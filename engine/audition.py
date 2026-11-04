"""Audition orchestrator: estimate -> run (concurrency, budget, progress, cancel).

Everything LLM-shaped is injected: ``llm`` needs only
``complete_structured(**kw)`` returning an object with ``.parsed``/``.text``/
``.content_type``/``.usage``/``.provider``/``.model``; ``session_source`` needs
the :class:`engine.corpus.SessionSource` protocol. Nothing here imports Hermes.
"""

from __future__ import annotations

import contextvars
import math
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Optional

from . import compare as _compare
from . import harness, judge as _judge, metrics as _metrics
from .corpus import collect_corpus
from .judge import ERROR, EXACT, INVALID, SKIPPED, UNJUDGED
from .pricing import call_cost, resolve_price, usage_cost
from .reliability import error_kind, is_transient, retry_after
from .sampling import stratified_sample
from .store import RunStore, new_run_id

DEFAULT_SETTINGS = {
    "sample": 20,
    "days": 14,
    "budget_usd": 1.5,
    "concurrency": 6,
    "baseline_sample": 8,
    "judge_model": "",
    "include_persona": True,
    "exclude_sources": [],
    "prices": {},
}

# Fallback provider ids when the providers package cannot be consulted.
KNOWN_PROVIDER_FALLBACK = frozenset({
    "openrouter", "nous", "anthropic", "openai", "deepseek", "gemini", "google",
    "xai", "groq", "together", "fireworks", "mistral", "ollama", "lmstudio", "custom",
})

MAX_CANDIDATES = 4
JUDGE_CALL_SHARE = 0.6  # rough estimate-only share of replays that reach the judge
MAX_RECORDED_ERRORS = 50
EXCERPT_USER_TAIL = 1200
TOOL_SURFACE_DEADLINE_S = 300.0
# Transient retry policy: attempts = 1 + len(RETRY_BACKOFF_S); each wait is
# backoff + up to JITTER_S, overridden (≤ RETRY_AFTER_CAP_S) by Retry-After.
RETRY_BACKOFF_S = (2.0, 6.0)
RETRY_JITTER_S = 1.0
RETRY_AFTER_CAP_S = 20.0


def _is_trust_error(exc: BaseException) -> bool:
    return type(exc).__name__ == "PluginLlmTrustError"


def _sleep(seconds: float) -> None:  # patched by tests
    time.sleep(seconds)


def _call_with_retry(call: Callable[[], Any],
                     rng: Any = None) -> tuple[Any, Optional[BaseException], int]:
    """(result, last_error, transient_hits). Transient failures retry with
    backoff + jitter, honouring Retry-After; everything else returns at once.
    Trust errors propagate."""
    transient_hits = 0
    last: Optional[BaseException] = None
    for attempt in range(len(RETRY_BACKOFF_S) + 1):
        try:
            return call(), None, transient_hits
        except Exception as exc:
            if _is_trust_error(exc):
                raise
            last = exc
            if not is_transient(exc) or attempt >= len(RETRY_BACKOFF_S):
                break
            transient_hits += 1
            wait_s = retry_after(exc)
            if wait_s is None:
                wait_s = RETRY_BACKOFF_S[attempt] + (
                    (rng.random() if rng else 0.5) * RETRY_JITTER_S)
            _sleep(min(wait_s, RETRY_AFTER_CAP_S))
    return None, last, transient_hits


def known_provider_ids() -> frozenset:
    """Provider ids Hermes knows, for ``provider:model`` spec parsing."""
    try:
        from providers import list_providers
        names = {str(p.name).strip().lower() for p in list_providers() if getattr(p, "name", "")}
        return frozenset(names) if names else KNOWN_PROVIDER_FALLBACK
    except Exception:
        return KNOWN_PROVIDER_FALLBACK


def parse_candidate_spec(spec: str, providers: Optional[Iterable[str]] = None) -> dict:
    """``provider:model`` when the prefix is a known provider, else a bare model id."""
    spec = str(spec or "").strip()
    known = {p.lower() for p in (providers or known_provider_ids())}
    provider, model = None, spec
    if ":" in spec:
        prefix, rest = spec.split(":", 1)
        if prefix.strip().lower() in known and rest.strip():
            provider, model = prefix.strip().lower(), rest.strip()
    return {"spec": spec, "provider": provider, "model": model}


def _int_setting(settings: dict, key: str, lo: int, hi: int) -> int:
    try:
        value = int(settings.get(key, DEFAULT_SETTINGS[key]))
    except (TypeError, ValueError):
        value = DEFAULT_SETTINGS[key]
    return max(lo, min(hi, value))


def _float_setting(settings: dict, key: str, lo: float, hi: float) -> float:
    try:
        value = float(settings.get(key, DEFAULT_SETTINGS[key]))
    except (TypeError, ValueError):
        value = DEFAULT_SETTINGS[key]
    return max(lo, min(hi, value))


def merged_settings(settings: Optional[dict] = None) -> dict:
    out = dict(DEFAULT_SETTINGS)
    for key, value in (settings or {}).items():
        if key in out and value is not None:
            out[key] = value
    return out


_EST_OUTPUT_TOKENS = 350      # assumed replay output
_EST_JUDGE_EXTRA_IN = 800     # judge prompt overhead beyond the excerpt
_EST_JUDGE_OUT = 120
_EST_SEED = "estimate"


def _replay_size_chars(decision: Any, include_persona: bool) -> int:
    """Chars the replay call sends for this decision (input side)."""
    transcript = harness.render_transcript(decision.prior_messages)
    persona = harness.persona_excerpt(decision.system_prompt, include_persona)
    tools_block = harness.available_tools_block(
        harness.resolve_tool_names(decision.pinned_tool_names, decision.session_tool_names))
    return len(harness.REPLAY_SYSTEM) + len(
        harness.build_replay_input(persona, tools_block, transcript))


def _estimate_costs(sampled: list, specs: list, judge_spec: dict,
                    settings: dict, baseline_n: int,
                    include_persona: bool,
                    current_model: Optional[dict],
                    catalog: Optional[dict]) -> dict:
    """List-price estimate of the whole run, or None-valued pieces when a
    needed price is unknown."""
    settings_prices = settings.get("prices") if isinstance(settings.get("prices"), dict) else {}
    if catalog is None:
        try:
            from .pricing import load_catalog
            catalog = load_catalog()
        except Exception:
            catalog = None
    current = current_model or {}
    current_price = resolve_price(current.get("provider"), current.get("model") or "",
                                  base_url=current.get("base_url"),
                                  settings_prices=settings_prices, catalog=catalog)
    judge_price = (resolve_price(judge_spec.get("provider"), judge_spec["model"],
                                 settings_prices=settings_prices, catalog=catalog)
                   if judge_spec.get("model") else current_price)
    replay_in = [(_replay_size_chars(d, include_persona) / 4.0) for d in sampled]
    excerpt_in = [(min(len(harness.render_transcript(d.prior_messages)),
                       _judge.EXCERPT_CAP) / 4.0 + _EST_JUDGE_EXTRA_IN) for d in sampled]
    judge_calls = int(math.ceil(len(sampled) * JUDGE_CALL_SHARE))
    # Judge sees the largest excerpts: it is called for the non-EXACT share.
    excerpt_in.sort(reverse=True)
    judge_in = excerpt_in[:judge_calls]

    def replay_total(price) -> float:
        return sum(usage_cost({"input_tokens": int(t), "output_tokens": _EST_OUTPUT_TOKENS},
                              price) or 0.0 for t in replay_in)

    def judge_total(price) -> float:
        return sum(usage_cost({"input_tokens": int(t), "output_tokens": _EST_JUDGE_OUT},
                              price) or 0.0 for t in judge_in)

    per_candidate, complete = [], True
    for spec in specs:
        price = resolve_price(spec.get("provider"), spec["model"],
                              base_url=spec.get("base_url"),
                              settings_prices=settings_prices, catalog=catalog)
        replay_usd = replay_total(price) if price else None
        judge_usd = judge_total(judge_price) if judge_price else None
        if replay_usd is None or judge_usd is None:
            complete = False
            per_candidate.append({"spec": spec["spec"], "replay_usd": replay_usd,
                                  "judge_usd": judge_usd,
                                  "total_usd": None if replay_usd is None else replay_usd,
                                  "price_known": price is not None})
        else:
            per_candidate.append({"spec": spec["spec"], "replay_usd": replay_usd,
                                  "judge_usd": judge_usd,
                                  "total_usd": replay_usd + judge_usd,
                                  "price_known": True})
    baseline_in = replay_in[:baseline_n]
    baseline_usd = (sum(usage_cost({"input_tokens": int(t), "output_tokens": _EST_OUTPUT_TOKENS},
                                   current_price) or 0.0 for t in baseline_in)
                    if current_price else None)
    baseline_judge = (judge_total(current_price) * (baseline_n / max(1, len(sampled)))
                      if current_price else None)
    if baseline_usd is None:
        complete = False
    base_total = None
    if baseline_usd is not None and baseline_judge is not None:
        base_total = baseline_usd + baseline_judge
    total = None
    if complete and all(c["total_usd"] is not None for c in per_candidate) and base_total is not None:
        total = sum(c["total_usd"] for c in per_candidate) + base_total
    return {"estimated_cost_usd": total, "per_candidate": per_candidate,
            "baseline_usd": base_total, "priced": complete}


def estimate(session_source: Any, settings: Optional[dict], candidate_specs: Iterable[str],
             *, current_session_id: Optional[str] = None, now: Optional[float] = None,
             current_model: Optional[dict] = None, catalog: Optional[dict] = None) -> dict:
    """What a run would do: corpus stats, planned calls, list-price estimate."""
    settings = merged_settings(settings)
    days = _int_setting(settings, "days", 1, 90)
    sample = _int_setting(settings, "sample", 4, 80)
    corpus = collect_corpus(
        session_source, days=days,
        exclude_sources=settings.get("exclude_sources") or (),
        current_session_id=current_session_id, now=now)
    replays = min(sample, len(corpus.decisions))
    baseline = min(_int_setting(settings, "baseline_sample", 0, 80), replays)
    specs = [parse_candidate_spec(s) for s in candidate_specs][:MAX_CANDIDATES]
    per_candidate = replays + int(math.ceil(replays * JUDGE_CALL_SHARE))
    out = {
        "window_days": days,
        "sessions_seen": corpus.sessions_seen,
        "sessions_used": corpus.sessions_used,
        "decisions_available": len(corpus.decisions),
        "by_category": dict(corpus.by_category),
        "spend_30d_usd": corpus.spend_30d_usd,
        "spend_known": corpus.spend_known,
        "usage_30d": dict(corpus.usage_30d),
        "sample_per_candidate": replays,
        "candidates": [s["spec"] for s in specs],
        "planned_calls_per_candidate": per_candidate,
        "baseline_calls": baseline + int(math.ceil(baseline * JUDGE_CALL_SHARE)),
        "planned_calls_total": per_candidate * len(specs) + baseline + int(math.ceil(baseline * JUDGE_CALL_SHARE)),
        "budget_usd": _float_setting(settings, "budget_usd", 0.05, 25.0),
    }
    sampled = stratified_sample(corpus.decisions, replays, _EST_SEED)
    judge_spec = (parse_candidate_spec(settings["judge_model"])
                  if str(settings.get("judge_model") or "").strip()
                  else {"provider": None, "model": None})
    out.update(_estimate_costs(
        sampled, specs, judge_spec, settings, baseline,
        bool(settings.get("include_persona", True)), current_model, catalog))
    return out


def _decision_excerpt(decision: Any) -> str:
    text = str(getattr(decision, "last_user_text", "") or "")
    return harness.redact(text[-EXCERPT_USER_TAIL:])


def _output_tokens(result: Any) -> Optional[int]:
    usage = getattr(result, "usage", None)
    value = getattr(usage, "output_tokens", None) if usage is not None else None
    return int(value) if isinstance(value, (int, float)) else None


def _run_item(llm: Any, spec: dict, decision: Any, judge_spec: dict,
              run_id: str, include_persona: bool, cand_price: Any,
              judge_price: Any) -> dict:
    """Replay one decision on one candidate, compare, judge unless EXACT.

    ``PluginLlmTrustError`` propagates: it aborts the whole run, not this item.
    """
    transcript = harness.render_transcript(decision.prior_messages)
    persona = harness.persona_excerpt(decision.system_prompt, include_persona)
    tools_block = harness.available_tools_block(
        harness.resolve_tool_names(decision.pinned_tool_names, decision.session_tool_names))
    item: dict[str, Any] = {
        "decision": decision, "spec": spec["spec"], "category": decision.category,
        "kind": decision.kind, "attempted": True, "verdict": None,
        "reason": "", "rendered": "", "cost_usd": None, "cost_source": "",
        "judge_cost_usd": None, "judge_cost_source": "", "latency_ms": None,
        "output_tokens": None, "errors": [],
    }
    latency_ms = 0.0

    def _replay() -> Any:
        nonlocal latency_ms
        started = time.monotonic()
        out = harness.call_replay(
            llm, provider=spec.get("provider"), model=spec.get("model"),
            persona=persona, tools_block=tools_block, transcript=transcript)
        latency_ms += (time.monotonic() - started) * 1000.0
        return out

    result, err, transient = _call_with_retry(_replay)
    if transient:
        item["transient_hits"] = transient
    if err is not None:
        item["errors"].append(f"{type(err).__name__}: {err}")
        item["error_kind"] = error_kind(err)
    item["latency_ms"] = latency_ms
    if result is None:
        item["verdict"] = ERROR
        return item
    cost, source = call_cost(getattr(result, "usage", None), cand_price)
    item["cost_usd"], item["cost_source"] = cost, source
    item["output_tokens"] = _output_tokens(result)

    candidate = harness.parse_next_action(result)
    item["rendered"] = harness.render_step(candidate)
    if candidate is None:
        item["verdict"] = INVALID
        return item
    if _compare.compare(decision.production, candidate) == _compare.EXACT:
        item["verdict"] = EXACT
        return item
    jout, jerr, jtransient = _call_with_retry(lambda: _judge.call_judge(
        llm, transcript=transcript,
        first_user=_judge.first_user_line(decision.prior_messages),
        production=decision.production, candidate=candidate,
        seed=f"{run_id}:{decision.id}:{spec['spec']}",
        provider=judge_spec.get("provider"), model=judge_spec.get("model")))
    if jtransient:
        item["transient_hits"] = item.get("transient_hits", 0) + jtransient
    if jerr is None:
        verdict, reason, _is_step1, jresult = jout
        jcost, jsource = call_cost(getattr(jresult, "usage", None), judge_price)
        item["judge_cost_usd"], item["judge_cost_source"] = jcost, jsource
        item["verdict"], item["reason"] = verdict, reason
    else:
        item["errors"].append(f"judge {type(jerr).__name__}: {jerr}")
        item.setdefault("error_kind", error_kind(jerr))
        item["verdict"] = UNJUDGED
    return item


class _Budget:
    """Shared spend/mean accounting for the submission guard."""

    def __init__(self, cap: float, concurrency: int = 1):
        self.cap = cap
        self.concurrency = max(1, int(concurrency))
        self.spent = 0.0
        self.done = 0
        self.lock = threading.Lock()

    def may_submit(self, in_flight: int) -> bool:
        with self.lock:
            mean = self.spent / self.done if self.done else 0.0
            if mean <= 0.0:
                # No estimate yet (or costs unreported): keep the pool busy but
                # don't race ahead of the spend data we need to project with.
                return in_flight < self.concurrency
            return self.spent + mean * (in_flight + 1) <= self.cap

    def record(self, item_cost: float) -> None:
        with self.lock:
            self.spent += item_cost
            self.done += 1


def _cancelled(cancel: Optional[Callable[[], bool]]) -> bool:
    try:
        return bool(cancel and cancel())
    except Exception:
        return False


def _compact_candidate(entry: dict) -> dict:
    """The candidate block persisted in the run record."""
    return {k: v for k, v in entry.items() if k not in ("items", "summary")}


def run(llm: Any, session_source: Any, settings: Optional[dict],
        candidate_specs: Iterable[str], *, store: Optional[Any] = None,
        progress: Optional[Callable[[dict], None]] = None,
        cancel: Optional[Callable[[], bool]] = None,
        run_id: Optional[str] = None, now: Optional[float] = None,
        current_model: Optional[dict] = None, profile: str = "",
        current_session_id: Optional[str] = None,
        max_seconds: Optional[float] = None,
        catalog: Optional[dict] = None) -> dict:
    """Run a full audition and return the persisted run record."""
    settings = merged_settings(settings)
    days = _int_setting(settings, "days", 1, 90)
    sample = _int_setting(settings, "sample", 4, 80)
    budget = _float_setting(settings, "budget_usd", 0.05, 25.0)
    concurrency = _int_setting(settings, "concurrency", 1, 16)
    baseline_n = _int_setting(settings, "baseline_sample", 0, 80)
    include_persona = bool(settings.get("include_persona", True))
    started = time.time() if now is None else float(now)
    # The deadline is wall-clock — `now` is an injection point for corpus
    # timestamps, not elapsed time.
    deadline = time.monotonic() + float(max_seconds) if max_seconds else None
    run_id = run_id or new_run_id()

    specs = [parse_candidate_spec(s) for s in candidate_specs if str(s).strip()][:MAX_CANDIDATES]
    judge_spec = (parse_candidate_spec(settings["judge_model"])
                  if str(settings.get("judge_model") or "").strip() else {"provider": None, "model": None})
    settings_prices = settings.get("prices") if isinstance(settings.get("prices"), dict) else {}
    current = current_model or {}

    def _catalog():
        if catalog is not None:
            return catalog
        try:
            from .pricing import load_catalog
            return load_catalog()
        except Exception:
            return None
    catalog_data = _catalog()
    prices = {
        s["spec"]: resolve_price(s.get("provider"), s["model"],
                                 base_url=s.get("base_url"), settings_prices=settings_prices,
                                 catalog=catalog_data)
        for s in specs
    }
    current_price = resolve_price(current.get("provider"), current.get("model") or "",
                                  base_url=current.get("base_url"),
                                  settings_prices=settings_prices, catalog=catalog_data)
    judge_price = (resolve_price(judge_spec.get("provider"), judge_spec["model"],
                                 settings_prices=settings_prices, catalog=catalog_data)
                   if judge_spec.get("model") else current_price)

    corpus = collect_corpus(
        session_source, days=days,
        exclude_sources=settings.get("exclude_sources") or (),
        current_session_id=current_session_id, now=started)
    sampled = stratified_sample(corpus.decisions, sample, run_id)
    baseline_ids = {d.id for d in sampled[:baseline_n]} if baseline_n else set()

    decision_records: dict[str, dict] = {}
    for decision in sampled:
        decision_records[decision.id] = {
            "id": decision.id, "session_id": decision.session_id, "source": decision.source,
            "category": decision.category, "kind": decision.kind,
            "session_model": decision.session_model,
            "excerpt": _decision_excerpt(decision),
            "production": harness.redact(harness.render_step(decision.production)),
            "replays": {},
        }
    record: dict[str, Any] = {
        "id": run_id, "status": "running", "stopped_reason": None,
        "created_at": datetime.fromtimestamp(started, timezone.utc).isoformat(timespec="seconds"),
        "finished_at": None, "profile": profile, "settings": dict(settings),
        "current_model": current,
        "corpus": {
            "sessions": corpus.sessions_used, "sessions_seen": corpus.sessions_seen,
            "decisions_available": len(corpus.decisions), "by_category": dict(corpus.by_category),
            "window_days": days, "spend_30d_usd": corpus.spend_30d_usd,
            "spend_known": corpus.spend_known, "usage_30d": dict(corpus.usage_30d),
        },
        "candidates": [], "baseline": None,
        "decisions": [decision_records[d.id] for d in sampled],
        "spent_usd": 0.0, "judge_spent_usd": 0.0, "errors": [],
        "budget_usd": budget, "sampled": len(sampled),
        "prices": {}, "usage_30d": dict(corpus.usage_30d), "verbosity": {},
        "current_monthly_usd": None, "current_monthly_source": None,
        "costs_unknown": [],
    }

    store = store if store is not None else RunStore()
    errors: list[str] = record["errors"]

    def save() -> None:
        try:
            store.save(record)
        except Exception as exc:
            if len(errors) < MAX_RECORDED_ERRORS:
                errors.append(f"store: {type(exc).__name__}: {exc}")

    def note_error(text: str) -> None:
        if len(errors) < MAX_RECORDED_ERRORS:
            errors.append(text)

    def emit(event: dict) -> None:
        if progress:
            try:
                progress(event)
            except Exception:
                pass

    save()

    # One interleaved work list: per decision the baseline item first, then each
    # candidate — budget/time/cancel cuts hit every candidate evenly.
    base_spec = {"spec": "baseline", "provider": None, "model": None}
    all_specs = {s["spec"]: s for s in specs} | {"baseline": base_spec}
    price_for = dict(prices)
    price_for["baseline"] = current_price
    work: list[tuple[dict, Any]] = []
    for decision in sampled:
        if decision.id in baseline_ids:
            work.append((base_spec, decision))
        for spec in specs:
            work.append((spec, decision))
    total_items = len(work)
    budget_state = _Budget(budget, concurrency)
    emit({"event": "start", "total": total_items, "run_id": run_id})

    item_results: list[Optional[dict]] = [None] * len(work)
    tallies: dict[str, dict[str, int]] = {
        label: {"ok": 0, "scored": 0} for label in all_specs}
    trust_abort = [False]
    stop_reason = [None]

    def item_cost(item: dict) -> float:
        return (item.get("cost_usd") or 0.0) + (item.get("judge_cost_usd") or 0.0)

    def past_deadline() -> bool:
        return deadline is not None and time.monotonic() >= deadline

    try:
        results = item_results
        in_flight: dict = {}
        pending = list(range(len(work)))
        # Per-spec adaptive caps: a rate_limited item halves its spec's cap (min 1).
        spec_cap: dict[str, int] = {label: max(1, concurrency) for label in all_specs}
        spec_busy: dict[str, int] = {label: 0 for label in all_specs}

        def _schedulable() -> Optional[int]:
            for i, wi in enumerate(pending):
                if spec_busy[work[wi][0]["spec"]] < spec_cap[work[wi][0]["spec"]]:
                    return i
            return None

        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            while pending or in_flight:
                while (pending and not trust_abort[0]
                       and not _cancelled(cancel) and not past_deadline()
                       and budget_state.may_submit(len(in_flight))):
                    pick = _schedulable()
                    if pick is None:
                        break
                    index = pending.pop(pick)
                    spec, decision = work[index]
                    task_ctx = contextvars.copy_context()
                    future = pool.submit(
                        task_ctx.run, _run_item, llm, spec, decision,
                        judge_spec, run_id, include_persona,
                        price_for.get(spec["spec"]), judge_price)
                    in_flight[future] = index
                    spec_busy[spec["spec"]] += 1
                if pending and not in_flight:
                    break
                if not in_flight:
                    break
                done, _pending = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
                for future in done:
                    index = in_flight.pop(future)
                    spec, decision = work[index]
                    spec_busy[spec["spec"]] -= 1
                    try:
                        item = future.result()
                    except Exception as exc:
                        if _is_trust_error(exc):
                            trust_abort[0] = True
                            item = {"decision": decision, "spec": spec["spec"],
                                    "category": decision.category, "kind": decision.kind,
                                    "attempted": False, "verdict": SKIPPED, "reason": "",
                                    "rendered": "", "cost_usd": None, "cost_source": "",
                                    "judge_cost_usd": None, "judge_cost_source": "",
                                    "latency_ms": None, "output_tokens": None, "errors": []}
                            note_error(f"trust: {exc}")
                        else:
                            item = {"decision": decision, "spec": spec["spec"],
                                    "category": decision.category, "kind": decision.kind,
                                    "attempted": True, "verdict": ERROR, "reason": "",
                                    "rendered": "", "cost_usd": None, "cost_source": "",
                                    "judge_cost_usd": None, "judge_cost_source": "",
                                    "latency_ms": None, "output_tokens": None,
                                    "errors": [f"{type(exc).__name__}: {exc}"]}
                    results[index] = item
                    if item.get("error_kind") == "rate_limited":
                        spec_cap[spec["spec"]] = max(1, spec_cap[spec["spec"]] // 2)
                    for err in item.get("errors") or []:
                        note_error(f"{spec['spec']} {decision.id}: {err}")
                    budget_state.record(item_cost(item))
                    record["spent_usd"] += item_cost(item)
                    record["judge_spent_usd"] += item.get("judge_cost_usd") or 0.0
                    replay = {"rendered": harness.redact(item.get("rendered") or ""),
                              "verdict": item["verdict"], "reason": item.get("reason") or "",
                              "cost_usd": item.get("cost_usd"),
                              "cost_source": item.get("cost_source") or "",
                              "latency_ms": item.get("latency_ms")}
                    if spec["spec"] == "baseline":
                        decision_records[decision.id]["baseline"] = replay
                    else:
                        decision_records[decision.id]["replays"][spec["spec"]] = replay
                    if item["verdict"] in _metrics.SCORED_VERDICTS:
                        tallies[spec["spec"]]["scored"] += 1
                        if item["verdict"] in _metrics.TAKEOVER_VERDICTS:
                            tallies[spec["spec"]]["ok"] += 1
                    save()
                    emit({"event": "item", "done": budget_state.done,
                          "total": total_items, "spent_usd": record["spent_usd"],
                          "budget_usd": budget, "candidate": spec["spec"],
                          "verdict": item["verdict"],
                          "tallies": {k: dict(v) for k, v in tallies.items()}})
        # Unscheduled items become SKIPPED, with the reason scheduling stopped.
        if trust_abort[0]:
            stop_reason[0] = "trust"
        elif _cancelled(cancel):
            stop_reason[0] = "cancelled"
        elif past_deadline():
            stop_reason[0] = "time"
        elif pending:
            stop_reason[0] = "budget"
        for index in range(len(work)):
            if results[index] is None:
                spec, decision = work[index]
                item = {"decision": decision, "spec": spec["spec"],
                        "category": decision.category, "kind": decision.kind,
                        "attempted": False, "verdict": SKIPPED, "reason": "",
                        "rendered": "", "cost_usd": None, "cost_source": "",
                        "judge_cost_usd": None, "judge_cost_source": "",
                        "latency_ms": None, "output_tokens": None, "errors": []}
                results[index] = item
                replay = {"rendered": "", "verdict": SKIPPED, "reason": "",
                          "cost_usd": None, "cost_source": "", "latency_ms": None}
                if spec["spec"] == "baseline":
                    decision_records[decision.id]["baseline"] = replay
                else:
                    decision_records[decision.id]["replays"][spec["spec"]] = replay

        # ── Summaries, prices, projection ────────────────────────────────────
        by_label: dict[str, list[dict]] = {}
        for item in results:
            if item is not None:
                by_label.setdefault(item["spec"], []).append(item)
        baseline_items = by_label.get("baseline", [])
        baseline_summary = _metrics.summarize_items(baseline_items)

        # Verbosity: candidate vs baseline output tokens on decisions both replayed.
        base_out_by_decision = {
            item["decision"].id: item.get("output_tokens") for item in baseline_items
            if item.get("output_tokens")}
        verbosity: dict[str, float] = {}
        for spec in specs:
            cand_pairs = [(item.get("output_tokens"),
                           base_out_by_decision.get(item["decision"].id))
                          for item in by_label.get(spec["spec"], [])
                          if item.get("output_tokens")
                          and base_out_by_decision.get(item["decision"].id)]
            if cand_pairs:
                cand_sum = sum(c for c, _b in cand_pairs)
                base_sum = sum(b for _c, b in cand_pairs)
                if base_sum > 0:
                    verbosity[spec["spec"]] = cand_sum / base_sum
        record["verbosity"] = verbosity

        # Monthly normalization: window usage/spend scaled to 30 days by the
        # actual span covered (clamped 1–30), so a 3-day-old history doesn't
        # under-report the month.
        usage_30d = corpus.usage_30d
        starts = [s["started_at"] for s in corpus.sessions_30d
                  if isinstance(s.get("started_at"), (int, float)) and s["started_at"] > 0]
        span_days = (min(30.0, max(1.0, (started - min(starts)) / 86400.0))
                     if starts else 30.0)
        record["usage_span_days"] = span_days
        monthly_scale = 30.0 / span_days
        if corpus.spend_known and corpus.spend_30d_usd is not None:
            record["current_monthly_usd"] = corpus.spend_30d_usd * monthly_scale
            record["current_monthly_source"] = "recorded"
        else:
            total = 0.0
            all_priced = bool(corpus.sessions_30d)
            for row in corpus.sessions_30d:
                row_price = resolve_price(row.get("provider"), row.get("model"),
                                          settings_prices=settings_prices,
                                          catalog=catalog_data)
                cost = usage_cost(row, row_price)
                if cost is None:
                    all_priced = False
                    break
                total += cost
            if all_priced:
                record["current_monthly_usd"] = total * monthly_scale
                record["current_monthly_source"] = "list price"
        current_list_cost = usage_cost(usage_30d, current_price)
        baseline_cpd = baseline_summary.get("cost_per_decision")
        record["baseline"] = {
            "decisions": len(baseline_items), "counts": baseline_summary["counts"],
            "scored": baseline_summary["scored"],
            "takeover_rate": baseline_summary.get("takeover_rate"),
            "critical_rate": baseline_summary.get("critical_rate"),
            "cost_usd": baseline_summary.get("cost_usd"),
            "cost_per_decision": baseline_cpd,
        }
        costs_unknown = []
        for spec in specs:
            items_for_spec = by_label.get(spec["spec"], [])
            summary = _metrics.summarize_items(items_for_spec)
            kinds: dict[str, Any] = {}
            for it in items_for_spec:
                kind = it.get("error_kind")
                if kind:
                    kinds[kind] = kinds.get(kind, 0) + 1
                    if "example" not in kinds and it.get("errors"):
                        kinds["example"] = str(it["errors"][-1])[:300]
            for it in items_for_spec:
                if it.get("errors") and "example" not in kinds:
                    kinds["example"] = str(it["errors"][-1])[:300]
                    break
            price = prices.get(spec["spec"])
            if summary.get("cost_per_decision") is None:
                costs_unknown.append(spec["model"])
            verb = verbosity.get(spec["spec"], 1.0)
            priced_tokens = dict(usage_30d)
            priced_tokens["output_tokens"] = int(usage_30d.get("output_tokens", 0) * verb)
            projected = usage_cost(priced_tokens, price)
            if projected is not None:
                projected *= monthly_scale
            denominator = (current_list_cost * monthly_scale
                           if current_list_cost is not None
                           else record["current_monthly_usd"])
            cost_ratio = (projected / denominator
                          if projected is not None and denominator else None)
            savings = (record["current_monthly_usd"] - projected
                       if projected is not None and record["current_monthly_usd"] is not None
                       else None)
            rec = _metrics.recommend(summary["scored"], summary.get("takeover_rate"),
                                     summary.get("critical_rate"), summary.get("by_category"),
                                     baseline_summary)
            record["prices"][spec["spec"]] = (
                {"input_per_m": price.input_per_m, "output_per_m": price.output_per_m,
                 "cache_read_per_m": price.cache_read_per_m, "source": price.source}
                if price else None)
            record["candidates"].append({
                "spec": spec["spec"], "provider": spec.get("provider"),
                "model": spec.get("model"),
                "counts": summary["counts"], "scored": summary["scored"],
                "attempted": summary.get("attempted"), "skipped": summary.get("skipped"),
                "takeover_rate": summary.get("takeover_rate"), "ci95": summary.get("ci95"),
                "critical_rate": summary.get("critical_rate"), "minor_rate": summary.get("minor_rate"),
                "by_category": summary.get("by_category"), "cost_usd": summary.get("cost_usd"),
                "cost_per_decision": summary.get("cost_per_decision"),
                "latency_ms_p50": summary.get("latency_ms_p50"),
                "unreliable": summary.get("unreliable"),
                "errors_by_kind": kinds,
                "verbosity": verb,
                "cost_ratio": cost_ratio,
                "projected_monthly_usd": projected,
                "savings_monthly_usd": savings,
                "recommendation": rec,
            })
        record["costs_unknown"] = costs_unknown
        if trust_abort[0]:
            record["status"] = "failed"
            record["failure"] = "trust"
        elif stop_reason[0] == "cancelled" or _cancelled(cancel):
            record["status"] = "cancelled"
            record["stopped_reason"] = "cancelled"
        else:
            record["status"] = "done"
            record["stopped_reason"] = stop_reason[0]
    except Exception as exc:
        if _is_trust_error(exc):
            record["status"] = "failed"
            record["failure"] = "trust"
            note_error(f"trust: {exc}")
        else:
            record["status"] = "failed"
            note_error(f"run: {type(exc).__name__}: {exc}")
    record["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    save()
    try:
        store.prune()
    except Exception:
        pass
    emit({"event": "finished", "run_id": run_id, "status": record["status"],
          "spent_usd": record["spent_usd"],
          "stopped_reason": record.get("stopped_reason"),
          "failure": record.get("failure")})
    return record
