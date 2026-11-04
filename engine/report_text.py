"""Text summaries shared by the tool, slash command and CLI surfaces."""

from __future__ import annotations

from typing import Any, Optional

_MISS_VERDICTS = {"CRITICAL_MISS", "MINOR_MISS", "INVALID"}


def _pct(rate: Optional[float]) -> str:
    return "?" if rate is None else f"{rate * 100:.0f}"


def _money(value: Optional[float], digits: int = 2) -> Optional[str]:
    if not isinstance(value, (int, float)):
        return None
    return f"${value:.{digits}f}"


def _model_label(current_model: Any) -> str:
    if not isinstance(current_model, dict):
        return str(current_model or "unknown")
    provider, model = current_model.get("provider"), current_model.get("model")
    if provider and model:
        return f"{provider}:{model}"
    return str(model or provider or "unknown")


def summary(record: dict) -> str:
    """The scorecard every surface shows after a run."""
    corpus = record.get("corpus") or {}
    decisions = record.get("decisions") or []
    window = corpus.get("window_days") or (record.get("settings") or {}).get("days") or "?"
    lines = [
        f"Understudy audition {record.get('id', '?')} — {len(decisions)} past decisions "
        f"from {corpus.get('sessions', '?')} conversations (last {window} days)",
    ]
    spend = corpus.get("spend_30d_usd")
    spend_text = (f"your spend, last 30 days: {_money(spend)}"
                  if isinstance(spend, (int, float)) else "spend not recorded")
    lines.append(f"Current model: {_model_label(record.get('current_model'))} · {spend_text}")

    for cand in record.get("candidates") or []:
        if not isinstance(cand, dict):
            continue
        lines.append("")
        provider = cand.get("provider") or "current provider"
        lines.append(f"{cand.get('model') or cand.get('spec') or '?'} ({provider})")
        scored = cand.get("scored") or 0
        takeover = cand.get("takeover_rate")
        counts = cand.get("counts") or {}
        taken = sum(counts.get(v, 0) for v in ("EXACT", "AGREE", "ALTERNATIVE", "BETTER"))
        ci = cand.get("ci95")
        ci_text = (f"{ci[0] * 100:.0f}–{ci[1] * 100:.0f}%"
                   if isinstance(ci, (list, tuple)) and len(ci) == 2 else "?")
        lines.append(f"  Could take over: {_pct(takeover)}% ({taken}/{scored}) · 95% CI {ci_text}")
        critical = counts.get("CRITICAL_MISS", 0) + counts.get("INVALID", 0)
        minor = counts.get("MINOR_MISS", 0)
        cat_bits = [f"{cat} {_pct(slot.get('takeover_rate'))}%"
                    for cat, slot in (cand.get("by_category") or {}).items()]
        miss_line = f"  Misses: {critical} critical, {minor} minor"
        if cat_bits:
            miss_line += " · " + " · ".join(cat_bits)
        extra = counts.get("ERROR", 0) + counts.get("UNJUDGED", 0) + counts.get("SKIPPED", 0)
        if extra:
            miss_line += f" · {extra} unscored"
        lines.append(miss_line)
        ratio = cand.get("cost_ratio")
        projected = cand.get("projected_monthly_usd")
        savings = cand.get("savings_monthly_usd")
        if isinstance(ratio, (int, float)) or isinstance(projected, (int, float)):
            cost_line = "  Cost: "
            if isinstance(ratio, (int, float)):
                cost_line += f"{ratio * 100:.0f}% of current"
            if isinstance(projected, (int, float)):
                cost_line += (" -> " if isinstance(ratio, (int, float)) else "")
                cost_line += f"~{_money(projected)}/mo"
            if isinstance(savings, (int, float)) and savings > 0:
                cost_line += f" (saves ~{_money(savings)}/mo)"
            span = record.get("usage_span_days")
            if span and span < 30:
                cost_line += (f" (based on {span:.1f} days of usage)" if span < 2
                              else f" (based on {int(span)} days of usage)")
            lines.append(cost_line)
        elif cand.get("model") in (record.get("costs_unknown") or []):
            lines.append(f"  Cost: unknown — budget cap could not be enforced; "
                         f"add a price under settings.prices")
        if cand.get("unreliable"):
            kinds = cand.get("errors_by_kind") or {}
            rate_limited = kinds.get("rate_limited", 0)
            timeouts = kinds.get("timeout", 0)
            invalid = kinds.get("invalid_params", 0)
            if rate_limited >= max(1, timeouts, invalid):
                lines.append(f"  Unreliable: the provider returned rate limits on {rate_limited} calls — rerun with lower concurrency or later.")
            elif timeouts >= 1:
                lines.append(f"  Unreliable: {timeouts} calls timed out — treat the scorecard with care.")
            elif invalid >= 1:
                lines.append(f"  Unreliable: the provider rejected {invalid} calls as invalid — the model may not fit this workload.")
            else:
                lines.append("  Unreliable: over 20% of its calls errored — treat the scorecard with care.")
        rec = cand.get("recommendation") or {}
        level = {"ready": "Ready", "partial": "Partial", "not_ready": "Not ready",
                 "insufficient": "Not enough data"}.get(rec.get("level"), "?")
        lines.append(f"  Verdict: {level} — {rec.get('text', '')}")

    baseline = record.get("baseline") or {}
    if baseline.get("scored"):
        counts = baseline.get("counts") or {}
        matched = sum(counts.get(v, 0) for v in ("EXACT", "AGREE", "ALTERNATIVE", "BETTER"))
        rate = baseline.get("takeover_rate")
        lines.append("")
        lines.append(f"Your current model matched its own past decisions on "
                     f"{matched}/{baseline['scored']} re-asked decisions ({_pct(rate)}%).")
    spent = record.get("spent_usd")
    budget = record.get("budget_usd")
    if isinstance(spent, (int, float)):
        lines.append(f"Spent {_money(spent)}"
                     + (f" of {_money(budget)}." if isinstance(budget, (int, float)) else "."))
    stopped = record.get("stopped_reason")
    if stopped and stopped != "cancelled":
        scored_total = sum((c.get("scored") or 0) for c in record.get("candidates") or [])
        total = len(decisions) * len(record.get("candidates") or [])
        lines.append(f"Stopped early ({stopped}): {scored_total} of {total} replays scored.")
    status = record.get("status")
    if status and status != "done":
        lines.append(f"Run status: {status}.")
    return "\n".join(lines)


def runs_list(runs: list[dict]) -> str:
    if not runs:
        return "No Understudy auditions yet."
    lines = ["Recent Understudy auditions:"]
    for run in runs:
        cands = ", ".join(run.get("candidates") or []) or "—"
        spent = run.get("spent_usd")
        spent_text = f" · {_money(spent)}" if isinstance(spent, (int, float)) else ""
        lines.append(f"  {run.get('id')} · {run.get('status') or '?'} · {cands}{spent_text} · {run.get('created_at') or ''}")
    return "\n".join(lines)


def details(record: dict, detail: str = "misses") -> str:
    """Per-decision dump: misses only, or every decision for detail == 'all'."""
    want_all = detail == "all"
    lines = []
    for decision in record.get("decisions") or []:
        replays = decision.get("replays") or {}
        interesting = {
            spec: r for spec, r in replays.items()
            if want_all or (r or {}).get("verdict") in _MISS_VERDICTS
        }
        if not interesting:
            continue
        lines.append(f"Decision {decision.get('id')} · {decision.get('category')} · {decision.get('kind')}")
        if decision.get("excerpt"):
            lines.append(f"  user: {decision['excerpt']}")
        lines.append(f"  production: {decision.get('production')}")
        for spec, replay in interesting.items():
            reason = (replay or {}).get("reason")
            lines.append(f"  {spec}: {(replay or {}).get('verdict')}"
                         + (f" — {reason}" if reason else ""))
            if (replay or {}).get("rendered"):
                lines.append(f"    candidate: {replay['rendered']}")
        lines.append("")
    if not lines:
        return "No critical or minor misses in this audition." if not want_all else "No decisions recorded."
    return "\n".join(lines).rstrip()
