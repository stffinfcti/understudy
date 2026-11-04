"""/understudy slash command."""

from __future__ import annotations

from typing import Any

from ..engine import audition, report_text
from ..engine.store import default_store
from . import runner

USAGE = (
    "Understudy — audition models on your real history.\n"
    "Usage:\n"
    "  /understudy <model…> [--sample N] [--days N] [--budget X] [--yes]\n"
    "    e.g. /understudy openrouter:moonshotai/kimi-k3 --sample 24\n"
    "  /understudy last    latest audition scorecard\n"
    "  /understudy runs    recent auditions\n"
    "  /understudy help    this text\n"
    "Candidates are provider:model or a bare model on your current provider. "
    "Runs cost real tokens; you are asked to confirm unless you pass --yes."
)


def _parse_args(raw: str) -> tuple[list[str], dict, bool, str]:
    """-> (candidates, overrides, yes, error)."""
    candidates, overrides, yes = [], {}, False
    tokens = str(raw or "").split()
    i = 0
    flags = {"--sample": "sample", "--days": "days", "--budget": "budget_usd"}
    while i < len(tokens):
        tok = tokens[i]
        if tok == "--yes":
            yes = True
        elif tok in flags:
            if i + 1 >= len(tokens):
                return [], {}, False, f"missing value for {tok}"
            value = tokens[i + 1]
            try:
                overrides[flags[tok]] = float(value) if tok == "--budget" else int(value)
            except ValueError:
                return [], {}, False, f"bad value for {tok}: {value!r}"
            i += 1
        elif tok.startswith("--"):
            return [], {}, False, f"unknown flag {tok!r}"
        else:
            candidates.append(tok)
        i += 1
    return candidates, overrides, yes, ""


def handle(ctx: Any, raw_args: str) -> str:
    raw = (raw_args or "").strip()
    if not raw or raw == "help":
        return USAGE
    store = default_store()
    if raw == "runs":
        return report_text.runs_list(store.list_runs())
    if raw == "last":
        record = store.load(None)
        return report_text.summary(record) if record else "No Understudy auditions yet."

    candidates, overrides, yes, error = _parse_args(raw)
    if error:
        return f"{error}\n\n{USAGE}"
    if not candidates or len(candidates) > 4:
        return USAGE

    settings = runner.get_settings(ctx, overrides)
    try:
        source = runner.session_source()
    except Exception as exc:
        return f"Understudy cannot open session history: {type(exc).__name__}: {exc}"
    try:
        est = audition.estimate(source, settings, candidates,
                                current_session_id=runner.current_session_id() or None,
                                current_model=runner.current_model())
        if not est["decisions_available"]:
            return ("Understudy found no eligible past decisions in the last "
                    f"{est['window_days']} days. Try --days 30 or more.")
        trust = runner.preflight_trust(ctx, candidates, settings.get("judge_model") or "")
        if trust:
            return trust
        n, budget = est["sample_per_candidate"], est["budget_usd"]
        if not yes:
            est_note = (f" Estimated ~${est['estimated_cost_usd']:.2f} at list prices."
                        if est.get("estimated_cost_usd") is not None else "")
            picked = ctx.ask_choice(
                f"Audition {', '.join(candidates)} on {n} of your past decisions?{est_note} "
                f"Spending stops at ${budget:.2f}.",
                [{"value": "run", "label": "Run audition"},
                 {"value": "cancel", "label": "Cancel"}])
            if picked != "run":
                return ("Audition not run. Rerun with --yes to skip the question "
                        "and start immediately.")
        label = candidates[0].split(":")[-1] if len(candidates) == 1 else f"{len(candidates)} models"
        sink = runner.ProgressSink(ctx, label)
        record = audition.run(
            ctx.llm, source, settings, candidates,
            store=store, progress=sink,
            current_model=runner.current_model(), profile=getattr(ctx, "profile_name", ""),
            current_session_id=runner.current_session_id() or None)
        from .tools import _write_artifacts
        _write_artifacts(record, store)
        try:
            store.save(record)
        except Exception:
            pass
        text = report_text.summary(record)
        if record.get("report_html"):
            text += f"\nFull report: {record['report_html']}"
        if record.get("card_png"):
            text += f"\nMEDIA:{record['card_png']}"
        try:
            sink.post_final(text)
        except Exception:
            pass
        if record.get("failure") == "trust":
            return runner.TRUST_HELP
        return text
    except Exception as exc:
        if runner.is_trust_error(exc):
            return runner.TRUST_HELP
        return f"Understudy audition failed: {type(exc).__name__}: {exc}"
    finally:
        try:
            source.close()
        except Exception:
            pass
