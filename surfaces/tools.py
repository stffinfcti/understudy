"""Agent tools: understudy_audition (run/dry-run) and understudy_report."""

from __future__ import annotations

from typing import Any

from ..engine import audition, report_text
from ..engine.store import default_store
from . import runner

AUDITION_SCHEMA = {
    "name": "understudy_audition",
    "description": (
        "Audition cheaper or different models on the user's real past Hermes conversations. "
        "Replays a sample of past decisions (replies and tool calls) on each candidate model, "
        "has a judge compare each one with what the current model actually did, and reports "
        "how often each candidate could have taken over, its critical misses, and projected "
        "monthly cost. Costs real tokens on the user's providers: call with dry_run=true "
        "first, tell the user the estimate, and only run after they agree. Never switches "
        "models itself."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 4,
                "description": "Models to audition, as provider:model (e.g. "
                               "openrouter:moonshotai/kimi-k3) or a bare model id on the "
                               "current provider.",
            },
            "sample": {
                "type": "integer",
                "minimum": 4,
                "maximum": 80,
                "description": "Past decisions to replay per candidate (default from "
                               "settings, usually 24).",
            },
            "days": {
                "type": "integer",
                "minimum": 1,
                "maximum": 90,
                "description": "How far back to sample conversations (default 14).",
            },
            "budget_usd": {
                "type": "number",
                "minimum": 0.05,
                "maximum": 25,
                "description": "Hard spending cap for this audition (default 1.50).",
            },
            "dry_run": {
                "type": "boolean",
                "description": "Only report what would be replayed and the estimate; "
                               "spends nothing.",
            },
        },
        "required": ["candidates"],
    },
}

REPORT_SCHEMA = {
    "name": "understudy_report",
    "description": (
        "Read an earlier Understudy audition: its scorecard, and optionally the decisions "
        "where a candidate fell short, with the transcript excerpt and both steps. Use to "
        "answer follow-up questions about an audition."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "run_id": {
                "type": "string",
                "description": "Audition id; omit for the latest.",
            },
            "detail": {
                "type": "string",
                "enum": ["summary", "misses", "all"],
                "description": "summary (default), misses (critical and minor misses with "
                               "excerpts), or all decisions.",
            },
        },
    },
}


def _json(data: Any) -> str:
    from tools.registry import tool_result
    return tool_result(data)


def _err(message: str, **extra: Any) -> str:
    from tools.registry import tool_error
    return tool_error(message, **extra)


def _write_artifacts(record: dict, store) -> None:
    """PNG card + HTML report next to the record; record gains card_png/report_html."""
    from ..engine import write_report_artifacts
    write_report_artifacts(record, store)


def _compact_record(record: dict) -> dict:
    """The small JSON the tool returns — never the raw record (context size)."""
    candidates = []
    for cand in record.get("candidates") or []:
        counts = cand.get("counts") or {}
        rec = cand.get("recommendation") or {}
        candidates.append({
            "spec": cand.get("spec"), "model": cand.get("model"),
            "provider": cand.get("provider"),
            "takeover_rate": cand.get("takeover_rate"), "ci95": cand.get("ci95"),
            "critical": counts.get("CRITICAL_MISS", 0) + counts.get("INVALID", 0),
            "minor": counts.get("MINOR_MISS", 0), "scored": cand.get("scored"),
            "cost_ratio": cand.get("cost_ratio"),
            "projected_monthly_usd": cand.get("projected_monthly_usd"),
            "savings_monthly_usd": cand.get("savings_monthly_usd"),
            "recommendation": {"level": rec.get("level"), "text": rec.get("text"),
                               "categories": rec.get("categories")},
            "by_category": {cat: slot.get("takeover_rate")
                            for cat, slot in (cand.get("by_category") or {}).items()},
        })
    baseline = record.get("baseline") or {}
    out = {
        "run_id": record.get("id"), "status": record.get("status"),
        "stopped_reason": record.get("stopped_reason"),
        "failure": record.get("failure"),
        "summary": report_text.summary(record),
        "candidates": candidates,
        "baseline": {"scored": baseline.get("scored"),
                     "takeover_rate": baseline.get("takeover_rate")},
        "spent_usd": record.get("spent_usd"),
        "costs_unknown": record.get("costs_unknown") or [],
        "card_png": record.get("card_png"), "report_html": record.get("report_html"),
    }
    if out["card_png"]:
        out["show_card"] = ("Put this line on its own line in your reply so the "
                            f"user sees the scorecard: MEDIA:{out['card_png']}")
    return out


def _overrides(args: dict) -> dict:
    return {key: args.get(key) for key in ("sample", "days", "budget_usd")
            if args.get(key) is not None}


def _candidates(args: dict) -> list[str]:
    raw = args.get("candidates")
    if isinstance(raw, str):
        raw = [raw]
    return [str(c).strip() for c in (raw or []) if str(c).strip()]


def audition_handler(ctx: Any, args: dict, **_kw: Any) -> str:
    candidates = _candidates(args)
    if not candidates:
        return _err("candidates is required: 1-4 model specs, e.g. [\"openrouter:moonshotai/kimi-k3\"]")
    if len(candidates) > 4:
        return _err("at most 4 candidates per audition")
    settings = runner.get_settings(ctx, _overrides(args))
    try:
        source = runner.session_source()
    except Exception as exc:
        return _err(f"cannot open session history: {type(exc).__name__}: {exc}")
    try:
        est = audition.estimate(source, settings, candidates,
                                current_session_id=runner.current_session_id() or None,
                                current_model=runner.current_model())
        if args.get("dry_run"):
            return _json({
                "dry_run": True,
                "estimate": est,
                "note": "No calls made and nothing spent. Share this estimate with the user; "
                        "run again with dry_run=false only after they agree.",
            })
        trust = runner.preflight_trust(ctx, candidates, settings.get("judge_model") or "")
        if trust:
            return _err(trust)
        if est["decisions_available"] == 0:
            return _json({"ran": False, "estimate": est,
                          "note": "No eligible past decisions found in the window — try a "
                                  "larger --days."})
        # Hermes caps tool calls at ~420s: run in a background thread, wait up
        # to 300s, and hand the user a run id if it is still going.
        import contextvars
        import threading
        from ..engine.store import new_run_id
        store = default_store()
        run_id = new_run_id()
        sink = runner.ProgressSink(ctx, candidates[0].split(":")[-1])
        box: dict = {}

        def _worker() -> None:
            try:
                record = audition.run(
                    ctx.llm, source, settings, candidates,
                    store=store, progress=sink, run_id=run_id,
                    current_model=runner.current_model(),
                    profile=getattr(ctx, "profile_name", ""),
                    current_session_id=runner.current_session_id() or None)
                _write_artifacts(record, store)
                try:
                    store.save(record)
                except Exception:
                    pass
                box["record"] = record
                if record.get("failure") == "trust":
                    sink.post_final(runner.TRUST_HELP)
                else:
                    text = report_text.summary(record)
                    if record.get("report_html"):
                        text += f"\nFull report: {record['report_html']}"
                    sink.post_final(text)
            except Exception as exc:  # surfaced via box; nothing raised across threads
                box["error"] = exc
            finally:
                try:
                    source.close()
                except Exception:
                    pass
            done.set()

        task_ctx = contextvars.copy_context()
        done = threading.Event()
        thread = threading.Thread(target=lambda: task_ctx.run(_worker), daemon=True)
        thread.start()
        done.wait(timeout=300.0)
        if not done.is_set():
            return _json({
                "status": "running", "run_id": run_id,
                "note": "Still running in the background. Live progress is in this chat "
                        "and the scorecard will be posted here when it finishes. "
                        "Use understudy_report later.",
            })
        if "error" in box:
            exc = box["error"]
            if runner.is_trust_error(exc):
                return _err(runner.TRUST_HELP)
            return _err(f"audition failed: {type(exc).__name__}: {exc}")
        record = box["record"]
        if record.get("failure") == "trust":
            return _err(runner.TRUST_HELP)
        return _json(_compact_record(record))
    except Exception as exc:
        if runner.is_trust_error(exc):
            return _err(runner.TRUST_HELP)
        return _err(f"audition failed: {type(exc).__name__}: {exc}")


def report_handler(ctx: Any, args: dict, **_kw: Any) -> str:
    del ctx
    run_id = (args.get("run_id") or "").strip() or None
    detail = args.get("detail") or "summary"
    store = default_store()
    record = store.load(run_id)
    if record is None:
        return _err(f"no audition found{'' if run_id else ' yet'}"
                    + (f" for id {run_id!r}" if run_id else ""))
    out: dict[str, Any] = {"run_id": record.get("id"), "status": record.get("status"),
                           "summary": report_text.summary(record)}
    if detail in ("misses", "all"):
        text = report_text.details(record, detail)
        if len(text) > 12000:
            text = text[:12000] + "\n…[cut at 12,000 chars — open the HTML report for everything]"
        out["details"] = text
    if record.get("report_html"):
        out["report_html"] = record["report_html"]
    if record.get("card_png"):
        out["card_png"] = record["card_png"]
    return _json(out)
