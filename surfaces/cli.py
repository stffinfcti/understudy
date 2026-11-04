"""``hermes understudy`` CLI: audition / estimate / list / show."""

from __future__ import annotations

import json
import sys
from typing import Any

from ..engine import audition, report_text
from ..engine.store import default_store
from . import runner


def _print(data: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    elif isinstance(data, str):
        print(data)
    else:
        print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def _settings(ctx: Any, args: Any) -> dict:
    overrides = {"sample": getattr(args, "sample", None),
                 "days": getattr(args, "days", None),
                 "budget_usd": getattr(args, "budget", None)}
    return runner.get_settings(ctx, {k: v for k, v in overrides.items() if v is not None})


def _progress_line(event: dict) -> None:
    if event.get("event") != "item":
        return
    done, total = event.get("done", 0), event.get("total", 0)
    spent = event.get("spent_usd")
    line = f"\rUnderstudy · {event.get('candidate') or ''} · {done}/{total} decisions"
    if isinstance(spent, (int, float)):
        line += f" · ${spent:.2f}"
    print(f"{line:<72}", end="", file=sys.stderr, flush=True)


def _audition(ctx: Any, args: Any, dry_run: bool) -> int:
    candidates = [c for c in getattr(args, "candidates", []) if str(c).strip()]
    if not candidates:
        print("understudy: at least one candidate is required", file=sys.stderr)
        return 2
    settings = _settings(ctx, args)
    as_json = bool(getattr(args, "json", False))
    try:
        source = runner.session_source()
    except Exception as exc:
        print(f"understudy: cannot open session history: {exc}", file=sys.stderr)
        return 1
    try:
        est = audition.estimate(source, settings, candidates,
                                current_session_id=runner.current_session_id() or None,
                                current_model=runner.current_model())
        if dry_run:
            _print(est, as_json)
            return 0
        if not est["decisions_available"]:
            print("Understudy found no eligible past decisions in the window; "
                  "try --days 30 or more.")
            return 1
        if not dry_run:
            trust = runner.preflight_trust(ctx, candidates, settings.get("judge_model") or "")
            if trust:
                print(trust, file=sys.stderr)
                return 1
        if not getattr(args, "yes", False):
            n, budget = est["sample_per_candidate"], est["budget_usd"]
            est_note = (f" Estimated ~${est['estimated_cost_usd']:.2f} at list prices."
                        if est.get("estimated_cost_usd") is not None else "")
            answer = input(
                f"Audition {', '.join(candidates)} on {n} of your past decisions?{est_note} "
                f"Spending stops at ${budget:.2f}. [y/N] ").strip().lower()
            if answer not in ("y", "yes"):
                print("Audition not run. Rerun with --yes to skip the question.")
                return 1
        store = default_store()
        record = audition.run(
            ctx.llm, source, settings, candidates,
            store=store, progress=None if as_json else _progress_line,
            current_model=runner.current_model(), profile=getattr(ctx, "profile_name", ""),
            current_session_id=runner.current_session_id() or None)
        from .tools import _write_artifacts
        _write_artifacts(record, store)
        try:
            store.save(record)
        except Exception:
            pass
        if not as_json:
            print("", file=sys.stderr)
        if record.get("failure") == "trust":
            print(runner.TRUST_HELP, file=sys.stderr)
            return 1
        if as_json:
            _print(record, True)
        else:
            text = report_text.summary(record)
            if record.get("report_html"):
                text += f"\nFull report: {record['report_html']}"
            print(text)
        return 0 if record.get("status") == "done" else 1
    except Exception as exc:
        if runner.is_trust_error(exc):
            print(runner.TRUST_HELP, file=sys.stderr)
            return 1
        print(f"understudy: audition failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            source.close()
        except Exception:
            pass


def make_handler(ctx: Any):
    def handler(args: Any) -> int:
        sub = getattr(args, "understudy_command", None)
        if sub == "audition":
            return _audition(ctx, args, dry_run=False)
        if sub == "estimate":
            return _audition(ctx, args, dry_run=True)
        if sub == "list":
            runs = default_store().list_runs()
            _print(runs if getattr(args, "json", False) else report_text.runs_list(runs),
                   getattr(args, "json", False))
            return 0
        if sub == "show":
            record = default_store().load(getattr(args, "run_id", None) or None)
            if record is None:
                print("understudy: no audition found", file=sys.stderr)
                return 1
            if getattr(args, "json", False):
                _print(record, True)
            elif getattr(args, "misses", False):
                print(report_text.details(record, "misses"))
            else:
                print(report_text.summary(record))
            return 0
        print("Usage: hermes understudy <audition|estimate|list|show> ...", file=sys.stderr)
        return 2
    return handler


def setup(subparser: Any, ctx: Any) -> None:
    subs = subparser.add_subparsers(dest="understudy_command")

    def _common(p: Any) -> None:
        p.add_argument("--sample", type=int, help="past decisions per candidate")
        p.add_argument("--days", type=int, help="days of history to sample")
        p.add_argument("--budget", type=float, help="hard spend cap in USD")

    p = subs.add_parser("audition", help="Run an audition (spends tokens)")
    p.add_argument("candidates", nargs="+", help="provider:model or bare model ids")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    p.add_argument("--json", action="store_true", help="print the full run record")
    _common(p)

    p = subs.add_parser("estimate", help="Show what an audition would do; spends nothing")
    p.add_argument("candidates", nargs="+", help="provider:model or bare model ids")
    p.add_argument("--json", action="store_true")
    _common(p)

    p = subs.add_parser("list", help="Recent auditions")
    p.add_argument("--json", action="store_true")

    p = subs.add_parser("show", help="Show an audition (latest by default)")
    p.add_argument("run_id", nargs="?", help="audition id")
    p.add_argument("--misses", action="store_true", help="list missed decisions")
    p.add_argument("--json", action="store_true")

    subparser.set_defaults(func=make_handler(ctx))
