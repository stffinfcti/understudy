"""Understudy dashboard backend — FastAPI router mounted at /api/plugins/understudy.

Routes: GET /runs, GET /runs/{id}, GET /runs/{id}/html, GET /estimate,
GET /settings, POST /runs, POST /runs/{id}/cancel. Runs happen on a worker
thread with the request's context copied over; live progress is pushed via
broadcast_plugin_event (and the desktop half polls /runs as a fallback).
"""

from __future__ import annotations

import contextvars
import threading
import time
from typing import Optional

from fastapi import APIRouter, Body, HTTPException, Query

import sys
from pathlib import Path as _Path

# The dashboard mounts this file via importlib.spec_from_file_location — a
# standalone module, not a package submodule — so relative imports fail. Add
# the plugin root so `import engine.*` resolves; nothing else is needed.
_ROOT = _Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

router = APIRouter()

_LOCK = threading.Lock()
_ACTIVE: dict[str, dict] = {}   # run_id -> {"cancel": bool, "started": float, "status": str}
_BROADCAST_MIN_S = 1.0


def _store():
    from engine.store import default_store
    return default_store()


def _settings() -> dict:
    from engine.audition import DEFAULT_SETTINGS
    out = dict(DEFAULT_SETTINGS)
    try:
        from hermes_cli.config import load_config
        raw = (((load_config() or {}).get("plugins") or {}).get("entries") or {}
               ).get("understudy") or {}
        for key, value in (raw.get("settings") or {}).items():
            if key in out and value is not None:
                out[key] = value
    except Exception:
        pass
    return out


def _source():
    from engine.corpus import HermesSessionSource
    return HermesSessionSource()


def _llm():
    from agent.plugin_llm import PluginLlm
    return PluginLlm(plugin_id="understudy")


def _current_model() -> dict:
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
        m = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
        return {"provider": m.get("provider") or "",
                "model": m.get("default") or m.get("model") or "",
                "base_url": m.get("base_url") or ""}
    except Exception:
        return {}


def _broadcast(event: str, payload: dict) -> None:
    try:
        from hermes_cli.plugin_events import broadcast_plugin_event
        broadcast_plugin_event("understudy", event, payload)
    except Exception:
        pass


def _stub(record: dict) -> dict:
    cands = []
    for c in record.get("candidates") or []:
        cands.append({"spec": c.get("spec"), "model": c.get("model"),
                      "takeover_rate": c.get("takeover_rate"),
                      "level": (c.get("recommendation") or {}).get("level")})
    return {"id": record.get("id"), "created_at": record.get("created_at"),
            "status": record.get("status"), "stopped_reason": record.get("stopped_reason"),
            "candidates": cands, "spent_usd": record.get("spent_usd"),
            "current_model": record.get("current_model"),
            "card_png": record.get("card_png"), "report_html": record.get("report_html")}


@router.get("/runs")
def list_runs() -> dict:
    runs = [_stub(r) for r in (_store().load(s["id"]) or s for s in _store().list_runs())]
    with _LOCK:
        for run_id, active in _ACTIVE.items():
            if not any(r["id"] == run_id for r in runs):
                runs.insert(0, {"id": run_id, "status": active.get("status", "running"),
                                "created_at": None, "candidates": [], "spent_usd": None})
    return {"runs": runs}


@router.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    record = _store().load(run_id)
    if record is None:
        raise HTTPException(404, f"no run {run_id}")
    try:
        from engine import report_text
        record = {**record, "summary": report_text.summary(record)}
    except Exception:
        pass
    return record


@router.get("/runs/{run_id}/html")
def get_run_html(run_id: str) -> dict:
    record = _store().load(run_id)
    if record is None:
        raise HTTPException(404, f"no run {run_id}")
    if record.get("report_html"):
        try:
            from pathlib import Path
            text = Path(record["report_html"]).read_text(encoding="utf-8")
            return {"html": text}
        except OSError:
            pass
    from engine import report_html
    import tempfile
    from pathlib import Path
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        report_html.render_html(record, Path(tmp.name))
        return {"html": Path(tmp.name).read_text(encoding="utf-8")}


@router.get("/estimate")
def estimate(candidates: str = Query(...), sample: Optional[int] = None,
             days: Optional[int] = None) -> dict:
    from engine import audition
    specs = [c.strip() for c in str(candidates or "").replace(" ", ",").split(",") if c.strip()]
    if not specs:
        raise HTTPException(400, "candidates is required")
    settings = _settings()
    if sample:
        settings["sample"] = sample
    if days:
        settings["days"] = days
    source = _source()
    try:
        return audition.estimate(source, settings, specs,
                                 current_model=_current_model())
    finally:
        try:
            source.close()
        except Exception:
            pass


@router.get("/settings")
def settings() -> dict:
    return _settings()


def _finish_artifacts(record: dict, store) -> None:
    try:
        from engine import write_report_artifacts as _write_artifacts_from_engine
        _write_artifacts_from_engine(record, store)
        store.save(record)
    except Exception:
        pass


@router.post("/runs")
def start_run(body: dict = Body(...)) -> dict:
    from engine import audition
    from engine.store import new_run_id
    candidates = body.get("candidates") or []
    if isinstance(candidates, str):
        candidates = [c.strip() for c in candidates.replace(" ", ",").split(",") if c.strip()]
    if not candidates or len(candidates) > 4:
        raise HTTPException(400, "1-4 candidates required")
    settings = _settings()
    for key in ("sample", "days"):
        if body.get(key):
            settings[key] = body[key]
    if body.get("budget_usd"):
        settings["budget_usd"] = body["budget_usd"]
    run_id = new_run_id()
    state = {"cancel": False, "status": "running", "started": time.time()}
    with _LOCK:
        _ACTIVE[run_id] = state
    store = _store()
    last_push = {"t": 0.0}
    ctx = contextvars.copy_context()

    def progress(event: dict) -> None:
        now = time.monotonic()
        kind = event.get("event")
        if kind == "finished" or (kind == "item" and now - last_push["t"] >= _BROADCAST_MIN_S):
            last_push["t"] = now
            _broadcast("progress", {**event, "run_id": run_id})

    def worker() -> None:
        source = None
        try:
            source = _source()
            record = audition.run(
                _llm(), source, settings, candidates,
                store=store, progress=progress, run_id=run_id,
                cancel=lambda: state["cancel"],
                current_model=_current_model())
            _finish_artifacts(record, store)
            state["status"] = record.get("status") or "done"
            _broadcast("finished", {"run_id": run_id, "status": state["status"],
                                    "spent_usd": record.get("spent_usd"),
                                    "card_png": record.get("card_png"),
                                    "report_html": record.get("report_html")})
        except Exception as exc:
            state["status"] = "failed"
            _broadcast("finished", {"run_id": run_id, "status": "failed",
                                    "error": f"{type(exc).__name__}: {exc}"})
        finally:
            if source is not None:
                try:
                    source.close()
                except Exception:
                    pass
            with _LOCK:
                _ACTIVE.pop(run_id, None)

    thread = threading.Thread(target=lambda: ctx.run(worker), daemon=True,
                              name=f"understudy-{run_id}")
    thread.start()
    return {"run_id": run_id}


@router.post("/runs/{run_id}/cancel")
def cancel_run(run_id: str) -> dict:
    with _LOCK:
        state = _ACTIVE.get(run_id)
        if state:
            state["cancel"] = True
            state["status"] = "cancelled"
            return {"run_id": run_id, "cancelled": True}
    raise HTTPException(404, f"no active run {run_id}")
