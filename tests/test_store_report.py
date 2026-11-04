"""Run-store persistence + retention, and text report checks."""

import json

from understudy.engine import report_text
from understudy.engine.store import RunStore, new_run_id


def _record(run_id, spent=0.5):
    return {"id": run_id, "status": "done", "created_at": "2026-01-01T00:00:00+00:00",
            "current_model": {"provider": "anthropic", "model": "claude-x"},
            "corpus": {"sessions": 3, "window_days": 14, "spend_30d_usd": 30.0,
                       "spend_known": True},
            "candidates": [{"spec": "openrouter:c", "provider": "openrouter", "model": "c",
                            "scored": 10, "takeover_rate": 0.9, "ci95": [0.6, 0.99],
                            "counts": {"EXACT": 5, "AGREE": 4, "MINOR_MISS": 1},
                            "critical_rate": 0.0, "minor_rate": 0.1,
                            "by_category": {"chat": {"n": 10, "takeover_rate": 0.9}},
                            "cost_ratio": 0.1, "projected_monthly_usd": 3.0,
                            "savings_monthly_usd": 27.0,
                            "recommendation": {"level": "ready", "text": "good"},
                            "unreliable": False}],
            "baseline": {"scored": 4, "takeover_rate": 0.75,
                         "counts": {"EXACT": 3, "MINOR_MISS": 1}},
            "decisions": [{"id": "s1:1", "category": "chat", "kind": "reply",
                           "excerpt": "hello", "production": "REPLY TO USER:\nhi",
                           "replays": {"openrouter:c": {"verdict": "MINOR_MISS",
                                                        "reason": "clumsy",
                                                        "rendered": "REPLY TO USER:\nyo"}}}],
            "spent_usd": spent, "budget_usd": 1.5}


def test_store_round_trip(tmp_path):
    store = RunStore(tmp_path)
    rec = _record("us_20260101_000000_abcd")
    store.save(rec)
    loaded = store.load("us_20260101_000000_abcd")
    assert loaded["id"] == rec["id"]
    assert loaded["candidates"][0]["takeover_rate"] == 0.9
    assert store.load(None)["id"] == rec["id"]          # latest
    assert store.load("nonexistent") is None
    # prefix resolution
    assert store.load("us_20260101")["id"] == rec["id"]


def test_store_keeps_newest_20(tmp_path):
    store = RunStore(tmp_path)
    for i in range(25):
        store.save(_record(f"us_202601{i:02d}_000000_x{i:02d}"))
    store.prune(keep=20)
    runs = store.list_runs()
    assert len(runs) == 20
    assert runs[0]["id"] == "us_20260124_000000_x24"
    assert not (tmp_path / "us_20260100_000000_x00.json").exists()


def test_run_id_format():
    rid = new_run_id()
    assert rid.startswith("us_") and len(rid.rsplit("_", 1)[1]) == 4


def test_summary_text_has_key_numbers():
    text = report_text.summary(_record("us_x"))
    assert "us_x" in text
    assert "90%" in text and "(9/10)" in text
    assert "95% CI 60–99%" in text
    assert "1 critical, 1 minor" in text or "0 critical, 1 minor" in text
    assert "chat 90%" in text
    assert "10% of current" in text and "$3.00/mo" in text and "saves ~$27.00/mo" in text
    assert "Ready" in text
    assert "3/4 re-asked" in text
    assert "$30.00" in text and "$0.50 of $1.50" in text


def test_summary_spend_not_recorded():
    rec = _record("us_y")
    rec["corpus"]["spend_30d_usd"] = None
    rec["corpus"]["spend_known"] = False
    rec["candidates"][0].update(cost_ratio=None, projected_monthly_usd=None,
                              savings_monthly_usd=None)
    text = report_text.summary(rec)
    assert "spend not recorded" in text
    assert "Cost:" not in text  # unknown parts omitted


def test_details_misses_only():
    rec = _record("us_z")
    text = report_text.details(rec, "misses")
    assert "s1:1" in text and "clumsy" in text
    rec["decisions"][0]["replays"]["openrouter:c"]["verdict"] = "AGREE"
    assert "No critical or minor misses" in report_text.details(rec, "misses")
    assert "s1:1" in report_text.details(rec, "all")


def test_runs_list():
    assert "No Understudy" in report_text.runs_list([])
    text = report_text.runs_list([{"id": "us_a", "status": "done",
                                   "candidates": ["m"], "spent_usd": 0.5,
                                   "created_at": "t"}])
    assert "us_a" in text and "m" in text
