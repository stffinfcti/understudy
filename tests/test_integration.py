"""Full audition against a real SessionDB populated through its public write API."""

import time

import pytest

from conftest import FakeLLM, FakeResult

pytestmark = pytest.mark.skipif(
    __import__("importlib.util", fromlist=["util"]).find_spec("hermes_state") is None,
    reason="hermes_state not importable on this interpreter")


def _make_db(tmp_path):
    from hermes_state import SessionDB
    return SessionDB(db_path=tmp_path / "state.db")


def _tc(name, args, call_id):
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": args}}


def test_full_audition(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    db = _make_db(tmp_path)
    now = time.time()
    try:
        # Session 1: coding session
        db.create_session("sess_code", "cli", model="anthropic/claude-x",
                          system_prompt="You are Hermes, a coding agent.")
        db.append_message("sess_code", "user", "fix the bug in main.py")
        db.append_message("sess_code", "assistant", None,
                          tool_calls=[_tc("read_file", '{"path": "main.py"}', "c1")])
        db.append_message("sess_code", "tool", "print('bug')", tool_name="read_file",
                          tool_call_id="c1")
        db.append_message("sess_code", "assistant", None,
                          tool_calls=[_tc("write_file", '{"path": "main.py", "content": "print(1)"}', "c2")])
        db.append_message("sess_code", "tool", "ok", tool_name="write_file",
                          tool_call_id="c2")
        db.append_message("sess_code", "assistant", "Fixed it.")
        # Session 2: chat
        db.create_session("sess_chat", "cli", model="anthropic/claude-x")
        db.append_message("sess_chat", "user", "what is rust?")
        db.append_message("sess_chat", "assistant", "A systems language.")
        db.append_message("sess_chat", "user", "and go?")
        db.append_message("sess_chat", "assistant", "Also one.")
        # Session 3: research + a subagent session that must be ignored
        db.create_session("sess_research", "telegram", model="anthropic/claude-x")
        db.append_message("sess_research", "user", "find news on x")
        db.append_message("sess_research", "assistant", None,
                          tool_calls=[_tc("web_search", '{"q": "x news"}', "c3")])
        db.append_message("sess_research", "tool", "results", tool_name="web_search",
                          tool_call_id="c3")
        db.append_message("sess_research", "assistant", "Here's what I found.")
        db.create_session("sess_sub", "subagent", model="anthropic/claude-x")
        db.append_message("sess_sub", "user", "child task")
        db.append_message("sess_sub", "assistant", "child answer")
        db.update_token_counts("sess_code", estimated_cost_usd=4.0, absolute=True)
        db.update_token_counts("sess_chat", actual_cost_usd=2.0, absolute=True)
    finally:
        db.close()

    from understudy.engine import audition
    from understudy.engine.corpus import HermesSessionSource
    from understudy.engine.store import RunStore

    llm = FakeLLM()
    llm.replay_fn = lambda kw, i: FakeResult(
        parsed={"action": "reply", "reply": "a plausible step"}, cost_usd=0.005)
    store = RunStore(tmp_path / "runs")
    source = HermesSessionSource(db=_make_db(tmp_path))
    try:
        record = audition.run(
            llm, source,
            {"sample": 20, "days": 14, "budget_usd": 5.0, "baseline_sample": 3,
             "concurrency": 2},
            ["openrouter:cand-a", "cand-b"],
            store=store, now=now, run_id="us_int",
            current_session_id="never-matches",
            current_model={"provider": "anthropic", "model": "claude-x"})
    finally:
        source.close()

    assert record["status"] == "done"
    assert record["corpus"]["sessions"] == 3
    assert {d["session_id"] for d in record["decisions"]} <= {
        "sess_code", "sess_chat", "sess_research"}
    assert not any(d["session_id"] == "sess_sub" for d in record["decisions"])
    cats = {d["category"] for d in record["decisions"]}
    assert "coding" in cats and "research" in cats
    assert record["corpus"]["spend_30d_usd"] == 6.0
    assert len(record["candidates"]) == 2
    for cand in record["candidates"]:
        assert cand["scored"] == len(record["decisions"])
        assert cand["takeover_rate"] is not None
        assert cand["cost_per_decision"] is not None
    assert record["baseline"]["scored"] == 3
    saved = store.load("us_int")
    assert saved["status"] == "done"
    # judge verdicts equivalent -> AGREE everywhere
    for cand in record["candidates"]:
        assert cand["counts"].get("AGREE", 0) == cand["scored"]
        assert cand["recommendation"]["level"] in ("ready", "partial",
                                                   "not_ready", "insufficient")
