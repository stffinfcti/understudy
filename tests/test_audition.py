"""Orchestrator: spec parsing, budget guard, cancel, end-to-end on stubs."""

import pytest

from understudy.engine import audition
from understudy.engine.audition import (
    _Budget, estimate, parse_candidate_spec, run,
)
from understudy.engine.store import RunStore

from conftest import assistant, session_dict, tool_call, user, StubSource, FakeLLM, FakeResult

NOW = 1_800_100_000.0


def _sessions_conv(n_sessions=3, decisions_per=8):
    sessions, convs = [], {}
    for i in range(n_sessions):
        sid = f"s{i}"
        sessions.append(session_dict(sid, started_at=NOW - 10000))
        conv = [user(f"q{i}")]
        for j in range(decisions_per):
            conv += [user(f"ask {j}"), assistant(f"answer {j}")]
        convs[sid] = conv
    return sessions, convs


def test_parse_candidate_spec():
    providers = {"openrouter", "anthropic"}
    assert parse_candidate_spec("openrouter:moonshotai/kimi", providers) == {
        "spec": "openrouter:moonshotai/kimi", "provider": "openrouter",
        "model": "moonshotai/kimi"}
    assert parse_candidate_spec("gpt-5-mini", providers) == {
        "spec": "gpt-5-mini", "provider": None, "model": "gpt-5-mini"}
    # unknown prefix stays a bare model id
    assert parse_candidate_spec("weird:model", providers)["provider"] is None
    # the fallback list knows common providers
    assert parse_candidate_spec("anthropic:claude-haiku")["provider"] == "anthropic"


def test_estimate_reports_counts():
    sessions, convs = _sessions_conv()
    src = StubSource(sessions, convs)
    est = estimate(src, {"sample": 10, "days": 14, "baseline_sample": 3},
                   ["openrouter:x"], now=NOW)
    assert est["decisions_available"] == 24
    assert est["sample_per_candidate"] == 10
    assert est["planned_calls_per_candidate"] == 16
    assert est["baseline_calls"] == 5


def _llm_mirror():
    llm = FakeLLM()
    def replay(kw, i):
        return FakeResult(parsed={"action": "reply", "reply": "answer 0"},
                          cost_usd=0.01)
    llm.replay_fn = replay
    return llm


def test_run_records_verdicts_and_persists(tmp_path):
    sessions, convs = _sessions_conv(4, 2)
    src = StubSource(sessions, convs)
    llm = _llm_mirror()
    store = RunStore(tmp_path)
    record = run(llm, src, {"sample": 8, "budget_usd": 5.0, "baseline_sample": 2},
                 ["openrouter:cand"], store=store, now=NOW, run_id="us_t1")
    assert record["status"] == "done"
    cand = record["candidates"][0]
    assert cand["scored"] == 8
    assert cand["counts"].get("AGREE", 0) + cand["counts"].get("BETTER", 0) \
        + cand["counts"].get("ALTERNATIVE", 0) + cand["counts"].get("EXACT", 0) \
        + cand["counts"].get("MINOR_MISS", 0) + cand["counts"].get("CRITICAL_MISS", 0) \
        + cand["counts"].get("INVALID", 0) == 8
    assert record["baseline"]["scored"] == 2
    saved = store.load("us_t1")
    assert saved["id"] == "us_t1" and saved["status"] == "done"
    replay_calls = [c for c in llm.calls if c.get("purpose") == "understudy.replay"]
    assert any(c.get("provider") == "openrouter" and c.get("model") == "cand"
               for c in replay_calls)
    baseline_calls = [c for c in replay_calls
                      if "provider" not in c and "model" not in c]
    assert len(baseline_calls) == 2


def test_exact_skips_judge():
    sessions, convs = _sessions_conv(1, 6)
    src = StubSource(sessions, convs)
    llm = FakeLLM()
    llm.replay_fn = lambda kw, i: FakeResult(
        parsed={"action": "reply", "reply": "answer 0"}, cost_usd=0.01)
    store_record = {}

    class NoopStore:
        def save(self, r):
            store_record["r"] = r

        def prune(self, keep=20):
            pass
    # force the candidate answer to equal production on a tool-call decision
    convs["s0"] = [user("do it"),
                   assistant(None, [tool_call("terminal", '{"command": "ls"}')]),
                   user("more"), assistant("ok")]
    llm.replay_fn = lambda kw, i: FakeResult(
        parsed={"action": "tool_calls",
                "tool_calls": [{"name": "terminal", "arguments": '{"command": "ls"}'}]},
        cost_usd=0.01)
    record = run(llm, src, {"sample": 4, "baseline_sample": 0},
                 ["cand"], store=NoopStore(), now=NOW, run_id="us_t2")
    judge_calls = [c for c in llm.calls if c.get("purpose") == "understudy.judge"]
    cand = record["candidates"][0]
    assert cand["counts"].get("EXACT", 0) == 1          # the terminal replay matched verbatim
    assert len(judge_calls) == 1                        # only the non-EXACT item was judged


def test_budget_guard_marks_skipped(tmp_path):
    sessions, convs = _sessions_conv(4, 8)
    src = StubSource(sessions, convs)
    llm = _llm_mirror()  # each replay costs 0.01 + judge calls add more
    store = RunStore(tmp_path)
    record = run(llm, src,
                 {"sample": 20, "budget_usd": 0.05, "concurrency": 1,
                  "baseline_sample": 0},
                 ["cand"], store=store, now=NOW, run_id="us_t3")
    cand = record["candidates"][0]
    skipped = cand["counts"].get("SKIPPED", 0)
    assert skipped > 0
    assert record["spent_usd"] <= 0.05 + 0.02  # last in-flight item may land just over
    done = sum(v for k, v in cand["counts"].items() if k != "SKIPPED")
    assert done < cand["attempted"] + done  # tautology guard; real check below
    assert done <= 5


def test_cancel_flag_marks_cancelled(tmp_path):
    sessions, convs = _sessions_conv(4, 8)
    src = StubSource(sessions, convs)
    llm = _llm_mirror()
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 3
    record = run(llm, src,
                 {"sample": 20, "budget_usd": 50.0, "concurrency": 1,
                  "baseline_sample": 0},
                 ["cand"], store=RunStore(tmp_path), now=NOW, run_id="us_t4",
                 cancel=cancel)
    assert record["status"] == "cancelled"


def test_invalid_and_error_verdicts(tmp_path):
    sessions, convs = _sessions_conv(3, 2)
    src = StubSource(sessions, convs)
    llm = FakeLLM()
    state = {"n": 0}

    def replay(kw, i):
        state["n"] += 1
        if state["n"] <= 2:
            return FakeResult(parsed=None, text="garbage", content_type="text",
                              cost_usd=0.01)
        raise RuntimeError("provider down")
    llm.replay_fn = replay
    record = run(llm, src, {"sample": 6, "baseline_sample": 0},
                 ["cand"], store=RunStore(tmp_path), now=NOW, run_id="us_t5")
    counts = record["candidates"][0]["counts"]
    assert counts.get("INVALID", 0) == 2
    assert counts.get("ERROR", 0) == 4
    assert record["errors"]


def test_progress_events(tmp_path):
    sessions, convs = _sessions_conv(3, 2)
    src = StubSource(sessions, convs)
    events = []
    record = run(_llm_mirror(), src,
                 {"sample": 5, "baseline_sample": 1},
                 ["cand"], store=RunStore(tmp_path), now=NOW, run_id="us_t6",
                 progress=events.append)
    kinds = [e["event"] for e in events]
    assert kinds[0] == "start" and kinds[-1] == "finished"
    assert sum(1 for k in kinds if k == "item") == len(record["decisions"]) + 1


def test_budget_math():
    b = _Budget(1.0)
    assert b.may_submit(0)
    for _ in range(10):
        b.record(0.4)
    assert not b.may_submit(0)
    b2 = _Budget(1.0)
    b2.record(0.9)
    assert not b2.may_submit(0)  # 0.9 + 0.9*(0+1) > 1.0
