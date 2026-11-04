"""Judge mapping and call shape."""

from understudy.engine import judge as judge_mod
from understudy.engine.judge import (
    AGREE, ALTERNATIVE, BETTER, CRITICAL_MISS, MINOR_MISS, UNJUDGED,
    call_judge, map_verdict,
)

from conftest import FakeLLM, FakeResult, assistant, tool_call, user


def _judge_result(verdict, severity="none", reason="r"):
    return FakeResult(parsed={"verdict": verdict, "severity": severity,
                              "reason": reason})


def test_map_verdict_candidate_perspective():
    # candidate is step 1
    assert map_verdict({"verdict": "step1_better", "severity": "major"}, True)[0] == BETTER
    assert map_verdict({"verdict": "step2_better", "severity": "major"}, True)[0] == CRITICAL_MISS
    assert map_verdict({"verdict": "step2_better", "severity": "minor"}, True)[0] == MINOR_MISS
    assert map_verdict({"verdict": "step2_better", "severity": "none"}, True)[0] == MINOR_MISS
    # candidate is step 2 (swapped)
    assert map_verdict({"verdict": "step2_better", "severity": "major"}, False)[0] == BETTER
    assert map_verdict({"verdict": "step1_better", "severity": "major"}, False)[0] == CRITICAL_MISS
    # equivalent / both_fine ignore severity entirely
    assert map_verdict({"verdict": "equivalent", "severity": "major"}, True)[0] == AGREE
    assert map_verdict({"verdict": "both_fine", "severity": "major"}, False)[0] == ALTERNATIVE
    assert map_verdict({"verdict": "bogus"}, True)[0] == UNJUDGED
    assert map_verdict(None, True)[0] == UNJUDGED


def test_call_judge_swaps_steps_by_seed(monkeypatch):
    seen = []

    def fake_random(seed):
        class R:
            def random(self):
                return 0.9  # >= 0.5 -> candidate is step 2
        return R()
    monkeypatch.setattr(judge_mod.random, "Random", fake_random)
    llm = FakeLLM()
    llm.judge_fn = lambda kw, i: _judge_result("step2_better", "major", "cand wins")
    prod = ("tool_calls", [("terminal", {"command": "ls"})])
    cand = ("tool_calls", [("terminal", {"command": "pwd"})])
    verdict, reason, is_step1, _ = call_judge(
        llm, transcript="[user] q", first_user="[user] q",
        production=prod, candidate=cand, seed="x")
    assert is_step1 is False
    assert verdict == BETTER
    prompt = llm.calls[0]["input"][0]["text"]
    assert "STEP 1:" in prompt and "STEP 2:" in prompt
    assert prompt.index("pwd") > prompt.index('"command": "ls"')  # cand rendered as step 2


def test_call_judge_kwargs():
    llm = FakeLLM()
    call_judge(llm, transcript="t", first_user=None,
               production=("reply", "a"), candidate=("reply", "b"),
               seed="s", provider="openrouter", model="m/x")
    kw = llm.calls[0]
    assert kw["purpose"] == "understudy.judge"
    assert kw["temperature"] == 0 and kw["max_tokens"] == 300
    assert kw["provider"] == "openrouter" and kw["model"] == "m/x"
    assert kw["json_schema"] is judge_mod.JUDGE_SCHEMA


def test_judge_exception_propagates():
    llm = FakeLLM()
    llm.judge_fn = lambda kw, i: RuntimeError("boom")
    try:
        call_judge(llm, transcript="t", first_user=None,
                   production=("reply", "a"), candidate=("reply", "b"), seed="s")
    except RuntimeError:
        pass
    else:
        raise AssertionError("judge failure should propagate for UNJUDGED mapping")


def test_excerpt_cap_and_first_user_prefix():
    t = "x" * 20000
    out = judge_mod.build_excerpt(t, "[user] first")
    assert len(out) <= judge_mod.EXCERPT_CAP + 20
    assert out.startswith("[user] first")
    short = judge_mod.build_excerpt("tiny", "[user] first")
    assert short == "tiny"


def test_first_user_line():
    msgs = [assistant("a"), user("the question"), assistant("b")]
    assert judge_mod.first_user_line(msgs) == "[user] the question"
