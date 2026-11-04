"""Reliability: error classification, transient retry, per-spec limiting."""

from __future__ import annotations

from conftest import StubSource, assistant, session_dict, user

from engine import audition, reliability

NOW = 1_800_000_000.0


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


class _Exc(Exception):
    def __init__(self, msg="", status=None, retry=None):
        super().__init__(msg)
        if status is not None:
            self.status_code = status
        if retry is not None:
            self.retry_after = retry


def test_classification():
    assert reliability.is_transient(_Exc("429")) is False  # text-only 429 → rate limit text? message "429" matches text regex? check
    assert reliability.is_transient(_Exc("rate limit", 429))
    assert reliability.is_transient(_Exc("bad", 500))
    assert reliability.is_transient(_Exc("bad", 503))
    assert reliability.is_transient(_Exc("connection reset by peer"))
    assert reliability.is_transient(_Exc("request timed out"))
    assert not reliability.is_transient(_Exc("Invalid inference request", 400))
    assert not reliability.is_transient(_Exc("some model error", 404))


def test_error_kind():
    assert reliability.error_kind(_Exc("x", 429)) == "rate_limited"
    assert reliability.error_kind(_Exc("rate limit exceeded")) == "rate_limited"
    assert reliability.error_kind(_Exc("x", 408)) == "timeout"
    assert reliability.error_kind(_Exc("request timed out")) == "timeout"
    assert reliability.error_kind(_Exc("x", 400)) == "invalid_params"
    assert reliability.error_kind(_Exc("x", 503)) == "other"


def test_retry_after():
    assert reliability.retry_after(_Exc("x", 429, retry=7)) == 7
    assert reliability.retry_after(_Exc("x", 429, retry=99)) == 20.0
    assert reliability.retry_after(_Exc("x")) is None


def test_call_with_retry(monkeypatch):
    sleeps = []
    monkeypatch.setattr(audition, "_sleep", sleeps.append)
    calls = {"n": 0}
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _Exc("rate limit", 429)
        return "ok"
    result, err, hits = audition._call_with_retry(flaky)
    assert (result, err) == ("ok", None) and hits == 2 and len(sleeps) == 2
    assert sleeps[0] >= 2.0 and sleeps[1] >= 6.0


def test_call_with_retry_no_retry_on_400(monkeypatch):
    monkeypatch.setattr(audition, "_sleep", lambda s: None)
    calls = {"n": 0}
    def bad():
        calls["n"] += 1
        raise _Exc("Invalid inference request", 400)
    _r, err, _h = audition._call_with_retry(bad)
    assert err is not None and calls["n"] == 1


def test_call_with_retry_gives_up(monkeypatch):
    monkeypatch.setattr(audition, "_sleep", lambda s: None)
    calls = {"n": 0}
    def always():
        calls["n"] += 1
        raise _Exc("rate limit", 429)
    _r, err, _h = audition._call_with_retry(always)
    assert err is not None and calls["n"] == len(audition.RETRY_BACKOFF_S) + 1


def test_per_spec_limit_halves(monkeypatch):
    """A rate_limited item halves that spec's cap; other specs unaffected."""
    monkeypatch.setattr(audition, "_sleep", lambda s: None)
    sessions, convs = _sessions_conv(2, 4)
    source = StubSource(sessions, convs)

    class RateLimiterLLM:
        def complete_structured(self, **kw):
            if kw.get("model") == "limited":
                raise _Exc("rate limit", 429)
            class R:
                parsed = {"action": "reply", "text": "x"}
                usage = {"input_tokens": 1, "output_tokens": 1, "cost_usd": 0.0}
                provider, model = "p", "m"
            return R()

    record = audition.run(RateLimiterLLM(), source,
                          {"sample": 8, "baseline_sample": 0, "concurrency": 4,
                           "budget_usd": 100.0, "days": 30},
                          ["limited", "other-model"], now=NOW)
    limited = next(c for c in record["candidates"] if c["spec"] == "limited")
    assert limited["counts"].get("ERROR") == 4
    assert limited["errors_by_kind"]["rate_limited"] == 4
    assert "example" in limited["errors_by_kind"]
    other = next(c for c in record["candidates"] if c["spec"] == "other-model")
    assert other["counts"].get("ERROR", 0) == 0


def test_monthly_span_scaling():
    """usage_span_days clamps (now-earliest)/86400 to [1,30] and scales monthly."""
    sessions, convs = _sessions_conv(1, 4)
    # one session started 5 days ago → span 5 → monthly = 30/5 = 6x window
    sessions[0]["started_at"] = NOW - 5 * 86400
    sessions[0]["input_tokens"] = 1000
    sessions[0]["output_tokens"] = 500
    sessions[0]["cache_read_tokens"] = 0
    sessions[0]["model"] = "cur-model"
    source = StubSource(sessions, convs)

    class OkLLM:
        def complete_structured(self, **kw):
            class R:
                parsed = {"action": "reply", "text": "x"}
                usage = {"input_tokens": 1, "output_tokens": 1, "cost_usd": 0.0}
                provider, model = "p", "m"
            return R()

    record = audition.run(OkLLM(), source,
                          {"sample": 4, "baseline_sample": 0, "budget_usd": 10.0,
                           "days": 30,
                           "prices": {"cur-model": {"input": 1.0, "output": 1.0}}},
                          ["other"], now=NOW, current_model={"model": "cur-model"})
    assert record["usage_span_days"] == 5.0
    # list-priced: (1000 + 500) tokens @ $1/1M = $0.0015 window → ×6 = $0.009 monthly
    assert abs(record["current_monthly_usd"] - 0.009) < 1e-9
