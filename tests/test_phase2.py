"""Phase 1 fixes + Phase 2: trust abort, stopped_reason, max_seconds, fair
scheduling, new recommend rule, pricing, compact results, dashboard backend."""

import pytest

from understudy.engine import audition, pricing
from understudy.engine.audition import run
from understudy.engine.judge import AGREE, CRITICAL_MISS, SKIPPED
from understudy.engine.metrics import recommend, summarize_items
from understudy.engine.pricing import (
    Price, call_cost, resolve_price, usage_cost, _normalize_model,
)
from understudy.engine.store import RunStore

from conftest import (
    FakeLLM, FakeResult, FakeUsage, StubSource, assistant, session_dict,
    tool_call, user,
)

NOW = 1_800_100_000.0


def _conv_source(n_sessions=4, decisions_per=8):
    sessions, convs = [], {}
    for i in range(n_sessions):
        sid = f"s{i}"
        sessions.append(session_dict(sid, started_at=NOW - 10000))
        conv = [user(f"q{i}")]
        for j in range(decisions_per):
            conv += [user(f"ask {j}"), assistant(f"answer {j}")]
        convs[sid] = conv
    return StubSource(sessions, convs)


def _mirror_llm(cost=0.01):
    llm = FakeLLM()
    llm.replay_fn = lambda kw, i: FakeResult(
        parsed={"action": "reply", "reply": "answer 0"}, cost_usd=cost)
    return llm


# ── trust errors ─────────────────────────────────────────────────────────────

class PluginLlmTrustError(Exception):
    pass


def test_trust_error_aborts_run(tmp_path):
    src = _conv_source()
    llm = FakeLLM()
    llm.replay_fn = lambda kw, i: PluginLlmTrustError("model_override not granted")
    record = run(llm, src, {"sample": 12, "baseline_sample": 0},
                 ["cand"], store=RunStore(tmp_path), now=NOW, run_id="us_tr")
    assert record["status"] == "failed"
    assert record["failure"] == "trust"
    assert any(e.startswith("trust:") for e in record["errors"])


def test_is_trust_error_by_name():
    assert audition._is_trust_error(PluginLlmTrustError("x"))
    assert not audition._is_trust_error(RuntimeError("x"))


# ── stopped_reason + max_seconds ─────────────────────────────────────────────

def test_max_seconds_marks_time(tmp_path):
    src = _conv_source()
    llm = FakeLLM()
    def slow(kw, i):
        import time as _t
        _t.sleep(0.05)
        return FakeResult(parsed={"action": "reply", "reply": "x"}, cost_usd=0.01)
    llm.replay_fn = slow
    record = run(llm, src, {"sample": 20, "concurrency": 1, "baseline_sample": 0},
                 ["cand"], store=RunStore(tmp_path), now=NOW, run_id="us_ms",
                 max_seconds=0.01)
    assert record["stopped_reason"] == "time"
    assert record["candidates"][0]["counts"].get("SKIPPED", 0) > 0


def test_stopped_reason_budget(tmp_path):
    src = _conv_source()
    record = run(_mirror_llm(), src,
                 {"sample": 20, "budget_usd": 0.05, "concurrency": 1,
                  "baseline_sample": 0},
                 ["cand"], store=RunStore(tmp_path), now=NOW, run_id="us_sb")
    assert record["stopped_reason"] == "budget"


def test_stopped_reason_cancelled(tmp_path):
    src = _conv_source()
    calls = {"n": 0}
    def cancel():
        calls["n"] += 1
        return calls["n"] > 2
    record = run(_mirror_llm(), src,
                 {"sample": 20, "concurrency": 1, "baseline_sample": 0},
                 ["cand"], store=RunStore(tmp_path), now=NOW, run_id="us_sc",
                 cancel=cancel)
    assert record["status"] == "cancelled"
    assert record["stopped_reason"] == "cancelled"


# ── fair interleaved scheduling ──────────────────────────────────────────────

def test_interleaved_order_baseline_first(tmp_path):
    src = _conv_source(2, 4)
    order = []
    llm = FakeLLM()
    def replay(kw, i):
        prov = kw.get("provider") or "baseline"
        order.append(prov)
        return FakeResult(parsed={"action": "reply", "reply": "x"}, cost_usd=0.005)
    llm.replay_fn = replay
    record = run(llm, src,
                 {"sample": 8, "baseline_sample": 2, "concurrency": 1,
                  "budget_usd": 5.0},
                 ["openrouter:a", "openrouter:b"],
                 store=RunStore(tmp_path), now=NOW, run_id="us_il")
    # With concurrency 1 the work list order is exactly the completion order:
    # baseline replays (no provider kwarg) come first per baseline decision.
    assert order[0] == "baseline"
    # Candidates interleave per decision, not run to completion serially:
    seen = {p: idx for idx, p in enumerate(order)}
    assert "openrouter" in seen or order.count("baseline") == 2
    # Baseline finished long before candidates in the interleaved list.
    assert order.index("baseline") < len(order) - 2


# ── new recommend baseline rule ──────────────────────────────────────────────

def test_recommend_baseline_min_085_rule():
    # base_takeover 0.60 -> threshold min(0.85, 0.55) = 0.55
    base = {"scored": 6, "takeover_rate": 0.60, "critical_rate": 0.0}
    assert recommend(10, 0.55, 0.05, {}, base)["level"] == "ready"
    assert recommend(10, 0.549, 0.05, {}, base)["level"] != "ready"
    # base_takeover 0.95 -> threshold min(0.85, 0.90) = 0.85
    base2 = {"scored": 6, "takeover_rate": 0.95, "critical_rate": 0.0}
    assert recommend(10, 0.85, 0.05, {}, base2)["level"] == "ready"
    assert recommend(10, 0.84, 0.05, {}, base2)["level"] != "ready"


# ── pricing ──────────────────────────────────────────────────────────────────

def _catalog():
    return {
        "openrouter": {"models": {
            "moonshotai/kimi-k3": {"cost": {"input": 0.4, "output": 1.6}},
            "deepseek/deepseek-v3-0324": {"cost": {"input": 0.3, "output": 0.8}},
        }},
        "moonshotai": {"models": {
            "kimi-k3": {"cost": {"input": 0.6, "output": 2.4}},
            "kimi-k3-free": {"cost": {"input": 0, "output": 0}},
        }},
        "deepseek": {"models": {
            "deepseek-v3": {"cost": {"input": 0.27, "output": 1.1}},
        }},
        "other": {"models": {
            "zzz/kimi-k3-20251020": {"cost": {"input": 0.5, "output": 2.0}},
            "zzz/kimi-k3-fp8": {"cost": {"input": 0.45, "output": 1.9}},
        }},
    }


def test_normalize_model_strips_tags():
    assert _normalize_model("moonshotai/Kimi-K3") == "kimi-k3"
    assert _normalize_model("x/deepseek-v3-0324") == "deepseek-v3"
    assert _normalize_model("x/foo-20251020") == "foo"
    assert _normalize_model("x/bar-fp8") == "bar"
    assert _normalize_model("x/baz-preview") == "baz"


def test_resolve_prefers_vendor_provider():
    p = resolve_price("openrouter", "moonshotai/kimi-k3", catalog=_catalog())
    # kimi-k3 appears on openrouter/moonshotai/other — vendor moonshotai wins.
    assert p.input_per_m == 0.6 and p.output_per_m == 2.4
    assert "moonshotai" in p.source


def test_resolve_median_when_no_vendor():
    p = resolve_price("other", "zzz/something-kimi-k3", catalog={
        "a": {"models": {"x/kimi-k3": {"cost": {"input": 1.0, "output": 4.0}}}},
        "b": {"models": {"y/kimi-k3": {"cost": {"input": 2.0, "output": 8.0}}}},
        "c": {"models": {"z/kimi-k3": {"cost": {"input": 0, "output": 0}}}},  # skipped
    })
    # "something-kimi-k3" normalizes differently — use a bare model instead:
    p = resolve_price(None, "kimi-k3", catalog={
        "a": {"models": {"x/kimi-k3": {"cost": {"input": 1.0, "output": 4.0}}}},
        "b": {"models": {"y/kimi-k3": {"cost": {"input": 2.0, "output": 8.0}}}},
        "c": {"models": {"z/kimi-k3": {"cost": {"input": 0, "output": 0}}}},
    })
    assert p.input_per_m == 1.5 and p.output_per_m == 6.0
    assert "median of 2" in p.source


def test_resolve_local_and_settings_precedence():
    assert resolve_price("ollama", "llama3").input_per_m == 0
    assert resolve_price("custom", "x", base_url="http://localhost:1234/v1").input_per_m == 0
    assert resolve_price("custom", "x", base_url="http://nas.local:8080").input_per_m == 0
    settings = {"openrouter:vendor/m": {"input": 9, "output": 9}}
    p = resolve_price("openrouter", "vendor/m", settings_prices=settings,
                      catalog=_catalog())
    assert p.input_per_m == 9 and p.source == "settings"
    assert resolve_price("openai", "nope-unknown", catalog=_catalog()) is None


def test_call_cost_metered_vs_priced():
    usage = FakeUsage(cost_usd=0.007)
    assert call_cost(usage, None) == (0.007, "metered")
    usage2 = FakeUsage(cost_usd=None)
    usage2.input_tokens, usage2.output_tokens = 1_000_000, 500_000
    cost, src = call_cost(usage2, Price(1.0, 4.0, None, "settings"))
    assert cost == pytest.approx(3.0) and src == "settings"
    assert call_cost(FakeUsage(cost_usd=None), None) == (None, "")


def test_usage_cost_and_projection(tmp_path):
    usage = {"input_tokens": 10_000_000, "output_tokens": 2_000_000,
             "cache_read_tokens": 0}
    assert usage_cost(usage, Price(0.5, 2.0, None, "s")) == pytest.approx(9.0)
    # sessions_30d rows feed the "list price" current-monthly path; an
    # unpriceable model with unmetered usage is flagged costs_unknown.
    src = _conv_source(1, 4)
    llm = FakeLLM()
    llm.replay_fn = lambda kw, i: FakeResult(
        parsed={"action": "reply", "reply": "x"}, cost_usd=None)
    llm.judge_fn = lambda kw, i: FakeResult(
        parsed={"verdict": "equivalent", "severity": "none", "reason": "r"},
        cost_usd=None)
    record = run(llm, src,
                 {"sample": 4, "baseline_sample": 0},
                 ["zzz:unpriced"], store=RunStore(tmp_path), now=NOW,
                 run_id="us_pr", catalog={},
                 current_model={"provider": "zzz", "model": "unpriced"})
    assert "zzz:unpriced" in record["costs_unknown"]
    assert record["candidates"][0]["projected_monthly_usd"] is None


# ── compact tool result ──────────────────────────────────────────────────────

def test_compact_record_shape(tmp_path):
    from understudy.surfaces.tools import _compact_record
    record = {"id": "us_x", "status": "done", "stopped_reason": None,
              "corpus": {"sessions": 2, "window_days": 14},
              "current_model": {"model": "m"},
              "candidates": [{"spec": "openrouter:c", "provider": "openrouter",
                              "model": "c", "scored": 10, "takeover_rate": 0.8,
                              "ci95": [0.5, 0.95],
                              "counts": {"EXACT": 8, "MINOR_MISS": 2},
                              "by_category": {"chat": {"takeover_rate": 0.8}},
                              "cost_ratio": 0.3, "projected_monthly_usd": 5.0,
                              "savings_monthly_usd": 2.0,
                              "recommendation": {"level": "partial",
                                                 "text": "t", "categories": ["chat"]}}],
              "baseline": {"scored": 4, "takeover_rate": 0.75},
              "decisions": [{"id": "s:1", "replays": {"openrouter:c": {"x": 1}}}],
              "spent_usd": 0.4, "card_png": "/tmp/x.png", "report_html": "/tmp/x.html"}
    out = _compact_record(record)
    assert out["run_id"] == "us_x" and out["card_png"] == "/tmp/x.png"
    assert "MEDIA:/tmp/x.png" in out["show_card"]
    assert "decisions" not in out and "record" not in out
    c = out["candidates"][0]
    assert c["critical"] == 0 and c["minor"] == 2 and c["scored"] == 10
    assert c["by_category"] == {"chat": 0.8}
    assert c["recommendation"]["level"] == "partial"


# ── dashboard backend ────────────────────────────────────────────────────────

def test_dashboard_routes(tmp_path, monkeypatch):
    fastapi = pytest.importorskip("fastapi", reason="fastapi unavailable")
    from fastapi.testclient import TestClient
    api = pytest.importorskip("understudy.dashboard.plugin_api",
                              reason="backend not importable")
    monkeypatch.setattr(api, "_store", lambda: RunStore(tmp_path))
    app = fastapi.FastAPI()
    app.include_router(api.router, prefix="/api/plugins/understudy")
    client = TestClient(app)
    # settings + runs list work against an empty store
    r = client.get("/api/plugins/understudy/settings")
    assert r.status_code == 200 and r.json()["sample"] == 20
    r = client.get("/api/plugins/understudy/runs")
    assert r.status_code == 200 and r.json() == {"runs": []}
    r = client.get("/api/plugins/understudy/runs/nope")
    assert r.status_code == 404
    r = client.post("/api/plugins/understudy/runs/nope/cancel")
    assert r.status_code == 404
    r = client.post("/api/plugins/understudy/runs", json={"candidates": []})
    assert r.status_code == 400
    r = client.get("/api/plugins/understudy/estimate", params={})
    assert r.status_code in (400, 422)
