"""Metrics: Wilson CI, rates, recommendation levels."""

import pytest

from understudy.engine import metrics
from understudy.engine.judge import (
    AGREE, CRITICAL_MISS, ERROR, EXACT, INVALID, MINOR_MISS, SKIPPED, UNJUDGED,
)
from understudy.engine.metrics import (
    project_cost, recommend, summarize_items, wilson_ci,
)


def _item(verdict, category="chat", cost=0.01, attempted=True):
    return {"verdict": verdict, "category": category, "cost_usd": cost,
            "latency_ms": 10, "attempted": attempted}


def test_wilson_ci_known_values():
    lo, hi = wilson_ci(20, 24)
    assert lo == pytest.approx(0.6415, abs=0.001)
    assert hi == pytest.approx(0.9333, abs=0.001)
    assert wilson_ci(0, 0) is None
    lo, hi = wilson_ci(5, 5)
    assert lo == pytest.approx(0.5657, abs=0.001)
    assert hi == pytest.approx(1.0, abs=1e-9)


def test_summarize_items_rates_and_exclusions():
    items = ([_item(EXACT)] * 6 + [_item(AGREE)] * 2 + [_item(MINOR_MISS)]
             + [_item(CRITICAL_MISS)] + [_item(INVALID)]
             + [_item(ERROR)] + [_item(UNJUDGED)] + [_item(SKIPPED, attempted=False)])
    s = summarize_items(items)
    assert s["scored"] == 11          # ERROR/UNJUDGED/SKIPPED excluded
    assert s["takeover_rate"] == pytest.approx(8 / 11)
    assert s["critical_rate"] == pytest.approx(2 / 11)
    assert s["minor_rate"] == pytest.approx(1 / 11)
    assert s["ci95"] is not None
    assert s["cost_per_decision"] == pytest.approx(0.01)
    assert s["skipped"] == 1


def test_unreliable_threshold():
    items = [_item(ERROR) for _ in range(3)] + [_item(EXACT) for _ in range(9)]
    assert summarize_items(items)["unreliable"] is True
    items = [_item(ERROR)] + [_item(EXACT) for _ in range(11)]
    assert summarize_items(items)["unreliable"] is False


def test_recommend_insufficient():
    rec = recommend(7, 0.9, 0.0, {}, None)
    assert rec["level"] == "insufficient"


def test_recommend_ready_without_baseline_edges():
    assert recommend(10, 0.85, 0.08, {}, {"scored": 0})["level"] == "ready"
    assert recommend(10, 0.849, 0.08, {}, {"scored": 0})["level"] != "ready"
    assert recommend(10, 0.9, 0.09, {}, {"scored": 0})["level"] != "ready"


def test_recommend_ready_vs_baseline():
    base = {"scored": 6, "takeover_rate": 0.80, "critical_rate": 0.0}
    assert recommend(10, 0.75, 0.08, {}, base)["level"] == "ready"   # 0.75 >= 0.80-0.05
    assert recommend(10, 0.749, 0.08, {}, base)["level"] != "ready"
    # baseline critical 0.05 -> ceiling max(0.08, 0.09) = 0.09
    base2 = {"scored": 6, "takeover_rate": 0.5, "critical_rate": 0.05}
    assert recommend(10, 0.9, 0.09, {}, base2)["level"] == "ready"
    assert recommend(10, 0.9, 0.10, {}, base2)["level"] == "not_ready"


def test_recommend_partial_categories():
    cats = {"coding": {"n": 5, "takeover_rate": 0.95, "critical_rate": 0.0},
            "chat": {"n": 6, "takeover_rate": 0.5, "critical_rate": 0.1}}
    rec = recommend(11, 0.7, 0.1, cats, {"scored": 0})
    assert rec["level"] == "partial"
    assert rec["categories"] == ["coding"]


def test_project_cost_nulls_and_ratio():
    out = project_cost(0.002, 0.01, 100.0)
    assert out["cost_ratio"] == pytest.approx(0.2)
    assert out["projected_monthly_usd"] == pytest.approx(20.0)
    assert out["savings_monthly_usd"] == pytest.approx(80.0)
    for bad in [(None, 0.01, 100.0), (0.002, None, 100.0),
                (0.002, 0.0, 100.0), (0.002, 0.01, None)]:
        out = project_cost(*bad)
        assert out["cost_ratio"] is None or out["projected_monthly_usd"] is None
    assert project_cost(0.002, None, 100.0)["projected_monthly_usd"] is None
