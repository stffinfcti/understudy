"""Aggregation: takeover rate with Wilson CI, cost projection, recommendation."""

from __future__ import annotations

import math
from typing import Any, Optional

from .judge import (
    AGREE, ALTERNATIVE, BETTER, CRITICAL_MISS, ERROR, EXACT, INVALID,
    MINOR_MISS, SKIPPED, UNJUDGED,
)

SCORED_VERDICTS = frozenset({EXACT, AGREE, ALTERNATIVE, BETTER, MINOR_MISS, CRITICAL_MISS, INVALID})
TAKEOVER_VERDICTS = frozenset({EXACT, AGREE, ALTERNATIVE, BETTER})
REPORTED_UNSCORED = (ERROR, UNJUDGED, SKIPPED)

WILSON_Z = 1.96
MIN_SCORED = 8
MIN_BASELINE_SCORED = 4
READY_TAKEOVER = 0.85
READY_MAX_CRITICAL = 0.08
PARTIAL_CATEGORY_N = 4
PARTIAL_TAKEOVER = 0.90
UNRELIABLE_ERROR_SHARE = 0.20


def wilson_ci(successes: int, n: int, z: float = WILSON_Z) -> Optional[tuple[float, float]]:
    """95% Wilson score interval on a binomial proportion."""
    if n <= 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _rate(part: int, n: int) -> Optional[float]:
    return (part / n) if n else None


def summarize_items(items: list[dict]) -> dict:
    """Counts, rates and CI for a list of per-decision replay records.

    Each item: ``{verdict, category, cost_usd, judge_cost_usd, latency_ms, attempted}``.
    """
    counts: dict[str, int] = {}
    by_category: dict[str, dict[str, int]] = {}
    for item in items:
        verdict = item.get("verdict")
        counts[verdict] = counts.get(verdict, 0) + 1
        if verdict in SCORED_VERDICTS:
            cat = item.get("category") or "other"
            slot = by_category.setdefault(cat, {})
            slot["n"] = slot.get("n", 0) + 1
            if verdict in TAKEOVER_VERDICTS:
                slot["takeover"] = slot.get("takeover", 0) + 1
            if verdict in (CRITICAL_MISS, INVALID):
                slot["critical"] = slot.get("critical", 0) + 1
    scored = sum(counts.get(v, 0) for v in SCORED_VERDICTS)
    takeover = sum(counts.get(v, 0) for v in TAKEOVER_VERDICTS)
    critical = counts.get(CRITICAL_MISS, 0) + counts.get(INVALID, 0)
    ci = wilson_ci(takeover, scored)
    attempted = sum(1 for item in items if item.get("attempted", True))
    costs = [float(item["cost_usd"]) for item in items
             if isinstance(item.get("cost_usd"), (int, float))]
    latencies = sorted(float(item["latency_ms"]) for item in items
                       if isinstance(item.get("latency_ms"), (int, float)))
    categories = {
        cat: {
            "n": slot["n"],
            "takeover_rate": _rate(slot.get("takeover", 0), slot["n"]),
            "critical_rate": _rate(slot.get("critical", 0), slot["n"]),
        }
        for cat, slot in sorted(by_category.items())
    }
    return {
        "counts": counts,
        "scored": scored,
        "attempted": attempted,
        "skipped": counts.get(SKIPPED, 0),
        "takeover_rate": _rate(takeover, scored),
        "ci95": [ci[0], ci[1]] if ci else None,
        "critical_rate": _rate(critical, scored),
        "minor_rate": _rate(counts.get(MINOR_MISS, 0), scored),
        "by_category": categories,
        "cost_usd": sum(costs) if costs else None,
        "cost_per_decision": (sum(costs) / len(costs)) if costs else None,
        "judge_cost_usd": sum(float(item["judge_cost_usd"]) for item in items
                              if isinstance(item.get("judge_cost_usd"), (int, float))),
        "latency_ms_p50": latencies[len(latencies) // 2] if latencies else None,
        "unreliable": attempted > 0 and (counts.get(ERROR, 0) / attempted) > UNRELIABLE_ERROR_SHARE,
    }


def recommend(scored: int, takeover_rate: Optional[float], critical_rate: Optional[float],
              by_category: dict, baseline: Optional[dict]) -> dict:
    """Map a candidate's summary to a recommendation level."""
    if scored < MIN_SCORED:
        return {"level": "insufficient",
                "text": f"only {scored} scored decisions — rerun with a larger --sample or more days.",
                "categories": []}
    takeover = takeover_rate or 0.0
    critical = critical_rate if critical_rate is not None else 1.0
    baseline_scored = int((baseline or {}).get("scored") or 0)
    if baseline_scored >= MIN_BASELINE_SCORED:
        base_takeover = (baseline or {}).get("takeover_rate") or 0.0
        base_critical = (baseline or {}).get("critical_rate") or 0.0
        if (takeover >= min(READY_TAKEOVER, base_takeover - 0.05)
                and critical <= max(READY_MAX_CRITICAL, base_critical + 0.04)):
            return {"level": "ready",
                    "text": "took over about as often as your current model matched itself, "
                            "with critical misses inside tolerance.",
                    "categories": []}
    elif takeover >= READY_TAKEOVER and critical <= READY_MAX_CRITICAL:
        return {"level": "ready",
                "text": "could have taken over most sampled decisions with few critical misses.",
                "categories": []}
    strong = [cat for cat, slot in (by_category or {}).items()
              if slot.get("n", 0) >= PARTIAL_CATEGORY_N
              and (slot.get("takeover_rate") or 0.0) >= PARTIAL_TAKEOVER
              and (slot.get("critical_rate") or 0.0) == 0.0]
    if strong:
        return {"level": "partial",
                "text": f"looks safe for: {', '.join(strong)}. Keep your current model for the rest.",
                "categories": strong}
    return {"level": "not_ready",
            "text": "fell short of the current model too often — not ready to take over.",
            "categories": []}


def project_cost(candidate_cost_per_decision: Optional[float],
                 baseline_cost_per_decision: Optional[float],
                 spend_30d_usd: Optional[float]) -> dict:
    """Legacy ratio-of-means projection; superseded by engine.pricing but kept
    for callers that only have spend + per-decision costs."""
    ratio = None
    if (isinstance(candidate_cost_per_decision, (int, float))
            and isinstance(baseline_cost_per_decision, (int, float))
            and baseline_cost_per_decision > 0):
        ratio = candidate_cost_per_decision / baseline_cost_per_decision
    projected = spend_30d_usd * ratio if (ratio is not None and spend_30d_usd is not None) else None
    savings = spend_30d_usd - projected if projected is not None else None
    return {"cost_ratio": ratio, "projected_monthly_usd": projected, "savings_monthly_usd": savings}
