"""Deterministic stratified sampling of the decision corpus (seed = run id)."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Optional

CATEGORY_FLOOR = 3
SESSION_CAP_DIVISOR = 8
SESSION_CAP_FLOOR = 2


def _allocate(available: dict[str, int], sample: int) -> dict[str, int]:
    """Per-category quota: proportional to availability, floored per present category."""
    present = {cat: n for cat, n in available.items() if n > 0}
    if not present or sample <= 0:
        return {}
    quota = {cat: min(CATEGORY_FLOOR, n) for cat, n in present.items()}
    budget = sample - sum(quota.values())
    if budget <= 0:
        # Sample smaller than the floors: keep floors for the largest categories.
        for cat in sorted(present, key=lambda c: present[c]):
            if budget == 0:
                break
            give = min(quota[cat], -budget)
            quota[cat] -= give
            budget += give
        return {c: q for c, q in quota.items() if q > 0}
    remaining = {cat: present[cat] - quota[cat] for cat in present}
    total_remaining = sum(remaining.values())
    if total_remaining <= 0:
        return quota
    # Largest-remainder proportional split of the leftover budget.
    shares = {cat: budget * remaining[cat] / total_remaining for cat in present}
    for cat in present:
        quota[cat] += min(int(shares[cat]), remaining[cat])
    left = budget - sum(quota[cat] - min(CATEGORY_FLOOR, present[cat]) for cat in present)
    for cat in sorted(present, key=lambda c: (shares[c] - int(shares[c]), present[c]), reverse=True):
        if left <= 0:
            break
        if present[cat] - quota[cat] > 0:
            quota[cat] += 1
            left -= 1
    return quota


def _interleave_kinds(decisions: list) -> list:
    """Order one session's decisions alternating reply/tool_calls where possible."""
    replies = [d for d in decisions if d.kind == "reply"]
    calls = [d for d in decisions if d.kind != "reply"]
    out = []
    while replies or calls:
        for pile in (calls, replies) if len(calls) >= len(replies) else (replies, calls):
            if pile:
                out.append(pile.pop(0))
    return out


def stratified_sample(decisions: list, sample: int, seed: str) -> list:
    """Pick up to ``sample`` decisions, stratified by category and spread over sessions."""
    sample = max(0, int(sample))
    if sample == 0 or not decisions:
        return []
    rng = random.Random(str(seed))
    by_category: dict[str, list] = defaultdict(list)
    for point in decisions:
        by_category[point.category].append(point)
    available = {cat: len(pts) for cat, pts in by_category.items()}
    quota = _allocate(available, sample)
    session_cap = max(SESSION_CAP_FLOOR, math.ceil(sample / SESSION_CAP_DIVISOR))

    per_session: dict[str, int] = defaultdict(int)
    picked: list = []
    picked_ids: set[str] = set()
    for cat, cat_quota in sorted(quota.items()):
        by_session: dict[str, list] = defaultdict(list)
        for point in by_category.get(cat, []):
            by_session[point.session_id].append(point)
        queues = [_interleave_kinds(pts) for pts in by_session.values()]
        for queue in queues:
            rng.shuffle(queue)
        sessions = [q for q in queues if q]
        rng.shuffle(sessions)
        got = 0
        while got < cat_quota and sessions:
            progressed = False
            for queue in list(sessions):
                if got >= cat_quota:
                    break
                if not queue:
                    sessions.remove(queue)
                    continue
                point = queue[0]
                if per_session[point.session_id] >= session_cap:
                    queue.pop(0)
                    if not queue:
                        sessions.remove(queue)
                    continue
                queue.pop(0)
                if not queue:
                    sessions.remove(queue)
                if point.id in picked_ids:
                    continue
                picked_ids.add(point.id)
                picked.append(point)
                per_session[point.session_id] += 1
                got += 1
                progressed = True
            if not progressed:
                break
    return picked
