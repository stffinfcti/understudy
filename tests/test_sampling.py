"""Deterministic stratified sampling."""

from understudy.engine.corpus import DecisionPoint
from understudy.engine.sampling import stratified_sample


def _pt(i, session="s1", category="chat", kind="reply"):
    return DecisionPoint(
        id=f"{session}:{i}", session_id=session, source="cli",
        session_model="m", started_at=0.0, kind=kind,
        production=("reply", "x"), prior_messages=[], category=category,
        session_tool_names=[])


def test_deterministic_same_seed():
    pts = [_pt(i, session=f"s{i % 8}", category="chat") for i in range(40)]
    a = stratified_sample(pts, 20, "run1")
    b = stratified_sample(pts, 20, "run1")
    assert [p.id for p in a] == [p.id for p in b]
    assert len(a) == 20


def test_different_seed_can_differ():
    pts = [_pt(i) for i in range(40)]
    a = stratified_sample(pts, 20, "aaa")
    b = stratified_sample(pts, 20, "bbb")
    assert [p.id for p in a] != [p.id for p in b]


def test_per_session_cap():
    pts = [_pt(i, session="one") for i in range(50)]
    out = stratified_sample(pts, 24, "s")
    # cap = max(2, ceil(24/8)) = 3
    assert len(out) == 3


def test_spreads_across_sessions():
    pts = [_pt(i, session=f"s{i % 8}") for i in range(64)]
    out = stratified_sample(pts, 24, "s")
    sessions = {p.session_id for p in out}
    assert len(sessions) >= 6  # 24 over 8 sessions with cap 3 -> uses many


def test_category_floor_respected():
    pts = ([_pt(i, category="coding") for i in range(30)]
           + [_pt(100 + i, category="chat") for i in range(2)])
    out = stratified_sample(pts, 24, "s")
    cats = {p.category for p in out}
    assert "chat" in cats and "coding" in cats
    assert sum(1 for p in out if p.category == "chat") == 2  # min(3, available)


def test_returns_fewer_when_fewer_exist():
    pts = [_pt(i, session=f"s{i % 3}") for i in range(5)]
    assert len(stratified_sample(pts, 24, "s")) == 5
    # One session only: the per-session cap (max(2, ceil(24/8))=3) binds.
    assert len(stratified_sample([_pt(i) for i in range(5)], 24, "s")) == 3


def test_empty():
    assert stratified_sample([], 10, "s") == []
    assert stratified_sample([_pt(1)], 0, "s") == []
