"""Decision-point extraction and corpus collection."""

import time

from understudy.engine import corpus
from understudy.engine.corpus import (
    categorize_calls, collect_corpus, decisions_from_session,
)

from conftest import assistant, session_dict, tool, tool_call, user

NOW = 1_800_100_000.0
DAY = 86400.0


def _conv(*msgs):
    return list(msgs)


def test_reply_and_tool_call_decisions_extracted():
    conv = _conv(
        user("hello"),
        assistant("hi there"),
        user("list files"),
        assistant(None, [tool_call("terminal", '{"command": "ls"}')]),
        tool("terminal", "a.txt"),
        assistant("done"),
    )
    pts = decisions_from_session(session_dict(), conv)
    assert [p.kind for p in pts] == ["reply", "tool_calls", "reply"]
    assert pts[0].id == "s1:1"
    assert pts[1].production == ("tool_calls", [("terminal", {"command": "ls"})])
    assert pts[0].production == ("reply", "hi there")


def test_decision_needs_an_earlier_user_message():
    conv = _conv(assistant("unprompted"), assistant("still no user"))
    assert decisions_from_session(session_dict(), conv) == []


def test_image_in_last_user_message_skips_decision():
    conv = _conv(
        user([{"type": "text", "text": "what is this"},
              {"type": "image_url", "image_url": {"url": "data:..."}}]),
        assistant("a cat"),
        user("thanks"),
        assistant("welcome"),
    )
    pts = decisions_from_session(session_dict(), conv)
    assert [p.id for p in pts] == ["s1:3"]


def test_subagent_and_delegate_sessions_excluded():
    conv = _conv(user("hi"), assistant("yo"))
    sessions = [
        session_dict("a", source="cli"),
        session_dict("b", source="subagent"),
        session_dict("c", source="kanban"),
        session_dict("d", source="cli", model_config='{"_delegate_from": "x"}'),
    ]
    src = _src(sessions, {s["id"]: conv for s in sessions})
    result = collect_corpus(src, days=14, now=NOW)
    assert {p.session_id for p in result.decisions} == {"a"}


def test_exclude_sources_and_window():
    old = NOW - 30 * DAY
    conv = _conv(user("hi"), assistant("yo"))
    sessions = [
        session_dict("a", source="cli", started_at=NOW - DAY),
        session_dict("b", source="telegram", started_at=NOW - DAY),
        session_dict("c", source="cli", started_at=old),
    ]
    src = _src(sessions, {s["id"]: conv for s in sessions})
    result = collect_corpus(src, days=14, exclude_sources=["telegram"], now=NOW)
    assert {p.session_id for p in result.decisions} == {"a"}
    assert result.by_category == {"chat": 1}


def test_current_session_excluded_else_recent_activity():
    conv = _conv(user("hi"), assistant("yo"))
    recent = NOW - 30
    stale = NOW - 3600
    sessions = [
        session_dict("live", started_at=NOW - DAY, last_activity_at=recent),
        session_dict("old", started_at=NOW - DAY, last_activity_at=stale),
    ]
    src = _src(sessions, {s["id"]: conv for s in sessions})
    # Named current session: only it is excluded, despite recent activity.
    result = collect_corpus(src, days=14, current_session_id="live", now=NOW)
    assert {p.session_id for p in result.decisions} == {"old"}
    # No id: the last-2-minutes rule excludes the active one.
    result = collect_corpus(src, days=14, now=NOW)
    assert {p.session_id for p in result.decisions} == {"old"}


def test_spend_30d_uses_best_known_cost():
    sessions = [
        session_dict("a", actual_cost_usd=2.0, estimated_cost_usd=5.0),
        session_dict("b", actual_cost_usd=None, estimated_cost_usd=1.0),
        session_dict("c", started_at=NOW - 40 * DAY, actual_cost_usd=9.0),
        session_dict("d"),
    ]
    result = collect_corpus(_src(sessions, {}), days=14, now=NOW)
    assert result.spend_known is True
    assert result.spend_30d_usd == 3.0
    empty = collect_corpus(_src([session_dict("x")], {}), days=14, now=NOW)
    assert empty.spend_known is False
    assert empty.spend_30d_usd is None


def test_categories():
    assert categorize_calls(["terminal"]) == "coding"
    assert categorize_calls(["write_file"]) == "coding"
    assert categorize_calls(["somefile_thing"]) == "coding"
    assert categorize_calls(["web_search"]) == "research"
    assert categorize_calls(["browser_open"]) == "research"
    assert categorize_calls(["todo_list"]) == "actions"
    assert categorize_calls(["kanban_move"]) == "actions"
    assert categorize_calls(["mcp_github__x"]) == "actions"
    assert categorize_calls(["clarify"]) == "other"


def test_reply_inherits_recent_tool_category_else_chat():
    conv = _conv(
        user("fix it"),
        assistant(None, [tool_call("write_file", '{"path": "a"}')]),
        tool("write_file", "ok"),
        assistant("fixed"),
        user("and?"),
        assistant("all done"),
    )
    pts = decisions_from_session(session_dict(), conv)
    assert pts[0].category == "coding"
    assert pts[1].category == "coding"   # inherits the calls 3 messages back
    # A reply with no tool calls in the previous 6 messages is "chat".
    conv2 = _conv(user("hi"), assistant("yo"), user("more"), assistant("sure"))
    assert [p.category for p in decisions_from_session(session_dict(), conv2)] == ["chat", "chat"]


def _src(sessions, convs):
    class _S:
        def iter_sessions(self, since):
            return iter(sessions)

        def get_session(self, sid):
            return next((s for s in sessions if s["id"] == sid), None)

        def get_conversation(self, sid):
            return convs.get(sid, [])
    return _S()
