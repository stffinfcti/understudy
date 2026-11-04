"""Corpus extraction: Hermes session history -> a list of replayable decision points.

A *decision point* is a moment where the user's agent committed to an action:
an assistant message that either made tool calls or replied to the user. The
engine works against an injected session source (``iter_sessions`` /
``get_session`` / ``get_conversation``), so it is unit-testable without Hermes;
:class:`HermesSessionSource` adapts ``SessionDB(read_only=True)``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Protocol

DAY_SECONDS = 86400.0
# When the caller cannot name the live session, anything touched this recently
# may be the session running the audition itself.
LIVE_SESSION_GRACE_SECONDS = 120.0
# Sources whose transcripts are other agents' work, not the user's agent.
NON_USER_AGENT_SOURCES = frozenset({"subagent", "kanban", "tool"})

CODING_TOOLS = frozenset({
    "terminal", "process_manage", "read_file", "write_file", "patch",
    "edit_file", "search_files", "execute_code",
})
RESEARCH_TOOLS = frozenset({"web_search", "web_extract", "x_search"})
ACTION_TOOLS = frozenset({
    "send_message", "cronjob_manage", "todo_list", "memory",
    "image_generate", "text_to_speech", "delegate_task",
})
REPLY_LOOKBACK_MESSAGES = 6
SPEND_WINDOW_DAYS = 30


@dataclass
class DecisionPoint:
    """One replayable moment: the transcript prefix plus the action the agent took."""

    id: str                      # "<session_id>:<index into the loaded conversation>"
    session_id: str
    source: str
    session_model: str
    started_at: float
    kind: str                    # "tool_calls" | "reply"
    production: tuple            # normalized action, see engine.compare
    prior_messages: list[dict]   # conversation messages before this one
    category: str
    session_tool_names: list[str]  # tool names used anywhere in the session
    pinned_tool_names: list[str] = field(default_factory=list)  # session's tools[] pin, if recorded
    system_prompt: str = ""
    last_user_text: str = ""


@dataclass
class CorpusResult:
    sessions_seen: int = 0
    sessions_used: int = 0
    decisions: list[DecisionPoint] = field(default_factory=list)
    by_category: dict[str, int] = field(default_factory=dict)
    window_days: int = 0
    spend_30d_usd: Optional[float] = None
    spend_known: bool = False
    # Token sums over sessions started in the last 30 days passing the filters —
    # the basis for monthly cost projection (engine/pricing.py).
    usage_30d: dict = field(default_factory=dict)
    sessions_30d: list = field(default_factory=list)  # per-session {model, provider, tokens...}


class SessionSource(Protocol):
    """What the engine needs from session storage (SessionDB-shaped)."""

    def iter_sessions(self, since: float) -> Iterable[dict]:
        """Session dicts (newest activity first) that may start after ``since`` (unix ts).

        Each dict needs: ``id``, ``source``, ``model``, ``started_at``,
        ``model_config``, ``estimated_cost_usd``, ``actual_cost_usd``.
        """

    def get_session(self, session_id: str) -> Optional[dict]:
        """Full session dict incl. ``system_prompt`` and pinned ``tool_names``."""

    def get_conversation(self, session_id: str) -> list[dict]:
        """OpenAI-format message dicts (role/content/tool_calls/tool_name)."""


def _content_has_image(content: Any) -> bool:
    if not isinstance(content, list):
        return False
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = str(block.get("type") or "")
        if "image" in btype or "image_url" in block or "image" in block:
            return True
    return False


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return "" if content is None else str(content)


def _call_name_and_args(call: Any) -> tuple[str, dict]:
    """Normalize one stored tool call to ``(name, args_dict)``."""
    if not isinstance(call, dict):
        return "", {}
    fn = call.get("function") if isinstance(call.get("function"), dict) else call
    name = str(fn.get("name") or call.get("name") or "")
    raw_args = fn.get("arguments", call.get("arguments"))
    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args)
        except (ValueError, TypeError):
            raw_args = {}
    return name, raw_args if isinstance(raw_args, dict) else {}


def _message_tool_calls(message: dict) -> list[tuple[str, dict]]:
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        return []
    out = [_call_name_and_args(c) for c in calls]
    return [pair for pair in out if pair[0]]


def _is_delegate_session(session: dict) -> bool:
    source = str(session.get("source") or "").strip().lower()
    if source in NON_USER_AGENT_SOURCES:
        return True
    raw = session.get("model_config")
    if isinstance(raw, str) and raw:
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            raw = None
    return isinstance(raw, dict) and bool(raw.get("_delegate_from"))


def categorize_calls(names: Iterable[str]) -> str:
    """Category one set of tool calls belongs to, first match wins."""
    names = [n for n in names]
    lowered = [n.lower() for n in names]
    if any(n in CODING_TOOLS or "file" in n or "terminal" in n or "code" in n for n in lowered):
        return "coding"
    if any(n in RESEARCH_TOOLS or n.startswith("browser") for n in lowered):
        return "research"
    if any(n in ACTION_TOOLS or n.startswith("kanban_") or n.startswith("mcp_") for n in lowered):
        return "actions"
    return "other"


def _session_cost(session: dict) -> Optional[float]:
    for key in ("actual_cost_usd", "estimated_cost_usd"):
        value = session.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    return None


def pinned_tool_names(session: dict) -> list[str]:
    """The session's tools[] pin (JSON name list), or [] when not recorded."""
    raw = (session or {}).get("tool_names")
    if isinstance(raw, str) and raw.strip():
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return []
    if isinstance(raw, (list, tuple)):
        return [str(n) for n in raw if n]
    return []


def decisions_from_session(session: dict, conversation: list[dict]) -> list[DecisionPoint]:
    """Extract every decision point from one session's loaded conversation."""
    session_id = str(session.get("id") or "")
    source = str(session.get("source") or "")
    session_model = str(session.get("model") or "")
    started_at = float(session.get("started_at") or 0.0)
    system_prompt = str(session.get("system_prompt") or "")
    pinned = pinned_tool_names(session)

    tool_names_seen: set[str] = set()
    for message in conversation:
        if message.get("role") == "assistant":
            tool_names_seen.update(name for name, _ in _message_tool_calls(message))

    points: list[DecisionPoint] = []
    last_user_index: Optional[int] = None
    last_user_has_image = False
    last_user_text = ""
    for index, message in enumerate(conversation):
        role = message.get("role")
        if role == "user":
            last_user_index = index
            content = message.get("content")
            last_user_has_image = _content_has_image(content)
            last_user_text = _text_of(content)
            continue
        if role != "assistant":
            continue
        calls = _message_tool_calls(message)
        text = _text_of(message.get("content")).strip()
        if calls:
            kind, production = "tool_calls", ("tool_calls", calls)
        elif text:
            kind, production = "reply", ("reply", text)
        else:
            continue
        if last_user_index is None or last_user_has_image:
            continue
        if kind == "tool_calls":
            category = categorize_calls(name for name, _ in calls)
        else:
            inherited = None
            for prev in conversation[max(0, index - REPLY_LOOKBACK_MESSAGES):index]:
                if prev.get("role") == "assistant":
                    prev_calls = _message_tool_calls(prev)
                    if prev_calls:
                        inherited = categorize_calls(name for name, _ in prev_calls)
            category = inherited if inherited is not None else "chat"
        points.append(DecisionPoint(
            id=f"{session_id}:{index}", session_id=session_id, source=source,
            session_model=session_model, started_at=started_at, kind=kind,
            production=production, prior_messages=conversation[:index],
            category=category, session_tool_names=sorted(tool_names_seen),
            pinned_tool_names=pinned,
            system_prompt=system_prompt, last_user_text=last_user_text,
        ))
    return points


def collect_corpus(
    session_source: SessionSource,
    *,
    days: int,
    exclude_sources: Iterable[str] = (),
    current_session_id: Optional[str] = None,
    now: Optional[float] = None,
) -> CorpusResult:
    """Build the decision corpus from sessions started inside the ``days`` window.

    ``current_session_id`` names the session running the audition; when it is
    None, every session with activity in the last :data:`LIVE_SESSION_GRACE_SECONDS`
    is excluded instead (any of them could be the live one).
    """
    now = time.time() if now is None else float(now)
    days = max(1, int(days))
    cutoff = now - days * DAY_SECONDS
    spend_cutoff = now - SPEND_WINDOW_DAYS * DAY_SECONDS
    since = min(cutoff, spend_cutoff)
    excluded_sources = {s.strip().lower() for s in exclude_sources if str(s).strip()}
    live_grace_cutoff = now - LIVE_SESSION_GRACE_SECONDS

    result = CorpusResult(window_days=days)
    spend = 0.0
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0}
    for session in session_source.iter_sessions(since):
        result.sessions_seen += 1
        session_id = str(session.get("id") or "")
        started_at = float(session.get("started_at") or 0.0)
        cost = _session_cost(session)
        source = str(session.get("source") or "").strip().lower()
        filtered = (source in excluded_sources or _is_delegate_session(session))
        if started_at >= spend_cutoff and not filtered:
            if cost is not None:
                spend += cost
                result.spend_known = True
            row = {"model": str(session.get("model") or ""),
                   "provider": str(session.get("billing_provider") or ""),
                   "started_at": started_at}
            for key in usage:
                value = session.get(key)
                row[key] = int(value) if isinstance(value, (int, float)) else 0
                usage[key] += row[key]
            result.sessions_30d.append(row)
        if started_at < cutoff:
            continue
        if filtered:
            continue
        if current_session_id:
            if session_id == current_session_id:
                continue
        else:
            last_active = session.get("last_active") or session.get("last_activity_at") or started_at
            try:
                last_active = float(last_active or 0.0)
            except (TypeError, ValueError):
                last_active = started_at
            if last_active >= live_grace_cutoff:
                continue
        full = session_source.get_session(session_id) or session
        conversation = session_source.get_conversation(session_id)
        if not conversation:
            continue
        points = decisions_from_session(full, conversation)
        if not points:
            continue
        result.sessions_used += 1
        result.decisions.extend(points)
    if result.spend_known:
        result.spend_30d_usd = spend
    result.usage_30d = usage
    for point in result.decisions:
        result.by_category[point.category] = result.by_category.get(point.category, 0) + 1
    return result


class HermesSessionSource:
    """``SessionSource`` over a read-only ``SessionDB`` on the active profile."""

    def __init__(self, db: Any = None, db_path: Any = None):
        if db is None:
            from hermes_state import SessionDB
            db = SessionDB(db_path=db_path, read_only=True)
        self._db = db

    def iter_sessions(self, since: float) -> Iterable[dict]:
        page, offset = 200, 0
        while True:
            rows = self._db.search_sessions(limit=page, offset=offset)
            if not rows:
                return
            for row in rows:
                yield row
            # Ordered by last_active DESC and started_at <= last_active, so once a
            # session's activity predates the window nothing later can qualify.
            last = rows[-1]
            last_active = last.get("last_active") or last.get("last_activity_at") or last.get("started_at") or 0.0
            if len(rows) < page or float(last_active or 0.0) < since:
                return
            offset += page

    def get_session(self, session_id: str) -> Optional[dict]:
        return self._db.get_session(session_id)

    def get_conversation(self, session_id: str) -> list[dict]:
        return self._db.get_messages_as_conversation(session_id)

    def close(self) -> None:
        self._db.close()
