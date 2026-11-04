"""Deterministic comparison of a production action and a candidate action.

Actions are normalized 2-tuples: ``("reply", text)`` or
``("tool_calls", [(name, args_dict), ...])``. Only an EXACT comparison skips
the judge; SAME_TOOLS and DIFFERENT both go to it.
"""

from __future__ import annotations

import difflib
import json
import shlex
from typing import Any, Optional

EXACT = "EXACT"
SAME_TOOLS = "SAME_TOOLS"
DIFFERENT = "DIFFERENT"

# Keys models fill with narration rather than intent; equal everywhere that matters.
_DROP_ARG_KEYS = frozenset({"timeout", "description", "reasoning", "explanation"})
_REPLY_EQUIVALENT_RATIO = 0.92


def _norm_text(text: Any) -> str:
    return " ".join(str(text or "").split())


def _norm_arg_value(value: Any) -> Any:
    if isinstance(value, str):
        return _norm_text(value)
    if isinstance(value, dict):
        return {k: _norm_arg_value(v) for k, v in value.items() if k not in _DROP_ARG_KEYS}
    if isinstance(value, list):
        return [_norm_arg_value(v) for v in value]
    return value


def _norm_args(args: Any) -> dict:
    if not isinstance(args, dict):
        return {}
    return {k: _norm_arg_value(v) for k, v in args.items() if k not in _DROP_ARG_KEYS}


def _call_parts(call: Any) -> tuple[str, dict]:
    """``(name, args_dict)`` from either the OpenAI ``function`` shape or a flat one."""
    if not isinstance(call, dict):
        return "", {}
    fn = call.get("function") if isinstance(call.get("function"), dict) else call
    name = str(fn.get("name") or call.get("name") or "")
    raw = fn.get("arguments", call.get("arguments"))
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            raw = {}
    return name, raw if isinstance(raw, dict) else {}


def normalize_tool_calls(calls: Any) -> list[tuple[str, dict]]:
    out = [_call_parts(c) for c in calls] if isinstance(calls, list) else []
    return [pair for pair in out if pair[0]]


def message_text(content: Any) -> str:
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


def normalize_production(message: dict) -> Optional[tuple]:
    """The stored assistant message -> its normalized action, or None when it took none."""
    calls = normalize_tool_calls(message.get("tool_calls"))
    if calls:
        return ("tool_calls", calls)
    text = message_text(message.get("content")).strip()
    if text:
        return ("reply", text)
    return None


def normalize_candidate(parsed: Any) -> Optional[tuple]:
    """A parsed NEXT_ACTION object -> its normalized action, or None (INVALID)."""
    if not isinstance(parsed, dict):
        return None
    action = parsed.get("action")
    if action == "reply":
        reply = parsed.get("reply")
        return ("reply", reply) if isinstance(reply, str) else None
    if action == "tool_calls":
        calls = normalize_tool_calls(parsed.get("tool_calls"))
        return ("tool_calls", calls) if calls else None
    return None


def _command_tokens(command: Any) -> Optional[list[str]]:
    if not isinstance(command, str):
        return None
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def _args_equal(name: str, left: dict, right: dict) -> bool:
    left, right = _norm_args(left), _norm_args(right)
    if name == "terminal":
        left_cmd, right_cmd = left.pop("command", None), right.pop("command", None)
        left_tok, right_tok = _command_tokens(left_cmd), _command_tokens(right_cmd)
        if left_tok is None or right_tok is None:
            if left_cmd != right_cmd:
                return False
        elif left_tok != right_tok:
            return False
    return left == right


def compare(production: Optional[tuple], candidate: Optional[tuple]) -> str:
    """EXACT only when same kind and materially identical; else SAME_TOOLS/DIFFERENT."""
    if production is None or candidate is None or production[0] != candidate[0]:
        return DIFFERENT
    if production[0] == "reply":
        a = _norm_text(production[1]).lower()
        b = _norm_text(candidate[1]).lower()
        return EXACT if a == b or difflib.SequenceMatcher(None, a, b).ratio() >= _REPLY_EQUIVALENT_RATIO else DIFFERENT
    prod_calls, cand_calls = production[1], candidate[1]
    if sorted(name for name, _ in prod_calls) != sorted(name for name, _ in cand_calls):
        return DIFFERENT
    prod_sorted = sorted(prod_calls, key=lambda c: (c[0], json.dumps(c[1], sort_keys=True, default=str)))
    cand_sorted = sorted(cand_calls, key=lambda c: (c[0], json.dumps(c[1], sort_keys=True, default=str)))
    for (name, p_args), (_, c_args) in zip(prod_sorted, cand_sorted):
        if not _args_equal(name, p_args, c_args):
            return SAME_TOOLS
    return EXACT
