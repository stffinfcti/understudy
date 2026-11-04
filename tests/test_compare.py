"""Deterministic action comparison."""

from understudy.engine.compare import (
    DIFFERENT, EXACT, SAME_TOOLS, compare, normalize_production,
)

from conftest import assistant, tool_call


def _prod(text=None, calls=None):
    return normalize_production(assistant(text, calls))


def test_reply_exact_with_whitespace_and_case():
    prod = _prod("The  answer\nis   42.")
    cand = ("reply", "the answer is 42")
    assert compare(prod, cand) == EXACT


def test_reply_different():
    assert compare(_prod("yes"), ("reply", "completely different text")) == DIFFERENT


def test_reply_near_miss_is_exact():
    prod = _prod("Here is the summary of the file you asked for.")
    cand = ("reply", "Here is the summary of the file you asked for!")
    assert compare(prod, cand) == EXACT


def test_tool_calls_exact_reordered_and_args():
    prod = _prod(calls=[tool_call("terminal", '{"command": "ls -la /tmp"}')])
    cand = ("tool_calls", [("terminal", {"command": "ls   -la    /tmp"})])
    assert compare(prod, cand) == EXACT  # shlex-equal commands


def test_tool_calls_dropped_keys_still_exact():
    prod = _prod(calls=[tool_call("write_file", '{"path": "a", "content": "x"}')])
    cand = ("tool_calls", [("write_file", {"path": "a", "content": "x",
                                          "description": "write the file",
                                          "reasoning": "because"})])
    assert compare(prod, cand) == EXACT


def test_tool_calls_same_tools_different_args():
    prod = _prod(calls=[tool_call("terminal", '{"command": "ls"}')])
    cand = ("tool_calls", [("terminal", {"command": "pwd"})])
    assert compare(prod, cand) == SAME_TOOLS


def test_tool_calls_different_names():
    prod = _prod(calls=[tool_call("terminal", '{}')])
    cand = ("tool_calls", [("web_search", {"q": "x"})])
    assert compare(prod, cand) == DIFFERENT


def test_multicall_multiset():
    prod = _prod(calls=[tool_call("read_file", '{"path": "a"}', "c1"),
                        tool_call("read_file", '{"path": "b"}', "c2")])
    cand = ("tool_calls", [("read_file", {"path": "b"}), ("read_file", {"path": "a"})])
    assert compare(prod, cand) == EXACT


def test_kind_mismatch():
    assert compare(_prod("hello"), ("tool_calls", [("terminal", {})])) == DIFFERENT
    assert compare(_prod(calls=[tool_call("t", '{}')]), ("reply", "hi")) == DIFFERENT
    assert compare(None, ("reply", "hi")) == DIFFERENT


def test_terminal_non_shlexable_falls_back():
    prod = _prod(calls=[tool_call("terminal", '{"command": "a >"}')])
    cand = ("tool_calls", [("terminal", {"command": "a >"})])
    assert compare(prod, cand) == EXACT


def test_whitespace_normalized_arg_values():
    prod = _prod(calls=[tool_call("write_file", '{"path": "a", "content": "x  y"}')])
    cand = ("tool_calls", [("write_file", {"path": "a", "content": "x y"})])
    assert compare(prod, cand) == EXACT
