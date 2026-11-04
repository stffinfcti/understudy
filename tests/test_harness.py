"""Transcript rendering, redaction, replay parsing, step rendering."""

from understudy.engine import harness
from understudy.engine.harness import (
    available_tools_block, build_replay_input, parse_next_action, redact,
    render_step, render_transcript,
)

from conftest import FakeResult, assistant, tool, tool_call, user


def test_transcript_blocks():
    conv = [user("hi"), assistant("hello"),
            assistant(None, [tool_call("terminal", '{"command": "ls"}')]),
            tool("terminal", "files")]
    out = render_transcript(conv)
    assert "[user] hi" in out
    assert "[assistant] hello" in out
    assert "[assistant -> tool calls]" in out
    assert 'terminal {"command": "ls"}' in out
    assert "[tool result: terminal] files" in out


def test_image_block_omitted():
    conv = [user([{"type": "text", "text": "look"},
                  {"type": "image_url", "image_url": {"url": "x"}}]),
            assistant("ok")]
    out = render_transcript(conv)
    assert "[image omitted]" in out and "look" in out


def test_tool_result_cap_keeps_head_and_tail():
    big = "A" * 1000 + "B" * 1000
    out = render_transcript([user("q"), tool("t", big)])
    assert "chars cut" in out
    assert "A" * 10 in out and "B" * 10 in out
    line = [l for l in out.split("\n") if "tool result" in l][0]
    assert len(line) < 1900


def test_transcript_cap_keeps_newest_and_prepends_first_user():
    conv = [user("FIRST " + "x" * 100)]
    for i in range(30):
        conv += [user("u" * 1500), assistant("a" * 1500)]
    out = render_transcript(conv)
    assert len(out) <= 42000
    assert out.startswith("[user, earlier in this conversation] FIRST")
    assert conv[-1]["content"][:50] in out  # newest kept


def test_redaction_each_kind():
    cases = [
        ("sk-" + "a1" * 15, "[redacted:api_key]"),
        ("ghp_" + "b2" * 20, "[redacted:github_token]"),
        ("github_pat_" + "c3" * 15, "[redacted:github_token]"),
        ("xoxb-" + "d4" * 10, "[redacted:slack_token]"),
        ("AKIA" + "E5" * 8, "[redacted:aws_key]"),
        ("ASIA" + "F6" * 8, "[redacted:aws_key]"),
        ("AIza" + "g7" * 12, "[redacted:google_key]"),
        ("sk_live_" + "h8" * 10, "[redacted:stripe_key]"),
        ("rk_live_" + "i9" * 10, "[redacted:stripe_key]"),
        ("pk_live_" + "j0" * 10, "[redacted:stripe_key]"),
        ("eyJ" + "a" * 10 + "." + "b" * 10 + "." + "c" * 10, "[redacted:jwt]"),
        ("password: hunter2secret", "password: [redacted:secret]"),
        ("api_key=\"k1" + "x" * 20 + "\"", "api_key=[redacted:secret]"),
    ]
    for raw, expected in cases:
        out = redact(f"token here {raw} end")
        assert expected in out, (raw, out)
        assert raw.split(":")[0] != raw or "[redacted" in out
    pem = ("-----BEGIN RSA PRIVATE KEY-----\nMIIabc\n"
           "-----END RSA PRIVATE KEY-----")
    assert "[redacted:private_key]" in redact(pem)
    # Plain text survives untouched.
    assert redact("hello world") == "hello world"


def test_parse_json_string_arguments():
    res = FakeResult(parsed={"action": "tool_calls", "tool_calls": [
        {"name": "terminal", "arguments": '{"command": "ls"}'}]})
    assert parse_next_action(res) == ("tool_calls", [("terminal", {"command": "ls"})])


def test_parse_object_arguments():
    res = FakeResult(parsed={"action": "tool_calls", "tool_calls": [
        {"name": "read_file", "arguments": {"path": "/a"}}]})
    assert parse_next_action(res) == ("tool_calls", [("read_file", {"path": "/a"})])


def test_parse_reply():
    res = FakeResult(parsed={"action": "reply", "reply": "sure thing"})
    assert parse_next_action(res) == ("reply", "sure thing")


def test_parse_garbage_is_none():
    for bad in [
        FakeResult(parsed=None, text="no json here", content_type="text"),
        FakeResult(parsed={"action": "fly"}, content_type="json"),
        FakeResult(parsed={"action": "tool_calls", "tool_calls": [{"name": ""}]},
                   content_type="json"),
        FakeResult(parsed=None, text="", content_type="text"),
    ]:
        assert parse_next_action(bad) is None


def test_parse_falls_back_to_text_json():
    res = FakeResult(parsed=None, content_type="text",
                     text='prefix {"action": "reply", "reply": "hi"} suffix')
    assert parse_next_action(res) == ("reply", "hi")


def test_available_tools_block():
    out = available_tools_block(["terminal", "read_file"], describe=lambda n: "desc " + n)
    assert "- terminal: desc terminal" in out
    assert "- read_file: desc read_file" in out
    assert available_tools_block([], describe=lambda n: "") == "(none listed)"


def test_build_replay_input_sections():
    out = build_replay_input("PERSONA", "TOOLS", "TRANSCRIPT")
    assert "AGENT PROFILE (excerpt of the agent's instructions):\nPERSONA" in out
    assert "AVAILABLE TOOLS:\nTOOLS" in out
    assert "TRANSCRIPT:\nTRANSCRIPT" in out
    assert out.endswith("What is the agent's next step?")


def test_render_step():
    assert render_step(("reply", "hi")) == "REPLY TO USER:\nhi"
    out = render_step(("tool_calls", [("terminal", {"command": "ls"})]))
    assert out == 'TOOL CALLS:\n1. terminal({"command": "ls"})'
