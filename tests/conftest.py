import sys
from pathlib import Path

# The plugin directory is the package; its parent must be importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest


class FakeUsage:
    def __init__(self, cost_usd=None, input_tokens=100, output_tokens=50):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = input_tokens + output_tokens
        self.cost_usd = cost_usd


class FakeResult:
    def __init__(self, parsed=None, text="", content_type="json", cost_usd=None,
                 provider="fake", model="fake-model"):
        self.parsed = parsed
        self.text = text
        self.content_type = content_type
        self.usage = FakeUsage(cost_usd)
        self.provider = provider
        self.model = model


class FakeLLM:
    """Scripted llm double: routes calls by ``purpose``/``schema_name``."""

    def __init__(self):
        self.calls = []
        self.replay_fn = None      # (kwargs, call_index) -> FakeResult or Exception
        self.judge_fn = None
        self.default_cost = 0.001
        self.default_judge = {"verdict": "equivalent", "severity": "none",
                              "reason": "same step."}

    def complete_structured(self, **kwargs):
        self.calls.append(kwargs)
        purpose = kwargs.get("purpose", "")
        try:
            if "judge" in purpose:
                out = self.judge_fn(kwargs, len(self.calls)) if self.judge_fn else None
                if out is None:
                    out = FakeResult(parsed=dict(self.default_judge),
                                     cost_usd=self.default_cost)
            else:
                out = self.replay_fn(kwargs, len(self.calls)) if self.replay_fn else None
                if out is None:
                    out = FakeResult(parsed={"action": "reply", "reply": "mirror"},
                                     cost_usd=self.default_cost)
        except Exception:
            raise
        if isinstance(out, Exception):
            raise out
        return out


def msg(role, content=None, **kw):
    m = {"role": role, "content": content}
    m.update(kw)
    return m


def user(text):
    return msg("user", text)


def assistant(text=None, tool_calls=None):
    return msg("assistant", text, tool_calls=tool_calls or [])


def tool_call(name, arguments, call_id="c1"):
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": arguments}}


def tool(name, output):
    return msg("tool", output, tool_name=name, tool_call_id="c1")


class StubSource:
    """In-memory SessionSource for unit tests."""

    def __init__(self, sessions=(), conversations=None):
        self.sessions = list(sessions)
        # No copy: tests mutate convs between StubSource() and run().
        self.conversations = conversations if conversations is not None else {}

    def iter_sessions(self, since):
        return iter(sorted(self.sessions,
                           key=lambda s: s.get("started_at") or 0, reverse=True))

    def get_session(self, session_id):
        for s in self.sessions:
            if s.get("id") == session_id:
                return s
        return None

    def get_conversation(self, session_id):
        return self.conversations.get(session_id, [])

    def close(self):
        pass


def session_dict(session_id="s1", source="cli", model="anthropic/claude-x",
                 started_at=None, **kw):
    base = {"id": session_id, "source": source, "model": model,
            "started_at": started_at if started_at is not None else 1_800_000_000.0,
            "model_config": None, "system_prompt": "You are Hermes.",
            "estimated_cost_usd": None, "actual_cost_usd": None}
    base.update(kw)
    return base


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def stub_source():
    return StubSource
