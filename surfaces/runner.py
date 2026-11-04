"""Shared plumbing for the tool / slash / CLI surfaces: settings, session source,
progress messages and the trust-gate error text."""

from __future__ import annotations

import time
from typing import Any, Optional

from ..engine import audition

_SETTING_DEFAULTS = dict(audition.DEFAULT_SETTINGS)

TRUST_HELP = (
    "Understudy needs permission to call other models on your account. "
    "Grant it on the consent screen when enabling the plugin, or set "
    "plugins.entries.understudy.llm.allow_model_override: true (and "
    "allow_provider_override: true for provider:model candidates) in config.yaml."
)


def get_settings(ctx: Any, overrides: Optional[dict] = None) -> dict:
    """Plugin settings from ``plugins.entries.understudy.settings.*`` with manifest defaults."""
    settings = dict(_SETTING_DEFAULTS)
    for key in settings:
        try:
            value = ctx.get_config(key, settings[key])
        except Exception:
            value = settings[key]
        if value is not None:
            settings[key] = value
    for key, value in (overrides or {}).items():
        if key in settings and value is not None:
            settings[key] = value
    return settings


def current_model() -> dict:
    """The user's active route: ``model.default`` or ``model.model``, plus
    provider and base_url when set."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
        model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
        return {"provider": model_cfg.get("provider") or "",
                "model": model_cfg.get("default") or model_cfg.get("model") or "",
                "base_url": model_cfg.get("base_url") or ""}
    except Exception:
        return {}


def required_overrides(candidate_specs, judge_model: str = "") -> list[str]:
    """Capabilities the run needs beyond plain ctx.llm access."""
    from ..engine.audition import parse_candidate_spec
    needs = set()
    for raw in candidate_specs or []:
        spec = parse_candidate_spec(str(raw))
        if spec.get("model"):
            needs.add("llm.model_override")
        if spec.get("provider"):
            needs.add("llm.provider_override")
    if str(judge_model or "").strip():
        judge = parse_candidate_spec(str(judge_model))
        if judge.get("model"):
            needs.add("llm.model_override")
        if judge.get("provider"):
            needs.add("llm.provider_override")
    return sorted(needs)


def preflight_trust(ctx, candidate_specs, judge_model: str = "") -> str:
    """TRUST_HELP text when a required override is not granted, else ""."""
    missing = [cap for cap in required_overrides(candidate_specs, judge_model)
               if not ctx.has_capability(cap)]
    return TRUST_HELP if missing else ""


def session_source() -> Any:
    from ..engine.corpus import HermesSessionSource
    return HermesSessionSource()


def current_session_id() -> str:
    try:
        from gateway.session_context import get_session_env
        return get_session_env("HERMES_SESSION_ID") or ""
    except Exception:
        return ""


def is_trust_error(exc: BaseException) -> bool:
    return type(exc).__name__ == "PluginLlmTrustError"


class ProgressSink:
    """Live ``Understudy · auditioning X · n/m decisions · $a of $b`` line via
    ctx.post_message + .edit, throttled. Degrades to a no-op outside a session."""

    THROTTLE_S = 1.5

    def __init__(self, ctx: Any, label: str):
        self._ctx = ctx
        self._label = label
        self._handle = None
        self._last_edit = 0.0
        self._pending = ""
        try:
            self._handle = ctx.post_message(f"Understudy · {label} · starting")
        except Exception:
            self._handle = None

    def _line(self, event: dict) -> str:
        done, total = event.get("done", 0), event.get("total", 0)
        spent, budget = event.get("spent_usd"), event.get("budget_usd")
        text = f"Understudy · {done}/{total} replays"
        if isinstance(spent, (int, float)):
            text += f" · ${spent:.2f}"
            if isinstance(budget, (int, float)):
                text += f" of ${budget:.2f}"
        for label, tally in (event.get("tallies") or {}).items():
            name = "your current model" if label == "baseline" else label
            text += f"\n  {name}: {tally.get('ok', 0)}/{tally.get('scored', 0)} could take over"
        return text

    def __call__(self, event: dict) -> None:
        if self._handle is None:
            return
        kind = event.get("event")
        if kind == "item":
            line = self._line(event)
            self._pending = line
            now = time.monotonic()
            if now - self._last_edit >= self.THROTTLE_S:
                self._last_edit = now
                try:
                    self._handle.edit(line)
                except Exception:
                    pass
        elif kind == "finished" and self._pending:
            try:
                self._handle.edit(self._pending)
            except Exception:
                pass

    def post_final(self, text: str) -> None:
        """Replace the progress message with the finished summary."""
        if self._handle is None:
            try:
                self._ctx.post_message(text)
            except Exception:
                pass
            return
        try:
            self._handle.edit(text)
        except Exception:
            try:
                self._ctx.post_message(text)
            except Exception:
                pass
