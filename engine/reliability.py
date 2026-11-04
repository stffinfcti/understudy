"""Transient-error classification, retry/backoff, and per-spec rate limiting.

Provider SDK exceptions vary wildly; classification inspects status attributes
first, then message text. Trust errors are handled by the caller, never here.
"""

from __future__ import annotations

import re
from typing import Any, Optional

TRANSIENT_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})
_TRANSIENT_TEXT = re.compile(
    r"rate.?limit|too many requests|overload|temporar|timeout|timed.?out|"
    r"connection (reset|refused|aborted)|service unavailable|gateway|"
    r"retry later|try again|capacity|throttl", re.I)
_INVALID_PARAMS = re.compile(
    r"invalid|bad request|unsupported|not allowed|400", re.I)
_TIMEOUT_TEXT = re.compile(r"timeout|timed.?out|deadline", re.I)
_RATE_LIMIT_TEXT = re.compile(r"rate.?limit|too many requests|429|throttl", re.I)


def _status(exc: BaseException) -> Optional[int]:
    for attr in ("status_code", "status", "code", "http_status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and 100 <= value <= 599:
            return value
    response = getattr(exc, "response", None)
    if response is not None:
        value = getattr(response, "status_code", None)
        if isinstance(value, int):
            return value
    return None


def retry_after(exc: BaseException) -> Optional[float]:
    """Seconds the provider asked us to wait, capped at 20."""
    for attr in ("retry_after", "retry_after_seconds"):
        value = getattr(exc, attr, None)
        if isinstance(value, (int, float)) and value > 0:
            return min(float(value), 20.0)
    headers = getattr(exc, "headers", None) or getattr(
        getattr(exc, "response", None), "headers", None)
    if isinstance(headers, dict):
        for key, value in headers.items():
            if str(key).lower() == "retry-after":
                try:
                    return min(float(value), 20.0)
                except (TypeError, ValueError):
                    return None
    return None


def is_transient(exc: BaseException) -> bool:
    """429s, 5xx, timeouts and connection failures — worth retrying/backing off."""
    status = _status(exc)
    if status is not None:
        if status in TRANSIENT_STATUS:
            return True
        if 400 <= status < 500:
            return False
        return status >= 500
    return bool(_TRANSIENT_TEXT.search(str(exc)))


def error_kind(exc: BaseException) -> str:
    """rate_limited | invalid_params | timeout | other — for errors_by_kind."""
    status = _status(exc)
    if status == 429 or _RATE_LIMIT_TEXT.search(str(exc)):
        return "rate_limited"
    if status in (408,) or _TIMEOUT_TEXT.search(str(exc)):
        return "timeout"
    if status is not None and 400 <= status < 500:
        return "invalid_params"
    if _INVALID_PARAMS.search(str(exc)):
        return "invalid_params"
    return "other"
