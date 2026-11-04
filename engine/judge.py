"""The judge: blind pairwise review of the production step vs the candidate step."""

from __future__ import annotations

import json
import random
from typing import Any, Optional

from . import harness
from .harness import _render_content, redact, render_step

# Verdicts recorded on each replayed decision.
EXACT = "EXACT"                    # deterministic match, judge skipped
AGREE = "AGREE"                    # judge: equivalent
ALTERNATIVE = "ALTERNATIVE"        # judge: both fine, different path
BETTER = "BETTER"                  # judge: candidate's step was better
MINOR_MISS = "MINOR_MISS"          # judge: candidate worse, severity minor/none
CRITICAL_MISS = "CRITICAL_MISS"    # judge: candidate worse, severity major
INVALID = "INVALID"                # candidate produced no usable action
ERROR = "ERROR"                    # candidate call failed after retry
UNJUDGED = "UNJUDGED"              # judge call failed
SKIPPED = "SKIPPED"                # never attempted (budget or cancel)

EXCERPT_CAP = 12000

JUDGE_SYSTEM = """You review whether a cheaper "understudy" model could have stood in for a user's current AI agent. You will see one moment from a real conversation and two possible next steps for the agent at that moment: Step 1 and Step 2. One is what the current agent actually did; the other is the understudy's. You are not told which is which, and you must not guess.

Judge each step on substance: does it serve what the user actually wanted, is it correct and safe given the transcript, and would it move the task forward as well? Ignore wording, formatting, length and politeness unless they change the outcome. Two different steps can both be fine — for example reading a file before editing it versus editing it directly when the content is already known.

Severity of the worse step:
- "major": it would mislead the user, take a wrong, unsafe or destructive action, ignore the request, or derail the task.
- "minor": it is slower, less thorough or clumsier, but the task would still end up fine.
- "none": neither step is worse."""

JUDGE_INPUT_TEMPLATE = """CONVERSATION SO FAR (most recent part):
{excerpt}

STEP 1:
{step1}

STEP 2:
{step2}

Compare Step 1 and Step 2."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["equivalent", "both_fine", "step1_better", "step2_better"],
        },
        "severity": {"type": "string", "enum": ["none", "minor", "major"]},
        "reason": {
            "type": "string",
            "description": "One sentence, at most 200 characters.",
        },
    },
    "required": ["verdict", "severity", "reason"],
}

_REASON_CAP = 200


def build_excerpt(transcript: str, first_user_line: Optional[str] = None) -> str:
    """Last 12,000 chars of the transcript; the first user line re-attached when cut."""
    excerpt = str(transcript or "")[-EXCERPT_CAP:]
    if len(str(transcript or "")) > EXCERPT_CAP and first_user_line:
        return f"{first_user_line}\n{excerpt}"
    return excerpt


def first_user_line(messages: list[dict]) -> Optional[str]:
    for message in messages or []:
        if message.get("role") == "user":
            return f"[user] {_render_content(message.get('content'))}"
    return None


def _judge_parsed(result: Any) -> Optional[dict]:
    parsed = getattr(result, "parsed", None)
    if isinstance(parsed, dict) and getattr(result, "content_type", "") == "json":
        return parsed
    text = getattr(result, "text", "") or ""
    start = text.find("{")
    if start < 0:
        return None
    try:
        parsed, _end = json.JSONDecoder().raw_decode(text, start)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def map_verdict(judged: Optional[dict], candidate_is_step1: bool) -> tuple[str, str]:
    """Judge JSON -> (verdict, reason) from the candidate's perspective."""
    if not isinstance(judged, dict):
        return UNJUDGED, ""
    verdict = judged.get("verdict")
    severity = judged.get("severity") or "none"
    reason = str(judged.get("reason") or "")[:_REASON_CAP]
    if verdict == "equivalent":
        return AGREE, reason
    if verdict == "both_fine":
        return ALTERNATIVE, reason
    better = "step1_better" if candidate_is_step1 else "step2_better"
    worse = "step2_better" if candidate_is_step1 else "step1_better"
    if verdict == better:
        return BETTER, reason
    if verdict == worse:
        return (CRITICAL_MISS if severity == "major" else MINOR_MISS), reason
    return UNJUDGED, reason


def call_judge(llm: Any, *, transcript: str, first_user: Optional[str],
               production: tuple, candidate: tuple, seed: str,
               provider: Optional[str] = None, model: Optional[str] = None) -> tuple[str, str, bool, Any]:
    """One blind review. Returns ``(verdict, reason, candidate_is_step1, result)``.

    ``result`` is the raw LLM result (for cost accounting); raises whatever the
    llm raises so the caller can mark UNJUDGED.
    """
    candidate_is_step1 = random.Random(f"{seed}").random() < 0.5
    step1, step2 = ((candidate, production) if candidate_is_step1
                    else (production, candidate))
    prompt = JUDGE_INPUT_TEMPLATE.format(
        excerpt=redact(build_excerpt(transcript, first_user)),
        step1=redact(render_step(step1)),
        step2=redact(render_step(step2)),
    )
    kwargs: dict[str, Any] = dict(
        system_prompt=JUDGE_SYSTEM,
        instructions="Compare the two steps as JSON matching the schema.",
        input=[{"type": "text", "text": prompt}],
        json_schema=JUDGE_SCHEMA,
        schema_name="understudy.judge",
        temperature=0,
        max_tokens=300,
        timeout=120,
        purpose="understudy.judge",
    )
    if provider:
        kwargs["provider"] = provider
    if model:
        kwargs["model"] = model
    result = harness.complete_with_fallback(llm, kwargs)
    verdict, reason = map_verdict(_judge_parsed(result), candidate_is_step1)
    return verdict, reason, candidate_is_step1, result
