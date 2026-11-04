---
name: understudy
description: "Audition cheaper or different models on the user's real Hermes history before switching: replays past decisions, judges them against the current model, reports takeover rate, misses and projected cost."
version: 1.0.0
author: stffinfcti
license: MIT
metadata:
  hermes:
    tags: [models, cost, evaluation, understudy]
---

# Understudy

Use this when the user asks whether a cheaper or different model could handle
their work, what a switch would save, or which model to pick — and whenever the
user names a specific candidate model ("could kimi-k3 handle my sessions?").

## Workflow

1. **Estimate first — never run blind.** Call `understudy_audition` with
   `dry_run: true` and the candidate list (1–4, `provider:model` or a bare
   model on the current provider). Tell the user what would be replayed: how
   many past decisions from how many conversations, and the planned call
   count. The audition spends real tokens on their providers, capped by
   `budget_usd` (default $1.50).
2. **Run only after the user agrees.** Call `understudy_audition` again with
   `dry_run: false`. Progress posts into the chat. Long runs continue in the
   background past the tool timeout and post the scorecard into the chat when
   they finish — you do not need to poll.
3. **Report honestly — and show the card.** After a finished run, put the
   `MEDIA:<card_png>` line from the result on its own line so the user sees
   the scorecard. Read it too: takeover rate with its 95% CI, critical and
   minor misses, the per-category breakdown, projected monthly cost, and the
   verdict. Do not oversell: if a category is weak or costs were unknown
   (budget not enforced), say so and name it.
4. **Follow-ups.** Use `understudy_report` (optionally `detail: "misses"`) to
   answer questions about what the candidate got wrong, or `run_id` for an
   older audition.

## Rules

- Never change the user's model configuration yourself. If the verdict is
  Ready and they want to switch, point them at the `/model` command.
- Understudy needs the `llm.model_override` (and `llm.provider_override` for
  `provider:model` candidates) capability granted; if a run is refused, tell
  the user to consent on the plugins screen or set
  `plugins.entries.understudy.llm.allow_model_override: true`.
- If too few decisions were found, widen `days` or `sample` before concluding
  anything.
