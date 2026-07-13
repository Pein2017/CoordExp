---
name: grill-me
description: Use when the user explicitly asks to grill, stress-test, or pressure-test a plan or decision before acting, optionally recording the outcome.
---

# Grill Me

Interrogate the plan until its consequential choices, assumptions, and evidence
are clear enough to act on.

## Method

- Inspect available files, artifacts, and code before asking about discoverable
  facts.
- Walk the decision tree one dependency at a time. Ask exactly one question,
  then wait for the answer.
- With every question, state why it matters and give a concrete recommendation.
- Focus on choices that materially affect meaning, evidence, cost,
  compatibility, or reversibility. Resolve routine implementation details
  yourself.
- Challenge vague goals, hidden assumptions, missing acceptance evidence, and
  premature commitment. Prefer a narrower probe when evidence is insufficient.
- Do not implement or launch the plan until the user confirms shared
  understanding and asks to proceed.

Use a compact conversational form, not a repeated template:

```text
Why this matters: ...
Recommendation: ...
Question: ...
```

Stop when remaining uncertainty no longer changes the decision or next action.
Summarize the resolved choice, open risk, and recommended next step.

Default to chat. If the user asks for a durable record, read
[recording.md](references/recording.md) after the decision resolves. Permission
to record does not authorize implementation.
