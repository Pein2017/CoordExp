---
name: grill-me
description: Use when the user explicitly asks to grill, stress-test, or pressure-test a plan or decision before acting through concise, round-based questioning, optionally recording the outcome.
---

# Grill Me

Interrogate the plan until its consequential choices, assumptions, and evidence
are clear enough to act on. Map decisions as a tree: a decision may unblock
others that depend on it.

## Method

- Inspect the environment instead of asking the user for discoverable facts.
  When fact-finding can run independently and subagents are allowed, dispatch
  narrow lanes without blocking unrelated questions.
- Work in rounds. In each round, ask the whole **frontier**: every important
  decision whose prerequisites are settled. Number the questions, recommend an
  answer for each, then wait.
- Recompute the frontier after every response. Defer any question that depends
  on an answer still open in the current round.
- Focus on choices that materially affect meaning, evidence, cost,
  compatibility, or reversibility. Resolve routine implementation details
  yourself.
- Challenge vague goals, hidden assumptions, missing acceptance evidence, and
  premature commitment. Prefer a narrower probe when evidence is insufficient.
- Do not implement or launch the plan until the user confirms shared
  understanding and asks to proceed.

Keep each frontier item compact:

```text
1. Question — recommendation (brief reason)
```

Stop when the frontier is empty: no consequential branch remains silently
assumed. Summarize the resolved choices, open risks, and recommended next step.

Default to chat. If the user asks for a durable record, read
[recording.md](references/recording.md) after the decision resolves. Permission
to record does not authorize implementation.
