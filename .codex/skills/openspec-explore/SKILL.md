---
name: openspec-explore
description: Use when the user wants OpenSpec-oriented exploration before or during a compatibility-sensitive change.
---

# OpenSpec Explore

Requires `openspec` CLI only when reading active change status.

Use this only when the conversation is already about whether an OpenSpec contract should exist or how an active OpenSpec change should evolve.

## Flow

1. Ground the question in current repo docs/code/artifacts before theorizing.
2. Separate hypothesis, contract requirement, implementation plan, measured result, and durable interpretation.
3. If an active change is named, read its artifacts and identify what is settled vs still exploratory.
4. Offer 2-3 concrete paths only when there is a real fork affecting compatibility, reproducibility, or eval validity.
5. End with the smallest next action: create/update artifact, use a repo-local plan, run a probe, or leave as discussion.

## Guardrails

- Do not create OpenSpec artifacts unless the user asks.
- Do not promote ordinary research brainstorming into OpenSpec ceremony.
- Do not encode benchmark claims without artifact-backed evidence and scope labels.
