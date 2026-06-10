---
name: workflow-self-distillation
description: Use when asked to review recent Codex work, memories, sessions, or task history to identify repeated manual workflows worth packaging as skills, custom subagents, automations, extensions, or deliberate skips.
---

# Workflow Self-Distillation

Use this to turn recent repeated work into durable agent assets without creating speculative clutter.

## Evidence Order

1. Read the user's brief or `self-distillation.md`.
2. Mine recent Codex threads, task summaries, memories, and rollout summaries.
3. Check Chronicle only if an enabled tool exists; use it for discovery and confirm important facts elsewhere.
4. Inventory existing skills, `agents/openai.yaml`, repo docs, hooks, ops helpers, and automations before proposing anything new.

Prefer a 30-day window unless the user gives another window. If history is shorter, say so.

## Parallel Split

When subagents are available, split read-only lanes by evidence source:

- memory and rollout summaries;
- recent Git history plus `progress/` / docs;
- existing assets and overlap risks;
- operations/artifact/data-transfer or another domain-heavy lane.

Give each subagent a self-contained prompt, forbid edits, and ask for evidence handles, confidence, recommended form, and overlap warnings.

## Candidate Test

Act only when the candidate:

- occurred at least twice, or is clearly likely to recur and costly;
- has stable inputs, repeatable steps, and a clear output or stopping condition;
- improves speed, consistency, reliability, or correctness;
- is not already adequately covered.

Choose the smallest form:

- skill: reusable workflow or playbook;
- custom subagent: bounded delegated role with clear inputs and output;
- automation: recurring reminder, monitor, report, or scheduled check;
- extend existing: when the gap belongs to a current skill;
- skip: one-off, sensitive, ambiguous, poorly evidenced, or overlapping.

## Output First

Before creating assets, produce a compact shortlist with:

- repeated workflow;
- evidence and dates;
- frequency / confidence;
- recommended form;
- why it is or is not worth creating.

Create only when the user explicitly asks for asset creation or approves the shortlist. Keep changes small, source-aware, and validator-clean.

## Verification

For skill changes:

```bash
python .codex/skills/.system/skill-creator/scripts/quick_validate.py .codex/skills/<skill>
python - <<'PY'
import pathlib, yaml
for path in pathlib.Path(".codex/skills").glob("*/agents/openai.yaml"):
    yaml.safe_load(path.read_text())
PY
```

For final reporting, include created/extended assets, deliberate skips, unresolved evidence gaps, and commands run.
