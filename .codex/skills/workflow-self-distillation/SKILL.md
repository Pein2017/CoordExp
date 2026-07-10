---
name: workflow-self-distillation
description: Use when asked to review recent Codex work, memories, sessions, task history, or recurring design/research preferences to identify durable skills, subagents, automations, docs, extensions, or deliberate skips.
---

# Workflow Self-Distillation

Use this to turn repeated work and recurring judgment preferences into durable agent assets.

## Evidence Order

1. Read the user's brief or `self-distillation.md`.
2. Mine recent Codex threads, task summaries, memories, and rollout summaries.
3. Check Chronicle only if an enabled tool exists; use it for discovery and confirm important facts elsewhere.
4. Inventory existing skills, `agents/openai.yaml`, `.codex/agents/*.toml`, repo docs, hooks, ops helpers, and automations before proposing anything new.

Prefer a 30-day window unless the user gives another window. If history is shorter, say so.

Classify provenance before recommending skill changes:

- personal/repo-local;
- official/plugin-managed;
- generated/vendor-provided;
- local wrapper around an official workflow.

Do not recommend removing or merging official/plugin-managed skills unless the user explicitly puts them in scope.

## Parallel Split

When subagents are explicitly requested or the parent workflow permits them, split broad evidence into read-only lanes instead of stacking all context in one thread:

- memory and rollout summaries;
- recent Git history plus `research/`, legacy `progress/` provenance, and docs;
- existing assets and overlap risks;
- operations/artifact/data-transfer or another domain-heavy lane.

Give each subagent a self-contained prompt, forbid edits, and ask for evidence handles, confidence, recommended form, and overlap warnings. The parent must synthesize, remove duplicate reasoning, and decide.

## Candidate Test

Act only when the candidate:

- occurred at least twice, or is clearly likely to recur and costly;
- has stable inputs, repeatable steps, and a clear output or stopping condition;
- improves speed, consistency, reliability, or correctness;
- is not already adequately covered.

Search recent memories and rollout summaries for prior consolidation, retirement, or restoration decisions. If prior decisions conflict with current state, report the conflict and explain whether provenance changes the recommendation.

Choose the smallest form:

- skill: reusable workflow or playbook;
- custom subagent: bounded delegated role with clear inputs and output;
- automation: recurring reminder, monitor, report, or scheduled check;
- extend existing: when the gap belongs to a current skill;
- docs or agent metadata: when the durable lesson is an authority boundary, delegation rule, or agent role tweak;
- skip: one-off, sensitive, ambiguous, poorly evidenced, or overlapping.

## Output First

Before creating assets, produce a compact shortlist with:

- repeated workflow;
- evidence and dates;
- frequency / confidence;
- recommended form;
- why it is or is not worth creating.

Create only when the user explicitly asks for asset creation or approves the shortlist. Validate changed skill/agent assets before reporting them complete.

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
