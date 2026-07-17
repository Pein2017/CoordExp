---
name: workflow-self-distillation
description: Use when reviewing Codex tasks, memories, sessions, or recurring CoordExp preferences to distill durable skills, subagents, automations, docs, extensions, or deliberate skips.
---

# Workflow Self-Distillation

Turn repeated work and judgment preferences into the smallest durable agent asset.

## Evidence Order

1. Read the user's brief or `self-distillation.md`.
2. Mine recent tasks, memories, and rollout summaries.
3. Use Chronicle only when enabled, and confirm its discoveries elsewhere.
4. Inventory skills, agent metadata, repo docs, hooks, helpers, and automations before proposing additions.

Prefer a 30-day window unless the user gives another window. If history is shorter, say so.

Classify each asset as personal/repo-local, official/plugin-managed,
generated/vendor-provided, or a local wrapper. Do not remove or merge official
or plugin-managed assets unless the user explicitly puts them in scope.

## Parallel Split

When subagents are requested or permitted, split independent read-only evidence
lanes: history/memory, repo/docs, asset overlap, and domain-heavy operations.
Give each lane a self-contained scope and request evidence handles, confidence,
recommended form, and overlap warnings. The parent synthesizes and decides.

## Candidate Test

Act only when the candidate:

- occurred at least twice, or is clearly likely to recur and costly;
- has stable inputs, repeatable steps, and a clear output or stopping condition;
- improves speed, consistency, reliability, or correctness;
- is not already adequately covered.

Search memory for prior consolidation, retirement, or restoration decisions and
report conflicts with the live state.

Choose the smallest form:

- skill: reusable workflow or playbook;
- custom subagent: bounded delegated role with clear inputs and output;
- automation: recurring reminder, monitor, report, or scheduled check;
- extend existing: when the gap belongs to a current skill;
- docs or agent metadata: authority, delegation, communication, or role guidance;
- skip: one-off, sensitive, ambiguous, poorly evidenced, or overlapping.

When evaluating an external implementation workflow, use a benchmark only if
it can change a durable routing or process decision. Include representative
expansion-prone work and a lean negative control, require independent contract
and quality adjudication, and never treat raw lines of code as adoption proof.

## Output First

Before creating assets, give a compact shortlist: workflow, dated evidence,
frequency/confidence, recommended form, and why it is worth creating or skipping.

Create only when explicitly asked or after approval. Put communication
preferences in `AGENTS.md` or memory, not a new skill. Validate changed assets.

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

Report created/extended assets, deliberate skips, unresolved evidence gaps, and commands run.
