---
name: coordexp-codebase
description: Navigate the CoordExp research codebase by routing through the canonical docs catalog, agent index, and implementation map.
---

# CoordExp Codebase Navigation

Use local `docs/` as the source of truth.
Do not maintain a second routing layer inside this skill.

## Primary Entry Points

1. `docs/AGENT_INDEX.md`
2. `docs/catalog.yaml`
3. `docs/PROJECT_CONTEXT.md`
4. `docs/SYSTEM_OVERVIEW.md`
5. `docs/IMPLEMENTATION_MAP.md`

## Precedence

When sources disagree, use:

1. `openspec/specs/`
2. `docs/`
3. `openspec/changes/<active-change>/`
4. `progress/`

## Task Routing

- Whole-project context:
  - `docs/AGENT_INDEX.md`
  - `docs/catalog.yaml`
  - `docs/PROJECT_CONTEXT.md`
- Data contract or preprocessing:
  - `docs/data/README.md`
  - `docs/data/CONTRACT.md`
  - `docs/data/PREPARATION.md`
- Stage-1 training:
  - `docs/training/README.md`
  - `docs/training/STAGE1_OBJECTIVE.md`
  - `configs/stage1/`
- Stage-2 training:
  - `docs/training/README.md`
  - `docs/training/STAGE2_DESIGN.md`
  - `docs/training/STAGE2_RUNBOOK.md`
  - `docs/training/METRICS.md`
  - `configs/stage2_two_channel/`
- Inference and evaluation:
  - `docs/eval/README.md`
  - `docs/eval/CONTRACT.md`
  - `docs/eval/WORKFLOW.md`
  - `docs/ARTIFACTS.md`
- Standards:
  - `docs/standards/README.md`

## Code Entry Points

- training entrypoint:
  - `src/sft.py`
- config loading:
  - `src/config/loader.py`
- datasets:
  - `src/datasets/`
  - `src/datasets/geometry.py`
- Stage-2 training:
  - `src/trainers/stage2_two_channel.py`
  - `src/trainers/rollout_matching/`
  - `src/trainers/teacher_forcing/`
- inference:
  - `src/infer/pipeline.py`
  - `scripts/run_infer.py`
- evaluation:
  - `src/eval/detection.py`
  - `scripts/evaluate_detection.py`

## Guardrails

- Use `progress/` only when the current docs do not answer a historical or empirical question.
- Prefer the catalog and agent index over blind repo-wide scans.
- Treat `configs/dlora/` as historical lineage, not the default training path.
