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
6. `openspec/specs/runtime-architecture-refactor-program/spec.md` for runtime structure and ownership seams

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
  - `docs/training/STAGE2_RUNBOOK.md`
  - `docs/training/METRICS.md`
  - `openspec/specs/stage2-ab-training/spec.md`
  - `openspec/specs/rollout-matching-sft/spec.md`
  - `openspec/specs/runtime-architecture-refactor-program/spec.md`
  - `configs/stage2_two_channel/`
- Inference and evaluation:
  - `docs/eval/README.md`
  - `docs/eval/CONTRACT.md`
  - `docs/eval/WORKFLOW.md`
  - `docs/ARTIFACTS.md`
  - `openspec/specs/inference-pipeline/spec.md`
  - `openspec/specs/inference-engine/spec.md`
  - `openspec/specs/detection-evaluator/spec.md`
  - `openspec/specs/runtime-architecture-refactor-program/spec.md`
- Standards:
  - `docs/standards/README.md`

## Code Entry Points

- training entrypoint:
  - `src/sft.py`
  - `scripts/train.sh`
  - `scripts/train_stage2.sh`
- bootstrap/runtime manifest helpers:
  - `src/bootstrap/`
- config loading:
  - `src/config/loader.py`
- datasets:
  - `src/datasets/`
  - `src/datasets/geometry.py`
- Stage-2 training:
  - `src/trainers/stage2_two_channel.py`
  - `src/trainers/stage2_two_channel/`
  - `src/trainers/stage2_rollout_aligned.py`
  - `src/trainers/rollout_aligned_targets.py`
  - `src/trainers/rollout_aligned_evaluator.py`
  - `src/trainers/rollout_matching_sft.py`
  - `src/trainers/rollout_runtime/`
  - `src/launchers/stage2_vllm_server.py`
  - `src/trainers/rollout_matching/`
  - `src/trainers/teacher_forcing/`
- inference:
  - `src/infer/pipeline.py`
  - `src/infer/engine.py`
  - `src/infer/artifacts.py`
  - `src/infer/backends.py`
  - `scripts/run_infer.py`
- evaluation:
  - `src/eval/detection.py`
  - `src/eval/orchestration.py`
  - `src/eval/artifacts.py`
  - `scripts/evaluate_detection.py`

## Guardrails

- Use `progress/` only when the current docs do not answer a historical or empirical question.
- Prefer the catalog and agent index over blind repo-wide scans.
- Treat `configs/fusion/` as the historical/experimental multi-dataset surface, not the default training path.
