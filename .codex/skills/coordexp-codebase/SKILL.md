---
name: coordexp-codebase
description: Use when navigating the CoordExp research codebase, locating current docs/specs/code entrypoints, or changing data, training, Stage-2, inference, evaluation, artifact, or provenance behavior.
---

# CoordExp Codebase Navigation

Use local `docs/` and `openspec/specs/` as the source of truth. This skill is a pointer layer only; do not duplicate the repo's docs catalog here.

## Primary Entry Points

1. `docs/AGENT_INDEX.md`
2. `docs/catalog.yaml`
3. `docs/PROJECT_CONTEXT.md`
4. `docs/SYSTEM_OVERVIEW.md`
5. `docs/IMPLEMENTATION_MAP.md`
6. `openspec/specs/runtime-architecture-refactor-program/spec.md` for runtime structure and ownership seams

## Working Pattern

1. Read the relevant docs/spec route first.
2. Use `rg` or `rtk rg` to narrow files and symbols.
3. For Python exploration or edits, use Serena MCP after narrowing.
4. Verify against the smallest realistic tests or artifacts named by `docs/IMPLEMENTATION_MAP.md`.

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
  - `docs/data/PACKING.md`
- Stage-1 training:
  - `docs/training/README.md`
  - `docs/training/STAGE1_OBJECTIVE.md`
  - `docs/data/PACKING.md`
  - `configs/stage1/`
- Stage-2 training:
  - `docs/training/README.md`
  - `docs/training/STAGE2_RUNBOOK.md`
  - `docs/training/METRICS.md`
  - `docs/training/LVIS.md`
  - `openspec/specs/stage2-ab-training/spec.md`
  - `openspec/specs/rollout-matching-sft/spec.md`
  - `openspec/specs/teacher-forcing-unified-loss-registry/spec.md`
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
  - `src/config/schema.py`
- datasets:
  - `src/datasets/`
  - `src/datasets/geometry.py`
- Stage-2 training:
  - `src/trainers/stage2_coordination.py`
  - `src/trainers/stage2_two_channel.py`
  - `src/trainers/stage2_two_channel/`
  - `src/trainers/stage2_ab/`
  - `src/trainers/stage2_rollout_aligned.py`
  - `src/trainers/rollout_aligned_targets.py`
  - `src/trainers/rollout_aligned_evaluator.py`
  - `src/trainers/rollout_matching_sft.py`
  - `src/trainers/rollout_runtime/`
  - `src/launchers/stage2_vllm_server.py`
  - `src/trainers/rollout_matching/`
  - `src/trainers/teacher_forcing/`
- inference:
  - `src/infer/pipeline.py::run_pipeline`
  - `src/infer/pipeline.py`
  - `src/infer/engine.py`
  - `src/infer/artifacts.py`
  - `src/infer/backends.py`
  - `scripts/run_infer.py`
  - `scripts/postop_confidence.py`
- evaluation:
  - `src/eval/detection.py::evaluate_and_save`
  - `src/eval/detection.py`
  - `src/eval/artifacts.py`
  - `src/eval/confidence_postop.py`
  - `src/eval/bbox_confidence.py`
  - `src/callbacks/detection_eval.py`
  - `scripts/evaluate_detection.py`
  - `scripts/evaluate_proxy_detection_bundle.py`

## Guardrails

- Use `progress/` only when the current docs do not answer a historical or empirical question.
- Prefer the catalog and agent index over blind repo-wide scans.
- Preserve offline-prepared JSONL, geometry/image alignment, Qwen3-VL chat-template compatibility, artifact names, manifest/provenance surfaces, and metric key semantics.
- Keep config-first behavior. Do not add new CLI flags for stable workflows when YAML can express the run.
- Update docs when changing user-facing defaults, entrypoints, artifact names, log keys, or recommended workflows.
- Update OpenSpec when changing stable training/evaluation behavior, config contracts, loss semantics, or normative metric semantics.
- Treat `configs/fusion/` as the historical/experimental multi-dataset surface, not the default training path.
