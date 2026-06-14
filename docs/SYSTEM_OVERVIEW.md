---
doc_id: docs.system-overview
layer: docs
doc_type: overview
status: canonical
domain: repo
summary: End-to-end flow from data intake to training, inference, evaluation, and artifacts.
updated: 2026-05-16
---

# System Overview

Purpose: map the end-to-end CoordExp flow from data intake to training, inference, evaluation, and reproducibility artifacts.
Authority: explanatory system guide for the current codebase; if this page conflicts with a spec or runbook, defer to `docs/PROJECT_CONTEXT.md` and `openspec/specs/`.
Read this after: `docs/PROJECT_CONTEXT.md`
Read this before: domain runbooks under `docs/data/`, `docs/training/`, and `docs/eval/`
Primary code handles: `src/config/loader.py`, `src/datasets/`, `src/sft.py`, `src/detection/runtime.py`, `src/detection/template.py`, `src/common/detection_sequence.py`, `src/common/detection_compact_rows.py`, `src/bootstrap/`, `src/trainers/metrics/`, `src/metrics/events.py`, `src/training/`, `src/training/surfaces.py::TrainingSurfaceResolver`, `src/trainers/stage2_rollout_correction.py`, `src/trainers/rollout_aligned_targets.py`, `src/trainers/rollout_aligned_evaluator.py`, `src/launchers/stage2_vllm_server.py`, `src/infer/pipeline.py`, `src/infer/runtime.py`, `src/infer/backend.py`, `src/infer/backend_sync.py`, `src/infer/backend_vllm_server.py`, `src/infer/constraints.py`, `src/infer/artifacts.py`, `src/eval/detection.py`, `src/eval/detection_records.py`, `src/eval/detection_geometry.py`, `src/eval/detection_coco.py`, `src/eval/detection_lvis.py`, `src/eval/detection_duplicate_guard.py`, `src/eval/detection_f1ish.py`, `src/eval/detection_orchestrator.py`, `src/eval/orchestration.py`, `src/eval/artifacts.py`
Verification search: `rg -n "detection/runtime|detection_sequence|detection_compact_rows|MetricEvent|flatten_metric_events|TrainingSurfaceResolver|stage1_compact_trie_ce|stage2_rollout_correction|stage2_rollout_runtime|pipeline_manifest|run_metadata|backends|artifacts|orchestration" src scripts configs docs`

## Flow At A Glance

```text
raw annotations / public datasets
  -> offline conversion + resize + coord-tokenization
  -> JSONL contract
  -> dataset build + chat-template encode
  -> training (Stage-1 baseline or Stage-2 rollout-aware)
  -> inference artifacts
  -> confidence post-op (optional for scored COCO)
  -> evaluation + visualizations
  -> reproducibility artifacts and logs
```

## 1. Data Intake And Offline Preparation

CoordExp expects offline-prepared JSONL rather than ad-hoc runtime transforms.

- Current contract docs:
  - [`docs/data/CONTRACT.md`](data/CONTRACT.md)
  - [`docs/data/PREPARATION.md`](data/PREPARATION.md)
- Main code handles:
  - `public_data/scripts/`
  - `src/datasets/geometry.py`
  - `src/datasets/builders/jsonlines.py`
- Key config surfaces:
  - `custom.train_jsonl`
  - `custom.val_jsonl`
  - `custom.emit_norm: none`
  - `custom.coord_tokens.*`

Important invariant:
- images are resized offline,
- geometry stays aligned with images,
- all training and evaluation consume those offline-prepared images as-is,
- runtime vision processors must not resize them,
- training uses `do_resize=false`.

## 2. Dataset Build And Template Encoding

Training and inference both pass through the same CoordExp-style multimodal formatting layer.

- Main code handles:
  - `src/datasets/dense_caption.py`
  - `src/datasets/builders/jsonlines.py`
  - `src/config/prompts.py`
  - `src/config/loader.py`
  - `src/detection/template.py`
  - `src/common/detection_sequence.py`
  - `src/common/detection_compact_rows.py`
- What happens here:
  - JSONL rows are read,
  - image paths are resolved,
  - assistant targets are rendered as CoordJSON,
  - compact detection sequence rows can be rendered or parsed through the strict template and common compatibility facade,
  - multimodal chat-template inputs are prepared for Qwen3-VL-compatible training/inference.

This is the layer to inspect when:
  - a JSONL record renders incorrectly,
  - prompt variants drift between train and infer,
  - tokenization or coord-token boundaries look wrong.

Compact detection sequence ownership:
- strict template behavior lives in `src/detection/template.py`;
- the strict factory-visible template IDs are `stage1_json_pretty` and `compact_full`;
- `src/common/detection_sequence.py` is the common compatibility facade;
- `src/common/detection_compact_rows.py` owns stdlib-only compact row markers, rendering, and splitting;
- `compact_no_desc`, `compact_no_bbox`, and `compact_min` remain helper/compatibility formats, not strict factory IDs.

## 3. Training Surfaces

### Shared Entry Point

- Entry point: `src/sft.py`
- Latest compact detection runtime policy: `src/detection/runtime.py`
- Shared lower-level config base: `configs/base.yaml`
- Typed config loading and validation:
  - `src/config/loader.py`
  - `src/config/schema.py`
- Bootstrap and provenance helpers:
  - `src/bootstrap/pipeline_manifest.py`
  - `src/bootstrap/trainer_setup.py`
  - `src/bootstrap/run_metadata.py`

### Stage-1 Baseline SFT

Use Stage-1 when you want teacher-forced baseline training without rollout-aware matching.

- Current config tree: `configs/stage1/`
- Main docs:
  - [`docs/training/README.md`](training/README.md)
  - [`docs/training/STAGE1_OBJECTIVE.md`](training/STAGE1_OBJECTIVE.md)
  - [`docs/data/PACKING.md`](data/PACKING.md)
- Main code handles:
  - `src/sft.py`
  - `src/detection/runtime.py`
  - `src/detection/template.py`
  - `src/common/detection_sequence.py`
  - `src/common/detection_compact_rows.py`
  - `src/metrics/dataset_metrics.py`
  - `src/metrics/events.py`
  - `src/trainers/losses/coord_soft_ce_w1.py`
  - `src/trainers/metrics/mixins.py`
  - `src/trainers/metrics/batch_contract.py`
  - `src/trainers/metrics/structural_close.py`
- `src/trainers/metrics/recursive_detection.py`
- `src/trainers/metrics/aggregate_tokens.py`
- `src/trainers/metrics/coord_losses.py`

### Stage-1 Detection Teacher Forcing

Use this surface for the canonical clean-break Stage-1 detection
teacher-forcing route, `stage1_detection_teacher_forcing`.

- Current config route: `configs/stage1/detection_teacher_forcing/`
- Runtime policy owner: `src/detection/runtime.py`
- Template owner: `src/detection/template.py`
- Compatibility sequence facade: `src/common/detection_sequence.py`
- Row helper: `src/common/detection_compact_rows.py`

Current source contract:
- `src/detection/runtime.py` owns detection runtime support/preflight,
  teacher-forcing runtime policy, prompt/mode/custom shim resolution, and
  `build_detection_training_dataset`.
- `src/sft.py` delegates these policies and keeps backward-compatible private aliases.
- packing/cache fail fast remains in force for non-prefix compact Stage-1
  detection teacher-forcing surfaces. The prefix-denoising V1 experiment adds a
  narrow `prefix_denoising` config surface and a dedicated hybrid-packed path
  with encoded sample cache disabled.
- no new CLI flags are introduced by this extraction.

Quarantined legacy/comparator note:
- `configs/archive/detection_scene_clean_break/stage1/` contains quarantined
  recursive-detection CE migration history and comparator/ablation material. Do
  not use it as the current public Stage-1 detection teacher-forcing route.

### Stage-2 Rollout-Aware Training

Use Stage-2 when you need rollout prefix plus GT correction supervision or vLLM server-mode training.

- Current config tree: `configs/stage2/rollout_correction/`
- Main docs:
  - [`docs/training/STAGE2_RUNBOOK.md`](training/STAGE2_RUNBOOK.md)
  - [`docs/training/METRICS.md`](training/METRICS.md)
  - [`openspec/specs/stage2-rollout-correction/spec.md`](../openspec/specs/stage2-rollout-correction/spec.md)
  - [`openspec/specs/rollout-matching-sft/spec.md`](../openspec/specs/rollout-matching-sft/spec.md) only for retired-contract rejection checks
  - [`openspec/specs/runtime-architecture-refactor-program/spec.md`](../openspec/specs/runtime-architecture-refactor-program/spec.md)
- Main code handles:
  - `src/trainers/stage2_rollout_correction.py`
  - `src/trainers/stage2_coordination.py`
  - `src/trainers/rollout_aligned_targets.py`
  - `src/trainers/rollout_aligned_evaluator.py`
  - `src/launchers/stage2_vllm_server.py`
  - `src/infer/runtime.py`
  - `src/infer/backend.py`
  - `src/infer/backend_vllm_server.py`
  - `src/infer/backend_sync.py`
  - `src/infer/rollout_dispatch.py`
  - `src/trainers/rollout_matching/parsing.py`
  - `src/trainers/rollout_matching/matching.py`
  - `src/trainers/teacher_forcing/module_registry.py`

Compatibility note:
- `src/trainers/stage2_rollout_correction.py` is the public Stage-2 trainer surface.
- Shared Stage-2 rollout prompt/decode/backend/trace behavior routes through
  `src/infer/*`; trainer modules own residual correction orchestration,
  post-rollout packing, and training/eval metric projection.
- Stage-2 historical rationale is summarized from the current runbook; use [`docs/training/STAGE2_RUNBOOK.md`](training/STAGE2_RUNBOOK.md) and stable specs for current behavior.

## 4. Inference, Confidence, And Evaluation

### Inference

- CLI / pipeline entry point:
  - `scripts/run_infer.py`
- Main runtime code:
  - `src/infer/pipeline.py`
  - `src/infer/runtime.py`
  - `src/infer/backend.py`
  - `src/infer/constraints.py`
  - `src/infer/artifacts.py`
- Config surfaces:
  - `configs/infer/`
  - `configs/bench/`

Primary artifact:
- `gt_vs_pred.jsonl`

### Confidence Post-Op

- CLI entry point:
  - `scripts/postop_confidence.py`
- Config surface:
  - `configs/postop/confidence.yaml`

Primary scored artifact:
- `gt_vs_pred_scored.jsonl`

### Evaluation

- Offline evaluator entry point:
  - `scripts/evaluate_detection.py`
- Main runtime code:
  - `src/eval/detection.py`
  - `src/eval/detection_records.py`
  - `src/eval/detection_geometry.py`
  - `src/eval/detection_coco.py`
  - `src/eval/detection_lvis.py`
  - `src/eval/detection_duplicate_guard.py`
  - `src/eval/detection_f1ish.py`
  - `src/eval/detection_orchestrator.py`
  - `src/eval/orchestration.py`
  - `src/eval/artifacts.py`
- Callback path for training-time offline eval:
  - `src/callbacks/detection_eval.py`

Important distinction:
- offline evaluator logs `eval_det_*`,
- trainer-native Stage-2 rollout evaluation logs `eval/detection/*, eval/parsing/*, eval/description/*, eval/config/*, eval/runtime/*`.

Import compatibility note:
- `src/eval/detection.py` is the import-compatible facade.
- `src/eval/detection_orchestrator.py` owns the durable orchestration entrypoint for decomposed detection eval.
- `SemanticDescEncoder` facade patch/import compatibility is preserved for existing callers.

## 5. Artifacts And Reproducibility

CoordExp writes paper-ready artifacts as part of normal execution.

- Artifact guide:
  - [`docs/ARTIFACTS.md`](ARTIFACTS.md)
- Current architecture contract:
  - [`runtime-architecture-refactor-program/spec.md`](../openspec/specs/runtime-architecture-refactor-program/spec.md)
- Training outputs usually include:
  - `resolved_config.json`
  - `runtime_env.json`
  - `effective_runtime.json`
  - `pipeline_manifest.json`
  - `experiment_manifest.json`
  - `run_metadata.json`
  - `logging.jsonl`
- Inference/eval outputs usually include:
  - `summary.json`
  - `resolved_config.path`
  - `metrics.json`
  - scored JSONLs and overlays when enabled

## 6. Where To Go Next

- Change data format or preprocessing:
  - [`docs/data/README.md`](data/README.md)
  - [`docs/IMPLEMENTATION_MAP.md`](IMPLEMENTATION_MAP.md)
- Change Stage-1 baseline behavior:
  - [`docs/training/README.md`](training/README.md)
  - [`docs/training/STAGE1_OBJECTIVE.md`](training/STAGE1_OBJECTIVE.md)
- Change Stage-2 training behavior:
  - [`docs/training/STAGE2_RUNBOOK.md`](training/STAGE2_RUNBOOK.md)
  - [`docs/training/METRICS.md`](training/METRICS.md)
  - [`openspec/specs/stage2-rollout-correction/spec.md`](../openspec/specs/stage2-rollout-correction/spec.md)
  - [`openspec/specs/runtime-architecture-refactor-program/spec.md`](../openspec/specs/runtime-architecture-refactor-program/spec.md)
  - [`docs/IMPLEMENTATION_MAP.md`](IMPLEMENTATION_MAP.md)
- Change infer/eval artifacts:
  - [`docs/eval/README.md`](eval/README.md)
  - [`docs/eval/WORKFLOW.md`](eval/WORKFLOW.md) for operational flow; [`docs/ARTIFACTS.md`](ARTIFACTS.md) owns the full artifact inventory
  - [`openspec/specs/inference-pipeline/spec.md`](../openspec/specs/inference-pipeline/spec.md)
  - [`openspec/specs/inference-engine/spec.md`](../openspec/specs/inference-engine/spec.md)
  - [`openspec/specs/detection-evaluator/spec.md`](../openspec/specs/detection-evaluator/spec.md)
  - [`docs/IMPLEMENTATION_MAP.md`](IMPLEMENTATION_MAP.md)
