---
doc_id: docs.system-overview
layer: docs
doc_type: overview
status: canonical
domain: repo
summary: End-to-end flow from data intake to training, inference, evaluation, and artifacts.
updated: 2026-03-09
---

# System Overview

Purpose: map the end-to-end CoordExp flow from data intake to training, inference, evaluation, and reproducibility artifacts.
Authority: explanatory system guide for the current codebase; if this page conflicts with a spec or runbook, defer to `docs/PROJECT_CONTEXT.md` and `openspec/specs/`.
Read this after: `docs/PROJECT_CONTEXT.md`
Read this before: domain runbooks under `docs/data/`, `docs/training/`, and `docs/eval/`
Primary code handles: `src/config/loader.py`, `src/datasets/`, `src/sft.py`, `src/trainers/stage2_two_channel.py`, `src/infer/pipeline.py`, `src/eval/detection.py`
Verification: `rg -n "stage2|duplicate_ul|run_infer|DetectionEvalCallback|coord_soft_ce_w1" src scripts configs docs`

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
- training uses `do_resize=false`.

## 2. Dataset Build And Template Encoding

Training and inference both pass through the same CoordExp-style multimodal formatting layer.

- Main code handles:
  - `src/datasets/dense_caption.py`
  - `src/datasets/builders/jsonlines.py`
  - `src/config/prompts.py`
  - `src/config/loader.py`
- What happens here:
  - JSONL rows are read,
  - image paths are resolved,
  - assistant targets are rendered as CoordJSON,
  - multimodal chat-template inputs are prepared for Qwen3-VL-compatible training/inference.

This is the layer to inspect when:
- a JSONL record renders incorrectly,
- prompt variants drift between train and infer,
- tokenization or coord-token boundaries look wrong.

## 3. Training Surfaces

### Shared Entry Point

- Entry point: `src/sft.py`
- Shared lower-level config base: `configs/base.yaml`
- Typed config loading and validation:
  - `src/config/loader.py`
  - `src/config/schema.py`

### Stage-1 Baseline SFT

Use Stage-1 when you want teacher-forced baseline training without rollout-aware matching.

- Current config tree: `configs/stage1/`
- Main docs:
  - [`docs/training/README.md`](training/README.md)
  - [`docs/training/STAGE1_OBJECTIVE.md`](training/STAGE1_OBJECTIVE.md)
  - [`docs/data/PACKING.md`](data/PACKING.md)
- Main code handles:
  - `src/sft.py`
  - `src/metrics/dataset_metrics.py`
  - `src/trainers/losses/coord_soft_ce_w1.py`
  - `src/trainers/metrics/mixins.py`

### Stage-2 Rollout-Aware Training

Use Stage-2 when you need rollout-time matching, clean-prefix Channel-B supervision, or vLLM server-mode training.

- Current config tree: `configs/stage2_two_channel/`
- Main docs:
  - [`docs/training/STAGE2_DESIGN.md`](training/STAGE2_DESIGN.md)
  - [`docs/training/STAGE2_RUNBOOK.md`](training/STAGE2_RUNBOOK.md)
  - [`openspec/specs/stage2-ab-training/spec.md`](../openspec/specs/stage2-ab-training/spec.md)
- Main code handles:
  - `src/trainers/stage2_two_channel.py`
  - `src/trainers/rollout_matching/parsing.py`
  - `src/trainers/rollout_matching/matching.py`
  - `src/trainers/teacher_forcing/module_registry.py`
  - `src/trainers/teacher_forcing/modules/duplicate_ul.py`

Compatibility note:
- `src/trainers/stage2_ab_training.py` is a compatibility wrapper.
- The active implementation lives in `src/trainers/stage2_two_channel.py`.

## 4. Inference, Confidence, And Evaluation

### Inference

- CLI / pipeline entry point:
  - `scripts/run_infer.py`
- Main runtime code:
  - `src/infer/pipeline.py`
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
- Callback path for training-time offline eval:
  - `src/callbacks/detection_eval.py`

Important distinction:
- offline evaluator logs `eval_det_*`,
- trainer-native Stage-2 rollout evaluation logs `eval_rollout/*`.

## 5. Artifacts And Reproducibility

CoordExp writes paper-ready artifacts as part of normal execution.

- Artifact guide:
  - [`docs/ARTIFACTS.md`](ARTIFACTS.md)
- Training outputs usually include:
  - `resolved_config.json`
  - `runtime_env.json`
  - `run_metadata.json`
  - `logging.jsonl`
- Inference/eval outputs usually include:
  - `summary.json`
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
  - [`docs/training/STAGE2_DESIGN.md`](training/STAGE2_DESIGN.md)
  - [`docs/training/STAGE2_RUNBOOK.md`](training/STAGE2_RUNBOOK.md)
  - [`docs/IMPLEMENTATION_MAP.md`](IMPLEMENTATION_MAP.md)
- Change infer/eval artifacts:
  - [`docs/eval/README.md`](eval/README.md)
  - [`docs/IMPLEMENTATION_MAP.md`](IMPLEMENTATION_MAP.md)
