---
doc_id: docs.coordexp-swift
layer: docs
doc_type: worktree-authority
status: canonical
domain: repo
summary: Current routing and evidence guide for the CoordExp-Swift rebuilt training, inference, and evaluation infrastructure.
tags: [coordexp-swift, training, inference, eval, routing]
updated: 2026-07-03
---

# CoordExp-Swift Worktree

This page is the first stop for the rebuilt CoordExp-Swift worktree under
`/data/CoordExp/.worktrees/CoordExp-swift`.

The active goal of this worktree is a locally owned, inspectable training and
inference stack for Qwen3-VL detection research. It replaces the old
MS-Swift-centered execution path for this worktree. Legacy/mainline docs and
paths remain useful as reference material only when this page or an active
OpenSpec change explicitly points to them.

## Current Verdict

CoordExp-Swift is a successful V1 backbone.

- Training: production-style supervised training completed with the rebuilt
  `src/` infrastructure and wrote a usable final checkpoint.
- Inference: the rebuilt `src/infer.py` plus `src/inference/` path runs real
  HF/Qwen generation with adapter and special-token embedding delta support.
- Evaluation: the rebuilt `src/eval/detection_consumer.py` consumes the Swift
  scored artifact family and writes bbox mAP/mRecall metrics.
- Validation scope: the accepted correctness gate is the fixed val200 run.
  A full validation-dataset or full benchmark run is not required for this V1
  readiness claim unless the user explicitly asks for one.

Tiny Wave 7 smokes are implementation-readiness evidence only. The val200 run is
the accepted local validation evidence because it uses real data, scored
artifacts, the standardized Swift evaluator, and expected mAP scale.

## Active Source Topology

```text
raw coord JSONL
  -> src/data/
  -> src/templates/
  -> src/qwen/ encoding and no-resize image planning
  -> src/packing/
  -> src/qwen/ forward inputs
  -> src/losses/
  -> src/training/supervised_trainer.py
  -> src/artifacts/
  -> src/infer.py + src/inference/
  -> src/eval/detection_consumer.py
```

Primary code handles:

- Training entry: `src/train.py`
- Training orchestration: `src/training/pipeline.py`
- Trainer core: `src/training/supervised_trainer.py`
- Data/examples: `src/data/`
- Template rendering: `src/templates/`
- Packing and supervision: `src/packing/`, `src/supervision/`
- Qwen loading, encoding, image, position, and forward helpers: `src/qwen/`
- Losses and token-type gates: `src/losses/`
- Adapter and DoRA source gates: `src/adapters/`
- Optimizer parameter groups: `src/optim/`
- Artifacts and checkpoints: `src/artifacts/`
- Inference entry: `src/infer.py`
- Inference runtime/backend/prompt/parser/scoring/artifacts: `src/inference/`
- Direct Swift evaluator: `src/eval/detection_consumer.py`
- Eval-forward helper: `src/eval/forward.py`

Do not create or route to a `src/infer/` package in this worktree. The public
entry is the file `src/infer.py`; the implementation package is
`src/inference/`.

## Active Config Routes

- Training production configs: `configs/coordexp_swift/prod/`
- Training smoke configs: `configs/coordexp_swift/smoke/`
- Inference configs: `configs/coordexp_swift/infer/`
- DeepSpeed helper config: `configs/coordexp_swift/deepspeed/`

The current accepted validation config is:

- `configs/coordexp_swift/infer/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_r16a32_step917_val200.yaml`

The full-dataset benchmark config is optional reference material:

- `configs/coordexp_swift/infer/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_r16a32_benchmark.yaml`

Do not treat the full-dataset benchmark config as a required readiness gate.

## Evidence Handles

Production training checkpoint:

- `outputs/prod/coordexp_swift/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_r16a32_llm_12000_accelerate8_ebs64_4epoch_warmup0p1-prod8-r16a32-ebs64-warmup0p1-20260702T170007Z/checkpoints/step-917`

Accepted val200 inference/eval run:

- `outputs/coordexp_swift/infer/val200/qwen3-vl-2b-desc-first-geo-sorted-pure-ce-dora-r16a32-step917-val200-20260703T035007Z`

Accepted evaluator metrics:

- `outputs/coordexp_swift/infer/val200/qwen3-vl-2b-desc-first-geo-sorted-pure-ce-dora-r16a32-step917-val200-20260703T035007Z/eval_coco_fixed_gt_scale/metrics.json`

Key values from that metrics file:

- `row_count: 200`
- `gt_object_count: 1444`
- `pred_object_count: 1243`
- `mAP: 0.4111788135144427`
- `mAP_50: 0.5616311086141887`
- `mAP_75: 0.43402742703563857`
- `mRecall: 0.4790356074108587`

The evaluator converts inline GT norm1000 coord-bin boxes to pixel `xyxy`.
Scored predictions are already parser-normalized pixel `xyxy`. Mixed-unit
COCO sidecars are invalid.

## Evaluation Gate Policy

For this worktree:

- Tiny/smoke inference runs prove implementation readiness only.
- The fixed val200 run is sufficient for V1 local validation and benchmark-style
  regression evidence.
- A full validation-dataset run is optional and requires an explicit user request.
- Official COCO test-dev submission remains a separate workflow and is not
  implied by local val200 acceptance.

This policy supersedes older wording that required full validation or full
benchmark inference before claiming the CoordExp-Swift V1 backbone is working.

## OpenSpec And Roadmap Authority

Active OpenSpec changes for the rebuild:

- `openspec/changes/rebuild-coordexp-swift-training-infra/`
- `openspec/changes/build-coordexp-swift-inference-infra/`
- `openspec/changes/standardize-coordexp-swift-detection-evaluator/`
- `openspec/changes/prepare-coordexp-swift-production-relaunch/`

Planning artifacts under `docs/superpowers/plans/` are useful provenance, but
the current status is this page plus the active OpenSpec task ledgers. If a dated
roadmap still contains unchecked boxes or old "blocked pending full benchmark"
wording, treat it as historical unless this page points to it as a live gate.

## Future Work Boundary

The V1 backbone intentionally does not claim rollout training, hidden-state
losses, feature-cache losses, vLLM execution, exact optimizer/RNG resume, or
DeepSpeed production benchmark completeness. Those should enter through a new
OpenSpec proposal after the V1 backbone is kept stable.
