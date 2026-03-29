---
doc_id: docs.training.lvis
layer: docs
doc_type: design-note
status: canonical
domain: training
summary: LVIS federated-label integration design, implementation notes, and migration guide for Stage-1, Stage-2, and evaluation.
updated: 2026-03-29
---

# LVIS Integration Guide

This page explains how CoordExp's COCO-oriented training/eval stack changes when
the dataset is LVIS.

The key upstream references are:

- the LVIS dataset paper (`https://www.lvisdataset.org/assets/lvis_v0.5.pdf`)
- the official `lvis-api` evaluation semantics (`lvis/eval.py`, `lvis/results.py`)

## What Changes Semantically Relative To COCO

COCO-style dense detection usually assumes that unlabeled categories in an image
are absent for evaluation purposes. LVIS does not.

LVIS is federated / partially labeled. For each category `c`, images fall into:

- `P_c`: exhaustive positive set, where all instances of `c` are annotated
- `N_c`: verified negative set, where `c` is verified absent
- outside `P_c ∪ N_c`: detections for `c` are not evaluated as ordinary false positives

In LVIS JSON, the practical image-level handles are:

- `neg_category_ids`
- `not_exhaustive_category_ids`

CoordExp therefore must not treat "missing from GT sequence" as "absent from the
image" when training or evaluating LVIS.

## What Stays The Same

These parts of the current pipeline remain valid:

- Stage-1 teacher-forced serialization stays dense object-sequence SFT.
- Standard token CE and bbox geometry losses remain valid for annotated LVIS
  objects.
- The existing JSONL contract, collators, and assistant payload shape still
  work.
- The current clean-prefix / FN-append Stage-2 structure remains the right
  backbone.

The default training geometry for LVIS should remain bbox-only.

Why bbox-only is the default:

- the active `stage2_two_channel` Channel-B target builder is explicitly bbox-only
- LVIS polygons are still useful for optional evaluation and ablations
- bbox-only keeps LVIS and COCO training behavior aligned at the objective level

Use the existing LVIS bbox exports in `public_data/lvis/rescale_32_1024_bbox_max60/`
as the default training source. Polygon-capable LVIS exports remain useful for
inference/eval experiments and later ablations.

Important artifact distinction:

- the currently materialized Stage-1 JSONLs under
  `public_data/lvis/rescale_32_1024_bbox_max60/` are valid dense bbox training
  inputs, but they do not yet carry top-level LVIS federated `metadata`
- those cached `train.coord.jsonl` / `val.coord.jsonl` exports are already
  pre-sorted for `custom.object_ordering: sorted`
  - verified against the runtime `(minY, minX)` contract over all rows
  - scan result: `0` unsorted rows in both train and val
- the richer federated metadata path is implemented in the converter/runtime so
  regenerated LVIS exports can support Stage-2 and LVIS-aware evaluation cleanly

## Stage-1 Changes

Minimal Stage-1 changes are preferred.

Implemented changes:

- LVIS converter rows now preserve federated metadata in top-level `metadata`.
- `JSONLinesBuilder` now keeps row metadata so it survives into runtime samples.
- Two explicit prompt variants were added:
  - `lvis_stage1_federated`
  - `lvis_stage2_federated`
- Dense prompt overrides now support full prompt replacement instead of suffix-only
  tweaks, so LVIS prompts do not accidentally inherit COCO wording like
  "detect every object".

Stage-1 policy:

- supervise only the verified annotation subset that is present in the sequence
- do not imply that omitted visible categories are absent
- do not change token CE / bbox geometry masking for annotated objects

This keeps Stage-1 close to the current infrastructure while avoiding the main
LVIS failure mode: teaching the model that every unlabeled category omission is
negative evidence.

The canonical LVIS Stage-1 preset now makes the teacher-forcing loss bundle
explicit:

- base CE stays active for text + JSON structure tokens
- `custom.coord_soft_ce_w1` enables coord-token hard CE + soft CE + W1 + gate
- `custom.bbox_geo` enables Stage-1 decoded-box CIoU supervision
- `custom.bbox_size_aux` enables the existing bbox size auxiliary loss

## Stage-2 / Channel-B Changes

The active Channel-B triage is now LVIS-aware.

For unmatched anchor objects:

- `verified_positive`: always dead
- `verified_negative`: always dead
- `not_exhaustive`: allowed into the explorer-support path
- unevaluable / outside verified sets: allowed into the explorer-support path

This yields a hybrid objective:

- FP-neutral for unevaluable / partially labeled categories
- ignore-aware under LVIS metadata
- pseudo-label-friendly only for ambiguous categories, not for verified-positive
  or verified-negative misses

The resulting behavior is intentionally minimal:

- no new pseudo-label subsystem was introduced
- existing explorer support and pseudo-positive promotion remain the mechanism
- LVIS policy only gates which unmatched anchors are even eligible

New Channel-B telemetry now separates:

- `train/triage/lvis_verified_positive_dead_count`
- `train/triage/lvis_verified_negative_dead_count`
- `train/triage/lvis_not_exhaustive_count`
- `train/triage/lvis_unevaluable_count`

## Evaluation Changes

The offline evaluator now supports LVIS-aware official-style evaluation through:

- `metrics: lvis`
- `metrics: both`

Behavior:

- `metrics: coco` keeps COCO semantics
- `metrics: lvis` requires LVIS federated metadata
- `metrics: both` means "official dataset metric + f1-ish"
  - on COCO-like data this is COCO + f1-ish
  - on LVIS data this is LVIS + f1-ish

Implementation details:

- detections are mapped to fixed LVIS category ids through the canonical LVIS
  category catalog carried in row metadata
- per-image detections are capped by `lvis_max_dets` (default `300`)
- LVIS ignore behavior is approximated inside the existing `pycocotools` path by
  emitting synthetic crowd-ignore annotations for:
  - `not_exhaustive` categories
  - unevaluable categories outside verified positive/negative sets

Useful LVIS metrics now include:

- `bbox_AP`, `bbox_AP50`, `bbox_AP75`
- `bbox_APs`, `bbox_APm`, `bbox_APl`
- `bbox_APr`, `bbox_APc`, `bbox_APf`
- `bbox_AR300`
- the corresponding `segm_*` metrics when polygon predictions are present

Extra LVIS diagnostics are emitted in eval summaries and counters:

- `lvis_diag_matched_verified_positive`
- `lvis_diag_verified_negative_unmatched`
- `lvis_diag_ignored_not_exhaustive`
- `lvis_diag_ignored_unevaluable`

## Code Handles

Primary implementation surfaces:

- `public_data/converters/lvis_converter.py`
- `src/datasets/builders/jsonlines.py`
- `src/config/prompt_variants.py`
- `src/config/prompts.py`
- `src/common/lvis_semantics.py`
- `src/config/schema.py`
- `src/eval/detection.py`
- `src/infer/engine.py`
- `src/infer/pipeline.py`
- `src/bootstrap/trainer_setup.py`
- `src/trainers/losses/bbox_geo.py`
- `src/trainers/metrics/mixins.py`
- `src/trainers/stage2_two_channel.py`
- `src/trainers/stage2_two_channel/target_builder.py`
- `src/trainers/stage2_two_channel/types.py`
- `src/trainers/stage2_rollout_aligned.py`

Useful config handles:

- `custom.train_jsonl`
- `custom.val_jsonl`
- `custom.extra.prompt_variant`
- `custom.bbox_geo`
- `rollout_matching.eval_detection.metrics`
- `rollout_matching.eval_detection.lvis_max_dets`
- `eval.metrics`
- `eval.lvis_max_dets`

Reusable prompt fragments:

- `configs/stage1/_shared/prompt_lvis_stage1_federated.yaml`
- `configs/stage2_two_channel/_shared/prompt_lvis_stage2_federated.yaml`

## Migration Guide

### 1. Export LVIS JSONL

Default bbox-only training export:

```bash
bash public_data/lvis/reproduce_max60_exports.sh
```

Current Stage-1 pretraining JSONLs:

- `/data/CoordExp/public_data/lvis/rescale_32_1024_bbox_max60/train.coord.jsonl`
- `/data/CoordExp/public_data/lvis/rescale_32_1024_bbox_max60/val.coord.jsonl`

### 2. Stage-1 Training

Ready-to-run Stage-1 config:

- `configs/stage1/lvis_bbox_max60_1024.yaml`

Key overrides in that config:

- extends the canonical 4B Stage-1 coord-token recipe (`profiles/4b/coord_soft_ce_gate_coco80_desc_first.yaml`)
- `custom.train_jsonl: public_data/lvis/rescale_32_1024_bbox_max60/train.coord.jsonl`
- `custom.val_jsonl: public_data/lvis/rescale_32_1024_bbox_max60/val.coord.jsonl`
- `custom.extra.prompt_variant: lvis_stage1_federated`
- `custom.object_field_order: desc_first`
- `custom.object_ordering: sorted`
- `custom.coord_soft_ce_w1: { ce_weight: 1.0, soft_ce_weight: 1.0, w1_weight: 1.0, gate_weight: 5.0 }`
- `custom.bbox_geo: { enabled: true, smoothl1_weight: 0.0, ciou_weight: 1.0 }`
- `custom.bbox_size_aux.enabled: true`
- `template.max_pixels: 1048576`
- `custom.offline_max_pixels: 1048576`

No extra presort step is needed for the current cached 1024 bbox exports. They
already satisfy the loader's top-left ordering invariant for
`custom.object_ordering: sorted`.

Example launch:

```bash
config=configs/stage1/lvis_bbox_max60_1024.yaml gpus=0,1 conda run -n ms bash scripts/train.sh
```

### 3. Stage-2 Training

Ready-to-run Stage-2 config:

- `configs/stage2_two_channel/lvis_bbox_max60_1024.yaml`

Key overrides in that config:

- `model.model: output/stage1/lvis_bbox_max60_1024/epoch_4-stage1-lvis_bbox_max60_1024-hard_ce_soft_ce_w1_ciou_bbox_size-merged`
- `custom.train_jsonl: public_data/lvis/rescale_32_1024_bbox_max60/train.coord.jsonl`
- `custom.val_jsonl: public_data/lvis/rescale_32_1024_bbox_max60/val.coord.jsonl`
- `custom.extra.prompt_variant: lvis_stage2_federated`
- `custom.object_field_order: desc_first`
- `custom.object_ordering: sorted`
- `stage2_ab.pipeline.bbox_geo: { smoothl1_weight: 0.0, ciou_weight: 1.0 }`
- `stage2_ab.pipeline.coord_reg: { coord_ce_weight: 1.0, soft_ce_weight: 1.0, w1_weight: 1.0, coord_gate_weight: 5.0 }`
- `template.max_pixels: 1048576`
- `custom.offline_max_pixels: 1048576`
- `rollout_matching.eval_detection.metrics: f1ish`

The conservative `f1ish` default is intentional for the current cached
`1024/max60` LVIS export. Switch the config to `metrics: lvis` or `metrics: both`
only after wiring in a metadata-bearing LVIS JSONL from the updated converter path.

Example direct learner launch:

```bash
config=configs/stage2_two_channel/lvis_bbox_max60_1024.yaml gpus=0,1 conda run -n ms bash scripts/train.sh
```

Example server-mode launch:

```bash
server_gpus=0,1,2,3,4,5 train_gpus=6,7 config=configs/stage2_two_channel/lvis_bbox_max60_1024.yaml conda run -n ms bash scripts/train_stage2.sh
```

### 4. LVIS Inference + Evaluation

The repo now includes an LVIS-oriented bench config:

- `configs/bench/lvis_bbox_max60_val_infer_eval_base.yaml`

That bench config is intentionally conservative because it points at the current
metadata-light Stage-1-style LVIS export:

- `infer.prompt_variant: lvis_stage2_federated`
- `infer.object_field_order: desc_first`
- `infer.object_ordering: sorted`
- `eval.metrics: f1ish`

To run official LVIS AP from the unified infer/eval pipeline, first swap
`infer.gt_jsonl` to a regenerated metadata-bearing LVIS JSONL from the updated
converter path, then change:

- `eval.metrics: lvis` or `both`
- `eval.lvis_max_dets: 300`

Example run:

```bash
PYTHONPATH=. conda run -n ms python scripts/run_infer.py --config configs/bench/lvis_bbox_max60_val_infer_eval_base.yaml
```

Offline evaluator example:

```bash
PYTHONPATH=. conda run -n ms python scripts/evaluate_detection.py --config configs/eval/detection.yaml --pred_jsonl output/bench/your_run/eval/gt_vs_pred_scored.jsonl --metrics lvis --lvis-max-dets 300
```

## Optional Improvements Later

Reasonable follow-ups, ordered from lowest to highest complexity:

- add dedicated LVIS smoke configs under `configs/stage1/` and `configs/stage2_two_channel/`
- add semantic-desc calibration specifically for long-tail LVIS categories
- add stricter parity tests against the external `lvis-api` package when that
  dependency is available in the runtime env
- add category-restricted pseudo-label buffering across steps or epochs instead
  of same-step explorer support only
