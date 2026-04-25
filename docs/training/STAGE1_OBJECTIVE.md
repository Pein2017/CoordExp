---
doc_id: docs.training.stage1-objective
layer: docs
doc_type: reference
status: canonical
domain: training
summary: Stage-1 objective surfaces and coord-token training behavior.
updated: 2026-04-25
---

# Coord Objective & Adapter

This document details the specialized training objectives and architectural adapters used for coordinate tokens in CoordExp.

Scope note:
- This page is primarily the Stage-1 / baseline coord-objective reference.
- The canonical Stage-1 packing contract now lives in [`../data/PACKING.md`](../data/PACKING.md):
  one hard `global_max_length` cap, offline static packing, full-length probing before plan build,
  and fail-fast when any atomic sample exceeds the cap.
- For Stage-2 pipeline-declared training, the canonical objective surface now lives under:
  - `stage2_ab.pipeline` for `custom.trainer_variant: stage2_two_channel`
  - `rollout_matching.pipeline` for `custom.trainer_variant: stage2_rollout_aligned`
- In those Stage-2 paths, `coord_reg`, `bbox_geo`, and `loss_duplicate_burst_unlikelihood` are declared through the pipeline surface described in:
  - `docs/training/STAGE2_RUNBOOK.md`
  - `docs/training/METRICS.md`
- Legacy `custom.coord_soft_ce_w1.*` authoring should not be used for pipeline-declared Stage-2 configs.
- For standard Stage-1 SFT, the active non-pipeline teacher-forcing surface is:
  - `custom.coord_soft_ce_w1.*`
  - `custom.bbox_geo.*`
  - `custom.bbox_size_aux.*`
- For the set-continuation Stage-1 experiment, the active surface is:
  - `custom.trainer_variant: stage1_set_continuation`
  - `custom.stage1_set_continuation.*`
  - top-level `benchmark.*`
  - `configs/stage1/set_continuation/production.yaml`
  - v1 rejects packing and uses repeated independent candidate forwards
- Narrow V1 exception:
  - `custom.bbox_format: cxcy_logw_logh` or `custom.bbox_format: cxcywh`
    defines an experimental Stage-1-only profile
  - under that profile, the allowed Stage-1 surface narrows to
    `custom.coord_soft_ce_w1.*` with hard CE plus positive coord/text gating
  - `custom.bbox_geo.*`, `custom.bbox_size_aux.*`, soft CE, W1, and trainer-side
    rollout/Stage-2 surfaces are intentionally out of scope and should be
    treated as invalid for that experiment

## Current Mechanism Note

Inference-only duplication studies on existing `merged` checkpoints now support
a more specific rollout-risk framing than the earlier generic "attention drifts
away from vision" explanation:

- the strongest onset-local separator is the early coordinate escape behavior at
  `x1` and `y1`
- healthy same-desc continuations usually evacuate probability mass away from
  the previous or local bbox neighborhood quickly
- duplicated continuations often keep `x1` / `y1` diffuse, high-entropy, or
  locally sticky long enough for rollout history to lock the model into a
  repeated-object basin
- late history overwrite still matters, but current control evidence suggests
  it is better treated as a secondary amplifier than as the sole root cause

Working interpretation:

- `softCE`, `W1`, and expectation-decoded geometry can preserve smooth local
  coordinate structure that looks acceptable under teacher forcing
- during rollout, that same local smoothness can lower the escape barrier
  between nearby same-desc instances
- once the model fails to separate from the previous or local basin at
  `coord_x1` / `coord_y1`, prior generated coord tokens and recent history can
  make duplication self-reinforcing

This does **not** yet prove that clean from-scratch pure CE fully solves the
problem. The current CE-side references on disk remain continuation-style
proxies unless a token-compatible pure-CE checkpoint is evaluated under the
same onset-local protocol.

## Current Mechanism Note

Inference-only duplication studies on existing `merged` checkpoints now support
a more specific rollout-risk framing than the earlier generic "attention drifts
away from vision" explanation:

- the strongest onset-local separator is the early coordinate escape behavior at
  `x1` and `y1`
- healthy same-desc continuations usually evacuate probability mass away from
  the previous or local bbox neighborhood quickly
- duplicated continuations often keep `x1` / `y1` diffuse, high-entropy, or
  locally sticky long enough for rollout history to lock the model into a
  repeated-object basin
- late history overwrite still matters, but current control evidence suggests
  it is better treated as a secondary amplifier than as the sole root cause

Working interpretation:

- `softCE`, `W1`, and expectation-decoded geometry can preserve smooth local
  coordinate structure that looks acceptable under teacher forcing
- during rollout, that same local smoothness can lower the escape barrier
  between nearby same-desc instances
- once the model fails to separate from the previous or local basin at
  `coord_x1` / `coord_y1`, prior generated coord tokens and recent history can
  make duplication self-reinforcing

This does **not** yet prove that clean from-scratch pure CE fully solves the
problem. The current CE-side references on disk remain continuation-style
proxies unless a token-compatible pure-CE checkpoint is evaluated under the
same onset-local protocol.

## Coord distribution loss (coord tokens)

CoordExp can supervise coordinate tokens with **distribution-based losses**
(recommended default for the existing `xyxy` Stage-1 baseline):

- Standard full-vocab CE is applied **only to non-coordinate tokens** (text + JSON structure).
- At `<|coord_*|>` positions, the model is supervised via:
  - `CE` (optional): hard CE over the 1000-bin coord vocabulary (ablation knob; default `0.0`)
  - `softCE`: soft cross-entropy between predicted coord-bin distribution `p` and a unimodal Gaussian soft label `q`
  - `W1`: 1D Wasserstein-1 distance on discrete bins via CDF differences between `p` and `q`
  - `gate`: coord-vocab gate loss that penalizes probability mass leaking to non-coord tokens

```yaml
custom:
  coord_soft_ce_w1:
    enabled: true
    # total_loss += ce_weight * CE + soft_ce_weight * softCE + w1_weight * W1
    #             + gate_weight * gate + adjacent_repulsion_weight * adjacent_repulsion
    ce_weight: 0.0
    soft_ce_weight: 1.0
    w1_weight: 1.0
    gate_weight: 1.0
    temperature: 1.0
    target_sigma: 2.0
    target_truncate: 16
    adjacent_repulsion_weight: 0.0
    adjacent_repulsion_filter_mode: same_desc
    adjacent_repulsion_margin_ratio: 0.05
    adjacent_repulsion_copy_margin: 0.8
```

**Notes**:
- Coord-token positions are identified from **labels** (teacher forcing), never from model predictions.
- No decoded coordinates (argmax/expectation/median) are computed for training or metrics.
- Because this objective is optimized under teacher forcing, it does not by
  itself test whether rollout can escape a previously emitted same-desc local
  basin. The active duplication-collapse analysis therefore treats early
  `coord_x1` / `coord_y1` escape from the previous/local neighborhood as the
  primary rollout diagnostic surface.
- Logged losses (train/eval parity, eval uses `eval_` prefix):
  - Stage-1 coord-family loss keys include `coord_softce_w1/loss`, `coord_softce_w1/soft_ce`, `coord_softce_w1/w1`, `coord_softce_w1/gate`, and `coord_softce_w1/adjacent_repulsion`
  - Stage-1 coord diagnostics include `coord_diag/loss`, `coord_diag/soft_ce`, `coord_diag/w1`, `coord_diag/gate`, `coord_diag/adjacent_repulsion`, `coord_diag/adjacent_repulsion_pair_count`, `coord_diag/adjacent_repulsion_applied_count`, `coord_diag/adjacent_repulsion_copy_score_mean`, plus `coord_diag/coord_vocab_mass`, `coord_diag/coord_tokens`, and the mode flag `coord_diag/enabled`
- Stage-2 note:
  - `stage2_two_channel` and `stage2_rollout_aligned` still use provenance-aware metric families, but the active single-pass Stage-2 contract now routes Channel-A through `loss/text/*`, `loss/coord/*`, and `coord_diag/*`, while Channel-B uses `loss/B_rollout_text/*`, `loss/B_coord/*`, and `coord_diag/B/*`.
  - Historical iterative groups such as `loss/A1_*`, `loss/A2_*`, `coord_diag/A1/*`, and `coord_diag/A2/*` are no longer part of the active Stage-2 contract.

## Stage-1 set-continuation objective

`custom.trainer_variant: stage1_set_continuation` adds an off-by-default
Stage-1 training paradigm for testing set-conditioned continuation instead of
ordinary fixed-order next-object SFT.

The object-level objective is:

```text
score(o) = log P(entry(o) | image, prompt, prefix)
loss/mp = -logsumexp(score(o) for o in remaining_candidates)
```

Important semantics:

- `entry(o)` is the full serialized object dictionary entry, including `desc`,
  `bbox_2d`, and the object-entry structural terminator.
- Candidate scoring is full-entry, not token-wise multi-positive mixing.
- Non-coordinate candidate-entry labels use ordinary full-vocab logprob.
- `<|coord_*|>` labels use coord-vocabulary-normalized logprob, so raw-text
  integer coordinate training is out of scope for this v1 path.
- The global detection-list close sequence is separate from object-entry end
  tokens. V1 uses the CoordJSON schema close sequence `]}` and never treats
  `<|im_end|>`, `<|end_of_text|>`, or tokenizer EOS as the stop target for this
  objective.
- V1 branch execution is naive repeated independent forward:
  `prefix + candidate_A`, `prefix + candidate_B`, and so on. Candidates do not
  attend to each other. Prefix gradients are non-detached but recomputed for
  each branch.
- V1 rejects `training.packing` and `training.eval_packing` because branch
  selection and structural-close spans are sample-local.

Subset-prefix sampling is configured under:

```yaml
custom:
  trainer_variant: stage1_set_continuation
  stage1_set_continuation:
    subset_sampling:
      empty_prefix_ratio: 0.30
      random_subset_ratio: 0.50
      leave_one_out_ratio: 0.20
      full_prefix_ratio: 0.0
      prefix_order: random
    candidates:
      mode: exact
      max_candidates: null
```

The checked-in production profile uses the `30/50/20/0` mixture above to keep
first-object and arbitrary-prefix coverage while still sampling
nearly-complete leave-one-out prefixes. Schema defaults remain `30/45/20/5`,
so the explicit production config is the source of truth for this run. A
non-zero `full_prefix_ratio` remains supported for explicit weak-close
ablations.

Candidate modes:

- `candidates.mode: exact` scores all observed remaining candidates.
- `candidates.mode: uniform_subsample` scores at most `max_candidates`, which
  must be positive when this mode is enabled.
- PEM uses exact remaining logZ in exact mode and the uniform-importance
  estimated remaining logZ in uniform-subsample mode.
- The current implementation does not enforce a same-budget controller.
  `same_budget_label` in the checked-in production config is an authored
  benchmark note; realized budget is reported through
  `mp/repeated_forward_token_ratio_vs_baseline`.

Structural-close controls:

- `structural_close.close_start_suppression_weight` adds
  `loss/anti_close_start = -log(1 - P_close_start(prefix))` when observed GT
  remains.
- `structural_close.final_schema_close_weight` adds weak teacher-forced
  close-sequence loss only when no observed GT remains.
- Object-entry close tokens remain supervised inside candidate entries.
- Compatibility aliases `loss/anti_stop`, `loss/eod`, and `stop/p_stop_*` are
  emitted for dashboards, but the authoritative v1 semantics are structural
  close-start and CoordJSON close-sequence probabilities.

PEM threshold loss is included in v1:

```yaml
positive_evidence_margin:
  objective: threshold_loss
  threshold_space: full_entry_logZ
  rho: 0.90
  threshold_calibration: fixed_rho_0.90_no_external_evaluator_v1
```

When `objective: threshold_loss`, `loss/pem` is optimized and
`loss/mp_diagnostic` records the MP loss without adding it to the total
objective.

Current parser contract: `objective=threshold_loss` requires exactly one of
`rho` or `log_rho`; `threshold_space` is fixed to `full_entry_logZ`; and
`threshold_calibration` must be a non-empty provenance string. The checked-in
production profile uses an authored fixed threshold, not an externally
calibrated evaluator threshold.

Branch-local auxiliary objectives are toggleable for `coord_soft_ce_w1`,
`bbox_geo`, and `bbox_size_aux`, but they are not inherited through ordinary
one-sequence Stage-1 mixins. When enabled, each scored candidate branch masks
labels down to that candidate entry, computes the auxiliary atom from the same
branch logits, and averages valid candidate atoms uniformly. Responsibility-
weighted aux is intentionally not a v1 mode.

### Ordinary SFT final-close control

The ordinary one-sequence Stage-1 SFT path can still downweight final global
CoordJSON close supervision with:

```yaml
custom:
  sft_structural_close:
    enabled: true
    final_close_weight: 0.0
```

This adds per-token base-CE weights for the final global CoordJSON close
sequence `]}` only. It does not mask object-entry close tokens and it does not
target chat-template EOS tokens. Fractional weights are supported in `[0, 1]`.
This path also rejects packing because the close-span mask is sequence-local.

### Production entry config

The canonical entry config lives at:

```text
configs/stage1/set_continuation/production.yaml
```

It is the all-feature production profile for continuing from the current SOTA
coord-token Stage-1 SFT checkpoint:

```text
output_remote/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate/epoch_4-from-base-2B/v0-20260227-050057/checkpoint-1332-merged-full
```

The profile enables exact full-entry MP scoring, the `30/50/20/0`
empty/random/leave-one-out/full prefix mixture, close-start suppression when
observed GT remains, final schema close supervision disabled, and fixed-rho
PEM threshold loss with `rho=0.90`. It remains 2B, COCO80 desc-first,
coord-token-only,
`val200`/`f1ish_annotated`, and `training.packing: false`.

Because this run continues from an already four-epoch fine-tuned SOTA
checkpoint, the production config uses reduced continuation learning rates:
`learning_rate=5e-5`, `vit_lr=1e-5`, `aligner_lr=5e-5`, and coord-offset
`embed_lr=head_lr=5e-5`.

The production entry intentionally keeps the original COCO Stage-1 coord-token
training surface first. LVIS-proxy should be evaluated as a follow-up
dataset-choice ablation, not silently mixed into the first objective-change
run.

Train-time detection evaluation is distributed by default through
`custom.eval_detection.distributed: true`. Under DDP, every rank reuses its live
training model replica to decode a deterministic shard of the eval JSONL, rank 0
merges the shard artifacts back into the canonical `gt_vs_pred.jsonl`, and only
rank 0 runs final detection scoring/log injection. This keeps val200 generation
from occupying only GPU 0 while preserving rank-0-owned metric semantics and the
existing `eval_detection/step_<N>/` artifact layout.

The profile pins `coord_soft_ce_w1`, `bbox_geo`, and `bbox_size_aux` disabled
so the first production run isolates the continuation objective. Its
`custom.extra.benchmark_report.same_budget_label` value is a comparison note,
not a runtime-enforced budget constraint.

## Stage-1 non-canonical bbox V1 experiments

When `custom.bbox_format` is `cxcy_logw_logh` or `cxcywh`, the Stage-1 loss
surface is intentionally much narrower than the existing `xyxy` baseline:

- model-facing `bbox_2d` slots become either `[cx, cy, u(w), u(h)]` or
  `[cx, cy, w, h]`
- coord-token bbox supervision is hard CE only
- `custom.coord_soft_ce_w1.enabled` must remain `true`
- `ce_weight > 0`
- `soft_ce_weight = 0`
- `w1_weight = 0`
- `gate_weight > 0`
- `text_gate_weight > 0`
- `temperature = 1.0`, `target_sigma = 2.0`, and `target_truncate = null`
  are compatibility-only defaults and do not define a soft-label path here
- `custom.bbox_geo.*` and `custom.bbox_size_aux.*` are out of scope for this
  experiment

This V1 profile exists to isolate the parameterization question under minimal
loss complexity. It is not the same recipe as the legacy `xyxy` Stage-1
baseline documented above.

Evaluation note:

- standalone inference/eval still emits canonical pixel `xyxy` standardized
  artifacts
- official score-aware evaluation remains available through a deterministic
  constant-score `gt_vs_pred_scored.jsonl` compatibility artifact
- confidence post-op remains unsupported for these non-canonical formats in
  this V1 path
- checkpoints trained on the legacy model-facing `xyxy` serialization are not
  semantically compatible with non-canonical `infer.bbox_format`
- if you force an old `xyxy` checkpoint through the `cxcy_logw_logh` or
  `cxcywh` infer
  path, the runtime may still emit canonicalized `xyxy` artifacts, but those
  outputs should be treated as a compatibility stress test rather than as valid
  non-canonical rollout behavior
- training data for this profile must come from the offline-prepared
  derived preset root such as
  `public_data/<dataset>/<preset>_cxcy_logw_logh/train.coord.jsonl` or
  `public_data/<dataset>/<preset>_cxcywh/train.coord.jsonl` rather
  than a runtime reinterpretation of canonical preset JSONL

## Stage-1 raw-text xyxy benchmark

The minimal raw-text benchmark keeps canonical `xyxy` geometry and the shared
norm1000 lattice, but removes coord-token rendering:

- train from canonical `train.norm.jsonl` / `val.norm.jsonl`
- set `custom.coord_tokens.enabled: false`
- keep `custom.coord_tokens.skip_bbox_norm: true`
- keep `custom.bbox_format: xyxy`
- keep `custom.coord_soft_ce_w1.enabled: false` for the pure-CE slice

Inference/eval for this benchmark must stay explicit:

- `infer.mode: text`
- `infer.pred_coord_mode: norm1000`
- `infer.bbox_format: xyxy`

Evaluation and visualization always canonicalize through
`norm1000 -> pixel-space xyxy` using the per-record image `width` and `height`
before drawing boxes or scoring metrics. Score-aware mAP for this benchmark
comes from numeric-span confidence post-op on the raw bbox integers rather than
from constant-score compatibility artifacts.

## Stage-1 bbox geometry loss

Stage-1 can also supervise decoded bbox geometry directly from the same
teacher-forced coord logits, without switching to a Stage-2 trainer variant.

```yaml
custom:
  bbox_geo:
    enabled: true
    parameterization: xyxy
    smoothl1_weight: 0.0
    ciou_weight: 1.0
    center_weight: 1.0
    size_weight: 0.25
```

Semantics:

- the Stage-1 trainer keeps standard base CE on text + structure tokens
- coord-token positions are still supervised from labels
- decoded bbox coordinates are produced by the same expectation decode used by
  the existing bbox-size auxiliary path
- outward `bbox_2d` supervision remains canonical `xyxy`; `parameterization:
  center_size` changes only the internal regression loss-space
- `smoothl1_weight` and `ciou_weight` gate the two decoded-box geometry atoms
- `parameterization: center_size` derives `(cx, cy, log_w, log_h)` from the
  canonical decoded box, applies stronger center supervision plus softer
  size supervision, and still keeps CIoU on canonical `xyxy`
- `center_weight` and `size_weight` only affect the internal regression term
  when `parameterization: center_size`; legacy configs that only specify
  `enabled`, `smoothl1_weight`, and `ciou_weight` continue to resolve to
  `parameterization: xyxy`
- this Stage-1 surface is intentionally config-first and narrow; Stage-2
  pipeline-declared configs should continue to express geometry through the
  `bbox_geo` objective module instead of `custom.bbox_geo`
- `loss/geo/bbox_smoothl1` stays the stable metric key for the configured bbox
  regression term, so compare it across runs only after joining against
  `resolved_config.json`

This is the intended way to run a legacy pure-SFT Stage-1 recipe with
hard CE + soft CE + W1 + CIoU + bbox-size aux enabled together. It does not
describe the narrower `cxcy_logw_logh` V1 experiment above.

## Decoded BBox Geometry Loss (Stage-1)

Standard Stage-1 SFT can optionally add decoded bbox geometry supervision from
the same forward logits used for coord-token teacher forcing.

```yaml
custom:
  bbox_geo:
    enabled: true
    parameterization: center_size
    smoothl1_weight: 0.01
    ciou_weight: 1.0
    center_weight: 1.0
    size_weight: 0.25
```

Semantics:

- the model still uses standard full-vocab CE on non-coord tokens
- coord-token positions still use `custom.coord_soft_ce_w1.*`
- decoded bbox losses are applied only to bbox-only Stage-1 samples where the
  coord-token stream forms explicit `bbox_2d` quartets
- expectation decoding is used for the Stage-1 geometry probe, matching the
  active teacher-forcing geometry baseline elsewhere in the repo
- canonical external `bbox_2d` / `xyxy` serialization, parsing, inference, and
  evaluation contracts do not change under `parameterization: center_size`
- keep `smoothl1_weight > 0` when you want the center/size regression branch to
  be active; `center_weight` and `size_weight` do not affect a CIoU-only setup

Metric handles:

- `loss/geo/bbox_geo`
- `loss/geo/bbox_smoothl1`
- `loss/geo/bbox_ciou`
- `bbox_geo/loss_per_sample`
- `bbox_geo/groups_total`
- `bbox_geo/coord_slots_total`

Cheaper debug loop:

```bash
PYTHONPATH=. conda run -n ms python -m src.sft \
  --config configs/stage1/smoke/lvis_bbox_max60_1024.yaml
```

For a center-size experiment, override the same `custom.bbox_geo` block with:

```yaml
custom:
  bbox_geo:
    enabled: true
    parameterization: center_size
    smoothl1_weight: 0.01
    ciou_weight: 1.0
    center_weight: 1.0
    size_weight: 0.25
```

## Coord-offset adapter (tie-head / single shared table)

When training with coord tokens, CoordExp can optionally avoid updating the full vocabulary embedding
and instead learn a small **offset adapter** over just the coord-token id range.

**Key idea**:
- Freeze the base `embed_tokens.weight` and `lm_head.weight`.
- Train a compact offset table only for `<|coord_0|>.. <|coord_999|>` token ids.

**Config**:
```yaml
custom:
  coord_offset:
    enabled: true
    # Default: Qwen3-VL-style tie-head (single/shared lookup table for embed + head).
    tie_head: true
    ids: { start: 151670, end: 152669 }  # <|coord_0|>.. <|coord_999|>
    # Optional: learning-rate overrides for the offset parameters.
    # When tie_head: true, only embed_lr is used (head_lr is ignored).
    embed_lr: 1.0e-4
    head_lr: 1.0e-4
    weight_decay: 0.0
```

**Semantics**:
- `tie_head: true` (recommended; default)
  - The adapter trains a **single** offset table and uses it for both:
    - embedding lookup (adds offsets to hidden states for coord tokens), and
    - output projection (adds logits for coord tokens via `hidden @ offset^T`).
  - This is equivalent to applying a single delta to the tied embedding/head table for coord tokens,
    which matches the intended tie-head routine of Qwen-family LMs.
- `tie_head: false` (legacy/ablation)
  - Trains separate `embed_offset` and `head_offset` tables (two independent deltas).
  - Export/merge may need to materialize `lm_head.weight` and disable tying to preserve separate behavior.

**Export/merge**:
- Use `scripts/merge_coord.sh` to merge LoRA/DoRA and bake the coord-offset adapter into a merged HF checkpoint.
  - With `tie_head: true`, the merged checkpoint can keep tied embeddings (single table).
  - With `tie_head: false`, the merged checkpoint may need an explicit `lm_head.weight` tensor and `tie_word_embeddings: false`.
