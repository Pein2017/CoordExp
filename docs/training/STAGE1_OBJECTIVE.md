---
doc_id: docs.training.stage1-objective
layer: docs
doc_type: reference
status: canonical
domain: training
summary: Stage-1 objective surfaces and coord-token training behavior.
updated: 2026-04-11
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

CoordExp can supervise coordinate tokens with **distribution-based losses** (recommended default):

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

This is the intended way to run a pure-SFT Stage-1 recipe with
hard CE + soft CE + W1 + CIoU + bbox-size aux enabled together.

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
