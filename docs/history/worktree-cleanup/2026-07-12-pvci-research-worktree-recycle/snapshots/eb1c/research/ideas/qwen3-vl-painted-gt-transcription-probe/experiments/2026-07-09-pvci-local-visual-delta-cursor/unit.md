---
title: PVCI Local Visual-Delta Cursor Probe
description: Tests whether the same-row stored painted-clean visual-feature delta remains effective when restricted to Qwen merged visual tokens overlapping the rendered mark.
type: idea
role: research-unit
authority: non_normative_research
promotion_status: not_promoted
unit_id: 2026-07-09-pvci-local-visual-delta-cursor
topic: qwen3-vl-painted-gt-transcription-probe
status: complete
tags:
  - coordexp-swift
  - research-unit
  - painted-gt
  - pvci
  - visual-delta
  - locality
  - non-pixel-cursor
updated: 2026-07-09
---

# PVCI Local Visual-Delta Cursor Probe

## Question

The completed
[PVCI Feature-Store Delta Cursor Probe](../2026-07-09-pvci-feature-store-delta-cursor/unit.md)
showed that same-row stored painted visual features, and same-row additive
`painted - clean` visual-feature deltas, reproduce the painted row-binding
effect during clean-image generation.

This unit asks whether that stored delta effect is spatially local after the
Qwen3-VL vision tower:

```text
capture phase:
  clean image -> Qwen vision tower -> clean visual features
  painted image -> Qwen vision tower -> painted visual features
  store full delta plus rendered-mark merged-token masks

generation phase:
  clean image bytes only
  + selected stored post-vision delta at Qwen3VLModel.get_image_features
  -> row-level binding behavior
```

The claim boundary is intentionally narrow. A positive result would support a
same-row post-vision local actuator for the rendered mark. It would not prove a
learned cursor, clean-image-only cursor generation, pixel-space causal locality,
STOP behavior, coverage, or production inference behavior.

## Evidence Scope

- Checkout or branch: `/data/CoordExp/.codex/worktrees/eb1c/CoordExp`,
  `codex/qwen3-vl-painted-gt-transcription-probe`.
- Parent unit:
  [2026-07-09 PVCI Feature-Store Delta Cursor](../2026-07-09-pvci-feature-store-delta-cursor/unit.md).
- Parent artifact root:
  `/data/CoordExp/outputs/painted_gt/pvci_feature_store_delta/e1_wrong_object_jitter_medium_val32`.
- Main checkpoint: E1 anti-copy checkpoint:
  `/data/CoordExp/outputs/painted_gt/pvci_step2/train_snap_radius/painted_gt_pvci_step2_snap_radius_train256_anti_copy_mix_step484_warm_start_dora_all_towers_accelerate8_ebs8_16epoch-pvci-step2-anticopy-retry3-20260708T1726Z/checkpoints/step-484`.
- Matched row panel:
  `/data/CoordExp/outputs/painted_gt/pvci_row_binding/materialized/heldout_val100_jitter_medium_wrong_object/conditions/stepwise__wrong_object_mark/stepwise.wrong_object_mark.examples.jsonl`.
- First execution scope: the same `val32` different-description primary rows
  used by the parent global feature-store delta unit.

## Condition Matrix

Primary controls and probes:

- `clean_no_cursor`: clean image bytes, no feature intervention.
- `stored_full_feature`: clean image bytes during generation, but the hook
  returns stored painted visual features captured earlier from the same row.
- `clean_plus_stored_delta`: clean image bytes during generation, clean visual
  features computed live, then full same-row stored `(painted - clean)` deltas
  added to main image embeddings and all deepstack visual features.
- `clean_plus_mark_delta`: same as full delta, but only at merged visual tokens
  whose pixel cells overlap the rendered mark `bbox_pixels`.
- `clean_plus_non_mark_delta`: same-row complement control; apply the full
  delta outside the rendered-mark mask.
- `clean_plus_shifted_mark_delta`: same-row spatial control; apply the delta at
  a deterministic rolled mark mask with the same token count.
- `clean_plus_wrong_row_mark_delta`: wrong-row local control; apply a
  matching-shape wrong row's mark-local delta.
- `clean_plus_zero_delta`: patch-path control; clean visual features computed
  live, then zero deltas added.
- `clean_plus_wrong_row_delta`: wrong-row global control; clean visual features
  computed live, then a stored full delta from another row with matching
  feature shape added.

## Required Diagnostics

- Generation-phase receipts must prove:
  - `painted_vision_forward_count_eval=0`;
  - no painted replay pixels are collated during generation;
  - every request hits the feature store unless a control explicitly records a
    fallback;
  - clean-image paths and hashes remain separate from painted capture-source
    paths and hashes.
- Feature-store manifest entries must record:
  - rendered mark `bbox_pixels`;
  - source and target bbox pixels when available;
  - decoded image width and height;
  - patch size, merge size, `image_grid_thw`, merged-grid shape, and split size;
  - mask construction policy and token-order assumption;
  - selected mark-token indices, shifted-token indices, and complement/hash
    receipts;
  - per-region image and deepstack delta L2 norms and ratios.
- Condition receipts must record local mask summaries, token counts, selected
  region norms, hook counts, clean vision forward counts, and fallback counts.

## Interpretation Rules

Supported only if observed:

- `clean_plus_stored_delta` remains a positive control for the full same-row
  effect on this run.
- `clean_plus_zero_delta` remains neutral relative to `clean_no_cursor`.
- `clean_plus_mark_delta` approaches the full-delta row-binding outcome while
  `clean_plus_shifted_mark_delta`, `clean_plus_wrong_row_mark_delta`, and
  wrong-row global controls do not.
- `clean_plus_non_mark_delta` is weaker than the mark-local intervention if the
  effect is local to the rendered mark; if it is comparable or stronger, the
  post-vision actuator is distributed or not cleanly mark-local.

Not supported by this unit:

- A learned non-pixel cursor exists.
- A source-image-disjoint approximation works.
- A cursor can be generated from clean image content alone.
- The effect is local in raw pixel space before the vision tower.
- E1 is detector-ready.
- STOP, coverage, self-prefix, or autonomous object selection is solved.

## Execution Plan

1. Extend the feature-store probe with local, complement, shifted, and wrong-row
   local delta conditions.
2. Add focused unit tests for bbox-to-merged-token mapping, mask application,
   selected-region norm receipts, and wrong-row local behavior.
3. Run a `debug4` smoke with all conditions and inspect receipts before
   scaling.
4. Run the same `val32` different-description primary panel as the parent unit.
5. Analyze row-binding outcomes against the parent clean, full-delta, zero, and
   wrong-row baselines.

## Research Unit Closeout

Observed:

- Focused tests passed for the probe helpers:
  `pytest tests/painted_gt/test_visual_feature_store_probe.py -q`
  reported 10 passing tests.
- A 4-row smoke completed with all nine conditions:
  `/data/CoordExp/outputs/painted_gt/pvci_local_delta/e1_wrong_object_jitter_medium_debug4`.
- The main 32-row different-description primary run completed:
  `/data/CoordExp/outputs/painted_gt/pvci_local_delta/e1_wrong_object_jitter_medium_val32`.
- Feature-store manifest:
  `/data/CoordExp/outputs/painted_gt/pvci_local_delta/e1_wrong_object_jitter_medium_val32/feature_store/visual_feature_store_manifest.json`.
  The tensor artifact SHA-256 is
  `aaee2180dc8b709df25604244bcc9fc13b40845a79a45b8a215da28aa1317e09`.
- Compact comparison:
  `/data/CoordExp/outputs/painted_gt/pvci_local_delta/e1_wrong_object_jitter_medium_val32/local_visual_delta_comparison.json`.
- The capture phase ran 32 clean and 32 painted vision forwards. Generation
  receipts recorded `painted_vision_forward_count_eval=0` for every
  intervention condition.
- `stored_full_feature` recorded 16 hook calls, 32 feature-store hits, 0 clean
  vision forwards, and 0 painted eval vision forwards.
- Delta conditions each recorded 16 clean vision forwards and 32 feature-store
  hits. The wrong-row global and wrong-row local controls each had 1
  zero-delta fallback because one row lacked a matching-shape wrong-row partner.
- One row had an empty non-mark complement because the rendered mark covered
  every merged visual token for that image. The complement condition applied a
  zero complement for that row and the receipts record
  `empty_non_mark_row_count=1`.
- Main 32-row rates:
  - `clean_no_cursor`: `prediction_rate=0.53125`,
    `source_object_row_rate=0.0588`, `rendered_mark_copy_rate=0.1176`,
    `target_row_rate=0.4706`, `third_object_row_rate=0.1176`.
  - `stored_full_feature`: `prediction_rate=0.8125`,
    `source_object_row_rate=0.5000`, `rendered_mark_copy_rate=0.1154`,
    `target_row_rate=0.0`, `third_object_row_rate=0.0`.
  - `clean_plus_stored_delta`: same rates as `stored_full_feature`.
  - `clean_plus_mark_delta`: `prediction_rate=0.8125`,
    `source_object_row_rate=0.4231`, `rendered_mark_copy_rate=0.2308`,
    `target_row_rate=0.0`, `third_object_row_rate=0.0`.
  - `clean_plus_non_mark_delta`: `prediction_rate=0.7500`,
    `source_object_row_rate=0.3750`, `rendered_mark_copy_rate=0.2083`,
    `target_row_rate=0.0417`, `third_object_row_rate=0.0417`.
  - `clean_plus_shifted_mark_delta`: `prediction_rate=0.5625`,
    `source_object_row_rate=0.1111`, `rendered_mark_copy_rate=0.1667`,
    `target_row_rate=0.3333`, `third_object_row_rate=0.0556`.
  - `clean_plus_wrong_row_mark_delta`: `prediction_rate=0.6875`,
    `source_object_row_rate=0.0455`, `rendered_mark_copy_rate=0.0909`,
    `target_row_rate=0.2727`, `third_object_row_rate=0.2727`.
  - `clean_plus_zero_delta`: same rates as `clean_no_cursor`.
  - `clean_plus_wrong_row_delta`: `prediction_rate=0.5625`,
    `source_object_row_rate=0.0556`, `rendered_mark_copy_rate=0.1111`,
    `target_row_rate=0.2778`, `third_object_row_rate=0.2778`.

Supported:

- A same-row post-vision rendered-mark local delta is a strong actuator. It
  recovers most of the full same-row delta effect, eliminates target-row
  behavior on this panel, and strongly increases source/rendered-mark binding
  relative to clean/no-cursor and zero-delta controls.
- The full same-row additive delta remains equivalent to stored full painted
  features on this panel.
- The zero-delta path remains neutral.
- Shifted mark-local, wrong-row mark-local, and wrong-row global controls do
  not reproduce the same-row mark/full-delta behavior.

Not supported yet:

- Pure mark-locality of the same-row post-vision delta. The non-mark complement
  also recovers a substantial part of the effect, so the actuator is not cleanly
  localized to rendered-mark-overlap tokens alone.
- A learned non-pixel cursor.
- Source-image-disjoint generalization.
- Production inference behavior.

Next decider:

- Treat the visual actuator as at least partly distributed after the vision
  tower. The next decisive probe should separate rendered-mark tokens from
  object-region/source-box tokens and/or fit a source-image-disjoint lightweight
  approximation to the local-plus-context delta. Do not design a learned cursor
  as if the evidence proved a purely local mark-token mechanism.

Promotion decision:

- Not promoted. This remains a research-only same-row oracle probe under
  `scripts/probes/painted_gt` plus local tests and artifacts. Do not promote to
  OpenSpec, stable docs, config schema, or production inference until a
  source-image-disjoint non-oracle mechanism exists.
