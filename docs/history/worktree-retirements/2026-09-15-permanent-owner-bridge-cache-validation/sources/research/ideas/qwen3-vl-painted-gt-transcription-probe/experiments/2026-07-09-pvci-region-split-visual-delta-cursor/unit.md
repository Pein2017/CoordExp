---
title: PVCI Region-Split Visual-Delta Cursor Probe
description: Splits the same-row post-vision delta into rendered-mark, source-object-box, rendered-halo, and source-ring regions to test whether the actuator follows paint, object region, or context.
type: idea
role: research-unit
authority: non_normative_research
promotion_status: not_promoted
unit_id: 2026-07-09-pvci-region-split-visual-delta-cursor
topic: qwen3-vl-painted-gt-transcription-probe
status: complete
tags:
  - coordexp-swift
  - research-unit
  - painted-gt
  - pvci
  - visual-delta
  - region-split
  - locality
updated: 2026-07-09
---

# PVCI Region-Split Visual-Delta Cursor Probe

## Question

The completed
[PVCI Local Visual-Delta Cursor Probe](../2026-07-09-pvci-local-visual-delta-cursor/unit.md)
showed that rendered-mark-overlap post-vision deltas are strong, but the
non-mark complement also carries substantial effect. This unit asks a sharper
question:

```text
Does the same-row visual actuator follow:
  A. the rendered paint bbox,
  B. the underlying marked/source object bbox,
  C. the rendered halo around that object,
  D. or the immediate visual context around the source object?
```

This is still a same-row oracle probe. It tests how an already captured
`painted - clean` post-vision delta decomposes across merged visual-token
regions. It does not define a learned cursor or production inference behavior.

## Evidence Scope

- Checkout or branch: `/data/CoordExp/.codex/worktrees/eb1c/CoordExp`,
  `codex/qwen3-vl-painted-gt-transcription-probe`.
- Parent unit:
  [2026-07-09 PVCI Local Visual-Delta Cursor](../2026-07-09-pvci-local-visual-delta-cursor/unit.md).
- Parent artifact root:
  `/data/CoordExp/outputs/painted_gt/pvci_local_delta/e1_wrong_object_jitter_medium_val32`.
- Main checkpoint: E1 anti-copy checkpoint:
  `/data/CoordExp/outputs/painted_gt/pvci_step2/train_snap_radius/painted_gt_pvci_step2_snap_radius_train256_anti_copy_mix_step484_warm_start_dora_all_towers_accelerate8_ebs8_16epoch-pvci-step2-anticopy-retry3-20260708T1726Z/checkpoints/step-484`.
- Matched row panel:
  `/data/CoordExp/outputs/painted_gt/pvci_row_binding/materialized/heldout_val100_jitter_medium_wrong_object/conditions/stepwise__wrong_object_mark/stepwise.wrong_object_mark.examples.jsonl`.
- Execution scope: the same `val32` different-description primary rows used by
  the parent local visual-delta unit.

## Condition Matrix

Core controls:

- `clean_no_cursor`: clean image bytes, no feature intervention.
- `stored_full_feature`: clean image bytes during generation, but the hook
  returns stored painted visual features.
- `clean_plus_stored_delta`: full same-row stored `(painted - clean)` delta.
- `clean_plus_zero_delta`: hook-path neutral control.

Region-split conditions:

- `clean_plus_mark_delta`: rendered mark bbox overlap.
- `clean_plus_source_box_delta`: underlying source/object bbox overlap.
- `clean_plus_rendered_halo_delta`: rendered mark bbox minus source/object bbox.
- `clean_plus_source_ring_delta`: one merged-token dilation around the source
  bbox, excluding the source bbox itself.

Negative controls:

- `clean_plus_shifted_mark_delta`: deterministic rolled mark-local mask.
- `clean_plus_wrong_row_mark_delta`: matching-shape wrong-row mark-local delta.
- `clean_plus_wrong_row_source_box_delta`: matching-shape wrong-row source-box
  delta.
- `clean_plus_wrong_row_delta`: matching-shape wrong-row full delta.

## Required Diagnostics

- Feature-store manifest entries must record rendered bbox, source bbox,
  selected token indices or hashes, region token counts, region overlaps, and
  selected-region image/deepstack L2 norms.
- Condition receipts must record hook counts, clean vision forward counts,
  painted eval vision forward counts, feature-store hits, fallback counts, and
  empty-region row counts.
- Analysis must report row-binding rates separately for source-object row,
  rendered-mark copy, target row, third-object row, prediction rate, and
  malformed/no-prediction count.

## Interpretation Rules

Supported only if observed:

- Source-box dominance: `source_box` approaches or exceeds `mark`, while
  `rendered_halo` and `source_ring` are weak.
- Paint-artifact dominance: `rendered_halo` is strong beyond source-box.
- Context-field evidence: `source_ring` is strong even without source-box.
- Distributed evidence: several regions partially recover the effect, or
  region strength tracks L2 energy/token budget more than semantic region.

Not supported by this unit:

- A learned non-pixel cursor.
- Source-image-disjoint generalization.
- A clean-image-only cursor generator.
- Pixel-space causal locality before the vision tower.
- Production inference or training behavior.

## Execution Plan

1. Extend the existing feature-store probe with source-box, rendered-halo,
   source-ring, and wrong-row source-box conditions.
2. Add focused tests for non-wrapping source-ring dilation, region mask
   application, selected-region norms, and wrong-row source-box behavior.
3. Run a `debug4` smoke with all core and region-split conditions.
4. Run the matched `val32` panel.
5. Materialize a compact region-split comparison JSON and close out this unit.

## Research Unit Closeout

Observed:

- Implemented the region-split feature-store probe in
  `scripts/probes/painted_gt/run_feature_store_delta_probe.py` and verified it
  with focused tests in `tests/painted_gt/test_visual_feature_store_probe.py`.
- Debug smoke artifact:
  `/data/CoordExp/outputs/painted_gt/pvci_region_split/e1_wrong_object_jitter_medium_debug4`.
- Main `val32` artifact root:
  `/data/CoordExp/outputs/painted_gt/pvci_region_split/e1_wrong_object_jitter_medium_val32`.
- Compact comparison JSON:
  `/data/CoordExp/outputs/painted_gt/pvci_region_split/e1_wrong_object_jitter_medium_val32/region_split_visual_delta_comparison.json`.
- Feature-store manifest:
  `/data/CoordExp/outputs/painted_gt/pvci_region_split/e1_wrong_object_jitter_medium_val32/feature_store/visual_feature_store_manifest.json`.
- Feature-store tensor SHA256:
  `2cb4dc20c4a4106c26e10404b4a2f30114640b4efbc64f3ba021b13ffafff5ff`.
- On the `val32` panel, `clean_no_cursor` had prediction rate `0.53125`
  and source-object row rate `0.0588`, while `stored_full_feature` and
  `clean_plus_stored_delta` both reached prediction rate `0.8125` and
  source-object row rate `0.5000`.
- Region split source-object row rates were:
  `mark=0.4231`, `non_mark=0.3750`, `source_ring=0.3636`,
  `source_box=0.2857`, and `rendered_halo=0.2727`.
- Negative controls did not restore same-row source binding:
  `shifted_mark=0.1111`, `wrong_row_mark=0.0455`,
  `wrong_row_source_box=0.0455`, and `wrong_row_full=0.0556`.
- Region-mask diagnostics show that source-box tokens were contained inside the
  rendered-mark mask for `31/32` rows, but exactly equal for only `5/32` rows.
  Rendered halo was empty for `5/32` rows. This means the probe can separate
  source-box from rendered-mark neighborhood on most rows, but still operates
  at Qwen merged visual-token granularity rather than precise pixel geometry.

Supported:

- The visual actuator is not purely the underlying source-object box. The
  rendered-mark neighborhood is stronger than source-box alone on this panel.
- The actuator is not purely local to the rendered mark either. `source_ring`
  and `non_mark` deltas both recover substantial source-object binding, which
  supports a distributed/contextual post-vision representation.
- Same-row row-specificity remains supported: shifted and wrong-row controls do
  not reproduce the source-binding recovery.
- The strongest safe interpretation is "mark-neighborhood plus distributed
  visual context", not "paint pixels alone" and not "object box alone".

Not supported yet:

- A learned non-pixel cursor.
- Source-image-disjoint generalization.
- Pixel-space causal locality before the vision tower.
- Production inference behavior.
- A claim that the rendered halo itself is the dominant actuator; it is
  meaningful but weaker than the full rendered-mark neighborhood in this panel.

Next decider:

- A source-image-disjoint variant: capture a same-class or same-description
  cursor-like visual feature on one image and test whether it can steer a
  different image. That is the next useful boundary between "same-image visual
  delta compression" and a potentially reusable cursor representation.

Promotion decision:

- Not promoted. This remains a same-row research probe under
  `scripts/probes/painted_gt` with local tests and artifact receipts only. No
  training, inference, config, or artifact contract should be updated from this
  unit alone.
