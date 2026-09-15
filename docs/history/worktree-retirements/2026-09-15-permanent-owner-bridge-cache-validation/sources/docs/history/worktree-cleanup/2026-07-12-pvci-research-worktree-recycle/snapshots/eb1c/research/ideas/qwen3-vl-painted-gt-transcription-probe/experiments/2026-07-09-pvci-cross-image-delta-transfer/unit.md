---
title: PVCI Cross-Image Visual-Delta Transfer Probe
description: Tests whether a painted-clean visual delta captured from one image can steer a different clean image when source and target share category and Qwen visual-feature shape.
type: idea
role: research-unit
authority: non_normative_research
promotion_status: not_promoted
unit_id: 2026-07-09-pvci-cross-image-delta-transfer
topic: qwen3-vl-painted-gt-transcription-probe
status: complete
tags:
  - coordexp-swift
  - research-unit
  - painted-gt
  - pvci
  - visual-delta
  - cross-image-transfer
  - cursor
updated: 2026-07-09
---

# PVCI Cross-Image Visual-Delta Transfer Probe

## Research Question

The completed
[PVCI Region-Split Visual-Delta Cursor Probe](../2026-07-09-pvci-region-split-visual-delta-cursor/unit.md)
showed that same-row painted-clean post-vision deltas strongly steer row
identity, with the strongest safe interpretation being mark-neighborhood plus
distributed visual context. This unit asks the next boundary question:

```text
Can a visual delta captured from image A steer a different clean image B,
or is the actuator only a same-image compression artifact?
```

This is not a learned cursor and not a production inference mode. It is a
research-only post-vision feature intervention through
`Qwen3VLModel.get_image_features`.

## Evidence Scope

- Checkout or branch: `/data/CoordExp/.codex/worktrees/eb1c/CoordExp`,
  `codex/qwen3-vl-painted-gt-transcription-probe`.
- Parent unit:
  [2026-07-09 PVCI Region-Split Visual-Delta Cursor](../2026-07-09-pvci-region-split-visual-delta-cursor/unit.md).
- Main checkpoint: E1 anti-copy checkpoint:
  `/data/CoordExp/outputs/painted_gt/pvci_step2/train_snap_radius/painted_gt_pvci_step2_snap_radius_train256_anti_copy_mix_step484_warm_start_dora_all_towers_accelerate8_ebs8_16epoch-pvci-step2-anticopy-retry3-20260708T1726Z/checkpoints/step-484`.
- Matched row panel:
  `/data/CoordExp/outputs/painted_gt/pvci_row_binding/materialized/heldout_val100_jitter_medium_wrong_object/conditions/stepwise__wrong_object_mark/stepwise.wrong_object_mark.examples.jsonl`.
- Initial scope: debug rows for artifact validation, then the same `val32`
  `different_description` panel if pairing and fallback counts are sane.

## Completion Promise

This unit is complete only when it produces:

- a feature-store artifact with explicit cross-image pair receipts;
- identity-conflict summaries for clean, same-row positive ceiling,
  same-class cross-image transfer, wrong-class control, random control, and
  zero-delta control;
- fallback counts proving whether missing pairs could explain a null result;
- a closeout that separates same-image actuator evidence from reusable
  cross-image cursor evidence.

## Evidence Gate

Support for reusable cross-image transfer requires:

- `clean_plus_cross_image_same_class_mark_delta` improves target-image matched
  row behavior over `clean_no_cursor`;
- wrong-class and random mark-delta controls do not show the same improvement;
- matched-source fallback count is low enough that the result is not mostly a
  zero-delta run.

If this gate fails, the correct conclusion is not that the visual actuator is
false. The supported conclusion is narrower: reusable cross-image cursor
transfer is unsupported by this probe, and the next decider should move to
layer or boundary localization.

## Pairing Policy

Cross-image sources are selected inside the stored feature-store panel:

- source and target rows must have identical Qwen visual feature signatures;
- source and target rows must come from different `source_example_id` images;
- same-class transfer matches target row category to source row marked-object
  category and chooses the nearest source mark geometry;
- wrong-class transfer chooses the nearest different-category source mark;
- random transfer uses a stable hash-ranked different-image source within the
  same feature signature group.

This intentionally keeps the first probe conservative. The existing
CoordExp-Swift geometry-flip augmentation pipeline may become useful for a
later same-object-transformed-image bridge probe, but it is not part of this
cross-image transfer unit.

## Condition Matrix

Core controls:

- `clean_no_cursor`: clean image bytes, no feature intervention.
- `clean_plus_zero_delta`: hook-path neutral control.
- `clean_plus_mark_delta`: same-row mark-local positive ceiling.
- `clean_plus_stored_delta`: same-row full-delta positive ceiling.

Cross-image transfer:

- `clean_plus_cross_image_same_class_mark_delta`: main condition.
- `clean_plus_cross_image_same_class_full_delta`: diagnostic upper bound, not
  the primary verdict.
- `clean_plus_cross_image_wrong_class_mark_delta`: semantic negative control.
- `clean_plus_cross_image_random_mark_delta`: generic transfer/noise control.

## Probe Run

- Scope: `debug16` first, then `val32` with a larger source pool after
  `debug16` showed same-class source scarcity.
- Baseline/comparison: matched clean and same-row positive-ceiling conditions.
- Stop condition: cross-image same-class mark delta either shows controlled lift
  over clean or fails under adequate pair coverage.

## Research Unit Closeout

Observed:

- Debug artifact root:
  `/data/CoordExp/outputs/painted_gt/pvci_cross_image_delta/e1_wrong_object_jitter_medium_debug16_pool96`.
  Same-class transfer coverage was too low for verdict: only `2/16` decoded
  rows had matched same-class cross-image sources. Wrong-class and random
  controls each matched `16/16`, proving the feature-store hook worked but the
  exact same-class source pool was sparse.
- Main artifact root:
  `/data/CoordExp/outputs/painted_gt/pvci_cross_image_delta/e1_wrong_object_jitter_medium_val32_pool512`.
  The requested `--pair-pool-rows 512` yielded `182` eligible feature-store
  rows after the upstream split/filter, with `32` decoded target rows.
- Main comparison artifact:
  `/data/CoordExp/outputs/painted_gt/pvci_cross_image_delta/e1_wrong_object_jitter_medium_val32_pool512/analysis/pvci_cross_image_delta_comparison.json`.
- `val32` official identity-conflict rates:

  | Condition | Pred rate | Target row | Source-object row | Mark-copy row | Third-object row |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | `clean_no_cursor` | `0.5313` | `0.4706` | `0.0588` | `0.1176` | `0.1176` |
  | `clean_plus_zero_delta` | `0.5313` | `0.4706` | `0.0588` | `0.1176` | `0.1176` |
  | `clean_plus_mark_delta` | `0.8125` | `0.0000` | `0.4231` | `0.2308` | `0.0000` |
  | `clean_plus_stored_delta` | `0.8125` | `0.0000` | `0.5000` | `0.1154` | `0.0000` |
  | `clean_plus_cross_image_same_class_mark_delta` | `0.5938` | `0.4211` | `0.0526` | `0.1053` | `0.1053` |
  | `clean_plus_cross_image_same_class_full_delta` | `0.4688` | `0.5333` | `0.0000` | `0.0000` | `0.0667` |
  | `clean_plus_cross_image_wrong_class_mark_delta` | `0.5938` | `0.5263` | `0.0526` | `0.1053` | `0.0526` |
  | `clean_plus_cross_image_random_mark_delta` | `0.6563` | `0.3810` | `0.0476` | `0.0476` | `0.2381` |

- `val32` cross-image source coverage:

  | Condition | Matched | Fallback zero | Category matches |
  | --- | ---: | ---: | ---: |
  | `same_class_mark_delta` | `14` | `18` | `14` |
  | `same_class_full_delta` | `14` | `18` | `14` |
  | `wrong_class_mark_delta` | `28` | `4` | `0` |
  | `random_mark_delta` | `28` | `4` | `1` |

- Matched-only official summaries did not rescue the main claim. For
  same-class mark transfer, the `14` matched rows had `0.6429` prediction rate,
  `0.2222` target-row rate, `0.1111` source-object-row rate, `0.2222`
  rendered-mark-copy rate, and `0.1111` third-object-row rate. The same-row
  positive ceilings remained much stronger, with source-object-row rates of
  `0.4231` for mark-local deltas and `0.5000` for full stored deltas.

Supported:

- Same-row post-vision visual deltas remain a strong causal actuator for this
  wrong-object row-binding panel.
- This probe does not show reusable cross-image cursor transfer. Cross-image
  deltas mostly behave like weak perturbations or retain the clean target
  behavior instead of reproducing the same-row wrong-object steering.
- Exact Qwen visual-feature signature matching is a real coverage limitation
  for cross-image tensor addition. The null result should be interpreted as
  unsupported transfer under this conservative hook, not as proof that no
  learned visual cursor can exist.

Not supported yet:

- Reusable cross-image cursor transfer.
- A learned cursor.
- Source-image-disjoint generalization.
- Production inference behavior.
- A conclusion that the visual actuator is merely an artifact; same-row
  positive-ceiling evidence still supports the actuator, only not this
  cross-image reuse.

Next decider:

- If portability remains important, run a same-object transformed-image bridge
  before copying a broad augmentation stack: keep image dimensions and Qwen
  feature signature fixed, apply small photometric or render-only perturbations,
  capture the painted-clean delta on the transformed view, and apply it to the
  original clean view. This tests whether deltas survive controlled image
  nuisance changes before the harder cross-image/category-transfer question.
- If the bridge also fails, move to hook-boundary/layer localization rather
  than more cross-image pairing.

Promotion decision:

- Not promoted. This remains a research-only feature-intervention probe.
