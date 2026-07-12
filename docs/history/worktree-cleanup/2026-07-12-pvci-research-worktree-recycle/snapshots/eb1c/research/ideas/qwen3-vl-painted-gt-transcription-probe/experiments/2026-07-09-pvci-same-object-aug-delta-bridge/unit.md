---
title: PVCI Same-Object Augmented Visual-Delta Bridge
description: Tests whether a painted-clean visual delta captured from a photometrically transformed view of the same row still steers the original clean image.
type: idea
role: research-unit
authority: non_normative_research
promotion_status: not_promoted
unit_id: 2026-07-09-pvci-same-object-aug-delta-bridge
topic: qwen3-vl-painted-gt-transcription-probe
status: complete
tags:
  - coordexp-swift
  - research-unit
  - painted-gt
  - pvci
  - visual-delta
  - augmentation
  - cursor
updated: 2026-07-09
---

# PVCI Same-Object Augmented Visual-Delta Bridge

## Research Question

The
[PVCI Cross-Image Visual-Delta Transfer Probe](../2026-07-09-pvci-cross-image-delta-transfer/unit.md)
failed to support reusable source-image-disjoint cross-image cursor transfer.
This bridge asks a narrower decider:

```text
If the visual delta is captured from a lightly transformed view of the same
row, can it still steer the original clean image?
```

This separates nuisance-view portability from the harder question of
cross-image or cross-instance transfer.

## Evidence Scope

- Checkout or branch: `/data/CoordExp/.codex/worktrees/eb1c/CoordExp`,
  `codex/qwen3-vl-painted-gt-transcription-probe`.
- Main checkpoint: E1 anti-copy checkpoint:
  `/data/CoordExp/outputs/painted_gt/pvci_step2/train_snap_radius/painted_gt_pvci_step2_snap_radius_train256_anti_copy_mix_step484_warm_start_dora_all_towers_accelerate8_ebs8_16epoch-pvci-step2-anticopy-retry3-20260708T1726Z/checkpoints/step-484`.
- Matched row panel:
  `/data/CoordExp/outputs/painted_gt/pvci_row_binding/materialized/heldout_val100_jitter_medium_wrong_object/conditions/stepwise__wrong_object_mark/stepwise.wrong_object_mark.examples.jsonl`.
- Main artifact root:
  `/data/CoordExp/outputs/painted_gt/pvci_same_object_aug_delta/e1_wrong_object_jitter_medium_val32`.
- Main comparison artifact:
  `/data/CoordExp/outputs/painted_gt/pvci_same_object_aug_delta/e1_wrong_object_jitter_medium_val32/analysis/pvci_same_object_aug_delta_comparison.json`.

## Completion Promise

This unit is complete only when it produces:

- original clean and painted feature-store deltas;
- same-row photometric augmented clean and painted feature-store deltas;
- clean, zero, original mark/full, and augmented mark/full identity-conflict
  summaries on the same decoded row slice;
- receipts proving the augmented conditions used the augmented feature store
  with no fallback zero deltas.

## Evidence Gate

Support for same-object nuisance portability requires:

- augmented mark or full deltas strongly reduce `target_row_rate` relative to
  `clean_no_cursor`;
- augmented mark or full deltas increase source-object or rendered-mark binding
  behavior relative to clean;
- fallback zero-delta count is zero;
- the result is interpreted only as same-object transformed-view portability,
  not cross-image generalization.

## Probe Design

The bridge uses deterministic photometric perturbations only:

- brightness, contrast, saturation, and sharpness are deterministically derived
  from `row_id`;
- image dimensions and bbox coordinates are preserved;
- Qwen visual feature signatures and local token masks are expected to remain
  aligned;
- decode still runs on the original clean image; only the stored source delta
  comes from the transformed clean/painted pair.

The existing CoordExp-Swift augmentation stack was not copied for this first
bridge because geometry-changing transforms would require a separate box/mask
alignment contract. This probe intentionally stays photometric and local.

## Research Unit Closeout

Observed:

- Debug artifact root:
  `/data/CoordExp/outputs/painted_gt/pvci_same_object_aug_delta/e1_wrong_object_jitter_medium_debug8`.
  The bridge smoke completed with separate original and augmented feature-store
  manifests. Augmented mark/full conditions each had `8/8` source-delta hits
  and `0` fallback zero deltas.
- `debug8` official identity-conflict rates:

  | Condition | Pred rate | Target row | Source-object row | Mark-copy row | Third-object row |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | `clean_no_cursor` | `0.5000` | `0.2500` | `0.0000` | `0.0000` | `0.2500` |
  | `clean_plus_zero_delta` | `0.5000` | `0.2500` | `0.0000` | `0.0000` | `0.2500` |
  | `clean_plus_mark_delta` | `0.8750` | `0.0000` | `0.5714` | `0.2857` | `0.0000` |
  | `clean_plus_same_object_aug_mark_delta` | `0.8750` | `0.0000` | `0.4286` | `0.4286` | `0.0000` |
  | `clean_plus_same_object_aug_full_delta` | `0.8750` | `0.0000` | `0.5714` | `0.2857` | `0.0000` |

- `val32` official identity-conflict rates:

  | Condition | Pred rate | Target row | Source-object row | Mark-copy row | Third-object row |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | `clean_no_cursor` | `0.5313` | `0.4706` | `0.0588` | `0.1176` | `0.1176` |
  | `clean_plus_zero_delta` | `0.5313` | `0.4706` | `0.0588` | `0.1176` | `0.1176` |
  | `clean_plus_mark_delta` | `0.8125` | `0.0000` | `0.4231` | `0.2308` | `0.0000` |
  | `clean_plus_stored_delta` | `0.8125` | `0.0000` | `0.5000` | `0.1154` | `0.0000` |
  | `clean_plus_same_object_aug_mark_delta` | `0.7500` | `0.0000` | `0.2917` | `0.3333` | `0.0000` |
  | `clean_plus_same_object_aug_full_delta` | `0.7813` | `0.0400` | `0.3600` | `0.2400` | `0.0000` |

- `val32` feature-store receipts:

  | Condition | Source-delta hits | Fallback zero |
  | --- | ---: | ---: |
  | `clean_plus_mark_delta` | `32` | `0` |
  | `clean_plus_stored_delta` | `32` | `0` |
  | `clean_plus_same_object_aug_mark_delta` | `32` | `0` |
  | `clean_plus_same_object_aug_full_delta` | `32` | `0` |

Supported:

- Same-object transformed-view visual deltas preserve strong actuator behavior.
  The transformed-source bridge still collapses target-row behavior and steers
  toward the wrong/painted object family.
- The actuator is not merely an exact pixel-cache artifact. It survives
  moderate deterministic photometric nuisance changes when row identity and
  geometry are preserved.
- This result reconciles with the cross-image unit: visual deltas are portable
  across same-object nuisance views, but source-image-disjoint transfer remains
  unsupported by the current post-vision tensor-addition probe.

Not supported yet:

- Cross-image cursor transfer.
- A learned cursor design.
- Robustness to geometry-changing transforms.
- A production inference mechanism.

Next decider:

- Localize the effective hook boundary or layer, because the same-object bridge
  suggests the actuator is meaningful while the cross-image probe shows it is
  not trivially reusable across images.
- If an augmentation bridge is extended, prefer one controlled geometric
  transform at a time with explicit bbox/mask remapping before copying a broad
  augmentation stack.

Promotion decision:

- Not promoted. This remains a research-only feature-intervention probe.
