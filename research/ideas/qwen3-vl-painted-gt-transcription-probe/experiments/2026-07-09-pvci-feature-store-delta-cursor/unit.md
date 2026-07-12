---
title: PVCI Feature-Store Delta Cursor Probe
description: Tests whether the full painted-feature replay effect can be reproduced from stored post-vision features or stored painted-clean deltas during clean-image generation.
type: idea
role: research-unit
authority: non_normative_research
promotion_status: not_promoted
unit_id: 2026-07-09-pvci-feature-store-delta-cursor
topic: qwen3-vl-painted-gt-transcription-probe
status: complete
tags:
  - coordexp-swift
  - research-unit
  - painted-gt
  - pvci
  - feature-store
  - visual-delta
  - non-pixel-cursor
updated: 2026-07-09
---

# PVCI Feature-Store Delta Cursor Probe

## Question

Can the full painted-feature replay effect be reproduced during generation
without running the painted image through the vision tower at inference time?

This unit follows the completed
[PVCI Feature-Replay Cursor Probe](../2026-07-09-pvci-feature-replay-cursor/unit.md).
That prior unit showed that clean input image bytes plus visual features
computed from the matched painted image reproduce the pixel-painted E1 anchor
nearly exactly on a 32-row different-description panel.

This unit asks a narrower follow-up:

```text
capture phase:
  clean image -> Qwen vision tower -> clean visual features
  painted image -> Qwen vision tower -> painted visual features
  store painted features and painted-clean deltas

generation phase:
  clean image bytes only
  + stored feature intervention at Qwen3VLModel.get_image_features
  -> row-level binding behavior
```

The first conditions are same-row oracle compression controls. They are useful
because they test feature-store mechanics and additive-delta equivalence, but
they are not a learned cursor and are not a generalizable non-pixel detector.

## Evidence Scope

- Checkout or branch: `/data/CoordExp/.codex/worktrees/eb1c/CoordExp`,
  `codex/qwen3-vl-painted-gt-transcription-probe`.
- Parent unit:
  [2026-07-09 PVCI Feature-Replay Cursor Probe](../2026-07-09-pvci-feature-replay-cursor/unit.md).
- Parent feature-replay artifact root:
  `/data/CoordExp/outputs/painted_gt/pvci_feature_replay/e1_wrong_object_jitter_medium_val32`.
- Main checkpoint: E1 anti-copy checkpoint:
  `/data/CoordExp/outputs/painted_gt/pvci_step2/train_snap_radius/painted_gt_pvci_step2_snap_radius_train256_anti_copy_mix_step484_warm_start_dora_all_towers_accelerate8_ebs8_16epoch-pvci-step2-anticopy-retry3-20260708T1726Z/checkpoints/step-484`.
- Matched row panel:
  `/data/CoordExp/outputs/painted_gt/pvci_row_binding/materialized/heldout_val100_jitter_medium_wrong_object/conditions/stepwise__wrong_object_mark/stepwise.wrong_object_mark.examples.jsonl`.
- Intended first execution scope: the same `val32` different-description
  primary rows used by the parent feature-replay unit.

## Condition Matrix

Primary conditions:

- `clean_no_cursor`: clean image bytes, no feature intervention.
- `stored_full_feature`: clean image bytes during generation, but the hook
  returns stored painted visual features captured earlier from the same row.
- `clean_plus_stored_delta`: clean image bytes during generation, clean visual
  features computed live, then same-row stored `(painted - clean)` deltas added
  to main image embeddings and all deepstack visual features.
- `clean_plus_zero_delta`: patch-path control; clean visual features computed
  live, then zero deltas added.
- `clean_plus_wrong_row_delta`: wrong-row oracle control; clean visual features
  computed live, then a stored delta from another row with matching feature
  shape added.

Interpretation labels:

- `stored_full_feature` and `clean_plus_stored_delta` are same-row oracle
  compression controls because their feature artifacts are derived from eval-row
  painted images in the capture phase.
- `clean_plus_wrong_row_delta` is a perturbation/control condition, not a
  proposed method.
- A future learned or held-out approximation condition must use a
  source-image-disjoint fit/eval split and must not consume eval-row painted
  features, rendered mark boxes, target/source ids, candidate objects, anchor
  outputs, or analyzer labels.

## Required Diagnostics

- Generation-phase receipts must prove:
  - `painted_vision_forward_count_eval=0`;
  - no replay `pixel_values` from painted images are collated during generation;
  - every request hits the feature store;
  - clean-image paths and hashes are recorded separately from any painted
    capture-source paths and hashes.
- Feature-store manifest must record:
  - producing script path and git commit;
  - command arguments;
  - config, model, adapter, embedding-delta, processor, tokenizer, and decode
    identities;
  - row ids and source image ids;
  - clean and painted capture-source hashes;
  - `image_grid_thw`, split sizes, tensor shapes, dtypes, and tensor-file hash;
  - condition matrix and claim labels.
- Intervention receipts must record:
  - hook boundary;
  - intervention mode;
  - clean vision forward count;
  - painted vision forward count during generation;
  - tensor shape parity;
  - delta norms or norm ratios by visual stream when applicable.
- Analyzer summaries must compare against:
  - clean/no-cursor;
  - parent full feature replay;
  - same-row pixel-painted E1 anchor subset.

## Interpretation Rules

Supported only if observed:

- `stored_full_feature` reproduces the parent feature-replay or pixel anchor
  outcome categories on the same rows while recording
  `painted_vision_forward_count_eval=0`.
- `clean_plus_stored_delta` closely matches `stored_full_feature` outcome
  categories. Exact token equality may tolerate small coordinate-token nudges.
- `clean_plus_zero_delta` behaves like `clean_no_cursor`.
- `clean_plus_wrong_row_delta` does not behave like the same-row intervention.

Not supported by this unit:

- A learned non-pixel cursor exists.
- A cursor can be generated from clean images alone.
- A source-image-disjoint approximation works.
- E1 is detector-ready.
- STOP, coverage, self-prefix, or autonomous object selection is solved.

## Research Unit Closeout

Observed:

- A 4-row smoke run completed across all five conditions:
  `/data/CoordExp/outputs/painted_gt/pvci_feature_store_delta/e1_wrong_object_jitter_medium_debug4`.
  It verified the feature-store receipts and showed the intended control
  ordering before scaling.
- The main 32-row different-description primary run completed:
  `/data/CoordExp/outputs/painted_gt/pvci_feature_store_delta/e1_wrong_object_jitter_medium_val32`.
- Feature-store manifest:
  `/data/CoordExp/outputs/painted_gt/pvci_feature_store_delta/e1_wrong_object_jitter_medium_val32/feature_store/visual_feature_store_manifest.json`.
  The tensor artifact SHA-256 is
  `010ccb6dd233d98238d47a0df5dfef962b0049d1c1f9e8550986e4c41e65c8a6`.
- The capture phase ran 32 clean and 32 painted vision forwards. Generation
  receipts recorded `painted_vision_forward_count_eval=0` for all intervention
  conditions.
- `stored_full_feature` recorded 16 hook calls, 32 feature-store hits, 0 clean
  vision forwards, and 0 painted eval vision forwards.
- `clean_plus_stored_delta`, `clean_plus_zero_delta`, and
  `clean_plus_wrong_row_delta` each recorded 16 clean vision forwards, 32
  feature-store hits, and 0 painted eval vision forwards. The wrong-row
  condition had 1 zero-delta fallback because one row lacked a matching-shape
  wrong-row partner.
- Compact comparison:
  `/data/CoordExp/outputs/painted_gt/pvci_feature_store_delta/e1_wrong_object_jitter_medium_val32/feature_store_delta_comparison.json`.
- Main 32-row rates:
  - `clean_no_cursor`: `prediction_rate=0.53125`,
    `source_object_row_rate=0.0588`, `target_row_rate=0.4706`.
  - `stored_full_feature`: `prediction_rate=0.8125`,
    `source_object_row_rate=0.5000`, `rendered_mark_copy_rate=0.1154`,
    `target_row_rate=0.0`.
  - `clean_plus_stored_delta`: same outcome metrics as
    `stored_full_feature`.
  - `clean_plus_zero_delta`: same outcome metrics and 32/32 exact raw/pred rows
    as `clean_no_cursor`.
  - `clean_plus_wrong_row_delta`: `prediction_rate=0.5625`,
    `source_object_row_rate=0.0556`, `target_row_rate=0.2778`,
    `third_object_row_rate=0.2778`.
- Exact agreement:
  - `stored_full_feature` and `clean_plus_stored_delta`: 23/32 exact raw/pred
    rows but 32/32 identical row-binding outcome categories.
  - Parent full feature replay and `stored_full_feature`: 22/32 exact raw/pred
    rows and 31/32 identical row-binding outcome categories.
  - `clean_no_cursor` and `clean_plus_zero_delta`: 32/32 exact raw/pred rows.

Supported:

- Same-row stored full visual features are sufficient to reproduce the parent
  full feature-replay effect during generation without running the painted
  image through the vision tower at eval/generation time.
- The same-row additive `(painted - clean)` visual-feature delta is sufficient
  to reproduce the stored-full outcome categories while keeping clean visual
  features live during generation.
- The zero-delta patch path is neutral: it exactly matches `clean_no_cursor`.
- Wrong-row deltas do not mimic the same-row intervention on this panel,
  supporting that the stored same-row effect is not merely a generic visual
  perturbation.

Not supported yet:

- A learned non-pixel cursor exists.
- A source-image-disjoint approximation works.
- A cursor can be generated from clean image content alone.
- The same-row delta is local to the marked region.
- Any production detector, STOP, coverage, self-prefix, or autonomous
  selection behavior is improved.

Next decider:

- Localize the same-row delta. The next cheapest probe should apply stored
  deltas only to merged visual tokens overlapping the rendered mark/source
  region, with non-mark, offset/background, and shuffled-row controls. If local
  deltas recover the stored-full behavior while controls do not, a compact
  region-cursor target becomes plausible. If only full-image deltas work, the
  actuator remains global or distributed.

Promotion decision:

- Keep as non-normative research. The current implementation is a probe-local
  same-row oracle compression harness and must not be promoted to a stable
  inference contract.
