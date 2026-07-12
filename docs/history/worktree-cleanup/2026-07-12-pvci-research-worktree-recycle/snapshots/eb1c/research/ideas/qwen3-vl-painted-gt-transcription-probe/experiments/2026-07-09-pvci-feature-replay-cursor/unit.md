---
title: PVCI Feature-Replay Cursor Probe
description: Tests whether the row-level painted visual actuator can be replayed after the vision encoder while feeding clean image bytes.
type: idea
role: research-unit
authority: non_normative_research
promotion_status: not_promoted
unit_id: 2026-07-09-pvci-feature-replay-cursor
topic: qwen3-vl-painted-gt-transcription-probe
status: complete
tags:
  - coordexp-swift
  - research-unit
  - painted-gt
  - pvci
  - feature-replay
  - non-pixel-cursor
updated: 2026-07-09
---

# PVCI Feature-Replay Cursor Probe

## Question

Can the row-level visual actuator observed in the PVCI corrupted wrong-object
probe be replayed without modifying the input image bytes?

The immediate target is not a learned hidden cursor, a selector, STOP policy, or
production detector. The target is a narrow feasibility gate:

```text
clean image bytes
+ teacher-prefix row prompt
+ visual features replayed from the matched painted image
-> row-level binding behavior similar to the pixel-painted E1 anchor
```

If feature replay cannot reproduce the pixel-painted anchor, then downstream
non-pixel cursor work is not trustworthy yet. If it can, then later work may
try to generate or learn such a feature delta without rendering paint into the
input pixels.

## Evidence Scope

- Checkout or branch: `/data/CoordExp/.codex/worktrees/eb1c/CoordExp`,
  `codex/qwen3-vl-painted-gt-transcription-probe`.
- Parent evidence:
  [2026-07-09 PVCI Corrupted Wrong-Object Row Binding](../2026-07-09-pvci-corrupted-wrong-object-row-binding/unit.md).
- Main checkpoint: E1 anti-copy checkpoint:
  `/data/CoordExp/outputs/painted_gt/pvci_step2/train_snap_radius/painted_gt_pvci_step2_snap_radius_train256_anti_copy_mix_step484_warm_start_dora_all_towers_accelerate8_ebs8_16epoch-pvci-step2-anticopy-retry3-20260708T1726Z/checkpoints/step-484`.
- Matched row panel:
  `/data/CoordExp/outputs/painted_gt/pvci_row_binding/materialized/heldout_val100_jitter_medium_wrong_object/conditions/stepwise__wrong_object_mark/stepwise.wrong_object_mark.examples.jsonl`.
- Pixel-painted anchor:
  `/data/CoordExp/outputs/painted_gt/pvci_row_binding/inference/e1_wrong_object_val100_jitter_medium/step484_anticopy_rp110_bs2`.
- Primary readout split: clean primary rows with different target/source
  descriptions. Same-description rows are geometry-only and must not define
  full phrase-binding claims.
- Intended first execution scope: small `val32` style subset before any full
  `825` row run.

## Procedure

1. Select rows from the corrupted wrong-object panel where:
   - `primary_credit_eligible` is true;
   - target `i` and source `j` descriptions differ;
   - `candidate_objects[]` is present;
   - the matched clean source image exists and has the same dimensions.
2. Materialize two matched inference inputs:
   - `clean_no_cursor`: clean source image bytes, no visual feature replay;
   - `feature_replay_painted`: clean source image bytes, but Qwen
     `get_image_features` returns the visual features computed from the
     matched painted row image.
3. Keep prompts, teacher prefix, generation settings, adapter, embedding delta,
   row ids, and candidate-object metadata matched to the E1 row-binding anchor.
4. Run deterministic HF generation with `temperature=0.0`, `top_p=1.0`,
   `repetition_penalty=1.10`, and `max_new_tokens=96`.
5. Analyze both conditions with the row-binding analyzer against target `i`,
   source object `j`, rendered/cursor geometry, and best third object.

Feature replay is a research-only inference hook. It bypasses the normal stable
inference contract and therefore must not be promoted to config/runtime docs
without a separate compatibility decision.

## Required Diagnostics

- Clean image bytes must be recorded and must not be modified for non-pixel
  conditions.
- The feature source image path and hash must be recorded separately from the
  clean input image path and hash.
- The hook must record how many times `get_image_features` was intercepted per
  batch.
- Analyzer summaries must include:
  - prediction/parse rate;
  - phrase identity;
  - geometry identity by IoU and L1;
  - source-object row rate;
  - rendered/cursor-copy rate;
  - target-row rate;
  - third-object row rate;
  - chimera rate;
  - split by same-description vs different-description.

## Interpretation Rules

Supported only if observed:

- `feature_replay_painted` moves substantially closer to the E1 pixel-painted
  anchor than `clean_no_cursor`, especially on different-description rows.
- The input artifact proves clean image bytes were used for
  `feature_replay_painted`.
- The hook replays both main image features and deepstack visual features.

Not supported by this unit:

- A learned internal cursor exists.
- A non-pixel cursor can be generated from the raw image alone.
- E1 is a detector-ready checkpoint.
- STOP, coverage, self-prefix, or autonomous object selection is solved.

## Research Unit Closeout

Observed:

- A tiny debug run over 4 different-description primary rows completed with
  the research-only hook active:
  `/data/CoordExp/outputs/painted_gt/pvci_feature_replay/e1_wrong_object_jitter_medium_debug4`.
  The feature-replay condition recorded 2 batches, 2 total hook calls, and
  replayed both main image embeddings and the three deepstack visual feature
  tensors.
- The main bounded run completed over 32 different-description primary rows:
  `/data/CoordExp/outputs/painted_gt/pvci_feature_replay/e1_wrong_object_jitter_medium_val32`.
- The feature-replay condition used clean input image files and separate
  painted feature-source files. The selection receipt records clean image
  SHA-256 values and painted feature-source SHA-256 values separately:
  `/data/CoordExp/outputs/painted_gt/pvci_feature_replay/e1_wrong_object_jitter_medium_val32/inputs/selection.json`.
- The feature-replay condition recorded 16 batches and 16 hook calls at
  `Qwen3VLModel.get_image_features`. The first receipt showed replayed
  `image_embeds` shapes `[[1014, 2048], [1014, 2048]]` and deepstack shapes
  `[[2028, 2048], [2028, 2048], [2028, 2048]]`.
- On the 32-row split, clean/no-cursor produced predictions for 17/32 rows
  (`prediction_rate=0.53125`) and mostly retained the scheduled target prior:
  `target_row_rate=0.4706`, `source_object_row_rate=0.0588`.
- On the same 32 rows, feature replay produced predictions for 26/32 rows
  (`prediction_rate=0.8125`) and matched the same-row pixel-painted anchor at
  analyzer level: `source_object_row_rate=0.4615`,
  `rendered_mark_copy_rate=0.1538`, `target_row_rate=0.0`,
  `chimera_row_rate=0.0769`.
- Row-wise comparison against the exact 32-row pixel-painted E1 anchor had
  30/32 identical `{raw_decode_text, pred}` rows. The 2 differing rows were
  small coordinate-token nudges, not outcome-category changes.
- Compact comparison:
  `/data/CoordExp/outputs/painted_gt/pvci_feature_replay/e1_wrong_object_jitter_medium_val32/feature_replay_comparison.json`.
- Analyzer outputs:
  `/data/CoordExp/outputs/painted_gt/pvci_feature_replay/e1_wrong_object_jitter_medium_val32/clean_no_cursor/row_binding/identity_conflict.json`,
  `/data/CoordExp/outputs/painted_gt/pvci_feature_replay/e1_wrong_object_jitter_medium_val32/feature_replay_painted/row_binding/identity_conflict.json`,
  `/data/CoordExp/outputs/painted_gt/pvci_feature_replay/e1_wrong_object_jitter_medium_val32/pixel_anchor_subset/row_binding/identity_conflict.json`.

Supported:

- The visible pixel paint effect can be replayed almost completely at the
  Qwen3-VL visual feature boundary for this row-level teacher-prefix panel.
- The decisive actuator for this probe is already present after
  `Qwen3VLModel.get_image_features`: replacing clean-image visual features with
  matched painted-image visual features is enough to recover the pixel-painted
  E1 row-binding behavior.
- Clean image bytes alone are not sufficient for this panel: without the replay
  cursor, the model often returns to the scheduled target prior or emits no
  usable row.

Not supported yet:

- This does not prove that a learned non-pixel cursor exists.
- This does not prove that a cursor can be generated from the raw image alone.
- This does not prove that E1 is detector-ready or that STOP, coverage,
  self-prefix, or autonomous selection is solved.
- This does not distinguish whether a future learned cursor should operate as
  a region-boundary delta, object-identity delta, pseudo-visual token, or
  hidden-state intervention.

Next decider:

- Test whether the replayed feature effect can be compressed into an explicit
  non-pixel intervention that does not require running the painted image through
  the vision tower. The cheapest next version should start with a stored
  feature delta or low-rank visual-feature delta on the same row-binding panel,
  with offset/background controls, before any larger learned selector or
  production rollout work.

Promotion decision:

- Keep as non-normative research. Do not promote to docs/OpenSpec yet, because
  the current hook is a temporary research probe and not a stable inference
  contract.
