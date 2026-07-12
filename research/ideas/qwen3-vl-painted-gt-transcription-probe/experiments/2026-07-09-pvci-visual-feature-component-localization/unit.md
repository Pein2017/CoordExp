---
title: PVCI Visual-Feature Component Localization
description: Splits the stored visual delta into image-embed and deepstack components to identify which Qwen3-VL post-vision return channel carries the row-binding actuator.
type: idea
role: research-unit
authority: non_normative_research
promotion_status: not_promoted
unit_id: 2026-07-09-pvci-visual-feature-component-localization
topic: qwen3-vl-painted-gt-transcription-probe
status: complete
tags:
  - coordexp-swift
  - research-unit
  - painted-gt
  - pvci
  - visual-delta
  - component-localization
  - qwen3-vl
updated: 2026-07-09
---

# PVCI Visual-Feature Component Localization

## Question

The same-row and same-object augmented visual deltas can steer decoding away
from the clean target row and toward the painted or wrong-object row family.
This unit asks where that actuator lives inside the current Qwen3-VL
`get_image_features` hook payload:

- the primary image embedding tuple consumed by token replacement;
- all deepstack image embeddings together;
- one specific deepstack layer.

The first pass intentionally does not inspect the visual tower internals or LLM
hidden states. It only splits the already effective post-vision return payload.

## Completion Promise

This unit is complete when:

- Evidence gate: at least `debug8`, and preferably `val32`, compares clean,
  zero-delta, full stored-delta, mark-delta, image-only, deepstack-all, and
  deepstack-layer-only conditions on the same wrong-object row-binding panel.
- Acceptable evidence: identity-conflict summaries report `prediction_rate`,
  `target_row_rate`, `source_object_row_rate`, `rendered_mark_copy_rate`, and
  `third_object_row_rate`, with feature-store receipts showing zero fallback
  deltas for active component conditions.
- Insufficient evidence: a component is called causal only because it changes
  raw output text without reducing target-row binding or increasing
  source-object or rendered-mark binding relative to clean and zero controls.

## Evidence Scope

- Checkout or branch: `/data/CoordExp/.codex/worktrees/eb1c/CoordExp`,
  `codex/qwen3-vl-painted-gt-transcription-probe`.
- Commit or diff scope: local diff adding component-localization conditions to
  `scripts/probes/painted_gt/run_feature_store_delta_probe.py`.
- Config: existing PVCI feature-store delta probe defaults unless overridden by
  the commands below.
- Checkpoint or model version: E1 anti-copy checkpoint:
  `/data/CoordExp/outputs/painted_gt/pvci_step2/train_snap_radius/painted_gt_pvci_step2_snap_radius_train256_anti_copy_mix_step484_warm_start_dora_all_towers_accelerate8_ebs8_16epoch-pvci-step2-anticopy-retry3-20260708T1726Z/checkpoints/step-484`.
- Artifact root:
  `/data/CoordExp/outputs/painted_gt/pvci_feature_component_localization/`.
- Commands:
  - `pytest tests/painted_gt/test_visual_feature_store_probe.py -q`
  - `python -m py_compile scripts/probes/painted_gt/run_feature_store_delta_probe.py`
  - `python scripts/probes/painted_gt/run_feature_store_delta_probe.py --max-rows 8 --conditions clean_no_cursor,clean_plus_zero_delta,clean_plus_mark_delta,clean_plus_stored_delta,clean_plus_image_only_delta,clean_plus_deepstack_all_only_delta,clean_plus_deepstack_0_only_delta,clean_plus_deepstack_1_only_delta,clean_plus_deepstack_2_only_delta --output-root /data/CoordExp/outputs/painted_gt/pvci_feature_component_localization/e1_wrong_object_jitter_medium_debug8 --force`
- Metrics or counters: identity-conflict summary rates plus
  `feature_store_receipts[*].fallback_zero_delta_count`.
- Sample window: first `debug8`; escalate to `val32` if the debug run is sane.
- Known limitations: same-row source deltas remain oracle interventions; this
  is not a learned cursor, cross-image transfer proof, or production inference
  mechanism.

## Procedure

1. Add component-only feature-store intervention modes without changing the
   existing wrong-row, cross-image, local-mask, or same-object augmented modes.
2. Keep source selection unchanged: component-only modes use the same row entry
   as the existing stored-delta oracle.
3. Apply full deltas for this first split:
   `image_only`, `deepstack_all_only`, and `deepstack_{0,1,2}_only`.
4. Run `debug8`; inspect identity-conflict summaries and fallback counts.
5. If a component clearly carries or partly carries the actuator, repeat the key
   winning or ambiguous conditions on `val32`.
6. If original-store evidence is ambiguous, decide whether to add same-object
   augmented component conditions in a separate unit or patch.

## Observations

- Direct observation: `debug8` and `val32` both identify the primary image
  embedding delta as the dominant row-binding actuator at this hook boundary.
  Deepstack-only deltas do not reproduce the steering behavior.
- Counterexample or negative result: `clean_plus_deepstack_all_only_delta`,
  `clean_plus_deepstack_0_only_delta`, and
  `clean_plus_deepstack_1_only_delta` behave like clean/zero controls on both
  `debug8` and `val32`; layer 2 slightly changes prediction rate but does not
  increase source-object or rendered-mark binding.
- Artifact handle:
  `/data/CoordExp/outputs/painted_gt/pvci_feature_component_localization/e1_wrong_object_jitter_medium_debug8`
  and
  `/data/CoordExp/outputs/painted_gt/pvci_feature_component_localization/e1_wrong_object_jitter_medium_val32`.

## Interpretation

- Supported reading: for the current Qwen3-VL `get_image_features` seam and
  same-row oracle delta, the effective visual steering signal is localized
  primarily in the returned image embedding tuple that is later scattered into
  the language token stream.
- Alternative reading: deepstack features may still matter during ordinary
  model computation, but additive painted-clean deltas at the returned
  deepstack channels do not act as the main row-binding control variable in
  this probe.
- Remaining uncertainty: this does not localize the causal mechanism inside the
  visual tower, the text tower, or the scatter/replacement path, and it does not
  prove a learned cursor or cross-image reusable actuator.

## Research Unit Closeout

Observed:

- `debug8` identity-conflict rates:

  | Condition | Pred rate | Target row | Source-object row | Mark-copy row | Third-object row |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | `clean_no_cursor` | `0.5000` | `0.2500` | `0.0000` | `0.0000` | `0.2500` |
  | `clean_plus_zero_delta` | `0.5000` | `0.2500` | `0.0000` | `0.0000` | `0.2500` |
  | `clean_plus_mark_delta` | `0.8750` | `0.0000` | `0.5714` | `0.2857` | `0.0000` |
  | `clean_plus_stored_delta` | `0.8750` | `0.0000` | `0.7143` | `0.1429` | `0.0000` |
  | `clean_plus_image_only_delta` | `0.8750` | `0.0000` | `0.5714` | `0.2857` | `0.0000` |
  | `clean_plus_deepstack_all_only_delta` | `0.5000` | `0.2500` | `0.0000` | `0.0000` | `0.2500` |
  | `clean_plus_deepstack_0_only_delta` | `0.5000` | `0.2500` | `0.0000` | `0.0000` | `0.2500` |
  | `clean_plus_deepstack_1_only_delta` | `0.5000` | `0.2500` | `0.0000` | `0.0000` | `0.2500` |
  | `clean_plus_deepstack_2_only_delta` | `0.6250` | `0.2000` | `0.0000` | `0.0000` | `0.4000` |

- `val32` identity-conflict rates:

  | Condition | Pred rate | Target row | Source-object row | Mark-copy row | Third-object row |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | `clean_no_cursor` | `0.5312` | `0.4706` | `0.0588` | `0.1176` | `0.1176` |
  | `clean_plus_zero_delta` | `0.5312` | `0.4706` | `0.0588` | `0.1176` | `0.1176` |
  | `clean_plus_mark_delta` | `0.8125` | `0.0000` | `0.4231` | `0.2308` | `0.0000` |
  | `clean_plus_stored_delta` | `0.8125` | `0.0000` | `0.5000` | `0.1154` | `0.0000` |
  | `clean_plus_image_only_delta` | `0.8125` | `0.0385` | `0.3846` | `0.2308` | `0.0000` |
  | `clean_plus_deepstack_all_only_delta` | `0.5625` | `0.5000` | `0.0556` | `0.1111` | `0.1111` |
  | `clean_plus_deepstack_0_only_delta` | `0.5625` | `0.5000` | `0.0556` | `0.1111` | `0.1111` |
  | `clean_plus_deepstack_1_only_delta` | `0.5312` | `0.4706` | `0.0588` | `0.1176` | `0.1176` |
  | `clean_plus_deepstack_2_only_delta` | `0.5938` | `0.4737` | `0.0526` | `0.1053` | `0.1579` |

- Feature-store receipts for all active component modes showed zero fallback
  deltas and `deepstack_layer_count: 3`. Component receipt summaries were:
  image-only `image_delta_enabled: true, deepstack_delta_layer_indices: []`;
  deepstack-all `image_delta_enabled: false, deepstack_delta_layer_indices:
  [0, 1, 2]`; layer-only modes enabled exactly their requested layer.

Evidence gate:

- Satisfied for the first hook-boundary localization question. `debug8` and
  `val32` both include clean, zero, full, mark, image-only, deepstack-all, and
  layer-only conditions on the same wrong-object panel, with official
  identity-conflict summaries and zero component fallback deltas.

Supported:

- The dominant additive actuator at this post-vision seam is in the primary
  image embeddings, not the deepstack return channels.
- The image-only delta is sufficient to nearly reproduce mark/full steering:
  on `val32`, it collapses target-row rate from `0.4706` to `0.0385` and raises
  source-object plus rendered-mark binding to `0.6154`.
- Deepstack-only deltas are not sufficient for the actuator in this setup:
  deepstack-all leaves target-row rate at `0.5000`, close to the clean/zero
  control.

Not supported yet:

- A causal location inside the vision tower before `get_image_features`.
- A causal location inside the language tower after image-token replacement.
- A learned cursor, cross-image cursor, or production inference mechanism.
- A conclusion that deepstack features are globally irrelevant; only this
  additive painted-clean intervention at the returned deepstack channels failed
  to steer row identity.

Next decider:

- Trace the image-embedding actuator one step deeper: either inspect the
  scatter/replacement boundary into `inputs_embeds`, or compare image-token
  position slices inside the language tower after replacement. This should stay
  focused on the primary image embedding path unless a later counterexample
  revives the deepstack branch.

Promotion decision:

- Not promoted. This remains a non-normative research result guiding the next
  mechanistic probe.
