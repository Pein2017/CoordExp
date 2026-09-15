---
title: PVCI Image-Embed Spatial Localization
description: Splits the winning primary image-embedding delta by coarse Qwen merged-token regions to test whether the row-binding actuator is local, halo-like, or distributed.
type: idea
role: research-unit
authority: non_normative_research
promotion_status: not_promoted
unit_id: 2026-07-09-pvci-image-embed-spatial-localization
topic: qwen3-vl-painted-gt-transcription-probe
status: complete
tags:
  - coordexp-swift
  - research-unit
  - painted-gt
  - pvci
  - visual-delta
  - image-embeds
  - spatial-localization
updated: 2026-07-09
---

# PVCI Image-Embed Spatial Localization

## Question

The
[PVCI Visual-Feature Component Localization](../2026-07-09-pvci-visual-feature-component-localization/unit.md)
localized the effective same-row visual-delta actuator to the primary
`image_embeds` channel at the Qwen3-VL `get_image_features` seam. This unit asks
where inside that winning image-token channel the actuator lives:

- merged visual tokens covering the rendered mark / painted object region;
- source-box region;
- rendered halo neighborhood;
- source-ring neighborhood;
- non-mark complement.

The region masks are coarse Qwen merged-token approximations, not exact pixel
attribution maps.

## Completion Promise

This unit is complete when:

- Evidence gate: `debug8`, and preferably `val32`, compares clean, zero, full
  stored delta, mixed mark delta, full image-only delta, deepstack-all sanity
  control, and image-only local-mask conditions on the same wrong-object
  row-binding panel.
- Acceptable evidence: identity-conflict summaries report `prediction_rate`,
  `target_row_rate`, `source_object_row_rate`,
  `rendered_mark_copy_rate`, and `third_object_row_rate`, with receipts proving
  active image-local conditions enable image deltas, disable all deepstack
  layers, and have zero fallback deltas.
- Insufficient evidence: a region is called causal only because it changes text
  form, without suppressing target-row binding or increasing source-object /
  rendered-mark binding toward the full image-only condition.

## Evidence Scope

- Checkout or branch: `/data/CoordExp/.codex/worktrees/eb1c/CoordExp`,
  `codex/qwen3-vl-painted-gt-transcription-probe`.
- Commit or diff scope: local diff adding image-embed spatial-localization
  conditions to `scripts/probes/painted_gt/run_feature_store_delta_probe.py`.
- Config: existing PVCI feature-store delta probe defaults unless overridden by
  the commands below.
- Checkpoint or model version: E1 anti-copy checkpoint:
  `/data/CoordExp/outputs/painted_gt/pvci_step2/train_snap_radius/painted_gt_pvci_step2_snap_radius_train256_anti_copy_mix_step484_warm_start_dora_all_towers_accelerate8_ebs8_16epoch-pvci-step2-anticopy-retry3-20260708T1726Z/checkpoints/step-484`.
- Artifact root:
  `/data/CoordExp/outputs/painted_gt/pvci_image_embed_spatial_localization/`.
- Commands:
  - `pytest tests/painted_gt/test_visual_feature_store_probe.py -q`
  - `python -m py_compile scripts/probes/painted_gt/run_feature_store_delta_probe.py`
  - `python scripts/probes/painted_gt/run_feature_store_delta_probe.py --max-rows 8 --conditions clean_no_cursor,clean_plus_zero_delta,clean_plus_mark_delta,clean_plus_stored_delta,clean_plus_image_only_delta,clean_plus_deepstack_all_only_delta,clean_plus_image_mark_delta,clean_plus_image_source_box_delta,clean_plus_image_rendered_halo_delta,clean_plus_image_source_ring_delta,clean_plus_image_non_mark_delta --output-root /data/CoordExp/outputs/painted_gt/pvci_image_embed_spatial_localization/e1_wrong_object_jitter_medium_debug8 --force`
- Metrics or counters: identity-conflict rates plus
  `feature_store_receipts[*].component_localization_summary` and
  `fallback_zero_delta_count`.
- Sample window: first `debug8`; escalate to `val32` if the debug run is sane.
- Known limitations: same-row source deltas remain oracle interventions; masks
  are approximate merged visual-token regions; this is not an attention map,
  learned cursor, or production inference mechanism.

## Procedure

1. Add explicit image-embed local-mask conditions without changing existing
   mixed local-mask or component-localization modes.
2. Apply the selected mask to `delta_image_embeds` only; all deepstack deltas
   must be disabled for the new conditions.
3. Keep clean, zero, mixed mark, full stored, full image-only, and
   deepstack-all controls in the same run.
4. Run `debug8`; inspect identity-conflict summaries and feature-store
   receipts.
5. If sane, run `val32`.
6. If a region wins or remains ambiguous, consider repeating only the winning
   or ambiguous local conditions on the same-object augmented feature store.

## Observations

- Direct observation: `debug8` and `val32` both show that image-local masks can
  partially reproduce the image-only steering actuator, but no single coarse
  region fully matches the full image-only delta.
- Counterexample or negative result: `clean_plus_image_mark_delta` is strong
  but insufficient by itself on `val32`; non-mark and source-ring masks also
  carry meaningful steering, which argues against a single exact bbox-token
  explanation.
- Artifact handle:
  `/data/CoordExp/outputs/painted_gt/pvci_image_embed_spatial_localization/e1_wrong_object_jitter_medium_debug8`
  and
  `/data/CoordExp/outputs/painted_gt/pvci_image_embed_spatial_localization/e1_wrong_object_jitter_medium_val32`.

## Interpretation

- Supported reading: the actuator is image-token-path dominant and spatially
  structured, but it is not fully localized to the rendered mark or source box
  under the current coarse merged-token masks.
- Alternative reading: coarse Qwen merged-token masks may blur the true source.
  The apparent distributed signal could reflect tokenization/merger granularity
  rather than genuinely global visual evidence.
- Remaining uncertainty: whether the effective signal is formed by multiple
  local regions, by broad context shifts caused by painting, or by imperfect
  pixel-to-token mask attribution.

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
  | `clean_plus_image_mark_delta` | `0.8750` | `0.0000` | `0.4286` | `0.2857` | `0.1429` |
  | `clean_plus_image_source_box_delta` | `0.8750` | `0.1429` | `0.2857` | `0.2857` | `0.1429` |
  | `clean_plus_image_rendered_halo_delta` | `0.6250` | `0.0000` | `0.0000` | `0.0000` | `0.6000` |
  | `clean_plus_image_source_ring_delta` | `0.6250` | `0.0000` | `0.6000` | `0.0000` | `0.2000` |
  | `clean_plus_image_non_mark_delta` | `0.8750` | `0.0000` | `0.2857` | `0.4286` | `0.1429` |

- `val32` identity-conflict rates:

  | Condition | Pred rate | Target row | Source-object row | Mark-copy row | Third-object row |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | `clean_no_cursor` | `0.5312` | `0.4706` | `0.0588` | `0.1176` | `0.1176` |
  | `clean_plus_zero_delta` | `0.5312` | `0.4706` | `0.0588` | `0.1176` | `0.1176` |
  | `clean_plus_mark_delta` | `0.8125` | `0.0000` | `0.4231` | `0.2308` | `0.0000` |
  | `clean_plus_stored_delta` | `0.8125` | `0.0000` | `0.5000` | `0.1154` | `0.0000` |
  | `clean_plus_image_only_delta` | `0.8125` | `0.0385` | `0.3846` | `0.2308` | `0.0000` |
  | `clean_plus_deepstack_all_only_delta` | `0.5625` | `0.5000` | `0.0556` | `0.1111` | `0.1111` |
  | `clean_plus_image_mark_delta` | `0.7500` | `0.1250` | `0.2917` | `0.2083` | `0.0417` |
  | `clean_plus_image_source_box_delta` | `0.6250` | `0.2500` | `0.2000` | `0.2500` | `0.0500` |
  | `clean_plus_image_rendered_halo_delta` | `0.6562` | `0.1905` | `0.1429` | `0.1429` | `0.1905` |
  | `clean_plus_image_source_ring_delta` | `0.5938` | `0.1579` | `0.3158` | `0.1053` | `0.1053` |
  | `clean_plus_image_non_mark_delta` | `0.7500` | `0.1250` | `0.3333` | `0.1667` | `0.0833` |

- Receipts showed zero fallback deltas for every active condition. The
  image-local conditions reported `image_delta_enabled: true`,
  `deepstack_delta_layer_indices: []`, `deepstack_layer_count: 3`, and the
  expected `image_mask_key` for each condition.

Evidence gate:

- Satisfied for the original-store image-embed spatial-localization question.
  The matched `debug8` and `val32` runs include clean, zero, mixed mark, full
  stored, full image-only, deepstack-all, and image-only local-mask conditions
  with receipt-backed component isolation.

Supported:

- The image-embed actuator is spatially structured and can be partially
  decomposed by merged-token masks.
- The rendered mark / source-box neighborhood is important but not sufficient:
  on `val32`, `clean_plus_image_mark_delta` reduces target-row rate from
  `0.4706` to `0.1250`, but does not match full image-only's `0.0385`.
- Non-mark and source-ring masks also carry actuator signal. This suggests
  painting changes a broader image-token context, or that the merged-token masks
  are too coarse to isolate the true object/paint locus exactly.
- The deepstack-all control remains close to clean/zero, reinforcing that this
  effect is specific to the primary image-token path.

Not supported yet:

- An exact visual-token attribution map.
- A claim that only the painted bbox or rendered mark carries the causal signal.
- A learned cursor, normal-inference attention explanation, or production
  intervention.
- Same-object augmented-store confirmation for the winning/ambiguous local
  masks.

Next decider:

- Either repeat the strongest/ambiguous local image masks on the same-object
  augmented feature store, or move one seam forward to test scatter equivalence:
  inject the winning image-only or image-local deltas after `<image_pad>`
  replacement in `inputs_embeds`.
- If choosing augmented repeat, prioritize `image_mark`, `image_non_mark`,
  `image_source_ring`, and full `image_only`; avoid expanding to the full matrix
  unless these contradict the current conclusion.

Promotion decision:

- Not promoted. This remains a research-only causal-intervention probe.
