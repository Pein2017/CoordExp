---
title: PVCI Post-Scatter Image-Token Equivalence
description: Tests whether the visual-delta actuator survives when injected after Qwen scatters image features into LM input embeddings.
type: idea
role: research-unit
authority: non_normative_research
promotion_status: not_promoted
unit_id: 2026-07-09-pvci-post-scatter-image-token-equivalence
topic: qwen3-vl-painted-gt-transcription-probe
status: complete
tags:
  - coordexp-swift
  - research-unit
  - pvci
  - qwen3-vl
updated: 2026-07-09
---

# PVCI Post-Scatter Image-Token Equivalence

## Question

Does the painted-GT visual-delta steering effect remain when the same stored
image delta is injected after Qwen3-VL has already replaced `<image_pad>` tokens
with image embeddings in `inputs_embeds`?

This resolves whether the previous `get_image_features` hook is merely a
convenient route into ordinary LM image-token embeddings, or whether the
pre-scatter feature tuple boundary itself is part of the effect.

## Completion Promise

This unit is complete when:

- Evidence gate: a debug and val-sized run compare clean baseline, existing
  `get_image_features` image-only positive control, post-scatter zero control,
  post-scatter full image delta, post-scatter spatial slices, and a wrong-row
  post-scatter sentinel under the same decode surface.
- Acceptable evidence: condition receipts show `get_image_features_patched:
  false` for post-scatter conditions, `hook_boundary:
  Qwen3VLModel.post_scatter_inputs_embeds`, nonzero applied hook counts, exact
  visual-token count agreement, and no painted pixels loaded during eval.
- Insufficient evidence: generated text alone without hook receipts, a run that
  changes prompts/checkpoints/decode config, or a post-scatter path that bypasses
  Qwen placeholder count checks or MRoPE construction.

## Evidence Scope

- Checkout or branch: `/data/CoordExp/.codex/worktrees/eb1c/CoordExp`,
  branch `codex/qwen3-vl-painted-gt-transcription-probe`.
- Commit or diff scope: dirty research-only probe runner and tests adding
  post-scatter conditions to `run_feature_store_delta_probe.py`.
- Config: same default config as `run_feature_store_delta_probe.py`.
- Checkpoint or model version: same step-484 PVCI adapter and base model used by
  the preceding feature-store delta probes.
- Artifact root:
  `/data/CoordExp/outputs/painted_gt/pvci_post_scatter_image_token_equivalence/`.
- Commands:
  - `pytest tests/painted_gt/test_visual_feature_store_probe.py -q`
  - `python /data/CoordExp/.codex/skills/coordexp-research-spec/scripts/validate_research_units.py research`
  - `python -m py_compile scripts/probes/painted_gt/run_feature_store_delta_probe.py`
  - `CUDA_VISIBLE_DEVICES=4 python scripts/probes/painted_gt/run_feature_store_delta_probe.py --max-rows 8 --conditions clean_no_cursor,clean_plus_zero_delta,clean_plus_image_only_delta,post_scatter_zero_delta,post_scatter_image_only_delta,post_scatter_image_mark_delta,post_scatter_image_non_mark_delta,post_scatter_image_source_ring_delta,post_scatter_wrong_row_image_delta --output-root /data/CoordExp/outputs/painted_gt/pvci_post_scatter_image_token_equivalence/e1_wrong_object_jitter_medium_debug8 --force`
  - `CUDA_VISIBLE_DEVICES=4 python scripts/probes/painted_gt/run_feature_store_delta_probe.py --max-rows 32 --conditions clean_no_cursor,clean_plus_zero_delta,clean_plus_image_only_delta,post_scatter_zero_delta,post_scatter_image_only_delta,post_scatter_image_mark_delta,post_scatter_image_non_mark_delta,post_scatter_image_source_ring_delta,post_scatter_wrong_row_image_delta --output-root /data/CoordExp/outputs/painted_gt/pvci_post_scatter_image_token_equivalence/e1_wrong_object_jitter_medium_val32 --force`
  - `python scripts/probes/painted_gt/analyze_identity_conflict.py --input <condition>/gt_vs_pred.jsonl --output-dir <condition>/identity_conflict --condition-name <condition>`
- Metrics or counters: prediction rate, target/source/mark/third binding rates,
  hook receipts, fallback counts, visual-token counts.
- Sample window: debug8 first, then val32 if debug receipts and behavior are
  sane.
- Known limitations: this is still a same-row oracle stored-delta intervention;
  it is not a learned cursor or deployable inference mechanism.

## Procedure

1. Add research-only post-scatter conditions to the existing feature-store
   probe while leaving normal Qwen vision feature extraction, placeholder count
   checks, MRoPE, and generation cache setup intact.
2. Run debug8 with clean baseline, zero control, existing image-only positive
   control, post-scatter image-only, post-scatter spatial slices, and a wrong-row
   sentinel.
3. Inspect receipts before interpreting metrics.
4. If debug8 passes, run val32 and compare post-scatter rates to the existing
   image-only positive control.
5. Decide whether same-object augmented local-mask repetition is still a
   necessary next probe.

## Observations

- Direct observation: post-scatter zero was byte-for-byte equivalent to clean
  output on both debug8 and val32. `clean_plus_zero_delta` was also
  byte-for-byte equivalent to clean.
- Direct observation: post-scatter full image delta was byte-for-byte equivalent
  to the existing `get_image_features` image-only positive control on both
  debug8 and val32.
- Receipt observation: all post-scatter condition receipts reported
  `get_image_features_patched: false`, `hook_boundary:
  Qwen3VLModel.post_scatter_inputs_embeds`, and `post_scatter_applied_hook_count:
  1` for each batch.
- Receipt observation: val32 post-scatter receipts covered `31424` visual tokens
  across 16 decode batches. The full, mark, non-mark, and source-ring
  post-scatter conditions had zero fallback source entries. The wrong-row
  sentinel had one fallback row because one selected row had no compatible
  wrong-row source.
- Artifact handle:
  `/data/CoordExp/outputs/painted_gt/pvci_post_scatter_image_token_equivalence/e1_wrong_object_jitter_medium_debug8`.
- Artifact handle:
  `/data/CoordExp/outputs/painted_gt/pvci_post_scatter_image_token_equivalence/e1_wrong_object_jitter_medium_val32`.

Val32 identity-conflict table:

| condition | pred | target | source | mark | third | chimera |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `clean_no_cursor` | `0.5312` | `0.4706` | `0.0588` | `0.1176` | `0.1176` | `0.0588` |
| `clean_plus_zero_delta` | `0.5312` | `0.4706` | `0.0588` | `0.1176` | `0.1176` | `0.0588` |
| `clean_plus_image_only_delta` | `0.8125` | `0.0385` | `0.3846` | `0.2308` | `0.0000` | `0.0385` |
| `post_scatter_zero_delta` | `0.5312` | `0.4706` | `0.0588` | `0.1176` | `0.1176` | `0.0588` |
| `post_scatter_image_only_delta` | `0.8125` | `0.0385` | `0.3846` | `0.2308` | `0.0000` | `0.0385` |
| `post_scatter_image_mark_delta` | `0.7500` | `0.1250` | `0.2917` | `0.2083` | `0.0417` | `0.0417` |
| `post_scatter_image_non_mark_delta` | `0.7500` | `0.1250` | `0.3333` | `0.1667` | `0.0833` | `0.0417` |
| `post_scatter_image_source_ring_delta` | `0.5938` | `0.1579` | `0.3158` | `0.1053` | `0.1053` | `0.0526` |
| `post_scatter_wrong_row_image_delta` | `0.6250` | `0.2500` | `0.1000` | `0.1500` | `0.2000` | `0.0000` |

## Interpretation

- Supported reading: the effective visual-delta actuator enters the decoder as
  ordinary image-token embeddings after Qwen scatters vision features into
  `inputs_embeds`. The earlier `get_image_features` hook is a convenient
  upstream boundary, not a required part of the effect.
- Supported reading: the post-scatter zero control preserving clean output
  argues against the language-model wrapper itself changing generation.
- Supported reading: the spatial-slice rates remain aligned with the previous
  image-embed spatial-localization run, so the localization story is not an
  artifact of the `get_image_features` tuple boundary.
- Alternative reading: this is still a same-row oracle delta and may encode
  more than a deployable cursor would learn.
- Remaining uncertainty: we have not yet shown where inside the LM stack this
  injected image-token signal becomes row-selection logits, nor whether a
  learned non-oracle cursor can reproduce it.

## Research Unit Closeout

Observed:

Post-scatter full image delta exactly reproduced the earlier image-only
`get_image_features` hook outputs on debug8 and val32. Post-scatter zero exactly
reproduced clean outputs on debug8 and val32. Receipts confirm the post-scatter
path left `get_image_features` unpatched and applied at
`Qwen3VLModel.post_scatter_inputs_embeds`.

Evidence gate:

Satisfied for hook-boundary equivalence.

Supported:

The actuator is present after image-feature scatter in the LM input embedding
stream. Immediate same-object augmented local-mask repetition is no longer the
most privileged next test for boundary validity.

Not supported yet:

This does not prove a learned cursor, causal sufficiency of any single local
region, or internal LM layer location.

Next decider:

Prefer a tiny early-layer residual or logit-onset probe if the next question is
where the image-token signal turns into row-binding behavior. Keep same-object
augmented local-mask repetition as a later robustness control if local spatial
attribution becomes the claim.

Promotion decision:

Not promoted.
