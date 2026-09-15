---
title: PVCI Post-Scatter Layer Onset
description: Locates where post-scatter visual-token steering becomes visible in next-token logit margins for divergence-conditioned rows.
type: idea
role: research-unit
authority: non_normative_research
promotion_status: not_promoted
unit_id: 2026-07-09-pvci-post-scatter-layer-onset
topic: qwen3-vl-painted-gt-transcription-probe
status: complete
tags:
  - coordexp-swift
  - research-unit
  - pvci
  - qwen3-vl
  - hidden-state
updated: 2026-07-09
---

# PVCI Post-Scatter Layer Onset

## Question

At which language-model layers does the post-scatter image-token actuator become
visible as a preference for the steered next token over the clean next token?

The immediate prior unit showed that injecting the same visual delta after Qwen
scatters image features into `inputs_embeds` exactly reproduces the earlier
`get_image_features` image-only positive control. This unit asks where that
signal becomes readable by the LM head for the two dominant divergence families:

- step-0 stop-vs-continue: clean emits `<|im_end|>`, post-scatter emits
  `<|object_ref_start|>`;
- step-1 description competition: clean and post-scatter share
  `<|object_ref_start|>` and then choose different first description tokens.

## Completion Promise

This unit is complete when:

- Evidence gate: a probe replays the common prefix immediately before clean vs
  post-scatter divergence, captures selected decoder-layer states, applies an
  LM-head logit lens to the clean-next and post-next tokens, and reports
  per-layer margin curves.
- Acceptable evidence: the report includes row ids, divergence step, competing
  token ids/text, final-logit margin, per-layer margin delta, earliest
  meaningful onset layer under an explicitly recorded rule, and receipt fields
  proving the same post-scatter visual-token intervention was used.
- Insufficient evidence: aggregate generation metrics without layer readouts,
  layer curves without divergence-conditioned prefixes, or probes that change
  prompt/template/checkpoint/decode identity relative to the post-scatter
  equivalence unit.

## Evidence Scope

- Checkout or branch: `/data/CoordExp/.codex/worktrees/eb1c/CoordExp`,
  branch `codex/qwen3-vl-painted-gt-transcription-probe`.
- Commit or diff scope: research-only script
  `scripts/probes/painted_gt/run_post_scatter_layer_onset_probe.py` and focused
  tests.
- Source artifacts:
  `/data/CoordExp/outputs/painted_gt/pvci_post_scatter_image_token_equivalence/e1_wrong_object_jitter_medium_val32`.
- Config: same resolved inference config as the post-scatter equivalence run.
- Checkpoint or model version: same step-484 PVCI adapter and base model used by
  the post-scatter equivalence run.
- Artifact root:
  `/data/CoordExp/outputs/painted_gt/pvci_post_scatter_layer_onset/`.
- Commands:
  - `pytest tests/painted_gt/test_post_scatter_layer_onset_probe.py tests/painted_gt/test_visual_feature_store_probe.py -q`
  - `python -m py_compile scripts/probes/painted_gt/run_post_scatter_layer_onset_probe.py scripts/probes/painted_gt/run_feature_store_delta_probe.py`
  - `python /data/CoordExp/.codex/skills/coordexp-research-spec/scripts/validate_research_units.py research`
  - `CUDA_VISIBLE_DEVICES=4 python scripts/probes/painted_gt/run_post_scatter_layer_onset_probe.py --max-events 4 --output-root /data/CoordExp/outputs/painted_gt/pvci_post_scatter_layer_onset/e1_wrong_object_jitter_medium_val32_step0_step1_debug4 --force`
  - `CUDA_VISIBLE_DEVICES=4 python scripts/probes/painted_gt/run_post_scatter_layer_onset_probe.py --output-root /data/CoordExp/outputs/painted_gt/pvci_post_scatter_layer_onset/e1_wrong_object_jitter_medium_val32_step0_step1_all --force`
- Metrics or counters: divergence-row count, selected-event count, per-layer
  post-token-vs-clean-token margin, final margin, onset layer.
- Sample window: start with all val32 rows whose first divergence is at step 0
  or step 1.
- Known limitations: logit-lens readouts are evidence of linear readability,
  not causal proof that a layer alone causes the generated token.

## Procedure

1. Select divergence-conditioned events from existing clean and post-scatter
   token traces.
2. Rebuild each common prefix from the clean prompt plus shared generated tokens
   before divergence.
3. Run clean and post-scatter image-only forward passes on that identical prefix.
4. Capture selected decoder-layer hidden states at the final prefix position.
5. Compute the margin `logit(post_next_token) - logit(clean_next_token)` at each
   layer and at the final logits.
6. Summarize onset by divergence family and inspect whether step-0
   stop-vs-continue and step-1 description competition emerge at the same or
   different depths.

## Observations

- Direct observation: among the 32 val rows, 20 rows diverged at step 0 or step
  1 and were selected by the probe. There were 9 step-0 stop-vs-continue events
  and 11 step-1 description-competition events.
- Direct observation: early layers had near-zero mean margin deltas. The overall
  mean layer delta stayed close to zero through layer 15, grew at layer 16
  (`0.7471`), layer 17 (`2.4635`), layer 19 (`3.8101`), layer 20 (`6.5996`),
  layer 21 (`8.9177`), layer 22 (`12.5895`), and layer 23 (`14.4822`).
- Direct observation: step-0 stop-vs-continue events reached the half-final
  delta threshold mostly at layers 17-21: `{17: 1, 18: 1, 19: 4, 20: 2, 21: 1}`.
- Direct observation: step-1 description-competition events reached the
  half-final delta threshold later, mostly at layers 20-23:
  `{20: 2, 21: 4, 22: 4, 23: 1}`.
- Direct observation: mean final delta margin was `12.6389` for step-0
  stop-vs-continue and `24.4318` for step-1 description competition.
- Counterexample or negative result: there is no evidence that early layers
  alone linearly expose the steered token preference; sign matches in early
  layers exist but are tiny and not robust by the half-final rule.
- Artifact handle:
  `/data/CoordExp/outputs/painted_gt/pvci_post_scatter_layer_onset/e1_wrong_object_jitter_medium_val32_step0_step1_debug4`.
- Artifact handle:
  `/data/CoordExp/outputs/painted_gt/pvci_post_scatter_layer_onset/e1_wrong_object_jitter_medium_val32_step0_step1_all`.

All-events summary:

| family | events | mean final delta | half-final onset layers |
| --- | ---: | ---: | --- |
| `step0_stop_vs_continue` | `9` | `12.6389` | `{17: 1, 18: 1, 19: 4, 20: 2, 21: 1}` |
| `step1_description_competition` | `11` | `24.4318` | `{20: 2, 21: 4, 22: 4, 23: 1}` |

## Interpretation

- Supported reading: the post-scatter image-token actuator is not merely a
  first-layer or shallow visual-token bias. It becomes strongly LM-head-readable
  only in late-middle language layers.
- Supported reading: stop-vs-continue control appears earlier than description
  identity selection. This suggests continuation gating and object-description
  binding may be separable downstream computations rather than one identical
  switch.
- Alternative reading: the logit-lens readout may understate early information
  that is present but not linearly aligned with the final LM head.
- Remaining uncertainty: this is linear readability, not causal proof. We still
  need residual patching to test whether replacing clean residual states with
  post-scatter states at layers 17-23 is sufficient to recover the final
  steered next-token margin.

## Research Unit Closeout

Observed:

The divergence-conditioned logit-lens probe selected 20 val32 events whose first
clean/post-scatter divergence occurred at generation step 0 or 1. The steered
token margin stayed near zero in early layers and rose sharply in late-middle
layers. Step-0 stop-vs-continue half-final onset concentrated at layers 17-21;
step-1 description competition concentrated at layers 20-23.

Evidence gate:

Satisfied for linear-readability onset. Not sufficient for causal localization.

Supported:

The post-scatter visual-token signal becomes LM-head-readable late in the text
stack, with continuation gating appearing earlier than description identity
selection.

Not supported yet:

Causal sufficiency of any layer or sublayer, and whether attention or MLP is the
dominant transformation that makes the signal readable.

Next decider:

Run a residual patch probe: capture post-scatter hidden states at layers
17-23, patch the clean run at the same prefix/layer, and measure recovery of
the post-scatter final next-token margin. If patching is sufficient, split the
dominant layers into attention/MLP sublayer routes.

Promotion decision:

Not promoted.
