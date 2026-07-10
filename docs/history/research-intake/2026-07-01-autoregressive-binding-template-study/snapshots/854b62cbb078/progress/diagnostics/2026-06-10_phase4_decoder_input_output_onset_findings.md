---
title: Phase 4 Decoder Input/Output Onset Findings
date: 2026-06-10
status: active-evidence-note
owner: codex
evidence_scope: phase4-decoder-input-output-onset-top4-allshards
---

# Phase 4 Decoder Input/Output Onset Findings

## Scope

This note records the first decoder-layer input/output sweep for the
autoregressive duplication mechanism study. It follows the residual-boundary
panel, which showed that decoder-layer input at layers `20/24` already carries
nearly the full patch-causal coordinate-slot basin.

Input manifest root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920
```

Main artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_decoder_input_output_layers_12_16_18_20_22_24_top4_allshards_summary.json
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_decoder_input_output_layers_12_16_18_20_22_24_top4_allshards_report.md
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_decoder_input_output_layers_12_16_18_20_22_24_top4_allshards_directionality.json
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_decoder_input_output_layers_12_16_18_20_22_24_top4_allshards_directionality.md
```

Probe settings:

- eight shard jobs launched across GPUs `0..7`;
- `target_top_k=4`;
- patch layers `12,16,18,20,22,24`;
- patch sites `decoder_layer_input,decoder_layer`;
- patch directions `masked_to_control,control_to_masked`;
- `top_k=8`.

Aggregate counts:

- `summary_count=8`;
- `residual_patch_row_count=1152`;
- `targeted_replay_case_count=2`;
- `checkpoint_count_sum=2`;
- paired directionality rows `48`.

## Findings

The main anchor `none_latest_ckpt32|post_y1/pre_x2` has a clear onset pattern:
layer `12` is weak, layer `16` is a transition/asymmetric repair point, and
layers `18+` carry the full bidirectional residual basin.

Anchor masked-to-control repair, decoder-layer input:

- `layer=12`: `prob_repair=0.000343`, `rank_repair=0.250`.
- `layer=16`: `prob_repair=0.009903`, `rank_repair=16.667`.
- `layer=18`: `prob_repair=0.013559`, `rank_repair=17.750`.
- `layer=20`: `prob_repair=0.014062`, `rank_repair=18.000`.
- `layer=22`: `prob_repair=0.014192`, `rank_repair=18.250`.
- `layer=24`: `prob_repair=0.013935`, `rank_repair=18.083`.

Anchor masked-to-control repair, decoder-layer output:

- `layer=12`: `prob_repair=0.000388`, `rank_repair=0.750`.
- `layer=16`: `prob_repair=0.009302`, `rank_repair=18.833`.
- `layer=18`: `prob_repair=0.012855`, `rank_repair=17.833`.
- `layer=20`: `prob_repair=0.013875`, `rank_repair=17.917`.
- `layer=22`: `prob_repair=0.014121`, `rank_repair=18.833`.
- `layer=24`: `prob_repair=0.013667`, `rank_repair=17.917`.

The same anchor's reverse direction separates the transition from later
symmetric damage:

- `decoder_layer_input|layer=12`: `prob_damage=0.000005`,
  `rank_damage=-0.250`.
- `decoder_layer_input|layer=16`: `prob_damage=0.000484`,
  `rank_damage=-1.583`.
- `decoder_layer_input|layer=18`: `prob_damage=0.012918`,
  `rank_damage=17.417`.
- `decoder_layer_input|layer=20`: `prob_damage=0.014434`,
  `rank_damage=20.417`.
- `decoder_layer|layer=16`: `prob_damage=0.000000`,
  `rank_damage=-1.750`.
- `decoder_layer|layer=18`: `prob_damage=0.014749`,
  `rank_damage=22.000`.

For `none_latest_ckpt32|post_x1/pre_y1`, the onset is also between layers
`16` and `18`:

- decoder-layer input repair rises from `0.000226` at layer `12` to `0.004563`
  at layer `16`, then `0.010460` at layer `18`;
- decoder-layer output repair rises from `0.000370` at layer `12` to
  `0.004269` at layer `16`, then `0.011106` at layer `18`.

The `aux_latest_ckpt32|post_x1/pre_y1` window shows the same broad transition:

- decoder-layer input repair is `0.000488` at layer `12`, `0.005941` at
  layer `16`, and `0.014286` at layer `18`;
- decoder-layer output repair is `0.000109` at layer `12`, `0.004470` at
  layer `16`, and `0.015300` at layer `18`.

`aux_latest_ckpt32|post_y1/pre_x2` remains weaker overall and is not the best
anchor for the onset claim. It has small probability effects across this sweep,
although some rank effects remain phase-specific.

## Mechanism Update

The current best picture is now more precise:

1. The residual coordinate-slot basin is not present at layer `12`.
2. Layer `16` is a transition point: it carries strong masked-to-control
   repair for the main anchor, but little reverse-direction symmetric damage.
3. Between layer `16` and layer `18`, the state becomes a full bidirectional
   residual basin: both repair and damage become large and stable.
4. Layers `18,20,22,24` mostly carry the basin forward rather than originating
   it.

This connects the earlier layer-16 attention evidence with the later residual
and MLP-input evidence. Layer `16` remains the likely routing/readout transition
zone, while the fully formed basin is visible by layer `18`.

## Next Probes

Recommended deterministic branch:

- run a finer layer sweep around the transition, especially layers `14,15,16,17,18`;
- include decoder-layer input/output and, if runtime allows, layer-16 attention
  group or self-attention input/output sites in the same top-4 anchor panel;
- keep `none_latest_ckpt32|post_y1/pre_x2` as the primary anchor because it
  cleanly separates weak layer-12 signal, asymmetric layer-16 repair, and
  bidirectional layer-18+ basin.

Guardrail:

- This is an all-shard top-4 panel. It identifies a transition interval, not a
  single causal operation. The next probe should resolve layer `14-18` more
  finely before making final origin claims.
