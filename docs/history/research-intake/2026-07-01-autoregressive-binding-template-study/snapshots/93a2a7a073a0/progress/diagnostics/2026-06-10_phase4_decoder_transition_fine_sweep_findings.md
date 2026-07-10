---
title: Phase 4 Decoder Transition Fine Sweep Findings
date: 2026-06-10
status: active-evidence-note
owner: codex
evidence_scope: phase4-decoder-input-output-transition-top4-allshards
---

# Phase 4 Decoder Transition Fine Sweep Findings

## Scope

This note records the follow-up fine sweep around the decoder-layer transition
identified in the coarse decoder input/output onset panel.

Input manifest root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920
```

Main artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_decoder_input_output_layers_14_15_16_17_18_top4_allshards_summary.json
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_decoder_input_output_layers_14_15_16_17_18_top4_allshards_report.md
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_decoder_input_output_layers_14_15_16_17_18_top4_allshards_directionality.json
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_decoder_input_output_layers_14_15_16_17_18_top4_allshards_directionality.md
```

Probe settings:

- eight shard jobs launched across GPUs `0..7`;
- `target_top_k=4`;
- patch layers `14,15,16,17,18`;
- patch sites `decoder_layer_input,decoder_layer`;
- patch directions `masked_to_control,control_to_masked`;
- `top_k=8`.

Aggregate counts:

- `summary_count=8`;
- `residual_patch_row_count=960`;
- `targeted_replay_case_count=2`;
- `checkpoint_count_sum=2`;
- paired directionality rows `40`.

## Main Anchor

The clearest anchor remains `none_latest_ckpt32|post_y1/pre_x2`. The fine
sweep moves the transition earlier and narrows it:

- `decoder_layer_input@14`: masked-to-control repair `0.009225`, reverse
  control-to-masked damage `0.000000`, rank repair `18.500`, rank damage
  `-2.833`.
- `decoder_layer_input@15`: repair `0.008552`, damage `0.000536`, rank repair
  `17.833`, rank damage `-1.500`.
- `decoder_layer_input@16`: repair `0.011091`, damage `0.001047`, rank repair
  `20.167`, rank damage `-1.250`.
- `decoder_layer_input@17`: repair `0.010274`, damage `0.000000`, rank repair
  `21.917`, rank damage `-1.667`.
- `decoder_layer_input@18`: repair `0.014757`, damage `0.013846`, rank repair
  `21.167`, rank damage `18.500`.

For decoder-layer output on the same anchor:

- `decoder_layer@14`: repair `0.008552`, damage `0.000536`, rank repair
  `17.833`, rank damage `-1.500`.
- `decoder_layer@15`: repair `0.011091`, damage `0.001047`, rank repair
  `20.167`, rank damage `-1.250`.
- `decoder_layer@16`: repair `0.010274`, damage `0.000000`, rank repair
  `21.917`, rank damage `-1.667`.
- `decoder_layer@17`: repair `0.014757`, damage `0.013846`, rank repair
  `21.167`, rank damage `18.500`.
- `decoder_layer@18`: repair `0.013712`, damage `0.015641`, rank repair
  `19.917`, rank damage `23.583`.

This makes the interpretation sharper than the coarse sweep: layers `14-16`
already carry strong repair for the masked coordinate slot, but they do not
behave like a full residual basin under the reverse direction. The symmetric
basin appears at the output of layer `17` and the input/output of layer `18`.

## Cross-Anchor Pattern

The `none_latest_ckpt32|post_x1/pre_y1` window shows a weaker but compatible
phase change:

- decoder-layer output repair rises from `0.001872` at layer `14` to
  `0.004615` at layer `15`, `0.004524` at layer `16`, then jumps to
  `0.011516` at layer `17` and `0.011563` at layer `18`;
- reverse damage is already visible across layers `14-18`, so this window is
  less clean for isolating a one-way repair stage.

The `aux_latest_ckpt32|post_x1/pre_y1` window has a similar late jump:

- decoder-layer output repair is `0.003303` at layer `14`, `0.005721` at
  layer `15`, `0.003730` at layer `16`, then `0.013070` at layer `17` and
  `0.013834` at layer `18`;
- reverse damage is again visible earlier, making it useful as a robustness
  check but not the cleanest origin anchor.

The `aux_latest_ckpt32|post_y1/pre_x2` window remains weak and mixed in this
panel. It should not anchor the origin claim.

## Mechanism Update

The current best picture is now:

1. A repair-capable coordinate-slot direction is already readable by layers
   `14-16` for the main anchor.
2. That early signal is asymmetric: patching masked state toward control
   repairs the coordinate slot, but patching control toward masked does not yet
   strongly damage it.
3. The full bidirectional basin emerges between layer `16` output and layer
   `18` input, with the cleanest boundary at `decoder_layer@17` /
   `decoder_layer_input@18`.
4. The previously observed layer-16 attention evidence is therefore likely a
   routing or readout precursor, not the final basin itself.

This is now a promising branch for deeper probing because it separates
"repair-capable direction exists" from "control state is causally locked into a
coordinate basin." The next highest-leverage question is what operation between
layer `16` output and layer `18` input turns the one-way repair direction into
a symmetric basin.

## Next Probes

Recommended deterministic branch:

- patch layer `17` module boundaries more finely: `self_attn_input`,
  `self_attn`, `post_attention_residual`, `post_attention_norm`, `mlp_input`,
  `mlp`, plus decoder-layer input/output as anchors;
- keep the primary anchor `none_latest_ckpt32|post_y1/pre_x2`;
- include `none_latest_ckpt32|post_x1/pre_y1` and
  `aux_latest_ckpt32|post_x1/pre_y1` only as robustness checks;
- if the layer-17 module boundary isolates the transition, follow with
  head-level or grouped attention patching inside layer `17`.

Guardrail:

- This panel uses top-4 selected replay cases. It is strong enough to choose
  the next causal probe, but not enough to claim the final origin of the
  duplication basin without layer-17 module and attention substructure tests.
