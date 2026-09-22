---
title: Phase 4 MLP Input/Output Patch Findings
date: 2026-06-10
status: active-evidence-note
owner: codex
evidence_scope: phase4-mlp-input-output-top4-allshards
---

# Phase 4 MLP Input/Output Patch Findings

## Scope

This note records the first MLP input-vs-output residual patch probe for the
autoregressive duplication mechanism study. It follows the grouped attention
result, which showed that patching more layer-16 attention heads together does
not repair the anchor coordinate-slot basin.

Input manifest root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920
```

Main artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_mlp_input_output_layers_20_24_top4_allshards_summary.json
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_mlp_input_output_layers_20_24_top4_allshards_report.md
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_mlp_input_output_layers_20_24_top4_allshards_directionality.json
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_mlp_input_output_layers_20_24_top4_allshards_directionality.md
```

Probe settings:

- eight shard jobs launched across GPUs `0..7`;
- `target_top_k=4`;
- patch layers `20,24`;
- patch sites `decoder_layer,mlp_input,mlp`;
- patch directions `masked_to_control,control_to_masked`;
- `top_k=8`.

Aggregate counts:

- `summary_count=8`;
- `residual_patch_row_count=576`;
- `targeted_replay_case_count=2`;
- `checkpoint_count_sum=2`;
- paired directionality rows `24`.

## Findings

The MLP input and MLP output patch summaries are identical for every inspected
phase/layer/site pair in this panel. That is expected if the MLP is a
positionwise deterministic transform: replacing the selected token's MLP input
with the control vector causes the selected token's MLP output to become the
same vector that direct MLP-output patching would have inserted.

For the anchor `none_latest_ckpt32|post_y1/pre_x2|masked_to_control`:

- `decoder_layer|layer=20`: `prob_repair=0.013875`,
  `rank_repair=17.917`.
- `decoder_layer|layer=24`: `prob_repair=0.013667`,
  `rank_repair=17.917`.
- `mlp_input|layer=20`: `prob_repair=0.006170`,
  `rank_repair=10.917`.
- `mlp|layer=20`: `prob_repair=0.006170`,
  `rank_repair=10.917`.
- `mlp_input|layer=24`: `prob_repair=0.008964`,
  `rank_repair=14.583`.
- `mlp|layer=24`: `prob_repair=0.008964`,
  `rank_repair=14.583`.

For the same anchor in the reverse direction:

- `decoder_layer|layer=20`: `prob_damage=0.013179`,
  `rank_damage=16.833`.
- `decoder_layer|layer=24`: `prob_damage=0.013004`,
  `rank_damage=16.083`.
- `mlp_input|layer=20`: `prob_damage=0.000000`,
  `rank_damage=-0.833`.
- `mlp|layer=20`: `prob_damage=0.000000`,
  `rank_damage=-0.833`.
- `mlp_input|layer=24`: `prob_damage=0.000000`,
  `rank_damage=-2.000`.
- `mlp|layer=24`: `prob_damage=0.000000`,
  `rank_damage=-2.000`.

The identical MLP-input/MLP-output rows also appear in the other top-4 windows.
Examples:

- `none_latest_ckpt32|post_x1/pre_y1|masked_to_control|site=mlp_input|layer=24`
  and `site=mlp|layer=24` both have `prob_repair=0.005997`,
  `rank_repair=2.750`.
- `aux_latest_ckpt32|post_x1/pre_y1|masked_to_control|site=mlp_input|layer=24`
  and `site=mlp|layer=24` both have `prob_repair=0.007177`,
  `rank_repair=3.083`.

## Mechanism Update

This result does not support a story where the MLP computation itself is the
place where a hidden coordinate-slot state is newly created from an otherwise
unformed input. Instead, the repair-relevant state is already present in the
selected token's residual stream as it enters the MLP at layers `20/24`.

Current best picture:

1. Grouped layer-16 attention heads do not directly reconstruct the anchor
   coordinate-slot basin.
2. MLP input and output patching at layers `20/24` are equivalent at the
   selected token.
3. Therefore, the residual stream immediately before the MLP at layers `20/24`
   already carries a substantial portion of the repair state.
4. Full decoder-layer residual patches remain stronger than MLP-only patches,
   so attention/residual-add components around the same layers may carry the
   remaining repair and symmetric-damage signal.

The next boundary should move from "inside the MLP" to "how the pre-MLP
residual state is formed."

## Next Probes

Recommended deterministic branch:

- add pre-attention input and post-attention/residual-add patch sites for
  layers `20/24`, if the module structure permits clean hooks;
- compare those with `mlp_input` and full decoder-layer patching on the same
  top-4 anchor rows;
- test whether layer-16 attention-group information affects the pre-MLP
  residual state at layers `20/24` via a cross-layer transfer or mediation
  probe.

Guardrail:

- This is an all-shard top-4 panel, not a full atlas. It narrows the causal
  boundary from MLP output to MLP input, but it does not yet identify which
  upstream operation writes the pre-MLP residual state.
