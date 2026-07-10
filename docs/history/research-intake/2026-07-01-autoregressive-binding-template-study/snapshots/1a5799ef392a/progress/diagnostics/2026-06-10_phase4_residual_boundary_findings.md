---
title: Phase 4 Residual Boundary Patch Findings
date: 2026-06-10
status: active-evidence-note
owner: codex
evidence_scope: phase4-residual-boundary-top4-allshards
---

# Phase 4 Residual Boundary Patch Findings

## Scope

This note records the first residual-boundary patch probe around layers `20`
and `24` for the autoregressive duplication mechanism study. It follows the
MLP input/output result, which showed that the repair-relevant state is already
present at the selected token before the MLP computation.

Input manifest root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920
```

Main artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_residual_boundary_layers_20_24_top4_allshards_v4_summary.json
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_residual_boundary_layers_20_24_top4_allshards_v4_report.md
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_residual_boundary_layers_20_24_top4_allshards_v4_directionality.json
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_residual_boundary_layers_20_24_top4_allshards_v4_directionality.md
```

Probe settings:

- eight shard jobs launched across GPUs `0..7`;
- `target_top_k=4`;
- patch layers `20,24`;
- patch sites `decoder_layer_input,decoder_layer,self_attn_input,self_attn,post_attention_residual,post_attention_norm,mlp_input`;
- patch directions `masked_to_control,control_to_masked`;
- `top_k=8`.

Aggregate counts:

- `summary_count=8`;
- `residual_patch_row_count=1344`;
- `targeted_replay_case_count=2`;
- `checkpoint_count_sum=2`;
- paired directionality rows `56`.

Invalid/interrupted artifact warning:

- Prefix `phase4_residual_boundary_layers_20_24_top4_allshards` failed because
  the first input pre-hook did not handle keyword-only module calls.
- Prefix `phase4_residual_boundary_layers_20_24_top4_allshards_v2` failed
  because `with_kwargs=True` pre-hooks must return `(args, kwargs)` even when
  the patched tensor came from positional args.
- Prefix `phase4_residual_boundary_layers_20_24_top4_allshards_v3` was
  interrupted before shards `03/04` wrote summaries.
- Prefix `phase4_residual_boundary_layers_20_24_top4_allshards_v4` is the
  valid completed run.

## Findings

The main anchor `none_latest_ckpt32|post_y1/pre_x2` shows that the strong
repair state is already present at decoder-layer input for layers `20/24`.
Patching the input to the whole decoder layer repairs nearly as strongly as
patching the decoder-layer output.

Anchor masked-to-control repair:

- `decoder_layer_input|layer=20`: `prob_repair=0.014062`,
  `rank_repair=18.000`.
- `decoder_layer|layer=20`: `prob_repair=0.013875`,
  `rank_repair=17.917`.
- `decoder_layer_input|layer=24`: `prob_repair=0.013935`,
  `rank_repair=18.083`.
- `decoder_layer|layer=24`: `prob_repair=0.013667`,
  `rank_repair=17.917`.

The self-attention boundary does not repair the masked path at the same anchor:

- `self_attn_input|layer=20`: `prob_repair=-0.000408`,
  `rank_repair=-1.417`.
- `self_attn|layer=20`: `prob_repair=-0.000861`,
  `rank_repair=-3.167`.
- `self_attn_input|layer=24`: `prob_repair=-0.000207`,
  `rank_repair=-0.417`.
- `self_attn|layer=24`: `prob_repair=-0.000224`,
  `rank_repair=-0.583`.

The post-attention, normalized post-attention, and MLP-input sites remain
identical in this panel:

- `post_attention_residual|layer=20`, `post_attention_norm|layer=20`, and
  `mlp_input|layer=20` all have `prob_repair=0.006170`,
  `rank_repair=10.917`.
- `post_attention_residual|layer=24`, `post_attention_norm|layer=24`, and
  `mlp_input|layer=24` all have `prob_repair=0.008964`,
  `rank_repair=14.583`.

The reverse direction has the same broad split. At
`none_latest_ckpt32|post_y1/pre_x2|control_to_masked`, decoder-layer input and
decoder-layer output carry strong damage, while attention and post-attention
sites behave differently:

- `decoder_layer_input|layer=20`: `prob_damage=0.014434`,
  `rank_damage=20.417`.
- `decoder_layer|layer=20`: `prob_damage=0.013179`,
  `rank_damage=16.833`.
- `self_attn_input|layer=20`: `prob_damage=0.000255`,
  `rank_damage=0.000`.
- `post_attention_residual|layer=20`: `prob_damage=0.000961`,
  `rank_damage=-0.833`.

## Mechanism Update

This result moves the boundary earlier than layer-20/24 submodule computation.
For the strongest anchor, the repair-relevant residual state is already present
at the input to decoder layers `20/24`. The layer's attention and MLP
submodules do not appear to newly create the main repair state at those layers.

Current best picture:

1. Layer-16 grouped attention does not directly reconstruct the masked-to-control
   coordinate-slot basin.
2. MLP input and output at layers `20/24` are equivalent, so the MLP computation
   itself is not the origin.
3. Decoder-layer input at layers `20/24` already carries nearly the full
   decoder-layer repair/damage signal.
4. Therefore, the main residual basin is likely written before layer `20`,
   then carried forward through the residual stream into layers `20/24`.

This changes the next question from "which submodule inside layer 20/24 writes
the state?" to "at which earlier layer does the decoder-layer input first
become patch-causal?"

## Next Probes

Recommended deterministic branch:

- run a decoder-layer-input layer sweep over earlier layers, especially
  `12,16,18,20,22,24`, on the same top-4 anchor windows;
- include decoder-layer output in the same panel so the first layer where
  input and output become causal can be distinguished;
- if layer `16` input/output is the transition, compare it with the already
  observed layer-16 attention-head routing evidence.

Guardrail:

- This is an all-shard top-4 panel. It is strong enough to redirect the next
  probe earlier in the residual stream, but it does not yet identify the first
  layer where the basin appears.
