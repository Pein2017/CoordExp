---
title: Phase 4 Layer 17 Module Boundary Findings
date: 2026-06-10
status: active-evidence-note
owner: codex
evidence_scope: phase4-layer17-module-boundary-top4-allshards
---

# Phase 4 Layer 17 Module Boundary Findings

## Scope

This note records the layer-17 module-boundary follow-up launched after the
decoder transition fine sweep narrowed the key interval to layer `17` output /
layer `18` input.

Input manifest root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920
```

Main artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_layer17_module_boundary_top4_allshards_summary.json
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_layer17_module_boundary_top4_allshards_report.md
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_layer17_module_boundary_top4_allshards_directionality.json
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_layer17_module_boundary_top4_allshards_directionality.md
```

Probe settings:

- eight shard jobs launched across GPUs `0..7`;
- `target_top_k=4`;
- patch layer `17`;
- patch sites `decoder_layer_input,self_attn_input,self_attn,post_attention_residual,post_attention_norm,mlp_input,mlp,decoder_layer`;
- patch directions `masked_to_control,control_to_masked`;
- `top_k=8`.

Aggregate counts:

- `summary_count=8`;
- `residual_patch_row_count=768`;
- `targeted_replay_case_count=2`;
- `checkpoint_count_sum=2`;
- paired directionality rows `32`.

## Primary Anchor

For the primary anchor `none_latest_ckpt32|post_y1/pre_x2`, layer `17` module
boundaries show a non-monotonic causal pattern:

| patch site | interpretation | repair | damage | rank repair | rank damage |
| --- | --- | ---: | ---: | ---: | ---: |
| `decoder_layer_input` | asymmetric repair | `0.010274` | `0.000000` | `21.917` | `-1.667` |
| `self_attn_input` | asymmetric repair | `0.007815` | `0.000000` | `19.833` | `-2.500` |
| `self_attn` | symmetric damage candidate | `0.015438` | `0.009726` | `22.833` | `11.250` |
| `post_attention_residual` | asymmetric repair | `0.010906` | `0.000000` | `18.667` | `-1.917` |
| `post_attention_norm` | asymmetric repair | `0.010906` | `0.000000` | `18.667` | `-1.917` |
| `mlp_input` | asymmetric repair | `0.010906` | `0.000000` | `18.667` | `-1.917` |
| `mlp` | asymmetric repair | `0.010906` | `0.000000` | `18.667` | `-1.917` |
| `decoder_layer` | symmetric damage candidate | `0.014757` | `0.013846` | `21.167` | `18.500` |

The hook semantics matter:

- `decoder_layer_input`, `self_attn_input`, `post_attention_residual`, and
  `mlp_input` are module-input patch sites.
- `self_attn`, `mlp`, and `decoder_layer` are module-output patch sites.
- `post_attention_residual` and `post_attention_norm` both patch the
  post-attention layernorm module boundary in the current implementation.

## Mechanism Update

The layer-17 result does not support a simple "MLP creates the basin" story.
Instead:

1. Before self-attention, the state is repair-capable but asymmetric.
2. The self-attention output is the first layer-17 module site that shows
   strong repair and nontrivial reverse damage for the primary anchor.
3. The intermediate post-attention and MLP boundary probes revert to an
   asymmetric-repair signature.
4. The full decoder-layer output again shows a strong bidirectional basin.

This suggests that layer-17 self-attention exposes or writes a basin-relevant
component, but the final decoder-layer output contains an additional wrapped or
residual effect that is not explained by the MLP site alone under the current
patch granularity.

## Cross-Anchor Context

The `post_x1/pre_y1` windows remain useful as robustness checks, but they are
less clean for origin claims:

- `none_latest_ckpt32|post_x1/pre_y1` shows symmetric effects at
  `self_attn_input`, `self_attn`, `decoder_layer_input`, and `decoder_layer`,
  but the directionality is weaker and less phase-specific than the primary
  anchor.
- `aux_latest_ckpt32|post_x1/pre_y1` shows strong reverse damage at
  `self_attn_input` and `self_attn`, with output-layer symmetry also present.
- `aux_latest_ckpt32|post_y1/pre_x2` remains mixed or weak, so it should not
  anchor the origin picture.

## Next Probes

Recommended deterministic branch:

- run layer-17 head-level or grouped attention patching, anchored on
  `none_latest_ckpt32|post_y1/pre_x2`;
- include self-attention input/output controls so head effects can be compared
  against the boundary-level self-attention repair/damage signature;
- if attention heads localize cleanly, follow with Q/K/V or attention-pattern
  analysis for the high-effect heads;
- if attention heads do not localize, inspect decoder-layer wrapper/residual
  behavior around the full layer output before attributing the basin to MLP.

Guardrail:

- The repeated equality among `post_attention_residual`, `post_attention_norm`,
  `mlp_input`, and `mlp` should be treated as an implementation-level patch
  granularity clue. It should not be over-read as proof that those mathematical
  subcomponents are causally identical.
