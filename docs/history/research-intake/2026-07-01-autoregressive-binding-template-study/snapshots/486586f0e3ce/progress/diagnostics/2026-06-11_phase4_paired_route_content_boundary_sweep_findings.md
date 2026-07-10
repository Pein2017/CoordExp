# Phase 4 Paired Route/Content Boundary Sweep Findings

## Scope

This note records a boundary sweep over the 14-row paired constructive vs
destructive manifest. It uses the same layer-17 head-1 duplicate-basin
route/content effects, but patches them at three boundaries:

- `self_attn_output`
- `post_attention_residual`
- `post_attention_norm`

Input manifest:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_paired_case_manifest_layer17_head1_duplicate_constructive_destructive/paired_case_manifest_rows.jsonl
```

Output artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_paired_route_content_boundary_sweep_layer17_head1_full_manifest/paired_route_content_probe_rows.jsonl
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_paired_route_content_boundary_sweep_layer17_head1_full_manifest/phase4_paired_route_content_probe_summary.json
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_paired_route_content_boundary_sweep_layer17_head1_full_manifest/phase4_paired_route_content_boundary_sweep_analysis_summary.json
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_paired_route_content_boundary_sweep_layer17_head1_full_manifest/phase4_paired_route_content_boundary_sweep_analysis_report.md
```

Run configuration:

- Candidate manifest rows: `14`
- Replay cases: `3`
- Probe rows: `126`
- Checkpoint: `none_latest_ckpt32`
- Phase: `post_y1/pre_x2`
- Region/component: `duplicate_basin`
- Layer/head: `17/1`
- Patch direction: `masked_to_control`
- Effect kinds: `total_delta`, `route_delta_masked_values`,
  `value_delta_control_route`

## Constructive Core

Constructive rows are the eight record-33 wine-glass rows.

At `self_attn_output`, the signal is strongly route-dominant:

| effect | prob recovery | radius-4 recovery | radius-8 recovery | expected abs-error recovery | rank recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| `route_delta_masked_values` | `0.027159` | `0.192719` | `0.334228` | `-14.6597` | `28.75` |
| `value_delta_control_route` | `0.000277` | `0.004091` | `0.005806` | `0.4323` | `1.125` |
| `total_delta` | `0.027560` | `0.190489` | `0.331801` | `-14.6318` | `29.125` |

At `post_attention_residual`, the same route delta becomes weakly harmful:

| effect | prob recovery | radius-4 recovery | radius-8 recovery | expected abs-error recovery | rank recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| `route_delta_masked_values` | `-0.001568` | `-0.011602` | `-0.023059` | `1.0717` | `-5.75` |
| `value_delta_control_route` | `-0.000985` | `-0.006942` | `-0.013786` | `0.5111` | `-3.125` |
| `total_delta` | `-0.002167` | `-0.014668` | `-0.028674` | `1.3752` | `-6.25` |

At `post_attention_norm`, the route delta becomes strongly destructive:

| effect | prob recovery | radius-4 recovery | radius-8 recovery | expected abs-error recovery | rank recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| `route_delta_masked_values` | `-0.011416` | `-0.084602` | `-0.168378` | `362.336` | `-597.75` |
| `value_delta_control_route` | `-0.002046` | `-0.012746` | `-0.027924` | `5.3992` | `-3.875` |
| `total_delta` | `-0.011124` | `-0.082571` | `-0.164298` | `339.497` | `-543.625` |

## Destructive Core

Destructive rows are records `114` and `50`.

At `self_attn_output`, the effects are smaller than constructive rows but still
positive on average:

| effect | prob recovery | radius-4 recovery | radius-8 recovery | expected abs-error recovery | rank recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| `route_delta_masked_values` | `0.002638` | `0.020507` | `0.033991` | `-3.9390` | `3.667` |
| `value_delta_control_route` | `0.000649` | `0.004638` | `0.011643` | `-10.9931` | `16.167` |
| `total_delta` | `0.004670` | `0.029671` | `0.052714` | `-15.7677` | `21.167` |

At `post_attention_residual`, destructive rows stay mixed:

| effect | prob recovery | radius-4 recovery | radius-8 recovery | expected abs-error recovery | rank recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| `route_delta_masked_values` | `0.001523` | `0.011894` | `0.020137` | `-1.2866` | `3.333` |
| `value_delta_control_route` | `-0.001566` | `-0.012241` | `-0.022902` | `4.9354` | `14.333` |
| `total_delta` | `0.000461` | `-0.001117` | `-0.003301` | `3.6464` | `15.667` |

At `post_attention_norm`, destructive rows are also harmful:

| effect | prob recovery | radius-4 recovery | radius-8 recovery | expected abs-error recovery | rank recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| `route_delta_masked_values` | `-0.004425` | `-0.031217` | `-0.055283` | `66.9305` | `-111.667` |
| `value_delta_control_route` | `-0.002758` | `-0.019625` | `-0.036328` | `89.9565` | `-160.167` |
| `total_delta` | `-0.002644` | `-0.020189` | `-0.036682` | `80.7032` | `-127.833` |

## Interpretation

The route-dominant constructive signal is highly boundary-local:

```text
self_attn_output:      strong coordinate repair, route-dominant
post_attention_residual: weakly harmful
post_attention_norm:   strongly destructive
```

This rules out the simple story that a beneficial value/route residual vector is
inserted into the stream and then cleanly carried downstream. The repair signal
is visible at the attention output interface, but the same vector is not a safe
additive intervention after residual addition or after post-attention norm.

Mechanistically, this points to a boundary-sensitive interaction:

1. The attention output contribution carries a useful route-selected coordinate
   basin signal.
2. That signal likely depends on being composed through the layer's native
   residual and normalization context.
3. Patching the same projected vector after the residual or norm boundary
   miscalibrates the coordinate slot, especially for the constructive rows.

The finding also explains the earlier apparent contradiction between the
manifest route/value hints and the first compact `post_attention_residual`
smoke: they were both mechanically valid, but they targeted different boundary
semantics.

## Next Probe

The next high-value causal probe should stay at `self_attn_output` and move from
within-row masked-to-control repair to cross-row swaps:

- constructive wine-glass route into destructive record-114 rows;
- destructive record-114 route into constructive wine-glass rows;
- value-only swap as a control;
- joint route+value swap to test nonlinearity.

The readout should distinguish:

- generic coordinate confidence changes;
- movement toward the target row's coordinate basin;
- movement toward the donor row's coordinate basin.

This is the right next step because the boundary sweep says the route-selected
attention output is the native intervention surface; cross-row swaps can now ask
whether constructive vs destructive behavior is caused by the selected route
pattern itself or by row-local downstream context.
