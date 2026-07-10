# Phase 4 Paired Route/Content Probe Findings

## Scope

This note records the paired manifest-filtered route/content causal probe for
the layer-17 head-1 duplicate-basin circuit.

The probe reruns the existing route/content patcher only on the compact
constructive/destructive manifest:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_paired_case_manifest_layer17_head1_duplicate_constructive_destructive/paired_case_manifest_rows.jsonl
```

The manifest had to include `pred_token_trace_path`; that field is now part of
the paired case manifest contract because replay reconstruction needs the full
generated-token trace.

## Artifacts

Comparable primary artifact, using `patch_site=self_attn_output`:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_paired_route_content_probe_layer17_head1_full_manifest_self_attn_output/paired_route_content_probe_rows.jsonl
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_paired_route_content_probe_layer17_head1_full_manifest_self_attn_output/phase4_paired_route_content_probe_summary.json
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_paired_route_content_probe_layer17_head1_full_manifest_self_attn_output/phase4_paired_route_content_probe_analysis_summary.json
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_paired_route_content_probe_layer17_head1_full_manifest_self_attn_output/phase4_paired_route_content_probe_analysis_report.md
```

Smoke artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_paired_route_content_probe_layer17_head1_smoke_constructive
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_paired_route_content_probe_layer17_head1_smoke_destructive
```

An initial smoke used `patch_site=post_attention_residual`; it passed
mechanically but produced a different sign pattern. The historical full
route/content artifact for these rows used `self_attn_output`, so the
interpretation below uses `self_attn_output` for comparability.

## Run Configuration

- Candidate manifest rows: `14`
- Replay cases: `3`
- Probe rows: `42`
- Checkpoint: `none_latest_ckpt32`
- Phase: `post_y1/pre_x2`
- Region/component: `duplicate_basin`
- Layer/head: `17/1`
- Patch direction: `masked_to_control`
- Patch site: `self_attn_output`
- Effect kinds:
  - `total_delta`
  - `route_delta_masked_values`
  - `value_delta_control_route`

## Summary

Constructive core, `n=8`:

| effect | prob recovery | radius-4 recovery | radius-8 recovery | expected abs-error recovery | rank recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| `route_delta_masked_values` | `0.027159` | `0.192719` | `0.334228` | `-14.6597` | `28.75` |
| `value_delta_control_route` | `0.000277` | `0.004091` | `0.005806` | `0.4323` | `1.125` |
| `total_delta` | `0.027560` | `0.190489` | `0.331801` | `-14.6318` | `29.125` |

Destructive core, `n=6`:

| effect | prob recovery | radius-4 recovery | radius-8 recovery | expected abs-error recovery | rank recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| `route_delta_masked_values` | `0.002638` | `0.020507` | `0.033991` | `-3.9390` | `3.667` |
| `value_delta_control_route` | `0.000649` | `0.004638` | `0.011643` | `-10.9931` | `16.167` |
| `total_delta` | `0.004670` | `0.029671` | `0.052714` | `-15.7677` | `21.167` |

## Interpretation

For the constructive wine-glass rows, the patch effect is almost entirely
route-delta driven at the attention-output boundary. The value-only term is
small, while route-only and total are nearly identical in probability, local
coordinate mass, and rank recovery.

For the destructive rows, the effects are much smaller and more heterogeneous:

- Route-only recovery is an order of magnitude smaller than constructive route
  recovery.
- Value-only contributes little probability but can contribute substantial
  expected-error/rank movement in some rows.
- Total recovery is larger than either component alone in the aggregate, which
  suggests nonlinear interaction or row-level heterogeneity rather than a clean
  additive value carrier.

The key refinement is that "route dominance" is boundary-specific. At
`self_attn_output`, constructive repair is route-dominant and matches the
historical artifact. At `post_attention_residual`, the same compact probe showed
different signs, so downstream residual/norm placement may transform or even
invert the apparent effect. That boundary dependence is now a mechanism target,
not just a plumbing detail.

## Next Step

The next high-value probe is a boundary sweep over the same 14-row manifest:

```text
patch_sites=self_attn_output,post_attention_residual,post_attention_norm
effect_kinds=total_delta,route_delta_masked_values,value_delta_control_route
patch_direction=masked_to_control
```

This should answer whether the route-dominant constructive signal is preserved,
attenuated, or transformed across the attention output, residual stream, and
post-attention norm boundary. If the boundary sweep is stable, then the next
cross-row experiment should swap route/value terms between constructive
wine-glass rows and destructive record-114 rows.
