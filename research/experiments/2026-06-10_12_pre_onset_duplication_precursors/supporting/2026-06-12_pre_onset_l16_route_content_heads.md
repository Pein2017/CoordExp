# Pre-Onset L16 Route/Content Head Probe

Date: 2026-06-12

## Scope

This slice follows `2026-06-12_pre_onset_l16_score_bias_heads.md`.
The score-bias probe left an ambiguity:

```text
Are L16H8/L16H13 merely visual routing sensors, while L16H12/L16H1 carry the
coordinate-logit consequence, or does the stronger value/content patch recover
the visual-head effect?
```

This run applies route/content patching to the same candidate heads:

- visual-source candidates: `L16H8`, `L16H13`;
- history comparator: `L16H1`;
- score-bias bridge head: `L16H12`.

The probe separates:

- `total_delta`;
- `route_delta_masked_values`;
- `value_delta_control_route`.

## Runs

All four heads used:

```text
token windows: /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/pre_onset_patch_token_windows.jsonl
region rows: /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/pre_onset_patch_region_rows.jsonl
layer: 16
region-kind: duplicate_basin
patch-components: duplicate_basin
patch-sites: self_attn_output,post_attention_residual
patch-directions: masked_to_control,control_to_masked
effect-kinds: total_delta,route_delta_masked_values,value_delta_control_route
```

Parallel launches:

```bash
CUDA_VISIBLE_DEVICES=0 ... --attention-head 8  --output-dir .../route_content_l16_head8
CUDA_VISIBLE_DEVICES=1 ... --attention-head 13 --output-dir .../route_content_l16_head13
CUDA_VISIBLE_DEVICES=2 ... --attention-head 1  --output-dir .../route_content_l16_head1
CUDA_VISIBLE_DEVICES=3 ... --attention-head 12 --output-dir .../route_content_l16_head12
```

Each run completed with:

```text
route_content_patch_row_count=192
checkpoint_count=3
replay_case_count=8
region_row_count=56
```

Artifact roots:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/route_content_l16_head8
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/route_content_l16_head13
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/route_content_l16_head1
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/route_content_l16_head12
```

## Durable Reducer

New reducer:

```text
src/analysis/autoregressive_duplication_mechanism/phase4_pre_onset_route_content_summary.py
```

New CLI:

```text
scripts/analysis/run_autoregressive_duplication_phase4_pre_onset_route_content_summary.py
```

New test:

```text
tests/analysis/autoregressive_duplication_mechanism/test_phase4_pre_onset_route_content_summary.py
```

Reduction command:

```bash
PYTHONDONTWRITEBYTECODE=1 \
python scripts/analysis/run_autoregressive_duplication_phase4_pre_onset_route_content_summary.py \
  --site-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/site_localization/pre_onset_site_localization_rows.jsonl \
  --route-content-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/route_content_l16_head8 \
  --route-content-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/route_content_l16_head13 \
  --route-content-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/route_content_l16_head1 \
  --route-content-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/route_content_l16_head12 \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/route_content_l16_heads_summary
```

Summary artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/route_content_l16_heads_summary
```

Produced:

```text
pre_onset_route_content_join_rows.jsonl
pre_onset_route_content_summary.json
pre_onset_route_content_report.md
```

The summary joins all 16 site-localization targets:

```text
ctm_repair_mtc_damage=9
mtc_repair_ctm_damage=2
same_direction_masked_to_control=5
```

## CTM/MTC Self-Attention Result

For the 9 CTM-repair/MTC-damage targets, `masked_to_control` at
`self_attn_output`:

| head | effect | prob | rank | mass16 | abs-error |
| ---: | --- | ---: | ---: | ---: | ---: |
| 8 | `total_delta` | 0.001758 | 35.56 | 0.0284 | -12.16 |
| 8 | `route_delta_masked_values` | 0.000846 | 34.22 | 0.0081 | -13.60 |
| 8 | `value_delta_control_route` | 0.001224 | 6.89 | 0.0260 | -1.30 |
| 13 | `total_delta` | 0.000337 | 10.11 | 0.0058 | -3.39 |
| 13 | `route_delta_masked_values` | 0.000245 | 10.78 | 0.0036 | -3.78 |
| 13 | `value_delta_control_route` | 0.000259 | 2.11 | 0.0037 | -0.93 |
| 1 | `total_delta` | 0.000198 | 3.56 | 0.0017 | -2.45 |
| 1 | `route_delta_masked_values` | 0.000185 | 5.22 | 0.0025 | -1.90 |
| 12 | `total_delta` | -0.000135 | -10.33 | -0.0038 | 0.84 |
| 12 | `route_delta_masked_values` | -0.000265 | -11.56 | -0.0030 | 3.43 |

The same CTM/MTC direction at `post_attention_residual` is much weaker:

| head | best effect | prob | rank | mass16 |
| ---: | --- | ---: | ---: | ---: |
| 8 | `total_delta` | 0.000350 | 12.33 | 0.0056 |
| 13 | `total_delta` | 0.000197 | 10.89 | 0.0026 |
| 1 | `total_delta` | 0.000132 | -0.33 | -0.0002 |
| 12 | `route_delta_masked_values` | 0.000090 | -0.22 | 0.0010 |

## Case-Level Peaks

Top `self_attn_output`, CTM/MTC, `masked_to_control` probability recoveries:

| head | effect | case | slot | prob | rank | mass16 |
| ---: | --- | --- | --- | ---: | ---: | ---: |
| 8 | `value_delta_control_route` | `aux_latest_ckpt32 r79 row9 bottle` | `x2` | 0.010477 | 65 | 0.2350 |
| 8 | `total_delta` | `aux_latest_ckpt32 r79 row9 bottle` | `x2` | 0.007330 | 56 | 0.1674 |
| 8 | `total_delta` | `none_latest_ckpt32 r114 row3 backpack` | `x1` | 0.005530 | 17 | 0.0114 |
| 8 | `route_delta_masked_values` | `none_latest_ckpt32 r114 row3 backpack` | `x1` | 0.004534 | 14 | 0.0090 |
| 8 | `route_delta_masked_values` | `none_latest_ckpt32 r114 row5 person` | `y1` | 0.002733 | 141 | 0.0650 |
| 13 | `value_delta_control_route` | `aux_latest_ckpt32 r79 row9 bottle` | `x2` | 0.001305 | 16 | 0.0308 |

## Read

This resolves the previous H12/H8 ambiguity in favor of H8:

1. `L16H8` is not merely a visual-source attention sensor. In route/content
   patching, it is the strongest CTM/MTC mover at the attention-output site.
2. H8's strongest x2 bottle effect is specifically `value_delta_control_route`,
   with probability recovery `0.010477`, rank recovery `65`, and mass16
   recovery `0.2350`. This is much larger than H12's score-bias-only peak and
   suggests the missing piece was value/content, not score routing alone.
3. `route_delta_masked_values` also matters for H8, especially for rank recovery
   and y1/person. So H8 is not purely value-only; it combines route and content.
4. `L16H13` follows the same sign as H8 but at smaller magnitude.
5. `L16H1` remains a weak history comparator. It is not carrying the dominant
   CTM/MTC route/content effect in this panel.
6. `L16H12` now looks less central for the CTM/MTC positives. Its earlier
   score-bias signal does not translate into stronger route/content repair.

Mechanism update:

```text
The strongest current bridge from L16 residual-boundary instability to
coordinate-logit repair is L16H8 at self_attn_output. Its effect is especially
clear on the x2 bottle bridge case and decomposes into a large value/content
component plus a route component. This supports a model where duplicate-onset
instability is not only a bad attention-score basin; the basin gates a
semantically useful head output whose value/content contribution can repair
or damage coordinate-basin logits.
```

## Caveats

- The panel remains small: 9 CTM/MTC positives, 2 MTC/CTM exceptions, and 5
  same-direction controls.
- The largest effect is case-local (`x2 bottle`), so this should be treated as
  a mechanistic candidate, not a population estimate.
- The `post_attention_residual` attenuation suggests downstream mixing or
  normalization changes the visible effect. This does not invalidate H8, but it
  means the exact transfer site matters.
- The reducer reports direction-specific metrics: `masked_to_control` uses
  recovery fields; `control_to_masked` uses damage fields.

## Next Step

Run Q/K origin for `L16H8` first, then optionally `L16H13`.

The decisive question:

```text
For H8, is the x2 bottle route/content repair primarily caused by key-side
duplicate-basin state, query-side current-token state, or both?
```

If Q/K origin points to key-side duplicate-basin state again, the active chain
becomes:

```text
duplicate-basin visual key state -> H8 route/content output -> L16 residual
boundary -> coord-basin logit repair/damage
```
