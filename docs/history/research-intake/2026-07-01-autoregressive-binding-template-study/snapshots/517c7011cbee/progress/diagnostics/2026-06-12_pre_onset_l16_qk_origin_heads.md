# Pre-Onset L16 Q/K Origin Head Probe

Date: 2026-06-12

## Scope

This slice follows `2026-06-12_pre_onset_l16_route_content_heads.md`.
The route/content split identified `L16H8` as the strongest current bridge from
the L16 residual-boundary effect to coordinate-logit repair, especially for the
`aux_latest_ckpt32 record79 row9 bottle x2` case.

This run asks:

```text
For H8, is the x2 bottle route/content repair primarily caused by key-side
duplicate-basin state, query-side current-token state, or both?
```

`L16H13` is included as the smaller-magnitude visual-route comparator.

## Runs

Both heads used:

```text
token windows: /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/pre_onset_patch_token_windows.jsonl
region rows: /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/pre_onset_patch_region_rows.jsonl
layer: 16
region-kind: duplicate_basin
patch-components: duplicate_basin
```

Parallel launches:

```bash
CUDA_VISIBLE_DEVICES=0 ... --attention-head 8  --output-dir .../qk_route_origin_l16_head8
CUDA_VISIBLE_DEVICES=1 ... --attention-head 13 --output-dir .../qk_route_origin_l16_head13
```

Each run completed with:

```text
qk_route_origin_row_count=16
checkpoint_count=3
replay_case_count=8
region_row_count=56
```

Artifact roots:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/qk_route_origin_l16_head8
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/qk_route_origin_l16_head13
```

## Durable Reducer

New reducer:

```text
src/analysis/autoregressive_duplication_mechanism/phase4_pre_onset_qk_origin_summary.py
```

New CLI:

```text
scripts/analysis/run_autoregressive_duplication_phase4_pre_onset_qk_origin_summary.py
```

New test:

```text
tests/analysis/autoregressive_duplication_mechanism/test_phase4_pre_onset_qk_origin_summary.py
```

Reduction command:

```bash
PYTHONDONTWRITEBYTECODE=1 \
python scripts/analysis/run_autoregressive_duplication_phase4_pre_onset_qk_origin_summary.py \
  --site-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/site_localization/pre_onset_site_localization_rows.jsonl \
  --qk-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/qk_route_origin_l16_head8 \
  --qk-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/qk_route_origin_l16_head13 \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/qk_route_origin_l16_heads_summary
```

Summary artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/qk_route_origin_l16_heads_summary
```

Produced:

```text
pre_onset_qk_origin_join_rows.jsonl
pre_onset_qk_origin_summary.json
pre_onset_qk_origin_report.md
```

The reducer joins all 16 site-localization targets:

```text
ctm_repair_mtc_damage=9
mtc_repair_ctm_damage=2
same_direction_masked_to_control=5
```

Finite Q/K coverage is lower than row coverage because some duplicate-basin
source components have zero materialized source tokens:

| signature | head | finite/rows |
| --- | ---: | ---: |
| `ctm_repair_mtc_damage` | 8 | 8/9 |
| `ctm_repair_mtc_damage` | 13 | 8/9 |
| `mtc_repair_ctm_damage` | 8 | 2/2 |
| `mtc_repair_ctm_damage` | 13 | 2/2 |
| `same_direction_masked_to_control` | 8 | 3/5 |
| `same_direction_masked_to_control` | 13 | 3/5 |

## CTM/MTC Head Averages

For CTM-repair/MTC-damage rows:

| head | attn delta | score delta | logsumexp delta | query masked-keys | key control-query |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.069908 | -0.519466 | 1.357933 | -0.410708 | -0.108759 |
| 13 | 0.109685 | 0.114917 | 2.033305 | 0.701920 | -0.587003 |

The head average alone is not the mechanism read. H8 has mixed CTM/MTC cases:
positive aux-checkpoint bridge cases and negative no-aligner contrast cases.
The case-level decomposition is the useful signal.

## Case-Level Origin

Top CTM/MTC score-delta rows:

| head | case | slot | attn delta | score delta | query masked-keys | key control-query |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 13 | `aux_latest_ckpt32 r47 row1 scissors` | `y2` | 0.508705 | 2.623532 | 3.412554 | -0.789021 |
| 13 | `aux_latest_ckpt32 r47 row1 scissors` | `y1` | 0.694581 | 2.396568 | 2.778995 | -0.382426 |
| 8 | `aux_latest_ckpt32 r47 row3 person` | `x2` | 0.016935 | 1.283377 | 0.149071 | 1.134306 |
| 8 | `aux_latest_ckpt32 r50 row8 person` | `x1` | -0.021817 | 1.073793 | -0.123529 | 1.197322 |
| 13 | `aux_latest_ckpt32 r79 row9 bottle` | `x2` | 0.025360 | 0.614843 | -0.711380 | 1.326223 |
| 8 | `aux_latest_ckpt32 r79 row9 bottle` | `x2` | 0.118679 | 0.497150 | -1.079660 | 1.576811 |

Top CTM/MTC key-delta rows:

| head | case | slot | score delta | query masked-keys | key control-query | query control-keys | key masked-query |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 8 | `aux_latest_ckpt32 r79 row9 bottle` | `x2` | 0.497150 | -1.079660 | 1.576811 | 2.128250 | -1.631100 |
| 13 | `aux_latest_ckpt32 r79 row9 bottle` | `x2` | 0.614843 | -0.711380 | 1.326223 | 2.007376 | -1.392532 |
| 8 | `aux_latest_ckpt32 r50 row8 person` | `x1` | 1.073793 | -0.123529 | 1.197322 | 0.032786 | 1.041007 |
| 8 | `aux_latest_ckpt32 r47 row3 person` | `x2` | 1.283377 | 0.149071 | 1.134306 | -0.264898 | 1.548275 |

## Read

This adds the missing origin detail for the H8 bridge case:

1. For the strongest route/content case, `aux_latest_ckpt32 r79 row9 bottle x2`,
   H8's score repair is key-side dominant:
   - `key_delta_control_query=+1.576811`;
   - `query_delta_masked_keys=-1.079660`;
   - `key_delta_masked_query=-1.631100`;
   - `query_delta_control_keys=+2.128250`.
2. This means the H8 repair is not explained by simply moving the current-token
   query toward the control stream. The control-side key/source state over the
   duplicate-basin component is doing the favorable work for the actual
   query/control pairing.
3. H13 behaves differently on its strongest CTM/MTC rows: the large scissors
   score deltas are query-side dominated (`query_delta_masked_keys` around
   `+2.78` to `+3.41`) while `key_delta_control_query` is negative. H13 is
   therefore not the same mechanism as H8, even though both are visual-source
   heads.
4. H8's head-level mean is mixed because no-aligner contrast cases have negative
   key deltas. That heterogeneity is useful: H8 appears to be a conditional
   bridge for particular auxiliary-checkpoint bridge cases, not a globally
   positive duplicate-basin route.

Mechanism update:

```text
For the current strongest bridge case, the chain is now:

duplicate-basin visual key/source state
  -> L16H8 Q/K route compatibility
  -> H8 route/content output at self_attn_output
  -> coordinate-basin logit repair for x2 bottle.

H13 is a nearby visual head but its largest effects look query-side dominated,
so it should be treated as a comparator rather than the same causal route.
```

## Caveats

- Q/K origin is strongest at the case level; aggregate CTM/MTC means mix
  auxiliary bridge cases with no-aligner contrast cases.
- Several same-direction rows have no finite Q/K values because the duplicate
  basin has zero materialized source tokens.
- This is still a score-origin decomposition, not a direct key-state
  intervention.

## Next Step

Run a key-state patch for `L16H8`, prioritizing the x2 bottle case and the
CTM/MTC auxiliary bridge rows. The target question is:

```text
Does transplanting duplicate-basin key/source state at H8 reproduce the
route/content repair without needing a broader residual-state patch?
```

If yes, the mechanism chain becomes much more concrete. If no, the Q/K signal is
diagnostic but the causal payload still lives in value/content or downstream
mixing.
