# Pre-Onset L16H8 Key-State Patch

Date: 2026-06-12

## Scope

This slice follows `2026-06-12_pre_onset_l16_qk_origin_heads.md`.
The Q/K origin probe showed that the strongest H8 route/content bridge case,
`aux_latest_ckpt32 record79 row9 bottle x2`, has key-side positive score origin:

```text
H8 x2 bottle:
key_delta_control_query=+1.576811
query_delta_masked_keys=-1.079660
```

This run tests whether directly transplanting H8 duplicate-basin key state is
enough to reproduce the larger route/content repair.

## Run

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \
python scripts/analysis/run_autoregressive_duplication_phase4_key_state_patch_shard.py \
  --token-windows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/pre_onset_patch_token_windows.jsonl \
  --region-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/pre_onset_patch_region_rows.jsonl \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/key_state_patch_l16_head8 \
  --device auto \
  --torch-dtype bfloat16 \
  --attn-implementation eager \
  --prompt-ordering random \
  --attention-layer 16 \
  --attention-head 8 \
  --region-kind duplicate_basin \
  --top-k 8 \
  --patch-directions masked_to_control,control_to_masked \
  --patch-components duplicate_basin
```

Run summary:

```text
attention_score_bias_patch_row_count=26
checkpoint_count=3
replay_case_count=8
region_row_count=56
score_bias_modes=["key_state_control"]
```

Artifact root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/key_state_patch_l16_head8
```

## Durable Reducer

New reducer:

```text
src/analysis/autoregressive_duplication_mechanism/phase4_pre_onset_key_state_summary.py
```

New CLI:

```text
scripts/analysis/run_autoregressive_duplication_phase4_pre_onset_key_state_summary.py
```

New test:

```text
tests/analysis/autoregressive_duplication_mechanism/test_phase4_pre_onset_key_state_summary.py
```

Reduction command:

```bash
PYTHONDONTWRITEBYTECODE=1 \
python scripts/analysis/run_autoregressive_duplication_phase4_pre_onset_key_state_summary.py \
  --site-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/site_localization/pre_onset_site_localization_rows.jsonl \
  --key-state-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/key_state_patch_l16_head8/attention_score_bias_patch_rows.jsonl \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/key_state_patch_l16_head8/site_join_summary
```

Summary artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/key_state_patch_l16_head8/site_join_summary
```

Produced:

```text
pre_onset_key_state_join_rows.jsonl
pre_onset_key_state_summary.json
pre_onset_key_state_report.md
```

The summary joins 13 materialized targets:

```text
ctm_repair_mtc_damage=8
mtc_repair_ctm_damage=2
same_direction_masked_to_control=3
```

## CTM/MTC Summary

For CTM-repair/MTC-damage rows:

| direction | targets | attn | prob | rank | mass16 | abs-error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `masked_to_control` | 8 | -0.031106 | 0.000576 | 22.75 | 0.0085 | -10.17 |
| `control_to_masked` | 8 | -0.142679 | -0.001633 | 4.12 | -0.0274 | 7.93 |

Top CTM/MTC `masked_to_control` rows:

| case | slot | prob | rank | mass16 | abs-error |
| --- | --- | ---: | ---: | ---: | ---: |
| `none_latest_ckpt32 r114 row5 person` | `y1` | 0.001822 | 101 | 0.0448 | -25.79 |
| `none_latest_ckpt32 r114 row3 backpack` | `x1` | 0.001601 | 7 | 0.0035 | -1.98 |
| `aux_latest_ckpt32 r79 row9 bottle` | `x2` | 0.000639 | 9 | 0.0058 | -1.30 |
| `aux_latest_ckpt32 r50 row8 person` | `x1` | 0.000397 | 17 | 0.0075 | -12.47 |
| `aux_latest_ckpt32 r47 row1 scissors` | `y1` | 0.000089 | 48 | 0.0024 | -26.42 |

For the key bridge case:

```text
H8 key-state masked_to_control, aux_latest_ckpt32 r79 row9 bottle x2:
prob recovery=0.000639
rank recovery=9
mass16 recovery=0.0058
abs-error recovery=-1.30
```

Compare with the stronger route/content result for the same case:

```text
H8 value_delta_control_route, aux_latest_ckpt32 r79 row9 bottle x2:
prob recovery=0.010477
rank recovery=65
mass16 recovery=0.2350
```

## Read

This is a useful negative result:

1. Direct H8 duplicate-basin key-state patching is not sufficient to reproduce
   the large x2 bottle route/content repair.
2. The x2 bottle key-state patch moves in the right direction, but only weakly:
   about `6%` of the route/content probability recovery and a much smaller
   mass16 recovery.
3. The biggest key-state gains occur in no-aligner contrast rows
   (`record114 y1 person`, `record114 x1 backpack`), not in the auxiliary
   x2 bottle bridge case. This suggests H8 key state is a diagnostic origin
   signal but not the full causal payload for the auxiliary bridge.
4. The asymmetric `control_to_masked` result for x2 bottle is strong and
   negative (`prob=-0.007330`, `mass16=-0.0903`), which still supports H8 key
   state as a vulnerability surface, even if key transplant alone is not enough
   for full repair.

Mechanism update:

```text
The current evidence separates origin from sufficiency:

H8 Q/K origin says duplicate-basin key/source state helps explain the score
compatibility of the x2 bottle repair.

H8 key-state patch says key state alone does not carry enough payload to
reproduce the route/content repair.

Therefore, the active mechanism should move from "key state causes repair" to
"key state gates route compatibility, while value/content or downstream
attention-output mixing carries most of the coordinate-basin effect."
```

## Caveats

- The key-state patch materialized 13 of the 16 site-localization targets.
- This is still a tiny mechanism panel, not a population estimate.
- The direct key-state intervention may be too narrow if Q/K score compatibility
  depends on paired query-key geometry rather than key vectors alone.

## Next Step

The next strongest test is not another key-only patch. It should target the
value/content side of H8:

```text
Run or reduce value-source/value-basin patching for L16H8 on the same panel,
with special attention to aux_latest_ckpt32 r79 row9 bottle x2.
```

The decisive question becomes:

```text
Does H8's duplicate-basin value/content contribution itself point into the
coordinate-basin repair direction, and can that explain the route/content
effect size that key-state patching cannot?
```
