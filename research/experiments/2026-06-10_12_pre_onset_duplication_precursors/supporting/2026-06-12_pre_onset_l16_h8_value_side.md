# Pre-Onset L16H8 Value-Side Probe

Date: 2026-06-12

## Scope

This slice follows `2026-06-12_pre_onset_l16_h8_key_state_patch.md`.
The key-state patch separated Q/K origin from sufficiency:

```text
H8 key-state patch for aux_latest_ckpt32 r79 row9 bottle x2:
prob recovery=0.000639
rank recovery=9
mass16 recovery=0.0058
```

That was directionally positive but far weaker than the route/content
`value_delta_control_route` result:

```text
H8 route/content value_delta_control_route for the same case:
prob recovery=0.010477
rank recovery=65
mass16 recovery=0.2350
```

This run tests the value side directly:

1. value-source patching: does swapping H8 duplicate-basin value contribution
   move coord logits?
2. value-basin projection: does the raw value delta geometrically point into the
   target coordinate basin?

## Runs

Value-source patch:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \
python scripts/analysis/run_autoregressive_duplication_phase4_value_source_patch_shard.py \
  --token-windows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/pre_onset_patch_token_windows.jsonl \
  --region-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/pre_onset_patch_region_rows.jsonl \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/value_source_patch_l16_head8 \
  --device auto \
  --torch-dtype bfloat16 \
  --attn-implementation eager \
  --prompt-ordering random \
  --attention-layers 16 \
  --attention-heads 8 \
  --source-region-kind duplicate_basin \
  --top-k 8 \
  --patch-directions masked_to_control,control_to_masked
```

Value-basin projection:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 \
python scripts/analysis/run_autoregressive_duplication_phase4_value_basin_projection_shard.py \
  --token-windows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/pre_onset_patch_token_windows.jsonl \
  --region-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/pre_onset_patch_region_rows.jsonl \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/value_basin_projection_l16_head8 \
  --device auto \
  --torch-dtype bfloat16 \
  --attn-implementation eager \
  --prompt-ordering random \
  --attention-layer 16 \
  --attention-head 8 \
  --region-kind duplicate_basin \
  --patch-components duplicate_basin
```

Run summaries:

```text
value_source_patch_row_count=26
value_basin_projection_row_count=16
checkpoint_count=3
replay_case_count=8
region_row_count=56
```

Artifact roots:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/value_source_patch_l16_head8
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/value_basin_projection_l16_head8
```

## Durable Reducer

New reducer:

```text
src/analysis/autoregressive_duplication_mechanism/phase4_pre_onset_value_side_summary.py
```

New CLI:

```text
scripts/analysis/run_autoregressive_duplication_phase4_pre_onset_value_side_summary.py
```

New test:

```text
tests/analysis/autoregressive_duplication_mechanism/test_phase4_pre_onset_value_side_summary.py
```

Reduction command:

```bash
PYTHONDONTWRITEBYTECODE=1 \
python scripts/analysis/run_autoregressive_duplication_phase4_pre_onset_value_side_summary.py \
  --site-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/site_localization/pre_onset_site_localization_rows.jsonl \
  --value-source-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/value_source_patch_l16_head8/value_source_patch_rows.jsonl \
  --value-basin-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/value_basin_projection_l16_head8/value_basin_projection_rows.jsonl \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/value_side_l16_head8_summary
```

Summary artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/value_side_l16_head8_summary
```

Produced:

```text
pre_onset_value_side_join_rows.jsonl
pre_onset_value_side_summary.json
pre_onset_value_side_report.md
```

## Value-Source Result

For CTM-repair/MTC-damage rows:

| direction | targets | prob | rank | mass16 | abs-error |
| --- | ---: | ---: | ---: | ---: | ---: |
| `masked_to_control` | 8 | 0.002045 | 39.75 | 0.0327 | -13.43 |
| `control_to_masked` | 8 | -0.002184 | 5.50 | -0.0400 | 4.94 |

Top CTM/MTC `masked_to_control` value-source rows:

| case | slot | prob | rank | mass16 | abs-error |
| --- | --- | ---: | ---: | ---: | ---: |
| `aux_latest_ckpt32 r79 row9 bottle` | `x2` | 0.007929 | 62 | 0.1719 | -11.95 |
| `none_latest_ckpt32 r114 row3 backpack` | `x1` | 0.005225 | 16 | 0.0102 | -2.84 |
| `none_latest_ckpt32 r114 row5 person` | `y1` | 0.002766 | 141 | 0.0718 | -40.43 |
| `aux_latest_ckpt32 r50 row8 person` | `x1` | 0.000658 | 23 | 0.0111 | -19.85 |

For the focal x2 bottle case:

```text
H8 value-source masked_to_control:
prob recovery=0.007929
rank recovery=62
mass16 recovery=0.1719
abs-error recovery=-11.95
```

This recovers most of the previous route/content effect:

```text
route/content value_delta_control_route:
prob recovery=0.010477
rank recovery=65
mass16 recovery=0.2350
```

## Value-Basin Projection Result

For CTM/MTC rows:

| signature | targets | target delta | centered delta | radius16 delta | target cosine | projected L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ctm_repair_mtc_damage` | 9 | 0.003408 | 0.000595 | 0.002475 | -0.001323 | 6.7744 |
| `mtc_repair_ctm_damage` | 2 | -0.025465 | 0.001183 | -0.025406 | 0.015582 | 9.7105 |

Top CTM/MTC target-delta rows:

| case | slot | target delta | centered delta | radius16 delta | target cosine |
| --- | --- | ---: | ---: | ---: | ---: |
| `aux_latest_ckpt32 r47 row1 scissors` | `y1` | 0.043036 | 0.000685 | 0.042942 | 0.000568 |
| `aux_latest_ckpt32 r47 row1 scissors` | `y2` | 0.042467 | 0.007601 | 0.042478 | 0.011876 |
| `none_latest_ckpt32 r114 row5 person` | `y1` | 0.004674 | -0.010858 | 0.004674 | -0.019409 |
| `aux_latest_ckpt32 r79 row9 bottle` | `x2` | -0.008292 | -0.004840 | -0.008329 | -0.006567 |

## Read

This is the strongest H8 causal evidence so far:

1. Value-source patching nearly reproduces the focal x2 bottle route/content
   repair. This is much stronger than key-state patching:

```text
x2 bottle H8:
key-state patch prob recovery       0.000639
value-source patch prob recovery    0.007929
route/content value patch recovery  0.010477
```

2. The H8 mechanism is now best described as:

```text
duplicate-basin key/source state gates route compatibility,
but duplicate-basin value/content carries most of the coordinate-basin payload.
```

3. The value-basin projection is not a simple standalone explanation. For x2
   bottle, raw target-delta and radius16-delta are negative, even though the
   intervention-level value-source patch strongly repairs the coordinate
   distribution. This suggests the causal payload depends on contextual
   attention/output mixing and downstream readout, not only on the raw projected
   value vector pointing toward the target coordinate token.
4. The strongest raw value-basin target deltas are the scissors y-slots, which
   were strong visual-routing/source cases but are not the focal x2 bridge. This
   keeps the mechanism case-conditional rather than one universal H8 geometry.

Mechanism update:

```text
For the current pre-onset CTM/MTC bridge, H8 is a value-bearing visual route.
Q/K/key-state evidence identifies the duplicate-basin key side as the route
compatibility gate. Value-source patching shows that the actual coordinate-logit
repair is mostly carried by the value/content contribution, especially in the
aux_latest x2 bottle case. Raw value-basin projection is insufficient as a
standalone explanation.
```

## Caveats

- Value-source patching materialized 13 targets, while value-basin projection
  materialized all 16 site-localization targets.
- The focal conclusion is case-level: `aux_latest_ckpt32 r79 row9 bottle x2`.
- Value-basin projection depends on the chosen coordinate-output projection and
  can miss context-dependent downstream effects.

## Next Step

The immediate L16H8 chain is now coherent enough to pause this branch of the
duplication-onset probe and turn back toward the broader project goal:

```text
Use the same evidence style on false-negative windows:
does visual evidence exist but fail to bind through language/prefix guidance,
or is the visual source genuinely absent?
```

For duplication, the next optional deepening would be a smaller case-specific
H8 value-source ablation around `aux_latest_ckpt32 r79 row9 bottle x2`, but the
current evidence already supports the key/value split.
