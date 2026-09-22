# Pre-Onset L16 Attention Source Routing

Date: 2026-06-12

## Scope

This slice follows `2026-06-12_pre_onset_rank_moving_site_localization.md`.
The previous reducer localized the clearest rank-moving pre-onset signal to the
layer-16 residual boundary, especially CTM-repair/MTC-damage coordinate targets.

This run asks a narrower routing question:

```text
At L16, do the clean CTM/MTC rank-moving targets route through visual source
heads, generated-history heads, or both?
```

The answer here is a source-category readout and source-mask comparison. It is
not yet a causal head patch or head ablation.

## Implementation

New reducer:

```text
src/analysis/autoregressive_duplication_mechanism/phase4_pre_onset_attention_source_summary.py
```

New CLI:

```text
scripts/analysis/run_autoregressive_duplication_phase4_pre_onset_attention_source_summary.py
```

New test:

```text
tests/analysis/autoregressive_duplication_mechanism/test_phase4_pre_onset_attention_source_summary.py
```

The reducer joins site-localization rows to attention-source rows by:

```text
checkpoint_label, record_idx, row_idx, phase, generated_token_index
```

It then summarizes source categories into four main families:

- `visual`;
- `generated_history`;
- `prompt_prefix`;
- `current_query`.

For family-level tables, source categories are summed within each
target/head/intervention before averaging. This matters because
`generated_history` is split across coord/text/structure/special buckets.

## Attention Source Readout Run

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \
python scripts/analysis/run_autoregressive_duplication_phase4_attention_source_shard.py \
  --token-windows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/pre_onset_patch_token_windows.jsonl \
  --region-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/pre_onset_patch_region_rows.jsonl \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/attention_source_l16_allheads \
  --device auto \
  --torch-dtype bfloat16 \
  --attn-implementation eager \
  --prompt-ordering random \
  --attention-layers 16 \
  --attention-heads 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --source-interventions no_op_control,duplicate_basin_mask
```

Readout summary:

```text
attention_source_row_count=4096
checkpoint_count=3
source_replay_case_count=8
attention_layers=[16]
attention_heads=[0..15]
```

Readout artifact root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/attention_source_l16_allheads
```

## Join Reduction Run

```bash
PYTHONDONTWRITEBYTECODE=1 \
python scripts/analysis/run_autoregressive_duplication_phase4_pre_onset_attention_source_summary.py \
  --site-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/site_localization/pre_onset_site_localization_rows.jsonl \
  --attention-source-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/attention_source_l16_allheads/attention_source_rows.jsonl \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/attention_source_l16_allheads/site_join_summary
```

Join artifact root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/attention_source_l16_allheads/site_join_summary
```

Produced:

```text
pre_onset_attention_source_join_rows.jsonl
pre_onset_attention_source_delta_rows.jsonl
pre_onset_attention_source_summary.json
pre_onset_attention_source_report.md
```

## Summary

Target counts:

| class/signature | targets |
| --- | ---: |
| all joined targets | 16 |
| `rank_moving_repair` | 11 |
| `ctm_repair_mtc_damage` | 9 |
| `mtc_repair_ctm_damage` | 2 |
| `same_direction_masked_to_control` | 5 |

Top L16 heads on the 9 CTM-repair/MTC-damage positives:

| readout | head | mean mass/delta |
| --- | ---: | ---: |
| control visual mass | 8 | 0.671559 |
| control visual mass | 13 | 0.610975 |
| control visual mass | 12 | 0.490524 |
| control generated-history mass | 1 | 0.577840 |
| control generated-history mass | 0 | 0.417895 |
| control generated-history mass | 4 | 0.389694 |
| visual control-minus-masked delta | 8 | 0.139792 |
| visual control-minus-masked delta | 13 | 0.060926 |
| generated-history control-minus-masked delta | 1 | 0.031004 |

Family averages over the 9 CTM-repair/MTC-damage positives:

| family | mean mass |
| --- | ---: |
| prompt prefix | 0.501527 |
| generated history | 0.196519 |
| visual | 0.174444 |
| current query | 0.127510 |

Slot-wise CTM/MTC family averages:

| slot | visual | generated history | prompt prefix | current query |
| --- | ---: | ---: | ---: | ---: |
| `x1` | 0.188830 | 0.184508 | 0.530778 | 0.095884 |
| `x2` | 0.129168 | 0.261065 | 0.478708 | 0.131059 |
| `y1` | 0.194080 | 0.150409 | 0.508733 | 0.146777 |
| `y2` | 0.162929 | 0.241795 | 0.437792 | 0.157484 |

## Read

The L16 source readout separates two routes:

1. Heads `8`, `13`, and `12` are high-visual-mass heads on the CTM/MTC
   positives. Under `duplicate_basin_mask`, the visual mass drops most strongly
   for heads `8` and `13`.
2. Head `1` is the clearest generated-history head. It has high control-side
   generated-history mass, but its duplicate-basin mask delta is smaller than
   the visual delta of head `8` and head `13`.
3. The slot pattern is not uniform. `x2` and `y2` carry more generated-history
   mass, while `x1` and `y1` are more balanced between visual and
   generated-history.
4. Case-level deltas are heterogeneous. The largest positive visual deltas are
   concentrated on `scissors` and `bottle`; `backpack` and some `person` rows
   can reverse direction under the same source mask. These are useful contrast
   cases rather than discardable noise.

Mechanism update:

```text
The L16 residual-boundary effect is not just a generated-history echo. The
clean CTM/MTC positives expose strong visual-source heads, especially L16H8 and
L16H13, whose attention mass is materially reduced by duplicate-basin masking.
Generated-history routing also exists, especially through L16H1, but the first
causal route to test should be visual-source head routing at L16, with H8/H13
as primary candidates and H1 as the history-loop comparator.
```

## Caveats

- This is a source-category attention readout, not a causal head patch.
- Source families are coarse. `generated_history` does not yet isolate same-desc
  rows from prior coordinate anchors.
- `prompt_prefix` mass is high across signatures, so a follow-up should check
  whether it is semantically meaningful routing or denominator/background mass.
- The CTM/MTC positive set has only 9 targets; it is a mechanism panel, not a
  validation-sized estimate.

## Next Step

Run a focused L16 head-level causal probe:

- primary visual-route candidates: `L16H8`, `L16H13`;
- history-loop comparator: `L16H1`;
- optional bridge head: `L16H12`;
- positive cases: the 9 CTM-repair/MTC-damage targets;
- contrast cases: the MTC/CTM and same-direction exceptions.

The decisive question is whether patching or score-biasing these heads moves
coord-token rank/logit outcomes in the same direction as the residual-boundary
L16 CTM/MTC effect.
