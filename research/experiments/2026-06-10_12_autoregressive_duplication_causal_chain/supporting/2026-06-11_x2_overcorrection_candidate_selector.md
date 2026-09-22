# X2 Overcorrection Candidate Selector

Date: 2026-06-11

## Scope

This note records a post-hoc candidate selector for future causal probes of the x2
overcorrection mechanism. It uses the enriched coordinate band-flow rows and does not
launch any GPU jobs.

The selector targets rows matching:

- `coord_slot == x2`;
- generated coordinate is above the target;
- masked top1 is exact or near target;
- patched top1 is below target;
- wide-radius target-near mass decreases.

This captures the mechanism from the previous note: the model has target-near x2
available in the masked readout, but the patch pushes the state downward into a
lower-coordinate attractor and removes target-near mass.

## Implementation

Added:

- `src/analysis/autoregressive_duplication_mechanism/phase4_x2_overcorrection_candidates.py`
- `scripts/analysis/run_autoregressive_duplication_phase4_x2_overcorrection_candidates.py`
- `tests/analysis/autoregressive_duplication_mechanism/test_phase4_x2_overcorrection_candidates.py`

Run command:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/analysis/run_autoregressive_duplication_phase4_x2_overcorrection_candidates.py \
  --rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/coord_basin_band_flow_layer17_head1/coord_basin_band_flow_rows.jsonl \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_candidates_whole_head
```

## Artifact Handles

- Output directory:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_candidates_whole_head`
- Candidate rows:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_candidates_whole_head/x2_overcorrection_candidate_rows.jsonl`
- Summary:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_candidates_whole_head/phase4_x2_overcorrection_candidate_summary.json`
- Report:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_candidates_whole_head/phase4_x2_overcorrection_candidate_report.md`

## Result

The selector found `7` candidates from `410` enriched band-flow rows:

- `6` from `aux_latest_ckpt32 #33`;
- `1` weak intermediate case from `no_aligner_parent_ckpt3668 #36`;
- `0` from `none_latest_ckpt32`;
- `0` from `aligner_parent_ckpt1824`.

Ranked candidates:

| rank | candidate | kind | gen-target | masked-target | patched-target | abs-delta | wide-delta | exact-delta | direction | score |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | `aux_latest_ckpt32_record33_row21_whole_head_x2_overcorrection` | masked_exact | 173 | 0 | -12 | 12 | -0.183812 | -0.036757 | below | 309.542 |
| 2 | `aux_latest_ckpt32_record33_row20_whole_head_x2_overcorrection` | masked_exact | 174 | 0 | -7 | 7 | -0.079195 | -0.037540 | below | 194.935 |
| 3 | `aux_latest_ckpt32_record33_row26_whole_head_x2_overcorrection` | masked_near | 185 | -4 | -11 | 7 | -0.104870 | 0.000000 | not-control | 124.720 |
| 4 | `aux_latest_ckpt32_record33_row23_whole_head_x2_overcorrection` | masked_near | 191 | 7 | -5 | -2 | -0.101951 | -0.026207 | below | 108.861 |
| 5 | `aux_latest_ckpt32_record33_row28_whole_head_x2_overcorrection` | masked_near | 195 | -5 | -5 | 0 | -0.078994 | 0.000000 | below | 85.944 |
| 6 | `aux_latest_ckpt32_record33_row18_whole_head_x2_overcorrection` | masked_near | 293 | -4 | -4 | 0 | -0.010332 | -0.000756 | below | 17.262 |
| 7 | `no_aligner_parent_ckpt3668_record36_row2_whole_head_x2_overcorrection` | masked_near | 517 | 6 | -1 | -5 | -0.000696 | -0.000091 | below | 6.866 |

Rows 21 and 20 are now promoted by criteria rather than hand selection. They remain the
cleanest future causal-probe candidates because masked top1 is exact target, while
patched top1 moves below target and target-near mass drops sharply.

## Mechanism Update

The selector makes the auxiliary finding operational:

1. the overcorrection signature is concentrated in `aux_latest_ckpt32`;
2. exact-masked cases are ranked highest;
3. `none_latest_ckpt32` contributes no rows under the same criteria, matching the
   corrective-downward movement interpretation;
4. the single no-aligner parent candidate is much weaker and has a tiny wide-radius loss,
   so it is useful as a possible control case rather than a primary target.

## Verification

- Focused direct test harness passed for
  `tests/analysis/autoregressive_duplication_mechanism/test_phase4_x2_overcorrection_candidates.py`.
- `python -m py_compile` passed for the selector module and CLI.
- Materializer completed with `candidate_count=7`.
- Inspected the generated report and candidate JSONL; rows 21 and 20 rank first.
