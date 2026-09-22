# X2 Overcorrection Coordinate Basin Linkage

Date: 2026-06-11

## Scope

Joined the validated x2 overcorrection candidates with coordinate-basin band-flow,
attention-route redistribution, and hidden-shift evidence. This is a CPU
post-hoc linkage step; no new GPU work was launched in this slice.

The goal was to test whether the same rows with strong duplicate-basin route
dependence and late hidden-state movement also show attraction into the
lower-coordinate basin: target mass loss, lower-control-anchor mass gain, and
downward expected-coordinate movement.

## Inputs

- Candidate rows:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_probe_panel_whole_head/candidate_probe_rows.jsonl`
- Coordinate basin rows:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/coord_basin_band_flow_layer17_head1/coord_basin_band_flow_rows.jsonl`
- Candidate attention/hidden rows:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_attention_routing_layer17_head1/candidate_attention_hidden_summary/candidate_attention_hidden_summary_rows.jsonl`

## Output

Output root:

`/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_coord_basin_linkage_layer17_head1`

Files:

- `candidate_coord_basin_rows.jsonl`
- `phase4_candidate_coord_basin_summary.json`
- `phase4_candidate_coord_basin_summary.md`

Counts:

- Candidates: `7`
- Joined rows: `7`
- Missing coordinate rows: `0`
- Coordinate slot: `x2`
- Patch component: `whole_head`

## Main Readout

Checkpoint means:

| checkpoint | n | dup attention delta | hidden L20 | target-wide loss | lower-control gain | expected downshift | strength | below-target top1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `aux_latest_ckpt32` | 6 | 0.399960 | 97.246207 | 0.093192 | 0.098923 | 9.679072 | 0.292431 | 5 |
| `no_aligner_parent_ckpt3668` | 1 | 0.001480 | 17.380585 | 0.000696 | 0.004743 | 2.928497 | 0.034724 | 1 |

Strongest rows:

| candidate | dup attention delta | hidden L20 | target-wide loss | lower-control gain | expected downshift | patched-target |
|---|---:|---:|---:|---:|---:|---:|
| row 21 exact | 0.602295 | 124.057 | 0.183812 | 0.216171 | 14.1583 | -12 |
| row 20 exact | 0.534668 | 123.955 | 0.079195 | 0.192730 | 12.8778 | -7 |
| row 23 near | 0.345703 | 107.419 | 0.101951 | 0.116587 | 11.9016 | -5 |
| row 28 near | 0.674316 | 111.955 | 0.078994 | 0.089200 | 9.6039 | -5 |

Descriptive correlations across the seven candidate-selected rows:

- duplicate attention delta vs lower-control gain: `r=0.787855`
- duplicate attention delta vs target-wide loss: `r=0.722713`
- duplicate attention delta vs expected-bin downshift: `r=0.841997`
- hidden L20 delta vs lower-control gain: `r=0.734039`
- hidden L20 delta vs target-wide loss: `r=0.842075`
- hidden L20 delta vs expected-bin downshift: `r=0.966909`

## Interpretation

The current row-linked chain is:

1. The auxiliary x2 candidates depend strongly on duplicate-basin attention at
   route head `17/1`.
2. Masking that duplicate basin causes large route redistribution into broader
   visual alternatives.
3. The same rows have large late hidden-state shifts around/after the route-head
   neighborhood.
4. The same rows lose target-near coordinate mass, gain lower-control mass, and
   shift expected x2 downward.

This is the clearest current evidence for the auxiliary checkpoint's
coordinate-slot attraction story: the overcorrection is not merely a top1 token
quirk. It is tied to route-head duplicate-basin dependence and a coordinate
distribution movement into a lower basin.

Guardrail: this remains candidate-selected evidence with `n=7`, including only
one no-aligner control candidate. Treat the correlations as mechanistic linkage
diagnostics, not population estimates or proof of final causal sufficiency.

## Reproduction

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/analysis/run_autoregressive_duplication_phase4_candidate_coord_basin_summary.py \
  --candidate-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_probe_panel_whole_head/candidate_probe_rows.jsonl \
  --coord-basin-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/coord_basin_band_flow_layer17_head1/coord_basin_band_flow_rows.jsonl \
  --candidate-attention-hidden-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_attention_routing_layer17_head1/candidate_attention_hidden_summary/candidate_attention_hidden_summary_rows.jsonl \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_coord_basin_linkage_layer17_head1
```

Verification:

```bash
python -m py_compile \
  src/analysis/autoregressive_duplication_mechanism/phase4_candidate_coord_basin_summary.py \
  scripts/analysis/run_autoregressive_duplication_phase4_candidate_coord_basin_summary.py \
  tests/analysis/autoregressive_duplication_mechanism/test_phase4_candidate_coord_basin_summary.py
```

The direct harness for
`tests/analysis/autoregressive_duplication_mechanism/test_phase4_candidate_coord_basin_summary.py`
passed.
