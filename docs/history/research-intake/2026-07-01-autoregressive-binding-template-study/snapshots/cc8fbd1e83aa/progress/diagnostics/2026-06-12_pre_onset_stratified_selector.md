# Pre-Onset Stratified Selector

## Scope

This slice adds a post-hoc selector over already-run pre-onset residual patch rows. It is a target-selection helper, not a new causal probe. The reason to add it now is that the four-case residual patch comparison split by sign:

- duplicate-basin mask helps the target in some rows;
- duplicate-basin mask harms the target in other rows;
- averaging these signs would hide the mechanism split.

## Implementation

Entrypoint:

`scripts/analysis/run_autoregressive_duplication_phase4_pre_onset_stratified_selector.py`

Module:

`src/analysis/autoregressive_duplication_mechanism/phase4_pre_onset_stratified_selector.py`

Test:

`tests/test_autoreg_pre_onset_stratified_selector.py`

The selector:

1. loads one or more `residual_patch_rows.jsonl` files;
2. collapses rows into unique `(checkpoint, record_idx, row_idx, phase, relative_row_offset)` cases;
3. classifies duplicate-basin masking as `mask_helps_rank`, `mask_helps_prob`, `mask_harms_rank`, `mask_harms_prob`, or `neutral`;
4. records best `masked_to_control` decoder-layer residual patch recovery;
5. writes ranked case rows, a summary JSON, and a compact Markdown report.

## Four-Case Artifact

Output root:

`/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_stratified_selector_four_case`

Files:

- `pre_onset_stratified_cases.jsonl`
- `pre_onset_stratified_summary.json`
- `pre_onset_stratified_report.md`

Command:

```bash
python scripts/analysis/run_autoregressive_duplication_phase4_pre_onset_stratified_selector.py \
  --residual-patch-rows /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_causal_patch_record33_no_aligner/residual_patch_layers20_24_26_27_mlp_decoder/residual_patch_rows.jsonl \
  --residual-patch-rows /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_causal_patch_comparison/no_aligner_record36_skis_offsetm1/residual_patch_layers20_24_26_27_mlp_decoder/residual_patch_rows.jsonl \
  --residual-patch-rows /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_causal_patch_comparison/no_aligner_record48_bus_offsetm1/residual_patch_layers20_24_26_27_mlp_decoder/residual_patch_rows.jsonl \
  --residual-patch-rows /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_causal_patch_comparison/none_latest_record33_bottle_offsetm2/residual_patch_layers20_24_26_27_mlp_decoder/residual_patch_rows.jsonl \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_stratified_selector_four_case \
  --top-k-per-sign 2
```

Summary:

- input paths: `4`
- input residual patch rows: `64`
- selected cases: `4`
- mask sign counts: `{"mask_harms_rank": 2, "mask_helps_prob": 1, "mask_helps_rank": 1}`

Ranked rows:

| case | sign | control rank/prob | masked rank/prob | desc | target | best decoder patch |
| --- | --- | --- | --- | --- | ---: | --- |
| `none_latest_ckpt32:33:21:post_y1/pre_x2:-2` | `mask_harms_rank` | 27/0.014387 | 31/0.013195 | bottle | 209 | L26 rank_rec +4.000 prob_rec +0.001229 |
| `no_aligner_parent_ckpt3668:33:2:post_y1/pre_x2:-2` | `mask_helps_rank` | 5/0.032961 | 2/0.036959 | wine glass | 145 | L27 rank_rec -3.000 prob_rec -0.003998 |
| `no_aligner_parent_ckpt3668:48:2:post_y1/pre_x2:-1` | `mask_harms_rank` | 1/0.061880 | 2/0.061519 | bus | 899 | L20 rank_rec +1.000 prob_rec +0.001672 |
| `no_aligner_parent_ckpt3668:36:2:post_y1/pre_x2:-1` | `mask_helps_prob` | 2/0.038703 | 2/0.038927 | skis | 448 | L20 rank_rec -1.000 prob_rec -0.000772 |

## Mechanism Use

This points to a clean matched-pair next probe:

- harmful-attractor row: `no_aligner_parent_ckpt3668:33:2:post_y1/pre_x2:-2`
- useful-support row: `no_aligner_parent_ckpt3668:48:2:post_y1/pre_x2:-1`

Both are from the same parent checkpoint and same phase. This makes them a better pair for attention/logit/residual comparison than mixing in the `none_latest_ckpt32` bottle row first, even though the bottle row is the strongest rank-sensitive support case.

## Verification

Ran:

```bash
python -m pytest tests/test_autoreg_pre_onset_stratified_selector.py -q
python scripts/analysis/run_autoregressive_duplication_phase4_pre_onset_stratified_selector.py ... --top-k-per-sign 2
```

The pytest command exited with status `0`. The selector run produced the summary and report under the four-case artifact root above.
