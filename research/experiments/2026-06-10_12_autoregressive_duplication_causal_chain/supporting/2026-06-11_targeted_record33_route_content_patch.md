# Targeted Record-33 Route/Content Patch

Date: 2026-06-11

## Scope

This focused slice follows the cross-phase synthesis top hit from the
20260611 Phase 1 manifest:

- checkpoint: `none_latest_ckpt32`
- record: `33`
- phase: `post_y1/pre_x2`
- layer/head: `17/1`
- component: `duplicate_basin`

The goal is not to re-prove the broad route/content result. The earlier full
panel already showed that the layer-17 head-1 duplicate-basin effect is
route-dominated. This run connects that broader mechanism back to the strongest
cross-phase case after the Phase 0/1/2/3 joint synthesis.

## Artifact Handles

Target panel root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel
```

Generated inputs:

```text
target_token_windows.jsonl
target_region_rows.jsonl
target_cross_phase_rows.jsonl
target_panel_summary.json
```

Route/content patch output:

```text
route_content_patch_layer17_head1/route_content_patch_rows.jsonl
route_content_patch_layer17_head1/phase4_route_content_patch_summary.json
route_content_patch_layer17_head1/route_content_patch_reduced_summary.json
route_content_patch_layer17_head1/route_content_patch_reduced_report.md
```

Run scale:

- target rows: `82`
- target cases: `5`
- route/content patch rows: `246`
- groups after reduction: `21`
- GPUs used: `CUDA_VISIBLE_DEVICES=0`
- patch direction: `masked_to_control`
- patch site: `self_attn_output`
- patch component: `duplicate_basin`
- effect kinds:
  `total_delta`, `route_delta_masked_values`,
  `value_delta_control_route`

## Command

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \
python scripts/analysis/run_autoregressive_duplication_phase4_route_content_patch_shard.py \
  --token-windows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/target_token_windows.jsonl \
  --region-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/target_region_rows.jsonl \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/route_content_patch_layer17_head1 \
  --device auto \
  --attention-layer 17 \
  --attention-head 1 \
  --region-kind duplicate_basin \
  --top-k 8 \
  --patch-directions masked_to_control \
  --patch-sites self_attn_output \
  --patch-components duplicate_basin \
  --effect-kinds total_delta,route_delta_masked_values,value_delta_control_route
```

Reduction:

```bash
PYTHONDONTWRITEBYTECODE=1 \
python scripts/analysis/reduce_autoregressive_duplication_phase4_route_content_patch.py \
  --rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/route_content_patch_layer17_head1/route_content_patch_rows.jsonl \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/route_content_patch_layer17_head1 \
  --top-n 25
```

## Key Reduced Table

Top groups by mean probability recovery:

| rank | checkpoint | record | phase | effect | rows | prob recovery | prob damage | rank recovery | abs-error recovery | mass16 recovery |
|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `none_latest_ckpt32` | 33 | `post_y1/pre_x2` | `total_delta` | 12 | 0.018371 | 0.004639 | 19.4167 | -9.8000 | 0.289986 |
| 2 | `none_latest_ckpt32` | 33 | `post_y1/pre_x2` | `route_delta_masked_values` | 12 | 0.018168 | 0.004436 | 19.3333 | -9.8141 | 0.291114 |
| 3 | `no_aligner_parent_ckpt3668` | 48 | `post_y1/pre_x2` | `total_delta` | 12 | 0.004967 | -0.003997 | 2.2500 | -1.1736 | 0.042783 |
| 4 | `no_aligner_parent_ckpt3668` | 48 | `post_y1/pre_x2` | `route_delta_masked_values` | 12 | 0.003690 | -0.005273 | 1.2500 | -0.9722 | 0.035087 |
| 5 | `aux_latest_ckpt32` | 33 | `post_y1/pre_x2` | `route_delta_masked_values` | 12 | 0.002815 | 0.000430 | -1.6667 | -2.4080 | 0.051820 |
| 14 | `none_latest_ckpt32` | 33 | `post_y1/pre_x2` | `value_delta_control_route` | 12 | 0.000121 | -0.013611 | 0.5000 | 0.2754 | 0.000942 |

Strong individual anchors include `none_latest_ckpt32` record 33 rows
`23-31`, where route-only probability recovery reaches about `0.0188-0.0343`
and expected absolute-error recovery reaches about `-8.78` to `-18.64` bins.

## Interpretation

The focused panel supports the same mechanism as the broader full Phase 4
route/content causal patch:

- The cross-phase top hit is genuinely route dominated. For
  `none_latest_ckpt32` record 33 at `post_y1/pre_x2`, the route-only effect
  (`0.018168`) nearly equals the total effect (`0.018371`).
- The value/content-only term is far smaller (`0.000121`) and does not explain
  the local coordinate-basin repair.
- The effect is phase-specific. The same record at `box_start/pre_x1` has much
  smaller probability recovery, even though rank and expected-error changes can
  look large because the distribution is broad at early box positions.
- The result is not universal across checkpoints. The no-aligner parent has one
  strong record-48 anchor, while the aux/latest record-33 effect is weaker and
  less clean in rank recovery.

This narrows the causal question: for the main record-33 failure, the key
origin is more likely query/key routing into the duplicate basin than visual
value content stored in the duplicate-basin tokens.

## Next Deterministic Step

Use the existing query/key route-origin and attention-score-bias probes on this
same targeted record-33 panel, rather than launching another broad sweep first.
The concrete test is whether the route delta comes from:

- decoder-side query state after the duplicate prefix,
- visual key changes induced by basin masking, or
- a learned coordinate-slot attraction that amplifies small routing changes at
  `post_y1/pre_x2`.

## Verification

- Route/content patch run completed with summary:
  `route_content_patch_row_count=246`, `checkpoint_count=3`,
  `replay_case_count=5`.
- Reducer completed with summary:
  `row_count=246`, `group_count=21`.
- Narrow syntax check:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m py_compile \
  scripts/analysis/reduce_autoregressive_duplication_phase4_route_content_patch.py
```
