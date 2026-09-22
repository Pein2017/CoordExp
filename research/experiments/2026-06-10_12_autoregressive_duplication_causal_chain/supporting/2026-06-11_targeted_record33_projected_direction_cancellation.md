# Targeted Record-33 Projected-Direction Cancellation

Date: 2026-06-11

## Scope

This slice follows the downstream-site patch for the cross-phase top case:

- checkpoint: `none_latest_ckpt32`
- record: `33`
- phase: `post_y1/pre_x2`
- layer/head: `17/1`

Prior targeted result:

- duplicate-basin route/content route-only output recovery: `0.018168`
- whole-head key+value source-state recovery: `0.013037`
- whole-head `self_attn_output` route-only recovery: `0.014328`

The previous result suggested that whole-head may underperform duplicate-basin
because other source buckets cancel the coordinate-target direction. This run
directly reads projected source-bucket deltas against the coordinate target
logit direction.

## Artifact Handles

Target panel root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel
```

Projected-direction output:

```text
projected_direction_component_cancellation_layer17_head1/projected_direction_rows.jsonl
projected_direction_component_cancellation_layer17_head1/phase4_projected_direction_summary.json
```

Reduced synthesis:

```text
targeted_projected_direction_cancellation_synthesis_layer17_head1/targeted_projected_direction_cancellation_synthesis_summary.json
targeted_projected_direction_cancellation_synthesis_layer17_head1/targeted_projected_direction_cancellation_synthesis_report.md
```

Run scale:

- target cases: `5`
- rows: `410`
- reduced groups: `35`
- layer/head: `17/1`
- components: `duplicate_basin`, `visual_near_ring`,
  `visual_far_background`, `non_region_complement`, `whole_head`
- GPU used: `CUDA_VISIBLE_DEVICES=0`

## Command

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \
python scripts/analysis/run_autoregressive_duplication_phase4_projected_direction_shard.py \
  --token-windows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/target_token_windows.jsonl \
  --region-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/target_region_rows.jsonl \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/projected_direction_component_cancellation_layer17_head1 \
  --device auto \
  --attention-layer 17 \
  --attention-head 1 \
  --region-kind duplicate_basin \
  --patch-components duplicate_basin,visual_near_ring,visual_far_background,non_region_complement,whole_head
```

## Result

Main case component projections:

| component | rows | pre-target fraction | projected target fraction | pre-target cosine | projected target cosine | projected delta L2 | source tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| `duplicate_basin` | 12 | 209.466 | 15.9077 | 0.212135 | 0.034267 | 72.3054 | 1 |
| `visual_near_ring` | 12 | -119.830 | -9.0936 | -0.065743 | -0.010536 | 64.5964 | 8 |
| `visual_far_background` | 12 | 4.693 | 0.3588 | 0.051837 | 0.009088 | 3.4266 | 331 |
| `non_region_complement` | 12 | -115.807 | -8.8152 | -0.046857 | -0.006967 | 66.5090 | 972 |
| `whole_head` | 12 | 93.659 | 7.0925 | 0.174443 | 0.029631 | 33.2567 | 973 |

Top cross-case projected target groups:

| rank | checkpoint | record | phase | component | projected target fraction |
|---:|---|---:|---|---|---:|
| 1 | `aux_latest_ckpt32` | 33 | `box_start/pre_x1` | `whole_head` | 17.4175 |
| 2 | `none_latest_ckpt32` | 33 | `post_y1/pre_x2` | `duplicate_basin` | 15.9077 |
| 3 | `none_latest_ckpt32` | 33 | `box_start/pre_x1` | `whole_head` | 14.8949 |
| 4 | `aux_latest_ckpt32` | 33 | `post_y1/pre_x2` | `duplicate_basin` | 11.3586 |
| 9 | `none_latest_ckpt32` | 33 | `post_y1/pre_x2` | `whole_head` | 7.0925 |

## Interpretation

This confirms cancellation.

For the main record-33 failure:

- `duplicate_basin` is the strongest coordinate-target-aligned component:
  projected target fraction `15.9077`.
- `visual_near_ring` is strongly anti-target: `-9.0936`.
- `non_region_complement` is also strongly anti-target: `-8.8152`.
- `visual_far_background` is small positive: `0.3588`.
- `whole_head` lands at `7.0925`, consistent with positive duplicate-basin
  signal being partially cancelled by other source buckets.

This explains why whole-head key+value and whole-head `self_attn_output`
patches were weaker than the earlier duplicate-basin route/content patch. The
whole-head vector is not a cleaner superset of duplicate-basin repair; it
contains substantial anti-target directions.

Mechanism status after this slice:

- The core positive repair vector is localized to the duplicate-basin route
  component.
- Near-ring and broad non-region sources oppose the coordinate target in
  projected residual space.
- Downstream residual/norm does not rescue the signal.
- The relevant mechanism is therefore a source-specific attention-output
  composition problem: an unusually aligned duplicate-basin route vector is
  competing with broader anti-target visual/source context.

## Next Deterministic Step

The next high-value test is to connect this cancellation picture to the
coordinate-token basin itself:

- inspect the output-logit distribution shift induced by positive
  duplicate-basin versus negative near-ring/non-region components;
- compare whether anti-target components push probability toward nearby
  coordinate bins, far coordinate bins, or the duplicated object's old bins;
- preserve the `<|coord_*|>` locality/smoothness framing from the roadmap.

This would decide whether the opposition is generic anti-target noise or a
structured attraction toward a competing coordinate basin.

## Verification

- GPU run completed with:
  `projected_direction_row_count=410`, `checkpoint_count=3`,
  `replay_case_count=5`.
- Reduced synthesis completed with:
  `rows=410`, `groups=35`.
- No code changes were made in this slice.
