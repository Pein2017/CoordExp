# Phase 4 Layer-17 Head-1 Route/Content Causal Patch Findings

Date: 2026-06-11

## Scope

This slice tests the causal side of the layer-17 head-1 route/content
decomposition. Earlier decomposition showed that the duplicate-basin support
and near-ring opposition were mostly `route_delta_masked_values`, not
`value_delta_control_route`. This run asks whether those separated effect
vectors actually move the next-coordinate basin when injected at
`self_attn_output`.

The experiment uses the same selected Phase 4 windows as the prior head-1
analysis and keeps the dynamic-exploration rule from the roadmap: if a path
looks likely to change the final mechanism picture, deeper in-flight analysis
is allowed and should be recorded.

## Artifact Handles

- Combined rows:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_route_content_patch_layer17_head1_route_value_visual_triplet_top4_allshards_rows.jsonl`
- Summary JSON:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_route_content_patch_layer17_head1_route_value_visual_triplet_top4_allshards_summary.json`
- Markdown report:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_route_content_patch_layer17_head1_route_value_visual_triplet_top4_allshards_report.md`
- Smoke:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_route_content_patch_layer17_head1_primary_smoke_shard-04-of-08`

Run scale:

- 8 shards
- 22,608 causal patch rows
- layer 17, head 1
- patch site: `self_attn_output`
- patch components:
  `duplicate_basin`, `visual_near_ring`, `visual_far_background`
- effect kinds:
  `total_delta`, `route_delta_masked_values`, `value_delta_control_route`
- directions:
  `masked_to_control`, `control_to_masked`

## Primary Result

For `none_latest_ckpt32`, phase `post_y1/pre_x2`, direction
`masked_to_control`, the duplicate-basin route-only effect recovers most of
the local coordinate-basin movement seen in the total duplicate-basin patch:

| component | effect | n | prob rec | rank rec | radius4 rec | radius8 rec | expected abs-error rec |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| duplicate_basin | total_delta | 62 | 0.003976 | 4.225806 | 0.027956 | 0.048536 | -3.342128 |
| duplicate_basin | route_delta_masked_values | 62 | 0.003703 | -0.500000 | 0.026634 | 0.045759 | -1.972511 |
| duplicate_basin | value_delta_control_route | 62 | 0.000343 | 3.741935 | 0.002422 | 0.004558 | -1.427814 |
| visual_near_ring | route_delta_masked_values | 62 | 0.000194 | 0.225806 | 0.000349 | 0.000028 | 2.831848 |
| visual_far_background | route_delta_masked_values | 62 | 0.000055 | -0.758065 | 0.000206 | 0.000388 | 1.124952 |

The means are noisy because `post_y1/pre_x2` contains several coordinate
anchors, including anchors with near-zero effect and anchors with large basin
movement. The exact record-33 rows show the high-effect anchors clearly:
duplicate-basin `route_delta_masked_values` reaches probability recoveries
around `0.022-0.036`, radius-4 recoveries around `0.17-0.24`, and expected
absolute-error improvements around `-9.7` to `-19.4` bins on the strong
coordinate anchors. The matching value-only rows are typically around
`0.0001-0.0019` probability recovery and much smaller radius recovery.

## Interpretation

This is causal support for the route-content decomposition:

- The duplicate-basin component is not just correlated with the target
  direction; injecting the separated route effect can move the local coordinate
  distribution toward the control basin.
- The value-only term is real but much smaller for masked-to-control repair in
  the main `none_latest_ckpt32` slice.
- Near-ring and far-background route effects do not reproduce the
  duplicate-basin repair. Near-ring often changes rank on individual anchors,
  but its phase-average local mass and expected-error behavior is weak or in
  the wrong direction.

Mechanistically, the next origin question should shift from "is it route or
value?" to "what creates the route delta?" The most promising next test is a
query/key-side routing analysis: decompose attention-score changes into query
state versus visual-key state, and patch attention probabilities or Q/K
projections for the duplicate-basin and near-ring source buckets.

## Verification

- `python -m py_compile src/analysis/autoregressive_duplication_mechanism/phase4_route_content.py scripts/analysis/run_autoregressive_duplication_phase4_route_content_patch_shard.py tests/analysis/autoregressive_duplication_mechanism/test_phase4_route_content_patch.py`
- Direct importlib harness for
  `tests/analysis/autoregressive_duplication_mechanism/test_phase4_route_content_patch.py`
- One-shard smoke on shard 04:
  `phase4_route_content_patch_layer17_head1_primary_smoke_shard-04-of-08`
- Full eight-GPU shard sweep:
  `phase4_route_content_patch_layer17_head1_route_value_visual_triplet_top4_allshards_shard-00-of-08`
  through
  `phase4_route_content_patch_layer17_head1_route_value_visual_triplet_top4_allshards_shard-07-of-08`

`python -m pytest tests/analysis/autoregressive_duplication_mechanism/test_phase4_route_content_patch.py -q`
still reports the local `Pytest: No tests collected` wrapper issue, so the
direct harness is the meaningful deterministic test for this slice.
