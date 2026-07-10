# Phase 4 Layer-17 Head-1 Visual Spatial Patch Findings

Date: 2026-06-11

## Scope

This note records the first full 8-GPU projected-contribution patch sweep for
the layer-17 head-1 branch of the autoregressive duplication mechanism study.
The probe patches the projected head-output contribution at
`self_attn_output` using duplicate-basin and visual-spatial source buckets.

Artifact root:
`/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920`

Full sweep artifact prefix:
`phase4_projected_patch_visual_spatial_buckets_layer17_head1_top4_allshards`

Machine summary:
`/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_projected_patch_visual_spatial_buckets_layer17_head1_top4_allshards_summary.json`

Human report:
`/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_projected_patch_visual_spatial_buckets_layer17_head1_top4_allshards_report.md`

Rows: 15,072 / expected 15,072. All eight shard row-count contracts passed.

## Probe Contract

- Layer/head: layer 17, head 1.
- Patch site: `self_attn_output`.
- Region basis: `duplicate_basin`.
- Patch components:
  `duplicate_basin`, `visual_near_ring`, `visual_far_background`,
  `visual_non_basin`, `non_region_complement`, `whole_head`.
- Patch directions:
  `masked_to_control`, `control_to_masked`.
- Main readout window:
  `post_y1/pre_x2`, where the next token is the coordinate slot most directly
  tied to x2 continuation after y1.

Direction convention:

- `masked_to_control`: does the component repair the duplicate-basin mask?
- `control_to_masked`: does transplanting the masked component into the control
  move the control toward the masked behavior?

## Primary Anchor Result

For `none_latest_ckpt32`, record 33, phase `post_y1/pre_x2`, the primary anchor
keeps the smoke-run pattern:

| component | direction | n | prob recovery mean | prob damage mean | rank recovery mean |
| --- | --- | ---: | ---: | ---: | ---: |
| duplicate_basin | masked_to_control | 12 | 0.018809 | 0.005344 | 19.833333 |
| visual_near_ring | control_to_masked | 12 | 0.019530 | 0.006066 | 20.000000 |
| visual_far_background | control_to_masked | 12 | 0.014434 | 0.000970 | 18.250000 |
| whole_head | masked_to_control | 12 | 0.012645 | -0.000820 | 18.416667 |

On this anchor, `visual_near_ring` is the strongest non-duplicate-basin
`control_to_masked` bucket and behaves like a compact causal source feeding the
masked-like coordinate-slot shift. This is promising because it connects the
duplicate basin to a concrete spatial source bucket rather than only to a
coarse residual or whole-head effect.

## Cross-Window Result

The broader `post_y1/pre_x2` table is more mixed and should prevent an
overclaim:

- For `none_latest_ckpt32`, `visual_far_background` has the largest aggregate
  `control_to_masked` probability recovery among visual buckets
  (`0.004950`), while `visual_near_ring` is close but smaller (`0.004097`).
- For `aligner_parent_ckpt1824`, `visual_near_ring` is the strongest aggregate
  `control_to_masked` visual bucket (`0.001527`), but the absolute signal is
  small.
- For `no_aligner_parent_ckpt3668`, `visual_far_background` is stronger than
  `visual_near_ring` in aggregate (`0.002970` vs `0.000522`).
- `visual_non_basin` and `non_region_complement` can match or exceed near-ring
  on some aggregates, which means the current bucket split is not yet isolating
  a unique near-ring-only mechanism.

The conclusion is therefore not "near-ring explains duplication." The sharper
read is: a visually sourced projected contribution can be causal at the layer-17
head-1 boundary, and the primary pathological anchor exposes a highly compact
near-ring effect, but the full aggregate still entangles near visual content,
far/background visual content, and broad non-region complements.

## Mechanistic Update

This branch remains attractive because it narrows the origin from generic
residual flow to a specific attention-head output boundary. However, the final
picture likely depends on a routing/content interaction:

- duplicate-basin query/key routing may select a small set of source positions;
- visual value content may carry object-local or scene-level continuation
  evidence;
- the downstream coordinate-slot basin may amplify small upstream differences
  into large x2-rank changes.

That interaction is more plausible than a single-token or single-bucket story.
Dynamic exploration should continue here, but the next probes should separate
source value content from routing and downstream coordinate attraction rather
than only adding more coarse visual buckets.

## Next Deterministic Step

Run a layer-17 head-1 source decomposition that keeps the successful
`self_attn_output` boundary but splits the mechanism into:

1. attention routing over duplicate-basin, near-ring, far/background, and
   complement source positions;
2. value/output contribution from those same source positions;
3. coordinate-slot target sensitivity before and after downstream residual
   propagation.

The immediate practical check should compare the same primary anchor against
the aggregate windows. A finding only on record 33 is still useful as a
pathological anchor, but it should be labeled as an anchor-specific mechanism
until a broader stratum reproduces it.

## Verification

- Full 8-shard GPU sweep completed.
- Shard row-count contract: passed for all eight shards.
- Aggregate row count: 15,072.
- Aggregate machine summary and Markdown report were written under the artifact
  root above.
