# Phase 4 Layer-17 Head-1 Value Source Findings

Date: 2026-06-11

## Scope

This note records the follow-up value-source readout launched after the
visual-spatial projected patch sweep. It keeps the same layer-17 head-1 branch
but asks a more direct source question: when the duplicate-basin intervention
is applied, how much of the head output still comes from the duplicate-basin
region versus broader visual or control regions?

Artifact root:
`/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920`

Shard prefix:
`phase4_value_source_layer17_head1_allshards_shard-*`

Machine summary:
`/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_value_source_layer17_head1_allshards_summary.json`

Human report:
`/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_value_source_layer17_head1_allshards_report.md`

Rows: 35,168 across eight shard files. Shard row-count contracts passed.

## Probe Contract

- Layer/head: layer 17, head 1.
- Readout: attention-weighted value-source contribution.
- Interventions: `no_op_control`, `duplicate_basin_mask`.
- Region kinds:
  `duplicate_basin`, `matched_gt_regions`, `rest_of_image`,
  `same_desc_component_envelope`, `spatial_basin_component_envelope`,
  `empty_top_left_control`, `empty_middle_left_control`.
- Main comparison: paired deltas
  `duplicate_basin_mask - no_op_control` at the same anchor/window/region.

## Primary Anchor Result

For `none_latest_ckpt32`, record 33, phase `post_y1/pre_x2`, the duplicate
basin mask removes a large amount of layer-17 head-1 value-source contribution
from the duplicate-basin region:

| region | n | attention mass delta | projected contribution fraction delta | projected projection-fraction delta |
| --- | ---: | ---: | ---: | ---: |
| duplicate_basin | 12 | -0.422098 | -0.483925 | -0.482850 |
| same_desc_component_envelope | 12 | -0.021522 | -0.040735 | -0.041145 |
| spatial_basin_component_envelope | 12 | -0.021522 | -0.040735 | -0.041145 |
| matched_gt_regions | 12 | 0.011357 | 0.003469 | 0.003521 |
| rest_of_image | 12 | 0.002083 | 0.000414 | 0.000363 |

This is a much sharper result than the visual-bucket patch aggregate: for the
pathological primary anchor, the intervention directly suppresses the
duplicate-basin source contribution instead of merely redistributing mass to a
large visual complement.

## Aggregate `post_y1/pre_x2` Result

The same direction appears in every checkpoint family, but the magnitude differs
substantially:

| checkpoint | duplicate-basin attention delta | duplicate-basin projected contribution-fraction delta |
| --- | ---: | ---: |
| `none_latest_ckpt32` | -0.123165 | -0.133397 |
| `aux_latest_ckpt32` | -0.088074 | -0.099209 |
| `no_aligner_parent_ckpt3668` | -0.027004 | -0.028464 |
| `aligner_parent_ckpt1824` | -0.010742 | -0.030905 |

The `none_latest_ckpt32` and `aux_latest_ckpt32` checkpoints therefore show the
largest duplicate-basin source sensitivity under this readout. The pure parent
checkpoints show the same sign but smaller magnitude.

## Mechanistic Update

This strengthens the layer-17 head-1 branch:

- the duplicate-basin mask changes the head's own source contribution, not only
  a downstream residual/logit readout;
- the strongest anchor has a large same-window drop in duplicate-basin
  attention mass and projected contribution fraction;
- broad regions like `rest_of_image` and `matched_gt_regions` do not absorb a
  comparable amount of the removed contribution on the primary anchor.

The result still does not close the mechanism. It tells us that the head's
value-source output is basin-sensitive, but not yet whether the decisive effect
is caused by attention routing, value content, output projection geometry, or a
downstream coordinate-slot basin that amplifies the source change.

## Next Deterministic Step

The next useful probe is a routing/content split at the same layer/head:

1. attention-source category readout for the same windows, to measure whether
   the basin mask primarily changes routing mass;
2. value-source patching by region kind, to test whether restoring only the
   duplicate-basin value contribution repairs the coordinate-slot target;
3. if those diverge, add a targeted coordinate-slot readout that projects the
   head-output delta onto the `<|coord_*|>` embedding manifold.

This keeps the current attractive branch alive while avoiding a premature
single-bucket explanation.

## Verification

- Full 8-shard GPU readout completed.
- Shard row-count contract: passed for all eight shards.
- Aggregate row count: 35,168.
- Aggregate machine summary and Markdown report were written under the artifact
  root above.
