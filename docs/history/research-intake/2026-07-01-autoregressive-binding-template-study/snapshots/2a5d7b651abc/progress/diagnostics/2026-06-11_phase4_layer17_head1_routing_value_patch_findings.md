# Phase 4 Layer-17 Head-1 Routing and Value-Patch Findings

Date: 2026-06-11

## Scope

This note records the follow-up routing/content split for the layer-17 head-1
branch of the autoregressive duplication mechanism study. It should be read
after the value-source note from the same date: the earlier readout showed that
duplicate-basin masking sharply reduces the head's duplicate-basin value-source
contribution; this slice asks whether that source is also causally sufficient
for logit repair.

Artifact root:
`/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920`

Attention-source routing report:
`/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_attention_source_layer17_head1_region_top4_allshards_report.md`

Attention-source machine summary:
`/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_attention_source_layer17_head1_region_top4_allshards_summary.json`

Value-source patch report:
`/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_value_source_patch_layer17_head1_top4_allshards_report.md`

Value-source patch machine summary:
`/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_value_source_patch_layer17_head1_top4_allshards_summary.json`

Artifact contracts:

- attention-source rows: 17,584 / expected 17,584;
- value-source patch rows: 1,472 / expected 1,472;
- all eight shard summaries are present for both probes.

## Routing Readout

The routing readout uses region memberships, not mutually exclusive source
categories. Region masses may overlap and should not be summed across rows.

For the primary anchor (`none_latest_ckpt32`, record 33,
`post_y1/pre_x2`), layer-17 head 1 routes strongly into the duplicate-basin
neighborhood:

| region | n | mean attention mass | mean tokens |
| --- | ---: | ---: | ---: |
| duplicate_basin | 12 | 0.427491 | 1.0 |
| same_desc_component_envelope | 12 | 0.811243 | 15.0 |
| spatial_basin_component_envelope | 12 | 0.811243 | 15.0 |
| matched_gt_regions | 12 | 0.933566 | 306.0 |
| rest_of_image | 12 | 0.953100 | 340.0 |
| empty_top_left_control | 12 | 0.001639 | 9.0 |
| empty_middle_left_control | 12 | 0.002708 | 9.0 |

This matters because the duplicate-basin row is a single-token region with
large mass. The same-desc/spatial envelope also carries large mass, but over a
small 15-token neighborhood rather than the full visual field.

Across all `post_y1/pre_x2` windows, duplicate-basin routing mass remains
checkpoint-dependent:

| checkpoint | duplicate-basin mean mass | duplicate-basin mean tokens |
| --- | ---: | ---: |
| `none_latest_ckpt32` | 0.165211 | 12.112903 |
| `aux_latest_ckpt32` | 0.201315 | 8.691358 |
| `no_aligner_parent_ckpt3668` | 0.126190 | 4.628319 |
| `aligner_parent_ckpt1824` | 0.081754 | 0.568966 |

The pathological primary anchor is therefore not representative in magnitude,
but it is a legitimate high-signal window: it is exactly where the next-token
coordinate target is most sensitive and the head routes unusually concentrated
mass into the duplicate basin.

## Causal Value-Source Patch

The value-source patch replaces only the `duplicate_basin` source contribution
inside the layer-17 head-1 pre-`o_proj` slice.

For the same primary anchor:

| direction | n | control prob | masked prob | patched prob | prob recovery | prob damage | rank recovery | rank damage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| masked_to_control | 12 | 0.026926 | 0.013462 | 0.032568 | 0.019106 | 0.005642 | 20.916667 | -3.416667 |
| control_to_masked | 12 | 0.026926 | 0.013462 | 0.015901 | 0.002440 | -0.011025 | 14.333333 | 3.166667 |

The `masked_to_control` direction is the decisive result: restoring just the
control duplicate-basin value contribution into the masked run repairs the
coordinate target beyond the original control probability mean. Several onset
rows also move to rank 1-4 after patching.

The reverse direction is asymmetric. Replacing the control contribution with
the masked duplicate-basin contribution damages the control state, but does not
perfectly recreate the masked state. This suggests the duplicate-basin value
contribution is sufficient for strong primary-anchor repair and partly
necessary for control-state integrity, while other downstream or neighboring
components also participate.

## Mechanistic Update

This is the strongest layer-17 head-1 evidence so far:

- routing evidence: the primary coordinate window routes concentrated attention
  into the duplicate-basin region;
- observational value-source evidence: duplicate-basin masking removes a large
  projected contribution from this head;
- causal patch evidence: restoring the duplicate-basin value contribution
  repairs the coordinate target at the same head-output slice.

The current best mechanism sketch is no longer a generic residual-flow story.
It is a local attention-head circuit:

1. at `post_y1/pre_x2`, layer-17 head 1 attends into the duplicate-basin
   neighborhood;
2. the attended value contribution carries target-relevant coordinate evidence;
3. downstream coordinate-slot dynamics amplify that contribution into large
   x2 probability/rank shifts.

The mechanism is still not closed. The next unresolved question is whether the
coordinate-slot amplification comes mostly from the coordinate-token embedding
basin, the unembedding/logit direction, or later decoder residual mixing.

## Next Deterministic Step

Run a coordinate-slot attraction probe on the same patched rows:

1. project the layer-17 head-1 duplicate-basin value delta onto the
   `<|coord_0|>` ... `<|coord_999|>` embedding/unembedding manifold;
2. compare target-bin direction against neighboring coordinate bins and the
   full coord-token subspace;
3. stratify the result by primary anchor versus aggregate `post_y1/pre_x2`
   windows.

This directly addresses the user's coordinate-slot basin concern without
leaving the successful layer-17 head-1 path.

## Verification

- Attention-source artifact contract passed:
  17,584 rows across eight shard files.
- Value-source patch artifact contract passed:
  1,472 rows across eight shard files.
- All required aggregate reports and machine summaries exist under the artifact
  root above.
