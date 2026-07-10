# Phase 4 Value-Source Patch Coordinate-Distribution Findings

Date: 2026-06-11

## Scope

This note records the dynamic coordinate-slice follow-up to the layer-17
head-1 duplicate-basin value-source patch. The previous causal patch note
showed target-bin probability/rank repair. This run preserves distribution
telemetry over the full `<|coord_0|>` ... `<|coord_999|>` slice so we can ask
whether the patch creates a local coordinate-basin shift rather than only a
single-bin spike.

Artifact root:
`/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920`

Shard prefix:
`phase4_value_source_patch_coord_distribution_layer17_head1_top4_allshards_shard-*`

Machine summary:
`/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_value_source_patch_coord_distribution_layer17_head1_top4_allshards_summary.json`

Human report:
`/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_value_source_patch_coord_distribution_layer17_head1_top4_allshards_report.md`

Rows: 1,472 / expected 1,472. All eight shard row-count contracts passed.

## Probe Contract

The probe reruns the same layer-17 head-1 duplicate-basin value-source
contribution patch as the previous causal run:

- source region: `duplicate_basin`;
- patch site: layer-17 head-1 source contribution before `o_proj`;
- directions: `masked_to_control`, `control_to_masked`;
- windows: same top replay cases and anchors as the previous value-source patch.

New row fields preserve:

- coord-slice entropy;
- expected coordinate bin;
- coordinate standard deviation;
- expected absolute error from the target bin;
- top-1 coord distance;
- target-centered local masses at radii 4, 8, and 16.

For every metric the row records control, masked, patched, recovery from masked,
and damage from control.

## Primary Anchor Result

For `none_latest_ckpt32`, record 33, phase `post_y1/pre_x2`:

| direction | n | target prob recovery | rank recovery | radius-4 mass recovery | radius-8 mass recovery | expected-error recovery | top1-distance recovery | entropy recovery |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `masked_to_control` | 12 | 0.019106 | 20.916667 | 0.130624 | 0.227583 | -10.066869 | -10.916667 | -0.322221 |
| `control_to_masked` | 12 | 0.002440 | 14.333333 | 0.012816 | 0.021504 | 5.914249 | -2.583333 | 0.415662 |

The `masked_to_control` result is the important direction. Restoring the control
duplicate-basin value contribution into the masked run does not merely increase
the target bin. It moves substantial probability mass into the local coordinate
neighborhood around the target and reduces the expected-bin error by about
10 bins on average.

## Aggregate `post_y1/pre_x2` Result

The same local-basin pattern is visible beyond the primary anchor, strongest in
`none_latest_ckpt32`:

| checkpoint | direction | n | radius-4 mass recovery | radius-8 mass recovery | expected-error recovery | top1-distance recovery |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `none_latest_ckpt32` | `masked_to_control` | 26 | 0.066577 | 0.115833 | -8.164935 | -13.961538 |
| `aux_latest_ckpt32` | `masked_to_control` | 69 | 0.008638 | 0.015041 | -2.407822 | -2.086957 |
| `aligner_parent_ckpt1824` | `masked_to_control` | 22 | 0.008573 | 0.013960 | -2.586537 | -1.227273 |
| `no_aligner_parent_ckpt3668` | `masked_to_control` | 67 | 0.007585 | 0.012660 | -2.313160 | 2.447761 |

The radius-mass and expected-error metrics support the coordinate-slot basin
interpretation: the causal layer-17 head-1 source contribution steers the
distribution toward the target-centered local coordinate neighborhood.

## Strongest Local Repairs

The strongest `masked_to_control` radius-4 repairs include:

- `no_aligner_parent_ckpt3668`, record 48, offset +1:
  probability recovery `0.046551`, radius-4 recovery `0.257676`,
  expected-error recovery `-5.659684`;
- `none_latest_ckpt32`, record 33, offsets 0, +2, +3:
  probability recoveries around `0.032` to `0.036`, radius-4 recoveries around
  `0.216` to `0.235`, and expected-error recoveries from about `-13.8` to
  `-19.7` bins.

This strengthens the prior finding: the primary pathological anchor is
especially clear, but it is not the only case where the patch creates a local
coord-basin movement.

## Mechanistic Update

The current best circuit story is now:

1. layer-17 head 1 routes strongly into the duplicate-basin neighborhood at
   coordinate continuation windows;
2. its duplicate-basin value contribution is causally sufficient for strong
   target-bin repair;
3. the repair is a local coordinate-slice movement, not only a single-token
   target-bin jump;
4. the static coord-token surface is ordered but jagged, so the local
   distribution shift can be large even when raw centered target-unembedding
   cosine is small.

This connects the visual/attention source mechanism with the coordinate-slot
basin concern. The layer-17 source contribution appears to push the downstream
state into a local coord-token basin around the target.

## Next Deterministic Step

The remaining ambiguity is where the basin amplification occurs after layer 17.
The next probe should compare coord-slice movement after patching:

- immediately at the layer-17 `self_attn_output` boundary;
- after the layer-17 decoder block output;
- after later residual boundaries already implicated by the residual sweeps.

That will locate whether the basin amplification is mostly inside layer 17,
later decoder mixing, or final logit/readout geometry.

## Verification

- Full 8-shard GPU run completed.
- Shard row-count contract: passed for all eight shards.
- Aggregate row count: 1,472.
- Aggregate machine summary and Markdown report were written under the artifact
  root above.
