# Coord-Token Embedding Geometry Findings

Date: 2026-06-11

## Scope

This note records a post-hoc static geometry probe for the added
`<|coord_0|>` ... `<|coord_999|>` token rows used by the current
autoregressive duplication mechanism study. It directly addresses the concern
that the coordinate-token slot may act as a basin/attractor whose row geometry
is local but not smooth.

Artifact root:
`/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920`

All-checkpoint summary:
`/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/coord_token_embedding_geometry_allcheckpoints_summary.json`

All-checkpoint report:
`/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/coord_token_embedding_geometry_allcheckpoints_report.md`

Per-checkpoint reports:

- `coord_token_embedding_geometry_aligner_parent_ckpt1824/coord_token_embedding_geometry_report.md`
- `coord_token_embedding_geometry_no_aligner_parent_ckpt3668/coord_token_embedding_geometry_report.md`
- `coord_token_embedding_geometry_latest_ckpt32/coord_token_embedding_geometry_report.md`

The `latest_ckpt32` static row geometry covers both manifest labels
`aux_latest_ckpt32` and `none_latest_ckpt32`, because both labels resolve to the
same physical checkpoint path.

## Probe Contract

The probe loads each checkpoint with the same Phase-4-compatible model loader
used by the autoregressive duplication mechanism runners. This matters for
`checkpoint-32`, which contains training-only instance-enumeration probe tensors
and is rejected by the ordinary inference adapter resolver.

For each checkpoint, the probe extracts:

- `base_input`;
- `base_output`;
- `coord_offset`;
- `effective_input`;
- `effective_output`;
- `effective_minus_base_input`;
- `effective_minus_base_output`.

Metrics are static row-geometry diagnostics over the 1000 coordinate bins:
distance-vs-numeric-bin correlation, radius-4 nearest-neighbor recall,
local-step size, second-difference-to-step ratio, linear directionality, and
contiguous cluster purity.

## Main Result

| checkpoint | surface | spearman dist-num | radius4 recall | local step | second/step | direction r2 | cluster purity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `aligner_parent_ckpt1824` | `base_input` | 0.522372 | 0.049563 | 0.000447 | 1.823571 | 0.973267 | 0.272724 |
| `aligner_parent_ckpt1824` | `coord_offset` | 0.547578 | 0.466562 | 0.044291 | 1.753213 | 1.000000 | 1.000000 |
| `aligner_parent_ckpt1824` | `effective_input` | 0.846337 | 0.466562 | 0.044337 | 1.753134 | 1.000000 | 1.000000 |
| `no_aligner_parent_ckpt3668` | `coord_offset` | 0.495408 | 0.491875 | 0.059584 | 1.739253 | 1.000000 | 1.000000 |
| `no_aligner_parent_ckpt3668` | `effective_input` | 0.834581 | 0.491875 | 0.059613 | 1.739172 | 1.000000 | 1.000000 |
| `latest_ckpt32` | `coord_offset` | 0.544571 | 0.466813 | 0.044490 | 1.752821 | 1.000000 | 1.000000 |
| `latest_ckpt32` | `effective_input` | 0.845334 | 0.466813 | 0.044537 | 1.752750 | 1.000000 | 1.000000 |

The base coord rows are almost unchanged across these checkpoints and have weak
locality by radius-4 nearest-neighbor recall. The effective coordinate surface
is created by the coord-offset rows.

## Mechanistic Read

The result supports a split interpretation:

- coordinate-token locality is real after applying the coord-offset surface;
- the surface is strongly ordered at a coarse scale (`directionality_r2` and
  cluster purity are essentially perfect);
- adjacent smoothness is not clean: `second_diff_to_step_ratio` stays around
  `1.74` to `1.75`, so the row path is jagged rather than a smooth numeric
  curve;
- `latest_ckpt32` is nearly identical to `aligner_parent_ckpt1824` in static
  coordinate-row geometry, so the tiny-step loss-only checkpoint did not
  visibly rewrite the static coord-token manifold;
- `no_aligner_parent_ckpt3668` has a larger coord-offset step size and slightly
  higher radius-4 recall, but the same roughness pattern.

This explains why the layer-17 head-1 causal path can strongly repair a
coordinate target while having only small raw centered target-coordinate
unembedding cosine in the projected-direction report. The coordinate rows form
a typed, ordered basin, but not a smooth final-logit line that every upstream
causal vector must align with directly.

## Next Deterministic Step

Continue with a dynamic coordinate-slot probe rather than another static
embedding probe:

1. take the successful layer-17 head-1 duplicate-basin value patch rows;
2. measure how the patch changes probability mass over the full
   `<|coord_0|>` ... `<|coord_999|>` slice, not only the target bin;
3. compare distribution shape around the target bin against the static
   coord-row neighborhood geometry.

That will directly test whether downstream coordinate-slot attraction turns
the layer-17 source contribution into a local coord-token basin shift.

## Verification

- `aligner_parent_ckpt1824`, `no_aligner_parent_ckpt3668`, and `latest_ckpt32`
  geometry probes completed.
- `latest_ckpt32` was loaded through the Phase-4-compatible loader, including
  training-only probe adapter support.
- All three per-checkpoint JSON summaries and Markdown reports were written.
- The all-checkpoint aggregate JSON and Markdown reports were written.
