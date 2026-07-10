# Phase 4 Layer-17 Head-1 Value Basin Projection Findings

## Scope

This note records the coordinate-token basin projection probe for the layer-17
head-1 duplicate-basin value-source delta. It follows the prior value-source,
Q/K route-origin, score-bias, and score-pattern probes.

Primary artifact root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920
```

Combined outputs:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_value_basin_projection_layer17_head1_duplicate_top4_allshards_rows.jsonl
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_value_basin_projection_layer17_head1_duplicate_top4_allshards_summary.json
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_value_basin_projection_layer17_head1_duplicate_top4_allshards_report.md
```

Run configuration:

- Region: `duplicate_basin`
- Patch component: `duplicate_basin`
- Attention layer/head: `17/1`
- Shards: `8/8`
- Total rows: `1256`
- Protocol: replayed selected windows with no-op control and duplicate-basin mask, then projected the control-minus-masked attention-weighted value contribution through `o_proj` and into the 1000 coordinate-token unembedding basin.

## Primary Slice

Primary diagnostic slice:

```text
checkpoint_label=none_latest_ckpt32
phase=post_y1/pre_x2
patch_component=duplicate_basin
```

Summary:

- Rows: `62`
- Active non-flat rows: `26`
- Active fraction: `0.419355`
- Active projected contribution delta L2 mean: `60.695`
- Active target centered coordinate-logit delta mean: `0.174823`
- Active radius-4 local-vs-outside mean margin: `0.176352`
- Active target rank among coordinate tokens, non-flat strict-rank convention: mean `260.154`, median `104.5`, min `4`, max `885`
- Active cosine against the output target-coordinate direction: mean `0.015676`, median `0.026403`
- All-row target centered coordinate-logit delta mean: `0.073313`

The all-row median is zero because many rows have exactly flat/zero coordinate
delta. The probe now marks those rows with `coord_output_flat_delta` and
`coord_output_nonzero_delta`; strict rank on a flat vector should not be read as
target evidence.

## Interpretation

The successful duplicate-basin value-source delta is not merely a large residual
vector. In the primary `post_y1/pre_x2` slice it carries a positive coordinate
basin component toward the target coordinate token and its local radius-4 basin.
This supports the current mechanism sketch:

```text
visual duplicate-basin key/value state
  -> layer-17 head-1 route/content contribution
  -> coordinate-basin repair signal at the next coordinate slot
```

The effect is mixed rather than universal. Active rows are only about 42% of the
primary slice, and even among active rows the coordinate rank is broad. That
means the value channel is plausibly a carrier of coordinate repair, but not a
complete explanation of success or failure. The next useful question is why the
active rows split into constructive and destructive cases.

## Cross-Checkpoint Pattern

From the report table, the primary `none_latest_ckpt32 post_y1/pre_x2` slice is
the strongest positive coordinate-basin projection among that checkpoint's four
coordinate phases:

- `box_start/pre_x1`: active target centered mean `0.077037`
- `post_x1/pre_y1`: active target centered mean `-0.003227`
- `post_y1/pre_x2`: active target centered mean `0.174823`
- `post_x2/pre_y2`: active target centered mean `-0.009036`

The aux checkpoint also shows positive active target-centered signal at
`box_start/pre_x1` and `post_y1/pre_x2`, but weaker than the primary slice here.
This suggests the x2 slot after y1 is a particularly informative place to study
coordinate-slot attraction/basin mechanics, not just a generic coordinate-token
readout.

## Next Probe

Recommended next step:

1. Stratify the primary active rows by constructive vs destructive
   coordinate-basin projection.
2. Join those strata back to onset ledger features: burst position/order,
   duplicated object geometry, local object crowding, and source-basin token
   counts.
3. If the constructive/destructive split is stable, probe whether the split is
   determined upstream by visual key selection, value content, or coordinate-slot
   residual state before the head fires.

This keeps the work aimed at the core origin of the behavior rather than proving
that the surface phenomenon exists.
