# Phase 2 Layer 17 Head 1 Attention Routing

Date: 2026-06-11

## Scope

This note records the first bounded Phase 2 attention/routing panel over the
validated Phase 1 manifest. The run is intentionally filtered to decoder layer
`17`, head `1`, because this lane has been repeatedly implicated by the Phase 4
route/content and mask-geometry probes.

This is routing evidence only. It must not be treated as causal proof before
Phase 3 perturbation and replay checks.

## Artifact Root

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133
```

Region artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/phase2_region_rows.jsonl
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/phase2_region_summary.json
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/phase2_region_report.md
```

Attention artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/shards/shard-*/attention_readouts/attention_region_rows.jsonl
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/phase2_attention_summary.json
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/phase2_attention_report.md
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/phase2_attention_suspect_heads.json
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/phase2_attention_suspect_heads_report.md
```

## Region Materialization

Region materialization completed for all selected windows:

- selected windows: `30`;
- region rows: `210`;
- region kinds per selected window:
  - `duplicate_basin`;
  - `same_desc_component_envelope`;
  - `spatial_basin_component_envelope`;
  - `matched_gt_regions`;
  - `empty_top_left_control`;
  - `empty_middle_left_control`;
  - `rest_of_image`;
- matched-GT enriched regions: `30`;
- matched-GT placeholders: `0`;
- manual-review-gated regions: `60`.

## Attention Readout

Layer/head filter:

```text
attention_layers = [17]
attention_heads = [1]
source_interventions = [no_op_control]
```

All four shards completed:

| Shard | GPU | replay cases | region rows | attention rows | checkpoint loads |
|---|---:|---:|---:|---:|---:|
| `shard-00-of-04` | 0 | 8 | 56 | 5040 | 3 |
| `shard-01-of-04` | 1 | 8 | 56 | 4424 | 2 |
| `shard-02-of-04` | 2 | 7 | 49 | 3920 | 3 |
| `shard-03-of-04` | 3 | 7 | 49 | 4200 | 3 |
| **Total** |  | **30** | **210** | **17584** |  |

Suspect-head aggregation:

- source attention rows: `17584`;
- head contrast rows: `240`;
- zero-token rows skipped: `1496`;
- top-k retained: `40`.

## Top Duplicate-Density Rows

The highest duplicate-basin density contrasts in this layer/head panel are:

| Checkpoint | Record | Phase | duplicate density | duplicate - matched GT | duplicate - rest |
|---|---:|---|---:|---:|---:|
| `none_latest_ckpt32` | 33 | `post_y1/pre_x2` | 0.427484 | 0.424433 | 0.424680 |
| `no_aligner_parent_ckpt3668` | 48 | `post_y1/pre_x2` | 0.394791 | 0.390315 | 0.391187 |
| `aux_latest_ckpt32` | 33 | `post_y1/pre_x2` | 0.355135 | 0.352033 | 0.352291 |
| `aux_latest_ckpt32` | 54 | `post_y1/pre_x2` | 0.273992 | 0.269895 | 0.263074 |
| `aligner_parent_ckpt1824` | 54 | `post_y1/pre_x2` | 0.248014 | 0.248014 | 0.238146 |
| `none_latest_ckpt32` | 33 | `box_start/pre_x1` | 0.224486 | 0.221500 | 0.221795 |
| `aux_latest_ckpt32` | 33 | `box_start/pre_x1` | 0.223419 | 0.220579 | 0.220806 |
| `no_aligner_parent_ckpt3668` | 34 | `post_y1/pre_x2` | 0.134792 | 0.134792 | 0.130183 |

The recurring high-density phase is `post_y1/pre_x2`, with record `33`
appearing strongly in both `none_latest_ckpt32` and `aux_latest_ckpt32`.
This lines up with the earlier route/content and mask-geometry evidence, but
the interpretation remains routing-only until perturbation tests confirm that
changing this route changes the target coordinate behavior.

## Smoke

Before the full bounded layer/head panel, a one-case attention smoke was run on
record `54`, checkpoint `aligner_parent_ckpt1824`, layer `17`, head `1`.

Smoke output:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/smoke/attention_readouts_one_case_layer17_head1
```

Smoke validation:

- replay cases: `1`;
- region rows: `7`;
- attention rows: `616`;
- phases covered: all eight Phase 1 token phases;
- region kinds covered: all seven Phase 2 region kinds.

## Next Step

Use this layer/head routing panel to prioritize Phase 3 perturbations. The
first perturbation lane should preserve no-op replay parity, then test
duplicate-basin masking on records with strong duplicate-density contrast,
especially record `33` and record `54` across `aux_latest_ckpt32` and
`none_latest_ckpt32`.
