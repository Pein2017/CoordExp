# Phase 4 Layer 17 Head 1 Source Region Control-vs-Mask Findings

## Question

Earlier layer-17 head-1 source-category readout showed that masking the
duplicate-basin visual region barely changed coarse source-category allocation:
visual-token mass stayed high. This follow-up asks whether the stable visual
category hides a local rerouting inside the visual field.

## Scope

- Artifact root: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920`
- Prefix: `phase4_attention_source_region_layer17_head1_control_vs_mask_top4_allshards`
- Summary JSON: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_attention_source_region_layer17_head1_control_vs_mask_top4_allshards_summary.json`
- Markdown report: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_attention_source_region_layer17_head1_control_vs_mask_top4_allshards_report.md`
- Shards: `8`
- Rows: `35168`
- Checkpoint labels represented: `aligner_parent_ckpt1824`, `no_aligner_parent_ckpt3668`, `aux_latest_ckpt32`, `none_latest_ckpt32`
- Intervention pair: `no_op_control` vs `duplicate_basin_mask`
- Target site: decoder layer `17`, attention head `1`

The shard summaries in this run use the older `phase2_attention` summary shape,
so the report derives counts and metrics directly from `attention_region_rows.jsonl`
files.

## Primary Anchor

Primary diagnostic anchor:

- checkpoint label: `none_latest_ckpt32`
- record: `33`
- role: `post_y1/pre_x2`
- duplicate-basin token count: `1`

Mean attention mass by visual region:

| Region | Control mass | Masked mass | Delta | Control density | Masked density | Tokens | n |
|---|---:|---:|---:|---:|---:|---:|---:|
| `duplicate_basin` | `0.427491` | `0.005393` | `-0.422098` | `0.427491` | `0.005393` | `1.0` | `12` |
| `same_desc_component_envelope` | `0.811243` | `0.789721` | `-0.021522` | `0.054083` | `0.052648` | `15.0` | `12` |
| `spatial_basin_component_envelope` | `0.811243` | `0.789721` | `-0.021522` | `0.054083` | `0.052648` | `15.0` | `12` |
| `matched_gt_regions` | `0.933566` | `0.944923` | `+0.011357` | `0.003051` | `0.003088` | `306.0` | `12` |
| `rest_of_image` | `0.953100` | `0.955183` | `+0.002083` | `0.002803` | `0.002809` | `340.0` | `12` |
| `empty_top_left_control` | `0.001639` | `0.003296` | `+0.001656` | `0.000182` | `0.000366` | `9.0` | `12` |
| `empty_middle_left_control` | `0.002708` | `0.002847` | `+0.000140` | `0.000301` | `0.000316` | `9.0` | `12` |

Anchor-row duplicate-basin masses:

| Row | Relative offset | Control mass | Masked mass | Delta |
|---:|---:|---:|---:|---:|
| `20` | `-3` | `0.000790` | `0.000026` | `-0.000764` |
| `21` | `-2` | `0.001984` | `0.000040` | `-0.001944` |
| `22` | `-1` | `0.000007` | `0.000002` | `-0.000005` |
| `23` | `0` | `0.593750` | `0.006744` | `-0.587006` |
| `24` | `1` | `0.000157` | `0.000014` | `-0.000144` |
| `25` | `2` | `0.593750` | `0.007477` | `-0.586273` |
| `26` | `3` | `0.765625` | `0.008118` | `-0.757507` |
| `27` | `4` | `0.384766` | `0.006409` | `-0.378357` |
| `28` | `5` | `0.781250` | `0.010071` | `-0.771179` |
| `29` | `6` | `0.691406` | `0.005493` | `-0.685913` |
| `30` | `7` | `0.464844` | `0.007996` | `-0.456848` |
| `31` | `8` | `0.851562` | `0.012329` | `-0.839233` |

## Cross-Case Pattern

At `post_y1/pre_x2`, nonempty duplicate-basin cases usually lose basin mass
under the duplicate-basin mask. The largest observed drops include:

| Checkpoint | Record | Tokens | Control mass | Masked mass | Delta |
|---|---:|---:|---:|---:|---:|
| `none_latest_ckpt32` | `33` | `1.0` | `0.427491` | `0.005393` | `-0.422098` |
| `none_latest_ckpt32` | `114` | `119.0` | `0.771910` | `0.497285` | `-0.274625` |
| `aux_latest_ckpt32` | `33` | `1.0` | `0.355069` | `0.055281` | `-0.299789` |
| `aux_latest_ckpt32` | `47` | `40.0` | `0.211151` | `0.057109` | `-0.154042` |
| `no_aligner_parent_ckpt3668` | `79` | `6.0` | `0.241132` | `0.084902` | `-0.156230` |

A few cases increase under the mask, especially when the duplicate-basin region
contains multiple cells or the case is not the primary high-confidence anchor
(`aligner_parent_ckpt1824` record `75`, `aux_latest_ckpt32` record `75`,
`no_aligner_parent_ckpt3668` record `34`). These should be treated as follow-up
cases, not as a reversal of the primary mechanism.

## Interpretation

The coarse source-category result was not wrong, but it was too coarse. Layer-17
head 1 keeps attending to visual tokens overall, yet the duplicate-basin cell
itself collapses from `0.427491` to `0.005393` mean mass in the primary anchor.
The rest-of-image mass is essentially unchanged (`0.953100` to `0.955183`).

So the current mechanism picture is:

- head 1 is a visual-source head;
- its causal duplicate effect is localized to a small visual basin/cell rather
  than to visual tokens globally;
- duplicate-basin masking reroutes attention inside the visual field while
  removing the high-density attractor at the duplicate cell;
- the next high-yield probe should separate attention probabilities from value
  content and inspect sub-visual-location/value vectors, not repeat broad
  source-category scans.

## Next Deterministic Step

Continue Phase 4 with a value/content probe at layer `17`, head `1`:

- capture the value vectors or attention output contribution for duplicate-basin
  tokens versus nearby same-desc/spatial-envelope tokens;
- compare control and duplicate-basin-mask states on the primary anchor first;
- then expand only to the nonempty cross-case rows listed above if the primary
  result is coherent.

