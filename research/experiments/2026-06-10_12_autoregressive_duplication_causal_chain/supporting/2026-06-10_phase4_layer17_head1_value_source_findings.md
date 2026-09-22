# Phase 4 Layer 17 Head 1 Value-Source Findings

## Question

The paired source-region readout showed that duplicate-basin masking collapses
attention on the duplicate visual cell while leaving broad visual-source mass
high. This value-source probe asks whether that local attention basin also
supplies a large aligned value contribution to the layer-17 head-1 output.

## Scope

- Artifact root: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920`
- Full-run prefix: `phase4_value_source_layer17_head1_top4_allshards`
- Summary JSON: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_value_source_layer17_head1_top4_allshards_summary.json`
- Markdown report: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_value_source_layer17_head1_top4_allshards_report.md`
- Smoke prefix: `phase4_value_source_layer17_head1_primary_smoke`
- Shards: `8`
- Rows: `35168`
- Replay cases: `30`
- Checkpoint loads across shards: `17`
- Checkpoint labels represented: `aligner_parent_ckpt1824`, `no_aligner_parent_ckpt3668`, `aux_latest_ckpt32`, `none_latest_ckpt32`
- Target site: decoder layer `17`, attention head `1`
- Intervention pair: `no_op_control` vs `duplicate_basin_mask`

Implementation support was added in commit `6778e668`:

- `src/analysis/autoregressive_duplication_mechanism/phase4_value_source.py`
- `scripts/analysis/run_autoregressive_duplication_phase4_value_source_shard.py`
- `tests/analysis/autoregressive_duplication_mechanism/test_phase4_value_source.py`

The probe captures `v_proj` outputs at the selected decoder layer and combines
them with the returned attention probabilities for the selected head. For GQA,
the query head is mapped to its KV head before slicing the value projection.

## Primary Anchor

Primary diagnostic anchor:

- checkpoint label: `none_latest_ckpt32`
- record: `33`
- role: `post_y1/pre_x2`
- duplicate-basin token count: `1`

Mean value-source metrics:

| Region | Control mass | Masked mass | Control contrib L2 | Masked contrib L2 | Control L2 frac | Masked L2 frac | Control proj frac | Masked proj frac |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `duplicate_basin` | `0.427491` | `0.005393` | `43.601439` | `0.563950` | `0.490544` | `0.007044` | `0.488726` | `0.006438` |
| `same_desc_component_envelope` | `0.811243` | `0.789721` | `75.637802` | `68.101449` | `0.884590` | `0.844747` | `0.882370` | `0.842026` |
| `spatial_basin_component_envelope` | `0.811243` | `0.789721` | `75.637802` | `68.101449` | `0.884590` | `0.844747` | `0.882370` | `0.842026` |
| `matched_gt_regions` | `0.933566` | `0.944923` | `84.714843` | `80.289953` | `0.991709` | `0.995176` | `0.991430` | `0.994936` |
| `rest_of_image` | `0.953100` | `0.955183` | `85.077721` | `80.361270` | `0.995923` | `0.996067` | `0.995811` | `0.995867` |
| `empty_top_left_control` | `0.001639` | `0.003296` | `0.078272` | `0.160415` | `0.000880` | `0.002126` | `0.000465` | `0.000467` |
| `empty_middle_left_control` | `0.002708` | `0.002847` | `0.163255` | `0.183411` | `0.001963` | `0.002372` | `0.001458` | `0.001825` |

The duplicate-basin source contributes about half of the total head-output norm
and projection in the control state. Under duplicate-basin masking, it becomes
nearly absent.

## Anchor Rows

The post-onset rows carry the effect. Examples:

| Row | Relative offset | Control mass | Masked mass | Control contrib L2 | Masked contrib L2 | Control proj frac | Masked proj frac |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `23` | `0` | `0.593750` | `0.006744` | `60.558849` | `0.705253` | `0.749570` | `0.007663` |
| `25` | `2` | `0.593750` | `0.007477` | `60.558849` | `0.781842` | `0.722941` | `0.008509` |
| `26` | `3` | `0.765625` | `0.008118` | `78.089043` | `0.848857` | `0.858654` | `0.009429` |
| `28` | `5` | `0.781250` | `0.010071` | `79.682693` | `1.053093` | `0.854167` | `0.011830` |
| `31` | `8` | `0.851562` | `0.012329` | `86.854134` | `1.289241` | `0.925366` | `0.016743` |

Rows before onset have near-zero duplicate-basin attention and value
contribution, matching the onset-local interpretation rather than a constant
image-wide bias.

## Cross-Case Pattern

At `post_y1/pre_x2`, most nonempty duplicate-basin cases lose value contribution
under the duplicate-basin mask. Largest L2-fraction drops:

| Checkpoint | Record | Tokens | Control L2 frac | Masked L2 frac | Delta |
|---|---:|---:|---:|---:|---:|
| `none_latest_ckpt32` | `33` | `1.0` | `0.490544` | `0.007044` | `-0.483500` |
| `aux_latest_ckpt32` | `33` | `1.0` | `0.413611` | `0.077914` | `-0.335697` |
| `none_latest_ckpt32` | `114` | `119.0` | `0.792117` | `0.541555` | `-0.250562` |
| `aux_latest_ckpt32` | `47` | `40.0` | `0.273198` | `0.073600` | `-0.199598` |
| `aligner_parent_ckpt1824` | `54` | `1.0` | `0.453844` | `0.284488` | `-0.169356` |
| `no_aligner_parent_ckpt3668` | `79` | `6.0` | `0.246943` | `0.091662` | `-0.155281` |

Some multi-token or weaker cases increase slightly under the mask
(`no_aligner_parent_ckpt3668` record `88`, `aligner_parent_ckpt1824` record `75`,
`aux_latest_ckpt32` record `75`). These are useful contrast cases, but they do
not overturn the primary finding.

## Mechanism Update

The deepest current picture is now:

- layer-17 head 1 is a visual-source head;
- the duplicate effect is localized to a small visual basin rather than visual
  tokens globally;
- the duplicate-basin cell supplies a large value contribution aligned with the
  head output at the onset-local coordinate slot;
- masking that cell removes both the attention mass and the value contribution,
  while broad visual-source and rest-of-image mass remain high;
- therefore, broad source-category explanations are too shallow. The current
  mechanism candidate is local high-density value routing from a visual basin
  into the coordinate-slot residual stream.

## Next Deterministic Step

Run a causal value-output patch rather than another observational readout:

- capture the layer-17 head-1 pre-`o_proj` vector or duplicate-basin value
  contribution in control;
- patch it into the duplicate-basin-masked state at `post_y1/pre_x2`;
- measure whether the correct next coordinate logit/rank is restored;
- also patch the masked/ablated contribution into control to test damage.

This directly tests whether the local value contribution is sufficient and
necessary, not merely correlated with the repair/damage direction.

