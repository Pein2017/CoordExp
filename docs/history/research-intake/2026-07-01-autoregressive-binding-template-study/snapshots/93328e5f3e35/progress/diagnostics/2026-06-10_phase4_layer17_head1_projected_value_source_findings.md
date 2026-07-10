# Phase 4 Layer 17 Head 1 Projected Value-Source Findings

## Question

The previous layer-transition patch localized the causal source-contribution
route to layer `17`, head `1`, but the patch target was the selected head's
pre-`o_proj` slice. This probe asks whether the duplicate-basin contribution is
already dominant before `o_proj`, or whether layer-17 `o_proj` creates the
dominant coordinate-slot direction from a weaker head-local vector.

## Scope

- Artifact root: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920`
- Full-run prefix: `phase4_value_source_projected_layer17_head1_top4_allshards`
- Summary JSON: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_value_source_projected_layer17_head1_top4_allshards_summary.json`
- Markdown report: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_value_source_projected_layer17_head1_top4_allshards_report.md`
- Smoke prefix: `phase4_value_source_projected_layer17_head1_primary_smoke`
- Shards: `8`
- Rows: `35168`
- Replay cases: `30`
- Checkpoint loads across shards: `17`
- Checkpoint labels represented: `aligner_parent_ckpt1824`, `no_aligner_parent_ckpt3668`, `aux_latest_ckpt32`, `none_latest_ckpt32`
- Target site: decoder layer `17`, attention head `1`
- Intervention pair: `no_op_control` vs `duplicate_basin_mask`

Implementation support was added in this slice:

- `src/analysis/autoregressive_duplication_mechanism/phase4_value_source.py`
- `tests/analysis/autoregressive_duplication_mechanism/test_phase4_value_source.py`

The probe projects each region's pre-`o_proj` source contribution through the
matching layer-17 attention output projection column block for head `1`. It
adds post-`o_proj` fields beside the original pre-`o_proj` fields, including
projected contribution norm, projected total norm, projected L2 fraction,
projected cosine, and projected projection fraction.

## Primary Anchor

Primary diagnostic anchor:

- checkpoint label: `none_latest_ckpt32`
- record: `33`
- role: `post_y1/pre_x2`
- duplicate-basin token count: `1`

Mean duplicate-basin metrics:

| Metric | Control | Duplicate-basin mask | Delta |
|---|---:|---:|---:|
| Attention mass | `0.427491` | `0.005393` | `-0.422098` |
| Pre-`o_proj` contribution L2 | `43.601439` | `0.563950` | `-43.037489` |
| Pre-`o_proj` L2 fraction | `0.490544` | `0.007044` | `-0.483500` |
| Pre-`o_proj` projection fraction | `0.488726` | `0.006438` | `-0.482288` |
| Projected contribution L2 | `74.437844` | `0.964282` | `-73.473562` |
| Projected L2 fraction | `0.490960` | `0.007035` | `-0.483925` |
| Projected projection fraction | `0.489351` | `0.006501` | `-0.482850` |

The post-`o_proj` fractions closely track the pre-`o_proj` fractions. For the
primary anchor, `o_proj` increases absolute norm scale, but it does not create
the dominance from a small head-local contribution.

## Cross-Case Pattern

Largest projected L2-fraction drops at `post_y1/pre_x2`:

| Checkpoint | Record | Tokens | Control projected L2 frac | Masked projected L2 frac | Delta | Control pre L2 frac | Masked pre L2 frac |
|---|---:|---:|---:|---:|---:|---:|---:|
| `none_latest_ckpt32` | `33` | `1.0` | `0.490960` | `0.007035` | `-0.483925` | `0.490544` | `0.007044` |
| `aux_latest_ckpt32` | `33` | `1.0` | `0.413795` | `0.077845` | `-0.335950` | `0.413611` | `0.077914` |
| `none_latest_ckpt32` | `114` | `119.0` | `0.793181` | `0.542319` | `-0.250863` | `0.792117` | `0.541555` |
| `aux_latest_ckpt32` | `47` | `40.0` | `0.278160` | `0.076447` | `-0.201713` | `0.273198` | `0.073600` |
| `aligner_parent_ckpt1824` | `54` | `1.0` | `0.476274` | `0.294360` | `-0.181914` | `0.453844` | `0.284488` |
| `no_aligner_parent_ckpt3668` | `79` | `6.0` | `0.247106` | `0.091817` | `-0.155289` | `0.246943` | `0.091662` |

The projected readout preserves the previous value-source ranking: the same
cases with strong pre-`o_proj` duplicate-basin contribution remain strong after
projection, and the mask removes the projected contribution at nearly the same
fractional scale.

## Mechanism Update

The layer-17 output projection appears to preserve and scale a head-local
duplicate-basin contribution rather than rotating a weak contribution into a
dominant coordinate-slot direction. This shifts the next question away from
"does `o_proj` create the basin?" and toward:

- whether the pre-`o_proj` duplicate-basin value vector already aligns with the
  residual shift that pushes the coordinate slot;
- whether the projected vector aligns with the target coordinate-logit
  direction;
- whether a post-`o_proj` projected-contribution patch gives the same repair as
  the pre-`o_proj` source-contribution patch.

## Verification

- Unit tests:
  `python - <<'PY' ... pytest.main(['-q','tests/analysis/autoregressive_duplication_mechanism/test_phase4_value_source.py']) ... PY`
  passed with `8 passed`.
- Neighboring attention patch tests:
  `python - <<'PY' ... pytest.main(['-q','tests/analysis/autoregressive_duplication_mechanism/test_phase4_attention_patch.py']) ... PY`
  passed with `11 passed`.
- Syntax check:
  `python -m py_compile src/analysis/autoregressive_duplication_mechanism/phase4_value_source.py`
  passed.
- CLI check:
  `python scripts/analysis/run_autoregressive_duplication_phase4_value_source_shard.py --help`
  exposed the expected value-source shard options.
- Real-model smoke:
  `phase4_value_source_projected_layer17_head1_primary_smoke` completed with
  `168` rows and no missing projected norm/fraction fields.
- Full 8-GPU sweep:
  `phase4_value_source_projected_layer17_head1_top4_allshards_shard-00-of-08`
  through `shard-07-of-08` completed with `35168` total rows.

`projected_contribution_total_cosine` is null for `2992` zero-vector cases,
which is expected for cosine with zero norm. Projected contribution norm,
projected total norm, projected L2 fraction, and projected projection fraction
are present for all rows.

## Next Deterministic Step

Run a directional post-`o_proj` test:

1. compare projected duplicate-basin contribution with the layer-17 residual
   shift direction and coordinate-logit target direction;
2. patch the projected contribution after `o_proj` and compare repair/damage
   against the existing pre-`o_proj` source-contribution patch.

This should separate "value vector already points into the coordinate basin"
from "the residual readout only becomes coordinate-basin aligned after
downstream mixing."
