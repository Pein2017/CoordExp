# Phase 4 Layer 17 Head 1 Projected Direction Component Findings

## Question

The component patch sweep showed that the duplicate-basin component repairs the
masked coordinate state, while the non-region complement has a strong opposite
direction. This probe asks whether that opposition is introduced by `o_proj` or
is already present in the pre-`o_proj` value-source mixture.

## Scope

- Artifact root: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920`
- Full-run prefix: `phase4_projected_direction_components_layer17_head1_top4_allshards`
- Summary JSON: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_projected_direction_components_layer17_head1_top4_allshards_summary.json`
- Markdown report: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_projected_direction_components_layer17_head1_top4_allshards_report.md`
- Smoke prefix: `phase4_projected_direction_components_layer17_head1_primary_smoke`
- Shards: `8`
- Rows: `3768`
- Replay cases: `30`
- Target site/source: decoder layer `17`, attention head `1`, `duplicate_basin`
- Components:
  - `duplicate_basin`
  - `whole_head`
  - `non_region_complement`

Direction comparison:

- pre-`o_proj`: component delta compared against `W_head.T @ target_logit_direction`;
- post-`o_proj`: projected component delta compared against the residual-space
  coordinate target-logit direction.

Implementation support was added in this slice:

- `src/analysis/autoregressive_duplication_mechanism/phase4_projected_direction.py`
- `scripts/analysis/run_autoregressive_duplication_phase4_projected_direction_shard.py`
- `tests/analysis/autoregressive_duplication_mechanism/test_phase4_projected_direction.py`

## Primary Anchor

Primary diagnostic anchor:

- checkpoint label: `none_latest_ckpt32`
- record: `33`
- role: `post_y1/pre_x2`

Mean direction metrics across the 12 primary-window rows:

| Component | Pre delta L2 | Post delta L2 | Pre target cos | Post target cos | Pre target projection | Post target projection | Post residual cos |
|---|---:|---:|---:|---:|---:|---:|---:|
| `duplicate_basin` | `43.063` | `73.508` | `0.211286` | `0.034142` | `212.408108` | `16.136784` | `0.106510` |
| `whole_head` | `20.835` | `34.045` | `0.161091` | `0.027696` | `93.122946` | `7.054535` | `0.540098` |
| `non_region_complement` | `39.682` | `67.743` | `-0.059968` | `-0.008849` | `-119.285157` | `-9.082249` | `0.159157` |

## Mechanism Update

The non-region complement is already an opposing coordinate direction before
`o_proj`; the sign is not created by the output projection. `o_proj` mostly
preserves and rescales the pre-existing sign structure while mapping it into
residual space.

This means the current layer-17 head-1 picture is more specific:

- duplicate-basin visual tokens provide a coordinate-supporting value-source
  delta;
- non-region tokens in the same head provide an opposing delta;
- the whole-head delta is smaller because these components cancel;
- `o_proj` is a carrier and scaler of this component balance, not the origin of
  the opposition.

The high post-residual cosine for `whole_head` does not contradict the patch
result. It means the whole head aligns with the broad residual shift, while the
coordinate-target direction is better explained by the component split.

## Verification

- Parser/helper/row tests:
  `python - <<'PY' ... pytest.main(['-q','tests/analysis/autoregressive_duplication_mechanism/test_phase4_projected_direction.py']) ... PY`
  passed with `8 passed`.
- Syntax and CLI check:
  `python -m py_compile src/analysis/autoregressive_duplication_mechanism/phase4_projected_direction.py scripts/analysis/run_autoregressive_duplication_phase4_projected_direction_shard.py`
  passed, and the shard CLI exposed `--patch-components`.
- Real-model smoke:
  `phase4_projected_direction_components_layer17_head1_primary_smoke`
  completed with `36` rows.
- Full 8-GPU sweep:
  `phase4_projected_direction_components_layer17_head1_top4_allshards_shard-00-of-08`
  through `shard-07-of-08` completed with `3768` rows.

## Next Deterministic Step

Split the non-region complement into source subsets:

1. visual non-basin tokens;
2. text/prefix tokens;
3. special/control tokens.

The goal is to identify which source subset supplies the opposing coordinate
direction instead of treating the complement as one residual bucket.

