# Phase 4 Layer 17 Head 1 Projected Patch Component Findings

## Question

The previous site sweep showed that the projected duplicate-basin contribution
repairs the primary coordinate decision when injected at `self_attn_output`, but
not at later residual/norm sites. This component sweep asks whether that repair
is a property of the duplicate-basin component specifically, or whether the
whole selected attention head output must be restored.

## Scope

- Artifact root: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920`
- Full-run prefix: `phase4_projected_contribution_patch_components_layer17_head1_top4_allshards`
- Summary JSON: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_projected_contribution_patch_components_layer17_head1_top4_allshards_summary.json`
- Markdown report: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_projected_contribution_patch_components_layer17_head1_top4_allshards_report.md`
- Smoke prefix: `phase4_projected_contribution_patch_components_layer17_head1_primary_smoke`
- Shards: `8`
- Rows: `7536`
- Replay cases: `30`
- Target site: decoder layer `17`, attention head `1`, `self_attn_output`
- Source region: `duplicate_basin`
- Patch components:
  - `duplicate_basin`
  - `whole_head`
  - `non_region_complement`
- Patch directions: `masked_to_control`, `control_to_masked`

Implementation support was added in this slice:

- `src/analysis/autoregressive_duplication_mechanism/phase4_projected_direction.py`
- `scripts/analysis/run_autoregressive_duplication_phase4_projected_contribution_patch_shard.py`
- `tests/analysis/autoregressive_duplication_mechanism/test_phase4_projected_direction.py`

## Primary Anchor

Primary diagnostic anchor:

- checkpoint label: `none_latest_ckpt32`
- record: `33`
- role: `post_y1/pre_x2`

Mean component metrics across the 12 primary-window rows:

| Component | Direction | Control prob | Masked prob | Patched prob | Prob recovery | Prob damage | Rank recovery | Rank damage | Delta L2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `duplicate_basin` | `masked_to_control` | `0.026926` | `0.013462` | `0.032271` | `0.018809` | `0.005344` | `19.833` | `-2.333` | `73.508` |
| `duplicate_basin` | `control_to_masked` | `0.026926` | `0.013462` | `0.015901` | `0.002440` | `-0.011025` | `13.667` | `3.833` | `73.508` |
| `whole_head` | `masked_to_control` | `0.026926` | `0.013462` | `0.026106` | `0.012645` | `-0.000820` | `18.417` | `-0.917` | `34.045` |
| `whole_head` | `control_to_masked` | `0.026926` | `0.013462` | `0.015095` | `0.001633` | `-0.011831` | `2.167` | `15.333` | `34.045` |
| `non_region_complement` | `masked_to_control` | `0.026926` | `0.013462` | `0.014629` | `0.001168` | `-0.012297` | `10.583` | `6.917` | `67.743` |
| `non_region_complement` | `control_to_masked` | `0.026926` | `0.013462` | `0.032948` | `0.019487` | `0.006022` | `19.583` | `-2.083` | `67.743` |

## Mechanism Update

The component split reveals cancellation inside the whole-head projected delta.
At the primary anchor, `duplicate_basin` is the strongest repair component for
`masked_to_control`, while `non_region_complement` is strongest in the opposite
`control_to_masked` direction. The whole-head delta is smaller than either
component in L2 and has weaker coordinate repair, consistent with partial
opposition between the region and non-region projected deltas.

This sharpens the current mechanism:

- the duplicate-basin value path is not merely a proxy for the whole selected
  attention head;
- the duplicate-basin component carries a specific attention-output repair
  direction for the masked state;
- the non-region complement carries a companion/opposing direction that is more
  effective when subtracted from the control state;
- the final picture likely depends on component balance inside the attention
  output, not only the total selected-head contribution magnitude.

The cross-case means remain smaller than the primary anchor, as in the site
sweep, but the component split is high-value because it exposes why the whole
head output can underperform the targeted duplicate-basin component.

## Verification

- Parser/row unit tests:
  `python - <<'PY' ... pytest.main(['-q','tests/analysis/autoregressive_duplication_mechanism/test_phase4_projected_direction.py']) ... PY`
  passed with `7 passed`.
- Syntax and CLI check:
  `python -m py_compile src/analysis/autoregressive_duplication_mechanism/phase4_projected_direction.py scripts/analysis/run_autoregressive_duplication_phase4_projected_contribution_patch_shard.py`
  passed, and the shard CLI exposed `--patch-components`.
- Real-model smoke:
  `phase4_projected_contribution_patch_components_layer17_head1_primary_smoke`
  completed with `72` rows.
- Full 8-GPU sweep:
  `phase4_projected_contribution_patch_components_layer17_head1_top4_allshards_shard-00-of-08`
  through `shard-07-of-08` completed with `7536` rows.

## Next Deterministic Step

The next clean probe is a component-aware direction decomposition:

1. compare duplicate-basin and non-region-complement projected deltas against
   the target coordinate logit direction;
2. measure whether their opposition is already visible before projection or is
   introduced/amplified by `o_proj`;
3. if the opposition is stable, test per-source visual/text subsets inside the
   non-region complement rather than treating it as one residual bucket.

