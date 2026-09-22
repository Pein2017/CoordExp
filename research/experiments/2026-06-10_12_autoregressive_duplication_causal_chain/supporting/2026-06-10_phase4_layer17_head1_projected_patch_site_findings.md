# Phase 4 Layer 17 Head 1 Projected Patch Site Findings

## Question

The previous projected-contribution patch failed when the projected
duplicate-basin delta was injected at the layer-17 post-attention residual
boundary. This site-control sweep asks whether the failure was due to the
projected vector itself or due to inserting it after the attention module's
native output/residual/norm path.

## Scope

- Artifact root: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920`
- Full-run prefix: `phase4_projected_contribution_patch_sites_layer17_head1_top4_allshards`
- Summary JSON: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_projected_contribution_patch_sites_layer17_head1_top4_allshards_summary.json`
- Markdown report: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_projected_contribution_patch_sites_layer17_head1_top4_allshards_report.md`
- Smoke prefix: `phase4_projected_contribution_patch_sites_layer17_head1_primary_smoke`
- Shards: `8`
- Rows: `7536`
- Replay cases: `30`
- Checkpoint loads across shards: `17`
- Target site: decoder layer `17`, attention head `1`
- Source region: `duplicate_basin`
- Patch sites:
  - `self_attn_output`
  - `post_attention_residual`
  - `post_attention_norm`
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

Mean site-sweep metrics across the 12 primary-window rows:

| Patch site | Direction | Control prob | Masked prob | Patched prob | Prob recovery | Prob damage | Rank recovery | Rank damage |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `self_attn_output` | `masked_to_control` | `0.026926` | `0.013462` | `0.032271` | `0.018809` | `0.005344` | `19.833` | `-2.333` |
| `self_attn_output` | `control_to_masked` | `0.026926` | `0.013462` | `0.015901` | `0.002440` | `-0.011025` | `13.667` | `3.833` |
| `post_attention_residual` | `masked_to_control` | `0.026926` | `0.013462` | `0.012079` | `-0.001382` | `-0.014847` | `-4.500` | `22.000` |
| `post_attention_residual` | `control_to_masked` | `0.026926` | `0.013462` | `0.021049` | `0.007587` | `-0.005877` | `11.583` | `5.917` |
| `post_attention_norm` | `masked_to_control` | `0.026926` | `0.013462` | `0.006290` | `-0.007172` | `-0.020636` | `-368.500` | `386.000` |
| `post_attention_norm` | `control_to_masked` | `0.026926` | `0.013462` | `0.007161` | `-0.006301` | `-0.019765` | `-296.167` | `313.667` |

The `self_attn_output` site nearly matches the earlier pre-`o_proj`
source-contribution repair:

| Patch | Direction | Patched prob | Prob recovery/damage | Patched rank | Rank recovery/damage |
|---|---|---:|---:|---:|---:|
| pre-`o_proj` source contribution | `masked_to_control` | `0.032568` | recovery `+0.019106` | `7.917` | recovery `+20.917` |
| projected `self_attn_output` | `masked_to_control` | `0.032271` | recovery `+0.018809` | `9.000` | recovery `+19.833` |
| pre-`o_proj` source contribution | `control_to_masked` | `0.015901` | damage `-0.011025` | `14.500` | damage `+3.167` |
| projected `self_attn_output` | `control_to_masked` | `0.015901` | damage `-0.011025` | `15.167` | damage `+3.833` |

This makes the causal site boundary much sharper than the previous negative
post-residual result.

## Cross-Case Pattern

At `post_y1/pre_x2`, mean `masked_to_control` recovery over all rows is modest
because many contrast windows have weak duplicate-basin deltas, but the site
ordering is consistent:

| Patch site | Direction | n | Mean prob recovery | Mean prob damage | Mean rank recovery | Mean rank damage |
|---|---|---:|---:|---:|---:|---:|
| `self_attn_output` | `masked_to_control` | `314` | `0.001286` | `-0.001127` | `2.290` | `4.379` |
| `post_attention_residual` | `masked_to_control` | `314` | near zero / weaker | worse than `self_attn_output` | weaker | larger damage |
| `post_attention_norm` | `masked_to_control` | `314` | negative | strongly damaging | strongly negative | strongly damaging |

The primary anchor and strongest cases show that the projected duplicate-basin
contribution is sufficient when injected at the attention module output, not
when injected later.

## Mechanism Update

The deepest current mechanism picture is now:

- layer-17 head-1 attends to a local duplicate visual basin;
- the duplicate-basin value contribution is large and survives the head's
  output projection;
- the projected contribution aligns with the layer-17 residual shift removed by
  duplicate-basin masking;
- replacing or injecting it at `self_attn_output` repairs the masked coordinate
  decision almost as strongly as the pre-`o_proj` source-contribution patch;
- injecting the same projected delta after attention residual construction or
  after post-attention norm fails.

So the causal route is not merely "a vector somewhere in residual space." It is
site-sensitive: the projected duplicate-basin contribution must enter through
the attention module output path before the layer's residual/norm machinery.
That explains the prior post-residual negative result without weakening the
layer-17 head-1 mechanism claim.

## Verification

- Unit tests:
  `python - <<'PY' ... pytest.main(['-q','tests/analysis/autoregressive_duplication_mechanism/test_phase4_projected_direction.py']) ... PY`
  passed with `6 passed`.
- Neighboring value-source and residual-patch tests:
  `python - <<'PY' ... pytest.main(['-q','tests/analysis/autoregressive_duplication_mechanism/test_phase4_value_source.py','tests/analysis/autoregressive_duplication_mechanism/test_phase4_residual_patch.py']) ... PY`
  passed with `24 passed`.
- Syntax and CLI check:
  `python -m py_compile src/analysis/autoregressive_duplication_mechanism/phase4_projected_direction.py scripts/analysis/run_autoregressive_duplication_phase4_projected_contribution_patch_shard.py`
  passed, and the shard CLI exposed `--patch-sites`.
- Real-model smoke:
  `phase4_projected_contribution_patch_sites_layer17_head1_primary_smoke`
  completed with `72` rows.
- Full 8-GPU sweep:
  `phase4_projected_contribution_patch_sites_layer17_head1_top4_allshards_shard-00-of-08`
  through `shard-07-of-08` completed with `7536` rows.

## Next Deterministic Step

The next clean probe is a contribution subset control at the successful
`self_attn_output` site:

1. patch whole head output at `self_attn_output`;
2. compare it to duplicate-basin-only projected contribution;
3. optionally subtract non-basin visual contribution to test whether the
   duplicate-basin component is specifically sufficient or whether the whole
   head output carries a necessary companion term.

This separates "local duplicate-basin projected value is enough" from "the
whole attention head output must be restored, with duplicate basin as the
dominant part."
