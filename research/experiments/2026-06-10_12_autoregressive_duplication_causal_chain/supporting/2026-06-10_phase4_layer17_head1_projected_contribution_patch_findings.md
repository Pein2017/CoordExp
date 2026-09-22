# Phase 4 Layer 17 Head 1 Projected-Contribution Patch Findings

## Question

The projected-direction probe showed that the layer-17 head-1 duplicate-basin
contribution survives `o_proj` and aligns with the layer-17 residual shift
removed by duplicate-basin masking. This patch asks whether that projected
contribution is causally sufficient when injected directly at the layer-17
post-attention residual boundary.

## Scope

- Artifact root: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920`
- Full-run prefix: `phase4_projected_contribution_patch_layer17_head1_top4_allshards`
- Summary JSON: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_projected_contribution_patch_layer17_head1_top4_allshards_summary.json`
- Markdown report: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_projected_contribution_patch_layer17_head1_top4_allshards_report.md`
- Smoke prefix: `phase4_projected_contribution_patch_layer17_head1_primary_smoke`
- Shards: `8`
- Rows: `2512`
- Replay cases: `30`
- Checkpoint loads across shards: `17`
- Target site: decoder layer `17`, attention head `1`
- Source region: `duplicate_basin`
- Patch site: layer-17 `post_attention_layernorm` input, labeled
  `post_attention_residual`
- Patch directions: `masked_to_control`, `control_to_masked`

Implementation support was added in this slice:

- `src/analysis/autoregressive_duplication_mechanism/phase4_projected_direction.py`
- `scripts/analysis/run_autoregressive_duplication_phase4_projected_contribution_patch_shard.py`
- `tests/analysis/autoregressive_duplication_mechanism/test_phase4_projected_direction.py`

Patch semantics:

- compute the projected duplicate-basin contribution in control and masked
  states;
- define `projected_delta = control_projected - masked_projected`;
- for `masked_to_control`, add `projected_delta` to the masked hidden state at
  the layer-17 post-attention residual boundary;
- for `control_to_masked`, add `-projected_delta` to the control hidden state
  at the same boundary.

This is an additive post-`o_proj` boundary patch, not a replacement of the
internal attention value path.

## Primary Anchor

Primary diagnostic anchor:

- checkpoint label: `none_latest_ckpt32`
- record: `33`
- role: `post_y1/pre_x2`

Mean projected-boundary patch metrics across the 12 primary-window rows:

| Direction | Control prob | Masked prob | Patched prob | Prob recovery | Prob damage | Rank recovery | Rank damage | Projected delta L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `masked_to_control` | `0.026926` | `0.013462` | `0.012079` | `-0.001382` | `-0.014847` | `-4.500` | `22.000` | `73.508282` |
| `control_to_masked` | `0.026926` | `0.013462` | `0.021049` | `0.007587` | `-0.005877` | `11.583` | `5.917` | `73.508282` |

For comparison, the earlier pre-`o_proj` source-contribution patch at the same
primary anchor reported:

| Direction | Patched prob | Prob recovery/damage | Patched rank | Rank recovery/damage |
|---|---:|---:|---:|---:|
| pre-`o_proj` `masked_to_control` | `0.032568` | recovery `+0.019106` | `7.917` | recovery `+20.917` |
| pre-`o_proj` `control_to_masked` | `0.015901` | damage `-0.011025` | `14.500` | damage `+3.167` |

The contrast is decisive for this patch site: pre-`o_proj` source-contribution
replacement repairs strongly, while additive projected-contribution injection at
the post-attention residual boundary does not repair the masked state. It even
slightly worsens the primary masked state on average. Removing the projected
delta from control does cause partial damage, so the projected contribution is
still relevant to the control computation.

## Cross-Case Pattern

Best `masked_to_control` projected-boundary recoveries at `post_y1/pre_x2` were
tiny:

| Checkpoint | Record | Tokens | Control prob | Masked prob | Patched prob | Prob recovery | Rank recovery |
|---|---:|---:|---:|---:|---:|---:|---:|
| `no_aligner_parent_ckpt3668` | `48` | `1.0` | `0.043973` | `0.034853` | `0.036437` | `0.001584` | `0.167` |
| `no_aligner_parent_ckpt3668` | `79` | `6.0` | `0.130530` | `0.121685` | `0.122643` | `0.000958` | `0.125` |
| `aligner_parent_ckpt1824` | `54` | `1.0` | `0.017791` | `0.015289` | `0.016153` | `0.000865` | `2.091` |

The strongest projected-boundary effect is therefore not sufficiency under
`masked_to_control`; it is partial control damage under `control_to_masked`,
especially for the primary anchor.

## Mechanism Update

Current interpretation:

- the layer-17 head-1 duplicate-basin route remains strongly implicated;
- the pre-`o_proj` source-contribution patch is causally sufficient for repair;
- the projected vector after `o_proj` is directionally aligned with the
  layer-17 residual shift;
- but additive injection of that projected vector at the post-attention
  residual boundary is not sufficient to recreate the control state.

This separates two claims that were previously entangled:

- **necessity / relevance**: supported by control damage and by the previous
  source-contribution patch;
- **standalone post-`o_proj` sufficiency at this boundary**: not supported.

The missing piece is likely site and interaction dependent:

- adding a large vector immediately before `post_attention_layernorm` can be
  normalized differently from the real attention path;
- same-layer MLP interaction may depend on the full attention output, not just
  the duplicate-basin component;
- the causal patch may need to target attention output before residual
  addition, post-attention norm output, or the whole head output rather than
  only the duplicate-basin projected component.

## Verification

- Unit tests:
  `python - <<'PY' ... pytest.main(['-q','tests/analysis/autoregressive_duplication_mechanism/test_phase4_projected_direction.py']) ... PY`
  passed with `4 passed`.
- Neighboring value-source and residual-patch tests:
  `python - <<'PY' ... pytest.main(['-q','tests/analysis/autoregressive_duplication_mechanism/test_phase4_value_source.py','tests/analysis/autoregressive_duplication_mechanism/test_phase4_residual_patch.py']) ... PY`
  passed with `24 passed`.
- Syntax and CLI check:
  `python -m py_compile src/analysis/autoregressive_duplication_mechanism/phase4_projected_direction.py scripts/analysis/run_autoregressive_duplication_phase4_projected_contribution_patch_shard.py`
  passed, and the shard CLI exposed the expected options.
- Real-model smoke:
  `phase4_projected_contribution_patch_layer17_head1_primary_smoke` completed
  with `24` rows.
- Full 8-GPU sweep:
  `phase4_projected_contribution_patch_layer17_head1_top4_allshards_shard-00-of-08`
  through `shard-07-of-08` completed with `2512` rows.

## Next Deterministic Step

Do a site-control sweep for the projected patch before adding more mechanism
complexity:

1. patch the same projected delta at post-attention norm output;
2. patch the whole head output contribution, not just duplicate-basin
   contribution;
3. compare those against the current post-attention residual-boundary additive
   patch and the earlier pre-`o_proj` source-contribution patch.

This should determine whether the failure is due to the contribution subset, the
normalization boundary, or the fact that only the internal pre-`o_proj` value
path preserves the right downstream computation.
