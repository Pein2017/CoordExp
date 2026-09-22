# Phase 4 Layer 17 Head 1 Projected Direction Findings

## Question

The projected value-source readout showed that layer-17 head-1 `o_proj`
preserves the duplicate-basin contribution's dominance. This probe asks whether
the projected duplicate-basin contribution points along the layer-17 residual
shift induced by duplicate-basin masking, and whether it already points toward a
raw coordinate-logit target direction.

## Scope

- Artifact root: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920`
- Full-run prefix: `phase4_projected_direction_layer17_head1_top4_allshards`
- Summary JSON: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_projected_direction_layer17_head1_top4_allshards_summary.json`
- Markdown report: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_projected_direction_layer17_head1_top4_allshards_report.md`
- Smoke prefix: `phase4_projected_direction_layer17_head1_primary_smoke`
- Shards: `8`
- Rows: `1256`
- Replay cases: `30`
- Checkpoint loads across shards: `17`
- Target site: decoder layer `17`, attention head `1`
- Source region: `duplicate_basin`

Implementation support was added in this slice:

- `src/analysis/autoregressive_duplication_mechanism/phase4_projected_direction.py`
- `scripts/analysis/run_autoregressive_duplication_phase4_projected_direction_shard.py`
- `tests/analysis/autoregressive_duplication_mechanism/test_phase4_projected_direction.py`

Direction definitions:

- projected contribution: duplicate-basin value contribution after multiplying
  the selected head slice by the layer-17 attention `o_proj` head-column block;
- residual shift: `control - duplicate_basin_mask` layer-17 decoder-layer
  output at the same token position, so positive alignment means the projected
  contribution points in the direction removed by masking;
- target-logit direction: target coordinate-token unembedding minus the mean
  coordinate-token unembedding.

## Primary Anchor

Primary diagnostic anchor:

- checkpoint label: `none_latest_ckpt32`
- record: `33`
- role: `post_y1/pre_x2`

Offset progression:

| Row | Offset | Control projected L2 | Masked projected L2 | Residual shift L2 | Delta-vs-residual cosine | Delta-vs-residual projection fraction | Control-vs-target cosine |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `20` | `-3` | `0.082661` | `0.002960` | `11.471893` | `-0.030586` | `-0.000212` | `0.034222` |
| `21` | `-2` | `0.350733` | `0.003975` | `10.349277` | `-0.139625` | `-0.004680` | `0.034280` |
| `22` | `-1` | `0.000634` | `0.000214` | `9.908087` | `-0.164186` | `-0.000005` | `0.034257` |
| `23` | `0` | `110.242310` | `0.960658` | `79.330269` | `0.086376` | `0.119684` | `0.034251` |
| `25` | `2` | `99.329498` | `0.774117` | `77.946678` | `0.064000` | `0.080833` | `0.034231` |
| `26` | `3` | `119.087448` | `1.047764` | `80.875717` | `0.178580` | `0.263321` | `0.034252` |
| `28` | `5` | `130.722443` | `1.926777` | `78.186119` | `0.304019` | `0.505876` | `0.034290` |
| `29` | `6` | `129.295746` | `0.873318` | `77.355560` | `0.330224` | `0.547658` | `0.034258` |
| `31` | `8` | `148.280807` | `2.204705` | `85.742790` | `0.499754` | `0.851872` | `0.034264` |

The onset-local pattern is now sharper:

- before onset, duplicate-basin projected contribution is near zero and
  residual alignment is near zero or negative;
- from onset onward, projected contribution becomes large;
- post-onset rows increasingly align with the layer-17 residual shift removed
  by duplicate-basin masking;
- the strongest primary row reaches about `0.50` cosine and `0.85` projection
  fraction against the residual shift.

## Cross-Case Pattern

Strongest `post_y1/pre_x2` residual alignment cases by mean
`delta_projected_vs_residual_cosine`:

| Checkpoint | Record | Tokens | Control projected L2 | Residual shift L2 | Delta-vs-residual cosine | Delta-vs-residual projection fraction | Control-vs-target cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| `none_latest_ckpt32` | `114` | `119.0` | `130.789058` | `131.011389` | `0.494346` | `0.450769` | `-0.005271` |
| `aux_latest_ckpt32` | `54` | `1.0` | `21.847185` | `14.599629` | `0.348635` | `0.215065` | `0.011358` |
| `aux_latest_ckpt32` | `47` | `40.0` | `60.357020` | `36.253961` | `0.317499` | `0.171198` | `0.004976` |
| `none_latest_ckpt32` | `33` | `1.0` | `61.857393` | `39.354152` | `0.122397` | `0.201480` | `0.034259` |
| `aux_latest_ckpt32` | `33` | `1.0` | `46.723965` | `45.779966` | `0.105153` | `0.109753` | `-0.003742` |

This confirms that the primary anchor is not unique, but the effect is
heterogeneous. Some cases have large projected contribution without strong
alignment to the selected residual-shift direction, which is exactly the kind
of contrast the next causal patch should separate.

## Mechanism Update

Current deepest picture:

- layer-17 head-1 duplicate-basin value contribution survives `o_proj`;
- after `o_proj`, that contribution aligns with the layer-17 residual shift
  that disappears under duplicate-basin masking, especially after onset;
- the same vector is not strongly aligned with a raw centered coordinate
  unembedding direction at layer 17.

The target-unembedding result should be interpreted cautiously: the centered
target-logit direction has small norm in some cases, so target projection
fractions are not stable evidence. Target cosine is more stable, and it remains
small. This suggests the layer-17 route is an intermediate residual-state
mechanism rather than a direct final-logit write by itself.

## Verification

- Unit tests:
  `python - <<'PY' ... pytest.main(['-q','tests/analysis/autoregressive_duplication_mechanism/test_phase4_projected_direction.py']) ... PY`
  passed with `2 passed`.
- Neighboring value-source tests:
  `python - <<'PY' ... pytest.main(['-q','tests/analysis/autoregressive_duplication_mechanism/test_phase4_value_source.py']) ... PY`
  passed with `8 passed`.
- Syntax and CLI check:
  `python -m py_compile src/analysis/autoregressive_duplication_mechanism/phase4_projected_direction.py scripts/analysis/run_autoregressive_duplication_phase4_projected_direction_shard.py`
  passed, and the shard CLI exposed the expected options.
- Real-model smoke:
  `phase4_projected_direction_layer17_head1_primary_smoke` completed with `12`
  rows.
- Full 8-GPU sweep:
  `phase4_projected_direction_layer17_head1_top4_allshards_shard-00-of-08`
  through `shard-07-of-08` completed with `1256` rows.

Cosine fields are null for `520` rows where one compared vector has zero norm;
these are mostly zero-contribution positions and are expected. Projection
fraction fields are present for all rows because the residual-shift denominator
is nonzero in this run.

## Next Deterministic Step

Patch the projected post-`o_proj` contribution at the layer-17 residual
boundary:

1. `masked_to_control`: add or replace the projected duplicate-basin
   contribution in the masked state at the layer-17 residual boundary;
2. `control_to_masked`: remove or replace it in the control state;
3. compare probability/rank repair with the existing pre-`o_proj`
   source-contribution patch.

If projected post-`o_proj` patching matches pre-`o_proj` repair, the mechanism
is almost fully localized to the layer-17 head-1 projected contribution. If it
does not, the missing piece is downstream interaction after the layer-17
residual boundary.
