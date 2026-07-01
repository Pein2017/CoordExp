# Phase 4 Layer 17 Head 1 Value-Source Contribution Patch Findings

## Question

The value-source readout showed that the duplicate-basin visual cell supplies a
large layer-17 head-1 value contribution in the primary control state. This
patch test asks whether that source contribution is causal, not just correlated:

- `masked_to_control`: replace the masked state's duplicate-basin source
  contribution with the control source contribution inside the layer-17 head-1
  pre-`o_proj` slice.
- `control_to_masked`: replace the control state's duplicate-basin source
  contribution with the masked source contribution inside the same slice.

## Scope

- Artifact root: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920`
- Full-run prefix: `phase4_value_source_patch_layer17_head1_top4_allshards`
- Summary JSON: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_value_source_patch_layer17_head1_top4_allshards_summary.json`
- Markdown report: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_value_source_patch_layer17_head1_top4_allshards_report.md`
- Primary smoke: `phase4_value_source_patch_layer17_head1_primary_smoke_v2`
- Shards: `8`
- Patch rows: `1472`
- Replay cases: `30`
- Checkpoint loads across shards: `17`
- Target site: decoder layer `17`, attention head `1`
- Source region: `duplicate_basin`
- Directions: `masked_to_control`, `control_to_masked`

Implementation support was added in commit `20458408`:

- `AttentionSourceContributionPatchHook`
- `attention_value_source_contribution`
- `collect_model_value_source_patch_for_case`
- `materialize_model_value_source_patch_for_shard`
- `scripts/analysis/run_autoregressive_duplication_phase4_value_source_patch_shard.py`

The patch changes only the selected source-region contribution inside the
selected head slice. It does not replace the whole attention head output.

## Primary Anchor

Primary diagnostic anchor:

- checkpoint label: `none_latest_ckpt32`
- record: `33`
- phase: `post_y1/pre_x2`

Aggregate result:

| Direction | n | Control prob | Masked prob | Patched prob | Prob recovery | Prob damage | Control rank | Masked rank | Patched rank | Rank recovery | Rank damage | Old L2 | Replacement L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `masked_to_control` | `12` | `0.026926` | `0.013462` | `0.032568` | `0.019106` | `0.005642` | `11.333` | `28.833` | `7.917` | `20.917` | `-3.417` | `0.563950` | `43.601439` |
| `control_to_masked` | `12` | `0.026926` | `0.013462` | `0.015901` | `0.002440` | `-0.011025` | `11.333` | `28.833` | `14.500` | `14.333` | `3.167` | `43.601439` | `0.563950` |

This is the strongest causal evidence so far. Inserting only the control
duplicate-basin contribution into the masked state recovers the target
coordinate probability and rank beyond the unpatched control mean for several
onset rows. Removing the contribution from control damages probability, but it
is not a perfect mirror because some residual/context paths still preserve part
of the target direction.

Representative onset rows for `masked_to_control`:

| Row | Relative offset | Control p/r | Masked p/r | Patched p/r | Prob recovery | Rank recovery |
|---:|---:|---:|---:|---:|---:|---:|
| `23` | `0` | `0.031918/5` | `0.005090/55` | `0.037187/4` | `0.032098` | `51` |
| `25` | `2` | `0.026650/12` | `0.011887/31` | `0.047842/1` | `0.035955` | `30` |
| `26` | `3` | `0.028764/13` | `0.008378/43` | `0.041422/2` | `0.033043` | `41` |
| `27` | `4` | `0.032935/7` | `0.019467/17` | `0.045263/1` | `0.025796` | `16` |
| `29` | `6` | `0.026678/14` | `0.010241/39` | `0.035617/6` | `0.025376` | `33` |
| `30` | `7` | `0.033612/7` | `0.015796/24` | `0.040123/3` | `0.024327` | `21` |

## Cross-Case Pattern

The causal signal is real but not globally uniform.

Strongest `post_y1/pre_x2` `masked_to_control` repairs:

| Checkpoint | Record | n | Prob recovery | Rank recovery | Prob damage | Rank damage |
|---|---:|---:|---:|---:|---:|---:|
| `none_latest_ckpt32` | `33` | `12` | `0.019106` | `20.917` | `0.005642` | `-3.417` |
| `aux_latest_ckpt32` | `79` | `5` | `0.004900` | `9.200` | `-0.002337` | `1.200` |
| `no_aligner_parent_ckpt3668` | `48` | `12` | `0.004784` | `1.917` | `-0.004336` | `1.750` |
| `none_latest_ckpt32` | `114` | `5` | `0.004596` | `9.000` | `-0.007071` | `13.400` |

Strongest `control_to_masked` probability damages:

| Checkpoint | Record | n | Prob recovery | Rank recovery | Prob damage | Rank damage |
|---|---:|---:|---:|---:|---:|---:|
| `none_latest_ckpt32` | `33` | `12` | `0.002440` | `14.333` | `-0.011025` | `3.167` |
| `none_latest_ckpt32` | `114` | `5` | `0.004582` | `-41.000` | `-0.007085` | `63.400` |
| `aux_latest_ckpt32` | `33` | `12` | `-0.003847` | `-1.167` | `-0.006981` | `-0.083` |
| `no_aligner_parent_ckpt3668` | `79` | `8` | `0.001869` | `-3.250` | `-0.006976` | `-8.750` |

The all-row average is much weaker than the primary anchor because the sweep
includes non-onset rows, empty or very low duplicate-basin contribution rows,
and heterogeneous contrast cases. Therefore the right interpretation is
onset-local causality, not a universal every-row head rule.

## Mechanism Update

The mechanism picture now has causal support:

- layer-17 head 1 is a visual-source head;
- the relevant route is local to the duplicate-basin visual cell;
- the duplicate-basin value contribution is sufficient to repair the masked
  coordinate-slot target in the primary onset anchor;
- removing that contribution from control damages the target probability,
  showing partial necessity;
- the asymmetry means other context or residual routes can still carry part of
  the coordinate target, so this is a core route rather than the only route.

## Next Deterministic Step

Run one of two targeted follow-ups:

1. Patch control duplicate-basin contribution into masked state across adjacent
   layers around the transition (`14`, `15`, `16`, `17`, `18`) to locate where
   this source-contribution route first becomes sufficient.
2. Decompose whether the repaired target direction comes from the visual value
   vector itself or from the `o_proj` orientation by patching the pre-`o_proj`
   source contribution versus post-`o_proj` projected contribution.

The first option is the cleaner next step if we want a layer-transition story;
the second is the cleaner next step if we want the geometric basis of the
coordinate-slot attraction.

