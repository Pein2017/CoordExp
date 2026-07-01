# Phase 4 Layer-Transition Source-Contribution Patch Findings

## Question

The previous source-contribution patch established that layer-17 head-1
duplicate-basin contribution is causally sufficient for strong primary-anchor
repair. This sweep asks whether that causal route appears gradually across the
nearby transition layers or sharply at layer `17`.

## Scope

- Artifact root: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920`
- Full-run prefix: `phase4_value_source_patch_layers14_18_head1_top4_allshards`
- Summary JSON: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_value_source_patch_layers14_18_head1_top4_allshards_summary.json`
- Markdown report: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_value_source_patch_layers14_18_head1_top4_allshards_report.md`
- Primary smoke: `phase4_value_source_patch_layers14_18_head1_primary_smoke`
- Layers: `14`, `15`, `16`, `17`, `18`
- Head: `1`
- Source region: `duplicate_basin`
- Patch directions: `masked_to_control`, `control_to_masked`
- Full rows: `7360`
- Replay cases: `30`
- Checkpoint loads across shards: `17`

The protocol is the same source-contribution patch used in commit `20458408`:
replace only the selected `duplicate_basin` value contribution inside the
selected head's pre-`o_proj` slice. It does not patch the whole attention head
output.

## Primary Anchor

Primary diagnostic anchor:

- checkpoint label: `none_latest_ckpt32`
- record: `33`
- phase: `post_y1/pre_x2`

`masked_to_control` by layer:

| Layer | Patched prob | Prob recovery | Patched rank | Rank recovery | Old L2 | Replacement L2 |
|---:|---:|---:|---:|---:|---:|---:|
| `14` | `0.013356` | `-0.000106` | `28.500` | `0.333` | `0.004540` | `0.007550` |
| `15` | `0.013467` | `0.000006` | `29.083` | `-0.250` | `0.001059` | `0.002086` |
| `16` | `0.013500` | `0.000039` | `28.917` | `-0.083` | `0.001016` | `0.004059` |
| `17` | `0.032568` | `0.019106` | `7.917` | `20.917` | `0.563950` | `43.601439` |
| `18` | `0.013345` | `-0.000117` | `29.250` | `-0.417` | `0.000098` | `0.000920` |

`control_to_masked` by layer:

| Layer | Patched prob | Prob damage | Patched rank | Rank damage | Old L2 | Replacement L2 |
|---:|---:|---:|---:|---:|---:|---:|
| `14` | `0.027807` | `0.000881` | `10.333` | `-1.000` | `0.007550` | `0.004540` |
| `15` | `0.027161` | `0.000235` | `11.167` | `-0.167` | `0.002086` | `0.001059` |
| `16` | `0.027218` | `0.000292` | `11.167` | `-0.167` | `0.004059` | `0.001016` |
| `17` | `0.015901` | `-0.011025` | `14.500` | `3.167` | `43.601439` | `0.563950` |
| `18` | `0.026835` | `-0.000091` | `11.250` | `-0.083` | `0.000920` | `0.000098` |

Layer `17` is the only tested layer with large `masked_to_control` recovery in
the primary anchor. The non-17 layers stay effectively at the masked baseline
even though the same patch operation is applied.

## Onset Rows

Representative `masked_to_control` rows:

| Layer | Row | Relative offset | Masked p/r | Patched p/r | Prob recovery | Rank recovery | Replacement L2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `17` | `23` | `0` | `0.005090/55` | `0.037187/4` | `0.032098` | `51` | `60.558849` |
| `17` | `25` | `2` | `0.011887/31` | `0.047842/1` | `0.035955` | `30` | `60.558849` |
| `17` | `26` | `3` | `0.008378/43` | `0.041422/2` | `0.033043` | `41` | `78.089043` |
| `17` | `27` | `4` | `0.019467/17` | `0.045263/1` | `0.025796` | `16` | `39.243725` |
| `17` | `29` | `6` | `0.010241/39` | `0.035617/6` | `0.025376` | `33` | `70.519180` |
| `17` | `30` | `7` | `0.015796/24` | `0.040123/3` | `0.024327` | `21` | `47.411205` |

Adjacent layers do not show comparable onset-row repair. Their replacement
contribution L2 values are also near zero in the primary anchor.

## Cross-Case Pattern

Layer-17 dominance is visible in the cases with meaningful probability repair,
but the cross-case picture remains heterogeneous. Strong layer-17 dominance
examples at `post_y1/pre_x2`:

| Checkpoint | Record | L17 prob recovery | Max non17 prob recovery | Margin | L17 rank recovery | Max non17 rank recovery | L17 replacement L2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `none_latest_ckpt32` | `33` | `0.019106` | `0.000039` | `0.019068` | `20.917` | `0.333` | `43.601439` |
| `aux_latest_ckpt32` | `79` | `0.004900` | `0.000051` | `0.004849` | `9.200` | `1.000` | `35.582244` |
| `no_aligner_parent_ckpt3668` | `48` | `0.004784` | `0.000184` | `0.004600` | `1.917` | `0.083` | `40.358287` |
| `none_latest_ckpt32` | `114` | `0.004596` | `0.000172` | `0.004424` | `9.000` | `2.600` | `77.863911` |

Some records have weak or negative layer-17 probability margins. These are not
contradictions of the primary mechanism; they are contrast cases where the
selected duplicate-basin route is not the dominant source of the target
coordinate slot.

## Mechanism Update

The layer-transition result sharpens the picture:

- the duplicate-basin value route does not gradually strengthen from layers
  `14` to `18`;
- it is sharply localized to layer `17` for the primary onset anchor;
- layer `17` is also where the source contribution L2 jumps by orders of
  magnitude;
- layer `18` no longer exposes the same source contribution at this pre-`o_proj`
  site, consistent with the route having already been mixed into later residual
  state rather than remaining as the same head-local source patch target.

This supports a mechanism boundary: the causal visual-basin contribution enters
the coordinate-slot computation through layer-17 head-1, then downstream layers
inherit its residual effect rather than recomputing the same source route.

## Next Deterministic Step

The clean follow-up is a post-`o_proj` contribution decomposition:

- project the duplicate-basin pre-`o_proj` contribution through layer-17 head-1
  `o_proj`;
- compare it with the residual shift direction and the coordinate-logit target
  direction;
- test whether patching the projected contribution after `o_proj` gives the
  same repair as pre-`o_proj` contribution patching.

That would separate two remaining mechanisms: whether the coordinate-slot basin
is primarily in the visual value vector itself, or in the orientation supplied
by the layer-17 output projection.

