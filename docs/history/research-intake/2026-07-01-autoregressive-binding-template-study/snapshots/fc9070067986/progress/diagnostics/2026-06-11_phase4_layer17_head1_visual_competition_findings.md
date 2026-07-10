# Phase 4 Layer-17 Head-1 Visual Competition Findings

Date: 2026-06-11

## Scope

This note connects the layer-17 head-1 value-source repair result with existing
projected-direction and projected-patch artifacts. The purpose is to sharpen the
mechanism from "duplicate-basin source contribution repairs the coord basin" to
"which source subsets inside the same head support or oppose the coordinate
direction."

No new GPU run was needed for this slice. The relevant all-shard artifacts
already existed under:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920
```

The key inputs were:

```text
phase4_projected_direction_components_layer17_head1_top4_allshards_report.md
phase4_projected_direction_source_buckets_layer17_head1_top4_allshards_report.md
phase4_projected_direction_visual_spatial_buckets_layer17_head1_top4_allshards_report.md
phase4_projected_patch_visual_spatial_buckets_layer17_head1_top4_allshards_report.md
```

## Artifact contracts

Projected direction rows:

- `phase4_projected_direction_layer17_head1_top4_allshards`: 1,256 rows
- `phase4_projected_direction_components_layer17_head1_top4_allshards`: 3,768 rows
- `phase4_projected_direction_source_buckets_layer17_head1_top4_allshards`: 7,536 rows
- `phase4_projected_direction_visual_spatial_buckets_layer17_head1_top4_allshards`: 10,048 rows

Projected visual-spatial patch rows:

- `phase4_projected_patch_visual_spatial_buckets_layer17_head1_top4_allshards`: 15,072 rows
- expected rows in report: 15,072
- `contract_ok=True`

The projected-direction row schema includes both:

- pre-`o_proj` delta against pulled-back target-coordinate direction:
  `delta_pre_o_proj_vs_target_logit_cosine`,
  `delta_pre_o_proj_target_logit_projection_fraction`
- post-`o_proj` projected delta against residual-space target-coordinate
  direction:
  `delta_projected_vs_target_logit_cosine`,
  `delta_projected_target_logit_projection_fraction`

## Primary directional decomposition

Primary slice:

- checkpoint: `none_latest_ckpt32`
- record: 33
- phase: `post_y1/pre_x2`
- layer/head: 17/1
- rows per component: 12

### Component split

| Component | Pre delta L2 | Post delta L2 | Pre target cosine | Post target cosine | Pre target projection | Post target projection | Post residual cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| `duplicate_basin` | 43.063 | 73.508 | 0.211286 | 0.034142 | 212.408108 | 16.136784 | 0.106510 |
| `non_region_complement` | 39.682 | 67.743 | -0.059968 | -0.008849 | -119.285157 | -9.082249 | 0.159157 |
| `whole_head` | 20.835 | 34.045 | 0.161091 | 0.027696 | 93.122946 | 7.054535 | 0.540098 |

Read:

- The duplicate-basin component points positively toward the coordinate target
  direction before and after `o_proj`.
- The non-region complement points in the opposite coordinate direction.
- The whole-head direction is still coordinate-supporting, but it is a
  cancellation product: duplicate-basin support minus non-region opposition.

This means the coordinate support is not created by `o_proj` from an arbitrary
value vector. The sign structure already exists pre-`o_proj`; `o_proj` maps it
into residual space while preserving the support/opposition split.

## Source-bucket localization

Primary `none_latest_ckpt32`, record 33, `post_y1/pre_x2`:

| Component | Token count | Pre target projection | Post target projection | Pre target cosine | Post target cosine | Post residual cosine |
|---|---:|---:|---:|---:|---:|---:|
| `duplicate_basin` | 1.0 | 212.408108 | 16.136784 | 0.211286 | 0.034142 | 0.106510 |
| `visual_non_basin` | 339.0 | -118.689661 | -9.004802 | -0.059451 | -0.008469 | 0.157225 |
| `text_prefix` | 484.0 | -0.579180 | -0.044032 | -0.014865 | -0.002968 | 0.103453 |
| `special_control` | 151.0 | -0.016335 | -0.001241 | -0.038857 | -0.006089 | -0.010122 |
| `non_region_complement` | 974.0 | -119.285157 | -9.082249 | -0.059968 | -0.008849 | 0.159157 |

Read:

- Almost all opposing coordinate projection comes from other visual tokens, not
  from text prefix or special/control tokens.
- Text/special buckets can have negative signs, but their magnitudes are tiny
  compared with visual non-basin opposition.
- The mechanism is therefore visual competition inside layer-17 head 1, not a
  language-template or special-token artifact.

## Visual-spatial localization

Primary `none_latest_ckpt32`, record 33, `post_y1/pre_x2`:

| Component | Token count | Pre target projection | Post target projection | Post target cosine | Post residual cosine |
|---|---:|---:|---:|---:|---:|
| `duplicate_basin` | 1.0 | 212.408108 | 16.136784 | 0.034142 | 0.106510 |
| `visual_near_ring` | 8.0 | -120.648186 | -9.164838 | -0.010813 | -0.016893 |
| `visual_far_background` | 331.0 | 1.958537 | 0.145139 | 0.007726 | 0.277686 |
| `visual_non_basin` | 339.0 | -118.689661 | -9.004802 | -0.008469 | 0.157225 |
| `whole_head` | 975.0 | 93.122946 | 7.054535 | 0.027696 | 0.540098 |

Read:

- The negative visual opposition is spatially local.
- `visual_near_ring`, only 8 tokens around the duplicate-basin visual token,
  explains essentially all of the negative `visual_non_basin` target projection.
- `visual_far_background`, despite 331 tokens, is small and slightly positive
  in the primary anchor.

This sharpens the picture from broad visual competition to local visual
neighborhood competition around the duplicated object/coordinate basin.

## Cross-case directional pattern

At all `post_y1/pre_x2` windows:

| Component | n | Pre target projection | Post target projection | Pre target cosine | Post target cosine |
|---|---:|---:|---:|---:|---:|
| `duplicate_basin` | 314 | 27.645129 | 2.100755 | 0.074990 | 0.012725 |
| `visual_near_ring` | 314 | -30.645097 | -2.324216 | -0.060344 | -0.010142 |
| `visual_far_background` | 314 | 5.351415 | 0.409280 | 0.001039 | 0.000370 |
| `visual_non_basin` | 314 | -25.293679 | -1.911581 | -0.025210 | -0.004153 |
| `whole_head` | 314 | 2.334946 | 0.184562 | 0.007804 | 0.001424 |

Cross-checkpoint post-`y1`/pre-`x2` means:

| Checkpoint | Duplicate pre projection | Near-ring pre projection | Far/background pre projection | Visual-non-basin pre projection |
|---|---:|---:|---:|---:|
| `aligner_parent_ckpt1824` | 1.121756 | -3.530995 | 65.488154 | 61.957165 |
| `aux_latest_ckpt32` | 58.803849 | -10.175707 | -13.604822 | -23.780525 |
| `no_aligner_parent_ckpt3668` | 17.613940 | -24.224646 | -4.980722 | -29.205365 |
| `none_latest_ckpt32` | 30.032609 | -94.453961 | -7.308979 | -101.762937 |

Read:

- The primary local-near-ring opposition is not merely an averaging artifact.
  It remains visible in the `none_latest_ckpt32` family average.
- The parent checkpoints differ materially. `aligner_parent_ckpt1824` has
  positive far/background projection large enough to swamp the near-ring
  opposition, while `none_latest_ckpt32` has strong near-ring opposition and
  mildly negative far/background.
- This supports treating checkpoint family as part of the mechanism, even while
  keeping the report labels stripped of broad checkpoint-family claims.

## Causal visual-spatial patch

Artifact:

```text
phase4_projected_patch_visual_spatial_buckets_layer17_head1_top4_allshards_report.md
```

Patch site:

```text
self_attn_output
```

Primary `none_latest_ckpt32`, record 33, `post_y1/pre_x2`:

| Component | Direction | n | Prob recovery | Prob damage | Rank recovery | Rank damage | Projected delta L2 |
|---|---|---:|---:|---:|---:|---:|---:|
| `duplicate_basin` | `masked_to_control` | 12 | 0.018809 | 0.005344 | 19.833333 | -2.333333 | 73.508282 |
| `duplicate_basin` | `control_to_masked` | 12 | 0.002440 | -0.011025 | 13.666667 | 3.833333 | 73.508282 |
| `visual_near_ring` | `masked_to_control` | 12 | 0.001532 | -0.011933 | 12.583333 | 4.916667 | 65.306174 |
| `visual_near_ring` | `control_to_masked` | 12 | 0.019530 | 0.006066 | 20.000000 | -2.500000 | 65.306174 |
| `visual_far_background` | `masked_to_control` | 12 | 0.000471 | -0.012994 | 1.583333 | 15.916667 | 3.635632 |
| `visual_far_background` | `control_to_masked` | 12 | 0.014434 | 0.000970 | 18.250000 | -0.750000 | 3.635632 |
| `visual_non_basin` | `masked_to_control` | 12 | 0.001074 | -0.012390 | 10.750000 | 6.750000 | 67.763769 |
| `visual_non_basin` | `control_to_masked` | 12 | 0.019883 | 0.006419 | 20.166667 | -2.666667 | 67.763769 |
| `whole_head` | `masked_to_control` | 12 | 0.012645 | -0.000820 | 18.416667 | -0.916667 | 34.044507 |
| `whole_head` | `control_to_masked` | 12 | 0.001633 | -0.011831 | 2.166667 | 15.333333 | 34.044507 |

Read:

- `duplicate_basin` is the repair component: `masked_to_control` strongly
  recovers the coordinate target.
- `visual_near_ring` is the anti-repair or masked-like component:
  `control_to_masked` strongly moves control toward the masked behavior, while
  `masked_to_control` barely repairs.
- `visual_non_basin` behaves like `visual_near_ring`, confirming near-ring is
  the dominant part of the non-basin visual effect in the primary anchor.
- `whole_head` repairs less than `duplicate_basin` because the head mixes
  duplicate-basin support with near-ring opposition.

Far/background has a small directional L2 in the primary anchor and weak
`masked_to_control` repair. Its `control_to_masked` rank effect is nontrivial,
but probability damage is much smaller than near-ring/non-basin. This makes it
a secondary interaction, not the main opposition source.

## Mechanistic picture after this slice

The current best-supported mechanism is:

1. Layer-17 head 1 routes strongly from the duplicate-basin visual token at the
   primary onset coordinate slot.
2. The duplicate-basin value contribution points toward the duplicated
   coordinate basin before `o_proj`.
3. The near visual neighborhood around that duplicate basin points against the
   duplicated coordinate direction.
4. `o_proj` preserves the sign structure and maps it into residual space.
5. The self-attention output is therefore a visual local-competition mixture:
   duplicate-basin support minus near-ring opposition.
6. The downstream residual stream carries the resulting coordinate-basin state
   into layer 18 with mild attenuation.

This is a more specific origin story than "attention copies the object." The
mechanism is a local visual competition inside one attention head, with a
coordinate-slot basin downstream that can amplify or preserve the winning
direction.

## Next deterministic question

The next question is whether this local competition is driven more by:

1. attention routing changes: query/key mass moves between duplicate-basin and
   near-ring visual keys;
2. value content: near-ring value vectors themselves point against the coordinate
   basin;
3. downstream coordinate-slot attraction: small source-vector differences are
   amplified by layer-17 output/residual geometry.

The clean next probe is a paired route/content decomposition for layer-17 head 1:

- freeze attention weights but swap value vectors for duplicate-basin vs
  near-ring buckets;
- or freeze values but patch attention mass/source weights into those buckets;
- then compare coord distribution fields at the same `post_y1/pre_x2` anchors.

This is the next place where new GPU work would be worthwhile.

## Verification

Commands used for this slice were read-only aggregation over existing artifacts:

```bash
find /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920 \
  -maxdepth 2 \( -name 'projected_direction_rows.jsonl' \
  -o -name 'phase4_projected_direction_summary.json' \
  -o -name '*projected_direction*report.md' \) -print
```

```bash
python - <<'PY'
# Aggregated visual-spatial direction and projected-patch rows.
PY
```

No model code was changed in this slice. No new GPU jobs were launched because
the directional and causal visual-spatial all-shard artifacts already existed
and directly answered the current deterministic question.
