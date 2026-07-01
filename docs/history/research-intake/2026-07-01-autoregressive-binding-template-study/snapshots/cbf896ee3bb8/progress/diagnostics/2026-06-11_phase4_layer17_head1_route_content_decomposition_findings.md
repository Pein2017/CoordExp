# Phase 4 Layer-17 Head-1 Route/Content Decomposition Findings

Date: 2026-06-11

## Scope

This slice follows the layer-17 head-1 visual-competition finding by separating
source contribution deltas into routing and value-content terms.

The previous result was:

- `duplicate_basin` visual tokens support the duplicated coordinate direction.
- `visual_near_ring` tokens around the duplicate basin oppose it.
- `visual_far_background` is much weaker in the primary anchor.

The new question is whether that support/opposition split comes mainly from:

1. attention routing changes into the source bucket; or
2. value-content changes inside the bucket.

## Implementation

New module:

```text
src/analysis/autoregressive_duplication_mechanism/phase4_route_content.py
```

New runner:

```text
scripts/analysis/run_autoregressive_duplication_phase4_route_content_decomposition_shard.py
```

The probe reuses the same control and duplicate-basin-mask forwards as the
projected-direction probe. For each component and coordinate anchor, it emits:

- `total_delta`: `A_control V_control - A_masked V_masked`
- `route_delta_masked_values`: `A_control V_masked - A_masked V_masked`
- `value_delta_control_route`: `A_control V_control - A_control V_masked`
- `route_delta_control_values`: `A_control V_control - A_masked V_control`
- `value_delta_masked_route`: `A_masked V_control - A_masked V_masked`

Each effect is compared both pre-`o_proj` against the pulled-back coordinate
target direction and post-`o_proj` against residual-space coordinate and
residual-shift directions.

## Artifact

Root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920
```

Prefix:

```text
phase4_route_content_layer17_head1_visual_triplet_top4_allshards
```

Generated files:

```text
phase4_route_content_layer17_head1_visual_triplet_top4_allshards_summary.json
phase4_route_content_layer17_head1_visual_triplet_top4_allshards_report.md
phase4_route_content_layer17_head1_visual_triplet_top4_allshards_shard-*/route_content_decomposition_rows.jsonl
phase4_route_content_layer17_head1_visual_triplet_top4_allshards_shard-*/phase4_route_content_decomposition_summary.json
```

Contract:

- components: `duplicate_basin`, `visual_near_ring`, `visual_far_background`
- effect kinds: five listed above
- rows: 18,840
- shards: 8
- layer/head: 17/1

Shard counts:

| Shard | Rows | Checkpoints | Replay cases |
|---|---:|---:|---:|
| `shard-00-of-08` | 2,820 | 3 | 4 |
| `shard-01-of-08` | 2,520 | 1 | 4 |
| `shard-02-of-08` | 2,520 | 2 | 4 |
| `shard-03-of-08` | 2,760 | 3 | 4 |
| `shard-04-of-08` | 2,580 | 2 | 4 |
| `shard-05-of-08` | 2,220 | 2 | 4 |
| `shard-06-of-08` | 1,680 | 2 | 3 |
| `shard-07-of-08` | 1,740 | 2 | 3 |

## Primary anchor result

Primary slice:

- checkpoint: `none_latest_ckpt32`
- record: 33
- phase: `post_y1/pre_x2`
- rows per component/effect: 12

### Duplicate-basin support

| Effect | Pre target projection | Post target projection | Pre target cosine | Post target cosine | Post residual cosine | Pre L2 | Post L2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `total_delta` | 212.408108 | 16.122491 | 0.211286 | 0.034121 | 0.106566 | 43.063349 | 73.506805 |
| `route_delta_masked_values` | 223.011597 | 16.924735 | 0.217133 | 0.034981 | 0.080040 | 44.138324 | 75.468068 |
| `value_delta_control_route` | -10.603491 | -0.806637 | -0.034178 | -0.006219 | 0.089781 | 13.326585 | 20.208272 |
| `route_delta_control_values` | 212.541879 | 16.130617 | 0.212169 | 0.034243 | 0.106534 | 43.051376 | 73.490125 |
| `value_delta_masked_route` | -0.133770 | -0.010164 | -0.034178 | -0.006208 | 0.089754 | 0.168124 | 0.254911 |

Read:

- Duplicate-basin coordinate support is route-driven.
- Holding masked values fixed, swapping masked route to control route is enough
  to produce an even larger target projection than the total observed delta.
- Value-content differences are negative under the control route, so they
  slightly oppose or modulate the routing effect rather than explain the
  support.

### Near-ring opposition

| Effect | Pre target projection | Post target projection | Pre target cosine | Post target cosine | Post residual cosine | Pre L2 | Post L2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `total_delta` | -120.648186 | -9.158681 | -0.067495 | -0.010795 | -0.016858 | 38.159065 | 65.292662 |
| `route_delta_masked_values` | -117.141923 | -8.888016 | -0.051916 | -0.008349 | -0.031769 | 37.733683 | 64.624710 |
| `value_delta_control_route` | -3.506270 | -0.265794 | -0.065145 | -0.012771 | 0.205142 | 2.751581 | 3.828361 |
| `route_delta_control_values` | -96.613268 | -7.339042 | -0.036926 | -0.005997 | -0.060855 | 39.521307 | 67.206331 |
| `value_delta_masked_route` | -24.034925 | -1.823266 | -0.098121 | -0.019255 | 0.198534 | 10.135518 | 14.178677 |

Read:

- Near-ring opposition is also mostly route-driven.
- Value-content terms are nonzero and negative, especially under the masked
  route, but route swapping accounts for most of the primary negative target
  projection.
- This strongly favors a query/key-routing explanation for local visual
  opposition.

### Far/background

| Effect | Pre target projection | Post target projection | Pre target cosine | Post target cosine | Post residual cosine | Pre L2 | Post L2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `total_delta` | 1.958537 | 0.148796 | 0.041012 | 0.007910 | 0.277769 | 2.301309 | 3.631397 |
| `route_delta_masked_values` | 0.690312 | 0.052178 | 0.030702 | 0.006108 | 0.237312 | 2.194751 | 3.514672 |
| `value_delta_control_route` | 1.268225 | 0.096200 | 0.050007 | 0.009262 | 0.196838 | 0.829311 | 1.246072 |

Read:

- Far/background is small in the primary anchor.
- Its target projection is mildly positive, not the main opposition.
- Both route and value terms are secondary compared with duplicate-basin and
  near-ring effects.

## Cross-window result

Across all `post_y1/pre_x2` windows, `n=314` per component/effect:

| Component | Effect | Pre target projection | Post target projection | Pre target cosine | Post target cosine |
|---|---|---:|---:|---:|---:|
| `duplicate_basin` | `total_delta` | 27.645129 | 2.097917 | 0.074990 | 0.012710 |
| `duplicate_basin` | `route_delta_masked_values` | 34.232401 | 2.597846 | 0.057431 | 0.009785 |
| `duplicate_basin` | `value_delta_control_route` | -6.587272 | -0.500708 | 0.005999 | 0.001144 |
| `visual_near_ring` | `total_delta` | -30.645097 | -2.326307 | -0.060344 | -0.010169 |
| `visual_near_ring` | `route_delta_masked_values` | -26.409796 | -2.004995 | -0.064643 | -0.011002 |
| `visual_near_ring` | `value_delta_control_route` | -4.235301 | -0.321450 | -0.032876 | -0.005577 |
| `visual_far_background` | `total_delta` | 5.351415 | 0.406425 | 0.001039 | 0.000317 |
| `visual_far_background` | `route_delta_masked_values` | 9.243700 | 0.701358 | 0.016841 | 0.002896 |
| `visual_far_background` | `value_delta_control_route` | -3.892285 | -0.295206 | -0.013289 | -0.002119 |

Read:

- Duplicate-basin support remains route-dominated across windows.
- Near-ring opposition remains route-dominated across windows.
- Far/background is mixed and weaker, supporting its interpretation as a
  secondary modifier.

## Checkpoint contrast

Pre-`o_proj` target projection at `post_y1/pre_x2`:

| Checkpoint | Component | Total | Route with masked values | Value under control route |
|---|---|---:|---:|---:|
| `aligner_parent_ckpt1824` | `duplicate_basin` | 1.121756 | 0.829350 | 0.292406 |
| `aligner_parent_ckpt1824` | `visual_near_ring` | -3.530995 | -2.935361 | -0.595634 |
| `aligner_parent_ckpt1824` | `visual_far_background` | 65.488154 | 59.868799 | 5.619353 |
| `aux_latest_ckpt32` | `duplicate_basin` | 58.803849 | 60.560917 | -1.757069 |
| `aux_latest_ckpt32` | `visual_near_ring` | -10.175707 | -10.235705 | 0.059999 |
| `aux_latest_ckpt32` | `visual_far_background` | -13.604822 | -20.151347 | 6.546526 |
| `no_aligner_parent_ckpt3668` | `duplicate_basin` | 17.613940 | 12.689420 | 4.924521 |
| `no_aligner_parent_ckpt3668` | `visual_near_ring` | -24.224646 | -23.159512 | -1.065134 |
| `no_aligner_parent_ckpt3668` | `visual_far_background` | -4.980722 | 13.305598 | -18.286318 |
| `none_latest_ckpt32` | `duplicate_basin` | 30.032609 | 70.347305 | -40.314699 |
| `none_latest_ckpt32` | `visual_near_ring` | -94.453961 | -75.424325 | -19.029637 |
| `none_latest_ckpt32` | `visual_far_background` | -7.308979 | -7.115194 | -0.193783 |

Read:

- Route dominates the main duplicate-basin and near-ring signs in all
  checkpoint groups.
- Value content is not irrelevant: for `none_latest_ckpt32`, duplicate-basin
  value content strongly cancels the route support, while near-ring value
  content adds additional opposition.
- The deepest current mechanism is therefore not just "value vector points at a
  coordinate." It is primarily "the query/key route selects local visual buckets
  whose aggregate value contributions have opposing coordinate signs."

## Mechanistic update

The current best picture is:

1. Layer-17 head 1 develops a query/key routing split at coordinate slots.
2. Routing into the duplicate-basin visual token creates coordinate-basin
   support.
3. Routing into the local near-ring visual neighborhood creates coordinate-basin
   opposition.
4. Value content modulates this route-driven split and can partially cancel or
   amplify it by checkpoint.
5. `o_proj` preserves the route/content sign structure into residual space.
6. The residual stream carries the resulting coordinate-basin state into layer
   18.

The next deterministic probe should focus on query/key routing itself:

- attention score/logit decomposition for duplicate-basin vs near-ring keys;
- query vector direction at the coordinate slot;
- key vectors for duplicate-basin and near-ring visual tokens;
- causal patch of query/key scores or attention probabilities before value
  aggregation.

## Verification

Commands/checks:

```bash
python -m py_compile \
  src/analysis/autoregressive_duplication_mechanism/phase4_route_content.py \
  scripts/analysis/run_autoregressive_duplication_phase4_route_content_decomposition_shard.py \
  tests/analysis/autoregressive_duplication_mechanism/test_phase4_route_content.py
```

Direct importlib test harness:

```text
tests/analysis/autoregressive_duplication_mechanism/test_phase4_route_content.py ran 1
```

Pytest:

```text
python -m pytest tests/analysis/autoregressive_duplication_mechanism/test_phase4_route_content.py -q
Pytest: No tests collected
```

This matches the known local pytest collection issue in this worktree.

GPU/artifact verification:

- smoke shard: `phase4_route_content_layer17_head1_primary_components_smoke_shard-03-of-08`,
  2,760 rows.
- full all-shard run: 18,840 rows.
- all 8 GPUs idle after completion.
