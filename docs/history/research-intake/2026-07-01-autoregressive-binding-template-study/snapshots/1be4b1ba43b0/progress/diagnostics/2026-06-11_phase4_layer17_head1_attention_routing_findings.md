# Phase 4 Layer-17 Head-1 Attention Routing Findings

Date: 2026-06-11

## Scope

This slice follows the route/content decomposition result. The previous result
showed that duplicate-basin support and near-ring opposition are primarily
route-driven. This probe measures the route directly from the model-returned
attention probabilities.

Important boundary:

- This is an attention routing mass/log-odds probe, not a raw pre-softmax q/k
  vector probe.
- Because the model applies positional machinery inside attention, using the
  returned attention probabilities is the faithful first readout of the route
  that actually feeds value aggregation.

## Implementation

New module:

```text
src/analysis/autoregressive_duplication_mechanism/phase4_attention_routing.py
```

New runner:

```text
scripts/analysis/run_autoregressive_duplication_phase4_attention_routing_shard.py
```

The probe emits, per component and coordinate anchor:

- control attention mass
- masked attention mass
- control-minus-masked mass delta
- control-minus-masked log-mass delta
- control-minus-masked logit-mass delta
- log mass relative to duplicate-basin mass
- visual attention share

Components used in the all-shard run:

```text
duplicate_basin,visual_near_ring,visual_far_background
```

## Artifact

Root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920
```

Prefix:

```text
phase4_attention_routing_layer17_head1_visual_triplet_top4_allshards
```

Generated files:

```text
phase4_attention_routing_layer17_head1_visual_triplet_top4_allshards_summary.json
phase4_attention_routing_layer17_head1_visual_triplet_top4_allshards_report.md
phase4_attention_routing_layer17_head1_visual_triplet_top4_allshards_shard-*/attention_routing_rows.jsonl
phase4_attention_routing_layer17_head1_visual_triplet_top4_allshards_shard-*/phase4_attention_routing_summary.json
```

Contract:

- rows: 3,768
- shards: 8
- layer/head: 17/1
- components: `duplicate_basin`, `visual_near_ring`, `visual_far_background`

Shard counts:

| Shard | Rows | Checkpoints | Replay cases |
|---|---:|---:|---:|
| `shard-00-of-08` | 564 | 3 | 4 |
| `shard-01-of-08` | 504 | 1 | 4 |
| `shard-02-of-08` | 504 | 2 | 4 |
| `shard-03-of-08` | 552 | 3 | 4 |
| `shard-04-of-08` | 516 | 2 | 4 |
| `shard-05-of-08` | 444 | 2 | 4 |
| `shard-06-of-08` | 336 | 2 | 3 |
| `shard-07-of-08` | 348 | 2 | 3 |

## Primary anchor

Primary slice:

- checkpoint: `none_latest_ckpt32`
- record: 33
- phase: `post_y1/pre_x2`
- rows per component: 12

| Component | Control mass | Masked mass | Mass delta | Log-mass delta | Logit-mass delta | Control visual share | Masked visual share |
|---|---:|---:|---:|---:|---:|---:|---:|
| `duplicate_basin` | 0.427491 | 0.005393 | 0.422098 | 3.839445 | 4.581482 | 0.446632 | 0.005678 |
| `visual_near_ring` | 0.176933 | 0.605071 | -0.428138 | -0.741123 | -2.166780 | 0.184900 | 0.628963 |
| `visual_far_background` | 0.348677 | 0.344718 | 0.003958 | 0.023095 | 0.026906 | 0.368467 | 0.365359 |

Read:

- Control routes strongly to the duplicate-basin visual token.
- Masking almost eliminates duplicate-basin attention mass.
- The missing mass moves mostly into the near-ring visual neighborhood.
- Far/background mass is nearly unchanged in the primary anchor.

This directly explains the route/content result:

```text
control: duplicate-basin route -> coordinate support
masked: near-ring route -> coordinate opposition
```

The primary route switch is not diffuse. It is a local exchange between the
duplicate visual token and its near-ring visual competitors.

## Primary row detail

For `post_y1/pre_x2`, the onset and post-onset rows show the starkest route
switch:

| Row | Offset | Duplicate control | Duplicate masked | Near-ring control | Near-ring masked |
|---:|---:|---:|---:|---:|---:|
| 23 | 0 | 0.593750 | 0.006744 | 0.219259 | 0.931176 |
| 25 | 2 | 0.593750 | 0.007477 | 0.266685 | 0.937607 |
| 26 | 3 | 0.765625 | 0.008118 | 0.142963 | 0.925138 |
| 27 | 4 | 0.384766 | 0.006409 | 0.547314 | 0.937842 |
| 28 | 5 | 0.781250 | 0.010071 | 0.155033 | 0.913291 |
| 29 | 6 | 0.691406 | 0.005493 | 0.238550 | 0.919373 |
| 30 | 7 | 0.464844 | 0.007996 | 0.465177 | 0.919651 |
| 31 | 8 | 0.851562 | 0.012329 | 0.068149 | 0.766460 |

Rows before the onset have tiny duplicate-basin mass in both conditions, so the
large route split is onset-local rather than a static property of the full
prefix.

## Cross-window result

Across all `post_y1/pre_x2` windows:

| Component | n | Control mass | Masked mass | Mass delta | Log-mass delta | Logit-mass delta | Control visual share | Masked visual share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `duplicate_basin` | 314 | 0.145066 | 0.086325 | 0.058741 | 0.648341 | 0.777140 | 0.157621 | 0.095937 |
| `visual_near_ring` | 314 | 0.125828 | 0.176017 | -0.050189 | -0.233796 | -0.351140 | 0.135735 | 0.190501 |
| `visual_far_background` | 314 | 0.654068 | 0.655785 | -0.001717 | -0.225679 | -0.348596 | 0.706644 | 0.713562 |

Read:

- The duplicate-basin/near-ring exchange persists across `post_y1/pre_x2`
  windows, but the primary anchor is much sharper than the mean.
- Far/background is high in absolute mass because it contains many tokens, but
  it does not explain the control-minus-masked route switch.

## Checkpoint contrast

At `post_y1/pre_x2`:

| Checkpoint | Component | Control mass | Masked mass | Mass delta | Log-mass delta |
|---|---|---:|---:|---:|---:|
| `aligner_parent_ckpt1824` | `duplicate_basin` | 0.081754 | 0.071011 | 0.010742 | 0.227205 |
| `aligner_parent_ckpt1824` | `visual_near_ring` | 0.030669 | 0.044916 | -0.014247 | -0.216484 |
| `aligner_parent_ckpt1824` | `visual_far_background` | 0.777780 | 0.750253 | 0.027527 | 0.071112 |
| `aux_latest_ckpt32` | `duplicate_basin` | 0.201315 | 0.113241 | 0.088074 | 1.073381 |
| `aux_latest_ckpt32` | `visual_near_ring` | 0.194516 | 0.264051 | -0.069535 | -0.396314 |
| `aux_latest_ckpt32` | `visual_far_background` | 0.549322 | 0.563538 | -0.014216 | -0.191592 |
| `no_aligner_parent_ckpt3668` | `duplicate_basin` | 0.126190 | 0.099185 | 0.027004 | 0.397174 |
| `no_aligner_parent_ckpt3668` | `visual_near_ring` | 0.158230 | 0.187752 | -0.029522 | -0.011889 |
| `no_aligner_parent_ckpt3668` | `visual_far_background` | 0.626558 | 0.622122 | 0.004436 | -0.225448 |
| `none_latest_ckpt32` | `duplicate_basin` | 0.165211 | 0.042047 | 0.123165 | 0.944785 |
| `none_latest_ckpt32` | `visual_near_ring` | 0.066054 | 0.162260 | -0.096206 | -0.442115 |
| `none_latest_ckpt32` | `visual_far_background` | 0.725324 | 0.749283 | -0.023959 | -0.548279 |

Read:

- All groups show the same sign pattern for duplicate-basin and near-ring mass
  deltas.
- `none_latest_ckpt32` has the sharpest duplicate-basin loss and near-ring gain.
- This matches its strong route/content decomposition and stronger visible
  duplication-basin mechanics.

## Mechanistic update

The current best mechanism is now:

1. At the coordinate slot, layer-17 head 1 normally routes to the duplicate-basin
   visual token.
2. That route supplies the coordinate-supporting value contribution.
3. When the duplicate-basin image region is masked, the same head reroutes
   locally into near-ring visual tokens.
4. Near-ring route/value aggregation supplies the coordinate-opposing component.
5. Far/background is large in token count and mass, but it is not the primary
   control-vs-mask route switch in the primary anchor.
6. The resulting self-attention output writes a coordinate-basin state into the
   residual stream, which layer 18 mostly carries forward.

This is the strongest current origin story:

```text
duplicate-basin visual evidence
  -> layer-17 head-1 local visual route selection
  -> support vs near-ring opposition value aggregation
  -> coord-slot basin state
  -> residual carry into layer 18
```

## Next deterministic question

The next deeper question is raw query/key geometry:

- what query vector at the coordinate slot selects duplicate-basin keys;
- how duplicate-basin keys differ from near-ring keys;
- whether the mask changes the query, the keys, or both;
- whether causal score/probability patching before value aggregation recreates
  the route switch.

That requires either a post-rotary q/k capture or a controlled attention-score
patch. This note intentionally stops at the returned attention route, which is
already enough to validate that route selection is the main immediate origin of
the support/opposition split.

## Verification

Commands/checks:

```bash
python -m py_compile \
  src/analysis/autoregressive_duplication_mechanism/phase4_attention_routing.py \
  scripts/analysis/run_autoregressive_duplication_phase4_attention_routing_shard.py \
  tests/analysis/autoregressive_duplication_mechanism/test_phase4_attention_routing.py
```

Direct importlib test harness:

```text
tests/analysis/autoregressive_duplication_mechanism/test_phase4_attention_routing.py ran 2
```

Smoke:

```text
phase4_attention_routing_layer17_head1_visual_triplet_primary_smoke_shard-04-of-08
attention_routing_row_count=516
```

Full all-shard artifact:

```text
phase4_attention_routing_layer17_head1_visual_triplet_top4_allshards
attention_routing_row_count=3768
```

All 8 GPUs were idle after completion.
