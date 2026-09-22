# FN Book Attention Edge Visual Routing

Date: 2026-06-12

## Scope

This note follows the hard book y2/extent failure:

```text
image_id = 139
gt_idx = 17
desc = book
target = [944, 716/717, 967, 826]
```

The prior coordslot probe showed that both the no-aligner parent and aux
checkpoint fail even after forcing `desc,x1,y1,x2`: the model predicts a short
box with `y2 ~= 730..739` instead of `826`. This probe asks whether the target
book is visually attended at the relevant coordinate-query states.

## Configs

Added reproducible one-case attention configs:

```text
configs/analysis/autoregressive_duplication_mechanism/book17_teacher_forced_attention_no_aligner.yaml
configs/analysis/autoregressive_duplication_mechanism/book17_teacher_forced_attention_no_aligner_lane_c.yaml
configs/analysis/autoregressive_duplication_mechanism/book17_teacher_forced_attention_aux.yaml
configs/analysis/autoregressive_duplication_mechanism/book17_teacher_forced_attention_aux_lane_c.yaml
```

The target case is the random-order teacher-forced row:

```text
case_id = row0:teacher_forced:depth10:gt17
source_line_idx = 0
prefix_depth = 10
```

The aux configs use `checkpoint-32-inference-clean`. The raw `checkpoint-32`
adapter is rejected by the standard inference adapter resolver because it still
contains training-only `instance_enumeration_probe` tensors.

## Artifacts

No-aligner parent root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_attention_book17_y2_extent/no_aligner_parent_ckpt3668
```

Aux checkpoint root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_attention_book17_y2_extent/aux_latest_ckpt32
```

Shared reducer:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_attention_book17_y2_extent/book17_teacher_forced_attention_edge_reduction.json
```

Each checkpoint root has:

```text
shards/shard_000-of-001/selected_cases.jsonl
shards/shard_000-of-001/candidate_region_rows.jsonl
shards/shard_000-of-001/attention_region_rows.jsonl
shards/shard_000-of-001/decision_context_rows.jsonl
shards/shard_000-of-001/summary.json
```

Both atlas runs completed with:

```text
selected_cases = 1
candidate_region_rows = 7
attention_region_rows = 15680
decision_context_rows = 1
attention_layer_count = 28
attention heads per layer = 16
query_roles = desc_end, box_start, pre_x1, post_x1, post_y1
visual_grid = 26 x 39
```

## Visual Token Geometry

The key result is already visible in region-token membership:

| region | token count | visual tokens |
| --- | ---: | --- |
| `target_gt` | 2 | `[883, 922]` |
| `shared_book_overlap` | 2 | `[883, 922]` |
| `target_y2_edge_band` | 1 | `[922]` |
| `same_desc_gt` | 3 | `[883, 922, 961]` |
| `same_desc_beyond_target_y2` | 1 | `[961]` |
| `context_ring` | 28 | neighborhood around both books |

So the nominal target book is not visually separable from the adjacent
same-desc book at the coarse visual-token grid. The target region is exactly
the shared overlap tokens, while the neighboring book contributes one extra
lower token just beyond the target y2 boundary.

## Attention Readout

Max normalized attention mass for the most coordinate-relevant query roles:

| checkpoint | role | `target_gt` | `target_y2_edge_band` | `same_desc_beyond_target_y2` | `context_ring` |
| --- | --- | ---: | ---: | ---: | ---: |
| no-aligner parent | `post_x1` | 0.2654 | 0.0712 | 0.0342 | 0.9553 |
| no-aligner parent | `post_y1` | 0.1921 | 0.0508 | 0.0742 | 0.8199 |
| aux checkpoint | `post_x1` | 0.4894 | 0.1319 | 0.0466 | 0.9442 |
| aux checkpoint | `post_y1` | 0.4540 | 0.0748 | 0.0810 | 0.9384 |

Aux has much stronger high-head attention to the target/shared-book tokens than
the no-aligner parent. But this does not rescue y2: the earlier coordslot probe
still places the true `826` bin at rank `155..161` for aux, worse than the
no-aligner parent rank `130..131`.

The edge-specific readout is the crucial part. The single target lower-edge
token `[922]` remains much weaker than the broad context-ring maxima, and it is
one of the shared-overlap tokens rather than a clean bottom-boundary token. The
neighbor's extra lower token `[961]` receives comparable or larger attention at
`post_y1` than the target edge token in both checkpoints.

## Mechanism Read

This is not a simple "model cannot see the book" case. The model does attend to
the visual neighborhood and, especially in aux, to the target/shared-book visual
tokens. The harder mechanism is spatial disentanglement at the visual-token
resolution:

1. The target box and adjacent same-desc book collapse onto nearly the same
   visual-token support.
2. The one token that distinguishes the neighbor below the target boundary is
   not cleanly suppressed.
3. The target's own y2-edge token is weak relative to broad context heads.
4. Stronger target/shared-token attention in aux does not translate into a
   correct y2 coordinate basin, so "more target attention" is insufficient.

This updates the false-negative hypothesis:

- some FNs are language/prefix-state repairable;
- this book case is closer to a visual-token resolution and coordinate-basin
  disentanglement failure, where evidence is present but not separable enough
  to drive the `y2` slot.

## Next Hook

The promising next causal probe is not more language guidance. It is a
visual/source intervention around the three relevant visual tokens:

```text
shared target tokens = [883, 922]
neighbor lower token = [961]
target y2 edge token = [922]
```

Useful next tests:

1. Patch or mask visual-token states for `[922]` and `[961]` before the coord
   slot and measure y2 rank movement.
2. Compare this hard book case with an FN whose target visual region is
   token-separable from same-desc objects.
3. Add a reducer that reports overlap between target GT, same-desc GT, and
   edge-band regions for all hard no-rescue cases.

## Verification

Commands run:

```bash
python scripts/analysis/run_autoreg_attention_evidence_routing.py \
  --config configs/analysis/autoregressive_duplication_mechanism/book17_teacher_forced_attention_no_aligner.yaml \
  --stages attention_atlas \
  --shard-index 0 \
  --num-shards 1 \
  --dry-run

python scripts/analysis/run_autoreg_attention_evidence_routing.py \
  --config configs/analysis/autoregressive_duplication_mechanism/book17_teacher_forced_attention_aux.yaml \
  --stages attention_atlas \
  --shard-index 0 \
  --num-shards 1 \
  --dry-run

CUDA_VISIBLE_DEVICES=0 python scripts/analysis/run_autoreg_attention_evidence_routing.py \
  --config configs/analysis/autoregressive_duplication_mechanism/book17_teacher_forced_attention_no_aligner.yaml \
  --stages attention_atlas \
  --shard-index 0 \
  --num-shards 1

CUDA_VISIBLE_DEVICES=1 python scripts/analysis/run_autoreg_attention_evidence_routing.py \
  --config configs/analysis/autoregressive_duplication_mechanism/book17_teacher_forced_attention_aux.yaml \
  --stages attention_atlas \
  --shard-index 0 \
  --num-shards 1
```

The first aux launch against raw `checkpoint-32` failed before model forward
with the expected training-only adapter rejection. The committed config uses
`checkpoint-32-inference-clean`, and the rerun completed successfully.
