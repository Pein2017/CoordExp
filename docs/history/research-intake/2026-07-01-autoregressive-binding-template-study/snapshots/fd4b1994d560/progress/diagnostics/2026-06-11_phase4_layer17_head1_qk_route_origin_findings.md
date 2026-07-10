# Phase 4 Layer-17 Head-1 Q/K Route-Origin Findings

Date: 2026-06-11

## Scope

This slice follows the route/content causal patch result. The previous causal
patch showed that duplicate-basin repair is largely available through
`route_delta_masked_values`, so this run asks where that route delta originates
in score space.

The probe captures post-q/k-norm, post-RoPE Q/K states from
`Qwen3VLTextAttention` and decomposes pre-softmax score changes by visual
source bucket. For each bucket it reports:

- full control score: `q_control * k_control`
- full masked score: `q_masked * k_masked`
- query-side hybrid: `q_control * k_masked`
- key-side hybrid: `q_control * k_control - q_control * k_masked`
- symmetric query/key hybrid terms with control keys or masked query

This is an origin-narrowing score readout, not yet a causal Q/K patch.

## Artifact Handles

- Combined rows:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_qk_route_origin_layer17_head1_visual_triplet_top4_allshards_rows.jsonl`
- Summary JSON:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_qk_route_origin_layer17_head1_visual_triplet_top4_allshards_summary.json`
- Markdown report:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_qk_route_origin_layer17_head1_visual_triplet_top4_allshards_report.md`
- Smoke:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_qk_route_origin_layer17_head1_visual_triplet_primary_smoke_shard-04-of-08`

Run scale:

- 8 shards
- 3,768 Q/K route-origin rows
- layer 17, head 1
- components:
  `duplicate_basin`, `visual_near_ring`, `visual_far_background`

## Primary Result

For `none_latest_ckpt32`, phase `post_y1/pre_x2`, the duplicate-basin attention
mass gain is paired with a large positive pre-softmax score gain. The hybrid
decomposition attributes that score gain mostly to the visual-key side under
the control query:

| component | n | attention mass delta | score mean delta | query delta with masked keys | key delta with control query | score logsumexp delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| duplicate_basin | 62 | 0.123165 | 1.860993 | -0.501067 | 2.362059 | 3.972339 |
| visual_near_ring | 62 | -0.096206 | 0.160204 | -0.287529 | 0.447733 | 0.630656 |
| visual_far_background | 62 | -0.023959 | -0.239746 | -0.212699 | -0.027047 | 0.077893 |

Interpretation:

- Duplicate-basin routing is not primarily explained by the coordinate-slot
  query becoming globally more compatible with masked visual keys. The
  `q_control * k_masked - q_masked * k_masked` term is negative on average.
- The positive duplicate-basin score shift appears mostly when the control
  query is paired with control duplicate-basin keys. That points toward
  duplicate-basin visual-key state changes as the dominant score-space origin.
- Near-ring loses attention mass even though its mean/logsumexp score terms are
  not uniformly negative. That suggests competition and normalization matter:
  duplicate-basin score/logsumexp gain can suppress near-ring mass even when
  near-ring raw scores do not simply collapse.

## Next Causal Step

The next high-leverage test is a key-side causal patch:

1. capture or recompute post-RoPE K states for duplicate-basin and near-ring
   visual source buckets;
2. patch masked duplicate-basin key states toward control at layer 17 head 1,
   ideally before the attention score matmul;
3. measure whether the patch restores duplicate-basin attention mass and then
   downstream coordinate-basin probability/radius mass.

If direct K-state patching is awkward in the HF attention path, a controlled
attention-score bias patch for the duplicate-basin bucket can serve as the
first causal proxy, followed by a lower-level K patch once the score target is
validated.

## Verification

- `python -m py_compile src/analysis/autoregressive_duplication_mechanism/phase4_qk_route_origin.py scripts/analysis/run_autoregressive_duplication_phase4_qk_route_origin_shard.py tests/analysis/autoregressive_duplication_mechanism/test_phase4_qk_route_origin.py`
- Direct importlib harness for
  `tests/analysis/autoregressive_duplication_mechanism/test_phase4_qk_route_origin.py`
- One-shard live GPU smoke on shard 04:
  `phase4_qk_route_origin_layer17_head1_visual_triplet_primary_smoke_shard-04-of-08`
- Full eight-GPU shard sweep:
  `phase4_qk_route_origin_layer17_head1_visual_triplet_top4_allshards_shard-00-of-08`
  through
  `phase4_qk_route_origin_layer17_head1_visual_triplet_top4_allshards_shard-07-of-08`

`python -m pytest tests/analysis/autoregressive_duplication_mechanism/test_phase4_qk_route_origin.py -q`
still reports the local `Pytest: No tests collected` wrapper issue, so the
direct importlib harness is the meaningful deterministic test for this slice.
