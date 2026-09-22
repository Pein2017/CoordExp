# Phase 4 Layer-17 Head-1 Attention Score-Pattern Patch Findings

Date: 2026-06-11

## Scope

This slice extends the scalar attention score-bias patch to a per-source score
pattern patch. Instead of adding one bucket-level logsumexp bias to every
duplicate-basin source token, the intervention adds the source-aligned
pre-softmax score delta:

```text
score_bias_i = score_control_i - score_masked_i
```

for each duplicate-basin visual source token. This tests whether the missing
piece after scalar score bias is the detailed within-bucket score pattern.

This is still a score-space intervention, not a direct K-state or value-state
patch.

## Artifact Handles

- Combined rows:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_attention_score_pattern_patch_layer17_head1_duplicate_top4_allshards_rows.jsonl`
- Summary JSON:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_attention_score_pattern_patch_layer17_head1_duplicate_top4_allshards_summary.json`
- Markdown report:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_attention_score_pattern_patch_layer17_head1_duplicate_top4_allshards_report.md`
- Smoke:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_attention_score_pattern_patch_layer17_head1_duplicate_primary_smoke_shard-04-of-08`

Run scale:

- 8 shards
- 1,472 per-source score-pattern patch rows
- layer 17, head 1
- component: `duplicate_basin`
- directions: `masked_to_control`, `control_to_masked`
- score bias mode: `per_source_delta`

## Primary Result

For `none_latest_ckpt32`, phase `post_y1/pre_x2`:

| mode | direction | n | score bias mean | score bias l2 | attention recovery | prob recovery | radius4 recovery | radius8 recovery | expected abs-error recovery |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| per_source_delta | masked_to_control | 26 | 1.860992 | 22.526543 | 0.304048 | 0.004902 | 0.032808 | 0.057736 | 8.639782 |
| per_source_delta | control_to_masked | 26 | -1.860992 | 22.526543 | -0.075530 | 0.006249 | 0.045468 | 0.083039 | -11.621593 |

Scalar score-bias baseline for the same slice:

| direction | n | score bias | attention recovery | prob recovery | radius4 recovery | expected abs-error recovery |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| masked_to_control | 26 | 3.972339 | 0.303442 | 0.004768 | 0.032295 | 1.580239 |
| control_to_masked | 26 | -3.972339 | -0.075007 | 0.006566 | 0.046194 | -12.068628 |

Checkpoint comparison for `post_y1/pre_x2`, `masked_to_control`:

| checkpoint | n | bias mean | bias l2 | attention recovery | prob recovery | radius4 recovery | expected abs-error recovery |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| aligner_parent_ckpt1824 | 22 | 1.336731 | 2.165712 | 0.036222 | 0.001143 | 0.007808 | -3.082536 |
| aux_latest_ckpt32 | 69 | 1.545436 | 9.863792 | 0.102233 | 0.000041 | 0.002310 | -1.179312 |
| no_aligner_parent_ckpt3668 | 67 | 0.110774 | 8.200753 | 0.042855 | 0.000271 | 0.002068 | -1.329218 |
| none_latest_ckpt32 | 26 | 1.860992 | 22.526543 | 0.304048 | 0.004902 | 0.032808 | 8.639782 |

## Interpretation

The per-source score pattern patch is a strong causal intervention on
duplicate-basin attention mass. It restores almost exactly the same attention
mass as the scalar bucket bias in the primary `none_latest_ckpt32` slice:

```text
scalar attention recovery:     0.303442
per-source attention recovery: 0.304048
```

It also gives similar coordinate probability and local radius recovery:

```text
scalar prob/radius4:     0.004768 / 0.032295
per-source prob/radius4: 0.004902 / 0.032808
```

But it does not restore the full coordinate-basin quality. Expected absolute
error is worse in the primary masked-to-control average, and rank/top1-distance
remain unstable. Therefore the missing mechanism is not just scalar bucket
mass, nor the source-token score pattern inside the duplicate basin.

This is a useful exclusion result:

1. Q/K score routing causally moves duplicate-basin attention.
2. Detailed score-pattern restoration is still insufficient to reproduce the
   earlier route-content/value-source coordinate repair.
3. The next high-leverage path is not another attention-score-only variant.
   It should patch the associated value contribution or patch visual key/value
   states together.

## Next Step

The next causal test should bind routing and content:

- patch duplicate-basin score pattern plus value contribution together;
- or patch post-RoPE K and V states for duplicate-basin visual tokens as a
  paired visual-state intervention;
- then compare against the earlier route-content projected patch, which already
  produced stronger coordinate-basin repair.

The central question becomes whether the duplicate-basin "attractor" is a
joint key-value visual state: key state opens the route, value state carries the
coordinate-basin direction.

## Verification

- `python -m py_compile src/analysis/autoregressive_duplication_mechanism/phase4_attention_score_bias_patch.py scripts/analysis/run_autoregressive_duplication_phase4_attention_score_bias_patch_shard.py tests/analysis/autoregressive_duplication_mechanism/test_phase4_attention_score_bias_patch.py`
- Direct importlib harness for
  `tests/analysis/autoregressive_duplication_mechanism/test_phase4_attention_score_bias_patch.py`
- One-shard live GPU smoke on shard 04:
  `phase4_attention_score_pattern_patch_layer17_head1_duplicate_primary_smoke_shard-04-of-08`
- Full eight-GPU shard sweep:
  `phase4_attention_score_pattern_patch_layer17_head1_duplicate_top4_allshards_shard-00-of-08`
  through
  `phase4_attention_score_pattern_patch_layer17_head1_duplicate_top4_allshards_shard-07-of-08`

`python -m pytest tests/analysis/autoregressive_duplication_mechanism/test_phase4_attention_score_bias_patch.py -q`
still reports the local `Pytest: No tests collected` wrapper issue, so the
direct importlib harness is the meaningful deterministic test for this slice.
