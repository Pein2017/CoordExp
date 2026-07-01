# Phase 4 Layer-17 Head-1 Attention Score-Bias Patch Findings

Date: 2026-06-11

## Scope

This slice follows the Q/K route-origin readout. That readout showed that the
duplicate-basin routing advantage in `none_latest_ckpt32` is largely a
pre-softmax score/logsumexp gap. This run tests a causal proxy: add the
duplicate-basin bucket score-logsumexp gap directly to the selected attention
scores before softmax.

This is not yet a low-level K-state patch. It is a controlled score-space
intervention: if a score bias alone restores attention but not the full
coordinate-basin behavior, then Q/K score routing is only part of the mechanism.

## Artifact Handles

- Combined rows:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_attention_score_bias_patch_layer17_head1_duplicate_top4_allshards_rows.jsonl`
- Summary JSON:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_attention_score_bias_patch_layer17_head1_duplicate_top4_allshards_summary.json`
- Markdown report:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_attention_score_bias_patch_layer17_head1_duplicate_top4_allshards_report.md`
- Smoke:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_attention_score_bias_patch_layer17_head1_duplicate_primary_smoke_shard-04-of-08`

Run scale:

- 8 shards
- 1,472 score-bias patch rows
- layer 17, head 1
- component: `duplicate_basin`
- directions: `masked_to_control`, `control_to_masked`

Rows with empty/invalid duplicate-basin score buckets are skipped, so row counts
are lower than the pure readout artifacts.

## Primary Result

For `none_latest_ckpt32`, phase `post_y1/pre_x2`:

| direction | n | score bias | attention recovery | prob recovery | radius4 recovery | radius8 recovery | expected abs-error recovery |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| masked_to_control | 26 | 3.972339 | 0.303442 | 0.004768 | 0.032295 | 0.057096 | 1.580239 |
| control_to_masked | 26 | -3.972339 | -0.075007 | 0.006566 | 0.046194 | 0.084109 | -12.068628 |

Checkpoint comparison for `post_y1/pre_x2`, `masked_to_control`:

| checkpoint | n | score bias | attention recovery | prob recovery | radius4 recovery | expected abs-error recovery |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| aligner_parent_ckpt1824 | 22 | 1.305042 | 0.035844 | 0.000636 | 0.004346 | -2.897174 |
| aux_latest_ckpt32 | 69 | 1.761452 | 0.102121 | 0.000275 | 0.002503 | -1.378155 |
| no_aligner_parent_ckpt3668 | 67 | 0.929683 | 0.042550 | 0.000264 | 0.002988 | -0.569427 |
| none_latest_ckpt32 | 26 | 3.972339 | 0.303442 | 0.004768 | 0.032295 | 1.580239 |

## Interpretation

The score-bias patch causally restores duplicate-basin attention mass in the
primary loss-only none checkpoint. The effect is much larger there than in the
SFT parents or `aux_latest_ckpt32`, consistent with the Q/K route-origin result
that `none_latest_ckpt32` has the strongest duplicate-basin score gap.

However, restoring duplicate-basin attention mass by a scalar bucket bias does
not fully reproduce the earlier route-content causal repair:

- local coordinate probability and radius mass improve;
- expected absolute error is mixed in the primary average;
- the intervention changes only bucket score mass, not the detailed within-
  bucket score distribution, value content, or visual-key representation.

This narrows the mechanism:

1. Q/K score routing is a causal contributor to duplicate-basin attraction.
2. Scalar bucket mass is not the whole mechanism behind the final coordinate
   basin. The within-bucket key structure and associated value vectors still
   matter.
3. The next deeper path should patch visual key states or score patterns at
   source-token granularity, rather than only applying a uniform bucket bias.

## Next Step

Move from scalar score-bias to key-pattern causality:

- patch duplicate-basin post-RoPE K states from masked toward control for
  layer-17 head-1 visual source tokens;
- or, as a near-term proxy, patch per-source score deltas instead of one scalar
  bucket bias;
- measure attention mass, coord probability/radius, expected absolute error,
  and route-content projected contribution together.

## Verification

- `python -m py_compile src/analysis/autoregressive_duplication_mechanism/phase4_attention_score_bias_patch.py scripts/analysis/run_autoregressive_duplication_phase4_attention_score_bias_patch_shard.py tests/analysis/autoregressive_duplication_mechanism/test_phase4_attention_score_bias_patch.py`
- Direct importlib harness for
  `tests/analysis/autoregressive_duplication_mechanism/test_phase4_attention_score_bias_patch.py`
- One-shard live GPU smoke on shard 04:
  `phase4_attention_score_bias_patch_layer17_head1_duplicate_primary_smoke_shard-04-of-08`
- Full eight-GPU shard sweep:
  `phase4_attention_score_bias_patch_layer17_head1_duplicate_top4_allshards_shard-00-of-08`
  through
  `phase4_attention_score_bias_patch_layer17_head1_duplicate_top4_allshards_shard-07-of-08`

`python -m pytest tests/analysis/autoregressive_duplication_mechanism/test_phase4_attention_score_bias_patch.py -q`
still reports the local `Pytest: No tests collected` wrapper issue, so the
direct importlib harness is the meaningful deterministic test for this slice.
