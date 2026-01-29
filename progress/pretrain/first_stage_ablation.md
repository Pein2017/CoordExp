# Stage-1 Pretraining Ablation (2026-01-26/27)

This note records the current Stage-1 ablation results for CoordExp, focusing on:

- whether the training/eval loss logging is correctly normalized under gradient accumulation (GAS),
- how the three loss variants behave on coordinate prediction diagnostics, and
- what these results imply for selecting Stage-2 initialization checkpoints.

Context / intended Stage-1 goal (see `progress/pretrain/first_stage.md` and `progress/full_idea.md`):

- Stage-1 is primarily meant to teach the model the *token protocol* (JSON-only structure + `<|coord_k|>` emission)
  and stabilize formatting.
- Stage-1 is *not* expected to fully solve high-precision localization; Stage-2 is expected to introduce
  rollout/matching/self-context forward and continuous geometric gradients.

---

## Experiments

All three experiments are Stage-1 runs on the same data setup (LVIS/COCO-like; no runtime resizing; `do_resize=false`)
with gradient accumulation enabled (GAS = 8 in logs: `accum/grad_steps: 8.0`).

### (A) softCE + W1 + gate (coord distribution supervision enabled)

Log:

- `output/1-26/stage_1/poly_prefer_semantic_max60-soft_ce_w1_gate/v0-20260126-162648/epoch_4-softce_w1_gate-LRs-2e-4_1e-4_4e-4-from-base-4B/logging.jsonl`

Key behavior:

- Base CE is computed on **non-coord** tokens only (coord targets masked from base CE).
- Coord tokens receive additional distribution losses:
  `coord_softce_w1/*` (softCE + W1 + gate).

### (B) pure hard CE baseline (no coord distribution supervision)

Log:

- `output/1-26/stage_1/poly_prefer_semantic_max60-pure_ce/v0-20260126-162638/epoch_4-pure_ce-LRs-2e-4_1e-4_4e-4-from-base-4B/logging.jsonl`

Key behavior:

- Standard SFT CE on all supervised tokens (including coord tokens).
- Coord diagnostics are still computed, but `coord_diag/enabled=0.0`.

### (C) mixed (hard CE + softCE/W1/gate for coord tokens)

Log:

- `output/1-26/stage_1/poly_prefer_semantic_max60-hard_ce_soft_ce_w1_gate/logging.jsonl`

Key behavior:

- Similar to (A) for `base_ce/*` + `coord_softce_w1/*`, but with a mixed setup.

---

## Implementation / Logging Audit (Gradient Accumulation)

### Summary

The prior issue was a **gradient-accumulation normalization mismatch** (train-only scaling error),
which could make `train.loss` appear approximately `1/GAS` of `eval_loss` for some modes.

After the fix (commit `dad60d6`), the new runs show consistent train/eval normalization:

- At every eval step, `train.loss / eval_loss ~ 1.0` (small noise expected because train logs are instantaneous,
  eval logs are dataset-averaged).
- For enabled modes, the sub-losses also match:
  - `train.base_ce/loss / eval_base_ce/loss ~ 1.0`
  - `train.coord_softce_w1/loss / eval_coord_softce_w1/loss ~ 1.0`

This strongly suggests the accumulation/scale bug is no longer present in these new logs.

### Quantitative check (ratios across all eval points)

Computed by aligning each `eval_*` row with the nearest preceding train row:

- softCE:
  - `loss/eval_loss`: median ~ 1.00, range ~ [0.98, 1.09]
  - `base_ce/loss / eval_base_ce/loss`: median ~ 0.99
  - `coord_softce_w1/loss / eval_coord_softce_w1/loss`: median ~ 1.00
- pure CE:
  - `loss/eval_loss`: median ~ 1.01, range ~ [0.99, 1.07]
- mixed:
  - `loss/eval_loss`: median ~ 1.00, range ~ [0.98, 1.05]
  - `base_ce` and `coord_softce_w1` ratios also ~ 1.0

Conclusion:

- The known scaling bug affected *training gradients* for the old pure-CE run (because the loss scaling was wrong).
- These restarted runs do not show signs of that bug anymore.

---

## Metrics Used (Cross-run comparable subset)

Important: `eval_loss` is **not directly comparable across runs** because enabled runs add extra coord losses
and also mask coord targets from base CE.

The following diagnostics are comparable across all three runs:

- `eval_coord_diag/expected_bin_mae`:
  mean absolute error in *bin index* between the expected-bin decode and the GT bin (coord positions only).
- `eval_coord_diag/expected_bin_abs_err_p90` (newly added):
  p90 of per-token `abs(expected_bin - gt_bin)` over coord positions.
  Intended to detect tail improvements that can be hidden by a mean-only metric.
- `eval_coord_diag/w1_to_delta` (newly added):
  W1 distance from the predicted coord distribution `p(k)` to a delta at the GT bin `t`,
  i.e. `E_p[|k - t|]` in **bins**. This is a Stage-2-aligned proxy for expectation/continuous geometry.
- `eval_coord_token_acc`:
  exact-match accuracy on coord tokens (hard hit rate).
- `eval_coord_diag/p_gt_mean`:
  average probability assigned to the GT bin at coord positions.
  High means a sharp distribution (often correlates with higher hard accuracy).

---

## Results

The three runs share 17 common eval steps: 40, 80, ..., 680 (soft/pure also include step 720).

### Snapshot at step 640 (all three runs have this eval point)

| run | expected_bin_mae | abs_err_p90 | coord_token_acc | p_gt_mean |
|---|---:|---:|---:|---:|
| softCE+W1+gate | 24.1203 | 55.4792 | 0.1714 | 0.0858 |
| pure CE | **23.6342** | **55.0611** | **0.2177** | **0.1665** |
| mixed | 24.2648 | 56.4770 | 0.2000 | 0.1036 |

Observations:

- Under these conditions, **pure CE is strongest on coord diagnostics** (lower mean + lower p90 + higher hard acc).
- softCE produces a much *flatter* distribution on GT (lower `p_gt_mean`), which reduces hard accuracy.
- mixed sits between pure CE and softCE on most metrics.

### Snapshot at step 720 (soft and pure only)

| run | expected_bin_mae | abs_err_p90 | coord_token_acc | p_gt_mean |
|---|---:|---:|---:|---:|
| softCE+W1+gate | 24.2960 | 57.2480 | 0.1786 | 0.0878 |
| pure CE | **23.9431** | **55.9131** | **0.2288** | **0.1685** |

### Curve shape / plateau

Across all runs, `expected_bin_mae` and p90 show the same qualitative regime change:

- Early training: very large errors (hundreds of bins) rapidly collapse by ~step 120.
- After ~step 120: a stable plateau emerges around:
  - `expected_bin_mae ~ 23.5 - 27 bins`
  - `p90 ~ 55 - 65 bins`

This plateau appears in all three runs, and differences between runs are relatively small compared
to the early collapse.

---

## Interpreting the "Floor" (bins -> pixels, patch scale intuition)

With 1000 bins (`0..999`) under the canonical mapping `c = k/999`, 1 bin step is ~1/999 in normalized coordinates.

For intuition, using a 640px image dimension (pixel coordinate uses `(W-1)*c`):

- 24 bins  ~= 24/999 * (640-1) ~= 15.4 px
- 55 bins  ~= 55/999 * (640-1) ~= 35.2 px

This aligns closely with a 32x32 patch representation:

- half-patch is ~16 px

Given the training setting here (no runtime scaling; LVIS raw resolutions), the observed plateau is consistent with
a **vision resolution / patchification floor**, rather than a metric implementation artifact.

---

## Stage-2 Checkpoint Recommendation (based on Stage-1 signals)

Stage-2 (per `progress/full_idea.md`) depends on rollout + parsing + matching, which often benefits from
sharp discrete token predictions (coord token exactness) to avoid parse/matching failures.

Recommendation:

1) Use **pure CE** as the primary Stage-2 initialization checkpoint:
   - best hard coord accuracy,
   - best mean/p90 on expected-bin diagnostics,
   - highest `p_gt_mean` (sharpness).

2) Also keep **softCE+W1+gate** as a secondary Stage-2 initialization checkpoint:
   - although it does not win Stage-1 `expected_bin_mae` or p90 here,
     it yields smoother coord distributions which may improve expectation-based continuous losses in Stage-2.
   - This is a Stage-2 hypothesis and should be validated by Stage-2 early signals (parse rate, matching quality,
     geometric loss descent), not Stage-1 metrics alone.

3) Treat **mixed** as optional:
   - Stage-1 metrics are generally between pure and softCE.
   - It is useful for completeness / paper ablations, but not strictly necessary to get Stage-2 moving quickly
     if compute is constrained.

---

## Notes / Caveats

- Absolute `eval_loss` is not comparable across the three runs because the loss compositions differ.
  Use `eval_coord_diag/*` and `eval_coord_token_acc` for cross-run comparisons.
- The apparent localization floor is likely tied to the visual representation scale (patch resolution),
  and may not be solvable by Stage-1 loss tweaking alone.
- The new p90 metric is meant to detect tail improvements; in this current ablation window,
  softCE does not show a clear advantage on p90 vs pure CE.
