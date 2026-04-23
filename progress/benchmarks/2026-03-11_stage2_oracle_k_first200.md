---
title: Oracle-K on Stage-2 High-Res 1024 Runs (First-200 COCO Val)
status: completed-diagnostic
scope: oracle-k-eval
topics: [oracle-k, stochastic-decoding, stage2, channel-b, unlikelihood, truncation, diagnostics]
references:
  - docs/PROJECT_CONTEXT.md
  - docs/eval/WORKFLOW.md
  - progress/diagnostics/2026-03-09_stage2_ul_capture_highres1024.md
---

# Oracle-K on Stage-2 High-Res 1024 Runs (First-200 COCO Val)

Date: 2026-03-11  
Status: completed diagnostic on a fixed 200-sample subset

This note records the first Oracle-K comparison across three related ~1024px Stage-2 checkpoints on COCO val.

Important naming clarification for this note:

- `ab_mixed` refers to the checkpoint at
  `output/stage2_ab/prod/ab_mixed/eff_size_96-b_ratio_0.75-n_softctx_iter_2-epoch_2_merged-ckpt-2442`,
  but for interpretation here it should be treated as the **Channel-A-only reference point** without effective Channel-B benefit.
- `ul_res_1024` is the first `300` steps after the high-res Channel-B run with `b_ratio=0.75`.
  This run still carried the **pre-fix unlikelihood capture bug**, so many correct duplicate-UL penalties were skipped.
- `ul_res_1024-v2` is the retraining from `ul_res_1024` after the **UL bug fix**.
  It improved latent recoverability, but also developed very long rollouts and frequent truncation pressure.

For the two Channel-B-related checkpoints, the training-side background and failure-mode interpretation should be read together with:

- `progress/diagnostics/2026-03-09_stage2_ul_capture_highres1024.md`

---

## 1) Experiment Setup

Dataset slice:

- JSONL: `public_data/coco/rescale_32_1024_bbox_max60/val.first200.coord.jsonl`
- Source: deterministic first `200` samples from `val.coord.jsonl`

Oracle-K sweep:

- `K = 8`
- temperatures: `0.2`, `0.5`, `0.8`, `1.0`
- seeds per temperature: `101`, `102`

Metrics reported below focus on:

- standard deterministic baseline metrics,
- Oracle-K recall at IoU `0.50`,
- recoverable vs systematic baseline false negatives,
- temperature/seed contribution to recovery.

Unless otherwise stated, “recoverable” means “recovered at least once among the 8 stochastic runs.”

---

## 2) Standard Baseline Snapshot

Deterministic eval on the same 200-sample subset:

| Checkpoint | bbox AP | bbox AP50 | F1-ish full micro @0.50 | Recall loc @0.50 | Recall full @0.50 |
|---|---:|---:|---:|---:|---:|
| `ab_mixed` | 0.272 | 0.336 | 0.464 | 0.338 | 0.335 |
| `ul_res_1024` | 0.368 | 0.469 | 0.616 | 0.554 | 0.546 |
| `ul_res_1024-v2` | 0.259 | 0.345 | 0.631 | 0.582 | 0.568 |

Immediate read:

- `ul_res_1024` has the strongest deterministic COCO-style quality.
- `ul_res_1024-v2` loses substantial AP relative to `ul_res_1024`, but slightly improves deterministic F1-ish full-match recall.
- `ab_mixed` is substantially worse than both `ul_res` runs on the deterministic decode.

---

## 3) Oracle-K Summary at IoU 0.50

| Checkpoint | Oracle recall loc | Oracle recall full | Delta loc | Delta full | Recoverable FN loc | Recoverable FN full | Recover fraction loc | Recover fraction full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ab_mixed` | 0.584 | 0.578 | +0.247 | +0.242 | 358 | 352 | 0.115 | 0.113 |
| `ul_res_1024` | 0.722 | 0.710 | +0.168 | +0.163 | 247 | 240 | 0.130 | 0.123 |
| `ul_res_1024-v2` | 0.767 | 0.759 | +0.186 | +0.191 | 273 | 281 | 0.183 | 0.181 |

Key observations:

- All three checkpoints have large best-of-K gains. A single deterministic decode is clearly leaving recoverable objects on the table.
- `ul_res_1024-v2` has the strongest latent capacity among the two `ul_res` runs:
  it reaches the best Oracle-K recall on both location and full matching.
- `ab_mixed` has the largest absolute delta from deterministic to Oracle-K.
  This is not because it is strongest overall, but because its baseline is weakest and therefore leaves more recoverable mass unused.

---

## 4) Recoverable vs Systematic FN Split

At IoU `0.50`, the baseline-FN split is:

| Checkpoint | Baseline FN loc | Recoverable share loc | Systematic share loc | Baseline FN full | Recoverable share full | Systematic share full |
|---|---:|---:|---:|---:|---:|---:|
| `ab_mixed` | 956 | 0.374 | 0.626 | 960 | 0.367 | 0.633 |
| `ul_res_1024` | 644 | 0.384 | 0.616 | 655 | 0.366 | 0.634 |
| `ul_res_1024-v2` | 604 | 0.452 | 0.548 | 624 | 0.450 | 0.550 |

Interpretation:

- `ul_res_1024-v2` is the strongest checkpoint if the question is:
  “How much of the remaining error is decoding-recoverable rather than systematic?”
- Roughly `45%` of its full-match misses are recoverable under K-sampling.
- `ul_res_1024` and `ab_mixed` look similar in recoverable-vs-systematic split:
  a little over one-third of full misses are recoverable, and nearly two-thirds remain systematic.

This is the main training implication:

- `ul_res_1024-v2` looks best for contrastive / GRPO-style exploitation of latent successful continuations.
- `ab_mixed` has a large absolute Oracle-K gain, but a large systematic remainder too, so “better decoding” alone is not enough.

---

## 5) Temperature Contribution

The number of baseline full-match FNs recovered by at least one run at each temperature:

| Checkpoint | t=0.2 | t=0.5 | t=0.8 | t=1.0 |
|---|---:|---:|---:|---:|
| `ab_mixed` | 89 | 198 | 176 | 219 |
| `ul_res_1024` | 82 | 120 | 135 | 147 |
| `ul_res_1024-v2` | 147 | 154 | 182 | 168 |

The number of baseline full-match FNs recovered **only** at one temperature band:

| Checkpoint | t=0.2 only | t=0.5 only | t=0.8 only | t=1.0 only |
|---|---:|---:|---:|---:|
| `ab_mixed` | 14 | 26 | 45 | 54 |
| `ul_res_1024` | 11 | 21 | 31 | 39 |
| `ul_res_1024-v2` | 19 | 22 | 34 | 29 |

Takeaways:

- High-temperature sampling is genuinely useful.
- For both `ul_res` checkpoints, `0.8` and `1.0` dominate the recovery pool.
- For `ab_mixed`, `1.0` is strongest and `0.5` is also very important.
- `0.2` is never the dominant temperature, but it is also never useless:
  it still contributes a non-trivial number of unique recoveries.

So the practical conclusion is not “just use the highest temperature.”
It is:

- keep a **diverse** sweep,
- but bias toward `0.8/1.0` for the `ul_res` family,
- and toward `0.5/1.0` for `ab_mixed`.

---

## 6) Most Productive Individual Runs

Top full-match recovery runs:

- `ul_res_1024`
  - `t=0.8,s=101` -> 103 recovered full FNs
  - `t=1.0,s=101` -> 92
  - `t=1.0,s=102` -> 90
- `ul_res_1024-v2`
  - `t=0.8,s=101` -> 130
  - `t=0.8,s=102` -> 119
  - `t=1.0,s=101` -> 116
- `ab_mixed`
  - `t=1.0,s=101` -> 164
  - `t=0.5,s=101` -> 159
  - `t=1.0,s=102` -> 125

This reinforces the temperature-level story:

- `ul_res_1024-v2` is especially effective when pushed into the `0.8` regime.
- `ab_mixed` responds unusually well to `1.0` and also strongly to `0.5`.

---

## 7) Narrative Interpretation Across The Three Checkpoints

### 7.1 `ab_mixed` (Channel-A-only reference)

- Worst deterministic baseline.
- Largest absolute Oracle-K lift.
- Still leaves a large systematic remainder.

Interpretation:

- this checkpoint has substantial latent capacity that deterministic decoding fails to realize,
- but the residual error profile is still not “mostly fixable by decoding”.

### 7.2 `ul_res_1024` (first Channel-B continuation with pre-fix UL capture bug)

- Best deterministic AP.
- Clear Oracle-K gain, but weaker latent recoverability than `ul_res_1024-v2`.

Interpretation:

- despite the UL-capture bug, the model lands on the best deterministic quality here,
- but the bug plausibly limits how well Channel-B pressure translated into robust latent alternative rollouts.

### 7.3 `ul_res_1024-v2` (UL bug fixed, but rollout length exploded)

- Worse deterministic AP than `ul_res_1024`.
- Best Oracle-K recall and best recoverable/systematic split.

Interpretation:

- the fixed UL training seems to have improved the model’s latent ability to produce correct objects under sampling,
- but that gain was offset in deterministic eval by rollout-length explosion / truncation pressure.

In other words:

- `ul_res_1024-v2` looks better as a **latent recoverability** model,
- `ul_res_1024` looks better as a **single-decode conservative** model.

That is exactly the kind of divergence Oracle-K was meant to expose.

---

## 8) Practical Training Implications

If the goal is contrastive or GRPO-style follow-up:

- `ul_res_1024-v2` is the best candidate.
  It has the highest fraction of recoverable misses and the strongest Oracle-K recall.
- `ab_mixed` is also interesting because its deterministic decode is especially conservative.
  It provides many recoverable misses, but also many systematic ones.

Concrete implication:

- for object-level contrast, prioritize baseline full-match FNs that are recovered at least once in
  `output/oracle_k/analysis_stage2_ab_prod_first200_20260310/ul_res_1024_v2_ckpt300/fn_objects.jsonl`
- for “decoder conservatism” studies, `ab_mixed` is the most extreme case
  because its best-of-K lift is huge relative to its baseline.

---

## 9) Artifact Handles

Standard eval:

- `output/eval/stage2_ab_prod_pair_first200_20260310_r2/ul_res_1024_ckpt300/`
- `output/eval/stage2_ab_prod_pair_first200_20260310_r2/ul_res_1024_v2_ckpt300/`
- `output/eval/stage2_ab_abmixed_ckpt2442_first200_20260310/ab_mixed_ckpt2442_baseline/`

Oracle-K summaries:

- `output/oracle_k/analysis_stage2_ab_prod_first200_20260310/ul_res_1024_ckpt300/summary.json`
- `output/oracle_k/analysis_stage2_ab_prod_first200_20260310/ul_res_1024_v2_ckpt300/summary.json`
- `output/oracle_k/analysis_stage2_ab_abmixed_ckpt2442_first200_20260310/ab_mixed_ckpt2442/summary.json`

Per-object audit files:

- `output/oracle_k/analysis_stage2_ab_prod_first200_20260310/ul_res_1024_ckpt300/fn_objects.jsonl`
- `output/oracle_k/analysis_stage2_ab_prod_first200_20260310/ul_res_1024_v2_ckpt300/fn_objects.jsonl`
- `output/oracle_k/analysis_stage2_ab_abmixed_ckpt2442_first200_20260310/ab_mixed_ckpt2442/fn_objects.jsonl`

---

## 10) Bottom Line

Oracle-K did exactly what we wanted here:

- it separated conservative single-decode quality from latent multi-sample capacity,
- it showed that `ul_res_1024-v2` is the strongest latent recoverability model,
- and it showed that `ab_mixed` leaves the most performance unrealized under deterministic decoding.

The two most important conclusions are:

- `ul_res_1024-v2` is the best candidate for recoverability-aware training.
- `ab_mixed` is the clearest evidence that K-sampling can uncover a large hidden margin beyond conservative performance.
