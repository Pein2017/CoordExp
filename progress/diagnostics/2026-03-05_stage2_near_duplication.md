---
title: Stage-2 AB Near-Duplication Diagnosis
status: active-diagnostic
scope: stage2-channel-b
topics: [stage2, channel-b, duplication, diagnostics]
references:
  - docs/PROJECT_CONTEXT.md
  - progress/directions/stage2_clean_prefix_v2.md
---

# Stage-2 AB Near-Duplication: Diagnosis + EM/MLE-Aligned Directions (2026-03-05)

Date: 2026-03-05  
Last updated: 2026-03-06  
Note: referenced run artifacts may be pruned; paths are best-effort pointers.

This note records a concrete failure mode observed in a Stage-2 AB run: **near-duplicate object enumeration** (same `desc` with highly-overlapping boxes, repeated many times in the same rollout). It summarizes evidence from monitor dumps + scalar logs, and frames the failure mode in an **EM-ish / maximum-likelihood** perspective so we can explore *learning-based* fixes (not just decode-time policies).

Update scope for 2026-03-06:
- Re-read `logging.jsonl` through the latest available step in this run (`global_step=1280`).
- Added the new eval checkpoint at **step 1200**.
- Added the new eval monitor dump at **`monitor_dumps/step_001200.json`**.
- Main new finding: the run is **not** a simple monotone collapse after step 900; it enters a late **oscillatory / two-mode regime** where average eval can improve while a few catastrophic duplicate-heavy samples remain or worsen.

Related notes:
- Design intent / algorithm: `docs/training/STAGE2_DESIGN.md` and `progress/directions/stage2_clean_prefix_v2.md`
- Prior failure mode summary (B-hot + length growth): `progress/diagnostics/2026-02-17_stage2_b_ratio_085_instability.md`
- Visual duplication analysis patterns (crowded books/people): `progress/diagnostics/2026-02-25_stage2_channel_a_visual_audit.md`

---

## 0) Artifacts (Provenance Handles)

Run (train + eval + monitor dumps):
- Run dir: `output/stage2_ab/prod/ab_mixed/eff_size_96-b_ratio_0.75-n_softctx_iter_2-epoch_2/v0-20260304-150926/`
- Train/eval scalars: `.../logging.jsonl`
- Eval monitor dumps: `.../monitor_dumps/step_000300.json`, `.../monitor_dumps/step_000600.json`, `.../monitor_dumps/step_000900.json`, `.../monitor_dumps/step_001200.json`
- Config: `configs/stage2_two_channel/prod/ab_mixed.yaml` (extends `configs/stage2_two_channel/base.yaml`)
- Metadata: `.../run_metadata.json` (git SHA `854195f...`, seed `17`)

Key config knobs (resolved):
- `stage2_ab.schedule.b_ratio = 0.75`
- `stage2_ab.n_softctx_iter = 2`
- `rollout_matching.max_new_tokens = 2560`
- `rollout_matching.decode_mode = greedy`, `temperature=0.0`, `top_p=1.0`, `top_k=-1`
- `rollout_matching.fp_cost = 0.5`, `fn_cost = 2.0`, `maskiou_gate = 0.3`

---

## 1) AB Schedule Sanity (Why `logging.jsonl` Looks B-Only)

The Stage-2 AB scheduler uses a deterministic "Bresenham-style" schedule:

- Select channel B at step `s` iff `floor((s+1)*b_ratio) > floor(s*b_ratio)`.

For `b_ratio=0.75`, this yields a periodic pattern:

- `A, B, B, B, A, B, B, B, ...` (exactly 1/4 A and 3/4 B).

CPU-only confirmation script (safe, no GPU needed):
- `temp/check_stage2_ab_schedule.py`
- Output (abridged): `ABBBABBBABBB...` and `A=25, B=75` over first 100 steps.

Important logging caveat:
- `configs/stage2_two_channel/base.yaml` uses `training.logging_steps = 10` and `logging_first_step = true`.
- This samples `global_step = 1, 10, 20, 30, ...`.
- The scheduler is keyed on the executed step index `s = global_step - 1`, so those sampled points are `s = 0, 9, 19, 29, ...`.
- With `b_ratio=0.75`, those `s` values are almost always channel **B**. So `logging.jsonl` can *appear* B-only even when A is running 25% of steps.

The scalar `stage2_ab/b_ratio_realized` (computed from recent executed steps) is consistently ~`0.75` in this run, which is consistent with the intended schedule.

---

## 2) What "Near-Duplication" Looks Like (Concrete Monitor Dump Evidence)

We define **near-duplicate** (operationally) as:
- same `desc`, and
- IoU(`bbox_i`, `bbox_j`) >= 0.9, with `bbox` in `norm1000` space (sufficient for diagnosing the behavior).

### 2.1 Step 300: "baseball bat" enumeration loop (one-sample hotspot)

From `monitor_dumps/step_000300.json`:
- Sample `sample_id=87140591469043`
- Parsed predictions: `pred=65`, GT: `gt=15`
- Label spam: `baseball bat` repeated **62 times** in the same output.

Near-duplicate concentration:
- Among the 62 `baseball bat` boxes:
  - IoU>=0.9 pairs: **85**
  - IoU>=0.95 pairs: **32**
  - max IoU observed: **~0.988**

This is not exact copy-paste duplication; it is **near-duplicate jitter** (tiny coordinate changes) repeated many times.

Qualitative rollout excerpt pattern (head and tail):
- Rollout begins emitting `{"desc": "baseball bat", "bbox_2d": ...}` and keeps repeating.
- Many boxes drift toward extreme coordinates (e.g., `... <|coord_999|>`), consistent with "fill the budget / drift" behavior under long enumerations.

### 2.2 Step 600: "book" enumeration loop (hotspot moves)

From `monitor_dumps/step_000600.json`:
- Sample `sample_id=87140591468546`
- Parsed predictions: `pred=38`, GT: `gt=17`
- Label spam: `book` repeated **33 times**
- Near-duplicate pairs among those `book` boxes: **5** (IoU>=0.9)

Again, this is a "same-class repeated object list" behavior, not invalid JSON or invalid coord tokens.

### 2.3 Key observation: the degenerate behavior is bursty + sample-concentrated

At both steps 300 and 600, the near-duplicate pairs are highly concentrated in 1-2 "hard" scenes (crowded instances like books/bats).
This suggests the failure is not simply "overall model is unstable"; it is a **specific attractor** triggered by certain image/prompt regimes.

### 2.4 Step 1200: new best eval checkpoint still contains catastrophic outliers

The extended run adds `monitor_dumps/step_001200.json`, and the surprising result is:

- **Eval mAP improves to a new best** at step 1200: `eval_rollout/mAP = 0.39849211`
- Yet the monitor dump still contains extremely degenerate samples, including cases that are clearly not explained by ordinary under-annotation.

Top step-1200 outliers (10-sample monitor dump):

- `base_idx=238`, `sample_id=87140591468782`
  - `pred=106`, `gt=16`, `fp=99`, `fn=9`, `parse_truncated=True`
  - repeated label: `book` **101 times**
  - same-desc IoU>=0.9 pairs: **793**
- `base_idx=190`, `sample_id=87140591468734`
  - `pred=105`, `gt=27`, `fp=98`, `fn=20`, `parse_truncated=True`
  - repeated label: `chair` **68 times**
  - same-desc IoU>=0.9 pairs: **174**
- `base_idx=307`, `sample_id=87140591468851`
  - `pred=106`, `gt=13`, `fp=98`, `fn=5`, `parse_truncated=True`
  - repeated label: `person` **92 times**
  - same-desc IoU>=0.9 pairs: **7**

Interpretation:
- The `book=101` and `chair=68` cases are unmistakable same-class repetition loops.
- The `person=92` street-scene case reinforces the earlier conclusion that this is **not mostly a dense-scene annotation issue**; the model is capable of entering a chaotic object-enumeration mode even when the image does not plausibly support that count.

So the new best checkpoint is "better on average" but **not clean in the tail**.

---

## 3) Scalar Signature: A Phase Transition Around Step ~530

Training-time rollout scalars show an abrupt jump into an over-generation regime:

- Step 520 (train snapshot):
  - `precision~0.610`, `recall~0.667`, `F1~0.637`
  - `pred/gt~1.09`, `rollout_len_mean~220`, `parse_truncated_rate=0.0`
- Step 530 (train snapshot):
  - `precision~0.354`, `recall~0.759`, `F1~0.483`
  - `pred/gt~2.14`, `rollout_len_mean~364`, `parse_truncated_rate=0.0625`
- Step 630 (worst observed train snapshot):
  - `precision~0.246`, `recall~0.751`, `F1~0.371`
  - `pred/gt~3.05`, `rollout_len_mean~582`, `parse_truncated_rate~0.156`

Interpretation:
- Recall stays high/increases (more proposals cover more GT).
- Precision collapses (FP count explodes).
- F1 collapses because it is precision-dominated when FP overwhelms.

Importantly, localization quality on matched predictions stays relatively stable:
- `rollout/matched_maskiou_mean` is ~0.79 on average in this run.
- `coord_diag/B/expected_bin_mae` improves over time (coord distributions get sharper/more accurate).

So the bottleneck is **set/cardinality** (duplicate enumeration), not geometry calibration.

### 3.1 Extended log read (through step 1280): late training is oscillatory, not purely monotone

The original read through step 900 suggested a late over-generation regime. The extended log confirms that regime, but adds an important refinement:

- after step 900, the run does **not** simply diverge monotonically,
- instead it **oscillates** between a "moderately bad" mode and a "catastrophically verbose" mode.

Aggregate train-rollout windows:

- Steps **300-890** (60 logged train windows):
  - `pred_per_sample_mean = 16.07`
  - `fp_total_mean = 1011.42`
  - `fn_total_mean = 173.40`
  - `precision_mean = 0.3903`
  - `recall_mean = 0.7562`
  - `parse_truncated_rate_mean = 0.0712`
  - `gen_new_tokens_p90_mean = 1607.26`
- Steps **900-1280** (39 logged train windows):
  - `pred_per_sample_mean = 24.13`
  - `fp_total_mean = 1755.00`
  - `fn_total_mean = 140.36`
  - `precision_mean = 0.2457`
  - `recall_mean = 0.8000`
  - `parse_truncated_rate_mean = 0.1534`
  - `gen_new_tokens_p90_mean = 2526.24`

So the late regime is still clearly worse on cardinality / truncation. But within that regime, the model keeps switching between two behaviors.

Examples of the "catastrophically verbose" mode:

- Step `1000`: `pred_per_sample=30.21`, `fp_total=2243`, `parse_truncated_rate=0.208`, `gen_new_tokens_p90=2560`
- Step `1010`: `pred_per_sample=30.89`, `fp_total=2308`, `parse_truncated_rate=0.229`, `gen_new_tokens_p90=2560`
- Step `1160`: `pred_per_sample=29.02`, `fp_total=2201`, `parse_truncated_rate=0.208`, `gen_new_tokens_p90=2560`
- Step `1260`: `pred_per_sample=28.31`, `fp_total=2165`, `parse_truncated_rate=0.219`, `gen_new_tokens_p90=2560`

Examples of the "moderately bad but partially recovered" mode:

- Step `930`: `pred_per_sample=16.27`, `fp_total=1135`, `parse_truncated_rate=0.073`, `gen_new_tokens_p90=2253.4`
- Step `1130`: `pred_per_sample=17.40`, `fp_total=1144`, `parse_truncated_rate=0.073`, `gen_new_tokens_p90=1586.9`
- Step `1170`: `pred_per_sample=17.80`, `fp_total=1251`, `parse_truncated_rate=0.073`, `gen_new_tokens_p90=2522.9`
- Step `1270`: `pred_per_sample=20.48`, `fp_total=1398`, `parse_truncated_rate=0.115`, `gen_new_tokens_p90=2560`

This is better described as a **bimodal / metastable decode regime** than as a one-way drift.

The important invariant across both modes is that matched localization quality stays fairly stable:

- post-900 `matched_maskiou_mean` still lives in a narrow band around `~0.78-0.80`

So again, the instability is mainly in **whether the model decides to keep enumerating objects**, not in where the matched boxes land.

### 3.2 Eval can improve while train-time rollout diagnostics still look chaotic

The new eval checkpoint at step 1200 is the clearest example of this decoupling.

Eval checkpoints so far:

- Step `300`: `mAP=0.39153625`, `precision=0.58327`, `recall=0.67597`, `pred/gt=3945/3404 = 1.1589`
- Step `600`: `mAP=0.38329532`, `precision=0.74130`, `recall=0.64483`, `pred/gt=2961/3404 = 0.8699`
- Step `900`: `mAP=0.38624761`, `precision=0.65900`, `recall=0.66539`, `pred/gt=3437/3404 = 1.0097`
- Step `1200`: `mAP=0.39849211`, `precision=0.64038`, `recall=0.67274`, `pred/gt=3576/3404 = 1.0505`

Step 1200 is therefore a real average-case improvement over step 300:

- `mAP`: `0.39849` vs `0.39154`
- `precision`: `0.64038` vs `0.58327`
- `recall`: `0.67274` vs `0.67597` (almost unchanged)

But the train-side rollout windows around that same region still show heavy over-generation, and the step-1200 monitor dump still has extreme tail failures.

Interpretation:
- the run is **not globally broken on every sample**,
- average eval quality can still improve,
- but the pathological duplicate-enumeration attractor remains alive on a subset of scenes and is strong enough to produce catastrophic outliers.

This means `train` rollout metrics are useful as an **online canary** for entering the bad basin, but they are **not a calibrated proxy** for eval mAP.

### 3.3 Train vs Eval FP/FN Can Flip (And Look Negatively Correlated)

In this run, the *training* rollout FP/FN (measured on the current training micro-window) and the *eval* rollout FP/FN (measured on `val_sample_limit=504`) move in opposite directions between the first two eval checkpoints:

- Eval step **300**:
  - **train** `rollout/fp_total=309`, `rollout/fn_total=239`
  - **eval** `eval_rollout/fp_total=1644`, `eval_rollout/fn_total=1103`
- Eval step **600**:
  - **train** `rollout/fp_total=962`, `rollout/fn_total=138`
  - **eval** `eval_rollout/fp_total=766`, `eval_rollout/fn_total=1209`

This can look "negatively correlated", but it is not necessarily a contradiction, because:

1. **They are different distributions.**
   - Train metrics are computed on a single rollout batch (`stage2/raw_rollouts=96`) at that training step.
   - Eval metrics are computed on a fixed validation slice (504 samples here), and are therefore much lower variance.

2. **Both are largely driven by a cardinality/verbosity shift (predicted count), not just geometry quality.**
   - At train step 300: `rollout/pred_per_sample ~ 8.17`
   - At train step 600: `rollout/pred_per_sample ~ 15.17`
   - Meanwhile eval predicted objects moved the *other* direction:
     - step 300: `eval_rollout/pred_objects=3945` (~7.83 preds/sample)
     - step 600: `eval_rollout/pred_objects=2961` (~5.88 preds/sample)

So train moving toward "over-generate to kill FN" (FP up, FN down) while eval moving toward "under-generate / more conservative" (FP down, FN up) is consistent with the numbers.

Practical takeaway:
- Treat *train* FP/FN primarily as an online diagnostic for **length/cardinality collapse** (over-generation regime),
  and treat *eval* FP/FN + mAP as the generalization signal.
- With only two eval points, do not over-interpret the sign; add more eval points (or a fixed train-eval slice) if we want to reason about correlations.

---

## 4) Mathematical Diagnosis (Why This Is a Stable Local Optimum)

### 4.1 Where the flat direction comes from (FP-neutral Channel-B)

Channel-B is designed to be **FP-neutral** (see `docs/training/STAGE2_DESIGN.md` and `progress/directions/stage2_clean_prefix_v2.md`):
- matched objects get geometry + (some) text supervision
- FN-injected objects get text supervision
- FP objects get *no* supervision inside their spans (to avoid punishing unlabeled true objects)
- closure/EOS is still supervised

Let `P` be the predicted set and `G` be the GT set.
The E-step does matching `M` (Hungarian / OT-ish) with stop-grad correspondences.
The M-step updates parameters using only matched + FN-injected targets.

If unmatched predictions in `P \\ M` incur ~0 loss (FP-neutral), then:
- Adding extra predictions can be (approximately) **loss-free**.
- Yet adding predictions can increase the probability that each GT object finds a match (reducing FN),
  which improves the parts of the objective that are actually supervised.

This creates an optimization pressure:
- increase |P| to reduce FN,
- tolerate FP because it is not penalized,
leading to a high-cardinality fixed point.

Near-duplication is the easiest way to do this without learning new semantics:
- keep emitting the same label,
- jitter coordinates slightly,
- occasionally one of these jittered boxes will match a GT instance under gating.

### 4.2 Why it looks EM-ish (latent variable interpretation)

We can interpret the matching assignment as a latent variable `z`:
- `z` chooses which predicted instance corresponds to each GT instance,
- with an "unassigned pool" for the rest.

Under FP-neutrality, the latent variables for "unassigned predictions" are effectively ignored.
This resembles EM with an unregularized number of mixture components: without a prior/penalty, the model can allocate extra components cheaply.

So the diagnosis is not "the model refuses to learn".
It is: **the current objective has an underconstrained degree of freedom: cardinality and self-collision.**

---

## 5) Diagnostics To Add (Make It Measurable)

These are cheap, run-local metrics that directly measure the failure mode:

Per-sample (rollout parsed preds):
- `dup/max_desc_count`: max over desc of count(desc) in the predicted set
- `dup/near_iou90_pairs_same_desc`: number of pairs (i<j) with same desc and IoU>=0.9
- `dup/near_iou90_pairs_any_desc`: same but no desc filter (less targeted)
- `dup/saturation_rate`: fraction of predicted boxes that contain `0` or `999` in any coord (boundary drift proxy)

Aggregates:
- mean/p90 of these metrics over the rollout batch
- correlate with `rollout/pred_per_sample`, `rollout/rollout_len_mean`, `rollout/parse_truncated_rate`

New recommendation from the extended log:
- track **tail-risk** metrics, not just means.
- The step-1200 checkpoint has the best eval mAP so far, yet still contains samples with `book=101`, `chair=68`, `person=92`.

So add at least:
- `dup/max_desc_count_max_over_batch`
- `dup/near_iou90_pairs_same_desc_max_over_batch`
- `dup/pred_objects_max_over_batch`
- `dup/parse_truncated_and_pred_gt_gt2_rate` (fraction of samples that are both truncated and >2x over GT count)

Acceptance check:
- these should drop sharply if we have a real fix (even if recall stays similar).

---

## 6) MLE/EM-Aligned Directions (No RL, No Hard Decode Policy)

The goal: preserve FP-neutrality for plausible unlabeled objects while penalizing *self-collision* behavior that is clearly redundant.

### 6.1 "Duplicate-of-matched" penalty (FP-neutral except for near-duplicates of matched)

Idea:
- Keep FP-neutral for generic FPs.
- But identify a subset of FPs that are almost certainly duplicates of already-matched predictions.

Procedure (Channel-B, after matching):
1. Let `M_p` be indices of matched predictions (pred side of Hungarian pairs).
2. For each unmatched pred `p`:
   - find `m = argmax_{m in M_p} IoU(bbox(p), bbox(m))` among *matched* preds with same `desc`
   - if IoU>=tau (e.g., 0.9), mark `p` as `dup_of_matched`
3. Apply a small penalty only to these `dup_of_matched` objects.

Note (empirical, this run only):
- In the step-300 monitor dump (10 eval samples), only **1** FP satisfied `(same desc, IoU>=0.9) w.r.t. a matched prediction`.
- In the step-600 monitor dump, this count was **0**.
So this rule targets a very specific duplicate subtype; it is still conceptually clean, but may not address the dominant "FP-FP near-dup jitter loops" by itself.

Why this is EM-ish / MLE-aligned:
- The "dup vs unlabeled-true" label is a latent variable.
- We infer it deterministically (E-step heuristic) from current outputs.
- We incorporate it as supervised signal in the M-step (regularized MLE / MAP).

Key property:
- It does NOT punish far-away extra objects (which could be unlabeled true instances).
- It targets exactly the harmful redundancy: repeated instances of the *same* thing.

Open question (implementation detail):
- What penalty best expresses "this object should not exist" under an autoregressive JSON format?
  Options: structure CE on the object's span, or an auxiliary "should-stop" loss, or a repulsion loss on geometry.

### 6.2 Repulsive set prior (self-collision regularizer)

Add a differentiable repulsion term between predicted objects of the same `desc`:

For predicted boxes `b_i` with labels `d_i`,
define a soft near-dup penalty:

`L_dup = lambda * sum_{i<j} 1[d_i = d_j] * softplus((IoU(b_i, b_j) - tau)/eps)`

where:
- `tau` is the near-dup threshold (e.g., 0.9),
- `eps` controls sharpness,
- IoU is computed on expectation-decoded `bbox` (CoordExp; differentiable almost everywhere).

This is equivalent to adding a "repulsive prior" on the set of boxes (point-process intuition).
It is not a hard constraint; the model can still output overlapping boxes when needed, but pays an energy cost.

### 6.3 Soft assignment / responsibility shaping (OT + entropy)

If we shift from hard assignment to soft responsibilities (Sinkhorn/OT),
we can add a term that prefers **peaked** responsibilities per GT, discouraging multiple similar preds sharing mass.

Conceptually:
- compute soft assignment matrix `A[p,g]` (rows preds, cols GT)
- add entropy penalty to encourage each GT to be "explained" by one pred

This is classical EM-style regularization ("avoid explaining one datapoint with many components").

Note:
- the current stack already has OT knobs (`ot_cost`, `ot_epsilon`, `ot_iters`), but as configured here it is very light (`ot_iters=1`).

---

## 7) Verification Plan (Minimal, Research-Grade)

When testing any near-dup fix, hold constant:
- seed, decoding settings, max_new_tokens, dataset slice

Verify:
1. `dup/near_iou90_pairs_same_desc` collapses (especially on the known hard scenes).
2. `rollout/pred_per_sample` stops growing in late training.
3. `rollout/parse_truncated_rate` does not trend upward with step.
4. Matched localization metrics (`matched_maskiou_mean`) remain stable (do not regress geometry).
5. Worst-sample tail metrics improve even when mean eval mAP is flat or slightly improved.

Qualitative:
- monitor dump rollouts should no longer show 20-60 repeats of the same class with jittered boxes.
- specifically, the known hard scenes should stop showing patterns like `book~101`, `chair~68`, `person~92` in a single sample.
