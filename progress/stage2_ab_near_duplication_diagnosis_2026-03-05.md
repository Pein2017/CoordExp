# Stage-2 AB Near-Duplication: Diagnosis + EM/MLE-Aligned Directions (2026-03-05)

Date: 2026-03-05  
Last updated: 2026-03-05  
Note: referenced run artifacts may be pruned; paths are best-effort pointers.

This note records a concrete failure mode observed in a Stage-2 AB run: **near-duplicate object enumeration** (same `desc` with highly-overlapping boxes, repeated many times in the same rollout). It summarizes evidence from monitor dumps + scalar logs, and frames the failure mode in an **EM-ish / maximum-likelihood** perspective so we can explore *learning-based* fixes (not just decode-time policies).

Related notes:
- Design intent / algorithm: `progress/full_idea.md`
- Prior failure mode summary (B-hot + length growth): `progress/stage_2_symptom.md`
- Visual duplication analysis patterns (crowded books/people): `progress/stage2_channel_a_only_coord_loss_visual_audit_2026-02-25.md`

---

## 0) Artifacts (Provenance Handles)

Run (train + eval + monitor dumps):
- Run dir: `output/stage2_ab/prod/ab_mixed/eff_size_96-b_ratio_0.75-n_softctx_iter_2-epoch_2/v0-20260304-150926/`
- Train/eval scalars: `.../logging.jsonl`
- Eval monitor dumps: `.../monitor_dumps/step_000300.json`, `.../monitor_dumps/step_000600.json`
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

---

## 4) Mathematical Diagnosis (Why This Is a Stable Local Optimum)

### 4.1 Where the flat direction comes from (FP-neutral Channel-B)

Channel-B is designed to be **FP-neutral** (see `progress/full_idea.md`):
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

Qualitative:
- monitor dump rollouts should no longer show 20-60 repeats of the same class with jittered boxes.

