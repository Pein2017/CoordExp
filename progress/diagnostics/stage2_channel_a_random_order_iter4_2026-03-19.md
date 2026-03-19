---
doc_id: progress.diagnostics.stage2-channel-a-random-order-iter4-2026-03-19
layer: progress
doc_type: diagnosis
status: historical-diagnostic
domain: research-history
summary: Partial-run diagnosis of the random-order Channel-A-only iter-4 artifact, focusing on loss dynamics, gradmon, and monitor-dump failure modes.
updated: 2026-03-19
---

# Stage-2 (Channel-A only) Diagnosis: Random-Order `n_softctx_iter=4` Partial Run (2026-03-19)

Date: 2026-03-19
Last updated: 2026-03-19
Note: this artifact was manually terminated before epoch completion; conclusions below are about the observed training dynamics up to termination, not about the hypothetical fully trained endpoint.

Run under analysis:
- Artifact root: `output/a_only/random_order-iter_4/epoch-1-eff_bs_128/v0-20260318-173914`
- Primary log: `output/a_only/random_order-iter_4/epoch-1-eff_bs_128/v0-20260318-173914/logging.jsonl`
- Monitor dumps:
  - `output/a_only/random_order-iter_4/epoch-1-eff_bs_128/v0-20260318-173914/monitor_dumps/step_000300.json`
  - `output/a_only/random_order-iter_4/epoch-1-eff_bs_128/v0-20260318-173914/monitor_dumps/step_000600.json`

Supporting comparison artifacts:
- Random-order `iter_1` baseline:
  - `output/a_only/random_order/epoch-1-iter_1-eff_bs_128/v0-20260318-094911/logging.jsonl`
  - `output/a_only/random_order/epoch-1-iter_1-eff_bs_128/v0-20260318-094911/monitor_dumps/step_000600.json`

Config reference:
- Source config snapshot: `output/a_only/random_order-iter_4/epoch-1-eff_bs_128/v0-20260318-173914/config_source.yaml`
- This is **not** a pure iter-count ablation against the `iter_1` run:
  - `n_softctx_iter` changes from `1` to `4`
  - `token_ce.config.struct_ce_weight` is reduced from `1.0` to `0.1`
  - `coord_reg.config.coord_ce_weight` is reduced from `0.04` to `0.02`
  - bbox loss weighting also shifts (`smoothl1_weight=2.0`, `ciou_weight=0.2`)

---

## 1) Executive read

The run does **not** show simple optimization divergence. Scalar losses fall substantially early, and the A2 loss bundle is strongly correlated with A2 coordinate quality. However, the run settles into a bad attractor:

- the final self-context pass (`A2`) remains consistently worse than the anchor pass (`A1`) on coord quality,
- eval remains far below the nearby `iter_1` random-order baseline at matched checkpoints,
- monitor dumps show repeated failures where predictions remain semantically plausible but lose **atomic objectness**, producing either too many valid boxes or boxes that cover a semantic region rather than an individual object.

Working interpretation:
- increasing soft self-context here is not producing pure noise,
- it is producing a *stable but wrong* geometry regime,
- the regime is best described as **semantic-area tracking without reliable instance separation**.

---

## 2) Artifact status at termination

Termination state from `logging.jsonl`:
- last logged progress: `79.78%`
- last logged elapsed time: `19h 37m 16s`
- last logged `train/samples_seen`: `9472`

This matters because:
- the run is incomplete,
- the fairest comparison is at shared checkpoint percentages (`32.79%`, `65.57%`) rather than the final row,
- even at those shared checkpoints, the iter-4 artifact already underperforms badly.

Runtime cost relative to the nearby `iter_1` baseline:
- `iter_1` mean `time/forward_s`: `6.78`
- `iter_4` mean `time/forward_s`: `27.49`
- `iter_1` mean `train_speed(iter/s)`: `0.0359`
- `iter_4` mean `train_speed(iter/s)`: `0.0106`

So the run is roughly 4x slower per forward and about 3.4x slower in iteration throughput, even before considering the worse object-level quality.

---

## 3) Loss dynamics from `logging.jsonl`

### 3.1 Early improvement is real

The scalar losses improve substantially from the start of training:

- total `loss`: `1.1606 -> 0.7831` best, then `0.7973` at termination
- `loss/A2_coord/coord_soft_ce`: `0.6348 -> 0.4361` best, then `0.4756`
- `loss/A2_coord/coord_token_ce`: `0.1258 -> 0.0858` best, then `0.0939`
- `loss/A2_coord/bbox_smoothl1`: `0.0564 -> 0.0193` best, then `0.0250`
- `loss/A2_coord/bbox_ciou`: `0.1841 -> 0.1306` best, then `0.1697`

This is not the signature of a run that immediately blows up numerically.

### 3.2 The run peaks mid-way and then degrades

The stronger story is:
- losses improve quickly,
- A2 coordinate quality improves with them,
- then both stop improving and partially regress before termination.

Key A2 coordinate diagnostics:
- `coord_diag/A2/acc_top5`: `0.2015 -> 0.3220` best, then `0.2460`
- `coord_diag/A2/expected_bin_mae`: `144.3 -> 75.2` best, then `92.6`

At representative train rows:
- first:
  - `A1 acc_top5=0.3381`, `A2 acc_top5=0.2015`
  - `A1 expected_bin_mae=74.4`, `A2 expected_bin_mae=144.3`
- middle:
  - `A1 acc_top5=0.3136`, `A2 acc_top5=0.2815`
  - `A1 expected_bin_mae=76.1`, `A2 expected_bin_mae=91.6`
- last:
  - `A1 acc_top5=0.2997`, `A2 acc_top5=0.2460`
  - `A1 expected_bin_mae=76.1`, `A2 expected_bin_mae=92.6`

Important point:
- A2 improves a lot from the initial state,
- but it never overtakes A1,
- and by termination it remains materially worse than A1.

Across the whole run:
- mean `A1 acc_top5`: `0.3277`
- mean `A2 acc_top5`: `0.2709`
- mean delta: `A2 - A1 = -0.0568`
- mean `A1 expected_bin_mae`: `71.9`
- mean `A2 expected_bin_mae`: `92.9`
- mean delta: `A2 - A1 = +21.0`

So the final self-context pass is not functioning as a superior refinement pass; it remains a degraded geometry context.

### 3.3 The losses are measuring a real geometry signal

The A2 losses are tightly correlated with A2 coordinate diagnostics:

- `loss/A2_coord/coord_soft_ce` vs `A2 expected_bin_mae`: `r=0.964`
- `loss/A2_coord/coord_token_ce` vs `A2 expected_bin_mae`: `r=0.964`
- `loss/A2_coord/bbox_smoothl1` vs `A2 expected_bin_mae`: `r=0.979`
- `loss/A2_coord/bbox_ciou` vs `A2 acc_top5`: `r=-0.852`

Interpretation:
- the scalar losses are not lying,
- lower A2 coord/bbox losses do correspond to better A2 coordinate behavior,
- the problem is that “better A2 coordinates” still do not translate into robust atomic detection behavior under this training setup.

---

## 4) Eval-level consequence

At shared eval checkpoints, the iter-4 run underperforms the nearby iter-1 run by a large margin.

At `32.79%`:
- `iter_1`: `mAP=0.4417`, `F1=0.6445`, `precision=0.6146`, `recall=0.6773`
- `iter_4`: `mAP=0.3170`, `F1=0.2751`, `precision=0.2259`, `recall=0.3519`

At `65.57%`:
- `iter_1`: `mAP=0.4466`, `F1=0.6486`, `precision=0.6317`, `recall=0.6664`
- `iter_4`: `mAP=0.3216`, `F1=0.3325`, `precision=0.3096`, `recall=0.3590`

The failure is **not** mainly parsing/format:
- `eval/parsing/sample_valid_pred_rate = 1.0` for iter-4 eval rows
- `eval/parsing/parse_truncated_rate` drops from `0.0215` to `0.0020`

The dominant issue is object-level matching:
- at `65.57%`, iter-4 has `pred_objects=4241`, `gt_objects_total=3657`
- but only `matched=1313`
- with `fp_total=2928` and `fn_total=2344`

This is consistent with “valid boxes, weak atomic instance grounding.”

---

## 5) What `gradmon` says

The loss-gradient monitor here is useful because it distinguishes “bad objective geometry” from “bad task attractor.”

Observed aggregate behavior:
- mean `gradmon/neg_cosine_pair_frac`: `0.240`
- mean `gradmon/neg_cos_to_total_frac`: `0.102`
- mean `gradmon/grad_norm_ratio_max_over_median`: `39.98`

Interpretation:
- pairwise term conflict exists, but it is not overwhelming,
- most terms remain positively aligned with the total gradient,
- the larger issue is **norm imbalance**, not catastrophic directional disagreement.

Dominant terms on monitor steps:
- `A2_coord/coord_soft_ce` is usually the largest gradient term,
- `A2_coord/bbox_ciou` is commonly second,
- occasional spikes come from `A2_coord/bbox_log_wh`.

Representative cosine-to-total behavior:
- `gradmon/cos_to_total/A2_coord/coord_soft_ce` mean: `0.816`
- `gradmon/cos_to_total/A2_coord/coord_token_ce` mean: `0.777`
- `gradmon/cos_to_total/A2_coord/bbox_ciou` mean: `0.485`

Interpretation:
- the A2 objective bundle is largely pulling in one direction,
- especially the coord-distribution losses,
- but it is pulling toward a geometry regime that does not preserve atomic objectness.

Practical note:
- mean `time/gradmon_s` is `28.92`, slightly above mean `time/forward_s` (`27.49`),
- so the monitor is very expensive in this setup,
- but for this diagnosis it was still valuable because it rules out “loss conflict” as the primary explanation.

---

## 6) Monitor-dump findings (`step_000300` -> `step_000600`)

### 6.1 Step 300: strong over-generation / collapse in crowded scenes

Aggregate sample-level stats from `step_000300.json`:
- average `pred/gt` ratio: `2.86`
- median `pred/gt` ratio: `1.60`
- max `pred/gt` ratio: `8.53`
- average matched objects: `2.5`
- average sample F1: `0.428`
- truncated samples: `3 / 10`

Worst samples:
- sample `87140591468898`: `gt=29`, `pred=128`, `matched=2`, `f1=0.025`
- sample `87140591468831`: `gt=21`, `pred=128`, `matched=3`, `f1=0.040`

This is not just “slightly too many boxes.” It is a severe crowded-scene failure mode.

### 6.2 Step 600: partial recovery, but the same crowded samples remain broken

Aggregate sample-level stats from `step_000600.json`:
- average `pred/gt` ratio: `1.67`
- median `pred/gt` ratio: `1.09`
- max `pred/gt` ratio: `6.10`
- average matched objects: `3.0`
- average sample F1: `0.480`
- truncated samples: `1 / 10`

So the run *does* recover somewhat:
- fewer truncated outputs,
- lower average over-generation,
- slightly better matching.

But the hard crowded samples remain poor:
- sample `87140591468831`: `gt=21`, `pred=128`, `matched=4`, `f1=0.054`
- sample `87140591468546`: `gt=17`, `pred=27`, `matched=2`, `f1=0.091`
- sample `87140591468591`: `gt=17`, `pred=22`, `matched=2`, `f1=0.103`

### 6.3 Fixed-sample evolution confirms “partial recovery, persistent non-atomicity”

Nine sample IDs are shared between `step_000300` and `step_000600`.

Examples of improvement:
- sample `87140591468602`: `matched 3 -> 6`, `f1 0.222 -> 0.480`
- sample `87140591468898`: `pred 128 -> 32`, `matched 2 -> 4`, `f1 0.025 -> 0.131`

Examples of persistent failure:
- sample `87140591468831`: `pred 128 -> 128`, `matched 3 -> 4`, `f1 0.040 -> 0.054`
- sample `87140591468591`: `pred 29 -> 22`, but `matched 2 -> 2`, `f1 0.087 -> 0.103`

Interpretation:
- the run learns to reduce some pathological over-generation,
- but on the hardest crowded samples it does **not** learn the missing atomic partition.

### 6.4 The worst sample is semantically right and atomically wrong

In the worst `step_000600` sample (`sample_id=87140591468831`):
- GT object counts:
  - `book: 13`
  - `chair: 3`
  - `potted plant: 3`
  - `bowl: 1`
  - `couch: 1`
- Predicted object counts:
  - `book: 125`
  - `potted plant: 1`
  - `couch: 1`
  - `chair: 1`

This is a useful signature:
- class semantics are mostly correct,
- but the model is not committing to individual instances,
- it keeps emitting many valid `book` boxes that do not match the GT partition.

That is exactly the “same semantic area, lost atomic object property” failure mode.

---

## 7) Comparison against the nearby `iter_1` baseline at the same sampled step

The pathology is not merely “crowded samples are hard for everyone.”

At `step_000600`, comparing the same sampled IDs:
- sample `87140591468561`
  - `iter_1`: `pred=10`, `matched=9`, `f1=0.750`
  - `iter_4`: `pred=15`, `matched=2`, `f1=0.138`
- sample `87140591468591`
  - `iter_1`: `pred=21`, `matched=12`, `f1=0.632`
  - `iter_4`: `pred=22`, `matched=2`, `f1=0.103`
- sample `87140591468602`
  - `iter_1`: `pred=11`, `matched=9`, `f1=0.783`
  - `iter_4`: `pred=13`, `matched=6`, `f1=0.480`

Summary over the 10 sampled images at step 600:
- `iter_1`: `1 / 10` samples with `f1 < 0.15`
- `iter_4`: `5 / 10` samples with `f1 < 0.15`

So the iter-4 run is not just slower or unfinished. It is qualitatively worse on several fixed samples that the iter-1 baseline already handles reasonably well.

---

## 8) Diagnosis

### 8.1 What is happening

The run appears to learn a geometry regime with the following properties:
- syntactically valid outputs,
- semantically plausible category choices,
- some early improvement in coarse coordinate calibration,
- persistent weakness in **instance-atomic** partitioning under self-context.

This is why the boxes look “flattened” or “divergent” in crowded scenes:
- the model is not blind to the semantic region,
- it is not choosing a stable one-box-per-object decomposition.

### 8.2 Why the scalar loss does not protect against this

The scalar loss mostly supervises A2 through:
- coord-distribution objectives,
- box geometry losses,
- a small struct-only stabilizer.

That is enough to improve coarse coordinate quality.
It is **not** enough, in this Channel-A-only setup, to force stable per-instance decomposition under iterative self-conditioning.

Put differently:
- the run is learning “where the semantic mass is,”
- not reliably “which atomic object each box should belong to.”

### 8.3 Why this does not disprove self-context as a research idea

This artifact does **not** show that self-context is inherently useless.
It shows that:
- higher-iteration soft self-context in Channel-A alone is risky,
- and can push the model toward a semantically correct but atomically wrong solution.

This is actually compatible with the intended Stage-2 design:
- Channel-A is the cheap self-context approximation,
- Channel-B is the path meant to restore true self-context and set/object correction.

The artifact is therefore better read as:
- **negative evidence for “just increase Channel-A soft iterations,”**
- not a general rejection of self-context as a research direction.

---

## 9) Recommended next actions

### 9.1 Controlled verification

Run a true iter-only ablation:
- same model init
- same loss weights
- same seed
- same data ordering
- only `n_softctx_iter in {1, 2, 4}`

Without that control, this artifact should be treated as strong warning evidence, not a clean causal proof.

### 9.2 Verification metrics to keep

For each eval checkpoint, record:
- `pred_objects / gt_objects_total`
- matched count
- `fp_total`
- `fn_total`
- count of fixed monitor samples with `f1 < 0.15`
- count of fixed monitor samples with `pred >= 64`

These metrics capture the “semantic-area without atomicity” failure more directly than aggregate loss alone.

### 9.3 Recommendation for research direction

Do not promote `n_softctx_iter=4` Channel-A-only as a default based on this artifact.

If self-context remains the research target, prefer:
- `n=2` as the safer Channel-A operating point,
- and test it together with an explicit object/set-corrective mechanism rather than relying on more soft iterations alone.

