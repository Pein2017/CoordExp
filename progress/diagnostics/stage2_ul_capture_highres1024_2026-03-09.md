---
title: Stage-2 AB Continuation @ ~1024px Before UL Boundary Fix
status: active-diagnostic
scope: stage2-channel-b
topics: [stage2, channel-b, duplicate-ul, diagnostics, resolution-1024]
references:
  - docs/PROJECT_CONTEXT.md
  - docs/training/STAGE2_DESIGN.md
  - progress/directions/stage2_clean_prefix_v2.md
  - progress/diagnostics/stage2_near_duplication_2026-03-05.md
---

# Stage-2 AB Continuation @ ~1024px: Pre-Fix UL Capture Gaps and Crowded-Scene Drop Risk (2026-03-09)

Date: 2026-03-09  
Status note: this document captures the run **before** the UL boundary-position fix was retrained into the model. It should be treated as the baseline to compare against the next post-fix continuation.

This note records a continuation run that combines:

- a **higher image-resolution cap** (`max_pixels = 32*32*1024 = 1048576`) with the COCO JSONL prepared under `rescale_32_1024_bbox_max60`,
- continuation from a previous Stage-2 checkpoint,
- and a known **pre-fix UL capture gap** where some duplicate continuations were identified but failed to become explicit duplicate-UL targets due to boundary/token alignment.

The main new diagnosis from this run is:

- average-case eval at step 300 is already strong,
- but the per-sample monitor dump shows that the remaining dataset-level risk is **not just obvious near-clone spam**,
- it is also **crowded same-class scenes** where healthy multiplicity can look superficially “duplicate-like”.

That is the exact failure mode where an overly aggressive duplicate cleaner / high `dup_drop_N` can hurt.

Related notes:

- prior duplicate-heavy baseline and tail-instability context: `progress/diagnostics/stage2_near_duplication_2026-03-05.md`
- current design intent: `docs/training/STAGE2_DESIGN.md`
- clean-prefix v2 direction: `progress/directions/stage2_clean_prefix_v2.md`

---

## 0) Artifacts

Run:

- Run dir: `output/stage2_ab/prod/ul-res_1024-continued-from-ckpt-2442/eff_size_96-b_ratio_0.75-n_softctx_iter_2-epoch_2/v0-20260309-011949/`
- Train / eval scalars: `.../logging.jsonl`
- Eval monitor dump used here: `.../monitor_dumps/step_000300.json`
- Config: `configs/stage2_two_channel/prod/ab_mixed_coco1024_bmajority.yaml`
- Launcher: `scripts/train_stage2.sh`

Resolution / dataset surfaces:

- `template.max_pixels = 1048576`
- `custom.train_jsonl = public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl`
- `custom.val_jsonl = public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl`

Continuation source:

- `model.model = output/stage2_ab/prod/ab_mixed/eff_size_96-b_ratio_0.75-n_softctx_iter_2-epoch_2_merged-ckpt-2442`

Important interpretation caveat:

- this run predates the later UL boundary-position fix,
- so any elevated `N_ul_skipped_no_divergence` here is part of the **baseline** rather than evidence against the fixed implementation.

---

## 1) Step-300 Scalar Read: Good Eval, Messy Train Windows

### 1.1 Train-side rollout at step 300 is still somewhat unstable

At the train snapshot for `global_step = 300`:

- `loss/B_rollout_text/duplicate_ul = 0.00011157`
- `stage2_ab/channel_b/dup/N_duplicates = 46`
- `stage2_ab/channel_b/dup/N_ul_boundaries = 2`
- `stage2_ab/channel_b/dup/N_ul_skipped_no_divergence = 44`
- `rollout/pred_per_sample = 13.34375`
- `rollout/precision = 0.42622951`
- `rollout/recall = 0.76470588`
- `rollout/f1 = 0.54736842`
- `rollout/parse_truncated_rate = 0.03125`

Interpretation:

- the train-side Channel-B windows still oscillate,
- duplicate handling is active but sparse,
- and a large fraction of duplicate-certified objects are still not becoming explicit UL targets in this pre-fix run.

### 1.2 Eval at step 300 is much healthier than the train-side canaries suggest

From the eval row at `global_step = 300`:

- `eval_rollout/mAP = 0.43986337`
- `eval_rollout/precision = 0.73588288`
- `eval_rollout/recall = 0.72356052`
- `eval_rollout/f1 = 0.72966968`
- `eval_rollout/pred_objects = 3347`
- `eval_rollout/gt_objects_total = 3404`
- `eval_rollout/parse_truncated_rate = 0.0`
- `eval_rollout/parse_dropped_invalid = 0.0`
- `eval_rollout/parse_dropped_ambiguous = 0.0`
- `eval_rollout/matched_maskiou_mean = 0.82540079`

Interpretation:

- average-case validation rollout is already good,
- the model is **not** over-generating on average (`pred/gt ≈ 0.983`),
- and the eval artifact is clean: no truncation, no invalid parse drops, no backend fallback issues.

### 1.3 This run is stronger than the earlier March-5 baseline at the first eval

For comparison, the prior duplicate-instability note recorded step-300 eval for the older run at:

- `mAP = 0.39153625`
- `precision = 0.58327`
- `recall = 0.67597`
- `pred/gt = 3945 / 3404 = 1.1589`

Current run at step 300:

- `mAP = 0.43986337`
- `precision = 0.73588288`
- `recall = 0.72356052`
- `pred/gt = 3347 / 3404 = 0.9833`

Important causal caution:

- this improvement is **confounded** by both checkpoint continuation and the ~1024px setup,
- so this note does **not** claim the gain is solely caused by higher resolution.

It does show that the combined setup is promising enough to preserve as a baseline.

---

## 2) Step-300 Monitor Dump: What The Real Per-Sample Risk Looks Like

The main monitor-dump discovery is that the residual pathology is **not** mostly “same-desc IoU>=0.9 clone spam”.

Across the 10 monitored eval samples:

- every sample has `parse_invalid_rollout = False`,
- every sample has `parse_dropped_invalid = 0`,
- every sample has `parse_dropped_ambiguous = 0`,
- every sample has `parse_truncated = False`,
- same-desc `IoU >= 0.9` duplicate pairs are `0` for every sample in this dump.

So the main dataset-level problem at this checkpoint is subtler:

- crowded same-class scenes,
- with many plausible same-class instances,
- plus a smaller number of true unhealthy over-enumeration cases.

### 2.1 Clearly unhealthy over-enumeration

`sample_id = 87140591468782`, `base_idx = 238`

- `pred = 38`, `gt = 16`, `matched = 10`, `fp = 28`, `fn = 6`
- dominant class: `book`
- predicted `book` count = `31`
- GT `book` count = `9`
- same-desc overlap is present but not mostly extreme (`near50_same = 6`, `near70_same = 2`, `near90_same = 0`)

Interpretation:

- this is a genuinely unhealthy “book over-enumeration” scene,
- and it is a good target for stronger duplicate suppression.

### 2.2 Healthy crowded same-class scenes that should **not** be over-dropped

These are the most important counterexamples.

`sample_id = 87140591468600`, `base_idx = 56`

- `pred = 5`, `gt = 5`, `matched = 5`, `fp = 0`, `fn = 0`
- dominant class: `sheep`
- predicted `sheep` count = `5`
- GT `sheep` count = `5`

`sample_id = 87140591469043`, `base_idx = 499`

- `pred = 16`, `gt = 15`, `matched = 10`, `fp = 6`, `fn = 5`
- dominant class: `baseball bat`
- predicted `baseball bat` count = `12`
- GT `baseball bat` count = `13`

`sample_id = 87140591468885`, `base_idx = 341`

- `pred = 18`, `gt = 15`, `matched = 10`, `fp = 8`, `fn = 5`
- dominant class: `person`
- predicted `person` count = `16`
- GT `person` count = `13`

`sample_id = 87140591468935`, `base_idx = 391`

- `pred = 15`, `gt = 11`, `matched = 8`, `fp = 7`, `fn = 3`
- dominant class: `bowl`
- predicted `bowl` count = `5`
- GT `bowl` count = `6`

Interpretation:

- high same-class multiplicity is often **correct** in this dataset,
- so a naive “many same-desc objects == unhealthy duplication” rule is wrong.

### 2.3 The most dangerous ambiguous case for `dup_drop_N`

`sample_id = 87140591468546`, `base_idx = 2`

- `pred = 19`, `gt = 17`, `matched = 5`, `fp = 14`, `fn = 12`
- dominant class: `book`
- predicted `book` count = `14`
- GT `book` count = `13`
- same-desc overlap is modest (`near50_same = 2`, `near70_same = 1`, `near90_same = 0`)

Interpretation:

- this is **not** a simple clone-spam case,
- the book multiplicity is actually close to GT,
- but matching quality is poor.

This is exactly the kind of scene where a too-high duplicate drop rate can make the dataset-level problem worse:

- many same-class candidates are legitimate,
- but localization / instance separation is messy,
- so over-dropping may remove healthy crowded-instance coverage rather than just deleting junk.

---

## 3) What This Means For The “Incorrect UL Capture” Story

The train log says this pre-fix run still had a large skip budget for duplicate UL construction. For example, at train step 300:

- `N_duplicates = 46`
- `N_ul_boundaries = 2`
- `N_ul_skipped_no_divergence = 44`

So the current run should be interpreted as:

- a baseline with **good average-case eval**,
- but still with incomplete UL coverage and residual crowded-scene instability.

The later UL boundary-position fix should, in theory, increase the fraction of duplicate-certified cases that become real UL negatives instead of skip-only accounting. But this monitor dump also makes clear that:

- not every high-multiplicity same-class scene should be pushed down,
- and success is **not** “make `max_desc_count` tiny everywhere”.

Success is:

- reduce unhealthy over-enumeration like `sample_id = 87140591468782`,
- while preserving healthy crowded-instance multiplicity like `sheep x5`, `baseball bat x12/13`, `person x16/13`, `bowl x5/6`.

---

## 4) Practical Baseline For The Next Retrain

Because the user planned to retrain after fixing a bug that made `dup_drop_N` too high, this run should be used as the **pre-fix comparison point**.

### 4.1 What should improve after the retrain

- fewer clearly unhealthy over-enumeration scenes,
- lower FP on samples like `87140591468782`,
- lower `N_ul_skipped_no_divergence` in train logs once the fixed code is actually in the retrained process.

### 4.2 What should **not** get worse

- healthy crowded same-class scenes should keep their multiplicity,
- samples like `87140591469043` and `87140591468885` should not be “fixed” by over-dropping valid same-class instances,
- `matched` should stay stable or improve on crowded scenes rather than collapsing with a lower `pred` count.

### 4.3 Recommended scorecard for the post-fix run

Check the next step-300 monitor dump for these sample archetypes:

- unhealthy over-enumeration:
  - `sample_id = 87140591468782`
- crowded same-class but plausibly healthy:
  - `sample_id = 87140591468546`
  - `sample_id = 87140591468885`
  - `sample_id = 87140591469043`
  - `sample_id = 87140591468935`

Desired direction:

- unhealthy sample:
  - `pred` down,
  - `fp` down substantially,
  - `matched` not harmed too much.
- healthy crowded samples:
  - `matched` stable or up,
  - `fn` stable or down,
  - dominant same-class counts still remain close to GT when the dataset truly contains many same-class instances.

---

## 5) Bottom Line

This ~1024px continuation run is a useful baseline because it shows two things at once:

1. the combined setup can already deliver a strong first eval checkpoint (`mAP 0.4399` at step 300), and
2. the remaining dataset-level flaw is not just obvious clone duplication; it is also the risk of **over-dropping healthy crowded same-class candidates**.

So the next retrain should be judged not only by “fewer duplicates”, but by the more precise question:

> did we reduce the unhealthy duplicate / over-enumeration cases **without** suppressing legitimate crowded same-class instance coverage?
