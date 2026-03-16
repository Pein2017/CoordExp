---
title: 2B COCO1024 Channel-A-Only Similarity, FN Hypotheses, and Prefix-Sensitivity Plan
status: active-diagnostic
scope: stage2-channel-a
topics: [stage2, 2b, channel-a, stage1-vs-stage2, fn-analysis, prefix, rollout-ordering]
references:
  - docs/PROJECT_CONTEXT.md
  - docs/SYSTEM_OVERVIEW.md
  - docs/data/CONTRACT.md
  - progress/diagnostics/stage2_ul_capture_highres1024_2026-03-09.md
  - progress/diagnostics/stage2_triage_posterior_coco1024_train_dynamics_2026-03-12.md
  - progress/diagnostics/stage2_channel_a_visual_audit_2026-02-25.md
  - progress/diagnostics/stage2_softctx_discretization_vs_stage1_bbox_2026-02-22.md
---

# 2B COCO1024 Channel-A-Only Similarity, FN Hypotheses, and Prefix-Sensitivity Plan (2026-03-16)

Date: 2026-03-16  
Status note: this note records a focused qualitative + subset-based comparison between the 2B Stage-1 checkpoint and the early high-res 2B Channel-A-only continuation, then proposes a concrete experiment plan for investigating prefix sensitivity and persistent FN behavior.

The short version is:

- on the monitored 10-image subset, the 2B Stage-1 checkpoint and the 2B A-only tuned model at recorded step `300` look **surprisingly similar**,
- both models already have strong semantic objectness,
- the main visible difference is **not** a broad capability jump, but a modest tendency for the A-only model to produce **more separable / more atomic crowded-scene objects** in a few hard cases,
- many strict-metric FPs in the dense scenes appear visually plausible and are likely amplified by incomplete GT annotation,
- the more important unresolved problem is therefore **FN / missed instance rollout behavior**, not raw FP count by itself,
- and the next high-value experiments should test whether the missing objects are due to:
  - rollout-prefix sensitivity,
  - sequence-length / EOS pressure,
  - or a stronger learned prior on the sorted training order than is actually helpful at eval time.

This note should be read together with:

- `progress/diagnostics/stage2_ul_capture_highres1024_2026-03-09.md`
- `progress/diagnostics/stage2_triage_posterior_coco1024_train_dynamics_2026-03-12.md`
- `progress/diagnostics/stage2_channel_a_visual_audit_2026-02-25.md`

---

## 0) Artifacts

2B A-only continuation run:

- Run dir:
  `output/stage2_ab/2b_1024/a_only_iter1/epoch_2-eff_size_64-n_softctx_iter_1-a_only/v0-20260309-102351/`
- Eval monitor dump used here:
  `.../monitor_dumps/step_000300.json`
- Train/eval scalars:
  `.../logging.jsonl`
- Resolved config:
  `.../resolved_config.json`
- Init model recorded in resolved config:
  `output/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate_merged-1332`

Fresh original-checkpoint parity rerun on the same 10 monitored images:

- Output dir:
  `temp/run_review/a_only_iter1_extreme_cases/original_ckpt_parity_step300_subset/`
- GT-vs-Pred JSONL:
  `.../gt_vs_pred.jsonl`
- Summary:
  `.../summary.json`
- Rendered overlays:
  `.../vis/`

Side-by-side comparison panels:

- Output dir:
  `temp/run_review/a_only_iter1_extreme_cases/original_vs_step300_pairs/`
- Per-sample summary:
  `.../summary.json`
- Contact sheet:
  `.../contact_sheet.png`

Important comparison caveat:

- the actual `checkpoint-300` weights are no longer present in the 2B A-only run dir,
- so the current comparison is:
  - a **fresh parity rerun** of the original Stage-1 checkpoint,
  - versus the **recorded** A-only step-`300` monitor dump,
- not a fresh rerun of the step-`300` weights.

This is still useful for qualitative evolution, but it is not a perfect same-backend checkpoint-vs-checkpoint rerun.

---

## 1) What The Current 2B Evidence Actually Supports

### 1.1 The two models are much closer than expected on the monitored subset

On the 10 monitored images, strict matching totals are:

| model | pred | matched | fp | fn |
|---|---:|---:|---:|---:|
| original Stage-1 2B parity rerun | `102` | `53` | `49` | `42` |
| recorded A-only step `300` dump | `104` | `56` | `48` | `39` |

So the A-only model is only modestly better on this subset under the strict metric:

- `+2` predictions,
- `+3` matched objects,
- `-1` FP,
- `-3` FN.

That is a real improvement, but it is **small**, not a qualitative jump.

### 1.2 Four of the ten monitored images are effectively unchanged

These samples are visually and numerically almost identical across the two models:

- `base000490`
- `base000045`
- `base000058`
- `base000318`

This matters because it suggests the A-only continuation is **not** radically rewriting the model’s general semantic behavior by step `300`.

### 1.3 The visible improvements are concentrated in a few hard cases

The main per-sample strict-metric gains are:

- `base000354`: `matched 7 -> 9`, `fp 32 -> 31`, `fn 22 -> 20`
- `base000047`: `matched 11 -> 12`, `fp 9 -> 8`, `fn 6 -> 5`
- `base000211`: `matched 2 -> 3`, `fn 1 -> 0`
- `base000259`: `matched 5 -> 6`, `fn 2 -> 1`

There are also two mild regressions:

- `base000017`: `matched 9 -> 8`
- `base000294`: `matched 4 -> 3`

Interpretation:

- A-only tuning is helping in some localized cases,
- but current evidence does **not** support a claim that it has already transformed the 2B model into a substantially different detector at step `300`.

### 1.4 The semantic picture is stronger than the raw FP counts suggest

Visual audit of the shared dense/crowded scenes suggests:

- many strict-metric FPs are semantically plausible objects,
- especially in audience / crowd / partially occluded regions,
- and some of the hardest scenes are strongly affected by annotation incompleteness or annotation-style mismatch.

So the current comparison should not be read as:

- “both models are bad because FP is high”.

A better reading is:

- both models already have strong objectness,
- both can detect many real visible objects outside the fully annotated subset,
- and the remaining high-value error analysis should focus more on **which GT objects are never or rarely emitted**, rather than treating every unmatched prediction as equally informative.

### 1.5 The most promising visible A-only gain is better crowded-scene separability

In the hardest dense scenes, especially `base000354`, the original Stage-1 checkpoint shows a stronger tendency toward:

- packed same-class proposals,
- compact same-region repetition,
- or harder-to-separate crowded outputs.

The A-only model at step `300` often looks slightly healthier there:

- more separable objects,
- somewhat cleaner atomic boxes,
- and less obvious local duplication.

This is probably the most optimistic signal from the current 2B comparison.

But it still needs to be described carefully:

- it is a **localized qualitative advantage**,
- not yet a robust claim of large overall rollout improvement.

---

## 2) What This Comparison Does **Not** Yet Verify

The current 2B A-only result should **not** be over-interpreted as the final answer for 2B geometry learning.

From `resolved_config.json` in the current A-only run:

- `bbox_geo.config.smoothl1_weight = 2.0`
- `bbox_geo.config.ciou_weight = 0.2`
- `bbox_geo.config.a1_smoothl1_weight = 0.0`
- `bbox_geo.config.a1_ciou_weight = 0.0`

And the objective list in the same artifact does **not** include an enabled `bbox_size_aux` module.

So this comparison is still missing the newer stronger geometry package the user wants to test on 2B:

- stronger CIoU emphasis,
- and explicit bbox-size auxiliary supervision.

Therefore the current evidence supports only the narrower statement:

- with the present 2B Stage-1 checkpoint and the current A-only continuation settings, the models look similar and only modestly different by step `300`.

It does **not** support the stronger statement:

- “A-only tuning is ineffective on 2B even with the newer geometry modules”.

That remains open and should be tested directly with a matched rerun.

---

## 3) Working Hypotheses For The Remaining FN Problem

The priority question now is:

- why are some visually clear objects still missing from rollout,
- even when the model otherwise looks semantically strong?

The current working hypotheses are:

### 3.1 Prefix sensitivity / rollout-path dependence

Hypothesis:

- once the model has emitted a partial object list, the resulting prefix strongly influences which objects it emits next,
- and early local decisions can push the continuation into a different sequence regime.

Why this is plausible here:

- training uses structured serialized lists,
- the run uses `custom.object_field_order = desc_first`,
- and the default training contract uses `custom.object_ordering = sorted`.

So the model may learn a strong next-object prior conditioned on:

- the objects already emitted,
- their order,
- and the current apparent list length / completion state.

If that prior is too strong, a semantically present object may still be skipped because the rollout has drifted onto a different continuation path.

### 3.2 Sequence-length / EOS pressure

Hypothesis:

- the model may know additional objects, but still prefer to end the list early because the partial sequence already “looks long enough” relative to training-time object-list patterns.

This is especially plausible in scenes where:

- several unmatched predictions are actually semantically correct visible objects,
- but the model still fails to enumerate all labeled GT instances.

In that regime, the problem may not be:

- “the model cannot see the object”.

It may instead be:

- “the continuation policy prefers to stop before listing every recoverable object”.

### 3.3 Sorted-order prior may be helping optimization while hurting real rollout recall

The data contract explicitly states:

- `custom.object_ordering: sorted` is the default,
- `random` ordering exists as an ablation mode,
- and `random` ordering is supported for dataset build, not as the default production path.

This creates a concrete possibility:

- sorted order may speed up learning and stabilize serialization,
- but it may also over-bias the model toward one canonical emission trajectory,
- making real rollout recall more fragile once the prefix deviates from the exact training-style object order.

This is a particularly important hypothesis because it yields a clean ablation with an existing config surface.

---

## 4) Experiment Design: How To Separate “Cannot See” From “Did Not Roll Out”

The next experiments should aim to partition FN into:

1. objects the model **never** emits,
2. objects the model emits **sometimes**, but not under greedy / canonical rollout,
3. objects the model emits only under certain prefix histories,
4. objects that are not truly missing but are represented by nearby or semantically plausible alternate boxes.

### 4.1 Experiment A: K-sampling FN coverage study

Goal:

- determine whether current FN are mostly “never emitted” or “sometimes emitted”.

Design:

- Use a fixed image set:
  - the existing 10 monitored images,
  - plus an expanded dense-scene subset where the user already observed annotation mismatch and high crowd density.
- Compare at least:
  - original Stage-1 2B checkpoint,
  - current A-only 2B continuation checkpoint,
  - and future stronger-geometry 2B checkpoints when available.
- For each image, run `K` rollouts under a small temperature grid:
  - `temperature ∈ {0.0, 0.2, 0.4}`
  - `top_p = 1.0`
  - `repetition_penalty = 1.1`
  - fixed seeds per sample.
- Suggested initial budget:
  - `K = 16` per temperature,
  - then expand to `K = 32` only on the images where recall is most ambiguous.

Outputs to record:

- per-image union-of-rollouts recall,
- per-GT-object hit rate across rollouts,
- per-image object-count distribution,
- parse validity rate,
- same-desc near-overlap counts.

Decision rule:

- if many FN objects are hit at least once across sampled rollouts, the bottleneck is probably **rollout policy / decoding / stop behavior**, not raw visual incapacity;
- if some GT objects are never emitted even across sampled rollouts, those are stronger candidates for true visual or representational weakness.

### 4.2 Experiment B: Prefix perturbation study

Goal:

- test whether the continuation path is highly sensitive to the rollout prefix.

Design:

For the same image, compare continuation under several prefix conditions:

1. no prefix beyond the image-only prompt,
2. canonical prefix:
   first `N` objects in the training-style sorted order,
3. same objects, but locally shuffled order,
4. same objects, but one adjacent swap,
5. same objects, but with one semantically plausible extra object inserted,
6. model-self prefix:
   use the model’s own first `N` emitted objects, then continue.

Evaluate for multiple prefix lengths:

- `N ∈ {1, 2, 4, 8}`

Outputs to record:

- continuation object set,
- EOS position,
- final pred count,
- GT hit set added **after** the prefix boundary,
- Jaccard overlap between continuations from different prefixes.

Decision rule:

- if small prefix changes cause large changes in later FN / recall, then prefix-path dependence is a real causal factor;
- if continuation is stable across prefix variants, then prefix sensitivity is probably not the main explanation.

### 4.3 Experiment C: Length-prior / EOS-pressure study

Goal:

- test whether the model stops because of a learned sequence-length prior rather than because no more objects are available.

Design:

- Reuse the sampled rollouts from Experiment A.
- For each image and rollout, log:
  - emitted object count,
  - generated token count,
  - whether additional GT objects remain unmatched at EOS.
- Compare those quantities to the training-data object-count distribution for the matched subset of images.

Optional stronger version:

- score the EOS / next-object competition at object boundaries if the chosen backend can expose per-step logprobs cleanly.

Decision rule:

- if longer-but-still-valid sampled rollouts consistently recover “missing” GT objects, then early stop pressure is likely part of the current FN pattern;
- if longer rollouts mostly add only noisy extras and almost never recover the missing GT instances, then EOS pressure is not the main bottleneck.

### 4.4 Experiment D: Sorted-vs-random object-ordering SFT ablation

Goal:

- test whether the canonical sorted-order prior is hurting real rollout recall.

Why this ablation is clean:

- `custom.object_ordering` already supports `sorted` and `random`,
- the data contract explicitly describes `random` as an ablation mode,
- so this can be tested without inventing a new training contract.

Minimal design:

- Start from the matched 2B Stage-1 recipe that produced
  `output/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate_merged-1332`
  or its nearest reproducible config equivalent.
- Change only:
  - `custom.object_ordering: random`
- Keep fixed:
  - model size,
  - dataset,
  - prompt variant (`coco_80`),
  - object field order (`desc_first`),
  - coord-token setup,
  - training budget.

Important runtime caveat:

- encoded sample cache only supports `custom.object_ordering='sorted'`,
- so this ablation must run with cache disabled / bypassed.

Primary readout:

- full-val rollout metrics,
- monitored dense-scene visualizations,
- and the Experiment-A FN coverage analysis.

Decision rule:

- if random-order SFT improves sampled recall coverage or reduces systematic FN concentration without harming format stability too much, then the sorted-order prior is likely part of the current rollout bottleneck;
- if random order hurts precision/format and does not improve FN coverage, then sorted order is probably helping more than hurting.

### 4.5 Experiment E: Stronger geometry package on 2B

Goal:

- test the currently unverified user hypothesis that stronger geometry supervision may matter on 2B.

Design:

- rerun the 2B A-only setup with the newer stronger geometry stack:
  - higher CIoU emphasis,
  - and explicit `bbox_size_aux`.
- keep the rest of the 2B recipe as matched as possible.

Reason to keep this separate from the prefix study:

- geometry tuning and prefix sensitivity answer different questions,
- and combining both changes in one run would make diagnosis much harder.

---

## 5) Recommended Execution Order

To maximize signal while minimizing confounds, the recommended order is:

1. **K-sampling FN coverage study** on current checkpoints  
   Fastest way to determine whether the FN problem is mostly incapacity or rollout policy.

2. **Prefix perturbation study** on the same images  
   Most direct test of the user’s main hypothesis about rollout-path dependence.

3. **Sorted-vs-random SFT ablation**  
   Clean structural test of whether the learned sorted-order prior is helping or harming recall.

4. **Stronger-geometry 2B rerun**  
   Important, but logically separate from the prefix/FN question.

This order gives the quickest route to the most decision-relevant question:

- does the model already know the “missing” objects and simply fail to roll them out?

---

## 6) Code And Config Handles For The Next Pass

Useful config / contract surfaces:

- `docs/data/CONTRACT.md`
- `configs/stage1/sft_base.yaml`
- `src/config/schema.py`

Concrete ordering / cache handles:

- `custom.object_ordering` schema support in `src/config/schema.py`
- dataset ordering logic in `src/datasets/dense_caption.py`
- cache caveat in `src/datasets/encoded_sample_cache.py`

Useful rollout / parity / visualization tools:

- `src/analysis/rollout_parity.py`
- `scripts/analysis/debug_rollout_collection_parity.py`
- `vis_tools/vis_monitor_dump_gt_vs_pred.py`
- `vis_tools/vis_coordexp.py`

These are the most direct existing handles for building:

- fixed-image rollout studies,
- prefix-controlled continuation studies,
- and side-by-side GT-vs-Pred visual audits.

---

## 7) Current Bottom Line

The current 2B result is encouraging, but subtle:

- the original Stage-1 2B checkpoint is already very strong,
- the A-only continuation at step `300` looks only modestly different overall,
- but it may already be improving crowded-scene separability in a few important cases.

The next research question is therefore no longer:

- “can the model detect objects at all?”

It is much more specifically:

- “why does rollout still omit some objects that the model plausibly knows?”

The highest-value next experiments are the ones that can convert today’s FN set into:

- **never-emitted objects**,
- **prefix-sensitive objects**,
- **EOS-suppressed objects**,
- and **annotation-policy ambiguities**.

That decomposition is likely to be much more informative than another raw FP/FN aggregate alone.
