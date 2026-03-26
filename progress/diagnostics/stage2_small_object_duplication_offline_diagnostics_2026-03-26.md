---
title: Stage-2 Small-Object Duplication Offline Diagnostics
status: active-diagnostic
scope: stage2-channel-b
topics: [stage2, duplication, small-objects, offline-diagnostics, decoding]
references:
  - docs/PROJECT_CONTEXT.md
  - progress/diagnostics/stage2_near_duplication_2026-03-05.md
  - progress/diagnostics/stage2_ul_capture_highres1024_2026-03-09.md
---

# Stage-2 Small-Object Duplication Offline Diagnostics (2026-03-26)

Date: 2026-03-26  
Authoring context: fixed-checkpoint offline diagnostics in isolated worktree  
Checkpoint under test: `output/stage2_ab/prod/pseudo_positive-ckpt_300_merged-v1`

This note records the offline diagnosis we ran for the observed **small-object duplication** failure mode in the Stage-2 two-channel setup.

The central question was:

- When the model correctly detects an object and then continues producing a **line of nearby shifted duplicates**, is that mainly because:
  - local visual features are too invariant under small spatial drift,
  - the autoregressive decode enters a local attractor and keeps enumerating the same mode,
  - or the objective/score surface fails to assign enough margin to the true next instance over a duplicate-like continuation?

The study used:

- a fixed checkpoint,
- crowded failure samples selected from `monitor_dumps`,
- offline decoding sweeps over temperature,
- prefix-conditioned continuation probes,
- and counterfactual teacher-forced scoring of `gt_next` versus duplicate candidates.

Important headline:

- The finished run gives **strong evidence for a prefix-conditioned local duplicate attractor with weak anti-duplicate margin**.
- The strongest trustworthy evidence comes from the **decode sweep** and the **exact-duplicate counterfactual**.
- The **prefix probe** and the **spatial-jitter branch** of the counterfactual probe are currently limited by probe-design issues, so they should not be over-interpreted.

---

## 0) Artifact Provenance

Implementation worktree:

- `/data/CoordExp/.worktrees/duplication-offline-diagnostics-cuda1`

Study config:

- `configs/analysis/small_object_duplication_diagnostics/ckpt300_crowded_cuda1.yaml`

Final artifact root:

- `output/analysis/ckpt300_crowded_cuda1_v2_bboxfix/`

Key output files:

- top-level summary: `output/analysis/ckpt300_crowded_cuda1_v2_bboxfix/report.md`
- study manifest: `output/analysis/ckpt300_crowded_cuda1_v2_bboxfix/study_manifest.json`
- crowded subset metadata: `output/analysis/ckpt300_crowded_cuda1_v2_bboxfix/subset/monitor_subset.meta.json`
- decode summaries: `output/analysis/ckpt300_crowded_cuda1_v2_bboxfix/decode/decode_manifest.json`
- decode aggregate rows: `output/analysis/ckpt300_crowded_cuda1_v2_bboxfix/decode/aggregate_metrics.jsonl`
- prefix probe rows: `output/analysis/ckpt300_crowded_cuda1_v2_bboxfix/prefix_probe/prefix_probe_rows.jsonl`
- counterfactual rows: `output/analysis/ckpt300_crowded_cuda1_v2_bboxfix/counterfactual/counterfactual_rows.jsonl`
- counterfactual summary: `output/analysis/ckpt300_crowded_cuda1_v2_bboxfix/counterfactual/counterfactual_manifest.json`

Crowded subset size:

- `8` monitor-selected samples containing at least one small GT object.

Representative high-risk subset members:

- `base_idx=62832` with monitor `fp_count=71`
- `base_idx=24202` with monitor `fp_count=59`
- `base_idx=39927` with monitor `fp_count=88`
- `base_idx=49922` with monitor `fp_count=92`

---

## 1) Harness Notes and One Important Correction

The first implementation pass exposed a diagnostics-layer bug:

- `_extract_bbox_px(...)` only treated `bbox_2d.points` as valid when they had polygon-like `8`-value coordinate lists.
- The parity artifacts actually store standard boxes as `points=[x1, y1, x2, y2]`.

Impact:

- initial local-pair, spillover, and prefix/counterfactual helper metrics were incorrectly zero or empty.

Correction:

- `_extract_bbox_px(...)` was fixed to accept `bbox_2d.points` of length `4` as `[x1, y1, x2, y2]`.

The final trustworthy run is therefore the **`v2_bboxfix`** artifact root, not the earlier partial run.

Why this matters:

- all findings below refer to the corrected artifact bundle in `ckpt300_crowded_cuda1_v2_bboxfix/`.

---

## 2) What the Decode Sweep Shows

### 2.1 Summary across temperatures

From `decode/decode_manifest.json`:

- `temp=0.0`
  - `mean_pred_count = 46.5`
  - `mean_fp_count = 34.625`
  - `mean_same_desc_local_pairs = 534.125`
  - `mean_same_desc_iou90_pairs = 6.75`
  - `spillover_rate = 0.375`
- `temp=0.1`
  - `mean_pred_count = 51.875`
  - `mean_fp_count = 39.75`
  - `mean_same_desc_local_pairs = 675.375`
  - `mean_same_desc_iou90_pairs = 19.625`
  - `spillover_rate = 0.375`
- `temp=0.3`
  - `mean_pred_count = 35.0`
  - `mean_fp_count = 23.375`
  - `mean_same_desc_local_pairs = 239.75`
  - `mean_same_desc_iou90_pairs = 0.625`
  - `spillover_rate = 0.25`
- `temp=0.7`
  - `mean_pred_count = 28.5`
  - `mean_fp_count = 16.125`
  - `mean_same_desc_local_pairs = 96.125`
  - `mean_same_desc_iou90_pairs = 0.125`
  - `spillover_rate = 0.375`

### 2.2 Main decode-level conclusion

The failure is dominated by **local same-desc drift**, not exact overlap duplication.

The strongest evidence is the gap between:

- `same_desc_local_pairs`
- and `same_desc_iou90_pairs`

At `temp=0.0`:

- local pairs: `534.125`
- IoU>=0.9 pairs: `6.75`

At `temp=0.1`:

- local pairs: `675.375`
- IoU>=0.9 pairs: `19.625`

Interpretation:

- the rollout is usually not repeating the exact same box,
- it is repeatedly generating **same-class boxes in a small neighborhood**,
- which matches the user-observed visual pattern of a shifted “line” of duplicates.

This is the cleanest artifact-level support for the hypothesis that **slight spatial drift still looks valid enough to keep the rollout going**.

### 2.3 Temperature matters, but does not solve the root cause

Temperature changes the severity, but not in a way that supports a purely decode-only story:

- `temp=0.1` is clearly the worst average setting in this pack.
- `temp=0.7` is best on average for FP count and local duplicate density.
- But some scenes still worsen under sampling, which means the model distribution itself already contains duplicate-heavy local continuations.

So the artifact-supported claim is:

- decoding dynamics modulate the failure,
- but the failure is not just an artifact of deterministic greedy decoding.

---

## 3) Strongest Per-Sample Evidence for a Local Attractor

### 3.1 `base_idx=49922`: the clearest “post-hit local attractor” case

At `temp=0.0` in `decode/temp_0/per_sample_metrics.jsonl`:

- `pred_count = 128`
- `fp_count = 123`
- `matched = 5`
- `precision = 0.039`
- `max_desc_count = 122`
- `same_desc_local_pairs = 2607`
- `same_desc_iou90_pairs = 20`
- `max_local_same_desc_cluster_size = 120`
- `first_small_match_desc = "book"`
- `first_small_match_spill_count = 58`

At `temp=0.1`:

- `fp_count = 121`
- `same_desc_local_pairs = 2963`
- `same_desc_iou90_pairs = 127`
- `first_small_match_spill_count = 66`

At `temp=0.7`:

- `pred_count = 30`
- `fp_count = 22`
- `max_desc_count = 24`
- `same_desc_local_pairs = 128`
- `first_small_match_spill_count = 9`

Interpretation:

- once the model enters the local “book” mode, it keeps emitting more nearby book-like objects,
- and it does so **after** the first small-object match,
- which is exactly the failure pattern of a local autoregressive attractor rather than a single isolated bad proposal.

### 3.2 `base_idx=3068`: duplication severity is temperature-sensitive

At `temp=0.0`:

- `pred_count = 61`
- `fp_count = 53`
- `max_desc_count = 54`
- `same_desc_local_pairs = 1193`
- `same_desc_iou90_pairs = 33`

At `temp=0.7`:

- `pred_count = 45`
- `fp_count = 30`
- `matched = 15`
- `precision = 0.333`
- `recall = 0.536`
- `max_desc_count = 25`
- `same_desc_local_pairs = 297`

Interpretation:

- here, higher temperature helps escape the local duplicate-heavy mode.

### 3.3 `base_idx=39927`: sampling can also destabilize a scene

At `temp=0.0`:

- `fp_count = 16`
- `same_desc_local_pairs = 105`

At `temp=0.1`:

- `fp_count = 91`
- `same_desc_local_pairs = 694`

At `temp=0.3`:

- `fp_count = 75`
- `same_desc_local_pairs = 1086`

At `temp=0.7`:

- `fp_count = 42`
- `same_desc_local_pairs = 203`

Interpretation:

- sampling does not uniformly suppress duplication,
- it can instead push the rollout into a worse duplicate-heavy branch.

This reinforces that the underlying score landscape already contains unstable local modes.

---

## 4) What the Counterfactual Scoring Shows

### 4.1 Exact-duplicate margin is often missing

From `counterfactual/counterfactual_manifest.json`:

- `gt_beats_duplicate_rate = 0.25`

Across the `4` sampled counterfactual scenes:

- `gt_next` beats best duplicate in `1/4`
- ties in `2/4`
- loses in `1/4`

Per-sample summary:

- sample `0`: tie
- sample `1`: `gt_next` wins by `+1.2391`
- sample `2`: duplicate wins by `-0.1095`
- sample `3`: tie

This is an important result.

It means that under the same prefix, the model often does **not** assign a higher score to the semantically correct next object than to a duplicate-like continuation.

That is direct evidence for **weak anti-duplicate margin in the model’s score surface**.

### 4.2 Example where duplicate beats `gt_next`

Sample `2` from `counterfactual_rows.jsonl`:

- `gt_next`
  - `desc_score = -0.6040`
  - `coord_score = -4.3954`
  - `full_score = -3.1316`
- `exact_duplicate`
  - `desc_score = -0.0413`
  - `coord_score = -3.7673`
  - `full_score = -3.0221`

Interpretation:

- the duplicate candidate is actually preferred,
- both at the descriptor level and at the coordinate level,
- which is consistent with the model being more comfortable staying in the same local same-class mode than transitioning to the true next instance.

### 4.3 Example where `gt_next` wins

Sample `1`:

- `gt_next full_score = -1.8002`
- `best duplicate full_score = -3.0392`

The most interesting detail is:

- `gt_next` wins mainly because its `desc_score` is much better,
- while the duplicate still has a competitive coordinate score.

Interpretation:

- the model can sometimes distinguish the correct next instance,
- but this distinction is not robust across scenes.

---

## 5) What We Can and Cannot Conclude about Representation-Level Invariance

### 5.1 What the artifacts do support

The artifacts support the following mechanism-level statement:

- after a same-class local prefix is established,
- the model often has **insufficient preference** for the true next instance over a duplicate-like continuation,
- and rollout can get trapped in a local same-desc neighborhood, emitting nearby shifted variants.

This is consistent with:

- local feature similarity under small drift,
- weak instance-level discrimination,
- and insufficient penalty for re-enumerating the same local mode.

### 5.2 What the artifacts do **not yet** cleanly prove

The spatial-jitter branch of the counterfactual probe is currently **not discriminative enough** to cleanly isolate representation sensitivity.

Observed limitation:

- every duplicate jitter variant (`±4`, `±8`, `±16`, all directions) produced the **same score** as the exact duplicate within a sample.

This likely means one of two things:

- the model is truly almost invariant to those small shifts in the current scoring setup,
- or the current probe implementation is not actually changing the effective scored token path in the intended way.

The decode sweep still strongly suggests a local-drift phenomenon, but the current jitter counterfactual is not sufficient to prove where in the stack that invariance originates.

---

## 6) Prefix Probe: Useful Goal, Unusable Current Artifact

The prefix probe was intended to answer:

- does the model show any coverage awareness or stop behavior under explicit partial prefixes,
- and does jittering the prefix box change the continuation tendency?

Current outcome:

- every row in `prefix_probe/prefix_probe_rows.jsonl` shows:
  - `pred_errors = ["empty_pred"]`
  - `generated_token_count = 512`
  - `finish_reason = "length"`
  - `continuation_pred_count = 0`

Therefore:

- the current prefix probe should be treated as **inconclusive**,
- not as evidence that the model is coverage-aware or duplication-free under prefixed continuation.

The likely interpretation is probe-format failure, not meaningful model success.

---

## 7) Overall Interpretation

The most defensible synthesis from this artifact bundle is:

1. The failure mode is primarily **prefix-conditioned local duplicate enumeration**.
2. The duplicated objects are usually **nearby shifted same-desc predictions**, not only exact-overlap clones.
3. Temperature changes how often the rollout falls into that mode, but does not remove the underlying attractor.
4. The counterfactual evidence shows that the model often lacks a **positive margin for `gt_next` over duplicate continuation**.
5. Therefore, the dominant issue is unlikely to be “decode randomness alone”; it is more consistent with a combination of:
   - weak instance discrimination under local same-class ambiguity,
   - autoregressive local-attractor dynamics,
   - and insufficient objective pressure against re-enumerating an already-covered local mode.

In short:

- **representation similarity is plausible,**
- **objective/score-surface under-penalization is definitely implicated,**
- and the observable failure in rollout looks like a **local attractor after a correct hit**.

---

## 8) Actionable Follow-Ups

### 8.1 Highest-value methodological fixes

- Fix the prefix probe so it produces parseable continuations.
  - Without this, we cannot directly test stop/coverage behavior under controlled prefixes.

- Fix the spatial-jitter counterfactual so jittered candidates actually induce distinguishable scored token paths.
  - Add a sanity assertion that `duplicate_jitter_*` changes serialized coord tokens relative to `exact_duplicate`.

- Re-run the exact same `8`-sample crowded pack after those probe fixes.
  - Keeping the subset fixed is important for causal comparison.

### 8.2 Likely mitigation directions suggested by the current artifacts

- Training/objective:
  - add stronger margin against duplicate continuation after a matched local same-class prefix,
  - especially for near-shift duplicates, not just exact same-desc IoU>=0.9 duplicates.

- Decoding:
  - local same-desc suppression or a duplicate-aware stop rule is likely to help,
  - but should be treated as a guardrail rather than the full fix.

- Representation:
  - if the repaired jitter counterfactual still shows score invariance under small shifts, that would strengthen the case for explicitly improving instance-level discrimination for small objects.

---

## 9) Bottom Line

This offline study does **not** support the idea that the observed small-object duplication is merely a superficial decode artifact.

It does support:

- a real **same-desc local-attractor regime** in autoregressive rollout,
- a strong bias toward **nearby shifted duplicates** rather than only exact duplicates,
- and a score surface that often fails to prefer the **true next instance** over duplicate-like continuations.

That is the main discovery from this pass.

