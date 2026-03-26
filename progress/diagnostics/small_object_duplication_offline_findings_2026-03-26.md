---
title: Small Object Duplication Offline Findings (Checkpoint 300, Fixed-Checkpoint Study)
status: active-diagnostic
scope: stage2-offline-analysis
topics: [small-object-duplication, offline-diagnostics, decoding, prefix-dynamics, teacher-forcing]
references:
  - progress/diagnostics/small_object_duplication_offline_protocol_2026-03-25.md
  - progress/diagnostics/stage2_near_duplication_2026-03-05.md
  - progress/diagnostics/stage2_pseudo_positive_k4_coord_only_findings_2026-03-24.md
  - src/analysis/small_object_duplication_study.py
---

# Small Object Duplication Offline Findings (Checkpoint 300, Fixed-Checkpoint Study) (2026-03-26)

Date: 2026-03-26  
Status note: this note records the completed offline diagnostic study requested for
`output/stage2_ab/prod/pseudo_positive-ckpt_300_merged-v1`, using monitor-dump-derived crowded samples,
temperature sweeps, one-object prefix continuation probes, and teacher-forced candidate scoring on `cuda0`.

The short version is:

- the duplication phenomenon is **not** just temperature-driven sampling noise,
- the strongest current evidence supports a **prefix-conditioned local attractor** rather than a global
  "the model always prefers duplicates" story,
- one-object prefix probes do **not** generically reproduce the failure, which suggests the bad regime often
  depends on accumulated autoregressive state rather than any arbitrary first correct small object,
- under the clean `focus_pred` prefix used by the current score probe, the model usually still ranks a
  real remaining GT object above a jittered duplicate,
- and the next best experiment is to score and continue from the **actual onset prefix** where the first
  duplicate cluster begins, not from `earliest_matched_small_or_first_matched`.

---

## 0) Artifacts

Study outputs:

- Run dir:
  `output/analysis/small-object-duplication-ckpt300-offline/`
- Top-level summary:
  `output/analysis/small-object-duplication-ckpt300-offline/summary.json`
- Cohort summary:
  `output/analysis/small-object-duplication-ckpt300-offline/cohort/summary.json`
- Decode summary:
  `output/analysis/small-object-duplication-ckpt300-offline/decode/summary.json`
- Prefix summary:
  `output/analysis/small-object-duplication-ckpt300-offline/prefix/summary.json`
- Score summary:
  `output/analysis/small-object-duplication-ckpt300-offline/score/summary.json`
- Raw stage outputs:
  `.../cohort/all_samples.jsonl`
  `.../cohort/duplication_cases.jsonl`
  `.../cohort/crowded_controls.jsonl`
  `.../decode/results.jsonl`
  `.../prefix/results.jsonl`
  `.../score/results.jsonl`

Study inputs:

- Checkpoint:
  `output/stage2_ab/prod/pseudo_positive-ckpt_300_merged-v1`
- Config:
  `configs/analysis/small_object_duplication/default.yaml`
- Monitor dumps:
  `output/stage2_ab/prod/pseudo_positive/k_4-eff_size_96-b_ratio_0.85-epoch_1-global_struct_ce_fixed-stronger_FN-from_300_v1/v0-20260325-025922/monitor_dumps/step_000002.json`
  `output/stage2_ab/prod/pseudo_positive/k_4-eff_size_96-b_ratio_0.85-epoch_1-global_struct_ce_fixed-stronger_FN-from_300_v1/v0-20260325-025922/monitor_dumps/step_000036.json`

Relevant harness implementation:

- Study module:
  [src/analysis/small_object_duplication_study.py](/data/CoordExp/.worktrees/small-object-duplication-offline-diagnostics/src/analysis/small_object_duplication_study.py)
- Protocol note:
  [small_object_duplication_offline_protocol_2026-03-25.md](/data/CoordExp/.worktrees/small-object-duplication-offline-diagnostics/progress/diagnostics/small_object_duplication_offline_protocol_2026-03-25.md)

---

## 1) Study Design

The completed study keeps the model fixed and asks three distinct questions:

1. Does the pathology persist under image-only decode, and how does temperature change it?
2. Can a controlled one-object prefix push the model into, or out of, the duplication regime?
3. Under a fixed prefix, does the model actually score a local duplicate above the true next object?

The default config implements:

- cohort mining from monitor dumps,
- image-only decode sweeps over `temperature in {0.0, 0.01, 0.05, 0.1, 0.2}`,
- one-object continuation probes from `pred`, `gt`, and small bbox jitters,
- teacher-forced candidate scoring for `remaining_gt`, `duplicate_jitter`, and `close`.

Important config surfaces:

- `monitor_dumps.top_duplication_cases = 12`
- `monitor_dumps.top_control_cases = 12`
- `prefix.max_cases = 8`
- `prefix.focus_policy = earliest_matched_small_or_first_matched`
- `prefix.sources = [pred, gt]`
- eight prefix jitters: `(+/-2, 0)`, `(0, +/-2)`, `(+/-4, 0)`, `(0, +/-4)`
- `scoring.max_cases = 8`
- `scoring.max_remaining_gt_candidates = 5`

Important scoring limitation:

- the current score stage always builds its prefix from `focus_pred`,
- it does **not** score the `focus_gt` or jitter-triggered prefixes that turned out to matter in the
  continuation study,
- so score results are informative about a clean anchor-like prefix state, but not yet definitive for
  the strongest trigger states.

---

## 2) Cohort Summary

The harness processed `43` monitor-dump samples and selected:

- `12` duplication cases
- `8` crowded controls

This means the offline study is grounded in a relatively small but nontrivial hard-case slice.

Interpretation note:

- this is a targeted diagnostic cohort, not an unbiased dataset sample,
- and the duplication-like metric used to mine it is best treated as a screening signal rather than a perfect
  definition of the failure.

---

## 3) Main Findings

### 3.1 Duplication is not purely sampling noise

On the mined duplication cohort, image-only decode already shows substantial duplication under greedy decode:

- `temperature = 0.0`:
  - mean predicted objects: `14.42`
  - mean small duplicate-like pair count: `10.0`
- `temperature = 0.01`:
  - mean duplicate-like pair count: `16.39`
- `temperature = 0.05`:
  - mean duplicate-like pair count: `33.89`
- `temperature = 0.1`:
  - mean duplicate-like pair count: `43.97`
- `temperature = 0.2`:
  - mean duplicate-like pair count: `34.25`

So temperature usually **amplifies** the pathology, but the base policy already contains it.

From the raw decode rows:

- `7/12` duplication cases still have `small_duplicate_like_pair_count > 0` at `temperature = 0.0`
- `8/12` do so at `0.01`
- `8/12` do so at `0.05`
- `8/12` do so at `0.1`
- `9/12` do so at `0.2`

This is the strongest evidence in the run that the regime is not a mere high-temperature artifact.

### 3.2 One-object prefix reproduction is sparse and highly state-specific

The most important finding in the whole study is that the one-object prefix probes do **not** generically
recreate the failure.

Among the first `8` selected duplication cases:

- only `2` cases produced any continuation at all under the one-object prefix study,
- the other `6` cases terminated immediately for **every** tested condition.

This strongly suggests:

- the bad regime often depends on more accumulated autoregressive state than "one first correct object",
- or the current focus policy is often selecting the wrong object relative to the true duplication onset,
- or both.

This is already a meaningful negative result: the failure is **not** well described as
"after any first correct small object, the model enters a duplicate loop."

### 3.3 There is a robust trigger case with a narrow, anisotropic basin

Case:

- `step_000002_sample_020_87140591556593`

This case is the clearest positive example of a local attractor.

In base image-only decode, it duplicates heavily across all tested temperatures.

In the one-object prefix study:

- `first_correct_gt` consistently reactivates long duplicate-heavy rollouts,
- `first_correct_pred_jitter_dx0_dy4` also consistently reactivates long duplicate-heavy rollouts,
- but the exact `first_correct_pred` prefix collapses immediately,
- and every other tested jitter also collapses immediately.

So the basin is:

- real,
- reproducible,
- but **very narrow**,
- and spatially **anisotropic**.

This is much closer to a "specific hidden-state / local-prefix attractor" story than to a generic
same-instance confusion story.

### 3.4 There is also a borderline, temperature-sensitive trigger regime

Case:

- `step_000036_sample_013_87140591518466`

This case behaves differently:

- base image-only decode stays benign in the offline harness,
- `first_correct_pred` and `first_correct_pred_jitter_dx-2_dy0` keep the continuation alive,
- but only the plain `first_correct_pred` variant becomes duplication-like at `temperature = 0.2`,
- most other jitters kill the continuation entirely.

This looks like a weaker, more sampling-sensitive instability:

- the underlying state is fragile,
- the local mode exists,
- but it is not a strong deterministic attractor like the previous case.

### 3.5 Under the clean `focus_pred` prefix, the model usually still prefers a real next object

The teacher-forced score probe evaluated `8` cases. The main summary metric is:

- `margin_gt_minus_duplicate = best_remaining_gt_full - best_duplicate_full`

Results:

- positive in `6/8` cases,
- negative in `2/8` cases,
- mean margin approximately `+0.14`.

So under the current clean `focus_pred` prefix:

- the model usually does **not** place the jittered duplicate above the best remaining GT object.

This means the current results do **not** support a strong global claim that:

- "the objective has simply trained the model to prefer duplicate re-enumeration over true next instances."

That may still happen in some trigger states, but it is not what the clean-prefix scorer shows by default.

### 3.6 Coverage awareness exists, but only weakly

The `close` candidate is better than the best duplicate in only `2/8` scored cases.

So the model sometimes knows to close the object list, but this signal is not strong or consistent enough
to reliably suppress local re-enumeration.

This supports a milder statement:

- the model has partial coverage awareness,
- but it is not robust enough to be protective in pathological local states.

### 3.7 The score breakdown points more to unstable spatial ranking than to pure desc confusion

From the raw scored candidates:

- duplicate beats GT on `full` score in `2/8` cases,
- duplicate beats GT on `coord` score in `2/8` cases,
- duplicate beats GT on `desc` score in only `1/8` case.

In several same-class cases, the desc scores are effectively tied and the separation comes almost entirely
from the coordinate side.

This suggests:

- the main weakness is **not** that the model cannot represent the category identity,
- it is that, in some contexts, the spatial/object continuation ranking becomes unstable enough for a local
  jittered repeat to compete with or beat the true next instance.

---

## 4) Best Current Causal Read

The strongest current interpretation is:

1. The pathology is **not purely decoding randomness**.
   Greedy decode already shows it on most of the mined hard cases.

2. The pathology is **not well explained by a generic one-object duplicate preference**.
   Most one-object prefixes do not recreate it at all.

3. The failure is best described as a **prefix-conditioned local attractor** that only activates for a subset
   of autoregressive states.

4. Small objects are still a plausible special vulnerability because:
   - same-desc competition is common,
   - tiny coordinate jitters preserve local visual plausibility,
   - and the score decomposition suggests the decisive margin often lives in the coordinate term.

5. The current objective is probably **insufficient but not globally inverted**.
   The clean-prefix scorer usually favors remaining GT over duplicate, which means the model retains some
   latent discriminative signal.

So the dominant issue appears to be:

- a state-evolution / local-basin problem,
- not just a universal inability to tell "same instance" from "different nearby instance."

---

## 5) Important Limitations

### 5.1 The scorer does not yet probe the actual trigger prefixes

The strongest duplication-triggering prefixes in the continuation study were:

- `focus_gt`
- and one specific jittered prefix in the robust case

But the current score stage only evaluates the `focus_pred` prefix.

So the scorer currently answers:

- "what does the model prefer under a clean anchor-like one-object prefix?"

It does **not** yet answer:

- "what does the model prefer at the exact prefix state that actually triggers the duplicate loop?"

### 5.2 The focus policy may miss the true onset boundary

The continuation study currently starts from:

- `earliest_matched_small_or_first_matched`

That may be too early for many failures.

If duplication commonly starts after more context has been accumulated, this study design will systematically
underestimate how reproducible the attractor really is.

### 5.3 The duplicate-like metric is useful but coarse

The same duplicate-like graph metric that usefully surfaces hard failures can also fire in crowded scenes,
especially at higher temperatures.

So pair counts are best interpreted as:

- a comparative diagnostic signal,
- not a final semantic definition of a bad duplicate.

---

## 6) Recommended Next Experiments

### 6.1 Recommended first: onset-conditioned continuation

Instead of prefixing from `earliest_matched_small_or_first_matched`, build prefixes from the **actual onset**
in the decoded duplicated trajectory:

- find the first boundary where the duplicate cluster begins,
- replay continuation from the exact prefix up to that point,
- and test controlled perturbations there.

Verification target:

- if duplication now reproduces for most currently failing cases, then the main driver is accumulated state,
  not single-object ambiguity.

### 6.2 Score the actual trigger prefixes

Extend the score stage so it can score:

- `focus_gt`,
- trigger jitters,
- and later multi-object onset prefixes,

not just the clean `focus_pred` prefix.

Verification target:

- compare `remaining_gt`, `duplicate_jitter`, and `close` at the exact trigger state.

### 6.3 Run a decode-time duplicate veto on the same cases

Add a local duplicate-aware suppression rule at inference time and rerun the current decode cohort.

Verification target:

- if the veto sharply reduces the pathology with limited collateral damage, then the model contains enough
  latent signal and the main gap is rollout control,
- if not, then the trigger-state representation/objective is genuinely too weak.

---

## 7) Actionable Implications

The current artifacts argue against jumping straight to a large retraining change based only on the claim that:

- "the model globally prefers duplicates over true next objects."

The evidence better supports this ordering of effort:

1. identify the actual onset state,
2. probe the exact trigger prefix,
3. then decide whether the best mitigation is:
   - stronger training-side duplicate suppression,
   - better coverage / close pressure,
   - or decode-time local duplicate control.

If the onset-prefix experiments confirm the same narrow-basin behavior seen in the best current positive case,
then the most likely root issue is a **state-local spatial attractor** rather than a broad semantic failure.
