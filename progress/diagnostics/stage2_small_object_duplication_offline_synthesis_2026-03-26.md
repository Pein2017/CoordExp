---
title: Stage-2 Small-Object Duplication Offline Diagnostics Synthesis
status: active-diagnostic
scope: stage2-channel-b
topics: [stage2, duplication, small-objects, offline-diagnostics, decoding, prefix-dynamics, teacher-forcing]
references:
  - progress/diagnostics/stage2_near_duplication_2026-03-05.md
  - progress/diagnostics/stage2_pseudo_positive_k4_coord_only_findings_2026-03-24.md
  - .worktrees/small-object-duplication-offline-diagnostics/progress/diagnostics/small_object_duplication_offline_protocol_2026-03-25.md
  - .worktrees/small-object-duplication-offline-diagnostics/progress/diagnostics/small_object_duplication_offline_findings_2026-03-26.md
  - .worktrees/duplication-offline-diagnostics-cuda1/progress/diagnostics/stage2_small_object_duplication_offline_diagnostics_2026-03-26.md
---

# Stage-2 Small-Object Duplication Offline Diagnostics Synthesis (2026-03-26)

Date: 2026-03-26  
Purpose: consolidate the protocol and findings from two parallel offline experiments run in separate worktrees against `output/stage2_ab/prod/pseudo_positive-ckpt_300_merged-v1`.

The short version is:

- Both experiments agree the pathology is not just high-temperature sampling noise.
- The shared best read is a prefix-conditioned local attractor or local-basin failure that emits nearby shifted same-class boxes.
- The broader harness shows the failure is not generically reproducible from any one-object prefix, which points to accumulated autoregressive state.
- The crowded deep-dive shows that once the right local state is reached, the model can lack a reliable margin for the true next instance over duplicate continuation.
- So the current evidence implicates both state-local rollout dynamics and insufficient anti-duplicate pressure, without yet supporting a blanket claim that the model globally prefers duplicates in all prefix states.

## 0) Source Documents and Artifact Roots

### Experiment A: broad fixed-checkpoint harness (`small-object-duplication-offline-diagnostics`)

- Worktree: `.worktrees/small-object-duplication-offline-diagnostics`
- Protocol note: `.worktrees/small-object-duplication-offline-diagnostics/progress/diagnostics/small_object_duplication_offline_protocol_2026-03-25.md`
- Findings note: `.worktrees/small-object-duplication-offline-diagnostics/progress/diagnostics/small_object_duplication_offline_findings_2026-03-26.md`
- Config: `configs/analysis/small_object_duplication/default.yaml`
- Artifact root: `output/analysis/small-object-duplication-ckpt300-offline/`
- Scope: `43` monitor-dump samples, `12` mined duplication cases, `8` crowded controls, decode temperatures `{0.0, 0.01, 0.05, 0.1, 0.2}`, one-object prefix probes, and teacher-forced candidate scoring.

### Experiment B: crowded subset deep-dive (`duplication-offline-diagnostics-cuda1`)

- Worktree: `.worktrees/duplication-offline-diagnostics-cuda1`
- Findings note: `.worktrees/duplication-offline-diagnostics-cuda1/progress/diagnostics/stage2_small_object_duplication_offline_diagnostics_2026-03-26.md`
- Config: `configs/analysis/small_object_duplication_diagnostics/ckpt300_crowded_cuda1.yaml`
- Trustworthy artifact root: `output/analysis/ckpt300_crowded_cuda1_v2_bboxfix/`
- Scope: `8` crowded monitor-selected samples with at least one small GT object, decode temperatures `{0.0, 0.1, 0.3, 0.7}`, per-sample local-pair metrics, prefix probes, and exact-duplicate counterfactual scoring.
- Important correction: `_extract_bbox_px(...)` had to be fixed to accept `bbox_2d.points=[x1, y1, x2, y2]`; earlier pre-`v2_bboxfix` artifacts should be treated as superseded.

## 1) Consolidated Protocol

The two experiments share the same high-level agenda:

1. Mine hard cases from monitor dumps around small-object duplication.
2. Hold the checkpoint fixed and test image-only decode under temperature sweeps.
3. Probe whether controlled prefixes can recreate or suppress the bad continuation.
4. Score real-next-object versus duplicate-like continuations under controlled prefix states.

Important difference in emphasis:

- Experiment A is the broader harness and is best for cohort-level statements about prevalence, prefix reproducibility, and clean-prefix scoring.
- Experiment B is the tighter crowded-pack deep-dive and is best for per-sample local-attractor evidence and exact-duplicate margin checks after the bbox parsing fix.

## 2) Shared Findings

### 2.1 Not purely sampling noise

- Experiment A: on `12` mined duplication cases, `7/12` already show `small_duplicate_like_pair_count > 0` at `temperature=0.0`; mean duplicate-like pair count then rises from `10.0` at `0.0` to `43.97` at `0.1`.
- Experiment B: greedy decode at `temp=0.0` already yields `mean_same_desc_local_pairs = 534.125` and `mean_fp_count = 34.625` on the crowded subset.
- Combined read: sampling modulates severity, but the base model already contains the failure mode.

### 2.2 Nearby shifted same-desc drift is more important than exact box cloning

- Experiment B gives the clearest evidence here: at `temp=0.0`, `mean_same_desc_local_pairs = 534.125` while `mean_same_desc_iou90_pairs = 6.75`; at `temp=0.1`, `675.375` versus `19.625`.
- This matches the observed "line of nearby shifted duplicates" pattern and suggests local same-class drift rather than simple exact-box repetition.
- Experiment A is compatible with the same story: the most informative positive trigger case depends on small spatial perturbations rather than only an exact copied box.

### 2.3 The failure is a local attractor, but not a generic one-object rule

- Experiment A: among the first `8` continuation cases, only `2` produce any continuation at all; `6` terminate immediately under every tested one-object condition. The best positive case (`step_000002_sample_020_87140591556593`) shows a real but narrow and anisotropic basin: `first_correct_gt` and one specific jitter (`dx0_dy4`) re-trigger duplication, while the exact `first_correct_pred` and most other jitters collapse.
- Experiment B: `base_idx=49922` is the clearest post-hit attractor case. Once the rollout enters the local `book` mode, it keeps emitting nearby same-desc objects (`fp_count=123`, `same_desc_local_pairs=2607`, `max_local_same_desc_cluster_size=120` at `temp=0.0`).
- Combined read: the attractor is real, but it activates only for a subset of local prefix states, often after more accumulated context than a generic first correct small object.

### 2.4 Temperature changes severity, but not monotonically or universally

- Experiment A: duplicate-like pair counts worsen as temperature increases through `0.1`, then soften at `0.2`, but remain elevated.
- Experiment B: some scenes improve at higher temperature (`base_idx=3068`), while others get much worse under sampling (`base_idx=39927`).
- Combined read: temperature is a control knob, not the root cause. It can either escape or enter a bad local branch depending on the scene.

## 3) Where the Two Experiments Diverge

### 3.1 Score-margin conclusions depend on the prefix state being scored

- Experiment A's clean `focus_pred` scorer is relatively encouraging: `margin_gt_minus_duplicate` is positive in `6/8` cases, negative in `2/8` cases, with mean about `+0.14`. Duplicate beats GT on `desc` in only `1/8` case and on `coord` in `2/8` cases.
- Experiment B's crowded deep-dive is harsher: on `4` sampled counterfactual scenes, `gt_next` beats the best exact duplicate only `1/4` of the time, ties `2/4`, and loses `1/4` (`gt_beats_duplicate_rate = 0.25`).
- Safest synthesis: the objective is probably insufficient but not globally inverted. Under clean anchor-like prefixes, the model often still prefers the real next object; under some crowded or trigger-like states, that margin can collapse or disappear.

### 3.2 Prefix-probe evidence is uneven in quality

- Experiment A's prefix study produces both strong negative evidence, since most one-object prefixes fail to recreate the loop, and one strong positive trigger case.
- Experiment B's prefix probe is not usable yet: every row ends with `pred_errors = ["empty_pred"]`, `finish_reason = "length"`, and `continuation_pred_count = 0`.
- Safest synthesis: use Experiment A for current prefix-state claims. Treat Experiment B's prefix probe as a tooling gap, not as evidence for or against coverage awareness.

### 3.3 Artifact trust is asymmetric

- Experiment A already records its protocol and completed findings as a coherent fixed-checkpoint study.
- Experiment B required an important diagnostics-layer correction in `_extract_bbox_px(...)`; only the `ckpt300_crowded_cuda1_v2_bboxfix` artifact root should be treated as final.
- Safest synthesis: any future cross-run comparison should keep the corrected `v2_bboxfix` subset fixed.

## 4) Best Current Causal Read

The strongest joint interpretation is:

1. The pathology is not just decode randomness. Greedy decode already shows it.
2. The pathology is not well described as "after any first correct small object, the model loops forever." Most clean one-object prefixes do not recreate it.
3. The failure is best described as a prefix-conditioned local attractor that activates only for some accumulated autoregressive states.
4. Small objects remain a plausible special vulnerability because small coordinate drift preserves local plausibility, same-desc competition is common, and the decisive margin often seems to live in the coordinate or continuation ranking.
5. The objective or score surface is implicated, but the evidence does not yet justify the stronger claim that the model globally prefers duplicate re-enumeration in every state.

In practice, the current evidence points to a combined problem:

- state-local rollout dynamics create the bad basin,
- and once the model is in that basin, anti-duplicate preference can be too weak to reliably move it back toward the true next instance or a clean stop.

## 5) Most Important Open Limitations

- The clean-prefix scorer in Experiment A does not yet score the actual trigger prefixes (`focus_gt`, trigger jitters, or later onset prefixes).
- The Experiment A focus policy (`earliest_matched_small_or_first_matched`) may often start too early and miss the real onset boundary.
- The Experiment B prefix probe currently fails at the artifact level and needs repair before it can inform causal claims.
- The duplicate-like mining metric is useful for diagnosis but is still only a screening signal, especially in crowded scenes.

## 6) Recommended Next Experiments

### 6.1 Highest priority: onset-conditioned continuation

- Build prefixes from the actual first boundary where the duplicate cluster begins in the decoded bad trajectory, not from `earliest_matched_small_or_first_matched`.
- Verification: duplication should reproduce on a much larger fraction of current hard cases if accumulated state is the main driver.

### 6.2 Score the actual trigger states

- Extend scoring so it can evaluate `focus_gt`, trigger jitters, and later multi-object onset prefixes, not only clean `focus_pred`.
- Verification: compare `remaining_gt`, `duplicate_jitter`, and `close` at the exact trigger state.

### 6.3 Repair and rerun the crowded-pack prefix probe

- Fix the prefix probe format in the `duplication-offline-diagnostics-cuda1` harness so continuations are parseable and non-empty.
- Verification: the rerun on the same corrected `8`-sample crowded pack should produce nonzero `continuation_pred_count` rows and usable per-condition continuation metrics.

### 6.4 Add a decode-time duplicate veto as a control experiment

- Run the same hard cases with a local same-desc duplicate suppression or stop rule.
- Verification: if the veto sharply reduces the pathology with limited collateral damage, then the model likely retains usable latent signal and the gap is largely rollout control; if not, the trigger-state representation or objective is genuinely too weak.

### 6.5 Only then decide on training-side changes

- If repaired trigger-prefix scoring still shows weak or flat margin under small spatial drift, prioritize stronger training-side pressure against near-shift duplicate continuation and better instance-level discrimination for small objects.
- Verification: rerun the fixed hard-case cohort and compare both duplicate metrics and matched-object retention.

## 7) Bottom Line

The two worktrees tell a consistent story once they are placed side by side:

- The duplication failure is real under fixed-checkpoint offline decode and is not just a temperature artifact.
- The most faithful current description is a state-local same-desc spatial attractor, not a universal duplicate preference after any first hit.
- The right trigger state can make the model's anti-duplicate margin weak or absent, but the clean-prefix scorer suggests that failure is not globally present in every prefix regime.
- The next decisive step is to move both continuation and scoring onto the actual onset prefix, then rerun the same fixed hard cases.
