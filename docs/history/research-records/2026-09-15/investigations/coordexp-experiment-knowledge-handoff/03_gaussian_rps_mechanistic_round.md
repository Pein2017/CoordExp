---
type: investigation
title: Gaussian Coordinate Soft-Target and Ranked Probability Score Mechanistic Round
description: Negative and validity evidence from a historical first-200 rollout comparison with incompatible token-health and artifact gates.
role: historical-negative-evidence-capsule
authority: non_normative_research
status: historical-synthesis
domain: gaussian-rps-rollout-anatomy
updated: 2026-07-18
---

# Gaussian Coordinate Soft-Target and Ranked Probability Score Mechanistic Round

## Question

What can the first-200 rollout anatomy reports say about the historical Gaussian
coordinate soft-target plus Ranked Probability Score (`gaussian_rps`) arm,
prefix denoising, and ledger variants after validity gates and token health are
applied?

## Source scope

The source family `outputs/analysis/gaussian_rps_mechanistic_round/` contains five reports. Each uses `dataset_slice=first_200`, `metric_family=guarded`, and a debug-oriented rollout anatomy. The reports compare pure cross-entropy, prefix-denoising sorted, ledger pre/post-merge, and Gaussian-plus-ranked-score checkpoints; they are not a common production benchmark.

## Verdict

- **GRPS-001 — validity is limited.** Four runs have Gate 0 `ok`; pure-CE sorted has Gate 0 `fail`. Every listed run has Gate 0b `warning` and effective metric family `debug-f1ish-only`. All reports set `production_training_recommendation: none` and explicitly prohibit H1–H5 or production-training claims.
- **GRPS-002 — token health dominates interpretation.** Pure CE has 200/200 traces ending at `<|im_end|>` with zero tokens after the first end marker. Prefix sorted has 25,113 tokens after the first end marker and 25,113 end-of-text events; ledger pre-merge has 27,740/27,740; ledger post-merge 23,390/23,390; Gaussian RPS 26,640/26,640. These are materially different continuation regimes.
- **GRPS-003 — raw/guarded counts are descriptive only.** Pure CE: 1,458 raw and 1,191 guarded predictions, under/equal/over images 67/88/45. Prefix sorted: 1,099/872 and 87/75/38. Ledger pre-merge: 1,377/1,086 and 81/83/36. Ledger post-merge: 1,326/1,070 and 78/85/37. Gaussian RPS: 1,399/1,128 and 75/88/37.
- **GRPS-004 — no Gaussian-plus-ranked-score superiority claim survives the gates.** The reports provide comparable descriptive counts, not a valid population metric or a production recommendation. Differences may reflect checkpoint, prompt, continuation, guard, and artifact-health changes.

## Negative controls and validity guards

- Do not compare first-200 guarded debug-F1ish values as official benchmark scores.
- Keep Gate 0, Gate 0b, metric-family, raw/guarded, and token-health fields adjacent to every headline.
- Treat extra tokens after the first `<|im_end|>` as a continuation/termination warning, not as evidence of improved recall.
- The pure-CE Gate 0 failure blocks it from serving as a clean comparator even though its marker health is better.

## Not claimed

This capsule does not claim that the Gaussian-plus-ranked-score objective improves F1 score, termination, duplication, or training outcomes; does not authorize production training; and does not convert first-200 guarded diagnostics into accepted validation evidence.

## Limitations

Runs use different checkpoints, training recipes, artifact roots, and sometimes different GPU/config wrappers. The slice is only 200 images. Guarded prediction counts are not GT-matched population metrics, and token traces can contain post-end material. No cross-run statistical test or accepted val200 artifact is present in this round.

## Continuation seeds

1. Re-run all variants with one canonical inference/evaluation contract and the same artifact gates.
2. Report raw and guarded predictions, end-marker health, and accepted metrics in one manifest before comparing objectives.
3. Investigate post-end continuation separately from coordinate/objective effects; require a clean Gate 0/0b result before promotion.

## Exact recovery snapshot handles

- `816fbb0a1:docs/history/output-markdown-union/2026-07-17/snapshots/c7a94ae52323/outputs/analysis/gaussian_rps_mechanistic_round/phase1_object_rollout/gaussian_rps_ckpt900_val200/rollout_anatomy/report.md`
- `816fbb0a1:docs/history/output-markdown-union/2026-07-17/snapshots/bd571a7da717/outputs/analysis/gaussian_rps_mechanistic_round/phase1_object_rollout/ledger_postmerge_ckpt928_val200/rollout_anatomy/report.md`
- `816fbb0a1:docs/history/output-markdown-union/2026-07-17/snapshots/9ea3e0371cd7/outputs/analysis/gaussian_rps_mechanistic_round/phase1_object_rollout/ledger_premerge_ckpt928_val200/rollout_anatomy/report.md`
- `816fbb0a1:docs/history/output-markdown-union/2026-07-17/snapshots/32f0720feeb4/outputs/analysis/gaussian_rps_mechanistic_round/phase1_object_rollout/prefix_denoise_sorted_ckpt908_val200/rollout_anatomy/report.md`
- `816fbb0a1:docs/history/output-markdown-union/2026-07-17/snapshots/31c72f6e9100/outputs/analysis/gaussian_rps_mechanistic_round/phase1_object_rollout/pure_ce_sorted_ckpt928_val200_len12000/rollout_anatomy/report.md`
