---
title: Stage-2 Rollout Failure Registry
type: investigation
role: historical-registry
authority: non_normative_research
status: historical-synthesis
updated: 2026-07-17
---

# Stage-2 Rollout Failure Registry

This registry preserves bounded Stage-2 rollout evidence for probe selection. It
does not define the active Stage-2 runtime. For current configuration and launch
ownership, follow [`docs/training/STAGE2_RUNBOOK.md`](../../../docs/training/STAGE2_RUNBOOK.md)
and the stable OpenSpec contracts it links. The entries below are historical
measurements, failure lessons, or artifact-trust warnings; old `stage2_ab`
configuration names are provenance handles only.

## Reading rules

- A result is comparable only within its declared checkpoint, dataset slice,
  decode identity, evaluator, and `comparable_group`.
- Invalid, empty, truncated, and duplicate outputs remain in diagnostic
  denominators. Parseable-only summaries are not launch evidence.
- `v2_bboxfix` is the trusted root for the crowded small-object deep-dive;
  pre-fix rows are superseded because the diagnostics parser misread standard
  four-coordinate boxes.
- Oracle-K and stochastic rollout results measure recoverability or support,
  not a deployable inference policy.

## Registry

| ID | Claim or lesson | Scope and artifact handle | Evidence and denominator | Currentity / boundary | Comparable group |
|---|---|---|---|---|---|
| S2-001 | Channel-A iteration appeared to improve geometry stability, but the old `n_softctx_iter` arm was configuration-uncertain. | Historical `stage2_ab` Channel-A; `progress/benchmarks/2026-02-01_stage2_channel_a_infer_eval.md`; 100-row rerun and an earlier 1,000-row run. | Run A had invalid geometry `5` vs `38` (intended iter≈2 vs 1); Run B had `empty_pred=1` in each arm and invalid geometry `0` vs `2`. Exact duplicates were `0/522` vs `10/463`. | Historical and noisy. The note explicitly says old configs are not current entrypoints; map only through the current runbook before reuse. | `stage2_ab_channel_a_decode_rp_sweep` |
| S2-002 | Repetition penalty can reduce catastrophic exact duplication without proving better model coverage. | Same Channel-A study; `temperature=0.1`, `top_p=0.9`, `max_new_tokens=2048`, `rp=1.1`, limit 100. | `a_only` exact duplicates `0/522`; iter-1 `10/463`, worst `11` repeats. AP was directional (`0.2496` vs `0.2241`) on only 100 rows. | Historical decode-control lesson, not a global default. Preserve exact/near-duplicate definitions and sample limit. | `stage2_ab_channel_a_decode_rp_sweep` |
| S2-003 | The unlikelihood-corrected `ul_res_1024-v2` had higher Oracle-K recoverability but also long-rollout and truncation pressure. | First-200 COCO slice; `progress/benchmarks/2026-03-11_stage2_oracle_k_first200.md`; K=8, temperatures `0.2,0.5,0.8,1.0`, two seeds. | `ul_res_1024-v2` Oracle full recall `0.759`, recoverable full-FN share about `0.450`; deterministic AP `0.259` while the pre-fix `ul_res_1024` AP was `0.368`. The pre-fix UL capture bug skipped correct duplicate penalties. | Historical result with a causal confound. Never claim the v2 gain is solely architectural; retain UL bug status and truncation counters. | `stage2_oracle_k_first200_ul_fix` |
| S2-004 | Temperature `0.7` was the best balance in one rollout-temperature refinement, not a universal policy. | Checkpoint `ul_res_1024-v2-ckpt_300_merged`; first-200 COCO; `progress/benchmarks/2026-03-11_stage2_rollout_temperature_refinement.md`; 2 seeds per temperature. | Mean full F1: `0.6203,0.6295,0.5999,0.5970` for `0.6,0.7,0.8,0.9`; mean empty predictions `14.5,13.5,19.0,23.5`; high-IoU duplicate records increased modestly at `0.7`. | Checkpoint- and protocol-specific. Re-run under current backend/evaluator before treating as a recommendation. | `stage2_rollout_temperature_refine_20260311` |
| S2-005 | Small-object duplication is a prefix-conditioned local attractor, not merely sampling noise or a universal one-object-prefix rule. | Fixed checkpoint `pseudo_positive-ckpt_300_merged-v1`; `progress/diagnostics/2026-03-26_stage2_small_object_duplication_offline_synthesis.md`; broad harness + crowded deep-dive. | Broad harness: `7/12` mined duplication cases already had duplicate-like pairs at greedy temperature `0.0`; crowded `v2_bboxfix`: temp `0.0` mean same-desc local pairs `534.125` across 8 selected samples. Most one-object prefixes did not reproduce the loop. | Mechanism hypothesis supported by selected diagnostics, not population validation. Next probe should use the actual duplication-onset prefix. | `stage2_small_object_duplication_fixed_ckpt300` |
| S2-006 | The corrected `v2_bboxfix` crowded run is trustworthy only after the bbox parser fix; the earlier root is superseded. | `output/analysis/ckpt300_crowded_cuda1_v2_bboxfix/`; companion note `progress/diagnostics/2026-03-26_stage2_small_object_duplication_crowded_deep_dive.md`. | `_extract_bbox_px(...)` was corrected to accept `bbox_2d.points=[x1,y1,x2,y2]`; earlier metrics could be zero or empty. All cross-run comparisons must pin the corrected root and its manifest. | Artifact-trust lesson. This is a hard prerequisite for any follow-up, not a model-quality result. | `stage2_small_object_duplication_crowded_v2_bboxfix` |
| S2-007 | Exact duplicate, near duplicate, crowded unlabeled object, and unsupported hallucination are distinct failure classes. | Stage-2 Channel-A notes and dense-enumeration manual review; see `progress/benchmarks/2026-02-01_stage2_channel_a_infer_eval.md` and `outputs/research/qwen3-vl-dense-enumeration/.../manual-review.md`. | Exact duplicates use identical `(description,bbox)`; near duplicates use an IoU threshold; manual review shows real unlabeled objects and unsupported classes can coexist. | Reusable diagnostic taxonomy. Do not collapse all unmatched predictions into “hallucination.” | `rollout_failure_taxonomy` |

## Probe handoff

The cheapest discriminators are: (1) rerun a fixed hard-case cohort under the
current backend while recording effective kwargs; (2) score `remaining_gt` vs
duplicate candidates at the actual onset prefix; (3) retain both raw rollouts and
strictly materialized evaluation artifacts; and (4) report exact/near duplicate,
empty, invalid, truncation, and prediction-tail counters together. A result that
does not preserve these fields is `unverified`, even when AP is available.
