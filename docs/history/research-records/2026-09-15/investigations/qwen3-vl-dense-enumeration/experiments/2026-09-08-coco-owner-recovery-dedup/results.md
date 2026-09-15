---
title: Matched geometric deduplication continuation results
description: Fewer repeats and FP, but no established annotated-owner recovery improvement.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
unit_id: 2026-09-08-coco-owner-recovery-dedup
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-09-08
---

# Fixed GPU contrast completed

**Lead-accepted execution and bounded readout; no architecture promotion.**
Both arms continued Rweak64 for exactly16 updates to80, with their own natural
train256 refresh at72. Final outputs are natural/unpostprocessed, all512 images
and3759 annotated owners retained. The previously used panel is not independent
confirmation. The initial dedup failure and exact65 resume are preserved in
[execution](execution.md); no scientific updates were added.

| Natural output | TP50 | FP50 | FN50 | Precision | Recall | Later-repeat candidates |
|---|---:|---:|---:|---:|---:|---:|
| Source |2225|3037|1534|42.28%|59.19%|1572|
| Rweak64 |2310|3369|1449|40.68%|61.45%|1524|
| Control80, lambda0 |2295|3008|1464|43.28%|61.05%|1339|
| Dedup80, lambda1 |2300|2830|1459|44.83%|61.19%|1132|

Primary matched contrast, dedup minus control:19 gained owner IDs and14 lost,
**net+5**, with diagnostic paired-image bootstrap95% total-equivalent interval
**[-11,+27]** (10000 draws, seed20260908). FP decreases178 (5.92%); later-repeat
candidates decrease207 (15.46%). Pair counts fall89306→54729, while affected
images remain37→37. Greedy-removable counts1337→1131 are a separate diagnostic;
no postprocessing is applied to the owner/FP table.

**Inference:** the point estimate is consistent with lower repeat burden and
annotation-relative FP while preserving approximately the same owner recovery,
but does not establish improved recovery or statistical equivalence. Broad
output suppression/jitter remains an alternative mechanism: valid outputs fall
5303→5130, and the repeat-pair count is concentrated in a few images. Do not
infer physical completeness or hallucination reduction from annotation-relative
FP. This round does not isolate the mechanism.

Against Rweak64, treatment gains70 owners and loses80 (net−10), while control
has net−15; extra continuation alone also lowers FP/repetition. Against Source,
treatment gains179 and loses104 (net+75), control net+70. Therefore comparing
treatment only against Rweak64 would overattribute the improvement to dedup.

Diagnostics: dedup-minus-control net owners at IoU60/80 are+3/+9, bootstrap
intervals[-14,+25]/[-2,+28]. Capped images9→7; parser-dropped predictions1926→1759;
invalid valid-parser boxes1→1. These remain separate from FP and never remove
images from the denominator.

Artifacts:
- [Exact metrics, owner IDs, bootstrap, per-image counts and raw hashes](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-08-coco-owner-recovery-dedup/round-v1/results.json)
- Control80 checkpoint ID:`5d7180be6ce325e910be6ed83375a97675957f1354d5a7e3585aaf6bc15c83f1`.
- Dedup80 checkpoint ID:`03399e0633021190868c48a4e7dfad2b76f02d80952408734e590e7fa9df18e5`.
- Reproduce: `conda run -n ms python scripts/research/reduce_coco_owner_recovery.py --root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-08-coco-owner-recovery-dedup/round-v1 --output /tmp/coco-owner-recovery-readout-new.json` (output must not already exist).

**Stop rule reached:** one matched round and final evaluation complete; no
extension, sweep, confirmation, commit, cleanup or promotion performed.

## Historical CPU/design package (before GPU authorization)

**Lead-accepted CPU recount and loss prototype; training effectiveness remains
untested at that checkpoint.** The [unit](unit.md) owns the single proposed next contrast and resource
request. Historical results and their old acceptance rules were not modified.

## Observation: the new predicate changes the repetition conclusion

Same512 images /3759 annotated owners, no dropped image or cap exclusion:

| Raw natural output | TP50 | FP50 | FN50 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Source |2225|3037|1534|42.28%|59.19%|
| Rweak64 |2310|3369|1449|40.68%|61.45%|

Rweak64 recovers188 Source-missed owner IDs and loses103, **net+85**. These are
unique category-compatible one-to-one matches, not repeated hits on one owner.

| Pixel IoU strictly >.95; category/description/GT ignored | Source | Rweak64 |
|---|---:|---:|
| Unordered overlapping pairs |124214|94546|
| Later predictions eligible against any earlier prediction |1572|1524|
| Deterministic greedy-removable predictions |1570|1521|
| Affected images |35|44|
| Cross-category pairs |2|7|

Greedy-removable means retain the first valid prediction, then remove a later
prediction iff it overlaps an already retained prediction at >.95. It is
order-dependent, not an optimal/minimum removal set. The training predicate
instead compares against **all** earlier predictions, including earlier
eligible ones. Overlap chains explain the small difference between the counts.
Pairs can grow quadratically and must not be called removable-box counts.

Source's top pair contributors are image885 (50201),397303 (49787),293858 (18721).
Rweak64's are885 (50174),566923 (30145),371472 (8276). Top-five images contribute
97.94% and97.47% of pairs respectively. Top removable counts are Source337/325/193
and Rweak336/249/176 for those respective images. All remain in reported totals.

**Inference:** the old narrower GT-attributed duplicate increase cannot be
restated as an increase under the user's new geometric rule. Rweak has fewer
total eligible/removable predictions, but more affected images. This does not
make it globally more stable, nor explain away its332 additional raw FP.
The sharp historical-count difference is mainly a scope change, not evidence
that cross-category pairs themselves are common.

## Separate postprocessing diagnostic, not training acceptance

Reapply the same owner matcher AFTER greedy removal:

| Postprocessed diagnostic | TP50 | FP50 | FN50 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Source |2225|1467|1534|60.27%|59.19%|
| Rweak64 |2308|1850|1451|55.51%|61.40%|

The paired postprocessed contrast is186 recovered /103 lost, net+83. Rweak
loses two previously matched owners after removal: TP preservation is not
guaranteed by the >.95 rule. Even after this diagnostic, Rweak has383 more FP
than Source. Deduplication addresses a substantial existing pathology in BOTH
arms, but is not by itself an explanation for all incremental Rweak FP.
No postprocessed result is evidence that the training regularizer works.

FP counts cover valid unmatched predictions relative to available annotations.
Two pixel-degenerate predictions per arm are invalid and separate. Upstream
`dropped_prediction_count` is2397 Source /1695 Rweak, parser failures0; these
parser-dropped items are excluded from valid-prediction FP and pair counts,
not interpreted as known physical objects or hallucinations.

## Executed CPU loss preparation

- Reuse the existing row-token grammar and current-prefix `TrajectoryScorer`
  CE interface. New loss has no GT/category requirement, uses exact later-row
  coordinate offsets, and converts bins to rounded pixel bboxes before IoU.
  The initially assumed bin/pixel equivalence was disproved by the consumer's
  integer rounding and corrected before acceptance.
- Selected proposal: geometric-mean coordinate-confidence unlikelihood,
  normalized by all valid generated rows then all images, lambda1; see exact
  equations in the unit. It remains a sampled surrogate, not a continuous IoU
  barrier. Jitter/shortening and weak cross-image transfer remain hypotheses.
- Saved Rweak64 train256 traces provide contiguous non-pad token IDs for all
  images, with253 native stops and3 caps. Token texts exactly reconstruct every
  raw decoded string. The new loss's per-image eligibility agrees with the
  saved-pixel recount for all256 images:504 later repeats across12 images.
  Two images contribute435/504 eligible rows (86.31%); signal is concentrated.
- CPU evaluation using saved generation logprobs gives image-mean surrogate
  loss0.00264085. An initial joint-four-coordinate probability formulation gave
  0.00016805 and was replaced by the length-normalized confidence formulation
  during train-only preparation. Neither number measures current-model replay
  gradients, gradient-norm matching, or natural owner recovery after training.
- Golden tests verify strict equality exclusion, cross/empty-description
  geometry eligibility, rounded-pixel behavior, overlap chains, no-repeat zero,
  exact coordinate-only gradient positions, descent lowering repeat confidence,
  normalization, finite extremes and malformed-complete-row rejection.
  The rematching counterexample explicitly loses a TP after category-blind
  dedup, so the test cannot silently assume owner preservation.
- Final focused suite: **5 passed**. In-memory threshold and bin-space
  mutations were both rejected by the tests. Lead recomputed the full saved
  recount, compared summary and every per-image row exactly, and validated
  both arms against the frozen512-image input. An initial verification attempt
  compared opposite arm serialization order; explicit original arm order fixed
  the verifier, with no data/code/result change.

## Artifacts and reproduction

Artifact root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-08-coco-owner-recovery-dedup/`

- `cpu-recount-v1/summary.json`: input identities, exact owner gain/loss sets,
  metrics, concentration and train signal supply.
- `cpu-recount-v1/per_image.jsonl`: all-image pair/removal evidence and raw versus
  rematched counts.
- `cpu-loss-v1/saved-train-replay.json`: all256 exact-token replay checks and
  saved-logprob surrogate diagnostics.
- `cpu-loss-v1/verification.json`: final CPU acceptance receipt/code identities.

Run from the worktree, choosing a NEW output directory/file on every rerun:

```bash
conda run -n ms python -m pytest -q tests/research/test_coco_geometric_dedup_loss.py tests/research/test_recount_coco_geometric_duplicates.py
conda run -n ms python scripts/research/recount_coco_geometric_duplicates.py --source <Source-gt_vs_pred_scored.jsonl> --rweak <Rweak-gt_vs_pred_scored.jsonl> --rweak-train <Rweak-train-gt_vs_pred_scored.jsonl> --output-dir <new-directory>
conda run -n ms python research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-08-coco-owner-recovery-dedup/check_saved_replay.py <new-json-file>
```

Exact input paths are in the summary. Earlier CPU-only intermediate recount
versions were moved, not deleted, to `/tmp/coco-recount-superseded.a1vx6k` and
`/tmp/coco-recount-precompact.nWVn62`; the final paths above are the accepted
versions. No old research artifact, shared trainer, index, skill or memory was
edited; no commit/push/archive/cleanup or dependency installation occurred.

## Recommended ruling

Approve or reject **one matched repair contrast**: both branches start at
Rweak64 and continue16 updates with unchanged Rweak loss, lambda0 versus1,
refresh training generations at0/+8, then evaluate the same512-image panel.
Reuse the admitted exact initial train capture; refresh+8 separately per arm.
Request <=4 simultaneous GPUs, <=4h elapsed, <=16 GPU-hours including the new
real-boundary smoke and final evaluation. No fresh confirmation, extra dose,
coefficient sweep or automatic extension. The existing identity-bound trainer
and cold consumer still need a minimal successor integration after approval.

**STOP at the CPU/design gate.** There is no unanswered discoverable question
blocking this package; the remaining user-owned decision is the bounded GPU
round. No scientific training-effect claim has been accepted.
