# Physical evaluation: what is being counted?

**Question:** are we seeing real owner gains/losses, annotation-relative matching changes, category/extent changes, or a changed review policy? A dataset annotation and a physical instance are related but not interchangeable.

## Evidence chain

[Human-audited rare-object genealogy](../../docs/history/research-records/2026-09-15/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-16-human-audited-rare-object-trajectory-genealogy/results.md) separated semantic support from part-sized, multi-instance and axis-wise box failures. [FP visual distribution](../../docs/history/research-records/2026-09-15/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-09-fp-visual-distribution/results.md) later separated strict repetition, visible objects lacking current GT coverage, class/extent errors and unresolved cases. These sampled audits are not an exhaustive physical census.

The [blind physical accounting](../../docs/history/research-records/2026-09-15/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-14-label-vs-compilation/physical/result.md) is a decisive measurement counterexample: annotation-relative F1 could improve while reviewed atomic-owner presence decreased. Many lost old predictions were GT50-unmatched, but that status alone does not prove that the instance had no annotation. Threshold, extent, class and assignment also matter. A union-of-predictions audit cannot see instances all compared models missed.

The [frozen supply audit](../../docs/history/research-records/2026-09-15/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-14-label-vs-compilation/supply/result.md) records nomination, candidate-successor and admission funnels. A no-nomination image or unknown-neutral HOLD group is not a missing-label count. Evaluation bias, information available to today's acquisition, and the causal effect of historical SFT annotation omissions are separate questions.

## Evaluation identities must remain explicit

The [accepted fourth-fit report](../experiments/2026-09-14-training-set-completion-curriculum/results.md) inherited class-agnostic one-to-one IoU≥0.5 matches and visually reviewed only residual unmatched valid predictions. Earlier stricter matched-row visual judgments remain historical observations; they do not retroactively veto that experiment's inherited matches. This is an experiment-specific convention, not a timeless physical truth rule.

The [fresh128 accepted result](../experiments/2026-09-17-readout-norm-fresh128/results.md) also uses class-agnostic one-to-one IoU≥0.5, whereas its human comparison export uses class-aware matching. Keep both named. The renderer's same-category pixel-IoU≥.30 pair count includes matched/matched pairs; the study's strict counter counts each later valid row once against any earlier bin-IoU>.95 box irrespective of category. Neither count certifies repeated physical identity. [Human-review discussion and CPU census](../experiments/2026-09-17-readout-norm-fresh128/human-review-duplication-discussion.md) preserve the definitions, examples and raw annotation context.

Keep the frozen training population and the later all-known supplemental population separately versioned. New discoveries do not retroactively become training targets. One predicted row credits at most one atomic owner; group boxes remain separate. Unknown class does not necessarily mean unknown physical identity, and a physical match does not establish category correctness.

Confirmed false objects, repeats, wrong category, wrong extent, malformed/invalid rows and unknown support are distinct debts. Unknown rows stay neither automatic positives nor negatives. They also limit any claim of physical-zero error; silence or a favorable aggregate is not exhaustive verification.

The2026-09-16 user clarification permits dense-scene group annotations when the
unit is explicitly a group. Keep individual owners, groups, body/object parts and
unresolved granularity separate. A valid individual box may contain neighbors;
overlap alone is not cross-owner failure. A group box may earn declared group
coverage, but cannot silently credit an unverified number of atomic owners.
Mixed annotation granularity is a measurement/supervision condition to record,
not a reason to tighten every box until existing valid owners become negatives.
The hand-sized person rejection remains an identity/part-as-whole example.

## Human evidence and review priority

The user's2026-09-17 review finds many real objects among nonduplicate annotation-unmatched predictions. This is useful evidence for protecting credible unlabeled owners, not a measured population rate or blanket label admission. Weakly visible objects, part/group ambiguity and reference-IoU misses remain separate from clear false objects. Judge at the actual unpainted model-input resolution; overlay strokes can obscure tiny evidence. No visual evidence sufficient for a decision means HOLD, not an invented absence or positive.

Original annotations remain intact. Any future visibility/granularity exclusions need a versioned symmetric evaluation, not raw GT deletion or retrospective score improvement. In image542582, a raw traffic-light crowd annotation is absent from the ordinary processed bank; this explains a missing evaluation context, not the truth of every unmatched prediction. A tiny box, a clipped boundary or a dense overlapping group is not automatically an error.

For the proposed recurrence study, review a fixed small set of changed owner clusters and suppression conflicts, including possible losses of credible unlabeled owners. Do not make exhaustive raw-proposal adjudication or teacher completion the gate to another bounded inference question. The previously stopped selective review still supplies no population physical-recall estimate.

## Reopening condition

User ruling2026-09-16: future unmatched diagnosis follows the project-wide
[TIDE-aligned review vocabulary](../../docs/eval/UNMATCHED_REVIEW.md). Co-DETR is
the preferred primary proxy, with no default VLM judge; unresolved instances go
to lead/subagent review. Detector agreement supports nomination, not automatic GT
or training admission. The
[detector-only retained-output diagnostic and user adjudication](../experiments/2026-09-16-codetr-only-review-proxy/results.md)
is closed. Crop/context detector support may assist selective review, but new
calibration remains unproven and is not the next research gate.

When a proposed result changes its matching/review rule, apply it symmetrically to the intended paired raw outputs or keep the result non-comparable. Preserve the previous evaluation and publish a new evaluation identity. A model judge remains a screening instrument until the intended error/admission boundary is independently tested; [the automated evaluator pilot](../../docs/history/research-records/2026-09-15/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-10-autonomous-unmatched-evaluator/results.md) did not create an oracle or a hard reward authority.

Do not move confirmation images into training, infer hallucination from FP alone, or remove uncertain owners to make a stage pass. The current user's latest explicit task rule takes precedence over an earlier scientific convention, without silently rewriting the earlier experiment's meaning.
