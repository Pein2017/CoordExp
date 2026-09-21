# Physical evaluation: what is being counted?

**Question:** are we seeing real owner gains/losses, annotation-relative matching changes, category/extent changes, or a changed review policy? A dataset annotation and a physical instance are related but not interchangeable.

## Evidence chain

[Human-audited rare-object genealogy](../../docs/history/research-records/2026-09-15/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-16-human-audited-rare-object-trajectory-genealogy/results.md) separated semantic support from part-sized, multi-instance and axis-wise box failures. [FP visual distribution](../../docs/history/research-records/2026-09-15/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-09-fp-visual-distribution/results.md) later separated strict repetition, visible objects lacking current GT coverage, class/extent errors and unresolved cases. These sampled audits are not an exhaustive physical census.

The [blind physical accounting](../../docs/history/research-records/2026-09-15/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-14-label-vs-compilation/physical/result.md) is a decisive measurement counterexample: annotation-relative F1 could improve while reviewed atomic-owner presence decreased. Many lost old predictions were GT50-unmatched, but that status alone does not prove that the instance had no annotation. Threshold, extent, class and assignment also matter. A union-of-predictions audit cannot see instances all compared models missed.

The [frozen supply audit](../../docs/history/research-records/2026-09-15/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-14-label-vs-compilation/supply/result.md) records nomination, candidate-successor and admission funnels. A no-nomination image or unknown-neutral HOLD group is not a missing-label count. Evaluation bias, information available to today's acquisition, and the causal effect of historical SFT annotation omissions are separate questions.

## Current rule differs from older physical summaries

The [accepted fourth-fit report](../experiments/2026-09-14-training-set-completion-curriculum/results.md) explicitly inherits class-agnostic one-to-one IoU≥0.5 matches and visually reviews only residual unmatched valid predictions. Previous stricter matched-row visual judgments are preserved as historical observations but do not veto these inherited matches. Do not compare the resulting total to an earlier all-row strict-extent total as if the estimator were unchanged.

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

## Reopening condition

User ruling2026-09-16: future unmatched diagnosis follows the project-wide
[TIDE-aligned review vocabulary](../../docs/eval/UNMATCHED_REVIEW.md). Co-DETR is
the preferred primary proxy, with no default VLM judge; unresolved instances go
to lead/subagent review. Detector agreement supports nomination, not automatic GT
or training admission. The
[detector-only retained-output diagnostic](../experiments/2026-09-16-codetr-only-review-proxy/unit.md)
examines the nearest existing evidence before proposing new calibration.

When a proposed result changes its matching/review rule, apply it symmetrically to the intended paired raw outputs or keep the result non-comparable. Preserve the previous evaluation and publish a new evaluation identity. A model judge remains a screening instrument until the intended error/admission boundary is independently tested; [the automated evaluator pilot](../../docs/history/research-records/2026-09-15/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-10-autonomous-unmatched-evaluator/results.md) did not create an oracle or a hard reward authority.

Do not move confirmation images into training, infer hallucination from FP alone, or remove uncertain owners to make a stage pass. The current user's latest explicit task rule takes precedence over an earlier scientific convention, without silently rewriting the earlier experiment's meaning.
