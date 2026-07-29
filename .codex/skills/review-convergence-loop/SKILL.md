---
name: review-convergence-loop
description: Manually run a bounded review and revision loop only when the user explicitly invokes $review-convergence-loop for a decision-important claim, launch or promotion gate, compatibility or material-cost decision, or irreversible change.
---

# Manual Review Convergence

Run only when the user explicitly invokes `$review-convergence-loop` for a
decision-important surface. Do not infer it from ordinary requests to review,
refine, iterate, or use subagents. Use `audit-review` for a one-shot audit and
the relevant implementation skill for ordinary changes.

1. Bind one current artifact version, decision, allowed mutation surface, and
   stop condition.
2. Apply `audit-review` to that fixed version. Use one reviewer per independent
   decision surface only when its judgment can change the decision; prefer a
   deterministic check or discriminator when available.
3. Let the lead disposition findings. Use the relevant implementation skill to
   apply only accepted and authorized revisions.
4. Verify the changed behavior, then use `audit-review` to recheck only accepted
   findings and materially changed evidence.
5. Stop under `audit-review`'s explicit iterative-review rule. Do not add a clean
   general review wave for reassurance.

Report the reviewed versions, accepted and rejected findings, verification,
unresolved gates, residual risk, next owner, and exact stop state.
