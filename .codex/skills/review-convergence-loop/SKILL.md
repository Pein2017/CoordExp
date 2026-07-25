---
name: review-convergence-loop
description: Iterate an explicitly review-gated CoordExp task through work, independent judgment, disposition, revision, and a bounded stop state.
---

# Review Convergence Loop

Run a **convergence loop** only when the user asks for iterative review,
subagents, or review-gated work. One-shot audits use `audit-review`; ordinary
implementation uses the relevant implementation skill.

## Loop

1. **Bind the loop.**
   - State objective, mutation boundary, authority surfaces, decision at stake,
     evidence that could change it, and stop condition.
   - Complete when every participant can tell what is in scope and what decision
     the loop must produce.

2. **Produce one current artifact.**
   - Inspect or create the smallest artifact appropriate to the authorized mode:
     research or design, guidance or specification, implementation, audit or
     launch, or packaging.
   - Complete when one version and owner are identifiable.

3. **Dispatch only independent judgment surfaces.**
   - Use subagents only when requested or permitted by the active workflow.
   - Give each lane exact evidence, one owned risk, mutation boundary, decision
     impact, and stop condition. Select models from the current model-routing
     guidance.
   - Use one reviewer per decision surface unless a real unresolved
     contradiction justifies another.
   - Complete when every required lane has returned evidence or is explicitly
     unresolved.

4. **Disposition findings.**
   - Use `audit-review` for severity and `fix`, `narrow`, `drop`, `probe`, or
     `needs user decision`.
   - Reject duplicates and findings without evidence. Do not patch through a
     research-semantic fork.
   - Complete when every material finding has one accepted disposition.

5. **Revise and verify.**
   - Change only authorized surfaces. Obtain a discriminator before acting on
     `probe`; revise scope or claim for `narrow`/`drop`.
   - Run the smallest verification that reaches the changed behavior.
   - Complete when accepted fixes are verified and the reviewed artifact version
     is explicit.

6. **Stop on decision convergence.**
   - Close a surface when a review produces no new accepted P0/P1 and no decision
     change. Further review requires changed evidence or a localized unresolved
     contradiction.
   - Finish as one of: `ready for user approval`, `approved to implement`,
     `implemented and verified`, `hold`, `probe required`,
     `needs user decision`, or `narrowed/dropped`.
   - Complete when the stop state, residual risk, and next owner are explicit.

Convergence means all conclusion-threatening findings have dispositions; it
does not mean every suggestion was implemented. A pilot may converge with
documented limitations and promotion blockers.

## Verification And Report

Use the checks owned by the changed artifact or its narrower skill. For skill
changes, run the skill validator, parse all changed UI metadata, and
forward-test one positive trigger plus one near-miss.

Report the reviewed version, lanes used, accepted and rejected findings,
revisions, verification, unresolved gates, and exact stop state.
