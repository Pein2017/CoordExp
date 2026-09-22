---
title: Independent Audit of the Individual-Trajectory versus Sampled-Union Object-Support Result
type: review
role: independent-audit
authority: non_normative_research
unit_id: 2026-07-21-individual-trajectory-versus-union-support-audit
topic: qwen3-vl-dense-enumeration
status: complete
updated: 2026-07-21
---

# Independent Audit

## Initial findings

An independent read-only reviewer reconstructed the route comparisons from the
unit, analyzer, rollout artifacts, and review decisions without inheriting the
main thread's conclusion. It found three issues:

1. Exact panel-wide best-trajectory identities and larger-budget union
   magnitudes were still provisional because many non-branch-bearing rows had
   not received crop adjudication.
2. A human-reviewed `uncertain` decision disappeared from the unresolved queue,
   which could make a trajectory look fully resolved even though the row could
   still alter coverage or safety.
3. The budget-root harmful-row count aggregated all seventeen trajectories and
   could be confused with the trajectory-specific harmful-row count used by the
   branch rule.

The reviewer independently confirmed the raw route-local comparisons for image
5001 with seed 21015, image 7511 with seed 21012, and image 2685 with seed
21008. It also identified a hard complementarity witness at four rows: five
images had a sampled-union owner lower bound above four, which no one four-row
trajectory can attain.

## Accepted corrections

The analyzer and tests were changed before final interpretation:

- reviewed `uncertain` rows remain represented as unresolved potential;
- every trajectory now reports coverage lower and upper bounds;
- every trajectory now reports harmful-row lower and upper bounds;
- every sampled route receives a conservative certificate only when its
  coverage lower bound exceeds greedy's coverage upper bound and its
  harmful-row upper bound does not exceed greedy's harmful-row lower bound;
- the budget aggregate is named `panel_harmful_row_count`, while the existing
  trajectory-specific field remains `harmful_row_count`; and
- positive and negative certificate tests were added.

The corrected analysis is:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-21-individual-trajectory-versus-union-support-audit/
support-audit-step4887-20260721a/analysis/reviewed-support-v3.json
```

## Final audit boundary

The evidence supports this wording:

> Three specific sampled routes satisfy conservative gain-and-safety
> inequalities, and complementary owner support exists across trajectories.

It does not support these stronger wordings:

- the analyzer formally fired the predeclared program-level Rule 1;
- every exact best-trajectory identity is final;
- every 32-row union-only magnitude is final; or
- the proposed self-imitation treatment is already effective.

The bounded self-imitation screen remains a justified next discriminator, not
a promoted training recipe.
