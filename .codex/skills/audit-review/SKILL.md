---
name: audit-review
description: Audit CoordExp run/artifact evidence, scientific claims or launch/promotion fidelity; not ordinary generic code review.
---

# CoordExp Decision Audit

This skill owns the common decision-audit procedure. Domain skills add a lens,
not another approval. Authority and general review limits remain in the
[user contract](../../AGENTS.md#acceptance-and-review).

## Bind the decision

Start from the exact run, candidate or claim. Identify its question, checkpoint,
effective config, inputs, runtime, evidence scope and acceptance/stop boundary.
Use a Git/artifact version; add content binding only at an identity-sensitive
boundary. Follow the originating brief and smallest live source/spec/research
owner, not a historical finding or handoff as a new requirement.

Choose only the needed lens:

- Run or artifact validity: [pipeline checklist](references/pipeline-checklist.md).
- Scientific mechanism fidelity before costly work or promotion:
  [risk audit](../model-innovation-risk-audit/SKILL.md).
- Claim/OpenSpec consistency: [governance checks](references/governance-claim-checks.md).
- Existing abnormal model behavior instead of a decision audit: `model-diagnosis`.

## Falsify and disposition

Trace only risks that could change the declared decision: selecting the wrong
condition, semantic misalignment, undeclared factors, invalid population or
attribution, or an uninterpretable primary observation. Resolve the actual
config-to-execution-to-artifact consumer, including installed upstream behavior
when it owns the result. Use the smallest permitted counterexample or existing
receipt; missing evidence blocks only a required gate or dependent claim.

Classify each material finding as `fix`, `narrow`, `drop`, `probe` or
`needs user decision`. A proposed repair changing cohort, predicate, estimand,
control, claim or stop rule is not an ordinary fix without the owner's approval.
Separate demonstrated defects from unproven requirements. Optional hardening
and P2 observations are not quotas or default blockers.

## Re-audit boundary

Keep one finding ledger per phase/target/risk. Bundle accepted corrections by
invariant and recheck those findings plus materially changed evidence. Another
bypass of the same invariant calls for one shared fix, not an audit-of-audit loop.
Arbitrary object-copy, monkey-patch or pickle tampering matters only when reachable
in the declared runtime or across a durable evidence boundary. A recommendation
to promote is not implementation, launch or publication permission.

Report decision-changing findings with evidence handle, impact, disposition,
next owner and smallest verification. Then give the bounded verdict, skipped
checks and residual uncertainty. Use [report template](references/report-template.md)
only for a requested standalone report. Stop when no unresolved material finding
changes acceptance; do not create a second review because the skill name changed.
