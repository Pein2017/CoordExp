---
name: audit-review
description: Audit read-only CoordExp runs, artifacts, research claims, launch or promotion gates, compatibility decisions, and user-owned semantic risks when current evidence must support a bounded decision. Use ordinary code review for generic diffs.
---

# Audit Review

Run a **decision audit**: reconstruct the claim from current evidence, expose
conclusion-changing risk, and return a bounded verdict. Do not implement fixes
unless the user separately authorizes implementation.

## Select The Branch

- **Run or artifact**: start at the exact artifact root and bind conclusions to
  its checkpoint, config, data scope, runtime, and metric-bearing status.
- **Claim validity**: name the decision-owning outcome, intervention or proxy,
  comparison, transfer assumption, preservation risk, and falsification gap.
- **Launch, promotion, or compatibility gate**: decide `approve`, `hold`,
  `reject`, `rerun gate`, `narrow`, or `needs user decision`.
- **Implementation versus vertical contract**: inspect code or a fixed diff only
  as evidence for one of the decisions above; route an ordinary engineering
  review to the platform review workflow.

Use `model-diagnosis` for an existing behavioral symptom,
`model-innovation-risk-audit` for a silent pre-launch contract risk, and the
owning code/config/runtime surface for a reproducible engineering failure.

## Audit

1. **Pin the decision and evidence.**
   - Record the question, snapshot or artifact identity, evidence scope, prior
     finding ledger, and stop condition.
   - Complete when another reviewer could inspect the same fixed point.

2. **Resolve authority and originating intent.**
   - Start with the user's brief, then follow the repository's current authority
     index to the smallest owning guidance, stable contract, config, test, code,
     and artifact set needed.
   - Treat handoffs, reviews, memories, and historical notes as provenance.
   - Complete when every requirement used in the audit has an owner.

3. **Trace only conclusion-changing risks.**
   - Check semantic alignment across touched data, geometry, ordering, prompts,
     tokens, forward behavior, objectives, metrics, runtime, and artifacts.
   - Verify upstream behavior when it owns the claim.
   - For an exploratory pilot, block only defects that can select the wrong
     condition, corrupt meaning-bearing alignment, change more than the declared
     factor, lose attribution, or make the primary observation uninterpretable.
   - Complete when each plausible P0/P1 has evidence or a named discriminator.

4. **Decide before prescribing.**
   - Classify each material finding as:
     - `fix`: intended direction remains valid;
     - `narrow`: reduce claim or supported scope;
     - `drop`: stop the path as framed;
     - `probe`: obtain missing discriminating evidence;
     - `needs user decision`: the fork changes user-owned meaning, cost,
       compatibility, publication, or irreversible behavior.
   - Before converting a research-semantic discrepancy into `fix`, compare it
     with the originating question. If the proposed repair changes the cohort,
     predicate, estimand, control, claim, or stop rule, classify it as `narrow`,
     `probe`, or `needs user decision` unless that change is already authorized.
   - Complete when every P0/P1 has one disposition and owner.

5. **Validate proportionately.**
   - Prefer targeted tests, artifact checks, installed-runtime probes, and a
     representative smoke over broad reruns.
   - If validation is unavailable or out of scope, state the exact missing check
     and the signal that would change the verdict.
   - Complete when the verdict is supported or explicitly bounded by skipped
     evidence.

## Re-Audit Boundary

Audit one fixed artifact version at a time. In an explicitly authorized outer
revision workflow, let the lead disposition findings and let the owning
implementation path apply accepted changes. Re-audit only those findings and
materially changed evidence; stop when no new accepted P0/P1 changes the
decision. The audit verdict never grants implementation or launch authority.

Freeze the target hash and finding ledger before the first independent review.
Group findings by violated invariant rather than by individual exploit. After
one correction bundle, recheck the accepted findings and changed evidence once.
If another bypass of the same invariant appears, return one systemic choke-point
finding to the implementation owner instead of opening an audit-of-audit loop.
Do not elevate arbitrary copy, proxy, monkey-patch, or pickle tampering to P1
without a reachable production or durable-evidence path.

Finish as `ready for user approval`, `hold`, `reject`, `rerun gate`, `probe
required`, `needs user decision`, or `narrowed/dropped`, with residual risk and
the next owner explicit.

## Report

Lead with severity-ranked findings:

- `P0`: likely invalidates correctness, reproducibility, or the claimed result;
- `P1`: substantial supported-workflow, contract, or interpretation risk;
- `P2`: maintainability or coverage risk with a plausible failure path.

Each finding includes an evidence handle, impact, disposition, smallest next
action, and verification. Then report confirmed OK checks, verdict, skipped
checks, and residual risk.

If no material finding remains, say so and name the residual risk rather than
manufacturing reassurance work.

## Conditional References

Load only for the active branch:

- [pipeline-checklist.md](references/pipeline-checklist.md) for end-to-end
  contract tracing;
- [governance-claim-checks.md](references/governance-claim-checks.md) for
  OpenSpec or claim-governance questions;
- [report-template.md](references/report-template.md) when the user requests a
  standalone report.
