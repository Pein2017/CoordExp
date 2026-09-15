---
title: Matched ablation depth-2 pilot receipt
type: research-pilot-receipt
package_id: image2299-matched-sequence-optimizer-ablation-depth2-pilot-v1
unit_id: 2026-08-31-image2299-matched-sequence-optimizer-ablation
status: complete
effective_depth: 2
updated: 2026-08-31
---

# Matched ablation depth-2 pilot receipt

## Lead disposition

The research unit is **accepted** against preflight v6, four cell-v5 receipts,
and final-v1. The depth-2 pilot itself is **revise**, not a blanket retain: it
reduced L0 transcript load for contract and final synthesis, but the runner
implementation crossed several silent-correctness seams and still required
direct L0 repair before the production-shaped preflight could freeze.

## Package accounting

All five L1 packages materialized two independent L2 outputs, so each records
`effective_depth: 2`. Raw L2 transcripts stayed behind their handles; L0 read
zero of them.

| L1 package handle | Lead disposition | Materialized L2 outputs / route evidence | L1 correction |
|---|---|---|---|
| `/root/matched_ablation_contract_lead` | accepted | two independent contract outputs; accepted unit and implementation plan | accepted packet; no L0 correction queue |
| `/root/matched_ablation_implementation_lead` | rework | runner core: Terra/medium; invariant tests: Luna/high | one bundled correction, but production-shaped replay still found blocking defects |
| `/root/matched_ablation_repair_lead` | rework then superseded by execution gate | runner/docs: Terra/medium; regression tests: Luna/high | one bundled correction; later L0 replay still found identity/runtime-gate defects |
| `/root/matched_ablation_execution_gate_lead` | accepted after lead replay | runner execution fixes: Terra/medium; finalizer regressions: Luna/high | one bundled correction; 33 tests in its packet, then final L0 fixture correction |
| `/root/matched_ablation_synthesis_lead` | accepted | artifact audit: Terra/medium; interpretation audit: Luna/high | zero; rehash and claim audit passed |

The contract package's exact L2 route labels were not repeated in its compact
L0 packet and are therefore not reconstructed here. No Sol escalation occurred.

## Accepted verifier

- Rehashed preflight v6, all four v5 cell receipts, and final v1 to the six
  authority hashes in [results](results.md).
- Replayed the final runner/test verification: Python compile, Ruff, 34 focused
  tests, binding check, and `git diff --check`.
- Independently checked the four-cell truth table: M+QP alone has exact warm/
  fresh-cold 41-owner success; both CE panels complete 32 scheduled checks with
  no warm success; G+QP is feasible but cap-gated at
  `27.98494582667317 > 1.125`.
- Confirmed the completed 46/46 artifact is an immutable external boundary and
  is not an ablation input or promoted payload.

## L0 intervention and drift

L0 made seven direct correction groups across the owned runner/test surfaces:
prompt/image/mainline and resource/cold-reference gates; same-owner-set debt
classification; G-basis reorthogonalization; per-cell identity-copy isolation;
the QP full-vocabulary runtime gate; warm/cold evaluator labeling; and matching
test-fixture updates. These are correction groups, not a count of shell or patch
calls. The implementation and repair L1 packets therefore did not, by
themselves, reduce L0 intervention.

No role or write-surface drift affected acceptance: L2 work stayed within its
owned code, tests, or read-only audit surfaces; the final synthesis changed
only this unit's records, its one experiment-index row, and the relevant
compass paragraph. No agent mutated model outputs, the 46/46 mainline, Notion,
or Git history during synthesis.

## Critical path and usage coverage

The four cell receipts cover `3971.590` aggregate cell-seconds; the largest
single-cell wall time is M+CE at about `1345.19 s`. A root end-to-end pilot
timer and agent token/priced-usage ledger were not captured, so no latency,
cost, token-saving, or model-superiority claim is made. GPU/model resource
admission and observed bounds are recorded in the preflight and cell receipts.

## Benefit, overhead, and recommendation

**Observed benefit:** depth-2 contract and synthesis packages compressed two
independent outputs into bounded packets; the final artifact and interpretation
audits found no discrepancy without another model run.

**Observed overhead/failure:** implementation decomposition split coupled
identity, numerical-null, runtime-gate, and cold-parity behavior across workers.
Five pre-final technical correction generations (preflight v1--v5 and cell
v1--v4 lineage) plus seven L0 correction groups were required before v6/v5.

**Recommendation: revise.** Retain effective depth 2 for independent contract,
fixture/audit, and final-synthesis surfaces. For the next GPU runner, keep one
L1 owner across implementation through a single production-shaped vertical
preflight, and use L2 only for a disjoint invariant fixture and read-only
artifact audit. There is no matched depth-1 control, so this pilot makes no
savings claim.
