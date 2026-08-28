# Verification Report: add-human13-k-union-greedy-overfit-probe

Verified: 2026-08-12.

## Summary

| Dimension | Status |
| --- | --- |
| Completeness | `25/27` tasks; `2` intentionally incomplete |
| Correctness | Implemented requirements and executed-arm scenarios covered; live census scenario incomplete |
| Coherence | Experiment-local design followed; no generic StateBank or production contract widened |

Verification evidence:

- `280` focused Human-13 tests passed across losses, manifest, collection,
  census adapters, packing/runner, live model/payload, launcher, eval, and
  analyzer.
- Ruff passed for the Human-13 source/test surface.
- `openspec validate add-human13-k-union-greedy-overfit-probe --strict` passed.
- `git diff --check` passed.
- Independent Sol-xhigh and Fable-xhigh execution audits found no P0 and no
  pooled-table mismatch. Their P1 findings were accepted as claim narrowing and
  provenance limitations in the owning review/results records.

## CRITICAL — required before archive

1. Task 5.2 is incomplete. The canonical manifest is sealed, but the live
   no-update census did not emit a valid artifact within its repair budget.
   Consequently A8-prime margin/applicability is unavailable. Either complete a
   separately authorized census successor or formally revise the change to
   close A8-prime as mechanically unavailable before archive.
2. Task 5.3 is incomplete because no valid no-update identity/runtime receipt
   exists. Training fail-closed tests exist, but the specified census receipt
   cannot be published retroactively.

## WARNING

1. A4 and A6 originally had no persisted launcher stderr. The deterministic
   CPU processor/payload path now reproduces their fail-closed errors, recorded
   in `analysis/execution-reconciliation-v1.json`, but that is posthoc evidence.
2. Eval receipts wrote an evaluation-contract digest into
   `resolved_arm_plan_sha256` rather than the actual training-plan digest. The
   immutable checkpoint/output hashes and counts remain valid; the posthoc
   crosswalk restores explicit linkage without rewriting artifacts.

## Correctness and coherence disposition

- Exact Source/K acquisition, manifest identity, duplicate-before-match owner
  ledger, frozen update contracts, no-padding panel accumulation, five executed
  arm schedules, checkpoint readback, full-panel HF readout, and analyzer
  projection are present and tested.
- A4 honored the specified atomic overlength failure instead of synthesizing a
  multi-pass objective.
- A6 honored sealed-donor validation and failed before model execution.
- A8-prime remained blocked rather than inventing a margin without the census.
- The implementation stayed experiment-local and did not claim prefix/image
  compute reuse, validation, generalization, production behavior, or a complete
  matrix winner.

## Final assessment

The bounded executed-arm observation is verified and complete. Two critical
OpenSpec tasks remain by design, so this change is **not ready for archive**.
No additional training or repair is implied by this report.
