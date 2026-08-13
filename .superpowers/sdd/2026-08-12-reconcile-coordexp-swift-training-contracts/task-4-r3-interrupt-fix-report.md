# Task 4 R3 parent-interruption fix report

Status: `DONE_WITH_CONCERNS`

## RED

- Command: `conda run -n ms pytest -q tests/training/test_reconcile_exact_resume_probe.py`
- Result: exit 1; `68 passed, 7 failed in 26.44s`.
- Expected failures: the probe lacked `_authenticate_parent_step_one`,
  `_terminate_process_group`, the `interrupt_parent` success-resumed seam, and
  the required process-group lifecycle behavior.

## GREEN and regressions

- `conda run -n ms pytest -q tests/training/test_reconcile_exact_resume_probe.py`
  -> `76 passed in 26.96s`.
- `conda run -n ms pytest -q tests/training/test_reconcile_exact_resume_probe.py tests/training/test_exact_resume.py tests/training/test_pipeline_exact_resume.py`
  -> `125 passed`.
- `openspec validate reconcile-coordexp-swift-training-contracts --strict`
  -> `Change 'reconcile-coordexp-swift-training-contracts' is valid`.
- `git diff --check` and `git diff --cached --check` -> exit 0.

## Files changed

- `scripts/probes/coordexp_swift/reconcile_exact_resume_probe.py`
- `tests/training/test_reconcile_exact_resume_probe.py`
- `openspec/changes/reconcile-coordexp-swift-training-contracts/tasks.md`
- `docs/superpowers/plans/2026-08-12-reconcile-coordexp-swift-training-contracts.md`
- this report

Implementation commit: `85afd3f2806122ef77619ba4cfef73fba1452917`.

## Behavior and claim boundary

`success-resumed` now starts the existing real two-rank torchrun parent argv in
its own session/process group, waits for and authenticates the production
step-1 manifest and latest completed RunWriter event, terminates and reaps the
full group with bounded TERM/KILL handling, re-authenticates unchanged durable
one-step progress, and only then launches the child once. Its receipt records
the controlled interruption, identities, progress, PID/PGID, signals,
returncode, durations, and bounded output tails. Production admission and all
files under `src/` are unchanged.

This is qualification-controller and model-free test evidence only. It does
not qualify exact resume, authorize an R3 packet, prove a GPU/model launch, or
check OpenSpec tasks 3.1-3.7.

## Self-review

- Confirmed stale step 1 is rejected through the production admission
  interface after a real authoritative step-2 RunWriter event.
- Confirmed child call count remains zero for timeout, early exit, mutation,
  and stale boundary failures.
- Confirmed TERM timeout escalates to KILL and real subprocess cleanup checks
  every non-zombie process-group member rather than only the launcher.
- Confirmed every post-`Popen` exception path attempts bounded group cleanup.
- Confirmed parent/child configs remain resume-compatible and no production
  admission behavior was weakened.
- Confirmed immutable attempt-1/attempt-2 receipts and tasks 3.1-3.7 were not
  changed.

## Concern

No GPU, model, cache preparation, qualification command, or R3 packet was
executed. The next target-bound launch still requires a newly frozen packet,
independent pre-cost closure, and fresh user authorization.
