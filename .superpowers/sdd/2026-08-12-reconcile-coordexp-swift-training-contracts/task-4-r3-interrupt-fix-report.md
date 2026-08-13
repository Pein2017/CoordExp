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

---

# Round 1/5 review fix

Status: `DONE`

## Findings addressed

1. Removed the post-`Popen` `os.getpgid(pid)` race. Because
   `start_new_session=True` makes the launched process the new session and
   process-group leader before exec, the controller now retains the
   deterministic intended `pgid == pid` and can clean same-group descendants
   even after the leader exits.
2. A parent is now a controlled interruption only when SIGTERM was actually
   sent and its launcher did not exit normally with return code 0. Both the
   real controller and the `success_resumed` receipt seam fail closed on this
   invariant before child launch or success-receipt publication.

## RED

- Command: `conda run -n ms pytest -q tests/training/test_reconcile_exact_resume_probe.py -k 'leader_exits_before_getpgid or normal_exit_after_admission'`
- Result: exit 1; `0 passed, 2 failed`.
- Leader-exit race: raw `ProcessLookupError` escaped from the old
  `os.getpgid(pid)` call after a real leader spawned a 60-second same-group CPU
  descendant and was reaped.
- Normal-exit race: the barrier-controlled admitted parent exited 0 before
  termination and the old controller did not raise, allowing the success path
  to continue.

## GREEN and regressions

- Targeted command above -> `2 passed`.
- `conda run -n ms pytest -q tests/training/test_reconcile_exact_resume_probe.py`
  -> `78 passed`.
- `conda run -n ms pytest -q tests/training/test_reconcile_exact_resume_probe.py tests/training/test_exact_resume.py tests/training/test_pipeline_exact_resume.py`
  -> `127 passed`.
- `openspec validate reconcile-coordexp-swift-training-contracts --strict`
  -> `Change 'reconcile-coordexp-swift-training-contracts' is valid`.
- `git diff --check` and `git diff --cached --check` -> exit 0.

## Files changed

- `scripts/probes/coordexp_swift/reconcile_exact_resume_probe.py`
- `tests/training/test_reconcile_exact_resume_probe.py`
- this appended report

Fix commit: `7fe6b57f16f8be8f1a5ee213d499d2ddcf969095`.

## Behavior and claim boundary

The real CPU leader-exit regression proves that no non-zombie member of the
intended process group survives the pre-PGID-observation race. The real CPU
barrier regression proves that an admitted parent which disappears normally
cannot be receipted as a controlled interruption, cannot launch the child, and
cannot publish `success-resumed-receipt.json`. Post-termination production
admission remains unchanged and no file under `src/` changed.

The review's `CUDA_VISIBLE_DEVICES=''` combined-suite failure was the expected
strict launcher-mapping preflight rejection for an empty device mapping. It is
not evidence for a production change and no production behavior was modified.

## Self-review and concern

- The primary `reconcile_probe.parent_exited_early` error is preserved when
  descendant cleanup succeeds; `reconcile_probe.parent_cleanup_failed` is used
  only if a real surviving-group cleanup attempt fails.
- The receipt invariant is checked before post-termination admission, child
  launch, or success receipt publication.
- No GPU, model, cache preparation, qualification command, or R3 packet ran.
- No new concern beyond the existing requirement for a fresh target-bound R3
  packet, independent pre-cost closure, and fresh user authorization.
