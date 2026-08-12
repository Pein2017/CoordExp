# Wave 1 focused baseline — Task 2

- UTC recorded: `2026-08-12`; CWD:
  `/data/CoordExp/.worktrees/CoordExp-swift`; branch: `coordexp-swift`.
- Implementation source remains bound to
  `71dab9772983a4680dfcea742aee949c79960560`: its descendant at execution time
  changes only the Task 1 OpenSpec matrix, receipt, and plan.
- The final command was run in one isolated process group.  Earlier overlapping
  launches caused by a terminal-wrapper early return were terminated before a
  terminal result and are excluded from this receipt.

## Executed command

```bash
cd /data/CoordExp/.worktrees/CoordExp-swift
conda run -n ms pytest -q \
  tests/config/test_train_config.py \
  tests/artifacts/test_checkpoint_payload_identity.py \
  tests/artifacts/test_checkpoint_writer.py \
  tests/artifacts/test_training_state.py \
  tests/artifacts/test_run_artifacts.py \
  tests/artifacts/test_provenance.py \
  tests/runtime/test_train_runtime.py \
  tests/training/test_exact_resume.py \
  tests/training/test_pipeline_exact_resume.py \
  tests/training/test_pipeline_cache_preflight.py \
  tests/training/test_pack_cache.py \
  tests/training/test_pack_cache_determinant_registry.py \
  tests/training/test_prepare_train_cache_cli.py \
  tests/inference/test_pipeline.py \
  tests/inference/test_artifacts.py \
  tests/adapters/test_inference_reload_status.py
```

Result: exit code `0`; `774 passed, 6 warnings in 104.93s (0:01:44)`; no
failures and no skips.

The six warnings are `multiprocessing.popen_fork` deprecation warnings emitted
by the named Gloo failure-control tests in
`tests/training/test_exact_resume.py`.  They are warnings, not selection or
scenario failures.  This CPU/unit/control-plane suite does not establish GPU,
production-launch, distributed interruption, or matched first-post-resume
update behavior.

## Row-by-row decision

- Every existing `accepted` row retains a current source owner, executes its
  named focused test module in this command, and is supported only within its
  stated claim boundary.
- The matrix's `gap` rows have named owners and bounded qualification needs,
  but this suite does not add the missing scenario-specific evidence.  They
  remain `gap`; none is a demonstrated production-source failure.
- There are no `remove` rows and no unexpected execution result to classify.

Decision: **continue with qualification**.  The stop/re-plan rule does not
permit a `src/` change because no gap has demonstrated one.  It also does not
permit entering Task 3: retained qualification gaps must be closed by their
own planned waves before an execution-promotion claim.
