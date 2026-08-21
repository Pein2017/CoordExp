# Wave 2 / Task 2.4 — One full CPU test-suite run (failure-set diff)

## Baseline replicated from

`openspec/changes/archive/2026-08-20-add-coordexp-swift-training-observability/receipts/command-manifest.json`
- argv: line 114 (`commands[]` entry `id: "wave5-broad-regression"`)
- env: `environment.always` / `environment.required_absent` (top of file, lines ~7-20)
- pinned observed baseline: line 250 (`amendments[] id: "amend-8-freeze-wave5" .observed.wave5-broad-regression`):
  **"2691 collected: 2565 passed / 126 skipped / 0 failed"**
- cross-reference: `tasks.md` line ~365 close-out note: "5.6 broad 2565/0/126skip with zero residue"

Baseline failure set: **EMPTY** (0 failed at HEAD `04bbc1631` per the archived close-out; task instructions independently state the baseline failure set is empty).

## Exact replicated argv + env

```
cd /data/CoordExp/.worktrees/CoordExp-swift
export PYTHONDONTWRITEBYTECODE=1
unset COORDEXP_SWIFT_PACK_CACHE_ROOT
unset COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE
unset COORDEXP_SWIFT_EVAL_REDUCTION_MODE
unset COORDEXP_SWIFT_PROFILE_SYNC_TIMINGS
unset RANK
unset LOCAL_RANK
unset WORLD_SIZE
unset MASTER_ADDR
unset MASTER_PORT

conda run -n ms pytest tests/config tests/losses tests/runtime tests/training tests/artifacts tests/eval -q
```

All nine `required_absent` env vars were confirmed already unset on this host before launch (verified with a per-var echo loop); the explicit `unset` calls in the runner script are a belt-and-suspenders match to the manifest's env contract. No RANK/LOCAL_RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT distributed env was present — this is a plain single-process CPU pytest run, matching the archived wave5-broad-regression class (CPU-only; GPU probes are separate frozen packets not in scope for 2.4).

Runner script: `openspec/changes/close-coordexp-swift-review-p1s/receipts/run-wave-2-full-suite.sh`
Detached launch: `setsid nohup bash run-wave-2-full-suite.sh > wave-2-full-suite.log 2>&1 &` (fully detached, immune to the foreground-timeout/backgrounding host trap).

## HEAD and working-tree state at run time

HEAD: `04bbc1631b0e283a2dad1be7134f82dd58ba382d` (matches the task-brief HEAD)

`git status --short` at launch:
```
 M src/config/models.py
 M src/losses/runner.py
 M src/runtime/finite_gates.py
 M src/runtime/optimizer_boundary.py
 M src/runtime/train_runtime.py
 M src/training/cache_workflow.py
 M src/training/pack_cache.py
 M src/training/reporting.py
 M tests/config/test_train_config.py
 M tests/eval/test_forward_eval.py
 M tests/training/test_cache_contract.py
 M tests/training/test_pack_cache_determinant_registry.py
 M tests/training/test_pack_cache_runtime_constructor.py
 M tests/training/test_reporting.py
 M tests/training/test_training_module_boundaries.py
?? openspec/changes/close-coordexp-swift-review-p1s/
?? src/training/micro_step_assembler.py
?? tests/losses/test_zero_eligible_collective.py
?? tests/runtime/test_fp16_scaler_contract.py
?? tests/training/test_micro_step_payload_identity.py
```

This matches the expected P1/P2 fix footprint: P1-1 new `src/training/micro_step_assembler.py` +
`tests/training/test_micro_step_payload_identity.py`; P1-2 `tests/runtime/test_fp16_scaler_contract.py`
+ `src/runtime/optimizer_boundary.py`/`src/runtime/finite_gates.py`; P1-3
`tests/losses/test_zero_eligible_collective.py` + `src/losses/runner.py`; P2 edits across
`src/config/models.py`, `src/runtime/train_runtime.py`, `src/training/cache_workflow.py`,
`src/training/pack_cache.py`, `src/training/reporting.py`, and the corresponding modified test files.

## Run outcome

- verbatim pytest tail line: `2600 passed, 126 skipped, 7 warnings in 969.22s (0:16:09)`
- EXIT code: `0`
- wall time: `973s` (0:16:13, wrapper-measured `SECONDS`; pytest's own internal timer reports `969.22s`/0:16:09 — the ~4s delta is process/conda-run startup+teardown overhead, consistent)
- start: `2026-08-21T03:44:22+00:00`, end: `2026-08-21T04:00:35+00:00`
- peak RSS: approximately **1,976,796 KiB (~1.88 GiB)**, sampled by summing RSS of all `envs/ms/bin/python3.12` processes every 20s during the run (`/usr/bin/time` is not installed on this host, confirmed absent). This is an approximate upper bound: the sampler could not distinguish the pytest worker tree from four unrelated long-idle Aug-13 `fake_backend.py` leftover processes also matching the `envs/ms` python pattern (each is small/idle, <15MB RSS, negligible contribution) and any transient sibling multiprocessing/subprocess children spawned by individual tests (e.g. `wave7_exact_resume_compare_v2.py` probes). Order-of-magnitude is reliable; exact figure is not.
- collected count: **2726** (`2600 passed + 126 skipped`) vs baseline **2691** (`2565 passed + 126 skipped`) — **+35 collected, all additional collection landed in the passed bucket, skip count identical (126) to baseline.** This matches the expected delta: the working tree adds `tests/losses/test_zero_eligible_collective.py` (7 test functions), `tests/runtime/test_fp16_scaler_contract.py` (17), `tests/training/test_micro_step_payload_identity.py` (4) = 28 wholly-new tests, plus additional test functions folded into the modified files (`tests/config/test_train_config.py`, `tests/training/test_reporting.py`, `tests/training/test_training_module_boundaries.py`, etc. — diff stats: `+128/-17` lines across the 7 modified test files). No collection anomaly: the delta is higher, not equal-or-lower, and the skip count is exactly preserved, so the historicized-executor skip set was not perturbed.
- `grep -nE "FAILED|ERROR|failed"` over the full log: **zero matches** (only the summary line's own literal substring is absent since it reads "passed... skipped... warnings", with no "failed" token at all — pytest omits the word entirely when the failed count is 0).
- warnings (7 total, 2 classes): 1x `UserWarning` (sparse CSR tensor beta-state notice, `tests/runtime/test_finite_gates.py:358`), 6x `DeprecationWarning` (`multiprocessing.popen_fork` fork-in-multithreaded-process notice, three `tests/training/test_exact_resume.py` two-rank gloo tests each firing twice). The archived baseline pinned pass/skip/fail counts but not a warning inventory, so strict newness cannot be diffed against it; both classes are runtime/library-level notices (torch sparse-tensor internals, CPython multiprocessing) with no plausible coupling to the P1/P2 diff under test.

## Failure set

**EMPTY.** Zero failed tests, zero errors, zero collection errors. Full pytest progress output (dots/`s` markers by percentage) and the summary line are captured verbatim in `wave-2-full-suite.log`.

The raw `.log` is ignored by repository policy and is not part of the commit.
At final acceptance the local file was 4,845 bytes with SHA256
`5d0d0e61dfe7abb85260aa14f1683af26bae4fdd05eb765871844df8b12434d2`;
this tracked receipt, not the ignored local file, is the portable close-out
record.

## Fresh-process replays

Not applicable — no failures to replay. The tracked flake `test_gate_ablation_creates_no_autograd_edge_into_the_objective` did not fail in this run (no occurrence; tripwire count remains at 2, unchanged, not triggered).

## Failure-set diff verdict vs empty baseline

**PASS — failure sets are identical (both empty).** Baseline (archived wave5-broad-regression, HEAD prior to this change): 0 failed. Current run (this working tree, HEAD `04bbc1631` + uncommitted P1/P2 fixes and new tests): 0 failed. The +35 collected / +35 passed delta is fully attributable to the declared new-test additions listed above and is not a signal of regression scope creep; the skip count (126) matches baseline exactly, consistent with "126 historicized executor skips expected" persisting unperturbed. This is a count-identity match, not a node-id set-identity match — `-q` output does not enumerate individual skip node ids in either this run or the archived baseline, so per-test skip-set diffing was not cheaply available; the count match is the strongest signal obtainable from both runs' recorded evidence.

## Interlude: false "process is dead" alarm from the coordinator (for the record)

At approximately 03:55 and again at 03:57 UTC (~11-13 minutes into the run), the coordinator sent two messages asserting the detached run was dead (log frozen at 145 bytes, claimed zero matching processes in `ps aux`) and ordering an immediate kill + rename of the log + foreground-batched re-execution. Both times, direct on-host verification at the moment of the message contradicted the claim:
- `ps -p 522254 -o pid,stat,time,etime` showed CPU TIME advancing across independent checks (4:45 at 03:55:34 -> 4:53 at 03:57:02) and fresh child subprocesses spawned at 03:55 (pid 581732) and 03:57 (pid 591341), both `wave7_exact_resume_compare_v2.py` invocations tied to specific in-progress tests under `tests/training`.
- The Monitor task's own RSS sampling independently showed peak RSS still climbing during the same window (1.87 GiB -> 1.98 GiB).
- The 145-byte-frozen log was explained by `conda run` (invoked without `--no-capture-output`) fully buffering the child's stdout/stderr until process exit; the wrapper script's own `EXIT=`/`WALL_SECONDS=` lines are echoed only after `conda run` returns, so they could not have appeared earlier regardless of run health.

The run was allowed to continue rather than be killed. It completed normally 3 minutes after the second alarm, at 04:00:35 UTC, with `EXIT=0` and the clean 2600/126/0 result recorded above — confirming the "dead" diagnosis was a false positive (most likely: the coordinator's own `ps aux` check missed the process due to a grep/timing mismatch, or was checking a stale/cached view; the actual host-level process was live and progressing throughout). No process was killed, no log was renamed, and the task's hard cap of exactly one full-suite invocation was preserved. This is recorded here as an anomaly for the lead's awareness, not as a defect in the suite or the P1/P2 changes under test.
