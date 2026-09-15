# W8 owner-bridge smoke: distributed collective choreography failure (2026-08-10)

Provenance record for OpenSpec `add-permanent-owner-bridge` task 5.1.

This record exists because the decisive logs live under the gitignored `outputs/`
tree (`.gitignore:54:/outputs/`). The decisive lines are **quoted verbatim** below so
this repository keeps a durable copy after the log tree is pruned or overwritten.

## Claim boundary (read first)

- What this record establishes: an 8-rank owner-bridge training smoke deadlocked on
  mismatched collective choreography and died; the ranks disagreed about how many
  NCCL works to enqueue inside one planned step.
- What this record does **not** establish: anything about atom construction, router
  behavior, `L_use`, AR decoding, loss values, metrics, or model quality. The run
  produced `completed_steps: 0` and a zero-byte `logging.jsonl`; **no step telemetry
  exists**, so no quality-bearing quantity was ever emitted. Any later reading of
  this failure as a modeling signal is unsupported.
- What this record explicitly does **not** claim: that the failure is repaired. The
  diagnosis section below is a hypothesis grounded in static code shape at the run's
  commit. Proof of repair belongs to current Wave-5 acceptance and its GPU smokes,
  not to this document.

## Run identity

| Field | Value |
| --- | --- |
| Run root | `outputs/smoke/coordexp_swift_owner_bridge/qwen3_vl_2b_desc_first_geo_sorted_xy_owner_bridge_stage1_dora_r16a32_llm_12000_accelerate8_ebs24_train256_val64_4epoch_warmup0p1-20260810T025027Z` |
| `run_id` | `qwen3_vl_2b_desc_first_geo_sorted_xy_owner_bridge_stage1_dora_r16a32_llm_12000_accelerate8_ebs24_train256_val64_4epoch_warmup0p1-ca563a0a5fc3` |
| `config_fingerprint` | `ca563a0a5fc36328b7767c45707631a340357f1bb58459d509865843d555e629` |
| `status` | `failed` |
| `completed_steps` | `0` (of `resolved_max_steps: 8`) |
| `runtime.world_size` | `8` |
| `created_at` | `2026-08-10T02:50:27.620691+00:00` |
| `updated_at` | `2026-08-10T03:05:11.443355+00:00` |
| `completed_at` | `null` |
| `consumed_packs` / `checkpoint_event_count` | `0` / `0` |
| `final_finite_status` / `final_optimizer_update_status` | `null` / `null` |
| Launch config | `configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_xy_owner_bridge_stage1_dora_r16a32_llm_12000_accelerate8_ebs24_train256_val64_4epoch_warmup0p1.yaml` |

Terminal error, verbatim from `run.json`:

```
RuntimeError: [/pytorch/third_party/gloo/gloo/transport/tcp/pair.cc:547] Connection closed by peer [192.168.7.7]:57000
```

## Source code identity at the failed run

The run executed the **clean** tree at `d9d9768407a9823a9c2e10d4ddd08e9321094eb2`
(`d9d976840`, "Allow long rank-skewed finite gates", `Mon Aug 10 02:48:44 2026 +0000`),
which is 1m43s before `created_at`. This is not inferred from the readiness directory
name alone — every source line number in the rank-0/rank-4 tracebacks resolves to the
matching statement in the committed blob at that commit:

| Traceback frame | Line content at `d9d9768` |
| --- | --- |
| `src/runtime/train_runtime.py:178` | `        return reduce_scalar_finite_reports(self._gather_rank_reports(report))` |
| `src/runtime/train_runtime.py:510` | `            reports = tuple(self.rank_report_gatherer(local_report))` |
| `src/training/pipeline.py:3109` | `        gathered_frames = _all_gather_cpu_bytes(` |
| `src/training/pipeline.py:2961` | `    distributed.all_gather(gathered, local, group=group)` |
| `src/training/supervised_trainer.py:576` | `                    pre_decision = self.runtime.pre_backward(` |

None of those line numbers match the current working tree, whose edits to those three
files post-date the run (`pipeline.py` 04:00:22Z, `supervised_trainer.py` 04:32:06Z,
`train_runtime.py` 05:02:15Z on 2026-08-10). So the failure is attributable to
`d9d9768` as committed, and any repair work in the dirty tree is later and untested by
this run.

## Raw observation: per-rank collective counters

Log root (gitignored):
`outputs/smoke/coordexp_swift_owner_bridge_readiness/d9d9768/w8-rank-logs/none_1ulisqjw/attempt_0/{0..7}/stderr.log`

### Ranks 0 and 4 — stopped at work 77, blocked in the Gloo `pre_backward` gather

`0/stderr.log:3-4`:

```
[rank0]:[E810 03:04:11.562447118 ProcessGroupNCCL.cpp:1794] [PG ID 0 PG GUID 0(default_pg) Rank 0] Observed flight recorder dump signal from another rank via TCPStore.
[rank0]:[E810 03:04:11.562692551 ProcessGroupNCCL.cpp:1858] [PG ID 0 PG GUID 0(default_pg) Rank 0] Received a dump signal due to a collective timeout from  rank 1 and we will try our best to dump the debug info. Last enqueued NCCL work: 77, last completed NCCL work: 77.This is most likely caused by incorrect usages of collectives, e.g., wrong sizes used across ranks, the order of collectives is not same for all ranks or the scheduled collective, for some reason, didn't run. Additionally, this can be caused by GIL deadlock or other reasons such as network errors or bugs in the communications library (e.g. NCCL), etc. 
```

`4/stderr.log:4`:

```
[rank4]:[E810 03:04:11.562661853 ProcessGroupNCCL.cpp:1858] [PG ID 0 PG GUID 0(default_pg) Rank 4] Received a dump signal due to a collective timeout from  rank 1 and we will try our best to dump the debug info. Last enqueued NCCL work: 77, last completed NCCL work: 77.This is most likely caused by incorrect usages of collectives, e.g., wrong sizes used across ranks, the order of collectives is not same for all ranks or the scheduled collective, for some reason, didn't run. Additionally, this can be caused by GIL deadlock or other reasons such as network errors or bugs in the communications library (e.g. NCCL), etc. 
```

Where those two ranks were parked — Python stack, `0/stderr.log:6-46` (rank 4 is
identical except for the `[rank4]` prefix and the Gloo peer port):

```
[rank0]: Traceback (most recent call last):
[rank0]:   File "<frozen runpy>", line 198, in _run_module_as_main
[rank0]:   File "<frozen runpy>", line 88, in _run_code
[rank0]:   File "/data/CoordExp/.worktrees/permanent-owner-bridge/src/train.py", line 33, in <module>
[rank0]:     raise SystemExit(main())
[rank0]:                      ^^^^^^
[rank0]:   File "/data/CoordExp/.worktrees/permanent-owner-bridge/src/train.py", line 26, in main
[rank0]:     result = dict(runner(config_path))
[rank0]:                   ^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/data/CoordExp/.worktrees/permanent-owner-bridge/src/training/pipeline.py", line 695, in run_training_pipeline
[rank0]:     return _run_initialized_training(
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/data/CoordExp/.worktrees/permanent-owner-bridge/src/training/pipeline.py", line 1786, in _run_initialized_training
[rank0]:     result = trainer.run()
[rank0]:              ^^^^^^^^^^^^^
[rank0]:   File "/data/CoordExp/.worktrees/permanent-owner-bridge/src/training/supervised_trainer.py", line 267, in run
[rank0]:     observation, consumed_count = self._run_streaming_planned_step(
[rank0]:                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/data/CoordExp/.worktrees/permanent-owner-bridge/src/training/supervised_trainer.py", line 294, in _run_streaming_planned_step
[rank0]:     return self._run_owner_bridge_planned_step(
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/data/CoordExp/.worktrees/permanent-owner-bridge/src/training/supervised_trainer.py", line 576, in _run_owner_bridge_planned_step
[rank0]:     pre_decision = self.runtime.pre_backward(
[rank0]:                    ^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/data/CoordExp/.worktrees/permanent-owner-bridge/src/runtime/train_runtime.py", line 178, in pre_backward
[rank0]:     return reduce_scalar_finite_reports(self._gather_rank_reports(report))
[rank0]:                                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/data/CoordExp/.worktrees/permanent-owner-bridge/src/runtime/train_runtime.py", line 510, in _gather_rank_reports
[rank0]:     reports = tuple(self.rank_report_gatherer(local_report))
[rank0]:                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/data/CoordExp/.worktrees/permanent-owner-bridge/src/training/pipeline.py", line 3109, in gather
[rank0]:     gathered_frames = _all_gather_cpu_bytes(
[rank0]:                       ^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/data/CoordExp/.worktrees/permanent-owner-bridge/src/training/pipeline.py", line 2961, in _all_gather_cpu_bytes
[rank0]:     distributed.all_gather(gathered, local, group=group)
[rank0]:   File "/root/miniconda3/envs/ms/lib/python3.12/site-packages/torch/distributed/c10d_logger.py", line 81, in wrapper
[rank0]:     return func(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/miniconda3/envs/ms/lib/python3.12/site-packages/torch/distributed/distributed_c10d.py", line 3949, in all_gather
[rank0]:     work.wait()
[rank0]: RuntimeError: [/pytorch/third_party/gloo/gloo/transport/tcp/pair.cc:547] Connection closed by peer [192.168.7.7]:57000
```

`4/stderr.log:46` differs only in the peer port:

```
[rank4]: RuntimeError: [/pytorch/third_party/gloo/gloo/transport/tcp/pair.cc:547] Connection closed by peer [192.168.7.7]:31485
```

Ranks 0/4 were therefore inside the **CPU/Gloo** `all_gather` of the per-rank finite
report (`TrainRuntime.pre_backward` → `_gather_rank_reports` → `_all_gather_cpu_bytes`),
not inside a CUDA collective. Their Gloo `RuntimeError` is a *consequence* of the other
six processes aborting at 03:05:11, not the initiating fault. `run.json`'s
`terminal_error` is rank 0's Gloo message (peer port `57000`).

### Ranks 1, 2, 3, 5, 6, 7 — enqueued 79, completed 77, hung on `SeqNum=78 BROADCAST`

Watchdog line, `{1,2,3,5,6,7}/stderr.log:3` (rank id and elapsed ms vary; sizes,
op type, sequence number and the 600 000 ms timeout are identical):

```
[rank1]:[E810 03:04:11.061727749 ProcessGroupNCCL.cpp:683] [Rank 1] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=78, OpType=BROADCAST, NumelIn=80, NumelOut=80, Timeout(ms)=600000) ran for 600096 milliseconds before timing out.
[rank2]:[E810 03:04:10.939039900 ProcessGroupNCCL.cpp:683] [Rank 2] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=78, OpType=BROADCAST, NumelIn=80, NumelOut=80, Timeout(ms)=600000) ran for 600087 milliseconds before timing out.
[rank3]:[E810 03:04:10.940469322 ProcessGroupNCCL.cpp:683] [Rank 3] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=78, OpType=BROADCAST, NumelIn=80, NumelOut=80, Timeout(ms)=600000) ran for 600057 milliseconds before timing out.
[rank5]:[E810 03:04:10.852112463 ProcessGroupNCCL.cpp:683] [Rank 5] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=78, OpType=BROADCAST, NumelIn=80, NumelOut=80, Timeout(ms)=600000) ran for 600011 milliseconds before timing out.
[rank6]:[E810 03:04:10.866714819 ProcessGroupNCCL.cpp:683] [Rank 6] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=78, OpType=BROADCAST, NumelIn=80, NumelOut=80, Timeout(ms)=600000) ran for 600027 milliseconds before timing out.
[rank7]:[E810 03:04:10.850617863 ProcessGroupNCCL.cpp:683] [Rank 7] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=78, OpType=BROADCAST, NumelIn=80, NumelOut=80, Timeout(ms)=600000) ran for 600003 milliseconds before timing out.
```

Process-group status, `{1,2,3,5,6,7}/stderr.log:4`:

```
[rank1]:[E810 03:04:11.063313477 ProcessGroupNCCL.cpp:2241] [PG ID 0 PG GUID 0(default_pg) Rank 1]  failure detected by watchdog at work sequence id: 78 PG status: last enqueued work: 79, last completed work: 77
[rank2]:[E810 03:04:10.940346318 ProcessGroupNCCL.cpp:2241] [PG ID 0 PG GUID 0(default_pg) Rank 2]  failure detected by watchdog at work sequence id: 78 PG status: last enqueued work: 79, last completed work: 77
[rank3]:[E810 03:04:10.941761910 ProcessGroupNCCL.cpp:2241] [PG ID 0 PG GUID 0(default_pg) Rank 3]  failure detected by watchdog at work sequence id: 78 PG status: last enqueued work: 79, last completed work: 77
[rank5]:[E810 03:04:10.853447105 ProcessGroupNCCL.cpp:2241] [PG ID 0 PG GUID 0(default_pg) Rank 5]  failure detected by watchdog at work sequence id: 78 PG status: last enqueued work: 79, last completed work: 77
[rank6]:[E810 03:04:10.868054967 ProcessGroupNCCL.cpp:2241] [PG ID 0 PG GUID 0(default_pg) Rank 6]  failure detected by watchdog at work sequence id: 78 PG status: last enqueued work: 79, last completed work: 77
[rank7]:[E810 03:04:10.852091492 ProcessGroupNCCL.cpp:2241] [PG ID 0 PG GUID 0(default_pg) Rank 7]  failure detected by watchdog at work sequence id: 78 PG status: last enqueued work: 79, last completed work: 77
```

Rank 1 was first to signal (`1/stderr.log:6`), which is why ranks 0/4 report the dump
signal as coming "from  rank 1":

```
[rank1]:[E810 03:04:11.063479079 ProcessGroupNCCL.cpp:2573] [PG ID 0 PG GUID 0(default_pg) Rank 1] First PG on this rank to signal dumping.
```

Teardown, e.g. `2/stderr.log:9-11` (also present for ranks 3, 6, 7):

```
[rank2]:[E810 03:05:11.145513712 ProcessGroupNCCL.cpp:744] [Rank 2] Some NCCL operations have failed or timed out. Due to the asynchronous nature of CUDA kernels, subsequent GPU operations might run on corrupted/incomplete data.
[rank2]:[E810 03:05:11.145572903 ProcessGroupNCCL.cpp:758] [Rank 2] To avoid data inconsistency, we are taking the entire process down.
[rank2]:[E810 03:05:11.149121725 ProcessGroupNCCL.cpp:2057] [PG ID 0 PG GUID 0(default_pg) Rank 2] Process group watchdog thread terminated with exception: [Rank 2] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=78, OpType=BROADCAST, NumelIn=80, NumelOut=80, Timeout(ms)=600000) ran for 600087 milliseconds before timing out.
```

Launcher verdict, tail of `w8-train.stdout.log`:

```
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
========================================================
src.train FAILED
--------------------------------------------------------
Failures:
  <NO_OTHER_FAILURES>
--------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2026-08-10_03:05:11
  host      : k8s-worker02
  rank      : 7 (local_rank: 7)
  exitcode  : -6 (pid: 2608408)
  error_file: <N/A>
  traceback : Signal 6 (SIGABRT) received by PID 2608408
========================================================
```

### Observation summary

| Ranks | Last enqueued NCCL work | Last completed NCCL work | Where blocked |
| --- | --- | --- | --- |
| 0, 4 | 77 | 77 | Gloo CPU `all_gather` in `pre_backward` (`_all_gather_cpu_bytes`) |
| 1, 2, 3, 5, 6, 7 | 79 | 77 | NCCL `WorkNCCL(SeqNum=78, OpType=BROADCAST, NumelIn=80, NumelOut=80)`, 600 s watchdog exceeded |

The 8 ranks split 2-vs-6 on how many NCCL works they enqueued inside the same planned
step. The six-rank group enqueued two works (78, 79) that ranks 0/4 never enqueued;
work 78 therefore had no matching participant and hung until the 600 s watchdog
tripped. Wall clock: the hanging broadcast was entered around 02:54:11Z (03:04:10.94Z
minus 600.087 s), watchdogs fired 03:04:10.85–03:04:11.06Z, processes aborted
03:05:11Z, `run.json` was finalized 03:05:11.443Z. Total run wall time ≈ 14m44s, of
which ≈10m was watchdog wait.

This is confirmed **runtime collective-choreography** evidence. It says the ranks
disagreed about collective count/order within one planned step. It says nothing about
what the bridge computes.

## Diagnosis (hypothesis, not proven by this run)

Separate from the observation above: the code shape at `d9d9768` **permits** a
rank-varying count of wrapped model forwards within one planned step, which would
produce exactly this counter skew.

In `_run_owner_bridge_planned_step` (`src/training/supervised_trainer.py` at
`d9d9768`), every rank builds the same number of stream slots
(`micro_steps_per_planned_step`), but each slot is independently *real* or *shadow*:

- a shadow slot calls `build_connected_shadow_loss(...)` — no model forward, so no
  DDP-side collectives;
- a real slot calls `execute_owner_bridge_micro_step(model=runtime_model, ...)` — a
  wrapped forward, which does enqueue collectives.

Uniform slot count therefore does **not** imply uniform wrapped-forward count. Ranks
whose window contained more real slots would enqueue more works than ranks with more
shadow slots, and the two groups would then meet at different collectives — the 2-vs-6
split observed above is consistent with that shape.

Limits of this hypothesis, stated deliberately:

- It is derived from reading the committed source, not from instrumentation in this
  run. FlightRecorder was disabled, so the hung collective has no captured stack:
  `[rank1]:[E810 03:04:11.063347235 ProcessGroupNCCL.cpp:730] Stack trace of the failed collective not found, potentially because FlightRecorder is disabled. You can enable it by setting TORCH_NCCL_TRACE_BUFFER_SIZE to a non-zero value.`
- The per-rank real/shadow composition for this specific step was **not** logged
  (`logging.jsonl` is 0 bytes), so the attribution of the 77-vs-79 gap to a specific
  slot pattern is inference, not measurement.
- Alternative causes not excluded by this run's evidence alone: a rank-conditional
  early `break` on the `pre_backward` gate path, or any other rank-divergent control
  flow that skips a collective.
- **No repair is proven here.** Any subsequent anchor/choreography change must be
  demonstrated by current Wave-5 acceptance and its own GPU smokes; this record must
  not be cited as evidence that the defect is fixed.

## Evidence inventory (hashes for re-verification)

Recorded 2026-08-10. All paths are relative to the worktree root
`/data/CoordExp/.worktrees/permanent-owner-bridge`. All are under gitignored
`outputs/`, which is why they are quoted above.

Run root
`outputs/smoke/coordexp_swift_owner_bridge/qwen3_vl_2b_desc_first_geo_sorted_xy_owner_bridge_stage1_dora_r16a32_llm_12000_accelerate8_ebs24_train256_val64_4epoch_warmup0p1-20260810T025027Z/`:

| File | sha256 | bytes | mtime (UTC) |
| --- | --- | --- | --- |
| `run.json` | `90917361c6c6c211bbb98f8486fa2826b5f8973837e6969a6a82b793e47bfadb` | 1394 | 2026-08-10 03:05:11.445 |
| `resolved_config.json` | `49245ea25526b9453bff1c47c155dd551c217d6b78251ea58f69334314021f57` | 7527 | 2026-08-10 02:50:27.617 |
| `logging.jsonl` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | 0 | 2026-08-10 02:50:27.617 |

(`e3b0c442…b855` is the sha256 of the empty string — the run emitted no telemetry.)

Log root
`outputs/smoke/coordexp_swift_owner_bridge_readiness/d9d9768/`:

| File | sha256 | bytes | mtime (UTC) |
| --- | --- | --- | --- |
| `w8-train.stdout.log` | `d368aa9fae3c28214694eaacf2eb68ea8a1f649a88dac8e87203a415cf2fe20e` | 46761 | 2026-08-10 03:05:14.098 |
| `w8-rank-logs/none_1ulisqjw/attempt_0/0/stderr.log` | `4f261cb3654e3ede888c0fe783b63925f38224449b85deb79c16b58e1de92428` | 4576 | 2026-08-10 03:05:11.449 |
| `.../1/stderr.log` | `0e047f84876bf6ba824814f7d573302ac9809b34e2b4074094cd11392338ce97` | 2017 | 2026-08-10 03:04:11.532 |
| `.../2/stderr.log` | `dc1f2a48076431c644c2f28af4fdeaee421edb0d4f00636af07bde3faa0ed62e` | 6421 | 2026-08-10 03:05:11.121 |
| `.../3/stderr.log` | `fffb46fe0cd962503e5d7494fa043c8b603ec95f4154d5b85ae6aee7a938f1ec` | 6421 | 2026-08-10 03:05:11.081 |
| `.../4/stderr.log` | `6752c237d8053be8e578423c5157dd0123d1baf1c3865198ccce1baf98283bf0` | 4576 | 2026-08-10 03:05:11.649 |
| `.../5/stderr.log` | `3788f86323f3f2d092176cb160f70dd7c8eeb11a38450431cff337e4174eb4a9` | 2017 | 2026-08-10 03:04:11.532 |
| `.../6/stderr.log` | `37b3157abedc291413911f7a0f8feb5f2bee133fac95e5d16c4581ffd97772a1` | 6421 | 2026-08-10 03:05:11.361 |
| `.../7/stderr.log` | `da786ff364532047edb4b51cecfe494e4e54ef676d8b926012cdd64472082392` | 6421 | 2026-08-10 03:05:11.049 |

Environment: PyTorch/NCCL from `/root/miniconda3/envs/ms` (Python 3.12), launched via
`conda run accelerate launch --multi_gpu --num_processes 8 --num_machines 1 --tee 3`,
host `k8s-worker02`.

## Evidence that does not exist

Named here so a later reader does not go looking for it or assume it was suppressed.

- **No step telemetry.** `logging.jsonl` is empty; `completed_steps: 0`; no
  checkpoint events, no materializations, no warning counts.
- **No flight-recorder trace.** `TORCH_NCCL_TRACE_BUFFER_SIZE` was unset, so the hung
  collective's stack was not captured on any rank.
- **Ranks 1 and 5 logs are truncated** at 8 lines / 2017 bytes (mtime 03:04:11.532Z).
  They stop after "preparing to dump debug info" and never wrote the `SIGABRT`
  teardown block that ranks 2/3/6/7 wrote. Their final state is known only through the
  launcher summary.
- **No per-rank real/shadow slot composition** for the failing planned step was
  recorded, which is the single measurement that would turn the diagnosis above from
  hypothesis into confirmation.
- **No stdout rank logs** beyond `w8-train.stdout.log`; the `attempt_0/{rank}/`
  directories contain `stderr.log` only.

## How to re-check this record

```bash
cd /data/CoordExp/.worktrees/permanent-owner-bridge
R=outputs/smoke/coordexp_swift_owner_bridge_readiness/d9d9768/w8-rank-logs/none_1ulisqjw/attempt_0
sha256sum $R/*/stderr.log
grep -inE "last enqueued" $R/*/stderr.log
grep -n "SeqNum=78" $R/*/stderr.log
python3 -m json.tool "outputs/smoke/coordexp_swift_owner_bridge/qwen3_vl_2b_desc_first_geo_sorted_xy_owner_bridge_stage1_dora_r16a32_llm_12000_accelerate8_ebs24_train256_val64_4epoch_warmup0p1-20260810T025027Z/run.json"
git show d9d9768:src/runtime/train_runtime.py | sed -n '178p;510p'
```

If the `outputs/` tree has been pruned, the quoted blocks above are the record; the
hashes let a restored copy be matched back to this document.
