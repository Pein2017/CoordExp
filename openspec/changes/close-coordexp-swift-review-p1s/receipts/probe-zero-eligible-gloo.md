# Receipt — task 2.2: timeout-bounded 2-rank gloo zero-eligible probe

**Verdict: PASS** (16/16 parent assertions, no hang, 5.62 s in-probe wall time)

Change: `close-coordexp-swift-review-p1s`, task 2.2 (P1-3).
Probe script: `scripts/probes/coordexp_swift/review_p1s_zero_eligible_gloo_probe.py`
Date: 2026-08-21. Host: CPU-only run (no GPU used), `torch 2.9.1+cu128`.

## Tree under probe

```
git rev-parse HEAD -> 04bbc1631b0e283a2dad1be7134f82dd58ba382d
```

The tree is **dirty**: the P1 fixes are uncommitted working-tree edits. The
surface this probe exercises is the P1-3 edit in `src/losses/runner.py` (the
zero-eligible decision moved out of `_build_denominator_from_token_sequences`
into `_resolve_streaming_denominators` / `_merge_global_denominators`, i.e.
after the cross-rank gather). Other dirty files (`src/runtime/*`,
`src/training/*`) carry the sibling P1-1/P1-2 work.

## Frozen argv

Green run (final):

```
bash -c 'export PYTHONDONTWRITEBYTECODE=1; export COORDEXP_PROBE_SCRATCH=/tmp/claude-0/-data-CoordExp--worktrees-CoordExp-swift/c9895ff9-1c0f-4476-97b9-d689ba469dba/scratchpad/probe-runs; time conda run -n ms python scripts/probes/coordexp_swift/review_p1s_zero_eligible_gloo_probe.py 2>&1'
```

Sensitivity run (pre-fix simulation, RED):

```
bash -c 'export PYTHONDONTWRITEBYTECODE=1; export COORDEXP_PROBE_SCRATCH=/tmp/claude-0/-data-CoordExp--worktrees-CoordExp-swift/c9895ff9-1c0f-4476-97b9-d689ba469dba/scratchpad/probe-runs; export COORDEXP_PROBE_JOIN_TIMEOUT=30; conda run -n ms python scripts/probes/coordexp_swift/review_p1s_zero_eligible_gloo_probe.py --simulate-pre-fix 2>&1'
```

The scratch path above is this session's scratchpad and is not load-bearing:
`COORDEXP_PROBE_SCRATCH` is optional and defaults to `/tmp/coordexp-swift-probes`.

Run from `/data/CoordExp/.worktrees/CoordExp-swift`. `COORDEXP_PROBE_SCRATCH`
defaults to `/tmp/coordexp-swift-probes`; each run makes a fresh uuid-named
rendezvous directory, so reruns cannot inherit a stale `file://` store.

## What is real, and what is not

Production stack exercised end to end (nothing about the gather is
re-implemented in the probe):

```
real torch.distributed gloo process group (file:// rendezvous, world_size=2)
  -> src.training.control_plane._build_rank_report_gatherer(2)      [production]
  -> src.runtime.train_runtime.TrainRuntime.gather_loss_denominators [production]
  -> src.training.supervised_trainer._runtime_loss_denominator_gatherer [production]
  -> src.losses.LossRunner.prepare_planned_step                      [production]
```

Deviations, and why:

- **The accelerator handed to `TrainRuntime` is a stand-in.** A real
  `accelerate.Accelerator` on a CPU-only gloo launch reports
  `distributed_type == MULTI_CPU`, which `validate_accelerator_runtime`
  rejects by design. The stand-in supplies rank / world size / device / a
  neutral `mixed_precision="no"` only; every denominator and gather surface
  below it is production code. The accelerator is not the surface under test
  in this probe (task 2.3 probes a real Accelerator directly).
- **The shards are hand-built `TokenSequence` objects**, as in
  `tests/losses/test_zero_eligible_collective.py`. No model, tokenizer, or
  packing pass is needed to produce a rank-local eligible count.
- **Two failure shapes, not one.** The task text asked for "rank 0 gets
  sequences with zero coordinate-type atoms". Under the strict production
  `LossesConfig`, `losses.protected.token_type_gate.groups` must be exactly
  `["desc_text", "schema", "coordinate", "eos"]` — a narrowed group tuple is
  rejected (this is the first failed attempt below), so a config-validated run
  cannot have a gate-only zero. Both shapes are therefore probed:
  - `zero_eligible_config` — production `LossesConfig` + `LossRunner.from_config`;
    rank 0's shard has segments with **no supervised atoms at all** (the real
    shape a shard takes when every atom of a segment is omitted at the segment
    boundary, cf. `OmittedPackedTokenAtom` /
    `logits_position_crosses_segment_boundary`). Both protected terms are zero
    on rank 0, so the converged failure names the first canonical term,
    `base_ce`.
  - `zero_eligible_gate` — the exact scenario the task and the unit tests
    describe: a directly constructed `LossRunner(token_type_gate_groups=("coordinate",))`,
    rank 0's shard has `desc_text` atoms only, so `base_ce` is nonzero while
    `token_type_gate` is zero. Proves the converged failure is **per term**.

## Hang bound

`JOIN_TIMEOUT_SECONDS = 90` (overridable via `COORDEXP_PROBE_JOIN_TIMEOUT`).
The parent joins both children under that bound and, on timeout, `terminate()`s
then `kill()`s them and fails loudly — the probe cannot itself hang. The bound
sits **below** the gloo control-group timeout
(`control_plane._RANK_REPORT_CONTROL_TIMEOUT_SECONDS = 120`), so a genuine
desync is reported by this probe rather than swallowed by the backend.

## Per-assertion verdicts (green run)

| # | Assertion | Verdict |
|---|-----------|---------|
| 1 | `no_hang_within_90s` — both children joined inside the bound | PASS |
| 2 | `both_children_exit_zero` — `{'probe-rank0': 0, 'probe-rank1': 0}` | PASS |
| 3 | `control_case_reported_by_both_ranks` | PASS |
| 4 | `control_denominators_are_global_sums` — scope `planned_step_global`, `eligible_segment_count == {base_ce: 4, token_type_gate: 4}` (2 per rank, summed), `selected_atom_count == {4, 4}`, `backend_gradient_scale == 2.0` | PASS |
| 5 | `zero_eligible_config.reported_by_both_ranks` | PASS |
| 6 | `zero_eligible_config.both_ranks_raise_the_expected_typed_code` — `loss.segment_balanced_zero_eligible` on both | PASS |
| 7 | `zero_eligible_config.both_rank_contexts_are_identical` | PASS |
| 8 | `zero_eligible_config.both_rank_messages_are_identical` | PASS |
| 9 | `zero_eligible_config.term_is_base_ce` | PASS |
| 10 | `zero_eligible_config.zero_eligible_ranks_names_the_zero_rank` — `[0]` on both | PASS |
| 11 | `zero_eligible_gate.reported_by_both_ranks` | PASS |
| 12 | `zero_eligible_gate.both_ranks_raise_the_expected_typed_code` | PASS |
| 13 | `zero_eligible_gate.both_rank_contexts_are_identical` | PASS |
| 14 | `zero_eligible_gate.both_rank_messages_are_identical` | PASS |
| 15 | `zero_eligible_gate.term_is_token_type_gate` | PASS |
| 16 | `zero_eligible_gate.zero_eligible_ranks_names_the_zero_rank` — `[0]` on both | PASS |

Converged failure, identical on both ranks (rank 0 is the zero-eligible rank,
rank 1 never had a zero count of its own):

```
LossContractError[loss.segment_balanced_zero_eligible]: segment_balanced reducer
requires at least one eligible segment on every rank | context:
{"context_count": 2, "eligible_segment_count": 2, "rank_count": 2,
 "selected_atom_count": 2, "skipped_segment_count": 2, "term": "token_type_gate",
 "zero_eligible_ranks": [0]}
```

Note the global `eligible_segment_count: 2` in the failure context: the peer's
shard alone makes the world sum nonzero, and the planned step still fails. The
pinned semantics ("which runs fail does not change") hold over the real
collective, not just under the unit tests' fake gatherers.

## Sensitivity (this probe has been observed RED for the right reason)

`--simulate-pre-fix` restores the pre-fix ordering in-process (monkeypatching
`src.losses.runner._build_denominator_from_token_sequences` to raise locally on
a zero eligible count, exactly where the removed raise stood — no source edit),
and reruns the same cases with a 30 s bound. Result:

- The control case still passes (the collective itself is healthy).
- `zero_eligible_config`: **rank 0 raises before the gather**, with the old
  pre-collective context `{"term": "base_ce", "context_count": 1,
  "selected_atom_count": 0, "skipped_segment_count": 2}` and **no**
  `zero_eligible_ranks` field — while **rank 1 blocks inside
  `distributed.all_gather`** in `control_plane._all_gather_cpu_bytes`.
- Neither child ever reached case 3. Both had to be SIGTERMed at the bound
  (`exit_codes={'probe-rank0': -15, 'probe-rank1': -15}`), and the parent
  printed `HANG DETECTED`.
- `SENSITIVITY VERDICT: PASS - the probe fails on the pre-fix ordering`.

This is the defect P1-3 removes, reproduced through the real gloo collective:
one rank aborting while its peer is already inside the gather.

## Failed attempts (kept as evidence)

**Attempt 1 — strict `LossesConfig` rejected the narrowed gate group tuple.**
The first version of the probe built the runner from a config with
`token_type_gate.groups: ["coordinate"]`, mirroring the unit tests. Both
children died at construction (exit 1, no hang), 5.44 s:

```
pydantic_core._pydantic_core.ValidationError: 1 validation error for LossesConfig
protected.token_type_gate
  Value error, losses.protected.token_type_gate.groups must be exactly the
  ordered tuple ['desc_text', 'schema', 'coordinate', 'eos']; got ['coordinate'].
  Omitting, duplicating, adding, or reordering a group is not supported.
```

Resolved by probing **both** shapes (see "What is real" above) rather than by
weakening the config: the config-validated shape (`zero_eligible_config`) and
the narrowed-gate shape the unit tests pin (`zero_eligible_gate`).

Parent verdicts of that attempt: `no_hang_within_90s` PASS,
`both_children_exit_zero` FAIL (`{'probe-rank0': 1, 'probe-rank1': 1}`),
both case checks FAIL (no rank reported). Full log in appendix C.

## Appendix A — green run, full output

```text
[parent] run_dir=/tmp/claude-0/-data-CoordExp--worktrees-CoordExp-swift/c9895ff9-1c0f-4476-97b9-d689ba469dba/scratchpad/probe-runs/zero-eligible-gloo-421f2b97777640809b50cfe0a315a217
[parent] torch=2.9.1+cu128 world_size=2
[parent] join_timeout_seconds=90 simulate_pre_fix=False
[Gloo] Rank 0 is connected to 1 peer ranks. Expected number of connected peer ranks is : 1
[Gloo] Rank 1 is connected to 1 peer ranks. Expected number of connected peer ranks is : 1
{"backend_gradient_scale": 2.0, "case": "control_all_ranks_nonzero", "context_count": {"base_ce": 2, "token_type_gate": 2}, "denominator_scope": "planned_step_global", "eligible_segment_count": {"base_ce": 4, "token_type_gate": 4}, "rank": 1, "selected_atom_count": {"base_ce": 4, "token_type_gate": 4}}
{"backend_gradient_scale": 2.0, "case": "control_all_ranks_nonzero", "context_count": {"base_ce": 2, "token_type_gate": 2}, "denominator_scope": "planned_step_global", "eligible_segment_count": {"base_ce": 4, "token_type_gate": 4}, "rank": 0, "selected_atom_count": {"base_ce": 4, "token_type_gate": 4}}
{"case": "zero_eligible_config", "code": "loss.segment_balanced_zero_eligible", "context": {"context_count": 2, "eligible_segment_count": 2, "rank_count": 2, "selected_atom_count": 2, "skipped_segment_count": 2, "term": "base_ce", "zero_eligible_ranks": [0]}, "message": "LossContractError[loss.segment_balanced_zero_eligible]: segment_balanced reducer requires at least one eligible segment on every rank | context: {\"context_count\": 2, \"eligible_segment_count\": 2, \"rank_count\": 2, \"selected_atom_count\": 2, \"skipped_segment_count\": 2, \"term\": \"base_ce\", \"zero_eligible_ranks\": [0]}", "rank": 1, "term": "base_ce", "zero_eligible_ranks": [0]}
{"case": "zero_eligible_config", "code": "loss.segment_balanced_zero_eligible", "context": {"context_count": 2, "eligible_segment_count": 2, "rank_count": 2, "selected_atom_count": 2, "skipped_segment_count": 2, "term": "base_ce", "zero_eligible_ranks": [0]}, "message": "LossContractError[loss.segment_balanced_zero_eligible]: segment_balanced reducer requires at least one eligible segment on every rank | context: {\"context_count\": 2, \"eligible_segment_count\": 2, \"rank_count\": 2, \"selected_atom_count\": 2, \"skipped_segment_count\": 2, \"term\": \"base_ce\", \"zero_eligible_ranks\": [0]}", "rank": 0, "term": "base_ce", "zero_eligible_ranks": [0]}
{"case": "zero_eligible_gate", "code": "loss.segment_balanced_zero_eligible", "context": {"context_count": 2, "eligible_segment_count": 2, "rank_count": 2, "selected_atom_count": 2, "skipped_segment_count": 2, "term": "token_type_gate", "zero_eligible_ranks": [0]}, "message": "LossContractError[loss.segment_balanced_zero_eligible]: segment_balanced reducer requires at least one eligible segment on every rank | context: {\"context_count\": 2, \"eligible_segment_count\": 2, \"rank_count\": 2, \"selected_atom_count\": 2, \"skipped_segment_count\": 2, \"term\": \"token_type_gate\", \"zero_eligible_ranks\": [0]}", "rank": 0, "term": "token_type_gate", "zero_eligible_ranks": [0]}
{"case": "zero_eligible_gate", "code": "loss.segment_balanced_zero_eligible", "context": {"context_count": 2, "eligible_segment_count": 2, "rank_count": 2, "selected_atom_count": 2, "skipped_segment_count": 2, "term": "token_type_gate", "zero_eligible_ranks": [0]}, "message": "LossContractError[loss.segment_balanced_zero_eligible]: segment_balanced reducer requires at least one eligible segment on every rank | context: {\"context_count\": 2, \"eligible_segment_count\": 2, \"rank_count\": 2, \"selected_atom_count\": 2, \"skipped_segment_count\": 2, \"term\": \"token_type_gate\", \"zero_eligible_ranks\": [0]}", "rank": 1, "term": "token_type_gate", "zero_eligible_ranks": [0]}
[parent] exit_codes={'probe-rank0': 0, 'probe-rank1': 0}
[parent] wall_seconds=5.62

[parent] --- verdicts ---
[parent] PASS no_hang_within_90s :: both children joined
[parent] PASS both_children_exit_zero :: {'probe-rank0': 0, 'probe-rank1': 0}
[parent] PASS control_case_reported_by_both_ranks :: ranks=[0, 1]
[parent] PASS control_denominators_are_global_sums :: {"0": {"backend_gradient_scale": 2.0, "case": "control_all_ranks_nonzero", "context_count": {"base_ce": 2, "token_type_gate": 2}, "denominator_scope": "planned_step_global", "eligible_segment_count": {"base_ce": 4, "token_type_gate": 4}, "rank": 0, "selected_atom_count": {"base_ce": 4, "token_type_gate": 4}}, "1": {"backend_gradient_scale": 2.0, "case": "control_all_ranks_nonzero", "context_count": {"base_ce": 2, "token_type_gate": 2}, "denominator_scope": "planned_step_global", "eligible_segment_count": {"base_ce": 4, "token_type_gate": 4}, "rank": 1, "selected_atom_count": {"base_ce": 4, "token_type_gate": 4}}}
[parent] PASS zero_eligible_config.reported_by_both_ranks :: ranks=[0, 1]
[parent] PASS zero_eligible_config.both_ranks_raise_the_expected_typed_code :: {"0": "loss.segment_balanced_zero_eligible", "1": "loss.segment_balanced_zero_eligible"}
[parent] PASS zero_eligible_config.both_rank_contexts_are_identical :: {"context_count": 2, "eligible_segment_count": 2, "rank_count": 2, "selected_atom_count": 2, "skipped_segment_count": 2, "term": "base_ce", "zero_eligible_ranks": [0]}
[parent] PASS zero_eligible_config.both_rank_messages_are_identical :: ["LossContractError[loss.segment_balanced_zero_eligible]: segment_balanced reducer requires at least one eligible segment on every rank | context: {\"context_count\": 2, \"eligible_segment_count\": 2, \"rank_count\": 2, \"selected_atom_count\": 2, \"skipped_segment_count\": 2, \"term\": \"base_ce\", \"zero_eligible_ranks\": [0]}"]
[parent] PASS zero_eligible_config.term_is_base_ce :: {"0": "base_ce", "1": "base_ce"}
[parent] PASS zero_eligible_config.zero_eligible_ranks_names_the_zero_rank :: {"0": [0], "1": [0]}
[parent] PASS zero_eligible_gate.reported_by_both_ranks :: ranks=[0, 1]
[parent] PASS zero_eligible_gate.both_ranks_raise_the_expected_typed_code :: {"0": "loss.segment_balanced_zero_eligible", "1": "loss.segment_balanced_zero_eligible"}
[parent] PASS zero_eligible_gate.both_rank_contexts_are_identical :: {"context_count": 2, "eligible_segment_count": 2, "rank_count": 2, "selected_atom_count": 2, "skipped_segment_count": 2, "term": "token_type_gate", "zero_eligible_ranks": [0]}
[parent] PASS zero_eligible_gate.both_rank_messages_are_identical :: ["LossContractError[loss.segment_balanced_zero_eligible]: segment_balanced reducer requires at least one eligible segment on every rank | context: {\"context_count\": 2, \"eligible_segment_count\": 2, \"rank_count\": 2, \"selected_atom_count\": 2, \"skipped_segment_count\": 2, \"term\": \"token_type_gate\", \"zero_eligible_ranks\": [0]}"]
[parent] PASS zero_eligible_gate.term_is_token_type_gate :: {"0": "token_type_gate", "1": "token_type_gate"}
[parent] PASS zero_eligible_gate.zero_eligible_ranks_names_the_zero_rank :: {"0": [0], "1": [0]}
[parent] VERDICT: PASS
[parent] wall_seconds=5.62


real	0m13.356s
user	0m59.406s
sys	0m3.442s
```

## Appendix B — sensitivity run (`--simulate-pre-fix`), full output

```text
Traceback (most recent call last):
  File "/data/CoordExp/.worktrees/CoordExp-swift/scripts/probes/coordexp_swift/review_p1s_zero_eligible_gloo_probe.py", line 449, in _child
    _zero_eligible_case(
  File "/data/CoordExp/.worktrees/CoordExp-swift/scripts/probes/coordexp_swift/review_p1s_zero_eligible_gloo_probe.py", line 335, in _zero_eligible_case
    assert raised.context.get("zero_eligible_ranks") == [ZERO_ELIGIBLE_RANK], (
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: {'term': 'base_ce', 'context_count': 1, 'selected_atom_count': 0, 'skipped_segment_count': 2}
Traceback (most recent call last):
  File "/data/CoordExp/.worktrees/CoordExp-swift/scripts/probes/coordexp_swift/review_p1s_zero_eligible_gloo_probe.py", line 449, in _child
    _zero_eligible_case(
  File "/data/CoordExp/.worktrees/CoordExp-swift/scripts/probes/coordexp_swift/review_p1s_zero_eligible_gloo_probe.py", line 298, in _zero_eligible_case
    runner.prepare_planned_step(
  File "/data/CoordExp/.worktrees/CoordExp-swift/src/losses/runner.py", line 356, in prepare_planned_step
    _resolve_streaming_denominators(
  File "/data/CoordExp/.worktrees/CoordExp-swift/src/losses/runner.py", line 819, in _resolve_streaming_denominators
    gathered_payloads = tuple(denominator_gatherer(local_payload))
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/CoordExp/.worktrees/CoordExp-swift/src/training/supervised_trainer.py", line 680, in <lambda>
    return lambda payload: gather_loss_denominators(
                           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/CoordExp/.worktrees/CoordExp-swift/src/runtime/train_runtime.py", line 483, in gather_loss_denominators
    reports = self._gather_rank_reports(payload)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/CoordExp/.worktrees/CoordExp-swift/src/runtime/train_runtime.py", line 776, in _gather_rank_reports
    reports = tuple(self.rank_report_gatherer(local_report))
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/CoordExp/.worktrees/CoordExp-swift/src/training/control_plane.py", line 313, in gather
    gathered_frames = _all_gather_cpu_bytes(
                      ^^^^^^^^^^^^^^^^^^^^^^
  File "/data/CoordExp/.worktrees/CoordExp-swift/src/training/control_plane.py", line 158, in _all_gather_cpu_bytes
    distributed.all_gather(gathered, local, group=group)
  File "/root/miniconda3/envs/ms/lib/python3.12/site-packages/torch/distributed/c10d_logger.py", line 81, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/root/miniconda3/envs/ms/lib/python3.12/site-packages/torch/distributed/distributed_c10d.py", line 3949, in all_gather
    work.wait()
RuntimeError: [/pytorch/third_party/gloo/gloo/transport/tcp/pair.cc:547] Connection closed by peer [192.168.7.7]:23568
Traceback (most recent call last):
  File "/data/CoordExp/.worktrees/CoordExp-swift/scripts/probes/coordexp_swift/review_p1s_zero_eligible_gloo_probe.py", line 478, in _child
    dist.barrier()
  File "/root/miniconda3/envs/ms/lib/python3.12/site-packages/torch/distributed/c10d_logger.py", line 81, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/root/miniconda3/envs/ms/lib/python3.12/site-packages/torch/distributed/distributed_c10d.py", line 4888, in barrier
    work.wait()
RuntimeError: [/pytorch/third_party/gloo/gloo/transport/tcp/pair.cc:547] Connection closed by peer [192.168.7.7]:23568

[parent] run_dir=/tmp/claude-0/-data-CoordExp--worktrees-CoordExp-swift/c9895ff9-1c0f-4476-97b9-d689ba469dba/scratchpad/probe-runs/zero-eligible-gloo-402b9b9ff6d74543a589ad178ede13c1
[parent] torch=2.9.1+cu128 world_size=2
[parent] join_timeout_seconds=30 simulate_pre_fix=True
[parent] SENSITIVITY RUN: the pre-fix rank-local raise is restored in-process; a FAIL below is the expected outcome and this run's process exit code is inverted.
[Gloo] Rank 0 is connected to 1 peer ranks. Expected number of connected peer ranks is : 1
[Gloo] Rank 1 is connected to 1 peer ranks. Expected number of connected peer ranks is : 1
{"backend_gradient_scale": 2.0, "case": "control_all_ranks_nonzero", "context_count": {"base_ce": 2, "token_type_gate": 2}, "denominator_scope": "planned_step_global", "eligible_segment_count": {"base_ce": 4, "token_type_gate": 4}, "rank": 1, "selected_atom_count": {"base_ce": 4, "token_type_gate": 4}}
{"backend_gradient_scale": 2.0, "case": "control_all_ranks_nonzero", "context_count": {"base_ce": 2, "token_type_gate": 2}, "denominator_scope": "planned_step_global", "eligible_segment_count": {"base_ce": 4, "token_type_gate": 4}, "rank": 0, "selected_atom_count": {"base_ce": 4, "token_type_gate": 4}}
{"case": "zero_eligible_config", "code": "loss.segment_balanced_zero_eligible", "context": {"context_count": 1, "selected_atom_count": 0, "skipped_segment_count": 2, "term": "base_ce"}, "message": "LossContractError[loss.segment_balanced_zero_eligible]: segment_balanced reducer requires at least one eligible segment | context: {\"context_count\": 1, \"selected_atom_count\": 0, \"skipped_segment_count\": 2, \"term\": \"base_ce\"}", "rank": 0, "term": "base_ce", "zero_eligible_ranks": null}
[parent] exit_codes={'probe-rank0': -15, 'probe-rank1': -15}
[parent] wall_seconds=30.17

[parent] --- verdicts ---
[parent] FAIL no_hang_within_30s :: hung=['probe-rank0', 'probe-rank1']
[parent] FAIL both_children_exit_zero :: {'probe-rank0': -15, 'probe-rank1': -15}
[parent] PASS control_case_reported_by_both_ranks :: ranks=[0, 1]
[parent] PASS control_denominators_are_global_sums :: {"0": {"backend_gradient_scale": 2.0, "case": "control_all_ranks_nonzero", "context_count": {"base_ce": 2, "token_type_gate": 2}, "denominator_scope": "planned_step_global", "eligible_segment_count": {"base_ce": 4, "token_type_gate": 4}, "rank": 0, "selected_atom_count": {"base_ce": 4, "token_type_gate": 4}}, "1": {"backend_gradient_scale": 2.0, "case": "control_all_ranks_nonzero", "context_count": {"base_ce": 2, "token_type_gate": 2}, "denominator_scope": "planned_step_global", "eligible_segment_count": {"base_ce": 4, "token_type_gate": 4}, "rank": 1, "selected_atom_count": {"base_ce": 4, "token_type_gate": 4}}}
[parent] FAIL zero_eligible_config.reported_by_both_ranks :: ranks=[0]
[parent] FAIL zero_eligible_gate.reported_by_both_ranks :: ranks=[]
[parent] HANG DETECTED: children did not converge the failure within 30s; terminated ['probe-rank0', 'probe-rank1']
[parent] VERDICT: FAIL
[parent] wall_seconds=30.17
[parent] SENSITIVITY VERDICT: PASS - the probe fails on the pre-fix ordering

```

## Appendix C — failed attempt 1, full output

```text
Traceback (most recent call last):
  File "/data/CoordExp/.worktrees/CoordExp-swift/scripts/probes/coordexp_swift/review_p1s_zero_eligible_gloo_probe.py", line 250, in _child
    runner = _loss_runner()
             ^^^^^^^^^^^^^^
  File "/data/CoordExp/.worktrees/CoordExp-swift/scripts/probes/coordexp_swift/review_p1s_zero_eligible_gloo_probe.py", line 126, in _loss_runner
    config = LossesConfig.model_validate(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/miniconda3/envs/ms/lib/python3.12/site-packages/pydantic/main.py", line 732, in model_validate
    return cls.__pydantic_validator__.validate_python(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
pydantic_core._pydantic_core.ValidationError: 1 validation error for LossesConfig
protected.token_type_gate
  Value error, losses.protected.token_type_gate.groups must be exactly the ordered tuple ['desc_text', 'schema', 'coordinate', 'eos']; got ['coordinate']. Omitting, duplicating, adding, or reordering a group is not supported. [type=value_error, input_value={'weight': 0.1, 'mode': '...groups': ['coordinate']}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/value_error
Traceback (most recent call last):
  File "/data/CoordExp/.worktrees/CoordExp-swift/scripts/probes/coordexp_swift/review_p1s_zero_eligible_gloo_probe.py", line 250, in _child
    runner = _loss_runner()
             ^^^^^^^^^^^^^^
  File "/data/CoordExp/.worktrees/CoordExp-swift/scripts/probes/coordexp_swift/review_p1s_zero_eligible_gloo_probe.py", line 126, in _loss_runner
    config = LossesConfig.model_validate(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/miniconda3/envs/ms/lib/python3.12/site-packages/pydantic/main.py", line 732, in model_validate
    return cls.__pydantic_validator__.validate_python(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
pydantic_core._pydantic_core.ValidationError: 1 validation error for LossesConfig
protected.token_type_gate
  Value error, losses.protected.token_type_gate.groups must be exactly the ordered tuple ['desc_text', 'schema', 'coordinate', 'eos']; got ['coordinate']. Omitting, duplicating, adding, or reordering a group is not supported. [type=value_error, input_value={'weight': 0.1, 'mode': '...groups': ['coordinate']}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/value_error

ERROR conda.cli.main_run:execute(127): `conda run python scripts/probes/coordexp_swift/review_p1s_zero_eligible_gloo_probe.py` failed. (See above for error)
[parent] run_dir=/tmp/claude-0/-data-CoordExp--worktrees-CoordExp-swift/c9895ff9-1c0f-4476-97b9-d689ba469dba/scratchpad/probe-runs/zero-eligible-gloo-1493a55bde31446cb91e30bf7692155e
[parent] torch=2.9.1+cu128 world_size=2
[Gloo] Rank 1 is connected to 1 peer ranks. Expected number of connected peer ranks is : 1
[Gloo] Rank 0 is connected to 1 peer ranks. Expected number of connected peer ranks is : 1
[parent] exit_codes={'probe-rank0': 1, 'probe-rank1': 1}
[parent] wall_seconds=5.44

[parent] --- verdicts ---
[parent] PASS no_hang_within_90s :: both children joined
[parent] FAIL both_children_exit_zero :: {'probe-rank0': 1, 'probe-rank1': 1}
[parent] FAIL control_case_reported_by_both_ranks :: ranks=[]
[parent] FAIL zero_eligible_case_reported_by_both_ranks :: ranks=[]
[parent] VERDICT: FAIL
[parent] wall_seconds=5.44

```
