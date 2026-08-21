# Receipt — task 2.3: fp16 launch fail-closed probe

**Verdict: PASS** (30/30 assertions, 7.09 s wall including interpreter start;
0.03 s in-probe)

Change: `close-coordexp-swift-review-p1s`, task 2.3 (P1-2).
Probe script: `scripts/probes/coordexp_swift/review_p1s_fp16_launch_refusal_probe.py`
Date: 2026-08-21. Host: CPU-only run (no GPU work, no CUDA tensor allocated),
`torch 2.9.1+cu128`, `accelerate 1.10.1`.

## Tree under probe

```
git rev-parse HEAD -> 04bbc1631b0e283a2dad1be7134f82dd58ba382d
```

The tree is **dirty**: the P1 fixes are uncommitted working-tree edits. The
surface this probe exercises is the P1-2 edit in `src/runtime/train_runtime.py`
(`_validate_fp16_scaler_contract`, called from `validate_accelerator_runtime`,
plus the shared `_resolve_active_fp16_scaler` lookup and the `declared_fp16`
declaration). Other dirty files carry the sibling P1-1/P1-3 work.

## Frozen argv

Green run (final):

```
bash -c 'export PYTHONDONTWRITEBYTECODE=1; time conda run -n ms python scripts/probes/coordexp_swift/review_p1s_fp16_launch_refusal_probe.py 2>&1'
```

Sensitivity run (pre-fix simulation, RED):

```
bash -c 'export PYTHONDONTWRITEBYTECODE=1; time conda run -n ms python scripts/probes/coordexp_swift/review_p1s_fp16_launch_refusal_probe.py --simulate-pre-fix 2>&1'
```

Run from `/data/CoordExp/.worktrees/CoordExp-swift`.

## What is real, and what is not

**No accelerator stand-in is used.** The task allowed falling back to the
bounded stand-in that `tests/runtime/test_fp16_scaler_contract.py` uses; it was
not needed. `accelerate.Accelerator(mixed_precision="fp16", cpu=True)` is a
genuine `accelerate.accelerator.Accelerator` that reports
`mixed_precision == "fp16"` with `scaler is None` — accelerate itself produces
the exact declared-fp16-without-a-scaler shape P1-2 refuses. Every case below
runs on a real Accelerator, and the scalers in cases (b) and (d) are real
`torch.amp.grad_scaler.GradScaler` objects, not mocks.

Entry points driven (both real):

- `src.runtime.train_runtime.validate_accelerator_runtime(...)` — the
  validation entry the fix hooks into.
- `src.runtime.train_runtime.TrainRuntime(...)` — the production constructor
  that calls it, with a real `torch.nn.Linear` model.

Deviations, and why:

- **`AcceleratorState._reset_state(True)` between cases** is private accelerate
  API. It is the only way to construct more than one `Accelerator` in one
  process; the alternative (one subprocess per case) would add nothing to what
  is being probed.
- **Case (b) constructs the disabled scaler explicitly.** The task assumed a
  host where a CUDA GradScaler self-disables. This host has a usable CUDA build
  (`torch.cuda.is_available() == True`), so `torch.amp.GradScaler("cuda")` stays
  enabled. `torch.amp.GradScaler("cuda", enabled=False)` is the supported way to
  obtain the same real object in its disabled state — identical class
  (`torch.amp.grad_scaler.GradScaler`), `is_enabled() is False`. The probe
  asserts both facts before using it.
- **`accelerator.prepare` is wrapped with a counting shim** (the real bound
  method is still called through). Real Accelerators have no `prepare_calls`
  counter, and the ordering claim — the refusal precedes every training-side
  mutation — needs to be measured, not assumed.
- **Case (d) is beyond the task**: an over-refusal control (declared fp16 with a
  real *enabled* scaler must NOT be refused). Constructing
  `GradScaler("cuda", enabled=True)` allocates no device memory.

## Per-case verdicts

### Case (a) — REAL Accelerator, fp16 declared, scaler absent → refused

`{"accelerator_type": "accelerate.accelerator.Accelerator", "device": "cpu",
"distributed_type": "DistributedType.NO", "mixed_precision": "fp16",
"num_processes": 1, "scaler_type": null}`

| Assertion | Verdict |
|-----------|---------|
| `a.accelerator_is_real_accelerate` (`accelerate.accelerator`) | PASS |
| `a.declares_fp16` | PASS |
| `a.scaler_is_absent` | PASS |
| `a.validate_accelerator_runtime_refuses` | PASS |
| `a.code_is_expected` → `runtime.fp16_scaler_missing` | PASS |
| `a.context_reports_absent_scaler` (`scaler_present: false`, `scaler_enabled: null`, `mixed_precision: "fp16"`) | PASS |
| `a.train_runtime_construction_refuses` | PASS |
| `a.train_runtime_code_is_expected` → `runtime.fp16_scaler_missing` | PASS |
| `a.refusal_precedes_accelerator_prepare` (`prepare` calls == 0) | PASS |

```
RuntimeContractError[runtime.fp16_scaler_missing]: training declares fp16 mixed
precision but no active, enabled GradScaler is reachable through the declared
scaler lookup | context: {"mixed_precision": "fp16", "scaler_enabled": null,
"scaler_present": false, "scaler_type": null}
```

### Case (b) — REAL Accelerator, fp16 declared, REAL disabled GradScaler → refused

`scaler_module: "torch.amp.grad_scaler"`, `scaler_type: "GradScaler"`,
`scaler_is_enabled: false`.

| Assertion | Verdict |
|-----------|---------|
| `b.scaler_is_a_real_torch_gradscaler` (`torch.amp.grad_scaler.GradScaler`) | PASS |
| `b.scaler_is_disabled` (`is_enabled() is False`) | PASS |
| `b.validate_accelerator_runtime_refuses` | PASS |
| `b.code_is_expected` → `runtime.fp16_scaler_missing` | PASS |
| `b.context_reports_present_but_disabled_scaler` (`scaler_present: true`, `scaler_enabled: false`, `scaler_type: "GradScaler"`) | PASS |
| `b.train_runtime_construction_refuses` | PASS |
| `b.train_runtime_code_is_expected` | PASS |
| `b.refusal_precedes_accelerator_prepare` (`prepare` calls == 0) | PASS |

```
RuntimeContractError[runtime.fp16_scaler_missing]: training declares fp16 mixed
precision but no active, enabled GradScaler is reachable through the declared
scaler lookup | context: {"mixed_precision": "fp16", "scaler_enabled": false,
"scaler_present": true, "scaler_type": "GradScaler"}
```

A present-but-disabled scaler is refused with the same code as an absent one,
and the context distinguishes the two cases — the gate is keyed on the *active*
lookup, not on attribute presence.

### Case (c) — CONTROL: REAL Accelerator, bf16 declared, no scaler → not refused

| Assertion | Verdict |
|-----------|---------|
| `c.declares_bf16` / `c.scaler_is_absent` | PASS |
| `c.validate_accelerator_runtime_accepts` (no refusal) | PASS |
| `c.train_runtime_constructs` | PASS |
| `c.declared_fp16_is_false` | PASS |
| `c.no_active_fp16_scaler` | PASS |
| `c.accelerator_prepare_was_reached` (`prepare` calls == 1) | PASS |

The genuine bf16/no-scaler run is untouched: it constructs and reaches
`accelerator.prepare` exactly once.

### Case (d) — OVER-REFUSAL CONTROL (beyond task): fp16 + REAL enabled GradScaler → not refused

| Assertion | Verdict |
|-----------|---------|
| `d.scaler_is_enabled` | PASS |
| `d.validate_accelerator_runtime_accepts` | PASS |
| `d.train_runtime_constructs` | PASS |
| `d.declared_fp16_is_true` | PASS |
| `d.resolves_that_exact_scaler` (`runtime._active_fp16_scaler() is scaler`) | PASS |
| `d.accelerator_prepare_was_reached` (`prepare` calls == 1) | PASS |

The gate does not over-refuse, and the runtime resolves the same scaler object
the launch gate saw — the "one declared scaler lookup" claim, measured.

## Sensitivity (this probe has been observed RED for the right reason)

`--simulate-pre-fix` replaces `src.runtime.train_runtime._validate_fp16_scaler_contract`
with a no-op in-process (the pre-P1-2 behaviour, no source edit) and reruns the
same four cases. Result: **6 assertions fail, all of them in cases (a) and (b)**:

```
failed=['a.validate_accelerator_runtime_refuses',
        'a.train_runtime_construction_refuses',
        'a.refusal_precedes_accelerator_prepare',
        'b.validate_accelerator_runtime_refuses',
        'b.train_runtime_construction_refuses',
        'b.refusal_precedes_accelerator_prepare']
SENSITIVITY VERDICT: PASS - the probe fails without the launch gate
```

The `refusal_precedes_accelerator_prepare` failures report `{"calls": 1}`: with
the gate removed, a run that declares fp16 with no usable scaler **constructs
successfully and proceeds into `accelerator.prepare`** — the exact silent
convergence P1-2 closes. Controls (c) and (d) stay green in the sensitivity run,
so the failures are attributable to the gate and not to probe breakage.

## Failed attempts

None. Both the green run and the sensitivity run behaved as designed on the
first execution of the finished script. The only pre-run exploration was a
one-off capability check (`torch.cuda.is_available()`, `accelerate` importable,
`Accelerator(mixed_precision="fp16", cpu=True)` yielding `scaler=None`), which
is what selected the real-Accelerator route over the stand-in.

## Appendix A — green run, full output

```text
[probe] torch=2.9.1+cu128 cuda_available=True
[probe] accelerate=1.10.1
[probe] simulate_pre_fix=False

[case a] REAL Accelerator, fp16 declared, scaler absent
  accelerator={"accelerator_type": "accelerate.accelerator.Accelerator", "device": "cpu", "distributed_type": "DistributedType.NO", "mixed_precision": "fp16", "num_processes": 1, "scaler_is_enabled": null, "scaler_module": null, "scaler_type": null}
  PASS a.accelerator_is_real_accelerate :: accelerate.accelerator
  PASS a.declares_fp16 ::
  PASS a.scaler_is_absent ::
  PASS a.validate_accelerator_runtime_refuses ::
  error=RuntimeContractError[runtime.fp16_scaler_missing]: training declares fp16 mixed precision but no active, enabled GradScaler is reachable through the declared scaler lookup | context: {"mixed_precision": "fp16", "scaler_enabled": null, "scaler_present": false, "scaler_type": null}
  PASS a.code_is_expected :: runtime.fp16_scaler_missing
  PASS a.context_reports_absent_scaler :: {"mixed_precision": "fp16", "scaler_present": false, "scaler_enabled": null, "scaler_type": null}
  PASS a.train_runtime_construction_refuses ::
  PASS a.train_runtime_code_is_expected :: runtime.fp16_scaler_missing
  PASS a.refusal_precedes_accelerator_prepare :: {"calls": 0}

[case b] REAL Accelerator, fp16 declared, REAL disabled GradScaler
  accelerator={"accelerator_type": "accelerate.accelerator.Accelerator", "device": "cpu", "distributed_type": "DistributedType.NO", "mixed_precision": "fp16", "num_processes": 1, "scaler_is_enabled": false, "scaler_module": "torch.amp.grad_scaler", "scaler_type": "GradScaler"}
  PASS b.scaler_is_a_real_torch_gradscaler :: torch.amp.grad_scaler.GradScaler
  PASS b.scaler_is_disabled ::
  PASS b.validate_accelerator_runtime_refuses ::
  error=RuntimeContractError[runtime.fp16_scaler_missing]: training declares fp16 mixed precision but no active, enabled GradScaler is reachable through the declared scaler lookup | context: {"mixed_precision": "fp16", "scaler_enabled": false, "scaler_present": true, "scaler_type": "GradScaler"}
  PASS b.code_is_expected :: runtime.fp16_scaler_missing
  PASS b.context_reports_present_but_disabled_scaler :: {"mixed_precision": "fp16", "scaler_present": true, "scaler_enabled": false, "scaler_type": "GradScaler"}
  PASS b.train_runtime_construction_refuses ::
  PASS b.train_runtime_code_is_expected :: runtime.fp16_scaler_missing
  PASS b.refusal_precedes_accelerator_prepare :: {"calls": 0}

[case c] CONTROL: REAL Accelerator, bf16 declared, no scaler
  accelerator={"accelerator_type": "accelerate.accelerator.Accelerator", "device": "cpu", "distributed_type": "DistributedType.NO", "mixed_precision": "bf16", "num_processes": 1, "scaler_is_enabled": null, "scaler_module": null, "scaler_type": null}
  PASS c.declares_bf16 ::
  PASS c.scaler_is_absent ::
  PASS c.validate_accelerator_runtime_accepts :: no refusal
  PASS c.train_runtime_constructs :: ok
  PASS c.declared_fp16_is_false ::
  PASS c.no_active_fp16_scaler ::
  PASS c.accelerator_prepare_was_reached :: {"calls": 1}

[case d] OVER-REFUSAL CONTROL: fp16 declared, REAL enabled GradScaler
  accelerator={"accelerator_type": "accelerate.accelerator.Accelerator", "device": "cpu", "distributed_type": "DistributedType.NO", "mixed_precision": "fp16", "num_processes": 1, "scaler_is_enabled": true, "scaler_module": "torch.amp.grad_scaler", "scaler_type": "GradScaler"}
  PASS d.scaler_is_enabled ::
  PASS d.validate_accelerator_runtime_accepts :: no refusal
  PASS d.train_runtime_constructs :: ok
  PASS d.declared_fp16_is_true ::
  PASS d.resolves_that_exact_scaler ::
  PASS d.accelerator_prepare_was_reached :: {"calls": 1}

[probe] --- summary ---
[probe] assertions=30 failed=[]
[probe] VERDICT: PASS
[probe] wall_seconds=0.03


real	0m7.093s
user	0m18.983s
sys	0m1.044s
```

## Appendix B — sensitivity run (`--simulate-pre-fix`), full output

```text
[probe] torch=2.9.1+cu128 cuda_available=True
[probe] accelerate=1.10.1
[probe] simulate_pre_fix=True
[probe] SENSITIVITY RUN: the P1-2 launch gate is neutralized in-process; a FAIL below is the expected outcome and this run's process exit code is inverted.

[case a] REAL Accelerator, fp16 declared, scaler absent
  accelerator={"accelerator_type": "accelerate.accelerator.Accelerator", "device": "cpu", "distributed_type": "DistributedType.NO", "mixed_precision": "fp16", "num_processes": 1, "scaler_is_enabled": null, "scaler_module": null, "scaler_type": null}
  PASS a.accelerator_is_real_accelerate :: accelerate.accelerator
  PASS a.declares_fp16 ::
  PASS a.scaler_is_absent ::
  FAIL a.validate_accelerator_runtime_refuses ::
  FAIL a.train_runtime_construction_refuses ::
  FAIL a.refusal_precedes_accelerator_prepare :: {"calls": 1}

[case b] REAL Accelerator, fp16 declared, REAL disabled GradScaler
  accelerator={"accelerator_type": "accelerate.accelerator.Accelerator", "device": "cpu", "distributed_type": "DistributedType.NO", "mixed_precision": "fp16", "num_processes": 1, "scaler_is_enabled": false, "scaler_module": "torch.amp.grad_scaler", "scaler_type": "GradScaler"}
  PASS b.scaler_is_a_real_torch_gradscaler :: torch.amp.grad_scaler.GradScaler
  PASS b.scaler_is_disabled ::
  FAIL b.validate_accelerator_runtime_refuses ::
  FAIL b.train_runtime_construction_refuses ::
  FAIL b.refusal_precedes_accelerator_prepare :: {"calls": 1}

[case c] CONTROL: REAL Accelerator, bf16 declared, no scaler
  accelerator={"accelerator_type": "accelerate.accelerator.Accelerator", "device": "cpu", "distributed_type": "DistributedType.NO", "mixed_precision": "bf16", "num_processes": 1, "scaler_is_enabled": null, "scaler_module": null, "scaler_type": null}
  PASS c.declares_bf16 ::
  PASS c.scaler_is_absent ::
  PASS c.validate_accelerator_runtime_accepts :: no refusal
  PASS c.train_runtime_constructs :: ok
  PASS c.declared_fp16_is_false ::
  PASS c.no_active_fp16_scaler ::
  PASS c.accelerator_prepare_was_reached :: {"calls": 1}

[case d] OVER-REFUSAL CONTROL: fp16 declared, REAL enabled GradScaler
  accelerator={"accelerator_type": "accelerate.accelerator.Accelerator", "device": "cpu", "distributed_type": "DistributedType.NO", "mixed_precision": "fp16", "num_processes": 1, "scaler_is_enabled": true, "scaler_module": "torch.amp.grad_scaler", "scaler_type": "GradScaler"}
  PASS d.scaler_is_enabled ::
  PASS d.validate_accelerator_runtime_accepts :: no refusal
  PASS d.train_runtime_constructs :: ok
  PASS d.declared_fp16_is_true ::
  PASS d.resolves_that_exact_scaler ::
  PASS d.accelerator_prepare_was_reached :: {"calls": 1}

[probe] --- summary ---
[probe] assertions=24 failed=['a.validate_accelerator_runtime_refuses', 'a.train_runtime_construction_refuses', 'a.refusal_precedes_accelerator_prepare', 'b.validate_accelerator_runtime_refuses', 'b.train_runtime_construction_refuses', 'b.refusal_precedes_accelerator_prepare']
[probe] VERDICT: FAIL
[probe] wall_seconds=0.03
[probe] SENSITIVITY VERDICT: PASS - the probe fails without the launch gate

```
