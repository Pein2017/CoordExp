## Scope

Audit target: `try/except` behavior in core training/inference/eval paths under `src/`, with emphasis on silent suppression patterns blocked by `tests/test_silent_failure_policy.py`.

Policy intent:
- Unexpected internal exceptions must fail fast.
- Operator-controlled input violations must fail fast.
- Continue-but-observable behavior is limited to explicit model-output consumer paths.
- Sink-scoped best-effort handling is allowed only for non-correctness diagnostics/I-O and must remain observable.

## 2026-03-04 Remediation Summary

The previously failing blanket suppression patterns (`except Exception: pass`, blanket `continue`, blanket `return`) in Stage-2 trainer paths were remediated by narrowing exception classes at the affected sites.

### P0 (previously failing policy checks, now resolved)

- `src/trainers/stage2_rollout_aligned.py`
  - `_cuda_memory_drain` best-effort cleanup handlers narrowed:
    - [stage2_rollout_aligned.py:2691](/data/CoordExp/src/trainers/stage2_rollout_aligned.py:2691)
    - [stage2_rollout_aligned.py:2696](/data/CoordExp/src/trainers/stage2_rollout_aligned.py:2696)
    - [stage2_rollout_aligned.py:2703](/data/CoordExp/src/trainers/stage2_rollout_aligned.py:2703)
  - vLLM engine class lookup fallback narrowed from blanket continue:
    - [stage2_rollout_aligned.py:2807](/data/CoordExp/src/trainers/stage2_rollout_aligned.py:2807)
  - atexit unregister / allocator-pool cleanup fallbacks narrowed:
    - [stage2_rollout_aligned.py:2904](/data/CoordExp/src/trainers/stage2_rollout_aligned.py:2904)
    - [stage2_rollout_aligned.py:2908](/data/CoordExp/src/trainers/stage2_rollout_aligned.py:2908)
    - [stage2_rollout_aligned.py:3011](/data/CoordExp/src/trainers/stage2_rollout_aligned.py:3011)
  - decode helper fallback narrowed:
    - [stage2_rollout_aligned.py:5341](/data/CoordExp/src/trainers/stage2_rollout_aligned.py:5341)

- `src/trainers/stage2_two_channel.py`
  - Stage2-AB avg-tokens override fallback narrowed:
    - [stage2_two_channel.py:434](/data/CoordExp/src/trainers/stage2_two_channel.py:434)

- `src/trainers/stage2_two_channel/executors.py`
  - avg-tokens override fallbacks narrowed:
    - [executors.py:78](/data/CoordExp/src/trainers/stage2_two_channel/executors.py:78)
    - [executors.py:104](/data/CoordExp/src/trainers/stage2_two_channel/executors.py:104)
  - DDP monitor-group init fallback return narrowed:
    - [executors.py:593](/data/CoordExp/src/trainers/stage2_two_channel/executors.py:593)

### P1 (classification outcome for previously flagged handlers)

Classified previously flagged handlers into two groups:

1. **Legitimate sink-scoped/best-effort cleanup**:
- CUDA allocator drain, atexit unregister, optional allocator pool teardown, tokenizer piece decode fallback.
- These remain best-effort but now use narrower exception classes where feasible.

2. **Core training control-path protections**:
- Stage2-AB DDP monitor-group fallback path and args override restoration.
- These were narrowed to avoid blanket suppression while preserving safe fallback semantics.

### P2 (residual broad `except Exception` inventory)

A broader repo scan still finds additional `except Exception` sites in large Stage-2 trainer surfaces that are not Tier-0/Tier-1 silent suppression violations (for example, warn-and-reraise or explicit diagnostics paths). These remain for future incremental tightening and are outside this remediation slice.

## Verification Evidence

### Policy tests (blocking)

- `conda run -n ms python -m pytest -q tests/test_silent_failure_policy.py`
  - Result: `3 passed`

### Policy-adjacent checks

- `conda run -n ms python -m pytest -q tests/test_no_silent_except_exception_pass.py tests/test_batch_extras_failure_not_silent.py tests/test_augmentation_curriculum_contract.py`
  - Result: `9 passed`

### Stage-2 representative smoke probe

Probe config:
- [temp/smoke_stage2_silent_failure_probe.yaml](/data/CoordExp/temp/smoke_stage2_silent_failure_probe.yaml)

Run command:
- `config=temp/smoke_stage2_silent_failure_probe.yaml gpus=7 conda run -n ms bash scripts/train.sh`

Observed evidence in log:
- One training step completed and metrics were emitted (smoke reached trainer runtime).
- Continue-but-observable warning during eval rollout decode:
  - [log_silent_failure_probe.log](/data/CoordExp/temp/log_silent_failure_probe.log):526
  - `Eval vLLM decode failed for sample_idx=0; skipping sample...`
- Unexpected eval rollout exception terminated the run non-zero (fail-fast):
  - [log_silent_failure_probe.log](/data/CoordExp/temp/log_silent_failure_probe.log):589
  - `RuntimeError: Eval vLLM rollout failed for all samples in a batch; aborting evaluation.`
- GPU binding confirmation for this probe:
  - [log_silent_failure_probe.log](/data/CoordExp/temp/log_silent_failure_probe.log):736
  - `CUDA_VISIBLE_DEVICES=7`

Interpretation:
- Fail-fast behavior for unexpected runtime failures is preserved.
- Observable warning behavior for permitted degraded processing paths is preserved.
