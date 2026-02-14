## Why

Stage-2 AB Channel-B stop-neutral masking removes supervision for top-level closure (`}` and `<|im_end|>`), which can bias training toward over-generation in B-dominant schedules.
Recent smoke evidence (for example `smoke_ab_mixed/v0-20260210-153258`) shows rollout-length inflation under stop-neutral without reliable training-time TP-hit gains.
We want to keep FP-neutral behavior for unlabeled-instance tolerance while restoring stop/closure learning signals.

## What Changes

- **BREAKING** Remove stop-neutral masking from the Stage-2 AB Channel-B contract (no runtime toggle; behavior is fixed by contract).
- Keep FP-neutral unchanged: unmatched predicted objects remain CE/geometry neutral.
- Re-enable Channel-B CE supervision on top-level close brace and `<|im_end|>` according to the unified target sequence.
- Update config/docs/specs/tests so the updated contract is explicit and reproducible.
- Refresh in-repo configs to the updated contract (no legacy stop-neutral knobs are supported).

## Capabilities

### New Capabilities
- None.

### Modified Capabilities
- `stage2-ab-training`: Channel-B SHALL supervise stop/closure tokens while retaining FP-neutral CE/geometry masking for unmatched predictions.

## Impact

- Affected code: Channel-B CE mask construction and related tests/telemetry.
- Affected docs/specs: Stage-2 AB contract, runbook, and metric interpretation guidance.
- Eval validity tradeoff: stronger stop prior vs. prior stop-neutral policy; FP-neutral protections remain in place.
- Validation focus: report training-time TP-hit efficiency and rollout-length/truncation indicators under the updated contract (no in-spec pass/fail gate).
- Migration note: resuming a run/checkpoint produced under stop-neutral masking is not supported for comparability; start a fresh run under the new objective.
