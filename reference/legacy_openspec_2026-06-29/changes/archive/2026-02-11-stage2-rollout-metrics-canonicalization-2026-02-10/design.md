# Design: Stage-2 Rollout Metrics Canonicalization

## Context

The Stage-2 stack reports rollout telemetry from two paths:
1. rollout-matching payload metrics (`rollout_matching_sft`), and
2. Stage-2 AB pending logs merged at optimizer-step boundaries.

In async + gradient-accumulation runs, some rollout config keys are present only for specific micro-steps and are currently divided by total micro-step count, yielding misleading fractional values.

## Goals

- Keep only high-signal scalar metrics in train logs.
- Improve semantic clarity for decode-related metrics.
- Eliminate aggregation artifacts for sparse gauges.

## Non-Goals

- No training-objective or matching-contract changes.
- No removal of legacy keys in this change.

## Data Flow

1. **Per-micro metric collection**
   - Stage-2 AB collects `batch_metrics` and stage-level losses.
2. **Pending accumulation**
   - `_PendingStage2Log` stores per-key sums and per-key observation counts.
3. **Step finalization**
   - Loss/ratio/gauge keys are averaged by key-presence count.
   - Counter keys remain summed.
4. **Rollout payload canonicalization**
   - Emit canonical decode-count metrics (`decode_non_beam_count`, `decode_beam_count`) plus legacy aliases.
5. **Artifacts**
   - `logging.jsonl` and TensorBoard now expose clearer canonical keys while preserving historical aliases.

## Compatibility

- Legacy keys removed from training logs are intentionally unsupported.
- Docs explicitly define the new minimal key contract.
- Consumers must migrate to canonical keys.

## Risks and Mitigations

- **Risk:** dashboards depending on removed legacy keys break.
  - **Mitigation:** publish the minimal key list and migrate dashboards in one pass.
- **Risk:** tests assume legacy-only keys.
  - **Mitigation:** keep legacy keys unchanged; add/adjust tests for canonical keys and aggregation semantics.
