# Proposal: Stage-2 Rollout Metrics Minimalization + Canonicalization

## Why

Stage-2 AB logs currently expose metric-key ambiguity in `logging.jsonl`:
- duplicated/alias keys with unclear ownership (`rollout/match_rate` vs `rollout/recall`, `rollout/decode_mode_*` vs `rollout/decode_*`),
- sparse Channel-B rollout config gauges diluted by gradient-accumulation averaging (`0.03125`-style artifacts),
- unclear canonical naming for non-beam decode counts.

These issues reduce dashboard interpretability and slow error analysis, especially for FP/format diagnostics.

## What Changes

This change introduces a contract cleanup (breaking at metric-key level):
- keep only a minimal high-signal metric set,
- use canonical names for decode-count keys,
- fix Stage-2 AB pending-log aggregation semantics so sparse gauge keys are averaged over key presence instead of total micro-steps,
- remove backward-compat aliases and verbose/duplicated gauges from training logs.

## Capabilities

### 1) Canonical decode-count metrics without aliases
- Keep only canonical non-beam/beam decode count keys.
- Remove legacy decode alias keys from training logs.

### 2) Micro-step aggregation semantics for sparse gauges
- Stage-2 pending logger averages gauge-like keys over key-observation count (not total micro-steps) to avoid GAS dilution.
- Preserve existing summed semantics for explicit counters and losses/ratios behavior where already expected.

### 3) Documentation alignment for metric semantics
- Update docs to distinguish:
  - canonical metrics,
  - compatibility aliases,
  - interpretation guidance for decode-mode sentinels under vLLM.

## Impact

- Backward compatibility: intentionally not preserved for removed metric keys.
- Reproducibility/eval validity: improved metric interpretability; no objective changes.
- Scope: logging/telemetry contract and docs only.
