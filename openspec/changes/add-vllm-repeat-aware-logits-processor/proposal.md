## Why

Channel-B rollouts in vLLM server mode currently ignore the existing `repeat_terminate` contract that is available in HF mode. This causes long degenerate generations that frequently hit `max_new_tokens`, increase truncation/invalid parse rates, and reduce training-time TP efficiency per rollout-second.

## What Changes

- Add server-side vLLM repeat-aware (`repeat_terminate`) logits processing for rollout generation.
- Wire repeat-termination behavior from existing YAML config (`custom.extra.rollout_matching.repeat_terminate`) into vLLM rollout serving whenever YAML enables it (no new runtime knobs).
- On current vLLM V1-default stack, activate repeat-aware logic via rollout-server startup plugin injection (e.g., `swift rollout --external_plugins <repo-owned-plugin>`, not per-request logits-processor payloads).
- Keep generation batching intact by forcing EOS per offending sequence rather than aborting full batches.
- Preserve existing HF-side guard behavior and align semantics across HF and vLLM backends.
- Add/adjust diagnostics so runs can verify that repeat-aware termination is active and reducing runaway tails (telemetry is emitted via the neutral `src.metrics` payload contracts).

## Capabilities

### New Capabilities
- None.

### Modified Capabilities
- `rollout-matching-sft`: vLLM backend SHALL honor repeat-termination semantics equivalent to HF guard behavior.
- `stage2-ab-training`: Channel-B rollout generation in vLLM server mode SHALL support repeat-aware early termination without breaking batching.

## Impact

- Affected code: rollout backend plumbing (rollout-matching public contracts), Stage-2 AB vLLM server-mode rollout integration, and rollout metrics/tests.
- Reproducibility: YAML-first config remains source of truth; no new ad-hoc runtime knobs.
- Performance/validity: expected reduction in max-length tail events, truncation, and invalid object drops under vLLM rollouts.
- External dependency touchpoint: ms-swift / vLLM rollout server engine path.
- Migration: legacy “HF-only/ignored-by-vLLM” assumptions must be removed from configs/docs, and unsupported activation paths must fail fast.
