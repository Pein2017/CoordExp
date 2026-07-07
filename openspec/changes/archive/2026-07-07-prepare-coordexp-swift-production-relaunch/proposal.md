# Prepare CoordExp-Swift Production Relaunch

## Why

The rebuilt CoordExp-Swift training stack has passed its V1 rebuild gates, but
the production relaunch now needs a focused readiness contract. The relaunch is
not a new objective or model-behavior experiment: it promotes several tiny but
fundamental infrastructure factors into explicit contract evidence before an
8-GPU production run is trusted.

The immediate target is the Qwen3-VL 2B desc-first geo-sorted pure-CE
language-only DoRA production profile with DoRA rank 16, alpha 32,
`training.effective_batch_size: 64`, `warmup_ratio: 0.1`, seed 17, four
epochs, packed length 12000, and Accelerate on 8 GPUs.

## What Changes

- Add a production-relaunch readiness contract for seed control, resolved
  production config evidence, scheduler warmup derivation, runtime-batch
  derivation, packing-cache evidence, FA2 proof policy, smoke-to-production
  promotion, and final checkpoint acceptance.
- Require the configured seed to be applied before Qwen load, fresh DoRA
  initialization, selected embedding setup, optimizer setup, and runtime setup.
- Treat deterministic supervised packing-cache reuse as approved production
  training infrastructure, while keeping hidden-state, KV, runtime feature
  caches, and DeepSpeed production support gated out of this relaunch.
- Add an artifact-backed smoke ladder before the production relaunch.
- Add the new r16/a32 EBS64 warmup0p1 production and smoke configs.

## Impact

- Operators get a concrete relaunch checklist with stop gates instead of
  launch folklore.
- Runtime determinism is strengthened without adding a new public config knob.
- The completed `rebuild-coordexp-swift-training-infra` OpenSpec remains
  historical baseline authority; this change is a follow-on readiness layer.
