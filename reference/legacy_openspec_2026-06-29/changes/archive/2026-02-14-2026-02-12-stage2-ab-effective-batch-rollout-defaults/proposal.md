## Why

Stage2-AB currently exposes several overlapping rollout batch-size knobs (generate/infer/decode) whose effective behavior depends on world-size, learner topology, and packing mode. This makes runs hard to scale, easy to misconfigure, and unnecessarily slow when rollout GPUs have ample memory but generation calls are forced into small or poorly balanced chunks.

We need a single, reproducible contract where `training.effective_batch_size` is the highest-truth demand signal (raw rollouts per optimizer step), while rollout servers satisfy that demand evenly under an explicit per-GPU generation-call cap.

## What Changes

- Standardize Stage2-AB Channel-B to a single execution pathway: **step-budgeted learn-to-completion with dynamic packing** (this becomes the only supported mode for Stage2-AB configs under `configs/stage2_ab/**`).
- Define `training.effective_batch_size` as the single source of truth for **global raw rollouts per optimizer step** (across learner ranks) and require it to be divisible by `per_device_train_batch_size × learner_world_size` (no ceil overshoot).
- Derive rollout request chunking automatically from rollout-server world size and learner world size, with a per-rollout-GPU **generation-call cap** controlled by `custom.extra.rollout_matching.decode_batch_size` (Stage2-AB YAML default under `configs/stage2_ab/**`: `4`).
- **BREAKING**: Remove legacy/overlapping knobs and modes:
  - Remove `stage2_ab.channel_b.mode` (including `micro` and `async`).
  - Remove `stage2_ab.channel_b.async` (and any `stage2_ab.channel_b.async.*` sub-config).
  - Remove `stage2_ab.channel_b.rollouts_per_step` (always derived from `training.effective_batch_size`).
  - Remove `stage2_ab.channel_b.enable_pipeline` (pipeline overlap becomes an internal implementation detail for vLLM server mode).
  - Remove `stage2_ab.channel_b.rollout_decode_batch_size` (use `custom.extra.rollout_matching.decode_batch_size`).
  - Remove `custom.extra.rollout_matching.rollout_generate_batch_size` and `custom.extra.rollout_matching.rollout_infer_batch_size` (use `decode_batch_size`).
  - Remove `custom.extra.rollout_matching.post_rollout_pack_scope: window` (standardize on dynamic `micro` scope).
  - Any config using removed keys MUST fail fast with actionable migration guidance.

## Capabilities

### New Capabilities

<!-- None: this change standardizes defaults and precedence within existing capabilities. -->

### Modified Capabilities

- `stage2-ab-training`: Define default Channel-B mode/packing behavior and formally specify how `training.effective_batch_size` maps to raw rollouts per optimizer step in multi-learner settings.
- `rollout-matching-sft`: Specify rollout request batching/derivation rules (including per-rollout-GPU generation-call cap) under a single decode batching knob that applies consistently across HF and vLLM backends.

## Impact

- Configs:
  - Update defaults across `configs/stage2_ab/**` (Channel-B mode, packing scope, and rollout batching knobs).
- Training/executors/scheduler:
  - Ensure the requested `effective_batch_size` is preserved through config loading and used for rollout budgeting when no explicit `rollouts_per_step` is provided.
- Rollout client/server interaction:
  - Add/standardize derived request chunk sizing using rollout-server world size (`/get_world_size/`) to honor a per-device decode cap without hardcoding a specific topology (e.g., 6 rollout GPUs / 2 learner GPUs).
- Docs/specs:
  - Update Stage2 runbook and metrics docs to remove async-mode guidance and document the standardized step-budgeted pipeline defaults.
- Reproducibility & eval:
  - Step-budgeted dynamic packing changes the default stepping behavior; this may affect throughput and (potentially) training dynamics compared to fixed windowing. The change is config-versioned and documented via updated specs.
