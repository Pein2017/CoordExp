## Why

Stage-2 online vLLM rollout needs to run from the same effective policy as the
learner while preserving the research contract that the base model remains
frozen and checkpoints save only adapters. The current full-sync path can send
CoordExp-only `modules_to_save` tensors such as `coord_offset_adapter.*` to
native vLLM, which fails with unknown weights; simply filtering those tensors
would make rollout semantics diverge from HF learner semantics.

## What Changes

- Define vLLM full-sync as a runtime materialized ordinary-weight snapshot of
  the learner's effective inference policy, not as a permanent base-model merge.
- Preserve adapter-only training and saving: LoRA/DoRA and token-row adapters
  remain adapter checkpoint state, and base model parameters remain frozen.
- Materialize CoordExp token-row adapters into the sync snapshot before vLLM
  `load_weights()` by adding row deltas to `embed_tokens.weight` and, when
  present or required, `lm_head.weight`.
- Remove learner-only adapter/module keys from the vLLM-bound sync snapshot
  after they have been materialized.
- Fail fast when active token-row adapter state cannot be materialized safely;
  silent skip is not allowed.
- Keep current Stage-2 vLLM server rollout on
  `rollout_matching.vllm.sync.mode=full` and
  `rollout_matching.vllm.enable_lora=false` for this change. True
  adapter-only vLLM sync with token-row runtime updates is deferred.
- Add targeted tests and a vLLM gate run before treating the path as usable for
  the 6-server / 2-learner online residual-trie experiment.

## Capabilities

### New Capabilities

- `vllm-full-sync-adapter-row-materialization`: Defines the runtime contract for
  converting learner adapter state into a vLLM-loadable ordinary-weight
  snapshot without changing training or checkpoint-save semantics.

### Modified Capabilities

- `coord_offset`: Clarifies that active coord/token-row offsets MUST be
  materialized into ordinary embedding/head rows for native vLLM full-sync
  instead of being skipped or sent as `coord_offset_adapter.*` tensors.
- `stage2-ab-training`: Defines Stage-2 vLLM rollout full-sync semantics for
  adapter-backed learners and keeps adapter-only save semantics intact.
- `silent-failure-policy`: Requires learner-side vLLM sync preparation
  failures to be observable and fail-fast, with DDP-rank-symmetric
  termination; stronger server-side load acknowledgement is deferred.

## Impact

- Affected code:
  - `src/trainers/rollout_runtime/vllm_server.py`
  - `src/trainers/rollout_runtime/vllm_engine.py`
  - likely new helper under `src/trainers/rollout_runtime/`
  - `src/tokens/row_offsets.py` only if a reusable discovery helper is needed
  - config/schema tests around `rollout_matching.vllm`
- Affected tests:
  - `tests/test_stage2_rollout_runtime.py`
  - coord-token / row-offset tests under `tests/coord_tokens/` or
    `tests/tokens/`
  - strict unknown-key / vLLM sync tests
- Affected operational surfaces:
  - Stage-2 vLLM server launch via `scripts/train_stage2.sh`
  - online residual-trie configs under `configs/stage2_two_channel/smoke/`
  - vLLM gate log and artifact roots under `temp/` and `output/stage2_ab/`
- No production dependency changes are intended.
- No upstream HF model file or native vLLM model file may be edited.

## Non-Goals

- Do not implement true vLLM adapter-only sync for multimodal LoRA/DoRA plus
  token-row adapters in this change.
- Do not permanently merge adapters into the base model for training.
- Do not change adapter checkpoint format or stop saving `coord_offset_adapter`
  through PEFT `modules_to_save`.
- Do not make `rollout_matching.vllm.enable_lora=true` the default Stage-2
  path when an active CoordExp token-row adapter exists.
- Do not make model-quality or val200 improvement an OpenSpec validity gate.
