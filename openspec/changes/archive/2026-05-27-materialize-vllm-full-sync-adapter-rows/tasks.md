# Tasks

Status note (2026-05-26): this completed change is historical/deferred for
active unified Stage-2 rollout-correction server training. Do not use its
native full-sync evidence or `src/trainers/rollout_runtime/*` implementation
locations as the active server sync route for `unify-inference-runtime`.
Unified Stage-2 server training uses adapter sync plus CoordExp coord-row
updates unless a later OpenSpec explicitly revives native full-sync.

These tasks are draft approval gates. Do not implement until the user approves
the OpenSpec and super-power plan.

## 1. Spec And Plan Review

- [x] 1.1 Review this OpenSpec change with subagents for contract clarity,
  implementation feasibility, and eval-validity risks.
- [x] 1.2 Revise proposal/design/spec/tasks from review findings.
- [x] 1.3 Validate the OpenSpec change with `openspec status --change
  materialize-vllm-full-sync-adapter-rows`.
- [x] 1.4 Wait for explicit user approval before editing production code.

## 2. Sync Materialization Helper

- [x] 2.1 Add a focused helper for vLLM full-sync state-dict materialization.
- [x] 2.2 Discover active coord/token-row adapter state after PEFT wrapping.
- [x] 2.3 Materialize tied-head and untied-head row deltas into cloned ordinary
  tensors.
- [x] 2.4 Fail fast if coord/token-row adapter keys are present but no active
  adapter can be discovered.
- [x] 2.5 Remove learner-only forbidden key families and validate the
  vLLM-bound state dict.
- [x] 2.6 Emit compact diagnostics for materialization applied/skipped and row
  count.

## 3. Runtime Integration

- [x] 3.1 Call the helper from server-mode full-sync before
  `_vllm_server_update_state_dict`.
- [x] 3.2 Call the helper from colocate full-sync before
  `engine.inner_model.load_weights`.
- [x] 3.3 Preserve adapter-only checkpoint saving and ensure temporary PEFT
  merge/unmerge leaves no lasting live-model mutation.
- [x] 3.4 Historical original full-sync path preserved
  `rollout_matching.vllm.enable_lora=false` for token-row adapters; this is
  superseded for active unified Stage-2 server rollout by adapter sync plus
  coord-row updates.

## 4. Tests

- [x] 4.1 Add toy parity tests for hook behavior versus materialized ordinary
  weights.
- [x] 4.2 Add row-level tied-head and untied-head assertions, including
  non-coordinate rows unchanged.
- [x] 4.3 Add state-dict hygiene tests for forbidden key removal across
  `modules_to_save.<adapter>`, `original_module`, CoordExp adapter keys, and
  LoRA/DoRA residue keys.
- [x] 4.4 Add fail-fast tests for missing embedding/head rows and shape or vocab
  mismatches.
- [x] 4.5 Add fail-fast tests for coord/token-row keys without a discoverable
  active adapter.
- [x] 4.6 Add tied-head missing-`lm_head.weight` tests that distinguish
  confirmed tied embeddings from unknown/untied embeddings.
- [x] 4.7 Add no-op tests when no coord/token-row adapter exists and assert the
  input mapping is not mutated.
- [x] 4.8 Add/update Stage-2 vLLM sync config/runtime tests for both server and
  colocate full-sync paths.

## 5. Documentation

- [x] 5.1 Update the Stage-2 runbook after implementation lands to document
  adapter-only save plus materialized full-sync runtime semantics.
- [x] 5.2 Mention that true vLLM adapter-only row-sync is a future performance
  project, not the current default.

## 6. Verification

- [x] 6.1 Run targeted unit tests with `conda run -n ms python -m pytest ...`.
- [x] 6.2 Run cfg-only for
  `configs/stage2_two_channel/smoke/compact_full_et_rmp_ce_ckpt3664_hf_coco80_view_train128_val64_4steps_online_residual_trie_tail_append_zero_fp_lr1e5_decode4_vllm_6srv2lr_gate.yaml`.
- [x] 6.3 Run the vLLM gate after approval and implementation.
- [x] 6.4 Report GPU memory, sync success/failure, first rollout status,
  invalid counters, and any OOM signal.

Evidence:
- Targeted tests:
  `conda run -n ms python -m pytest tests/tokens/test_vllm_sync_materialization.py tests/test_stage2_rollout_runtime.py tests/test_training_config_strict_unknown_keys.py -q`
  passed with `205 passed in 1.58s`. This is sync-materialization evidence,
  not evidence for the unified runtime generated-token logprob trace contract.
- OpenSpec strict validation:
  `openspec validate materialize-vllm-full-sync-adapter-rows --strict`
  passed.
- cfg-only for the vLLM 6-server / 2-learner gate exited with
  `status=ok`.
- Runtime vLLM 6:2 gate completed without OOM or vLLM unknown-key failure:
  historical output root
  `output/stage2_ab/smoke/coco80_view_online_residual_trie_train128_val64/tail_append_zero_fp_lr1e5_decode4_vllm6srv2lr_gate/.../v3-20260523-183336`.
  The log includes `materialized coord_offset_adapter rows for vLLM full-sync:
  rows=1002`, `rollout/backend_vllm=1`, zero invalid rollouts, zero parse
  truncation, final eval recall `0.53409091`, eval FN total `205`, and no
  CUDA OOM.
- Adapter-save gate completed without OOM:
  historical output root
  `output/stage2_ab/smoke/coco80_view_online_residual_trie_train128_val64/tail_append_zero_fp_lr1e5_decode4_vllm6srv2lr_save_gate/.../v1-20260523-184751`.
  Checkpoint `checkpoint-1` is 38M and contains `adapter_model.safetensors`,
  `adapter_config.json`, and `modules_to_save=["coord_offset_adapter"]`, with
  no full-model shard files.
