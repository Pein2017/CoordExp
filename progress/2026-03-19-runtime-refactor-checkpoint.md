# Runtime Refactor Checkpoint

Date: 2026-03-19
Worktree: `/data/CoordExp/.worktrees/refactor-core-runtime-architecture`
Branch: `change/refactor-core-runtime-architecture`

## What Landed

Phase 2 is now effectively completed in code:
- explicit payload/types moved into `src/trainers/stage2_two_channel/types.py`
- Channel-B target construction moved behind:
  - `src/trainers/stage2_two_channel/target_builder.py`
  - `src/trainers/stage2_two_channel/rollout_views.py`
  - `src/trainers/stage2_two_channel/objective_runner.py`
- Channel-B executor/coordination moved behind:
  - `src/trainers/stage2_two_channel/coordination.py`
  - `src/trainers/stage2_two_channel/executors.py`

Phase 3 has started in code:
- `src/trainers/rollout_runtime/vllm_config.py`
- `src/trainers/rollout_runtime/vllm_engine.py`
- `src/trainers/rollout_runtime/vllm_server.py`
- `src/trainers/rollout_runtime/vllm_infer.py`

Recent grouped commits:
- `285fa3f` `refactor(stage2): extract nonpipeline channel-b loop`
- `1d82e2d` `refactor(stage2): extract step-mode accumulation helper`
- `b8cddbd` `refactor(stage2): extract channel-b pipeline loop`
- `b6f7765` `refactor(rollout): extract vllm config resolution`
- `5caac6a` `refactor(rollout): extract vllm engine creation`
- `2215ef4` `refactor(rollout): extract vllm server client lifecycle`
- `4cf4eb4` `refactor(rollout): extract vllm lifecycle helpers`
- `8455cf5` `refactor(rollout): extract vllm sync delegates`
- `0771ce5` `refactor(rollout): extract vllm server sync helpers`
- `2762b0d` `refactor(rollout): extract vllm runtime helpers`
- `cc171f0` `refactor(rollout): extract vllm infer delegates`
- `f70d60c` `refactor(rollout): extract vllm server infer helpers`
- `36eb09e` `refactor(rollout): extract vllm colocate rollout helper`
- `80171b7` `refactor(rollout): extract vllm server dispatch helpers`
- `2215ef4` `refactor(rollout): extract vllm server client lifecycle`

## Validation Status

Phase 2 / executor bundle that stayed green:
- `conda run -n ms python -m pytest -q tests/test_stage2_ab_training.py`
- `conda run -n ms python -m pytest -q tests/test_stage2_two_channel_training.py`
- `conda run -n ms python -m pytest -q tests/test_stage2_ab_ddp_phase_monitor_disable.py`
- `conda run -n ms python -m pytest -q tests/test_stage2_ab_disable_average_tokens_across_devices.py`

Phase 3 / rollout-runtime bundle that stayed green:
- `conda run -n ms python -m pytest -q tests/test_stage2_rollout_aligned.py`
- `conda run -n ms python -m pytest -q tests/test_stage2_ab_training.py`
- `conda run -n ms python -m pytest -q tests/test_stage2_ab_vllm_server_mode_smoke.py`
- `conda run -n ms python -m pytest -q tests/test_stage2_rollout_import_boundaries.py`

## Current Focus

Phase 3 is in progress.

The rollout-runtime package now owns:
- vLLM config resolution
- colocated engine creation/shutdown
- vLLM server client/communicator lifecycle
- vLLM server sync/update helpers
- vLLM TP-group infer helper
- vLLM colocate rollout helper
- vLLM server infer/dispatch helpers

The next natural seam is still in `src/trainers/stage2_rollout_aligned.py`:
- the remaining `*_rollout_many_*` server dispatch/metadata/orchestration code
- then the broader shared rollout dispatch/runtime interface

## Handoff Notes

- Preserve trainer-facing adapters while internals move:
  - `_rollout_many`
  - `_ensure_vllm_engine`
  - `_ensure_vllm_server_client`
  - `_ensure_vllm_server_communicator_rank0`
  - `_shutdown_vllm_server_client`
  - `_shutdown_vllm_colocate_engine`
  - `_sync_vllm_server_rollout_model_if_needed`
  - `_sync_vllm_server_full_weights`
  - `_vllm_server_update_state_dict`
  - `_vllm_infer_tp_group`
  - `_rollout_many_vllm_colocate`
  - `_rollout_many_vllm_server`
- Preserve Stage-2 monkeypatch/import compatibility in the two-channel path:
  - `parse_rollout_for_matching`
  - `points_from_coord_tokens`
  - `hungarian_match_maskiou`
  - `_build_canonical_prefix_text_data`
  - `_build_canonical_prefix_data`
  - `_build_dead_anchor_suppression_targets`
  - `_compute_duplicate_diagnostics`
  - `_sequential_dedup_bbox_objects`
