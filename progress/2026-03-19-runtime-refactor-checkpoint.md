# Runtime Refactor Checkpoint

Date: 2026-03-19
Worktree: `/data/CoordExp/.worktrees/refactor-core-runtime-architecture`
Branch: `change/refactor-core-runtime-architecture`

## What Landed

Late 2026-03-19 completion pass closed the remaining phases that were actually implemented in this change:
- `src/trainers/rollout_runtime/dispatch.py` and `src/trainers/rollout_runtime/vllm_server.py` now cover the shared rollout runtime seam while `src/trainers/stage2_rollout_aligned.py` keeps compatibility adapters intact.
- `src/trainers/rollout_aligned_targets.py` now owns rollout-aligned target construction helpers, with `_build_labels_and_coord_targets_for_sample` and `_build_labels_and_coord_targets_for_batch` preserved as stage2-local wrappers.
- `src/trainers/rollout_aligned_evaluator.py` now owns the evaluation reduction / COCO / monitor-dump / callback tail behind `finalize_rollout_aligned_evaluation`, and `RolloutMatchingSFTTrainer.evaluate` delegates into it while still passing the stage2-local monkeypatch surfaces explicitly.
- `src/bootstrap/pipeline_manifest.py` and `src/bootstrap/trainer_setup.py` now own pipeline-manifest projection plus trainer callback/class assembly, while `src/sft.py` preserves `_build_pipeline_manifest` and `resolve_trainer_cls` as compatibility surfaces. Typed config remains the authored source; variant-default projection is still helper-owned in this pass.
- `src/infer/artifacts.py` now owns infer artifact path resolution plus summary/resolved-meta payload assembly, and `src/eval/orchestration.py` now owns top-level detection evaluation/save orchestration while `src/eval/detection.py` remains the public API surface and still contains the deeper ingest / COCO / F1-ish helpers.
- No operator-facing entrypoints changed in this pass, so no follow-on docs delta was required to preserve current runbook usage.

Earlier 2026-03-19 foundation slices that fed the completion pass included:
- explicit payload/types moved into `src/trainers/stage2_two_channel/types.py`
- Channel-B target construction moved behind:
  - `src/trainers/stage2_two_channel/target_builder.py`
  - `src/trainers/stage2_two_channel/rollout_views.py`
  - `src/trainers/stage2_two_channel/objective_runner.py`
- Channel-B executor/coordination moved behind:
  - `src/trainers/stage2_two_channel/coordination.py`
  - `src/trainers/stage2_two_channel/executors.py`
- `src/trainers/rollout_runtime/vllm_config.py`
- `src/trainers/rollout_runtime/vllm_engine.py`
- `src/trainers/rollout_runtime/vllm_server.py`
- `src/trainers/rollout_runtime/vllm_infer.py`
- `PreparedVLLMServerRollout`, `build_vllm_server_infer_requests`, and `prepare_vllm_server_rollout` now move server rollout request building, seed planning, and per-rank cap orchestration behind `src/trainers/rollout_runtime/vllm_server.py`
- `src/trainers/stage2_rollout_aligned.py::_rollout_many_vllm_server` is now a thinner trainer-facing adapter over runtime preparation + dispatch
- `src/trainers/rollout_runtime/dispatch.py` now owns shared rollout backend dispatch for `_rollout_many`, `_rollout_many_vllm`, and `_rollout_many_vllm_traced`
- `src/bootstrap/run_metadata.py` now owns run-metadata/provenance composition and writing
- `src/sft.py` keeps compatibility wrappers for `_collect_dependency_provenance`, `_collect_launcher_metadata_from_env`, and `_attach_encoded_sample_cache_run_metadata`, but the rank-0 `run_metadata.json` path is no longer assembled inline
- `src/infer/backends.py` now owns backend-specific infer batching for HF and vLLM while `src/infer/engine.py` stays the public engine surface
- `src/eval/artifacts.py` now owns detection artifact/report writing while `src/eval/detection.py` stays the public evaluator surface

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

Additional green bundles after later slices:
- `PYTHONPATH=. conda run -n ms python -m pytest -q tests/test_stage2_rollout_aligned.py tests/test_stage2_ab_training.py tests/test_stage2_ab_vllm_server_mode_smoke.py tests/test_stage2_rollout_import_boundaries.py tests/test_stage2_ab_prompt_alignment_contract.py tests/test_prepare_samples_for_rollout_vllm_multimodal.py`
- `PYTHONPATH=. conda run -n ms python -m pytest -q tests/test_run_manifest_files.py tests/test_dependency_provenance.py tests/test_launcher_metadata_env.py tests/test_encoded_sample_cache_runtime_config.py tests/test_stage1_static_packing_runtime_config.py tests/test_run_metadata_file.py`
- `PYTHONPATH=. conda run -n ms python -m pytest -q tests/test_infer_batch_decoding.py tests/test_unified_infer_pipeline.py tests/test_confidence_postop.py tests/test_bbox_confidence.py`
- `PYTHONPATH=. conda run -n ms python -m pytest -q tests/test_detection_eval_output_parity.py tests/test_detection_eval_ingestion_diagnostics.py tests/test_confidence_postop.py tests/test_bbox_confidence.py`

## Final Validation

Late-pass targeted parity bundles that stayed green:
- `PYTHONPATH=. conda run -n ms python -m pytest -q tests/test_stage2_rollout_aligned.py tests/test_packed_labels_and_coord_targets.py tests/test_stage2_ab_training.py tests/test_stage2_ab_vllm_server_mode_smoke.py tests/test_stage2_rollout_import_boundaries.py tests/test_detection_eval_output_parity.py`
  - Result: `172 passed, 2 skipped`
- `PYTHONPATH=. conda run -n ms python -m pytest -q tests/test_stage2_ab_config_contract.py tests/test_training_config_strict_unknown_keys.py tests/test_run_manifest_files.py tests/test_dependency_provenance.py tests/test_launcher_metadata_env.py tests/test_trainer_metrics_payload_contract.py tests/test_stage1_static_packing_runtime_config.py tests/test_unified_infer_pipeline.py tests/test_infer_batch_decoding.py tests/test_confidence_postop.py tests/test_bbox_confidence.py tests/test_detection_eval_output_parity.py tests/test_detection_eval_ingestion_diagnostics.py`
  - Result: `201 passed`
- `PYTHONPATH=. conda run -n ms python -m pytest -q tests/test_coord_geometry_invariants.py tests/test_chat_template_regression.py tests/test_prompt_variants.py tests/test_stage2_ab_config_contract.py tests/test_training_config_strict_unknown_keys.py tests/test_stage2_ab_training.py tests/test_stage2_two_channel_training.py tests/test_stage2_rollout_aligned.py tests/test_stage2_rollout_import_boundaries.py tests/test_stage2_ab_vllm_server_mode_smoke.py tests/test_unified_infer_pipeline.py tests/test_detection_eval_output_parity.py tests/test_detection_eval_ingestion_diagnostics.py tests/test_confidence_postop.py tests/test_bbox_confidence.py tests/test_run_manifest_files.py tests/test_dependency_provenance.py tests/test_launcher_metadata_env.py tests/test_trainer_metrics_payload_contract.py`
  - Result: `421 passed, 4 skipped`
- `openspec validate refactor-core-runtime-architecture --type change --strict --no-interactive`
  - Result: `Change 'refactor-core-runtime-architecture' is valid`

## Current Focus

The refactor program is complete for the scoped OpenSpec change.

Contract stability notes:
- trainer-facing rollout adapters remain preserved
- rollout-aligned target/eval metric keys remained unchanged
- infer artifact names and fields remained unchanged
- detection evaluator output artifacts remained unchanged
- no public CLI or runbook entrypoint changed, so no follow-on OpenSpec delta or operator-doc edit was needed

Reproducibility handles recorded for the completion pass:
- worktree: `/data/CoordExp/.worktrees/refactor-core-runtime-architecture`
- branch: `change/refactor-core-runtime-architecture`
- config path used for smoke-style verification: none; completion used targeted pytest parity bundles rather than a new runtime smoke config
- run name: not applicable for the code-only parity bundle
- seed: deterministic test-owned seeds as encoded in the existing test fixtures
- expected preserved artifacts: `resolved_config.json`, `runtime_env.json`, `run_metadata.json`, `logging.jsonl`, `gt_vs_pred.jsonl`, `pred_token_trace.jsonl`, `summary.json`, `metrics.json`, `per_image.json`, `per_class.csv`, `coco_gt.json`, `coco_preds.json`, `vis_resources/gt_vs_pred.jsonl`

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
