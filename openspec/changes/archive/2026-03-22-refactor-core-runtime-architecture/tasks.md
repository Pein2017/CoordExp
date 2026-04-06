## Workstream 0. Contract Freeze And Interface Snapshot

- [x] 0.1 Document the early-phase contract freeze in implementation notes for:
  - `stage2_ab.pipeline` schema and strict key behavior,
  - Stage-2 clean-prefix / triage / dead-anchor semantics,
  - trainer metric key families,
  - infer/eval artifact contracts,
  - reproducibility artifact files.
- [x] 0.2 Add or tighten explicit interface snapshots for the runtime payloads that move between batch prep and loss execution in the Stage-2 trainers.
- [x] 0.3 Bootstrap the brownfield execution-planning layer from the validated OpenSpec change so execution phases, context files, and later plans trace back to the same contract source. Every `.planning/` artifact must cite:
  - source change path,
  - source workstream/task ranges,
  - and its local execution purpose.
- [x] 0.4 Identify and preserve trainer-level compatibility adapters relied on by tests and monkeypatching, including:
  - `_rollout_many`
  - `_ensure_vllm_engine`
  - `src/trainers/stage2_two_channel/__init__.py`
  - `src/trainers/stage2_ab_training.py`
  - `src/trainers/stage2_ab/__init__.py`
  - `src/trainers/stage2_ab/executors.py`
  - `src/trainers/stage2_ab/scheduler.py`
- [x] 0.5 Capture a baseline verification report before the first extraction slice:
  - `conda run -n ms python -m pytest -q tests/test_coord_geometry_invariants.py`
  - `conda run -n ms python -m pytest -q tests/test_chat_template_regression.py`
  - `conda run -n ms python -m pytest -q tests/test_prompt_variants.py`
  - `conda run -n ms python -m pytest -q tests/test_stage2_ab_config_contract.py`
  - `conda run -n ms python -m pytest -q tests/test_training_config_strict_unknown_keys.py`
  - `conda run -n ms python -m pytest -q tests/test_stage2_ab_training.py`
  - `conda run -n ms python -m pytest -q tests/test_stage2_two_channel_training.py`
  - `conda run -n ms python -m pytest -q tests/test_stage2_rollout_aligned.py`
  - `conda run -n ms python -m pytest -q tests/test_stage2_rollout_import_boundaries.py`
  - `conda run -n ms python -m pytest -q tests/test_unified_infer_pipeline.py`
  - `conda run -n ms python -m pytest -q tests/test_detection_eval_output_parity.py`
  - `conda run -n ms python -m pytest -q tests/test_run_manifest_files.py`
  - `conda run -n ms python -m pytest -q tests/test_dependency_provenance.py`
  - `conda run -n ms python -m pytest -q tests/test_launcher_metadata_env.py`
  - `conda run -n ms python -m pytest -q tests/test_trainer_metrics_payload_contract.py`
- [x] 0.6 Emit named Phase-1 planning outputs under `.planning/phases/01-contract-freeze-and-execution-charter/`:
  - `01-CONTRACT-MATRIX.md`
  - `01-EXECUTION-CHARTER.md`
  - `01-COMPAT-ADAPTER-REGISTER.md`
  - `01-BASELINE-VERIFICATION.md`

## Workstream 1. Stage-2 Two-Channel Internal Boundary Extraction

- [x] 1.1 Introduce explicit internal types for Channel-A / Channel-B prepared payloads, including the metadata currently threaded through `_rollout_matching_meta`.
- [x] 1.2 Extract Channel-B clean-prefix target construction out of `src/trainers/stage2_two_channel.py` into dedicated package-local modules without changing semantics.
- [x] 1.3 Extract channel-specific objective execution and log projection out of `compute_loss` into a dedicated `objective_runner` layer.
- [x] 1.4 Keep `src/trainers/stage2_two_channel.py` as the public assembly surface, with the current `scheduler.py`, `executors.py`, compatibility wrapper, and legacy Stage-2 import shims preserved.
- [x] 1.5 Validate after each logical slice:
  - `conda run -n ms python -m pytest -q tests/test_stage2_ab_training.py`
  - `conda run -n ms python -m pytest -q tests/test_stage2_two_channel_training.py`
  - `conda run -n ms python -m pytest -q tests/test_stage2_ab_config_contract.py`
  - `conda run -n ms python -m pytest -q tests/test_training_config_strict_unknown_keys.py`

## Workstream 2. Channel-B Executor And Coordination Cleanup

- [x] 2.1 Extract Channel-B producer/consumer execution from `src/trainers/stage2_two_channel/executors.py` into a narrower execution helper or module.
- [x] 2.2 Extract DDP monitored-barrier and timeout policy into explicit coordination helpers so queue/runtime logic is no longer mixed with synchronization rules.
- [x] 2.3 Preserve current no-sync, packed-step weighting, timeout behavior, and legacy Stage-2 compatibility imports.
- [x] 2.4 Validate Channel-B runtime behavior with targeted tests:
  - `conda run -n ms python -m pytest -q tests/test_stage2_ab_training.py`
  - `conda run -n ms python -m pytest -q tests/test_stage2_two_channel_training.py`
  - `conda run -n ms python -m pytest -q tests/test_stage2_ab_ddp_phase_monitor_disable.py`
  - `conda run -n ms python -m pytest -q tests/test_stage2_ab_disable_average_tokens_across_devices.py`

## Workstream 3. Shared Rollout Runtime Extraction

- [x] 3.1 Extract shared rollout backend/decode lifecycle from `src/trainers/stage2_rollout_aligned.py` into a dedicated runtime package.
- [x] 3.2 Preserve trainer-facing adapter methods initially so existing tests and monkeypatches continue to work.
- [x] 3.3 Move vLLM local/server lifecycle and rollout dispatch behind a narrower trainer-to-runtime interface.
- [x] 3.4 Validate shared runtime parity:
  - `conda run -n ms python -m pytest -q tests/test_stage2_rollout_aligned.py`
  - `conda run -n ms python -m pytest -q tests/test_stage2_ab_training.py`
  - `conda run -n ms python -m pytest -q tests/test_stage2_ab_vllm_server_mode_smoke.py`
  - `conda run -n ms python -m pytest -q tests/test_stage2_rollout_import_boundaries.py`

## Workstream 4. Rollout-Aligned Target Builder And Evaluator Split

- [x] 4.1 Extract rollout-aligned train-time target construction from `_prepare_batch_inputs` into dedicated modules.
- [x] 4.2 Extract rollout-aligned evaluation flow from `evaluate` into an evaluator module while preserving metric and artifact behavior.
- [x] 4.3 Keep current rollout metric keys and eval artifact expectations stable during the first extraction pass.
- [x] 4.4 Validate rollout-aligned parity:
  - `conda run -n ms python -m pytest -q tests/test_stage2_rollout_aligned.py`
  - `conda run -n ms python -m pytest -q tests/test_detection_eval_output_parity.py`

## Workstream 5. Training Bootstrap And Manifest Authority Cleanup

- [x] 5.1 Move packing/bootstrap policy out of `src/sft.py` into focused bootstrap helpers without changing resolved runtime behavior.
- [x] 5.2 Consolidate pipeline-manifest projection behind a single bootstrap helper layer so typed config remains the normative authored source for Stage-2 pipeline structure.
- [x] 5.3 Move trainer assembly/injection and provenance writing into dedicated bootstrap helpers while preserving current artifacts:
  - `resolved_config.json`
  - `runtime_env.json`
  - `run_metadata.json`
  - `logging.jsonl`
- [x] 5.4 Validate bootstrap and provenance parity:
  - `conda run -n ms python -m pytest -q tests/test_stage2_ab_config_contract.py`
  - `conda run -n ms python -m pytest -q tests/test_training_config_strict_unknown_keys.py`
  - `conda run -n ms python -m pytest -q tests/test_run_manifest_files.py`
  - `conda run -n ms python -m pytest -q tests/test_dependency_provenance.py`
  - `conda run -n ms python -m pytest -q tests/test_launcher_metadata_env.py`
  - `conda run -n ms python -m pytest -q tests/test_trainer_metrics_payload_contract.py`

## Workstream 6. Inference Backend Split

- [x] 6.1 Split `src/infer/engine.py` into backend adapters plus output/artifact orchestration.
- [x] 6.2 Preserve strict preflight behavior, current prompt rendering, existing JSONL/summary/token-trace artifact fields, and the public `src.infer.engine` API consumed by `src/infer/pipeline.py`. Compatibility-preserving exports or shims are acceptable while internals move.
- [x] 6.3 Validate infer parity:
  - `conda run -n ms python -m pytest -q tests/test_unified_infer_pipeline.py`
  - `conda run -n ms python -m pytest -q tests/test_confidence_postop.py`
  - `conda run -n ms python -m pytest -q tests/test_bbox_confidence.py`

## Workstream 7. Detection Evaluator Split

- [x] 7.1 Split top-level detection evaluation orchestration and artifact/report writing into focused layers without changing output semantics; deeper ingest / COCO / F1-ish helpers may remain in `src/eval/detection.py` during this pass.
- [x] 7.2 Preserve:
  - `metrics.json`
  - `per_image.json`
  - `per_class.csv`
  - `coco_gt.json`
  - `coco_preds.json`
  - `vis_resources/gt_vs_pred.jsonl`
- [x] 7.3 Preserve the public `src.eval.detection` API boundary consumed by:
  - `src/infer/pipeline.py`
  - `src/callbacks/detection_eval.py`
  - `scripts/evaluate_detection.py`
  Compatibility-preserving exports or shims are acceptable while internals move.
- [x] 7.4 Validate eval parity:
  - `conda run -n ms python -m pytest -q tests/test_detection_eval_output_parity.py`
  - `conda run -n ms python -m pytest -q tests/test_detection_eval_ingestion_diagnostics.py`
  - `conda run -n ms python -m pytest -q tests/test_confidence_postop.py`
  - `conda run -n ms python -m pytest -q tests/test_bbox_confidence.py`

## Workstream 8. Final Cross-Cut Verification And Governance Check

- [x] 8.1 Run the full targeted verification bundle after the last internal extraction slice:
  - `conda run -n ms python -m pytest -q tests/test_coord_geometry_invariants.py`
  - `conda run -n ms python -m pytest -q tests/test_chat_template_regression.py`
  - `conda run -n ms python -m pytest -q tests/test_prompt_variants.py`
  - `conda run -n ms python -m pytest -q tests/test_stage2_ab_config_contract.py`
  - `conda run -n ms python -m pytest -q tests/test_training_config_strict_unknown_keys.py`
  - `conda run -n ms python -m pytest -q tests/test_stage2_ab_training.py`
  - `conda run -n ms python -m pytest -q tests/test_stage2_two_channel_training.py`
  - `conda run -n ms python -m pytest -q tests/test_stage2_rollout_aligned.py`
  - `conda run -n ms python -m pytest -q tests/test_stage2_rollout_import_boundaries.py`
  - `conda run -n ms python -m pytest -q tests/test_stage2_ab_vllm_server_mode_smoke.py`
  - `conda run -n ms python -m pytest -q tests/test_unified_infer_pipeline.py`
  - `conda run -n ms python -m pytest -q tests/test_detection_eval_output_parity.py`
  - `conda run -n ms python -m pytest -q tests/test_detection_eval_ingestion_diagnostics.py`
  - `conda run -n ms python -m pytest -q tests/test_confidence_postop.py`
  - `conda run -n ms python -m pytest -q tests/test_bbox_confidence.py`
  - `conda run -n ms python -m pytest -q tests/test_run_manifest_files.py`
  - `conda run -n ms python -m pytest -q tests/test_dependency_provenance.py`
  - `conda run -n ms python -m pytest -q tests/test_launcher_metadata_env.py`
  - `conda run -n ms python -m pytest -q tests/test_trainer_metrics_payload_contract.py`
- [x] 8.2 Confirm stable contracts remained unchanged; if not, add follow-on OpenSpec deltas before landing those behavior changes.
- [x] 8.3 Update operator-facing docs only where current entrypoints, runbooks, or architecture pointers changed.
- [x] 8.4 Record exact reproducibility handles for the refactor program:
  - config path used for any smoke verification,
  - run name,
  - seed,
  - and expected output artifacts.
