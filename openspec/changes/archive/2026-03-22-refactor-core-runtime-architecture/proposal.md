## Why

CoordExp's current runtime-critical code has reached the point where architectural coupling, not missing functionality, is the main engineering bottleneck.

The strongest pressure is in the Stage-2 training stack:

- `src/trainers/stage2_rollout_aligned.py` is a true god module that mixes rollout backend lifecycle, vLLM local/server orchestration, target construction, monitoring, post-rollout packing, teacher-forced loss execution, and production-style evaluation.
- `src/trainers/stage2_two_channel.py` is on a healthier path because it already has package seams in `scheduler.py`, `executors.py`, and the import-compat wrapper, but the remaining hotspots still braid runtime setup, clean-prefix domain logic, objective execution, and metric projection into a few oversized methods.
- `src/sft.py` is the repo entrypoint and currently duplicates trainer and pipeline-manifest policy that also exists in typed config/schema code.
- `src/infer/engine.py` and `src/eval/detection.py` are not the highest-risk modules, but both are carrying more backend/artifact orchestration than they should if the stack is expected to keep scaling.

This change is not about changing research behavior first. It is about making the stack more decoupled, scalable, elegant, and maintainable without losing reproducibility or correctness.

The program should preserve these guardrails throughout:

- config-first engineering; no new CLI flags by default,
- preserve geometry invariants via `src/datasets/geometry.py`,
- maintain Qwen3-VL chat-template compatibility,
- keep Stage-2 clean-prefix and unified-loss semantics stable unless a later, explicit contract change is approved,
- preserve reproducibility artifacts and stable metric/artifact contracts unless a follow-on spec delta is intentionally authored.

## What Changes

- Introduce a staged refactor program for runtime-critical modules, starting with internal seam extraction and contract freeze rather than broad behavioral change.
- Freeze the current public training/inference/eval contracts before the first implementation slice:
  - `stage2_ab.pipeline` schema and strict unknown-key behavior,
  - current Stage-2 clean-prefix / triage / dead-anchor semantics,
  - current inference/eval artifact contracts,
  - current reproducibility artifacts and trainer metric key families.
- Extend the existing `src/trainers/stage2_two_channel/` package decomposition by extracting:
  - Channel-B target construction,
  - objective execution / metric projection,
  - explicit intermediate payload/types shared between batch prep and loss execution,
  - and Channel-B step-execution runtime helpers.
- Treat the legacy Stage-2 compatibility shims as explicit protected surfaces during early phases:
  - `src/trainers/stage2_ab_training.py`
  - `src/trainers/stage2_ab/__init__.py`
  - `src/trainers/stage2_ab/executors.py`
  - `src/trainers/stage2_ab/scheduler.py`
- Extract a shared rollout runtime layer from `src/trainers/stage2_rollout_aligned.py` so both Stage-2 trainers can use the same backend and decode lifecycle through a narrower interface.
- Extract bootstrap/setup responsibilities from `src/sft.py`, especially:
  - packing policy/setup,
  - trainer assembly/injection,
  - pipeline-manifest serialization,
  - provenance/run-metadata writing.
- Split `src/infer/engine.py` later into backend adapters plus artifact-writing orchestration, while preserving JSONL, summary, and token-trace contracts.
- Split `src/eval/detection.py` later by first extracting artifact/report writing and top-level orchestration, while preserving output parity; deeper ingest, COCO, F1-ish, and visualization helpers can remain in the public module until a follow-on slice justifies a fuller split.
- Treat the current public infer/eval module boundaries as compatibility surfaces while internals move:
  - `src.infer.engine` as consumed by `src/infer/pipeline.py`
  - `src.eval.detection` as consumed by `src/infer/pipeline.py`, `src/callbacks/detection_eval.py`, and `scripts/evaluate_detection.py`
- Require all `.planning/` artifacts to cite the source OpenSpec change path and relevant workstream/task ranges so the planning layer stays derivative and refreshable.
- Add explicit verification gates after each phase so architectural progress does not outpace behavioral parity.
- Add a brownfield planning layer in `.planning/` where phase execution steps and context files derive from the validated OpenSpec change instead of redefining scope in parallel.
- Treat any stable contract change discovered during the refactor as out-of-scope for silent implementation and require a follow-on OpenSpec delta before landing that slice.

## Capabilities

### New Capabilities

- `runtime-architecture-refactor-program`: A staged engineering workflow for decoupling the CoordExp runtime stack while preserving correctness, reproducibility, geometry invariants, and stable operator-facing contracts.

### Modified Capabilities

- `stage2-ab-training`: internal trainer/module structure only in the initial phases; stable behavior must remain unchanged.
- `rollout-matching-sft`: internal runtime decomposition only in the initial phases; stable behavior must remain unchanged.
- `inference-engine`: internal backend/artifact decomposition only in the initial phases; stable behavior must remain unchanged.
- `detection-evaluator`: internal evaluator decomposition only in the initial phases; stable behavior must remain unchanged.

## Impact

- Primary code surfaces:
  - `src/trainers/stage2_rollout_aligned.py`
  - `src/trainers/stage2_two_channel.py`
  - `src/trainers/stage2_two_channel/executors.py`
  - `src/trainers/stage2_ab_training.py`
  - `src/trainers/stage2_ab/`
  - `src/sft.py`
  - `src/infer/engine.py`
  - `src/infer/pipeline.py`
  - `src/eval/detection.py`
  - `src/callbacks/detection_eval.py`
  - `scripts/evaluate_detection.py`
- Expected new internal module families:
  - `src/trainers/stage2_two_channel/`
  - `src/trainers/rollout_runtime/`
- `src/bootstrap/`
- `src/infer/backends/`
- `src/eval/{artifacts,orchestration}.py`
- Expected planning artifacts for execution:
  - `.planning/PROJECT.md`
  - `.planning/REQUIREMENTS.md`
  - `.planning/ROADMAP.md`
  - `.planning/phases/*`
- Stable authorities that should remain unchanged during early phases:
  - `src/config/schema.py`
  - `src/datasets/geometry.py`
  - `src/trainers/teacher_forcing/`
- Main verification surfaces:
  - `tests/test_coord_geometry_invariants.py`
  - `tests/test_chat_template_regression.py`
  - `tests/test_prompt_variants.py`
  - `tests/test_stage2_ab_training.py`
  - `tests/test_stage2_two_channel_training.py`
  - `tests/test_stage2_rollout_aligned.py`
  - `tests/test_stage2_rollout_import_boundaries.py`
  - `tests/test_stage2_ab_vllm_server_mode_smoke.py`
  - `tests/test_stage2_ab_config_contract.py`
  - `tests/test_training_config_strict_unknown_keys.py`
  - `tests/test_unified_infer_pipeline.py`
  - `tests/test_detection_eval_output_parity.py`
  - `tests/test_detection_eval_ingestion_diagnostics.py`
  - `tests/test_confidence_postop.py`
  - `tests/test_bbox_confidence.py`
  - `tests/test_run_manifest_files.py`
  - `tests/test_dependency_provenance.py`
  - `tests/test_launcher_metadata_env.py`
  - `tests/test_trainer_metrics_payload_contract.py`
- Operator-facing behavior is intended to remain stable in the initial refactor phases.
- If implementation requires changing stable contracts, metric semantics, or artifact schemas, additional spec updates will be required before that slice lands.
