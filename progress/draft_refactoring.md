# CoordExp Architectural Refactor Diagnosis

Date: 2026-03-18

Scope:
- Canonical docs: `docs/PROJECT_CONTEXT.md`, `docs/SYSTEM_OVERVIEW.md`, `docs/IMPLEMENTATION_MAP.md`
- Hotspots: `src/trainers/stage2_rollout_aligned.py`, `src/trainers/stage2_two_channel.py`, `src/trainers/stage2_two_channel/executors.py`, `src/sft.py`, `src/config/schema.py`, `src/eval/detection.py`, `src/infer/engine.py`
- Guardrails applied: config-first, preserve geometry via `src/datasets/geometry.py`, keep Qwen3-VL chat-template compatibility, do not touch upstream HF model files

## Summary

1. `src/trainers/stage2_rollout_aligned.py` is the clearest true god module in the repo. `RolloutMatchingSFTTrainer` owns rollout backend lifecycle, vLLM local/server orchestration, target construction, packing, monitoring, training loss execution, and production-style evaluation inside one class. That is the highest architectural risk.
2. `src/trainers/stage2_two_channel.py` is still too dense, but it already contains the right incremental decomposition direction. The package split into `scheduler.py`, `executors.py`, and a dynamic compatibility wrapper is a strong precedent; the remaining hotspots are `_prepare_batch_inputs_b` and `compute_loss`.
3. `src/config/schema.py` is large but cohesive. It is acting as the strict contract authority for `stage2_ab.pipeline` and unknown-key rejection, and the tests I ran support keeping it centralized for now.
4. `src/sft.py` is a legitimate orchestration entrypoint, but it currently duplicates trainer/pipeline policy that already exists in typed config code. The biggest maintainability issue there is duplicated authority, not raw size.
5. `src/infer/engine.py` and `src/eval/detection.py` are moderately overloaded and worth splitting later, but they are a lower priority than the Stage-2 trainers because they are less entangled with the core training-time research loop.

Confirmed baseline:
- `conda run -n ms python -m pytest -q tests/test_stage2_ab_config_contract.py` -> 49 passed
- `conda run -n ms python -m pytest -q tests/test_training_config_strict_unknown_keys.py` -> 50 passed
- `conda run -n ms python -m pytest -q tests/test_stage2_ab_training.py` -> 78 passed
- `conda run -n ms python -m pytest -q tests/test_stage2_rollout_aligned.py` -> 77 passed
- `conda run -n ms python -m pytest -q tests/test_unified_infer_pipeline.py` -> 26 passed
- `conda run -n ms python -m pytest -q tests/test_detection_eval_output_parity.py` -> 15 passed

## Evidence-backed diagnosis

### 1. `src/trainers/stage2_rollout_aligned.py`

Classification:
- True god module.

Current responsibilities:
- Trainer state initialization for rollout, monitor dumps, vLLM local/server lifecycle, pending logs, and wallclock tracking in `RolloutMatchingSFTTrainer.__init__` at `src/trainers/stage2_rollout_aligned.py:1305-1358`.
- Backend lifecycle and engine creation in methods like `_ensure_vllm_engine` at `src/trainers/stage2_rollout_aligned.py:4044-4364`.
- Shared rollout dispatch API in `_rollout_many` at `src/trainers/stage2_rollout_aligned.py:6643-6728`, including use by both rollout-aligned eval and the Stage-2 two-channel trainer.
- Train-time target building and post-rollout packing in `_prepare_batch_inputs` at `src/trainers/stage2_rollout_aligned.py:7136-7771`.
- Teacher-forcing loss execution in `compute_loss` at `src/trainers/stage2_rollout_aligned.py:8703-8878`.
- Production-style rollout evaluation, confidence post-op integration, semantic desc monitoring, DDP metric reduction, and eval metric emission in `evaluate` at `src/trainers/stage2_rollout_aligned.py:8974-10073`.

Why it is hard to maintain:
- `_prepare_batch_inputs` alone reads rollout config, gates qualitative monitor dumps, runs rollouts, parses responses, matches against GT, builds prefix and FN targets, re-encodes assistant tokens, and packs batches. That is orchestration plus domain logic plus monitoring in one method: `src/trainers/stage2_rollout_aligned.py:7136-7771`.
- The vLLM server path is deeply embedded in the trainer. `_rollout_many_vllm_server` handles request seeding, server world-size discovery, per-rank request caps, retry logic, recursive split fallback, thread-pool fanout, and debug dumping inside a single method: `src/trainers/stage2_rollout_aligned.py:6147-6507`.
- `evaluate` is not a thin adapter over reusable evaluator logic. It re-implements rollout-time inference, optional confidence scoring, semantic desc monitoring, distributed reductions, and metric naming in one place: `src/trainers/stage2_rollout_aligned.py:9002-9067` and `src/trainers/stage2_rollout_aligned.py:9656-9775`.
- The class is a hidden shared dependency. `Stage2ABTrainingTrainer._prepare_batch_inputs_b` calls `self._rollout_many(...)`, so rollout runtime changes inside this file spill into the two-channel trainer as well: `src/trainers/stage2_two_channel.py:2158-2178`.

Refactor seams to extract:
- Shared rollout runtime package, for example `src/trainers/rollout_runtime/`, owning `_rollout_many`, `_rollout_many_hf`, `_rollout_many_vllm`, `_rollout_many_vllm_server`, `_ensure_vllm_engine`, and sync helpers.
- Rollout-aligned target builder module, for example `src/trainers/rollout_aligned/target_builder.py`, owning the parse -> match -> `Y_train` assembly path now embedded in `_prepare_batch_inputs`.
- Rollout-aligned evaluator module, for example `src/trainers/rollout_aligned/evaluator.py`, owning the `evaluate` loop and its eval-only detection/confidence hooks.

Risks and tradeoffs:
- Tests monkeypatch `trainer._rollout_many` and `trainer._ensure_vllm_engine`, so preserving trainer-level adapter methods or a compatibility wrapper matters for low-risk migration. Evidence handles include `tests/test_stage2_rollout_aligned.py` references surfaced around `_rollout_many` and `_ensure_vllm_engine`.
- Extracting backend logic too aggressively before defining a narrow runtime interface could move complexity around without reducing it.
- The metric and artifact surface is already large. Internal extraction is safe; changing `rollout/*`, `eval/*`, or confidence artifact semantics is not.

Incremental or broader contract shift:
- Incremental if the class keeps the same public methods, config keys, metrics, and artifact contracts.
- OpenSpec required if you change rollout/eval behavior, stable metric keys, or target-building semantics. Relevant specs/docs are `openspec/specs/teacher-forcing-unified-loss-registry/spec.md`, `docs/training/STAGE2_RUNBOOK.md`, and `docs/eval/CONTRACT.md`.

### 2. `src/trainers/stage2_two_channel.py` plus `src/trainers/stage2_two_channel/executors.py` and `scheduler.py`

Classification:
- Partially decomposed, but still contains two major god methods and one oversized executor.

What is already good:
- The trainer is already assembled from mixins and a base trainer: `Stage2ABTrainingTrainer(Stage2ABSchedulerMixin, Stage2ABChannelExecutorsMixin, RolloutMatchingSFTTrainer)` at `src/trainers/stage2_two_channel.py:984-988`.
- The schedule seam is clean and readable. `_stage2_channel_for_step` is a focused, deterministic Bresenham-style scheduler in `src/trainers/stage2_two_channel/scheduler.py:153-175`.
- The package-level wrapper intentionally preserves monkeypatch/import compatibility. `_load_impl` and `_Stage2ModuleProxy.__setattr__` in `src/trainers/stage2_two_channel/__init__.py:29-49` and `src/trainers/stage2_two_channel/__init__.py:56-66` are useful compatibility infrastructure, not accidental complexity.

Current responsibilities:
- Channel-B rollout config, dual decode request setup, anchor/explorer rollout execution, triage, clean-prefix reconstruction, dead-anchor suppression target building, monitor candidate assembly, and batch metrics in `_prepare_batch_inputs_b` at `src/trainers/stage2_two_channel.py:2071-3519`.
- Channel-A and Channel-B objective routing, multi-iteration soft self-context forward, multiple `TeacherForcingContext` passes, and log-key projection in `compute_loss` at `src/trainers/stage2_two_channel.py:3522-4693`.
- Channel-B threaded producer/consumer execution, DDP barrier policy, no-sync behavior, queue backpressure, and packing in `_stage2_b_step_budgeted_train` at `src/trainers/stage2_two_channel/executors.py:556-1132`.

Why it is hard to maintain:
- `_prepare_batch_inputs_b` mixes runtime setup and the actual research contract. The same method reads `stage2_ab.channel_b.triage_posterior.*`, runs anchor and explorer rollouts, deduplicates, associates anchor/explorer objects, classifies `shielded_anchor` vs `dead_anchor`, rebuilds the clean prefix, injects FN tails, computes dead-anchor suppression targets, and emits diagnostics: `src/trainers/stage2_two_channel.py:2090-2152`, `src/trainers/stage2_two_channel.py:2394-2767`, `src/trainers/stage2_two_channel.py:3290-3506`.
- `compute_loss` is simultaneously an objective runner and a metric compiler. It builds context objects, runs the unified loss pipeline in several variants, and then manually maps pipeline metrics back to `loss/A1_*`, `loss/A2_*`, `loss/B_*`, `coord_diag/*`, and `train/triage/*`: `src/trainers/stage2_two_channel.py:3525-3695`, `src/trainers/stage2_two_channel.py:4056-4235`, `src/trainers/stage2_two_channel.py:4305-4665`.
- `executors.py` is a real improvement over keeping everything in `stage2_two_channel.py`, but the Channel-B executor is still a concurrency-heavy runtime unit. It owns timeout derivation, DDP monitor group initialization, producer thread management, bounded queue behavior, and packing-prefill logic: `src/trainers/stage2_two_channel/executors.py:603-701` and `src/trainers/stage2_two_channel/executors.py:948-1045`.

Refactor seams to extract:
- `src/trainers/stage2_two_channel/channel_b_target_builder.py` or `clean_prefix_builder.py`
  - Own the `_prepare_batch_inputs_b` domain path.
  - Move pure helpers such as `_sequential_dedup_bbox_objects`, `_build_dead_anchor_suppression_targets`, `_build_canonical_prefix_data`, `_build_canonical_closed_container_text`, and the inline rollout-view/triage loop.
- `src/trainers/stage2_two_channel/objective_runner.py`
  - Own `compute_loss` context assembly, pipeline calls, and log projection.
  - Make the boundary explicit between research semantics and metric formatting.
- `src/trainers/stage2_two_channel/channel_b_executor.py`
  - Own the producer/consumer, queue, and DDP coordination pieces now embedded in `_stage2_b_step_budgeted_train`.

Risks and tradeoffs:
- The `_rollout_matching_meta` payload is an implicit interface between batch prep and loss computation. Splitting without first making that payload explicit will create hidden coupling across files.
- The current wrapper package is preserving compatibility for tests and monkeypatching. Any file split should retain that behavior.
- The temptation will be to also change semantics while extracting. That is the main risk to avoid.

Incremental or broader contract shift:
- Incremental if you first freeze intermediate payloads and keep clean-prefix, triage, and metric names identical.
- OpenSpec required if you change clean-prefix semantics, `loss_dead_anchor_suppression`, `stage2_ab.pipeline` meaning, or the canonical logging contract under `loss/*`, `train/triage/*`, and `coord_diag/*`. Relevant specs are `openspec/specs/stage2-ab-training/spec.md` and `openspec/specs/teacher-forcing-unified-loss-registry/spec.md`.

### 3. `src/sft.py`

Classification:
- Large orchestration module, not the worst god module, but it has too much policy knowledge.

Current responsibilities:
- Trainer selection in `resolve_trainer_cls` at `src/sft.py:57-84`.
- Pipeline manifest construction in `_build_pipeline_manifest` at `src/sft.py:1024-1357`.
- Packing policy parse and dataset wrapping around `src/sft.py:1846-1976`.
- Trainer class/mixin/callback composition and construction around `src/sft.py:2659-2741`.
- Rollout and Stage-2 config injection at `src/sft.py:2791-2951`.
- Run metadata emission and cleanup at `src/sft.py:3091-3182`.

Why it is hard to maintain:
- `_build_pipeline_manifest` still contains variant-specific module knowledge, default configs, and serialization policy even though `src/config/schema.py` now already validates Stage-2 pipeline structure strictly. That is duplicated authority: `src/config/schema.py:1526-1753` vs `src/sft.py:1024-1357`.
- `main` acts as an entrypoint plus a trainer service locator plus a config injector plus a provenance writer. That is survivable, but it makes every trainer refactor touch the largest startup path.
- Trainer-specific config injection is not localized. Rollout config, packing knobs, object ordering, and pipeline manifests are all injected from `main`: `src/sft.py:2862-2875` and `src/sft.py:2925-2951`.

Refactor seams to extract:
- `src/training/bootstrap/packing.py`
  - Own `_parse_packing_config`, `_recompute_gas_for_packing`, static packing setup, and related validation.
- `src/training/bootstrap/trainer_setup.py`
  - Own `resolve_trainer_cls`, mixin assembly, callback assembly, and trainer construction.
- `src/config/pipeline_manifest.py` or a method on typed config objects
  - Own manifest serialization so typed config and manifest generation cannot drift.
- `src/training/bootstrap/provenance.py`
  - Own `run_metadata.json` composition and cleanup helpers.

Risks and tradeoffs:
- `sft.py` is the main entrypoint, so over-splitting it into tiny helpers would make navigation worse.
- Reproducibility artifacts like `resolved_config.json`, `run_metadata.json`, and `logging.jsonl` are part of the practical contract. Byte-level or field-level drift matters even if behavior stays the same.

Incremental or broader contract shift:
- Internal extraction only, no OpenSpec required, if config semantics and emitted metadata stay the same.
- OpenSpec required only if you change config contract, pipeline checksum meaning, or emitted stable artifact fields.

### 4. `src/config/schema.py`

Classification:
- Large but cohesive.

Why it is not a top refactor target:
- It is acting as a strict schema authority, which matches the repo’s config-first posture.
- `_validate_section_keys_strict` at `src/config/schema.py:62-83` enforces fail-fast unknown-key behavior.
- `Stage2PipelineConfig.from_mapping` at `src/config/schema.py:1526-1753` validates canonical module order, allowed presets, required config keys, channel dependencies, and clean-prefix contract requirements.
- `Stage2ABConfig.from_mapping` at `src/config/schema.py:1769-1879` rejects legacy flat knobs and enforces top-level `stage2_ab.*` structure.
- The contract tests I ran for Stage-2 config and strict unknown keys both passed.

What I would keep as-is:
- Keep `schema.py` as the single typed authority for `stage2_ab` and training config validation.
- If it ever gets split, split by stable config domain sections, not by arbitrary size thresholds.

What I would not do first:
- Do not start the architecture refactor by scattering Stage-2 schema across multiple files unless there is a clear ownership boundary and the tests prove no drift.

Contract impact:
- Any actual schema change is OpenSpec territory.

### 5. `src/infer/engine.py`

Classification:
- Moderately mixed, but still recognizably an inference orchestrator.

Current responsibilities:
- Backend load and selection in `load_model` at `src/infer/engine.py:469-568`.
- vLLM server request/response adaptation in `_generate_vllm_server` at `src/infer/engine.py:836-938`.
- End-to-end JSONL reading, preflight validation, batch generation, prediction parsing, token-trace writing, output JSONL emission, and summary emission in `infer` at `src/infer/engine.py:1218-1548`.

Why it is harder to extend than it needs to be:
- Backend-specific concerns are embedded directly in `InferenceEngine`, so extending or testing a new backend means opening the same class that also owns JSONL stream control and output serialization.
- `infer` is doing both orchestration and artifact writing. It builds resolved metadata, canonicalizes errors, flushes batch generation, writes traces, and emits final summary in one method: `src/infer/engine.py:1253-1388`.
- The module is managing deterministic HF vs best-effort vLLM behavior, which is important, but that concern belongs at a backend-adapter boundary rather than inside the outer inference loop.

Refactor seams to extract:
- `src/infer/backends/hf.py`
- `src/infer/backends/vllm_local.py`
- `src/infer/backends/vllm_server.py`
- `src/infer/writer.py` or `src/infer/artifacts.py`

Risks and tradeoffs:
- Preserve current JSONL fields and trace behavior. `pred_token_trace.jsonl` is especially useful for confidence post-op.
- Preserve prompt generation through `get_template_prompts` and current object field ordering so train/infer prompt drift does not reappear.

Incremental or broader contract shift:
- Internal extraction only if output JSONL and summary are unchanged.
- OpenSpec only if the infer artifact contract changes.

### 6. `src/eval/detection.py`

Classification:
- Moderately mixed, but still evaluator-centric rather than a true god module.

Current responsibilities:
- JSONL ingestion and diagnostics via `load_jsonl`.
- GT reconstruction and COCO conversion via helpers including `preds_to_gt_records`.
- COCO eval, F1-ish eval, semantic matching, visualization materialization, overlay rendering, and artifact writing.
- The end-to-end save path in `evaluate_and_save` at `src/eval/detection.py:1421-1512` calls `_prepare_all`, `_run_coco_eval`, `evaluate_f1ish`, `materialize_gt_vs_pred_vis_resource`, `write_outputs`, and optional overlays.

Why it is not urgent:
- The module is broad, but the responsibilities are all still inside the offline evaluation domain.
- Unlike the trainers, it is not also managing training-time backends, DDP coordination, and target construction.
- The targeted parity test I ran passed, which lowers the urgency of reorganizing it immediately.

Where the seams are:
- `src/eval/detection/ingest.py`
- `src/eval/detection/coco.py`
- `src/eval/detection/f1ish.py`
- `src/eval/detection/artifacts.py`
- `src/eval/detection/vis.py`

Risks and tradeoffs:
- Keep `metrics.json`, `per_image.json`, `coco_gt.json`, `coco_preds.json`, `per_class.csv`, and `vis_resources/gt_vs_pred.jsonl` stable.
- Preserve current score-contract strictness and semantic matching behavior.

Incremental or broader contract shift:
- Internal extraction only if artifact names and metric meanings remain stable.
- OpenSpec required if evaluation artifact schema or stable metric semantics change.

## Recommended refactor plan

### 1. Continue the `stage2_two_channel/` package split first

Why this is first:
- It hits the active Stage-2 training path.
- The package and compatibility wrapper already exist.
- The most overloaded logic is already partly isolated, which lowers migration risk.

Module boundaries:
- `src/trainers/stage2_two_channel/channel_b_target_builder.py`
- `src/trainers/stage2_two_channel/objective_runner.py`
- `src/trainers/stage2_two_channel/types.py`

Migration sequence:
1. Introduce explicit dataclasses for the batch-builder to loss-runner boundary.
2. Move pure helpers and Channel-B target-construction code out of `_prepare_batch_inputs_b`.
3. Reduce `compute_loss` to channel dispatch plus calls into an objective runner.
4. Leave `Stage2ABTrainingTrainer` as the public assembly point and keep the compatibility wrapper.

What should stay fixed:
- `stage2_ab.pipeline` contract.
- Clean-prefix and dead-anchor semantics.
- Current log-key names.

Verification:
- `conda run -n ms python -m pytest -q tests/test_stage2_ab_training.py`
- `conda run -n ms python -m pytest -q tests/test_stage2_ab_config_contract.py`
- `conda run -n ms python -m pytest -q tests/test_training_config_strict_unknown_keys.py`

Contract impact:
- Internal only if semantics and metrics stay identical.

### 2. Extract a shared rollout runtime from `RolloutMatchingSFTTrainer`

Why this is second:
- Both Stage-2 systems depend on the same rollout backend machinery.
- This reduces duplicate future work and shrinks the largest god class.

Module boundaries:
- `src/trainers/rollout_runtime/backend.py`
- `src/trainers/rollout_runtime/vllm_local.py`
- `src/trainers/rollout_runtime/vllm_server.py`
- `src/trainers/rollout_runtime/decode.py`

Migration sequence:
1. Move rollout backend selection and decode-request logic behind a narrow service interface.
2. Keep trainer methods like `_rollout_many` as thin delegators so tests and monkeypatches still work.
3. Move vLLM engine/server lifecycle and sync helpers next.

What should stay fixed:
- Existing config keys under `rollout_matching.*`
- Per-rank request-cap behavior
- Seed derivation and logging behavior

Verification:
- `conda run -n ms python -m pytest -q tests/test_stage2_rollout_aligned.py`
- `conda run -n ms python -m pytest -q tests/test_stage2_ab_training.py`

Contract impact:
- Internal only unless you change rollout/eval semantics or public metrics.

### 3. Shrink `executors.py` by isolating Channel-B step execution

Why this matters:
- `Stage2ABChannelExecutorsMixin` is a good decomposition, but `_stage2_b_step_budgeted_train` is still a concurrency-heavy hotspot.
- The queue/thread/DDP barrier logic is fragile enough to deserve its own ownership boundary.

Module boundaries:
- `src/trainers/stage2_two_channel/channel_b_step_executor.py`
- Optional `src/trainers/stage2_two_channel/ddp_coordination.py`

Migration sequence:
1. Extract producer/consumer pipeline code.
2. Extract DDP timeout and monitored-barrier policy.
3. Leave `executors.py` as a thin coordination shell.

What should stay fixed:
- `stage2_ab.channel_b.ddp_phase_timeout_s`
- `stage2_ab.channel_b.producer_wait_timeout_s`
- Existing no-sync semantics and packed-step weighting

Verification:
- `conda run -n ms python -m pytest -q tests/test_stage2_ab_training.py`
- `conda run -n ms python -m pytest -q tests/test_stage2_ab_ddp_phase_monitor_disable.py`
- `conda run -n ms python -m pytest -q tests/test_stage2_ab_disable_average_tokens_across_devices.py`

Contract impact:
- Internal only.

### 4. Remove duplicated pipeline authority from `sft.py`

Why this is important:
- It is a lower-risk cleanup that reduces future config drift.
- Today both `schema.py` and `_build_pipeline_manifest` know too much about Stage-2 pipeline structure.

Module boundaries:
- `src/config/pipeline_manifest.py`
- `src/training/bootstrap/trainer_setup.py`
- `src/training/bootstrap/provenance.py`

Migration sequence:
1. Make typed config objects the source for manifest serialization.
2. Move trainer injection logic out of `main`.
3. Keep `main` as the single CLI entrypoint.

What should stay fixed:
- Manifest checksum behavior
- Trainer injection timing
- Reproducibility artifact contents

Verification:
- `conda run -n ms python -m pytest -q tests/test_stage2_ab_config_contract.py`
- `conda run -n ms python -m pytest -q tests/test_training_config_strict_unknown_keys.py`
- `conda run -n ms python -m pytest -q tests/test_run_manifest_files.py`

Contract impact:
- Internal only if emitted manifests and metadata remain identical.

### 5. Split inference backends before splitting evaluation internals

Why this order:
- `src/infer/engine.py` is a cleaner candidate for backend extraction than `src/eval/detection.py`, and inference backends are the more obvious extension seam.

Module boundaries:
- `src/infer/backends/hf.py`
- `src/infer/backends/vllm_local.py`
- `src/infer/backends/vllm_server.py`
- `src/infer/artifacts.py`

Verification:
- `conda run -n ms python -m pytest -q tests/test_unified_infer_pipeline.py`
- `conda run -n ms python -m pytest -q tests/test_confidence_postop.py`

Contract impact:
- Internal only if `gt_vs_pred.jsonl`, `pred_token_trace.jsonl`, and `summary.json` stay stable.

### 6. Split `src/eval/detection.py` only after the trainer work

Why later:
- It is broad, but it is not the core runtime bottleneck for research iteration.
- The trainer files are where change coupling is currently most dangerous.

Module boundaries:
- `src/eval/detection/ingest.py`
- `src/eval/detection/coco.py`
- `src/eval/detection/f1ish.py`
- `src/eval/detection/artifacts.py`

Verification:
- `conda run -n ms python -m pytest -q tests/test_detection_eval_output_parity.py`
- `conda run -n ms python -m pytest -q tests/test_confidence_postop.py`

Contract impact:
- Internal only if eval artifact and metric contracts stay stable.

## What should stay as-is for now

- `src/config/schema.py` as the typed config authority.
- `src/datasets/geometry.py` as the geometry authority. Do not duplicate or re-interpret geometry helpers elsewhere.
- `src/trainers/stage2_two_channel/scheduler.py` as the schedule seam.
- `src/trainers/stage2_two_channel/__init__.py` dynamic compatibility wrapper until import and monkeypatch compatibility is intentionally redesigned.
- `src/trainers/teacher_forcing/` registry modules as the canonical loss/math implementation. The trainers should call into them, not absorb more of that math.
- Single-dataset assumptions and packing-first efficiency posture. Do not spend early refactor cycles building broader multi-dataset or fusion-config abstractions.

## OpenSpec trigger points

OpenSpec is not needed for internal file/package extraction that preserves behavior.

OpenSpec is needed if the refactor changes any of the following:
- `stage2_ab.pipeline` schema or semantics
- clean-prefix Channel-B semantics, dead-anchor suppression behavior, or triage definitions
- stable training metric keys under `loss/*`, `coord_diag/*`, `train/triage/*`, or rollout/eval key families
- infer/eval artifact schema documented in `docs/eval/CONTRACT.md`

## Verification plan

Use these as the minimum gates during refactor work:

- Config/contract gates:
  - `conda run -n ms python -m pytest -q tests/test_stage2_ab_config_contract.py`
  - `conda run -n ms python -m pytest -q tests/test_training_config_strict_unknown_keys.py`

- Stage-2 trainer gates:
  - `conda run -n ms python -m pytest -q tests/test_stage2_ab_training.py`
  - `conda run -n ms python -m pytest -q tests/test_stage2_rollout_aligned.py`

- Infer/eval gates:
  - `conda run -n ms python -m pytest -q tests/test_unified_infer_pipeline.py`
  - `conda run -n ms python -m pytest -q tests/test_detection_eval_output_parity.py`
  - `conda run -n ms python -m pytest -q tests/test_confidence_postop.py`

- Artifact checks after each phase:
  - Training still writes `resolved_config.json`, `runtime_env.json`, `run_metadata.json`, and `logging.jsonl`.
  - Inference still writes `gt_vs_pred.jsonl`, optional `pred_token_trace.jsonl`, and `summary.json`.
  - Evaluation still writes `metrics.json`, `per_image.json`, and when enabled `vis_resources/gt_vs_pred.jsonl`.

- Migration discipline:
  - Do not change geometry helpers outside `src/datasets/geometry.py`.
  - Do not change prompt rendering behavior without verifying Qwen3-VL chat-template compatibility.
  - Do not rename stable metrics or artifact fields as part of the first extraction pass.
