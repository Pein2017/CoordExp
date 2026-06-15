# Handoff: Unified Inference Runtime Migration

Audience: a fresh Codex agent working from an older/previous CoordExp checkout who needs to understand and port or review the migration.

Repo/worktree: `/data/CoordExp/.worktrees/unified-training-infra-refactor`

Branch: `codex/unified-training-infra-refactor`

Primary OpenSpec change: `openspec/changes/unify-inference-runtime/`

Primary implementation plan: `docs/superpowers/plans/2026-05-26-unified-inference-runtime-refactor.md`

## One-Sentence Upgrade Summary

This migration removes the old split between offline inference and Stage-2 online rollout inference by making `src.infer.*` the shared inference root for prompt construction, decode requests, backend dispatch, trace/logprob handling, parsing adapters, artifact shaping, and vLLM lifecycle/server behavior.

## Why This Exists

The previous repo had multiple overlapping inference paths:

- Offline eval/infer used `src.infer.engine`, `src.infer.backends`, and script-local glue.
- Stage-2 online rollout used trainer-local modules under `src.trainers.rollout_runtime.*` and methods on `Stage2RolloutRuntime`.
- Prompt/message formatting, image payload normalization, decode parameter projection, vLLM/HF dispatch, logprob trace handling, and artifact/provenance behavior could drift between offline and online paths.

The refactor makes that drift much harder to introduce. The new rule is:

- Shared inference behavior lives in `src.infer.*`.
- Stage-2 trainers may orchestrate training/eval and residual-correction objective logic, but they should not own generic prompt/backend/decode/trace/parser behavior.
- Deleted legacy modules should not be resurrected as compatibility shims.

## New Runtime Layout

Use these modules as the new canonical surfaces:

- `src/infer/runtime.py`: shared runtime dataclasses/configs, decode-request construction, offline artifact loop, offline engine facade, distributed output/preflight helpers, and rollout owner policy readers.
- `src/infer/prompt.py`: shared offline and online prompt/message construction, visual metadata stamping, one-image payload normalization, and Stage-2 rollout prompt preparation.
- `src/infer/backend.py`: HF/offline backend behavior, HF rollout generation, logits processors, generated-token logprob trace handling, vLLM OpenAI-compatible response adaptation, and left-padding cleanup.
- `src/infer/backend_vllm_config.py`: vLLM engine config normalization.
- `src/infer/backend_vllm_engine.py`: colocated vLLM lazy engine init, eval lifecycle, sleep/wake validation, raw-engine access, shutdown, and best-effort allocator/sleep-mode cleanup.
- `src/infer/backend_vllm_infer.py`: colocated vLLM rollout inference.
- `src/infer/backend_vllm_server.py`: vLLM server request construction/dispatch, server specs, timeouts, world sizes, adapter/full sync, chunking, and strict per-rank request caps.
- `src/infer/backend_sync.py`: shared backend sync/provenance helpers.
- `src/infer/rollout_dispatch.py`: shared Stage-2 rollout dispatch for normal and traced HF/vLLM rollouts.
- `src/infer/parsing.py`: Stage-2 rollout prediction parser adapter that bridges compact-full and coord-token parsing without importing trainer packages.
- `src/infer/artifacts.py`: inference summaries plus Stage-2 eval confidence trace validation/scoring and raw rollout artifact payload shaping.
- `src/infer/constraints.py` and `src/infer/_constraints_impl.py`: compact-grammar and stop-pressure facade replacing old standalone modules.

## Deleted / Do Not Reintroduce

These old surfaces were intentionally removed:

- `src/infer/engine.py`
- `src/infer/backends.py`
- `src/infer/compact_grammar.py`
- `src/infer/stop_pressure.py`
- `src/trainers/rollout_runtime/`

If an older branch imports one of these, migrate the caller to the matching `src.infer.*` surface instead of adding a compatibility file.

## Stage-2 Trainer Boundary After Migration

`src/trainers/stage2_rollout_runtime.py` is still present, but it has been reduced to trainer-owned orchestration:

- Keeps Stage-2 rollout/eval loops, metric aggregation, DDP coordination, post-rollout packing, debug dump orchestration, and trainer resource hooks.
- Delegates prompt preparation to `src.infer.prompt.prepare_rollout_prompt_samples_from_owner`.
- Delegates normal/traced rollout dispatch to `src.infer.rollout_dispatch.rollout_many` and `rollout_many_traced`.
- Delegates decode request construction to `src.infer.runtime.build_decode_request_from_rollout_owner` or `build_decode_request_from_rollout_matching_config`.
- Delegates parser behavior to `src.infer.parsing.parse_stage2_detection_rollout_predictions`.
- Delegates confidence trace scoring and raw rollout artifact payload shaping to `src.infer.artifacts`.
- Delegates vLLM colocate/server lifecycle/config/chunking/sync to `src.infer.backend_vllm_engine` and `src.infer.backend_vllm_server`.

Thin wrapper methods remain in `Stage2RolloutRuntime` for internal callers/tests, but they should stay pass-through. Do not add new decode/backend/parser logic there.

## Prompt and Visual Input Alignment

The new contract is that offline and online inference must share prompt/input construction rather than duplicating templates.

Important behavior:

- One-image-only is the active codebase assumption for rollout prompt prep.
- Prompt messages preserve image-before-text ordering.
- Visual metadata is stamped and checked so Stage-2 residual target construction can detect incompatible source metadata.
- Training still must not silently resize images; preserve geometry/image alignment.
- Qwen chat/template formatting should be routed through the shared prompt helpers, not script-local copies.

Relevant tests:

- `tests/test_detection_prompt_input_codec.py`
- `tests/test_prompt_parity_guard.py`
- `tests/test_prepare_samples_for_rollout_vllm_multimodal.py`
- `tests/test_stage2_ab_prompt_alignment_contract.py`
- `tests/test_prompt_variants.py`

## Decode / Logprob Trace Contract

The unified decode layer now treats generated-token logprob traces as a first-class requirement for metric-bearing rollouts.

Important behavior:

- Decode parameters project through shared `DetectionDecodeRequest`/runtime helpers.
- HF trace-required generation fails fast if aligned generated-token scores are unavailable.
- OpenAI-compatible vLLM server responses fail fast on empty choices or missing trace fields when traces are required.
- Local vLLM selected-token logprobs must match the generated token ID; do not silently choose an arbitrary logprob entry.
- Stop/length/greedy/sampling settings should be represented through the shared runtime/decode request surfaces.

Relevant tests:

- `tests/test_decode_backend_trace_contract.py`
- `tests/test_decode_provenance_contract.py`
- `tests/test_infer_decode_request_mapping.py`
- `tests/test_infer_pipeline_shared_decode_request.py`
- `tests/test_rollout_matching_decoding_cfg.py`
- `tests/test_qwen_generation_contract.py`

## vLLM Notes

vLLM support is now split by responsibility:

- Colocated engine config/lifecycle: `src.infer.backend_vllm_config` and `src.infer.backend_vllm_engine`.
- Colocated rollout inference: `src.infer.backend_vllm_infer`.
- Server-mode rollout, adapter sync, request chunking, and server caps: `src.infer.backend_vllm_server`.

Important policy:

- vLLM logprob support is treated as required for trace-bearing paths. Missing or malformed trace data should fail fast.
- ms-swift adapter sync behavior was preserved. The server-mode adapter/full sync logic lives in `src.infer.backend_vllm_server`; do not replace it with a simpler generic reload without auditing coord adapter synchronization.
- The old trainer-local vLLM runtime package was deleted.

Relevant tests:

- `tests/test_vllm_server_rollout_contract.py`
- `tests/test_vllm_server_adapter_payload.py`
- `tests/test_vllm_server_chunking.py`
- `tests/test_vllm_server_multimodal_payload_contract.py`
- `tests/test_ddp_vllm_sync_failure_propagation.py`

## Offline Scripts and Evaluation

Scripts and eval helpers were migrated to shared runtime/provenance surfaces:

- `scripts/run_infer.py`
- `scripts/evaluate_detection.py`
- `scripts/export_coco_submission.py`
- `scripts/stamp_inference_provenance.py`
- `src/callbacks/stage1_detection_eval.py`
- `src/eval/detection_orchestrator.py`
- `src/eval/proxy_eval_bundle.py`

The intent is that offline inference, offline eval, Stage-1 eval callbacks, and Stage-2 online rollouts use the same decode/prompt/provenance foundation.

## Config / Docs Scope

Updated active config/docs surfaces include:

- `configs/stage2_rollout_correction/`
- `docs/ARTIFACTS.md`
- `docs/IMPLEMENTATION_MAP.md`
- `docs/SYSTEM_OVERVIEW.md`
- `docs/eval/CONTRACT.md`
- `docs/eval/WORKFLOW.md`
- `docs/training/README.md`
- `docs/training/STAGE2_RUNBOOK.md`
- `openspec/changes/unify-inference-runtime/`

There are also edits in `openspec/changes/materialize-vllm-full-sync-adapter-rows/` because the vLLM adapter-sync contract had to reference the new backend ownership.

## Import/Layout Gates

The main regression guard is `tests/test_infer_layout_import_gates.py`.

It asserts that deleted modules stay absent and active code does not import the old runtime layout. Historical planning docs and tests may mention old names intentionally; active source should not import them.

Useful manual checks:

```bash
test ! -e src/infer/engine.py
test ! -e src/infer/backends.py
test ! -e src/infer/compact_grammar.py
test ! -e src/infer/stop_pressure.py
test ! -d src/trainers/rollout_runtime

rg -n "from src\\.infer\\.(engine|backends|compact_grammar|stop_pressure)\\b|import src\\.infer\\.(engine|backends|compact_grammar|stop_pressure)\\b|from src\\.trainers\\.rollout_runtime\\b|import src\\.trainers\\.rollout_runtime\\b" src scripts configs docs openspec/specs -g '!docs/superpowers/**'
```

## Verification Already Run

Use the `ms` conda environment. The following passed in this worktree:

```bash
/root/miniconda3/bin/conda run -n ms python -m py_compile src/infer/backend_vllm_engine.py src/trainers/stage2_rollout_runtime.py
```

```bash
/root/miniconda3/bin/conda run -n ms python -m pytest tests/test_stage2_rollout_runtime.py -q
# 90 passed, 2 warnings
```

```bash
/root/miniconda3/bin/conda run -n ms python -m pytest tests/test_stage2_rollout_runtime.py tests/test_vllm_server_rollout_contract.py tests/test_vllm_server_adapter_payload.py tests/test_vllm_server_chunking.py tests/test_ddp_vllm_sync_failure_propagation.py tests/test_prepare_samples_for_rollout_vllm_multimodal.py tests/test_rollout_matching_decoding_cfg.py -q
# 139 passed, 2 warnings
```

```bash
/root/miniconda3/bin/conda run -n ms python -m pytest tests/test_detection_prompt_input_codec.py tests/test_decode_backend_trace_contract.py tests/test_decode_provenance_contract.py tests/test_parser_policy_parity.py tests/test_unified_infer_pipeline.py tests/test_infer_batch_decoding.py tests/test_qwen_generation_contract.py tests/test_infer_layout_import_gates.py -q
# 154 passed
```

```bash
/root/miniconda3/bin/conda run -n ms python -m pytest tests/test_artifact_contract_docs.py tests/test_experiment_manifest_file.py tests/test_run_manifest_files.py tests/test_run_metadata_file.py tests/test_coco_submission_export_provenance.py tests/test_evaluate_detection_provenance.py tests/test_stamp_inference_provenance.py tests/test_run_infer_legacy_shared_runtime.py tests/test_infer_pipeline_shared_decode_request.py tests/test_infer_decode_request_mapping.py tests/test_score_policy_fingerprint.py -q
# 73 passed
```

```bash
/root/miniconda3/bin/conda run -n ms python -m pytest tests/test_inference_runtime_backend_facade.py tests/test_infer_constraints_facade.py tests/test_prompt_parity_guard.py tests/test_infer_decode_request_mapping.py tests/test_infer_pipeline_shared_decode_request.py tests/test_stage2_ab_prompt_alignment_contract.py tests/test_infer_layout_import_gates.py -q
# 75 passed, 2 warnings
```

```bash
git diff --check
# passed
```

```bash
rg -n "^- \\[ \\]" openspec/changes/unify-inference-runtime/tasks.md
# no unchecked tasks
```

## Verification Not Run / Environment Gap

This failed because `openspec` was not installed or on `PATH`, including inside `ms`:

```bash
openspec validate unify-inference-runtime --strict
/root/miniconda3/bin/conda run -n ms openspec validate unify-inference-runtime --strict
```

Run this later if the target environment has the OpenSpec CLI available.

No production-scale training, full-dataset offline eval, live multi-process GPU vLLM server smoke, or colocated vLLM GPU smoke was run. Before trusting this for a training-scale run, execute one real Stage-2 rollout-correction smoke on the target machine, preferably starting with:

```bash
configs/stage2_rollout_correction/smoke/compact_full_hf_1step.yaml
```

Then add a live vLLM server or colocate smoke if that backend is the intended production path.

## Subagent Review Results

Two read-only review passes found no blocking issues:

- vLLM lifecycle extraction audit: no findings; confirmed no wrapper recursion, monkeypatch seams preserved, shutdown cleanup semantics retained, and no new Stage-2 import cycle.
- Final OpenSpec 6.2 closure audit: no findings; confirmed `Stage2RolloutRuntime` is now orchestration/thin wrappers while shared prompt/backend/decode/trace/parser behavior is under `src.infer.*`.

## Recommended Next Agent Workflow

1. Start from `docs/IMPLEMENTATION_MAP.md`, `docs/SYSTEM_OVERVIEW.md`, and `openspec/changes/unify-inference-runtime/tasks.md`.
2. Inspect `src/infer/runtime.py`, `src/infer/prompt.py`, `src/infer/backend.py`, `src/infer/rollout_dispatch.py`, and `src/infer/backend_vllm_server.py` before changing any scripts or trainers.
3. If porting to an older repo, first migrate callers away from deleted modules, then delete the modules, then enable layout gates.
4. Keep `Stage2RolloutRuntime` as orchestration only. Do not add new prompt/decode/backend/parser code there.
5. Preserve ms-swift adapter sync semantics when touching vLLM server mode.
6. Use targeted tests before broad tests. The suites above are the best smoke matrix for this migration.

## Current Dirty Scope

This worktree has a large intentional refactor diff. Major categories:

- New shared inference modules under `src/infer/`.
- Deleted legacy inference/runtime modules listed above.
- Stage-2 runtime reduction and tests.
- Offline scripts/eval/provenance migration.
- Docs/OpenSpec updates.
- Stage-2 rollout-correction config updates.

Do not assume unrelated user dirt should be reverted. Check `git status --short --branch` and inspect diffs before staging or committing.
