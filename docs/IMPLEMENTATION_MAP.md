---
doc_id: docs.implementation-map
layer: docs
doc_type: implementation-map
status: canonical
domain: repo
summary: Small source and test routing map for the current CoordExp-Swift implementation.
updated: 2026-08-19
---

# Implementation Map

Use this page to locate the current owner before opening a broad source area.
It describes `main`'s CoordExp-Swift route; stable contract details belong to
the linked OpenSpecs. Historical paths are listed only to prevent accidental
reuse.

## Current entry and assembly map

| Question | Current owner | First verification surface |
| --- | --- | --- |
| Training entry | `src/train.py` | `tests/training/` |
| Training assembly | `src/training/pipeline.py` | `tests/training/test_pipeline_assembly.py` |
| Planned-step loop | `src/training/supervised_trainer.py` | `tests/training/test_supervised_trainer.py` |
| Config schema and resolution | `src/config/loader.py`, `src/config/models.py`, `src/config/resolve.py` | `tests/config/` and strict-config tests |
| Raw JSONL and geometry | `src/data/` | `docs/data/CONTRACT.md`, `tests/data/` |
| Prompt and semantic spans | `src/templates/` | `tests/templates/` |
| Qwen load/encode/forward | `src/qwen/` | `tests/qwen/` |
| Packing | `src/packing/`, `src/training/pack_cache.py` | `tests/packing/`, `tests/training/test_pack_cache.py` |
| Token supervision | `src/supervision/` | `tests/supervision/` |
| Loss assembly | `src/losses/` | `tests/losses/` |
| Runtime and optimization | `src/runtime/`, `src/optim/`, `src/adapters/` | `tests/runtime/`, `tests/optim/`, `tests/adapters/` |
| Training run files | `src/artifacts/run_writer.py` | `tests/artifacts/test_run_artifacts.py` |
| Training checkpoints | `src/artifacts/checkpoints.py` | `tests/artifacts/test_checkpoint_writer.py` |
| Inference payload manifest | `src/artifacts/checkpoint_payload.py` | `tests/artifacts/test_checkpoint_payload_identity.py` |
| Exact training state | `src/artifacts/training_state.py` | `tests/artifacts/test_training_state.py` |
| Exact resume admission and restore | `src/training/exact_resume.py` | `tests/training/test_exact_resume.py`, `tests/training/test_pipeline_exact_resume.py` |
| Inference entry and pipeline | `src/infer.py`, `src/inference/pipeline.py` | `tests/inference/test_pipeline.py` |
| Inference composition/backend | `src/inference/runtime.py`, `src/inference/backend.py` | `tests/inference/test_config_runtime.py`, `tests/inference/test_scoring.py` |
| Inference artifacts | `src/inference/artifacts.py`, `src/inference/merge.py` | `tests/inference/` |
| Forward-only eval | `src/eval/forward.py` | `tests/eval/test_forward_eval.py` |
| Detection eval | `src/eval/detection_consumer.py` | `tests/eval/test_detection_consumer.py` |
| Coordinate-token targets | `src/coordinate_targets.py` | `tests/templates/test_renderer.py`, `tests/losses/test_runner.py` |
| Planned-step schedule | `src/training/schedule.py` | `tests/training/test_schedule.py` |
| Runtime seeding | `src/runtime/seeding.py` | `tests/runtime/test_train_runtime.py` |
| Detection category registry | `src/eval/detection_categories.py` | `tests/eval/test_detection_consumer.py` |
| Config tracing CLI | `src/trace_config.py` | `tests/config/test_train_config.py` |
| Qwen alias checks | `src/common/qwen_aliases.py` | `tests/qwen/test_token_identity.py` |
| Visualization helpers | `src/vis/` | `tests/test_gt_vs_pred_visualization.py`, `scripts/visualize_detection.py` |

Normative routes: [`coordexp-swift-training-artifacts`](../openspec/specs/coordexp-swift-training-artifacts/spec.md)
for run/checkpoint publication and
[`coordexp-swift-infer-config-runtime`](../openspec/specs/coordexp-swift-infer-config-runtime/spec.md)
for explicit inference payload composition.

## Current config route

Open these roots first:

- `configs/coordexp_swift/prod/`
- `configs/coordexp_swift/smoke/`
- `configs/coordexp_swift/infer/`

The loader requires strict typed config resolution. Do not infer a current
schema from an archived YAML file or an old plan.

## Current semantic seams

- `src/data/` validates raw records and geometry.
- `src/templates/` renders prompt/assistant content and semantic spans.
- `src/qwen/encoding.py` aligns rendered spans with physical tokens and image
  positions.
- `src/packing/` creates physical packed segments and remaps supervision.
- `src/supervision/tokens.py` is the token-level record interface.
- `src/losses/runner.py` owns configured loss assembly and normalization.
- `src/runtime/train_runtime.py` owns the Accelerate-only replicated-DDP
  execution boundary, finite gates, optimizer, and scheduler behavior.
- `src/artifacts/run_writer.py` owns rank-zero `run.json`,
  `resolved_config.json`, and `logging.jsonl`; `src/artifacts/checkpoints.py`
  owns synchronized staged PEFT adapter and optional selected-token delta
  payloads plus `final.json` and `best.json`. Its
  `_run_exact_training_state_callback` seam is where the opt-in exact
  training-state sibling is published, after the inference payload commits and
  before either alias updates.
- `src/training/pack_cache.py` owns immutable cache v3 outside the run tree;
  `rebuild` publishes only to a previously absent semantic fingerprint target,
  while the run retains compact materialization bindings.
- `src/inference/backend.py` owns backend-neutral requests, results, dual
  likelihood semantics, validation, and session lifecycle.
- `src/inference/hf_backend.py` owns dynamic HF composition and generation;
  `src/inference/execution_model.py` owns immutable composed snapshots; and
  `src/inference/vllm_backend.py` owns offline vLLM generation and raw replay.
- `src/inference/runtime.py` owns the processor-only frontend and strict
  backend launch projection; `src/inference/vllm_qualification.py` retains
  explicit audit probes while normal sessions record a non-blocking operational
  preflight. BF16 vLLM is the normal throughput path and FP32 comparison is
  claim-specific diagnostic evidence.
  Unknown versions may attempt policy-only execution; raw tracing remains
  restricted to versions with known pre-processor logprob-capture ordering.
- `src/eval/detection_consumer.py` owns score-provenance validation, coordinate
  conversion, and detection metrics.

## Stable contract routes

Use the exact relevant spec, not a proposal copy:

- `openspec/specs/coordexp-swift-config-runtime/spec.md`
- `openspec/specs/coordexp-swift-data-template-encoding/spec.md`
- `openspec/specs/coordexp-swift-packing-forward/spec.md`
- `openspec/specs/coordexp-swift-supervision-losses/spec.md`
- `openspec/specs/coordexp-swift-pack-cache-semantic-identity/spec.md`
- `openspec/specs/coordexp-swift-adapters-embeddings-optim/spec.md`
- `openspec/specs/coordexp-swift-training-artifacts/spec.md`
- `openspec/specs/coordexp-swift-training-resume/spec.md`
- `openspec/specs/coordexp-swift-infer-pipeline/spec.md`
- `openspec/specs/coordexp-swift-infer-backend-trace/spec.md`
- `openspec/specs/coordexp-swift-infer-execution-model/spec.md`
- `openspec/specs/coordexp-swift-infer-scoring-artifacts/spec.md`
- `openspec/specs/coordexp-swift-detection-evaluator/spec.md`

## Historical route quarantine

The following names are historical, archived, or comparator-only in current
documentation and must not be used as current Swift entrypoints:

- `src/sft.py`, `src/trainers/`, `src/datasets/`, `src/detection/`, and the
  old `src/infer/` package;
- `configs/stage1/`, `configs/stage2/`, and
  `configs/archive/detection_scene_clean_break/`;
- dated historical plans, old architecture proposals, and archived
  OpenSpec changes.

Consult [`docs/history/README.md`](history/README.md) only for explicit
provenance reconstruction. Use
[`docs/architecture/README.md`](architecture/README.md) for accepted current
architecture. Do not
promote historical material by copying its wording into a canonical current
page.
