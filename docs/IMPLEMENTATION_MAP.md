---
doc_id: docs.implementation-map
layer: docs
doc_type: implementation-map
status: canonical
domain: repo
summary: Small source and test routing map for the current CoordExp-Swift implementation.
updated: 2026-08-20
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
| Training facade | `src/training/pipeline.py` | `tests/training/test_pipeline_assembly.py` |
| Model-free execution plan | `src/training/execution_plan.py` | `tests/training/test_pipeline_assembly.py` |
| Pre-model rank convergence | `src/training/control_plane.py` | `tests/runtime/test_rank_report_collective.py` |
| Initialized-run lifetime | `src/training/session.py` | `tests/training/test_training_session.py` |
| Cache admission and hydration | `src/training/cache_workflow.py`, `src/training/cache_contract.py` | `tests/training/test_pipeline_cache_preflight.py` |
| Completed-step reporting | `src/training/reporting.py` | `tests/training/test_pipeline_assembly.py` |
| Typed distributed metric reduction | `src/runtime/metrics.py` | `tests/runtime/test_metrics.py` |
| Optimizer-boundary decision and update receipt | `src/runtime/optimizer_boundary.py` | `tests/runtime/test_wave3_optimizer_boundary.py` |
| Rank-zero observation publication and derived sinks | `src/artifacts/observation_publisher.py` | `tests/artifacts/test_observation_publisher.py` |
| Cached micro-step schema | `src/training/micro_steps.py` | `tests/training/test_pack_cache_determinant_registry.py` |
| Forward-input preparation | `src/training/forward_input_provider.py` | `tests/training/test_forward_input_provider.py` |
| Planned-step loop | `src/training/supervised_trainer.py` | `tests/training/test_supervised_trainer.py` |
| Config schema and resolution | `src/config/loader.py`, `src/config/models.py`, `src/config/resolve.py` | `tests/config/` and strict-config tests |
| Raw JSONL and geometry | `src/data/` | `docs/data/CONTRACT.md`, `tests/data/` |
| Prompt and semantic spans | `src/templates/` | `tests/templates/` |
| Qwen load/encode/forward | `src/qwen/` | `tests/qwen/` |
| Packing | `src/packing/`, `src/training/pack_cache.py` | `tests/packing/`, `tests/training/test_pack_cache.py` |
| Token supervision | `src/supervision/` | `tests/supervision/` |
| Loss assembly | `src/losses/` | `tests/losses/` |
| Runtime and optimization | `src/runtime/`, `src/optim/`, `src/adapters/` | `tests/runtime/`, `tests/optim/`, `tests/adapters/` |
| Training run files | `src/artifacts/run_writer.py` (facade), `src/artifacts/run_schema.py`, `src/artifacts/run_state.py` | `tests/artifacts/test_run_artifacts.py`, `tests/artifacts/test_run_schema.py`, `tests/artifacts/test_run_state.py` |
| Generic content/weight/repository identity | `src/artifacts/identity.py` | `tests/artifacts/test_identity_compatibility.py` |
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
- `src/losses/runner.py` owns configured loss assembly and normalization over
  the closed term inventory in `src/losses/bindings.py`, which declares each
  implemented term's role, normalizer, and zero policy. That inventory is
  private and compiled with the code: there is no registry, import path, or
  callable config for selecting a loss.
- `src/runtime/train_runtime.py` owns the Accelerate-only replicated-DDP
  execution boundary, finite gates, optimizer, and scheduler behavior.
- `src/training/pipeline.py` is the training facade only: it builds the
  immutable model-free plan (`execution_plan.py`), opens the pre-model rank
  control plane (`control_plane.py`), admits the cache workflow
  (`cache_workflow.py`/`cache_contract.py`), and runs exactly one
  `TrainingSession` (`session.py`), which owns model/runtime assembly,
  exact-resume, eval/checkpoint/finalization, and the forward-input provider
  lifetime. `src/training/reporting.py` owns completed-step rows.
- One observation has exactly four owners and no generic coordinator:
  `src/runtime/metrics.py` owns typed cross-rank reduction (each metric
  declares its reducer; there is no name-derived or mean fallback),
  `src/runtime/optimizer_boundary.py` owns the all-rank boundary decision and
  the single update receipt, `src/training/reporting.py` owns canonical row
  construction, and `src/artifacts/observation_publisher.py` owns rank-zero
  publication plus the derived console and TensorBoard sinks. `session.py`
  composes them. The training facade is not a second row, reducer, or sink
  owner.
- `src/training/forward_input_provider.py` owns forward-input preparation.
  The strict config field `training.forward_input_provider_mode` is the only
  selector; no environment variable may replace it. `synchronous` is the
  default reference implementation, and `overlapped` is an explicit
  experimental selection with depth-one, CPU-only producer semantics. Every
  supported mode builds a provider; assembly never omits one.
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
