---
doc_id: docs.architecture.independent-architecture-review
layer: docs
doc_type: architecture-review
status: review
domain: architecture
summary: Independent architecture review of CoordExp current state, using the existing simplification proposal as hypothesis rather than ground truth.
updated: 2026-05-31
---

# CoordExp Independent Architecture Review

## Scope and stance

This document is an independent architecture review of the current CoordExp
repository. It uses the proposal documents in this directory as one input source:

- `docs/architecture/README.md`
- `docs/architecture/CODEBASE_SIMPLIFICATION_PROPOSAL.md`
- `docs/architecture/OWNERSHIP_BOUNDARIES.md`
- `docs/architecture/SIMPLIFICATION_ROADMAP.md`

Those proposal documents are treated as architectural hypotheses and migration
material, not as proof of the current architecture.

This review is grounded in the current repository docs, source layout, config
schemas, active config trees, and symbolic/code inspection. It does not propose
implementation patches and does not modify code.

No tests or smoke runs were executed for this review. The evidence scope is
source and documentation inspection only.

## Executive thesis

CoordExp is not architectureless. It is already converging around a coherent
research-runtime architecture:

```text
offline prepared JSONL
  -> geometry/image-root contract
  -> surface-specific dataset/rendering path
  -> YAML-first training or inference runtime
  -> canonical gt_vs_pred artifact family
  -> score-aware eval and typed metrics/provenance
```

The current center of gravity is not yet a clean package boundary such as
`src/training/` or a single `CoordExpRecord` IR. The actual center of gravity is
the combination of:

- canonical docs and stable specs that define what is reportable;
- strict config loading through `src/config/loader.py` and `src/config/schema.py`;
- the broad training entrypoint `src/sft.py`;
- the latest compact detection stack under `src/detection/`;
- the active Stage-2 trainer/runtime stack under `src/trainers/` plus `src/infer/`;
- the canonical artifact/eval contracts under `src/infer/`, `src/eval/`, and `src/metrics/`;
- provenance/bootstrap writers under `src/bootstrap/`.

The architecture proposal is mostly correct as a migration direction, especially
about reducing `src.sft` responsibility, making training pipelines executable,
clarifying lifecycle status, and converging metric/artifact contracts. Its main
weakness is that it sometimes describes target concepts as if they were already
the runtime truth. In particular:

- `TrainingSurfaceResolver` and `src/training/pipelines/*` are real but still
  shadow descriptors, not executable launch owners.
- `DataView / CoordExpRecord` does not exist as a production-wide canonical IR.
  Current data representations include raw JSONL dicts, `ConversationRecord`,
  `NormalizedDetectionSample`, `DetectionDocument`, rendered template spans,
  tokenized sidecars, and `gt_vs_pred` pixel artifacts.
- `DetectionTemplate` is canonical for strict latest compact surfaces, but
  standard Stage-1 rendering and inference salvage still use compatibility
  layers.
- `stage2_rollout_correction` is the only active public Stage-2 trainer/config
  variant, but `rollout_matching.*` remains the active runtime/backend/decode/eval
  namespace and `src/trainers/stage2_rollout_runtime.py` remains internal active
  code.
- `MetricEvent` exists and is canonical for several metric families, but not all
  metric producers are fully event-first yet.

The right next architecture move is not a broad package rename. It is to make
the already-emerging seams deeper:

```text
resolve config once
  -> choose a real executable training surface
  -> feed a normalized record/document IR
  -> render through one strict template owner per surface
  -> decode through explicit request/result objects
  -> emit artifact and metric events through one compatibility-aware layer
```

## Resolved next-action boundaries

These boundaries were resolved after the review to constrain the next
architecture action before implementation.

### Decision

The next architecture action is **detection-only first**. It should define the
canonical semantic in-memory object for active detection workflows, not a
universal `CoordExpRecord` for every historical or future CoordExp task family.

Within that detection scope, representation ownership should be layered:

| Representation | Owner | Canonical only for |
| --- | --- | --- |
| Raw JSONL records and image references | data/dataset loading | on-disk storage and intake contract |
| Detection semantic object | detection domain code | in-memory detection meaning |
| Rendered supervision text | detection template code | prompt/completion formatting |
| Token-level training view | detection tokenization/objective code | loss alignment, token spans, masks, and sidecars |
| Runtime rollout/correction view | Stage-2 training runtime | rollout assignments and residual correction targets |
| Decoded prediction object | inference/runtime parsing boundary | parsed model predictions plus invalid/drop metadata |
| Official eval rows/artifacts | eval/artifact code | `gt_vs_pred`, `gt_vs_pred_scored`, metrics, and provenance |

No layer should treat another layer's representation as canonical. Raw JSONL,
rendered assistant text, token spans, rollout targets, decoded predictions, and
eval rows are owned views, adapters, artifacts, or runtime projections around
the semantic detection object.

### Rationale

Detection is the active pressure point where CoordExp currently repeats the
same concepts across data rows, normalized samples, rendered strings, token
sidecars, rollout targets, decoded predictions, and eval artifacts. A
repo-wide universal record would be tempting, but it would likely become a
shallow abstraction before a second active task family proves the need.

### Consequence

The first implementation slice should prove the seam on an active detection
path before broad cleanup:

```text
raw detection JSONL
  -> DetectionDocument-like semantic object
  -> strict detection template render
  -> token/span supervision
  -> decoded prediction/eval artifact
```

Stage-2 rollout correction, inference, and eval should become downstream
consumers of this seam, but broad trainer refactors, retired public trainer
variants, historical runtime-fusion data paths, and artifact-contract changes
remain out of scope until the semantic seam is proven.

The seam design must cover **Stage-1 and Stage-2 together**. Stage-1 and
Stage-2 share image identity, image dimensions, geometry frame, object identity,
object ordering, class/text description, coordinate representation, template
semantics, parse validity, decoded prediction structure, and eval artifact
meaning. A Stage-1-only design would be too narrow.

The intended split is:

```text
shared detection semantic layer:
  DetectionDocument-like GT/image/object semantics

Stage-1 projection:
  DetectionDocument -> teacher-forced rendered target -> token/span supervision

Stage-2 projection:
  DetectionDocument + rollout predictions
    -> duplicate filtering / assignment / residual correction events
    -> correction render/token supervision

Inference projection:
  generated text -> decoded prediction objects + invalid/drop metadata

Eval projection:
  GT semantic objects + decoded predictions -> gt_vs_pred / scored artifacts
```

The first implementation slice may still use Stage-1 compact / teacher-forcing
detection as the safer proving path, but only if the design explicitly states
how Stage-2 rollout correction will consume the same semantic seam. "Stage-1
first" must not become "Stage-1 only."

The first seam migration must be **strictly behavior-preserving**. It may
change ownership and routing, but it must not intentionally change object
ordering, coordinate normalization, bbox/poly interpretation, template text,
chat-template boundary behavior, token labels or masks, loss semantics, rollout
assignment policy, duplicate filtering behavior, invalid/drop accounting,
`gt_vs_pred` / `gt_vs_pred_scored` schema, metric meaning, artifact names, or
config defaults.

Any change to those semantics must be split into a separate explicit decision.
If the change affects stable training/eval behavior, config schema, artifact
names, loss semantics, or normative metric meaning, it should be routed through
the appropriate docs and, when compatibility-sensitive, OpenSpec.

### Evidence

- Scope: `none-yet` for implementation; source/docs inspection only.
- Handles: `src/detection/ir.py`, `src/detection/template.py`,
  `src/detection/tokenization.py`, `src/detection/dataset.py`,
  `src/training/objectives/teacher_forcing.py`, `src/trainers/stage2_rollout_correction.py`,
  `src/infer/runtime.py`, `src/eval/detection_orchestrator.py`,
  `docs/data/CONTRACT.md`, `docs/eval/CONTRACT.md`, and this review.

## Evidence base

Primary current docs inspected:

- `docs/AGENT_INDEX.md`
- `docs/catalog.yaml`
- `docs/PROJECT_CONTEXT.md`
- `docs/SYSTEM_OVERVIEW.md`
- `docs/IMPLEMENTATION_MAP.md`
- `docs/data/README.md`
- `docs/data/CONTRACT.md`
- `docs/data/PREPARATION.md`
- `docs/data/PACKING.md`
- `docs/training/README.md`
- `docs/training/STAGE1_OBJECTIVE.md`
- `docs/training/STAGE2_RUNBOOK.md`
- `docs/training/METRICS.md`
- `docs/eval/README.md`
- `docs/eval/CONTRACT.md`
- `docs/eval/WORKFLOW.md`
- `docs/ARTIFACTS.md`

Primary code/config surfaces inspected:

- `src/sft.py`
- `src/config/loader.py`
- `src/config/schema.py`
- `src/config/rollout_matching_schema.py`
- `src/training/surfaces.py`
- `src/training/pipelines/*`
- `src/training_runtime/plan.py`
- `src/training_runtime/stage2_projection.py`
- `src/datasets/dense_caption.py`
- `src/datasets/builders/jsonlines.py`
- `src/datasets/geometry.py`
- `src/detection/data.py`
- `src/detection/ir.py`
- `src/detection/dataset.py`
- `src/detection/runtime.py`
- `src/detection/template.py`
- `src/detection/tokenization.py`
- `src/detection/objective.py`
- `src/training/objectives/teacher_forcing.py`
- `src/trainers/stage2_rollout_correction.py`
- `src/trainers/stage2_rollout_correction_impl.py`
- `src/trainers/stage2_rollout_runtime.py`
- `src/trainers/rollout_aligned_targets.py`
- `src/trainers/rollout_aligned_evaluator.py`
- `src/trainers/rollout_matching/*`
- `src/training/stage2/assignment.py`
- `src/training/stage2/duplicate_filter.py`
- `src/training/stage2/planners.py`
- `src/training/ordering.py`
- `src/infer/pipeline.py`
- `src/infer/runtime.py`
- `src/infer/backend.py`
- `src/infer/artifacts.py`
- `src/infer/rollout_dispatch.py`
- `src/eval/detection.py`
- `src/eval/detection_orchestrator.py`
- `src/eval/detection_records.py`
- `src/eval/orchestration.py`
- `src/eval/artifacts.py`
- `src/common/coord_standardizer.py`
- `src/common/prediction_parsing.py`
- `src/metrics/events.py`
- `src/training/observability/*`
- `scripts/run_infer.py`
- `scripts/evaluate_detection.py`
- `scripts/postop_confidence.py`
- `scripts/run_infer_eval.sh`
- `scripts/run_vis.sh`
- `configs/stage1/`
- `configs/stage2_rollout_correction/`
- `configs/infer/`
- `configs/eval/`

## 1. Current-state architecture

### 1.1 Authority and contract structure

The repository already separates authority layers clearly:

| Layer | Current role | Evidence |
| --- | --- | --- |
| `docs/` | Current operator-facing truth for workflows, routing, artifact names, recommended behavior | `docs/PROJECT_CONTEXT.md`, `docs/AGENT_INDEX.md`, `docs/catalog.yaml` |
| `openspec/specs/` | Stable compatibility contracts for training/eval/config/artifact semantics | referenced from `docs/AGENT_INDEX.md`, `docs/training/STAGE2_RUNBOOK.md`, `docs/eval/README.md` |
| `progress/` | Historical evidence, diagnostics, benchmark reports, and design derivation | `docs/PROJECT_CONTEXT.md` |
| `docs/architecture/` | Proposal material and this review, not executable truth | `docs/architecture/README.md` |

This is a strength. It means current behavior can usually be reconstructed by
following the docs spine before diving into source.

The weak spot is that docs sometimes describe future seams alongside current
ones. For example, the unified training surface vocabulary is current design
direction, but not yet the sole executable runtime shape.

### 1.2 Data and preprocessing

Current data architecture has one strong contract and multiple active
representations.

Canonical contract:

- JSONL records contain `images`, `objects`, `width`, and `height`.
- Objects use exactly one geometry field: `bbox_2d` or `poly`.
- Training coordinates are pre-normalized norm1000 integers or pre-tokenized
  `<|coord_k|>` values.
- Runtime resize is forbidden for coordinate data.
- Non-canonical bbox parameterizations such as `cxcy_logw_logh` and `cxcywh`
  must be authored as offline sibling branches.

Important current split:

| Path | Status | Role |
| --- | --- | --- |
| Legacy processed/preset roots such as `public_data/coco/rescale_32_1024_bbox_max60` | Active | Used by current configs and comparator surfaces |
| Phase-1 public-data views under `public_data/coco/views/**` with `meta.json` image-store semantics | Active in docs/code/provenance, not fully replacing all configs | Separates reusable image stores from annotation views |
| `src/datasets/dense_caption.py::BaseCaptionDataset` plus `src/datasets/builders/jsonlines.py::JSONLinesBuilder` | Active despite legacy naming | Standard Stage-1 JSON/CoordJSON dataset/render path |
| `src/detection/data.py::NormalizedDetectionSample` plus `src/detection/dataset.py::DetectionTrainingDataset` | Active for latest compact detection | Strict typed detection row path, currently bbox-oriented |
| `src/detection/ir.py::DetectionDocument` | Existing adapter/future semantic IR | Not yet the production-wide spine |

The proposal is correct that a canonical data IR would reduce repeated
transformations. It is incorrect if read as saying the repo already has a
production-wide `CoordExpRecord`/`DataView`.

### 1.3 Template, rendering, parsing, and token spans

Current template architecture is partially converged.

Strict owner:

- `src/detection/template.py` defines `DetectionSequenceTemplate`.
- Factory-visible strict template IDs are `stage1_json_pretty` and
  `compact_full`.
- `CompactFullTemplate` owns strict compact rendering, parsing, render spans,
  terminal semantics, and structural row behavior for latest compact surfaces.

Compatibility layers:

- `src/common/detection_sequence.py` remains a compatibility facade for
  generated-text repair and helper formats.
- `src/common/detection_compact_rows.py` owns stdlib-only compact marker/render
  helpers.
- Helper formats such as `compact_no_desc`, `compact_no_bbox`, and `compact_min`
  still exist as helper/compatibility formats, not strict factory template IDs.
- Inference parsing deliberately supports salvage paths through
  `src/common/prediction_parsing.py` and `src/common/coord_standardizer.py`.

Token/objective sidecars:

- `src/detection/tokenization.py` maps rendered char spans into token spans.
- `src/detection/objective.py` builds recursive-detection targets.
- `src/detection/teacher_forcing/target_builder.py` builds teacher-forcing target
  IR.
- `src/training/objectives/teacher_forcing.py` runs the teacher-forcing
  objective.

Current architectural truth:

```text
strict latest compact training:
  RawDetectionRow
    -> NormalizedDetectionSample
    -> DetectionTemplate render
    -> tokenization/spans
    -> objective sidecars

standard Stage-1:
  raw JSONL dict / ConversationRecord
    -> JSONLinesBuilder
    -> CoordJSON or compatibility compact rendering
    -> Swift template encode

inference:
  prompt/runtime path
    -> generated text
    -> strict parse where possible
    -> salvage parse where configured/needed
    -> canonical gt_vs_pred artifact
```

This is a reasonable research architecture, but it is not yet the single
canonical `DetectionTemplate` spine proposed in the simplification document.

### 1.4 Training infrastructure

The live training architecture is still centered on `src/sft.py`.

Actual current responsibilities in `src/sft.py` include:

- trainer class selection;
- config loading and runtime mutation glue;
- dataset and eval dataset construction;
- static packing and encoded-sample cache preflight;
- effective-batch and gradient-accumulation derivation;
- detection runtime support and latest compact runtime handoff;
- Stage-2 projection injection;
- manifest/effective-runtime/run-metadata writing;
- callback/data-collator wiring;
- compatibility aliases for logic that has moved elsewhere.

Extracted but not fully owning execution:

| Module | Current role |
| --- | --- |
| `src/training_runtime/plan.py` | Import-safe trainer-variant setup contract, collator family, packing owner, Stage-2 namespace requirement |
| `src/training/surfaces.py` | Strict shadow surface resolver and objective-profile validator |
| `src/training/pipelines/stage1_json_ce.py` | Shadow descriptor only |
| `src/training/pipelines/stage1_compact_trie_ce.py` | Shadow descriptor only |
| `src/training/pipelines/stage2_rollout_correction.py` | Shadow descriptor only |
| `src/detection/runtime.py` | Latest compact detection runtime preflight, shim resolution, dataset building |
| `src/bootstrap/*` | Manifest, run metadata, experiment manifest, and policy provenance helpers |

The proposal is correct that `src.sft` is the primary architecture hotspot.

### 1.5 Stage-1 training

Current Stage-1 has at least three important surfaces:

| Surface | Current status | Canonical handles |
| --- | --- | --- |
| Stage-1 JSON CE baseline | Active baseline/regression surface | `configs/stage1/sft_base.yaml`, `configs/stage1/profiles/`, `src/datasets/dense_caption.py`, `src/trainers/losses/coord_soft_ce_w1.py` |
| Stage-1 compact teacher-forcing | Active compact-full direction | `configs/stage1/teacher_forcing/`, `src/detection/runtime.py`, `src/detection/dataset.py`, `src/training/objectives/teacher_forcing.py` |
| Stage-1 compact recursive detection CE | Legacy/comparator and ablation surface | `configs/stage1/recursive_detection_ce/`, `src/detection/objective.py`, recursive detection metrics |

Important distinction:

- New compact-full Stage-1 direction is teacher-forcing based.
- Recursive-detection CE configs are comparator/history surfaces, not the active
  teacher-forcing route.
- The proposed `stage1_compact_trie_ce` surface name is directionally aligned
  with compact-full token/trie objectives, but the current active config family
  is named and authored around `objective.id: teacher_forcing`.

This creates a terminology hazard. A target architecture should either make the
surface name match the active objective vocabulary or clearly define how
`stage1_compact_trie_ce` includes the current teacher-forcing objective atoms.

### 1.6 Stage-2 rollout/correction training

Current active Stage-2 public contract:

- `custom.trainer_variant: stage2_rollout_correction`
- `stage2_rollout_correction.pipeline.objective[]` contains exactly one enabled
  `residual_set_correction`
- `stage2_rollout_correction.pipeline.diagnostics[]` is empty
- active configs live under `configs/stage2_rollout_correction/`
- old public variants fail fast:
  `stage2_ab_training`, `stage2_two_channel`, `rollout_matching_sft`,
  `stage2_rollout_aligned`, and public `stage2_rollout_runtime`

Current split that matters:

| Namespace/module | Current ownership |
| --- | --- |
| `stage2_rollout_correction.pipeline.*` | Active objective ownership |
| `stage2_rollout_correction.correction.*` | Correction target and policy knobs |
| `rollout_matching.*` | Still active runtime/backend/decode/eval namespace |
| `src/trainers/stage2_rollout_correction.py` | Public trainer import facade |
| `src/trainers/stage2_rollout_correction_impl.py` | Main Stage-2 trainer implementation |
| `src/trainers/stage2_rollout_runtime.py` | Internal active trainer-owned rollout runtime helper/facade |
| `src/infer/runtime.py`, `src/infer/backend.py`, `src/infer/rollout_dispatch.py` | Shared decode/backend request projection and rollout dispatch |
| `src/training/stage2/assignment.py` | Greedy IoU assignment seam |
| `src/training/stage2/duplicate_filter.py` | Duplicate filtering seam |
| `src/training/stage2/planners.py` | Shadow planning and legacy realizer adapter |

The proposal is correct that public Stage-2 has collapsed toward
`stage2_rollout_correction`. It is incomplete if it implies all old runtime
names are gone. The public variants are gone; internal runtime/helper names are
not.

### 1.7 Inference and decoding runtime

Current inference architecture is partly unified and partly compatibility-bound.

Stable operator path:

- `scripts/run_infer.py --config configs/infer/...`
- `src/infer/pipeline.py::run_pipeline`

Compatibility paths:

- `scripts/run_infer.py` still supports legacy flag-only invocation.
- Legacy flags can override YAML fields when `--config` is present.
- `scripts/run_infer_eval.sh` is explicitly legacy/debug and refuses official
  COCO/LVIS/both metrics because it cannot prove scored-artifact provenance.
- `scripts/run_vis.sh` is manual/debug.

Runtime seams:

- `src/infer/runtime.py::DetectionDecodeRequest` is the shared decode request.
- `src/infer/runtime.py::build_decode_request_from_infer_config` maps infer YAML
  into that request.
- `src/infer/runtime.py::build_decode_request_from_rollout_owner` still resolves
  Stage-2 rollout facts from owner-like trainer objects.
- `src/infer/runtime.py::InferenceRuntime` normalizes legacy backend results into
  `DetectionDecodeResult`.
- `src/infer/backend.py` owns HF/vLLM backend generation and trace normalization.
- `src/infer/rollout_dispatch.py` bridges rollout calls to concrete handles.

Notable current-state correction:

- There is no `src/infer/engine.py` or `src/infer/backends.py` file in the
  current tree. Current implementation uses `src/infer/runtime.py` and
  `src/infer/backend.py` plus focused vLLM modules.

The proposal's `DecodeRuntime` recommendation is directionally right, but the
current code already has `DetectionDecodeRequest` and `InferenceRuntime` rather
than a clean `DecodeRuntime.generate(DecodeBatchRequest)` abstraction. Stage-2
still partially resolves decode state from trainer/owner-shaped objects.

### 1.8 Evaluation and metrics

Current eval architecture is strong and more decomposed than the proposal might
imply.

Eval path:

```text
gt_vs_pred.jsonl or gt_vs_pred_scored.jsonl
  -> scripts/evaluate_detection.py
  -> src/eval/detection.py facade
  -> src/eval/detection_orchestrator.py
  -> src/eval/detection_records.py
  -> COCO/LVIS/F1-ish/duplicate guard modules
  -> metrics.json, per_image.json, sidecars, overlays
```

Important artifact contract:

- `gt_vs_pred.jsonl` is the raw inference/debug artifact.
- `gt_vs_pred_scored.jsonl` is required for official COCO/LVIS/both metrics.
- Comparable scored artifacts need score-bearing provenance.
- `resolved_config.path` lets downstream jobs recover the authoritative
  `resolved_config.json`.
- Guarded outputs are additive companions, not replacements for raw artifacts.

Metric architecture:

- `src/metrics/events.py::MetricEvent` exists and has explicit identity axes.
- `flatten_metric_events` reduces typed events and emits identity-checked legacy
  aliases.
- `src/training/observability/events.py::DiagnosticEvent` exists for bounded
  structured diagnostics.
- `src/training/observability/legacy.py::adapt_legacy_metric` tolerates
  historical flat metrics.

Current limitation:

- `MetricEvent` is canonical for several Stage-1 and compact metric families,
  but Stage-2 and offline evaluation still expose many flat metric families and
  artifact-level metrics. The event-first architecture is present but not yet
  universal.

### 1.9 Artifacts and provenance

Artifact architecture is a genuine strength of the repository.

Training artifacts:

- `resolved_config.json`
- `runtime_env.json`
- `effective_runtime.json`
- `pipeline_manifest.json` when applicable
- `experiment_manifest.json`
- `train_data_provenance.json`
- `eval_data_provenance.json`
- `run_metadata.json`
- `config_source.yaml`
- `base_config_source.yaml`
- `monitor_dumps/`
- `eval_detection/step_<global_step>/` when materialized

Inference/eval artifacts:

- `gt_vs_pred.jsonl`
- `pred_token_trace.jsonl`
- `pred_confidence.jsonl`
- `gt_vs_pred_scored.jsonl`
- `gt_vs_pred_guarded.jsonl`
- `gt_vs_pred_scored_guarded.jsonl`
- `summary.json`
- `resolved_config.json`
- `resolved_config.path`
- `metrics.json`
- `metrics_guarded.json`
- `per_image.json`
- `per_class.csv`
- `coco_gt.json`
- `coco_preds.json`
- `matches*.jsonl`
- `duplicate_guard_report.json`
- `vis_resources/gt_vs_pred.jsonl`

Ownership is distributed but documented:

- `src/bootstrap/*` for training manifests/provenance;
- `src/infer/artifacts.py` for infer summaries, score provenance, and artifact
  facts;
- `src/eval/artifacts.py` and `src/eval/orchestration.py` for evaluator outputs;
- `docs/ARTIFACTS.md` for the cross-cutting artifact contract.

The proposal correctly treats artifacts as architecture, not leftovers.

## 2. Architectural diagnosis

### 2.1 What is already converging well

| Convergence | Evidence |
| --- | --- |
| YAML-first workflows | `src/config/loader.py`, `src/config/schema.py`, `scripts/run_infer.py --config`, Stage-2 runbook |
| Offline-prepared JSONL and no runtime resize | `docs/data/CONTRACT.md`, `docs/data/PREPARATION.md`, dataset/runtime guards |
| Public Stage-2 collapse to `stage2_rollout_correction` | `configs/stage2_rollout_correction/`, strict schema rejection of old variants |
| Compact template ownership | `src/detection/template.py`, strict factory IDs |
| Shared inference decode request vocabulary | `DetectionDecodeRequest`, infer and rollout decode builders |
| Canonical artifact family | `gt_vs_pred*.jsonl`, `resolved_config.path`, score provenance, guarded companions |
| Metric identity direction | `MetricEvent`, alias registry, reserved alias collision guard |
| Provenance-first execution | `effective_runtime.json`, `pipeline_manifest.json`, `experiment_manifest.json`, `run_metadata.json` |

### 2.2 Redundancy and drift

#### `src.sft` remains too broad

`src.sft.py` is the largest remaining orchestration funnel. It owns or bridges
too many concepts:

- config -> runtime mutation;
- trainer selection;
- dataset construction;
- packing/cache policy;
- detection runtime preflight;
- Stage-2 projection;
- artifact payload construction;
- callback/collator/trainer setup.

The module has become the place where architecture decisions meet. That is
useful for compatibility but poor for locality.

#### Shadow surfaces are not deep modules

`src/training/surfaces.py` validates a clean top-level domain model:

```text
run, surface, data, template, supervision, objectives, observability, artifacts, runtime, experimental
```

But the pipeline classes currently expose identity descriptors only. They do not
own dataset construction, collator selection, trainer construction, objective
wiring, runtime policy, or artifact emission.

This is a classic shallow seam:

- The interface says "pipeline".
- The implementation is mostly a name and lifecycle.
- The real complexity remains in `src.sft` and specialized modules.

#### Stage-2 names are cleaned publicly but not internally

Publicly retired:

- `stage2_ab_training`
- `stage2_two_channel`
- `rollout_matching_sft`
- `stage2_rollout_aligned`
- public `stage2_rollout_runtime`

Still active internally:

- `rollout_matching.*` runtime config namespace;
- `src/trainers/stage2_rollout_runtime.py`;
- `src/trainers/rollout_matching/*`;
- `rollout_aligned_targets.py`;
- `rollout_aligned_evaluator.py`.

This is not wrong. It reflects an incremental migration. But it is a source of
confusion because the same word can be retired at the public config surface and
active in implementation internals.

#### Data has a strong contract but not one representation

Current data concepts include:

- raw JSONL dicts;
- `ConversationRecord`;
- `BaseCaptionDataset`;
- `JSONLinesBuilder`;
- `RawDetectionRow`;
- `NormalizedDetectionSample`;
- `DetectionDocument`;
- rendered assistant text and char spans;
- tokenized labels/role spans;
- recursive-detection sidecars;
- teacher-forcing target IR;
- standardized pixel `gt_vs_pred` records.

Some of these are necessary, but the current system repeats validation,
geometry normalization, bbox checks, object ordering, and parse/salvage policy
across layers.

#### Strict parsing and salvage parsing are both current

Strict parsing is needed for training/eval contract safety. Salvage parsing is
needed for inference diagnostics and robust artifact production. The issue is
not their coexistence. The issue is that current naming can make "parser"
ambiguous without saying:

- strict template parser;
- compatibility compact parser;
- CoordJSON salvage parser;
- prediction text standardizer;
- evaluator artifact ingestion.

#### Metric identity is partially, not fully, canonical

`MetricEvent` is a good deepening direction. It captures denominator, reducer,
semantic axes, objective identity, parser mode, and aliases. However, Stage-2
telemetry and offline eval still expose a large set of flat families. Some are
valid artifacts or operator metrics, but the semantic identity layer is not yet
universal.

#### Config system is strict but conceptually split

Current config reality:

- `TrainingConfig` and `DetectionTrainingConfig` both exist.
- Stage-2 uses `custom.trainer_variant: stage2_rollout_correction` plus
  `stage2_rollout_correction.*` plus active `rollout_matching.*`.
- Stage-1 standard paths still use `custom.*` heavily.
- Latest detection configs reject `custom` entirely and use top-level
  `data`, `prompt`, `detection_template`, `token_rows`, `objective`, `packing`,
  `evaluation`, and `validation`.
- Shadow unified surfaces use a different closed top-level domain set.

The proposal's domain schema is directionally right, but the current repository
has not selected it as the only public config surface.

### 2.3 Technical debt and legacy layers

| Debt | Why it matters | Current severity |
| --- | --- | --- |
| Empty/remnant `configs/stage2_two_channel/` directory | Misleads navigation even though active YAML is gone | Low |
| `dense_caption` naming for active Stage-1 dataset path | Obscures current detection/grounding role | Medium |
| `src/trainers/stage2_rollout_runtime.py` active internal name | Looks retired from public docs but remains implementation-critical | Medium |
| `rollout_matching.*` active runtime namespace | Correct but confusing beside removed `rollout_matching.pipeline` | Medium |
| `src/common/detection_sequence.py` compatibility facade | Needed for salvage/helpers, but competes with strict template ownership | Medium |
| `src.sft.py` compatibility aliases and orchestration logic | Centralizes too much change risk | High |
| Multiple data/render/parse representations | Makes geometry and ordering invariants harder to audit | High |
| Partial `MetricEvent` adoption | New metric names can still drift unless event-first contract spreads | Medium |

## 3. Comparison against the existing proposal

### 3.1 Major recommendations

| Proposal recommendation | Verdict | Reasoning |
| --- | --- | --- |
| A. Activate surface-owned training pipelines | Agree | Current `TrainingSurfaceResolver` and pipeline descriptors are real but shadow-only. The recommendation addresses the main training architecture gap. |
| B. Introduce a canonical data IR | Agree with modification | Needed, but the repo already has `NormalizedDetectionSample` and `DetectionDocument`. The migration should first decide whether to promote `DetectionDocument`, replace it, or create `CoordExpRecord` beneath it. |
| C. Unify offline inference and Stage-2 rollout decode | Partially agree | Shared decode request/result concepts already exist. The missing piece is removing owner-shaped trainer resolution and promoting an explicit batch runtime API. |
| D. Make `MetricEvent` canonical | Partially agree | `MetricEvent` already exists and is well-designed. The work is adoption and boundary clarification, not greenfield creation. |
| E. Convert config into a domain plan | Agree with caution | Current config is strict but split across legacy `custom.*`, latest detection schema, Stage-2 schema, and shadow domains. A `RunPlan` is valuable, but migration must preserve ms-swift projection and current active config ergonomics. |
| F. Create an architecture lifecycle registry | Agree | The repo already has lifecycle language in docs/tests, but no single machine-readable registry. This would reduce retired-name confusion. |

### 3.2 Proposal claims that are correct

| Claim | Review verdict | Evidence |
| --- | --- | --- |
| `src.sft` is still a broad orchestration funnel | Correct | Symbol overview and line-level search show trainer selection, dataset build, runtime policy, Stage-2 projection, and manifest writing in `src/sft.py` |
| Training is YAML-first | Correct | `ConfigLoader`, Stage-2 runbook, inference/eval workflow |
| Runtime image resizing is forbidden | Correct | Data contract, preprocessing docs, dataset no-resize guards |
| Public Stage-2 collapsed toward `stage2_rollout_correction` | Correct | Active configs and schema reject old variants |
| `rollout_matching.pipeline` is retired | Correct | Schema/projection reject it |
| Artifact contracts are central architecture | Correct | `docs/ARTIFACTS.md`, infer/eval artifacts, score provenance |
| Metric identity should not be flat-key-only | Correct and partly implemented | `MetricEvent`, alias registry, docs/training/METRICS.md |

### 3.3 Proposal claims that are incomplete or risky

| Claim | Review verdict | Correction |
| --- | --- | --- |
| Three canonical training surfaces are already executable owners | Incomplete | They are shadow design vocabulary. Live launch still routes through `src.sft.py` and specialized runtime modules. |
| `stage2_rollout_runtime` is retired | Partially wrong | Retired as public trainer variant. `src/trainers/stage2_rollout_runtime.py` remains active internal code. |
| `rollout_matching` is retired | Partially wrong | `rollout_matching.pipeline` is retired. `rollout_matching.*` runtime/backend/decode/eval config remains active for Stage-2. |
| Canonical `DataView / CoordExpRecord` is the current architecture | Wrong as current-state description | No production-wide `CoordExpRecord` exists. Current IR candidates are split. |
| `DetectionTemplate` is the single render/parse layer | Incomplete | True for strict latest surfaces. Standard Stage-1 and inference salvage still use compatibility layers. |
| Runtime bbox conversion is forbidden | Needs precision | Training must not convert canonical sources into non-canonical model-facing branches at runtime. Inference/eval still converts non-canonical predictions back to canonical pixel `xyxy`. |
| Config should just become `run/surface/data/...` | Directionally right but oversimplified | Current latest detection config already has a different top-level schema, and ms-swift projection remains a real boundary. |

### 3.4 Proposal items that should be postponed or narrowed

| Proposal item | Why narrow it |
| --- | --- |
| Large `src/data/` package migration | Risk of renaming active `src/datasets/` without first choosing the canonical record/document interface |
| Deleting compatibility parsers | Inference diagnostics still need salvage behavior and raw-output evidence |
| Removing `rollout_matching.*` namespace wholesale | It still owns active Stage-2 runtime/backend/decode/eval knobs |
| Treating `stage1_compact_trie_ce` as production truth | Active compact Stage-1 configs are teacher-forcing oriented; naming must match semantics before promotion |
| MetricEvent-first offline eval rewrite | Valuable, but artifact-level metric contracts already work and need parity-focused migration |

## 4. Recommended target architecture

### 4.1 Target principles

Use the proposal's core direction, but phrase it around current repo facts:

1. One current concept should have one owner.
2. Public lifecycle must be explicit: active, compatibility, retired.
3. Runtime should validate prepared data, not repair it.
4. Training surfaces should own execution, not only identity.
5. Strict template parsing and inference salvage parsing should be separate,
   named policies.
6. Decode comparability should be expressed through explicit request/result
   objects, not trainer owner objects.
7. Artifacts and metrics are contracts, not byproducts.
8. Compatibility aliases must be tolerant-read, not clean-write defaults.

### 4.2 Canonical execution paths

#### Training

Target training flow:

```text
ConfigLoader
  -> ResolvedTrainingRun / RunPlan
  -> executable TrainingPipeline
  -> DatasetBundle
  -> CollatorBundle
  -> TrainerBundle
  -> Artifact/Metric/Diagnostic writers
```

Recommended executable surfaces:

| Target surface | Current source of truth |
| --- | --- |
| `stage1_json_ce` | Standard Stage-1 JSON/CoordJSON baseline |
| `stage1_compact_teacher_forcing` or carefully defined `stage1_compact_trie_ce` | Active compact-full teacher-forcing direction |
| `stage2_rollout_correction` | Active Stage-2 rollout prefix plus residual/GT correction |

Naming note:

- If the surface remains `stage1_compact_trie_ce`, document exactly how it maps
  to `objective.id: teacher_forcing`, valid-set marginal atoms, token CE, trie
  CE, and coordinate objectives.
- If the active direction is semantically teacher-forcing first, consider
  renaming the target surface before making it the public stable surface.

#### Data

Target data flow:

```text
JSONL/view metadata/image root
  -> DataView
  -> CoordExpRecord or DetectionDocument
  -> surface-specific normalized sample
  -> template rendering
  -> tokenization/objective/eval
```

Recommendation:

- Do not introduce `CoordExpRecord` as a parallel second IR until the role of
  `DetectionDocument` is decided.
- The likely deep module is:

```text
Raw JSONL + image-root metadata
  -> one validated record/document object
```

That object should own:

- image path resolution semantics;
- width/height;
- object list;
- source object identity;
- bbox/poly geometry;
- coordinate surface;
- prepared bbox provenance;
- object ordering plan;
- metadata needed by training, inference, eval, and visualization.

#### Templates and parsers

Target ownership:

| Concern | Owner |
| --- | --- |
| Strict training/eval template IDs | `src/detection/template.py` or submodules under `src/detection/templates/` |
| Low-level compact row grammar | `src/common/detection_compact_rows.py` until migrated |
| Compatibility generated-output salvage | Explicit compatibility parser module, not hidden inside the strict template owner |
| Inference parser policy | `src/infer/parsing.py` or `src/detection/evaluation.py` with explicit metric-eligibility flags |
| Token span projection | `src/detection/tokenization.py` |

The important target is not "one parser function." It is one named parser
policy per evidence surface.

#### Decode runtime

Target decode flow:

```text
PromptBundle[]
  + DecodeBatchRequest
  + BackendSpec
  -> DecodeRuntime.generate(...)
  -> DetectionDecodeResult[]
```

Current `DetectionDecodeRequest` should probably survive. The missing target
object is a batch request that also carries:

- prompt bundles;
- model/backend identity;
- adapter sync identity;
- provenance/fingerprint facts;
- trace requirements;
- parse policy expectations.

Stage-2 should build this request from explicit runtime facts, not from a whole
trainer owner object.

#### Evaluation and metrics

Target eval/metric flow:

```text
canonical artifact
  -> evaluator records
  -> metric events and artifact reports
  -> flat aliases / metrics.json / summaries
```

The current artifact split should remain:

- raw `gt_vs_pred.jsonl` for debugging and F1-ish/debug surfaces;
- scored `gt_vs_pred_scored.jsonl` for official metrics;
- guarded companions as additive safety views;
- `resolved_config.path` for downstream provenance recovery.

`MetricEvent` should become the source of scalar semantic identity where
practical, but offline evaluation can still own rich artifacts and reports.

### 4.3 Ownership boundaries

Recommended target module ownership:

| Concept | Target owner | Notes |
| --- | --- | --- |
| YAML inheritance and strict validation | `src/config/` | Keep list replacement explicit or replace objective lists with keyed objective maps |
| Resolved run plan | `src/config/` plus `src/training/` | Should be immutable before execution |
| Training surface execution | `src/training/pipelines/` | Must own build/run methods, not only identity |
| Standard dataset contract and image-root resolution | Either promoted `src/data/` or current `src/datasets/` with renamed deeper modules | Avoid ceremonial package move before interface is chosen |
| Strict detection document/template | `src/detection/` | Keep strict surface separate from inference salvage |
| Stage-2 assignment/planning | `src/training/stage2/` | Already moving in right direction |
| Decode runtime | `src/infer/` | Reduce owner-shaped Stage-2 adapters over time |
| Eval artifact ingestion and metrics | `src/eval/` | Keep `src/eval/detection.py` as facade only if compatibility needs it |
| Metric identity and aliasing | `src/metrics/` | Continue event-first migration |
| Observability diagnostics | `src/training/observability/` | Keep bounded profiles and separate from scalar metrics |
| Training artifacts/provenance | `src/bootstrap/` and training artifact module | Avoid rebuilding artifact payloads in `src.sft` |

## 5. Refactoring strategy

### 5.1 Highest-impact changes

#### 1. Make training pipelines executable

Impact:

- Reduces `src.sft` complexity.
- Makes surface ownership real.
- Makes future experiments ask "which surface owns this?" before adding flags.

Sequence:

1. Define an executable pipeline protocol.
2. Implement no-op parity wrappers that call current `src.sft` helpers.
3. Route one Stage-1 JSON smoke through the pipeline.
4. Route one compact teacher-forcing smoke through the pipeline.
5. Route Stage-2 config setup through a Stage-2 pipeline.
6. Keep `src.sft` as CLI-compatible entrypoint.

Verification:

- `--cfg-only` parity for representative configs.
- Artifact parity for `resolved_config.json`, `effective_runtime.json`, and
  manifests.
- Existing targeted training runtime tests.

#### 2. Decide the canonical record/document IR

Impact:

- Reduces repeated geometry and image-root handling.
- Clarifies whether `DetectionDocument` is the semantic IR or a temporary
  adapter.
- Makes train/infer/eval parity easier to audit.

Sequence:

1. Inventory fields required by `BaseCaptionDataset`, `DetectionTrainingDataset`,
   inference standardization, eval, visualization, and artifact writers.
2. Decide whether to promote `DetectionDocument`, create `CoordExpRecord`, or
   layer one below the other.
3. Add constructors from raw JSONL and Phase-1 view metadata.
4. Adapt one Stage-1 standard path and one latest compact path.
5. Only then consider package migration from `src/datasets/` to `src/data/`.

Verification:

- Same fixture through standard Stage-1 render, compact render, inference
  standardization, and eval ingestion.
- Geometry invariants for `bbox_2d`, `poly`, norm1000, coord-token surfaces,
  and non-canonical bbox branches.

#### 3. Replace owner-shaped decode with explicit requests

Impact:

- Makes offline inference and Stage-2 rollout more comparable.
- Reduces hidden trainer coupling.
- Clarifies backend/provenance/fingerprint ownership.

Sequence:

1. Keep `DetectionDecodeRequest`.
2. Add `BackendSpec` and `DecodeBatchRequest` around it.
3. Adapt offline inference first.
4. Adapt Stage-2 rollout dispatch second.
5. Keep owner-shaped wrappers as compatibility only.

Verification:

- Decode policy fingerprint parity.
- Prompt token parity.
- HF/vLLM trace normalization tests.
- Stage-2 rollout runtime tests.

#### 4. Move artifact payload construction out of `src.sft`

Impact:

- Makes reproducibility contracts easier to preserve during training refactors.
- Concentrates artifact ownership.

Sequence:

1. Extract effective-runtime payload assembly into a training artifact module.
2. Keep exact artifact names and payload fields.
3. Move Stage-2 policy provenance assembly to a single owner.
4. Keep `src.sft` only delegating.

Verification:

- Manifest/artifact tests.
- Snapshot selected `effective_runtime.json` fields before/after.

#### 5. Create a lifecycle registry

Impact:

- Makes active/compatibility/retired state machine-checkable.
- Prevents retired experiments from leaking into configs/scripts.

Initial registry should include:

- training surfaces;
- public trainer variants;
- config namespaces;
- template IDs and helper formats;
- artifact names and aliases;
- scripts/wrappers;
- retired objective/mechanism names.

Verification:

- Active configs do not reference retired names.
- Compatibility entries require owner/reason/expiry.
- Retired names only appear in rejection tests, historical docs, or archived
  evidence.

### 5.2 Low-risk cleanup opportunities

| Cleanup | Rationale | Risk |
| --- | --- | --- |
| Remove or tombstone empty `configs/stage2_two_channel/` | Avoid stale navigation | Low |
| Add explicit lifecycle labels to `src/trainers/stage2_rollout_runtime.py` docstring | Clarify internal-active vs public-retired | Low |
| Add a small architecture note explaining `rollout_matching.*` current runtime ownership | Avoid config mistakes | Low |
| Tighten docs around Stage-1 compact teacher-forcing vs recursive-detection comparator | Prevent wrong config copying | Low |
| Rename docs references to non-existent `src/infer/engine.py` / `src/infer/backends.py` where present | Keep navigation current | Low |
| Add a short parser-policy matrix | Clarify strict vs salvage parse eligibility | Low |

### 5.3 Migration sequencing

Recommended sequence:

1. Lifecycle registry and tombstones.
2. Documentation-only clarification of current active/compatibility/retired
   names.
3. Extract artifact/effective-runtime payload builders from `src.sft`.
4. Make Stage-1 JSON pipeline executable.
5. Make compact teacher-forcing pipeline executable.
6. Make Stage-2 rollout-correction setup pipeline executable.
7. Decide and introduce/promote canonical record/document IR.
8. Convert owner-shaped decode to explicit batch request.
9. Expand `MetricEvent` adoption across Stage-2 and offline eval where useful.
10. Delete compatibility only after registry and tests prove no active callers.

This order favors low-risk ownership clarification before behavior-sensitive
data/decode changes.

### 5.4 Risks

| Risk | Why it matters | Mitigation |
| --- | --- | --- |
| ms-swift projection drift | Training config and template defaults can change silently | Snapshot resolved TrainArguments-relevant fields |
| Geometry/image alignment drift | Invalidates grounding/eval claims | Keep no-resize and offline-prepared invariants fail-fast |
| Stage-2 DDP/post-rollout packing drift | Can deadlock or change loss normalization | Preserve current Stage-2 runtime tests and manifest fields |
| Prompt/token parity drift | Breaks train/infer/rollout comparability | Add prompt-token parity checks |
| Artifact contract drift | Breaks benchmark reproducibility | Preserve artifact names or migrate with explicit docs/spec tests |
| Parser-policy confusion | Salvage output can be mistaken for metric-bearing strict output | Add explicit parser policy in artifacts/provenance |
| Metric alias collision | Flat dashboards can lie about semantics | Continue `MetricEvent` alias registry and reserved-key guard |

## 6. Review findings by priority

### P1 findings

1. `src.sft` remains the primary orchestration hotspot.

Impact:

Many policies still meet in one entrypoint, so refactors can accidentally change
training behavior, artifacts, cache/packing policy, or Stage-2 runtime setup.

Fix direction:

Make surface pipelines executable and move artifact/runtime helpers behind
deeper modules while preserving the `python -m src.sft --config ...` entrypoint.

2. Training surfaces are not executable owners yet.

Impact:

The architecture says "surface-owned pipeline", but execution still says
"global runner plus conditionals."

Fix direction:

Promote pipeline descriptors to real build/run owners through parity-preserving
steps.

3. No production-wide data IR exists.

Impact:

Data/geometry/order/metadata knowledge is repeated across dataset builders,
template renderers, inference standardization, and eval.

Fix direction:

Choose and promote a canonical record/document IR before moving packages.

4. Stage-2 has public cleanup but internal naming split.

Impact:

Implementers can confuse retired public variants with active internal helper
modules and active runtime namespaces.

Fix direction:

Add lifecycle labels and avoid deleting internal names until explicit
replacement seams exist.

### P2 findings

1. `MetricEvent` is strong but partially adopted.

Fix direction:

Convert metric producers one family at a time and keep alias parity tests.

2. Parser policy is implicit.

Fix direction:

Name strict, compatibility, and salvage parser modes in artifacts and docs.

3. Config domain model is split.

Fix direction:

Build a real `RunPlan` projection layer before forcing public config migration.

4. Latest compact detection is bbox-oriented while global JSONL supports `poly`.

Fix direction:

Document bbox-only support for latest compact paths or deliberately add polygon
support through data, template, spans, targets, parse, and eval.

5. Empty/remnant directories and historical names still mislead navigation.

Fix direction:

Remove, tombstone, or lifecycle-register them.

## 7. Minimal target concept set

The smallest concept set that explains most current CoordExp behavior is:

1. PreparedDataContract

Strict JSONL plus image-root semantics, width/height, `bbox_2d|poly`, coordinate
surface, object ordering, and offline preparation provenance.

2. TrainingSurface

One of the current active executable surfaces:

- Stage-1 JSON CE baseline;
- Stage-1 compact teacher-forcing/trie direction;
- Stage-2 rollout correction.

3. DetectionDocument or CoordExpRecord

One validated record/document representation that can feed templates, training,
inference, eval, and visualization.

4. DetectionTemplate

Strict surface renderer/parser/span owner, distinct from compatibility salvage
parsers.

5. Objective

Surface-bound supervision primitive with typed atoms, not loose flat config
modules.

6. DecodeRequest/DecodeRuntime

Explicit generation request/result/fingerprint contract shared by offline
inference and Stage-2 rollout.

7. Artifact

Canonical output family with provenance: raw, scored, guarded, eval sidecars,
and run manifests.

8. MetricEvent/DiagnosticEvent

Typed scalar identity and bounded structured diagnostics, flattened only at the
logging/output boundary.

9. LifecycleState

Active, compatibility, or retired classification for config keys, variants,
templates, artifacts, scripts, and mechanisms.

## 8. Recommended next decision

Before implementing the proposal roadmap, make one architecture decision:

```text
What is the canonical record/document interface?
```

Options:

| Option | Description | Trade-off |
| --- | --- | --- |
| Promote `DetectionDocument` | Treat existing `src/detection/ir.py` as the semantic document spine | Reuses work, but may need expansion to cover all standard Stage-1/eval needs |
| Introduce `CoordExpRecord` below `DetectionDocument` | Use `CoordExpRecord` for raw prepared data and `DetectionDocument` for rendered semantic detection | Clean layering, but adds a new concept |
| Keep `NormalizedDetectionSample` as the central IR | Make current latest-detection sample broader and production-wide | Simpler, but the name may be too detection-template-specific |

Recommendation:

Choose a two-layer model only if both layers earn their keep:

```text
CoordExpRecord:
  prepared data contract, image path, dimensions, objects, geometry, metadata

DetectionDocument:
  semantic detection document for template/render/object identity/span use
```

If that distinction does not produce leverage, promote one interface and delete
the other. The deletion test should decide.

## 9. Summary verdict

The proposal is a strong migration hypothesis, but it should be edited mentally
as follows:

```text
not "CoordExp has the target architecture"
but "CoordExp is halfway through converging toward it"
```

Most of the recommended endpoints are reasonable. The high-value corrections are:

- make training surfaces executable before moving more logic into them;
- do not claim a canonical data IR until one is selected and routed through
  active paths;
- keep strict parser ownership separate from inference salvage;
- treat `stage2_rollout_runtime` and `rollout_matching.*` as public-retired but
  internal-active where appropriate;
- make lifecycle state machine-checkable;
- deepen existing seams before broad package renames.

The repository should become smaller, but the safest path is not deletion first.
The safest path is:

```text
classify -> deepen -> route one active path -> prove parity -> deprecate -> delete
```

That sequence preserves research meaning while still reducing code volume,
indirection, config complexity, duplicated execution paths, and compatibility
burden.
