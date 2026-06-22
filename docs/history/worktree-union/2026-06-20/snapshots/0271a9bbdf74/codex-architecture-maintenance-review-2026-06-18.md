# Codex Architecture and Maintenance Review: CoordExp

Date: 2026-06-18

Repository: `/data/CoordExp`

Scope: static inspection of git-tracked repository contents only. I did not run training, inference, evaluation, or other expensive jobs. I treated docs, configs, tests, and source code as evidence, with current implementation taking precedence where docs and code appeared to differ.

## Executive Thesis

CoordExp is not best understood as a conventional ML package with a few scripts around it. It is a research runtime with a serious evidence and reproducibility control plane. The strongest parts of the repository are the places where it admits that fact: strict YAML configs, explicit artifact contracts, no-resize geometry invariants, eval provenance, historical progress notes kept separate from current truth, and tests that lock down behavior that would otherwise drift silently.

My architectural read is that CoordExp should evolve by deepening a small number of central seams rather than by broad renaming or package reshuffling. The important seams are:

- offline JSONL and image/geometry alignment
- semantic detection scene representation
- template/render/decode contracts
- resolved training runtime plans
- Stage-1 and Stage-2 training surfaces
- shared rollout/inference decode request/result contracts
- `gt_vs_pred` style artifacts and scored evaluation
- metric events and manifests

The main danger is not lack of structure. The repo has a lot of structure. The danger is that several structures coexist at different maturity levels: mature docs, strict schemas, shadow pipeline descriptors, legacy compatibility paths, owner-shaped trainer internals, multiple detection data representations, and workbench-style analysis modules. Future research will be safest if the codebase makes its active path narrower and more executable while keeping compatibility adapters explicit and temporary.

My strongest recommendation is: do not start with a giant rewrite. Start by making the current active path mechanically legible. Promote the already-present concepts of `DetectionScene`, `ResolvedTrainingRun`, runtime plans, `DetectionDecodeRequest`, scored eval artifacts, and `MetricEvent` into active joining points. Then thin the large orchestration files one policy cluster at a time.

## Evidence Base

I inspected the repository through the project routing docs first, then verified key claims against source, configs, tests, and tracked file structure.

Representative commands and inspections:

- `git status --short --branch`: clean tracked tree at inspection start, on `main...origin/main [ahead 1]`.
- `git ls-files` grouped by top-level directory, extension, and selected subtrees.
- CodeGraph index status: 1,181 indexed files, 20,333 nodes, 50,268 edges.
- Authoritative docs inspected: `docs/AGENT_INDEX.md`, `docs/catalog.yaml`, `docs/PROJECT_CONTEXT.md`, `docs/SYSTEM_OVERVIEW.md`, `docs/IMPLEMENTATION_MAP.md`.
- Domain docs inspected: `docs/data/CONTRACT.md`, `docs/data/PREPARATION.md`, `docs/data/PACKING.md`, `docs/training/STAGE1_OBJECTIVE.md`, `docs/training/STAGE2_RUNBOOK.md`, `docs/training/METRICS.md`, `docs/eval/CONTRACT.md`, `docs/eval/WORKFLOW.md`, `docs/ARTIFACTS.md`.
- Architecture docs inspected: `docs/architecture/INDEPENDENT_ARCHITECTURE_REVIEW.md`, `docs/architecture/proposals/2026-05-31-simplification/CODEBASE_SIMPLIFICATION_PROPOSAL.md`, `docs/architecture/OWNERSHIP_BOUNDARIES.md`, `docs/architecture/SIMPLIFICATION_ROADMAP.md`.
- Stable specs inspected: `openspec/specs/runtime-architecture-refactor-program/spec.md`, `openspec/specs/stage2-rollout-correction/spec.md`.
- Source areas inspected: `src/sft.py`, `src/config/schema.py`, `src/training/surfaces.py`, `src/training_runtime/plan.py`, `src/training/pipelines/*`, `src/detection/*`, `src/common/detection_*`, `src/infer/*`, `src/trainers/stage2_*`, `src/eval/*`, `src/metrics/events.py`.
- Configs inspected: `configs/stage1/detection_teacher_forcing/prod/compact_support2.yaml`, `configs/stage2/rollout_correction/base.yaml`, `configs/infer/pipeline.yaml`, `configs/eval/detection.yaml`.
- Tests inspected as contract evidence: `tests/test_stage2_rollout_correction_contract.py`, `tests/test_training_config_strict_unknown_keys.py`, plus route and file-map evidence for broader test coverage.

Tracked top-level file counts showed this rough shape:

- `openspec`: 621 tracked files
- `src`: 396
- `tests`: 347
- `configs`: 231
- `progress`: 228
- `docs`: 143
- `scripts`: 97
- `public_data`: 83

That distribution matters. This is a repo where documentation, specs, configs, tests, and progress records are not accessory material. They are part of the architecture.

## Mental Model

I would divide CoordExp into eight conceptual layers.

### 1. Authority and Evidence Plane

This layer is `docs/`, `openspec/`, `progress/`, and `docs/catalog.yaml`.

The repo is unusually explicit about evidence hierarchy. `docs/AGENT_INDEX.md` points agents toward `docs/PROJECT_CONTEXT.md`, `docs/SYSTEM_OVERVIEW.md`, `docs/IMPLEMENTATION_MAP.md`, relevant domain docs, stable `openspec/specs/`, and only then historical `progress/`. `docs/PROJECT_CONTEXT.md` says current docs are current truth, `openspec/specs/` are stable compatibility contracts, active changes are only relevant when explicitly named, and `progress/` is historical evidence. That is the right separation for a research codebase.

This layer should be protected because research repositories often collapse under ambiguous truth sources. CoordExp has mostly avoided that.

### 2. Config and Runtime-Plan Plane

This layer is `configs/`, `src/config/schema.py`, `src/training/surfaces.py`, `src/training_runtime/plan.py`, and config contract tests.

The project is config-first by design. `docs/standards/CODE_STYLE.md` and `docs/standards/REPO_HYGIENE.md` both reinforce strict validation, YAML-first behavior changes, and fail-fast unknown keys. `tests/test_training_config_strict_unknown_keys.py` locks down removed keys and variants. `src/training_runtime/plan.py` provides active runtime-plan decisions such as Stage-2 post-rollout packing ownership and fail-fast retired variants.

This layer is partly mature and partly transitional. `src/training/surfaces.py` has a promising `TrainingSurfaceResolver`, but its pipeline descriptors are shadow identity descriptors rather than executable training pipelines. The repo has the vocabulary of surfaces; the implementation still routes much of the real execution through broad orchestration code.

### 3. Data, Image, and Geometry Plane

This layer is `public_data/`, `docs/data/*`, `src/datasets/*`, `src/detection/data.py`, `src/detection/scene.py`, `src/datasets/geometry.py`, and offline preparation scripts.

The core invariant is clear: images are resized offline, geometry is aligned to the offline image view, and training uses `do_resize=false`. `docs/SYSTEM_OVERVIEW.md` states this directly. `docs/data/CONTRACT.md` defines strict JSONL rows, image references, objects, geometry keys, coordinate spaces, and raw JSONL versus model CoordJSON. `docs/data/PREPARATION.md` repeats the golden rule that training and eval use offline images and the model processor must not silently resize.

This is one of the most important layers in the repo. If future changes damage it, everything downstream can look plausible while being wrong.

### 4. Detection Semantic, Template, and Token Plane

This layer is `src/detection/scene.py`, `src/detection/data.py`, `src/detection/ir.py`, `src/detection/template.py`, `src/detection/template_contracts.py`, `src/common/detection_sequence.py`, `src/common/detection_compact_rows.py`, and related tests.

The right central concept is already present: `DetectionScene`. `src/detection/scene.py` describes it as the semantic in-memory authority for detection examples, with raw JSONL storage, render/token/eval/trainer views as projections. That is exactly the kind of object this codebase needs.

The template layer is also thoughtfully separated. `src/detection/template_contracts.py` owns compact-template metadata shared across training, inference, token-row artifacts, and provenance. `src/detection/template.py` owns strict rendering and parsing for training-facing templates. `src/common/detection_sequence.py` intentionally keeps looser compatibility parsing for inference and non-training callers, including salvage behavior and helper formats.

The problem is not absence of good concepts. The problem is overlapping maturity: `DetectionScene`, `NormalizedDetectionSample`, raw detection rows, `DetectionDocument` in `src/detection/ir.py`, rendered strings, token rows, eval records, and legacy caption records all coexist. Some coexistence is necessary during migration, but the repo needs a clearer declaration of which representation is canonical at each boundary.

### 5. Training Plane

This layer is `src/sft.py`, `src/training/*`, `src/training_runtime/*`, `src/trainers/*`, `configs/stage1/*`, `configs/stage2/*`, and training docs.

The public story is good:

- Stage-1 detection teacher forcing is the baseline surface.
- Stage-2 rollout-aware correction is the active rollout surface.
- Removed Stage-2 variants fail fast.
- Runtime fusion is legacy or experimental.
- Packing, cache, template, image-root, and artifact decisions are documented and tested.

The implementation shape is less clean. `src/sft.py` is about 4,340 lines and handles config loading, legacy custom shims, image-root resolution, ms-swift pipeline initialization, coord-template adapter setup, token embeddings, encoded cache, bbox-format contracts, dataset construction, static packing, trainer class composition, callbacks, Stage-2 runtime projection, selected pipeline manifests, effective runtime payloads, run metadata, and experiment manifests. That is too much load in one entrypoint.

`src/trainers/stage2_rollout_runtime.py` is about 4,585 lines, and `src/trainers/stage2_rollout_correction_impl.py` is about 5,923 lines. Those numbers do not prove bad architecture by themselves, but the surrounding evidence does: owner-shaped runtime state still leaks into inference and rollout dispatch, while the active contracts want explicit decode/runtime facts and shared request/result structures.

### 6. Inference and Rollout Decode Plane

This layer is `src/infer/runtime.py`, `src/infer/backend.py`, `src/infer/pipeline.py`, `src/infer/rollout_dispatch.py`, inference configs, and Stage-2 rollout integration.

This area is moving in the right direction. `src/infer/runtime.py` defines decode facts, runtime facts, `InferenceRuntime`, and builders from infer configs and rollout owner-like objects. `src/infer/backend.py` defines `DetectionDecodeResult` and backend handle projection. `src/infer/rollout_dispatch.py` has owner-to-handle translation plus a more owner-free dispatch core. `docs/SYSTEM_OVERVIEW.md` and `docs/training/STAGE2_RUNBOOK.md` both describe shared prompt/decode/backend trace through `src/infer/*`.

The design target is clear: Stage-2 trainers should not privately own decoding semantics. They should construct explicit facts or requests and call a shared runtime. The current code is partway there, but not done.

### 7. Evaluation, Artifact, and Metric Plane

This layer is `docs/eval/*`, `docs/ARTIFACTS.md`, `src/eval/*`, `src/metrics/*`, inference/eval configs, and artifact tests.

This is one of the best-designed parts of the repo. The distinction between raw `gt_vs_pred` artifacts, scored artifacts, guarded artifacts, COCO/LVIS claims, f1-ish debug metrics, score provenance, and visualization expectations is explicit. `docs/eval/CONTRACT.md` says COCO evaluation consumes scored artifacts and raw eval is debug/f1-ish only. `docs/eval/WORKFLOW.md` repeats that official COCO/LVIS/both claims must consume scored artifacts. `docs/ARTIFACTS.md` documents stable artifact names and training manifests.

`src/metrics/events.py` is also a good direction: typed `MetricEvent`, aliases, reducers, flattening, and collision guards are the right response to metric-name drift. This layer should be deepened, not casually rewritten.

### 8. Analysis and Experiment Workbench Plane

This layer is `src/analysis/`, `configs/analysis/`, `scripts/`, `progress/`, and one-off artifacts under controlled locations.

The tracked counts are large: `src/analysis` has 89 tracked Python files, and `configs/analysis` has 93 tracked config files. That can be appropriate for a research codebase, but only if analysis code has a clear lifecycle. The docs already point in that direction: `progress/` is historical evidence, `docs/standards/REPO_HYGIENE.md` defines script and asset lifecycle, and current behavior belongs in docs/specs/configs/tests/source rather than progress.

My concern is that workbench code can become de facto production because it exists and works. The architecture should make that promotion explicit.

## Strengths to Preserve

### The repo has a real authority hierarchy

The docs routing is not decorative. It gives agents and humans a way to decide what is current, what is stable, what is experimental, and what is historical. `docs/catalog.yaml` makes this even stronger by listing active surfaces, config roots, docs, progress notes, and code handles.

This should be preserved. In a research codebase with many experiments, the ability to say "this progress note is evidence, not current contract" is a structural advantage.

### Geometry and image alignment are treated as first-class contracts

The no-resize and geometry alignment rules appear in multiple places and are backed by code routes. `docs/SYSTEM_OVERVIEW.md` frames offline prep as the place where image resizing and coordinate conversion happen. `docs/data/CONTRACT.md` distinguishes raw JSONL rows, geometry keys, coord spaces, and model CoordJSON. `docs/data/PREPARATION.md` says offline-prepared images are the training/eval source and model processors must not resize. `configs/stage1/detection_teacher_forcing/prod/compact_support2.yaml` explicitly sets detection fields such as `bbox_format: xyxy`, `strict_parse: true`, and no packing.

This is exactly the right paranoia. Silent geometry bugs are worse than loud crashes.

### The public Stage-2 surface has been narrowed

The Stage-2 docs and specs have converged on one active trainer surface: `stage2_rollout_correction`, with residual-set correction as the single active objective. `openspec/specs/stage2-rollout-correction/spec.md` says old split public variants are removed, `rollout_matching.pipeline.*` is gone, and remaining `rollout_matching.*` is a private migration/runtime handle. `tests/test_stage2_rollout_correction_contract.py` checks that old variants and old keys fail fast.

That is a good cleanup pattern: remove public ambiguity first, keep temporary runtime migration handles explicit, and test the boundary.

### Artifact semantics are strong

The artifact contract is unusually mature. `docs/ARTIFACTS.md` preserves stable artifact names as ownership moves. `docs/eval/CONTRACT.md` and `docs/eval/WORKFLOW.md` distinguish raw, scored, guarded, debug, and official metric inputs. This gives future research an audit trail.

Do not casually rename or flatten these artifacts. The names are part of the reproducibility API.

### Strict config validation protects research meaning

`src/config/schema.py` is large, but it encodes a lot of important governance: strict keys, removed mechanism rejection, defaults, migration shims, and domain-specific constraints. Tests such as `tests/test_training_config_strict_unknown_keys.py` and `tests/test_stage2_rollout_correction_contract.py` make config drift visible.

Strict config parsing is especially important because this repo uses YAML as a public research interface. A silently ignored key could invalidate a run.

### The codebase has good embryonic seams

Several abstractions are the right shape even if they are not fully active:

- `DetectionScene` in `src/detection/scene.py`
- `TrainingSurfaceResolver` and `ResolvedTrainingRun` in `src/training/surfaces.py`
- `resolve_training_runtime_plan` in `src/training_runtime/plan.py`
- `DetectionDecodeRequest` and `DetectionDecodeResult` style boundaries in `src/infer/*`
- `MetricEvent` in `src/metrics/events.py`
- template contracts in `src/detection/template_contracts.py`

The next development phase should turn these into executable joints, not invent unrelated replacements.

### Tests are contract-oriented

The test tree is large and appears to focus heavily on contracts: config strictness, Stage-2 variant removal, packing/cache behavior, manifests, inference artifacts, eval parsing, metrics, and detection templates. That is the right kind of testing for this repo. Pure unit tests would not be enough here; the important failures are often cross-boundary contract failures.

## Problems and Architectural Risks

### `src/sft.py` is still the main overload point

`src/sft.py` is about 4,340 lines and owns too many kinds of decision:

- config loading and detection-vs-legacy branching
- legacy `custom.*` shims
- image-root inference
- coord-template adapter setup
- token embedding adapter installation
- encoded sample cache decisions
- bbox-format contract checks
- dataset construction for detection and legacy paths
- static packing and Stage-2 post-rollout packing decisions
- trainer class composition and callback setup
- Stage-2 runtime projection into trainers
- pipeline manifests, effective runtime payloads, run metadata, experiment manifests

This file is acting as an executable compatibility matrix. That is understandable historically, but it is the wrong long-term center. It makes behavior hard to audit because the "why" of a run is scattered through imperative orchestration rather than materialized as a resolved plan.

I would not split it by file length alone. I would split it by policy cluster, preserving the current entrypoint as a wrapper until tests prove equivalence.

### Training pipeline descriptors are too shallow

`src/training/pipelines/base.py` defines `PipelineLifecycle` and a `TrainingPipeline` protocol, but the current pipeline descriptors such as `src/training/pipelines/stage1_json_ce.py` and `src/training/pipelines/stage2_rollout_correction.py` are identity descriptors with `SHADOW` lifecycle. `src/training/surfaces.py` validates domains and objectives and returns `ResolvedTrainingRun`, but the active execution still largely happens elsewhere.

This creates an architectural split: docs and configs speak in surfaces, while execution still speaks in entrypoint branches and trainer setup. The fix is not to invent a huge abstract pipeline framework. The fix is to let the resolved run object become the thing that active code consumes.

### Stage-2 internals remain owner-shaped

The Stage-2 surface is conceptually cleaner than its implementation. `src/trainers/stage2_rollout_runtime.py` and `src/trainers/stage2_rollout_correction_impl.py` are both very large. `src/infer/runtime.py` has builders that still resolve decode requests from owner-like objects. `src/infer/rollout_dispatch.py` translates Stage-2 owner state into backend handles before reaching owner-free dispatch.

That is not a moral failure. It is exactly what a staged migration looks like. But it should be treated as unfinished architecture work. The stable target should be:

- Stage-2 target construction remains in trainer/training code.
- Shared prompt/decode/backend behavior lives in `src/infer`.
- Trainer state is projected into explicit facts, requests, and handles.
- Backend dispatch does not need to understand trainer owners.

### Detection representation is overpopulated

The repo currently has several representations in circulation:

- raw JSONL row contracts in docs and `src/detection/data.py`
- `RawDetectionRow` and related typed rows
- `NormalizedDetectionSample`
- `DetectionScene`
- `DetectionDocument` and slot-style `DetectionGeometry` in `src/detection/ir.py`
- rendered assistant text
- token rows and span metadata
- eval records and decoded detection results
- legacy caption/conversation records

Some of these are boundary views and should exist. But `src/detection/scene.py` and `src/detection/ir.py` appear to encode overlapping "semantic IR" ambitions. The newer `DetectionScene` is better aligned with the architecture docs because it explicitly includes image reference, coordinate frame, object ordering, and projection methods.

My recommendation is to make `DetectionScene` the canonical semantic in-memory representation for detection. Keep raw rows as storage, rendered text as template output, decode/eval records as downstream artifacts, and legacy normalized samples as compatibility bridges. Decide whether `DetectionDocument` has a distinct job. If it does not, retire or relabel it.

### Strict and salvage parsing need clearer policy names

`src/detection/template.py` owns strict training-facing parsing. `src/common/detection_sequence.py` intentionally has compatibility parsing that strips generation suffixes and returns `None` instead of strict template errors. This is a useful distinction, not a mistake.

The risk is that future developers may treat both as interchangeable "the parser." They are not. One is a validity contract. The other is a tolerant artifact/debug compatibility layer. That distinction should be present in module names, call-site names, and tests.

### Config layering is strict but cognitively expensive

The config system is disciplined, but there are too many partially overlapping domains:

- legacy `custom.*`
- top-level detection training config
- shadow surface domains
- `stage2_rollout_correction.*`
- still-active `rollout_matching.*` runtime/backend/eval handles
- semantic `packing` fields plus adapter fields under `training`

The Stage-2 base config illustrates the auditability problem. `configs/stage2/rollout_correction/base.yaml` has residual objective `expected_num_rollouts: 4`, while `triage_posterior.num_rollouts: 2`. The implementation in `src/trainers/stage2_rollout_correction_impl.py` chooses `expected_num_rollouts` from residual-set options when present, so this is probably not a functional bug. But it is an audit smell: a reader can see two rollout counts and must inspect implementation to know which one matters.

The architecture needs a resolved run plan that says, explicitly and in one place, "the effective rollout count is K, from source X, for reason Y."

### The analysis workbench is large enough to need lifecycle enforcement

`src/analysis` and `configs/analysis` are both large. That is normal for active research, but large workbench areas need guardrails:

- one-off exploration should remain easy
- repeated analysis should get a stable script/config owner
- evidence should land in `progress/`
- stable behavior should be promoted to docs/specs/configs/tests/source
- stale or superseded analysis should be marked historical rather than silently relied on

The existing repo hygiene docs already say much of this. The missing piece is enforcement by convention, tests, or a lightweight registry.

### `src/config/schema.py` is a useful monolith, but still a monolith

At about 4,831 lines, `src/config/schema.py` is large. I would be careful here: config schema code is one of the places where monolithic code can be safer than distributed cleverness, because strict parsing and migration errors need to be coherent. But as more runtime-plan logic moves out of `src/sft.py`, this file should eventually be decomposed by config domain or converted into smaller schema modules with one public loader facade.

Do not do this first. The schema file is currently protecting many contracts. Refactor it only after active run-plan objects and tests make behavior easy to preserve.

## Central Surfaces to Protect

These are the surfaces I would treat as high-risk and compatibility-sensitive:

- Offline JSONL data contract: `docs/data/CONTRACT.md`, `public_data`, `src/detection/data.py`.
- Image/geometry alignment and no-resize behavior: `docs/data/PREPARATION.md`, `docs/SYSTEM_OVERVIEW.md`, `src/datasets/geometry.py`, model processor settings.
- Stage-1 compact detection teacher forcing: `configs/stage1/detection_teacher_forcing/prod/compact_support2.yaml`, `src/detection/runtime.py`, `src/sft.py` compatibility path.
- Stage-2 rollout correction: `configs/stage2/rollout_correction/*`, `src/trainers/stage2_rollout_correction.py`, `src/trainers/stage2_rollout_correction_impl.py`, `src/trainers/stage2_rollout_runtime.py`.
- Shared infer/decode runtime: `src/infer/runtime.py`, `src/infer/backend.py`, `src/infer/rollout_dispatch.py`.
- Template contracts and parser policies: `src/detection/template.py`, `src/detection/template_contracts.py`, `src/common/detection_sequence.py`.
- Artifact names and schemas: `docs/ARTIFACTS.md`, `docs/eval/CONTRACT.md`, `src/eval/*`.
- Metric identity and aliases: `docs/training/METRICS.md`, `src/metrics/events.py`.
- Strict config loader behavior: `src/config/schema.py`, `tests/test_training_config_strict_unknown_keys.py`.

These should be changed only with targeted tests and explicit migration notes.

## Development Recommendations

### 1. Make the active lifecycle registry executable

The repo already has lifecycle language in architecture docs and specs: active, compatibility, retired, shadow, historical. I would turn that into something checkable.

Concrete target:

- A small tracked registry of active public surfaces, compatibility handles, retired variants, and historical-only configs.
- Tests that active configs do not reference retired surfaces.
- Tests that retired public keys still fail fast with clear errors.
- Docs generated or checked against the same registry where practical.

This is high-leverage because it attacks ambiguity without changing training behavior.

### 2. Promote a resolved run plan before splitting everything

Before a broad refactor, create or strengthen a resolved training run object that active execution consumes. It should answer:

- which surface is active
- which dataset/source contract is active
- which template and parser policy are active
- which objective is active
- which rollout count and decode settings are effective
- who owns packing
- which artifact and metric namespaces will be emitted
- what compatibility handles were used

The repository already has `ResolvedTrainingRun` and runtime plans. The move is to make them operational rather than descriptive.

This is the missing bridge between config strictness and code auditability.

### 3. Thin `src/sft.py` by policy cluster, not by aesthetic slicing

I would extract in this order:

1. Token embedding and coordinate-template adapter setup.
2. Packing and encoded-cache planning.
3. Dataset bundle construction and image-root resolution.
4. Run manifest, effective runtime, and provenance writing.
5. Trainer class/callback composition.

Each extraction should leave a small wrapper or compatibility function in `src/sft.py` first. Tests should prove that existing Stage-1 and Stage-2 config behavior did not change. This avoids a risky "new training entrypoint" rewrite.

### 4. Declare `DetectionScene` as the canonical detection semantic object

`DetectionScene` should become the center of detection data flow. Raw JSONL rows, normalized samples, template rendering, Stage-2 supervision views, inference decoded results, and eval records should be projections from or to this concept.

Concrete actions:

- Add or strengthen projection tests across raw row -> `DetectionScene` -> template render -> parse/decode -> eval record.
- Decide whether `src/detection/ir.py::DetectionDocument` is still needed.
- Rename or document migration bridges so they are not mistaken for the canonical path.
- Keep geometry frame and coordinate chart explicit. Do not infer coordinate semantics from field names.

### 5. Continue owner-to-facts migration in rollout decoding

`src/infer/runtime.py` and `src/infer/rollout_dispatch.py` already point toward explicit facts, handles, and requests. Continue that direction.

Concrete target:

- Stage-2 code builds a small immutable decode/runtime context.
- Shared infer code consumes that context.
- Backends do not need trainer owner objects.
- Trainer-private methods do not duplicate prompt, parser, generation, or backend behavior.

Be careful not to move training target construction, assignment, DDP behavior, or loss semantics into `src/infer`. `openspec/specs/runtime-architecture-refactor-program/spec.md` is right to keep shared runtime small.

### 6. Make parser policy explicit at every boundary

I would name parser choices by intent:

- `strict_training_parse`
- `tolerant_artifact_parse`
- `salvage_debug_parse`

The exact names can differ, but call sites should not just say "parse detection sequence" when the policy affects validity. This is especially important where metric claims are made.

### 7. Decompose metrics by adopting `MetricEvent` family by family

Do not rewrite all metrics at once. Continue moving high-value families to `MetricEvent` and alias registries:

- Stage-1 coord/token/template losses
- Stage-2 rollout/correction metrics
- eval artifact metrics
- launch-health diagnostics

Preserve tolerant legacy reads, but make clean current keys canonical.

### 8. Treat analysis code as a workbench with promotion rules

For `src/analysis`, `configs/analysis`, and `scripts`, I would define three states:

- exploratory: allowed to be rough, tied to a progress note or temporary artifact
- repeated: has stable CLI/config and minimal tests or smoke checks
- promoted: becomes part of training/infer/eval/source docs and contract tests

This lets research stay fast without letting accidental utilities become hidden infrastructure.

## Documentation Recommendations

The docs are already good, so I would not add more broad overview pages. I would add or update only high-clarity pages:

- A current "active surface registry" page or generated table.
- A "resolved run plan" explanation once the object is active.
- A detection representation map showing raw row, `DetectionScene`, normalized compatibility sample, template render, token rows, decode result, eval record.
- A parser policy page or table distinguishing strict, tolerant, and salvage behavior.
- A Stage-2 effective rollout count/config-resolution note, specifically explaining `expected_num_rollouts` versus `triage_posterior.num_rollouts`.

Avoid adding another long architecture manifesto unless it replaces or supersedes an older one. There are already several architecture documents. The next docs should reduce ambiguity, not add a parallel explanation.

## Test Recommendations

The highest-value additional tests would be cross-boundary contract tests rather than isolated pure unit tests:

- Active config registry test: no active config references retired public keys or historical-only surfaces.
- Resolved run plan snapshot tests for Stage-1 compact TF and Stage-2 rollout correction smoke configs.
- DetectionScene projection tests: raw JSONL -> scene -> template render -> strict parse -> decoded/eval record.
- Parser policy tests: strict training failures versus tolerant artifact parse versus salvage debug parse.
- Stage-2 rollout effective settings tests: prove which rollout count, backend, parser mode, assignment policy, and packing owner are effective for base/prod/smoke configs.
- Artifact metric provenance tests: official COCO/LVIS paths require scored artifacts and comparable score provenance.
- Manifest completeness tests: resolved config path, git SHA, data roots, template contract, coordinate/tokenization settings, and effective runtime payload.

I would avoid broad slow integration tests as the first response. Start with cheap contract tests that fail when research meaning changes.

## Research Direction Advice

### Where new experiments should live first

Use this rule:

- If a new idea is a knob over existing behavior, start in `configs/`.
- If it changes interpretation, evidence, or run outcome, record it in `progress/` with scope and artifact handles.
- If it changes stable behavior, config schema, artifact names, metrics, loss semantics, or eval validity, use `openspec/`.
- If it is a repeated analysis workflow, put it in `scripts/` or `src/analysis/` with a clear config and progress note.
- If it is core training/infer/eval behavior, add a module under `src/` behind an existing surface or a clearly registered new surface.
- If it is risky, compatibility-sensitive, or likely to churn, use a branch and keep the public surface closed until the behavior is proven.

### How to keep experimentation fast without corrupting the architecture

Keep the stable path narrow. Let experiments branch off through configs and explicit modules, not by adding hidden cases to the central entrypoint.

I would use this promotion ladder:

1. Prototype: config override, `src/analysis`, script, or branch.
2. Evidence: progress note with scope, config, artifact root, metrics, and failure modes.
3. Stabilization: tests around the behavior that actually matters.
4. Surface: config schema or registered surface if multiple runs depend on it.
5. Documentation: current docs only when it becomes recommended or stable.
6. Spec: OpenSpec only for compatibility-sensitive contracts.

This prevents every idea from becoming permanent infrastructure while preserving the evidence needed to resurrect good ideas later.

## Priority Plan

### Do first

1. Build an active/compat/retired lifecycle registry and test active configs against it.
2. Make resolved run plans first-class for Stage-1 compact TF and Stage-2 rollout correction smoke configs.
3. Add snapshot-style tests for effective Stage-2 rollout settings, especially rollout count, parser policy, packing owner, and artifact namespaces.
4. Write a detection representation map and decide whether `DetectionDocument` remains distinct from `DetectionScene`.
5. Extract one low-risk cluster from `src/sft.py`, preferably manifest/provenance writing or token adapter setup, while preserving wrapper behavior.

### Do next

1. Move more Stage-2 decode dispatch from owner-shaped state to explicit facts and handles.
2. Promote `MetricEvent` adoption for one metric family at a time.
3. Extract dataset bundle construction out of `src/sft.py` once `DetectionScene` projection tests are strong.
4. Tighten analysis workbench lifecycle with lightweight metadata or docs.
5. Decompose `src/config/schema.py` only after resolved plans reduce the need to inspect raw parser internals.

### Can wait

- Broad package renames.
- A new universal pipeline framework.
- Wholesale deletion of compatibility configs.
- Full metric namespace rewrite.
- Major eval artifact renaming.
- Aggressive cleanup of historical progress files.

### Risky or not worth it now

- Replacing `src/sft.py` with a new entrypoint in one step.
- Removing tolerant/salvage parsers because strict parsing feels cleaner.
- Deleting `rollout_matching.*` handles before replacement runtime config is proven.
- Changing geometry coordinate conventions as part of architecture cleanup.
- Introducing new CLI flags where config schema should carry the behavior.
- Treating `progress/` as current documentation.

## What I Would Leave Alone

I would leave these mostly alone except for narrow compatibility fixes:

- Stable artifact names such as `gt_vs_pred.jsonl`, `gt_vs_pred_scored.jsonl`, guarded artifacts, and metric outputs.
- The no-resize image preprocessing contract.
- The strict config validation posture.
- The separation between current docs, stable specs, active changes, and historical progress.
- Existing Stage-1 compact TF and Stage-2 rollout correction public config names.
- The eval rule that COCO/LVIS claims consume scored artifacts.
- Clean-write/tolerant-read artifact behavior.

These are the repo's load-bearing walls.

## Where Docs and Code Seem Slightly Out of Sync

I did not find a catastrophic contradiction in the inspected path, but I did find places where a reader has to understand implementation details to reconcile docs and configs.

The best example is Stage-2 rollout count. `docs/training/STAGE2_RUNBOOK.md` describes the default pseudo-positive profile around `triage_posterior.num_rollouts: 4` and says lower K is rejected in that default profile. `configs/stage2/rollout_correction/base.yaml` has `stage2_rollout_correction.residual_set.expected_num_rollouts: 4`, but `triage_posterior.num_rollouts: 2`. The implementation in `src/trainers/stage2_rollout_correction_impl.py` appears to use residual-set `expected_num_rollouts` when residual options exist, so the effective value is likely 4 for that path.

I would not call this a bug without running targeted config/runtime tests. I would call it a maintainability smell: the effective setting is not obvious from the YAML. A resolved run plan would fix this better than a prose note alone.

## My Opinionated Target Architecture

If I were evolving this repo, I would aim for this hierarchy:

1. `docs/` and `openspec/specs/` define current behavior and stable contracts.
2. `configs/` define public research runs.
3. `src/config` parses YAML strictly into config objects.
4. `src/training_runtime` resolves config objects into explicit run plans.
5. `src/detection/scene.py` owns the semantic detection example.
6. `src/detection/template.py` and `src/detection/template_contracts.py` own render/parse/template identity.
7. `src/sft.py` remains a thin compatibility entrypoint.
8. Surface-specific training modules construct datasets, collators, trainers, losses, and manifests from the resolved plan.
9. `src/infer` owns decode requests, backend dispatch, and decode results shared by inference and rollout training.
10. `src/eval` consumes artifacts and produces scored/guarded metrics under stable artifact contracts.
11. `src/metrics` owns typed metric identities, aliases, reducers, and logging adapters.
12. `src/analysis`, `scripts`, and `progress` remain fast-moving evidence and diagnostic layers with explicit promotion paths.

This target is close to what the repo already says it wants. The work is convergence, not reinvention.

## Self-Reflection

### Assumptions I made

I assumed tracked repository contents are the intended source of truth, as requested. I assumed current docs are meaningful unless code/config/tests contradict them. I assumed no expensive runtime jobs should be run. I treated existing architecture docs as evidence and hypotheses, not as final truth.

### What I inspected first and why

I started with `docs/AGENT_INDEX.md`, `docs/catalog.yaml`, `docs/PROJECT_CONTEXT.md`, `docs/SYSTEM_OVERVIEW.md`, and `docs/IMPLEMENTATION_MAP.md` because the repository explicitly tells agents to route through those files before broad source search. Then I checked file counts and code handles to avoid simply repeating docs. After that I focused on the central behavior surfaces: config schema, training entrypoint, detection representation, Stage-2 trainers, inference runtime, eval artifacts, and metric events.

### What another strong model might disagree with

Another model might argue for a faster clean-break rewrite around a new `src/pipelines` or `src/data` hierarchy. I think that would be too risky right now because the current repo has valuable compatibility contracts, artifact semantics, and historical reproducibility constraints. The safer path is to make active concepts executable, then retire compatibility paths with tests.

Another model might also favor leaving `src/sft.py` alone because it works and is heavily tested indirectly. I disagree. It should not be attacked casually, but it is too broad to remain the long-term place where research meaning is assembled.

### Problems I am most confident I detected

I am most confident about these:

- `src/sft.py` is overloaded as an orchestration and compatibility hub.
- Training pipeline descriptors are too shallow relative to the surface vocabulary.
- Stage-2 decode/runtime integration is still too owner-shaped.
- Detection has too many overlapping representation layers.
- Parser policy boundaries need clearer names and tests.
- Config strictness is strong, but effective runtime settings are not always obvious from YAML alone.
- Artifact/eval contracts are unusually strong and should be protected.

### Problems I might have missed

I may have missed runtime performance bottlenecks, distributed training edge cases, subtle eval bugs that require artifact replay, data-specific geometry failures, CI-only failures, and untracked experimental state. I also did not inspect every test file or every analysis module. My review is strongest on architecture, contracts, and maintainability, not on empirical model quality or performance.

## Bottom Line

CoordExp already has the bones of a serious research system: strict configs, strong artifact contracts, geometry paranoia, documented current behavior, historical evidence separation, and contract-heavy tests. The maintenance challenge is to keep the active path narrow enough that future experiments are auditable.

I would evolve the repo by:

- making lifecycle and resolved run plans executable
- promoting `DetectionScene` as the semantic detection center
- thinning `src/sft.py` gradually
- completing owner-to-facts migration in rollout decoding
- keeping eval/artifact/metric contracts stable
- letting experiments begin in configs/progress/analysis but requiring promotion before they become infrastructure

The codebase does not need a dramatic rewrite. It needs convergence around the good concepts it already contains.
