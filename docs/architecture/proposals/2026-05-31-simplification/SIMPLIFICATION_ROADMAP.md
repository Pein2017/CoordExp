---
doc_id: docs.architecture.simplification-roadmap
layer: docs
doc_type: architecture-roadmap
status: proposal
domain: architecture
summary: Concrete refactoring roadmap, validation plan, and migration risks for CoordExp architecture simplification.
updated: 2026-05-31
---

# CoordExp Architecture Simplification Roadmap

## Purpose

This roadmap translates the architecture simplification proposal into implementation phases. It is designed for incremental PRs that preserve behavior while moving ownership into clearer modules.

Do not implement this roadmap as one giant refactor. The safe path is a sequence of small migrations with compatibility tests and artifact parity checks.

## Migration strategy

The migration should follow this pattern:

```text
1. Declare the target contract.
2. Add tests that describe current behavior and target ownership.
3. Move logic behind an equivalent facade.
4. Route one active path through the new owner.
5. Prove artifact / metric / config parity.
6. Deprecate compatibility path with owner and expiry.
7. Delete compatibility only after all active configs stop using it.
```

## Phase 0: Architecture registry and guardrails

### Goal

Create a machine-readable lifecycle registry so active, compatibility, and retired concepts are explicit.

### Proposed files

```text
architecture_state.yaml
# or docs/architecture/architecture_state.yaml if source-root policy prefers docs-local metadata
```

### Contents

```yaml
training_surfaces:
  active:
    - stage1_json_ce
    - stage1_compact_trie_ce
    - stage2_rollout_correction
  retired:
    - stage2_ab_training
    - stage2_two_channel
    - rollout_matching_sft
    - stage2_rollout_aligned
    - stage2_rollout_runtime

entrypoints:
  active:
    - python -m src.sft --config
    - scripts/run_infer.py --config
  compatibility:
    - scripts/run_infer.py legacy flags
    - scripts/run_infer_eval.sh
    - scripts/run_vis.sh

artifacts:
  active:
    - gt_vs_pred.jsonl
    - gt_vs_pred_scored.jsonl
    - metrics.json
  compatibility:
    - pred.jsonl
```

### Validation

Add tests that:

- active configs do not reference retired trainer variants;
- retired variant names raise canonical errors;
- compatibility entries include owner, reason, and expiry;
- new active entrypoints must be listed before use in tests or docs.

### Rationale

Without a registry, retired experiments can sneak back into active configs through helper scripts or comparator configs. A registry turns architecture intent into a CI-visible contract.

## Phase 1: Make training surface resolution mandatory

### Goal

Every training run should resolve to a canonical surface before dataset/trainer setup.

### Work items

1. Add a single `resolve_run_plan(config)` path.
2. Ensure current legacy training configs resolve to one of:
   - `stage1_json_ce`
   - `stage1_compact_trie_ce`
   - `stage2_rollout_correction`
3. Keep old config fields supported through projection.
4. Record the resolved surface in `effective_runtime.json`, `pipeline_manifest.json`, or successor artifact.

### Validation

- Golden tests for representative Stage-1 and Stage-2 configs.
- `--cfg-only` tests that print resolved surface identity.
- Failure tests for retired variants.

### Rationale

A training run should not begin by asking many local conditionals what it is. It should know its surface once, early, and pass that identity downstream.

## Phase 2: Slim `src.sft` into an entrypoint

### Goal

Extract policy from `src.sft` into owned modules without changing behavior.

### Target extraction

```text
src/training/entrypoint.py
  parse args, logging bootstrap, call runner

src/training/runner.py
  high-level run orchestration

src/training/runtime/packing.py
  packing config, static packing cache, accumulation windows

src/training/runtime/cache.py
  encoded sample cache config, fingerprints, eligibility

src/training/runtime/checkpointing.py
  checkpoint mode, resume, final checkpoint behavior

src/training/adapters/token_embeddings_adapter.py
  token-embeddings adapter setup

src/training/artifacts.py
  effective runtime payload, manifests, provenance
```

### Migration steps

1. Move pure helper functions first.
2. Keep import-compatible wrappers in `src.sft` temporarily.
3. Add tests that import the new owners directly.
4. Replace internal calls in `src.sft` with calls to new owners.
5. Reduce `src.sft` to argument parsing and delegation.

### Validation

- Existing Stage-1 smoke tests.
- Existing Stage-2 smoke tests.
- Static packing tests.
- Encoded cache tests.
- Checkpoint mode tests.
- Run metadata / manifest tests.

### Rationale

This phase reduces the main architecture hotspot. It also makes later surface-owned pipeline work much safer because each policy already has a module owner.

## Phase 3: Make `TrainingPipeline` executable

### Goal

Turn pipeline descriptors into executable owners.

### Proposed protocol

```python
class TrainingPipeline(Protocol):
    @property
    def identity(self) -> TrainingPipelineIdentity: ...

    def validate(self, plan: RunPlan) -> None: ...
    def build_dataset(self, plan: RunPlan, context: BuildContext) -> DatasetBundle: ...
    def build_collator(self, plan: RunPlan, dataset: DatasetBundle) -> Any: ...
    def build_trainer(self, plan: RunPlan, context: BuildContext, dataset: DatasetBundle) -> Any: ...
    def run(self, plan: RunPlan, context: BuildContext) -> TrainResult: ...
```

### Surface responsibilities

#### `stage1_json_ce`

- Standard JSON/CoordJSON SFT.
- Default dataset-level static packing allowed.
- Default ms-swift trainer path.
- Token CE objective.

#### `stage1_compact_trie_ce`

- Compact-full template.
- Trie CE / coord objective setup.
- Compact token role validation.
- Prefix-rollin only if still active under this surface.

#### `stage2_rollout_correction`

- Raw sample metadata preservation.
- Identity collator.
- Rollout decode request construction.
- Post-rollout packing owned by trainer/pipeline.
- Teacher-forcing correction objective.

### Validation

- One golden-thread test per surface.
- Artifact parity between old runner and new pipeline route.
- Same resolved runtime payload before/after migration.

### Rationale

A pipeline abstraction is only useful if it owns execution. Identity-only descriptors document intent but do not simplify the code path.

## Phase 4: Introduce canonical data IR

### Goal

Move active data loading toward a single `CoordExpRecord` or equivalent IR.

### Proposed types

```python
@dataclass(frozen=True)
class ImageRef:
    path: str
    width: int
    height: int
    image_id: str | int | None = None

@dataclass(frozen=True)
class DetectionObject:
    desc: str
    geometry_key: Literal["bbox_2d", "poly"]
    geometry_value: tuple[int, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class CoordExpRecord:
    image: ImageRef
    objects: tuple[DetectionObject, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
```

### Work items

1. Define IR and strict constructors from current JSONL records.
2. Make image-root resolution part of `DataView` creation.
3. Move bbox format validation into IR construction.
4. Make `DetectionTemplate` consume IR or a normalized sample derived from IR.
5. Keep current dataset classes as wrappers around IR until all callers migrate.

### Validation

- JSONL loading diagnostics parity.
- Image path resolution tests.
- Geometry invariant tests.
- Object ordering tests.
- Chat-template regression tests.
- Offline-prepared bbox branch tests.

### Rationale

The same record shape should feed training, inference, visualization, and evaluation. Without a canonical IR, each subsystem will keep its own small parser and its own edge cases.

## Phase 5: Unify offline inference and Stage-2 rollout decode

### Goal

Make offline inference and training-time rollout use the same decode request/result API.

### Proposed API

```python
@dataclass(frozen=True)
class BackendSpec:
    backend: Literal["hf", "vllm"]
    mode: str
    model: str | None
    adapter: str | None
    options: Mapping[str, Any]

@dataclass(frozen=True)
class DecodeBatchRequest:
    prompts: tuple[PromptBundle, ...]
    decode: DetectionDecodeRequest
    backend: BackendSpec
    provenance: Mapping[str, Any]

class DecodeRuntime:
    def generate(self, request: DecodeBatchRequest) -> tuple[DetectionDecodeResult, ...]: ...
```

### Work items

1. Add `BackendSpec` and `DecodeBatchRequest`.
2. Adapt offline inference to use the batch API.
3. Adapt Stage-2 rollout to build the same request type from explicit facts.
4. Keep owner-shaped adapters as compatibility wrappers.
5. Move local vLLM auto-launch out of `scripts/run_infer.py` into `src/infer/launch.py`.

### Validation

- Prompt token parity tests.
- HF/vLLM trace normalization tests.
- Offline inference artifact parity tests.
- Stage-2 rollout runtime tests.
- vLLM server-mode smoke tests if available.

### Rationale

Decode policy is a comparability contract. If offline inference and Stage-2 rollout diverge here, training diagnostics and benchmark artifacts become difficult to compare.

## Phase 6: MetricEvent-first metrics

### Goal

Route all metric producers through typed metric events and one reducer.

### Work items

1. Add a reducer that consumes `MetricEvent[]` and emits flat metrics.
2. Convert Stage-1 training metrics to events.
3. Convert Stage-2 pending logs and rollout telemetry to events.
4. Convert offline COCO/LVIS/F1-ish results to events.
5. Move compatibility flat names into alias tables.

### Validation

- Alias collision tests.
- Existing metric key parity tests.
- Offline eval output parity tests.
- Stage-2 DDP metric reduction tests.
- Metrics JSON schema tests.

### Rationale

Flat metric names are presentation. Metric identity should be semantic. This avoids accidental collisions and makes metric evolution safer.

## Phase 7: Public config domain schema

### Goal

Make the cleaner domain schema the public target while preserving old configs through projection.

### Target domains

```yaml
run:
surface:
data:
template:
supervision:
objectives:
observability:
artifacts:
runtime:
experimental:
```

### Work items

1. Add domain-schema examples for one Stage-1 and one Stage-2 config.
2. Implement `RunPlan` construction from the domain schema.
3. Implement projection from domain schema to current ms-swift arguments.
4. Mark `custom.*` as compatibility in docs and registry.
5. Migrate active smoke configs first.

### Validation

- Config strict unknown key tests.
- Inheritance/list ownership tests.
- `--cfg-only` parity tests.
- Effective runtime artifact parity tests.

### Rationale

Configuration should describe CoordExp concepts, not leak historical implementation details. Projection to ms-swift should be a boundary, not the public architecture.

## Phase 8: Compatibility cleanup

### Goal

Delete or hard-gate paths that are no longer needed by active configs and tests.

### Candidate cleanup list

```text
- pred.jsonl fallback alias
- legacy run_infer flag-only mode
- old stage2 variant names outside rejection tests
- dense-caption public naming in active docs/configs
- custom.fusion_config references from supported training surface
- compact helper formats if no active callers remain
- runtime bbox conversion branches
- runtime image resize escape hatches
```

### Validation

- Repository-wide grep tests for retired names in active configs.
- Official config smoke tests.
- Artifact reader compatibility tests for explicitly legacy fixtures.
- Docs link checks if available.

### Rationale

Compatibility paths are useful only when temporary. If kept indefinitely, they become a parallel architecture.

## Risk areas and migration concerns

### 1. ms-swift integration risk

`TrainArguments`, `RLHFArguments`, trainer factories, and template behavior are external integration points. Moving projection logic can accidentally change default values or nested argument mutation.

**Mitigation:** snapshot resolved `TrainArguments`-relevant fields before and after each extraction.

### 2. Stage-2 DDP and post-rollout packing risk

Stage-2 has special behavior around identity collators, raw metadata preservation, local normalization, pending-log aggregation, and variable post-rollout pack counts.

**Mitigation:** keep Stage-2 behavior behind parity tests before moving trainer internals. Avoid changing loss normalization and DDP reduction semantics during ownership refactors.

### 3. Prompt and token parity risk

Prompt templates, chat templates, coord tokens, stop tokens, and compact grammar must remain aligned between training, inference, and rollout.

**Mitigation:** require prompt-token and visual parity tests for decode runtime changes.

### 4. Coordinate token ID and adapter risk

Coord-offset adapters, trainable token rows, tokenizer expansion, and LoRA merge/export behavior depend on stable token IDs.

**Mitigation:** keep coord-token row setup isolated and add explicit token-ID contract tests before moving export or adapter logic.

### 5. Dataset provenance risk

Offline-prepared data depends on image-root semantics, metadata files, width/height, coordinate surface, and bbox parameterization. Small changes can invalidate experiments.

**Mitigation:** validate data IR construction against known prepared JSONL fixtures and view metadata.

### 6. Evaluation comparability risk

Changing artifact names, score provenance, confidence post-op, or duplicate guard behavior can make old benchmark claims incomparable.

**Mitigation:** keep artifact schema stable. Any new artifact reader should support legacy fixtures only through explicit compatibility mode.

### 7. vLLM backend compatibility risk

vLLM server-mode behavior depends on installed vLLM capabilities, trace details, adapter sync, and server responses.

**Mitigation:** keep backend capability validation strict and separate operational server launch logic from decode policy.

### 8. Large-file refactor risk

Moving too much at once will blur behavior changes and ownership changes.

**Mitigation:** prefer extraction-only PRs followed by routing PRs followed by deletion PRs.

## Suggested PR sequence

### PR 1: Add proposal docs and lifecycle registry

- Add architecture documents.
- Add `architecture_state.yaml` if accepted.
- Add no source behavior changes.

### PR 2: Surface-resolution smoke tests

- Require all representative training configs to resolve to a surface.
- Add retired-variant rejection tests.

### PR 3: Extract `src.sft` packing/cache helpers

- Move pure helpers.
- Keep wrappers.
- Prove tests pass.

### PR 4: Extract token-embeddings adapter setup

- Move adapter logic to `src/training/adapters/`.
- Keep behavior identical.

### PR 5: Make Stage-1 pipeline executable

- Route one Stage-1 smoke config through pipeline object.
- Add parity tests.

### PR 6: Make Stage-2 pipeline executable

- Route Stage-2 setup through pipeline object.
- Keep trainer implementation unchanged initially.

### PR 7: Add data IR behind existing dataset wrappers

- Build IR but keep old dataset APIs.
- Validate parity.

### PR 8: Add DecodeRuntime batch API

- Offline inference first.
- Stage-2 rollout second.

### PR 9: MetricEvent reducer adoption

- Add reducer.
- Convert one metric family at a time.

### PR 10: Compatibility gates and cleanup

- Explicitly gate legacy CLI/artifacts.
- Delete retired paths only after active configs are clean.

## Done criteria

The architecture simplification program is complete when:

1. All active training configs resolve to one of the three canonical surfaces.
2. `src.sft` is a thin entrypoint with no surface-specific branching.
3. Active data loading uses a canonical record IR.
4. Offline inference and Stage-2 rollout share decode request/result contracts.
5. Metrics are produced as `MetricEvent` and flattened by one reducer.
6. Compatibility paths have owner, reason, and expiry.
7. Retired variants cannot be used by active configs.
8. Official artifacts use canonical names and provenance.
9. Docs, configs, and tests agree on active architecture.
