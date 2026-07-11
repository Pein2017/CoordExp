---
doc_id: docs.architecture.codebase-simplification-proposal
layer: docs
doc_type: architecture-proposal
status: proposal
domain: architecture
summary: Architecture diagnosis and target simplification model for CoordExp.
updated: 2026-05-31
---

# CoordExp Codebase Simplification Proposal

## Purpose

CoordExp has accumulated multiple research iterations, training surfaces, rollout variants, dataset paths, evaluation utilities, and compatibility abstractions. The repository now contains enough current contracts to converge around a smaller architecture, but several legacy paths still live inside active execution code.

This proposal consolidates the architectural direction for simplifying the repository. It focuses on reducing duplicate concepts, clarifying ownership boundaries, and making future experiments explicit rather than letting them become permanent infrastructure by accident.

The intended outcome is a codebase where each major concept has one canonical owner and one canonical execution path.

## Executive summary

CoordExp should converge around the following minimal architecture:

```text
Offline data preparation
  -> canonical JSONL / image-root / geometry contract
  -> canonical DataView / record IR
  -> canonical DetectionTemplate render/parse/token-span layer
  -> one of three training surfaces
  -> shared DecodeRuntime for inference and Stage-2 rollouts
  -> canonical gt_vs_pred artifacts
  -> shared evaluation and MetricEvent reduction
```

The target set of core concepts should be:

```text
1. RunPlan
   Immutable resolved config. No free-form YAML dictionaries past this boundary.

2. Surface
   One of:
   - stage1_json_ce
   - stage1_compact_trie_ce
   - stage2_rollout_correction

3. DataView / CoordExpRecord
   Validated offline JSONL plus image-root plus geometry contract.
   Runtime does not resize images or convert bbox formats.

4. DetectionTemplate
   Renderer, parser, token spans, compact row syntax, and terminal semantics.

5. Objective
   Surface-bound loss or supervision primitive.

6. DecodeRuntime
   Shared generation API for offline inference and Stage-2 rollout training.

7. Artifact / MetricEvent
   Canonical outputs, provenance, metric identities, aliases, and reducers.
```

Everything else should be classified as one of:

```text
ACTIVE          Executable, documented, tested, and used by official configs.
COMPATIBILITY   Temporary bridge for old configs/artifacts; must have an owner and expiry.
RETIRED         Historical only; may appear in docs or failure-message tests, not active code paths.
```

## High-level diagnosis

The current repository is not directionless. The architecture already contains strong convergence signals:

- Training is intended to be YAML-first.
- Data should be offline-prepared and consumed as-is.
- Runtime image resizing is forbidden because it breaks coordinate alignment.
- Compact detection templates are becoming the canonical formatting layer.
- Stage-2 has been collapsed conceptually toward `stage2_rollout_correction`.
- Inference is moving toward a shared runtime and backend abstraction.
- Metrics are moving toward typed `MetricEvent` identities rather than ad hoc flat keys.

The main problem is that these newer contracts are not yet the only execution paths. Older abstractions still sit in active modules, especially around `src.sft`, dataset construction, Stage-2 trainer internals, legacy inference flags, and compatibility facades.

The codebase currently behaves like a research notebook that has grown a skeleton, muscles, and three spare wings. The proposal is not to make it smaller by deleting capability blindly. The proposal is to make every capability declare its owner, lifecycle, and execution path.

## Main sources of redundancy and unclear ownership

### 1. `src.sft` is still a broad orchestration funnel

`src.sft` currently resolves config, applies runtime policy, handles image-root resolution, installs coord-token adapters, validates cache and packing policy, builds datasets, applies static packing, selects trainers, builds manifests, and bridges current and legacy detection configs.

This creates three issues:

1. Surface-specific decisions live in a global entrypoint.
2. New experiments tend to add more conditionals to the runner.
3. Ownership is hard to determine because many subsystem policies converge in one file.

**Recommendation:** keep `src.sft` as a thin entrypoint and move decisions into surface-owned pipelines and runtime modules.

### 2. Training surfaces exist as shadow concepts, not executable owners

The current surface resolver already describes a clean future model, but pipeline descriptors are still mostly identity-only. This means the architecture says “surface-owned pipeline,” while execution still says “global runner plus conditionals.”

**Recommendation:** make `TrainingSurfaceResolver` active and make each `TrainingPipeline` executable.

### 3. Dataset naming and responsibilities still reflect dense-caption history

The current dataset layer contains valid and useful behavior, but names like `dense_caption`, summary/dense mode, augmentation hooks, and caption-era builders obscure the current goal: detection/grounding with a strict offline geometry contract.

**Recommendation:** introduce a canonical data IR and route all active training/inference data through it. Keep dense-caption compatibility only as a temporary facade if needed.

### 4. Inference and rollout are close to unified, but still have owner-shaped adapters

Offline inference and Stage-2 rollout both need prompt construction, decode policy, backend dispatch, trace normalization, and artifacts. Current code already has shared request/result concepts, but Stage-2 still passes around trainer-owner-shaped state in many helpers.

**Recommendation:** define a shared `DecodeRuntime` API that receives explicit facts rather than trainer objects.

### 5. Evaluation is decomposed, but metric and artifact compatibility still leak

The evaluator has a durable orchestration layer and separate COCO/LVIS/F1-ish modules, but legacy names such as `pred.jsonl` and facade patching still exist. Training-time metrics and offline metrics also use different flat naming conventions.

**Recommendation:** make `MetricEvent` the canonical metric production format across training, rollout, and offline evaluation. Flat names should be generated aliases, not source-level truths.

### 6. Config has two competing shapes

Current configs still expose older `custom.*` fields and ms-swift-shaped sections, while the newer surface schema already defines a cleaner domain model: `run`, `surface`, `data`, `template`, `supervision`, `objectives`, `observability`, `artifacts`, `runtime`, and `experimental`.

**Recommendation:** use the domain schema as the public shape and project into ms-swift / legacy structures only at the boundary.

### 7. Experiment-specific code has become semi-permanent

The repository already rejects several removed Stage-2 variants and old mechanisms, which is good. But the surrounding code still contains compatibility names, comparator configs, retired wrappers, and historical helper formats.

**Recommendation:** make lifecycle state explicit and machine-checkable: active, compatibility, retired.

## Recommended target architecture

### Package-level target

```text
src/config/
  Own YAML loading, inheritance, validation, and RunPlan construction.

src/training/
  Own training surfaces, pipelines, runtime policy, packing, cache, adapters, and trainer projection.

src/data/
  Own JSONL loading, image resolution, record IR, and offline geometry validation.

src/detection/
  Own detection templates, render/parse, token spans, compact row syntax, and sequence semantics.

src/infer/
  Own prompt policy, decode requests/results, backend execution, rollout decode, and inference artifacts.

src/eval/
  Own artifact evaluation, COCO/LVIS/F1-ish metrics, duplicate guard, and reports.

src/metrics/
  Own MetricEvent, reducers, aliases, compatibility metric names, and event sinks.

scripts/
  Thin user-facing entrypoints only. No research logic.
```

### Canonical training surfaces

The active training surface set should be closed:

```text
stage1_json_ce
  Plain Stage-1 JSON / CoordJSON token CE baseline.

stage1_compact_trie_ce
  Stage-1 compact-full template with trie CE and optional coordinate objectives.

stage2_rollout_correction
  Stage-2 rollout prefix plus GT correction / residual-set supervision.
```

Removed or old names should not be accepted except through explicit failure messages:

```text
stage2_ab_training
stage2_two_channel
rollout_matching_sft
stage2_rollout_aligned
stage2_rollout_runtime
```

### Canonical data flow

```text
public_data/offline tools
  -> prepared JSONL + images + metadata
  -> DataView validation
  -> CoordExpRecord
  -> DetectionTemplate rendering
  -> tokenizer/template encoding
  -> training or inference
```

No active path should silently:

- resize images at runtime;
- convert bbox parameterizations at runtime;
- infer missing geometry contracts;
- mutate object order without a declared surface policy;
- mix prepared and unprepared coordinate surfaces.

### Canonical decode flow

```text
PromptBundle[]
  + DetectionDecodeRequest
  + BackendSpec
  -> DecodeRuntime.generate(...)
  -> DetectionDecodeResult[]
  -> parser / artifact writer / metric consumer
```

Offline inference and Stage-2 rollout should share this path.

### Canonical metric flow

```text
metric producer
  -> MetricEvent[]
  -> reducer
  -> flat log aliases, metrics.json, summary.json, provenance sidecars
```

Metrics should not originate as arbitrary flat dictionaries unless they are immediately converted to typed events.

## Simplification principles

### Principle 1: One concept, one owner

Every architectural concept should have one owner module. Compatibility facades may import or re-export that owner, but source-level edits should target the owner.

Examples:

```text
Detection sequence format -> src/detection/template.py
Decode policy             -> src/infer/runtime.py
Metric identity           -> src/metrics/events.py
Training surface          -> src/training/surfaces.py
Offline JSONL contract    -> src/data/ or current data contract owner during migration
```

### Principle 2: Entry points should not own research logic

Scripts and CLI modules should parse config, call the correct library entrypoint, and print a short result. They should not perform vLLM server planning, data policy validation, trainer-specific branching, or metric transformation directly.

### Principle 3: Config resolves into a typed plan before execution

After config loading, execution code should consume typed plans and explicit facts. It should not continue to inspect raw YAML dictionaries or `custom.extra` payloads deep in the stack.

### Principle 4: Compatibility must be named and expiring

Compatibility code must declare:

```text
owner
reason
accepted callers
removal condition
expiry or review date
```

No compatibility path should be treated as a default active path.

### Principle 5: Runtime must not repair offline data mistakes

If images, geometry, bbox formats, coordinate surfaces, or metadata are wrong, runtime should fail fast. Silent repair creates incomparable experiments.

### Principle 6: Stage-2 should use components, not trainer-local sprawl

The Stage-2 trainer should coordinate components. It should not own parser behavior, decode backend policy, assignment algorithms, target construction, packing policy, and metric taxonomy all in one class.

### Principle 7: Artifacts are architecture, not leftovers

`gt_vs_pred.jsonl`, scored artifacts, provenance sidecars, metrics files, and run manifests should be designed as stable contracts. Evaluation should consume those contracts instead of rediscovering intent from local paths or legacy aliases.

## Major recommendations and reasoning

### Recommendation A: Activate surface-owned training pipelines

**Problem:** The repo already has a closed surface registry, but execution still routes through a global runner with many conditionals.

**Target:** Each surface owns dataset construction, collator choice, objective wiring, trainer class selection, metric event production, packing/cache eligibility, and validation.

**Reasoning:** This prevents future research work from adding more special cases to `src.sft`. It also makes it obvious where a Stage-1 vs Stage-2 change belongs.

**First moves:**

1. Add executable methods to `TrainingPipeline`.
2. Resolve every config into `ResolvedTrainingRun` or a successor `RunPlan`.
3. Move Stage-1 and Stage-2 branching out of `src.sft`.
4. Keep `src.sft` as a compatibility entrypoint.

### Recommendation B: Introduce a canonical data IR

**Problem:** The active data path still carries dense-caption naming and modes while the project now centers detection/grounding.

**Target:** `CoordExpRecord` or equivalent typed IR, with image path, width/height, objects, geometry, metadata, and coordinate surface validated before rendering.

**Reasoning:** A canonical IR prevents divergence between training, inference, evaluation, visualization, and preprocessing. It also makes offline vs runtime responsibility testable.

**First moves:**

1. Define `CoordExpRecord`, `DetectionObject`, and `ImageRef`.
2. Convert current JSONL loading into IR construction.
3. Make templates consume IR rather than loose dictionaries.
4. Keep old dataset classes as adapters until all active configs migrate.

### Recommendation C: Unify offline inference and Stage-2 rollout decode

**Problem:** Decode policy is shared conceptually, but trainer-owned rollout helpers still infer behavior from owner objects.

**Target:** A `DecodeRuntime` that takes explicit request objects and backend specs.

**Reasoning:** Rollout training and reportable inference must be comparable. Shared decode fingerprints, trace normalization, and backend projection reduce subtle differences between HF and vLLM execution.

**First moves:**

1. Define `DecodeBatchRequest` and `BackendSpec`.
2. Adapt offline inference to call `DecodeRuntime.generate`.
3. Adapt Stage-2 rollout to build the same request type.
4. Remove owner-shaped backend interpretation after parity tests pass.

### Recommendation D: Make `MetricEvent` the canonical metric layer

**Problem:** Stage-1, Stage-2, rollout, and offline eval produce different flat metric surfaces.

**Target:** All metric producers emit typed `MetricEvent` objects, then a reducer generates flat aliases and artifact files.

**Reasoning:** Metric identity should survive renaming. Aliases are needed for compatibility, but aliases should not define semantics.

**First moves:**

1. Define event producers for Stage-1 metrics.
2. Convert Stage-2 pending-log snapshots into events.
3. Convert offline eval results into events.
4. Add alias collision tests.

### Recommendation E: Convert config into a domain plan

**Problem:** Public config still exposes old `custom.*` and ms-swift-shaped internals.

**Target:** Public configs resolve into a domain plan with `run`, `surface`, `data`, `template`, `supervision`, `objectives`, `observability`, `artifacts`, `runtime`, and optional `experimental`.

**Reasoning:** Config is the boundary between experiment design and implementation. If it remains a bag of old fields, old concepts keep leaking into new code.

**First moves:**

1. Make the surface-domain schema loadable for smoke configs.
2. Add a projection layer to current `TrainingConfig` / ms-swift arguments.
3. Migrate one Stage-1 config and one Stage-2 config.
4. Mark `custom.*` as internal compatibility.

### Recommendation F: Create an architecture lifecycle registry

**Problem:** The repo knows some variants are removed, but lifecycle state is scattered through code and docs.

**Target:** A machine-readable registry such as `architecture_state.yaml` listing active, compatibility, and retired surfaces, scripts, artifacts, config keys, and mechanisms.

**Reasoning:** This prevents retired experiments from re-entering through new configs or helper scripts.

**First moves:**

1. Add registry file.
2. Add tests that active configs only reference active surfaces.
3. Add tests that retired names fail with canonical messages.
4. Require owner/expiry metadata for compatibility entries.

## Non-goals

This proposal does not require immediate deletion of all legacy code. It also does not require changing model behavior, loss math, or benchmark claims in one large PR. The migration should be incremental and test-led.

This proposal also does not attempt to replace ms-swift. The goal is to isolate ms-swift projection behind a clean CoordExp plan boundary.

## Expected end state

A future contributor should be able to answer these questions quickly:

- Which training surface am I modifying?
- Which data IR am I consuming?
- Which template owns this serialization?
- Which decode runtime path generated this artifact?
- Which metric event defines this number?
- Is this path active, compatibility, or retired?

If those answers are easy, the codebase has converged.
