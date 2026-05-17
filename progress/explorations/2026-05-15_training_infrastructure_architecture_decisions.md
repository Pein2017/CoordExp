---
doc_id: progress.explorations.training-infrastructure-architecture-decisions-2026-05-15
layer: progress
doc_type: exploration
status: in-progress-decision-record
domain: training-architecture
summary: Decision record from the training infrastructure architecture audit and grill-me loop, focused on duplicate handling, Stage-2 assignment cleanup, and training-time heuristic removal.
updated: 2026-05-15
---

# Training Infrastructure Architecture Decisions (2026-05-15)

This note exports the resolved decisions from the architecture-audit /
`grill-me` discussion so a future context window can continue without relying on
chat history.

This is a planning and decision record only. It does not claim implementation
has happened.

## Current Supersession Notice

This progress note records the live discussion chronologically, so earlier
sections intentionally include names and flow sketches that were later
superseded. For implementation, the Superpowers spec and plan are authoritative
over older sections in this note.

Current implementation terms:

- Use `SupervisionPlan` as the provisional semantic-planning name.
- Do not create public `TargetPlan` APIs unless a later design explicitly
  reopens and resolves the naming decision.
- Use `SupervisionSpan.label_positions`; do not create new
  `prediction_positions` fields.
- Treat duplicate filtering before assignment as the current Stage-2 shadow
  planning contract; assignment runs on accepted survivors, not raw predictions.
- Preserve Stage-2 eval artifacts and experiment provenance until a deliberate
  migration replaces them.

## Source Context

User request:

- Audit the training infrastructure / pipeline / framework at codebase level.
- Design a cleaner architecture around the current confirmed direction.
- Primary future direction is Stage-1 training:
  - standard SFT / CE;
  - entry-trie multiple-positive / multiple-target CE;
  - optional coordinate / regression losses;
  - compact-full template as the primary target.
- Treat old compatibility paths, unhealthy research ideas, and failed
  heuristics critically.
- Use `grill-me` to clarify true research decisions before implementation.

Audit inputs:

- Six read-only subagents inspected:
  - training pipeline and control flow;
  - data / template / tokenizer / span representation;
  - loss modules and multiple-positive trie training;
  - config hierarchy, experiment management, and monitoring;
  - Stage-2 rollout / assignment design;
  - dead code, legacy behavior, and documentation cleanup.
- Current docs and code authority were read from:
  - `docs/PROJECT_CONTEXT.md`;
  - `docs/SYSTEM_OVERVIEW.md`;
  - `docs/IMPLEMENTATION_MAP.md`;
  - `docs/training/STAGE1_OBJECTIVE.md`;
  - `docs/training/STAGE2_RUNBOOK.md`;
  - `docs/training/METRICS.md`;
  - `docs/ARTIFACTS.md`.

## High-Level Diagnosis

The architecture is split between the current desired Stage-1 compact-full
direction and older Stage-2 / rollout / duplicate-control machinery.

Core issues:

- `src/sft.py` is still a broad orchestration entrypoint rather than a thin
  launcher over resolved training surfaces.
- Runtime is selected by `custom.trainer_variant`, not by a first-class
  `TrainingSurface`.
- The repo has multiple incompatible supervision representations:
  legacy label/token-type masks, compact recursive `TokenTarget` sidecars, and
  Stage-2 raw segment metadata.
- Compact-full has good strict template machinery, but training/inference and
  token-role ownership are split across strict and compatibility facades.
- Latest compact recursive CE cannot safely use packing/cache because sidecar
  offset rewriting is not represented as a first-class segment map.
- Stage-2 still carries Hungarian matching and duplicate-negative training
  mechanisms that should not shape the future Stage-1-first architecture.
- Observability is spread across `custom.extra`, `MetricEvent`, Stage-2 pending
  aggregation, monitor dumps, eval artifacts, and docs instead of a typed
  config domain.
- Some `progress/` notes record negative or unhealthy directions but still look
  active enough to steer future work.

## Resolved Decisions

### Decision 1: Duplicate Handling Canonical Path

Decision:

- Canonical duplicate handling is pre-forward filtering plus diagnostics.
- Duplicate predictions should be filtered before target realization and before
  the teacher-forced positive forward pass.
- Duplicate objects should not enter the main positive target sequence.

Canonical flow:

```text
rollout / predicted objects
  -> parse
  -> duplicate filter / survivor selection
  -> greedy IoU assignment / pairing on duplicate survivors
  -> build positive supervision sequence from survivors + valid FN insertions
  -> forward pass
  -> losses on explicit positive spans
```

Keep:

- duplicate filtering before forward;
- duplicate diagnostic metrics;
- duplicate cluster / near-IoU / survivor reason telemetry;
- optional posthoc guarded eval artifacts, explicitly labeled as analysis views.

Remove from canonical training:

- duplicate-negative objectives;
- handcrafted anti-duplicate losses at duplicated sequence positions;
- duplicate objects in positive teacher-forced targets.

Rationale:

- Duplicate predictions are invalid positive supervision candidates.
- The model needs to learn to emit valid objects, not merely be punished for
  nearby or repeated boxes.
- Handcrafted local negative losses are brittle and easy to misalign with
  legitimate nearby same-class crowded objects.

### Decision 2: Duplicate-Negative Training Full Removal

Decision:

- Duplicate-negative training is removed, not merely deprecated.
- Do not keep an experiment-only implementation in the clean codebase unless
  the user later explicitly asks to resurrect it from git history.

Remove:

- `loss_duplicate_burst_unlikelihood` training objective code;
- default/canonical config entries for the module;
- tests that preserve the old duplicate-negative behavior;
- docs that describe duplicate-negative training as active or supported.

Keep minimally:

- one short historical/progress note explaining why it was removed;
- git history and old run artifacts as provenance;
- duplicate filtering and diagnostics.

Implementation consequence:

- Any previously drafted plan language that keeps duplicate-negative loss as an
  experiment-only module is superseded by this decision.

### Decision 3: Adjacent Repulsion Full Removal

Decision:

- Adjacent repulsion should be completely removed along with duplicate-negative
  training.

Rationale:

- The user classified adjacent repulsion as `治标不治本`: it treats symptoms
  rather than teaching valid object emission.
- Forcing the model not to generate a nearby bounding box does not teach it how
  to emit the correct missing or distinct object.
- It has appeared as anti-duplication / damping heuristic clutter and should
  not remain in the clean loss architecture.

Remove:

- adjacent-repulsion training code;
- adjacent-repulsion config knobs;
- zero-weight adjacent-repulsion config clutter;
- tests/docs that preserve it as an active objective.

Keep:

- historical notes and old artifacts only.

### Decision 4: EOS Loosen / Forced Continuation Training Removal

Decision:

- Remove training-time EOS loosen, forced-continuation, and stop-gate hacks from
  canonical training code/configs.
- Keep monitoring metrics and diagnostic probes for future comparison.

Remove from training:

- EOS-loosen training configs as production-looking presets;
- forced-continuation loss knobs;
- stop-gate training mechanisms;
- docs that present these as current supported training strategy.

Keep as diagnostics:

- teacher-forced continue-vs-EOS probes;
- decode stop diagnostics;
- metrics that help explain early stopping or conservative decode;
- historical negative-result notes.

Rationale:

- These mechanisms can make the model output more, but do not necessarily make
  it output valid objects.
- Valid object emission should be learned through compact-full structure,
  explicit object-entry spans, trie multi-positive CE, positive target coverage,
  and coordinate/regression objectives where needed.

### Decision 5: Hungarian / Rollout-Aligned Removal Is Staged

Decision:

- Do not fully remove the old Stage-2 executable path in the first cleanup wave.
- De-canonicalize Hungarian and rollout-aligned compatibility now.
- Remove them only after a greedy-IoU replacement path exists and passes smoke.

First cleanup wave:

- fully remove duplicate-negative training code/config/tests;
- fully remove adjacent repulsion code/config/tests;
- remove training-time EOS/forced-continuation hacks while preserving
  diagnostics;
- remove old import shims and dead aliases where safe;
- mark old rollout-aligned / Hungarian Stage-2 as legacy and not architecture
  shaping.

Second cleanup wave:

- introduce greedy IoU assignment seam;
- migrate active `stage2_two_channel` to it;
- smoke-test the replacement;
- then delete old Hungarian / rollout-aligned executable paths if no longer
  needed.

Rationale:

- Duplicate-negative and adjacent-repulsion training are rejected ideas and can
  be removed now.
- Hungarian / rollout-aligned code is entangled with runnable Stage-2 code,
  configs, tests, eval artifacts, and old diagnosis paths.
- Removing it before a greedy-IoU replacement exists risks losing a runnable
  Stage-2 baseline.

### Decision 6: Cleanup First, Reusable Abstractions Second

Decision:

- The first implementation pass should be cleanup-only.
- Remove rejected mechanisms and clean code/config/tests/docs before adding the
  new reusable abstractions.
- After cleanup, introduce the reusable abstractions that are justified by the
  simplified architecture.

First pass:

- remove duplicate-negative training code/config/tests/docs;
- remove adjacent repulsion code/config/tests/docs;
- remove training-time EOS loosen / forced-continuation / stop-gate mechanisms,
  while preserving diagnostic probes and monitoring metrics;
- remove safe legacy import shims and dead aliases;
- update minimal docs/progress notes so future agents do not resurrect removed
  mechanisms.

Second pass:

- introduce reusable architecture pieces such as `TrainingSurface`,
  `DetectionTemplateCodec`, `EncodedDetectionView`, `SupervisionSpan`,
  `ObjectiveRunner`, `AssignmentStrategy`, `DuplicateFilter`,
  `ObservabilityConfig`, and `DiagnosticsArtifactWriter`;
- add these only where cleanup has exposed a real repeated responsibility, not
  as speculative abstraction.

Rationale:

- Cleanup-only first keeps the blast radius small.
- It separates research decisions from architecture construction.
- It avoids hiding regressions behind simultaneous deletion and abstraction
  changes.
- The reusable abstractions should be advocated from a clean surface, not built
  around preserving unhealthy legacy behavior.

### Decision 7: First Cleanup Wave Includes Safe Dead Shims

Decision:

- The first cleanup wave should remove both rejected training mechanisms and
  safe dead import/config shims.
- The pass should remain behavior-preserving outside the rejected mechanisms.

Remove in the first cleanup wave:

- `loss_duplicate_burst_unlikelihood` training code/config/tests/docs;
- adjacent-repulsion training code/config/tests/docs;
- EOS-loosen / forced-continuation / stop-gate training hooks;
- stale `stage1_set_continuation` bytecode / empty surfaces;
- `stage2_ab_training` import shim if no live imports require it;
- `rollout_matching_sft` import shim if no live imports require it;
- old config aliases that already fail fast;
- docs that advertise removed names as live.

Do not remove in the first cleanup wave:

- executable `stage2_rollout_aligned` trainer;
- Hungarian matching implementation;
- posthoc eval duplicate guard;
- monitoring metrics and diagnostic probes;
- major `src.sft` architecture.

Rationale:

- Removing rejected mechanisms but leaving dead shims would keep the repo
  visually and semantically haunted.
- Removing safe shims improves architecture clarity without requiring the new
  abstraction layer.
- Runnable Stage-2 / Hungarian and diagnostic/eval analysis surfaces are staged
  separately to avoid unnecessary behavior loss.

### Decision 8: Preserve Progress Evidence, Demote Authority

Decision:

- Do not delete old `progress/` notes merely because they promoted removed
  ideas.
- Preserve them as historical evidence, mechanism studies, negative results, or
  superseded directions.
- Demote their authority through explicit statuses and router/index updates.

Keep as evidence:

- duplication-collapse studies;
- EOS-loosen negative results;
- adjacent-repulsion / anti-dup diagnostics;
- Stage-2 self-context history;
- stop-gate and continuation diagnostics;
- old rollout and matching investigations.

Change statuses where appropriate, using the repo's existing hyphenated style:

- `active-reference` -> `concluded-negative`;
- `active-reference` -> `mechanism-evidence`;
- `active-reference` -> `superseded`;
- `active-reference` -> `archive-only`.

Update routing surfaces:

- `progress/index.yaml`;
- `progress/diagnostics/README.md`;
- `progress/explorations/README.md`;
- `docs/catalog.yaml` if it points to removed ideas as active current guidance.

Rationale:

- The notes explain why rejected mechanisms are being removed.
- Deleting them would erase the causal trail.
- Leaving them as active references would invite future agents to reuse failed
  ideas.
- The correct cleanup is status demotion and router hygiene, not evidence
  deletion.

### Decision 9: OpenSpec Only For Direct Stable-Contract Contradictions

Decision:

- Do not start OpenSpec work by default in the first cleanup pass.
- First cleanup should focus on code, configs, tests, current docs, and
  progress routers.
- Touch OpenSpec only if an existing stable spec directly says a removed
  mechanism is required current behavior.

First cleanup pass:

- remove rejected training mechanisms;
- remove safe dead shims;
- update current docs under `docs/`;
- update `progress/` routing/statuses;
- avoid new OpenSpec changes unless required.

Only update OpenSpec if:

- a stable spec requires duplicate-negative training as current behavior;
- a stable spec requires adjacent repulsion as current behavior;
- a stable spec requires EOS loosen / forced-continuation training hacks as
  current behavior;
- a stable spec would mislead future implementation after cleanup.

Otherwise:

- record in the cleanup plan that an OpenSpec audit is required before
  archiving the Stage-2 / Hungarian path.

Rationale:

- OpenSpec is downgraded governance in this repo and should not inflate an
  implementation cleanup by default.
- Current docs and executable repo truth should move first.
- Normative stable specs must not contradict removed behavior.

### Decision 10: Delete Positive Support Tests, Add Absence Tests

Decision:

- Remove tests that positively preserve removed mechanisms.
- Keep or add a small number of negative/absence tests so removed mechanisms do
  not silently return.

Delete:

- tests proving `loss_duplicate_burst_unlikelihood` math or active objective
  behavior;
- tests proving adjacent-repulsion math or active objective behavior;
- tests proving EOS-loosen / forced-continuation training behavior;
- tests preserving old import shims as supported APIs.

Convert or add:

- tests that canonical configs reject removed knobs;
- tests that removed modules are absent from active objective registries;
- tests that duplicate diagnostics still work without duplicate-negative
  training;
- tests that EOS/continue diagnostic probes still run outside training;
- tests that posthoc eval duplicate guard is explicitly labeled as an analysis
  view;
- tests that old import/config aliases fail fast with actionable messages when
  relevant.

Rationale:

- Keeping old positive behavior tests would force the codebase to preserve
  rejected ideas.
- Deleting all tests would allow removed mechanisms to creep back silently.
- Absence tests encode the cleanup contract without preserving the unhealthy
  mechanism.

### Decision 11: Clean New Metric Writes, Tolerant Old Artifact Reads

Decision:

- New training runs must not emit metrics for removed training mechanisms.
- Artifact readers and analysis tools may remain tolerant of old metric keys so
  historical runs stay inspectable.

New run write path must not emit:

- `loss_duplicate_burst_unlikelihood` metrics;
- adjacent-repulsion metrics;
- EOS-loosen training metrics;
- forced-continuation / stop-gate training metrics.

Historical read path may still read:

- old duplicate-negative metrics;
- old adjacent-repulsion metrics;
- old EOS/continue hack metrics;
- old rollout/Hungarian metrics.

Reader requirements:

- label removed keys as legacy or removed if surfaced;
- do not treat removed keys as current metric contracts;
- do not require current configs to emit old keys for parity.

Rationale:

- Old runs are evidence and should remain inspectable.
- New metric contracts should not preserve rejected mechanisms.
- The correct split is strict clean writes and tolerant historical reads.

### Decision 12: Rebuild Bottom-Up After Cleanup

Decision:

- After the cleanup-only first wave, introduce reusable abstractions bottom-up.
- Start with the supervision contract rather than top-level orchestration.
- Do not introduce `TrainingSurface` before the lower-level span/objective
  representation is clean enough to support it.

Post-cleanup abstraction order:

1. `SupervisionSpan` and `TargetDistribution`;
2. `DetectionTemplateCodec` and `EncodedDetectionView`;
3. `ObjectiveRunner` consuming spans;
4. `TrainingSurface` resolving configs into dataset/objective/packing/artifact
   plans;
5. `ObservabilityConfig` and metric registry.

Rationale:

- The deepest duplication is the incompatible supervision representation across
  legacy Stage-1, compact recursive CE, and Stage-2.
- A top-down `TrainingSurface` first would risk becoming a wrapper over the same
  messy mask/meta/sidecar world.
- A shared span contract gives template codecs, objective modules, packing maps,
  metrics, and surfaces a reusable core.

### Decision 13: Cross-Stage Span Contract With Stage-1 And Stage-2 Wrappers

Decision:

- `SupervisionSpan` and `TargetDistribution` should be designed as cross-stage
  contracts from the start.
- The implementation should prepare wrappers/adapters for both Stage-1 and
  Stage-2 because the next research direction mainly focuses on Stage-2.
- Stage-1 compact-full can still be the first clean validation surface, but the
  contract must not be Stage-1-only.

Required wrapper/adapters:

- Stage-1 compact-full recursive CE adapter:
  - converts compact template spans, object-entry trie targets, coordinate
    targets, EOS/stop diagnostics, and type/schema roles into shared spans.
- Stage-1 JSON CE adapter:
  - converts ordinary JSON chat supervision into hard-token spans and optional
    coordinate/regression spans.
- Stage-2 Channel-A adapter:
  - converts GT-anchored teacher-forced supervision into the same span format.
- Stage-2 Channel-B adapter:
  - converts rollout-filtered clean-prefix / FN-injection target plans into
    spans after duplicate filtering and assignment.
- Stage-2 diagnostic wrapper:
  - keeps rollout parse, duplicate filtering, assignment, EOS/continue, and
    decode diagnostics attached as provenance or metric events rather than
    training hacks.

Design requirements:

- Include optional `stage`, `channel`, `segment_id`, and `provenance` fields so
  Stage-2 can represent Channel-A / Channel-B without inventing a second meta
  system.
- Keep duplicate-negative targets out of the canonical target-distribution
  family.
- Keep removed EOS/forced-continuation behavior out of training spans while
  allowing diagnostic events to reference stop/continue positions.
- Make the Stage-2 wrapper consume a cleaned target plan, not raw historical
  `tail_desc_pos` / `tail_ignore_pos` / duplicate-negative metadata.

Rationale:

- A Stage-1-only span type would likely need a disruptive rename or reshape
  when Stage-2 work resumes.
- A cross-stage type with narrow first adapters allows Stage-2 research to reuse
  the core abstraction without preserving old Stage-2 heuristics.
- The wrapper boundary lets the cleanup proceed first while making the next
  Stage-2 research idea architecture-compatible.

### Decision 14: Stage-2 Uses A Clean Target Plan Before Span Conversion

Decision:

- Stage-2 should not be adapted directly from the old raw target-builder
  metadata into `SupervisionSpan`.
- Introduce a clean object-level `Stage2TargetPlan` first, then convert that
  target plan into shared spans through Stage-2 Channel-A / Channel-B adapters.
- `Stage2TargetPlan` is the Stage-2 counterpart to Stage-1's normalized
  detection document plus object-entry target plan.

Required shape:

```text
rollout text / sampled predictions
  -> RolloutViews
  -> DuplicateFilter, before positive target realization
  -> AssignmentStrategy over accepted survivors, default future strategy = greedy IoU
  -> Stage2TargetPlan, object-level supervision intent
  -> DetectionTemplateCodec
  -> EncodedDetectionView
  -> SupervisionSpan + TargetDistribution
  -> ObjectiveRunner
```

`Stage2TargetPlan` should own:

- accepted rollout objects;
- matched ground-truth objects;
- false-negative insertions;
- ignored or filtered predictions with reason codes;
- channel identity, such as Channel-A or Channel-B;
- object ordering intent;
- target text / template-ready object entries before tokenization;
- provenance references back to rollout diagnostics.

`Stage2TargetPlan` should not own:

- duplicate clustering as a training objective;
- duplicate-negative targets;
- Hungarian-specific assignment state;
- raw token positions such as historical `tail_desc_pos` or
  `tail_ignore_pos`;
- raw rollout token traces except by diagnostic/provenance reference;
- EOS/forced-continuation training spans.

Rationale:

- A thin wrapper over historical Stage-2 metadata would pollute the shared span
  contract with old rollout-aligned assumptions.
- Stage-2's next research direction needs a reusable assignment/target
  abstraction, not another loss-side metadata bridge.
- Object-level target planning keeps assignment, filtering, target
  realization, tokenization, and loss math separate enough to test and replace.

Consequence:

- The unified architecture should be built around a shared supervision spine
  for both Stage-1 and Stage-2.
- Stage-specific code should live mainly in plan builders and adapters, not in
  the objective/loss modules.

### Decision 15: Duplicate Filtering Is Separate From Stage-2 Target Planning

Decision:

- Duplicate filtering should be a separate pre-target-planning module.
- `DuplicateFilter` runs before `Stage2TargetPlan` construction.
- `Stage2TargetPlan` receives survivor objects plus provenance references to
  duplicate diagnostics.
- `Stage2TargetPlan` should not decide what is a duplicate.
- `SupervisionSpan` and `ObjectiveRunner` should never receive
  duplicate-negative spans, anti-duplicate penalty targets, or handcrafted
  duplicate-position loss metadata.

Canonical flow:

```text
RolloutViews
  -> DuplicateFilter
  -> AssignmentStrategy over accepted survivors
  -> Stage2TargetPlan
  -> DetectionTemplateCodec
  -> EncodedDetectionView
  -> SupervisionSpan
  -> ObjectiveRunner
```

`DuplicateFilter` owns:

- duplicate clustering;
- survivor selection;
- dropped-object reason codes;
- duplicate diagnostics and telemetry;
- optional links to posthoc guarded-eval analysis views.

`Stage2TargetPlan` owns:

- clean object-level supervision intent;
- accepted rollout objects;
- false-negative insertions;
- ignored objects with reason references;
- channel identity and object ordering.

Rationale:

- Duplicate handling is target hygiene and diagnosis, not a learning objective.
- Putting duplicate filtering inside `Stage2TargetPlan` would make target
  planning a mixed policy sink.
- Putting duplicate filtering inside loss would resurrect the rejected
  duplicate-negative training mechanism.

### Decision 16: Use Professional Pipeline / Component Architecture

Decision:

- Replace casual "routine", "helper", and "plugin" terminology with a mature
  pipeline/component architecture.
- `TrainingPipeline` is the config-visible orchestration unit for one training
  surface.
- Reusable mechanisms are typed `TrainingComponent` / `Strategy`
  implementations selected through internal registries.
- Do not expose a broad dynamic plugin system unless a future need justifies
  third-party or runtime-loaded extensions.

Approved public concepts:

- `TrainingSurface`;
- `TrainingPipeline`;
- `TrainingComponent`;
- `StrategyRegistry`;
- `TargetPlan`;
- `SupervisionSpan`;
- `TargetDistribution`;
- `ObjectiveModule`;
- `ObservabilityService`;
- `DiagnosticsProbe`.

Avoided public concepts:

- routine;
- helper;
- plugin;
- trainer-variant branch;
- callback soup.

Pipeline responsibility:

- own the ordered lifecycle for a training surface;
- resolve and validate data, target planning, template, span adaptation,
  objectives, observability, and artifact boundaries;
- make the stage dataflow readable from one orchestration module;
- avoid implementing mechanism details.

Component responsibility:

- implement one narrow typed contract;
- receive explicit inputs and return explicit outputs;
- avoid reading global config directly;
- avoid writing artifacts directly except through injected observability
  services;
- be reusable across Stage-1 and Stage-2 when the contract is genuinely shared.

Target module direction:

```text
src/training/pipelines/
  stage1_json_ce.py
  stage1_compact_trie_ce.py
  stage2_two_channel.py

src/training/components/
  assignment/
  duplicate_filtering/
  target_planning/
  span_adapters/
  objectives/
  observability/
```

Rationale:

- The architecture should read like professional training infrastructure, not
  a bag of helper hooks.
- Pipelines make workflow order explicit.
- Components preserve reuse without creating an unbounded plugin ecosystem.
- The split gives Stage-1 and Stage-2 a common engineering language while
  preserving their different target-planning needs.

### Decision 17: Use A Pipeline Protocol, Not A Heavy Base Class

Decision:

- Stage-1 and Stage-2 should share a small `TrainingPipeline` protocol.
- They should not share a heavy inheritance base class with many protected
  hooks.
- Shared reuse should happen through typed contracts and components below the
  pipeline layer.

Conceptual protocol:

```python
class TrainingPipeline(Protocol):
    def validate(self, context: TrainingContext) -> PipelineValidationReport: ...
    def build_dataset_plan(self, context: TrainingContext) -> DatasetPlan: ...
    def build_target_plan(self, batch: SourceBatch) -> TargetPlan: ...
    def build_supervision(self, target_plan: TargetPlan) -> SupervisionBatch: ...
    def build_objectives(self, context: TrainingContext) -> ObjectiveRunner: ...
    def build_observability(self, context: TrainingContext) -> ObservabilityService: ...
```

Pipeline implementations:

- `Stage1JsonCePipeline`;
- `Stage1CompactTrieCePipeline`;
- `Stage2TwoChannelPipeline`.

Shared layer:

- `TrainingContext`;
- `TargetPlan` interface;
- `DetectionTemplateCodec`;
- `EncodedDetectionView`;
- `SupervisionSpan`;
- `ObjectiveRunner`;
- `ObservabilityService`;
- typed component registries.

Rationale:

- Stage-1 and Stage-2 differ most at target-planning time.
- A shared base implementation would likely become another conditional
  orchestration tangle.
- A protocol gives a consistent lifecycle while allowing each pipeline to own
  its stage-specific orchestration.
- If repeated implementation appears later, extract concrete shared helpers
  then; do not guess the hierarchy upfront.

### Decision 18: Use A Typed Target-Plan Family For Flexibility And Scale

Decision:

- `TargetPlan` should be a narrow protocol/interface.
- Concrete target plans should be stage-specific and approach-specific
  dataclasses.
- This is the most flexible and scalable design for future research because
  target-building ideas can vary without polluting a single shared mega-object.

Do not use:

```python
@dataclass
class TargetPlan:
    stage1_objects: ...
    stage2_rollouts: ...
    channel_b_clean_prefix: ...
    false_negative_insertions: ...
    trie_targets: ...
    assignment_result: ...
    duplicate_filter_result: ...
```

Use a narrow shared contract:

```python
class TargetPlan(Protocol):
    @property
    def sample_id(self) -> str: ...
    @property
    def stage(self) -> TrainingStage: ...
    @property
    def template_id(self) -> str: ...
    @property
    def object_entries(self) -> Sequence[TemplateObjectEntry]: ...
    @property
    def provenance(self) -> TargetPlanProvenance: ...
```

Initial concrete plans:

- `Stage1JsonTargetPlan`;
- `Stage1CompactTrieTargetPlan`;
- `Stage2ChannelATargetPlan`;
- `Stage2ChannelBTargetPlan`.

Future research can add plans such as:

- `Stage2GreedyIoUTargetPlan`;
- `Stage2SelfCorrectionTargetPlan`;
- `Stage2ContrastiveRolloutTargetPlan`;
- `Stage1AlternativeTrieTargetPlan`;
- `Stage1CoordinateFocusedTargetPlan`.

Adapter contract:

```text
TargetPlan implementation
  -> matching SpanAdapter
  -> SupervisionBatch[SupervisionSpan]
```

Example adapters:

- `Stage1JsonSpanAdapter`;
- `Stage1CompactTrieSpanAdapter`;
- `Stage2ChannelASpanAdapter`;
- `Stage2ChannelBSpanAdapter`.

Design rules:

- shared fields stay minimal and semantic;
- approach-specific fields live only on concrete plan dataclasses;
- span adapters declare which concrete plan type they consume;
- objective modules consume only `SupervisionSpan` / `TargetDistribution`, not
  target-plan internals;
- new research ideas add a new plan and adapter when their supervision intent
  is genuinely different;
- do not add optional fields to the base protocol merely for one experiment.

Rationale:

- The target-plan layer is the main research variation seam.
- A giant unified dataclass would be superficially convenient but would quickly
  accumulate optional fields, hidden invariants, and invalid combinations.
- A protocol plus typed concrete plans keeps extension cheap while preserving
  explicit data flow.
- This design supports both stable current paths and future Stage-2 ideas
  without making Stage-1 carry Stage-2 metadata.

### Decision 19: Separate Semantic Supervision Planning From Span Adaptation

Decision:

- Semantic supervision planning and token-span adaptation should be separate
  steps.
- The semantic planning object should use the current working name
  `SupervisionPlan`.
- `SupervisionPlan` is still in discussion and should be treated as provisional
  naming, not final stable API vocabulary.
- The approved architecture is:

```text
SupervisionPlanner
  -> SupervisionPlan
  -> span adapter
  -> SupervisionBatch[SupervisionSpan]
  -> ObjectiveRunner
```

Do not allow:

```text
target planner
  -> SupervisionSpan directly
```

Rationale:

- Semantic supervision intent should be testable without tokenizer, chat
  template, packing, or loss code.
- Span adaptation should own template/token alignment and token-position
  projection.
- Objective modules should consume only `SupervisionSpan` /
  `TargetDistribution`.
- Future research target-building ideas should not need to know template byte
  layout, special-token positions, coordinate slot tokenization, or packing
  rewrite mechanics.

Naming follow-up:

- The term `TargetPlan` is rejected as unclear and too generic for this core
  abstraction.
- `SupervisionPlan` is acceptable as the current working name, but the user has
  noted that `Plan` can still feel ambiguous in code implementation.
- Revisit naming before final implementation planning if a clearer professional
  name emerges.

### Decision 20: SupervisionPlan Is Per-Example / Per-Channel Only

Decision:

- `SupervisionPlan` should represent semantic supervision for one training
  example.
- For Stage-2, each channel-specific example should get its own
  `SupervisionPlan`.
- Shared runtime, batch, tokenizer, config, and observability information
  should live in `SupervisionContext`, not inside `SupervisionPlan`.

Approved flow:

```text
SupervisionPlanner
  -> SupervisionPlan          # one example or one Stage-2 channel example
  + SupervisionContext        # shared runtime/context/provenance info
  -> SpanAdapter
  -> SupervisionBatch
```

`SupervisionPlan` owns:

- semantic supervision intent for one example;
- object entries;
- channel identity when relevant;
- accepted objects;
- false-negative insertions;
- trie branches or multi-positive object-entry structure;
- object ordering intent;
- object-level provenance.

`SupervisionContext` owns:

- template id;
- tokenizer and coordinate vocabulary references;
- assignment config snapshot;
- duplicate-filter config snapshot;
- observability handles;
- batch and run identifiers;
- runtime configuration needed by adapters and components.

Rationale:

- Keeping `SupervisionPlan` per-example makes it easy to test, serialize,
  diff, and reason about.
- Batch/global context would turn `SupervisionPlan` into a hidden state carrier.
- Stage-2 can represent Channel-A and Channel-B as separate plans for the same
  source sample while sharing one context.

### Decision 21: SupervisionPlan Is Optionally Serializable For Diagnostics

Decision:

- `SupervisionPlan` dataclasses should be structurally serializable for
  diagnostics.
- Serialization is optional and diagnostic-only.
- Do not make serialized `SupervisionPlan` records a required training cache or
  stable runtime artifact contract in the initial architecture.

Possible diagnostic artifact:

```text
output_dir/diagnostics/supervision_plans/step_000100.jsonl
```

Record contents may include:

- sample id;
- stage;
- channel;
- template id;
- object entries;
- accepted rollout objects;
- false-negative insertions;
- ignored predictions with reason references;
- assignment summary;
- duplicate-filter summary;
- provenance links.

Rules:

- training code should not depend on reading serialized `SupervisionPlan`
  records back;
- `ObservabilityService` owns optional emission;
- serialized records are debug views, not the canonical internal ABI;
- old diagnostic records may evolve as long as core run artifacts remain
  explicit about schema/version.

Rationale:

- This gives Stage-2 a clear "what did this example intend to teach?" debug
  surface before tokenization.
- It helps inspect assignment, duplicate filtering, false-negative insertion,
  and compact-full rendering decisions.
- Keeping it diagnostic-only avoids turning JSON debug output into a brittle
  cache contract.

### Decision 22: SupervisionContext Is Small, Frozen, And Dependency-Only

Decision:

- `SupervisionContext` should be a small, immutable dependency/context object.
- It should carry resolved policy and service references needed for planning,
  template encoding, span adaptation, objectives, and diagnostics.
- It should not carry mutable per-example state, raw config dictionaries,
  trainer/model handles, or arbitrary extension dictionaries.

Allowed contents:

- template id and template policy;
- tokenizer reference;
- coordinate vocabulary;
- span policy;
- objective policy;
- assignment and duplicate-filter policy snapshots when needed;
- run identity / provenance handles;
- observability handle.

Rejected contents:

- raw source examples;
- rollout objects;
- raw YAML dictionaries;
- mutable metric accumulators;
- model handles;
- trainer handles;
- batch tensors;
- per-example decisions;
- miscellaneous `extra` or `state` dictionaries.

Boundary:

```text
SupervisionContext provides resolved services and policy.
SupervisionPlan carries per-example semantic supervision.
SupervisionBatch carries token-level training spans.
```

Rationale:

- A loose context object would recreate `custom.extra` as Python state.
- Small immutable context makes components testable and keeps data flow
  explicit.
- Per-example decisions should be visible on `SupervisionPlan`, not hidden in a
  shared context object.

### Decision 23: SpanAdapter Consumes EncodedDetectionView

Decision:

- `SpanAdapter` should consume `SupervisionPlan` plus an already-built
  `EncodedDetectionView`.
- `SpanAdapter` should not own template rendering, tokenizer calls, rendered
  text parsing, or compact-full schema-token discovery.
- `DetectionTemplateCodec` owns rendering, tokenization alignment, and token
  role projection.

Approved flow:

```text
SupervisionPlan
  -> DetectionTemplateCodec
  -> EncodedDetectionView
  -> SpanAdapter
  -> SupervisionBatch
```

Responsibilities:

- `DetectionTemplateCodec`
  - renders exact assistant bytes;
  - applies chat/template encoding;
  - projects char spans to token spans;
  - identifies schema, description, coordinate, punctuation/json, and stop
    token roles;
  - produces `EncodedDetectionView`.
- `EncodedDetectionView`
  - is the authoritative token-level representation for a sample.
- `SpanAdapter`
  - maps semantic supervision intent onto token positions exposed by
    `EncodedDetectionView`;
  - emits `SupervisionSpan` objects.
- `ObjectiveModule`
  - consumes `SupervisionSpan` / `TargetDistribution` plus logits.

Rationale:

- Compact-full token roles must be a first-class contract.
- Allowing each adapter to tokenize or parse rendered text would create drift
  between Stage-1 and Stage-2.
- This separation keeps template bytes, token alignment, span construction, and
  loss math independently testable.

### Decision 24: EncodedDetectionView Is The Authoritative Tokenized View

Decision:

- `EncodedDetectionView` should be the authoritative tokenized representation
  for one detection supervision example.
- It should be rich enough to eliminate downstream string parsing.
- It should not contain objective, loss, assignment, duplicate-filter, model
  output, or metric-accumulator state.

Minimum conceptual contents:

```python
@dataclass(frozen=True)
class EncodedDetectionView:
    sample_id: str
    template_id: str
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    labels: tuple[int, ...]
    assistant_span: TokenSpan
    object_entries: tuple[EncodedObjectEntry, ...]
    schema_spans: tuple[TokenSpan, ...]
    description_spans: tuple[TokenSpan, ...]
    coordinate_slots: tuple[EncodedCoordinateSlot, ...]
    stop_spans: tuple[TokenSpan, ...]
    token_roles: tuple[TokenRole, ...]
    prediction_positions: tuple[int, ...]
    provenance: EncodingProvenance
```

For compact-full, `EncodedObjectEntry` should expose:

```python
@dataclass(frozen=True)
class EncodedObjectEntry:
    object_id: str
    object_index: int
    entry_span: TokenSpan
    object_ref_start_span: TokenSpan
    description_span: TokenSpan
    box_start_span: TokenSpan
    coordinate_slots: tuple[EncodedCoordinateSlot, ...]
```

Coordinate slots:

```python
@dataclass(frozen=True)
class EncodedCoordinateSlot:
    object_id: str
    slot_name: Literal["x1", "y1", "x2", "y2"]
    token_span: TokenSpan
    token_id: int
    coord_value: int
```

Allowed contents:

- token ids;
- attention mask;
- labels;
- assistant spans;
- object-entry spans;
- schema-token spans;
- description spans;
- coordinate slots;
- stop-token spans;
- token roles;
- prediction positions;
- encoding provenance.

Rejected contents:

- loss weights;
- objective selection;
- assignment outputs;
- duplicate-filter decisions;
- regression loss values;
- model logits;
- metric accumulators.

Boundary:

```text
EncodedDetectionView = what tokens exist and what semantic roles they have.
SupervisionSpan = which token positions receive which supervision.
ObjectiveModule = how loss is computed from those targets.
```

Rationale:

- Compact-full schema, free-text description, and coordinate-token structure
  must be represented explicitly for Stage-1 and Stage-2.
- Downstream span adapters and objectives should not rediscover token roles by
  parsing strings.
- Keeping objective/loss state out preserves a clean split between encoding,
  supervision, and optimization.

### Decision 25: SupervisionSpan Is Semantic-Region-Level

Decision:

- `SupervisionSpan` should represent a semantic region, not a single token by
  default.
- It should contain explicit prediction positions so causal-shift semantics are
  never implicit.
- `TargetDistribution` attached to the span may define per-position targets
  when needed.

Conceptual shape:

```python
@dataclass(frozen=True)
class SupervisionSpan:
    span_id: str
    sample_id: str
    stage: TrainingStage
    channel: str | None
    object_id: str | None
    role: SpanRole
    token_span: TokenSpan
    prediction_positions: tuple[int, ...]
    target: TargetDistribution
    weight: float
    mask_policy: SpanMaskPolicy
    provenance: SpanProvenance
```

Examples:

- Description CE span:
  - role: `description_text`;
  - token span: free-text description region;
  - prediction positions: each description-token prediction position;
  - target: hard-token CE.
- Compact object-entry trie span:
  - role: `object_entry`;
  - token span: full object entry region;
  - prediction positions: branch decision positions;
  - target: multi-positive token CE / trie distribution.
- Coordinate span:
  - role: `coordinate_slot`;
  - token span: coordinate token;
  - prediction positions: coordinate-token prediction position;
  - target: hard CE, coordinate soft CE, or regression-linked target.

Rejected alternatives:

- one span per token by default:
  - too noisy;
  - loses object/region semantics;
  - awkward for trie and object-level stats.
- one span per object only:
  - too coarse;
  - cannot cleanly mask or weight schema, description, and coordinate roles.
- one global span per example:
  - insufficient for diagnostics and mixed objectives.

Rationale:

- Region-level spans preserve semantic meaning for monitoring.
- Explicit prediction positions avoid off-by-one and causal-shift ambiguity.
- Multi-positive trie CE can attach multiple possible targets to one semantic
  region.
- Coordinate losses can target coordinate slots only.
- Token-type statistics become natural aggregations over span roles.
- Loss modules do not need to parse template structure.

### Decision 26: TargetDistribution Is A Registered Dataclass Family

Decision:

- `TargetDistribution` should be a small registered family of dataclass
  variants.
- It should not be an arbitrary open protocol.
- New distribution types may be added, but only through deliberate registry,
  objective, metric, and config-schema updates.

Initial distribution family:

- `HardTokenDistribution`
  - kind: `hard_token`;
  - one token target per prediction position.
- `MultiPositiveTokenDistribution`
  - kind: `multi_positive_token`;
  - sparse candidate token ids and optional weights per prediction position;
  - central representation for entry-trie multiple-positive CE.
- `CoordinateSoftTokenDistribution`
  - kind: `coordinate_soft_token`;
  - neighborhood distribution over coordinate tokens.
- `BoxRegressionDistribution`
  - kind: `box_regression`;
  - coordinate/bbox regression target attached to object or coordinate slots.

Rejected distribution types:

- `DuplicateNegativeDistribution`;
- `AdjacentRepulsionDistribution`;
- `ForcedContinuationDistribution`;
- `StopGateDistribution`.

Validation contract:

```text
TargetDistribution.kind
  -> ObjectiveModule supports kind
  -> metric schema supports kind
  -> config schema permits kind
```

Example:

```python
@dataclass(frozen=True)
class MultiPositiveTokenDistribution:
    kind: Literal["multi_positive_token"]
    targets_by_position: Mapping[int, SparseTokenTargetSet]
    normalization: str
```

Rationale:

- Loss-target semantics are core training contracts and should not be arbitrary
  runtime duck typing.
- The family must still be extensible for future research.
- A registered dataclass family preserves both controlled semantics and
  research flexibility.
- Objective runners can validate that each span target is supported before loss
  computation.

## External Stack Exploration: ms-swift And Qwen3-VL Loss Boundary

Status:

- Read-only exploration completed after Decision 26.
- Four subagents inspected:
  - ms-swift SFT / `Seq2SeqTrainer` loss path;
  - Qwen3-VL `transformers` forward, vision tower, multimodal RoPE, and loss
    path;
  - ms-swift Qwen/Qwen3-VL template and multimodal input preparation;
  - current CoordExp custom trainer/loss/sidecar injection points.

Key constraints:

- Qwen3-VL is not a plain text causal LM wrapper.
  - `Qwen3VLModel` owns the visual tower, language model, and `rope_deltas`.
  - `Qwen3VLModel.forward` replaces image/video placeholder embeddings with
    visual features and validates placeholder-feature count agreement.
  - Vision features come from grid-aware visual processing and DeepStack visual
    features are injected again in the text decoder.
  - `get_rope_index` constructs multimodal 3-axis position ids from text,
    image/video placeholder positions, and `image_grid_thw` /
    `video_grid_thw`.
- Custom CoordExp losses should not alter the Qwen3-VL multimodal pre-forward
  contract.
  - Preserve `input_ids`, `pixel_values`, `pixel_values_videos`,
    `image_grid_thw`, `video_grid_thw`, `attention_mask`, `position_ids`, and
    `cache_position` unless explicitly refactoring multimodal embedding
    semantics.
  - Do not precompute or mutate `inputs_embeds` casually.
  - Do not force custom `position_ids` casually.
- For ordinary multimodal supervised training, preserve full logits.
  - Qwen3-VL supports `logits_to_keep`, but ms-swift disables auto
    `logits_to_keep` for multimodal models.
  - If `logits_to_keep` is forced later, span sidecars must be projected
    through the same label/logit slicing.
- ms-swift loss semantics are shifted next-token semantics.
  - Labels are target-token labels.
  - Built-in per-token CE shifts labels with `torch.roll(labels, shifts=-1)`.
  - A label-position span `[s, e)` corresponds to logits positions `[s - 1,
    e - 1)` except index `0`, which is masked.
  - `loss_scale` is shifted the same way before multiplying per-token losses.
- ms-swift `channel` loss is logging-only in the SFT path.
  - It updates custom metrics from per-token slices but does not change scalar
    loss semantics.
  - CoordExp channel-aware objectives should stay inside CoordExp objective
    modules, not rely on upstream channel logging.
- ms-swift `args.loss_type` / `compute_loss_func` is available but not the
  preferred clean seam for CoordExp's future unified objectives.
  - Stock custom CE can recompute per-token CE and ignore already-scaled
    `outputs.loss` in some combinations.
  - A CoordExp objective runner should be tested explicitly if it enters this
    seam.
- ms-swift templates are not always equivalent to raw Hugging Face
  `tokenizer.apply_chat_template`.
  - The default ms-swift template backend is `swift`.
  - Exact model-ready parity should either use the Swift template encoding path
    or validate constant token/label alignment against it.
- `EncodedDetectionView` must not be text-only.
  - It should either contain or be paired with the true model-input multimodal
    payload and alignment evidence.
  - Span adaptation must not mutate visual grids, MRoPE positions, packing
    state, or trainer-only model-input keys.
- Current CoordExp already has useful boundaries.
  - Stage-1 batch extras are explicit `BatchExtras` and must not be forwarded
    to the model.
  - Latest detection sidecars are stripped by a registered model-input
    boundary.
  - Recursive detection and Stage-2 already own model-forward/logits-based loss
    computation and are safer first candidates for a unified objective result
    contract than additive Stage-1 MRO mixins.
  - Stage-1 recursive detection emits typed `MetricEvent`; Stage-2 still emits
    dynamic flat metric maps, so metric unification remains a design fork.

Implication for the next decision:

- The future `ObjectiveModule` boundary should be designed around:
  - normal model forward through ms-swift / Qwen3-VL;
  - full unsliced `outputs.logits` unless explicitly projected;
  - shifted-logit coordinates;
  - typed batch extras / supervision spans;
  - no access to or mutation of Qwen3-VL multimodal embedding/position inputs.

### Decision 27: Separate Model Forward From Objective Modules

Decision:

- Model forward and objective computation must be separate architecture
  boundaries.
- `ObjectiveModule` should not call `model(**inputs)` directly.
- A trainer-facing bridge should own model-input preparation, sidecar stripping,
  ms-swift/Qwen3-VL forward context, full-logit preservation, and forward
  provenance.
- `ObjectiveModule` should consume only post-forward outputs, supervision
  spans, target distributions, and an objective context.

Approved flow:

```text
TrainerLossBridge
  -> prepare model inputs
  -> strip sidecars
  -> call model once under ms-swift / Qwen3-VL context
  -> ModelForwardResult
  -> ObjectiveRunner
  -> ObjectiveModule(s)
  -> ObjectiveResult
```

Conceptual result object:

```python
@dataclass(frozen=True)
class ModelForwardResult:
    logits: Tensor
    outputs: ModelOutput
    labels: Tensor | None
    model_input_keys: tuple[str, ...]
    num_items_in_batch: int | None
    provenance: ForwardProvenance
```

`TrainerLossBridge` owns:

- preserving the ms-swift / Qwen3-VL model-input contract;
- `template.forward_context(...)`;
- sidecar stripping;
- labels handling at the model boundary;
- preserving full logits unless an explicit projection layer is introduced;
- `num_items_in_batch` and distributed-loss accounting inputs;
- forward provenance and model-input key auditing.

`ObjectiveModule` owns:

- reading `ModelForwardResult.logits`;
- consuming `SupervisionSpan` and `TargetDistribution`;
- computing objective-specific loss terms;
- emitting typed metric and diagnostic events;
- reducing objective-local losses according to resolved objective policy.

`ObjectiveModule` must not access or mutate:

- raw model inputs;
- `input_ids`;
- `pixel_values`;
- `pixel_values_videos`;
- `image_grid_thw`;
- `video_grid_thw`;
- `position_ids`;
- `inputs_embeds`;
- Qwen3-VL visual placeholder logic;
- Qwen3-VL RoPE / `rope_deltas`;
- ms-swift template internals;
- raw YAML dictionaries;
- trainer/model handles.

Rationale:

- Qwen3-VL owns fragile multimodal embedding replacement, DeepStack visual
  injection, and multimodal RoPE.
- ms-swift owns important trainer/template context around forward execution.
- The future objective layer should be a logits-and-spans layer, not a model
  execution layer.
- Recursive detection and Stage-2 already approximate this safer boundary and
  are natural first migration candidates.

### Decision 28: Keep Raw Logits And Use PredictionCoordinateMapper

Decision:

- `ModelForwardResult.logits` should preserve raw model output coordinates.
- Do not silently shift, slice, or normalize logits inside
  `ModelForwardResult`.
- Use an explicit `PredictionCoordinateMapper` to map label/token positions to
  logits positions.

Approved boundary:

```text
ModelForwardResult.logits
  raw model output shape and positions

PredictionCoordinateMapper
  label position -> logits position
  span token positions -> prediction positions
  logits_to_keep / packing projection when explicitly enabled
```

Normal causal LM mapping:

```text
label position p -> logit position p - 1
```

Design rules:

- `ModelForwardResult` records what the model actually returned.
- `SupervisionSpan.prediction_positions` should be explicit and compatible
  with the mapper.
- `ObjectiveModule` should use the mapper or already-mapped span positions,
  never implicit causal-shift assumptions.
- If future training enables `logits_to_keep`, packing, or sidecar offset
  rewriting, that projection belongs in the mapper / segment-map boundary.

Rationale:

- Raw logits are easier to debug and compare against upstream ms-swift/HF
  behavior.
- Silent shifting would hide the model-output contract.
- The mapper centralizes the fragile off-by-one, `logits_to_keep`, and packing
  coordinate logic.

### Decision 29: TrainerLossBridge Constructs PredictionCoordinateMapper

Decision:

- `TrainerLossBridge` should construct `PredictionCoordinateMapper`.
- `SupervisionSpan` carries logical prediction positions derived from
  `EncodedDetectionView`.
- `ObjectiveRunner` validates and maps those positions against actual raw logits
  before objective computation.

Approved flow:

```text
DetectionTemplateCodec
  -> EncodedDetectionView
  -> SpanAdapter
  -> SupervisionSpan(prediction_positions in logical sequence coordinates)

TrainerLossBridge
  -> model forward
  -> PredictionCoordinateMapper(actual logits projection)
  -> ModelForwardResult

ObjectiveRunner
  -> validate/map span prediction positions
  -> ObjectiveModule
```

`PredictionCoordinateMapper` depends on bridge-level facts:

- labels shape;
- raw logits shape;
- causal LM shift;
- `logits_to_keep` status;
- packing / padding-free status;
- `text_position_ids`;
- `cu_seq_lens`;
- segment maps;
- model-input provenance.

Rationale:

- `SpanAdapter` is template-aware but should not know trainer runtime details.
- `ObjectiveModule` should not guess causal shift, packing projection, or
  logits slicing.
- The bridge is the layer that has enough information to validate the model
  output against logical supervision positions.

### Decision 30: ObjectiveRunner Materializes ObjectiveBatch

Decision:

- `ObjectiveRunner` should handle coordinate mapping and tensor gathering.
- `ObjectiveModule` should receive objective-ready `ObjectiveBatch` tensors by
  default.
- Objective modules should not each reimplement causal-shift, packing,
  `logits_to_keep`, or span-position mapping logic.

Approved flow:

```text
ObjectiveRunner
  -> group spans by objective / distribution kind
  -> validate positions through PredictionCoordinateMapper
  -> gather logits, targets, masks, and weights
  -> ObjectiveBatch
  -> ObjectiveModule.compute(...)
```

Conceptual module API:

```python
class ObjectiveModule(Protocol):
    id: str
    supported_distribution_kinds: frozenset[str]

    def compute(
        self,
        batch: ObjectiveBatch,
        context: ObjectiveContext,
    ) -> ObjectiveResult:
        ...
```

Conceptual batch:

```python
@dataclass(frozen=True)
class ObjectiveBatch:
    logits: Tensor
    target: TargetDistributionBatch
    weights: Tensor
    mask: Tensor
    spans: tuple[SupervisionSpan, ...]
    provenance: ObjectiveBatchProvenance
```

Default objective responsibilities:

- `TokenCEObjective`
  - objective-ready logits + token ids + weights -> CE.
- `TrieCEObjective`
  - objective-ready logits + sparse positive sets + weights -> multi-positive
    CE.
- `CoordSoftCEObjective`
  - objective-ready logits + coordinate distributions + weights -> soft CE.
- `BoxRegressionObjective`
  - prepared coordinate/logit-derived values + bbox targets -> regression.

Exception:

- Future advanced objectives may request raw spans / mapper through an explicit
  advanced interface, but this should not be the default module contract.

Rationale:

- Centralized coordinate mapping prevents repeated off-by-one and projection
  bugs.
- Objective modules stay focused on math rather than trainer/model coordinate
  semantics.
- This design keeps raw logits available for debugging while presenting
  objective-specific tensors for loss computation.

### Decision 31: Standard CE Is An Explicit ObjectiveModule

Decision:

- Standard cross-entropy should be represented as an explicit
  `TokenCEObjective`.
- It should run through `ObjectiveRunner` like trie CE, coordinate soft CE, and
  regression objectives.
- The canonical future architecture should not rely on implicit model / HF /
  ms-swift CE as a hidden base loss.

Canonical objective stack:

```text
ObjectiveRunner
  -> TokenCEObjective
  -> TrieCEObjective
  -> CoordSoftCEObjective
  -> BoxRegressionObjective
```

Plain SFT target config shape:

```yaml
objectives:
  - id: token_ce
    weight: 1.0
```

Compact trie target config shape:

```yaml
objectives:
  - id: trie_ce
    weight: 1.0
  - id: coord_soft_ce
    enabled: false
```

Mixed-objective target config shape:

```yaml
objectives:
  - id: token_ce
    spans: [description_text, schema_token]
    weight: 1.0
  - id: trie_ce
    spans: [object_entry]
    weight: 1.0
  - id: box_regression
    spans: [coordinate_slot]
    weight: 0.1
```

Bridge consequence:

- When `ObjectiveRunner` owns the canonical loss, the bridge should generally
  remove `labels` before model forward so Qwen3-VL / HF does not compute an
  implicit CE loss that is ignored or double-counted.
- The model still receives all multimodal inputs; only scalar loss ownership is
  moved into CoordExp's objective runner.

Migration caveat:

- Legacy ordinary SFT paths may temporarily keep implicit ms-swift CE during
  migration.
- The target architecture should still treat CE as explicit and componentized.

Rationale:

- One loss-accounting path is easier to reason about than hidden base loss plus
  additive extras.
- Objective-level logging becomes consistent.
- Stage-1 and Stage-2 share the same loss composition mechanism.
- Denominators, weights, and masks become explicit and auditable.

### Decision 32: Objective-Local Normalization With Explicit Precision Policy

Decision:

- Each objective should compute and report its own normalized loss.
- `ObjectiveRunner` should combine already-normalized objective losses using
  configured objective weights.
- Do not force CE tokens, trie branch decisions, coordinate slots, and bbox
  regression targets under one global denominator.
- Each objective must declare and follow an explicit numeric precision policy.

Conceptual result:

```python
@dataclass(frozen=True)
class ObjectiveResult:
    objective_id: str
    loss: Tensor
    weighted_loss: Tensor
    numerator: Tensor
    denominator: Tensor
    weight: float
    precision_policy: ObjectivePrecisionPolicy
    metric_events: tuple[MetricEvent, ...]
    diagnostic_events: tuple[DiagnosticEvent, ...]
```

Combination:

```text
total_loss = sum(result.weighted_loss for result in objective_results)
```

Objective denominator examples:

- `token_ce`
  - supervised token count or weighted token mass.
- `trie_ce`
  - branch decision count or weighted support mass.
- `coord_soft_ce`
  - coordinate-slot count or weighted coordinate mass.
- `box_regression`
  - object count or coordinate-slot count.

Monitoring keys should include:

- `loss/<objective>`;
- `loss/<objective>_weighted`;
- `denom/<objective>`;
- `weight/<objective>`;
- `loss/total`.

Precision policy:

- Objective modules must not blindly inherit bf16 for all math.
- Numerically sensitive operations should upcast to `float32` when needed.
- The policy should be explicit per objective and visible in diagnostics or
  resolved config.

Initial precision guidance:

- `token_ce`
  - may follow PyTorch / ms-swift practice of upcasting logits to `float32`
    before CE when needed.
- `trie_ce`
  - should use `float32` for log-softmax / log-sum-exp / sparse
    multi-positive aggregation unless a validated lower-precision path exists.
- `coord_soft_ce`
  - should use `float32` for soft target probabilities, normalization, and
    distribution diagnostics.
- `box_regression`
  - should use `float32` for geometry, IoU-style math, CIoU/GIoU-like terms,
    and small-denominator divisions.
- final weighted loss
  - may be cast back only according to the trainer/autocast policy after the
    objective has computed stable normalized values.

Rationale:

- Different objectives have different natural units and denominators.
- Objective-local normalization makes loss weights meaningful and auditable.
- Precision mistakes can silently destabilize trie CE, coordinate
  distributions, and regression math.
- Explicit precision policy keeps mixed-precision training compatible without
  sacrificing numerically fragile computations.

### Decision 33: Precision Policy Is Per-Objective With Safe Defaults

Decision:

- Precision policy should be per objective.
- Each objective should provide safe defaults.
- Resolved precision policy should be emitted in run artifacts and/or
  diagnostics.
- Explicit overrides are allowed when needed, but default configs should remain
  concise.

Concise config:

```yaml
objectives:
  - id: trie_ce
    weight: 1.0
```

Resolved policy example:

```json
{
  "id": "trie_ce",
  "precision": {
    "logits": "float32",
    "target_distribution": "float32",
    "reduction": "float32",
    "final_loss": "trainer_default"
  }
}
```

Explicit override example:

```yaml
objectives:
  - id: trie_ce
    weight: 1.0
    precision:
      logits: float32
      target_distribution: float32
      reduction: float32
      final_loss: trainer_default
```

Rationale:

- A global precision policy is too blunt for mixed CE, trie, coordinate, and
  regression objectives.
- Hardcoding hides reproducibility-relevant behavior.
- Safe defaults keep configs readable while still documenting resolved numeric
  behavior.
- Diagnostics can detect accidental bf16-only math in numerically sensitive
  objective paths.

### Decision 34: MetricEvent Is The Canonical Metric Surface

Decision:

- Objective and diagnostic metrics should move toward typed `MetricEvent`
  emission everywhere, including Stage-2.
- Stage-2 dynamic flat metric maps may remain temporarily through a
  compatibility adapter during migration.
- New objective modules and new monitoring/diagnostic modules should emit typed
  events directly.

Target flow:

```text
ObjectiveModule
  -> ObjectiveResult.metric_events
  -> MetricEvent
  -> MetricRegistry / flatten_metric_events
  -> SwiftMetricReporter
```

Stage-2 migration bridge:

```text
Stage2FlatMetricAdapter
  dynamic flat metrics -> MetricEvent where mapping is known
  unmapped legacy metrics -> legacy / flat diagnostic event
```

Rules:

- typed events are canonical for new code;
- flat maps are tolerated only at compatibility boundaries;
- removed mechanisms must not emit new metric keys;
- historical readers may still understand old keys and label them legacy /
  removed;
- metric reducers, units, denominators, span roles, token types, stage,
  channel, and objective identity should be explicit event fields rather than
  inferred from string-key patterns.

Rationale:

- Metric-key string heuristics are fragile, especially under DDP/global
  reduction.
- Typed events make loss components, span stats, trie stats, coordinate
  diagnostics, assignment diagnostics, and duplicate diagnostics auditable.
- This matches the earlier clean-write / tolerant-read decision for removed
  mechanisms and historical artifacts.

### Decision 35: Separate MetricEvent And DiagnosticEvent Streams

Decision:

- Scalar or reduced training/eval quantities should use `MetricEvent`.
- Structured debug, provenance, sampled records, artifact-linked payloads, and
  verbose traces should use a separate `DiagnosticEvent`.
- Both event streams should be coordinated by `ObservabilityService`.

`MetricEvent` examples:

- `loss/token_ce`;
- `loss/trie_ce`;
- `denom/trie_ce`;
- `span_count/coordinate_slot`;
- `trie/support_size_mean`;
- `coord/token_acc`;
- `stage2/assignment_iou_mean`;
- `duplicate/filter_drop_rate`.

`DiagnosticEvent` examples:

- supervision-plan snapshot;
- duplicate cluster members;
- assignment pair list;
- ignored prediction reason list;
- rollout parse failure record;
- EOS/continue probe trace;
- coordinate regression outlier sample;
- raw decoded text reference.

Unified owner:

```text
ObservabilityService
  -> MetricEvent sink
  -> DiagnosticEvent sink
  -> artifact writers
  -> legacy adapters
```

Rationale:

- Metrics need reducer semantics, DDP behavior, scalar logging, stable keys, and
  compact training curves.
- Diagnostics need structured payloads, sampling, JSONL artifacts, provenance
  links, and looser schemas.
- Mixing them would either pollute metric logs or make diagnostics too weak.
- A shared observability owner keeps lifecycle, enablement, artifact paths, and
  legacy compatibility centralized.

### Decision 36: Diagnostic Artifacts Use Profile-Based Bounded Sampling

Decision:

- Diagnostic artifact emission should be profile-based.
- Default level should be `standard`.
- Standard diagnostics should be bounded by per-step and per-run sample limits.
- Debug diagnostics may emit richer traces for smoke/debug runs.

Conceptual config:

```yaml
observability:
  diagnostics:
    level: standard   # off | standard | debug
```

Level semantics:

- `off`
  - scalar `MetricEvent`s only;
  - no structured diagnostic JSONL except hard failure records.
- `standard`
  - bounded diagnostic samples;
  - summary artifacts;
  - enough for normal training diagnosis;
  - safe default for Stage-1 and Stage-2.
- `debug`
  - richer per-example traces;
  - supervision-plan snapshots;
  - assignment lists;
  - duplicate clusters;
  - rollout parse records;
  - intended for smoke/debug runs, not full production by default.

Default bounded policy:

```yaml
observability:
  diagnostics:
    level: standard
    sample_per_step: 8
    max_records_per_run: 5000
    emit_supervision_plan_samples: true
    emit_assignment_samples: true
    emit_duplicate_samples: true
    emit_raw_rollout_samples: false
```

Rationale:

- Stage-2 rollouts can generate large diagnostic payloads.
- Full traces can slow training and inflate artifacts.
- Diagnostics off by default would make failures hard to diagnose.
- Bounded standard diagnostics provides a black-box recorder without turning the
  output directory into a landfill.

### Decision 37: Components Receive Typed Runtime Plans, Not Raw Config

Decision:

- Components should never read raw YAML/config dictionaries directly.
- Config loading and schema validation should happen once.
- `TrainingSurfaceResolver` should build typed runtime plans before training.
- Components receive typed config objects only.

Approved flow:

```text
YAML config
  -> ConfigLoader / schema validation
  -> TrainingSurfaceResolver
  -> ResolvedTrainingRun
  -> TrainingPipeline
  -> typed component configs
```

Key runtime plans:

- `ResolvedTrainingRun`;
- `ObjectivePlan`;
- `ObservabilityPlan`;
- `TemplatePlan`;
- `DatasetPlan`;
- `AssignmentPlan`;
- `DuplicateFilteringPlan`;
- `PrecisionPlan`;
- `ArtifactPlan`.

Component construction examples:

```python
TokenCEObjective(config: TokenCEObjectiveConfig)
TrieCEObjective(config: TrieCEObjectiveConfig)
GreedyIoUAssignment(config: GreedyIoUAssignmentConfig)
ObservabilityService(config: ObservabilityPlan)
```

Rejected pattern:

```python
class TrieCEObjective:
    def compute(...):
        weight = raw_config["custom"]["extra"]["trie"]["weight"]
```

Rationale:

- Prevents `custom.extra` from reappearing under a new name.
- Makes config inheritance and list-replacement issues visible at resolve time.
- Enables config-lint tests before training starts.
- Makes run artifacts record the actual resolved behavior.
- Keeps components testable without full repo config.

### Decision 38: Objective Config Uses Keyed Profiles And Resolves To Ordered List

Decision:

- Objective authoring config should move toward named profiles with keyed,
  patchable entries.
- Runtime execution should still receive a deterministic ordered objective list.
- The config hierarchy and knobs must be designed deliberately, not merely
  converted from YAML lists to dictionaries.

Authoring shape:

```yaml
objective_profile: stage1_compact_trie

objectives:
  trie_ce:
    enabled: true
    weight: 1.0
  token_ce:
    enabled: true
    spans: [description_text, schema_token]
    weight: 1.0
  coord_soft_ce:
    enabled: false
```

Resolved runtime shape:

```json
{
  "objectives": [
    {"id": "token_ce", "weight": 1.0, "order": 10},
    {"id": "trie_ce", "weight": 1.0, "order": 20}
  ]
}
```

Rules:

- keyed objective entries are the authoring surface;
- `enabled: false` disables an objective without deleting sibling entries;
- the resolver validates incompatible objective combinations;
- the resolver assigns deterministic execution order;
- the resolved config records the final ordered objective list;
- config knobs should belong to the narrowest relevant hierarchy;
- avoid dumping experimental knobs into broad `custom` or objective-wide
  `extra` maps.

Rationale:

- YAML list replacement is dangerous for objective ablations because changing
  one inherited objective can silently drop sibling objectives.
- Keyed entries make inheritance and ablation safer.
- Ordered runtime output preserves deterministic execution.
- A designed hierarchy keeps the objective config from becoming another
  unstructured experimental attic.

### Decision 39: Use A Small Explicit Top-Level Config Hierarchy

Decision:

- The rebuilt training architecture should use a small set of top-level config
  domains with clear ownership.
- Avoid using broad `custom.extra` or objective-wide `extra` maps as hiding
  places for research semantics.
- Keep research semantics out of `runtime`.
- Keep execution mechanics out of `objectives`.

Approved top-level domains:

- `run`;
- `surface`;
- `data`;
- `template`;
- `supervision`;
- `objectives`;
- `observability`;
- `artifacts`;
- `runtime`.

Conceptual shape:

```yaml
run:
  name: ...
  output_dir: ...
  seed: ...

surface:
  id: stage1_compact_trie

data:
  train_jsonl: ...
  val_jsonl: ...
  geometry:
    do_resize: false
  packing:
    enabled: false

template:
  id: compact_full
  coordinate_surface: coord_token
  bbox_format: xyxy

supervision:
  builder: stage1_compact_trie
  stage1:
    object_ordering: ...
  stage2:
    channel_policy: ...
    assignment:
      strategy: greedy_iou
    duplicate_filtering:
      enabled: true

objectives:
  profile: stage1_compact_trie
  entries:
    token_ce:
      enabled: true
      weight: 1.0
    trie_ce:
      enabled: true
      weight: 1.0
    coord_soft_ce:
      enabled: false

observability:
  metrics:
    typed_events: true
  diagnostics:
    level: standard
    sample_per_step: 8
    max_records_per_run: 5000

artifacts:
  manifest: true
  resolved_config: true
  supervision_plan_samples: true

runtime:
  trainer_backend: ms_swift
  precision:
    default: bf16
  distributed: ...
```

Ownership rule:

- `surface`
  - chooses the training pipeline.
- `supervision`
  - builds semantic `SupervisionPlan` objects.
- `template`
  - renders, tokenizes, and encodes.
- `objectives`
  - computes loss from spans and target distributions.
- `observability`
  - owns metrics and diagnostics.
- `runtime`
  - owns trainer backend, distributed execution, and systems-level precision
    defaults, not research objective semantics.

Rationale:

- The hierarchy separates research semantics from execution mechanics.
- It makes config ownership inspectable.
- It reduces accidental coupling between data/template/objective/runtime
  settings.
- It gives future schema validation a stable map.

### Decision 40: Shared Config Hierarchy With Surface-Specific Schemas

Decision:

- Stage-1 and Stage-2 should share the same top-level config hierarchy.
- Each supported training surface should have a separate surface-specific schema
  selected by `surface.id`.
- Avoid one giant schema with many optional sections and conditional ignores.

Conceptual schema family:

```text
BaseTrainingConfig
  run
  surface
  data
  template
  objectives
  observability
  artifacts
  runtime

Stage1JsonCeConfig(BaseTrainingConfig)
  supervision.stage1_json
  objective constraints

Stage1CompactTrieConfig(BaseTrainingConfig)
  supervision.stage1_compact_trie
  trie objective constraints
  compact_full template constraints

Stage2TwoChannelConfig(BaseTrainingConfig)
  supervision.stage2_two_channel
  assignment / duplicate filtering constraints
  rollout diagnostics constraints
```

Resolver flow:

```text
surface.id
  -> choose surface schema
  -> validate required sections
  -> reject irrelevant/incompatible sections
  -> produce ResolvedTrainingRun
```

Rationale:

- Shared top-level domains keep config layout familiar across stages.
- Surface-specific schemas prevent invalid Stage-1 / Stage-2 combinations from
  hiding behind optional fields.
- The resolver can produce clearer errors than a huge conditional schema.
- This approach keeps extensibility for future research surfaces without
  weakening validation for current surfaces.

### Decision 41: Canonical Configs Are Strict; Experimental Knobs Require Opt-In

Decision:

- Canonical surface schemas should be strict and fail fast.
- Unknown or irrelevant config keys should not be silently ignored.
- Removed mechanism keys should fail with actionable removal notes.
- Temporary research knobs must live under an explicit `experimental` block and
  require pipeline opt-in before being consumed.

Canonical behavior:

- unknown key -> fail fast;
- irrelevant Stage-2 key in Stage-1 surface -> fail fast;
- irrelevant Stage-1 key in Stage-2 surface -> fail fast;
- removed mechanism key -> fail fast with removal rationale;
- disabled objective with active risky subconfig -> warning or fail depending
  on risk.

Experimental block shape:

```yaml
experimental:
  owner: xiaoyan
  expires_after: 2026-06-15
  notes: "Temporary probe for ..."
  knobs:
    ...
```

Rules:

- core components must not read `experimental` silently;
- a pipeline must explicitly opt into a named experimental block;
- experimental blocks should be excluded from canonical docs unless promoted;
- expired experimental blocks should fail or warn during config lint;
- experiment artifacts should record any active experimental block.

Rationale:

- Strict validation prevents dead config from pretending it matters.
- Explicit experimental namespaces preserve research agility without poisoning
  canonical configs.
- Removed mechanisms such as duplicate-negative training and adjacent repulsion
  should not be reintroduced by stale YAML.
- The resolver should behave like a contract gate, not a suggestion box.

### Decision 42: compact_full Is The Canonical Default For Stage-1 And Stage-2

Decision:

- `compact_full` should be the canonical default template for both Stage-1 and
  Stage-2 detection/object training.
- Stage-2 should use the same compact object-entry structure that the project
  wants at inference time.
- Template codec remains selectable for explicit baselines and ablations.

Target defaults:

```yaml
surface:
  id: stage1_compact_trie

template:
  id: compact_full
```

```yaml
surface:
  id: stage2_two_channel

template:
  id: compact_full
```

Explicit baseline:

```yaml
surface:
  id: stage1_json_ce

template:
  id: stage1_json_pretty
```

Rules:

- `compact_full`
  - canonical default for detection/object training in both stages.
- `stage1_json_pretty`
  - explicit baseline/debug template.
- other compact helper formats
  - compatibility helpers only, not canonical training surfaces.

Rationale:

- The rebuilt architecture centers explicit object entries, schema tokens,
  description spans, and coordinate slots.
- Stage-2 should not preserve avoidable train/eval template mismatch.
- Keeping the codec selectable preserves ablation and baseline flexibility
  without weakening the canonical path.

### Decision 43: Stage-2 Channels Share compact_full Codec And Mostly Share Span Adapter

Decision:

- Stage-2 Channel-A and Channel-B should use the same
  `CompactFullTemplateCodec`.
- Channel-A and Channel-B should have separate `SupervisionPlanner`s because
  their semantic supervision sources differ.
- Once they produce compact-full `SupervisionPlan`s, they should use a shared
  compact span adapter by default.
- Any channel-specific behavior in span adaptation should be narrow and
  explicit.

Approved flow:

```text
Stage2ChannelASupervisionPlanner
  -> Stage2ChannelASupervisionPlan
  -> CompactFullTemplateCodec
  -> EncodedDetectionView
  -> Stage2CompactSpanAdapter

Stage2ChannelBSupervisionPlanner
  -> Stage2ChannelBSupervisionPlan
  -> CompactFullTemplateCodec
  -> EncodedDetectionView
  -> Stage2CompactSpanAdapter
```

Channel planner semantics:

- Channel-A
  - GT-anchored teacher-forced supervision.
- Channel-B
  - rollout survivors;
  - false-negative insertions;
  - ignored predictions;
  - assignment provenance;
  - duplicate-filter provenance.

Allowed narrow channel-specific span policies:

- span weight policy;
- supervised span selection;
- diagnostic provenance attachment.

Rationale:

- Compact-full rendered syntax should be identical across Stage-2 channels.
- Channel-specific rollout complexity belongs in supervision planning, not in
  template rendering or objective modules.
- Sharing the codec and span adapter reduces train/eval/template drift while
  keeping Channel-B semantics explicit.

### Decision 44: Share CompactFullSpanProjector, Keep Stage-Specific SpanAdapters

Decision:

- Stage-1 compact trie CE and Stage-2 compact training should share a lower
  level `CompactFullSpanProjector`.
- Stage-1 and Stage-2 should still use separate stage-specific `SpanAdapter`s.

Approved structure:

```text
CompactFullSpanProjector
  common token-role / object-entry / coordinate-slot projection utilities

Stage1CompactTrieSpanAdapter
  uses CompactFullSpanProjector
  emits trie CE spans and optional coordinate / regression spans

Stage2CompactSpanAdapter
  uses CompactFullSpanProjector
  emits Channel-A / Channel-B spans plus Stage-2 provenance
```

Shared projector owns:

- object entry spans;
- schema token spans;
- description spans;
- coordinate slots;
- prediction positions;
- stop positions;
- compact-full token-role projection.

Stage-specific adapters own:

- Stage-1
  - full GT object-set learning;
  - entry-trie multi-positive CE;
  - object ordering / permutation semantics.
- Stage-2
  - rollout-conditioned accepted objects;
  - false-negative insertion;
  - assignment provenance;
  - duplicate-filter provenance;
  - channel identity.

Rationale:

- One universal compact adapter would conflate Stage-1 and Stage-2 supervision
  semantics.
- Duplicating compact-full projection would recreate token-role drift.
- The projector/adapters split gives reusable syntax mechanics without erasing
  stage-specific research meaning.

### Decision 45: Assignment Is A Separate Stage-2 Component

Decision:

- Stage-2 assignment should be a separate component after duplicate filtering
  and before supervision planning.
- `AssignmentStrategy` owns the pairing algorithm.
- `AssignmentResult` owns pairing evidence and diagnostics.
- Stage-2 supervision planners consume assignment results rather than
  implementing assignment internally.

Approved flow:

```text
RolloutViews
  -> DuplicateFilter
  -> DuplicateFilterResult
  -> AssignmentStrategy
  -> AssignmentResult
  -> Stage2ChannelBSupervisionPlanner
  -> Stage2ChannelBSupervisionPlan
```

Assignment strategies:

- `GreedyIoUAssignment`
  - canonical future default.
- `LegacyHungarianAssignment`
  - migration-only compatibility path, not canonical default.

`AssignmentResult` owns:

- prediction-to-ground-truth matches;
- unmatched predictions;
- unmatched ground-truth objects;
- IoU scores;
- reason codes;
- assignment diagnostics.

`Stage2ChannelBSupervisionPlanner` owns:

- accepted matched rollout objects;
- false-negative insertions;
- ignored predictions;
- final object ordering;
- semantic supervision intent.

Rationale:

- Assignment is evidence used by supervision planning, not supervision planning
  itself.
- Keeping assignment separate makes greedy IoU and future Stage-2 pairing ideas
  replaceable.
- This prevents Channel-B construction from becoming coupled to one assignment
  algorithm.

### Decision 46: Duplicate Filtering Runs Before Assignment

Decision:

- Stage-2 duplicate filtering should run before assignment.
- `AssignmentStrategy` should pair ground-truth objects against accepted
  duplicate survivors, not against the raw predicted object list.
- `DuplicateFilter` should emit survivor / ignored-prediction diagnostics before
  target realization.

Canonical flow:

```text
RolloutViews
  -> DuplicateFilter
  -> DuplicateFilterResult
  -> accepted survivors
  -> AssignmentStrategy
  -> AssignmentResult
  -> Stage2ChannelBSupervisionPlanner
```

Rationale:

- Duplicate filtering is a train-side target-construction policy, not an
  assignment algorithm.
- Assignment over raw duplicate predictions can leak duplicate objects into the
  target-realization path and make the later positive sequence ambiguous.
- Assignment over accepted survivors keeps greedy-IoU pairing, FN insertion, and
  span construction aligned with the exact object set that can enter the
  forward pass.
- Diagnostics still report duplicate clusters and survivor reasons before
  assignment, while assignment diagnostics report pairings for the filtered
  survivor set.

### Decision 47: Duplicate Filtering Is Deterministic In The Canonical Path

Decision:

- Canonical duplicate filtering should be deterministic and rule-based.
- Stochastic or learned survivor selection may only be introduced later as an
  explicit experimental strategy.
- Default duplicate filtering must use stable tie-breakers for reproducibility.

Canonical inputs:

- `RolloutViews`;
- duplicate thresholds;
- confidence scores when available;
- stable object ordering.

Canonical outputs:

- survivor predictions;
- dropped predictions;
- reason codes;
- diagnostics.

Default survivor priority:

1. higher confidence if available;
2. better duplicate-cluster representative under the configured geometric
   policy;
3. earlier / stable order tie-breaker.

Canonical config:

```yaml
supervision:
  stage2:
    duplicate_filtering:
      strategy: deterministic_iou_cluster
```

Experimental-only future variant:

```yaml
experimental:
  knobs:
    duplicate_filtering:
      strategy: stochastic_survivor_probe
```

Rationale:

- Reproducible Stage-2 training examples are more important than clever survivor
  sampling in the canonical path.
- Deterministic diagnostics are easier to interpret and regression-test.
- Stochastic or learned duplicate handling would be a research idea, not an
  infrastructure default.

### Decision 48: Stage-2 Channel-B Planner Owns False-Negative Insertion

Decision:

- False-negative insertion should be owned by
  `Stage2ChannelBSupervisionPlanner`.
- Assignment identifies unmatched ground-truth objects.
- Duplicate filtering identifies accepted rollout survivors and ignored
  predictions.
- The Channel-B supervision planner builds the final compact-full supervised
  object sequence.
- Object ordering should be an explicit policy.

Approved flow:

```text
AssignmentResult
  -> unmatched ground-truth objects

DuplicateFilterResult
  -> accepted rollout survivors
  -> ignored predictions

Stage2ChannelBSupervisionPlanner
  -> accepted rollout survivors
  -> false-negative insertions
  -> explicit object ordering policy
  -> Stage2ChannelBSupervisionPlan
```

Initial ordering policies:

- `sorted`
  - explicit canonical-order option;
  - merge accepted survivors and false-negative objects, then apply canonical
    object ordering.
- `tail_append`
  - current default compatibility mode;
  - preserve accepted rollout survivor order and append false negatives at the
    tail.
- `seeded_random_shuffle`
  - explicit permutation / augmentation option;
  - must be seeded, recorded in provenance, and controlled by config;
  - never an implicit side effect.

Rationale:

- False-negative insertion is semantic supervision construction, not assignment
  or duplicate filtering.
- `tail_append` remains the current default because existing Stage-2 checkpoints
  and configs were authored around clean-prefix plus FN-tail construction.
- `sorted` may become a future canonical default only after an explicit
  migration decision and validation.
- `sorted` keeps Stage-2 targets closer to canonical Stage-1 / inference object
  ordering when it is deliberately enabled.
- Random shuffling can be useful for permutation robustness or multiple-positive
  research, but it must be explicit, reproducible, and visible in diagnostics.

### Decision 49: ObjectOrderingStrategy Is Shared Across Stage-1 And Stage-2

Decision:

- Object ordering should be a shared `ObjectOrderingStrategy` component.
- Stage-1 and Stage-2 supervision planners should use the same ordering
  mechanism.
- Nontrivial or random ordering policies must emit provenance.

Initial strategies:

- `sorted`
  - future canonical target after migration approval.
- `seeded_random_shuffle`
  - explicit permutation / augmentation option.
- `tail_append_legacy`
  - current Stage-2 compatibility/default mode until a separate migration
    promotes another ordering policy.
- `source_order`
  - preserve source/dataset order when explicitly requested.

Usage:

```text
Stage1CompactTrieSupervisionPlanner
  -> ObjectOrderingStrategy
  -> SupervisionPlan

Stage2ChannelBSupervisionPlanner
  -> ObjectOrderingStrategy
  -> SupervisionPlan
```

Required provenance for nontrivial ordering:

```json
{
  "strategy": "seeded_random_shuffle",
  "seed": 12345,
  "input_object_ids": ["a", "b", "c"],
  "output_object_ids": ["b", "c", "a"]
}
```

Rationale:

- Stage-1 compact trie CE and Stage-2 Channel-B both care about object ordering.
- Duplicated ordering logic would create Stage-1 / Stage-2 drift.
- Shared ordering makes permutation augmentation, sorted canonical ordering, and
  legacy tail behavior explicit and testable.

### Decision 50: Stage-2 Object Ordering Runs After Filtering And FN Insertion

Decision:

- Stage-2 Channel-B object ordering should run after duplicate filtering and
  false-negative insertion.
- Ordering applies to the final semantic target object set, not the raw rollout
  object set.

Canonical Channel-B flow:

```text
RolloutViews
  -> DuplicateFilter
  -> AssignmentStrategy over accepted survivors
  -> Stage2ChannelBSupervisionPlanner
       -> collect accepted survivors
       -> add false-negative objects
       -> ObjectOrderingStrategy
       -> Stage2ChannelBSupervisionPlan
```

Rationale:

- Duplicate filtering decides which rollout predictions survive.
- False-negative insertion adds missing ground-truth objects.
- Only after those steps is the final target object set known.
- Ordering before filtering/insertion would produce stale or misleading
  ordering provenance.

### Decision 51: SupervisionPlan Is Semantic-Only, Not Rendered Text

Decision:

- `SupervisionPlan` should contain semantic object entries, ordering,
  supervision intent, and provenance.
- It should not contain full rendered assistant text.
- Rendered assistant bytes belong to `DetectionTemplateCodec` /
  `EncodedDetectionView`.

Approved ownership:

```text
SupervisionPlan
  semantic object entries
  ordering
  stage / channel
  assignment / duplicate / false-negative provenance
  objective intent

DetectionTemplateCodec
  renders compact_full / stage1_json_pretty bytes

EncodedDetectionView
  rendered bytes
  token ids
  spans
  roles
```

Allowed text inside `SupervisionPlan`:

- semantic field values such as object descriptions.

Rejected text inside `SupervisionPlan`:

- full assistant payload;
- rendered compact-full rows;
- rendered JSON payload;
- tokenized prompt strings.

Rationale:

- Rendering is a template responsibility.
- Storing rendered text in `SupervisionPlan` would duplicate template ownership.
- Stage-1 and Stage-2 planners should not format strings differently.
- Semantic planning should remain testable without tokenizer/template code.

### Decision 52: EncodedDetectionView Stores Rendered Text For Diagnostics

Decision:

- `EncodedDetectionView` should store rendered assistant text for diagnostics
  and provenance.
- Token ids, labels, token roles, spans, and prediction positions remain the
  training authority.
- Downstream training logic should not parse rendered text to recover roles or
  objectives.

Conceptual shape:

```python
@dataclass(frozen=True)
class EncodedDetectionView:
    rendered_assistant_text: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    token_roles: tuple[TokenRole, ...]
    object_entries: tuple[EncodedObjectEntry, ...]
    ...
```

Authority split:

```text
training authority:
  token ids
  labels
  token roles
  spans
  prediction positions

diagnostic/provenance view:
  rendered assistant text
```

Rationale:

- Rendered text makes token/span alignment easier to debug.
- Diagnostic artifacts can show the exact assistant payload.
- Parser/eval parity checks can compare rendered text and decoded text.
- Keeping token IDs/spans authoritative prevents string parsing from creeping
  back into loss logic.

### Decision 53: Separate EncodedDetectionView From ModelInputBundle

Decision:

- `EncodedDetectionView` should stay focused on semantic token-role
  representation.
- Exact ms-swift / Qwen3-VL model-forward payload should live in a separate
  `ModelInputBundle`.
- Link them through an `EncodedTrainingExample`.

Approved split:

```text
EncodedDetectionView
  rendered text
  token ids
  labels
  token roles
  object spans
  coordinate slots
  logical prediction positions

ModelInputBundle
  input_ids
  labels
  attention_mask
  pixel_values
  image_grid_thw
  pixel_values_videos
  video_grid_thw
  position_ids
  text_position_ids
  cu_seq_lens
  model-only kwargs
```

Conceptual linkage:

```python
@dataclass(frozen=True)
class EncodedTrainingExample:
    view: EncodedDetectionView
    model_inputs: ModelInputBundle
    sidecars: TrainingSidecars
```

Rationale:

- `EncodedDetectionView` is for template/token-role semantics.
- `ModelInputBundle` is for exact backend/model-forward payload.
- Qwen3-VL/ms-swift inputs include fragile vision/grid/position keys that span
  logic should not casually mutate.
- A separate bundle lets `TrainerLossBridge` own model-forward safety.
- The token view remains testable without GPU/model payload complexity.

### Decision 54: ModelInputBundle Uses A Strict Backend Key Registry

Decision:

- `ModelInputBundle` should enforce a strict backend-specific allowed-key
  registry.
- Arbitrary extra keys should not be forwarded to the model.
- Sidecars are forbidden in `ModelInputBundle` and should live separately.

Conceptual shape:

```python
@dataclass(frozen=True)
class ModelInputBundle:
    tensors: Mapping[str, Tensor]
    backend: Literal["ms_swift_qwen3_vl"]
    allowed_keys: frozenset[str]
    provenance: ModelInputProvenance
```

Allowed key registry example:

```text
ms_swift_qwen3_vl:
  input_ids
  labels
  attention_mask
  pixel_values
  pixel_values_videos
  image_grid_thw
  video_grid_thw
  position_ids
  text_position_ids
  cu_seq_lens_q
  cu_seq_lens_k
  logits_to_keep
  output_router_logits
```

Forbidden sidecar examples:

- `recursive_detection_targets`;
- `supervision_spans`;
- `detection_metadata`;
- `assignment_result`;
- `duplicate_filter_result`;
- `rollout_meta`;
- `metric_events`.

Rationale:

- Prevents accidental forwarding of sidecars into Qwen3-VL.
- Makes the backend/model boundary auditable.
- Lets tests fail before forward when an unknown model-input key appears.
- Allows explicit backend extensions without arbitrary escape hatches.

### Decision 55: TrainingSidecars Are Strict Typed Groups

Decision:

- `TrainingSidecars` should be strict typed dataclasses grouped by purpose.
- Do not use loose arbitrary batch keys as the canonical sidecar mechanism.
- Sidecars must not be forwarded into the model.

Conceptual shape:

```python
@dataclass(frozen=True)
class TrainingSidecars:
    supervision: SupervisionSidecars
    diagnostics: DiagnosticSidecars
    dataset: DatasetSidecars
    stage2: Stage2Sidecars | None
```

Conceptual contents:

```text
SupervisionSidecars:
  supervision_plan_id
  supervision_spans
  encoded_view_ref
  objective_metadata

DiagnosticSidecars:
  supervision_plan_snapshot_ref
  encoding_alignment_report
  parser_manifest
  provenance_links

DatasetSidecars:
  sample_id
  dataset_name
  source_index
  image_path
  original_geometry
  resize_policy

Stage2Sidecars:
  rollout_view_ref
  assignment_result
  duplicate_filter_result
  channel
  ignored_predictions
  rollout_metrics
```

Rules:

- strict dataclasses, not loose dictionaries;
- no tensors that belong in `ModelInputBundle`;
- no raw YAML/config;
- no model/trainer handles;
- sidecars are stripped or consumed before model forward;
- each sidecar group has a clear owner;
- unknown sidecar keys fail before forward.

Rationale:

- Batch extras tend to become ad hoc hidden state unless typed.
- Grouped sidecars give supervision, diagnostics, dataset provenance, and
  Stage-2 metadata clear ownership.
- Strict sidecars prevent accidental model-forward pollution while keeping
  objective/diagnostic data available to the bridge and runner.

### Decision 56: EncodedTrainingExample Is Internal; Dicts Exist At Backend Boundary

Decision:

- Dataset / codec / planner internals should use typed
  `EncodedTrainingExample` objects.
- ms-swift / HF-compatible dictionaries should be produced only at the
  collator/trainer boundary.
- The collator is the adapter from the typed CoordExp world to the backend
  batch-dict world.

Internal dataset output:

```python
EncodedTrainingExample(
    view=EncodedDetectionView(...),
    model_inputs=ModelInputBundle(...),
    sidecars=TrainingSidecars(...),
)
```

Backend boundary output:

```text
dict[str, Tensor | sidecar_payload]
```

Rule:

```text
Dataset / codec / planner:
  typed objects

Collator / TrainerLossBridge:
  backend-compatible dicts with strict model-input and sidecar split
```

Rationale:

- Internal architecture stays typed and readable.
- ms-swift/HF can keep receiving the batch dictionaries they expect.
- Strict validation can happen before converting typed objects into dicts.
- If the backend changes later, only the adapter boundary should need major
  rewriting.

### Decision 57: Packing Is Pending Until PackingSegmentMap Exists

Decision:

- Packing remains disabled for compact-full Stage-1 and Stage-2 supervision in
  the clean architecture's initial implementation.
- Packing enablement is explicitly pending.
- Do not enable packing until `PackingSegmentMap` exists and is covered by
  alignment tests.

Default policy:

```yaml
data:
  packing:
    enabled: false
```

Future enablement requirements:

- `PackingSegmentMap` maps original example positions to packed batch
  positions.
- It rewrites `EncodedDetectionView` spans.
- It rewrites `SupervisionSpan.prediction_positions`.
- It rewrites sidecar references.
- It records segment provenance.
- It validates label/logit alignment after packing.
- It handles ms-swift padding-free / Qwen position behavior explicitly.

Rationale:

- Compact-full supervision is position-sensitive.
- `EncodedDetectionView`, `SupervisionSpan`, `PredictionCoordinateMapper`,
  `ModelInputBundle`, `TrainingSidecars`, object entries, coordinate slots, and
  trie branch positions all depend on exact offsets.
- Enabling packing before exact segment-map rewriting risks silently training
  losses on the wrong positions.

### Decision 58: Encoded-Sample Cache Is Pending Until Fingerprints Are Complete

Decision:

- Encoded-sample caching remains disabled initially for compact-full
  supervision.
- Cache enablement is pending a robust `EncodedSampleFingerprint` contract.
- Do not enable cache reuse for compact-full Stage-1 or Stage-2 until every
  supervision-shaping dependency is represented in the fingerprint.

Default policy:

```yaml
data:
  encoded_sample_cache:
    enabled: false
```

Fingerprint requirements:

- template codec version;
- template schema;
- tokenizer version;
- ms-swift template backend;
- object ordering strategy and seed policy;
- coordinate vocabulary;
- special token rows;
- supervision planner version;
- span adapter version;
- source row hash;
- stage / surface id;
- compact-full codec id;
- Stage-2 assignment / duplicate / FN policy for generated examples when
  applicable.

Fingerprint should exclude:

- volatile diagnostics;
- sampled diagnostic artifact choices;
- non-semantic logging options.

Rationale:

- Cache speed is not worth stale span/sidecar risk.
- Wrong cache hits can silently misalign supervision and logits.
- The cache contract must know all dependencies that shape rendered tokens,
  spans, sidecars, and semantic supervision.

### Decision 59: Implementation Starts With Cleanup, Then Bottom-Up Rebuild

Decision:

- The first implementation wave should be cleanup/deletion first.
- The architecture rebuild should happen bottom-up after cleanup.
- This supersedes any impulse to introduce the full new architecture before
  removing rejected mechanisms.

First cleanup/deletion wave removes:

- duplicate-negative training;
- adjacent repulsion training;
- EOS-loosen / forced-continuation training hooks;
- safe dead shims;
- active docs/configs that advertise removed mechanisms as current behavior.

First cleanup/deletion wave keeps:

- diagnostics/probes;
- duplicate filtering diagnostics;
- posthoc guarded eval as explicit analysis view;
- runnable old Stage-2 / Hungarian path until greedy-IoU replacement smoke
  exists.

Bottom-up rebuild order:

1. `SupervisionPlan`;
2. `EncodedDetectionView`, `ModelInputBundle`, and `EncodedTrainingExample`;
3. `SupervisionSpan` and `TargetDistribution`;
4. `TrainerLossBridge` and `PredictionCoordinateMapper`;
5. `ObjectiveRunner` and `ObjectiveModule`;
6. `ObservabilityService`;
7. Stage-2 assignment / duplicate-filter / supervision-planner stack;
8. `TrainingPipeline` and resolver.

Rationale:

- Cleanup reduces the number of old concepts the new architecture must
  accommodate.
- It prevents designing adapters for rejected mechanisms.
- It makes later test failures easier to interpret.
- It keeps the first implementation slice smaller and less magical.

### Decision 60: Use A Staged Validation Ladder Before Production Runs

Decision:

- The rebuild should use a staged validation ladder.
- Each architecture layer must prove its own boundary before integration.
- Exact synthetic tests and tiny smokes should come before production-scale
  training.
- Full training runs are not the first validation mechanism.

Validation ladder:

```text
Level 0: config/schema validation
Level 1: semantic planning validation
Level 2: template/encoding validation
Level 3: span/coordinate validation
Level 4: objective math validation
Level 5: bridge/model-forward validation
Level 6: integrated tiny training smoke
Level 7: Stage-2 rollout-planning smoke
```

Level responsibilities:

- Level 0: config/schema validation
  - unknown knobs fail;
  - removed knobs fail;
  - objective profiles resolve deterministically;
  - precision policies resolve;
  - surface-specific schemas reject irrelevant sections.
- Level 1: semantic planning validation
  - Stage-1 compact supervision objects are correct;
  - Stage-2 Channel-B accepted / false-negative / ignored objects are correct;
  - assignment and duplicate provenance is preserved;
  - object ordering is deterministic or seeded.
- Level 2: template/encoding validation
  - compact-full rendered bytes are exact;
  - token ids align with ms-swift encoding;
  - schema / description / coordinate token roles are correct;
  - rendered text is diagnostic-only;
  - `ModelInputBundle` keys are strict.
- Level 3: span/coordinate validation
  - label-position to logit-position shift is correct;
  - span prediction positions are valid;
  - coordinate slots map to correct tokens;
  - `logits_to_keep` / packing are rejected or explicitly unsupported for now;
  - no sidecar enters model inputs.
- Level 4: objective math validation
  - `TokenCEObjective` matches PyTorch / ms-swift CE on fixed tensors;
  - trie CE multi-positive log-sum-exp is correct in float32;
  - coordinate soft CE normalizes correctly;
  - bbox regression uses float32 and stable denominators;
  - objective-local denominator / weight metrics are correct.
- Level 5: trainer-loss bridge validation
  - sidecars are stripped;
  - model is called once;
  - Qwen3-VL multimodal inputs are preserved;
  - full logits are kept;
  - labels are removed when `ObjectiveRunner` owns loss;
  - `PredictionCoordinateMapper` is constructed.
- Level 6: integrated tiny training smoke
  - one tiny compact-full Stage-1 batch;
  - one forward/backward step;
  - `ObjectiveRunner` emits loss and metrics;
  - no Qwen3-VL placeholder / grid / position failure;
  - artifacts include resolved config and metric events.
- Level 7: Stage-2 rollout-planning smoke
  - synthetic rollout predictions plus ground truth;
  - greedy IoU assignment;
  - deterministic duplicate filtering;
  - false-negative insertion;
  - sorted/random ordering provenance;
  - Channel-B `SupervisionPlan` diagnostic sample.

Future command shape:

```bash
conda run -n ms python -m pytest tests/test_training_config_strict_unknown_keys.py -q
conda run -n ms python -m pytest tests/test_compact_full_encoding_contract.py -q
conda run -n ms python -m pytest tests/test_supervision_span_coordinate_mapper.py -q
conda run -n ms python -m pytest tests/test_objective_runner_math.py -q
conda run -n ms python -m pytest tests/test_trainer_loss_bridge_qwen3vl_contract.py -q
conda run -n ms python -m pytest tests/test_stage2_supervision_planning_smoke.py -q
```

Rationale:

- The architecture is position-sensitive and multimodal-forward-sensitive.
- Tiny exact tests catch contract drift faster than full training.
- Production runs are too expensive and too slow to be first-line validation.
- This ladder creates a clean proof path from config to loss to Stage-2 planning.

### Decision 61: Start With A Compact-Full Golden Thread Fixture

Decision:

- The first golden fixture should be a canonical compact-full multimodal Stage-1
  detection example.
- It should exercise the full source-to-objective path.
- A Stage-2 rollout-planning golden fixture should be added second.

First golden fixture contents:

- one image;
- three objects;
- mixed descriptions;
- nearby boxes that are not duplicates;
- one object with coordinate tokens near boundary values;
- compact-full rendered target;
- ms-swift encoded model inputs;
- `EncodedDetectionView`;
- `SupervisionPlan`;
- `SupervisionSpan` list;
- `PredictionCoordinateMapper` expectations;
- `ObjectiveBatch` expectations;
- diagnostic JSON snapshot.

Golden thread:

```text
source row
  -> SupervisionPlan
  -> compact_full rendered text
  -> EncodedDetectionView
  -> ModelInputBundle
  -> SupervisionSpan
  -> ObjectiveBatch
```

Second golden fixture:

```text
ground truth objects:
  A, B, C

rollout predictions:
  A matched
  duplicate of A
  C matched but lower quality duplicate exists
  missing B

expected:
  assignment result
  duplicate survivor / drop reasons
  false-negative insertion for B
  sorted final object order
  Stage2ChannelBSupervisionPlan
```

Rationale:

- The Stage-1 compact-full fixture validates the central template, span, and
  objective path.
- It tests compact-full token roles, coordinate slots, and ms-swift encoding
  parity.
- It is small enough to inspect by hand when it fails.
- The Stage-2 fixture reuses the same target-format anchor while testing
  assignment, duplicate filtering, and false-negative insertion.

### Decision 62: Golden Fixtures Use Static Snapshots Plus Builders

Decision:

- Golden fixtures should use both static source / expected JSON snapshots and
  small test builders.
- Static fixtures should be human-readable and reviewable.
- Builders should construct typed objects through real code.

Target fixture layout:

```text
tests/fixtures/training_architecture/
  compact_full_stage1_source.json
  compact_full_stage1_expected.json
  stage2_rollout_source.json
  stage2_rollout_expected.json

tests/helpers/training_architecture_fixture_builder.py
```

Pattern:

```text
static source fixture
  -> human-readable input

builder
  -> constructs typed objects through real code

static expected snapshot
  -> small canonical JSON expectations for important fields
```

Rules:

- keep snapshots small and semantic;
- do not snapshot full logits;
- snapshot token ids only for tiny fixtures where alignment matters;
- snapshot provenance enough to debug ordering, assignment, and filtering;
- include schema/version fields;
- update snapshots deliberately when architecture changes.

Rationale:

- Static fixtures are easy to inspect and review.
- Builders prevent hand-written snapshots from drifting away from code reality.
- Expected snapshots make accidental output changes visible in diffs.
- Small semantic snapshots avoid turning tests into giant tensor archives.

## Target Architecture Direction

The desired clean architecture is:

```text
YAML config
  -> TrainingSurfaceResolver
  -> ResolvedTrainingRun
  -> DatasetPlan + DetectionTemplateCodec
  -> EncodedDetectionView
  -> SupervisionBatch[SupervisionSpan]
  -> ObjectiveRunner
  -> CoordExpTrainer / Swift adapter
  -> typed metrics + diagnostics writers
  -> manifests and artifacts
```

Central abstractions to design:

- `TrainingSurface`
  - Resolved training mode: Stage-1 compact trie CE, Stage-1 JSON CE, optional
    simplified Stage-2.
- `DetectionTemplateCodec`
  - Single owner for render, strict parse, salvage parse, prompt snippets,
    schema-token inventory, tokenizer stop contract, and token-role projection.
- `EncodedDetectionView`
  - Encoded sample with `input_ids`, `labels`, `attention_mask`,
    assistant/stop spans, schema spans, desc spans, coord slots, object entries,
    and sidecar positions.
- `SupervisionSpan`
  - Typed target span with role, object id, channel, token positions, weight,
    mask policy, and target distribution.
- `TargetDistribution`
  - Keep canonical types such as hard token CE, sparse multi-positive target,
    coord soft target, and bbox regression target.
  - Do not include duplicate-negative target as a canonical distribution.
- `ObjectiveRunner`
  - Executes objectives from spans instead of trainer mixins or loosely typed
    metadata.
- `PackingSegmentMap`
  - Maps unpacked encoded views into packed positions and rewrites span/sidecar
    offsets.
- `AssignmentStrategy`
  - Stage-2 pairing abstraction; default future implementation is greedy IoU,
    not Hungarian.
- `DuplicateFilter`
  - Removes duplicate predicted/rollout objects before positive target
    realization and emits diagnostics.
- `ObservabilityConfig`
  - Typed config root for loss components, span stats, token-type stats, trie
    stats, coordinate diagnostics, decode/rollout diagnostics, monitor dumps,
    and metric strictness.

## Near-Term Implementation Implications

The first concrete cleanup wave should focus on:

1. Removing duplicate-negative training:
   - code;
   - configs;
   - tests;
   - active docs.
2. Removing adjacent repulsion:
   - code;
   - configs;
   - tests;
   - active docs.
3. Removing training-time EOS/forced-continuation hacks:
   - training code/configs/tests;
   - production-looking docs;
   - preserve metrics/probes.
4. Keeping duplicate diagnostics and posthoc guarded eval:
   - clearly labeled as analysis views, never silent canonical metrics.
5. De-canonicalizing Hungarian / rollout-aligned:
   - docs and plan language only for now;
   - code removal after greedy-IoU replacement smoke.

## Future Plan Caveat

When the super-power implementation plan is drafted or revised, it must encode
the decisions in this note. In particular, it must not preserve
duplicate-negative training as an experiment-only module unless the user later
explicitly reverses the full-removal decision.

The approved current decision is full removal of duplicate-negative training and
adjacent repulsion, with diagnostic-only retention for EOS/continue probes.

## Naming Supersession

Some earlier sections in this decision record use `TargetPlan` or
`Stage2TargetPlan` while the naming discussion was still unresolved. Those
names are historical and not canonical for the next implementation plan.

The current working name is `SupervisionPlan`, still explicitly provisional and
still open to a final naming pass before stable APIs are introduced.

Implementation plans should not create new public `TargetPlan` APIs unless the
user explicitly reopens and reverses the naming decision.

## Position-Semantics Supersession

Some earlier sections use the phrase `prediction_positions`. The refined
Superpowers spec makes this explicit to avoid off-by-one ambiguity:

- `SupervisionSpan.label_positions` stores target-token label positions.
- `PredictionCoordinateMapper` is the only component that maps label position
  `p` to model-logit row `p - 1`.
- implementation plans should not store already-shifted logit rows in
  `SupervisionSpan` under the ambiguous name `prediction_positions`.

## Stage-2 Rollout Template Boundary Addendum

Decision date: 2026-05-17

The A2 random ET-RMP-CE compact-full checkpoint smoke exposed a Stage-2
architecture gap that must be treated as a blocking design adjustment:

- The A2 compact-full adapter at `checkpoint-3600` and `checkpoint-3664` can be
  loaded into Stage-2 through the Qwen3-VL base model plus `model.adapters`.
- The Stage-2 tiny launch can complete one training step and write normal
  artifacts, but the current Stage-2 rollout path is still CoordJSON-shaped.
  It uses CoordJSON rollout prompting/parsing/false-negative append behavior
  rather than the compact-full sequence surface used to train A2.
- The same `checkpoint-3664` produces valid compact-full infer/eval outputs
  through the compact-full infer pipeline with compact grammar decoding.

Resolved decision:

- Stage-2 rollout I/O must become a first-class template-aware boundary.
- `compact_full` Stage-2 rollouts require compact-full prompt construction,
  unconstrained default rollout decoding, compact-full parsing,
  compact-full false-negative append serialization, compact-full supervision
  conversion, and template-aware artifacts.
- CoordJSON rollout parsing/appending remains legacy-only for explicit
  CoordJSON surfaces. It must not be selected implicitly for compact-full
  checkpoints through legacy `custom.json_format` defaults.
- Compact-full Stage-2 readiness cannot be claimed from process exit, loss
  logging, or `invalid_rollout=0` alone. It requires a real-backend A2 smoke
  that proves at least one valid generated compact-full predicted object
  before assignment, duplicate-filter, and false-negative metrics are
  interpreted as model-quality signals.
- Migration is dual-surface and explicit: `compact_full` is canonical for
  A2-style checkpoints and new Stage-2 work, while `coordjson` remains runnable
  only as an explicit legacy surface. Implicit fallback or mixed surface
  selection is forbidden.
- Compact-full Stage-2 training must face the model's unconstrained rollout
  distribution by default. Do not hide force-continuation, duplicate-burst, or
  special-basin failures behind grammar-constrained decoding. Compact grammar
  decoding may be kept only as a labeled diagnostic/control comparison.
- Malformed or empty compact-full rollouts should not be dropped from Channel-B
  training by default. They fall back to GT/FN append-only supervision because
  an empty rollout can mean the model failed to recognize image objects and
  should receive a correction signal. The fallback remains visible through
  invalid/empty rollout metrics and raw artifacts, and it does not count as
  valid-rollout evidence.
- GT/FN fallback supervision uses the same initial loss weight as normal
  Channel-B correction (`fallback_loss_weight=1.0`). It must carry explicit
  provenance such as `rollout_context=fallback_gt_fn_append_only` and separate
  metrics such as `loss/B_fallback/*`, `rollout/fallback_loss_share`, and
  `rollout/invalid_fallback_gt_fn_rate`. If fallback exceeds roughly 30-40% of
  Channel-B samples over a monitoring window, the run should warn that the
  rollout distribution is unhealthy.
- A2 compact-full Stage-2 rollout I/O acceptance uses two gates:
  - Gate 1, launch/I/O wiring: 2-4 samples, unconstrained greedy decoding,
    compact-full parser only, no CoordJSON fallback, raw output artifacts
    preserved, and at least one valid predicted compact-full object.
  - Gate 2, rollout readiness: 16-32 samples, unconstrained greedy decoding,
    `sample_valid_pred_rate >= 0.75`, parser-template mismatch rate equal to
    zero, parse truncation, empty-valid-object, and GT/FN fallback cases
    reported explicitly, and raw rollouts plus parsed objects materialized for
    manual inspection.

Implementation consequence:

- Add a `Stage2RolloutTemplatePolicy` / rollout codec seam before Stage-2
  assignment and duplicate filtering.
- Add compact-full `fallback_gt_fn_append_only` behavior for malformed or empty
  model rollouts, while hard-failing configuration-level template/parser
  mismatches.
- Reuse the current compact-full infer/eval path as the parity reference,
  especially strict compact-full parsing and artifact serialization. Compact
  grammar decode is not the training default.
- Record the resolved rollout template, parser, decode policy, and append policy
  in resolved config and artifacts.

## Continue The Grill-Me Loop

Next decisions still worth asking when the context resumes:

1. Should the first implementation pass be a narrow cleanup-only pass, or should
   it also introduce the first `TrainingSurface` skeleton?
   - Recommended: cleanup-only first, then `TrainingSurface`.
2. Should posthoc eval duplicate guard stay in `src/eval/` indefinitely, or be
   moved under an explicit `analysis_views` / guarded artifact namespace?
   - Recommended: keep for now, rename/label as guarded analysis view.
3. Should EOS/continue diagnostic probes live under `src/analysis/` only, with
   no import path from training modules?
   - Recommended: yes.
4. Should `progress/index.yaml` add explicit statuses such as
   `active-reference`, `mechanism-evidence`, `concluded-negative`,
   `superseded`, and `archive-only`?
   - Recommended: yes, after cleanup note lands.
5. Should old Stage-2 `rollout_aligned` stay runnable until greedy-IoU smoke
   passes, or be made explicitly non-discoverable but still importable?
   - Recommended: runnable but legacy-labeled until replacement smoke.

## Minimal Verification For The Cleanup Wave

Before cleanup:

```bash
rg -n "loss_duplicate_burst_unlikelihood|adjacent_repulsion|stop_signal_damping|eos_loosen|force.*continu|hungarian|linear_sum_assignment" \
  src configs docs tests --glob '!progress/**' --glob '!docs/superpowers/**'
```

After cleanup:

```bash
rtk conda run -n ms python -m pytest \
  tests/test_training_config_strict_unknown_keys.py \
  tests/test_stage2_ab_config_contract.py \
  tests/test_stage2_two_channel_training.py \
  tests/test_detection_eval_output_parity.py \
  tests/test_confidence_postop.py -q
```

Expected cleanup outcome:

- duplicate-negative and adjacent-repulsion training symbols are absent from
  live code/config/tests/docs;
- training-time EOS/forced-continuation hacks are absent from live training
  code/config/tests/docs;
- diagnostic probes and monitoring metrics remain available;
- duplicate filtering and diagnostics remain;
- posthoc guarded eval remains explicitly labeled;
- Hungarian remains only as legacy executable path until greedy-IoU replacement.
