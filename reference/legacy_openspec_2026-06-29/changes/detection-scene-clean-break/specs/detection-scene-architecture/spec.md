## ADDED Requirements

### Requirement: DetectionScene clean-break contract activates only after approval gates

This OpenSpec change SHALL define the target clean-break architecture, not
current runnable behavior.

Normative behavior:

- current docs and stable specs MUST remain authoritative for existing
  workflows until the user approves production implementation;
- the archive checkpoint MUST be recorded before deletion or rename cleanup
  begins;
- implementation MUST NOT claim old public names or historical paths are
  retired until the corresponding cleanup slice has run;
- retained canonical behavior MUST NOT be removed before its clean-break
  replacement seam and characterization checks are named;
- a historical path MAY be deleted or quarantined only after it is classified as
  delete or quarantine and covered by the archive checkpoint plus search gates.

#### Scenario: Approval of the target contract does not change current runtime authority

- **GIVEN** this OpenSpec change is reviewed but production implementation is
  not yet approved
- **WHEN** an operator asks how current CoordExp runs are launched
- **THEN** current docs and stable specs remain the authority
- **AND** the DetectionScene clean-break vocabulary is treated as target
  architecture rather than already-implemented behavior.

### Requirement: DetectionScene is the canonical semantic object for active detection workflows

Active detection workflows SHALL use `DetectionScene` as the central in-memory
semantic object for one image's detection meaning.

Normative behavior:

- `DetectionScene` MUST represent exactly one detection image/example;
- `DetectionScene` MUST carry image identity, exactly one resolved image
  reference, image dimensions, coordinate frame, coordinate space, bbox chart,
  ground-truth detection objects, object ordering policy, and metadata required
  by Stage-1, Stage-2, inference, and eval projections;
- `DetectionObject` MUST represent one annotated object and its label or
  description plus geometry and object metadata;
- `DetectionGeometry` MUST make bbox or poly representation explicit and MUST
  not infer bbox chart or coordinate space from coordinate position alone;
- raw JSONL records MUST remain the on-disk storage contract but MUST NOT be the
  canonical in-memory semantic object;
- rendered text, token supervision, rollout states, decoded predictions, and
  eval rows MUST be treated as owned projections or artifacts, not as the
  semantic center;
- `DetectionScene` MUST NOT contain rendered assistant text, token IDs, token
  spans, rollout assignments, correction events, raw backend output, eval metric
  fields, or visualization layout metadata;
- `DetectionDocument` MUST NOT be introduced as the new public semantic term for
  this architecture.

#### Scenario: Detection semantics are represented without storage or render leakage

- **GIVEN** a raw detection JSONL record with image metadata and object geometry
- **WHEN** it is loaded into the canonical detection semantic layer
- **THEN** the result is a `DetectionScene`
- **AND** callers that need detection semantics do not inspect rendered assistant
  text, token sidecars, rollout targets, or eval rows as their source of truth.

### Requirement: DetectionScene does not absorb runtime, trainer, or evaluator ownership

`DetectionScene` SHALL remain a semantic interchange object rather than a broad
runtime framework.

Normative behavior:

- prompt rendering and chat-template boundary behavior MUST remain owned by
  template/prompt runtime layers;
- backend decode, backend lifecycle, trace conversion, parser policy, and
  prompt/decode/model provenance MUST remain owned by shared inference/runtime
  layers;
- Stage-2 assignment, duplicate filtering, correction-event construction,
  target IR construction, DDP coordination, loss execution, and training metric
  projection MUST remain owned by Stage-2 correction/training layers;
- evaluator math, score application, duplicate-control post-op, and official
  metric reporting MUST remain evaluator/artifact responsibilities;
- `DetectionScene` MAY be consumed by those layers, but MUST NOT become their
  owner or a dumping ground for their policy decisions.

#### Scenario: Scene object does not own Stage-2 target policy

- **GIVEN** Stage-2 rollout correction consumes a `DetectionScene`
- **WHEN** duplicate filtering, assignment, and correction-event construction
  run
- **THEN** those policies are owned by Stage-2 correction modules
- **AND** `DetectionScene` supplies semantic GT/image/object state without
  embedding the trainer's policy decisions.

### Requirement: Detection representation ownership is layered and non-overlapping

The clean-break detection stack SHALL assign each representation to one owning
layer and SHALL prevent other layers from treating that representation as
canonical beyond its purpose.

Normative behavior:

- data/dataset loading owns `RawDetectionRecord` intake;
- detection domain code owns `DetectionScene`, `DetectionObject`, and
  `DetectionGeometry`;
- detection template code owns `RenderedDetectionSequence` and
  `DetectionSequenceTemplate`;
- detection tokenization/objective code owns `DetectionSupervisionView`;
- shared inference/runtime parsing owns strict parser policy and
  `DecodedDetectionResult`;
- Stage-2 correction modules own `RolloutPrediction` only as a rollout-context
  view over shared runtime decoded output, and own `DetectionAssignment` and
  `CorrectionEvent`;
- eval/artifact code owns `DetectionEvalRecord` and
  `ScoredDetectionEvalRecord`;
- no layer MAY require another layer's private representation to recover its
  own canonical meaning.

#### Scenario: Token supervision does not become semantic authority

- **GIVEN** Stage-1 needs labels, masks, coordinate spans, and sidecars
- **WHEN** it constructs training inputs
- **THEN** those inputs are represented as `DetectionSupervisionView`
- **AND** object identity, geometry, and ordering semantics are derived from the
  source `DetectionScene`, not reconstructed from token positions.

### Requirement: DetectionScene preserves explicit geometry, ordering, and metric eligibility semantics

`DetectionScene` and its derived views SHALL preserve hard detection semantics
needed for training, rollout correction, inference, eval, and visualization.

Normative behavior:

- each object MUST expose exactly one explicit geometry kind such as `bbox_2d`
  or `poly` at serialization boundaries that use JSON object payloads;
- bbox geometry MUST preserve `[x1, y1, x2, y2]` meaning unless a separate
  semantic-change decision approves another outward representation;
- offline model-facing bbox branches that intentionally store non-`xyxy` slots
  under a historical `bbox_2d` key MUST carry explicit bbox-chart provenance
  and MUST NOT be consumed as canonical `xyxy` at inference, eval, or
  visualization boundaries;
- polygon geometry MUST preserve ordered vertices and valid arity;
- a generic serialized key named `geometry` MUST NOT replace concrete outward
  geometry keys unless a future schema explicitly introduces it as an internal
  typed union separate from existing JSON payloads;
- object instance ordering and per-object field ordering MUST remain separate
  policies;
- strict metric-bearing results MUST be distinguished from diagnostic
  non-metric salvage or incomplete-geometry records.

#### Scenario: Field ordering does not change object instance ordering

- **GIVEN** a `DetectionScene` whose objects are projected into rendered target
  text
- **WHEN** a surface changes per-object field order
- **THEN** the object instance order remains governed by the explicit object
  ordering policy
- **AND** concrete geometry keys remain explicit in the rendered payload.

### Requirement: Stage-1 and Stage-2 projections share DetectionScene semantics

Stage-1 teacher-forcing detection and Stage-2 rollout correction SHALL be
designed as projections from the same `DetectionScene` semantic layer.

Normative behavior:

- Stage-1 projection MUST follow `DetectionScene -> RenderedDetectionSequence ->
  DetectionSupervisionView`;
- Stage-2 projection MUST follow `DetectionScene + RolloutPrediction ->
  DetectionAssignment -> CorrectionEvent -> DetectionSupervisionView`;
- `RolloutPrediction` MUST be derived from shared inference/runtime decode and
  parser policy rather than from Stage-2-local raw-text parsing or diagnostic
  salvage;
- Stage-2 design MUST constrain the shared semantic layer before Stage-1-only
  implementation claims the seam is complete;
- Stage-1-first implementation MAY be used as a proving slice only when the
  Stage-2 consumption path is specified and kept compatible with the shared
  scene semantics.

#### Scenario: Stage-2 correction consumes scene semantics instead of Stage-1 artifacts

- **GIVEN** a Stage-2 rollout-correction sample with GT objects and parsed
  rollout predictions
- **WHEN** correction targets are constructed
- **THEN** matching and correction events consume `DetectionScene` and
  `RolloutPrediction`
- **AND** they do not depend on Stage-1 rendered target text or Stage-1 token
  sidecars as semantic inputs.

#### Scenario: Stage-2 rollout prediction uses shared parse provenance

- **GIVEN** raw model output from an online rollout backend
- **WHEN** Stage-2 prepares rollout-correction targets
- **THEN** parser policy, invalid/drop metadata, prompt/decode/model provenance,
  and metric-bearing status come from the shared inference/runtime boundary
- **AND** Stage-2 does not construct a metric-bearing rollout prediction through
  a private parser or diagnostic salvage path.

### Requirement: Clean-break cleanup preserves hard detection invariants, not historical runtime compatibility

The clean-break architecture SHALL allow historical runtime compatibility to be
dropped after an archive checkpoint while preserving hard detection invariants
for retained canonical surfaces.

Normative behavior:

- a pre-cleanup commit, branch, or tag MUST be identified and recorded before
  large deletion or rename work;
- old behavior MUST be preserved by archive checkpoint, progress notes,
  historical docs, and artifact references rather than live compatibility code;
- retained canonical surfaces MUST preserve image/geometry alignment,
  no-silent-resize behavior, explicit coordinate frames, explicit object
  ordering, explicit bbox/poly meaning, template-owned chat/render boundaries,
  template/tokenization-derived supervision, explicit Stage-2 correction target
  construction, explicit invalid/drop metadata, raw/scored eval separation,
  metric/provenance interpretability, and deterministic config resolution;
- changes to those retained semantics MUST be approved as separate semantic
  changes.

#### Scenario: Deleting a retired surface does not require preserving its runtime path

- **GIVEN** the archive checkpoint has been recorded
- **AND** a historical config namespace is classified as delete or quarantine
- **WHEN** the cleanup removes that namespace from active code and docs
- **THEN** no compatibility adapter is required for that historical runtime path
- **AND** retained canonical surfaces still preserve their stated detection
  semantics.

### Requirement: Canonical naming replaces misleading historical public concepts

The clean-break architecture SHALL rename misleading public concepts when their
current names preserve historical mechanisms rather than target meaning.

Normative behavior:

- new public concepts MUST prefer the canonical vocabulary:
  `DetectionScene`, `DetectionObject`, `DetectionGeometry`,
  `RenderedDetectionSequence`, `DetectionSequenceTemplate`,
  `DetectionSupervisionView`, `RolloutPrediction`, `DetectionAssignment`,
  `CorrectionEvent`, `DecodedDetectionResult`, `DetectionEvalRecord`, and
  `ScoredDetectionEvalRecord`;
- new public Stage-1 detection teacher-forcing surfaces MUST NOT use
  `recursive_detection_ce` as the target architecture name;
- new public Stage-2 runtime/decode/eval surfaces MUST NOT use
  `rollout_matching` as the target architecture name;
- old names MAY remain only as explicitly classified archive history,
  temporary migration handles, rejection tests, or private implementation
  details scheduled for rename/delete.

#### Scenario: A remaining old name is classified rather than treated as active authority

- **WHEN** active source, configs, docs, tests, and non-archived OpenSpec are
  searched after a cleanup slice
- **THEN** remaining mentions of old public names are classified as retained
  canonical, migration handle, rejection/absence test, private implementation
  detail, or archive history
- **AND** unclassified historical names do not remain as current authority.

### Requirement: DetectionScene implementation requires readiness gates before runtime code changes

Production implementation of the clean-break detection architecture SHALL NOT
start until the readiness gates are named in the implementation plan.

Normative behavior:

- the implementation plan MUST name the archive checkpoint gate;
- the implementation plan MUST classify surfaces as keep/rename, temporary
  migration handle, quarantine, or delete;
- the implementation plan MUST name semantic parity requirements for retained
  surfaces;
- the implementation plan MUST define Stage-1, Stage-2, inference/eval, naming,
  config, verification, and governance gates;
- the user MUST explicitly approve implementation after reviewing the OpenSpec
  and implementation plan.

#### Scenario: OpenSpec approval does not start production implementation

- **GIVEN** this OpenSpec change has been drafted and reviewed
- **WHEN** the planning phase completes
- **THEN** production runtime code remains unchanged
- **AND** implementation begins only after the user explicitly approves the
  implementation phase.
