# runtime-architecture-refactor-program Specification

## Purpose
Define the permanent architectural contract for runtime-critical CoordExp refactors so internal decomposition preserves stable behavior, reproducibility artifacts, and operator-facing compatibility while code ownership shifts into narrower seams.
## Requirements
### Requirement: Runtime-critical refactors preserve stable behavior through explicit interface boundaries
The runtime-critical CoordExp stack SHALL be refactored through explicit internal interface boundaries rather than large in-place rewrites.

Normative behavior:

- the refactor MUST preserve current stable behavior for:
  - typed config loading and strict unknown-key handling,
  - geometry and coordinate invariants,
  - Qwen3-VL chat-template-compatible prompt/rendering behavior,
  - stable metric key families,
  - stable infer/eval artifact contracts,
  - run-manifest and provenance artifacts,
- any internal boundary introduced during the refactor MUST define the payloads it exchanges explicitly enough for targeted tests to validate them,
- if a slice cannot preserve one of the above stable behaviors, that slice MUST be blocked pending an explicit follow-on spec delta for the behavior change.

#### Scenario: Internal extraction preserves stable training and artifact contracts
- **GIVEN** a refactor slice that extracts trainer or runtime helpers into dedicated modules
- **WHEN** the targeted parity and contract tests are run
- **THEN** stable config, metric, geometry, and artifact behavior remains unchanged
- **AND** the slice is treated as an internal decomposition rather than a silent contract change.

### Requirement: Stage-2 rollout correction exposes dedicated runtime seams by concern
The Stage-2 rollout-correction implementation SHALL separate step execution, correction target construction, rollout runtime, and objective execution through dedicated seams by concern.

Normative behavior:

- the `stage2_rollout_correction` path MUST NOT reintroduce an A/B scheduler seam,
- rollout-correction target construction and residual supervision assembly MUST be isolatable from step-execution and DDP coordination logic,
- objective execution / metric projection MUST be isolatable from rollout/target construction,
- trainer-facing compatibility adapters MAY remain during migration, but responsibility ownership MUST move toward the dedicated seams.

#### Scenario: rollout-correction target construction can be tested without executor runtime coupling
- **WHEN** rollout-correction residual target construction is exercised in targeted tests
- **THEN** it can be invoked and validated without requiring the full threaded executor / DDP coordination path
- **AND** the resulting supervision payload remains compatible with trainer loss execution.

### Requirement: Shared rollout runtime uses compatibility-preserving backend interfaces
Rollout generation and backend orchestration shared across Stage-2 trainers SHALL be mediated through compatibility-preserving backend/runtime interfaces rather than remaining embedded inline in one trainer class.

Normative behavior:

- backend selection, decode request handling, backend lifecycle, and rollout fanout MUST be isolatable from trainer-specific target construction,
- trainer-facing adapter methods MAY remain during migration so existing tests and monkeypatch-based compatibility are preserved,
- extracted runtime interfaces MUST preserve rollout payload fields required by strict parsing, matching, and supervision construction.

#### Scenario: Shared rollout runtime extraction preserves trainer-facing rollout calls
- **GIVEN** a trainer implementation that historically called a trainer-local rollout method
- **WHEN** rollout runtime ownership is moved into dedicated runtime helpers
- **THEN** the trainer can still exercise rollout generation through a compatibility-preserving adapter
- **AND** returned rollout payloads remain valid for downstream parsing and matching.

### Requirement: Bootstrap, inference, and evaluation decomposition preserve authoritative contracts
Entry-point and offline-runtime decomposition SHALL preserve the authoritative contracts owned by typed config, inference artifacts, and evaluation outputs.

Normative behavior:

- `sft` bootstrap decomposition MUST not duplicate or silently diverge from typed config authority,
- inference backend decomposition MUST preserve current JSONL, summary, and token-trace artifact contracts,
- evaluation decomposition MUST preserve current output artifact semantics and parity expectations,
- architecture work MAY reorganize code ownership, but MUST NOT silently rename or redefine stable outputs.

#### Scenario: Infer/eval module split preserves artifact parity
- **GIVEN** an infer or evaluation slice that moves backend/artifact logic into dedicated submodules
- **WHEN** infer/eval parity tests are run against the refactored implementation
- **THEN** emitted artifacts remain byte-compatible or contract-compatible with the pre-refactor behavior
- **AND** downstream workflows continue to consume them without migration changes.

### Requirement: Inference runtime refactor removes overlapping active layouts

Runtime architecture cleanup SHALL converge offline and online detection decode
behavior into `src/infer` and SHALL remove overlapping active import surfaces
after migration.

Normative behavior:

- final active code MUST NOT depend on the retired trainer rollout-runtime
  package;
- final active code MUST NOT depend on the legacy Stage-2 rollout runtime module
  for prompt rendering, backend lifecycle, decode request conversion, trace
  normalization, or parser policy; any remaining Stage-2 module with that
  responsibility must be renamed or reduced to a thin trainer-owned facade;
- final active code MUST NOT depend on legacy infer engine/backend aliases;
- final active caller-facing code MUST NOT import retired infer decode-constraint
  modules or facades; compact-row inference uses free decode unless a future
  change introduces a new explicitly scoped mechanism;
- scripts, callbacks, analysis utilities, tests, configs, docs, and
  non-archived OpenSpec references in scope MUST be updated rather than routed
  through compatibility aliases;
- any temporary migration shim MUST be deleted before the refactor is complete;
- archived history may retain old names when it is clearly historical.

#### Scenario: Search gate rejects active legacy import surfaces

- **WHEN** implementation claims the unified inference runtime refactor is
  complete
- **THEN** search gates over active `src`, `scripts`, `tests`, `configs`,
  `docs`, and non-archived OpenSpec surfaces find no active imports of retired
  trainer rollout-runtime, legacy Stage-2 rollout-runtime, legacy infer engine,
  infer backend alias, or retired infer decode-constraint import surfaces
- **AND** archived records are the only allowed remaining historical mentions.

### Requirement: Shared runtime stays small and caller boundaries remain explicit

The shared inference runtime SHALL avoid a broad framework shape and SHALL keep
caller-specific downstream behavior outside `src/infer`.

Normative behavior:

- `src/infer` MAY own prompt, backend, parsing, artifacts, constraints,
  checkpoint resolution, visualization helpers, and offline pipeline
  orchestration;
- Stage-2 target construction, duplicate filtering, greedy IoU assignment,
  DDP coordination, loss execution, and training metric projection MUST remain
  trainer-owned;
- `src/infer` MUST NOT import rollout-correction target builder, residual
  boundary adapters, target IR constructors, greedy IoU matching helpers, or
  duplicate-control training logic;
- COCO/LVIS evaluator math MUST remain evaluator-owned;
- module splits under `src/infer` SHOULD happen only when they create a
  focused independently testable policy/adapter and avoid variant-specific
  layout sprawl.

#### Scenario: Residual target construction remains trainer-owned

- **WHEN** Stage-2 rollout generation moves to the shared inference runtime
- **THEN** the runtime returns decoded/parsed rollout results and trace
  metadata
- **AND** residual correction event construction and target IR creation remain
  in Stage-2 trainer-owned modules.
