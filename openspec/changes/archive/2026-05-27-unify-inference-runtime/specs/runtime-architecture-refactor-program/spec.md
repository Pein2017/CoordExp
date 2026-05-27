## ADDED Requirements

### Requirement: Inference runtime refactor removes overlapping active layouts

Runtime architecture cleanup SHALL converge offline and online detection decode
behavior into `src/infer` and SHALL remove overlapping active import surfaces
after migration.

Normative behavior:

- final active code MUST NOT depend on `src.trainers.rollout_runtime.*`;
- final active code MUST NOT depend on `src.trainers.stage2_rollout_runtime`
  for prompt rendering, backend lifecycle, decode request conversion, trace
  normalization, or parser policy; any remaining Stage-2 module with that
  responsibility must be renamed or reduced to a thin trainer-owned facade;
- final active code MUST NOT depend on `src.infer.engine` or
  `src.infer.backends`;
- final active caller-facing code MUST NOT import `src.infer.compact_grammar`
  or `src.infer.stop_pressure`; compact grammar and stop-pressure behavior must
  be exposed through the shared constraints facade or private helper modules;
- scripts, callbacks, analysis utilities, tests, configs, docs, and
  non-archived OpenSpec references in scope MUST be updated rather than routed
  through compatibility aliases;
- any temporary migration shim MUST be deleted before the refactor is complete;
- archived history may retain old names when it is clearly historical.

#### Scenario: Search gate rejects active legacy import surfaces

- **WHEN** implementation claims the unified inference runtime refactor is
  complete
- **THEN** search gates over active `src`, `scripts`, `tests`, `configs`,
  `docs`, and non-archived OpenSpec surfaces find no active imports of
  `src.trainers.rollout_runtime`, decode-owning
  `src.trainers.stage2_rollout_runtime`, `src.infer.engine`,
  `src.infer.backends`, `src.infer.compact_grammar`, or
  `src.infer.stop_pressure`
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
