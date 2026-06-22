## ADDED Requirements

### Requirement: Active training configs migrate as a breaking hierarchy change

The SFT pipeline hierarchy migration SHALL be breaking for active repo-owned
training configs.

Normative behavior:

- active repo-owned configs identified by current docs/catalog routing at
  migration/archive time MUST migrate to the target hierarchy in the
  implementation slice;
- the first migration slice MUST include the catalog-canonical Stage-1 research
  teacher-forcing config
  `configs/stage1/detection_teacher_forcing/prod/compact_support2.yaml`;
- the first migration slice MUST include active Standard SFT launch-prep leaves
  from `/data/CoordExp` main for sorted/random ordering with object and bbox
  closure, while preserving their status as current work;
- active docs and catalog entries MUST route to the migrated configs;
- archive and historical configs MAY remain as evidence, but MUST NOT be routed
  as active runnable configs;
- the migration MUST preserve strict unknown-key fail-fast behavior;
- the migration MUST NOT add a new templating system or stable CLI flags.

#### Scenario: Active config routing is migrated

- **GIVEN** docs/catalog routing names active Stage-1 or Stage-2 training config
  families
- **WHEN** this change is implemented
- **THEN** those active config families migrate to the target hierarchy
- **AND** docs/catalog routing no longer advertises the old hierarchy as active.

#### Scenario: Catalog canonical compact support config migrates from flat dialect

- **GIVEN** docs/catalog routing points to
  `configs/stage1/detection_teacher_forcing/prod/compact_support2.yaml`
- **WHEN** this change is implemented
- **THEN** the active migrated config uses
  `pipeline.id: stage1_research_teacher_forcing`
- **AND** it uses `objective.id: research_teacher_forcing`
- **AND** it expresses ordering, bbox format, coordinate surface, and strict
  parse through `sample_factory.target_sequence`.

#### Scenario: Historical config is not advertised as active

- **GIVEN** a config under an archive or historical evidence root
- **WHEN** docs and catalog routing are updated
- **THEN** the config may remain inspectable
- **AND** it is not listed as an active launch surface.

### Requirement: Old and new sequence-control paths cannot be dual-authored

The config loader SHALL reject active target-hierarchy configs that author both
old `custom.*` sequence controls and new `sample_factory.target_sequence`
controls.

Normative behavior:

- duplicate old/new authoring MUST fail even when values match;
- validation errors MUST name both conflicting paths;
- explicit migration tests MAY construct dual-authoring fixtures only to prove
  rejection behavior;
- active repo-owned migrated configs MUST author sequence controls only in the
  new target hierarchy.

#### Scenario: Matching duplicate values still fail

- **GIVEN** a migrated config with both `custom.object_ordering: sorted` and
  `sample_factory.target_sequence.object_ordering: sorted`
- **WHEN** the config is loaded
- **THEN** loading fails before dataset construction
- **AND** the error names both paths.

### Requirement: Resolved config records normalized hierarchy

Resolved config artifacts SHALL record normalized hierarchy even when
implementation internals still use older module names.

Normative behavior:

- resolved config artifacts MUST include normalized `pipeline.id`;
- resolved config artifacts MUST include normalized `objective.id`;
- resolved config artifacts MUST include normalized `sample_factory` and
  `sample_factory.target_sequence`;
- active migrated configs MUST reject `objective.id: teacher_forcing` rather
  than resolving it as an alias;
- resolved config artifacts MUST NOT expose `custom.trainer_variant` as the
  active selector after migration.

#### Scenario: Resolved config rejects legacy research objective

- **GIVEN** an active migrated config using `objective.id: teacher_forcing`
- **WHEN** resolved config construction starts
- **THEN** loading fails before the artifact is written
- **AND** the error directs the author to `objective.id:
  research_teacher_forcing`.

### Requirement: Active config names are semantic and unambiguous

Active migrated config filenames and inheritance leaves SHALL use semantic axes
rather than ambiguous numbers or version-like labels.

Normative behavior:

- filenames SHOULD expose the relevant pipeline, objective, template, object
  ordering, object field order, prompt variant, and packing identity;
- filenames MUST NOT use `number`, `version`, `v1`, `v2`, or similar migration
  labels as the only distinction between active leaves;
- inheritance should keep shared base config pieces small and named by the
  semantic contract they provide;
- historical/archive files may keep old names as evidence, but active routing
  MUST NOT depend on ambiguous version-like names.

#### Scenario: Paired launch leaves are named by semantic differences

- **GIVEN** paired active Stage-1 Standard SFT launch configs for sorted and
  random object ordering
- **WHEN** docs/catalog routing names them after migration
- **THEN** the filenames expose the object ordering and template closure
  semantics
- **AND** they do not differ only by an ambiguous number or version-like suffix.
