## ADDED Requirements

### Requirement: Standard Stage-1 detection-sequence SFT is a protected active lane

The active Standard SFT lane for object/bbox target sequences SHALL be selected
with `pipeline.id: stage1_standard_sft`.

Normative behavior:

- this lane uses `objective.id: standard_ce`;
- this lane may use `detection_template.id` for compact template identity;
- this lane uses `sample_factory.id: detection_sequence` for current object/bbox
  sequence materialization;
- this lane uses `sample_factory.target_sequence` for object ordering, field
  order, bbox format, coordinate surface, and strict parsing;
- sorted/random ordering and object/bbox closure launch prep from main is
  active research work and MUST NOT be classified as legacy merely because it
  touches detection sequence code;
- active Standard SFT launch-prep leaves from main are protected first-slice
  migration inputs when they launch sorted/random ordering with object and bbox
  closure;
- optional geometry or soft-CE auxiliaries may be explicit additions under the
  Standard SFT objective.

#### Scenario: Standard SFT object/bbox config resolves

- **GIVEN** an active Stage-1 config with `pipeline.id: stage1_standard_sft`
- **AND** `objective.id: standard_ce`
- **AND** `sample_factory.id: detection_sequence`
- **WHEN** the config is loaded
- **THEN** the Standard SFT path is selected
- **AND** no research teacher-forcing token tracing is required.

### Requirement: Research teacher forcing owns fine-grained Stage-1 objective behavior

Fine-grained Stage-1 detection teacher-forcing research SHALL be selected with
`pipeline.id: stage1_research_teacher_forcing` and
`objective.id: research_teacher_forcing`.

Normative behavior:

- this lane owns token role/value/span tracing, valid sets, branch state,
  force/weight policies, and exact label/logit positions;
- recursive-detection / ET-RMP behavior remains preserved comparator and
  ablation lineage under this research lane;
- the old public name `objective.id: teacher_forcing` MUST be rejected for
  active migrated configs;
- the catalog-canonical
  `configs/stage1/detection_teacher_forcing/prod/compact_support2.yaml` route
  is a protected first-slice migration input and must migrate to
  `objective.id: research_teacher_forcing`;
- Standard SFT MUST NOT depend on research teacher-forcing target IR sidecars;
- preserving ET-RMP does not make it the default Standard SFT route.

#### Scenario: ET-RMP remains preserved research comparator

- **GIVEN** an ET-RMP config or infer/eval comparator route
- **WHEN** the SFT hierarchy migration is implemented
- **THEN** ET-RMP remains inspectable or runnable as preserved comparator
  lineage
- **AND** Standard SFT remains selected separately through
  `pipeline.id: stage1_standard_sft`.

#### Scenario: Canonical compact support route migrates to research teacher forcing

- **GIVEN** docs/catalog routing points to
  `configs/stage1/detection_teacher_forcing/prod/compact_support2.yaml`
- **WHEN** active Stage-1 research teacher-forcing configs are migrated
- **THEN** the migrated route uses
  `pipeline.id: stage1_research_teacher_forcing`
- **AND** it uses `objective.id: research_teacher_forcing`
- **AND** `objective.id: teacher_forcing` is rejected as active authoring.

### Requirement: Stage-1 packing policy distinguishes Standard SFT from research teacher forcing

Stage-1 packing SHALL be treated as first-class for Standard SFT and as a
required long-term capability for research teacher forcing.

Normative behavior:

- Standard SFT may use the efficient static/packed path when its data path
  supports it;
- research teacher forcing MUST reject packing until exact sidecar
  atom-position remapping is implemented and tested;
- research teacher forcing MUST NOT be documented as permanently unpacked;
- packing/cache fingerprints MUST include pipeline, objective, template, target
  sequence, and hard length identity.

#### Scenario: Research teacher forcing packing remains fail-fast before remap

- **GIVEN** a research teacher-forcing config with packing enabled before
  remapping support exists
- **WHEN** config validation runs
- **THEN** validation fails before training
- **AND** the error names the missing exact atom-position remapping contract.
