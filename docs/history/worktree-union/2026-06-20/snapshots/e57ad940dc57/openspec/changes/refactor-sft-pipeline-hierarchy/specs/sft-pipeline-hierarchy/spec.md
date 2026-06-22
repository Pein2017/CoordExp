## ADDED Requirements

### Requirement: Active training family is selected by top-level pipeline id

Active repo-owned training configs SHALL select the training family with a
top-level `pipeline.id` mapping.

Normative behavior:

- `pipeline.id` MUST be one of:
  - `stage1_standard_sft`
  - `stage1_research_teacher_forcing`
  - `stage2_rollout_correction`
- active target-hierarchy configs MUST NOT author `custom.trainer_variant` as a
  selector;
- active target-hierarchy configs MUST NOT author scalar `pipeline_id`;
- active target-hierarchy configs MUST NOT expose `surface.id` as public config
  vocabulary;
- docs/catalog routing MUST identify the pipeline-selection owner with
  pipeline vocabulary rather than surface vocabulary.

#### Scenario: Standard SFT pipeline id loads

- **GIVEN** an active config with `pipeline.id: stage1_standard_sft`
- **WHEN** the training config is loaded
- **THEN** the config resolves to the Standard SFT training family
- **AND** no `custom.trainer_variant` selector is required.

#### Scenario: Custom trainer variant is rejected as active selector

- **GIVEN** an active target-hierarchy config with
  `custom.trainer_variant: stage2_rollout_correction`
- **WHEN** the training config is loaded
- **THEN** loading fails before trainer construction
- **AND** the error directs the author to `pipeline.id:
  stage2_rollout_correction`.

#### Scenario: Surface id is not public config language

- **GIVEN** an active target-hierarchy config with `surface.id:
  stage1_standard_sft`
- **WHEN** the training config is loaded
- **THEN** loading fails before trainer construction
- **AND** the error directs the author to `pipeline.id`.

### Requirement: Sequence materialization is owned by sample factory target sequence

Active repo-owned Stage-1 target-sequence materialization SHALL author
sequence-format controls under `sample_factory.target_sequence`.

Normative behavior:

- the current object/bbox task family uses `sample_factory.id:
  detection_sequence`;
- `sample_factory.target_sequence.task_family` MUST be `detection` for the
  current object/bbox route;
- `sample_factory.target_sequence` MUST own:
  - `object_ordering`
  - `object_field_order`
  - `bbox_format`
  - `coordinate_surface`
  - `strict_parse`
- `detection_template.id` remains the stable top-level template identity and
  MUST NOT move under `sample_factory.target_sequence` in this change;
- active configs MUST migrate historical `custom.detection_template_id` values
  to top-level `detection_template.id`;
- active configs MUST migrate historical `custom.detection_sequence_format`
  intent into the combination of `detection_template.id` and
  `sample_factory.id`;
- active target-hierarchy configs MUST NOT author the old `custom.*` sequence
  paths for these fields;
- active target-hierarchy configs MUST author parser strictness only through
  `sample_factory.target_sequence.strict_parse`;
- active target-hierarchy configs MUST NOT author duplicate parser strictness
  through `detection_template.strict_parse` or separate evaluation parser
  strictness paths;
- old and new sequence-control paths MUST fail when authored together, even if
  their values match, except inside explicit migration tests.

#### Scenario: Target sequence owns field order

- **GIVEN** an active Stage-1 Standard SFT config
- **AND** it authors `sample_factory.target_sequence.object_field_order:
  geometry_first`
- **WHEN** the config is loaded
- **THEN** geometry-first compact row materialization uses that value
- **AND** the same resolved value appears in cache and provenance identity.

#### Scenario: Detection template identity remains top-level

- **GIVEN** an active config with `detection_template.id:
  compact_object_box_closed`
- **WHEN** the config is loaded
- **THEN** template family and structural-row identity resolve from
  `detection_template.id`
- **AND** no `sample_factory.target_sequence.template_id` key is required.

#### Scenario: Historical template id path is rejected

- **GIVEN** an active target-hierarchy config with
  `custom.detection_template_id: compact_object_box_closed`
- **WHEN** the config is loaded
- **THEN** loading fails before sample construction
- **AND** the error directs the author to top-level `detection_template.id`.

#### Scenario: Dual sequence authoring fails

- **GIVEN** an active target-hierarchy config that authors both
  `custom.object_field_order: desc_first` and
  `sample_factory.target_sequence.object_field_order: desc_first`
- **WHEN** the config is loaded
- **THEN** loading fails before sample construction
- **AND** the error identifies the duplicate old and new sequence-control paths.

#### Scenario: Catalog canonical flat research-TF config is migrated explicitly

- **GIVEN** the catalog-canonical Stage-1 research teacher-forcing config
  `configs/stage1/detection_teacher_forcing/prod/compact_support2.yaml`
- **WHEN** it is migrated into the target hierarchy
- **THEN** `data.object_ordering: random_permutation` becomes
  `sample_factory.target_sequence.object_ordering: random_permutation`
- **AND** `prompt.prompt_variant_enabled: true` becomes
  `prompt.variant: coco_80`
- **AND** flat `detection_template.coordinate_surface`,
  `detection_template.bbox_format`, and `detection_template.strict_parse`
  become fields under `sample_factory.target_sequence`
- **AND** `objective.id` is changed to `research_teacher_forcing`, not kept as
  `teacher_forcing`.

#### Scenario: Duplicate parser strictness fails

- **GIVEN** an active target-hierarchy config with
  `sample_factory.target_sequence.strict_parse: true`
- **AND** it also authors `detection_template.strict_parse: true`
- **WHEN** the config is loaded
- **THEN** loading fails before target rendering
- **AND** the error names `sample_factory.target_sequence.strict_parse` as the
  single active parser-policy source.

### Requirement: Standard SFT uses public objective id standard_ce

Standard Stage-1 SFT SHALL author ordinary assistant-label cross entropy through
`objective.id: standard_ce`.

Normative behavior:

- `standard_ce` is the durable public objective id for pure CE Standard SFT;
- optional geometry or soft-CE auxiliaries MAY be authored beneath the
  `standard_ce` objective;
- optional auxiliaries MUST NOT turn the run into the research teacher-forcing
  lane by themselves;
- `token_ce` MAY remain an implementation module, loss term, metric component,
  or internal symbol;
- `token_ce` MUST NOT be required as the public objective id for active
  Standard SFT configs unless a later OpenSpec promotes it.

#### Scenario: Standard CE loads without research tracing

- **GIVEN** a config with `pipeline.id: stage1_standard_sft`
- **AND** `objective.id: standard_ce`
- **WHEN** the training runtime is resolved
- **THEN** ordinary assistant-label CE is selected
- **AND** fine-grained research teacher-forcing tracing is not required.

#### Scenario: Standard CE may carry disabled auxiliaries

- **GIVEN** a config with `objective.id: standard_ce`
- **AND** `objective.auxiliaries.coord_soft_ce.enabled: false`
- **WHEN** the config is loaded
- **THEN** the disabled auxiliary remains explicit
- **AND** the public objective identity remains `standard_ce`.

### Requirement: Research teacher forcing uses one public objective family

Fine-grained Stage-1 research teacher forcing SHALL author
`objective.id: research_teacher_forcing`.

Normative behavior:

- `research_teacher_forcing` owns token role/value/span tracing, branch state,
  valid sets, force/weight policies, and exact label/logit position metadata;
- research tactics SHOULD be internal weighted terms under the
  `research_teacher_forcing` objective rather than new public pipeline ids;
- `teacher_forcing` is not the desired long-term public objective id;
- active migrated configs MUST reject `objective.id: teacher_forcing` rather
  than accepting it as a migration alias;
- ET-RMP and recursive-detection behavior remain preserved research/comparator
  behavior, not the default Standard SFT route.

#### Scenario: Research teacher forcing loads weighted terms

- **GIVEN** a config with `pipeline.id: stage1_research_teacher_forcing`
- **AND** `objective.id: research_teacher_forcing`
- **AND** internal weighted terms are authored under `objective.terms`
- **WHEN** the training runtime is resolved
- **THEN** fine-grained token tracing and term composition are enabled
- **AND** no new public pipeline id is required for each term.

#### Scenario: Legacy teacher_forcing objective is rejected

- **GIVEN** an active migrated config with `objective.id: teacher_forcing`
- **WHEN** config loading runs
- **THEN** loading fails before objective construction
- **AND** the error directs the author to `objective.id:
  research_teacher_forcing`.

### Requirement: Provenance and cache identity include normalized training hierarchy

Resolved configs, cache fingerprints, manifests, and training provenance SHALL
record the effective normalized training hierarchy without removing existing
non-hierarchy discriminators.

Normative behavior:

- the identity MUST include:
  - `pipeline.id`
  - `objective.id`
  - `detection_template.id`
  - `sample_factory.id`
  - `sample_factory.target_sequence.object_ordering`
  - `sample_factory.target_sequence.object_field_order`
  - `sample_factory.target_sequence.bbox_format`
  - `sample_factory.target_sequence.coordinate_surface`
  - `sample_factory.target_sequence.strict_parse`
  - resolved train prompt variant
  - resolved prompt hash
  - tokenizer identity
  - chat-template identity
  - effective packing length;
- these fields are additive to the current implementation's existing
  discriminators, including dataset/source identity, train/eval split, sample
  limits, tokenizer/model identity, system prompt text, prompt hash, packing
  knobs, dataloader shuffle, and offline image-pixel budget where those keys
  already exist;
- this change MUST NOT add a new image-root or view-store discriminator unless
  that discriminator is already present in the current key set being extended;
- encoded-sample cache fingerprints MUST change when any of those fields
  changes;
- static-packing cache fingerprints MUST change when any of those fields
  changes;
- Stage-2 provenance MUST record top-level `pipeline.id` without changing the
  existing Stage-2 internal correction pipeline objective namespace;
- `pipeline.id: stage2_rollout_correction` MUST be sufficient to build Stage-2
  policy provenance even when `custom.trainer_variant` is absent;
- migrated active Stage-2 provenance MUST NOT emit `trainer_variant` as a
  compatibility selector field.

#### Scenario: Prompt variant change invalidates encoded cache

- **GIVEN** two Stage-1 configs that differ only by resolved train prompt variant
- **WHEN** encoded-sample cache fingerprints are computed
- **THEN** the fingerprints differ.

#### Scenario: Existing dataset discriminator remains active

- **GIVEN** two configs with identical normalized hierarchy
- **AND** different training JSONL source identity or sample-limit identity
- **WHEN** encoded-sample and static-packing cache fingerprints are computed
- **THEN** the fingerprints differ.

### Requirement: Token embedding adapter is high-level and independent

Compact structural token embedding adaptation SHALL be authored through a
high-level `token_embeddings_adapter` namespace.

Normative behavior:

- `token_embeddings_adapter` derives required structural rows from
  `detection_template.id`;
- `token_embeddings_adapter` MUST NOT infer required row sets from
  `sample_factory.target_sequence.object_field_order`;
- flat `token_rows` authoring is deprecated by this program and MUST be
  rejected after migration for active configs;
- `custom.token_embeddings_adapter` is deprecated by this program and MUST be
  rejected after migration for active configs.

#### Scenario: Token row fields are rejected after adapter migration

- **GIVEN** an active migrated config that authors flat `token_rows`
- **WHEN** the config is loaded
- **THEN** loading fails before adapter setup
- **AND** the error directs the author to `token_embeddings_adapter`.

#### Scenario: Field-order change invalidates encoded cache

- **GIVEN** two configs that differ only by
  `sample_factory.target_sequence.object_field_order`
- **WHEN** encoded-sample cache fingerprints are computed
- **THEN** the fingerprints differ.

#### Scenario: Pipeline id appears in run provenance

- **GIVEN** a Stage-2 rollout-correction run
- **WHEN** resolved config and policy provenance are written
- **THEN** they include `pipeline.id: stage2_rollout_correction`
- **AND** Stage-2 internal objective provenance still uses
  `stage2_rollout_correction.pipeline.objective[]`.

#### Scenario: Pipeline id builds Stage-2 policy provenance

- **GIVEN** a Stage-2 config with `pipeline.id: stage2_rollout_correction`
- **AND** no `custom.trainer_variant`
- **WHEN** effective runtime, pipeline manifest, run metadata, and experiment
  manifest are written
- **THEN** each manifest carrier includes Stage-2 policy provenance
- **AND** that provenance records assignment strategy, duplicate-filter policy,
  object ordering policy, rollout template family, invalid-rollout policy, and
  fallback loss weight.

### Requirement: Strict parser policy has one normalized training source

Active Stage-1 target-hierarchy configs SHALL author parser strictness through
`sample_factory.target_sequence.strict_parse`.

Normative behavior:

- target rendering, strict parsing, Stage-1 eval callbacks, eval/infer artifact
  materialization, cache fingerprints, and provenance MUST consume the same
  normalized strict-parse value;
- active configs MUST fail fast if they author conflicting or duplicate
  strict-parse/parser-policy paths;
- legacy `detection_template.strict_parse` and separate evaluation parser
  strictness authoring MAY appear only in migration/rejection fixtures or
  derived resolved fields after this migration.

#### Scenario: Conflicting strict parser policy fails

- **GIVEN** an active target-hierarchy config with
  `sample_factory.target_sequence.strict_parse: true`
- **AND** a legacy authored parser strictness path sets the effective value to
  false
- **WHEN** config loading runs
- **THEN** loading fails before dataset construction
- **AND** the error names the conflicting parser-policy paths.

### Requirement: Stage-1 research teacher forcing packing requires exact remapping

Research teacher forcing SHALL treat Stage-1 packing as a long-term supported
requirement, but it MUST reject packed execution until exact atom-position
remapping is implemented and tested.

Normative behavior:

- Standard SFT owns the high-throughput packed-forward path;
- research teacher forcing MAY reject packing while atom/span remapping is not
  implemented;
- research teacher forcing MUST NOT be documented as inherently unpacked;
- packed research teacher forcing MUST prove label/logit positions, token-role
  spans, valid sets, and force/weight policies are remapped exactly across
  packed segments;
- the migration MUST preserve the current exact-packing-mapping guard and packed
  target-IR sidecar rejection until the remapper proves segment offsets,
  `encoded_len`, `cu_seq_lens`, sample ids, and image-grid slices remain aligned;
- Stage-2 rollout correction keeps its existing post-rollout packing contract
  and does not become a Stage-1 SFT subtype.

#### Scenario: Research teacher forcing rejects unimplemented packing

- **GIVEN** `pipeline.id: stage1_research_teacher_forcing`
- **AND** training packing is enabled before exact atom-position remapping is
  implemented
- **WHEN** the config is loaded
- **THEN** loading fails before training starts
- **AND** the error states that packed research teacher forcing requires exact
  atom-position remapping tests.

#### Scenario: Packed research teacher forcing proves remapped positions

- **GIVEN** packed research teacher-forcing support is implemented
- **WHEN** a packed sample contains multiple original examples
- **THEN** label/logit positions, token roles, valid sets, and force/weight
  policies are remapped to the packed physical positions exactly.
