# training-config-hierarchy Specification (Delta)

## Purpose

Define the canonical reusable YAML hierarchy for training configs so shared
concern groups are explicit, repo-global defaults stay truly global, and leaf
profiles remain easy to audit without weakening the existing typed config
contract. This hierarchy is intended to scale across COCO, LVIS, and future
prepared single-dataset families under the repo's current single-dataset
training default.

## ADDED Requirements

### Requirement: Repo-global training base MUST remain dataset-agnostic and prompt-agnostic

The repo-global training base under `configs/base.yaml` SHALL define only
truly global defaults and MUST NOT silently author dataset identity or prompt
identity for downstream training runs.

Normative behavior:

- `configs/base.yaml` MAY define shared defaults for:
  - model settings
  - template settings
  - training defaults
  - data-loader defaults
- `configs/base.yaml` MUST NOT define dataset-specific training identity such as:
  - `custom.train_jsonl`
  - `custom.val_jsonl`
  - dataset-specific `custom.object_ordering`
  - dataset-specific prompt variants
- downstream training configs that need dataset or prompt identity MUST receive
  those values from dedicated shared facets or explicit leaf overrides.

#### Scenario: Global base does not silently inject dataset paths

- **GIVEN** a downstream training config that extends `configs/base.yaml`
- **WHEN** the config does not also extend a dataset facet or explicitly author
  dataset paths
- **THEN** the resolved config does not silently inherit a dataset-specific
  `custom.train_jsonl` / `custom.val_jsonl` pair from the repo-global base.

### Requirement: Training hierarchy MAY compose reusable facets using existing extends semantics

Training configs SHALL support reusable YAML facets for shared concern groups
using the existing `extends` / deep-merge model rather than a new templating or
schema system.

Normative behavior:

- reusable shared facets MAY be introduced for concern groups such as:
  - dataset identity
  - prompt identity
  - stage-local objective bundles
  - stage-local observability bundles
- those facet patterns MUST remain open-ended for future prepared dataset
  families; COCO- and LVIS-specific facets are examples, not a closed allowlist.
- the preferred authored hierarchy for canonical configs is:
  - universal base
  - stage-wise base
  - shared/common reusable package settings
  - specialized experiment leaf
- canonical repo-owned training configs SHOULD stay within a maximum authored
  semantic depth of 4 levels from universal base to specialized experiment
  leaf.
- derivative wrappers such as smoke/runtime-limit overlays MAY sit on top of a
  specialized experiment leaf, but they MUST stay narrow and do not count as an
  additional shared/common semantic layer.
- facet composition MUST remain within the existing typed top-level sections
  already supported by the training schema.
- the hierarchy refactor MUST NOT require new top-level config sections solely
  for reuse.
- the hierarchy refactor MUST preserve current strict unknown-key fail-fast
  behavior.

#### Scenario: Shared facet composes without adding a new schema surface

- **GIVEN** a training leaf that extends a shared dataset facet and a shared
  prompt facet
- **WHEN** the config is loaded through the current typed loader
- **THEN** the resolved config is accepted without introducing new top-level
  schema groups
- **AND** unknown keys still fail fast under the existing strict parser.

#### Scenario: Future dataset facet uses the same hierarchy contract

- **GIVEN** a new prepared single-dataset family that follows the existing
  training JSONL/image contract
- **WHEN** a new shared dataset facet is authored for that family using the
  existing typed keys
- **THEN** the facet composes through the same `extends` hierarchy model
- **AND** adding that dataset family does not require new top-level schema
  sections or dataset-name conditionals in the loader.

#### Scenario: Canonical config stays within the preferred authored depth budget

- **GIVEN** a canonical repo-owned training config
- **WHEN** its inheritance chain is inspected from universal base to specialized
  experiment leaf
- **THEN** the authored hierarchy stays within the preferred 4-level semantic
  depth
  budget
- **AND** reusable shared/common package settings may still be composed as
  sibling parents within that layer.

### Requirement: Stage-local runtime bases MUST own runtime defaults only

Stage-local runtime bases SHALL remain explicit, but they MUST narrow their
ownership to stage-specific runtime behavior rather than bundling unrelated
dataset or prompt identity.

Normative behavior:

- `configs/stage1/sft_base.yaml` MUST remain the Stage-1 runtime base.
- `configs/stage2_two_channel/base.yaml` MUST remain the Stage-2 two-channel
  runtime base.
- for canonical migrated families, both stage-local runtime bases SHOULD layer
  on top of `configs/base.yaml` so the universal base remains truly universal.
- those stage-local bases MAY define stage-specific runtime defaults, but MUST
  NOT be the hidden long-term home for reusable dataset identity or reusable
  prompt identity.
- once a shared dataset facet or shared prompt facet is introduced for a leaf
  family, the corresponding stage-local base MUST NOT continue to silently
  select that family's dataset paths or prompt variant.

#### Scenario: Stage base does not hide reusable dataset identity

- **GIVEN** a representative Stage-1 or Stage-2 leaf config
- **WHEN** dataset identity is authored through a shared dataset facet
- **THEN** the stage-local base does not need to carry the same dataset path
  contract redundantly.
- **AND** the migrated leaf family does not continue to depend on a hidden
  stage-base dataset or prompt default.

### Requirement: Legacy fusion surfaces MUST be shut down coherently with the hierarchy cleanup

The canonical training-config hierarchy SHALL target the repo's single-dataset
training default and MUST NOT preserve legacy fusion authoring as a hidden
exception to base-purity rules.

Normative behavior:

- legacy example config surfaces under `configs/fusion/` MAY remain in-tree as
  dormant reference assets.
- legacy `custom.fusion_config` capability in schema/runtime/tests/docs MUST be
  disabled in the same cleanup rather than preserved as an active side contract.
- multi-dataset or fusion-first config authoring is outside this hierarchy
  contract unless a separate future change explicitly reintroduces it.
- base-purity or facet-migration rules MUST NOT leave `configs/fusion/` as an
  unreviewed consumer of hidden dataset defaults.

#### Scenario: Fusion is shut down coherently while dormant assets remain

- **GIVEN** a hierarchy cleanup that removes dataset defaults from
  `configs/base.yaml`
- **WHEN** legacy fusion config surfaces are reviewed
- **THEN** the `configs/fusion/` folder may remain only as dormant reference
  material
- **AND** `custom.fusion_config` is disabled across schema, runtime, docs,
  specs, and tests
- **AND** the cleanup does not leave them as silent consumers of the old hidden
  base defaults.

### Requirement: Artifact path composition MUST support reusable roots plus one explicit subdir

Canonical training configs SHALL be able to configure artifact roots once and
author the shared directory suffix only once, while keeping the resolved
`training.output_dir` / `training.logging_dir` behavior stable.

Normative behavior:

- under the existing `training` section, repo-owned internal keys MAY define:
  - `training.output_root`
  - `training.logging_root`
  - `training.artifact_subdir`
- when those keys are present, the repo-owned loader SHOULD derive:
  - `training.output_dir = training.output_root / training.artifact_subdir`
  - `training.logging_dir = training.logging_root / training.artifact_subdir`
  before downstream trainer initialization.
- canonical migrated repo-owned configs MUST use one explicit
  `training.artifact_subdir` instead of duplicating the same suffix in both
  `training.output_dir` and `training.logging_dir`.
- canonical migrated repo-owned configs MUST NOT directly author duplicated
  `training.output_dir` / `training.logging_dir` as a parallel legacy path.

#### Scenario: Shared artifact suffix is authored once

- **GIVEN** a canonical training leaf that sets `training.artifact_subdir`
- **AND** reusable roots for output and tensorboard logs
- **WHEN** the config is materialized through the repo-owned loader
- **THEN** the resolved `training.output_dir` and `training.logging_dir` share
  the same authored suffix
- **AND** the config does not need to duplicate that suffix in two separate leaf
  fields.

### Requirement: High-signal leaf identity MUST remain explicit

Training leaf configs SHALL remain directly auditable for high-signal run
identity and MUST NOT rely on generated or opaque conventions in the first
hierarchy optimization slice.

Normative behavior:

- representative training leaves MUST continue to author explicit values for:
  - `model.model`
  - `training.run_name`
  - `training.artifact_subdir`
- the hierarchy optimization MUST NOT replace those fields with mandatory
  opaque naming conventions in this slice.

#### Scenario: Leaf still exposes run identity after facet migration

- **GIVEN** a training leaf migrated onto the new facet layout
- **WHEN** an operator opens the leaf YAML
- **THEN** the model path, run identity, and shared artifact suffix remain
  visible in the leaf
- **AND** they do not need to reconstruct those values from generated naming
  conventions.

### Requirement: List-valued training bundles MUST have a single hierarchy owner and MUST NOT overlap across parents

The config hierarchy MUST assign one clear owner in the inheritance chain to
every list-valued bundle whose semantics depend on ordering or full
replacement.

Normative behavior:

- the hierarchy refactor MUST NOT assume semantic list merging for list-valued
  sections.
- list-valued training bundles such as objective or diagnostic manifests MUST be
  owned by one facet or one leaf at a time.
- two reusable parents in the same inheritance chain MUST NOT each author the
  same list-valued config key and rely on merge precedence to produce the final
  bundle.
- a later facet MAY intentionally replace a whole list-valued bundle, but the
  hierarchy MUST NOT split one intended bundle across multiple parents and
  expect additive merge behavior.

#### Scenario: Objective list is owned by one facet

- **GIVEN** a training config whose objective manifest is authored in a shared
  facet
- **WHEN** the leaf composes that facet with other shared parents
- **THEN** the resolved objective list comes from one explicit owner
- **AND** the hierarchy does not rely on implicit concatenation across parents.

#### Scenario: Overlapping list ownership is rejected by the repo hierarchy contract

- **GIVEN** two reusable parents in the same inheritance chain each author the
  same list-valued config key
- **WHEN** the hierarchy contract is validated through repo-owned checks
- **THEN** the overlap is treated as an invalid hierarchy pattern
- **AND** the config tree does not rely on silent last-writer-wins replacement
  for shared list assembly.

### Requirement: Hierarchy refactors MUST prove resolved-config parity on representative leaves

A training-config hierarchy refactor SHALL be treated as behavior-preserving
only when representative leaf configs resolve to the same intended runtime
contract after migration.

Normative behavior:

- representative parity coverage MUST include at minimum:
  - minimal composed fixture coverage for every newly introduced shared dataset
    facet and shared prompt facet
  - one Stage-1 LVIS production leaf
  - one Stage-1 LVIS smoke leaf
  - one Stage-1 COCO leaf
  - one Stage-2 LVIS production leaf
  - one canonical Stage-2 COCO prod leaf
- parity coverage MUST expand beyond the minimum representative set for every
  canonical leaf family that consumes a newly introduced shared dataset or
  prompt facet.
- the hierarchy contract MUST remain reusable for future prepared dataset
  families without requiring new loader/schema branches.
- parity checks MUST verify the resolved values for:
  - dataset paths
  - `custom.offline_max_pixels`
  - `template.max_pixels`
  - prompt variant / field order
  - object ordering
  - run identity
  - `training.output_dir`
  - `training.logging_dir`
- raw ownership checks MUST verify that canonical migrated leaves still author:
  - `model.model`
  - `training.run_name`
  - `training.artifact_subdir`
- named regression suites MUST be stabilized or partitioned before they are used
  as parity evidence when unrelated pre-existing failures already exist.
- the hierarchy refactor MUST preserve existing strict config validation and
  canonical Stage-2 profile loading behavior while those representative leaves
  are migrated.

#### Scenario: Representative leaves preserve their resolved contract

- **GIVEN** representative training leaves migrated onto the new facet layout
- **WHEN** they are loaded through `ConfigLoader.load_materialized_training_config(...)`
- **THEN** their resolved dataset, prompt, ordering, and run-identity contract
  matches the intended pre-migration values
- **AND** the existing strict-config and canonical-profile tests continue to
  pass.
