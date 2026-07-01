## ADDED Requirements

### Requirement: Training configs SHALL support a first-class experiment metadata section
The training config schema SHALL support an optional top-level `experiment`
section for authored run intent that is separate from `training.*`,
`custom.*`, and provenance artifacts.

Normative behavior:

- the `experiment` section MUST accept structured, human-authored fields for:
  - run purpose,
  - hypothesis,
  - key deviations from baseline,
  - important runtime-setting notes,
  - optional comments or observations,
- the `experiment` section MAY additionally carry stable grouping aids such as
  a short title, tags, or a baseline reference,
- unknown `experiment.*` keys MUST fail fast with dotted-path reporting,
- when present, the resolved `experiment` payload MUST be preserved in
  `resolved_config.json` as part of the canonical resolved training config.

#### Scenario: Authored experiment narrative survives config resolution
- **GIVEN** a training config that authors `experiment.purpose`,
  `experiment.hypothesis`, and `experiment.key_deviations`
- **WHEN** the config is loaded and materialized for training
- **THEN** those fields remain available in the resolved training config
- **AND** the loader does not require the operator to encode the same meaning in
  `training.run_name` or `training.artifact_subdir`.

### Requirement: Training runs SHALL emit a unified experiment manifest artifact
Training bootstrap SHALL emit `experiment_manifest.json` under the run output
directory as the primary overview artifact for retrospective analysis.

Normative behavior:

- `experiment_manifest.json` MUST contain:
  - run identity (`config_path`, `base_config_path`, `run_name`, `output_dir`,
    `dataset_seed`),
  - the authored `experiment` payload or an explicit empty/null authored block
    when the config does not provide one,
  - a hard-runtime summary derived from executed runtime artifacts,
  - a provenance summary derived from run-level provenance capture,
  - artifact pointers for the authoritative run files,
- the artifact MUST be emitted on rank 0 before training proceeds past normal
  bootstrap artifact creation,
- the artifact MUST be JSON and human-readable (`indent=2` style or
  equivalent).

#### Scenario: Operator gets one run-level entrypoint for hard and soft context
- **GIVEN** a successful training bootstrap on rank 0
- **WHEN** the output directory is inspected
- **THEN** `experiment_manifest.json` is present
- **AND** an operator can recover the run’s authored purpose, runtime summary,
  provenance summary, and authoritative artifact locations from that one file.

### Requirement: Experiment manifests SHALL complement rather than redefine authoritative artifacts
The new experiment metadata surfaces MUST complement the existing config,
runtime, pipeline, and provenance artifacts rather than silently replacing
their authority.

Normative behavior:

- `resolved_config.json` MUST remain the authoritative exact-config record,
- `effective_runtime.json` MUST remain the authoritative executed-runtime
  record,
- `pipeline_manifest.json` MUST remain the authoritative pipeline-structure
  record,
- `run_metadata.json` MUST remain the low-level provenance artifact for git,
  upstream dependency, launcher, and cache metadata,
- `experiment_manifest.json` MUST summarize and point to those artifacts rather
  than redefining their semantics.

#### Scenario: Experiment prose does not replace runtime truth
- **GIVEN** a run whose authored experiment notes mention important runtime
  settings
- **WHEN** an operator needs the exact executed value of a runtime knob
- **THEN** the operator can recover that exact value from
  `effective_runtime.json` or `resolved_config.json`
- **AND** `experiment_manifest.json` does not become a conflicting source of
  truth.

### Requirement: Legacy training configs SHALL remain runnable during migration
The training runtime SHALL continue to accept configs that do not yet author a
top-level `experiment` section.

Normative behavior:

- absence of `experiment` MUST NOT fail config loading,
- runs launched from legacy configs MUST still emit `experiment_manifest.json`,
- in that case the manifest MUST make the absence explicit via a null or empty
  authored experiment payload rather than inventing synthetic prose.

#### Scenario: Legacy config produces a partial experiment manifest
- **GIVEN** a training config that omits the top-level `experiment` section
- **WHEN** the run starts on rank 0
- **THEN** `experiment_manifest.json` is still emitted
- **AND** the hard-runtime, provenance, and artifact-pointer sections are
  populated
- **AND** the authored experiment section is explicitly absent rather than
  auto-generated from naming conventions.
