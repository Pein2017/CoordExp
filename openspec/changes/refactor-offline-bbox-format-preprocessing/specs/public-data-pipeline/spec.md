## MODIFIED Requirements

### Requirement: Command Grammar and Semantics
The runner SHALL support the following command grammar:
`public_data/run.sh <dataset> <command> [runner-flags] [-- <passthrough-args>]`

The runner SHALL interpret only the defined `runner-flags` and SHALL pass all
remaining arguments after `--` verbatim to the underlying command
implementation (dataset plugin or shared python script).
For `all`, passthrough arguments after `--` SHALL be forwarded only to dataset
plugin steps (`download` and `convert`) to avoid ambiguity with shared
preprocessing options.

Supported commands:
- `download`: dataset-specific internet download into `public_data/<dataset>/raw/`
- `convert`: dataset-specific parsing/conversion into the CoordExp JSONL contract
- `rescale`: shared smart-resize into a preset directory
- `coord`: shared coord-token conversion within the preset directory
- `bbox-format`: shared offline derivation of non-canonical bbox-format branches
  from canonical preset artifacts
- `validate`: validate raw artifacts, canonical preset artifacts, and eligible
  derived bbox-format branches for contract/template compliance
- `all`: run `download -> convert -> rescale -> coord -> validate` in order
  (default preset resolution required); `all` remains the canonical workflow and
  SHALL NOT implicitly create non-canonical bbox-format branches
- `help`: print usage and exit 0

Supported runner flags:
- `--preset <name>`: preset directory name under `public_data/<dataset>/` (used
  by `rescale|coord|bbox-format|validate|all`)
- `--conda-env <name>`: override the default conda environment name (default: `ms`)
- `--skip-image-check`: skip image existence checks during validation (annotation-only flows)
- `--raw-only`: for `validate`, validate only raw artifacts
- `--preset-only`: for `validate`, validate only canonical preset artifacts and
  derived bbox-format branches under that preset

Supported environment interfaces:
- `PUBLIC_DATA_MAX_OBJECTS=<N>` enables optional max-object filtering for the
  `coord` command only.
- The runner MUST fail fast when `PUBLIC_DATA_MAX_OBJECTS` is set for
  `rescale`, `bbox-format`, `validate`, or `all`, with guidance to run a
  staged flow instead.
- The runner forwards `max_objects` into the pipeline planner only for `coord`;
  naming and effective preset resolution are owned by the planner.
- If `max_objects` is provided by multiple input surfaces and values disagree,
  execution MUST fail fast with an actionable error (no silent override).

Preset resolution rules:
- for commands that require a preset (`rescale`, `coord`, `bbox-format`) the
  user SHALL provide `--preset <name>`,
- for `all`, the runner SHALL:
  - use `--preset <name>` when provided, otherwise
  - use the dataset plugin default preset, otherwise
  - exit non-zero and print that a preset is required,
- for `validate`:
  - if `--raw-only` is set, the runner SHALL NOT require or resolve a preset,
  - otherwise (default behavior or `--preset-only`), the runner SHALL:
    - use `--preset <name>` when provided, otherwise
    - use the dataset plugin default preset, otherwise
    - exit non-zero and print that a preset is required.

Mutual exclusivity:
- for `validate`, `--raw-only` and `--preset-only` SHALL be mutually exclusive;
  setting both SHALL be an error.

#### Scenario: User runs an unknown command
- **GIVEN** the user runs `public_data/run.sh vg does-not-exist`
- **THEN** the runner exits non-zero and prints a usage message listing
  supported commands.

#### Scenario: User passes mutually-exclusive validate flags
- **GIVEN** the user runs
  `public_data/run.sh <dataset> validate --raw-only --preset-only`
- **THEN** the runner exits non-zero and prints that `--raw-only` and
  `--preset-only` cannot be used together.

#### Scenario: User runs a shared step without required inputs
- **GIVEN** `public_data/<dataset>/raw/train.jsonl` does not exist
- **WHEN** the user runs `public_data/run.sh <dataset> rescale --preset <preset>`
- **THEN** the runner exits non-zero and prints which required input file is
  missing.

#### Scenario: User runs coord-token conversion without preset inputs
- **GIVEN** `public_data/<dataset>/<preset>/train.jsonl` does not exist
- **WHEN** the user runs `public_data/run.sh <dataset> coord --preset <preset>`
- **THEN** the runner exits non-zero and prints which required preset file is
  missing.

#### Scenario: User runs bbox-format derivation without canonical preset inputs
- **GIVEN** `public_data/<dataset>/<preset>/train.jsonl` does not exist
- **WHEN** the user runs
  `public_data/run.sh <dataset> bbox-format --preset <preset> -- --bbox-format cxcy_logw_logh`
- **THEN** the runner exits non-zero and prints which required canonical source
  file is missing.

#### Scenario: User runs `all` without a preset and the dataset has no default
- **GIVEN** the dataset plugin does not define a default preset
- **WHEN** the user runs `public_data/run.sh <dataset> all`
- **THEN** the runner exits non-zero and prints that `--preset <name>` is
  required.

#### Scenario: User assumes passthrough args affect shared steps inside `all`
- **GIVEN** the user wants to tune shared preprocessing options for `rescale`,
  `coord`, or `bbox-format`
- **WHEN** the user runs
  `public_data/run.sh <dataset> all --preset <preset> -- --image-factor 64`
- **THEN** the runner prints a message explaining that `all` runs shared steps
  with runner defaults and that users should run the shared steps separately to
  tune their options.

#### Scenario: `all` does not implicitly create non-canonical bbox branches
- **GIVEN** the user runs `public_data/run.sh <dataset> all --preset <preset>`
- **WHEN** canonical preprocessing completes successfully
- **THEN** the runner produces canonical raw/rescale/coord/validate outputs only
- **AND** it does not silently create sibling derived preset roots such as
  `<preset>_cxcy_logw_logh/`.

#### Scenario: Conda environment is unavailable
- **GIVEN** `conda` or the requested conda environment is not available on the system
- **WHEN** the user runs a command that invokes python
- **THEN** the runner exits non-zero and prints a message indicating the missing
  dependency.

## ADDED Requirements

### Requirement: Shared offline bbox-format derivation step
The system SHALL provide a shared `bbox-format` step that derives
non-canonical model-facing bbox-format branches from canonical preset artifacts.

Normative behavior:
- the step SHALL invoke a shared python entrypoint via
  `conda run -n <conda-env> python ...`,
- the step SHALL consume canonical `train.jsonl` and `val.jsonl`
  (when present) from the preset root,
- the step SHALL accept only canonical `bbox_2d`-only preset sources for the
  first implementation surface,
- the step SHALL fail fast on any `poly` geometry or mixed-geometry source,
- the step SHALL require the desired bbox format to be passed explicitly
  through the shared derivation interface,
- the step SHALL write derived artifacts only under
  `public_data/<dataset>/<preset>_<format>/`,
- for each available split, the step SHALL emit:
  - `<split>.jsonl` containing offline-prepared norm1000 integer bbox tuples
    for the requested chart,
  - `<split>.coord.jsonl` containing the tokenized version of the same prepared
    tuples,
- the step SHALL fail fast if the canonical source artifacts are missing,
  malformed, or already marked as non-canonical.

#### Scenario: User derives a cxcy-logw-logh branch from a canonical preset
- **WHEN** the user runs
  `public_data/run.sh <dataset> bbox-format --preset <preset> -- --bbox-format cxcy_logw_logh`
- **THEN** the runner emits the derived branch under
  `<preset>_cxcy_logw_logh/`
- **AND** it leaves the canonical preset artifacts unchanged.

#### Scenario: Derivation rejects a non-canonical preset source
- **WHEN** the shared derivation step discovers that the source artifact is
  already stamped as a non-canonical prepared bbox format
- **THEN** the command fails fast
- **AND** it explains that only canonical preset artifacts are valid sources.

### Requirement: Validation covers derived bbox-format branches
The public-data validation flow SHALL validate eligible derived bbox-format
branches under a preset in addition to raw and canonical preset outputs.

Normative behavior:
- `validate --preset <preset>` SHALL inspect canonical preset artifacts and any
  discovered sibling derived preset roots for that canonical preset, including
  `<preset>_cxcy_logw_logh/` when present,
- validation for derived branches SHALL check:
  - required branch files for both the numeric split JSONL and coord-token
    split JSONL surfaces,
  - manifest presence and schema,
  - record-level prepared bbox-format metadata,
  - record-level slot-order metadata,
  - that numeric split JSONL uses norm1000 integer slots on the same lattice as
    the corresponding coord-token split JSONL,
  - geometry arity and coord-token/norm lattice validity,
- validation SHALL fail fast on missing manifest/provenance fields or geometry
  contract violations in derived branches.

#### Scenario: Validate inspects a derived branch
- **WHEN** the user runs `public_data/run.sh <dataset> validate --preset <preset>`
- **AND** `<preset>_cxcy_logw_logh/` exists under that canonical preset root
- **THEN** validation includes the derived branch files
- **AND** reports derived-branch provenance/contract errors if present.

#### Scenario: Validate fails on missing branch manifest
- **WHEN** a derived bbox-format branch is present but `manifest.json` is missing
- **THEN** validation fails fast
- **AND** it reports the missing derived-branch provenance artifact.
