# public-data-pipeline Specification

## Purpose
Define the public-data preprocessing pipeline contract for emitting CoordExp JSONL datasets with validated geometry/schema and reproducible transforms.

## Requirements
### Requirement: Unified Shell Entrypoint and Working Directory
The system SHALL provide a unified shell entrypoint at `public_data/run.sh` to run public dataset preparation pipelines with a consistent interface across datasets.

The runner SHALL be executed from the CoordExp repository root (the directory that contains `public_data/` and `src/`).

#### Scenario: User runs unified entrypoint from repo root
- **GIVEN** the user is in the repo root (`.`)
- **WHEN** the user runs `public_data/run.sh <dataset> <command> [args...]`
- **THEN** the runner executes the requested command for the specified dataset and exits non-zero on failure.

#### Scenario: User runs unified entrypoint from a non-root directory
- **GIVEN** the user is not in the repo root
- **WHEN** the user runs `public_data/run.sh <dataset> <command>`
- **THEN** the runner exits non-zero and prints a message instructing the user to run from the repo root.

### Requirement: Command Grammar and Semantics
The runner SHALL support the following command grammar:
`public_data/run.sh <dataset> <command> [runner-flags] [-- <passthrough-args>]`

The runner SHALL interpret only the defined `runner-flags` and SHALL pass all remaining arguments after `--` verbatim to the underlying command implementation (dataset plugin or shared python script).
For `all`, passthrough arguments after `--` SHALL be forwarded only to dataset plugin steps (`download` and `convert`) to avoid ambiguity with shared preprocessing options.

Supported commands:
- `download`: dataset-specific internet download into `public_data/<dataset>/raw/`
- `convert`: dataset-specific parsing/conversion into the CoordExp JSONL contract
- `rescale`: shared smart-resize into a preset directory
- `coord`: shared coord-token conversion within the preset directory
- `validate`: validate both raw and preset artifacts for contract/template compliance
- `all`: run `download -> convert -> rescale -> coord -> validate` in order (default preset resolution required)
- `help`: print usage and exit 0

Supported runner flags:
- `--preset <name>`: preset directory name under `public_data/<dataset>/` (used by `rescale|coord|validate|all`)
- `--conda-env <name>`: override the default conda environment name (default: `ms`)
- `--skip-image-check`: skip image existence checks during validation (annotation-only flows)
- `--raw-only`: for `validate`, validate only raw artifacts
- `--preset-only`: for `validate`, validate only preset artifacts

Supported environment interfaces:
- `PUBLIC_DATA_MAX_OBJECTS=<N>` enables optional max-object filtering for the `coord` command only.
- The runner MUST fail fast when `PUBLIC_DATA_MAX_OBJECTS` is set for `rescale`, `validate`, or `all`, with guidance to run a two-step flow (`rescale/all` first, then `coord` with max-object filtering).
- The runner forwards `max_objects` into the pipeline planner only for `coord`; naming and effective preset resolution are owned by the planner.
- If `max_objects` is provided by multiple input surfaces and values disagree, execution MUST fail fast with an actionable error (no silent override).

Preset resolution rules:
- For commands that require a preset (`rescale`, `coord`) the user SHALL provide `--preset <name>`.
- For `all`, the runner SHALL:
  - use `--preset <name>` when provided, otherwise
  - use the dataset plugin default preset, otherwise
  - exit non-zero and print that a preset is required.
- For `validate`:
  - if `--raw-only` is set, the runner SHALL NOT require or resolve a preset.
  - otherwise (default behavior or `--preset-only`), the runner SHALL:
    - use `--preset <name>` when provided, otherwise
    - use the dataset plugin default preset, otherwise
    - exit non-zero and print that a preset is required.

Mutual exclusivity:
- For `validate`, `--raw-only` and `--preset-only` SHALL be mutually exclusive; setting both SHALL be an error.

#### Scenario: User runs an unknown command
- **GIVEN** the user runs `public_data/run.sh vg does-not-exist`
- **THEN** the runner exits non-zero and prints a usage message listing supported commands.

#### Scenario: User passes mutually-exclusive validate flags
- **GIVEN** the user runs `public_data/run.sh <dataset> validate --raw-only --preset-only`
- **THEN** the runner exits non-zero and prints that `--raw-only` and `--preset-only` cannot be used together.

#### Scenario: User runs a shared step without required inputs
- **GIVEN** `public_data/<dataset>/raw/train.jsonl` does not exist
- **WHEN** the user runs `public_data/run.sh <dataset> rescale --preset <preset>`
- **THEN** the runner exits non-zero and prints which required input file is missing.

#### Scenario: User runs coord-token conversion without preset inputs
- **GIVEN** `public_data/<dataset>/<preset>/train.jsonl` does not exist
- **WHEN** the user runs `public_data/run.sh <dataset> coord --preset <preset>`
- **THEN** the runner exits non-zero and prints which required preset file is missing.

#### Scenario: User runs `all` without a preset and the dataset has no default
- **GIVEN** the dataset plugin does not define a default preset
- **WHEN** the user runs `public_data/run.sh <dataset> all`
- **THEN** the runner exits non-zero and prints that `--preset <name>` is required.

#### Scenario: User assumes passthrough args affect shared steps inside `all`
- **GIVEN** the user wants to tune shared preprocessing options for `rescale` or `coord`
- **WHEN** the user runs `public_data/run.sh <dataset> all --preset <preset> -- --image-factor 64`
- **THEN** the runner prints a message explaining that `all` runs shared steps with runner defaults, and users should run `rescale`/`coord` as separate steps to tune their options.

#### Scenario: Conda environment is unavailable
- **GIVEN** `conda` or the requested conda environment is not available on the system
- **WHEN** the user runs a command that invokes python (e.g., `rescale`)
- **THEN** the runner exits non-zero and prints a message indicating the missing dependency (conda/env).

### Requirement: Dataset-Specific Download and Conversion
The system SHALL allow dataset-specific customization for internet downloading and raw-to-contract conversion, because datasets originate from different sources and use different metadata/fields.

#### Scenario: Different datasets implement different download logic
- **GIVEN** dataset A requires multiple mirrors and dataset B requires authentication or special URLs
- **WHEN** the user runs `public_data/run.sh <dataset> download`
- **THEN** the dataset-specific download logic is executed without requiring changes to shared preprocessing scripts.

### Requirement: Geometry Compatibility (bbox_2d Required, poly Optional)
For compatibility across diverse sources, the dataset conversion step SHALL always be able to emit bounding-box geometry in the contract form:
`bbox_2d: [x1, y1, x2, y2]`.

Polygon geometry (`poly`) support is OPTIONAL and MAY be implemented per dataset when the source provides polygon annotations.

Dataset conversion SHALL normalize source-specific geometry keys and formats into the CoordExp contract keys:
- Output geometry keys SHALL be only `bbox_2d` or `poly` (plus optional `poly_points`).
- Legacy source keys (e.g., `bbox`, `polygon`, `segmentation`, `coords`, etc.) SHALL NOT appear in the emitted contract JSONL.

#### Scenario: Source bbox key is not `bbox_2d`
- **GIVEN** a source dataset stores bbox geometry under a different field name (e.g., `bbox` or `coords`)
- **WHEN** the dataset plugin runs `convert`
- **THEN** the emitted contract JSONL uses `bbox_2d` with the correct `[x1,y1,x2,y2]` ordering.

#### Scenario: Source provides polygon only
- **GIVEN** a source dataset provides polygon geometry but not bounding boxes
- **WHEN** the dataset plugin runs `convert`
- **THEN** the plugin derives `bbox_2d` from the polygon (or emits `poly` when supported and enabled), and the record remains compliant with `docs/data/JSONL_CONTRACT.md`.

#### Scenario: Source provides both bbox and polygon
- **GIVEN** a source dataset provides both bbox and polygon geometry
- **WHEN** the dataset plugin runs `convert`
- **THEN** the plugin emits exactly one geometry per object.
- **AND** by default, the plugin emits `bbox_2d` (bbox-only is the default geometry mode).
- **AND** `poly` emission is opt-in and occurs only when explicitly enabled by the user via dataset-specific passthrough options.

#### Scenario: User explicitly enables poly emission for a dataset that supports polygons
- **GIVEN** a dataset plugin supports polygon emission
- **WHEN** the user runs `public_data/run.sh <dataset> convert -- <dataset-specific-flag-to-enable-poly>`
- **THEN** the emitted contract JSONL contains `poly` geometries where available (still exactly one geometry per object).

#### Scenario: Converter fails on missing geometry or invalid arity
- **GIVEN** a source object with empty `desc`, missing geometry, multiple geometry fields, or invalid geometry arity
- **WHEN** the dataset plugin runs `convert`
- **THEN** conversion fails fast with an actionable validation error
- **AND** no geometry-less/ambiguous object is emitted into the contract JSONL.


#### Scenario: Runtime builder fails on missing geometry or invalid arity
- **GIVEN** a record reaching runtime assistant-payload emission with an object that has empty `desc`, missing geometry, multiple geometry fields, or invalid geometry arity
- **WHEN** runtime payload construction executes
- **THEN** runtime validation fails fast with an actionable error
- **AND** no geometry-less/ambiguous object is serialized into assistant payload output.


### Requirement: Dataset Plugin Contract (Shell)
The unified runner SHALL load a dataset plugin implemented as an **executable shell script** at `public_data/datasets/<dataset>.sh`.

Each dataset plugin:
- SHALL be invoked by the runner (the runner SHALL NOT `source` it).
- SHALL implement the following subcommands:
  - `download`: download raw artifacts/images into the dataset raw directory.
  - `convert`: parse raw artifacts and write the CoordExp JSONL contract outputs.
  - `default-preset`: print the plugin default preset name to stdout and exit 0; exit non-zero if no default exists.
- SHALL accept runner-provided paths/settings as **explicit flags** (CLI contract; no env-var contract).
- MUST NOT require environment variables for runner-provided values (the runner does not export contract env vars).
- SHALL return a non-zero exit code on failure.

The runner SHALL pass required locations/settings as explicit flags to plugin subcommands:
- `--repo-root <abs>`
- `--dataset <id>`
- `--dataset-dir <abs>`
- `--raw-dir <abs>`
- `--raw-image-dir <abs>`
- `--raw-train-jsonl <abs>`
- `--raw-val-jsonl <abs>`
- `--conda-env <name>`

The runner SHALL forward passthrough arguments after `--` to plugin subcommands verbatim.

#### Scenario: Runner invokes dataset plugin successfully
- **GIVEN** `public_data/datasets/vg.sh` exists
- **WHEN** the user runs `public_data/run.sh vg convert`
- **THEN** the runner executes `public_data/datasets/vg.sh convert ...` with explicit flags and exits non-zero on plugin failure.

#### Scenario: Runner resolves default preset via plugin
- **GIVEN** `public_data/datasets/lvis.sh` defines `default-preset`
- **WHEN** the user runs `public_data/run.sh lvis all` without `--preset`
- **THEN** the runner calls `public_data/datasets/lvis.sh default-preset` and uses the returned preset name.

#### Scenario: Runner fails when plugin file is missing
- **GIVEN** `public_data/datasets/unknown.sh` does not exist
- **WHEN** the user runs `public_data/run.sh unknown download`
- **THEN** the runner exits non-zero and prints that the dataset is unknown and the plugin file is missing.

### Requirement: Standardized Contract Output Locations
Within the unified runner, the system SHALL standardize where converted JSONLs are written so downstream steps can be shared.

By default, dataset conversion output:
- SHALL write train JSONL to `public_data/<dataset>/raw/train.jsonl`.
- SHALL write val JSONL to `public_data/<dataset>/raw/val.jsonl` when a split exists.

#### Scenario: Conversion output is in standard locations
- **GIVEN** the user runs `public_data/run.sh vg convert`
- **THEN** `public_data/vg/raw/train.jsonl` exists on success, and `public_data/vg/raw/val.jsonl` exists when the dataset defines a val split.

#### Scenario: Underlying dataset converter defaults differ
- **GIVEN** an underlying dataset converter script has a different default output directory
- **WHEN** the user runs the unified runner's `convert` command
- **THEN** the runner/plugin writes to the runner-standard raw locations regardless of the underlying script defaults.

### Requirement: Shared Rescale Step
The system SHALL provide a shared `rescale` step that runs the existing resizing pipeline to produce a preset directory with relative image paths.

The rescale step:
- SHALL invoke `public_data/scripts/rescale_jsonl.py` via `conda run -n <conda-env> python ...` (default conda env: `ms`).
- SHALL write preset outputs under `public_data/<dataset>/<preset>/`.
- SHALL rescale `train.jsonl` and SHOULD also rescale `val.jsonl` when `public_data/<dataset>/raw/val.jsonl` exists.

#### Scenario: User rescale-preprocesses a dataset
- **GIVEN** `public_data/<dataset>/raw/train.jsonl` exists
- **WHEN** the user runs `public_data/run.sh <dataset> rescale --preset <preset-name> [options...]`
- **THEN** the runner writes `public_data/<dataset>/<preset-name>/train.jsonl` and `public_data/<dataset>/<preset-name>/images/`.

### Requirement: Shared Coord-Token Conversion Step
The system SHALL provide a shared `coord` step that converts preset JSONLs to coord-token JSONLs for training.

The coord step:
- SHALL invoke `public_data/scripts/convert_to_coord_tokens.py` via `conda run -n <conda-env> python ...` (default conda env: `ms`).
- SHALL produce `train.coord.jsonl` (and `val.coord.jsonl` when applicable) under the preset directory.

#### Scenario: User converts preset JSONL to coord tokens
- **GIVEN** `public_data/<dataset>/<preset>/train.jsonl` exists
- **WHEN** the user runs `public_data/run.sh <dataset> coord --preset <preset>`
- **THEN** `public_data/<dataset>/<preset>/train.coord.jsonl` exists on success.

### Requirement: Validation Covers Raw and Preset Outputs
The system SHALL provide a `validate` step to verify that prepared datasets meet baseline expectations for both:
- raw outputs (`public_data/<dataset>/raw/*.jsonl`)
- preset outputs (`public_data/<dataset>/<preset>/*.jsonl` and `*.coord.jsonl`)

Validation SHALL enforce the structural requirements in `docs/data/JSONL_CONTRACT.md` for both `bbox_2d` and `poly` geometries.

The validate step:
- SHALL validate that required keys exist (`images`, `objects`, `width`, `height`).
- SHALL enforce that `images[0]` is a relative path resolved against the JSONL directory (error on absolute paths). If `--skip-image-check` is set, file-existence checks are skipped but the relative-path requirement still applies.
- SHALL accept geometry coordinates as pixel-space numbers or coord tokens (`<|coord_k|>`) and SHALL report malformed/out-of-range coord tokens as validation errors (must not crash).
- SHALL reject legacy/unsupported geometry keys in emitted JSONLs (e.g., `bbox`, `polygon`, `line`, `line_points`).
- SHALL bound expensive image-open checks using deterministic first-N sampling for both raw and preset validation targets (default `N=64` per target file) instead of unbounded full-file opens.
- SHOULD run `scripts/tools/inspect_chat_template.py --index 0` on at least one coord-token JSONL sample to confirm prompt/template compatibility.
- SHALL validate both raw and preset artifacts by default, unless `--raw-only` or `--preset-only` is specified.
- SHALL validate `train.jsonl` and SHOULD also validate `val.jsonl` when present, for both raw and preset artifacts.

#### Scenario: User validates raw + preset outputs
- **GIVEN** the user has produced `public_data/<dataset>/raw/train.jsonl`
- **AND** the user has produced `public_data/<dataset>/<preset>/train.coord.jsonl`
- **WHEN** the user runs `public_data/run.sh <dataset> validate --preset <preset>`
- **THEN** the runner validates both raw and preset artifacts and exits non-zero if violations are detected.

#### Scenario: User runs validate without an available preset
- **GIVEN** the user runs `public_data/run.sh <dataset> validate`
- **AND** the user did not pass `--preset`
- **AND** the dataset plugin does not define a default preset
- **AND** `--raw-only` is NOT set
- **THEN** the runner exits non-zero and prints that `--preset <name>` is required to validate preset outputs.

#### Scenario: User validates without image files present (annotation-only)
- **GIVEN** the user has a JSONL but does not have the corresponding images on disk yet
- **WHEN** the user runs `public_data/run.sh <dataset> validate --raw-only --skip-image-check`
- **THEN** the runner validates JSON structure and geometry requirements without failing due to missing image files.

#### Scenario: Raw image-open validation is bounded by deterministic sampling
- **GIVEN** raw `train.jsonl` contains many records
- **WHEN** validation runs with image checks enabled
- **THEN** image-open checks use deterministic first-N sampling (default `N=64`) per target file rather than opening every image.

#### Scenario: Template sanity check is best-effort
- **GIVEN** the system cannot run `scripts/tools/inspect_chat_template.py` (no cached model or missing deps)
- **WHEN** the user runs `public_data/run.sh <dataset> validate --preset <preset>`
- **THEN** the runner warns and skips the template check while still enforcing the JSONL contract validation.

#### Scenario: User validates raw-only without a preset
- **GIVEN** the user has produced `public_data/<dataset>/raw/train.jsonl`
- **WHEN** the user runs `public_data/run.sh <dataset> validate --raw-only`
- **THEN** the runner validates only the raw artifacts and does not require a preset.

### Requirement: Environment and Reproducibility Defaults
The unified runner SHALL default to the project run environment conventions.

The runner:
- SHALL use `conda run -n <conda-env> python ...` by default for Python steps (default conda env: `ms`).
- SHALL allow selecting the conda environment via the explicit `--conda-env <name>` runner flag.
- SHALL emit human-readable stage logs so users can tell whether downloading/preprocessing is active.

#### Scenario: User sees stage logs during long downloads
- **GIVEN** a dataset download takes a long time (large image zips)
- **WHEN** the user runs `public_data/run.sh <dataset> download`
- **THEN** the runner prints stage banners and surfaces download progress from the dataset-specific logic.


### Requirement: Stable External Runner and Plugin Contract During Internal Refactor
The system MUST preserve the existing external interfaces of:
- `public_data/run.sh` command grammar and behavior contract,
- dataset shell plugin contract under `public_data/datasets/*.sh`,
while routing execution through the unified shared internal pipeline/factory implementation.

This preservation requirement applies throughout migration and after cutover.

#### Scenario: Existing runner invocation remains valid
- **WHEN** a user runs an existing command such as `./public_data/run.sh coco all --preset rescale_32_768_bbox`
- **THEN** command semantics remain valid without requiring new CLI flags
- **AND** execution is handled by unified internal pipeline stages under the hood.

#### Scenario: Existing dataset shell plugins remain valid integration points
- **WHEN** a dataset plugin implements the current shell contract in `public_data/datasets/<dataset>.sh`
- **THEN** the runner continues to invoke it successfully during and after migration
- **AND** plugin authors are not required to rewrite to a new external interface for this change.


### Requirement: Core Orchestration is Dataset-Agnostic
Shared pipeline orchestration MUST NOT hard-code dataset-specific processing logic in core execution flow.

Dataset-specific behavior MUST be encapsulated in adapter implementations resolved by the registry/factory.

Boundary definition:
- `public_data/run.sh` and dataset shell plugins are compatibility wrappers/integration surfaces.
- "Core orchestration" refers to the internal pipeline planner/stage executor implementation.

#### Scenario: Adding a new dataset avoids core orchestrator edits
- **WHEN** a new dataset is introduced through a new adapter and registry registration
- **THEN** shared orchestrator code path does not require dataset-conditional branches for conversion behavior.

#### Scenario: Dataset-specific fast paths are encapsulated
- **WHEN** a dataset needs a specialized optimization path
- **THEN** that behavior is implemented in adapter/plugin-owned integration surfaces
- **AND** core stage orchestration remains dataset-agnostic.


### Requirement: Optional Max-Object Filtering and Canonical Suffix Naming
Max-object filtering MUST be optional and disabled by default in unified pipeline execution.

When max-object filtering is enabled with value `N`, the effective preset/output naming MUST use canonical suffix `_max{N}` (for example `_max60`), so filtered artifacts are self-describing.

The planner MUST be the single source of truth for effective preset naming and output directory selection.
- `public_data/run.sh` MUST pass `--preset <base>` without pre-resolving an effective preset path.
- `max_objects` MUST be accepted only for `coord`; non-`coord` uses MUST fail fast with actionable two-step guidance.
- Legacy `_max_<N>` suffixes are invalid for this contract and MUST fail fast with an actionable rename hint.

If the effective preset/output name already contains the same canonical suffix, the system MUST NOT append it again.

#### Scenario: Default run has no max-object filter and no suffix
- **WHEN** pipeline execution runs without max-object filtering configured
- **THEN** no object-count filtering stage is applied
- **AND** output preset naming remains unchanged.

#### Scenario: Enabled max-object filter appends deterministic canonical suffix
- **WHEN** max-object filtering is enabled with `N=60`
- **THEN** effective output preset naming includes suffix `_max60`
- **AND** generated artifacts are written under the suffixed preset directory.

#### Scenario: Max-object filter outside coord fails fast
- **WHEN** max-object filtering is requested for `rescale`, `validate`, or `all`
- **THEN** execution fails fast with actionable guidance to run a two-step flow (`rescale/all` first, then `coord` with max filtering).

#### Scenario: Existing canonical suffix is not duplicated
- **WHEN** effective preset naming already ends with `_max60`
- **THEN** enabling max-object filtering with `N=60` does not append a second suffix token.

#### Scenario: Legacy underscore suffix is rejected
- **WHEN** preset naming includes legacy `_max_60`
- **THEN** planner resolution fails fast with an actionable migration hint to rename/rebuild as `_max60`.

### Requirement: Canonical Preset Artifact Layout
The unified pipeline MUST emit pixel-space preset outputs as canonical `<split>.jsonl` files.

Per split (`train` or `val`), artifact naming MUST be:
- `<split>.jsonl` for pixel-space records,
- `<split>.norm.jsonl` for norm1000 integer records,
- `<split>.coord.jsonl` for coord-token records.

The pipeline MUST NOT duplicate pixel-space outputs into additional alias files.

#### Scenario: Rescale writes canonical pixel-space artifact
- **WHEN** rescale/pipeline execution writes preset artifacts
- **THEN** the pixel-space artifact path is `<split>.jsonl`
- **AND** no `<split>.raw.jsonl` compatibility copy is emitted.

### Requirement: Derived Preset Image Reuse Uses Hardlinks
When max-object filtering creates a derived preset (`effective_preset != base_preset`), derived preset images MUST be materialized as hardlinks to the base preset resized images.

Hardlink materialization contract:
- Derived preset `images/` MUST be a real directory (never a symlink).
- For each referenced `images/...` path in derived preset JSONL, destination file MUST exist and be the same inode as the corresponding base preset image.
- In `coord` mode, hardlink materialization MUST execute after max-object filtering writes the derived `<split>.jsonl`, so only retained records drive hardlink creation.
- Materialization is append-only and idempotent:
  - create link when destination is missing,
  - no-op when destination already links to the same inode,
  - fail fast when destination exists but is a different inode.
- The planner MUST precheck same-filesystem compatibility and fail fast on hardlink errors.
- The planner MUST NOT fall back to byte-copy for derived presets.

#### Scenario: Derived preset links base images
- **WHEN** max-object filtering produces derived preset `..._max60`
- **THEN** derived `images/` is a real directory containing hardlinks to base preset images at matching relative paths.

#### Scenario: Dropped records do not materialize derived hardlinks
- **WHEN** max-object filtering drops a sample during derived `coord` processing
- **THEN** the dropped sample's `images/...` path is absent from the derived preset `images/` tree.

#### Scenario: Cross-device hardlink attempt fails fast
- **WHEN** base and derived preset roots are on different filesystems
- **THEN** hardlink materialization fails fast with actionable guidance to co-locate outputs on one filesystem.

### Requirement: Preset Image Immutability and Fresh Rescale Targets
Preset `images/` are immutable once written by the rescale stage.

Rescale/full execution MUST require a fresh preset target and MUST NOT overwrite existing preset artifacts in place.

Fail-fast gating for rescale target freshness:
- If target preset already contains any existing preset artifacts (`images/`, `<split>.jsonl`, `<split>.norm.jsonl`, `<split>.coord.jsonl`, `pipeline_manifest.json`, or filter-stats files), execution MUST fail fast.
- This includes symlink/file hazards at `images/`; the system MUST fail rather than attempting in-place repair/rematerialization.

`pipeline_manifest.json` still records rescale parameters (`max_pixels`, `min_pixels`, `image_factor`) for provenance and auditability, but strict freshness gating is authoritative for write safety.

Rebuild behavior:
- Rebuild is manual (fresh preset name or deliberate full deletion of existing preset directory).
- Rebuild MUST NOT be in-place overwrite of existing preset `images/`.

#### Scenario: Existing preset artifacts fail fast regardless of params
- **GIVEN** preset directory already contains previous artifacts
- **WHEN** user reruns rescale/full targeting that preset
- **THEN** execution fails fast and instructs the user to rebuild safely.
