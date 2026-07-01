## ADDED Requirements

### Requirement: Dataset Adapter Registry and Factory Instantiation
The system MUST provide a registry-based adapter factory that resolves a dataset id to a concrete adapter implementation without editing shared pipeline orchestration code.

Adapter registration MUST support at least `lvis`, `coco`, and `vg` at launch.

#### Scenario: Known dataset id resolves to adapter
- **WHEN** pipeline execution is requested for dataset id `coco`
- **THEN** the factory returns the registered COCO adapter
- **AND** shared pipeline orchestration proceeds without dataset-specific branching in core stage runner logic.

#### Scenario: Unknown dataset id fails fast
- **WHEN** pipeline execution is requested for an unregistered dataset id
- **THEN** the factory fails fast with an actionable error listing available dataset ids.

### Requirement: Adapter Contract Owns Source-Specific Ingestion
Each dataset adapter MUST implement a source-ingestion contract that includes:
- raw data acquisition hooks,
- annotation parsing,
- source-specific normalization into canonical intermediate records for shared transforms.

The adapter contract MUST NOT own shared post-ingestion transforms (rescale/filter/coord conversion/output formatting).

#### Scenario: Adapter produces canonical intermediate records
- **WHEN** an adapter parses raw annotations for its dataset
- **THEN** it emits canonical intermediate records consumable by shared transform stages
- **AND** downstream stages are dataset-agnostic.

#### Scenario: Shared transforms are not duplicated in adapters
- **WHEN** a new adapter is implemented
- **THEN** common post-ingestion steps (rescale, optional max-object filtering, coord-token conversion, formatting) are invoked through shared stages rather than re-implemented in adapter code.

### Requirement: Canonical Intermediate Record Contract is Explicit
Canonical intermediate records at the adapter->pipeline boundary MUST follow a defined schema:
- required keys: `images`, `objects`, `width`, `height`,
- object-level invariant: exactly one geometry key per object (`bbox_2d` or `poly`) plus non-empty `desc`,
- coordinate space at this boundary is pixel-space,
- optional metadata is allowed.

Canonical intermediate records MUST be structurally compatible with `docs/data/JSONL_CONTRACT.md`, except that in-memory `images` paths may be absolute before writer relativization.

#### Scenario: Intermediate record violates geometry invariant
- **WHEN** an adapter emits an object containing both `bbox_2d` and `poly`
- **THEN** pipeline validation fails fast before downstream shared transforms continue.

#### Scenario: Intermediate record uses absolute image path in-memory
- **WHEN** adapter emits canonical intermediate record with absolute `images[0]`
- **THEN** writer stage converts persisted output paths to be relative to output JSONL directory.

### Requirement: Always-On Structural Checks Are Separate from Optional Full Validation
The unified pipeline MUST run minimal structural checks unconditionally before expensive/output-level validation:
- required top-level keys (`images`, `objects`, `width`, `height`),
- exactly one geometry key per object (`bbox_2d` or `poly`),
- non-empty `desc`,
- geometry type/arity sanity (list type and expected coordinate arity).

Full contract validation (including checks such as image existence and broader validator policies) MAY run through an optional validation stage.

#### Scenario: Always-on structural check fails before optional validation stage
- **WHEN** a record violates one-geometry-per-object invariant
- **THEN** pipeline fails fast even if optional validation stage is disabled.

#### Scenario: Optional validation stage disabled with valid structural checks
- **WHEN** optional validation stage is disabled and records pass always-on structural checks
- **THEN** pipeline may proceed without running full validator stage logic.

### Requirement: Shared Stage Pipeline is Composable and Deterministic
The unified internal pipeline MUST execute dataset processing through explicit ordered stages:
- ingestion output normalization,
- smart resize under configured pixel budget,
- optional object-count filtering,
- coordinate normalization to norm1000 integers,
- coord-token expansion,
- standardized JSONL formatting/writing,
- validation hook stage.

Stage execution order MUST be deterministic for a given configuration.

#### Scenario: Same config yields same stage order
- **WHEN** the same dataset and pipeline configuration are run twice
- **THEN** stage order and output artifact set are identical.

#### Scenario: Stage composition differs only by config
- **WHEN** `max_objects` is unset
- **THEN** the max-object filtering stage is skipped
- **AND** all other configured stages run in the same deterministic order.

#### Scenario: Max-object semantics are drop-not-truncate
- **WHEN** max-object filtering is enabled with threshold `N`
- **THEN** records with `len(objects) > N` are dropped entirely
- **AND** records are not truncated to first `N` objects.

#### Scenario: Max-object stage placement is stable
- **WHEN** max-object filtering is enabled
- **THEN** filtering runs after resize/path normalization and before coordinate normalization/token expansion.

### Requirement: Deterministic Geometry Canonicalization and Ordering is Preserved
The unified pipeline MUST preserve deterministic geometry behavior currently used by shared helpers:
- polygon canonicalization semantics from `public_data/converters/sorting.py::canonicalize_poly`,
- object ordering semantics from `public_data/converters/sorting.py::sort_objects_tlbr`.

#### Scenario: Canonical polygon ordering matches shared helper semantics
- **WHEN** polygon objects are processed by unified pipeline stages
- **THEN** vertex ordering and start-point canonicalization follow the same semantics as `canonicalize_poly`.

#### Scenario: Object order remains top-to-bottom then left-to-right
- **WHEN** objects are emitted in final artifacts
- **THEN** ordering follows shared `sort_objects_tlbr` semantics deterministically.

### Requirement: Unified Writer Preserves Contract and Relative Paths
The output formatter/writer used by the unified pipeline MUST preserve `docs/data/JSONL_CONTRACT.md` invariants and MUST produce image paths relative to the output JSONL directory.

It MUST support generation of:
- pixel-space train/val raw outputs,
- norm1000 numeric train/val outputs,
- coordinate-token expanded train/val JSONL outputs as first-class artifacts.

Per split (`train` or `val`), artifact naming MUST be:
- `<split>.raw.jsonl` for pixel-space records,
- `<split>.norm.jsonl` for norm1000 integer records,
- `<split>.coord.jsonl` for coord-token records.

#### Scenario: Writer emits contract-compliant artifact
- **WHEN** a transformed record is written by unified writer
- **THEN** output JSONL remains contract-compliant for geometry keys and top-level required fields.

#### Scenario: Writer emits coord-token expanded output
- **WHEN** coord-token stage is enabled in the pipeline plan
- **THEN** the pipeline emits `train.coord.jsonl` and `val.coord.jsonl` (when val exists) as first-class outputs.

#### Scenario: Writer emits explicit coordinate-space variants
- **WHEN** a split is processed through full pipeline stages
- **THEN** persisted outputs clearly separate pixel-space, norm1000 numeric, and coord-token variants using the required filenames.

### Requirement: Legacy Runner Mapping Is Explicit During Migration
When unified internals are invoked through legacy runner surfaces, artifact mapping MUST be explicit to avoid ambiguity:
- legacy rescale view `<split>.jsonl` (pixel-space) maps to canonical `<split>.raw.jsonl`,
- normalized numeric artifact remains explicit as `<split>.norm.jsonl`,
- coord-token artifact remains `<split>.coord.jsonl`.

#### Scenario: Legacy runner compatibility mapping for rescale output
- **WHEN** users interact through legacy `run.sh` rescale-compatible flows
- **THEN** pixel-space compatibility output corresponds to canonical `<split>.raw.jsonl`
- **AND** parity gates compare legacy pixel outputs against canonical raw outputs.

### Requirement: Validation Hook Stage is a First-Class Pipeline Stage
If validation is enabled in pipeline plan configuration, validation MUST run as an explicit stage in the stage planner rather than only as an out-of-band manual check.

#### Scenario: Validation enabled in stage plan
- **WHEN** pipeline plan includes validation stage
- **THEN** validator executes as a stage in the same deterministic stage sequence
- **AND** validation failure causes pipeline failure for that run.
