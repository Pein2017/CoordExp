## ADDED Requirements

### Requirement: Offline bbox-format derivation consumes canonical preset split datasets
The system SHALL derive non-canonical model-facing bbox-format datasets only
from validated canonical preset split datasets in `public_data/` whose
`bbox_2d` geometry remains canonical `xyxy`.

Normative behavior:
- the source artifact for offline bbox-format derivation MUST be canonical
  preset split JSONL such as `train.jsonl` / `val.jsonl` produced by the shared
  public-data pipeline,
- the source records MUST remain canonical `xyxy` at the record boundary,
- the source records MUST contain `bbox_2d` geometry only for this first
  implementation surface,
- the derivation stage MUST fail fast if any source record contains `poly` or
  if geometry types are mixed across the source artifact,
- the derivation stage MUST NOT accept an already-derived non-canonical branch
  as input for another bbox-format conversion,
- the derivation stage MUST fail fast if the source artifact lacks the metadata
  required to prove canonical provenance.

#### Scenario: Canonical preset dataset is accepted for cxcy-logw-logh derivation
- **WHEN** the user runs offline bbox-format derivation on a canonical
  `train.jsonl`
- **THEN** the system emits a derived `cxcy_logw_logh` branch
- **AND** the source `bbox_2d` values are interpreted as canonical `xyxy`.

#### Scenario: Non-canonical source is rejected
- **WHEN** the user attempts to derive a bbox-format branch from an artifact that
  already declares a non-canonical prepared bbox format
- **THEN** the derivation step fails fast
- **AND** it reports that offline bbox-format derivation accepts canonical
  sources only.

### Requirement: Derived bbox-format branches are physically separate from canonical presets
The system SHALL store non-canonical prepared bbox-format datasets in a
dedicated branch root under the canonical preset rather than overwriting the
canonical preset artifacts.

Normative behavior:
- the canonical preset root under `public_data/<dataset>/<preset>/` MUST remain
  unchanged,
- the derived branch root MUST be
  `public_data/<dataset>/<preset>_<format>/`,
- each derived branch MUST emit:
  - `train.jsonl` containing offline-prepared norm1000 integer bbox tuples in
    the derived chart,
  - `train.coord.jsonl` containing the tokenized version of the same prepared
    tuples,
  - `val.jsonl` containing offline-prepared norm1000 integer bbox tuples in the derived
    chart when a canonical val source exists,
  - `val.coord.jsonl` containing the tokenized version of the same prepared
    tuples when a canonical val source exists,
  - `manifest.json`,
- derived branch generation MUST NOT overwrite canonical
  `train.jsonl|train.norm.jsonl|train.coord.jsonl` artifacts in the preset root.

#### Scenario: Derived branch writes into a separate root
- **WHEN** offline derivation is run for `cxcy_logw_logh`
- **THEN** the outputs are written under
  `public_data/<dataset>/<preset>_cxcy_logw_logh/`
- **AND** the canonical preset files remain in place.

#### Scenario: Missing val split preserves split semantics
- **WHEN** the canonical preset contains only `train.jsonl`
- **THEN** the derived branch emits only `train.jsonl` and
  `train.coord.jsonl`
- **AND** it does not fabricate val artifacts.

### Requirement: Derived branches are self-describing and auditable
Each derived bbox-format branch SHALL carry explicit provenance at both the
branch and record level.

Normative behavior:
- `manifest.json` in the branch root MUST record:
  - canonical source artifact path(s),
  - source bbox format,
  - derived bbox format,
  - derived bbox slot order,
  - coord-token/norm contract version,
  - numeric split contract stating that `<split>.jsonl` uses norm1000 integer
    slots on the same lattice as `<split>.coord.jsonl`,
  - bbox-format conversion version,
  - generator command or resolved configuration,
- each emitted record MUST include metadata that identifies:
  - `prepared_bbox_format`,
  - `prepared_bbox_slot_order`,
  - `prepared_bbox_source_format`,
  - `prepared_bbox_conversion_version`,
  - canonical source lineage sufficient for downstream validation,
- downstream consumers MUST be able to validate branch provenance without
  relying exclusively on directory naming.

#### Scenario: Derived record remains self-describing after file relocation
- **WHEN** a derived `train.coord.jsonl` is copied outside its original branch
  directory
- **THEN** downstream validation can still recover the prepared bbox-format
  identity from record metadata
- **AND** it does not depend only on the original path.

#### Scenario: Manifest records source lineage
- **WHEN** a derived branch is generated
- **THEN** its `manifest.json` records the canonical source artifact and emitted
  bbox-format metadata
- **AND** operators can audit how the branch was produced.

### Requirement: Derived bbox-format generation is deterministic and versioned
Offline bbox-format derivation SHALL be deterministic for a fixed source
artifact set and resolved derivation configuration.

Normative behavior:
- identical canonical inputs and identical derivation settings MUST produce
  byte-identical derived outputs,
- for `cxcy_logw_logh`, the derived chart MUST serialize bbox slots in the
  order `[cx, cy, logw, logh]`,
- any intentional change to the conversion semantics MUST bump the
  bbox-format conversion version recorded in manifests and record metadata,
- downstream compatibility checks MUST treat a conversion-version mismatch as a
  provenance mismatch.

#### Scenario: Repeated derivation is byte-identical
- **WHEN** the same canonical source artifact is derived twice with the same
  bbox format and settings
- **THEN** the emitted branch artifacts are byte-identical.

#### Scenario: Conversion-version mismatch invalidates compatibility
- **WHEN** a consumer requests a derived branch whose recorded conversion version
  differs from the supported version
- **THEN** the consumer fails fast
- **AND** it reports the provenance/version mismatch explicitly.
