## ADDED Requirements

### Requirement: Encoded cache fingerprints include prepared bbox-branch provenance
Encoded training cache identity SHALL include the prepared bbox-format branch
provenance for any dataset that uses offline-derived non-canonical bbox-format
artifacts.

Normative behavior:
- the fingerprint MUST include:
  - prepared bbox format,
  - prepared bbox slot order,
  - canonical source lineage,
  - bbox-format conversion version,
  - branch root or equivalent manifest identity,
- cache manifests MUST record the same prepared-branch provenance,
- cache reuse across datasets or runs that differ in prepared bbox-format
  provenance MUST be treated as invalid.

#### Scenario: Cache reuse is blocked across different prepared branches
- **GIVEN** two runs share the same cache root
- **AND** one uses canonical `xyxy` data while the other uses an offline
  `cxcy_logw_logh` branch
- **WHEN** cache lookup resolves the fingerprint
- **THEN** the fingerprints differ
- **AND** cache reuse does not cross the branch boundary.

#### Scenario: Cache reuse is blocked between cxcy_logw_logh and cxcywh
- **GIVEN** two runs share the same cache root
- **AND** one uses an offline `cxcy_logw_logh` branch while the other uses an
  offline `cxcywh` branch
- **WHEN** cache lookup resolves the fingerprint
- **THEN** the fingerprints differ
- **AND** cache reuse does not cross the branch boundary.

#### Scenario: Cache manifest records prepared branch identity
- **WHEN** encoded-sample caching is enabled for a derived bbox-format branch
- **THEN** the cache manifest records the prepared bbox-format provenance
- **AND** operators can audit which branch populated the cache.

### Requirement: Ambiguous bbox-format provenance is cache-ineligible
Datasets with ambiguous or missing prepared bbox-format provenance SHALL be
treated as cache-ineligible for non-canonical bbox-format training.

Normative behavior:
- if a non-canonical bbox-format run cannot prove the dataset branch identity
  from manifest or record metadata, cache initialization MUST:
  - fail fast when `ineligible_policy=error`, or
  - bypass cache reuse when `ineligible_policy=bypass`,
- the cache runtime MUST NOT silently assume canonical `xyxy` provenance for a
  non-canonical bbox-format experiment,
- partial legacy caches built under the removed online-conversion path MUST NOT
  be treated as valid hits for the offline-prepared branch workflow.

#### Scenario: Missing provenance prevents cache reuse
- **GIVEN** encoded-sample caching is enabled for a non-canonical bbox-format run
- **AND** the dataset artifacts do not expose the required prepared-branch
  provenance
- **WHEN** cache initialization runs
- **THEN** the cache is treated as ineligible
- **AND** the configured ineligible policy is applied explicitly.

#### Scenario: Legacy online-conversion cache is not reused
- **GIVEN** a cache directory was created before offline-prepared bbox-format
  branch provenance became part of cache identity
- **WHEN** a non-canonical bbox-format run initializes cache lookup
- **THEN** the legacy cache is not reused
- **AND** the system treats it as provenance-incompatible.
