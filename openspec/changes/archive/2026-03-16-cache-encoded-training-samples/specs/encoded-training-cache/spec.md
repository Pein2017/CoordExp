# encoded-training-cache Specification (delta: deterministic reusable encoded samples for training)

## Purpose
Define an optional deterministic cache for pre-encoded CoordExp training samples
so supported runs can reuse rendered/encoded payloads instead of rebuilding them
inside dataloader workers on every epoch.

## Requirements

## ADDED Requirements

### Requirement: Encoded training cache uses a canonical YAML namespace
The system SHALL expose encoded-sample caching only through the canonical typed
namespace `training.encoded_sample_cache`.

Normative behavior:
- The cache MUST be opt-in through YAML configuration.
- The cache MUST NOT require new CLI flags.
- Unknown keys under `training.encoded_sample_cache` MUST fail fast.
- The canonical v1 fields are:
  - `training.encoded_sample_cache.enabled: bool`
  - `training.encoded_sample_cache.root_dir: str | null`
  - `training.encoded_sample_cache.ineligible_policy: \"error\" | \"bypass\"`
  - `training.encoded_sample_cache.wait_timeout_s: int`
- When `training.encoded_sample_cache.root_dir` is omitted, the cache root MUST
  default to `<training.output_dir>/cache/encoded_samples`.
- `training.encoded_sample_cache.ineligible_policy` MUST default to `error`.
- `training.encoded_sample_cache.wait_timeout_s` MUST default to `7200`, and
  `0` MUST mean wait indefinitely.

#### Scenario: Omitted root_dir uses the run-scoped default cache root
- **GIVEN** encoded-sample caching is enabled
- **AND** `training.encoded_sample_cache.root_dir` is omitted
- **WHEN** training resolves the cache location
- **THEN** the cache root is `<training.output_dir>/cache/encoded_samples`
- **AND** no extra CLI flag is required.

### Requirement: Encoded cache is limited to explicitly eligible deterministic runs
The system SHALL allow encoded-sample caching only when the encoded payload is a
pure function of stable base-record identity plus fixed configuration.

Normative behavior:
- V1 cache-eligible runs MUST be plain JSONL-backed `BaseCaptionDataset` paths
  with one stable base record per `base_idx`.
- The cache MUST NOT be treated as eligible when encoded output depends on
  runtime RNG, requested dataloader index, or epoch.
- The following paths MUST be treated as ineligible in v1:
  - augmentation,
  - preprocessors that mutate encoded content at fetch time,
  - non-empty `curriculum_state`,
  - `custom.object_ordering: random`,
  - fusion/mixing schedules,
  - hard-sample or epoch-varying sample plans that change dataset identity at
    fetch time.
- When `training.encoded_sample_cache.ineligible_policy=error`, startup MUST
  fail fast with actionable guidance.
- When `training.encoded_sample_cache.ineligible_policy=bypass`, training MUST
  continue without cache reuse and MUST emit an explicit bypass reason.

#### Scenario: Ineligible non-deterministic dataset path is rejected
- **GIVEN** encoded-sample caching is enabled
- **AND** the active dataset path uses non-deterministic sample mutation or an
  epoch-varying schedule
- **WHEN** training initializes dataset caching
- **AND** `training.encoded_sample_cache.ineligible_policy=error`
- **THEN** the system does not silently reuse encoded samples
- **AND** it fails fast with actionable guidance.

#### Scenario: Ineligible request is bypassed only when authored explicitly
- **GIVEN** encoded-sample caching is enabled
- **AND** the active dataset path is ineligible for deterministic cache reuse
- **AND** `training.encoded_sample_cache.ineligible_policy=bypass`
- **WHEN** training initializes dataset caching
- **THEN** the system continues uncached
- **AND** it records an explicit cache-bypass reason.

### Requirement: Encoded cache reuse is guarded by a stable fingerprint and artifact manifest
The system SHALL guard cache reuse with a stable fingerprint and a manifest-backed
artifact layout.

Normative behavior:
- The cache directory MUST resolve to `<cache_root>/<fingerprint>/`.
- The fingerprint MUST include all resolved surfaces that can change encoded
  payloads or their deterministic planning lengths, including:
  - dataset source identity,
  - dataset seed,
  - prompts/system prompts,
  - template length surfaces,
  - object ordering and object field ordering,
  - coord-token behavior,
  - dataset mode (`dense` vs `summary`),
  - offline image-budget invariants,
  - and artifact format/schema version.
- Cache lookup identity MUST be `base_idx`, not requested dataloader index.
- The cache directory MUST contain a `manifest.json` that records fingerprint,
  artifact format version, sample count, preserved payload keys, shard
  inventory, and build provenance.
- Cross-run reuse MAY occur only when an explicit or default root contains a
  manifest whose fingerprint matches exactly.

#### Scenario: Shared root reuses an identical fingerprint across runs
- **GIVEN** two training runs point to the same explicit
  `training.encoded_sample_cache.root_dir`
- **AND** the resolved fingerprint is identical between the two runs
- **WHEN** the second run initializes dataset caching
- **THEN** it reuses the existing `<root>/<fingerprint>/manifest.json`
- **AND** it does not rebuild encoded payloads for that fingerprint.

### Requirement: Encoded cache build is single-writer and fetch bypasses hot-path encoding
The system SHALL build encoded caches safely under distributed launch and SHALL
reuse them in dataset fetch without re-running hot-path encoding.

Normative behavior:
- Cache build for a missing fingerprint MUST use a single writer.
- Non-writer ranks/processes MUST wait for a complete manifest up to
  `training.encoded_sample_cache.wait_timeout_s`.
- Partial or interrupted builds MUST NOT be treated as valid cache hits.
- The writer MUST publish the manifest only after all cache shard files are
  complete.
- On cache hit, dataset fetch MUST load the stored encoded payload by `base_idx`
  instead of rebuilding the conversation payload and re-running
  `template.encode(...)` for that fetch.
- Cached payloads MUST preserve the downstream sample contract required by
  Stage-1 packing and Stage-2 training, including sample provenance and
  teacher-forcing side information.

#### Scenario: Cache hit bypasses repeated sample encoding
- **GIVEN** a deterministic training config with encoded-sample caching enabled
- **AND** a matching cache artifact already exists for the resolved dataset and
  encoding fingerprint
- **WHEN** the dataset fetches a sample during training
- **THEN** it loads the encoded sample payload from the cache by `base_idx`
- **AND** it does not re-run the full render-plus-`template.encode(...)` path
  for that sample fetch.

#### Scenario: Non-writer rank waits for a complete manifest
- **GIVEN** encoded-sample caching is enabled under distributed launch
- **AND** one rank/process is building a missing fingerprint
- **WHEN** another rank/process initializes dataset caching
- **THEN** it waits for the manifest to reach a complete state
- **AND** it does not read partial cache artifacts as a valid hit.

### Requirement: Encoded cache provenance is visible in startup logs and run artifacts
The system SHALL expose enough provenance for operators to verify whether the
encoded-sample cache was built, reused, or bypassed.

Normative behavior:
- Startup logs SHALL record whether the cache was `built`, `reused`, or
  `bypassed`.
- Startup logs SHALL record the active fingerprint, resolved cache root, and
  resolved cache directory.
- When bypass occurs, startup logs SHALL record the bypass reason.
- Run artifacts under `training.output_dir` SHALL record the same cache
  provenance so operators can recover it after startup.
- Cache provenance MUST NOT require new canonical per-step metric keys and MUST
  NOT change model or rollout semantics.

#### Scenario: Cache usage is visible in run outputs
- **GIVEN** encoded-sample caching is enabled for a supported deterministic run
- **WHEN** training starts
- **THEN** the run outputs include cache provenance information
- **AND** operators can determine whether the cache was built, reused, or
  bypassed from startup logs and run artifacts.
