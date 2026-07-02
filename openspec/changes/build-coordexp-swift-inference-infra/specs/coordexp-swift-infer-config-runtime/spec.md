## ADDED Requirements

### Requirement: Strict inference config
The system SHALL define a strict `InferConfig` for offline inference separate from `TrainConfig`.
The config MUST include `schema_version`, `run`, `model`, `data`, `template`,
`backend`, `generation`, `scoring`, `artifacts`, and `debug` sections. Adapter
and special-token embedding delta sections MAY be absent, but when present they
MUST be explicit.

#### Scenario: Valid production inference config
- **WHEN** a production inference config declares base model, dataset JSONL,
  template, HF backend, generation settings, artifact root, scoring enabled,
  and decode batch size greater than one
- **THEN** config validation succeeds and returns an `InferConfig`

#### Scenario: Training-only keys rejected
- **WHEN** an inference config includes training-only sections such as
  `optimizer`, `training`, or training `checkpoint`
- **THEN** config validation fails before model loading

#### Scenario: Unknown keys rejected
- **WHEN** an inference config includes an unknown top-level or nested key
- **THEN** config validation fails with a config-contract error that names the
  offending path

### Requirement: Inference config namespace
Inference configs SHALL live under `configs/coordexp_swift/infer/`.
Legacy `configs/infer/*` files MUST be treated as reference-only and MUST NOT
be loaded as canonical CoordExp-swift V1 inference configs unless a later
compatibility change explicitly promotes them.

#### Scenario: CoordExp-swift config path
- **WHEN** `python -m src.infer --config configs/coordexp_swift/infer/base.yaml`
  is invoked
- **THEN** the config loader treats the file as a CoordExp-swift inference
  config candidate

#### Scenario: Legacy config path
- **WHEN** `python -m src.infer --config configs/infer/pipeline.yaml` is invoked
- **THEN** validation fails with a message that legacy inference configs are
  reference-only for this worktree

### Requirement: Shallow production leaves
The system SHALL support shallow inference config inheritance for shared model, runtime, and template defaults.
Production leaf configs MUST explicitly declare dataset, adapter/checkpoint
selection when used, generation, scoring, and artifact-root fields.

#### Scenario: Explicit production leaf
- **WHEN** a leaf config inherits shared model/template defaults and explicitly
  declares data, generation, scoring, and artifact fields
- **THEN** resolved config writing succeeds

#### Scenario: Hidden inherited dataset
- **WHEN** a production leaf omits `data` because a parent file provided it
- **THEN** validation fails unless the config is marked as debug or smoke

### Requirement: Resolved config artifacts
Every inference run SHALL write self-contained resolved config artifacts under the run directory.
The canonical artifacts are `configs/resolved.json` and
`configs/resolved.yaml`. Inference SHALL NOT introduce a top-level
`resolved_config.json` artifact as the canonical V1 config evidence.

#### Scenario: Resolved config written
- **WHEN** inference initializes a run directory
- **THEN** `configs/resolved.json` and `configs/resolved.yaml` exist before
  model generation starts

#### Scenario: Manifest links config artifacts
- **WHEN** `run_manifest.json` is written
- **THEN** it records paths and fingerprints for both resolved config artifacts

### Requirement: Model adapter delta identity
Inference SHALL load a base model from a configured `model_cache/<model-id>` path and MAY apply explicit adapter or embedding-delta payloads.
When adapter or delta payloads are used, the runtime MUST record resolved
payload identity and MUST validate base and tokenizer identity before
generation.

#### Scenario: Base-only inference
- **WHEN** a config declares only a base model and no adapter or embedding delta
- **THEN** runtime setup succeeds and records base-only model identity

#### Scenario: Final checkpoint alias
- **WHEN** a config points to `checkpoint-final` metadata
- **THEN** runtime resolves concrete adapter and embedding-delta payload paths
  and records the resolved identities

#### Scenario: Wrong adapter base
- **WHEN** an adapter payload declares an incompatible base model identity
- **THEN** runtime setup fails before generation

#### Scenario: PEFT irregular load result
- **WHEN** adapter loading reports missing adapter keys, unexpected keys,
  disabled adapter status, an unexpected active adapter list, irregular status
  fields, or an unexpected merged state
- **THEN** runtime setup fails before generation and records the failed identity
  check diagnostically

#### Scenario: Warning-only PEFT load path
- **WHEN** an adapter loader relies only on warning output instead of capturing
  the PEFT `load_result` or equivalent missing/unexpected-key evidence
- **THEN** the implementation is noncompliant and adapter-enabled runtime setup
  cannot be accepted

#### Scenario: Missing delta identity
- **WHEN** a special-token embedding delta lacks required base/tokenizer
  metadata
- **THEN** runtime setup fails before generation

#### Scenario: Delta token mismatch
- **WHEN** an embedding delta records token strings or token ids that disagree
  with the runtime tokenizer identity
- **THEN** runtime setup fails before generation

### Requirement: Owner-neutral shared runtime APIs
Inference-facing config, Qwen, and artifact helpers SHALL be callable without training-owned configuration classes.
The implementation MUST provide or deepen owner-neutral APIs for Qwen loading,
resolved config writing, manifest primitives, adapter identity, and
special-token embedding delta identity. Inference modules MUST NOT depend on
`TrainConfig`, `ResolvedTrainConfig`, `ResolvedStepSchedule`, or
`src.training.*` except for explicitly approved type-free utilities. The V1
allowlist for `src.training.*` imports is empty unless a source-study note and
OpenSpec patch name a concrete exception.

#### Scenario: Inference import residue check
- **WHEN** inference-facing modules are inspected by tests or residue checks
- **THEN** they do not import `TrainConfig`, `ResolvedTrainConfig`,
  `ResolvedStepSchedule`, `load_train_config`, or any unallowlisted
  `src.training.*` module

#### Scenario: Config-neutral Qwen load
- **WHEN** inference runtime loads a base model
- **THEN** the Qwen owner receives config-neutral options such as model path,
  dtype, attention implementation, processor policy, patch policy, and
  load-model flag rather than a `TrainConfig`

#### Scenario: Type-neutral artifact writing
- **WHEN** inference writes resolved configs, manifests, or provenance
- **THEN** the write path does not require training step schedules, training
  checkpoint aliases, or `ResolvedTrainConfig`

#### Scenario: Training loader not reused
- **WHEN** inference loads or resolves an inference config
- **THEN** the call path does not call `load_train_config()` and does not return
  `ResolvedTrainConfig`

### Requirement: Thin public entrypoint
The public inference entry SHALL be `src/infer.py`, invoked as `python -m src.infer --config CONFIG.yaml`.
Implementation modules SHALL live under `src/inference/`. The repository MUST
NOT create both `src/infer.py` and a sibling `src/infer/` package.

#### Scenario: Help command resolves entry
- **WHEN** `python -m src.infer --help` is invoked
- **THEN** the command resolves the thin inference entrypoint without path hacks

#### Scenario: Package collision avoided
- **WHEN** the source tree is inspected
- **THEN** `src/inference/` exists for implementation modules and `src/infer/`
  does not exist as a sibling package
