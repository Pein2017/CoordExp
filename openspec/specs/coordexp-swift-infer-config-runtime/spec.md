# coordexp-swift-infer-config-runtime Specification

## Purpose
Canonical CoordExp-Swift inference configuration contract for strict YAML
loading, inheritance, runtime projection, and provenance on repository `main`.
The contract is owned by the rebuilt `src/inference/` runtime and is separate
from the archived MS-Swift/mainline inference configuration surfaces.
## Requirements
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

Inference SHALL load a base model from a configured
`model_cache/<model-id>` path and MAY apply explicit `adapter.path` and
`embedding_delta.path` payloads. It MUST NOT require `checkpoint-final`,
`checkpoint.json`, or `checkpoint_handoff.json` metadata to resolve those
paths. Runtime MUST record the identities it actually loads before generation.

For a standard PEFT/DoRA adapter, runtime MUST validate the configured base
identifier and model compatibility, `peft_type: LORA`, `use_dora: true`, target
modules, tensor shapes, nonempty LoRA A/B and DoRA magnitude-vector state, and
the PEFT load result/status. It MUST NOT claim immutable base-config or
tokenizer-content validation absent from standard adapter metadata. When a
selected-token embedding delta is configured, runtime MUST additionally
validate that payload's recorded base-config hash, tokenizer hash, token
strings/ids, tensor key, shape, source tensor dtype, and tied-weight semantics.
The declared source dtype MUST match the actual payload tensor. Runtime MAY
convert that validated tensor into the installed delta-parameter dtype, but it
MUST record both source and runtime dtypes when they differ.

#### Scenario: Base-only inference

- **WHEN** a config declares only a base model and no adapter or embedding
  delta
- **THEN** runtime setup MUST succeed and record base-only model identity.

#### Scenario: Explicit adapter and delta paths

- **WHEN** a config declares `adapter.path` and optional
  `embedding_delta.path`
- **THEN** runtime MUST load those concrete payloads directly
- **AND** MUST record their actual loader identities without resolving
  checkpoint-final or handoff metadata.

#### Scenario: Final checkpoint alias

- **WHEN** a new canonical inference config points to `checkpoint-final`
  metadata instead of explicit adapter and optional delta paths
- **THEN** config or runtime validation MUST fail before generation.

#### Scenario: Validated delta dtype conversion

- **WHEN** an embedding-delta payload tensor matches its declared source dtype
  and every other payload identity, but the installed runtime delta parameter
  uses a different supported dtype
- **THEN** runtime MAY convert the validated tensor into the installed dtype
- **AND** MUST record both the source and runtime tensor dtypes
- **BUT** a mismatch between the payload tensor and its declared source dtype
  MUST fail before conversion.

#### Scenario: Wrong adapter base

- **WHEN** an adapter payload declares an incompatible base identifier or model
  contract
- **THEN** runtime setup MUST fail before generation.

#### Scenario: PEFT irregular load result

- **WHEN** adapter loading reports missing adapter keys, unexpected keys,
  disabled adapter status, an unexpected active adapter list, irregular status
  fields, or an unexpected merged state
- **THEN** runtime setup MUST fail before generation and record the failed
  identity check diagnostically.

#### Scenario: Warning-only PEFT load path

- **WHEN** an adapter loader relies only on warning output instead of capturing
  the PEFT `load_result` or equivalent missing/unexpected-key evidence
- **THEN** the implementation MUST be rejected for adapter-enabled runtime
  setup.

#### Scenario: Adapter base contents change at the same path

- **WHEN** adapter-only inference uses a base path whose contents changed while
  its standard adapter identifier/model/shape contract still matches
- **THEN** runtime MUST apply the declared standard PEFT checks
- **AND** MUST NOT claim immutable base/tokenizer hash validation.

#### Scenario: Missing delta identity

- **WHEN** a selected-token embedding delta lacks required base/tokenizer
  metadata
- **THEN** runtime setup MUST fail before generation.

#### Scenario: Delta token mismatch

- **WHEN** an embedding delta records token strings or token ids that disagree
  with the runtime tokenizer identity
- **THEN** runtime setup MUST fail before generation.

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

### Requirement: CUDA-required non-dry inference runtime
CoordExp-swift inference SHALL require CUDA for every non-dry execution path.
The runtime MUST fail before Qwen model loading if no CUDA device is visible.
This applies to direct single-rank execution and controller/worker execution.
This feature MUST NOT silently fall back to CPU execution.

#### Scenario: CUDA unavailable
- **WHEN** a non-dry inference run starts with no visible CUDA device
- **THEN** the run fails before Qwen model loading
- **AND** the failure summary does not claim benchmark eligibility

#### Scenario: Dry run without CUDA
- **WHEN** `debug.dry_run: true` is configured and no CUDA device is visible
- **THEN** config and run-directory validation may complete without model
  loading
- **AND** no scored artifact set or benchmark eligibility claim is written

### Requirement: No public GPU-id config surface
The V1 data-parallel inference resource boundary SHALL be the visible CUDA environment, not stable config GPU ids.
The strict inference config MUST NOT add a public stable field such as
`gpu_ids` or `num_gpus` for this change. Tests MAY use private overrides for
device discovery, but production users restrict devices through
`CUDA_VISIBLE_DEVICES`.

#### Scenario: User restricts devices
- **WHEN** the user wants inference on four specific GPUs
- **THEN** the supported V1 mechanism is launching with
  `CUDA_VISIBLE_DEVICES=<four tokens>`
- **AND** the resolved inference config remains focused on semantic inference
  settings rather than cluster resource selection

### Requirement: Parallelism metadata in resolved runtime evidence
Every data-parallel inference run SHALL record its resolved parallelism policy.
The evidence MUST include parent visible CUDA tokens, active rank count,
per-device decode batch size, worker binding policy, shard-plan fingerprint,
and whether the run used direct single-process or controller/worker execution.

#### Scenario: Parallel run metadata
- **WHEN** a multi-rank inference run completes
- **THEN** manifest or provenance evidence records active rank count,
  per-device batch size, and rank-to-device mapping

#### Scenario: Direct single-rank path
- **WHEN** only one active rank is needed and HF is selected
- **THEN** manifest or summary evidence MAY record that the direct
  single-process path was used
- **AND** non-dry direct execution still records CUDA-required runtime evidence
- **AND** the manifest records direct-runtime device evidence after model load,
  including logical device and model first-parameter device when available

#### Scenario: Fresh single-rank vLLM worker
- **WHEN** only one active rank is needed and vLLM is selected
- **THEN** manifest or summary evidence records controller/worker execution
- **AND** the vLLM engine runs in a fresh rank-local worker process

### Requirement: Strict backend-discriminated inference config
`backend.type` SHALL accept `hf` and `vllm`. Shared `model` config SHALL contain
base-model identity, dtype, and processor policy, but MUST NOT contain
backend-only execution controls. HF MUST require a strict `backend.hf` block
containing `attn_implementation` and `patch_embed_linearization`; vLLM MUST
require a strict `backend.vllm` block containing finite
`gpu_memory_utilization` in `(0, 1]`. The non-selected backend block MUST be
rejected, unknown fields MUST fail validation, and arbitrary engine kwargs MUST
NOT be exposed as stable config.

#### Scenario: Explicit vLLM backend
- **WHEN** a config selects `backend.type: vllm` with a valid vLLM block
- **THEN** config validation succeeds without loading either backend

#### Scenario: HF with vLLM settings
- **WHEN** a config selects HF and includes `backend.vllm`
- **THEN** validation fails before model loading

#### Scenario: vLLM carries an HF attention implementation
- **WHEN** a config selects vLLM but includes `backend.hf` or legacy
  `model.attn_implementation`
- **THEN** validation fails instead of recording an ignored execution setting

### Requirement: Explicit dual-backend precision roles
Dynamic HF SHALL remain a first-class backend for direct base, DoRA, and
selected-token embedding-delta composition. vLLM SHALL remain a first-class
offline backend over a validated execution model. FP32 SHALL be the strict
cross-backend parity mode. BF16 vLLM MAY be used for supported throughput runs,
but its artifacts and documentation MUST NOT claim strict HF parity.

#### Scenario: Dynamic HF composed checkpoint
- **WHEN** `backend.type: hf` selects a base plus DoRA plus embedding delta
- **THEN** runtime loads that composition directly without requiring a vLLM
  execution-model snapshot

#### Scenario: BF16 vLLM throughput run
- **WHEN** `backend.type: vllm` and `model.dtype: bf16` are selected
- **THEN** inference remains executable but the run cannot satisfy the strict
  FP32 cross-backend parity gate

### Requirement: Deterministic evidence-bearing scored inference
Canonical inference MUST require `temperature: 0.0`, `top_p: 1.0`, scoring
enabled, token trace writing enabled, and parse diagnostics writing enabled.
Unsupported stochastic or evidence-disabling values MUST fail validation
instead of being ignored. Explicit positive repetition penalties remain
supported and MUST be projected to the selected backend.

#### Scenario: Stochastic config rejected
- **WHEN** temperature is nonzero or top-p is below one
- **THEN** validation fails with a deterministic-inference contract error

#### Scenario: Required trace disabled
- **WHEN** scoring is enabled but token trace writing is false
- **THEN** validation fails before runtime setup

### Requirement: Optional raw-model likelihood trace
Inference config SHALL expose `artifacts.include_raw_model_logprob` with default
`false`. Enabling it MUST request aligned raw-model likelihoods for every
generated non-pad token. It MUST NOT change selected-token scoring policy,
generation behavior, parsing, or evaluator input rows.

#### Scenario: Raw likelihood omitted by default
- **WHEN** the field is absent
- **THEN** inference runs with policy likelihood only and records raw likelihood
  tracing as disabled

#### Scenario: Raw likelihood requested
- **WHEN** the field is true
- **THEN** backend setup requires raw-likelihood support and failures are fatal

### Requirement: Qualified vLLM version
The runtime MUST accept only versions whose Qwen3-VL prompt, image, stop,
policy-logprob, raw-replay, CUDA-binding, process-mode, and cleanup probes are
recorded in a matching passed runtime-qualification receipt. The initial
candidate is `0.14.1`; it becomes accepted only after its Wave 0 receipt passes.
The receipt MUST bind the probe implementation, fixture, exhaustive source-base
snapshot assets, installed dependencies, every loaded source module from the
qualified vLLM/Transformers/PEFT/Qwen utility boundary after execution, named
required execution owners, effective engine arguments, and observed results.
Runtime matching MUST distinguish invariant semantics
from explicitly qualified engine-argument values. An unqualified version,
source drift, invariant mismatch, or unprobed argument value MUST fail before
engine construction and MUST NOT be treated as benchmark evidence.
Runtime-owned qualification receipts MUST live outside an active OpenSpec
change and MUST bind repository sources by repo-relative path plus hash and
installed sources by package-relative path plus hash. Recorded absolute paths
MAY remain as diagnostics but MUST NOT determine whether an unchanged checkout
or environment is qualified.

#### Scenario: Unqualified installed version
- **WHEN** the installed vLLM version is outside the qualified set
- **THEN** runtime fails with the observed and supported versions

#### Scenario: Version string matches but receipt does not
- **WHEN** vLLM reports `0.14.1` but installed source, source-base assets,
  tokenizer/processor/template identity, or invariant engine semantics do not
  match the passed qualification receipt
- **THEN** runtime fails before engine construction

#### Scenario: Qualified composed derivative
- **WHEN** a materialized model derives from the qualified source base,
  preserves its architecture/tokenizer/processor/template identities, and has
  a passed composition-fidelity receipt with dynamic-HF behavioral diagnostics
- **THEN** its distinct execution-model fingerprint is accepted without being
  mistaken for runtime qualification drift

#### Scenario: Unprobed concurrency value
- **WHEN** `max_num_seqs` differs from every value covered by a passed
  concurrency receipt
- **THEN** runtime fails before engine construction and names the qualified
  values

#### Scenario: Equivalent checkout at a different root
- **WHEN** the same qualified repository and package source bytes are available
  under different absolute installation paths
- **THEN** qualification succeeds through relative identities without editing
  the receipts

#### Scenario: OpenSpec change is archived
- **WHEN** the change that introduced vLLM is archived
- **THEN** runtime still resolves its accepted receipts from the stable
  inference-owned qualification directory
