## ADDED Requirements

### Requirement: Run Artifacts Record Executed Environment Provenance

Every training run SHALL record enough non-secret provenance to distinguish
the code and critical dependency state that actually executed. The run record
MUST identify the repository commit, whether relevant tracked or untracked
changes were present, a stable digest or explicit unavailable status for that
local state, and the resolved versions plus available source or binary
identities for critical training dependencies. Provenance collection MUST NOT
copy credentials or secret environment values into artifacts.

#### Scenario: A dirty checkout starts training

- **WHEN** relevant tracked or untracked changes are present
- **THEN** the run record MUST mark the executed tree dirty and retain a stable
  non-secret identity for the local state
- **AND** the repository commit alone MUST NOT be presented as complete
  execution provenance.

#### Scenario: A dependency identity is unavailable

- **WHEN** a critical source or binary identity cannot be obtained safely
- **THEN** the run record MUST preserve the dependency version and an explicit
  unavailable status with a bounded reason
- **AND** it MUST NOT substitute an assumed identity.

## MODIFIED Requirements

### Requirement: Minimal Inference Checkpoint Payloads

Scheduled checkpoint saving SHALL always materialize an independently loadable
minimal inference payload: a standard PEFT/DoRA adapter directory when adapter
training is enabled, a compact selected-token embedding-delta directory when
that trainable surface is enabled, and a self-authenticating manifest binding
those learned files. The inference payload MUST NOT save base-model weights and
MUST NOT require optimizer, scheduler, scaler, RNG, dataloader, iterator, or
sampler state to be loadable. New inference payloads MUST NOT require
`checkpoint.json`, `checkpoint_handoff.json`, setup receipts, duplicated
identity graphs, or an exact-training-state sibling.

Exact training state, when explicitly enabled, MUST be published as a typed
`training_state/` sibling rather than added to or inferred from the inference
payload. When disabled, no exact-state sibling or identity may be written.
Inference loaders MUST ignore the sibling and consume only explicit adapter and
selected-token payload paths covered by the inference manifest.

All ranks SHALL enter checkpoint-save operations in the same order when
distributed collectives require it, but only rank zero SHALL materialize the
durable inference payload. Supported distributed saving SHALL be limited to
replicated DDP: all ranks enter a pre-save barrier; rank zero unwraps the model
and writes only the configured adapter into staging using safe serialization
with embedding-layer saving disabled, then writes the optional compact selected
embedding delta and inference manifest. The adapter safetensor MUST contain
required LoRA A/B and DoRA magnitude-vector state and MUST NOT contain full
embedding, LM-head, or base-model tensors. Rank zero MUST atomically commit the
inference payload only after validation and broadcast a bounded success/error
descriptor to every rank. Every rank MUST continue or raise the same named
checkpoint-save error. Failed inference staging MUST be removed. Final and best
aliases MUST update only after all payloads required by the selected checkpoint
mode have committed successfully.

#### Scenario: Adapter-only checkpoint is saved

- **WHEN** adapter training reaches a scheduled checkpoint step without
  selected-token embedding training
- **THEN** the step directory MUST contain the standard adapter payload and
  manifest needed by the inference adapter loader
- **AND** MUST use `adapter_model.safetensors`
- **AND** MUST NOT contain a copy of base-model, full-embedding, or LM-head
  weights.

#### Scenario: Adapter and selected embeddings are saved

- **WHEN** both adapter and selected-token embeddings are trainable
- **THEN** the inference manifest MUST cover both payload directories
- **AND** inference MUST be able to compose them from explicit config paths.

#### Scenario: Exact state is disabled

- **WHEN** a scheduled checkpoint is saved with exact training state disabled
- **THEN** the inference payload and aliases MUST retain their normal behavior
- **AND** no `training_state/` sibling or exact-state identity may be written.

#### Scenario: Exact state is enabled

- **WHEN** a scheduled checkpoint is saved with exact training state enabled
- **THEN** the inference payload MUST commit independently before the typed
  sibling is published
- **AND** final or best aliases and the completed checkpoint event MUST update
  only after the exact-state publication succeeds.

#### Scenario: Exact-state publication fails on one rank

- **WHEN** the inference payload commits but one required rank fails during
  exact-state publication
- **THEN** every live rank MUST observe the same bounded checkpoint failure
  without hanging
- **AND** no final or best alias and no completed exact checkpoint event may
  reference that step
- **AND** the inference payload MUST remain typed as inference-only rather than
  being treated as partial exact state.

#### Scenario: Rank-zero inference payload save fails

- **WHEN** rank zero fails while saving or validating the inference payload
- **THEN** every rank MUST observe the same checkpoint-save failure without
  hanging
- **AND** no final or best alias MUST reference the incomplete step
- **AND** no committed partial inference checkpoint directory may remain.

#### Scenario: Existing checkpoint is used

- **WHEN** an older CoordExp-Swift checkpoint contains a standard adapter and
  optional compatible selected-token embedding payload plus extra historical
  metadata
- **THEN** inference MUST load the configured payload paths
- **AND** MUST ignore unrelated training-state or historical metadata
- **AND** MUST NOT require that metadata to be regenerated.
