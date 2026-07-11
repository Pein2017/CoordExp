## MODIFIED Requirements

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
strings/ids, tensor key, shape, dtype, and tied-weight semantics.

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
