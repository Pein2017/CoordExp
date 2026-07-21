# coordexp-swift-infer-execution-model Specification

## Purpose
Define how backends that cannot load CoordExp DoRA and selected-token payloads
resolve, verify, and execute an immutable standard-model snapshot without
changing the trained composition.
## Requirements
### Requirement: Exact execution-model composition
The system SHALL resolve one immutable execution model for the configured base,
optional DoRA adapter, and optional selected-token embedding delta. Base-only
vLLM MAY use the base directory directly only after a receipt hashes every
required model shard plus config, tokenizer, and processor identity. Any
configured adapter or embedding delta MUST be materialized into a standard HF
checkpoint before vLLM loading. Every worker MUST revalidate the supplied
execution-model receipt before engine construction. The system MUST NOT drop,
approximate, or natively reinterpret DoRA or the selected-token delta.

#### Scenario: Base-only execution
- **WHEN** vLLM inference declares no adapter and no embedding delta
- **THEN** the execution model resolves directly to the content-hashed base
  model and workers revalidate its immutable receipt before engine loading

#### Scenario: Base shard changes after controller resolution
- **WHEN** a base weight shard changes in place before a worker starts
- **THEN** worker receipt validation fails before engine construction

#### Scenario: DoRA and delta execution
- **WHEN** vLLM inference declares a DoRA adapter and selected-token delta
- **THEN** DoRA is merged first and the delta is folded exactly once into the
  tied input/output weight before the standard checkpoint is published

#### Scenario: Qualified source base with composition-proven derivative
- **WHEN** a composed execution model has a different output fingerprint but
  binds the qualified source-base fingerprint and a passed composition-fidelity
  receipt
- **THEN** runtime qualification validates the source family while per-run
  execution identity validates the exact materialized output

### Requirement: Content-addressed materialization identity
Each derived execution model MUST live under
`model_cache/coordexp_swift/vllm_materialized/<composition-key>/snapshot/`.
The composition key MUST bind base model weight shards, base config, tokenizer identity, every
processor/chat-template asset copied into or loaded from the snapshot, adapter
config and tensor payloads, embedding metadata and tensor payloads, target
dtype, materialization algorithm version, and relevant Transformers and PEFT
versions. The materialization receipt MUST separately record an exhaustive
fingerprint of the published snapshot bytes. Output location and worker count
MUST NOT alter either semantic identity.

#### Scenario: Identical composition reuses cache
- **WHEN** two runs resolve byte-identical model, adapter, delta, dtype, and
  algorithm identities
- **THEN** both resolve the same composition key and snapshot fingerprint and
  the second run validates and reuses the existing execution model

#### Scenario: Payload changes in place
- **WHEN** an adapter or delta tensor changes while its path stays constant
- **THEN** the execution-model composition key changes

#### Scenario: Receipt avoids self-referential hashing
- **WHEN** the published execution snapshot is fingerprinted
- **THEN** `coordexp_materialization.json` is stored beside `snapshot/` and is
  not part of the exhaustive snapshot-file manifest it records

### Requirement: Atomic single-builder publication
Materialization MUST use an inter-process lock at
`.locks/<composition-key>.lock` and a unique staging directory. The final
composition directory MUST become visible only after the snapshot, receipt,
hashes, tied-weight checks, and residue checks pass. Multi-rank
controller execution MUST materialize once before launching GPU workers.

#### Scenario: Concurrent cache miss
- **WHEN** multiple processes request the same missing fingerprint
- **THEN** one process builds and publishes while the others wait and then
  validate the completed entry

#### Scenario: Failed build
- **WHEN** model merging, delta folding, saving, or validation fails
- **THEN** no completed fingerprint directory is published

### Requirement: Strict cache-hit validation
A cache hit MUST validate its materialization manifest, source fingerprint,
required checkpoint files, file hashes, model config, tokenizer identity,
processor/chat-template identity, absence of adapter or parametrization
residue, and tied input/output semantics. A corrupt completed entry MUST fail
with a concrete contract error and MUST NOT be silently repaired or partially
reused.

#### Scenario: Corrupt cached tensor
- **WHEN** a completed cached model shard hash differs from its manifest
- **THEN** execution fails before vLLM engine construction

#### Scenario: Processor file changes in place
- **WHEN** a copied preprocessor, processor, or chat-template asset changes
- **THEN** the execution-model fingerprint changes or cache validation fails
  before engine construction

### Requirement: Owner-defined deterministic composition
Adapter identity MUST be inspected by the adapter owner and MUST hash
`adapter_config.json` plus every adapter tensor file. DoRA composition MUST load
the adapter as frozen through PEFT and call
`merge_and_unload(safe_merge=True, adapter_names=["default"])`. Selected-token
identity MUST be inspected by the embedding-delta owner and MUST hash metadata,
tensor payload, tensor key/shape/dtype, token ids/strings, base identity, and
tokenizer identity. The base MUST be loaded in target dtype before merge, and
the FP32 delta MUST be folded directly into target-dtype tied rows without a
later whole-model cast.

#### Scenario: Adapter tensor payload changes
- **WHEN** any adapter tensor file changes while the adapter path is unchanged
- **THEN** adapter identity and the execution-model composition key change

#### Scenario: Target-dtype selected-row fold
- **WHEN** an FP32 selected-token delta is materialized into a BF16 execution model
- **THEN** each selected tied row is computed once against the BF16 base row and
  stored in BF16 without a subsequent whole-model dtype conversion

### Requirement: Composition fidelity and behavioral diagnosis

The system SHALL resolve one immutable execution model for the configured base,
optional DoRA adapter, and optional selected-token embedding delta. Any
configured adapter or embedding delta MUST be materialized into a standard HF
checkpoint before vLLM loading. DoRA MUST be merged first and the selected-
token delta MUST be folded exactly once into the tied input/output weight.

The blocking execution receipt MUST bind and validate the current source
payload identities, tensor compatibility, target dtype, merge/fold outcomes,
tied-weight structure, standard Qwen3-VL snapshot files, absence of adapter
residue, exhaustive snapshot identity, and atomic cache publication. A
composition-fidelity or HF/vLLM behavioral-comparison receipt MUST NOT be
required for ordinary vLLM loading.

Before publication, adapter merge evidence MUST exist exactly when an adapter
identity is configured and MUST bind that identity with status `merged`.
Embedding-delta fold evidence MUST exist exactly when a delta identity is
configured and MUST bind that identity, status `folded`, one row addition, and
tied input/output storage. The complete materialization MUST also attest tied
input/output structure. Missing, unexpected, or identity-mismatched outcomes
MUST prevent cache publication.

Explicit composition probes MAY bind and validate exact merged-target,
selected-row, prompt, logit, or generated-token comparisons as optional
diagnostics. A failed or stale optional comparison MUST NOT invalidate a
structurally valid execution model unless it demonstrates that the current
materialized snapshot violates one of the blocking composition invariants.

#### Scenario: Valid composition without comparison sidecar

- **WHEN** base, DoRA, and embedding delta materialize into a structurally
  valid tied Qwen3-VL snapshot but no composition-fidelity sidecar exists
- **THEN** ordinary vLLM runtime accepts the execution-model receipt and
  proceeds to engine loading

#### Scenario: Stale behavioral comparison

- **WHEN** an old HF comparison receipt is missing, source-stale, or exceeds a
  historical numerical tolerance while the current structural receipt passes
- **THEN** the condition is recorded only when inspected diagnostically
- **AND** does not prevent ordinary vLLM execution

#### Scenario: Invalid selected-token fold

- **WHEN** selected-token ids, shapes, dtype, base/tokenizer identity, or tied
  input/output structure is incompatible during materialization
- **THEN** materialization fails and no completed execution snapshot is
  published

#### Scenario: Dynamic and materialized BF16 behavior differs

- **WHEN** current structural composition checks pass but dynamic and
  materialized HF logits or greedy ids differ
- **THEN** an explicit comparison records the behavioral difference without
  rejecting the structurally valid execution model

#### Scenario: Reloaded merged target differs

- **WHEN** a reloaded materialized DoRA target differs from the merge outcome
  bound by the current execution receipt
- **THEN** the execution model is rejected before vLLM loading

#### Scenario: Folded selected row differs after target-dtype cast

- **WHEN** a selected embedding row does not equal the current one-addition
  fold outcome after target-dtype casting
- **THEN** materialization fails before completed snapshot publication

#### Scenario: Clean cache reuses durable FP32 proof

- **WHEN** a clean checkout reuses a structurally valid content-addressed cache
  entry and a linked durable FP32 comparison is available
- **THEN** runtime may expose that comparison as diagnostic evidence
- **AND** cache acceptance remains based on the current structural receipt
