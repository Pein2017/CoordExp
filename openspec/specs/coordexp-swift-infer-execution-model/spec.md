# coordexp-swift-infer-execution-model Specification

## Purpose
TBD - created by archiving change add-coordexp-swift-vllm-inference-backend. Update Purpose after archive.
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
The derived checkpoint MUST be reloaded through HF before adapter-enabled vLLM
support is claimed. Its merged DoRA target weights MUST be bitwise identical to
the owner-recorded target-dtype weights produced before snapshot publication.
Its selected embedding rows MUST be bitwise equal to the effective rows of the
existing dynamic HF composition after target-dtype casting. Both models MUST
retain tied input/output storage and use identical prompt ids on one real
multimodal no-resize fixture.

The same fixture MUST also record dynamic-HF versus materialized-HF FP32
fixed-prefix full-vocabulary and selected-vocabulary logit differences plus
greedy generated ids. Those values are behavioral diagnostics, not composition
fidelity failures, because unmerged DoRA and one folded BF16 linear weight use
different floating-point operation orderings. The diagnostic MUST retain the
reference thresholds `rtol=1e-4`, full-vocabulary `atol=5e-3`, and
selected-vocabulary `atol=2e-3` without claiming execution identity when they
are exceeded.

Canonical HF inference MUST continue to use dynamic DoRA plus the selected-token
delta. Materialized HF MUST be the exact executable-model oracle for vLLM. The
receipt MUST record fixture/source fingerprints, merged-target identities,
compared positions, shapes, dtypes, maximum absolute/relative differences,
generated ids, composition checks, behavioral checks, and thresholds.
Passed composition receipts used as runtime authority MUST be preserved in a
stable inference-owned qualification directory and keyed by composition key.
When a newly materialized cache entry has no local composition sidecar, runtime
MAY bind a durable receipt only after exact linkage validation against the new
execution-model receipt. It MUST NOT infer, weaken, or silently regenerate the
proof.

#### Scenario: Dynamic and materialized BF16 behavior differs
- **WHEN** exact state-composition checks pass but dynamic and materialized HF
  logits exceed a reference threshold or greedy ids differ
- **THEN** the receipt records the difference without replacing canonical HF or
  rejecting the correctly materialized execution model

#### Scenario: Reloaded merged target differs
- **WHEN** any reloaded materialized DoRA target weight differs from the
  owner-recorded pre-save merged target identity
- **THEN** the execution model is rejected before vLLM loading

#### Scenario: Folded selected row differs after target-dtype cast
- **WHEN** any selected embedding row is not bitwise equal between dynamic and
  materialized HF after target-dtype casting
- **THEN** the execution model is rejected before vLLM loading

#### Scenario: Clean cache reuses durable FP32 proof
- **WHEN** a clean checkout materializes the byte-identical FP32 composition
  and the cache has no local composition sidecar
- **THEN** runtime binds the durable content-addressed receipt after exact
  composition and snapshot linkage validation
