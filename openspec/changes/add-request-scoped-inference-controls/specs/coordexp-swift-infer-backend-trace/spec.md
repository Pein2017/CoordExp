## ADDED Requirements

### Requirement: Request-scoped decode generation policy
The system SHALL represent generation behavior as an immutable backend-neutral
request policy tagged as `greedy` or `sampled`. The policy MUST include positive
`max_new_tokens`, positive finite `repetition_penalty`, finite `temperature`,
finite `top_p` in `(0, 1]`, and the declared
`temperature_top_p_categorical_v1` sampling profile. Greedy mode MUST require `temperature: 0.0`,
`top_p: 1.0`, and no sampling seed. Sampled mode MUST require positive
`temperature` and exactly one non-negative signed-64-bit-safe sampling seed on
each request. The default public inference path MUST remain greedy.

All requests in one Hugging Face generation call MUST use identical policy
fields. Sampled requests in the same call MAY use different request-owned
sampling seeds. Invalid or batch-incompatible policies MUST fail before model
generation.

The effective generation configuration MUST NOT inherit undeclared model-owned
sampling behavior. The declared profile MUST disable top-k filtering, typical,
minimum-probability, epsilon, and eta sampling cutoffs; use one beam and one
returned sequence; disable constraints, forced/suppressed-token rules, and
watermarking; and explicitly own Qwen end-of-assistant and padding identifiers.
The execution receipt MUST serialize every effective argument capable of
changing token selection or stopping.

#### Scenario: Explicit default greedy request
- **WHEN** the public inference pipeline constructs a decode request from its
  current default generation configuration
- **THEN** the request uses a validated greedy policy with neutral sampling
  fields and no sampling seed
- **AND** generated behavior remains deterministic greedy decoding

#### Scenario: Greedy authored sampling fields normalized
- **WHEN** a valid greedy inference config contains a non-neutral sampling-only
  value that the current deterministic backend does not execute
- **THEN** resolved-config provenance preserves the authored value
- **AND** the request execution policy normalizes `temperature` to `0.0` and
  `top_p` to `1.0`
- **AND** the execution receipt records those normalized executed values

#### Scenario: Valid sampled batch
- **WHEN** four requests use one identical sampled policy and each carries one
  valid request-owned sampling seed
- **THEN** backend validation accepts the batch for one generation call

#### Scenario: Sampled request without seed
- **WHEN** a sampled request omits its sampling seed
- **THEN** validation fails before model generation and names the request

#### Scenario: Greedy request with seed
- **WHEN** a greedy request carries a scientific sampling seed
- **THEN** validation fails before model generation

#### Scenario: Incompatible policies in one batch
- **WHEN** requests in one generation call differ in mode, temperature, top-p
  probability, repetition penalty, or maximum generated-token count
- **THEN** validation fails before model generation and identifies the differing
  policy fields

### Requirement: Request-owned sampled randomness
For sampled generation, the Hugging Face backend SHALL construct one
device-compatible random-number generator per request from that request's
sampling seed. Because the installed stock Qwen/Hugging Face sampler has no
per-request generator seam, the backend SHALL use the authorized local
score-preserving custom generation branch inside the existing batched backend.
That branch MUST reuse prepared logits processors, stopping criteria, cache
updates, score timing, and decoder-only output shape while binding each
categorical draw to the corresponding request generator. The backend MUST NOT use process-global reseeding, batch-position
seeds, or one shared batch generator as request-level evidence. When results are
joined by request identity, same-policy and same-seed generated token
identifiers MUST be invariant to compatible request reordering and batch
scheduling.

Support for request-owned sampled randomness MUST be established by an executed
probe against the installed Hugging Face and Compute Unified Device Architecture
runtime. Source inspection, a fake model, or a batch-size-one loop MUST NOT be
accepted as attestation for the four-request batched claim. If the installed
custom branch cannot satisfy the probe, sampled backend support MUST remain
unavailable rather than silently changing execution semantics.

#### Scenario: Same-seed replay
- **WHEN** four sampled requests are executed twice with identical request
  identities, policies, prompts, and seeds
- **THEN** generated token identifiers match exactly after joining results by
  request identity

#### Scenario: Reversed scheduling replay
- **WHEN** the same four sampled requests are executed again in reversed request
  order under the same policy and seeds
- **THEN** generated token identifiers match the original results after joining
  by request identity

#### Scenario: Natural three-request tail batch
- **WHEN** a pre-materialized research schedule ends in one three-request tail
  batch using the same sampled policy and request-owned generator semantics
- **THEN** that cardinality passes its own replay and generator-binding
  attestation before metric-bearing use
- **AND** it is not represented as a padded four-request batch or a sequence of
  batch-size-one calls

#### Scenario: Three-request versus four-request cardinality parity
- **WHEN** the same three request identities, prompts, policies, and seeds are
  executed once as a natural three-request batch and once in a four-request
  attestation batch with one predeclared independent fourth request
- **THEN** the shared requests have identical generated token identifiers after
  joining by request identity
- **AND** selected-token score traces and recomputed row-local scores agree
  after float32 promotion with absolute and relative tolerance `1e-6`
- **AND** the independent fourth request is not treated as padding or included
  in metric-bearing tail results

#### Scenario: Distinct request-to-generator seed mapping
- **WHEN** four sampled requests carry four distinct valid seeds
- **THEN** the generator object at each request execution index reports the seed
  owned by the request at that same index
- **AND** no output difference is required solely because two seeds differ
- **AND** scientific output-diversity acceptance remains outside this stable
  backend contract

#### Scenario: Unsupported custom sampled runtime
- **WHEN** the custom branch cannot consume independent request-owned generators
  in one four-request scored generation call while preserving required runtime
  semantics
- **THEN** the sampling attestation fails
- **AND** the implementation does not fall back to process-global reseeding,
  shared batch randomness, or batch size one

### Requirement: Result-bound decode execution receipt
Every decode result SHALL contain an immutable execution receipt bound to that
request and generated output. The receipt MUST record schema version, request
identity, complete decode generation policy and fingerprint, sampling seed or
null, random-generator kind, device, and initial seed or null, request execution
index, batch request-order fingerprint, normalized serializable generation
arguments actually executed, prompt token count and token-identifier hash,
generated token count and token-identifier hash, backend identity, backend mode,
response family, stop reason, and receipt fingerprint.

The receipt MUST additionally bind the sampling-profile identifier and
fingerprint; model-identity and tokenizer-identity fingerprints; the custom
sampler's versioned algorithm identity and executable code hash when sampled;
the active attention implementation; runtime package versions relevant to
generation; the installed-runtime identity fingerprint; model evaluation mode;
the complete sanitized effective generation arguments; and a canonical hash of
the enclosing result's float32-promoted token score trace. Receipt fingerprints MAY differ
across compatible request reorderings because execution indices and batch-order
fingerprints differ; replay comparison MUST join generated identifiers by
request identity and validate each run's local order binding separately.

The receipt fingerprint MUST cover every stable receipt field except itself.
Decode-result validation MUST reject any disagreement between the receipt and
the enclosing result's request identity, policy fingerprint, prompt tokens,
generated tokens, score trace, model identity, tokenizer identity, backend
identity, runtime identity, custom-sampler identity, or stop reason. Before sampled generation,
the backend MUST reject a generator whose reported initial seed or list index
does not match the request seed and execution index. The batch request-order
fingerprint MUST cover the ordered request identities, prompt-token hashes, and
common policy fingerprint. In-memory object addresses and process-global random
state MUST NOT serve as generator identity.

Receipt construction MUST defensively convert all nested fields to recursively
immutable canonical values before fingerprinting. A frozen outer dataclass that
retains caller-owned mutable mappings or sequences does not satisfy this
requirement.

#### Scenario: Valid greedy receipt
- **WHEN** a greedy request completes
- **THEN** its result contains a receipt with the executed greedy policy,
  null sampling seed, null random-generator evidence, bound prompt and generated
  token hashes, and matching backend and stop evidence

#### Scenario: Valid sampled receipt
- **WHEN** a sampled request completes with a request-owned generator
- **THEN** its result receipt records the exact request seed, generator kind and
  device, initial seed read from the actual generator, execution index, batch
  request-order fingerprint, sampled generation arguments, and output binding

#### Scenario: Swapped generator order
- **WHEN** two sampled generator objects are swapped while request seeds and
  request order remain unchanged
- **THEN** backend validation fails before generation because actual generator
  initial seeds do not match their request execution indices

#### Scenario: Swapped result receipt
- **WHEN** a receipt from one request or generated output is attached to another
  decode result
- **THEN** validation fails before artifact or research-attempt materialization

#### Scenario: Receipt serialization replay
- **WHEN** a receipt is serialized to and reloaded from its canonical
  machine-readable representation
- **THEN** its fingerprint recomputes identically and all result bindings remain
  valid

#### Scenario: Cross-model or cross-sampler receipt swap
- **WHEN** a structurally compatible receipt from another model, tokenizer, or
  custom-sampler implementation is attached to a result
- **THEN** validation fails on the corresponding identity or code fingerprint

#### Scenario: Nested receipt mutation is ineffective
- **WHEN** caller-owned mappings or sequences used to construct a receipt are
  mutated after receipt construction
- **THEN** the receipt's canonical values and fingerprint remain unchanged

### Requirement: Score, cache, stop, and padding attestation
Before sampled support is exposed to research orchestration, the installed-model
CUDA gate SHALL attest the custom branch at batch sizes four and three using the
exact configured attention implementation. It MUST validate score-tuple length
and generated-step alignment, transition-score recomputation, cache-update
behavior, Qwen `<|im_end|>` retention and stop reason, finished-row sample-then-
pad behavior, generated output type and shape, and request-generator draw
progression. A fixed-prefix pre-draw comparison MUST show that custom and stock
generation see equivalent processed logits before categorical selection.

Same-seed and reversed-order comparisons MUST bind exact generated token
identifiers and compare selected-token score traces and recomputed
`compact-object-selected-token-score-v1` row-local scores after float32
promotion with absolute and relative tolerance `1e-6`. Batch-size-one execution
MAY be used only as a non-claiming pre-draw diagnostic fixture; it MUST NOT be a
runtime fallback or metric-bearing arm.

One production verifier invocation MUST load the model exactly once and execute
the complete batch-size-four forward/reversed, batch-size-three
forward/reversed, stock/custom processed-logit parity, capability mint, and
capability-gated four-request sampled-production replay for each exact
calibration temperature `0.2`, `0.4`, and `0.6`. The production command MUST
NOT expose a single-temperature or incomplete-aggregate write mode.

For each policy, the admitted sampled-production call MUST exactly replay
request order, generated result artifacts, result-receipt bindings, and
canonical `compact-object-selected-token-score-v1` evidence from that policy's
attested four-request forward case. It MUST expose a pre-generation live-state
seal with the frozen runtime's exact 589 adapter-plus-selected-token-embedding
tensors, 80,248,832 payload bytes, non-negative payload-hash and total-seal
timings, and explicit evidence that base-model tensor values were not hashed.

Only after all three policies pass MAY the verifier perform one append-only
write. The typed aggregate MUST contain exactly three independently
fingerprinted policy entries in canonical temperature order. Every entry MUST
preserve its sampled-runtime attestation bundle and fingerprint unchanged and
MUST contain a separately fingerprinted admitted-production replay bound to
that same bundle and exact decode-generation-policy fingerprint. The aggregate
MUST reject missing, duplicate, or unexpected policies and bundle/replay swaps
across policies, including swaps followed by recomputation of outer
fingerprints. Missing, mis-scoped, or mismatched live-state diagnostics, a
replay mismatch, or failure of any one policy MUST stop execution before any
output is written.

A persisted aggregate is evidence, not a serializable capability. A worker MAY
obtain sampled-production admission only through the public verifier-owned
rebind seam, selecting one exact attested policy fingerprint. Rebind MUST fully
validate all three bundles, derive model, tokenizer, and generation identities
from immutable backend-owned runtime bindings, validate the current execution
device, installed runtime, custom sampler, model structure and configuration,
active adapter state, behavior-changing tokenizer state, and exact bounded
adapter-plus-selected-token-embedding values against the persisted portable
live-state seal, and then mint a new non-serializable capability bound to that
worker's exact backend, model, and tokenizer objects. A bare bundle, wildcard
policy, caller-supplied stale identity, or capability used with another backend
object MUST fail closed.

#### Scenario: Score alignment drift
- **WHEN** generated identifiers replay but score steps, selected-token log
  probabilities, or recomputed row-local scores exceed the frozen tolerance
- **THEN** sampled support remains unavailable

#### Scenario: Stop or padding drift
- **WHEN** the custom branch loses `<|im_end|>`, changes stop reason, misaligns
  score steps, or advances finished rows differently from declared sample-then-
  pad semantics
- **THEN** sampled support remains unavailable

#### Scenario: Serialized evidence without admitted production execution

- **WHEN** the four attestation layouts pass but the capability-gated production
  replay is absent, differs from the four-request forward case, or reports the
  wrong live payload identity
- **THEN** no canonical attestation output is written
- **AND** serialized attestation evidence alone does not grant production
  admission

#### Scenario: Incomplete multi-policy verifier execution

- **WHEN** any one of temperatures `0.2`, `0.4`, or `0.6` is absent, duplicated,
  unexpected, or fails its complete attestation or admitted replay
- **THEN** the verifier performs no canonical append-only write

#### Scenario: Equivalent worker runtime rebind

- **WHEN** a new worker loads the complete aggregate, selects one exact policy
  fingerprint, and presents an equivalent backend-owned runtime and portable
  live state
- **THEN** the verifier mints a new process-local capability bound only to that
  worker's backend, model, and tokenizer objects

#### Scenario: Persisted or active runtime mismatch

- **WHEN** aggregate, bundle, replay, policy, model, tokenizer, generation
  configuration, device, installed runtime, adapter, selected-token embedding,
  or backend-object identity differs from the attested contract
- **THEN** rebind or sampled production fails before generation
