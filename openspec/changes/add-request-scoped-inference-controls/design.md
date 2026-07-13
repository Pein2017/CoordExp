## Context

The current Hugging Face (HF) inference backend receives a batch of
`DecodeRequest` values but owns generation behavior partly through hard-coded
arguments. It always executes greedy decoding and its result records only a
run-provided generation-config fingerprint. Consequently, a future research
runner cannot prove which sampled policy and random seed produced each result,
and batch- or process-owned randomness could silently change paired results
when requests are regrouped or reordered.

The current prompt builder creates an open assistant turn by removing authored
assistant messages and applying the Qwen chat template with a generation
prompt. A spatial-scope experiment also needs to place already accepted object
rows inside that same open assistant turn. Representing those rows as another
chat message would add a new boundary; independently tokenizing and concatenating
the rows would assume a token boundary that may not exist.

These are stable execution seams shared by more than one possible research
question. The scientific protocol itself remains owned by `research/`,
`scripts/research/`, and `src/analysis/`.

## Goals / Non-Goals

**Goals:**

- Represent greedy and sampled generation as one explicit, immutable,
  backend-neutral request contract.
- Make sampled randomness request-owned and reproducible after joining results
  by request identity, independent of compatible batch scheduling or request
  order.
- Bind executed generation evidence to each decode result through an immutable
  receipt that can be validated and serialized.
- Append optional canonical text inside the existing open assistant turn while
  preserving exact full-prompt text and token evidence.
- Preserve current production behavior when the new controls are absent:
  deterministic greedy generation, Qwen `<|im_end|>` stopping, no-resize image
  processing, and per-device batch semantics.
- Fail before scientific interpretation when the installed HF runtime cannot
  attest request-scoped sampling for a four-request batch.

**Non-Goals:**

- Add sampling fields to the public `InferConfig` or promote a sampled
  production preset.
- Add Vision Language Model (VLM) architecture changes, training objectives,
  visual-feature interventions, or model-specific research mechanisms.
- Encode cohort identities, dense-scene labels, random-seed derivation
  namespaces, temperature search, experiment-arm scheduling, tile geometry,
  masked-full-canvas construction, accepted-row admission, matching,
  non-maximum suppression, bootstrap statistics, or scientific thresholds in
  stable inference code.
- Change canonical one-input-row-to-one-output-row raw or scored artifacts.
- Add vLLM execution, batch-size-one fallback, or process-global random-state
  compatibility.
- Claim that this compatibility change alone makes any research unit
  execution-ready.

## Decisions

### 1. Use a tagged request generation policy, not independent loose knobs

Introduce an immutable `DecodeGenerationPolicy` owned by
`src/inference/backend.py`. Its operational meaning is the complete set of
batch-compatible arguments that determine how one request is decoded:

```text
DecodeGenerationPolicy
  mode: greedy | sampled
  sampling_profile: temperature_top_p_categorical_v1
  max_new_tokens: positive integer
  repetition_penalty: positive finite number
  temperature: zero for greedy, positive finite number for sampled
  top_p: one for greedy, value in (0, 1] for sampled
```

`DecodeRequest` owns one policy plus `sampling_seed`. A sampled request requires
one non-negative signed-64-bit-safe seed; a greedy request forbids a scientific
sampling seed. All requests in one HF call must have identical policy fields,
while sampled requests may and normally will have different seeds.

The public pipeline constructs the explicit greedy execution policy from its
current resolved config, so public behavior does not change. Sampling-only
values authored in a greedy config remain present in resolved-config
provenance, but the executed policy normalizes them to `temperature: 0.0` and
`top_p: 1.0`; the execution receipt records the normalized values actually
used. A research runner may construct sampled requests directly without making
research controls permanent public configuration.

The named `temperature_top_p_categorical_v1` profile is part of the stable
policy meaning. It MUST build an explicit effective generation configuration
rather than inherit the model's `generation_config.json`. In sampled mode it
executes categorical sampling after repetition-penalty, temperature, and
nucleus-probability processing with top-k disabled (`top_k: 0`),
`typical_p: 1.0`, no minimum-probability or epsilon/eta cutoff, one beam, one
returned sequence, no constraints, no forced/suppressed-token policy, no
watermark, and explicit Qwen `<|im_end|>` and padding identifiers. Greedy mode
uses the corresponding neutral profile. Every effective field that can alter
token selection or stopping is serialized in the execution receipt. This
prevents the active Qwen model's authored sampling defaults from silently
becoming part of the experiment.

**Alternative considered: add `do_sample`, `temperature`, `top_p`, and `seed`
as unrelated `DecodeRequest` fields.** Rejected because contradictory states
such as greedy decoding with a positive temperature remain representable and
validation becomes distributed.

**Alternative considered: one process or batch seed.** Rejected because output
then depends on batch membership, position, worker scheduling, or unrelated
requests and cannot support paired request-level estimands.

### 2. Use a narrow score-preserving custom sampler and treat it as an executed runtime claim

Installed Transformers `4.57.1` and the active Qwen generation class use the
stock `GenerationMixin._sample` implementation, whose categorical draw calls
`torch.multinomial` without a generator argument. The normal generation path
therefore cannot consume one request-owned generator per row. The user has
authorized a narrow custom branch inside the existing `HFGenerateBackend`; this
is not a second inference pipeline or a batch-size-one fallback.

For sampled mode, the backend constructs one CUDA-device-compatible
random-number generator per request from that request's seed. A local
`custom_generate` callable with an explicit `request_generators` parameter
reuses Hugging Face input preparation, cache updates, logits processors,
stopping criteria, output-score structure, and decoder-only output type, while
replacing only the final categorical draw with one generator-bound draw per
request. No process-global reseeding is allowed. Finished rows retain stock
sample-then-pad semantics so generator draw counts, score timing, Qwen
`<|im_end|>` handling, and current token-trace alignment remain comparable to
the installed path.

This design is conditional on an executable attestation probe using the
implemented custom branch, installed Transformers implementation, the exact
configured attention implementation used by the metric-bearing run, and Compute
Unified Device Architecture (CUDA) runtime. The current frozen checkpoint
configuration uses Scaled Dot-Product Attention (`sdpa`); Flash Attention 2 is
not covered unless separately attested. The
probe uses four requests in one call, then repeats the same
request identities and seeds in the same and reversed scheduling orders. Joined
generated token identifiers must agree exactly for same-seed requests. Each
actual generator object's reported initial seed and generator-list index must
match the request at the corresponding execution index. Output diversity across
different seeds is not a stable backend acceptance rule; the frozen research
temperature-calibration panel owns that stochastic scientific gate.

If the custom branch cannot preserve request-owned replay, reversed-order
replay, scored batched output, cache/stopping/padding semantics, and exact
request-to-generator binding, sampled backend support remains unavailable and
the research unit stops. It must not silently switch to batch size one, loop
over requests while reporting one batched call, or reseed global state.

**Alternative considered: adopt vLLM solely for sampling.** Rejected for this
unit because the stable inference configuration intentionally reserves vLLM,
and changing backend ownership would add prompt, score-trace, and artifact
parity questions unrelated to the scientific hypothesis.

### 3. Return a result-bound decode execution receipt

Add an immutable `DecodeExecutionReceipt` owned by the backend and required by
`DecodeResult`. Its stable operational fields are:

```text
schema_version
request_id
decode_generation_policy and its fingerprint
sampling_seed or null
random_generator_kind, random_generator_device, and
  random_generator_initial_seed or null
request_execution_index and batch_request_order_fingerprint
executed_generation_arguments
prompt_token_count and prompt_token_identifiers_hash
generated_token_count and generated_token_identifiers_hash
score_trace_count and canonical_float32_score_trace_hash
backend, backend_mode, and response_family
model_identity_fingerprint and tokenizer_identity_fingerprint
sampling_profile_fingerprint and installed_runtime_identity_fingerprint
custom_sampler_identity and custom_sampler_code_hash or null
stop_reason
receipt_fingerprint
```

The token hashes use Secure Hash Algorithm 256-bit (SHA-256) over a canonical
JavaScript Object Notation (JSON) representation of the integer identifiers.
The receipt fingerprint covers every stable field except itself. Validation
binds receipt request identity, policy fingerprint, prompt hash, generated hash,
backend identity, and stop reason to the enclosing `DecodeResult`.

`executed_generation_arguments` contains the normalized serializable arguments
actually passed to generation, excluding non-serializable model tensors and
generator objects. Generator kind, device, initial seed read back from the
actual generator object, and its request execution index attest how request
randomness was materialized without relying on unstable in-process object
addresses. Before generation, the backend requires the generator at each list
index to report the seed owned by the request at that same execution index. The
batch request-order fingerprint binds the ordered request identities, prompt
hashes, and common policy fingerprint for the actual call. Behavioral replay is
still required; receipt self-consistency alone is not claimed to prove causal
generator-to-output ownership.

The receipt also records the sampling-profile identifier and fingerprint,
model-identity and tokenizer-identity fingerprints, attention
implementation, installed Transformers, PyTorch, Flash Attention and CUDA
versions, model training/evaluation mode, logical generator device, and
effective sanitized generation arguments. When sampled, the custom sampler is
identified by a versioned algorithm identifier and a hash of its executable
source/code payload rather than a function name alone. The bound score trace is
the exact `DecodeResult` token trace after float32 promotion, including token
identifier, generated-step index, stop/padding classification, and log
probability. Reversed request order is expected
to change execution indices, order fingerprints, and receipt fingerprints;
attestation compares generated identifiers by request identity while
separately validating each run's local order binding.

Receipt values are recursively canonical and immutable. Mapping- or
sequence-shaped inputs are defensively converted to canonical tuple-based
values before fingerprinting; a frozen outer dataclass containing mutable nested
objects is not sufficient.

Research code may embed this receipt inside a richer per-attempt record. The
stable receipt does not own arm, cell, spatial-scope, seed-namespace, or
scientific-result fields.

**Alternative considered: record only one run-level generation policy.**
Rejected because it cannot detect ignored, missing, or swapped per-request
seeds.

**Alternative considered: let the backend write a receipt sidecar directly.**
Rejected because a logging side effect can drift from returned results and
complicates sharded ownership. Returning the receipt makes result-receipt
binding testable at the interface.

### 4. Model prompt conditioning as an open-assistant continuation

Introduce immutable `AssistantContinuation` input owned by
`src/inference/prompt.py`. Its operational meaning is canonical text that must
continue the assistant response already opened by the Qwen generation prompt;
it is not a completed assistant message and not a new conversation turn.

Prompt construction proceeds as follows:

1. Build the ordinary open-assistant `chat_text` through the existing template.
2. Locate and record the ordinary prompt's final open-assistant turn and its
   content-start boundary. Earlier completed system and user turns may contain
   their required terminators.
3. Validate the continuation as nonempty canonical text containing no image
   placeholder, assistant terminator, end-of-sequence token, or chat-turn
   opener.
4. Require the interval from the final assistant-content start to the
   continuation start to contain no assistant terminator, end-of-sequence token,
   new turn opener, or pre-existing assistant content.
5. Append the text directly to the open assistant turn with no inserted
   separator or terminal token.
6. Tokenize the complete combined text with special-token insertion disabled.
   Never assume `tokenize(base) + tokenize(continuation)` equals full-prompt
   tokenization.
7. Fail if tokenization truncates or the combined prompt no longer contains
   exactly one image placeholder.

Existing field ownership is frozen: `prompt_text` remains the authored task
prompt, while `chat_text` remains the exact full ordinary or continued chat
text. Continued prompt evidence serializes `chat_text` explicitly as
`full_chat_text` without repurposing `prompt_text`, and records the full token
identifiers, full-prompt fingerprint, continuation text hash, open-assistant
content start, continuation byte and character spans, and a token impact span.
The token impact span begins at the longest common token prefix between the
ordinary and continued full prompts, so it honestly includes any boundary token
that full-prompt retokenization changed. It is not claimed to be an
independently tokenized continuation span.

When no continuation is supplied, the current chat-template text and token path
remain unchanged and golden fixtures must remain byte- and token-identical.

**Alternative considered: add prior rows as a completed assistant chat
message.** Rejected because it inserts a terminal boundary and tests
conversation history rather than continuation within the current response.

**Alternative considered: accept pretokenized continuation identifiers.**
Rejected because it hides boundary retokenization and permits token/text drift.

### 5. Keep experimental orchestration outside stable inference ownership

The future research runner owns request identity construction, seed derivation,
attempt ordering, batching into groups of four, accepted-row selection, image
variants, per-attempt artifacts, and paired scientific comparisons. Stable
inference only guarantees that a valid request is executed with attested prompt
and generation semantics.

Residue tests will reject stable configuration or inference fields containing
experiment arm names, cohort identifiers, grid policies, calibration
thresholds, matching settings, non-maximum-suppression settings, or research
run-root names.

## Risks / Trade-offs

- **[The custom sampler drifts from installed HF score, cache, stop, or padding
  semantics]** → Stop at the executable attestation gate; do not use global
  reseeding or batch-size-one fallback.
- **[Model generation defaults silently add top-k or another warper]** → Build
  and receipt the complete sanitized profile; fail when an unowned effective
  processor or constraint remains active.
- **[Batch order or cardinality preserves tokens but changes scientific
  scores]** → Compare selected-token score traces and recomputed row-local scores
  after float32 promotion with absolute and relative tolerance `1e-6`; stop if
  the tolerance or deterministic score-ordering gate fails.
- **[Full-prompt retokenization changes a boundary token]** → Record the token
  impact span from the longest common prefix and compare full prompt identifiers
  rather than pretending continuation tokenization is additive.
- **[Execution receipts duplicate some fields already present in
  `DecodeResult`]** → Keep duplication limited to binding-critical fingerprints
  and validate equality; the redundancy is deliberate attestation evidence.
- **[A caller constructs sampled requests with incompatible policies in one
  batch]** → Fail before model generation with request identities and differing
  fields in the error context.
- **[A research runner treats receipt presence as scientific validity]** → The
  receipt attests execution only; research admission, comparison, and inference
  remain outside this capability and must carry their own ledger.
- **[Prompt validation rejects a legitimate future continuation syntax]** →
  Keep the seam generic but conservative; expand accepted syntax through a
  later compatibility change with golden-token evidence.

## Migration Plan

1. Add validation-only policy, continuation, and receipt types with unit tests.
2. Record the installed stock Qwen/Hugging Face path's lack of request-owned
   generator support and select the authorized custom-generation seam.
3. Implement request-owned generators, the minimal score-preserving custom
   sampler, and result-bound receipts; adapt the public pipeline to construct
   the explicit greedy policy.
4. Add assistant continuation and golden prompt fixtures.
5. Re-run current greedy backend, prompt, pipeline, trace, and shard tests to
   prove no-continuation and greedy compatibility, including a greedy config
   whose authored sampling-only values normalize to neutral executed values.
6. Execute the four-request CUDA attestation through the implemented branch.
   Only after this verification gate may research orchestration consume the
   seams.

Rollback deletes the new optional continuation path and sampled request path,
then restores direct greedy request fields. Existing production configuration
and artifact schemas require no data migration.

## Open Questions

- Which stable strings should name the custom sampler, attested generator
  implementation, and logical/physical device evidence in receipts? The
  implementation task must select deterministic machine-readable values and
  golden-test them before artifact consumption.
