## ADDED Requirements

### Requirement: HF exact-history chosen-token evidence

`HFBackendSession` SHALL expose an HF-specific, single-request exact-history
surface without adding exact-history operations to the backend-neutral
`BackendSession` protocol.

The surface MUST provide:

- `special_token_ids`, including the backend-owned `im_end` and `pad` token
  IDs;
- `prepare_exact_history(request: DecodeRequest) -> HFExactHistory`;
- `extend_exact_history(history, token_ids) -> HFExactHistory`; and
- `teacher_forced_evidence(history, continuation_token_ids) ->
  tuple[HFChosenTokenEvidence, ...]`.

`HFExactHistory` MUST expose the request ID, exact conditioning token IDs, and
their canonical SHA-256. Native tensors, loaded model/tokenizer/processor
objects, and the unforgeable session binding MUST remain private to the live HF
session. A history MUST be immutable and usable only by the live session that
created it.

Preparation MUST reuse the existing request media, processor, prompt-token,
and image-grid validation. Extension MUST append caller-supplied integer token
IDs in order without decoding or re-tokenizing them and MUST return a new
history without mutating its parent. Extension MUST NOT claim that attention or
position tensors have already been updated.

Teacher-forced evidence MUST materialize attention and Qwen M-RoPE positions
from the full conditioning-plus-continuation sequence at the time of use. It
MUST return exactly one evidence record per non-empty continuation token, in
the same order, with the token ID, `raw_model_logprob` under the existing FP32
raw-likelihood semantics, and `candidate_vocab_rank`. Rank MUST equal one plus
the count of raw FP32 logits strictly greater than the selected token logit, so
tied logits share a rank. The operation MUST NOT return full logits, mutate the
history, apply generation processors, or choose scientific interpretation.

Invalid token IDs, an empty evidence continuation, a forged or cross-session
history, or use after session close MUST raise the existing
`RuntimeContractError` family before a model forward.

#### Scenario: Prepare one verified multimodal history

- **WHEN** a live HF session prepares one valid `DecodeRequest`
- **THEN** the returned history's conditioning IDs equal the processor-verified
  executed prompt IDs
- **AND** its SHA-256 equals the canonical hash of those exact IDs
- **AND** no native HF object is available through its public attributes

#### Scenario: Append a persisted exact prefix

- **WHEN** the caller extends a history with persisted prefix token IDs
- **THEN** the child history contains the parent IDs followed by those exact IDs
  in order
- **AND** the parent history remains unchanged
- **AND** no text decode or tokenization path is invoked

#### Scenario: Observe alternative boundary tokens

- **WHEN** the caller separately requests one-token evidence for its chosen row
  opener and for `special_token_ids["im_end"]` on the same history
- **THEN** both records are conditioned on the identical history
- **AND** the caller can compute its experiment-owned opener-minus-terminal
  margin without receiving full logits or a backend-selected label

#### Scenario: Score a complete candidate sequence

- **WHEN** the caller requests evidence for a non-empty multi-token candidate
  row
- **THEN** one ordered evidence record is returned for every supplied token
- **AND** every log-probability and vocabulary rank is conditioned on the exact
  preceding history and earlier supplied candidate tokens

#### Scenario: Reuse a history in the wrong lifecycle

- **WHEN** a history is passed to another HF session or used after its owning
  session closes
- **THEN** the operation raises `RuntimeContractError` before model execution

#### Scenario: Research semantics remain caller-owned

- **WHEN** two callers use the same exact-history evidence surface for a
  boundary comparison and a candidate-row score
- **THEN** the backend returns only mechanical token evidence
- **AND** cohort, candidate role, row phase, control, owner policy, aggregation,
  estimand, uncertainty, claim boundary, and stop rule remain outside the
  backend result

### Requirement: HF observed runtime receipt settings

The HF backend session receipt SHALL include
`effective_settings.observed_model_dtype` and
`effective_settings.observed_attn_implementation` as observations of the
loaded runtime, distinct from declared launch dtype and configured backend
options.

`observed_model_dtype` MUST be either a backend-neutral mapping containing
sorted parameter dtype names and element counts or `null` when the runtime
cannot establish them. `observed_attn_implementation` MUST be the value
observed on the loaded model or `null` when unavailable. The backend MUST NOT
substitute a configured or declared value and label it as observed.

#### Scenario: Qwen runtime observations are available

- **WHEN** an HF Qwen model exposes its parameter dtypes and active attention
  implementation at session open
- **THEN** the receipt records their observed values
- **AND** a caller need not access the loaded model merely to report them

#### Scenario: Runtime observation is unavailable

- **WHEN** the loaded model does not expose an active attention implementation
  or observable parameter dtype information
- **THEN** the corresponding observed field is `null`
- **AND** the configured launch value remains separately identifiable and is
  not relabeled as an observation
