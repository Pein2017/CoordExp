## 1. Contract-First Interface Tests

- [ ] 1.1 Add failing backend-interface tests for tagged greedy and sampled
  generation-policy validation, request-owned seed requirements, four-request
  batch compatibility, and rejection of process- or batch-owned seed semantics.
- [ ] 1.2 Add failing execution-receipt tests for canonical fingerprints,
  serialization replay, prompt/output hash binding, and intentional
  result-receipt and generator-order swaps. Include model/tokenizer/custom-
  sampler swaps, score-trace binding, and nested-mutation resistance.
- [x] 1.3 Add failing golden prompt tests for full-text open-assistant
  continuation tokenization, boundary-retokenization impact spans, forbidden
  control tokens only inside the final open-assistant interval, legal earlier
  completed-turn terminators, exact single-image-placeholder evidence, backend
  parity, frozen prompt-field meanings, and unchanged no-continuation prompts.
- [ ] 1.4 Add a residue test proving stable inference configuration and source do
  not acquire spatial-scope arm names, cohort identifiers, grid policies,
  calibration thresholds, matching or non-maximum-suppression settings, or
  research output-root names.

## 2. Installed-Runtime Sampling Attestation Gate

- [ ] 2.1 Add a bounded one-time probe that records the installed stock Qwen and
  Hugging Face sampling path, demonstrates that it has no per-request generator
  seam, and records the authorized `custom_generate` implementation seam without
  making a model call.
- [ ] 2.2 Add contract fixtures for the custom sampler's four-row generator
  mapping, sanitized temperature-plus-nucleus generation profile, score tensor
  timing, sample-then-pad behavior, cache update, Qwen stop, and returned output
  shape. These fixtures are wiring evidence, not CUDA attestation.
- [ ] 2.3 Keep sampled support unavailable until the implemented branch passes
  the later four-request CUDA attestation; never use batch-size-one fallback,
  serial disguise, shared randomness, or process-global reseeding.

## 3. Request-Scoped Backend Implementation

- [ ] 3.1 Implement immutable decode-generation-policy validation and replace
  direct `DecodeRequest` generation fields with the policy plus optional
  request-owned sampling seed; update all callers without compatibility
  pass-through fields.
- [ ] 3.2 Implement batch compatibility checks and request-owned sampled
  generators through the narrow score-preserving custom Hugging Face branch,
  including explicit neutralization of inherited model sampling defaults, while
  preserving scored token alignment, cache updates, sample-then-pad behavior,
  Qwen `<|im_end|>` stopping, and greedy defaults.
- [ ] 3.3 Implement immutable result-bound decode execution receipts, canonical
  token hashes and fingerprints, actual-generator initial-seed and index
  evidence, batch request-order fingerprints, executed-argument normalization,
  model/tokenizer/runtime/sampling-profile/custom-sampler identity binding,
  canonical float32 score-trace binding, recursively immutable nested values,
  and strict result-receipt validation.
- [ ] 3.4 Update the public inference pipeline to construct the explicit greedy
  policy from current resolved configuration, preserve authored sampling-only
  provenance while normalizing ignored greedy execution knobs, and prove that
  `generation.batch_size: 4` remains a per-device execution quantity rather
  than a seed or scientific-arm control.

## 4. Open-Assistant Continuation Implementation

- [x] 4.1 Implement immutable assistant-continuation input, forbidden-boundary
  validation, append-without-separator semantics, and complete-prompt
  retokenization in `src/inference/prompt.py`.
- [x] 4.2 Extend prompt records with full-prompt fingerprint, continuation text
  hash, frozen `prompt_text` and `chat_text` meanings, serialized
  `full_chat_text`, final assistant-content start, byte and character spans,
  longest-common-prefix token impact span, and exact image-placeholder and
  open-assistant interval evidence.
- [x] 4.3 Preserve the existing no-continuation chat-template path exactly and
  update backend prompt-parity validation to use complete combined prompt token
  identifiers when a continuation is present.

## 5. Verification and Independent Gate

- [ ] 5.1 Run targeted backend, prompt, pipeline, scored-trace, and data-parallel
  shard tests in the repository-standard conda environment named `ms`, including
  batch-size-four sampled and greedy cases.
- [ ] 5.2 Re-run the installed-runtime four-request attestation through the
  implemented backend and verify each returned receipt binds the exact request
  seed to the actual generator initial seed and execution index, plus the batch
  order, sanitized executed arguments, prompt, generated output, and stop
  reason. Verify exact same-seed replay after reversed request ordering by
  request identity while allowing order-dependent receipt fingerprints. Repeat
  the replay and generator-binding gate for the natural three-request tail
  cardinality required by the frozen optional replication schedule. Compare the
  shared three requests across natural batch size three and an attestation batch
  size four with one independent fourth request. On the exact configured
  attention implementation, attest processed-logit, cache, score-step,
  transition-score, `<|im_end|>`, stop-reason, sample-then-pad, output-shape, and
  generator-draw semantics; bind selected-token score traces and recomputed
  row-local scores with the frozen float32 tolerance.
- [ ] 5.3 Run strict OpenSpec validation, whitespace checks, the stable-surface
  residue check, and a no-continuation/no-sampling regression comparison against
  the pre-change greedy fixtures.
- [ ] 5.4 Obtain separate independent engineering-standards and
  intent-and-contract audit verdicts; do not expose these seams to the research
  runner while any priority-zero or priority-one finding remains unresolved.
