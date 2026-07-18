## 1. Wave 0 - Contract And Qualification

- [x] 1.1 Inspect current config, runtime, backend, prompt/image, artifact, merge, worker, evaluator, and stable-spec ownership.
- [x] 1.2 Inspect installed vLLM, Transformers, PEFT, and Qwen3-VL source for multimodal, likelihood, DoRA, and process semantics.
- [x] 1.3 Record the vLLM and execution-model source studies with accepted lessons and rejected shortcuts.
- [x] 1.4 Add an executable real base-only qualification probe for prompt ids, no-resize image input, special tokens, stop behavior, processed logprobs, raw replay, CUDA binding, explicit engine process mode, and cleanup.
- [x] 1.5 Run the qualification probe on vLLM 0.14.1 and commit a receipt binding the probe, fixture, exhaustive model/tokenizer/processor/template assets, every loaded vLLM/Transformers/PEFT/Qwen utility source module after execution plus required execution owners, dependency, CUDA, engine-process, executed image bytes, returned placeholder ranges, prompt, stop, likelihood, replay, and cleanup identities; otherwise revise the candidate version after a concrete failure.
- [x] 1.6 Gate Wave 0 with OpenSpec validation plus independent standards and intent-contract reviews; resolve every P0/P1.

## 2. Wave 1 - Shared Contracts And HF Regression

- [x] 2.1 Add failing config tests for strict HF/vLLM block discrimination, migration of HF-only attention/patch controls, deterministic-only inference, mandatory evidence, raw-trace opt-in, and qualified-version failure.
- [x] 2.2 Implement the strict config projection, migrate canonical configs to `backend.hf`, and remove the reserved-vLLM rejection.
- [x] 2.3 Add semantic decode request with separate unexpanded input and expected executed prompt ids plus media SHA-256, likelihood pair, injectable backend-session opener, launch, and receipt contracts without native backend objects.
- [x] 2.4 Move the existing HF implementation behind an HF session with backend-private image tensor materialization and actual runtime receipts.
- [x] 2.5 Add HF raw-logit tracing and prove it against a teacher-forced FP32 reference.
- [x] 2.6 Refactor pipeline orchestration to semantic requests and session-owned execution while retaining the old HF path only as a temporary parity oracle.
- [x] 2.7 Run fresh real HF single-row and heterogeneous-batch artifact parity between the old and new paths, requiring byte-exact evaluator-bearing rows, exact projected sidecars, zero removed structured leaves, and zero additions or changed structured leaves outside the executable verifier's explicitly enumerated backend-session, likelihood, transform/media, identity-derivation, config-schema, and device-provenance policy.
- [x] 2.8 Delete `InferenceRuntime`, old backend factories, `model_inputs`, and pipeline `_batches` residue, then gate Wave 1 with targeted HF/config/pipeline/evaluator tests, OpenSpec validation, residue checks, and two independent reviews.

## 3. Wave 2 - Execution Model

- [x] 3.1 Add failing tests for base-only and composed fingerprint determinants, worker revalidation, cache hits, payload changes, invalid counts, tied weights, corrupt entries, locks, and failed atomic publication.
- [x] 3.2 Deepen adapter and selected-token owners with reusable identity inspection and deterministic merge/fold operations.
- [x] 3.3 Implement content-addressed execution-model resolution under `model_cache/coordexp_swift/vllm_materialized`.
- [x] 3.4 Implement controller-side locking, staging, standard HF save, manifest/hash validation, and worker receipt loading.
- [ ] 3.5 Add dynamic-HF versus materialized-HF parity tooling with the specified target-dtype row equality and FP32 logit tolerances, then run it on one real base-plus-DoRA-plus-delta checkpoint.
- [ ] 3.6 Gate Wave 2 with materialization tests, real parity receipt, residue checks, OpenSpec validation, and two independent reviews.

## 4. Wave 3 - Single-GPU vLLM Tracer Bullet

- [ ] 4.1 Add failing vLLM adapter tests for engine kwargs, unexpanded-to-executed prompt projection, image path mutation, result ids/order, stop retention, and processed chosen-token likelihood extraction.
- [ ] 4.2 Implement the offline vLLM session with qualified version enforcement, TP=1, DP=1, per-device concurrency, no-resize media, and explicit process mode.
- [ ] 4.3 Normalize vLLM outputs into backend-neutral results and actual engine/model/processor receipts.
- [ ] 4.4 Run one real base-only row through prompt, vLLM, parsing, policy scoring, artifacts, and the unchanged evaluator.
- [ ] 4.5 Run a real concurrency probe at the largest initially supported `max_num_seqs`, and run the fixed fixture twice in fresh workers with exact prompt ids, generated ids, stop reason, parser/scored rows, and artifact ordering plus the approved likelihood tolerances.
- [ ] 4.6 Gate Wave 3 with exact fixed-fixture parity, the concurrency and repeatability receipts, process-exit cleanup checks, OpenSpec validation, and two independent reviews.

## 5. Wave 4 - Dual Likelihood Evidence

- [ ] 5.1 Add failing alignment tests for missing, duplicate, shifted, positive, non-finite, token-mismatched, and stop-missing raw likelihood evidence.
- [ ] 5.2 Implement vLLM teacher-forced prompt-logprob replay and discard replay-only generated tokens.
- [ ] 5.3 Add nullable `raw_model_logprob` to token trace artifacts and likelihood semantics to provenance, manifest, summaries, and strict merge identity.
- [ ] 5.4 Preserve policy-only selected replay and evaluator score ownership; add explicit regression tests against accidental raw-score promotion.
- [ ] 5.5 Run the repeated-token HF/vLLM likelihood probe and enforce median 0.002, P99 0.02, max 0.05, and per-object log-score 0.01 tolerances.
- [ ] 5.6 Gate Wave 4 with trace/artifact/merge tests, executed numeric receipt, OpenSpec validation, and two independent reviews.

## 6. Wave 5 - Checkpoint And Distributed Integration

- [ ] 6.1 Pass execution-model receipts through controller/worker CLI and validate one shared fingerprint before every vLLM engine starts.
- [ ] 6.2 Preserve deterministic batch-block sharding while mapping batch size to vLLM maximum sequence concurrency.
- [ ] 6.3 Extend worker/merge failure handling for engine startup, CUDA OOM, bounded worker timeout, full owned-process-tree termination, orphan process, malformed shard, backend identity, and likelihood mismatches.
- [ ] 6.4 Add production-like vLLM configs for base smoke, adapter-plus-delta smoke, and matched step-4887 val200 evaluation.
- [ ] 6.5 Run real one-GPU composed-checkpoint, two-active-rank, and eight-active-rank smokes through evaluator consumption; the eight-rank fixture must contain at least eight nonempty decode blocks.
- [ ] 6.6 Gate Wave 5 with exact row coverage/order and one table-driven injected test/receipt for every 6.3 failure class (engine startup, CUDA OOM, worker timeout, process-tree termination, orphan process, malformed shard, backend identity, and likelihood mismatch), each proving terminal diagnostics, complete cleanup, and absence of canonical top-level artifacts; then run targeted tests, OpenSpec validation, and two independent reviews.

## 7. Wave 6 - Acceptance And Documentation

- [ ] 7.1 Run matched HF/vLLM val200 inference and evaluation with identical checkpoint, dataset, prompt, and generation policy.
- [ ] 7.2 Verify exact input/GT identity and absolute mAP/mRecall differences at most 0.005; report parse/drop/stop/truncation counters, throughput, and peak memory.
- [ ] 7.3 Update canonical architecture, implementation-map, upstream, and infer/eval workflow docs with accepted behavior and operational commands.
- [ ] 7.4 Run the complete targeted inference/evaluator suite, real smoke receipts, OpenSpec delta inspection, residue searches, markdown/config hygiene, and `git diff --check`.
- [ ] 7.5 Run four independent final review lanes, record triage, and resolve every accepted P0/P1 finding.
- [ ] 7.6 Sync accepted delta specs and archive the change only after all implementation and acceptance tasks pass.
