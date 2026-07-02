## 1. Source Studies And Approval Gates

- [x] 1.1 Obtain user approval for source studies and probes only; source implementation remains blocked until the study note is written and reviewed.
- [x] 1.2 Study owner-boundary paths in current CoordExp-swift Qwen/template/config/artifact/adapter code for prompt rendering, tokenizer ids, no-resize image planning, image batch materialization, resolved config writing, manifest writing, adapter reload, and special-token embedding delta validation.
- [x] 1.3 Define owner-neutral runtime seams for Qwen loading, resolved config writing, artifact manifests, adapter identity, and embedding-delta identity so inference-facing modules do not depend on `TrainConfig`, `ResolvedTrainConfig`, `ResolvedStepSchedule`, `load_train_config()`, or unallowlisted `src.training.*`; the V1 training-import allowlist is empty unless explicitly patched into OpenSpec.
- [x] 1.4 Study installed Transformers Qwen3-VL `generate` output, `scores`, `compute_transition_scores`, prompt-padding behavior, stop-token handling, tokenizer special-token decoding, processor vision fields, and model vision config fields.
- [x] 1.5 Study existing `src/adapters/` owner code and PEFT adapter load/status behavior, then define fatal checks for adapter base identity, captured PEFT `load_result.missing_keys`, captured PEFT `load_result.unexpected_keys`, `set_adapter`, `get_model_status().enabled`, expected active adapter names, irregular fields, and unexpected merged state; warning-only PEFT load paths are not sufficient.
- [x] 1.6 Study special-token embedding delta identity checks for base config SHA, tokenizer SHA, token strings, and token ids.
- [x] 1.7 Study old `/data/CoordExp/src/infer/*`, current eval mAP artifact contract, confidence post-op, and legacy inference configs as reference-only material.
- [x] 1.8 Triage legacy tests importing `src.infer` or legacy inference configs; classify each as port-to-`tests/inference/`, retained with an explicit allowlist reason, or removed/archived with rationale while preserving the no-`src/infer/` package contract.
- [x] 1.9 Write a source-study note under this change or proposal directory summarizing exact handles, accepted lessons, rejected legacy behavior, implementation constraints, adapter ownership, legacy test triage, and any OpenSpec contradictions.
- [x] 1.10 Patch OpenSpec and the superpower roadmap if source studies contradict current contract wording.
  - Source-study result: no immediate OpenSpec rewrite was required; stale legacy docs are recorded as reference-only implementation risk in `source-studies/2026-07-02-wave1-gate-resolution.md`.
- [x] 1.11 Obtain explicit user approval for source implementation after source-study review, covering config/runtime, backend trace, prompt/image parity, parser/geometry, scoring/artifacts, eval-consumer, and benchmark acceptance contract surfaces.
  - 2026-07-02 completed: user approved implementation kickoff in this worktree after OpenSpec and review-convergence planning; final production benchmark launch remains separately approval-gated by 8.6.

## 2. Smoke Fixtures And Production Handles

- [x] 2.1 Pin a tiny real single-image inference fixture that uses the same offline JSONL example family as training and preserves image path, dimensions, GT objects, and prompt-relevant fields.
- [x] 2.2 Pin a two-row batched trace fixture with different prompt lengths for HF score-alignment testing.
- [x] 2.3 Pin or create an adapter-enabled smoke fixture using a `checkpoint-final` alias or explicit concrete checkpoint path plus any special-token embedding delta expected by the target checkpoint.
- [x] 2.4 Name the production benchmark config leaf, dataset path, base model path, adapter checkpoint path when used, artifact root, and mAP evaluator command before launching final benchmark inference.
- [x] 2.5 Record that sample-limited and tiny runs are smoke evidence only and not final benchmark evidence.

## 3. Config Runtime And Entry Surface

- [x] 3.1 Add failing tests for strict `InferConfig`: valid production leaf, rejected training-only keys, rejected unknown keys, rejected legacy `configs/infer/*`, reserved but not implemented vLLM backend, and debug-only `batch_size: 1`.
- [x] 3.2 Implement `InferConfig` models and loader behavior without importing `TrainConfig`.
- [x] 3.3 Add `src/infer.py` as the thin public entrypoint and `src/inference/` package skeleton; verify no `src/infer/` sibling package exists.
- [x] 3.4 Implement resolved config writing for inference using `configs/resolved.json` and `configs/resolved.yaml`.
- [x] 3.5 Implement runtime assembly for base model, optional adapter, optional embedding delta, and resolved identity recording through existing Qwen/adapter/artifact owners; inference runtime coordinates adapter-owner results rather than inlining PEFT status policy.
- [x] 3.6 Add owner-boundary AST residue tests preventing inference-facing imports or calls involving `TrainConfig`, `ResolvedTrainConfig`, `ResolvedStepSchedule`, `load_train_config()`, and any unallowlisted `src.training.*` module across `src/infer.py`, `src/inference/**`, and inference-facing owner APIs.
- [x] 3.7 Add or extend adapter-owner tests for inference reload/status checks under `tests/adapters/`, then connect them to inference config/runtime tests.
- [x] 3.8 Run config/runtime tests, adapter-owner tests, and a dry path that writes resolved configs before model generation.

## 4. Backend Trace

- [x] 4.1 Add failing tests for backend-neutral `DecodeRequest`, `DecodeResult`, and `TokenTrace` consumption without HF object leakage.
- [x] 4.2 Add failing tests for HF scored generation arguments, missing score failure, missing required trace field failure, and special-token-preserving raw trace decode.
- [x] 4.3 Add failing tests for batched prompt-width alignment with variable prompt lengths and post-stop padding exclusion.
- [x] 4.4 Implement HF `generate_batch` with `return_dict_in_generate=True`, `output_scores=True`, deterministic greedy defaults, Qwen `<|im_end|>` stop policy, and normalized transition logprob gathering.
- [x] 4.5 Run a real tiny HF/Qwen trace probe that verifies generated ids, token text, logprobs, stop token, and parser-facing stripped view.
  - 2026-07-02 Wave 3 implementation note: targeted fake-HF tests now cover one-row scored trace extraction, special-token preservation, post-stop pad exclusion, and two-row variable prompt-width alignment. The real Qwen3-VL image probe remains deferred to Wave 7 smoke because loading the full local 2B VL model is too expensive for this small backend-only wave; this does not weaken the backend-trace spec.
  - 2026-07-02 Wave 7 partial evidence: `outputs/coordexp_swift/infer/wave7_real_smokes/wave7-real-base-single-smoke-20260702T182753Z/pred_token_trace.jsonl` records generated ids, token text, and finite logprobs over a real Qwen run. The tiny run reached length stop rather than `<|im_end|>`, so real stop-token evidence remains open.
  - 2026-07-02 Wave 7 completion evidence: `outputs/coordexp_swift/infer/wave7_real_smokes/wave7-real-production-adapter-smoke/pred_token_trace.jsonl` records generated ids, token text, selected logprob replay, `is_stop=2`, and `is_pad=2`; `gt_vs_pred.jsonl` and `parse_diagnostics.jsonl` prove parser-facing stripped compact rows were accepted.

## 5. Prompt Image And Parsing

- [x] 5.1 Add failing tests for prompt-token parity between local rendered prompts and backend prompt token ids.
- [x] 5.2 Add failing tests for mandatory no-resize `image_plan.jsonl` fields, processor-model vision parity, invalid-dimension terminal failure before processor reshape, and no benchmark-eligible row artifacts on invalid required input.
- [x] 5.3 Implement inference prompt helpers that reuse training template semantics and record prompt/template fingerprints.
- [x] 5.4 Implement image-plan materialization for inference batches through existing Qwen no-resize helpers.
- [x] 5.5 Add failing parser tests for valid compact object, unsupported JSON assistant response, accepted-with-drops, all-spans-dropped, generated prediction order preservation, degenerate bbox, and out-of-range coordinates.
- [x] 5.6 Implement compact object-box-closed parser, salvage diagnostics, inline parser status, sidecar diagnostics, and shared geometry conversion.
- [x] 5.7 Run prompt/image/parser tests and a tiny real no-resize image smoke.

## 6. Scoring And Artifact Contracts

- [x] 6.1 Add failing tests for `exp(sum(selected_token_logprobs) / n_selected)`, exact `n_selected == 8` compact-object policy, empty selected-token set, non-finite logprob, duplicate-span ambiguity, object-span contiguity, and description/category token exclusion.
- [x] 6.2 Implement selected-token alignment and scoring with persisted replay evidence: row id, object span id, generated-step indices, token ids, token text, selected logprobs, selected count, and score-policy fingerprinting.
- [x] 6.3 Add failing tests for raw/scored row-count parity, no extra diagnostic rows in raw/scored JSONL, scored rows with `pred: []`, inline GT preservation, row-local non-empty score source, integer score version, finite score range `[0.0, 1.0]`, scored provenance sidecar, portable raw/scored SHA binding, and trace-based score recomputation.
- [x] 6.4 Implement `gt_vs_pred.jsonl`, `gt_vs_pred_scored.jsonl`, `gt_vs_pred_scored.jsonl.provenance.json`, `pred_token_trace.jsonl`, `parse_diagnostics.jsonl`, `image_plan.jsonl`, `summary.json`, and inference manifest writing.
- [x] 6.5 Run artifact/scoring tests and inspect a tiny scored artifact set for benchmark eligibility fields.

## 7. Pipeline And Eval Consumer

- [x] 7.1 Add failing tests for end-to-end pipeline orchestration over a tiny real or fixture-backed input, including batching, counters, terminal failure status, and no metric reduction inside `src.infer`.
- [x] 7.2 Implement inference pipeline orchestration across runtime, prompt, backend, parsing, scoring, and artifact modules.
- [x] 7.3 Add failing tests for the minimal rebuilt `src.eval` detection consumer that consumes `gt_vs_pred_scored.jsonl` with provenance and writes `metrics.json` or an explicitly named metric artifact.
- [x] 7.4 Implement the minimal rebuilt evaluator consumer and verify it refuses missing or mismatched score provenance before metric computation; do not implement a legacy bridge unless the user explicitly changes the V1 default.
- [x] 7.5 Run pipeline/evaluator tests and a tiny scored fixture through the named evaluator consumer.

## 8. Real Smokes And Benchmark Readiness

- [x] 8.1 Run the tiny real HF/Qwen inference smoke and verify prompt parity, image plan, trace, parser, scoring, and artifacts.
  - 2026-07-02 partial: base single-row smoke completed and wrote required artifacts, but generated output produced `scoreable_prediction_count=0`, so non-empty selected-token scoring remains uncovered by real model output.
  - 2026-07-02 completion evidence: production-adapter smoke `outputs/coordexp_swift/infer/wave7_real_smokes/wave7-real-production-adapter-smoke` completed over the two-row real fixture with `scoreable_prediction_count=4`, `parser_failure_count=0`, required artifacts present, and evaluator consumer metrics at `eval/metrics.json` reporting `pred_object_count=4` and `scored_pred_count=4`.
- [x] 8.2 Run the two-row batched trace smoke and verify prompt-width alignment and stop/pad handling.
  - 2026-07-02 partial: two-row batch completed and wrote required artifacts, but no real `is_stop=true` or `is_pad=true` trace rows were observed; stop/pad remains targeted-test evidence only.
  - 2026-07-02 completion evidence: production-adapter smoke `pred_token_trace.jsonl` contains 48 trace rows with `is_stop=2` and `is_pad=2` under `generation.batch_size=2`.
- [x] 8.3 Run the adapter-enabled real smoke and verify checkpoint-final or explicit checkpoint resolution, PEFT identity/status checks, embedding-delta metadata checks, and scored artifact output.
  - 2026-07-02 completed: adapter smoke passed at `outputs/coordexp_swift/infer/wave7_real_smokes/wave7-real-adapter-smoke-20260702T192643Z` using explicit adapter path and smoke-only repaired delta payload `outputs/coordexp_swift/infer/wave7_real_smokes/repaired_special_token_embeddings_step2`; `run_manifest.json` records `adapter_identity.status=validated`, `adapter_identity.requires_grad={"default": false}`, `model_identity.embedding_delta.status=loaded`, and `model_identity.embedding_delta.load.loaded=true`; `summary.json` records `scored_artifact_materialized=true` and `benchmark_eligible=false`.
- [x] 8.4 Run OpenSpec validation, targeted pytest suites, markdown/config hygiene checks, residue checks for `src/infer/` package collision, unapproved `src.infer` legacy test imports, stale `resolved_config.json`, legacy `configs/infer/` authority, constant-score fallback, and old `object_ordering: sorted` assumptions.
- [x] 8.5 Prepare a benchmark launch packet naming production config, dataset, model, adapter, artifact root, evaluator command, expected evidence scope, and rollback path.
- [ ] 8.6 Stop for explicit user approval before launching implementation-derived production benchmark or claiming final inference correctness.

## 9. Review Convergence

- [ ] 9.1 Dispatch isolated review lanes after implementation-plan drafting: contract/spec auditor, upstream HF/Qwen/PEFT tracer, eval/artifact auditor, architecture/module-boundary auditor, and smoke/benchmark auditor.
- [ ] 9.2 Triage findings as P0, P1, P2, wrong, duplicate, or non-blocking.
- [ ] 9.3 Patch all accepted P0/P1 findings in OpenSpec and superpower docs before requesting user approval.
- [ ] 9.4 Leave implementation unchecked and blocked until the user explicitly approves kickoff.
