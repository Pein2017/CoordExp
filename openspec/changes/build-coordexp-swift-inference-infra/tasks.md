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
- [ ] 1.11 Obtain explicit user approval for source implementation after source-study review, covering config/runtime, backend trace, prompt/image parity, parser/geometry, scoring/artifacts, eval-consumer, and benchmark acceptance contract surfaces.

## 2. Smoke Fixtures And Production Handles

- [ ] 2.1 Pin a tiny real single-image inference fixture that uses the same offline JSONL example family as training and preserves image path, dimensions, GT objects, and prompt-relevant fields.
- [ ] 2.2 Pin a two-row batched trace fixture with different prompt lengths for HF score-alignment testing.
- [ ] 2.3 Pin or create an adapter-enabled smoke fixture using a `checkpoint-final` alias or explicit concrete checkpoint path plus any special-token embedding delta expected by the target checkpoint.
- [ ] 2.4 Name the production benchmark config leaf, dataset path, base model path, adapter checkpoint path when used, artifact root, and mAP evaluator command before launching final benchmark inference.
- [ ] 2.5 Record that sample-limited and tiny runs are smoke evidence only and not final benchmark evidence.

## 3. Config Runtime And Entry Surface

- [ ] 3.1 Add failing tests for strict `InferConfig`: valid production leaf, rejected training-only keys, rejected unknown keys, rejected legacy `configs/infer/*`, reserved but not implemented vLLM backend, and debug-only `batch_size: 1`.
- [ ] 3.2 Implement `InferConfig` models and loader behavior without importing `TrainConfig`.
- [ ] 3.3 Add `src/infer.py` as the thin public entrypoint and `src/inference/` package skeleton; verify no `src/infer/` sibling package exists.
- [ ] 3.4 Implement resolved config writing for inference using `configs/resolved.json` and `configs/resolved.yaml`.
- [ ] 3.5 Implement runtime assembly for base model, optional adapter, optional embedding delta, and resolved identity recording through existing Qwen/adapter/artifact owners; inference runtime coordinates adapter-owner results rather than inlining PEFT status policy.
- [ ] 3.6 Add owner-boundary AST residue tests preventing inference-facing imports or calls involving `TrainConfig`, `ResolvedTrainConfig`, `ResolvedStepSchedule`, `load_train_config()`, and any unallowlisted `src.training.*` module across `src/infer.py`, `src/inference/**`, and inference-facing owner APIs.
- [ ] 3.7 Add or extend adapter-owner tests for inference reload/status checks under `tests/adapters/`, then connect them to inference config/runtime tests.
- [ ] 3.8 Run config/runtime tests, adapter-owner tests, and a dry path that writes resolved configs before model generation.

## 4. Backend Trace

- [ ] 4.1 Add failing tests for backend-neutral `DecodeRequest`, `DecodeResult`, and `TokenTrace` consumption without HF object leakage.
- [ ] 4.2 Add failing tests for HF scored generation arguments, missing score failure, missing required trace field failure, and special-token-preserving raw trace decode.
- [ ] 4.3 Add failing tests for batched prompt-width alignment with variable prompt lengths and post-stop padding exclusion.
- [ ] 4.4 Implement HF `generate_batch` with `return_dict_in_generate=True`, `output_scores=True`, deterministic greedy defaults, Qwen `<|im_end|>` stop policy, and normalized transition logprob gathering.
- [ ] 4.5 Run a real tiny HF/Qwen trace probe that verifies generated ids, token text, logprobs, stop token, and parser-facing stripped view.

## 5. Prompt Image And Parsing

- [ ] 5.1 Add failing tests for prompt-token parity between local rendered prompts and backend prompt token ids.
- [ ] 5.2 Add failing tests for mandatory no-resize `image_plan.jsonl` fields, processor-model vision parity, invalid-dimension terminal failure before processor reshape, and no benchmark-eligible row artifacts on invalid required input.
- [ ] 5.3 Implement inference prompt helpers that reuse training template semantics and record prompt/template fingerprints.
- [ ] 5.4 Implement image-plan materialization for inference batches through existing Qwen no-resize helpers.
- [ ] 5.5 Add failing parser tests for valid compact object, unsupported JSON assistant response, accepted-with-drops, all-spans-dropped, generated prediction order preservation, degenerate bbox, and out-of-range coordinates.
- [ ] 5.6 Implement compact object-box-closed parser, salvage diagnostics, inline parser status, sidecar diagnostics, and shared geometry conversion.
- [ ] 5.7 Run prompt/image/parser tests and a tiny real no-resize image smoke.

## 6. Scoring And Artifact Contracts

- [ ] 6.1 Add failing tests for `exp(sum(selected_token_logprobs) / n_selected)`, exact `n_selected == 8` compact-object policy, empty selected-token set, non-finite logprob, duplicate-span ambiguity, object-span contiguity, and description/category token exclusion.
- [ ] 6.2 Implement selected-token alignment and scoring with persisted replay evidence: row id, object span id, generated-step indices, token ids, token text, selected logprobs, selected count, and score-policy fingerprinting.
- [ ] 6.3 Add failing tests for raw/scored row-count parity, no extra diagnostic rows in raw/scored JSONL, scored rows with `pred: []`, inline GT preservation, row-local non-empty score source, integer score version, finite score range `[0.0, 1.0]`, scored provenance sidecar, portable raw/scored SHA binding, and trace-based score recomputation.
- [ ] 6.4 Implement `gt_vs_pred.jsonl`, `gt_vs_pred_scored.jsonl`, `gt_vs_pred_scored.jsonl.provenance.json`, `pred_token_trace.jsonl`, `parse_diagnostics.jsonl`, `image_plan.jsonl`, `summary.json`, and inference manifest writing.
- [ ] 6.5 Run artifact/scoring tests and inspect a tiny scored artifact set for benchmark eligibility fields.

## 7. Pipeline And Eval Consumer

- [ ] 7.1 Add failing tests for end-to-end pipeline orchestration over a tiny real or fixture-backed input, including batching, counters, terminal failure status, and no metric reduction inside `src.infer`.
- [ ] 7.2 Implement inference pipeline orchestration across runtime, prompt, backend, parsing, scoring, and artifact modules.
- [ ] 7.3 Add failing tests for the minimal rebuilt `src.eval` detection consumer that consumes `gt_vs_pred_scored.jsonl` with provenance and writes `metrics.json` or an explicitly named metric artifact.
- [ ] 7.4 Implement the minimal rebuilt evaluator consumer and verify it refuses missing or mismatched score provenance before metric computation; do not implement a legacy bridge unless the user explicitly changes the V1 default.
- [ ] 7.5 Run pipeline/evaluator tests and a tiny scored fixture through the named evaluator consumer.

## 8. Real Smokes And Benchmark Readiness

- [ ] 8.1 Run the tiny real HF/Qwen inference smoke and verify prompt parity, image plan, trace, parser, scoring, and artifacts.
- [ ] 8.2 Run the two-row batched trace smoke and verify prompt-width alignment and stop/pad handling.
- [ ] 8.3 Run the adapter-enabled real smoke and verify checkpoint-final or explicit checkpoint resolution, PEFT identity/status checks, embedding-delta metadata checks, and scored artifact output.
- [ ] 8.4 Run OpenSpec validation, targeted pytest suites, markdown/config hygiene checks, residue checks for `src/infer/` package collision, unapproved `src.infer` legacy test imports, stale `resolved_config.json`, legacy `configs/infer/` authority, constant-score fallback, and old `object_ordering: sorted` assumptions.
- [ ] 8.5 Prepare a benchmark launch packet naming production config, dataset, model, adapter, artifact root, evaluator command, expected evidence scope, and rollback path.
- [ ] 8.6 Stop for explicit user approval before launching implementation-derived production benchmark or claiming final inference correctness.

## 9. Review Convergence

- [ ] 9.1 Dispatch isolated review lanes after implementation-plan drafting: contract/spec auditor, upstream HF/Qwen/PEFT tracer, eval/artifact auditor, architecture/module-boundary auditor, and smoke/benchmark auditor.
- [ ] 9.2 Triage findings as P0, P1, P2, wrong, duplicate, or non-blocking.
- [ ] 9.3 Patch all accepted P0/P1 findings in OpenSpec and superpower docs before requesting user approval.
- [ ] 9.4 Leave implementation unchecked and blocked until the user explicitly approves kickoff.
