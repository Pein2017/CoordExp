## 1. Config + Schema Surface

- [x] 1.1 Add `rollout_matching.eval_rollout_backend` to the typed config schema (`src/config/rollout_matching_schema.py`) and fail fast on invalid values (`null|hf|vllm`).
- [x] 1.2 Tighten schema-level length guardrails in `TrainingConfig.from_mapping` (`src/config/schema.py`) so **effective eval backend == vLLM** enforces the same vLLM length-coherence checks as training vLLM:
  - `rollout_matching.vllm.max_model_len >= global_max_length`,
  - `rollout_matching.max_new_tokens < rollout_matching.vllm.max_model_len`.
- [x] 1.3 Add a schema/runtime invariant for full-sync-only vLLM: whenever the effective backend is `vllm` (train or eval), `rollout_matching.vllm.enable_lora` MUST be `false` (fail fast with actionable guidance if violated).
- [x] 1.4 Add schema regression tests for:
  - `eval_rollout_backend: null` inherits,
  - eval-only vLLM triggers the length guardrails,
  - eval-only vLLM + `enable_lora: true` fails fast (strict parsing + invariant).

## 2. Evaluation Backend Override Wiring

- [x] 2.1 Implement a single authoritative resolver for rollout backend selection (e.g., `_effective_rollout_backend(context="train"|"eval")`) and apply it consistently so eval-only vLLM cannot silently fall back to HF.
- [x] 2.2 Wire the eval override through **all** evaluation call-sites in `src/trainers/stage2_rollout_aligned.py`, including:
  - traced/confidence scoring path, and
  - non-traced `_rollout_many(...)` path (avoid `_rollout_many` re-resolving backend via training defaults).
- [x] 2.3 Ensure the override is rank-symmetric (DDP safe) and does not leak into training-time rollouts.
- [x] 2.4 Add minimal logging for resolved eval backend + vLLM mode at evaluation start (reproducibility).

## 3. Full-Sync Enforcement (no adapter-only sync)

- [x] 3.1 Add a runtime guard: if the effective rollout backend is `vllm` and config requests adapter-only sync (e.g., `rollout_matching.vllm.enable_lora: true`), fail fast with actionable guidance.
- [x] 3.2 Ensure both colocate and server sync paths are full-sync-only:
  - remove/disable adapter upload branches,
  - ensure no code path calls `add_lora` / LoRA adapter sync when effective backend is vLLM.
- [x] 3.3 Add a unit test that asserts `enable_lora: true` fails fast when eval-only vLLM is enabled (covers the override case explicitly).

## 4. Minimal GPU Memory: Eval-Scoped Offload + vLLM Lifecycle

- [x] 4.1 Ensure rollout offload works for eval-only vLLM colocate even when training rollouts use HF (i.e., the offload decision must respect the *effective eval backend*).
- [x] 4.1.1 Keep an explicit compatibility guard: when the effective backend is vLLM colocate (train or eval) and `rollout_matching.offload.enabled: true`, fail fast if `deepspeed.enabled: true` (DeepSpeed/ZeRO not supported for offload in this trainer).
- [x] 4.2 Implement eval-scoped offload semantics:
  - `rollout_matching.offload.enabled: true` defaults `offload_model=true` and `offload_optimizer=true` when missing,
  - offload/restore occurs once per `evaluate()` call (not per batch).
- [x] 4.3 Implement standard colocate vLLM eval lifecycle (no new YAML knobs):
  - default to `rollout_matching.vllm.enable_sleep_mode=false` (stable standard mode),
  - avoid in-process vLLM shutdown/teardown during training.
- [x] 4.4 Gate sleep lifecycle to explicit opt-in:
  - do not force `EngineArgs(enable_sleep_mode=true)` by default,
  - run sleep/wake API preflight only when `rollout_matching.vllm.enable_sleep_mode=true`.
- [x] 4.5 Add a regression test that runs a tiny eval loop (mocked or minimal) and asserts:
  - offload/restore happens once per `evaluate()` call,
  - sleep/wake lifecycle assertions are only required when sleep mode is explicitly enabled.
- [x] 4.6 Add a regression test for the **startup preflight**: if eval-only vLLM colocate is configured with `enable_sleep_mode=true` but the runtime lacks required lifecycle APIs, trainer initialization MUST fail fast (before training begins) with actionable guidance.

## 5. Eval Robustness: Token-Trace Fallback + Per-Sample Decode Skip

- [x] 5.1 Add a parser regression test that feeds a fake traced vLLM output into the traced-output parser and asserts strict invariants: `len(token_ids)==len(token_logprobs)==len(generated_token_text)` and finiteness checks.
- [x] 5.2 Wire eval-step confidence scoring to degrade gracefully on missing/invalid traces when `score_mode=confidence_postop` and backend is vLLM:
  - emit a warning + counter/metric (`eval/trace_fallback_count`),
  - fall back to constant-score policy for that evaluation window,
  - do not abort training.
- [x] 5.3 Add a regression test that simulates a token-trace invariant violation during eval and asserts we fall back (no exception escapes `evaluate()`).
- [x] 5.4 Add per-sample vLLM decode error handling for eval-only vLLM:
  - skip the failed sample (do not crash eval),
  - increment `eval/vllm_decode_error_count`,
  - ensure training continues.
- [x] 5.5 Add a regression test that injects a per-sample decode error and asserts:
  - the sample is skipped,
  - `eval/vllm_decode_error_count` increments,
  - `evaluate()` returns normally.
- [x] 5.6 Add an engine-level failure regression for eval-only vLLM:
  - simulate vLLM engine init failure (e.g., raising during engine construction or an eval-time OOM in engine init),
  - assert `evaluate()` fails fast with actionable guidance (no silent fallback to HF).
- [x] 5.7 Remove/adjust "experimental" warnings/comments once the above regressions are in place and passing.

## 6. Example Configs + Runbook Notes

- [x] 6.1 Add a new prod-style config `configs/stage2_two_channel/prod/desc_first_a_only_eval_vllm_colocate.yaml` extending `configs/stage2_two_channel/prod/desc_first_a_only.yaml` that enables vLLM colocate for eval only (Channel-A-only schedule).
- [x] 6.2 Add a smoke config that forces rapid eval (small `val_sample_limit`, small `eval_steps`) to validate lifecycle + tracing on a handful of samples.
- [x] 6.3 Document recommended knobs for A100-80GB (e.g., `gpu_memory_utilization`, `max_model_len`, `max_num_seqs`) in an appropriate existing doc or change note.

## 7. Validation Checklist (Manual / Smoke)

- [ ] 7.1 Smoke-run: 8-GPU Stage-2 Channel-A-only training under plain DDP (`deepspeed.enabled=false`) with vLLM eval override; verify evaluation completes and training resumes (no post-eval OOM).
- [ ] 7.2 Verify full-sync invariant is enforced by intentionally misconfiguring adapter-only settings and observing fail-fast behavior.
- [ ] 7.3 Verify confidence scoring path runs under vLLM traced eval; if trace fallback occurs, it is logged/metric'd and training continues.
- [ ] 7.4 Optional dependency verification: validate ms-swift multimodal `RequestConfig(logprobs=True)` returns `logprobs.content` for an image prompt if a local ms-swift checkout is available (example: `conda run -n ms pytest /data/ms-swift/tests/infer/test_mllm.py::test_stream`).
