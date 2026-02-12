## 0. Prerequisites (Refactor First)

- [x] 0.1 Land the schema/module refactor work that clarifies canonical helper ownership and leaves compatibility shims as needed (e.g., `openspec/changes/src-ambiguity-cleanup-2026-02-11`).
- [x] 0.2 When implementing this change, avoid introducing new parallel helpers/regexes; reuse canonical helper surfaces under `src/common/*` and canonical coord-token helpers under `src/coord_tokens/codec.py` where applicable.

## 1. Contract and Config Plumbing

- [x] 1.1 Update rollout config parsing so the full `custom.extra.rollout_matching.repeat_terminate` subtree is available to vLLM server startup path.
- [x] 1.2 Define and implement how the full `repeat_terminate` subtree is transmitted into the separately-launched vLLM server process at startup.
  - Recommended compliant approach: the server launcher (`scripts/stage2_ab_server_train.sh`) exports an env-var JSON blob for the subtree to the server process (e.g., `COORDEXP_VLLM_REPEAT_TERMINATE_CONFIG_JSON`).
- [x] 1.3 Implement repo-owned `swift rollout` plugin injection via `--external_plugins <repo-owned-plugin>` (server process only) to register repeat-aware logits processing without modifying external dependency source code.
  - Guard: `COORDEXP_ENABLE_VLLM_REPEAT_TERMINATE_INJECTION=1`
  - Fallback (only if required by environment): `PYTHONPATH` + repo-owned `sitecustomize.py` may be used to load the same plugin code before engine init.
- [x] 1.4 Add startup validation that fails fast when repeat-aware processing is enabled but cannot be activated.
- [x] 1.5 Keep config-first behavior (no new CLI flags) and verify existing YAMLs still parse.
- [x] 1.6 Update legacy config/docs comments that still say repeat-terminate is HF-only or ignored by vLLM (including `configs/stage2_ab/base_rollout_matching_sft.yaml`).

## 2. vLLM Server Repeat-Aware Processing

- [x] 2.1 Implement vLLM repeat-aware logits processing (repeat-terminate, force per-sequence EOS) in rollout serving path.
- [x] 2.2 Wire repeat thresholds (`min_new_tokens`, repeat/ngram limits, `max_object_keys`) into processor state.
- [x] 2.3 Ensure processing preserves batching (no full-batch abort/cancel when one sequence triggers).
- [x] 2.4 Ensure vLLM V1 path does not depend on per-request `logits_processors` payload injection for activation.
- [x] 2.5 Ensure the implementation does **not** modify external library source code (ms-swift/vLLM). Use supported extension points or a repo-owned wrapper/injection module loaded at server startup.
- [x] 2.6 Expose a deterministic "repeat termination triggered" signal from the server/processor to the learner-facing response path.
  - Normative schema: `/infer/` returns per-output wrapper `{"response": <...>, "coordexp": {"repeat_terminate_triggered": 0|1}}`.
  - The learner counts triggers from `coordexp.repeat_terminate_triggered` (not finish-reason heuristics) to emit `rollout/repeat_terminate_triggered_sequences`.

## 3. Stage-2 Integration

- [x] 3.1 Integrate Channel-B vLLM rollout calls with startup-validated repeat-aware activation state and deterministic behavior.
- [x] 3.2 Add telemetry/log keys `rollout/repeat_terminate_active` and `rollout/repeat_terminate_triggered_sequences`.
- [x] 3.3 Confirm HF guard behavior remains unchanged for `rollout_backend: hf`.
- [x] 3.4 Ensure all metrics emitted via the neutral trainer-metrics payload `metrics` map are **global** aggregates after grad-accum aggregation and DDP all-reduce (no rank-local training metrics).
  - Counters use global sum; wall time uses global max; boolean-style keys use global max; rates use ratio-of-global-sums.
- [x] 3.5 Ensure tail-control metrics match the spec:
  - `rollout/parse_truncated_rate` uses ratio of global sums (not mean of rank-local ratios),
  - `rollout/gen_new_tokens_p99` uses all-reduce max over rank-local p99 (simple conservative proxy).

## 4. Tests

- [x] 4.1 Add unit tests for repeat-aware trigger conditions (consecutive token repeats, repeated n-grams, optional object-key cap).
- [x] 4.2 Add server-side test proving only offending sequences are force-EOS terminated within a batch.
- [x] 4.3 Add integration test for YAML full-subtree propagation in vLLM mode and fail-fast on missing processor.
- [x] 4.4 Add regression test that vLLM rollout activation does not rely on request-time logits-processor kwargs.
- [x] 4.5 Add regression test asserting FP/matching and geometry supervision semantics are unchanged by repeat-aware activation.
- [x] 4.6 Add regression test asserting telemetry key presence/semantics:
  - `rollout/repeat_terminate_active` is emitted and is in `{0,1}` after global aggregation,
  - `rollout/repeat_terminate_triggered_sequences` is emitted and is a non-negative global counter (sum-aggregated).
- [x] 4.7 Add regression test asserting `rollout/repeat_terminate_triggered_sequences` is sourced from an explicit server/processor signal (not inferred from stop-reason strings).
- [x] 4.8 Run targeted tests: `conda run -n ms python -m pytest tests/test_stage2_ab_training.py`.
- [x] 4.9 Run boundary + payload contract tests: `conda run -n ms python -m pytest tests/test_stage2_rollout_import_boundaries.py tests/test_trainer_metrics_payload_contract.py`.

## 5. Lightweight Validation (No Baseline Gate)

- [x] 5.1 Run a bounded Stage-2 AB smoke with vLLM server mode and `repeat_terminate.enabled: true`.
- [x] 5.2 Verify audit metrics keys are present and have sane ranges:
  - `rollout/repeat_terminate_active` is in `{0,1}`,
  - `rollout/repeat_terminate_triggered_sequences` is `>= 0`,
  - `rollout/parse_truncated_rate` is in `[0, 1]`,
  - `rollout/gen_new_tokens_p99` is `>= 0`,
  - `rollout/parse_dropped_invalid` is `>= 0`.
- [x] 5.3 Record run metadata (config path, run name, output dir, git SHA) in change notes for reviewer audit.
  - 2026-02-12 bounded smoke: `config=/data/CoordExp/temp/smoke_stage2_ab_repeat_stop.yaml`
  - run name: `smoke_contract_repeat_stop`
  - output dir: `/data/CoordExp/output/stage2_ab/smoke_contract/repeat_stop/v9-20260212-003102/smoke_contract_repeat_stop`
  - git SHA: `9495f7d969acaafd5d14333ffde4357c63f3ec25`
  - metrics spot-check:
    - `rollout/repeat_terminate_active=1.0`
    - `rollout/repeat_terminate_triggered_sequences=0.0`
    - `rollout/parse_truncated_rate=1.0`
    - `rollout/gen_new_tokens_p99=128.0`
    - `rollout/parse_dropped_invalid=1.0`

## 6. Docs and Contract Alignment

- [x] 6.1 Update `docs/training/STAGE2_RUNBOOK.md` guidance to remove HF-only `repeat_terminate` wording and document vLLM server-mode repeat termination activation (startup injection/wrapper approach).
- [x] 6.2 Update `docs/training/METRICS_LOSSES.md` to align training metric aggregation language with this change (training-time `rollout/*` keys are global aggregates; remove/qualify any "rollout/* are always rank-local" statements).
- [x] 6.3 Update `docs/training/STAGE2_RUNBOOK.md` to match `openspec/specs/rollout-matching-sft/spec.md` for vLLM server mode + multi-process learner support (including rank0-only sync ordering and `vllm.sync.mode: full` under `world_size > 1`).
