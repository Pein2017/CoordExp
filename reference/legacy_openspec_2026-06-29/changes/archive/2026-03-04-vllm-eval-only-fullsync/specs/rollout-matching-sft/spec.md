## ADDED Requirements

### Requirement: Evaluation can override rollout backend independently of training
The system SHALL allow Stage-2 evaluation (`eval_step`) rollouts to select a rollout backend independently of the training-time rollout backend.

Normative behavior:
- `rollout_matching.rollout_backend` continues to define the rollout backend for training-time rollouts (e.g., Stage-2 Channel-B).
- A new optional key `rollout_matching.eval_rollout_backend` SHALL be supported:
  - when `null` or missing, evaluation rollouts MUST use `rollout_matching.rollout_backend`,
  - when set to `hf` or `vllm`, evaluation rollouts MUST use `rollout_matching.eval_rollout_backend` and MUST NOT affect training rollouts.
- `rollout_matching.eval_rollout_backend` MUST accept only `null`, `hf`, or `vllm` (case-insensitive). Missing MUST be treated as `null`. Any other value MUST fail fast with actionable guidance.
- If the effective evaluation rollout backend resolves to `vllm`, the system MUST enforce the same vLLM length-coherence guardrails as when `rollout_matching.rollout_backend: vllm`, including:
  - `rollout_matching.max_new_tokens < rollout_matching.vllm.max_model_len` (to avoid truncation/overflow), and
  - `rollout_matching.vllm.max_model_len >= global_max_length` (to avoid silent truncation drift between training and rollouts).

#### Scenario: Eval backend override uses vLLM while training uses HF
- **GIVEN** `rollout_matching.rollout_backend: hf`
- **AND** `rollout_matching.eval_rollout_backend: vllm`
- **WHEN** the trainer runs `evaluate()` at an `eval_step`
- **THEN** evaluation rollouts are generated via the vLLM backend
- **AND** training-time rollouts (if any) continue to use the HF backend.

#### Scenario: Missing eval override inherits training backend
- **GIVEN** `rollout_matching.rollout_backend: vllm`
- **AND** `rollout_matching.eval_rollout_backend` is missing (or null)
- **WHEN** the trainer runs `evaluate()`
- **THEN** evaluation rollouts use the vLLM backend.

### Requirement: vLLM rollouts require full merged-weight sync (no adapter-only sync)
In this stack, vLLM rollouts are supported only via full merged-weight synchronization into the vLLM engine ("full sync"). Adapter-only synchronization (vLLM LoRA upload / `add_lora`) is unsupported and MUST be rejected.

Normative behavior:
- When the effective rollout backend is `vllm` (training or eval):
  - The system MUST perform a full merged-weight sync into vLLM before issuing rollouts.
  - The system MUST NOT use vLLM adapter-only sync (no `add_lora` / adapter-only upload path).
- If configuration requests vLLM adapter-only sync while the effective backend is vLLM (e.g., `rollout_matching.vllm.enable_lora: true`), the run MUST fail fast before starting rollouts with actionable guidance.

#### Scenario: Adapter-only sync is rejected for vLLM rollouts
- **GIVEN** a config where the effective rollout backend is `vllm`
- **AND** config requests vLLM LoRA / adapter-only sync for rollout weights
- **WHEN** rollout generation begins (training or eval)
- **THEN** the run fails fast with an error stating that vLLM rollouts require full merged-weight sync.

### Requirement: Eval-only colocate vLLM MAY release GPU memory after evaluation (optional sleep-after-eval)
When evaluation rollouts use vLLM in colocate mode, the system MUST preserve training correctness and MUST be DDP-safe. vLLM sleep mode is an optional/advanced optimization and MUST be disabled by default in standard colocate mode due to observed teardown incompatibilities in our environment.

Normative behavior:
- When `rollout_matching.vllm.mode: colocate` and the effective evaluation rollout backend resolves to `vllm`:
  - The system MUST NOT attempt to "shutdown" vLLM in-process as part of the evaluation lifecycle (DDP safety).
  - Default behavior MUST NOT require vLLM sleep mode:
    - The system MUST NOT force vLLM sleep mode enablement at engine construction time (e.g., do not unconditionally set `enable_sleep_mode=true`).
    - Absence of vLLM sleep/wake APIs MUST NOT fail the run.
  - If vLLM sleep mode is explicitly enabled for the run (advanced / operator-controlled):
    - The system MUST ensure the vLLM engine is awake before issuing any evaluation rollouts.
      - If the engine was previously slept, the system MUST call `LLMEngine.wake_up()` (or a version-equivalent wake method) before generating any evaluation rollouts.
    - The system SHOULD call vLLM sleep at the end of `evaluate()` to release GPU allocations between eval windows.
      - Recommended: `LLMEngine.sleep(level=2)` (or a version-equivalent sleep method).
    - The system MUST ensure vLLM is configured to support sleep mode so sleep/wake actually affects GPU allocations.
      - This MUST be enabled at engine construction time (e.g., `EngineArgs(enable_sleep_mode=true)` in vLLM 0.11.x).
      - Failure to enable sleep mode (or lack of required vLLM APIs) MUST fail fast with actionable guidance *before training begins*, so the run cannot silently proceed with incorrect memory expectations.

#### Scenario: Optional colocated vLLM sleep-after-eval lifecycle
- **GIVEN** evaluation rollouts use `vllm` with `vllm.mode: colocate`
- **AND** vLLM sleep mode is enabled for the run (advanced)
- **WHEN** `evaluate()` completes
- **THEN** vLLM engine resources are transitioned to a low-GPU-memory state (recommended: sleep level `2`)
- **AND** the next `evaluate()` call wakes the vLLM engine before issuing rollouts.

### Requirement: Optional HF offload is supported for vLLM evaluation rollouts under plain DDP (colocate vLLM)
To reduce peak GPU memory when evaluation uses vLLM colocate mode, the system SHALL support offloading HF training state under operator control **when training runs under plain DDP** (one full model replica per rank).

Normative behavior:
- The system MUST support `rollout_matching.offload.enabled: true` during evaluation-only colocate vLLM.
- If `rollout_matching.offload.enabled: true`, the system MUST offload the requested training state before issuing vLLM rollouts and MUST restore the requested state after evaluation completes.
- Offload defaults:
  - When `rollout_matching.offload.enabled: true` and `rollout_matching.offload.offload_model` is missing, it MUST default to `true`.
  - When `rollout_matching.offload.enabled: true` and `rollout_matching.offload.offload_optimizer` is missing, it MUST default to `true`.
- Offloading MUST NOT be enabled under runtimes that partition or alias optimizer/model state (e.g., DeepSpeed/ZeRO). In particular, if `deepspeed.enabled: true` and rollout offload is requested for vLLM colocate (train or eval), the run MUST fail fast before starting rollouts with actionable guidance.

#### Scenario: DeepSpeed is rejected for eval-time offload under colocate vLLM
- **GIVEN** `deepspeed.enabled: true`
- **AND** evaluation rollouts use `vllm` with `vllm.mode: colocate`
- **AND** rollout offload is enabled
- **WHEN** evaluation begins
- **THEN** the run fails fast with an error stating that eval-time offload is only supported under plain DDP (no DeepSpeed/ZeRO).

### Requirement: vLLM traced rollouts satisfy token-trace invariants for confidence scoring (eval-safe fallback)
When evaluation requests confidence scoring derived from token logprob traces, vLLM traced rollouts MUST produce a well-formed trace for every sample. If traces are invalid during evaluation, the system MUST degrade gracefully so training can continue.

Normative behavior:
- If eval-step scoring mode requires token traces (e.g., confidence post-op scoring), vLLM evaluation MUST request logprobs and MUST run in greedy mode (`temperature=0`).
- For each evaluated sample, the vLLM backend MUST return:
  - `token_ids` (generated completion token ids),
  - `token_logprobs` (list[number]),
  - `generated_token_text` (list[string]),
  and MUST satisfy `len(token_ids) == len(token_logprobs) == len(generated_token_text)`.
- If any sample violates these invariants during **evaluation**, the system MUST:
  - emit a warning (rate-limited) that confidence scoring fell back due to invalid token traces,
  - increment an explicit metric/counter `eval/trace_fallback_count` by 1 for each affected sample, and
  - continue evaluation (do not abort training), falling back to a safe score policy equivalent to `score_mode: constant` using `constant_score` for the affected evaluation window.

#### Scenario: Trace length mismatch falls back during evaluation
- **GIVEN** eval-step confidence scoring is enabled
- **AND** evaluation backend is `vllm`
- **WHEN** any vLLM rollout returns `len(token_ids) != len(token_logprobs)` or `len(token_ids) != len(generated_token_text)`
- **THEN** evaluation continues and uses constant-score fallback for confidence scoring, and a warning is emitted indicating a token-trace invariant violation.

### Requirement: Eval-only vLLM rollouts MUST skip per-sample decode failures; engine-level failures MUST fail fast
Evaluation rollouts are observability and model-quality measurement. This stack MUST be robust to rare sample-level decode failures, but MUST fail fast on engine-level failures (which indicate misconfiguration or an environment/runtime problem and should not be silently ignored).

Normative behavior:
- When the effective evaluation rollout backend resolves to `vllm`:
  - If an individual sample fails to decode/roll out due to a runtime error localized to that sample, the system MUST:
    - skip that sample (exclude it from downstream eval aggregation),
    - increment `eval/vllm_decode_error_count` by 1,
    - and continue evaluation.
  - If evaluation rollouts cannot proceed due to a vLLM engine-level failure (e.g., engine construction failure, missing/unsupported required lifecycle APIs for the configured mode (e.g., wake/sleep when sleep mode is enabled), OOM during eval, or a fatal runtime error), the system MUST:
    - fail fast with an actionable error message (do not hang),
    - and MUST NOT silently fall back to a different backend (e.g., HF) for that evaluation window.
- This best-effort behavior is scoped to **evaluation only**. Training-time rollouts (e.g., Stage-2 Channel-B) MAY continue to fail fast to preserve training semantics.
- This change intentionally does **not** define an "eval-window abort and continue training" metric path (e.g., no `eval/vllm_eval_aborted`). Engine-level vLLM failures are treated as fatal configuration/runtime errors and MUST fail fast.

#### Scenario: Per-sample decode errors are skipped but engine failures are fatal
- **GIVEN** evaluation backend resolves to `vllm`
- **WHEN** one sample decode fails but the vLLM engine remains healthy
- **THEN** evaluation skips that sample, increments `eval/vllm_decode_error_count`, and continues
- **AND WHEN** a later engine-level vLLM failure occurs
- **THEN** evaluation fails fast with an actionable error and does not fall back to HF.
