## Context

Stage-2 AB Channel-B uses rollout generation from data -> decode -> parse/match -> teacher-forced training. In HF backend, `repeat_terminate` already has a guard path; in vLLM server mode this guard is currently absent, so long degenerate tails can consume rollout budget and degrade training-time TP efficiency. We need parity across backends while preserving throughput-critical batching in rollout serving.
For the current stack (vLLM V1-default), per-request `logits_processors` are not a viable activation path in rollout request payloads, so repeat-aware behavior must be attached at rollout-server startup/engine config.

## Goals / Non-Goals

**Goals:**
- Apply repeat-aware early termination in vLLM rollout serving with behavior aligned to existing YAML repeat-termination semantics.
- Preserve per-sequence batching behavior (only terminate offending sequences).
- Keep config-first flow and avoid new CLI flags.
- Keep Qwen3 chat-template compatibility and existing geometry/loss contracts unchanged.
- Avoid introducing new duplicate helper modules during implementation; reuse canonical helper surfaces (e.g., `src/common/*`) established by schema/module refactors.

**Non-Goals:**
- No change to matching/loss formulation.
- No change to upstream HF internals.
- No custom RL loop or architecture fork.

## Decisions

1. Use a server-side vLLM logits processor (repeat-aware, config-driven by `repeat_terminate`) instead of client-side trimming.
- Rationale: only server-side intervention saves generation compute.
- Alternative: parse-time truncation on learner side.
- Rejected: does not reduce rollout latency or token waste.

2. Implement termination as per-sequence EOS forcing, not full-batch abort.
- Rationale: preserves batching and avoids penalizing healthy sequences.
- Alternative: cancel entire request batch on any repeat trigger.
- Rejected: large throughput loss and unstable async queue utilization.

3. In vLLM mode, hard-require startup-time processor activation for this stack; do not rely on request-time processor payloads.
- Rationale: request payload path is non-executable for V1-default deployment.
- Alternative: keep dual path (startup + request-time) with fallback.
- Rejected: introduces ambiguous behavior and version-coupled reproducibility drift.

4. Keep YAML as source of truth and propagate the full `repeat_terminate` subtree into vLLM rollout server startup config.
- Rationale: reproducible config-first workflow; no new ad-hoc flags.
- Alternative: partial key propagation or derived defaults in rollout client.
- Rejected: parity breaks with HF and silently changes effective thresholds.

5. Fail fast when repeat-aware behavior is enabled by config but cannot be activated.
- Rationale: silent fail-open breaks reproducibility and training validity.
- Alternative: best-effort no-op when processor unavailable.
- Rejected: hides contract violations and invalidates run-to-run comparisons.

6. Require explicit activation visibility in logs/metrics with fixed keys.
- Rationale: avoids silent behavior drift between HF and vLLM runs.
- Alternative: free-form log messages only.
- Rejected: difficult to audit and regression-test.

7. Do not modify external dependency source code to implement repeat-aware vLLM behavior.
- Rationale: preserve dependency integrity while still enabling server-side repeat termination.
- Alternative: patch ms-swift/vLLM source files in-place.
- Rejected: increases maintenance risk and violates the intended dependency boundary.

## Activation Mechanism (vLLM Server Mode)

The vLLM rollout backend for Stage-2 AB runs as a separately-launched server process (external dependency stack, launched via `swift rollout`). For the current vLLM V1-default workflow:
- Activation MUST happen at **server startup** (request-time injection is not a viable contract for correctness).
- The learner-side YAML is the only source of truth for `custom.extra.rollout_matching.repeat_terminate`, so the full subtree MUST be transmitted to the server startup process in a reproducible way.

Why startup injection is required (non-exhaustive):
- The rollout server request schema (`RequestConfig`) has no typed place to carry a logits-processor payload; the server endpoint accepts `infer_requests` + `request_config` only. Therefore, repeat-aware activation MUST NOT depend on learner-side per-request injection of `logits_processors`.

Compliant approach (repo-owned, no external library source edits; recommended):
- The Stage-2 AB server launcher starts `swift rollout` with a repo-owned plugin loaded via `--external_plugins <path>`.
  - This import happens before engine init in each worker process, so the plugin can install startup-time patches safely.
- The Stage-2 AB server launcher exports environment variables to the **server process only**:
  - guard: `COORDEXP_ENABLE_VLLM_REPEAT_TERMINATE_INJECTION=1`
  - config: `COORDEXP_VLLM_REPEAT_TERMINATE_CONFIG_JSON=<json>` containing the full `repeat_terminate` subtree.
    - If env-var JSON quoting/length is problematic, a compliant alternative is `COORDEXP_VLLM_REPEAT_TERMINATE_CONFIG_JSON_PATH=<path>` pointing to a generated JSON artifact.
- The repo-owned plugin MUST:
  - validate the presence and schema of the repeat-terminate config when `repeat_terminate.enabled: true`,
  - patch the rollout server's vLLM SamplingParams construction so a per-request repeat-aware processor is attached via `SamplingParams.logits_processors` (server-side; no learner-side request injection),
  - preserve batching by forcing per-sequence EOS (no full-batch abort),
  - expose an explicit trigger signal (per sequence) back to the learner-facing response path.

Alternative compliant approach (fallback):
- If `--external_plugins` is unavailable in the deployment stack, a repo-owned `sitecustomize.py` injection loaded via `PYTHONPATH` may be used to install the same patches before engine init. This fallback MUST still respect the "no external library source edits" rule.

Explicit trigger signal transport (normative):
- The vLLM server `/infer/` response MUST support a wrapper envelope per output item:
  - `{"response": <ChatCompletionResponse-dict>, "coordexp": {"repeat_terminate_triggered": 0|1}}`
  - This is additive-only: existing clients that only read `response` remain compatible.
- The learner MUST derive `rollout/repeat_terminate_triggered_sequences` from this explicit wrapper signal, not from stop-reason heuristics.

If no supported extension point exists in the dependency stack, this change MUST introduce a small, repo-owned wrapper around server startup that performs the injection without modifying dependency source code.

## Risks / Trade-offs

- [Risk] vLLM plugin/load-path mismatch causes rollout startup failure -> Mitigation: add startup validation and clear error messages before training begins.
- [Risk] Aggressive thresholds may cut useful long generations -> Mitigation: keep thresholds config-driven and retain existing defaults semantics from YAML.
- [Risk] Backend-parity edge cases between HF and vLLM tokenization -> Mitigation: parity tests using the same prompts and guard settings.
- [Risk] External ms-swift/vLLM dependency changes behavior across versions -> Mitigation: pin tested versions in run metadata and document compatibility assumptions.

## Migration Plan

Sequencing note:
- If also planning to land `remove-stage2-ab-stop-neutral`, land this repeat-aware vLLM change first to isolate decode-time tail-control effects from training-objective changes.

1. Update rollout backend contract spec deltas for vLLM repeat-aware support.
2. Implement server-side logits processor wiring and config mapping at rollout-server startup.
3. Remove/replace any “HF-only / ignored by vLLM” config comments/docs that conflict with the new contract.
4. Add unit/integration tests for full-subtree propagation, fail-fast startup validation, and per-sequence EOS/no-batch-abort behavior.
5. Run a bounded Stage-2 AB smoke with vLLM server mode to confirm repeat-aware activation, per-sequence EOS (no batch abort), and audit metrics emission.
6. Rollback path: disable repeat-aware processor via YAML and revert to existing behavior if regressions appear.

## Resolved Choices

- Runtime path: startup-time processor activation in vLLM rollout server for this stack.
- Error policy: fail fast when `repeat_terminate.enabled: true` but processor activation is unavailable.
- Telemetry contract: expose concrete activation + trigger counters for auditability.
