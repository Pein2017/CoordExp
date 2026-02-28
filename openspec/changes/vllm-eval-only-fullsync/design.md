## Context

Current state:
- Stage-2 trainers (`stage2_rollout_aligned`, `stage2_two_channel`) implement evaluation as rollout-first decoding (no teacher-forced loss), then parse + match + optionally run detection metrics. This means evaluation can use a different rollout backend without affecting the training objective.
- Rollout backends already support `hf` and `vllm`, and vLLM supports `colocate` (local engine) and `server` (external rollout server) modes.

User target:
- 8x A100-80GB.
- Training uses DoRA/LoRA; evaluation should use **vLLM without a separate rollout server**.
- Training is plain DDP: `dp == world_size` and each rank holds one full model replica (no tensor parallelism).
- Adapter-only sync is not supported in this stack; vLLM rollouts must always use **full merged-weight sync** (covers multimodal/image prompts).
- Prefer stable eval-only colocate behavior in long-lived DDP jobs; avoid in-process lifecycle paths that have shown allocator teardown instability.
- vLLM token-logprob tracing for eval-step confidence scoring should be treated as production behavior (not "experimental") once validated.
- For eval-time HF offload (model + optimizer), we target plain DDP (one full model replica per rank). DeepSpeed/ZeRO is treated as incompatible for offload in this change.

Constraints:
- Config-first (YAML), no new CLI flags.
- Preserve Qwen3-VL chat template compatibility.
- Preserve geometry invariants (coords never dropped/reordered).
- Do not edit upstream HF model internals.

## Goals / Non-Goals

**Goals:**
- Add a config-driven path to use **vLLM for evaluation rollouts only** while keeping training rollouts (if any) on the configured training backend.
- Guarantee **full sync** for vLLM rollouts:
  - colocate: sync by loading full merged weights into vLLM (no vLLM LoRA adapter uploads).
  - server: sync by full weight push (already required under DDP).
- Keep colocate lifecycle DDP-safe and stable:
  - initialize colocated vLLM lazily for evaluation windows,
  - keep standard colocate sleep mode disabled by default,
  - optionally offload HF training state during evaluation for VRAM headroom.
- Make vLLM traced logprob outputs for multimodal prompts part of the supported contract by adding regressions/guards.

**Non-Goals:**
- Replacing the entire rollout-matching training path with vLLM (this change is scoped to evaluation).
- Adding bespoke RL loops or architectural model forks.
- Changing dataset formats or coordinate conventions.

## Decisions

1. Backend override: add `rollout_matching.eval_rollout_backend` (optional)

Decision:
- Introduce `rollout_matching.eval_rollout_backend: null|hf|vllm` (default: `null` meaning inherit `rollout_matching.rollout_backend`).
- Evaluation code path uses `eval_rollout_backend` if provided; training uses existing `rollout_backend`.

Why:
- The user wants "vLLM for eval only" even when training schedule later enables rollouts (e.g., low `b_ratio` Channel-B). A dedicated eval override avoids coupling evaluation choice to training rollout choice.
- Backwards compatible: existing configs continue to behave identically when `eval_rollout_backend` is unset.
- Length-coherence guardrails must still apply: eval-only vLLM should not bypass existing protections against silent truncation drift (e.g., `vllm.max_model_len >= global_max_length`).

Alternatives considered:
- Rely on `b_ratio=0` and keep only `rollout_backend`. Rejected because it does not generalize to future low-`b_ratio` Channel-B, and it does not guarantee eval-only behavior under other schedules.

2. Full-sync enforcement for vLLM rollouts

Decision:
- Treat vLLM rollouts as supported **only** via full merged-weight sync ("full sync").
- Disallow adapter-only sync (vLLM LoRA upload / `add_lora`) for vLLM rollouts in this trainer:
  - Require `rollout_matching.vllm.enable_lora=false` when the effective backend is `vllm`, and fail fast if violated.

Why:
- Adapter-only sync is a frequent source of multimodal instability and correctness drift; eliminating it removes a large branch of "maybe works" behavior.
- Full sync is the robust path already used when `enable_lora=false` by merging adapters into the training model weights prior to syncing.

Alternatives considered:
- Allow `enable_lora=true` and rely on operator caution. Rejected: too easy to misconfigure and undermines eval validity.

3. Standard colocate lifecycle (sleep disabled by default) + optional HF offload (DDP-only)

Decision:
- When evaluation uses vLLM in colocate mode, default to standard colocate behavior:
  - keep vLLM sleep mode disabled by default,
  - do not require sleep/wake lifecycle APIs in default runs,
  - do not force `EngineArgs(enable_sleep_mode=true)`.
- Optional HF offload is operator-controlled via `rollout_matching.offload.enabled` and is scoped to the full evaluation window:
  - when enabled, offload model + optimizer by default (unless explicitly overridden),
  - offload happens once per `evaluate()` call (not per batch),
  - restore happens after evaluation completes.
- Compatibility: offload is supported only under plain DDP. DeepSpeed/ZeRO is treated as incompatible and must fail fast when offload is requested.

Why:
- In this environment, sleep-mode lifecycle handling has shown allocator teardown instability in long-lived DDP jobs (hard abort signatures including `CUDAPluggableAllocator::raw_delete` / pointer-free mismatch).
- Standard colocate with sleep disabled avoids that failure mode while preserving eval-only backend semantics.
- Compatibility: rollout offload assumes each rank owns a full set of model parameters and optimizer state tensors and can move them to/from CPU deterministically. DeepSpeed/ZeRO partitions and aliases optimizer/model state; we treat DeepSpeed as incompatible for offload and require fail-fast when offload is requested under DeepSpeed.
- We intentionally do **not** support an in-process "shutdown" lifecycle during training:
  - vLLM contains a helper `destroy_distributed_environment()` that calls `torch.distributed.destroy_process_group()` (default group) when `torch.distributed` is initialized.
  - Some vLLM shutdown paths (notably `v1/executor/multiproc_executor.py`) call `destroy_distributed_environment()` as part of executor shutdown.
  - In a long-lived training job that uses `torch.distributed` for DDP/FSDP/ZeRO, any shutdown path that destroys the default process group is a P0 correctness risk (it can break training collectives after eval).
  - Therefore, this change does not perform in-process teardown/shutdown during training.

Alternatives considered:
- Enable sleep mode by default and sleep vLLM after eval. Rejected as default due teardown instability in this environment; remains an advanced opt-in path.
- Use vLLM server mode to externalize memory. Rejected for this change because the user explicitly wants no separate rollout server.
- Shutdown colocated vLLM in-process after eval. Rejected for this change due to DDP-safety risk (see above); revisit only if vLLM teardown can be isolated from the training process group (e.g., out-of-process engine, or a proven-safe shutdown path under `external_launcher`).

4. Make vLLM logprob tracing non-experimental via regressions and eval-safe fallback

Decision:
- Keep the decoding constraints required for stable token traces:
  - traced mode requires greedy decoding (`temperature=0`, no sampling).
- Add tests/regressions that:
  - validate vLLM traced response parsing invariants (length alignment, finiteness),
  - cover multimodal RequestConfig(logprobs=True) behavior through upstream ms-swift tests,
  - cover confidence scoring pipeline assumptions using a small deterministic fixture.
- Eval-only robustness policy:
  - per-sample vLLM decode failures are skipped (metric: `eval/vllm_decode_error_count`),
  - token-trace invariant violations during eval confidence scoring fall back to constant-score (metric: `eval/trace_fallback_count`),
  - engine-level vLLM failures (init / missing required lifecycle APIs for configured mode / eval OOM) fail fast (no silent fallback to HF).

Why:
- Eval-step COCO scoring can depend on confidence scoring from token traces; trace corruption is an eval-validity risk.
- The core trace contract (1:1 alignment of generated token text and token logprobs) is already explicitly required by `confidence-postop`; we should enforce it at the rollout boundary.

## Risks / Trade-offs

- [Risk] Optional sleep-mode behavior can vary by vLLM version → Mitigation: keep sleep mode disabled by default; gate preflight and lifecycle checks to explicit sleep-mode runs.
- [Risk] Offloading HF state during eval may be slow (CPU transfer) → Mitigation: keep it operator-controlled; document expected overhead and suggest eval_steps cadence.
- [Risk] Full merged-weight sync into vLLM can be expensive → Mitigation: sync only once per `global_step` and only for eval windows; do not sync per batch.
- [Risk] DDP synchronization hazards if eval hooks diverge across ranks → Mitigation: ensure eval override + lifecycle logic runs on all ranks (no rank gating) and avoids asymmetric early returns.

## Migration Plan

- Add config keys with safe defaults (inherit existing behavior when unset).
- Provide new example profile(s) that extend existing prod configs and only change eval backend + vLLM colocate lifecycle.
- Keep existing `rollout_backend` semantics unchanged for training.
- Rollback is config-only: remove `eval_rollout_backend` and disable eval lifecycle/offload.

## Extension: `b_ratio > 0` (Channel-B) and why it "feels similar" to eval rollouts

When `stage2_ab.schedule.b_ratio > 0`, the training loop will run some **Channel-B steps**. Those steps require rollouts during training (unlike pure SFT teacher-forcing), which makes them operationally similar to eval rollouts:

- Both are "rollout windows": a segment of the training loop that must do inference (decode) and then return to training.
- Both need a consistent snapshot of model weights for rollout validity (and in this stack, vLLM requires **full merged-weight sync**).
- Both can benefit from the same memory-management primitives:
  - optionally offload HF training state during rollout,
  - run a decode backend (HF or vLLM),
  - restore HF state and resume training.

This change is intentionally scoped to **evaluation-only** vLLM. For `b_ratio > 0`, we can still unify infrastructure by reusing:

- the same backend resolver (`eval_rollout_backend ?? rollout_backend`), and
- the same "rollout window" lifecycle hooks (offload + optional vLLM sleep/wake when explicitly enabled), but applied to the Channel-B rollout window if/when we migrate Channel-B rollouts to vLLM.

Practically:
- `b_ratio > 0` does **not** force us to partition GPUs today if Channel-B rollouts remain on HF.
- If we later choose vLLM for Channel-B rollouts, then `b_ratio` becomes the key driver of whether we should stay "collocate" or introduce a learner/actor split.

## Performance intuition: separation vs collocate vs Channel-B with HF backend

There are three relevant modes once Channel-B exists (`b_ratio > 0`):

1. **Channel-B with HF backend (no vLLM)**:
   - Simplest mechanically (one stack).
   - Typically slowest for decoding throughput and least stable for very long-seq, high-throughput rollout workloads.

2. **Collocate (same 8 GPUs alternate between learner and vLLM actor inside the same processes)**:
   - Highest reuse of hardware; no reserved actor pool.
   - Overhead comes from:
     - full-sync into vLLM per rollout window,
     - HF offload/restore CPU transfers,
     - optional vLLM sleep/wake boundaries (only when explicitly enabled).
   - Best fit when rollouts are **infrequent** (low `b_ratio` or eval-only) because the overhead amortizes.

3. **Separation (dedicated actor GPUs for vLLM, dedicated learner GPUs for training)**:
   - Highest rollout throughput and least thrash once rollouts are common.
   - Complexity cost:
     - you need a weight-sync channel from learners to actors,
     - you need a result-return channel from actors to learners,
     - scheduling/backpressure becomes real (queueing).
   - Best fit when rollouts are **frequent** (higher `b_ratio`) because you avoid repeated offload/restore and keep vLLM hot.

Rule-of-thumb (qualitative, for planning):
- Eval-only rollouts and very low `b_ratio`: **collocate** tends to be faster-to-implement and good enough.
- Moderate/high `b_ratio` where a significant fraction of steps require rollouts: **separation** tends to be faster end-to-end.

## Open Questions

- Should eval-only vLLM be available for all trainers that inherit `RolloutMatchingSFTTrainer`, or limited to Stage-2 variants only?
- What is the preferred default for eval offload on non-A100 hardware (e.g., smaller GPUs)?
- If operators need stronger memory isolation than standard colocate + offload, should we prioritize server-mode guidance or an out-of-process lifecycle design?
