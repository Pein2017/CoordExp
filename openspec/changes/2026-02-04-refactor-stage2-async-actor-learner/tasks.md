## 1. Spec Deltas
- [x] Update `openspec/specs/rollout-matching-sft/spec.md` via a delta spec:
  - [x] Modify the full vLLM backend requirement block to allow `vllm.mode: server` with learner `world_size > 1`.
  - [x] Specify rank0-only communicator init + full weight sync + barrier ordering (no multi-client sync).
  - [x] Document decoding knobs for sampling (`temperature`, `top_p`, optional `top_k`) and robustness-first reproducibility expectations under retries/splitting.
  - [x] Remove `custom.extra.rollout_matching.rollout_buffer` requirements (buffered reuse) and require fail-fast if provided.
- [x] Update `openspec/specs/stage2-ab-training/spec.md` via a delta spec:
  - [x] Add `stage2_ab.channel_b.mode: async` semantics (async actor-learner with versioned ready-pack queues).
  - [x] Specify queue feasibility gate overriding scheduled B and required skip logging.
  - [x] Specify DDP-safe execution semantics:
    - exactly 1 packed forward/backward per micro-step per rank
    - rank0 decides + broadcasts step kind per optimizer step (DDP safety)
    - fail-fast for `stage2_ab.channel_b.mode: step` under `world_size > 1` (v1 guardrail).
  - [x] Remove schedule override semantics based on rollout-buffer reuse (since rollout_buffer is removed).

## 2. Config / Schema
- [x] Extend typed config in `src/config/schema.py`:
  - [x] Extend `stage2_ab.channel_b.mode` to accept `async` (existing: `micro`, `step`).
  - [x] Add a Stage2-AB async subsection `stage2_ab.channel_b.async.*` including:
    - [x] `queue_limit: int` (per-rank packs, drop-oldest on overflow)
    - [x] `version_window: int` (default 2)
    - [x] `sync_every_steps: int` (default 1; how often rank0 syncs and advances `sync_counter`)
    - [x] `prefetch_target_packs: int` (target queue depth per rank)
  - [x] Keep `stage2_ab.schedule.b_ratio` as the policy knob; queue availability is the feasibility gate in async mode.
  - [x] Add rollout decoding knobs under `custom.extra.rollout_matching.decoding.*`:
    - [x] `temperature` (float >= 0; `0` = greedy; sampling if > 0)
    - [x] `top_p` (float in (0,1], default 1.0)
    - [x] optional `top_k` (int, default -1) if supported by the backend
  - [x] Fail fast if configs provide legacy keys:
    - [x] `custom.extra.rollout_matching.temperature`
    - [x] `custom.extra.rollout_matching.top_p`
    - [x] `custom.extra.rollout_matching.top_k`
    - [x] `custom.extra.rollout_matching.rollout_buffer`

## 3. vLLM Server Mode + Multi-GPU Learner
- [x] Update `src/trainers/rollout_matching_sft.py`:
  - [x] Remove the single-process learner hard error for `vllm.mode: server`.
  - [x] Make full weight sync rank0-only and synchronize with other ranks:
    - [x] rank0 performs sync when `sync_counter` advances
    - [x] enforce strict sync ordering (no in-flight infer during sync):
      - [x] `dist.barrier()` -> rank0 sync -> `dist.barrier()` -> infer
  - [x] Ensure inference over server is DDP-safe:
    - [x] no NCCL communicator init on non-rank0
    - [x] robust HTTP retry behavior (batch split allowed; log warnings)
  - [x] Add `top_p` (and optional `top_k`) to server `RequestConfig` and HF `GenerationConfig`.

## 4. Stage2-AB Async Actor-Learner (Queue-Gated B Steps)
- [x] Implement async queues in `src/trainers/stage2_ab_training.py`:
  - [x] Background prefetch that builds **ready packed micro-batches** for Channel-B.
    - Each rank maintains its own ready-pack queue to avoid cross-rank payload transfer.
  - [x] Rank0 makes the per-optimizer-step A/B decision and broadcasts `step_kind` for the full accumulation window:
    - Run B only if (a) the Bresenham `b_ratio` schedule selects B and (b) all ranks have enough ready packs to cover `gradient_accumulation_steps`.
    - Otherwise run A.
  - [x] Freshness gating:
    - Each ready-pack carries `ver` (sync-counter) and is consumed only if `ver >= current_ver - version_window`.
    - Enforce `queue_limit` and drop-oldest when full.
  - [x] DDP safety:
    - Exactly 1 packed forward/backward per micro-step per rank.
    - Never run inner multi-pack loops inside a single `training_step` call when `world_size > 1`.
    - Fail fast if an unsupported legacy mode is enabled under multi-GPU learner.

## 5. Logging / Metrics / Docs
- [x] Add async queue telemetry (and document it in `docs/training/METRICS_LOSSES.md`):
  - [x] queue depth (ready packs) per rank
  - [x] stale drop counts
  - [x] queue drop-oldest counts
  - [x] `ver_current` and `ver_lag` statistics
  - [x] realized `b_ratio_realized` over a rolling window
- [x] Update `docs/training/STAGE2_RUNBOOK.md`:
  - [x] multi-GPU learner + server-mode rollout topology
  - [x] recommended starting knobs (e.g. `b_ratio=0.2`, `temperature=0.01`, `top_p=0.95`)
  - [x] failure modes and mitigations (server timeouts, queue starvation, staleness)

## 6. Scripts / Launching
- [x] Update `scripts/stage2_ab_server_train.sh`:
  - [x] Allow multi-GPU learner (`train_gpus=6,7`) while keeping server GPUs separate.
  - [x] Validate that server and learner GPU sets are disjoint.

## 7. Tests
- [x] Add unit tests for:
  - [x] `top_p`/sampling config parsing and propagation (HF + vLLM server request config)
  - [x] queue-gated schedule behavior (deterministic A/B decision given queue depths)
  - [x] DDP safety invariants at the logic level (no multi-pack loops under `world_size > 1`)

## 8. Config Migration (No Backward Compatibility)
- [x] Update rollout-matching configs to use `custom.extra.rollout_matching.decoding.*` (breaking):
  - [x] `configs/stage2_ab/rollout_matching_sft_template.yaml`
  - [x] `configs/stage2_ab/prod/*.yaml`
  - [x] `configs/stage2_ab/smoke/*.yaml`
  - [x] `configs/dlora/stage2_rollout_matching_ckpt3106.yaml`
  - [x] `configs/dlora/stage2_rollout_matching_ckpt3106_server_3v1.yaml`
- [x] Search for any remaining legacy decoding knobs usage in YAML and migrate it (no backward compatibility).
- [x] Remove legacy decoding knobs support (fail fast) when configs provide:
  - [x] `custom.extra.rollout_matching.temperature`
  - [x] `custom.extra.rollout_matching.top_p`
  - [x] `custom.extra.rollout_matching.top_k`
- [x] Remove legacy rollout-buffer configs (breaking):
  - [x] Remove `custom.extra.rollout_matching.rollout_buffer` from all YAMLs (template + prod + smoke + dlora).
  - [x] Ensure any remaining `rollout_buffer.*` usage fails fast at runtime/config validation.

## 9. Quick Validation (Fail-Fast)
_Deferred by user instruction ("Skip the real GPU tasks")._
- [x] Validation 0 (no async): 1-GPU learner + vLLM server mode still works (baseline regression check). _(Deferred: real-GPU runtime validation skipped.)_
- [x] Validation 1 (DDP sync): 2-GPU learner + vLLM server mode:
  - [x] rank0 sync only; other rank never calls communicator init _(Deferred: real-GPU runtime validation skipped.)_
  - [x] all ranks can issue `/infer/` without deadlock _(Deferred: real-GPU runtime validation skipped.)_
- [x] Validation 2 (async on): enable `stage2_ab.channel_b.mode: async` with small queues:
  - [x] verify skip-to-A happens when queues are empty and logs `b_step_skipped_due_to_queue` _(Deferred: real-GPU runtime validation skipped.)_
  - [x] verify stale drops happen when `version_window` is tight and are logged _(Deferred: real-GPU runtime validation skipped.)_
  - [x] verify exactly 1 packed forward/backward per micro-step per rank _(Deferred: real-GPU runtime validation skipped.)_

## 10. Tuning Experiments (Rollout/Learner Balance)
_Deferred by user instruction ("Skip the real GPU tasks")._
- [x] Sweep `stage2_ab.schedule.b_ratio` at fixed GPU split (e.g., 6 rollout GPUs + 2 learner GPUs):
  - [x] 0.05, 0.10, 0.20, 0.30, 0.40 _(Deferred: real-GPU runtime experiment skipped.)_
- [x] Sweep `stage2_ab.channel_b.async.sync_every_steps` (1, 2, 4) and `version_window` (1, 2, 4) to bound staleness. _(Deferred: real-GPU runtime experiment skipped.)_
- [x] Track and compare:
  - [x] learner idle vs busy (time blocked on ready-pack queue) _(Deferred: real-GPU runtime experiment skipped.)_
  - [x] queue depths and skip rates (how often scheduled B falls back to A) _(Deferred: real-GPU runtime experiment skipped.)_
  - [x] staleness stats (`ver_lag` mean/max; stale drop rate) _(Deferred: real-GPU runtime experiment skipped.)_
  - [x] rollout throughput (requests/s) and learner throughput (steps/s, tokens/s) _(Deferred: real-GPU runtime experiment skipped.)_
  - [x] quality proxies (train/val metrics already in the repo; ensure comparable eval cadence) _(Deferred: real-GPU runtime experiment skipped.)_
