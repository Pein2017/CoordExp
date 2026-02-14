## Context

Stage2-AB Channel-B (rollout-matching supervision) runs a rollout phase (HF `generate()` or vLLM `/infer/`) followed by a teacher-forced learn-to-completion phase that may use dynamic post-rollout packing to `global_max_length`.

Today the pipeline exposes multiple batch-related knobs across:
- raw rollout budgeting (how many raw rollouts per optimizer step)
- rollout request chunking (how many requests to issue per RPC / per server)
- decode microbatching (how many sequences are decoded in one backend call)

These knobs interact with:
- learner world size (DDP ranks)
- rollout server world size (number of dedicated inference devices)
- `training.effective_batch_size` (ms-swift derives `gradient_accumulation_steps` via ceil)
- packing scope (`post_rollout_pack_scope`) and Channel-B execution modes (`micro`/`step`/`async`)

As a result, scaling Stage2-AB to heterogeneous topologies (e.g., 6 rollout GPUs + 2 learner GPUs) requires manual tuning and is easy to misconfigure (underutilization, too many small RPCs, or exceeding rollout GPU memory limits).

Constraints:
- Config-first: all behavior MUST be driven by YAML (no new CLI flags).
- Preserve Qwen3-VL chat-template and geometry invariants (no coord drop/reorder; `do_resize=false` remains standard).
- Do not patch upstream HF model files.
- Keep behavior deterministic and paper-reproducible (stable scheduling, stable request ordering, logged metadata).

## Goals / Non-Goals

**Goals:**
- Make `training.effective_batch_size` the single source of truth for **global raw rollouts per optimizer step** in Stage2-AB Channel-B.
- Standardize Stage2-AB Channel-B to a single execution pathway: **step-budgeted learn-to-completion with dynamic packing**.
- Remove all legacy/overlapping rollout batch-size knobs in favor of a single decode batching control with clear semantics (a per-rollout-GPU cap per generation call).
- Enforce a per-rollout-GPU generation-call cap (Stage2-AB YAML default: `4` sequences per call) while still distributing requests evenly across rollout devices.
- Derive rollout request chunk sizing automatically from rollout-server world size and learner world size (no hardcoded 6-2 topology assumptions).
- Reduce end-user tuning surface: no legacy fallback paths; configs either follow the standardized pattern or fail fast with actionable guidance.

**Non-Goals:**
- Introducing new rollout backends or changing decode algorithms.
- Changing geometry tokenization / coord vocab / coordinate semantics.
- Reworking ms-swift’s `effective_batch_size -> gradient_accumulation_steps` derivation logic.
- Guaranteeing that *data consumption* per optimizer step always equals the requested `effective_batch_size` in all edge cases (e.g., non-divisible values under ceil-derived accumulation); Stage2-AB will prioritize honoring the rollout budget contract.

## Decisions

0) **Single Channel-B execution pathway (remove micro/async modes)**

- Stage2-AB Channel-B MUST execute in the step-budgeted learn-to-completion pathway.
- `stage2_ab.channel_b.mode` is removed from the supported config surface.
  - If a config provides `stage2_ab.channel_b.mode` (any value), the trainer MUST fail fast with migration guidance.

Rationale:
- Supporting multiple execution modes creates a combinatorial config surface and makes throughput tuning non-reproducible.
- The core performance goal (high rollout throughput with bounded memory) is achieved by step-budgeted execution + derived decode batching; cross-step async queues are not required.

1) **Single source of truth for Channel-B raw rollout budgeting**

- The global number of raw rollouts collected for one Channel-B optimizer step MUST equal:
  - `rollouts_per_step := training.effective_batch_size`.
- `stage2_ab.channel_b.rollouts_per_step` is removed from the supported config surface.
  - If a config provides `stage2_ab.channel_b.rollouts_per_step` (any value), the trainer MUST fail fast with migration guidance.
- Stage2-AB training MUST require `training.effective_batch_size` to be divisible by:
  - `training.per_device_train_batch_size × learner_world_size`,
  so that `training.gradient_accumulation_steps` is an exact integer (no ceil overshoot).

Rationale:
- Users reason about throughput and learner demand in terms of “how many raw rollouts are needed per optimizer update”.
- The ceil-derived accumulation “realized batch size” is an implementation detail and can silently overshoot. Defaults should respect user intent.

2) **Per-rollout-GPU decode cap as the primary rollout batching control**

Define a single semantic target:
- `custom.extra.rollout_matching.decode_batch_size`: maximum number of sequences decoded per rollout GPU in one backend generation call.
- Stage2-AB YAML default under `configs/stage2_ab/**`: `4` (user constraint; safe for memory-bounded rollout GPUs).
- Code fallback when unset: `1` (conservative; avoids unexpected OOM on small GPUs).

Legacy knobs are removed (no fallback):
- If a config provides any of:
  - `custom.extra.rollout_matching.rollout_generate_batch_size`
  - `custom.extra.rollout_matching.rollout_infer_batch_size`
  - `stage2_ab.channel_b.rollout_decode_batch_size`
  the trainer MUST fail fast with guidance to use `custom.extra.rollout_matching.decode_batch_size` instead.

3) **Derived request chunk sizing from rollout and learner world sizes**

To respect the cap without hardcoding topology:
- Query each configured vLLM server’s `${base_url}/get_world_size/` and cache one `server_world_size` per server entry.
- Let `server_world_sizes = [s_0, s_1, ...]` be those values, and let:
  - `S = sum(server_world_sizes)` (total rollout inference device count across servers; DP replicas)
  - `W = learner_world_size` (training DDP world size)
- **Feasibility (normative)**: the cap can only be preserved under DDP if:
  - `decode_batch_size * S >= W`
  If not, the trainer MUST fail fast with actionable guidance (otherwise every learner rank issuing at least one request would exceed the global cap).
- Choose a per-learner-rank request chunk size:
  - `chunk = floor(decode_batch_size * S / W)` (this is guaranteed to be `>= 1` under the feasibility rule above).

Backend notes:
- HF backend and vLLM colocate mode: generation happens locally per learner rank/device, so the per-call decode batch size is simply `decode_batch_size` (no world-size derivation).
- vLLM server mode: server request distribution MUST be capacity-aware (proportional to `server_world_sizes`) so that, when all learner ranks generate concurrently, each rollout GPU sees bounded decode work.

4) **Default packing scope aligned with dynamic-per-step training**

- Stage2-AB Channel-B uses dynamic-per-step packing semantics.
- Therefore, post-rollout packing is standardized to **micro-scope dynamic packing** (pack based on completion availability within the step, not a fixed gradient-accumulation window).
- `custom.extra.rollout_matching.post_rollout_pack_scope` is removed from the supported config surface (no fallback).
  - If a config provides `post_rollout_pack_scope` (any value), the trainer MUST fail fast with migration guidance.

5) **Determinism and observability**

- Rollout budgeting and rank-local shares MUST remain deterministic given:
  - `rollouts_per_step`, learner world size, and global step.
- Derived decode/request chunk parameters MUST be logged once (and again if they change due to dynamic server discovery).
- Server request distribution MUST remain deterministic (stable contiguous chunking, weighted by server world sizes when servers are heterogeneous).

## Risks / Trade-offs

- [Breaking config surface] → Mitigation: configs under `configs/stage2_ab/**` are updated in-tree; other configs must migrate by following the standardized pattern (fail-fast errors include exact replacement keys).
- [Non-divisible effective_batch_size under ceil-derived GAS] → Mitigation: fail fast with actionable guidance to choose a divisible `effective_batch_size` (required for paper-reproducible step semantics).
- [World-size discovery failures in server mode] → Mitigation: fail fast if `/get_world_size/` is unreachable (server must be healthy before training starts).
- [Uneven rollout completion latency leads to temporary queue imbalance] → Mitigation: keep bounded producer/consumer queue; derived chunk sizing limits peak outstanding work.
