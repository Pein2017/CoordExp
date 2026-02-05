# Design: Stage-2 Async Actor-Learner (Server Rollouts, Multi-GPU Learner)

## Scope / Topology (v1)
We target a strict role split for memory safety:
- **Rollout server:** a separate `swift rollout` process running vLLM on dedicated GPUs (e.g. GPUs `0-5`).
- **Learner:** `torchrun` training job on dedicated GPUs (e.g. GPUs `6-7`, `world_size=2`).

There are no rollout “actor ranks” inside the learner torchrun world in v1.

## Key Constraints
- vLLM server weight sync MUST use `sync.mode: full`.
- Stage2-AB Channel-B under multi-GPU learner MUST be DDP-safe:
  - exactly one packed forward/backward per micro-step per rank
  - no per-rank variation in collective sync behavior inside a micro-step
- Scheduling policy remains YAML-driven via `stage2_ab.schedule.b_ratio`.
  Queue availability acts as a feasibility gate (no mid-accumulation switching).
- Sampling is allowed (e.g. `temperature=0.01`, `top_p=0.95`) with robustness prioritized over strict determinism.
- The legacy `custom.extra.rollout_matching.rollout_buffer` reuse mechanism is removed in this refactor.
  Async prefetch + versioned queues are the intended throughput mechanism in Stage-2.

## High-Level Data Flow

### Channel-A (hot path)
Unchanged: dataloader -> prepare A inputs -> forward/backward on learner ranks.

### Channel-B (cold path, async)
Each learner rank runs a local prefetch pipeline that produces “ready packs”:

1) **Sample source** (per rank):
- Uses the rank’s own dataloader shard (DistributedSampler).
- Produces raw samples (`messages`, `images`, GT objects metadata).

2) **Rollout generation** (server):
- Each rank issues requests to the rollout server over HTTP.
- Rank0 is responsible for **full weight sync** and advancing `sync_counter` (`ver`).
- Weight sync ordering is strict (DDP-safe, no in-flight infer during sync):
  - `dist.barrier()` -> rank0 sync -> `dist.barrier()` -> infer
- If async prefetch workers issue `/infer/` requests, they must participate in the sync fence:
  - `/infer/` issuance must be gated so no background worker can be mid-`/infer/` during the sync critical section.

3) **Preprocess** (CPU):
- strict parse rollout -> predicted objects
- Hungarian matching + gating
- construct `y_train_ids` (reordered GT with FN append)
- build loss masks/meta

4) **Teacher-forced encode** (CPU):
- `template.encode(... return_length=True)` to get collator-ready per-sample encodings and `encoded_len`.

5) **Packing** (CPU):
- Build one packed micro-batch per micro-step, targeting `packing_length` and respecting `packing_min_fill_ratio`.
- This produces exactly **one packed batch dict** per queue item.

6) **Learner consumption**:
- On B micro-steps, training pops exactly one packed batch from the ready queue and runs one forward/backward.

## Scheduling (Policy vs Feasibility)
We use two gates:
1) **Policy gate:** Bresenham-style `b_ratio` schedule decides “wants B” at optimizer-step granularity.
2) **Feasibility gate:** require that all ranks have at least `gradient_accumulation_steps` ready packs available
   (or a stricter token-volume threshold) before committing to a B optimizer step.

If policy wants B but feasibility fails, we run A for that optimizer step and log:
- `stage2_ab/async/b_step_skipped_due_to_queue = 1`

## Freshness / Versioning
- Maintain a monotonic `sync_counter` (`ver`).
- Each rollout/pack is tagged with `ver`.
- Learner consumes only if `ver >= current_ver - version_window`.
- If a ready item is stale, drop it and increment a counter.
- Enforce `queue_limit` by dropping oldest first.

## Data Sourcing / Accounting (Async Channel-B)
Async Channel-B prefetch uses an **independent** sample stream (separate from Channel-A’s main training dataloader stream):
- Channel-A always consumes the main ms-swift training dataloader.
- Channel-B async prefetch consumes a separate iterator (same dataset distribution, different stream).

Implications:
- `stage2_ab.schedule.b_ratio` is a **step-level policy knob** (how often we *want* B), not a guarantee that Channel-A and Channel-B see the same underlying samples.
- We must log enough counters to make the effective data mix auditable (e.g., executed B-step ratio, queue skip counts, stale drops).

## Why Per-Rank Ready Queues (Not Rank0 “Send Packs”)
We intentionally keep data-plane local to each learner rank:
- avoids transmitting large packed batches cross-rank
- preserves standard DDP dataset sharding semantics
- reduces single-rank bottlenecks (rank0 remains the control plane only)

If we later need globally optimal fill across ranks, we can add a “pack-plan broadcast” optimization,
but v1 favors correctness and throughput stability.

## Decoding Knobs (Sampling)
Sampling is controlled by:
- `temperature` (sampling if `temperature > 0`; recommended starting point `0.01`)
- `top_p` (default `1.0`; recommended starting point `0.95`)
- optional `top_k` (default `-1`)

Robustness is preferred:
- on server infer failures/timeouts, split batches and retry
- this may perturb sampling outcomes due to changed request chunking; seed and decoding params must be logged.
