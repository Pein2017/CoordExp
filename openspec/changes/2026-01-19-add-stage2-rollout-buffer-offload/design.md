## Context
Stage_2 rollout-matching SFT is implemented as a two-phase pipeline:

raw samples
  -> (A) rollout generation (no grad)
  -> parse / token-aligned object extraction
  -> Hungarian matching
  -> build canonical teacher-forced target `Y_train` (rollout prefix + FN append)
  -> (B) teacher-forced forward + masked CE + coord losses (with optional post-rollout packing)

This is correctness-aligned but can be throughput- and memory-inefficient:
- Rollout generation time dominates step time and is highly variable (rank stragglers).
- vLLM colocate rollouts share GPU memory with training activations/optimizer state and can OOM on peaks.
- When rollouts are slow, the job performs very few optimizer steps, so improvement signals are noisy.

ms-swift's GRPO implementation uses two patterns that can be reused without adopting GRPO loss:
1) Offload context around rollout inference (to reduce peak colocate memory).
2) Buffering rollout results across multiple training updates (to amortize generation cost).

This change keeps CoordExp's stage_2 algorithm (token-aligned parse + Hungarian + append + coord supervision),
but changes *scheduling* and *memory policy* to be closer to ms-swift RLHF infrastructure robustness.

## Terminology (Trainer step semantics)
To avoid ambiguity around gradient accumulation, this design uses the following terms:

- **micro-step**: one invocation of `Trainer.training_step(...)` on one dataloader batch.
- **optimizer step**: one parameter update step (after all micro-steps in the current gradient accumulation window),
  which increments `TrainerState.global_step`.
- **accumulation window**: the sequence of `gradient_accumulation_steps` micro-steps that contribute gradients to a
  single optimizer step.

## Batch-size units (raw vs packed)
Stage_2 uses two different "batch size" concepts:

- **raw micro-batch size** (`B_raw`): the number of raw dataset samples per micro-step per rank. This is
  `per_device_train_batch_size` for the identity collator (a Python list length).
- **forward batch size** (`B_fwd`): the first dimension of the tensor batch passed into `model(...)` for one micro-step.
  - When post-rollout packing is enabled, `B_fwd = 1` because the trainer packs multiple segments into one padding-free
    packed row.
  - When post-rollout packing is disabled, `B_fwd = B_raw` (standard padded batch).

For effective batch size accounting under packing, we treat each packed row as one atomic "training unit":
- global effective batch size (packed-row units) per optimizer step is `world_size * gradient_accumulation_steps * 1`.
- the number of raw samples/segments per packed row is variable (depends on sequence lengths and packing policy) and
  should be logged as a throughput diagnostic, not treated as the definition of batch size.

## Design Overview
### A) Rollout Buffer (E-step / M-step scheduling)
Add an opt-in rollout buffer under `custom.extra.rollout_matching.rollout_buffer`:

- When enabled, the trainer runs the full rollout+match+encode+pack pipeline once ("E-step") to produce a
  *ready-to-train* set of micro-step batches for one optimizer step (i.e., one accumulation window worth of work).
- The trainer then reuses that prepared accumulation window for `m_steps` optimizer steps ("M-steps") before
  generating a new rollout window.

Key properties:
- No changes to `Y_train` construction semantics.
- No changes to the loss definition; only changes *when* we regenerate rollouts.
- Bounded staleness: reuse window length is explicit and finite (`m_steps`).

Implementation sketch (conceptual):
- Maintain trainer-local state:
  - `rollout_window_step0` (global_step when buffer was created)
  - `rollout_window_used_steps` (how many optimizer steps consumed)
  - `cached_micro_batches` (a list of prepared model batches, one per micro-step in the accumulation window)
  - `cached_micro_step_idx` (which cached micro-batch to use next)
- In `training_step` (called per micro-step):
  - if caching is enabled and `cached_micro_batches` are valid for the current rollout window, train on
    `cached_micro_batches[cached_micro_step_idx]`
  - else build `cached_micro_batches` for the next accumulation window (using the next `gradient_accumulation_steps`
    raw micro-batches), and start a new window

Note: This intentionally mirrors the GRPO "generate once, update multiple times" scheduling, but uses an SFT loss.

Cached batch immutability:
- HF/Swift trainer internals and the stage_2 implementation may mutate input dicts (e.g., via `pop`).
- Therefore cached prepared batches must be treated as read-only; reuse should pass a safe copy into the training stack.

### B) Stage_2 Dataloader Wrapper (repeat accumulation windows)
If the trainer reuses cached work for M-steps, the upstream dataloader must not advance through the dataset and
silently drop samples (or accidentally change the effective micro-batch composition under gradient accumulation).

We therefore add a stage_2-only dataloader wrapper that repeats each *accumulation window* `m_steps` times when
`rollout_buffer.enabled=true` and `m_steps > 1`.

Concretely, on each rank:
- Let `gas = gradient_accumulation_steps`.
- The wrapper groups the underlying dataloader micro-batches into chunks of length `gas`.
- Each chunk is yielded `m_steps` times before advancing to the next chunk.

Example (`gas=3`, `m_steps=2`): the per-rank sequence of micro-batches becomes:
`A,B,C, A,B,C, D,E,F, D,E,F, ...`

This is intentionally similar to GRPO's "generate once, update multiple times" pipeline:
- the same raw data are presented for the reuse window,
- the trainer does not need to discard fresh batches,
- behavior remains deterministic under seeding and DDP.

This wrapper is stage_2-specific and MUST NOT affect stage_1.

Implementation note:
- `src/sft.py` does not directly control dataloader iteration. The wrapper should be implemented via a stage_2-only
  trainer override (e.g., `get_train_dataloader()` returning a wrapped iterator) or a sampler/batch_sampler wrapper.

End-of-epoch / partial windows:
- If the underlying dataloader yields a final partial accumulation window (< `gas` micro-batches), we cannot safely
  "repeat it as optimizer steps" without changing HF Trainer accumulation behavior.
- Therefore the buffered-reuse behavior should apply only to full windows. The final partial window should be processed
  once (no reuse), with a warning suggesting `dataloader_drop_last=true` or `m_steps=1` if the user wants strict reuse.

Eval / predict:
- Buffering should be disabled for eval/predict. Even when enabled in config, evaluation should behave as `m_steps=1`
  (no reuse) to avoid confusing or stale metrics.

### C) Rollout Offload Context (colocate vLLM memory relief)
Add an opt-in offload context under `custom.extra.rollout_matching.offload`:
- `offload_model` (bool): move training model parameters to CPU during rollout inference
- `offload_optimizer` (bool): move optimizer state to CPU during rollout inference

This context is applied only around vLLM colocate rollout inference calls. The goal is to reduce peak GPU memory by
freeing training state while vLLM performs multimodal prefill and KV-cache allocations.

Design constraints:
- Offloading MUST occur only during rollout generation (no grad).
- The trainer MUST restore model/optimizer state to GPU before teacher-forced forward/backprop.
- Offload MUST be opt-in and default to disabled (performance trade-off).
- If offload is requested but cannot be applied safely under the current setup (e.g., incompatible DeepSpeed/ZeRO
  behavior), the trainer should fail fast with actionable guidance rather than silently degrading.
- When vLLM is lazily initialized on the first rollout, offload should cover vLLM engine initialization and LoRA
  adapter loading, not only the infer call, since those are often peak-allocation moments.

## Notes / Compatibility
- Server-mode vLLM rollouts are out-of-scope for this change, but the design keeps the rollout backend boundary
  explicit so server mode can be added later.
- The rollout buffer is compatible with post-rollout packing. In buffered mode:
  - post-rollout packing (and its carry buffer) is updated only during E-steps,
  - cached batches are reused as-is during M-steps (no repacking / no carry-buffer mutation).
- LoRA sync into vLLM remains on `global_step` boundaries; sync SHOULD occur outside the offload context.
- Checkpoint/resume: the rollout buffer is runtime-only state and is not persisted. On resume, the buffer starts empty
  and the next training step regenerates rollouts.
