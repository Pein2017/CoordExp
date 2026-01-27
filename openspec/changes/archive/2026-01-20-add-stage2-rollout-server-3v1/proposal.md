# Change: 2026-01-20-add-stage2-rollout-server-3v1

## Why
Stage_2 rollout-matching (`custom.trainer_variant: rollout_matching_sft`) is currently fastest with vLLM rollouts, but the default integration is **colocate**: a vLLM inference engine lives on the same GPU(s) as the training model used for the teacher-forced forward/backward pass.

With long-context multimodal runs (e.g. `global_max_length: 16000` and large `max_new_tokens`), colocate mode can become unstable and frequently OOMs because GPU memory must simultaneously hold:
- training model weights + activations + optimizer state (esp. with long sequences), and
- a second inference model instance + a large vLLM KV cache reservation.

We want a reproducible, EM-style loop that stabilizes memory by separating rollout generation from SFT training:
- E-step: generate rollouts at high throughput using vLLM with dedicated GPU memory
- M-step: perform teacher-forced SFT (matching/masking) on packed sequences and update parameters
- repeat using the latest updated parameters

We also want **fast parameter synchronization** across phases without offload/reload from disk, and we must keep DoRA ("dlora") as the tuning method.

## What Changes
- Add an optional **vLLM server** rollout mode for stage_2 rollout-matching so that rollouts can run on a dedicated GPU subset (e.g. 3 GPUs) while SFT training runs on a different GPU (e.g. 1 GPU).
- Define a config-driven **3v1 actor/learner topology**:
  - 3 GPUs: vLLM rollout server (`swift rollout` or compatible ms-swift vLLM server)
  - 1 GPU: stage_2 learner (`python -m src.sft ...`) doing parse/match/packing + teacher-forcing SFT
- Use **in-memory weight sync** between learner and rollout server (no checkpoint reloading):
  - default: full merged weights sync (GRPO-style; robust for multimodal + DoRA)
  - optional: adapter-only sync when vLLM LoRA is enabled and compatible
- Keep the stage_2 math unchanged: strict parsing, Hungarian matching, mandatory FN-append target construction, and post-rollout packing remain on the learner.

This proposal is intentionally config-first and avoids new CLI hyperparameter flags.

## Scope
In scope:
- Stage_2 trainer variant `rollout_matching_sft` only.
- Rollout backend `vllm` gains a `mode: server` option (colocate remains the default).
- Deterministic EM-like iteration boundaries and explicit weight-sync points.
- Compatibility with:
  - Qwen3-VL family and existing coord-token/vision features
  - DoRA (dlora) tuning
  - existing rollout_buffer (E-step generate, M-step reuse)
  - existing post-rollout packing (micro/window scheduling)

Out of scope (for this change):
- Asynchronous pipelining (generate next window while training current window).
- Cross-rank global packing across multiple training GPUs (token-balanced scheduling across ranks).
- New loss functions, new matching policies, or new parsing behavior.

## Impact
Expected improvements:
- Eliminates colocate vLLM + training VRAM contention, reducing OOM instability.
- Allows higher vLLM `gpu_memory_utilization` on rollout GPUs to maximize throughput.
- Enables EM-style training where rollouts always come from the latest (or explicitly buffered) parameters.

Risks:
- Server connectivity / NCCL communicator issues can break training.
- Full-weight sync cost can become non-trivial if synced too frequently.
- vLLM LoRA sync may be unstable for multimodal/DoRA (hence full-sync default).

Mitigations:
- Keep `rollout_buffer.m_steps > 1` as the recommended throughput knob to reduce rollout+sync frequency.
- Provide fail-fast validation and clear fallbacks (`rollout_backend: hf` or `vllm.mode: colocate`).

## Rollout/Experiment Plan
- Add a new config template for 3v1 server mode (based on `configs/dlora/stage2_rollout_matching_ckpt3106.yaml`).
- Smoke test (small sample limit) and compare against colocate:
  - stability (no OOM)
  - rollout tokens/s (server)
  - step time and packing fill ratio (learner)
  - rollout quality metrics (f1/recall/truncation) on E-steps only when buffering is enabled.
- Use greedy decoding (`temperature: 0.0`) to minimize nondeterminism and simplify debugging.
