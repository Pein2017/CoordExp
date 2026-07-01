# Change: Refactor Stage-2 Async Actor-Learner (Multi-GPU Learner + vLLM Server)

## Why
Stage-2 AB training currently supports vLLM server mode for rollouts, but it is restricted to a single-process learner (`world_size == 1`).
In practice, long-context multimodal rollouts and training cannot fit on the same GPU without OOM, so we need a **hard role split**:
- dedicated GPUs for rollout only (vLLM server), and
- dedicated GPUs for learning only (multi-GPU learner with DDP).

We also want to complete the `progress/full_idea.md` migration to an **async actor-learner** model where rollouts are generated and preprocessed ahead of time and consumed by the learner with bounded staleness.

## What Changes
- Enable `custom.extra.rollout_matching.vllm.mode: server` to work with a **multi-process learner** (`torchrun`, `world_size > 1`) via:
  - rank0-only full weight sync to the server, and
  - safe multi-rank rollout inference (no deadlocks, no duplicated weight sync).
- Add a Stage-2 AB async actor-learner path that:
  - prefetches rollouts + preprocessing into bounded queues,
  - uses a **sync-counter version** (`ver`) for freshness gating (`version_window`), and
  - keeps Channel-B as a cold path governed by `stage2_ab.schedule.b_ratio` (policy) and queue availability (feasibility gate).
- Enforce **DDP-safe Channel-B execution semantics** for multi-GPU learners:
  - exactly **one packed forward/backward per micro-step per rank** (no inner multi-pack loops that can desynchronize DDP collectives).
- Support rollout sampling for vLLM server mode (optional):
  - `temperature > 0` triggers sampling; add `top_p` (and optionally `top_k`) knobs.
  - Robustness is prioritized over strict bitwise reproducibility when retries / batch-splitting happen.
- **BREAKING (config):** rollout decoding knobs move to `custom.extra.rollout_matching.decoding.*`.
  - Legacy keys like `custom.extra.rollout_matching.temperature` are removed and MUST fail fast.
- **BREAKING (config):** remove `custom.extra.rollout_matching.rollout_buffer` (buffered reuse of prepared windows).
  - The new Stage-2 pipeline is explicitly async actor-learner; reuse via `rollout_buffer` is not part of the intended workflow and complicates staleness/versioning.
  - Legacy `rollout_buffer.*` keys are removed and MUST fail fast.
- Update launch scripts so a single entrypoint can run:
  - vLLM rollout server on GPUs `0-5`, and
  - learner `torchrun` on GPUs `6-7`.

Non-goals (v1):
- Do not introduce “actor ranks” inside the learner `torchrun` world (i.e. do not split one torchrun job into learner ranks + rollout ranks).
  Rollouts remain served by an external vLLM server process.

## Impact
Affected specs:
- `openspec/specs/rollout-matching-sft/spec.md` (server mode learner world-size constraint, determinism notes, decoding knobs)
- `openspec/specs/stage2-ab-training/spec.md` (async Channel-B mode, queue feasibility gate, rank0 broadcast, DDP safety guardrails)

Affected code (expected):
- `src/trainers/rollout_matching_sft.py` (vLLM server client init/sync/infer under DDP; decoding knobs)
- `src/trainers/stage2_ab_training.py` (async queue path; DDP-safe B execution semantics; scheduling gate)
- `src/config/schema.py` (new Stage2-AB async config knobs)
- `docs/training/STAGE2_RUNBOOK.md` and `docs/training/METRICS_LOSSES.md` (new behavior + new metrics)
- `scripts/stage2_ab_server_train.sh` (allow multi-GPU learner)
