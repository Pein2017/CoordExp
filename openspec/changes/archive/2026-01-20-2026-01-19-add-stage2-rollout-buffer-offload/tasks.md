## 1. Implementation
- [x] 1.1 Add YAML knobs under `custom.extra.rollout_matching`:
  - [x] `rollout_buffer` (mapping):
    - [x] `enabled` (bool, default false)
    - [x] `m_steps` (int, default 1; number of optimizer steps to reuse the buffered rollout window)
  - [x] `offload` (mapping):
    - [x] `enabled` (bool, default false)
    - [x] `offload_model` (bool, default false)
    - [x] `offload_optimizer` (bool, default false)
- [x] 1.2 Add a stage_2-only "accumulation-window repeater" for the training dataloader when
      `rollout_buffer.enabled=true` and `m_steps > 1`:
      - repeat each *gradient accumulation window* (group of `gradient_accumulation_steps` micro-batches) `m_steps`
        times per rank (e.g., `A,B,C, A,B,C, ...` for `gas=3, m_steps=2`).
      - implement via a trainer override (`get_train_dataloader`) or a sampler/batch_sampler wrapper; avoid changing
        baseline stage_1 behavior.
- [x] 1.3 Update `src/trainers/rollout_matching_sft.py`:
  - [x] Add a rollout buffer that caches one *accumulation window* worth of prepared micro-step batches and reuses it
        across `m_steps` optimizer steps.
        - [x] Ensure cached batches are safe to reuse even if the training stack mutates inputs (copy-on-reuse).
        - [x] Ensure reuse is disabled for eval/predict (no caching when `model.training == False`).
        - [x] Handle final partial accumulation windows by processing once without reuse (warn + suggest mitigations).
  - [x] Add a rollout offload context applied during vLLM colocate rollouts (no-op for HF backend).
        - [x] Ensure offload covers vLLM engine init and LoRA adapter loading/sync (first-step peak).
  - [x] Add lightweight logs for buffer window index / reuse count.
- [x] 1.4 Tests:
  - [x] Verify dataloader wrapper repeats batches deterministically.
  - [x] Verify rollout buffering does not change batch contents vs baseline for the first window step.
  - [x] Verify cached-batch reuse does not fail due to input dict mutation (e.g., keys popped during compute_loss).
  - [x] Verify offload context does not break training_step (smoke-level unit/integration test).

## 2. Docs & Examples
- [x] 2.1 Update `docs/STAGE2_ROLLOUT_MATCHING_RUNBOOK.md` with rollout_buffer/offload knobs and recommended settings.
- [x] 2.2 Add/modify a stage_2 config example under `configs/dlora/` demonstrating buffered rollouts.

## 3. Validation
- [x] 3.1 Run `openspec validate 2026-01-19-add-stage2-rollout-buffer-offload --strict`.
NOTE: The following items are operational benchmarking guidance (not OpenSpec-gated tasks) and may require
cluster resources and dataset access beyond this repo.

- 3.2 Run a small monitor run (e.g., 128/32) and record steps/sec and rollout time split with and without buffering.
      - In buffered mode, ensure logs/metrics distinguish E-steps vs M-steps:
            - rollout generation timing SHOULD be logged as 0 (or omitted) on M-steps to avoid double-counting
            - add a boolean flag/counter (e.g., `rollout/buffer_reuse=1`) to make dashboards interpretable.
