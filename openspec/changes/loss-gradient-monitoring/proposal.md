## Why

Multi-loss training in CoordExp (Stage-1 Scheme A and Stage-2 AB / rollout-aligned) can become unstable: sudden loss spikes, oscillations, or divergence. Scalar loss logs alone are not enough to diagnose *why* this happens.

For multi-loss objectives, the common failure modes are:
- **(A) Gradient domination:** some loss terms contribute much larger gradient norms and effectively overwhelm other supervision signals.
- **(B) Gradient conflict:** loss gradients frequently point in opposing directions (negative cosine similarity), causing optimization to "fight itself" even when scalar losses look reasonable.

We need a lightweight, opt-in, monitoring-only tool to measure domination vs conflict **without changing any optimization logic, loss formulation, rollout logic, or Hungarian matching**.

## What Changes

- Add a monitoring-only component `LossGradientMonitor` that, every **N optimizer steps** (configurable, default `100`), logs:
  - raw per-term loss values (each term kept separate before summation),
  - EMA-normalized loss values,
  - per-term gradient norm on a shared parameter probe block: `||∇θ_shared Li||_2`,
  - `max(norm) / median(norm)` across terms,
  - fraction / percentage of **negative pairwise cosine similarities** across per-term gradients.
  - per-term cosine similarity to the net coord update direction: `cos(∇Li, ∇L_total)` ("cos_to_total").

- **Automatic loss-term discovery (before summation):**
  - **Stage-2 (atomic, coord-only):**
    - For Channel-A monitoring, losses are tracked at the **atomic level** (A1_* vs A2_*), using per-atom tensors produced by the objective modules (`bbox_geo` + `coord_reg`) and weighted exactly as in the summed objective.
    - Only **coordinate-token supervision terms** are included (coord/geo). Text CE terms are excluded.
  - **Stage-1 (coord-only):**
    - Monitor only coordinate-token supervision terms from `CoordSoftCEW1Result` (softCE, W1, gate, optional coord-CE).
  - **Packed-sequence contract:**
    - The monitor reuses the additive scalar tensors already produced by the active loss helpers / objective modules.
    - It does **not** introduce a second packed-token position gatherer or reconstruct supervision spans from `input_ids` after the fact.
    - For Stage-2 packed trainers, monitor values are computed on each participating packed forward and then aggregated into the optimizer-step log using the existing per-step metric buffers / reducers.

- Integrate into training loops:
  - Stage-1 (SFT path): integrate into the existing metrics/loss mixin stack without refactoring trainers.
  - Stage-2 two-channel (`custom.trainer_variant: stage2_two_channel`): integrate into its packed-forward `compute_loss` path, then feed the existing `_PendingStage2Log` step reducer.
  - Stage-2 rollout-aligned (`custom.trainer_variant: stage2_rollout_aligned`): integrate into its packed-forward `compute_loss` path, then feed the existing `PendingTrainRolloutLog` step reducer.

- Configuration and activation:
  - Config-first enable/disable via `custom.extra.loss_gradient_monitor` (disabled by default).
  - No new CLI flags.

## Capabilities

### New Capabilities
- `loss-gradient-monitoring`: diagnose multi-loss instability by measuring per-loss gradient domination and gradient conflict.

### Modified Capabilities
- `trainer-metrics-components`: extend the metrics surface with optional `gradmon/*` diagnostics keys (best-effort, sparse-emitted).
- `stage2-ab-training` and `rollout-matching-sft`: add monitoring hooks that do not interfere with rollout generation/parsing/matching.
- Stage-1 training pipeline: add monitoring without changing the objective.

## Impact

- Default behavior is unchanged (monitor is off by default).
- Opt-in runs add minor overhead only on monitor steps (default: once per 100 optimizer steps).
- Adds new diagnostic metric keys under `gradmon/*` and a short enablement section in [docs/training/METRICS.md](../../docs/training/METRICS.md).
- Under DDP, monitor metrics are computed locally on each rank first, buffered locally for the step, and then synchronized at the same optimizer-step reducer boundary used by the current training metrics.
