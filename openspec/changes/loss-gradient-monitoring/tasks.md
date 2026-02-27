## 1. Config Surface + Injection (YAML-First)

- [ ] 1.1 Add `custom.extra.loss_gradient_monitor` parsing/injection in `src/sft.py`:
  - Read a mapping from `custom.extra["loss_gradient_monitor"]` when present.
  - Attach it to every trainer instance as `trainer.loss_gradient_monitor_cfg` (Stage-1 and Stage-2 variants).
  - Log a single init line on rank0 when enabled (interval, param selector strategy).

- [ ] 1.2 Keep schema changes minimal:
  - Do not add new CLI flags.
  - Keep the config under `custom.extra` (freeform mapping), but validate required fields at runtime in the monitor (fail closed: warn-once + disable monitor, do not crash training).

## 2. Implement The Monitoring Component

- [ ] 2.1 Add `src/trainers/monitoring/loss_gradient_monitor.py` implementing `LossGradientMonitor`:
  - Inputs:
    - `loss_terms: Mapping[str, torch.Tensor]` (additive scalars before summation; require-grad),
    - `model` (possibly DDP-wrapped),
    - `trainer` (for `state.global_step`, rank detection, reporter access).
  - Outputs:
    - metrics dict with `gradmon/*` keys (floats) to be merged into existing log payloads.
  - Core computations (only on monitor steps):
    - EMA of per-term abs(loss) and `loss_ema_norm`,
    - per-term gradient norms on a shared parameter probe block,
    - `max(norm)/median(norm)` ratio,
    - negative cosine pair fraction/percent.
    - per-term cosine similarity to net coord update direction:
      - `cos_to_total(term) = cos(g_term, g_total)` where `g_total = Σ g_term`.
  - Performance/compat:
    - only run every `interval_steps` optimizer steps (default 100),
    - only run on last gradient-accumulation micro-step when available (`accelerator.sync_gradients`) when `require_sync_gradients=true`,
    - wrap extra autograd grad calls in `model.no_sync()` when available (DDP) to avoid extra allreduces,
    - cast grad tensors to `float32` for norm/dot,
    - never write into `.grad`, never call optimizer hooks, never change the returned training loss.
  - Reliability:
    - best-effort: unexpected exceptions warn once and disable monitor for the rest of the run.

- [ ] 2.2 Optional: export `LossGradientMonitor` from `src/trainers/monitoring/__init__.py`.

## 3. Stage Integrations (Minimal Changes)

- [ ] 3.1 Stage-1 integration (SFT path):
  - Hook point: `CoordSoftCEW1LossMixin._maybe_add_coord_softce_w1_loss` in `src/trainers/metrics/mixins.py`.
  - Build `loss_terms` dict *before* summation:
    - Coord-only atoms from `CoordSoftCEW1Result`:
      - `S1/coord_soft_ce`, `S1/coord_w1`, optional `S1/coord_ce`, optional `S1/coord_gate`.
    - Exclude base text CE (monitor is coord-only by requirement).
  - Call monitor only when enabled; merge returned metrics into the existing reporter flow.

- [ ] 3.2 Stage-2 rollout-aligned integration:
  - Hook point: `RolloutMatchingSFTTrainer.compute_loss` in `src/trainers/stage2_rollout_aligned.py`.
  - Term discovery:
    - derive **atomic coord/geo tensors** from `pipeline_result.state` (module-provided contrib tensors):
      - bbox geo atoms: `bbox_smoothl1_contrib`, `bbox_ciou_contrib`
      - coord reg atoms: `coord_token_ce_contrib`, `coord_soft_ce_contrib`, `coord_w1_contrib`,
        `coord_el1_contrib`, `coord_ehuber_contrib`, `coord_entropy_contrib`, `coord_gate_contrib`
    - multiply each atom by the corresponding objective module `spec.weight` so the monitored term matches the additive contribution in `total_loss`,
    - name terms under Stage-2 provenance keys, e.g. `B_coord/bbox_smoothl1`, `B_coord/coord_soft_ce`, etc,
    - exclude `token_ce` atoms and exclude `text_gate_contrib` (coord-only).
  - Logging:
    - add monitor metrics into the existing pending micro-batch buffer so they are logged once per optimizer step.

- [ ] 3.3 Stage-2 two-channel integration:
  - Hook point: `Stage2ABTrainingTrainer.compute_loss` in `src/trainers/stage2_two_channel.py`.
  - Term discovery:
    - Channel-B step: same as rollout-aligned (`B_coord/*` atomic coord/geo terms from `pipeline_ctx.state`, module-weighted),
    - Channel-A step:
      - **atomic provenance split**:
        - `A2_coord/*` atoms from the self-context pathway (`pipeline_ctx.state`, module-weighted),
        - `A1_coord/*` atoms from the optional A1 anchor pathway (the explicit `run_bbox_geo_module` / `run_coord_reg_module` calls), weighted by the trainer’s module-weight multipliers and the anchor sub-weights already baked into contrib tensors.
      - exclude all text CE atoms; exclude `text_gate_contrib` (coord-only).
  - Logging:
    - add monitor metrics into the Stage-2 pending log buffer (`_PendingStage2Log`) as mean-like keys.

- [ ] 3.4 Guardrails:
  - Ensure monitor logic never touches rollout generation/parsing/matching code paths.
  - Ensure all ranks take the same “monitor or skip” branch for a given optimizer step to avoid DDP mismatches.

## 4. Logging Contract + Docs

- [ ] 4.1 Update `docs/training/METRICS_LOSSES.md`:
  - Document `gradmon/*` keys, their meanings, and sparse emission semantics.
  - Include a short enable/disable snippet for `custom.extra.loss_gradient_monitor`.

- [ ] 4.2 Update `openspec/specs/trainer-metrics-components/spec.md`:
  - Register `gradmon/*` as an optional diagnostics surface with best-effort semantics.
  - Clarify aggregation semantics: mean-like, sparse-emitted, no per-dataset buckets.

## 5. Verification (Unit Tests + Smoke)

- [ ] 5.1 Add unit tests for monitor math on a tiny toy model (CPU is fine):
  - Construct two loss terms with known gradient directions on a shared parameter vector:
    - expect `neg_cosine_pair_frac` to be `1.0` when gradients oppose,
    - expect `0.0` when gradients align.
  - Verify `cos_to_total` signs match expectations (e.g., one term negative when it opposes the summed direction).
  - Verify gradient norms are positive and stable.

- [ ] 5.2 Add a unit test that enabling the monitor does not change the objective:
  - Run one training-step forward/backward with monitor off vs on,
  - Assert the returned scalar loss is identical (or within tight tolerance),
  - Assert parameter grads (after real backward) match within tolerance.

- [ ] 5.3 Add a Stage-2-focused unit test for term discovery:
  - Verify that the monitor sees exactly the additive terms that are summed into `total`
    for both Channel-A and Channel-B steps (coord-only atomic terms with provenance split: `A1_coord/*`, `A2_coord/*`, `B_coord/*`).

- [ ] 5.4 Add smoke YAML configs enabling the monitor with `interval_steps: 1`:
  - One Stage-1 smoke under `configs/stage1/smoke/`,
  - One Stage-2 two-channel smoke under `configs/stage2_two_channel/smoke/`,
  - Optional: rollout-aligned smoke under `configs/stage2_rollout_aligned/smoke/` (if available in the repo).
  - Run for a few optimizer steps and verify `gradmon/*` keys appear.

Validation commands (examples):
- `conda run -n ms python -m pytest -q tests/test_loss_gradient_monitor.py`
- `conda run -n ms python -m pytest -q tests/test_stage2_ab_training.py`
- `conda run -n ms python -m pytest -q tests/test_stage2_rollout_aligned.py`
