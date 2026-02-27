## Context

CoordExp has two distinct multi-loss training pipelines that can exhibit instability:

1. **Stage-1 SFT (Scheme A)** mixes:
   - base next-token CE on **non-coord** tokens (coord slots are masked out of CE), and
   - additional coord supervision terms (softCE + W1 + optional gates) computed from the model logits.

   Relevant code handles:
   - Stage-1 loss composition lives in [`src/trainers/metrics/mixins.py`](src/trainers/metrics/mixins.py):
     - `CoordSoftCEW1LossMixin.compute_loss` (wraps base CE, masks coord labels)
     - `CoordSoftCEW1LossMixin._maybe_add_coord_softce_w1_loss` (adds coord loss term)
   - Coord multi-term construction lives in [`src/trainers/losses/coord_soft_ce_w1.py`](src/trainers/losses/coord_soft_ce_w1.py):
     - `compute_coord_soft_ce_w1_loss(...) -> CoordSoftCEW1Result`
       which exposes additive terms:
       `softce_contrib`, `w1_contrib`, `ce_contrib`, `gate_contrib`, and `coord_loss = sum(contribs)`.

2. **Stage-2 teacher-forcing pipelines** (two-channel AB and rollout-aligned) combine multiple objective modules:
   - `token_ce`, `bbox_geo`, `coord_reg` (and optional A1 anchor terms in Stage-2 two-channel).

   Relevant code handles:
   - Teacher-forcing objective composition lives in [`src/trainers/teacher_forcing/objective_pipeline.py`](src/trainers/teacher_forcing/objective_pipeline.py):
     - `run_teacher_forcing_pipeline(...) -> PipelineResult(total_loss, module_losses, metrics, state)`
     - Critically, the objective modules already expose **per-atom tensors** in `ModuleResult.state` (e.g. `bbox_smoothl1_contrib`, `coord_soft_ce_contrib`) that can be used to build atomic loss terms before summation.
   - Stage-2 rollout-aligned integrates the pipeline in [`src/trainers/stage2_rollout_aligned.py`](src/trainers/stage2_rollout_aligned.py):
     - `RolloutMatchingSFTTrainer.compute_loss` computes one teacher-forced forward and returns `pipeline_result.total_loss`.
   - Stage-2 two-channel integrates the pipeline in [`src/trainers/stage2_two_channel.py`](src/trainers/stage2_two_channel.py):
     - `Stage2ABTrainingTrainer.compute_loss` runs:
       - `pipeline_ctx` for Channel-A self-context or Channel-B rollout-context, and
       - `pipeline_gt` for Channel-A GT-anchor token CE,
       - plus optional A1 anchor terms (`a1_bbox_obj`, `a1_coord_obj`),
       - then forms `total = pipeline_gt.total_loss + pipeline_ctx.total_loss + a1_bbox_obj + a1_coord_obj`.

Existing metrics already log per-term scalar values (`loss/<provenance>/<atom>` for Stage-2, and Stage-1 coord monitors), but do not diagnose **gradient domination** or **gradient conflict**.

This change adds a monitoring-only module that uses autograd to compute per-loss gradient norms and gradient cosine similarity statistics on a small shared parameter probe block.

Constraints:
- Config-first (YAML). No new CLI flags.
- Must not change objective or optimizer behavior.
- Must work under AMP and DDP.
- Must not interfere with rollout parsing, matching, or masking semantics.

## Goals / Non-Goals

**Goals**
- Automatically discover the set of additive loss terms `Li` *before summation* for:
  - Stage-1 (**coord-token supervision terms only**; exclude text CE),
  - Stage-2 rollout-aligned (**coord/geo atoms only**, derived from pipeline/module atom tensors),
  - Stage-2 two-channel (**atomic coord/geo terms** with Channel-A provenance split `A1_*` vs `A2_*`).
- Compute, on a shared parameter probe block `θ_shared`:
  - per-term gradient norm `||∇θ_shared Li||_2`,
  - percentage of negative cosine similarity across all loss-pairs.
- Log every `interval_steps` optimizer steps (default `100`), using the existing reporting/logging stack.
- Keep overhead small by:
  - activating only every N steps,
  - probing only a small, output-adjacent parameter block by default.

**Non-Goals**
- No dynamic reweighting, gradient surgery (PCGrad), or optimizer changes.
- No changes to loss formulations, masking, or Hungarian matching.
- No attempt to compute full-model per-loss gradients.
- No refactor of the trainer architecture.

## Decisions

### 1) Define “Loss Terms” as Atomic Coord/Geo Contributions (Coord-Only)

We define the monitored terms `Li` as the **atomic, additive, coordinate-token supervision contributions** that are summed (possibly after weighting) into the trainer’s total loss for the step.

“Coord-only” means:
- include coordinate-token-position losses:
  - bbox geometry: SmoothL1 / CIoU atoms (derived from coord token logits),
  - coord distribution regularizers: softCE/W1/coord-token-CE/etc (computed on coord token positions),
- exclude text-token losses:
  - `token_ce` struct/desc CE atoms,
  - `text_gate` (operates on text positions by construction).

Stage-specific term discovery:
- **Stage-2 (rollout-aligned):**
  - Terms come from atomic tensors in `pipeline_result.state`, sourced from:
    - `bbox_geo`: `bbox_smoothl1_contrib`, `bbox_ciou_contrib`
    - `coord_reg`: `coord_token_ce_contrib`, `coord_soft_ce_contrib`, `coord_w1_contrib`,
      `coord_el1_contrib`, `coord_ehuber_contrib`, `coord_entropy_contrib`, `coord_gate_contrib`
  - Each atom tensor is multiplied by the effective module weight (the objective spec’s `weight`) so the monitored `Li` matches the additive term used in `total_loss`.
  - Term names follow Stage-2 provenance conventions (example): `B_coord/bbox_smoothl1`, `B_coord/coord_soft_ce`, etc.
- **Stage-2 (two-channel):**
  - Channel-B step: same as rollout-aligned (atomic `B_coord/*` atoms from `pipeline_ctx.state`).
  - Channel-A step: atomic coord/geo terms are provenance-split:
    - `A2_coord/*`: atoms from the self-context forward (`pipeline_ctx.state`) with module-weighting applied.
    - `A1_coord/*`: optional atoms from the A1 anchor pathway when enabled (the explicit `run_bbox_geo_module` / `run_coord_reg_module` calls used to form `a1_bbox_obj` / `a1_coord_obj`), with the trainer’s module-weight multipliers applied.
  - Only atoms that actually participate in the summed scalar `total` are included (when the corresponding effective weight is non-zero and the atom tensor exists).
- **Stage-1:**
  - Coord auxiliary atomic contributions are treated as:
    - `S1/coord_soft_ce`, `S1/coord_w1`, optional `S1/coord_ce`, optional `S1/coord_gate`
    - sourced from `CoordSoftCEW1Result.{softce_contrib,w1_contrib,ce_contrib,gate_contrib}`.
  - Base text CE is excluded (coord-only).

Atomic coord/geo term allowlist (MUST be monitored when present):
- bbox geometry atoms:
  - `bbox_smoothl1` (SmoothL1 on decoded bbox corners; from `bbox_smoothl1_contrib`)
  - `bbox_ciou` (CIoU on decoded bbox corners; from `bbox_ciou_contrib`)
- coord distribution atoms:
  - `coord_token_ce` (hard CE on coord bins; from `coord_token_ce_contrib`)
  - `coord_soft_ce` (soft CE on coord bins; from `coord_soft_ce_contrib`)
  - `coord_w1` (W1 distance / CDF loss on coord bins; from `coord_w1_contrib`)
  - `coord_el1` (expected L1 on bins; from `coord_el1_contrib`)

Other coord atoms MAY be monitored when configured (non-zero effective weight), but are not required by this change:
- `coord_ehuber`, `coord_entropy`, `coord_gate`

Stage-2 Channel-A enumeration (explicit provenance split):
- For any monitored atom `<atom>` above, the term naming MUST support:
  - `A1_coord/<atom>`: A1 coord/geo terms (anchor pathway; optional, only when A1 anchor is enabled and included in `total`)
  - `A2_coord/<atom>`: A2 coord/geo terms (self-context pathway; whenever the corresponding module is enabled)
- Channel-B rollout-context terms use:
  - `B_coord/<atom>`

Rationale:
- This matches the question we care about for instability diagnosis in grounding: "which coordinate supervision atoms dominate/conflict in the *actual* gradient update?"
- The term count remains bounded (typically `2 + up to ~7` atoms per provenance), and monitoring is sparse.

### 2) A Small, Output-Adjacent Shared Parameter Probe Block (Default)

We must pick `θ_shared` such that:
- it is shared across all loss terms,
- gradient computation is cheap enough to run periodically,
- it is stable across Stage-1 and Stage-2.

Default probe block:
- Select a small, output-adjacent transformer block parameter set, for example:
  - the **final language transformer layernorm parameters** (small and always present), or
  - a similarly small last-block subset (configurable).

Configuration:
- `custom.extra.loss_gradient_monitor.param_block` supports:
  - `strategy: auto_last_lm_layernorm` (default),
  - `strategy: regex` with `include` / `exclude` patterns over `model.named_parameters()` keys,
  - optional caps: `max_params` and/or `max_numel` to guarantee bounded memory.

Failure mode:
- If no parameters are selected, the monitor disables itself for the run (warn-once) and training continues.

### 3) Gradient Computation Strategy (Best-Effort, Sparse)

On monitor steps, we compute gradients per term using:
- `torch.autograd.grad(Li, params_shared, create_graph=False, allow_unused=True, retain_graph=True)`

Key properties:
- `retain_graph=True` is necessary because the training loop still needs to run a real backward pass on the full summed objective after `compute_loss` returns.
- We never write into `.grad` and never call `optimizer.step()`. This remains monitoring-only.
- Grads are reduced to float scalars via `L2` norms and dot products computed in `float32`.

DDP considerations:
- Computing per-term grads should not trigger extra allreduces.
- When the model exposes `no_sync()` (DDP), monitoring gradient calls run inside `model.no_sync()` so that autograd does not synchronize gradients across ranks on these extra backward computations.
- Monitor activation is gated deterministically by `global_step` and (when available) `accelerator.sync_gradients`, so all ranks either run or skip together.

AMP considerations:
- Monitoring computations operate on the same loss tensors used for training (pre-GradScaler).
- Norm/dot computations cast grads to `float32` for numeric stability.

### 4) Cosine Similarity + Negative Pair Fraction

For terms `{Li}`, compute per-term gradient vectors `gi = ∇θ_shared Li`.

For all pairs `(i,j)` with non-zero norms:
- `cos(i,j) = dot(gi, gj) / (||gi|| * ||gj|| + eps)`
- Count negative cos pairs and report:
  - `neg_pair_frac = (# cos<0) / (# valid pairs)`
  - `neg_pair_pct = 100 * neg_pair_frac`

We do **not** log the full pairwise matrix by default to avoid metric explosion.
Optional debug mode may emit `gradmon/cos/<term_i>__<term_j>` for a small term set.

### 4.1) Cosine To Total (Term vs Net Coord Update Direction)

To improve interpretability without logging a full similarity matrix, the monitor SHALL also compute cosine similarity between each term gradient and the net coord/geo gradient direction:

- Define `g_total = Σ_i gi` across all monitored terms.
- For each term `i` with non-trivial norm, compute:
  - `cos_to_total(i) = dot(gi, g_total) / (||gi|| * ||g_total|| + eps)`

Interpretation:
- `cos_to_total(i) < 0` indicates term `i` is opposing the net coord/geo update direction on the shared probe block.

### 5) EMA-Normalized Loss Scalars

To make scalar loss magnitudes comparable across runs and across terms, we maintain an EMA per term:
- `ema_i <- beta * ema_i + (1-beta) * abs(raw_loss_i)`
- `loss_ema_norm_i = raw_loss_i / (ema_i + eps)`

Defaults:
- `beta = 0.98` (configurable),
- `eps = 1e-8`.

EMA state is stored on the trainer (or inside the monitor instance) keyed by the discovered term name.

### 6) Logging Contract (Sparse)

Keys are emitted only on monitor steps. Proposed canonical keys:

- Per-term scalars:
  - `gradmon/loss_raw/<term>`
  - `gradmon/loss_ema_norm/<term>`
  - `gradmon/grad_norm/<term>`
  - `gradmon/cos_to_total/<term>`
- Aggregate diagnostics:
  - `gradmon/grad_norm_ratio_max_over_median`
  - `gradmon/neg_cosine_pair_frac`
  - `gradmon/neg_cosine_pair_pct`
  - `gradmon/neg_cos_to_total_frac` (fraction of monitored terms with `cos_to_total < 0`)
  - `gradmon/num_terms`
  - `gradmon/shared_param_count`
  - `gradmon/shared_param_numel`
- Optional timing:
  - `time/gradmon_s` (measured wall time for the monitoring block)

Stage-2 integration respects the existing “micro-batch aggregation -> optimizer-step log” contract:
- Monitor metrics are added into the same pending log buffers as other mean-like scalars so they are logged once per optimizer step.

### 7) Enablement Surface (Config-First)

Monitor is disabled by default.

Config entry (example):
```yaml
custom:
  extra:
    loss_gradient_monitor:
      enabled: true
      interval_steps: 100
      ema_beta: 0.98
      # Coord-only monitor (exclude text CE and text_gate).
      coord_only: true
      # Stage-2 Channel-A uses atomic provenance split (A1_coord/* vs A2_coord/*).
      granularity: atomic
      # Optional: only compute on last accumulation micro-step.
      require_sync_gradients: true
      param_block:
        strategy: auto_last_lm_layernorm
        # strategy: regex
        # include: "model\\.layers\\.(\\d+)\\."
        # exclude: "visual|vision|embed"
        max_params: 64
        max_numel: 200000
      reduce_across_ranks: false
```

Parsing/injection:
- Config is read from `custom.extra.loss_gradient_monitor` in `src/sft.py` and attached to trainer instances (Stage-1 and Stage-2).

## Risks / Trade-offs

- **Overhead on monitor steps:** computing multiple `autograd.grad` calls adds backward-like work.
  - Mitigation: sparse interval (default 100), small probe block, coord-only atom list is bounded.
- **Probe block representativeness:** gradients on a small block may not reflect full-model behavior.
  - Mitigation: allow regex-based selection and caps; document that it is a diagnostic probe.
- **DDP interaction with extra backward passes:** must avoid unexpected gradient synchronization or reducer state issues.
  - Mitigation: deterministic gating, `no_sync()` wrapping, best-effort failure policy.
- **Key sprawl:** per-term keys can grow if term naming is too granular.
  - Mitigation: define terms as a fixed coord-only atom allowlist (no text CE atoms; no full cosine matrix by default).

## Migration Plan

No migration required.
- The monitor is disabled by default and does not change existing objectives.
- Enabling is opt-in via `custom.extra.loss_gradient_monitor.enabled: true`.

## Open Questions

- Default `param_block.strategy`:
  - confirm the most stable “small but representative” block across Qwen3-VL variants (final LM layernorm is the current recommendation).
- Should Stage-1 always split coord auxiliary terms into atoms (softCE/W1/gate/coordCE), or allow a `granularity: total` option to group them into `S1/coord_total`?
- Should we optionally all-reduce some statistics across ranks (e.g. negative pair numerator/denominator) for less noisy logs in DDP?
