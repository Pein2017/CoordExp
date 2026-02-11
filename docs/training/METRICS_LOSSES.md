# Training Metrics and Losses (Quick Reference)

This doc is a compact "API surface" for interpreting `Trainer.log()` keys produced during
training/evaluation runs.

Notes:
- Keys shown here are the *train* names. During evaluation, ms-swift prefixes keys with
  `eval_` (e.g. `coord_diag/loss` -> `eval_coord_diag/loss`).
- Unless otherwise stated, metrics are computed on **supervised next-token positions**
  (i.e. `labels[:, 1:] != -100`).

Stage-2 note (rollout-matching SFT):
- Stage_2 (`custom.trainer_variant: rollout_matching_sft`) uses masked losses on a single
  teacher-forced forward:
  - rollout prefix: coord-token supervision only (prefix text CE is masked out),
  - appended GT tail: normal CE on JSON structure, coord-token supervision on coord slots,
    and CE for `desc` string *values* is intentionally masked out to avoid amplifying noisy GT labels.
- As a result, token-type metrics like `desc_token_frac` / `desc_token_acc` may be near-zero
  or not meaningful for stage_2 runs (because those positions are not supervised).
- Stage_2 runbook: `STAGE2_RUNBOOK.md`.

Stage-2 AB note (Channel-B path):
- Stage_2 AB (`custom.trainer_variant: stage2_ab_training`) uses unified Channel-B by default:
  rollout prefix + FN injection + one teacher-forced forward with explicit CE masks.
- Channel-B CE/geometry semantics:
  - matched prefix: structure CE ON, desc/coord CE OFF,
  - FP prefix: structure/desc/coord CE OFF,
  - FN-injected: structure+desc CE ON, coord CE OFF,
  - geometry loss on matched + FN, FP geometry OFF.
- Legacy `reordered_gt_sft` remains opt-in ablation behavior.

## Stage-2 Rollout-Matching Metrics (Training Logs)

Stage_2 (`custom.trainer_variant: rollout_matching_sft`) logs additional keys
under `rollout/*`, `packing/*`, and `time/*` to help diagnose failures and
performance during training.

For evaluation, Stage_2 uses a production-style evaluator (rollout -> parse ->
Hungarian match) and reports metrics under `eval_rollout_*` keys. This evaluator
intentionally skips teacher-forced encoding/loss computation, so `eval_loss` is
not reported for this trainer variant.

Important semantics:
- **Aggregated logging:** metrics are accumulated across gradient-accumulation micro-batches and
  logged once per optimizer step (same step index as `train/loss`).
- **Rank-local (training logs):** `rollout/*` keys logged during training are
  rank-local (not all-reduced), so they can vary across GPUs.
- **All-reduced (eval):** `eval_rollout_*` keys are aggregated over the full
  evaluation dataset and summed across ranks.

### Minimal train metrics (default)

Stage-2 training now emits a deliberately minimal, high-signal metric set (no
backward-compat aliases).

Rollout quality:
- `rollout/parse_dropped_invalid`
- `rollout/parse_obj_drop_frac`
- `rollout/parse_truncated_rate`
- `rollout/sample_valid_pred_rate`
- `rollout/sample_any_match_rate`
- `rollout/gating_rejection_rate`
- `rollout/precision`, `rollout/recall`, `rollout/f1`
- `rollout/fp_per_sample`, `rollout/fn_per_sample`
- `rollout/matched_maskiou_mean`

Rollout length / decode mode:
- `rollout/rollout_len_mean`, `rollout/rollout_len_p90`
- `rollout/decode_non_beam_count`, `rollout/decode_beam_count`
- `rollout/do_sample`, `rollout/temperature`

Throughput / packing:
- `time/rollout_generate_s`
- `time/rollout_parse_match_s`
- `rollout/gen_tokens_per_s`
- `packing/post_rollout_fill`

Loss breakdown:
- `loss/ce`
- `loss/coord`

Stage-2 AB scheduling:
- `stage2/channel_a`
- `stage2/channel_b`
- `stage2_ab/async/b_ratio_realized`

## Stage-2 Rollout-Matching Metrics (Eval)

When `custom.trainer_variant: rollout_matching_sft` runs evaluation (`training.eval_strategy != no`),
it reports production-style metrics derived from rollout -> parse -> Hungarian matching.

Returned keys (prefixed with `eval_`):
- `eval_rollout_precision`, `eval_rollout_recall`, `eval_rollout_f1`
- `eval_rollout_pred_objects`, `eval_rollout_gt_objects`, `eval_rollout_matched`
- `eval_rollout_fp`, `eval_rollout_fn`
- `eval_rollout_parse_truncated_rate`
- `eval_rollout_parse_dropped_invalid`, `eval_rollout_parse_dropped_ambiguous`
- `eval_rollout_sample_valid_pred_rate`, `eval_rollout_sample_any_match_rate`
- `eval_rollout_matched_maskiou_mean`

Optional desc monitor keys (when enabled):
- `eval_rollout_desc_pairs_total`
- `eval_rollout_desc_exact_acc_on_matched`
- `eval_rollout_desc_sem_enabled`
- `eval_rollout_desc_sem_acc_on_matched`
- `eval_rollout_desc_sem_sim_mean`, `eval_rollout_desc_sem_sim_count`

Config tip:
- For Stage_2 runs, prefer `training.metric_for_best_model: eval_rollout_f1` (and
  `training.greater_is_better: true`) to select best checkpoints by on-policy rollout quality.

## Loss Composition (Stage-1 / Scheme A)

Coord-offset adapter note (tie-head):
- Some stage-1 configs use `custom.coord_offset.enabled: true` to train coord-token rows via a lightweight
  offset adapter (instead of updating the full vocab embedding/head).
- By default `custom.coord_offset.tie_head: true`, which enforces a **single shared offset table** used for
  both embedding lookup and lm_head logits (Qwen-family tie-head semantics).
- Set `custom.coord_offset.tie_head: false` only for ablations that intentionally train embedding vs head
  offsets separately (export/merge then may need to materialize `lm_head.weight` and disable tying).

When `custom.coord_soft_ce_w1.enabled: true`:

1) Base LM loss (full vocab CE)
- **What:** model-native cross-entropy over the full vocabulary.
- **Where applied:** only on **non-coord** targets (coord targets are masked to `-100`).
- **Normalization:** mean over the number of supervised non-coord tokens (packing-safe).
- **In loss:** YES (train and eval_loss).

Diagnostics (logged as metrics, prefixed with `eval_` during evaluation):
- `base_ce/loss`
  - **What:** the base CE term after masking coord targets (i.e. non-coord CE only).
  - **In loss:** YES (this is the base term of the objective).
- `base_ce/noncoord_tokens`
  - **What:** supervised non-coord token count used by the base CE term (sanity-check denominator).
  - **In loss:** NO (diagnostic only).
- `base_ce/noncoord_tokens_per_sample`
  - **What:** `base_ce/noncoord_tokens / pack/num_samples` (packed runs only; batch-wide aggregate).
  - **In loss:** NO (diagnostic only; helps interpret scale per original sample).
- `base_ce/loss_per_sample`
  - **What:** approximate base-CE contribution per original sample:
    `base_ce/loss * base_ce/noncoord_tokens / pack/num_samples`.
  - **In loss:** NO (diagnostic only).

2) Coord-token loss (coord-gated distribution losses)
- **What:** extra supervision computed from the same forward logits at GT coord positions.
- **Where applied:** only at positions whose GT label is a coord token (1000-bin ordered vocab).
- **Normalization:** mean over the number of GT coord-token positions (packing-safe).
- **In loss:** YES (added to the base loss).

### Coord-token loss breakdown (`coord_diag/*`)

The coord loss is:

`coord_diag/loss = soft_ce_weight * softCE + w1_weight * W1 + ce_weight * CE + gate_weight * gate`

Compatibility note:
- The same breakdown is also logged under `coord_softce_w1/*` keys (legacy alias):
  - `coord_softce_w1/loss`, `coord_softce_w1/soft_ce`, `coord_softce_w1/w1`, `coord_softce_w1/ce`, `coord_softce_w1/gate`

- `coord_diag/enabled`
  - **What:** whether coord-gated softCE+W1(+gate) is active (`1.0`) or this is a pure-CE ablation (`0.0`).
  - **In loss:** NO (this is a tag for grouping/comparability across ablations).

- `coord_diag/loss`
  - **What:** coord loss term (already includes weights). When `coord_diag/enabled=1`, this is the
    coord term *added* to the training loss. When `coord_diag/enabled=0`, this is diagnostic-only.
  - **In loss:** YES iff `custom.coord_soft_ce_w1.enabled: true` (otherwise diagnostic-only).

- `coord_diag/soft_ce`
  - **What:** soft cross-entropy between the predicted coord distribution and a Gaussian
    soft target centered at the GT bin.
  - **In loss:** YES iff `custom.coord_soft_ce_w1.enabled: true` (otherwise diagnostic-only).

- `coord_diag/w1`
  - **What:** 1D Wasserstein-1 distance computed via CDF differences on the ordered bins.
  - **In loss:** YES iff `custom.coord_soft_ce_w1.enabled: true` (otherwise diagnostic-only).

- `coord_diag/ce`
  - **What:** optional CE-on-bins term (pure cross-entropy over the 1000-bin coord vocab).
  - **In loss:** YES iff `custom.coord_soft_ce_w1.enabled: true` and `ce_weight != 0` (otherwise diagnostic-only).

- `coord_diag/gate`
  - **What:** coord-vocab "mass leak" penalty at GT coord positions:
      `gate = -log(sum_{i in coord_vocab} softmax(full_logits / T)[i])`
  - **In loss:** YES iff `custom.coord_soft_ce_w1.enabled: true` and `gate_weight != 0` (otherwise diagnostic-only).

- `coord_diag/coord_vocab_mass`
  - **What:** mean probability mass inside the coord sub-vocabulary at GT coord positions.
    This is derived from the gate computation (approximately `exp(-gate)`).
  - **In loss:** NO (diagnostic only).

- `coord_diag/coord_tokens`
  - **What:** number of GT coord-token positions in the current batch (or mean count over
    logging windows). Useful for sanity-checking the denominator.
  - **In loss:** NO (diagnostic only).

- `coord_diag/acc_top5`
  - **What:** top-5 accuracy within the 1000-bin coord sub-vocabulary at GT coord positions.
  - **In loss:** NO (distribution-quality monitor).

- `coord_diag/p_gt_mean`
  - **What:** mean predicted probability assigned to the GT coord bin (after temperature).
  - **In loss:** NO (distribution-quality monitor).

- `coord_diag/margin_mean`
  - **What:** mean `(max_logit - gt_logit)` within coord vocab (after temperature); lower is better.
  - **In loss:** NO (distribution-quality monitor).

- `coord_diag/expected_bin_mae`
  - **What:** mean absolute error between the expected coord bin index (under the predicted
    coord-vocab distribution) and the GT bin index. Units are **bins** (0..999).
  - **In loss:** NO (distribution-quality monitor; often more informative than top-k early on).

- `coord_diag/expected_bin_abs_err_p90`
  - **What:** 90th percentile (p90) of the per-token absolute error
    `abs(expected_bin - gt_bin)` at GT coord positions. Units are **bins** (0..999).
  - **Why:** tail-sensitive complement to `expected_bin_mae` (mean-only summaries can hide heavy tails).
  - **In loss:** NO (distribution-quality monitor).

- `coord_diag/w1_to_delta`
  - **What:** W1 distance from the predicted coord distribution `p(k)` to a delta at the GT bin `t`,
    i.e. `E_p[|k - t|]`. Units are **bins** (0..999).
  - **Why:** expectation-friendly proxy for continuous geometry: distinguishes "flat but not closer"
    from "flat and near-GT mass". More shape-sensitive than `abs(E[k] - t)`.
  - **In loss:** NO (distribution-quality monitor).

- `coord_diag/coord_tokens_per_sample`
  - **What:** `coord_diag/coord_tokens / pack/num_samples` (packed runs only; batch-wide aggregate).
  - **In loss:** NO (diagnostic only).

- `coord_diag/loss_per_sample`
  - **What:** approximate coord-loss contribution per original sample:
    `coord_diag/loss * coord_diag/coord_tokens / pack/num_samples`.
  - **In loss:** NO (diagnostic only).

- `stage1/total_loss_per_sample_est`
  - **What:** `base_ce/loss_per_sample + coord_diag/loss_per_sample` (approx. total per-sample objective).
  - **In loss:** NO (diagnostic only; useful for packed runs).

## Token Accuracy and Token-Type Metrics

These metrics help interpret "does it output the correct token id", separated by token
categories. They are all **metrics-only** (not part of the loss).

**Config**:
- Enable with `custom.token_type_metrics.enabled: true`.
- Defaults: `include: ["lvis"]`, `exclude: []`.
- Set `custom.token_type_metrics.log_top5: false` to skip top-k metrics (can reduce logging overhead).
- Works on padded and packed batches: token types are computed per sample pre-pack and concatenated; if alignment fails the metrics are skipped (training continues).
- NaN-safe: batches with zero supervised tokens are skipped.

- `token_acc` (from ms-swift)
  - **What:** top-1 token accuracy over supervised tokens (argmax vs GT).
  - **In loss:** NO.

- `token_acc_top5` (CoordExp aggregate metric)
  - **What:** top-5 token accuracy over supervised tokens.
  - **In loss:** NO.
  - **Config:** set `custom.token_type_metrics.log_top5: false` to skip top-k metrics (can reduce logging overhead).

- `text_token_acc`
  - **What:** top-1 accuracy over supervised tokens that are not GT coord tokens.
  - **In loss:** NO.

If `custom.token_type_metrics.enabled: true`, we also log per-type splits (all metrics-only):
- `desc_token_frac`, `format_token_frac`, `coord_token_frac`
  - **What:** fraction of supervised tokens belonging to each GT type.

- `desc_token_acc`, `format_token_acc`, `coord_token_acc`
  - **What:** top-1 accuracy within each GT type.

- `desc_token_acc_top5`, `format_token_acc_top5`, `coord_token_acc_top5`
  - **What:** top-5 accuracy within each GT type.

## Coord Vocab "Both Ways" Monitors (`coord_monitor/*`)

These are diagnostics to disambiguate two failure directions:
- GT coord slot -> predicted non-coord token (coord slot collapse)
- GT non-coord slot (format/desc) -> predicted coord token (coord intrusion)

All `coord_monitor/*` keys are **metrics-only** (not part of the loss).

### Type flip rates (argmax-based)

- `coord_monitor/flip_coord_to_noncoord`
  - **What:** among GT coord slots, fraction where argmax prediction is NOT a coord token.

- `coord_monitor/flip_text_to_coord`
  - **What:** among GT non-coord slots, fraction where argmax prediction IS a coord token.

- `coord_monitor/flip_format_to_coord`
  - **What:** among GT FORMAT slots (e.g. `}`, `]`, `:`), fraction predicted as coord.

- `coord_monitor/flip_desc_to_coord`
  - **What:** among GT DESC slots, fraction predicted as coord.

### Coord-vocab mass (softmax-mass-based)

These report mean probability mass inside the coord sub-vocabulary, conditioned on GT type:

- `coord_monitor/coord_vocab_mass_at_gt_coord`
- `coord_monitor/coord_vocab_mass_at_gt_text`
- `coord_monitor/coord_vocab_mass_at_gt_format`
- `coord_monitor/coord_vocab_mass_at_gt_desc`

Interpretation:
- Low `..._at_gt_coord` means coord slots are bleeding probability into non-coord tokens.
- High `..._at_gt_format` / `..._at_gt_desc` means coord tokens are intruding into text/format slots.

Config:
- Set `custom.token_type_metrics.coord_monitor_mass: false` to skip these mass diagnostics.
- Optionally cap compute cost with `custom.token_type_metrics.coord_monitor_mass_max_tokens: <int>` (0 = no cap).

## Packed-Run Per-Sample Helpers

When packing is enabled, one "training unit" is a concatenation of multiple original samples.
To make scales more intuitive, CoordExp logs a pack-size helper:
- `pack/num_samples`
  - **What:** number of original samples concatenated into the current unit (batch-wide aggregate).
  - **Note:** in non-packed runs, this is effectively the batch size.

Stage_2 (rollout-matching / Stage2-AB) also logs packing-aware step helpers that are stable even when
post-rollout packing is used:
- `train/samples_total`
  - **What:** total number of raw (unpacked) samples that contributed to the current optimizer step.
  - **Channel-A:** dataset samples packed into the learner sequences.
  - **Channel-B:** rollout samples packed into the learner sequences (after parse/match/FN-append).
- `train/samples_seen`
  - **What:** cumulative `train/samples_total` over the run (rank-local in multi-GPU; exact in server-mode world_size=1).
  - **Why:** use this as a packing-aware "progress" axis for eval scheduling and throughput comparisons.
- `train/micro_steps`
  - **What:** number of micro-steps accumulated into the current optimizer step (≈ `gradient_accumulation_steps`).

## Reduction / Aggregation Semantics (Important)

Internally, per-step values are pushed into ms-swift's `MeanMetric` containers.
For scalar floats, this typically means "average over logging steps" (not token-weighted).

Where a metric is intended to be token-weighted, CoordExp computes a per-token mean first,
then updates the metric with a scalar.

Grad-accumulation note:
- Train `loss` is intended to be comparable to `eval_loss` (mean objective per optimizer step), even when
  `gradient_accumulation_steps > 1`.
- CoordExp logs two runtime helpers (metrics-only; prefixed with `eval_` during eval):
  - `accum/grad_steps`: configured `gradient_accumulation_steps`.
  - `accum/current_grad_steps`: per-update value (may differ on the last partial update in an epoch).
