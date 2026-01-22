# Training Metrics and Losses (Quick Reference)

This doc is a compact "API surface" for interpreting `Trainer.log()` keys produced during
training/evaluation runs.

Notes:
- Keys shown here are the *train* names. During evaluation, ms-swift prefixes keys with
  `eval_` (e.g. `coord_softce_w1/loss` -> `eval_coord_softce_w1/loss`).
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
- Stage_2 runbook: `docs/STAGE2_ROLLOUT_MATCHING_RUNBOOK.md`.

## Stage-2 Rollout-Matching Metrics (Training Logs)

Stage_2 (`custom.trainer_variant: rollout_matching_sft`) logs additional keys
under `rollout/*`, `packing/*`, and `time/*` to help diagnose failures and
performance during training.

For evaluation, Stage_2 uses a production-style evaluator (rollout -> parse ->
Hungarian match) and reports metrics under `eval_rollout_*` keys. This evaluator
intentionally skips teacher-forced encoding/loss computation, so `eval_loss` is
not reported for this trainer variant.

Important semantics:
- **Optimizer-step units:** when rollout buffering is enabled, "E-step vs M-step"
  is defined on optimizer steps (`TrainerState.global_step`), not micro-steps.
- **Aggregated logging:** metrics are accumulated across gradient-accumulation micro-batches and
  logged once per optimizer step (same step index as `train/loss`).
- **Buffering:** interpret rollout-quality metrics on **E-steps only** by
  filtering to `rollout/buffer_reuse == 0`. M-steps reuse cached targets and are
  not an on-policy rollout signal.
- **Rank-local (training logs):** `rollout/*` keys logged during training are
  rank-local (not all-reduced), so they can vary across GPUs.
- **All-reduced (eval):** `eval_rollout_*` keys are aggregated over the full
  evaluation dataset and summed across ranks.

### Buffer / EM window diagnostics

- `rollout/buffer_reuse`
  - **What:** 1.0 on M-steps (reuse), 0.0 on E-steps (fresh rollout).
- `rollout/buffer_window_step0`
  - **What:** optimizer-step index where the current reuse window started (E-step step id).
- `rollout/buffer_completed_steps`
  - **What:** how many optimizer steps in the current window have completed so far.
### Rollout timing / throughput

- `time/rollout_generate_s`
- `time/rollout_parse_match_s`
- `time/rollout_teacher_encode_s`
  - **Note:** these are forced to 0.0 on buffer reuse steps to avoid double counting.

- `rollout/gen_new_tokens_total|mean|p90|p99`
  - **What:** generated assistant token counts (after stage_2 suffix trimming).
  - **Why:** helps detect "always hit max_new_tokens" pathologies.

- `rollout/gen_tokens_per_s`
  - **What:** `gen_new_tokens_total / time/rollout_generate_s`.
  - **Why:** detects rollout slowdowns (KV cache pressure / chunked prefill regressions).

### Parse health

- `rollout/parse_dropped_invalid`, `rollout/parse_dropped_ambiguous`
  - **What:** number of predicted objects dropped by strict parsing.
- `rollout/parse_truncated`, `rollout/parse_truncated_rate`
  - **What:** sample count and rate where rollouts are truncated mid-object (suffix-trimmed).
- `rollout/parse_obj_total`
  - **What:** `valid_pred_objects + dropped_invalid + dropped_ambiguous` (object-level accounting).
- `rollout/parse_obj_valid_frac`, `rollout/parse_obj_drop_frac`
  - **What:** object-level valid/drop fractions.
- `rollout/sample_valid_pred_rate`
  - **What:** fraction of samples that yield at least one valid predicted object.
- `rollout/sample_any_match_rate`
  - **What:** fraction of samples that produce at least one supervised match.

### Matching quality (rollout-level)

- `rollout/gt_objects`, `rollout/valid_pred_objects`
  - **What:** GT and valid predicted object counts (post-parse).
- `rollout/matched_for_supervision`, `rollout/excluded_from_supervision`
  - **What:** matched objects that were used vs excluded due to target-construction failure.
- `rollout/fn_appended`, `rollout/fn`
  - **What:** GT objects not supervised via prefix matching; appended as FN in the tail.
- `rollout/gating_rejections`, `rollout/gating_rejection_rate`
  - **What:** how often candidate pairs were rejected by the `maskiou_gate` threshold.

- `rollout/precision`, `rollout/recall`, `rollout/f1`
  - **What:** object-level precision/recall/F1 derived from matched-for-supervision.
  - **Note:** `rollout/recall` is the same as `rollout/match_rate` (kept for compatibility).

- `rollout/matched_maskiou_mean`, `rollout/matched_maskiou_count`
  - **What:** mean maskIoU over matched pairs (norm1000-space, virtual canvas).
  - **Why:** disambiguates “more matches” vs “better geometry”.

### Desc monitoring (optional; metrics only)

Stage_2 can optionally monitor whether rollout `desc` strings stay semantically
aligned with GT on geometry-matched pairs. This is **metrics-only** and does not
affect the training loss.

- `rollout/desc_pairs_total`
  - **What:** number of geometry-matched pairs evaluated for desc monitoring.
- `rollout/desc_exact_acc_on_matched`
  - **What:** exact-match accuracy of normalized `desc` strings on matched pairs.
- `rollout/desc_sem_enabled`
  - **What:** 1.0 when the semantic embedding model is available for this step.
- `rollout/desc_sem_acc_on_matched`
  - **What:** semantic accuracy on matched pairs (exact OR cosine-sim >= threshold).
- `rollout/desc_sem_sim_mean`, `rollout/desc_sem_sim_count`
  - **What:** mean cosine similarity and count over matched pairs with embeddings.

### Supervision construction coverage

- `rollout/excluded_rate`
  - **What:** `excluded_from_supervision / (matched_for_supervision + excluded_from_supervision)`.
  - **Why:** detects OT/target-construction instability.

- `rollout/prefix_coord_targets_total`, `rollout/prefix_coord_targets_per_matched`
  - **What:** total coord slots supervised in the prefix and average per matched object.

- `rollout/tail_ignore_frac`
  - **What:** fraction of appended tail tokens that are ignored for CE due to `"desc"` masking.

### Length / packing diagnostics (stage_2)

- `rollout/prompt_len_mean|p90`
- `rollout/rollout_len_mean|p90`
- `rollout/train_len_mean|p90`
- `rollout/append_len_mean|p90`
- `rollout/encoded_len_mean|p90`
  - **What:** token-length summaries for prompt / rollout / training target / encoded length.
  - **Why:** explains OOM risk and packing fill changes between 1k/4k/8k max_new_tokens.

- `packing/post_rollout_fill`
- `packing/post_rollout_segments`
- `packing/post_rollout_buffer`
  - **What:** post-rollout packing stats (carry-only mode).

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

### Coord-token loss breakdown (`coord_softce_w1/*`)

The coord loss is:

`coord_softce_w1/loss = soft_ce_weight * softCE + w1_weight * W1 + gate_weight * gate`

- `coord_softce_w1/loss`
  - **What:** total coord loss added to the training loss (already includes weights).
  - **In loss:** YES (train and eval_loss when `custom.coord_soft_ce_w1.enabled: true`).

- `coord_softce_w1/soft_ce`
  - **What:** soft cross-entropy between the predicted coord distribution and a Gaussian
    soft target centered at the GT bin.
  - **In loss:** YES (train and eval_loss).

- `coord_softce_w1/w1`
  - **What:** 1D Wasserstein-1 distance computed via CDF differences on the ordered bins.
  - **In loss:** YES (train and eval_loss).

- `coord_softce_w1/gate`
  - **What:** coord-vocab "mass leak" penalty at GT coord positions:
      `gate = -log(sum_{i in coord_vocab} softmax(full_logits / T)[i])`
  - **In loss:** YES iff `gate_weight != 0` (train and eval_loss).

- `coord_softce_w1/coord_vocab_mass`
  - **What:** mean probability mass inside the coord sub-vocabulary at GT coord positions.
    This is derived from the gate computation (approximately `exp(-gate)`).
  - **In loss:** NO (diagnostic only).

- `coord_softce_w1/coord_tokens`
  - **What:** number of GT coord-token positions in the current batch (or mean count over
    logging windows). Useful for sanity-checking the denominator.
  - **In loss:** NO (diagnostic only).

- `coord_softce_w1/acc_top5`
  - **What:** top-5 accuracy within the 1000-bin coord sub-vocabulary at GT coord positions.
  - **In loss:** NO (distribution-quality monitor).

- `coord_softce_w1/p_gt_mean`
  - **What:** mean predicted probability assigned to the GT coord bin (after temperature).
  - **In loss:** NO (distribution-quality monitor).

- `coord_softce_w1/margin_mean`
  - **What:** mean `(max_logit - gt_logit)` within coord vocab (after temperature); lower is better.
  - **In loss:** NO (distribution-quality monitor).

- `coord_softce_w1/coord_tokens_per_sample`
  - **What:** `coord_softce_w1/coord_tokens / pack/num_samples` (packed runs only; batch-wide aggregate).
  - **In loss:** NO (diagnostic only).

- `coord_softce_w1/loss_per_sample`
  - **What:** approximate coord-loss contribution per original sample:
    `coord_softce_w1/loss * coord_softce_w1/coord_tokens / pack/num_samples`.
  - **In loss:** NO (diagnostic only).

- `stage1/total_loss_per_sample_est`
  - **What:** `base_ce/loss_per_sample + coord_softce_w1/loss_per_sample` (approx. total per-sample objective).
  - **In loss:** NO (diagnostic only; useful for packed runs).

## Token Accuracy and Token-Type Metrics

These metrics help interpret "does it output the correct token id", separated by token
categories. They are all **metrics-only** (not part of the loss).

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
