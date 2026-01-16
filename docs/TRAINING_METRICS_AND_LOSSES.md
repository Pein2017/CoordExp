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

## Loss Composition (Stage-1 / Scheme A)

When `custom.coord_soft_ce_w1.enabled: true`:

1) Base LM loss (full vocab CE)
- **What:** model-native cross-entropy over the full vocabulary.
- **Where applied:** only on **non-coord** targets (coord targets are masked to `-100`).
- **Normalization:** mean over the number of supervised non-coord tokens (packing-safe).
- **In loss:** YES (this is the primary training loss term).

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
  - **In loss:** YES (train), NO (eval-only logging).

- `coord_softce_w1/soft_ce`
  - **What:** soft cross-entropy between the predicted coord distribution and a Gaussian
    soft target centered at the GT bin.
  - **In loss:** YES (train), NO (eval-only logging).

- `coord_softce_w1/w1`
  - **What:** 1D Wasserstein-1 distance computed via CDF differences on the ordered bins.
  - **In loss:** YES (train), NO (eval-only logging).

- `coord_softce_w1/gate`
  - **What:** coord-vocab "mass leak" penalty at GT coord positions:
      `gate = -log(sum_{i in coord_vocab} softmax(full_logits / T)[i])`
  - **In loss:** YES iff `gate_weight != 0` (train), NO (eval-only logging).

- `coord_softce_w1/coord_vocab_mass`
  - **What:** mean probability mass inside the coord sub-vocabulary at GT coord positions.
    This is derived from the gate computation (approximately `exp(-gate)`).
  - **In loss:** NO (diagnostic only).

- `coord_softce_w1/coord_tokens`
  - **What:** number of GT coord-token positions in the current batch (or mean count over
    logging windows). Useful for sanity-checking the denominator.
  - **In loss:** NO (diagnostic only).

## Token Accuracy and Token-Type Metrics

These metrics help interpret "does it output the correct token id", separated by token
categories. They are all **metrics-only** (not part of the loss).

- `token_acc` (from ms-swift)
  - **What:** top-1 token accuracy over supervised tokens (argmax vs GT).
  - **In loss:** NO.

- `token_acc_top5` (CoordExp aggregate metric)
  - **What:** top-5 token accuracy over supervised tokens.
  - **In loss:** NO.

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

## Reduction / Aggregation Semantics (Important)

Internally, per-step values are pushed into ms-swift's `MeanMetric` containers.
For scalar floats, this typically means "average over logging steps" (not token-weighted).

Where a metric is intended to be token-weighted, CoordExp computes a per-token mean first,
then updates the metric with a scalar.
