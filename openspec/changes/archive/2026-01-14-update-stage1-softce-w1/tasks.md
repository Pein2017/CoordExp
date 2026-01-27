## 1. Implementation
- [x] 1.1 Add a config surface (YAML) to enable Stage-1 coord-gated `softCE+W1` loss mode and its weights/temperatures as placeholders.
- [x] 1.2 Add a new loss module `src/coord_tokens/soft_ce_w1.py` that provides:
  - [x] Gaussian-kernel unimodal soft-label builder over 0..999 bins (per coord token) with configurable sigma/width.
  - [x] `softCE` against a soft target (log-softmax based; NaN-safe).
  - [x] 1D Wasserstein-1 loss over bins using CDF differences (vectorized; stable under bf16/fp16).
  - [x] a combined per-token coord loss: `λ_softCE * softCE + λ_W1 * W1`, computed only at coord positions.
- [x] 1.3 Integrate coord-vocab gating at coord positions by slicing logits to ordered coord-token IDs (0..999 bins) before computing coord losses.
- [x] 1.4 Integrate Stage-1 training behaviour in a **single forward pass**:
  - [x] mask coord-token targets to `ignore_index` for the base CE loss (non-coord tokens only),
  - [x] compute coord `softCE+W1` from the same forward logits (no second forward),
  - [x] ensure packed and non-packed batches behave identically.
- [x] 1.7 Add unit tests for:
  - [x] label builder unimodality + normalization,
  - [x] W1(CDF) correctness on simple distributions,
  - [x] coord-vocab gating (no non-coord probability mass at coord positions in the loss path).
- [x] 1.8 Add a Stage-1 retrain config YAML under `configs/` (placeholders only; no default hyperparams) and document the expected run command.
- [x] 1.9 Add/extend logging keys so Stage-1 runs report: `loss`, `token_acc`, `coord_token_acc`, plus coord-loss breakdown terms (softCE/W1) when enabled.

## 2. Retraining Checklist (operational, paper-ready)
NOTE: This section is an operational checklist for running Stage-1 experiments. It is intentionally NOT tracked
as OpenSpec tasks (i.e., no checkboxes) so it doesn't block archiving code/spec changes.

- 2.1 Select the Stage-1 base checkpoint to retrain from (`<BASE_CKPT_PATH>`).
- 2.2 Select the Stage-1 train/val JSONL(s) (`<TRAIN_JSONL>`, `<VAL_JSONL>`), ensuring coord-token mode is enabled.
- 2.3 Run Stage-1 retraining with fixed seed and clear run naming (`<RUN_NAME>`), saving the resolved config with the output.
- 2.4 Verify:
  - JSON parse rate on a held-out sample set,
  - coord-token accuracy and coord-loss curves are stable (no spikes / NaNs),
  - basic detection evaluator sanity check (optional, offline).
- 2.5 Export the resulting Stage-1 checkpoint to the expected location (`<STAGE1_CKPT_OUT>`), and record metrics + config snapshot.
