# Change: Retrain Stage-1 with coord-gated softCE+W1 supervision

## Why
Stage-1 SFT is currently treated as “coord-token pretraining complete”, but the loss recipe is still optimized around expectation-decoding and continuous box losses (L1/GIoU). This proposal updates Stage-1 to a simpler, more stable, **token-native** geometry supervision that matches the project’s JSON-only output contract: **apply standard CE only to non-coord tokens**, and supervise `<|coord_*|>` positions using **softCE + 1D Wasserstein-1 (CDF)** under a **coord-vocab gate**.

The goal is to retrain a Stage-1 checkpoint that:
- emits valid JSON reliably (unchanged),
- learns a well-shaped coordinate distribution at `<|coord_*|>` slots (unimodal and calibratable),
- does **not** rely on Expect→float→L1/GIoU to make Stage-1 work.

## What Changes
- Add an opt-in Stage-1 loss mode that:
  - applies full-vocab CE only on **non-coord** tokens (text + JSON structure),
  - applies **coord-vocab-gated** `softCE(Gaussian kernel) + W1(CDF)` only on `<|coord_*|>` token positions,
  - does not require expectation decoding (including top-k expectation) or box-level regression losses (L1/GIoU).
- Ensure the implementation is **single-forward**:
  - one model forward produces logits,
  - the native CE path computes loss on non-coord tokens by masking coord labels to `ignore_index`,
  - coord `softCE+W1` is computed from the same logits (no second forward).
- **BREAKING**: Remove legacy expectation-decoding and box-level aux losses:
  - `custom.coord_loss` (L1/GIoU/poly losses) and its trainer/collator wiring
  - `custom.coord_expectation_metrics` (decoded-coordinate diagnostics)
  - configs that still reference these keys SHALL fail fast with a clear error
- Add utilities to:
  - build unimodal soft labels over bins (per coord token),
  - compute 1D Wasserstein-1 via CDF differences on ordered bins,
  - combine and log these losses in a packing-compatible way.
- Provide a reproducible Stage-1 retraining config (YAML-first) and an evaluation checklist.

## Impact
- Affected specs:
  - `coord-token-mode` (loss helpers extended for Stage-1 retraining)
- Affected code (expected):
  - `src/coord_tokens/` (new soft-label + W1 loss helpers; coord-vocab gating use)
  - `src/sft.py` and/or collators/metrics (masking coord tokens from base CE when enabled; logging)
  - `configs/` (new Stage-1 retrain YAML that enables the mode)
- Backward compatibility:
  - **BREAKING**: legacy coord aux-loss / expectation-metric config keys are removed and will error if present.
  - Coord-token mode requires distribution losses; coord tokens cannot be enabled without `custom.coord_soft_ce_w1.enabled`.

## Non-goals
- No Stage-2 EM-ish / matching changes in this proposal.
- No changes to the dataset JSONL contract or chat template format.
- No attempt to preserve legacy expectation-decoding or box-level losses for ablations (those are removed by design).
