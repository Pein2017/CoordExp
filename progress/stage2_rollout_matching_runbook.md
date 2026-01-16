# Stage-2 (rollout-matching) runbook

This note is a minimal “paper-ready” checklist for running the rollout-matching SFT trainer (`custom.trainer_variant: rollout_matching_sft`).

## 1) Naming + determinism

- Pick a `run_name` that encodes: dataset, base ckpt, decode mode, gate/top-k, seed.
  - Example: `stage2_lvis_base3106_greedy_gate0p30_top10_seed17`
- Keep `training.seed` fixed (e.g. 17) for reproducible debugging.

## 2) Config

- Start from `configs/rollout_matching_sft_template.yaml`.
- Set:
  - `custom.train_jsonl`
  - `custom.val_jsonl`
  - `custom.extra.rollout_matching.*` (decode + matching knobs)
- Ensure:
  - `training.packing: false` (required; rollout generation does not support packing)

## 3) Command

Run from repo root:

`PYTHONPATH=. /root/miniconda3/envs/ms/bin/python -m src.sft --config <yaml> [--base_config <yaml>]`

## 4) Health counters to watch

The trainer logs rollout health without logging IoU/maskIoU metrics (maskIoU is internal to matching only).

Key counters to watch in logs/tensorboard:
- `rollout/parse_dropped_invalid`
- `rollout/parse_truncated`
- `rollout/valid_pred_objects`
- `rollout/matched_for_supervision`
- `rollout/excluded_from_supervision`
- `rollout/fn_appended`
- `rollout/gating_rejections`

Loss breakdown:
- `loss/ce`
- `loss/coord`
- `loss/coord_prefix`
- `loss/coord_tail`

## 5) Smoke-test recipe (small N)

- Set a strict sample limit (e.g. `debug.enabled: true` + `debug.train_sample_limit`, `debug.val_sample_limit`).
- Use greedy decoding first (`decode_mode: greedy`, temperature ~0) to stabilize parsing.
- Verify:
  - No NaNs/Infs.
  - Parsing produces some valid objects (dropped-invalid is not ~100%).
  - FN append is non-zero early (expected) and decreases as matching improves.

