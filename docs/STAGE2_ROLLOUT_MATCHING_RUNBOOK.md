# Stage-2 (Rollout-Matching) SFT Runbook

This doc is a minimal “paper-ready” checklist for running the rollout-matching SFT
trainer (stage_2), enabled via:

`custom.trainer_variant: rollout_matching_sft`

Authoritative requirements live under:
- `openspec/changes/2026-01-15-add-rollout-matching-trainer/specs/rollout-matching-sft/spec.md`

## What Stage-2 Does (One Forward Pass)

Stage_2 performs:

rollout (no grad) -> strict parse -> match -> build one teacher-forced target -> masked losses

The canonical assistant training target is:

`Y_train = Y_rollout_prefix + SerializeAppend(FN_gt_objects) + EOS`

Key policies:
- Rollout parsing is STRICT (no JSON repair). Invalid predicted objects are DROPPED.
- Missing GT objects (FN) are ALWAYS appended in the tail (recall recovery stays mandatory).
- Coord supervision stays token-distributional (softCE + W1 + gate) at coord slots.
- `desc` string VALUE tokens are not supervised by CE in stage_2 (JSON structure remains supervised).

## Hard Constraints / Gotchas

- Packing is NOT supported:
  - Set `training.packing: false`.
  - The trainer fails fast if packing is enabled.
- The rollout prefix is treated as immutable in token space:
  - Only suffix-only trimming is allowed (no decode+re-encode of earlier tokens).

## Rollout Parsing Policy (Current Rollouts)

The current rollout behavior (20-sample smoke at
`output/infer/rollout_ckpt3106_smoke/pred.jsonl`) commonly includes a trailing
`<|im_end|>` token and occasionally true truncation mid-object.

Parsing policy:
- Treat `<|im_end|>` as a hard stop (strip it, even when fused into the final token).
- If the rollout is truncated mid-object, suffix-trim to the last complete object boundary.
- Make the prefix append-ready by dropping the final top-level `}` (open JSON object).
- Failure fallback:
  - If no opening `{` exists, or no append-ready prefix can be produced, use
    `Y_rollout_prefix = "{"` (no prefix supervision; all GT become FN and are appended).

## Recommended Matching Knobs (Starting Point)

These are reasonable smoke defaults (tune later with logs):
- `candidate_top_k: 5`
- `maskiou_gate: 0.3`

Interpretation:
- `candidate_top_k` prunes GT candidates per predicted object before expensive geometry.
- `maskiou_gate` rejects low-quality matches early; rejected GT remain FN and are appended.

## Config Checklist

Start from: `configs/rollout_matching_sft_template.yaml`

Set:
- `custom.train_jsonl`
- `custom.val_jsonl`
- `custom.extra.rollout_matching.*` (decode + matching knobs)
- `training.packing: false`

Decoding notes:
- Start with greedy (`decode_mode: greedy`, `temperature: 0.0`) for stability.
- Ensure `max_new_tokens` is large enough to avoid systematic truncation
  (LVIS dense outputs can be ~11k text tokens in the tail).

## Command

From repo root:

`PYTHONPATH=. /root/miniconda3/envs/ms/bin/python -m src.sft --config <yaml> [--base_config <yaml>]`

4 GPUs:

`PYTHONPATH=. /root/miniconda3/envs/ms/bin/torchrun --nproc_per_node 4 -m src.sft --config <yaml> [--base_config <yaml>]`

## Health Counters to Watch

The trainer logs rollout health without logging IoU/maskIoU numeric metrics
(maskIoU is internal to matching only).

Parsing/matching counters:
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

## Minimal Preflight Validation

- Spec validity:
  - `openspec validate 2026-01-15-add-rollout-matching-trainer --strict`
- Unit tests:
  - `PYTHONPATH=. /root/miniconda3/envs/ms/bin/python -m pytest -q tests/test_rollout_matching_sft.py -q`

