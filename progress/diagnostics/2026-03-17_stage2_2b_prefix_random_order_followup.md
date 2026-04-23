---
title: 2B Prefix-Control Repair and Random-Order Stage-1 Follow-Up
date: 2026-03-17
status: in_progress
topics: [stage2, 2b, prefix, fn-analysis, stage1, random-order]
tags: [2b, diagnostics, prefix, random-order, stage1, follow-up]
summary: Follow-up note for the 2B FN investigation after the repaired Hard-16 prefix rerun completed. Prefix-control repair is now validated at the artifact level, yielding nonzero continuation-only recovery. The random-order Stage-1 ablation surface is implemented and aligned to the 1024 data, but the queued smoke training job has not yet executed.
---

# 2B Prefix-Control Repair and Random-Order Stage-1 Follow-Up (2026-03-17)

This note extends:

- `progress/diagnostics/2026-03-17_stage2_2b_fn_factor_results.md`

It tracks two follow-up threads:

1. repaired prefix-control intervention for the fixed 2B checkpoint pair
2. random-order Stage-1 ablation surface for the same 2B family

Fixed checkpoints:

- original:
  `output/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate_merged-1332`
- A-only:
  `output/stage2_ab/2b_1024/a_only_iter1/merged_ckpt-900`

Fixed datasets:

- train:
  `public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl`
- val:
  `public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl`

## Prefix-Control Repair

### What was fixed in code

The original Hard-16 prefix study mixed a real model-behavior question with a flawed intervention path.

The repair now does three important things:

1. baseline / self-prefix rows with pixel-space `bbox_2d` are converted back into norm1000 coord tokens before prefix serialization
2. assistant prefill uses the official chat template with `continue_final_message=True` instead of raw string concatenation
3. serialized object prefixes remain explicitly append-ready, and the close helper repairs duplicate comma boundaries deterministically

Implementation handles:

- `src/analysis/rollout_fn_factor_study.py`
- `tests/test_rollout_fn_factor_study.py`

Focused validation passed after the repair:

- `conda run -n ms ruff check src/analysis/rollout_fn_factor_study.py tests/test_rollout_fn_factor_study.py`
- `PYTHONPATH=. conda run -n ms python -m pytest tests/test_rollout_fn_factor_study.py -q`
  - result:
    `12 passed`

### What is now ruled out

The repaired prompt surface is not accidentally sealed by the processor.

A direct processor probe on:

- `output/stage2_ab/2b_1024/a_only_iter1/merged_ckpt-900`

confirmed that the assistant-prefill tail remains open and does **not** end with `<|im_end|>`.

So the remaining prefix behavior should no longer be attributed to a trivial chat-template closure bug.

### Final result of the repaired Hard-16 prefix rerun

The repaired rerun completed and rewrote:

- `output/analysis/rollout-fn-factor-2b-hard16-full-20260317/report.md`
- `output/analysis/rollout-fn-factor-2b-hard16-full-20260317/prefix_stage/stage_manifest.json`
- `output/analysis/rollout-fn-factor-2b-hard16-full-20260317/recovery/*.summary.json`

The repaired outcome is materially different from the earlier collapsed-prefix read.

Clean prefix-order cells:

- all `24 / 24` are now `health_valid=True`

Broken / switched stress cells:

- still `parse_invalid`

Recovered continuation-only signal:

- train / A-only:
  `prefix_sensitive_miss = 30`
- train / original:
  `34`
- val / A-only:
  `15`
- val / original:
  `15`

So the repaired prefix intervention now yields a real continuation-sensitive recovery channel instead of global collapse.

### What the repaired prefix result actually says

The repaired result is still not a clean endorsement of the original “training order wins” hypothesis.

Across the `94` true `prefix_sensitive_miss` rows in the refreshed Hard-16 recovery tables:

- train-order recovered `60`
- random-order recovered `60`
- self-prefix recovered `40`

The overlap structure is mixed:

- both train-order and random-order:
  `39`
- self-prefix only:
  `26`
- all three:
  `13`
- train-order only:
  `8`
- random-order only:
  `7`

So the current prefix read is:

- prefix state matters
- continuation can be unblocked by injected prefix context
- but the repaired evidence does **not** show a strong train-order advantage over random-order

This points more toward a general continuation-state / conditioning effect than a narrow sorted-order prior alone.

## Random-Order Stage-1 Ablation Surface

### New configs

Added:

- `configs/stage1/profiles/2b/coord_ce_soft_ce_w1_gate_coco80_desc_first_random_order.yaml`
- `configs/stage1/smoke/2b_coord_ce_soft_ce_w1_gate_coco80_desc_first_random_order.yaml`

These configs now explicitly align to the `1024` JSONLs:

- `public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl`
- `public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl`

and the smoke config now writes to smoke-only output roots instead of inheriting the production experiment root.

The intended ablation change remains:

- `custom.object_ordering: random`

with explicit cache disable:

- `training.encoded_sample_cache.enabled: false`

### Validation already completed

The config / dataset-ordering surface is already validated by:

- materialized config checks
- `tests/test_dense_caption_prompt_override.py`
- `tests/test_encoded_sample_cache.py`
- `tests/test_dataset_multworker_determinism_probe.py`

Previously verified result:

- `10 passed`

### Smoke execution status

A queued smoke launcher exists in:

- tmux session:
  `stage1_random_order_smoke_20260317`

but the smoke has **not** actually started.

Current state:

- there is no `temp/stage1_random_order_smoke_20260317.log`
- no smoke output has been written under the smoke output root
- the tmux session is still just a waiting shell

The most likely issue is that its wait loop matches the analysis command pattern too broadly and never advances cleanly.

So the random-order Stage-1 training result is still pending.

## Current Read

The follow-up now supports three bounded conclusions.

1. The earlier prefix-collapse conclusion was partly confounded by a real intervention bug, and that bug is now fixed.
2. After the repair, prefix continuation is a real secondary recovery channel, but it does not strongly favor train-order over random-order.
3. The random-order Stage-1 ablation surface is implementation-ready and aligned to the correct `1024` data, but it still needs a real smoke execution result.

## Next Actions

The most useful next checks are now:

1. launch the random-order Stage-1 smoke directly, without the hanging wait-shell wrapper
2. decide whether to rerun a true Hard-32 prefix extension or treat the refreshed Hard-16 prefix result as sufficient
3. if needed, build a more targeted prefix analysis that compares unique recovery sets for:
   - train-order only
   - random-order only
   - self-prefix only
