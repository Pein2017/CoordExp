---
doc_id: progress.diagnostics.residual_set_refactor_preflight_2026_05_22
layer: progress
doc_type: diagnostic-note
status: current
domain: training
summary: Stage-2 residual-set refactor integration tests and prepared-rollout fixture preflight.
tags: [stage2, residual-set, prepared-rollout, smoke, ckpt3664]
updated: 2026-05-22
---

# Residual-Set Refactor Preflight

Scope: deterministic CPU/integration preflight plus a tiny 4-GPU `max_steps=1`
smoke for the Stage-2 residual-set / UL correction refactor. This is not a
model-quality claim and does not replace full validation.

Config:

- `configs/stage2_two_channel/smoke/compact_full_residual_set_ckpt3664_hf_1step.yaml`

Prepared rollout fixture:

- `output/stage2_ab/prepared_rollouts/train8_ckpt3664.jsonl`
- generated with:

```bash
conda run -n ms python scripts/tools/prepare_stage2_residual_rollouts.py \
  --config configs/stage2_two_channel/smoke/compact_full_residual_set_ckpt3664_hf_1step.yaml \
  --out output/stage2_ab/prepared_rollouts/train8_ckpt3664.jsonl \
  --mode fixture \
  --train-sample-limit 8 \
  --expected-num-rollouts 4 \
  --seed 17 \
  --greedy-rollouts 1 \
  --sampling-rollouts 3 \
  --include-debug-cases invalid_bbox_dirty_prefix,exact_duplicate_attempt
```

Fixture evidence:

- `prepared_rows=33`
- `samples=8`;
- raw per-sample K distribution before exact-token dedup is `{4: 7, 5: 1}`;
- training-side exact-token dedup with `legacy_reencode_fallback=false` yields
  `K_after_dedup={4: 8}`;
- first sample includes one exact duplicate rollout attempt before exact-token
  dedup; the only dropped reason is
  `exact_duplicate_response_token_ids=1`;
- `debug_case=invalid_bbox_dirty_prefix` is present;
- required prepared-rollout v1 fields are present;
- prepared rows overlap the configured first 8 training samples by runtime
  dataset `sample_id` keys, where JSONL-backed fixture rows use the same
  dataset namespace plus `base_idx` contract as `BaseCaptionDataset`;
- raw `image_id`/`image_path` remain present for provenance and secondary
  lookup.

Verification:

```bash
conda run -n ms python -m pytest \
  tests/test_stage2_ab_config_contract.py \
  tests/test_stage2_residual_boundary_adapter.py \
  tests/test_stage2_residual_set_correction.py \
  tests/test_stage2_residual_ul_consensus.py \
  tests/test_stage2_residual_set_loss_module.py \
  tests/test_stage2_teacher_forcing_adapter_contract.py \
  tests/test_stage2_ab_training.py \
  -q
# 447 passed in 5.21s

openspec validate add-stage2-residual-set-ul-correction --type change --strict --no-interactive
# Change 'add-stage2-residual-set-ul-correction' is valid

conda run -n ms python -m pytest tests/test_prepare_stage2_residual_rollouts.py -q
# 9 passed in 11.31s

conda run -n ms python -m pytest \
  tests/test_stage2_ab_training.py \
  -k "offline_residual_set or compact_full_residual_path"
# 7 passed, 162 deselected in 0.66s
```

GPU smoke:

- First current-HEAD attempt:
  `temp/train_logs/compact_full_residual_set_ckpt3664_hf_1step-20260522T031325Z.log`
  failed before forward with
  `ValueError: stage2-ab Channel-B step mode produced no post-rollout segments`.
  Root cause: the prepared fixture used raw COCO `image_id` as `sample_id`, while
  the live identity-collator training path receives encoded dataset samples
  whose stable lookup key is the runtime `sample_id` attached by
  `BaseCaptionDataset` (`dataset` namespace plus `base_idx`).
- Fixed by regenerating
  `output/stage2_ab/prepared_rollouts/train8_ckpt3664.jsonl` with runtime
  `sample_id` values while preserving raw `image_id`/`image_path`.
- Passing smoke command, using idle GPUs available at launch time:

```bash
config=configs/stage2_two_channel/smoke/compact_full_residual_set_ckpt3664_hf_1step.yaml \
gpus=0,1,2,6 \
conda run -n ms bash scripts/train.sh
```

- Launcher log:
  `temp/train_logs/compact_full_residual_set_ckpt3664_hf_1step-20260522T033227Z.log`
- Artifact root:
  `output/stage2_ab/smoke/compact_full_residual_set_ckpt3664_hf_1step/smoke_1step-compact_full-residual_set-ckpt3664-hf-unconstrained/v5-20260522-033424`
- Phase trace segment counts:
  rank00=`8`, rank01=`7`, rank02=`8`, rank03=`8`.
- `logging.jsonl` residual metrics check:
  rows=`2`, metric_rows=`1`,
  `stage2_ab/channel_b/residual_set/sequence_count=31.0`,
  `stage2_ab/channel_b/residual_set/atom_count=81.0`,
  `stage2_ab/channel_b/residual_set/prepared/K_total=25.0`.
- Training summary:
  `train_loss=14.22993088`, `train_runtime=5.8188`, `global_step/max_steps=1/1`.
