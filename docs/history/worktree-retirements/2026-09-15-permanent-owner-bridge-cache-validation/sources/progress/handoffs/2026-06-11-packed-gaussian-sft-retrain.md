# Handoff: Restart Packed Gaussian Stage-1 SFT

Date: 2026-06-11

## Repo State

- Branch: `codex/fully-compact-2x2-ablation`
- Remote/upstream: `origin/codex/fully-compact-2x2-ablation`
- Code fix commit: `c4b10744 Fix packed Gaussian coord soft CE scaling`
- Sync command on the other machine:

```bash
git fetch origin
git checkout codex/fully-compact-2x2-ablation
git pull --ff-only
```

## What Changed

- Fixed packed Gaussian SFT aux loss scaling so it uses true accumulation-window coord-token `token_mean`.
- Made packed batch contract validation run on the always-on Stage-1 SFT path before batch extras are popped.
- Did not launch production training after this fix.
- Do not use `grammar-on`; treat grammar-constrained decode/eval as legacy and unreliable for this study.

Key files:

- `src/trainers/metrics/sft_gaussian_coord_soft_ce.py`
- `src/trainers/losses/sft_gaussian_coord_soft_ce.py`
- `src/trainers/metrics/batch_contract.py`
- `src/trainers/metrics/aggregate_tokens.py`
- `tests/test_sft_gaussian_coord_soft_ce.py`
- `tests/test_grad_accum_loss_scale_mixin.py`

## Verification Already Run

```bash
/root/miniconda3/envs/ms/bin/python -m pytest \
  tests/test_sft_gaussian_coord_soft_ce.py \
  tests/test_grad_accum_loss_scale_mixin.py \
  tests/test_train_batch_contract.py \
  tests/test_packing_wrapper.py \
  tests/test_batch_extras_contract.py \
  tests/test_training_runtime_profile.py -q
```

Result: `76 passed, 10 warnings`.

The warnings are the existing multiprocessing fork deprecation warnings from `tests/test_packing_wrapper.py`.

Also run:

```bash
git diff --cached --check
```

Result before commit: passed.

## Config To Restart

Run the Gaussian coordinate-soft CE packed Stage-1 SFT production config:

```text
configs/stage1/recursive_detection_ce_latest/prod/compact_full_random_sft_coord_gauss_softce_mix0p5_frac0p04_cap8_llm_lora_packed.yaml
```

This is the intended new training arm:

- random object order
- pure SFT/CE path, not ET-RMP/trie
- Gaussian coord soft CE auxiliary loss
- `mix=0.5`
- `r95_axis_fraction=0.04`
- `cap_bins=8`
- packed training
- global max length `12000`
- LLM-only LoRA
- LoRA rank `16`, alpha `32`
- vision tower frozen
- aligner frozen

Matched hard-CE control config, if needed for comparison:

```text
configs/stage1/recursive_detection_ce_latest/prod/compact_full_random_sft_llm_lora_packed.yaml
```

4-GPU smoke config before production:

```text
configs/stage1/recursive_detection_ce_latest/smoke/compact_full_random_sft_coord_gauss_softce_llm_lora_packed_4gpu_smoke.yaml
```

## Run From Repo Root

All paths below are relative to the repo root.

Recommended smoke command:

```bash
tmux new-session -d -s coordexp_gauss_softce_smoke_4gpu \
  "bash -lc 'COORDEXP_TRAIN_HEARTBEAT=1 config=configs/stage1/recursive_detection_ce_latest/smoke/compact_full_random_sft_coord_gauss_softce_llm_lora_packed_4gpu_smoke.yaml gpus=0,1,2,3 conda run --no-capture-output -n ms bash scripts/train.sh'"
```

Monitor:

```bash
tmux attach -t coordexp_gauss_softce_smoke_4gpu
```

If the smoke is healthy, launch production:

```bash
tmux new-session -d -s coordexp_gauss_softce_prod_4gpu \
  "bash -lc 'COORDEXP_TRAIN_HEARTBEAT=1 config=configs/stage1/recursive_detection_ce_latest/prod/compact_full_random_sft_coord_gauss_softce_mix0p5_frac0p04_cap8_llm_lora_packed.yaml gpus=0,1,2,3 conda run --no-capture-output -n ms bash scripts/train.sh'"
```

Monitor:

```bash
tmux attach -t coordexp_gauss_softce_prod_4gpu
```

`scripts/train.sh` accepts environment variables only. Do not pass the config as a positional argument.

## Expected Artifact Name

Config artifact subdir:

```text
compact_full_fullobj_random_sft_coord_gauss_softce_mix0p5_frac0p04_cap8_llm_lora_packed_r16_a32_bsz16_4epoch_tokenrows_v2
```

Relative artifact location under the configured output root:

```text
outputs/stage1_2b/recursive_detection_ce_latest/compact_full_fullobj_random_sft_coord_gauss_softce_mix0p5_frac0p04_cap8_llm_lora_packed_r16_a32_bsz16_4epoch_tokenrows_v2
```

Note: the YAML base currently sets `training.output_root` to `/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest`. If the other machine does not have `/data/CoordExp`, either run in that mounted layout or update the YAML output root intentionally before launch.

## Preflight Checklist On Other Machine

Run these from repo root before production:

```bash
python -m pytest \
  tests/test_sft_gaussian_coord_soft_ce.py \
  tests/test_grad_accum_loss_scale_mixin.py \
  tests/test_train_batch_contract.py \
  tests/test_packing_wrapper.py \
  tests/test_batch_extras_contract.py \
  tests/test_training_runtime_profile.py -q
```

Confirm data files exist:

```bash
test -s public_data/coco/rescale_32_1024_bbox/train.coord.jsonl
test -s public_data/coco/rescale_32_1024_bbox/val.coord.jsonl
test -d public_data/coco/rescale_32_1024_bbox/train
test -d public_data/coco/rescale_32_1024_bbox/val
```

Confirm GPUs are free:

```bash
nvidia-smi
```

## Residual Notes

- A broader docs-parity test involving `tests/test_stage1_metric_key_parity.py` still reports legacy undocumented `coord_diag/*` metrics. I did not fix that because it is unrelated to the packed Gaussian SFT scaling/contract issue.
- This handoff is only for retraining. For later eval, use free decode and avoid grammar-constrained decode.
