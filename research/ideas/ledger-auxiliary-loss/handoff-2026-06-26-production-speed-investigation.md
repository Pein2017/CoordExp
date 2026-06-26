---
title: Ledger Production Speed Investigation Handoff
updated: 2026-06-26
branch: codex/ledger-auxiliary-loss
claim_scope: handoff
---

# Ledger Production Speed Investigation Handoff

This handoff is for continuing the ledger auxiliary-loss branch on another host
after the saved-adapter blocker was fixed, the 128 warm-start smoke passed, and
the first full production restart exposed a serious throughput / GPU-utilization
problem.

## Checkout

Current branch:

```bash
cd /data/CoordExp/.worktrees/ledger-auxiliary-loss
git branch --show-current
# codex/ledger-auxiliary-loss
```

Remote branch:

```text
origin/codex/ledger-auxiliary-loss
```

On another host, from an existing CoordExp clone:

```bash
cd /data/CoordExp
git fetch origin
git worktree add .worktrees/ledger-auxiliary-loss origin/codex/ledger-auxiliary-loss
cd .worktrees/ledger-auxiliary-loss
```

If the worktree already exists:

```bash
cd /data/CoordExp/.worktrees/ledger-auxiliary-loss
git fetch origin
git pull --ff-only origin codex/ledger-auxiliary-loss
```

Use the `ms` environment and local ms-swift checkout for raw Python probes:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ms
export PYTHONPATH=/data/ms-swift:$PWD:${PYTHONPATH:-}
```

## Current Code State

The current branch contains the ledger adapter-save/reload fix and regression
tests. Important files:

- `src/sft.py`
  - `_ensure_coverage_ledger_head_trainable_after_prepare_model(...)`
  - promotes `coverage_ledger_head` into active PEFT `modules_to_save` after
    warm-start adapter load / `prepare_model`.
- `src/trainers/metrics/teacher_forcing.py`
  - `_trainer_return_outputs(...)`
  - adapts the rich ledger capture result into a HF Trainer-compatible
    `(loss, logits)` output for eval/prediction steps.
- `tests/test_coverage_ledger_head_install.py`
  - regression for warm-start PEFT promotion of `coverage_ledger_head`.
- `tests/test_coverage_ledger_bridge_integration.py`
  - regression for HF eval output shape.

The key cause of the original production failure was:

- `ms-swift` freezes the model before loading `args.adapters`.
- The warm-start pure-CE adapter only had `token_embeddings_adapter` in
  `modules_to_save`.
- `coverage_ledger_head` was installed, but remained frozen / not PEFT-saveable
  after `prepare_model`.
- The existing training guard correctly failed with
  `coverage_ledger_head was not active after prepare_model`.

## Verification Already Completed

Targeted regression suite:

```bash
PYTHONPATH=/data/ms-swift:$PWD /root/miniconda3/envs/ms/bin/python -m pytest \
  tests/test_coverage_ledger_bridge_integration.py \
  tests/test_coverage_ledger_head_install.py \
  tests/test_coverage_ledger_smoke_configs.py \
  tests/test_final_checkpoint_coverage_ledger.py \
  tests/tokens/test_token_embeddings_adapter_optimizer.py \
  tests/test_infer_checkpoint_resolution.py \
  tests/test_infer_batch_decoding.py::test_hf_adapter_inference_drops_training_only_ledger_head_before_swift -q
```

Result:

```text
60 passed in 9.07s
```

Whitespace check:

```bash
git diff --check
```

Result: passed.

Exact production-derived 128 warm-start smoke passed after the fixes:

```text
log:
/data/CoordExp/.worktrees/ledger-auxiliary-loss/temp/train_logs/ledger_fix_prod_warm_128b_20260626_015114/coverage_ledger_closed_hard_sft_prod_warm_128-20260626T015115Z.log

run root:
/data/CoordExp/.worktrees/ledger-auxiliary-loss/temp/detection_teacher_forcing/output/coverage_ledger_closed_hard_sft_prod_warm_128_adapter_save/smoke-prod-warm-coverage-ledger-closed-hard-sft-128/v2-20260626-015252

checkpoint:
/data/CoordExp/.worktrees/ledger-auxiliary-loss/temp/detection_teacher_forcing/output/coverage_ledger_closed_hard_sft_prod_warm_128_adapter_save/smoke-prod-warm-coverage-ledger-closed-hard-sft-128/v2-20260626-015252/checkpoint-2
```

Smoke evidence:

- loaded warm-start adapter:
  `/data/CoordExp/outputs/stage1_2b/adapter_views/pure_ce_sorted_checkpoint_928_token_embeddings_adapter-5e5zj73f`
- `Promoted coverage_ledger_head into active PEFT modules_to_save after prepare_model`
- `Model after tuner: PeftModelForCausalLM`
- `eval_loss: 1.697914481163025`
- checkpoint `adapter_config.json` had
  `modules_to_save = ['token_embeddings_adapter', 'coverage_ledger_head']`
- saved checkpoint tensors included both `token_embeddings_adapter` and the
  three `coverage_ledger_head` projection weights.

Inference adapter view check also passed:

- `prepare_adapter_checkpoint_for_inference(...)` dropped only
  `coverage_ledger_head`
- inference-view `modules_to_save = ['token_embeddings_adapter']`
- inference-view safetensors had `ledger_key_count = 0`
- token adapter tensors remained present.

## Production Restart That Was Killed

Production config:

```text
configs/stage1/detection_teacher_forcing/prod/coverage_ledger_closed_hard_sft.yaml
```

Launch command shape:

```bash
cd /data/CoordExp/.worktrees/ledger-auxiliary-loss
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ms
export PYTHONPATH=/data/ms-swift:$PWD:${PYTHONPATH:-}
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
config=configs/stage1/detection_teacher_forcing/prod/coverage_ledger_closed_hard_sft.yaml \
  gpus=0,1,2,3,4,5,6,7 \
  train_log_dir=/data/CoordExp/.worktrees/ledger-auxiliary-loss/temp/train_logs/ledger_prod_restart_20260626_021219 \
  bash scripts/train.sh
```

tmux session was:

```text
ledger_prod_restart_20260626_021219
```

It was killed by request at:

```text
2026-06-26 02:38:46 UTC
```

Launcher log:

```text
/data/CoordExp/.worktrees/ledger-auxiliary-loss/temp/train_logs/ledger_prod_restart_20260626_021219/coverage_ledger_closed_hard_sft-20260626T021220Z.log
```

Partial run root:

```text
/data/CoordExp/outputs/stage1_2b/detection_teacher_forcing/coverage_ledger_closed_hard_sft_bsz32_2epoch/coverage-ledger-closed-hard-sft-bsz32-2epoch/v8-20260626-021435
```

No checkpoint was created. This is expected because `save_delay_steps=600`.
The partial run root only contains startup manifests and `logging.jsonl`.

After killing the tmux session:

- no tmux sessions remained;
- no lingering `torchrun`, `src.sft`, or `scripts/train.sh` processes remained;
- all 8 GPUs returned to `0 MiB` and `0%` utilization.

## Production Config Facts

Resolved production run characteristics:

- model:
  `model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`
- warm-start adapter:
  `/data/CoordExp/outputs/stage1_2b/adapter_views/pure_ce_sorted_checkpoint_928_token_embeddings_adapter-5e5zj73f`
- template:
  `compact_object_box_closed`
- parser:
  `strict_expected`
- global max length:
  `12000`
- static train packs:
  `14832`
- static eval packs:
  `632`
- training epochs:
  `2`
- max steps:
  `928`
- effective batch:
  `32`
- per-device train batch:
  `1`
- grad accumulation:
  `4`
- world size:
  `8`
- LR:
  `2.0e-5`
- packing length precompute workers:
  `16`

Important: the ms-swift args dump includes `packing=False`, but this is not the
custom static-packing truth. The later custom logs show:

```text
Packing enabled (static): length=12000 ...
Static packing stats: packs=14832 avg_fill=0.950 single_long=0 skipped_long=0
Static eval packing stats: packs=632 avg_fill=0.950 single_long=0 skipped_long=0
```

## Why The Production Run Was Stopped

The user observed long stretches of imbalanced GPU utilization, with some GPUs
around `1%` while others were near `100%`. Training speed was also far too slow
for the intended 2-epoch production run.

Observed progress before kill, parsed from the log:

```text
Train:   1%|          | 5/928 [17:36<57:40:09, 224.93s/it]
```

Earlier first scalar metrics:

```text
global_step/max_steps: 1/928
loss: 6.42231941
grad_norm: 14.85559273
teacher_forcing/loss/coverage_ledger_auxiliary/contribution: 0.13533109
teacher_forcing/loss/token_type_mass/contribution: 0.01239158
teacher_forcing/loss/continuation_margin/contribution: 0.01839574
teacher_forcing/loss/total: 1.47024882
remaining_time: 2d 7h 23m 11s
train_speed(iter/s): 0.004649
```

The run got past the previous correctness blockers:

- JSONL precheck passed for train and val.
- static packing completed with 16 workers, no serial fallback.
- adapter was loaded for training.
- `coverage_ledger_head` promotion fired.
- `Model after tuner: PeftModelForCausalLM`.
- training loop started and produced metrics.

The current blocker is therefore performance / rank imbalance, not adapter
saving.

## Suggested Next Investigation

Do not immediately relaunch the full production run without a speed diagnostic.
Recommended first passes:

1. Reproduce for only a few steps with the same production config surface and
   record per-rank timing. Keep it in tmux, but stop after enough evidence.
2. Instrument or inspect static-pack distribution by rank:
   - packed text length;
   - object count;
   - image count;
   - vision token / pixel burden;
   - number of samples per pack;
   - whether the slow rank repeatedly receives heavier packs.
3. Check whether static packing equalizes sequence fill but not vision/object
   compute. The train pack fill is high (`avg_fill=0.950`), but equal text fill
   may still hide large image/object imbalance.
4. Profile the ledger auxiliary path separately:
   - time spent in coverage-ledger forward capture;
   - time spent in coverage-ledger loss construction;
   - CPU/GPU synchronization points;
   - per-rank all-reduce wait.
5. Compare with a pure-CE baseline run using the same static packs and 12k
   length. If baseline also shows the same imbalance, focus on packing /
   multimodal sequence distribution. If baseline is healthy, focus on ledger
   capture/loss.
6. Consider a tiny diagnostic config that keeps:
   - same model;
   - same warm-start adapter;
   - same `compact_object_box_closed` template;
   - same `max_length=12000`;
   - same world size 8;
   - `max_steps` around 8-16;
   - more frequent logging.

Useful runtime checks:

```bash
tmux ls
pgrep -af "torchrun|src.sft|coverage_ledger_closed_hard_sft|scripts/train.sh"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
tail -c 4000 /data/CoordExp/.worktrees/ledger-auxiliary-loss/temp/train_logs/ledger_prod_restart_20260626_021219/coverage_ledger_closed_hard_sft-20260626T021220Z.log | cat -v
```

Relevant code surfaces to inspect:

- `src/datasets/wrappers/packed_caption.py`
- `src/sft.py`
- `src/trainers/metrics/teacher_forcing.py`
- `src/training/bridge.py`
- `src/training/coverage_ledger/`

Relevant tests already added or updated:

- `tests/test_coverage_ledger_head_install.py`
- `tests/test_coverage_ledger_bridge_integration.py`
- `tests/test_coverage_ledger_smoke_configs.py`
- `tests/test_final_checkpoint_coverage_ledger.py`
- `tests/test_infer_checkpoint_resolution.py`
- `tests/test_infer_batch_decoding.py::test_hf_adapter_inference_drops_training_only_ledger_head_before_swift`

## Relaunch Gate

Before relaunching full production, the next agent should be able to answer:

- Is the utilization imbalance caused by static-pack compute skew, ledger loss
  overhead, DDP/all-reduce waiting, or something else?
- What is the expected step time for the chosen 12k packed config?
- Does a short diagnostic run show stable all-rank utilization?
- Is there a safe config/code change that preserves the research meaning while
  making the 2-epoch run feasible?

The adapter-saving mechanism itself is not the current blocker.
