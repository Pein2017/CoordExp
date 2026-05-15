---
doc_id: progress.diagnostics.a2-eos-loosen-ablation-2026-05-14
layer: progress
doc_type: ablation-tracking-note
status: Concluded
domain: compact-detection
summary: Negative result for the clean A2+EOS-loosen ablation; EOS loosening alone increased bursty duplicate emission and did not improve valid-object recall on the stable compact-full support2 ET-RMP-CE baseline.
---

# A2 EOS-Loosen Clean Ablation

## Corrected Framing

A2 remains the current trusted compact-full anchor:

```text
configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2.yaml
```

A3/A4 are prefix-rollin mechanism probes, not fair causal replacements for A2.
A4 is A3+EOS loosen, not A2+EOS loosen. A2-vs-A3/A4 comparisons confound
prefix state distribution, reduced supervised-token density, support/balance
surface changes, EOS behavior, and COCO incomplete-label ambiguity.

The clean EOS ablation is A2E:

```text
A2 + objective.eos.eos_trust_weight
no prefix-rollin
no prefix masking
no reduced object-token supervision density
```

## Hypothesis

If A2E improves valid-object emission while preserving A2 duplicate/parse
stability, the useful part of A4 may be EOS/missing-label calibration rather
than prefix-rollin. If A2E increases duplicate or collapse behavior, EOS
loosening needs objectness or duplicate gating. If A2E does not help, A4's
behavior is not explained by EOS loosening alone.

## 2026-05-15 Rollout Evaluation Conclusion

Status: `Concluded`.

A2E is a negative ablation under the matched val200 greedy `rp=1.10` rollout
surface. Training completed cleanly, but free rollout shows that simply pushing
the EOS/continuation signal does not reveal more valid objects. Instead, it
increases concentrated duplicate bursts, empty predictions, and degenerate boxes.

Evaluation surfaces:

```text
checkpoint:
/data/CoordExp/output_remote/stage1_2b/recursive_detection_ce_latest/compact_full_support2_eos_loosen/compact-full-support2-eos-loosen-a2e/v0-20260514-062803/checkpoint-3664

cap1024 run:
/data/CoordExp/output_remote/infer/recursive_detection_ce_latest/compact_full_support2_eos_loosen_a2e_greedy_cap1024_rp110_ckpt3664_val200_bsz8_temp0_rp1p10_max1024_chatfix_8gpu

cap3084 run:
/data/CoordExp/output_remote/infer/recursive_detection_ce_latest/compact_full_support2_eos_loosen_a2e_greedy_cap3084_rp110_ckpt3664_val200_bsz8_temp0_rp1p10_max3084_chatfix_8gpu
```

Matched comparison against the existing A2 anchor artifacts:

| run | raw AP | raw AP50 | raw AP75 | raw F1@0.50 | guarded AP | guarded AP50 | guarded F1@0.50 | pred | degenerate | duplicate suppressed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A2 baseline cap1024/rp1.10 | 0.4184 | 0.5654 | 0.4385 | 0.6122 | 0.4045 | 0.5470 | 0.6070 | 1151 | 2 | 246 |
| A2E cap1024/rp1.10 | 0.3213 | 0.4106 | 0.3392 | 0.3258 | 0.3101 | 0.3945 | 0.3851 | 1285 | 45 | 695 |
| A2 baseline cap3084/rp1.10 | 0.4184 | 0.5654 | 0.4385 | 0.6122 | 0.4045 | 0.5470 | 0.6070 | 1151 | 2 | 246 |
| A2E cap3084/rp1.10 | 0.3270 | 0.4145 | 0.3493 | 0.3295 | 0.3159 | 0.4006 | 0.4021 | 1373 | 51 | 794 |

Parser/eval health signals:

```text
A2 cap1024: errors_total=1, empty_pred=0, eval degenerate=2
A2E cap1024: errors_total=107, empty_pred=52, eval degenerate=45
A2E cap3084: errors_total=1367, empty_pred=53, eval degenerate=51
```

Duplicate-burst profile:

```text
A2 baseline cap1024:
  affected records = 63
  duplicate-suppressed predictions = 246
  worst inspected/suppressed example = 30 / 24

A2E cap1024:
  affected records = 29
  duplicate-suppressed predictions = 695
  worst inspected/suppressed examples:
    000000016010.jpg = 128 / 123
    000000007511.jpg = 127 / 119
    000000013348.jpg = 127 / 117
    000000005001.jpg = 101 / 96

A2E cap3084:
  affected records = 28
  duplicate-suppressed predictions = 794
  worst inspected/suppressed examples:
    000000017899.jpg = 256 / 238
    000000011197.jpg = 158 / 128
    000000017959.jpg = 117 / 110
    000000013659.jpg = 117 / 96
```

Interpretation:

- EOS loosen alone is not a valid-object recall solution.
- The objective increases emission pressure, but the added mass is mostly
  unstable: empty outputs on many images and severe repeated-object bursts on a
  smaller set of images.
- Longer generation budget does not rescue the behavior. Cap3084 slightly
  improves raw AP over cap1024 (`0.3213 -> 0.3270`) but worsens prediction
  count, duplicate suppression, degenerate boxes, and invalid-geometry counters.
- A2 remains the trusted anchor. A2E should not be promoted as a replacement.
- Any future EOS/continuation relaxation should be paired with explicit
  objectness, duplicate, or termination gating.

## Configs

Production ablation config:

```text
configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_eos_loosen.yaml
```

DDP8 preflight config:

```text
configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_eos_loosen_ddp8_preflight.yaml
```

Core contract:

```text
objective.variant = random_permutation_et_rmp_ce
objective.trie_support_weight = 2.0
objective.trie_balance_weight = 1.0
objective.state_weighting = uniform_permutation
objective.normalization = semantic_image_bucket_balanced
objective.eos.eos_trust_weight.source = empirical_unlabeled_poisson_v0
experiment.surface = ablation
experiment.ablation_id = A2E-support2-eos-loosen
```

Expected production artifact root pattern:

```text
/data/CoordExp/output_remote/stage1_2b/recursive_detection_ce_latest/compact_full_support2_eos_loosen/compact-full-support2-eos-loosen-a2e/v0-<UTC>
```

Latest production artifact root:

```text
/data/CoordExp/output_remote/stage1_2b/recursive_detection_ce_latest/compact_full_support2_eos_loosen/compact-full-support2-eos-loosen-a2e/v0-20260514-062803
```

Launcher log:

```text
temp/a2e_eos_loosen_launch_logs/prod/compact_full_support2_eos_loosen-20260514T062522Z.log
```

## Implementation Status

Implemented as an opt-in A2 objective extension:

- `src/config/schema.py`: `objective.eos` is allowed for
  `random_permutation_et_rmp_ce`; prefix-only sections remain rejected.
- `src/detection/runtime.py`: latest detection dataset receives EOS trust
  config whenever `objective.eos` is present.
- `src/detection/dataset.py`: EOS trust is computed per image for A2 without
  adding roll-in metadata.
- `src/detection/objective.py`: EOS trust changes only assistant `<|im_end|>`
  token-target weights.

Verified unit/schema command:

```bash
conda run -n ms python -m pytest \
  tests/test_detection_training_dataset.py \
  tests/test_prefix_rollin_schema.py \
  -q
```

Observed:

```text
59 passed
```

## Smoke Result

DDP2 smoke was used initially while a true DDP8 window was unavailable. After
the user's earlier regular 8-GPU tests were terminated, the guarded launcher
observed a clean all-8-GPU window and completed the exact DDP8 smoke before
production launch.

Command:

```bash
PYTHONPATH=. \
OMP_NUM_THREADS=8 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
COORDEXP_TRAIN_HEARTBEAT=1 \
CUDA_VISIBLE_DEVICES=0,5 \
conda run --no-capture-output -n ms torchrun \
  --master_addr=127.0.0.1 \
  --master_port=29618 \
  --nproc_per_node=2 \
  -m src.sft \
  --config configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_eos_loosen_ddp8_preflight.yaml
```

Artifact root:

```text
temp/recursive_detection_ce_latest/output/compact_full_support2_eos_loosen_ddp8_preflight/smoke-compact-full-support2-eos-loosen-ddp8-preflight/v0-20260514-035850
```

Observed:

```text
world_size = 2
global_effective = 128
max_steps = 4
recursive_detection_ce/trie_support_weight = 2.0
recursive_detection_ce/trie_balance_weight = 1.0
recursive_detection_ce/eos_trust_weight logged on train/eval
train_heartbeat.rank0.jsonl and train_heartbeat.rank1.jsonl present
resolved_config.json, effective_runtime.json, runtime_env.json, run_metadata.json,
experiment_manifest.json, train_data_provenance.json, eval_data_provenance.json present
```

This DDP2 smoke remains useful as an early objective-path check. It is
superseded for launch gating by the exact DDP8 smoke below.

Exact DDP8 smoke command:

```bash
PYTHONPATH=. \
OMP_NUM_THREADS=8 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
COORDEXP_TRAIN_HEARTBEAT=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
conda run --no-capture-output -n ms torchrun \
  --master_addr=127.0.0.1 \
  --master_port=29620 \
  --nproc_per_node=8 \
  -m src.sft \
  --config configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_eos_loosen_ddp8_preflight.yaml
```

Exact DDP8 smoke artifact root:

```text
temp/recursive_detection_ce_latest/output/compact_full_support2_eos_loosen_ddp8_preflight/smoke-compact-full-support2-eos-loosen-ddp8-preflight/v1-20260514-062139
```

Observed in the DDP8 smoke artifact:

```text
exit code = 0
train_heartbeat.rank0.jsonl ... train_heartbeat.rank7.jsonl present
global_step/max_steps = 4/4
recursive_detection_ce/eos_trust_weight logged
recursive_detection_ce/trie_support_weight logged
recursive_detection_ce/trie_balance_weight logged
resolved_config.json, effective_runtime.json, runtime_env.json, run_metadata.json,
experiment_manifest.json, train_data_provenance.json, eval_data_provenance.json present
```

Note: the guard's first smoke-artifact verifier printed the older DDP2 `v0`
path because its glob matched `v0-*`. The helper was fixed to match `v*-*`, and
the DDP8 smoke artifact above was manually verified before final launch-state
reporting.

## Production Launch Command

Preferred wrapper:

```bash
config=configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_eos_loosen.yaml \
gpus=all \
COORDEXP_TRAIN_HEARTBEAT=1 \
train_log_dir=temp/prod_launch_a2e_eos_loosen_20260514 \
conda run --no-capture-output -n ms bash scripts/train.sh
```

Actual production launch used the guarded script's explicit DDP8 torchrun
command after a second clean all-8-GPU window:

```bash
PYTHONPATH=. \
OMP_NUM_THREADS=8 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
COORDEXP_TRAIN_HEARTBEAT=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
conda run --no-capture-output -n ms torchrun \
  --master_addr=127.0.0.1 \
  --master_port=29619 \
  --nproc_per_node=8 \
  -m src.sft \
  --config /data/CoordExp/.worktrees/boundary-tail-direction-gate/configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_eos_loosen.yaml
```

Launch evidence:

```text
2026-05-14T06:25:20Z clean 8-GPU window detected
2026-05-14T06:25:20Z production command started
world_size = 8
per_device_train_batch_size = 16
actual_global_effective_batch_size = 128
num_train_epochs = 4
save_delay_steps = 600
train_sample_count = 117247
val_sample_count = 4951
```

Current run artifact evidence:

```text
/data/CoordExp/output_remote/stage1_2b/recursive_detection_ce_latest/compact_full_support2_eos_loosen/compact-full-support2-eos-loosen-a2e/v0-20260514-062803/config_source.yaml
/data/CoordExp/output_remote/stage1_2b/recursive_detection_ce_latest/compact_full_support2_eos_loosen/compact-full-support2-eos-loosen-a2e/v0-20260514-062803/effective_runtime.json
/data/CoordExp/output_remote/stage1_2b/recursive_detection_ce_latest/compact_full_support2_eos_loosen/compact-full-support2-eos-loosen-a2e/v0-20260514-062803/experiment_manifest.json
/data/CoordExp/output_remote/stage1_2b/recursive_detection_ce_latest/compact_full_support2_eos_loosen/compact-full-support2-eos-loosen-a2e/v0-20260514-062803/logging.jsonl
/data/CoordExp/output_remote/stage1_2b/recursive_detection_ce_latest/compact_full_support2_eos_loosen/compact-full-support2-eos-loosen-a2e/v0-20260514-062803/resolved_config.json
/data/CoordExp/output_remote/stage1_2b/recursive_detection_ce_latest/compact_full_support2_eos_loosen/compact-full-support2-eos-loosen-a2e/v0-20260514-062803/run_metadata.json
/data/CoordExp/output_remote/stage1_2b/recursive_detection_ce_latest/compact_full_support2_eos_loosen/compact-full-support2-eos-loosen-a2e/v0-20260514-062803/train_heartbeat.rank0.jsonl ... train_heartbeat.rank7.jsonl
```

Resolved production config check:

```text
objective.variant = random_permutation_et_rmp_ce
objective.trie_support_weight = 2.0
objective.trie_balance_weight = 1.0
objective.state_weighting = uniform_permutation
objective.normalization = semantic_image_bucket_balanced
objective.eos.eos_trust_weight.source = empirical_unlabeled_poisson_v0
objective.rollin = null
experiment.surface = ablation
experiment.ablation_id = A2E-support2-eos-loosen
```

First training metric rows show the intended objective surface is active:

```text
global_step/max_steps = 1/3664 then 10/3664
recursive_detection_ce/eos_trust_weight = 0.32792658 then 0.28877722
recursive_detection_ce/eos_unweighted_ce logged
recursive_detection_ce/eos_weighted_loss logged
recursive_detection_ce/trie_support_weight = 2.0
recursive_detection_ce/trie_balance_weight = 1.0
```

## First Evaluation Path

After training, evaluate the best/final checkpoint on the same A2 val200
surface first:

- greedy cap1024/rp1.10;
- greedy cap3084/rp1.10 if cheap;
- raw and guarded metrics;
- duplicate guard report;
- parse/drop counters;
- prediction count;
- manual audit packet if emission or duplicate behavior changes.

Do not treat `val200` as full validation. Do not compare A2E against A3/A4 as a
fair prefix-rollin result; that requires supervised-token-matched prefix-rollin
or an A2/A3 mixture in a separate ablation.

## Historical Launch Blocker

Resolved: after the user's regular 8-GPU tests were terminated, the faster
guard observed a clean window, completed DDP8 smoke, and started production.

At 2026-05-14 04:05 UTC, an 8-GPU launch was blocked by hidden GPU contexts
outside this container's visible process table:

```text
0, 0 MiB, 0%
1, 28039 MiB, 100%
2, 24251 MiB, 0%
3, 0 MiB, 0%
4, 26905 MiB, 100%
5, 0 MiB, 0%
6, 24177 MiB, 100%
7, 26443 MiB, 55%
```

`nvidia-smi --query-compute-apps` showed PIDs with process name `[Not Found]`
for the occupied GPUs, and those PIDs were not visible via `ps` inside this
container. Do not launch the production A2E run under the 8-GPU name until all
eight GPUs are actually available.

Continuation audit at 2026-05-14 04:10 UTC reran the unit/config checks and
rechecked the node. Verification still passed:

```text
tests/test_detection_training_dataset.py
tests/test_prefix_rollin_schema.py
tests/test_eos_prior.py
tests/test_latest_training_config_contract.py
tests/test_recursive_detection_ce_sft_wiring.py
165 passed
```

Config materialization still showed:

```text
prod: variant=random_permutation_et_rmp_ce support=2.0 balance=1.0 eos=empirical_unlabeled_poisson_v0 rollin=None surface=ablation
smoke: variant=random_permutation_et_rmp_ce support=2.0 balance=1.0 eos=empirical_unlabeled_poisson_v0 rollin=None surface=smoke
```

The 8-GPU launch remained blocked:

```text
0, 26755 MiB, 100%
1, 0 MiB, 0%
2, 24251 MiB, 0%
3, 30483 MiB, 100%
4, 0 MiB, 0%
5, 24855 MiB, 0%
6, 25041 MiB, 0%
7, 0 MiB, 0%
```

`nvidia-smi --query-compute-apps` again reported `[Not Found]` process names
on the occupied GPUs, so the production launch is still pending a real 8-GPU
window.

Continuation recheck at 2026-05-14 04:13 UTC still did not have an 8-GPU
window:

```text
0, 26755 MiB, 54%
1, 0 MiB, 0%
2, 0 MiB, 0%
3, 0 MiB, 0%
4, 0 MiB, 0%
5, 24855 MiB, 70%
6, 25041 MiB, 31%
7, 0 MiB, 0%
```

Compute-app query:

```text
00000000:67:00.0, 1575994, [Not Found], 26746
00000000:E6:00.0, 1575995, [Not Found], 24846
00000000:E7:00.0, 1580592, [Not Found], 25032
```

The PIDs were not visible through `ps` inside this container. Full DDP8 smoke
and production launch remain blocked.

Continuation recheck at 2026-05-14 04:12 UTC still did not have an 8-GPU
window:

```text
0, 0 MiB, 0%
1, 27969 MiB, 100%
2, 0 MiB, 0%
3, 0 MiB, 0%
4, 0 MiB, 0%
5, 0 MiB, 0%
6, 25041 MiB, 57%
7, 31671 MiB, 100%
```

Compute-app query:

```text
00000000:68:00.0, 1586905, [Not Found], 27960
00000000:E7:00.0, 1580592, [Not Found], 25032
00000000:E8:00.0, 1586255, [Not Found], 31662
```

Full DDP8 smoke and production launch remain pending a genuinely clear node.

Continuation recheck at 2026-05-14 04:13 UTC still did not have an 8-GPU
window:

```text
0, 0 MiB, 0%
1, 27969 MiB, 51%
2, 0 MiB, 0%
3, 0 MiB, 0%
4, 29069 MiB, 100%
5, 0 MiB, 0%
6, 25041 MiB, 100%
7, 31671 MiB, 0%
```

Compute-app query:

```text
00000000:68:00.0, 1586905, [Not Found], 27960
00000000:E5:00.0, 1587880, [Not Found], 29060
00000000:E7:00.0, 1580592, [Not Found], 25032
00000000:E8:00.0, 1586255, [Not Found], 31662
```

Full DDP8 smoke and production launch remain blocked.

Continuation recheck at 2026-05-14 04:15 UTC still did not have an 8-GPU
window:

```text
0, 31875 MiB, 0%
1, 27969 MiB, 0%
2, 24025 MiB, 0%
3, 29655 MiB, 0%
4, 29069 MiB, 68%
5, 0 MiB, 0%
6, 0 MiB, 0%
7, 0 MiB, 0%
```

Compute-app query:

```text
00000000:67:00.0, 1591929, [Not Found], 31866
00000000:68:00.0, 1586905, [Not Found], 27960
00000000:6C:00.0, 1589204, [Not Found], 24016
00000000:6D:00.0, 1590050, [Not Found], 29646
00000000:E5:00.0, 1587880, [Not Found], 29060
```

`ps` inside this container did not see those PIDs. Full DDP8 smoke and
production launch remain blocked by external CUDA contexts.

Guarded launch watcher armed at 2026-05-14 04:17 UTC:

```bash
tmux new-session -d -s a2e_eos_loosen_guard \
  'cd /data/CoordExp/.worktrees/boundary-tail-direction-gate && \
   bash temp/a2e_eos_loosen_wait_and_launch.sh'
```

The watcher waits for a genuinely clear 8-GPU window, runs the exact DDP8 smoke
with
`configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_eos_loosen_ddp8_preflight.yaml`,
verifies the smoke artifact contains EOS trust/support metrics, waits for a
clear 8-GPU window again, and only then launches production with
`configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_eos_loosen.yaml`.

Watcher status immediately after launch: waiting before DDP8 smoke; external
`[Not Found]` CUDA contexts still occupy multiple GPUs. Guard log:

```text
temp/a2e_eos_loosen_launch_logs/guard.log
```

Watcher recheck at 2026-05-14 04:18 UTC remained blocked before DDP8 smoke:

```text
0, 31875 MiB, 0%
1, 0 MiB, 0%
2, 24025 MiB, 0%
3, 29655 MiB, 0%
4, 0 MiB, 0%
5, 28931 MiB, 0%
6, 26561 MiB, 100%
7, 28001 MiB, 48%
```

The tmux guard is still alive as `a2e_eos_loosen_guard`; no smoke or production
run has started yet.

Watcher recheck at 2026-05-14 04:21 UTC remained blocked before DDP8 smoke:

```text
0, 0 MiB, 0%
1, 28139 MiB, 38%
2, 0 MiB, 0%
3, 0 MiB, 0%
4, 0 MiB, 0%
5, 0 MiB, 0%
6, 26561 MiB, 0%
7, 28001 MiB, 0%
```

Manual GPU check at 2026-05-14 04:22 UTC also remained blocked:

```text
0, 0 MiB, 0%
1, 28139 MiB, 99%
2, 0 MiB, 0%
3, 0 MiB, 0%
4, 28319 MiB, 78%
5, 0 MiB, 0%
6, 26561 MiB, 100%
7, 28001 MiB, 74%
```

No A2E DDP8 smoke or production artifact has been created yet by the watcher.
Process check at this point found only the guard script matching A2E; no
`src.sft` A2E training process was running. `fuser` was not available in this
container, and the external NVML PIDs remained invisible through `ps`.

Watcher recheck at 2026-05-14 04:23 UTC remained blocked before DDP8 smoke:

```text
0, 24613 MiB, 100%
1, 28139 MiB, 0%
2, 28761 MiB, 100%
3, 27555 MiB, 11%
4, 28319 MiB, 0%
5, 0 MiB, 0%
6, 26561 MiB, 0%
7, 0 MiB, 0%
```

Manual GPU check at 2026-05-14 04:24 UTC showed seven occupied GPUs:

```text
0, 24613 MiB, 100%
1, 28139 MiB, 43%
2, 28761 MiB, 85%
3, 27555 MiB, 100%
4, 28319 MiB, 3%
5, 28757 MiB, 0%
6, 26561 MiB, 58%
7, 0 MiB, 0%
```

The guard remains alive and no A2E `src.sft` process has started.

Watcher recheck at 2026-05-14 04:28 UTC remained blocked before DDP8 smoke:

```text
0, 0 MiB, 0%
1, 24891 MiB, 100%
2, 28761 MiB, 87%
3, 27555 MiB, 100%
4, 28319 MiB, 13%
5, 28757 MiB, 0%
6, 30415 MiB, 100%
7, 26191 MiB, 87%
```

Manual GPU check at 2026-05-14 04:28 UTC also remained blocked on GPUs 1-7.
Only the guard script matched A2E; no A2E `src.sft` process had started.

Watcher recheck at 2026-05-14 04:34 UTC remained blocked before DDP8 smoke:

```text
0, 27579 MiB, 100%
1, 0 MiB, 0%
2, 30055 MiB, 0%
3, 25995 MiB, 100%
4, 28767 MiB, 0%
5, 23523 MiB, 100%
6, 0 MiB, 0%
7, 27743 MiB, 59%
```

Manual GPU check at 2026-05-14 04:34 UTC also remained blocked:

```text
0, 27579 MiB, 100%
1, 0 MiB, 0%
2, 30055 MiB, 0%
3, 25995 MiB, 0%
4, 28767 MiB, 24%
5, 23523 MiB, 91%
6, 0 MiB, 0%
7, 27743 MiB, 100%
```

The guard remains the only A2E process. Exact DDP8 smoke and production remain
pending.

Watcher recheck at 2026-05-14 04:35 UTC remained blocked before DDP8 smoke:

```text
0, 0 MiB, 0%
1, 31063 MiB, 100%
2, 30055 MiB, 0%
3, 25995 MiB, 0%
4, 28767 MiB, 78%
5, 23523 MiB, 88%
6, 0 MiB, 0%
7, 27743 MiB, 85%
```

Manual GPU check at 2026-05-14 04:35 UTC also remained blocked:

```text
0, 0 MiB, 0%
1, 31063 MiB, 0%
2, 30055 MiB, 100%
3, 25995 MiB, 29%
4, 28767 MiB, 32%
5, 23523 MiB, 37%
6, 0 MiB, 0%
7, 27743 MiB, 40%
```

Still no A2E `src.sft` process.

Long monitor through 2026-05-14 05:20 UTC never observed a clean 8-GPU window.
The watcher was still blocked before DDP8 smoke at attempt 63:

```text
0, 0 MiB, 0%
1, 30101 MiB, 74%
2, 24055 MiB, 70%
3, 0 MiB, 0%
4, 28663 MiB, 0%
5, 0 MiB, 0%
6, 24353 MiB, 34%
7, 28679 MiB, 56%
```

Manual GPU check at 2026-05-14 05:20 UTC also remained blocked:

```text
0, 0 MiB, 0%
1, 30101 MiB, 0%
2, 24055 MiB, 100%
3, 0 MiB, 0%
4, 28663 MiB, 31%
5, 0 MiB, 0%
6, 24353 MiB, 0%
7, 28679 MiB, 3%
```

The guard remains the only A2E process; exact DDP8 smoke and production have
not started.

Watcher recheck at 2026-05-14 05:21 UTC remained blocked before DDP8 smoke:

```text
0, 0 MiB, 0%
1, 30101 MiB, 0%
2, 24055 MiB, 40%
3, 26265 MiB, 100%
4, 28663 MiB, 46%
5, 0 MiB, 0%
6, 24353 MiB, 100%
7, 0 MiB, 0%
```

Manual GPU check at 2026-05-14 05:21 UTC also remained blocked, and the guard
remained the only A2E process.

Extended monitor through 2026-05-14 05:57 UTC still did not observe a clean
8-GPU window. The watcher remained blocked before DDP8 smoke at attempt 99:

```text
0, 29285 MiB, 84%
1, 27415 MiB, 0%
2, 0 MiB, 0%
3, 27963 MiB, 0%
4, 29537 MiB, 100%
5, 0 MiB, 0%
6, 29713 MiB, 0%
7, 24977 MiB, 0%
```

Manual GPU check at 2026-05-14 05:57 UTC also remained blocked:

```text
0, 29285 MiB, 0%
1, 27415 MiB, 0%
2, 0 MiB, 0%
3, 27963 MiB, 0%
4, 29537 MiB, 0%
5, 0 MiB, 0%
6, 29713 MiB, 100%
7, 24977 MiB, 31%
```

The node is closer to draining at times, but no all-8-GPU clear interval has
been observed by the guard. DDP8 smoke and production remain unstarted.

Watcher recheck at 2026-05-14 05:58 UTC showed the node re-filled before the
guard could start smoke:

```text
0, 29285 MiB, 100%
1, 27415 MiB, 0%
2, 27967 MiB, 0%
3, 27963 MiB, 100%
4, 29537 MiB, 15%
5, 27157 MiB, 0%
6, 29713 MiB, 100%
7, 24977 MiB, 55%
```

No DDP8 smoke launch occurred by attempt 101.

Extended monitor through 2026-05-14 06:10 UTC still did not observe a clean
8-GPU window. The watcher was still blocked before DDP8 smoke at attempt 113:

```text
0, 31289 MiB, 100%
1, 0 MiB, 0%
2, 27881 MiB, 12%
3, 0 MiB, 0%
4, 0 MiB, 0%
5, 28605 MiB, 62%
6, 0 MiB, 0%
7, 0 MiB, 0%
```

Manual GPU check at 2026-05-14 06:11 UTC also remained blocked:

```text
0, 31289 MiB, 95%
1, 0 MiB, 0%
2, 27881 MiB, 100%
3, 25185 MiB, 68%
4, 0 MiB, 0%
5, 0 MiB, 0%
6, 0 MiB, 0%
7, 30593 MiB, 100%
```

The node repeatedly drops to a few occupied GPUs but has not produced an
all-8-GPU clear guard tick. No A2E `src.sft` process has started.

Guard adjustment at 2026-05-14 06:13 UTC:

- killed and restarted `a2e_eos_loosen_guard`,
- kept the same strict all-8-GPU clear predicate and same smoke-before-prod
  sequencing,
- changed polling from 60 seconds to 5 seconds because the node appears to
  cycle through short low-pressure intervals,
- throttled blocked-attempt logging with `LOG_EVERY_ATTEMPTS=12` to avoid
  flooding the log while polling faster.

Restart command:

```bash
tmux new-session -d -s a2e_eos_loosen_guard \
  'cd /data/CoordExp/.worktrees/boundary-tail-direction-gate && \
   POLL_SECONDS=5 LOG_EVERY_ATTEMPTS=12 \
   bash temp/a2e_eos_loosen_wait_and_launch.sh'
```

Immediate restarted-guard status: still blocked before DDP8 smoke; no A2E
`src.sft` process was running.

Faster-polling guard recheck at 2026-05-14 06:15 UTC was still blocked before
DDP8 smoke. The restarted guard logged every 12th 5-second attempt; attempt 24
was still blocked:

```text
0, 31289 MiB, 79%
1, 24423 MiB, 0%
2, 0 MiB, 0%
3, 25185 MiB, 0%
4, 28211 MiB, 100%
5, 24877 MiB, 97%
6, 29303 MiB, 0%
7, 30593 MiB, 100%
```

Manual GPU check at 2026-05-14 06:15 UTC also remained blocked:

```text
0, 0 MiB, 0%
1, 24423 MiB, 100%
2, 0 MiB, 0%
3, 25185 MiB, 8%
4, 28211 MiB, 26%
5, 24877 MiB, 0%
6, 29303 MiB, 49%
7, 30593 MiB, 100%
```

The faster guard remains the only A2E process; no DDP8 smoke has started.

Watcher recheck at 2026-05-14 04:45 UTC remained blocked before DDP8 smoke:

```text
0, 30851 MiB, 38%
1, 27269 MiB, 100%
2, 27227 MiB, 14%
3, 0 MiB, 0%
4, 27699 MiB, 0%
5, 24605 MiB, 100%
6, 24915 MiB, 4%
7, 24041 MiB, 65%
```

Manual GPU check at 2026-05-14 04:45 UTC also remained blocked:

```text
0, 30851 MiB, 0%
1, 27269 MiB, 71%
2, 27227 MiB, 0%
3, 0 MiB, 0%
4, 27699 MiB, 62%
5, 24605 MiB, 100%
6, 24915 MiB, 83%
7, 24041 MiB, 100%
```

The guard remains the only A2E process; DDP8 smoke has not started.

Watcher recheck at 2026-05-14 04:46 UTC remained blocked before DDP8 smoke:

```text
0, 30851 MiB, 0%
1, 27269 MiB, 100%
2, 27227 MiB, 54%
3, 0 MiB, 0%
4, 0 MiB, 0%
5, 24605 MiB, 27%
6, 24915 MiB, 44%
7, 24041 MiB, 0%
```

Manual GPU check at 2026-05-14 04:46 UTC also remained blocked:

```text
0, 30851 MiB, 0%
1, 27269 MiB, 0%
2, 27227 MiB, 0%
3, 0 MiB, 0%
4, 0 MiB, 0%
5, 24605 MiB, 83%
6, 24915 MiB, 0%
7, 24041 MiB, 100%
```

No A2E `src.sft` process was running.

Watcher recheck at 2026-05-14 04:53 UTC remained blocked before DDP8 smoke:

```text
0, 29207 MiB, 0%
1, 26043 MiB, 62%
2, 23433 MiB, 100%
3, 0 MiB, 0%
4, 24553 MiB, 100%
5, 23451 MiB, 0%
6, 0 MiB, 0%
7, 29291 MiB, 0%
```

Manual GPU check just before the guard tick at 2026-05-14 04:53 UTC also
remained blocked, and no A2E `src.sft` process was running.

Long monitor through 2026-05-14 05:08 UTC never observed a clean 8-GPU window.
The watcher was still blocked before DDP8 smoke at attempt 51:

```text
0, 25973 MiB, 100%
1, 0 MiB, 0%
2, 28443 MiB, 40%
3, 30969 MiB, 0%
4, 26143 MiB, 65%
5, 24039 MiB, 46%
6, 30825 MiB, 100%
7, 31475 MiB, 41%
```

Manual GPU check at 2026-05-14 05:09 UTC also remained blocked:

```text
0, 0 MiB, 0%
1, 0 MiB, 0%
2, 28443 MiB, 100%
3, 30969 MiB, 73%
4, 26143 MiB, 100%
5, 24039 MiB, 100%
6, 30825 MiB, 100%
7, 31475 MiB, 100%
```

The guard remains the only A2E process. No exact DDP8 smoke or production
launch has started.

Watcher recheck at 2026-05-14 05:09 UTC remained blocked before DDP8 smoke:

```text
0, 0 MiB, 0%
1, 0 MiB, 0%
2, 28443 MiB, 0%
3, 30969 MiB, 0%
4, 26143 MiB, 100%
5, 24039 MiB, 100%
6, 30825 MiB, 0%
7, 31475 MiB, 0%
```

Manual GPU check at 2026-05-14 05:09 UTC also remained blocked:

```text
0, 0 MiB, 0%
1, 25603 MiB, 43%
2, 28443 MiB, 100%
3, 0 MiB, 0%
4, 26143 MiB, 22%
5, 24039 MiB, 100%
6, 30825 MiB, 100%
7, 31475 MiB, 100%
```

Still no A2E `src.sft` process.

Watcher recheck at 2026-05-14 04:51 UTC remained blocked before DDP8 smoke:

```text
0, 0 MiB, 0%
1, 0 MiB, 0%
2, 23433 MiB, 74%
3, 30893 MiB, 100%
4, 24553 MiB, 0%
5, 23451 MiB, 85%
6, 0 MiB, 0%
7, 29291 MiB, 0%
```

Manual GPU check at 2026-05-14 04:51 UTC also remained blocked:

```text
0, 0 MiB, 0%
1, 0 MiB, 0%
2, 23433 MiB, 47%
3, 30893 MiB, 0%
4, 24553 MiB, 0%
5, 23451 MiB, 100%
6, 0 MiB, 0%
7, 29291 MiB, 100%
```

Additional host-level introspection did not expose an owner: `/proc/driver/nvidia/clients`
was empty, NVML accounting returned no rows, and the current NVML PIDs had no
`/proc/<pid>` entries in this container.

Watcher recheck at 2026-05-14 04:52 UTC remained blocked before DDP8 smoke:

```text
0, 0 MiB, 0%
1, 26043 MiB, 74%
2, 23433 MiB, 0%
3, 30893 MiB, 100%
4, 24553 MiB, 0%
5, 23451 MiB, 16%
6, 0 MiB, 0%
7, 29291 MiB, 100%
```

No A2E `src.sft` process was running.
