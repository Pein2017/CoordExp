---
doc_id: progress.audits.instance-trie-gaussian-smoke-behavior-2026-05-14
layer: progress
doc_type: audit
status: concluded
domain: training
summary: Smoke and 8-GPU preflight behavior audit for Instance-Trie Gaussian SoftCE.
updated: 2026-05-14
---

# Instance-Trie Gaussian SoftCE Smoke Behavior Audit (2026-05-14)

Scope: `/data/CoordExp/.worktrees/instance-trie-gaussian-softce` on branch `codex/instance-trie-gaussian-softce`.

Status: smoke/preflight audit concluded. This is production-readiness evidence for the 8-GPU launch path, not a final validation-quality result.

## Contract

`A5-instance-trie-gaussian` keeps the compact-full support2 recursive detection setup unchanged except for coordinate-token supervision:

- schema/control tokens: hard CE
- description and trie-entry ambiguity: ET-RMP support + balance
- coordinate tokens: pure full-vocabulary SoftCE against an active-branch remaining-instance Gaussian mixture
- no coordinate support/balance weighting
- no oracle decoding, reranking, or inference/eval change

## Evidence

Target-shape audit:

```text
artifact: temp/instance_trie_gaussian/target_shape_audit.json
target_distribution: instance_trie_gaussian
candidate_leak_count: 0
nonfinite_target_count: 0
teacher_candidate_missing_count: 0
real_probe_status: ok
real_probe repeated_desc: orange
real_probe x1 candidate_count/effective/top1: 4 / 4.0 / 0.25
```

Tiny smoke:

```text
run:
  temp/recursive_detection_ce_latest/output/compact_full_instance_trie_gaussian_softce_a5_tiny/smoke-compact-full-instance_trie_gaussian_softce_a5-tiny/v0-20260514-121444
launcher log:
  temp/instance_trie_gaussian/train_logs/compact_full_support2_instance_trie_gaussian_softce_a5_tiny-20260514T121300Z.log
train_loss: 12.67943001
coord_soft_ce/is_instance_trie_gaussian: 1.0
coord_soft_ce/balance_loss: 0.0
x1/y1/x2/y2 effective candidate counts: 2.77777767 / 1.11110139 / 1.0 / 1.0
```

DDP8 preflight:

```text
run:
  temp/recursive_detection_ce_latest/output/compact_full_instance_trie_gaussian_softce_a5_ddp8_preflight/smoke-compact-full-instance_trie_gaussian_softce_a5-ddp8-preflight/v0-20260514-122519
launcher log:
  temp/instance_trie_gaussian/train_logs/compact_full_support2_instance_trie_gaussian_softce_a5_ddp8_preflight-20260514T122339Z.log
resolved batch shape:
  per_device_train_batch_size: 16
  world_size: 8
  global_effective: 128
  max_steps: 4
train loss series: 11.4318, 11.7557, 11.0908, 10.4684
eval loss series: 11.9755, 11.4867, 10.9156, 10.6649
coord weighted-loss series:
  train: 21.6260, 21.4509, 20.7999, 20.0667
  eval: 21.9242, 21.0397, 20.0072, 19.5494
max trainer memory: 60.61 GiB
```

Focused tests:

```bash
conda run -n ms python -m pytest -q \
  tests/test_instance_trie_gaussian_coord_softce.py \
  tests/test_iou_gibbs_coord_softce.py \
  tests/test_recursive_detection_ce_target_builder.py \
  tests/test_recursive_detection_ce_loss_adapter.py \
  tests/test_latest_training_config_contract.py \
  tests/test_instance_trie_gaussian_config_diff.py \
  tests/test_instance_trie_gaussian_target_shape_audit.py
```

Result: `174 passed in 7.73s`.

## Findings

- P0/P1: none found from target-shape audit, tiny smoke, focused tests, or DDP8 preflight artifacts.
- P2: coordinate metric names still include legacy aggregate words such as `support_bin_count` and `support_mixture` for backward-compatible logging. Slot-level metrics use the clearer candidate/posterior names and should be preferred for this ablation.
- P3: launcher log contains expected warnings about deprecated `NCCL_ASYNC_ERROR_HANDLING` and `torch_dtype`; they did not block DDP startup or training.

## Confirmed OK

- Resolved configs enable `objective.coord_soft_ce.target_distribution: instance_trie_gaussian`.
- The DDP8 preflight used 8 ranks with per-device batch size 16 and global effective batch 128.
- `coord_soft_ce/is_instance_trie_gaussian` is `1.0` in train and eval logs.
- `coord_soft_ce/balance_loss` is `0.0`, confirming coordinate tokens are not using support/balance weighting.
- No non-finite values were found in `logging.jsonl` or `train_heartbeat.rank0.jsonl` for tiny or DDP8 runs.
- The DDP8 launcher log contains no `Traceback`, `RuntimeError`, `ChildFailedError`, or CUDA OOM.
- Posterior diagnostics show the intended causal sharpening pattern: x1 remains multi-candidate, while y1/x2/y2 are near single-instance after teacher-forced prefix conditioning.
- GPUs were free after the DDP8 preflight completed.

## Residual Risks

- This evidence is smoke/preflight scope only. It does not prove final AP, full-val behavior, or free-rollout quality.
- The no-training target-shape probe confirms localized multi-peak targets, but production training can still reveal optimization or decoding behavior not visible in four DDP steps.
- Production launch should use the checked-in production config and retain the launcher log redirection from `scripts/train.sh`.

## Launch Readiness

The setup is production-ready for an 8-GPU launch path from a runtime and objective-wiring standpoint, pending explicit user approval to start the real production run.
