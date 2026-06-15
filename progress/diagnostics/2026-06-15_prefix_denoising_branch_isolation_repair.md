# Prefix-Denoising Branch-Isolation Repair

Date: 2026-06-15

Scope: repair and launch-health note for the prefix-denoising SFT V1 branch after
audit review found that historical packed smokes forwarded clean and noisy
branches as one concatenated causal row.

## Repair Summary

The prefix-denoising objective now replays clean and noisy branch segments as
separate model forwards before computing CE and optional local-window KL. Static
packing still groups multiple hybrid samples for dataloader throughput, but the
loss path no longer lets a noisy branch attend to clean-branch answer tokens from
the same packed item.

Other launch-gate fixes:

- hard CE and local KL fail fast on non-finite logits/losses;
- degenerate GT boxes are skipped with `skip_reason=degenerate_gt_bbox` instead
  of aborting dataset construction;
- prefix-denoising dataset eligibility summaries are recorded in
  `runtime.prefix_denoising.dataset.{train,eval}`;
- static packing fingerprints identify the objective as
  `prefix_denoising_sft` and include the prefix-denoising dataset summary;
- `docs/training/METRICS.md` now documents the required `llm_loss`, branch CE,
  top1/top5 token accuracy, and KL metric families.

## Verification

CPU regression and integration slice:

```bash
env -u CODEX_CI python -m pytest -q \
  tests/test_prefix_denoising_geometry.py \
  tests/test_prefix_denoising_builder.py \
  tests/test_prefix_denoising_collator.py \
  tests/test_prefix_denoising_loss.py \
  tests/test_prefix_denoising_metrics.py \
  tests/test_prefix_denoising_runtime_integration.py \
  tests/test_prefix_denoising_config_contract.py \
  tests/test_teacher_forcing_config_contract.py \
  tests/test_training_runtime_sft_integration.py \
  tests/test_training_runtime_profile.py \
  tests/test_packing_attention_backend_gate.py \
  tests/test_stage1_static_packing_runtime_config.py \
  tests/test_packing_cache_fingerprints.py \
  tests/test_packing_template_contracts.py
```

Result: `285 passed in 2.95s`.

Post-provenance focused slice:

```bash
env -u CODEX_CI python -m pytest -q \
  tests/test_stage1_static_packing_runtime_config.py::test_static_packing_fingerprint_includes_dataset_source_identity \
  tests/test_stage1_static_packing_runtime_config.py::test_static_packing_fingerprint_marks_prefix_denoising_objective \
  tests/test_packing_cache_fingerprints.py
```

Result: `24 passed in 1.58s`.

Compile check:

```bash
python -m py_compile \
  src/detection/prefix_denoising/dataset.py \
  src/detection/prefix_denoising/builder.py \
  src/detection/prefix_denoising/loss.py \
  src/detection/packing.py \
  src/trainers/metrics/prefix_denoising.py \
  src/sft.py \
  tests/test_prefix_denoising_builder.py \
  tests/test_prefix_denoising_loss.py \
  tests/test_prefix_denoising_runtime_integration.py \
  tests/test_training_runtime_sft_integration.py \
  tests/test_stage1_static_packing_runtime_config.py
```

Result: exit 0.

Patch hygiene:

```bash
git diff --check
```

Result: exit 0.

Cfg-only smoke checks:

```bash
env -u CODEX_CI python -m src.sft --config configs/stage1/detection_teacher_forcing/smoke/compact_full_prefix_denoising_ce_only_tiny.yaml --cfg-only
env -u CODEX_CI python -m src.sft --config configs/stage1/detection_teacher_forcing/smoke/compact_full_prefix_denoising_kl_w0p05_tiny.yaml --cfg-only
```

Result: both returned `status=ok`.

GPU smoke checks were run sequentially on `CUDA_VISIBLE_DEVICES=0` while the
node already had active training processes on all GPUs.

CE-only:

```bash
env -u CODEX_CI CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -m src.sft --config configs/stage1/detection_teacher_forcing/smoke/compact_full_prefix_denoising_ce_only_tiny.yaml
```

Artifact root:

```text
temp/detection_teacher_forcing/output/compact_full_prefix_denoising_ce_only_tiny/smoke-compact-full-prefix-denoising-ce-only-tiny/v18-20260615-022333
```

Result: completed 2/2 steps. Final metric row included:

```text
llm_loss=11.35028648
prefix_denoising/clean_full/loss/ce=11.3485527
prefix_denoising/noisy_full/loss/ce=11.35202122
prefix_denoising/global/loss/ce_balanced=11.35028648
prefix_denoising/global/token_acc/full_vocab/top1=0.2972973
prefix_denoising/global/token_acc/full_vocab/top5=0.44144145
```

KL-on:

```bash
env -u CODEX_CI CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -m src.sft --config configs/stage1/detection_teacher_forcing/smoke/compact_full_prefix_denoising_kl_w0p05_tiny.yaml
```

Artifact root:

```text
temp/detection_teacher_forcing/output/compact_full_prefix_denoising_kl_w0p05_tiny/smoke-compact-full-prefix-denoising-kl-w0p05-tiny/v13-20260615-022423
```

Result: completed 2/2 steps. Final metric row included:

```text
llm_loss=11.35028648
prefix_denoising/kl/local_window/raw=0.0000076
prefix_denoising/kl/local_window/weighted=0.00000038
prefix_denoising/kl/local_window/site_count=16
prefix_denoising/kl/local_window/candidate_site_count=16
prefix_denoising/kl/local_window/support_bin_count=16.5625
prefix_denoising/global/token_acc/full_vocab/top1=0.2972973
prefix_denoising/global/token_acc/full_vocab/top5=0.44144145
```

## Artifact Checks

Both smoke runs recorded the repaired runtime payload:

```text
runtime.prefix_denoising.enabled=true
runtime.prefix_denoising.packing_enabled=true
runtime.prefix_denoising.packing_mode=static
runtime.prefix_denoising.dataset.train.source_rows=4
runtime.prefix_denoising.dataset.train.eligible_rows=4
runtime.prefix_denoising.dataset.train.skipped_rows=0
runtime.prefix_denoising.dataset.eval.source_rows=1
runtime.prefix_denoising.dataset.eval.eligible_rows=1
runtime.prefix_denoising.dataset.eval.skipped_rows=0
```

Both static packing plans used:

```text
setup_fingerprint_sha256=68cee581d038bf801e45e7e5668aa3d033bf37120ced4d1a3a978fb859d8ca9d
fingerprint.detection_packing_contract.metadata.objective_variant=prefix_denoising_sft
fingerprint.detection_packing_contract.metadata.normalization_policy=branch_balanced_clean_noisy_ce
fingerprint.detection_packing_contract.metadata.loss_mask_version=prefix_denoising_branch_isolated_hard_ce_v1
fingerprint.prefix_denoising_dataset_summary.skip_counters={}
raw_plan=[[0, 1, 2, 3]]
avg_fill=0.9335
skipped_long=0
```

The smoke run metadata recorded `git_dirty=true` at
`git_sha=e74b5be9188755334266a78324140960d241cb3e` because the repair was
verified before committing the branch. Treat these smoke artifacts as
post-repair launch evidence but not clean-head provenance.

## Launch-Health Read

Status: healthy for V1 launch wiring after branch-isolation repair.

This is still tiny launch evidence only. It verifies that the repaired objective
launches with Qwen3-VL, static packing, branch-isolated CE, standard
`llm_loss`/top1/top5 monitors, local-window KL diagnostics, static-packing
objective fingerprinting, and durable prefix-denoising dataset skip summaries.
It is not rollout evidence and should not be interpreted as exposure-bias
improvement.
