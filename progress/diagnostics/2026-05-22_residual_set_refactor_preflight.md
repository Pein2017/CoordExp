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

Scope: deterministic CPU/integration preflight for the Stage-2 residual-set /
UL correction refactor. This is not a model-quality claim and does not replace
a GPU training smoke.

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
- prepared rows overlap the configured first 8 training samples by
  sample/image/path join keys.

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
# 7 passed in 11.74s
```

GPU smoke:

- Not launched in this preflight note. The plan command requests 4 GPUs via
  `gpus=0,1,2,3 conda run -n ms bash scripts/train.sh`; treat that as a
  separate high-cost smoke before making model-quality or production-training
  claims.
