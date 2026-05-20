---
doc_id: progress.diagnostics.residual_set_stage2_smoke_2026_05_20
layer: progress
doc_type: diagnostic-note
status: historical
domain: training
summary: Tiny Stage-2 residual-set correction smoke evidence for ckpt3664 configs.
tags: [stage2, residual-set, smoke, ckpt3664]
updated: 2026-05-20
---

# Residual-Set Stage-2 Smoke

Scope: tiny smoke only, not full validation. Both runs used single-GPU HF rollout
through `scripts/train.sh`, compact-full coord-token rows, checkpoint
`/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_et_rmp_ce_support2_bsz16_4epoch_tokenrows_v2/compact-full-et-rmp-ce-support2-bsz16-4epoch-tokenrows-v2/v0-20260504-071356/checkpoint-3664`, and seed `17`.

Configs:

- `configs/stage2_two_channel/smoke/compact_full_residual_set_ckpt3664_hf_1step.yaml`
- `configs/stage2_two_channel/smoke/compact_full_residual_set_ckpt3664_hf_thorough.yaml`

Schema note: current `stage2_ab.pipeline.objective` canonical order requires
`token_ce` before `residual_set_correction`, so these B-only residual smokes keep
`token_ce` enabled on Channel-A only and `residual_set_correction` enabled on
Channel-B only.

Commands:

```bash
config=configs/stage2_two_channel/smoke/compact_full_residual_set_ckpt3664_hf_1step.yaml gpus=0 conda run -n ms bash scripts/train.sh
config=configs/stage2_two_channel/smoke/compact_full_residual_set_ckpt3664_hf_thorough.yaml gpus=0 conda run -n ms bash scripts/train.sh
```

Artifact roots:

- `output/stage2_ab/smoke/compact_full_residual_set_ckpt3664_hf_1step/smoke_1step-compact_full-residual_set-ckpt3664-hf-unconstrained/v0-20260520-215526`
- `output/stage2_ab/smoke/compact_full_residual_set_ckpt3664_hf_thorough/smoke_thorough-compact_full-residual_set-ckpt3664-hf-unconstrained/v0-20260520-220301`

Evidence:

- 1-step train: residual atom count `126`; invalid rollout `0`; parse dropped invalid/ambiguous `0/0`; parse truncated rate `0`; strict drop invalid `0`; prompt-token mismatch rate `0`; UL atoms `0`; unlabeled-consistent count `0`.
- 1-step eval: precision `0.5952381`; recall `0.46296296`; F1 `0.52083333`; mAP `0.40028878`; eval parse drops `0/0`; COCO eval ok `1`.
- Thorough 4-step train: residual atom counts `152, 148, 100, 201`; invalid rollout `0` on all logged steps; parse drops `0/0`; strict drop invalid `0`; prompt-token mismatch rate `0`; UL atoms `0`; unlabeled-consistent counts `1, 0, 0, 1`.
- Thorough eval: precision `0.6185567`; recall `0.54054054`; F1 `0.57692308`; mAP `0.46263948`; eval parse drops `0/0`; COCO eval ok `1`.

No `target_position/logit_position` invariant failures or tracebacks were found
in the launcher logs. No separate UL-cluster artifact file was materialized in
these tiny runs; strict UL promotion did not fire (`ul_atom_count=0`).

