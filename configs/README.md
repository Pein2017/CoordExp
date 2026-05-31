---
doc_id: configs.index
layer: configs
doc_type: router
status: canonical
domain: training
updated: 2026-05-25
---

# Configs

`configs/` is for durable, runnable configuration. Keep experiments small and
named around the research decision they represent, not around every launch
attempt.

## Training Surfaces

- `configs/stage1/sft_base.yaml`: baseline Stage-1 runtime base.
- `configs/stage1/profiles/`: small legacy Stage-1 SFT profile set. Keep only
  non-redundant 2B/4B hyperparameter surfaces that remain useful as baselines
  or comparison anchors.
- `configs/stage1/detection_teacher_forcing/`: canonical
  `stage1_detection_teacher_forcing` Stage-1 detection teacher-forcing surface.
  New compact-full teacher-forcing production and smoke configs belong here.
- `configs/stage1/recursive_detection_ce/prod/`: legacy/comparator compact
  recursive-detection CE production and comparison configs. Keep only for
  migration history or explicitly named comparisons.
- `configs/stage1/recursive_detection_ce/ablation/`: legacy/comparator
  mechanism-level recursive-detection CE ablations.
- `configs/stage1/recursive_detection_ce/smoke/`: migration smoke and
  preflight overlays for the legacy recursive-detection CE configs.
- `configs/stage2_rollout_correction/`: active Stage-2 rollout-correction
  surface. These configs use rollout prefix plus GT/residual correction and the
  canonical `stage2_rollout_correction` namespace.

## Policy

- Do not add one YAML per debugging attempt. Put one-off launch variations under
  `temp/` or pass temporary overrides through a copied local file.
- Keep a new tracked training YAML only when it represents a durable production
  profile, a reusable smoke/preflight overlay, or a named ablation whose result
  should remain reproducible.
- Prefer a small prod config plus a smoke overlay over separate tiny/DDP/single
  GPU files for every variant.
- If a config exists only to remember an old run, record the artifact path in
  `progress/` and remove the runnable YAML from `configs/`.
- The retired Stage-2 AB/two-channel config root is removed; do not add active
  configs under old `stage2_ab` or `stage2_two_channel` names.
- `configs/analysis/`, `configs/infer/`, `configs/eval/`, `configs/postop/`,
  and `configs/bench/` are tool/config inputs, not training profile families.
  Clean them separately from Stage-1/Stage-2 training YAML.
