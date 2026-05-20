---
doc_id: configs.index
layer: configs
doc_type: router
status: canonical
domain: training
updated: 2026-05-20
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
- `configs/stage1/recursive_detection_ce/prod/`: compact
  recursive detection production/comparison configs.
- `configs/stage1/recursive_detection_ce/ablation/`: current
  mechanism-level recursive detection ablations that are still worth launching
  or comparing.
- `configs/stage1/recursive_detection_ce/smoke/`: minimal smoke and
  preflight overlays for the retained recursive detection configs.
- `configs/stage2_two_channel/`: compact Stage-2 two-channel surface. This is
  intentionally narrow while Stage-2 is not the main generation path.

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
- `configs/analysis/`, `configs/infer/`, `configs/eval/`, `configs/postop/`,
  and `configs/bench/` are tool/config inputs, not training profile families.
  Clean them separately from Stage-1/Stage-2 training YAML.
