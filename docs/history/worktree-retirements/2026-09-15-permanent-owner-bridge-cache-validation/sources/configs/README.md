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

- `configs/coordexp_swift/prod/`: canonical Swift production-style training
  configs.
- `configs/coordexp_swift/smoke/`: canonical Swift implementation and
  promotion smokes.
- `configs/coordexp_swift/infer/`: canonical Swift inference configs.
- `configs/coordexp_swift/deepspeed/`: Swift backend helper configuration;
  production support remains governed by the Swift contracts.

The former `configs/stage1/`, `configs/stage2/`, and related root-level
training families are MS-Swift/mainline compatibility or historical surfaces.
They are not the current `main` training route. The old route is preserved on
the `ms-swift` branch and should be used only for explicit legacy reproduction.

## Policy

- Do not add one YAML per debugging attempt. Put one-off launch variations under
  `temp/` or pass temporary overrides through a copied local file.
- Keep a new tracked training YAML only when it represents a durable production
  profile, a reusable smoke/preflight overlay, or a named ablation whose result
  should remain reproducible.
- Prefer a small prod config plus a smoke overlay over separate tiny/DDP/single
  GPU files for every variant.
- If a config exists only to remember an old run, record the artifact path in
  `research/` or a current handoff note and remove the runnable YAML from
  `configs/`. Do not create new `progress/` notes.
- The retired Stage-2 AB/two-channel config root is removed; do not add active
  configs under old `stage2_ab` or `stage2_two_channel` names.
- `configs/analysis/`, `configs/infer/`, `configs/eval/`, `configs/postop/`,
  and `configs/bench/` are tool/config inputs, not training profile families.
  Clean them separately from Stage-1/Stage-2 training YAML.
