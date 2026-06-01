# Shared Latest Detection Overlays

This directory contains authoring snippets for latest compact detection configs
that parse through `DetectionTrainingConfig`.

Quarantined legacy/comparator launch configs under
`configs/archive/detection_scene_clean_break/stage1/recursive_detection_ce/`
are preserved for historical inspection and do not extend these snippets. Edits here do not affect production or smoke launch
configs until their `extends` chains are migrated. New canonical compact
teacher-forcing configs belong under
`configs/stage1/detection_teacher_forcing/`.

Detection owns these top-level sections:

- `data`
- `prompt`
- `detection_template`
- `token_rows`
- `objective`
- `packing`
- `evaluation`
- `validation`

`custom.*` is rejected for detection. Do not extend legacy shared
Stage-1 SFT overlays from `configs/_shared/datasets/` or
`configs/_shared/prompts/`.

Authoring notes:

- `data.object_ordering` uses `random_permutation`, not legacy `random`.
- Top-level `packing` is the semantic latest-detection owner.
- `training.packing` and `training.eval_packing` remain runtime adapter fields
  until the loader derives them from latest `packing`.
- Latest recursive detection currently requires both semantic and runtime
  packing to stay disabled.
- `objectives/random_order_sft.yaml` is coord-token latest compact SFT only; it
  does not cover raw-text norm1000 profiles
  geometry-loss ablations.
