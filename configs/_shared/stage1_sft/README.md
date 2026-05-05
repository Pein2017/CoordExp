# Shared Stage-1 SFT Overlays

This directory documents the legacy Stage-1 SFT overlay family.

Existing shared overlays under `configs/_shared/datasets/` and
`configs/_shared/prompts/` are authored for legacy `TrainingConfig` profiles.
They use the `custom.*` namespace for dataset, ordering, and prompt behavior,
including:

- `custom.train_jsonl`
- `custom.val_jsonl`
- `custom.object_ordering`
- `custom.object_field_order`
- `custom.extra.prompt_variant`

Latest compact detection configs must not extend those legacy overlays. Latest
detection uses `LatestDetectionTrainingConfig` and top-level `data`, `prompt`,
`detection_template`, `token_rows`, `objective`, `packing`, `evaluation`, and
`validation` sections instead.
