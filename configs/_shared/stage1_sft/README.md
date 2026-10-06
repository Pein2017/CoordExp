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

The former compact detection config family used a separate
`DetectionTrainingConfig` schema and could not extend these overlays.
These are historical authoring boundaries; see the [config router](../../README.md)
for this checkout's maintained training surface and qualification boundary.
