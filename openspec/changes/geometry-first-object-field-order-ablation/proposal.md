## Why

For autoregressive multimodal causal models (e.g., Qwen-VL series), the order of emitted JSON fields can materially affect decoding dynamics, exposure bias, and token-level supervision patterns in detection training.

CoordExp currently hardcodes per-object field order as `desc -> geometry` in multiple training paths:
- stage-1 dataset serialization,
- stage-2 Channel-A teacher-forced payload construction,
- stage-2 Channel-B FN append serialization.

To run controlled ablations and answer "which field order is better for AR detection behavior," we need infrastructure that:
- is config-first and reproducible,
- keeps current behavior as the default baseline,
- changes only per-object field order (not object instance ordering),
- applies consistently across stage-1 and stage-2 training pipelines.

## What Changes

- Add a new YAML config key: `custom.object_field_order`.
  - Allowed values: `desc_first` (default), `geometry_first`.
  - `geometry_first` means the existing geometry key (`bbox_2d` or `poly`) is emitted before `desc` within each object payload; no synthetic key named `geometry` is introduced.
  - Terminology bridge: this is a generalized form of "bbox-first" from `progress/full_idea.md`; they are equivalent when geometry is `bbox_2d`.
- Keep object instance ordering unchanged:
  - `custom.object_ordering` (`sorted` / `random`) remains the only control for object sequence.
- Apply `custom.object_field_order` consistently to:
  - stage-1 assistant payload serialization,
  - stage-2 Channel-A teacher-forced payload serialization,
  - stage-2 Channel-B FN append serialization used to build `Y_train`.
- Adjust dense prompt wording so instruction order matches configured field order:
  - `geometry_first` prompts explicitly request the geometry key (`bbox_2d`/`poly`) before `desc`,
  - `desc_first` remains baseline wording.
- Preserve strict JSON and geometry contracts (no coordinate drop/reorder, no geometry semantics changes).
- Keep assistant outputs constrained to `desc` + one geometry key (`bbox_2d` or `poly`); optional source metadata such as `poly_points` is not emitted into assistant text payloads.
- Keep rollout parsing/matching behavior unchanged except for accepting either field order as valid schema order.

## Capabilities

### New Capabilities

- Configurable per-object field serialization order for training targets via `custom.object_field_order`.

### Modified Capabilities

- `rollout-matching-sft`: FN append serialization order becomes config-driven while preserving raw rollout object appearance order.
- `stage2-ab-training`: Channel-A/Channel-B constructed training payloads honor the same configured field order.

## Impact

- Schema/config:
  - `src/config/schema.py`
  - `src/config/loader.py`
- Prompt generation:
  - `src/config/prompts.py`
- Stage-1 serialization:
  - `src/datasets/builders/jsonlines.py`
  - `src/datasets/dense_caption.py`
  - `src/datasets/unified_fusion_dataset.py`
- Stage-2 serialization:
  - `src/trainers/stage2_ab_training.py`
  - `src/trainers/rollout_matching/parsing.py`
  - `src/trainers/rollout_matching_sft.py` (config plumbing only if needed)
- Tests:
  - dataset runtime serialization tests
  - stage2/rollout append serialization tests
  - config strict-validation tests

No new CLI flags. No model architecture changes. No change to object enumeration policy.
