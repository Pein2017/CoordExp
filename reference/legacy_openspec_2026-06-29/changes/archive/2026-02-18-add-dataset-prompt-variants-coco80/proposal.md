## Why

CoordExp currently uses one global dense prompt, but COCO stage-1 runs are intentionally restricted to the canonical 80 classes. Without an explicit prompt-variant mechanism, training and inference can drift in labeling policy, which weakens reproducibility and eval validity for COCO-focused experiments.

## What Changes

- Add a centralized prompt-variant registry with a backward-compatible `default` variant and a new `coco_80` variant.
- Add YAML-first variant selectors:
  - training: `custom.extra.prompt_variant`
  - inference pipeline: `infer.prompt_variant`
- Route both training prompt resolution and inference message construction through the same variant-aware prompt resolver.
- Define deterministic resolver behavior: resolving the same variant key with the same mode/order inputs MUST yield identical system/user prompt text across repeated calls and across training/inference surfaces.
- Use a frozen canonical COCO class source for `coco_80` (aligned to `public_data/coco/raw/categories.json` names/order at snapshot time) to prevent drift from local data-path differences.
- Keep summary mode behavior unchanged and keep Qwen3-VL chat-template compatibility.
- Keep runtime behavior prompt-only for class restriction (no new runtime label filtering/drop logic).
- Add focused tests and docs for variant resolution and COCO usage.

## Capabilities

### New Capabilities
- `dataset-prompt-variants`: Provide deterministic, dataset-aware prompt variants (starting with COCO-80) that are selectable from YAML and applied consistently across training and inference.

### Modified Capabilities
- None.

## Impact

- Affected code:
  - `src/config/prompts.py`
  - `src/config/loader.py`
  - `src/infer/engine.py`
  - `src/infer/pipeline.py`
  - new registry module under `src/config/`
  - tests under `tests/`
  - docs under `docs/data/` and `public_data/coco/`
- API/config impact:
  - New optional config keys: `custom.extra.prompt_variant` and `infer.prompt_variant`.
  - No new CLI flags.
- YAML placement examples (for docs and configs):
  - training (`configs/stage1/*.yaml`):
    - `custom.extra.prompt_variant: coco_80`
  - inference (`configs/infer/*.yaml`):
    - `infer.prompt_variant: coco_80`
- Correctness/reproducibility:
  - Variant choice becomes explicit and versioned in config/artifacts.
  - Inference artifacts (`resolved_config.json` and summary outputs) record the resolved prompt variant so audits can verify train/infer prompt-policy parity.
  - COCO experiments gain a stable closed-class prompt policy while preserving current defaults elsewhere.
