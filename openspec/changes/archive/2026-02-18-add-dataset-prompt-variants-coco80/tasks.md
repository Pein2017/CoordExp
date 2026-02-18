## 1. Registry and Prompt Assembly

- [x] 1.1 Add a centralized prompt-variant registry module under `src/config/` with `default` and `coco_80` entries.
- [x] 1.2 Freeze the `coco_80` class list in code aligned to `public_data/coco/raw/categories.json` snapshot names/order.
- [x] 1.3 Update `src/config/prompts.py` to assemble dense prompts from the selected variant while preserving existing `default` behavior.
- [x] 1.4 Ensure summary-mode prompts remain unchanged by variant selection.
- [x] 1.5 Implement strict unknown-variant validation with descriptive error text that includes both unknown key and available variant keys.

## 2. Training and Inference Integration

- [x] 2.1 Read `custom.extra.prompt_variant` in training prompt resolution (`src/config/loader.py`) and pass it to variant-aware prompt builders.
- [x] 2.2 Add `infer.prompt_variant` support in inference pipeline loading (`src/infer/pipeline.py`) and `InferenceConfig`.
- [x] 2.3 Refactor inference message builders in `src/infer/engine.py` to use the same variant-aware prompt resolver across HF and vLLM paths.
- [x] 2.4 Record resolved `prompt_variant` in inference artifacts (`resolved_config.json` and summary output metadata).
- [x] 2.5 Preserve fusion prompt override precedence and verify no behavior regression for dataset-level `prompt_user`/`prompt_system`.

## 3. Tests and Verification

- [x] 3.1 Add tests for prompt variant resolution: default fallback, deterministic repeated resolution, and cross-surface parity.
- [x] 3.2 Add tests that validate `coco_80` content constraints (compact canonical list, 80 classes, no duplicated class names).
- [x] 3.3 Add regression tests for unknown-variant error payload (includes unknown key and available keys).
- [x] 3.4 Add/extend inference and fusion regression tests to confirm variant wiring, artifact metadata emission, and override precedence.
- [x] 3.5 Run: `PYTHONPATH=. conda run -n ms python -m pytest -q tests/test_prompt_variants.py tests/test_infer_batch_decoding.py tests/test_unified_infer_pipeline.py tests/test_fusion_config.py`.
- [x] 3.6 Run OpenSpec validation: `openspec validate add-dataset-prompt-variants-coco80 --type change --strict`.

## 4. Docs and Reproducibility Records

- [x] 4.1 Update `docs/data/README.md` with variant overview and training/inference YAML key paths.
- [x] 4.2 Update `public_data/coco/README.md` with `coco_80` usage and parity guidance for inference.
- [x] 4.3 Add reproducibility notes at `openspec/changes/add-dataset-prompt-variants-coco80/implementation-notes.md` recording config path, run name, seed, selected prompt variant, and output artifact paths.
