## 1. Implementation
- [x] Update fusion dataclasses to support Qwen3-VL-style containers (`targets` + `sources`) and optional fields used in upstream configs.
- [x] Update fusion config parser to accept YAML/JSON with `extends` support and strict validation (unique dataset IDs, required fields, template validation).
- [x] Upgrade `src/datasets/unified_fusion_dataset.py` to:
  - [x] load all datasets listed across `targets` + `sources` and treat them uniformly for training
  - [x] compute per-dataset quotas (`quota_i = round(len(pool_i) * ratio_i)`; ratio defaults to 1.0)
  - [x] build eval dataset from any dataset entry with a non-null `val_jsonl` (missing/null skips)
- [x] Add/extend dataset wrappers:
  - [x] `vg` wrapper (public source defaults)
  - [x] generic `jsonl` wrapper for arbitrary JSONLs (so fusion is not blocked by wrapper registry)
- [x] Wire fusion into training entrypoint:
  - [x] relax `CustomConfig` validation so either `train_jsonl` OR `fusion_config` is required
  - [x] in `src/sft.py`, instantiate `FusionCaptionDataset` when `custom.fusion_config` is provided
  - [x] build an eval dataset from fusion config when `fusion_config` is set (do not require `custom.val_jsonl`)
- [x] Add example configs under `configs/fusion/` and a training config that uses them (e.g., LVIS + VG mix).

## 2. Documentation
- [x] Add `docs/data/FUSION_DATASET.md` explaining:
  - schema (`targets`/`sources`, ratios, sampling rules)
  - interaction with packing + coord-token mode
  - recommended patterns for LVIS+VG
  - when to prefer offline merge (`public_data/scripts/merge_jsonl.py`)
- [x] Update existing docs that state fusion is disabled/deprecated to reflect the new optional capability.

## 3. Validation & Tests
- [x] Add unit tests for fusion config parsing (extends merge, unique name validation, missing field errors).
- [x] Add a small smoke test that instantiates fusion dataset from a tiny config and iterates a few samples (no GPU).
- [x] Verify `scripts/inspect_chat_template.py` runs on a fused sample (coord-token mode).
- [x] Verify that a fusion training config can be loaded without errors (config-only / dataset-only smoke).
