# Design Notes: Add Fusion Config Training (Qwen3-VL-Compatible Schema)

## Reference Implementation
Use Qwen3-VL as the behavioral reference:
- `Qwen3-VL/src/datasets/fusion.py`
- `Qwen3-VL/src/datasets/fusion_types.py`
- `Qwen3-VL/src/datasets/unified_fusion_dataset.py`
- `Qwen3-VL/docs/data/UNIFIED_FUSION_DATASET.md`

## CoordExp Constraints / Deltas
- CoordExp prompts are global English prompts selected by:
  - ordering (`custom.object_ordering`)
  - coord representation (`custom.coord_tokens.enabled`)
  Therefore, `template:` in fusion configs is required/validated for config compatibility and typo-checking, but does not necessarily imply a different template class (v1 keeps a single shared template instance).
- CoordExp defaults to packing (`training.packing: true`). Fusion should remain compatible, but docs must call out recommended settings and any pitfalls.
- YAML prompt overrides in training configs are disabled; fusion config MAY still accept per-dataset `user_prompt`/`system_prompt` as dataset-level overrides (not via `prompts:` section).

## Proposed Integration Points
- `src/config/schema.py`:
  - Allow `custom.fusion_config` and relax the requirement that `custom.train_jsonl` must always be set.
- `src/sft.py`:
  - If `custom.fusion_config` is set, instantiate `FusionCaptionDataset` for training and for evaluation (eval includes entries with non-null `val_jsonl`).
- `src/datasets/*`:
  - Replace/upgrade the existing deprecated fusion helpers with a Qwen3-VL-compatible *schema* but CoordExp-specific *mixing semantics*:
    - accept `targets` and `sources` containers
    - treat all entries uniformly (no target/source split) for training quotas
    - compute quotas per-entry from its own pool size and `ratio`
  - Add generic dataset wrapper (`jsonl`) so fusion is not blocked by wrapper registry for new datasets.
