# Change: Add coord-offset tuning for Qwen3-VL

## Why
- Coord tokens were added after pretrain; their embedding and lm_head rows remain frozen under current dlora recipes, limiting learning signal.
- We need a safe way to tune only coord token rows while keeping the rest of the vocabulary and backbone stable, and to give them dedicated learning rates.

## What Changes
- Introduce a coord-offset adapter that adds trainable offsets for coord token IDs on both the token embedding and lm_head, leaving base weights frozen.
- Keep existing DoRA/LoRA on all other linear layers (LLM, vision tower, aligner) via ms-swift.
- Add optimizer support for dedicated LR buckets for coord-offset parameters.
- Provide YAML knobs to enable/disable the feature and configure coord IDs and LRs.

## Impact
- Affects training pipeline entry (`src/sft.py`), coord-token utilities, optimizer plugin, and configs for dlora runs.
- New docs/specs describing coord-offset behavior and configuration.
