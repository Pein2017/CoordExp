# CoordExp Project Overview

## Purpose
- Extend Qwen3-VL with coordinate-specialized tokens, expectation-based continuous box decoding, and order-invariant matching to advance open-vocabulary detection/grounding while remaining within the ms-swift SFT pipeline.
- Keep experiments YAML/config-driven, reproducible, and ready for paper-ready runs (single-dataset by default, fusion via `custom.fusion_config`).

## Tech Stack
- Python 3 on Linux, running inside `conda run -n ms` with `ms-swift` (trainer/orchestrator) plus Hugging Face Qwen3-VL checkpoints.
- Geometry-focused tooling under `src/datasets/geometry.py` and shared dataset/data-contract helpers in `public_data/`.
- Config-first YAML loader (`configs/` + `src/config/`) and ms-swift for training/inference.

## Structure
- `src/`: training stack (`src/sft.py` entrypoint), dataset builders, config loader, callbacks, inference/eval helpers.
- `configs/`: YAML experiment definitions (base, dlora, LoRA, fusion overrides).
- `scripts/` and `public_data/scripts/`: utilities for vocab expansion, JSONL validation, data packing, LVIS pipeline, etc.
- `docs/`: runbooks, data contracts, standards (`docs/standards/CODE_STYLE.md`), and evaluation/training guides.

## Key Commands
- `PYTHONPATH=. conda run -n ms python -m src.sft --config <yaml> [--base_config <yaml>] [--debug]` (main training entrypoint).
- `PYTHONPATH=. conda run -n ms python scripts/tools/expand_coord_vocab.py --src <base> --dst <expanded>` (coord vocab prep).
- `PYTHONPATH=. conda run -n ms python scripts/tools/inspect_chat_template.py --jsonl <file> --index <n>` (preview prompt rendering).
- `PYTHONPATH=. conda run -n ms python public_data/scripts/validate_jsonl.py <path>` (JSONL contract check).
- `bash public_data/scripts/lvis_full_pipeline.sh` (LVIS data prep reproducible pipeline).
