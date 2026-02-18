# CoordExp Suggested Commands

- `PYTHONPATH=. conda run -n ms python -m src.sft --config configs/<run>.yaml [--base_config configs/base.yaml]` – run a YAML-first SFT training experiment (single dataset or extend with `custom.fusion_config`).
- `PYTHONPATH=. conda run -n ms python scripts/tools/expand_coord_vocab.py --src <base> --dst <coord-expanded>` – regenerate coord tokens before training to avoid ID drift.
- `PYTHONPATH=. conda run -n ms python scripts/tools/inspect_chat_template.py --jsonl <file> --index 0` – preview how a JSONL record renders through the Qwen3-VL chat template and prompt.
- `PYTHONPATH=. conda run -n ms python public_data/scripts/validate_jsonl.py <path.jsonl>` – verify JSONL records adhere to the canonical contract before packing.
- `bash public_data/scripts/lvis_full_pipeline.sh` (with optional `MAX_BLOCKS`, `FACTOR`, etc.) – run the LVIS raw-to-packed data pipeline for reproducible geometry.
- `git status --porcelain` and `git diff` – check working tree cleanliness before editing; repo safety requires avoiding destructive commands unless explicitly requested.
