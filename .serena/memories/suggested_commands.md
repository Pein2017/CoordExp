# Command Cribsheet (Memory)

Role separation:
- Memory role: short, high-frequency command recall.
- Canonical docs for full run procedures: `docs/data/README.md`, `docs/training/STAGE2_RUNBOOK.md`, `public_data/README.md`.
- Update trigger: when script CLIs or recommended launcher patterns change.

Environment baseline:
- Prefer `PYTHONPATH=. conda run -n ms ...`

Training:
- `PYTHONPATH=. conda run -n ms python -m src.sft --config <yaml> [--base_config <yaml>] [--debug|--verbose]`
- `PYTHONPATH=. conda run -n ms torchrun --nproc_per_node 4 -m src.sft --config <yaml> [--base_config <yaml>]`

Validation/sanity:
- `conda run -n ms python public_data/scripts/validate_jsonl.py <jsonl>`
- `conda run -n ms python scripts/tools/inspect_chat_template.py --jsonl <path> --index 0`

Inference/eval:
- `CKPT=<ckpt> GT_JSONL=<gt.jsonl> OUTPUT_BASE_DIR=<out_dir> bash scripts/run_infer_eval.sh`
- `PYTHONPATH=. conda run -n ms python scripts/evaluate_detection.py --pred_jsonl <pred.jsonl> --out_dir <eval_out> --metrics both`

Coord-token verification:
- `conda run -n ms python scripts/tools/verify_coord_tokens.py --original <base_ckpt> --merged <merged_ckpt> [--adapter <adapter_ckpt>]`
