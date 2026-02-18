# Suggested Commands (run from repo root)

Environment tip:
- Prefer `PYTHONPATH=. conda run -n ms ...` for reproducibility.
- Some bash scripts hardcode `PYTHON_BIN=/root/miniconda3/envs/ms/bin/python`; override via env if needed.

Training:
- `PYTHONPATH=. conda run -n ms python -m src.sft --config <yaml> [--base_config <yaml>] [--debug|--verbose]`
- Multi-GPU: `PYTHONPATH=. conda run -n ms torchrun --nproc_per_node 4 -m src.sft --config <yaml> [--base_config <yaml>]`

Tests:
- `conda run -n ms python -m pytest tests/`

Inspect/validate data:
- `conda run -n ms python scripts/tools/inspect_chat_template.py --jsonl <path/to/data.jsonl> --index 0`
- `conda run -n ms python scripts/tools/verify_coord_tokens.py --input <jsonl>`
- `conda run -n ms python public_data/scripts/validate_jsonl.py <jsonl>`

Inference -> eval:
- `CKPT=<ckpt_or_merged_model> GT_JSONL=<gt.jsonl> OUTPUT_BASE_DIR=<out_dir> bash scripts/run_infer_eval.sh`
- (Eval only) `PYTHONPATH=. conda run -n ms python scripts/evaluate_detection.py --pred_jsonl <pred.jsonl> --out_dir <eval_out> --metrics coco|f1ish|both`

Public data runner:
- `./public_data/run.sh lvis all --preset rescale_32_768_bbox`
- `./public_data/run.sh vg all --preset rescale_32_768_bbox -- --objects-version 1.2.0`
