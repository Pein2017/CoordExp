# Command Cribsheet (Memory)

Role separation:
- Memory role: short, high-frequency command recall only.
- Canonical docs for full procedures: `docs/data/PREPARATION.md`, `docs/training/STAGE2_RUNBOOK.md`, `docs/eval/WORKFLOW.md`, `public_data/README.md`.
- Update trigger: when recommended launcher patterns or script CLIs change.

Environment baseline:
- Prefer `PYTHONPATH=. conda run -n ms ...` from repo root.

Training:
- `PYTHONPATH=. conda run -n ms python -m src.sft --config <yaml> [--base_config <yaml>] [--debug|--verbose]`
- `bash scripts/train.sh config=<yaml> gpus=0`
- `bash scripts/train.sh config=<yaml> gpus=0,1,2,3`

Validation and sanity:
- `conda run -n ms python public_data/scripts/validate_jsonl.py <jsonl>`
- `conda run -n ms python scripts/tools/inspect_chat_template.py --jsonl <path> --index 0`
- `PYTHONPATH=. conda run -n ms python -m src.sft --config <yaml> --debug`

Inference and evaluation:
- `PYTHONPATH=. conda run -n ms python scripts/run_infer.py --config configs/infer/pipeline.yaml`
- `PYTHONPATH=. conda run -n ms python scripts/postop_confidence.py --config configs/postop/confidence.yaml`
- `PYTHONPATH=. conda run -n ms python scripts/evaluate_detection.py --config configs/eval/detection.yaml`
