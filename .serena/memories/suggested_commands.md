# Command Cribsheet (Memory)

Role separation:
- Memory role: short, high-frequency command recall only.
- Canonical docs for full procedures: `docs/data/PREPARATION.md`, `docs/training/STAGE2_RUNBOOK.md`, `docs/eval/WORKFLOW.md`, `public_data/README.md`.
- Update trigger: when recommended launcher patterns or script CLIs change.

Environment baseline:
- Prefer `PYTHONPATH=. conda run -n ms ...` from repo root.

Training:
- `PYTHONPATH=. conda run -n ms python -m src.sft --config <yaml> [--base_config <yaml>] [--debug|--verbose]`
- `config=<yaml> gpus=0 conda run -n ms bash scripts/train.sh`
- `config=<yaml> gpus=0,1,2,3 conda run -n ms bash scripts/train.sh`
- `server_gpus=0,1,2,3 train_gpus=4,5 config=<yaml> conda run -n ms bash scripts/train_stage2.sh`
- `server_gpus=0,1,2,3,4,5 train_gpus=6,7 config=<yaml> conda run -n ms bash scripts/train_stage2.sh`

Validation and sanity:
- `conda run -n ms python public_data/scripts/validate_jsonl.py <jsonl>`
- `conda run -n ms python scripts/tools/inspect_chat_template.py --jsonl <path> --index 0`
- `PYTHONPATH=. conda run -n ms python -m src.sft --config <yaml> --debug`
- `conda run -n ms python -m pytest -q tests/test_stage1_metric_key_parity.py`
- `conda run -n ms python -m pytest -q tests/test_stage2_rollout_aligned.py tests/test_stage2_rollout_import_boundaries.py`
- `conda run -n ms python -m pytest -q tests/test_run_manifest_files.py tests/test_run_metadata_file.py`
- `conda run -n ms python -m pytest -q tests/test_unified_infer_pipeline.py tests/test_detection_eval_output_parity.py`

Inference and evaluation:
- `PYTHONPATH=. conda run -n ms python scripts/run_infer.py --config configs/infer/pipeline.yaml`
- `PYTHONPATH=. conda run -n ms python scripts/postop_confidence.py --config configs/postop/confidence.yaml`
- `PYTHONPATH=. conda run -n ms python scripts/evaluate_detection.py --config configs/eval/detection.yaml`
