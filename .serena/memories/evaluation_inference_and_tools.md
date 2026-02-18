# Evaluation, Inference, and Common Tools

Inference:
- Entry: `scripts/run_infer.py` (uses `src/infer/engine.py`). Produces `pred.jsonl` compatible with evaluator.
- Full workflow helper: `scripts/run_infer_eval.sh` (env-driven; set `CKPT`, `GT_JSONL`, `OUTPUT_BASE_DIR`, etc).
  - Note: the script hardcodes a `PYTHON_BIN`; override if your env differs.

Detection evaluation:
- CLI entry: `scripts/evaluate_detection.py` -> `src/eval/detection.py::evaluate_and_save`.
- Supports `--metrics coco|f1ish|both` and unknown-desc policies (`bucket|drop|semantic`).
- Semantic matching uses a sentence-transformers embedding model (configurable via flags).
- Outputs: summary JSON + optional overlays under `--out_dir` (overwrites).

Template / data inspection:
- `scripts/tools/inspect_chat_template.py` renders one JSONL record through the model chat template (good for prompt + coord sanity).
- Public-data validator: `public_data/scripts/validate_jsonl.py`.

Coord vocab/token utilities:
- Vocab: `scripts/tools/expand_coord_vocab.py`, `scripts/tools/verify_coord_vocab.py`.
- JSONL conversion/verification: `scripts/tools/convert_to_coord_tokens.py`, `scripts/tools/verify_coord_tokens.py`.
- Coord-offset tooling: `scripts/tools/inject_coord_offsets.py`.

Checkpoint inspection:
- `scripts/tools/inspect_checkpoint_modules.py` helps verify adapter/module layout.
