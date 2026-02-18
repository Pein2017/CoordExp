# Inference/Eval Tooling (Memory)

Role separation:
- Memory role: fast map of executable inference/eval utilities and sharp edges.
- Canonical docs: inference/eval sections in `docs/` runbooks and dataset docs.
- Canonical code paths: `scripts/run_infer.py`, `scripts/evaluate_detection.py`, `src/infer/`, `src/eval/`.
- Update trigger: when CLI behavior, artifact paths, or deprecated flag handling changes.

Inference path:
- Primary runner is `scripts/run_infer.py` (YAML-first; legacy flag mode still present).
- Pipeline implementation entry is `src/infer/pipeline.py`.
- End-to-end helper is `scripts/run_infer_eval.sh`.

Evaluator reminders:
- Entry: `scripts/evaluate_detection.py`.
- Metrics suites: COCO/F1-ish/both.
- Deprecated options (`--unknown-policy`, `--semantic-fallback`) fail fast.

Inspection/utilities quick map:
- Template sanity: `scripts/tools/inspect_chat_template.py`
- JSONL validation: `public_data/scripts/validate_jsonl.py`
- Coord vocab/token helpers: `scripts/tools/expand_coord_vocab.py`, `scripts/tools/verify_coord_vocab.py`, `scripts/tools/verify_coord_tokens.py`
- Checkpoint module inspection: `scripts/tools/inspect_checkpoint_modules.py`
