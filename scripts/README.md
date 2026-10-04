# Script ownership

Training and inference are configuration-first module entrypoints:
`python -m src.train --config ...` and `python -m src.infer --config ...`.
Main's runnable model configurations live under `configs/coordexp_infras/`.
Do not introduce another launcher around retired MS-Swift or Stage-2 APIs.

`scripts/` owns thin operational entrypoints; reusable behavior belongs to its
`src/` owner. The offline evaluator is `scripts/evaluate_detection.py`.
Annotation operations delegate to `src/coco_refinement/` or
`src/label_studio_coco_refinement/`. Visualization is not evaluation evidence.
Self-contained artifact reducers may remain under `scripts/analysis/`; new
research experiments belong to the canonical research checkout, not a revived
`src.analysis` tree in main. Tooling needs a current consumer and must not
duplicate a source-owned API.

Public-data recovery is a separate contract. Two legacy diagnostics still have
data-preparation callers: `scripts/tools/inspect_chat_template.py` and
`scripts/analysis/measure_gt_max_new_tokens.py`. Their removed model APIs make
them unsupported as runnable diagnostics. Migrate their rendering/token-budget
consumers deliberately before retirement; do not silently change regenerated
samples or reintroduce the old model stack.

Use this checkout's source, typed configuration and tests for current behavior.
`docs/BRANCH_AND_WORKTREE_POLICY.md` owns checkout boundaries and
`docs/OUTPUT_STORAGE_POLICY.md` owns artifacts and explicit transfers.
Git history supplies obsolete command implementations; this directory is not
their archive.
