## 1. Baseline and Guardrails

- [x] 1.1 Capture baseline parity references for Stage-2, inference pipeline, and detection evaluation outputs (metric keys + artifact schema) from current main behavior.
- [x] 1.2 Document reproducibility checkpoints for this change in a working note: config paths, run names, seeds, output artifact locations.
- [x] 1.3 Add/confirm fail-fast guardrails list for invariant-critical paths (queue/version gating, required batch fields, artifact path resolution).
- [x] 1.4 Add a baseline preflight gate (syntax/import collection sanity) and fail the refactor task chain early if baseline is not build-clean.

## 2. Stage-2 / Rollout Contract Boundaries

- [x] 2.1 Introduce public rollout-matching contract modules and remove Stage-2 imports of private underscore-prefixed rollout symbols.
- [x] 2.2 Add an AST-based regression guard test (pytest/static AST import walk, not regex) that fails when underscore-private rollout imports are reintroduced from Stage-2 paths.
- [x] 2.3 Decompose Stage-2 AB internals into scheduler, async queue manager, and channel executors while preserving trainer-variant entrypoint behavior.
- [x] 2.4 Apply exception taxonomy in Stage-2/rollout critical paths: critical invariants fail fast, diagnostics remain explicit best-effort.
- [x] 2.5 Validate Stage-2 contract parity with targeted tests and logs (channel schedule determinism, async queue feasibility/fallback counters).
- [x] 2.6 Remove temporary compatibility shims/re-exports after all callsites are migrated and parity checks pass.

## 3. Coord/Data Utility Consolidation

- [x] 3.1 Consolidate coordinate conversion/validation helper ownership into canonical shared utils and remove duplicated helper paths where equivalent.
- [x] 3.2 Unify dense/fusion sample-to-encode flow through shared helpers while preserving geometry ordering and metadata semantics.
- [x] 3.3 Ensure dataset and evaluator consumers both use canonical helper contracts for coord-token and numeric geometry handling.
- [x] 3.4 Validate geometry invariants and determinism with existing/new targeted tests (no coord drop/reorder, stable conversion outcomes).
- [x] 3.5 Preserve helper ownership boundaries: transforms remain authoritative in `src/datasets/geometry.py`; shared coord-utils remain pure/import-light and avoid dataset<->eval import cycles.

## 4. Inference Pipeline and Engine Robustness

- [x] 4.1 Refactor inference pipeline into explicit config/artifact resolution phase plus stage execution phase, with `resolved_config.json` as the single canonical resolved manifest (no parallel manifest for the same contract).
- [x] 4.2 Implement resilient HF attention backend selection with explicit fallback diagnostics, preserving output contract.
- [x] 4.3 Keep inference backend runtime selection behind explicit contract interfaces without changing standardized output payload semantics.
- [x] 4.4 Unify relative image-root/path-resolution behavior across infer/eval/vis surfaces, explicitly including `src/infer/vis.py`.
- [x] 4.5 Validate infer/eval/vis compatibility using YAML-driven pipeline runs, schema checks on `gt_vs_pred.jsonl` and `summary.json`, and contract checks on `resolved_config.json` field completeness.
- [x] 4.6 Add targeted validation for inference-engine sample-scoped structured errors: failed samples emit structured error entries and summary failure counters.
- [x] 4.7 Define and lock a minimal stable `resolved_config.json` schema contract (`schema_version` major and additive-only key evolution within a major).
- [x] 4.8 Record resolved image-root decision breadcrumbs in `resolved_config.json` (`root_image_dir`, `root_image_dir_source`) for eval/vis reproducibility.
- [x] 4.9 Record selected attention backend (including fallback choice when used) in run artifacts (`summary.json` and/or `resolved_config.json`).

## 5. Detection Evaluator Parity and Diagnostics

- [x] 5.1 Standardize evaluator ingestion diagnostics to path + 1-based line explicit parse errors, including a clipped payload snippet for triage.
- [x] 5.2 Remove any remaining duplicate evaluator conversion/validation helpers that are not already shared; keep evaluator on canonical shared geometry contracts.
- [x] 5.3 Verify metric/artifact schema parity for COCO + F1-ish outputs and match artifact generation.

## 6. Trainer Metrics Contract Inversion

- [x] 6.1 Introduce neutral trainer-metrics payload contracts and migrate metrics components to consume them.
- [x] 6.2 Preserve documented stable metric key names and evaluation prefix behavior during migration.
- [x] 6.3 Preserve explicit first-failure signaling for diagnostics-only metric paths while keeping training non-blocking.
- [x] 6.4 Validate metric parity with targeted regression tests and key-set comparisons.
- [x] 6.5 Standardize neutral payload `schema_version` as integer major version (initial major `1`) and add explicit rejection tests for missing/non-integer/unsupported-major versions.

## 7. OpenSpec, Docs, and Config Synchronization

- [x] 7.1 Update affected docs/runbooks to reflect final contract surfaces (Stage-2 runbook, metrics/losses, eval and data contract docs as needed).
- [x] 7.2 Update YAML configs/scripts only where required by finalized contract behavior (config-first, no new CLI flags).
- [x] 7.3 Verify operational entrypoints remain synchronized with refactor behavior (`scripts/train.sh`, `scripts/stage2_ab_server_train.sh`, `scripts/run_infer.py`, `scripts/evaluate_detection.py`, `scripts/run_infer_eval.sh`, `scripts/run_vis.sh`).
- [x] 7.4 Add explicit server-mode deploy-readiness gate criteria for Stage-2 ops (backend/mode/server URL/model preconditions) and sync them with runbook/script checks.
- [x] 7.5 Update evaluator docs to mark deprecated CLI knobs as deprecated/ignored and point to the canonical pipeline config path.
- [x] 7.6 Ensure all OpenSpec deltas in this change remain aligned with implemented behavior before apply/archive.

## 8. Validation Commands and Exit Checks

- [x] 8.1 Run baseline syntax/import preflight for known critical modules before feature-level suites:
      `PYTHONPATH=. conda run -n ms python -m py_compile src/infer/engine.py`
      `PYTHONPATH=. conda run -n ms python -m pytest --collect-only -q tests/test_unified_infer_pipeline.py`
- [x] 8.2 Run focused test suites for changed surfaces using repo-standard environment:
      `PYTHONPATH=. conda run -n ms python -m pytest -q tests/test_stage2_ab_training.py tests/test_rollout_matching_sft.py tests/test_unified_infer_pipeline.py tests/test_coord_standardizer.py tests/test_coord_utils.py tests/test_token_type_metrics.py tests/test_stage1_metric_key_parity.py tests/test_batch_extras_contract.py`
- [x] 8.3 Run Stage-2 server-mode operational smoke with a pinned smoke config (manual/GPU path) via `scripts/stage2_ab_server_train.sh`; explicitly verify backend/mode/server URL/model gate diagnostics and key telemetry checks.
- [x] 8.4 After 8.1 preflight passes, from repository root run ops smoke checks for non-Stage2 operational entrypoints with pinned smoke configs/artifacts:
      `PYTHONPATH=. conda run -n ms python scripts/run_infer.py --config configs/bench/a_only_ckpt_6064_infer_eval.yaml`
      `PYTHONPATH=. conda run -n ms python scripts/evaluate_detection.py --pred_jsonl output/bench/a_only_ckpt_6064/gt_vs_pred.jsonl --out_dir output/bench/a_only_ckpt_6064/eval_manual --metrics both --unknown-policy semantic --num-workers 0`
      `CKPT=output/stage2_ab/a_only_ckpt_6064 GT_JSONL=public_data/lvis/rescale_32_768_bbox_max60/val.bbox_only.max60.coord.jsonl OUTPUT_BASE_DIR=output/bench/smoke_infer_eval MODE=auto PRED_COORD_MODE=auto DEVICE=cuda:0 LIMIT=10 OVERLAY=0 NUM_WORKERS=0 scripts/run_infer_eval.sh`
      `PRED_JSONL=output/bench/a_only_ckpt_6064/gt_vs_pred.jsonl SAVE_DIR=output/bench/a_only_ckpt_6064/vis_smoke ROOT_IMAGE_DIR=\"${ROOT_IMAGE_DIR:?set ROOT_IMAGE_DIR to dataset image root}\" LIMIT=10 scripts/run_vis.sh`
- [x] 8.5 Run any additional contract/parity tests introduced by this change on touched modules.
- [x] 8.6 Run OpenSpec strict validation for this change:
      (from repository root)
      `openspec validate refactor-src-modernization --strict`
- [x] 8.7 Record final reproducibility checklist (config path, run name, seed, artifacts) in the change notes before handoff.
- [x] 8.8 Post-merge reconciliation on `main`: run merged contract/boundary suite and strict OpenSpec validation against the final integrated branch state.
