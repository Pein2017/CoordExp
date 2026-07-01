# 2026-02-11-refactor-src-modernization — working notes

Last updated: 2026-02-11

This file is a *working note* for baseline parity references, reproducibility
checkpoints, and invariant guardrails. It is intentionally descriptive (no new
metrics/results are invented here).

## 0) Post-merge follow-up: ambiguity cleanup (2026-02-11)

After the main refactor landed and strict validation passed, a quick architecture
audit identified a few remaining ambiguity surfaces:
- metrics helpers whose *canonical* module path is documented as `src/metrics/*`
  but still implemented under `src/trainers/metrics/*`,
- duplicated payload-contract implementations (neutral trainer-metrics),
- duplicated coord-vocab gate-loss math in multiple callsites,
- conflicting schema TypedDict definitions for geometry and conversation records.

These are addressed as follow-up tasks under section 9 in `tasks.md` so the
change record remains paper-ready and bisectable.

## 1) Baseline parity references (current main behavior)

### 1.1 Stage-2 / rollout-matching / Stage-2 AB

Primary references:
- Metric keys + semantics: `docs/training/METRICS_LOSSES.md`
- Stage-2 AB behavior and invariants: `docs/training/STAGE2_RUNBOOK.md`
- Regression tests:
  - `tests/test_rollout_matching_sft.py`
  - `tests/test_stage2_ab_training.py`
  - `tests/test_stage1_metric_key_parity.py`

Stable key families (non-exhaustive; see `docs/training/METRICS_LOSSES.md` for the
canonical list):
- `rollout/*`, `packing/*`, `time/*`
- `eval_rollout/*`
- `stage2_ab/async/*` (async actor/learner telemetry)
- `stage2_ab/channel_b/*` (Channel-B construction diagnostics)

### 1.2 Unified inference pipeline

Primary references:
- Artifact contract + schema checklist: `docs/eval/README.md`
- Pipeline config reference: `configs/infer/pipeline.yaml`
- Regression test: `tests/test_unified_infer_pipeline.py`

Canonical artifacts (YAML pipeline):
- `<run_dir>/gt_vs_pred.jsonl`: per-line JSON dict with stable top-level keys
  (see `docs/eval/README.md`), with geometry under `gt[]` / `pred[]` entries.
- `<run_dir>/summary.json`: run summary + counters.
- `<run_dir>/resolved_config.json`: resolved stage toggles + artifact paths +
  redacted config snapshot (schema-versioned; see OpenSpec delta in
  `openspec/changes/archive/2026-02-14-2026-02-11-refactor-src-modernization/specs/inference-pipeline/spec.md`).

### 1.3 Detection evaluator

Primary references:
- Evaluator behavior + output artifacts: `docs/eval/README.md`
- Entrypoint: `scripts/evaluate_detection.py`

Canonical evaluator artifacts (see `docs/eval/README.md`):
- `metrics.json` (metrics + counters)
- `per_image.json`
- When COCO enabled: `per_class.csv`, `coco_gt.json`, `coco_preds.json`
- When F1-ish enabled: `matches.jsonl` and optional `matches@<thr>.jsonl`

## 2) Reproducibility checkpoints (configs, seeds, artifacts)

Stage-2 AB:
- Smoke configs: `configs/stage2_ab/smoke/*.yaml` (sample limits + run naming)
- Production configs: `configs/stage2_ab/prod/*.yaml` (canonical knobs)
- Output dirs: `training.output_dir` (checkpoints), `training.logging_dir` (TB)
- Run name: `training.run_name`

Inference/eval:
- Unified pipeline: `configs/infer/pipeline.yaml`
- Benchmark configs: `configs/bench/*.yaml` (recommended for paper-ready runs)
- Output dirs:
  - unified pipeline: `<run.output_dir>/<run.name>/...`
  - bench configs: `run.output_dir` + `run.name`
- Seeds:
  - unified pipeline: `infer.generation.seed`
  - bench configs: `infer.generation.seed`

## 3) Fail-fast guardrails (invariant-critical paths)

These are the invariants that should fail fast (not silently degrade):
- Stage-2 AB queue feasibility + version-window gating (async Channel-B).
- Required batch fields / batch-extras contract integrity.
- Artifact path resolution + run-dir determinism for infer/eval/vis.

Regression anchors:
- `tests/test_batch_extras_contract.py` (required extras/fields)
- `tests/test_unified_infer_pipeline.py` (artifact resolution + schema checks)
- `tests/test_stage2_ab_training.py` (Stage-2 AB invariants + contract behavior)

## 4) Baseline preflight gate (must pass before refactor work continues)

Run:

```bash
bash openspec/changes/archive/2026-02-14-2026-02-11-refactor-src-modernization/preflight.sh
```

## 5) Final validation / smoke checklist (this change)

This section records **what was actually run** (configs + commands + artifact dirs)
so the change can be reproduced and audited during review/handoff. No new results
are claimed here beyond “command completed and artifacts were written”.

### 5.1 Pre-merge smoke provenance (historical)

The entries below were captured during the candidate-worktree phase and are kept
for reproducibility provenance only. They are not the final post-merge validation
gate for this change.

- Stage-2 AB server-mode smoke used a pinned config under a worktree path and
  produced artifacts under `output/stage2_ab/smoke/...`.
- Unified infer/eval smoke used `temp/a_only_ckpt_6064_infer_eval_smoke_fast.yaml`
  and wrote artifacts under `output/bench_smoke/...`.
- Manual evaluator and wrapper smokes were executed against the same artifact
  families documented above.

### 5.2 Post-merge reconciliation on `main`

After integrating 5.2 + selected 5.3 improvements into `main`, final contract
alignment is:

- Preserved from 5.2 (authoritative strict surfaces):
  - `src/infer/pipeline.py` strict `resolved_config.json` validation
  - `src/eval/detection.py` strict ingestion diagnostics (path+line + non-object rejection)
  - `src/infer/engine.py` parser ownership via `src/common/prediction_parsing.py`
  - Stage-2/rollout boundary split in `src/trainers/rollout_matching/` and
    `src/trainers/stage2_ab/`

- Imported from 5.3 (targeted improvements):
  - `src/common/geometry/coord_utils.py` import-light ownership (no dataset-layer import)
  - `src/metrics/*` neutral metrics contract package + compatibility shim in
    `src/trainers/metrics/reporter.py`
  - Boundary tests: `tests/test_coord_helper_boundaries.py`,
    `tests/test_coord_geometry_invariants.py`

### 5.3 Final validation run on merged `main`

Commands executed:

```bash
PYTHONPATH=. conda run -n ms python -m pytest -q \
  tests/test_stage2_rollout_import_boundaries.py \
  tests/test_trainer_metrics_payload_contract.py \
  tests/test_detection_eval_ingestion_diagnostics.py \
  tests/test_detection_eval_output_parity.py \
  tests/test_infer_batch_decoding.py \
  tests/test_stage2_ab_training.py \
  tests/test_coord_utils.py \
  tests/test_dataset_runtime_contracts.py \
  tests/test_stage1_metric_key_parity.py \
  tests/test_coord_helper_boundaries.py \
  tests/test_coord_geometry_invariants.py \
  tests/test_unified_infer_pipeline.py
```

- Result: `83 passed`

```bash
openspec validate 2026-02-11-refactor-src-modernization --strict
```

- Result: `Change '2026-02-11-refactor-src-modernization' is valid`
