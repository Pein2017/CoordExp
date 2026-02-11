# refactor-src-modernization — working notes

Last updated: 2026-02-11

This file is a *working note* for baseline parity references, reproducibility
checkpoints, and invariant guardrails. It is intentionally descriptive (no new
metrics/results are invented here).

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
- `eval_rollout_*`
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
  `openspec/changes/refactor-src-modernization/specs/inference-pipeline/spec.md`).

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
bash openspec/changes/refactor-src-modernization/preflight.sh
```

## 5) Final validation / smoke checklist (this change)

This section records **what was actually run** (configs + commands + artifact dirs)
so the change can be reproduced and audited during review/handoff. No new results
are claimed here beyond “command completed and artifacts were written”.

### 5.1 Stage-2 AB server-mode operational smoke (1 step, forced Channel-B)

- Config: `temp/stage2_ab_server_smoke_1step.yaml` (extends `configs/stage2_ab/smoke/ab_mixed.yaml`)
- Key overrides:
  - `training.max_steps: 1` (I/O disabled: `save_strategy=no`, `eval_strategy=no`)
  - `custom.extra.rollout_matching.max_new_tokens: 128`
  - Server URL pinned to avoid collisions: `http://127.0.0.1:8003` (default configs use `:8000`)
  - `stage2_ab.schedule.b_ratio: 1.0` (force Channel-B on the single step)
- Launcher topology (8 GPUs total):
  - vLLM server GPUs: `0,1,2,3,4,5`
  - learner/train GPUs: `6,7`
- Command:

  ```bash
  bash scripts/stage2_ab_server_train.sh \
    server_gpus=0,1,2,3,4,5 \
    train_gpus=6,7 \
    config=/data/CoordExp/.worktrees/refactor-src-modernizatio-5.2/temp/stage2_ab_server_smoke_1step.yaml \
    vllm_gpu_memory_utilization=0.70 \
    wait_timeout=1200
  ```

- Output run dir:
  - `output/stage2_ab/smoke/ab_mixed/v4-20260211-010233/smoke_ab_mixed`
  - Key artifacts: `run_metadata.json`, `logging.jsonl`
  - `run_metadata.json` captures git SHA/dirty status + resolved config path.

### 5.2 Unified infer+eval pipeline smoke (HF backend, small limit)

- Config: `temp/a_only_ckpt_6064_infer_eval_smoke_fast.yaml`
  - `run.name: a_only_ckpt_6064_smoke_fast_2026-02-11`
  - `run.output_dir: output/bench_smoke`
  - `infer.generation.seed: 42`
  - `infer.limit: 5`
- Command:

  ```bash
  PYTHONPATH=. conda run -n ms python scripts/run_infer.py \
    --config temp/a_only_ckpt_6064_infer_eval_smoke_fast.yaml
  ```

- Output run dir:
  - `output/bench_smoke/a_only_ckpt_6064_smoke_fast_2026-02-11`
  - Key artifacts: `gt_vs_pred.jsonl`, `summary.json`, `resolved_config.json`, `eval/`

### 5.3 Detection evaluator smoke (manual entrypoint; deprecated knob tolerated)

- Command:

  ```bash
  PYTHONPATH=. conda run -n ms python scripts/evaluate_detection.py \
    --pred_jsonl output/bench/a_only_ckpt_6064/gt_vs_pred.jsonl \
    --out_dir output/bench/a_only_ckpt_6064/eval_manual \
    --metrics both \
    --unknown-policy semantic \
    --num-workers 0
  ```

- Output dir: `output/bench/a_only_ckpt_6064/eval_manual`

### 5.4 Legacy wrapper infer+eval smoke (shell wrapper)

- Command:

  ```bash
  CKPT=output/stage2_ab/a_only_ckpt_6064 \
  GT_JSONL=public_data/lvis/rescale_32_768_bbox_max60/val.bbox_only.max60.coord.jsonl \
  OUTPUT_BASE_DIR=output/bench/smoke_infer_eval \
  MODE=auto \
  PRED_COORD_MODE=auto \
  DEVICE=cuda:0 \
  LIMIT=10 \
  OVERLAY=0 \
  NUM_WORKERS=0 \
  scripts/run_infer_eval.sh
  ```

- Output dir: `output/bench/smoke_infer_eval` (writes `gt_vs_pred.jsonl`, `summary.json`, `eval/`)

### 5.5 Visualization wrapper smoke (shell wrapper)

- Command:

  ```bash
  PRED_JSONL=output/bench/a_only_ckpt_6064/gt_vs_pred.jsonl \
  SAVE_DIR=output/bench/a_only_ckpt_6064/vis_smoke \
  ROOT_IMAGE_DIR=/data/CoordExp/public_data/lvis/rescale_32_768_bbox_max60 \
  LIMIT=10 \
  scripts/run_vis.sh
  ```

- Output dir: `output/bench/a_only_ckpt_6064/vis_smoke`
