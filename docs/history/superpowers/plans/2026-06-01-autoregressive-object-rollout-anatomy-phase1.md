# Autoregressive Object Rollout Anatomy Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run the CPU-safe launch phase for checkpoint-3664 autoregressive object rollout anatomy: Gate 0/0b artifact validity plus Lane A free-decode rollout anatomy.

**Architecture:** Keep Phase 1 as a small analysis surface that reads existing inference/eval artifacts and writes machine-readable analysis outputs under `/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200`. Do not load the model or use GPUs until shard-safe Lane B/C/D wrappers exist.

**Tech Stack:** Python stdlib JSON/CSV/hashlib/pathlib, existing `src.infer.artifacts.load_comparable_artifact`, pytest fixtures, YAML config for launch reproducibility.

---

## Files

- Create: `configs/analysis/autoreg_object_rollout/ckpt3664_val200.yaml`
- Create: `scripts/analysis/run_autoreg_object_rollout_anatomy.py`
- Create: `src/analysis/autoreg_object_rollout.py`
- Create: `tests/test_autoreg_object_rollout.py`
- Read only during validation: `/data/CoordExp/outputs/infer/recursive_detection_ce_latest/compact_full_support2_tokenrows_v2_ckpt3664_val200_bsz8_temp0_rep1p10_prompt_offset_fix_8gpu`
- Read only during validation: `/data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl`

## Task 1: Config And Entrypoint

- [ ] **Step 1: Create the analysis config**

Write `configs/analysis/autoreg_object_rollout/ckpt3664_val200.yaml` with these fields:

```yaml
run:
  output_dir: /data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200
  scope_label: val200_first_200
  checkpoint: /data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_et_rmp_ce_support2_bsz16_4epoch_tokenrows_v2/compact-full-et-rmp-ce-support2-bsz16-4epoch-tokenrows-v2/v0-20260504-071356/checkpoint-3664
inputs:
  artifact_root: /data/CoordExp/outputs/infer/recursive_detection_ce_latest/compact_full_support2_tokenrows_v2_ckpt3664_val200_bsz8_temp0_rep1p10_prompt_offset_fix_8gpu
  dataset_jsonl: /data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl
  dataset_slice:
    kind: first_n
    n: 200
analysis:
  truncate_at_first_im_end: true
  require_scored_provenance: false
  metric_family: guarded
```

- [ ] **Step 2: Create the CLI wrapper**

Create `scripts/analysis/run_autoreg_object_rollout_anatomy.py` that accepts `--config` and calls `run_phase1_from_config(config_path)`.

- [ ] **Step 3: Run import smoke**

Run:

```bash
PYTHONPATH=/data/CoordExp python scripts/analysis/run_autoreg_object_rollout_anatomy.py --help
```

Expected: exits 0 and shows `--config`.

## Task 2: Gate 0 And Gate 0b Analyzer

- [ ] **Step 1: Write failing tests**

In `tests/test_autoreg_object_rollout.py`, add fixtures with two source rows, two trace rows, raw/scored/guarded artifacts, matches, duplicate guard mapping, summary, and resolved config. Tests must assert:

```python
assert gate["status"] == "ok"
assert resolved["dataset_slice"]["kind"] == "first_n"
assert resolved["artifact_hashes"]["gt_vs_pred.jsonl"]["line_count"] == 2
assert rows[0]["source_line_idx"] == 0
assert rows[0]["coco_image_id"] == 139
assert rows[0]["guarded_pred_idx"] == 0
assert rows[1]["suppressed_by_guard"] is True
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
PYTHONPATH=/data/CoordExp pytest tests/test_autoreg_object_rollout.py -q
```

Expected before implementation: import or function-not-found failure.

- [ ] **Step 3: Implement `src.analysis.autoreg_object_rollout`**

Implement:

```python
def run_phase1_from_config(config_path: str | Path) -> dict[str, Any]: ...
def run_phase1(config: Mapping[str, Any]) -> dict[str, Any]: ...
def build_resolved_inputs(config: Mapping[str, Any]) -> dict[str, Any]: ...
def build_gate_report(resolved: Mapping[str, Any]) -> dict[str, Any]: ...
def build_lane_a_rows(resolved: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]: ...
```

Use physical JSONL line number as `source_line_idx`; rename eval match `image_id` to `eval_record_idx`; keep COCO id as `coco_image_id`; hash source files; call `load_comparable_artifact(..., require_score=True)` and record failure as `debug-f1ish-only` instead of aborting.

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
PYTHONPATH=/data/CoordExp pytest tests/test_autoreg_object_rollout.py -q
```

Expected: all tests pass.

## Task 3: Lane A Rollout Anatomy Outputs

- [ ] **Step 1: Write row-label and EOS tests**

Add tests asserting first-`<|im_end|>` truncation:

```python
assert per_image[0]["first_im_end_index"] == 5
assert per_image[0]["tokens_after_first_im_end"] == 2
assert per_image[0]["endoftext_after_im_end_count"] == 2
assert summary["token_summary"]["used_token_count"] == 6
```

Add tests asserting row labels:

```python
assert per_row[0]["raw_match_label"] == "tp_like"
assert per_row[1]["row_label"] == "duplicate_suppressed"
```

- [ ] **Step 2: Implement outputs**

Write:

```text
resolved_inputs.json
gate_report.json
rollout_anatomy/per_row.jsonl
rollout_anatomy/per_image.jsonl
rollout_anatomy/coverage_survival.csv
rollout_anatomy/summary.json
rollout_anatomy/report.md
```

The report must include `production_training_recommendation: none`.

- [ ] **Step 3: Run unit tests**

Run:

```bash
PYTHONPATH=/data/CoordExp pytest tests/test_autoreg_object_rollout.py -q
```

Expected: all tests pass.

## Task 4: Run Phase 1 On Current Val200 Artifact

- [ ] **Step 1: Run the CPU-only analysis**

Run:

```bash
PYTHONPATH=/data/CoordExp python scripts/analysis/run_autoreg_object_rollout_anatomy.py \
  --config configs/analysis/autoreg_object_rollout/ckpt3664_val200.yaml
```

Expected: exits 0 and writes the Phase 1 output tree under `/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200`.

- [ ] **Step 2: Verify output counts**

Run:

```bash
PYTHONPATH=/data/CoordExp python - <<'PY'
import json
from pathlib import Path
root = Path('/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200')
summary = json.loads((root / 'rollout_anatomy/summary.json').read_text())
gate = json.loads((root / 'gate_report.json').read_text())
assert gate['gate0']['status'] == 'ok'
assert summary['scope']['dataset_slice'] == 'first_200'
assert summary['counts']['images'] == 200
assert summary['counts']['gt_objects'] == 1444
assert summary['counts']['raw_predictions'] == 1141
print('phase1 artifact checks ok')
PY
```

Expected: prints `phase1 artifact checks ok`.

## Task 5: 8-GPU Hold/Launch Decision

- [ ] **Step 1: Check GPU readiness**

Run:

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits
```

Expected: 8 devices visible, enough memory free for one worker per device.

- [ ] **Step 2: Decide next stage**

If Phase 1 passes, do not launch Lane B/C/D directly. Record this launch decision:

```text
Lane A: completed CPU-only
Lane B: hold until config compatibility, boundary mode, and shard/range output are implemented
Lane C: hold until Lane A match-state target selection and shard/merge output are implemented
Lane D: hold until compact-full rendering/position inventory adapter is implemented
8-GPU status: ready for the next shard-safe implementation wave
```

If any Phase 1 gate fails, fix Phase 1 before any GPU job.
