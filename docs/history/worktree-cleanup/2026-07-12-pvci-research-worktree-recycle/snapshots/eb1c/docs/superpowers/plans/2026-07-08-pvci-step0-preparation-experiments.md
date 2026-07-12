# PVCI Step 0 Preparation Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run Step 0 causal-preparation experiments for the PVCI/PICD/VCI direction: wrong-mark attribution, held-out GT-mark generalization, and mark-coarseness tolerance.

**Architecture:** Reuse the existing CoordExp-Swift painted-GT materialization, HF inference, parser, and debug metric/report surfaces. Add only the missing branch-local coarseness materialization and report glue required to test whether visible paint generalizes and how precise future internal cursors must be. Do not implement hidden cursors, feature-space marks, selector heads, coverage depletion, or null-instance STOP in this step.

**Tech Stack:** Python 3.12, PyTorch/Transformers Qwen3-VL via existing `src.infer`, Pillow painted-image materialization, CoordExp-Swift JSONL/config/artifact contracts, pytest.

## Global Constraints

- Worktree: `/data/CoordExp/.codex/worktrees/eb1c/CoordExp`
- Branch: `codex/qwen3-vl-painted-gt-transcription-probe`
- Source overfit stepwise adapter: `/data/CoordExp/outputs/painted_gt/train_overfit_gate/painted_gt_stepwise_teacher_prefix_geo_gate256_overfit16_warm_start_dora_all_towers_accelerate8_ebs8/checkpoints/step-484/adapter`
- Source special-token embedding payload: `/data/CoordExp/outputs/painted_gt/train_overfit_gate/painted_gt_stepwise_teacher_prefix_geo_gate256_overfit16_warm_start_dora_all_towers_accelerate8_ebs8/checkpoints/step-484/special_token_embeddings`
- Held-out val200 JSONL: `/data/CoordExp/.worktrees/CoordExp-swift/outputs/coordexp_swift/infer/val200_inputs/coco_val200_len12000.rebased_images.coord.jsonl`
- Primary output root: `/data/CoordExp/outputs/painted_gt/pvci_step0`
- Decode: HF backend, `temperature: 0.0`, `top_p: 1.0`, `repetition_penalty: 1.10`, free raw generation.
- GPU coexistence: use all visible GPUs only through CoordExp-Swift data-parallel inference with `generation.batch_size: 1` per device; if memory becomes tight, restrict to a subset of GPUs instead of raising batch size.
- Step 0 must remain a measurement suite. Architecture implementation begins only after the Step 0 report.

---

### Task 1: Record Scope And Existing Wrong-Mark Evidence

**Files:**
- Create: `research/ideas/qwen3-vl-painted-gt-transcription-probe/pvci-step0-plan.md`
- Create/Update after metrics: `research/ideas/qwen3-vl-painted-gt-transcription-probe/pvci-step0-results.md`

**Interfaces:**
- Consumes existing artifact: `/data/CoordExp/outputs/painted_gt/counterfactual_inference/counterfactual_summary_gate256_overfit16.json`
- Produces a durable research note that separates existing train256 evidence from new held-out/coarseness evidence.

- [ ] Write `pvci-step0-plan.md` with experiment scope, surfaces, artifact roots, and pass/fail interpretation.
- [ ] Extract existing wrong-object mark metrics from `counterfactual_summary_gate256_overfit16.json`.
- [ ] Record current conclusion: wrong-object mark is a steering actuator if marked-object IoU/F1 is high while scheduled-target F1 is low.

### Task 2: Add Coarseness Materialization Support With TDD

**Files:**
- Modify: `src/painted_gt/painting.py`
- Modify: `src/painted_gt/materialization.py`
- Modify: `src/painted_gt/counterfactuals.py`
- Modify: `scripts/probes/painted_gt/materialize_counterfactual_conditions.py`
- Test: `tests/painted_gt/test_materialization.py`
- Test: `tests/painted_gt/test_counterfactual_controls.py`

**Interfaces:**
- Produces coarseness variants for stepwise teacher-prefix condition rows:
  - `tight_outline_center`
  - `box_1p5_outline_center`
  - `box_2p0_outline_center`
  - `grid_snapped_box_outline_center`
  - `center_blob`
  - `center_point`
  - `outline_only`
- Each row must record `mark_coarseness_variant`, original target bbox, effective painted bbox or center primitive, and nearest-GT overlap diagnostics.

- [ ] Write failing tests asserting coarseness variants materialize deterministic painted plans and metadata.
- [ ] Verify tests fail before production edits.
- [ ] Implement minimal coarseness helper functions and metadata.
- [ ] Add CLI option to materialize selected coarseness variants for stepwise conditions.
- [ ] Verify targeted tests pass.

### Task 3: Add Held-Out Source Override For Materialization

**Files:**
- Modify: `scripts/probes/painted_gt/materialize_counterfactual_conditions.py`
- Modify: `scripts/probes/painted_gt/materialize_stepwise_teacher_prefix_condition.py`
- Test: `tests/painted_gt/test_counterfactual_controls.py`

**Interfaces:**
- Produces `--input-jsonl` for materializers. When supplied, it overrides `config.data.train.path` but keeps template/model/preflight defaults from `--config`.
- Source identity must record the override path, sha256, sample limit, source config, and config fingerprint.

- [ ] Write failing test or CLI-level unit that proves `--input-jsonl` changes loaded source identity.
- [ ] Verify the test fails.
- [ ] Implement `--input-jsonl`.
- [ ] Verify targeted tests pass.

### Task 4: Materialize Step 0 Conditions

**Files:**
- Output only under `/data/CoordExp/outputs/painted_gt/pvci_step0/materialized/`

**Interfaces:**
- Consumes materialization scripts and val200/train source JSONLs.
- Produces condition manifests and JSONLs for inference configs.

- [ ] Materialize held-out val200 Step 0B conditions: `stepwise__painted_correct`, `stepwise__unpainted_same_prompt`, and `stepwise__wrong_object_mark`, using sample size `100` first.
- [ ] Materialize train256 Step 0C coarseness conditions for all listed variants.
- [ ] If train256 coarseness runs produce a clear top subset, materialize the strongest 2-3 variants on val100.
- [ ] Verify condition manifests, geometry audit, preflight status, row counts, and JSONL loadability.

### Task 5: Generate And Run Inference Configs

**Files:**
- Create: `configs/coordexp_swift/infer/painted_gt/pvci_step0/*.yaml`
- Output: `/data/CoordExp/outputs/painted_gt/pvci_step0/inference/`

**Interfaces:**
- Each config consumes a materialized `example_jsonl`, the step-484 adapter, and the step-484 special-token embedding payload.
- Each config uses `generation.batch_size: 1`, `temperature: 0.0`, `top_p: 1.0`, `repetition_penalty: 1.10`, and `max_new_tokens` sufficient for one compact row.

- [ ] Generate YAML configs for each Step 0B and Step 0C condition.
- [ ] Parse YAML configs.
- [ ] Launch with `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m src.infer --config <config>`.
- [ ] Monitor GPU memory and relaunch on fewer GPUs if coexistence with user training causes OOM.
- [ ] Verify each run writes `summary.json`, `gt_vs_pred.jsonl`, `resolved_config.json`, and token/parse diagnostics.

### Task 6: Write Step 0 Report

**Files:**
- Create/Update: `research/ideas/qwen3-vl-painted-gt-transcription-probe/pvci-step0-results.md`

**Interfaces:**
- Consumes debug metric JSONs, inference summaries, condition manifests, and existing wrong-mark report.
- Produces the go/no-go interpretation for PICD/VCI.

- [ ] Run `scripts/probes/painted_gt/write_debug_metric_report.py` for every completed condition.
- [ ] Aggregate row counts, F1, precision, recall, row validity, parse/drop counters, and mark-attribution facts.
- [ ] Interpret:
  - held-out tight GT mark works or fails;
  - wrong mark steers or disrupts;
  - coarse mark tolerance implies coarse selector, center pointer, or tight-localization requirement.
- [ ] Record exact artifact roots, commands, scope, caveats, and next recommended branch.

## Verification

Run targeted checks:

```bash
pytest tests/painted_gt/test_materialization.py tests/painted_gt/test_counterfactual_controls.py -q
python -m py_compile src/painted_gt/painting.py src/painted_gt/materialization.py src/painted_gt/counterfactuals.py scripts/probes/painted_gt/materialize_counterfactual_conditions.py scripts/probes/painted_gt/materialize_stepwise_teacher_prefix_condition.py
git diff --check -- src/painted_gt scripts/probes/painted_gt tests/painted_gt configs/coordexp_swift/infer/painted_gt/pvci_step0 research/ideas/qwen3-vl-painted-gt-transcription-probe docs/superpowers/plans
```

After GPU runs, verify:

```bash
python scripts/probes/painted_gt/write_debug_metric_report.py --mode stepwise_teacher_prefix --run-dir <run_dir>
python - <<'PY'
from pathlib import Path
import json
for p in Path('/data/CoordExp/outputs/painted_gt/pvci_step0/inference').glob('*/summary.json'):
    d=json.loads(p.read_text())
    print(p.parent.name, d.get('row_count'), d.get('scoreable_prediction_count'), d.get('terminal_status'))
PY
```
