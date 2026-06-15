# FN-Rescue Desc-X1 Binding Mechanism Phase 4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Phase-4 analysis pipeline that links Phase-3 causal outcomes to instance-level attention evidence and desc->x1 hidden/logit probe availability.

**Architecture:** Add a focused CPU-first Phase-4 module and runner.  Phase-4A scans existing instance-level attention rows and joins them to Phase-3 case-linked intervention rows.  Phase-4B writes an explicit hidden/logit linkage report from existing Lane-D artifacts when available, or records the blocker when not available.

**Tech Stack:** Python, JSONL artifacts, YAML configs, existing CoordExp analysis helpers, pytest.

---

### Task 1: Phase-4 Config And Dry-Run

**Files:**
- Create: `src/analysis/autoreg_fn_rescue_desc_x1_phase4.py`
- Create: `scripts/analysis/run_autoreg_fn_rescue_desc_x1_phase4.py`
- Create: `configs/analysis/autoreg_fn_rescue_desc_x1_phase4/ckpt3664_val200.yaml`
- Create: `configs/analysis/autoreg_fn_rescue_desc_x1_phase4/ckpt3664_val200_smoke.yaml`
- Create: `tests/test_autoreg_fn_rescue_desc_x1_phase4.py`

- [x] **Step 1: Write failing config/dry-run test**

Create a test that writes a tiny config, loads it, and asserts the dry-run plan
reports the Phase-3 root, FN-rescue root, Lane-D root, and stage list.

- [x] **Step 2: Implement minimal config dataclasses**

Add `Phase4Paths`, `Phase4Config`, `load_phase4_config`, and
`build_phase4_dry_run_plan`.

- [x] **Step 3: Add runner dry-run**

Create a runner with `--config`, `--stages`, and `--dry-run`.

- [x] **Step 4: Verify**

Run:

```bash
PYTHONPATH=$PWD python - <<'PY'
import pytest, sys
sys.exit(pytest.main(['tests/test_autoreg_fn_rescue_desc_x1_phase4.py', '-q']))
PY
```

Expected: tests pass.

### Task 2: Instance-Level Attention Binding Table

**Files:**
- Modify: `src/analysis/autoreg_fn_rescue_desc_x1_phase4.py`
- Modify: `scripts/analysis/run_autoreg_fn_rescue_desc_x1_phase4.py`
- Modify: `tests/test_autoreg_fn_rescue_desc_x1_phase4.py`

- [x] **Step 1: Write failing fixture test**

Create tiny Phase-3 `case_mechanism_rows.jsonl` and FN-rescue
`rescue_attention_region_rows.jsonl` fixtures.  Assert that
`materialize_instance_attention_binding` writes `rows.jsonl`, `summary.json`,
and `report.md`.

- [x] **Step 2: Implement join logic**

Join on `(case_id, rescue_tier)` and restrict attention rows to
`aggregation_scope=instance`.  Aggregate by region kind and record
`target_attention_mass`, `competitor_attention_mass`,
`rollout_prediction_attention_mass`, `wrong_source_attention_mass`,
`context_ring_attention_mass`, and `far_background_attention_mass`.

- [x] **Step 3: Implement top-head summary**

For each case, record top instance heads by mean attention mass as
`top_instance_attention_heads`.

- [x] **Step 4: Implement bucket summaries**

Write bucket-level counts and mean target-vs-competitor margins grouped by
`mechanism_bucket`.

- [x] **Step 5: Verify**

Run the Phase-4 test file and compile the new module/runner.

### Task 3: Desc-X1 Probe Linkage Manifest

**Files:**
- Modify: `src/analysis/autoreg_fn_rescue_desc_x1_phase4.py`
- Modify: `scripts/analysis/run_autoreg_fn_rescue_desc_x1_phase4.py`
- Modify: `tests/test_autoreg_fn_rescue_desc_x1_phase4.py`

- [x] **Step 1: Write failing absent-artifact test**

Assert that `materialize_desc_x1_probe_linkage` writes a summary/report with
`status=blocked_missing_probe_rows` when Lane-D `probe_rows.jsonl` is absent.

- [x] **Step 2: Write available-artifact fixture test**

Create tiny `probe_rows.jsonl` with roles `desc_end` and `pre_x1`.  Assert the
summary reports available roles and joined case count.

- [x] **Step 3: Implement linkage manifest**

Read optional `probe_rows.jsonl`, join by `case_id`, and summarize available
roles, layer groups, row count, and joined Phase-3 case count.

- [x] **Step 4: Verify**

Run the Phase-4 tests.

### Task 4: Real Artifact Smoke And Findings

**Files:**
- Modify: `progress/diagnostics/2026-06-02_fn_rescue_attention_binding_findings.md`

- [x] **Step 1: Run dry-run**

```bash
PYTHONPATH=$PWD python scripts/analysis/run_autoreg_fn_rescue_desc_x1_phase4.py \
  --config configs/analysis/autoreg_fn_rescue_desc_x1_phase4/ckpt3664_val200_smoke.yaml \
  --stages instance_attention_binding,desc_x1_probe_linkage,report \
  --dry-run
```

- [x] **Step 2: Run smoke**

```bash
PYTHONPATH=$PWD python scripts/analysis/run_autoreg_fn_rescue_desc_x1_phase4.py \
  --config configs/analysis/autoreg_fn_rescue_desc_x1_phase4/ckpt3664_val200_smoke.yaml \
  --stages instance_attention_binding,desc_x1_probe_linkage,report
```

- [x] **Step 3: Run full CPU analysis**

```bash
PYTHONPATH=$PWD python scripts/analysis/run_autoreg_fn_rescue_desc_x1_phase4.py \
  --config configs/analysis/autoreg_fn_rescue_desc_x1_phase4/ckpt3664_val200.yaml \
  --stages instance_attention_binding,desc_x1_probe_linkage,report
```

- [x] **Step 4: Artifact check**

Check that full root contains:

```text
instance_attention_binding/rows.jsonl
instance_attention_binding/summary.json
instance_attention_binding/report.md
desc_x1_probe_linkage/summary.json
desc_x1_probe_linkage/report.md
summary.json
report.md
```

- [x] **Step 5: Update findings**

Append Phase-4 smoke/full summary, including bucket-level target-vs-competitor
instance attention margins and whether hidden/logit linkage is available.
