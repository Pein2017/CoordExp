# FN-Rescue Desc-X1 Logit Binding Phase 5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a CPU-first Phase-5 analysis that joins Phase-4 instance-attention rows with Lane-D `x1_logit_lens_*` probe rows.

**Architecture:** Add one focused analysis module plus one runner and YAML configs.  The module validates artifact coverage, materializes a joined per-role/per-layer logit-binding table, writes bucket summaries, and records role coverage gaps for requested roles such as `post_x1` and `pre_y1`.

**Tech Stack:** Python, JSONL artifacts, YAML configs, pytest, existing CoordExp analysis conventions.

---

### Task 1: Config, Dry-Run, And Coverage Audit

**Files:**
- Create: `src/analysis/autoreg_fn_rescue_desc_x1_phase5.py`
- Create: `scripts/analysis/run_autoreg_fn_rescue_desc_x1_phase5.py`
- Create: `configs/analysis/autoreg_fn_rescue_desc_x1_phase5/ckpt3664_val200.yaml`
- Create: `configs/analysis/autoreg_fn_rescue_desc_x1_phase5/ckpt3664_val200_smoke.yaml`
- Create: `tests/test_autoreg_fn_rescue_desc_x1_phase5.py`

- [x] **Step 1: Write failing config/dry-run test**

Create a tiny config with `artifact_root`, `phase4_root`, and `lane_d_root`.
Assert `build_phase5_dry_run_plan` reports the expected input paths, requested
roles, layer groups, and probe availability.

- [x] **Step 2: Implement config dataclasses**

Add `Phase5Paths`, `Phase5ProbeConfig`, `Phase5Config`,
`load_phase5_config`, and `build_phase5_dry_run_plan`.

- [x] **Step 3: Add runner dry-run**

Create a CLI runner with `--config`, `--stages`, and `--dry-run`.

- [x] **Step 4: Verify**

Run:

```bash
PYTHONPATH=$PWD python - <<'PY'
import pytest, sys
sys.exit(pytest.main(['tests/test_autoreg_fn_rescue_desc_x1_phase5.py', '-q']))
PY
```

Expected during RED: import failure.  Expected after implementation: tests pass.

### Task 2: X1 Logit Binding Probe

**Files:**
- Modify: `src/analysis/autoreg_fn_rescue_desc_x1_phase5.py`
- Modify: `scripts/analysis/run_autoreg_fn_rescue_desc_x1_phase5.py`
- Modify: `tests/test_autoreg_fn_rescue_desc_x1_phase5.py`

- [x] **Step 1: Write failing fixture test**

Create Phase-4 `instance_attention_binding/rows.jsonl` and Lane-D
`probe_rows.jsonl` fixtures.  Assert that `materialize_x1_logit_binding_probe`
writes joined rows where `x1_logit_lens_rank` and
`x1_logit_lens_target_minus_top1` are preserved and bucket summaries are
computed.

- [x] **Step 2: Implement join logic**

Join by `case_id`.  Preserve Phase-4 mechanism fields and probe role/layer
fields.  Treat rows without `x1_logit_lens_available=true` as coverage rows but
exclude them from rank and margin means.

- [x] **Step 3: Implement role coverage audit**

For every requested role and layer group, report whether any available
state-dependent logit-lens row exists.  Record missing requested roles.

- [x] **Step 4: Implement critical-slice summary**

Count rows where target attention is positive but x1 rank is poor or
target-minus-top1 is negative.  This is the next mechanism slice for cases where
attention sees the target but coordinate binding remains weak.

- [x] **Step 5: Verify**

Run the Phase-5 test file and compile the new module/runner.

### Task 3: Real Artifact Runs And Findings

**Files:**
- Modify: `progress/diagnostics/2026-06-02_fn_rescue_attention_binding_findings.md`

- [x] **Step 1: Run dry-run**

```bash
PYTHONPATH=$PWD python scripts/analysis/run_autoreg_fn_rescue_desc_x1_phase5.py \
  --config configs/analysis/autoreg_fn_rescue_desc_x1_phase5/ckpt3664_val200_smoke.yaml \
  --stages x1_logit_binding_probe,report \
  --dry-run
```

- [x] **Step 2: Run smoke**

```bash
PYTHONPATH=$PWD python scripts/analysis/run_autoreg_fn_rescue_desc_x1_phase5.py \
  --config configs/analysis/autoreg_fn_rescue_desc_x1_phase5/ckpt3664_val200_smoke.yaml \
  --stages x1_logit_binding_probe,report
```

- [x] **Step 3: Run full CPU analysis**

```bash
PYTHONPATH=$PWD python scripts/analysis/run_autoreg_fn_rescue_desc_x1_phase5.py \
  --config configs/analysis/autoreg_fn_rescue_desc_x1_phase5/ckpt3664_val200.yaml \
  --stages x1_logit_binding_probe,report
```

- [x] **Step 4: Artifact check**

Check that full root contains:

```text
x1_logit_binding_probe/rows.jsonl
x1_logit_binding_probe/summary.json
x1_logit_binding_probe/report.md
summary.json
report.md
```

- [x] **Step 5: Update findings**

Append Phase-5 smoke/full results, including coverage gaps and whether the
critical positive-attention-but-poor-x1-logit slice exists.

### Task 4: Phase-5B Coord-Slot Extension

**Files:**
- Modify: `src/analysis/autoreg_hidden_state_probe.py`
- Modify: `src/analysis/autoreg_fn_rescue_desc_x1_phase5.py`
- Modify: `scripts/analysis/run_autoreg_fn_rescue_desc_x1_phase5.py`
- Modify: `tests/test_autoreg_hidden_state_probe.py`
- Modify: `tests/test_autoreg_fn_rescue_desc_x1_phase5.py`
- Create: `configs/analysis/autoreg_hidden_state_probe/ckpt3664_lane_d_coord_slot_logit_lens_postx1.yaml`
- Create: `configs/analysis/autoreg_fn_rescue_desc_x1_phase5/ckpt3664_val200_coordslot.yaml`

- [x] **Step 1: Add coord-slot TDD coverage**

Verify that `pre_x1` targets `x1`, `post_x1` targets `y1`, and `post_y1`
targets `x2`.

- [x] **Step 2: Add Lane-D coord-slot logit lens fields**

Add `coord_slot_logit_lens_*` artifact fields while preserving existing
`x1_logit_lens_*` fields for backward compatibility.

- [x] **Step 3: Run Lane-D coord-slot extraction**

Run 8 shards with tmux session `autoreg_lane_d_coordslot_ckpt3664`.

- [x] **Step 4: Add Phase-5 coord-slot linkage stage**

Materialize `coord_slot_logit_binding_probe/rows.jsonl`, `summary.json`, and
`report.md`.

- [x] **Step 5: Update findings**

Record the slot-aware result: `post_x1::y1` improves over `pre_x1::x1`, but
the target slot is still often below a competing top coordinate.
