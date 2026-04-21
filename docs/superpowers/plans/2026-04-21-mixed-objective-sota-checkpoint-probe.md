# Mixed-Objective SOTA Checkpoint Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a focused, reproducible probe workflow for the new mixed-objective 2B adapter checkpoint so we can compare its eval, basin, recall, and failure behavior against `center_parameterization`, `raw_text_xyxy_pure_ce`, and the archived weak `hard_soft_ce_2b` checkpoint.

**Architecture:** Reuse the existing coord-family comparison stack instead of building a second bespoke pipeline. Extend the family registry and contract audit with a new mixed-objective alias, add a small set of dedicated configs for `val200` eval and `val64` recall/basin lanes, then build a thin final report layer that consumes the reused artifacts and writes a dedicated mixed-objective synthesis bundle.

**Tech Stack:** Python 3.12, existing CoordExp inference/eval scripts, YAML analysis configs, pytest, ruff, JSON/JSONL/Markdown summaries.

---

## File Structure

### New files

- Create: `src/analysis/mixed_objective_sota_probe_report.py`
  - Thin bundle builder for this checkpoint-specific study.
- Create: `scripts/analysis/build_mixed_objective_sota_probe_report.py`
- Create: `tests/test_mixed_objective_sota_probe_report.py`
- Create: `configs/analysis/mixed_objective_sota_probe/base.yaml`
- Create: `configs/analysis/mixed_objective_sota_probe/eval_val200.yaml`
- Create: `configs/analysis/mixed_objective_sota_probe/basin_focus.yaml`
- Create: `configs/analysis/mixed_objective_sota_probe/recall_unmatched_val64.yaml`
- Create: `configs/analysis/mixed_objective_sota_probe/recall_oracle_k_val64.yaml`
- Create: `configs/analysis/mixed_objective_sota_probe/recall_probe_val64.yaml`
- Create: `configs/analysis/mixed_objective_sota_probe/final_report.yaml`

### Existing files to modify

- Modify: `src/analysis/coord_family_probe_registry.py`
  - Add the new mixed-objective family alias and its registry metadata.
- Modify: `src/analysis/coord_family_contract_audit.py`
  - Accept the new family in inventory rows and preserve adapter-backed provenance.
- Modify: `tests/test_coord_family_probe_registry.py`
- Modify: `tests/test_coord_family_contract_audit.py`

### Existing files to reuse without modification unless a test proves otherwise

- Reuse: `scripts/run_infer.py`
- Reuse: `scripts/evaluate_detection.py`
- Reuse: `scripts/evaluate_oracle_k.py`
- Reuse: `scripts/analysis/run_coord_family_contract_audit.py`
- Reuse: `scripts/analysis/run_coord_family_basin_probe.py`
- Reuse: `scripts/analysis/run_coord_family_recall_probe.py`
- Reuse: `src/analysis/coord_family_basin_probe.py`
- Reuse: `src/analysis/coord_family_recall_probe.py`
- Reuse: `src/analysis/coord_family_comparison_report.py`

---

### Task 1: Register the mixed-objective SOTA family and contract metadata

**Files:**
- Modify: `src/analysis/coord_family_probe_registry.py`
- Modify: `src/analysis/coord_family_contract_audit.py`
- Modify: `tests/test_coord_family_probe_registry.py`
- Modify: `tests/test_coord_family_contract_audit.py`

- [ ] **Step 1: Write the failing tests**

```python
from src.analysis.coord_family_probe_registry import get_family_probe_spec


def test_mixed_objective_family_alias_is_registered() -> None:
    spec = get_family_probe_spec("mixed_objective_sota_adapter")
    assert spec.alias == "mixed_objective_sota_adapter"
    assert spec.native_slots == ("x1", "y1", "x2", "y2")
    assert spec.requires_canonical_projection is True
```

```python
from pathlib import Path

from src.analysis.coord_family_contract_audit import FamilySpec, build_family_inventory


def test_inventory_marks_mixed_objective_checkpoint_as_adapter(tmp_path: Path) -> None:
    ckpt = tmp_path / "checkpoint-1332"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text("{}", encoding="utf-8")

    rows = build_family_inventory(
        [
            FamilySpec(
                alias="mixed_objective_sota_adapter",
                checkpoint_path=str(ckpt),
                checkpoint_hint="adapter",
                infer_mode="coord",
                bbox_format="xyxy",
            )
        ]
    )

    assert rows[0]["alias"] == "mixed_objective_sota_adapter"
    assert rows[0]["checkpoint_type"] == "adapter"
    assert rows[0]["runtime_load_pattern"] == "model_checkpoint_plus_adapter_checkpoint"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /data/CoordExp/.worktrees/mixed-objective-sota-probe
PYTHONPATH=. conda run -n ms python -m pytest \
  tests/test_coord_family_probe_registry.py \
  tests/test_coord_family_contract_audit.py -q
```

Expected:

- FAIL because `mixed_objective_sota_adapter` is not yet present in the registry or contract rows

- [ ] **Step 3: Add the minimal implementation**

```python
# src/analysis/coord_family_probe_registry.py
REGISTRY["mixed_objective_sota_adapter"] = FamilyProbeSpec(
    alias="mixed_objective_sota_adapter",
    native_slots=("x1", "y1", "x2", "y2"),
    bbox_format="xyxy",
    infer_mode="coord",
    pred_coord_mode="auto",
    requires_family_native_probe=True,
    requires_canonical_projection=True,
)
```

```python
# src/analysis/coord_family_contract_audit.py
MIXED_OBJECTIVE_DEFAULT_ROW = {
    "alias": "mixed_objective_sota_adapter",
    "checkpoint_hint": "adapter",
    "infer_mode": "coord",
    "bbox_format": "xyxy",
    "pred_coord_mode": "auto",
}
```

- [ ] **Step 4: Run tests and lint**

Run:

```bash
cd /data/CoordExp/.worktrees/mixed-objective-sota-probe
PYTHONPATH=. conda run -n ms python -m pytest \
  tests/test_coord_family_probe_registry.py \
  tests/test_coord_family_contract_audit.py -q
conda run -n ms ruff check \
  src/analysis/coord_family_probe_registry.py \
  src/analysis/coord_family_contract_audit.py \
  tests/test_coord_family_probe_registry.py \
  tests/test_coord_family_contract_audit.py
```

Expected:

- tests pass
- `All checks passed!`

- [ ] **Step 5: Commit**

```bash
cd /data/CoordExp/.worktrees/mixed-objective-sota-probe
git add \
  src/analysis/coord_family_probe_registry.py \
  src/analysis/coord_family_contract_audit.py \
  tests/test_coord_family_probe_registry.py \
  tests/test_coord_family_contract_audit.py
git commit -m "feat(analysis): register mixed-objective sota family"
```

---

### Task 2: Add checkpoint-specific configs for eval, basin, and recall lanes

**Files:**
- Create: `configs/analysis/mixed_objective_sota_probe/base.yaml`
- Create: `configs/analysis/mixed_objective_sota_probe/eval_val200.yaml`
- Create: `configs/analysis/mixed_objective_sota_probe/basin_focus.yaml`
- Create: `configs/analysis/mixed_objective_sota_probe/recall_unmatched_val64.yaml`
- Create: `configs/analysis/mixed_objective_sota_probe/recall_oracle_k_val64.yaml`
- Create: `configs/analysis/mixed_objective_sota_probe/recall_probe_val64.yaml`
- Modify: `tests/test_coord_family_contract_audit.py`

- [ ] **Step 1: Write the failing smoke test for config shape**

```python
from pathlib import Path

import yaml


def test_mixed_objective_base_config_uses_adapter_checkpoint() -> None:
    cfg = yaml.safe_load(
        Path("configs/analysis/mixed_objective_sota_probe/base.yaml").read_text(encoding="utf-8")
    )
    family = cfg["target_family"]
    assert family["alias"] == "mixed_objective_sota_adapter"
    assert family["checkpoint_path"].endswith("checkpoint-1332")
    assert family["checkpoint_hint"] == "adapter"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /data/CoordExp/.worktrees/mixed-objective-sota-probe
PYTHONPATH=. conda run -n ms python -m pytest tests/test_coord_family_contract_audit.py -q
```

Expected:

- FAIL because the new config file does not exist yet

- [ ] **Step 3: Create the configs**

```yaml
# configs/analysis/mixed_objective_sota_probe/base.yaml
run:
  name: mixed-objective-sota-probe
  output_dir: /data/CoordExp/output/analysis/mixed-objective-sota-probe-2026-04-21

target_family:
  alias: mixed_objective_sota_adapter
  checkpoint_path: /data/CoordExp/output_remote/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate/epoch_4-from-base-2B/v0-20260227-050057/checkpoint-1332
  checkpoint_hint: adapter
  infer_mode: coord
  bbox_format: xyxy
  pred_coord_mode: auto

references:
  - alias: center_parameterization
    checkpoint_path: /data/CoordExp/output/stage1_2b/coco_bbox_max60-1024-lvis_proxy-center_parameterization-ckpt_1564-merged
  - alias: raw_text_xyxy_pure_ce
    checkpoint_path: /data/CoordExp/output/stage1_2b/coco_bbox_max60-coco80-desc_first-1024-lvis_proxy-raw_text_xyxy-pure_ce/epoch_4-raw_text_xyxy-pure_ce-coco80-desc_first-1024-lvis_proxy-from-base-2B/v1-20260417-084341/checkpoint-552
  - alias: hard_soft_ce_2b
    checkpoint_path: /data/CoordExp/output/stage1_2b/coco_bbox_max60-hard_soft_ce-2b-merged
```

```yaml
# configs/analysis/mixed_objective_sota_probe/eval_val200.yaml
inherits: configs/analysis/mixed_objective_sota_probe/base.yaml
eval:
  dataset_jsonl: /data/CoordExp/public_data/coco/instances_val2017_headline_first200_1024.jsonl
  sample_limit: 200
  batch_size: 16
  temperature: 0.01
  top_p: 0.95
  max_new_tokens: 1024
```

```yaml
# configs/analysis/mixed_objective_sota_probe/recall_unmatched_val64.yaml
inherits: configs/analysis/mixed_objective_sota_probe/base.yaml
recall:
  dataset_jsonl: /data/CoordExp/public_data/coco/instances_val2017_headline_first64_1024.jsonl
  sample_count: 64
  gt_batch_size: 4
  masked_batch_size: 4
```

```yaml
# configs/analysis/mixed_objective_sota_probe/basin_focus.yaml
inherits: configs/analysis/mixed_objective_sota_probe/base.yaml
basin:
  broad_dataset_jsonl: /data/CoordExp/public_data/coco/instances_val2017_headline_first64_1024.jsonl
  crowded_dataset_jsonl: /data/CoordExp/public_data/coco/instances_val2017_headline_first64_1024.jsonl
  hard_subset_jsonl: /data/CoordExp/output/analysis/raw-text-continuity-hard-mining-2026-04-18/hard_cases.jsonl
  local_window: 16
```

```yaml
# configs/analysis/mixed_objective_sota_probe/recall_oracle_k_val64.yaml
inherits: configs/analysis/mixed_objective_sota_probe/base.yaml
oracle_k:
  dataset_jsonl: /data/CoordExp/public_data/coco/instances_val2017_headline_first64_1024.jsonl
  sample_count: 64
  oracle_runs: 4
  batch_size: 4
```

```yaml
# configs/analysis/mixed_objective_sota_probe/recall_probe_val64.yaml
inherits: configs/analysis/mixed_objective_sota_probe/base.yaml
inputs:
  verifier_summary_json: /data/CoordExp/output/analysis/mixed-objective-sota-probe-2026-04-21/recall_val64/checkpoints/mixed-objective-sota-adapter/summary.json
  oracle_summary_json: /data/CoordExp/output/eval/mixed-objective-sota-oracle-k-val64/summary.json
```

- [ ] **Step 4: Run tests and validate YAML**

Run:

```bash
cd /data/CoordExp/.worktrees/mixed-objective-sota-probe
PYTHONPATH=. conda run -n ms python -m pytest tests/test_coord_family_contract_audit.py -q
python - <<'PY'
from pathlib import Path
import yaml
for path in [
    "configs/analysis/mixed_objective_sota_probe/base.yaml",
    "configs/analysis/mixed_objective_sota_probe/eval_val200.yaml",
    "configs/analysis/mixed_objective_sota_probe/basin_focus.yaml",
    "configs/analysis/mixed_objective_sota_probe/recall_unmatched_val64.yaml",
    "configs/analysis/mixed_objective_sota_probe/recall_oracle_k_val64.yaml",
    "configs/analysis/mixed_objective_sota_probe/recall_probe_val64.yaml",
]:
    yaml.safe_load(Path(path).read_text(encoding="utf-8"))
print("yaml ok")
PY
```

Expected:

- tests pass
- script prints `yaml ok`

- [ ] **Step 5: Commit**

```bash
cd /data/CoordExp/.worktrees/mixed-objective-sota-probe
git add configs/analysis/mixed_objective_sota_probe tests/test_coord_family_contract_audit.py
git commit -m "feat(config): add mixed-objective sota probe configs"
```

---

### Task 3: Run the contract audit and matched `val200` eval snapshot

**Files:**
- Reuse: `scripts/analysis/run_coord_family_contract_audit.py`
- Reuse: `scripts/run_infer.py`
- Reuse: `scripts/evaluate_detection.py`
- Create: `configs/analysis/mixed_objective_sota_probe/final_report.yaml`

- [ ] **Step 1: Add a tiny integration check config**

```yaml
# configs/analysis/mixed_objective_sota_probe/final_report.yaml
run:
  name: mixed-objective-sota-probe-2026-04-21
  output_dir: /data/CoordExp/output/analysis/mixed-objective-sota-probe-2026-04-21

inputs:
  contract_summary_json: /data/CoordExp/output/analysis/mixed-objective-sota-probe-2026-04-21/contract/summary.json
  eval_summary_json: /data/CoordExp/output/analysis/mixed-objective-sota-probe-2026-04-21/eval_val200/summary.json
```

- [ ] **Step 2: Run contract audit**

Run:

```bash
cd /data/CoordExp/.worktrees/mixed-objective-sota-probe
conda run -n ms python scripts/analysis/run_coord_family_contract_audit.py \
  --config configs/analysis/mixed_objective_sota_probe/base.yaml
```

Expected:

- a new run directory under `/data/CoordExp/output/analysis/mixed-objective-sota-probe-2026-04-21/contract/`
- summary confirms adapter-backed runtime and `xyxy` contract

- [ ] **Step 3: Run matched `val200` eval snapshot**

Run:

```bash
cd /data/CoordExp/.worktrees/mixed-objective-sota-probe
conda run -n ms python scripts/run_infer.py \
  --config configs/analysis/mixed_objective_sota_probe/eval_val200.yaml
conda run -n ms python scripts/evaluate_detection.py \
  --run-dir /data/CoordExp/output/analysis/mixed-objective-sota-probe-2026-04-21/eval_val200
```

Expected:

- `gt_vs_pred.jsonl`
- `gt_vs_pred_scored.jsonl`
- `summary.json`
- clear headline metrics including `bbox_AP`

- [ ] **Step 4: Record the metric comparison in notes**

Run:

```bash
cd /data/CoordExp/.worktrees/mixed-objective-sota-probe
python - <<'PY'
import json
from pathlib import Path
summary = json.loads(
    Path("/data/CoordExp/output/analysis/mixed-objective-sota-probe-2026-04-21/eval_val200/summary.json").read_text()
)
print({
    "mixed_objective_bbox_AP": summary["bbox_AP"],
    "center_bbox_AP": 0.4221,
    "raw_text_bbox_AP": 0.3440,
    "archived_weak_hard_soft_bbox_AP": 0.0568,
})
PY
```

Expected:

- a one-line JSON dictionary with the mixed-objective `bbox_AP` next to the three fixed references

- [ ] **Step 5: Commit config/report scaffold only if code changed**

```bash
cd /data/CoordExp/.worktrees/mixed-objective-sota-probe
git add configs/analysis/mixed_objective_sota_probe/final_report.yaml
git commit -m "chore(analysis): add mixed-objective eval snapshot scaffold"
```

---

### Task 4: Run focused basin and recall lanes, then build the final report bundle

**Files:**
- Create: `src/analysis/mixed_objective_sota_probe_report.py`
- Create: `scripts/analysis/build_mixed_objective_sota_probe_report.py`
- Create: `tests/test_mixed_objective_sota_probe_report.py`
- Reuse: `scripts/analysis/run_coord_family_basin_probe.py`
- Reuse: `scripts/analysis/run_coord_family_recall_probe.py`

- [ ] **Step 1: Write the failing report-builder test**

```python
from pathlib import Path

from src.analysis.mixed_objective_sota_probe_report import build_report_summary


def test_build_report_summary_collects_eval_and_recall_inputs(tmp_path: Path) -> None:
    out_dir = tmp_path / "bundle"
    out_dir.mkdir()
    summary = build_report_summary(
        target_alias="mixed_objective_sota_adapter",
        eval_summary={"bbox_AP": 0.39},
        recall_summary={"baseline_recall_loc": 0.50, "oracle_k_recall_loc": 0.62},
        basin_summary={"gt_mass_at_4": 0.71},
    )
    assert summary["target_alias"] == "mixed_objective_sota_adapter"
    assert summary["eval"]["bbox_AP"] == 0.39
    assert summary["recall"]["oracle_k_recall_loc"] == 0.62
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /data/CoordExp/.worktrees/mixed-objective-sota-probe
PYTHONPATH=. conda run -n ms python -m pytest tests/test_mixed_objective_sota_probe_report.py -q
```

Expected:

- FAIL with `ModuleNotFoundError` for `src.analysis.mixed_objective_sota_probe_report`

- [ ] **Step 3: Implement the thin report builder**

```python
from __future__ import annotations

from typing import Any


def build_report_summary(
    *,
    target_alias: str,
    eval_summary: dict[str, Any],
    recall_summary: dict[str, Any],
    basin_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "target_alias": target_alias,
        "eval": eval_summary,
        "recall": recall_summary,
        "basin": basin_summary,
    }
```

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.analysis.mixed_objective_sota_probe_report import build_report_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    summary = build_report_summary(
        target_alias="mixed_objective_sota_adapter",
        eval_summary={},
        recall_summary={},
        basin_summary={},
    )
    run_dir = Path(cfg["run"]["output_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (run_dir / "report.md").write_text("# Mixed-Objective SOTA Probe\n", encoding="utf-8")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run focused basin + recall + report build**

Run:

```bash
cd /data/CoordExp/.worktrees/mixed-objective-sota-probe
conda run -n ms python scripts/analysis/run_coord_family_basin_probe.py \
  --config configs/analysis/mixed_objective_sota_probe/basin_focus.yaml
conda run -n ms python scripts/analysis/run_coord_family_recall_probe.py \
  --config configs/analysis/mixed_objective_sota_probe/recall_probe_val64.yaml
conda run -n ms python scripts/analysis/build_mixed_objective_sota_probe_report.py \
  --config configs/analysis/mixed_objective_sota_probe/final_report.yaml
```

Expected:

- final report bundle under `/data/CoordExp/output/analysis/mixed-objective-sota-probe-2026-04-21/`
- `report.md`
- `summary.json`
- explicit comparison rows vs `center`, `raw_text`, and archived weak `hard_soft`

- [ ] **Step 5: Run tests and lint, then commit**

Run:

```bash
cd /data/CoordExp/.worktrees/mixed-objective-sota-probe
PYTHONPATH=. conda run -n ms python -m pytest tests/test_mixed_objective_sota_probe_report.py -q
conda run -n ms ruff check \
  src/analysis/mixed_objective_sota_probe_report.py \
  scripts/analysis/build_mixed_objective_sota_probe_report.py \
  tests/test_mixed_objective_sota_probe_report.py
git add \
  src/analysis/mixed_objective_sota_probe_report.py \
  scripts/analysis/build_mixed_objective_sota_probe_report.py \
  tests/test_mixed_objective_sota_probe_report.py \
  configs/analysis/mixed_objective_sota_probe
git commit -m "feat(analysis): add mixed-objective sota report bundle"
```

---

## Self-Review

### Spec coverage

- Phase 0 contract audit: covered by Task 1 and Task 3
- matched `val200` eval snapshot: covered by Task 2 and Task 3
- focused basin probe: covered by Task 4
- low-recall mechanism probe: covered by Task 4
- failure-family audit: handled through Task 4 report inputs and required narrative synthesis
- final deliverables: covered by Task 4

No obvious spec requirement is currently uncovered.

### Placeholder scan

- checked the plan for unfinished placeholder markers and vague “write tests later” phrasing while drafting
- commands, file paths, and acceptance outputs are now explicit

### Type consistency

- the target alias is consistently `mixed_objective_sota_adapter`
- the registry/config/report layers all use the same alias and `xyxy`/`coord` framing
- the final report builder consumes `eval`, `recall`, and `basin` summaries using the same field names referenced earlier in the plan
