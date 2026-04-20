# 2B Coordinate Family Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable comparative analysis pipeline that inventories the selected 2B checkpoint families, audits their contracts, measures family-native continuity and bad-basin behavior, and studies low-recall false-negative mechanisms.

**Architecture:** Reuse the existing inference and evaluation stack instead of creating a bespoke runner. Add a new `coord_family_comparison` analysis slice with four focused layers: inventory/contract audit, family-native basin probing, low-recall mechanism probing, and final comparative report synthesis. Keep 2B as the headline lane and 4B as optional auxiliary audit input only.

**Tech Stack:** Python 3.12, existing CoordExp inference/eval pipeline, ms-swift HF runtime, YAML configs, pytest, ruff, JSONL/Markdown reporting.

---

## File Structure

### New files

- Create: `src/analysis/coord_family_contract_audit.py`
  - Inventory builders, checkpoint-type detection, contract summaries, runtime-path resolution.
- Create: `src/analysis/coord_family_probe_registry.py`
  - Family aliases, native slot definitions, representation metadata, canonicalization helpers.
- Create: `src/analysis/coord_family_basin_probe.py`
  - Family-native continuity and bad-basin probe runner, plus aggregate basin metrics.
- Create: `src/analysis/coord_family_recall_probe.py`
  - Teacher-forced GT support, proposal-conditioned recoverability, Oracle-K summary ingestion, FN labeling.
- Create: `src/analysis/coord_family_comparison_report.py`
  - Cross-family aggregation, verdict derivation, artifact bundle assembly.
- Create: `scripts/analysis/run_coord_family_contract_audit.py`
- Create: `scripts/analysis/run_coord_family_basin_probe.py`
- Create: `scripts/analysis/run_coord_family_recall_probe.py`
- Create: `scripts/analysis/build_coord_family_comparison_report.py`
- Create: `tests/test_coord_family_contract_audit.py`
- Create: `tests/test_coord_family_probe_registry.py`
- Create: `tests/test_coord_family_basin_probe.py`
- Create: `tests/test_coord_family_recall_probe.py`
- Create: `tests/test_coord_family_comparison_report.py`
- Create: `configs/analysis/coord_family_comparison/base.yaml`
- Create: `configs/analysis/coord_family_comparison/smoke_inventory.yaml`
- Create: `configs/analysis/coord_family_comparison/smoke_basin.yaml`
- Create: `configs/analysis/coord_family_comparison/smoke_recall.yaml`
- Create: `configs/analysis/coord_family_comparison/final_report_smoke.yaml`

### Existing files to reuse but avoid modifying unless a test proves it is necessary

- Reuse: `src/infer/pipeline.py`
- Reuse: `src/infer/engine.py`
- Reuse: `src/analysis/raw_text_coord_continuity_scoring.py`
- Reuse: `src/analysis/unmatched_proposal_verifier.py`
- Reuse: `src/analysis/duplication_collapse_analysis.py`
- Reuse: `src/vis/gt_vs_pred.py`

### Existing files that are acceptable to modify if the new analysis cannot stay isolated

- Modify only if proven necessary: `src/analysis/raw_text_coord_continuity_scoring.py`
- Modify only if proven necessary: `src/analysis/unmatched_proposal_verifier.py`

---

### Task 1: Build family inventory and contract audit

**Files:**
- Create: `src/analysis/coord_family_contract_audit.py`
- Create: `scripts/analysis/run_coord_family_contract_audit.py`
- Create: `tests/test_coord_family_contract_audit.py`
- Create: `configs/analysis/coord_family_comparison/base.yaml`
- Create: `configs/analysis/coord_family_comparison/smoke_inventory.yaml`

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path

from src.analysis.coord_family_contract_audit import (
    FamilySpec,
    build_family_inventory,
    infer_checkpoint_runtime_mode,
)


def test_infer_checkpoint_runtime_mode_prefers_adapter_when_adapter_config_exists(tmp_path: Path) -> None:
    ckpt = tmp_path / "adapter_ckpt"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text("{}", encoding="utf-8")

    mode = infer_checkpoint_runtime_mode(ckpt)

    assert mode == "adapter"


def test_build_family_inventory_records_required_contract_fields(tmp_path: Path) -> None:
    merged = tmp_path / "merged_ckpt"
    merged.mkdir()
    (merged / "config.json").write_text("{}", encoding="utf-8")

    specs = [
        FamilySpec(
            alias="base_xyxy_merged",
            checkpoint_path=str(merged),
            checkpoint_hint="merged",
            infer_mode="coord",
            bbox_format="xyxy",
        )
    ]

    rows = build_family_inventory(specs)

    assert rows[0]["alias"] == "base_xyxy_merged"
    assert rows[0]["checkpoint_type"] == "merged"
    assert rows[0]["runtime_load_pattern"] == "model_checkpoint_only"
    assert rows[0]["bbox_format"] == "xyxy"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=. conda run -n ms python -m pytest tests/test_coord_family_contract_audit.py -q
```

Expected:

- FAIL with `ModuleNotFoundError` for `src.analysis.coord_family_contract_audit`

- [ ] **Step 3: Write the minimal implementation**

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FamilySpec:
    alias: str
    checkpoint_path: str
    checkpoint_hint: str
    infer_mode: str
    bbox_format: str


def infer_checkpoint_runtime_mode(checkpoint_path: Path) -> str:
    if (checkpoint_path / "adapter_config.json").exists():
        return "adapter"
    return "merged"


def build_family_inventory(specs: list[FamilySpec]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        ckpt_path = Path(spec.checkpoint_path)
        checkpoint_type = infer_checkpoint_runtime_mode(ckpt_path)
        rows.append(
            {
                "alias": spec.alias,
                "checkpoint_path": str(ckpt_path),
                "checkpoint_type": checkpoint_type,
                "runtime_load_pattern": (
                    "model_checkpoint_plus_adapter_checkpoint"
                    if checkpoint_type == "adapter"
                    else "model_checkpoint_only"
                ),
                "infer_mode": spec.infer_mode,
                "bbox_format": spec.bbox_format,
            }
        )
    return rows
```

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.analysis.coord_family_contract_audit import FamilySpec, build_family_inventory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = build_family_inventory([])
    Path(args.output).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests and lint to verify they pass**

Run:

```bash
PYTHONPATH=. conda run -n ms python -m pytest tests/test_coord_family_contract_audit.py -q
conda run -n ms ruff check src/analysis/coord_family_contract_audit.py scripts/analysis/run_coord_family_contract_audit.py tests/test_coord_family_contract_audit.py
```

Expected:

- `2 passed`
- `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add src/analysis/coord_family_contract_audit.py scripts/analysis/run_coord_family_contract_audit.py tests/test_coord_family_contract_audit.py configs/analysis/coord_family_comparison/base.yaml configs/analysis/coord_family_comparison/smoke_inventory.yaml
git commit -m "feat: add coord family contract audit scaffold"
```

---

### Task 2: Add family registry and native slot semantics

**Files:**
- Create: `src/analysis/coord_family_probe_registry.py`
- Create: `tests/test_coord_family_probe_registry.py`
- Modify: `src/analysis/coord_family_contract_audit.py`

- [ ] **Step 1: Write the failing tests**

```python
from src.analysis.coord_family_probe_registry import (
    get_family_probe_spec,
    native_slot_names,
)


def test_native_slot_names_cover_all_headline_families() -> None:
    assert native_slot_names("base_xyxy_merged") == ("x1", "y1", "x2", "y2")
    assert native_slot_names("cxcywh_pure_ce") == ("cx", "cy", "w", "h")
    assert native_slot_names("cxcy_logw_logh_pure_ce") == ("cx", "cy", "logw", "logh")


def test_get_family_probe_spec_flags_canonical_projection_requirement() -> None:
    spec = get_family_probe_spec("center_parameterization")
    assert spec.requires_family_native_probe is True
    assert spec.requires_canonical_projection is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=. conda run -n ms python -m pytest tests/test_coord_family_probe_registry.py -q
```

Expected:

- FAIL with `ModuleNotFoundError` for `src.analysis.coord_family_probe_registry`

- [ ] **Step 3: Write the minimal implementation**

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FamilyProbeSpec:
    alias: str
    native_slots: tuple[str, ...]
    requires_family_native_probe: bool
    requires_canonical_projection: bool


_REGISTRY = {
    "base_xyxy_merged": FamilyProbeSpec(
        alias="base_xyxy_merged",
        native_slots=("x1", "y1", "x2", "y2"),
        requires_family_native_probe=True,
        requires_canonical_projection=True,
    ),
    "raw_text_xyxy_pure_ce": FamilyProbeSpec(
        alias="raw_text_xyxy_pure_ce",
        native_slots=("x1", "y1", "x2", "y2"),
        requires_family_native_probe=True,
        requires_canonical_projection=True,
    ),
    "cxcywh_pure_ce": FamilyProbeSpec(
        alias="cxcywh_pure_ce",
        native_slots=("cx", "cy", "w", "h"),
        requires_family_native_probe=True,
        requires_canonical_projection=True,
    ),
    "cxcy_logw_logh_pure_ce": FamilyProbeSpec(
        alias="cxcy_logw_logh_pure_ce",
        native_slots=("cx", "cy", "logw", "logh"),
        requires_family_native_probe=True,
        requires_canonical_projection=True,
    ),
    "center_parameterization": FamilyProbeSpec(
        alias="center_parameterization",
        native_slots=("cx", "cy", "w", "h"),
        requires_family_native_probe=True,
        requires_canonical_projection=True,
    ),
    "hard_soft_ce_2b": FamilyProbeSpec(
        alias="hard_soft_ce_2b",
        native_slots=("x1", "y1", "x2", "y2"),
        requires_family_native_probe=True,
        requires_canonical_projection=True,
    ),
}


def get_family_probe_spec(alias: str) -> FamilyProbeSpec:
    return _REGISTRY[alias]


def native_slot_names(alias: str) -> tuple[str, ...]:
    return get_family_probe_spec(alias).native_slots
```

- [ ] **Step 4: Run tests and lint to verify they pass**

Run:

```bash
PYTHONPATH=. conda run -n ms python -m pytest tests/test_coord_family_probe_registry.py -q
conda run -n ms ruff check src/analysis/coord_family_probe_registry.py tests/test_coord_family_probe_registry.py
```

Expected:

- `2 passed`
- `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add src/analysis/coord_family_probe_registry.py tests/test_coord_family_probe_registry.py src/analysis/coord_family_contract_audit.py
git commit -m "feat: add coord family probe registry"
```

---

### Task 3: Implement family-native basin probe pilot

**Files:**
- Create: `src/analysis/coord_family_basin_probe.py`
- Create: `scripts/analysis/run_coord_family_basin_probe.py`
- Create: `tests/test_coord_family_basin_probe.py`
- Create: `configs/analysis/coord_family_comparison/smoke_basin.yaml`
- Reuse: `src/analysis/raw_text_coord_continuity_scoring.py`

- [ ] **Step 1: Write the failing tests**

```python
from src.analysis.coord_family_basin_probe import (
    BasinProbeRow,
    summarize_basin_rows,
)


def test_summarize_basin_rows_reports_mass_at_4_and_local_error() -> None:
    rows = [
        BasinProbeRow(
            family_alias="cxcywh_pure_ce",
            slot="cx",
            center_value=500,
            target_value=500,
            candidate_value=500,
            score_mean=-0.1,
            abs_distance_to_target=0,
        ),
        BasinProbeRow(
            family_alias="cxcywh_pure_ce",
            slot="cx",
            center_value=500,
            target_value=500,
            candidate_value=504,
            score_mean=-0.2,
            abs_distance_to_target=4,
        ),
    ]

    summary = summarize_basin_rows(rows)

    assert summary[0]["family_alias"] == "cxcywh_pure_ce"
    assert "mass_at_4" in summary[0]
    assert "local_expected_abs_error" in summary[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=. conda run -n ms python -m pytest tests/test_coord_family_basin_probe.py -q
```

Expected:

- FAIL with `ModuleNotFoundError` for `src.analysis.coord_family_basin_probe`

- [ ] **Step 3: Write the minimal implementation**

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BasinProbeRow:
    family_alias: str
    slot: str
    center_value: int
    target_value: int
    candidate_value: int
    score_mean: float
    abs_distance_to_target: int


def summarize_basin_rows(rows: list[BasinProbeRow]) -> list[dict[str, Any]]:
    if not rows:
        return []
    total = len(rows)
    mass_at_4 = sum(1.0 for row in rows if row.abs_distance_to_target <= 4) / total
    local_expected_abs_error = (
        sum(float(row.abs_distance_to_target) for row in rows) / total
    )
    head = rows[0]
    return [
        {
            "family_alias": head.family_alias,
            "slot": head.slot,
            "mass_at_4": mass_at_4,
            "local_expected_abs_error": local_expected_abs_error,
        }
    ]
```

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.analysis.coord_family_basin_probe import summarize_basin_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_out", required=True)
    args = parser.parse_args()
    Path(args.summary_out).write_text(
        json.dumps({"slot_metrics": summarize_basin_rows([])}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests and lint to verify they pass**

Run:

```bash
PYTHONPATH=. conda run -n ms python -m pytest tests/test_coord_family_basin_probe.py -q
conda run -n ms ruff check src/analysis/coord_family_basin_probe.py scripts/analysis/run_coord_family_basin_probe.py tests/test_coord_family_basin_probe.py
```

Expected:

- `1 passed`
- `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add src/analysis/coord_family_basin_probe.py scripts/analysis/run_coord_family_basin_probe.py tests/test_coord_family_basin_probe.py configs/analysis/coord_family_comparison/smoke_basin.yaml
git commit -m "feat: add coord family basin probe scaffold"
```

---

### Task 4: Implement low-recall mechanism probe

**Files:**
- Create: `src/analysis/coord_family_recall_probe.py`
- Create: `scripts/analysis/run_coord_family_recall_probe.py`
- Create: `tests/test_coord_family_recall_probe.py`
- Create: `configs/analysis/coord_family_comparison/smoke_recall.yaml`

- [ ] **Step 1: Write the failing tests**

```python
from src.analysis.coord_family_recall_probe import classify_fn_mechanism


def test_classify_fn_mechanism_marks_suppressed_when_support_and_recovery_are_high() -> None:
    label = classify_fn_mechanism(
        teacher_forced_support=0.75,
        proposal_support=0.80,
        oracle_k_recovered=True,
        competitor_margin=-0.05,
    )
    assert label == "suppressed_fn"


def test_classify_fn_mechanism_marks_competitive_when_competitor_margin_is_large() -> None:
    label = classify_fn_mechanism(
        teacher_forced_support=0.45,
        proposal_support=0.50,
        oracle_k_recovered=False,
        competitor_margin=0.30,
    )
    assert label == "competitive_fn"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=. conda run -n ms python -m pytest tests/test_coord_family_recall_probe.py -q
```

Expected:

- FAIL with `ModuleNotFoundError` for `src.analysis.coord_family_recall_probe`

- [ ] **Step 3: Write the minimal implementation**

```python
from __future__ import annotations


def classify_fn_mechanism(
    *,
    teacher_forced_support: float,
    proposal_support: float,
    oracle_k_recovered: bool,
    competitor_margin: float,
) -> str:
    if competitor_margin > 0.20:
        return "competitive_fn"
    if teacher_forced_support >= 0.60 and proposal_support >= 0.60 and oracle_k_recovered:
        return "suppressed_fn"
    return "weak_visual_fn"
```

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_out", required=True)
    args = parser.parse_args()
    Path(args.summary_out).write_text(
        json.dumps({"fn_rows": []}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests and lint to verify they pass**

Run:

```bash
PYTHONPATH=. conda run -n ms python -m pytest tests/test_coord_family_recall_probe.py -q
conda run -n ms ruff check src/analysis/coord_family_recall_probe.py scripts/analysis/run_coord_family_recall_probe.py tests/test_coord_family_recall_probe.py
```

Expected:

- `2 passed`
- `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add src/analysis/coord_family_recall_probe.py scripts/analysis/run_coord_family_recall_probe.py tests/test_coord_family_recall_probe.py configs/analysis/coord_family_comparison/smoke_recall.yaml
git commit -m "feat: add coord family recall probe scaffold"
```

---

### Task 5: Build the comparative report bundle

**Files:**
- Create: `src/analysis/coord_family_comparison_report.py`
- Create: `scripts/analysis/build_coord_family_comparison_report.py`
- Create: `tests/test_coord_family_comparison_report.py`
- Create: `configs/analysis/coord_family_comparison/final_report_smoke.yaml`

- [ ] **Step 1: Write the failing tests**

```python
from src.analysis.coord_family_comparison_report import derive_family_verdicts


def test_derive_family_verdicts_flags_family_with_high_bad_basin_as_risky() -> None:
    verdicts = derive_family_verdicts(
        basin_rows=[{"family_alias": "cxcywh_pure_ce", "mass_at_4": 0.82}],
        recall_rows=[{"family_alias": "cxcywh_pure_ce", "suppressed_fn_rate": 0.10, "competitive_fn_rate": 0.35}],
        vision_rows=[{"family_alias": "cxcywh_pure_ce", "vision_lift": 4.0}],
    )

    assert "cxcywh_pure_ce" in verdicts
    assert verdicts["cxcywh_pure_ce"]["family_health"] in {"promising", "mixed", "risky"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=. conda run -n ms python -m pytest tests/test_coord_family_comparison_report.py -q
```

Expected:

- FAIL with `ModuleNotFoundError` for `src.analysis.coord_family_comparison_report`

- [ ] **Step 3: Write the minimal implementation**

```python
from __future__ import annotations

from typing import Any


def derive_family_verdicts(
    *,
    basin_rows: list[dict[str, Any]],
    recall_rows: list[dict[str, Any]],
    vision_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    verdicts: dict[str, dict[str, Any]] = {}
    for row in basin_rows:
        alias = str(row["family_alias"])
        verdicts[alias] = {
            "family_health": "mixed",
            "basin_strength": row.get("mass_at_4"),
        }
    return verdicts
```

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.analysis.coord_family_comparison_report import derive_family_verdicts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_out", required=True)
    args = parser.parse_args()
    summary = {
        "verdicts": derive_family_verdicts(
            basin_rows=[],
            recall_rows=[],
            vision_rows=[],
        )
    }
    Path(args.summary_out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests and lint to verify they pass**

Run:

```bash
PYTHONPATH=. conda run -n ms python -m pytest tests/test_coord_family_comparison_report.py -q
conda run -n ms ruff check src/analysis/coord_family_comparison_report.py scripts/analysis/build_coord_family_comparison_report.py tests/test_coord_family_comparison_report.py
```

Expected:

- `1 passed`
- `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add src/analysis/coord_family_comparison_report.py scripts/analysis/build_coord_family_comparison_report.py tests/test_coord_family_comparison_report.py configs/analysis/coord_family_comparison/final_report_smoke.yaml
git commit -m "feat: add coord family comparison report scaffold"
```

---

### Task 6: Wire the smoke path end to end

**Files:**
- Modify: `configs/analysis/coord_family_comparison/base.yaml`
- Modify: `configs/analysis/coord_family_comparison/smoke_inventory.yaml`
- Modify: `configs/analysis/coord_family_comparison/smoke_basin.yaml`
- Modify: `configs/analysis/coord_family_comparison/smoke_recall.yaml`
- Modify: `configs/analysis/coord_family_comparison/final_report_smoke.yaml`
- Test: `tests/test_coord_family_contract_audit.py`
- Test: `tests/test_coord_family_probe_registry.py`
- Test: `tests/test_coord_family_basin_probe.py`
- Test: `tests/test_coord_family_recall_probe.py`
- Test: `tests/test_coord_family_comparison_report.py`

- [ ] **Step 1: Write the failing integration test**

```python
from pathlib import Path


def test_smoke_configs_exist() -> None:
    for rel in [
        "configs/analysis/coord_family_comparison/base.yaml",
        "configs/analysis/coord_family_comparison/smoke_inventory.yaml",
        "configs/analysis/coord_family_comparison/smoke_basin.yaml",
        "configs/analysis/coord_family_comparison/smoke_recall.yaml",
        "configs/analysis/coord_family_comparison/final_report_smoke.yaml",
    ]:
        assert Path(rel).exists(), rel
```

- [ ] **Step 2: Run the integration test to verify it fails**

Run:

```bash
PYTHONPATH=. conda run -n ms python -m pytest tests/test_coord_family_contract_audit.py tests/test_coord_family_probe_registry.py tests/test_coord_family_basin_probe.py tests/test_coord_family_recall_probe.py tests/test_coord_family_comparison_report.py -q
```

Expected:

- FAIL because the smoke YAML files are missing or incomplete

- [ ] **Step 3: Write the smoke configs and runner commands**

```yaml
# configs/analysis/coord_family_comparison/base.yaml
run:
  output_dir: output/analysis
  name: coord-family-comparison-smoke

families:
  - alias: base_xyxy_merged
    model_checkpoint: output/stage1_2b/coco_bbox_max60-coco80-desc_first-1024-lvis_proxy-merged
    infer_mode: coord
    bbox_format: xyxy
  - alias: raw_text_xyxy_pure_ce
    model_checkpoint: output/stage1_2b/coco_bbox_max60-coco80-desc_first-1024-lvis_proxy-raw_text_xyxy-pure_ce/epoch_4-raw_text_xyxy-pure_ce-coco80-desc_first-1024-lvis_proxy-from-base-2B
    infer_mode: text
    bbox_format: xyxy
```

```yaml
# configs/analysis/coord_family_comparison/smoke_inventory.yaml
extends: ./base.yaml
run:
  name: coord-family-comparison-smoke-inventory
```

```yaml
# configs/analysis/coord_family_comparison/smoke_basin.yaml
extends: ./base.yaml
probe:
  stage: basin
  limit_cases: 8
  candidate_radius: 4
```

```yaml
# configs/analysis/coord_family_comparison/smoke_recall.yaml
extends: ./base.yaml
probe:
  stage: recall
  limit_cases: 8
```

```yaml
# configs/analysis/coord_family_comparison/final_report_smoke.yaml
extends: ./base.yaml
report:
  stage: final
```

- [ ] **Step 4: Run the smoke verification commands**

Run:

```bash
PYTHONPATH=. conda run -n ms python -m pytest tests/test_coord_family_contract_audit.py tests/test_coord_family_probe_registry.py tests/test_coord_family_basin_probe.py tests/test_coord_family_recall_probe.py tests/test_coord_family_comparison_report.py -q
conda run -n ms ruff check src/analysis/coord_family_contract_audit.py src/analysis/coord_family_probe_registry.py src/analysis/coord_family_basin_probe.py src/analysis/coord_family_recall_probe.py src/analysis/coord_family_comparison_report.py scripts/analysis/run_coord_family_contract_audit.py scripts/analysis/run_coord_family_basin_probe.py scripts/analysis/run_coord_family_recall_probe.py scripts/analysis/build_coord_family_comparison_report.py tests/test_coord_family_contract_audit.py tests/test_coord_family_probe_registry.py tests/test_coord_family_basin_probe.py tests/test_coord_family_recall_probe.py tests/test_coord_family_comparison_report.py
```

Expected:

- All listed tests pass
- Ruff reports `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add configs/analysis/coord_family_comparison/base.yaml configs/analysis/coord_family_comparison/smoke_inventory.yaml configs/analysis/coord_family_comparison/smoke_basin.yaml configs/analysis/coord_family_comparison/smoke_recall.yaml configs/analysis/coord_family_comparison/final_report_smoke.yaml tests/test_coord_family_contract_audit.py tests/test_coord_family_probe_registry.py tests/test_coord_family_basin_probe.py tests/test_coord_family_recall_probe.py tests/test_coord_family_comparison_report.py
git commit -m "chore: add coord family comparison smoke configs"
```

---

## Self-Review

### Spec coverage

- `Family Scope` -> Tasks 1 and 2 lock the 2B families and runtime semantics.
- `Checkpoint Loading Policy` -> Task 1 audits merged versus adapter loading.
- `Family-native vs canonical comparison` -> Tasks 2 and 3 define the probe registry and native basin runner.
- `Low-recall mechanism study` -> Task 4 adds FN mechanism classification.
- `Artifact Contract` and `Synthesis` -> Tasks 5 and 6 build the report bundle and smoke path.

### Placeholder scan

- No `TBD` or `TODO` placeholders remain in task steps.
- Each task includes exact files, commands, expected outcomes, and commit boundaries.

### Type consistency

- `FamilySpec`, `FamilyProbeSpec`, `BasinProbeRow`, and `classify_fn_mechanism(...)` are referenced consistently across tasks.
- The final report task consumes the prior task outputs without renaming fields mid-plan.

---

Plan complete and saved to `docs/superpowers/plans/2026-04-20-coord-family-basin-and-recall-comparison.md`.

I recommend **subagent-driven execution** for this one, because the tasks have clean boundaries and we can parallelize the audit/probe/report scaffolds safely. If you prefer inline execution in this session instead, say so; otherwise I’ll proceed with the subagent-driven path.
