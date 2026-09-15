# PVCI Step-1 Pre-Commit Probes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run the Step-1 pixel-side pre-commit probe suite that determines whether the painted-outline effect is object-identity steering, box-geometry leakage, boundary-shape dependence, or a mixture, then deliver a route-decision report for PVCI/PICD follow-up.

**Architecture:** Keep the first executable wave offline and artifact-backed. Extend the existing painted-GT counterfactual materializer with boundary-ablation variants, add a pure leakage analyzer over existing `gt_vs_pred.jsonl` rows, generate matched inference configs, run the compact `val32` matrix first, selectively promote informative variants to `val100`, and write a final research note. Feature-space outline transplant and hidden-state cursor probes remain designed Phase-2 candidates and are not implemented by this plan.

**Tech Stack:** Python 3.12, existing CoordExp-Swift inference artifacts, Pillow painted-image materialization, existing `src.painted_gt` counterfactual materializer, existing HF `src.infer`, pytest, YAML.

## Global Constraints

- Worktree: `/data/CoordExp/.codex/worktrees/eb1c/CoordExp`
- Branch: `codex/qwen3-vl-painted-gt-transcription-probe`
- Design source: `docs/superpowers/specs/2026-07-08-pvci-step1-precommit-probes-design.md`
- Stepwise painted-GT adapter: `/data/CoordExp/outputs/painted_gt/train_overfit_gate/painted_gt_stepwise_teacher_prefix_geo_gate256_overfit16_warm_start_dora_all_towers_accelerate8_ebs8/checkpoints/step-484/adapter`
- Special-token embedding delta: `/data/CoordExp/outputs/painted_gt/train_overfit_gate/painted_gt_stepwise_teacher_prefix_geo_gate256_overfit16_warm_start_dora_all_towers_accelerate8_ebs8/checkpoints/step-484/special_token_embeddings`
- Held-out JSONL source: `/data/CoordExp/.worktrees/CoordExp-swift/outputs/coordexp_swift/infer/val200_inputs/coco_val200_len12000.rebased_images.coord.jsonl`
- First executable scope: `val32` boundary-ablation matrix, with selective `val100` promotion only after the `val32` report.
- Frozen decode surface: HF backend, `temperature=0.0`, `top_p=1.0`, `repetition_penalty=1.10`, `max_new_tokens=96`.
- Primary leakage prediction: first valid parsed row from `gt_vs_pred.jsonl` field `pred[0]`.
- Primary GT: target-row GT from `gt_vs_pred.jsonl` field `gt[0]`.
- Primary mark: first current-object mark from `gt_vs_pred.jsonl` field `painted_gt.marks[0]`.
- Do not train a new teacher in Step 1.1 or Step 1.2.
- Do not compare checkpoints in the first executable wave.
- Do not implement feature-space outline transplant before the pixel-side route report.
- Do not implement hidden boundary cursor before the pixel-side route report.
- Do not report target-row debug F1 as COCO mAP.
- Use OpenSpec only if a later slice promotes stable public config schema, evaluator semantics, artifact names, or reusable forward hooks.

---

## File Structure

- Create `src/painted_gt/mark_geometry_leakage.py`: pure functions for bbox IoU/L1, first-prediction extraction, per-row leakage records, and aggregate leakage summaries.
- Create `scripts/probes/painted_gt/analyze_mark_geometry_leakage.py`: CLI wrapper that reads one or more `gt_vs_pred.jsonl` files and writes `mark_geometry_leakage.json` plus `mark_geometry_leakage.rows.jsonl`.
- Modify `src/painted_gt/counterfactuals.py`: add boundary-ablation mark variants and render metadata.
- Modify `scripts/probes/painted_gt/materialize_counterfactual_conditions.py`: expose the new variants through the existing `--mark-coarseness-variant` choices automatically by importing `MARK_COARSENESS_VARIANTS`.
- Create `scripts/probes/painted_gt/build_pvci_step1_infer_configs.py`: write matched inference YAMLs for materialized Step-1 condition roots.
- Create `scripts/probes/painted_gt/write_pvci_step1_route_report.py`: aggregate debug metrics and leakage outputs into a Markdown route report and JSON summary.
- Create `tests/painted_gt/test_mark_geometry_leakage.py`: unit tests for leakage math and row analysis.
- Modify or extend `tests/painted_gt/test_counterfactual_controls.py`: tests for new mark variants, metadata, and representative pixels.
- Create `tests/painted_gt/test_pvci_step1_config_builder.py`: tests for generated YAML paths, decode settings, and config parseability.
- Create final research note after runs: `research/ideas/qwen3-vl-painted-gt-transcription-probe/pvci-step1-precommit-probes-2026-07-08.md`.
- Create first-wave config family: `configs/coordexp_swift/infer/painted_gt/pvci_step1/`.
- Use artifact root: `/data/CoordExp/outputs/painted_gt/pvci_step1`.

## Task 1: Mark Geometry Leakage Analyzer

**Files:**
- Create: `src/painted_gt/mark_geometry_leakage.py`
- Create: `scripts/probes/painted_gt/analyze_mark_geometry_leakage.py`
- Test: `tests/painted_gt/test_mark_geometry_leakage.py`

**Interfaces:**
- Consumes: `gt_vs_pred.jsonl` rows whose relevant fields are `row_id`, `example_id`, `valid_prediction_count`, `pred`, `gt`, and `painted_gt.marks`.
- Produces: `analyze_row(row: Mapping[str, Any], *, source_path: str | None = None) -> dict[str, Any]`.
- Produces: `summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]`.
- Produces CLI:
  `python scripts/probes/painted_gt/analyze_mark_geometry_leakage.py --input <gt_vs_pred.jsonl> --output-dir <dir> --condition-name <name>`.
- Writes: `<output-dir>/mark_geometry_leakage.rows.jsonl` and `<output-dir>/mark_geometry_leakage.json`.

- [ ] **Step 1: Write failing tests for bbox math**

Add `tests/painted_gt/test_mark_geometry_leakage.py` with:

```python
from src.painted_gt.mark_geometry_leakage import bbox_iou_xyxy, bbox_l1_xyxy


def test_bbox_iou_xyxy_handles_identical_partial_and_empty() -> None:
    assert bbox_iou_xyxy([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert round(bbox_iou_xyxy([0, 0, 10, 10], [5, 5, 15, 15]), 6) == round(25 / 175, 6)
    assert bbox_iou_xyxy([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_bbox_l1_xyxy_returns_mean_absolute_coordinate_distance() -> None:
    assert bbox_l1_xyxy([0, 0, 10, 10], [1, 3, 7, 20]) == 4.25
```

Run:

```bash
pytest tests/painted_gt/test_mark_geometry_leakage.py -q
```

Expected before implementation: import failure for `src.painted_gt.mark_geometry_leakage`.

- [ ] **Step 2: Implement bbox helpers minimally**

Create `src/painted_gt/mark_geometry_leakage.py` with:

```python
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def bbox_iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = _bbox(a)
    bx1, by1, bx2, by2 = _bbox(b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else inter / union


def bbox_l1_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    aa = _bbox(a)
    bb = _bbox(b)
    return sum(abs(x - y) for x, y in zip(aa, bb, strict=True)) / 4.0


def _bbox(values: Sequence[float]) -> tuple[float, float, float, float]:
    if len(values) != 4:
        raise ValueError(f"bbox must have four values, got {len(values)}")
    return tuple(float(value) for value in values)  # type: ignore[return-value]
```

Run:

```bash
pytest tests/painted_gt/test_mark_geometry_leakage.py -q
```

Expected: the two bbox tests pass.

- [ ] **Step 3: Write failing tests for row-level leakage analysis**

Extend the test file:

```python
from src.painted_gt.mark_geometry_leakage import analyze_row, summarize_rows


def _row(*, pred_bbox, gt_bbox, mark_bbox, description="bear", mark_variant="outline_only"):
    return {
        "row_id": "row-1",
        "example_id": "example-1",
        "valid_prediction_count": 1 if pred_bbox is not None else 0,
        "pred": [] if pred_bbox is None else [{"bbox": pred_bbox, "description": description, "generated_order": 0}],
        "gt": [{"bbox": gt_bbox, "description": "bear", "object_id": "gt-1"}],
        "painted_gt": {
            "mark_coarseness_variant": mark_variant,
            "control_family": "stepwise",
            "control_name": "painted_correct",
            "target_object_id": "gt-1",
            "marks": [{"bbox_pixels": mark_bbox, "object_id": "gt-1", "render_mode": mark_variant}],
        },
    }


def test_analyze_row_compares_first_prediction_to_gt_and_mark() -> None:
    record = analyze_row(_row(pred_bbox=[0, 0, 10, 10], gt_bbox=[0, 0, 10, 10], mark_bbox=[0, 0, 20, 20]))
    assert record["has_prediction"] is True
    assert record["pred_index_used"] == 0
    assert record["class_match"] is True
    assert record["pred_gt_iou"] == 1.0
    assert record["pred_mark_iou"] == 0.25
    assert record["closer_to"] == "gt"


def test_analyze_row_records_missing_prediction_without_geometry_scores() -> None:
    record = analyze_row(_row(pred_bbox=None, gt_bbox=[0, 0, 10, 10], mark_bbox=[0, 0, 20, 20]))
    assert record["has_prediction"] is False
    assert record["pred_gt_iou"] is None
    assert record["closer_to"] == "no_prediction"


def test_summarize_rows_reports_variant_level_rates() -> None:
    records = [
        analyze_row(_row(pred_bbox=[0, 0, 10, 10], gt_bbox=[0, 0, 10, 10], mark_bbox=[0, 0, 20, 20])),
        analyze_row(_row(pred_bbox=[0, 0, 20, 20], gt_bbox=[0, 0, 10, 10], mark_bbox=[0, 0, 20, 20])),
    ]
    summary = summarize_rows(records)
    variant = summary["by_mark_variant"]["outline_only"]
    assert variant["row_count"] == 2
    assert variant["closer_to_gt_rate"] == 0.5
    assert variant["closer_to_mark_rate"] == 0.5
```

Run:

```bash
pytest tests/painted_gt/test_mark_geometry_leakage.py -q
```

Expected before implementation: failures for missing `analyze_row` and `summarize_rows`.

- [ ] **Step 4: Implement row analyzer and summary**

Add these functions in `src/painted_gt/mark_geometry_leakage.py`:

```python
def analyze_row(row: Mapping[str, Any], *, source_path: str | None = None) -> dict[str, Any]:
    pred_items = list(row.get("pred") or [])
    gt_items = list(row.get("gt") or [])
    painted_gt = dict(row.get("painted_gt") or {})
    marks = list(painted_gt.get("marks") or [])
    gt = gt_items[0] if gt_items else {}
    mark = marks[0] if marks else {}
    pred = pred_items[0] if pred_items else None
    variant = str(painted_gt.get("mark_coarseness_variant") or mark.get("mark_coarseness_variant") or "unknown")
    base = {
        "source_path": source_path,
        "row_id": row.get("row_id"),
        "example_id": row.get("example_id"),
        "control_family": painted_gt.get("control_family"),
        "control_name": painted_gt.get("control_name"),
        "mark_coarseness_variant": variant,
        "target_object_id": painted_gt.get("target_object_id") or gt.get("object_id"),
        "gt_description": gt.get("description"),
        "gt_bbox": gt.get("bbox"),
        "mark_bbox": mark.get("bbox_pixels"),
        "mark_source_bbox": mark.get("source_bbox_pixels"),
        "mark_render_mode": mark.get("render_mode"),
        "valid_prediction_count": row.get("valid_prediction_count", len(pred_items)),
        "dropped_prediction_count": row.get("dropped_prediction_count", 0),
        "has_prediction": pred is not None,
        "pred_index_used": 0 if pred is not None else None,
    }
    if pred is None or not gt.get("bbox") or not mark.get("bbox_pixels"):
        return {
            **base,
            "pred_description": None if pred is None else pred.get("description"),
            "pred_bbox": None if pred is None else pred.get("bbox"),
            "class_match": False,
            "pred_gt_iou": None,
            "pred_mark_iou": None,
            "pred_gt_l1": None,
            "pred_mark_l1": None,
            "closer_to": "no_prediction" if pred is None else "missing_reference",
        }
    pred_bbox = pred["bbox"]
    gt_bbox = gt["bbox"]
    mark_bbox = mark["bbox_pixels"]
    pred_gt_l1 = bbox_l1_xyxy(pred_bbox, gt_bbox)
    pred_mark_l1 = bbox_l1_xyxy(pred_bbox, mark_bbox)
    if pred_gt_l1 < pred_mark_l1:
        closer_to = "gt"
    elif pred_mark_l1 < pred_gt_l1:
        closer_to = "mark"
    else:
        closer_to = "tie"
    return {
        **base,
        "pred_description": pred.get("description"),
        "pred_bbox": pred_bbox,
        "class_match": _normalize_desc(pred.get("description")) == _normalize_desc(gt.get("description")),
        "pred_gt_iou": bbox_iou_xyxy(pred_bbox, gt_bbox),
        "pred_mark_iou": bbox_iou_xyxy(pred_bbox, mark_bbox),
        "pred_gt_l1": pred_gt_l1,
        "pred_mark_l1": pred_mark_l1,
        "closer_to": closer_to,
    }


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "metric_name": "mark_geometry_leakage_v1",
        "row_count": len(rows),
        "overall": _summarize_group(rows),
        "by_mark_variant": {
            key: _summarize_group([row for row in rows if row.get("mark_coarseness_variant") == key])
            for key in sorted({str(row.get("mark_coarseness_variant")) for row in rows})
        },
    }


def _summarize_group(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pred_rows = [row for row in rows if row.get("has_prediction")]
    return {
        "row_count": len(rows),
        "prediction_count": len(pred_rows),
        "prediction_rate": _ratio(len(pred_rows), len(rows)),
        "class_match_rate": _ratio(sum(1 for row in pred_rows if row.get("class_match")), len(pred_rows)),
        "closer_to_gt_rate": _ratio(sum(1 for row in pred_rows if row.get("closer_to") == "gt"), len(pred_rows)),
        "closer_to_mark_rate": _ratio(sum(1 for row in pred_rows if row.get("closer_to") == "mark"), len(pred_rows)),
        "closer_tie_rate": _ratio(sum(1 for row in pred_rows if row.get("closer_to") == "tie"), len(pred_rows)),
        "mean_pred_gt_iou": _mean(row.get("pred_gt_iou") for row in pred_rows),
        "mean_pred_mark_iou": _mean(row.get("pred_mark_iou") for row in pred_rows),
        "mean_pred_gt_l1": _mean(row.get("pred_gt_l1") for row in pred_rows),
        "mean_pred_mark_l1": _mean(row.get("pred_mark_l1") for row in pred_rows),
    }
```

Also add `_normalize_desc`, `_ratio`, and `_mean` helpers in the same file. `_normalize_desc` lowercases and strips whitespace. `_ratio` returns `None` for denominator `0`. `_mean` skips `None` and returns `None` for no values.

Run:

```bash
pytest tests/painted_gt/test_mark_geometry_leakage.py -q
```

Expected: pass.

- [ ] **Step 5: Add CLI and test it against a tiny JSONL fixture**

Add `scripts/probes/painted_gt/analyze_mark_geometry_leakage.py`:

```python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.painted_gt.mark_geometry_leakage import analyze_row, summarize_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze painted-mark geometry leakage.")
    parser.add_argument("--input", action="append", required=True, help="gt_vs_pred.jsonl path; repeatable.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--condition-name", default="mark_geometry_leakage")
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for raw_path in args.input:
        path = Path(raw_path).expanduser().resolve()
        for row in _read_jsonl(path):
            rows.append(analyze_row(row, source_path=str(path)))
    summary = summarize_rows(rows)
    summary["condition_name"] = args.condition_name
    rows_path = output_dir / "mark_geometry_leakage.rows.jsonl"
    summary_path = output_dir / "mark_geometry_leakage.json"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False, ensure_ascii=True, sort_keys=True) + "\n")
    summary["rows_jsonl"] = str(rows_path)
    summary_path.write_text(
        json.dumps(summary, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": str(summary_path), "row_count": len(rows)}, sort_keys=True))
    return 0


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise SystemExit(f"row must be a JSON object: {path}:{line_number}")
            rows.append(payload)
    if not rows:
        raise SystemExit(f"input has no rows: {path}")
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
```

Add a CLI test to `tests/painted_gt/test_mark_geometry_leakage.py` that writes two JSONL rows, invokes `main` with an explicit argument list, and asserts `mark_geometry_leakage.json` plus `mark_geometry_leakage.rows.jsonl` exist and contain `row_count == 2`.

Run:

```bash
pytest tests/painted_gt/test_mark_geometry_leakage.py -q
python -m py_compile src/painted_gt/mark_geometry_leakage.py scripts/probes/painted_gt/analyze_mark_geometry_leakage.py
```

- [ ] **Step 6: Validate on existing Step-0 artifacts**

Run:

```bash
python scripts/probes/painted_gt/analyze_mark_geometry_leakage.py \
  --input /data/CoordExp/outputs/painted_gt/pvci_step0/inference/coarseness_val32/outline_only_step484_rp110_bs2/gt_vs_pred.jsonl \
  --output-dir /data/CoordExp/outputs/painted_gt/pvci_step1/analysis/step0_existing/outline_only \
  --condition-name step0_outline_only
```

Expected:

- JSON printed with `"row_count": 256`;
- `/data/CoordExp/outputs/painted_gt/pvci_step1/analysis/step0_existing/outline_only/mark_geometry_leakage.json` exists;
- no code changes beyond the analyzer and tests.

- [ ] **Step 7: Commit Task 1**

Run:

```bash
git add src/painted_gt/mark_geometry_leakage.py \
  scripts/probes/painted_gt/analyze_mark_geometry_leakage.py \
  tests/painted_gt/test_mark_geometry_leakage.py
git diff --cached --check
git commit -m "feat: add painted mark geometry leakage analyzer"
```

## Task 2: Boundary-Ablation Variants

**Files:**
- Modify: `src/painted_gt/counterfactuals.py`
- Test: `tests/painted_gt/test_counterfactual_controls.py`

**Interfaces:**
- Extends `MARK_COARSENESS_VARIANTS` with:
  `top_edge_only`, `bottom_edge_only`, `left_edge_only`, `right_edge_only`,
  `top_left_edges`, `four_corners_only`, `dashed_outline`, `inner_outline`,
  `shifted_single_edge`, `wrong_aspect_outline`, `background_outline`.
- Extends `_mark_with_coarseness` to emit `render_mode`, `bbox_pixels`, `source_bbox_pixels`, `center_pixels`, `nearest_gt_iou`, and variant-specific metadata.
- Extends `_materialize_control_image` to render each new `render_mode` deterministically.

- [ ] **Step 1: Write failing tests for variant registry and metadata**

In `tests/painted_gt/test_counterfactual_controls.py`, add:

```python
from src.painted_gt.counterfactuals import MARK_COARSENESS_VARIANTS


def test_step1_boundary_variants_are_registered() -> None:
    expected = {
        "top_edge_only",
        "bottom_edge_only",
        "left_edge_only",
        "right_edge_only",
        "top_left_edges",
        "four_corners_only",
        "dashed_outline",
        "inner_outline",
        "shifted_single_edge",
        "wrong_aspect_outline",
        "background_outline",
    }
    assert expected.issubset(set(MARK_COARSENESS_VARIANTS))
```

Add a materialization test that calls `materialize_counterfactual_conditions` with `mark_coarseness_variant="four_corners_only"` and asserts the first `counterfactual_plan.jsonl` mark contains:

```python
assert mark["mark_coarseness_variant"] == "four_corners_only"
assert mark["render_mode"] == "four_corners_only"
assert mark["source_bbox_pixels"] == mark["bbox_pixels"]
assert "nearest_gt_iou" in mark
```

Run:

```bash
pytest tests/painted_gt/test_counterfactual_controls.py -q
```

Expected before implementation: failure because the variants are not registered.

- [ ] **Step 2: Register boundary variants**

Modify `MARK_COARSENESS_VARIANTS` in `src/painted_gt/counterfactuals.py` by appending exactly:

```python
    "top_edge_only",
    "bottom_edge_only",
    "left_edge_only",
    "right_edge_only",
    "top_left_edges",
    "four_corners_only",
    "dashed_outline",
    "inner_outline",
    "shifted_single_edge",
    "wrong_aspect_outline",
    "background_outline",
```

Run:

```bash
pytest tests/painted_gt/test_counterfactual_controls.py::test_step1_boundary_variants_are_registered -q
```

Expected: pass.

- [ ] **Step 3: Implement mark geometry transformations**

Extend `_mark_with_coarseness`:

```python
    elif variant in {
        "top_edge_only",
        "bottom_edge_only",
        "left_edge_only",
        "right_edge_only",
        "top_left_edges",
        "four_corners_only",
        "dashed_outline",
    }:
        render_mode = variant
    elif variant == "inner_outline":
        render_mode = "outline_only"
        bbox = _inset_bbox(
            original_bbox,
            fraction=0.08,
            image_width=raw_example.image.width,
            image_height=raw_example.image.height,
        )
        center = _center(bbox)
    elif variant == "shifted_single_edge":
        render_mode = "shifted_single_edge"
        bbox = _shifted_edge_bbox(
            original_bbox,
            image_width=raw_example.image.width,
            image_height=raw_example.image.height,
            fraction=0.12,
        )
        center = _center(bbox)
    elif variant == "wrong_aspect_outline":
        render_mode = "outline_only"
        bbox = _wrong_aspect_bbox(
            original_bbox,
            image_width=raw_example.image.width,
            image_height=raw_example.image.height,
            x_factor=1.35,
            y_factor=0.75,
        )
        center = _center(bbox)
    elif variant == "background_outline":
        render_mode = "outline_only"
        bbox = _background_bbox(
            raw_example,
            avoid_bbox=original_bbox,
        )
        center = _center(bbox)
```

Add helpers in `src/painted_gt/counterfactuals.py` near existing bbox helpers:

- `_inset_bbox(bbox, *, fraction, image_width, image_height)`;
- `_shifted_edge_bbox(bbox, *, image_width, image_height, fraction)`;
- `_wrong_aspect_bbox(bbox, *, image_width, image_height, x_factor, y_factor)`;
- `_background_bbox(raw_example, *, avoid_bbox)`.

Each helper must call `_clip_bbox` before returning. `_background_bbox` must choose a deterministic same-size box from four corners, picking the first candidate whose IoU with every GT object is below `LOW_GT_OVERLAP_IOU`; if no candidate is clean, pick the candidate with the lowest maximum GT IoU and record the actual `nearest_gt_iou`.

Run:

```bash
pytest tests/painted_gt/test_counterfactual_controls.py -q
```

Expected: metadata tests pass, rendering tests still fail until renderer modes are added.

- [ ] **Step 4: Write failing pixel-render tests**

Add tests that materialize one small fixture image and inspect the resulting PNG:

```python
def test_four_corners_only_renders_corners_without_full_edges(tmp_path: Path) -> None:
    root = _materialize_single_variant(tmp_path, "four_corners_only")
    image_path, mark = _first_image_and_mark(root)
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
    x1, y1, x2, y2 = mark["bbox_pixels"]
    magenta = (255, 0, 255)
    assert rgb.getpixel((x1, y1)) == magenta
    assert rgb.getpixel((x2, y1)) == magenta
    mid_x = round((x1 + x2) / 2)
    assert rgb.getpixel((mid_x, y1)) != magenta


def test_top_edge_only_renders_top_edge_without_bottom_edge(tmp_path: Path) -> None:
    root = _materialize_single_variant(tmp_path, "top_edge_only")
    image_path, mark = _first_image_and_mark(root)
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
    x1, y1, x2, y2 = mark["bbox_pixels"]
    magenta = (255, 0, 255)
    mid_x = round((x1 + x2) / 2)
    assert rgb.getpixel((mid_x, y1)) == magenta
    assert rgb.getpixel((mid_x, y2)) != magenta
```

Use or create local test helpers `_materialize_single_variant` and `_first_image_and_mark` in the test file. These helpers should reuse the existing fake raw-example fixture style already present in `tests/painted_gt/test_counterfactual_controls.py`.

Run:

```bash
pytest tests/painted_gt/test_counterfactual_controls.py -q
```

Expected before rendering implementation: failures because the render modes are skipped.

- [ ] **Step 5: Implement deterministic rendering modes**

Extend `_materialize_control_image` in `src/painted_gt/counterfactuals.py` after the existing style extraction:

- `top_edge_only`: draw `[x1, y1] -> [x2, y1]`;
- `bottom_edge_only`: draw `[x1, y2] -> [x2, y2]`;
- `left_edge_only`: draw `[x1, y1] -> [x1, y2]`;
- `right_edge_only`: draw `[x2, y1] -> [x2, y2]`;
- `top_left_edges`: draw top edge and left edge;
- `four_corners_only`: draw short corner line segments with length `max(thickness * 4, min(width, height) // 12)`;
- `dashed_outline`: draw each edge as repeated dash segments with dash length `max(thickness * 4, 6)` and gap length `max(thickness * 3, 4)`;
- `shifted_single_edge`: draw the shifted top edge stored in `bbox_pixels`;
- `outline_only`: already supported and reused by `inner_outline`, `wrong_aspect_outline`, and `background_outline`.

Do not add fill, class text, ids, or color changes.

Run:

```bash
pytest tests/painted_gt/test_counterfactual_controls.py tests/painted_gt/test_counterfactual_materialize_cli.py -q
python -m py_compile src/painted_gt/counterfactuals.py scripts/probes/painted_gt/materialize_counterfactual_conditions.py
```

- [ ] **Step 6: Commit Task 2**

Run:

```bash
git add src/painted_gt/counterfactuals.py tests/painted_gt/test_counterfactual_controls.py
git diff --cached --check
git commit -m "feat: add PVCI boundary ablation marks"
```

## Task 3: Step-1 Inference Config Builder

**Files:**
- Create: `scripts/probes/painted_gt/build_pvci_step1_infer_configs.py`
- Test: `tests/painted_gt/test_pvci_step1_config_builder.py`
- Output when executed: `configs/coordexp_swift/infer/painted_gt/pvci_step1/*.yaml`

**Interfaces:**
- Consumes a materialized root containing `conditions/stepwise__painted_correct/stepwise.painted_correct.examples.jsonl`.
- Produces YAML configs extending `../stepwise_counterfactual_painted_correct_overfit16_step484_rp110_gate256.yaml`.
- Preserves frozen decode: `batch_size: 2`, `max_new_tokens: 96`, `temperature: 0.0`, `top_p: 1.0`, `repetition_penalty: 1.10`.

- [ ] **Step 1: Write failing tests for config rendering**

Create `tests/painted_gt/test_pvci_step1_config_builder.py`:

```python
from pathlib import Path
import importlib.util
import yaml

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/probes/painted_gt/build_pvci_step1_infer_configs.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_pvci_step1_infer_configs", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_render_config_preserves_frozen_decode_surface() -> None:
    module = _load_module()
    payload = module.render_config_payload(
        variant="four_corners_only",
        example_jsonl="/tmp/examples.jsonl",
        artifact_root="/tmp/inference",
        output_dir="boundary_val32_four_corners_only_step484_rp110_bs2",
        run_name="pvci-step1-boundary-val32-four-corners-only-step484-rp110-bs2",
    )
    assert payload["extends"] == "../stepwise_counterfactual_painted_correct_overfit16_step484_rp110_gate256.yaml"
    assert payload["data"]["input_jsonl"] == "/tmp/examples.jsonl"
    assert payload["generation"] == {
        "batch_size": 2,
        "max_new_tokens": 96,
        "temperature": 0.0,
        "top_p": 1.0,
        "repetition_penalty": 1.10,
    }


def test_write_config_matrix_creates_one_yaml_per_variant(tmp_path: Path) -> None:
    module = _load_module()
    materialized_root = tmp_path / "materialized"
    for variant in ("outline_only", "four_corners_only"):
        condition = materialized_root / variant / "conditions/stepwise__painted_correct"
        condition.mkdir(parents=True)
        (condition / "stepwise.painted_correct.examples.jsonl").write_text('{"example_id":"x"}\n', encoding="utf-8")
    config_root = tmp_path / "configs"
    written = module.write_config_matrix(
        materialized_root=materialized_root,
        config_root=config_root,
        inference_artifact_root=Path("/tmp/inference"),
        scope="boundary_val32",
        variants=("outline_only", "four_corners_only"),
    )
    assert len(written) == 2
    loaded = yaml.safe_load(written[0].read_text())
    assert loaded["generation"]["batch_size"] == 2
```

Run:

```bash
pytest tests/painted_gt/test_pvci_step1_config_builder.py -q
```

Expected before implementation: script import failure.

- [ ] **Step 2: Implement config builder**

Create `scripts/probes/painted_gt/build_pvci_step1_infer_configs.py` with functions:

```python
def render_config_payload(
    *,
    variant: str,
    example_jsonl: str,
    artifact_root: str,
    output_dir: str,
    run_name: str,
) -> dict:
    return {
        "extends": "../stepwise_counterfactual_painted_correct_overfit16_step484_rp110_gate256.yaml",
        "run": {
            "name": run_name,
            "artifact_root": artifact_root,
            "output_dir": output_dir,
            "collision_policy": "fail",
        },
        "data": {"input_jsonl": example_jsonl},
        "generation": {
            "batch_size": 2,
            "max_new_tokens": 96,
            "temperature": 0.0,
            "top_p": 1.0,
            "repetition_penalty": 1.10,
        },
        "scoring": {"enabled": True},
        "debug": {"smoke": False, "dry_run": False},
    }
```

Also implement:

```python
def write_config_matrix(
    *,
    materialized_root: Path,
    config_root: Path,
    inference_artifact_root: Path,
    scope: str,
    variants: Sequence[str],
) -> list[Path]:
    config_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for variant in variants:
        example_jsonl = (
            materialized_root
            / variant
            / "conditions"
            / "stepwise__painted_correct"
            / "stepwise.painted_correct.examples.jsonl"
        )
        if not example_jsonl.is_file():
            raise SystemExit(f"missing materialized examples for {variant}: {example_jsonl}")
        output_dir = f"{scope}_{variant}_step484_rp110_bs2"
        payload = render_config_payload(
            variant=variant,
            example_jsonl=str(example_jsonl.resolve()),
            artifact_root=str(inference_artifact_root.resolve()),
            output_dir=output_dir,
            run_name=f"pvci-step1-{scope.replace('_', '-')}-{variant.replace('_', '-')}-step484-rp110-bs2",
        )
        path = config_root / f"{output_dir}.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        written.append(path)
    return written
```

The config writer uses `yaml.safe_dump(payload, sort_keys=False)`. File names must be:

```text
{scope}_{variant}_step484_rp110_bs2.yaml
```

Run:

```bash
pytest tests/painted_gt/test_pvci_step1_config_builder.py -q
python -m py_compile scripts/probes/painted_gt/build_pvci_step1_infer_configs.py
```

- [ ] **Step 3: Commit Task 3**

Run:

```bash
git add scripts/probes/painted_gt/build_pvci_step1_infer_configs.py \
  tests/painted_gt/test_pvci_step1_config_builder.py
git diff --cached --check
git commit -m "tool: add PVCI step1 inference config builder"
```

## Task 4: Route Report Aggregator

**Files:**
- Create: `scripts/probes/painted_gt/write_pvci_step1_route_report.py`
- Test: `tests/painted_gt/test_pvci_step1_route_report.py`

**Interfaces:**
- Consumes per-run `per_target_step_debug_f1.json`.
- Consumes per-run `mark_geometry_leakage.json`.
- Produces `pvci_step1_route_summary.json`.
- Produces `pvci_step1_route_report.md`.

- [ ] **Step 1: Write failing tests for report aggregation**

Create `tests/painted_gt/test_pvci_step1_route_report.py`:

```python
from pathlib import Path
import importlib.util
import json

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/probes/painted_gt/write_pvci_step1_route_report.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("write_pvci_step1_route_report", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_route_summary_keeps_debug_metric_and_leakage_side_by_side(tmp_path: Path) -> None:
    module = _load_module()
    run_dir = tmp_path / "four_corners"
    analysis_dir = tmp_path / "analysis_four_corners"
    run_dir.mkdir()
    analysis_dir.mkdir()
    (run_dir / "per_target_step_debug_f1.json").write_text(json.dumps({
        "metric_name": "per_target_step_debug_f1",
        "row_count": 32,
        "f1": 0.5,
        "precision": 0.6,
        "recall": 0.43,
        "step_validity": 0.9,
    }), encoding="utf-8")
    (analysis_dir / "mark_geometry_leakage.json").write_text(json.dumps({
        "metric_name": "mark_geometry_leakage_v1",
        "row_count": 32,
        "overall": {
            "closer_to_gt_rate": 0.75,
            "closer_to_mark_rate": 0.25,
            "mean_pred_gt_iou": 0.6,
            "mean_pred_mark_iou": 0.4,
        },
        "by_mark_variant": {"four_corners_only": {"row_count": 32}},
    }), encoding="utf-8")
    summary = module.build_route_summary([
        module.VariantInputs(
            variant="four_corners_only",
            run_dir=run_dir,
            leakage_dir=analysis_dir,
            scope="val32",
        )
    ])
    assert summary["variant_count"] == 1
    assert summary["variants"][0]["variant"] == "four_corners_only"
    assert summary["variants"][0]["f1"] == 0.5
    assert summary["variants"][0]["closer_to_gt_rate"] == 0.75
```

Run:

```bash
pytest tests/painted_gt/test_pvci_step1_route_report.py -q
```

Expected before implementation: import failure.

- [ ] **Step 2: Implement aggregator**

Create `scripts/probes/painted_gt/write_pvci_step1_route_report.py` with:

- dataclass `VariantInputs(variant: str, run_dir: Path, leakage_dir: Path, scope: str)`;
- `build_route_summary(inputs: Sequence[VariantInputs]) -> dict[str, Any]`;
- `write_route_report(summary: Mapping[str, Any], *, output_md: Path, output_json: Path) -> None`;
- CLI args:
  - `--variant variant=scope=/path/to/run_dir=/path/to/leakage_dir`, repeatable;
  - `--output-md`;
  - `--output-json`.

Markdown report sections:

```text
# PVCI Step-1 Route Report
## Scope
## Variant Summary
## Leakage Interpretation
## Boundary Ablation Interpretation
## Route Verdict
## Caveats
## Next Probe Recommendation
```

The initial route verdict should be conservative and data-driven:

- if no variants are provided, exit nonzero;
- if `outline_only` and `tight_outline_center` are absent, state that baseline controls are missing;
- if any loose or shifted variants have `closer_to_mark_rate > closer_to_gt_rate`, flag geometry-leakage risk;
- if partial/corner variants preserve at least half of `outline_only` F1, flag lightweight boundary cursor plausible;
- always state that this is target-row debug evidence, not final mAP.

Run:

```bash
pytest tests/painted_gt/test_pvci_step1_route_report.py -q
python -m py_compile scripts/probes/painted_gt/write_pvci_step1_route_report.py
```

- [ ] **Step 3: Commit Task 4**

Run:

```bash
git add scripts/probes/painted_gt/write_pvci_step1_route_report.py \
  tests/painted_gt/test_pvci_step1_route_report.py
git diff --cached --check
git commit -m "tool: add PVCI step1 route report"
```

## Task 5: Materialize Val32 Boundary Matrix And Generate Configs

**Files:**
- Output: `/data/CoordExp/outputs/painted_gt/pvci_step1/materialized/boundary_val32/`
- Output: `configs/coordexp_swift/infer/painted_gt/pvci_step1/`

**Interfaces:**
- Uses existing materializer script:
  `scripts/probes/painted_gt/materialize_counterfactual_conditions.py`.
- Uses config builder from Task 3.

- [ ] **Step 1: Materialize every val32 variant**

Run this loop from repo root:

```bash
VARIANTS=(
  tight_outline_center
  outline_only
  box_1p5_outline_center
  box_2p0_outline_center
  grid_snapped_box_outline_center
  center_point
  center_blob
  semi_transparent_fill
  top_edge_only
  bottom_edge_only
  left_edge_only
  right_edge_only
  top_left_edges
  four_corners_only
  dashed_outline
  inner_outline
  shifted_single_edge
  wrong_aspect_outline
  background_outline
)
for variant in "${VARIANTS[@]}"; do
  python scripts/probes/painted_gt/materialize_counterfactual_conditions.py \
    --input-jsonl /data/CoordExp/.worktrees/CoordExp-swift/outputs/coordexp_swift/infer/val200_inputs/coco_val200_len12000.rebased_images.coord.jsonl \
    --output-root /data/CoordExp/outputs/painted_gt/pvci_step1/materialized/boundary_val32/"${variant}" \
    --schedule-kind geo_sorted \
    --size 32 \
    --sample-limit 200 \
    --condition stepwise__painted_correct \
    --mark-coarseness-variant "${variant}" \
    --skip-preflight \
    --force
done
```

Expected:

- each variant root has `counterfactual_conditions_manifest.json`;
- each variant root has `conditions/stepwise__painted_correct/stepwise.painted_correct.examples.jsonl`;
- each condition has `row_count` near the object count of the selected 32 images, not necessarily exactly 32.

- [ ] **Step 2: Validate manifests and row counts**

Run:

```bash
python - <<'PY'
from pathlib import Path
import json
root = Path('/data/CoordExp/outputs/painted_gt/pvci_step1/materialized/boundary_val32')
bad = []
for manifest in sorted(root.glob('*/counterfactual_conditions_manifest.json')):
    data = json.loads(manifest.read_text())
    cond = data['conditions']['stepwise__painted_correct']
    example_jsonl = Path(cond['example_jsonl'])
    row_count = sum(1 for _ in example_jsonl.open())
    print(manifest.parent.name, row_count, cond['row_count'])
    if row_count != cond['row_count'] or row_count <= 0:
        bad.append(str(manifest))
if bad:
    raise SystemExit({'bad': bad})
PY
```

- [ ] **Step 3: Generate inference configs**

Run:

```bash
python scripts/probes/painted_gt/build_pvci_step1_infer_configs.py \
  --materialized-root /data/CoordExp/outputs/painted_gt/pvci_step1/materialized/boundary_val32 \
  --config-root configs/coordexp_swift/infer/painted_gt/pvci_step1 \
  --inference-artifact-root /data/CoordExp/outputs/painted_gt/pvci_step1/inference/boundary_val32 \
  --scope boundary_val32
```

Expected:

- one YAML per variant under `configs/coordexp_swift/infer/painted_gt/pvci_step1/`;
- each YAML extends `../stepwise_counterfactual_painted_correct_overfit16_step484_rp110_gate256.yaml`;
- each YAML points to the matching materialized `stepwise.painted_correct.examples.jsonl`.

- [ ] **Step 4: Parse generated YAMLs**

Run:

```bash
python - <<'PY'
from pathlib import Path
import yaml
root = Path('configs/coordexp_swift/infer/painted_gt/pvci_step1')
paths = sorted(root.glob('boundary_val32_*_step484_rp110_bs2.yaml'))
for path in paths:
    payload = yaml.safe_load(path.read_text())
    assert payload['generation']['batch_size'] == 2, path
    assert payload['generation']['max_new_tokens'] == 96, path
    assert payload['generation']['repetition_penalty'] == 1.10, path
print('yaml_ok', len(paths))
PY
```

Expected: `yaml_ok 19`.

- [ ] **Step 5: Commit generated configs**

Run:

```bash
git add configs/coordexp_swift/infer/painted_gt/pvci_step1
git diff --cached --check
git commit -m "config: add PVCI step1 val32 inference matrix"
```

## Task 6: Run Val32 Inference, Metrics, And Leakage

**Files:**
- Output: `/data/CoordExp/outputs/painted_gt/pvci_step1/inference/boundary_val32/`
- Output: `/data/CoordExp/outputs/painted_gt/pvci_step1/analysis/boundary_val32/`
- Output: `/data/CoordExp/outputs/painted_gt/pvci_step1/reports/val32/`

**Interfaces:**
- Consumes configs generated in Task 5.
- Runs `python -m src.infer --config <config>`.
- Computes `per_target_step_debug_f1.json`.
- Computes `mark_geometry_leakage.json`.

- [ ] **Step 1: Run inference for every val32 config**

Run sequentially unless GPU memory permits parallel tmux lanes:

```bash
for config in configs/coordexp_swift/infer/painted_gt/pvci_step1/boundary_val32_*_step484_rp110_bs2.yaml; do
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python -m src.infer --config "$config"
done
```

Expected per run:

- `summary.json` exists in the configured run directory;
- `gt_vs_pred.jsonl` exists;
- no OOM or adapter-loading failure.

- [ ] **Step 2: Compute debug F1 for every run**

Run:

```bash
for run_dir in /data/CoordExp/outputs/painted_gt/pvci_step1/inference/boundary_val32/*_step484_rp110_bs2; do
  python scripts/probes/painted_gt/write_debug_metric_report.py \
    --mode stepwise_teacher_prefix \
    --run-dir "$run_dir"
done
```

Expected: each run directory has `per_target_step_debug_f1.json`.

- [ ] **Step 3: Compute leakage for every run**

Run:

```bash
for run_dir in /data/CoordExp/outputs/painted_gt/pvci_step1/inference/boundary_val32/*_step484_rp110_bs2; do
  variant="$(basename "$run_dir" | sed 's/^boundary_val32_//' | sed 's/_step484_rp110_bs2$//')"
  python scripts/probes/painted_gt/analyze_mark_geometry_leakage.py \
    --input "$run_dir/gt_vs_pred.jsonl" \
    --output-dir /data/CoordExp/outputs/painted_gt/pvci_step1/analysis/boundary_val32/"$variant" \
    --condition-name "boundary_val32_${variant}"
done
```

Expected: each analysis dir has `mark_geometry_leakage.json` and `mark_geometry_leakage.rows.jsonl`.

- [ ] **Step 4: Build first route report**

Run the route report script with all variants. If the CLI accepts repeated `--variant` values, generate them in shell:

```bash
args=()
for run_dir in /data/CoordExp/outputs/painted_gt/pvci_step1/inference/boundary_val32/*_step484_rp110_bs2; do
  variant="$(basename "$run_dir" | sed 's/^boundary_val32_//' | sed 's/_step484_rp110_bs2$//')"
  args+=(--variant "${variant}=val32=${run_dir}=/data/CoordExp/outputs/painted_gt/pvci_step1/analysis/boundary_val32/${variant}")
done
python scripts/probes/painted_gt/write_pvci_step1_route_report.py \
  "${args[@]}" \
  --output-md /data/CoordExp/outputs/painted_gt/pvci_step1/reports/val32/pvci_step1_route_report.md \
  --output-json /data/CoordExp/outputs/painted_gt/pvci_step1/reports/val32/pvci_step1_route_summary.json
```

Expected:

- report exists;
- JSON summary contains all `19` variants;
- report explicitly labels scope `val32`.

- [ ] **Step 5: Decide selective val100 promotion set**

Use the `val32` route summary. Promote at least:

- `tight_outline_center`;
- `outline_only`;
- the strongest partial-boundary variant;
- the weakest but informative partial-boundary variant;
- `background_outline`;
- any loose or shifted variant with ambiguous `closer_to_gt_rate` versus `closer_to_mark_rate`.

Do not promote every variant unless the `val32` report is noisy enough that the route would otherwise be unreliable.

## Task 7: Selective Val100 Promotion And Final Report

**Files:**
- Output: `/data/CoordExp/outputs/painted_gt/pvci_step1/materialized/boundary_val100/`
- Output: `/data/CoordExp/outputs/painted_gt/pvci_step1/inference/boundary_val100/`
- Output: `/data/CoordExp/outputs/painted_gt/pvci_step1/analysis/boundary_val100/`
- Output: `/data/CoordExp/outputs/painted_gt/pvci_step1/reports/final/`
- Create: `research/ideas/qwen3-vl-painted-gt-transcription-probe/pvci-step1-precommit-probes-2026-07-08.md`

**Interfaces:**
- Uses the same materialization, config, inference, metric, leakage, and report scripts as Tasks 5 and 6.
- Final conclusion must be one of:
  `proceed_feature_cursor`, `proceed_hidden_cursor`, `boundary_precision_required`, `geometry_leakage_dominant`, `inconclusive_requires_phase2_probe`, or `stop_direction`.

- [ ] **Step 1: Materialize selected val100 variants**

For each selected variant:

```bash
python scripts/probes/painted_gt/materialize_counterfactual_conditions.py \
  --input-jsonl /data/CoordExp/.worktrees/CoordExp-swift/outputs/coordexp_swift/infer/val200_inputs/coco_val200_len12000.rebased_images.coord.jsonl \
  --output-root /data/CoordExp/outputs/painted_gt/pvci_step1/materialized/boundary_val100/"${variant}" \
  --schedule-kind geo_sorted \
  --size 100 \
  --sample-limit 200 \
  --condition stepwise__painted_correct \
  --mark-coarseness-variant "${variant}" \
  --skip-preflight \
  --force
```

- [ ] **Step 2: Generate val100 configs**

Run:

```bash
python scripts/probes/painted_gt/build_pvci_step1_infer_configs.py \
  --materialized-root /data/CoordExp/outputs/painted_gt/pvci_step1/materialized/boundary_val100 \
  --config-root configs/coordexp_swift/infer/painted_gt/pvci_step1 \
  --inference-artifact-root /data/CoordExp/outputs/painted_gt/pvci_step1/inference/boundary_val100 \
  --scope boundary_val100 \
  --variants "${selected_variants_csv}"
```

The config builder must accept a comma-separated `--variants` value in addition to default all-subdirs discovery.

- [ ] **Step 3: Run val100 inference, metrics, and leakage**

Run the same command patterns as Task 6, replacing `boundary_val32` with `boundary_val100`.

- [ ] **Step 4: Generate final route report**

Run `write_pvci_step1_route_report.py` with both val32 and val100 variants and outputs:

```text
/data/CoordExp/outputs/painted_gt/pvci_step1/reports/final/pvci_step1_route_report.md
/data/CoordExp/outputs/painted_gt/pvci_step1/reports/final/pvci_step1_route_summary.json
```

The final report must include:

- fixed checkpoint/decode surface;
- artifact roots;
- variant table for val32;
- promoted variant table for val100;
- leakage interpretation;
- boundary-dependence interpretation;
- route verdict;
- caveats and next probe recommendation.

- [ ] **Step 5: Write durable research note**

Create `research/ideas/qwen3-vl-painted-gt-transcription-probe/pvci-step1-precommit-probes-2026-07-08.md` with:

```md
# PVCI Step-1 Pre-Commit Probes - 2026-07-08

## Scope

## Fixed Surfaces

## Artifact Roots

## Val32 Boundary Matrix

## Val100 Promotion

## Identity Versus Geometry Leakage

## Boundary Ablation

## Route Verdict

## Interpretation

## Caveats

## Next Step
```

The note must link, not duplicate, large JSONL artifacts.

- [ ] **Step 6: Update research index**

Modify `research/ideas/qwen3-vl-painted-gt-transcription-probe/index.md` and `overview.md` to link the Step-1 note and summarize the route verdict in one short paragraph.

- [ ] **Step 7: Commit final docs/configs**

Run:

```bash
git add configs/coordexp_swift/infer/painted_gt/pvci_step1 \
  research/ideas/qwen3-vl-painted-gt-transcription-probe/pvci-step1-precommit-probes-2026-07-08.md \
  research/ideas/qwen3-vl-painted-gt-transcription-probe/index.md \
  research/ideas/qwen3-vl-painted-gt-transcription-probe/overview.md
git diff --cached --check
git commit -m "docs: record PVCI step1 route verdict"
```

## Task 8: Final Verification And Handoff

**Files:**
- No required source edits.
- Optional: create handoff file with the `handoff` skill if context is long.

**Interfaces:**
- Confirms source tests, script compile, config parse, and artifact completeness.
- Confirms route conclusion is evidence-scoped and not overstated.

- [ ] **Step 1: Run focused tests**

Run:

```bash
pytest tests/painted_gt/test_mark_geometry_leakage.py \
  tests/painted_gt/test_counterfactual_controls.py \
  tests/painted_gt/test_counterfactual_materialize_cli.py \
  tests/painted_gt/test_pvci_step1_config_builder.py \
  tests/painted_gt/test_pvci_step1_route_report.py -q
```

- [ ] **Step 2: Compile touched scripts**

Run:

```bash
python -m py_compile \
  src/painted_gt/mark_geometry_leakage.py \
  src/painted_gt/counterfactuals.py \
  scripts/probes/painted_gt/analyze_mark_geometry_leakage.py \
  scripts/probes/painted_gt/build_pvci_step1_infer_configs.py \
  scripts/probes/painted_gt/write_pvci_step1_route_report.py \
  scripts/probes/painted_gt/materialize_counterfactual_conditions.py
```

- [ ] **Step 3: Parse generated YAMLs**

Run:

```bash
python - <<'PY'
from pathlib import Path
import yaml
paths = sorted(Path('configs/coordexp_swift/infer/painted_gt/pvci_step1').glob('*.yaml'))
for path in paths:
    with path.open() as handle:
        yaml.safe_load(handle)
print('yaml_ok', len(paths))
PY
```

- [ ] **Step 4: Verify required artifacts exist**

Run:

```bash
python - <<'PY'
from pathlib import Path
required = [
    Path('/data/CoordExp/outputs/painted_gt/pvci_step1/reports/val32/pvci_step1_route_report.md'),
    Path('/data/CoordExp/outputs/painted_gt/pvci_step1/reports/val32/pvci_step1_route_summary.json'),
    Path('/data/CoordExp/outputs/painted_gt/pvci_step1/reports/final/pvci_step1_route_report.md'),
    Path('/data/CoordExp/outputs/painted_gt/pvci_step1/reports/final/pvci_step1_route_summary.json'),
    Path('research/ideas/qwen3-vl-painted-gt-transcription-probe/pvci-step1-precommit-probes-2026-07-08.md'),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit({'missing': missing})
print('artifact_ok', len(required))
PY
```

- [ ] **Step 5: Run diff hygiene**

Run:

```bash
git diff --check
git status --short --branch
```

- [ ] **Step 6: Final response contract**

Report:

- route verdict;
- metrics table handle;
- artifact roots;
- changed files and commits;
- verification commands and outcomes;
- skipped checks;
- recommended next branch: feature-space cursor, hidden cursor, teacher strengthening, or stop.

## Execution Policy

Use `superpowers:subagent-driven-development` for implementation and review because Tasks 1 through 4 are separable code/tooling slices and Tasks 5 through 8 are execution/reporting slices. Each code task must be TDD-first. After each code task, run task-scoped review before moving on. For GPU execution tasks, use a read-only audit pass before interpreting the final route report.

The work is complete only when the final research note exists, the final route report exists, and the final response gives a concrete route verdict or a clearly scoped inconclusive verdict with the smallest next probe that would resolve it.
