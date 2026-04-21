from __future__ import annotations

import json
from pathlib import Path

from src.analysis.coord_family_invalid_geometry_audit import run_invalid_geometry_audit


def test_run_invalid_geometry_audit_classifies_overflow_and_extent_failures(tmp_path: Path) -> None:
    gt_vs_pred_path = tmp_path / "gt_vs_pred.jsonl"
    output_dir = tmp_path / "analysis"
    gt_vs_pred_path.write_text(
        json.dumps(
            {
                "image": "img.png",
                "image_id": 7,
                "width": 100,
                "height": 80,
                "errors": ["invalid_geometry", "invalid_geometry"],
                "error_entries": [
                    {"code": "invalid_geometry"},
                    {"code": "invalid_geometry"},
                ],
                "raw_output_json": {
                    "objects": [
                        {"desc": "person", "bbox_2d": [10, 10, 50, 120]},
                        {"desc": "cup", "bbox_2d": [30, 20, 20, 40]},
                        {"desc": "cat", "bbox_2d": [1, 1, 10, 10]},
                    ]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        f"""
run:
  name: coord-family-invalid-geometry-smoke
  output_dir: {output_dir.as_posix()}

families:
  - family_alias: base_xyxy_merged
    gt_vs_pred_jsonl: {gt_vs_pred_path.as_posix()}
    raw_coord_mode: pixel
        """.strip(),
        encoding="utf-8",
    )

    result = run_invalid_geometry_audit(config_path, repo_root=tmp_path)

    summary_path = output_dir / "coord-family-invalid-geometry-smoke" / "summary.json"
    rows_path = output_dir / "coord-family-invalid-geometry-smoke" / "invalid_rows.jsonl"
    assert result["summary_json"] == str(summary_path)
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    family = summary["families"]["base_xyxy_merged"]
    assert family["invalid_geometry_error_entries"] == 2
    assert family["invalid_raw_object_count"] == 2
    assert family["alignment_rate"] == 1.0
    failure_families = {row["failure_family"] for row in rows}
    assert "overflow_bottom" in failure_families
    assert "non_positive_width" in failure_families
