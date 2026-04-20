from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from src.analysis.raw_text_coord_manual_review import (
    build_contact_sheet,
    build_manual_review_bundle,
)


def test_build_contact_sheet_writes_png(tmp_path: Path) -> None:
    panel_path = tmp_path / "panel.png"
    Image.new("RGB", (120, 80), color=(200, 100, 50)).save(panel_path)
    out_path = tmp_path / "sheet.png"
    build_contact_sheet(
        case_id="case-a",
        panels=[
            {
                "model_alias": "base",
                "variant": "baseline",
                "center_kind": "pred",
                "figure_path": str(panel_path),
            }
        ],
        output_path=out_path,
        title="focus",
    )
    assert out_path.exists()


def test_build_manual_review_bundle_materializes_interface(tmp_path: Path) -> None:
    final_bundle_dir = tmp_path / "final"
    final_bundle_dir.mkdir()
    (final_bundle_dir / "summary.json").write_text(
        json.dumps(
            {
                "verdicts": {
                    "q1_base_numeric_continuity": {"verdict": "strongly supported"},
                    "q2_pure_ce_enhances_continuity": {"verdict": "partially supported"},
                }
            }
        ),
        encoding="utf-8",
    )
    heatmap_dir = tmp_path / "heatmap"
    (heatmap_dir / "figures").mkdir(parents=True)
    image_path = heatmap_dir / "figures" / "case-a__base__baseline__pred.png"
    Image.new("RGB", (120, 80), color=(20, 120, 180)).save(image_path)
    source_image_path = tmp_path / "source.png"
    Image.new("RGB", (120, 80), color=(180, 220, 240)).save(source_image_path)
    source_jsonl_path = tmp_path / "source_gt_vs_pred.jsonl"
    source_jsonl_path.write_text(
        json.dumps(
            {
                "image": str(source_image_path),
                "width": 120,
                "height": 80,
                "gt": [
                    {"desc": "person", "type": "bbox_2d", "points": [8, 10, 28, 60]},
                    {"desc": "person", "bbox_2d": [72, 12, 98, 64]},
                ],
                "pred": [
                    {"desc": "person", "type": "bbox_2d", "points": [10, 10, 30, 60]},
                    {"desc": "person", "bbox_2d": [70, 12, 96, 64]},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (heatmap_dir / "selected_cases.jsonl").write_text(
        json.dumps(
            {
                "case_id": "case-a",
                "image_id": 1,
                "line_idx": 0,
                "slot": "y1",
                "pred_value": 10,
                "gt_value": 12,
                "wrong_anchor_advantage_at_4": 0.4,
                "selection_reason": "demo",
                "images": [str(source_image_path)],
                "source_gt_vs_pred_jsonl": str(source_jsonl_path),
                "object_index": 1,
                "source_object_index": 0,
                "top_desc": "person",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (heatmap_dir / "heatmaps.jsonl").write_text(
        json.dumps(
            {
                "case_id": "case-a",
                "model_alias": "base",
                "variant": "baseline",
                "center_kind": "pred",
                "figure_path": str(image_path),
                "grid_json": {"x_values": [0], "y_values": [0], "scores": [[0.0]]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "manual.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run:",
                f"  final_bundle_dir: {final_bundle_dir}",
                "  review_subdir: manual_review",
                "sources:",
                "  - label: demo_focus",
                f"    path: {heatmap_dir}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = build_manual_review_bundle(config_path)

    review_dir = Path(result["review_dir"])
    assert (review_dir / "manifest.json").exists()
    assert (review_dir / "review.md").exists()
    assert (review_dir / "annotation_workbook.md").exists()
    assert (review_dir / "bbox_annotations_template.jsonl").exists()
    assert (review_dir / "case_annotations_template.jsonl").exists()
    assert (review_dir / "panel_annotations_template.jsonl").exists()
    assert list((review_dir / "contact_sheets").glob("*.png"))
    assert list((review_dir / "bbox_audit").rglob("*.png"))
    review_text = (review_dir / "review.md").read_text(encoding="utf-8")
    assert "These are not raw attention maps." in review_text
