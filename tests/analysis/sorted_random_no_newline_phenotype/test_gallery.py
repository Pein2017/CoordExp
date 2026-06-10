from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.analysis.sorted_random_no_newline_phenotype.gallery import (
    GALLERY_LABELS,
    build_galleries,
)


def test_gallery_builds_indices_jpgs_and_non_overlapping_right_panel_legend(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "blank.jpg"
    Image.new("RGB", (48, 32), color=(245, 245, 245)).save(image_path)
    artifact_root = tmp_path / "artifacts"

    metadata = build_galleries(
        artifact_root,
        native_cases=[_case("native-1", image_path=image_path)],
        fn_cases=[_case("fn-1", image_path=image_path)],
    )

    assert (artifact_root / "gallery" / "index.md").is_file()
    assert (artifact_root / "fn_probe" / "gallery" / "index.md").is_file()
    assert (artifact_root / "gallery" / "images" / "native-1.jpg").is_file()
    assert (
        artifact_root / "fn_probe" / "gallery" / "images" / "fn-1.jpg"
    ).is_file()

    index_text = (artifact_root / "gallery" / "index.md").read_text(
        encoding="utf-8"
    )
    fn_index_text = (artifact_root / "fn_probe" / "gallery" / "index.md").read_text(
        encoding="utf-8"
    )
    for label in GALLERY_LABELS:
        assert label in index_text
        assert label in fn_index_text

    for item in metadata["gallery"] + metadata["fn_probe_gallery"]:
        assert item["legend_placement"] == "right_panel"
        assert item["legend_overlaps_image"] is False
        assert _boxes_do_not_overlap(item["image_panel_bbox"], item["legend_bbox"])


def test_gallery_uses_labeled_placeholder_when_image_source_is_missing(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"

    metadata = build_galleries(
        artifact_root,
        native_cases=[_case("missing-native", image_path=tmp_path / "missing.jpg")],
        fn_cases=[],
    )

    assert (artifact_root / "gallery" / "images" / "missing-native.jpg").is_file()
    index_text = (artifact_root / "gallery" / "index.md").read_text(
        encoding="utf-8"
    )
    assert "Missing image placeholder" in index_text
    assert metadata["gallery"][0]["image_source_status"] == "missing_placeholder"


def _case(case_id: str, *, image_path: Path) -> dict[str, object]:
    return {
        "case_id": case_id,
        "image_id": case_id,
        "image_path": str(image_path),
        "width": 48,
        "height": 32,
        "gt_objects": [{"desc": "chair", "bbox": [4, 4, 20, 20]}],
        "random_predictions": [{"desc": "chair", "bbox": [2, 2, 16, 18]}],
        "sorted_predictions": [{"desc": "chair", "bbox": [5, 5, 21, 21]}],
        "fn_target": {"desc": "chair", "bbox": [24, 8, 36, 24]},
        "emitted_same_desc": [{"desc": "chair", "bbox": [2, 2, 16, 18]}],
        "residual_same_desc": [{"desc": "chair", "bbox": [24, 8, 36, 24]}],
        "x1_peaks": [{"x1": 24, "score": 0.91}],
    }


def _boxes_do_not_overlap(
    left: tuple[int, int, int, int] | list[int],
    right: tuple[int, int, int, int] | list[int],
) -> bool:
    left_x1, left_y1, left_x2, left_y2 = left
    right_x1, right_y1, right_x2, right_y2 = right
    return (
        left_x2 <= right_x1
        or right_x2 <= left_x1
        or left_y2 <= right_y1
        or right_y2 <= left_y1
    )
