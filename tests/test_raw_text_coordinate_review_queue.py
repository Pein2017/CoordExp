from pathlib import Path

from PIL import Image

from src.analysis.raw_text_coordinate_review_queue import (
    build_review_queue_rows,
    materialize_review_gallery,
    write_review_gallery_html,
)


def test_build_review_queue_rows_exposes_notion_friendly_columns() -> None:
    rows = build_review_queue_rows(
        shortlist=[
            {
                "case_uid": "base_only:1:0:0:first_burst_onset:src2:on3",
                "review_bucket": "FP",
                "model_alias": "base_only",
                "selection_rank": 1,
                "source_gt_vs_pred_jsonl": "/tmp/source.jsonl",
                "line_idx": 7,
                "object_index": 3,
                "source_object_index": 2,
                "gt_idx": None,
            }
        ]
    )

    assert rows[0]["case_uid"] == "base_only:1:0:0:first_burst_onset:src2:on3"
    assert rows[0]["bucket"] == "FP"
    assert rows[0]["status"] == "unreviewed"
    assert rows[0]["bbox_judgment"] == ""
    assert rows[0]["source_gt_vs_pred_jsonl"] == "/tmp/source.jsonl"
    assert rows[0]["object_index"] == 3


def test_write_review_gallery_html_renders_case_cards(tmp_path: Path) -> None:
    output_path = tmp_path / "review.html"
    write_review_gallery_html(
        rows=[
            {
                "case_uid": "base_only:1:0:0:first_burst_onset:src2:on3",
                "review_bucket": "FP",
                "model_alias": "base_only",
                "image_id": 1,
                "line_idx": 0,
                "detail": "duplicate burst | desc=person | source=2 -> onset=3",
                "review_image": "cases/case-1/rendered/vis_0000.png",
            },
            {
                "case_uid": "base_only:2:1:1:labeled_fn:gt4",
                "review_bucket": "FN",
                "model_alias": "base_only",
                "image_id": 2,
                "line_idx": 1,
                "detail": "labeled FN | desc=clock | gt_idx=4",
                "review_image": "cases/case-2/rendered/vis_0000.png",
            },
        ],
        output_path=output_path,
        title="Demo Review",
    )

    html_text = output_path.read_text(encoding="utf-8")

    assert "Demo Review" in html_text
    assert "cases/case-1/rendered/vis_0000.png" in html_text
    assert "duplicate burst | desc=person | source=2 -&gt; onset=3" in html_text
    assert "FN Cases (1)" in html_text


def test_materialize_review_gallery_renders_local_assets(tmp_path: Path) -> None:
    image_path = tmp_path / "demo.png"
    Image.new("RGB", (16, 16), color=(255, 255, 255)).save(image_path)
    source_jsonl = tmp_path / "source.jsonl"
    source_jsonl.write_text(
        (
            '{"record_idx": 0, "image": "%s", "width": 16, "height": 16, '
            '"coord_mode": "pixel", "gt": [{"type": "bbox_2d", "points": [1, 1, 8, 8], "desc": "person"}], '
            '"pred": [{"type": "bbox_2d", "points": [1, 1, 8, 8], "desc": "person"}]}\n'
        )
        % image_path.as_posix(),
        encoding="utf-8",
    )

    rows = materialize_review_gallery(
        shortlist=[
            {
                "case_uid": "base_only:1:0:0:first_burst_onset:src0:on0",
                "review_bucket": "FP",
                "model_alias": "base_only",
                "image_id": 1,
                "line_idx": 0,
                "source_gt_vs_pred_jsonl": source_jsonl.as_posix(),
                "object_index": 0,
                "source_object_index": 0,
                "gt_idx": None,
                "selection_rank": 1,
                "serializer_surface": "model_native",
            }
        ],
        output_dir=tmp_path / "gallery",
        title="Demo Gallery",
    )

    assert len(rows) == 1
    assert (tmp_path / "gallery" / "review.html").exists()
    assert (tmp_path / "gallery" / "manifest.json").exists()
    assert (
        tmp_path / "gallery" / rows[0]["review_image"]
    ).exists()
