from __future__ import annotations

import json
from pathlib import Path

from src.analysis.post_x1_instance_basin_tomography.gallery import materialize_gallery, select_gallery_rows
from src.analysis.post_x1_instance_basin_tomography.jsonl import write_jsonl


def test_select_gallery_rows_prioritizes_competitor_and_reference_anchor() -> None:
    rows = [
        {
            "case_id": "case-target",
            "checkpoint_role": "fullobj_random_pure_ce_ckpt3668",
            "winner_bucket": "target_instance",
            "comparison_role": "clean_pair",
            "template_contract": {"template_contract_id": "compact_full_no_newline_native_v1"},
        },
        {
            "case_id": "case-competitor",
            "checkpoint_role": "et_rmp_ce_ckpt3664",
            "winner_bucket": "same_desc_competitor",
            "comparison_role": "reference_anchor",
            "template_contract": {"template_contract_id": "compact_full_newline_native_v1"},
        },
    ]

    selected = select_gallery_rows(rows, max_rows=1)

    assert selected[0]["case_id"] == "case-competitor"
    assert selected[0]["reference_anchor_caveat"] == "template_objective_confounded_reference"


def test_materialize_gallery_writes_visible_contract_and_caveat(tmp_path: Path) -> None:
    write_jsonl(
        tmp_path / "slot_posterior_rows.jsonl",
        [
            {
                "case_id": "case-competitor",
                "image_id": "img-1",
                "checkpoint_role": "et_rmp_ce_ckpt3664",
                "slot": "y1",
                "winner_bucket": "same_desc_competitor",
                "prefix_mode": "post_x1",
                "comparison_role": "reference_anchor",
                "target_gt_box": [1, 2, 3, 4],
                "same_desc_competitor_gt_boxes": [[5, 6, 7, 8]],
                "template_contract": {
                    "template_contract_id": "compact_full_newline_native_v1",
                    "row_separator": "newline",
                },
            }
        ],
    )

    summary = materialize_gallery(tmp_path, max_rows=8)

    index = (tmp_path / "gallery" / "index.md").read_text(encoding="utf-8")
    assert summary["gallery_rows"] == 1
    assert "compact_full_newline_native_v1" in index
    assert "template_objective_confounded_reference" in index
    assert "same_desc_competitor" in index
    payload = json.loads((tmp_path / "gallery" / "gallery_summary.json").read_text(encoding="utf-8"))
    assert payload["artifact_schema_version"] == "a3.3.v1"
