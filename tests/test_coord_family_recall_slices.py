from __future__ import annotations

import json
from pathlib import Path

from src.analysis.coord_family_recall_slices import run_recall_slice_analysis


def test_run_recall_slice_analysis_materializes_family_buckets(tmp_path: Path) -> None:
    subset_path = tmp_path / "subset.jsonl"
    oracle_path = tmp_path / "fn_objects.jsonl"
    output_dir = tmp_path / "analysis"

    subset_row = {
        "image_id": 1,
        "file_name": "img.png",
        "width": 100,
        "height": 100,
        "objects": [
            {"desc": "cat", "category_name": "cat", "category_id": 1, "bbox_2d": ["<|coord_0|>", "<|coord_0|>", "<|coord_99|>", "<|coord_99|>"]},
            {"desc": "cat", "category_name": "cat", "category_id": 1, "bbox_2d": ["<|coord_10|>", "<|coord_10|>", "<|coord_19|>", "<|coord_19|>"]},
            {"desc": "dog", "category_name": "dog", "category_id": 2, "bbox_2d": ["<|coord_50|>", "<|coord_50|>", "<|coord_59|>", "<|coord_59|>"]},
        ],
    }
    subset_path.write_text(json.dumps(subset_row) + "\n", encoding="utf-8")
    oracle_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "record_idx": 0,
                        "gt_idx": 1,
                        "baseline_fn_loc": True,
                        "ever_recovered_loc": True,
                        "systematic_loc": False,
                        "recover_count_loc": 2,
                        "recover_fraction_loc": 0.5,
                    }
                ),
                json.dumps(
                    {
                        "record_idx": 0,
                        "gt_idx": 2,
                        "baseline_fn_loc": True,
                        "ever_recovered_loc": False,
                        "systematic_loc": True,
                        "recover_count_loc": 0,
                        "recover_fraction_loc": 0.0,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "slices.yaml"
    config_path.write_text(
        f"""
run:
  name: coord-family-recall-slices-smoke
  output_dir: {output_dir.as_posix()}

families:
  - family_alias: raw_text_xyxy_pure_ce
    oracle_fn_objects_jsonl: {oracle_path.as_posix()}
    subset_jsonl: {subset_path.as_posix()}
        """.strip(),
        encoding="utf-8",
    )

    result = run_recall_slice_analysis(config_path, repo_root=tmp_path)

    summary_path = output_dir / "coord-family-recall-slices-smoke" / "summary.json"
    rows_path = output_dir / "coord-family-recall-slices-smoke" / "slice_rows.jsonl"
    assert result["summary_json"] == str(summary_path)
    assert rows_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    family = summary["families"]["raw_text_xyxy_pure_ce"]
    assert family["overall"]["baseline_fn_count_loc"] == 2
    assert family["overall"]["recoverable_fn_count_loc"] == 1
    assert family["by_same_desc_bucket"][0]["same_desc_bucket"] == "pair"
    assert family["top_categories"][0]["category_name"] == "cat"

