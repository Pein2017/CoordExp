from __future__ import annotations

import json
from pathlib import Path

from src.analysis.raw_text_coordinate_decode_bias_study import (
    hydrate_case_rows,
    load_study_config,
    run_study,
)

_BASE_MODEL_PATH = (
    "/data/CoordExp/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp"
)
_ADAPTER_PATH = (
    "/data/CoordExp/output/stage1_2b/"
    "coco_bbox_max60-coco80-desc_first-1024-lvis_proxy-raw_text_xyxy-pure_ce/"
    "epoch_4-raw_text_xyxy-pure_ce-coco80-desc_first-1024-lvis_proxy-from-base-2B/"
    "v1-20260417-084341/checkpoint-552"
)


def _write_source_jsonl(path: Path) -> None:
    rows = [
        {
            "image_id": 101,
            "width": 100,
            "height": 100,
            "objects": [
                {"desc": "cup", "bbox_2d": [10, 20, 30, 40]},
                {"desc": "plate", "bbox_2d": [50, 60, 70, 80]},
            ],
        },
        {
            "image_id": 202,
            "width": 200,
            "height": 100,
            "objects": [
                {"desc": "book", "bbox_2d": [20, 10, 80, 50]},
                {"desc": "lamp", "bbox_2d": [100, 5, 150, 75]},
            ],
        },
    ]
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_config(*, config_path: Path, output_dir: Path, source_jsonl: Path) -> None:
    config_path.write_text(
        f"""
run:
  name: raw-text-decode-bias-smoke
  output_dir: {output_dir.as_posix()}
  stages: [hydrate, counterfactual_eos, counterfactual_repeat_penalty]

study:
  history_scope: full_model_history
  val200_source_jsonl: {source_jsonl.as_posix()}
  val200_source_indices: [0, 1]

models:
  base_only:
    alias: base_only
    base_path: {_BASE_MODEL_PATH}
    adapter_path: null
    prompt_variant: coco_80
    object_field_order: desc_first
    coord_mode: norm1000_text
  base_plus_adapter:
    alias: base_plus_adapter
    base_path: {_BASE_MODEL_PATH}
    adapter_path: {_ADAPTER_PATH}
    prompt_variant: coco_80
    object_field_order: desc_first
    coord_mode: norm1000_text
        """.strip(),
        encoding="utf-8",
    )


def test_load_study_config_parses_raw_text_only_model_block_and_val200_indices(
    tmp_path: Path,
) -> None:
    source_jsonl = tmp_path / "val200.jsonl"
    _write_source_jsonl(source_jsonl)
    config_path = tmp_path / "study.yaml"
    _write_config(
        config_path=config_path,
        output_dir=tmp_path,
        source_jsonl=source_jsonl,
    )

    cfg = load_study_config(config_path)

    assert cfg.study.history_scope == "full_model_history"
    assert tuple(cfg.study.val200_source_indices) == (0, 1)
    assert cfg.models.base_only.adapter_path is None
    assert cfg.models.base_only.coord_mode == "norm1000_text"
    assert cfg.models.base_plus_adapter.adapter_path == _ADAPTER_PATH
    assert cfg.models.base_plus_adapter.coord_mode == "norm1000_text"


def test_hydrate_case_rows_writes_frozen_candidate_texts() -> None:
    hydrated = hydrate_case_rows(
        case_rows=[
            {
                "case_uid": "base_only:val200:0",
                "model_alias": "base_only",
                "source_jsonl": "/tmp/source.jsonl",
                "source_index": 0,
                "image_id": 101,
                "object_field_order": "desc_first",
                "source_row": {
                    "image_id": 101,
                    "width": 100,
                    "height": 100,
                    "objects": [
                        {"desc": "cup", "bbox_2d": [10, 20, 30, 40]},
                        {"desc": "plate", "bbox_2d": [50, 60, 70, 80]},
                    ],
                },
            }
        ]
    )

    assert hydrated[0]["case_uid"] == "base_only:val200:0"
    assert hydrated[0]["hydration_version"] == "raw_text_decode_bias_v1"
    assert hydrated[0]["baseline_assistant_text"].startswith('{"objects": [')
    assert hydrated[0]["stop_now_candidate_text"].startswith(
        hydrated[0]["baseline_assistant_text"]
    )
    assert hydrated[0]["stop_now_candidate_text"].endswith("]}")
    assert hydrated[0]["continue_with_gt_candidate_text"].startswith(
        hydrated[0]["baseline_assistant_text"]
    )
    assert '"desc": "plate"' in hydrated[0]["continue_with_gt_candidate_text"]


def test_run_study_materializes_run_dir_and_hydrated_inputs(tmp_path: Path) -> None:
    source_jsonl = tmp_path / "val200.jsonl"
    _write_source_jsonl(source_jsonl)
    config_path = tmp_path / "study.yaml"
    _write_config(
        config_path=config_path,
        output_dir=tmp_path,
        source_jsonl=source_jsonl,
    )

    result = run_study(config_path)
    run_dir = tmp_path / "raw-text-decode-bias-smoke"

    assert result["run_dir"] == str(run_dir)
    assert (run_dir / "stage_manifest.json").exists()
    assert (run_dir / "counterfactual_inputs" / "hydrated_cases.jsonl").exists()

