from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from public_data.view_contracts import write_view_metadata

from src.common.coord_standardizer import CoordinateStandardizer
from src.common.geometry.bbox_parameterization import (
    xyxy_norm1000_to_cxcy_logw_logh_bins,
    xyxy_norm1000_to_cxcywh_bins,
)
from src.config.schema import PromptOverrides, TrainingConfig
from src.sft import _validate_bbox_format_contract


def _base_training_payload() -> dict:
    return {
        "template": {"truncation_strategy": "raise"},
        "custom": {
            "train_jsonl": "train.jsonl",
            "user_prompt": "prompt",
            "emit_norm": "none",
            "json_format": "standard",
            "object_field_order": "desc_first",
        },
    }


def _write_qwen_coord_token_view_meta(view_root: Path) -> None:
    write_view_metadata(
        view_root / "meta.json",
        {
            "schema_version": 1,
            "kind": "annotation_view",
            "dataset": "coco",
            "view": "coco80/len-12000",
            "image_store": "public_data/coco/images/res-1024",
            "path_anchor": "repo_root",
            "image_path_semantics": "image_store_relative",
            "coordinate_space": "norm1000",
            "coordinate_storage": "integer",
            "coordinate_range": [0, 999],
            "coordinate_chart": "xyxy",
            "assistant_coordinate_rendering": "qwen_coord_tokens",
            "primary_jsonl": {"train": "train.jsonl", "val": "val.jsonl"},
            "sample_policy": {
                "type": "length_budget",
                "max_total_tokens": 12000,
            },
            "length_budget_scope": {"rendered_families": ["assistant"]},
            "length_budget_template_id": "compact_full",
            "summary": {},
        },
    )


def test_custom_bbox_format_accepts_cxcy_logw_logh_and_rejects_unknown() -> None:
    payload = _base_training_payload()
    payload["custom"]["bbox_format"] = "cxcy_logw_logh"
    payload["custom"]["coord_tokens"] = {"enabled": True, "skip_bbox_norm": True}
    payload["custom"]["coord_soft_ce_w1"] = {
        "enabled": True,
        "ce_weight": 1.0,
        "soft_ce_weight": 0.0,
        "w1_weight": 0.0,
        "gate_weight": 1.0,
        "text_gate_weight": 1.0,
    }

    cfg = TrainingConfig.from_mapping(payload, PromptOverrides())
    assert cfg.custom.bbox_format == "cxcy_logw_logh"

    bad_payload = _base_training_payload()
    bad_payload["custom"]["bbox_format"] = "corners_plus"
    with pytest.raises(ValueError, match="custom.bbox_format"):
        TrainingConfig.from_mapping(bad_payload, PromptOverrides())


def test_coord_standardizer_converts_cxcy_logw_logh_predictions_to_xyxy() -> None:
    serialized = xyxy_norm1000_to_cxcy_logw_logh_bins([100, 200, 400, 700])
    standardizer = CoordinateStandardizer(
        "text",
        pred_coord_mode="norm1000",
        bbox_format="cxcy_logw_logh",
    )
    errors: list[str] = []
    raw_text = (
        '{"objects":[{"bbox_2d":['
        f"{serialized[0]},{serialized[1]},{serialized[2]},{serialized[3]}"
        '],"desc":"car"}]}'
    )

    preds = standardizer.process_prediction_text(
        raw_text,
        width=999,
        height=999,
        errors=errors,
    )

    assert errors == []
    assert preds[0]["points"] == [100, 200, 400, 699]
    assert preds[0]["points_text"] == "100 200 400 699"


def test_validate_bbox_format_contract_rejects_stage2_cxcy_logw_logh() -> None:
    with pytest.raises(ValueError, match="custom.bbox_format=cxcy_logw_logh"):
        _validate_bbox_format_contract(
            custom_config=SimpleNamespace(bbox_format="cxcy_logw_logh"),
            trainer_variant="stage2_rollout_correction",
        )


def test_validate_bbox_format_contract_rejects_raw_text_on_coord_surface() -> None:
    with pytest.raises(
        ValueError,
        match=r"custom\.coord_tokens\.enabled=false; use a \*\.norm\.jsonl surface",
    ):
        _validate_bbox_format_contract(
            custom_config=SimpleNamespace(
                bbox_format="xyxy",
                coord_tokens=SimpleNamespace(enabled=False),
                train_jsonl="public_data/coco/demo/train.coord.jsonl",
            ),
            trainer_variant="",
        )


def test_validate_bbox_format_contract_rejects_coord_mode_on_norm_surface() -> None:
    with pytest.raises(
        ValueError,
        match=r"custom\.coord_tokens\.enabled=true; use a \*\.coord\.jsonl surface",
    ):
        _validate_bbox_format_contract(
            custom_config=SimpleNamespace(
                bbox_format="xyxy",
                coord_tokens=SimpleNamespace(enabled=True),
                train_jsonl="public_data/coco/demo/train.norm.jsonl",
            ),
            trainer_variant="",
        )


def test_validate_bbox_format_contract_accepts_canonical_view_coord_rendering(
    tmp_path: Path,
) -> None:
    view_root = tmp_path / "public_data/coco/views/coco80/len-12000"
    view_root.mkdir(parents=True)
    _write_qwen_coord_token_view_meta(view_root)

    _validate_bbox_format_contract(
        custom_config=SimpleNamespace(
            bbox_format="xyxy",
            coord_tokens=SimpleNamespace(enabled=True),
            train_jsonl=str(view_root / "train.jsonl"),
            val_jsonl=str(view_root / "val.jsonl"),
        ),
        trainer_variant="",
    )


def test_validate_bbox_format_contract_rejects_raw_text_mode_on_canonical_view(
    tmp_path: Path,
) -> None:
    view_root = tmp_path / "public_data/coco/views/coco80/len-12000"
    view_root.mkdir(parents=True)
    _write_qwen_coord_token_view_meta(view_root)

    with pytest.raises(
        ValueError,
        match="assistant_coordinate_rendering=qwen_coord_tokens",
    ):
        _validate_bbox_format_contract(
            custom_config=SimpleNamespace(
                bbox_format="xyxy",
                coord_tokens=SimpleNamespace(enabled=False),
                train_jsonl=str(view_root / "train.jsonl"),
            ),
            trainer_variant="",
        )


def test_custom_bbox_format_accepts_cxcywh() -> None:
    payload = _base_training_payload()
    payload["custom"]["bbox_format"] = "cxcywh"
    payload["custom"]["coord_tokens"] = {"enabled": True, "skip_bbox_norm": True}
    payload["custom"]["coord_soft_ce_w1"] = {
        "enabled": True,
        "ce_weight": 1.0,
        "soft_ce_weight": 0.0,
        "w1_weight": 0.0,
        "gate_weight": 1.0,
        "text_gate_weight": 1.0,
    }

    cfg = TrainingConfig.from_mapping(payload, PromptOverrides())
    assert cfg.custom.bbox_format == "cxcywh"


def test_coord_standardizer_converts_cxcywh_predictions_to_xyxy() -> None:
    serialized = xyxy_norm1000_to_cxcywh_bins([100, 200, 400, 700])
    standardizer = CoordinateStandardizer(
        "text",
        pred_coord_mode="norm1000",
        bbox_format="cxcywh",
    )
    errors: list[str] = []
    raw_text = (
        '{"objects":[{"bbox_2d":['
        f"{serialized[0]},{serialized[1]},{serialized[2]},{serialized[3]}"
        '],"desc":"car"}]}'
    )

    preds = standardizer.process_prediction_text(
        raw_text,
        width=999,
        height=999,
        errors=errors,
    )

    assert errors == []
    assert preds[0]["points"] == [100, 200, 400, 699]
    assert preds[0]["points_text"] == "100 200 400 699"


def test_validate_bbox_format_contract_rejects_stage2_cxcywh() -> None:
    with pytest.raises(ValueError, match="custom.bbox_format=cxcywh"):
        _validate_bbox_format_contract(
            custom_config=SimpleNamespace(bbox_format="cxcywh"),
            trainer_variant="stage2_rollout_correction",
        )
