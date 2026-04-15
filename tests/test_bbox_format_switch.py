from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.common.coord_standardizer import CoordinateStandardizer
from src.common.geometry.bbox_parameterization import (
    xyxy_norm1000_to_cxcy_logw_logh_bins,
    xyxy_norm1000_to_cxcywh_bins,
)
from src.config.schema import PromptOverrides, TrainingConfig
from src.sft import _validate_bbox_format_contract
from src.trainers.losses.bbox_geo import compute_stage1_bbox_geo_loss
from src.trainers.losses.bbox_size_aux import compute_stage1_bbox_size_aux_loss


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


def _build_coord_id_map(vocab_size: int, coord_token_ids: list[int]) -> torch.Tensor:
    coord_id_map = torch.full((vocab_size,), -1, dtype=torch.long)
    for idx, tok_id in enumerate(coord_token_ids):
        coord_id_map[int(tok_id)] = int(idx)
    return coord_id_map


def _perfect_next_token_logits(labels: torch.Tensor, *, vocab: int) -> torch.Tensor:
    seq_len = max(int(labels.shape[1]) - 1, 0)
    logits = torch.full((int(labels.shape[0]), seq_len, vocab), -20.0)
    if seq_len <= 0:
        return logits
    next_labels = labels[:, 1:]
    for row in range(int(next_labels.shape[0])):
        for col in range(int(next_labels.shape[1])):
            logits[row, col, int(next_labels[row, col].item())] = 20.0
    return logits


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


def test_stage1_bbox_losses_accept_cxcy_logw_logh_serialization() -> None:
    vocab = 1200
    coord_token_ids = [100 + i for i in range(1000)]
    coord_id_map = _build_coord_id_map(vocab, coord_token_ids)
    encoded_bins = xyxy_norm1000_to_cxcy_logw_logh_bins([200, 300, 500, 700])
    labels = torch.tensor(
        [[0, 100 + encoded_bins[0], 100 + encoded_bins[1], 100 + encoded_bins[2], 100 + encoded_bins[3]]],
        dtype=torch.long,
    )
    logits = _perfect_next_token_logits(labels, vocab=vocab)

    geo = compute_stage1_bbox_geo_loss(
        logits=logits,
        labels=labels,
        coord_token_ids=coord_token_ids,
        coord_id_map=coord_id_map,
        tokenizer=None,
        cfg={"smoothl1_weight": 1.0, "ciou_weight": 1.0},
        decode_temperature=1.0,
        bbox_format="cxcy_logw_logh",
    )
    size = compute_stage1_bbox_size_aux_loss(
        logits=logits,
        labels=labels,
        coord_token_ids=coord_token_ids,
        coord_id_map=coord_id_map,
        tokenizer=None,
        cfg={"log_wh_weight": 1.0, "oversize_penalty_weight": 0.0, "eps": 1e-6},
        decode_temperature=1.0,
        bbox_format="cxcy_logw_logh",
    )

    assert geo is not None
    assert size is not None
    assert float(geo.total_loss.detach().cpu().item()) == pytest.approx(0.0, abs=1e-8)
    assert float(size.total_loss.detach().cpu().item()) == pytest.approx(0.0, abs=1e-8)


def test_validate_bbox_format_contract_rejects_stage2_cxcy_logw_logh() -> None:
    with pytest.raises(ValueError, match="custom.bbox_format=cxcy_logw_logh"):
        _validate_bbox_format_contract(
            custom_config=SimpleNamespace(bbox_format="cxcy_logw_logh"),
            trainer_variant="stage2_two_channel",
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


def test_stage1_bbox_geo_loss_accepts_cxcywh_serialization() -> None:
    vocab = 1200
    coord_token_ids = [100 + i for i in range(1000)]
    coord_id_map = _build_coord_id_map(vocab, coord_token_ids)
    encoded_bins = xyxy_norm1000_to_cxcywh_bins([200, 300, 500, 700])
    labels = torch.tensor(
        [[0, 100 + encoded_bins[0], 100 + encoded_bins[1], 100 + encoded_bins[2], 100 + encoded_bins[3]]],
        dtype=torch.long,
    )
    logits = _perfect_next_token_logits(labels, vocab=vocab)

    geo = compute_stage1_bbox_geo_loss(
        logits=logits,
        labels=labels,
        coord_token_ids=coord_token_ids,
        coord_id_map=coord_id_map,
        tokenizer=None,
        cfg={"smoothl1_weight": 1.0, "ciou_weight": 1.0},
        decode_temperature=1.0,
        bbox_format="cxcywh",
    )

    assert geo is not None
    assert float(geo.total_loss.detach().cpu().item()) == pytest.approx(0.0, abs=1e-8)


def test_validate_bbox_format_contract_rejects_stage2_cxcywh() -> None:
    with pytest.raises(ValueError, match="custom.bbox_format=cxcywh"):
        _validate_bbox_format_contract(
            custom_config=SimpleNamespace(bbox_format="cxcywh"),
            trainer_variant="stage2_two_channel",
        )
