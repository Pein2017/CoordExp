from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from src.config.loader import ConfigLoader
from src.config.schema import PromptOverrides, TrainingConfig
from src.datasets.dense_caption import BaseCaptionDataset
from src.common.coord_standardizer import CoordinateStandardizer
from src.infer.engine import GenerationConfig, InferenceConfig, InferenceEngine
from src.sft import (
    PackingRuntimeConfig,
    _build_static_packing_fingerprint,
    _validate_bbox_format_contract,
)
from src.trainers.losses.bbox_geo import compute_stage1_bbox_geo_loss
from src.trainers.losses.bbox_size_aux import compute_stage1_bbox_size_aux_loss


class _Template:
    def __init__(self) -> None:
        self.system = "system"
        self.max_length = 256
        self.max_pixels = 16384


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


def test_custom_bbox_format_accepts_cxcywh_and_rejects_unknown() -> None:
    payload = _base_training_payload()
    payload["custom"]["bbox_format"] = "cxcywh"

    cfg = TrainingConfig.from_mapping(payload, PromptOverrides())
    assert cfg.custom.bbox_format == "cxcywh"

    bad_payload = _base_training_payload()
    bad_payload["custom"]["bbox_format"] = "corners_plus"
    with pytest.raises(ValueError, match="custom.bbox_format"):
        TrainingConfig.from_mapping(bad_payload, PromptOverrides())


def test_prompt_resolution_and_inference_parity_reflect_bbox_format() -> None:
    train_prompts = ConfigLoader.resolve_prompts(
        {
            "custom": {
                "object_ordering": "sorted",
                "object_field_order": "desc_first",
                "bbox_format": "cxcywh",
                "coord_tokens": {"enabled": True},
            }
        }
    )

    assert "[cx, cy, w, h]" in train_prompts.system
    assert "[cx, cy, w, h]" in train_prompts.user
    assert "implied top-left" in train_prompts.user

    engine = InferenceEngine(
        InferenceConfig(
            gt_jsonl="dummy.jsonl",
            model_checkpoint="dummy",
            mode="text",
            bbox_format="cxcywh",
            object_ordering="sorted",
        ),
        GenerationConfig(),
    )
    messages = engine._build_messages(Image.new("RGB", (16, 16), color=(0, 0, 0)))

    assert messages[0]["content"][0]["text"] == train_prompts.system
    assert messages[1]["content"][0]["text"] == train_prompts.user


def test_dataset_runtime_bbox_conversion_is_model_facing_only() -> None:
    dataset = BaseCaptionDataset(
        base_records=[
            {
                "images": ["dummy.jpg"],
                "width": 100,
                "height": 80,
                "objects": [
                    {"desc": "cat", "bbox_2d": [10, 20, 30, 60]},
                ],
            }
        ],
        template=_Template(),
        user_prompt="Locate objects",
        emit_norm="none",
        json_format="standard",
        bbox_format="cxcywh",
    )

    prepared = dataset._prepare_record_for_cache(base_idx=0)
    assert prepared["objects"][0]["bbox_2d"] == [20, 40, 20, 40]
    assert prepared["objects"][0]["_bbox_xyxy_original"] == [10, 20, 30, 60]

    builder = dataset._create_builder("dense")
    rendered, _messages = dataset._render_prepared_record(
        record=prepared,
        builder=builder,
        system_prompt=None,
    )
    assert rendered["assistant_payload"]["objects"][0]["bbox_2d"] == [
        "<|coord_20|>",
        "<|coord_40|>",
        "<|coord_20|>",
        "<|coord_40|>",
    ]
    assert rendered["objects"]["bbox"][0] == [10, 20, 30, 60]


def test_coord_standardizer_converts_cxcywh_predictions_to_xyxy() -> None:
    standardizer = CoordinateStandardizer("text", pred_bbox_format="cxcywh")
    errors: list[str] = []
    raw_text = '{"obj":{"bbox_2d":[50,25,20,10],"desc":"car"}}'
    preds = standardizer.process_prediction_text(
        raw_text,
        width=100,
        height=50,
        errors=errors,
    )

    assert errors == []
    assert preds[0]["points"] == [40, 20, 60, 30]
    assert preds[0]["points_text"] == "40 20 60 30"


def test_stage1_bbox_losses_accept_cxcywh_serialization() -> None:
    vocab = 1200
    coord_token_ids = [100 + i for i in range(1000)]
    coord_id_map = _build_coord_id_map(vocab, coord_token_ids)
    cxcywh_bins = [200, 300, 300, 400]
    labels = torch.tensor(
        [[0, 100 + cxcywh_bins[0], 100 + cxcywh_bins[1], 100 + cxcywh_bins[2], 100 + cxcywh_bins[3]]],
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
    size = compute_stage1_bbox_size_aux_loss(
        logits=logits,
        labels=labels,
        coord_token_ids=coord_token_ids,
        coord_id_map=coord_id_map,
        tokenizer=None,
        cfg={"log_wh_weight": 1.0, "oversize_penalty_weight": 0.0, "eps": 1e-6},
        decode_temperature=1.0,
        bbox_format="cxcywh",
    )

    assert geo is not None
    assert size is not None
    assert float(geo.total_loss.detach().cpu().item()) == pytest.approx(0.0, abs=1e-8)
    assert float(size.total_loss.detach().cpu().item()) == pytest.approx(0.0, abs=1e-8)


def test_static_packing_fingerprint_tracks_bbox_format() -> None:
    template = _Template()
    train_args = SimpleNamespace(max_model_len=512)
    packing_cfg = PackingRuntimeConfig(enabled=True, mode="static", packing_length=256)

    xyxy_cfg = SimpleNamespace(
        user_prompt="prompt",
        emit_norm="none",
        json_format="standard",
        object_ordering="sorted",
        object_field_order="desc_first",
        bbox_format="xyxy",
        use_summary=False,
        offline_max_pixels=None,
        coord_tokens=None,
        fusion_config=None,
        system_prompt_dense=None,
        system_prompt_summary=None,
    )
    cxcywh_cfg = SimpleNamespace(**{**xyxy_cfg.__dict__, "bbox_format": "cxcywh"})

    fp_xyxy = _build_static_packing_fingerprint(
        training_config=SimpleNamespace(template={}, training={}),
        custom_config=xyxy_cfg,
        template=template,
        train_args=train_args,
        dataset_seed=7,
        packing_cfg=packing_cfg,
        train_jsonl="train.jsonl",
    )
    fp_cxcywh = _build_static_packing_fingerprint(
        training_config=SimpleNamespace(template={}, training={}),
        custom_config=cxcywh_cfg,
        template=template,
        train_args=train_args,
        dataset_seed=7,
        packing_cfg=packing_cfg,
        train_jsonl="train.jsonl",
    )

    assert fp_xyxy["custom_bbox_format"] == "xyxy"
    assert fp_cxcywh["custom_bbox_format"] == "cxcywh"
    assert fp_xyxy != fp_cxcywh


def test_stage2_variants_fail_fast_for_non_xyxy_bbox_format() -> None:
    with pytest.raises(ValueError, match="custom.bbox_format=cxcywh"):
        _validate_bbox_format_contract(
            custom_config=SimpleNamespace(bbox_format="cxcywh"),
            trainer_variant="stage2_two_channel",
        )
