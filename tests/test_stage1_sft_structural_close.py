from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from src.config.schema import PromptOverrides, TrainingConfig
from src.data_collators.batch_extras_collator import build_dataset_metrics_collator


class _FakeTemplate:
    """Character-level template sufficient for structural-close span tests."""

    def __init__(self) -> None:
        self.system = "system"
        self.data_collator = _base_collator

    def encode(
        self, rendered: Mapping[str, Any], return_length: bool = True
    ) -> dict[str, Any]:
        assistant = next(
            message
            for message in rendered["messages"]
            if message.get("role") == "assistant"
        )
        text = _message_text(assistant) or ""
        ids = [ord(ch) for ch in text]
        return {
            "input_ids": ids,
            "labels": list(ids),
            "attention_mask": [1] * len(ids),
            "length": len(ids),
        }


def _message_text(message: Mapping[str, Any]) -> str | None:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        texts = [
            item.get("text")
            for item in content
            if isinstance(item, Mapping) and item.get("type") == "text"
        ]
        return "\n".join(str(item) for item in texts if item is not None)
    return None


def _base_collator(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    rows = [row["encoded"] for row in batch]
    max_len = max(len(row["input_ids"]) for row in rows)
    input_ids = []
    labels = []
    attention_mask = []
    for row in rows:
        pad = max_len - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [0] * pad)
        labels.append(row["labels"] + [-100] * pad)
        attention_mask.append(row["attention_mask"] + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def _raw_row(template: _FakeTemplate, assistant_text: str) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": [{"type": "text", "text": "detect"}]},
        {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
    ]
    return {
        "messages": messages,
        "assistant_payload": {"objects": []},
        "metadata": {"dataset": "unit"},
        "encoded": template.encode({"messages": messages}),
    }


def _base_payload() -> dict[str, Any]:
    return {
        "template": {"truncation_strategy": "raise"},
        "training": {"packing": False, "eval_packing": False},
        "custom": {
            "train_jsonl": "train.jsonl",
            "user_prompt": "prompt",
            "emit_norm": "none",
            "json_format": "standard",
            "object_field_order": "desc_first",
            "coord_tokens": {"enabled": True, "skip_bbox_norm": True},
        },
    }


def test_sft_structural_close_config_parses_and_rejects_packing() -> None:
    payload = _base_payload()
    payload["custom"]["sft_structural_close"] = {
        "enabled": True,
        "final_close_weight": 0.0,
    }

    cfg = TrainingConfig.from_mapping(payload, PromptOverrides())

    assert cfg.custom.sft_structural_close.enabled is True
    assert cfg.custom.sft_structural_close.final_close_weight == pytest.approx(0.0)

    packed = _base_payload()
    packed["training"] = {"packing": True}
    packed["custom"]["sft_structural_close"] = {
        "enabled": True,
        "final_close_weight": 0.0,
    }
    with pytest.raises(ValueError, match="sft_structural_close.*training.packing=false"):
        TrainingConfig.from_mapping(packed, PromptOverrides())


def test_sft_structural_close_collator_weights_only_global_final_close() -> None:
    from src.config.schema import Stage1SFTStructuralCloseConfig

    template = _FakeTemplate()
    assistant_text = '{"objects": [{"desc": "cat", "bbox_2d": [<|coord_1|>, <|coord_2|>, <|coord_3|>, <|coord_4|>]}]}'
    collator = build_dataset_metrics_collator(
        template,
        _base_collator,
        sft_structural_close_cfg=Stage1SFTStructuralCloseConfig(
            enabled=True,
            final_close_weight=0.0,
        ),
    )

    batch = collator([_raw_row(template, assistant_text)])

    weights = batch["sft_structural_close_token_weights"][0]
    labels = batch["labels"][0]
    final_positions = [
        int(index)
        for index, value in enumerate(labels.tolist())
        if value in (ord("]"), ord("}"))
    ]
    assert final_positions[-2:] == [len(assistant_text) - 2, len(assistant_text) - 1]
    assert weights[final_positions[-2]].item() == pytest.approx(0.0)
    assert weights[final_positions[-1]].item() == pytest.approx(0.0)
    assert any(weights[index].item() == pytest.approx(1.0) for index in final_positions[:-2])


def test_sft_structural_close_loss_mixin_applies_fractional_token_weights() -> None:
    from src.trainers.metrics.mixins import SFTStructuralCloseLossMixin

    class _FakeModel:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def __call__(self, **kwargs):
            self.calls.append(dict(kwargs))
            input_ids = kwargs["input_ids"]
            logits = torch.zeros(
                (input_ids.shape[0], input_ids.shape[1], 32),
                dtype=torch.float32,
            )
            logits[:, :, 1] = 4.0
            return SimpleNamespace(logits=logits)

    class _Trainer(SFTStructuralCloseLossMixin):
        sft_structural_close_cfg = SimpleNamespace(
            enabled=True,
            final_close_weight=0.25,
        )

    trainer = _Trainer()
    model = _FakeModel()
    inputs = {
        "input_ids": torch.tensor([[0, 1, 1, 1]], dtype=torch.long),
        "labels": torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
        "sft_structural_close_token_weights": torch.tensor(
            [[1.0, 1.0, 0.25, 0.25]], dtype=torch.float32
        ),
    }

    loss, outputs = trainer.compute_loss(model, inputs, return_outputs=True)

    assert outputs is not None
    assert "sft_structural_close_token_weights" not in model.calls[0]
    logits = model.calls[0]["input_ids"].new_zeros((1, 3, 32), dtype=torch.float32)
    logits[:, :, 1] = 4.0
    labels_next = torch.tensor([[1, 2, 3]], dtype=torch.long)
    weights_next = torch.tensor([[1.0, 0.25, 0.25]], dtype=torch.float32)
    ce = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 32),
        labels_next.reshape(-1),
        reduction="none",
    ).view_as(labels_next)
    expected = (ce * weights_next).sum() / weights_next.sum()
    assert loss.item() == pytest.approx(float(expected.item()))
