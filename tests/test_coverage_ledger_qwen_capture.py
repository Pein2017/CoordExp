from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from src.training.coverage_ledger.qwen_capture import CoverageLedgerForwardCapture


@dataclass
class _FakeBackboneOutput:
    last_hidden_state: torch.Tensor
    past_key_values: object | None = None
    rope_deltas: object | None = None

    def __getitem__(self, index: int) -> torch.Tensor:
        if index != 0:
            raise IndexError(index)
        return self.last_hidden_state


@dataclass
class _FakeConditionalOutput:
    logits: torch.Tensor
    hidden_states: torch.Tensor


class _FakeLowerQwen:
    def __init__(self, *, hidden_size: int, image_token_id: int) -> None:
        self.config = SimpleNamespace(image_token_id=image_token_id)
        self.visual = SimpleNamespace(spatial_merge_size=1)
        self.hidden_size = hidden_size
        self.get_image_features_calls = 0
        self.forward_calls: list[dict[str, Any]] = []
        self.image_embeds = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor | None = None,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        self.get_image_features_calls += 1
        return (self.image_embeds,), ()

    def __call__(self, **kwargs: Any) -> _FakeBackboneOutput:
        self.forward_calls.append(dict(kwargs))
        if kwargs.get("pixel_values") is not None:
            self.get_image_features(kwargs["pixel_values"], kwargs.get("image_grid_thw"))
        input_ids = kwargs["input_ids"]
        hidden = torch.arange(
            input_ids.numel() * self.hidden_size,
            dtype=torch.float32,
            device=input_ids.device,
        ).reshape(*input_ids.shape, self.hidden_size)
        return _FakeBackboneOutput(last_hidden_state=hidden)


class _FakeQwenForConditionalGeneration:
    def __init__(self, *, image_token_id: int = 32000, vocab_size: int = 7) -> None:
        self.config = SimpleNamespace(
            model_type="qwen3_vl",
            image_token_id=image_token_id,
            text_config=SimpleNamespace(vocab_size=vocab_size),
        )
        self.model = _FakeLowerQwen(hidden_size=4, image_token_id=image_token_id)
        self.lm_head = torch.nn.Linear(4, vocab_size, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(
                torch.arange(vocab_size * 4, dtype=torch.float32).reshape(vocab_size, 4)
                / 10.0
            )

    def __call__(self, **kwargs: Any) -> _FakeConditionalOutput:
        outputs = self.model(**kwargs)
        hidden_states = outputs[0]
        return _FakeConditionalOutput(
            logits=self.lm_head(hidden_states),
            hidden_states=hidden_states,
        )


class _FakeTrainableWrapper:
    def __init__(self, base_model: _FakeQwenForConditionalGeneration) -> None:
        self.config = SimpleNamespace(model_type="peft")
        self.base_model = base_model

    def get_base_model(self) -> _FakeQwenForConditionalGeneration:
        return self.base_model


def _qwen_inputs(*, image_token_id: int = 32000) -> dict[str, Any]:
    return {
        "input_ids": torch.tensor([[11, image_token_id, image_token_id, 12]]),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
        "pixel_values": torch.ones((1, 3, 2, 2), dtype=torch.float32),
        "image_grid_thw": torch.tensor([[1, 1, 2]], dtype=torch.long),
        "labels": torch.tensor([[11, image_token_id, image_token_id, 12]]),
        "training_sidecars": object(),
        "supervision_spans": ("sidecar-only",),
    }


def test_capture_returns_full_products_with_same_forward_parity() -> None:
    model = _FakeQwenForConditionalGeneration()
    inputs = _qwen_inputs()

    capture = CoverageLedgerForwardCapture().capture(
        model=model,
        inputs=inputs,
        ignored_keys=("labels", "training_sidecars", "supervision_spans"),
        packing_enabled=False,
        where="test",
    )
    assert model.model.get_image_features_calls == 1
    assert len(model.model.forward_calls) == 1
    assert set(model.model.forward_calls[0]) == {
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
    }

    normal = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        pixel_values=inputs["pixel_values"],
        image_grid_thw=inputs["image_grid_thw"],
    )

    assert torch.allclose(capture.logits, normal.logits)
    assert torch.equal(capture.final_hidden_states, normal.hidden_states)
    assert torch.equal(capture.image_embeds, model.model.image_embeds)
    assert capture.logits.shape[:2] == inputs["input_ids"].shape
    assert model.model.get_image_features_calls == 2
    assert len(model.model.forward_calls) == 2


def test_capture_rejects_logits_to_keep_before_qwen_forward() -> None:
    model = _FakeQwenForConditionalGeneration()
    inputs = {**_qwen_inputs(), "logits_to_keep": 1}

    with pytest.raises(ValueError, match="logits_to_keep"):
        CoverageLedgerForwardCapture().capture(
            model=model,
            inputs=inputs,
            ignored_keys=("labels", "training_sidecars", "supervision_spans"),
            packing_enabled=False,
            where="test",
        )

    assert model.model.forward_calls == []
    assert model.model.get_image_features_calls == 0


def test_capture_unwraps_trainable_wrapper_without_copying_base_model() -> None:
    model = _FakeQwenForConditionalGeneration()
    wrapper = _FakeTrainableWrapper(model)

    capture = CoverageLedgerForwardCapture().capture(
        model=wrapper,
        inputs=_qwen_inputs(),
        ignored_keys=("labels", "training_sidecars", "supervision_spans"),
        packing_enabled=False,
        where="test",
    )

    assert capture.logits.shape == (1, 4, model.config.text_config.vocab_size)
    assert model.model.get_image_features_calls == 1
    assert len(model.model.forward_calls) == 1


def test_capture_rejects_missing_empty_or_count_mismatched_image_embeds() -> None:
    model = _FakeQwenForConditionalGeneration()
    model.model.image_embeds = torch.empty((0, 4), dtype=torch.float32)

    with pytest.raises(ValueError, match="image_embeds.*empty|empty.*image_embeds"):
        CoverageLedgerForwardCapture().capture(
            model=model,
            inputs=_qwen_inputs(),
            ignored_keys=("labels", "training_sidecars", "supervision_spans"),
            packing_enabled=False,
            where="test",
        )

    mismatch_model = _FakeQwenForConditionalGeneration()
    mismatch_inputs = _qwen_inputs()
    mismatch_inputs["image_grid_thw"] = torch.tensor([[1, 1, 1]], dtype=torch.long)

    with pytest.raises(ValueError, match="image.*count|count.*image"):
        CoverageLedgerForwardCapture().capture(
            model=mismatch_model,
            inputs=mismatch_inputs,
            ignored_keys=("labels", "training_sidecars", "supervision_spans"),
            packing_enabled=False,
            where="test",
        )


def test_capture_rejects_non_qwen_conditional_generation_object() -> None:
    class _NotQwen:
        config = SimpleNamespace(model_type="qwen3_vl")

    with pytest.raises(TypeError, match="Qwen3-VL.*conditional"):
        CoverageLedgerForwardCapture().capture(
            model=_NotQwen(),
            inputs=_qwen_inputs(),
            ignored_keys=("labels", "training_sidecars", "supervision_spans"),
            packing_enabled=False,
            where="test",
        )
