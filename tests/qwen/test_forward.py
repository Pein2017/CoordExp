from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from src.common.errors import QwenForwardContractError
from src.config.loader import load_train_config
from src.data import load_raw_examples
from src.packing.planner import plan_packed_sequences
from src.qwen.encoding import encode_rendered_example
from src.qwen.forward import build_qwen_forward_inputs, run_qwen_forward
from src.qwen.loading import load_qwen_components
from src.qwen.positions import build_qwen_position_inputs
from src.templates import render_example


FIXTURE_CONFIG = Path("tests/fixtures/smoke/qwen3_vl_single_image_pack/config.yaml")
IMAGE_TOKEN_ID = 151655


def test_forward_inputs_concatenate_pack_positions_and_visual_payloads() -> None:
    examples = _fake_examples()
    pack = plan_packed_sequences(examples, global_max_length=32)[0]
    positions = build_qwen_position_inputs(pack, examples)

    forward_inputs = build_qwen_forward_inputs(pack, examples, positions)

    assert tuple(forward_inputs.input_ids.shape) == (1, pack.length)
    assert tuple(forward_inputs.position_ids.shape) == (4, 1, pack.length)
    assert torch.equal(forward_inputs.position_ids, positions.position_ids)
    assert tuple(forward_inputs.pixel_values.shape) == (48, 8)
    assert forward_inputs.image_grid_thw.tolist() == [[1, 4, 6], [1, 4, 6]]
    assert forward_inputs.to_model_kwargs()["labels"] is None
    assert forward_inputs.to_model_kwargs()["use_cache"] is False
    assert forward_inputs.to_model_kwargs()["logits_to_keep"] == 0
    assert "inputs_embeds" not in forward_inputs.to_model_kwargs()
    artifact = forward_inputs.to_artifact_dict()
    assert artifact["placeholder_token_count"] == 12
    assert artifact["expected_visual_token_count"] == 12
    assert artifact["pixel_values_shape"] == [48, 8]
    assert artifact["position_ids_shape"] == [4, 1, pack.length]


def test_forward_runner_disables_model_loss_cache_and_partial_logits() -> None:
    examples = _fake_examples()
    pack = plan_packed_sequences(examples, global_max_length=32)[0]
    positions = build_qwen_position_inputs(pack, examples)
    forward_inputs = build_qwen_forward_inputs(pack, examples, positions)
    model = FakeQwenModel(vocab_size=17, loss=torch.tensor(3.0))

    result = run_qwen_forward(model, forward_inputs, expected_vocab_size=17)

    assert model.calls == 1
    assert model.last_kwargs["labels"] is None
    assert model.last_kwargs["use_cache"] is False
    assert model.last_kwargs["logits_to_keep"] == 0
    assert "inputs_embeds" not in model.last_kwargs
    assert tuple(result.logits.shape) == (1, pack.length, 17)
    assert result.receipt.model_loss_present is True
    assert result.receipt.output_logits_shape == (1, pack.length, 17)
    assert result.receipt.to_artifact_dict()["model_loss_ignored"] is True


def test_forward_inputs_accept_explicit_logits_to_keep_positions() -> None:
    examples = _fake_examples()
    pack = plan_packed_sequences(examples, global_max_length=32)[0]
    positions = build_qwen_position_inputs(pack, examples)
    forward_inputs = build_qwen_forward_inputs(
        pack,
        examples,
        positions,
        logits_to_keep_positions=(0, 3, 11),
    )
    model = FakeQwenModel(vocab_size=17)

    result = run_qwen_forward(model, forward_inputs, expected_vocab_size=17)

    kept = model.last_kwargs["logits_to_keep"]
    assert isinstance(kept, torch.Tensor)
    assert kept.tolist() == [0, 3, 11]
    assert tuple(result.logits.shape) == (1, 3, 17)
    assert result.logits_position_ids == (0, 3, 11)
    artifact = result.receipt.to_artifact_dict()
    assert artifact["logits_to_keep"] == [0, 3, 11]
    assert artifact["output_logits_shape"] == [1, 3, 17]


def test_forward_runner_rejects_manual_logits_to_keep_override_with_owned_path_hint() -> None:
    examples = _fake_examples()
    pack = plan_packed_sequences(examples, global_max_length=32)[0]
    positions = build_qwen_position_inputs(pack, examples)
    forward_inputs = build_qwen_forward_inputs(pack, examples, positions)
    model = FakeQwenModel(vocab_size=17)

    with pytest.raises(QwenForwardContractError) as exc_info:
        run_qwen_forward(
            model,
            forward_inputs,
            expected_vocab_size=17,
            extra_model_kwargs={"logits_to_keep": 1},
        )

    assert exc_info.value.code == "qwen.forward_logits_to_keep"
    assert "logits_to_keep_positions" in str(exc_info.value)
    assert model.calls == 0


@pytest.mark.parametrize(
    ("extra_kwargs", "code"),
    [
        ({"inputs_embeds": torch.zeros((1, 2, 3))}, "qwen.forward_inputs_embeds"),
        ({"labels": torch.zeros((1, 2), dtype=torch.long)}, "qwen.forward_labels"),
        ({"use_cache": True}, "qwen.forward_use_cache"),
        ({"use_cache": 1}, "qwen.forward_use_cache"),
        ({"logits_to_keep": 1}, "qwen.forward_logits_to_keep"),
        ({"pixel_values_videos": torch.zeros((1, 8))}, "qwen.forward_video_payload"),
        ({"video_grid_thw": torch.zeros((1, 3), dtype=torch.long)}, "qwen.forward_video_payload"),
        ({"input_ids": torch.zeros((1, 2), dtype=torch.long)}, "qwen.forward_boundary_override"),
        (
            {"position_ids": torch.zeros((4, 1, 2), dtype=torch.long)},
            "qwen.forward_boundary_override",
        ),
        ({"pixel_values": torch.zeros((1, 8))}, "qwen.forward_boundary_override"),
        (
            {"image_grid_thw": torch.zeros((1, 3), dtype=torch.long)},
            "qwen.forward_boundary_override",
        ),
        ({"past_key_values": object()}, "qwen.forward_boundary_override"),
        ({"cache_position": torch.arange(2)}, "qwen.forward_boundary_override"),
    ],
)
def test_forward_runner_rejects_unsafe_overrides(
    extra_kwargs: dict[str, Any],
    code: str,
) -> None:
    examples = _fake_examples()
    pack = plan_packed_sequences(examples, global_max_length=32)[0]
    positions = build_qwen_position_inputs(pack, examples)
    forward_inputs = build_qwen_forward_inputs(pack, examples, positions)
    model = FakeQwenModel(vocab_size=17)

    with pytest.raises(QwenForwardContractError) as exc_info:
        run_qwen_forward(
            model,
            forward_inputs,
            expected_vocab_size=17,
            extra_model_kwargs=extra_kwargs,
        )

    assert exc_info.value.code == code
    assert model.calls == 0


def test_forward_runner_rejects_sliced_or_wrong_vocab_logits() -> None:
    examples = _fake_examples()
    pack = plan_packed_sequences(examples, global_max_length=32)[0]
    positions = build_qwen_position_inputs(pack, examples)
    forward_inputs = build_qwen_forward_inputs(pack, examples, positions)

    with pytest.raises(QwenForwardContractError) as exc_info:
        run_qwen_forward(
            FakeQwenModel(vocab_size=17, seq_length=pack.length - 1),
            forward_inputs,
            expected_vocab_size=17,
        )
    assert exc_info.value.code == "qwen.forward_logits_shape"

    with pytest.raises(QwenForwardContractError) as exc_info:
        run_qwen_forward(
            FakeQwenModel(vocab_size=16),
            forward_inputs,
            expected_vocab_size=17,
        )
    assert exc_info.value.code == "qwen.forward_logits_shape"


def test_forward_inputs_reject_placeholder_grid_mismatch_before_model_call() -> None:
    examples = _fake_examples()
    pack = plan_packed_sequences(examples, global_max_length=32)[0]
    positions = build_qwen_position_inputs(pack, examples)
    bad_examples = (
        replace(examples[0], image_encoding=_fake_image_encoding(grid=(1, 4, 8))),
        examples[1],
    )

    with pytest.raises(QwenForwardContractError) as exc_info:
        build_qwen_forward_inputs(pack, bad_examples, positions)

    assert exc_info.value.code == "qwen.forward_grid_mismatch"


def test_real_smoke_forward_inputs_use_encoded_visual_payloads_without_model_load() -> None:
    resolved = load_train_config(FIXTURE_CONFIG)
    components = load_qwen_components(resolved.config, load_model=False)
    examples = []
    for raw in load_raw_examples(resolved.config.data.train):
        rendered = render_example(raw, resolved.config.template)
        examples.append(
            encode_rendered_example(
                raw,
                rendered,
                components=components,
                processor_config=resolved.config.model.processor,
                global_max_length=resolved.config.packing.global_max_length,
                materialize_image_pixels=False,
            )
        )
    examples = tuple(examples)
    assert all(example.image_encoding.pixel_values is None for example in examples)
    pack = plan_packed_sequences(examples, global_max_length=12_000)[0]
    positions = build_qwen_position_inputs(pack, examples)

    forward_inputs = build_qwen_forward_inputs(pack, examples, positions)

    assert tuple(forward_inputs.input_ids.shape) == (1, 2068)
    assert tuple(forward_inputs.position_ids.shape) == (4, 1, 2068)
    assert forward_inputs.image_grid_thw.tolist() == [[1, 52, 78], [1, 72, 54]]
    assert tuple(forward_inputs.pixel_values.shape) == (7944, 1536)
    assert forward_inputs.receipt.segment_count == 2
    assert forward_inputs.receipt.placeholder_token_count == 1986


def _fake_examples() -> tuple["FakeEncodedExample", ...]:
    return (
        FakeEncodedExample(
            example_id="ex-0",
            input_ids=(10, 11, 12, *([IMAGE_TOKEN_ID] * 6), 13, 14),
            image_pad_physical_start=3,
            image_pad_physical_end=9,
            image_encoding=_fake_image_encoding(fill=1.0),
        ),
        FakeEncodedExample(
            example_id="ex-1",
            input_ids=(20, 21, *([IMAGE_TOKEN_ID] * 6), 22),
            image_pad_physical_start=2,
            image_pad_physical_end=8,
            image_encoding=_fake_image_encoding(fill=2.0),
        ),
    )


def _fake_image_encoding(
    *,
    grid: tuple[int, int, int] = (1, 4, 6),
    fill: float = 1.0,
) -> "FakeImageEncoding":
    return FakeImageEncoding(
        image_grid_thw=grid,
        merged_visual_tokens=grid[0] * grid[1] * grid[2] // 4,
        pixel_values=torch.full((grid[0] * grid[1] * grid[2], 8), fill),
        image_grid_thw_tensor=torch.tensor([list(grid)], dtype=torch.long),
        plan=FakeImagePlan(merge_size=2),
    )


@dataclass(frozen=True)
class FakeEncodedExample:
    example_id: str
    input_ids: tuple[int, ...]
    image_pad_physical_start: int
    image_pad_physical_end: int
    image_encoding: "FakeImageEncoding"


@dataclass(frozen=True)
class FakeImageEncoding:
    image_grid_thw: tuple[int, int, int]
    merged_visual_tokens: int
    pixel_values: torch.Tensor
    image_grid_thw_tensor: torch.Tensor
    plan: "FakeImagePlan"


@dataclass(frozen=True)
class FakeImagePlan:
    merge_size: int


class FakeQwenModel:
    def __init__(
        self,
        *,
        vocab_size: int,
        seq_length: int | None = None,
        loss: torch.Tensor | None = None,
    ) -> None:
        self.config = SimpleNamespace(text_config=SimpleNamespace(vocab_size=vocab_size))
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.loss = loss
        self.calls = 0
        self.last_kwargs: dict[str, Any] = {}

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        self.calls += 1
        self.last_kwargs = dict(kwargs)
        seq_length = self.seq_length
        if seq_length is None:
            logits_to_keep = kwargs["logits_to_keep"]
            seq_length = (
                int(logits_to_keep.numel())
                if isinstance(logits_to_keep, torch.Tensor)
                else int(kwargs["input_ids"].shape[1])
            )
        return SimpleNamespace(
            logits=torch.zeros((1, seq_length, self.vocab_size), dtype=torch.float32),
            loss=self.loss,
            past_key_values=None,
            rope_deltas=None,
        )
