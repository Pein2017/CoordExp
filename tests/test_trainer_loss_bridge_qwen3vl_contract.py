from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from src.training.bridge import TrainerLossBridge, TrainerLossBridgeSettings
from src.training.coverage_ledger.head import CoverageLedgerHead
from src.training.coverage_ledger.sidecars import (
    CoverageLedgerObjectEntry,
    CoverageLedgerSidecar,
)
from src.training.encoding.model_inputs import FORWARDED_MODEL_INPUT_KEYS, ModelInputBundle
from src.training.objectives.types import ObjectiveSpec
from src.training.sidecars import SupervisionSidecars, TrainingSidecars
from src.training.supervision.batch import SupervisionBatch
from src.training.supervision.distributions import HardTokenDistribution
from src.training.supervision.spans import SupervisionSpan


@dataclass
class _FakeOutputs:
    logits: torch.Tensor
    loss: torch.Tensor


class _FakeModel:
    def __init__(self, logits: torch.Tensor, *, model_type: str = "qwen3_vl") -> None:
        self.config = SimpleNamespace(model_type=model_type)
        self.logits = logits
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> _FakeOutputs:
        self.calls.append(dict(kwargs))
        return _FakeOutputs(
            logits=self.logits,
            loss=torch.tensor(999.0, dtype=torch.float32),
        )


@dataclass
class _FakeBackboneOutput:
    last_hidden_state: torch.Tensor

    def __getitem__(self, index: int) -> torch.Tensor:
        if index != 0:
            raise IndexError(index)
        return self.last_hidden_state


class _FakeLowerQwen:
    def __init__(self, *, hidden_size: int, image_token_id: int) -> None:
        self.config = SimpleNamespace(image_token_id=image_token_id)
        self.visual = SimpleNamespace(patch_size=16, spatial_merge_size=1)
        self.hidden_size = hidden_size
        self.image_embeds = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        self.get_image_features_calls = 0
        self.forward_calls: list[dict[str, Any]] = []

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
        ).reshape(*input_ids.shape, self.hidden_size)
        return _FakeBackboneOutput(last_hidden_state=hidden)


class _FakeQwenConditionalModel:
    def __init__(self, *, image_token_id: int = 32000, vocab_size: int = 6) -> None:
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
        self.coverage_ledger_head = CoverageLedgerHead(
            hidden_size=4,
            visual_dim=4,
            ledger_projection_dim=4,
            normalize_eps=1.0e-6,
        )


def _span(
    *,
    sample_id: str = "sample-1",
    label_positions: tuple[int, ...] = (1,),
    token_id: int = 2,
) -> SupervisionSpan:
    return SupervisionSpan(
        sample_id=sample_id,
        role="schema",
        label_positions=label_positions,
        distribution=HardTokenDistribution(token_id=token_id),
    )


def _batch(*spans: SupervisionSpan) -> SupervisionBatch:
    return SupervisionBatch(spans=spans, batch_id="batch-1")


def _coverage_ledger_sidecars(
    *,
    image_grid_thw: tuple[int, int, int] = (1, 1, 2),
    processed_width: int = 32,
    processed_height: int = 16,
) -> TrainingSidecars:
    return TrainingSidecars(
        supervision=SupervisionSidecars(
            payloads=(
                CoverageLedgerSidecar(
                    sample_id="sample-1",
                    prompt_end_position=0,
                    object_entries=(
                        CoverageLedgerObjectEntry(
                            object_instance_id="sample-1:ann-1",
                            source_object_index=0,
                            emitted_order_index=0,
                            image_index=0,
                            bbox_norm1000_xyxy=(10, 20, 900, 900),
                            box_start_position=2,
                            coord_label_positions=(3, 3, 3, 3),
                            object_ref_end_position=1,
                            box_end_position=3,
                        ),
                    ),
                    image_grid_thw=image_grid_thw,
                    processed_width=processed_width,
                    processed_height=processed_height,
                    image_identity="fake-image",
                ),
            )
        )
    )


def test_trainer_loss_bridge_calls_model_once_and_uses_runner_loss_not_output_loss() -> None:
    logits = torch.tensor(
        [[[0.0, -1.0, 3.0, 0.5], [0.0, 2.0, -1.0, 0.5]]],
        dtype=torch.float32,
    )
    model = _FakeModel(logits)
    raw_batch = {
        "input_ids": torch.tensor([[11, 12]], dtype=torch.long),
        "attention_mask": torch.ones((1, 2), dtype=torch.long),
        "labels": torch.tensor([[11, 12]], dtype=torch.long),
        "compute_loss_func": object(),
        "loss_scale": torch.tensor([42.0]),
        "supervision_spans": ("sidecar",),
        "training_sidecars": {"sidecar": True},
        "duplicate_filter_result": {"ignored": True},
    }
    supervision = _batch(_span(sample_id="sample-1", label_positions=(1,), token_id=2))

    result = TrainerLossBridge().compute_loss(
        model=model,
        raw_batch=raw_batch,
        supervision=supervision,
        objectives=(ObjectiveSpec("token_ce"),),
        sample_id_to_batch_index={"sample-1": 0},
    )

    expected = -torch.log_softmax(logits[0, 0], dim=-1)[2]
    assert len(model.calls) == 1
    assert set(model.calls[0]) == {"input_ids", "attention_mask"}
    assert result.loss.item() == pytest.approx(expected.item())
    assert result.loss.item() != pytest.approx(999.0)
    assert result.objective_result.loss is result.loss
    assert result.outputs.loss.item() == pytest.approx(999.0)
    assert type(result.model_inputs) is ModelInputBundle
    assert result.coordinate_mapper.logits_shape == tuple(logits.shape)


def test_trainer_loss_bridge_uses_qwen_capture_when_coverage_ledger_enabled() -> None:
    model = _FakeQwenConditionalModel()
    image_token_id = model.config.image_token_id
    raw_batch = {
        "input_ids": torch.tensor(
            [[11, image_token_id, image_token_id, image_token_id, image_token_id]]
        ),
        "attention_mask": torch.ones((1, 5), dtype=torch.long),
        "pixel_values": torch.ones((1, 3, 2, 2), dtype=torch.float32),
        "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),
        "labels": torch.tensor(
            [[11, image_token_id, image_token_id, image_token_id, image_token_id]]
        ),
        "training_sidecars": _coverage_ledger_sidecars(
            image_grid_thw=(1, 2, 2),
            processed_width=32,
            processed_height=32,
        ),
        "supervision_spans": ("sidecar-only",),
        "duplicate_filter_result": {"sidecar": True},
    }

    result = TrainerLossBridge(
        settings=TrainerLossBridgeSettings(
            coverage_ledger={
                "enabled": True,
                "coverage_weight": 0.5,
                "region_anchor_weight": 0.25,
                "temperature": 1.0,
                "pos_weight": 1.0,
            }
        )
    ).compute_loss(
        model=model,
        raw_batch=raw_batch,
        supervision=_batch(_span(sample_id="sample-1", label_positions=(1,), token_id=2)),
        objectives=(ObjectiveSpec("token_ce"),),
        sample_id_to_batch_index={"sample-1": 0},
    )

    assert len(model.model.forward_calls) == 1
    assert model.model.get_image_features_calls == 1
    assert set(model.model.forward_calls[0]) == {
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
    }
    assert result.outputs.logits.shape[:2] == raw_batch["input_ids"].shape
    assert result.outputs.final_hidden_states.shape == (1, 5, 4)
    assert torch.equal(result.outputs.image_embeds, model.model.image_embeds)
    assert result.loss.item() > result.objective_result.loss.item()
    assert "training_sidecars" not in model.model.forward_calls[0]
    assert "supervision_spans" not in model.model.forward_calls[0]
    assert result.coordinate_mapper.logits_shape == tuple(result.outputs.logits.shape)


def test_trainer_loss_bridge_preserves_registered_forward_keys() -> None:
    logits = torch.zeros((1, 2, 5), dtype=torch.float32)
    model = _FakeModel(logits)
    raw_batch = {
        "input_ids": torch.ones((1, 2), dtype=torch.long),
        "attention_mask": torch.ones((1, 2), dtype=torch.long),
        "token_type_ids": torch.zeros((1, 2), dtype=torch.long),
        "pixel_values": torch.zeros((1, 3, 2, 2), dtype=torch.float32),
        "pixel_values_videos": torch.zeros((1, 3, 2, 2), dtype=torch.float32),
        "image_grid_thw": torch.tensor([[1, 1, 1]], dtype=torch.long),
        "video_grid_thw": torch.tensor([[1, 1, 1]], dtype=torch.long),
        "position_ids": torch.arange(6, dtype=torch.long).reshape(3, 1, 2),
        "second_per_grid_ts": torch.tensor([1.0], dtype=torch.float32),
        "cross_attention_mask": torch.ones((1, 2), dtype=torch.long),
        "cache_position": torch.arange(2, dtype=torch.long),
        "past_key_values": ("cached",),
        "use_cache": True,
        "max_length_q": 2,
        "max_length_k": 2,
        "cu_seq_lens": torch.tensor([0, 2], dtype=torch.int32),
        "cu_seq_lens_q": torch.tensor([0, 2], dtype=torch.int32),
        "cu_seq_lens_k": torch.tensor([0, 2], dtype=torch.int32),
        "output_router_logits": True,
        "text_position_ids": torch.arange(2, dtype=torch.long).reshape(1, 2),
    }

    TrainerLossBridge().compute_loss(
        model=model,
        raw_batch=raw_batch,
        supervision=SupervisionBatch(),
        objectives=(ObjectiveSpec("token_ce"),),
    )

    forwarded = model.calls[0]
    assert set(FORWARDED_MODEL_INPUT_KEYS).issubset(forwarded)
    assert "text_position_ids" not in forwarded
    assert tuple(forwarded["position_ids"].shape) == (4, 1, 2)
    assert torch.equal(forwarded["position_ids"][0], raw_batch["text_position_ids"])


def test_trainer_loss_bridge_rejects_logits_to_keep_by_default() -> None:
    model = _FakeModel(torch.zeros((1, 2, 5), dtype=torch.float32))

    with pytest.raises(ValueError, match="logits_to_keep.*projection"):
        TrainerLossBridge().compute_loss(
            model=model,
            raw_batch={
                "input_ids": torch.ones((1, 2), dtype=torch.long),
                "logits_to_keep": 1,
            },
            supervision=SupervisionBatch(),
            objectives=(ObjectiveSpec("token_ce"),),
        )

    assert model.calls == []


def test_trainer_loss_bridge_rejects_unimplemented_logits_projection_switch() -> None:
    model = _FakeModel(torch.zeros((1, 2, 5), dtype=torch.float32))
    bridge = TrainerLossBridge(
        settings=TrainerLossBridgeSettings(allow_logits_projection=True)
    )

    with pytest.raises(NotImplementedError, match="logits_to_keep.*not implemented"):
        bridge.compute_loss(
            model=model,
            raw_batch={
                "input_ids": torch.ones((1, 2), dtype=torch.long),
                "logits_to_keep": 1,
            },
            supervision=SupervisionBatch(),
            objectives=(ObjectiveSpec("token_ce"),),
        )

    assert model.calls == []


def test_trainer_loss_bridge_rejects_sliced_logits_against_input_ids_shape() -> None:
    model = _FakeModel(torch.zeros((1, 1, 5), dtype=torch.float32))

    with pytest.raises(ValueError, match="full logits|required|sliced"):
        TrainerLossBridge().compute_loss(
            model=model,
            raw_batch={"input_ids": torch.ones((1, 2), dtype=torch.long)},
            supervision=SupervisionBatch(),
            objectives=(ObjectiveSpec("token_ce"),),
        )


def test_trainer_loss_bridge_rejects_2d_logits_for_multi_sample_batch() -> None:
    model = _FakeModel(torch.zeros((3, 5), dtype=torch.float32))

    with pytest.raises(ValueError, match="full logits|required|batched|2D"):
        TrainerLossBridge().compute_loss(
            model=model,
            raw_batch={"input_ids": torch.ones((2, 3), dtype=torch.long)},
            supervision=_batch(
                _span(sample_id="sample-a", label_positions=(1,), token_id=2)
            ),
            objectives=(ObjectiveSpec("token_ce"),),
            sample_id_to_batch_index={"sample-a": 0},
        )

    assert len(model.calls) == 1


def test_trainer_loss_bridge_rejects_model_owned_loss_mode_until_implemented() -> None:
    with pytest.raises(NotImplementedError, match="runner_owns_loss=False"):
        TrainerLossBridge(
            settings=TrainerLossBridgeSettings(runner_owns_loss=False)
        ).compute_loss(
            model=_FakeModel(torch.zeros((1, 2, 5), dtype=torch.float32)),
            raw_batch={
                "input_ids": torch.ones((1, 2), dtype=torch.long),
                "labels": torch.ones((1, 2), dtype=torch.long),
            },
            supervision=SupervisionBatch(),
            objectives=(ObjectiveSpec("token_ce"),),
        )
