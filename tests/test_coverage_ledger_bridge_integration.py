from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import src.training.bridge.loss_bridge as loss_bridge_module
from src.metrics.events import flatten_metric_events
from src.trainers.batch_extras import BatchExtras
from src.training.bridge import TrainerLossBridge, TrainerLossBridgeSettings
from src.training.coverage_ledger.head import CoverageLedgerHead
from src.training.coverage_ledger.loss import (
    CoverageLedgerDebugRows,
    CoverageLedgerLossConfig,
    CoverageLedgerLossResult,
    compute_coverage_ledger_loss,
)
from src.training.coverage_ledger.metrics import WEIGHTED_LOSS_KEY
from src.training.coverage_ledger.sidecars import (
    CoverageLedgerObjectEntry,
    CoverageLedgerSidecar,
)
from src.training.coverage_ledger.visual_regions import (
    map_norm1000_bbox_to_visual_token_region,
    pool_object_visual_embeddings,
)
from src.training.objectives.types import ObjectiveSpec
from src.training.sidecars import SupervisionSidecars, TrainingSidecars
from src.training.supervision.batch import SupervisionBatch
from src.training.supervision.distributions import HardTokenDistribution
from src.training.supervision.spans import SupervisionSpan
from src.training.teacher_forcing.packing_offsets import PackedSegmentOffset


@dataclass
class _Outputs:
    logits: torch.Tensor


@dataclass(frozen=True)
class _CaptureResult:
    logits: torch.Tensor
    final_hidden_states: torch.Tensor
    image_embeds: torch.Tensor
    outputs: Any


class _FakeQwenModel(torch.nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3_vl")
        self.model = SimpleNamespace(
            visual=SimpleNamespace(patch_size=2, spatial_merge_size=1),
            config=SimpleNamespace(patch_size=2, spatial_merge_size=1),
        )
        self.logits = logits
        self.calls: list[dict[str, Any]] = []

    def forward(self, **kwargs: Any) -> _Outputs:
        self.calls.append(dict(kwargs))
        return _Outputs(logits=self.logits)


class _FakeCapture:
    calls: list[dict[str, Any]] = []
    result: _CaptureResult | None = None

    def capture(self, **kwargs: Any) -> _CaptureResult:
        self.__class__.calls.append(dict(kwargs))
        if self.__class__.result is None:
            raise AssertionError("test did not install a fake capture result")
        return self.__class__.result


def _raw_batch(*, time_steps: int = 5) -> dict[str, Any]:
    return {
        "input_ids": torch.tensor([[0, 2, 3, 4, 5][:time_steps]], dtype=torch.long),
        "attention_mask": torch.ones((1, time_steps), dtype=torch.long),
        "pixel_values": torch.ones((1, 3, 4, 4), dtype=torch.float32),
        "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),
    }


def _packed_raw_batch() -> dict[str, Any]:
    return {
        "input_ids": torch.tensor([[0, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=torch.long),
        "attention_mask": torch.ones((1, 10), dtype=torch.long),
        "pixel_values": torch.ones((2, 3, 4, 4), dtype=torch.float32),
        "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]], dtype=torch.long),
        "position_ids": torch.arange(10, dtype=torch.long).reshape(1, 1, 10).repeat(4, 1, 1),
    }


def _sidecar(sample_id: str = "sample-1") -> CoverageLedgerSidecar:
    return CoverageLedgerSidecar(
        sample_id=sample_id,
        prompt_end_position=1,
        object_entries=(
            CoverageLedgerObjectEntry(
                object_instance_id=f"{sample_id}:object-0",
                source_object_index=0,
                emitted_order_index=0,
                image_index=0,
                bbox_norm1000_xyxy=(0, 0, 999, 999),
                box_start_position=2,
                coord_label_positions=(3, 4, 4, 4),
                object_ref_end_position=1,
                box_end_position=3,
            ),
        ),
        image_grid_thw=(1, 2, 2),
        processed_width=4,
        processed_height=4,
        image_identity=f"{sample_id}.jpg",
    )


def _shifted_sidecar(
    *,
    sample_id: str,
    token_delta: int,
    segment_index: int,
) -> CoverageLedgerSidecar:
    base = _sidecar(sample_id=sample_id)
    entry = base.object_entries[0]
    return CoverageLedgerSidecar(
        sample_id=base.sample_id,
        prompt_end_position=base.prompt_end_position + token_delta,
        object_entries=(
            CoverageLedgerObjectEntry(
                object_instance_id=entry.object_instance_id,
                source_object_index=entry.source_object_index,
                emitted_order_index=entry.emitted_order_index,
                image_index=entry.image_index,
                bbox_norm1000_xyxy=entry.bbox_norm1000_xyxy,
                box_start_position=entry.box_start_position + token_delta,
                coord_label_positions=tuple(
                    position + token_delta for position in entry.coord_label_positions
                ),
                object_ref_end_position=entry.object_ref_end_position + token_delta,
                box_end_position=entry.box_end_position + token_delta,
            ),
        ),
        image_grid_thw=base.image_grid_thw,
        processed_width=base.processed_width,
        processed_height=base.processed_height,
        image_identity=base.image_identity,
        packed_source_sample_id=sample_id,
        packed_row_index=0,
        packed_segment_index=segment_index,
        packed_token_start=token_delta,
        packed_token_end=token_delta + 5,
    )


def _supervision(sample_id: str = "sample-1") -> SupervisionBatch:
    return SupervisionBatch(
        spans=(
            SupervisionSpan(
                sample_id=sample_id,
                role="schema",
                label_positions=(1,),
                distribution=HardTokenDistribution(token_id=2),
                provenance="unit",
            ),
        ),
        batch_id="unit",
    )


def _settings(enabled: bool = True, **overrides: Any) -> TrainerLossBridgeSettings:
    cfg = {
        "enabled": enabled,
        "coverage_weight": 0.5,
        "region_anchor_weight": 0.25,
        "temperature": 1.0,
        "pos_weight": 1.0,
        "log_auc": True,
        "log_accuracy": True,
    }
    cfg.update(overrides)
    return TrainerLossBridgeSettings(coverage_ledger=cfg)


def _install_identity_head(model: _FakeQwenModel) -> CoverageLedgerHead:
    head = CoverageLedgerHead(
        hidden_size=3,
        visual_dim=3,
        ledger_projection_dim=3,
        normalize_eps=1.0e-6,
    )
    with torch.no_grad():
        eye = torch.eye(3, dtype=torch.float32)
        head.state_projection.weight.copy_(eye)
        head.region_anchor_state_projection.weight.copy_(eye)
        head.object_projection.weight.copy_(eye)
    model.add_module("coverage_ledger_head", head)
    return head


def _enabled_sidecars(*payloads: Any) -> TrainingSidecars:
    return TrainingSidecars(
        supervision=SupervisionSidecars(payloads=tuple(payloads))
    )


def test_disabled_coverage_ledger_uses_existing_model_path_without_head_or_sidecar() -> None:
    logits = torch.zeros((1, 5, 6), dtype=torch.float32)
    model = _FakeQwenModel(logits)

    result = TrainerLossBridge(
        settings=TrainerLossBridgeSettings(coverage_ledger={"enabled": False})
    ).compute_loss(
        model=model,
        raw_batch=_raw_batch(),
        supervision=SupervisionBatch(),
        objectives=(ObjectiveSpec("token_ce"),),
    )

    assert len(model.calls) == 1
    assert result.loss.item() == pytest.approx(0.0)
    assert result.metric_events == ()


def test_disabled_coverage_ledger_ignores_ledger_sidecar_and_does_not_require_head() -> None:
    logits = torch.zeros((1, 5, 6), dtype=torch.float32)
    model = _FakeQwenModel(logits)
    sidecar = _sidecar()

    result = TrainerLossBridge(
        settings=TrainerLossBridgeSettings(coverage_ledger={"enabled": False})
    ).compute_loss(
        model=model,
        raw_batch={**_raw_batch(), "training_sidecars": _enabled_sidecars(sidecar)},
        supervision=SupervisionBatch(),
        objectives=(ObjectiveSpec("token_ce"),),
    )

    assert len(model.calls) == 1
    assert result.training_sidecars.supervision.payloads == (sidecar,)


def test_enabled_coverage_ledger_requires_exactly_one_sidecar_and_head() -> None:
    model = _FakeQwenModel(torch.zeros((1, 5, 6), dtype=torch.float32))
    _install_identity_head(model)

    with pytest.raises(ValueError, match="exactly one CoverageLedgerSidecar"):
        TrainerLossBridge(settings=_settings()).compute_loss(
            model=model,
            raw_batch=_raw_batch(),
            supervision=SupervisionBatch(),
            objectives=(ObjectiveSpec("token_ce"),),
        )

    with pytest.raises(ValueError, match="exactly one CoverageLedgerSidecar"):
        TrainerLossBridge(settings=_settings()).compute_loss(
            model=model,
            raw_batch={
                **_raw_batch(),
                "training_sidecars": _enabled_sidecars(_sidecar("a"), _sidecar("b")),
            },
            supervision=SupervisionBatch(),
            objectives=(ObjectiveSpec("token_ce"),),
        )

    missing_head = _FakeQwenModel(torch.zeros((1, 5, 6), dtype=torch.float32))
    with pytest.raises(ValueError, match="exactly one coverage_ledger_head"):
        TrainerLossBridge(settings=_settings()).compute_loss(
            model=missing_head,
            raw_batch={**_raw_batch(), "training_sidecars": _enabled_sidecars(_sidecar())},
            supervision=SupervisionBatch(),
            objectives=(ObjectiveSpec("token_ce"),),
        )


def test_enabled_coverage_ledger_uses_capture_sums_loss_and_combines_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logits = torch.tensor(
        [[[0.0, 0.0, 2.0, -1.0, -1.0, -1.0]] + [[0.0] * 6 for _ in range(4)]],
        dtype=torch.float32,
    )
    model = _FakeQwenModel(logits)
    _install_identity_head(model)
    hidden = torch.zeros((1, 5, 3), dtype=torch.float32)
    hidden[0, 1] = torch.tensor([-1.0, 0.0, 0.0])
    hidden[0, 2] = torch.tensor([1.0, 0.0, 0.0])
    hidden[0, 3] = torch.tensor([1.0, 0.0, 0.0])
    image_embeds = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    capture_outputs = SimpleNamespace(marker="lower-qwen-output")
    _FakeCapture.calls = []
    _FakeCapture.result = _CaptureResult(
        logits=logits,
        final_hidden_states=hidden,
        image_embeds=image_embeds,
        outputs=capture_outputs,
    )
    monkeypatch.setattr(loss_bridge_module, "CoverageLedgerForwardCapture", _FakeCapture)

    sidecar = _sidecar()
    result = TrainerLossBridge(settings=_settings()).compute_loss(
        model=model,
        raw_batch={**_raw_batch(), "training_sidecars": _enabled_sidecars(sidecar)},
        supervision=_supervision(),
        objectives=(ObjectiveSpec("token_ce"),),
        sample_id_to_batch_index={"sample-1": 0},
    )
    regions = tuple(
        map_norm1000_bbox_to_visual_token_region(
            entry.bbox_norm1000_xyxy,
            image_grid_thw=sidecar.image_grid_thw,
            processed_width=sidecar.processed_width,
            processed_height=sidecar.processed_height,
            patch_size=2,
            spatial_merge_size=1,
        )
        for entry in sidecar.object_entries
    )
    expected_ledger = compute_coverage_ledger_loss(
        head=model.coverage_ledger_head,
        final_hidden_states=hidden,
        pooled_visual_object_embeddings=pool_object_visual_embeddings(
            image_embeds,
            regions,
        ),
        sidecar=sidecar,
        config=CoverageLedgerLossConfig(
            coverage_weight=0.5,
            region_anchor_weight=0.25,
            temperature=1.0,
            pos_weight=1.0,
        ),
        sample_id_to_batch_index={"sample-1": 0},
    )

    assert len(_FakeCapture.calls) == 1
    assert model.calls == []
    assert "training_sidecars" not in _FakeCapture.calls[0]["inputs"]
    assert result.outputs is _FakeCapture.result
    assert result.outputs.logits is logits
    assert result.outputs.final_hidden_states is hidden
    assert result.outputs.image_embeds is image_embeds
    assert result.loss.item() == pytest.approx(
        result.objective_result.loss.item() + expected_ledger.weighted_loss.item()
    )
    metric_keys = {event.key for event in result.metric_events}
    assert "training/objectives/token_ce/loss" in metric_keys
    assert not any(
        key.startswith("training/objectives/coverage_ledger/")
        for key in metric_keys
    )
    assert "teacher_forcing/ledger/object_count" in metric_keys
    assert "teacher_forcing/ledger/coverage_state_count" in metric_keys
    assert "teacher_forcing/ledger/coverage_pair_count" in metric_keys
    assert "teacher_forcing/ledger/row_object_binding_pair_count" in metric_keys
    assert WEIGHTED_LOSS_KEY in metric_keys


def _fake_coverage_ledger_result(*, weighted_loss: float) -> CoverageLedgerLossResult:
    scalar = torch.tensor(float(weighted_loss), dtype=torch.float32)
    return CoverageLedgerLossResult(
        total_loss=scalar,
        coverage_loss=scalar,
        region_anchor_loss=scalar,
        weighted_loss=scalar,
        coverage_weight=1.0,
        region_anchor_weight=1.0,
        metric_events=(),
        debug_rows=CoverageLedgerDebugRows(
            coverage_state_positions=(1,),
            region_anchor_positions=(2,),
            region_anchor_object_indices=(0,),
            coverage_targets=torch.ones((1, 1), dtype=torch.float32),
            coverage_logits=torch.zeros((1, 1), dtype=torch.float32),
            region_anchor_targets=torch.ones((1, 1), dtype=torch.float32),
            region_anchor_logits=torch.zeros((1, 1), dtype=torch.float32),
            object_count=1,
            coverage_state_count=1,
            coverage_pair_count=1,
            region_anchor_pair_count=1,
        ),
    )


def test_bridge_accepts_multiple_packed_coverage_ledger_sidecars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _FakeQwenModel(torch.zeros((1, 10, 32), dtype=torch.float32))
    model.add_module(
        "coverage_ledger_head",
        CoverageLedgerHead(
            hidden_size=8,
            visual_dim=8,
            ledger_projection_dim=4,
            normalize_eps=1.0e-6,
        ),
    )
    hidden = torch.zeros((1, 10, 8), dtype=torch.float32)
    image_embeds = torch.cat(
        (
            torch.ones((4, 8), dtype=torch.float32),
            torch.full((4, 8), 7.0, dtype=torch.float32),
        ),
        dim=0,
    )
    _FakeCapture.calls = []
    _FakeCapture.result = _CaptureResult(
        logits=torch.zeros((1, 10, 32), dtype=torch.float32),
        final_hidden_states=hidden,
        image_embeds=image_embeds,
        outputs=SimpleNamespace(),
    )
    monkeypatch.setattr(loss_bridge_module, "CoverageLedgerForwardCapture", _FakeCapture)
    calls: list[dict[str, Any]] = []

    def fake_compute_coverage_ledger_loss(**kwargs: Any) -> CoverageLedgerLossResult:
        calls.append(kwargs)
        pooled_mean = float(kwargs["pooled_visual_object_embeddings"].mean().item())
        if pooled_mean == pytest.approx(1.0):
            return _fake_coverage_ledger_result(weighted_loss=0.25)
        if pooled_mean == pytest.approx(7.0):
            return _fake_coverage_ledger_result(weighted_loss=0.75)
        raise AssertionError(f"unexpected pooled visual mean: {pooled_mean}")

    monkeypatch.setattr(
        loss_bridge_module,
        "compute_coverage_ledger_loss",
        fake_compute_coverage_ledger_loss,
    )
    first = _shifted_sidecar(
        sample_id="a",
        token_delta=0,
        segment_index=0,
    )
    second = _shifted_sidecar(
        sample_id="b",
        token_delta=5,
        segment_index=1,
    )

    result = TrainerLossBridge(
        settings=TrainerLossBridgeSettings(
            packing_enabled=True,
            coverage_ledger={"enabled": True},
        )
    ).compute_loss(
        model=model,
        raw_batch={**_packed_raw_batch(), "training_sidecars": _enabled_sidecars(first, second)},
        batch_extras=BatchExtras(
            packed_segment_offsets=(
                PackedSegmentOffset(
                    sample_id="a",
                    packed_row_index=0,
                    segment_index=0,
                    token_start=0,
                    token_end=5,
                ),
                PackedSegmentOffset(
                    sample_id="b",
                    packed_row_index=0,
                    segment_index=1,
                    token_start=5,
                    token_end=10,
                ),
            )
        ),
        supervision=SupervisionBatch(),
        objectives=(),
        sample_id_to_batch_index={"a": 0, "b": 0},
    )

    assert len(calls) == 2
    assert [float(call["pooled_visual_object_embeddings"].mean().item()) for call in calls] == [
        pytest.approx(1.0),
        pytest.approx(7.0),
    ]
    assert result.loss.item() == pytest.approx(0.5)
    flat = flatten_metric_events(result.metric_events)
    assert flat[WEIGHTED_LOSS_KEY] == pytest.approx(0.5)
    assert _FakeCapture.calls[0]["packing_enabled"] is True


def test_enabled_coverage_ledger_keeps_full_logits_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _FakeQwenModel(torch.zeros((1, 5, 6), dtype=torch.float32))
    _install_identity_head(model)
    _FakeCapture.calls = []
    _FakeCapture.result = _CaptureResult(
        logits=torch.zeros((1, 2, 6), dtype=torch.float32),
        final_hidden_states=torch.zeros((1, 5, 3), dtype=torch.float32),
        image_embeds=torch.ones((4, 3), dtype=torch.float32),
        outputs=SimpleNamespace(),
    )
    monkeypatch.setattr(loss_bridge_module, "CoverageLedgerForwardCapture", _FakeCapture)

    with pytest.raises(ValueError, match="full logits aligned"):
        TrainerLossBridge(settings=_settings()).compute_loss(
            model=model,
            raw_batch={**_raw_batch(), "training_sidecars": _enabled_sidecars(_sidecar())},
            supervision=SupervisionBatch(),
            objectives=(ObjectiveSpec("token_ce"),),
        )
