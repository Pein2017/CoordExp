from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import src.training.bridge.loss_bridge as loss_bridge_module
from src.metrics.events import flatten_metric_events
from src.training.bridge import TrainerLossBridge, TrainerLossBridgeSettings
from src.training.coverage_ledger.head import CoverageLedgerHead
from src.training.coverage_ledger.loss import (
    CoverageLedgerLossConfig,
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
                bbox_norm1000_xyxy=(0, 0, 1000, 1000),
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
    assert result.outputs.outputs is capture_outputs
    assert result.loss.item() == pytest.approx(
        result.objective_result.loss.item() + expected_ledger.weighted_loss.item()
    )
    metric_keys = {event.key for event in result.metric_events}
    assert "training/objectives/token_ce/loss" in metric_keys
    assert WEIGHTED_LOSS_KEY in metric_keys


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
