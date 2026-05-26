from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.tokens.row_offsets import CoordOffsetAdapter
from src.trainers.rollout_runtime.vllm_server import (
    _filter_vllm_adapter_lora_tensors,
    _sync_vllm_server_coord_offset_adapter,
    _vllm_adapter_peft_config,
)


def test_vllm_adapter_payload_strips_modules_to_save_config() -> None:
    payload, dropped = _vllm_adapter_peft_config(
        SimpleNamespace(
            to_dict=lambda: {
                "r": 16,
                "target_modules": ["q_proj"],
                "modules_to_save": ["coord_offset_adapter"],
            }
        )
    )

    assert payload["modules_to_save"] is None
    assert dropped == ("coord_offset_adapter",)


def test_vllm_adapter_payload_keeps_only_lora_tensors() -> None:
    kept, dropped = _filter_vllm_adapter_lora_tensors(
        OrderedDict(
            [
                ("base_model.model.q_proj.lora_A.default.weight", torch.ones(1)),
                ("base_model.model.q_proj.lora_B.default.weight", torch.ones(1)),
                ("modules_to_save.default.coord_ids", torch.ones(1)),
                (
                    "base_model.model.coord_offset_adapter.modules_to_save.default.coord_embed_offsets",
                    torch.ones(1),
                ),
            ]
        )
    )

    assert list(kept) == [
        "base_model.model.q_proj.lora_A.default.weight",
        "base_model.model.q_proj.lora_B.default.weight",
    ]
    assert dropped == (
        "modules_to_save.default.coord_ids",
        "base_model.model.coord_offset_adapter.modules_to_save.default.coord_embed_offsets",
    )


class _TinyCoordModel(nn.Module):
    def __init__(self, adapter: CoordOffsetAdapter | None = None) -> None:
        super().__init__()
        if adapter is not None:
            self.coord_offset_adapter = adapter


class _FakeClient:
    def __init__(self) -> None:
        self.calls = []

    def update_token_row_offsets(
        self,
        coord_ids: torch.Tensor,
        embed_offset: torch.Tensor,
        *,
        head_offset: torch.Tensor | None = None,
        tie_head: bool = True,
    ) -> None:
        self.calls.append(
            {
                "coord_ids": coord_ids.detach().cpu().clone(),
                "embed_offset": embed_offset.detach().cpu().clone(),
                "head_offset": (
                    head_offset.detach().cpu().clone()
                    if torch.is_tensor(head_offset)
                    else None
                ),
                "tie_head": bool(tie_head),
            }
        )


def test_vllm_adapter_coord_offset_sync_sends_row_payload() -> None:
    adapter = CoordOffsetAdapter(
        coord_ids=[2, 5],
        tie_head=True,
        embed_dim=4,
        head_dim=4,
        base_dtype=torch.float32,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        adapter.embed_offset.copy_(
            torch.tensor(
                [[0.25, -0.5, 0.75, 1.0], [-1.25, 0.5, 0.125, -0.75]],
                dtype=torch.float32,
            )
        )
    client = _FakeClient()

    _sync_vllm_server_coord_offset_adapter(
        owner=SimpleNamespace(model=_TinyCoordModel(adapter)),
        client=client,
        logger=SimpleNamespace(info=lambda *args, **kwargs: None),
        dropped_modules_to_save=("coord_offset_adapter",),
        dropped_param_names=("base_model.model.coord_offset_adapter.embed_offset",),
    )

    assert len(client.calls) == 1
    call = client.calls[0]
    assert torch.equal(call["coord_ids"], torch.tensor([2, 5]))
    assert torch.allclose(call["embed_offset"], adapter.embed_offset.detach())
    assert call["head_offset"] is None
    assert call["tie_head"] is True


def test_vllm_adapter_coord_offset_sync_fails_if_declared_but_missing() -> None:
    with pytest.raises(RuntimeError, match="Refusing to run rollouts"):
        _sync_vllm_server_coord_offset_adapter(
            owner=SimpleNamespace(model=_TinyCoordModel(None)),
            client=_FakeClient(),
            logger=SimpleNamespace(info=lambda *args, **kwargs: None),
            dropped_modules_to_save=("coord_offset_adapter",),
            dropped_param_names=(),
        )
