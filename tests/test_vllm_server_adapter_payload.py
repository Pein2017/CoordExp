from __future__ import annotations

import sys
import types
from collections import OrderedDict
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.tokens.row_offsets import TokenEmbeddingsAdapter
from src.infer.backend_vllm_server import (
    _build_vllm_adapter_sync_provenance,
    _filter_vllm_adapter_lora_tensors,
    _sync_vllm_server_token_embeddings_adapter,
    _vllm_adapter_peft_config,
    sync_vllm_server_adapter,
    sync_vllm_server_rollout_model_if_needed,
)


def test_vllm_adapter_payload_strips_modules_to_save_config() -> None:
    payload, dropped = _vllm_adapter_peft_config(
        SimpleNamespace(
            to_dict=lambda: {
                "r": 16,
                "target_modules": ["q_proj"],
                "modules_to_save": ["token_embeddings_adapter"],
            }
        )
    )

    assert payload["modules_to_save"] is None
    assert dropped == ("token_embeddings_adapter",)


def test_vllm_adapter_payload_keeps_only_lora_tensors() -> None:
    kept, dropped = _filter_vllm_adapter_lora_tensors(
        OrderedDict(
            [
                ("base_model.model.q_proj.lora_A.default.weight", torch.ones(1)),
                ("base_model.model.q_proj.lora_B.default.weight", torch.ones(1)),
                ("modules_to_save.default.token_ids", torch.ones(1)),
                (
                    "base_model.model.token_embeddings_adapter.modules_to_save.default.embed_offset",
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
        "modules_to_save.default.token_ids",
        "base_model.model.token_embeddings_adapter.modules_to_save.default.embed_offset",
    )


class _TinyCoordModel(nn.Module):
    def __init__(self, adapter: TokenEmbeddingsAdapter | None = None) -> None:
        super().__init__()
        if adapter is not None:
            self.token_embeddings_adapter = adapter


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


def test_vllm_adapter_token_embeddings_adapter_sync_sends_row_payload() -> None:
    adapter = TokenEmbeddingsAdapter(
        token_ids=[2, 5],
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

    _sync_vllm_server_token_embeddings_adapter(
        owner=SimpleNamespace(model=_TinyCoordModel(adapter)),
        client=client,
        logger=SimpleNamespace(info=lambda *args, **kwargs: None),
        dropped_modules_to_save=("token_embeddings_adapter",),
        dropped_param_names=("base_model.model.token_embeddings_adapter.embed_offset",),
    )

    assert len(client.calls) == 1
    call = client.calls[0]
    assert torch.equal(call["coord_ids"], torch.tensor([2, 5]))
    assert torch.allclose(call["embed_offset"], adapter.embed_offset.detach())
    assert call["head_offset"] is None
    assert call["tie_head"] is True


def test_vllm_adapter_token_embeddings_adapter_sync_fails_if_declared_but_missing() -> None:
    with pytest.raises(RuntimeError, match="Refusing to run rollouts"):
        _sync_vllm_server_token_embeddings_adapter(
            owner=SimpleNamespace(model=_TinyCoordModel(None)),
            client=_FakeClient(),
            logger=SimpleNamespace(info=lambda *args, **kwargs: None),
            dropped_modules_to_save=("token_embeddings_adapter",),
            dropped_param_names=(),
        )


def test_vllm_adapter_token_embeddings_adapter_sync_fails_with_unpatched_client() -> None:
    adapter = TokenEmbeddingsAdapter(
        token_ids=[2],
        tie_head=True,
        embed_dim=2,
        head_dim=2,
        base_dtype=torch.float32,
        device=torch.device("cpu"),
    )

    with pytest.raises(RuntimeError, match="update_token_row_offsets"):
        _sync_vllm_server_token_embeddings_adapter(
            owner=SimpleNamespace(model=_TinyCoordModel(adapter)),
            client=SimpleNamespace(),
            logger=SimpleNamespace(info=lambda *args, **kwargs: None),
            dropped_modules_to_save=("token_embeddings_adapter",),
            dropped_param_names=(),
        )


def test_vllm_adapter_sync_provenance_records_stable_lora_and_coord_digests() -> None:
    lora_params = OrderedDict(
        [
            ("base_model.model.q_proj.lora_A.default.weight", torch.ones(2, 2)),
            ("base_model.model.q_proj.lora_B.default.weight", torch.arange(4).reshape(2, 2)),
        ]
    )
    coord_ids = torch.tensor([2, 5], dtype=torch.long)
    embed_offset = torch.tensor([[0.25, -0.5], [0.75, 1.0]], dtype=torch.float32)

    provenance = _build_vllm_adapter_sync_provenance(
        sync_mode="adapter",
        lora_params=lora_params,
        vllm_peft_config={"r": 16, "target_modules": ["q_proj"]},
        dropped_modules_to_save=("token_embeddings_adapter",),
        dropped_param_names=("token_embeddings_adapter.embed_offset",),
        coord_ids=coord_ids,
        embed_offset=embed_offset,
        head_offset=None,
        tie_head=True,
        coord_row_status="requested",
        worker_verified=False,
        worker_verified_status="unavailable_fire_and_forget",
        rank_symmetric_failure=True,
    )
    same = _build_vllm_adapter_sync_provenance(
        sync_mode="adapter",
        lora_params=lora_params,
        vllm_peft_config={"target_modules": ["q_proj"], "r": 16},
        dropped_modules_to_save=("token_embeddings_adapter",),
        dropped_param_names=("token_embeddings_adapter.embed_offset",),
        coord_ids=coord_ids,
        embed_offset=embed_offset,
        head_offset=None,
        tie_head=True,
        coord_row_status="requested",
        worker_verified=False,
        worker_verified_status="unavailable_fire_and_forget",
        rank_symmetric_failure=True,
    )
    changed = _build_vllm_adapter_sync_provenance(
        sync_mode="adapter",
        lora_params=OrderedDict(
            [
                ("base_model.model.q_proj.lora_A.default.weight", torch.zeros(2, 2)),
                ("base_model.model.q_proj.lora_B.default.weight", torch.arange(4).reshape(2, 2)),
            ]
        ),
        vllm_peft_config={"r": 16, "target_modules": ["q_proj"]},
        dropped_modules_to_save=("token_embeddings_adapter",),
        dropped_param_names=("token_embeddings_adapter.embed_offset",),
        coord_ids=coord_ids,
        embed_offset=embed_offset,
        head_offset=None,
        tie_head=True,
        coord_row_status="requested",
        worker_verified=False,
        worker_verified_status="unavailable_fire_and_forget",
        rank_symmetric_failure=True,
    )

    assert provenance == same
    assert provenance["schema_version"] == "coordexp_vllm_adapter_sync_v1"
    assert provenance["sync_policy"]["mode"] == "adapter"
    assert provenance["server_identity"] == {
        "sync_schema": "coordexp_vllm_adapter_sync_v1",
        "coord_row_api": "coordexp_token_row_offsets_v1",
        "client_patch": "coordexp_vllm_client_token_row_offsets_v1",
        "worker_extension_cls": "src.infer.backend_sync.CoordExpWeightSyncWorkerExtension",
    }
    assert provenance["lora"]["tensor_count"] == 2
    assert provenance["lora"]["digest"].startswith("sha256:")
    assert provenance["coord_rows"]["status"] == "requested"
    assert provenance["coord_rows"]["digest"].startswith("sha256:")
    assert provenance["worker_verified"]["verified"] is False
    assert (
        provenance["worker_verified"]["status"]
        == "unavailable_fire_and_forget"
    )
    assert provenance["rank_symmetric_failure"] is True
    assert provenance["lora"]["digest"] != changed["lora"]["digest"]


class _TinyPeftCoordModel(nn.Module):
    def __init__(self, adapter: TokenEmbeddingsAdapter) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.token_embeddings_adapter = adapter
        self.peft_config = {
            "default": SimpleNamespace(
                to_dict=lambda: {
                    "r": 16,
                    "target_modules": ["q_proj"],
                    "modules_to_save": ["token_embeddings_adapter"],
                }
            )
        }
        self.merge_calls = 0
        self.unmerge_calls = 0

    def merge_adapter(self) -> None:
        self.merge_calls += 1

    def unmerge_adapter(self) -> None:
        self.unmerge_calls += 1


class _FakeFlattenedTensorBucket:
    def __init__(self, named_tensors: list[tuple[str, torch.Tensor]]) -> None:
        self.named_tensors = named_tensors

    def get_metadata(self) -> dict[str, object]:
        return {
            "tensor_count": len(self.named_tensors),
            "tensor_names": [name for name, _ in self.named_tensors],
        }

    def get_flattened_tensor(self) -> torch.Tensor:
        pieces = [tensor.detach().reshape(-1).cpu() for _, tensor in self.named_tensors]
        return torch.cat(pieces) if pieces else torch.empty(0)


class _RecordingAdapterSyncClient:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    def update_adapter_flattened_param(
        self,
        peft_config: dict[str, object],
        metadata: dict[str, object],
        flattened: torch.Tensor,
    ) -> None:
        self.events.append(
            (
                "adapter",
                {
                    "peft_config": dict(peft_config),
                    "metadata": dict(metadata),
                    "flattened": flattened.detach().cpu().clone(),
                },
            )
        )

    def update_token_row_offsets(
        self,
        coord_ids: torch.Tensor,
        embed_offset: torch.Tensor,
        *,
        head_offset: torch.Tensor | None = None,
        tie_head: bool = True,
    ) -> None:
        self.events.append(
            (
                "coord_rows",
                {
                    "coord_ids": coord_ids.detach().cpu().clone(),
                    "embed_offset": embed_offset.detach().cpu().clone(),
                    "head_offset": (
                        head_offset.detach().cpu().clone()
                        if torch.is_tensor(head_offset)
                        else None
                    ),
                    "tie_head": bool(tie_head),
                },
            )
        )

    def reset_prefix_cache(self) -> None:
        self.events.append(("reset_prefix_cache", {}))

    def reset_mm_cache(self) -> None:
        self.events.append(("reset_mm_cache", {}))


def _install_fake_peft_sync_modules(
    monkeypatch: pytest.MonkeyPatch,
    lora_params: "OrderedDict[str, torch.Tensor]",
) -> None:
    accelerate_mod = types.ModuleType("accelerate")
    accelerate_utils_mod = types.ModuleType("accelerate.utils")
    accelerate_utils_mod.is_peft_model = lambda model: True  # type: ignore[attr-defined]
    accelerate_mod.utils = accelerate_utils_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "accelerate", accelerate_mod)
    monkeypatch.setitem(sys.modules, "accelerate.utils", accelerate_utils_mod)

    peft_mod = types.ModuleType("peft")
    peft_utils_mod = types.ModuleType("peft.utils")
    peft_save_mod = types.ModuleType("peft.utils.save_and_load")
    peft_save_mod.get_peft_model_state_dict = (  # type: ignore[attr-defined]
        lambda model, named_state: lora_params
    )
    peft_utils_mod.save_and_load = peft_save_mod  # type: ignore[attr-defined]
    peft_mod.utils = peft_utils_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "peft", peft_mod)
    monkeypatch.setitem(sys.modules, "peft.utils", peft_utils_mod)
    monkeypatch.setitem(sys.modules, "peft.utils.save_and_load", peft_save_mod)


def test_vllm_adapter_sync_stores_backend_identity_without_reordering_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = TokenEmbeddingsAdapter(
        token_ids=[2, 5],
        tie_head=True,
        embed_dim=2,
        head_dim=2,
        base_dtype=torch.float32,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        adapter.embed_offset.copy_(
            torch.tensor([[0.25, -0.5], [0.75, 1.0]], dtype=torch.float32)
        )
    model = _TinyPeftCoordModel(adapter)
    owner = SimpleNamespace(model=model, state=SimpleNamespace(global_step=17))
    client = _RecordingAdapterSyncClient()
    lora_params = OrderedDict(
        [
            ("base_model.model.q_proj.lora_A.default.weight", torch.ones(2, 2)),
            ("base_model.model.q_proj.lora_B.default.weight", torch.arange(4).reshape(2, 2)),
            ("base_model.model.token_embeddings_adapter.embed_offset", adapter.embed_offset),
        ]
    )
    _install_fake_peft_sync_modules(monkeypatch, lora_params)
    monkeypatch.setattr(
        "src.infer.backend_vllm_server._import_swift_rollout_utils",
        lambda: SimpleNamespace(
            get_gather_if_zero3_context=lambda owner: (lambda params: nullcontext()),
            patch_lora_merge=lambda model: nullcontext(),
            patch_lora_unmerge=lambda model: nullcontext(),
            FlattenedTensorBucket=_FakeFlattenedTensorBucket,
        ),
    )

    sync_vllm_server_adapter(
        owner=owner,
        client=client,
        logger=SimpleNamespace(
            info=lambda *args, **kwargs: None,
            warning=lambda *args, **kwargs: None,
        ),
    )

    assert [event[0] for event in client.events] == [
        "adapter",
        "coord_rows",
        "reset_prefix_cache",
        "reset_mm_cache",
    ]
    assert model.merge_calls == 1
    assert model.unmerge_calls == 1
    provenance = owner._vllm_server_last_backend_sync_identity
    assert owner._vllm_server_last_sync_provenance == provenance
    assert provenance["sync_policy"]["mode"] == "adapter"
    assert provenance["sync_policy"]["frequency"] == "per_global_step"
    assert provenance["sync_policy"]["global_step"] == 17
    assert provenance["lora"]["tensor_count"] == 2
    assert provenance["lora"]["dropped_modules_to_save"] == ["token_embeddings_adapter"]
    assert provenance["coord_rows"]["status"] == "requested"
    assert provenance["requested"] == {
        "adapter_update": True,
        "coord_row_update": True,
    }
    assert provenance["worker_verified"] == {
        "verified": False,
        "status": "unavailable_fire_and_forget",
    }


def test_vllm_server_sync_failure_broadcast_aborts_nonzero_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: 1)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_backend", lambda: "gloo")
    calls = {"broadcast": 0}

    def _broadcast(tensor: torch.Tensor, src: int) -> None:
        assert src == 0
        calls["broadcast"] += 1
        tensor.fill_(1)

    def _broadcast_object_list(
        objects: list[object],
        src: int,
        device: torch.device | None = None,
    ) -> None:
        assert src == 0
        objects[0] = "RuntimeError: synthetic rank0 failure"

    monkeypatch.setattr(dist, "broadcast", _broadcast)
    monkeypatch.setattr(dist, "broadcast_object_list", _broadcast_object_list)
    owner = SimpleNamespace(
        state=SimpleNamespace(global_step=5),
        model=SimpleNamespace(device=torch.device("cpu")),
        _vllm_server_last_synced_step=-1,
        _effective_vllm_server_sync_mode=lambda: "adapter",
        _ensure_vllm_server_client=lambda: pytest.fail(
            "nonzero rank must not create the vLLM client"
        ),
    )

    with pytest.raises(RuntimeError, match="aborting all ranks.*synthetic rank0 failure"):
        sync_vllm_server_rollout_model_if_needed(owner=owner)

    assert calls["broadcast"] == 2
