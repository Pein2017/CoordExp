from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from peft.utils.save_and_load import get_peft_model_state_dict

from src.sft import (
    _append_train_arg_module_to_save,
    _install_coverage_ledger_head_for_training,
    _require_wrapped_coverage_ledger_head_for_training,
)
from src.training.bridge import TrainerLossBridge, TrainerLossBridgeSettings
from src.training.coverage_ledger.head import (
    CoverageLedgerHead,
    install_coverage_ledger_head,
)


class _PreparedToyModel(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int = 8,
        visual_dim: int | None = 6,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.backbone = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        self.config = SimpleNamespace(hidden_size=hidden_size)
        if visual_dim is not None:
            self.config.vision_config = SimpleNamespace(hidden_size=visual_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def _ledger_cfg(enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        enabled=enabled,
        ledger_projection_dim=4,
        normalize_eps=1.0e-6,
    )


def _training_config(coverage_ledger_cfg: object) -> SimpleNamespace:
    return SimpleNamespace(
        objective=SimpleNamespace(
            terms=SimpleNamespace(coverage_ledger=coverage_ledger_cfg)
        )
    )


def test_public_sidecar_builder_imports_do_not_cycle() -> None:
    script = """
from src.training.coverage_ledger import build_coverage_ledger_sidecar
from src.training.coverage_ledger.sidecar_builder import build_coverage_ledger_sidecar as direct
import src.sft
assert build_coverage_ledger_sidecar is direct
print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "ok" in result.stdout


def test_enabled_config_installs_coverage_ledger_head_on_prepared_model() -> None:
    prepared_model = _PreparedToyModel()

    head = _install_coverage_ledger_head_for_training(
        prepared_model,
        _training_config(_ledger_cfg(enabled=True)),
    )

    assert head is prepared_model.coverage_ledger_head
    assert isinstance(head, CoverageLedgerHead)
    parameters = dict(prepared_model.named_parameters())
    assert "coverage_ledger_head.state_projection.weight" in parameters
    assert "coverage_ledger_head.region_anchor_state_projection.weight" in parameters
    assert "coverage_ledger_head.object_projection.weight" in parameters
    assert head.state_projection.in_features == 8
    assert head.region_anchor_state_projection.in_features == 8
    assert head.object_projection.in_features == 6
    assert all(param.requires_grad for param in head.parameters())


def test_disabled_config_does_not_install_coverage_ledger_head() -> None:
    prepared_model = _PreparedToyModel()

    head = _install_coverage_ledger_head_for_training(
        prepared_model,
        _training_config(_ledger_cfg(enabled=False)),
    )

    assert head is None
    assert not hasattr(prepared_model, "coverage_ledger_head")


def test_coverage_ledger_head_params_update_and_round_trip_state_dict() -> None:
    model = _PreparedToyModel()
    head = install_coverage_ledger_head(model, _ledger_cfg(), visual_dim=6)
    assert head is not None
    before = {name: param.detach().clone() for name, param in head.named_parameters()}

    optimizer = torch.optim.SGD(head.parameters(), lr=0.1)
    hidden = torch.ones((2, 8), dtype=next(model.parameters()).dtype)
    visual = torch.ones((2, 6), dtype=next(model.parameters()).dtype)
    loss = (
        head.state_projection(hidden).sum()
        + head.region_anchor_state_projection(hidden).sum()
        + head.object_projection(visual).sum()
    )
    loss.backward()
    optimizer.step()

    after = dict(head.named_parameters())
    assert any(not torch.equal(before[name], after[name]) for name in before)

    saved = {name: value.detach().clone() for name, value in model.state_dict().items()}
    assert "coverage_ledger_head.state_projection.weight" in saved
    assert "coverage_ledger_head.region_anchor_state_projection.weight" in saved
    assert "coverage_ledger_head.object_projection.weight" in saved

    with torch.no_grad():
        for param in head.parameters():
            param.add_(10.0)

    model.load_state_dict(saved)

    restored = model.state_dict()
    for name, value in saved.items():
        if name.startswith("coverage_ledger_head."):
            assert torch.equal(restored[name], value)


def test_install_reuses_matching_head_and_rejects_incompatible_existing_head() -> None:
    model = _PreparedToyModel()
    installed = install_coverage_ledger_head(model, _ledger_cfg(), visual_dim=6)

    assert install_coverage_ledger_head(model, _ledger_cfg(), visual_dim=6) is installed

    with pytest.raises(ValueError, match="coverage_ledger_head.*incompatible"):
        install_coverage_ledger_head(
            model,
            SimpleNamespace(
                enabled=True,
                ledger_projection_dim=5,
                normalize_eps=1.0e-6,
            ),
            visual_dim=6,
        )


def test_install_infers_visual_dim_from_sample_embeddings_when_config_lacks_field() -> None:
    model = _PreparedToyModel(visual_dim=None)
    sample_image_embeddings = torch.zeros((1, 3, 7), dtype=torch.float32)

    head = install_coverage_ledger_head(
        model,
        _ledger_cfg(),
        sample_image_embeddings=sample_image_embeddings,
    )

    assert head is not None
    assert head.region_anchor_state_projection.in_features == 8
    assert head.object_projection.in_features == 7


def test_install_prefers_vision_out_hidden_size_for_post_merger_visual_dim() -> None:
    model = _PreparedToyModel()
    model.config.vision_config.hidden_size = 3
    model.config.vision_config.out_hidden_size = 9

    head = install_coverage_ledger_head(model, _ledger_cfg())

    assert head is not None
    assert head.region_anchor_state_projection.in_features == 8
    assert head.object_projection.in_features == 9


def test_modules_to_save_append_preserves_token_embeddings_adapter_entry() -> None:
    train_args = SimpleNamespace(
        modules_to_save=["token_embeddings_adapter"],
        training_args=SimpleNamespace(modules_to_save=["token_embeddings_adapter"]),
    )

    _append_train_arg_module_to_save(train_args, "coverage_ledger_head")
    _append_train_arg_module_to_save(train_args, "coverage_ledger_head")

    assert train_args.modules_to_save == [
        "token_embeddings_adapter",
        "coverage_ledger_head",
    ]
    assert train_args.training_args.modules_to_save == [
        "token_embeddings_adapter",
        "coverage_ledger_head",
    ]


def test_peft_adapter_state_dict_includes_coverage_ledger_head_modules_to_save() -> None:
    model = _PreparedToyModel(hidden_size=4, visual_dim=None)
    install_coverage_ledger_head(model, _ledger_cfg(), visual_dim=3)
    model.config = {"tie_word_embeddings": False, "model_type": "toy"}
    peft_model = get_peft_model(
        model,
        LoraConfig(
            target_modules=["backbone"],
            r=2,
            lora_alpha=2,
            modules_to_save=["coverage_ledger_head"],
        ),
    )

    _require_wrapped_coverage_ledger_head_for_training(peft_model, _ledger_cfg())
    adapter_state = get_peft_model_state_dict(peft_model)

    assert {
        "base_model.model.coverage_ledger_head.state_projection.weight",
        "base_model.model.coverage_ledger_head.region_anchor_state_projection.weight",
        "base_model.model.coverage_ledger_head.object_projection.weight",
    }.issubset(adapter_state)


def test_loss_bridge_resolves_and_updates_active_peft_coverage_ledger_head() -> None:
    model = _PreparedToyModel(hidden_size=4, visual_dim=None)
    install_coverage_ledger_head(model, _ledger_cfg(), visual_dim=3)
    model.config = {"tie_word_embeddings": False, "model_type": "toy"}
    peft_model = get_peft_model(
        model,
        LoraConfig(
            target_modules=["backbone"],
            r=2,
            lora_alpha=2,
            modules_to_save=["coverage_ledger_head"],
        ),
    )
    wrapper = peft_model.base_model.model.coverage_ledger_head
    active_adapter = wrapper.active_adapter
    assert isinstance(active_adapter, str)
    active_head = wrapper.modules_to_save[active_adapter]

    bridge = TrainerLossBridge(
        settings=TrainerLossBridgeSettings(coverage_ledger=_ledger_cfg())
    )
    resolved_head = bridge._require_coverage_ledger_head(peft_model)

    assert resolved_head is active_head
    assert isinstance(resolved_head, CoverageLedgerHead)

    before = active_head.state_projection.weight.detach().clone()
    optimizer = torch.optim.SGD(active_head.parameters(), lr=0.1)
    loss = (
        resolved_head.state_projection(torch.ones((1, 4))).sum()
        + resolved_head.region_anchor_state_projection(torch.ones((1, 4))).sum()
        + resolved_head.object_projection(torch.ones((1, 3))).sum()
    )
    loss.backward()
    optimizer.step()

    assert not torch.equal(before, active_head.state_projection.weight)
