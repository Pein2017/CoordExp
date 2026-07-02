from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from src.common.errors import RuntimeContractError


def test_inference_adapter_status_accepts_valid_peft_receipt() -> None:
    from src.adapters.dora import validate_inference_adapter_status

    receipt = validate_inference_adapter_status(
        load_result=SimpleNamespace(missing_keys=[], unexpected_keys=[]),
        status=SimpleNamespace(
            enabled=True,
            active_adapters=["default"],
            merged_adapters=[],
            requires_grad=False,
        ),
        expected_adapter_name="default",
    )

    assert receipt.status == "validated"
    assert receipt.adapter_name == "default"
    assert receipt.missing_keys == ()
    assert receipt.unexpected_keys == ()
    assert receipt.to_artifact_dict()["active_adapters"] == ["default"]


def test_inference_dora_adapter_loader_captures_load_result_and_status() -> None:
    from src.adapters.dora import load_inference_dora_adapter

    model = FakePeftModel()
    result = load_inference_dora_adapter(
        config=SimpleNamespace(
            adapter=SimpleNamespace(
                type="dora",
                path="/tmp/adapter",
                name="default",
            )
        ),
        qwen=SimpleNamespace(model=model, base_model_path="/tmp/base"),
    )

    assert model.loaded_path == "/tmp/adapter"
    assert model.loaded_adapter_name == "default"
    assert model.active_adapter == "default"
    assert result["status"] == "validated"
    assert result["adapter_path"] == "/tmp/adapter"
    assert result["base_model_path"] == "/tmp/base"
    assert result["load_result_available"] is True
    assert result["load_result_api"] == "peft.PeftModel.load_adapter"


def test_inference_dora_adapter_loader_accepts_transformers_mixin_with_equivalent_evidence(
    tmp_path: Path,
) -> None:
    from src.adapters.dora import load_inference_dora_adapter

    adapter_dir = _write_adapter_payload(tmp_path / "adapter", base_model="/tmp/base")
    model = FakeTransformersMixinModel(_saved_adapter_state())

    result = load_inference_dora_adapter(
        config=SimpleNamespace(
            adapter=SimpleNamespace(type="dora", path=str(adapter_dir), name="default")
        ),
        qwen=SimpleNamespace(model=model, base_model_path="/tmp/base"),
    )

    assert model.loaded_path == str(adapter_dir)
    assert model.active_adapter == "default"
    assert result["status"] == "validated"
    assert result["load_result_available"] is False
    assert result["load_result_api"] == "transformers.PeftAdapterMixin.load_adapter"
    assert result["adapter_payload_evidence"]["lora_A_count"] == 1
    assert result["adapter_payload_evidence"]["lora_B_count"] == 1
    assert result["adapter_payload_evidence"]["lora_magnitude_vector_count"] == 1
    assert result["adapter_state_evidence"]["state_checked"] is True
    assert result["adapter_state_evidence"]["normalized_saved_key_count"] == 3
    assert result["adapter_status_evidence"]["available_adapters"] == ["default"]
    assert result["adapter_status_evidence"]["num_adapter_layers"] == 1


def test_inference_dora_adapter_state_allows_transformers_materialized_prefix(
    tmp_path: Path,
) -> None:
    from src.adapters.dora import load_inference_dora_adapter

    adapter_dir = _write_adapter_payload(tmp_path / "adapter", base_model="/tmp/base")
    state = {
        key.removeprefix("base_model.model."): value
        for key, value in _saved_adapter_state().items()
    }
    model = FakeTransformersMixinModel(state)

    result = load_inference_dora_adapter(
        config=SimpleNamespace(
            adapter=SimpleNamespace(type="dora", path=str(adapter_dir), name="default")
        ),
        qwen=SimpleNamespace(model=model, base_model_path="/tmp/base"),
    )

    assert result["adapter_state_evidence"]["state_checked"] is True


def test_inference_dora_adapter_loader_rejects_transformers_mixin_missing_state(
    tmp_path: Path,
) -> None:
    from src.adapters.dora import load_inference_dora_adapter

    adapter_dir = _write_adapter_payload(tmp_path / "adapter", base_model="/tmp/base")
    model = FakeTransformersMixinModel({})

    with pytest.raises(RuntimeContractError) as exc_info:
        load_inference_dora_adapter(
            config=SimpleNamespace(
                adapter=SimpleNamespace(type="dora", path=str(adapter_dir), name="default")
            ),
            qwen=SimpleNamespace(model=model, base_model_path="/tmp/base"),
        )

    assert exc_info.value.code == "adapter.inference_state_empty"


def test_inference_dora_adapter_loader_rejects_transformers_mixin_extra_state(
    tmp_path: Path,
) -> None:
    from src.adapters.dora import load_inference_dora_adapter

    adapter_dir = _write_adapter_payload(tmp_path / "adapter", base_model="/tmp/base")
    state = _saved_adapter_state()
    state["base_model.model.layers.0.self_attn.q_proj.other_adapter.lora_A.weight"] = torch.ones(
        1, 1
    )
    model = FakeTransformersMixinModel(state)

    with pytest.raises(RuntimeContractError) as exc_info:
        load_inference_dora_adapter(
            config=SimpleNamespace(
                adapter=SimpleNamespace(type="dora", path=str(adapter_dir), name="default")
            ),
            qwen=SimpleNamespace(model=model, base_model_path="/tmp/base"),
        )

    assert exc_info.value.code == "adapter.inference_state_mismatch"
    assert exc_info.value.context["extra_materialized_keys"]


def test_inference_dora_adapter_loader_rejects_zero_adapter_layers(
    tmp_path: Path,
) -> None:
    from src.adapters.dora import load_inference_dora_adapter

    adapter_dir = _write_adapter_payload(tmp_path / "adapter", base_model="/tmp/base")
    model = FakeTransformersMixinModel(
        _saved_adapter_state(),
        status_overrides={"num_adapter_layers": 0},
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        load_inference_dora_adapter(
            config=SimpleNamespace(
                adapter=SimpleNamespace(type="dora", path=str(adapter_dir), name="default")
            ),
            qwen=SimpleNamespace(model=model, base_model_path="/tmp/base"),
        )

    assert exc_info.value.code == "adapter.inference_status_no_layers"


def test_inference_dora_adapter_loader_rejects_missing_dora_payload(
    tmp_path: Path,
) -> None:
    from src.adapters.dora import load_inference_dora_adapter

    adapter_dir = _write_adapter_payload(
        tmp_path / "adapter",
        base_model="/tmp/base",
        tensors={"base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(1, 1)},
    )
    model = FakeTransformersMixinModel(_saved_adapter_state())

    with pytest.raises(RuntimeContractError) as exc_info:
        load_inference_dora_adapter(
            config=SimpleNamespace(
                adapter=SimpleNamespace(type="dora", path=str(adapter_dir), name="default")
            ),
            qwen=SimpleNamespace(model=model, base_model_path="/tmp/base"),
        )

    assert exc_info.value.code == "adapter.inference_payload_shape"


@pytest.mark.parametrize(
    ("load_result", "status", "expected_code"),
    [
        (
            SimpleNamespace(missing_keys=["base_model.model.q.lora_A.weight"], unexpected_keys=[]),
            SimpleNamespace(
                enabled=True,
                active_adapters=["default"],
                merged_adapters=[],
                requires_grad=False,
            ),
            "adapter.inference_load_result_irregular",
        ),
        (
            SimpleNamespace(missing_keys=[], unexpected_keys=["extra.weight"]),
            SimpleNamespace(
                enabled=True,
                active_adapters=["default"],
                merged_adapters=[],
                requires_grad=False,
            ),
            "adapter.inference_load_result_irregular",
        ),
        (
            SimpleNamespace(missing_keys=[], unexpected_keys=[]),
            SimpleNamespace(
                enabled=False,
                active_adapters=["default"],
                merged_adapters=[],
                requires_grad=False,
            ),
            "adapter.inference_status_disabled",
        ),
        (
            SimpleNamespace(missing_keys=[], unexpected_keys=[]),
            SimpleNamespace(
                enabled=True,
                active_adapters=["other"],
                merged_adapters=[],
                requires_grad=False,
            ),
            "adapter.inference_active_adapter_mismatch",
        ),
        (
            SimpleNamespace(missing_keys=[], unexpected_keys=[]),
            SimpleNamespace(
                enabled=True,
                active_adapters=["default"],
                merged_adapters=["default"],
                requires_grad=False,
            ),
            "adapter.inference_merged_state",
        ),
        (
            SimpleNamespace(missing_keys=[], unexpected_keys=[]),
            SimpleNamespace(
                enabled="irregular",
                active_adapters=["default"],
                merged_adapters=[],
                requires_grad=False,
            ),
            "adapter.inference_status_irregular",
        ),
    ],
)
def test_inference_adapter_status_rejects_irregular_peft_receipts(
    load_result: SimpleNamespace,
    status: SimpleNamespace,
    expected_code: str,
) -> None:
    from src.adapters.dora import validate_inference_adapter_status

    with pytest.raises(RuntimeContractError) as exc_info:
        validate_inference_adapter_status(
            load_result=load_result,
            status=status,
            expected_adapter_name="default",
        )

    assert exc_info.value.code == expected_code


class FakePeftModel:
    def __init__(self) -> None:
        self.loaded_path: str | None = None
        self.loaded_adapter_name: str | None = None
        self.active_adapter: str | None = None

    def load_adapter(
        self,
        path: str,
        *,
        adapter_name: str,
        is_trainable: bool,
    ) -> SimpleNamespace:
        assert is_trainable is False
        self.loaded_path = path
        self.loaded_adapter_name = adapter_name
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    def set_adapter(self, adapter_name: str) -> None:
        self.active_adapter = adapter_name

    def get_model_status(self) -> SimpleNamespace:
        return SimpleNamespace(
            enabled=True,
            active_adapters=[self.active_adapter],
            merged_adapters=[],
            requires_grad=False,
        )


class FakeTransformersMixinModel:
    def __init__(
        self,
        adapter_state: dict[str, torch.Tensor],
        *,
        status_overrides: dict[str, object] | None = None,
    ) -> None:
        self.loaded_path: str | None = None
        self.loaded_adapter_name: str | None = None
        self.active_adapter: str | None = None
        self.adapter_state = adapter_state
        self.status_overrides = dict(status_overrides or {})

    def load_adapter(
        self,
        path: str,
        *,
        adapter_name: str,
        is_trainable: bool,
    ) -> None:
        assert is_trainable is False
        self.loaded_path = path
        self.loaded_adapter_name = adapter_name
        return None

    def set_adapter(self, adapter_name: str) -> None:
        self.active_adapter = adapter_name

    def get_adapter_state_dict(self, adapter_name: str) -> dict[str, torch.Tensor]:
        assert adapter_name == "default"
        return dict(self.adapter_state)

    def get_model_status(self) -> SimpleNamespace:
        payload = {
            "enabled": True,
            "active_adapters": [self.active_adapter],
            "available_adapters": ["default"],
            "merged_adapters": [],
            "num_adapter_layers": 1,
            "requires_grad": False,
        }
        payload.update(self.status_overrides)
        return SimpleNamespace(**payload)


def _write_adapter_payload(
    path: Path,
    *,
    base_model: str,
    tensors: dict[str, torch.Tensor] | None = None,
) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    config = {
        "base_model_name_or_path": base_model,
        "peft_type": "LORA",
        "use_dora": True,
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj"],
    }
    (path / "adapter_config.json").write_text(
        json.dumps(config, sort_keys=True),
        encoding="utf-8",
    )
    save_file(tensors or _saved_adapter_state(), str(path / "adapter_model.safetensors"))
    return path


def _saved_adapter_state() -> dict[str, torch.Tensor]:
    return {
        "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(1, 1),
        "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(1, 1),
        "base_model.model.layers.0.self_attn.q_proj.lora_magnitude_vector": torch.ones(1),
    }
