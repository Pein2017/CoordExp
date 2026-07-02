from __future__ import annotations

from types import SimpleNamespace

import pytest

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
