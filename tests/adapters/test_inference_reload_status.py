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
