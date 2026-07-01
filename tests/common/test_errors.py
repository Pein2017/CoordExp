from __future__ import annotations

from src.common.errors import (
    ArtifactContractError,
    ConfigContractError,
    CoordExpError,
    DataContractError,
    EncodingContractError,
    LossContractError,
    PackingContractError,
    QwenForwardContractError,
    RuntimeContractError,
    TemplateContractError,
)


def test_contract_error_formats_code_message_and_context() -> None:
    err = ConfigContractError(
        "unknown field",
        code="config.unknown_field",
        context={"field": "extra", "path": "config.yaml"},
    )

    assert isinstance(err, CoordExpError)
    assert err.code == "config.unknown_field"
    assert err.message == "unknown field"
    assert err.context == {"field": "extra", "path": "config.yaml"}
    assert str(err) == (
        "ConfigContractError[config.unknown_field]: unknown field | "
        'context: {"field": "extra", "path": "config.yaml"}'
    )


def test_contract_error_copies_context_and_preserves_cause() -> None:
    context = {"example_id": "ex-1"}
    cause = ValueError("raw failure")

    err = RuntimeContractError(
        "backend rejected setup",
        code="runtime.backend_setup",
        context=context,
        cause=cause,
    )
    context["example_id"] = "mutated"

    assert err.context == {"example_id": "ex-1"}
    assert err.cause is cause
    assert err.__cause__ is cause


def test_approved_contract_error_names_are_available() -> None:
    error_types = [
        ConfigContractError,
        DataContractError,
        TemplateContractError,
        EncodingContractError,
        PackingContractError,
        QwenForwardContractError,
        LossContractError,
        RuntimeContractError,
        ArtifactContractError,
    ]

    assert all(issubclass(error_type, CoordExpError) for error_type in error_types)
