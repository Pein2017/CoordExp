"""Cryptographically bound parity evidence for materialized execution models."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.common.errors import RuntimeContractError


EXECUTION_MODEL_PARITY_VERSION = "coordexp-swift-execution-model-parity-v1"
EXECUTION_MODEL_PARITY_NAME = "coordexp_materialization_parity.json"
FULL_LOGIT_RTOL = 1e-4
FULL_LOGIT_ATOL = 5e-3
SELECTED_LOGIT_RTOL = 1e-4
SELECTED_LOGIT_ATOL = 2e-3


def compare_execution_models(
    *,
    dynamic_model: Any,
    materialized_model: Any,
    native_inputs: Mapping[str, Any],
    selected_token_ids: list[int] | tuple[int, ...],
    generation_kwargs: Mapping[str, object],
    compared_position_count: int = 4,
) -> dict[str, Any]:
    """Execute the numeric and exact checks required by the parity contract."""

    import torch

    input_ids = native_inputs.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        _fail("native_inputs.input_ids", type(input_ids).__name__)
    if input_ids.shape[0] != 1:
        _fail("native_inputs.input_ids.shape", list(input_ids.shape))
    if compared_position_count <= 0:
        _fail("compared_position_count", compared_position_count)

    selected_ids = tuple(int(item) for item in selected_token_ids)
    if not selected_ids or len(set(selected_ids)) != len(selected_ids):
        _fail("selected_token_ids", list(selected_ids))
    dynamic_tied = _model_has_tied_storage(dynamic_model)
    materialized_tied = _model_has_tied_storage(materialized_model)
    dynamic_rows = _effective_selected_rows(dynamic_model, selected_ids)
    materialized_rows = _effective_selected_rows(materialized_model, selected_ids)
    rows_equal = bool(torch.equal(dynamic_rows, materialized_rows))

    prompt_width = int(input_ids.shape[1])
    start = max(0, prompt_width - compared_position_count)
    positions = list(range(start, prompt_width))
    with torch.inference_mode():
        dynamic_outputs = dynamic_model(
            **dict(native_inputs),
            return_dict=True,
            use_cache=False,
        )
        dynamic_logits = dynamic_outputs.logits[:, positions, :].float().cpu()
        del dynamic_outputs
        materialized_outputs = materialized_model(
            **dict(native_inputs),
            return_dict=True,
            use_cache=False,
        )
        materialized_logits = materialized_outputs.logits[:, positions, :].float().cpu()
        del materialized_outputs

        dynamic_generated = dynamic_model.generate(
            **dict(native_inputs),
            **dict(generation_kwargs),
        )
        materialized_generated = materialized_model.generate(
            **dict(native_inputs),
            **dict(generation_kwargs),
        )
    dynamic_generated_ids = [
        int(item) for item in dynamic_generated[0, prompt_width:].detach().cpu().tolist()
    ]
    materialized_generated_ids = [
        int(item)
        for item in materialized_generated[0, prompt_width:].detach().cpu().tolist()
    ]
    selected_index = torch.tensor(selected_ids, dtype=torch.long)
    dynamic_selected = dynamic_logits.index_select(-1, selected_index)
    materialized_selected = materialized_logits.index_select(-1, selected_index)
    full_evidence = _numeric_evidence(
        dynamic_logits,
        materialized_logits,
        rtol=FULL_LOGIT_RTOL,
        atol=FULL_LOGIT_ATOL,
    )
    selected_evidence = _numeric_evidence(
        dynamic_selected,
        materialized_selected,
        rtol=SELECTED_LOGIT_RTOL,
        atol=SELECTED_LOGIT_ATOL,
    )
    return {
        "exact_checks": {
            "prompt_ids": True,
            "greedy_generated_ids": dynamic_generated_ids
            == materialized_generated_ids,
            "selected_rows_target_dtype": rows_equal,
            "dynamic_tied_weights": dynamic_tied,
            "materialized_tied_weights": materialized_tied,
        },
        "full_vocab": full_evidence,
        "selected_vocab": selected_evidence,
        "compared_positions": positions,
        "full_vocab_shape": list(dynamic_logits.shape),
        "selected_vocab_shape": list(dynamic_selected.shape),
        "dtypes": {
            "dynamic_logits": str(dynamic_logits.dtype),
            "materialized_logits": str(materialized_logits.dtype),
            "dynamic_selected_rows": str(dynamic_rows.dtype),
            "materialized_selected_rows": str(materialized_rows.dtype),
        },
        "dynamic_generated_ids": dynamic_generated_ids,
        "materialized_generated_ids": materialized_generated_ids,
    }


def build_execution_model_parity_receipt(
    *,
    execution_model: Mapping[str, object],
    fixture_identity: Mapping[str, object],
    comparison: Mapping[str, object],
) -> dict[str, Any]:
    source = _require_mapping(execution_model, "source_identity")
    receipt: dict[str, Any] = {
        "version": EXECUTION_MODEL_PARITY_VERSION,
        "status": "passed",
        "composition_key": _require_string(execution_model, "composition_key"),
        "snapshot_fingerprint": _require_string(
            execution_model,
            "snapshot_fingerprint",
        ),
        "source_fingerprints": {
            "base": _require_string(_require_mapping(source, "base"), "fingerprint"),
            "adapter": _optional_fingerprint(source.get("adapter")),
            "embedding_delta": _optional_fingerprint(
                source.get("embedding_delta")
            ),
        },
        "target_dtype": _require_string(execution_model, "target_dtype"),
        "algorithm_version": _require_string(
            execution_model,
            "algorithm_version",
        ),
        "package_versions": dict(
            _require_mapping(execution_model, "package_versions")
        ),
        "fixture_identity": dict(fixture_identity),
        "comparison": dict(comparison),
        "thresholds": {
            "full_vocab": {"rtol": FULL_LOGIT_RTOL, "atol": FULL_LOGIT_ATOL},
            "selected_vocab": {
                "rtol": SELECTED_LOGIT_RTOL,
                "atol": SELECTED_LOGIT_ATOL,
            },
        },
    }
    _validate_comparison(receipt["comparison"])
    receipt["digest"] = _digest(receipt)
    return receipt


def validate_execution_model_parity_receipt(
    receipt: Mapping[str, object],
    *,
    execution_model: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    payload = dict(receipt)
    if payload.get("version") != EXECUTION_MODEL_PARITY_VERSION:
        _fail("version", payload.get("version"))
    if payload.get("status") != "passed":
        _fail("status", payload.get("status"))
    expected_digest = _digest(payload)
    if payload.get("digest") != expected_digest:
        raise RuntimeContractError(
            "execution-model parity receipt digest mismatch",
            code="inference.execution_model_parity_digest_mismatch",
            context={
                "expected_digest": expected_digest,
                "actual_digest": payload.get("digest"),
            },
        )
    thresholds = _require_mapping(payload, "thresholds")
    if thresholds != {
        "full_vocab": {"rtol": FULL_LOGIT_RTOL, "atol": FULL_LOGIT_ATOL},
        "selected_vocab": {
            "rtol": SELECTED_LOGIT_RTOL,
            "atol": SELECTED_LOGIT_ATOL,
        },
    }:
        _fail("thresholds", thresholds)
    _validate_comparison(_require_mapping(payload, "comparison"))
    if execution_model is not None:
        expected = build_execution_model_parity_linkage(execution_model)
        observed = {
            key: payload.get(key)
            for key in (
                "composition_key",
                "snapshot_fingerprint",
                "source_fingerprints",
                "target_dtype",
                "algorithm_version",
                "package_versions",
            )
        }
        if observed != expected:
            raise RuntimeContractError(
                "execution-model parity receipt is bound to a different composition",
                code="inference.execution_model_parity_linkage_mismatch",
                context={"expected": expected, "observed": observed},
            )
    return payload


def build_execution_model_parity_linkage(
    execution_model: Mapping[str, object],
) -> dict[str, object]:
    source = _require_mapping(execution_model, "source_identity")
    return {
        "composition_key": _require_string(execution_model, "composition_key"),
        "snapshot_fingerprint": _require_string(
            execution_model,
            "snapshot_fingerprint",
        ),
        "source_fingerprints": {
            "base": _require_string(_require_mapping(source, "base"), "fingerprint"),
            "adapter": _optional_fingerprint(source.get("adapter")),
            "embedding_delta": _optional_fingerprint(
                source.get("embedding_delta")
            ),
        },
        "target_dtype": _require_string(execution_model, "target_dtype"),
        "algorithm_version": _require_string(
            execution_model,
            "algorithm_version",
        ),
        "package_versions": dict(
            _require_mapping(execution_model, "package_versions")
        ),
    }


def write_execution_model_parity_receipt(
    path: str | Path,
    receipt: Mapping[str, object],
) -> Path:
    validated = validate_execution_model_parity_receipt(receipt)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(validated, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return output


def load_execution_model_parity_receipt(path: str | Path) -> dict[str, Any]:
    receipt_path = Path(path)
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeContractError(
            "execution-model parity receipt is missing",
            code="inference.execution_model_parity_missing",
            context={"path": str(receipt_path)},
            cause=exc,
        ) from exc
    if not isinstance(payload, dict):
        _fail("receipt", type(payload).__name__)
    return validate_execution_model_parity_receipt(payload)


def _validate_comparison(comparison: Mapping[str, object]) -> None:
    exact_checks = _require_mapping(comparison, "exact_checks")
    required_exact = {
        "prompt_ids",
        "greedy_generated_ids",
        "selected_rows_target_dtype",
        "dynamic_tied_weights",
        "materialized_tied_weights",
    }
    missing = sorted(required_exact.difference(exact_checks))
    failed = sorted(
        key for key in required_exact if exact_checks.get(key) is not True
    )
    if missing or failed:
        raise RuntimeContractError(
            "execution-model parity exact checks did not pass",
            code="inference.execution_model_parity_exact_check",
            context={"missing": missing, "failed": failed},
        )
    for name, rtol, atol in (
        ("full_vocab", FULL_LOGIT_RTOL, FULL_LOGIT_ATOL),
        ("selected_vocab", SELECTED_LOGIT_RTOL, SELECTED_LOGIT_ATOL),
    ):
        evidence = _require_mapping(comparison, name)
        if evidence.get("allclose") is not True:
            _fail(f"comparison.{name}.allclose", evidence.get("allclose"))
        observed_rtol = float(evidence.get("rtol", float("nan")))
        observed_atol = float(evidence.get("atol", float("nan")))
        if observed_rtol != rtol or observed_atol != atol:
            _fail(
                f"comparison.{name}.thresholds",
                {"rtol": observed_rtol, "atol": observed_atol},
            )
        for field in ("max_abs_diff", "max_rel_diff"):
            value = float(evidence.get(field, float("nan")))
            if value < 0.0 or value != value or value in (float("inf"), float("-inf")):
                _fail(f"comparison.{name}.{field}", value)
    for field in ("compared_positions", "full_vocab_shape", "selected_vocab_shape"):
        value = comparison.get(field)
        if not isinstance(value, list) or not value:
            _fail(f"comparison.{field}", value)
    dtypes = _require_mapping(comparison, "dtypes")
    if not dtypes:
        _fail("comparison.dtypes", dtypes)


def _model_has_tied_storage(model: Any) -> bool:
    embedding = model.get_input_embeddings()
    output = model.get_output_embeddings()
    embedding_weight = getattr(getattr(embedding, "base", embedding), "weight", None)
    output_weight = getattr(getattr(output, "base", output), "weight", None)
    return embedding_weight is not None and embedding_weight is output_weight


def _effective_selected_rows(model: Any, selected_ids: tuple[int, ...]) -> Any:
    import torch

    embedding = model.get_input_embeddings()
    base = getattr(embedding, "base", embedding)
    weight = getattr(base, "weight", None)
    if not isinstance(weight, torch.Tensor):
        _fail("model.input_embedding.weight", type(weight).__name__)
    index = torch.tensor(selected_ids, dtype=torch.long, device=weight.device)
    rows = weight.index_select(0, index)
    delta = getattr(embedding, "shared_embed_delta", None)
    if delta is not None:
        selection = getattr(embedding, "selection", None)
        if selection is None or tuple(selection.token_ids) != selected_ids:
            _fail("model.selected_token_order", selected_ids)
        rows = rows + delta.to(device=rows.device, dtype=rows.dtype)
    return rows.detach().cpu()


def _numeric_evidence(left: Any, right: Any, *, rtol: float, atol: float) -> dict[str, object]:
    import torch

    if left.shape != right.shape:
        _fail("comparison.shape", {"left": list(left.shape), "right": list(right.shape)})
    difference = torch.abs(left - right)
    denominator = torch.maximum(torch.abs(right), torch.full_like(right, 1e-12))
    relative = difference / denominator
    return {
        "allclose": bool(torch.allclose(left, right, rtol=rtol, atol=atol)),
        "rtol": rtol,
        "atol": atol,
        "max_abs_diff": float(difference.max().item()),
        "max_rel_diff": float(relative.max().item()),
    }


def _optional_fingerprint(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        _fail("source_identity", type(value).__name__)
    return _require_string(value, "fingerprint")


def _require_mapping(owner: Mapping[str, object], field: str) -> Mapping[str, object]:
    value = owner.get(field)
    if not isinstance(value, Mapping):
        _fail(field, type(value).__name__)
    return value


def _require_string(owner: Mapping[str, object], field: str) -> str:
    value = owner.get(field)
    if not isinstance(value, str) or not value:
        _fail(field, value)
    return value


def _digest(receipt: Mapping[str, object]) -> str:
    payload = {key: value for key, value in receipt.items() if key != "digest"}
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fail(field: str, value: object) -> None:
    raise RuntimeContractError(
        "execution-model parity receipt contains invalid evidence",
        code="inference.execution_model_parity_invalid",
        context={"field": field, "value": value},
    )


__all__ = [
    "EXECUTION_MODEL_PARITY_NAME",
    "EXECUTION_MODEL_PARITY_VERSION",
    "build_execution_model_parity_receipt",
    "compare_execution_models",
    "load_execution_model_parity_receipt",
    "validate_execution_model_parity_receipt",
    "write_execution_model_parity_receipt",
]
