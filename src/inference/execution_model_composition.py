"""Composition fidelity and behavioral diagnostics for execution models."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.common.errors import RuntimeContractError


EXECUTION_MODEL_COMPOSITION_VERSION = "coordexp-swift-execution-model-composition-v1"
EXECUTION_MODEL_COMPOSITION_NAME = "coordexp_composition_fidelity.json"
COMPOSITION_PROBE_RELATIVE_PATH = Path(
    "scripts/probes/coordexp_swift/execution_model_composition.py"
)
FULL_LOGIT_RTOL = 1e-4
FULL_LOGIT_ATOL = 5e-3
SELECTED_LOGIT_RTOL = 1e-4
SELECTED_LOGIT_ATOL = 2e-3


def compare_execution_models(
    *,
    dynamic_model: Any,
    materialized_model: Any,
    dynamic_native_inputs: Mapping[str, Any],
    materialized_native_inputs: Mapping[str, Any],
    selected_token_ids: list[int] | tuple[int, ...],
    generation_kwargs: Mapping[str, object],
    expected_merged_target_identity: Mapping[str, object] | None,
    expected_folded_selected_rows_sha256: str | None,
    compared_position_count: int = 4,
) -> dict[str, Any]:
    """Prove composition fidelity and record dynamic-HF behavior drift."""

    import torch

    dynamic_input_ids = _require_prompt_ids(
        dynamic_native_inputs,
        owner="dynamic_native_inputs",
    )
    materialized_input_ids = _require_prompt_ids(
        materialized_native_inputs,
        owner="materialized_native_inputs",
    )
    prompt_ids_equal = bool(torch.equal(dynamic_input_ids, materialized_input_ids))
    if not prompt_ids_equal:
        raise RuntimeContractError(
            "dynamic and materialized processors produced different prompt ids",
            code="inference.execution_model_composition_prompt_mismatch",
            context={
                "dynamic_shape": list(dynamic_input_ids.shape),
                "materialized_shape": list(materialized_input_ids.shape),
            },
        )
    if compared_position_count <= 0:
        _fail("compared_position_count", compared_position_count)

    selected_ids = tuple(int(item) for item in selected_token_ids)
    if not selected_ids or len(set(selected_ids)) != len(selected_ids):
        _fail("selected_token_ids", list(selected_ids))
    dynamic_tied = _model_has_tied_storage(dynamic_model)
    materialized_tied = _model_has_tied_storage(materialized_model)
    dynamic_rows = _effective_selected_rows(dynamic_model, selected_ids)
    materialized_rows = _effective_selected_rows(materialized_model, selected_ids)
    dynamic_rows_sha256 = _tensor_sha256(dynamic_rows)
    materialized_rows_sha256 = _tensor_sha256(materialized_rows)
    rows_equal = bool(torch.equal(dynamic_rows, materialized_rows))
    observed_merged_target_identity: Mapping[str, object] | None = None
    if expected_merged_target_identity is not None:
        expected_targets = _target_names(expected_merged_target_identity)
        from src.adapters.dora import inspect_merged_dora_target_weights

        observed_merged_target_identity = inspect_merged_dora_target_weights(
            materialized_model,
            expected_targets,
        )
    merged_targets_equal = (
        None
        if expected_merged_target_identity is None
        else dict(expected_merged_target_identity)
        == observed_merged_target_identity
    )

    prompt_width = int(dynamic_input_ids.shape[1])
    start = max(0, prompt_width - compared_position_count)
    positions = list(range(start, prompt_width))
    with torch.inference_mode():
        dynamic_outputs = dynamic_model(
            **dict(dynamic_native_inputs),
            return_dict=True,
            use_cache=False,
        )
        dynamic_native_logits = dynamic_outputs.logits[:, positions, :]
        dynamic_native_logits_dtype = str(dynamic_native_logits.dtype)
        dynamic_logits = dynamic_native_logits.float().cpu()
        del dynamic_outputs
        materialized_outputs = materialized_model(
            **dict(materialized_native_inputs),
            return_dict=True,
            use_cache=False,
        )
        materialized_native_logits = materialized_outputs.logits[:, positions, :]
        materialized_native_logits_dtype = str(materialized_native_logits.dtype)
        materialized_logits = materialized_native_logits.float().cpu()
        del materialized_outputs

        dynamic_generated = dynamic_model.generate(
            **dict(dynamic_native_inputs),
            **dict(generation_kwargs),
        )
        materialized_generated = materialized_model.generate(
            **dict(materialized_native_inputs),
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
    generated_ids_equal = dynamic_generated_ids == materialized_generated_ids
    return {
        "composition_checks": {
            "prompt_ids": prompt_ids_equal,
            "selected_rows_target_dtype": rows_equal,
            "dynamic_tied_weights": dynamic_tied,
            "materialized_tied_weights": materialized_tied,
            "merged_target_weights": merged_targets_equal,
        },
        "behavior_checks": {
            "greedy_generated_ids_match": generated_ids_equal,
            "full_vocab_within_reference_tolerance": full_evidence["allclose"],
            "selected_vocab_within_reference_tolerance": selected_evidence[
                "allclose"
            ],
        },
        "merged_target_weight_identity": {
            "expected": (
                None
                if expected_merged_target_identity is None
                else _target_identity_summary(expected_merged_target_identity)
            ),
            "observed": (
                None
                if observed_merged_target_identity is None
                else _target_identity_summary(observed_merged_target_identity)
            ),
        },
        "selected_row_identity": {
            "dynamic_effective_sha256": dynamic_rows_sha256,
            "materializer_folded_sha256": expected_folded_selected_rows_sha256,
            "reloaded_materialized_sha256": materialized_rows_sha256,
        },
        "full_vocab": full_evidence,
        "selected_vocab": selected_evidence,
        "compared_positions": positions,
        "full_vocab_shape": list(dynamic_logits.shape),
        "selected_vocab_shape": list(dynamic_selected.shape),
        "dtypes": {
            "dynamic_native_logits": dynamic_native_logits_dtype,
            "materialized_native_logits": materialized_native_logits_dtype,
            "comparison_logits": str(dynamic_logits.dtype),
            "dynamic_selected_rows": str(dynamic_rows.dtype),
            "materialized_selected_rows": str(materialized_rows.dtype),
        },
        "dynamic_generated_ids": dynamic_generated_ids,
        "materialized_generated_ids": materialized_generated_ids,
    }


def build_execution_model_composition_receipt(
    *,
    execution_model: Mapping[str, object],
    fixture_identity: Mapping[str, object],
    probe_identity: Mapping[str, object],
    resolved_config_identity: Mapping[str, object],
    comparison: Mapping[str, object],
) -> dict[str, Any]:
    source = _require_mapping(execution_model, "source_identity")
    materialization_identity = _materialization_identity(execution_model)
    receipt: dict[str, Any] = {
        "version": EXECUTION_MODEL_COMPOSITION_VERSION,
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
        "materialization_identity": materialization_identity,
        "fixture_identity": dict(fixture_identity),
        "probe_identity": dict(probe_identity),
        "resolved_config_identity": dict(resolved_config_identity),
        "comparison": dict(comparison),
        "thresholds": {
            "full_vocab": {"rtol": FULL_LOGIT_RTOL, "atol": FULL_LOGIT_ATOL},
            "selected_vocab": {
                "rtol": SELECTED_LOGIT_RTOL,
                "atol": SELECTED_LOGIT_ATOL,
            },
        },
    }
    _validate_comparison(
        receipt["comparison"],
        target_dtype=receipt["target_dtype"],
    )
    _validate_fixture_identity(receipt["fixture_identity"])
    _validate_probe_identity(receipt["probe_identity"])
    _validate_resolved_config_identity(receipt["resolved_config_identity"])
    _validate_owner_binding(
        receipt["comparison"],
        materialization_identity=materialization_identity,
    )
    receipt["digest"] = _digest(receipt)
    return receipt


def validate_execution_model_composition_receipt(
    receipt: Mapping[str, object],
    *,
    execution_model: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    payload = dict(receipt)
    if payload.get("version") != EXECUTION_MODEL_COMPOSITION_VERSION:
        _fail("version", payload.get("version"))
    if payload.get("status") != "passed":
        _fail("status", payload.get("status"))
    expected_digest = _digest(payload)
    if payload.get("digest") != expected_digest:
        raise RuntimeContractError(
            "execution-model composition fidelity receipt digest mismatch",
            code="inference.execution_model_composition_digest_mismatch",
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
    _validate_comparison(
        _require_mapping(payload, "comparison"),
        target_dtype=_require_string(payload, "target_dtype"),
    )
    _validate_fixture_identity(_require_mapping(payload, "fixture_identity"))
    _validate_probe_identity(_require_mapping(payload, "probe_identity"))
    _validate_resolved_config_identity(
        _require_mapping(payload, "resolved_config_identity")
    )
    if execution_model is not None:
        expected = build_execution_model_composition_linkage(execution_model)
        observed = {
            key: payload.get(key)
            for key in (
                "composition_key",
                "snapshot_fingerprint",
                "source_fingerprints",
                "target_dtype",
                "algorithm_version",
                "package_versions",
                "materialization_identity",
            )
        }
        if observed != expected:
            raise RuntimeContractError(
                "execution-model composition fidelity receipt is bound to a different composition",
                code="inference.execution_model_composition_linkage_mismatch",
                context={"expected": expected, "observed": observed},
            )
    return payload


def _validate_fixture_identity(value: Mapping[str, object]) -> None:
    row_id = value.get("row_id")
    row_index = value.get("row_index")
    max_new_tokens = value.get("max_new_tokens")
    if not isinstance(row_id, str) or not row_id:
        _fail("fixture_identity.row_id", row_id)
    if isinstance(row_index, bool) or not isinstance(row_index, int) or row_index < 0:
        _fail("fixture_identity.row_index", row_index)
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens <= 0
    ):
        _fail("fixture_identity.max_new_tokens", max_new_tokens)
    sha_fields = (
        "input_jsonl_sha256",
        "prompt_ids_sha256",
        "dynamic_executed_prompt_ids_sha256",
        "materialized_executed_prompt_ids_sha256",
        "processor_fingerprint",
        "image_file_sha256",
        "executed_media_sha256",
        "generation_fingerprint",
    )
    for field in sha_fields:
        _require_sha256(value, field)
    prompt_hashes = {
        value["prompt_ids_sha256"],
        value["dynamic_executed_prompt_ids_sha256"],
        value["materialized_executed_prompt_ids_sha256"],
    }
    if len(prompt_hashes) != 1:
        _fail("fixture_identity.prompt_ids", "mismatch")


def _validate_probe_identity(value: Mapping[str, object]) -> None:
    relative_path = value.get("path")
    if relative_path != COMPOSITION_PROBE_RELATIVE_PATH.as_posix():
        _fail("probe_identity.path", relative_path)
    expected = _require_sha256(value, "sha256")
    probe_path = Path(__file__).resolve().parents[2] / COMPOSITION_PROBE_RELATIVE_PATH
    observed = _sha256_file(probe_path)
    if observed != expected:
        raise RuntimeContractError(
            "execution-model composition probe source differs from receipt",
            code="inference.execution_model_composition_probe_drift",
            context={
                "path": str(probe_path),
                "expected": expected,
                "observed": observed,
            },
        )


def _validate_resolved_config_identity(value: Mapping[str, object]) -> None:
    _require_sha256(value, "fingerprint")
    entry_path = value.get("entry_config_path")
    if not isinstance(entry_path, str) or not entry_path:
        _fail("resolved_config_identity.entry_config_path", entry_path)
    sources = value.get("sources")
    if not isinstance(sources, list) or not sources:
        _fail("resolved_config_identity.sources", sources)
    for index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            _fail(f"resolved_config_identity.sources.{index}", source)
        path = source.get("path")
        if not isinstance(path, str) or not path:
            _fail(f"resolved_config_identity.sources.{index}.path", path)
        _require_sha256(source, "sha256")


def _sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def build_execution_model_composition_linkage(
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
        "materialization_identity": _materialization_identity(execution_model),
    }


def write_execution_model_composition_receipt(
    path: str | Path,
    receipt: Mapping[str, object],
) -> Path:
    validated = validate_execution_model_composition_receipt(receipt)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(validated, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return output


def load_execution_model_composition_receipt(path: str | Path) -> dict[str, Any]:
    receipt_path = Path(path)
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeContractError(
            "execution-model composition fidelity receipt is missing",
            code="inference.execution_model_composition_missing",
            context={"path": str(receipt_path)},
            cause=exc,
        ) from exc
    if not isinstance(payload, dict):
        _fail("receipt", type(payload).__name__)
    return validate_execution_model_composition_receipt(payload)


def _validate_comparison(
    comparison: Mapping[str, object],
    *,
    target_dtype: str,
) -> None:
    composition_checks = _require_mapping(comparison, "composition_checks")
    required_composition = {
        "prompt_ids",
        "selected_rows_target_dtype",
        "dynamic_tied_weights",
        "materialized_tied_weights",
    }
    missing = sorted(required_composition.difference(composition_checks))
    failed = sorted(
        key
        for key in required_composition
        if composition_checks.get(key) is not True
    )
    if missing or failed:
        raise RuntimeContractError(
            "execution-model composition fidelity checks did not pass",
            code="inference.execution_model_composition_exact_check",
            context={"missing": missing, "failed": failed},
        )
    if "merged_target_weights" not in composition_checks:
        _fail("comparison.composition_checks.merged_target_weights", "missing")
    merged_check = composition_checks.get("merged_target_weights")
    if merged_check not in (True, None):
        _fail("comparison.composition_checks.merged_target_weights", merged_check)
    behavior_checks = _require_mapping(comparison, "behavior_checks")
    required_behavior = {
        "greedy_generated_ids_match",
        "full_vocab_within_reference_tolerance",
        "selected_vocab_within_reference_tolerance",
    }
    missing_behavior = sorted(required_behavior.difference(behavior_checks))
    invalid_behavior = sorted(
        key
        for key in required_behavior
        if not isinstance(behavior_checks.get(key), bool)
    )
    if missing_behavior or invalid_behavior:
        _fail(
            "comparison.behavior_checks",
            {
                "missing": missing_behavior,
                "invalid": invalid_behavior,
            },
        )
    merged_identity = _require_mapping(comparison, "merged_target_weight_identity")
    expected_merged = merged_identity.get("expected")
    observed_merged = merged_identity.get("observed")
    if expected_merged != observed_merged:
        _fail("comparison.merged_target_weight_identity", "mismatch")
    if expected_merged is not None:
        if not isinstance(expected_merged, Mapping):
            _fail("comparison.merged_target_weight_identity.expected", expected_merged)
        _target_identity_summary(expected_merged)
    if (expected_merged is None) != (merged_check is None):
        _fail(
            "comparison.composition_checks.merged_target_weights",
            merged_check,
        )
    selected_rows = _require_mapping(comparison, "selected_row_identity")
    if "materializer_folded_sha256" not in selected_rows:
        _fail("comparison.selected_row_identity.materializer_folded_sha256", "missing")
    selected_hashes = {
        _require_sha256(selected_rows, field)
        for field in ("dynamic_effective_sha256", "reloaded_materialized_sha256")
    }
    folded_hash = selected_rows.get("materializer_folded_sha256")
    if folded_hash is not None:
        selected_hashes.add(_require_sha256(selected_rows, "materializer_folded_sha256"))
    if len(selected_hashes) != 1:
        _fail("comparison.selected_row_identity", "mismatch")
    for name, rtol, atol in (
        ("full_vocab", FULL_LOGIT_RTOL, FULL_LOGIT_ATOL),
        ("selected_vocab", SELECTED_LOGIT_RTOL, SELECTED_LOGIT_ATOL),
    ):
        evidence = _require_mapping(comparison, name)
        if not isinstance(evidence.get("allclose"), bool):
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
    expected_behavior = {
        "greedy_generated_ids_match": comparison.get("dynamic_generated_ids")
        == comparison.get("materialized_generated_ids"),
        "full_vocab_within_reference_tolerance": _require_mapping(
            comparison, "full_vocab"
        ).get("allclose"),
        "selected_vocab_within_reference_tolerance": _require_mapping(
            comparison, "selected_vocab"
        ).get("allclose"),
    }
    if dict(behavior_checks) != expected_behavior:
        _fail(
            "comparison.behavior_checks.consistency",
            {"expected": expected_behavior, "observed": dict(behavior_checks)},
        )
    for field in ("compared_positions", "full_vocab_shape", "selected_vocab_shape"):
        value = comparison.get(field)
        if not isinstance(value, list) or not value:
            _fail(f"comparison.{field}", value)
    dtypes = _require_mapping(comparison, "dtypes")
    native_dtype = {
        "bf16": "torch.bfloat16",
        "fp16": "torch.float16",
        "fp32": "torch.float32",
    }.get(target_dtype)
    if native_dtype is None:
        _fail("target_dtype", target_dtype)
    expected_dtypes = {
        "dynamic_native_logits": native_dtype,
        "materialized_native_logits": native_dtype,
        "comparison_logits": "torch.float32",
        "dynamic_selected_rows": native_dtype,
        "materialized_selected_rows": native_dtype,
    }
    if dict(dtypes) != expected_dtypes:
        _fail(
            "comparison.dtypes",
            {"expected": expected_dtypes, "observed": dict(dtypes)},
        )


def _validate_owner_binding(
    comparison: Mapping[str, object],
    *,
    materialization_identity: Mapping[str, object],
) -> None:
    merged = _require_mapping(comparison, "merged_target_weight_identity")
    if merged.get("expected") != materialization_identity.get(
        "merged_target_weight_identity"
    ):
        _fail("comparison.merged_target_weight_identity.owner", "mismatch")
    selected = _require_mapping(comparison, "selected_row_identity")
    if selected.get("materializer_folded_sha256") != materialization_identity.get(
        "folded_selected_rows_sha256"
    ):
        _fail("comparison.selected_row_identity.owner", "mismatch")


def _materialization_identity(
    execution_model: Mapping[str, object],
) -> dict[str, object]:
    if execution_model.get("mode") != "materialized":
        _fail("execution_model.mode", execution_model.get("mode"))
    materialization = _require_mapping(execution_model, "materialization")
    source = _require_mapping(execution_model, "source_identity")
    adapter_merge = materialization.get("adapter_merge")
    delta_fold = materialization.get("embedding_delta_fold")
    if (source.get("adapter") is None) != (adapter_merge is None):
        _fail("materialization.adapter_merge", adapter_merge)
    if (source.get("embedding_delta") is None) != (delta_fold is None):
        _fail("materialization.embedding_delta_fold", delta_fold)
    target_identity = None
    if adapter_merge is not None:
        if not isinstance(adapter_merge, Mapping):
            _fail("materialization.adapter_merge", adapter_merge)
        merge = _require_mapping(adapter_merge, "merge")
        target_identity = _target_identity_summary(
            _require_mapping(merge, "target_weight_identity")
        )
    folded_rows = None
    if delta_fold is not None:
        if not isinstance(delta_fold, Mapping):
            _fail("materialization.embedding_delta_fold", delta_fold)
        folded_rows = _require_sha256(delta_fold, "selected_rows_after_sha256")
    return {
        "merged_target_weight_identity": target_identity,
        "folded_selected_rows_sha256": folded_rows,
    }


def _model_has_tied_storage(model: Any) -> bool:
    embedding = model.get_input_embeddings()
    output = model.get_output_embeddings()
    embedding_weight = getattr(getattr(embedding, "base", embedding), "weight", None)
    output_weight = getattr(getattr(output, "base", output), "weight", None)
    return embedding_weight is not None and embedding_weight is output_weight


def _require_prompt_ids(native_inputs: Mapping[str, Any], *, owner: str) -> Any:
    import torch

    input_ids = native_inputs.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        _fail(f"{owner}.input_ids", type(input_ids).__name__)
    if input_ids.shape[0] != 1:
        _fail(f"{owner}.input_ids.shape", list(input_ids.shape))
    return input_ids


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


def _tensor_sha256(tensor: Any) -> str:
    import torch

    if not isinstance(tensor, torch.Tensor):
        _fail("tensor", type(tensor).__name__)
    payload = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


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


def _target_names(identity: Mapping[str, object]) -> tuple[str, ...]:
    targets = identity.get("targets")
    if not isinstance(targets, list) or not targets:
        _fail("expected_merged_target_identity.targets", targets)
    names: list[str] = []
    for item in targets:
        if not isinstance(item, Mapping):
            _fail("expected_merged_target_identity.targets", type(item).__name__)
        names.append(_require_string(item, "target_name"))
    if len(set(names)) != len(names):
        _fail("expected_merged_target_identity.targets", names)
    return tuple(names)


def _target_identity_summary(identity: Mapping[str, object]) -> dict[str, object]:
    target_count = identity.get("target_count")
    fingerprint = identity.get("fingerprint")
    if not isinstance(target_count, int) or target_count <= 0:
        _fail("merged_target_identity.target_count", target_count)
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        _fail("merged_target_identity.fingerprint", fingerprint)
    return {"target_count": target_count, "fingerprint": fingerprint}


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


def _require_sha256(owner: Mapping[str, object], field: str) -> str:
    value = _require_string(owner, field)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
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
        "execution-model composition fidelity receipt contains invalid evidence",
        code="inference.execution_model_composition_invalid",
        context={"field": field, "value": value},
    )


__all__ = [
    "EXECUTION_MODEL_COMPOSITION_NAME",
    "EXECUTION_MODEL_COMPOSITION_VERSION",
    "build_execution_model_composition_receipt",
    "compare_execution_models",
    "load_execution_model_composition_receipt",
    "validate_execution_model_composition_receipt",
    "write_execution_model_composition_receipt",
]
