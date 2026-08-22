"""Fail-closed validation of admitted BF16 vLLM qualification receipts."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, NoReturn, cast

from src.common.errors import RuntimeContractError
from src.inference.backend import BackendLaunch
from src.inference.vllm_qualification_producer import (
    CONTRACT_VERSION,
    DEFAULT_ADMISSION_ROOT,
    EXPECTED_RECEIPT_FILENAMES,
    build_launch_contract,
    build_qualification_identity,
    validate_receipt_set,
)


RUNTIME_QUALIFICATION_VERSION = CONTRACT_VERSION
CONCURRENCY_QUALIFICATION_VERSION = CONTRACT_VERSION
QUALIFIED_MAX_NUM_SEQS = frozenset((1, 4))
QUALIFICATION_RECEIPT = DEFAULT_ADMISSION_ROOT / EXPECTED_RECEIPT_FILENAMES["runtime"]
CONCURRENCY_QUALIFICATION_RECEIPT = (
    DEFAULT_ADMISSION_ROOT / EXPECTED_RECEIPT_FILENAMES["concurrency"]
)
FORCED_REPLAY_SINGLE_QUALIFICATION_RECEIPT = (
    DEFAULT_ADMISSION_ROOT / EXPECTED_RECEIPT_FILENAMES["forced_replay"]
)


def validate_admitted_receipt_set(
    *,
    identity: Mapping[str, object],
    receipt_root: str | Path = DEFAULT_ADMISSION_ROOT,
) -> dict[str, dict[str, object]]:
    return validate_receipt_set(
        receipt_root=receipt_root,
        identity=identity,
        require_artifacts=False,
    )


def validate_vllm_runtime_qualification(
    *,
    launch: BackendLaunch,
    engine_kwargs: Mapping[str, object],
    receipt_root: str | Path = DEFAULT_ADMISSION_ROOT,
    identity_builder: Callable[[Path], Mapping[str, object]] = build_qualification_identity,
    receipt_path: str | Path | None = None,
    concurrency_receipt_path: str | Path | None = None,
) -> dict[str, object]:
    """Bind one production launch to the complete admitted receipt set."""

    _validate_launch_shape(launch=launch, engine_kwargs=engine_kwargs)
    root = _resolve_receipt_root(
        receipt_root=receipt_root,
        receipt_path=receipt_path,
        secondary_receipt_path=concurrency_receipt_path,
    )
    candidate_identity = _read_candidate_identity(root)
    current_identity = dict(
        identity_builder(_identity_config_path(candidate_identity))
    )
    receipts = validate_admitted_receipt_set(
        identity=current_identity,
        receipt_root=root,
    )
    _validate_launch_identity(launch=launch, identity=current_identity)
    _validate_launch_config(
        launch=launch,
        engine_kwargs=engine_kwargs,
        identity=current_identity,
    )
    mode = "runtime" if launch.batch_size == 1 else "concurrency"
    selected = receipts[mode]
    candidate_version = _vllm_version(current_identity)
    source_identity = cast(Mapping[str, object], current_identity["source"])
    config_identity = cast(Mapping[str, object], current_identity["config"])
    model_identity = cast(Mapping[str, object], current_identity["model"])
    return {
        "status": "passed",
        "candidate_version": candidate_version,
        "model_dtype": "bf16",
        "max_num_seqs": launch.batch_size,
        "receipt_path": str(root / EXPECTED_RECEIPT_FILENAMES[mode]),
        "receipt_digest": selected["digest"],
        "qualification_set": {
            kind: receipt["digest"] for kind, receipt in receipts.items()
        },
        "source_fingerprint": source_identity["fingerprint"],
        "config_fingerprint": config_identity["fingerprint"],
        "model_snapshot_fingerprint": model_identity["snapshot_fingerprint"],
    }


def validate_vllm_forced_replay_qualification(
    *,
    launch: BackendLaunch,
    processor_identity: Mapping[str, object],
    receipt_root: str | Path = DEFAULT_ADMISSION_ROOT,
    identity_builder: Callable[[Path], Mapping[str, object]] = build_qualification_identity,
    receipt_path: str | Path | None = None,
    baseline_receipt_path: str | Path | None = None,
) -> dict[str, object]:
    """Bind forced replay to the admitted processor and BF16 model identity."""

    if launch.batch_size != 1:
        raise RuntimeContractError(
            "forced replay is qualified only at max_num_seqs=1",
            code="vllm_backend.raw_replay_qualification_max_num_seqs",
            context={"observed": launch.batch_size, "qualified": [1]},
        )
    root = _resolve_receipt_root(
        receipt_root=receipt_root,
        receipt_path=receipt_path,
        secondary_receipt_path=baseline_receipt_path,
    )
    candidate_identity = _read_candidate_identity(root)
    current_identity = dict(
        identity_builder(_identity_config_path(candidate_identity))
    )
    receipts = validate_admitted_receipt_set(
        identity=current_identity,
        receipt_root=root,
    )
    _validate_launch_identity(launch=launch, identity=current_identity)
    receipt = receipts["forced_replay"]
    evidence = receipt["evidence"]
    if not isinstance(evidence, Mapping):
        _fail("forced-replay evidence is malformed", field="evidence")
    expected_processor_sha = evidence.get("processor_source_sha256")
    if processor_identity.get("source_sha256") != expected_processor_sha:
        raise RuntimeContractError(
            "forced-replay processor differs from admitted qualification",
            code="vllm_backend.raw_replay_qualification_processor_drift",
            context={
                "expected": expected_processor_sha,
                "observed": processor_identity.get("source_sha256"),
            },
        )
    return {
        "status": "passed",
        "candidate_version": _vllm_version(current_identity),
        "model_dtype": "bf16",
        "qualified_request_count": evidence["request_count"],
        "processor_source_sha256": expected_processor_sha,
        "receipt_path": str(
            root / EXPECTED_RECEIPT_FILENAMES["forced_replay"]
        ),
        "receipt_digest": receipt["digest"],
    }


def _validate_launch_shape(
    *,
    launch: BackendLaunch,
    engine_kwargs: Mapping[str, object],
) -> None:
    if launch.backend != "vllm":
        raise RuntimeContractError(
            "vLLM qualification received a non-vLLM launch",
            code="vllm_backend.launch_backend",
            context={"backend": launch.backend},
        )
    if launch.model_dtype != "bf16":
        raise RuntimeContractError(
            "vLLM qualification is BF16-only",
            code="vllm_backend.qualification_dtype",
            context={"observed": launch.model_dtype, "qualified": ["bf16"]},
        )
    if launch.batch_size not in QUALIFIED_MAX_NUM_SEQS:
        raise RuntimeContractError(
            "vLLM batch concurrency has no admitted qualification",
            code="vllm_backend.qualification_max_num_seqs",
            context={
                "observed": launch.batch_size,
                "qualified": sorted(QUALIFIED_MAX_NUM_SEQS),
            },
        )
    if engine_kwargs.get("max_num_seqs") != launch.batch_size:
        raise RuntimeContractError(
            "vLLM engine max_num_seqs differs from the launch",
            code="vllm_backend.qualification_engine_argument",
            context={
                "field": "max_num_seqs",
                "expected": launch.batch_size,
                "observed": engine_kwargs.get("max_num_seqs"),
            },
        )


def _validate_launch_identity(
    *,
    launch: BackendLaunch,
    identity: Mapping[str, object],
) -> None:
    execution = launch.execution_model_identity
    if not isinstance(execution, Mapping):
        _fail("launch lacks execution-model identity", field="execution_model_identity")
    model = identity.get("model")
    if not isinstance(model, Mapping):
        _fail("qualification model identity is malformed", field="identity.model")
    source = execution.get("source_identity")
    observed = {
        "mode": execution.get("mode"),
        "composition_key": execution.get("composition_key"),
        "snapshot_fingerprint": execution.get("snapshot_fingerprint"),
        "receipt_fingerprint": execution.get("receipt_fingerprint"),
        "source_fingerprints": {
            "base": _fingerprint(source, "base"),
            "adapter": _fingerprint(source, "adapter"),
            "embedding_delta": _fingerprint(source, "embedding_delta"),
        },
        "target_dtype": execution.get("target_dtype", launch.model_dtype),
    }
    if observed != model:
        raise RuntimeContractError(
            "execution model differs from admitted qualification",
            code="vllm_backend.qualification_model_identity",
            context={"expected": dict(model), "observed": observed},
        )
    composition = execution.get("composition_fidelity")
    receipt = composition.get("receipt") if isinstance(composition, Mapping) else None
    resolved_config = (
        receipt.get("resolved_config_identity")
        if isinstance(receipt, Mapping)
        else None
    )
    config = identity.get("config")
    expected_config = config.get("fingerprint") if isinstance(config, Mapping) else None
    if isinstance(composition, Mapping) and (
        not isinstance(resolved_config, Mapping)
        or resolved_config.get("fingerprint") != expected_config
    ):
        raise RuntimeContractError(
            "execution-model composition is bound to a different config",
            code="vllm_backend.qualification_config_identity",
            context={
                "expected": expected_config,
                "observed": (
                    resolved_config.get("fingerprint")
                    if isinstance(resolved_config, Mapping)
                    else None
                ),
            },
        )


def _validate_launch_config(
    *,
    launch: BackendLaunch,
    engine_kwargs: Mapping[str, object],
    identity: Mapping[str, object],
) -> None:
    config = identity.get("config")
    contracts = config.get("launch_contracts") if isinstance(config, Mapping) else None
    expected = (
        contracts.get(f"seq{launch.batch_size}")
        if isinstance(contracts, Mapping)
        else None
    )
    observed = build_launch_contract(launch=launch, engine_kwargs=engine_kwargs)
    if expected != observed:
        raise RuntimeContractError(
            "vLLM launch/config differs from admitted qualification",
            code="vllm_backend.qualification_config_identity",
            context={
                "mode": f"seq{launch.batch_size}",
                "expected_fingerprint": (
                    expected.get("fingerprint")
                    if isinstance(expected, Mapping)
                    else None
                ),
                "observed_fingerprint": observed["fingerprint"],
            },
        )


def _read_candidate_identity(root: Path) -> dict[str, Any]:
    path = root / EXPECTED_RECEIPT_FILENAMES["runtime"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeContractError(
            "vLLM qualification receipt is unavailable",
            code="vllm_backend.qualification_receipt",
            context={"path": str(path)},
            cause=exc,
        ) from exc
    identity = payload.get("identity") if isinstance(payload, Mapping) else None
    if not isinstance(identity, Mapping):
        _fail("receipt lacks qualification identity", field="identity")
    return dict(identity)


def _identity_config_path(identity: Mapping[str, object]) -> Path:
    config = identity.get("config")
    value = config.get("entry_path") if isinstance(config, Mapping) else None
    if not isinstance(value, str) or not value:
        _fail("receipt lacks config entry path", field="identity.config.entry_path")
    return Path(value).expanduser().resolve()


def _resolve_receipt_root(
    *,
    receipt_root: str | Path,
    receipt_path: str | Path | None,
    secondary_receipt_path: str | Path | None,
) -> Path:
    explicit = [
        Path(path).expanduser().resolve()
        for path in (receipt_path, secondary_receipt_path)
        if path is not None
    ]
    if explicit:
        parents = {path.parent for path in explicit}
        if len(parents) != 1:
            _fail(
                "explicit qualification receipts have different roots",
                field="receipt_root",
            )
        return next(iter(parents))
    return Path(receipt_root).expanduser().resolve()


def _fingerprint(source: object, name: str) -> object:
    value = source.get(name) if isinstance(source, Mapping) else None
    return value.get("fingerprint") if isinstance(value, Mapping) else None


def _vllm_version(identity: Mapping[str, object]) -> str:
    runtime = identity.get("runtime")
    packages = runtime.get("packages") if isinstance(runtime, Mapping) else None
    value = packages.get("vllm") if isinstance(packages, Mapping) else None
    if not isinstance(value, str) or not value:
        _fail("runtime identity lacks vLLM version", field="runtime.packages.vllm")
    return value


def _fail(message: str, *, field: str) -> NoReturn:
    raise RuntimeContractError(
        message,
        code="vllm_backend.qualification_receipt",
        context={"field": field},
    )


__all__ = [
    "CONCURRENCY_QUALIFICATION_RECEIPT",
    "CONCURRENCY_QUALIFICATION_VERSION",
    "FORCED_REPLAY_SINGLE_QUALIFICATION_RECEIPT",
    "QUALIFICATION_RECEIPT",
    "QUALIFIED_MAX_NUM_SEQS",
    "RUNTIME_QUALIFICATION_VERSION",
    "validate_admitted_receipt_set",
    "validate_vllm_forced_replay_qualification",
    "validate_vllm_runtime_qualification",
]
