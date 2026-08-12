#!/usr/bin/env python3
"""Build and validate the CPU-only Wave 8 compatibility matrix.

This module never launches a model, CUDA work, cache preparation, training, or
evaluation.  It authenticates already-produced evidence and publishes a small
absent-only plan or terminal matrix receipt.
"""

from __future__ import annotations

import argparse
from collections import namedtuple
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probes.coordexp_swift import (  # noqa: E402
    wave7_determinism_preflight,
    wave7_exact_resume_postrun,
    wave7_exact_resume_sequence,
)
from src.artifacts.provenance import PINNED_RUNTIME_BASELINE_SHA256  # noqa: E402
from src.qwen.parity import (  # noqa: E402
    assert_absent_artifact_target,
    canonical_json_bytes,
    write_strict_json_atomic,
)


PLAN_SCHEMA = "coordexp-swift-wave8-compatibility-matrix-plan-v1"
RECEIPT_SCHEMA = "coordexp-swift-wave8-compatibility-matrix-receipt-v1"
PINNED_BASELINE_SHA256 = PINNED_RUNTIME_BASELINE_SHA256
MAX_JSON_BYTES = 16 * 1024 * 1024

CELL_NAMES = (
    "cache_admission",
    "packed_parity_all_layer_fa2",
    "protected_loss",
    "exact_resume",
    "native_runtime",
)
CELL_SCHEMAS = {
    name: f"coordexp-swift-wave8-{name.replace('_', '-')}-compatibility-v1"
    for name in CELL_NAMES
}
CELL_CHECKS = {
    "cache_admission": (
        "current_cache_identity_admitted",
        "admission_precedes_model_setup",
        "immutable_cache_unchanged",
    ),
    "packed_parity_all_layer_fa2": (
        "real_packed_forward_backward_compatibility",
        "all_text_layers_attested",
        "exact_boundaries_attested",
        "no_backend_fallback",
    ),
    "protected_loss": (
        "protected_loss_compatible",
        "zero_weight_diagnostic_detached",
        "no_performance_claim",
    ),
    "exact_resume": (
        "same_world_size_exact_resume_compatible",
        "passed_sequence_and_comparison_bound",
        "postrun_bound",
    ),
    "native_runtime": (
        "pinned_runtime_baseline_admitted",
        "mapped_native_runtime_attested",
        "flash_attention_2_selected",
    ),
}

NONPERFORMANCE_CLAIM = {
    "classification": "compatibility_only_nonperformance",
    "performance_promotion": False,
    "establishes": [
        "pinned_baseline_compatibility_evidence",
        "wave7_to_wave8_gate_evidence",
    ],
    "does_not_establish": [
        "throughput",
        "latency",
        "memory_or_resource_efficiency",
        "training_quality",
        "dependency_or_backend_upgrade",
        "wave9_launch_authorization",
    ],
}

InputContract = namedtuple(
    "InputContract", "schema hash_field status_field passed_status"
)
INPUT_CONTRACTS = {
    "wave7_sequence": InputContract(
        wave7_exact_resume_sequence.RECEIPT_SCHEMA,
        "receipt_payload_sha256",
        "status",
        "passed",
    ),
    "wave7_comparison": InputContract(
        wave7_exact_resume_postrun.FINAL_COMPARISON_SCHEMA,
        "receipt_payload_sha256",
        "status",
        "passed",
    ),
    "wave7_postrun": InputContract(
        wave7_exact_resume_postrun.RECEIPT_SCHEMA,
        "receipt_payload_sha256",
        "status",
        "passed",
    ),
    "native_runtime": InputContract(
        wave7_determinism_preflight.NATIVE_REFERENCE_SCHEMA,
        "receipt_sha256",
        "terminal_status",
        "passed",
    ),
    "config_identity": InputContract(
        "coordexp-swift-wave8-config-identity-v1",
        "receipt_payload_sha256",
        "status",
        "passed",
    ),
    "cache_identity": InputContract(
        "coordexp-swift-wave8-cache-identity-v1",
        "receipt_payload_sha256",
        "status",
        "passed",
    ),
    "resume_identity": InputContract(
        "coordexp-swift-wave8-resume-identity-v1",
        "receipt_payload_sha256",
        "status",
        "passed",
    ),
}


class Wave8MatrixError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "wave8_matrix.invalid",
        context: Mapping[str, Any] | None = None,
    ) -> None:
        self.code = code
        self.context = dict(context or {})
        super().__init__(message)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise Wave8MatrixError(
            "artifact is unreadable",
            code="wave8_matrix.input_unreadable",
            context={"path": str(path), "error_type": type(exc).__name__},
        ) from exc
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def load_strict_json_bytes(encoded: bytes, *, owner: str) -> Any:
    """Load bounded strict JSON, rejecting duplicate keys and non-finite values."""

    if len(encoded) > MAX_JSON_BYTES:
        raise Wave8MatrixError(
            f"{owner} exceeds the strict JSON byte bound",
            code="wave8_matrix.json_oversize",
        )

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate key: {key}")
            value[key] = item
        return value

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite constant: {value}")

    try:
        result = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
        canonical_json_bytes(result)
    except (UnicodeDecodeError, TypeError, ValueError) as exc:
        raise Wave8MatrixError(
            f"{owner} is not strict JSON",
            code="wave8_matrix.strict_json",
            context={"owner": owner, "error_type": type(exc).__name__},
        ) from exc
    return result


def _regular_file(path: Path, *, owner: str) -> Path:
    requested = Path(os.path.abspath(path.expanduser()))
    try:
        info = requested.lstat()
    except OSError as exc:
        raise Wave8MatrixError(
            f"{owner} is missing",
            code="wave8_matrix.input_missing",
            context={"path": str(requested)},
        ) from exc
    if requested.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise Wave8MatrixError(
            f"{owner} must be one regular non-symlink file",
            code="wave8_matrix.input_topology",
            context={"path": str(requested)},
        )
    resolved = requested.resolve(strict=True)
    if resolved != requested:
        raise Wave8MatrixError(
            f"{owner} must not traverse a symlink",
            code="wave8_matrix.input_topology",
            context={"path": str(requested), "resolved": str(resolved)},
        )
    return resolved


def _load_json_file(path: Path, *, owner: str) -> dict[str, Any]:
    source = _regular_file(path, owner=owner)
    try:
        value = load_strict_json_bytes(source.read_bytes(), owner=owner)
    except OSError as exc:
        raise Wave8MatrixError(
            f"{owner} is unreadable",
            code="wave8_matrix.input_unreadable",
        ) from exc
    if not isinstance(value, dict):
        raise Wave8MatrixError(
            f"{owner} must be one JSON object",
            code="wave8_matrix.json_schema",
        )
    return value


def _require_canonical_file(
    path: Path, payload: Mapping[str, Any], *, owner: str
) -> None:
    expected = canonical_json_bytes(dict(payload)) + b"\n"
    try:
        observed = path.read_bytes()
    except OSError as exc:
        raise Wave8MatrixError(
            f"{owner} is unreadable",
            code="wave8_matrix.input_unreadable",
        ) from exc
    if observed != expected:
        raise Wave8MatrixError(
            f"{owner} is not canonical JSON bytes",
            code="wave8_matrix.noncanonical",
        )


def _exact_fields(value: Mapping[str, Any], expected: set[str], *, owner: str) -> None:
    observed = set(value)
    if observed != expected:
        raise Wave8MatrixError(
            f"{owner} fields are not exact",
            code="wave8_matrix.json_schema",
            context={
                "owner": owner,
                "missing": sorted(expected - observed),
                "unexpected": sorted(observed - expected),
            },
        )


def _authenticate(
    path: Path,
    *,
    schema: str,
    hash_field: str,
    status_field: str,
    passed_status: str,
    owner: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    source = _regular_file(path, owner=owner)
    payload = _load_json_file(source, owner=owner)
    observed_digest = payload.get(hash_field)
    unsigned = dict(payload)
    unsigned.pop(hash_field, None)
    recomputed = _sha256_bytes(canonical_json_bytes(unsigned))
    if (
        not _is_sha256(observed_digest)
        or observed_digest != recomputed
        or payload.get("schema") != schema
        or payload.get(status_field) != passed_status
    ):
        raise Wave8MatrixError(
            f"{owner} signature, schema, or status is invalid",
            code="wave8_matrix.input_authentication",
            context={
                "path": str(source),
                "expected_schema": schema,
                "observed_schema": payload.get("schema"),
                "observed_status": payload.get(status_field),
            },
        )
    return payload, {
        "path": str(source),
        "file_sha256": _sha256_file(source),
        "payload_sha256": recomputed,
        "schema": schema,
    }


def _input_paths(inputs: Mapping[str, Mapping[str, Any]]) -> dict[str, Path]:
    if set(inputs) != set(INPUT_CONTRACTS):
        raise Wave8MatrixError(
            "Wave 8 inputs are not exact",
            code="wave8_matrix.input_inventory",
            context={
                "missing": sorted(set(INPUT_CONTRACTS) - set(inputs)),
                "unexpected": sorted(set(inputs) - set(INPUT_CONTRACTS)),
            },
        )
    result: dict[str, Path] = {}
    for name, value in inputs.items():
        if not isinstance(value, Mapping) or set(value) != {"path", "schema"}:
            raise Wave8MatrixError(
                "input request fields are not exact",
                code="wave8_matrix.input_inventory",
                context={"input": name},
            )
        contract = INPUT_CONTRACTS[name]
        if value["schema"] != contract.schema:
            raise Wave8MatrixError(
                "input request schema differs from the pinned contract",
                code="wave8_matrix.input_schema",
                context={"input": name},
            )
        result[name] = Path(os.path.abspath(Path(str(value["path"])).expanduser()))
    return result


def _authenticate_inputs(paths: Mapping[str, Path]) -> dict[str, dict[str, str]]:
    bindings: dict[str, dict[str, str]] = {}
    for name, contract in INPUT_CONTRACTS.items():
        owner = name.replace("_", " ")
        if name.startswith("wave7_"):
            owner = f"Wave 7 {name.removeprefix('wave7_').replace('_', ' ')}"
        _, binding = _authenticate(
            paths[name],
            schema=contract.schema,
            hash_field=contract.hash_field,
            status_field=contract.status_field,
            passed_status=contract.passed_status,
            owner=owner,
        )
        bindings[name] = binding
    return bindings


def _validate_wave7_evidence(
    *,
    wave7_root: Path,
    sequence_path: Path,
    comparison_path: Path,
    postrun_path: Path,
) -> dict[str, Any]:
    """Reuse the Wave 7 validators; do not duplicate exact-resume semantics."""

    root = wave7_root.expanduser().resolve(strict=True)
    sequence_payload = wave7_exact_resume_sequence._strict_json_file(
        sequence_path, owner="Wave 7 sequence receipt"
    )
    signed_sequence = dict(sequence_payload)
    sequence_digest = signed_sequence.pop("receipt_payload_sha256", None)
    if not _is_sha256(sequence_digest):
        raise Wave8MatrixError(
            "Wave 7 sequence signature is missing",
            code="wave8_matrix.wave7_sequence",
        )
    wave7_exact_resume_sequence._verify_signed_payload(
        sequence_payload,
        expected_payload_sha256=sequence_digest,
        owner="Wave 7 sequence receipt",
    )
    wave7_exact_resume_sequence._validate_terminal_payload(signed_sequence)
    if sequence_payload.get("status") != "passed":
        raise Wave8MatrixError(
            "Wave 7 sequence did not pass",
            code="wave8_matrix.wave7_sequence",
        )
    admitted = wave7_exact_resume_postrun.authenticate_inputs(
        r5_root=root,
        sequence_receipt=sequence_path,
        final_comparison=comparison_path,
    )
    postrun_payload, _ = wave7_exact_resume_postrun._authenticate_receipt_snapshot(
        postrun_path,
        schema=wave7_exact_resume_postrun.RECEIPT_SCHEMA,
        status="passed",
        owner="Wave 7 postrun receipt",
    )
    sequence_input = postrun_payload.get("sequence_receipt")
    comparison_input = postrun_payload.get("final_comparison")
    if not wave7_exact_resume_postrun._binding_matches(
        sequence_input,
        path=sequence_path.resolve(),
        payload=sequence_payload,
    ):
        raise Wave8MatrixError(
            "Wave 7 postrun does not bind the current sequence",
            code="wave8_matrix.wave7_postrun",
        )
    comparison_payload = wave7_exact_resume_postrun._authenticate_receipt(
        comparison_path,
        schema=wave7_exact_resume_postrun.FINAL_COMPARISON_SCHEMA,
        status="passed",
        owner="Wave 7 comparison receipt",
    )
    if not wave7_exact_resume_postrun._binding_matches(
        comparison_input,
        path=comparison_path.resolve(),
        payload=comparison_payload,
    ):
        raise Wave8MatrixError(
            "Wave 7 postrun does not bind the current comparison",
            code="wave8_matrix.wave7_postrun",
        )
    return {
        "status": "passed",
        "sequence_schema": sequence_payload["schema"],
        "comparison_schema": comparison_payload["schema"],
        "postrun_schema": postrun_payload["schema"],
        "checkpoint_dir": str(admitted["checkpoint_dir"]),
    }


def _validate_native_runtime_evidence(binding: Mapping[str, Any]) -> dict[str, Any]:
    """Reuse the mapped-native receipt validator from the Wave 7 preflight."""

    wave7_determinism_preflight._validate_native_reference_binding(binding)
    return {
        "status": "passed",
        "schema": binding["schema"],
        "baseline_sha256": PINNED_BASELINE_SHA256,
    }


def _wave7_gate(
    *, wave7_root: Path, paths: Mapping[str, Path], bindings: Mapping[str, Any]
) -> dict[str, Any]:
    sequence_contract = INPUT_CONTRACTS["wave7_sequence"]
    sequence = _load_json_file(paths["wave7_sequence"], owner="Wave 7 sequence")
    if (
        sequence.get("schema") != sequence_contract.schema
        or sequence.get(sequence_contract.status_field)
        != sequence_contract.passed_status
    ):
        raise Wave8MatrixError(
            "Wave 7 sequence must be current and passed",
            code="wave8_matrix.wave7_gate",
        )
    result = _validate_wave7_evidence(
        wave7_root=wave7_root,
        sequence_path=paths["wave7_sequence"],
        comparison_path=paths["wave7_comparison"],
        postrun_path=paths["wave7_postrun"],
    )
    native = _validate_native_runtime_evidence(bindings["native_runtime"])
    return {"wave7": result, "native_runtime": native}


def _target_map(cell_targets: Mapping[str, Path]) -> dict[str, dict[str, str]]:
    if set(cell_targets) != set(CELL_NAMES):
        raise Wave8MatrixError(
            "matrix cell targets are not exact",
            code="wave8_matrix.cell_inventory",
        )
    result: dict[str, dict[str, str]] = {}
    seen: set[Path] = set()
    for name in CELL_NAMES:
        target = Path(cell_targets[name]).expanduser().resolve(strict=False)
        if target in seen:
            raise Wave8MatrixError(
                "matrix cell targets must be distinct",
                code="wave8_matrix.cell_topology",
            )
        seen.add(target)
        result[name] = {"path": str(target), "schema": CELL_SCHEMAS[name]}
    return result


def _signed(payload: Mapping[str, Any], *, hash_field: str) -> dict[str, Any]:
    unsigned = dict(payload)
    unsigned.pop(hash_field, None)
    return {
        **unsigned,
        hash_field: _sha256_bytes(canonical_json_bytes(unsigned)),
    }


def _publish_absent(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        target = assert_absent_artifact_target(path)
        write_strict_json_atomic(target, payload)
    except Exception as exc:
        if isinstance(exc, Wave8MatrixError):
            raise
        raise Wave8MatrixError(
            "artifact target already exists or cannot be published absent-only",
            code="wave8_matrix.publication",
            context={"path": str(path), "error_type": type(exc).__name__},
        ) from exc


def prepare_plan(
    *,
    plan_path: Path,
    receipt_path: Path,
    wave7_root: Path,
    inputs: Mapping[str, Mapping[str, Any]],
    cell_targets: Mapping[str, Path],
) -> dict[str, Any]:
    """Authenticate current inputs and publish one immutable matrix plan."""

    plan_target = Path(plan_path).expanduser().resolve(strict=False)
    receipt_target = Path(receipt_path).expanduser().resolve(strict=False)
    if plan_target == receipt_target:
        raise Wave8MatrixError(
            "plan and receipt targets must be distinct",
            code="wave8_matrix.target_topology",
        )
    try:
        assert_absent_artifact_target(plan_target)
        assert_absent_artifact_target(receipt_target)
    except Exception as exc:
        raise Wave8MatrixError(
            "plan or receipt target already exists",
            code="wave8_matrix.publication",
        ) from exc
    paths = _input_paths(inputs)
    bindings = _authenticate_inputs(paths)
    gate = _wave7_gate(wave7_root=Path(wave7_root), paths=paths, bindings=bindings)
    runtime_baseline = {
        "schema_version": 3,
        "baseline_sha256": PINNED_BASELINE_SHA256,
        "attention_backend": "flash_attention_2",
    }
    identity_bundle_sha256 = _sha256_bytes(
        canonical_json_bytes(
            {
                "runtime_baseline": runtime_baseline,
                "native_runtime": bindings["native_runtime"],
                "config_identity": bindings["config_identity"],
                "cache_identity": bindings["cache_identity"],
                "resume_identity": bindings["resume_identity"],
            }
        )
    )
    body = {
        "schema": PLAN_SCHEMA,
        "status": "prepared",
        "route": ["wave7", "wave8", "wave9"],
        "wave7_root": str(Path(wave7_root).expanduser().resolve(strict=True)),
        "inputs": bindings,
        "runtime_baseline": runtime_baseline,
        "identity_bundle_sha256": identity_bundle_sha256,
        "wave7_gate": gate,
        "cells": _target_map(cell_targets),
        "receipt_target": str(receipt_target),
        "claim_scope": NONPERFORMANCE_CLAIM,
    }
    plan = _signed(body, hash_field="plan_payload_sha256")
    _publish_absent(plan_target, plan)
    persisted = _load_json_file(plan_target, owner="published matrix plan")
    if persisted != plan:
        raise Wave8MatrixError(
            "published matrix plan differs after reload",
            code="wave8_matrix.publication_reload",
        )
    return plan


_PLAN_FIELDS = {
    "schema",
    "status",
    "route",
    "wave7_root",
    "inputs",
    "runtime_baseline",
    "identity_bundle_sha256",
    "wave7_gate",
    "cells",
    "receipt_target",
    "claim_scope",
    "plan_payload_sha256",
}


def _load_plan(path: Path, *, validate_live: bool) -> dict[str, Any]:
    plan_path = _regular_file(path, owner="matrix plan")
    plan = _load_json_file(plan_path, owner="matrix plan")
    _require_canonical_file(plan_path, plan, owner="matrix plan")
    _exact_fields(plan, _PLAN_FIELDS, owner="matrix plan")
    observed = plan.get("plan_payload_sha256")
    unsigned = dict(plan)
    unsigned.pop("plan_payload_sha256", None)
    recomputed = _sha256_bytes(canonical_json_bytes(unsigned))
    if (
        observed != recomputed
        or not _is_sha256(observed)
        or plan["schema"] != PLAN_SCHEMA
        or plan["status"] != "prepared"
        or plan["route"] != ["wave7", "wave8", "wave9"]
        or plan["claim_scope"] != NONPERFORMANCE_CLAIM
    ):
        raise Wave8MatrixError(
            "matrix plan signature or fixed contract is invalid",
            code="wave8_matrix.plan_authentication",
        )
    expected_runtime = {
        "schema_version": 3,
        "baseline_sha256": PINNED_BASELINE_SHA256,
        "attention_backend": "flash_attention_2",
    }
    if plan["runtime_baseline"] != expected_runtime:
        raise Wave8MatrixError(
            "matrix plan baseline drifted",
            code="wave8_matrix.baseline_drift",
        )
    cells = plan["cells"]
    if not isinstance(cells, dict) or set(cells) != set(CELL_NAMES):
        raise Wave8MatrixError(
            "matrix plan cell inventory drifted",
            code="wave8_matrix.cell_inventory",
        )
    for name in CELL_NAMES:
        if cells[name] != {
            "path": str(Path(cells[name].get("path", "")).resolve(strict=False)),
            "schema": CELL_SCHEMAS[name],
        }:
            raise Wave8MatrixError(
                "matrix cell target contract drifted",
                code="wave8_matrix.cell_inventory",
                context={"cell": name},
            )
    inputs = plan["inputs"]
    if not isinstance(inputs, dict) or set(inputs) != set(INPUT_CONTRACTS):
        raise Wave8MatrixError(
            "matrix input bindings drifted",
            code="wave8_matrix.input_inventory",
        )
    paths: dict[str, Path] = {}
    for name, binding in inputs.items():
        if not isinstance(binding, dict):
            raise Wave8MatrixError(
                "matrix input binding is malformed",
                code="wave8_matrix.input_binding",
            )
        _exact_fields(
            binding,
            {"path", "file_sha256", "payload_sha256", "schema"},
            owner=f"{name} binding",
        )
        if (
            binding["schema"] != INPUT_CONTRACTS[name].schema
            or not _is_sha256(binding["file_sha256"])
            or not _is_sha256(binding["payload_sha256"])
        ):
            raise Wave8MatrixError(
                "matrix input binding identity is malformed",
                code="wave8_matrix.input_binding",
            )
        paths[name] = Path(binding["path"])
    identity_bundle_sha256 = _sha256_bytes(
        canonical_json_bytes(
            {
                "runtime_baseline": expected_runtime,
                "native_runtime": inputs["native_runtime"],
                "config_identity": inputs["config_identity"],
                "cache_identity": inputs["cache_identity"],
                "resume_identity": inputs["resume_identity"],
            }
        )
    )
    if plan["identity_bundle_sha256"] != identity_bundle_sha256:
        raise Wave8MatrixError(
            "matrix identity bundle drifted",
            code="wave8_matrix.identity_bundle",
        )
    if validate_live:
        observed_bindings = _authenticate_inputs(paths)
        if observed_bindings != inputs:
            raise Wave8MatrixError(
                "matrix input bytes drifted",
                code="wave8_matrix.input_drift",
            )
        gate = _wave7_gate(
            wave7_root=Path(plan["wave7_root"]),
            paths=paths,
            bindings=observed_bindings,
        )
        if gate != plan["wave7_gate"]:
            raise Wave8MatrixError(
                "matrix Wave 7 validation projection drifted",
                code="wave8_matrix.wave7_gate",
            )
    return plan


def validate_plan(path: Path) -> dict[str, Any]:
    return _load_plan(Path(path), validate_live=True)


def _validate_cell(plan: Mapping[str, Any], name: str) -> dict[str, Any]:
    spec = plan["cells"][name]
    payload, binding = _authenticate(
        Path(spec["path"]),
        schema=CELL_SCHEMAS[name],
        hash_field="receipt_payload_sha256",
        status_field="status",
        passed_status="passed",
        owner=f"matrix cell {name}",
    )
    _exact_fields(
        payload,
        {
            "schema",
            "status",
            "cell",
            "plan_payload_sha256",
            "baseline_sha256",
            "identity_bundle_sha256",
            "checks",
            "claim_scope",
            "receipt_payload_sha256",
        },
        owner=f"matrix cell {name}",
    )
    expected_checks = {check: True for check in CELL_CHECKS[name]}
    if (
        payload["cell"] != name
        or payload["plan_payload_sha256"] != plan["plan_payload_sha256"]
        or payload["baseline_sha256"] != PINNED_BASELINE_SHA256
        or payload["identity_bundle_sha256"] != plan["identity_bundle_sha256"]
        or payload["checks"] != expected_checks
        or payload["claim_scope"] != NONPERFORMANCE_CLAIM
    ):
        raise Wave8MatrixError(
            "matrix cell did not satisfy its exact compatibility contract",
            code="wave8_matrix.cell_contract",
            context={"cell": name},
        )
    return {"cell": name, "status": "passed", "binding": binding}


def _error_row(name: str, status: str, exc: Exception | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {"cell": name, "status": status, "binding": None}
    if exc is not None:
        row["error_code"] = str(getattr(exc, "code", type(exc).__name__))
    return row


def aggregate(plan_path: Path) -> dict[str, Any]:
    """Publish a passed, blocked, or failed receipt without executing cells."""

    source = Path(plan_path).expanduser().resolve(strict=True)
    plan = _load_plan(source, validate_live=False)
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    plan_error: Exception | None = None
    try:
        _load_plan(source, validate_live=True)
    except Exception as exc:
        plan_error = exc
        failures.append(str(getattr(exc, "code", type(exc).__name__)))
    if plan_error is not None:
        rows = [_error_row(name, "not_evaluated") for name in CELL_NAMES]
        status = "failed"
    else:
        for name in CELL_NAMES:
            path = Path(plan["cells"][name]["path"])
            if not path.exists() and not path.is_symlink():
                rows.append(_error_row(name, "missing"))
                continue
            try:
                rows.append(_validate_cell(plan, name))
            except Exception as exc:
                rows.append(_error_row(name, "failed", exc))
                failures.append(str(getattr(exc, "code", type(exc).__name__)))
        if failures:
            status = "failed"
        elif any(row["status"] == "missing" for row in rows):
            status = "blocked"
        else:
            status = "passed"
    body = {
        "schema": RECEIPT_SCHEMA,
        "status": status,
        "passed": status == "passed",
        "plan": {
            "path": str(source),
            "file_sha256": _sha256_file(source),
            "payload_sha256": plan["plan_payload_sha256"],
            "schema": PLAN_SCHEMA,
        },
        "runtime_baseline": plan["runtime_baseline"],
        "input_bindings": plan["inputs"],
        "identity_bundle_sha256": plan["identity_bundle_sha256"],
        "cells": rows,
        "failures": sorted(set(failures)),
        "claim_scope": NONPERFORMANCE_CLAIM,
    }
    receipt = _signed(body, hash_field="receipt_payload_sha256")
    _publish_absent(Path(plan["receipt_target"]), receipt)
    return receipt


_RECEIPT_FIELDS = {
    "schema",
    "status",
    "passed",
    "plan",
    "runtime_baseline",
    "input_bindings",
    "identity_bundle_sha256",
    "cells",
    "failures",
    "claim_scope",
    "receipt_payload_sha256",
}


def validate_receipt(path: Path) -> dict[str, Any]:
    receipt_path = _regular_file(Path(path), owner="matrix receipt")
    receipt = _load_json_file(receipt_path, owner="matrix receipt")
    _require_canonical_file(receipt_path, receipt, owner="matrix receipt")
    _exact_fields(receipt, _RECEIPT_FIELDS, owner="matrix receipt")
    observed = receipt.get("receipt_payload_sha256")
    unsigned = dict(receipt)
    unsigned.pop("receipt_payload_sha256", None)
    recomputed = _sha256_bytes(canonical_json_bytes(unsigned))
    if observed != recomputed or not _is_sha256(observed):
        raise Wave8MatrixError(
            "matrix receipt signature is invalid",
            code="wave8_matrix.receipt_authentication",
        )
    if (
        receipt["schema"] != RECEIPT_SCHEMA
        or receipt["status"] not in {"passed", "blocked", "failed"}
        or receipt["passed"] is not (receipt["status"] == "passed")
        or receipt["claim_scope"] != NONPERFORMANCE_CLAIM
    ):
        raise Wave8MatrixError(
            "matrix receipt terminal contract is invalid",
            code="wave8_matrix.receipt_contract",
        )
    plan_binding = receipt["plan"]
    if not isinstance(plan_binding, dict):
        raise Wave8MatrixError(
            "matrix receipt plan binding is malformed",
            code="wave8_matrix.receipt_contract",
        )
    _exact_fields(
        plan_binding,
        {"path", "file_sha256", "payload_sha256", "schema"},
        owner="matrix receipt plan binding",
    )
    plan_path = Path(plan_binding["path"])
    plan = _load_plan(
        plan_path, validate_live=receipt.get("status") in {"passed", "blocked"}
    )
    if plan_binding != {
        "path": str(plan_path.resolve()),
        "file_sha256": _sha256_file(plan_path.resolve()),
        "payload_sha256": plan["plan_payload_sha256"],
        "schema": PLAN_SCHEMA,
    }:
        raise Wave8MatrixError(
            "matrix receipt plan binding drifted",
            code="wave8_matrix.receipt_plan",
        )
    if (
        receipt["runtime_baseline"] != plan["runtime_baseline"]
        or receipt["input_bindings"] != plan["inputs"]
        or receipt["identity_bundle_sha256"] != plan["identity_bundle_sha256"]
        or not isinstance(receipt["cells"], list)
        or len(receipt["cells"]) != len(CELL_NAMES)
        or [row.get("cell") for row in receipt["cells"]] != list(CELL_NAMES)
    ):
        raise Wave8MatrixError(
            "matrix receipt bindings or cell inventory drifted",
            code="wave8_matrix.receipt_contract",
        )
    allowed_statuses = {
        "passed": {"passed"},
        "blocked": {"passed", "missing"},
        "failed": {"passed", "missing", "failed", "not_evaluated"},
    }
    for row in receipt["cells"]:
        if (
            not isinstance(row, dict)
            or row.get("status") not in allowed_statuses[receipt["status"]]
        ):
            raise Wave8MatrixError(
                "matrix receipt cell state is incoherent",
                code="wave8_matrix.receipt_contract",
            )
        if row["status"] == "passed":
            if set(row) != {"cell", "status", "binding"} or not isinstance(
                row["binding"], dict
            ):
                raise Wave8MatrixError(
                    "passed matrix cell row is malformed",
                    code="wave8_matrix.receipt_contract",
                )
        elif row["status"] == "failed":
            if set(row) != {"cell", "status", "binding", "error_code"}:
                raise Wave8MatrixError(
                    "failed matrix cell row is malformed",
                    code="wave8_matrix.receipt_contract",
                )
        elif set(row) != {"cell", "status", "binding"} or row["binding"] is not None:
            raise Wave8MatrixError(
                "non-passed matrix cell row is malformed",
                code="wave8_matrix.receipt_contract",
            )
    if receipt["status"] == "passed":
        expected_rows = [_validate_cell(plan, name) for name in CELL_NAMES]
        if receipt["cells"] != expected_rows or receipt["failures"] != []:
            raise Wave8MatrixError(
                "passed matrix receipt lacks five live passed cells",
                code="wave8_matrix.receipt_pass",
            )
    elif receipt["status"] == "blocked":
        if receipt["failures"] or not any(
            row.get("status") == "missing" for row in receipt["cells"]
        ):
            raise Wave8MatrixError(
                "blocked matrix receipt has no missing cell",
                code="wave8_matrix.receipt_blocked",
            )
    elif not receipt["failures"]:
        raise Wave8MatrixError(
            "failed matrix receipt has no failure code",
            code="wave8_matrix.receipt_failed",
        )
    return receipt


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="publish an authenticated matrix plan")
    plan.add_argument("--plan", type=Path, required=True)
    plan.add_argument("--receipt", type=Path, required=True)
    plan.add_argument("--wave7-root", type=Path, required=True)
    for name in INPUT_CONTRACTS:
        plan.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    for name in CELL_NAMES:
        plan.add_argument(f"--cell-{name.replace('_', '-')}", type=Path, required=True)
    aggregate_parser = commands.add_parser(
        "aggregate", help="publish a terminal receipt from existing cells"
    )
    aggregate_parser.add_argument("--plan", type=Path, required=True)
    validate = commands.add_parser("validate", help="validate one plan or receipt")
    validate.add_argument("--kind", choices=("plan", "receipt"), required=True)
    validate.add_argument("--artifact", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "plan":
            inputs = {
                name: {
                    "path": getattr(args, name),
                    "schema": contract.schema,
                }
                for name, contract in INPUT_CONTRACTS.items()
            }
            cells = {name: getattr(args, f"cell_{name}") for name in CELL_NAMES}
            result = prepare_plan(
                plan_path=args.plan,
                receipt_path=args.receipt,
                wave7_root=args.wave7_root,
                inputs=inputs,
                cell_targets=cells,
            )
        elif args.command == "aggregate":
            result = aggregate(args.plan)
        elif args.kind == "plan":
            result = validate_plan(args.artifact)
        else:
            result = validate_receipt(args.artifact)
    except Exception as exc:
        error = {
            "status": "failed",
            "error_code": str(getattr(exc, "code", type(exc).__name__)),
            "error_type": type(exc).__name__,
        }
        sys.stderr.buffer.write(canonical_json_bytes(error) + b"\n")
        return 2
    sys.stdout.buffer.write(canonical_json_bytes(result) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
