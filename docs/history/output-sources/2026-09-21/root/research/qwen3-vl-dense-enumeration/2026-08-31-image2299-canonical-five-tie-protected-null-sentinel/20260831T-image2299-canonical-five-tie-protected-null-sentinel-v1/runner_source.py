#!/usr/bin/env python3
"""Test the canonical five-tie suffix with one protected-null child residual."""

from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
from safetensors.torch import load_file, save_file
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research import run_image2299_dyadic_norm_release_distillation as parent
from scripts.research import run_image2299_protected_null_output_distillation as protected_null


recursive = parent.recursive
base = parent.base
full_root = parent.full_root
token_ids_sha256 = parent.token_ids_sha256

SCHEMA_VERSION = "image2299.canonical_five_tie_protected_null_sentinel.v1"
UNIT_ID = "2026-08-31-image2299-canonical-five-tie-protected-null-sentinel"
OUTPUT_ROOT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration") / UNIT_ID
PARENT_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-30-image2299-dyadic-norm-release-distillation/"
    "20260830T-image2299-dyadic-norm-release-distillation-v1"
)
PARENT_RECEIPT = PARENT_ROOT / "receipt.json"
PARENT_RECEIPT_SHA256 = "9811bdb632c5c564607537fb80892c38b66a584b31a09da97fa887b6fc90a5e6"
PARENT_RUNNER_SHA256 = "8820277e8e4698599fca30e1e740790fd6cf8216345faaac9f2da9e4a9134c5c"
PARENT_PAYLOAD = PARENT_ROOT / "dyadic_norm_release_output_residual.safetensors"
PARENT_PAYLOAD_SHA256 = "10455e0587bfd103bf3c539cdc655f21b1ee778a02ef27be5f80a6e9f8c34704"
PARENT_RESIDUAL_ROWS_SHA256 = "f85c0022306c426d5cc752d9ac383ea313c1af6fc737f42bee058b8891b1b1ec"
PARENT_ROUTE_SHA256 = "c89c6f9900303227cf781caf4a39dc203a09a53b0d319930754cda184529530e"
PARENT_ROUTE_LENGTH = 370
PARENT_NORMALIZED_NORM = 1.0972734315870298
TARGET_PATH = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-26-image2299-full-root-detached-margin/target-root/target-root-v2/target.json"
)
TARGET_SHA256 = "22c24c53af14f8a0969bb09efe3150d63bdca19ca3c85baac046450d62576988"
START_CHECKPOINT = parent.START_CHECKPOINT
PROMPT_TOKEN_SHA256 = parent.PROMPT_TOKEN_SHA256
IMAGE_SHA256 = parent.IMAGE_SHA256
START_SURFACE_SHA256 = parent.START_SURFACE_SHA256
FROZEN_SURFACE_SHA256 = parent.FROZEN_SURFACE_SHA256

EOS = 151645
ROW_TOKENS = 9
CANONICAL_OWNER_ORDER = (
    "gt:2299:45",
    "gt:2299:11",
    "gt:2299:9",
    "gt:2299:8",
    "gt:2299:43",
)
CANONICAL_ROWS = {
    "gt:2299:45": [151646, 48731, 151647, 151648, 152019, 152427, 152031, 152471, 151649],
    "gt:2299:11": [151646, 48731, 151647, 151648, 152042, 151871, 152054, 151904, 151649],
    "gt:2299:9": [151646, 48731, 151647, 151648, 152132, 151860, 152145, 151902, 151649],
    "gt:2299:8": [151646, 48731, 151647, 151648, 152229, 151826, 152256, 151872, 151649],
    "gt:2299:43": [151646, 48731, 151647, 151648, 152245, 152418, 152272, 152483, 151649],
}
CANONICAL_ROW_SHA256 = {
    "gt:2299:45": "74cfa577c711a0e6ff27bdb5e9ad24a20b765c27248fca6d92ff484cdbf311c7",
    "gt:2299:11": "269bc215ee885b2cf705fbb20cf3621dd77afb62e1160f7210740ce4d8a43067",
    "gt:2299:9": "4a86b343b1a8ac35d2180e1911f2ccfb0cf5b1e7ff3798b38efd80e61c979434",
    "gt:2299:8": "6cedfa02aa66bb88369b15cf125e6e1cd0d81a8125fbf1b5d4a9463390bdc691",
    "gt:2299:43": "264a1f5cd4c0af8745bcbe54ba8f3f71672f78e92dca7c2db8bf767591331829",
}
CANDIDATE_ROUTE_SHA256 = "9491c9c4027cdf050caacc0d41d261418bbdc21d3676227f53fb7c4ccc0c4572"
CANDIDATE_ROUTE_LENGTH = 415
ALL_OWNER_IDS = tuple(f"gt:2299:{index}" for index in range(46))
PARENT_OWNER_IDS = tuple(owner for owner in ALL_OWNER_IDS if owner not in CANONICAL_OWNER_ORDER)
PARENT_SELECTED_TOKEN_IDS = [151645, 151646, 151820, 151867, 151935, 152032, 152190, 152242, 152305]

MARGIN = 0.01
NORM_CAP_NUMERATOR = 9
NORM_CAP_DENOMINATOR = 8
NORM_CAP = NORM_CAP_NUMERATOR / NORM_CAP_DENOMINATOR
PROJECTED_NORM_FLOOR = 1.0e-3
PROTECTED_NULL_TOLERANCE = 1.0e-10
MAX_POSITIVES = 46
MAX_RANK = 46
MAX_CHILD_ROWS = 26
MAX_CONSTRAINTS = 1_196
MAX_VARIABLES = 1_196
MAX_SOLVES = 1
MAX_WARM_CANDIDATES = 1
RESOURCE_BOUND = {
    "gpu_count": 1,
    "world_size": 1,
    "positive_states_max": MAX_POSITIVES,
    "positive_rank_max": MAX_RANK,
    "selected_child_rows_max": MAX_CHILD_ROWS,
    "constraint_count_max": MAX_CONSTRAINTS,
    "variable_count_max": MAX_VARIABLES,
    "solve_count_max": MAX_SOLVES,
    "warm_candidate_count_max": MAX_WARM_CANDIDATES,
    "model_load_count_max": 2,
    "wall_time_seconds_max": 1_800,
    "max_peak_cuda_reserved_bytes": 16 * 2**30,
    "output_artifact_bytes_max": 200_000_000,
}
TERMINAL_STATUSES = {
    "cold_greedy_46_owner_success",
    "static_route_hold",
    "nullspace_infeasible",
    "certified_infeasible",
    "numerical_solver_hold",
    "norm_cap_exceeded",
    "runtime_margin_hold",
    "canonical_greedy_negative",
    "technical_hold",
}

ProtectedNullHold = parent.ProtectedNullHold
SparseOutputResidual = parent.SparseOutputResidual
_hold = parent._hold
_sha256 = parent._sha256
_value_sha256 = parent._value_sha256
_tensor_sha256 = parent._tensor_sha256
_atomic_json = parent._atomic_json
_capture_route = parent._capture_route
_greedy_route = parent._greedy_route
_evaluate_route = parent._evaluate_route
_protected_hidden_matrix = protected_null._protected_hidden_matrix
_row_basis = protected_null._row_basis
_row_norms = parent._row_norms
_collapsed_rows = parent._collapsed_rows
_full_vocab_violation = parent._full_vocab_violation
_residual_identity = parent._residual_identity
_surface_snapshot = parent._surface_snapshot
_artifact_bytes = parent._artifact_bytes
_require_one_gpu = parent._require_one_gpu
_target_only_constraints = parent._target_only_constraints
_compact_constraints = parent._compact_constraints
_solve_target_only_minimum_normalized = parent._solve_target_only_minimum_normalized


class NullspaceInfeasible(ProtectedNullHold):
    """The frozen positive/protected projection has no admitted child surface."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise _hold(f"{label} is unreadable: {error}") from error
    if not isinstance(value, dict):
        raise _hold(f"{label} is not an object")
    return value


def _payload_identity(path: Path) -> tuple[list[int], np.ndarray, dict[str, Any]]:
    tensors = load_file(path, device="cpu")
    token_ids = tensors.get("selected_token_ids")
    rows = tensors.get("residual_rows")
    if (
        token_ids is None
        or token_ids.ndim != 1
        or token_ids.dtype != torch.int64
        or rows is None
        or rows.ndim != 2
        or rows.shape[0] != token_ids.numel()
        or rows.dtype != torch.float64
        or not bool(torch.isfinite(rows).all())
    ):
        raise _hold("residual payload tensor schema is invalid")
    ids = list(map(int, token_ids.tolist()))
    array = rows.numpy()
    identity = _residual_identity(ids, array) | {"payload_sha256": _sha256(path)}
    return ids, array, identity


def _load_parent_receipt() -> dict[str, Any]:
    if not PARENT_RECEIPT.is_file() or _sha256(PARENT_RECEIPT) != PARENT_RECEIPT_SHA256:
        raise _hold("immutable dyadic parent receipt drifted")
    return _load_json(PARENT_RECEIPT, label="dyadic parent receipt")


def _load_parent_payload(receipt: Mapping[str, Any]) -> dict[str, Any]:
    payload_record = dict(receipt.get("payload", {}))
    metadata_path = PARENT_PAYLOAD.with_suffix(".json")
    manifest_path = PARENT_ROOT / "augmented_checkpoint_manifest.json"
    if (
        Path(str(payload_record.get("payload", ""))) != PARENT_PAYLOAD
        or Path(str(payload_record.get("metadata", ""))) != metadata_path
        or Path(str(payload_record.get("manifest", ""))) != manifest_path
        or not PARENT_PAYLOAD.is_file()
        or _sha256(PARENT_PAYLOAD) != PARENT_PAYLOAD_SHA256
        or not metadata_path.is_file()
        or not manifest_path.is_file()
    ):
        raise _hold("dyadic parent payload paths or SHA drifted")
    ids, rows, identity = _payload_identity(PARENT_PAYLOAD)
    metadata = _load_json(metadata_path, label="dyadic parent residual metadata")
    manifest = _load_json(manifest_path, label="dyadic parent augmented manifest")
    if (
        ids != PARENT_SELECTED_TOKEN_IDS
        or tuple(rows.shape) != (9, 2048)
        or identity["residual_rows_sha256"] != PARENT_RESIDUAL_ROWS_SHA256
        or identity != payload_record.get("identity")
        or metadata.get("schema_version") != parent.SCHEMA_VERSION
        or metadata.get("unit_id") != parent.UNIT_ID
        or metadata.get("selected_token_ids") != PARENT_SELECTED_TOKEN_IDS
        or float(metadata.get("normalized_norm", math.inf)) != PARENT_NORMALIZED_NORM
        or metadata.get("cap") != NORM_CAP
        or metadata.get("cap_numerator") != NORM_CAP_NUMERATOR
        or metadata.get("cap_denominator") != NORM_CAP_DENOMINATOR
        or metadata.get("residual_identity") != identity
        or manifest.get("schema_version") != f"{parent.SCHEMA_VERSION}.augmented_checkpoint_manifest"
        or manifest.get("unit_id") != parent.UNIT_ID
        or Path(str(manifest.get("residual_payload", ""))) != PARENT_PAYLOAD
        or manifest.get("residual_payload_sha256") != PARENT_PAYLOAD_SHA256
        or Path(str(manifest.get("residual_metadata", ""))) != metadata_path
        or manifest.get("residual_metadata_sha256") != _sha256(metadata_path)
        or manifest.get("load_order") != [
            "load frozen base checkpoint",
            "install output-head residual module",
            "load residual payload",
        ]
    ):
        raise _hold("dyadic parent payload, metadata, manifest, or selected rows drifted")
    return {"selected_token_ids": ids, "residual_rows": rows, "identity": identity}


def _validate_parent_receipt(
    receipt: Mapping[str, Any], *, parent_contract: Mapping[str, Any], payload: Mapping[str, Any],
) -> tuple[list[int], list[str]]:
    warm = dict(receipt.get("warm_final", {}))
    cold = dict(receipt.get("cold_final", {}))
    warm_gate = dict(warm.get("gate", {}))
    cold_gate = dict(cold.get("gate", {}))
    warm_generic = dict(warm_gate.get("generic", {}))
    route = list(map(int, warm.get("generated_token_ids", ())))
    owners = list(map(str, warm_generic.get("matched_owner_ids", ())))
    source = dict(receipt.get("runner_source_snapshot", {}))
    source_path = Path(str(source.get("path", "")))
    warm_surface = dict(receipt.get("warm_surface", {}))
    cold_surface = dict(cold.get("surface", {}))
    cold_frozen = dict(cold.get("frozen_surface", {}))
    parity = dict(receipt.get("warm_cold", {}))
    sealed_binding = dict(receipt.get("bindings", {}))
    if (
        receipt.get("schema_version") != parent.SCHEMA_VERSION
        or receipt.get("unit_id") != parent.UNIT_ID
        or receipt.get("status") != "cold_greedy_38_person_success"
        or _sha256(Path(parent.__file__)) != PARENT_RUNNER_SHA256
        or source.get("sha256") != PARENT_RUNNER_SHA256
        or not source_path.is_file()
        or _sha256(source_path) != PARENT_RUNNER_SHA256
        or sealed_binding.get("schema_version") != parent.SCHEMA_VERSION
        or sealed_binding.get("unit_id") != parent.UNIT_ID
        or sealed_binding.get("status") != "verified"
        or sealed_binding.get("runner_sha256") != PARENT_RUNNER_SHA256
        or sealed_binding.get("checkpoint") != str(START_CHECKPOINT)
        or sealed_binding.get("checkpoint_readback_sha256")
        != _value_sha256(parent_contract["runtime_contract"]["checkpoint_readback"])
        or sealed_binding.get("controlled_route_sha256") != PARENT_ROUTE_SHA256
        or sealed_binding.get("controlled_route_token_count") != PARENT_ROUTE_LENGTH
        or sealed_binding.get("target_token_ids") != PARENT_SELECTED_TOKEN_IDS
        or sealed_binding.get("norm_cap") != NORM_CAP
        or sealed_binding.get("norm_cap_numerator") != NORM_CAP_NUMERATOR
        or sealed_binding.get("norm_cap_denominator") != NORM_CAP_DENOMINATOR
        or receipt.get("counts") != {
            "base_greedy": 1,
            "model_loads": 2,
            "solves": 1,
            "warm_candidates": 1,
            "zero_wrapper_controlled_capture": 1,
        }
        or len(route) != PARENT_ROUTE_LENGTH
        or token_ids_sha256(route) != PARENT_ROUTE_SHA256
        or route[-1:] != [EOS]
        or EOS in route[:-1]
        or warm.get("generated_token_ids_sha256") != PARENT_ROUTE_SHA256
        or cold.get("generated_token_ids") != route
        or cold.get("generated_token_ids_sha256") != PARENT_ROUTE_SHA256
        or warm.get("ledger") != cold.get("ledger")
        or warm_gate.get("passed") is not True
        or cold_gate != warm_gate
        or set(owners) != set(PARENT_OWNER_IDS)
        or len(owners) != 41
        or dict(warm.get("residual_identity", {})).get("payload_sha256") != PARENT_PAYLOAD_SHA256
        or dict(cold.get("residual_identity", {})) != dict(warm.get("residual_identity", {}))
        or receipt.get("payload", {}).get("identity") != payload["identity"]
        or not parity
        or not all(value is True for value in parity.values())
        or warm_surface.get("before") != warm_surface.get("after")
        or warm_surface.get("frozen_before") != warm_surface.get("frozen_after")
        or cold_surface.get("before") != cold_surface.get("after")
        or cold_frozen.get("before") != cold_frozen.get("after")
        or warm_surface.get("before") != cold_surface.get("before")
        or warm_surface.get("frozen_before") != cold_frozen.get("before")
    ):
        raise _hold("dyadic parent warm/cold/pass/surface/payload evidence drifted")
    receipts = list(dict(warm.get("ledger", {})).get("matcher", {}).get("prediction_receipts", ()))
    owner_order = [str(item.get("strict_match_gt_owner_id", "")) for item in receipts]
    if (
        len(owner_order) != 41
        or set(owner_order) != set(PARENT_OWNER_IDS)
        or any(item.get("strict_match_status") != "matched" for item in receipts)
    ):
        raise _hold("dyadic parent strict prediction ledger drifted")
    return route, owner_order


def _load_target() -> dict[str, Any]:
    if not TARGET_PATH.is_file() or _sha256(TARGET_PATH) != TARGET_SHA256:
        raise _hold("immutable target library drifted")
    target = _load_json(TARGET_PATH, label="target library")
    rows = list(target.get("rows", ()))
    by_owner = {str(item.get("owner", "")): dict(item) for item in rows}
    global_match = dict(target.get("global_match", {}))
    if (
        len(rows) != 46
        or len(by_owner) != 46
        or set(by_owner) != set(ALL_OWNER_IDS)
        or global_match.get("committed_owner_count") != 46
        or set(map(str, global_match.get("committed_owner_ids", ()))) != set(ALL_OWNER_IDS)
        or global_match.get("valid_prediction_count") != 46
        or global_match.get("dropped_prediction_count") != 0
        or global_match.get("person_prediction_count") != 38
        or global_match.get("tie_prediction_count") != 8
    ):
        raise _hold("target library 46-owner global matcher binding drifted")
    for owner in CANONICAL_OWNER_ORDER:
        tokens = list(map(int, by_owner[owner].get("token_ids", ())))
        if (
            tokens != CANONICAL_ROWS[owner]
            or token_ids_sha256(tokens) != CANONICAL_ROW_SHA256[owner]
            or by_owner[owner].get("description") != "tie"
        ):
            raise _hold(f"canonical row binding drifted for {owner}")
    return target


def _candidate_route(parent_route: Sequence[int]) -> list[int]:
    route = [
        *map(int, parent_route[: PARENT_ROUTE_LENGTH - 1]),
        *(token for owner in CANONICAL_OWNER_ORDER for token in CANONICAL_ROWS[owner]),
        EOS,
    ]
    if (
        len(parent_route) != PARENT_ROUTE_LENGTH
        or list(map(int, parent_route[-1:])) != [EOS]
        or len(route) != CANDIDATE_ROUTE_LENGTH
        or token_ids_sha256(route) != CANDIDATE_ROUTE_SHA256
        or route[-1] != EOS
        or EOS in route[:-1]
        or (len(route) - 1) // ROW_TOKENS != 46
    ):
        raise _hold("canonical 415-token route structure or identity drifted")
    return route


def _binding_contract() -> dict[str, Any]:
    receipt = _load_parent_receipt()
    parent_contract = parent._binding_contract()
    payload = _load_parent_payload(receipt)
    parent_route, parent_owner_order = _validate_parent_receipt(
        receipt, parent_contract=parent_contract, payload=payload,
    )
    target = _load_target()
    candidate = _candidate_route(parent_route)
    suffix_owners = [*parent_owner_order, *CANONICAL_OWNER_ORDER]
    suffix_hashes = [
        token_ids_sha256(candidate[index * ROW_TOKENS : (index + 1) * ROW_TOKENS])
        for index in range(46)
    ]
    if (
        len(suffix_owners) != 46
        or set(suffix_owners) != set(ALL_OWNER_IDS)
        or suffix_hashes[-5:] != [CANONICAL_ROW_SHA256[owner] for owner in CANONICAL_OWNER_ORDER]
        or candidate[:369] != parent_route[:369]
        or parent_contract["runtime_contract"]["target"] != target
    ):
        raise _hold("canonical route owner, row, target, or parent-prefix binding drifted")
    return {
        "parent_contract": parent_contract,
        "parent_receipt": receipt,
        "parent_payload": payload,
        "parent_route_tokens": parent_route,
        "parent_owner_order": parent_owner_order,
        "candidate_route_tokens": candidate,
        "candidate_owner_order": suffix_owners,
        "candidate_row_sha256": suffix_hashes,
        "target": target,
        "runtime_contract": parent_contract["runtime_contract"],
    }


def _binding_receipt(contract: Mapping[str, Any] | None = None) -> dict[str, Any]:
    bound = _binding_contract() if contract is None else contract
    payload = bound["parent_payload"]
    return {
        "schema_version": SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "status": "verified",
        "execution_surface": "cpu_only_no_model_load_no_cuda",
        "parent_receipt": str(PARENT_RECEIPT),
        "parent_receipt_sha256": PARENT_RECEIPT_SHA256,
        "parent_runner_sha256": PARENT_RUNNER_SHA256,
        "parent_payload": str(PARENT_PAYLOAD),
        "parent_payload_sha256": PARENT_PAYLOAD_SHA256,
        "parent_identity": payload["identity"],
        "parent_route_sha256": PARENT_ROUTE_SHA256,
        "parent_route_token_count": PARENT_ROUTE_LENGTH,
        "parent_owner_ids": sorted(PARENT_OWNER_IDS),
        "parent_owner_count": 41,
        "target": str(TARGET_PATH),
        "target_sha256": TARGET_SHA256,
        "canonical_owner_order": list(CANONICAL_OWNER_ORDER),
        "canonical_rows": [
            {
                "owner": owner,
                "token_ids": CANONICAL_ROWS[owner],
                "token_ids_sha256": CANONICAL_ROW_SHA256[owner],
            }
            for owner in CANONICAL_OWNER_ORDER
        ],
        "candidate_route_sha256": CANDIDATE_ROUTE_SHA256,
        "candidate_route_token_count": CANDIDATE_ROUTE_LENGTH,
        "candidate_owner_ids": sorted(ALL_OWNER_IDS),
        "candidate_owner_count": 46,
        "candidate_row_sha256": bound["candidate_row_sha256"],
        "checkpoint": str(START_CHECKPOINT),
        "checkpoint_readback_sha256": _value_sha256(
            bound["runtime_contract"]["checkpoint_readback"]
        ),
        "prompt_token_ids_sha256": PROMPT_TOKEN_SHA256,
        "image_sha256": IMAGE_SHA256,
        "margin": MARGIN,
        "norm_cap": NORM_CAP,
        "norm_cap_numerator": NORM_CAP_NUMERATOR,
        "norm_cap_denominator": NORM_CAP_DENOMINATOR,
        "resource_bound": RESOURCE_BOUND,
        "runner_sha256": _sha256(Path(__file__)),
    }


def _prepare_output(output: Path) -> dict[str, Any]:
    if output.exists():
        raise _hold(f"refusing overwrite: {output}")
    output.mkdir(parents=True)
    snapshot = output / "runner_source.py"
    source = Path(__file__).read_bytes()
    snapshot.write_bytes(source)
    if snapshot.read_bytes() != source:
        raise _hold("successor-local runner snapshot readback drifted")
    return {"path": str(snapshot), "sha256": _sha256(snapshot)}


def _protected_positive_basis(
    protected_hidden: np.ndarray, positive_hidden: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    protected = np.asarray(protected_hidden, dtype=np.float64)
    positive = np.asarray(positive_hidden, dtype=np.float64)
    if protected.ndim != 2 or positive.ndim != 2 or protected.shape[1] != positive.shape[1]:
        raise ValueError("protected and positive hidden states must share hidden width")
    q0, protected_info = _row_basis(protected)
    projected = positive - (positive @ q0) @ q0.T
    original_norms = np.linalg.norm(positive, axis=1)
    projected_norms = np.linalg.norm(projected, axis=1)
    if (
        not 1 <= len(positive) <= MAX_POSITIVES
        or np.any(original_norms < PROJECTED_NORM_FLOOR)
        or np.any(projected_norms < PROJECTED_NORM_FLOOR)
    ):
        raise NullspaceInfeasible(
            "HOLD: positive count or original/projected norm violates the frozen bound"
        )
    basis, positive_info = _row_basis(projected)
    null_max = float(np.max(np.abs(protected @ basis), initial=0.0))
    if not 1 <= basis.shape[1] <= min(len(positive), MAX_RANK) or null_max > PROTECTED_NULL_TOLERANCE:
        raise NullspaceInfeasible(
            "HOLD: protected-null positive basis is rank-invalid or numerically non-null"
        )
    return basis, {
        "protected": protected_info,
        "projected_positive": positive_info,
        "positive_count": int(len(positive)),
        "rank": int(basis.shape[1]),
        "original_norm_min": float(original_norms.min()),
        "projected_norm_min": float(projected_norms.min()),
        "protected_null_max_abs": null_max,
        "basis_sha256": _tensor_sha256(basis),
    }


def _capture_parent_candidate(
    *, model: Any, wrapper: SparseOutputResidual, native_inputs: Mapping[str, Any],
    parent_route: Sequence[int], candidate_route: Sequence[int], pad_token_id: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], list[dict[str, Any]], np.ndarray, np.ndarray, dict[str, Any]]:
    anchor = _capture_route(
        model=model,
        output_head=wrapper,
        native_inputs=native_inputs,
        route_tokens=parent_route,
        pad_token_id=pad_token_id,
    )
    candidate = _capture_route(
        model=model,
        output_head=wrapper,
        native_inputs=native_inputs,
        route_tokens=candidate_route,
        pad_token_id=pad_token_id,
    )
    anchor_top1 = anchor["logits"].argmax(dim=1).tolist()
    candidate_top1 = candidate["logits"].argmax(dim=1).tolist()
    if anchor_top1 != list(map(int, parent_route)):
        raise _hold("parent teacher capture does not exactly replay its ordinary route")
    states = [
        {
            "stage": 1,
            "position": position,
            "target_token_id": int(target),
            "hidden": candidate["hidden"][position].double().numpy(),
            "logits": candidate["logits"][position],
        }
        for position, (target, top1) in enumerate(zip(candidate_route, candidate_top1, strict=True))
        if int(target) != int(top1)
    ]
    if (
        not 1 <= len(states) <= MAX_POSITIVES
        or any(int(item["position"]) < PARENT_ROUTE_LENGTH - 1 for item in states)
    ):
        raise _hold("canonical candidate positive states escaped the bounded suffix")
    positions = [int(item["position"]) for item in states]
    protected, selection = _protected_hidden_matrix(
        ordinary_route=parent_route,
        ordinary_hidden=anchor["hidden"],
        controlled_route=candidate_route,
        controlled_hidden=candidate["hidden"],
        controlled_positive_positions=positions,
    )
    positive = np.stack([np.asarray(item["hidden"], dtype=np.float64) for item in states])
    basis, info = _protected_positive_basis(protected, positive)
    info.update({
        "protected_hidden_sha256": _tensor_sha256(protected),
        "protected_hidden_shape": list(protected.shape),
        "protected_null_apply_contract": (
            "FP64 payload; hidden cast FP64; hidden@D.T FP64; correction cast to logits dtype"
        ),
        "protected_null_tolerance": PROTECTED_NULL_TOLERANCE,
        "selection": selection,
    })
    return anchor, candidate, states, protected, basis, info


def _child_constraints(
    states: Sequence[Mapping[str, Any]], basis: np.ndarray,
) -> tuple[list[int], list[dict[str, Any]]]:
    child_ids = sorted({int(item["target_token_id"]) for item in states})
    if not 1 <= len(child_ids) <= MAX_CHILD_ROWS:
        raise _hold("dynamic child target-token surface violates 1..26")
    constraints = _target_only_constraints(states, basis, target_token_ids=child_ids)
    expected = len(states) * len(child_ids)
    if len(constraints) != expected or len(constraints) > MAX_CONSTRAINTS:
        raise _hold("dynamic target-only constraint count violates the frozen bound")
    by_position: dict[int, list[dict[str, Any]]] = {}
    for item in constraints:
        by_position.setdefault(int(item["position"]), []).append(item)
    for items in by_position.values():
        fixed = [item for item in items if not item["competitor_trainable"]]
        movable = [item for item in items if item["competitor_trainable"]]
        if (
            len(fixed) != 1
            or int(fixed[0]["competitor_token_id"]) in child_ids
            or {int(item["competitor_token_id"]) for item in movable}
            != set(child_ids) - {int(items[0]["target_token_id"])}
        ):
            raise _hold("constraint partition is not every other S plus one fixed V-minus-S maximum")
    return child_ids, constraints


def _compose_residuals(
    parent_ids: Sequence[int], parent_rows: np.ndarray,
    child_ids: Sequence[int], child_rows: np.ndarray,
) -> tuple[list[int], np.ndarray]:
    p_ids, c_ids = list(map(int, parent_ids)), list(map(int, child_ids))
    p_rows = np.asarray(parent_rows, dtype=np.float64)
    c_rows = np.asarray(child_rows, dtype=np.float64)
    if (
        len(p_ids) != len(set(p_ids))
        or len(c_ids) != len(set(c_ids))
        or p_rows.ndim != 2
        or c_rows.ndim != 2
        or p_rows.shape[0] != len(p_ids)
        or c_rows.shape[0] != len(c_ids)
        or p_rows.shape[1:] != c_rows.shape[1:]
        or not np.isfinite(p_rows).all()
        or not np.isfinite(c_rows).all()
    ):
        raise ValueError("parent/child residual shapes or ids are invalid")
    by_id = {token: p_rows[index].copy() for index, token in enumerate(p_ids)}
    for index, token in enumerate(c_ids):
        by_id[token] = by_id.get(token, np.zeros(p_rows.shape[1], dtype=np.float64)) + c_rows[index]
    union = sorted(by_id)
    return union, np.stack([by_id[token] for token in union])


def _effective_child_rows(
    *, parent_ids: Sequence[int], parent_rows: np.ndarray, child_ids: Sequence[int],
    composed_ids: Sequence[int], composed_rows: np.ndarray,
) -> np.ndarray:
    parent_by_id = dict(zip(map(int, parent_ids), np.asarray(parent_rows, dtype=np.float64), strict=True))
    composed_by_id = dict(zip(map(int, composed_ids), np.asarray(composed_rows, dtype=np.float64), strict=True))
    rows = np.stack([
        composed_by_id[int(token)] - parent_by_id.get(
            int(token), np.zeros(np.asarray(composed_rows).shape[1], dtype=np.float64),
        )
        for token in child_ids
    ])
    check_ids, check_rows = _compose_residuals(parent_ids, parent_rows, child_ids, rows)
    if list(map(int, composed_ids)) != check_ids or not np.array_equal(composed_rows, check_rows):
        raise _hold("composed payload cannot be exactly decomposed into parent plus child")
    return rows


def _final_gate(
    evaluation: Mapping[str, Any], *, binding: Mapping[str, Any],
) -> dict[str, Any]:
    generic = recursive._node_gate(evaluation, parent_owner_ids=PARENT_OWNER_IDS)
    owners = list(map(str, generic.get("matched_owner_ids", ())))
    target_rows = list(binding["target"].get("rows", ()))
    descriptions = {str(item.get("owner", "")): str(item.get("description", "")) for item in target_rows}
    people = [owner for owner in owners if descriptions.get(owner) == "person"]
    ties = [owner for owner in owners if descriptions.get(owner) == "tie"]
    tokens = list(map(int, evaluation.get("generated_token_ids", ())))
    receipts = list(dict(evaluation.get("ledger", {})).get("matcher", {}).get("prediction_receipts", ()))
    statuses = [str(item.get("strict_match_status", "")) for item in receipts]
    debt = dict(generic.get("debt", {}))
    if generic.get("passed") is not True:
        debt["generic_gate"] = True
    if set(owners) != set(ALL_OWNER_IDS) or len(owners) != 46:
        debt["all_46_owners"] = True
    if set(people) != {owner for owner in ALL_OWNER_IDS if descriptions.get(owner) == "person"} or len(people) != 38:
        debt["all_38_persons"] = True
    if set(ties) != {owner for owner in ALL_OWNER_IDS if descriptions.get(owner) == "tie"} or len(ties) != 8:
        debt["all_8_ties"] = True
    if len(receipts) != 46 or statuses != ["matched"] * 46:
        debt["exactly_46_strict_predictions"] = True
    if len(tokens) != CANDIDATE_ROUTE_LENGTH or tokens[-1:] != [EOS] or EOS in tokens[:-1]:
        debt["natural_46_row_eos"] = True
    return {
        "passed": not debt,
        "debt": debt,
        "owner_equivalent": not debt,
        "person_count": len(people),
        "tie_count": len(ties),
        "strict_prediction_count": len(receipts),
        "generic": generic,
    }


def _save_composed_payload(
    output: Path, *, binding: Mapping[str, Any], parent_ids: Sequence[int],
    parent_rows: np.ndarray, child_ids: Sequence[int], child_rows: np.ndarray,
    composed_ids: Sequence[int], composed_rows: np.ndarray, solution: Mapping[str, Any],
    basis_info: Mapping[str, Any], positive_positions: Sequence[int],
) -> dict[str, Any]:
    if (
        list(map(int, composed_ids)) != sorted(set(map(int, [*parent_ids, *child_ids])))
        or np.asarray(composed_rows).shape != (len(composed_ids), 2048)
        or float(solution["normalized_norm"]) > NORM_CAP
    ):
        raise _hold("refusing to persist an invalid composed residual")
    parent_identity = deepcopy(dict(binding["parent_payload"]["identity"]))
    child_identity = _residual_identity(child_ids, child_rows)
    payload = output / "canonical_five_tie_composed_output_residual.safetensors"
    save_file(
        {
            "selected_token_ids": torch.tensor(list(map(int, composed_ids)), dtype=torch.int64),
            "residual_rows": torch.from_numpy(np.asarray(composed_rows, dtype=np.float64)),
        },
        payload,
        metadata={"schema_version": SCHEMA_VERSION, "unit_id": UNIT_ID},
    )
    composed_identity = _residual_identity(composed_ids, composed_rows) | {
        "payload_sha256": _sha256(payload),
    }
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "runner_sha256": _sha256(Path(__file__)),
        "base_checkpoint": str(START_CHECKPOINT),
        "parent_receipt_sha256": PARENT_RECEIPT_SHA256,
        "parent_payload_sha256": PARENT_PAYLOAD_SHA256,
        "target_sha256": TARGET_SHA256,
        "parent_route_sha256": PARENT_ROUTE_SHA256,
        "candidate_route_sha256": CANDIDATE_ROUTE_SHA256,
        "margin": MARGIN,
        "normalization": "per-selected-original-base-output-row L2; child normalized Frobenius cap",
        "child_normalized_norm": float(solution["normalized_norm"]),
        "cap": NORM_CAP,
        "cap_numerator": NORM_CAP_NUMERATOR,
        "cap_denominator": NORM_CAP_DENOMINATOR,
        "positive_positions": list(map(int, positive_positions)),
        "child_selected_token_ids": list(map(int, child_ids)),
        "composed_selected_token_ids": list(map(int, composed_ids)),
        "basis": deepcopy(dict(basis_info)),
        "parent_identity": parent_identity,
        "child_identity": child_identity,
        "composed_identity": composed_identity,
    }
    metadata_path = payload.with_suffix(".json")
    _atomic_json(metadata_path, metadata)
    manifest = {
        "schema_version": f"{SCHEMA_VERSION}.augmented_checkpoint_manifest",
        "unit_id": UNIT_ID,
        "base_checkpoint": str(START_CHECKPOINT),
        "base_checkpoint_readback": binding["runtime_contract"]["checkpoint_readback"],
        "base_weight_payload_mode": "reference_only_no_copy_no_mutation",
        "parent_identity": parent_identity,
        "child_identity": child_identity,
        "composed_identity": composed_identity,
        "composed_payload": str(payload),
        "composed_payload_sha256": composed_identity["payload_sha256"],
        "composed_metadata": str(metadata_path),
        "composed_metadata_sha256": _sha256(metadata_path),
        "load_order": [
            "load frozen base checkpoint",
            "install exactly one output-head residual wrapper against the original base head",
            "load composed union payload",
        ],
    }
    manifest_path = output / "augmented_checkpoint_manifest.json"
    _atomic_json(manifest_path, manifest)
    return {
        "payload": str(payload),
        "metadata": str(metadata_path),
        "manifest": str(manifest_path),
        "parent_identity": parent_identity,
        "child_identity": child_identity,
        "composed_identity": composed_identity,
    }


def _load_composed_payload(
    payload: Path, *, binding: Mapping[str, Any],
) -> dict[str, Any]:
    metadata_path = payload.with_suffix(".json")
    manifest_path = payload.parent / "augmented_checkpoint_manifest.json"
    if not payload.is_file() or not metadata_path.is_file() or not manifest_path.is_file():
        raise _hold("cold composed payload set is incomplete")
    composed_ids, composed_rows, composed_identity = _payload_identity(payload)
    metadata = _load_json(metadata_path, label="composed residual metadata")
    manifest = _load_json(manifest_path, label="composed augmented manifest")
    child_ids = list(map(int, metadata.get("child_selected_token_ids", ())))
    parent_ids = binding["parent_payload"]["selected_token_ids"]
    parent_rows = binding["parent_payload"]["residual_rows"]
    child_rows = _effective_child_rows(
        parent_ids=parent_ids,
        parent_rows=parent_rows,
        child_ids=child_ids,
        composed_ids=composed_ids,
        composed_rows=composed_rows,
    )
    child_identity = _residual_identity(child_ids, child_rows)
    if (
        metadata.get("schema_version") != SCHEMA_VERSION
        or metadata.get("unit_id") != UNIT_ID
        or metadata.get("runner_sha256") != _sha256(Path(__file__))
        or metadata.get("base_checkpoint") != str(START_CHECKPOINT)
        or metadata.get("parent_receipt_sha256") != PARENT_RECEIPT_SHA256
        or metadata.get("parent_payload_sha256") != PARENT_PAYLOAD_SHA256
        or metadata.get("target_sha256") != TARGET_SHA256
        or metadata.get("parent_route_sha256") != PARENT_ROUTE_SHA256
        or metadata.get("candidate_route_sha256") != CANDIDATE_ROUTE_SHA256
        or metadata.get("margin") != MARGIN
        or metadata.get("cap") != NORM_CAP
        or metadata.get("cap_numerator") != NORM_CAP_NUMERATOR
        or metadata.get("cap_denominator") != NORM_CAP_DENOMINATOR
        or not 0.0 <= float(metadata.get("child_normalized_norm", math.inf)) <= NORM_CAP
        or not 1 <= len(child_ids) <= MAX_CHILD_ROWS
        or child_ids != sorted(set(child_ids))
        or metadata.get("composed_selected_token_ids") != composed_ids
        or metadata.get("parent_identity") != binding["parent_payload"]["identity"]
        or metadata.get("child_identity") != child_identity
        or metadata.get("composed_identity") != composed_identity
        or manifest.get("schema_version") != f"{SCHEMA_VERSION}.augmented_checkpoint_manifest"
        or manifest.get("unit_id") != UNIT_ID
        or Path(str(manifest.get("composed_payload", ""))) != payload
        or manifest.get("composed_payload_sha256") != composed_identity["payload_sha256"]
        or Path(str(manifest.get("composed_metadata", ""))) != metadata_path
        or manifest.get("composed_metadata_sha256") != _sha256(metadata_path)
        or manifest.get("parent_identity") != binding["parent_payload"]["identity"]
        or manifest.get("child_identity") != child_identity
        or manifest.get("composed_identity") != composed_identity
        or manifest.get("load_order") != [
            "load frozen base checkpoint",
            "install exactly one output-head residual wrapper against the original base head",
            "load composed union payload",
        ]
    ):
        raise _hold("cold parent/child/composed binding or payload schema is invalid")
    return {
        "metadata": metadata,
        "parent_ids": parent_ids,
        "parent_rows": parent_rows,
        "child_ids": child_ids,
        "child_rows": child_rows,
        "composed_ids": composed_ids,
        "composed_rows": composed_rows,
        "parent_identity": binding["parent_payload"]["identity"],
        "child_identity": child_identity,
        "composed_identity": composed_identity,
    }


def _cold_verify(*, payload: Path, result: Path) -> None:
    if result.exists():
        raise _hold(f"refusing overwrite: {result}")
    _require_one_gpu()
    binding = _binding_contract()
    loaded = _load_composed_payload(payload, binding=binding)
    contract = binding["runtime_contract"]
    setup = contract["setup"]
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    with base.open_backend_session(setup["frontend"].launch) as opened:
        if type(opened) is not base.HFBackendSession or opened._model is None or opened._tokenizer is None:
            raise _hold("cold verification requires concrete FP32 HF backend")
        model, tokenizer = opened._model, opened._tokenizer
        model.eval()
        native_inputs, prompts, _grids, _media = opened._materialize_native_inputs(setup["requests"][:1])
        if (
            token_ids_sha256(prompts[0]) != PROMPT_TOKEN_SHA256
            or _sha256(Path(setup["raw_example"].image.path)) != IMAGE_SHA256
        ):
            raise _hold("cold prompt/image identity drifted")
        names, parameters, surface_before, frozen_before = _surface_snapshot(model)
        sentinels = full_root._full_root_nontrainable_sentinels(model)
        base_head = model.get_output_embeddings()
        wrapper = SparseOutputResidual(
            base_head, loaded["parent_ids"], torch.from_numpy(loaded["parent_rows"]),
        ).to(next(model.parameters()).device)
        model.set_output_embeddings(wrapper)
        _anchor, _candidate, states, protected, basis, basis_info = _capture_parent_candidate(
            model=model,
            wrapper=wrapper,
            native_inputs=native_inputs,
            parent_route=binding["parent_route_tokens"],
            candidate_route=binding["candidate_route_tokens"],
            pad_token_id=int(tokenizer.pad_token_id),
        )
        metadata = loaded["metadata"]
        if (
            [int(item["position"]) for item in states] != metadata.get("positive_positions")
            or sorted({int(item["target_token_id"]) for item in states}) != loaded["child_ids"]
            or _tensor_sha256(protected) != dict(metadata.get("basis", {})).get("protected_hidden_sha256")
            or _tensor_sha256(basis) != dict(metadata.get("basis", {})).get("basis_sha256")
            or basis_info["selection"] != dict(metadata.get("basis", {})).get("selection")
        ):
            raise _hold("cold positive/protected/basis/selected-row recapture drifted")
        null_max = float(
            np.max(np.abs(protected @ loaded["child_rows"].T), initial=0.0)
        )
        if null_max > PROTECTED_NULL_TOLERANCE:
            raise _hold("cold child residual exceeds the protected-null tolerance")
        violation = _full_vocab_violation(
            states,
            selected_token_ids=loaded["child_ids"],
            residual_rows=loaded["child_rows"],
        )
        if violation is not None:
            raise _hold("cold child residual fails the exhaustive full-vocabulary margin recheck")
        wrapper.set_payload(loaded["composed_ids"], torch.from_numpy(loaded["composed_rows"]))
        route = _greedy_route(model, native_inputs, int(tokenizer.pad_token_id))
        evaluation = _evaluate_route(
            tokenizer=tokenizer,
            tokens=route,
            contract=contract,
            parent_owner_ids=PARENT_OWNER_IDS,
            label="canonical-five-tie-protected-null-final",
        )
        gate = _final_gate(evaluation, binding=binding)
        model.set_output_embeddings(base_head)
        full_root._assert_full_root_sentinels(model, sentinels)
        names_after, parameters_after, surface_after, frozen_after = _surface_snapshot(model)
        if names != names_after or any(
            left is not right for left, right in zip(parameters, parameters_after, strict=True)
        ):
            raise _hold("cold base parameter identity changed after residual removal")
        runtime = opened.receipt.to_artifact_dict()
    _atomic_json(result, {
        "schema_version": SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "status": "cold_verification_complete",
        "generated_token_ids": route,
        "generated_token_ids_sha256": token_ids_sha256(route),
        "ledger": evaluation["ledger"],
        "gate": gate,
        "parent_identity": loaded["parent_identity"],
        "child_identity": loaded["child_identity"],
        "composed_identity": loaded["composed_identity"],
        "positive_positions": [int(item["position"]) for item in states],
        "protected_hidden_sha256": _tensor_sha256(protected),
        "basis_sha256": _tensor_sha256(basis),
        "protected_selection": basis_info["selection"],
        "protected_null_max_abs": null_max,
        "protected_null_tolerance": PROTECTED_NULL_TOLERANCE,
        "surface": {"before": surface_before, "after": surface_after},
        "frozen_surface": {"before": frozen_before, "after": frozen_after},
        "runtime": runtime,
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        "model_load_count": 1,
        "wrapper_count": 1,
    })


def _cuda_metric(name: str) -> int:
    if not torch.cuda.is_available():
        return 0
    return int(getattr(torch.cuda, name)(0))


def run(*, run_id: str) -> Path:
    output = OUTPUT_ROOT / base._safe_run_id(run_id)
    started = time.monotonic()
    snapshot = _prepare_output(output)
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "run_id": output.name,
        "status": "technical_hold",
        "runner_source_snapshot": snapshot,
        "counts": {
            "model_loads": 0,
            "solves": 0,
            "warm_candidates": 0,
            "parent_greedy": 0,
            "static_candidate_gates": 0,
            "warm_wrapper_count": 0,
        },
    }
    phase = "binding"
    terminal_status: str | None = None
    binding = warm_final = payload_record = None
    model = wrapper = base_head = opened = native_inputs = tokenizer = None
    try:
        _require_one_gpu()
        binding = _binding_contract()
        contract = binding["runtime_contract"]
        receipt["bindings"] = _binding_receipt(binding)
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
        setup = contract["setup"]
        phase = "single_warm_model_load"
        with base.open_backend_session(setup["frontend"].launch) as opened:
            if type(opened) is not base.HFBackendSession or opened._model is None or opened._tokenizer is None:
                raise _hold("sentinel requires concrete FP32 HF backend")
            receipt["counts"]["model_loads"] = 1
            model, tokenizer = opened._model, opened._tokenizer
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            native_inputs, prompts, _grids, _media = opened._materialize_native_inputs(setup["requests"][:1])
            if (
                token_ids_sha256(prompts[0]) != PROMPT_TOKEN_SHA256
                or _sha256(Path(setup["raw_example"].image.path)) != IMAGE_SHA256
            ):
                raise _hold("warm prompt/image identity drifted")
            names, parameters, surface_before, frozen_before = _surface_snapshot(model)
            sentinels = full_root._full_root_nontrainable_sentinels(model)
            base_head = model.get_output_embeddings()

            phase = "parent_zero_child_composition_replay"
            parent_ids = binding["parent_payload"]["selected_token_ids"]
            parent_rows = binding["parent_payload"]["residual_rows"]
            zero_ids, zero_rows = _compose_residuals(
                parent_ids,
                parent_rows,
                [],
                np.zeros((0, parent_rows.shape[1]), dtype=np.float64),
            )
            if zero_ids != parent_ids or not np.array_equal(zero_rows, parent_rows):
                raise _hold("zero-child composition is not bit-exact parent payload")
            wrapper = SparseOutputResidual(base_head, zero_ids, torch.from_numpy(zero_rows)).to(
                next(model.parameters()).device
            )
            model.set_output_embeddings(wrapper)
            receipt["counts"]["warm_wrapper_count"] = 1
            parent_route = _greedy_route(model, native_inputs, int(tokenizer.pad_token_id))
            receipt["counts"]["parent_greedy"] = 1
            parent_evaluation = _evaluate_route(
                tokenizer=tokenizer,
                tokens=parent_route,
                contract=contract,
                parent_owner_ids=binding["parent_contract"]["blocks"][-1]["parent_owner_ids"],
                label="dyadic-norm-release-final",
            )
            parent_gate = parent._final_gate(parent_evaluation, binding=binding["parent_contract"])
            expected_parent = binding["parent_receipt"]["warm_final"]
            if (
                parent_route != binding["parent_route_tokens"]
                or parent_evaluation["ledger"] != expected_parent["ledger"]
                or parent_gate != expected_parent["gate"]
            ):
                raise _hold("zero-child composition failed exact parent route/ledger replay")
            receipt["parent_replay"] = {
                "generated_token_ids_sha256": token_ids_sha256(parent_route),
                "ledger_sha256": _value_sha256(parent_evaluation["ledger"]),
                "gate": parent_gate,
                "zero_child_parent_payload_bit_exact": True,
            }

            phase = "production_static_candidate_gate"
            static_evaluation = _evaluate_route(
                tokenizer=tokenizer,
                tokens=binding["candidate_route_tokens"],
                contract=contract,
                parent_owner_ids=PARENT_OWNER_IDS,
                label="canonical-five-tie-static",
            )
            static_gate = _final_gate(static_evaluation, binding=binding)
            receipt["counts"]["static_candidate_gates"] = 1
            receipt["static_candidate"] = {
                "generated_token_ids_sha256": CANDIDATE_ROUTE_SHA256,
                "ledger": static_evaluation["ledger"],
                "gate": static_gate,
            }
            if not static_gate["passed"]:
                terminal_status = "static_route_hold"
                raise _hold("canonical candidate failed the production parser/global matcher gate")

            phase = "parent_augmented_anchor_candidate_capture"
            _anchor, _candidate, states, protected, basis, basis_info = _capture_parent_candidate(
                model=model,
                wrapper=wrapper,
                native_inputs=native_inputs,
                parent_route=binding["parent_route_tokens"],
                candidate_route=binding["candidate_route_tokens"],
                pad_token_id=int(tokenizer.pad_token_id),
            )
            child_ids, constraints = _child_constraints(states, basis)
            receipt["capture"] = {
                "parent_route_sha256": PARENT_ROUTE_SHA256,
                "candidate_route_sha256": CANDIDATE_ROUTE_SHA256,
                "positive_count": len(states),
                "positive_positions": [int(item["position"]) for item in states],
                "positive_target_token_ids": [int(item["target_token_id"]) for item in states],
                "protected_count": len(protected),
                "child_selected_token_ids": child_ids,
                "basis": basis_info,
            }

            phase = "one_repaired_highs_slsqp_solve"
            solved = _solve_target_only_minimum_normalized(
                constraints,
                rank=basis.shape[1],
                row_norms=_row_norms(base_head, child_ids),
                target_token_ids=child_ids,
            )
            receipt["counts"]["solves"] = 1
            receipt["child_program"] = {
                "positive_count": len(states),
                "rank": int(basis.shape[1]),
                "child_row_count": len(child_ids),
                "constraint_count": len(constraints),
                "constraints": _compact_constraints(constraints),
                "solver": {
                    key: value
                    for key, value in solved.items()
                    if key not in {"normalized_rows", "scaled_basis_rows"}
                },
            }
            if not solved.get("feasible"):
                if solved.get("solver_classification") == "certified_infeasible":
                    terminal_status = "certified_infeasible"
                else:
                    terminal_status = "numerical_solver_hold"
                raise _hold("repaired child solver did not produce an accepted minimum-norm primal")
            if (
                solved.get("selected_token_ids") != child_ids
                or int(solved.get("variable_count", -1)) != len(child_ids) * basis.shape[1]
                or int(solved.get("variable_count", -1)) > MAX_VARIABLES
            ):
                raise _hold("child solve escaped the dynamic S/rank surface")
            if float(solved["normalized_norm"]) > NORM_CAP:
                terminal_status = "norm_cap_exceeded"
                raise _hold("child minimum normalized norm exceeds 9/8")
            raw_child_rows = _collapsed_rows(solved, basis)
            composed_ids, composed_rows = _compose_residuals(
                parent_ids, parent_rows, child_ids, raw_child_rows,
            )
            child_rows = _effective_child_rows(
                parent_ids=parent_ids,
                parent_rows=parent_rows,
                child_ids=child_ids,
                composed_ids=composed_ids,
                composed_rows=composed_rows,
            )
            null_max = float(np.max(np.abs(protected @ child_rows.T), initial=0.0))
            receipt["child_program"]["protected_null_max_abs"] = null_max
            receipt["child_program"]["protected_null_tolerance"] = PROTECTED_NULL_TOLERANCE
            if null_max > PROTECTED_NULL_TOLERANCE:
                terminal_status = "nullspace_infeasible"
                raise _hold("effective child residual is not numerically protected-null")
            violation = _full_vocab_violation(
                states,
                selected_token_ids=child_ids,
                residual_rows=child_rows,
            )
            receipt["child_program"]["runtime_full_vocab_recheck"] = {
                "state_count": len(states),
                "passed": violation is None,
                "violation": None if violation is None else {
                    "position": int(violation["state"]["position"]),
                    "target_token_id": int(violation["state"]["target_token_id"]),
                    "competitor_token_id": int(violation["competitor_token_id"]),
                    "corrected_margin": float(violation["corrected_margin"]),
                },
            }
            if violation is not None:
                terminal_status = "runtime_margin_hold"
                raise _hold("child runtime-dtype exhaustive full-vocabulary margin failed")

            phase = "one_warm_ordinary_greedy_candidate"
            wrapper.set_payload(composed_ids, torch.from_numpy(composed_rows))
            final_route = _greedy_route(model, native_inputs, int(tokenizer.pad_token_id))
            receipt["counts"]["warm_candidates"] = 1
            final_evaluation = _evaluate_route(
                tokenizer=tokenizer,
                tokens=final_route,
                contract=contract,
                parent_owner_ids=PARENT_OWNER_IDS,
                label="canonical-five-tie-protected-null-final",
            )
            final_gate = _final_gate(final_evaluation, binding=binding)
            warm_final = {
                "generated_token_ids": final_route,
                "generated_token_ids_sha256": token_ids_sha256(final_route),
                "ledger": deepcopy(dict(final_evaluation["ledger"])),
                "gate": final_gate,
                "parent_identity": deepcopy(dict(binding["parent_payload"]["identity"])),
                "child_identity": _residual_identity(child_ids, child_rows),
                "composed_identity": _residual_identity(composed_ids, composed_rows),
                "positive_positions": [int(item["position"]) for item in states],
                "protected_hidden_sha256": _tensor_sha256(protected),
                "basis_sha256": _tensor_sha256(basis),
                "protected_selection": basis_info["selection"],
                "protected_null_max_abs": null_max,
                "protected_null_tolerance": PROTECTED_NULL_TOLERANCE,
            }
            receipt["warm_final"] = warm_final
            if not final_gate["passed"]:
                terminal_status = "canonical_greedy_negative"
                raise _hold("sole canonical ordinary-greedy candidate failed the 46-owner gate")

            payload_record = _save_composed_payload(
                output,
                binding=binding,
                parent_ids=parent_ids,
                parent_rows=parent_rows,
                child_ids=child_ids,
                child_rows=child_rows,
                composed_ids=composed_ids,
                composed_rows=composed_rows,
                solution=solved,
                basis_info=basis_info,
                positive_positions=warm_final["positive_positions"],
            )
            warm_final["composed_identity"] = payload_record["composed_identity"]
            receipt["payload"] = payload_record
            model.set_output_embeddings(base_head)
            full_root._assert_full_root_sentinels(model, sentinels)
            names_after, parameters_after, surface_after, frozen_after = _surface_snapshot(model)
            if (
                names_after != names
                or any(left is not right for left, right in zip(parameters, parameters_after, strict=True))
                or surface_after != surface_before
                or frozen_after != frozen_before
            ):
                raise _hold("warm residual execution mutated a frozen model surface")
            receipt["warm_surface"] = {
                "before": surface_before,
                "after": surface_after,
                "frozen_before": frozen_before,
                "frozen_after": frozen_after,
            }
            receipt["runtime"] = {"warm": opened.receipt.to_artifact_dict()}

        phase = "fresh_successor_snapshot_cold_verification"
        model = wrapper = base_head = opened = native_inputs = tokenizer = None
        states = protected = basis = raw_child_rows = child_rows = composed_rows = None
        parameter = parameters = parameters_after = sentinels = final_evaluation = None
        _anchor = _candidate = _grids = _media = None
        gc.collect()
        torch.cuda.empty_cache()
        receipt["resources_before_cold"] = {
            "cuda_allocated_bytes_after_reference_drop": _cuda_metric("memory_allocated"),
            "cuda_reserved_bytes_after_empty_cache": _cuda_metric("memory_reserved"),
            "gc_collected_before_cold": True,
        }
        cold_path = output / "cold_verification.json"
        snapshot_path = Path(snapshot["path"])
        repo = str(Path(parent.__file__).resolve().parents[2])
        env = dict(os.environ)
        env["PYTHONPATH"] = repo + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        remaining = max(1.0, RESOURCE_BOUND["wall_time_seconds_max"] - (time.monotonic() - started))
        completed = subprocess.run(
            [
                sys.executable,
                str(snapshot_path),
                "--cold-verify",
                "--payload",
                str(payload_record["payload"]),
                "--cold-result",
                str(cold_path),
            ],
            cwd=repo,
            env=env,
            text=True,
            capture_output=True,
            timeout=remaining,
            check=False,
        )
        if completed.returncode != 0 or not cold_path.is_file():
            raise _hold(
                "fresh successor-local cold subprocess failed: "
                + (completed.stderr[-2_000:] or completed.stdout[-2_000:] or f"exit {completed.returncode}")
            )
        cold = _load_json(cold_path, label="cold verification result")
        receipt["counts"]["model_loads"] = 2
        receipt["runtime"]["cold"] = cold.get("runtime")
        receipt["cold_final"] = cold
        warm_cold = {
            "successor_schema_exact": cold.get("schema_version") == SCHEMA_VERSION
            and cold.get("unit_id") == UNIT_ID,
            "route_exact": warm_final["generated_token_ids"] == cold.get("generated_token_ids"),
            "ledger_exact": warm_final["ledger"] == cold.get("ledger"),
            "parent_identity_exact": warm_final["parent_identity"] == cold.get("parent_identity"),
            "child_identity_exact": warm_final["child_identity"] == cold.get("child_identity"),
            "composed_identity_exact": warm_final["composed_identity"] == cold.get("composed_identity"),
            "selected_rows_exact": warm_final["child_identity"]["selected_token_ids"]
            == dict(cold.get("child_identity", {})).get("selected_token_ids"),
            "positive_positions_exact": warm_final["positive_positions"] == cold.get("positive_positions"),
            "protected_hidden_exact": warm_final["protected_hidden_sha256"]
            == cold.get("protected_hidden_sha256"),
            "basis_exact": warm_final["basis_sha256"] == cold.get("basis_sha256"),
            "protected_selection_exact": warm_final["protected_selection"]
            == cold.get("protected_selection"),
            "warm_gate_passed": warm_final["gate"].get("passed") is True,
            "cold_gate_passed": dict(cold.get("gate", {})).get("passed") is True,
            "warm_surface_frozen": receipt["warm_surface"]["before"]
            == receipt["warm_surface"]["after"],
            "cold_surface_frozen": dict(cold.get("surface", {})).get("before")
            == dict(cold.get("surface", {})).get("after"),
            "surface_exact": receipt["warm_surface"]["before"]
            == dict(cold.get("surface", {})).get("before"),
            "frozen_surface_exact": receipt["warm_surface"]["frozen_before"]
            == dict(cold.get("frozen_surface", {})).get("before"),
            "warm_protected_null": float(warm_final["protected_null_max_abs"])
            <= PROTECTED_NULL_TOLERANCE,
            "cold_protected_null": float(cold.get("protected_null_max_abs", math.inf))
            <= PROTECTED_NULL_TOLERANCE,
            "cold_one_wrapper": cold.get("wrapper_count") == 1,
        }
        receipt["warm_cold"] = warm_cold
        if not all(warm_cold.values()):
            raise _hold("warm/cold route, ledger, identities, rows, states, null, or surface mismatch")
        terminal_status = "cold_greedy_46_owner_success"
    except BaseException as error:
        if terminal_status is None:
            if phase == "production_static_candidate_gate":
                terminal_status = "static_route_hold"
            elif isinstance(error, NullspaceInfeasible):
                terminal_status = "nullspace_infeasible"
            else:
                terminal_status = "technical_hold"
        receipt["stop"] = {
            "phase": phase,
            "reason": str(error),
            "error_type": type(error).__name__,
            "traceback": traceback.format_exc(),
        }
    finally:
        if model is not None and base_head is not None:
            try:
                model.set_output_embeddings(base_head)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
        elapsed = time.monotonic() - started
        receipt["status"] = terminal_status if terminal_status in TERMINAL_STATUSES else "technical_hold"
        receipt["wall_time_seconds"] = elapsed
        warm_peak_reserved = _cuda_metric("max_memory_reserved")
        cold_peak_reserved = int(dict(receipt.get("cold_final", {})).get("peak_cuda_reserved_bytes", 0))
        receipt["resources"] = {
            "peak_cuda_allocated_bytes": _cuda_metric("max_memory_allocated"),
            "warm_peak_cuda_reserved_bytes": warm_peak_reserved,
            "cold_peak_cuda_reserved_bytes": cold_peak_reserved,
            "peak_cuda_reserved_bytes": max(warm_peak_reserved, cold_peak_reserved),
            "artifact_bytes_before_receipt": _artifact_bytes(output),
            "predeclared_bound": RESOURCE_BOUND,
        }
        receipt["claim_boundary"] = (
            "One Image2299 composed-augmented-model ordinary-greedy 46-owner result only; "
            "not base-model, transfer, or general enumeration evidence."
        )
        if (
            elapsed > RESOURCE_BOUND["wall_time_seconds_max"]
            or receipt["counts"]["solves"] > MAX_SOLVES
            or receipt["counts"]["warm_candidates"] > MAX_WARM_CANDIDATES
            or receipt["counts"]["model_loads"] > 2
            or receipt["counts"]["warm_wrapper_count"] > 1
            or receipt["resources"]["peak_cuda_reserved_bytes"]
            > RESOURCE_BOUND["max_peak_cuda_reserved_bytes"]
            or receipt["resources"]["artifact_bytes_before_receipt"]
            > RESOURCE_BOUND["output_artifact_bytes_max"]
        ):
            receipt["pre_resource_status"] = receipt["status"]
            receipt["status"] = "technical_hold"
            receipt.setdefault(
                "stop",
                {"phase": "resource_enforcement", "reason": "predeclared resource bound exceeded"},
            )
        _atomic_json(output / "receipt.json", receipt)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-bindings", action="store_true", help="CPU-only immutable binding check")
    parser.add_argument("--run-id")
    parser.add_argument("--cold-verify", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--payload", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--cold-result", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.check_bindings:
        if args.run_id or args.cold_verify or args.payload or args.cold_result:
            raise SystemExit("--check-bindings cannot be combined with execution arguments")
        print(json.dumps(_binding_receipt(), indent=2, sort_keys=True))
        return
    if args.cold_verify:
        if args.run_id or args.payload is None or args.cold_result is None:
            raise SystemExit("internal --cold-verify requires --payload and --cold-result only")
        _cold_verify(payload=args.payload, result=args.cold_result)
        return
    if not args.run_id or args.payload or args.cold_result:
        raise SystemExit("one-GPU execution requires exactly --run-id")
    print(run(run_id=args.run_id))


if __name__ == "__main__":
    main()
