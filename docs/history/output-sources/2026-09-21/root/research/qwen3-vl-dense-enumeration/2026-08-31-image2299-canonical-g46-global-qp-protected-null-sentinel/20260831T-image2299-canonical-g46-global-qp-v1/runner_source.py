#!/usr/bin/env python3
"""Compile the frozen canonical Image2299 G46 route with one direct global QP."""

from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
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

from scripts.research import (
    run_image2299_global_target_only_protected_null_distillation as qp,
)
from scripts.research import (
    run_image2299_matched_sequence_optimizer_ablation as matched,
)
from scripts.research import (
    run_image2299_protected_null_output_distillation as protected,
)


base = protected.base
recursive = protected.recursive
full_root = protected.full_root
token_ids_sha256 = protected.token_ids_sha256

SCHEMA_VERSION = "image2299.canonical_g46_global_qp_protected_null_sentinel.v1"
UNIT_ID = "2026-08-31-image2299-canonical-g46-global-qp-protected-null-sentinel"
OUTPUT_ROOT = (
    Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration") / UNIT_ID
)
TARGET_PATH = matched.TARGET_PATH
TARGET_SHA256 = "22c24c53af14f8a0969bb09efe3150d63bdca19ca3c85baac046450d62576988"
EOS, ROW_TOKENS, MARGIN = 151645, 9, 0.01
G41_OWNER_LEDGER = tuple(f"gt:2299:{owner}" for owner in matched.OWNER_LEDGER)
MISSING_TIE_OWNERS = (
    "gt:2299:45",
    "gt:2299:11",
    "gt:2299:9",
    "gt:2299:8",
    "gt:2299:43",
)
OWNER_LEDGER = (*G41_OWNER_LEDGER, *MISSING_TIE_OWNERS)
ALL_OWNER_IDS = tuple(f"gt:2299:{owner}" for owner in range(46))
G46_PRE_EOS_SHA256 = "d5e7dc703727f2f3db265df848f4c64fbc67e9fb8417eca23d0ae0c8e69c8dbb"
G46_ROUTE_SHA256 = "96c2cbe15fdb4c09822d6a472f2510605c5701cfafcc9e33324f14ce90e0cb49"
G46_ROUTE_LENGTH = 415
PROTECTED_NULL_TOLERANCE = 1.0e-10

# Generous deterministic admission bounds; a route or S above these is a HOLD, never trimmed.
MAX_POSITIVES = G46_ROUTE_LENGTH
MAX_SELECTED_ROWS = G46_ROUTE_LENGTH
MAX_RANK = G46_ROUTE_LENGTH
MAX_CONSTRAINTS = MAX_POSITIVES * MAX_SELECTED_ROWS
MAX_VARIABLES = MAX_SELECTED_ROWS * MAX_RANK
RESOURCE_BOUND = {
    "gpu_count": 1,
    "world_size": 1,
    "positive_states_max": MAX_POSITIVES,
    "selected_output_rows_max": MAX_SELECTED_ROWS,
    "positive_rank_max": MAX_RANK,
    "constraint_count_max": MAX_CONSTRAINTS,
    "variable_count_max": MAX_VARIABLES,
    "solve_count_max": 1,
    "warm_candidate_count_max": 1,
    "model_load_count_max": 2,
    "wall_time_seconds_max": 7200,
    "max_peak_cuda_reserved_bytes": 16 * 2**30,
    "output_artifact_bytes_max": 300_000_000,
}
TERMINAL_STATUSES = {
    "cold_greedy_46_owner_success",
    "static_route_hold",
    "nullspace_infeasible",
    "certified_infeasible",
    "numerical_solver_hold",
    "runtime_margin_hold",
    "canonical_greedy_negative",
    "technical_hold",
}

ProtectedNullHold = protected.ProtectedNullHold
_hold = protected._hold
_sha256 = protected._sha256
_value_sha256 = protected._value_sha256
_tensor_sha256 = protected._tensor_sha256
_atomic_json = protected._atomic_json
_capture_route = protected._capture_route
_greedy_route = protected._greedy_route
_evaluate_route = protected._evaluate_route
_row_basis = protected._row_basis
_protected_hidden_matrix = protected._protected_hidden_matrix
_target_only_constraints = qp._target_only_constraints
_full_vocab_violation = protected._full_vocab_violation
_row_norms = protected._row_norms
_collapsed_rows = protected._collapsed_rows
_residual_identity = protected._residual_identity
_surface_snapshot = protected._surface_snapshot
_artifact_bytes = protected._artifact_bytes
_require_one_gpu = protected._require_one_gpu
SparseOutputResidual = protected.SparseOutputResidual
_solve_target_only_minimum_normalized = qp._solve_target_only_minimum_normalized


class NullspaceInfeasible(ProtectedNullHold):
    """The requested positive surface disappeared after protected projection."""


def _json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise _hold(f"{label} is unreadable: {error}") from error
    if not isinstance(value, dict):
        raise _hold(f"{label} is not an object")
    return value


def _g46_route(target: Mapping[str, Any]) -> tuple[list[int], list[str], list[str]]:
    rows = {str(row.get("owner", "")): dict(row) for row in target.get("rows", ())}
    if set(rows) != set(ALL_OWNER_IDS) or len(rows) != 46:
        raise _hold("target library does not provide exactly the canonical 46 owners")
    route_rows = [
        list(map(int, rows[owner].get("token_ids", ()))) for owner in OWNER_LEDGER
    ]
    route = [token for row in route_rows for token in row] + [EOS]
    hashes = [token_ids_sha256(row) for row in route_rows]
    if (
        any(len(row) != ROW_TOKENS for row in route_rows)
        or len(route) != G46_ROUTE_LENGTH
        or token_ids_sha256(route[:-1]) != G46_PRE_EOS_SHA256
        or token_ids_sha256(route) != G46_ROUTE_SHA256
        or route[-1:] != [EOS]
        or EOS in route[:-1]
    ):
        raise _hold("frozen canonical G46 route identity drifted")
    return route, list(OWNER_LEDGER), hashes


def _static_gate(
    *, target: Mapping[str, Any], route: Sequence[int], owners: Sequence[str]
) -> dict[str, Any]:
    """CPU-only production-shaped parser/global-matcher gate for canonical target rows."""
    rows = {str(row.get("owner", "")): dict(row) for row in target.get("rows", ())}
    expected = [list(map(int, rows[owner].get("token_ids", ()))) for owner in owners]
    observed = [
        list(map(int, route[i : i + ROW_TOKENS]))
        for i in range(0, len(route) - 1, ROW_TOKENS)
    ]
    descriptions = {
        owner: str(row.get("description", "")) for owner, row in rows.items()
    }
    people = [owner for owner in owners if descriptions.get(owner) == "person"]
    ties = [owner for owner in owners if descriptions.get(owner) == "tie"]
    debt: dict[str, bool] = {}
    if observed != expected or len(observed) != 46:
        debt["canonical_rows"] = True
    if list(owners) != list(OWNER_LEDGER) or len(set(owners)) != 46:
        debt["strict_owner_ledger"] = True
    if len(people) != 38 or len(ties) != 8:
        debt["person_tie_counts"] = True
    if list(route[-1:]) != [EOS] or EOS in route[:-1]:
        debt["natural_terminal_eos"] = True
    return {
        "passed": not debt,
        "debt": debt,
        "person_count": len(people),
        "tie_count": len(ties),
        "strict_owner_count": len(owners),
        "natural_row_aligned_eos": not debt.get("natural_terminal_eos", False),
    }


def _binding_contract() -> dict[str, Any]:
    static = matched._static_binding()
    if not TARGET_PATH.is_file() or _sha256(TARGET_PATH) != TARGET_SHA256:
        raise _hold("immutable target library SHA drifted")
    target = _json(TARGET_PATH, label="target library")
    route, owners, hashes = _g46_route(target)
    gate = _static_gate(target=target, route=route, owners=owners)
    if list(static["m_owner_ledger"]) != list(G41_OWNER_LEDGER) or not gate["passed"]:
        raise _hold("G41 owner ledger or G46 parser gate drifted")
    return {
        "target": target,
        "route_tokens": route,
        "owner_ledger": owners,
        "row_sha256": hashes,
        "static_gate": gate,
        "g41_route_sha256": matched.G_ROUTE_SHA256,
    }


def _runtime_contract(binding: Mapping[str, Any]) -> Mapping[str, Any]:
    """GPU-only runtime admission; deliberately absent from --check-bindings."""
    runtime = matched.parent._binding_contract()["runtime_contract"]
    if runtime["target"] != binding["target"]:
        raise _hold("frozen runtime target drifted from CPU-bound target library")
    return runtime


def _binding_receipt(binding: Mapping[str, Any] | None = None) -> dict[str, Any]:
    bound = _binding_contract() if binding is None else binding
    return {
        "schema_version": SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "status": "verified",
        "execution_surface": "cpu_only_no_model_load_no_cuda",
        "target": str(TARGET_PATH),
        "target_sha256": TARGET_SHA256,
        "g41_owner_ledger": list(G41_OWNER_LEDGER),
        "missing_tie_owners": list(MISSING_TIE_OWNERS),
        "owner_ledger": list(OWNER_LEDGER),
        "route_sha256": G46_ROUTE_SHA256,
        "pre_eos_sha256": G46_PRE_EOS_SHA256,
        "route_token_count": G46_ROUTE_LENGTH,
        "row_sha256": bound["row_sha256"],
        "static_gate": bound["static_gate"],
        "checkpoint": str(matched.parent.START_CHECKPOINT),
        "margin": MARGIN,
        "protected_null_tolerance": PROTECTED_NULL_TOLERANCE,
        "resource_bound": RESOURCE_BOUND,
        "runner_sha256": _sha256(Path(__file__)),
    }


def _protected_positive_basis(
    protected_hidden: np.ndarray, positive_hidden: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    protected_rows, positive_rows = (
        np.asarray(protected_hidden, dtype=np.float64),
        np.asarray(positive_hidden, dtype=np.float64),
    )
    if (
        protected_rows.ndim != 2
        or positive_rows.ndim != 2
        or protected_rows.shape[1] != positive_rows.shape[1]
    ):
        raise NullspaceInfeasible(
            "protected and positive matrices have incompatible shapes"
        )
    q0, protected_info = _row_basis(protected_rows)
    projected = positive_rows - (positive_rows @ q0) @ q0.T
    if (
        not len(positive_rows)
        or not np.isfinite(projected).all()
        or np.any(np.linalg.norm(projected, axis=1) <= 1e-3)
    ):
        raise NullspaceInfeasible("positive state is zero after protected projection")
    basis, positive_info = _row_basis(projected)
    # Wide canonical routes previously exposed SVD cancellation around 1e-9.
    # Reprojecting the same span is numerical stabilization, not a new surface.
    for _ in range(3):
        basis = basis - q0 @ (q0.T @ basis)
        basis, _r = np.linalg.qr(basis, mode="reduced")
    null_max = float(np.max(np.abs(protected_rows @ basis), initial=0.0))
    if not 1 <= basis.shape[1] <= MAX_RANK or null_max > PROTECTED_NULL_TOLERANCE:
        raise NullspaceInfeasible("protected-null basis rank or tolerance failed")
    return basis, {
        "protected": protected_info,
        "projected_positive": positive_info,
        "rank": int(basis.shape[1]),
        "protected_null_max_abs": null_max,
        "basis_sha256": _tensor_sha256(basis),
        "reorthogonalization_passes": 3,
    }


def _capture_g46(
    *,
    model: Any,
    base_head: Any,
    native_inputs: Mapping[str, Any],
    route: Sequence[int],
    pad_token_id: int,
) -> tuple[list[int], list[dict[str, Any]], np.ndarray, np.ndarray, dict[str, Any]]:
    ordinary = _greedy_route(model, native_inputs, pad_token_id)
    ordinary_capture = _capture_route(
        model=model,
        output_head=base_head,
        native_inputs=native_inputs,
        route_tokens=ordinary,
        pad_token_id=pad_token_id,
    )
    candidate = _capture_route(
        model=model,
        output_head=base_head,
        native_inputs=native_inputs,
        route_tokens=route,
        pad_token_id=pad_token_id,
    )
    top1 = candidate["logits"].argmax(dim=1).tolist()
    states = [
        {
            "stage": 1,
            "position": position,
            "target_token_id": int(target),
            "hidden": candidate["hidden"][position].double().numpy(),
            "logits": candidate["logits"][position],
        }
        for position, (target, actual) in enumerate(zip(route, top1, strict=True))
        if int(target) != int(actual)
    ]
    if len(states) > MAX_POSITIVES:
        raise _hold(
            "positive state count exceeds explicit deterministic resource bound"
        )
    protected_rows, selection = _protected_hidden_matrix(
        ordinary_route=ordinary,
        ordinary_hidden=ordinary_capture["hidden"],
        controlled_route=route,
        controlled_hidden=candidate["hidden"],
        controlled_positive_positions=[int(item["position"]) for item in states],
    )
    if not states:
        raise _hold("G46 has no non-top1 positions; one-QP program is undefined")
    basis, info = _protected_positive_basis(
        protected_rows, np.stack([item["hidden"] for item in states])
    )
    info.update(
        {
            "selection": selection,
            "protected_hidden_sha256": _tensor_sha256(protected_rows),
            "protected_hidden_shape": list(protected_rows.shape),
            "protected_null_tolerance": PROTECTED_NULL_TOLERANCE,
        }
    )
    return ordinary, states, protected_rows, basis, info


def _global_program(
    states: Sequence[Mapping[str, Any]], basis: np.ndarray
) -> tuple[list[int], list[dict[str, Any]]]:
    selected = sorted({int(item["target_token_id"]) for item in states})
    if not selected or len(selected) > MAX_SELECTED_ROWS or basis.shape[1] > MAX_RANK:
        raise _hold(
            "dynamic S or projected rank exceeds explicit deterministic resource bound"
        )
    constraints = _target_only_constraints(states, basis, target_token_ids=selected)
    if (
        len(constraints) > MAX_CONSTRAINTS
        or len(selected) * basis.shape[1] > MAX_VARIABLES
    ):
        raise _hold("global QP dimensions exceed explicit deterministic resource bound")
    return selected, constraints


def _final_gate(
    evaluation: Mapping[str, Any], *, binding: Mapping[str, Any]
) -> dict[str, Any]:
    generic = recursive._node_gate(evaluation, parent_owner_ids=G41_OWNER_LEDGER)
    owners = list(map(str, generic.get("matched_owner_ids", ())))
    descriptions = {
        str(row.get("owner", "")): str(row.get("description", ""))
        for row in binding["target"].get("rows", ())
    }
    receipts = list(
        dict(evaluation.get("ledger", {}))
        .get("matcher", {})
        .get("prediction_receipts", ())
    )
    tokens = list(map(int, evaluation.get("generated_token_ids", ())))
    debt = dict(generic.get("debt", {}))
    if tokens != binding["route_tokens"]:
        debt["exact_g46_route"] = True
    if set(owners) != set(binding["owner_ledger"]) or len(owners) != 46:
        debt["exact_g46_owner_ledger"] = True
    if sum(descriptions.get(owner) == "person" for owner in owners) != 38:
        debt["all_38_persons"] = True
    if sum(descriptions.get(owner) == "tie" for owner in owners) != 8:
        debt["all_8_ties"] = True
    if len(receipts) != 46 or any(
        item.get("strict_match_status") != "matched" for item in receipts
    ):
        debt["strict_46_predictions"] = True
    if tokens[-1:] != [EOS] or EOS in tokens[:-1]:
        debt["natural_terminal_eos"] = True
    return {
        "passed": not debt,
        "debt": debt,
        "person_count": sum(descriptions.get(owner) == "person" for owner in owners),
        "tie_count": sum(descriptions.get(owner) == "tie" for owner in owners),
        "strict_owner_count": len(owners),
        "generic": generic,
    }


def _production_static_gate(
    tokenizer: Any, *, binding: Mapping[str, Any], runtime: Mapping[str, Any]
) -> dict[str, Any]:
    evaluation = _evaluate_route(
        tokenizer=tokenizer,
        tokens=binding["route_tokens"],
        contract=runtime,
        parent_owner_ids=G41_OWNER_LEDGER,
        label="canonical-g46-global-qp-static-admission",
    )
    gate = _final_gate(evaluation, binding=binding)
    return {"passed": gate["passed"], "debt": gate["debt"], "ledger": evaluation["ledger"]}


def _first_divergence(
    expected: Sequence[int], observed: Sequence[int]
) -> dict[str, int | None] | None:
    for position, (left, right) in enumerate(zip(expected, observed)):
        if int(left) != int(right):
            return {
                "position": position,
                "expected_token_id": int(left),
                "observed_token_id": int(right),
            }
    if len(expected) != len(observed):
        position = min(len(expected), len(observed))
        return {
            "position": position,
            "expected_token_id": int(expected[position])
            if position < len(expected)
            else None,
            "observed_token_id": int(observed[position])
            if position < len(observed)
            else None,
        }
    return None


def _save_payload(
    output: Path,
    *,
    selected: Sequence[int],
    rows: np.ndarray,
    solution: Mapping[str, Any],
    capture: Mapping[str, Any],
    binding: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        list(selected) != sorted(set(selected))
        or rows.shape != (len(selected), 2048)
        or not np.isfinite(rows).all()
    ):
        raise _hold("refusing malformed direct residual payload")
    payload = output / "g46_global_qp_protected_null_output_residual.safetensors"
    save_file(
        {
            "selected_token_ids": torch.tensor(selected, dtype=torch.int64),
            "residual_rows": torch.from_numpy(rows),
        },
        payload,
        metadata={"schema_version": SCHEMA_VERSION, "unit_id": UNIT_ID},
    )
    identity = _residual_identity(selected, rows) | {"payload_sha256": _sha256(payload)}
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "runner_sha256": _sha256(Path(__file__)),
        "target_sha256": TARGET_SHA256,
        "route_sha256": G46_ROUTE_SHA256,
        "selected_token_ids": list(selected),
        "minimum_normalized_norm": float(solution["normalized_norm"]),
        "margin": MARGIN,
        "protected_null_tolerance": PROTECTED_NULL_TOLERANCE,
        "capture": deepcopy(dict(capture)),
        "identity": identity,
        "checkpoint_readback": runtime["checkpoint_readback"],
    }
    metadata_path = payload.with_suffix(".json")
    _atomic_json(metadata_path, metadata)
    return {
        "payload": str(payload),
        "metadata": str(metadata_path),
        "identity": identity,
    }


def _load_payload(
    payload: Path, *, binding: Mapping[str, Any], runtime: Mapping[str, Any]
) -> tuple[list[int], np.ndarray, dict[str, Any]]:
    metadata_path = payload.with_suffix(".json")
    if not payload.is_file() or not metadata_path.is_file():
        raise _hold("cold payload set is incomplete")
    tensors, metadata = (
        load_file(payload, device="cpu"),
        _json(metadata_path, label="payload metadata"),
    )
    ids, rows = tensors.get("selected_token_ids"), tensors.get("residual_rows")
    if (
        ids is None
        or rows is None
        or ids.dtype != torch.int64
        or rows.dtype != torch.float64
        or rows.ndim != 2
    ):
        raise _hold("cold payload tensor schema is invalid")
    selected, array = list(map(int, ids.tolist())), rows.numpy()
    identity = _residual_identity(selected, array) | {
        "payload_sha256": _sha256(payload)
    }
    if (
        selected != sorted(set(selected))
        or not np.isfinite(array).all()
        or metadata.get("schema_version") != SCHEMA_VERSION
        or metadata.get("unit_id") != UNIT_ID
        or metadata.get("runner_sha256") != _sha256(Path(__file__))
        or metadata.get("target_sha256") != TARGET_SHA256
        or metadata.get("route_sha256") != G46_ROUTE_SHA256
        or metadata.get("selected_token_ids") != selected
        or metadata.get("identity") != identity
        or metadata.get("checkpoint_readback") != runtime["checkpoint_readback"]
    ):
        raise _hold("cold direct payload binding drifted")
    return selected, array, metadata


def _cold_verify(*, payload: Path, result: Path) -> None:
    if result.exists():
        raise _hold(f"refusing overwrite: {result}")
    _require_one_gpu()
    binding = _binding_contract()
    runtime = _runtime_contract(binding)
    selected, rows, metadata = _load_payload(payload, binding=binding, runtime=runtime)
    setup = runtime["setup"]
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    with base.open_backend_session(setup["frontend"].launch) as opened:
        if (
            type(opened) is not base.HFBackendSession
            or opened._model is None
            or opened._tokenizer is None
        ):
            raise _hold("cold verification requires FP32 HF backend")
        model, tokenizer = opened._model, opened._tokenizer
        model.eval()
        native_inputs, _prompts, _grids, _media = opened._materialize_native_inputs(
            setup["requests"][:1]
        )
        names, parameters, surface_before, frozen_before = _surface_snapshot(model)
        sentinels = full_root._full_root_nontrainable_sentinels(model)
        base_head = model.get_output_embeddings()
        ordinary, states, protected_rows, basis, capture = _capture_g46(
            model=model,
            base_head=base_head,
            native_inputs=native_inputs,
            route=binding["route_tokens"],
            pad_token_id=int(tokenizer.pad_token_id),
        )
        if (
            selected != sorted({int(item["target_token_id"]) for item in states})
            or metadata["capture"]["positive_positions"]
            != [int(item["position"]) for item in states]
            or metadata["capture"]["protected_hidden_sha256"]
            != _tensor_sha256(protected_rows)
            or metadata["capture"]["basis_sha256"] != _tensor_sha256(basis)
        ):
            raise _hold("cold G46 state/S/protected/basis recapture drifted")
        null_max = float(np.max(np.abs(protected_rows @ rows.T), initial=0.0))
        if null_max > PROTECTED_NULL_TOLERANCE:
            raise _hold("cold residual violates protected-null tolerance")
        violation = _full_vocab_violation(
            states, selected_token_ids=selected, residual_rows=rows
        )
        if violation is not None:
            raise _hold("cold residual fails exhaustive full-vocabulary recheck")
        wrapper = SparseOutputResidual(base_head, selected, torch.from_numpy(rows)).to(
            next(model.parameters()).device
        )
        model.set_output_embeddings(wrapper)
        route = _greedy_route(model, native_inputs, int(tokenizer.pad_token_id))
        evaluation = _evaluate_route(
            tokenizer=tokenizer,
            tokens=route,
            contract=runtime,
            parent_owner_ids=G41_OWNER_LEDGER,
            label="canonical-g46-global-qp-cold",
        )
        gate = _final_gate(evaluation, binding=binding)
        model.set_output_embeddings(base_head)
        full_root._assert_full_root_sentinels(model, sentinels)
        names_after, parameters_after, surface_after, frozen_after = _surface_snapshot(
            model
        )
        if (
            names != names_after
            or any(a is not b for a, b in zip(parameters, parameters_after, strict=True))
            or surface_before != surface_after
            or frozen_before != frozen_after
        ):
            raise _hold("cold residual changed base parameter identity")
        runtime = opened.receipt.to_artifact_dict()
    _atomic_json(
        result,
        {
            "schema_version": SCHEMA_VERSION,
            "unit_id": UNIT_ID,
            "status": "cold_verification_complete",
            "base_ordinary_route_sha256": token_ids_sha256(ordinary),
            "generated_token_ids": route,
            "generated_token_ids_sha256": token_ids_sha256(route),
            "first_divergence": _first_divergence(binding["route_tokens"], route),
            "ledger": evaluation["ledger"],
            "gate": gate,
            "residual_identity": _residual_identity(selected, rows)
            | {"payload_sha256": _sha256(payload)},
            "positive_positions": [int(item["position"]) for item in states],
            "protected_hidden_sha256": _tensor_sha256(protected_rows),
            "basis_sha256": _tensor_sha256(basis),
            "protected_null_max_abs": null_max,
            "protected_null_tolerance": PROTECTED_NULL_TOLERANCE,
            "surface": {"before": surface_before, "after": surface_after},
            "frozen_surface": {"before": frozen_before, "after": frozen_after},
            "runtime": runtime,
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
            "model_load_count": 1,
            "wrapper_count": 1,
        },
    )


def _prepare_output(output: Path) -> dict[str, Any]:
    if output.exists():
        raise _hold(f"refusing overwrite: {output}")
    output.mkdir(parents=True)
    snapshot = output / "runner_source.py"
    source = Path(__file__).read_bytes()
    snapshot.write_bytes(source)
    if snapshot.read_bytes() != source:
        raise _hold("runner snapshot readback drifted")
    return {"path": str(snapshot), "sha256": _sha256(snapshot)}


def _cuda_metric(name: str) -> int:
    return int(getattr(torch.cuda, name)(0)) if torch.cuda.is_available() else 0


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
            "base_greedy": 0,
            "warm_wrapper_count": 0,
        },
    }
    phase, terminal, model, base_head = "binding", None, None, None
    try:
        binding = _binding_contract()
        receipt["bindings"] = _binding_receipt(binding)
        receipt["static_candidate"] = {
            "gate": binding["static_gate"],
            "route_sha256": G46_ROUTE_SHA256,
        }
        if not binding["static_gate"]["passed"]:
            terminal = "static_route_hold"
            raise _hold("canonical G46 production static gate failed")
        _require_one_gpu()
        runtime = _runtime_contract(binding)
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
        setup = runtime["setup"]
        phase = "single_warm_model_load"
        with base.open_backend_session(setup["frontend"].launch) as opened:
            if (
                type(opened) is not base.HFBackendSession
                or opened._model is None
                or opened._tokenizer is None
            ):
                raise _hold("G46 requires FP32 HF backend")
            receipt["counts"]["model_loads"] = 1
            model, tokenizer = opened._model, opened._tokenizer
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            native_inputs, _prompts, _grids, _media = opened._materialize_native_inputs(
                setup["requests"][:1]
            )
            production_static = _production_static_gate(
                tokenizer, binding=binding, runtime=runtime
            )
            receipt["static_candidate"]["production_parser_global_matcher"] = (
                production_static
            )
            if not production_static["passed"]:
                terminal = "static_route_hold"
                raise _hold("canonical G46 production parser/global matcher gate failed")
            names, parameters, surface_before, frozen_before = _surface_snapshot(model)
            sentinels = full_root._full_root_nontrainable_sentinels(model)
            base_head = model.get_output_embeddings()
            phase = "capture_frozen_g46"
            ordinary, states, protected_rows, basis, capture = _capture_g46(
                model=model,
                base_head=base_head,
                native_inputs=native_inputs,
                route=binding["route_tokens"],
                pad_token_id=int(tokenizer.pad_token_id),
            )
            receipt["counts"]["base_greedy"] = 1
            selected, constraints = _global_program(states, basis)
            capture.update(
                {
                    "ordinary_route_sha256": token_ids_sha256(ordinary),
                    "positive_positions": [int(item["position"]) for item in states],
                    "positive_target_token_ids": [
                        int(item["target_token_id"]) for item in states
                    ],
                    "selected_token_ids": selected,
                }
            )
            phase = "one_global_qp"
            solved = _solve_target_only_minimum_normalized(
                constraints,
                rank=basis.shape[1],
                row_norms=_row_norms(base_head, selected),
                target_token_ids=selected,
            )
            receipt["counts"]["solves"] = 1
            receipt["global_program"] = {
                "P": len(states),
                "S": len(selected),
                "rank": int(basis.shape[1]),
                "variables": len(selected) * basis.shape[1],
                "constraints": len(constraints),
                "constraints_compact": qp._compact_constraints(constraints),
                "solver": {
                    key: value
                    for key, value in solved.items()
                    if key not in {"normalized_rows", "scaled_basis_rows"}
                },
            }
            if not solved.get("feasible"):
                terminal = (
                    "certified_infeasible"
                    if solved.get("solver_classification") == "certified_infeasible"
                    else "numerical_solver_hold"
                )
                raise _hold("one global QP did not produce a finite feasible primal")
            if (
                solved.get("selected_token_ids") != selected
                or int(solved.get("variable_count", -1))
                != len(selected) * basis.shape[1]
            ):
                raise _hold("one global QP escaped frozen S/rank dimensions")
            rows = _collapsed_rows(solved, basis)
            null_max = float(np.max(np.abs(protected_rows @ rows.T), initial=0.0))
            receipt["global_program"].update(
                {
                    "minimum_normalized_norm": float(solved["normalized_norm"]),
                    "protected_null_max_abs": null_max,
                    "protected_null_tolerance": PROTECTED_NULL_TOLERANCE,
                }
            )
            if null_max > PROTECTED_NULL_TOLERANCE:
                terminal = "nullspace_infeasible"
                raise _hold("direct global residual violates protected-null tolerance")
            violation = _full_vocab_violation(
                states, selected_token_ids=selected, residual_rows=rows
            )
            receipt["global_program"]["full_vocab"] = {"passed": violation is None}
            if violation is not None:
                terminal = "runtime_margin_hold"
                raise _hold(
                    "direct global residual failed exhaustive full-vocabulary pass"
                )
            phase = "one_warm_ordinary_greedy"
            wrapper = SparseOutputResidual(
                base_head, selected, torch.from_numpy(rows)
            ).to(next(model.parameters()).device)
            model.set_output_embeddings(wrapper)
            receipt["counts"]["warm_wrapper_count"] = 1
            route = _greedy_route(model, native_inputs, int(tokenizer.pad_token_id))
            receipt["counts"]["warm_candidates"] = 1
            evaluation = _evaluate_route(
                tokenizer=tokenizer,
                tokens=route,
                contract=runtime,
                parent_owner_ids=G41_OWNER_LEDGER,
                label="canonical-g46-global-qp-warm",
            )
            gate = _final_gate(evaluation, binding=binding)
            warm = {
                "generated_token_ids": route,
                "generated_token_ids_sha256": token_ids_sha256(route),
                "first_divergence": _first_divergence(binding["route_tokens"], route),
                "ledger": deepcopy(dict(evaluation["ledger"])),
                "gate": gate,
                "positive_positions": capture["positive_positions"],
                "protected_hidden_sha256": _tensor_sha256(protected_rows),
                "basis_sha256": _tensor_sha256(basis),
                "protected_null_max_abs": null_max,
            }
            receipt["warm_final"] = warm
            if not gate["passed"]:
                terminal = "canonical_greedy_negative"
                raise _hold("sole warm ordinary greedy failed canonical G46 gate")
            payload = _save_payload(
                output,
                selected=selected,
                rows=rows,
                solution=solved,
                capture=capture | {"basis_sha256": _tensor_sha256(basis)},
                binding=binding,
                runtime=runtime,
            )
            warm["residual_identity"] = payload["identity"]
            receipt["payload"] = payload
            model.set_output_embeddings(base_head)
            full_root._assert_full_root_sentinels(model, sentinels)
            names_after, parameters_after, surface_after, frozen_after = (
                _surface_snapshot(model)
            )
            if (
                names != names_after
                or any(
                    a is not b
                    for a, b in zip(parameters, parameters_after, strict=True)
                )
                or surface_before != surface_after
                or frozen_before != frozen_after
            ):
                raise _hold("warm residual changed frozen model surface")
            receipt["warm_surface"] = {
                "before": surface_before,
                "after": surface_after,
                "frozen_before": frozen_before,
                "frozen_after": frozen_after,
            }
            receipt["runtime"] = {"warm": opened.receipt.to_artifact_dict()}
        phase = "fresh_subprocess_cold"
        model = base_head = None
        gc.collect()
        torch.cuda.empty_cache()
        cold_path = output / "cold_verification.json"
        repo = str(Path(__file__).resolve().parents[2])
        env = dict(os.environ)
        env["PYTHONPATH"] = repo + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        completed = subprocess.run(
            [
                sys.executable,
                str(snapshot["path"]),
                "--cold-verify",
                "--payload",
                str(payload["payload"]),
                "--cold-result",
                str(cold_path),
            ],
            cwd=repo,
            env=env,
            text=True,
            capture_output=True,
            timeout=max(
                1.0,
                RESOURCE_BOUND["wall_time_seconds_max"] - (time.monotonic() - started),
            ),
            check=False,
        )
        if completed.returncode != 0 or not cold_path.is_file():
            raise _hold(
                "fresh cold subprocess failed: "
                + (
                    completed.stderr[-2000:]
                    or completed.stdout[-2000:]
                    or str(completed.returncode)
                )
            )
        cold = _json(cold_path, label="cold verification")
        receipt["counts"]["model_loads"] = 2
        receipt["cold_final"] = cold
        receipt["runtime"]["cold"] = cold.get("runtime")
        parity = {
            "route_exact": warm["generated_token_ids"]
            == cold.get("generated_token_ids"),
            "ledger_exact": warm["ledger"] == cold.get("ledger"),
            "identity_exact": warm["residual_identity"]
            == cold.get("residual_identity"),
            "positions_exact": warm["positive_positions"]
            == cold.get("positive_positions"),
            "protected_exact": warm["protected_hidden_sha256"]
            == cold.get("protected_hidden_sha256"),
            "basis_exact": warm["basis_sha256"] == cold.get("basis_sha256"),
            "warm_gate": gate["passed"] is True,
            "cold_gate": dict(cold.get("gate", {})).get("passed") is True,
            "cold_one_wrapper": cold.get("wrapper_count") == 1,
        }
        receipt["warm_cold"] = parity
        if not all(parity.values()):
            raise _hold("warm/cold parity failed")
        terminal = "cold_greedy_46_owner_success"
    except BaseException as error:
        terminal = terminal or (
            "nullspace_infeasible"
            if isinstance(error, NullspaceInfeasible)
            else "technical_hold"
        )
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
        receipt["status"] = (
            terminal if terminal in TERMINAL_STATUSES else "technical_hold"
        )
        receipt["wall_time_seconds"] = elapsed
        receipt["resources"] = {
            "peak_cuda_allocated_bytes": _cuda_metric("max_memory_allocated"),
            "peak_cuda_reserved_bytes": _cuda_metric("max_memory_reserved"),
            "artifact_bytes_before_receipt": _artifact_bytes(output),
            "predeclared_bound": RESOURCE_BOUND,
        }
        if (
            elapsed > RESOURCE_BOUND["wall_time_seconds_max"]
            or receipt["counts"]["solves"] > 1
            or receipt["counts"]["warm_candidates"] > 1
            or receipt["counts"]["model_loads"] > 2
            or receipt["counts"]["warm_wrapper_count"] > 1
            or receipt["resources"]["peak_cuda_reserved_bytes"]
            > RESOURCE_BOUND["max_peak_cuda_reserved_bytes"]
            or receipt["resources"]["artifact_bytes_before_receipt"]
            > RESOURCE_BOUND["output_artifact_bytes_max"]
        ):
            receipt["status"] = "technical_hold"
        _atomic_json(output / "receipt.json", receipt)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-bindings", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--cold-verify", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--payload", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--cold-result", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.check_bindings:
        if args.run_id or args.cold_verify or args.payload or args.cold_result:
            raise SystemExit(
                "--check-bindings cannot be combined with execution arguments"
            )
        print(json.dumps(_binding_receipt(), indent=2, sort_keys=True))
        return
    if args.cold_verify:
        if args.run_id or args.payload is None or args.cold_result is None:
            raise SystemExit(
                "internal --cold-verify requires --payload and --cold-result"
            )
        _cold_verify(payload=args.payload, result=args.cold_result)
        return
    if not args.run_id or args.payload or args.cold_result:
        raise SystemExit("one-GPU execution requires exactly --run-id")
    print(run(run_id=args.run_id))


if __name__ == "__main__":
    main()
