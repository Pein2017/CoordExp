#!/usr/bin/env python3
"""Distill Image2299 with a protected-null residual on nine target rows only."""

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
from scipy.optimize import minimize
from safetensors.torch import load_file, save_file
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research import run_image2299_protected_null_output_distillation as predecessor


recursive = predecessor.recursive
base = predecessor.base
full_root = predecessor.full_root
token_ids_sha256 = predecessor.token_ids_sha256

SCHEMA_VERSION = "image2299.target_only_protected_null_distillation.v1"
UNIT_ID = "2026-08-30-image2299-target-only-protected-null-distillation"
OUTPUT_ROOT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration") / UNIT_ID
SOURCE_RECEIPT = (
    predecessor.OUTPUT_ROOT
    / "20260830T-image2299-protected-null-output-distillation-v1"
    / "receipt.json"
)
SOURCE_RECEIPT_SHA256 = "d41d0418b6f035a3124334107de86e436f5e94072d832042e59cb430e122e3da"
PREDECESSOR_RUNNER_SHA256 = "d1d64e648f7b0ac473aa5fc0386c6316a8a51a9a9376b14fcbe4fc9a48320463"
TERMINAL_WITNESS_RECEIPT_SHA256 = "d9cb113ae3f7cbe4b080fbe4f8faf76f68d36d9ca2844940b27560d330ab4495"
MODEL_RECEIPT_SHA256 = "a2912cab11797ddab8a556be1c17603d537271981047d40b99ffa6f27515278c"
START_SURFACE_SHA256 = "2ac0e8a963d91ac7a4ba42322272889efd74bd46578c1ac44e16e531d1246676"
FROZEN_SURFACE_SHA256 = "c16f0b52a2d5ce5ccfa1b239a0934a9ed27945278fc65e97d56cb8d50c4877d5"
PROMPT_TOKEN_SHA256 = "33f11458039a9b8c01e1329eeedad91ac68824cc5454c9df4b15e47d50458ffb"
IMAGE_SHA256 = "cd7199a37188c9ac6481520175866cbd78fa6fba35f4290bb0b60c78afcb2df3"
START_CHECKPOINT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-30-image2299-ota-sqp-lite/"
    "20260830T-image2299-ota-sqp-lite-r32-step-v1/checkpoint-selected-r32-step"
)
CONTROLLED_ROUTE_SHA256 = "c89c6f9900303227cf781caf4a39dc203a09a53b0d319930754cda184529530e"
ORDINARY_ROUTE_SHA256 = "5df0ac25aa871ddc0550298e70cd6012ce4dc3b6c6068b97dfa0cddca2f02a67"
PROTECTED_HIDDEN_SHA256 = "d3ba5b61b4783da8c8e1d7973d32c337e3abae96efb2b255c48173bd875a9ead"
BASIS_SHA256 = "f913a3ae539119deb7c61641ae39458e343d7f39404f571b7e37af65907ea438"
POSITIVE_POSITIONS = [301, 324, 328, 342, 346, 351, 355, 360, 364, 365, 367, 369]
TARGET_TOKEN_IDS = [151645, 151646, 151820, 151867, 151935, 152032, 152190, 152242, 152305]
STAGE_POSITIVE_COUNTS = [1, 3, 5, 7, 12]
STAGE_CONSTRAINT_COUNTS = [9, 27, 45, 63, 108]

MARGIN = 0.01
NORM_CAP = 1.0
MAX_POSITIVES = 12
MAX_RANK = 12
MAX_SOLVES = 5
MAX_WARM_CANDIDATES = 5
RESOURCE_BOUND = {
    "gpu_count": 1,
    "world_size": 1,
    "positive_states_max": MAX_POSITIVES,
    "positive_rank_max": MAX_RANK,
    "selected_output_rows": len(TARGET_TOKEN_IDS),
    "constraint_count_max": STAGE_CONSTRAINT_COUNTS[-1],
    "variable_count_max": len(TARGET_TOKEN_IDS) * MAX_RANK,
    "solve_count_max": MAX_SOLVES,
    "warm_candidate_count_max": MAX_WARM_CANDIDATES,
    "model_load_count_max": 2,
    "wall_time_seconds_max": 1_800,
    "max_peak_cuda_reserved_bytes": 16 * 2**30,
    "output_artifact_bytes_max": 100_000_000,
}
TERMINAL_STATUSES = {
    "cold_greedy_38_person_success",
    "target_only_infeasible",
    "norm_cap_exceeded",
    "runtime_margin_hold",
    "stage_exhausted",
    "technical_hold",
}

ProtectedNullHold = predecessor.ProtectedNullHold
SparseOutputResidual = predecessor.SparseOutputResidual
_hold = predecessor._hold
_sha256 = predecessor._sha256
_value_sha256 = predecessor._value_sha256
_tensor_sha256 = predecessor._tensor_sha256
_atomic_json = predecessor._atomic_json
_capture_route = predecessor._capture_route
_protected_positive_basis = predecessor._protected_positive_basis
_protected_hidden_matrix = predecessor._protected_hidden_matrix
_greedy_route = predecessor._greedy_route
_evaluate_route = predecessor._evaluate_route
_stage_gate = predecessor._stage_gate
_row_norms = predecessor._row_norms
_collapsed_rows = predecessor._collapsed_rows
_full_vocab_violation = predecessor._full_vocab_violation
_residual_identity = predecessor._residual_identity
_surface_snapshot = predecessor._surface_snapshot
_artifact_bytes = predecessor._artifact_bytes
_require_one_gpu = predecessor._require_one_gpu


def _load_source_receipt() -> dict[str, Any]:
    if not SOURCE_RECEIPT.is_file() or _sha256(SOURCE_RECEIPT) != SOURCE_RECEIPT_SHA256:
        raise _hold("immutable predecessor receipt drifted")
    try:
        receipt = json.loads(SOURCE_RECEIPT.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise _hold(f"predecessor receipt is unreadable: {error}") from error
    if not isinstance(receipt, dict):
        raise _hold("predecessor receipt is not an object")
    return receipt


def _binding_contract() -> dict[str, Any]:
    receipt = _load_source_receipt()
    capture = dict(receipt.get("capture", {}))
    basis = dict(capture.get("basis", {}))
    source = dict(receipt.get("runner_source_snapshot", {}))
    source_path = Path(str(source.get("path", "")))
    inherited = predecessor._binding_contract()
    if (
        receipt.get("schema_version") != predecessor.SCHEMA_VERSION
        or receipt.get("unit_id") != predecessor.UNIT_ID
        or receipt.get("status") != "stage_exhausted"
        or source.get("sha256") != PREDECESSOR_RUNNER_SHA256
        or not source_path.is_file()
        or _sha256(source_path) != PREDECESSOR_RUNNER_SHA256
        or _sha256(Path(predecessor.__file__)) != PREDECESSOR_RUNNER_SHA256
        or predecessor.SOURCE_RECEIPT_SHA256 != TERMINAL_WITNESS_RECEIPT_SHA256
        or predecessor.MODEL_RECEIPT_SHA256 != MODEL_RECEIPT_SHA256
        or predecessor.START_SURFACE_SHA256 != START_SURFACE_SHA256
        or predecessor.FROZEN_SURFACE_SHA256 != FROZEN_SURFACE_SHA256
        or predecessor.PROMPT_TOKEN_SHA256 != PROMPT_TOKEN_SHA256
        or predecessor.IMAGE_SHA256 != IMAGE_SHA256
        or Path(recursive.START_CHECKPOINT) != START_CHECKPOINT
        or capture.get("ordinary_route_sha256") != ORDINARY_ROUTE_SHA256
        or int(capture.get("ordinary_route_token_count", -1)) != 307
        or capture.get("controlled_route_sha256") != CONTROLLED_ROUTE_SHA256
        or int(capture.get("controlled_route_token_count", -1)) != 370
        or list(capture.get("positive_positions", ())) != POSITIVE_POSITIONS
        or int(capture.get("positive_count", -1)) != MAX_POSITIVES
        or int(capture.get("protected_count", -1)) != 664
        or list(basis.get("protected_hidden_shape", ())) != [664, 2048]
        or int(basis.get("rank", -1)) != MAX_RANK
        or basis.get("protected_hidden_sha256") != PROTECTED_HIDDEN_SHA256
        or basis.get("basis_sha256") != BASIS_SHA256
        or receipt.get("bindings") != predecessor._binding_receipt()
    ):
        raise _hold("predecessor receipt, recapture, or helper source binding drifted")
    return inherited | {"predecessor_receipt": receipt}


def _binding_receipt() -> dict[str, Any]:
    contract = _binding_contract()
    runtime = contract["runtime_contract"]
    return {
        "schema_version": SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "status": "verified",
        "execution_surface": "cpu_only_no_model_load_no_cuda",
        "source_receipt": str(SOURCE_RECEIPT),
        "source_receipt_sha256": SOURCE_RECEIPT_SHA256,
        "predecessor_runner_sha256": PREDECESSOR_RUNNER_SHA256,
        "terminal_witness_receipt_sha256": TERMINAL_WITNESS_RECEIPT_SHA256,
        "model_receipt_sha256": MODEL_RECEIPT_SHA256,
        "checkpoint": str(START_CHECKPOINT),
        "checkpoint_readback_sha256": _value_sha256(runtime["checkpoint_readback"]),
        "start_surface_sha256": START_SURFACE_SHA256,
        "frozen_surface_sha256": FROZEN_SURFACE_SHA256,
        "prompt_token_ids_sha256": PROMPT_TOKEN_SHA256,
        "image_sha256": IMAGE_SHA256,
        "ordinary_route_sha256": ORDINARY_ROUTE_SHA256,
        "ordinary_route_token_count": 307,
        "controlled_route_sha256": CONTROLLED_ROUTE_SHA256,
        "controlled_route_token_count": 370,
        "positive_positions": POSITIVE_POSITIONS,
        "positive_count": MAX_POSITIVES,
        "protected_count": 664,
        "protected_hidden_sha256": PROTECTED_HIDDEN_SHA256,
        "protected_hidden_shape": [664, 2048],
        "basis_sha256": BASIS_SHA256,
        "rank": MAX_RANK,
        "target_token_ids": TARGET_TOKEN_IDS,
        "stage_positive_counts": STAGE_POSITIVE_COUNTS,
        "stage_constraint_counts": STAGE_CONSTRAINT_COUNTS,
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
        raise _hold("successor runner source snapshot readback drifted")
    return {"path": str(snapshot), "sha256": _sha256(snapshot)}


def _target_only_constraints(
    states: Sequence[Mapping[str, Any]], basis: np.ndarray,
    target_token_ids: Sequence[int] = TARGET_TOKEN_IDS,
) -> list[dict[str, Any]]:
    selected = list(map(int, target_token_ids))
    selected_set = set(selected)
    if len(selected) != len(selected_set):
        raise ValueError("target token ids must be unique")
    constraints: list[dict[str, Any]] = []
    for state in states:
        target = int(state["target_token_id"])
        logits = state["logits"]
        if target not in selected_set or logits.ndim != 1 or max(selected) >= logits.numel():
            raise ValueError("state target/logits do not match the fixed target partition")
        feature = np.asarray(state["hidden"], dtype=np.float64) @ np.asarray(basis, dtype=np.float64)
        fixed_logits = logits.clone()
        fixed_logits[torch.tensor(selected, dtype=torch.long)] = -torch.inf
        fixed_competitor = int(fixed_logits.argmax().item())
        for competitor in selected:
            if competitor != target:
                constraints.append({
                    "stage": int(state["stage"]),
                    "position": int(state["position"]),
                    "kind": "movable_target_partition",
                    "target_token_id": target,
                    "competitor_token_id": competitor,
                    "competitor_trainable": True,
                    "raw_margin": float(logits[target].item() - logits[competitor].item()),
                    "feature": feature,
                })
        constraints.append({
            "stage": int(state["stage"]),
            "position": int(state["position"]),
            "kind": "fixed_non_target_partition_max",
            "target_token_id": target,
            "competitor_token_id": fixed_competitor,
            "competitor_trainable": False,
            "raw_margin": float(logits[target].item() - logits[fixed_competitor].item()),
            "feature": feature,
        })
    return constraints


def _solve_target_only_minimum_normalized(
    constraints: Sequence[Mapping[str, Any]], *, rank: int,
    row_norms: Mapping[int, float], target_token_ids: Sequence[int] = TARGET_TOKEN_IDS,
) -> dict[str, Any]:
    selected = list(map(int, target_token_ids))
    selected_set = set(selected)
    if not constraints or rank <= 0 or len(selected) != len(selected_set):
        return {"feasible": False, "reason": "empty_constraint_basis_or_target_set"}
    if set(map(int, row_norms)) != selected_set or min(map(float, row_norms.values())) <= 0.0:
        return {"feasible": False, "reason": "target_row_norm_contract"}
    token_to_row = {token: index for index, token in enumerate(selected)}
    width = len(selected) * rank
    matrix = np.zeros((len(constraints), width), dtype=np.float64)
    rhs = np.empty(len(constraints), dtype=np.float64)
    for index, item in enumerate(constraints):
        feature = np.asarray(item["feature"], dtype=np.float64)
        target = int(item["target_token_id"])
        competitor = int(item["competitor_token_id"])
        trainable = bool(item.get("competitor_trainable"))
        kind = str(item.get("kind", ""))
        if feature.shape != (rank,) or target not in selected_set or target == competitor:
            return {"feasible": False, "reason": "malformed_constraint"}
        if trainable != (competitor in selected_set) or (
            kind == "fixed_non_target_partition_max" and trainable
        ) or (kind == "movable_target_partition" and not trainable):
            return {"feasible": False, "reason": "competitor_partition_contract"}
        start = token_to_row[target] * rank
        matrix[index, start : start + rank] += float(row_norms[target]) * feature
        if trainable:
            start = token_to_row[competitor] * rank
            matrix[index, start : start + rank] -= float(row_norms[competitor]) * feature
        rhs[index] = MARGIN - float(item["raw_margin"])
    impossible = (np.linalg.norm(matrix, axis=1) <= 1.0e-14) & (rhs > 1.0e-9)
    if np.any(impossible) or not np.isfinite(matrix).all() or not np.isfinite(rhs).all():
        return {"feasible": False, "reason": "zero_feature_positive_rhs"}

    gram = matrix @ matrix.T

    def objective(lam: np.ndarray) -> tuple[float, np.ndarray]:
        gradient = gram @ lam - rhs
        return float(0.5 * lam @ gram @ lam - rhs @ lam), gradient

    outcome = minimize(
        objective,
        np.zeros(len(constraints), dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        bounds=[(0.0, None)] * len(constraints),
        options={"ftol": 1.0e-15, "gtol": 1.0e-10, "maxiter": 20_000, "maxls": 100},
    )
    normalized = matrix.T @ np.asarray(outcome.x, dtype=np.float64)
    slacks = matrix @ normalized - rhs
    if not np.isfinite(normalized).all() or float(slacks.min(initial=math.inf)) < -2.0e-7:
        return {
            "feasible": False,
            "reason": "dual_solver_did_not_produce_feasible_primal",
            "solver_message": str(outcome.message),
            "minimum_slack": float(slacks.min(initial=math.inf)),
        }
    normalized_rows = normalized.reshape(len(selected), rank)
    return {
        "feasible": True,
        "selected_token_ids": selected,
        "normalized_rows": normalized_rows,
        "scaled_basis_rows": np.stack([
            normalized_rows[index] * float(row_norms[token])
            for index, token in enumerate(selected)
        ]),
        "normalized_norm": float(np.linalg.norm(normalized)),
        "minimum_slack": float(slacks.min(initial=math.inf)),
        "constraint_count": len(constraints),
        "variable_count": width,
        "solver_success": bool(outcome.success),
        "solver_message": str(outcome.message),
        "iterations": int(outcome.nit),
    }


def _compact_constraints(constraints: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in item.items() if key != "feature"}
        | {"feature_sha256": _tensor_sha256(np.asarray(item["feature"], dtype=np.float64))}
        for item in constraints
    ]


def _save_augmented_payload(
    output: Path, *, residual_rows: np.ndarray, solution: Mapping[str, Any],
    basis_info: Mapping[str, Any], contract: Mapping[str, Any],
) -> dict[str, Any]:
    token_ids = list(map(int, solution["selected_token_ids"]))
    if token_ids != TARGET_TOKEN_IDS:
        raise _hold("refusing to persist a residual outside the nine fixed target rows")
    payload = output / "target_only_protected_null_output_residual.safetensors"
    save_file(
        {
            "selected_token_ids": torch.tensor(token_ids, dtype=torch.int64),
            "residual_rows": torch.from_numpy(np.asarray(residual_rows, dtype=np.float64)),
        },
        payload,
        metadata={"schema_version": SCHEMA_VERSION, "unit_id": UNIT_ID},
    )
    identity = _residual_identity(token_ids, residual_rows)
    identity["payload_sha256"] = _sha256(payload)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "base_checkpoint": str(recursive.START_CHECKPOINT),
        "source_receipt_sha256": SOURCE_RECEIPT_SHA256,
        "controlled_route_sha256": CONTROLLED_ROUTE_SHA256,
        "selected_output_rows_contract": "exactly the fixed nine target-token rows",
        "normalization": "per-selected-base-output-row L2; global normalized Frobenius cap",
        "normalized_norm": float(solution["normalized_norm"]),
        "cap": NORM_CAP,
        "basis": deepcopy(dict(basis_info)),
        "residual_identity": identity,
    }
    metadata_path = payload.with_suffix(".json")
    _atomic_json(metadata_path, metadata)
    manifest = {
        "schema_version": f"{SCHEMA_VERSION}.augmented_checkpoint_manifest",
        "unit_id": UNIT_ID,
        "base_checkpoint": str(recursive.START_CHECKPOINT),
        "base_checkpoint_readback": contract["runtime_contract"]["checkpoint_readback"],
        "base_weight_payload_mode": "reference_only_no_copy_no_mutation",
        "residual_payload": str(payload),
        "residual_payload_sha256": identity["payload_sha256"],
        "residual_metadata": str(metadata_path),
        "residual_metadata_sha256": _sha256(metadata_path),
        "load_order": ["load frozen base checkpoint", "install output-head residual module", "load residual payload"],
    }
    manifest_path = output / "augmented_checkpoint_manifest.json"
    _atomic_json(manifest_path, manifest)
    return {
        "payload": str(payload),
        "metadata": str(metadata_path),
        "manifest": str(manifest_path),
        "identity": identity,
    }


def _recapture(
    *, model: Any, tokenizer: Any, native_inputs: Mapping[str, Any],
    binding: Mapping[str, Any], base_head: torch.nn.Module,
) -> tuple[list[int], dict[str, torch.Tensor], dict[str, torch.Tensor], list[dict[str, Any]], np.ndarray, np.ndarray, dict[str, Any]]:
    pad = int(tokenizer.pad_token_id)
    ordinary_route = _greedy_route(model, native_inputs, pad)
    ordinary = _capture_route(
        model=model, output_head=base_head, native_inputs=native_inputs,
        route_tokens=ordinary_route, pad_token_id=pad,
    )
    controlled_tokens = binding["controlled_route_tokens"]
    controlled = _capture_route(
        model=model, output_head=base_head, native_inputs=native_inputs,
        route_tokens=controlled_tokens, pad_token_id=pad,
    )
    if (
        len(ordinary_route) != 307
        or token_ids_sha256(ordinary_route) != ORDINARY_ROUTE_SHA256
        or len(controlled_tokens) != 370
        or token_ids_sha256(controlled_tokens) != CONTROLLED_ROUTE_SHA256
        or ordinary["logits"].argmax(dim=1).tolist() != ordinary_route
    ):
        raise _hold("ordinary/controlled route recapture drifted")
    intervention_stage = {
        position: int(block["stage"])
        for block in binding["blocks"] for position in block["positions"]
    }
    positive_states = []
    controlled_top1 = controlled["logits"].argmax(dim=1).tolist()
    for position, (target, top1) in enumerate(zip(controlled_tokens, controlled_top1, strict=True)):
        if position in intervention_stage and int(target) != int(top1):
            positive_states.append({
                "stage": intervention_stage[position],
                "position": position,
                "target_token_id": int(target),
                "hidden": controlled["hidden"][position].double().numpy(),
                "logits": controlled["logits"][position],
            })
        elif position not in intervention_stage and int(target) != int(top1):
            raise _hold("controlled route has an unrecorded non-top1 intervention")
    if [item["position"] for item in positive_states] != POSITIVE_POSITIONS:
        raise _hold("positive-position recapture drifted")
    protected, selection = _protected_hidden_matrix(
        ordinary_route=ordinary_route,
        ordinary_hidden=ordinary["hidden"],
        controlled_route=controlled_tokens,
        controlled_hidden=controlled["hidden"],
        controlled_positive_positions=POSITIVE_POSITIONS,
    )
    positive = np.stack([item["hidden"] for item in positive_states])
    basis, basis_info = _protected_positive_basis(protected, positive)
    if (
        len(protected) != 664
        or _tensor_sha256(protected) != PROTECTED_HIDDEN_SHA256
        or basis.shape != (2048, MAX_RANK)
        or _tensor_sha256(basis) != BASIS_SHA256
    ):
        raise _hold("protected-state or basis recapture drifted")
    basis_info.update({
        "protected_hidden_sha256": PROTECTED_HIDDEN_SHA256,
        "protected_hidden_shape": [664, 2048],
        "protected_null_apply_contract": (
            "FP64 payload; hidden cast FP64; hidden@D.T FP64; correction cast to logits dtype"
        ),
        "protected_null_tolerance": 1.0e-10,
    })
    return ordinary_route, ordinary, controlled, positive_states, protected, basis, basis_info | {"selection": selection}


def _cold_verify(*, payload: Path, result: Path) -> None:
    if result.exists():
        raise _hold(f"refusing overwrite: {result}")
    _require_one_gpu()
    binding = _binding_contract()
    contract = binding["runtime_contract"]
    setup = contract["setup"]
    tensors = load_file(payload, device="cpu")
    metadata_path = payload.with_suffix(".json")
    if not metadata_path.is_file():
        raise _hold("cold residual metadata is missing")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    token_ids = tensors.get("selected_token_ids")
    rows = tensors.get("residual_rows")
    actual_identity = None if token_ids is None or rows is None else _residual_identity(token_ids.tolist(), rows)
    if actual_identity is not None:
        actual_identity["payload_sha256"] = _sha256(payload)
    if (
        metadata.get("schema_version") != SCHEMA_VERSION
        or metadata.get("unit_id") != UNIT_ID
        or metadata.get("source_receipt_sha256") != SOURCE_RECEIPT_SHA256
        or metadata.get("controlled_route_sha256") != CONTROLLED_ROUTE_SHA256
        or dict(metadata.get("basis", {})).get("basis_sha256") != BASIS_SHA256
        or dict(metadata.get("basis", {})).get("protected_hidden_sha256") != PROTECTED_HIDDEN_SHA256
        or token_ids is None
        or token_ids.tolist() != TARGET_TOKEN_IDS
        or rows is None
        or tuple(rows.shape) != (len(TARGET_TOKEN_IDS), 2048)
        or rows.dtype != torch.float64
        or not bool(torch.isfinite(rows).all())
        or metadata.get("residual_identity") != actual_identity
    ):
        raise _hold("cold successor residual payload/schema/selected-row contract is invalid")
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    with base.open_backend_session(setup["frontend"].launch) as opened:
        if type(opened) is not base.HFBackendSession or opened._model is None or opened._tokenizer is None:
            raise _hold("cold verification requires concrete FP32 HF backend")
        model, tokenizer = opened._model, opened._tokenizer
        model.eval()
        native_inputs, prompts, _grids, _media = opened._materialize_native_inputs(setup["requests"][:1])
        if token_ids_sha256(prompts[0]) != PROMPT_TOKEN_SHA256:
            raise _hold("cold prompt token identity drifted")
        names, parameters, surface_before, frozen_before = _surface_snapshot(model)
        sentinels = full_root._full_root_nontrainable_sentinels(model)
        base_head = model.get_output_embeddings()
        ordinary_route, _ordinary, _controlled, _states, protected, basis, basis_info = _recapture(
            model=model, tokenizer=tokenizer, native_inputs=native_inputs,
            binding=binding, base_head=base_head,
        )
        protected_null_max_abs = float(np.max(np.abs(protected @ rows.double().numpy().T), initial=0.0))
        if protected_null_max_abs > 1.0e-10:
            raise _hold("cold residual exceeds numerical protected-null bound 1e-10")
        wrapper = SparseOutputResidual(base_head, TARGET_TOKEN_IDS, rows).to(next(model.parameters()).device)
        model.set_output_embeddings(wrapper)
        route = _greedy_route(model, native_inputs, int(tokenizer.pad_token_id))
        evaluation = _evaluate_route(
            tokenizer=tokenizer,
            tokens=route,
            contract=contract,
            parent_owner_ids=contract["parent_owner_ids"],
            label="target-only-protected-null-final",
        )
        gate = _stage_gate(evaluation, block=binding["blocks"][-1])
        residual_identity = _residual_identity(TARGET_TOKEN_IDS, rows)
        residual_identity["payload_sha256"] = _sha256(payload)
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
        "base_ordinary_route_sha256": token_ids_sha256(ordinary_route),
        "generated_token_ids": route,
        "generated_token_ids_sha256": token_ids_sha256(route),
        "ledger": evaluation["ledger"],
        "gate": gate,
        "residual_identity": residual_identity,
        "protected_hidden_sha256": PROTECTED_HIDDEN_SHA256,
        "basis_sha256": basis_info["basis_sha256"],
        "protected_null_max_abs": protected_null_max_abs,
        "protected_null_tolerance": 1.0e-10,
        "surface": {"before": surface_before, "after": surface_after},
        "frozen_surface": {"before": frozen_before, "after": frozen_after},
        "runtime": runtime,
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        "model_load_count": 1,
    })


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
        "counts": {"model_loads": 0, "solves": 0, "warm_candidates": 0, "base_greedy": 0, "zero_wrapper_greedy": 0},
        "stages": [],
    }
    stage = "single_warm_model_load"
    terminal_status: str | None = None
    warm_final: dict[str, Any] | None = None
    payload: dict[str, Any] | None = None
    model = wrapper = base_head = opened = tokenizer = native_inputs = prompts = None
    try:
        _require_one_gpu()
        binding = _binding_contract()
        contract = binding["runtime_contract"]
        receipt["bindings"] = _binding_receipt()
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
        setup = contract["setup"]
        with base.open_backend_session(setup["frontend"].launch) as opened:
            if type(opened) is not base.HFBackendSession or opened._model is None or opened._tokenizer is None:
                raise _hold("distillation requires concrete FP32 HF backend")
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

            stage = "capture_raw_ordinary_and_controlled"
            ordinary_route, _ordinary, controlled, positive_states, protected, basis, basis_info = _recapture(
                model=model, tokenizer=tokenizer, native_inputs=native_inputs,
                binding=binding, base_head=base_head,
            )
            receipt["counts"]["base_greedy"] = 1
            wrapper = SparseOutputResidual(
                base_head,
                TARGET_TOKEN_IDS,
                torch.zeros((len(TARGET_TOKEN_IDS), controlled["hidden"].shape[1]), dtype=torch.float64),
            ).to(next(model.parameters()).device)
            model.set_output_embeddings(wrapper)
            zero_controlled = _capture_route(
                model=model, output_head=wrapper, native_inputs=native_inputs,
                route_tokens=binding["controlled_route_tokens"], pad_token_id=int(tokenizer.pad_token_id),
            )
            zero_route = _greedy_route(model, native_inputs, int(tokenizer.pad_token_id))
            receipt["counts"]["zero_wrapper_greedy"] = 1
            if (
                zero_route != ordinary_route
                or not torch.equal(zero_controlled["logits"], controlled["logits"])
                or not torch.equal(zero_controlled["hidden"], controlled["hidden"])
            ):
                raise _hold("zero target-only residual failed hidden/logit/generate parity")
            receipt["capture"] = {
                "ordinary_route_sha256": ORDINARY_ROUTE_SHA256,
                "ordinary_route_token_count": len(ordinary_route),
                "controlled_route_sha256": CONTROLLED_ROUTE_SHA256,
                "controlled_route_token_count": len(binding["controlled_route_tokens"]),
                "positive_count": len(positive_states),
                "positive_positions": POSITIVE_POSITIONS,
                "protected_count": len(protected),
                "protected_selection": basis_info.pop("selection"),
                "basis": basis_info,
                "target_token_ids": TARGET_TOKEN_IDS,
                "zero_wrapper_hidden_logits_generate_exact": True,
            }

            stage = "five_target_only_solves"
            row_norms = _row_norms(base_head, TARGET_TOKEN_IDS)
            final_solution = None
            final_rows = None
            for stage_number, block in enumerate(binding["blocks"], start=1):
                states = [item for item in positive_states if item["stage"] <= stage_number]
                constraints = _target_only_constraints(states, basis)
                expected_positive = STAGE_POSITIVE_COUNTS[stage_number - 1]
                expected_constraints = STAGE_CONSTRAINT_COUNTS[stage_number - 1]
                if len(states) != expected_positive or len(constraints) != expected_constraints:
                    raise _hold(f"stage {stage_number} cumulative state/constraint count drifted")
                solved = _solve_target_only_minimum_normalized(
                    constraints, rank=MAX_RANK, row_norms=row_norms,
                )
                receipt["counts"]["solves"] += 1
                stage_record = {
                    "stage": stage_number,
                    "expected_route_sha256": block["expected_route_sha256"],
                    "positive_count": len(states),
                    "constraint_count": len(constraints),
                    "constraints": _compact_constraints(constraints),
                    "solver": {key: value for key, value in solved.items() if key not in {"normalized_rows", "scaled_basis_rows"}},
                }
                receipt["stages"].append(stage_record)
                if not solved.get("feasible"):
                    terminal_status = "target_only_infeasible"
                    raise _hold(f"stage {stage_number} target-only inequalities are infeasible")
                if solved["selected_token_ids"] != TARGET_TOKEN_IDS or solved["variable_count"] != 108:
                    raise _hold("target-only solve escaped the exact 108-variable nine-row surface")
                if float(solved["normalized_norm"]) > NORM_CAP + 1.0e-8:
                    terminal_status = "norm_cap_exceeded"
                    raise _hold(f"stage {stage_number} minimum normalized norm exceeds 1")
                rows = _collapsed_rows(solved, basis)
                null_max = float(np.max(np.abs(protected @ rows.T), initial=0.0))
                stage_record["protected_null_max_abs"] = null_max
                stage_record["protected_null_tolerance"] = 1.0e-10
                if null_max > 1.0e-10:
                    raise _hold("collapsed target-only residual is not numerically protected-null")
                violation = _full_vocab_violation(
                    states,
                    selected_token_ids=TARGET_TOKEN_IDS,
                    residual_rows=rows,
                )
                stage_record["runtime_full_vocab_recheck"] = {
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
                    raise _hold(f"stage {stage_number} runtime-dtype full-vocabulary margin failed")
                wrapper.set_payload(TARGET_TOKEN_IDS, torch.from_numpy(rows))
                candidate_route = _greedy_route(model, native_inputs, int(tokenizer.pad_token_id))
                receipt["counts"]["warm_candidates"] += 1
                evaluation = _evaluate_route(
                    tokenizer=tokenizer,
                    tokens=candidate_route,
                    contract=contract,
                    parent_owner_ids=(block.get("parent_owner_ids") or contract["parent_owner_ids"]),
                    label=(
                        "target-only-protected-null-final"
                        if stage_number == 5 else f"target-only-protected-null-stage-{stage_number}"
                    ),
                )
                gate = _stage_gate(evaluation, block=block)
                stage_record.update({
                    "candidate_route_sha256": token_ids_sha256(candidate_route),
                    "candidate_route_token_count": len(candidate_route),
                    "gate": gate,
                    "residual_identity": _residual_identity(TARGET_TOKEN_IDS, rows),
                })
                if not gate["passed"]:
                    terminal_status = "stage_exhausted"
                    raise _hold(f"stage {stage_number} sole ordinary-greedy candidate failed its gate")
                final_solution, final_rows = solved, rows
                if stage_number == 5:
                    warm_final = {
                        "generated_token_ids": candidate_route,
                        "generated_token_ids_sha256": token_ids_sha256(candidate_route),
                        "ledger": deepcopy(dict(evaluation["ledger"])),
                        "gate": gate,
                        "residual_identity": _residual_identity(TARGET_TOKEN_IDS, rows),
                        "protected_hidden_sha256": PROTECTED_HIDDEN_SHA256,
                        "basis_sha256": BASIS_SHA256,
                        "protected_null_max_abs": null_max,
                        "protected_null_tolerance": 1.0e-10,
                    }
            if warm_final is None or final_solution is None or final_rows is None:
                terminal_status = "stage_exhausted"
                raise _hold("five stages ended without warm final acceptance")
            payload = _save_augmented_payload(
                output,
                residual_rows=final_rows,
                solution=final_solution,
                basis_info=basis_info,
                contract=binding,
            )
            warm_final["residual_identity"]["payload_sha256"] = payload["identity"]["payload_sha256"]
            receipt["payload"] = payload
            receipt["warm_final"] = warm_final
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

        stage = "fresh_subprocess_cold_verification"
        del (
            model, wrapper, base_head, names, parameters, names_after, parameters_after,
            parameter, opened, tokenizer, native_inputs, prompts, _grids, _media, sentinels,
            _ordinary, controlled, positive_states, protected, basis, rows, final_rows,
        )
        gc.collect()
        torch.cuda.empty_cache()
        receipt["resources_before_cold"] = {
            "cuda_allocated_bytes_after_reference_drop": int(torch.cuda.memory_allocated(0)),
            "cuda_reserved_bytes_after_empty_cache": int(torch.cuda.memory_reserved(0)),
            "gc_collected_before_cold": True,
        }
        cold_path = output / "cold_verification.json"
        snapshot_path = Path(snapshot["path"])
        env = dict(os.environ)
        repo = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = repo + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        remaining = max(1.0, RESOURCE_BOUND["wall_time_seconds_max"] - (time.monotonic() - started))
        completed = subprocess.run(
            [
                sys.executable,
                str(snapshot_path),
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
            timeout=remaining,
            check=False,
        )
        if completed.returncode != 0 or not cold_path.is_file():
            raise _hold(
                "fresh successor cold subprocess failed: "
                + (completed.stderr[-2_000:] or completed.stdout[-2_000:] or f"exit {completed.returncode}")
            )
        cold = json.loads(cold_path.read_text(encoding="utf-8"))
        receipt["counts"]["model_loads"] = 2
        receipt["runtime"]["cold"] = cold.get("runtime")
        receipt["cold_final"] = cold
        warm_cold = {
            "successor_schema_exact": cold.get("schema_version") == SCHEMA_VERSION and cold.get("unit_id") == UNIT_ID,
            "route_exact": warm_final["generated_token_ids"] == cold.get("generated_token_ids"),
            "ledger_exact": warm_final["ledger"] == cold.get("ledger"),
            "residual_exact": warm_final["residual_identity"] == cold.get("residual_identity"),
            "selected_ids_exact": cold.get("residual_identity", {}).get("selected_token_ids") == TARGET_TOKEN_IDS,
            "warm_gate_passed": warm_final["gate"].get("passed") is True,
            "cold_gate_passed": dict(cold.get("gate", {})).get("passed") is True,
            "warm_surface_frozen": receipt["warm_surface"]["before"] == receipt["warm_surface"]["after"],
            "cold_surface_frozen": dict(cold.get("surface", {})).get("before") == dict(cold.get("surface", {})).get("after"),
            "surface_exact": receipt["warm_surface"]["before"] == dict(cold.get("surface", {})).get("before"),
            "frozen_surface_exact": receipt["warm_surface"]["frozen_before"] == dict(cold.get("frozen_surface", {})).get("before"),
            "protected_hidden_exact": warm_final["protected_hidden_sha256"] == cold.get("protected_hidden_sha256"),
            "basis_exact": warm_final["basis_sha256"] == cold.get("basis_sha256"),
            "warm_protected_null": float(warm_final["protected_null_max_abs"]) <= 1.0e-10,
            "cold_protected_null": float(cold.get("protected_null_max_abs", math.inf)) <= 1.0e-10,
        }
        receipt["warm_cold"] = warm_cold
        if not all(warm_cold.values()):
            raise _hold("warm/cold route, ledger, residual, gate, binding, or frozen surface mismatch")
        terminal_status = "cold_greedy_38_person_success"
    except BaseException as error:
        if terminal_status is None:
            terminal_status = "technical_hold"
        receipt["stop"] = {
            "stage": stage,
            "reason": str(error),
            "error_type": type(error).__name__,
            "traceback": traceback.format_exc(),
        }
    finally:
        elapsed = time.monotonic() - started
        receipt["status"] = terminal_status if terminal_status in TERMINAL_STATUSES else "technical_hold"
        receipt["wall_time_seconds"] = elapsed
        warm_peak_reserved = int(torch.cuda.max_memory_reserved(0))
        cold_peak_reserved = int(dict(receipt.get("cold_final", {})).get("peak_cuda_reserved_bytes", 0))
        receipt["resources"] = {
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
            "warm_peak_cuda_reserved_bytes": warm_peak_reserved,
            "cold_peak_cuda_reserved_bytes": cold_peak_reserved,
            "peak_cuda_reserved_bytes": max(warm_peak_reserved, cold_peak_reserved),
            "artifact_bytes_before_receipt": _artifact_bytes(output),
            "predeclared_bound": RESOURCE_BOUND,
        }
        receipt["claim_boundary"] = (
            "One Image2299 frozen-r32 augmented-model ordinary-greedy result only; "
            "not transfer, general enumeration learning, a base-r32 greedy result, or 8/8 tie recovery."
        )
        if (
            elapsed > RESOURCE_BOUND["wall_time_seconds_max"]
            or receipt["counts"]["solves"] > MAX_SOLVES
            or receipt["counts"]["warm_candidates"] > MAX_WARM_CANDIDATES
            or receipt["counts"]["model_loads"] > 2
            or receipt["resources"]["peak_cuda_reserved_bytes"] > RESOURCE_BOUND["max_peak_cuda_reserved_bytes"]
            or receipt["resources"]["artifact_bytes_before_receipt"] > RESOURCE_BOUND["output_artifact_bytes_max"]
        ):
            receipt["pre_resource_status"] = receipt["status"]
            receipt["status"] = "technical_hold"
            receipt.setdefault("stop", {"stage": "resource_enforcement", "reason": "predeclared resource bound exceeded"})
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
