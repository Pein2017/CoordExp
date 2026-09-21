#!/usr/bin/env python3
"""Solve all Image2299 target-only protected-null constraints in one program."""

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

from scripts.research import run_image2299_target_only_protected_null_distillation as staged


recursive = staged.recursive
base = staged.base
full_root = staged.full_root
token_ids_sha256 = staged.token_ids_sha256

SCHEMA_VERSION = "image2299.global_target_only_protected_null_distillation.v1"
UNIT_ID = "2026-08-30-image2299-global-target-only-protected-null-distillation"
OUTPUT_ROOT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration") / UNIT_ID
SOURCE_RECEIPT = (
    staged.OUTPUT_ROOT
    / "20260830T-image2299-target-only-protected-null-distillation-v1"
    / "receipt.json"
)
SOURCE_RECEIPT_SHA256 = "e1ed1e101bf0f3b8485491ee87bfdb9f226a4451580da1885222ba4cd8b4729e"
PREDECESSOR_RUNNER_SHA256 = "184bf01c50a042fbba699490da65434818e0d39ff498210e8a8b0d23863bf769"
TERMINAL_WITNESS_RECEIPT_SHA256 = staged.TERMINAL_WITNESS_RECEIPT_SHA256
MODEL_RECEIPT_SHA256 = staged.MODEL_RECEIPT_SHA256
START_SURFACE_SHA256 = staged.START_SURFACE_SHA256
FROZEN_SURFACE_SHA256 = staged.FROZEN_SURFACE_SHA256
PROMPT_TOKEN_SHA256 = staged.PROMPT_TOKEN_SHA256
IMAGE_SHA256 = staged.IMAGE_SHA256
START_CHECKPOINT = staged.START_CHECKPOINT
CONTROLLED_ROUTE_SHA256 = staged.CONTROLLED_ROUTE_SHA256
ORDINARY_ROUTE_SHA256 = staged.ORDINARY_ROUTE_SHA256
PROTECTED_HIDDEN_SHA256 = staged.PROTECTED_HIDDEN_SHA256
BASIS_SHA256 = staged.BASIS_SHA256
POSITIVE_POSITIONS = list(staged.POSITIVE_POSITIONS)
POSITIVE_STAGES = [1, 2, 2, 3, 3, 4, 4, 5, 5, 5, 5, 5]
TARGET_TOKEN_IDS = list(staged.TARGET_TOKEN_IDS)

MARGIN = staged.MARGIN
NORM_CAP = staged.NORM_CAP
POSITIVE_COUNT = 12
PROTECTED_COUNT = 664
RANK = 12
CONSTRAINT_COUNT = 108
VARIABLE_COUNT = 108
MAX_SOLVES = 1
MAX_WARM_CANDIDATES = 1
RESOURCE_BOUND = {
    "gpu_count": 1,
    "world_size": 1,
    "positive_states_max": POSITIVE_COUNT,
    "positive_rank_max": RANK,
    "selected_output_rows": len(TARGET_TOKEN_IDS),
    "constraint_count_max": CONSTRAINT_COUNT,
    "variable_count_max": VARIABLE_COUNT,
    "solve_count_max": MAX_SOLVES,
    "warm_candidate_count_max": MAX_WARM_CANDIDATES,
    "model_load_count_max": 2,
    "wall_time_seconds_max": 1_200,
    "max_peak_cuda_reserved_bytes": 16 * 2**30,
    "output_artifact_bytes_max": 100_000_000,
}
TERMINAL_STATUSES = {
    "cold_greedy_38_person_success",
    "global_target_only_infeasible",
    "norm_cap_exceeded",
    "runtime_margin_hold",
    "global_greedy_negative",
    "technical_hold",
}

ProtectedNullHold = staged.ProtectedNullHold
SparseOutputResidual = staged.SparseOutputResidual
_hold = staged._hold
_sha256 = staged._sha256
_value_sha256 = staged._value_sha256
_tensor_sha256 = staged._tensor_sha256
_atomic_json = staged._atomic_json
_capture_route = staged._capture_route
_greedy_route = staged._greedy_route
_evaluate_route = staged._evaluate_route
_row_norms = staged._row_norms
_collapsed_rows = staged._collapsed_rows
_full_vocab_violation = staged._full_vocab_violation
_residual_identity = staged._residual_identity
_surface_snapshot = staged._surface_snapshot
_artifact_bytes = staged._artifact_bytes
_require_one_gpu = staged._require_one_gpu
_recapture = staged._recapture
_target_only_constraints = staged._target_only_constraints
_solve_target_only_minimum_normalized = staged._solve_target_only_minimum_normalized
_compact_constraints = staged._compact_constraints


def _load_source_receipt() -> dict[str, Any]:
    if not SOURCE_RECEIPT.is_file() or _sha256(SOURCE_RECEIPT) != SOURCE_RECEIPT_SHA256:
        raise _hold("immutable staged predecessor receipt drifted")
    try:
        receipt = json.loads(SOURCE_RECEIPT.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise _hold(f"staged predecessor receipt is unreadable: {error}") from error
    if not isinstance(receipt, dict):
        raise _hold("staged predecessor receipt is not an object")
    return receipt


def _validate_staged_outcome(
    receipt: Mapping[str, Any], inherited: Mapping[str, Any],
) -> dict[str, Any]:
    stages = list(receipt.get("stages", ()))
    if len(stages) != 2 or len(inherited.get("blocks", ())) < 2:
        raise _hold("staged predecessor did not stop at its frozen second stage")
    first, second = map(dict, stages)
    first_gate = dict(first.get("gate", {}))
    second_gate = dict(second.get("gate", {}))
    first_generic = dict(first_gate.get("generic", {}))
    second_generic = dict(second_gate.get("generic", {}))
    first_solver = dict(first.get("solver", {}))
    second_solver = dict(second.get("solver", {}))
    first_block, second_block = map(dict, inherited["blocks"][:2])
    second_raw = dict(second_generic.get("raw_hard_counters", {}))
    expected_raw = {key: 0 for key in recursive.RAW_COUNTER_KEYS}
    expected_raw.update({
        "matcher_unmatched_count": 1,
        "unmatched_person_count": 1,
        "unsupported_person_count": 1,
    })
    first_owners = list(map(str, first_generic.get("matched_owner_ids", ())))
    second_owners = list(map(str, second_generic.get("matched_owner_ids", ())))
    if (
        int(first.get("stage", -1)) != 1
        or int(first.get("positive_count", -1)) != 1
        or int(first.get("constraint_count", -1)) != 9
        or first_solver.get("feasible") is not True
        or int(first_solver.get("constraint_count", -1)) != 9
        or int(first_solver.get("variable_count", -1)) != VARIABLE_COUNT
        or list(first_solver.get("selected_token_ids", ())) != TARGET_TOKEN_IDS
        or first.get("candidate_route_sha256") != first.get("expected_route_sha256")
        or first.get("expected_route_sha256") != first_block.get("expected_route_sha256")
        or int(first.get("candidate_route_token_count", -1)) != len(first_block["expected_route_tokens"])
        or first_gate.get("passed") is not True
        or first_gate.get("route_exact") is not True
        or set(first_owners) != set(map(str, first_block["expected_owner_ids"]))
        or int(second.get("stage", -1)) != 2
        or int(second.get("positive_count", -1)) != 3
        or int(second.get("constraint_count", -1)) != 27
        or float(second_solver.get("normalized_norm", math.inf)) != 0.7897506392158977
        or second_solver.get("feasible") is not True
        or int(second_solver.get("constraint_count", -1)) != 27
        or int(second_solver.get("variable_count", -1)) != VARIABLE_COUNT
        or list(second_solver.get("selected_token_ids", ())) != TARGET_TOKEN_IDS
        or dict(second.get("runtime_full_vocab_recheck", {})).get("passed") is not True
        or int(dict(second.get("runtime_full_vocab_recheck", {})).get("state_count", -1)) != 3
        or int(second.get("candidate_route_token_count", -1)) != 352
        or len(second_block["expected_route_tokens"]) != 343
        or second.get("expected_route_sha256") != second_block.get("expected_route_sha256")
        or second.get("candidate_route_sha256") == second.get("expected_route_sha256")
        or second_gate.get("passed") is not False
        or second_gate.get("route_exact") is not False
        or set(second_owners) != set(map(str, second_block["expected_owner_ids"]))
        or len(second_owners) != len(second_block["expected_owner_ids"])
        or list(map(str, second_generic.get("gained_owner_ids", ())))
        != ["gt:2299:32", "gt:2299:35"]
        or second_raw != expected_raw
    ):
        raise _hold("staged predecessor stage-1/stage-2 decision evidence drifted")
    return {
        "stage1_exact_success": True,
        "stage2_normalized_norm": 0.7897506392158977,
        "stage2_runtime_recheck_passed": True,
        "stage2_candidate_token_count": 352,
        "stage2_expected_token_count": 343,
        "stage2_intended_owner_set_exact": True,
        "stage2_single_unmatched_unsupported_row": True,
        "cold_absent": True,
    }


def _binding_contract() -> dict[str, Any]:
    receipt = _load_source_receipt()
    capture = dict(receipt.get("capture", {}))
    basis = dict(capture.get("basis", {}))
    source = dict(receipt.get("runner_source_snapshot", {}))
    source_path = Path(str(source.get("path", "")))
    inherited = staged._binding_contract()
    outcome = _validate_staged_outcome(receipt, inherited)
    if (
        receipt.get("schema_version") != staged.SCHEMA_VERSION
        or receipt.get("unit_id") != staged.UNIT_ID
        or receipt.get("status") != "stage_exhausted"
        or source.get("sha256") != PREDECESSOR_RUNNER_SHA256
        or not source_path.is_file()
        or _sha256(source_path) != PREDECESSOR_RUNNER_SHA256
        or _sha256(Path(staged.__file__)) != PREDECESSOR_RUNNER_SHA256
        or capture.get("ordinary_route_sha256") != ORDINARY_ROUTE_SHA256
        or int(capture.get("ordinary_route_token_count", -1)) != 307
        or capture.get("controlled_route_sha256") != CONTROLLED_ROUTE_SHA256
        or int(capture.get("controlled_route_token_count", -1)) != 370
        or list(capture.get("positive_positions", ())) != POSITIVE_POSITIONS
        or int(capture.get("positive_count", -1)) != POSITIVE_COUNT
        or int(capture.get("protected_count", -1)) != PROTECTED_COUNT
        or list(basis.get("protected_hidden_shape", ())) != [PROTECTED_COUNT, 2048]
        or int(basis.get("rank", -1)) != RANK
        or basis.get("protected_hidden_sha256") != PROTECTED_HIDDEN_SHA256
        or basis.get("basis_sha256") != BASIS_SHA256
        or receipt.get("bindings") != staged._binding_receipt()
        or receipt.get("counts") != {
            "base_greedy": 1,
            "model_loads": 1,
            "solves": 2,
            "warm_candidates": 2,
            "zero_wrapper_greedy": 1,
        }
        or any(key in receipt for key in ("cold_final", "warm_final", "warm_cold", "payload"))
    ):
        raise _hold("staged predecessor receipt, recapture, source, or no-cold binding drifted")
    return inherited | {"predecessor_receipt": receipt, "predecessor_outcome": outcome}


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
        "positive_count": POSITIVE_COUNT,
        "protected_count": PROTECTED_COUNT,
        "protected_hidden_sha256": PROTECTED_HIDDEN_SHA256,
        "protected_hidden_shape": [PROTECTED_COUNT, 2048],
        "basis_sha256": BASIS_SHA256,
        "rank": RANK,
        "target_token_ids": TARGET_TOKEN_IDS,
        "constraint_count": CONSTRAINT_COUNT,
        "variable_count": VARIABLE_COUNT,
        "margin": MARGIN,
        "norm_cap": NORM_CAP,
        "staged_outcome": contract["predecessor_outcome"],
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
        raise _hold("global successor runner source snapshot readback drifted")
    return {"path": str(snapshot), "sha256": _sha256(snapshot)}


def _solve_global_target_only(
    states: Sequence[Mapping[str, Any]], basis: np.ndarray, row_norms: Mapping[int, float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if (
        len(states) != POSITIVE_COUNT
        or [int(item["position"]) for item in states] != POSITIVE_POSITIONS
        or [int(item["stage"]) for item in states] != POSITIVE_STAGES
    ):
        raise ValueError("global program requires the exact twelve recaptured positive states")
    constraints = _target_only_constraints(states, basis)
    if len(constraints) != CONSTRAINT_COUNT:
        raise ValueError("global program requires exactly 108 target-only constraints")
    solved = _solve_target_only_minimum_normalized(
        constraints,
        rank=RANK,
        row_norms=row_norms,
        target_token_ids=TARGET_TOKEN_IDS,
    )
    return constraints, solved


def _final_gate(evaluation: Mapping[str, Any], *, binding: Mapping[str, Any]) -> dict[str, Any]:
    block = dict(binding["blocks"][-1])
    if int(block.get("stage", -1)) != 5:
        raise _hold("final controlled-owner gate binding drifted")
    return staged._stage_gate(evaluation, block=block)


def _save_augmented_payload(
    output: Path,
    *,
    residual_rows: np.ndarray,
    solution: Mapping[str, Any],
    basis_info: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    token_ids = list(map(int, solution["selected_token_ids"]))
    rows = np.asarray(residual_rows, dtype=np.float64)
    if token_ids != TARGET_TOKEN_IDS or rows.shape != (len(TARGET_TOKEN_IDS), 2048):
        raise _hold("refusing to persist outside the fixed nine-row global surface")
    payload = output / "global_target_only_protected_null_output_residual.safetensors"
    save_file(
        {
            "selected_token_ids": torch.tensor(token_ids, dtype=torch.int64),
            "residual_rows": torch.from_numpy(rows),
        },
        payload,
        metadata={"schema_version": SCHEMA_VERSION, "unit_id": UNIT_ID},
    )
    identity = _residual_identity(token_ids, rows)
    identity["payload_sha256"] = _sha256(payload)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "base_checkpoint": str(START_CHECKPOINT),
        "source_receipt_sha256": SOURCE_RECEIPT_SHA256,
        "controlled_route_sha256": CONTROLLED_ROUTE_SHA256,
        "selected_output_rows_contract": "exactly the fixed nine target-token rows",
        "normalization": "per-selected-base-output-row L2; one global normalized Frobenius cap",
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
        "base_checkpoint": str(START_CHECKPOINT),
        "base_checkpoint_readback": contract["runtime_contract"]["checkpoint_readback"],
        "base_weight_payload_mode": "reference_only_no_copy_no_mutation",
        "residual_payload": str(payload),
        "residual_payload_sha256": identity["payload_sha256"],
        "residual_metadata": str(metadata_path),
        "residual_metadata_sha256": _sha256(metadata_path),
        "load_order": [
            "load frozen base checkpoint",
            "install output-head residual module",
            "load residual payload",
        ],
    }
    manifest_path = output / "augmented_checkpoint_manifest.json"
    _atomic_json(manifest_path, manifest)
    return {
        "payload": str(payload),
        "metadata": str(metadata_path),
        "manifest": str(manifest_path),
        "identity": identity,
    }


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
        raise _hold("cold global successor residual payload/schema/surface is invalid")
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
        ordinary_route, _ordinary, _controlled, states, protected, basis, basis_info = _recapture(
            model=model,
            tokenizer=tokenizer,
            native_inputs=native_inputs,
            binding=binding,
            base_head=base_head,
        )
        if (
            [int(item["position"]) for item in states] != POSITIVE_POSITIONS
            or basis_info["basis_sha256"] != BASIS_SHA256
            or _tensor_sha256(protected) != PROTECTED_HIDDEN_SHA256
        ):
            raise _hold("cold positive/protected/basis recapture drifted")
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
            parent_owner_ids=binding["blocks"][-1]["parent_owner_ids"],
            label="global-target-only-protected-null-final",
        )
        gate = _final_gate(evaluation, binding=binding)
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
        "protected_hidden_sha256": _tensor_sha256(protected),
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
        "counts": {
            "model_loads": 0,
            "solves": 0,
            "warm_candidates": 0,
            "base_greedy": 0,
            "zero_wrapper_controlled_capture": 0,
        },
    }
    phase = "single_warm_model_load"
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

            phase = "recapture_frozen_states"
            ordinary_route, _ordinary, controlled, states, protected, basis, basis_info = _recapture(
                model=model,
                tokenizer=tokenizer,
                native_inputs=native_inputs,
                binding=binding,
                base_head=base_head,
            )
            receipt["counts"]["base_greedy"] = 1
            wrapper = SparseOutputResidual(
                base_head,
                TARGET_TOKEN_IDS,
                torch.zeros((len(TARGET_TOKEN_IDS), controlled["hidden"].shape[1]), dtype=torch.float64),
            ).to(next(model.parameters()).device)
            model.set_output_embeddings(wrapper)
            zero_controlled = _capture_route(
                model=model,
                output_head=wrapper,
                native_inputs=native_inputs,
                route_tokens=binding["controlled_route_tokens"],
                pad_token_id=int(tokenizer.pad_token_id),
            )
            receipt["counts"]["zero_wrapper_controlled_capture"] = 1
            if (
                not torch.equal(zero_controlled["logits"], controlled["logits"])
                or not torch.equal(zero_controlled["hidden"], controlled["hidden"])
            ):
                raise _hold("zero global residual failed hidden/logit parity")
            receipt["capture"] = {
                "ordinary_route_sha256": token_ids_sha256(ordinary_route),
                "ordinary_route_token_count": len(ordinary_route),
                "controlled_route_sha256": token_ids_sha256(binding["controlled_route_tokens"]),
                "controlled_route_token_count": len(binding["controlled_route_tokens"]),
                "positive_count": len(states),
                "positive_positions": [int(item["position"]) for item in states],
                "protected_count": len(protected),
                "protected_selection": basis_info.pop("selection"),
                "basis": basis_info,
                "target_token_ids": TARGET_TOKEN_IDS,
                "zero_wrapper_hidden_logits_exact": True,
            }

            phase = "one_global_target_only_solve"
            constraints, solved = _solve_global_target_only(
                states,
                basis,
                _row_norms(base_head, TARGET_TOKEN_IDS),
            )
            receipt["counts"]["solves"] = 1
            receipt["global_program"] = {
                "positive_count": len(states),
                "constraint_count": len(constraints),
                "constraints": _compact_constraints(constraints),
                "solver": {
                    key: value
                    for key, value in solved.items()
                    if key not in {"normalized_rows", "scaled_basis_rows"}
                },
            }
            if not solved.get("feasible"):
                terminal_status = "global_target_only_infeasible"
                raise _hold("global target-only inequalities are infeasible")
            if solved["selected_token_ids"] != TARGET_TOKEN_IDS or solved["variable_count"] != VARIABLE_COUNT:
                raise _hold("global solve escaped the exact 108-variable nine-row surface")
            if float(solved["normalized_norm"]) > NORM_CAP:
                terminal_status = "norm_cap_exceeded"
                raise _hold("global minimum normalized norm exceeds 1")
            rows = _collapsed_rows(solved, basis)
            null_max = float(np.max(np.abs(protected @ rows.T), initial=0.0))
            receipt["global_program"]["protected_null_max_abs"] = null_max
            receipt["global_program"]["protected_null_tolerance"] = 1.0e-10
            if null_max > 1.0e-10:
                raise _hold("global residual is not numerically protected-null")
            violation = _full_vocab_violation(
                states,
                selected_token_ids=TARGET_TOKEN_IDS,
                residual_rows=rows,
            )
            receipt["global_program"]["runtime_full_vocab_recheck"] = {
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
                raise _hold("global runtime-dtype full-vocabulary margin failed")

            phase = "one_final_ordinary_greedy_candidate"
            wrapper.set_payload(TARGET_TOKEN_IDS, torch.from_numpy(rows))
            candidate_route = _greedy_route(model, native_inputs, int(tokenizer.pad_token_id))
            receipt["counts"]["warm_candidates"] = 1
            evaluation = _evaluate_route(
                tokenizer=tokenizer,
                tokens=candidate_route,
                contract=contract,
                parent_owner_ids=binding["blocks"][-1]["parent_owner_ids"],
                label="global-target-only-protected-null-final",
            )
            gate = _final_gate(evaluation, binding=binding)
            warm_final = {
                "generated_token_ids": candidate_route,
                "generated_token_ids_sha256": token_ids_sha256(candidate_route),
                "ledger": deepcopy(dict(evaluation["ledger"])),
                "gate": gate,
                "residual_identity": _residual_identity(TARGET_TOKEN_IDS, rows),
                "protected_hidden_sha256": _tensor_sha256(protected),
                "basis_sha256": basis_info["basis_sha256"],
                "protected_null_max_abs": null_max,
                "protected_null_tolerance": 1.0e-10,
            }
            receipt["warm_final"] = warm_final
            if not gate["passed"]:
                terminal_status = "global_greedy_negative"
                raise _hold("global residual sole ordinary-greedy candidate failed the final gate")

            payload = _save_augmented_payload(
                output,
                residual_rows=rows,
                solution=solved,
                basis_info=basis_info,
                contract=binding,
            )
            warm_final["residual_identity"]["payload_sha256"] = payload["identity"]["payload_sha256"]
            receipt["payload"] = payload
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

        phase = "fresh_subprocess_cold_verification"
        model = wrapper = base_head = opened = tokenizer = native_inputs = prompts = None
        _ordinary = controlled = zero_controlled = states = protected = basis = rows = None
        parameters = parameters_after = parameter = sentinels = evaluation = None
        _grids = _media = None
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
                "fresh global successor cold subprocess failed: "
                + (completed.stderr[-2_000:] or completed.stdout[-2_000:] or f"exit {completed.returncode}")
            )
        cold = json.loads(cold_path.read_text(encoding="utf-8"))
        receipt["counts"]["model_loads"] = 2
        receipt["runtime"]["cold"] = cold.get("runtime")
        receipt["cold_final"] = cold
        warm_cold = {
            "successor_schema_exact": cold.get("schema_version") == SCHEMA_VERSION
            and cold.get("unit_id") == UNIT_ID,
            "route_exact": warm_final["generated_token_ids"] == cold.get("generated_token_ids"),
            "ledger_exact": warm_final["ledger"] == cold.get("ledger"),
            "payload_exact": warm_final["residual_identity"] == cold.get("residual_identity"),
            "selected_ids_exact": cold.get("residual_identity", {}).get("selected_token_ids")
            == TARGET_TOKEN_IDS,
            "warm_gate_passed": warm_final["gate"].get("passed") is True,
            "cold_gate_passed": dict(cold.get("gate", {})).get("passed") is True,
            "warm_surface_frozen": receipt["warm_surface"]["before"] == receipt["warm_surface"]["after"],
            "cold_surface_frozen": dict(cold.get("surface", {})).get("before")
            == dict(cold.get("surface", {})).get("after"),
            "surface_exact": receipt["warm_surface"]["before"]
            == dict(cold.get("surface", {})).get("before"),
            "frozen_surface_exact": receipt["warm_surface"]["frozen_before"]
            == dict(cold.get("frozen_surface", {})).get("before"),
            "protected_hidden_exact": warm_final["protected_hidden_sha256"]
            == cold.get("protected_hidden_sha256"),
            "basis_exact": warm_final["basis_sha256"] == cold.get("basis_sha256"),
            "warm_protected_null": float(warm_final["protected_null_max_abs"]) <= 1.0e-10,
            "cold_protected_null": float(cold.get("protected_null_max_abs", math.inf)) <= 1.0e-10,
        }
        receipt["warm_cold"] = warm_cold
        if not all(warm_cold.values()):
            raise _hold("warm/cold route, ledger, payload, binding, null, or surface mismatch")
        terminal_status = "cold_greedy_38_person_success"
    except BaseException as error:
        if terminal_status is None:
            terminal_status = "technical_hold"
        receipt["stop"] = {
            "phase": phase,
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
