#!/usr/bin/env python3
"""Release only the frozen Image2299 global normalized residual cap to 9/8."""

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

from scripts.research import run_image2299_global_target_only_protected_null_distillation as predecessor


recursive = predecessor.recursive
base = predecessor.base
full_root = predecessor.full_root
token_ids_sha256 = predecessor.token_ids_sha256

SCHEMA_VERSION = "image2299.dyadic_norm_release_distillation.v1"
UNIT_ID = "2026-08-30-image2299-dyadic-norm-release-distillation"
OUTPUT_ROOT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration") / UNIT_ID
SOURCE_RECEIPT = (
    predecessor.OUTPUT_ROOT
    / "20260830T-image2299-global-target-only-protected-null-distillation-v2"
    / "receipt.json"
)
SOURCE_RECEIPT_SHA256 = "3bc288a62e6b6ef23ef9f08dc53b33e9d5bafd3f690d8cb7c226f20577a28cdf"
PREDECESSOR_RUNNER_SHA256 = "3974b1fbab3409fd7ebabc0fe6aedf50ad6ed0c1308a3ef8582d8221124bf4ba"
TERMINAL_WITNESS_RECEIPT_SHA256 = predecessor.TERMINAL_WITNESS_RECEIPT_SHA256
MODEL_RECEIPT_SHA256 = predecessor.MODEL_RECEIPT_SHA256
START_SURFACE_SHA256 = predecessor.START_SURFACE_SHA256
FROZEN_SURFACE_SHA256 = predecessor.FROZEN_SURFACE_SHA256
PROMPT_TOKEN_SHA256 = predecessor.PROMPT_TOKEN_SHA256
IMAGE_SHA256 = predecessor.IMAGE_SHA256
START_CHECKPOINT = predecessor.START_CHECKPOINT
CONTROLLED_ROUTE_SHA256 = predecessor.CONTROLLED_ROUTE_SHA256
ORDINARY_ROUTE_SHA256 = predecessor.ORDINARY_ROUTE_SHA256
PROTECTED_HIDDEN_SHA256 = predecessor.PROTECTED_HIDDEN_SHA256
BASIS_SHA256 = predecessor.BASIS_SHA256
POSITIVE_POSITIONS = list(predecessor.POSITIVE_POSITIONS)
POSITIVE_STAGES = list(predecessor.POSITIVE_STAGES)
TARGET_TOKEN_IDS = list(predecessor.TARGET_TOKEN_IDS)

MARGIN = predecessor.MARGIN
NORM_CAP_NUMERATOR = 9
NORM_CAP_DENOMINATOR = 8
NORM_CAP = NORM_CAP_NUMERATOR / NORM_CAP_DENOMINATOR
POSITIVE_COUNT = predecessor.POSITIVE_COUNT
PROTECTED_COUNT = predecessor.PROTECTED_COUNT
RANK = predecessor.RANK
CONSTRAINT_COUNT = predecessor.CONSTRAINT_COUNT
VARIABLE_COUNT = predecessor.VARIABLE_COUNT
MAX_SOLVES = predecessor.MAX_SOLVES
MAX_WARM_CANDIDATES = predecessor.MAX_WARM_CANDIDATES
RESOURCE_BOUND = deepcopy(predecessor.RESOURCE_BOUND)
TERMINAL_STATUSES = {
    "cold_greedy_38_person_success",
    "certified_infeasible",
    "numerical_solver_hold",
    "norm_cap_exceeded",
    "runtime_margin_hold",
    "global_greedy_negative",
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
_greedy_route = predecessor._greedy_route
_evaluate_route = predecessor._evaluate_route
_row_norms = predecessor._row_norms
_collapsed_rows = predecessor._collapsed_rows
_full_vocab_violation = predecessor._full_vocab_violation
_residual_identity = predecessor._residual_identity
_surface_snapshot = predecessor._surface_snapshot
_artifact_bytes = predecessor._artifact_bytes
_require_one_gpu = predecessor._require_one_gpu
_recapture = predecessor._recapture
_target_only_constraints = predecessor._target_only_constraints
_compact_constraints = predecessor._compact_constraints
_normalized_primal_system = predecessor._normalized_primal_system
_legacy_dual_diagnostic = predecessor._legacy_dual_diagnostic
_solve_target_only_minimum_normalized = predecessor._solve_target_only_minimum_normalized
_solve_global_target_only = predecessor._solve_global_target_only
_final_gate = predecessor._final_gate


def _require_frozen_cap() -> None:
    if (
        NORM_CAP_NUMERATOR != 9
        or NORM_CAP_DENOMINATOR != 8
        or NORM_CAP != 1.125
    ):
        raise _hold("dyadic norm-release cap drifted from exact 9/8")


def _load_source_receipt() -> dict[str, Any]:
    if not SOURCE_RECEIPT.is_file() or _sha256(SOURCE_RECEIPT) != SOURCE_RECEIPT_SHA256:
        raise _hold("immutable cap-1 predecessor receipt drifted")
    try:
        receipt = json.loads(SOURCE_RECEIPT.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise _hold(f"cap-1 predecessor receipt is unreadable: {error}") from error
    if not isinstance(receipt, dict):
        raise _hold("cap-1 predecessor receipt is not an object")
    return receipt


def _validate_cap1_outcome(receipt: Mapping[str, Any]) -> dict[str, Any]:
    source = dict(receipt.get("runner_source_snapshot", {}))
    source_path = Path(str(source.get("path", "")))
    capture = dict(receipt.get("capture", {}))
    basis = dict(capture.get("basis", {}))
    program = dict(receipt.get("global_program", {}))
    solver = dict(program.get("solver", {}))
    lp = dict(solver.get("lp_diagnostics", {}))
    primal = dict(solver.get("primal_diagnostics", {}))
    dual = dict(solver.get("legacy_dual_diagnostic", {}))
    if (
        receipt.get("schema_version") != predecessor.SCHEMA_VERSION
        or receipt.get("unit_id") != predecessor.UNIT_ID
        or receipt.get("status") != "norm_cap_exceeded"
        or source.get("sha256") != PREDECESSOR_RUNNER_SHA256
        or not source_path.is_file()
        or _sha256(source_path) != PREDECESSOR_RUNNER_SHA256
        or _sha256(Path(predecessor.__file__)) != PREDECESSOR_RUNNER_SHA256
        or receipt.get("bindings") != predecessor._binding_receipt()
        or receipt.get("counts") != {
            "base_greedy": 1,
            "model_loads": 1,
            "solves": 1,
            "warm_candidates": 0,
            "zero_wrapper_controlled_capture": 1,
        }
        or int(capture.get("positive_count", -1)) != POSITIVE_COUNT
        or list(capture.get("positive_positions", ())) != POSITIVE_POSITIONS
        or int(capture.get("protected_count", -1)) != PROTECTED_COUNT
        or list(capture.get("target_token_ids", ())) != TARGET_TOKEN_IDS
        or capture.get("zero_wrapper_hidden_logits_exact") is not True
        or basis.get("protected_hidden_sha256") != PROTECTED_HIDDEN_SHA256
        or basis.get("basis_sha256") != BASIS_SHA256
        or int(basis.get("rank", -1)) != RANK
        or int(program.get("positive_count", -1)) != POSITIVE_COUNT
        or int(program.get("constraint_count", -1)) != CONSTRAINT_COUNT
        or solver.get("feasible") is not True
        or solver.get("solver_classification") != "feasible_primal_minimum_norm"
        or list(solver.get("selected_token_ids", ())) != TARGET_TOKEN_IDS
        or int(solver.get("constraint_count", -1)) != CONSTRAINT_COUNT
        or int(solver.get("variable_count", -1)) != VARIABLE_COUNT
        or float(solver.get("normalized_norm", math.inf)) != 1.0972734315870298
        or float(solver.get("minimum_slack", -math.inf)) != -5.329070518200751e-15
        or lp.get("method") != "highs"
        or lp.get("success") is not True
        or int(lp.get("status", -1)) != 0
        or primal.get("method") != "SLSQP"
        or primal.get("success") is not True
        or float(primal.get("minimum_slack", -math.inf)) != -5.329070518200751e-15
        or float(primal.get("objective", math.inf)) != 0.6020044918333882
        or float(dual.get("normalized_norm_lower_bound", math.inf)) != 1.0972734315870256
        or dict(receipt.get("stop", {})).get("phase") != "one_global_target_only_solve"
        or any(key in receipt for key in ("warm_final", "cold_final", "warm_cold", "payload"))
    ):
        raise _hold("cap-1 receipt, repaired source, solve evidence, or zero-warm stop drifted")
    return {
        "status": "norm_cap_exceeded",
        "constraint_count": CONSTRAINT_COUNT,
        "variable_count": VARIABLE_COUNT,
        "highs_feasible": True,
        "slsqp_normalized_norm": 1.0972734315870298,
        "slsqp_minimum_slack": -5.329070518200751e-15,
        "slsqp_objective": 0.6020044918333882,
        "dual_normalized_norm_lower_bound": 1.0972734315870256,
        "above_cap_1": True,
        "below_cap_9_8": True,
        "one_solve": True,
        "no_warm_candidate": True,
        "no_payload": True,
        "no_cold": True,
    }


def _binding_contract() -> dict[str, Any]:
    _require_frozen_cap()
    inherited = predecessor._binding_contract()
    receipt = _load_source_receipt()
    outcome = _validate_cap1_outcome(receipt)
    return inherited | {"cap1_receipt": receipt, "cap1_outcome": outcome}


def _binding_receipt() -> dict[str, Any]:
    contract = _binding_contract()
    inherited = predecessor._binding_receipt()
    return inherited | {
        "schema_version": SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "source_receipt": str(SOURCE_RECEIPT),
        "source_receipt_sha256": SOURCE_RECEIPT_SHA256,
        "predecessor_runner_sha256": PREDECESSOR_RUNNER_SHA256,
        "norm_cap": NORM_CAP,
        "norm_cap_numerator": NORM_CAP_NUMERATOR,
        "norm_cap_denominator": NORM_CAP_DENOMINATOR,
        "cap1_outcome": contract["cap1_outcome"],
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
        raise _hold("dyadic norm-release successor runner source snapshot readback drifted")
    return {"path": str(snapshot), "sha256": _sha256(snapshot)}


def _save_augmented_payload(
    output: Path,
    *,
    residual_rows: np.ndarray,
    solution: Mapping[str, Any],
    basis_info: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    _require_frozen_cap()
    token_ids = list(map(int, solution["selected_token_ids"]))
    rows = np.asarray(residual_rows, dtype=np.float64)
    if token_ids != TARGET_TOKEN_IDS or rows.shape != (len(TARGET_TOKEN_IDS), 2048):
        raise _hold("refusing to persist outside the fixed nine-row dyadic norm-release surface")
    payload = output / "dyadic_norm_release_output_residual.safetensors"
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
        "cap_numerator": NORM_CAP_NUMERATOR,
        "cap_denominator": NORM_CAP_DENOMINATOR,
        "selected_token_ids": token_ids,
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
        or metadata.get("cap") != NORM_CAP
        or metadata.get("cap_numerator") != NORM_CAP_NUMERATOR
        or metadata.get("cap_denominator") != NORM_CAP_DENOMINATOR
        or metadata.get("selected_token_ids") != TARGET_TOKEN_IDS
        or float(metadata.get("normalized_norm", math.inf)) > NORM_CAP
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
        raise _hold("cold dyadic norm-release successor residual payload/schema/surface is invalid")
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
            label="dyadic-norm-release-final",
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
                if solved.get("solver_classification") == "certified_infeasible":
                    terminal_status = "certified_infeasible"
                    raise _hold("HiGHS certified the global target-only inequalities infeasible")
                terminal_status = "numerical_solver_hold"
                raise _hold("global target-only numerical solver did not produce an accepted primal")
            if solved["selected_token_ids"] != TARGET_TOKEN_IDS or solved["variable_count"] != VARIABLE_COUNT:
                raise _hold("global solve escaped the exact 108-variable nine-row surface")
            if float(solved["normalized_norm"]) > NORM_CAP:
                terminal_status = "norm_cap_exceeded"
                raise _hold("global minimum normalized norm exceeds 9/8")
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
                label="dyadic-norm-release-final",
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
                "fresh dyadic norm-release successor cold subprocess failed: "
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
