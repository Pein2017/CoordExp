#!/usr/bin/env python3
"""Distill the frozen Image2299 terminal witness with a protected-null output residual."""

from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
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
from torch import nn

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research import run_image2299_terminal_eos_seal as terminal


recursive = terminal.recursive
ota = terminal.ota
base = terminal.base
full_root = base.full_root
token_ids_sha256 = terminal.token_ids_sha256

SCHEMA_VERSION = "image2299.protected_null_output_distillation.v1"
UNIT_ID = "2026-08-30-image2299-protected-null-output-distillation"
OUTPUT_ROOT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration") / UNIT_ID
SOURCE_RECEIPT = (
    terminal.OUTPUT_ROOT / "20260830T-image2299-terminal-eos-seal-v1" / "receipt.json"
)
SOURCE_RECEIPT_SHA256 = "d9cb113ae3f7cbe4b080fbe4f8faf76f68d36d9ca2844940b27560d330ab4495"
SOURCE_RUNNER_SHA256 = "1247d4cc901f2b95116cfd8b8c7fa5166ad42a5e0ff9c97d316b902c336baf0a"
CONTROLLED_ROUTE_SHA256 = "c89c6f9900303227cf781caf4a39dc203a09a53b0d319930754cda184529530e"
MODEL_RECEIPT_SHA256 = "a2912cab11797ddab8a556be1c17603d537271981047d40b99ffa6f27515278c"
START_SURFACE_SHA256 = "2ac0e8a963d91ac7a4ba42322272889efd74bd46578c1ac44e16e531d1246676"
FROZEN_SURFACE_SHA256 = "c16f0b52a2d5ce5ccfa1b239a0934a9ed27945278fc65e97d56cb8d50c4877d5"
PROMPT_TOKEN_SHA256 = "33f11458039a9b8c01e1329eeedad91ac68824cc5454c9df4b15e47d50458ffb"
IMAGE_SHA256 = "cd7199a37188c9ac6481520175866cbd78fa6fba35f4290bb0b60c78afcb2df3"

MARGIN = 0.01
NORM_CAP = 1.0
PROJECTED_NORM_FLOOR = 1.0e-3
INITIAL_TOP_K = 8
MAX_POSITIVES = 30
MAX_RANK = 30
MAX_CLOSURES_PER_STAGE = 2
MAX_SOLVES = 15
MAX_WARM_CANDIDATES = 15
ROW_OPEN_TOKEN = recursive.ROW_OPEN[0]
RESOURCE_BOUND = {
    "gpu_count": 1,
    "world_size": 1,
    "positive_states_max": MAX_POSITIVES,
    "positive_rank_max": MAX_RANK,
    "solve_count_max": MAX_SOLVES,
    "warm_candidate_count_max": MAX_WARM_CANDIDATES,
    "model_load_count_max": 2,
    "wall_time_seconds_max": 1_800,
    "max_peak_cuda_reserved_bytes": 16 * 2**30,
    "output_artifact_bytes_max": 100_000_000,
}
TERMINAL_STATUSES = {
    "cold_greedy_38_person_success",
    "nullspace_infeasible",
    "norm_cap_exceeded",
    "stage_exhausted",
    "technical_hold",
}


class ProtectedNullHold(RuntimeError):
    """An immutable binding, numerical, runtime, or acceptance contract failed."""


def _hold(message: str) -> ProtectedNullHold:
    return ProtectedNullHold(f"HOLD: {message}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _value_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _tensor_sha256(value: torch.Tensor | np.ndarray) -> str:
    array = value.detach().cpu().contiguous().numpy() if isinstance(value, torch.Tensor) else value
    array = np.ascontiguousarray(array)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_source_receipt() -> dict[str, Any]:
    if not SOURCE_RECEIPT.is_file() or _sha256(SOURCE_RECEIPT) != SOURCE_RECEIPT_SHA256:
        raise _hold("immutable terminal witness receipt drifted")
    try:
        value = json.loads(SOURCE_RECEIPT.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise _hold(f"terminal witness receipt is unreadable: {error}") from error
    if not isinstance(value, dict):
        raise _hold("terminal witness receipt is not an object")
    return value


def _derive_five_blocks(receipt: Mapping[str, Any]) -> list[dict[str, Any]]:
    transcript = list(receipt.get("intervention_transcript", ()))
    if len(transcript) != 5:
        raise _hold("terminal witness does not contain exactly five interventions")
    blocks: list[dict[str, Any]] = []
    for stage, raw in enumerate(transcript[:4], start=1):
        item = dict(raw)
        position = int(item.get("x1_position", -1))
        prefix = list(map(int, item.get("forced_prefix_tokens", ())))
        expected = list(map(int, item.get("generated_token_ids", ())))
        positions = list(range(position - 4, position + 1))
        targets = prefix[position - 4 : position + 1]
        if (
            item.get("depth") != stage - 1
            or position < 4
            or len(prefix) != position + 1
            or len(targets) != 5
            or targets[:4] != list(recursive.ROW_OPEN)
            or targets[-1] != int(item.get("x1_token_id", -1))
            or token_ids_sha256(prefix) != item.get("forced_prefix_sha256")
            or not expected
            or expected[-1] != base.EOS
            or base.EOS in expected[:-1]
            or token_ids_sha256(expected) != item.get("generated_token_ids_sha256")
            or expected[: len(prefix)] != prefix
        ):
            raise _hold(f"stage {stage} source-node route/block binding drifted")
        blocks.append({
            "stage": stage,
            "kind": "appended_row_open_through_x1",
            "positions": positions,
            "target_token_ids": targets,
            "x1_position": position,
            "x1_token_id": targets[-1],
            "expected_route_tokens": expected,
            "expected_route_sha256": item["generated_token_ids_sha256"],
            "expected_owner_ids": list(map(str, item.get("matched_owner_ids", ()))),
            "parent_owner_ids": list(map(str, item.get("parent_owner_ids", ()))),
        })

    final = dict(transcript[4])
    prefix = list(map(int, final.get("forced_prefix_tokens", ())))
    route = list(map(int, final.get("sealed_route_tokens", ())))
    positions = list(range(len(prefix) - base.ROW_TOKENS, len(route)))
    targets = route[len(prefix) - base.ROW_TOKENS :]
    if (
        final.get("control") != "terminal_complete_row_plus_eos"
        or len(prefix) < base.ROW_TOKENS
        or len(targets) != base.ROW_TOKENS + 1
        or targets[:4] != list(recursive.ROW_OPEN)
        or targets[-2] != terminal.ROW_CLOSE
        or targets[-1] != base.EOS
        or route != [*prefix, base.EOS]
        or token_ids_sha256(prefix) != final.get("forced_prefix_sha256")
        or token_ids_sha256(route) != CONTROLLED_ROUTE_SHA256
        or final.get("sealed_route_sha256") != CONTROLLED_ROUTE_SHA256
    ):
        raise _hold("stage 5 final-row/EOS block binding drifted")
    selected = dict(dict(receipt.get("selection", {})).get("candidate", {}))
    blocks.append({
        "stage": 5,
        "kind": "final_complete_row_plus_eos",
        "positions": positions,
        "target_token_ids": targets,
        "expected_route_tokens": route,
        "expected_route_sha256": CONTROLLED_ROUTE_SHA256,
        "expected_owner_ids": list(map(str, selected.get("matched_owner_ids", ()))),
        "expected_person_owner_ids": list(map(str, selected.get("matched_person_owner_ids", ()))),
        "expected_tie_owner_ids": list(map(str, selected.get("matched_tie_owner_ids", ()))),
        "parent_owner_ids": list(blocks[0]["parent_owner_ids"]),
    })
    if sum(len(item["positions"]) for item in blocks) != MAX_POSITIVES:
        raise _hold("five intervention blocks no longer have the static <=30-positive bound")
    return blocks


def _binding_contract() -> dict[str, Any]:
    receipt = _load_source_receipt()
    contract = recursive._start_contract()
    blocks = _derive_five_blocks(receipt)
    source = dict(receipt.get("runner_source_snapshot", {}))
    source_path = Path(str(source.get("path", "")))
    selected = dict(dict(receipt.get("selection", {})).get("candidate", {}))
    fresh = dict(receipt.get("fresh_replay", {}))
    controlled = blocks[-1]["expected_route_tokens"]
    expected_owners = blocks[-1]["expected_owner_ids"]
    if (
        receipt.get("schema_version") != terminal.SCHEMA_VERSION
        or receipt.get("unit_id") != terminal.UNIT_ID
        or receipt.get("status") != "controlled_38_person_success"
        or not source_path.is_file()
        or source.get("sha256") != SOURCE_RUNNER_SHA256
        or _sha256(source_path) != SOURCE_RUNNER_SHA256
        or recursive.MODEL_RECEIPT_SHA256 != MODEL_RECEIPT_SHA256
        or recursive.START_SURFACE_SHA256 != START_SURFACE_SHA256
        or recursive.FROZEN_SURFACE_SHA256 != FROZEN_SURFACE_SHA256
        or base.PROMPT_TOKEN_SHA256 != PROMPT_TOKEN_SHA256
        or base.IMAGE_SHA256 != IMAGE_SHA256
        or dict(receipt.get("bindings", {})).get("checkpoint_readback")
        != contract.get("checkpoint_readback")
        or selected.get("sealed_route_tokens") != controlled
        or selected.get("sealed_route_sha256") != CONTROLLED_ROUTE_SHA256
        or int(selected.get("matched_person_count", -1)) != 38
        or int(selected.get("matched_tie_count", -1)) != 3
        or int(selected.get("matched_owner_count", -1)) != 41
        or len(expected_owners) != 41
        or len(set(expected_owners)) != 41
        or dict(selected.get("gate", {})).get("debt")
        or fresh.get("final_generated_token_ids") != controlled
        or fresh.get("final_ledger") != dict(selected.get("evaluation", {})).get("ledger")
        or dict(fresh.get("final_gate_debt", {}))
    ):
        raise _hold("terminal witness/checkpoint/route/owner identity drifted")
    return {
        "receipt": receipt,
        "runtime_contract": contract,
        "blocks": blocks,
        "controlled_route_tokens": controlled,
        "controlled_route_sha256": CONTROLLED_ROUTE_SHA256,
        "expected_owner_ids": expected_owners,
        "expected_person_owner_ids": blocks[-1]["expected_person_owner_ids"],
        "expected_tie_owner_ids": blocks[-1]["expected_tie_owner_ids"],
    }


def _binding_receipt() -> dict[str, Any]:
    contract = _binding_contract()
    runtime = contract["runtime_contract"]
    blocks = contract["blocks"]
    return {
        "schema_version": SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "status": "verified",
        "execution_surface": "cpu_only_no_model_load_no_cuda",
        "source_receipt": str(SOURCE_RECEIPT),
        "source_receipt_sha256": SOURCE_RECEIPT_SHA256,
        "source_runner_sha256": SOURCE_RUNNER_SHA256,
        "model_receipt_sha256": MODEL_RECEIPT_SHA256,
        "checkpoint": str(recursive.START_CHECKPOINT),
        "checkpoint_readback_sha256": _value_sha256(runtime["checkpoint_readback"]),
        "start_surface_sha256": START_SURFACE_SHA256,
        "frozen_surface_sha256": FROZEN_SURFACE_SHA256,
        "prompt_token_ids_sha256": PROMPT_TOKEN_SHA256,
        "image_sha256": IMAGE_SHA256,
        "controlled_route_sha256": CONTROLLED_ROUTE_SHA256,
        "controlled_route_token_count": len(contract["controlled_route_tokens"]),
        "five_blocks": [
            {
                key: value for key, value in item.items()
                if key not in {"expected_route_tokens", "expected_owner_ids", "parent_owner_ids"}
            }
            | {"expected_route_token_count": len(item["expected_route_tokens"])}
            for item in blocks
        ],
        "stage_route_sha256": [item["expected_route_sha256"] for item in blocks],
        "static_task_position_count": sum(len(item["positions"]) for item in blocks),
        "positive_state_upper_bound": MAX_POSITIVES,
        "rank_upper_bound": MAX_RANK,
        "resource_bound": RESOURCE_BOUND,
        "runner_sha256": _sha256(Path(__file__)),
    }


class SparseOutputResidual(nn.Module):
    """One untied selected-row linear residual applied after an existing output head."""

    def __init__(
        self,
        base_head: nn.Module,
        selected_token_ids: Sequence[int],
        residual_rows: torch.Tensor,
    ) -> None:
        super().__init__()
        self.base_head = base_head
        self.register_buffer("selected_token_ids", torch.empty(0, dtype=torch.long))
        self.residual_rows = nn.Parameter(torch.empty((0, 0), dtype=torch.float64), requires_grad=False)
        self.set_payload(selected_token_ids, residual_rows)

    @property
    def weight(self) -> torch.Tensor:
        return self.base_head.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return getattr(self.base_head, "bias", None)

    def set_payload(self, selected_token_ids: Sequence[int], residual_rows: torch.Tensor) -> None:
        token_ids = torch.as_tensor(list(map(int, selected_token_ids)), dtype=torch.long)
        rows = torch.as_tensor(residual_rows).detach()
        if rows.ndim != 2 or rows.shape[0] != token_ids.numel():
            raise ValueError("residual rows must align one-to-one with selected token ids")
        if token_ids.numel() != torch.unique(token_ids).numel() or bool((token_ids < 0).any()):
            raise ValueError("selected token ids must be unique and non-negative")
        if rows.dtype not in {torch.float32, torch.float64} or not bool(torch.isfinite(rows).all()):
            raise ValueError("residual rows must be finite float32/float64")
        self.selected_token_ids = token_ids.to(self.selected_token_ids.device)
        self.residual_rows = nn.Parameter(
            rows.to(device=self.residual_rows.device, dtype=torch.float64).contiguous(),
            requires_grad=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.base_head(hidden_states)
        if self.selected_token_ids.numel() == 0:
            return logits
        correction = hidden_states.to(self.residual_rows.dtype) @ self.residual_rows.t()
        selected = self.selected_token_ids.to(logits.device)
        index = selected.view(*([1] * (correction.ndim - 1)), -1).expand_as(correction)
        result = logits.clone()
        result.scatter_add_(-1, index, correction.to(logits.dtype))
        return result


def _capture_route(
    *, model: Any, output_head: nn.Module, native_inputs: Mapping[str, Any],
    route_tokens: Sequence[int], pad_token_id: int,
) -> dict[str, torch.Tensor]:
    captured: list[torch.Tensor] = []

    def capture(_module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        captured.append(args[0].detach().cpu().clone())

    handle = output_head.register_forward_pre_hook(capture)
    try:
        with torch.inference_mode():
            logits = full_root._teacher_forced_route_logits(
                model=model,
                native_inputs=native_inputs,
                route_tokens=route_tokens,
                pad_token_id=pad_token_id,
            ).detach().cpu()
    finally:
        handle.remove()
    if len(captured) != 1:
        raise _hold("output-head hidden-state capture did not execute exactly once")
    hidden = captured[0]
    if hidden.ndim == 3 and hidden.shape[0] == 1:
        hidden = hidden[0]
    if hidden.ndim != 2 or tuple(hidden.shape[:1]) != (len(route_tokens),):
        raise _hold("captured output-head hidden states do not align to route labels")
    if logits.ndim != 2 or logits.shape[0] != len(route_tokens):
        raise _hold("captured raw logits do not align to route labels")
    if not bool(torch.isfinite(hidden).all()) or not bool(torch.isfinite(logits).all()):
        raise _hold("captured hidden states or raw logits are non-finite")
    return {"hidden": hidden, "logits": logits}


def _row_basis(matrix: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("row-basis input must be a finite matrix")
    if matrix.shape[0] == 0:
        return np.zeros((matrix.shape[1], 0), dtype=np.float64), {
            "rank": 0, "singular_values": [], "tolerance": 0.0,
        }
    _u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    tolerance = max(matrix.shape) * np.finfo(np.float64).eps * float(singular[0])
    rank = int(np.sum(singular > tolerance))
    return vh[:rank].T.copy(), {
        "rank": rank,
        "singular_values": singular.tolist(),
        "tolerance": tolerance,
    }


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
        len(positive) == 0
        or np.any(original_norms < PROJECTED_NORM_FLOOR)
        or np.any(projected_norms < PROJECTED_NORM_FLOOR)
    ):
        raise _hold("positive state has original/projected norm below 1e-3")
    basis, positive_info = _row_basis(projected)
    protected_null = protected @ basis
    if basis.shape[1] > MAX_RANK or float(np.max(np.abs(protected_null), initial=0.0)) > 1.0e-10:
        raise _hold("protected-null positive basis is rank-invalid or numerically non-null")
    return basis, {
        "protected": protected_info,
        "projected_positive": positive_info,
        "positive_count": int(len(positive)),
        "rank": int(basis.shape[1]),
        "original_norm_min": float(original_norms.min()),
        "projected_norm_min": float(projected_norms.min()),
        "protected_null_max_abs": float(np.max(np.abs(protected_null), initial=0.0)),
        "basis_sha256": _tensor_sha256(basis),
    }


def _protected_hidden_matrix(
    *,
    ordinary_route: Sequence[int],
    ordinary_hidden: torch.Tensor,
    controlled_route: Sequence[int],
    controlled_hidden: torch.Tensor,
    controlled_positive_positions: Sequence[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Exclude ordinary states that are the same prefix-state as a controlled positive."""
    positives = set(map(int, controlled_positive_positions))
    if (
        ordinary_hidden.ndim != 2
        or controlled_hidden.ndim != 2
        or ordinary_hidden.shape[0] != len(ordinary_route)
        or controlled_hidden.shape[0] != len(controlled_route)
        or ordinary_hidden.shape[1] != controlled_hidden.shape[1]
        or any(position < 0 or position >= len(controlled_route) for position in positives)
    ):
        raise ValueError("protected hidden-state routes/captures/positions are misaligned")
    positive_prefixes = {tuple(map(int, controlled_route[:position])) for position in positives}
    ordinary_excluded = [
        position for position in range(len(ordinary_route))
        if tuple(map(int, ordinary_route[:position])) in positive_prefixes
    ]
    ordinary_excluded_set = set(ordinary_excluded)
    ordinary_kept = [
        ordinary_hidden[position].double().numpy()
        for position in range(len(ordinary_route))
        if position not in ordinary_excluded_set
    ]
    controlled_kept = [
        controlled_hidden[position].double().numpy()
        for position in range(len(controlled_route))
        if position not in positives
    ]
    rows = [*ordinary_kept, *controlled_kept]
    if not rows:
        raise ValueError("protected hidden-state matrix is empty")
    matrix = np.stack(rows)
    return matrix, {
        "ordinary_state_count": len(ordinary_route),
        "ordinary_excluded_shared_positive_prefix_positions": ordinary_excluded,
        "ordinary_protected_count": len(ordinary_kept),
        "controlled_state_count": len(controlled_route),
        "controlled_positive_count": len(positives),
        "controlled_protected_count": len(controlled_kept),
        "protected_count": len(rows),
    }


def _constraint_key(item: Mapping[str, Any]) -> tuple[int, int, int]:
    return int(item["position"]), int(item["target_token_id"]), int(item["competitor_token_id"])


def _solve_minimum_normalized(
    constraints: Sequence[Mapping[str, Any]],
    *, rank: int, row_norms: Mapping[int, float],
) -> dict[str, Any]:
    selected = sorted({
        int(item[key]) for item in constraints
        for key in ("target_token_id", "competitor_token_id")
    })
    if not constraints or rank <= 0 or not selected:
        return {"feasible": False, "reason": "empty_constraint_or_basis"}
    token_to_row = {token: index for index, token in enumerate(selected)}
    width = len(selected) * rank
    g = np.zeros((len(constraints), width), dtype=np.float64)
    rhs = np.empty(len(constraints), dtype=np.float64)
    for index, item in enumerate(constraints):
        feature = np.asarray(item["feature"], dtype=np.float64)
        target = int(item["target_token_id"])
        competitor = int(item["competitor_token_id"])
        if feature.shape != (rank,) or target == competitor:
            return {"feasible": False, "reason": "malformed_constraint"}
        target_norm = float(row_norms[target])
        competitor_norm = float(row_norms[competitor])
        if min(target_norm, competitor_norm) <= 0.0:
            return {"feasible": False, "reason": "nonpositive_output_row_norm"}
        g[index, token_to_row[target] * rank : (token_to_row[target] + 1) * rank] += target_norm * feature
        g[index, token_to_row[competitor] * rank : (token_to_row[competitor] + 1) * rank] -= competitor_norm * feature
        rhs[index] = MARGIN - float(item["raw_margin"])
    impossible = (np.linalg.norm(g, axis=1) <= 1.0e-14) & (rhs > 1.0e-9)
    if np.any(impossible) or not np.isfinite(g).all() or not np.isfinite(rhs).all():
        return {"feasible": False, "reason": "zero_feature_positive_rhs"}

    gram = g @ g.T

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
    normalized = g.T @ np.asarray(outcome.x, dtype=np.float64)
    margins = g @ normalized - rhs
    if not np.isfinite(normalized).all() or float(margins.min(initial=math.inf)) < -2.0e-7:
        return {
            "feasible": False,
            "reason": "dual_solver_did_not_produce_feasible_primal",
            "solver_message": str(outcome.message),
            "minimum_slack": float(margins.min(initial=math.inf)),
        }
    normalized_rows = normalized.reshape(len(selected), rank)
    scaled_rows = np.stack([
        normalized_rows[index] * float(row_norms[token])
        for index, token in enumerate(selected)
    ])
    norm = float(np.linalg.norm(normalized))
    return {
        "feasible": True,
        "selected_token_ids": selected,
        "normalized_rows": normalized_rows,
        "scaled_basis_rows": scaled_rows,
        "normalized_norm": norm,
        "minimum_slack": float(margins.min(initial=math.inf)),
        "constraint_count": len(constraints),
        "variable_count": width,
        "solver_success": bool(outcome.success),
        "solver_message": str(outcome.message),
        "iterations": int(outcome.nit),
    }


def _greedy_route(model: Any, native_inputs: Mapping[str, Any], pad: int) -> list[int]:
    return full_root._greedy_release(
        model=model,
        native_inputs=native_inputs,
        prefix=(),
        eos_token_id=base.EOS,
        pad_token_id=pad,
    )


def _evaluate_route(
    *, tokenizer: Any, tokens: Sequence[int], contract: Mapping[str, Any],
    parent_owner_ids: Sequence[str], label: str,
) -> dict[str, Any]:
    return recursive.manifold._match_evaluation(
        tokenizer=tokenizer,
        tokens=tokens,
        target=contract["target"],
        witness_tokens=tokens,
        raw_example=contract["setup"]["raw_example"],
        parent_owners=parent_owner_ids,
        label=label,
    )


def _stage_gate(
    evaluation: Mapping[str, Any], *, block: Mapping[str, Any],
) -> dict[str, Any]:
    route = list(map(int, evaluation.get("generated_token_ids", ())))
    expected = list(map(int, block["expected_route_tokens"]))
    if block["stage"] < 5:
        generic = recursive._node_gate(evaluation, parent_owner_ids=block["parent_owner_ids"])
        owners = list(map(str, generic.get("matched_owner_ids", ())))
        debt = dict(generic.get("debt", {}))
        if route != expected:
            debt["exact_source_node_route"] = True
        if owners != list(map(str, block["expected_owner_ids"])):
            debt["source_node_owner_ledger"] = True
        return {"passed": not debt, "debt": debt, "route_exact": route == expected, "generic": generic}

    parent = list(map(str, block["expected_owner_ids"]))
    generic = recursive._node_gate(evaluation, parent_owner_ids=block["parent_owner_ids"])
    owners = list(map(str, generic.get("matched_owner_ids", ())))
    people = [owner for owner in owners if owner in set(map(str, block["expected_person_owner_ids"]))]
    ties = [owner for owner in owners if owner in set(map(str, block["expected_tie_owner_ids"]))]
    debt = dict(generic.get("debt", {}))
    if set(owners) != set(parent) or len(owners) != 41:
        debt["controlled_owner_set"] = True
    if set(people) != set(map(str, block["expected_person_owner_ids"])) or len(people) != 38:
        debt["all_38_persons"] = True
    if set(ties) != set(map(str, block["expected_tie_owner_ids"])) or len(ties) != 3:
        debt["same_three_ties"] = True
    return {
        "passed": not debt,
        "debt": debt,
        "route_exact": route == expected,
        "owner_equivalent": not debt,
        "generic": generic,
    }


def _first_route_drift(actual: Sequence[int], expected: Sequence[int]) -> dict[str, Any] | None:
    for position, (left, right) in enumerate(zip(actual, expected)):
        if int(left) != int(right):
            return {"position": position, "actual_token_id": int(left), "expected_token_id": int(right)}
    if len(actual) != len(expected):
        position = min(len(actual), len(expected))
        return {
            "position": position,
            "actual_token_id": None if position >= len(actual) else int(actual[position]),
            "expected_token_id": None if position >= len(expected) else int(expected[position]),
        }
    return None


def _row_norms(output_head: nn.Module, token_ids: Sequence[int]) -> dict[int, float]:
    weight = output_head.weight.detach()
    result = {}
    for token in sorted(set(map(int, token_ids))):
        if token < 0 or token >= weight.shape[0]:
            raise _hold(f"selected token {token} is outside output-head vocabulary")
        result[token] = float(weight[token].double().norm().item())
    return result


def _collapsed_rows(solution: Mapping[str, Any], basis: np.ndarray) -> np.ndarray:
    rows = np.asarray(solution["scaled_basis_rows"], dtype=np.float64) @ np.asarray(basis, dtype=np.float64).T
    if not np.isfinite(rows).all():
        raise _hold("collapsed residual contains non-finite values")
    return rows


def _corrected_logits(
    state: Mapping[str, Any], *, selected_token_ids: Sequence[int], residual_rows: np.ndarray,
) -> torch.Tensor:
    logits = state["logits"].clone()
    correction = np.asarray(state["hidden"], dtype=np.float64) @ np.asarray(residual_rows, dtype=np.float64).T
    logits[torch.tensor(selected_token_ids, dtype=torch.long)] += torch.from_numpy(correction).to(logits.dtype)
    return logits


def _full_vocab_violation(
    states: Sequence[Mapping[str, Any]], *, selected_token_ids: Sequence[int],
    residual_rows: np.ndarray,
) -> dict[str, Any] | None:
    violations = []
    for state in states:
        logits = _corrected_logits(state, selected_token_ids=selected_token_ids, residual_rows=residual_rows)
        target = int(state["target_token_id"])
        target_logit = float(logits[target].item())
        logits[target] = -torch.inf
        competitor_logit, competitor = logits.max(dim=0)
        margin = target_logit - float(competitor_logit.item())
        if not math.isfinite(margin) or margin < MARGIN - 2.0e-6:
            violations.append({
                "state": state,
                "competitor_token_id": int(competitor.item()),
                "corrected_margin": margin,
                "shortfall": MARGIN - margin,
            })
    return max(violations, key=lambda item: item["shortfall"]) if violations else None


def _constraint_for(state: Mapping[str, Any], competitor: int, basis: np.ndarray, kind: str) -> dict[str, Any]:
    target = int(state["target_token_id"])
    logits = state["logits"]
    return {
        "stage": int(state["stage"]),
        "position": int(state["position"]),
        "kind": kind,
        "target_token_id": target,
        "competitor_token_id": int(competitor),
        "raw_margin": float(logits[target].item() - logits[int(competitor)].item()),
        "feature": np.asarray(state["hidden"], dtype=np.float64) @ basis,
    }


def _initial_constraints(states: Sequence[Mapping[str, Any]], basis: np.ndarray, x1_tokens: Sequence[int]) -> list[dict[str, Any]]:
    constraints: list[dict[str, Any]] = []
    for state in states:
        target = int(state["target_token_id"])
        competitors = set(map(int, torch.topk(state["logits"], INITIAL_TOP_K).indices.tolist()))
        competitors.update((base.EOS, ROW_OPEN_TOKEN))
        if state.get("is_x1"):
            competitors.update(map(int, x1_tokens))
        competitors.discard(target)
        constraints.extend(_constraint_for(state, competitor, basis, "initial_active_competitor") for competitor in sorted(competitors))
    unique = {_constraint_key(item): item for item in constraints}
    return list(unique.values())


def _residual_identity(token_ids: Sequence[int], rows: torch.Tensor | np.ndarray) -> dict[str, Any]:
    return {
        "selected_token_ids": list(map(int, token_ids)),
        "selected_token_ids_sha256": _tensor_sha256(np.asarray(token_ids, dtype=np.int64)),
        "residual_rows_sha256": _tensor_sha256(rows),
        "residual_shape": list(np.asarray(rows).shape),
        "residual_dtype": str(np.asarray(rows).dtype),
    }


def _save_augmented_payload(
    output: Path, *, token_ids: Sequence[int], residual_rows: np.ndarray,
    solution: Mapping[str, Any], basis_info: Mapping[str, Any], contract: Mapping[str, Any],
) -> dict[str, Any]:
    payload = output / "protected_null_output_residual.safetensors"
    tensors = {
        "selected_token_ids": torch.tensor(list(map(int, token_ids)), dtype=torch.int64),
        "residual_rows": torch.from_numpy(np.asarray(residual_rows, dtype=np.float64)),
    }
    save_file(tensors, payload, metadata={"schema_version": SCHEMA_VERSION, "unit_id": UNIT_ID})
    identity = _residual_identity(token_ids, residual_rows)
    identity["payload_sha256"] = _sha256(payload)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "base_checkpoint": str(recursive.START_CHECKPOINT),
        "source_receipt_sha256": SOURCE_RECEIPT_SHA256,
        "controlled_route_sha256": CONTROLLED_ROUTE_SHA256,
        "normalization": "per-selected-base-output-row L2; global normalized Frobenius cap",
        "normalized_norm": float(solution["normalized_norm"]),
        "cap": NORM_CAP,
        "basis": deepcopy(dict(basis_info)),
        "residual_identity": identity,
    }
    metadata_path = output / "protected_null_output_residual.json"
    _atomic_json(metadata_path, metadata)
    manifest = {
        "schema_version": f"{SCHEMA_VERSION}.augmented_checkpoint_manifest",
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


def _surface_snapshot(model: Any) -> tuple[list[str], list[torch.nn.Parameter], dict[str, Any], str]:
    names, parameters = ota._step_trainable_surface(model, r32_step=True)
    surface = full_root._full_root_surface_snapshot(names, parameters)[1]
    frozen = base._frozen_surface(model, names)
    if surface.get("aggregate_sha256") != START_SURFACE_SHA256 or frozen != FROZEN_SURFACE_SHA256:
        raise _hold("frozen r32 DoRA/non-DoRA surface identity drifted")
    return names, parameters, surface, frozen


def _compact_constraints(constraints: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in item.items() if key != "feature"}
        | {"feature_sha256": _tensor_sha256(np.asarray(item["feature"], dtype=np.float64))}
        for item in constraints
    ]


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
    if token_ids is None or rows is None or token_ids.ndim != 1 or rows.ndim != 2 or rows.shape[0] != token_ids.numel():
        raise _hold("cold residual payload schema is invalid")
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
        ordinary_route = _greedy_route(model, native_inputs, int(tokenizer.pad_token_id))
        ordinary_capture = _capture_route(
            model=model, output_head=base_head, native_inputs=native_inputs,
            route_tokens=ordinary_route, pad_token_id=int(tokenizer.pad_token_id),
        )
        controlled_capture = _capture_route(
            model=model, output_head=base_head, native_inputs=native_inputs,
            route_tokens=binding["controlled_route_tokens"], pad_token_id=int(tokenizer.pad_token_id),
        )
        positive_positions = {
            position for block in binding["blocks"] for position in block["positions"]
            if int(binding["controlled_route_tokens"][position])
            != int(controlled_capture["logits"][position].argmax().item())
        }
        protected_hidden, protected_selection = _protected_hidden_matrix(
            ordinary_route=ordinary_route,
            ordinary_hidden=ordinary_capture["hidden"],
            controlled_route=binding["controlled_route_tokens"],
            controlled_hidden=controlled_capture["hidden"],
            controlled_positive_positions=sorted(positive_positions),
        )
        protected_hidden_sha256 = _tensor_sha256(protected_hidden)
        expected_protected_sha256 = dict(metadata.get("basis", {})).get("protected_hidden_sha256")
        if protected_hidden_sha256 != expected_protected_sha256:
            raise _hold("cold protected hidden-state matrix identity drifted")
        protected_correction = protected_hidden @ rows.double().numpy().T
        protected_null_max_abs = float(np.max(np.abs(protected_correction), initial=0.0))
        if protected_null_max_abs > 1.0e-10:
            raise _hold("cold residual exceeds numerical protected-null bound 1e-10")
        wrapper = SparseOutputResidual(base_head, token_ids.tolist(), rows).to(next(model.parameters()).device)
        model.set_output_embeddings(wrapper)
        route = _greedy_route(model, native_inputs, int(tokenizer.pad_token_id))
        evaluation = _evaluate_route(
            tokenizer=tokenizer,
            tokens=route,
            contract=contract,
            parent_owner_ids=contract["parent_owner_ids"],
            label="protected-null-final",
        )
        gate = _stage_gate(evaluation, block=binding["blocks"][-1])
        residual_identity = _residual_identity(token_ids.tolist(), rows)
        residual_identity["payload_sha256"] = _sha256(payload)
        model.set_output_embeddings(base_head)
        full_root._assert_full_root_sentinels(model, sentinels)
        _names, _parameters, surface_after, frozen_after = _surface_snapshot(model)
        if names != _names or any(left is not right for left, right in zip(parameters, _parameters, strict=True)):
            raise _hold("cold base parameter identity changed after residual removal")
        runtime = opened.receipt.to_artifact_dict()
    _atomic_json(result, {
        "schema_version": SCHEMA_VERSION,
        "status": "cold_verification_complete",
        "generated_token_ids": route,
        "generated_token_ids_sha256": token_ids_sha256(route),
        "ledger": evaluation["ledger"],
        "gate": gate,
        "residual_identity": residual_identity,
        "protected_hidden_sha256": protected_hidden_sha256,
        "protected_selection": protected_selection,
        "protected_null_max_abs": protected_null_max_abs,
        "protected_null_tolerance": 1.0e-10,
        "surface": {"before": surface_before, "after": surface_after},
        "frozen_surface": {"before": frozen_before, "after": frozen_after},
        "runtime": runtime,
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        "model_load_count": 1,
    })


def _require_one_gpu() -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or torch.cuda.device_count() != 1:
        raise _hold("protected-null distillation requires one visible GPU and world size 1")


def _prepare_output(output: Path) -> dict[str, Any]:
    if output.exists():
        raise _hold(f"refusing overwrite: {output}")
    output.mkdir(parents=True)
    snapshot = output / "runner_source.py"
    source = Path(__file__).read_bytes()
    snapshot.write_bytes(source)
    if snapshot.read_bytes() != source:
        raise _hold("runner source snapshot readback drifted")
    return {"path": str(snapshot), "sha256": _sha256(snapshot)}


def _artifact_bytes(output: Path) -> int:
    return sum(path.stat().st_size for path in output.iterdir() if path.is_file())


def run(*, run_id: str) -> Path:
    _require_one_gpu()
    binding = _binding_contract()
    contract = binding["runtime_contract"]
    run_id = base._safe_run_id(run_id)
    output = OUTPUT_ROOT / run_id
    started = time.monotonic()
    snapshot = _prepare_output(output)
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "run_id": run_id,
        "status": "technical_hold",
        "runner_source_snapshot": snapshot,
        "bindings": _binding_receipt(),
        "counts": {"model_loads": 0, "solves": 0, "warm_candidates": 0, "base_greedy": 0, "zero_wrapper_greedy": 0},
        "stages": [],
    }
    stage = "single_warm_model_load"
    warm_final: dict[str, Any] | None = None
    payload: dict[str, Any] | None = None
    base_head = wrapper = model = None
    terminal_status: str | None = None
    try:
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
            if token_ids_sha256(prompts[0]) != PROMPT_TOKEN_SHA256 or _sha256(Path(setup["raw_example"].image.path)) != IMAGE_SHA256:
                raise _hold("warm prompt/image identity drifted")
            names, parameters, surface_before, frozen_before = _surface_snapshot(model)
            sentinels = full_root._full_root_nontrainable_sentinels(model)
            base_head = model.get_output_embeddings()

            stage = "capture_raw_ordinary_and_controlled"
            ordinary_route = _greedy_route(model, native_inputs, int(tokenizer.pad_token_id))
            receipt["counts"]["base_greedy"] = 1
            ordinary = _capture_route(
                model=model, output_head=base_head, native_inputs=native_inputs,
                route_tokens=ordinary_route, pad_token_id=int(tokenizer.pad_token_id),
            )
            controlled = _capture_route(
                model=model, output_head=base_head, native_inputs=native_inputs,
                route_tokens=binding["controlled_route_tokens"], pad_token_id=int(tokenizer.pad_token_id),
            )
            ordinary_top1 = ordinary["logits"].argmax(dim=1).tolist()
            if ordinary_top1 != ordinary_route:
                raise _hold("raw argmax teacher logits do not reproduce ordinary HF greedy")

            all_target_ids = sorted({token for block in binding["blocks"] for token in block["target_token_ids"]})
            hidden_width = int(controlled["hidden"].shape[1])
            wrapper = SparseOutputResidual(
                base_head, all_target_ids,
                torch.zeros((len(all_target_ids), hidden_width), dtype=torch.float64),
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
                raise _hold("zero residual wrapper failed hidden/logit/generate parity")

            stage = "protected_null_basis"
            intervention_positions = {
                position: block["stage"]
                for block in binding["blocks"] for position in block["positions"]
            }
            controlled_tokens = binding["controlled_route_tokens"]
            controlled_top1 = controlled["logits"].argmax(dim=1).tolist()
            positive_states: list[dict[str, Any]] = []
            for position, (target, top1) in enumerate(zip(controlled_tokens, controlled_top1, strict=True)):
                is_positive = position in intervention_positions and int(target) != int(top1)
                if is_positive:
                    block = binding["blocks"][intervention_positions[position] - 1]
                    positive_states.append({
                        "stage": int(block["stage"]),
                        "position": position,
                        "target_token_id": int(target),
                        "hidden": controlled["hidden"][position].double().numpy(),
                        "logits": controlled["logits"][position],
                        "is_x1": position == block.get("x1_position"),
                    })
                else:
                    if position not in intervention_positions and int(target) != int(top1):
                        raise _hold("controlled route has an unrecorded non-top1 intervention")
            if not positive_states or len(positive_states) > MAX_POSITIVES:
                raise _hold("observed positive count is outside 1..30")
            positive_matrix = np.stack([item["hidden"] for item in positive_states])
            protected_matrix, protected_selection = _protected_hidden_matrix(
                ordinary_route=ordinary_route,
                ordinary_hidden=ordinary["hidden"],
                controlled_route=controlled_tokens,
                controlled_hidden=controlled["hidden"],
                controlled_positive_positions=[item["position"] for item in positive_states],
            )
            try:
                basis, basis_info = _protected_positive_basis(protected_matrix, positive_matrix)
            except ProtectedNullHold:
                terminal_status = "nullspace_infeasible"
                raise
            if basis.shape[1] == 0:
                terminal_status = "nullspace_infeasible"
                raise _hold("projected positive span has rank zero")
            basis_info["protected_hidden_sha256"] = _tensor_sha256(protected_matrix)
            basis_info["protected_hidden_shape"] = list(protected_matrix.shape)
            basis_info["protected_null_apply_contract"] = (
                "FP64 payload; hidden cast FP64; hidden@D.T FP64; correction cast to logits dtype"
            )
            basis_info["protected_null_tolerance"] = 1.0e-10
            for item in positive_states:
                item["feature"] = np.asarray(item["hidden"], dtype=np.float64) @ basis
            x1_tokens = [int(item["x1_token_id"]) for item in binding["blocks"][:4]]
            constraints: list[dict[str, Any]] = []
            accepted_states: list[dict[str, Any]] = []
            current_solution: dict[str, Any] | None = None
            current_rows: np.ndarray | None = None

            receipt["capture"] = {
                "ordinary_route_sha256": token_ids_sha256(ordinary_route),
                "ordinary_route_token_count": len(ordinary_route),
                "controlled_route_sha256": CONTROLLED_ROUTE_SHA256,
                "controlled_route_token_count": len(controlled_tokens),
                "raw_argmax_matches_ordinary_generate": True,
                "zero_wrapper_hidden_logits_generate_exact": True,
                "positive_count": len(positive_states),
                "protected_count": len(protected_matrix),
                "protected_selection": protected_selection,
                "positive_positions": [int(item["position"]) for item in positive_states],
                "basis": basis_info,
            }

            stage = "five_staged_solves"
            for block in binding["blocks"]:
                stage_number = int(block["stage"])
                new_states = [item for item in positive_states if item["stage"] == stage_number]
                constraints.extend(_initial_constraints(new_states, basis, x1_tokens))
                constraints = list({_constraint_key(item): item for item in constraints}.values())
                accumulated_states = [item for item in positive_states if item["stage"] <= stage_number]
                stage_record: dict[str, Any] = {
                    "stage": stage_number,
                    "expected_route_sha256": block["expected_route_sha256"],
                    "positive_count_added": len(new_states),
                    "closures": [],
                    "attempts": [],
                }
                receipt["stages"].append(stage_record)
                accepted = False
                for attempt in range(MAX_CLOSURES_PER_STAGE + 1):
                    if receipt["counts"]["solves"] >= MAX_SOLVES:
                        terminal_status = "stage_exhausted"
                        raise _hold("global 15-solve bound exhausted")
                    selected_for_norm = {
                        int(item[key]) for item in constraints
                        for key in ("target_token_id", "competitor_token_id")
                    }
                    norms = _row_norms(base_head, selected_for_norm)
                    solved = _solve_minimum_normalized(constraints, rank=basis.shape[1], row_norms=norms)
                    receipt["counts"]["solves"] += 1
                    attempt_record = {
                        "attempt": attempt,
                        "constraint_count": len(constraints),
                        "solver": {key: value for key, value in solved.items() if key not in {"normalized_rows", "scaled_basis_rows"}},
                    }
                    stage_record["attempts"].append(attempt_record)
                    if not solved.get("feasible"):
                        terminal_status = "nullspace_infeasible"
                        raise _hold(f"stage {stage_number} protected-null inequalities are infeasible")
                    if float(solved["normalized_norm"]) > NORM_CAP + 1.0e-8:
                        terminal_status = "norm_cap_exceeded"
                        raise _hold(f"stage {stage_number} minimum normalized norm exceeds 1")
                    rows = _collapsed_rows(solved, basis)
                    selected_ids = list(map(int, solved["selected_token_ids"]))
                    protected_correction = protected_matrix @ rows.T
                    null_max = float(np.max(np.abs(protected_correction), initial=0.0))
                    if null_max > 1.0e-10:
                        terminal_status = "nullspace_infeasible"
                        raise _hold("collapsed residual is not numerically null on protected states")
                    violation = _full_vocab_violation(
                        [*accumulated_states, *accepted_states],
                        selected_token_ids=selected_ids,
                        residual_rows=rows,
                    )
                    attempt_record["protected_null_max_abs"] = null_max
                    attempt_record["protected_null_tolerance"] = 1.0e-10
                    if violation is not None:
                        competitor = int(violation["competitor_token_id"])
                        closure = _constraint_for(violation["state"], competitor, basis, "full_vocab_competitor_closure")
                        if attempt >= MAX_CLOSURES_PER_STAGE or _constraint_key(closure) in {_constraint_key(item) for item in constraints}:
                            terminal_status = "stage_exhausted"
                            raise _hold(f"stage {stage_number} full-vocabulary closure exhausted")
                        constraints.append(closure)
                        stage_record["closures"].append({
                            "kind": "full_vocab_competitor",
                            "position": int(closure["position"]),
                            "target_token_id": int(closure["target_token_id"]),
                            "competitor_token_id": competitor,
                            "corrected_margin": float(violation["corrected_margin"]),
                        })
                        continue

                    wrapper.set_payload(selected_ids, torch.from_numpy(rows))
                    candidate_route = _greedy_route(model, native_inputs, int(tokenizer.pad_token_id))
                    receipt["counts"]["warm_candidates"] += 1
                    evaluation = _evaluate_route(
                        tokenizer=tokenizer,
                        tokens=candidate_route,
                        contract=contract,
                        parent_owner_ids=(block.get("parent_owner_ids") or contract["parent_owner_ids"]),
                        label="protected-null-final" if stage_number == 5 else f"protected-null-stage-{stage_number}",
                    )
                    gate = _stage_gate(evaluation, block=block)
                    attempt_record.update({
                        "candidate_route_sha256": token_ids_sha256(candidate_route),
                        "candidate_route_token_count": len(candidate_route),
                        "gate": gate,
                        "normalized_norm": float(solved["normalized_norm"]),
                        "residual_identity": _residual_identity(selected_ids, rows),
                    })
                    if gate["passed"]:
                        accepted = True
                        current_solution, current_rows = solved, rows
                        stage_record["accepted_attempt"] = attempt
                        stage_record["accepted_route_sha256"] = token_ids_sha256(candidate_route)
                        if stage_number == 5:
                            warm_final = {
                                "generated_token_ids": candidate_route,
                                "generated_token_ids_sha256": token_ids_sha256(candidate_route),
                                "ledger": deepcopy(dict(evaluation["ledger"])),
                                "gate": gate,
                                "surface": surface_before,
                                "frozen_surface_sha256": frozen_before,
                                "residual_identity": _residual_identity(selected_ids, rows),
                                "protected_hidden_sha256": basis_info["protected_hidden_sha256"],
                                "protected_null_max_abs": null_max,
                                "protected_null_tolerance": 1.0e-10,
                            }
                        break

                    drift = _first_route_drift(candidate_route, block["expected_route_tokens"])
                    if drift is None or drift["expected_token_id"] is None or drift["actual_token_id"] is None:
                        terminal_status = "stage_exhausted"
                        raise _hold(f"stage {stage_number} failed without a closable earliest drift")
                    position = int(drift["position"])
                    if position >= len(controlled_tokens):
                        terminal_status = "stage_exhausted"
                        raise _hold(f"stage {stage_number} earliest drift is outside controlled capture")
                    drift_state = {
                        "stage": stage_number,
                        "position": position,
                        "target_token_id": int(drift["expected_token_id"]),
                        "hidden": controlled["hidden"][position].double().numpy(),
                        "logits": controlled["logits"][position],
                        "is_x1": False,
                    }
                    closure = _constraint_for(drift_state, int(drift["actual_token_id"]), basis, "earliest_drift_closure")
                    if attempt >= MAX_CLOSURES_PER_STAGE or _constraint_key(closure) in {_constraint_key(item) for item in constraints}:
                        terminal_status = "stage_exhausted"
                        raise _hold(f"stage {stage_number} earliest-drift closure exhausted")
                    constraints.append(closure)
                    accepted_states.append(drift_state)
                    stage_record["closures"].append({"kind": "earliest_drift", **drift})
                if not accepted:
                    terminal_status = "stage_exhausted"
                    raise _hold(f"stage {stage_number} exhausted two closures")

            if warm_final is None or current_solution is None or current_rows is None:
                terminal_status = "stage_exhausted"
                raise _hold("five stages ended without warm final acceptance")
            payload = _save_augmented_payload(
                output,
                token_ids=current_solution["selected_token_ids"],
                residual_rows=current_rows,
                solution=current_solution,
                basis_info=basis_info,
                contract=binding,
            )
            warm_final["residual_identity"]["payload_sha256"] = payload["identity"]["payload_sha256"]
            receipt["constraints"] = _compact_constraints(constraints)
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
            model,
            wrapper,
            base_head,
            parameters,
            parameters_after,
            parameter,
            opened,
            tokenizer,
            native_inputs,
            prompts,
            _grids,
            _media,
            sentinels,
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
                sys.executable, str(snapshot_path), "--cold-verify",
                "--payload", str(payload["payload"]), "--cold-result", str(cold_path),
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
                "fresh cold subprocess failed: "
                + (completed.stderr[-2_000:] or completed.stdout[-2_000:] or f"exit {completed.returncode}")
            )
        cold = json.loads(cold_path.read_text(encoding="utf-8"))
        receipt["counts"]["model_loads"] = 2
        receipt["runtime"]["cold"] = cold.get("runtime")
        receipt["cold_final"] = cold
        warm_cold = {
            "route_exact": warm_final["generated_token_ids"] == cold.get("generated_token_ids"),
            "ledger_exact": warm_final["ledger"] == cold.get("ledger"),
            "residual_exact": warm_final["residual_identity"] == cold.get("residual_identity"),
            "warm_gate_passed": warm_final["gate"].get("passed") is True,
            "cold_gate_passed": dict(cold.get("gate", {})).get("passed") is True,
            "warm_surface_frozen": receipt["warm_surface"]["before"] == receipt["warm_surface"]["after"],
            "cold_surface_frozen": dict(cold.get("surface", {})).get("before") == dict(cold.get("surface", {})).get("after"),
            "protected_hidden_exact": warm_final["protected_hidden_sha256"] == cold.get("protected_hidden_sha256"),
            "warm_protected_null": float(warm_final["protected_null_max_abs"]) <= 1.0e-10,
            "cold_protected_null": float(cold.get("protected_null_max_abs", math.inf)) <= 1.0e-10,
        }
        receipt["warm_cold"] = warm_cold
        if not all(warm_cold.values()):
            raise _hold("warm/cold route, ledger, residual, gate, or frozen surface mismatch")
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
        receipt["status"] = terminal_status
        receipt["wall_time_seconds"] = elapsed
        receipt["resources"] = {
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
            "artifact_bytes_before_receipt": _artifact_bytes(output),
            "predeclared_bound": RESOURCE_BOUND,
        }
        receipt["claim_boundary"] = (
            "One Image2299 frozen-r32 augmented-model ordinary-greedy result only; "
            "not transfer, general enumeration learning, a base-r32 greedy result, or 8/8 tie recovery."
        )
        if terminal_status not in TERMINAL_STATUSES:
            receipt["status"] = "technical_hold"
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
