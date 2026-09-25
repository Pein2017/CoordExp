"""Selected-output-row minimum-Frobenius QP on explicit captured states.

The solver and FP32 separator are retained from the finite-panel research method.
A certificate covers these supplied hidden states and full-vocabulary evidence,
not natural greedy trajectories, physical owners or transfer. No model is loaded.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

MARGIN = 0.01
CERTIFICATE_TOLERANCE = 2e-5
MAX_OUTER_SOLVES = 64
MAX_ACTIVE_CONSTRAINTS = 20000
MAX_ACTIVE_DUAL_BYTES = 512 * 1024 * 1024

class HoldError(RuntimeError):
    """A fail-closed mechanics boundary, reported as HOLD by the CLI."""

def sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()

class SelectedOutputRowsHook:
    """Add an immutable residual to selected output rows only."""

    def __init__(self, module: Any, selected_token_ids: Any, residual_rows: Any) -> None:
        import torch

        self.module = module
        self.selected_token_ids = torch.as_tensor(selected_token_ids, dtype=torch.long)
        self.residual_rows = torch.as_tensor(residual_rows, dtype=torch.float64)
        if self.residual_rows.ndim != 2:
            raise ValueError("residual_rows must be a matrix")
        if self.residual_rows.shape[0] != self.selected_token_ids.numel():
            raise ValueError("selected ids and residual rows differ")
        if len(set(int(x) for x in self.selected_token_ids.tolist())) != int(
            self.selected_token_ids.numel()
        ):
            raise ValueError("selected token ids must be unique")
        self.handle: Any | None = None
        self.call_count = 0

    def _hook(self, _module: Any, args: tuple[Any, ...], output: Any) -> Any:
        import torch

        if not isinstance(output, torch.Tensor) or not args or not isinstance(args[0], torch.Tensor):
            raise RuntimeError("output-head hook requires tensor input/output")
        self.call_count += 1
        if self.selected_token_ids.numel() == 0:
            return output
        hidden = args[0]
        if hidden.shape[-1] != self.residual_rows.shape[1]:
            raise RuntimeError("output-head hidden width differs from payload")
        ids = self.selected_token_ids.to(output.device)
        rows = self.residual_rows.to(device=output.device, dtype=hidden.dtype)
        delta = torch.matmul(hidden, rows.transpose(0, 1)).to(output.dtype)
        changed = output.clone()
        changed[..., ids] = changed[..., ids] + delta
        return changed

    def __enter__(self) -> "SelectedOutputRowsHook":
        if self.handle is not None:
            raise RuntimeError("output-head hook already installed")
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *_exc: Any) -> None:
        assert self.handle is not None
        self.handle.remove()
        self.handle = None

def recover_fixed_max(
    top_ids: Sequence[int],
    top_logits: Sequence[float],
    selected_ids: set[int],
    *,
    target_id: int,
) -> tuple[int, float]:
    """Return the exact best unchanged competitor from a sufficient top-K."""

    if len(top_ids) != len(top_logits):
        raise ValueError("top ids/logits differ")
    if any(float(left) < float(right) for left, right in zip(top_logits, top_logits[1:])):
        raise ValueError("top-K logits must be sorted descending")
    for token_id, logit in zip(top_ids, top_logits, strict=True):
        token_id = int(token_id)
        if token_id != int(target_id) and token_id not in selected_ids:
            return token_id, float(logit)
    raise HoldError("top-K does not contain an unchanged non-target competitor")

def select_trainable_rows(
    target_ids: Sequence[int],
    top_ids: Any,
    top_logits: Any,
    *,
    target_logits: Sequence[float] | None = None,
    margin: float = MARGIN,
) -> tuple[int, ...]:
    """Select route targets whose exact base full-vocabulary margin is deficient."""

    import numpy as np

    targets = np.asarray(target_ids, dtype=np.int64)
    ids = np.asarray(top_ids, dtype=np.int64)
    logits = np.asarray(top_logits, dtype=np.float64)
    if ids.shape != logits.shape or ids.ndim != 2 or ids.shape[0] != targets.size:
        raise ValueError("top-K evidence shape differs from target positions")
    explicit_targets = (
        None if target_logits is None else np.asarray(target_logits, dtype=np.float64)
    )
    if explicit_targets is not None and explicit_targets.shape != targets.shape:
        raise ValueError("target logits shape differs from target positions")
    selected: set[int] = set()
    for pos, target in enumerate(targets.tolist()):
        competitor = max(
            float(value)
            for token, value in zip(ids[pos], logits[pos], strict=True)
            if int(token) != target
        )
        if explicit_targets is None:
            target_matches = [
                float(value)
                for token, value in zip(ids[pos], logits[pos], strict=True)
                if int(token) == target
            ]
            if len(target_matches) != 1:
                raise HoldError("target logit must be explicit when target is outside top-K")
            target_logit = target_matches[0]
        else:
            target_logit = float(explicit_targets[pos])
        if target_logit - competitor < margin:
            selected.add(target)
    return tuple(sorted(selected))

@dataclass(frozen=True)
class _Constraint:
    position: int
    target_row: int | None
    competitor_row: int | None
    rhs: float
    kind: str
    competitor_token_id: int

    @property
    def key(self) -> tuple[int, int | None, int | None, str]:
        return (self.position, self.target_row, self.competitor_row, self.kind)

def _lhs(constraint: _Constraint, coefficients: Any, state: Any) -> float:
    value = 0.0
    if constraint.target_row is not None:
        value += float(coefficients[constraint.target_row] @ state)
    if constraint.competitor_row is not None:
        value -= float(coefficients[constraint.competitor_row] @ state)
    return value

def solve_minimum_frobenius(
    *,
    hidden_states: Any,
    target_ids: Sequence[int],
    route_token_ids: Sequence[int],
    base_route_logits: Any,
    top_ids: Any,
    top_logits: Any,
    margin: float = MARGIN,
    certificate_tolerance: float = CERTIFICATE_TOLERANCE,
    max_outer_solves: int = MAX_OUTER_SOLVES,
    max_active_constraints: int = MAX_ACTIVE_CONSTRAINTS,
) -> dict[str, Any]:
    """Solve the selected-row QP with an exhaustive cutting-plane separator."""

    import numpy as np
    from scipy.optimize import minimize

    hidden = np.asarray(hidden_states, dtype=np.float64)
    targets = np.asarray(target_ids, dtype=np.int64)
    route_ids = np.asarray(route_token_ids, dtype=np.int64)
    route_logits = np.asarray(base_route_logits, dtype=np.float64)
    top_token_ids = np.asarray(top_ids, dtype=np.int64)
    top_values = np.asarray(top_logits, dtype=np.float64)
    if hidden.ndim != 2 or hidden.shape[0] != targets.size:
        raise ValueError("hidden states differ from target positions")
    if route_logits.shape != (targets.size, route_ids.size):
        raise ValueError("route logits shape differs")
    if top_token_ids.shape != top_values.shape or top_values.shape[0] != targets.size:
        raise ValueError("top-K shape differs")
    if len(set(route_ids.tolist())) != route_ids.size:
        raise ValueError("route token ids must be unique")
    if top_values.shape[1] <= route_ids.size:
        raise HoldError("K must exceed the number of unique route targets")
    route_index = {int(token): index for index, token in enumerate(route_ids)}
    if any(int(token) not in route_index for token in targets):
        raise HoldError("target token is absent from route-logit evidence")
    target_base_logits = np.asarray(
        [route_logits[pos, route_index[int(target)]] for pos, target in enumerate(targets)],
        dtype=np.float64,
    )
    selected_ids = select_trainable_rows(
        targets,
        top_token_ids,
        top_values,
        target_logits=target_base_logits,
        margin=margin,
    )
    selected_set = set(selected_ids)
    selected_index = {token: index for index, token in enumerate(selected_ids)}
    if not selected_ids:
        raise HoldError("no selected rows: zero residual already satisfies teacher-forced margins")

    _, singular, vh = np.linalg.svd(hidden, full_matrices=False)
    if singular.size == 0 or singular[0] == 0.0:
        raise HoldError("captured hidden states have zero span")
    rank_tolerance = max(hidden.shape) * np.finfo(np.float64).eps * singular[0]
    rank = int(np.sum(singular > rank_tolerance))
    basis = vh[:rank]
    projected = hidden @ basis.T

    fixed: list[tuple[int, float]] = []
    for pos, target in enumerate(targets.tolist()):
        fixed.append(
            recover_fixed_max(
                top_token_ids[pos],
                top_values[pos],
                selected_set,
                target_id=target,
            )
        )

    deficient_position_count = sum(
        1
        for pos, target in enumerate(targets.tolist())
        if target_base_logits[pos]
        - max(
            float(value)
            for token, value in zip(top_token_ids[pos], top_values[pos], strict=True)
            if int(token) != target
        )
        < margin
    )

    def base(token: int, pos: int) -> float:
        return float(route_logits[pos, route_index[token]])

    def candidate_constraints(pos: int) -> list[_Constraint]:
        target = int(targets[pos])
        target_row = selected_index.get(target)
        result: list[_Constraint] = []
        for competitor in selected_ids:
            if competitor == target:
                continue
            result.append(
                _Constraint(
                    pos,
                    target_row,
                    selected_index[competitor],
                    margin - base(target, pos) + base(competitor, pos),
                    "selected",
                    competitor,
                )
            )
        # A fixed target was not deficient at base and all fixed rows remain
        # unchanged.  Only selected competitors can threaten it after fitting.
        if target_row is not None:
            fixed_id, fixed_logit = fixed[pos]
            result.append(
                _Constraint(
                    pos,
                    target_row,
                    None,
                    margin - base(target, pos) + fixed_logit,
                    "fixed_max",
                    fixed_id,
                )
            )
        return result

    active: list[_Constraint] = []
    active_keys: set[tuple[int, int | None, int | None, str]] = set()
    coefficients = np.zeros((len(selected_ids), rank), dtype=np.float64)
    dual = np.zeros(0, dtype=np.float64)
    outer_solve_count = 0
    optimizer_iterations = 0

    def separate(current: Any) -> tuple[float, list[_Constraint], _Constraint | None]:
        worst = -math.inf
        worst_constraint: _Constraint | None = None
        additions: list[_Constraint] = []
        for pos in range(targets.size):
            state_worst = -math.inf
            state_constraint: _Constraint | None = None
            for constraint in candidate_constraints(pos):
                violation = constraint.rhs - _lhs(constraint, current, projected[pos])
                if violation > worst:
                    worst, worst_constraint = violation, constraint
                if violation > state_worst:
                    state_worst, state_constraint = violation, constraint
            if (
                state_constraint is not None
                and state_worst > certificate_tolerance
                and state_constraint.key not in active_keys
            ):
                additions.append(state_constraint)
        return worst, additions, worst_constraint

    while True:
        worst, additions, largest = separate(coefficients)
        if worst <= certificate_tolerance:
            break
        if outer_solve_count >= max_outer_solves:
            raise HoldError("cutting-plane outer solve limit reached")
        if not additions:
            raise HoldError("violated constraints remain but no cutting plane was added")
        if len(active) + len(additions) > max_active_constraints:
            raise HoldError("active constraint ceiling reached before certificate")
        active.extend(additions)
        active_keys.update(item.key for item in additions)
        if len(active) * 8 > MAX_ACTIVE_DUAL_BYTES:
            raise HoldError("active dual allocation ceiling reached")
        start = np.pad(dual, (0, len(active) - dual.size))

        def reconstruct(lambdas: Any) -> Any:
            current = np.zeros_like(coefficients)
            for weight, constraint in zip(lambdas, active, strict=True):
                if weight == 0.0:
                    continue
                z = projected[constraint.position]
                if constraint.target_row is not None:
                    current[constraint.target_row] += weight * z
                if constraint.competitor_row is not None:
                    current[constraint.competitor_row] -= weight * z
            return current

        def objective(lambdas: Any) -> tuple[float, Any]:
            current = reconstruct(lambdas)
            value = 0.5 * float(np.sum(current * current)) - float(
                np.dot(lambdas, [item.rhs for item in active])
            )
            gradient = np.asarray(
                [
                    _lhs(item, current, projected[item.position]) - item.rhs
                    for item in active
                ],
                dtype=np.float64,
            )
            return value, gradient

        outer_solve_count += 1
        optimizer_ftol = 1e-14
        for segment_index in range(1, 4):
            polish = optimizer_ftol == 0.0
            result = minimize(
                objective,
                start,
                jac=True,
                method="L-BFGS-B",
                bounds=[(0.0, None)] * len(active),
                options={
                    "ftol": optimizer_ftol,
                    "gtol": 1e-10,
                    "maxiter": 4000,
                    "maxls": 50,
                },
            )
            optimizer_iterations += int(result.nit)
            candidate_dual = np.asarray(result.x, dtype=np.float64)
            candidate_coefficients = reconstruct(candidate_dual)
            _, candidate_gradient = objective(candidate_dual)
            candidate_primal = 0.5 * float(
                np.sum(candidate_coefficients * candidate_coefficients)
            )
            candidate_dual_objective = float(
                np.dot(candidate_dual, [item.rhs for item in active])
            ) - candidate_primal
            candidate_worst_violation, candidate_additions, _ = separate(
                candidate_coefficients
            )
            progress = {
                "schema": "human13_output_qp_solver_progress.v1",
                "diagnostic_outer_solve_index": outer_solve_count,
                "diagnostic_segment_index": segment_index,
                "diagnostic_segment_limit": 3,
                "diagnostic_polish": polish,
                "diagnostic_optimizer_ftol": optimizer_ftol,
                "diagnostic_active_constraint_count": len(active),
                "diagnostic_active_set_sha256": sha256_json([item.key for item in active]),
                "diagnostic_result_success": bool(result.success),
                "diagnostic_result_status": int(result.status),
                "diagnostic_result_message": str(result.message),
                "diagnostic_result_nit": int(result.nit),
                "diagnostic_result_nfev": int(result.nfev),
                "diagnostic_result_fun": float(result.fun),
                "diagnostic_lambda_l2_norm": float(np.linalg.norm(candidate_dual)),
                "diagnostic_lambda_max": float(np.max(candidate_dual)),
                "diagnostic_projected_gradient_kkt_inf_norm": float(
                    np.linalg.norm(
                        candidate_dual
                        - np.maximum(0.0, candidate_dual - candidate_gradient),
                        ord=np.inf,
                    )
                ),
                "diagnostic_full_registered_worst_primal_violation": float(
                    candidate_worst_violation
                ),
                "diagnostic_primal_half_frobenius_squared": candidate_primal,
                "diagnostic_dual_objective": candidate_dual_objective,
                "diagnostic_candidate_gap": candidate_primal - candidate_dual_objective,
                "diagnostic_active_complementarity_max": float(
                    np.max(np.abs(candidate_dual * candidate_gradient))
                ),
            }
            if hasattr(result, "njev"):
                progress["diagnostic_result_njev"] = int(result.njev)
            print(
                json.dumps(progress, sort_keys=True, separators=(",", ":")),
                flush=True,
            )
            if result.success:
                if (
                    candidate_worst_violation <= certificate_tolerance
                    or candidate_additions
                    or segment_index == 3
                ):
                    break
                start = candidate_dual
                optimizer_ftol = 0.0
                continue
            if (
                int(result.status) == 1
                and "TOTAL NO. OF ITERATIONS REACHED LIMIT" in str(result.message)
                and segment_index < 3
            ):
                start = candidate_dual
                continue
            raise HoldError(f"QP dual optimizer failed: {result.message}")
        dual = candidate_dual
        coefficients = candidate_coefficients

    residual_rows = coefficients @ basis
    # Replay the actual hook arithmetic boundary: FP64 payload rows are cast to
    # FP32 and multiplied by FP32 hidden states before addition to FP32 logits.
    fp32_delta = (
        hidden.astype(np.float32) @ residual_rows.astype(np.float32).T
    ).astype(np.float32)
    minimum_fp32_margin = math.inf
    worst_fp32: dict[str, Any] | None = None
    for pos, target in enumerate(targets.tolist()):
        target_delta = (
            float(fp32_delta[pos, selected_index[target]]) if target in selected_set else 0.0
        )
        target_logit = np.float32(base(target, pos)) + np.float32(target_delta)
        for competitor in selected_ids:
            if competitor == target:
                continue
            margin_value = float(
                target_logit
                - (
                    np.float32(base(competitor, pos))
                    + np.float32(fp32_delta[pos, selected_index[competitor]])
                )
            )
            if margin_value < minimum_fp32_margin:
                minimum_fp32_margin = margin_value
                worst_fp32 = {"position": pos, "competitor_token_id": competitor, "kind": "selected"}
        if target in selected_set:
            fixed_id, fixed_logit = fixed[pos]
            margin_value = float(target_logit - np.float32(fixed_logit))
            if margin_value < minimum_fp32_margin:
                minimum_fp32_margin = margin_value
                worst_fp32 = {"position": pos, "competitor_token_id": fixed_id, "kind": "fixed_max"}
    if minimum_fp32_margin < margin - certificate_tolerance:
        raise HoldError(
            f"FP32 hook replay misses registered margin: {minimum_fp32_margin}"
        )

    primal = 0.5 * float(np.sum(coefficients * coefficients))
    dual_objective = float(np.dot(dual, [item.rhs for item in active])) - primal
    row_energy = np.sum(residual_rows * residual_rows, axis=1)
    sv_residual = np.linalg.svd(residual_rows, compute_uv=False)
    energy = sv_residual * sv_residual
    effective_rank = (
        float(energy.sum() ** 2 / np.sum(energy * energy)) if np.any(energy) else 0.0
    )
    rank_95 = (
        int(np.searchsorted(np.cumsum(energy) / energy.sum(), 0.95) + 1)
        if np.any(energy)
        else 0
    )
    worst, _, _ = separate(coefficients)
    largest_required = max(
        (constraint for pos in range(targets.size) for constraint in candidate_constraints(pos)),
        key=lambda item: item.rhs,
    )
    return {
        "selected_token_ids": np.asarray(selected_ids, dtype=np.int64),
        "residual_rows": np.asarray(residual_rows, dtype=np.float64),
        "receipt": {
            "position_count": int(targets.size),
            "deficient_position_count": deficient_position_count,
            "selected_row_count": len(selected_ids),
            "hidden_span_rank": rank,
            "free_variable_count": len(selected_ids) * rank,
            "hidden_span_singular_values": singular.tolist(),
            "registered_constraint_count": sum(
                len(candidate_constraints(pos)) for pos in range(targets.size)
            ),
            "active_constraint_count": len(active),
            "outer_solve_count": outer_solve_count,
            "optimizer_iteration_count": optimizer_iterations,
            "max_fp64_violation": float(worst),
            "minimum_primal_slack": float(-worst),
            "minimum_fp32_hook_margin": float(minimum_fp32_margin),
            "certificate_tolerance": certificate_tolerance,
            "full_vocab_partition_certificate": {
                "top_k": int(top_values.shape[1]),
                "unique_route_target_count": int(route_ids.size),
                "k_exceeds_unique_targets": bool(top_values.shape[1] > route_ids.size),
                "selected_rows_are_route_targets": selected_set.issubset(set(route_ids.tolist())),
                "target_excluded_from_competitors": True,
                "worst_fp32_constraint": worst_fp32,
            },
            "objective_half_frobenius_squared": primal,
            "dual_objective": dual_objective,
            "duality_gap": primal - dual_objective,
            "residual_rank": int(np.sum(sv_residual > (sv_residual[0] * 1e-12))) if sv_residual.size else 0,
            "residual_effective_rank": effective_rank,
            "residual_rank_95_percent_energy": rank_95,
            "residual_first_direction_energy_share": (
                float(energy[0] / energy.sum()) if np.any(energy) else 0.0
            ),
            "largest_row_energy_share": float(row_energy.max() / row_energy.sum()) if row_energy.sum() else 0.0,
            "largest_required_margin_constraint": {
                "position": largest_required.position,
                "kind": largest_required.kind,
                "competitor_token_id": largest_required.competitor_token_id,
                "rhs": largest_required.rhs,
            },
        },
    }

SOURCE_PATHS = ("probes/output_qp.py", "src/artifacts/git_identity.py", "src/artifacts/json_values.py", "src/artifacts/__init__.py")


def qualify(capture: Path, output: Path, *, source_receipt: Path | None = None) -> None:
    """Qualify explicit arrays under current clean code; old receipts cannot resume."""
    import numpy as np
    from src.artifacts.git_identity import capture_source_identity, verify_source_identity, SourceIdentityError
    from src.artifacts import publish_json_exclusive

    identity = capture_source_identity(SOURCE_PATHS)
    before = capture.read_bytes()
    input_hash = hashlib.sha256(before).hexdigest()
    if source_receipt is not None:
        previous = json.loads(source_receipt.read_text())
        if previous.get("schema") != "output_qp.qualified_arrays.v1":
            raise SourceIdentityError("historical/unsupported for continuation: legacy QP receipt")
        verify_source_identity(previous.get("source_identity", {}), required_paths=SOURCE_PATHS)
        if previous.get("input_sha256") != input_hash:
            raise SourceIdentityError("historical/unsupported for continuation: capture bytes changed")
    if output.exists():
        raise FileExistsError(output)
    with np.load(capture, allow_pickle=False) as arrays:
        required = {"hidden_states", "target_ids", "route_token_ids", "base_route_logits", "top_ids", "top_logits"}
        if set(arrays.files) != required or any(not np.isfinite(arrays[name]).all() for name in required):
            raise ValueError("capture needs exactly six finite numeric arrays")
        result = solve_minimum_frobenius(**{name: arrays[name] for name in required})
    if capture.read_bytes() != before:
        raise ValueError("capture changed while solving")
    verify_source_identity(identity, required_paths=SOURCE_PATHS)
    payload = {
        "schema": "output_qp.qualified_arrays.v1", "source_identity": identity,
        "input_sha256": input_hash, "input_path": str(capture.resolve()),
        "selected_token_ids": result["selected_token_ids"].tolist(),
        "residual_rows": result["residual_rows"].tolist(), "certificate": result["receipt"],
        "scope": "supplied-state FP32 margin certificate; no model or natural rollout",
    }
    publish_json_exclusive(output, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path, help="Explicit six-array NPZ; no pickle")
    parser.add_argument("--output", required=True, type=Path, help="Fresh JSON destination")
    parser.add_argument("--source-receipt", type=Path, help="Reuse only an exact current qualification; legacy is rejected")
    args = parser.parse_args()
    qualify(args.capture, args.output, source_receipt=args.source_receipt)


if __name__ == "__main__":
    main()
