"""CPU reducer for the frozen Lane A spatial-progress scores."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from probes.training_set_completion.spatial_progress_gate.runtime import (
    ADMISSION,
    ADMISSION_SHA256,
    ATOL,
    BUDGET,
    BUDGET_SHA256,
    ROOT,
    WALL_LIMIT,
    WALL_LIMIT_SHA256,
    _cells,
    _digest,
    _validate_scores,
)


def binding(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def write_new(path: Path, value: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return binding(path)


def logsumexp(values: list[float]) -> float | None:
    if not values:
        return None
    peak = max(values)
    return peak + math.log(sum(math.exp(value - peak) for value in values))


def _row_check(row: dict[str, Any]) -> None:
    tokens = row.get("token_ids")
    logprobs = row.get("token_logprobs")
    if not isinstance(tokens, list) or not isinstance(logprobs, list) or len(tokens) != len(logprobs):
        raise ValueError(f"candidate token/logprob length mismatch: {row.get('candidate_id')}")
    if tokens[0] != 151646 or tokens[-1] != 151649:
        raise ValueError(f"candidate opener/terminator omitted: {row.get('candidate_id')}")
    if row.get("token_sha256") != _digest(tokens):
        raise ValueError(f"candidate token identity changed: {row.get('candidate_id')}")
    if any(not math.isfinite(float(value)) for value in logprobs):
        raise ValueError(f"candidate has nonfinite logprob: {row.get('candidate_id')}")
    total = sum(float(value) for value in logprobs)
    if abs(total - float(row.get("row_sum_logprob"))) > 1e-5:
        raise ValueError(f"candidate row probability accounting changed: {row.get('candidate_id')}")


def _expected_ids(boundary: dict[str, Any]) -> set[str]:
    return {str(candidate["id"]) for name in ("A", "N") for candidate in boundary["candidate_sets"][name]}


def _validate_cell(result: dict[str, Any], cell: dict[str, Any]) -> None:
    if result.get("status") != "candidate_complete":
        raise ValueError(f"cell is not complete: {cell['id']}")
    if result.get("cell_id") != cell["id"] or result.get("condition") != cell["condition"]:
        raise ValueError(f"cell identity changed: {cell['id']}")
    _validate_scores(result, _expected_ids(cell))
    for row in result["rows"].values():
        _row_check(row)
    falsification = result.get("probability_accounting_falsification", {})
    if not falsification.get("passed") or not all(item.get("rejected") for item in falsification.get("checks", [])):
        raise ValueError(f"probability-accounting falsification failed: {cell['id']}")
    prefix = result.get("history_prefix", {}).get("condition", {})
    if not prefix.get("changed_prefix_check", {}).get("passed"):
        raise ValueError(f"changed prefix check failed: {cell['id']}")
    if cell["condition"] == "native":
        parity = result.get("native_trace_parity", {})
        if not parity.get("passed") or float(parity.get("max_full_vocab_logprob_abs_error", ATOL + 1)) > ATOL:
            raise ValueError(f"native source parity failed: {cell['id']}")


def _artifact_bytes(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def _saved_wall_elapsed(root: Path) -> tuple[float | None, dict[str, Any] | None]:
    local_start = root / "coordination" / "first-model-launch.json"
    shared_start = root.parent / "2026-09-21-visual-instance-binding" / "wall-start.json"
    # Recovery has its own dispatch-bound continuation wall.  A predecessor
    # lane marker is only a legacy fallback for the old package root.
    marker = local_start if local_start.is_file() else shared_start
    if not marker.is_file():
        return None, None
    payload = json.loads(marker.read_text())
    started = float(payload.get("started_epoch", payload.get("started_unix")))
    receipts = sorted(root.glob("qualification*/receipt.json")) + sorted(root.glob("cells/*/receipt.json"))
    if not receipts:
        return 0.0, {"start": binding(marker), "terminal_receipts": []}
    terminal = max(receipts, key=lambda path: path.stat().st_mtime)
    return max(0.0, terminal.stat().st_mtime - started), {
        "start": binding(marker),
        "terminal_receipt": binding(terminal),
        "terminal_time_basis": "retained receipt filesystem mtime",
    }


def _attempts(root: Path, cell: dict[str, Any]) -> list[Path]:
    base = root / "cells" / cell["id"]
    return [path for path in [base, *sorted(root.glob(f"cells/{cell['id']}--retry-*"))] if path.is_dir()]


def _load_cell(root: Path, cell: dict[str, Any]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    failed: list[dict[str, Any]] = []
    chosen: dict[str, Any] | None = None
    for directory in _attempts(root, cell):
        receipt_path = directory / "receipt.json"
        score_path = directory / "scores.json"
        if not receipt_path.is_file():
            failed.append({"path": str(directory), "status": "missing_receipt"})
            continue
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("status") != "candidate_complete" or not score_path.is_file():
            failed.append({"path": binding(receipt_path), "status": receipt.get("status"), "error": receipt.get("error"), "counters": receipt.get("counters", {})})
            continue
        result = json.loads(score_path.read_text())
        try:
            _validate_cell(result, cell)
        except (AssertionError, KeyError, TypeError, ValueError) as error:
            failed.append({"path": binding(receipt_path), "status": "technical_invalid", "error": str(error), "counters": receipt.get("counters", {})})
            continue
        chosen = {
            "directory": directory, "result": result, "receipt": receipt,
            "receipt_binding": binding(receipt_path), "scores_binding": binding(score_path),
        }
    return chosen, failed


def _qualification(root: Path, boundary: dict[str, Any]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    failed: list[dict[str, Any]] = []
    chosen = None
    for name in ("qualification-01", "qualification-repair-01"):
        directory = root / name
        receipt_path = directory / "receipt.json"
        result_path = directory / "result.json"
        if not directory.exists():
            continue
        if not receipt_path.is_file():
            failed.append({"path": str(directory), "status": "missing_receipt"})
            continue
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("status") != "candidate_complete" or not result_path.is_file():
            failed.append({"path": binding(receipt_path), "status": receipt.get("status"), "error": receipt.get("error"), "counters": receipt.get("counters", {})})
            continue
        result = json.loads(result_path.read_text())
        try:
            if result.get("status") != "candidate_complete" or not result.get("passed"):
                raise ValueError("qualification did not pass")
            for condition in ("native", "x1_before_N786"):
                value = result["conditions"][condition]
                _validate_cell(value, {**boundary, "id": value["cell_id"], "condition": condition})
        except (AssertionError, KeyError, TypeError, ValueError) as error:
            failed.append({"path": binding(receipt_path), "status": "technical_invalid", "error": str(error), "counters": receipt.get("counters", {})})
            continue
        chosen = {"directory": directory, "result": result, "receipt": receipt, "receipt_binding": binding(receipt_path), "result_binding": binding(result_path)}
    return chosen, failed


def _condition_rows(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return result["rows"]


def _effects(boundary: dict[str, Any], by_condition: dict[str, dict[str, Any]]) -> dict[str, Any]:
    native = by_condition.get("native")
    before = by_condition.get("x1_before_N786")
    if native is None or before is None:
        return {"status": "HOLD_missing_native_or_before"}
    shams = {name: by_condition.get(name) for name in ("sham_y2_minus4", "sham_y2_plus4")}
    candidates = {}
    for name in ("A", "N"):
        for candidate in boundary["candidate_sets"][name]:
            ident = str(candidate["id"])
            nrow = native["rows"][ident]
            brow = before["rows"][ident]
            item: dict[str, Any] = {
                "candidate_id": ident, "set": name, "owner": candidate["owner"], "values": candidate["values"],
                "native_logprob": nrow["row_sum_logprob"], "before_logprob": brow["row_sum_logprob"],
                "x1_before_minus_native": float(brow["row_sum_logprob"] - nrow["row_sum_logprob"]),
                "native_xy1_logprob": nrow["conditional_xy1"]["logprob_sum"],
                "before_xy1_logprob": brow["conditional_xy1"]["logprob_sum"],
                "x1_before_minus_native_xy1": float(brow["conditional_xy1"]["logprob_sum"] - nrow["conditional_xy1"]["logprob_sum"]),
                "shams": {},
            }
            for sham_name, sham in shams.items():
                if sham is not None:
                    srow = sham["rows"][ident]
                    item["shams"][sham_name] = {
                        "logprob": srow["row_sum_logprob"],
                        "minus_native": float(srow["row_sum_logprob"] - nrow["row_sum_logprob"]),
                        "ordering_minus_sham": float((brow["row_sum_logprob"] - nrow["row_sum_logprob"]) - (srow["row_sum_logprob"] - nrow["row_sum_logprob"])),
                        "xy1_minus_native": float(srow["conditional_xy1"]["logprob_sum"] - nrow["conditional_xy1"]["logprob_sum"]),
                    }
            candidates[ident] = item
    fixed = str(boundary["fixed_crossed_N"])
    fixed_effect = next((item for item in candidates.values() if str(item["owner"]) == fixed), None)
    finite = {}
    for name in ("A", "N"):
        native_mass = native["candidate_sets"][name]["row_logsumexp"]
        before_mass = before["candidate_sets"][name]["row_logsumexp"]
        finite[name] = {"native": native_mass, "before": before_mass, "before_minus_native": float(before_mass - native_mass)}
    return {
        "status": "candidate", "boundary_id": boundary["id"], "source_boundary_id": boundary["source_boundary_id"],
        "kind": boundary["kind"], "model": boundary["model"], "image_id": boundary["image_id"],
        "fixed_crossed_N": fixed, "finite_set_masses": finite,
        "fixed_crossed_N_effect": fixed_effect, "candidates": candidates,
        "N_before_minus_native": None if fixed_effect is None else fixed_effect["x1_before_minus_native"],
        "N_before_minus_native_xy1": None if fixed_effect is None else fixed_effect["x1_before_minus_native_xy1"],
        "native_alias": {"condition_id": "x1_after_N786", "alias_of": "native", "executed_once": True},
        "sham_conditions": [name for name, value in shams.items() if value is not None],
        "prediction_direction": "positive fixed-N before-minus-after effect is H1 direction",
        "candidate_support_only": True,
    }


def reduce(root: Path = ROOT, out: Path | None = None) -> dict[str, Any]:
    admission_binding = binding(ADMISSION)
    budget_binding = binding(BUDGET)
    wall_binding = binding(WALL_LIMIT)
    if admission_binding["sha256"] != ADMISSION_SHA256 or budget_binding["sha256"] != BUDGET_SHA256 or wall_binding["sha256"] != WALL_LIMIT_SHA256:
        raise ValueError("frozen selection binding changed")
    admission = json.loads(ADMISSION.read_text())
    cells = _cells(admission)
    qualification_boundary = next(item for item in admission["lane_a"]["boundaries"] if item["id"] == "tied-14038-failure-before-row8")
    qualification, qualification_failures = _qualification(root, qualification_boundary)
    reduced_cells: list[dict[str, Any]] = []
    missing: list[str] = []
    failed_attempts = list(qualification_failures)
    by_boundary: dict[str, dict[str, dict[str, Any]]] = {}
    counters = {"model_forwards": 0, "vision_forwards": 0, "gpu_seconds": 0.0, "artifact_bytes": 0, "candidate_rows": 0}
    for failure in qualification_failures:
        for key in ("model_forwards", "vision_forwards", "candidate_rows"):
            counters[key] += int(failure.get("counters", {}).get(key, 0))
        counters["gpu_seconds"] += float(failure.get("counters", {}).get("gpu_seconds", 0.0))
        failure_path = failure.get("path", "")
        directory = Path(failure_path.get("path", "")) if isinstance(failure_path, dict) else Path(failure_path)
        if directory.is_file():
            directory = directory.parent
        if directory.is_dir():
            counters["artifact_bytes"] += _artifact_bytes(directory)
    for cell in cells:
        chosen, failures = _load_cell(root, cell)
        failed_attempts.extend(failures)
        for failure in failures:
            for key in ("model_forwards", "vision_forwards", "candidate_rows"):
                counters[key] += int(failure.get("counters", {}).get(key, 0))
            counters["gpu_seconds"] += float(failure.get("counters", {}).get("gpu_seconds", 0.0))
            failure_path = failure.get("path", "")
            directory = Path(failure_path.get("path", "")) if isinstance(failure_path, dict) else Path(failure_path)
            if directory.is_file():
                directory = directory.parent
            if directory.is_dir():
                counters["artifact_bytes"] += _artifact_bytes(directory)
        if chosen is None:
            missing.append(cell["id"])
            continue
        result, receipt, directory = chosen["result"], chosen["receipt"], chosen["directory"]
        counters["model_forwards"] += int(receipt.get("counters", {}).get("model_forwards", 0))
        counters["vision_forwards"] += int(receipt.get("counters", {}).get("vision_forwards", 0))
        counters["candidate_rows"] += int(receipt.get("counters", {}).get("candidate_rows", 0))
        counters["gpu_seconds"] += float(receipt.get("counters", {}).get("gpu_seconds", 0.0))
        counters["artifact_bytes"] += _artifact_bytes(directory)
        reduced_cells.append({
            "id": cell["id"], "boundary_id": cell["id"].split("--", 1)[0], "condition": cell["condition"],
            "model": cell["model"], "image_id": cell["image_id"], "scores": chosen["scores_binding"], "receipt": chosen["receipt_binding"],
            "candidate_sets": result["candidate_sets"], "rows": result["rows"], "history_prefix": result["history_prefix"],
        })
        by_boundary.setdefault(cell["id"].split("--", 1)[0], {})[cell["condition"]] = result
    if qualification is not None:
        qreceipt = qualification["receipt"]
        qdir = qualification["directory"]
        counters["model_forwards"] += int(qreceipt.get("counters", {}).get("model_forwards", 0))
        counters["vision_forwards"] += int(qreceipt.get("counters", {}).get("vision_forwards", 0))
        counters["candidate_rows"] += int(qreceipt.get("counters", {}).get("candidate_rows", 0))
        counters["gpu_seconds"] += float(qreceipt.get("counters", {}).get("gpu_seconds", 0.0))
        counters["artifact_bytes"] += _artifact_bytes(qdir)
    effects = []
    for boundary in admission["lane_a"]["boundaries"]:
        effects.append(_effects(boundary, by_boundary.get(boundary["id"], {})))
    wall_elapsed, wall_elapsed_basis = _saved_wall_elapsed(root)
    result = {
        "schema": "spatial_progress_gate.reduction.v2", "status": "candidate" if qualification is not None and not missing else "technical_HOLD",
        "qualification": None if qualification is None else {"campaign": qualification["directory"].name, "result": qualification["result_binding"], "receipt": qualification["receipt_binding"]},
        "qualification_HOLDs": qualification_failures, "cells": reduced_cells, "missing_cells": missing,
        "effects": effects, "failed_attempts": failed_attempts,
        "costs": {**counters, "wall_elapsed_seconds": wall_elapsed, "wall_elapsed_basis": wall_elapsed_basis, "hard_wall_limit_seconds": 7200},
        "optional_release": {"status": "not_run", "reason": "requires lead outcome-selected direction ruling"},
        "bindings": {"admission": admission_binding, "budget_estimate": budget_binding, "wall_limit": wall_binding},
        "limits": admission["limits"], "tolerances": admission["tolerances"],
        "claim_limits": admission["claim_limits"],
    }
    write_new(out or root / "reduction.json", result)
    return result
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = reduce(args.root, args.out)
    print(json.dumps({"status": result["status"], "cells": len(result["cells"]), "missing": result["missing_cells"], "costs": result["costs"]}, indent=2))


if __name__ == "__main__":
    main()
