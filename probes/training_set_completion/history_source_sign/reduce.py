"""Deterministic CPU reducer for Lane A source-path cells."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from probes.training_set_completion.numerical_feedback.select import rows as parsed_rows


ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-21-history-source-sign"
)
PLAN = ROOT / "selection" / "shared-admission.json"
EOS, ROW_OPEN, ROW_END = 151645, 151646, 151649


def binding(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def write_once(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def logsumexp(values: list[float]) -> float:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("finite candidate log-mass inputs are required")
    peak = max(values)
    return peak + math.log(sum(math.exp(value - peak) for value in values))


def _cells(plan: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for key in ("failure_boundaries", "control_boundaries"):
        lane = "failure" if key == "failure_boundaries" else "control"
        for boundary in plan["lane_a"][key]:
            for condition in ("native", "cut_A", "cut_C"):
                item = dict(boundary)
                item.update(
                    lane=lane,
                    condition=condition,
                    admission_boundary_id=boundary["id"],
                    id=f"{boundary['id']}--{condition}",
                )
                out.append(item)
    return out


def _iou(left: list[int], right: list[int]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _classify_release(release: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    anchors = [
        (role, candidate["owner"], candidate["description_tokens"], candidate["values"])
        for role in ("A", "C", "N")
        for candidate in spec["candidate_sets"][role]
    ]
    classified = []
    counts = {name: 0 for name in (
        "A", "C", "N", "other_credible_owner", "UNKNOWN",
        "invalid_geometry", "malformed_output", "EOS", "administrative_cap",
    )}
    rows = parsed_rows(release["token_ids"])
    for row in rows:
        if not row["valid"]:
            label, owner, score = "invalid_geometry", None, None
        else:
            choices = [
                (_iou(row["values"], values), role, owner)
                for role, owner, description, values in anchors
                if row["description_tokens"] == description
            ]
            score, label, owner = max(choices, default=(0.0, "UNKNOWN", None))
            if score < 0.5:
                label, owner = "UNKNOWN", None
        counts[label] += 1
        classified.append({
            "row_index": row["index"], "values": row["values"], "label": label,
            "owner": owner, "best_anchor_iou": score,
        })
    complete_token_count = sum(int(row["end"]) - int(row["start"]) for row in rows)
    counts["malformed_output"] = int(
        complete_token_count != len(release["token_ids"])
        and release["stop"]["reason"] != "eos"
    )
    counts["EOS"] = int(release["stop"]["reason"] == "eos")
    counts["administrative_cap"] = int(release["stop"]["reason"] in {"rows", "source_cap"})
    return {
        "rule": "same-description best frozen A/C/N anchor IoU>=0.5; unmatched is UNKNOWN",
        "other_credible_owner_scope": "none admitted outside the frozen A/C/N anchors",
        "rows": classified,
        "counts": counts,
    }


def _runtime_paths(root: Path, cell_id: str) -> tuple[Path, Path] | None:
    roots = [root / "runtime" / cell_id]
    roots.extend(sorted(root.glob(f"runtime/{cell_id}--repair-*")))
    ready = [
        (directory / "release.json", directory / "receipt.json")
        for directory in roots
        if (directory / "release.json").is_file() and (directory / "receipt.json").is_file()
    ]
    return ready[-1] if ready else None


def _check_row(item: dict[str, Any]) -> dict[str, Any]:
    tokens = item.get("token_ids")
    logps = item.get("token_logprobs")
    if not isinstance(tokens, list) or not isinstance(logps, list) or len(tokens) != len(logps):
        raise ValueError(f"candidate token/logprob length mismatch: {item.get('id')}")
    if len(tokens) < 2 or int(tokens[0]) != ROW_OPEN or int(tokens[-1]) != ROW_END:
        raise ValueError(f"candidate opener/terminator omitted: {item.get('id')}")
    if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in logps):
        raise ValueError(f"candidate has nonfinite token logprob: {item.get('id')}")
    total = sum(float(value) for value in logps)
    if abs(total - float(item["logprob_sum"])) > 1e-6:
        raise ValueError(f"candidate logprob sum mismatch: {item.get('id')}")
    return {
        "id": item["id"],
        "owner": item["owner"],
        "source": item["source"],
        "values": item["values"],
        "token_count": len(tokens),
        "token_sha256": item["token_sha256"],
        "logprob_sum": total,
        "logprob_includes": item.get("logprob_includes"),
    }


def reduce(root: Path, out: Path) -> dict[str, Any]:
    plan_path = root / "selection" / "shared-admission.json"
    plan = json.loads(plan_path.read_text())
    expected = _cells(plan)
    cells: list[dict[str, Any]] = []
    missing: list[str] = []
    receipts: list[dict[str, Any]] = []
    total_forwards = total_free = total_bytes = 0
    total_elapsed = 0.0
    for spec in expected:
        paths = _runtime_paths(root, spec["id"])
        if paths is None:
            missing.append(spec["id"])
            continue
        release_path, receipt_path = paths
        release = json.loads(release_path.read_text())
        receipt = json.loads(receipt_path.read_text())
        if release.get("status") != "candidate_complete" or receipt.get("status") != "candidate_complete":
            missing.append(spec["id"])
            continue
        if release.get("cell_id") != spec["id"] or receipt.get("cell_id") != spec["id"]:
            raise ValueError(f"cell identity changed: {spec['id']}")
        masses = release["candidate_scores"]["finite_set_log_masses"]
        reconstructed: dict[str, float] = {}
        rows: dict[str, list[dict[str, Any]]] = {}
        for name, items in release["candidate_scores"]["sets"].items():
            checked = [_check_row(item) for item in items]
            rows[name] = checked
            reconstructed[name] = logsumexp([item["logprob_sum"] for item in checked])
            if abs(reconstructed[name] - float(masses[name])) > 1e-6:
                raise ValueError(f"finite-set mass mismatch in {spec['id']}:{name}")
        cells.append({
            "id": spec["id"],
            "lane": spec["lane"],
            "condition": spec["condition"],
            "boundary_id": spec["admission_boundary_id"],
            "source_boundary_id": spec["source_boundary_id"],
            "model": spec["model"],
            "image_id": spec["image_id"],
            "source_row": release["source_row"],
            "rows": rows,
            "finite_set_log_masses": reconstructed,
            "finite_set_contrasts": {
                "A_minus_N": reconstructed["A"] - reconstructed["N"],
                "C_minus_N": reconstructed["C"] - reconstructed["N"],
                "A_minus_C": reconstructed["A"] - reconstructed["C"],
            },
            "first_candidate_fork": release["candidate_scores"]["first_candidate_fork"],
            "release": release["release"],
            "release_classification": _classify_release(release["release"], spec),
            "release_binding": binding(release_path),
        })
        receipts.append(binding(receipt_path))
        total_forwards += int(receipt.get("model_forwards", 0))
        total_free += int(receipt.get("free_cells", 0))
        total_bytes += int(receipt.get("retained_tensor_bytes", 0))
        total_elapsed += float(receipt.get("elapsed_gpu_seconds", 0.0))

    by_boundary: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for cell in cells:
        by_boundary.setdefault((cell["lane"], cell["boundary_id"]), {})[cell["condition"]] = cell
    contrasts: list[dict[str, Any]] = []
    for (lane, boundary_id), conditions in sorted(by_boundary.items()):
        native = conditions.get("native")
        if native is None:
            continue
        for condition in ("cut_A", "cut_C"):
            current = conditions.get(condition)
            if current is None:
                continue
            contrasts.append({
                "lane": lane,
                "boundary_id": boundary_id,
                "condition": condition,
                "minus_native_finite_set_log_masses": {
                    name: current["finite_set_log_masses"][name] - native["finite_set_log_masses"][name]
                    for name in ("A", "C", "N")
                },
                "minus_native_contrasts": {
                    name: current["finite_set_contrasts"][name] - native["finite_set_contrasts"][name]
                    for name in ("A_minus_N", "C_minus_N", "A_minus_C")
                },
                "first_fork_margin_delta": (
                    current["first_candidate_fork"]["top2"]["margin"]
                    - native["first_candidate_fork"]["top2"]["margin"]
                ),
            })
    qualification_receipts = []
    for receipt_path in sorted((root / "qualification").glob("attempt-*/receipt.json")):
        receipt = json.loads(receipt_path.read_text())
        result_path = receipt_path.parent / "result.json"
        result = json.loads(result_path.read_text()) if result_path.is_file() else {}
        status = "candidate_complete" if result.get("status") == "candidate_complete" and result.get("passed") else "technical_HOLD"
        qualification_receipts.append({"path": binding(receipt_path), "status": status, "error": receipt.get("error")})
        total_forwards += int(receipt.get("model_forwards", 0))
        total_free += int(receipt.get("free_cells", 0))
        total_bytes += int(receipt.get("retained_tensor_bytes", 0))
        total_elapsed += float(receipt.get("elapsed_gpu_seconds", 0.0))
    failed_runtime_receipts = []
    for receipt_path in sorted((root / "runtime").glob("*/receipt.json")):
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("status") == "candidate_complete":
            continue
        failed_runtime_receipts.append({"path": binding(receipt_path), "status": receipt.get("status"), "error": receipt.get("error")})
        total_forwards += int(receipt.get("model_forwards", 0))
        total_bytes += int(receipt.get("retained_tensor_bytes", 0))
        total_elapsed += float(receipt.get("elapsed_gpu_seconds", 0.0))
    passing_qualification = next((
        item["path"] for item in reversed(qualification_receipts)
        if item["status"] == "candidate_complete"
    ), None)
    result = {
        "schema": "history_source_sign.reduction.v2",
        "status": "candidate" if not missing else "technical_HOLD",
        "protocol_compliance": "violated",
        "qualification_attempt_overrun": 1,
        "plan": binding(plan_path),
        "qualification": passing_qualification,
        "expected_cells": len(expected),
        "completed_cells": len(cells),
        "missing_cells": missing,
        "cells": cells,
        "within_boundary_contrasts": contrasts,
        "counters": {
            "model_forwards": total_forwards,
            "free_cells": total_free,
            "retained_tensor_bytes": total_bytes,
            "elapsed_gpu_seconds": total_elapsed,
            "allocated_gpu_hours_ceiling": 4,
            "model_forward_ceiling": 40000,
            "free_cell_ceiling": 144,
            "retained_byte_ceiling": 16 * 1024**3,
        },
        "interpretation_scope": [
            "candidate row log masses are finite tested sets, not complete owner probabilities",
            "native/cut effects are paired within boundary before any failure/control comparison",
            "release metrics are conditional route evidence and do not establish physical memory",
        ],
        "receipts": receipts,
        "qualification_receipts": qualification_receipts,
        "failed_runtime_receipts": failed_runtime_receipts,
    }
    write_once(out, result)
    return result


def selfcheck() -> None:
    plan = json.loads(PLAN.read_text())
    boundaries = {(cell["lane"], cell["admission_boundary_id"]) for cell in _cells(plan)}
    assert boundaries == {
        ("failure", "tied-14038-failure-before-row8"),
        ("control", "tied-14038-control-before-row7"),
    }
    complete = {
        "id": "x", "owner": "o", "source": "s", "values": [1, 2, 3, 4],
        "token_ids": [ROW_OPEN, 7, 151647, 151648, 151671, 151672, 151673, 151674, ROW_END],
        "token_logprobs": [-1.0] * 9, "token_sha256": "x", "logprob_sum": -9.0,
    }
    _check_row(complete)
    corrupt = dict(complete, token_ids=complete["token_ids"][1:], token_logprobs=[-1.0] * 8, logprob_sum=-8.0)
    try:
        _check_row(corrupt)
    except ValueError as exc:
        assert "opener/terminator omitted" in str(exc)
    else:
        raise AssertionError("dropped opener was not rejected")
    assert _iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    print("PASS boundary separation, complete-row accounting, and owner-anchor matching")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    out = args.out or args.root / "reduction.json"
    result = reduce(args.root, out)
    print(json.dumps({"status": result["status"], "completed_cells": result["completed_cells"], "missing": result["missing_cells"]}, indent=2))


if __name__ == "__main__":
    main()
