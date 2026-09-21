"""Versioned CPU-only correction for the Lane A control candidate identity error."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-21-history-source-sign"
)
LANE_B_ROOT = ROOT.parent / "2026-09-21-native-row-choice"
BOUNDARY = "tied-14038-control-before-row7"
EXCLUDED_ID = "A-actual-greedy"
EXCLUDED_SHA = "0739bb69023add13cbffe7cecf0656bf6980c9bf4f9c811c2e7d3e5398f2a442"


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def binding(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def write_once(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def logsumexp(rows: list[dict[str, Any]]) -> float:
    values = [float(row["logprob_sum"]) for row in rows]
    peak = max(values)
    return peak + math.log(sum(math.exp(value - peak) for value in values))


def correct(source: dict[str, Any], source_path: Path, out: Path) -> dict[str, Any]:
    plan_path = ROOT / "selection" / "shared-admission.json"
    plan = json.loads(plan_path.read_text())
    control = next(item for item in plan["lane_a"]["control_boundaries"] if item["id"] == BOUNDARY)
    frozen = next(item for item in control["candidate_sets"]["A"] if item["id"] == EXCLUDED_ID)
    if frozen["token_sha256"] != EXCLUDED_SHA:
        raise ValueError("excluded token identity changed")
    if control["designated_A"] == control["native_next_owner"]:
        raise ValueError("frozen control no longer reproduces the A/N identity conflict")
    if frozen["owner"] != control["designated_A"]:
        raise ValueError("frozen mislabeled row no longer carries the designated-A label")

    controls = [cell for cell in source["cells"] if cell["boundary_id"] == BOUNDARY]
    failures = [cell for cell in source["cells"] if cell["lane"] == "failure"]
    if len(controls) != 3 or {cell["condition"] for cell in controls} != {"native", "cut_A", "cut_C"}:
        raise ValueError("control trio changed")
    corrected = []
    for cell in controls:
        original_a = cell["rows"]["A"]
        excluded = [row for row in original_a if row["id"] == EXCLUDED_ID]
        retained = [row for row in original_a if row["id"] != EXCLUDED_ID]
        if len(excluded) != 1 or excluded[0]["token_sha256"] != EXCLUDED_SHA or len(retained) != 2:
            raise ValueError(f"symmetric control exclusion failed: {cell['condition']}")
        masses = dict(cell["finite_set_log_masses"])
        masses["A"] = logsumexp(retained)
        relative = {
            "A_minus_N": masses["A"] - masses["N"],
            "C_minus_N": masses["C"] - masses["N"],
            "A_minus_C": masses["A"] - masses["C"],
        }
        first = cell["release"]["first_complete_row"]
        exact_native_n = cell["condition"] == "native" and first and first["values"] == frozen["values"]
        corrected.append({
            "condition": cell["condition"],
            "support": {"A_original": 3, "A_corrected": 2, "C": len(cell["rows"]["C"]), "N": len(cell["rows"]["N"])},
            "individual_rows": {"A_retained": retained, "A_excluded_invalid_audit_only": excluded, "C": cell["rows"]["C"], "N": cell["rows"]["N"]},
            "finite_set_log_masses": masses, "finite_set_contrasts": relative,
            "first_fork": cell["first_candidate_fork"],
            "first_release": {
                "values": first["values"] if first else None,
                "identity": "N:annotation:-125" if exact_native_n else "HOLD",
                "identity_basis": "independently frozen exact native next row" if exact_native_n else "no independent owner identity; geometry alone is a proxy",
            },
            "release_binding": cell["release_binding"],
        })
    by_condition = {cell["condition"]: cell for cell in corrected}
    effects = []
    for condition in ("cut_A", "cut_C"):
        current, native = by_condition[condition], by_condition["native"]
        effects.append({
            "condition": condition,
            "label": "POSTHOC_CORRECTED_SENSITIVITY",
            "minus_native_finite_set_log_masses": {
                name: current["finite_set_log_masses"][name] - native["finite_set_log_masses"][name]
                for name in ("A", "C", "N")
            },
            "minus_native_contrasts": {
                name: current["finite_set_contrasts"][name] - native["finite_set_contrasts"][name]
                for name in ("A_minus_N", "C_minus_N", "A_minus_C")
            },
            "first_fork_margin_delta": current["first_fork"]["top2"]["margin"] - native["first_fork"]["top2"]["margin"],
        })

    result = {
        "schema": "history_source_sign.control_sensitivity.v1", "status": "candidate",
        "label": "POSTHOC_CORRECTED_SENSITIVITY", "protocol_debt": True,
        "correction_rule": {
            "boundary": BOUNDARY, "candidate_id": EXCLUDED_ID, "token_sha256": EXCLUDED_SHA,
            "action": "exclude this exact mislabeled row from control A in native, cut_A, and cut_C; add or substitute nothing",
            "frozen_control_A_and_dependent_effects": "INVALID_identity_incoherence_audit_only",
        },
        "bindings": {
            "immutable_admission": binding(plan_path), "as_run_reduction": binding(source_path),
            "reducer": binding(Path(__file__).resolve()),
            "lead_ruling": binding(ROOT / "coordination" / "control-identity-ruling-03.txt"),
            "lane_b_reduction_unchanged": binding(LANE_B_ROOT / "reduction-integrated.json"),
            "release_files": [cell["release_binding"] for cell in source["cells"]],
        },
        "control_cells": corrected, "corrected_within_control_effects": effects,
        "regression": {
            "identity_conflict_reproduced": True, "symmetric_exclusion_conditions": [cell["condition"] for cell in corrected],
            "failure_cells_unchanged_sha256": digest(failures),
            "lane_b_values_unchanged_sha256": binding(LANE_B_ROOT / "reduction-integrated.json")["sha256"],
        },
        "claim_limit": "Finite tested-set sensitivity after an evaluation-only identity correction; failure/control comparisons inherit this label, and release geometry is not confirmed physical identity.",
    }
    write_once(out, result)
    return result


def selfcheck(source_path: Path) -> None:
    source = json.loads(source_path.read_text())
    plan = json.loads((ROOT / "selection" / "shared-admission.json").read_text())
    control = plan["lane_a"]["control_boundaries"][0]
    bad = next(row for row in control["candidate_sets"]["A"] if row["id"] == EXCLUDED_ID)
    assert control["designated_A"] == "annotation:-141"
    assert control["native_next_owner"] == "annotation:-125"
    assert bad["token_sha256"] == EXCLUDED_SHA
    cells = [cell for cell in source["cells"] if cell["boundary_id"] == BOUNDARY]
    assert len(cells) == 3 and all(sum(row["id"] == EXCLUDED_ID for row in cell["rows"]["A"]) == 1 for cell in cells)
    print("PASS frozen identity conflict and symmetric three-cell exclusion preconditions")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "reduction-integrated.json")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck(args.source)
        return
    result = correct(json.loads(args.source.read_text()), args.source, args.out or ROOT / "control-sensitivity-v1.json")
    print(json.dumps({"status": result["status"], "label": result["label"], "effects": len(result["corrected_within_control_effects"])}, indent=2))


if __name__ == "__main__":
    main()
