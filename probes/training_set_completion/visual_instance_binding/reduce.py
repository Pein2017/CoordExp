"""CPU reducer for the frozen visual-binding response matrices."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-21-visual-instance-binding")
ATOL = 1e-5


def binding(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def logsumexp(values: list[float]) -> float | None:
    if not values:
        return None
    peak = max(values)
    return peak + math.log(sum(math.exp(value - peak) for value in values))


def check_condition(result: dict[str, Any]) -> None:
    rows = result["rows"]
    for group in result["candidate_sets"].values():
        ids = [str(item) for item in group["row_ids"]]
        observed = logsumexp([float(rows[item]["row_sum_logprob"]) for item in ids])
        expected = group["row_logsumexp"]
        if (observed is None) != (expected is None) or (observed is not None and abs(observed - float(expected)) > ATOL):
            raise ValueError("candidate-set logsumexp changed")
    for row in rows.values():
        tokens, logps = row["token_ids"], row["token_logprobs"]
        if tokens[0] != 151646 or tokens[-1] != 151649 or len(tokens) != len(logps):
            raise ValueError("complete-row accounting lost entry or terminator")
        if abs(sum(float(value) for value in logps) - float(row["row_sum_logprob"])) > ATOL:
            raise ValueError("row logprob sum changed")


def reduce(root: Path, out: Path) -> dict[str, Any]:
    states: list[dict[str, Any]] = []
    totals = {"model_forwards": 0, "vision_forwards": 0, "gpu_seconds": 0.0, "retained_tensor_bytes": 0}
    failures: list[dict[str, Any]] = []
    primary_root = root / "qualification-01"
    for directory in sorted(primary_root.glob("*")):
        scores_path, receipt_path = directory / "scores.json", directory / "receipt.json"
        if not scores_path.is_file() or not receipt_path.is_file():
            failures.append({"state_id": directory.name, "status": "missing_receipt_or_scores"})
            continue
        receipt, scores = json.loads(receipt_path.read_text()), json.loads(scores_path.read_text())
        if receipt.get("scores") != binding(scores_path) or receipt.get("status") != "candidate_complete":
            failures.append({"state_id": directory.name, "status": receipt.get("status"), "error": receipt.get("error")})
            continue
        check_condition(scores["conditions"]["clean"])
        for condition in ("ablate_A", "ablate_N", "ablate_unrelated"):
            check_condition(scores["conditions"][condition])
        if not scores["source_native_replay_parity"]["passed"] or not scores["clean_compositor_full_logit_parity"]["passed"] or not scores["actual_condition_difference"]["passed"]:
            raise ValueError(f"technical parity failed: {directory.name}")
        for key in totals:
            totals[key] += receipt.get("counters", {}).get(key, 0)
        response = scores["response_matrix"]
        rows = []
        for region, candidates in response.items():
            for candidate_id, effect in candidates.items():
                rows.append({"region": region, "candidate_id": candidate_id, "effect_logprob": float(effect), "candidate_set": "A" if candidate_id in scores["candidate_sets"]["A"]["row_ids"] else "N"})
        states.append({"state_id": scores["state_id"], "model": scores["model"], "image_id": scores["image_id"], "stratum": "control" if "control" in scores["state_id"] else "target_first_revisit", "response_matrix": rows, "selectivity": {"A_removal_A_effects": [row["effect_logprob"] for row in rows if row["region"] == "A" and row["candidate_set"] == "A"], "A_removal_N_effects": [row["effect_logprob"] for row in rows if row["region"] == "A" and row["candidate_set"] == "N"], "N_removal_A_effects": [row["effect_logprob"] for row in rows if row["region"] == "N" and row["candidate_set"] == "A"], "N_removal_N_effects": [row["effect_logprob"] for row in rows if row["region"] == "N" and row["candidate_set"] == "N"]}, "parity": {"native": scores["source_native_replay_parity"], "clean": scores["clean_compositor_full_logit_parity"]}, "scores": binding(scores_path), "receipt": binding(receipt_path)})
    if failures:
        status = "candidate_with_failures"
    else:
        status = "candidate"
    audit = []
    for receipt_path in sorted((root / "states").glob("*/receipt.json")):
        receipt = json.loads(receipt_path.read_text())
        audit.append({"state_id": receipt.get("state_id"), "status": receipt.get("status"), "receipt": binding(receipt_path), "counters": receipt.get("counters", {}), "disposition": "redundant_pre_ruling_execution_excluded_from_primary_evidence"})
    result = {"schema": "visual_instance_binding.reduction.v1", "status": status, "primary_campaign": "qualification-01", "states": states, "failure_or_control_strata": {"target_first_revisit": sum(item["stratum"] == "target_first_revisit" for item in states), "control": sum(item["stratum"] == "control" for item in states)}, "counters": totals, "failed_attempts": failures, "audit_only_redundant_execution": audit, "response_matrix_definition": "logP(candidate | region ablated) - logP(candidate | clean), complete row including entry and terminator", "interpretation": "Finite visual intervention candidate from qualification-01 only; redundant scientific-01 outputs are retained as protocol debt and excluded from denominators.", "admission": binding(root / "selection" / "shared-admission-binding.json")}
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def selfcheck() -> None:
    row = {"token_ids": [151646, 1, 151649], "token_logprobs": [-1.0, -2.0, -3.0], "row_sum_logprob": -6.0}
    state = {"rows": {"a": row}, "candidate_sets": {"A": {"row_ids": ["a"], "row_logsumexp": -6.0}, "N": {"row_ids": [], "row_logsumexp": None}}}
    check_condition(state)
    try:
        check_condition({**state, "rows": {"a": {**row, "token_ids": [1, 1, 151649]}}})
    except ValueError as error:
        assert "entry" in str(error)
    else:
        raise AssertionError("dropped entry was not rejected")
    print("visual_instance_binding reducer selfcheck: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    result = reduce(args.root, args.out or args.root / "reduction.json")
    print(json.dumps({"status": result["status"], "states": len(result["states"]), "failed": len(result["failed_attempts"])}, indent=2))


if __name__ == "__main__":
    main()
