"""Deterministic CPU reducer for the frozen Lane B native row scores."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-21-native-row-choice"
)
ROW_OPEN, ROW_END = 151646, 151649
STATE_DIRS = (
    "qualification/tied-14038-first-revisit",
    "states-repair-1/tied-885-first-revisit",
    "states/untied-885-first-revisit",
)


def binding(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def write_once(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def logsumexp(values: list[float]) -> float | None:
    if not values:
        return None
    peak = max(values)
    return peak + math.log(sum(math.exp(value - peak) for value in values))


def check_row(row: dict[str, Any]) -> float:
    tokens, logps = row["token_ids"], row["token_logprobs"]
    if len(tokens) != len(logps) or len(tokens) < 2:
        raise ValueError("row token/logprob length mismatch")
    if tokens[0] != ROW_OPEN or tokens[-1] != ROW_END:
        raise ValueError("row opener or terminator was dropped")
    total = sum(float(value) for value in logps)
    if not math.isfinite(total) or abs(total - float(row["row_sum_logprob"])) > 1e-5:
        raise ValueError("row probability accounting changed")
    return total


def reduce(root: Path, out: Path) -> dict[str, Any]:
    states = []
    totals = {"model_forwards": 0, "gpu_seconds": 0.0, "retained_tensor_bytes": 0}
    receipts = []
    for relative in STATE_DIRS:
        directory = root / relative
        receipt_path, scores_path = directory / "receipt.json", directory / "scores.json"
        receipt, scores = json.loads(receipt_path.read_text()), json.loads(scores_path.read_text())
        if receipt.get("status") != "candidate_complete" or scores.get("status") != "candidate_complete":
            raise ValueError(f"incomplete Lane B state: {scores.get('state_id')}")
        if receipt.get("scores") != binding(scores_path):
            raise ValueError(f"score binding changed: {scores.get('state_id')}")
        falsification_path = directory / "probability-accounting-falsification.json"
        falsification = json.loads(falsification_path.read_text())
        if not falsification.get("passed") or not all(item.get("rejected") for item in falsification["checks"]):
            raise ValueError(f"probability falsification failed: {scores.get('state_id')}")
        row_scores = {name: check_row(row) for name, row in scores["rows"].items()}
        for name in ("A", "C", "N"):
            expected = scores["candidate_sets"][name]["row_logsumexp"]
            observed = logsumexp([row_scores[row_id] for row_id in scores["candidate_sets"][name]["row_ids"]])
            if (expected is None) != (observed is None) or (
                expected is not None and abs(float(expected) - float(observed)) > 1e-6
            ):
                raise ValueError(f"candidate-set mass changed: {scores.get('state_id')}:{name}")
        actual = scores["actual_greedy_id"]
        best_n = max(scores["candidate_sets"]["N"]["row_ids"], key=row_scores.__getitem__)
        parity = scores["actual_row_trace_parity"]
        if not parity.get("passed") or not parity.get("entry_included") or not parity.get("terminator_included"):
            raise ValueError(f"source trace parity failed: {scores.get('state_id')}")
        states.append({
            "state_id": scores["state_id"], "model": scores["model"], "image_id": scores["image_id"],
            "common_description_tokens": scores["common_description_tokens"],
            "actual_greedy_id": actual, "actual_greedy_logprob": row_scores[actual],
            "best_N_id": best_n, "best_N_owner": scores["rows"][best_n]["owner"],
            "best_N_logprob": row_scores[best_n], "best_N_minus_actual": row_scores[best_n] - row_scores[actual],
            "positive_local_witness": row_scores[best_n] > row_scores[actual],
            "finite_set_log_masses": {
                name: scores["candidate_sets"][name]["row_logsumexp"] for name in ("A", "C", "N")
            },
            "max_trace_top2_abs_error": max(item["top2_max_abs_error"] for item in parity["tokens"]),
            "scores": binding(scores_path), "falsification": binding(falsification_path),
        })
        for key in totals:
            totals[key] += receipt["counters"][key]
        receipts.append(binding(receipt_path))

    failed = []
    for receipt_path in sorted((root / "states").glob("*/receipt.json")):
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("status") == "candidate_complete":
            continue
        failed.append({"receipt": binding(receipt_path), "status": receipt.get("status"), "error": receipt.get("error")})
        for key in totals:
            totals[key] += receipt["counters"][key]
    result = {
        "schema": "native_row_choice.reduction.v1", "status": "candidate",
        "states": states, "positive_witness_count": sum(item["positive_local_witness"] for item in states),
        "tested_state_count": len(states), "optional_continuations_run": 0,
        "counters": totals, "receipts": receipts, "failed_attempts": failed,
        "interpretation": "Negative finite-candidate diagnostic; unresolved outside the frozen rows and not a global row-MAP or owner-mass claim.",
    }
    write_once(out, result)
    return result


def selfcheck() -> None:
    row = {"token_ids": [ROW_OPEN, ROW_END], "token_logprobs": [-1.0, -2.0], "row_sum_logprob": -3.0}
    assert check_row(row) == -3.0
    try:
        check_row({**row, "token_ids": [7, ROW_END]})
    except ValueError as exc:
        assert "opener or terminator" in str(exc)
    else:
        raise AssertionError("dropped opener was not rejected")
    print("PASS complete-row probability accounting falsification")


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
    print(json.dumps({"status": result["status"], "states": result["tested_state_count"], "positive": result["positive_witness_count"]}, indent=2))


if __name__ == "__main__":
    main()
