"""Focused independent CPU verifier for the v4 first-divergence correction."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-21-history-source-sign"
)
ROW_OPEN, EOS = 151646, 151645
COORD_MIN, COORD_MAX = 151670, 152669


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def verify(root: Path) -> dict[str, Any]:
    source_path = root / "first-divergence-v1.json"
    value = json.loads(source_path.read_text())
    if value.get("status") != "candidate" or value["denominators"] != {
        "failure_pairs": 12, "corrected_control_pairs": 8,
        "total_pairs": 20, "condition_level_diagnostics": 60,
    }:
        raise ValueError("pair denominators changed")
    for item in value["bindings"].values():
        entries = item if isinstance(item, list) else [item]
        for entry in entries:
            path = Path(entry["path"])
            if sha(path) != entry["sha256"] or path.stat().st_size != entry["size_bytes"]:
                raise ValueError(f"bound input changed: {path}")
    native_boundaries = {}
    for item in value["row_boundary_top2"]:
        if item["token_ids"] != [ROW_OPEN, EOS]:
            raise ValueError("row-boundary endpoint is not opener/EOS")
        if item["condition"] == "native":
            native_boundaries[item["lane"]] = item["margin"]
    if native_boundaries != {"failure": 8.674028396606445, "control": 8.91995620727539}:
        raise ValueError("native row-boundary margins changed")

    pair_count = condition_count = 0
    for group in value["pair_groups"]:
        native_row = group["retained_native_first_row_tokens"]
        for pair in group["pairs"]:
            pair_count += 1
            j, prefix = pair["first_unequal_index"], pair["common_prefix_tokens"]
            tokens = (pair["A_token_id"], pair["N_token_id"])
            if tokens == (ROW_OPEN, EOS) or not all(COORD_MIN <= token <= COORD_MAX for token in tokens):
                raise ValueError("candidate fork was conflated with opener/EOS")
            expected_reachability = "native_greedy_reachable" if native_row[:j] == prefix else "off_native_conditional"
            if pair["native_prefix_status"] != expected_reachability:
                raise ValueError("native-prefix status changed")
            native = pair["conditions"]["native"]["A_minus_N_margin"]
            for condition, diagnostic in pair["conditions"].items():
                condition_count += 1
                if diagnostic["common_prefix_max_logprob_abs_error"] > 2e-4:
                    raise ValueError("common-prefix score mismatch")
                expected = None if condition == "native" else diagnostic["A_minus_N_margin"] - native
                if diagnostic["cut_minus_own_native"] != expected:
                    raise ValueError("cut-minus-native arithmetic changed")
    if (pair_count, condition_count) != (20, 60):
        raise ValueError("pair traversal changed")

    frozen = json.loads((root / "coordination" / "root-acceptance" / "manifest-bindings.json").read_text())
    if frozen["candidate_sha256"] != sha(root / "candidate-manifest-v3.json") or frozen["errors"]:
        raise ValueError("v3 manifest identity changed")
    for name, expected in frozen["files"].items():
        path = Path(name)
        if sha(path) != expected["sha256"] or path.stat().st_size != expected["size_bytes"]:
            raise ValueError(f"v3-bound file changed: {path}")
    return {
        "schema": "history_source_sign.first_divergence_verification.v1",
        "status": "PASS", "source_sha256": sha(source_path),
        "pair_count": pair_count, "condition_level_diagnostics": condition_count,
        "opener_EOS_not_candidate_endpoint": True,
        "v3_bound_files_unchanged": frozen["unique_files"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = verify(args.root)
    if args.out:
        write_once(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
