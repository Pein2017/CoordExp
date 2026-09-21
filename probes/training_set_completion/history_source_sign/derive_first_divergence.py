"""CPU-only correction for Lane A row-boundary versus A/N fork margins."""
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
PLAN = ROOT / "selection" / "shared-admission.json"
ROW_OPEN, REF_END, BOX_START, ROW_END, EOS = 151646, 151647, 151648, 151649, 151645
COORD_MIN, COORD_MAX = 151670, 152669
ATOL = 2e-4
EXCLUDED_CONTROL = {
    "id": "A-actual-greedy",
    "token_sha256": "0739bb69023add13cbffe7cecf0656bf6980c9bf4f9c811c2e7d3e5398f2a442",
}


def binding(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def write_once(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def runtime_path(cell_id: str) -> Path:
    paths = [ROOT / "runtime" / cell_id, *sorted(ROOT.glob(f"runtime/{cell_id}--repair-*"))]
    ready = [path / "release.json" for path in paths if (path / "release.json").is_file()]
    if not ready:
        raise ValueError(f"missing completed cell: {cell_id}")
    return ready[-1]


def validate_row(row: dict[str, Any]) -> None:
    tokens, logps = row["token_ids"], row["token_logprobs"]
    if len(tokens) != len(logps) or len(tokens) < 2:
        raise ValueError(f"token/logprob alignment failed: {row['id']}")
    if tokens[0] != ROW_OPEN or tokens[-1] != ROW_END:
        raise ValueError(f"row boundary failed: {row['id']}")
    if abs(sum(float(value) for value in logps) - float(row["logprob_sum"])) > 1e-5:
        raise ValueError(f"row score accounting failed: {row['id']}")


def first_divergence(left: list[int], right: list[int]) -> int:
    if len(left) != len(right):
        raise ValueError("pair row lengths differ")
    try:
        return next(index for index, values in enumerate(zip(left, right)) if values[0] != values[1])
    except StopIteration as exc:
        raise ValueError("pair rows are identical") from exc


def token_role(tokens: list[int], index: int) -> str:
    ref_end = tokens.index(REF_END)
    roles = {ref_end: "description_end", ref_end + 1: "box_start", ref_end + 2: "x1",
             ref_end + 3: "y1", ref_end + 4: "x2", ref_end + 5: "y2", ref_end + 6: "terminator"}
    if index == 0:
        return "row_opener"
    if index < ref_end:
        return "description"
    return roles.get(index, "malformed")


def derive(root: Path, out: Path) -> dict[str, Any]:
    plan = json.loads((root / "selection" / "shared-admission.json").read_text())
    boundaries = [
        ("failure", item) for item in plan["lane_a"]["failure_boundaries"]
    ] + [("control", item) for item in plan["lane_a"]["control_boundaries"]]
    pair_groups, release_bindings, row_boundaries = [], [], []
    total_pairs = total_condition_diagnostics = 0

    for lane, boundary in boundaries:
        cells: dict[str, tuple[dict[str, Any], Path]] = {}
        for condition in ("native", "cut_A", "cut_C"):
            path = runtime_path(f"{boundary['id']}--{condition}")
            value = json.loads(path.read_text())
            if value.get("status") != "candidate_complete" or value.get("condition") != condition:
                raise ValueError(f"cell not complete: {boundary['id']}:{condition}")
            cells[condition] = (value, path)
            release_bindings.append(binding(path))
            top = value["candidate_scores"]["first_candidate_fork"]["top2"]
            if top["token_ids"] != [ROW_OPEN, EOS]:
                raise ValueError("saved row-boundary top-two is no longer opener/EOS")
            row_boundaries.append({
                "lane": lane, "boundary_id": boundary["id"], "condition": condition,
                "endpoint": "row_boundary_top2_opener_vs_EOS", "token_ids": top["token_ids"],
                "logits": top["logits"], "margin": top["margin"],
                "cut_minus_own_native": None,
            })
        native_margin = next(item["margin"] for item in row_boundaries if item["boundary_id"] == boundary["id"] and item["condition"] == "native")
        for item in row_boundaries:
            if item["boundary_id"] == boundary["id"] and item["condition"] != "native":
                item["cut_minus_own_native"] = item["margin"] - native_margin

        native_first = cells["native"][0]["release"]["first_complete_row"]
        if native_first is None:
            raise ValueError(f"native first row absent: {boundary['id']}")
        native_tokens = cells["native"][0]["release"]["token_ids"][native_first["start"]:native_first["end"]]
        native_sets = cells["native"][0]["candidate_scores"]["sets"]
        a_ids = [row["id"] for row in native_sets["A"]]
        if lane == "control":
            excluded = [row for row in native_sets["A"] if row["id"] == EXCLUDED_CONTROL["id"]]
            if len(excluded) != 1 or excluded[0]["token_sha256"] != EXCLUDED_CONTROL["token_sha256"]:
                raise ValueError("corrected control exclusion identity changed")
            a_ids = [ident for ident in a_ids if ident != EXCLUDED_CONTROL["id"]]
        n_ids = [row["id"] for row in native_sets["N"]]

        records = []
        for a_id in a_ids:
            for n_id in n_ids:
                conditions = {}
                native_pair_margin = None
                pair_meta = None
                for condition in ("native", "cut_A", "cut_C"):
                    sets = cells[condition][0]["candidate_scores"]["sets"]
                    a_row = next(row for row in sets["A"] if row["id"] == a_id)
                    n_row = next(row for row in sets["N"] if row["id"] == n_id)
                    validate_row(a_row)
                    validate_row(n_row)
                    j = first_divergence(a_row["token_ids"], n_row["token_ids"])
                    if a_row["token_ids"][:j] != n_row["token_ids"][:j]:
                        raise ValueError("common-prefix token mismatch")
                    prefix_error = max(
                        [abs(float(a_row["token_logprobs"][index]) - float(n_row["token_logprobs"][index])) for index in range(j)] or [0.0]
                    )
                    if prefix_error > ATOL:
                        raise ValueError(f"common-prefix score mismatch: {boundary['id']}:{a_id}:{n_id}:{condition}")
                    margin = float(a_row["token_logprobs"][j]) - float(n_row["token_logprobs"][j])
                    if condition == "native":
                        native_pair_margin = margin
                    conditions[condition] = {
                        "A_logprob": float(a_row["token_logprobs"][j]),
                        "N_logprob": float(n_row["token_logprobs"][j]),
                        "A_minus_N_margin": margin,
                        "cut_minus_own_native": None,
                        "common_prefix_max_logprob_abs_error": prefix_error,
                        "source": binding(cells[condition][1]),
                    }
                    current_meta = {
                        "first_unequal_index": j, "role": token_role(a_row["token_ids"], j),
                        "common_prefix_tokens": a_row["token_ids"][:j],
                        "A_token_id": a_row["token_ids"][j], "N_token_id": n_row["token_ids"][j],
                    }
                    if pair_meta is not None and current_meta != pair_meta:
                        raise ValueError("pair tokens changed across conditions")
                    pair_meta = current_meta
                assert native_pair_margin is not None and pair_meta is not None
                for condition in ("cut_A", "cut_C"):
                    conditions[condition]["cut_minus_own_native"] = conditions[condition]["A_minus_N_margin"] - native_pair_margin
                pair_meta["native_prefix_status"] = (
                    "native_greedy_reachable" if native_tokens[:pair_meta["first_unequal_index"]] == pair_meta["common_prefix_tokens"]
                    else "off_native_conditional"
                )
                pair_meta.update({"A_id": a_id, "N_id": n_id, "conditions": conditions})
                records.append(pair_meta)
        expected = 12 if lane == "failure" else 8
        if len(records) != expected:
            raise ValueError(f"pair denominator changed: {boundary['id']} {len(records)} != {expected}")
        total_pairs += len(records)
        total_condition_diagnostics += 3 * len(records)
        pair_groups.append({
            "lane": lane, "boundary_id": boundary["id"], "A_candidate_count": len(a_ids),
            "N_candidate_count": len(n_ids), "pair_count": len(records),
            "control_exclusion": EXCLUDED_CONTROL if lane == "control" else None,
            "retained_native_first_row_tokens": native_tokens, "pairs": records,
        })

    candidate_tokens = {
        (pair["A_token_id"], pair["N_token_id"])
        for group in pair_groups for pair in group["pairs"]
    }
    falsification_passed = all(
        ROW_OPEN not in tokens and EOS not in tokens and all(COORD_MIN <= token <= COORD_MAX for token in tokens)
        for tokens in candidate_tokens
    ) and all(item["token_ids"] == [ROW_OPEN, EOS] for item in row_boundaries)
    if not falsification_passed:
        raise ValueError("opener/EOS versus candidate-divergence falsification failed")

    result = {
        "schema": "history_source_sign.first_divergence.v1", "status": "candidate",
        "correction": "saved first_candidate_fork is row-boundary opener-versus-EOS, not an owner-candidate fork",
        "tolerance": {"common_prefix_logprob_atol": ATOL},
        "row_boundary_top2": row_boundaries,
        "pair_groups": pair_groups,
        "denominators": {"failure_pairs": 12, "corrected_control_pairs": 8, "total_pairs": total_pairs,
                         "condition_level_diagnostics": total_condition_diagnostics},
        "falsification": {
            "passed": True,
            "saved_row_boundary_token_ids": [ROW_OPEN, EOS],
            "candidate_first_divergence_token_pairs": [list(tokens) for tokens in sorted(candidate_tokens)],
            "statement": "Every candidate endpoint is a coordinate-token pair and differs from opener/EOS.",
        },
        "bindings": {
            "shared_admission": binding(root / "selection" / "shared-admission.json"),
            "candidate_v3": binding(root / "candidate-manifest-v3.json"),
            "correction_ruling": binding(root / "coordination" / "fork-surface-correction-04.txt"),
            "release_files": release_bindings,
            "producer": binding(Path(__file__).resolve()),
        },
        "claim_limits": [
            "Pairwise first-divergence margins are conditional on the exact common prefix and do not establish complete-row or global sequence preference.",
            "Off-native common prefixes are conditional diagnostics, not native-greedy reachable decisions.",
            "A numerical candidate token does not establish the physical owner of a cut release row.",
        ],
    }
    write_once(out, result)
    return result


def selfcheck() -> None:
    left = [ROW_OPEN, 2190, REF_END, BOX_START, 152458, 152340, 152520, 152367, ROW_END]
    right = [ROW_OPEN, 2190, REF_END, BOX_START, 152452, 152347, 152524, 152385, ROW_END]
    j = first_divergence(left, right)
    assert j == 4 and token_role(left, j) == "x1"
    assert [ROW_OPEN, EOS] != [left[j], right[j]]
    assert all(COORD_MIN <= token <= COORD_MAX for token in (left[j], right[j]))
    print("PASS opener/EOS is not the candidate first-divergence endpoint")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    result = derive(args.root, args.out or args.root / "first-divergence-v1.json")
    print(json.dumps({"status": result["status"], **result["denominators"], "falsification": result["falsification"]["passed"]}, indent=2))


if __name__ == "__main__":
    main()
