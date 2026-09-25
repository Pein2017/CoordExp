"""CPU-only side-by-side response matrices for coordinate and full-row paths."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


FIELDS = (("x1", "x1_logprob"), ("y1_given_candidate_x1", "y1_given_x1_logprob"), ("joint_corner", "x1_y1_conditional_logprob"), ("full_row", "row_sum_logprob"))


def binding(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def integrate(root: Path) -> dict[str, Any]:
    states = []
    for scores_path in sorted((root / "qualification-01").glob("*/scores.json")):
        scores = json.loads(scores_path.read_text())
        clean = scores["conditions"]["clean"]["rows"]
        rows = []
        candidate_ids = [*scores["candidate_sets"]["A"]["row_ids"], *scores["candidate_sets"]["N"]["row_ids"]]
        for ident in candidate_ids:
            row_effects = {"candidate_id": ident, "candidate_set": "A" if ident in scores["candidate_sets"]["A"]["row_ids"] else "N", "prefix_route": "native_reachable" if ident == scores["actual_row_id"] else "supplied_candidate_under_native_prefix", "regions": {}}
            for region, condition in (("A", "ablate_A"), ("N", "ablate_N"), ("unrelated", "ablate_unrelated")):
                altered = scores["conditions"][condition]["rows"][ident]
                row_effects["regions"][region] = {name: float(altered[field] - clean[ident][field]) for name, field in FIELDS}
            rows.append(row_effects)
        states.append({"state_id": scores["state_id"], "model": scores["model"], "image_id": scores["image_id"], "stratum": "control" if "control" in scores["state_id"] else "target_first_revisit", "fixed_text_history": True, "candidate_prefix_labels": {row["candidate_id"]: row["prefix_route"] for row in rows}, "rows": rows})
    result = {"schema": "visual_instance_binding.response_matrices.v1", "status": "candidate", "primary_campaign": "qualification-01", "definition": "Each region entry reports ablated minus clean log probability for the same candidate row under the fixed native text prefix.", "columns": [name for name, _ in FIELDS], "prefix_route_labels": {"native_reachable": "the frozen native next row at the boundary", "supplied_candidate_under_native_prefix": "a frozen A/N candidate scored conditionally after the same native prefix"}, "states": states, "producer": binding(Path(__file__)), "excluded_execution": "scientific-01 states/ outputs are redundant pre-ruling execution and are excluded from this matrix."}
    out = root / "response-matrices.json"
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    result["binding"] = binding(out)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = integrate(args.root)
    print(json.dumps({"status": result["status"], "states": len(result["states"]), "output": str(args.root / "response-matrices.json")}, indent=2))


if __name__ == "__main__":
    main()
