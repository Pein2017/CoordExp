"""Repair only missing consumer alias provenance in an immutable final join."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from alias_provenance import ALIAS_FIELDS, derive_required_alias_provenance


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha(value: Any) -> str:
    return sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def binding(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "sha256": sha_bytes(path.read_bytes()), "size_bytes": path.stat().st_size}


def load_exact(path: Path, expected_sha: str, label: str) -> Any:
    require(path.is_file() and sha_bytes(path.read_bytes()) == expected_sha, f"changed {label}: {path}")
    return json.loads(path.read_text())


def projection(index: dict[str, Any], kind: str) -> Any:
    if kind == "source_identity":
        return [{"job_id": row["job_id"], "image_id": row["image_id"], "example_id": row["example_id"],
                 "shard": row["shard"], "source_identity": row["source_identity"],
                 "source_job_ordinal": row["source_job_ordinal"], "source_job_sha256": row["source_job_sha256"],
                 "source_row_ordinal": row["source_row_ordinal"], "source_row_sha256": row["source_row_sha256"]}
                for row in index["rows"]]
    if kind == "tokens_history":
        return [{"job_id": row["job_id"], "c_token_ids_sha256": row["c"]["token_ids_sha256"],
                 "w_token_ids_sha256": row["w"]["token_ids_sha256"],
                 "exact_history": row.get("exact_history"),
                 "history_index": row["source_identity"]["history_index"],
                 "candidate_index": row["source_identity"]["candidate_index"]}
                for row in index["rows"]]
    if kind == "labels":
        return [{"job_id": row["job_id"], "job_status": row.get("job_status"),
                 "c_physical_review": row["c"]["physical_review"],
                 "w_physical_review": row["w"]["physical_review"],
                 "physical_pair_review": row.get("physical_pair_review")}
                for row in index["rows"]]
    if kind == "groups":
        return [{"visual_group_id": group["visual_group_id"], "image_id": group["image_id"],
                 "job_ids": group["job_ids"], "review_status": group.get("review_status")}
                for group in index["groups"]]
    raise ValueError(kind)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-index", type=Path, required=True)
    parser.add_argument("--input-index-sha256", required=True)
    parser.add_argument("--input-summary", type=Path, required=True)
    parser.add_argument("--input-summary-sha256", required=True)
    parser.add_argument("--root-decisions", type=Path, required=True)
    parser.add_argument("--root-decisions-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "no final-v2 overwrite")

    old_index = load_exact(args.input_index, args.input_index_sha256, "final-v1 index")
    old_summary = load_exact(args.input_summary, args.input_summary_sha256, "final-v1 summary")
    decisions = load_exact(args.root_decisions, args.root_decisions_sha256, "root decisions")
    require(old_index.get("schema") == "owner_successor_scale.physical_admission_review_index.v1"
            and old_index.get("status") == "review_complete" and len(old_index["rows"]) == 502,
            "complete final-v1 review index")
    require(decisions.get("schema") == "owner_successor_scale.physical_training_decisions.v1"
            and decisions.get("status") == "root_decisions_complete", "immutable complete root decisions")
    decision_index_binding = {k: binding(args.input_index)[k] for k in ("path", "sha256")}
    require(decisions["review_indexes"] == [decision_index_binding], "root decisions do not bind final-v1")

    new_index = copy.deepcopy(old_index)
    additions = derive_required_alias_provenance(new_index["rows"], new_index["groups"])
    require(len(additions) == 140 and all(set(value["added"]) == set(ALIAS_FIELDS) for value in additions),
            "expected exact 140-row/two-field repair")
    addition_jobs = {value["job_id"] for value in additions}
    stripped = copy.deepcopy(new_index)
    for row in stripped["rows"]:
        if row["job_id"] in addition_jobs:
            for field in ALIAS_FIELDS:
                row.pop(field)
    require(stripped == old_index, "final-v2 changes more than missing alias provenance")

    groups = {group["visual_group_id"]: group for group in new_index["groups"]}
    rows = {row["job_id"]: row for row in new_index["rows"]}
    require(len(decisions["decisions"]) == len(groups) == 429, "root/group decision coverage")
    require({value["visual_group_id"] for value in decisions["decisions"]} == set(groups),
            "root group membership changed")
    admitted_missing_before, admitted_missing_after = [], []
    for value in decisions["decisions"]:
        if value["disposition"] != "admit":
            continue
        job_id = value["canonical_job_id"]
        require(job_id in groups[value["visual_group_id"]]["job_ids"], "root canonical job/group membership")
        if any(field not in next(row for row in old_index["rows"] if row["job_id"] == job_id) for field in ALIAS_FIELDS):
            admitted_missing_before.append(job_id)
        if any(field not in rows[job_id] for field in ALIAS_FIELDS):
            admitted_missing_after.append(job_id)
    require(admitted_missing_before and not admitted_missing_after, "real admitted-row failure/repaired check")

    args.output_dir.mkdir(parents=True)
    index_path = args.output_dir / "consumer-review-index-v2.json"
    index_path.write_text(json.dumps(new_index, indent=2, sort_keys=True) + "\n")
    additions_path = args.output_dir / "alias-additions-v2.jsonl"
    additions_path.write_text("".join(json.dumps(value, sort_keys=True) + "\n" for value in additions))
    new_summary = copy.deepcopy(old_summary)
    new_summary["consumer_review_index"] = binding(index_path)
    new_summary["transport_correction"] = {
        "status": "consumer_required_alias_provenance_derived_from_frozen_exact_visual_groups",
        "input_final_v1": binding(args.input_index), "alias_additions": binding(additions_path),
        "rows_repaired": len(additions), "fields_added": list(ALIAS_FIELDS),
        "root_decisions_unchanged": binding(args.root_decisions),
    }
    summary_path = args.output_dir / "candidate-summary-v2.json"
    summary_path.write_text(json.dumps(new_summary, indent=2, sort_keys=True) + "\n")

    invariants = {}
    for name in ("source_identity", "tokens_history", "labels", "groups"):
        before, after = projection(old_index, name), projection(new_index, name)
        require(before == after, f"changed {name} projection")
        invariants[name] = {"unchanged": True, "sha256": canonical_sha(before)}
    proof = {
        "schema": "owner_successor_scale.physical_admission_alias_transport_proof.v2",
        "status": "passed_exact_alias_addition_only",
        "input_index": binding(args.input_index), "output_index": binding(index_path),
        "input_summary": binding(args.input_summary), "output_summary": binding(summary_path),
        "root_decisions": binding(args.root_decisions), "alias_additions": binding(additions_path),
        "counts": {"rows": 502, "groups": 429, "repaired_rows": len(additions),
                   "existing_alias_rows_unchanged": 502 - len(additions),
                   "root_decisions": len(decisions["decisions"]),
                   "root_admits": sum(x["disposition"] == "admit" for x in decisions["decisions"]),
                   "root_admitted_rows_missing_before": len(admitted_missing_before),
                   "root_admitted_rows_missing_after": len(admitted_missing_after)},
        "derivation": {
            "execution_aliases_same_visual_group": "all job_ids in the row's frozen exact visual group",
            "same_image_aliases": "other job_ids in the row's frozen exact visual group; never self",
            "co_image_only_alias_inference": False,
        },
        "unchanged_projections": invariants,
        "root_group_membership_unchanged": True,
        "old_real_failure_example": admitted_missing_before[0],
    }
    proof_path = args.output_dir / "transport-correction-proof-v2.json"
    proof_path.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"index": binding(index_path), "summary": binding(summary_path),
                      "proof": binding(proof_path), "counts": proof["counts"]}, indent=2))


if __name__ == "__main__":
    main()
