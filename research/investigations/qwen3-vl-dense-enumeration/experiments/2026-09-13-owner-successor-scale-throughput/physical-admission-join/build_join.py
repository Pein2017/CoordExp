"""Build a proposal-only 502-row review join for the real training consumer."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from alias_provenance import derive_required_alias_provenance


ADMISSION_INDEX_SCHEMA = "owner_successor_scale.physical_admission_review_index.v1"
def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binding(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "sha256": digest(path), "size_bytes": path.stat().st_size}


def load_bound(value: dict[str, Any], label: str) -> tuple[Path, Any]:
    path = Path(value["path"]).resolve()
    require(path.is_file(), f"missing {label}: {path}")
    require(digest(path) == value["sha256"], f"changed {label}: {path}")
    return path, json.loads(path.read_text())


def normalized_binding(value: dict[str, Any], label: str) -> dict[str, str]:
    path = Path(value["path"]).resolve()
    result = {"path": str(path), "sha256": value["sha256"]}
    require(path.is_file() and digest(path) == result["sha256"], f"changed {label}")
    return result


def source_shards(index: dict[str, Any]) -> dict[str, dict[str, dict[str, str]]]:
    sources = index["source_bindings"]
    if "source_shards" in sources:
        candidates = sources["source_shards"]
    elif "shards" in sources:
        candidates = sources["shards"]
    else:
        candidates = sources
    result = {}
    referenced = {row["shard"] for row in index["rows"]}
    for name in sorted(referenced):
        require(name in candidates, f"missing source shard {name}")
        source = candidates[name]
        require(all(key in source for key in ("jobs", "rows", "images")), f"incomplete source shard {name}")
        result[name] = {
            key: normalized_binding(source[key], f"{name} {key}")
            for key in ("jobs", "rows", "images")
        }
    return result


def identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(key) for key in (
        "job_id", "visual_group_id", "image_id", "example_id", "shard",
        "source_job_ordinal", "source_job_sha256", "source_row_ordinal", "source_row_sha256",
    ))


def normalize_review_row(row: dict[str, Any]) -> dict[str, Any]:
    row = copy.deepcopy(row)
    for side in ("c", "w"):
        review = row[side]["physical_review"]
        review.setdefault("status", review.get("admission_status", "HOLD"))
    return row


def validate_and_overlay(
    base: dict[str, Any], base_path: Path, proposal: dict[str, Any], proposal_path: Path,
    expected_schema: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    require(proposal.get("schema") == expected_schema, f"unexpected proposal schema: {proposal_path}")
    source = proposal.get("source_review_index")
    require(isinstance(source, dict), f"proposal lacks bound source review index: {proposal_path}")
    expected_source = {key: binding(base_path)[key] for key in ("path", "sha256")}
    require(normalized_binding(source, f"proposal source {proposal_path}") == expected_source,
            f"proposal does not bind its declared frozen index: {proposal_path}")
    base_rows = {row["job_id"]: row for row in base["rows"]}
    proposal_rows = {row["job_id"]: row for row in proposal["rows"]}
    require(set(base_rows) == set(proposal_rows), f"proposal job coverage mismatch: {proposal_path}")
    for job_id, row in proposal_rows.items():
        require(identity(row) == identity(base_rows[job_id]), f"proposal changed frozen identity: {job_id}")
    base_groups = {group["visual_group_id"]: group for group in base["groups"]}
    proposal_groups = {group["visual_group_id"]: group for group in proposal["groups"]}
    require(set(base_groups) == set(proposal_groups), f"proposal group coverage mismatch: {proposal_path}")
    for group_id, group in proposal_groups.items():
        require(group["job_ids"] == base_groups[group_id]["job_ids"], f"proposal changed group jobs: {group_id}")
    return ({key: normalize_review_row(value) for key, value in proposal_rows.items()}, proposal_groups)


def validate_recovery_manifest(manifest: dict[str, Any], frozen: dict[str, Any], frozen_path: Path) -> None:
    require(manifest.get("schema") == "physical_review_recovery_batches.v1", "recovery manifest schema")
    expected_source = {key: binding(frozen_path)[key] for key in ("path", "sha256")}
    require(normalized_binding(manifest["source_index"], "recovery manifest source") == expected_source,
            "recovery manifest does not bind frozen recovery index")
    groups = [value for batch in manifest["batches"] for value in batch["groups"]]
    jobs = [value for batch in manifest["batches"] for value in batch["job_ids"]]
    require(len(groups) == len(set(groups)) == 80, "recovery manifest exact 80-group partition")
    require(len(jobs) == len(set(jobs)) == 98, "recovery manifest exact 98-job partition")
    require(set(groups) == {g["visual_group_id"] for g in frozen["groups"]}, "recovery group union")
    require(set(jobs) == {r["job_id"] for r in frozen["rows"]}, "recovery job union")
    for batch in manifest["batches"]:
        input_path, value = load_bound(batch["input"], f"recovery batch {batch['batch']} input")
        require(value.get("schema") == "physical_review_recovery_batch.v1", f"recovery input schema: {input_path}")
        require(normalized_binding(value["source_index"], f"recovery input source {input_path}") == expected_source,
                f"recovery input does not bind frozen recovery index: {input_path}")
        require([g["visual_group_id"] for g in value["groups"]] == batch["groups"], "recovery batch groups")
        require([r["job_id"] for r in value["rows"]] == batch["job_ids"], "recovery batch jobs")


def apply_recovery_decisions(
    rows: dict[str, dict[str, Any]], groups: dict[str, dict[str, Any]], resume_manifest: dict[str, Any],
) -> tuple[set[str], set[str], list[dict[str, Any]], list[dict[str, Any]]]:
    reviewed_groups, reviewed_jobs, sources, invalid = set(), set(), [], []
    for batch in resume_manifest["batches"]:
        path = Path(batch["input"]["path"]).parent / "decisions.jsonl"
        if not path.exists():
            continue
        sources.append(binding(path))
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            group_id = None
            try:
                decision = json.loads(line)
                group_id = decision.get("visual_group_id")
                require(group_id in batch["groups"] and group_id not in reviewed_groups,
                        f"duplicate/out-of-batch recovery group at {path}:{number}")
                expected = groups[group_id]["job_ids"]
                actual = [job["job_id"] for job in decision.get("jobs", [])]
                require(actual == expected, f"recovery exact job coverage at {path}:{number}")
                for viewed in decision.get("viewed", []):
                    require(viewed.get("detail") == "original", f"recovery view detail at {path}:{number}")
                    normalized_binding(viewed, f"recovery viewed evidence at {path}:{number}")
                for value in decision["jobs"]:
                    require(value["c"]["status"] in {"candidate_accept", "HOLD"}, "recovery c status")
                    require(value["w"]["status"] in {"candidate_accept", "HOLD"}, "recovery w status")
                    require(value.get("pair_distinct") in {True, False, None}, "recovery pair status")
                    for side in ("c", "w"):
                        require(all(value[side].get(key) for key in ("reason", "newness", "newness_reason")),
                                f"recovery {side} rationale")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                invalid.append({"path": str(path.resolve()), "line": number,
                                "visual_group_id": group_id, "error": str(error)})
                continue
            pair_candidates = []
            for value in decision["jobs"]:
                job_id = value["job_id"]
                for side in ("c", "w"):
                    rows[job_id][side]["physical_review"] = {
                        "status": value[side]["status"], "admission_status": value[side]["status"],
                        "reason": value[side]["reason"], "newness": value[side]["newness"],
                        "newness_reason": value[side]["newness_reason"],
                        "reviewer_identity": decision["reviewer"], "review_route": "individual view_image",
                    }
                positive = (value["c"]["status"] == value["w"]["status"] == "candidate_accept"
                            and value.get("pair_distinct") is True)
                rows[job_id]["job_status"] = "candidate_accept" if positive else "HOLD"
                rows[job_id]["physical_pair_review"] = {
                    "status": rows[job_id]["job_status"], "distinct_physical_entities": value.get("pair_distinct"),
                    "reason": value.get("pair_reason"), "requires_root_admission": True,
                    "not_a_training_target": True,
                }
                pair_candidates.append(positive)
                reviewed_jobs.add(job_id)
            groups[group_id]["review_status"] = "candidate_accept" if any(pair_candidates) else "HOLD"
            groups[group_id]["recovery_decision_source"] = {"path": str(path.resolve()), "line": number}
            reviewed_groups.add(group_id)
    return reviewed_groups, reviewed_jobs, sources, invalid


def apply_root_overrides(rows: dict[str, dict[str, Any]], groups: dict[str, dict[str, Any]], root: dict[str, Any]) -> set[str]:
    require(root.get("schema") == "owner_successor_scale.root_individual_review.v1"
            and root.get("training_admission") is False, "root override boundary")
    overridden = set()
    for record in root["records"]:
        group_id = record["group"]
        require(group_id in groups and int(groups[group_id]["image_id"]) == int(record["image"]), "root group/image")
        overridden.add(group_id)
        positive = record["c"] == record["w"] == "candidate_supported"
        for job_id in groups[group_id]["job_ids"]:
            for side in ("c", "w"):
                value = record[side]
                rows[job_id][side]["physical_review"].update({
                    "status": value, "admission_status": value, "reason": record["reason"],
                    "root_override_not_training_admission": True,
                })
            rows[job_id]["job_status"] = "candidate_supported_root_review_not_admission" if positive else "HOLD"
        groups[group_id]["review_status"] = "candidate_supported_root_review_not_admission" if positive else "HOLD"
        groups[group_id]["root_override"] = copy.deepcopy(record)
    return overridden


def positive_summary(rows: dict[str, dict[str, Any]], groups: dict[str, dict[str, Any]], pool: dict[str, Any]) -> list[dict[str, Any]]:
    order = {int(image): number for number, image in enumerate(pool["image_ids"])}
    positives = []
    for group_id, group in groups.items():
        if group.get("review_status") not in {"candidate_accept", "candidate_supported_root_review_not_admission"}:
            continue
        jobs = [rows[job_id] for job_id in group["job_ids"] if rows[job_id].get("job_status") in {
            "candidate_accept", "candidate_supported_root_review_not_admission",
        }]
        require(jobs, f"positive group without pair-positive exact job: {group_id}")
        jobs.sort(key=lambda row: (row["source_identity"]["history_index"],
                                   row["source_identity"]["candidate_index"], row["job_id"]))
        reasons = [{"job_id": row["job_id"], "c": row["c"]["physical_review"]["reason"],
                    "w": row["w"]["physical_review"]["reason"]} for row in jobs]
        positives.append({
            "visual_group_id": group_id, "image_id": int(group["image_id"]),
            "pool_ordinal": order[int(group["image_id"])],
            "exact_candidate_job_ids": [row["job_id"] for row in jobs],
            "proposed_canonical_job_id": jobs[0]["job_id"],
            "canonical_basis": "earliest frozen history/candidate/job tuple; proposal only, root must sign",
            "history_alias_issue": ("root must choose one canonical exact history" if len(jobs) > 1 else None),
            "same_image_aliases": sorted({alias for row in jobs for alias in row.get("same_image_aliases", [])}),
            "execution_aliases_same_visual_group": sorted({alias for row in jobs for alias in row.get("execution_aliases_same_visual_group", [])}),
            "possible_near_visual_alias_group_ids": group.get("possible_near_visual_alias_group_ids", []),
            "visual_review_alias_cluster_id": group.get("visual_review_alias_cluster_id"),
            "reasons": reasons,
            "root_override_applied": "root_override" in group,
            "disposition": "candidate_for_root_decision_not_admitted",
        })
    return sorted(positives, key=lambda value: (value["pool_ordinal"], value["visual_group_id"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    inputs = json.loads(args.inputs.read_text())
    require(inputs.get("schema") == "owner_successor_scale.physical_admission_join_inputs.v1", "join inputs schema")
    pool_path, pool = load_bound(inputs["pool"], "pool")
    confirmation_path, _ = load_bound(inputs["confirmation_selection"], "confirmation selection")
    root_path, root = load_bound(inputs["root_review_overrides"], "root review overrides")

    rows: dict[str, dict[str, Any]] = {}
    groups: dict[str, dict[str, Any]] = {}
    shards: dict[str, dict[str, dict[str, str]]] = {}
    completed_bindings = []
    partition_counts = []
    for partition in inputs["completed_partitions"]:
        frozen_path, frozen = load_bound(partition["frozen_index"], f"{partition['name']} frozen index")
        proposal_path, proposal = load_bound(partition["proposal"], f"{partition['name']} proposal")
        proposal_rows, proposal_groups = validate_and_overlay(
            frozen, frozen_path, proposal, proposal_path, partition["proposal_schema"],
        )
        require(not set(rows).intersection(proposal_rows), f"duplicate jobs in {partition['name']}")
        require(not set(groups).intersection(proposal_groups), f"duplicate groups in {partition['name']}")
        rows.update(proposal_rows); groups.update(copy.deepcopy(proposal_groups))
        for name, value in source_shards(frozen).items():
            require(name not in shards or shards[name] == value, f"conflicting source shard {name}")
            shards[name] = value
        completed_bindings.append({"name": partition["name"], "frozen_index": binding(frozen_path),
                                   "proposal": binding(proposal_path)})
        partition_counts.append({"name": partition["name"], "reviewed_jobs": len(proposal_rows),
                                 "reviewed_groups": len(proposal_groups)})

    completed_jobs = len(rows)
    completed_groups = len(groups)
    require(completed_jobs == 404 and completed_groups == 349, "frozen completed-review 404-job/349-group coverage")

    recovery_path, recovery = load_bound(inputs["recovery"]["frozen_index"], "recovery frozen index")
    resume_path, resume = load_bound(inputs["recovery"]["resume_manifest"], "recovery resume manifest")
    validate_recovery_manifest(resume, recovery, recovery_path)
    recovery_rows = {row["job_id"]: normalize_review_row(row) for row in recovery["rows"]}
    recovery_groups = {group["visual_group_id"]: copy.deepcopy(group) for group in recovery["groups"]}
    reviewed_recovery_groups, reviewed_recovery_jobs, decision_sources, invalid_decisions = apply_recovery_decisions(
        recovery_rows, recovery_groups, resume,
    )
    require(not set(rows).intersection(recovery_rows) and not set(groups).intersection(recovery_groups), "recovery overlap")
    rows.update(recovery_rows); groups.update(recovery_groups)
    for name, value in source_shards(recovery).items():
        require(name not in shards or shards[name] == value, f"conflicting recovery source shard {name}")
        shards[name] = value

    require(len(rows) == 502 and len(set(rows)) == 502, "exact 502 candidate-job union")
    require(sum(len(group["job_ids"]) for group in groups.values()) == 502, "group/job exact coverage")
    require({job for group in groups.values() for job in group["job_ids"]} == set(rows), "group/job union")
    derive_required_alias_provenance(list(rows.values()), list(groups.values()))
    overridden = apply_root_overrides(rows, groups, root)
    positives = positive_summary(rows, groups, pool)
    candidate_images = {item["image_id"] for item in positives}
    candidate_package_ceiling = sum(min(2, sum(item["image_id"] == image for item in positives)) for image in candidate_images)
    reviewed_job_count = completed_jobs + len(reviewed_recovery_jobs)
    reviewed_group_count = completed_groups + len(reviewed_recovery_groups)
    pending_jobs = 502 - reviewed_job_count
    pending_groups = len(groups) - reviewed_group_count

    args.output_dir.mkdir(parents=True, exist_ok=True)
    aggregate = {
        "schema": ADMISSION_INDEX_SCHEMA,
        "status": "review_complete" if pending_jobs == 0 else "review_ready_pending_view_image",
        "scope": "proposal-only exact 502-job transport; root decisions absent",
        "source_bindings": shards, "rows": list(rows.values()), "groups": list(groups.values()),
        "counts": {"candidate_local_w_rows": 502, "exact_visual_groups": len(groups),
                   "reviewed_jobs": reviewed_job_count, "pending_jobs": pending_jobs,
                   "reviewed_groups": reviewed_group_count, "pending_groups": pending_groups},
        "admission_boundary": "No root training decisions in this artifact.",
    }
    aggregate_path = args.output_dir / "consumer-review-index-v1.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    summary = {
        "schema": "owner_successor_scale.physical_admission_candidate_summary.v1",
        "status": ("candidate_proposals_incomplete_root_decision_blocked_by_pending_recovery"
                   if pending_jobs else "candidate_proposals_complete_pending_root_decisions"),
        "counts": {
            "frozen_jobs": 502, "frozen_visual_groups": len(groups),
            "completed_proposal_jobs": reviewed_job_count, "pending_recovery_jobs": pending_jobs,
            "completed_proposal_groups": reviewed_group_count, "pending_recovery_groups": pending_groups,
            "effective_candidate_groups_after_root_overrides": len(positives),
            "effective_candidate_exact_jobs_after_root_overrides": sum(len(x["exact_candidate_job_ids"]) for x in positives),
            "effective_candidate_images_after_root_overrides": len(candidate_images),
            "candidate_package_ceiling_under_max2_per_image": candidate_package_ceiling,
        },
        "fixed_selection_rule": "pool order; distinct-image-first; at most two packages/image; floor 32 packages/16 images",
        "selection_not_run": True,
        "floor_readiness": f"HOLD_pending_{pending_jobs}_recovery_jobs" if pending_jobs else "pending_root_decisions",
        "root_override_groups": sorted(overridden),
        "candidate_positive_groups": positives,
        "pending_recovery": {"resume_manifest": binding(resume_path),
                             "decision_sources": decision_sources,
                             "invalid_decisions": invalid_decisions,
                             "pending_group_ids": sorted(set(recovery_groups) - reviewed_recovery_groups),
                             "pending_job_ids": sorted(set(recovery_rows) - reviewed_recovery_jobs)},
        "sources": {"pool": binding(pool_path), "confirmation_selection": binding(confirmation_path),
                    "root_review_overrides": binding(root_path), "completed_partitions": completed_bindings,
                    "recovery_frozen_index": binding(recovery_path)},
        "partition_counts": partition_counts,
        "consumer_review_index": binding(aggregate_path),
        "root_decisions_created": False,
    }
    summary_path = args.output_dir / "candidate-summary-v1.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"consumer_review_index": binding(aggregate_path), "candidate_summary": binding(summary_path),
                      "counts": summary["counts"]}, indent=2))


if __name__ == "__main__":
    main()
