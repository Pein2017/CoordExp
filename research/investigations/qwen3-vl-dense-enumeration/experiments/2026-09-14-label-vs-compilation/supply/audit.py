#!/usr/bin/env python3
"""Project the frozen 2026-09-13 supply ledger into bounded scalar counts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


BASE = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-13-owner-successor-scale-throughput"
)
FILES = {
    "prior_unit": Path(
        "/data/CoordExp/.worktrees/research-probes/research/investigations/"
        "qwen3-vl-dense-enumeration/experiments/"
        "2026-09-13-owner-successor-scale-throughput/unit.md"
    ),
    "supply_recovery_result": Path(
        "/data/CoordExp/.worktrees/research-probes/research/investigations/"
        "qwen3-vl-dense-enumeration/experiments/"
        "2026-09-13-owner-successor-scale-throughput/supply-recovery/result.md"
    ),
    "completion": BASE / "supply-recovery/completion.json",
    "verification": BASE / "supply-recovery/verification-and-cost.json",
    "image_records": BASE / "supply-recovery/image-records.json",
    "ordered_jobs": BASE / "supply-recovery/ordered-jobs.json",
    "ordered_results": BASE / "supply-recovery/ordered-results.json",
    "review_index": BASE / "physical-admission-join/final-v2/consumer-review-index-v2.json",
    "root_decisions": BASE / "physical-admission/root-training-decisions-v2.json",
    "transport_proof": BASE / "physical-admission-join/final-v2/transport-correction-proof-v2.json",
    "physical_bank": BASE / "training/physical-bank.json",
}
BURDEN_KEYS = (
    "free_strict_repeats_including_history",
    "free_geometry_invalid",
    "free_other_malformed",
    "cap",
)


def load(path: Path):
    with path.open() as handle:
        return json.load(handle)


def binding(path: Path) -> dict:
    raw = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }


def burden_counts(rows: list[dict]) -> dict:
    counts = Counter()
    for row in rows:
        burden = row["burden"]
        flags = [burden[key] > 0 for key in BURDEN_KEYS]
        for key, flag in zip(BURDEN_KEYS, flags, strict=True):
            counts[f"{key}_positive"] += int(flag)
        counts["any_machine_burden"] += int(any(flags))
        counts["no_machine_burden"] += int(not any(flags))
        counts["eos"] += int(burden["eos"] > 0)
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    completion = load(FILES["completion"])
    verification = load(FILES["verification"])
    records = load(FILES["image_records"])["records"]
    jobs = load(FILES["ordered_jobs"])["jobs"]
    results = load(FILES["ordered_results"])["rows"]
    review = load(FILES["review_index"])
    decisions = load(FILES["root_decisions"])["decisions"]
    transport = load(FILES["transport_proof"])
    bank = load(FILES["physical_bank"])

    by_result = {row["job_id"]: row for row in results}
    groups = {group["visual_group_id"]: group for group in review["groups"]}
    decision_by_group = {row["visual_group_id"]: row for row in decisions}
    admitted_groups = {
        group_id
        for group_id, row in decision_by_group.items()
        if row["disposition"] == "admit"
    }
    selected_groups = {p["provenance"]["visual_group_id"] for p in bank["packages"]}
    selected_results = [by_result[p["package_id"]] for p in bank["packages"]]
    candidate_w = [r for r in results if r["local_w"]["status"] == "candidate_local_w"]

    # Frozen-contract and exact-join checks: fail rather than silently change scope.
    assert len(records) == completion["images"] == 4096
    assert len(jobs) == len(results) == completion["conditional_calls"] == 589
    assert len(by_result) == len(results)
    assert set(by_result) == {job["job_id"] for job in jobs}
    assert len(review["rows"]) == len(candidate_w) == 502
    assert len(groups) == len(decision_by_group) == 429
    assert selected_groups <= admitted_groups
    assert len(bank["packages"]) == bank["counts"]["packages"] == 53
    assert len({p["image_id"] for p in bank["packages"]}) == bank["counts"]["images"] == 39

    nomination_images = {job["image_id"] for job in jobs}
    candidate_w_images = {row["image_id"] for row in candidate_w}
    admitted_images = {groups[group_id]["image_id"] for group_id in admitted_groups}
    no_nomination = [row for row in records if row["nominations"] == 0]
    root_reason_counts = Counter(
        (row["disposition"], row["reason"]) for row in decisions
    )
    witness_status = Counter(row["local_w"]["status"] for row in results)
    held_status = Counter(item["status"] for row in records for item in row["held"])

    suffix_candidates = []
    for package, row in zip(bank["packages"], selected_results, strict=True):
        burden = row["burden"]
        flags = {key: burden[key] for key in BURDEN_KEYS if burden[key] > 0}
        if not flags:
            continue
        group = groups[package["provenance"]["visual_group_id"]]
        suffix_candidates.append(
            {
                "package_id": package["package_id"],
                "image_id": package["image_id"],
                "visual_group_id": package["provenance"]["visual_group_id"],
                "machine_burden": flags,
                "free_valid_rows": burden["free_valid_rows"],
                "free_row_starts": burden["free_row_starts"],
                "eos": burden["eos"],
                "representative_card": group["representative_card"]["path"],
                "raw_record": {
                    "path": str(FILES["ordered_results"]),
                    "selector": f".rows[] | select(.job_id == {json.dumps(package['package_id'])})",
                },
            }
        )

    result = {
        "schema": "label_vs_compilation.supply_evidence_audit.v1",
        "status": "candidate_cpu_read_only_existing_evidence_accounting",
        "source_bindings": {name: binding(path) for name, path in FILES.items()},
        "acquisition": {
            "frozen_images": len(records),
            "images_with_gt_backed_nomination": len(nomination_images),
            "images_without_gt_backed_nomination": len(no_nomination),
            "no_nomination_no_recorded_filter_reason": sum(not row["held"] for row in no_nomination),
            "no_nomination_with_recorded_gt_filter_hold": sum(bool(row["held"]) for row in no_nomination),
            "recorded_gt_filter_hold_rows": dict(held_status),
            "conditional_nomination_jobs": len(jobs),
            "distinct_nominated_histories": len(
                {(job["image_id"], job["history_index"]) for job in jobs}
            ),
            "nomination_count_per_image": dict(
                sorted(Counter(row["nominations"] for row in records).items())
            ),
        },
        "conditional_immediate_witness": {
            "job_status_counts": dict(witness_status),
            "images_with_any_candidate_local_w": len(candidate_w_images),
            "nominated_images_without_candidate_local_w": len(nomination_images - candidate_w_images),
        },
        "physical_admission": {
            "candidate_local_w_rows": len(candidate_w),
            "exact_visual_groups": len(groups),
            "root_disposition_counts": dict(Counter(row["disposition"] for row in decisions)),
            "root_reason_counts": [
                {"disposition": key[0], "reason": key[1], "groups": value}
                for key, value in sorted(root_reason_counts.items())
            ],
            "candidate_w_images": len(candidate_w_images),
            "root_admitted_images": len(admitted_images),
            "candidate_w_images_without_root_admitted_group": len(candidate_w_images - admitted_images),
            "root_admitted_groups": len(admitted_groups),
            "final_packages": len(bank["packages"]),
            "final_images": len({p["image_id"] for p in bank["packages"]}),
            "admitted_groups_excluded_by_max2_per_image_selection": sorted(admitted_groups - selected_groups),
            "aim_images": 64,
            "observed_gap_to_aim_images": 64 - len(admitted_images),
            "floor_packages": 32,
            "floor_images": 16,
        },
        "later_continuation_machine_burden": {
            "definition": (
                "Existing machine ledger only: a candidate-local-w continuation has a later strict repeat, "
                "geometry-invalid row, other malformed content, or cap. This is not semantic or physical bad-suffix adjudication."
            ),
            "candidate_local_w_rows": burden_counts(candidate_w),
            "selected_packages": burden_counts(selected_results),
            "selected_package_review_candidates": suffix_candidates,
            "packages_excluded_by_this_burden_under_frozen_admission_contract": 0,
        },
        "technical_and_unknown": {
            "pending_natural": completion["pending_natural"],
            "pending_conditional": completion["pending_conditional"],
            "interrupted_incomplete_image_attempts": verification["interrupted_incomplete_image_attempts"],
            "uncommitted_forwards": verification["uncommitted_forwards"],
            "current_review_pending_groups": review["counts"]["pending_groups"],
            "current_review_pending_jobs": review["counts"]["pending_jobs"],
            "transport_repair_status": transport["status"],
            "current_packages_lost_to_technical_failure": 0,
            "root_hold_groups_are_unknown_neutral_not_negative_labels": sum(
                row["disposition"] == "hold" for row in decisions
            ),
        },
        "claim_boundary": [
            "No-GT-backed nomination is an acquisition-path fact, not evidence of an unlabeled physical owner.",
            "The prediction-conditioned ledger cannot count owners missed by every prediction and cannot support an original-image census.",
            "The observed 25-image gap to the 64-image aim has no identified missing-label counterfactual in these artifacts.",
            "The seven selected-package suffix candidates are machine-burden review cases, not missing-owner candidates or package rejections.",
            "All physical HOLDs remain unknown-neutral; no negative label or positive owner is inferred here.",
        ],
    }

    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
