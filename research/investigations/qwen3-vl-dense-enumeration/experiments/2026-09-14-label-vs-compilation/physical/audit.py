#!/usr/bin/env python3
"""Independently rejoin frozen blind32 decisions to source and GT50 rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


OLD = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-13-owner-successor-scale-throughput/evaluation/paired-consumer-v1"
)
PREP = OLD / "blind32-review-preparation-v1"
DEFAULT_OUTPUT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-14-label-vs-compilation/physical/result.json"
)
ARMS = ("N16-anchor", "A", "B")
PAIRS = (("N16-anchor", "A"), ("N16-anchor", "B"), ("A", "B"))
FILES = {
    "prior_physical_receipt_v1": DEFAULT_OUTPUT.with_name("result-v1.json"),
    "comparison": OLD / "blind-review-comparison-v1.json",
    "source_map": OLD / "blind-review-source-map.json",
    "queue": OLD / "blind-review-queue.jsonl",
    "manifest": PREP / "manifest.json",
    "gt50_N16-anchor": OLD.parent / "paired-preparation/n16-anchor-rows-896.jsonl",
    "gt50_A": OLD / "A-consumer.jsonl",
    "gt50_B": OLD / "B-consumer.jsonl",
}
DECISION_FILES = tuple(sorted((PREP / "batches").glob("batch-*/decisions.jsonl")))
GT50_CLASSES = (
    "has_GT50_matched_source_prediction",
    "supported_but_GT50_unmatched",
    "unidentifiable",
)


def json_file(path: Path):
    with path.open() as handle:
        return json.load(handle)


def jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def binding(path: Path) -> dict:
    raw = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }


def aggregate_gt50(rows: list[dict]) -> dict:
    counts = Counter()
    for row in rows:
        for key in ("tp", "fp", "fn"):
            counts[key] += row["score"]["50"][key]
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "annotated_owner_denominator": tp + fn,
        "predicted_box_denominator": tp + fp,
        "precision": tp / (tp + fp),
        "recall": tp / (tp + fn),
        "f1": 2 * tp / (2 * tp + fp + fn),
    }


def compare_sets(sets: dict[str, set[str]], source: str, target: str) -> dict:
    gained = sorted(sets[target] - sets[source])
    lost = sorted(sets[source] - sets[target])
    retained = sorted(sets[source] & sets[target])
    return {
        "source": source,
        "target": target,
        "gained": gained,
        "lost": lost,
        "retained": retained,
        "counts": {
            "gained": len(gained),
            "lost": len(lost),
            "retained": len(retained),
            "net": len(gained) - len(lost),
        },
    }


def summarize_clusters(clusters: list[dict]) -> dict:
    sets = {
        arm: {cluster["key"] for cluster in clusters if cluster["presence"][arm]}
        for arm in ARMS
    }
    return {
        "clusters": len(clusters),
        "per_arm_present": {arm: len(sets[arm]) for arm in ARMS},
        "comparisons": {
            f"{source}->{target}": compare_sets(sets, source, target)
            for source, target in PAIRS
        },
    }


def compact_summary(summary: dict) -> dict:
    return {
        "clusters": summary["clusters"],
        "per_arm_present": summary["per_arm_present"],
        "comparisons": {
            key: value["counts"] for key, value in summary["comparisons"].items()
        },
    }


def classify_cluster_gt50_source(
    cluster: dict,
    arm: str,
    source_by_proposal: dict[str, dict],
    gt_rows: dict[str, dict[int, dict]],
) -> dict:
    sources = [
        source_by_proposal[proposal_id]
        for proposal_id in cluster["proposal_ids"]
        if source_by_proposal[proposal_id]["source_arm"] == arm
    ]
    if not sources:
        raise AssertionError(f"cluster {cluster['key']} has no {arm} source prediction")

    prediction_indices: list[int] = []
    matched_indices: list[int] = []
    problems: list[str] = []
    row = gt_rows.get(arm, {}).get(cluster["image_id"])
    for source in sources:
        index = source.get("source_prediction_index")
        if row is None:
            problems.append("bound_source_row_missing")
            continue
        if not isinstance(index, int) or index < 0 or index >= len(row["parsed"]["pred"]):
            problems.append("source_prediction_index_has_no_exact_parsed_prediction")
            continue
        prediction_indices.append(index)
        match_indices = {match["pred_index"] for match in row["score"]["50"]["matches"]}
        if index in match_indices:
            matched_indices.append(index)

    if problems:
        classification = "unidentifiable"
    elif matched_indices:
        classification = "has_GT50_matched_source_prediction"
    else:
        classification = "supported_but_GT50_unmatched"
    return {
        "key": cluster["key"],
        "image_id": cluster["image_id"],
        "bound_arm": arm,
        "classification": classification,
        "has_any_extent_or_class_caveat": bool(cluster["extent_or_class_caveats"]),
        "source_prediction_indices": sorted(set(prediction_indices)),
        "gt50_matched_source_prediction_indices": sorted(set(matched_indices)),
        "unidentifiable_reasons": sorted(set(problems)),
    }


def stratify_records(records: list[dict]) -> dict:
    counts = Counter(record["classification"] for record in records)
    return {
        "clusters": len(records),
        "class_counts": {name: counts[name] for name in GT50_CLASSES},
        "records": sorted(records, key=lambda record: record["key"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    assert len(DECISION_FILES) == 8
    decisions = [row for path in DECISION_FILES for row in jsonl(path)]
    source_rows = json_file(FILES["source_map"])["rows"]
    queue_images = jsonl(FILES["queue"])
    manifest = json_file(FILES["manifest"])
    reference = json_file(FILES["comparison"])

    image_ids = {row["image_id"] for row in decisions}
    manifest_ids = {
        image_id for batch in manifest["batches"] for image_id in batch["images"]
    }
    source_by_proposal = {row["proposal_id"]: row for row in source_rows}
    queue_rows = [
        {"image_id": image["image_id"], **proposal}
        for image in queue_images
        for proposal in image["proposals"]
    ]
    queue_by_proposal = {row["proposal_id"]: row for row in queue_rows}

    assert len(decisions) == len(image_ids) == manifest["images"] == 32
    assert image_ids == manifest_ids
    assert len(source_rows) == len(source_by_proposal) == 607
    assert len(queue_images) == len({row["image_id"] for row in queue_images}) == 32
    assert len(queue_rows) == len(queue_by_proposal) == manifest["proposal_ids"] == 607
    assert set(source_by_proposal) == set(queue_by_proposal)
    assert {row["image_id"] for row in source_rows} == image_ids
    assert {row["image_id"] for row in queue_rows} == image_ids
    assert {row["source_arm"] for row in source_rows} == set(ARMS)

    owners: list[dict] = []
    groups: list[dict] = []
    unresolved_proposals: list[dict] = []
    non_owner_proposals: list[dict] = []
    assigned: list[str] = []
    decision_by_image = {row["image_id"]: row for row in decisions}

    for decision in decisions:
        image_id = decision["image_id"]
        for owner in decision["owners"]:
            proposal_ids = owner["proposal_ids"]
            assigned.extend(proposal_ids)
            owners.append(
                {
                    "key": f"{image_id}:{owner['owner_id']}",
                    "image_id": image_id,
                    "owner_id": owner["owner_id"],
                    "proposal_ids": proposal_ids,
                    "extent_or_class_caveats": owner["extent_or_class_caveats"],
                    "presence": {
                        arm: any(source_by_proposal[p]["source_arm"] == arm for p in proposal_ids)
                        for arm in ARMS
                    },
                }
            )
        for group in decision["group_coverage"]:
            proposal_ids = group["proposal_ids"]
            assigned.extend(proposal_ids)
            groups.append(
                {
                    "key": f"{image_id}:{group['group_id']}",
                    "image_id": image_id,
                    "group_id": group["group_id"],
                    "proposal_ids": proposal_ids,
                    "extent_or_class_caveats": group["extent_or_class_caveats"],
                    "presence": {
                        arm: any(source_by_proposal[p]["source_arm"] == arm for p in proposal_ids)
                        for arm in ARMS
                    },
                }
            )
        for unresolved in decision["unresolved"]:
            for proposal_id in unresolved["proposal_ids"]:
                assigned.append(proposal_id)
                unresolved_proposals.append(
                    {
                        "image_id": image_id,
                        "proposal_id": proposal_id,
                        "source_arm": source_by_proposal[proposal_id]["source_arm"],
                        "axes": unresolved["axes"],
                    }
                )
        for item in decision["non_owner_evidence"]:
            for proposal_id in item["proposal_ids"]:
                assigned.append(proposal_id)
                non_owner_proposals.append(
                    {
                        "image_id": image_id,
                        "proposal_id": proposal_id,
                        "source_arm": source_by_proposal[proposal_id]["source_arm"],
                    }
                )

    assert len(assigned) == len(set(assigned)) == 607
    assert set(assigned) == set(source_by_proposal)
    assert len({row["key"] for row in owners}) == len(owners) == 198
    assert len({row["key"] for row in groups}) == len(groups) == 13

    # GT50 rows use exactly the same 32 image IDs; no 896-panel aggregate leaks in.
    gt_rows: dict[str, dict[int, dict]] = {}
    for arm in ARMS:
        all_rows = jsonl(FILES[f"gt50_{arm}"])
        assert len(all_rows) == len({row["image_id"] for row in all_rows}) == 896
        selected = {row["image_id"]: row for row in all_rows if row["image_id"] in image_ids}
        assert set(selected) == image_ids
        gt_rows[arm] = selected

    physical = summarize_clusters(owners)
    caveat_free_owners = [row for row in owners if not row["extent_or_class_caveats"]]
    caveat_free = summarize_clusters(caveat_free_owners)
    group_summary = summarize_clusters(groups)

    source_correspondence_counts = Counter()
    for source in source_rows:
        row = gt_rows.get(source["source_arm"], {}).get(source["image_id"])
        index = source.get("source_prediction_index")
        if row is None or not isinstance(index, int) or index < 0 or index >= len(row["parsed"]["pred"]):
            source_correspondence_counts["unidentifiable"] += 1
        elif index in {match["pred_index"] for match in row["score"]["50"]["matches"]}:
            source_correspondence_counts["has_GT50_matched_source_prediction"] += 1
        else:
            source_correspondence_counts["supported_but_GT50_unmatched"] += 1

    owner_by_key = {row["key"]: row for row in owners}
    gt50_stratification = {}
    for target in ("A", "B"):
        comparison = physical["comparisons"][f"N16-anchor->{target}"]
        lost = [
            classify_cluster_gt50_source(owner_by_key[key], "N16-anchor", source_by_proposal, gt_rows)
            for key in comparison["lost"]
        ]
        gained = [
            classify_cluster_gt50_source(owner_by_key[key], target, source_by_proposal, gt_rows)
            for key in comparison["gained"]
        ]
        lost_summary = stratify_records(lost)
        gained_summary = stratify_records(gained)
        caveat_free_lost = stratify_records(
            [record for record in lost if not record["has_any_extent_or_class_caveat"]]
        )
        caveat_free_gained = stratify_records(
            [record for record in gained if not record["has_any_extent_or_class_caveat"]]
        )
        gt50_stratification[f"N16-anchor->{target}"] = {
            "lost_clusters_classified_on_N16_source": lost_summary,
            "gained_clusters_classified_on_destination_source": gained_summary,
            "cluster_net_by_class": {
                name: gained_summary["class_counts"][name] - lost_summary["class_counts"][name]
                for name in GT50_CLASSES
            },
            "caveat_free_only": {
                "lost_clusters_classified_on_N16_source": caveat_free_lost,
                "gained_clusters_classified_on_destination_source": caveat_free_gained,
                "cluster_net_by_class": {
                    name: caveat_free_gained["class_counts"][name]
                    - caveat_free_lost["class_counts"][name]
                    for name in GT50_CLASSES
                },
            },
        }

    # Independent result must exactly replay the frozen comparison headline.
    assert physical["per_arm_present"] == {
        arm: reference["per_arm"][arm]["atomic_physical_owners_present"] for arm in ARMS
    }
    assert group_summary["per_arm_present"] == {
        arm: reference["per_arm"][arm]["group_coverage_present_separate"] for arm in ARMS
    }
    for source, target in PAIRS:
        key = f"{source}->{target}"
        ours = physical["comparisons"][key]
        theirs = reference["comparisons_atomic_owners_only"][key]
        assert ours["gained"] == theirs["gained"]
        assert ours["lost"] == theirs["lost"]
        assert ours["retained"] == theirs["retained"]

    unresolved_by_arm = Counter(row["source_arm"] for row in unresolved_proposals)
    unresolved_by_image = Counter(row["image_id"] for row in unresolved_proposals)
    uncertainty_net_bounds = {}
    for source, target in PAIRS:
        key = f"{source}->{target}"
        observed = physical["comparisons"][key]["counts"]["net"]
        uncertainty_net_bounds[key] = {
            "observed_net": observed,
            "conservative_per_proposal_distinct_owner_min": observed - unresolved_by_arm[source],
            "conservative_per_proposal_distinct_owner_max": observed + unresolved_by_arm[target],
            "source_unresolved_proposals": unresolved_by_arm[source],
            "target_unresolved_proposals": unresolved_by_arm[target],
        }

    owner_by_image = {
        image_id: [row for row in owners if row["image_id"] == image_id]
        for image_id in sorted(image_ids)
    }
    group_by_image = {
        image_id: [row for row in groups if row["image_id"] == image_id]
        for image_id in sorted(image_ids)
    }
    per_image = []
    for image_id in sorted(image_ids):
        image_owners = owner_by_image[image_id]
        image_caveat_free = [row for row in image_owners if not row["extent_or_class_caveats"]]
        owner_sets = {
            arm: {row["key"] for row in image_owners if row["presence"][arm]}
            for arm in ARMS
        }
        caveat_sets = {
            arm: {row["key"] for row in image_caveat_free if row["presence"][arm]}
            for arm in ARMS
        }
        per_image.append(
            {
                "image_id": image_id,
                "atomic_present": {arm: len(owner_sets[arm]) for arm in ARMS},
                "atomic_caveat_free_present": {arm: len(caveat_sets[arm]) for arm in ARMS},
                "physical_comparisons": {
                    f"{source}->{target}": compare_sets(owner_sets, source, target)["counts"]
                    for source, target in PAIRS
                },
                "caveat_free_comparisons": {
                    f"{source}->{target}": compare_sets(caveat_sets, source, target)["counts"]
                    for source, target in PAIRS
                },
                "group_coverage_present_separate": {
                    arm: sum(row["presence"][arm] for row in group_by_image[image_id])
                    for arm in ARMS
                },
                "unresolved_proposals": {
                    arm: sum(
                        row["image_id"] == image_id and row["source_arm"] == arm
                        for row in unresolved_proposals
                    )
                    for arm in ARMS
                },
                "gt50": {
                    arm: {
                        key: gt_rows[arm][image_id]["score"]["50"][key]
                        for key in ("tp", "fp", "fn", "f1")
                    }
                    for arm in ARMS
                },
            }
        )

    result = {
        "schema": "label_vs_compilation.physical32_independent_audit.v2",
        "status": "candidate_exact_cpu_join_with_gt50_source_stratification_root_acceptance_pending",
        "source_bindings": {
            **{name: binding(path) for name, path in FILES.items()},
            "decisions": [binding(path) for path in DECISION_FILES],
        },
        "denominators": {
            "images": len(image_ids),
            "queue_proposals": len(queue_rows),
            "source_map_rows": len(source_rows),
            "decision_assigned_proposals": len(assigned),
            "atomic_owner_clusters": len(owners),
            "atomic_owner_clusters_with_any_caveat": len(owners) - len(caveat_free_owners),
            "atomic_owner_clusters_without_caveat": len(caveat_free_owners),
            "dense_group_clusters_separate": len(groups),
            "unresolved_proposals": len(unresolved_proposals),
            "non_owner_evidence_proposals": len(non_owner_proposals),
        },
        "gt50_exact32": {
            arm: aggregate_gt50(list(gt_rows[arm].values())) for arm in ARMS
        },
        "source_prediction_gt50_correspondence": {
            "method": (
                "Use source_map.source_prediction_index directly against the bound natural row's "
                "score.50.matches[].pred_index; no box rematching."
            ),
            "all_607_source_prediction_counts": {
                name: source_correspondence_counts[name] for name in GT50_CLASSES
            },
            "cluster_rule": (
                "has_GT50_matched_source_prediction if any exact participating prediction for the bound arm "
                "is matched; supported_but_GT50_unmatched if all exact correspondences exist and none is matched; "
                "otherwise unidentifiable."
            ),
            "n16_loss_and_destination_gain_stratification": gt50_stratification,
        },
        "physical_atomic": compact_summary(physical),
        "descriptive_sensitivity_excluding_any_caveat_cluster": compact_summary(caveat_free),
        "dense_group_coverage_separate": compact_summary(group_summary),
        "uncertainty": {
            "policy": "Unresolved proposals remain neutral and are not atomic owners, losses, or gains.",
            "proposal_counts_by_arm": {arm: unresolved_by_arm[arm] for arm in ARMS},
            "proposal_counts_by_image": dict(sorted(unresolved_by_image.items())),
            "atomic_total_upper_bound_if_every_unresolved_proposal_is_distinct": {
                arm: physical["per_arm_present"][arm] + unresolved_by_arm[arm] for arm in ARMS
            },
            "paired_net_bounds_if_every_unresolved_proposal_can_be_a_distinct_arm_only_owner": uncertainty_net_bounds,
            "boundary": (
                "Conservative proposal-count bound only; unresolved proposals may alias one another or existing owners. "
                "It is not an adjudication or exhaustive-recall interval."
            ),
        },
        "per_image": per_image,
        "verification": {
            "comparison_sha256_expected": "b4da082ae70e63cdc0e6001b26e937966993bd14b006c8f6ea384ec27c21cbaf",
            "comparison_sha256_observed": binding(FILES["comparison"])["sha256"],
            "all_32_ids_match_manifest_queue_source_and_each_gt_arm": True,
            "all_607_proposals_assigned_exactly_once": True,
            "all_607_source_prediction_indices_have_exact_bound_row_correspondence": (
                source_correspondence_counts["unidentifiable"] == 0
            ),
            "independent_headline_equals_frozen_comparison": True,
        },
        "claim_boundary": [
            "Physical counts cover the finite source-blind union of generated proposals on exactly 32 images, not exhaustive image recall.",
            "Dense groups are reported separately and never added to atomic owner counts.",
            "Caveat-free sensitivity is descriptive exclusion of every atomic cluster carrying any class or extent caveat; it is not a new benchmark or gate.",
            "supported_but_GT50_unmatched is only a stored metric classification; class, extent, threshold, or assignment can explain it, so it is never called truly unlabeled.",
            "Cluster match strata do not exactly decompose aggregate TP changes because retained physical clusters and multiple predictions can change GT assignment.",
            "GT-unmatched does not necessarily mean unannotated, and these results do not identify missing-label causality.",
        ],
    }
    assert result["verification"]["comparison_sha256_observed"] == result["verification"]["comparison_sha256_expected"]

    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)


if __name__ == "__main__":
    main()
