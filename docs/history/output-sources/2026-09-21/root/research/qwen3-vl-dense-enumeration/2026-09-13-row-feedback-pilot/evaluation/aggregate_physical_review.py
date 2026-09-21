"""Aggregate a completed dense-8 physical review after source-map unblinding.

The review record stays source blind: it contains proposal groups, reviewer
decisions, and image-local owner IDs only.  This consumer verifies that record
against the public queue before joining the sealed source map to make per-arm
counts.  Its counts cover reviewed proposals only; they do not estimate scene
recall and never turn an unmatched proposal into a negative.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

WORKTREE = Path("/data/CoordExp/.worktrees/row-feedback-pilot-20260913")
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

from probes.row_feedback.evaluation import binding, read, require, validate_selection  # noqa: E402


QUEUE_SCHEMA = "row_feedback.dense8_blind_review_queue.v1"
SOURCE_MAP_SCHEMA = "row_feedback.dense8_blind_review_source_map.v1"
REVIEW_SCHEMA = "row_feedback.dense8_physical_review_record.v1"
REPORT_SCHEMA = "row_feedback.dense8_physical_review_aggregate.v1"
FINISHED_STATUS = "completed_source_blind_physical_review"
DECISIONS = frozenset(("clearly_visible_plausible", "uncertain", "invalid"))


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _publish_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode() + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _read_bound(reference: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    require(isinstance(reference, Mapping), f"{label} binding")
    path = Path(str(reference.get("path", ""))).resolve()
    require(path.is_file() and binding(path) == dict(reference), f"{label} binding drift")
    value = read(path)
    require(isinstance(value, dict), f"{label} JSON object")
    return value


def _no_arm_leak(value: Any) -> None:
    if isinstance(value, Mapping):
        require(not ({"arm", "source_arm", "source_prediction_index", "source_map"} & set(value)),
                "source arm leaked into blind review record")
        for child in value.values():
            _no_arm_leak(child)
    elif isinstance(value, list):
        for child in value:
            _no_arm_leak(child)


def _proposal_key(proposal: Mapping[str, Any]) -> tuple[str, tuple[float, float, float, float]]:
    description = proposal.get("description")
    bbox = proposal.get("bbox")
    require(isinstance(description, str) and description, "proposal description")
    require(proposal.get("bbox_format") == "xyxy_native_pixels", "proposal bbox format")
    require(isinstance(bbox, list) and len(bbox) == 4
            and all(type(item) in (int, float) and not isinstance(item, bool) for item in bbox),
            "proposal bbox")
    return description, tuple(float(item) for item in bbox)


def _queue_index(queue: Mapping[str, Any], *, selection: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[int, list[str]]]:
    require(queue.get("schema") == QUEUE_SCHEMA and queue.get("status") == "ready_for_physical_review",
            "blind queue status")
    expected_ids = [int(value) for value in selection["dense_review_ids"]]
    rows = queue.get("rows")
    require(queue.get("images") == len(rows) == 8 and len(expected_ids) == 8, "fixed dense8 denominator")
    require([int(row.get("image_id")) for row in rows] == expected_ids, "queue image/order coverage")
    by_id: dict[str, dict[str, Any]] = {}
    image_ids: dict[int, list[str]] = {}
    for row in rows:
        require(row.get("source_blind") is True, "queue must be source blind")
        image_id = int(row["image_id"])
        proposals = row.get("proposals")
        require(isinstance(proposals, list), "queue proposals")
        ids: list[str] = []
        for proposal in proposals:
            proposal_id = proposal.get("proposal_id")
            require(isinstance(proposal_id, str) and len(proposal_id) == 64 and proposal_id not in by_id,
                    "queue proposal identity")
            _proposal_key(proposal)
            by_id[proposal_id] = {"image_id": image_id, **dict(proposal)}
            ids.append(proposal_id)
        image_ids[image_id] = ids
    return by_id, image_ids


def _review_groups(review: Mapping[str, Any], *, queue: Mapping[str, Any], image_ids: Mapping[int, Sequence[str]],
                   proposals: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    require(review.get("schema") == REVIEW_SCHEMA and review.get("status") == FINISHED_STATUS
            and review.get("demo") is False, "review must be finished non-demo")
    _no_arm_leak(review)
    require(review.get("queue") == binding(queue["_path"]), "review queue binding")
    require(set(review.get("decision_values", DECISIONS)) == DECISIONS, "review decision vocabulary")
    rows = review.get("rows")
    require(isinstance(rows, list) and len(rows) == 8, "review dense8 row count")
    queue_rows = {int(row["image_id"]): row for row in queue["rows"]}
    require([int(row.get("image_id")) for row in rows] == [int(row["image_id"]) for row in queue["rows"]],
            "review image/order coverage")
    per_proposal: dict[str, dict[str, Any]] = {}
    groups_by_image: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        image_id = int(row["image_id"])
        require(row.get("review_id") == queue_rows[image_id].get("review_id"), "review identity")
        require(row.get("exhaustive_visible_census_performed") is False, "review cannot claim exhaustive census")
        groups = row.get("groups")
        require(isinstance(groups, list), "review groups")
        expected_group_ids: dict[tuple[str, tuple[float, float, float, float]], set[str]] = defaultdict(set)
        for proposal_id in image_ids[image_id]:
            expected_group_ids[_proposal_key(proposals[proposal_id])].add(proposal_id)
        observed_group_ids: dict[tuple[str, tuple[float, float, float, float]], set[str]] = {}
        normalized: list[dict[str, Any]] = []
        for group in groups:
            public_id = group.get("public_id")
            description, bbox = _proposal_key({"description": group.get("description"), "bbox": group.get("bbox_xyxy_native_pixels"),
                                               "bbox_format": "xyxy_native_pixels"})
            proposal_ids = group.get("proposal_ids")
            decision, owner = group.get("visibility_decision"), group.get("physical_owner_id")
            require(isinstance(public_id, str) and public_id and isinstance(proposal_ids, list) and proposal_ids,
                    "review display group")
            require(decision in DECISIONS, "unknown review decision")
            if decision == "clearly_visible_plausible":
                require(isinstance(owner, str) and owner, "plausible group requires image-local owner")
            else:
                require(owner is None, "uncertain/invalid owner must remain neutral")
            key = description, bbox
            require(key not in observed_group_ids, "duplicate review display group")
            observed_group_ids[key] = set(proposal_ids)
            normalized.append({"public_id": public_id, "description": description, "bbox": list(bbox),
                               "proposal_ids": list(proposal_ids), "decision": decision, "owner": owner})
            for proposal_id in proposal_ids:
                require(proposal_id not in per_proposal and proposal_id in proposals
                        and proposals[proposal_id]["image_id"] == image_id, "unknown/duplicate review proposal")
                require(_proposal_key(proposals[proposal_id]) == key, "review literal class/bbox differs from queue")
                per_proposal[proposal_id] = {"image_id": image_id, "decision": decision, "owner": owner,
                                             "public_id": public_id, "description": description, "bbox": list(bbox)}
        require(observed_group_ids == expected_group_ids, "review display groups/proposal coverage differs from queue")
        groups_by_image[image_id] = normalized
    require(set(per_proposal) == set(proposals), "review proposal coverage")
    return per_proposal, groups_by_image


def _source_index(source_map: Mapping[str, Any], *, selection: Mapping[str, Any], proposals: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    require(source_map.get("schema") == SOURCE_MAP_SCHEMA and source_map.get("status") == "sealed_not_for_reviewer",
            "sealed source-map status")
    rows = source_map.get("rows")
    require(isinstance(rows, list), "source-map rows")
    result: dict[str, dict[str, Any]] = {}
    prediction_indices: set[tuple[int, str, int]] = set()
    for row in rows:
        proposal_id, arm = row.get("proposal_id"), row.get("source_arm")
        require(isinstance(proposal_id, str) and proposal_id in proposals and proposal_id not in result,
                "source-map proposal coverage")
        require(arm in ("S", "F") and type(row.get("source_prediction_index")) is int
                and row.get("source_prediction_index") >= 0, "source-map arm/order")
        require(int(row.get("image_id")) == proposals[proposal_id]["image_id"], "source-map image identity")
        order = int(row["image_id"]), str(arm), int(row["source_prediction_index"])
        require(order not in prediction_indices, "source-map duplicate per-arm prediction index")
        prediction_indices.add(order)
        result[proposal_id] = dict(row)
    require(set(result) == set(proposals), "source-map exact proposal coverage")
    return result


def aggregate(*, selection_path: str | Path, manifest_path: str | Path, queue_path: str | Path,
              source_map_path: str | Path, review_path: str | Path, output: str | Path) -> dict[str, Any]:
    """Validate the blind artifacts, then unblind per-arm reviewed-proposal counts."""
    selection_path, manifest_path, queue_path, source_map_path, review_path, output = (Path(value).resolve() for value in
        (selection_path, manifest_path, queue_path, source_map_path, review_path, output))
    require(not output.exists(), "aggregate output already exists")
    selection = read(selection_path)
    validate_selection(selection, verify_files=True)
    manifest = read(manifest_path)
    require(manifest.get("schema") == "row_feedback.dense8_blind_review_manifest.v1"
            and manifest.get("status") == "ready_for_render", "prepare-review manifest status")
    queue = read(queue_path)
    queue["_path"] = str(queue_path)
    source_map = read(source_map_path)
    review = read(review_path)
    expected_selection = binding(selection_path)
    require(queue.get("selection") == expected_selection and source_map.get("selection") == expected_selection,
            "queue/source-map selection binding")
    require(manifest.get("queue") == binding(queue_path)
            and manifest.get("source_map") == binding(source_map_path),
            "prepare-review manifest queue/source-map binding")
    proposals, image_ids = _queue_index(queue, selection=selection)
    review_by_proposal, groups_by_image = _review_groups(review, queue=queue, image_ids=image_ids, proposals=proposals)
    sources = _source_index(source_map, selection=selection, proposals=proposals)

    per_image: list[dict[str, Any]] = []
    union: list[dict[str, Any]] = []
    for image_id in [int(value) for value in selection["dense_review_ids"]]:
        arm_rows: dict[str, list[dict[str, Any]]] = {"S": [], "F": []}
        for proposal_id in image_ids[image_id]:
            arm_rows[sources[proposal_id]["source_arm"]].append({
                "proposal_id": proposal_id, "source_prediction_index": sources[proposal_id]["source_prediction_index"],
                **review_by_proposal[proposal_id],
            })
        arm_report: dict[str, Any] = {}
        owners: dict[str, set[str]] = defaultdict(set)
        for arm in ("S", "F"):
            rows = sorted(arm_rows[arm], key=lambda row: (row["source_prediction_index"], row["proposal_id"]))
            plausible = [row for row in rows if row["decision"] == "clearly_visible_plausible"]
            owner_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in plausible:
                owner_rows[row["owner"]].append(row)
                owners[row["owner"]].add(arm)
            arm_report[arm] = {
                "proposed_exact_proposals": len(rows),
                "review_display_groups_touched": len({row["public_id"] for row in rows}),
                "plausible_proposals": len(plausible),
                "uncertain_proposals": sum(row["decision"] == "uncertain" for row in rows),
                "invalid_proposals": sum(row["decision"] == "invalid" for row in rows),
                "reviewed_physical_owner_count": len(owner_rows),
                "same_owner_additional_proposals": sum(len(items) - 1 for items in owner_rows.values()),
                "within_arm_recurrence_order": [
                    {"physical_owner_id": owner, "source_prediction_indices": [item["source_prediction_index"] for item in items],
                     "proposal_ids": [item["proposal_id"] for item in items]}
                    for owner, items in sorted(owner_rows.items()) if len(items) > 1
                ],
            }
        owner_changes = {"retained": [], "gained": [], "lost": []}
        for owner, arms in sorted(owners.items()):
            owner_changes["retained" if arms == {"S", "F"} else "gained" if arms == {"F"} else "lost"].append(owner)
        union.extend({"image_id": image_id, "physical_owner_id": owner, "arms": sorted(arms)}
                     for owner, arms in sorted(owners.items()))
        per_image.append({"image_id": image_id, "display_groups": len(groups_by_image[image_id]),
                          "exact_proposal_ids": len(image_ids[image_id]), "per_arm": arm_report,
                          "reviewed_owner_changes_F_vs_S": owner_changes})

    per_arm_summary = {
        arm: {
            field: sum(int(row["per_arm"][arm][field]) for row in per_image)
            for field in ("proposed_exact_proposals", "review_display_groups_touched", "plausible_proposals",
                          "uncertain_proposals", "invalid_proposals", "reviewed_physical_owner_count",
                          "same_owner_additional_proposals")
        }
        for arm in ("S", "F")
    }
    owner_change_summary = {
        category: sum(len(row["reviewed_owner_changes_F_vs_S"][category]) for row in per_image)
        for category in ("retained", "gained", "lost")
    }

    result = {
        "schema": REPORT_SCHEMA, "status": "candidate_awaiting_root_unblinding_acceptance",
        "claim_boundary": "Counts cover only the fixed dense8 queue's reviewed proposals and image-local physical-owner assignments. They do not estimate exhaustive scene recall; uncertain proposals remain neutral and unmatched proposals are never negatives.",
        "selection": binding(selection_path), "prepare_review_manifest": binding(manifest_path),
        "queue": binding(queue_path), "source_map": binding(source_map_path),
        "review": binding(review_path), "denominators": {"images": 8, "exact_proposal_ids": len(proposals),
                                                            "display_groups": sum(len(groups) for groups in groups_by_image.values())},
        "per_arm_reviewed_proposal_summary": per_arm_summary,
        "reviewed_owner_change_counts_F_vs_S": owner_change_summary,
        "per_image": per_image, "reviewed_physical_owner_union": union,
        "unblinding_scope": "Sealed source-map arms joined only after completed source-blind review; owner IDs are image-local.",
    }
    result["report_payload_sha256"] = digest(result)
    _publish_exclusive(output, result)
    return result


def _fixture(selection_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    selection = read(selection_path)
    validate_selection(selection, verify_files=True)
    ids = [int(value) for value in selection["dense_review_ids"]]
    records = {int(row["image_id"]): row for row in selection["records"]}
    queue_rows, source_rows, review_rows = [], [], []
    for index, image_id in enumerate(ids):
        record = records[image_id]
        proposals = []
        if index == 0:
            raw = [("person", [10, 10, 30, 30], "S", 0), ("person", [10, 10, 30, 30], "F", 0),
                   ("person", [12, 10, 32, 30], "S", 1), ("cup", [40, 10, 60, 30], "F", 1)]
        else:
            raw = [("chair", [10, 10, 30, 30], "S" if index % 2 else "F", 0)]
        groups: dict[tuple[str, tuple[int, int, int, int]], list[str]] = defaultdict(list)
        for ordinal, (description, bbox, arm, source_index) in enumerate(raw):
            proposal_id = hashlib.sha256(f"fixture:{image_id}:{ordinal}".encode()).hexdigest()
            proposals.append({"proposal_id": proposal_id, "description": description, "bbox": bbox,
                              "bbox_format": "xyxy_native_pixels"})
            source_rows.append({"image_id": image_id, "proposal_id": proposal_id, "source_arm": arm,
                                "source_prediction_index": source_index})
            groups[(description, tuple(bbox))].append(proposal_id)
        review_groups = []
        for group_index, ((description, bbox), proposal_ids) in enumerate(groups.items(), start=1):
            uncertain = index == 0 and description == "cup"
            review_groups.append({"public_id": f"P{group_index:03d}", "description": description,
                                  "bbox_xyxy_native_pixels": list(bbox), "proposal_ids": proposal_ids,
                                  "visibility_decision": "uncertain" if uncertain else "clearly_visible_plausible",
                                  "physical_owner_id": None if uncertain else "owner-a" if index == 0 else "owner-1"})
        queue_rows.append({"review_id": f"row-feedback-dense:{image_id}", "image_id": image_id,
                           "image_path": record["image_path"], "width": record["width"], "height": record["height"],
                           "source_blind": True, "proposals": proposals})
        review_rows.append({"review_id": f"row-feedback-dense:{image_id}", "image_id": image_id,
                            "groups": review_groups, "image_notes": "", "exhaustive_visible_census_performed": False})
    queue = {"schema": QUEUE_SCHEMA, "status": "ready_for_physical_review", "selection": binding(selection_path),
             "images": 8, "rows": queue_rows}
    source = {"schema": SOURCE_MAP_SCHEMA, "status": "sealed_not_for_reviewer", "selection": binding(selection_path), "rows": source_rows}
    review = {"schema": REVIEW_SCHEMA, "status": FINISHED_STATUS, "demo": False, "queue": None,
              "decision_values": sorted(DECISIONS), "rows": review_rows}
    return queue, source, review


def self_test(selection_path: str | Path) -> dict[str, Any]:
    """Exercise source-blind, exact-coverage, alias, and neutral-uncertainty contracts."""
    selection_path = Path(selection_path).resolve()
    queue, source, review = _fixture(selection_path)
    with tempfile.TemporaryDirectory(prefix="row-feedback-physical-review-") as directory:
        root = Path(directory)
        queue_path, source_path, review_path = root / "queue.json", root / "source-map.json", root / "review.json"
        _publish_exclusive(queue_path, queue)
        _publish_exclusive(source_path, source)
        manifest_path = root / "manifest.json"
        _publish_exclusive(manifest_path, {"schema": "row_feedback.dense8_blind_review_manifest.v1",
                                           "status": "ready_for_render", "queue": binding(queue_path),
                                           "source_map": binding(source_path)})
        review["queue"] = binding(queue_path)
        _publish_exclusive(review_path, review)
        report = aggregate(selection_path=selection_path, manifest_path=manifest_path, queue_path=queue_path,
                           source_map_path=source_path, review_path=review_path, output=root / "report.json")
        first = report["per_image"][0]
        require(first["reviewed_owner_changes_F_vs_S"]["retained"] == ["owner-a"], "fixture cross-arm owner")
        require(first["per_arm"]["S"]["same_owner_additional_proposals"] == 1, "fixture within-arm alias")
        require(first["per_arm"]["F"]["uncertain_proposals"] == 1, "fixture uncertainty neutral")
        failures: dict[str, str] = {}
        for label, mutate in {
            "missing": lambda value: value["rows"][0]["groups"][0]["proposal_ids"].pop(),
            "extra": lambda value: value["rows"][0]["groups"][0]["proposal_ids"].append("f" * 64),
            "mutated": lambda value: value["rows"][0]["groups"][0].update({"description": "dog"}),
            "arm_leak": lambda value: value["rows"][0].update({"arm": "S"}),
        }.items():
            candidate = copy.deepcopy(review)
            mutate(candidate)
            path = root / f"{label}.json"
            _publish_exclusive(path, candidate)
            try:
                aggregate(selection_path=selection_path, queue_path=queue_path, source_map_path=source_path,
                          manifest_path=manifest_path, review_path=path, output=root / f"{label}-report.json")
            except ValueError as exc:
                failures[label] = str(exc)
        stale_map = copy.deepcopy(source)
        for row in stale_map["rows"]:
            if row["image_id"] == int(read(selection_path)["dense_review_ids"][0]):
                row["source_arm"] = "F" if row["source_arm"] == "S" else "S"
        stale_map_path = root / "stale-source-map.json"
        _publish_exclusive(stale_map_path, stale_map)
        try:
            aggregate(selection_path=selection_path, manifest_path=manifest_path, queue_path=queue_path,
                      source_map_path=stale_map_path, review_path=review_path, output=root / "stale-map-report.json")
        except ValueError as exc:
            failures["stale_source_map"] = str(exc)
        require(set(failures) == {"missing", "extra", "mutated", "arm_leak", "stale_source_map"},
                "fixture falsifiers")
        return {"status": "passed", "falsifiers": failures, "source_blind_review_verified": True,
                "fixture_first_image": {"retained": ["owner-a"], "S_same_owner_additional": 1,
                                        "F_uncertain_proposals": 1}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--queue", type=Path)
    parser.add_argument("--source-map", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = self_test(args.selection)
    else:
        require(args.manifest is not None and args.queue is not None and args.source_map is not None
                and args.review is not None and args.output is not None,
                "aggregate requires manifest, queue, source-map, review, and output")
        result = aggregate(selection_path=args.selection, manifest_path=args.manifest, queue_path=args.queue,
                           source_map_path=args.source_map, review_path=args.review, output=args.output)
        result = {"output": binding(args.output), "report_payload_sha256": result["report_payload_sha256"]}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
