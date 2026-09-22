"""CPU-only single-image blind32 renderer, batch freezer, and exact-ID join."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw

from probes.native_owner_scale import evaluation as native
from src.eval.detection_categories import COCO_80_CLASS_NAMES, normalize_coco_category_name
from src.vis.rendering import _font, _load_image, _save


BLIND_IMAGES = 32
BATCHES = 8
IMAGES_PER_BATCH = 4
ARMS = ("N16-anchor", "A", "B")
DECISION_CATEGORIES = ("owners", "group_coverage", "unresolved", "non_owner_evidence")
UNRESOLVED_AXES = {"entity", "class", "extent"}
NON_OWNER_STATUSES = {"unsupported", "extent_mismatch"}
COCO80 = {normalize_coco_category_name(name) for name in COCO_80_CLASS_NAMES}
SAVE_ATTESTATION = "reviewer_attests_decision_saved_before_next_view"


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return native.read_jsonl(path)


def binding(path: str | Path) -> dict[str, Any]:
    return native.binding(path)


def publish(path: str | Path, value: Any) -> None:
    native.publish(path, value)


def _literal_key(proposal: Mapping[str, Any]) -> str:
    value = {"description": proposal["description"], "bbox": proposal["bbox"],
             "bbox_format": proposal.get("bbox_format", "xyxy")}
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _validate_bbox(box: Any, width: int, height: int) -> list[float]:
    require(isinstance(box, list) and len(box) == 4, "proposal bbox shape")
    values = [float(value) for value in box]
    require(all(math.isfinite(value) for value in values), "proposal bbox finite")
    require(0 <= values[0] <= values[2] <= width and 0 <= values[1] <= values[3] <= height,
            "proposal bbox outside literal native frame")
    return values


def validate_queue(queue_path: str | Path, *, expected_images: int = BLIND_IMAGES) -> list[dict[str, Any]]:
    queue = read_jsonl(queue_path)
    require(len(queue) == expected_images, "blind image denominator")
    image_ids = [int(item["image_id"]) for item in queue]
    require(len(set(image_ids)) == len(image_ids), "duplicate blind image")
    proposal_ids: set[tuple[int, str]] = set()
    for item in queue:
        require(item.get("source_blind") is True, "queue is not source blind")
        require(isinstance(item.get("proposals"), list) and item["proposals"], "empty blind proposals")
        width, height = int(item["image_width"]), int(item["image_height"])
        require(width > 0 and height > 0, "source dimensions")
        path = Path(str(item.get("literal_source_canvas_path", item.get("image_path", ""))))
        require(path.is_absolute() and path.is_file(), "literal source frame missing")
        with Image.open(path) as source:
            require(source.size == (width, height), "literal source dimensions changed")
        for proposal in item["proposals"]:
            require(set(proposal).isdisjoint({"source_arm", "source_prediction_index", "model_outcome", "gt"}),
                    "arm/model/GT leak in anonymous proposal")
            proposal_id = str(proposal.get("proposal_id", ""))
            key = (int(item["image_id"]), proposal_id)
            require(proposal_id and key not in proposal_ids, "duplicate proposal ID")
            proposal_ids.add(key)
            description = normalize_coco_category_name(proposal.get("description"))
            require(description in COCO80, "proposal outside canonical COCO80")
            _validate_bbox(proposal.get("bbox"), width, height)
            require(proposal.get("bbox_format", "xyxy") == "xyxy", "proposal bbox format")
    return queue


def alias_groups(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Alias only byte-equivalent literal description/bbox transport proposals."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for proposal in item["proposals"]:
        grouped[_literal_key(proposal)].append(proposal)
    result = []
    for literal, rows in sorted(grouped.items(), key=lambda pair: pair[0]):
        proposal = rows[0]
        alias_id = hashlib.sha256(f"{item['image_id']}:{literal}".encode()).hexdigest()
        result.append({"alias_id": alias_id, "proposal_ids": sorted(str(row["proposal_id"]) for row in rows),
                       "description": proposal["description"],
                       "canonical_category": normalize_coco_category_name(proposal["description"]),
                       "bbox": list(proposal["bbox"]), "bbox_format": "xyxy"})
    require(sum(len(row["proposal_ids"]) for row in result) == len(item["proposals"]),
            "literal alias coverage")
    return result


def render_item(item: Mapping[str, Any], output: str | Path) -> dict[str, Any]:
    """Render native-resolution overlays with exactly one literal candidate each."""
    output = Path(output)
    require(not output.exists(), f"image render output occupied: {output}")
    output.mkdir(parents=True)
    source_path = Path(str(item.get("literal_source_canvas_path", item.get("image_path", ""))))
    source = _load_image(source_path)
    width, height = int(item["image_width"]), int(item["image_height"])
    require(source.size == (width, height), "render source dimensions changed")
    rendered = []
    for ordinal, alias in enumerate(alias_groups(item), 1):
        image = source.copy()
        draw = ImageDraw.Draw(image)
        box = _validate_bbox(alias["bbox"], width, height)
        draw.rectangle(tuple(box), outline=(255, 0, 255), width=3)
        label = f"{ordinal}: {alias['canonical_category']} [{alias['alias_id'][:10]}]"
        font = _font("regular", 14)
        x, y = int(box[0]), max(0, int(box[1]) - 20)
        draw.rectangle((x, y, min(width, x + max(36, 8 * len(label))), min(height, y + 19)), fill="white")
        draw.text((x + 1, y), label, fill=(160, 0, 160), font=font)
        path = output / f"candidate-{ordinal:04d}-{alias['alias_id'][:12]}.png"
        _save(image, path)
        with Image.open(path) as check:
            require(check.size == (width, height), "candidate overlay resized")
            check.verify()
        rendered.append({**alias, "overlay": binding(path), "canvas_size": [width, height],
                         "overlay_candidates": 1})
    return {"image_id": int(item["image_id"]), "review_id": item["review_id"],
            "original_image": binding(source_path), "source_dimensions": [width, height],
            "candidates": rendered}


def _assert_reviewer_safe(value: Any) -> None:
    banned_keys = {"source_arm", "source_prediction_index", "source_map", "model_outcomes",
                   "gt", "ground_truth", "candidate_score"}
    banned_values = set(ARMS)
    if isinstance(value, Mapping):
        require(not banned_keys.intersection(value), "reviewer batch leaks source/model/GT fields")
        for child in value.values():
            _assert_reviewer_safe(child)
    elif isinstance(value, list):
        for child in value:
            _assert_reviewer_safe(child)
    elif isinstance(value, str):
        require(value not in banned_values, "reviewer batch leaks arm label")


def prepare_review(*, queue_path: str | Path, output: str | Path,
                   expected_images: int = BLIND_IMAGES, batch_size: int = IMAGES_PER_BATCH) -> dict[str, Any]:
    """Render one-image cards and freeze source-blind reviewer batches."""
    queue_path, output = Path(queue_path), Path(output)
    require(not output.exists(), f"blind review output occupied: {output}")
    require(expected_images == BLIND_IMAGES and batch_size == IMAGES_PER_BATCH,
            "production blind32 must remain eight batches of four")
    queue = validate_queue(queue_path, expected_images=expected_images)
    output.mkdir(parents=True)
    rendered = [render_item(item, output / "render" / f"{int(item['image_id']):012d}") for item in queue]
    batches = []
    for start in range(0, len(rendered), batch_size):
        batch_number = start // batch_size + 1
        batch = {
            "schema": "owner_successor_scale.blind_review.batch.v1",
            "status": "frozen_source_blind_unreviewed",
            "batch_id": f"blind32-batch-{batch_number:02d}",
            "images": rendered[start:start + batch_size],
            "review_contract": {
                "one_image_per_view": True, "view_detail": "original",
                "save_each_image_decision_before_next_view": True,
                "physical_presence": "class-agnostic within COCO80-compatible objects; retain class/extent caveats; not strict box TP",
                "atomic": "clearly separable physical instances; one owner ID may bind multiple proposal aliases",
                "groups": "report coherent dense group coverage separately; never count it as atomic owner or with children",
                "uncertainty": "entity, class and extent are separate; unresolved remains neutral",
                "negative_boundary": "no hallucination or negative label from missing GT",
            },
            "decision_schema": {
                "required": ["image_id", "review_id", "reviewer", "saved_before_next_view_attestation", "viewed", "owners",
                             "group_coverage", "unresolved", "non_owner_evidence"],
                "saved_before_next_view_attestation": SAVE_ATTESTATION,
                "owners": ["owner_id", "proposal_ids", "extent_or_class_caveats"],
                "group_coverage": ["group_id", "proposal_ids", "extent_or_class_caveats"],
                "unresolved": ["proposal_ids", "axes", "image_grounded_reason"],
                "non_owner_evidence_optional": ["proposal_ids", "status=unsupported|extent_mismatch",
                                                "image_grounded_reason"],
            },
        }
        _assert_reviewer_safe(batch)
        path = output / "batches" / f"batch-{batch_number:02d}" / "input.json"
        publish(path, batch)
        batches.append({"batch_id": batch["batch_id"], "input": binding(path),
                        "images": [row["image_id"] for row in batch["images"]]})
    require(len(batches) == BATCHES and all(len(row["images"]) == IMAGES_PER_BATCH for row in batches),
            "blind32 batch partition")
    manifest = {"schema": "owner_successor_scale.blind_review.preparation.v1",
                "status": "rendered_and_batches_frozen_not_reviewed", "queue": binding(queue_path),
                "images": len(rendered), "batches": batches,
                "literal_alias_groups": sum(len(row["candidates"]) for row in rendered),
                "proposal_ids": sum(len(alias["proposal_ids"]) for row in rendered for alias in row["candidates"]),
                "source_map_read": False, "visual_labels": 0,
                "boundary": "One native-resolution source sample per card and one candidate per overlay; no source arm/model/GT fields in reviewer batches."}
    publish(output / "manifest.json", manifest)
    return manifest


def _proposal_universe(queue: Sequence[Mapping[str, Any]]) -> tuple[set[tuple[int, str]], dict[int, list[set[str]]]]:
    keys = {(int(item["image_id"]), str(proposal["proposal_id"]))
            for item in queue for proposal in item["proposals"]}
    aliases = {int(item["image_id"]): [set(row["proposal_ids"]) for row in alias_groups(item)] for item in queue}
    return keys, aliases


def _verified_binding(value: Any, label: str) -> Path:
    require(isinstance(value, Mapping) and set(value) == {"path", "sha256", "size_bytes"},
            f"{label} binding shape")
    path = Path(str(value["path"]))
    require(path.is_file(), f"{label} missing")
    require(binding(path) == dict(value), f"{label} binding changed")
    return path


def _preparation_views(*, preparation_path: str | Path, queue_path: str | Path,
                       queue: Sequence[Mapping[str, Any]]) -> dict[int, set[tuple[str, str]]]:
    """Load exact per-image view bindings from the frozen preparation and batches."""
    preparation = read(preparation_path)
    require(preparation.get("schema") == "owner_successor_scale.blind_review.preparation.v1",
            "preparation schema")
    require(preparation.get("status") == "rendered_and_batches_frozen_not_reviewed",
            "preparation status")
    require(preparation.get("queue") == binding(queue_path), "preparation queue binding mismatch")
    require(preparation.get("images") == len(queue), "preparation image denominator")
    queue_by_id = {int(item["image_id"]): item for item in queue}
    queue_ids = [int(item["image_id"]) for item in queue]
    allowed: dict[int, set[tuple[str, str]]] = {}
    prepared_ids: list[int] = []
    batches = preparation.get("batches")
    require(isinstance(batches, list) and batches, "preparation batches")
    if len(queue) == BLIND_IMAGES:
        require(len(batches) == BATCHES, "blind32 preparation batch denominator")
    for batch_record in batches:
        require(isinstance(batch_record, Mapping), "preparation batch record")
        batch_path = _verified_binding(batch_record.get("input"), "preparation batch")
        batch = read(batch_path)
        require(batch.get("schema") == "owner_successor_scale.blind_review.batch.v1",
                "review batch schema")
        require(batch.get("status") == "frozen_source_blind_unreviewed", "review batch status")
        require(batch.get("batch_id") == batch_record.get("batch_id"), "review batch ID mismatch")
        _assert_reviewer_safe(batch)
        images = batch.get("images")
        require(isinstance(images, list) and images, "empty review batch")
        batch_ids = [int(item["image_id"]) for item in images]
        require(batch_ids == [int(value) for value in batch_record.get("images", [])],
                "preparation batch image mismatch")
        if len(queue) == BLIND_IMAGES:
            require(len(batch_ids) == IMAGES_PER_BATCH, "blind32 batch size")
        prepared_ids.extend(batch_ids)
        for item in images:
            image_id = int(item["image_id"])
            require(image_id in queue_by_id and image_id not in allowed,
                    "unknown or duplicate prepared image")
            queue_item = queue_by_id[image_id]
            require(item.get("review_id") == queue_item["review_id"], "prepared review ID mismatch")
            dimensions = [int(queue_item["image_width"]), int(queue_item["image_height"])]
            require(item.get("source_dimensions") == dimensions, "prepared source dimensions mismatch")
            original = item.get("original_image")
            _verified_binding(original, "prepared original image")
            views = {(str(original["path"]), str(original["sha256"]))}
            candidates = item.get("candidates")
            require(isinstance(candidates, list), "prepared candidates")
            expected_aliases = {row["alias_id"]: row for row in alias_groups(queue_item)}
            require({str(row.get("alias_id")) for row in candidates} == set(expected_aliases),
                    "prepared candidate alias coverage")
            covered: list[str] = []
            for candidate in candidates:
                alias = expected_aliases[str(candidate["alias_id"])]
                require(candidate.get("proposal_ids") == alias["proposal_ids"] and
                        candidate.get("description") == alias["description"] and
                        candidate.get("bbox") == alias["bbox"] and
                        candidate.get("bbox_format") == alias["bbox_format"],
                        "prepared candidate literal mismatch")
                require(candidate.get("canvas_size") == dimensions and
                        candidate.get("overlay_candidates") == 1,
                        "prepared candidate rendering contract")
                overlay = candidate.get("overlay")
                _verified_binding(overlay, "prepared candidate overlay")
                views.add((str(overlay["path"]), str(overlay["sha256"])))
                covered.extend(str(value) for value in candidate["proposal_ids"])
            require(sorted(covered) == sorted(str(row["proposal_id"]) for row in queue_item["proposals"]),
                    "prepared proposal coverage")
            allowed[image_id] = views
    require(prepared_ids == queue_ids, "preparation image order/coverage")
    return allowed


def validate_reviews(queue: Sequence[Mapping[str, Any]], review_paths: Sequence[str | Path], *,
                     preparation_path: str | Path, queue_path: str | Path) -> tuple[list[dict[str, Any]], dict[tuple[int, str], str]]:
    decisions = [row for path in review_paths for row in read_jsonl(path)]
    queue_ids = [int(item["image_id"]) for item in queue]
    require(len(decisions) == len(queue_ids), "review image coverage")
    require([int(row["image_id"]) for row in decisions] == queue_ids, "review image order/duplication")
    keys, aliases = _proposal_universe(queue)
    categories: dict[tuple[int, str], str] = {}
    seen_owner_ids: set[tuple[int, str]] = set()
    seen_group_ids: set[tuple[int, str]] = set()
    by_image = {int(item["image_id"]): item for item in queue}
    allowed_views = _preparation_views(preparation_path=preparation_path, queue_path=queue_path,
                                       queue=queue)
    for decision in decisions:
        image_id = int(decision["image_id"])
        require(decision.get("review_id") == by_image[image_id]["review_id"], "review ID mismatch")
        require(isinstance(decision.get("reviewer"), str) and decision["reviewer"], "missing reviewer")
        require(decision.get("saved_before_next_view_attestation") == SAVE_ATTESTATION,
                "missing saved-before-next-view workflow attestation")
        viewed = decision.get("viewed")
        require(isinstance(viewed, list) and viewed, "no actual viewed evidence")
        for view in viewed:
            require(set(view) >= {"path", "sha256", "detail"} and view["detail"] == "original",
                    "view evidence shape/detail")
            require(binding(view["path"])["sha256"] == view["sha256"], "viewed path/hash changed")
            require((str(view["path"]), str(view["sha256"])) in allowed_views[image_id],
                    "viewed evidence belongs to a different image or preparation")
        for category in DECISION_CATEGORIES:
            require(isinstance(decision.get(category), list), f"missing {category}")
            for entry in decision[category]:
                proposal_ids = [str(value) for value in entry.get("proposal_ids", [])]
                require(proposal_ids and len(proposal_ids) == len(set(proposal_ids)), "empty/duplicate decision proposal IDs")
                for proposal_id in proposal_ids:
                    key = (image_id, proposal_id)
                    require(key in keys and key not in categories, "unknown or multiply assigned proposal ID")
                    categories[key] = category
                if category == "owners":
                    owner = (image_id, str(entry.get("owner_id", "")))
                    require(owner[1] and owner not in seen_owner_ids, "duplicate/missing within-image owner ID")
                    seen_owner_ids.add(owner)
                    require(isinstance(entry.get("extent_or_class_caveats"), list), "owner caveat list")
                elif category == "group_coverage":
                    group = (image_id, str(entry.get("group_id", "")))
                    require(group[1] and group not in seen_group_ids, "duplicate/missing within-image group ID")
                    seen_group_ids.add(group)
                    require(isinstance(entry.get("extent_or_class_caveats"), list), "group caveat list")
                elif category == "unresolved":
                    axes = set(entry.get("axes", []))
                    require(axes and axes <= UNRESOLVED_AXES, "unresolved axes")
                    require(str(entry.get("image_grounded_reason", "")).strip(), "unresolved image-grounded reason")
                else:
                    require(entry.get("status") in NON_OWNER_STATUSES, "non-owner evidence status")
                    require(str(entry.get("image_grounded_reason", "")).strip(), "non-owner image-grounded reason")
        # Literal-identical transport aliases must never be split into multiple decisions.
        for alias in aliases[image_id]:
            locations = {(categories[(image_id, proposal_id)], next(
                index for index, entry in enumerate(decision[categories[(image_id, proposal_id)]])
                if proposal_id in entry["proposal_ids"])) for proposal_id in alias}
            require(len(locations) == 1, "literal alias split/double-count")
    require(set(categories) == keys, "review proposal omission")
    return decisions, categories


def _changes(owners: Sequence[Mapping[str, Any]], left: str, right: str) -> dict[str, Any]:
    left_set = {row["key"] for row in owners if row["presence"][left]}
    right_set = {row["key"] for row in owners if row["presence"][right]}
    return {"gained": sorted(right_set - left_set), "retained": sorted(left_set & right_set),
            "lost": sorted(left_set - right_set),
            "counts": {"gained": len(right_set - left_set), "retained": len(left_set & right_set),
                       "lost": len(left_set - right_set)}}


def compare(*, queue_path: str | Path, preparation_path: str | Path, source_map_path: str | Path,
            review_paths: Sequence[str | Path], output: str | Path,
            expected_images: int = BLIND_IMAGES) -> dict[str, Any]:
    """Join source identities only after exact review decisions are frozen."""
    queue_path, preparation_path = Path(queue_path), Path(preparation_path)
    source_map_path, output = Path(source_map_path), Path(output)
    require(not output.exists(), f"blind comparison output occupied: {output}")
    queue = validate_queue(queue_path, expected_images=expected_images)
    decisions, categories = validate_reviews(queue, review_paths,
                                              preparation_path=preparation_path,
                                              queue_path=queue_path)
    keys, _ = _proposal_universe(queue)
    mapping_value = read(source_map_path)
    mapping = mapping_value.get("rows", [])
    sources: dict[tuple[int, str], set[str]] = defaultdict(set)
    source_rows: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    identities = set()
    for row in mapping:
        key = (int(row["image_id"]), str(row["proposal_id"]))
        require(key in keys and row.get("source_arm") in ARMS, "source map key/arm")
        identity = (key, row["source_arm"], int(row["source_prediction_index"]))
        require(identity not in identities, "duplicate source-map row")
        identities.add(identity); sources[key].add(row["source_arm"]); source_rows[key].append(row)
    require(set(sources) == keys, "source-map proposal omission")
    owners, groups = [], []
    for decision in decisions:
        image_id = int(decision["image_id"])
        for kind, target, id_field in (("owners", owners, "owner_id"),
                                       ("group_coverage", groups, "group_id")):
            for entry in decision[kind]:
                entry_keys = [(image_id, str(proposal_id)) for proposal_id in entry["proposal_ids"]]
                presence = {arm: any(arm in sources[key] for key in entry_keys) for arm in ARMS}
                target.append({"key": f"{image_id}:{entry[id_field]}", "image_id": image_id,
                               id_field: entry[id_field], "proposal_ids": entry["proposal_ids"],
                               "extent_or_class_caveats": entry["extent_or_class_caveats"],
                               "presence": presence})
    comparisons = {f"{left}->{right}": _changes(owners, left, right)
                   for left, right in (("N16-anchor", "A"), ("N16-anchor", "B"), ("A", "B"))}
    per_arm = {}
    for arm in ARMS:
        selected = {key for key in keys if arm in sources[key]}
        per_arm[arm] = {
            "source_rows": sum(row["source_arm"] == arm for rows in source_rows.values() for row in rows),
            "unique_proposal_ids": len(selected),
            "atomic_physical_owners_present": sum(row["presence"][arm] for row in owners),
            "group_coverage_present_separate": sum(row["presence"][arm] for row in groups),
            "proposal_decisions": dict(Counter(categories[key] for key in selected)),
            "unresolved_proposals": sum(categories[key] == "unresolved" for key in selected),
        }
    result = {
        "schema": "owner_successor_scale.blind_review.comparison.v1",
        "status": "exact_id_join_complete_root_acceptance_pending",
        "bindings": {"queue": binding(queue_path), "preparation": binding(preparation_path),
                     "source_map": binding(source_map_path),
                     "reviews": [binding(path) for path in review_paths],
                     "consumer": binding(Path(__file__).resolve())},
        "denominators": {"images": len(queue), "queue_unique_proposal_ids": len(keys),
                         "source_map_rows": len(mapping), "source_map_unique_proposal_ids": len(sources),
                         "review_assigned_proposal_ids": len(categories), "atomic_owner_clusters": len(owners),
                         "group_coverage_clusters_separate": len(groups),
                         "unresolved_proposal_ids": sum(value == "unresolved" for value in categories.values()),
                         "non_owner_evidence_proposal_ids": sum(value == "non_owner_evidence" for value in categories.values())},
        "comparisons_atomic_owners_only": comparisons,
        "per_arm": per_arm, "owners": owners, "group_coverage": groups,
        "boundary": "Physical-owner presence is class-agnostic within COCO80-compatible proposals with caveats; not strict TP or exhaustive recall. Group coverage is never added to atomic owner gains. Unresolved stays neutral and missing GT creates no negative/hallucination label.",
    }
    publish(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--queue", type=Path, required=True); prepare.add_argument("--output", type=Path, required=True)
    join = sub.add_parser("compare")
    join.add_argument("--queue", type=Path, required=True); join.add_argument("--preparation", type=Path, required=True)
    join.add_argument("--source-map", type=Path, required=True)
    join.add_argument("--review", action="append", type=Path, required=True); join.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare": result = prepare_review(queue_path=args.queue, output=args.output)
    else: result = compare(queue_path=args.queue, preparation_path=args.preparation,
                           source_map_path=args.source_map,
                           review_paths=args.review, output=args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
