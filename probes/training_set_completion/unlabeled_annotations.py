"""Export accepted non-GT physical owners beside the frozen COCO annotations.

The export is an annotation artifact, not a new review pass.  It copies the
eleven acquired input records unchanged and adds only ``unlabeled`` rows that
are already accepted in the fixed owner catalog.  A masked teacher description
is never converted into a class label here.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


B = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum")
ROOT_V1 = B / "annotations-with-unlabeled-v1"
ROOT = B / "annotations-with-unlabeled-v3"
CATALOG = B / "target-owners-complete-v4.json"
ACQUISITION = B / "stage01-acquisition-v1-retry1-config-batch2/manifest.json"
COMPLETE_ACCEPTANCE = B / "third-complete-bank-preparation-v2/root-acceptance.json"
COMPLETE_BANK = B / "third-complete-bank-preparation-v2/bank.json"
SCHEMA = "training_set_completion.annotations_with_unlabeled.v1"
REVIEW_INDEX_SCHEMA = "training_set_completion.unlabeled_review_source_index.v1"
EXTRA_EVIDENCE_SCHEMA = "training_set_completion.third_fit_admissions_with_evidence.v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def binding(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    require(resolved.is_file(), f"source is not a file: {resolved}")
    return {"path": str(resolved), "sha256": file_hash(resolved), "size_bytes": resolved.stat().st_size}


def read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def _as_bindings(value: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep recorded evidence bindings and prove they still name the same bytes."""
    rows = []
    for item in value:
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
            continue
        observed = binding(item["path"])
        require(observed["sha256"] == item.get("sha256"), f"recorded evidence changed: {item['path']}")
        rows.append(observed)
    return rows


def _coord_tokens(bins: list[int]) -> list[str]:
    return [f"<|coord_{value}|>" for value in bins]


def validate_bbox(bins: Any, *, owner_id: str) -> list[int]:
    require(isinstance(bins, list) and len(bins) == 4 and all(type(value) is int for value in bins), f"{owner_id}: bbox bins")
    x1, y1, x2, y2 = bins
    require(0 <= x1 < x2 <= 999 and 0 <= y1 < y2 <= 999, f"{owner_id}: invalid bbox area")
    return list(bins)


def _masked_owner_ids(path: Path) -> set[str]:
    receipt = read(path)
    owners = receipt.get("masked_description_owner_ids")
    require(isinstance(owners, list) and len(owners) == 13 and all(isinstance(owner, str) for owner in owners), "complete-bank class mask")
    return set(owners)


def _bank_descriptions(path: Path) -> dict[str, str]:
    bank = read(path)
    result: dict[str, str] = {}
    for route in bank.get("routes", []):
        for trace in route.get("provenance", {}).get("trace", []):
            owner = trace.get("owner_id")
            description = trace.get("edited_fields", {}).get("selected_description")
            if isinstance(owner, str) and isinstance(description, str) and description:
                require(owner not in result or result[owner] == description, f"conflicting bank description: {owner}")
                result[owner] = description
    require(len(result) == 232, "complete-bank owner description coverage")
    return result


def _review_binding(image_id: int) -> dict[str, Any]:
    path = B / "stage01-owner-reviews-v1" / f"image-{image_id:012d}" / "review.json"
    return binding(path)


def _path_binding(path: str | Path) -> dict[str, Any]:
    return binding(Path(path))


def _unique_bindings(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    answer = []
    for row in rows:
        key = (str(row["path"]), str(row["sha256"]))
        if key not in seen:
            seen.add(key)
            answer.append(dict(row))
    return answer


def _crop_path(path: str | Path) -> bool:
    value = Path(path)
    parts = {part.lower() for part in value.parts}
    name = value.name.lower()
    return "crops" in parts or "crop" in name or "context" in name or "tight" in name or "evidence" in parts


def _packet_original(path: Path) -> list[dict[str, Any]]:
    packet = read(path)
    candidates: list[Any] = [packet.get("original_image"), packet.get("source", {}).get("original_image"), packet.get("rendered", {}).get("original_image")]
    rendered = packet.get("rendered", {})
    if isinstance(rendered.get("original_image_reference"), str):
        candidates.append({"path": rendered["original_image_reference"]})
    result = []
    for candidate in candidates:
        if isinstance(candidate, Mapping) and isinstance(candidate.get("path"), str):
            result.append(_path_binding(candidate["path"]))
    return _unique_bindings(result)


def _stage01_proposals() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    root = B / "stage01-review-extraction-v2/proposal-rows"
    for path in sorted(root.glob("image-*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            proposal_id = row.get("proposal_id")
            if isinstance(proposal_id, str):
                rows[proposal_id] = row
    return rows


def _bank_originals(path: Path) -> dict[int, dict[str, Any]]:
    bank = read(path)
    result = {}
    for route in bank.get("routes", []):
        image_id = route.get("image_id")
        image = route.get("image_identity", {}).get("image_path")
        if isinstance(image_id, int) and isinstance(image, str):
            result[image_id] = _path_binding(image)
    require(len(result) == 11, "complete-bank original-image bindings")
    return result


def _find_owner_nodes(value: Any, owner_id: str) -> list[Mapping[str, Any]]:
    answer: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        if value.get("owner_id") == owner_id:
            answer.append(value)
        for nested in value.values():
            answer.extend(_find_owner_nodes(nested, owner_id))
    elif isinstance(value, list):
        for nested in value:
            answer.extend(_find_owner_nodes(nested, owner_id))
    return answer


def _paths_in(value: Any) -> list[str]:
    answer: list[str] = []
    if isinstance(value, str) and (value.endswith(".png") or value.endswith(".jpg") or value.endswith(".json")):
        answer.append(value)
    elif isinstance(value, Mapping):
        for nested in value.values():
            answer.extend(_paths_in(nested))
    elif isinstance(value, list):
        for nested in value:
            answer.extend(_paths_in(nested))
    return answer


def _visual_entry(*, owner_id: str, image_id: int, decision_source: Mapping[str, Any], reason: str | None, originals: Iterable[Mapping[str, Any]], overlays: Iterable[Mapping[str, Any]], crops: Iterable[Mapping[str, Any]], proposal_id: str | None, exceptions: Iterable[str] = ()) -> dict[str, Any]:
    entry = {
        "image_id": image_id,
        "stable_owner_id": owner_id,
        "decision": {"source": dict(decision_source), "proposal_id": proposal_id, "reason": reason},
        "visual": {"original": _unique_bindings(originals), "bbox_overlay": _unique_bindings(overlays), "crop": _unique_bindings(crops)},
        "exceptions": list(exceptions),
    }
    missing = [name for name, rows in entry["visual"].items() if not rows]
    if missing:
        entry["exceptions"].append("missing persisted " + "/".join(missing) + " binding")
    entry["status"] = "complete" if not entry["exceptions"] else "exception"
    return entry


def _review_source_index(*, candidates: Iterable[Mapping[str, Any]], complete_bank_path: Path) -> dict[str, Any]:
    """Resolve an owner to the exact persisted review decision and visual files."""
    stage_rows = _stage01_proposals()
    first = read(B / "first-fit-new-owner-admissions-v1/admissions.json")["entries"]
    first_by_owner = {str(row["owner_id"]): row for row in first}
    second = read(B / "second-fit-root-rulings-v1/admissions.json")["admissions"]
    second_by_owner = {str(row["owner_id"]): row for row in second}
    bank_originals = _bank_originals(complete_bank_path)
    entries: dict[str, dict[str, Any]] = {}
    for record in candidates:
        owner, image_id = str(record["owner_id"]), int(record["image_id"])
        proposal_id = record.get("reference_proposal_id")
        if isinstance(record.get("_extra_review_evidence"), Mapping):
            evidence, decision = record["_extra_review_evidence"], record.get("_extra_decision")
            require(isinstance(decision, Mapping) and isinstance(decision.get("source"), Mapping), f"extra decision source: {owner}")
            decision_source = dict(decision["source"])
            require(binding(decision_source["path"]) == decision_source, f"extra decision source changed: {owner}")
            originals = _as_bindings(evidence.get("original", []))
            overlays = _as_bindings(evidence.get("bbox_overlay", []))
            crops = _as_bindings(evidence.get("crop", []))
            entries[owner] = _visual_entry(owner_id=owner, image_id=image_id, decision_source=decision_source, reason=decision.get("reason"), originals=originals, overlays=overlays, crops=crops, proposal_id=proposal_id)
            continue
        if isinstance(proposal_id, str) and proposal_id.startswith("stage01:"):
            source = stage_rows.get(proposal_id)
            require(source is not None, f"missing stage01 proposal row: {owner}")
            review_path = B / "stage01-owner-reviews-v1" / f"image-{image_id:012d}" / "review.json"
            review = read(review_path)
            nodes = _find_owner_nodes(review, owner)
            visual_paths = _paths_in(source.get("review_evidence") or source.get("raw_review_decision", {}).get("visual_evidence") or [])
            for node in nodes:
                visual_paths.extend(_paths_in(node.get("evidence", {})))
            bindings = []
            for path in visual_paths:
                candidate = Path(path) if Path(path).is_absolute() else review_path.parent / path
                if candidate.is_file():
                    bindings.append(_path_binding(candidate))
            # Some early owner reviews stored a crop but referred to the policy
            # overlay through the packet rather than repeating its full path.
            policy = source.get("policy")
            if isinstance(policy, str):
                overlay = B / "stage01-review-packets-v1" / f"image-{image_id:012d}" / f"pred-overlay-{policy}.png"
                if overlay.is_file():
                    bindings.append(_path_binding(overlay))
            node_reason = next((node.get("reason") or node.get("note") for node in nodes if node.get("reason") or node.get("note")), None)
            entries[owner] = _visual_entry(owner_id=owner, image_id=image_id, decision_source=binding(review_path), reason=source.get("review_reason") or source.get("raw_review_decision", {}).get("reason") or node_reason, originals=[bank_originals[image_id]], overlays=[item for item in bindings if "overlay" in Path(item["path"]).name.lower()], crops=[item for item in bindings if _crop_path(item["path"]) or "newowner-" in Path(item["path"]).name], proposal_id=proposal_id)
            continue
        if owner in first_by_owner:
            admission = first_by_owner[owner]
            sources = _as_bindings(admission.get("sources", []))
            packet = next((Path(item["path"]) for item in sources if Path(item["path"]).name == "packet.json"), None)
            entries[owner] = _visual_entry(owner_id=owner, image_id=image_id, decision_source=binding(B / "first-fit-new-owner-admissions-v1/admissions.json"), reason=admission.get("ruling"), originals=_packet_original(packet) if packet else [], overlays=[item for item in sources if "overlay" in Path(item["path"]).name.lower()], crops=[item for item in sources if _crop_path(item["path"])], proposal_id=proposal_id)
            continue
        if owner in second_by_owner:
            admission = second_by_owner[owner]
            evidence = _as_bindings(admission.get("evidence", []))
            entries[owner] = _visual_entry(owner_id=owner, image_id=image_id, decision_source=binding(B / "second-fit-root-rulings-v1/admissions.json"), reason=admission.get("reason"), originals=[item for item in evidence if Path(item["path"]).name == "original.jpg"], overlays=[item for item in evidence if "overlay" in Path(item["path"]).name.lower()], crops=[item for item in evidence if _crop_path(item["path"])], proposal_id=proposal_id)
            continue
        # Four catalog records use a fixed/manual reference.  Their evidence is
        # explicit in their owner review; select that review rather than a broad
        # image-level binding.
        review_path = B / "stage01-owner-reviews-v1" / f"image-{image_id:012d}" / "review.json"
        review = read(review_path)
        nodes = _find_owner_nodes(review, owner)
        paths = []
        for node in nodes:
            for path in _paths_in(node.get("evidence", {})):
                candidate = Path(path) if Path(path).is_absolute() else review_path.parent / path
                if candidate.is_file():
                    paths.append(_path_binding(candidate))
        # The full-reference render is named in the first-fit owner review for
        # the steering wheel and in the stage01 review for the spoon.
        if owner == "stablenew:rear-steering-wheel":
            paths.extend([_path_binding(B / "first-fit-owner-reviews-v1/image-000000388795/rear-steering-wheel-root-reference.png"), _path_binding(review_path.parent / "newowner-steering-wheel.png")])
        if owner == "stableNew:right-white-spoon":
            paths.extend([_path_binding(review_path.parent / "root-right-spoon-full-extent-overlay.png"), _path_binding(review_path.parent / "root-right-spoon-full-extent-crop.png")])
        bindings = _unique_bindings(paths)
        reason = next((str(node.get("reason") or node.get("note")) for node in nodes if node.get("reason") or node.get("note")), None)
        entries[owner] = _visual_entry(owner_id=owner, image_id=image_id, decision_source=binding(review_path), reason=reason, originals=[bank_originals[image_id]], overlays=[item for item in bindings if "overlay" in Path(item["path"]).name.lower() or "reference" in Path(item["path"]).name.lower()], crops=[item for item in bindings if _crop_path(item["path"]) or "newowner-steering-wheel" in Path(item["path"]).name], proposal_id=proposal_id)
    require(len(entries) == len(list(candidates)), "review-source owner coverage")
    return {"schema": REVIEW_INDEX_SCHEMA, "status": "candidate_ready", "entries": entries, "content_sha256": digest({"schema": REVIEW_INDEX_SCHEMA, "status": "candidate_ready", "entries": entries})}


def _catalog_candidates(catalog: Mapping[str, Any]) -> list[dict[str, Any]]:
    require(catalog.get("schema") == "training_set_completion.target_owners.v4" and catalog.get("status") == "lead-accepted", "accepted v4 target catalog")
    records = catalog.get("records")
    require(isinstance(records, list) and len(records) == 232, "v4 target catalog denominator")
    result = [dict(record) for record in records if record.get("role") != "gt_atomic"]
    require(len(result) == 63, "v4 accepted non-GT denominator")
    require(all(record.get("physical_match") == "separate_owner_metric" for record in result), "non-GT record lacks accepted physical owner")
    require(all(isinstance(record.get("owner_id"), str) and record["owner_id"] for record in result), "non-GT stable owner ID")
    return result


def _extra_candidates(path: Path | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if path is None:
        return [], []
    source = read(path)
    if isinstance(source, Mapping) and source.get("schema") == EXTRA_EVIDENCE_SCHEMA:
        require(source.get("status") == "candidate_ready" and isinstance(source.get("entries"), list), "extra evidence schema")
        rows = []
        for entry in source["entries"]:
            require(isinstance(entry, Mapping) and isinstance(entry.get("admission"), Mapping) and isinstance(entry.get("review_evidence"), Mapping) and isinstance(entry.get("decision"), Mapping), "extra evidence entry")
            row = dict(entry["admission"])
            row["_extra_review_evidence"] = entry["review_evidence"]
            row["_extra_decision"] = entry["decision"]
            rows.append(row)
    else:
        rows = source.get("admissions") if isinstance(source, Mapping) else source
    require(isinstance(rows, list), "extra root admissions must be a JSON list or {admissions: [...]}")
    source_binding = binding(path)
    candidates = []
    for row in rows:
        require(isinstance(row, Mapping), "extra root admission row")
        accepted = row.get("status") in ("accepted", "lead-accepted") or str(row.get("reference_status", "")).startswith("qualified_")
        require(accepted and row.get("physical_match") == "separate_owner_metric", "extra admission is not an accepted separate physical owner")
        candidate = dict(row)
        candidate.setdefault("role", "new")
        candidates.append(candidate)
    return candidates, [source_binding]


def materialize_extra_evidence(*, admissions_path: Path, output: Path) -> dict[str, Any]:
    """Freeze explicit visual bindings for the accepted third-fit additions."""
    require(not output.exists(), f"collision: {output}")
    source = read(admissions_path)
    rows = source.get("admissions") if isinstance(source, Mapping) else source
    require(isinstance(rows, list) and len(rows) == 14, "third-fit admission denominator")
    decision_source = binding(admissions_path)
    entries = []
    for admission in rows:
        require(admission.get("status") == "lead-accepted" and admission.get("physical_match") == "separate_owner_metric", "accepted third-fit admission")
        sources = _as_bindings(admission.get("sources", []))
        originals = [item for item in sources if Path(item["path"]).name == "original.jpg"]
        crops = [item for item in sources if _crop_path(item["path"])]
        overlays = [item for item in sources if item not in originals and item not in crops and Path(item["path"]).suffix == ".png"]
        require(len(originals) == 1 and crops and overlays, f"third-fit explicit visual evidence: {admission.get('owner_id')}")
        entries.append({"admission": dict(admission), "review_evidence": {"original": _unique_bindings(originals), "bbox_overlay": _unique_bindings(overlays), "crop": _unique_bindings(crops)}, "decision": {"source": decision_source, "proposal_id": admission.get("reference_proposal_id"), "reason": admission.get("ruling")}})
    value = {"schema": EXTRA_EVIDENCE_SCHEMA, "status": "candidate_ready", "source_admissions": decision_source, "entries": entries}
    value["content_sha256"] = digest(value)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(canonical(value)); handle.flush(); os.fsync(handle.fileno())
    require(read(output) == value, "extra-evidence publication readback")
    return value


def _evidence(record: Mapping[str, Any], *, catalog_binding: Mapping[str, Any], mask_binding: Mapping[str, Any], review_source: Mapping[str, Any], review_index_binding: Mapping[str, Any], extra_admissions: list[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    direct = _as_bindings(record.get("sources", []))
    for key in ("root_admission", "root_reference_ruling", "reference_ruling", "reference_receipt"):
        value = record.get(key)
        if isinstance(value, Mapping) and isinstance(value.get("path"), str):
            direct.extend(_as_bindings([value]))
    # Dedupe by immutable bytes while retaining a compact provenance surface.
    def unique(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        seen: set[tuple[str, str]] = set()
        answer = []
        for row in rows:
            key = (str(row["path"]), str(row["sha256"]))
            if key not in seen:
                seen.add(key)
                answer.append(dict(row))
        return answer

    crop = [item for item in direct if "crop" in Path(item["path"]).name.lower()]
    admission = [item for item in direct if "admission" in Path(item["path"]).name.lower()] + list(extra_admissions)
    geometry = [item for item in direct if "ruling" in Path(item["path"]).name.lower() or "reference" in Path(item["path"]).name.lower()]
    visual = review_source["visual"]
    recorded = _unique_bindings([*visual["original"], *visual["bbox_overlay"], *visual["crop"], review_source["decision"]["source"]])
    return {
        "target_catalog": [dict(catalog_binding)],
        "class_mask_ruling": [dict(mask_binding)],
        "review_source_index": [dict(review_index_binding)],
        "admission": unique(admission),
        "geometry": unique(geometry),
        "crop": unique(crop),
        "recorded_visual_or_review": recorded,
    }


def unlabeled_record(record: Mapping[str, Any], *, descriptions: Mapping[str, str], masked: set[str], catalog_binding: Mapping[str, Any], mask_binding: Mapping[str, Any], review_source: Mapping[str, Any], review_index_binding: Mapping[str, Any], extra_admissions: list[Mapping[str, Any]]) -> dict[str, Any]:
    owner_id = record.get("owner_id")
    image_id = record.get("image_id")
    require(isinstance(owner_id, str) and isinstance(image_id, int), "unlabeled identity")
    bins = validate_bbox(record.get("reference_coord_bins_1000"), owner_id=owner_id)
    unknown = owner_id in masked
    description = None if unknown else (record.get("category") or descriptions.get(owner_id))
    require(unknown or isinstance(description, str), f"verified owner has no accepted description: {owner_id}")
    return {
        "stable_owner_id": owner_id,
        "bbox_2d": _coord_tokens(bins),
        "bbox_2d_bins_1000": bins,
        "coordinate_convention": "normalized discrete xyxy bins 0..999; native <|coord_N|> spelling",
        "category_id": None,
        "category_name": description,
        "desc": description,
        "class_status": "unknown" if unknown else "verified",
        "physical_status": "valid_unlabeled",
        "geometry_status": "reasonable",
        "reference_status": record.get("reference_status"),
        "reference_proposal_id": record.get("reference_proposal_id"),
        "provenance": _evidence(record, catalog_binding=catalog_binding, mask_binding=mask_binding, review_source=review_source, review_index_binding=review_index_binding, extra_admissions=extra_admissions),
    }


def validate(rows: list[Mapping[str, Any]], *, expected_non_gt: int, masked: set[str]) -> None:
    require(len(rows) == 11, "eleven annotation rows")
    owners = []
    for row in rows:
        require(isinstance(row.get("objects"), list) and isinstance(row.get("unlabeled"), list), "annotation/object lists")
        gt_ids = {str(item.get("coco_ann_id")) for item in row["objects"] if item.get("coco_ann_id") is not None}
        local = []
        for item in row["unlabeled"]:
            owner = item.get("stable_owner_id")
            require(isinstance(owner, str) and owner not in gt_ids, "GT/non-GT duplicate")
            validate_bbox(item.get("bbox_2d_bins_1000"), owner_id=owner)
            require(item.get("bbox_2d") == _coord_tokens(item["bbox_2d_bins_1000"]), "native coordinate spelling")
            require(item.get("physical_status") == "valid_unlabeled" and item.get("geometry_status") == "reasonable", "unlabeled physical/geometry status")
            expected_class = "unknown" if owner in masked else "verified"
            require(item.get("class_status") == expected_class, "class mask promotion")
            if expected_class == "unknown":
                require(item.get("category_name") is None and item.get("desc") is None, "masked class leaked into export")
            else:
                require(isinstance(item.get("category_name"), str) and item["category_name"] == item.get("desc"), "verified description")
            require(set(item.get("provenance", {})) == {"target_catalog", "class_mask_ruling", "review_source_index", "admission", "geometry", "crop", "recorded_visual_or_review"}, "provenance fields")
            visual = item["provenance"]["recorded_visual_or_review"]
            paths = [Path(source["path"]) for source in visual]
            require(any(path.name == "original.jpg" or "/public_data/" in str(path) for path in paths), "owner lacks original visual binding")
            require(any("overlay" in path.name.lower() or "reference" in path.name.lower() for path in paths), "owner lacks bbox-overlay binding")
            require(any(_crop_path(path) or "newowner-" in path.name for path in paths), "owner lacks crop binding")
            local.append(owner)
        require(len(local) == len(set(local)), "duplicate non-GT owner within image")
        owners.extend(local)
    require(len(owners) == expected_non_gt and len(set(owners)) == expected_non_gt, "unlabeled owner denominator")
    require(masked <= set(owners), "masked owner missing from export")


def _publish_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    require(not path.exists(), f"collision: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    content = b"".join(canonical(row) for row in rows)
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    require(path.read_bytes() == content, "annotations publication readback")


def build(*, output: Path = ROOT, catalog_path: Path = CATALOG, acquisition_path: Path = ACQUISITION, complete_acceptance_path: Path = COMPLETE_ACCEPTANCE, complete_bank_path: Path = COMPLETE_BANK, extra_root_admissions: Path | None = None) -> dict[str, Any]:
    require(not output.exists() or not any(output.iterdir()), f"collision: {output}")
    catalog, acquisition = read(catalog_path), read(acquisition_path)
    catalog_binding, mask_binding = binding(catalog_path), binding(complete_acceptance_path)
    masked, descriptions = _masked_owner_ids(complete_acceptance_path), _bank_descriptions(complete_bank_path)
    candidates = _catalog_candidates(catalog)
    extras, extra_bindings = _extra_candidates(extra_root_admissions)
    candidates.extend(extras)
    require(len({str(item.get("owner_id")) for item in candidates}) == len(candidates), "duplicate accepted non-GT owner")
    # Later root admissions may explicitly keep their class unknown.  They can
    # extend the immutable export without weakening the fixed v2 mask ruling.
    unknown_owners = masked | {
        str(item["owner_id"])
        for item in extras
        if item.get("class_status") == "unknown" or item.get("class_policy") == "mask_description"
    }
    by_image: dict[int, list[dict[str, Any]]] = {}
    for candidate in candidates:
        by_image.setdefault(int(candidate["image_id"]), []).append(candidate)
    source_records = {int(record["case"]["input_record"]["image_id"]): record["case"]["input_record"] for record in acquisition.get("records", [])}
    require(len(source_records) == 11, "acquisition source rows")
    output.mkdir(parents=True, exist_ok=True)
    review_index = _review_source_index(candidates=candidates, complete_bank_path=complete_bank_path)
    require(all(entry["status"] == "complete" for entry in review_index["entries"].values()), "review-source exceptions: " + json.dumps({owner: entry["exceptions"] for owner, entry in review_index["entries"].items() if entry["exceptions"]}, sort_keys=True))
    review_index_path = output / "review-source-index.json"
    with review_index_path.open("xb") as handle:
        handle.write(canonical(review_index)); handle.flush(); os.fsync(handle.fileno())
    require(read(review_index_path) == review_index, "review-source index publication readback")
    review_index_binding = binding(review_index_path)
    annotations_path = output / "annotations.jsonl"
    # Attach the index only after every owner has a complete persisted visual
    # chain, so a row cannot silently fall back to image-level review evidence.
    rows = []
    for image_id in sorted(source_records):
        original = source_records[image_id]
        row = copy.deepcopy(original)
        row["unlabeled"] = [unlabeled_record(candidate, descriptions=descriptions, masked=unknown_owners, catalog_binding=catalog_binding, mask_binding=mask_binding, review_source=review_index["entries"][str(candidate["owner_id"])], review_index_binding=review_index_binding, extra_admissions=extra_bindings) for candidate in sorted(by_image.get(image_id, []), key=lambda value: str(value["owner_id"]))]
        rows.append(row)
    validate(rows, expected_non_gt=len(candidates), masked=unknown_owners)
    _publish_jsonl(annotations_path, rows)
    manifest = {
        "schema": SCHEMA,
        "status": "candidate_ready",
        "annotations": binding(annotations_path),
        "sources": {"catalog": catalog_binding, "acquisition_manifest": binding(acquisition_path), "complete_bank_acceptance": mask_binding, "complete_bank": binding(complete_bank_path), "review_source_index": review_index_binding, "v1_snapshot_producer": binding(ROOT_V1 / "producer.py"), "producer": binding(Path(__file__)), "extra_root_admissions": extra_bindings},
        "counts": {"images": len(rows), "gt_objects": sum(len(row["objects"]) for row in rows), "valid_unlabeled": len(candidates), "class_unknown": len(unknown_owners), "class_verified": len(candidates) - len(unknown_owners)},
        "replay": "python -m probes.training_set_completion.unlabeled_annotations --output <new-empty-output-dir>" + (f" --extra-root-admissions {extra_root_admissions}" if extra_root_admissions else ""),
    }
    manifest["content_sha256"] = digest(manifest)
    manifest_path = output / "manifest.json"
    with manifest_path.open("xb") as handle:
        handle.write(canonical(manifest)); handle.flush(); os.fsync(handle.fileno())
    require(read(manifest_path) == manifest, "manifest publication readback")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--acquisition", type=Path, default=ACQUISITION)
    parser.add_argument("--complete-acceptance", type=Path, default=COMPLETE_ACCEPTANCE)
    parser.add_argument("--complete-bank", type=Path, default=COMPLETE_BANK)
    parser.add_argument("--extra-root-admissions", type=Path, default=None, help="optional accepted third-fit admission JSON for a later immutable export")
    parser.add_argument("--write-extra-evidence", type=Path, default=None, help="freeze explicit visual bindings for a supplied third-fit admission file")
    args = parser.parse_args()
    if args.write_extra_evidence is not None:
        require(args.extra_root_admissions is not None, "--write-extra-evidence requires --extra-root-admissions")
        print(json.dumps(materialize_extra_evidence(admissions_path=args.extra_root_admissions, output=args.write_extra_evidence), sort_keys=True))
        return
    print(json.dumps(build(output=args.output, catalog_path=args.catalog, acquisition_path=args.acquisition, complete_acceptance_path=args.complete_acceptance, complete_bank_path=args.complete_bank, extra_root_admissions=args.extra_root_admissions), sort_keys=True))


if __name__ == "__main__":
    main()
