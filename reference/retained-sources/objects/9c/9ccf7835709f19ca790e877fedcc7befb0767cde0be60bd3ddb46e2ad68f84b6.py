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
ROOT = B / "annotations-with-unlabeled-v1"
CATALOG = B / "target-owners-complete-v4.json"
ACQUISITION = B / "stage01-acquisition-v1-retry1-config-batch2/manifest.json"
COMPLETE_ACCEPTANCE = B / "third-complete-bank-preparation-v2/root-acceptance.json"
COMPLETE_BANK = B / "third-complete-bank-preparation-v2/bank.json"
SCHEMA = "training_set_completion.annotations_with_unlabeled.v1"


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


def _evidence(record: Mapping[str, Any], *, catalog_binding: Mapping[str, Any], mask_binding: Mapping[str, Any], review: Mapping[str, Any], extra_admissions: list[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
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
    return {
        "target_catalog": [dict(catalog_binding)],
        "class_mask_ruling": [dict(mask_binding)],
        "review": [dict(review)],
        "admission": unique(admission),
        "geometry": unique(geometry),
        "crop": unique(crop),
        "recorded_visual_or_review": unique(direct),
    }


def unlabeled_record(record: Mapping[str, Any], *, descriptions: Mapping[str, str], masked: set[str], catalog_binding: Mapping[str, Any], mask_binding: Mapping[str, Any], extra_admissions: list[Mapping[str, Any]]) -> dict[str, Any]:
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
        "provenance": _evidence(record, catalog_binding=catalog_binding, mask_binding=mask_binding, review=_review_binding(image_id), extra_admissions=extra_admissions),
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
            require(set(item.get("provenance", {})) == {"target_catalog", "class_mask_ruling", "review", "admission", "geometry", "crop", "recorded_visual_or_review"}, "provenance fields")
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
    require(not output.exists(), f"collision: {output}")
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
    rows = []
    for image_id in sorted(source_records):
        original = source_records[image_id]
        row = copy.deepcopy(original)
        row["unlabeled"] = [unlabeled_record(candidate, descriptions=descriptions, masked=unknown_owners, catalog_binding=catalog_binding, mask_binding=mask_binding, extra_admissions=extra_bindings) for candidate in sorted(by_image.get(image_id, []), key=lambda value: str(value["owner_id"]))]
        rows.append(row)
    validate(rows, expected_non_gt=len(candidates), masked=unknown_owners)
    output.mkdir(parents=True)
    annotations_path = output / "annotations.jsonl"
    _publish_jsonl(annotations_path, rows)
    manifest = {
        "schema": SCHEMA,
        "status": "candidate_ready",
        "annotations": binding(annotations_path),
        "sources": {"catalog": catalog_binding, "acquisition_manifest": binding(acquisition_path), "complete_bank_acceptance": mask_binding, "complete_bank": binding(complete_bank_path), "extra_root_admissions": extra_bindings},
        "counts": {"images": len(rows), "gt_objects": sum(len(row["objects"]) for row in rows), "valid_unlabeled": len(candidates), "class_unknown": len(unknown_owners), "class_verified": len(candidates) - len(unknown_owners)},
        "replay": "python -m probes.training_set_completion.unlabeled_annotations --output <new-empty-output-dir>",
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
    args = parser.parse_args()
    print(json.dumps(build(output=args.output, catalog_path=args.catalog, acquisition_path=args.acquisition, complete_acceptance_path=args.complete_acceptance, complete_bank_path=args.complete_bank, extra_root_admissions=args.extra_root_admissions), sort_keys=True))


if __name__ == "__main__":
    main()
