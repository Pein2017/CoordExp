"""CPU-only freeze of the new owner-successor confirmation panel.

This is intentionally an identity/materialization receipt, not an evaluator
or inference entrypoint.  The prior evaluation selection is the authority for
the earlier exclusion reconciliation; its selected fresh256 is added
explicitly because it was exposed after that reconciliation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

# Keep the replay command independent of shell-specific PYTHONPATH setup.
sys.path.insert(0, str(Path(__file__).resolve().parents[6]))
from probes.native_owner_scale import evaluation as prior_helpers


BASE = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration")
UNIT = "2026-09-13-owner-successor-scale-throughput"
OUTPUT = BASE / UNIT / "evaluation"
PRIOR = BASE / "2026-09-12-native-owner-scale-and-state/evaluation/selection.json"
SOURCE = Path("/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/val.coord.jsonl")
NATIVE_SOURCE = Path(
    "/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000_xy_sorted/val.coord.jsonl"
)
PANEL_SIZE = 256
REVIEW_SIZE = 32
PANEL_SALT = "owner-successor-scale-throughput-confirmation-2026-09-13:"
REVIEW_SALT = "owner-successor-scale-throughput-review-2026-09-13:"


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _file_binding(path: Path) -> dict[str, Any]:
    # Reuse the accepted helper's exact file hash/binding convention.
    return prior_helpers.binding(path)


def _ids_digest(ids: Iterable[int]) -> str:
    return _sha([int(x) for x in ids])


def _load_rows(path: Path) -> tuple[list[dict[str, Any]], dict[int, tuple[int, dict[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    by_id: dict[int, tuple[int, dict[str, Any]]] = {}
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        image_id = int(row["image_id"])
        if image_id in by_id:
            raise ValueError(f"duplicate image_id in {path}: {image_id}")
        by_id[image_id] = (line_no, row)
        rows.append(row)
    return rows, by_id


def _coord_value(token: str) -> int:
    prefix, value = token.rsplit("_", 1)
    if prefix not in {"<|coord"} or not value.endswith("|>"):
        raise ValueError(f"not a native coordinate token: {token}")
    value = value[:-2]
    number = int(value)
    if not 0 <= number <= 1000:
        raise ValueError(f"coordinate bin outside norm1000: {token}")
    return number


def _objects_digest(objects: list[dict[str, Any]]) -> str:
    return _sha(objects)


def _verify_native_geometry(source: dict[str, Any], native: dict[str, Any]) -> dict[str, Any]:
    if int(source["image_id"]) != int(native["image_id"]):
        raise ValueError("native/source image identity mismatch")
    if (source["width"], source["height"], source["images"]) != (
        native["width"],
        native["height"],
        native["images"],
    ):
        raise ValueError(f"native physical identity mismatch for {source['image_id']}")
    source_objects = source["objects"]
    native_objects = native["objects"]
    if sorted(_sha(obj) for obj in source_objects) != sorted(_sha(obj) for obj in native_objects):
        raise ValueError(f"native object multiset mismatch for {source['image_id']}")
    anchors: list[tuple[int, int]] = []
    for obj in native_objects:
        bbox = obj["bbox_2d"]
        if len(bbox) != 4:
            raise ValueError(f"native bbox arity mismatch for {source['image_id']}")
        coords = tuple(_coord_value(str(token)) for token in bbox)
        if not (coords[0] < coords[2] and coords[1] < coords[3]):
            raise ValueError(f"native degenerate bbox for {source['image_id']}")
        anchors.append((coords[0], coords[1]))
    if anchors != sorted(anchors):
        raise ValueError(f"native geo_sorted_xy order mismatch for {source['image_id']}")
    physical = (SOURCE.parent / source["images"][0]).resolve()
    native_physical = (NATIVE_SOURCE.parent / native["images"][0]).resolve()
    if physical != native_physical or not physical.is_file():
        raise ValueError(f"missing or mismatched physical image for {source['image_id']}: {physical}")
    try:
        from PIL import Image

        with Image.open(physical) as image:
            actual_size = tuple(image.size)
    except Exception as exc:  # pragma: no cover - environment/data failure
        raise ValueError(f"cannot inspect physical image for {source['image_id']}: {exc}") from exc
    if actual_size != (int(source["width"]), int(source["height"])):
        raise ValueError(f"physical dimensions disagree for {source['image_id']}: {actual_size}")
    return {
        "image_id": int(source["image_id"]),
        "example_id": f"coco2017_val_{int(source['image_id']):012d}",
        "source_line": None,
        "native_line": None,
        "physical_image_path": str(physical),
        "width": int(source["width"]),
        "height": int(source["height"]),
        "object_count": len(source_objects),
        "source_objects_sha256": _objects_digest(source_objects),
        "native_objects_sha256": _objects_digest(native_objects),
        "native_geometry": "norm1000_xyxy; geo_sorted_xy=(x1,y1); native/source object multiset equal",
    }


def _check_binding(path: Path, declared: dict[str, Any], label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing provenance binding ({label}): {path}")
    current = _file_binding(path)
    # The previous selection is immutable authority, but one source was
    # rewritten after its freeze while preserving its 384 projected IDs.  We
    # carry both hashes and verify that identity projection below.
    if current["sha256"] == declared["sha256"] and current["size_bytes"] == declared["size_bytes"]:
        return {"declared": declared, "current": current, "status": "byte_exact"}
    return {"declared": declared, "current": current, "status": "source_changed_identity_rechecked"}


def _prior_boundary(prior: dict[str, Any]) -> tuple[set[int], list[dict[str, Any]]]:
    if prior.get("status") != "frozen_before_new_outputs":
        raise ValueError("prior evaluation selection is not frozen_before_new_outputs")
    if prior.get("source") != _file_binding(SOURCE) or prior.get("native_source") != _file_binding(NATIVE_SOURCE):
        raise ValueError("prior evaluation source binding changed")
    excluded = {int(x) for x in prior["excluded_image_ids"]}
    fresh = {int(x) for x in prior["image_ids"]}
    if len(excluded) != len(prior["excluded_image_ids"]):
        raise ValueError("prior excluded identity list is not unique")
    if len(fresh) != len(prior["image_ids"]) or len(fresh) != PANEL_SIZE:
        raise ValueError("prior fresh panel identity/denominator")
    if excluded & fresh:
        raise ValueError("prior fresh panel overlaps prior exclusions")
    sources: list[dict[str, Any]] = []
    for source in prior["exclusion_sources"]:
        ids = [int(x) for x in source["image_ids"]]
        if len(ids) != source["image_count"] or len(set(ids)) != len(ids):
            raise ValueError(f"prior exclusion source identity count: {source['role']}")
        check = _check_binding(Path(source["binding"]["path"]), source["binding"], source["role"])
        # A changed source must still expose the exact declared identity set.
        if check["status"] != "byte_exact":
            current = _read(Path(source["binding"]["path"]))
            projected = prior_helpers.image_ids(current)
            if not set(ids) <= projected:
                raise ValueError(f"changed prior source lost declared IDs: {source['role']}")
        sources.append(
            {
                "role": source["role"],
                "image_count": len(ids),
                "image_ids_sha256": _ids_digest(sorted(ids)),
                "declared_binding": check["declared"],
                "current_binding": check["current"],
                "binding_status": check["status"],
            }
        )
    if set().union(*(set(int(x) for x in s["image_ids"]) for s in prior["exclusion_sources"])) != excluded:
        raise ValueError("prior exclusion_sources do not exactly reconcile to excluded_image_ids")
    boundary = excluded | fresh
    sources.append(
        {
            "role": "previous-fresh256-now-exposed",
            "image_count": len(fresh),
            "image_ids_sha256": _ids_digest(sorted(fresh)),
            "selection_binding": _file_binding(PRIOR),
            "binding_status": "byte_exact",
        }
    )
    return boundary, sources


def freeze(output: Path = OUTPUT) -> dict[str, Any]:
    if (output / "confirmation-selection.json").exists():
        raise ValueError(f"refusing to overwrite frozen packet: {output / 'confirmation-selection.json'}")
    source_rows, source_by_id = _load_rows(SOURCE)
    native_rows, native_by_id = _load_rows(NATIVE_SOURCE)
    if len(source_rows) != len(native_rows) or set(source_by_id) != set(native_by_id):
        raise ValueError("native/source val universes differ")
    prior = _read(PRIOR)
    boundary, exclusion_sources = _prior_boundary(prior)
    universe = sorted(source_by_id)
    panel = prior_helpers.select_ids(universe, boundary, PANEL_SIZE, PANEL_SALT)
    review = prior_helpers.select_ids(panel, (), REVIEW_SIZE, REVIEW_SALT)
    records = []
    for image_id in panel:
        line, source = source_by_id[image_id]
        native_line, native = native_by_id[image_id]
        record = _verify_native_geometry(source, native)
        record["source_line"] = line
        record["native_line"] = native_line
        records.append(record)
    packet = {
        "schema": "native_owner_successor_scale_throughput.confirmation_selection.v1",
        "status": "frozen_cpu_no_model_calls",
        "unit": UNIT,
        "source": _file_binding(SOURCE),
        "native_source": _file_binding(NATIVE_SOURCE),
        "source_images": len(universe),
        "exclusion_counts": {
            "prior_excluded_ids": len(set(int(x) for x in prior["excluded_image_ids"])),
            "previous_fresh256": len(set(int(x) for x in prior["image_ids"])),
            "union_excluded_ids": len(boundary),
            "union_excluded_source_images": len(boundary & set(universe)),
            "eligible_source_images": len(set(universe) - boundary),
        },
        "excluded_image_ids": sorted(boundary),
        "exclusion_sources": exclusion_sources,
        "image_ids": panel,
        "image_ids_sha256": prior_helpers.digest(panel),
        "blind_review_ids": review,
        "blind_review_ids_sha256": prior_helpers.digest(review),
        "panel_salt": PANEL_SALT,
        "blind_review_salt": REVIEW_SALT,
        "selection_rule": "SHA256(salt + decimal image ID), first256 from processed COCO val universe after complete prior exclusion chain plus previous fresh256; blind32 is selected before any outputs; no visual/count/output backfill",
        "records": records,
        "provenance_boundary": "Disjoint from the prior evaluation selection's complete declared exclusion_sources and its previously fresh256, with source/native hashes and physical image paths rechecked. This is not a pretraining/SFT-disjointness claim.",
        "claim_boundary": "CPU identity/materialization freeze only; no model calls, inference ownership, consumer choice, or quality claim.",
    }
    output.mkdir(parents=True, exist_ok=True)
    prior_helpers.publish(output / "confirmation-selection.json", packet)
    return packet


def replay(packet_path: Path = OUTPUT / "confirmation-selection.json") -> dict[str, Any]:
    """Recompute the frozen identity boundary and native materialization checks."""

    packet = _read(packet_path)
    if packet.get("status") != "frozen_cpu_no_model_calls":
        raise ValueError("confirmation packet is not CPU-frozen")
    if packet["source"] != _file_binding(SOURCE) or packet["native_source"] != _file_binding(NATIVE_SOURCE):
        raise ValueError("confirmation source binding changed")
    source_rows, source_by_id = _load_rows(SOURCE)
    _, native_by_id = _load_rows(NATIVE_SOURCE)
    prior = _read(PRIOR)
    boundary, _ = _prior_boundary(prior)
    universe = sorted(source_by_id)
    expected_panel = prior_helpers.select_ids(universe, boundary, PANEL_SIZE, PANEL_SALT)
    expected_review = prior_helpers.select_ids(expected_panel, (), REVIEW_SIZE, REVIEW_SALT)
    if packet["image_ids"] != expected_panel or packet["blind_review_ids"] != expected_review:
        raise ValueError("confirmation IDs are not reproducible from frozen salts")
    if packet["excluded_image_ids"] != sorted(boundary):
        raise ValueError("confirmation exclusion boundary changed")
    if packet["image_ids_sha256"] != prior_helpers.digest(expected_panel):
        raise ValueError("confirmation panel digest mismatch")
    if packet["blind_review_ids_sha256"] != prior_helpers.digest(expected_review):
        raise ValueError("confirmation review digest mismatch")
    records = {int(row["image_id"]): row for row in packet["records"]}
    if set(records) != set(expected_panel):
        raise ValueError("confirmation materialization record set mismatch")
    for image_id in expected_panel:
        _, source = source_by_id[image_id]
        _, native = native_by_id[image_id]
        expected = _verify_native_geometry(source, native)
        stored = records[image_id]
        for key in (
            "physical_image_path",
            "width",
            "height",
            "object_count",
            "source_objects_sha256",
            "native_objects_sha256",
        ):
            if stored.get(key) != expected[key]:
                raise ValueError(f"materialization record mismatch for {image_id}: {key}")
    return {
        "status": "passed",
        "source_images": len(source_rows),
        "excluded_source_images": len(boundary & set(universe)),
        "eligible_source_images": len(set(universe) - boundary),
        "panel_images": len(expected_panel),
        "review_images": len(expected_review),
        "physical_geometry_records": len(records),
        "model_calls": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if args.replay:
        print(json.dumps(replay(args.output / "confirmation-selection.json"), sort_keys=True))
    else:
        packet = freeze(args.output)
        print(json.dumps({"path": str(args.output / 'confirmation-selection.json'), "image_count": len(packet['image_ids']), "review_count": len(packet['blind_review_ids']), "image_ids_sha256": packet['image_ids_sha256'], "blind_review_ids_sha256": packet['blind_review_ids_sha256']}, sort_keys=True))


if __name__ == "__main__":
    main()
