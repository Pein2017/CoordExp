#!/usr/bin/env python3
"""Validate and canonically extract all third-fit native step32 reviews."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable


B = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum")
OUT = B / "third-fit-review-extraction-v1"
REVIEWS = B / "third-fit-owner-reviews-v1"
PACKETS = B / "third-fit-review-packets-v1"
TARGET = B / "target-owners-complete-v4.json"
PARENT = B / "parent16-v4-physical-ledger-v1/ledger.json"
SCORED = B / "third-fit-selectors-v1/scored-step-32.json"
RAW_ROOT = B / "third-fit-v1/readback-recovery/rows"
RULINGS = B / "third-fit-root-rulings-v1/rulings.json"
IMAGE_IDS = (25274, 59571, 99937, 210457, 219546, 323322, 351017, 388795, 417044, 477415, 528944)
PHYSICAL = {"true_unique", "repeat", "false", "unknown", "invalid_output"}
EXTENT = {"reasonable", "wrong", "unknown"}
CLASS = {"verified", "wrong", "unknown"}
CE = {"positive", "mask"}
EXPECTED_TARGET_SHA256 = "b5c6534259a46583466dc5b92ae0b4b7854cc5adc4de8bc84801f3d96bc69f64"
EXPECTED_PARENT_SHA256 = "f0536ed6a5a158f8bf296f1e736a5b0bdbee9d52349179f4b0ef1064bfcfce5e"
PREDICTION_ID = re.compile(r"^p[0-9]+$")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def binding(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    require("/2026-09-14-training-set-completion-curriculum/B/" not in str(path), f"noncanonical /B/ path: {path}")
    return {"path": str(path), "sha256": file_hash(path), "size_bytes": path.stat().st_size}


def validate_declared_binding(value: dict[str, Any], label: str) -> dict[str, Any]:
    require(isinstance(value, dict) and "path" in value and "sha256" in value, f"missing declared binding: {label}")
    actual = binding(Path(value["path"]))
    require(actual["sha256"] == value["sha256"], f"declared hash mismatch: {label}")
    if "size_bytes" in value:
        require(actual["size_bytes"] == value["size_bytes"], f"declared size mismatch: {label}")
    return actual


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical(value))


def flatten_paths(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            result.extend(flatten_paths(item))
        return result
    if isinstance(value, dict):
        result = []
        for item in value.values():
            result.extend(flatten_paths(item))
        return result
    return []


def evidence_id(path_value: str, catalog: dict[str, dict[str, Any]]) -> str:
    path = Path(path_value).resolve(strict=True)
    require("/2026-09-14-training-set-completion-curriculum/B/" not in str(path), f"noncanonical evidence path: {path}")
    key = str(path)
    if key not in catalog:
        catalog[key] = binding(path)
    return key


def declared_evidence_id(value: dict[str, Any], label: str, catalog: dict[str, dict[str, Any]]) -> str:
    actual = validate_declared_binding(value, label)
    catalog[actual["path"]] = actual
    return actual["path"]


def validate_native_hashes(raw: dict[str, Any], image_id: int) -> None:
    require(raw["image_id"] == image_id and raw["checkpoint_step"] == 32, f"native identity: {image_id}")
    require(raw["empty_assistant_prefix"] is True, f"native prefix is not empty: {image_id}")
    require(digest(raw["generated_token_ids"]) == raw["generated_token_ids_sha256"], f"native generated token hash: {image_id}")
    require(digest(raw["prompt_token_ids"]) == raw["prompt_token_ids_sha256"], f"native prompt token hash: {image_id}")
    for key in ("manifest", "model_receipt"):
        validate_declared_binding(raw[key], f"native {key}: {image_id}")


def validate_packet_groups(packet: dict[str, Any], image_id: int) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    require(packet["schema"] == "training_set_completion.third_fit_step32_review_packet.v1", f"packet schema: {image_id}")
    require(packet["image_id"] == image_id and packet["checkpoint_step"] == 32, f"packet identity: {image_id}")
    rows = packet["rendered"]["raw_rows"]
    groups = packet["rendered"]["visual_groups"]
    require(len(rows) == packet["counts"]["raw_row_count"], f"packet row count: {image_id}")
    require(sum(row["status"] == "parsed_valid" for row in rows) == packet["counts"]["valid_prediction_count"], f"packet valid count: {image_id}")
    require(sum(row["status"] != "parsed_valid" for row in rows) == packet["counts"]["raw_invalid_or_dropped_count"], f"packet drop count: {image_id}")
    require(len(groups) == packet["counts"]["visual_group_count"], f"packet group count: {image_id}")
    require([row["generated_order"] for row in rows] == list(range(len(rows))), f"raw generated order is not contiguous: {image_id}")
    require([row["prediction_id"] for row in rows] == [f"p{i}" for i in range(len(rows))], f"raw prediction order mismatch: {image_id}")
    group_by_id = {group["visual_group_id"]: group for group in groups}
    require(len(group_by_id) == len(groups), f"duplicate visual group ID: {image_id}")
    member_ids: list[str] = []
    for group in groups:
        members = group["member_prediction_ids"]
        require(group["member_count"] == len(members) and len(members) > 0, f"group member count: {image_id}:{group['visual_group_id']}")
        member_ids.extend(members)
    raw_ids = [row["prediction_id"] for row in rows]
    require(len(member_ids) == len(set(member_ids)) and set(member_ids) == set(raw_ids), f"visual groups do not partition raw rows: {image_id}")
    for row in rows:
        require(row["visual_group_id"] in group_by_id, f"unknown row group: {image_id}:{row['prediction_id']}")
        require(row["prediction_id"] in group_by_id[row["visual_group_id"]]["member_prediction_ids"], f"row group membership mismatch: {image_id}:{row['prediction_id']}")
    return rows, group_by_id


def validate_packet_sources(packet: dict[str, Any], image_id: int) -> None:
    source = packet["source"]
    target = validate_declared_binding(source["target_catalog_v4"], f"packet target catalog: {image_id}")
    parent = validate_declared_binding(source["parent_v4_physical_ledger"], f"packet parent ledger: {image_id}")
    require(target["path"] == str(TARGET.resolve()) and target["sha256"] == EXPECTED_TARGET_SHA256, f"packet target identity: {image_id}")
    require(parent["path"] == str(PARENT.resolve()) and parent["sha256"] == EXPECTED_PARENT_SHA256, f"packet parent identity: {image_id}")
    for key in ("acquisition_manifest", "original_image", "scored_json"):
        validate_declared_binding(source[key], f"packet source {key}: {image_id}")


def validate_packet_vs_scored(packet: dict[str, Any], scored: dict[str, Any], rows: list[dict[str, Any]], image_id: int) -> None:
    require(digest(scored) == packet["scored_row_sha256"], f"packet scored-row hash: {image_id}")
    source_rows = scored["raw_rows"]
    require(len(source_rows) == len(rows), f"packet/scored row count: {image_id}")
    source_by_id = {row["prediction_id"]: row for row in source_rows}
    require(len(source_by_id) == len(source_rows) and set(source_by_id) == {row["prediction_id"] for row in rows}, f"packet/scored ID partition: {image_id}")
    for row in rows:
        source = source_by_id[row["prediction_id"]]
        require(source["prediction_id"] == row["prediction_id"] and source["generated_order"] == row["generated_order"], f"packet/scored row identity: {image_id}:{row['prediction_id']}")
        require(source["status"] == row["status"], f"packet/scored status: {image_id}:{row['prediction_id']}")
        require(source.get("description") == row.get("description"), f"packet/scored description: {image_id}:{row['prediction_id']}")
        require(source.get("coord_bins_1000") == row.get("coord_bins_1000"), f"packet/scored coords: {image_id}:{row['prediction_id']}")
        require(source.get("bbox_pixel_xyxy") == row.get("raw_bbox_pixel_xyxy"), f"packet/scored bbox: {image_id}:{row['prediction_id']}")
        require(source["raw_span_sha256"] == row["raw_span_sha256"], f"packet/scored span hash: {image_id}:{row['prediction_id']}")


def validate_scored_vs_native(scored: dict[str, Any], raw: dict[str, Any], image_id: int) -> None:
    require(scored["image_id"] == raw["image_id"] and scored["route_id"] == raw["route_id"], f"scored/native identity: {image_id}")
    source = scored["readback_source"]
    for key in (
        "image_id", "route_id", "decode_stop_reason", "empty_assistant_prefix", "executed_media_sha256",
        "generated_token_ids", "generated_token_ids_sha256", "prompt_token_ids",
        "observed_image_grid_thw", "raw_decode_text",
    ):
        require(source[key] == raw[key], f"scored/native mismatch {image_id}:{key}")
    require(scored["decode_stop_reason"] == raw["decode_stop_reason"], f"scored/native stop reason: {image_id}")
    require(scored["token_count"] == len(raw["generated_token_ids"]), f"scored/native token count: {image_id}")


def validate_review_binding(review: dict[str, Any], packet_path: Path, image_id: int) -> None:
    require(review["schema"] == "third_fit_step32_owner_review.v1", f"review schema: {image_id}")
    require(review["status"] == "candidate_ready", f"review status: {image_id}")
    require(review["image_id"] == image_id and review["checkpoint_step"] == 32, f"review identity: {image_id}")
    target = validate_declared_binding(review["target_catalog"], f"review target: {image_id}")
    packet = validate_declared_binding(review["source_packet"], f"review packet: {image_id}")
    require(target["path"] == str(TARGET.resolve()) and target["sha256"] == EXPECTED_TARGET_SHA256, f"review target identity: {image_id}")
    require(packet["path"] == str(packet_path.resolve()) and packet["sha256"] == file_hash(packet_path), f"review packet identity: {image_id}")


def expected_semantics(decision: dict[str, Any], raw_status: str, target_ids: set[str]) -> tuple[bool, dict[str, str]]:
    fixed = decision.get("owner_id") is not None and str(decision["owner_id"]) in target_ids
    physical_geometry = raw_status == "parsed_valid" and decision["physical_status"] in {"true_unique", "repeat"} and decision["extent"] == "reasonable"
    coverage = physical_geometry and fixed
    direct_unique = coverage and decision["physical_status"] == "true_unique"
    ce = {
        "bbox": "positive" if direct_unique else "mask",
        "description": "positive" if direct_unique and decision["class"] == "verified" else "mask",
    }
    return coverage, ce


def validate_decision(decision: dict[str, Any], raw: dict[str, Any], target_ids: set[str], image_id: int) -> None:
    for field in ("prediction_id", "generated_order", "visual_group_id", "owner_id", "physical_status", "extent", "class", "coverage_eligible", "direct_CE", "reason", "evidence_paths"):
        require(field in decision, f"decision missing {field}: {image_id}:{decision.get('prediction_id')}")
    prediction_id = decision["prediction_id"]
    require(decision["generated_order"] == raw["generated_order"], f"decision order: {image_id}:{prediction_id}")
    require(decision["visual_group_id"] == raw["visual_group_id"], f"decision group: {image_id}:{prediction_id}")
    require(decision["physical_status"] in PHYSICAL, f"physical domain: {image_id}:{prediction_id}")
    require(decision["extent"] in EXTENT, f"extent domain: {image_id}:{prediction_id}")
    require(decision["class"] in CLASS, f"class domain: {image_id}:{prediction_id}")
    require(type(decision["coverage_eligible"]) is bool, f"coverage type: {image_id}:{prediction_id}")
    require(set(decision["direct_CE"]) == {"bbox", "description"} and set(decision["direct_CE"].values()) <= CE, f"CE domain: {image_id}:{prediction_id}")
    if raw["status"] != "parsed_valid":
        require(decision["physical_status"] == "invalid_output", f"parser-dropped row not invalid_output: {image_id}:{prediction_id}")
    if decision["physical_status"] == "invalid_output":
        require(raw["status"] != "parsed_valid", f"parsed-valid row marked invalid_output: {image_id}:{prediction_id}")
    expected_coverage, expected_ce = expected_semantics(decision, raw["status"], target_ids)
    require(decision["coverage_eligible"] == expected_coverage, f"coverage contract: {image_id}:{prediction_id}")
    require(decision["direct_CE"] == expected_ce, f"CE contract: {image_id}:{prediction_id} {decision['direct_CE']} != {expected_ce}")


def recompute_summary(rows: list[dict[str, Any]], target_ids: set[str], parent_ids: set[str], scored: dict[str, Any]) -> dict[str, Any]:
    covered = {str(row["owner_id"]) for row in rows if row["coverage_eligible"]}
    missing = target_ids - covered
    retained = covered & parent_ids
    lost = parent_ids - covered
    new = covered - parent_ids
    return {
        "target_owner_count": len(target_ids), "covered_owner_ids": sorted(covered), "missing_owner_ids": sorted(missing),
        "covered_owner_count": len(covered), "missing_owner_count": len(missing),
        "physical_repeat_row_count": sum(row["physical_status"] == "repeat" for row in rows),
        "confirmed_false_row_count": sum(row["physical_status"] == "false" for row in rows),
        "physical_unknown_row_count": sum(row["physical_status"] == "unknown" for row in rows),
        "invalid_output_row_count": sum(row["physical_status"] == "invalid_output" for row in rows),
        "class_wrong_row_count": sum(row["class"] == "wrong" for row in rows),
        "class_unknown_row_count": sum(row["class"] == "unknown" for row in rows),
        "raw_stop_reason": scored["decode_stop_reason"], "cap_debt": scored["cap_debt"],
        "parent_covered_owner_ids": sorted(parent_ids), "retained_parent_owner_ids": sorted(retained),
        "lost_parent_owner_ids": sorted(lost), "newly_covered_fixed_owner_ids": sorted(new),
    }


def validate_review_summary(summary: dict[str, Any], rows: list[dict[str, Any]], target_ids: set[str], parent_ids: set[str], scored: dict[str, Any], image_id: int) -> dict[str, Any]:
    checks = recompute_summary(rows, target_ids, parent_ids, scored)
    for key, expected in checks.items():
        require(key in summary, f"review summary missing {key}: {image_id}")
        actual = summary[key]
        if key.endswith("_owner_ids"):
            actual = sorted(map(str, actual))
        require(actual == expected, f"review summary mismatch {image_id}:{key}")
    return checks


def apply_root_patch(row: dict[str, Any], patch: dict[str, Any], raw: dict[str, Any], target_ids: set[str], ruling_binding: dict[str, Any], catalog: dict[str, dict[str, Any]]) -> dict[str, Any]:
    allowed = {"owner_id", "physical_status", "extent", "class", "coverage_eligible", "direct_CE"}
    updates = patch["updates"]
    require(set(updates) <= allowed, f"unsupported root patch fields: {sorted(set(updates) - allowed)}")
    require("reason" in patch and patch["reason"], f"root patch lacks reason: {row['image_id']}:{row['prediction_id']}")
    result = dict(row)
    result["original_candidate_fields"] = {key: row.get(key) for key in sorted(allowed)}
    result.update(updates)
    patch_evidence_ids = []
    for index, item in enumerate(patch.get("evidence", [])):
        patch_evidence_ids.append(declared_evidence_id(item, f"root patch evidence {row['image_id']}:{row['prediction_id']}:{index}", catalog))
    result["root_ruling"] = {
        "reason": patch["reason"], "updates": updates, "evidence_ids": patch_evidence_ids,
        "source_ruling": ruling_binding,
    }
    require(result["physical_status"] in PHYSICAL and result["extent"] in EXTENT and result["class"] in CLASS, f"root patch domain: {row['image_id']}:{row['prediction_id']}")
    expected_coverage, expected_ce = expected_semantics(result, raw["status"], target_ids)
    require(result["coverage_eligible"] == expected_coverage, f"root patch coverage: {row['image_id']}:{row['prediction_id']}")
    require(result["direct_CE"] == expected_ce, f"root patch CE: {row['image_id']}:{row['prediction_id']}")
    return result


def candidate_records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        require(all(isinstance(item, dict) for item in value), "new candidate list contains non-object")
        return value
    if isinstance(value, dict):
        if set(value) == {"admission_pending_root", "note"}:
            require(value["admission_pending_root"] == [], "unsupported nonempty candidate metadata")
            return []
        if all(isinstance(item, dict) for item in value.values()):
            return list(value.values())
        return [value]
    raise ValueError("new_owner_candidates must be a list or object")


def candidate_prediction_ids(candidate: dict[str, Any]) -> set[str]:
    result: set[str] = set()
    for key, value in candidate.items():
        if "prediction" not in key and "proposal" not in key:
            continue
        for text in flatten_paths(value):
            if PREDICTION_ID.fullmatch(text):
                result.add(text)
            elif ":p" in text:
                tail = text.rsplit(":", 1)[-1]
                if PREDICTION_ID.fullmatch(tail):
                    result.add(tail)
    return result


def candidate_evidence_paths(candidate: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for key, value in candidate.items():
        if "evidence" in key or key.endswith("_path") or key.endswith("_paths"):
            result.extend(path for path in flatten_paths(value) if path.startswith("/"))
    return result


def extract() -> dict[str, Any]:
    review_paths = {image_id: REVIEWS / f"image-{image_id:012d}" / "review.json" for image_id in IMAGE_IDS}
    missing_reviews = [image_id for image_id, path in review_paths.items() if not path.is_file()]
    require(not missing_reviews, f"missing review.json for image IDs: {missing_reviews}")

    require(file_hash(TARGET) == EXPECTED_TARGET_SHA256, "fixed v4 target catalog hash mismatch")
    require(file_hash(PARENT) == EXPECTED_PARENT_SHA256, "parent v4 physical ledger hash mismatch")
    target = json.loads(TARGET.read_text())
    parent = json.loads(PARENT.read_text())
    scored = json.loads(SCORED.read_text())
    rulings = json.loads(RULINGS.read_text())
    require(target["atomic_target_count"] == len(target["records"]) == 232, "fixed v4 target count")
    require(parent["summary"]["qualified_covered_owner_count"] == 155, "parent v4 covered count")
    require(parent["summary"]["target_owner_count"] == 232, "parent v4 target count")
    require(scored["expected_image_ids"] == list(IMAGE_IDS) and scored["missing_image_ids"] == [], "scored all-image partition")
    require([row["image_id"] for row in scored["rows"]] == list(IMAGE_IDS), "scored image order")

    target_by_image: dict[int, set[str]] = defaultdict(set)
    for row in target["records"]:
        target_by_image[int(row["image_id"])].add(str(row["owner_id"]))
    require(set(target_by_image) == set(IMAGE_IDS) and sum(map(len, target_by_image.values())) == 232, "target all-image partition")
    parent_by_image = {int(row["image_id"]): set(map(str, row["covered_owner_ids"])) for row in parent["images"]}
    require(set(parent_by_image) == set(IMAGE_IDS) and sum(map(len, parent_by_image.values())) == 155, "parent all-image partition")
    scored_by_image = {int(row["image_id"]): row for row in scored["rows"]}
    require(rulings["status"] == "lead-accepted", "third-fit root rulings are not lead-accepted")
    ruling_binding = binding(RULINGS)
    patch_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for patch in rulings["patches"]:
        key = (int(patch["image_id"]), str(patch["prediction_id"]))
        require(key not in patch_by_key, f"duplicate root patch: {key}")
        patch_by_key[key] = patch

    evidence_catalog: dict[str, dict[str, Any]] = {}
    flat: list[dict[str, Any]] = []
    full_results: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    extracted_candidates: list[dict[str, Any]] = []
    review_sources: list[dict[str, Any]] = []
    packet_sources: list[dict[str, Any]] = []
    native_sources: list[dict[str, Any]] = []
    notes_sources: list[dict[str, Any]] = []

    for image_id in IMAGE_IDS:
        review_path = review_paths[image_id]
        packet_path = PACKETS / f"image-{image_id:012d}" / "packet.json"
        raw_path = RAW_ROOT / f"step-00032-image-{image_id:012d}.json"
        review = json.loads(review_path.read_text())
        packet = json.loads(packet_path.read_text())
        native = json.loads(raw_path.read_text())
        scored_row = scored_by_image[image_id]

        validate_review_binding(review, packet_path, image_id)
        validate_packet_sources(packet, image_id)
        packet_rows, group_by_id = validate_packet_groups(packet, image_id)
        validate_packet_vs_scored(packet, scored_row, packet_rows, image_id)
        validate_native_hashes(native, image_id)
        validate_scored_vs_native(scored_row, native, image_id)
        require(review["raw_row_count"] == len(review["decisions"]) == len(packet_rows), f"review row count: {image_id}")
        raw_by_id = {row["prediction_id"]: row for row in packet_rows}
        decision_by_id = {row["prediction_id"]: row for row in review["decisions"]}
        require(len(decision_by_id) == len(review["decisions"]) and set(decision_by_id) == set(raw_by_id), f"review/raw ID partition: {image_id}")
        require([row["prediction_id"] for row in review["decisions"]] == [row["prediction_id"] for row in packet_rows], f"review/raw decision order: {image_id}")

        review_binding = binding(review_path)
        packet_binding = binding(packet_path)
        native_binding = binding(raw_path)
        review_sources.append(review_binding)
        packet_sources.append(packet_binding)
        native_sources.append(native_binding)
        notes_path = review_path.parent / "review-notes.jsonl"
        notes_binding = None
        if notes_path.is_file():
            notes_lines = [json.loads(line) for line in notes_path.read_text().splitlines() if line.strip()]
            require(notes_lines, f"empty review notes: {image_id}")
            notes_binding = binding(notes_path)
            notes_binding["record_count"] = len(notes_lines)
            notes_sources.append(notes_binding)

        rendered = packet["rendered"]
        image_evidence = {
            "original_image": declared_evidence_id(rendered["original_image"], f"rendered original: {image_id}", evidence_catalog),
            "target_catalog_overlay": declared_evidence_id(rendered["target_catalog_overlay"], f"rendered target overlay: {image_id}", evidence_catalog),
            "raw_generated_overlay": declared_evidence_id(rendered["raw_generated_overlay"], f"rendered raw overlay: {image_id}", evidence_catalog),
        }

        image_flat: list[dict[str, Any]] = []
        image_source_flat: list[dict[str, Any]] = []
        for packet_row in packet_rows:
            prediction_id = packet_row["prediction_id"]
            source_decision = decision_by_id[prediction_id]
            validate_decision(source_decision, packet_row, target_by_image[image_id], image_id)
            group = group_by_id[packet_row["visual_group_id"]]
            context_id = declared_evidence_id(group["context_crop"], f"context crop {image_id}:{prediction_id}", evidence_catalog)
            tight_id = declared_evidence_id(group["tight_crop"], f"tight crop {image_id}:{prediction_id}", evidence_catalog)
            require(packet_row["shared_crop_paths"] == {"context": context_id, "tight": tight_id}, f"row crop paths: {image_id}:{prediction_id}")
            reviewer_evidence_ids = [evidence_id(path, evidence_catalog) for path in flatten_paths(source_decision["evidence_paths"])]
            row = {
                "schema": "training_set_completion.third_fit_step32_flat_decision.v1",
                "image_id": image_id, "checkpoint_step": 32,
                "proposal_id": f"third-fit:step32:image-{image_id:012d}:{prediction_id}",
                "prediction_id": prediction_id, "generated_order": packet_row["generated_order"],
                "visual_group_id": packet_row["visual_group_id"], "raw_status": packet_row["status"],
                "raw_span_sha256": packet_row["raw_span_sha256"], "raw_source_row_sha256": packet_row["raw_source_row_sha256"],
                "owner_id": source_decision.get("owner_id"), "physical_status": source_decision["physical_status"],
                "extent": source_decision["extent"], "class": source_decision["class"],
                "coverage_eligible": source_decision["coverage_eligible"], "direct_CE": source_decision["direct_CE"],
                "reason": source_decision["reason"], "source_review_decision_sha256": digest(source_decision),
            }
            image_source_flat.append(row)
            patch = patch_by_key.get((image_id, prediction_id))
            if patch is not None:
                row = apply_root_patch(row, patch, packet_row, target_by_image[image_id], ruling_binding, evidence_catalog)
            flat.append(row)
            image_flat.append(row)
            full_results.append({
                **row,
                "schema": "training_set_completion.third_fit_step32_full_review_result.v1",
                "raw": {
                    "description": packet_row.get("description"), "coord_bins_1000": packet_row.get("coord_bins_1000"),
                    "bbox_pixel_xyxy": packet_row.get("raw_bbox_pixel_xyxy"), "drop_reason": packet_row.get("drop_reason"),
                },
                "visual_evidence_ids": {**image_evidence, "context_crop": context_id, "tight_crop": tight_id},
                "reviewer_evidence_ids": reviewer_evidence_ids,
                "source_bindings": {"review": review_binding, "packet": packet_binding, "native_readback": native_binding, "review_notes": notes_binding},
            })

        validate_review_summary(review["summary"], image_source_flat, target_by_image[image_id], parent_by_image[image_id], scored_row, image_id)
        recomputed = recompute_summary(image_flat, target_by_image[image_id], parent_by_image[image_id], scored_row)
        source_candidates = candidate_records(review["new_owner_candidates"])
        for index, candidate in enumerate(source_candidates):
            referred = candidate_prediction_ids(candidate)
            require(referred <= set(raw_by_id), f"new candidate references unknown prediction IDs: {image_id}:{index}:{sorted(referred - set(raw_by_id))}")
            candidate_evidence = [evidence_id(path, evidence_catalog) for path in candidate_evidence_paths(candidate)]
            extracted_candidates.append({
                "schema": "training_set_completion.third_fit_step32_new_owner_candidate_reference.v1",
                "image_id": image_id, "candidate_index": index, "status": "not_admitted_fixed_v4_denominator_unchanged",
                "source_candidate": candidate, "referenced_prediction_ids": sorted(referred),
                "evidence_ids": candidate_evidence, "source_review": review_binding,
            })
        images.append({
            "image_id": image_id, **recomputed,
            "raw_row_count": len(image_flat), "parsed_valid_row_count": sum(row["raw_status"] == "parsed_valid" for row in image_flat),
            "parser_drop_count": sum(row["raw_status"] != "parsed_valid" for row in image_flat),
            "physical_status_row_counts": dict(sorted(Counter(row["physical_status"] for row in image_flat).items())),
            "extent_row_counts": dict(sorted(Counter(row["extent"] for row in image_flat).items())),
            "class_row_counts": dict(sorted(Counter(row["class"] for row in image_flat).items())),
            "positive_bbox_row_count": sum(row["direct_CE"]["bbox"] == "positive" for row in image_flat),
            "positive_description_row_count": sum(row["direct_CE"]["description"] == "positive" for row in image_flat),
            "new_owner_candidate_count": len(source_candidates),
            "native_readback": {"token_count": scored_row["token_count"], "decode_stop_reason": scored_row["decode_stop_reason"], "natural_eos": scored_row["natural_eos"], "capped": scored_row["capped"], "cap_debt": scored_row["cap_debt"]},
            "source_bindings": {"review": review_binding, "packet": packet_binding, "native_readback": native_binding, "review_notes": notes_binding},
        })

    require(len(flat) == len(full_results) == 327, f"all-row extraction count: {len(flat)}")
    require(sum(row["raw_status"] == "parsed_valid" for row in flat) == 322, "parsed-valid count")
    require(sum(row["raw_status"] != "parsed_valid" for row in flat) == 5, "parser-drop count")
    require({(row["image_id"], row["prediction_id"]) for row in flat} == {(row["image_id"], row["prediction_id"]) for row in full_results}, "full-result/decision partition")
    actual_keys = {(row["image_id"], row["prediction_id"]) for row in flat}
    require(set(patch_by_key) <= actual_keys, f"root patch targets absent rows: {sorted(set(patch_by_key) - actual_keys)}")
    covered_total = sum(row["covered_owner_count"] for row in images)
    missing_total = sum(row["missing_owner_count"] for row in images)
    require(covered_total + missing_total == 232, "fixed-v4 cumulative partition")

    OUT.mkdir(parents=True, exist_ok=True)
    decisions_path = OUT / "decisions.jsonl"
    full_results_path = OUT / "full-review-results.jsonl"
    candidates_path = OUT / "new-owner-candidate-references.jsonl"
    decisions_path.write_bytes(b"".join(canonical(row) for row in flat))
    full_results_path.write_bytes(b"".join(canonical(row) for row in full_results))
    candidates_path.write_bytes(b"".join(canonical(row) for row in extracted_candidates))
    evidence_path = OUT / "evidence-catalog.json"
    write_json(evidence_path, {
        "schema": "training_set_completion.third_fit_step32_evidence_catalog.v1",
        "binding_count": len(evidence_catalog), "bindings": [evidence_catalog[path] for path in sorted(evidence_catalog)],
    })
    ledger_path = OUT / "perimageledger.json"
    ledger = {
        "schema": "training_set_completion.third_fit_step32_review_extraction.v1", "status": "candidate_ready",
        "scope": {"checkpoint_step": 32, "image_ids": list(IMAGE_IDS), "fixed_v4_target_count": 232, "parent_v4_covered_count": 155},
        "images": images,
        "summary": {
            "image_count": 11, "raw_row_count": 327, "parsed_valid_row_count": 322, "parser_drop_count": 5,
            "physical_status_row_counts": dict(sorted(Counter(row["physical_status"] for row in flat).items())),
            "extent_row_counts": dict(sorted(Counter(row["extent"] for row in flat).items())),
            "class_row_counts": dict(sorted(Counter(row["class"] for row in flat).items())),
            "positive_bbox_row_count": sum(row["direct_CE"]["bbox"] == "positive" for row in flat),
            "positive_description_row_count": sum(row["direct_CE"]["description"] == "positive" for row in flat),
            "covered_physical_owner_count": covered_total, "missing_physical_owner_count": missing_total,
            "parent_covered_owner_count": 155,
            "retained_parent_owner_count": sum(len(row["retained_parent_owner_ids"]) for row in images),
            "lost_parent_owner_count": sum(len(row["lost_parent_owner_ids"]) for row in images),
            "newly_covered_fixed_owner_count": sum(len(row["newly_covered_fixed_owner_ids"]) for row in images),
            "new_owner_candidate_reference_count": len(extracted_candidates),
            "root_patch_count": len(patch_by_key),
            "fixed_v4_denominator_unchanged": True,
        },
        "source_bindings": {
            "target_catalog_v4": binding(TARGET), "parent_v4_physical_ledger": binding(PARENT), "scored_step32": binding(SCORED),
            "root_rulings": ruling_binding,
            "reviews": review_sources, "packets": packet_sources, "native_readbacks": native_sources, "review_notes": notes_sources,
        },
        "artifacts": {
            "flat_decisions": binding(decisions_path), "full_review_results": binding(full_results_path),
            "evidence_catalog": binding(evidence_path), "new_owner_candidate_references": binding(candidates_path),
        },
        "policy": {
            "coverage": "Parsed-valid physical identity (true_unique or repeat), reasonable raw extent, and fixed-v4 owner membership; class-independent.",
            "direct_CE": "Repeat, pending-new, unknown, false, invalid, and wrong-extent rows are fully masked. Fixed-v4 true-unique reasonable rows receive bbox positive; description is positive only for class verified.",
            "new_candidates": "References are preserved for separate root v5 annotation work and never modify the fixed-v4 denominator here.",
            "no_gt_iou_physical_proxy": True,
        },
    }
    write_json(ledger_path, ledger)
    receipt = {
        "schema": "training_set_completion.third_fit_step32_review_extraction_receipt.v1", "status": "candidate_ready",
        "counts": {"images": 11, "raw_rows": 327, "parsed_valid": 322, "parser_drops": 5, "fixed_v4_targets": 232, "parent_covered": 155, "covered": covered_total, "missing": missing_total, "new_owner_candidate_references": len(extracted_candidates)},
        "per_image_ledger": binding(ledger_path), "flat_decisions": binding(decisions_path),
        "full_review_results": binding(full_results_path), "evidence_catalog": binding(evidence_path),
        "new_owner_candidate_references": binding(candidates_path), "validator": binding(Path(__file__)),
    }
    write_json(OUT / "receipt.json", receipt)
    return receipt


def main() -> None:
    print(json.dumps(extract(), indent=2))


if __name__ == "__main__":
    main()
