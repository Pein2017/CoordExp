#!/usr/bin/env python3
"""Validate and canonically extract all second-fit step16 owner reviews."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

B = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum")
OUT = B / "second-fit-review-extraction-v1"
REVIEWS = B / "second-fit-owner-reviews-v1"
PACKETS = B / "second-fit-review-packets-v1"
TARGET = B / "target-owners-complete-v3.json"
PARENT = B / "parent16-v3-physical-ledger-v2/ledger.json"
PARENT_ACCEPT = B / "parent16-v3-physical-ledger-v2/root-acceptance.json"
PARENT_CLASSES = B / "parent16-v3-physical-ledger-v2/root-effective-class-decisions.json"
READBACK = B / "second-fit-readback-recovery-v1/readback-step-16.json"
RAW_ROOT = B / "second-fit-readback-recovery-v1/rows"
RULINGS = B / "second-fit-root-rulings-v1/rulings.json"
STEP64_PACKETS = B / "second-fit-step64-introduced-owner-packets-v1"
IMAGE_IDS = (25274, 59571, 99937, 210457, 219546, 323322, 351017, 388795, 417044, 477415, 528944)
STEP64_IMAGE_IDS = (25274, 59571, 99937, 219546, 351017, 388795, 417044, 477415, 528944)
PHYSICAL = {"true_unique", "repeat", "false", "unknown", "invalid_output"}
EXTENT = {"reasonable", "wrong", "unknown"}
CLASS = {"verified", "wrong", "unknown"}


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


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical(value))


def packet_paths(image_id: int) -> tuple[Path, Path]:
    return (
        REVIEWS / f"image-{image_id:012d}" / "review.json",
        PACKETS / f"image-{image_id:012d}" / "packet.json",
    )


def evidence_binding(path_value: str, catalog: dict[str, dict[str, Any]]) -> str:
    path = Path(path_value).resolve(strict=True)
    require("/2026-09-14-training-set-completion-curriculum/B/" not in str(path), f"noncanonical evidence path: {path}")
    key = str(path)
    if key not in catalog:
        catalog[key] = binding(path)
    return key


def packet_evidence(row: dict[str, Any], packet: dict[str, Any], catalog: dict[str, dict[str, Any]]) -> list[str]:
    rendered = packet["rendered"]
    values = [
        rendered["original_image"]["path"], rendered["target_catalog_overlay"]["path"],
        rendered["raw_generated_overlay"]["path"], row["shared_crop_paths"]["tight"],
        row["shared_crop_paths"]["context"],
    ]
    return [evidence_binding(value, catalog) for value in values]


def validate_source_decision(decision: dict[str, Any], raw: dict[str, Any], target_ids: set[str]) -> None:
    for field in ("prediction_id", "generated_order", "visual_group_id", "owner_id", "physical_status", "extent", "class", "coverage_eligible", "direct_CE", "reason", "evidence_paths"):
        require(field in decision, f"decision missing {field}: {decision.get('prediction_id')}")
    require(decision["physical_status"] in PHYSICAL, f"physical domain: {decision['prediction_id']}")
    require(decision["extent"] in EXTENT, f"extent domain: {decision['prediction_id']}")
    require(decision["class"] in CLASS, f"class domain: {decision['prediction_id']}")
    require(type(decision["coverage_eligible"]) is bool, f"coverage bool: {decision['prediction_id']}")
    require(set(decision["direct_CE"]) == {"bbox", "description"}, f"CE fields: {decision['prediction_id']}")
    require(decision["direct_CE"]["bbox"] in {"positive", "mask"} and decision["direct_CE"]["description"] in {"positive", "mask"}, f"CE domains: {decision['prediction_id']}")
    require(decision["generated_order"] == raw["generated_order"], f"order mismatch: {decision['prediction_id']}")
    require(decision["visual_group_id"] == raw["visual_group_id"], f"group mismatch: {decision['prediction_id']}")
    parsed = raw["status"] == "parsed_valid"
    physical_support = parsed and decision["physical_status"] in {"true_unique", "repeat"} and decision["extent"] == "reasonable" and str(decision.get("owner_id")) in target_ids
    unique_positive = parsed and decision["physical_status"] == "true_unique" and decision["extent"] == "reasonable" and str(decision.get("owner_id")) in target_ids
    expected_ce = {"bbox": "positive" if unique_positive else "mask", "description": "positive" if unique_positive and decision["class"] == "verified" else "mask"}
    require(decision["direct_CE"] == expected_ce, f"source CE contract mismatch {decision['prediction_id']}: {decision['direct_CE']} != {expected_ce}")
    if decision["coverage_eligible"]:
        require(physical_support, f"source coverage asserted without physical support: {decision['prediction_id']}")


def apply_rulings(decision: dict[str, Any], patches: dict[tuple[int, str], dict[str, Any]], catalog: dict[str, dict[str, Any]]) -> dict[str, Any]:
    key = (int(decision["image_id"]), decision["prediction_id"])
    result = dict(decision)
    patch = patches.get(key)
    if patch:
        allowed = {"owner_id", "physical_status", "extent", "class", "coverage_eligible", "direct_CE"}
        require(set(patch["updates"]) <= allowed, f"unsupported ruling keys: {set(patch['updates']) - allowed}")
        result.update(patch["updates"])
        result["root_ruling"] = {
            "reason": patch["reason"], "updates": patch["updates"],
            "evidence_ids": [evidence_binding(item["path"], catalog) for item in patch.get("evidence", [])],
        }
    return result


def semantic_projection(decision: dict[str, Any], target_ids: set[str], raw_status: str) -> dict[str, Any]:
    parsed = raw_status == "parsed_valid"
    physical_support = parsed and decision["physical_status"] in {"true_unique", "repeat"} and decision["extent"] == "reasonable" and str(decision.get("owner_id")) in target_ids
    unique_positive = parsed and decision["physical_status"] == "true_unique" and decision["extent"] == "reasonable" and str(decision.get("owner_id")) in target_ids
    expected_ce = {
        "bbox": "positive" if unique_positive else "mask",
        "description": "positive" if unique_positive and decision["class"] == "verified" else "mask",
    }
    # Root rulings may carry these derived values; they must agree with the canonical projection.
    if "root_ruling" in decision:
        updates = decision["root_ruling"]["updates"]
        if "coverage_eligible" in updates: require(bool(updates["coverage_eligible"]) == physical_support, f"ruling coverage mismatch {decision['proposal_id']}")
        if "direct_CE" in updates: require(updates["direct_CE"] == expected_ce, f"ruling CE mismatch {decision['proposal_id']}")
    result = dict(decision)
    result["coverage_eligible"] = physical_support
    result["direct_CE"] = expected_ce
    result["eligibility_checks"] = {
        "parser_valid": parsed, "physical_identity": decision["physical_status"] in {"true_unique", "repeat"},
        "extent_reasonable": decision["extent"] == "reasonable", "atomic_target_owner": str(decision.get("owner_id")) in target_ids,
        "unique_for_direct_CE": decision["physical_status"] == "true_unique",
    }
    return result


def corrected_parent(parent: dict[str, Any], rulings: dict[str, Any]) -> dict[int, set[str]]:
    patches = {(int(p["image_id"]), p["prediction_id"]): p for p in rulings["patches"] if p["phase"] == "first_fit16_parent"}
    covered_by_image: dict[int, set[str]] = {}
    for image in parent["images"]:
        image_id = int(image["image_id"]); target_ids = set(image["target_owner_ids"]); covered = set()
        for row in image["all_step16_rows"]:
            effective = dict(row)
            patch = patches.get((image_id, row["prediction_id"]))
            if patch: effective.update(patch["updates"])
            support = effective["raw_status"] == "parsed_valid" and effective["physical_status"] in {"true_unique", "repeat"} and effective["effective_extent"] == "reasonable" and str(effective.get("effective_owner_id")) in target_ids
            # Root patches use `extent`; this is the authoritative post-ledger correction.
            if patch:
                support = effective["raw_status"] == "parsed_valid" and effective["physical_status"] in {"true_unique", "repeat"} and patch["updates"].get("extent", effective["effective_extent"]) == "reasonable" and str(effective.get("effective_owner_id")) in target_ids
            if support: covered.add(str(effective["effective_owner_id"]))
        covered_by_image[image_id] = covered
    require(sum(map(len, covered_by_image.values())) == 153, "corrected parent coverage must be 153")
    return covered_by_image


def load_supplement(image_id: int, evidence_catalog: dict[str, dict[str, Any]]) -> dict[str, Any]:
    path = REVIEWS / f"image-{image_id:012d}" / "supplemental-step64.json"
    data = json.loads(path.read_text())
    source = STEP64_PACKETS / f"image-{image_id:012d}" / "packet.json"
    packet = json.loads(source.read_text()); candidate = packet["candidate_groups"][0]
    # Reviewers use small image-local schemas; project the same required fields.
    decision = data.get("decision", {})
    semantic = data.get("selected_owner_decision", {})
    reference = data.get("selected_gt_reference", data.get("selected_reference", data.get("selected_gt_owner", data.get("target_owner", {}))))
    raw = data.get("raw_proposal", data.get("selected_candidate", semantic))
    owner_value = decision.get("owner_id", semantic.get("owner_id", reference.get("owner_id", data.get("selected_owner_id"))))
    require(owner_value is not None, f"step64 supplement lacks owner: {image_id}")
    owner = str(owner_value)
    prediction = decision.get("prediction_id", semantic.get("prediction_id", raw.get("prediction_id", candidate["members"][0]["prediction_id"])))
    order = int(decision.get("generated_order", semantic.get("generated_order", raw.get("generated_order", candidate["members"][0]["generated_order"]))))
    physical = decision.get("physical_status", semantic.get("physical_status"))
    extent = decision.get("extent", semantic.get("extent"))
    klass = decision.get("class", semantic.get("class"))
    require(owner == str(packet["selected_gt_reference"]["owner_plan"]["owner_id"]), f"step64 owner mismatch: {image_id}")
    require(prediction == candidate["members"][0]["prediction_id"] and order == candidate["members"][0]["generated_order"], f"step64 raw identity mismatch: {image_id}")
    require(physical in PHYSICAL and extent in EXTENT and klass in CLASS, f"step64 decision domains: {image_id}")
    for item in packet["visual_evidence"].values(): evidence_binding(item["path"], evidence_catalog)
    for item in candidate["shared_visual_evidence"].values(): evidence_binding(item["path"], evidence_catalog)
    context = packet["output_context_do_not_mistake_for_clean_output"]
    return {
        "image_id": image_id, "selected_owner_id": owner, "prediction_id": prediction, "generated_order": order,
        "physical_status": physical, "extent": extent, "class": klass,
        "qualified_physical_geometry": physical in {"true_unique", "repeat"} and extent == "reasonable",
        "coord_bins_1000": candidate["coord_bins_1000"], "selected_reference_iou": candidate["selected_reference_iou"],
        "before_first_anomaly": context["first_observable_anomaly_generated_order"] is None or order < context["first_observable_anomaly_generated_order"],
        "before_sustained_repeat_or_collapse": context["sustained_repeat_or_collapse_generated_order"] is None or order < context["sustained_repeat_or_collapse_generated_order"],
        "output_capped": context["capped"], "output_stop_reason": context["stop_reason"],
        "first_anomaly_generated_order": context["first_observable_anomaly_generated_order"],
        "sustained_repeat_or_collapse_generated_order": context["sustained_repeat_or_collapse_generated_order"],
        "source_supplement": binding(path), "source_packet": binding(source),
        "claim_boundary": "Selected-owner check only; no full step64 coverage or output-cleanliness claim.",
    }


def extract() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    target = json.loads(TARGET.read_text()); parent = json.loads(PARENT.read_text()); readback = json.loads(READBACK.read_text()); rulings = json.loads(RULINGS.read_text())
    require(rulings["status"] == "lead-accepted" and rulings["fixed_v3_target_count"] == 228, "root rulings not accepted")
    require(file_hash(RULINGS) == "4e0899818afa35217a20e45e15be4fecce32c8a1396d1dbd9e1ba9cd6853ae0a", "root rulings hash mismatch")
    target_by_image: dict[int, set[str]] = defaultdict(set)
    for row in target["records"]: target_by_image[int(row["image_id"])].add(str(row["owner_id"]))
    require(sum(map(len, target_by_image.values())) == 228, "target v3 count")
    parent_by_image = corrected_parent(parent, rulings)
    second_patches = {(int(p["image_id"]), p["prediction_id"]): p for p in rulings["patches"] if p["phase"] == "second_fit16"}
    require(len(second_patches) == 2, "expected exactly two second-fit root patches")
    saved_by_image = {int(row["image_id"]): row for row in readback["rows"]}
    evidence_catalog: dict[str, dict[str, Any]] = {}
    review_sources = []; packet_sources = []; raw_sources = []
    flat = []; images = []
    for image_id in IMAGE_IDS:
        review_path, packet_path = packet_paths(image_id)
        review = json.loads(review_path.read_text()); packet = json.loads(packet_path.read_text())
        require(int(review["image_id"]) == image_id and int(review["checkpoint_step"]) == 16, f"review identity: {image_id}")
        require(packet["image_id"] == image_id and packet["dose_step"] == 16, f"packet identity: {image_id}")
        raw_rows = packet["rendered"]["raw_rows"]; decisions = review["decisions"]
        require(len(raw_rows) == len(decisions) == int(packet["counts"]["raw_row_count"]), f"row count: {image_id}")
        raw_by_id = {row["prediction_id"]: row for row in raw_rows}; decision_by_id = {row["prediction_id"]: row for row in decisions}
        require(len(raw_by_id) == len(raw_rows) and set(raw_by_id) == set(decision_by_id), f"raw decision ID partition: {image_id}")
        target_ids = target_by_image[image_id]
        review_sources.append(binding(review_path)); packet_sources.append(binding(packet_path))
        raw_path = RAW_ROOT / f"step-00016-image-{image_id:012d}.json"; raw_data = json.loads(raw_path.read_text()); saved = saved_by_image[image_id]
        for key in ("generated_token_ids_sha256", "decode_stop_reason", "route_id", "image_id"):
            require(raw_data[key] == saved[key], f"readback identity mismatch {image_id}:{key}")
        raw_sources.append(binding(raw_path))
        for raw in raw_rows:
            source = decision_by_id[raw["prediction_id"]]
            validate_source_decision(source, raw, target_ids)
            decision = {
                "schema": "training_set_completion.second_fit_step16_flat_decision.v1",
                "image_id": image_id, "checkpoint_step": 16,
                "proposal_id": f"second-fit:step16:image-{image_id:012d}:{raw['prediction_id']}",
                "prediction_id": raw["prediction_id"], "generated_order": raw["generated_order"],
                "visual_group_id": raw["visual_group_id"], "raw_status": raw["status"],
                "raw_span_sha256": raw["raw_span_sha256"], "owner_id": source.get("owner_id"),
                "physical_status": source["physical_status"], "extent": source["extent"], "class": source["class"],
                "coverage_eligible": source["coverage_eligible"], "direct_CE": source["direct_CE"],
                "reason": source["reason"], "evidence_ids": packet_evidence(raw, packet, evidence_catalog),
                "source_review_decision_sha256": digest(source),
            }
            decision = apply_rulings(decision, second_patches, evidence_catalog)
            decision = semantic_projection(decision, target_ids, raw["status"])
            flat.append(decision)
        image_rows = [row for row in flat if row["image_id"] == image_id]
        covered = sorted({str(row["owner_id"]) for row in image_rows if row["coverage_eligible"]})
        missing = sorted(target_ids - set(covered)); parent_covered = parent_by_image[image_id]
        token_ids = raw_data["generated_token_ids"]
        natural_eos = raw_data["decode_stop_reason"] == "im_end" and bool(token_ids) and token_ids[-1] == 151645
        images.append({
            "image_id": image_id, "target_owner_count": len(target_ids), "covered_owner_ids": covered,
            "missing_owner_ids": missing, "covered_owner_count": len(covered), "missing_owner_count": len(missing),
            "parent_corrected_covered_owner_ids": sorted(parent_covered),
            "retained_parent_owner_ids": sorted(parent_covered & set(covered)),
            "lost_parent_owner_ids": sorted(parent_covered - set(covered)),
            "newly_covered_fixed_owner_ids": sorted(set(covered) - parent_covered),
            "raw_row_count": len(image_rows), "parser_drop_count": sum(r["raw_status"] != "parsed_valid" for r in image_rows),
            "physical_status_row_counts": dict(sorted(Counter(r["physical_status"] for r in image_rows).items())),
            "extent_row_counts": dict(sorted(Counter(r["extent"] for r in image_rows).items())),
            "class_row_counts": dict(sorted(Counter(r["class"] for r in image_rows).items())),
            "positive_bbox_row_count": sum(r["direct_CE"]["bbox"] == "positive" for r in image_rows),
            "positive_description_row_count": sum(r["direct_CE"]["description"] == "positive" for r in image_rows),
            "raw_readback": {"token_count": len(token_ids), "token_ids_sha256": raw_data["generated_token_ids_sha256"], "decode_stop_reason": raw_data["decode_stop_reason"], "natural_eos": natural_eos, "capped": raw_data["decode_stop_reason"] == "length"},
        })
    require(len(flat) == 1178, f"flat row count {len(flat)}")
    require(sum(row["raw_status"] != "parsed_valid" for row in flat) == 612, "parser drop count")
    require(set(second_patches) <= {(row["image_id"], row["prediction_id"]) for row in flat}, "unused second-fit ruling")
    covered_total = sum(row["covered_owner_count"] for row in images); missing_total = sum(row["missing_owner_count"] for row in images)
    require(covered_total + missing_total == 228, "cumulative target partition")
    decisions_path = OUT / "decisions.jsonl"
    decisions_path.write_bytes(b"".join(canonical(row) for row in flat))
    # Selected-owner supplements form a separate table and never block step16 publication.
    missing_supplements = [image_id for image_id in STEP64_IMAGE_IDS if not (REVIEWS / f"image-{image_id:012d}" / "supplemental-step64.json").is_file()]
    available_supplements = [image_id for image_id in STEP64_IMAGE_IDS if image_id not in missing_supplements]
    step64_rows = [load_supplement(image_id, evidence_catalog) for image_id in available_supplements]
    require(all(row["qualified_physical_geometry"] for row in step64_rows), "persisted selected step64 owner not qualified")
    step64 = {
        "schema": "training_set_completion.second_fit_step64_selected_owner_table.v1",
        "status": "needs_context" if missing_supplements else "candidate_targeted_checks_extracted",
        "expected_row_count": 9, "row_count": len(step64_rows), "rows": step64_rows,
        "qualified_count": sum(row["qualified_physical_geometry"] for row in step64_rows),
        "missing_supplement_image_ids": missing_supplements,
        "claim_boundary": "Selected-owner checks only; no full step64 owner coverage, false-positive, or clean-output claim.",
    }
    write_json(OUT / "step64-selected-owner-table.json", step64)
    if missing_supplements:
        pending = {
            "schema": "training_set_completion.second_fit_review_extraction_pending.v1", "status": "needs_context",
            "validated_step16_review_count": 11, "validated_step16_raw_rows": len(flat), "validated_parser_drops": 612,
            "root_rulings": binding(RULINGS), "missing_step64_supplement_image_ids": missing_supplements,
            "partial_step64_table": binding(OUT / "step64-selected-owner-table.json"),
            "note": "Canonical step16 extraction is independently publishable; only completion of the separate nine-owner step64 table awaits persisted same-image supplements.",
        }
        write_json(OUT / "pending-inputs.json", pending)
    parent_class = json.loads(PARENT_CLASSES.read_text())
    ledger = {
        "schema": "training_set_completion.second_fit_step16_review_extraction.v1", "status": "candidate_ready",
        "scope": {"checkpoint_step": 16, "image_ids": list(IMAGE_IDS), "target_v3_count": 228},
        "images": images,
        "summary": {
            "image_count": 11, "raw_row_count": len(flat), "parser_drop_count": 612,
            "physical_status_row_counts": dict(sorted(Counter(r["physical_status"] for r in flat).items())),
            "extent_row_counts": dict(sorted(Counter(r["extent"] for r in flat).items())),
            "class_row_counts": dict(sorted(Counter(r["class"] for r in flat).items())),
            "positive_bbox_row_count": sum(r["direct_CE"]["bbox"] == "positive" for r in flat),
            "positive_description_row_count": sum(r["direct_CE"]["description"] == "positive" for r in flat),
            "covered_physical_owner_count": covered_total, "missing_physical_owner_count": missing_total,
            "cumulative_physical_zero_missing_owners": missing_total == 0,
            "all_images_physical_zero": all(row["missing_owner_count"] == 0 for row in images),
            "parent_pre_ruling_covered_owner_count": parent["summary"]["covered_physical_owner_count"],
            "parent_corrected_covered_owner_count": sum(map(len, parent_by_image.values())),
            "parent_corrected_missing_owner_count": 228 - sum(map(len, parent_by_image.values())),
            "net_covered_owner_change_vs_corrected_parent": covered_total - sum(map(len, parent_by_image.values())),
            "root_effective_parent_class_counts": parent_class["counts"],
            "root_effective_parent_class_corrections": parent_class["changes"],
        },
        "evidence_catalog": [evidence_catalog[path] for path in sorted(evidence_catalog)],
        "source_bindings": {
            "target_v3": binding(TARGET), "root_rulings": binding(RULINGS), "parent_ledger_v2": binding(PARENT),
            "parent_root_acceptance": binding(PARENT_ACCEPT), "parent_effective_classes": binding(PARENT_CLASSES),
            "consolidated_step16_readback": binding(READBACK), "reviews": review_sources,
            "packets": packet_sources, "individual_raw_readbacks": raw_sources,
        },
        "artifacts": {"flat_decisions": binding(decisions_path), "step64_selected_owner_table": binding(OUT / "step64-selected-owner-table.json")},
        "limitations": [
            "Owner coverage requires physical identity, parsed-valid raw geometry, reasonable extent, and fixed-v3 membership; class errors remain separate.",
            "Root image25274 rulings correct both parent and second-fit comparisons without changing the fixed-v3 denominator.",
            "Step64 rows are selected-owner checks only and do not establish full-output coverage or cleanliness.",
        ],
    }
    write_json(OUT / "ledger.json", ledger)
    receipt = {
        "schema": "training_set_completion.second_fit_review_extraction_receipt.v1",
        "status": "candidate_step16_ready_step64_pending" if missing_supplements else "candidate_ready",
        "ledger": binding(OUT / "ledger.json"), "flat_decisions": binding(decisions_path),
        "step64_selected_owner_table": binding(OUT / "step64-selected-owner-table.json"),
        "counts": {"images": 11, "raw_rows": 1178, "parser_drops": 612, "targets": 228, "covered": covered_total, "missing": missing_total, "step64_selected_owner_checks": len(step64_rows), "step64_selected_owner_checks_expected": 9},
        "validator": binding(Path(__file__)),
    }
    write_json(OUT / "receipt.json", receipt)
    if not missing_supplements and (OUT / "pending-inputs.json").exists(): (OUT / "pending-inputs.json").unlink()
    return receipt


def refresh_step64_only() -> dict[str, Any]:
    """Refresh only the separate selected-owner table; never touch accepted step16 files."""
    OUT.mkdir(parents=True, exist_ok=True)
    ledger_path = OUT / "ledger.json"; decisions_path = OUT / "decisions.jsonl"
    require(file_hash(ledger_path) == "21be09a7f5bfe21c74d3ef61277e6feb65f49a2a78efe640cee9686b419de0c7", "accepted step16 ledger changed")
    require(file_hash(decisions_path) == "13d734cf66b3d7f5b07d5b9f9ea63c470740ba903ab39034ac2e9a949bc6423b", "accepted step16 decisions changed")
    evidence_catalog: dict[str, dict[str, Any]] = {}
    missing = [image_id for image_id in STEP64_IMAGE_IDS if not (REVIEWS / f"image-{image_id:012d}" / "supplemental-step64.json").is_file()]
    available = [image_id for image_id in STEP64_IMAGE_IDS if image_id not in missing]
    rows = [load_supplement(image_id, evidence_catalog) for image_id in available]
    require(all(row["qualified_physical_geometry"] for row in rows), "persisted selected step64 owner not qualified")
    table = {
        "schema": "training_set_completion.second_fit_step64_selected_owner_table.v1",
        "status": "needs_context" if missing else "candidate_targeted_checks_extracted",
        "expected_row_count": 9, "row_count": len(rows), "rows": rows,
        "qualified_count": sum(row["qualified_physical_geometry"] for row in rows),
        "missing_supplement_image_ids": missing,
        "evidence_catalog": [evidence_catalog[path] for path in sorted(evidence_catalog)],
        "frozen_step16_bindings": {"ledger": binding(ledger_path), "flat_decisions": binding(decisions_path)},
        "claim_boundary": "Selected-owner checks only; no full step64 owner coverage, false-positive, or clean-output claim.",
    }
    table_path = OUT / "step64-selected-owner-table.json"; write_json(table_path, table)
    receipt = {
        "schema": "training_set_completion.second_fit_step64_selected_owner_receipt.v1",
        "status": "needs_context" if missing else "candidate_ready",
        "table": binding(table_path), "expected_rows": 9, "rows": len(rows),
        "qualified_physical_geometry": sum(row["qualified_physical_geometry"] for row in rows),
        "missing_supplement_image_ids": missing,
        "frozen_step16_ledger_sha256": file_hash(ledger_path),
        "frozen_step16_decisions_sha256": file_hash(decisions_path),
        "validator": binding(Path(__file__)),
    }
    write_json(OUT / "step64-selected-owner-receipt.json", receipt)
    if missing:
        pending = {
            "schema": "training_set_completion.second_fit_review_extraction_pending.v1", "status": "needs_context",
            "missing_step64_supplement_image_ids": missing, "partial_step64_table": binding(table_path),
            "step16_status": "root_accepted_immutable", "step16_ledger_sha256": file_hash(ledger_path),
            "step16_decisions_sha256": file_hash(decisions_path),
        }
        write_json(OUT / "pending-inputs.json", pending)
    elif (OUT / "pending-inputs.json").exists():
        (OUT / "pending-inputs.json").unlink()
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-step64-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(refresh_step64_only() if args.refresh_step64_only else extract(), indent=2))


if __name__ == "__main__":
    main()
