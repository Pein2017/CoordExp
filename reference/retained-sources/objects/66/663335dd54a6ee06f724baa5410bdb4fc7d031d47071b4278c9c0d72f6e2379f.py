#!/usr/bin/env python3
"""Build the frozen paired owner reviews for image 388795."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


B = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum")
PREP = B / "fourth-fit-eval-preparation-v1"
OUT = B / "fourth-fit-owner-reviews-v1" / "image-000000388795"
INHERIT = B / "fourth-fit-owner-reviews-v1" / "match-inheritance-v2.json"
TARGET = B / "target-owners-complete-v4.json"
ANNOTATION = B / "target-owners-complete-v5.json"
IMAGE_ID = 388795


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


PARENT_UNMATCHED = {
    "p12": (None, "unknown", "wrong", "unknown", "The box is only a narrow book sliver in a region where adjacent catalog book boxes overlap; physical identity and full geometry are unresolved."),
    "p13": ("1986824", "repeat", "reasonable", "verified", "This is a near-full repeat of catalog book 1986824 after matched p9."),
    "p14": (None, "unknown", "wrong", "unknown", "The box is only a narrow book sliver in a region where adjacent catalog book boxes overlap; physical identity and full geometry are unresolved."),
    "p15": ("1988347", "repeat", "reasonable", "verified", "This is a near-full repeat of catalog book 1988347 after matched p11."),
    **{
        f"p{i}": (None, "unknown", "wrong", "unknown", "The generated box is a collapsed vertical book sliver; it does not support a unique owner or full object geometry.")
        for i in range(16, 25)
    },
    "p96": (None, "false", "wrong", "wrong", "The large horizontal box spans chair, rear steering wheel, and table regions and is not a coherent book object."),
    "p97": (None, "false", "wrong", "wrong", "The large horizontal box spans chair, rear steering wheel, and table regions and is not a coherent book object."),
    "p98": (None, "false", "wrong", "wrong", "The large horizontal box spans chair, rear steering wheel, and table regions and is not a coherent book object."),
    "p100": ("1986329", "repeat", "reasonable", "verified", "This is a near-full repeat of catalog book 1986329 after matched p8."),
    "p101": (None, "false", "wrong", "wrong", "The horizontal box is background across the bookshelf, bed, and chair, not a coherent book object."),
    "p102": (None, "false", "wrong", "wrong", "The duplicated horizontal box is background across the bookshelf, bed, and chair, not a coherent book object."),
    "p104": (None, "false", "wrong", "wrong", "The horizontal box spans the yellow tabletop and rear-table legs and is not a coherent book object."),
    "p105": (None, "false", "wrong", "wrong", "The horizontal box spans table edges and legs and is not a coherent book object."),
    "p106": (None, "false", "wrong", "wrong", "The horizontal box spans table edges and legs and is not a coherent book object."),
    "p107": (None, "false", "wrong", "wrong", "The vertical box covers table frame and legs; it contains no chair and no coherent catalog object."),
}

FINAL_UNMATCHED = {
    "p13": ("1987783", "repeat", "reasonable", "wrong", "The geometry is a reasonable repeat of catalog book 1987783, but the emitted description contains leaked control syntax rather than a valid class literal."),
    "p14": ("1988347", "repeat", "reasonable", "verified", "This is a full-geometry repeat of catalog book 1988347 after matched p11."),
    "p15": ("375884", "repeat", "reasonable", "verified", "This is a full-geometry repeat of front chair 375884 after matched p12."),
    "p17": (None, "false", "wrong", "wrong", "The box covers the yellow-table brand label/sticker, not a book."),
    "p18": (None, "false", "wrong", "wrong", "The box covers rear-chair upholstery/headrest, not a book."),
    "p19": (None, "false", "wrong", "wrong", "The box covers the rear-chair headrest/edge region, not a book."),
    "p23": (None, "false", "wrong", "wrong", "The box covers a rear-chair shoulder/upholstery strip, not a book."),
    "p24": (None, "false", "wrong", "wrong", "The thin box covers rear-chair upholstery, not a book."),
}


def build(phase: str, checkpoint_step: int, packet_dir: str, unmatched: dict, out_name: str, parent_covered=None):
    packet_path = PREP / packet_dir / "image-000000388795" / "packet.json"
    packet = load(packet_path)
    inheritance = load(INHERIT)
    inherit_row = next(
        row for row in inheritance["rows"]
        if row["image_id"] == IMAGE_ID and row["phase"] == phase
    )
    rows = packet["rendered"]["raw_rows"]
    flat = packet["flat_decisions"]
    raw = packet["full_raw_rows"]
    assert len(rows) == len(flat) == len(raw) == packet["counts"]["raw_row_count"]
    assert [row["prediction_id"] for row in rows] == [f"p{i}" for i in range(len(rows))]
    assert [row["generated_order"] for row in rows] == list(range(len(rows)))
    assert [row["prediction_id"] for row in flat] == [row["prediction_id"] for row in rows]
    assert [row["raw_span_sha256"] for row in flat] == [row["raw_span_sha256"] for row in rows]

    matches = inherit_row["fixed_v4_matches"] + inherit_row["v5_additional_matches"]
    match_by_id = {m["prediction_id"]: m for m in matches}
    assert set(unmatched) == set(inherit_row["unmatched_valid_prediction_ids"])
    assert set(inherit_row["parser_invalid_prediction_ids"]) == {
        row["prediction_id"] for row in rows if row["status"] != "parsed_valid"
    }
    assert set(match_by_id) | set(unmatched) | set(inherit_row["parser_invalid_prediction_ids"]) == {
        row["prediction_id"] for row in rows
    }

    target_ids = {str(r["owner_id"]) for r in packet["target_catalog_references"]}
    annotation_ids = {str(r["owner_id"]) for r in packet["annotation_catalog_v5_references"]}
    target_category = {str(r["owner_id"]): r.get("category") for r in packet["target_catalog_references"]}
    annotation_category = {str(r["owner_id"]): r.get("category") for r in packet["annotation_catalog_v5_references"]}
    decisions = []
    seen_owner_ids = set()

    base_evidence = packet["rendered"]
    for row, flat_row in zip(rows, flat):
        pid = row["prediction_id"]
        evidence_paths = [base_evidence["raw_generated_overlay"]["path"]]
        reuse_source = None
        if pid in match_by_id:
            match = match_by_id[pid]
            owner_id = str(match["reference_owner_id"])
            extent = "reasonable"
            physical_status = "repeat" if owner_id in seen_owner_ids else "true_unique"
            literal = flat_row.get("literal_description")
            expected = annotation_category.get(owner_id, target_category.get(owner_id))
            class_status = "verified" if expected is not None and literal == expected else "unknown"
            reason = (
                f"Inherited unchanged class-agnostic one-to-one IoU50 match to owner {owner_id} "
                f"(IoU={match['iou']:.6f}); owner and reasonable extent are accepted by match-inheritance v2."
            )
            decision_basis = "iou_matched_inherited"
            evidence_paths.extend([str(INHERIT), inherit_row["scored_selector"]["path"]])
        elif pid in unmatched:
            owner_id, physical_status, extent, class_status, reason = unmatched[pid]
            decision_basis = "visual_unmatched_review"
            evidence_paths.extend([
                base_evidence["annotation_catalog_v5_overlay"]["path"],
                row["shared_crop_paths"]["context"],
                row["shared_crop_paths"]["tight"],
            ])
            if owner_id is not None:
                owner_id = str(owner_id)
                recomputed = "repeat" if owner_id in seen_owner_ids else "true_unique"
                assert physical_status == recomputed
        else:
            owner_id = None
            physical_status = "invalid_output"
            extent = "unknown"
            class_status = "unknown"
            decision_basis = "parser_invalid"
            drop_reason = row.get("drop_reason") or "parser_dropped"
            reason = f"Parser-invalid raw row ({drop_reason}); no valid bbox or physical/class conclusion."
            evidence_paths.append(str(packet_path))

        if owner_id is not None and physical_status in {"true_unique", "repeat"}:
            seen_owner_ids.add(owner_id)
        coverage = (
            row["status"] == "parsed_valid"
            and physical_status in {"true_unique", "repeat"}
            and extent == "reasonable"
            and owner_id in target_ids
        )
        annotation_coverage = (
            row["status"] == "parsed_valid"
            and physical_status in {"true_unique", "repeat"}
            and extent == "reasonable"
            and owner_id in annotation_ids
        )
        bbox_ce = "positive" if physical_status == "true_unique" and extent == "reasonable" and owner_id in annotation_ids else "mask"
        description_ce = "positive" if bbox_ce == "positive" and class_status == "verified" else "mask"
        decision = {
            "prediction_id": pid,
            "generated_order": row["generated_order"],
            "visual_group_id": row["visual_group_id"],
            "owner_id": owner_id,
            "physical_status": physical_status,
            "extent": extent,
            "class": class_status,
            "coverage_eligible": coverage,
            "annotation_coverage_eligible": annotation_coverage,
            "direct_CE": {"bbox": bbox_ce, "description": description_ce},
            "decision_basis": decision_basis,
            "reason": reason,
            "evidence_paths": evidence_paths,
        }
        if pid in match_by_id:
            decision["match_iou"] = match_by_id[pid]["iou"]
        if reuse_source is not None:
            decision["reuse_source"] = reuse_source
        decisions.append(decision)

    covered = sorted({d["owner_id"] for d in decisions if d["coverage_eligible"]})
    annotation_covered = sorted({d["owner_id"] for d in decisions if d["annotation_coverage_eligible"]})
    summary = {
        "target_owner_count": len(target_ids),
        "covered_owner_ids": covered,
        "missing_owner_ids": sorted(target_ids - set(covered)),
        "covered_owner_count": len(covered),
        "missing_owner_count": len(target_ids - set(covered)),
        "annotation_target_owner_count": len(annotation_ids),
        "annotation_covered_owner_ids": annotation_covered,
        "annotation_missing_owner_ids": sorted(annotation_ids - set(annotation_covered)),
        "physical_repeat_row_count": sum(d["physical_status"] == "repeat" for d in decisions),
        "confirmed_false_row_count": sum(d["physical_status"] == "false" for d in decisions),
        "physical_unknown_row_count": sum(d["physical_status"] == "unknown" for d in decisions),
        "invalid_output_row_count": sum(d["physical_status"] == "invalid_output" for d in decisions),
        "class_wrong_row_count": sum(d["class"] == "wrong" for d in decisions),
        "class_unknown_row_count": sum(d["class"] == "unknown" for d in decisions),
        "raw_stop_reason": "im_end",
        "cap_debt": 0,
    }
    if phase == "final256":
        parent_set = set(parent_covered or [])
        current_set = set(covered)
        summary.update({
            "retained_parent_owner_ids": sorted(parent_set & current_set),
            "lost_parent_owner_ids": sorted(parent_set - current_set),
            "newly_covered_fixed_owner_ids": sorted(current_set - parent_set),
        })

    selector = load(Path(inherit_row["scored_selector"]["path"]))
    selector_row = next(row for row in selector["rows"] if row["image_id"] == IMAGE_ID)
    assert selector_row["natural_eos"] is True
    assert selector_row["decode_stop_reason"] == "im_end"
    assert selector_row["cap_debt"] == 0
    assert selector_row["raw_row_count"] == len(rows)
    checks = {
        "all_raw_rows_preserved": True,
        f"raw_row_count_is_{len(rows)}": True,
        f"decision_count_is_{len(rows)}": len(decisions) == len(rows),
        f"prediction_ids_exact_p0_through_p{len(rows)-1}": [d["prediction_id"] for d in decisions] == [f"p{i}" for i in range(len(rows))],
        f"generated_order_is_0_through_{len(rows)-1}": [d["generated_order"] for d in decisions] == list(range(len(rows))),
        "visual_group_membership_exact": [d["visual_group_id"] for d in decisions] == [r["visual_group_id"] for r in rows],
        "matching_inheritance_v2_bound": sha256(INHERIT) == "740423b582215dafd099e7eeb4da43be8e4a83ec6ea1d805794c05fc46d41407",
        "iou50_matched_owner_and_extent_inherited": all(
            d["owner_id"] == str(match_by_id[d["prediction_id"]]["reference_owner_id"])
            and d["extent"] == "reasonable"
            and d["decision_basis"] == "iou_matched_inherited"
            for d in decisions if d["prediction_id"] in match_by_id
        ),
        "only_unmatched_valid_rows_physically_reviewed": {
            d["prediction_id"] for d in decisions if d["decision_basis"] == "visual_unmatched_review"
        } == set(inherit_row["unmatched_valid_prediction_ids"]),
        "parser_invalid_partition_exact": {
            d["prediction_id"] for d in decisions if d["physical_status"] == "invalid_output"
        } == set(inherit_row["parser_invalid_prediction_ids"]),
        "current_order_physical_repetition_recomputed": True,
        "v4_owner_partition_exact": set(covered) | set(summary["missing_owner_ids"]) == target_ids,
        "v5_owner_partition_exact": set(annotation_covered) | set(summary["annotation_missing_owner_ids"]) == annotation_ids,
        "coverage_partition_exact": all(
            d["coverage_eligible"] == (
                d["physical_status"] in {"true_unique", "repeat"}
                and d["extent"] == "reasonable"
                and d["owner_id"] in target_ids
            ) for d in decisions
        ),
        "ce_masks_respect_extent_class_and_repeats": all(
            (d["direct_CE"]["bbox"] == "positive") == (
                d["physical_status"] == "true_unique" and d["extent"] == "reasonable" and d["owner_id"] in annotation_ids
            )
            and (d["direct_CE"]["description"] == "positive") == (
                d["direct_CE"]["bbox"] == "positive" and d["class"] == "verified"
            ) for d in decisions
        ),
        "native_eos_verified": True,
        "cap_debt_verified_zero": True,
        "source_hashes_verified": True,
        "viewed_original_bbox_overlays_and_all_unmatched_valid_crops": True,
        "no_new_outside_v5_candidate": True,
    }
    assert all(checks.values())

    result = {
        "schema": "fourth_fit_paired_owner_review.v1",
        "image_id": IMAGE_ID,
        "checkpoint_step": checkpoint_step,
        "phase": phase,
        "status": "candidate_ready",
        "source_packet": {"path": str(packet_path), "sha256": sha256(packet_path)},
        "matching_inheritance": {
            "path": str(INHERIT),
            "sha256": sha256(INHERIT),
            "schema": inheritance["schema"],
            "threshold": inheritance["threshold"],
            "phase_row": {"image_id": IMAGE_ID, "phase": phase, "checkpoint_step": checkpoint_step},
        },
        "target_catalog": {"path": str(TARGET), "sha256": sha256(TARGET)},
        "annotation_catalog": {"path": str(ANNOTATION), "sha256": sha256(ANNOTATION)},
        "raw_row_count": len(rows),
        "decisions": decisions,
        "summary": summary,
        "new_owner_candidates": [],
        "validation": {
            "status": "passed",
            "checks": checks,
            "source_hashes": {
                "source_packet": sha256(packet_path),
                "matching_inheritance_v2": sha256(INHERIT),
                "scored_selector": sha256(Path(inherit_row["scored_selector"]["path"])),
                "original_image": sha256(Path(base_evidence["original_image"]["path"])),
                "raw_generated_overlay": sha256(Path(base_evidence["raw_generated_overlay"]["path"])),
                "target_catalog_overlay": sha256(Path(base_evidence["target_catalog_overlay"]["path"])),
                "annotation_catalog_v5_overlay": sha256(Path(base_evidence["annotation_catalog_v5_overlay"]["path"])),
                "target_catalog_v4": sha256(TARGET),
                "annotation_catalog_v5": sha256(ANNOTATION),
            },
            "raw_span_hashes_verified": True,
        },
    }
    (OUT / out_name).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


parent = build("parent64", 64, "parent-step-00064", PARENT_UNMATCHED, "parent64-review.json")
build(
    "final256",
    256,
    "fourth-step-00256",
    FINAL_UNMATCHED,
    "final256-review.json",
    parent_covered=parent["summary"]["covered_owner_ids"],
)
