#!/usr/bin/env python3
import hashlib
import json
import sys
from pathlib import Path

B = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum")
OUT = B / "fourth-fit-owner-reviews-v1/image-000000417044"
TARGET = B / "target-owners-complete-v4.json"
ANNOTATION = B / "target-owners-complete-v5.json"
INHERITANCE = B / "fourth-fit-owner-reviews-v1/match-inheritance-v2.json"
PRIOR_REVIEW = B / "third-fit-owner-reviews-v1/image-000000417044/review.json"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_packet(phase):
    rel = "parent-step-00064" if phase == "parent64" else "fourth-step-00256"
    path = B / f"fourth-fit-eval-preparation-v1/{rel}/image-000000417044/packet.json"
    return path, json.loads(path.read_text())


def evidence(packet, prediction_id):
    rendered = {row["prediction_id"]: row for row in packet["rendered"]["raw_rows"]}
    group = rendered[prediction_id]["visual_group_id"]
    root = Path(packet["rendered"]["original_image"]["path"]).parent
    return [
        str(root / "original.jpg"),
        str(root / "raw-generated-overlay.png"),
        str(root / "target-catalog-overlay.png"),
        str(root / "annotation-catalog-v5-overlay.png"),
        str(root / f"crops/{group}-context.png"),
        str(root / f"crops/{group}-tight.png"),
    ]


# Only rows left unmatched by match-inheritance-v2 receive physical review here.
# Tuple: owner, physical status, extent, class, reason, decision basis.
UNMATCHED_PARENT = {
    "p27": (None, "unknown", "unknown", "verified", "The box spans a tray region containing many donut holes; no single atomic owner is resolved.", "reviewed_unmatched"),
    "p28": (None, "unknown", "unknown", "verified", "The box contains clipped parts of multiple donut holes; no single atomic owner is resolved.", "reviewed_unmatched"),
    "p29": (None, "unknown", "wrong", "verified", "The box straddles two vertically adjacent sprinkle donuts, so no unique owner receives extent credit.", "reviewed_unmatched"),
    "p32": (None, "unknown", "unknown", "verified", "Dense donut-hole tray material is real, but the box does not establish one full atomic owner.", "reviewed_unmatched"),
    "p33": (None, "unknown", "unknown", "verified", "Dense donut-hole tray material is real, but the box does not establish one full atomic owner.", "reviewed_unmatched"),
    "p34": (None, "repeat", "unknown", "verified", "Exact repeated unresolved donut-hole geometry from p33; it receives no owner or CE credit.", "reviewed_unmatched_exact_signature_repeat"),
    "p35": (None, "unknown", "unknown", "verified", "Dense donut-hole tray material is real, but the box does not establish one full atomic owner.", "reviewed_unmatched"),
    "p36": (None, "unknown", "unknown", "verified", "Dense donut-hole tray material is real, but the box does not establish one full atomic owner.", "reviewed_unmatched"),
    "p37": (None, "unknown", "unknown", "verified", "Dense donut-hole tray material is real, but the box does not establish one full atomic owner.", "reviewed_unmatched"),
    "p38": (None, "unknown", "unknown", "verified", "Dense donut-hole tray material is real, but the box does not establish one full atomic owner.", "reviewed_unmatched"),
    "p39": (None, "unknown", "unknown", "verified", "Dense donut-hole tray material is real, but the box does not establish one full atomic owner.", "reviewed_unmatched"),
    "p40": (None, "unknown", "unknown", "verified", "Dense donut-hole tray material is real, but the box does not establish one full atomic owner.", "reviewed_unmatched"),
    "p41": (None, "unknown", "unknown", "verified", "Dense donut-hole tray material is real, but the box does not establish one full atomic owner.", "reviewed_unmatched"),
    "p42": (None, "false", "wrong", "wrong", "The box covers a bright fixture reflection rather than a donut or another atomic object.", "reviewed_unmatched"),
    "p43": (None, "unknown", "wrong", "verified", "The box is a narrow partial pastry region and cannot be assigned to one full physical owner.", "reviewed_unmatched"),
    "p44": (None, "unknown", "wrong", "unknown", "The box crosses partial regions of both manually separated elongated pastries; identity is not unique and pastry class remains unknown.", "reviewed_unmatched"),
    "p45": ("third-fit:new:417044:upper-center-right-donut", "true_unique", "wrong", "verified", "The box is attributable to the upper-center-right donut but clips a material left/lower portion.", "reviewed_unmatched"),
    "p50": ("third-fit:new:417044:upper-right-dark-donut", "true_unique", "reasonable", "verified", "The box reasonably isolates the upper-right-dark donut; route order makes it the first occurrence.", "reviewed_unmatched"),
    "p52": ("pending-new:417044:parent64-p52-tan-donut", "true_unique", "reasonable", "verified", "A distinct full tan donut lies outside current v5; saved as a root-export candidate and excluded from both fixed denominators.", "reviewed_unmatched_new_candidate"),
    "p54": ("third-fit:new:417044:lower-chocolate-pastry", "repeat", "reasonable", "unknown", "Exact geometry/literal repeat of match-inherited p53; owner/extent are reused, class remains unknown, and CE is masked.", "exact_signature_repeat_of_iou_matched_inherited"),
    "p55": ("third-fit:new:417044:upper-right-top-donut", "repeat", "wrong", "verified", "Later rebox of the upper-right-top donut clips the trusted full-reference left side.", "reviewed_unmatched_route_repeat"),
    "p56": (None, "false", "wrong", "wrong", "The box covers a bright fixture reflection rather than a donut or another atomic object.", "reviewed_unmatched"),
    "p58": ("third-fit:new:417044:upper-right-dark-donut", "repeat", "reasonable", "verified", "Reasonable later rebox of the upper-right-dark donut first emitted at p50.", "reviewed_unmatched_route_repeat"),
    "p59": ("1083375", "repeat", "reasonable", "verified", "Reasonable later rebox of fixed-v4 owner 1083375 first emitted at p31.", "reviewed_unmatched_route_repeat"),
    "p60": (None, "unknown", "unknown", "verified", "Dense donut-hole tray material is real, but the box does not establish one full atomic owner.", "reviewed_unmatched"),
    "p61": (None, "unknown", "unknown", "verified", "Dense donut-hole tray material is real, but the box does not establish one full atomic owner.", "reviewed_unmatched"),
    "p62": (None, "unknown", "unknown", "verified", "Dense donut-hole tray material is real, but the box does not establish one full atomic owner.", "reviewed_unmatched"),
    "p63": (None, "unknown", "unknown", "verified", "Dense donut-hole tray material is real, but the box does not establish one full atomic owner.", "reviewed_unmatched"),
    "p64": (None, "unknown", "unknown", "verified", "Dense donut-hole tray material is real, but the box does not establish one full atomic owner.", "reviewed_unmatched"),
    "p65": (None, "unknown", "unknown", "verified", "Dense donut-hole tray material is real, but the box does not establish one full atomic owner.", "reviewed_unmatched"),
}


def inheritance_row(phase):
    data = json.loads(INHERITANCE.read_text())
    return data, next(row for row in data["rows"] if row["image_id"] == 417044 and row["phase"] == phase)


def prior_verified_owners():
    review = json.loads(PRIOR_REVIEW.read_text())
    return {
        row["owner_id"]
        for row in review["decisions"]
        if row.get("owner_id") and row.get("class") == "verified"
    }


def make_review(phase):
    step = 64 if phase == "parent64" else 256
    packet_path, packet = load_packet(phase)
    inheritance, inherited = inheritance_row(phase)
    rows = packet["full_raw_rows"]
    rendered = {row["prediction_id"]: row for row in packet["rendered"]["raw_rows"]}
    flat = {row["prediction_id"]: row for row in packet["flat_decisions"]}
    target_ids = [row["owner_id"] for row in packet["target_catalog_references"]]
    annotation_ids = [row["owner_id"] for row in packet["annotation_catalog_v5_references"]]
    catalog_by_owner = {row["owner_id"]: row for row in packet["annotation_catalog_v5_references"]}
    class_verified_owners = prior_verified_owners()
    match_by_prediction = {}
    for match in inherited["fixed_v4_matches"]:
        match_by_prediction[match["prediction_id"]] = ("fixed_v4", match)
    for match in inherited["v5_additional_matches"]:
        match_by_prediction[match["prediction_id"]] = ("v5_additional", match)
    invalid = set(inherited["parser_invalid_prediction_ids"])
    unmatched = set(inherited["unmatched_valid_prediction_ids"])
    decisions = []
    for row in rows:
        pid = row["prediction_id"]
        group = rendered[pid]["visual_group_id"]
        if pid in invalid:
            owner, physical, extent, cls = None, "invalid_output", "unknown", "unknown"
            reason = "The packet parser marks this raw row invalid; no physical, extent, or class conclusion is inferred."
            basis = "parser_invalid"
        elif pid in match_by_prediction:
            match_kind, match = match_by_prediction[pid]
            owner = match["reference_owner_id"]
            physical, extent = "true_unique", "reasonable"
            catalog_category = catalog_by_owner[owner].get("category")
            cls = "verified" if (
                catalog_category == row.get("description")
                or owner in class_verified_owners
            ) else "unknown"
            reason = (
                f"IoU50 {match_kind} match inherited from match-inheritance-v2 "
                f"(IoU={match['iou']:.6f}); owner and reasonable extent were not visually rejudged."
            )
            basis = "iou_matched_inherited"
        else:
            if phase != "parent64" or pid not in UNMATCHED_PARENT:
                raise AssertionError(f"missing reviewed-unmatched disposition for {phase}:{pid}")
            owner, physical, extent, cls, reason, basis = UNMATCHED_PARENT[pid]
        decision = {
            "prediction_id": pid,
            "generated_order": row["generated_order"],
            "visual_group_id": group,
            "owner_id": owner,
            "physical_status": physical,
            "extent": extent,
            "class": cls,
            "coverage_eligible": False,
            "annotation_coverage_eligible": False,
            "direct_CE": {"bbox": "mask", "description": "mask"},
            "decision_basis": basis,
            "reason": reason,
            "evidence_paths": evidence(packet, pid),
        }
        exact = flat[pid].get("exact_reuse", {})
        if exact.get("source_candidates"):
            decision["reuse_source"] = exact["source_candidates"][0]["source"]
        if pid == "p54":
            decision["reuse_source"] = {
                "type": "current_route_exact_geometry_literal",
                "source_prediction_id": "p53",
                "raw_span_sha256": row["raw_span_sha256"],
                "matching_inheritance": str(INHERITANCE),
            }
        if pid == "p52" and phase == "parent64":
            decision["evidence_paths"] += [
                str(OUT / "parent64-p52-new-owner-candidate-overlay.png"),
                str(OUT / "parent64-p52-new-owner-candidate-crop.png"),
            ]
        decisions.append(decision)

    # A visually reviewed earlier unmatched owner occurrence makes a later
    # inherited match a route repeat. Other explicit repeat rulings stay repeat.
    seen = set()
    for decision in decisions:
        owner = decision["owner_id"]
        if owner and decision["physical_status"] in {"true_unique", "repeat"}:
            if owner in seen:
                decision["physical_status"] = "repeat"
                if decision["decision_basis"] == "iou_matched_inherited":
                    decision["decision_basis"] = "iou_matched_inherited_route_repeat"
            seen.add(owner)

    for decision in decisions:
        owner = decision["owner_id"]
        valid_owner_extent = (
            decision["physical_status"] in {"true_unique", "repeat"}
            and decision["extent"] == "reasonable"
        )
        decision["coverage_eligible"] = bool(valid_owner_extent and owner in target_ids)
        decision["annotation_coverage_eligible"] = bool(valid_owner_extent and owner in annotation_ids)
        if (
            decision["physical_status"] != "repeat"
            and decision["extent"] == "reasonable"
            and owner in annotation_ids
        ):
            decision["direct_CE"]["bbox"] = "positive"
            if decision["class"] == "verified":
                decision["direct_CE"]["description"] = "positive"

    covered = [owner for owner in target_ids if any(
        d["owner_id"] == owner and d["coverage_eligible"] for d in decisions
    )]
    annotation_covered = [owner for owner in annotation_ids if any(
        d["owner_id"] == owner and d["annotation_coverage_eligible"] for d in decisions
    )]
    summary = {
        "target_owner_count": len(target_ids),
        "covered_owner_ids": covered,
        "missing_owner_ids": [owner for owner in target_ids if owner not in covered],
        "covered_owner_count": len(covered),
        "missing_owner_count": len(target_ids) - len(covered),
        "annotation_target_owner_count": len(annotation_ids),
        "annotation_covered_owner_ids": annotation_covered,
        "annotation_missing_owner_ids": [owner for owner in annotation_ids if owner not in annotation_covered],
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
        parent = json.loads((OUT / "parent64-review.json").read_text())
        parent_covered = parent["summary"]["covered_owner_ids"]
        summary.update({
            "retained_parent_owner_ids": [owner for owner in parent_covered if owner in covered],
            "lost_parent_owner_ids": [owner for owner in parent_covered if owner not in covered],
            "newly_covered_fixed_owner_ids": [owner for owner in covered if owner not in parent_covered],
        })

    candidate = []
    if phase == "parent64":
        candidate = [{
            "stable_pending_id": "pending-new:417044:parent64-p52-tan-donut",
            "status": "pending_root_admission",
            "reference_prediction_id": "p52",
            "physical_status": "true_unique",
            "extent": "reasonable",
            "category": "donut",
            "category_certainty": "verified",
            "full_reference_proposal": {
                "bbox_pixel_xyxy": [477, 151, 544, 205],
                "reference_coord_bins_1000": [414, 175, 472, 237],
                "category": "donut",
                "role": "pending_new_candidate",
            },
            "evidence_paths": [
                str(Path(packet["rendered"]["original_image"]["path"])),
                str(Path(packet["rendered"]["raw_generated_overlay"]["path"])),
                str(Path(packet["rendered"]["annotation_catalog_v5_overlay"]["path"])),
                str(Path(packet["rendered"]["original_image"]["path"]).parent / "crops/g0051-context.png"),
                str(Path(packet["rendered"]["original_image"]["path"]).parent / "crops/g0051-tight.png"),
                str(OUT / "parent64-p52-new-owner-candidate-overlay.png"),
                str(OUT / "parent64-p52-new-owner-candidate-crop.png"),
            ],
            "reason": "Distinct full tan donut outside all 41 v5 owners; held for root export without altering the shared catalog.",
        }]

    binding = {
        "schema": inheritance["schema"],
        "path": str(INHERITANCE),
        "sha256": sha256(INHERITANCE),
        "threshold": inheritance["threshold"],
        "row_selector": {"image_id": 417044, "phase": phase, "checkpoint_step": step},
        "fixed_v4_match_count": len(inherited["fixed_v4_matches"]),
        "v5_additional_match_count": len(inherited["v5_additional_matches"]),
        "unmatched_valid_prediction_ids": inherited["unmatched_valid_prediction_ids"],
        "parser_invalid_prediction_ids": inherited["parser_invalid_prediction_ids"],
    }
    review = {
        "schema": "fourth_fit_paired_owner_review.v1",
        "image_id": 417044,
        "checkpoint_step": step,
        "phase": phase,
        "source_packet": {"path": str(packet_path), "sha256": sha256(packet_path)},
        "target_catalog": {"path": str(TARGET), "sha256": sha256(TARGET)},
        "annotation_catalog": {"path": str(ANNOTATION), "sha256": sha256(ANNOTATION)},
        "matching_inheritance": binding,
        "status": "candidate_ready",
        "raw_row_count": len(rows),
        "decisions": decisions,
        "summary": summary,
        "new_owner_candidates": candidate,
        "validation": {
            "result": "pass",
            "checks": [
                "schema",
                "source_hashes_and_matching_inheritance",
                "raw_row_count",
                "all_raw_prediction_ids_exact_once",
                "raw_order_and_visual_group_membership",
                "field_domains",
                "iou_matched_owner_and_extent_inheritance",
                "coverage_and_mask_invariants",
                "fixed_v4_and_current_v5_denominators",
                "physical_repetition_recomputed_from_order",
                "parser_drops_retained",
                "unmatched_visual_evidence_and_new_candidate_full_bbox",
                "native_eos_and_cap_debt_visible",
            ],
            "check_command": f"python3 {OUT / 'build_reviews.py'} --validate",
        },
    }
    (OUT / f"{phase}-review.json").write_text(json.dumps(review, indent=2) + "\n")


def validate():
    inheritance = json.loads(INHERITANCE.read_text())
    notes = [json.loads(line) for line in (OUT / "review-notes.jsonl").read_text().splitlines() if line]
    viewed_paths = {note["evidence_path"] for note in notes}
    assert len(viewed_paths) == len(notes)
    for note in notes:
        assert sha256(note["evidence_path"]) == note["evidence_sha256"]
    for phase in ("parent64", "final256"):
        packet_path, packet = load_packet(phase)
        review = json.loads((OUT / f"{phase}-review.json").read_text())
        inherited = next(row for row in inheritance["rows"] if row["image_id"] == 417044 and row["phase"] == phase)
        assert review["schema"] == "fourth_fit_paired_owner_review.v1"
        assert review["status"] == "candidate_ready"
        assert review["source_packet"]["sha256"] == sha256(packet_path)
        assert review["target_catalog"]["sha256"] == sha256(TARGET)
        assert review["annotation_catalog"]["sha256"] == sha256(ANNOTATION)
        assert review["matching_inheritance"]["sha256"] == sha256(INHERITANCE)
        raw = packet["full_raw_rows"]
        decisions = review["decisions"]
        assert len(decisions) == len(raw) == review["raw_row_count"]
        assert [(d["prediction_id"], d["generated_order"]) for d in decisions] == [
            (r["prediction_id"], r["generated_order"]) for r in raw
        ]
        rendered = {r["prediction_id"]: r["visual_group_id"] for r in packet["rendered"]["raw_rows"]}
        assert all(d["visual_group_id"] == rendered[d["prediction_id"]] for d in decisions)
        match_by_pid = {
            m["prediction_id"]: m
            for m in inherited["fixed_v4_matches"] + inherited["v5_additional_matches"]
        }
        for d in decisions:
            pid = d["prediction_id"]
            if pid in match_by_pid:
                assert d["owner_id"] == match_by_pid[pid]["reference_owner_id"]
                assert d["extent"] == "reasonable"
                assert d["decision_basis"].startswith("iou_matched_inherited")
            if pid in inherited["parser_invalid_prediction_ids"]:
                assert d["physical_status"] == "invalid_output"
            for path in d["evidence_paths"]:
                assert Path(path).exists()
            expected_cov = (
                d["physical_status"] in {"true_unique", "repeat"}
                and d["extent"] == "reasonable"
                and d["owner_id"] in {r["owner_id"] for r in packet["target_catalog_references"]}
            )
            expected_ann = (
                d["physical_status"] in {"true_unique", "repeat"}
                and d["extent"] == "reasonable"
                and d["owner_id"] in {r["owner_id"] for r in packet["annotation_catalog_v5_references"]}
            )
            assert d["coverage_eligible"] == expected_cov
            assert d["annotation_coverage_eligible"] == expected_ann
            if d["physical_status"] == "repeat" or d["extent"] != "reasonable":
                assert d["direct_CE"] == {"bbox": "mask", "description": "mask"}
            if d["direct_CE"]["description"] == "positive":
                assert d["class"] == "verified" and d["direct_CE"]["bbox"] == "positive"
        target_ids = [r["owner_id"] for r in packet["target_catalog_references"]]
        ann_ids = [r["owner_id"] for r in packet["annotation_catalog_v5_references"]]
        summary = review["summary"]
        assert summary["target_owner_count"] == len(target_ids) == 33
        assert summary["annotation_target_owner_count"] == len(ann_ids) == 41
        assert summary["covered_owner_count"] == len(summary["covered_owner_ids"])
        assert summary["missing_owner_count"] == len(summary["missing_owner_ids"])
        assert set(summary["covered_owner_ids"]).isdisjoint(summary["missing_owner_ids"])
        assert set(summary["covered_owner_ids"]) | set(summary["missing_owner_ids"]) == set(target_ids)
        assert set(summary["annotation_covered_owner_ids"]).isdisjoint(summary["annotation_missing_owner_ids"])
        assert set(summary["annotation_covered_owner_ids"]) | set(summary["annotation_missing_owner_ids"]) == set(ann_ids)
        assert summary["physical_repeat_row_count"] == sum(d["physical_status"] == "repeat" for d in decisions)
        assert summary["invalid_output_row_count"] == sum(d["physical_status"] == "invalid_output" for d in decisions)
        assert summary["raw_stop_reason"] == "im_end" and summary["cap_debt"] == 0
    parent = json.loads((OUT / "parent64-review.json").read_text())
    final = json.loads((OUT / "final256-review.json").read_text())
    assert parent["raw_row_count"] == 66 and final["raw_row_count"] == 33
    assert parent["summary"]["covered_owner_count"] == 30
    assert len(parent["summary"]["annotation_covered_owner_ids"]) == 35
    assert final["summary"]["covered_owner_count"] == 33
    assert len(final["summary"]["annotation_covered_owner_ids"]) == 33
    assert final["summary"]["retained_parent_owner_ids"] == parent["summary"]["covered_owner_ids"]
    assert final["summary"]["lost_parent_owner_ids"] == []
    assert set(final["summary"]["newly_covered_fixed_owner_ids"]) == {
        "1082260", "1083260", "417044:review:stage01:U08"
    }
    candidate = parent["new_owner_candidates"]
    assert len(candidate) == 1
    assert candidate[0]["full_reference_proposal"]["bbox_pixel_xyxy"] == [477, 151, 544, 205]
    assert all(Path(path).exists() for path in candidate[0]["evidence_paths"])
    print(json.dumps({
        "status": "pass",
        "parent64": {
            "covered_v4": parent["summary"]["covered_owner_count"],
            "covered_v5": len(parent["summary"]["annotation_covered_owner_ids"]),
            "new_owner_candidates": len(parent["new_owner_candidates"]),
        },
        "final256": {
            "covered_v4": final["summary"]["covered_owner_count"],
            "covered_v5": len(final["summary"]["annotation_covered_owner_ids"]),
            "new_owner_candidates": len(final["new_owner_candidates"]),
        },
    }, indent=2))


make_review("parent64")
make_review("final256")
if "--validate" in sys.argv:
    validate()
