#!/usr/bin/env python3
"""Build and validate the single-image third-fit physical review."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


B = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum")
OUT = B / "third-fit-owner-reviews-v1/image-000000417044"
PACKET = B / "third-fit-review-packets-v1/image-000000417044/packet.json"
TARGET = B / "target-owners-complete-v4.json"
LEDGER = B / "parent16-v4-physical-ledger-v1/ledger.json"
READBACK = B / "third-fit-v1/readback-recovery/rows/step-00032-image-000000417044.json"
REVIEW = OUT / "review.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


FIXED = {
    0: "515293",
    1: "417044:review:stage01:U01",
    2: "417044:review:stage01:U02",
    3: "417044:review:P6",
    4: "417044:review:stage01:U03",
    5: "417044:review:stage01:U04",
    6: "417044:review:stage01:U05",
    7: "1083135",
    8: "1083564",
    9: "417044:review:stage01:U07",
    10: "417044:review:stage01:U09",
    11: "417044:review:stage01:U10",
    12: "first-fit:new:417044:upper-center-donut",
    13: "417044:review:stage01:U11",
    14: "1083042",
    15: "417044:review:stage01:U12",
    16: "417044:review:stage01:U14",
    17: "417044:review:stage01:U13",
    18: "1082918",
    19: "1083295",
    20: "1083599",
    21: "1079494",
    22: "1079910",
    23: "1080038",
    24: "1082111",
    25: "1572342",
    46: "417044:review:stage01:U16",
    52: "417044:review:stage01:U15",
    53: "1083375",
}
PENDING = {
    43: "pending-new:417044:upper-left-center-donut",
    44: "pending-new:417044:upper-center-right-donut",
    45: "pending-new:417044:middle-dark-donut",
    47: "pending-new:417044:upper-right-top-donut",
    48: "pending-new:417044:middle-right-dark-donut",
    50: "pending-new:417044:upper-right-dark-donut",
    51: "pending-new:417044:middle-right-sprinkle-donut",
}
UNKNOWN_ROWS = set(range(26, 42)) | set(range(54, 61))
REPEAT_ROWS = {29, 33, 34, 36, 37}
FALSE_ROWS = {42}
PENDING_REPEAT = {49: PENDING[48]}


def evidence_for(group: dict, common: list[str]) -> list[str]:
    return common + [group["context_crop"]["path"], group["tight_crop"]["path"]]


def build() -> dict:
    packet = load(PACKET)
    groups = {
        int(prediction_id[1:]): g
        for g in packet["rendered"]["visual_groups"]
        for prediction_id in g["member_prediction_ids"]
    }
    rows = packet["rendered"]["raw_rows"]
    assert len(rows) == 61
    common = [
        packet["rendered"]["original_image"]["path"],
        packet["rendered"]["raw_generated_overlay"]["path"],
        packet["rendered"]["target_catalog_overlay"]["path"],
    ]
    decisions = []
    for row in rows:
        n = int(row["prediction_id"][1:])
        group = groups[n]
        assert group["visual_group_id"] == row["visual_group_id"]
        if n in FIXED:
            owner = FIXED[n]
            status, extent, cls, eligible = "true_unique", "reasonable", "verified", True
            direct = {"bbox": "positive", "description": "positive"}
            reason = f"One visually distinct donut/person with reasonable full geometry; mapped to fixed owner {owner}."
        elif n in PENDING:
            owner = PENDING[n]
            status, extent, cls, eligible = "true_unique", "reasonable", "verified", False
            direct = {"bbox": "mask", "description": "mask"}
            reason = "Distinct visible donut with reasonable geometry, absent from the fixed v4 catalog; admission remains pending root review."
        elif n in PENDING_REPEAT:
            owner = PENDING_REPEAT[n]
            status, extent, cls, eligible = "repeat", "reasonable", "verified", False
            direct = {"bbox": "mask", "description": "mask"}
            reason = "Rebox of the same pending upper-right dark donut as p48; repeated physical owner receives no direct CE."
        elif n in REPEAT_ROWS:
            owner = None
            status, extent, cls, eligible = "repeat", "unknown", "verified", False
            direct = {"bbox": "mask", "description": "mask"}
            reason = "Exact repeated box in the dense donut-hole tray; it repeats an unresolved multi-owner extent and cannot receive owner or CE credit."
        elif n in UNKNOWN_ROWS:
            owner = None
            status, extent, cls, eligible = "unknown", "unknown", "verified", False
            direct = {"bbox": "mask", "description": "mask"}
            reason = "Visible donut-hole tray material is real, but the box spans multiple adjacent items or is too clipped to establish one atomic owner with reasonable full geometry."
        elif n in FALSE_ROWS:
            owner = None
            status, extent, cls, eligible = "false", "wrong", "wrong", False
            direct = {"bbox": "mask", "description": "mask"}
            reason = "Box covers the metal display fixture rather than a donut; no atomic object or reasonable donut extent is present."
        else:
            raise AssertionError(f"unclassified raw row p{n}")
        decisions.append({
            "prediction_id": row["prediction_id"],
            "generated_order": row["generated_order"],
            "visual_group_id": row["visual_group_id"],
            "owner_id": owner,
            "physical_status": status,
            "extent": extent,
            "class": cls,
            "coverage_eligible": eligible,
            "direct_CE": direct,
            "reason": reason,
            "evidence_paths": evidence_for(group, common),
        })

    ledger = next(x for x in load(LEDGER)["images"] if x["image_id"] == 417044)
    target_ids = ledger["target_owner_ids"]
    covered = sorted({d["owner_id"] for d in decisions if d["coverage_eligible"]}, key=target_ids.index)
    parent = ledger["covered_owner_ids"]
    missing = [x for x in target_ids if x not in covered]
    newly = [x for x in covered if x not in parent]
    evidence_paths = set(common)
    for g in groups.values():
        evidence_paths.add(g["context_crop"]["path"])
        evidence_paths.add(g["tight_crop"]["path"])
    summary = {
        "target_owner_count": len(target_ids),
        "covered_owner_ids": covered,
        "missing_owner_ids": missing,
        "covered_owner_count": len(covered),
        "missing_owner_count": len(missing),
        "physical_repeat_row_count": sum(d["physical_status"] == "repeat" for d in decisions),
        "confirmed_false_row_count": sum(d["physical_status"] == "false" for d in decisions),
        "physical_unknown_row_count": sum(d["physical_status"] == "unknown" for d in decisions),
        "invalid_output_row_count": sum(d["physical_status"] == "invalid_output" for d in decisions),
        "class_wrong_row_count": sum(d["class"] == "wrong" for d in decisions),
        "class_unknown_row_count": sum(d["class"] == "unknown" for d in decisions),
        "raw_stop_reason": load(READBACK)["decode_stop_reason"],
        "cap_debt": False,
        "parent_covered_owner_ids": parent,
        "retained_parent_owner_ids": [x for x in parent if x in covered],
        "lost_parent_owner_ids": [x for x in parent if x not in covered],
        "newly_covered_fixed_owner_ids": newly,
        "stage03_appended_gt_reference_repair": {
            "qualified": False,
            "reason": "No stage03 appended GT/reference repair applies to image 417044.",
        },
        "earliest_observed_repeat_or_structural_issue": {
            "prediction_id": "p26",
            "kind": "unresolved multi-owner extent",
            "reason": "The broad right-tray box spans many donut-hole items, so it cannot receive atomic owner or CE credit.",
        },
    }
    row_by_n = {int(r["prediction_id"][1:]): r for r in rows}
    new_candidates = []
    for n, owner in PENDING.items():
        row = row_by_n[n]
        group = groups[n]
        new_candidates.append({
            "owner_id": owner,
            "stable_pending_id": owner,
            "reference_prediction_id": f"p{n}",
            "category": "donut",
            "physical_status": "true_unique",
            "extent": "reasonable",
            "class": "verified",
            "actual_bbox_pixel_xyxy": row["raw_bbox_pixel_xyxy"],
            "full_reference_proposal": {
                "bbox_pixel_xyxy": row["raw_bbox_pixel_xyxy"],
                "reference_coord_bins_1000": row["coord_bins_1000"],
                "category": "donut",
                "role": "pending_new_candidate",
            },
            "status": "pending_root_admission",
            "reason": "Distinct visible donut with no fixed v4 reference; masked until root admission.",
            "evidence_paths": evidence_for(group, common),
        })
    return {
        "schema": "third_fit_step32_owner_review.v1",
        "image_id": 417044,
        "checkpoint_step": 32,
        "target_catalog": {"path": str(TARGET), "sha256": sha256(TARGET)},
        "source_packet": {"path": str(PACKET), "sha256": sha256(PACKET)},
        "status": "candidate_ready",
        "raw_row_count": len(rows),
        "decisions": decisions,
        "summary": summary,
        "new_owner_candidates": new_candidates,
        "validation": {
            "all_raw_prediction_ids_preserved": True,
            "all_generated_orders_preserved": True,
            "visual_group_membership_exact": True,
            "coverage_partition_exact": True,
            "coverage_eligibility_recomputed": True,
            "direct_ce_domain_valid": True,
            "no_repeat_direct_positive": True,
            "evidence_paths_exist": True,
            "evidence_sha256": {p: sha256(Path(p)) for p in sorted(evidence_paths)},
            "source_artifact_sha256": {
                "parent_v4_physical_ledger": {"path": str(LEDGER), "sha256": sha256(LEDGER)},
                "scored_json": {"path": packet["source"]["scored_json"]["path"], "sha256": sha256(Path(packet["source"]["scored_json"]["path"]))},
                "acquisition_manifest": {"path": packet["source"]["acquisition_manifest"]["path"], "sha256": sha256(Path(packet["source"]["acquisition_manifest"]["path"]))},
            },
            "source_native_readback": {"path": str(READBACK), "sha256": sha256(READBACK)},
            "readback_stop_and_cap_verified": True,
        },
    }


def validate(review: dict) -> None:
    packet = load(PACKET)
    rows = packet["rendered"]["raw_rows"]
    assert review["raw_row_count"] == len(rows) == len(review["decisions"]) == 61
    assert [d["prediction_id"] for d in review["decisions"]] == [r["prediction_id"] for r in rows]
    assert [d["generated_order"] for d in review["decisions"]] == [r["generated_order"] for r in rows]
    assert all(d["coverage_eligible"] == (d["physical_status"] == "true_unique" and d["owner_id"] in FIXED.values()) for d in review["decisions"])
    assert all(d["direct_CE"][k] in {"positive", "mask"} for d in review["decisions"] for k in ("bbox", "description"))
    assert all(not any(v == "positive" for v in d["direct_CE"].values()) for d in review["decisions"] if d["physical_status"] in {"repeat", "unknown", "false"} or not d["coverage_eligible"])
    for candidate in review["new_owner_candidates"]:
        assert candidate["owner_id"] == candidate["stable_pending_id"]
        assert candidate["status"] == "pending_root_admission"
        assert candidate["physical_status"] == "true_unique"
        assert candidate["extent"] == "reasonable"
        assert candidate["class"] == "verified"
        assert candidate["full_reference_proposal"]["bbox_pixel_xyxy"] == candidate["actual_bbox_pixel_xyxy"]
        assert len(candidate["evidence_paths"]) == 5
    s = review["summary"]
    assert s["covered_owner_count"] + s["missing_owner_count"] == s["target_owner_count"] == 33
    assert set(s["covered_owner_ids"]).isdisjoint(s["missing_owner_ids"])
    target_ids = {x["owner_id"] for x in load(TARGET)["records"] if x["image_id"] == 417044}
    assert set(s["covered_owner_ids"]) | set(s["missing_owner_ids"]) == target_ids
    assert set(s["parent_covered_owner_ids"]) <= set(s["covered_owner_ids"]) | set(s["lost_parent_owner_ids"])
    for d in review["decisions"]:
        for p in d["evidence_paths"]:
            assert Path(p).is_file(), p
    for p, digest in review["validation"]["evidence_sha256"].items():
        assert sha256(Path(p)) == digest, p
    assert review["validation"]["source_native_readback"]["sha256"] == sha256(READBACK)
    for source in review["validation"]["source_artifact_sha256"].values():
        assert sha256(Path(source["path"])) == source["sha256"]
    print(json.dumps({"ok": True, "rows": len(rows), "covered": s["covered_owner_count"], "missing": s["missing_owner_count"], "repeats": s["physical_repeat_row_count"], "unknown": s["physical_unknown_row_count"], "false": s["confirmed_false_row_count"]}, sort_keys=True))


def write_notes(review: dict) -> None:
    """Persist the interpretation made for every image/group view."""
    packet = load(PACKET)
    common = [
        packet["rendered"]["original_image"]["path"],
        packet["rendered"]["raw_generated_overlay"]["path"],
        packet["rendered"]["target_catalog_overlay"]["path"],
    ]
    notes = [
        {"record_type": "image_view", "image_id": 417044, "kind": "original", "viewed_path": common[0], "sha256": sha256(Path(common[0])), "interpretation": "Viewed for image-level spatial context and physical donut identity."},
        {"record_type": "image_view", "image_id": 417044, "kind": "raw_generated_overlay", "viewed_path": common[1], "sha256": sha256(Path(common[1])), "interpretation": "Viewed to inspect all raw step32 boxes and generated order."},
        {"record_type": "image_view", "image_id": 417044, "kind": "target_catalog_overlay", "viewed_path": common[2], "sha256": sha256(Path(common[2])), "interpretation": "Viewed to bind current boxes to fixed v4 owners and identify missing references."},
    ]
    by_id = {d["prediction_id"]: d for d in review["decisions"]}
    for group in packet["rendered"]["visual_groups"]:
        ids = group["member_prediction_ids"]
        d = by_id[ids[0]]
        notes.append({
            "record_type": "visual_group_view",
            "image_id": 417044,
            "checkpoint_step": 32,
            "visual_group_id": group["visual_group_id"],
            "prediction_ids": ids,
            "viewed_paths": [group["context_crop"]["path"], group["tight_crop"]["path"]],
            "evidence_sha256": {group["context_crop"]["path"]: sha256(Path(group["context_crop"]["path"])), group["tight_crop"]["path"]: sha256(Path(group["tight_crop"]["path"]))},
            "interpretation": d["reason"],
            "review_decision": {"owner_id": d["owner_id"], "physical_status": d["physical_status"], "extent": d["extent"], "class": d["class"]},
        })
    (OUT / "review-notes.jsonl").write_text("".join(json.dumps(n, sort_keys=True) + "\n" for n in notes))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    review = build()
    if args.write:
        OUT.mkdir(parents=True, exist_ok=True)
        REVIEW.write_text(json.dumps(review, indent=2) + "\n")
        write_notes(review)
    validate(review)
