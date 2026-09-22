import hashlib
import json
from pathlib import Path

BASE = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum")
PACKET_PATH = BASE / "third-fit-review-packets-v1/image-000000351017/packet.json"
TARGET_PATH = BASE / "target-owners-complete-v4.json"
PARENT_PATH = BASE / "parent16-v4-physical-ledger-v1/ledger.json"
OUT = BASE / "third-fit-owner-reviews-v1/image-000000351017"
REVIEW_PATH = OUT / "review.json"
NOTES_PATH = OUT / "review-notes.jsonl"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


packet = json.load(open(PACKET_PATH))
rows = packet["rendered"]["raw_rows"]
groups = {g["visual_group_id"]: g for g in packet["rendered"]["visual_groups"]}
target_ids = sorted(str(x["owner_id"]) for x in packet["target_catalog_references"])
original = packet["rendered"]["original_image"]["path"]
raw_overlay = packet["rendered"]["raw_generated_overlay"]["path"]
target_overlay = packet["rendered"]["target_catalog_overlay"]["path"]

parent_covered = [
    "1489041", "1870183", "1961511", "2094819", "466970", "476801",
    "478719", "499060", "666212", "667309", "667754", "667769",
    "667794", "90429", "91179", "95431", "96050",
]

# owner_id, physical, extent, class, interpretation
spec = {
    "p0": ("466970", "true_unique", "reasonable", "verified", "Left foreground person with coherent full visible extent."),
    "p1": ("476801", "true_unique", "reasonable", "verified", "Left-middle bearded person, distinct from p0."),
    "p2": ("1961511", "true_unique", "reasonable", "verified", "Single dining table; box is broad but follows the visible tabletop and lower body."),
    "p3": ("667309", "true_unique", "reasonable", "verified", "Wine glass held by the left-middle person."),
    "p4": ("pending-root:crowd-shelf-351017-a", "unknown", "unknown", "unknown", "Same occluded shelf bottle-like region as the earlier pending-a candidate; crowd overlap prevents atomic admission."),
    "p5": (None, "unknown", "unknown", "unknown", "Broad blurred box spans multiple neighboring shelf bottle-like fragments; no stable single physical owner."),
    "p6": ("92415", "true_unique", "reasonable", "verified", "Dark bottle beside the held wine glass; same physical owner as fixed GT92415 with reasonable geometry."),
    "p7": ("90429", "true_unique", "reasonable", "verified", "Left of two adjacent dark foreground bottles."),
    "p8": ("91179", "true_unique", "reasonable", "verified", "Right of two adjacent dark foreground bottles, distinct from p7."),
    "p9": ("95431", "true_unique", "reasonable", "verified", "Background bottle owner95431 with coherent visible extent."),
    "p10": ("95431", "repeat", "wrong", "verified", "Narrow right-side partial rebox of the same bottle emitted at p9, despite low overlap."),
    "p11": ("478719", "true_unique", "reasonable", "verified", "Center rear person; broad box remains coherent with the same accepted visible person extent."),
    "p12": ("1489041", "true_unique", "reasonable", "verified", "Right tall foreground bottle."),
    "p13": ("1870183", "true_unique", "reasonable", "verified", "Same labeled background bottle as fixed owner1870183; geometry matches the previously accepted physical support."),
    "p14": ("666212", "true_unique", "reasonable", "verified", "Central foreground stemmed wine glass."),
    "p15": ("pending-root:crowd-shelf-351017-e", "unknown", "unknown", "unknown", "Partially occluded shelf bottle-like member behind the center person's head; remains an unadmitted crowd candidate."),
    "p16": ("667754", "true_unique", "reasonable", "verified", "Small upper central wine glass."),
    "p17": ("499060", "true_unique", "reasonable", "verified", "Right foreground person; left-clipped relative to GT but coherent with previously accepted visible extent."),
    "p18": ("667794", "true_unique", "reasonable", "verified", "Right-center stemmed glass; its visible glass extent is coherent despite occlusion by the straw vessel."),
    "p19": ("96050", "true_unique", "reasonable", "verified", "Narrow shelf bottle owner96050. It does not also cover adjacent owner2094819, whose white-label bottle lies left of this box."),
    "p20": ("667769", "true_unique", "reasonable", "verified", "Rightmost labeled stemmed wine glass."),
    "p21": (None, "false", "wrong", "wrong", "Bottle box covers yellow wall, a black shelf edge, and foreground hair; no bottle is present."),
    "p22": ("pending-new:image-000000351017:writing-pen-right", "true_unique", "wrong", "wrong", "Real writing pen held by the right person, but the box includes substantial hand/background and the emitted fork class is wrong."),
    "p23": ("pending-new:image-000000351017:dark-vessel-midright", "true_unique", "reasonable", "verified", "Distinct dark bottle-like vessel immediately right of wine glass667769, with coherent visible occlusion-bounded extent."),
    "p24": ("pending-new:image-000000351017:writing-pen-right", "repeat", "wrong", "wrong", "Small upper-tip rebox of the same writing pen first emitted at p22; fork class remains wrong."),
    "p25": ("pending-new:image-000000351017:dark-vessel-right", "true_unique", "reasonable", "verified", "Second distinct dark bottle-like vessel at the right person's shirt edge, separate from p23."),
}

sheet_for = {}
for start in range(0, 26, 6):
    end = min(start + 5, 25)
    for i in range(start, end + 1):
        sheet_for[f"p{i}"] = str(OUT / f"crop-sheet-p{start:02d}-p{end:02d}.png")

notes = [{
    "record_type": "overview_view",
    "image_id": 351017,
    "checkpoint_step": 32,
    "viewed_paths": [original, target_overlay, raw_overlay],
    "interpretation": "Restaurant table with four people, labeled tableware and dense shelf bottles. Raw overlay has 26 parsed-valid rows. Fixed owner96316 is the small far-left shelf bottle and has no raw box; raw p19 is owner96050 and does not also cover adjacent owner2094819.",
}]
decisions = []
for row in rows:
    pid = row["prediction_id"]
    owner, physical, extent, cls, interpretation = spec[pid]
    crop_paths = [row["shared_crop_paths"]["context"], row["shared_crop_paths"]["tight"]]
    evidence = [original, target_overlay, raw_overlay, *crop_paths, sheet_for[pid]]
    if pid in {"p4", "p5", "p9", "p10", "p13", "p15", "p19", "p21"}:
        evidence.append(str(OUT / "shelf-owner-detail.png"))
    if pid in {"p6"}:
        evidence.append(str(OUT / "new-fixed-92415-detail.png"))
    if pid in {"p20", "p22", "p23", "p24", "p25"}:
        evidence.append(str(OUT / "table-tail-detail.png"))
    notes.append({
        "record_type": "visual_group_view",
        "image_id": 351017,
        "checkpoint_step": 32,
        "prediction_id": pid,
        "generated_order": row["generated_order"],
        "visual_group_id": row["visual_group_id"],
        "raw_bbox_pixel_xyxy": row["raw_bbox_pixel_xyxy"],
        "coord_bins_1000": row["coord_bins_1000"],
        "description": row["description"],
        "viewed_paths": evidence,
        "interpretation": interpretation,
        "provisional_decision": {
            "owner_id": owner,
            "physical_status": physical,
            "extent": extent,
            "class": cls,
        },
    })
    eligible = owner in target_ids and physical in {"true_unique", "repeat"} and extent == "reasonable"
    positive = eligible and physical == "true_unique"
    decisions.append({
        "proposal_id": f"third-fit:step32:image-000000351017:{pid}",
        "prediction_id": pid,
        "generated_order": row["generated_order"],
        "visual_group_id": row["visual_group_id"],
        "owner_id": owner,
        "physical_status": physical,
        "extent": extent,
        "class": cls,
        "coverage_eligible": eligible,
        "direct_CE": {
            "bbox": "positive" if positive else "mask",
            "description": "positive" if positive and cls == "verified" else "mask",
        },
        "reason": "Viewed original, target/raw bbox overlays, and this group's context and tight crops: " + interpretation,
        "evidence_paths": evidence,
    })

with open(NOTES_PATH, "w") as f:
    for note in notes:
        f.write(json.dumps(note, sort_keys=True) + "\n")

covered = sorted({x["owner_id"] for x in decisions if x["coverage_eligible"] and x["owner_id"] in target_ids})
missing = sorted(set(target_ids) - set(covered))
retained = sorted(set(parent_covered) & set(covered))
lost = sorted(set(parent_covered) - set(covered))
new_fixed = sorted(set(covered) - set(parent_covered))

new_candidates = [
    {
        "owner_id": "pending-root:crowd-shelf-351017-a",
        "source_prediction_ids": ["p4"],
        "reference_proposals": ["third-fit:step32:image-000000351017:p4"],
        "actual_bbox_pixel_xyxy": [371, 60, 403, 128],
        "full_reference_proposal": None,
        "physical_status": "unknown",
        "extent": "unknown",
        "class": "unknown",
        "category": "bottle-like",
        "status": "carried_pending_root_not_admitted",
        "valid_unlabeled_candidate": False,
        "reason": "Crowd overlap still prevents a stable full atomic reference.",
        "evidence_paths": [raw_overlay, rows[4]["shared_crop_paths"]["context"], rows[4]["shared_crop_paths"]["tight"], str(OUT / "shelf-owner-detail.png")],
    },
    {
        "owner_id": "pending-root:crowd-shelf-351017-e",
        "source_prediction_ids": ["p15"],
        "reference_proposals": ["third-fit:step32:image-000000351017:p15"],
        "actual_bbox_pixel_xyxy": [721, 25, 755, 79],
        "full_reference_proposal": None,
        "physical_status": "unknown",
        "extent": "unknown",
        "class": "unknown",
        "category": "bottle-like",
        "status": "carried_pending_root_not_admitted",
        "valid_unlabeled_candidate": False,
        "reason": "Occlusion behind the center person's head prevents a stable full atomic reference.",
        "evidence_paths": [raw_overlay, rows[15]["shared_crop_paths"]["context"], rows[15]["shared_crop_paths"]["tight"], str(OUT / "shelf-owner-detail.png")],
    },
    {
        "owner_id": "pending-new:image-000000351017:writing-pen-right",
        "source_prediction_ids": ["p22", "p24"],
        "reference_proposals": ["third-fit:step32:image-000000351017:p22", "third-fit:step32:image-000000351017:p24"],
        "actual_bbox_pixel_xyxy": [910, 433, 1011, 513],
        "full_reference_proposal": {
            "category": "pen",
            "bbox_pixel_xyxy": [930, 433, 1011, 506],
            "reference_coord_bins_1000": [745, 520, 810, 608],
            "status": "root_review_required_non_v4_category",
        },
        "physical_status": "true_unique_with_repeat",
        "extent": "wrong",
        "class": "wrong",
        "category": "pen",
        "status": "pending_root_noncatalog_class",
        "valid_unlabeled_candidate": False,
        "reason": "The real object is a writing pen rather than the emitted fork; p24 is its partial repeat.",
        "evidence_paths": [raw_overlay, rows[22]["shared_crop_paths"]["context"], rows[22]["shared_crop_paths"]["tight"], rows[24]["shared_crop_paths"]["context"], rows[24]["shared_crop_paths"]["tight"], str(OUT / "table-tail-detail.png")],
    },
    {
        "owner_id": "pending-new:image-000000351017:dark-vessel-midright",
        "source_prediction_ids": ["p23"],
        "reference_proposals": ["third-fit:step32:image-000000351017:p23"],
        "actual_bbox_pixel_xyxy": [958, 337, 1001, 423],
        "full_reference_proposal": {
            "category": "bottle",
            "bbox_pixel_xyxy": [958, 337, 1001, 423],
            "reference_coord_bins_1000": [768, 405, 802, 509],
            "status": "candidate_visible_extent",
        },
        "physical_status": "true_unique",
        "extent": "reasonable",
        "class": "verified",
        "category": "bottle",
        "status": "pending_root",
        "valid_unlabeled_candidate": True,
        "reason": "Distinct dark bottle-like vessel with coherent visible occlusion-bounded extent; not in fixed v4.",
        "evidence_paths": [raw_overlay, rows[23]["shared_crop_paths"]["context"], rows[23]["shared_crop_paths"]["tight"], str(OUT / "table-tail-detail.png")],
    },
    {
        "owner_id": "pending-new:image-000000351017:dark-vessel-right",
        "source_prediction_ids": ["p25"],
        "reference_proposals": ["third-fit:step32:image-000000351017:p25"],
        "actual_bbox_pixel_xyxy": [1000, 349, 1031, 409],
        "full_reference_proposal": {
            "category": "bottle",
            "bbox_pixel_xyxy": [1000, 349, 1031, 409],
            "reference_coord_bins_1000": [801, 419, 826, 492],
            "status": "candidate_visible_extent",
        },
        "physical_status": "true_unique",
        "extent": "reasonable",
        "class": "verified",
        "category": "bottle",
        "status": "pending_root",
        "valid_unlabeled_candidate": True,
        "reason": "Second distinct dark bottle-like vessel at the shirt edge; not in fixed v4.",
        "evidence_paths": [raw_overlay, rows[25]["shared_crop_paths"]["context"], rows[25]["shared_crop_paths"]["tight"], str(OUT / "table-tail-detail.png")],
    },
]

review = {
    "schema": "third_fit_step32_owner_review.v1",
    "image_id": 351017,
    "checkpoint_step": 32,
    "target_catalog": {"path": str(TARGET_PATH), "sha256": sha256(TARGET_PATH)},
    "source_packet": {"path": str(PACKET_PATH), "sha256": sha256(PACKET_PATH)},
    "status": "candidate_ready",
    "raw_row_count": len(rows),
    "decisions": decisions,
    "summary": {
        "target_owner_count": len(target_ids),
        "covered_owner_ids": covered,
        "missing_owner_ids": missing,
        "covered_owner_count": len(covered),
        "missing_owner_count": len(missing),
        "physical_repeat_row_count": sum(x["physical_status"] == "repeat" for x in decisions),
        "confirmed_false_row_count": sum(x["physical_status"] == "false" for x in decisions),
        "physical_unknown_row_count": sum(x["physical_status"] == "unknown" for x in decisions),
        "invalid_output_row_count": sum(x["physical_status"] == "invalid_output" for x in decisions),
        "class_wrong_row_count": sum(x["class"] == "wrong" for x in decisions),
        "class_unknown_row_count": sum(x["class"] == "unknown" for x in decisions),
        "raw_stop_reason": "im_end",
        "cap_debt": 0,
        "parent_covered_owner_ids": parent_covered,
        "retained_parent_owner_ids": retained,
        "lost_parent_owner_ids": lost,
        "newly_covered_fixed_owner_ids": new_fixed,
        "appended_gt_repair": {
            "qualified": False,
            "owner_id": "96316",
            "reference_coord_bins_1000": [265, 85, 278, 156],
            "reason": "Owner96316 is a visible small shelf bottle, but no third-fit step32 raw row covers its physical identity with reasonable geometry.",
        },
        "earliest_observed_repeat_or_structural_issue": {
            "kind": "physical_repeat_wrong_extent",
            "prediction_id": "p10",
            "generated_order": 10,
            "reason": "Partial low-overlap rebox of owner95431 after new fixed owner92415 was already recovered at p6.",
        },
        "scientific_ruling": "A parent-absent fixed owner, 92415, enters natural greedy at p6 before the first repeat at p10 and before natural im_end. This is offset by loss of parent owner2094819, so fixed coverage remains 17/25. Intended appended owner96316 is still absent.",
    },
    "new_owner_candidates": new_candidates,
    "validation": {},
}

# Deterministic validation of rows, group membership, policies, partitions, and hashes.
assert [x["prediction_id"] for x in decisions] == [x["prediction_id"] for x in rows]
assert [x["generated_order"] for x in decisions] == list(range(26))
assert len({x["proposal_id"] for x in decisions}) == 26
assert {(g["visual_group_id"], p) for g in groups.values() for p in g["member_prediction_ids"]} == {(x["visual_group_id"], x["prediction_id"]) for x in decisions}
assert set(covered) | set(missing) == set(target_ids)
assert not set(covered) & set(missing)
assert covered == sorted({x["owner_id"] for x in decisions if x["coverage_eligible"] and x["owner_id"] in target_ids})
assert all(x["coverage_eligible"] == (x["owner_id"] in target_ids and x["physical_status"] in {"true_unique", "repeat"} and x["extent"] == "reasonable") for x in decisions)
assert all(x["direct_CE"] == {"bbox": "mask", "description": "mask"} for x in decisions if x["physical_status"] != "true_unique" or not x["coverage_eligible"])
assert all(x["physical_status"] in {"true_unique", "repeat", "false", "unknown", "invalid_output"} for x in decisions)
assert all(x["extent"] in {"reasonable", "wrong", "unknown"} for x in decisions)
assert all(x["class"] in {"verified", "wrong", "unknown"} for x in decisions)
assert sha256(TARGET_PATH) == packet["source"]["target_catalog_v4"]["sha256"]
assert sha256(PARENT_PATH) == packet["source"]["parent_v4_physical_ledger"]["sha256"]
assert len(notes) == 27
assert {x["prediction_id"] for x in notes if x["record_type"] == "visual_group_view"} == {x["prediction_id"] for x in rows}
valid_new = [x for x in new_candidates if x["valid_unlabeled_candidate"]]
assert all(x["owner_id"].startswith("pending-new:") and x["actual_bbox_pixel_xyxy"] and x["full_reference_proposal"] and x["physical_status"] == "true_unique" and x["extent"] == "reasonable" and x["class"] == "verified" and x["evidence_paths"] for x in valid_new)

declared = [
    packet["rendered"]["original_image"],
    packet["rendered"]["raw_generated_overlay"],
    packet["rendered"]["target_catalog_overlay"],
]
for group in groups.values():
    declared.extend([group["context_crop"], group["tight_crop"]])
for item in declared:
    assert Path(item["path"]).is_file()
    assert sha256(item["path"]) == item["sha256"]
all_evidence = sorted({p for x in decisions for p in x["evidence_paths"]})
assert all(Path(p).is_file() for p in all_evidence)

review["validation"] = {
    "status": "passed",
    "raw_prediction_ids_exact": True,
    "visual_group_membership_exact": True,
    "generated_orders_exact": True,
    "decision_count": len(decisions),
    "coverage_partition_exact": True,
    "coverage_eligibility_recomputed": True,
    "parent_retention_recomputed": True,
    "field_domains_checked": True,
    "mask_policy_checked": True,
    "source_packet_sha256_checked": True,
    "target_catalog_sha256_checked": True,
    "parent_ledger_sha256_checked": True,
    "packet_declared_visual_hashes_checked": len(declared),
    "decision_evidence_paths_exist": len(all_evidence),
    "review_notes_line_count": len(notes),
    "valid_unlabeled_candidate_contract_checked": len(valid_new),
}
with open(REVIEW_PATH, "w") as f:
    json.dump(review, f, indent=2)
    f.write("\n")
print(REVIEW_PATH)
print(NOTES_PATH)
