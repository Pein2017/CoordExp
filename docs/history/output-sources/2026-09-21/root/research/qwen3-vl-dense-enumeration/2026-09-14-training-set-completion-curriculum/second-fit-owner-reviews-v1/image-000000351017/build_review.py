import hashlib
import json
from pathlib import Path

BASE = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum")
PACKET_PATH = BASE / "second-fit-review-packets-v1/image-000000351017/packet.json"
TARGET_PATH = BASE / "target-owners-complete-v3.json"
OUT_DIR = BASE / "second-fit-owner-reviews-v1/image-000000351017"
OUT_PATH = OUT_DIR / "review.json"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


packet = json.load(open(PACKET_PATH))
rows = packet["rendered"]["raw_rows"]
row_by_id = {row["prediction_id"]: row for row in rows}
original = packet["rendered"]["original_image"]["path"]
raw_overlay = packet["rendered"]["raw_generated_overlay"]["path"]
target_overlay = packet["rendered"]["target_catalog_overlay"]["path"]

parent_covered = [
    "1489041", "1870183", "1961511", "2094819", "466970", "476801",
    "478719", "499060", "666212", "667309", "667754", "667769",
    "667794", "90429", "91179", "95431", "96050",
]
all_targets = sorted(str(x["owner_id"]) for x in packet["target_catalog_references"])
missing = sorted(set(all_targets) - set(parent_covered))

spec = {
    "p0": ("466970", "true_unique", "reasonable", "verified", "Left foreground person; full visible body support is coherent."),
    "p1": ("476801", "true_unique", "reasonable", "verified", "Left-middle bearded person; distinct from p0."),
    "p2": ("1961511", "true_unique", "reasonable", "verified", "The single visible dining table."),
    "p3": ("667309", "true_unique", "reasonable", "verified", "Wine glass held by the left-middle person."),
    "p4": ("pending-root:crowd-shelf-351017-a", "unknown", "unknown", "unknown", "Occluded shelf bottle-like member in the separate crowd region; atomic boundary remains unresolved and it is not admitted."),
    "p5": ("pending-root:crowd-shelf-351017-b", "unknown", "unknown", "unknown", "Blurred shelf bottle-like member in the separate crowd region; atomic boundary remains unresolved and it is not admitted."),
    "p6": ("pending-root:crowd-shelf-351017-c", "unknown", "unknown", "unknown", "Blurred shelf bottle-like member overlapping neighbors in the separate crowd region; not admitted."),
    "p7": ("pending-root:crowd-shelf-351017-d", "unknown", "unknown", "unknown", "Narrow shelf bottle-like fragment in the separate crowd region; full atomic extent remains unresolved and it is not admitted."),
    "p8": ("90429", "true_unique", "reasonable", "verified", "Left of two adjacent dark foreground bottles."),
    "p9": ("91179", "true_unique", "reasonable", "verified", "Right of two adjacent dark foreground bottles; distinct from p8."),
    "p10": ("95431", "true_unique", "reasonable", "verified", "Existing background bottle with coherent visible extent."),
    "p11": ("95431", "repeat", "wrong", "verified", "Partial lower rebox of the same physical background bottle already emitted at p10."),
    "p12": ("478719", "true_unique", "reasonable", "verified", "Center rear person; distinct from the other person boxes."),
    "p13": (None, "unknown", "unknown", "unknown", "Box straddles adjacent foreground bottle boundaries 1868425 and 1489041; no stable single owner."),
    "p14": ("1489041", "true_unique", "reasonable", "verified", "Right tall foreground bottle."),
    "p15": ("1870183", "true_unique", "reasonable", "verified", "Existing labeled background bottle with coherent visible extent."),
    "p16": ("666212", "true_unique", "reasonable", "verified", "Central foreground stemmed wine glass."),
    "p17": ("pending-root:crowd-shelf-351017-e", "unknown", "unknown", "unknown", "Partially occluded shelf bottle-like member behind the center person's head; crowd overlap prevents atomic admission."),
    "p18": ("667754", "true_unique", "reasonable", "verified", "Small upper central wine glass."),
    "p19": ("499060", "true_unique", "reasonable", "verified", "Right foreground person; distinct from the other three persons."),
    "p20": ("667794", "true_unique", "reasonable", "verified", "Right-center stemmed wine glass."),
    "p21": ("2094819", "true_unique", "reasonable", "verified", "Large labeled background bottle."),
    "p22": ("96050", "true_unique", "reasonable", "verified", "Narrow labeled background bottle immediately right of p21."),
    "p23": ("667769", "true_unique", "reasonable", "verified", "Rightmost labeled stemmed wine glass."),
    "p24": ("96050", "repeat", "reasonable", "verified", "Second coherent box of the same physical bottle already emitted at p22; repeat despite IoU below 0.95."),
    "p25": ("96050", "repeat", "wrong", "verified", "Narrower partial rebox of owner 96050 already emitted at p22/p24."),
    "p26": ("96050", "repeat", "wrong", "verified", "Right-edge partial rebox of owner 96050; it also reaches toward the adjacent shelf object."),
    "p27": ("pending-new:image-000000351017:shelf-bottle-right", "true_unique", "unknown", "verified", "A distinct red-capped shelf bottle-like object immediately right of owner 96050; occlusion by the foreground head leaves full extent unresolved, so it remains pending root."),
    "p28": ("499060", "repeat", "wrong", "verified", "A substantially left-clipped rebox of the same right foreground person already emitted at p19."),
    "p29": ("pending-new:image-000000351017:shelf-bottle-right", "repeat", "wrong", "verified", "Right-side sliver rebox of the same physical shelf bottle-like object first emitted at p27, despite negligible IoU."),
    "p30": ("pending-new:image-000000351017:table-utensil-right", "true_unique", "unknown", "unknown", "A real small horizontal utensil/pen-overlap region beside the right hand; physical object is distinct, but category and full boundary are too blurred for admission."),
    "p31": ("pending-new:image-000000351017:dark-vessel-midright", "true_unique", "unknown", "unknown", "A distinct dark vessel-like object immediately right of wine glass 667769; occlusion and reflections leave category and full extent unresolved."),
    "p32": ("pending-new:image-000000351017:dark-vessel-right", "true_unique", "unknown", "unknown", "A second dark vessel-like object at the right person's shirt edge; physical separation is visible, but category and full extent remain unresolved."),
}

decisions = []
for row in rows:
    pid = row["prediction_id"]
    owner, physical, extent, cls, reason = spec[pid]
    eligible = owner in all_targets and physical in {"true_unique", "repeat"} and extent == "reasonable"
    unique_positive = eligible and physical == "true_unique"
    ev = [
        original,
        target_overlay,
        raw_overlay,
        row["shared_crop_paths"]["context"],
        row["shared_crop_paths"]["tight"],
    ]
    if pid in {"p4", "p5", "p6", "p7", "p10", "p11", "p15", "p17", "p21", "p22", "p24", "p25", "p26", "p27", "p29"}:
        ev.append(str(OUT_DIR / "shelf-detail-gt96316.png"))
    if pid in {"p23", "p30", "p31", "p32"}:
        ev.append(str(OUT_DIR / "suffix-table-detail.png"))
    decisions.append({
        "proposal_id": f"second-fit:step16:image-000000351017:{pid}",
        "prediction_id": pid,
        "generated_order": row["generated_order"],
        "visual_group_id": row["visual_group_id"],
        "owner_id": owner,
        "physical_status": physical,
        "extent": extent,
        "class": cls,
        "coverage_eligible": eligible,
        "direct_CE": {
            "bbox": "positive" if unique_positive else "mask",
            "description": "positive" if unique_positive and cls == "verified" else "mask",
        },
        "reason": "Viewed original, target/raw overlays, and the corresponding context and tight crops: " + reason,
        "evidence_paths": ev,
    })

carried = []
for suffix, pid in zip("abcde", ["p4", "p5", "p6", "p7", "p17"]):
    carried.append({
        "owner_id": f"pending-root:crowd-shelf-351017-{suffix}",
        "source_prediction_ids": [pid],
        "reference_proposals": [f"second-fit:step16:image-000000351017:{pid}"],
        "category": "bottle-like",
        "status": "carried_pending_root_not_admitted",
        "direct_CE": {"bbox": "mask", "description": "mask"},
        "reason": "Earlier crowd-overlapping shelf candidate remains unresolved and is not admitted by this review.",
    })

new_candidates = carried + [
    {
        "owner_id": "pending-new:image-000000351017:shelf-bottle-right",
        "source_prediction_ids": ["p27", "p29"],
        "reference_proposals": [
            "second-fit:step16:image-000000351017:p27",
            "second-fit:step16:image-000000351017:p29",
        ],
        "category": "bottle",
        "physical_status": "true_unique_with_repeat",
        "extent": "unknown",
        "class": "verified",
        "status": "pending_root",
        "direct_CE": {"bbox": "mask", "description": "mask"},
        "reason": "Distinct visible shelf bottle-like object; p29 is a partial rebox of p27 and neither row changes the fixed-v3 denominator.",
    },
    {
        "owner_id": "pending-new:image-000000351017:table-utensil-right",
        "source_prediction_ids": ["p30"],
        "reference_proposals": ["second-fit:step16:image-000000351017:p30"],
        "category": "unknown utensil/pen overlap",
        "physical_status": "true_unique",
        "extent": "unknown",
        "class": "unknown",
        "status": "pending_root",
        "direct_CE": {"bbox": "mask", "description": "mask"},
        "reason": "A real small object is visible, but blur and overlap prevent a reliable fork-versus-pen and full-extent ruling.",
    },
    {
        "owner_id": "pending-new:image-000000351017:dark-vessel-midright",
        "source_prediction_ids": ["p31"],
        "reference_proposals": ["second-fit:step16:image-000000351017:p31"],
        "category": "unknown vessel",
        "physical_status": "true_unique",
        "extent": "unknown",
        "class": "unknown",
        "status": "pending_root",
        "direct_CE": {"bbox": "mask", "description": "mask"},
        "reason": "Distinct dark vessel-like object beside known wine glass 667769; occlusion prevents category and extent admission.",
    },
    {
        "owner_id": "pending-new:image-000000351017:dark-vessel-right",
        "source_prediction_ids": ["p32"],
        "reference_proposals": ["second-fit:step16:image-000000351017:p32"],
        "category": "unknown vessel",
        "physical_status": "true_unique",
        "extent": "unknown",
        "class": "unknown",
        "status": "pending_root",
        "direct_CE": {"bbox": "mask", "description": "mask"},
        "reason": "Second distinct dark vessel-like object at the shirt edge; occlusion prevents category and extent admission.",
    },
]

review = {
    "schema": "second_fit_step16_owner_review.v1",
    "image_id": 351017,
    "checkpoint_step": 16,
    "target_catalog": {"path": str(TARGET_PATH), "sha256": sha256(TARGET_PATH)},
    "source_packet": {"path": str(PACKET_PATH), "sha256": sha256(PACKET_PATH)},
    "status": "candidate_ready",
    "raw_row_count": len(rows),
    "decisions": decisions,
    "summary": {
        "target_owner_count": len(all_targets),
        "covered_owner_ids": parent_covered,
        "missing_owner_ids": missing,
        "covered_owner_count": len(parent_covered),
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
        "retained_parent_owner_ids": parent_covered,
        "lost_parent_owner_ids": [],
        "newly_covered_fixed_owner_ids": [],
        "appended_gt_repair": {
            "qualified": False,
            "owner_id": "96316",
            "reference_coord_bins_1000": [265, 85, 278, 156],
            "reason": "The fixed GT is visibly a small shelf bottle, but no step16 raw row covers its physical identity with reasonable geometry; nearest shelf predictions are different objects.",
        },
        "earliest_observed_repeat_or_structural_issue": {
            "kind": "physical_repeat_wrong_extent",
            "prediction_id": "p11",
            "generated_order": 11,
            "reason": "Partial rebox of owner 95431 appears before any absent fixed owner is recovered.",
        },
        "scientific_ruling": "No previously absent fixed owner enters natural step16 output. The output reaches natural im_end without parser errors or cap debt, but after owner repeats and unresolved/new physical objects; stage03's appended owner 96316 is not naturally recovered at step16.",
    },
    "new_owner_candidates": new_candidates,
    "validation": {},
}

# Deterministic validation, including every packet-declared visual hash.
assert [x["prediction_id"] for x in decisions] == [x["prediction_id"] for x in rows]
assert [x["generated_order"] for x in decisions] == list(range(33))
assert len({x["proposal_id"] for x in decisions}) == 33
assert {
    (group["visual_group_id"], pid)
    for group in packet["rendered"]["visual_groups"]
    for pid in group["member_prediction_ids"]
} == {(x["visual_group_id"], x["prediction_id"]) for x in decisions}
assert set(review["summary"]["covered_owner_ids"]) | set(review["summary"]["missing_owner_ids"]) == set(all_targets)
assert not (set(review["summary"]["covered_owner_ids"]) & set(review["summary"]["missing_owner_ids"]))
assert all(x["coverage_eligible"] == (x["owner_id"] in all_targets and x["physical_status"] in {"true_unique", "repeat"} and x["extent"] == "reasonable") for x in decisions)
assert all(x["direct_CE"] == {"bbox": "mask", "description": "mask"} for x in decisions if x["physical_status"] != "true_unique" or not x["coverage_eligible"])
assert all(x["physical_status"] in {"true_unique", "repeat", "false", "unknown", "invalid_output"} for x in decisions)
assert all(x["extent"] in {"reasonable", "wrong", "unknown"} for x in decisions)
assert all(x["class"] in {"verified", "wrong", "unknown"} for x in decisions)
assert sha256(TARGET_PATH) == packet["source"]["target_catalog_v3"]["sha256"]

declared = [
    packet["rendered"]["original_image"],
    packet["rendered"]["raw_generated_overlay"],
    packet["rendered"]["target_catalog_overlay"],
]
for group in packet["rendered"]["visual_groups"]:
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
    "field_domains_checked": True,
    "mask_policy_checked": True,
    "source_packet_sha256_checked": True,
    "target_catalog_sha256_checked": True,
    "packet_declared_visual_hashes_checked": len(declared),
    "decision_evidence_paths_exist": len(all_evidence),
}
with open(OUT_PATH, "w") as f:
    json.dump(review, f, indent=2)
    f.write("\n")
print(OUT_PATH)
