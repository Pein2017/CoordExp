import hashlib
import json
from pathlib import Path


B = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum")
OWNED = B / "third-fit-owner-reviews-v1/image-000000059571"
PACKET_PATH = B / "third-fit-review-packets-v1/image-000000059571/packet.json"
CATALOG_PATH = B / "target-owners-complete-v4.json"
SCORED_PATH = B / "third-fit-selectors-v1/scored-step-32.json"
PARENT_LEDGER_PATH = B / "parent16-v4-physical-ledger-v1/ledger.json"
REVIEW_PATH = OWNED / "review.json"
NOTES_PATH = OWNED / "review-notes.jsonl"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


packet = json.loads(PACKET_PATH.read_text())
catalog = json.loads(CATALOG_PATH.read_text())
scored = json.loads(SCORED_PATH.read_text())
parent = json.loads(PARENT_LEDGER_PATH.read_text())
groups = {g["visual_group_id"]: g for g in packet["rendered"]["visual_groups"]}
rows = packet["rendered"]["raw_rows"]
target_ids = sorted(r["owner_id"] for r in catalog["records"] if r["image_id"] == 59571)
parent_image = next(r for r in parent["images"] if r.get("image_id") == 59571 and "covered_owner_ids" in r)
parent_covered = sorted(parent_image["covered_owner_ids"])
scored_row = next(r for r in scored["rows"] if r["image_id"] == 59571)


rules = {}


def put(group_ids, owner_id, physical, extent, cls, reason):
    for gid in group_ids:
        assert gid not in rules, gid
        rules[gid] = {
            "owner_id": owner_id,
            "physical": physical,
            "extent": extent,
            "class": cls,
            "reason": reason,
        }


def gids(a, b):
    return [f"g{i:04d}" for i in range(a, b + 1)]


put(["g0000"], "1126212", "owner", "reasonable", "verified", "The box reasonably covers the fixed oven/cooktop owner.")
put(["g0001"], "191081", "owner", "reasonable", "verified", "The box reasonably covers the central woman.")
put(["g0002"], "1487030", "owner", "reasonable", "verified", "The box reasonably covers the foreground bottle.")
put(["g0003"], None, "false", "wrong", "wrong", "The bottle box is on the central woman's face; no bottle occupies the proposed extent.")
put(["g0004"], "new:59571:pink-pump-bottle", "owner", "reasonable", "verified", "The box reasonably covers the admitted pink pump-bottle owner.")
put(["g0005"], "new:59571:cabinet-dark-bottle", "owner", "reasonable", "verified", "The box has the admitted cabinet dark bottle as its dominant object; its small rightward excess remains reasonable.")
put(gids(6, 8), None, "unknown", "unknown", "unknown", "The cabinet reflection/glassware crop does not resolve one distinct atomic bottle owner or a trustworthy whole-object boundary.")
put(["g0009"], None, "false", "wrong", "wrong", "The small box lies on a wall-mounted display element, not a bottle.")
put(["g0010"], None, "false", "wrong", "wrong", "The small box lies on a tan wall-display element, not a bottle.")
put(["g0011"], "new:59571:back-counter-metal-container", "owner", "reasonable", "unknown", "The box reasonably covers the admitted back-counter metal-container owner; its generated bottle class is not sufficiently verifiable against the category-null target.")
put(["g0012"], "1722837", "owner", "reasonable", "verified", "The box reasonably covers the lower foreground person.")
put(["g0013"], "676828", "owner", "reasonable", "verified", "The box reasonably covers the fixed white cup.")
put(["g0014"], "1120658", "owner", "reasonable", "verified", "The box reasonably covers the fixed microwave.")
put(["g0015"], "684390", "owner", "reasonable", "verified", "The box reasonably covers the fixed lidded cup.")
put(["g0016"], "684894", "owner", "reasonable", "verified", "The box reasonably covers the fixed red cup.")
put(["g0017"], None, "unknown", "wrong", "verified", "The person box merges the photographer and the standing man, so it cannot support one atomic person owner.")
put(["g0018"], "1885496", "owner", "reasonable", "verified", "The box reasonably covers the small white cup.")
put(["g0019"], "1217358", "owner", "reasonable", "verified", "The box reasonably covers the lower-right foreground person.")
put(["g0020"], None, "false", "wrong", "wrong", "The proposed book box is on camera/monitor equipment; no book occupies the extent.")
put(["g0021"], None, "invalid_output", "unknown", "unknown", "The raw parser reports a malformed object span; no physical or class judgment is inferred.")
put(["g0022"], "pending-new:59571:upper-right-horizontal-book", "owner", "reasonable", "verified", "The box isolates one horizontal blue book-like object on the upper-right shelf, outside the fixed-v4 catalog.")
put(["g0023"], None, "false", "wrong", "wrong", "The narrow box lies on wooden shelf trim above fixed bottle 2094968; it does not cover a book or that bottle.")
put(["g0024"], "pending-new:59571:upper-right-horizontal-book", "owner", "reasonable", "verified", "This is a slightly larger rebox of the same horizontal blue shelf book-like object as p22.")
put(["g0025"], None, "false", "wrong", "wrong", "The narrow box lies on the opposite wooden shelf/column edge, not a book.")
put(gids(26, 38), None, "unknown", "wrong", "wrong", "The box lies on wall-display graphics, trim, or hardware; no atomic physical book boundary is supported.")
put(["g0039"], None, "invalid_output", "unknown", "unknown", "The raw parser reports invalid geometry; no physical or class judgment is inferred.")
put(["g0040"], None, "false", "wrong", "wrong", "The exact-signature group lies on wall trim/hardware with no book present.")
put(["g0041"], None, "false", "wrong", "wrong", "The exact-signature group lies on wall trim/hardware with no book present.")
put(["g0042"], None, "unknown", "wrong", "wrong", "The oversized upper-right shelf box spans multiple objects and cannot support one atomic book owner.")
put(["g0043"], None, "unknown", "wrong", "wrong", "The right-edge shelf box spans container/trim fragments and cannot support one atomic book owner.")
put(["g0044"], None, "unknown", "wrong", "wrong", "The clipped image-edge box does not establish one atomic book owner.")

assert set(rules) == set(groups), (sorted(set(groups) - set(rules)), sorted(set(rules) - set(groups)))

contact_sheets = [
    {"path": str(p), "sha256": sha256(p)}
    for p in sorted((OWNED / "contact-sheets").glob("groups-*.png"))
]
assert len(contact_sheets) == 4
extra_evidence = [
    {"path": str(p), "sha256": sha256(p)}
    for p in sorted((OWNED / "extra-evidence").glob("*.png"))
]

original = packet["rendered"]["original_image"]
target_overlay = packet["rendered"]["target_catalog_overlay"]
raw_overlay = packet["rendered"]["raw_generated_overlay"]
common_visual_evidence = [original["path"], target_overlay["path"], raw_overlay["path"]]

seen_owners = set()
first_in_group = set()
decisions = []
for row in sorted(rows, key=lambda r: r["generated_order"]):
    gid = row["visual_group_id"]
    rule = rules[gid]
    owner_id = rule["owner_id"]
    physical = rule["physical"]
    reason = rule["reason"]
    if physical == "owner":
        if owner_id in seen_owners:
            physical = "repeat"
            reason = "Same physical owner as an earlier generated row. " + reason
        else:
            physical = "true_unique"
            seen_owners.add(owner_id)
    elif gid in first_in_group and physical == "false":
        physical = "repeat"
        reason = "Exact duplicate of the preceding false-region prediction in this visual group. " + reason
    first_in_group.add(gid)
    extent = rule["extent"]
    cls = rule["class"]
    coverage_eligible = (
        row["status"] == "parsed_valid"
        and owner_id in target_ids
        and physical in {"true_unique", "repeat"}
        and extent == "reasonable"
    )
    direct_bbox = "mask"
    direct_description = "mask"
    if physical == "true_unique" and owner_id in target_ids and extent == "reasonable":
        direct_bbox = "positive"
        if cls == "verified":
            direct_description = "positive"
    group = groups[gid]
    sheet_path = contact_sheets[min(int(gid[1:]) // 12, 3)]["path"]
    evidence_paths = common_visual_evidence + [group["context_crop"]["path"], group["tight_crop"]["path"], sheet_path]
    if gid in {"g0005", "g0006", "g0007", "g0008"}:
        evidence_paths += [e["path"] for e in extra_evidence]
    decisions.append({
        "prediction_id": row["prediction_id"],
        "generated_order": row["generated_order"],
        "visual_group_id": gid,
        "owner_id": owner_id,
        "physical_status": physical,
        "extent": extent,
        "class": cls,
        "coverage_eligible": coverage_eligible,
        "direct_CE": {"bbox": direct_bbox, "description": direct_description},
        "reason": reason,
        "evidence_paths": evidence_paths,
    })

covered = sorted({d["owner_id"] for d in decisions if d["coverage_eligible"]})
missing = sorted(set(target_ids) - set(covered))
retained = sorted(set(covered) & set(parent_covered))
lost = sorted(set(parent_covered) - set(covered))
newly_covered = sorted(set(covered) - set(parent_covered))


def count(field, value):
    return sum(d[field] == value for d in decisions)


summary = {
    "target_owner_count": len(target_ids),
    "covered_owner_ids": covered,
    "missing_owner_ids": missing,
    "covered_owner_count": len(covered),
    "missing_owner_count": len(missing),
    "physical_repeat_row_count": count("physical_status", "repeat"),
    "confirmed_false_row_count": count("physical_status", "false"),
    "physical_unknown_row_count": count("physical_status", "unknown"),
    "invalid_output_row_count": count("physical_status", "invalid_output"),
    "class_wrong_row_count": count("class", "wrong"),
    "class_unknown_row_count": count("class", "unknown"),
    "raw_stop_reason": scored_row["decode_stop_reason"],
    "cap_debt": scored_row["cap_debt"],
    "parent_covered_owner_ids": parent_covered,
    "retained_parent_owner_ids": retained,
    "lost_parent_owner_ids": lost,
    "newly_covered_fixed_owner_ids": newly_covered,
    "stage03_appended_reference_repair": {
        "qualified": False,
        "owner_id": "2094968",
        "natural_prediction_id": None,
        "coverage_eligible": False,
        "finding": "No third-fit step32 row covers fixed bottle owner 2094968. The nearby p23 box ends above the owner's reference extent and lies on wooden trim."
    },
    "all_fixed_owner_report": {
        "covered": covered,
        "missing": missing,
        "owner_2094968": "missing"
    },
    "earliest_observed_issues": {
        "false_or_structural": {"prediction_id": "p3", "finding": "bottle box on the central woman's face"},
        "invalid_output": {"prediction_id": "p21", "drop_reason": "malformed_object_span"},
        "physical_rebox": {"prediction_id": "p24", "repeats_prediction_id": "p22", "owner_id": "pending-new:59571:upper-right-horizontal-book"},
        "exact_signature_repeat": {"prediction_id": "p41", "repeats_prediction_id": "p40"}
    },
    "before_stop_diagnosis": {
        "natural_eos": scored_row["natural_eos"],
        "capped_by_limit": scored_row["capped_by_limit"],
        "token_count": scored_row["token_count"],
        "raw_row_count": scored_row["raw_row_count"],
        "valid_prediction_count": scored_row["valid_prediction_count"],
        "raw_invalid_or_dropped_count": packet["counts"]["raw_invalid_or_dropped_count"],
        "finding": "The decode reaches natural im_end before cap, but after p21 it enters a book-labeled scan across shelf and wall regions, including exact duplicate groups, without recovering owner 2094968."
    },
    "fixed_owner_special_cases": {
        "pig_chef_figurine": {
            "owner_id": "second-fit:new:59571:pig-chef-figurine",
            "status": "missing",
            "reference_policy": "Full reference includes the attached placard and base; no row covers that full physical owner."
        },
        "foreground_work_island": {
            "owner_id": "first-fit:new:59571:foreground-work-island",
            "status": "missing",
            "class_policy": "unknown/mask_description",
            "finding": "p0 covers the oven/cooktop rather than the full foreground work-island owner."
        }
    }
}

new_owner_candidates = [
    {
        "owner_id": "pending-new:59571:upper-right-horizontal-book",
        "status": "pending_root_unlabeled_admission",
        "category": "book",
        "reference_prediction_id": "p24",
        "actual_prediction_bboxes_coord_bins_1000": {
            "p22": [934, 361, 979, 380],
            "p24": [928, 361, 979, 383]
        },
        "full_reference_proposal_coord_bins_1000": [928, 361, 979, 383],
        "full_reference_proposal_status": "reasonable_from_p24",
        "physical_status": "true_unique_at_p22_repeat_at_p24",
        "extent": "reasonable",
        "class": "verified",
        "coverage_effect": "none_until_root_admission",
        "reason": "The original, overlay, and both group crops show one horizontal blue book-like object on the upper-right shelf. p24 supplies the more complete reasonable reference proposal.",
        "evidence_paths": common_visual_evidence + [
            groups["g0022"]["context_crop"]["path"], groups["g0022"]["tight_crop"]["path"],
            groups["g0024"]["context_crop"]["path"], groups["g0024"]["tight_crop"]["path"],
            contact_sheets[1]["path"], contact_sheets[2]["path"]
        ]
    }
]

notes = []
by_group_decisions = {}
for d in decisions:
    by_group_decisions.setdefault(d["visual_group_id"], []).append(d)
for gid in sorted(groups):
    group = groups[gid]
    ds = by_group_decisions[gid]
    rule = rules[gid]
    sheet_path = contact_sheets[min(int(gid[1:]) // 12, 3)]["path"]
    evidence_paths = common_visual_evidence + [group["context_crop"]["path"], group["tight_crop"]["path"], sheet_path]
    if gid in {"g0005", "g0006", "g0007", "g0008"}:
        evidence_paths += [e["path"] for e in extra_evidence]
    notes.append({
        "schema": "third_fit_step32_visual_interpretation_note.v1",
        "image_id": 59571,
        "visual_group_id": gid,
        "member_prediction_ids": group["member_prediction_ids"],
        "viewed_surfaces": ["original", "raw_bbox_overlay", "target_overlay", "context_crop", "tight_crop", "same_image_contact_sheet"],
        "interpretation": rule["reason"],
        "row_outcomes": [
            {
                "prediction_id": d["prediction_id"],
                "owner_id": d["owner_id"],
                "physical_status": d["physical_status"],
                "extent": d["extent"],
                "class": d["class"]
            } for d in ds
        ],
        "evidence_paths": evidence_paths
    })

NOTES_PATH.write_text("".join(json.dumps(n, sort_keys=False) + "\n" for n in notes))

review = {
    "schema": "third_fit_step32_owner_review.v1",
    "image_id": 59571,
    "checkpoint_step": 32,
    "target_catalog": {"path": str(CATALOG_PATH), "sha256": sha256(CATALOG_PATH)},
    "source_packet": {"path": str(PACKET_PATH), "sha256": sha256(PACKET_PATH)},
    "status": "candidate_ready",
    "raw_row_count": len(rows),
    "source_evidence": {
        "original_image": original,
        "target_catalog_overlay": target_overlay,
        "raw_generated_overlay": raw_overlay,
        "scored_row": {"path": str(SCORED_PATH), "sha256": sha256(SCORED_PATH)},
        "parent_ledger": {"path": str(PARENT_LEDGER_PATH), "sha256": sha256(PARENT_LEDGER_PATH)},
        "contact_sheets": contact_sheets,
        "extra_evidence": extra_evidence,
        "review_notes": {"path": str(NOTES_PATH), "sha256": sha256(NOTES_PATH), "record_count": len(notes)},
        "crop_hashes_bound_by_source_packet": True
    },
    "decisions": decisions,
    "summary": summary,
    "new_owner_candidates": new_owner_candidates,
    "validation": {"script_path": str(Path(__file__).resolve()), "result": "pending", "checks": []}
}


def validate(obj):
    checks = []
    assert obj["schema"] == "third_fit_step32_owner_review.v1"
    checks.append("schema")
    assert obj["raw_row_count"] == 52 == len(obj["decisions"]) == len(rows)
    checks.append("raw_row_count")
    by_id = {d["prediction_id"]: d for d in obj["decisions"]}
    assert len(by_id) == len(rows) and set(by_id) == {r["prediction_id"] for r in rows}
    checks.append("all_raw_prediction_ids_exact_once")
    for row in rows:
        d = by_id[row["prediction_id"]]
        assert d["generated_order"] == row["generated_order"]
        assert d["visual_group_id"] == row["visual_group_id"]
    checks.append("raw_order_and_visual_group_membership")
    for gid, group in groups.items():
        assert set(group["member_prediction_ids"]) == {d["prediction_id"] for d in obj["decisions"] if d["visual_group_id"] == gid}
    checks.append("exact_group_membership")
    assert {d["physical_status"] for d in obj["decisions"]} <= {"true_unique", "repeat", "false", "unknown", "invalid_output"}
    assert {d["extent"] for d in obj["decisions"]} <= {"reasonable", "wrong", "unknown"}
    assert {d["class"] for d in obj["decisions"]} <= {"verified", "wrong", "unknown"}
    assert {d["direct_CE"]["bbox"] for d in obj["decisions"]} <= {"positive", "mask"}
    assert {d["direct_CE"]["description"] for d in obj["decisions"]} <= {"positive", "mask"}
    checks.append("field_domains")
    for d in obj["decisions"]:
        expected = d["owner_id"] in target_ids and d["physical_status"] in {"true_unique", "repeat"} and d["extent"] == "reasonable"
        assert d["coverage_eligible"] == expected
        if d["physical_status"] == "repeat" or (d["owner_id"] and d["owner_id"].startswith("pending-new:")):
            assert d["direct_CE"] == {"bbox": "mask", "description": "mask"}
        if d["physical_status"] in {"unknown", "invalid_output", "false"} or d["extent"] != "reasonable":
            assert d["direct_CE"] == {"bbox": "mask", "description": "mask"}
    checks.append("coverage_and_mask_invariants")
    assert set(obj["summary"]["covered_owner_ids"]).isdisjoint(obj["summary"]["missing_owner_ids"])
    assert set(obj["summary"]["covered_owner_ids"]) | set(obj["summary"]["missing_owner_ids"]) == set(target_ids)
    assert obj["summary"]["covered_owner_count"] + obj["summary"]["missing_owner_count"] == 23
    checks.append("fixed_target_partition")
    assert obj["summary"]["covered_owner_ids"] == covered
    assert obj["summary"]["newly_covered_fixed_owner_ids"] == ["new:59571:back-counter-metal-container", "new:59571:cabinet-dark-bottle"]
    assert obj["summary"]["lost_parent_owner_ids"] == []
    assert obj["summary"]["retained_parent_owner_ids"] == parent_covered
    checks.append("parent_retention_and_new_coverage")
    assert by_id["p5"]["owner_id"] == "new:59571:cabinet-dark-bottle" and by_id["p5"]["coverage_eligible"]
    assert by_id["p11"]["owner_id"] == "new:59571:back-counter-metal-container" and by_id["p11"]["coverage_eligible"]
    assert by_id["p11"]["direct_CE"] == {"bbox": "positive", "description": "mask"}
    assert all(d["owner_id"] != "2094968" for d in obj["decisions"])
    assert not obj["summary"]["stage03_appended_reference_repair"]["qualified"]
    checks.append("special_owner_decisions")
    assert by_id["p24"]["physical_status"] == "repeat" and by_id["p24"]["owner_id"] == by_id["p22"]["owner_id"]
    assert by_id["p41"]["physical_status"] == "repeat"
    assert all(by_id[f"p{i}"]["physical_status"] == "repeat" for i in range(43, 49))
    checks.append("physical_and_exact_repeats")
    assert obj["summary"]["invalid_output_row_count"] == 2
    assert [d["prediction_id"] for d in obj["decisions"] if d["physical_status"] == "invalid_output"] == ["p21", "p39"]
    checks.append("parser_drops_retained")
    assert len(notes) == len(groups) == 45
    assert {n["visual_group_id"] for n in notes} == set(groups)
    assert sha256(NOTES_PATH) == obj["source_evidence"]["review_notes"]["sha256"]
    checks.append("persistent_visual_notes_exact_group_coverage")
    assert sha256(PACKET_PATH) == obj["source_packet"]["sha256"]
    assert sha256(CATALOG_PATH) == obj["target_catalog"]["sha256"]
    assert sha256(original["path"]) == original["sha256"]
    assert sha256(target_overlay["path"]) == target_overlay["sha256"]
    assert sha256(raw_overlay["path"]) == raw_overlay["sha256"]
    for group in groups.values():
        assert sha256(group["context_crop"]["path"]) == group["context_crop"]["sha256"]
        assert sha256(group["tight_crop"]["path"]) == group["tight_crop"]["sha256"]
    for evidence in contact_sheets + extra_evidence:
        assert sha256(evidence["path"]) == evidence["sha256"]
    checks.append("source_and_visual_evidence_hashes")
    assert all(Path(p).exists() for d in obj["decisions"] for p in d["evidence_paths"])
    assert all(Path(p).exists() for c in obj["new_owner_candidates"] for p in c["evidence_paths"])
    checks.append("all_evidence_paths_exist")
    return checks


review["validation"]["checks"] = validate(review)
review["validation"]["result"] = "pass"
REVIEW_PATH.write_text(json.dumps(review, indent=2, sort_keys=False) + "\n")
loaded = json.loads(REVIEW_PATH.read_text())
assert loaded["validation"]["checks"] == validate(loaded)
print(json.dumps({
    "review_path": str(REVIEW_PATH),
    "review_sha256": sha256(REVIEW_PATH),
    "notes_path": str(NOTES_PATH),
    "notes_sha256": sha256(NOTES_PATH),
    "covered": loaded["summary"]["covered_owner_count"],
    "missing": loaded["summary"]["missing_owner_count"],
    "repeats": loaded["summary"]["physical_repeat_row_count"],
    "false": loaded["summary"]["confirmed_false_row_count"],
    "unknown": loaded["summary"]["physical_unknown_row_count"],
    "invalid": loaded["summary"]["invalid_output_row_count"],
    "class_wrong": loaded["summary"]["class_wrong_row_count"],
    "class_unknown": loaded["summary"]["class_unknown_row_count"],
    "new_candidates": len(loaded["new_owner_candidates"]),
    "validation": loaded["validation"]["result"]
}, indent=2))
