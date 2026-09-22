import hashlib
import json
from pathlib import Path


B = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum")
OWNED = B / "second-fit-owner-reviews-v1/image-000000059571"
PACKET_PATH = B / "second-fit-review-packets-v1/image-000000059571/packet.json"
CATALOG_PATH = B / "target-owners-complete-v3.json"
SCORED_PATH = B / "second-fit-selectors-v1/scored-step-16.json"
PARENT_LEDGER_PATH = B / "parent16-v3-physical-ledger-v2/ledger.json"
REVIEW_PATH = OWNED / "review.json"


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

target_records = [r for r in catalog["records"] if r["image_id"] == 59571]
target_ids = sorted(r["owner_id"] for r in target_records)
parent_image = next(r for r in parent["images"] if r.get("image_id") == 59571 and "covered_owner_ids" in r)
parent_covered = sorted(parent_image["covered_owner_ids"])
scored_row = next(r for r in scored["rows"] if r["image_id"] == 59571)


# Each visual group receives an identity/geometry/class decision. Owner-bearing
# groups are expanded in raw generation order: the first row for that physical
# owner is true_unique and every later row is repeat, even across different
# boxes or within one exact-signature group.
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


put(["g0000"], None, "unknown", "unknown", "unknown", "Tiny far-left shelf region cannot be resolved as one atomic book owner under the retained shelf exclusions.")
put(["g0001"], None, "unknown", "wrong", "verified", "The row covers a horizontal stack of several books rather than one atomic owner.")
put(["g0002"], None, "repeat", "wrong", "verified", "This is another box on the same non-atomic horizontal book stack as p1.")
put(gids(3, 6), None, "unknown", "unknown", "verified", "The box lies on overlapping bookshelf spines; an atomic owner boundary remains unresolved.")
put(["g0007"], "1126212", "owner", "reasonable", "verified", "The box reasonably covers the fixed oven/cooktop owner.")
put(["g0008"], "191081", "owner", "reasonable", "verified", "The box reasonably covers the central woman.")
put(["g0009"], "1487030", "owner", "reasonable", "verified", "The box reasonably covers the foreground bottle.")
put(["g0010"], "new:59571:pink-pump-bottle", "owner", "reasonable", "verified", "The box reasonably covers the admitted pink pump-bottle owner.")
put(gids(11, 13), None, "unknown", "wrong", "unknown", "The mirror-region box does not establish one atomic physical wine-glass owner.")
put(["g0014"], "1722837", "owner", "reasonable", "verified", "The box reasonably covers the lower foreground person.")
put(["g0015"], "676828", "owner", "reasonable", "verified", "The box reasonably covers the fixed white cup.")
put(["g0016"], "1120658", "owner", "reasonable", "verified", "The box reasonably covers the fixed microwave.")
put(["g0017"], "684390", "owner", "reasonable", "verified", "The box reasonably covers the fixed lidded cup.")
put(["g0018"], "684894", "owner", "reasonable", "verified", "The box reasonably covers the fixed red cup.")
put(["g0019"], None, "unknown", "wrong", "verified", "The box merges the two right-side people and cannot support one physical owner.")
put(["g0020"], "1885496", "owner", "reasonable", "verified", "The box reasonably covers the small white cup.")
put(["g0021"], "1217358", "owner", "reasonable", "verified", "The box reasonably covers the lower-right foreground person.")
put(["g0022"], "2094968", "owner", "reasonable", "wrong", "The box isolates fixed bottle owner 2094968, but the generated description is person.")
put(["g0023"], None, "false", "wrong", "wrong", "The box is on the ceiling beam; no person is present.")
put(["g0024"], None, "unknown", "wrong", "wrong", "The box spans multiple upper-right shelf objects; it is not a person and has no atomic owner assignment.")
put(gids(25, 32), "pending-new:59571:upper-right-shelf-container", "owner", "reasonable", "wrong", "The box isolates the same small red-blue shelf container at the image edge; person is the wrong class.")
put(gids(33, 43), None, "invalid_output", "unknown", "unknown", "The raw parser reports invalid geometry; no physical or class judgment is inferred from the malformed row.")
put(gids(44, 70), None, "false", "wrong", "wrong", "The box lies on ceiling, beam, rack, or hanging hardware with no person present.")
put(["g0071"], "pending-new:59571:chef-display-statue", "owner", "reasonable", "wrong", "The box reasonably isolates the decorative chef display statue; person is the wrong class.")
put(["g0072"], "pending-new:59571:chef-display-statue", "owner", "wrong", "wrong", "The box repeats only the chef statue's hat/head region and is not a reasonable whole-object extent.")
put(gids(73, 76), None, "false", "wrong", "wrong", "The box lies on books, shelving, or the wooden column with no person present.")
put(gids(77, 101), None, "false", "wrong", "wrong", "The box lies on product displays or the microwave area with no person inside the proposed extent.")
put(gids(102, 107), "193722", "owner", "wrong", "verified", "The crop is a small fragment of the photographer/camera-side person, so identity is resolved but whole-person extent is wrong.")
put(["g0108"], None, "unknown", "wrong", "verified", "The box overlaps both right-side people and does not isolate one person owner.")
put(gids(109, 110), "193722", "owner", "wrong", "verified", "The crop is another small fragment of photographer owner 193722; whole-person extent is wrong.")
put(gids(111, 115), None, "unknown", "wrong", "verified", "The box overlaps both right-side people and cannot support one owner.")
put(gids(116, 117), "202810", "owner", "wrong", "verified", "The box identifies the rightmost standing man but covers only a small head/upper-body fragment.")
put(gids(118, 123), "1217358", "owner", "wrong", "verified", "The box repeats only a narrow fragment of lower-right person owner 1217358.")
put(["g0124"], "1217358", "owner", "reasonable", "verified", "This repeat box is close to the fixed visible extent of lower-right person owner 1217358.")
put(["g0125"], "1217358", "owner", "wrong", "verified", "The box repeats a narrow fragment of lower-right person owner 1217358.")
put(gids(126, 129), None, "invalid_output", "unknown", "unknown", "The raw parser reports zero-width or otherwise invalid geometry.")
put(gids(130, 135), None, "false", "wrong", "wrong", "The box lies on the ceiling/rack region with no person present.")
put(["g0136"], "pending-new:59571:chef-display-statue", "owner", "wrong", "wrong", "This repeats the chef display statue but misses its head and is not a reasonable whole-object extent.")
put(["g0137"], None, "unknown", "wrong", "wrong", "The box mixes the statue hat edge with background shelving and cannot support one owner.")
put(gids(138, 141), None, "false", "wrong", "wrong", "The box lies on rack or bookshelf structure with no person present.")
put(gids(142, 143), "pending-new:59571:chef-display-statue", "owner", "wrong", "wrong", "The box repeats only a small head/hat fragment of the chef display statue.")
put(["g0144"], "1722837", "owner", "reasonable", "verified", "The box reasonably repeats the fixed lower foreground person owner.")
put(["g0145"], "1722837", "owner", "wrong", "verified", "The tall box includes a fragment of owner 1722837 plus unrelated display area; extent is wrong.")
put(gids(146, 151), None, "false", "wrong", "wrong", "The box lies on product-display shelves with no person inside the proposed extent.")
put(gids(152, 166), None, "unknown", "wrong", "verified", "The large or horizontal box mixes display structure with multiple right-side people and does not isolate one owner.")
put(gids(167, 174), "202810", "owner", "wrong", "verified", "The box repeats only a narrow fragment of rightmost standing person owner 202810.")
put(gids(175, 176), None, "invalid_output", "unknown", "unknown", "The raw parser reports invalid geometry.")
put(["g0177"], "202810", "owner", "wrong", "verified", "The box repeats a tiny shirt fragment of owner 202810.")
put(gids(178, 181), None, "invalid_output", "unknown", "unknown", "The raw parser reports invalid geometry.")
put(["g0182"], "202810", "owner", "wrong", "verified", "The box repeats a tiny shirt fragment of owner 202810.")
put(gids(183, 186), None, "invalid_output", "unknown", "unknown", "The raw parser reports invalid geometry.")
put(["g0187"], "202810", "owner", "wrong", "verified", "The box repeats a tiny shirt fragment of owner 202810.")
put(["g0188"], None, "invalid_output", "unknown", "unknown", "The raw parser reports invalid geometry.")

assert set(rules) == set(groups), (sorted(set(groups) - set(rules)), sorted(set(rules) - set(groups)))

contact_sheets = []
for path in sorted((OWNED / "contact-sheets").glob("groups-*.png")):
    contact_sheets.append({"path": str(path), "sha256": sha256(path)})
assert len(contact_sheets) == 16

original = packet["rendered"]["original_image"]
target_overlay = packet["rendered"]["target_catalog_overlay"]
raw_overlay = packet["rendered"]["raw_generated_overlay"]
common_visual_evidence = [original["path"], target_overlay["path"], raw_overlay["path"]]

seen_owners = set()
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
    extent = rule["extent"]
    cls = rule["class"]
    fixed_owner = owner_id in target_ids
    coverage_eligible = (
        row["status"] == "parsed_valid"
        and fixed_owner
        and physical in {"true_unique", "repeat"}
        and extent == "reasonable"
    )
    direct_bbox = "mask"
    direct_description = "mask"
    if physical == "true_unique" and fixed_owner and extent == "reasonable":
        direct_bbox = "positive"
        if cls == "verified":
            direct_description = "positive"
    group = groups[gid]
    sheet_index = int(gid[1:]) // 12
    sheet_path = contact_sheets[sheet_index]["path"]
    if row["status"] == "parser_dropped":
        evidence_paths = [str(PACKET_PATH), group["context_crop"]["path"], group["tight_crop"]["path"], sheet_path]
    else:
        evidence_paths = common_visual_evidence + [group["context_crop"]["path"], group["tight_crop"]["path"], sheet_path]
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
        "qualified": True,
        "owner_id": "2094968",
        "natural_prediction_id": "p22",
        "physical_status": "true_unique",
        "extent": "reasonable",
        "class": "wrong",
        "coverage_eligible": True,
        "direct_CE": {"bbox": "positive", "description": "mask"},
        "finding": "The previously absent bottle owner enters natural greedy output before degeneration, but it is described as person."
    },
    "earliest_observed_issues": {
        "physical_repeat": {"prediction_id": "p2", "finding": "second box on the same non-atomic book stack"},
        "post_repair_false_or_structural": {"prediction_id": "p23", "finding": "person box on the ceiling beam immediately after repaired owner p22"},
        "exact_signature_repeat": {"prediction_id": "p27", "repeats_prediction_id": "p26"},
        "invalid_output": {"prediction_id": "p36", "drop_reason": "geometry_invalid"}
    },
    "before_cap_diagnosis": {
        "natural_eos": scored_row["natural_eos"],
        "capped_by_limit": scored_row["capped_by_limit"],
        "token_count": scored_row["token_count"],
        "raw_row_count": scored_row["raw_row_count"],
        "valid_prediction_count": scored_row["valid_prediction_count"],
        "raw_invalid_or_dropped_count": packet["counts"]["raw_invalid_or_dropped_count"],
        "finding": "After p22, person-labeled boxes sweep unrelated scene regions, repeat fragments of real people, produce invalid edge geometry, and continue until the length cap without natural EOS."
    }
}

new_owner_candidates = [
    {
        "owner_id": "pending-new:59571:upper-right-shelf-container",
        "reference_prediction_id": "p25",
        "reference_coord_bins_1000": [982, 361, 999, 388],
        "category": None,
        "status": "pending_root",
        "class_observation": "generated person is wrong",
        "reason": "The crop shows one small red-blue container-like shelf object at the image edge; exact category is unresolved.",
        "evidence_paths": [groups["g0025"]["context_crop"]["path"], groups["g0025"]["tight_crop"]["path"], contact_sheets[2]["path"]]
    },
    {
        "owner_id": "pending-new:59571:chef-display-statue",
        "reference_prediction_id": "p121",
        "reference_coord_bins_1000": [60, 508, 158, 938],
        "category": None,
        "status": "pending_root",
        "class_observation": "generated person is wrong; object is a decorative statue",
        "reason": "The crop reasonably isolates a distinct decorative chef display statue not present as an atomic fixed-v3 owner.",
        "evidence_paths": [groups["g0071"]["context_crop"]["path"], groups["g0071"]["tight_crop"]["path"], contact_sheets[5]["path"]]
    }
]

review = {
    "schema": "second_fit_step16_owner_review.v1",
    "image_id": 59571,
    "checkpoint_step": 16,
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
        "crop_hashes_bound_by_source_packet": True
    },
    "decisions": decisions,
    "summary": summary,
    "new_owner_candidates": new_owner_candidates,
    "validation": {
        "script_path": str(Path(__file__).resolve()),
        "result": "pending",
        "checks": []
    }
}


def validate(obj):
    checks = []
    assert obj["schema"] == "second_fit_step16_owner_review.v1"
    checks.append("schema")
    assert obj["raw_row_count"] == 342 == len(obj["decisions"]) == len(rows)
    checks.append("raw_row_count")
    by_id = {d["prediction_id"]: d for d in obj["decisions"]}
    assert len(by_id) == len(rows)
    assert set(by_id) == {r["prediction_id"] for r in rows}
    checks.append("all_raw_prediction_ids_exact_once")
    for row in rows:
        d = by_id[row["prediction_id"]]
        assert d["generated_order"] == row["generated_order"]
        assert d["visual_group_id"] == row["visual_group_id"]
    checks.append("raw_order_and_visual_group_membership")
    for gid, group in groups.items():
        packet_members = set(group["member_prediction_ids"])
        review_members = {d["prediction_id"] for d in obj["decisions"] if d["visual_group_id"] == gid}
        assert packet_members == review_members
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
        if d["physical_status"] == "repeat":
            assert d["direct_CE"] == {"bbox": "mask", "description": "mask"}
        if d["owner_id"] and d["owner_id"].startswith("pending-new:"):
            assert not d["coverage_eligible"]
            assert d["direct_CE"] == {"bbox": "mask", "description": "mask"}
        if d["physical_status"] in {"unknown", "invalid_output", "false"} or d["extent"] != "reasonable":
            assert d["direct_CE"] == {"bbox": "mask", "description": "mask"}
    checks.append("coverage_and_mask_invariants")
    assert set(obj["summary"]["covered_owner_ids"]).isdisjoint(obj["summary"]["missing_owner_ids"])
    assert set(obj["summary"]["covered_owner_ids"]) | set(obj["summary"]["missing_owner_ids"]) == set(target_ids)
    assert obj["summary"]["covered_owner_count"] + obj["summary"]["missing_owner_count"] == len(target_ids)
    checks.append("fixed_target_partition")
    assert obj["summary"]["covered_owner_ids"] == covered
    assert obj["summary"]["newly_covered_fixed_owner_ids"] == ["2094968"]
    assert obj["summary"]["lost_parent_owner_ids"] == []
    checks.append("exact_coverage_reproduction")
    assert obj["summary"]["invalid_output_row_count"] == 77
    assert [d["prediction_id"] for d in obj["decisions"] if d["prediction_id"] == "p22"][0] == "p22"
    p22 = by_id["p22"]
    assert p22["owner_id"] == "2094968" and p22["physical_status"] == "true_unique"
    assert p22["extent"] == "reasonable" and p22["class"] == "wrong" and p22["coverage_eligible"]
    assert p22["direct_CE"] == {"bbox": "positive", "description": "mask"}
    checks.append("repair_owner_p22")
    assert sha256(PACKET_PATH) == obj["source_packet"]["sha256"]
    assert sha256(CATALOG_PATH) == obj["target_catalog"]["sha256"]
    assert sha256(original["path"]) == original["sha256"]
    assert sha256(target_overlay["path"]) == target_overlay["sha256"]
    assert sha256(raw_overlay["path"]) == raw_overlay["sha256"]
    for group in groups.values():
        assert sha256(group["context_crop"]["path"]) == group["context_crop"]["sha256"]
        assert sha256(group["tight_crop"]["path"]) == group["tight_crop"]["sha256"]
    for sheet in contact_sheets:
        assert sha256(sheet["path"]) == sheet["sha256"]
    checks.append("source_and_visual_evidence_hashes")
    assert all(Path(p).exists() for d in obj["decisions"] for p in d["evidence_paths"])
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
    "covered": loaded["summary"]["covered_owner_count"],
    "missing": loaded["summary"]["missing_owner_count"],
    "repeats": loaded["summary"]["physical_repeat_row_count"],
    "false": loaded["summary"]["confirmed_false_row_count"],
    "unknown": loaded["summary"]["physical_unknown_row_count"],
    "invalid": loaded["summary"]["invalid_output_row_count"],
    "class_wrong": loaded["summary"]["class_wrong_row_count"],
    "class_unknown": loaded["summary"]["class_unknown_row_count"],
    "validation": loaded["validation"]["result"],
}, indent=2))
