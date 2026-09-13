"""Persist the scale owner's physical review of manifest cards 0--24."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from probes.dora_owner_learning.candidate_opportunity import require, score
from probes.native_owner_scale.scale import binding, publish, read, validate_selection
from probes.source_rweak_row_cross.run import native_record
from src.data.geometry import iou_xyxy


ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-12-native-owner-scale-and-state/scale"
)
RESULT = ROOT / "acquisition-full-v2.json"
MANIFEST = ROOT / "visual-review-full-v2" / "manifest.json"
OUTPUT = ROOT / "visual-review-full-v2" / "review-first25-scale-owner.json"


# Human observations from the immutable full+h+c/w cards. HOLD is neutral.
OBSERVATIONS = {
    4: {
        "c": ("clear", "substantial single utensil", "spoon supported"),
        "w": ("clear", "right-edge clipped; bowl visible, handle mostly off-frame", "spoon/scoop supported"),
        "nonduplicate": True,
        "verdict": "admit",
        "reason": "two spatially separate spoons; root independently confirmed the clipped w",
        "root_preconfirmed": True,
    },
    6: {
        "c": ("clear", "substantial person, slight image-edge clipping", "person supported"),
        "w": ("clear", "complete stemmed glass", "wine glass supported"),
        "nonduplicate": True,
        "verdict": "admit",
        "reason": "right-side person and central wine glass are separate from the two blue h owners",
    },
    8: {
        "c": ("clear object but identity conflicts with h", "broad right couch section", "couch supported"),
        "w": ("clear", "complete screen", "tv supported"),
        "nonduplicate": True,
        "verdict": "HOLD",
        "reason": "red c is physically part of the same sectional couch already covered by broad blue h",
        "c_absent_from_h": False,
    },
    10: {
        "c": ("clear", "broad tabletop", "dining table supported"),
        "w": ("visible only at top edge", "edge-clipped book spine", "book plausible"),
        "nonduplicate": False,
        "verdict": "HOLD",
        "reason": "green w is the same top-edge book as multiple blue h boxes (nearest IoU 0.913)",
    },
    16: {
        "c": ("clear", "partly occluded seated audience person", "person supported"),
        "w": ("clear", "head and torso of adjacent seated person", "person supported"),
        "nonduplicate": True,
        "verdict": "admit",
        "reason": "adjacent but visibly distinct audience members; neither is a blue h chair owner",
    },
    18: {
        "c": ("clear", "right-edge partial upholstered couch", "couch supported"),
        "w": ("clear region", "one box spans a visible stack of books", "book class but single-owner extent unsupported"),
        "nonduplicate": None,
        "verdict": "HOLD",
        "reason": "w extent contains multiple stacked books, so a genuine single owner is not established",
    },
    19: {
        "c": ("unresolved", "tiny train-window patch", "chair not visually resolved"),
        "w": ("unresolved", "adjacent tiny train-window patch", "chair not visually resolved"),
        "nonduplicate": None,
        "verdict": "HOLD",
        "reason": "neither small dark window patch supports a confident physical chair owner",
    },
    25: {
        "c": ("plausible with GT", "partly clipped black backpack/equipment at lower left", "backpack plausible"),
        "w": ("clear pixels but wrong class", "tiny distant lift-chair region", "snowboard unsupported"),
        "nonduplicate": None,
        "verdict": "HOLD",
        "reason": "w encloses a distant ski-lift chair rather than a snowboard",
    },
    28: {
        "c": ("small but GT-supported", "compact object on monitor stand", "remote supported"),
        "w": ("clear", "complete keyboard", "keyboard supported"),
        "nonduplicate": True,
        "verdict": "admit",
        "reason": "remote and keyboard are separate single objects absent from blue h book/chair boxes",
    },
    30: {
        "c": ("clear", "stovetop/oven appliance surface", "oven supported"),
        "w": ("clear", "substantial standing woman", "person supported"),
        "nonduplicate": True,
        "verdict": "admit",
        "reason": "one oven/range and one distinct person; no h owner occupies either region",
    },
    31: {
        "c": ("clear", "complete blue cup on left pole", "cup supported"),
        "w": ("clear", "complete blue cup on right pole", "cup supported"),
        "nonduplicate": True,
        "verdict": "admit",
        "reason": "two visibly separate blue cups; w has a distinct GT owner not present in h",
    },
    33: {
        "c": ("clear", "left tabletop", "dining table supported"),
        "w": ("clear", "partial right covered table", "dining table supported"),
        "nonduplicate": False,
        "verdict": "HOLD",
        "reason": "w matches GT owner 1965639 already in h and is a partial jittered box on that same table",
    },
    34: {
        "c": ("small but GT-supported", "single narrow book spine", "book supported"),
        "w": ("clear", "complete right automotive chair", "chair supported"),
        "nonduplicate": True,
        "verdict": "admit",
        "reason": "book spine and right chair are distinct from h; w is not the blue left chair",
    },
    35: {
        "c": ("clear", "left/central plant and pot", "potted plant supported"),
        "w": ("clear", "right plant with separate pot", "potted plant supported"),
        "nonduplicate": True,
        "verdict": "admit",
        "reason": "adjacent foliage belongs to two visibly separate pots; neither is a blue h vase box",
    },
    36: {
        "c": ("clear", "complete left-tray donut", "donut supported"),
        "w": ("clear", "separate donut above c", "donut supported"),
        "nonduplicate": True,
        "verdict": "admit",
        "reason": "root-confirmed unlabeled c and separate w; both differ from blue h-covered left-edge donuts",
        "root_preconfirmed": True,
    },
    43: {
        "c": ("weak", "tiny narrow pedestrian candidate under signal", "person uncertain"),
        "w": ("unresolved", "tiny right-edge road patch", "car uncertain"),
        "nonduplicate": None,
        "verdict": "HOLD",
        "reason": "w physical support, class, and extent are not confidently resolved",
    },
    50: {
        "c": ("unsupported", "box lies on accordion/player torso", "cow unsupported"),
        "w": ("partial", "right-edge cow fragment", "cow plausible"),
        "nonduplicate": None,
        "verdict": "HOLD",
        "reason": "c box does not visibly support a cow owner",
    },
    54: {
        "c": ("clear", "single background flower pot", "potted plant supported"),
        "w": ("clear", "partial red background chair", "chair supported"),
        "nonduplicate": False,
        "verdict": "HOLD",
        "reason": "w is a shifted extension of the same red chair already boxed in blue h",
    },
    60: {
        "c": ("clear", "single copper bowl on shelf", "bowl supported"),
        "w": ("clear", "dark vase under sunflowers", "vase supported"),
        "nonduplicate": False,
        "verdict": "HOLD",
        "reason": "w matches GT owner 1152176 already in h and visually repeats its blue box",
    },
    65: {
        "c": ("plausible with GT", "partly occluded pedestrian", "person supported"),
        "w": ("unresolved", "thin distant storefront/road patch", "person unsupported"),
        "nonduplicate": None,
        "verdict": "HOLD",
        "reason": "w does not resolve as one physical person",
    },
    71: {
        "c": ("clear", "single large tan bowl", "bowl supported"),
        "w": ("clear", "right-edge hanging ladle/spoon", "spoon supported"),
        "nonduplicate": True,
        "verdict": "admit",
        "reason": "the hanging right utensil is distinct from the other blue h spoon locations",
    },
    72: {
        "c": ("weak with GT", "chair/stool largely occluded by dog", "chair plausible"),
        "w": ("unresolved", "tiny white doorway sliver", "chair unsupported"),
        "nonduplicate": None,
        "verdict": "HOLD",
        "reason": "w has no confident physical chair support or extent",
    },
    75: {
        "c": ("small but GT-supported", "single crowd person", "person supported"),
        "w": ("weak", "very low-resolution crowd sliver", "person plausible but boundary uncertain"),
        "nonduplicate": None,
        "verdict": "HOLD",
        "reason": "w single-owner extent cannot be separated confidently from adjacent crowd people",
    },
    79: {
        "c": ("tiny but GT-supported", "one tennis ball", "sports ball supported"),
        "w": ("visible", "tennis-ball cluster region", "sports ball supported"),
        "nonduplicate": False,
        "verdict": "HOLD",
        "reason": "w substantially overlaps and visually repeats the same ball already boxed by blue h",
    },
    80: {
        "c": ("clear despite occlusion", "visible front of bus behind crowd", "bus supported"),
        "w": ("clear", "single right-edge standing person", "person supported"),
        "nonduplicate": True,
        "verdict": "admit",
        "reason": "occluded bus front and rightmost person are genuine distinct owners absent from h",
    },
}


def main() -> None:
    result, manifest = read(RESULT), read(MANIFEST)
    selection = validate_selection(result["selection"]["path"])
    source = read(selection["sources"]["stable50_universe"]["path"])
    by_job = {row["job_id"]: row for row in selection["nominations"]}
    by_example = {row["example_id"]: row for row in source["eval_records"]}
    by_result = {row["job_id"]: row for row in result["rows"]}
    cards = manifest["cards"][:25]
    require(len(cards) == 25 and [card["admission_priority"] for card in cards] == list(OBSERVATIONS),
            "frozen first25 review partition")
    decisions = []
    for card in cards:
        job = by_job[card["job_id"]]
        row = by_result[card["job_id"]]
        frozen = by_example[job["example_id"]]
        observation = OBSERVATIONS[job["admission_priority"]]
        h_parsed = native_record(job["h_text"], frozen["case"], frozen["golden"], "supplied_prefix")
        w_parsed = native_record(row["local_w"]["w_text"], frozen["case"], frozen["golden"], "free")
        w_score = score(w_parsed, seed=None, length=len(row["local_w"]["w_token_ids"]), stop="free")
        w_box = w_parsed["pred"][0]["bbox"]
        nearest = max(
            ({"iou": iou_xyxy(w_box, item["bbox"]), "index": index,
              "description": item["description"], "bbox_xyxy_pixels": item["bbox"]}
             for index, item in enumerate(h_parsed["pred"])),
            key=lambda item: item["iou"],
            default=None,
        )
        w_owners = w_score["50"]["owners"]
        decision = {
            "job_id": job["job_id"],
            "admission_priority": job["admission_priority"],
            "card": card["card"],
            "c": {
                "provenance": job["source"],
                "owner_id": job["owner_id"],
                "bbox_xyxy_pixels": job["c_bbox_xyxy_pixels"],
                "support": observation["c"][0],
                "extent": observation["c"][1],
                "class": observation["c"][2],
                "max_any_class_history_iou": job["max_any_class_history_iou"],
                "absent_from_h": observation.get("c_absent_from_h", True),
            },
            "w": {
                "bbox_xyxy_pixels": w_box,
                "support": observation["w"][0],
                "extent": observation["w"][1],
                "class": observation["w"][2],
                "gt_owner_ids_iou50": w_owners,
                "gt_owner_ids_already_in_h": [owner for owner in w_owners if owner in job["h_owner_ids50"]],
                "closest_history_box": nearest,
                "physical_nonduplicate_vs_h_plus_c": observation["nonduplicate"],
            },
            "verdict": observation["verdict"],
            "reason": observation["reason"],
            "unknown_is_neutral": observation["verdict"] == "HOLD",
            "root_preconfirmed": observation.get("root_preconfirmed", False),
        }
        require(
            decision["verdict"] == "HOLD"
            or (decision["c"]["absent_from_h"] is True
                and decision["w"]["physical_nonduplicate_vs_h_plus_c"] is True),
            "admit physical invariants",
        )
        decisions.append(decision)
    counts = Counter(decision["verdict"] for decision in decisions)
    publish(OUTPUT, {
        "schema": "native_owner_scale.visual_review_sidecar.v1",
        "status": "candidate_worker_review_root_acceptance_pending",
        "reviewer": "/root/sol_high_native_owner_scale",
        "partition": {"manifest_start_inclusive": 0, "manifest_end_exclusive": 25},
        "result": binding(RESULT),
        "manifest": binding(MANIFEST),
        "policy": (
            "image-level physical single-owner support, extent and class; c absent from literal h; "
            "w nonduplicate of h+c; GT identity and closest h box aid but do not replace sight; unknown is HOLD"
        ),
        "counts": dict(counts),
        "decisions": decisions,
    })
    print(json.dumps({"output": binding(OUTPUT), "counts": dict(counts)}, indent=2))


if __name__ == "__main__":
    main()
