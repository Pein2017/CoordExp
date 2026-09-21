#!/usr/bin/env python3
"""Materialize and validate the image417044 step16 physical-owner review."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

B = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum")
OUT = B / "second-fit-owner-reviews-v1/image-000000417044"
PACKET_PATH = B / "second-fit-review-packets-v1/image-000000417044/packet.json"
TARGET_PATH = B / "target-owners-complete-v3.json"
PARENT_PATH = B / "parent16-v3-physical-ledger-v2/ledger.json"
PARENT_ACCEPTANCE = B / "parent16-v3-physical-ledger-v2/root-acceptance.json"
STEP64_PACKET = B / "second-fit-step64-introduced-owner-packets-v1/image-000000417044/packet.json"
PROTOCOL = B / "second-fit-owner-reviews-v1/review-protocol.md"

OWNER_BY_ORDER = {
    0: "515293", 1: "417044:review:stage01:U01", 2: "417044:review:stage01:U02",
    3: "417044:review:P6", 4: "417044:review:stage01:U03", 5: "417044:review:stage01:U04",
    6: "417044:review:stage01:U05", 7: "1083135", 8: "417044:review:stage01:U06",
    9: "1083564", 10: "417044:review:stage01:U07", 11: "417044:review:stage01:U09",
    12: "417044:review:stage01:U10", 13: "first-fit:new:417044:upper-center-donut",
    14: "417044:review:stage01:U11", 15: "1083042", 16: "417044:review:stage01:U12",
    17: "417044:review:stage01:U13", 18: "1082918", 19: "1083295", 20: "1083599",
    21: "1079494", 22: "1079910", 23: "1080038", 24: "1082111", 25: "1572342",
}
FALSE_ORDERS = {26, 27, 28, 29, 30, 31, 34, 35}
INVALID_ORDERS = {32, 33}


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def binding(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "sha256": file_hash(path), "size_bytes": path.stat().st_size}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def evidence_paths(row: dict[str, Any], packet: dict[str, Any]) -> list[str]:
    return [
        packet["rendered"]["original_image"]["path"],
        packet["rendered"]["target_catalog_overlay"]["path"],
        packet["rendered"]["raw_generated_overlay"]["path"],
        row["shared_crop_paths"]["tight"], row["shared_crop_paths"]["context"],
    ]


def positive_reason(order: int, owner_id: str) -> str:
    if order == 0:
        return "Original, overlays, and both crops show the single person at fixed owner 515293 with a reasonable visible-body extent."
    if order == 17:
        return "Original, overlays, and both crops show one distinct chocolate-frosted donut; the box isolates fixed owner 417044:review:stage01:U13 with reasonable extent."
    return f"Original, overlays, and both crops show one distinct donut/baked item isolated at fixed owner {owner_id} with reasonable extent."


def false_reason(order: int) -> str:
    if order == 26:
        return "The viewed box encloses a cluster of many powdered donut holes, not one atomic object; extent is wrong."
    if order in {27, 28, 29, 30, 31}:
        return "The viewed box spans multiple donut holes and/or partial neighbors, so it does not isolate one atomic object; extent is wrong."
    if order == 34:
        return "The viewed right-edge box is a fragment across the multi-hole tray and does not isolate one atomic object; extent is wrong."
    return "The viewed narrow clipped right-edge strip contains parts of multiple donut holes and does not isolate one atomic object; extent is wrong."


def build() -> tuple[dict[str, Any], dict[str, Any]]:
    packet = json.loads(PACKET_PATH.read_text()); target = json.loads(TARGET_PATH.read_text())
    parent = json.loads(PARENT_PATH.read_text()); step64 = json.loads(STEP64_PACKET.read_text())
    rows = packet["rendered"]["raw_rows"]
    target_ids = sorted(str(row["owner_id"]) for row in packet["target_catalog_references"])
    parent_image = next(row for row in parent["images"] if int(row["image_id"]) == 417044)
    decisions = []
    for row in rows:
        order = int(row["generated_order"])
        base = {
            "prediction_id": row["prediction_id"], "generated_order": order,
            "visual_group_id": row["visual_group_id"], "raw_status": row["status"],
            "raw_span_sha256": row["raw_span_sha256"], "owner_id": None,
            "evidence_paths": evidence_paths(row, packet),
        }
        if order in OWNER_BY_ORDER:
            base.update({
                "owner_id": OWNER_BY_ORDER[order], "physical_status": "true_unique",
                "extent": "reasonable", "class": "verified", "coverage_eligible": True,
                "direct_CE": {"bbox": "positive", "description": "positive"},
                "reason": positive_reason(order, OWNER_BY_ORDER[order]),
            })
        elif order in FALSE_ORDERS:
            base.update({
                "physical_status": "false", "extent": "wrong", "class": "verified",
                "coverage_eligible": False, "direct_CE": {"bbox": "mask", "description": "mask"},
                "reason": false_reason(order),
            })
        else:
            require(order in INVALID_ORDERS, f"unruled order {order}")
            base.update({
                "physical_status": "invalid_output", "extent": "wrong", "class": "unknown",
                "coverage_eligible": False, "direct_CE": {"bbox": "mask", "description": "mask"},
                "reason": "Native parser dropped the malformed object span with degenerate x1=x2=999; generated axes are preserved and receive no owner credit.",
                "drop_reason": row["drop_reason"],
            })
        decisions.append(base)
    covered = sorted({str(row["owner_id"]) for row in decisions if row["coverage_eligible"]})
    missing = sorted(set(target_ids) - set(covered))
    parent_covered = sorted(parent_image["covered_owner_ids"])
    review = {
        "schema": "second_fit_step16_owner_review.v1", "status": "candidate_ready",
        "image_id": 417044, "checkpoint_step": 16,
        "target_catalog": binding(TARGET_PATH), "source_packet": binding(PACKET_PATH),
        "review_protocol": binding(PROTOCOL), "raw_row_count": len(rows), "decisions": decisions,
        "summary": {
            "target_owner_count": len(target_ids), "covered_owner_ids": covered, "missing_owner_ids": missing,
            "covered_owner_count": len(covered), "missing_owner_count": len(missing),
            "physical_repeat_row_count": 0, "confirmed_false_row_count": len(FALSE_ORDERS),
            "physical_unknown_row_count": 0, "invalid_output_row_count": len(INVALID_ORDERS),
            "class_wrong_row_count": 0, "class_unknown_row_count": len(INVALID_ORDERS),
            "raw_stop_reason": "im_end", "cap_debt": 0,
            "parent_covered_owner_ids": parent_covered,
            "retained_parent_owner_ids": sorted(set(parent_covered) & set(covered)),
            "lost_parent_owner_ids": sorted(set(parent_covered) - set(covered)),
            "newly_covered_fixed_owner_ids": sorted(set(covered) - set(parent_covered)),
            "stage03_appended_gt_reference_repair_qualified_at_step16": False,
            "stage03_appended_owner_id": "1083260",
            "earliest_observed_repeat_or_structural_issue": {
                "generated_order": 26, "prediction_id": "p26", "kind": "multi_object_cluster_box",
                "note": "First row after 26 geometry-qualified unique fixed owners; it boxes many powdered donut holes. No physical repeat is observed before natural EOS.",
            },
        },
        "new_owner_candidates": [],
        "scientific_assessment": "At native greedy step16, the selected stage03 owner 1083260 is still missing. The stream naturally terminates after 26 qualified fixed owners, then eight wrong-extent multi-object/fragment rows and two malformed parser drops. This is image-local evidence only.",
        "parent_context": {"ledger": binding(PARENT_PATH), "root_acceptance": binding(PARENT_ACCEPTANCE)},
        "validation": {
            "all_raw_prediction_ids_preserved": sorted(row["prediction_id"] for row in decisions) == sorted(row["prediction_id"] for row in rows),
            "exact_visual_group_membership": all(row["visual_group_id"] == packet["rendered"]["raw_rows"][i]["visual_group_id"] for i, row in enumerate(decisions)),
            "target_partition_exact": sorted(covered + missing) == target_ids and not (set(covered) & set(missing)),
            "all_evidence_viewed": True,
            "same_image_only": True,
            "claim_boundary": "Candidate single-image physical review; no all-stage pass claim or target mutation.",
        },
    }
    candidate = step64["candidate_groups"][0]
    supplemental = {
        "schema": "second_fit_step64_selected_owner_supplement.v1", "status": "candidate_ready",
        "image_id": 417044, "checkpoint_step": 64, "selected_owner_id": "1083260",
        "source_packet": binding(STEP64_PACKET),
        "decision": {
            "prediction_id": candidate["members"][0]["prediction_id"],
            "generated_order": candidate["members"][0]["generated_order"],
            "owner_id": "1083260", "physical_status": "true_unique", "extent": "reasonable", "class": "verified",
            "coord_bins_1000": candidate["coord_bins_1000"], "selected_reference_iou": candidate["selected_reference_iou"],
            "reason": "The original image, selected overlay, and tight/context crops show the same upper-left glazed donut as fixed GT owner1083260; the candidate closely follows the visible donut extent.",
            "evidence": {
                "original": step64["visual_evidence"]["original_image"],
                "overlay": step64["visual_evidence"]["selected_gt_and_candidates_overlay"],
                **candidate["shared_visual_evidence"],
            },
        },
        "output_context": {
            "capped": True, "stop_reason": "length", "token_count": 3084,
            "first_observable_anomaly_generated_order": 39,
            "sustained_repeat_or_collapse_generated_order": 152,
            "selected_owner_before_first_anomaly": candidate["members"][0]["generated_order"] < 39,
            "selected_owner_before_sustained_collapse": candidate["members"][0]["generated_order"] < 152,
        },
        "assessment": "The intended fixed owner is physically and geometrically qualified at step64 before the first anomaly. The output later caps and structurally loops, so this is not a clean-output claim and no other step64 row was assessed.",
        "claim_boundary": "Targeted selected-owner review only; IoU alone was not used as owner truth and no other output row was judged.",
    }
    return review, supplemental


def validate(review: dict[str, Any], supplemental: dict[str, Any]) -> dict[str, Any]:
    packet = json.loads(PACKET_PATH.read_text()); rows = packet["rendered"]["raw_rows"]
    require(review["status"] == "candidate_ready" and review["raw_row_count"] == 36, "review identity/count")
    require([d["prediction_id"] for d in review["decisions"]] == [r["prediction_id"] for r in rows], "raw ID/order mismatch")
    require([d["visual_group_id"] for d in review["decisions"]] == [r["visual_group_id"] for r in rows], "group mismatch")
    require(len({d["prediction_id"] for d in review["decisions"]}) == 36, "duplicate decisions")
    domains = {"physical_status": {"true_unique", "repeat", "false", "unknown", "invalid_output"}, "extent": {"reasonable", "wrong", "unknown"}, "class": {"verified", "wrong", "unknown"}}
    for d in review["decisions"]:
        for key, values in domains.items(): require(d[key] in values, f"domain {key}")
        require(d["direct_CE"]["bbox"] in {"positive", "mask"} and d["direct_CE"]["description"] in {"positive", "mask"}, "CE domain")
        expected = d["physical_status"] == "true_unique" and d["extent"] == "reasonable" and d["owner_id"] is not None
        require(d["coverage_eligible"] == expected, f"coverage contract {d['prediction_id']}")
        if not expected: require(d["direct_CE"] == {"bbox": "mask", "description": "mask"}, f"mask contract {d['prediction_id']}")
        for path in d["evidence_paths"]: require(Path(path).is_file(), f"missing evidence {path}")
    summary = review["summary"]
    require(summary["target_owner_count"] == 33 and summary["covered_owner_count"] == 26 and summary["missing_owner_count"] == 7, "coverage counts")
    require(set(summary["covered_owner_ids"]).isdisjoint(summary["missing_owner_ids"]), "partition overlap")
    require(len(summary["covered_owner_ids"] + summary["missing_owner_ids"]) == 33, "partition size")
    require(summary["retained_parent_owner_ids"] == summary["parent_covered_owner_ids"], "parent owner lost")
    require(summary["newly_covered_fixed_owner_ids"] == ["417044:review:stage01:U13"], "new coverage")
    require(summary["confirmed_false_row_count"] == 8 and summary["invalid_output_row_count"] == 2 and summary["physical_repeat_row_count"] == 0, "debt counts")
    require(supplemental["decision"]["owner_id"] == "1083260" and supplemental["decision"]["physical_status"] == "true_unique", "step64 decision")
    require(supplemental["output_context"]["selected_owner_before_first_anomaly"] and supplemental["output_context"]["capped"], "step64 timing/cap boundary")
    for value in supplemental["decision"]["evidence"].values(): require(file_hash(Path(value["path"])) == value["sha256"], "step64 evidence hash")
    return {"status": "validated", "step16_raw_rows": 36, "step16_covered_fixed_owners": 26, "step16_missing_fixed_owners": 7, "step16_false_rows": 8, "step16_invalid_rows": 2, "step64_selected_owner_qualified": True}


def main() -> None:
    review, supplemental = build()
    result = validate(review, supplemental)
    review["validation"]["validator_result"] = result
    (OUT / "review.json").write_bytes(canonical(review))
    (OUT / "supplemental-step64.json").write_bytes(canonical(supplemental))
    # Replay from the just-written artifacts as the final consumer check.
    result2 = validate(json.loads((OUT / "review.json").read_text()), json.loads((OUT / "supplemental-step64.json").read_text()))
    print(json.dumps({**result2, "review": binding(OUT / "review.json"), "supplemental": binding(OUT / "supplemental-step64.json")}, indent=2))


if __name__ == "__main__":
    main()
