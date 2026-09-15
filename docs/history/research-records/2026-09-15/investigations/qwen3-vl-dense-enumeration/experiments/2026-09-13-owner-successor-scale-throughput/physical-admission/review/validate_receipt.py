#!/usr/bin/env python3
"""Read-only fail-closed checks for the physical-admission receipt."""

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

WORK = Path("/data/CoordExp/.worktrees/research-probes")
OUT = Path(
    "/data/CoordExp/outputs/research/"
    "qwen3-vl-dense-enumeration/2026-09-13-owner-successor-scale-throughput/"
    "physical-admission"
)
U = (
    WORK
    / "research/investigations/qwen3-vl-dense-enumeration/experiments/"
    "2026-09-13-owner-successor-scale-throughput/physical-admission"
)


def load(path):
    with path.open() as f:
        return json.load(f)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(message):
    raise AssertionError(message)


def main():
    preflight = load(U / "preflight.json")
    index = load(OUT / "review-index-v1.json")
    review = load(OUT / "physical-review-v1.json")
    receipt = load(OUT / "individual-view-receipt-v1.json")
    manifest = load(OUT / "manifest.json")
    census = load(OUT / "census-v1.json")

    # Frozen sealed inputs, including the completed-slice omission.
    for shard, expected in (("shard-2", 19), ("shard-3", 39)):
        spec = preflight["shards"][shard]
        terminal = load(Path(spec["bindings"]["terminal"]["path"]))
        if terminal.get("status") != "completed" or terminal.get("exit_code") != 0:
            fail(f"{shard}: terminal is not completed/exit0")
        if spec["terminal"]["status"] != "completed" or spec["terminal"]["exit_code"] != 0:
            fail(f"{shard}: preflight terminal is not completed/exit0")
        if spec["candidate_local_w_count"] != expected:
            fail(f"{shard}: candidate_local_w count changed")
    if preflight["shards"]["shard-3"]["excluded_job_ids_not_in_completed_rows"] != [
        "39654:h0:c0"
    ]:
        fail("shard-3 completed-slice exclusion changed")

    if len(index["rows"]) != 58 or len({r["visual_group_id"] for r in index["rows"]}) != 50:
        fail("candidate row/group denominator changed")
    if len(review["rows"]) != 58 or len({r["visual_group_id"] for r in review["rows"]}) != 50:
        fail("review row/group denominator changed")

    # Exact candidate row -> job -> image joins, and no absent shard-3 job.
    supply = Path(
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
        "2026-09-13-owner-successor-scale-throughput/supply/remainder-v1"
    )
    source_rows, source_jobs, source_images = {}, {}, {}
    for shard in ("shard-2", "shard-3"):
        base = supply / shard
        for ordinal, line in enumerate((base / "rows.jsonl").read_text().splitlines()):
            obj = json.loads(line)
            source_rows[obj["job_id"]] = (shard, ordinal, obj)
        for ordinal, line in enumerate((base / "jobs.jsonl").read_text().splitlines()):
            obj = json.loads(line)
            source_jobs[obj["job_id"]] = (shard, ordinal, obj)
        for line in (base / "images.jsonl").read_text().splitlines():
            obj = json.loads(line)
            source_images[obj["image_id"]] = (shard, obj)
    absent = "39654:h0:c0"
    if absent in {r["job_id"] for r in index["rows"]}:
        fail("absent shard-3 job entered candidate index")
    for row in index["rows"]:
        jid, iid, shard = row["job_id"], row["image_id"], row["shard"]
        if jid not in source_rows or jid not in source_jobs:
            fail(f"missing source join for {jid}")
        if iid not in source_images:
            fail(f"missing image join for {jid}/{iid}")
        rshard, rord, robj = source_rows[jid]
        jshard, jord, jobj = source_jobs[jid]
        ishard, iobj = source_images[iid]
        if (rshard, jshard, ishard) != (shard, shard, shard):
            fail(f"shard mismatch for {jid}")
        if rord != row["source_row_ordinal"] or jord != row["source_job_ordinal"]:
            fail(f"source ordinal mismatch for {jid}")
        if robj["image_id"] != iid or jobj["image_id"] != iid:
            fail(f"job/image id mismatch for {jid}")
        if iobj["image_id"] != iid:
            fail(f"image record mismatch for {jid}")

    # Resolved positive classes must be canonical COCO-80.  HOLD/outside-class
    # evidence remains neutral and is not mapped into another class.
    sys.path.insert(0, str(WORK))
    from src.eval.detection_categories import COCO_80_CATEGORY_IDS

    for row in review["rows"]:
        for side in ("c", "w"):
            proposal = row[side]
            status = proposal["physical_review"]["admission_status"]
            if status == "candidate_accept":
                name = proposal["description"]
                if name not in COCO_80_CATEGORY_IDS:
                    fail(f"non-COCO positive resolved: {row['job_id']} {side} {name}")
                if not proposal["canonical_coco80"]["member"]:
                    fail(f"canonical membership false for resolved positive: {row['job_id']} {side}")

    # All one-sample card/crop bindings are immutable and decision evidence does
    # not point at the historical contact sheets.
    if receipt["visual_group_count"] != 50 or receipt["individual_card_count"] != 50:
        fail("individual card receipt denominator changed")
    if receipt["crop_view_count"] != 48:
        fail("crop receipt count changed")
    views = {v["visual_group_id"]: v for v in receipt["views"]}
    if set(views) != {r["visual_group_id"] for r in index["rows"]}:
        fail("individual-view groups do not match index groups")
    for v in views.values():
        card = Path(v["card_binding"]["path"])
        if not card.exists() or sha256(card) != v["card_binding"]["sha256"]:
            fail(f"card binding mismatch: {card}")
        if v["card_view"]["contact_sheet_evidence"]:
            fail("contact sheet used as card decision evidence")
        for crop in v["crop_views"]:
            path = Path(crop["path"])
            if not path.exists() or sha256(path) != crop["sha256"]:
                fail(f"crop binding mismatch: {path}")
    for row in review["rows"]:
        ev = row["review_evidence"]
        if ev["contact_sheet_decision_evidence"]:
            fail(f"contact sheet evidence on {row['job_id']}")
        if ev["individual_card_binding"]["sha256"] != views[row["visual_group_id"]]["card_binding"]["sha256"]:
            fail(f"row/card binding mismatch: {row['job_id']}")

    # Fail closed on worker-only labels, training targets, and duplicate aliases.
    if any(r["training_target"] for r in review["rows"]):
        fail("worker proposal became training target")
    if any(g["training_target"] for g in review["groups"]):
        fail("visual group became training target")
    if any(g["review_status"] == "lead-accepted" for g in review["groups"]):
        fail("lead acceptance written by worker")
    for cluster in review["visual_alias_clusters"]:
        group_ids = set(cluster["group_ids"])
        cluster_rows = [r for r in review["rows"] if r["visual_group_id"] in group_ids]
        if any(r["training_target"] for r in cluster_rows):
            fail(f"alias cluster has atomic training target: {cluster['visual_alias_cluster_id']}")
        if any(r["visual_review_alias_cluster_id"] != cluster["visual_alias_cluster_id"] for r in cluster_rows):
            fail(f"alias provenance missing: {cluster['visual_alias_cluster_id']}")

    # Root owns admission: these artifacts can only be candidates/HOLD.
    if review["counts"]["candidate_accept_jobs"] != 9 or review["counts"]["hold_jobs"] != 49:
        fail("candidate/HOLD job counts changed")
    if review["counts"]["canonical_candidate_pairs_after_alias_collapse"] != 6:
        fail("alias-collapsed canonical count changed")
    if review["rows"][[r["visual_group_id"] for r in review["rows"]].index("PA-0003")]["w"]["physical_review"]["admission_status"] != "candidate_accept":
        fail("PA-0003 w individual-review correction missing")

    checks = {
        "sealed_terminals_completed_exit0": True,
        "candidate_denominator_58": True,
        "visual_group_denominator_50": True,
        "exact_rows_jobs_images_join": True,
        "shard3_absent_job_excluded": True,
        "all_resolved_classes_in_coco80": True,
        "individual_cards_50": True,
        "individual_crops_48": True,
        "contact_sheets_historical_only": True,
        "alias_execution_history_preserved": True,
        "aliases_not_training_targets": True,
        "no_training_or_lead_acceptance": True,
        "status_domain": True,
    }
    out = {
        "schema": "owner_successor_scale.physical_admission_validation.v2",
        "status": "passed",
        "checks": checks,
        "observed_counts": review["counts"],
        "side_status_counts": review["side_status_counts"],
        "review_sha256": sha256(OUT / "physical-review-v1.json"),
        "receipt_sha256": sha256(OUT / "individual-view-receipt-v1.json"),
        "index_sha256": sha256(OUT / "review-index-v1.json"),
        "manifest_sha256": sha256(OUT / "manifest.json"),
        "command": "PYTHONPATH=/data/CoordExp/.worktrees/research-probes python3 "
        + str(U / "review/validate_receipt.py"),
        "note": "No worker labels, training export, GPU call, source edit, or confirmation-panel admission.",
    }
    (U / "review/validation-v1.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    (OUT / "validation-v1.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
