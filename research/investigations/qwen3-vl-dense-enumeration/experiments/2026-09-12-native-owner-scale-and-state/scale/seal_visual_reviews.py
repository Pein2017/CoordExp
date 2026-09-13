"""Seal the disjoint 25+25 worker reviews with root's physical overlays."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from probes.dora_owner_learning.candidate_opportunity import require
from probes.native_owner_scale.scale import binding, distinct_first_admission, publish, read


ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-12-native-owner-scale-and-state/scale"
)
RESULT = ROOT / "acquisition-full-v2.json"
REVIEW_ROOT = ROOT / "visual-review-full-v2"
MANIFEST = REVIEW_ROOT / "manifest.json"
FIRST = REVIEW_ROOT / "review-first25-scale-owner.json"
LAST = REVIEW_ROOT / "luna-last25-review.json"
ROOT_FIRST = REVIEW_ROOT / "root-review.json"
ROOT_LAST = REVIEW_ROOT / "root-last25-rulings.json"
OUTPUT = REVIEW_ROOT / "final-reviews.json"
EXPECTED_ADMIT_PRIORITIES = (4, 6, 16, 28, 30, 31, 34, 35, 36, 71,
                             90, 92, 110, 114, 116, 150)


def _normalize(value: str) -> str:
    require(value.lower() in ("admit", "hold"), "known physical review verdict")
    return "accept" if value.lower() == "admit" else "neutral"


def main() -> None:
    result, manifest = read(RESULT), read(MANIFEST)
    first, last = read(FIRST), read(LAST)
    cards = manifest["cards"]
    require(len(cards) == 50 and len(result["rows"]) == 156, "review/result denominator")
    candidate_ids = [card["job_id"] for card in cards]
    first_rows, last_rows = first["decisions"], last["reviews"]
    require([row["job_id"] for row in first_rows] == candidate_ids[:25], "first25 identity/order")
    require([row["assigned_id"] for row in last_rows] == candidate_ids[25:], "last25 identity/order")

    worker = {}
    for row in first_rows:
        worker[row["job_id"]] = {
            "verdict": _normalize(row["verdict"]), "reason": row["reason"],
            "sidecar": str(FIRST), "card": row["card"]["path"],
        }
    for row in last_rows:
        worker[row["assigned_id"]] = {
            "verdict": _normalize(row["verdict"]), "reason": row["reason"],
            "sidecar": str(LAST), "card": row["card_path"],
        }
    require(set(worker) == set(candidate_ids) and len(worker) == 50, "exact worker review union")

    overlays = {}
    for path in (ROOT_FIRST, ROOT_LAST):
        root_rows = read(path)["decisions"]
        for row in root_rows:
            require(row["job_id"] in worker and row["admission_priority"]
                    == next(card["admission_priority"] for card in cards
                            if card["job_id"] == row["job_id"]), "root overlay identity")
            overlays[row["job_id"]] = {
                "verdict": _normalize(row["verdict"]), "reason": row["reason"],
                "sidecar": str(path),
            }

    by_result = {row["job_id"]: row for row in result["rows"]}
    decisions = {}
    for row in result["rows"]:
        job_id = row["job_id"]
        if job_id not in worker:
            require(row["local_w"]["status"] != "candidate_local_w", "candidate lacks review")
            decisions[job_id] = {
                "status": "neutral", "reason": row["local_w"]["status"],
                "evidence_paths": [str(RESULT)], "mechanical_noncandidate": True,
            }
            continue
        effective = overlays.get(job_id, worker[job_id])
        evidence = [worker[job_id]["card"], worker[job_id]["sidecar"]]
        if job_id in overlays:
            evidence.append(overlays[job_id]["sidecar"])
        accepted = effective["verdict"] == "accept"
        decisions[job_id] = {
            "status": effective["verdict"],
            "reason": effective["reason"],
            "worker_verdict": worker[job_id]["verdict"],
            "root_overlay_applied": job_id in overlays,
            "c_single_owner_absent_from_h": True if accepted else None,
            "w_single_owner_nonduplicate": True if accepted else None,
            "evidence_paths": evidence,
            "mechanical_noncandidate": False,
        }

    accepted_rows = [by_result[job_id] for job_id, value in decisions.items()
                     if value["status"] == "accept"]
    require(tuple(sorted(row["admission_priority"] for row in accepted_rows))
            == EXPECTED_ADMIT_PRIORITIES, "root final physical admit set")
    admitted = distinct_first_admission(accepted_rows, maximum=32)
    require(len(admitted) == 16 and len({row["example_id"] for row in admitted}) == 11,
            "N16 across 11 distinct images")
    counts = Counter(value["status"] for value in decisions.values())
    require(counts == {"accept": 16, "neutral": 140}, "final 156 review denominator")

    publish(OUTPUT, {
        "schema": "native_owner_scale.visual_reviews.final.v1",
        "status": "lead_accepted_physical_bank_admission",
        "lead_acceptance": {
            "owner": "/root", "scope": "physical c+w bank only",
            "stop_rule": "exact N16; no fill and no threshold relaxation",
        },
        "acquisition_result": binding(RESULT),
        "manifest": binding(MANIFEST),
        "worker_sidecars": [binding(FIRST), binding(LAST)],
        "root_overlays": [binding(ROOT_FIRST), binding(ROOT_LAST)],
        "counts": dict(counts),
        "candidate_decisions": {"accept": 16, "neutral": 34},
        "accepted_priorities_frozen_order": list(EXPECTED_ADMIT_PRIORITIES),
        "admitted_priorities_distinct_first": [row["admission_priority"] for row in admitted],
        "accepted_distinct_images": 11,
        "decisions": decisions,
        "claim_boundary": (
            "physical local c+w admission only; supplied/forced rows receive no natural credit; "
            "HOLD is neutral uncertainty and never a negative label"
        ),
    })
    print(json.dumps({"output": binding(OUTPUT), "counts": dict(counts),
                      "admitted_priorities_distinct_first":
                          [row["admission_priority"] for row in admitted]}, indent=2))


if __name__ == "__main__":
    main()
