"""Cold aggregate of the frozen 156-row scale acquisition denominator."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from probes.dora_owner_learning.candidate_opportunity import require
from probes.native_owner_scale.scale import binding, publish, read, validate_selection


ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-12-native-owner-scale-and-state/scale"
)
RESULT = ROOT / "acquisition-full-v2.json"
OUTPUT = ROOT / "acquisition-full-v2-summary.json"


def main() -> None:
    result = read(RESULT)
    require(result["mode"] == "full" and len(result["rows"]) == 156, "full denominator")
    selection = validate_selection(result["selection"]["path"])
    by_job = {row["job_id"]: row for row in selection["nominations"]}
    rows = result["rows"]
    candidates = [row for row in rows if row["local_w"]["status"] == "candidate_local_w"]
    burden_keys = tuple(rows[0]["burden"])
    local_by_stratum = Counter(row["stratum"] for row in candidates)
    attempted_by_stratum = Counter(row["stratum"] for row in rows)
    boundary_attempts = Counter(by_job[row["job_id"]]["history_boundary"] for row in rows)
    boundary_candidates = Counter(by_job[row["job_id"]]["history_boundary"] for row in candidates)
    h_owner_losses = sum(
        len(set(by_job[row["job_id"]]["h_owner_ids50"]) - set(row["full_score"]["50"]["owners"]))
        for row in rows
    )
    gt_c_match_failures = sum(
        by_job[row["job_id"]]["owner_id"] not in set(row["full_score"]["50"]["owners"])
        for row in rows
        if by_job[row["job_id"]]["source"] == "coco_gt_annotation"
    )
    summary = {
        "schema": "native_owner_scale.acquisition_summary.v1",
        "status": "cold_reduced_visual_acceptance_pending",
        "result": binding(RESULT),
        "denominators": result["denominators"],
        "cost": result["cost"],
        "local_w": {
            "status_counts": dict(Counter(row["local_w"]["status"] for row in rows)),
            "candidate_by_stratum": dict(local_by_stratum),
            "attempted_by_stratum": dict(attempted_by_stratum),
            "candidate_rate_by_stratum": {
                key: local_by_stratum[key] / attempted_by_stratum[key]
                for key in attempted_by_stratum
            },
            "candidate_distinct_images": len({row["example_id"] for row in candidates}),
            "provenance": dict(Counter(by_job[row["job_id"]]["source"] for row in candidates)),
        },
        "history_boundary": {
            "attempts": dict(boundary_attempts),
            "candidate_local_w": dict(boundary_candidates),
        },
        "full_continuation_burden": {
            "stop_reason": dict(Counter(row["stop_reason"] for row in rows)),
            "sums": {key: sum(row["burden"][key] for row in rows) for key in burden_keys},
            "full_overlap_sums": {
                key: sum(row["full_overlap_counts"][key] for row in rows)
                for key in rows[0]["full_overlap_counts"]
            },
            "free_iou50_owner_matches": sum(len(row["free_score"]["50"]["owners"]) for row in rows),
            "supplied_h_iou50_owner_losses_in_full_sequence": h_owner_losses,
            "gt_c_iou50_match_failures_in_full_sequence": gt_c_match_failures,
        },
        "interpretation_boundary": (
            "local-w counts are machine candidates pending image review; supplied h and forced c receive no "
            "natural credit; full suffix repeats and malformed burden do not invalidate a genuine local c+w"
        ),
    }
    publish(OUTPUT, summary)
    print(json.dumps({"summary": binding(OUTPUT), "local_w": summary["local_w"]}, indent=2))


if __name__ == "__main__":
    main()
