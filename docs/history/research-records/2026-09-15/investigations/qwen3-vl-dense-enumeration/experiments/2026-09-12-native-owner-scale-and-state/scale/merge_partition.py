"""One-off immutable merge for the frozen scale slice4 + remainder152 partition.

The general reducer sorts rows by admission priority, while the launch packet
stores the four representative slice IDs in presentation order. Partition
identity is therefore a set invariant; final output order remains the frozen
admission priority.
"""
from __future__ import annotations

import json
from pathlib import Path

from probes.dora_owner_learning.candidate_opportunity import require
from probes.native_owner_scale.scale import binding, publish, read, validate_acquisition, validate_selection


ROOT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-native-owner-scale-and-state/scale")
PACKET = ROOT / "preparation" / "acquisition-v2-remainder.json"
SLICE = ROOT / "acquisition-slice-v2-execfix" / "result.json"
REMAINDER = ROOT / "acquisition-remainder-v2" / "result.json"
OUTPUT = ROOT / "acquisition-full-v2.json"


def main() -> None:
    packet = validate_acquisition(PACKET)
    slice_result, remainder_result = read(SLICE), read(REMAINDER)
    require(slice_result["schema"] == remainder_result["schema"]
            == "native_owner_scale.acquisition_result.v1", "partition result schemas")
    require(slice_result["mode"] == "slice" and remainder_result["mode"] == "remainder",
            "partition result modes")
    selection = validate_selection(packet["sources"]["selection"]["path"])
    by_job = {row["job_id"]: row for row in selection["nominations"]}
    slice_ids = [row["job_id"] for row in slice_result["rows"]]
    remainder_ids = [row["job_id"] for row in remainder_result["rows"]]
    require(set(slice_ids) == set(packet["modes"]["slice"])
            and set(remainder_ids) == set(packet["modes"]["remainder"]),
            "observed/frozen partition identity")
    require(set(slice_ids).isdisjoint(remainder_ids)
            and set(slice_ids) | set(remainder_ids) == set(packet["modes"]["full"])
            and len(slice_ids) + len(remainder_ids) == len(set(packet["modes"]["full"])) == 156,
            "exactly 156 unique frozen IDs")
    rows = [*slice_result["rows"], *remainder_result["rows"]]
    for row in rows:
        frozen = by_job[row["job_id"]]
        require(all(row[key] == frozen[key] for key in
                    ("example_id", "owner_id", "stratum", "candidate_index", "admission_priority")),
                "merged row differs from frozen nomination")
    rows.sort(key=lambda row: by_job[row["job_id"]]["admission_priority"])
    require([row["admission_priority"] for row in rows] == list(range(156)),
            "frozen admission priority")
    result = {"schema": "native_owner_scale.acquisition_result.v1",
        "status": "candidate_cpu_verified_visual_review_pending", "mode": "full",
        "packet": binding(PACKET), "selection": packet["sources"]["selection"],
        "partitions": {"slice": binding(SLICE), "remainder": binding(REMAINDER)},
        "merge_consumer": binding(Path(__file__).resolve()),
        "denominators": {"source_universe": 384, "raw_images": 128,
            "frozen_nominations": 156, "executed": 156, "terminal_failures": 0,
            "candidate_local_w": sum(row["local_w"]["status"] == "candidate_local_w" for row in rows)},
        "rows": rows, "cost": {key: slice_result["cost"][key] + remainder_result["cost"][key]
                                for key in ("model_forwards", "image_forwards", "allocated_gpu_hours")},
        "raw_row_sources": [*slice_result["raw_row_sources"], *remainder_result["raw_row_sources"]],
        "claim_boundary": "machine local-w candidacy is not visual owner acceptance and forced rows receive no natural credit",
    }
    publish(OUTPUT, result)
    print(json.dumps({"output": binding(OUTPUT), "denominators": result["denominators"],
                      "cost": result["cost"]}))


if __name__ == "__main__":
    main()
