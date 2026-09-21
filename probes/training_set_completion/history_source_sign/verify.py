"""Independent CPU check of the frozen shared admission bytes and caps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from probes.training_set_completion.history_source_sign.prepare import OUT_A, digest, make_plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=OUT_A / "selection" / "shared-admission.json")
    args = parser.parse_args()
    saved = json.loads(args.plan.read_text())
    recomputed = make_plan()
    assert saved == recomputed
    assert len(saved["source_pool"]) == 11
    assert len({item["image_id"] for item in saved["source_pool"]}) == 6
    assert saved["lane_a"]["counts"] == {"source_trajectories": 11, "owner_grounded_failure": 1, "failure_hold": 10, "admitted_matched_pairs": 1, "control_hold": 0}
    assert saved["lane_b"]["counts"] == {"source_trajectories": 11, "ready": 3, "hold": 8}
    assert all(state["candidate_sets"]["unique_row_count"] <= 13 for state in saved["lane_b"]["states"])
    print(json.dumps({"status": "PASS", "plan_digest": digest(saved), "lane_a": saved["lane_a"]["counts"], "lane_b": saved["lane_b"]["counts"]}, indent=2))


if __name__ == "__main__":
    main()
