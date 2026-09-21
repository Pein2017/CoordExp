"""Freeze the shared admission for spatial-progress and visual-binding probes."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


OLD = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-21-history-source-sign/selection/shared-admission.json")
PROFILE = Path("research/experiments/2026-09-21-spatial-progress-gate/task-profile.md")
OUT_A = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-21-spatial-progress-gate")
OUT_B = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-21-visual-instance-binding")
COORD_BASE = 151670


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def binding(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def candidate(state: dict[str, Any], set_name: str, ident: str) -> dict[str, Any]:
    rows = [row for row in state["candidate_sets"][set_name] if row["id"] == ident]
    if len(rows) != 1:
        raise ValueError(f"candidate is not unique: {state['id']}/{set_name}/{ident}")
    return copy.deepcopy(rows[0])


def edited_condition(boundary: dict[str, Any], name: str, role: str, value: int) -> dict[str, Any]:
    row = boundary["source_row"]
    role_index = {"x1": 0, "y1": 1, "x2": 2, "y2": 3}[role]
    absolute_offset = int(row["coordinate_offsets"][role_index])
    old_value = int(row["values"][role_index])
    return {
        "id": name,
        "history_tokens": "edited_native_prefix",
        "edited_row_index": int(row["index"]),
        "role": role,
        "absolute_offset": absolute_offset,
        "old_value": old_value,
        "new_value": value,
        "old_token_id": COORD_BASE + old_value,
        "new_token_id": COORD_BASE + value,
        "delta_bins": value - old_value,
    }


def mask(owner: str, normalized: list[int], image_size: list[int]) -> dict[str, Any]:
    width, height = image_size
    x1, y1, x2, y2 = normalized
    pixels = [
        max(0, min(width, math.floor(x1 * width / 1000))),
        max(0, min(height, math.floor(y1 * height / 1000))),
        max(0, min(width, math.ceil(x2 * width / 1000))),
        max(0, min(height, math.ceil(y2 * height / 1000))),
    ]
    if not pixels[0] < pixels[2] or not pixels[1] < pixels[3]:
        raise ValueError(f"empty mask for {owner}")
    return {
        "owner": owner,
        "normalized_xyxy": normalized,
        "pixel_xyxy": pixels,
        "pixel_area": (pixels[2] - pixels[0]) * (pixels[3] - pixels[1]),
        "fill": "integer-rounded mean RGB of the clipped 8-pixel surrounding ring, excluding the mask",
    }


def make_plan() -> dict[str, Any]:
    old = json.loads(OLD.read_text())
    if old.get("status") != "frozen_before_model_comparisons" or len(old.get("source_pool", [])) != 11:
        raise ValueError("accepted predecessor admission changed")
    old_states = {state["id"]: state for state in old["lane_b"]["states"]}
    failure = copy.deepcopy(old["lane_a"]["failure_boundaries"][0])
    control = copy.deepcopy(old["lane_a"]["control_boundaries"][0])
    if failure["id"] != "tied-14038-failure-before-row8" or control["id"] != "tied-14038-control-before-row7":
        raise ValueError("tied14038 boundary binding changed")
    n_common = [
        candidate(old_states["tied-14038-first-revisit"], "N", ident)
        for ident in ("N-annotation--127", "N-annotation--120", "N-annotation--142", "N-annotation--128")
    ]
    a_failure = [copy.deepcopy(row) for row in failure["candidate_sets"]["A"]]
    a_control = [
        candidate(control, "A", "A-native-source"),
        candidate(control, "A", "A-annotation"),
    ]
    lane_a_boundaries = []
    for raw, a_rows in ((failure, a_failure), (control, a_control)):
        source_record = next(item for item in old["source_pool"] if item["source_boundary_id"] == raw["source_boundary_id"])
        rows = raw["source_row_index_A"]
        # The accepted record carries parsed row coordinates on the source-pool failure.
        if raw["kind"] == "target_first_revisit":
            source_row = source_record["source_row"]
        else:
            native = [int(token) for token in next(
                item for item in json.loads(Path(old["panel"]["path"]).read_text())["all_boundaries"]
                if item["id"] == raw["source_boundary_id"]
            )["native_tokens"]]
            from probes.training_set_completion.numerical_feedback.select import rows as parse_rows
            source_row = parse_rows(native)[rows]
        base = {
            "id": raw["id"],
            "source_boundary_id": raw["source_boundary_id"],
            "model": raw["model"],
            "image_id": raw["image_id"],
            "kind": raw["kind"],
            "prefix_end": raw["prefix_end"],
            "source_row": source_row,
            "designated_A": raw["designated_A"],
            "native_next_owner": raw["native_next_owner"],
            "fixed_crossed_N": "annotation:-128",
            "fixed_crossed_N_x1": 786,
            "candidate_sets": {"A": a_rows, "N": copy.deepcopy(n_common)},
            "actual_row": {
                "index": raw["native_next_row_index"],
                "owner": raw["native_next_owner"],
            },
        }
        base["conditions"] = [
            {"id": "native", "history_tokens": "exact_native_prefix", "alias_of": None},
            edited_condition(base, "x1_before_N786", "x1", 784),
            {**edited_condition(base, "x1_after_N786", "x1", 788), "alias_of": "native"},
            edited_condition(base, "sham_y2_minus4", "y2", int(source_row["values"][3]) - 4),
            edited_condition(base, "sham_y2_plus4", "y2", int(source_row["values"][3]) + 4),
        ]
        base["unique_execution_conditions"] = ["native", "x1_before_N786", "sham_y2_minus4", "sham_y2_plus4"]
        lane_a_boundaries.append(base)

    image_sizes = {885: [1248, 832], 14038: [1248, 832]}
    visual_states: list[dict[str, Any]] = []
    for state_id in ("tied-885-first-revisit", "untied-885-first-revisit", "tied-14038-first-revisit"):
        old_state = old_states[state_id]
        is_885 = int(old_state["image_id"]) == 885
        if is_885:
            a_owner, a_box = "annotation:2154963", [1, 1, 94, 22]
            n_id, n_owner, n_box = "N-annotation-542506", "annotation:542506", [448, 0, 519, 28]
            control_box = [100, 700, 193, 721]
        else:
            a_owner, a_box = "annotation:-125", [784, 670, 851, 704]
            n_id, n_owner, n_box = "N-annotation--128", "annotation:-128", [786, 751, 837, 784]
            control_box = [650, 670, 717, 704]
        visual_states.append({
            "id": state_id,
            "source_boundary_id": old_state["source_boundary_id"],
            "model": old_state["model"],
            "image_id": old_state["image_id"],
            "kind": "target_first_revisit",
            "prefix_end": old_state["prefix_end"],
            "designated_A": a_owner,
            "designated_N": n_owner,
            "actual_row": {"candidate_id": old_state["candidate_sets"]["actual_greedy_id"], "owner": a_owner},
            "candidate_sets": {
                "A": copy.deepcopy(old_state["candidate_sets"]["A"]),
                "N": [candidate(old_state, "N", n_id)],
            },
            "masks": {
                "A": mask(a_owner, a_box, image_sizes[int(old_state["image_id"])]),
                "N": mask(n_owner, n_box, image_sizes[int(old_state["image_id"])]),
                "unrelated": mask("unrelated_region", control_box, image_sizes[int(old_state["image_id"])]),
            },
            "conditions": ["clean", "ablate_A", "ablate_N", "ablate_unrelated"],
        })
    visual_states.append({
        "id": "tied-14038-control-before-row7",
        "source_boundary_id": control["source_boundary_id"],
        "model": "tied",
        "image_id": 14038,
        "kind": "matched_non_target_revisit",
        "prefix_end": control["prefix_end"],
        "designated_A": "annotation:-141",
        "designated_N": "annotation:-128",
        "actual_row": {"candidate_id": "actual-native-next", "owner": "annotation:-125", "values": [788, 670, 850, 697]},
        "candidate_sets": {"A": copy.deepcopy(a_control), "N": [candidate(old_states["tied-14038-first-revisit"], "N", "N-annotation--128")]},
        "masks": {
            "A": mask("annotation:-141", [785, 660, 851, 690], image_sizes[14038]),
            "N": mask("annotation:-128", [786, 751, 837, 784], image_sizes[14038]),
            "unrelated": mask("unrelated_region", [650, 660, 716, 690], image_sizes[14038]),
        },
        "conditions": ["clean", "ablate_A", "ablate_N", "ablate_unrelated"],
        "match": copy.deepcopy(control["match"]),
    })

    lane_a_holds = [
        {
            "source_boundary_id": item["source_boundary_id"],
            "reason": (
                "HOLD_no_owner_preserving_crossing: retained N candidates lie lexicographically ahead of A"
                if item["image_id"] == 885 and item["lane_b_status"] == "ready"
                else "HOLD_owner_identity_or_crossing_unavailable_from_accepted_evidence"
            ),
        }
        for item in old["source_pool"]
        if item["source_boundary_id"] != "tied-14038-failure"
    ]
    lane_b_holds = [
        {"source_boundary_id": item["source_boundary_id"], "reason": item["audit"]["reason"]}
        for item in old["source_pool"] if item["lane_b_status"] != "ready"
    ]
    lane_b_control_holds = [
        {"source_boundary_id": ident, "reason": "HOLD_no_matched_non_target_revisit_boundary at comparable exposure/stage"}
        for ident in ("tied-885-failure", "untied-885-failure")
    ]
    plan = {
        "schema": "spatial_progress_visual_binding.shared_admission.v1",
        "status": "frozen_before_intervention_scores",
        "profile": binding(PROFILE),
        "accepted_predecessor_admission": binding(OLD),
        "panel": copy.deepcopy(old["panel"]),
        "source_pool_rule": old["source_pool_rule"],
        "source_pool": copy.deepcopy(old["source_pool"]),
        "source_bindings": copy.deepcopy(old["source_bindings"]),
        "lane_a": {
            "question": "Does moving an owner-preserving historical reference across fixed N in lexicographic order directionally increase N preference?",
            "boundaries": lane_a_boundaries,
            "holds": lane_a_holds,
            "counts": {"source_trajectories": 11, "admitted_failure": 1, "failure_hold": 10, "admitted_control": 1, "control_hold": 0, "unique_scientific_cells": 8},
            "condition_rule": "x1 784/788 crosses fixed N x1=786; native is the after arm; y2 +/-4 are equal-magnitude non-order shams",
        },
        "lane_b": {
            "question": "Do separately localized A and N ablations have distinct correctly directed effects, and is separation lost at recurrence?",
            "states": visual_states,
            "failure_holds": lane_b_holds,
            "control_holds": lane_b_control_holds,
            "counts": {"source_trajectories": 11, "admitted_failure": 3, "failure_hold": 8, "admitted_control": 1, "control_hold": 2, "scientific_cells": 16},
            "compositor_rule": "decoded RGB; exact rectangle; integer-rounded clipped 8-pixel surrounding-ring mean; unchanged complement; PNG output",
        },
        "limits": {"allocated_gpu_hours": 8, "model_forwards": 100000, "free_release_cells": 192, "retained_bytes": 16 * 1024**3},
        "tolerances": {"full_vocab_replay_atol": 2e-4, "clean_compositor_full_vocab_atol": 2e-4, "unchanged_pixel_atol": 0, "score_reduction_atol": 1e-5},
        "claim_limits": [
            "candidate sets are finite tested rows, not total owner probability",
            "matched boundaries are observational controls",
            "identity HOLD is not a physical negative",
            "visual interventions do not identify training origin",
            "secondary releases, if any, are outcome-selected",
        ],
    }
    return plan


def selfcheck(plan: dict[str, Any]) -> None:
    assert len(plan["source_pool"]) == 11
    assert len(plan["lane_a"]["boundaries"]) == 2
    for boundary in plan["lane_a"]["boundaries"]:
        by_id = {item["id"]: item for item in boundary["conditions"]}
        assert by_id["x1_before_N786"]["new_value"] < 786 < by_id["x1_after_N786"]["new_value"]
        assert by_id["x1_after_N786"]["alias_of"] == "native"
        assert abs(by_id["sham_y2_minus4"]["delta_bins"]) == abs(by_id["sham_y2_plus4"]["delta_bins"]) == 4
    assert len(plan["lane_b"]["states"]) == 4
    for state in plan["lane_b"]["states"]:
        a, n = state["masks"]["A"]["pixel_xyxy"], state["masks"]["N"]["pixel_xyxy"]
        assert a[2] <= n[0] or n[2] <= a[0] or a[3] <= n[1] or n[3] <= a[1]
        ratio = state["masks"]["A"]["pixel_area"] / state["masks"]["N"]["pixel_area"]
        assert 0.5 <= ratio <= 2.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    plan = make_plan()
    selfcheck(plan)
    if args.selfcheck:
        print("PASS pool, crossing direction, shams, visual separation and area comparability")
        return
    write_new(OUT_A / "selection" / "shared-admission.json", plan)
    write_new(OUT_B / "selection" / "shared-admission-binding.json", {"status": "frozen", "shared_admission": binding(OUT_A / "selection" / "shared-admission.json")})
    print(json.dumps({"lane_a": plan["lane_a"]["counts"], "lane_b": plan["lane_b"]["counts"]}, indent=2))


if __name__ == "__main__":
    main()
