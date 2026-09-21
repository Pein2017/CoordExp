"""Freeze the shared owner-grounded admission and finite row candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from probes.training_set_completion.numerical_feedback.select import rows, same

PANEL = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-distribution-census/shared-panel.json")
OUT_A = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-21-history-source-sign")
OUT_B = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-21-native-row-choice")
PROFILE = Path("research/experiments/2026-09-21-history-source-sign/task-profile.md")
BASE = 151670
ROW_OPEN, REF_END, BOX_START, ROW_END = 151646, 151647, 151648, 151649

# This is the bounded full-image/context audit frozen before intervention outcomes.
# A/C owner IDs refer only to the source-panel annotation snapshot named below.
AUDIT = {
    "untied-885-failure": {"lane_b": True, "a": 2154963, "reason": "A is the isolated top-left spectator; both numerical rows overlap the same annotation at IoU>0.5."},
    "tied-885-failure": {"lane_b": True, "a": 2154963, "reason": "A is the isolated top-left spectator; both numerical rows overlap the same annotation at IoU>0.6."},
    "untied-5586-failure": {"lane_b": False, "reason": "HOLD_identity: target boxes start at image origin and overlap no bound person annotation; the crowd does not supply a unique owner."},
    "tied-5586-failure": {"lane_b": False, "reason": "HOLD_identity: target boxes start at image origin and overlap no bound person annotation; the crowd does not supply a unique owner."},
    "untied-7511-failure": {"lane_b": False, "reason": "HOLD_identity: tiny left-edge person boxes have low overlap and multiple plausible refined person owners."},
    "untied-14038-failure": {"lane_b": False, "reason": "HOLD_identity: the first numerical pair changes its best bound book owner (-141 to -119) in a dense overlapping stack."},
    "tied-14038-failure": {"lane_b": True, "lane_a": True, "a": -125, "c": -141, "reason": "Rows 7/8 share bound book -125; row 6 supplies already-covered neighboring book -141."},
    "untied-632-failure": {"lane_b": False, "reason": "HOLD_identity: target book extents have at most 0.296 IoU and span multiple shelf books."},
    "tied-632-failure": {"lane_b": False, "reason": "HOLD_identity: target book extents have zero overlap with the bound individual book boxes."},
    "untied-417044-failure": {"lane_b": False, "reason": "HOLD_identity: crowded clipped donuts leave two plausible owners and no unique >=0.5 match."},
    "tied-417044-failure": {"lane_b": False, "reason": "HOLD_identity: crowded clipped donuts leave two plausible owners and no unique >=0.5 match."},
}


def binding(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _coords(obj: dict[str, Any]) -> list[int]:
    return [int(re.search(r"(\d+)", token).group(1)) for token in obj["bbox_2d"]]


def _iou(a: list[int], b: list[int]) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa, bb = max(0, a[2] - a[0]) * max(0, a[3] - a[1]), max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / (aa + bb - inter) if aa + bb - inter else 0.0


def _first_pair(parsed: list[dict[str, Any]]) -> tuple[int, int]:
    for later in range(len(parsed)):
        for earlier in range(later):
            if same(parsed[earlier], parsed[later], 8):
                return earlier, later
    raise ValueError("failure trajectory has no numerical revisit")


def _source_panel(boundary: dict[str, Any]) -> Path:
    path = Path(boundary["raw_path"]).resolve().parents[3] / "panel.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _case(boundary: dict[str, Any], source_panel: dict[str, Any]) -> dict[str, Any]:
    matches = [case for group in source_panel["groups"] for case in group["cases"]
               if group["key"] == boundary["group"] and int(case["input_record"]["image_id"]) == int(boundary["image_id"])]
    if len(matches) != 1:
        raise ValueError(f"nonunique source case for {boundary['id']}")
    return matches[0]


def _row_tokens(row: dict[str, Any], native: list[int]) -> list[int]:
    value = [int(x) for x in native[int(row["start"]):int(row["end"])]]
    if value[0] != ROW_OPEN or value[-1] != ROW_END:
        raise ValueError("native row boundary changed")
    return value


def _annotation_row(obj: dict[str, Any], description_tokens: list[int]) -> list[int]:
    values = _coords(obj)
    if not values[0] < values[2] or not values[1] < values[3]:
        raise ValueError("annotation candidate has invalid geometry")
    return [ROW_OPEN, *description_tokens, REF_END, BOX_START, *[BASE + x for x in values], ROW_END]


def _candidate(ident: str, owner: str, source: str, tokens: list[int], rule: str) -> dict[str, Any]:
    parsed = rows(tokens)
    if len(parsed) != 1 or int(parsed[0]["start"]) != 0 or int(parsed[0]["end"]) != len(tokens):
        raise ValueError(f"candidate is not one complete row: {ident}")
    return {"id": ident, "owner": owner, "source": source, "tokens": tokens,
            "token_sha256": digest(tokens), "description_tokens": parsed[0]["description_tokens"],
            "values": parsed[0]["values"], "valid": parsed[0]["valid"], "construction_rule": rule}


def _unique(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result, seen = [], set()
    for item in items:
        key = tuple(item["tokens"])
        if key not in seen:
            result.append(item)
            seen.add(key)
    return result


def _owner(objects: list[dict[str, Any]], owner_id: int) -> dict[str, Any]:
    matches = [obj for obj in objects if int(obj["coco_ann_id"]) == owner_id]
    if len(matches) != 1:
        raise ValueError(f"owner {owner_id} not unique")
    return matches[0]


def _sets(boundary: dict[str, Any], case: dict[str, Any], source_index: int, actual_index: int,
          a_owner: int, c_source_index: int | None = None, c_owner: int | None = None) -> dict[str, Any]:
    native = [int(x) for x in boundary["native_tokens"]]
    parsed = rows(native)
    source, actual = parsed[source_index], parsed[actual_index]
    objects = list(case["input_record"]["objects"])
    a_obj = _owner(objects, a_owner)
    if a_obj["desc"] not in case["input_record"]["metadata"].get("allowed_descriptions", [a_obj["desc"]]):
        pass  # legacy records need not carry this optional metadata field
    a = _unique([
        _candidate("A-native-source", f"annotation:{a_owner}", "native", _row_tokens(source, native), "first earlier numerical match"),
        _candidate("A-actual-greedy", f"annotation:{a_owner}", "native", _row_tokens(actual, native), "actual native greedy revisit row"),
        _candidate("A-annotation", f"annotation:{a_owner}", "bound_annotation", _annotation_row(a_obj, source["description_tokens"]), "bound coherent owner extent"),
    ])[:4]
    covered_ids = {a_owner}
    c: list[dict[str, Any]] = []
    if c_source_index is not None and c_owner is not None:
        c_row, c_obj = parsed[c_source_index], _owner(objects, c_owner)
        c = _unique([
            _candidate("C-native-source", f"annotation:{c_owner}", "native", _row_tokens(c_row, native), "covered reciprocal source row"),
            _candidate("C-annotation", f"annotation:{c_owner}", "bound_annotation", _annotation_row(c_obj, c_row["description_tokens"]), "bound coherent owner extent"),
        ])[:4]
        covered_ids.add(c_owner)
    eligible_n = sorted((obj for obj in objects if obj["desc"] == a_obj["desc"] and int(obj["coco_ann_id"]) not in covered_ids),
                        key=lambda obj: (*_coords(obj), int(obj["coco_ann_id"])))[:4]
    n = [_candidate(f"N-annotation-{obj['coco_ann_id']}", f"annotation:{obj['coco_ann_id']}", "bound_annotation",
                    _annotation_row(obj, source["description_tokens"]), "first four unvisited same-description owners in stable geometry order")
         for obj in eligible_n]
    if not n:
        raise ValueError(f"no credible N candidates for {boundary['id']}")
    all_unique = {tuple(item["tokens"]) for item in [*a, *c, *n]}
    if len(all_unique) > 13 or max(map(len, (a, c, n))) > 4:
        raise ValueError("finite candidate cap exceeded")
    return {"A": a, "C": c, "N": n, "actual_greedy_id": "A-actual-greedy",
            "unique_row_count": len(all_unique), "common_description_tokens": source["description_tokens"]}


def make_plan() -> dict[str, Any]:
    panel = json.loads(PANEL.read_text())
    pool = [b for b in panel["all_boundaries"] if b["kind"] == "failure" and "-train-" not in b["id"]]
    pool.sort(key=lambda b: (b["split"], int(b["image_id"]), b["model"]))
    if len(pool) != 11 or len({int(b["image_id"]) for b in pool}) != 6 or set(AUDIT) != {b["id"] for b in pool}:
        raise ValueError("frozen six-image/11-trajectory source pool changed")
    records, sources = [], {}
    for boundary in pool:
        source_path = _source_panel(boundary)
        source_panel = json.loads(source_path.read_text())
        case = _case(boundary, source_panel)
        native = [int(x) for x in boundary["native_tokens"]]
        parsed = rows(native)
        earlier, later = _first_pair(parsed)
        audit = dict(AUDIT[boundary["id"]])
        record: dict[str, Any] = {
            "source_boundary_id": boundary["id"], "model": boundary["model"], "split": boundary["split"],
            "image_id": int(boundary["image_id"]), "group": boundary["group"], "batch_index": int(boundary["batch_index"]),
            "selection_order": len(records), "first_numerical_pair": [earlier, later], "prefix_end": int(parsed[later]["start"]),
            "source_row": parsed[earlier], "actual_row": parsed[later], "audit": audit,
            "image": binding(Path(case["image_path"])), "image_plan": case.get("image_plan"),
            "annotation_snapshot_sha256": digest(case["input_record"]["objects"]),
            "annotation_count": len(case["input_record"]["objects"]), "source_panel": binding(source_path),
            "raw": binding(Path(boundary["raw_path"])), "trace": binding(Path(boundary["trace_path"])),
            "receipt": binding(Path(boundary["receipt_path"])), "native_token_sha256": digest(native),
            "native_token_count": len(native), "lane_a_status": "HOLD", "lane_b_status": "HOLD",
        }
        if audit.get("lane_b"):
            record["candidate_sets"] = _sets(boundary, case, earlier, later, int(audit["a"]),
                                               earlier - 1 if audit.get("lane_a") else None,
                                               int(audit["c"]) if audit.get("lane_a") else None)
            record["lane_b_status"] = "ready"
        if audit.get("lane_a"):
            if (boundary["id"], earlier, later) != ("tied-14038-failure", 7, 8):
                raise ValueError("Lane A owner-grounded boundary changed")
            record["lane_a_status"] = "ready"
        records.append(record)
        sources[boundary["id"]] = {key: record[key] for key in ("source_panel", "raw", "trace", "receipt", "image")}

    target = next(item for item in records if item["source_boundary_id"] == "tied-14038-failure")
    boundary = next(item for item in pool if item["id"] == target["source_boundary_id"])
    source_panel = json.loads(Path(target["source_panel"]["path"]).read_text())
    case = _case(boundary, source_panel)
    parsed = rows(boundary["native_tokens"])
    # Matched control: one row earlier, after covered book A(row6) and covered tv C(row5),
    # the native next row is new book -125 rather than a revisit of designated A=-141.
    control_sets = _sets(boundary, case, 6, 7, -141, 5, 33179)
    control = {
        "id": "tied-14038-control-before-row7", "source_boundary_id": boundary["id"], "model": "tied",
        "image_id": 14038, "kind": "matched_non_target_revisit", "prefix_end": int(parsed[7]["start"]),
        "source_row_index_A": 6, "source_row_index_C": 5, "native_next_row_index": 7,
        "designated_A": "annotation:-141", "designated_C": "annotation:33179", "native_next_owner": "annotation:-125",
        "match": {"failure_id": "tied-14038-failure", "same_model": True, "same_image": True,
                  "description_A_vs_native_next": "book", "generation_row_delta": -1,
                  "history_exposure": "A once; C once", "residual_mismatch": "C is tv while A/N are book"},
        "candidate_sets": control_sets, "status": "ready",
    }
    failure = {
        "id": "tied-14038-failure-before-row8", "source_boundary_id": boundary["id"], "model": "tied",
        "image_id": 14038, "kind": "target_first_revisit", "prefix_end": int(parsed[8]["start"]),
        "source_row_index_A": 7, "source_row_index_C": 6, "native_next_row_index": 8,
        "designated_A": "annotation:-125", "designated_C": "annotation:-141", "native_next_owner": "annotation:-125",
        "candidate_sets": target["candidate_sets"], "status": "ready",
    }
    lane_b_states = [{
        "id": item["source_boundary_id"].replace("-failure", "-first-revisit"),
        "source_boundary_id": item["source_boundary_id"], "model": item["model"], "image_id": item["image_id"],
        "prefix_end": item["prefix_end"], "source_row_index_A": item["first_numerical_pair"][0],
        "actual_row_index": item["first_numerical_pair"][1], "candidate_sets": item["candidate_sets"], "status": "ready",
    } for item in records if item["lane_b_status"] == "ready"]
    return {
        "schema": "history_source_sign.shared_admission.v1", "status": "frozen_before_model_comparisons",
        "profile": binding(PROFILE), "panel": binding(PANEL), "source_pool_rule": "existing mature failure boundaries; six legacy images; stable split,image,model order; no replacement",
        "source_pool": records, "source_bindings": sources,
        "lane_a": {"question": "signed same-prefix source effect with separately matched non-revisit control",
                   "barrier": "block all target queries from source row box_end query onward from attending source row content [row_start,box_end); full-prefix recomputation",
                   "conditions": ["native", "cut_A", "cut_C"], "failure_boundaries": [failure], "control_boundaries": [control],
                   "counts": {"source_trajectories": 11, "owner_grounded_failure": 1, "failure_hold": 10,
                              "admitted_matched_pairs": 1, "control_hold": 0}},
        "lane_b": {"question": "finite native complete-row witness under original distribution", "states": lane_b_states,
                   "counts": {"source_trajectories": 11, "ready": len(lane_b_states), "hold": 11 - len(lane_b_states)}},
        "limits": {"allocated_gpu_hours": 4, "model_forwards": 40000, "free_release_cells": 144, "retained_bytes": 16 * 1024**3},
        "tolerances": {"noop_full_vocab_atol": 2e-4, "barrier_invariance_atol": 2e-4, "numerical_repeat_coord_tolerance": 8},
        "claim_limits": ["finite tested row sets are not complete owner mass", "matched boundaries are observational controls", "identity HOLD is not a physical negative"],
    }


def selfcheck() -> None:
    row = lambda value: [ROW_OPEN, 9, REF_END, BOX_START, *[BASE + x for x in value], ROW_END]
    parsed = rows(row([1, 2, 3, 4]) + row([2, 2, 3, 4]))
    assert _first_pair(parsed) == (0, 1)
    assert _annotation_row({"bbox_2d": ["<|coord_1|>", "<|coord_2|>", "<|coord_3|>", "<|coord_4|>"]}, [9]) == row([1, 2, 3, 4])
    assert _iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    print("PASS earliest-pair, coherent-row, IoU")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    plan = make_plan()
    write_new(OUT_A / "selection" / "shared-admission.json", plan)
    write_new(OUT_B / "selection" / "shared-admission-binding.json", {"status": "frozen", "shared_admission": binding(OUT_A / "selection" / "shared-admission.json")})
    print(json.dumps({"lane_a": plan["lane_a"]["counts"], "lane_b": plan["lane_b"]["counts"]}, indent=2))


if __name__ == "__main__":
    main()
