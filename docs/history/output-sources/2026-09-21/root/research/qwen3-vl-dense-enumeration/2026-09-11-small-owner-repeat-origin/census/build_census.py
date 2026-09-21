#!/usr/bin/env python3
"""Build an eight-case, CPU-only census of frozen Stable50 natural rows.

This is descriptive token/geometry bookkeeping.  It does not inspect GT,
images, logits, or model state and makes no visual-truth or causal claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer


WORKTREE = Path("/data/CoordExp/.worktrees/research-probes")
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

from probes.dora_owner_learning.geometric_dedup import (  # noqa: E402
    DUPLICATE_IOU_THRESHOLD,
    _exact_token_text_frame,
    trajectory_layout,
)
from src.data.geometry import iou_xyxy  # noqa: E402


DEFAULT_INPUTS = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-11-stable50-geometric-dedup/inputs.json"
)
EXPECTED_IDS = [9813, 158044, 248167, 274509, 351017, 417044, 477415, 502725]
EXPECTED_STRICT_REPEATS = [0, 5, 11, 10, 136, 291, 4, 38]

OBJECT_START = "<|object_ref_start|>"
OBJECT_END = "<|object_ref_end|>"
BOX_START = "<|box_start|>"
BOX_END = "<|box_end|>"
COORD_RE = re.compile(r"<\|coord_(0|[1-9][0-9]{0,2})\|>")
CANONICAL_ROW_RE = re.compile(
    "^"
    + re.escape(OBJECT_START)
    + r"(?P<raw_description>.*?)"
    + re.escape(OBJECT_END)
    + re.escape(BOX_START)
    + r"(?P<coords>(?:<\|coord_[^>]+\|>){4})"
    + re.escape(BOX_END),
    re.DOTALL,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def exact_char_to_token_interval(
    char_start: int,
    char_end: int,
    token_spans: list[tuple[int, int]],
) -> tuple[int, int]:
    starts = [i for i, (start, end) in enumerate(token_spans) if end > start and start == char_start]
    ends = [i + 1 for i, (start, end) in enumerate(token_spans) if end > start and end == char_end]
    if len(starts) != 1 or len(ends) != 1 or ends[0] <= starts[0]:
        raise ValueError(f"character interval [{char_start}, {char_end}) is not an exact token interval")
    return starts[0], ends[0]


def unvalidated_pixel_box(coord_bins: list[int], width: int, height: int) -> list[int]:
    """Project canonical 0..999 endpoints exactly, without declaring geometry valid."""

    return [
        round(coord_bins[0] * width / 1000),
        round(coord_bins[1] * height / 1000),
        round(coord_bins[2] * width / 1000),
        round(coord_bins[3] * height / 1000),
    ]


def normalized_area(pixel_box: list[int], width: int, height: int) -> float:
    x1, y1, x2, y2 = pixel_box
    return max(0, x2 - x1) * max(0, y2 - y1) / (width * height)


def native_rows_by_order(layout: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for value in layout["row_spans"]:
        order = int(value["generated_order"])
        rows[order] = {"kind": "accepted", "value": value}
    for value in layout["parser_drop_rows"]:
        order = value.get("generated_order")
        if isinstance(order, int) and not isinstance(order, bool):
            if order in rows:
                raise ValueError(f"generated order {order} is both accepted and dropped")
            rows[order] = {"kind": "dropped", "value": value}
    return rows


def build_raw_rows(
    *,
    case: dict[str, Any],
    layout: dict[str, Any],
    action_text: str,
    token_spans: list[tuple[int, int]],
    token_pieces: list[str],
) -> list[dict[str, Any]]:
    action_ids = case["action_ids"]
    width, height = int(case["image_width"]), int(case["image_height"])
    starts = [match.start() for match in re.finditer(re.escape(OBJECT_START), action_text)]
    native = native_rows_by_order(layout)
    if set(native) != set(range(len(starts))):
        raise ValueError("native accepted/dropped generated orders do not cover every raw object start")

    rows: list[dict[str, Any]] = []
    prior_valid: list[dict[str, Any]] = []
    for order, candidate_start in enumerate(starts):
        candidate_end = starts[order + 1] if order + 1 < len(starts) else len(action_text)
        entry = native[order]
        value = entry["value"]
        char_start = int(value["char_start"])
        char_end = int(value["char_end"])
        if char_start != candidate_start or char_end > candidate_end:
            raise ValueError(f"native row {order} is outside its raw object candidate")
        token_start, token_end = exact_char_to_token_interval(char_start, char_end, token_spans)
        raw_text = action_text[char_start:char_end]
        row_token_ids = action_ids[token_start:token_end]
        row_token_pieces = token_pieces[token_start:token_end]
        if "".join(row_token_pieces) != raw_text:
            raise ValueError(f"row {order} token pieces do not reconstruct raw text")

        canonical = CANONICAL_ROW_RE.match(raw_text)
        is_complete = canonical is not None and canonical.end() == len(raw_text)
        raw_description = canonical.group("raw_description") if canonical else None
        coord_tokens = COORD_RE.findall(canonical.group("coords")) if canonical else []
        coord_bins = [int(value) for value in coord_tokens] if len(coord_tokens) == 4 else None
        projected = unvalidated_pixel_box(coord_bins, width, height) if coord_bins else None
        geometry_valid = entry["kind"] == "accepted"
        if geometry_valid:
            pixel_box = [int(value) for value in entry["value"]["bbox_pixel_xyxy"]]
            if projected != pixel_box:
                raise ValueError(f"row {order} native pixel box differs from exact endpoint projection")
            description = str(entry["value"]["description"])
            reason = None
        else:
            pixel_box = None
            description = raw_description.strip() if raw_description is not None else None
            reason = str(entry["value"].get("reason", "unknown_drop"))

        prior_max_iou = None
        prior_max_index = None
        strict_refs: list[int] = []
        if geometry_valid:
            scored = [
                (float(iou_xyxy(tuple(projected), tuple(previous["pixel_box_xyxy"]))), previous["raw_row_index"])
                for previous in prior_valid
            ]
            if scored:
                prior_max_iou, prior_max_index = max(scored, key=lambda item: item[0])
            strict_refs = [index for overlap, index in scored if overlap > DUPLICATE_IOU_THRESHOLD]

        previous_same = rows[-1] if rows and rows[-1].get("description") == description else None
        delta_bins = None
        delta_pixels = None
        previous_iou = None
        if previous_same is not None and coord_bins is not None and previous_same.get("coord_bins") is not None:
            delta_bins = [a - b for a, b in zip(coord_bins, previous_same["coord_bins"])]
            delta_pixels = [a - b for a, b in zip(projected, previous_same["projected_pixel_xyxy_unvalidated"])]
            if geometry_valid and previous_same["geometry_valid"]:
                previous_iou = float(iou_xyxy(tuple(projected), tuple(previous_same["pixel_box_xyxy"])))

        row = {
            "raw_row_index": order,
            "object_span_id": str(value.get("object_span_id", "")),
            "native_status": "accepted" if geometry_valid else reason,
            "geometry_valid": geometry_valid,
            "complete_canonical_row": is_complete,
            "char_start": char_start,
            "char_end_exclusive": char_end,
            "candidate_char_end_exclusive": candidate_end,
            "token_start": token_start,
            "token_end_exclusive": token_end,
            "prefix_token_count_at_start": token_start,
            "prefix_token_count_after_end": token_end,
            "token_ids": row_token_ids,
            "token_texts": row_token_pieces,
            "raw_text": raw_text,
            "raw_text_sha256": sha256_text(raw_text),
            "raw_description": raw_description,
            "description": description,
            "coord_bins": coord_bins,
            "projected_pixel_xyxy_unvalidated": projected,
            "pixel_box_xyxy": pixel_box,
            "area_fraction_of_image": normalized_area(projected, width, height) if projected else None,
            "prior_valid_max_iou": prior_max_iou,
            "prior_valid_max_iou_row_index": prior_max_index,
            "strict_iou_gt_0_95_repeat": bool(strict_refs),
            "strict_repeat_reference_row_indices": strict_refs,
            "same_description_as_previous_complete_row": previous_same is not None,
            "previous_same_description_row_index": previous_same["raw_row_index"] if previous_same else None,
            "coord_bin_delta_from_previous_same_description_row": delta_bins,
            "pixel_endpoint_delta_from_previous_same_description_row": delta_pixels,
            "iou_to_previous_same_description_valid_row": previous_iou,
        }
        rows.append(row)
        if geometry_valid:
            prior_valid.append(row)
    return rows


def row_pointer(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "raw_row_index": row["raw_row_index"],
        "description": row["description"],
        "native_status": row["native_status"],
        "coord_bins": row["coord_bins"],
        "pixel_box_xyxy": row["pixel_box_xyxy"],
        "projected_pixel_xyxy_unvalidated": row["projected_pixel_xyxy_unvalidated"],
        "token_start": row["token_start"],
        "token_end_exclusive": row["token_end_exclusive"],
        "prefix_token_count_at_start": row["prefix_token_count_at_start"],
        "prefix_token_count_after_end": row["prefix_token_count_after_end"],
    }


def probe_seed_pointer(row: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return a complete-row seed receipt plus same-extent translation bounds."""

    if row is None:
        return None
    value = row_pointer(row)
    bins = row["coord_bins"]
    if bins is None:
        value.update(
            {
                "full_canvas_0_0_999_999": False,
                "same_extent_translation_possible": False,
                "translation_delta_bin_ranges": None,
            }
        )
        return value
    x1, y1, x2, y2 = bins
    dx = [-min(x1, x2), 999 - max(x1, x2)]
    dy = [-min(y1, y2), 999 - max(y1, y2)]
    value.update(
        {
            "full_canvas_0_0_999_999": bins == [0, 0, 999, 999],
            "same_extent_translation_possible": dx != [0, 0] or dy != [0, 0],
            "translation_delta_bin_ranges": {"dx_inclusive": dx, "dy_inclusive": dy},
            "translation_semantics": "add one dx to x1,x2 and one dy to y1,y2; endpoint differences and native valid/invalid ordering are preserved",
        }
    )
    return value


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first_valid = next((row for row in rows if row["geometry_valid"]), None)
    first_repeat = next((row for row in rows if row["strict_iou_gt_0_95_repeat"]), None)
    first_invalid = next((row for row in rows if not row["geometry_valid"]), None)
    repeat_description = first_repeat["description"] if first_repeat else None
    first_repeat_description_valid = next(
        (row for row in rows if row["geometry_valid"] and row["description"] == repeat_description),
        None,
    )

    run_start = None
    run_end = None
    if first_repeat is not None:
        position = first_repeat["raw_row_index"]
        start = position
        while start > 0 and rows[start - 1]["complete_canonical_row"] and rows[start - 1]["description"] == repeat_description:
            start -= 1
        end = position + 1
        while end < len(rows) and rows[end]["complete_canonical_row"] and rows[end]["description"] == repeat_description:
            end += 1
        run_start, run_end = rows[start], rows[end - 1]
        loop_rows = rows[start:end]
    else:
        loop_rows = []

    early_seed = None
    late_seed = None
    if first_repeat is not None:
        repeat_position = first_repeat["raw_row_index"]
        if repeat_position > 0 and rows[repeat_position - 1]["complete_canonical_row"]:
            early_seed = rows[repeat_position - 1]
        repeated_description_emissions = [
            row
            for row in rows[repeat_position:]
            if row["complete_canonical_row"] and row["description"] == repeat_description
        ]
        if repeated_description_emissions:
            late_seed = repeated_description_emissions[min(7, len(repeated_description_emissions) - 1)]

    intervention_boundaries = None
    if first_repeat is not None:
        intervention_boundaries = {
            "before_first_strict_repeat_row": {
                "prefix_token_count": first_repeat["token_start"],
                "next_raw_row_index": first_repeat["raw_row_index"],
                "semantics": "history ends immediately before the complete row opener",
            },
            "after_first_strict_repeat_row": {
                "prefix_token_count": first_repeat["token_end_exclusive"],
                "completed_raw_row_index": first_repeat["raw_row_index"],
                "semantics": "history includes the complete row through box_end",
            },
            "before_contiguous_repeat_description_run": {
                "prefix_token_count": run_start["token_start"],
                "next_raw_row_index": run_start["raw_row_index"],
                "semantics": "history ends immediately before the first complete same-description row in the run containing the first strict repeat",
            },
            "after_contiguous_repeat_description_run_onset_row": {
                "prefix_token_count": run_start["token_end_exclusive"],
                "completed_raw_row_index": run_start["raw_row_index"],
                "semantics": "history includes the run onset row through box_end",
            },
        }

    return {
        "first_valid_row": row_pointer(first_valid),
        "first_strict_iou_gt_0_95_repeat": row_pointer(first_repeat),
        "first_geometry_invalid_or_malformed_row": row_pointer(first_invalid),
        "repeat_description": repeat_description,
        "first_valid_instance_of_repeat_description": row_pointer(first_repeat_description_valid),
        "loop_run_definition": (
            "maximal contiguous sequence of complete canonical rows with the first strict-repeat row's "
            "description, containing that first strict-repeat row"
        ),
        "loop_run_onset": row_pointer(run_start),
        "loop_run_end": row_pointer(run_end),
        "loop_run_raw_row_count": len(loop_rows),
        "loop_run_valid_row_count": sum(row["geometry_valid"] for row in loop_rows),
        "loop_run_invalid_row_count": sum(not row["geometry_valid"] for row in loop_rows),
        "loop_run_strict_repeat_count": sum(row["strict_iou_gt_0_95_repeat"] for row in loop_rows),
        "probe_seed_candidates": {
            "early_last_complete_row_before_first_strict_repeat": probe_seed_pointer(early_seed),
            "late_last_complete_row_after_up_to_8_repeat_description_emissions": probe_seed_pointer(late_seed),
            "late_selection_emission_count": (
                min(8, len(repeated_description_emissions)) if first_repeat is not None else 0
            ),
            "late_selection_semantics": (
                "starting with the first strict-repeat row as emission 1, select emission 8; "
                "if fewer survive, select the last available complete same-description row"
            ),
        },
        "candidate_complete_row_intervention_boundaries": intervention_boundaries,
    }


def build_case(case: dict[str, Any], tokenizer: Tokenizer, inputs_path: Path, online_index: int) -> dict[str, Any]:
    action_ids = [int(value) for value in case["action_ids"]]
    action_text, token_spans = _exact_token_text_frame(action_ids, tokenizer)
    token_pieces = [tokenizer.decode([value], skip_special_tokens=False) for value in action_ids]
    layout = trajectory_layout(
        action_ids,
        tokenizer,
        image_width=int(case["image_width"]),
        image_height=int(case["image_height"]),
        row_id=str(case["example_id"]),
    )
    rows = build_raw_rows(
        case=case,
        layout=layout,
        action_text=action_text,
        token_spans=token_spans,
        token_pieces=token_pieces,
    )
    strict_count = sum(row["strict_iou_gt_0_95_repeat"] for row in rows)
    if strict_count != len(layout["duplicate_row_indices"]):
        raise ValueError("raw-row strict-repeat count differs from reusable trajectory layout")
    if sha256_json(action_ids) != case["action_ids_sha256"]:
        raise ValueError("embedded action token digest mismatch")
    if [row["raw_row_index"] for row in rows if row["strict_iou_gt_0_95_repeat"]] != layout["duplicate_row_indices"]:
        raise ValueError("raw-row strict-repeat indices differ from reusable trajectory layout")
    for key in (
        "action_token_count",
        "valid_row_count",
        "duplicate_row_indices",
        "parser_drops",
        "invalid_geometry_rows",
    ):
        if layout[key] != case["initial_layout"][key]:
            raise ValueError(f"fresh layout field {key} differs from frozen initial layout")

    image_id = int(case["image_id"])
    return {
        "image_id": image_id,
        "example_id": case["example_id"],
        "image_path": case["group"]["image_path"],
        "image_width": int(case["image_width"]),
        "image_height": int(case["image_height"]),
        "image_area_pixels": int(case["image_width"]) * int(case["image_height"]),
        "stop_reason": case["stop_reason"],
        "stable_action_source": {
            "path": str(inputs_path),
            "json_pointer": f"/online_cases/{online_index}/action_ids",
            "action_ids_sha256": case["action_ids_sha256"],
        },
        "action_token_count": len(action_ids),
        "action_token_ids": action_ids,
        "action_token_texts": token_pieces,
        "decoded_action_text": action_text,
        "decoded_action_text_sha256": sha256_text(action_text),
        "raw_object_start_count": len(rows),
        "complete_canonical_row_count": sum(row["complete_canonical_row"] for row in rows),
        "valid_row_count": sum(row["geometry_valid"] for row in rows),
        "invalid_or_malformed_row_count": sum(not row["geometry_valid"] for row in rows),
        "strict_class_blind_once_per_later_row_repeat_count": strict_count,
        "onset_summary": summarize(rows),
        "raw_rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    inputs = json.loads(args.inputs.read_text())
    tokenizer_path = Path(inputs["model"]["base_model_path"]) / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    online = inputs["online_cases"]
    observed_ids = [int(case["image_id"]) for case in online]
    if observed_ids != EXPECTED_IDS:
        raise ValueError(f"unexpected online case order: {observed_ids}")

    cases = [build_case(case, tokenizer, args.inputs, index) for index, case in enumerate(online)]
    observed_counts = [case["strict_class_blind_once_per_later_row_repeat_count"] for case in cases]
    if observed_counts != EXPECTED_STRICT_REPEATS:
        raise ValueError(f"strict-repeat baseline mismatch: {observed_counts}")

    base_config_root = Path(inputs["config"]["run"]["artifact_root"])
    base_config_run = base_config_root / inputs["config"]["run"]["name"]
    result = {
        "schema": "small_owner_repeat_origin.census.v1",
        "status": "complete_cpu_descriptive_census",
        "scope": {
            "image_ids": EXPECTED_IDS,
            "case_count": len(cases),
            "no_gt_use": True,
            "no_image_or_visual_truth_use": True,
            "no_model_or_gpu_calls": True,
            "claim_boundary": (
                "Frozen Stable50 natural token/row geometry only. Describes generated row onset, "
                "validity, strict class-blind recurrence, and endpoint drift; does not establish "
                "a visual false positive, hallucination, object identity, or causal mechanism."
            ),
        },
        "definitions": {
            "strict_repeat": "later geometry-valid row has native pixel IoU > 0.95 with any earlier geometry-valid row, class-blind; counted once per later row",
            "pixel_geometry": "round(coord_bin * native decoded image extent / 1000), matching src.data.geometry.coord_bins_to_pixel_xyxy for valid rows",
            "projected_pixel_xyxy_unvalidated": "same endpoint projection retained for canonical geometry-invalid rows; not a parser-valid box",
            "area_fraction_of_image": "max(0,x2-x1)*max(0,y2-y1)/(image_width*image_height), with no cross-image normalization",
            "prior_valid_max_iou": "maximum class-blind native pixel IoU against earlier geometry-valid rows; null when no prior valid row or current row invalid",
            "token_offsets": "zero-based half-open offsets into the unchanged embedded Stable50 action_token_ids",
        },
        "sources": {
            "stable50_inputs": {"path": str(args.inputs), "sha256": sha256_file(args.inputs)},
            "stable50_action_origin": "inputs.json /online_cases/*/action_ids (embedded frozen natural outputs)",
            "stable50_adapter_root": inputs["model"]["current_adapter"]["root"],
            "tokenizer": {"path": str(tokenizer_path), "sha256": sha256_file(tokenizer_path)},
            "base_config_artifact_run_resolved_from_inputs": {
                "path": str(base_config_run),
                "note": "source config/data run named by inputs; Stable50 actions themselves are the embedded online_cases values",
            },
            "reused_layout_helper": {
                "path": str(WORKTREE / "probes/dora_owner_learning/geometric_dedup.py"),
                "sha256_registered_in_inputs": inputs["source_files"][str(WORKTREE / "probes/dora_owner_learning/geometric_dedup.py")],
                "sha256_observed": sha256_file(WORKTREE / "probes/dora_owner_learning/geometric_dedup.py"),
            },
        },
        "acceptance": {
            "expected_image_ids": EXPECTED_IDS,
            "observed_image_ids": observed_ids,
            "expected_strict_repeat_counts": EXPECTED_STRICT_REPEATS,
            "observed_strict_repeat_counts": observed_counts,
            "strict_repeat_counts_match": observed_counts == EXPECTED_STRICT_REPEATS,
            "all_action_id_digests_match": True,
            "all_row_texts_reconstruct_from_original_token_slices": True,
            "all_fresh_layouts_match_frozen_initial_layout": True,
        },
        "cases": cases,
        "unknowns": [
            "The token/geometry census does not identify whether any row corresponds to a visible physical instance.",
            "It does not determine whether the first small box is an owner, noise, or a visual false positive.",
            "Temporal row recurrence and coordinate drift do not by themselves identify the model-internal causal state or the effect of a future intervention.",
            "Capped traces end with an incomplete row, so post-cap continuation behavior is unobserved.",
        ],
    }
    if result["sources"]["reused_layout_helper"]["sha256_registered_in_inputs"] != result["sources"]["reused_layout_helper"]["sha256_observed"]:
        raise ValueError("reused geometric_dedup.py differs from its inputs-registered identity")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    compact = {
        "schema": "small_owner_repeat_origin.census_summary.v1",
        "status": result["status"],
        "scope": result["scope"],
        "sources": result["sources"],
        "acceptance": result["acceptance"],
        "cases": [
            {
                "image_id": case["image_id"],
                "action_token_count": case["action_token_count"],
                "raw_object_start_count": case["raw_object_start_count"],
                "valid_row_count": case["valid_row_count"],
                "invalid_or_malformed_row_count": case["invalid_or_malformed_row_count"],
                "strict_class_blind_once_per_later_row_repeat_count": case[
                    "strict_class_blind_once_per_later_row_repeat_count"
                ],
                "onset_summary": case["onset_summary"],
            }
            for case in cases
        ],
        "unknowns": result["unknowns"],
        "full_census": {"path": str(args.out), "sha256": sha256_file(args.out)},
    }
    summary_path = args.out.with_name("summary.json")
    summary_path.write_text(json.dumps(compact, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
