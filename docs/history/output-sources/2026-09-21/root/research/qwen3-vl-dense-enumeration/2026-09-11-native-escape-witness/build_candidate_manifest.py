#!/usr/bin/env python3
"""Build and consume the bounded native-escape candidate manifest.

This is a CPU-only, source-bound packaging step.  It does not call a model,
inspect GT, or adjudicate a physical owner.  The only rows considered for
candidate selection are the first five complete rows in the frozen
``early_original_translated`` continuation for each requested case.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


SOURCE_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-11-small-owner-repeat-origin"
)
OUTPUT_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-11-native-escape-witness"
)
MANIFEST_PATH = OUTPUT_ROOT / "candidate_manifest.json"
FIGURE_ROOT = OUTPUT_ROOT / "candidate-visualizations"
TEST_ROOT = OUTPUT_ROOT / "tests"
PACKET_PATH = SOURCE_ROOT / "packet.json"
COHORT = ("351017", "417044", "477415", "502725")
SOURCE_JOB_ID = "early_original_translated"
H_JOB_ID = "early_original_native"
MAX_CANDIDATES = 2
FIRST_COMPLETE_LIMIT = 5
IOU_LIMIT = 0.95
WORKTREE = Path("/data/CoordExp/.worktrees/research-probes")
TOKENIZER_PATH = (
    Path("/data/Qwen3-VL/model_cache/models/Qwen/")
    / "Qwen3-VL-2B-Instruct-coordexp-natural-adjacent/tokenizer.json"
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_CANONICAL_ROW = re.compile(
    r"^<\|object_ref_start\|>.*?<\|object_ref_end\|>"
    r"<\|box_start\|>(?:<\|coord_(?:0|[1-9][0-9]{0,2})\|>){4}"
    r"<\|box_end\|>$",
    re.DOTALL,
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_ids(ids: Sequence[int]) -> str:
    return sha256_bytes(
        json.dumps(list(ids), separators=(",", ":"), ensure_ascii=False).encode()
    )


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_source_census() -> Any:
    """Load the frozen source row parser without importing the source as a package."""

    if str(WORKTREE) not in sys.path:
        sys.path.insert(0, str(WORKTREE))
    path = SOURCE_ROOT / "census" / "build_census.py"
    spec = importlib.util.spec_from_file_location("native_escape_source_census", path)
    require(spec is not None and spec.loader is not None, "cannot load source census parser")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_tokenizer() -> Any:
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(TOKENIZER_PATH))


def json_digest(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    )


def iou_xyxy(first: Sequence[int], second: Sequence[int]) -> float:
    """Class-blind endpoint IoU; zero-area boxes have zero overlap."""

    ax1, ay1, ax2, ay2 = (float(v) for v in first)
    bx1, by1, bx2, by2 = (float(v) for v in second)
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0.0 else intersection / union


def packet_case(packet: Mapping[str, Any], case_id: str) -> tuple[int, Mapping[str, Any]]:
    for index, case in enumerate(packet["cases"]):
        if case.get("case_id") == case_id:
            return index, case
    raise ValueError(f"frozen packet is missing case {case_id}")


def source_record_path(case_id: str) -> Path:
    """Resolve the one completed translated continuation for a frozen case."""

    candidates = [
        SOURCE_ROOT / "full" / f"rank-{index}" / "records.jsonl"
        for index in range(8)
    ] + [
        SOURCE_ROOT / "repair-01" / f"rank-{index}" / "records.jsonl"
        for index in range(8)
    ]
    matches: list[Path] = []
    for path in candidates:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("case_id") == case_id and row.get("job_id") == SOURCE_JOB_ID:
                matches.append(path)
                break
    require(len(matches) == 1, f"expected exactly one source record for {case_id}, got {matches}")
    return matches[0]


def source_record(path: Path, case_id: str) -> Mapping[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matches = [row for row in rows if row.get("case_id") == case_id and row.get("job_id") == SOURCE_JOB_ID]
    require(len(matches) == 1, f"source record {path} must contain one {case_id}/{SOURCE_JOB_ID}")
    return matches[0]


def parse_rows(ids: Sequence[int], case: Mapping[str, Any], tokenizer: Any, census: Any) -> list[dict[str, Any]]:
    """Use the frozen CPU parser/layout to preserve exact row token intervals."""

    from probes.dora_owner_learning.geometric_dedup import (
        _exact_token_text_frame,
        trajectory_layout,
    )

    values = list(ids)
    text, token_spans = _exact_token_text_frame(values, tokenizer)
    width = int(case["source_case"]["image_width"])
    height = int(case["source_case"]["image_height"])
    layout = trajectory_layout(
        values,
        tokenizer,
        image_width=width,
        image_height=height,
        row_id=str(case["case_id"]),
    )
    return census.build_raw_rows(
        case={"action_ids": values, "image_width": width, "image_height": height},
        layout=layout,
        action_text=text,
        token_spans=token_spans,
        token_pieces=[tokenizer.decode([token], skip_special_tokens=False) for token in values],
    )


def compact_row(row: Mapping[str, Any], *, include_tokens: bool = True) -> dict[str, Any]:
    """Retain parser provenance without copying GT or unrelated model fields."""

    result: dict[str, Any] = {
        "generated_order": int(row["raw_row_index"]),
        "raw_row_index": int(row["raw_row_index"]),
        "free_complete_ordinal": None,
        "object_span_id": row.get("object_span_id", ""),
        "native_status": row["native_status"],
        "geometry_valid": bool(row["geometry_valid"]),
        "complete_canonical_row": bool(row["complete_canonical_row"]),
        "token_start": int(row["token_start"]),
        "token_end_exclusive": int(row["token_end_exclusive"]),
        "token_length": int(row["token_end_exclusive"] - row["token_start"]),
        "description": row.get("description"),
        "raw_description": row.get("raw_description"),
        "coord_bins": row.get("coord_bins"),
        "bbox_pixel_xyxy": row.get("pixel_box_xyxy"),
        "projected_pixel_xyxy_unvalidated": row.get("projected_pixel_xyxy_unvalidated"),
        "raw_text": row["raw_text"],
        "raw_text_sha256": row["raw_text_sha256"],
    }
    if include_tokens:
        ids = list(row["token_ids"])
        texts = list(row["token_texts"])
        result.update(
            {
                "token_ids": ids,
                "token_texts": texts,
                "token_ids_sha256": sha256_ids(ids),
                "token_text_reconstructs_raw": "".join(texts) == row["raw_text"],
            }
        )
    return result


def image_envelope(case: Mapping[str, Any]) -> dict[str, Any]:
    source = case["source_case"]
    plan = source["image_plan"]
    path = Path(source["image_path"])
    require(path.is_file(), f"source image is missing: {path}")
    observed = sha256_file(path)
    require(observed == plan["image_content_sha256"], f"source image hash mismatch for {case['case_id']}")
    return {
        "row_id": source["row_id"],
        "row_index": int(source["row_index"]),
        "image_id": int(source["input_record"]["image_id"]),
        "image_path": str(path),
        "image_sha256": observed,
        "image_width": int(source["image_width"]),
        "image_height": int(source["image_height"]),
        "image_grid_thw": list(plan["observed_image_grid_thw"]),
        "merged_visual_tokens": int(plan["merged_visual_tokens"]),
        "logical_transform_id": plan["logical_transform_id"],
        "backend_prompt_token_count": int(plan["backend_prompt_token_count"]),
    }


def _h_case_envelope(case: Mapping[str, Any], tokenizer: Any, census: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    h_job = next(job for job in case["jobs"] if job["job_id"] == H_JOB_ID)
    h_ids = list(h_job["extension_ids"])
    h_rows = parse_rows(h_ids, case, tokenizer, census)
    h_text = tokenizer.decode(h_ids, skip_special_tokens=False)
    require(h_ids == list(case["baseline_action_ids"][: len(h_ids)]), f"{case['case_id']}: h is not baseline prefix")
    valid_rows = [row for row in h_rows if row["geometry_valid"]]
    h = {
        "job_id": H_JOB_ID,
        "boundary": "early",
        "image_condition": "original",
        "history_condition": "native",
        "ids": h_ids,
        "length": len(h_ids),
        "ids_sha256": sha256_ids(h_ids),
        "text": h_text,
        "text_sha256": sha256_text(h_text),
        "valid_row_count": len(valid_rows),
        "rows": [compact_row(row) for row in h_rows],
    }
    return h, valid_rows


def source_free_row_record(
    row: Mapping[str, Any], *, ordinal: int | None = None, selection_state: str | None = None,
    selection_reason: str | None = None, max_iou_to_h: float | None = None,
    max_iou_to_h_row_index: int | None = None, max_iou_to_prior_candidates: float | None = None,
) -> dict[str, Any]:
    result = compact_row(row)
    result["free_complete_ordinal"] = ordinal
    if selection_state is not None:
        result["selection_state"] = selection_state
    if selection_reason is not None:
        result["selection_reason"] = selection_reason
    if max_iou_to_h is not None:
        result["max_iou_to_h"] = float(max_iou_to_h)
        result["max_iou_to_h_row_index"] = max_iou_to_h_row_index
    if max_iou_to_prior_candidates is not None:
        result["max_iou_to_prior_candidates"] = float(max_iou_to_prior_candidates)
    return result


def record_free_rows(
    record: Mapping[str, Any], case: Mapping[str, Any], tokenizer: Any, census: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = parse_rows(record["action_ids"], case, tokenizer, census)
    prefix_length = len(record["extension_ids"])
    free_rows = [row for row in rows if int(row["token_start"]) >= prefix_length]
    complete = [row for row in free_rows if row["complete_canonical_row"]]
    first_five = complete[:FIRST_COMPLETE_LIMIT]
    invalid = [row for row in free_rows if not row["geometry_valid"]]
    incomplete = [row for row in free_rows if not row["complete_canonical_row"]]
    return first_five, invalid, incomplete


def choose_candidates(
    first_five: Sequence[Mapping[str, Any]], h_valid_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    held: list[dict[str, Any]] = []
    for ordinal, row in enumerate(first_five, start=1):
        box = row.get("pixel_box_xyxy")
        h_scores = [iou_xyxy(box, hrow["pixel_box_xyxy"]) for hrow in h_valid_rows] if box else []
        max_h = max(h_scores, default=0.0)
        max_h_index = None
        if h_scores:
            max_h_index = int(h_valid_rows[h_scores.index(max_h)]["raw_row_index"])
        prior_scores = [iou_xyxy(box, prior["bbox_pixel_xyxy"]) for prior in selected] if box else []
        max_prior = max(prior_scores, default=0.0)
        if not row["geometry_valid"]:
            state, reason = "held", f"geometry_invalid:{row['native_status']}"
        elif max_h > IOU_LIMIT:
            state, reason = "held", "class_blind_iou_to_h_gt_0.95"
        elif max_prior > IOU_LIMIT:
            state, reason = "held", "class_blind_iou_to_prior_candidate_gt_0.95"
        elif len(selected) >= MAX_CANDIDATES:
            state, reason = "held", "selection_cap_after_two"
        else:
            state, reason = "selected", None
        marked = source_free_row_record(
            row,
            ordinal=ordinal,
            selection_state=state,
            selection_reason=reason,
            max_iou_to_h=max_h,
            max_iou_to_h_row_index=max_h_index,
            max_iou_to_prior_candidates=max_prior,
        )
        if state == "selected":
            marked["candidate_index"] = len(selected) + 1
            selected.append(marked)
        else:
            held.append(marked)
    return selected, held


def candidate_from_marked(
    row: Mapping[str, Any], case_id: str, source_case_row_id: str
) -> dict[str, Any]:
    require(row["selection_state"] == "selected", f"{case_id}: candidate row is not selected")
    ids = list(row["token_ids"])
    return {
        "candidate_id": f"{case_id}-c{int(row['candidate_index']):02d}",
        "case_id": case_id,
        "source_job_id": SOURCE_JOB_ID,
        "source_image_condition": "original",
        "visual_admission": "candidate_pending_root_review",
        "source_row_id": row.get("object_span_id") or f"{case_id}:generated:{int(row['generated_order'])}",
        "source_case_row_id": source_case_row_id,
        "source_free_complete_ordinal": int(row["free_complete_ordinal"]),
        "source_free_complete_order": int(row["free_complete_ordinal"]) - 1,
        "source_generated_order": int(row["generated_order"]),
        "source_raw_row_index": int(row["raw_row_index"]),
        "description": row.get("description"),
        "geometry_valid": bool(row["geometry_valid"]),
        "coord_bins": row["coord_bins"],
        "bbox_pixel_xyxy": row["bbox_pixel_xyxy"],
        "max_iou_to_h": float(row["max_iou_to_h"]),
        "max_iou_to_h_row_index": row["max_iou_to_h_row_index"],
        "max_iou_to_prior_candidates": float(row["max_iou_to_prior_candidates"]),
        "max_class_blind_iou_to_h": float(row["max_iou_to_h"]),
        "max_class_blind_iou_to_prior_candidates": float(row["max_iou_to_prior_candidates"]),
        "c_ids": ids,
        "c_length": len(ids),
        "c_ids_sha256": sha256_ids(ids),
        "c_text": row["raw_text"],
        "c_text_sha256": row["raw_text_sha256"],
        "raw_span_text": row["raw_text"],
        "raw_span_sha256": row["raw_text_sha256"],
        "c_token_texts": list(row["token_texts"]),
        "c_token_text_reconstructs": "".join(row["token_texts"]) == row["raw_text"],
        "source_token_start": int(row["token_start"]),
        "source_token_end_exclusive": int(row["token_end_exclusive"]),
    }


def draw_candidate_figures(case: Mapping[str, Any], candidate: Mapping[str, Any], figure_root: Path) -> dict[str, Any]:
    from PIL import Image, ImageDraw, ImageFont

    image_path = Path(case["image"]["image_path"])
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    box = [int(value) for value in candidate["bbox_pixel_xyxy"]]
    require(box[2] > box[0] and box[3] > box[1], f"{candidate['candidate_id']}: figure box is invalid")
    font = ImageFont.load_default()
    color = (220, 30, 30)

    full = image.copy()
    draw = ImageDraw.Draw(full)
    draw.rectangle(box, outline=color, width=max(3, round(min(width, height) / 320)))
    label = candidate["candidate_id"]
    label_box = draw.textbbox((0, 0), label, font=font)
    label_width = label_box[2] - label_box[0] + 8
    label_height = label_box[3] - label_box[1] + 6
    label_x = max(0, min(box[0], width - label_width))
    label_y = max(0, box[1] - label_height)
    draw.rectangle((label_x, label_y, label_x + label_width, label_y + label_height), fill=color)
    draw.text((label_x + 4, label_y + 3), label, fill=(255, 255, 255), font=font)

    margin = max(24, round(max(box[2] - box[0], box[3] - box[1]) * 0.75))
    crop_box = (
        max(0, box[0] - margin),
        max(0, box[1] - margin),
        min(width, box[2] + margin),
        min(height, box[3] + margin),
    )
    crop = image.crop(crop_box)
    crop_draw = ImageDraw.Draw(crop)
    relative = [box[0] - crop_box[0], box[1] - crop_box[1], box[2] - crop_box[0], box[3] - crop_box[1]]
    crop_draw.rectangle(relative, outline=color, width=max(3, round(min(crop.size) / 180)))
    crop_label_y = max(0, relative[1] - label_height)
    crop_draw.rectangle((max(0, relative[0]), crop_label_y, max(0, relative[0]) + label_width, crop_label_y + label_height), fill=color)
    crop_draw.text((max(0, relative[0]) + 4, crop_label_y + 3), label, fill=(255, 255, 255), font=font)

    # Keep figures human-reviewable without changing the underlying source pixels.
    max_side = 1400
    if max(full.size) > max_side:
        scale = max_side / max(full.size)
        full = full.resize((round(full.width * scale), round(full.height * scale)), Image.Resampling.LANCZOS)
    crop_max = 900
    if max(crop.size) > crop_max:
        scale = crop_max / max(crop.size)
        crop = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.Resampling.LANCZOS)

    figure_root.mkdir(parents=True, exist_ok=True)
    stem = candidate["candidate_id"]
    full_path = figure_root / f"{stem}-full.png"
    crop_path = figure_root / f"{stem}-crop.png"
    full.save(full_path, format="PNG", optimize=False)
    crop.save(crop_path, format="PNG", optimize=False)
    return {
        "candidate_id": candidate["candidate_id"],
        "full_path": str(full_path),
        "full_sha256": sha256_file(full_path),
        "crop_path": str(crop_path),
        "crop_sha256": sha256_file(crop_path),
        "source_image_path": str(image_path),
        "source_image_sha256": case["image"]["image_sha256"],
        "annotation": "candidate geometry only; no GT or classification annotation",
    }


def build_manifest() -> dict[str, Any]:
    require(PACKET_PATH.is_file(), f"missing frozen packet: {PACKET_PATH}")
    packet = json.loads(PACKET_PATH.read_text(encoding="utf-8"))
    tokenizer = load_tokenizer()
    census = load_source_census()
    packet_sha = sha256_file(PACKET_PATH)
    cases: list[dict[str, Any]] = []
    all_figures: list[dict[str, Any]] = []

    for case_id in COHORT:
        case_index, case = packet_case(packet, case_id)
        source_path = source_record_path(case_id)
        record = source_record(source_path, case_id)
        require(record["job_id"] == SOURCE_JOB_ID, f"{case_id}: wrong source job")
        require(record["boundary"] == "early", f"{case_id}: source is not early")
        require(record["image_condition"] == "original", f"{case_id}: source is not original image")
        require(record["history_condition"] == "translated", f"{case_id}: source is not translated history")
        h, h_valid_rows = _h_case_envelope(case, tokenizer, census)
        source_image = image_envelope(case)
        source_path_sha = sha256_file(source_path)
        first_five, invalid_rows, incomplete_rows = record_free_rows(record, case, tokenizer, census)
        selected_marked, held_marked = choose_candidates(first_five, h_valid_rows)
        candidates = [
            candidate_from_marked(row, case_id, str(source_image["row_id"]))
            for row in selected_marked
        ]
        figures = [draw_candidate_figures({"image": source_image}, candidate, FIGURE_ROOT) for candidate in candidates]
        all_figures.extend(figures)

        # The source prefix is recorded separately from h: it is translated
        # context used only to identify where c came from, never the emitted h.
        translated_prefix = list(record["extension_ids"])
        translated_prefix_text = tokenizer.decode(translated_prefix, skip_special_tokens=False)
        source_complete_all = [
            row for row in parse_rows(record["action_ids"], case, tokenizer, census)
            if int(row["token_start"]) >= len(record["extension_ids"]) and row["complete_canonical_row"]
        ]
        marked_by_order = {
            int(row["raw_row_index"]): row for row in (selected_marked + held_marked)
        }
        case_entry = {
            "case_id": case_id,
            "packet_case_index": case_index,
            "source_case": source_image,
            "source_case_ref": {
                "packet_path": str(PACKET_PATH),
                "packet_sha256": packet_sha,
                "packet_case_index": case_index,
                "row_id": source_image["row_id"],
            },
            "prompt": {
                "token_ids": list(case["prompt_token_ids"]),
                "length": len(case["prompt_token_ids"]),
                "ids_sha256": sha256_ids(case["prompt_token_ids"]),
            },
            "prompt_token_ids": list(case["prompt_token_ids"]),
            "baseline": {
                "token_ids": list(case["baseline_action_ids"]),
                "length": len(case["baseline_action_ids"]),
                "ids_sha256": sha256_ids(case["baseline_action_ids"]),
            },
            "baseline_action_ids": list(case["baseline_action_ids"]),
            "h": h,
            "h_ids": list(h["ids"]),
            "h_ids_sha256": h["ids_sha256"],
            "h_source_job_id": H_JOB_ID,
            "h_complete_row_count": sum(bool(row["complete_canonical_row"]) for row in h["rows"]),
            "source_translated_record": {
                "path": str(source_path),
                "sha256": source_path_sha,
                "packet_path": str(PACKET_PATH),
                "packet_sha256": packet_sha,
                "case_id": case_id,
                "source_row_id": case["source_case"]["row_id"],
                "job_id": record["job_id"],
                "boundary": record["boundary"],
                "image_condition": record["image_condition"],
                "history_condition": record["history_condition"],
                "record_request_id": record.get("request_id"),
                "record_source_row_id": record.get("source_row_id"),
                "action_ids_length": len(record["action_ids"]),
                "action_ids_sha256": sha256_ids(record["action_ids"]),
                "action_text_sha256": sha256_text(record["text"]),
                "free_ids_length": len(record["free_ids"]),
                "free_ids_sha256": sha256_ids(record["free_ids"]),
                "translated_prefix_ids": translated_prefix,
                "translated_prefix_length": len(translated_prefix),
                "translated_prefix_ids_sha256": sha256_ids(translated_prefix),
                "translated_prefix_text_sha256": sha256_text(translated_prefix_text),
            },
            "source_free": {
                "selection_scope": "first five complete rows only; no rejection search or backfill",
                "raw_row_count": len([row for row in parse_rows(record["action_ids"], case, tokenizer, census) if int(row["token_start"]) >= len(record["extension_ids"])]),
                "complete_row_count": len(source_complete_all),
                "first_five_complete_count": len(first_five),
                "valid_row_count": sum(bool(row["geometry_valid"]) for row in [r for r in parse_rows(record["action_ids"], case, tokenizer, census) if int(r["token_start"]) >= len(record["extension_ids"])]),
                "invalid_row_count": len(invalid_rows),
                "incomplete_row_count": len(incomplete_rows),
                "first_five_complete_rows": [
                    marked_by_order[int(row["raw_row_index"])] for row in first_five
                ],
                "raw_invalid_rows": [source_free_row_record(row) for row in invalid_rows],
                "raw_incomplete_rows": [source_free_row_record(row) for row in incomplete_rows],
            },
            "candidates": candidates,
            "held_rows": held_marked,
            "figures": figures,
            "selection_counts": {
                "first_five_complete": len(first_five),
                "eligible_geometry_valid": sum(bool(row["geometry_valid"]) for row in first_five),
                "selected": len(candidates),
                "held": len(held_marked),
                "source_complete_shortfall": max(0, FIRST_COMPLETE_LIMIT - len(first_five)),
            },
        }
        cases.append(case_entry)

    selected_count = sum(len(case["candidates"]) for case in cases)
    manifest = {
        "schema": "native_escape_witness.candidate_manifest.v1",
        "status": "candidate_only",
        "admission": "not_visually_admitted; not_launchable; lead acceptance required",
        "claim_boundary": "CPU packaging only; no inference, GT classification, physical-owner adjudication, or mechanism claim",
        "source_root": str(SOURCE_ROOT),
        "source_packet": {"path": str(PACKET_PATH), "sha256": packet_sha},
        "frozen_cohort": list(COHORT),
        "selection_contract": {
            "source_job_id": SOURCE_JOB_ID,
            "source_boundary": "early",
            "source_image_condition": "original",
            "source_history_condition": "translated",
            "emitted_history_prefix_job_id": H_JOB_ID,
            "first_complete_rows_only": FIRST_COMPLETE_LIMIT,
            "max_candidates_per_case": MAX_CANDIDATES,
            "iou_threshold_inclusive": IOU_LIMIT,
            "iou_semantics": "class-blind pixel xyxy endpoint IoU; candidate must have IoU <= 0.95 to every valid h row and prior chosen candidate",
            "order": "frozen cohort order, then source raw generated order; no rejection search or backfill",
        },
        "cases": cases,
        "figures": all_figures,
        "counts": {
            "case_count": len(cases),
            "candidate_count": selected_count,
            "held_row_count": sum(len(case["held_rows"]) for case in cases),
            "source_invalid_row_count": sum(case["source_free"]["invalid_row_count"] for case in cases),
            "source_incomplete_row_count": sum(case["source_free"]["incomplete_row_count"] for case in cases),
            "source_shortfall_case_ids": [
                case["case_id"] for case in cases if case["selection_counts"]["source_complete_shortfall"]
            ],
            "hold_case_ids": [case["case_id"] for case in cases if not case["candidates"]],
        },
        "build_receipt": {
            "builder_path": str(Path(__file__).resolve()),
            "builder_sha256": sha256_file(Path(__file__).resolve()) if Path(__file__).is_file() else None,
            "no_model_calls": True,
            "no_gt_or_classification_use": True,
            "source_parser_path": str(SOURCE_ROOT / "census" / "build_census.py"),
            "source_parser_sha256": sha256_file(SOURCE_ROOT / "census" / "build_census.py"),
        },
    }
    return manifest


def _validate_hash(value: Any, field: str) -> None:
    require(isinstance(value, str) and _HEX64.fullmatch(value), f"{field} must be SHA-256")


def consume_candidate(manifest: Mapping[str, Any], case_id: str, candidate_id: str, tokenizer: Any | None = None) -> dict[str, Any]:
    """Validate the literal h+c contract used by a future native runner."""

    require(manifest.get("status") == "candidate_only", "manifest is not candidate-only")
    case = next((value for value in manifest["cases"] if value.get("case_id") == case_id), None)
    require(case is not None, f"unknown case {case_id}")
    candidate = next((value for value in case["candidates"] if value.get("candidate_id") == candidate_id), None)
    require(candidate is not None, f"unknown candidate {candidate_id}")
    h = case["h"]
    require(h["job_id"] == H_JOB_ID and h["history_condition"] == "native", "h identity is not native early")
    h_ids = list(h["ids"])
    c_ids = list(candidate["c_ids"])
    require(sha256_ids(h_ids) == h["ids_sha256"], "h token identity hash mismatch")
    require(sha256_ids(c_ids) == candidate["c_ids_sha256"], "c token identity hash mismatch")
    require(len(c_ids) == candidate["c_length"] == candidate["source_token_end_exclusive"] - candidate["source_token_start"], "c length mismatch")
    require(candidate["source_job_id"] == SOURCE_JOB_ID, "candidate did not come from early_original_translated")
    require(candidate["geometry_valid"] is True, "candidate geometry is not valid")
    if tokenizer is None:
        tokenizer = load_tokenizer()
    h_text = tokenizer.decode(h_ids, skip_special_tokens=False)
    c_text = tokenizer.decode(c_ids, skip_special_tokens=False)
    require(h_text == h["text"], "h literal text mismatch")
    require(c_text == candidate["c_text"], "c literal text mismatch")
    require(_CANONICAL_ROW.fullmatch(c_text) is not None, "c is not one complete canonical row")
    require(sha256_text(c_text) == candidate["c_text_sha256"], "c text hash mismatch")
    require("".join(candidate["c_token_texts"]) == c_text, "c token pieces do not reconstruct c")
    return {"case_id": case_id, "candidate_id": candidate_id, "h_ids": h_ids, "c_ids": c_ids, "action_ids": h_ids + c_ids, "h_text": h_text, "c_text": c_text}


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    require(manifest.get("schema") == "native_escape_witness.candidate_manifest.v1", "wrong manifest schema")
    require(manifest.get("frozen_cohort") == list(COHORT), "cohort order changed")
    require(manifest.get("selection_contract", {}).get("source_job_id") == SOURCE_JOB_ID, "source job contract changed")
    require(len(manifest.get("cases", [])) == len(COHORT), "case count changed")
    for case_id in COHORT:
        case = next(case for case in manifest["cases"] if case["case_id"] == case_id)
        require(case["source_translated_record"]["job_id"] == SOURCE_JOB_ID, f"{case_id}: source job changed")
        require(case["source_translated_record"]["history_condition"] == "translated", f"{case_id}: source history changed")
        require(case["h"]["job_id"] == H_JOB_ID, f"{case_id}: h job changed")
        require(len(case["candidates"]) <= MAX_CANDIDATES, f"{case_id}: candidate cap exceeded")
        previous = 0
        for candidate in case["candidates"]:
            require(candidate["source_free_complete_ordinal"] > previous, f"{case_id}: candidate order changed")
            previous = candidate["source_free_complete_ordinal"]
            consume_candidate(manifest, case_id, candidate["candidate_id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args()
    manifest = build_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
