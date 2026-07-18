#!/usr/bin/env python3
"""Historical random-versus-sorted full-row likelihood screen for image 2299.

This is deliberately a one-time research script.  It reconstructs the June
2026 compact-row runtime exactly enough to score one frozen, hand-relabelled
image under the two matched pure cross-entropy adapters.  It is not a general
inference entry point and it must not be used to compare current checkpoints
with the historical adapters.

The historical row has no newline or closing marker::

    <|object_ref_start|>person<|box_start|><|coord_x1|>...

``score-arm`` loads one adapter and writes one deterministic JSON arm.  The
``merge`` command combines the eight arms (two adapters and four person-only
owner ranks) without loading a model.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import types
from typing import Any


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


SCRIPT_SCHEMA_VERSION = "historical_random_sorted_image2299_screen.v1"
EXPECTED_PROMPT_HASH = "6c03d11f37f6ebd2676d6eae2a1a3329534243b512bbfdaed0bacbd5c4d6cca2"
EXPECTED_IMAGE_ID = "2299"
EXPECTED_PERSON_COUNT = 38
PARENT_PERSON_RANKS = (0, 1)
OWNER_PERSON_RANKS = (2, 3, 4, 14)
# Stable identities frozen from the 2026-07-18 human-relabelled packet.
EXPECTED_OWNER_BOXES = {
    2: ([625, 86, 711, 356], -2, 2),
    3: ([331, 87, 423, 369], -23, 3),
    4: ([708, 91, 784, 368], -9, 4),
    14: ([639, 266, 734, 514], -6, 19),
}
TERMINAL_TOKEN = "<|im_end|>"
LEGACY_COORD_SOURCE_REVISION = "834b2e0e4^"
LEGACY_COORD_SOURCE_PATH = "src/tokens/row_offsets.py"
DEFAULT_BASE_MODEL = Path(
    "/data/CoordExp/model_cache_remote/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp"
)
DEFAULT_JSONL = Path(
    "/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/val.coord.jsonl"
)
DEFAULT_IMAGE = Path(
    "/data/CoordExp/public_data/coco/rescale_32_1024_bbox/images/val2017/000000002299.jpg"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-18-historical-random-versus-geometry-sorted-image2299-screen"
)
DEFAULT_CHECKPOINTS = {
    "random": Path(
        "/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/"
        "compact_full_fullobj_random_sft_bsz16_4epoch_tokenrows_v2/"
        "compact-full-fullobj-random-sft-bsz16-4epoch-tokenrows-v2/"
        "v1-20260601-062428/checkpoint-3668"
    ),
    "sorted": Path(
        "/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/"
        "compact_full_fullobj_sorted_sft_bsz16_4epoch_tokenrows_v2/"
        "compact-full-fullobj-sorted-sft-bsz16-4epoch-tokenrows-v2/"
        "v1-20260601-062429/checkpoint-3668"
    ),
}
EXPECTED_IMAGE_SHA256 = "cd7199a37188c9ac6481520175866cbd78fa6fba35f4290bb0b60c78afcb2df3"
EXPECTED_RECORD_SHA256 = "ce19853c74a595f22cc183ce450e561f2da3216e54a1e499cfbca1be7e1c425b"

# One-time paired-sampling screen.  The same explicit seeds are used for the
# random-order and geometry-sorted adapters.  Do not silently replace this
# with a process-dependent random seed: the paired comparison is the point of
# this arm.
SAMPLING_SEEDS = tuple(range(24))
SAMPLING_TEMPERATURE = 0.4
SAMPLING_TOP_P = 0.95
SAMPLING_REPETITION_PENALTY = 1.0
SAMPLING_MAX_NEW_TOKENS = 9
MATCH_IOU_THRESHOLD = 0.5
MATCH_AMBIGUITY_MARGIN = 0.05

# This is the historical COCO-80 order from git revision a7ced2708.  The
# compact prompt builder joined this tuple with ``, ``; keeping the literal
# tuple here makes the hash independent of the current source tree.
COCO_80_CLASS_NAMES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def coord_token(value: int) -> str:
    value = int(value)
    if not 0 <= value <= 999:
        raise ValueError(f"coordinate token must be in [0,999], got {value}")
    return f"<|coord_{value}|>"


def render_legacy_row(description: str, bbox: Sequence[int]) -> str:
    """Render exactly one old compact row, without separators or end markers."""

    if not isinstance(description, str) or not description:
        raise ValueError("description must be non-empty")
    if len(bbox) != 4:
        raise ValueError("bbox must contain four coordinates")
    return (
        "<|object_ref_start|>"
        + description
        + "<|box_start|>"
        + "".join(coord_token(int(value)) for value in bbox)
    )


def parse_legacy_generation(text: str) -> dict[str, Any]:
    """Parse exactly one generated historical row, terminal, or malformed text.

    This parser intentionally accepts only the old compact row.  It does not
    attempt to repair whitespace, missing coordinate tokens, extra rows, or
    free-form explanations: those cases are evidence about the sampled
    rollout and are therefore classified as ``malformed``.
    """

    if not isinstance(text, str):
        raise TypeError("generated text must be a string")
    normalized = text.strip()
    if normalized == TERMINAL_TOKEN:
        return {"status": "terminal", "text": text}
    classes = "|".join(re.escape(name) for name in sorted(COCO_80_CLASS_NAMES, key=len, reverse=True))
    pattern = re.compile(
        r"^<\|object_ref_start\|>(?P<desc>" + classes + r")<\|box_start\|>"
        r"<\|coord_(?P<x1>\d+)\|><\|coord_(?P<y1>\d+)\|>"
        r"<\|coord_(?P<x2>\d+)\|><\|coord_(?P<y2>\d+)\|>"
        r"(?P<terminal><\|im_end\|>)?$"
    )
    match = pattern.fullmatch(normalized)
    if match is None:
        return {"status": "malformed", "text": text, "reason": "not_exactly_one_legacy_row"}
    bbox = [int(match.group(slot)) for slot in ("x1", "y1", "x2", "y2")]
    if any(value < 0 or value > 999 for value in bbox):
        return {"status": "malformed", "text": text, "reason": "coordinate_out_of_range"}
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return {"status": "malformed", "text": text, "reason": "non_positive_box_extent"}
    return {
        "status": "row",
        "text": text,
        "description": match.group("desc"),
        "bbox": bbox,
        "terminal_after_row": match.group("terminal") is not None,
    }


def legacy_sampling_suffix_is_complete(
    generated_token_ids: Sequence[int],
    *,
    object_ref_start_token_id: int,
    box_start_token_id: int,
    coordinate_token_ids: set[int],
    terminal_token_id: int,
) -> bool:
    """Return whether a sampled suffix has a terminal or complete row.

    Descriptions are not assumed to be one token (``traffic light`` is a
    useful counterexample), so the stopping rule waits for the structural
    marker followed by four coordinate tokens rather than a fixed token count.
    ``max_new_tokens=9`` remains a safety horizon for malformed generations.
    """

    ids = [int(value) for value in generated_token_ids]
    if int(terminal_token_id) in ids:
        return True
    start_positions = [index for index, value in enumerate(ids) if value == int(object_ref_start_token_id)]
    for start in start_positions:
        box_positions = [index for index in range(start + 1, len(ids)) if ids[index] == int(box_start_token_id)]
        for box in box_positions:
            coordinates = ids[box + 1 : box + 5]
            if len(coordinates) == 4 and all(value in coordinate_token_ids for value in coordinates):
                return True
    return False


def bbox_iou(bbox_a: Sequence[int | float], bbox_b: Sequence[int | float]) -> float:
    """Return inclusive-coordinate-independent xyxy intersection-over-union."""

    if len(bbox_a) != 4 or len(bbox_b) != 4:
        raise ValueError("both bounding boxes must contain four coordinates")
    ax1, ay1, ax2, ay2 = (float(value) for value in bbox_a)
    bx1, by1, bx2, by2 = (float(value) for value in bbox_b)
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0.0 else intersection / union


def match_parsed_row_to_candidates(
    parsed: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float = MATCH_IOU_THRESHOLD,
    ambiguity_margin: float = MATCH_AMBIGUITY_MARGIN,
) -> dict[str, Any]:
    """Match one parsed row to frozen objects using exact category then IoU.

    A high-overlap tie between same-category physical objects is reported as
    ``ambiguous`` rather than being forced into an arbitrary instance.  This
    matters in the dense person scene: the experiment measures discovery, but
    must not invent an identity when geometry cannot distinguish two people.
    """

    status = str(parsed.get("status"))
    if status != "row":
        return {"status": status, "reason": str(parsed.get("reason", status))}
    description = str(parsed.get("description"))
    bbox = parsed.get("bbox")
    if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)) or len(bbox) != 4:
        return {"status": "unmatched", "reason": "parsed_bbox_missing"}
    all_scored = [
        {
            "global_object_rank": int(item["global_object_rank"]),
            "person_only_rank": item.get("person_only_rank"),
            "desc": str(item.get("desc")),
            "coco_ann_id": item.get("coco_ann_id"),
            "iou": bbox_iou(bbox, item["bbox"]),
            "category_exact": str(item.get("desc")) == description,
        }
        for item in candidates
    ]
    all_scored.sort(key=lambda item: (-float(item["iou"]), int(item["global_object_rank"])))
    scored = [item for item in all_scored if bool(item["category_exact"])]
    if not scored:
        return {
            "status": "unmatched",
            "reason": "no_exact_coco_category",
            "description": description,
            "best_iou": float(all_scored[0]["iou"]) if all_scored else 0.0,
            "candidate_rankings": all_scored,
        }
    best_iou = float(scored[0]["iou"])
    result: dict[str, Any] = {
        "description": description,
        "parsed_bbox": [int(value) for value in bbox],
        "best_iou": best_iou,
        "category_exact": True,
        "all_candidate_rankings": all_scored,
        "candidate_rankings": scored,
    }
    if best_iou < float(iou_threshold):
        result.update({"status": "unmatched", "reason": "best_iou_below_threshold"})
        return result
    near = [item for item in scored if best_iou - float(item["iou"]) <= float(ambiguity_margin)]
    if len(near) > 1:
        result.update({
            "status": "ambiguous",
            "reason": "multiple_same_category_objects_within_iou_margin",
            "ambiguous_global_object_ranks": [int(item["global_object_rank"]) for item in near],
            "ambiguous_person_only_ranks": [item.get("person_only_rank") for item in near],
        })
        return result
    winner = scored[0]
    result.update({
        "status": "matched",
        "matched_global_object_rank": int(winner["global_object_rank"]),
        "matched_person_only_rank": winner.get("person_only_rank"),
        "matched_coco_ann_id": winner.get("coco_ann_id"),
    })
    return result


def compact_row_token_labels() -> tuple[str, ...]:
    return (
        "object_ref_start",
        "description",
        "box_start",
        "x1",
        "y1",
        "x2",
        "y2",
    )


def build_historical_prompts(*, ordering: str = "sorted") -> dict[str, Any]:
    """Build the exact a7ced2708 compact no-separator prompt and its hash."""

    if ordering not in {"sorted", "random"}:
        raise ValueError("ordering must be sorted or random")
    pattern = (
        "<|object_ref_start|>{desc}<|box_start|>"
        "<|coord_x1|><|coord_y1|><|coord_x2|><|coord_y2|>"
    )
    system = (
        "You are a general-purpose object detection and grounding assistant. "
        "Output one compact detection row per object by concatenating rows directly "
        "with no separator, no newline, and no extra text. "
        f"Use this row pattern exactly: {pattern}. "
        "Descriptions are raw class text; bbox coords are four coord tokens in x1 y1 x2 y2 order."
    )
    user = (
        "Locate each clearly visible object instance in the image. "
        f"Return compact rows using this exact pattern: {pattern}. "
        "Use one row per object; concatenate rows directly with no separator and do not insert newline characters. "
        "Restrict `desc` to this COCO-80 class list: "
        + ", ".join(COCO_80_CLASS_NAMES)
        + "."
    )
    payload = {
        "ordering": ordering,
        "coord_mode": "coord_tokens",
        "prompt_variant": "coco_80",
        "object_field_order": "desc_first",
        "bbox_format": "xyxy",
        "detection_sequence_format": "compact_full",
        "row_separator": "none",
        "system_prompt": system,
        "user_prompt": user,
        "do_resize": False,
    }
    prompt_hash = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {"ordering": ordering, "system": system, "user": user, "hash": prompt_hash, "payload": payload}


def verify_historical_prompt(ordering: str = "sorted") -> dict[str, Any]:
    prompt = build_historical_prompts(ordering=ordering)
    if prompt["hash"] != EXPECTED_PROMPT_HASH:
        raise ValueError(
            f"historical prompt hash mismatch: expected {EXPECTED_PROMPT_HASH}, got {prompt['hash']}"
        )
    return prompt


def _coord_value(token: str) -> int:
    if not token.startswith("<|coord_") or not token.endswith("|>"):
        raise ValueError(f"not a coordinate token: {token!r}")
    return int(token[len("<|coord_") : -2])


def _global_geometry_key(obj: Mapping[str, Any]) -> tuple[int, int]:
    bbox = obj["bbox_2d"]
    return (_coord_value(str(bbox[1])), _coord_value(str(bbox[0])))


def load_image_record(jsonl_path: Path, image_path: Path = DEFAULT_IMAGE) -> tuple[int, dict[str, Any], str]:
    """Return 1-based source line index and the record containing image 2299."""

    expected_name = image_path.name
    found: list[tuple[int, dict[str, Any], str]] = []
    with jsonl_path.open("rb") as handle:
        for line_index, raw_line in enumerate(handle, start=1):
            line = raw_line.decode("utf-8")
            record = json.loads(line)
            images = record.get("images")
            if isinstance(images, list) and any(str(path).endswith(expected_name) for path in images):
                found.append((line_index, record, hashlib.sha256(raw_line).hexdigest()))
    if len(found) != 1:
        raise ValueError(f"expected one record ending with {expected_name}, found {len(found)}")
    line_index, record, record_sha256 = found[0]
    return line_index, record, record_sha256


def geometry_sorted_objects(record: Mapping[str, Any], *, require_expected_person_count: bool = True) -> list[dict[str, Any]]:
    raw = record.get("objects")
    if not isinstance(raw, list):
        raise ValueError("record.objects must be a list")
    objects = [dict(obj) for obj in raw if isinstance(obj, Mapping)]
    objects.sort(key=_global_geometry_key)
    for global_rank, obj in enumerate(objects):
        obj["global_object_rank"] = global_rank
    persons = [obj for obj in objects if str(obj.get("desc")) == "person"]
    if require_expected_person_count and len(persons) != EXPECTED_PERSON_COUNT:
        raise ValueError(f"expected {EXPECTED_PERSON_COUNT} persons, found {len(persons)}")
    for person_rank, obj in enumerate(persons):
        obj["person_only_rank"] = person_rank
    category_counts: dict[str, int] = {}
    for obj in objects:
        desc = str(obj["desc"])
        obj["category_rank"] = category_counts.get(desc, 0)
        category_counts[desc] = int(obj["category_rank"]) + 1
    return objects


def select_image2299_rows(jsonl_path: Path = DEFAULT_JSONL, image_path: Path = DEFAULT_IMAGE) -> dict[str, Any]:
    line_index, record, record_sha256 = load_image_record(jsonl_path, image_path)
    objects = geometry_sorted_objects(record)
    by_global = {int(obj["global_object_rank"]): obj for obj in objects}
    people = [obj for obj in objects if obj.get("desc") == "person"]
    for rank in PARENT_PERSON_RANKS + OWNER_PERSON_RANKS:
        obj = people[rank]
        if obj.get("desc") != "person":
            raise ValueError(f"person rank {rank} is not person: {obj.get('desc')!r}")
    for person_rank, (expected_bbox, expected_ann_id, expected_global) in EXPECTED_OWNER_BOXES.items():
        obj = people[person_rank]
        actual_bbox = [_coord_value(str(token)) for token in obj["bbox_2d"]]
        if actual_bbox != expected_bbox or obj.get("coco_ann_id") != expected_ann_id or int(obj["global_object_rank"]) != expected_global:
            raise ValueError(
                f"image2299 owner identity mismatch for person rank {person_rank}: "
                f"bbox={actual_bbox}, ann={obj.get('coco_ann_id')}, global={obj.get('global_object_rank')}"
            )
    image_sha256 = sha256_file(image_path)
    if image_sha256 != EXPECTED_IMAGE_SHA256:
        raise ValueError(f"image-2299 checksum mismatch: expected {EXPECTED_IMAGE_SHA256}, got {image_sha256}")
    if record_sha256 != EXPECTED_RECORD_SHA256:
        raise ValueError(f"image-2299 JSONL record checksum mismatch: expected {EXPECTED_RECORD_SHA256}, got {record_sha256}")
    return {
        "source_line_index": line_index,
        "record": record,
        "objects": objects,
        "people": people,
        "by_global_rank": by_global,
        "by_person_rank": {int(obj["person_only_rank"]): obj for obj in people},
        "jsonl_sha256": sha256_file(jsonl_path),
        "record_sha256": record_sha256,
        "image_sha256": image_sha256,
    }


def candidate_rows(packet: Mapping[str, Any], owner_person_rank: int) -> list[dict[str, Any]]:
    by_person = packet["by_person_rank"]
    if owner_person_rank not in OWNER_PERSON_RANKS:
        raise ValueError(f"owner_person_rank must be one of {OWNER_PERSON_RANKS}")
    prefix_objects = [by_person[rank] for rank in PARENT_PERSON_RANKS] + [by_person[owner_person_rank]]
    for obj in prefix_objects:
        if obj.get("desc") != "person":
            raise ValueError("all forced prefix objects must be person")
    result: list[dict[str, Any]] = []
    emitted_global = {int(obj["global_object_rank"]) for obj in prefix_objects}
    for obj in packet["objects"]:
        bbox = [_coord_value(str(token)) for token in obj["bbox_2d"]]
        global_rank = int(obj["global_object_rank"])
        result.append({
            "global_object_rank": global_rank,
            "person_only_rank": obj.get("person_only_rank"),
            "category_rank": int(obj["category_rank"]),
            "desc": str(obj["desc"]),
            "category_id": obj.get("category_id"),
            "coco_ann_id": obj.get("coco_ann_id"),
            "bbox": bbox,
            "row_text": render_legacy_row(str(obj["desc"]), bbox),
            "emitted": global_rank in emitted_global,
            "uncovered": global_rank not in emitted_global,
            "emitted_person": str(obj["desc"]) == "person" and global_rank in emitted_global,
            "uncovered_person": str(obj["desc"]) == "person" and global_rank not in emitted_global,
        })
    return result


def append_prefix_rows(packet: Mapping[str, Any], owner_person_rank: int) -> tuple[list[int], list[str]]:
    by_person = packet["by_person_rank"]
    rows: list[str] = []
    for rank in PARENT_PERSON_RANKS + (owner_person_rank,):
        obj = by_person[rank]
        bbox = [_coord_value(str(token)) for token in obj["bbox_2d"]]
        rows.append(render_legacy_row(str(obj["desc"]), bbox))
    return [], rows


def _tokenize_row(tokenizer: Any, row_text: str) -> list[int]:
    ids = tokenizer(row_text, add_special_tokens=False, return_tensors=None)["input_ids"]
    if not isinstance(ids, list) or not ids:
        raise ValueError(f"tokenizer returned invalid row ids for {row_text!r}")
    return [int(v) for v in ids]


def score_token_sequence(logits: Any, *, boundary_length: int, row_token_ids: Sequence[int]) -> dict[str, Any]:
    """Decompose next-token log probabilities for one complete row.

    ``logits[j]`` predicts token ``j+1``.  The returned slots are always
    float64 Python values, even though model computation is float32.
    """

    import torch

    logits_tensor = logits if isinstance(logits, torch.Tensor) else torch.as_tensor(logits)
    if logits_tensor.ndim != 2:
        raise ValueError("logits must have shape [sequence_length, vocabulary]")
    tokens = [int(v) for v in row_token_ids]
    if boundary_length < 1 or boundary_length + len(tokens) > logits_tensor.shape[0]:
        raise ValueError("boundary_length and row token length exceed logits")
    log_probs = torch.log_softmax(logits_tensor.to(dtype=torch.float32), dim=-1)
    values = [float(log_probs[boundary_length + index - 1, token].item()) for index, token in enumerate(tokens)]
    labels = list(compact_row_token_labels())
    if len(values) != len(labels):
        raise ValueError("historical row must tokenize to seven semantic tokens")
    by_label = {label: value for label, value in zip(labels, values)}
    structural = [by_label["object_ref_start"], by_label["box_start"]]
    coordinates = [by_label[key] for key in ("x1", "y1", "x2", "y2")]
    total = math.fsum(values)
    return {
        "token_count": len(tokens),
        "token_ids": tokens,
        "token_log_probs": values,
        "total_log_probability": total,
        "mean_log_probability": total / len(values),
        "structural_log_probability": math.fsum(structural),
        "structural_mean_log_probability": math.fsum(structural) / len(structural),
        "description_log_probability": by_label["description"],
        "coordinate_log_probabilities": {key: by_label[key] for key in ("x1", "y1", "x2", "y2")},
        "coordinate_log_probability": math.fsum(coordinates),
    }


def terminal_score(logits: Any, *, boundary_length: int, terminal_token_id: int, object_ref_start_token_id: int | None = None) -> dict[str, Any]:
    import torch

    tensor = logits if isinstance(logits, torch.Tensor) else torch.as_tensor(logits)
    if tensor.ndim != 2 or boundary_length < 1 or boundary_length > tensor.shape[0]:
        raise ValueError("invalid logits or boundary_length for terminal score")
    log_probs = torch.log_softmax(tensor.to(dtype=torch.float32), dim=-1)
    value = float(log_probs[boundary_length - 1, int(terminal_token_id)].item())
    result = {"token_id": int(terminal_token_id), "token_log_probability": value}
    if object_ref_start_token_id is not None:
        result["object_ref_start_token_id"] = int(object_ref_start_token_id)
        result["object_ref_start_log_probability"] = float(
            log_probs[boundary_length - 1, int(object_ref_start_token_id)].item()
        )
        result["continue_minus_terminal_margin"] = result["object_ref_start_log_probability"] - value
    return result


def _rankdata(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (float(values[index]), index))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and float(values[order[end]]) == float(values[order[cursor]]):
            end += 1
        rank = (cursor + end - 1) / 2.0 + 1.0
        for pos in order[cursor:end]:
            ranks[pos] = rank
        cursor = end
    return ranks


def pearson(values_a: Sequence[float], values_b: Sequence[float]) -> float | None:
    if len(values_a) != len(values_b) or not values_a:
        raise ValueError("correlation vectors must have equal non-zero lengths")
    mean_a = sum(float(v) for v in values_a) / len(values_a)
    mean_b = sum(float(v) for v in values_b) / len(values_b)
    da = [float(v) - mean_a for v in values_a]
    db = [float(v) - mean_b for v in values_b]
    den = math.sqrt(math.fsum(v * v for v in da) * math.fsum(v * v for v in db))
    return None if den == 0 else math.fsum(x * y for x, y in zip(da, db)) / den


def spearman(values_a: Sequence[float], values_b: Sequence[float]) -> float | None:
    return pearson(_rankdata(values_a), _rankdata(values_b))


def merge_arm_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(records) != 8:
        raise ValueError(f"merge expects exactly eight arm records, got {len(records)}")
    index: dict[tuple[str, int], Mapping[str, Any]] = {}
    for record in records:
        adapter = str(record.get("adapter_role"))
        owner = int(record.get("owner_person_rank", record.get("owner_global_rank")))
        key = (adapter, owner)
        if adapter not in {"random", "sorted"} or owner not in OWNER_PERSON_RANKS or key in index:
            raise ValueError(f"invalid or duplicate arm key: {key}")
        index[key] = record
    merged: dict[str, Any] = {"schema_version": SCRIPT_SCHEMA_VERSION, "owners": {}, "correlations": {}}
    for owner in OWNER_PERSON_RANKS:
        random_record = index[("random", owner)]
        sorted_record = index[("sorted", owner)]
        random_candidates = {int(item["global_object_rank"]): item for item in random_record["candidate_scores"]}
        sorted_candidates = {int(item["global_object_rank"]): item for item in sorted_record["candidate_scores"]}
        if set(random_candidates) != set(sorted_candidates):
            raise ValueError(f"candidate rank set differs for owner {owner}")
        ranks = sorted(random_candidates)
        target_global_rank = next(
            rank
            for rank, item in random_candidates.items()
            if item.get("person_only_rank") is not None
            and int(item["person_only_rank"]) == owner
        )
        random_values = [float(random_candidates[rank]["score"]["total_log_probability"]) for rank in ranks]
        sorted_values = [float(sorted_candidates[rank]["score"]["total_log_probability"]) for rank in ranks]
        deltas: dict[str, Any] = {}
        for rank in ranks:
            rc = random_candidates[rank]
            sc = sorted_candidates[rank]
            components = [
                "total_log_probability", "mean_log_probability", "structural_log_probability",
                "description_log_probability", "coordinate_log_probability",
            ]
            components.extend(f"coordinate_{key}_log_probability" for key in ("x1", "y1", "x2", "y2"))
            delta: dict[str, float] = {}
            for component in components:
                if component.startswith("coordinate_") and component.endswith("_log_probability") and component not in {"coordinate_log_probability"}:
                    slot = component[len("coordinate_") : -len("_log_probability")]
                    a = float(sc["score"]["coordinate_log_probabilities"][slot])
                    b = float(rc["score"]["coordinate_log_probabilities"][slot])
                else:
                    a = float(sc["score"][component])
                    b = float(rc["score"][component])
                delta[component] = a - b
            deltas[str(rank)] = {"global_object_rank": rank, **delta}
        def rank_of(values: Mapping[int, Mapping[str, Any]], target: int) -> int:
            ordered = sorted(values, key=lambda key: float(values[key]["score"]["total_log_probability"]), reverse=True)
            return ordered.index(target) + 1
        def logsum(values: Sequence[float]) -> float:
            import torch
            return float(torch.logsumexp(torch.tensor(list(values), dtype=torch.float64), dim=0).item())
        random_terminal = float(random_record["terminal_score"]["token_log_probability"])
        sorted_terminal = float(sorted_record["terminal_score"]["token_log_probability"])
        random_object_start = float(random_record["terminal_score"]["object_ref_start_log_probability"])
        sorted_object_start = float(sorted_record["terminal_score"]["object_ref_start_log_probability"])
        emitted = [rank for rank in ranks if bool(random_candidates[rank].get("emitted"))]
        uncovered = [rank for rank in ranks if bool(random_candidates[rank].get("uncovered"))]
        emitted_person = [rank for rank in ranks if bool(random_candidates[rank].get("emitted_person"))]
        uncovered_person = [rank for rank in ranks if bool(random_candidates[rank].get("uncovered_person"))]
        tie_rows = [rank for rank in ranks if str(random_candidates[rank].get("desc")) == "tie"]
        def entropy(values: Sequence[float]) -> float:
            import torch
            x = torch.tensor(list(values), dtype=torch.float64)
            p = torch.softmax(x, dim=0)
            return float((-(p * torch.log(p)).sum()).item())
        def mass(values: Mapping[int, Mapping[str, Any]], selected: Sequence[int]) -> float:
            return logsum([float(values[r]["score"]["total_log_probability"]) for r in selected]) if selected else float("-inf")
        def probability(log_mass: float, log_denominator: float) -> float:
            if not math.isfinite(log_mass) or not math.isfinite(log_denominator):
                return 0.0
            return float(math.exp(min(0.0, log_mass - log_denominator)))
        def restricted_probabilities(values: Mapping[int, Mapping[str, Any]]) -> dict[str, float]:
            all_mass = mass(values, ranks)
            person_mass = mass(values, [r for r in ranks if str(values[r].get("desc")) == "person"])
            tie_mass = mass(values, [r for r in ranks if str(values[r].get("desc")) == "tie"])
            emitted_local = [r for r in ranks if bool(values[r].get("emitted"))]
            uncovered_local = [r for r in ranks if bool(values[r].get("uncovered"))]
            emitted_person_local = [r for r in ranks if bool(values[r].get("emitted_person"))]
            uncovered_person_local = [r for r in ranks if bool(values[r].get("uncovered_person"))]
            emitted_mass = mass(values, emitted_local)
            uncovered_mass = mass(values, uncovered_local)
            emitted_person_mass = mass(values, emitted_person_local)
            uncovered_person_mass = mass(values, uncovered_person_local)
            return {
                "person_probability_over_all_46": probability(person_mass, all_mass),
                "tie_probability_over_all_46": probability(tie_mass, all_mass),
                "emitted_probability_over_all_46": probability(emitted_mass, all_mass),
                "uncovered_probability_over_all_46": probability(uncovered_mass, all_mass),
                "emitted_person_probability_given_person": probability(emitted_person_mass, person_mass),
                "uncovered_person_probability_given_person": probability(uncovered_person_mass, person_mass),
            }
        def top_agreement(k: int) -> float:
            a = {int(item["global_object_rank"]) for item in sorted(random_candidates.values(), key=lambda x: float(x["score"]["total_log_probability"]), reverse=True)[:k]}
            b = {int(item["global_object_rank"]) for item in sorted(sorted_candidates.values(), key=lambda x: float(x["score"]["total_log_probability"]), reverse=True)[:k]}
            return len(a & b) / max(1, k)
        x_mean = sum(random_values) / len(random_values)
        y_mean = sum(sorted_values) / len(sorted_values)
        slope_den = math.fsum((x - x_mean) ** 2 for x in random_values)
        slope = None if slope_den == 0 else math.fsum((x - x_mean) * (y - y_mean) for x, y in zip(random_values, sorted_values)) / slope_den
        intercept = None if slope is None else y_mean - slope * x_mean
        r2 = None if slope is None else pearson(random_values, sorted_values)
        r2 = None if r2 is None else r2 * r2
        merged["owners"][str(owner)] = {
            "owner_person_rank": owner,
            "owner_global_rank": int(random_candidates[target_global_rank].get("global_object_rank")),
            "random": {
                "top_candidates": sorted(random_candidates.values(), key=lambda x: float(x["score"]["total_log_probability"]), reverse=True)[:10],
                "target_owner_next_rank": rank_of(random_candidates, target_global_rank),
                "target_owner_score": random_candidates[target_global_rank]["score"],
                "all_candidate_logsumexp": mass(random_candidates, ranks),
                "person_candidate_logsumexp": mass(random_candidates, [r for r in ranks if str(random_candidates[r].get("desc")) == "person"]),
                "emitted_logsumexp": mass(random_candidates, emitted),
                "uncovered_logsumexp": mass(random_candidates, uncovered),
                "emitted_person_logsumexp": mass(random_candidates, emitted_person),
                "uncovered_person_logsumexp": mass(random_candidates, uncovered_person),
                "tie_logsumexp": mass(random_candidates, tie_rows),
                "candidate_restricted_probabilities": restricted_probabilities(random_candidates),
                "candidate_normalized_exact_row_entropy": entropy(random_values),
                "terminal_comparison": {"terminal_log_probability": random_terminal, "object_ref_start_log_probability": random_object_start, "continue_minus_terminal_margin": random_object_start - random_terminal, "target_minus_terminal_full_row": float(random_candidates[target_global_rank]["score"]["total_log_probability"]) - random_terminal},
            },
            "sorted": {
                "top_candidates": sorted(sorted_candidates.values(), key=lambda x: float(x["score"]["total_log_probability"]), reverse=True)[:10],
                "target_owner_next_rank": rank_of(sorted_candidates, target_global_rank),
                "target_owner_score": sorted_candidates[target_global_rank]["score"],
                "all_candidate_logsumexp": mass(sorted_candidates, ranks),
                "person_candidate_logsumexp": mass(sorted_candidates, [r for r in ranks if str(sorted_candidates[r].get("desc")) == "person"]),
                "emitted_logsumexp": mass(sorted_candidates, emitted),
                "uncovered_logsumexp": mass(sorted_candidates, uncovered),
                "emitted_person_logsumexp": mass(sorted_candidates, emitted_person),
                "uncovered_person_logsumexp": mass(sorted_candidates, uncovered_person),
                "tie_logsumexp": mass(sorted_candidates, tie_rows),
                "candidate_restricted_probabilities": restricted_probabilities(sorted_candidates),
                "candidate_normalized_exact_row_entropy": entropy(sorted_values),
                "terminal_comparison": {"terminal_log_probability": sorted_terminal, "object_ref_start_log_probability": sorted_object_start, "continue_minus_terminal_margin": sorted_object_start - sorted_terminal, "target_minus_terminal_full_row": float(sorted_candidates[target_global_rank]["score"]["total_log_probability"]) - sorted_terminal},
            },
            "sorted_minus_random_by_candidate": deltas,
            "top1_agreement": top_agreement(1),
            "top3_agreement": top_agreement(3),
            "top5_agreement": top_agreement(5),
            "sorted_vs_random_affine_fit": {"slope": slope, "intercept": intercept, "r2": r2},
        }
        merged["correlations"][str(owner)] = {
            "pearson_total_log_probability": pearson(random_values, sorted_values),
            "spearman_total_log_probability": spearman(random_values, sorted_values),
            "candidate_ranks": ranks,
        }
    return merged


def _atomic_json_dump(path: Path, payload: Mapping[str, Any], *, force: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {path}; pass --force")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _load_legacy_coord_module() -> types.ModuleType:
    source = subprocess.check_output(
        ["git", "show", f"{LEGACY_COORD_SOURCE_REVISION}:{LEGACY_COORD_SOURCE_PATH}"],
        text=True,
    )
    module = types.ModuleType("coordexp_historical_row_offsets")
    module.__file__ = f"git:{LEGACY_COORD_SOURCE_REVISION}:{LEGACY_COORD_SOURCE_PATH}"
    sys.modules[module.__name__] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def _load_historical_model(base_model: Path, checkpoint: Path, device: str) -> tuple[Any, Any, dict[str, Any]]:
    """Load exact legacy coordinate adapter before loading the LoRA modules."""

    import torch
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    module = _load_legacy_coord_module()
    weights_path = checkpoint / "adapter_model.safetensors"
    weights = load_file(str(weights_path), device="cpu")
    ids_key = "base_model.model.coord_offset_adapter.coord_ids"
    embed_key = "base_model.model.coord_offset_adapter.embed_offset"
    coord_ids = [int(v) for v in weights[ids_key].tolist()]
    expected_embed = weights[embed_key].detach().cpu()
    processor = AutoProcessor.from_pretrained(str(base_model), local_files_only=True, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(base_model),
        torch_dtype=torch.float32,
        attn_implementation="eager",
        local_files_only=True,
        trust_remote_code=True,
    )
    module.install_coord_offset_adapter(model, coord_ids=coord_ids, tie_head=True, dtype="auto")
    model = PeftModel.from_pretrained(model, str(checkpoint), local_files_only=True, is_trainable=False)
    model = model.to(device=device).eval()
    active_hook_adapter = module.reattach_coord_offset_hooks(model)
    if active_hook_adapter is None:
        raise RuntimeError("legacy coord offset hooks were not reattached after PEFT loading")
    active = model.base_model.model.coord_offset_adapter.modules_to_save["default"].embed_offset.detach().cpu()
    common_active = active.to(dtype=torch.float32)
    common_expected = expected_embed.to(dtype=torch.float32)
    diff = (common_active - common_expected).abs()
    if not torch.equal(active, expected_embed) and float(diff.max().item()) != 0.0:
        raise RuntimeError(f"legacy coord offset tensor mismatch: max_abs_diff={float(diff.max().item())}")
    receipt = {
        "source_revision": LEGACY_COORD_SOURCE_REVISION,
        "source_path": LEGACY_COORD_SOURCE_PATH,
        "coord_id_count": len(coord_ids),
        "coord_ids_sha256": sha256_json(coord_ids),
        "checkpoint_tensor_dtype": str(expected_embed.dtype),
        "active_tensor_dtype": str(active.dtype),
        "active_tensor_shape": list(active.shape),
        "max_abs_diff_float32": float(diff.max().item()),
        "mean_abs_diff_float32": float(diff.mean().item()),
        "exact_tensor_equality": bool(torch.equal(active, expected_embed)),
        "model_dtype": str(next(model.parameters()).dtype),
        "attention_implementation": "eager",
        "hooks_reattached": True,
    }
    return model, processor, receipt


def _repeat_model_inputs(base_inputs: Mapping[str, Any], batch_size: int) -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {}
    for key, value in base_inputs.items():
        if not isinstance(value, torch.Tensor) or key in {"input_ids", "attention_mask"}:
            continue
        if value.ndim == 0:
            result[key] = value
        elif key in {"pixel_values", "pixel_values_videos"}:
            result[key] = value.repeat((batch_size,) + (1,) * (value.ndim - 1))
        elif value.shape[0] == 1:
            result[key] = value.repeat((batch_size,) + (1,) * (value.ndim - 1))
        else:
            raise ValueError(f"cannot repeat model input {key!r} with shape {tuple(value.shape)}")
    return result


def _forward_batch(
    model: Any,
    base_inputs: Mapping[str, Any],
    full_token_sequences: Sequence[Sequence[int]],
    device: str,
    *,
    logits_to_keep: int,
) -> Any:
    import torch

    lengths = {len(sequence) for sequence in full_token_sequences}
    if len(lengths) != 1:
        raise ValueError("candidate batch sequences must have equal lengths")
    input_ids = torch.tensor(full_token_sequences, dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    kwargs = _repeat_model_inputs(base_inputs, len(full_token_sequences))
    kwargs.update({
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": False,
        "return_dict": True,
        # Qwen3-VL accepts this integer and returns only the tail needed by
        # the score decomposition.  Avoid transferring a full vocabulary
        # tensor for every long image prefix.
        "logits_to_keep": int(logits_to_keep),
    })
    kwargs = {key: value.to(device=device) if isinstance(value, torch.Tensor) else value for key, value in kwargs.items()}
    with torch.inference_mode():
        output = model(**kwargs)
    logits = getattr(output, "logits", None)
    if logits is None:
        raise RuntimeError("historical model forward returned no logits")
    return logits.detach().to(device="cpu", dtype=torch.float32)


def _score_arm(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    adapter_role = str(args.adapter_role)
    checkpoint = Path(args.checkpoint or DEFAULT_CHECKPOINTS[adapter_role]).expanduser().resolve(strict=True)
    base_model = Path(args.base_model).expanduser().resolve(strict=True)
    image_path = Path(args.image).expanduser().resolve(strict=True)
    jsonl_path = Path(args.jsonl).expanduser().resolve(strict=True)
    output_root = Path(args.output_root).expanduser().resolve()
    output_path = output_root / "arms" / f"{adapter_role}-owner-person-{int(args.owner_rank)}.json"
    if output_path.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {output_path}; pass --force")
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA is required for score-arm")
    packet = select_image2299_rows(jsonl_path, image_path)
    prompt = verify_historical_prompt("sorted")
    candidates = candidate_rows(packet, int(args.owner_rank))
    _, prefix_rows = append_prefix_rows(packet, int(args.owner_rank))
    device = str(args.device)
    model, processor, coord_receipt = _load_historical_model(base_model, checkpoint, device)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": prompt["system"]}]},
        {"role": "user", "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": prompt["user"]},
        ]},
    ]
    base_inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    tokenizer = processor.tokenizer
    base_prompt_ids = [int(v) for v in base_inputs["input_ids"][0].tolist()]
    prefix_ids: list[int] = []
    for row in prefix_rows:
        ids = _tokenize_row(tokenizer, row)
        if len(ids) != 7:
            raise RuntimeError(f"historical row tokenized to {len(ids)} tokens, expected 7: {row}")
        prefix_ids.extend(ids)
    candidate_token_ids = []
    for item in candidates:
        ids = _tokenize_row(tokenizer, item["row_text"])
        if len(ids) != 7:
            raise RuntimeError(f"candidate row tokenized to {len(ids)} tokens, expected 7: {item['row_text']}")
        item["token_ids"] = ids
        candidate_token_ids.append(ids)
    terminal_id = tokenizer.convert_tokens_to_ids(TERMINAL_TOKEN)
    object_ref_start_id = tokenizer.convert_tokens_to_ids("<|object_ref_start|>")
    if terminal_id is None or int(terminal_id) < 0:
        raise RuntimeError("tokenizer has no <|im_end|> token")
    terminal_id = int(terminal_id)
    prefix_sequence = base_prompt_ids + prefix_ids
    boundary_len = len(prefix_sequence)
    scores: list[dict[str, Any]] = []
    batch_size = int(args.batch_size)
    for start in range(0, len(candidates), batch_size):
        group = candidates[start : start + batch_size]
        full_sequences = [prefix_sequence + item["token_ids"] for item in group]
        logits = _forward_batch(model, base_inputs, full_sequences, device, logits_to_keep=8)
        for row_index, item in enumerate(group):
            score = score_token_sequence(logits[row_index], boundary_length=1, row_token_ids=item["token_ids"])
            scores.append({
                "global_object_rank": item["global_object_rank"],
                "person_only_rank": item["person_only_rank"],
                "category_rank": item["category_rank"],
                "desc": item["desc"],
                "category_id": item["category_id"],
                "coco_ann_id": item["coco_ann_id"],
                "bbox": item["bbox"],
                "row_text": item["row_text"],
                "token_ids": item["token_ids"],
                "emitted": item["emitted"],
                "uncovered": item["uncovered"],
                "emitted_person": item["emitted_person"],
                "uncovered_person": item["uncovered_person"],
                "score": score,
            })
    terminal_sequence = [prefix_sequence]
    terminal_logits = _forward_batch(model, base_inputs, [prefix_sequence], device, logits_to_keep=1)
    terminal = terminal_score(
        terminal_logits[0], boundary_length=1, terminal_token_id=terminal_id,
        object_ref_start_token_id=None if object_ref_start_id is None else int(object_ref_start_id),
    )
    payload = {
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "adapter_role": adapter_role,
        "training_ordering": "random_permutation" if adapter_role == "random" else "sorted",
        "readout_prompt_ordering": "sorted",
        "model": {
            "base_model": str(base_model),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint / "adapter_model.safetensors"),
            "runtime_dtype": "torch.float32",
            "device": device,
        },
        "image": {
            "image_id": EXPECTED_IMAGE_ID,
            "path": str(image_path),
            "sha256": packet["image_sha256"],
            "source_jsonl": str(jsonl_path),
            "source_jsonl_full_file_sha256": packet["jsonl_sha256"],
            "source_jsonl_record_sha256": packet["record_sha256"],
            "source_line_index_1_based": packet["source_line_index"],
        },
        "prompt": {"hash": prompt["hash"], "system": prompt["system"], "user": prompt["user"], "payload": prompt["payload"]},
        "legacy_row_schema": {
            "row_pattern": "<|object_ref_start|>person<|box_start|><|coord_x1|><|coord_y1|><|coord_x2|><|coord_y2|>",
            "separator": "none",
            "terminal": TERMINAL_TOKEN,
            "prefix_person_only_ranks": list(PARENT_PERSON_RANKS) + [int(args.owner_rank)],
            },
        "owner_person_rank": int(args.owner_rank),
        "owner_global_rank": next(item["global_object_rank"] for item in candidates if item.get("person_only_rank") == int(args.owner_rank)),
        "prefix_rows": prefix_rows,
        "prefix_token_ids": prefix_ids,
        "prefix_token_count": len(prefix_ids),
        "base_prompt_token_count": len(base_prompt_ids),
        "coord_offset_load_receipt": coord_receipt,
        "candidate_count": len(scores),
        "candidate_scores": sorted(scores, key=lambda item: int(item["global_object_rank"])),
        "terminal_score": terminal,
        "execution": {"batch_size": batch_size, "torch_version": torch.__version__, "image_forward": True},
    }
    _atomic_json_dump(output_path, payload, force=bool(args.force))
    return payload


def _seed_sampling(seed: int) -> None:
    """Set the explicit per-sample seed used by both adapter arms."""

    import torch

    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _sample_arm(args: argparse.Namespace) -> dict[str, Any]:
    """Run paired stochastic continuation from one frozen owner prefix.

    This is intentionally separate from ``score-arm``.  The likelihood screen
    is deterministic teacher-forced scoring; this arm observes the model's own
    next-row choices under a fixed seed list.  It never claims that a sampled
    row is a new ground-truth annotation: matching is only to the frozen
    image-2299 candidate packet and can be ambiguous or unmatched.
    """

    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    adapter_role = str(args.adapter_role)
    checkpoint = Path(args.checkpoint or DEFAULT_CHECKPOINTS[adapter_role]).expanduser().resolve(strict=True)
    base_model = Path(args.base_model).expanduser().resolve(strict=True)
    image_path = Path(args.image).expanduser().resolve(strict=True)
    jsonl_path = Path(args.jsonl).expanduser().resolve(strict=True)
    output_root = Path(args.output_root).expanduser().resolve()
    owner_rank = int(args.owner_rank)
    output_path = output_root / "sample-arms" / f"{adapter_role}-owner-person-{owner_rank}.json"
    if output_path.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {output_path}; pass --force")
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA is required for sample-arm")
    packet = select_image2299_rows(jsonl_path, image_path)
    prompt = verify_historical_prompt("sorted")
    candidates = candidate_rows(packet, owner_rank)
    _, prefix_rows = append_prefix_rows(packet, owner_rank)
    model, processor, coord_receipt = _load_historical_model(base_model, checkpoint, str(args.device))
    messages = [
        {"role": "system", "content": [{"type": "text", "text": prompt["system"]}]},
        {"role": "user", "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": prompt["user"]},
        ]},
    ]
    base_inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    tokenizer = processor.tokenizer
    base_prompt_ids = [int(value) for value in base_inputs["input_ids"][0].tolist()]
    prefix_ids: list[int] = []
    for row in prefix_rows:
        ids = _tokenize_row(tokenizer, row)
        if len(ids) != 7:
            raise RuntimeError(f"historical row tokenized to {len(ids)} tokens, expected 7: {row}")
        prefix_ids.extend(ids)
    terminal_id = tokenizer.convert_tokens_to_ids(TERMINAL_TOKEN)
    object_ref_start_id = tokenizer.convert_tokens_to_ids("<|object_ref_start|>")
    box_start_id = tokenizer.convert_tokens_to_ids("<|box_start|>")
    coordinate_ids = {int(tokenizer.convert_tokens_to_ids(coord_token(value))) for value in range(1000)}
    if any(value is None or int(value) < 0 for value in (terminal_id, object_ref_start_id, box_start_id)):
        raise RuntimeError("historical tokenizer is missing a required row or terminal token")
    prefix_sequence = base_prompt_ids + prefix_ids
    device = str(args.device)
    generated_rows: list[dict[str, Any]] = []

    class _LegacyRowStoppingCriteria(StoppingCriteria):
        def __init__(self, start_length: int) -> None:
            self.start_length = int(start_length)

        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
            del scores, kwargs
            return torch.tensor(
                [
                    legacy_sampling_suffix_is_complete(
                        row[self.start_length :].tolist(),
                        object_ref_start_token_id=int(object_ref_start_id),
                        box_start_token_id=int(box_start_id),
                        coordinate_token_ids=coordinate_ids,
                        terminal_token_id=int(terminal_id),
                    )
                    for row in input_ids
                ],
                dtype=torch.bool,
                device=input_ids.device,
            )

    stopping_criteria = StoppingCriteriaList([_LegacyRowStoppingCriteria(len(prefix_sequence))])
    for seed in SAMPLING_SEEDS:
        _seed_sampling(seed)
        generation_inputs: dict[str, Any] = {}
        for key, value in base_inputs.items():
            generation_inputs[key] = value.to(device=device) if isinstance(value, torch.Tensor) else value
        generation_inputs["input_ids"] = torch.tensor([prefix_sequence], dtype=torch.long, device=device)
        generation_inputs["attention_mask"] = torch.ones((1, len(prefix_sequence)), dtype=torch.long, device=device)
        with torch.inference_mode():
            output_ids = model.generate(
                **generation_inputs,
                do_sample=True,
                temperature=SAMPLING_TEMPERATURE,
                top_p=SAMPLING_TOP_P,
                repetition_penalty=SAMPLING_REPETITION_PENALTY,
                max_new_tokens=SAMPLING_MAX_NEW_TOKENS,
                num_return_sequences=1,
                use_cache=True,
                stopping_criteria=stopping_criteria,
            )
        if not isinstance(output_ids, torch.Tensor) or output_ids.ndim != 2 or output_ids.shape[0] != 1:
            raise RuntimeError("historical sampling returned unexpected generate output shape")
        new_token_ids = [int(value) for value in output_ids[0, len(prefix_sequence):].detach().cpu().tolist()]
        text = tokenizer.decode(new_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        parsed = parse_legacy_generation(text)
        matched = match_parsed_row_to_candidates(parsed, candidates)
        generated_rows.append({
            "seed": int(seed),
            "generated_token_ids": new_token_ids,
            "generated_text": text,
            "parsed": parsed,
            "match": matched,
        })
    payload = {
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "adapter_role": adapter_role,
        "training_ordering": "random_permutation" if adapter_role == "random" else "sorted",
        "readout_prompt_ordering": "sorted",
        "model": {
            "base_model": str(base_model),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint / "adapter_model.safetensors"),
            "runtime_dtype": "torch.float32",
            "device": device,
        },
        "image": {
            "image_id": EXPECTED_IMAGE_ID,
            "path": str(image_path),
            "sha256": packet["image_sha256"],
            "source_jsonl": str(jsonl_path),
            "source_jsonl_full_file_sha256": packet["jsonl_sha256"],
            "source_jsonl_record_sha256": packet["record_sha256"],
            "source_line_index_1_based": packet["source_line_index"],
        },
        "prompt": {"hash": prompt["hash"], "system": prompt["system"], "user": prompt["user"], "payload": prompt["payload"]},
        "legacy_row_schema": {
            "row_pattern": "<|object_ref_start|>person<|box_start|><|coord_x1|><|coord_y1|><|coord_x2|><|coord_y2|>",
            "separator": "none",
            "terminal": TERMINAL_TOKEN,
            "prefix_person_only_ranks": list(PARENT_PERSON_RANKS) + [owner_rank],
        },
        "owner_person_rank": owner_rank,
        "owner_global_rank": next(item["global_object_rank"] for item in candidates if item.get("person_only_rank") == owner_rank),
        "prefix_rows": prefix_rows,
        "prefix_token_ids": prefix_ids,
        "prefix_token_count": len(prefix_ids),
        "base_prompt_token_count": len(base_prompt_ids),
        "coord_offset_load_receipt": coord_receipt,
        "sampling": {
            "seed_list": list(SAMPLING_SEEDS),
            "seed_list_sha256": sha256_json(list(SAMPLING_SEEDS)),
            "temperature": SAMPLING_TEMPERATURE,
            "top_p": SAMPLING_TOP_P,
            "repetition_penalty": SAMPLING_REPETITION_PENALTY,
            "max_new_tokens": SAMPLING_MAX_NEW_TOKENS,
            "stopping_rule": "terminal token or one object_ref_start + box_start + four coordinate tokens",
            "iou_threshold": MATCH_IOU_THRESHOLD,
            "ambiguity_margin": MATCH_AMBIGUITY_MARGIN,
            "paired_across_adapter_roles": True,
        },
        "candidate_count": len(candidates),
        "candidate_rows": candidates,
        "samples": generated_rows,
        "execution": {"torch_version": torch.__version__, "image_forward": True},
    }
    _atomic_json_dump(output_path, payload, force=bool(args.force))
    return payload


def _merge(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root).expanduser().resolve(strict=True)
    arm_paths = sorted((root / "arms").glob("*.json"))
    if len(arm_paths) != 8:
        raise FileNotFoundError(f"merge requires eight arm JSON files under {root / 'arms'}, found {len(arm_paths)}")
    records = [json.loads(path.read_text()) for path in arm_paths]
    merged = merge_arm_records(records)
    merged["source_arm_paths"] = [str(path) for path in arm_paths]
    merged["image_id"] = EXPECTED_IMAGE_ID
    merged["prompt_hash"] = EXPECTED_PROMPT_HASH
    output_path = root / "summary" / "summary.json"
    _atomic_json_dump(output_path, merged, force=bool(args.force))
    report = ["# Historical random-versus-sorted image 2299 screen", "", f"Prompt hash: `{EXPECTED_PROMPT_HASH}`", ""]
    for owner in OWNER_PERSON_RANKS:
        item = merged["owners"][str(owner)]
        corr = merged["correlations"][str(owner)]
        report.extend([
            f"## Person-only owner rank {owner} (global rank {item['owner_global_rank']})",
            f"- Target rank (random / sorted): {item['random']['target_owner_next_rank']} / {item['sorted']['target_owner_next_rank']}",
            f"- Pearson correlation: {corr['pearson_total_log_probability']}",
            f"- Spearman correlation: {corr['spearman_total_log_probability']}",
            f"- Continue-minus-terminal margin (random / sorted): {item['random']['terminal_comparison']['continue_minus_terminal_margin']:.6f} / {item['sorted']['terminal_comparison']['continue_minus_terminal_margin']:.6f}",
            "",
        ])
        for adapter in ("random", "sorted"):
            top = item[adapter]["top_candidates"][:5]
            report.append(f"### {adapter} top candidates")
            report.extend(
                f"- global rank {row['global_object_rank']} / person rank {row.get('person_only_rank')} "
                f"/ {row['desc']}: {row['score']['total_log_probability']:.6f}"
                for row in top
            )
            report.append("")
    report_path = root / "summary" / "report.md"
    if report_path.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {report_path}; pass --force")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report), encoding="utf-8")
    return merged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    score = sub.add_parser("score-arm")
    score.add_argument("--adapter-role", choices=("random", "sorted"), required=True)
    score.add_argument("--owner-rank", type=int, choices=OWNER_PERSON_RANKS, required=True, help="geometry-sorted person-only rank")
    score.add_argument("--checkpoint", type=Path)
    score.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    score.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    score.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    score.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    score.add_argument("--device", default="cuda")
    score.add_argument("--batch-size", type=int, default=4)
    score.add_argument("--force", action="store_true")
    sample = sub.add_parser("sample-arm")
    sample.add_argument("--adapter-role", choices=("random", "sorted"), required=True)
    sample.add_argument("--owner-rank", type=int, choices=OWNER_PERSON_RANKS, required=True, help="geometry-sorted person-only rank")
    sample.add_argument("--checkpoint", type=Path)
    sample.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    sample.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    sample.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    sample.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    sample.add_argument("--device", default="cuda")
    sample.add_argument("--force", action="store_true")
    merge = sub.add_parser("merge")
    merge.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    merge.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "score-arm":
        _score_arm(args)
    elif args.command == "sample-arm":
        _sample_arm(args)
    elif args.command == "merge":
        _merge(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
