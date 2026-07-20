#!/usr/bin/env python3
"""Build a small, deterministic visual review packet from exact rollout traces.

This is intentionally a one-purpose research helper.  It does not decide
whether a coordinate is trustworthy.  It only gathers one exact row, its
canonical ledger owner, and reviewer-proposed coordinates into a packet that
can be inspected before a training state bank is assembled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont


SCHEMA_VERSION = "exact_trace_coordinate_review_packet.v1"
DEFAULT_CROP_HALO = 48
DEFAULT_RESIZE_FACTOR = 4
BLIND_IMAGE_IDS = {
    "1584", "2685", "4134", "5001", "6040", "7511", "10707",
    "13348", "13923", "14038", "14439", "16228",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    return sha256_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))


def _require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _int_image_id(value: Any) -> str:
    text = str(value)
    try:
        return str(int(text))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid image_id: {value!r}") from exc


def _validate_bins(value: Any, name: str) -> list[int]:
    values = _require_list(value, name)
    if len(values) != 4 or any(not isinstance(item, (int, float)) for item in values):
        raise ValueError(f"{name} must contain four numeric values")
    result = [int(item) for item in values]
    if any(item < 0 or item > 999 for item in result):
        raise ValueError(f"{name} values must be in [0, 999]")
    if result[0] >= result[2] or result[1] >= result[3]:
        raise ValueError(f"{name} must be a non-empty xyxy box")
    return result


def _validate_hash(values: Any, declared: Any, name: str) -> str:
    if not isinstance(values, list) or any(not isinstance(item, int) for item in values):
        raise ValueError(f"{name} token ids are not an integer list")
    actual = sha256_json(values)
    if declared != actual:
        raise ValueError(f"{name} hash mismatch: declared={declared!r} actual={actual}")
    return actual


def _case_manifest_entries(document: Any) -> list[Mapping[str, Any]]:
    if isinstance(document, list):
        entries = document
    elif isinstance(document, Mapping):
        entries = document.get("cases")
    else:
        entries = None
    if not isinstance(entries, list) or not entries:
        raise ValueError("case manifest must contain a non-empty cases list")
    result: list[Mapping[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("case manifest entries must be objects")
        result.append(entry)
    return result


def _image_path(image: Mapping[str, Any]) -> Path:
    prompt = image.get("prompt")
    if not isinstance(prompt, Mapping) or not isinstance(prompt.get("image_path"), str):
        raise ValueError("trace image lacks prompt.image_path")
    return Path(str(prompt["image_path"])).resolve(strict=True)


def _norm_to_px(box: Sequence[float], width: int, height: int) -> tuple[float, float, float, float]:
    return (
        float(box[0]) * width / 1000.0,
        float(box[1]) * height / 1000.0,
        float(box[2]) * width / 1000.0,
        float(box[3]) * height / 1000.0,
    )


def _px_box(box: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
    converted = _norm_to_px(box, width, height)
    return tuple(max(0, min(int(round(value)), limit)) for value, limit in zip(converted, (width - 1, height - 1, width - 1, height - 1)))  # type: ignore[return-value]


def _draw_box(draw: ImageDraw.ImageDraw, box: Sequence[float], *, width: int, height: int, color: str, label: str, line_width: int = 3) -> None:
    draw.rectangle(_px_box(box, width, height), outline=color, width=line_width)
    x1, y1, _, _ = _px_box(box, width, height)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(12, min(width, height) // 38))
    except OSError:
        font = ImageFont.load_default()
    draw.text((x1 + 2, max(0, y1 - 18)), label, fill=color, font=font)


def _same_category_nearby(
    ledger: Sequence[Mapping[str, Any]],
    category: str,
    focus: Sequence[float],
    *,
    distance_norm: float = 180.0,
) -> list[Mapping[str, Any]]:
    fx = (float(focus[0]) + float(focus[2])) / 2.0
    fy = (float(focus[1]) + float(focus[3])) / 2.0
    matches: list[tuple[float, Mapping[str, Any]]] = []
    for entity in ledger:
        if str(entity.get("description")) != category:
            continue
        box = entity.get("bbox_norm1000")
        if not isinstance(box, list) or len(box) != 4:
            continue
        cx = (float(box[0]) + float(box[2])) / 2.0
        cy = (float(box[1]) + float(box[3])) / 2.0
        distance = math.hypot(cx - fx, cy - fy)
        if distance <= distance_norm:
            matches.append((distance, entity))
    return [entity for _, entity in sorted(matches, key=lambda item: (item[0], str(item[1].get("entity_id"))))]


def _draw_full(
    image: Image.Image,
    *,
    predicted: Sequence[float],
    reference: Sequence[float],
    category: str,
    owner_id: str,
    nearby: Sequence[Mapping[str, Any]],
    output: Path,
) -> None:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for entity in nearby:
        _draw_box(
            draw,
            _validate_bins(entity.get("bbox_norm1000"), f"entity {entity.get('entity_id')} bbox"),
            width=canvas.width,
            height=canvas.height,
            color="#2f75b5",
            label=f"near {entity.get('entity_id')}",
            line_width=1,
        )
    _draw_box(draw, reference, width=canvas.width, height=canvas.height, color="#27ae60", label=f"reference {owner_id}")
    _draw_box(draw, predicted, width=canvas.width, height=canvas.height, color="#e74c3c", label=f"prediction {category}")
    canvas.save(output, format="PNG")


def _draw_crop(
    image: Image.Image,
    *,
    predicted: Sequence[float],
    reference: Sequence[float],
    category: str,
    owner_id: str,
    nearby: Sequence[Mapping[str, Any]],
    output: Path,
    halo: int,
    resize_factor: int,
    resample_name: str,
) -> None:
    width, height = image.size
    union = [
        min(float(predicted[0]), float(reference[0])),
        min(float(predicted[1]), float(reference[1])),
        max(float(predicted[2]), float(reference[2])),
        max(float(predicted[3]), float(reference[3])),
    ]
    ux1, uy1, ux2, uy2 = _norm_to_px(union, width, height)
    left = max(0, int(math.floor(ux1 - halo)))
    top = max(0, int(math.floor(uy1 - halo)))
    right = min(width, int(math.ceil(ux2 + halo)))
    bottom = min(height, int(math.ceil(uy2 + halo)))
    if right <= left or bottom <= top:
        raise ValueError("crop is empty")
    canvas = image.convert("RGB").crop((left, top, right, bottom))
    if resample_name == "bicubic":
        resample = Image.Resampling.BICUBIC
    else:
        resample = Image.Resampling.NEAREST
    canvas = canvas.resize((canvas.width * resize_factor, canvas.height * resize_factor), resample=resample)
    draw = ImageDraw.Draw(canvas)

    def draw_local(box: Sequence[float], color: str, label: str, line_width: int) -> None:
        px = _norm_to_px(box, width, height)
        local = tuple((value - offset) * resize_factor for value, offset in zip(px, (left, top, left, top)))
        draw.rectangle(tuple(int(round(value)) for value in local), outline=color, width=line_width * resize_factor)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", max(20, min(canvas.size) // 25))
        except OSError:
            font = ImageFont.load_default()
        draw.text((int(local[0]) + 4, max(0, int(local[1]) - 25)), label, fill=color, font=font)

    for entity in nearby:
        draw_local(
            _validate_bins(entity.get("bbox_norm1000"), f"entity {entity.get('entity_id')} bbox"),
            "#2f75b5",
            f"near {entity.get('entity_id')}",
            1,
        )
    draw_local(reference, "#27ae60", f"reference {owner_id}", 3)
    draw_local(predicted, "#e74c3c", f"prediction {category}", 3)
    canvas.save(output, format="PNG")


def _load_case(entry: Mapping[str, Any]) -> dict[str, Any]:
    required = ("case_id", "exact_trace_path", "row_index", "split_role", "proposed_review_coordinate")
    for key in required:
        if key not in entry:
            raise ValueError(f"case missing {key}")
    case_id = str(entry["case_id"])
    trace_path = Path(str(entry["exact_trace_path"])).resolve(strict=True)
    trace = _read_json(trace_path)
    images = _require_list(trace.get("images"), "trace.images")
    if len(images) != 1 or not isinstance(images[0], Mapping):
        raise ValueError(f"{case_id}: expected exactly one trace image")
    image = images[0]
    image_id = _int_image_id(image.get("image_id"))
    if image_id in BLIND_IMAGE_IDS:
        raise ValueError(f"{case_id}: blind image {image_id} is not allowed")
    row_index = entry["row_index"]
    if not isinstance(row_index, int) or row_index < 0:
        raise ValueError(f"{case_id}: row_index must be a non-negative integer")
    extended = image.get("extended_root_greedy")
    if not isinstance(extended, Mapping):
        raise ValueError(f"{case_id}: trace lacks extended_root_greedy")
    rows = _require_list(extended.get("rows"), "extended_root_greedy.rows")
    row_matches = [row for row in rows if isinstance(row, Mapping) and row.get("row_index") == row_index]
    if len(row_matches) != 1:
        raise ValueError(f"{case_id}: expected exactly one row_index={row_index}, found {len(row_matches)}")
    row = row_matches[0]
    if row.get("accepted_complete_row") is not True or row.get("status") != "success":
        raise ValueError(f"{case_id}: row is not an accepted complete successful row")
    parsed = _require_list(row.get("parsed_predictions"), f"{case_id}.parsed_predictions")
    parse_evidence = row.get("parse_evidence")
    if not isinstance(parse_evidence, Mapping) or parse_evidence.get("parse_status") != "accepted":
        raise ValueError(f"{case_id}: parse evidence is not accepted")
    evidence_predictions = _require_list(parse_evidence.get("predictions"), f"{case_id}.parse_evidence.predictions")
    if len(parsed) != 1 or len(evidence_predictions) != 1:
        raise ValueError(f"{case_id}: exact review case must contain exactly one parsed prediction")
    prediction = parsed[0]
    if not isinstance(prediction, Mapping) or prediction != evidence_predictions[0]:
        raise ValueError(f"{case_id}: parsed prediction disagrees with parse evidence")
    strict_owners = _require_list(row.get("strict_matched_owner_ids"), f"{case_id}.strict_matched_owner_ids")
    if len(strict_owners) != 1 or not isinstance(strict_owners[0], str) or not strict_owners[0]:
        raise ValueError(f"{case_id}: expected exactly one strict matched owner")
    owner_id = strict_owners[0]

    predicted = _validate_bins(prediction.get("coord_bins"), f"{case_id}.predicted coord_bins")
    proposed_axis = entry["proposed_review_coordinate"]
    if proposed_axis not in {"x1", "y1", "x2", "y2"}:
        raise ValueError(f"{case_id}: proposed_review_coordinate must be one of x1, y1, x2, y2")
    prefix = row.get("prefix_token_ids")
    prefix_hash = _validate_hash(prefix, row.get("prefix_token_ids_sha256"), f"{case_id}.prefix")
    input_prefix = row.get("input_prefix_token_ids")
    if input_prefix is not None:
        input_prefix_hash = _validate_hash(
            input_prefix,
            row.get("input_prefix_token_ids_sha256"),
            f"{case_id}.input_prefix",
        )
        if input_prefix != prefix or input_prefix_hash != prefix_hash:
            raise ValueError(f"{case_id}: input prefix disagrees with exact prefix")
    raw = row.get("raw_generated_token_ids")
    raw_hash = _validate_hash(raw, row.get("raw_generated_token_ids_sha256"), f"{case_id}.raw row")
    raw_text = row.get("raw_generated_text")
    if not isinstance(raw_text, str) or row.get("raw_generated_text_sha256") != sha256_bytes(raw_text.encode()):
        raise ValueError(f"{case_id}: raw generated text hash mismatch")
    image_path = _image_path(image)
    image_stem = image_path.stem.lstrip("0") or "0"
    if image_stem.isdigit() and image_stem != image_id:
        raise ValueError(f"{case_id}: image_id does not match image filename")
    actual_image_hash = sha256_file(image_path)
    declared_image_hash = image.get("prompt", {}).get("image_sha256")
    if declared_image_hash != actual_image_hash:
        raise ValueError(f"{case_id}: image hash mismatch")
    with Image.open(image_path) as opened:
        actual_size = opened.size
        image_copy = opened.convert("RGB")
    prompt = image.get("prompt")
    if isinstance(prompt, Mapping):
        declared_width, declared_height = prompt.get("width"), prompt.get("height")
        if (
            (declared_width is not None and int(declared_width) != actual_size[0])
            or (declared_height is not None and int(declared_height) != actual_size[1])
        ):
            raise ValueError(f"{case_id}: trace image dimensions disagree with image file")
    ledger = _require_list(image.get("entity_ledger"), f"{case_id}.entity_ledger")
    owner_matches = [entity for entity in ledger if isinstance(entity, Mapping) and str(entity.get("entity_id")) == owner_id]
    if len(owner_matches) != 1:
        raise ValueError(f"{case_id}: owner {owner_id!r} missing or duplicated in ledger")
    owner = owner_matches[0]
    category = str(prediction.get("description") or owner.get("description") or "unknown")
    if str(owner.get("description")) != category:
        raise ValueError(f"{case_id}: prediction category does not match strict owner category")
    canonical_reference = _validate_bins(owner.get("bbox_norm1000"), f"{case_id}.owner bbox")
    nearby = [
        entity for entity in _same_category_nearby(ledger, category, canonical_reference)
        if str(entity.get("entity_id")) != owner_id
    ]
    trace_hash = sha256_file(trace_path)
    return {
        "case_id": case_id,
        "split_role": str(entry["split_role"]),
        "proposed_review_axis": proposed_axis,
        "image_id": image_id,
        "row_index": row_index,
        "exact_trace_path": str(trace_path),
        "source_trace_sha256": trace_hash,
        "trace_sha256": trace_hash,
        "image_path": str(image_path),
        "image_sha256": actual_image_hash,
        "source_image_sha256": actual_image_hash,
        "image_width": actual_size[0],
        "image_height": actual_size[1],
        "owner_id": owner_id,
        "category": category,
        "predicted_norm1000_xyxy": predicted,
        "reference_norm1000_xyxy": canonical_reference,
        "per_axis_delta_predicted_minus_reference": [
            predicted[index] - canonical_reference[index] for index in range(4)
        ],
        "exact_prefix_token_ids": list(prefix),
        "exact_prefix_token_ids_sha256": prefix_hash,
        "exact_raw_generated_token_ids": list(raw),
        "exact_raw_generated_token_ids_sha256": raw_hash,
        "exact_raw_generated_text": raw_text,
        "exact_raw_generated_text_sha256": row["raw_generated_text_sha256"],
        "review": {
            "entity_review_status": "pending",
            "geometry_review_status": "pending",
            "earlier_coordinates_accepted": None,
            "accepted_coordinate_bins": None,
            "reviewer": None,
            "comment": None,
        },
        "_image": image_copy,
        "_ledger": [entity for entity in ledger if isinstance(entity, Mapping)],
        "_nearby": nearby,
    }


def build_packet(case_manifest_path: Path, output_dir: Path, *, crop_halo: int = DEFAULT_CROP_HALO, resize_factor: int = DEFAULT_RESIZE_FACTOR, crop_resample: str = "bicubic") -> dict[str, Any]:
    entries = _case_manifest_entries(_read_json(case_manifest_path))
    for entry in entries:
        if entry.get("split_role") not in {"train", "eval"}:
            raise ValueError(f"split_role must be train or eval, got {entry.get('split_role')!r}")
    cases = [_load_case(entry) for entry in entries]
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("case_id values must be unique")
    output_dir.mkdir(parents=True, exist_ok=True)
    packet_cases: list[dict[str, Any]] = []
    for case in sorted(cases, key=lambda item: item["case_id"]):
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", case["case_id"])
        full_path = output_dir / f"{slug}.full.png"
        crop_path = output_dir / f"{slug}.crop.png"
        _draw_full(
            case["_image"],
            predicted=case["predicted_norm1000_xyxy"],
            reference=case["reference_norm1000_xyxy"],
            category=case["category"],
            owner_id=case["owner_id"],
            nearby=case["_nearby"],
            output=full_path,
        )
        _draw_crop(
            case["_image"],
            predicted=case["predicted_norm1000_xyxy"],
            reference=case["reference_norm1000_xyxy"],
            category=case["category"],
            owner_id=case["owner_id"],
            nearby=case["_nearby"],
            output=crop_path,
            halo=crop_halo,
            resize_factor=resize_factor,
            resample_name=crop_resample,
        )
        serial = {key: value for key, value in case.items() if not key.startswith("_")}
        serial["artifacts"] = {
            "full_image_png": str(full_path.resolve()),
            "crop_png": str(crop_path.resolve()),
            "crop_halo_pixels": crop_halo,
            "crop_resize_factor": resize_factor,
            "crop_resample": crop_resample,
            "full_image_png_sha256": sha256_file(full_path),
            "crop_png_sha256": sha256_file(crop_path),
        }
        packet_cases.append(serial)
    packet = {
        "schema_version": SCHEMA_VERSION,
        "case_manifest_path": str(case_manifest_path.resolve()),
        "case_manifest_sha256": sha256_file(case_manifest_path),
        "artifacts": {
            "crop_halo_pixels": crop_halo,
            "crop_resize_factor": resize_factor,
            "crop_resample": crop_resample,
        },
        "cases": packet_cases,
    }
    packet_path = output_dir / "review-packet.json"
    packet_path.write_text(json.dumps(packet, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return packet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crop-halo", type=int, default=DEFAULT_CROP_HALO)
    parser.add_argument("--resize-factor", type=int, default=DEFAULT_RESIZE_FACTOR)
    parser.add_argument("--crop-resample", choices=("bicubic", "nearest"), default="bicubic")
    args = parser.parse_args()
    if args.crop_halo < 0 or args.resize_factor < 1:
        raise SystemExit("--crop-halo must be >=0 and --resize-factor must be >=1")
    build_packet(
        args.case_manifest,
        args.output_dir,
        crop_halo=args.crop_halo,
        resize_factor=args.resize_factor,
        crop_resample=args.crop_resample,
    )


if __name__ == "__main__":
    main()
