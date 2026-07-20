#!/usr/bin/env python3
"""Build a visual review packet from exact-prefix sampled-rescue artifacts.

The packet is deliberately a review aid, not a label generator.  It only
includes sampler rows already marked ``selected_verified_uncovered_rescue``
and the frozen greedy harmful row.  Every review decision remains pending;
this helper never admits a sampled row to a training state bank.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.build_exact_trace_coordinate_review_packet import (
    BLIND_IMAGE_IDS,
    _norm_to_px,
    _px_box,
    _validate_bins,
    sha256_bytes,
    sha256_file,
    sha256_json,
)


SCHEMA_VERSION = "exact_prefix_sampled_rescue_review_packet.v1"
DEFAULT_CROP_HALO = 64
DEFAULT_RESIZE_FACTOR = 4
PALETTE = (
    "#2f75b5",
    "#27ae60",
    "#f39c12",
    "#16a085",
    "#8e44ad",
    "#d35400",
    "#2980b9",
    "#c0392b",
    "#7f8c8d",
    "#1abc9c",
    "#9b59b6",
    "#34495e",
)
RESCUE_COLOR = "#e74c3c"
HARMFUL_COLOR = "#8e44ad"


class ReviewPacketError(ValueError):
    """Raised when an input artifact cannot support an exact review packet."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReviewPacketError(f"cannot read JSON artifact {path}: {exc}") from exc


def _ids(value: Any, label: str, *, allow_empty: bool = False) -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ReviewPacketError(f"{label} must be a token-id sequence")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ReviewPacketError(f"{label} contains an invalid token id")
        result.append(int(item))
    if not result and not allow_empty:
        raise ReviewPacketError(f"{label} must not be empty")
    return result


def _hash_ids(value: Any, label: str, *, allow_empty: bool = False) -> tuple[list[int], str]:
    ids = _ids(value, label, allow_empty=allow_empty)
    observed = sha256_json(ids)
    return ids, observed


def _image_id(value: Any) -> str:
    try:
        return str(int(str(value)))
    except (TypeError, ValueError) as exc:
        raise ReviewPacketError(f"invalid image_id: {value!r}") from exc


def _ledger(artifact: Mapping[str, Any]) -> list[dict[str, Any]]:
    selected = artifact.get("selected_source_identity")
    raw = selected.get("positive_owner_ledger") if isinstance(selected, Mapping) else None
    if not isinstance(raw, list) or not raw:
        raise ReviewPacketError("artifact lacks selected_source_identity.positive_owner_ledger")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise ReviewPacketError("positive_owner_ledger contains a non-object")
        entity_id = str(item.get("entity_id", ""))
        if not entity_id or entity_id in seen:
            raise ReviewPacketError("positive_owner_ledger has a missing or duplicate entity_id")
        description = str(item.get("description", ""))
        if not description:
            raise ReviewPacketError(f"ledger owner {entity_id} lacks description")
        box = _validate_bins(item.get("bbox_norm1000"), f"ledger owner {entity_id} bbox")
        seen.add(entity_id)
        result.append({
            "entity_id": entity_id,
            "description": description,
            "bbox_norm1000": box,
            "verification": item.get("verification"),
            "source": item.get("source"),
        })
    return sorted(result, key=lambda item: item["entity_id"])


def _ledger_hash(ledger: Sequence[Mapping[str, Any]]) -> str:
    return sha256_json(list(ledger))


def _resolve_image_path(source_jsonl: Path, image_id: str) -> Path:
    source_jsonl = source_jsonl.resolve(strict=True)
    for line_number, line in enumerate(source_jsonl.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReviewPacketError(f"invalid source JSONL at line {line_number}: {exc}") from exc
        if not isinstance(record, Mapping) or _image_id(record.get("image_id")) != image_id:
            continue
        images = record.get("images")
        if not isinstance(images, list) or len(images) < 1 or not isinstance(images[0], str):
            raise ReviewPacketError(f"source record {image_id} lacks images[0]")
        raw_path = Path(images[0]).expanduser()
        candidates = [raw_path] if raw_path.is_absolute() else [source_jsonl.parent / raw_path]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise ReviewPacketError(f"source record {image_id} image path does not exist: {images[0]}")
    raise ReviewPacketError(f"image_id {image_id} not found in source JSONL {source_jsonl}")


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _draw_labelled_box(
    draw: ImageDraw.ImageDraw,
    box: Sequence[float],
    *,
    width: int,
    height: int,
    color: str,
    label: str,
    line_width: int = 2,
) -> None:
    px = _px_box(box, width, height)
    draw.rectangle(px, outline=color, width=line_width)
    draw.text((px[0] + 2, max(0, px[1] - 17)), label, fill=color, font=_font(max(12, min(width, height) // 44)))


def _draw_full(
    image: Image.Image,
    *,
    ledger: Sequence[Mapping[str, Any]],
    overlays: Sequence[Mapping[str, Any]],
    output: Path,
) -> None:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for index, owner in enumerate(ledger):
        _draw_labelled_box(
            draw,
            owner["bbox_norm1000"],
            width=canvas.width,
            height=canvas.height,
            color=PALETTE[index % len(PALETTE)],
            label=f"ledger {owner['description']}:{owner['entity_id']}",
            line_width=2,
        )
    for overlay in overlays:
        _draw_labelled_box(
            draw,
            overlay["bbox_norm1000"],
            width=canvas.width,
            height=canvas.height,
            color=overlay["color"],
            label=overlay["label"],
            line_width=4,
        )
    canvas.save(output, format="PNG")


def _crop_bounds(
    boxes: Sequence[Sequence[float]], *, width: int, height: int, halo: int
) -> tuple[int, int, int, int]:
    if not boxes:
        raise ReviewPacketError("cannot build a crop without a focus box")
    union = [
        min(float(box[0]) for box in boxes),
        min(float(box[1]) for box in boxes),
        max(float(box[2]) for box in boxes),
        max(float(box[3]) for box in boxes),
    ]
    ux1, uy1, ux2, uy2 = _norm_to_px(union, width, height)
    left = max(0, int(math.floor(ux1 - halo)))
    top = max(0, int(math.floor(uy1 - halo)))
    right = min(width, int(math.ceil(ux2 + halo)))
    bottom = min(height, int(math.ceil(uy2 + halo)))
    if right <= left or bottom <= top:
        raise ReviewPacketError("crop is empty")
    return left, top, right, bottom


def _draw_crop(
    image: Image.Image,
    *,
    ledger: Sequence[Mapping[str, Any]],
    overlays: Sequence[Mapping[str, Any]],
    focus_boxes: Sequence[Sequence[float]],
    output: Path,
    halo: int,
    resize_factor: int,
) -> tuple[int, int, int, int]:
    width, height = image.size
    left, top, right, bottom = _crop_bounds(focus_boxes, width=width, height=height, halo=halo)
    canvas = image.convert("RGB").crop((left, top, right, bottom))
    canvas = canvas.resize((canvas.width * resize_factor, canvas.height * resize_factor), resample=Image.Resampling.BICUBIC)
    draw = ImageDraw.Draw(canvas)

    def local_box(box: Sequence[float]) -> tuple[int, int, int, int]:
        px = _norm_to_px(box, width, height)
        values = tuple((value - offset) * resize_factor for value, offset in zip(px, (left, top, left, top)))
        return tuple(int(round(value)) for value in values)  # type: ignore[return-value]

    for index, owner in enumerate(ledger):
        draw.rectangle(local_box(owner["bbox_norm1000"]), outline=PALETTE[index % len(PALETTE)], width=2 * resize_factor)
    for overlay in overlays:
        box = local_box(overlay["bbox_norm1000"])
        draw.rectangle(box, outline=overlay["color"], width=4 * resize_factor)
        draw.text((box[0] + 4, max(0, box[1] - 27)), overlay["label"], fill=overlay["color"], font=_font(max(20, min(canvas.size) // 24)))
    canvas.save(output, format="PNG")
    return left, top, right, bottom


def _prediction_from_owner_matches(
    owner_matches: Any, owner_id: str, *, label: str
) -> list[float]:
    if not isinstance(owner_matches, list):
        raise ReviewPacketError(f"{label} lacks owner_matches")
    matches = [item for item in owner_matches if isinstance(item, Mapping) and str(item.get("matched_entity_id")) == owner_id]
    if len(matches) != 1:
        raise ReviewPacketError(f"{label} must have exactly one owner match for {owner_id}")
    value = matches[0].get("predicted_bbox_norm1000")
    return [float(item) for item in _validate_bins(value, f"{label} predicted bbox")]


def _validate_artifact(path: Path) -> dict[str, Any]:
    artifact = _read_json(path)
    if artifact.get("schema_version") != "exact_prefix_sampled_rescue.v1":
        raise ReviewPacketError(f"{path}: schema_version is not exact_prefix_sampled_rescue.v1")
    selected = artifact.get("selected_source_identity")
    prefix = artifact.get("prefix")
    frozen = artifact.get("frozen_source")
    greedy = artifact.get("greedy")
    if not isinstance(selected, Mapping) or not isinstance(prefix, Mapping) or not isinstance(frozen, Mapping) or not isinstance(greedy, Mapping):
        raise ReviewPacketError(f"{path}: missing exact source, prefix, frozen, or greedy section")
    image_id = _image_id(selected.get("image_id"))
    if image_id in BLIND_IMAGE_IDS:
        raise ReviewPacketError(f"{path}: blind image {image_id} is not allowed")
    prefix_ids, prefix_hash = _hash_ids(prefix.get("token_ids"), f"{path}: prefix token_ids")
    if prefix_hash != str(prefix.get("token_ids_sha256")):
        raise ReviewPacketError(f"{path}: prefix token hash mismatch")
    prompt_ids, prompt_hash = _hash_ids(selected.get("prompt_token_ids"), f"{path}: prompt token_ids")
    if prompt_hash != str(selected.get("prompt_token_ids_sha256")):
        raise ReviewPacketError(f"{path}: prompt token hash mismatch")
    covered = sorted({str(value) for value in prefix.get("covered_owner_ids", [])})
    uncovered = sorted({str(value) for value in prefix.get("uncovered_owner_ids", [])})
    if not covered or not uncovered:
        raise ReviewPacketError(f"{path}: prefix must contain covered and uncovered owner ids")
    ledger = _ledger(artifact)
    ledger_ids = {item["entity_id"] for item in ledger}
    if not set(covered).issubset(ledger_ids) or not set(uncovered).issubset(ledger_ids):
        raise ReviewPacketError(f"{path}: prefix owner set references unknown ledger ids")
    source_jsonl = Path(str(selected.get("source_jsonl", ""))).expanduser().resolve(strict=True)
    image_path = _resolve_image_path(source_jsonl, image_id)
    image_sha = sha256_file(image_path)
    if image_sha != str(selected.get("image_sha256")):
        raise ReviewPacketError(f"{path}: source image SHA-256 mismatch")
    with Image.open(image_path) as opened:
        actual_size = opened.size
    if [actual_size[1], actual_size[0]] != [int(selected.get("height")), int(selected.get("width"))]:
        raise ReviewPacketError(f"{path}: source image dimensions disagree with artifact")
    frozen_kind = str(frozen.get("harmful_kind", ""))
    if frozen_kind not in {"duplicate", "premature_terminal"}:
        raise ReviewPacketError(f"{path}: unsupported frozen harmful_kind {frozen_kind!r}")
    frozen_ids, frozen_hash = _hash_ids(greedy.get("candidate_token_ids"), f"{path}: greedy candidate token_ids")
    if frozen_hash != str(greedy.get("candidate_token_ids_sha256")):
        raise ReviewPacketError(f"{path}: greedy candidate token hash mismatch")
    greedy_owner_ids = sorted({str(value) for value in greedy.get("strict_matched_owner_ids", [])})
    if frozen_kind == "duplicate" and len(greedy_owner_ids) != 1:
        raise ReviewPacketError(f"{path}: duplicate greedy row must have one strict owner")
    if frozen_kind == "premature_terminal" and greedy_owner_ids:
        raise ReviewPacketError(f"{path}: premature terminal row must have no strict owners")
    samples = artifact.get("samples")
    if not isinstance(samples, list):
        raise ReviewPacketError(f"{path}: samples must be a list")
    selected_samples: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping) or sample.get("selected_verified_uncovered_rescue") is not True:
            continue
        seed = sample.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ReviewPacketError(f"{path}: selected sample {index} has invalid seed")
        sample_prefix, sample_prefix_hash = _hash_ids(sample.get("prefix_token_ids"), f"{path}: sample prefix")
        if sample_prefix_hash != prefix_hash:
            raise ReviewPacketError(f"{path}: selected sample {seed} prefix differs from frozen prefix")
        candidate_ids, candidate_hash = _hash_ids(sample.get("candidate_token_ids"), f"{path}: sample {seed} candidate")
        if candidate_hash != str(sample.get("candidate_token_ids_sha256")):
            raise ReviewPacketError(f"{path}: sample {seed} candidate hash mismatch")
        owner_ids = sorted({str(value) for value in sample.get("strict_matched_owner_ids", [])})
        verified_ids = sorted({str(value) for value in sample.get("verified_uncovered_owner_ids", [])})
        if len(owner_ids) != 1 or owner_ids != verified_ids or owner_ids[0] not in set(uncovered):
            raise ReviewPacketError(f"{path}: selected sample {seed} owner is not one verified uncovered owner")
        prediction_box = _prediction_from_owner_matches(sample.get("owner_matches"), owner_ids[0], label=f"{path}: sample {seed}")
        selected_samples.append({
            "seed": seed,
            "candidate_token_ids": candidate_ids,
            "candidate_token_ids_sha256": candidate_hash,
            "owner_id": owner_ids[0],
            "predicted_norm1000_xyxy": prediction_box,
            "sample_index": index,
        })
    if not selected_samples:
        # It is useful to review a harmful row even when a seed block found no
        # rescue, but the packet must not silently turn it into a positive case.
        selected_samples = []
    greedy_box: list[float] | None = None
    if frozen_kind == "duplicate":
        greedy_box = _prediction_from_owner_matches(greedy.get("owner_matches"), greedy_owner_ids[0], label=f"{path}: greedy duplicate")
    return {
        "artifact_path": str(path.resolve()),
        "artifact_sha256": sha256_file(path),
        "image_id": image_id,
        "image_path": str(image_path),
        "image_sha256": image_sha,
        "image_width": actual_size[0],
        "image_height": actual_size[1],
        "source_jsonl": str(source_jsonl),
        "prompt_token_ids": prompt_ids,
        "prompt_token_ids_sha256": prompt_hash,
        "prefix_token_ids": prefix_ids,
        "prefix_token_ids_sha256": prefix_hash,
        "covered_owner_ids": covered,
        "uncovered_owner_ids": uncovered,
        "row_index": int(prefix.get("row_index")),
        "harmful_kind": frozen_kind,
        "ledger": ledger,
        "ledger_sha256": _ledger_hash(ledger),
        "checkpoint_id": str(artifact.get("source_checkpoint_identity", {}).get("checkpoint_id", "")),
        "greedy": {
            "candidate_token_ids": frozen_ids,
            "candidate_token_ids_sha256": frozen_hash,
            "strict_matched_owner_ids": greedy_owner_ids,
            "predicted_norm1000_xyxy": greedy_box,
        },
        "selected_samples": selected_samples,
    }


def _group_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        item["image_id"], item["prefix_token_ids_sha256"], item["harmful_kind"], item["row_index"], item["ledger_sha256"], item["image_sha256"], item["checkpoint_id"],
    )


def _review(case_id: str, role: str, *, owner_id: str | None = None) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "role": role,
        "entity_review_status": "pending",
        "geometry_review_status": "pending",
        "accepted_entity_id": None,
        "accepted_coordinate_bins": None,
        "reviewer": None,
        "comment": None,
        "owner_id_from_sampler": owner_id,
    }


def build_packet(
    artifact_paths: Sequence[Path],
    output_dir: Path,
    *,
    crop_halo: int = DEFAULT_CROP_HALO,
    resize_factor: int = DEFAULT_RESIZE_FACTOR,
) -> dict[str, Any]:
    if not artifact_paths:
        raise ReviewPacketError("at least one --artifact is required")
    items = [_validate_artifact(path.resolve(strict=True)) for path in artifact_paths]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for item in items:
        groups.setdefault(_group_key(item), []).append(item)
    output_dir.mkdir(parents=True, exist_ok=True)
    packet_cases: list[dict[str, Any]] = []
    packet_groups: list[dict[str, Any]] = []
    for group_index, key in enumerate(sorted(groups, key=str)):
        group = sorted(groups[key], key=lambda item: item["artifact_path"])
        first = group[0]
        output_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"image-{first['image_id']}-row-{first['row_index']}-{first['harmful_kind']}")
        image = Image.open(first["image_path"]).convert("RGB")
        ledger = first["ledger"]
        overlays: list[dict[str, Any]] = []
        for item in group:
            if item["greedy"]["predicted_norm1000_xyxy"] is not None:
                overlays.append({
                    "bbox_norm1000": item["greedy"]["predicted_norm1000_xyxy"],
                    "color": HARMFUL_COLOR,
                    "label": f"greedy:{item['greedy']['strict_matched_owner_ids'][0]}",
                })
        selected = []
        seen_sample_keys: set[tuple[int, str]] = set()
        for item in group:
            for sample in item["selected_samples"]:
                sample_key = (int(sample["seed"]), str(sample["candidate_token_ids_sha256"]))
                if sample_key in seen_sample_keys:
                    continue
                seen_sample_keys.add(sample_key)
                selected.append((item, sample))
                overlays.append({
                    "bbox_norm1000": sample["predicted_norm1000_xyxy"],
                    "color": RESCUE_COLOR,
                    "label": f"rescue:{sample['seed']}:{sample['owner_id']}",
                })
        focus_boxes: list[Sequence[float]] = [
            owner["bbox_norm1000"] for owner in ledger if owner["entity_id"] in set(first["uncovered_owner_ids"])
        ]
        focus_boxes.extend(overlay["bbox_norm1000"] for overlay in overlays)
        if not focus_boxes:
            focus_boxes = [ledger[0]["bbox_norm1000"]]
        full_path = output_dir / f"{output_stem}.full.png"
        crop_path = output_dir / f"{output_stem}.crop.png"
        _draw_full(image, ledger=ledger, overlays=overlays, output=full_path)
        crop_bounds = _draw_crop(image, ledger=ledger, overlays=overlays, focus_boxes=focus_boxes, output=crop_path, halo=crop_halo, resize_factor=resize_factor)
        def case_artifacts(case_stem: str, case_overlays: Sequence[Mapping[str, Any]], case_focus: Sequence[Sequence[float]]) -> dict[str, Any]:
            case_full = output_dir / f"{case_stem}.full.png"
            case_crop = output_dir / f"{case_stem}.crop.png"
            _draw_full(image, ledger=ledger, overlays=case_overlays, output=case_full)
            case_bounds = _draw_crop(
                image,
                ledger=ledger,
                overlays=case_overlays,
                focus_boxes=case_focus,
                output=case_crop,
                halo=crop_halo,
                resize_factor=resize_factor,
            )
            return {
                "full_image_png": str(case_full.resolve()),
                "crop_png": str(case_crop.resolve()),
                "full_image_png_sha256": sha256_file(case_full),
                "crop_png_sha256": sha256_file(case_crop),
                "crop_bounds_px_xyxy": list(case_bounds),
                "crop_halo_pixels": crop_halo,
                "crop_resize_factor": resize_factor,
            }
        packet_groups.append({
            "group_index": group_index,
            "image_id": first["image_id"],
            "row_index": first["row_index"],
            "harmful_kind": first["harmful_kind"],
            "prefix_token_ids": first["prefix_token_ids"],
            "prefix_token_ids_sha256": first["prefix_token_ids_sha256"],
            "covered_owner_ids": first["covered_owner_ids"],
            "uncovered_owner_ids": first["uncovered_owner_ids"],
            "ledger": ledger,
            "ledger_sha256": first["ledger_sha256"],
            "artifact_paths": [item["artifact_path"] for item in group],
            "artifact_sha256": [item["artifact_sha256"] for item in group],
            "selected_sample_count": len(selected),
            "artifacts": {
                "full_image_png": str(full_path.resolve()),
                "crop_png": str(crop_path.resolve()),
                "full_image_png_sha256": sha256_file(full_path),
                "crop_png_sha256": sha256_file(crop_path),
                "crop_bounds_px_xyxy": list(crop_bounds),
                "crop_halo_pixels": crop_halo,
                "crop_resize_factor": resize_factor,
            },
        })
        harmful_item = group[0]
        harmful_case_id = f"{output_stem}-greedy-harmful"
        harmful_overlays = [
            overlay
            for overlay in overlays
            if str(overlay.get("label", "")).startswith("greedy:")
        ]
        harmful_focus = [
            owner["bbox_norm1000"]
            for owner in ledger
            if owner["entity_id"] in set(first["uncovered_owner_ids"])
        ]
        harmful_focus.extend(overlay["bbox_norm1000"] for overlay in harmful_overlays)
        if not harmful_focus:
            harmful_focus = [ledger[0]["bbox_norm1000"]]
        packet_cases.append({
            "case_id": harmful_case_id,
            "role": "frozen_greedy_harmful_row",
            "group_index": group_index,
            "image_id": first["image_id"],
            "image_path": first["image_path"],
            "image_sha256": first["image_sha256"],
            "row_index": first["row_index"],
            "harmful_kind": first["harmful_kind"],
            "prefix_token_ids": first["prefix_token_ids"],
            "prefix_token_ids_sha256": first["prefix_token_ids_sha256"],
            "prompt_token_ids_sha256": first["prompt_token_ids_sha256"],
            "covered_owner_ids": first["covered_owner_ids"],
            "uncovered_owner_ids": first["uncovered_owner_ids"],
            "greedy": first["greedy"],
            "artifact_paths": [item["artifact_path"] for item in group],
            "review": _review(harmful_case_id, "frozen_greedy_harmful_row", owner_id=(first["greedy"]["strict_matched_owner_ids"] or [None])[0]),
            "artifacts": case_artifacts(
                harmful_case_id,
                harmful_overlays,
                harmful_focus,
            ),
        })
        for item, sample in selected:
            case_id = f"{output_stem}-sample-rescue-seed-{sample['seed']}-{sample['candidate_token_ids_sha256'][:12]}"
            owner = next(owner for owner in ledger if owner["entity_id"] == sample["owner_id"])
            rescue_overlay = {
                "bbox_norm1000": sample["predicted_norm1000_xyxy"],
                "color": RESCUE_COLOR,
                "label": f"rescue:{sample['seed']}",
            }
            rescue_focus = [owner["bbox_norm1000"], sample["predicted_norm1000_xyxy"]]
            packet_cases.append({
                "case_id": case_id,
                "role": "selected_verified_uncovered_rescue",
                "group_index": group_index,
                "image_id": first["image_id"],
                "image_path": first["image_path"],
                "image_sha256": first["image_sha256"],
                "row_index": first["row_index"],
                "harmful_kind": first["harmful_kind"],
                "owner_id": sample["owner_id"],
                "owner_description": owner["description"],
                "owner_reference_norm1000_xyxy": owner["bbox_norm1000"],
                "predicted_norm1000_xyxy": sample["predicted_norm1000_xyxy"],
                "candidate_token_ids": sample["candidate_token_ids"],
                "candidate_token_ids_sha256": sample["candidate_token_ids_sha256"],
                "prefix_token_ids": first["prefix_token_ids"],
                "prefix_token_ids_sha256": first["prefix_token_ids_sha256"],
                "prompt_token_ids_sha256": first["prompt_token_ids_sha256"],
                "source_artifact_path": item["artifact_path"],
                "source_artifact_sha256": item["artifact_sha256"],
                "review": _review(case_id, "selected_verified_uncovered_rescue", owner_id=sample["owner_id"]),
                "artifacts": case_artifacts(case_id, [rescue_overlay], rescue_focus),
            })
    packet = {
        "schema_version": SCHEMA_VERSION,
        "input_artifacts": [str(path.resolve()) for path in sorted(artifact_paths, key=lambda path: str(path))],
        "input_artifact_sha256": [sha256_file(path.resolve(strict=True)) for path in sorted(artifact_paths, key=lambda path: str(path))],
        "never_auto_admit": True,
        "review_contract": {
            "entity_review_status": "pending",
            "geometry_review_status": "pending",
            "accepted_coordinate_bins_require_human_review": True,
            "sampled_rows_are_candidates_not_training_labels": True,
        },
        "groups": packet_groups,
        "cases": sorted(packet_cases, key=lambda item: item["case_id"]),
    }
    packet_path = output_dir / "review-packet.json"
    packet_path.write_text(json.dumps(packet, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, action="append", required=True, help="exact_prefix_sampled_rescue.v1 JSON artifact; repeat for seed blocks")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crop-halo", type=int, default=DEFAULT_CROP_HALO)
    parser.add_argument("--resize-factor", type=int, default=DEFAULT_RESIZE_FACTOR)
    args = parser.parse_args()
    if args.crop_halo < 0 or args.resize_factor < 1:
        raise SystemExit("--crop-halo must be >=0 and --resize-factor must be >=1")
    try:
        build_packet(args.artifact, args.output_dir, crop_halo=args.crop_halo, resize_factor=args.resize_factor)
    except (OSError, ReviewPacketError) as exc:
        raise SystemExit(f"review packet build failed: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
