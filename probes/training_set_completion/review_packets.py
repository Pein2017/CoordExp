"""Prepare one-image visual review packets for stage-01 acquisition rows.

This module is CPU-only and deliberately keeps every parsed prediction and
every parser-dropped proposal visible.  It renders viewing aids; it does not
make owner, class, or truth decisions.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

WORKTREE = Path(__file__).resolve().parents[2]
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

from probes.training_set_completion import artifacts as artifact_primitives
from src.data.geometry import coord_bins_to_pixel_xyxy, iou_xyxy


BASE = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration")
DEFAULT_ACQUISITION = BASE / (
    "2026-09-14-training-set-completion-curriculum/"
    "stage01-acquisition-v1-retry1-config-batch2/rows.jsonl"
)
DEFAULT_SUPPORT = BASE / (
    "2026-09-14-training-set-completion-curriculum/"
    "stage01-support-seed-v1/candidate-support-ledger-v1.json"
)
DEFAULT_OUTPUT = BASE / (
    "2026-09-14-training-set-completion-curriculum/stage01-review-packets-v1"
)
IMAGE_IDS = (25274, 59571, 99937, 210457, 219546, 323322, 351017, 388795, 417044, 477415, 528944)
SCHEMA = "training_set_completion.stage01_review_packets.v1"
CAP = 3084


_canonical = artifact_primitives.canonical
_file_hash = artifact_primitives.file_hash
_binding = artifact_primitives.binding


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _policy(row: Mapping[str, Any]) -> str:
    request = row["request"]
    temperature = float(request["temperature"])
    return "greedy-t0p0" if request["kind"] == "greedy" else f"sample-t{temperature:g}".replace(".", "p")


def _request_order(row: Mapping[str, Any]) -> tuple[int, int]:
    request = row["request"]
    return (0 if request["kind"] == "greedy" else 1, int(round(float(request["temperature"]) * 10)))


def _coord_bins(value: Any) -> list[int] | None:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            result = [int(x) for x in value]
        except (TypeError, ValueError):
            return None
        if all(0 <= x <= 1000 for x in result):
            return result
    return None


def _coord_bins_from_drop(drop: Mapping[str, Any]) -> list[int] | None:
    spans = drop.get("coord_token_spans")
    if not isinstance(spans, list) or len(spans) != 4:
        return None
    values: list[int] = []
    for span in spans:
        text = span.get("text") if isinstance(span, Mapping) else None
        if not isinstance(text, str) or not text.startswith("<|coord_") or not text.endswith("|>"):
            return None
        try:
            values.append(int(text[len("<|coord_") : -len("|>")]))
        except ValueError:
            return None
    return values if all(0 <= x <= 1000 for x in values) else None


def _pixel_from_bins(bins: Sequence[int], width: int, height: int) -> list[int]:
    return list(coord_bins_to_pixel_xyxy(bins, image_width=width, image_height=height, field="coord_bins_1000"))


def _pixel_from_bins_for_viewing(bins: Sequence[int], width: int, height: int) -> list[int]:
    """Convert even order-invalid raw bins so parser drops stay drawable."""
    return [round(int(bins[0]) * width / 1000), round(int(bins[1]) * height / 1000),
            round(int(bins[2]) * width / 1000), round(int(bins[3]) * height / 1000)]


def _safe_iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1, y1, x2, y2 = (float(x) for x in left)
    if x1 >= x2 or y1 >= y2:
        return 0.0
    return float(iou_xyxy(tuple(left), tuple(right)))


def _nearest(box: Sequence[float] | None, gt: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if box is None:
        return {"nearest_gt_owner_id": None, "nearest_gt_iou": None, "second_nearest_gt_iou": None}
    values = sorted(
        ((_safe_iou(box, g["bbox_pixel_xyxy"]), str(g["owner_id"])) for g in gt),
        key=lambda item: (-item[0], item[1]),
    )
    if not values:
        return {"nearest_gt_owner_id": None, "nearest_gt_iou": 0.0, "second_nearest_gt_iou": 0.0}
    return {
        "nearest_gt_owner_id": values[0][1],
        "nearest_gt_iou": round(values[0][0], 8),
        "second_nearest_gt_iou": round(values[1][0], 8) if len(values) > 1 else 0.0,
    }


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _draw_box(draw: ImageDraw.ImageDraw, box: Sequence[float], colour: tuple[int, int, int], label: str, width: int = 3) -> None:
    x1, y1, x2, y2 = (round(float(x)) for x in box)
    if x1 < x2 and y1 < y2:
        draw.rectangle((x1, y1, x2, y2), outline=colour, width=width)
    else:
        # Keep invalid order visible as a cross and its normalized envelope.
        draw.rectangle((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)), outline=colour, width=width)
        draw.line((x1, y1, x2, y2), fill=colour, width=width)
        draw.line((x1, y2, x2, y1), fill=colour, width=width)
    draw.text((max(0, min(x1, x2) + 2), max(0, min(y1, y2) + 2)), label, fill=colour, font=_font(12))


def _crop(image: Image.Image, box: Sequence[float], path: Path) -> None:
    width, height = image.size
    x1, y1, x2, y2 = (int(round(float(x))) for x in box)
    left, right = max(0, min(x1, x2)), min(width, max(x1, x2))
    top, bottom = max(0, min(y1, y2)), min(height, max(y1, y2))
    if right <= left:
        right = min(width, left + 1)
    if bottom <= top:
        bottom = min(height, top + 1)
    image.crop((left, top, right, bottom)).save(path, format="PNG", optimize=False)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_source_line_number"] = line_no
            row["_source_line_sha256"] = hashlib.sha256(line.encode()).hexdigest()
            rows.append(row)
    _require(len(rows) == 44, f"expected 44 acquisition rows, got {len(rows)}")
    _require(len({r["request"]["request_id"] for r in rows}) == 44, "duplicate request IDs")
    return rows


def _gt_for_image(support: Mapping[str, Any], image_id: int, width: int, height: int) -> list[dict[str, Any]]:
    image = next((x for x in support["images"] if int(x["image_id"]) == image_id), None)
    _require(image is not None, f"support ledger missing image {image_id}")
    result = []
    for owner in image["gt_owners"]:
        bins = owner["bbox_coord_canvas"]["coord_bins_1000"]
        result.append({
            "owner_id": str(owner["owner_id"]),
            "category_name": owner["category_name"],
            "is_crowd": bool(owner["is_crowd"]),
            "owner_role": owner["owner_role"],
            "coord_bins_1000": list(bins),
            "bbox_pixel_xyxy": _pixel_from_bins(bins, width, height),
        })
    return result


def _proposal_from_valid(obj: Mapping[str, Any], *, row: Mapping[str, Any], index: int, gt: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    bins = _coord_bins(obj.get("coord_bins"))
    box = list(obj.get("bbox", [])) if isinstance(obj.get("bbox"), list) else None
    _require(bins is not None and box is not None and len(box) == 4, "valid prediction lacks bins/pixel bbox")
    expected = _pixel_from_bins(bins, row["parsed"]["image_width"], row["parsed"]["image_height"])
    _require(expected == [int(x) for x in box], f"coord_bins/pixel bbox conversion mismatch for {row['request']['request_id']}:{index}")
    nearest = _nearest(box, gt)
    return {
        "proposal_id": f"{row['request']['request_id']}:p{obj.get('generated_order', index)}",
        "source_request_id": row["request"]["request_id"],
        "source_line_number_1_based": row["_source_line_number"],
        "source_line_sha256": row["_source_line_sha256"],
        "policy": _policy(row),
        "temperature": row["request"]["temperature"],
        "seed": row["request"]["seed"],
        "generated_order": obj.get("generated_order", index),
        "status": "parsed_valid",
        "description": obj.get("description"),
        "coord_bins_1000": bins,
        "bbox_pixel_xyxy": box,
        "raw_span_sha256": obj.get("raw_span_sha256"),
        **nearest,
    }


def _proposal_from_drop(drop: Mapping[str, Any], *, row: Mapping[str, Any], gt: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    bins = _coord_bins_from_drop(drop)
    box = _pixel_from_bins_for_viewing(bins, row["parsed"]["image_width"], row["parsed"]["image_height"]) if bins else None
    nearest = _nearest(box, gt)
    return {
        "proposal_id": f"{row['request']['request_id']}:p{drop.get('generated_order', 'drop')}",
        "source_request_id": row["request"]["request_id"],
        "source_line_number_1_based": row["_source_line_number"],
        "source_line_sha256": row["_source_line_sha256"],
        "policy": _policy(row),
        "temperature": row["request"]["temperature"],
        "seed": row["request"]["seed"],
        "generated_order": drop.get("generated_order"),
        "status": "parser_dropped_raw_visible",
        "drop_reason": drop.get("reason"),
        "drop_code": drop.get("code"),
        "description": (drop.get("raw_text") or "").split("<|object_ref_end|>")[0].replace("<|object_ref_start|>", "") or None,
        "coord_bins_1000": bins,
        "bbox_pixel_xyxy": box,
        "raw_span_sha256": drop.get("raw_span_sha256"),
        "raw_text": drop.get("raw_text"),
        **nearest,
    }


def _flags(proposals: list[dict[str, Any]]) -> None:
    for proposal in proposals:
        flags: list[str] = []
        if proposal["status"] == "parser_dropped_raw_visible":
            flags.append("invalid" if proposal.get("drop_reason") == "geometry_invalid" else "malformed")
        iou = proposal.get("nearest_gt_iou")
        if iou is not None and iou < 0.5:
            flags.append("gt_unmatched")
        if proposal.get("second_nearest_gt_iou", 0.0) > 0.0 and iou is not None and abs(iou - proposal["second_nearest_gt_iou"]) <= 0.05:
            flags.append("ambiguous_nearest_gt")
        proposal["review_flags"] = flags
    valid = [p for p in proposals if p["bbox_pixel_xyxy"] is not None]
    for index, proposal in enumerate(valid):
        repeated = [other["proposal_id"] for other in valid[:index] if _safe_iou(proposal["bbox_pixel_xyxy"], other["bbox_pixel_xyxy"]) > 0.95]
        if repeated:
            proposal["repeated_of"] = repeated
            proposal["review_flags"].append("repeated")


def _render_gt(image: Image.Image, gt: Sequence[Mapping[str, Any]], path: Path, image_id: int) -> None:
    canvas = image.copy(); draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), f"GT overlay image={image_id} atomic/group preserved", fill=(0, 0, 0), font=_font(18))
    for owner in gt:
        colour = (180, 0, 180) if owner["is_crowd"] else (0, 170, 70)
        _draw_box(draw, owner["bbox_pixel_xyxy"], colour, f"G:{owner['owner_id']} {owner['category_name']}")
    canvas.save(path, format="PNG", optimize=False)


def _render_policy(image: Image.Image, gt: Sequence[Mapping[str, Any]], proposals: Sequence[Mapping[str, Any]], path: Path, image_id: int, policy: str) -> None:
    canvas = image.copy(); draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), f"Prediction overlay image={image_id} policy={policy} raw proposals={len(proposals)}", fill=(0, 0, 0), font=_font(18))
    # Light GT outlines establish geometry without assigning proposal truth.
    for owner in gt:
        _draw_box(draw, owner["bbox_pixel_xyxy"], (130, 130, 130), f"G:{owner['owner_id']}", width=2)
    for proposal in proposals:
        box = proposal.get("bbox_pixel_xyxy")
        if box is None:
            continue
        flags = set(proposal.get("review_flags", []))
        colour = (235, 45, 45) if "invalid" in flags else (245, 145, 0) if "repeated" in flags else (120, 70, 200) if "gt_unmatched" in flags else (0, 100, 220)
        label = f"P{proposal.get('generated_order')} {proposal.get('description') or '?'}"
        if "invalid" in flags:
            label = "INVALID " + label
        _draw_box(draw, box, colour, label)
    canvas.save(path, format="PNG", optimize=False)


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_canonical(value))


def build(*, acquisition: Path = DEFAULT_ACQUISITION, support: Path = DEFAULT_SUPPORT, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    _require(not output.exists(), f"refusing to overwrite existing output: {output}")
    acquisition = acquisition.resolve(strict=True); support = support.resolve(strict=True)
    rows = _load_rows(acquisition); support_data = json.loads(support.read_text())
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_image[int(row["image_id"])].append(row)
    _require(list(sorted(by_image)) == sorted(IMAGE_IDS), "acquisition image cohort mismatch")
    _require(all(len(by_image[i]) == 4 for i in IMAGE_IDS), "not exactly four policies per image")
    output.mkdir(parents=True)
    packets = []
    total_proposals = total_invalid = total_crops = 0
    for ordinal, image_id in enumerate(IMAGE_IDS):
        image_rows = sorted(by_image[image_id], key=_request_order)
        first = image_rows[0]["parsed"]
        image_path = Path(first["image_path"]).resolve(strict=True)
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        _require([image.width, image.height] == [first["image_width"], first["image_height"]], f"image dimensions differ for {image_id}")
        gt = _gt_for_image(support_data, image_id, image.width, image.height)
        packet_dir = output / f"image-{image_id:012d}"
        packet_dir.mkdir()
        gt_path = packet_dir / "gt-overlay.png"; _render_gt(image, gt, gt_path, image_id)
        policy_payloads = []; all_proposals: list[dict[str, Any]] = []
        for row in image_rows:
            parsed = row["parsed"]; proposals = [_proposal_from_valid(obj, row=row, index=index, gt=gt) for index, obj in enumerate(parsed["pred"])]
            drops = parsed.get("dropped_predictions", [])
            proposals.extend(_proposal_from_drop(drop, row=row, gt=gt) for drop in drops)
            proposals.sort(key=lambda p: (p["generated_order"] if isinstance(p["generated_order"], int) else CAP + 1, p["proposal_id"]))
            _flags(proposals)
            policy = _policy(row); overlay_path = packet_dir / f"pred-overlay-{policy}.png"; _render_policy(image, gt, proposals, overlay_path, image_id, policy)
            for p in proposals:
                flags = set(p["review_flags"])
                if flags.intersection({"gt_unmatched", "ambiguous_nearest_gt", "repeated", "invalid"}) and p.get("bbox_pixel_xyxy") is not None:
                    crop_path = packet_dir / "crops" / f"{policy}-p{p['generated_order']}-{'_'.join(sorted(flags))}.png"
                    crop_path.parent.mkdir(exist_ok=True)
                    _crop(image, p["bbox_pixel_xyxy"], crop_path); p["crop_path"] = str(crop_path.resolve()); total_crops += 1
                p["source_row"] = {"request_id": row["request"]["request_id"], "line_number_1_based": row["_source_line_number"], "line_sha256": row["_source_line_sha256"]}
            all_proposals.extend(proposals); total_proposals += len(proposals); total_invalid += sum("invalid" in p["review_flags"] for p in proposals)
            policy_payloads.append({"policy": policy, "request": row["request"], "source_row": {"line_number_1_based": row["_source_line_number"], "line_sha256": row["_source_line_sha256"]}, "parsed_counts": {"valid": parsed["valid_prediction_count"], "dropped": parsed["dropped_prediction_count"], "raw_visible": len(proposals), "stop_reason": row["decode_stop_reason"]}, "overlay_path": str(overlay_path.resolve()), "proposals": proposals})
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for p in all_proposals:
            groups[(p.get("description"), tuple(p.get("coord_bins_1000") or ()), p.get("status"))].append(p)
        identical = [{"signature": {"description": key[0], "coord_bins_1000": list(key[1]), "status": key[2]}, "proposal_ids": [p["proposal_id"] for p in vals], "provenance": [{"request_id": p["source_request_id"], "temperature": p["temperature"], "seed": p["seed"], "source_line_number_1_based": p["source_line_number_1_based"]} for p in vals]} for key, vals in sorted(groups.items(), key=lambda x: str(x[0])) if len(vals) > 1]
        overview = packet_dir / "overview.png"; _render_policy(image, gt, all_proposals, overview, image_id, "all-four-policies-viewing-overview")
        packet = {"schema": SCHEMA, "status": "candidate_visual_review_packet_ready", "ordinal": ordinal, "image_id": image_id, "row_id": f"coco2017_train_{image_id:012d}", "image": {"path": str(image_path), "sha256": _file_hash(image_path), "size": [image.width, image.height]}, "gt": gt, "gt_overlay_path": str(gt_path.resolve()), "policies": policy_payloads, "identical_proposal_groups_viewing_aid": identical, "overview_path": str(overview.resolve()), "review_boundary": "Rendering only; no owner/class/truth decision. Parser-dropped raw rows remain visible in JSON and are drawn when coordinate bins exist."}
        _write_json(packet_dir / "packet.json", packet); packets.append({"image_id": image_id, "path": str((packet_dir / "packet.json").resolve()), "overview_path": str(overview.resolve()), "gt_overlay_path": str(gt_path.resolve()), "policy_count": len(policy_payloads), "raw_proposal_count": len(all_proposals), "invalid_raw_count": sum("invalid" in p["review_flags"] for p in all_proposals)})
    receipt = {"schema": SCHEMA, "status": "candidate_packets_ready", "acquisition": _binding(acquisition), "support_ledger": _binding(support), "cohort_image_ids": list(IMAGE_IDS), "image_count": len(packets), "policy_count": 44, "raw_proposal_count": total_proposals, "geometry_invalid_raw_count": total_invalid, "crop_count": total_crops, "packets": packets, "no_truth_decisions": True, "code_sources": {"producer": _binding(Path(__file__)), "artifact_primitives": _binding(Path(artifact_primitives.__file__))}}
    receipt["receipt_sha256"] = artifact_primitives.digest({k: v for k, v in receipt.items() if k != "receipt_sha256"})
    _write_json(output / "receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition", type=Path, default=DEFAULT_ACQUISITION)
    parser.add_argument("--support-ledger", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(acquisition=args.acquisition, support=args.support_ledger, output=args.output), indent=2))


if __name__ == "__main__":
    main()
