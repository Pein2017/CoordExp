"""Render per-image readback review packets for first-fit preparation.

This is a CPU-only viewing aid. It preserves every scored raw row, including
parser-dropped rows and order-invalid axes. It makes no owner, class, truth,
matching, or first-fit decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont


COHORT = (25274, 59571, 99937, 210457, 219546, 323322, 351017, 388795, 417044, 477415, 528944)


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def binding(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "sha256": file_hash(path), "size_bytes": path.stat().st_size}


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def font(size: int = 14) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_raw_box(draw: ImageDraw.ImageDraw, box: Sequence[float], colour: tuple[int, int, int], label: str) -> None:
    """Draw raw axes without sorting them; invalid order is shown as a cross."""
    x1, y1, x2, y2 = (round(float(value)) for value in box)
    if x1 < x2 and y1 < y2:
        draw.rectangle((x1, y1, x2, y2), outline=colour, width=3)
    else:
        draw.rectangle((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)), outline=colour, width=3)
        draw.line((x1, y1, x2, y2), fill=colour, width=3)
        draw.line((x1, y2, x2, y1), fill=colour, width=3)
    draw.text((max(0, min(x1, x2) + 2), max(0, min(y1, y2) + 2)), label, fill=colour, font=font(13))


def bins_to_pixels(bins: Sequence[float], width: int, height: int) -> list[int]:
    require(len(bins) == 4, "coordinate bins must have four axes")
    return [round(float(bins[0]) * width / 1000), round(float(bins[1]) * height / 1000),
            round(float(bins[2]) * width / 1000), round(float(bins[3]) * height / 1000)]


def raw_box(row: Mapping[str, Any], width: int, height: int) -> list[int] | None:
    box = row.get("bbox_pixel_xyxy")
    if isinstance(box, list) and len(box) == 4:
        return [round(float(value)) for value in box]
    bins = row.get("coord_bins_1000")
    if isinstance(bins, list) and len(bins) == 4:
        return bins_to_pixels(bins, width, height)
    return None


def crop_frame(box: Sequence[float] | None, width: int, height: int, padding: int) -> tuple[int, int, int, int]:
    if box is None:
        return (0, 0, width, height)
    x1, y1, x2, y2 = (float(value) for value in box)
    left = max(0, min(width - 1, round(min(x1, x2)) - padding))
    top = max(0, min(height - 1, round(min(y1, y2)) - padding))
    right = min(width, max(left + 1, round(max(x1, x2)) + padding + 1))
    bottom = min(height, max(top + 1, round(max(y1, y2)) + padding + 1))
    return (left, top, right, bottom)


def render_target_overlay(image: Image.Image, refs: Sequence[Mapping[str, Any]], path: Path, image_id: int) -> None:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), f"Target catalog overlay image={image_id} references={len(refs)}", fill=(0, 0, 0), font=font(18))
    for ref in refs:
        box = ref["bbox_pixel_xyxy"]
        colour = (0, 170, 70) if ref.get("role") == "gt_atomic" else (180, 0, 180)
        draw_raw_box(draw, box, colour, f"T:{ref['owner_id']} {ref.get('category') or '?'}")
    canvas.save(path, format="PNG", optimize=False)


def render_raw_overlay(image: Image.Image, rows: Sequence[Mapping[str, Any]], path: Path, image_id: int) -> None:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), f"Raw generated overlay image={image_id} rows={len(rows)}", fill=(0, 0, 0), font=font(18))
    for row in rows:
        box = row.get("_raw_box_pixel_xyxy")
        label = f"{row.get('prediction_id', 'p?')} {row.get('description') or '?'}"
        if row.get("status") not in {"parsed_valid"}:
            label = "RAW INVALID " + label
        if box is None:
            draw.text((8, 32 + 18 * int(row.get("generated_order") or 0)), label + " (no axes)", fill=(235, 45, 45), font=font(13))
            continue
        colour = (0, 100, 220) if row.get("status") == "parsed_valid" else (235, 45, 45)
        draw_raw_box(draw, box, colour, label)
    canvas.save(path, format="PNG", optimize=False)


def render_crop(image: Image.Image, row: Mapping[str, Any], path: Path, padding: int, kind: str) -> tuple[int, int, int, int]:
    box = row.get("_raw_box_pixel_xyxy")
    frame = crop_frame(box, image.width, image.height, padding)
    left, top, right, bottom = frame
    canvas = image.crop(frame)
    draw = ImageDraw.Draw(canvas)
    if box is not None:
        shifted = [float(box[0]) - left, float(box[1]) - top, float(box[2]) - left, float(box[3]) - top]
        label = f"{row.get('prediction_id', 'p?')} {row.get('description') or '?'} [{kind}]"
        if row.get("status") != "parsed_valid":
            label = "RAW INVALID " + label
        draw_raw_box(draw, shifted, (0, 100, 220) if row.get("status") == "parsed_valid" else (235, 45, 45), label)
    else:
        draw.text((2, 2), f"{row.get('prediction_id', 'p?')} {row.get('description') or '?'} [{kind}] no axes", fill=(235, 45, 45), font=font(13))
    canvas.save(path, format="PNG", optimize=False)
    return frame


def load_sources(scored_path: Path, acquisition_path: Path | None, target_path: Path | None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path, Path]:
    scored = json.loads(scored_path.read_text())
    require(scored.get("schema", "").startswith("training_set_completion.readback_selector_preparation"), "unsupported scored schema")
    source = scored.get("sources", {})
    acquisition_path = (acquisition_path or Path(source["acquisition_manifest"]["path"])).resolve(strict=True)
    target_path = (target_path or Path(source["target_owners"]["path"])).resolve(strict=True)
    acquisition = json.loads(acquisition_path.read_text())
    target = json.loads(target_path.read_text())
    require(isinstance(scored.get("rows"), list), "scored rows must be a list")
    return scored, acquisition, target, acquisition_path, target_path


def build(*, scored_path: Path, output: Path, acquisition_path: Path | None = None, target_path: Path | None = None) -> dict[str, Any]:
    scored_path = scored_path.resolve(strict=True)
    require(not output.exists(), f"refusing to overwrite existing output: {output}")
    scored, acquisition, target, acquisition_path, target_path = load_sources(scored_path, acquisition_path, target_path)
    acquisition_records = {int(row["image_id"]): row for row in acquisition["records"]}
    target_by_image: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in COHORT}
    for ref in target["records"]:
        image_id = int(ref["image_id"])
        require(image_id in target_by_image, f"target catalog image outside cohort: {image_id}")
        bins = ref.get("reference_coord_bins_1000")
        require(isinstance(bins, list) and len(bins) == 4, f"target reference missing coords: {image_id}:{ref.get('owner_id')}")
        target_by_image[image_id].append(dict(ref))
    scored_rows = scored["rows"]
    seen: set[int] = set()
    output.mkdir(parents=True)
    packet_summaries = []
    for row_index, scored_row in enumerate(scored_rows):
        image_id = int(scored_row["image_id"])
        require(image_id in acquisition_records, f"scored image absent from acquisition manifest: {image_id}")
        require(image_id not in seen, f"duplicate scored image row: {image_id}")
        seen.add(image_id)
        image_record = acquisition_records[image_id]
        image_path = Path(image_record["image_file"]["path"]).resolve(strict=True)
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
        width, height = image.size
        require([width, height] == [int(image_record["case"]["image_width"]), int(image_record["case"]["image_height"])], f"image dimensions differ: {image_id}")
        refs = []
        for ref in target_by_image[image_id]:
            item = dict(ref)
            item["bbox_pixel_xyxy"] = bins_to_pixels(ref["reference_coord_bins_1000"], width, height)
            refs.append(item)
        raw_rows = []
        for source_row in scored_row.get("raw_rows", []):
            item = dict(source_row)
            item["_raw_box_pixel_xyxy"] = raw_box(item, width, height)
            item["raw_axes_preserved"] = item.get("_raw_box_pixel_xyxy") is not None
            raw_rows.append(item)
        raw_rows.sort(key=lambda item: (int(item.get("generated_order", 10**9)), str(item.get("prediction_id", ""))))
        packet_dir = output / f"image-{image_id:012d}"
        crops_dir = packet_dir / "crops"
        crops_dir.mkdir(parents=True)
        target_overlay = packet_dir / "target-catalog-overlay.png"
        raw_overlay = packet_dir / "raw-generated-overlay.png"
        render_target_overlay(image, refs, target_overlay, image_id)
        render_raw_overlay(image, raw_rows, raw_overlay, image_id)
        rendered_rows = []
        for item in raw_rows:
            prediction_id = str(item.get("prediction_id", f"p{item.get('generated_order', 'unknown')}"))
            stem = prediction_id.replace("/", "_")
            tight_path = crops_dir / f"{stem}-tight.png"
            context_path = crops_dir / f"{stem}-context.png"
            tight_frame = render_crop(image, item, tight_path, 8, "tight")
            context_frame = render_crop(image, item, context_path, 64, "context")
            rendered_rows.append({
                "prediction_id": prediction_id,
                "generated_order": item.get("generated_order"),
                "status": item.get("status"),
                "drop_reason": item.get("drop_reason"),
                "description": item.get("description"),
                "coord_bins_1000": item.get("coord_bins_1000"),
                "raw_bbox_pixel_xyxy": item.get("bbox_pixel_xyxy"),
                "render_box_pixel_xyxy": item.get("_raw_box_pixel_xyxy"),
                "raw_axes_preserved": item.get("_raw_box_pixel_xyxy") is not None,
                "tight_crop": {"path": str(tight_path.resolve()), "frame_pixel_xyxy": list(tight_frame), "sha256": file_hash(tight_path)},
                "context_crop": {"path": str(context_path.resolve()), "frame_pixel_xyxy": list(context_frame), "sha256": file_hash(context_path)},
            })
        packet = {
            "schema": "training_set_completion.first_fit_visualization_packet.v1",
            "status": "diagnostic_visualization_only",
            "image_id": image_id,
            "scored_row_index": row_index,
            "scored_row_sha256": hashlib.sha256(canonical(scored_row)).hexdigest(),
            "route_id": scored_row.get("route_id"),
            "source": {"scored_json": binding(scored_path), "acquisition_manifest": binding(acquisition_path), "target_catalog": binding(target_path), "original_image": binding(image_path), "dimensions": [width, height]},
            "target_catalog_references": [{"owner_id": ref["owner_id"], "role": ref.get("role"), "category": ref.get("category"), "reference_coord_bins_1000": ref["reference_coord_bins_1000"], "bbox_pixel_xyxy": ref["bbox_pixel_xyxy"]} for ref in refs],
            "rendered": {"original_image_reference": str(image_path), "target_catalog_overlay": {"path": str(target_overlay.resolve()), "sha256": file_hash(target_overlay)}, "raw_generated_overlay": {"path": str(raw_overlay.resolve()), "sha256": file_hash(raw_overlay)}, "raw_rows": rendered_rows},
            "acceptance_boundary": {"no_first_fit_results": True, "no_owner_or_class_decisions": True, "raw_rows_all_preserved": True, "invalid_axes_drawn_without_sorting": True, "crop_frame_may_use_envelope_for_viewing_only": True, "matching_metrics_are_not_recomputed": True},
        }
        packet_path = packet_dir / "packet.json"
        packet_path.write_bytes(canonical(packet))
        packet_summaries.append({"image_id": image_id, "path": str(packet_path.resolve()), "target_reference_count": len(refs), "raw_row_count": len(raw_rows), "invalid_or_nonvalid_raw_rows": sum(item.get("status") != "parsed_valid" for item in raw_rows), "crop_count": len(raw_rows) * 2})
    receipt = {"schema": "training_set_completion.first_fit_visualization_preparation_receipt.v1", "status": "diagnostic_visualization_ready", "sources": {"scored_json": binding(scored_path), "acquisition_manifest": binding(acquisition_path), "target_catalog": binding(target_path)}, "expected_cohort_image_ids": list(COHORT), "rendered_image_ids": sorted(seen), "missing_image_ids": [image_id for image_id in COHORT if image_id not in seen], "packets": packet_summaries, "no_first_fit_results": True, "no_owner_or_class_decisions": True}
    receipt["receipt_sha256"] = hashlib.sha256(canonical(receipt)).hexdigest()
    (output / "render_receipt.json").write_bytes(canonical(receipt))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--acquisition-manifest", type=Path)
    parser.add_argument("--target-owners", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(scored_path=args.scored, output=args.output, acquisition_path=args.acquisition_manifest, target_path=args.target_owners), indent=2))


if __name__ == "__main__":
    main()
