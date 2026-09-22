"""Render original-image successor overlays from existing native records only."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_reducer(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("native_escape_reducer", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def first_complete_free(record: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    return [
        row for row in record["parser_partition"]["free_rows"]
        if str(row.get("raw_span_text", "")).endswith("<|box_end|>")
    ][:limit]


def annotated_free(record: dict[str, Any], native: dict[str, Any], reducer: Any) -> list[dict[str, Any]]:
    repeated = set(reducer.strict_repeat_orders(record))
    native_valid = native["parser_partition"]["valid_free_rows"]
    result = []
    for index, row in enumerate(first_complete_free(record), 1):
        same_native = []
        if row.get("bbox") is not None:
            same_native = [
                other for other in native_valid
                if other.get("description") == row.get("description")
                and reducer.iou(row["bbox"], other["bbox"]) > 0.5
            ]
        result.append({
            "free_ordinal": index,
            "generated_order": row.get("generated_order"),
            "description": row.get("description"),
            "bbox": row.get("bbox"),
            "parser_disposition": row["parser_disposition"],
            "reason": row.get("reason"),
            "strict_repeat": row.get("generated_order") in repeated,
            "same_description_native_suffix_iou_gt_0_5": bool(same_native),
        })
    return result


def draw_rows(draw: ImageDraw.ImageDraw, items: list[dict[str, Any]],
              color: tuple[int, int, int], prefix: str) -> None:
    font = ImageFont.load_default(size=18)
    for item in items:
        bbox = item.get("bbox")
        if bbox is None:
            continue
        flags = (" R" if item.get("strict_repeat") else "") + (
            " N" if item.get("same_description_native_suffix_iou_gt_0_5") else "")
        label = f"{prefix}{item['free_ordinal']} {item.get('description')}{flags}"
        draw.rectangle(tuple(bbox), outline=color, width=4)
        x, y = bbox[0], max(0, bbox[1] - 22)
        draw.rectangle((x, y, x + max(90, 9 * len(label)), y + 22), fill=(0, 0, 0))
        draw.text((x + 2, y + 2), label, fill=color, font=font)


def panel(image: Image.Image, native_items: list[dict[str, Any]],
          candidate_items: list[dict[str, Any]], candidate: dict[str, Any],
          summary: dict[str, Any]) -> Image.Image:
    width, height = image.size
    header = 90
    canvas = Image.new("RGB", (2 * width, height + header), "white")
    canvas.paste(image, (0, header))
    canvas.paste(image, (width, header))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=20)
    draw.text((12, 8), "Native h-only: first 5 complete free rows; h omitted",
              fill=(0, 90, 220), font=font)
    draw.text((width + 12, 8),
              "Forced c dashed magenta (not free/credited); green F1-F5 are fresh",
              fill=(0, 120, 40), font=font)
    draw.text((width + 12, 36),
              f"stop={summary['stop_reason']} tokens={summary['free_token_count']} "
              f"valid={summary['valid_complete_free_row_count']} "
              f"repeat={summary['strict_class_blind_repeat_later_row_count']} "
              f"geom-invalid={summary['burden']['geometry_invalid_complete_count']} "
              f"other-malformed={summary['burden']['other_complete_malformed_count']}",
              fill=(20, 20, 20), font=font)
    left = ImageDraw.Draw(canvas)
    shifted_native = [dict(item, bbox=[
        item["bbox"][0], item["bbox"][1] + header,
        item["bbox"][2], item["bbox"][3] + header])
        for item in native_items if item.get("bbox") is not None]
    draw_rows(left, shifted_native, (0, 120, 255), "N")
    cbox = candidate["bbox_pixel_xyxy"]
    shifted_c = (cbox[0] + width, cbox[1] + header,
                 cbox[2] + width, cbox[3] + header)
    for offset in range(0, 12, 8):
        left.rectangle((shifted_c[0] + offset, shifted_c[1],
                        shifted_c[2], shifted_c[3]), outline=(255, 0, 180), width=2)
    left.text((shifted_c[0] + 3, max(header, shifted_c[1] - 22)),
              f"C forced {candidate['description']}", fill=(255, 0, 180), font=font)
    shifted_candidate = [dict(item, bbox=[
        item["bbox"][0] + width, item["bbox"][1] + header,
        item["bbox"][2] + width, item["bbox"][3] + header])
        for item in candidate_items if item.get("bbox") is not None]
    draw_rows(left, shifted_candidate, (0, 220, 70), "F")
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--reduction", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    packet = json.loads(args.packet.read_text())
    case = {case["case_id"]: case for case in packet["cases"]}
    records = rows(args.records)
    by_job = {row["job_id"]: row for row in records}
    reductions = {
        item["job_id"]: item
        for item in json.loads(args.reduction.read_text())["cells"]
    }
    reducer = load_reducer(Path(__file__).with_name("reduce.py"))
    args.out_dir.mkdir(parents=True, exist_ok=False)
    figures = []
    for record in records:
        if record["kind"] != "h_plus_c":
            continue
        case_id = record["case_id"]
        native = by_job["h_only"]
        native_items = annotated_free(native, native, reducer)
        candidate_items = annotated_free(record, native, reducer)
        image = Image.open(case[case_id]["source_case"]["image_path"]).convert("RGB")
        rendered = panel(image, native_items, candidate_items,
                         record["forced_candidate"], reductions[record["job_id"]])
        path = args.out_dir / f"{record['job_id']}-successors.png"
        rendered.save(path)
        figures.append({
            "case_id": case_id,
            "job_id": record["job_id"],
            "candidate_id": record["forced_candidate"]["candidate_id"],
            "path": str(path.resolve()),
            "sha256": sha(path),
            "forced_candidate": {
                "description": record["forced_candidate"]["description"],
                "bbox": record["forced_candidate"]["bbox_pixel_xyxy"],
                "credited_as_free": False,
            },
            "native_h_only_first_complete_free": native_items,
            "candidate_first_complete_free": candidate_items,
            "legend": {
                "N": "same-description geometry overlaps some native h-only free row at IoU > 0.5; not certified physical owner",
                "R": "strict class-blind IoU > 0.95 recurrence to any earlier valid row, counted once",
                "h": "not drawn",
                "forced_c": "magenta, not included in free ordinals or credit",
            },
        })
    manifest = {
        "schema": "native_escape_witness.successor_visualizations.v1",
        "records": str(args.records.resolve()),
        "records_sha256": sha(args.records),
        "reduction": str(args.reduction.resolve()),
        "reduction_sha256": sha(args.reduction),
        "producer": str(Path(__file__).resolve()),
        "producer_sha256": sha(Path(__file__)),
        "figures": figures,
        "claim_boundary": (
            "Original-image display for root adjudication. Geometry overlap is not "
            "GT truth or physical-owner certification; h and forced c are excluded "
            "from fresh free-row ordinals and forced c is never credited."
        ),
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
