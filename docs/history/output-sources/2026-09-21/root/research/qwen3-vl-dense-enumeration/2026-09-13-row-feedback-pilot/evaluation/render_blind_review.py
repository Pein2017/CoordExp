"""Render the source-blind dense review queue without reading its source map."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageDraw, ImageFont


QUEUE_SCHEMA = "row_feedback.dense8_blind_review_queue.v1"
RECORD_SCHEMA = "row_feedback.dense8_physical_review_record.v1"
DEMO_STATUS = "DEMO_SYNTHETIC_NO_SCIENTIFIC_OUTPUTS"
DECISIONS = ("clearly_visible_plausible", "uncertain", "invalid")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binding(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_sha256(path), "size_bytes": path.stat().st_size}


def publish_json(path: Path, value: Any) -> None:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _font(*, bold: bool = False, size: int = 18) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(str(Path("/usr/share/fonts/truetype/dejavu") / name), size)
    except OSError:
        return ImageFont.load_default()


def _box(value: Mapping[str, Any], *, width: int, height: int) -> tuple[float, float, float, float]:
    require(value.get("bbox_format") == "xyxy_native_pixels", "proposal bbox format")
    raw = value.get("bbox")
    require(isinstance(raw, list) and len(raw) == 4, "proposal bbox")
    require(
        all(type(item) in (int, float) and not isinstance(item, bool) and math.isfinite(item) for item in raw),
        "proposal bbox values",
    )
    box = tuple(float(item) for item in raw)
    x1, y1, x2, y2 = box
    require(0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height, "proposal bbox outside native image")
    return box


def _color(public_id: str) -> tuple[int, int, int]:
    raw = bytes.fromhex(hashlib.sha256(public_id.encode()).hexdigest()[:6])
    return tuple(50 + value % 156 for value in raw)


def _label(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, color: tuple[int, int, int]) -> None:
    font = _font(bold=True, size=18)
    x, y = xy
    bounds = draw.textbbox((x, y), text, font=font)
    draw.rectangle((bounds[0] - 3, bounds[1] - 3, bounds[2] + 3, bounds[3] + 3), fill="white", outline=color, width=2)
    draw.text((x, y), text, fill="black", font=font)


def _groups(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    width, height = int(row["width"]), int(row["height"])
    grouped: dict[tuple[str, tuple[float, float, float, float]], list[str]] = {}
    for proposal in row["proposals"]:
        proposal_id = proposal.get("proposal_id")
        description = proposal.get("description")
        require(isinstance(proposal_id, str) and len(proposal_id) == 64, "proposal identity")
        require(isinstance(description, str) and description, "proposal class")
        key = (description, _box(proposal, width=width, height=height))
        grouped.setdefault(key, []).append(proposal_id)
    result = []
    for index, ((description, box), proposal_ids) in enumerate(grouped.items(), start=1):
        result.append(
            {
                "public_id": f"P{index:03d}",
                "proposal_ids": proposal_ids,
                "description": description,
                "bbox_xyxy_native_pixels": list(box),
            }
        )
    return result


def _render_context(image: Image.Image, groups: list[dict[str, Any]], output: Path) -> None:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    line_width = max(3, min(image.size) // 250)
    for group in groups:
        box = tuple(group["bbox_xyxy_native_pixels"])
        color = _color(group["public_id"])
        draw.rectangle(box, outline=color, width=line_width)
        _label(draw, (box[0] + 3, max(3, box[1] + 3)), group["public_id"], color)
    canvas.save(output)


def _render_crop(image: Image.Image, group: Mapping[str, Any], output: Path) -> None:
    x1, y1, x2, y2 = group["bbox_xyxy_native_pixels"]
    padding = max(12, int(round(max(x2 - x1, y2 - y1) * 0.18)))
    region = (
        max(0, math.floor(x1 - padding)),
        max(0, math.floor(y1 - padding)),
        min(image.width, math.ceil(x2 + padding)),
        min(image.height, math.ceil(y2 + padding)),
    )
    crop = image.crop(region)
    header = 42
    coords = ",".join(f"{value:g}" for value in group["bbox_xyxy_native_pixels"])
    header_text = f"{group['public_id']}  {group['description']}  [{coords}]"
    font = _font(bold=True, size=16)
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    text_bounds = measure.textbbox((0, 0), header_text, font=font)
    canvas_width = max(crop.width, text_bounds[2] - text_bounds[0] + 12)
    crop_x = (canvas_width - crop.width) // 2
    canvas = Image.new("RGB", (canvas_width, crop.height + header), "white")
    canvas.paste(crop, (crop_x, header))
    draw = ImageDraw.Draw(canvas)
    local_box = (
        x1 - region[0] + crop_x,
        y1 - region[1] + header,
        x2 - region[0] + crop_x,
        y2 - region[1] + header,
    )
    color = _color(str(group["public_id"]))
    draw.rectangle(local_box, outline=color, width=max(3, min(image.size) // 250))
    draw.text((6, 8), header_text, fill="black", font=font)
    canvas.save(output)


def render(queue_path: str | Path, output: str | Path, *, demo: bool = False) -> dict[str, Any]:
    queue_path, output = Path(queue_path), Path(output)
    require(not output.exists(), "render output already exists")
    queue = json.loads(queue_path.read_text())
    require(queue.get("schema") == QUEUE_SCHEMA, "queue schema")
    expected_status = DEMO_STATUS if demo else "ready_for_physical_review"
    require(queue.get("status") == expected_status, "queue/demo status")
    rows = queue.get("rows")
    require(isinstance(rows, list) and rows, "queue rows")
    require(demo or len(rows) == queue.get("images") == 8, "scientific dense8 denominator")
    require(demo or all(row.get("source_blind") is True for row in rows), "queue is not source blind")
    review_ids = [row.get("review_id") for row in rows]
    require(len(review_ids) == len(set(review_ids)), "duplicate review identity")
    proposal_ids = [proposal.get("proposal_id") for row in rows for proposal in row.get("proposals", [])]
    require(len(proposal_ids) == len(set(proposal_ids)), "duplicate proposal identity")

    output.mkdir(parents=True)
    record_rows, public_map, rendered_files, markdown = [], [], [], [
        "# DEMO physical-review rendering" if demo else "# Dense-8 source-blind physical review",
        "",
        "Review only the listed proposals. This is not an exhaustive census; unlisted or unmatched objects are not negatives.",
        "",
    ]
    for image_index, row in enumerate(rows, start=1):
        require(row.get("source_blind") is True, "review row is not source blind")
        image_path = Path(str(row["image_path"])).resolve()
        require(image_path.is_file(), "review image missing")
        image = Image.open(image_path).convert("RGB")
        require(image.size == (int(row["width"]), int(row["height"])), "review image geometry")
        groups = _groups(row)
        require(sum(len(group["proposal_ids"]) for group in groups) == len(row["proposals"]), "proposal accounting")
        folder = output / f"{image_index:02d}-{int(row['image_id'])}"
        folder.mkdir()
        raw_path = folder / "raw.png"
        context_path = folder / "context-boxes.png"
        image.save(raw_path)
        _render_context(image, groups, context_path)
        rendered_files.extend((binding(raw_path), binding(context_path)))
        markdown += [f"## Image {image_index}: {int(row['image_id'])}", "", f"![raw]({raw_path.relative_to(output)})", "", f"![context]({context_path.relative_to(output)})", ""]
        review_groups = []
        for group in groups:
            crop_path = folder / f"{group['public_id']}.png"
            _render_crop(image, group, crop_path)
            rendered_files.append(binding(crop_path))
            public_map.append(
                {
                    "review_id": row["review_id"],
                    "image_id": int(row["image_id"]),
                    **group,
                }
            )
            coords = ", ".join(f"{value:g}" for value in group["bbox_xyxy_native_pixels"])
            markdown += [
                f"### {group['public_id']} - {group['description']} - [{coords}]",
                "",
                f"Proposal IDs: {', '.join(group['proposal_ids'])}",
                "",
                f"![{group['public_id']}]({crop_path.relative_to(output)})",
                "",
            ]
            review_groups.append(
                {
                    **group,
                    "visibility_decision": None,
                    "physical_owner_id": None,
                    "notes": "",
                }
            )
        record_rows.append(
            {
                "review_id": row["review_id"],
                "image_id": int(row["image_id"]),
                "groups": review_groups,
                "image_notes": "",
                "exhaustive_visible_census_performed": False,
            }
        )
    sheet_path = output / "review-sheet.md"
    _write_text_exclusive(sheet_path, "\n".join(markdown))
    record = {
        "schema": RECORD_SCHEMA,
        "status": "DEMO_TEMPLATE_ONLY" if demo else "pending_source_blind_physical_review",
        "demo": demo,
        "queue": binding(queue_path),
        "decision_values": list(DECISIONS),
        "instructions": {
            "visibility_decision": "Assign one declared decision to every public group.",
            "physical_owner_id": "Assign consistent image-local owner IDs across groups that depict the same physical instance; exact duplicate proposals are already grouped for display only.",
            "scope": "Judge proposal visibility/class/geometry and alias identity only. Do not perform an exhaustive scene census and do not infer unmatched negatives.",
        },
        "rows": record_rows,
    }
    record_path = output / "review-record-template.json"
    map_path = output / "public-proposal-map.json"
    publish_json(record_path, record)
    publish_json(
        map_path,
        {
            "schema": "row_feedback.dense8_public_proposal_map.v1",
            "demo": demo,
            "queue": binding(queue_path),
            "rows": public_map,
        },
    )
    manifest = {
        "schema": "row_feedback.dense8_review_render_manifest.v1",
        "status": "DEMO_RENDERED_NO_SCIENTIFIC_OUTPUTS" if demo else "rendered_for_source_blind_review",
        "demo": demo,
        "queue": binding(queue_path),
        "review_sheet": binding(sheet_path),
        "review_record_template": binding(record_path),
        "public_proposal_map": binding(map_path),
        "rendered_files": rendered_files,
        "images": len(rows),
        "input_proposals": len(proposal_ids),
        "display_groups": len(public_map),
        "exact_duplicate_grouping_only": True,
        "source_map_read": False,
    }
    publish_json(output / "manifest.json", manifest)
    return manifest


def _write_text_exclusive(path: Path, text: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    result = render(args.queue, args.output, demo=args.demo)
    print(json.dumps({key: result[key] for key in ("status", "images", "input_proposals", "display_groups")}, sort_keys=True))


if __name__ == "__main__":
    main()
