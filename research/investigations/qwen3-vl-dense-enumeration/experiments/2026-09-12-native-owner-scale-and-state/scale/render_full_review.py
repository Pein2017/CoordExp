"""Render bound full-canvas and zoom cards for the frozen scale acquisition.

Blue boxes are literal native-history owners, red is the nominated c, and
green is the immediate complete free-continuation w.  The cards are evidence
for human review, not automatic owner admission.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps

from probes.dora_owner_learning.candidate_opportunity import file_hash, require
from probes.native_owner_scale.scale import binding, publish, read, validate_selection
from probes.source_rweak_row_cross.run import native_record


ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-12-native-owner-scale-and-state/scale"
)
RESULT = ROOT / "acquisition-full-v2.json"
OUTPUT = ROOT / "visual-review-full-v2"
CARD_SIZE = (1600, 1680)
FULL_PANEL = (1500, 760)
ZOOM_PANEL = (735, 500)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _fit(image: Image.Image, size: tuple[int, int]) -> tuple[Image.Image, float, int, int]:
    scale = min(size[0] / image.width, size[1] / image.height)
    resized = image.resize((round(image.width * scale), round(image.height * scale)))
    canvas = Image.new("RGB", size, "white")
    left = (size[0] - resized.width) // 2
    top = (size[1] - resized.height) // 2
    canvas.paste(resized, (left, top))
    return canvas, scale, left, top


def _box_after_transform(
    box: Sequence[int], crop: Sequence[int], scale: float, left: int, top: int
) -> tuple[int, int, int, int]:
    return tuple(
        round((box[i] - crop[i % 2]) * scale + (left if i % 2 == 0 else top))
        for i in range(4)
    )


def _intersects(a: Sequence[int], b: Sequence[int]) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _draw_boxes(
    panel: Image.Image,
    crop: Sequence[int],
    scale: float,
    left: int,
    top: int,
    history_boxes: Iterable[Sequence[int]],
    c_box: Sequence[int],
    w_box: Sequence[int],
) -> None:
    draw = ImageDraw.Draw(panel)
    for box in history_boxes:
        if _intersects(box, crop):
            draw.rectangle(_box_after_transform(box, crop, scale, left, top), outline=(30, 100, 230), width=2)
    for box, colour, label in (
        (c_box, (230, 20, 20), "c"),
        (w_box, (20, 170, 40), "w"),
    ):
        if not _intersects(box, crop):
            continue
        xyxy = _box_after_transform(box, crop, scale, left, top)
        width = 7
        draw.rectangle(xyxy, outline=colour, width=width)
        label_xy = (max(0, xyxy[0]), max(0, xyxy[1] - 30))
        draw.rectangle((label_xy[0], label_xy[1], label_xy[0] + 32, label_xy[1] + 28), fill="black")
        draw.text((label_xy[0] + 7, label_xy[1]), label, fill=colour, font=_font(24))


def _context_crop(image: Image.Image, box: Sequence[int]) -> tuple[int, int, int, int]:
    width, height = box[2] - box[0], box[3] - box[1]
    side = min(max(max(width, height) * 4, 128), max(image.width, image.height))
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    left = max(0, min(round(cx - side / 2), image.width - side))
    top = max(0, min(round(cy - side / 2), image.height - side))
    return left, top, min(image.width, round(left + side)), min(image.height, round(top + side))


def _panel(
    image: Image.Image,
    crop: Sequence[int],
    size: tuple[int, int],
    history_boxes: Sequence[Sequence[int]],
    c_box: Sequence[int],
    w_box: Sequence[int],
) -> Image.Image:
    source = image.crop(tuple(crop))
    panel, scale, left, top = _fit(source, size)
    _draw_boxes(panel, crop, scale, left, top, history_boxes, c_box, w_box)
    return panel


def _card(
    image: Image.Image,
    job: Mapping[str, Any],
    row: Mapping[str, Any],
    history_boxes: Sequence[Sequence[int]],
) -> Image.Image:
    c_box = job["c_bbox_xyxy_pixels"]
    w_box = row["local_w"]["w_bbox_xyxy_pixels"]
    card = Image.new("RGB", CARD_SIZE, "white")
    draw = ImageDraw.Draw(card)
    title = (
        f"priority {job['admission_priority']:03d} | {job['job_id']} | {job['stratum']} | "
        f"c provenance: {job['source']}"
    )
    detail = (
        f"c: {job['category']} owner={job['owner_id']} max-IoU(h)={job['max_any_class_history_iou']:.4f} | "
        f"w: {row['local_w']['w_description']} max-IoU(h+c)="
        f"{row['local_w']['max_any_class_prefix_iou']:.4f} | h rows={job['h_row_count']}"
    )
    question = (
        "Review: red c is one physical owner and absent from blue native h; "
        "green w is one physical owner and nonduplicate of h+c. Unknown stays neutral."
    )
    draw.text((28, 24), title, fill="black", font=_font(28))
    draw.text((28, 70), detail, fill="black", font=_font(25))
    draw.text((28, 112), question, fill=(80, 80, 80), font=_font(22))
    draw.text((28, 150), "blue=h   red=c   green=w", fill=(20, 20, 20), font=_font(24))

    full_crop = (0, 0, image.width, image.height)
    card.paste(_panel(image, full_crop, FULL_PANEL, history_boxes, c_box, w_box), (50, 205))
    draw.text((50, 975), "c context", fill=(230, 20, 20), font=_font(28))
    draw.text((815, 975), "w context", fill=(20, 150, 40), font=_font(28))
    card.paste(_panel(image, _context_crop(image, c_box), ZOOM_PANEL,
                      history_boxes, c_box, w_box), (50, 1020))
    card.paste(_panel(image, _context_crop(image, w_box), ZOOM_PANEL,
                      history_boxes, c_box, w_box), (815, 1020))
    draw.text((50, 1540), f"c bbox={c_box}", fill=(80, 80, 80), font=_font(22))
    draw.text((815, 1540), f"w bbox={w_box}", fill=(80, 80, 80), font=_font(22))
    return card


def main() -> None:
    require(not OUTPUT.exists(), f"review output collision: {OUTPUT}")
    result = read(RESULT)
    require(result["schema"] == "native_owner_scale.acquisition_result.v1"
            and result["mode"] == "full" and len(result["rows"]) == 156,
            "complete merged acquisition required")
    selection = validate_selection(result["selection"]["path"])
    require(result["selection"] == binding(result["selection"]["path"]), "selection binding")
    source = read(selection["sources"]["stable50_universe"]["path"])
    by_example = {row["example_id"]: row for row in source["eval_records"]}
    by_job = {row["job_id"]: row for row in selection["nominations"]}
    candidates = [row for row in result["rows"]
                  if row["local_w"]["status"] == "candidate_local_w"]
    require(len(candidates) == result["denominators"]["candidate_local_w"] == 50,
            "candidate card denominator")

    cards_dir = OUTPUT / "cards"
    cards_dir.mkdir(parents=True)
    cards = []
    for row in candidates:
        job = by_job[row["job_id"]]
        frozen = by_example[row["example_id"]]
        image_path = Path(job["image"]["image_path"])
        require(file_hash(image_path) == job["image"]["image_sha256"], "review image identity")
        parsed_h = native_record(job["h_text"], frozen["case"], frozen["golden"], "supplied_prefix")
        require(not parsed_h["dropped_predictions"]
                and len(parsed_h["pred"]) == job["h_row_count"], "literal h boxes")
        history_boxes = [item["bbox"] for item in parsed_h["pred"]]
        with Image.open(image_path) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
            require(image.size == (job["image"]["image_width"], job["image"]["image_height"]),
                    "review image dimensions")
            card = _card(image, job, row, history_boxes)
        path = cards_dir / f"{job['admission_priority']:03d}-{job['image_id']}-{job['owner_id']}.png"
        card.save(path, format="PNG", optimize=False)
        cards.append({
            "job_id": row["job_id"],
            "admission_priority": job["admission_priority"],
            "stratum": job["stratum"],
            "c_provenance": job["source"],
            "c_category": job["category"],
            "c_bbox_xyxy_pixels": job["c_bbox_xyxy_pixels"],
            "max_any_class_history_iou": job["max_any_class_history_iou"],
            "w_description": row["local_w"]["w_description"],
            "w_bbox_xyxy_pixels": row["local_w"]["w_bbox_xyxy_pixels"],
            "max_any_class_prefix_iou": row["local_w"]["max_any_class_prefix_iou"],
            "image": binding(image_path),
            "card": binding(path),
        })

    draft_decisions = {}
    card_by_job = {card["job_id"]: card for card in cards}
    for row in result["rows"]:
        if row["job_id"] in card_by_job:
            draft_decisions[row["job_id"]] = {
                "status": "pending_root_visual_review",
                "c_single_owner_absent_from_h": None,
                "w_single_owner_nonduplicate": None,
                "evidence_paths": [card_by_job[row["job_id"]]["card"]["path"]],
            }
        else:
            draft_decisions[row["job_id"]] = {
                "status": "mechanical_neutral_pending_root_ack",
                "reason": row["local_w"]["status"],
                "evidence_paths": [str(RESULT)],
            }
    publish(OUTPUT / "review-draft.json", {
        "schema": "native_owner_scale.visual_reviews.draft.v1",
        "status": "root_decisions_required_not_training_admissible",
        "decisions": draft_decisions,
    })
    publish(OUTPUT / "manifest.json", {
        "schema": "native_owner_scale.full_visual_review.v1",
        "status": "candidate_cards_root_review_pending",
        "result": binding(RESULT),
        "selection": result["selection"],
        "review_scope": (
            "blue literal h, red nominated c, green immediate local w; visual owner identity "
            "is required and machine IoU remains only a conservative review aid"
        ),
        "counts": {"executed": 156, "candidate_cards": len(cards),
                   "mechanical_noncandidate": 156 - len(cards)},
        "cards": cards,
        "review_draft": binding(OUTPUT / "review-draft.json"),
        "no_training_admission": True,
    })
    print(json.dumps({"manifest": binding(OUTPUT / "manifest.json"),
                      "cards": len(cards)}, indent=2))


if __name__ == "__main__":
    main()
