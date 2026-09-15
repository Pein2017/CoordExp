"""Render review-only per-owner full-image and marked-context COCO22 views."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw

from probes.training_set_completion import coco22_acquisition as discovery
from probes.training_set_completion import review_packets as visuals
from src.data.geometry import coord_bins_to_pixel_xyxy


SCHEMA = "training_set_completion.coco22_bbox_context_review.v1"
require = discovery.require
binding = discovery.binding
publish = discovery.publish
read = discovery.read


def _tag(canvas: Image.Image, identifier: str) -> None:
    draw = ImageDraw.Draw(canvas)
    width = min(canvas.width, max(110, len(identifier) * 8 + 12))
    draw.rectangle((0, 0, width, min(canvas.height, 24)), fill=(0, 0, 0))
    draw.text((4, 3), identifier, fill=(255, 255, 255), font=visuals._font(13))


def annotated_bbox_views(image: Image.Image, box: Sequence[int], identifier: str,
                         full_path: Path | None, context_path: Path) -> dict[str, Any]:
    """Use one renderer for candidate proposals and confirmed source-GT cards."""
    require(len(box) == 4 and all(type(value) is int for value in box), "pixel bbox shape")
    x1, y1, x2, y2 = box
    require(all(-image.width <= x <= 2 * image.width for x in (x1, x2))
            and all(-image.height <= y <= 2 * image.height for y in (y1, y2)),
            "pixel bbox must be near the original image")
    if full_path is not None:
        canvas = image.copy()
        visuals._draw_box(ImageDraw.Draw(canvas), box, (0, 110, 230), identifier, width=4)
        _tag(canvas, identifier)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(full_path, format="PNG", optimize=False)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    extent_x = max(abs(x2 - x1) * 1.8, image.width * .18,
                   len(identifier) * 8 + 24, 240)
    extent_y = max(abs(y2 - y1) * 1.8, image.height * .18, 120)
    left = max(0, math.floor(cx - extent_x / 2))
    top = max(0, math.floor(cy - extent_y / 2))
    right = min(image.width, max(left + 1, math.ceil(cx + extent_x / 2)))
    bottom = min(image.height, max(top + 1, math.ceil(cy + extent_y / 2)))
    context = image.crop((left, top, right, bottom))
    local = [x1 - left, y1 - top, x2 - left, y2 - top]
    visuals._draw_box(ImageDraw.Draw(context), local, (255, 80, 0), identifier, width=4)
    _tag(context, identifier)
    context_path.parent.mkdir(parents=True, exist_ok=True)
    context.save(context_path, format="PNG", optimize=False)
    return {"original_bbox_pixel_xyxy": list(box), "context_bounds_pixel_xyxy": [left, top, right, bottom],
            "bbox_pixel_xyxy_within_context": local,
            "individual_full_image_bbox_overlay": binding(full_path) if full_path is not None else None,
            "context_crop_with_bbox_and_id": binding(context_path)}


def _index_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    require(len(rows) == 169 and len({row["owner_id"] for row in rows}) == 169,
            "169 unique GT source cards")
    return rows


def _write_index(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    require(not path.exists(), "review index collision")
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    require(sum(1 for _ in path.open()) == len(rows), "review index readback denominator")


def gt_contexts(*, gt_index_path: Path, output: Path) -> dict[str, Any]:
    require(not output.exists(), "GT context root occupied")
    source = _index_rows(gt_index_path)
    output.mkdir(parents=True)
    grouped: dict[int, list[dict[str, Any]]] = {}
    for card in source:
        grouped.setdefault(card["image_id"], []).append(card)
    require(len(grouped) == 11, "11 GT source images")
    index = []
    for image_id, cards in grouped.items():
        image_binding = cards[0]["visual"]["original"]
        require(binding(image_binding["path"]) == image_binding, "GT original visual source bytes")
        with Image.open(image_binding["path"]) as original:
            image = original.convert("RGB")
        for card in cards:
            require(card["visual"]["original"] == image_binding
                    and card["status"] == "pending_visual_review"
                    and card["decision"] is None, "GT source card/decision identity")
            bins = [int(token.removeprefix("<|coord_").removesuffix("|>")) for token in card["bbox_2d"]]
            require(list(coord_bins_to_pixel_xyxy(bins, image_width=image.width,
                                                  image_height=image.height, field="review_gt_bins"))
                    == card["bbox_pixel_xyxy"], "GT bbox production conversion")
            for name in ("bbox_overlay", "crop"):
                require(binding(card["visual"][name]["path"]) == card["visual"][name],
                        "GT predecessor visual bytes")
            filename = hashlib.sha256(card["owner_id"].encode()).hexdigest()[:16] + ".png"
            view = annotated_bbox_views(image, card["bbox_pixel_xyxy"], card["owner_id"],
                                        None, output / f"image-{image_id:012d}" / filename)
            index.append({"image_id": image_id, "owner_id": card["owner_id"],
                          "source_card": card, "original": image_binding,
                          "full_image_individual_bbox_overlay": card["visual"]["bbox_overlay"],
                          "marked_context": view["context_crop_with_bbox_and_id"],
                          "bbox_pixel_xyxy": view["original_bbox_pixel_xyxy"],
                          "plain_crop_auxiliary": card["visual"]["crop"],
                          "decision": None})
    _write_index(output / "index.jsonl", index)
    receipt = {"schema": f"{SCHEMA}.gt_context_receipt", "status": "GT_visual_views_pending_native_review",
               "source_gt_index": binding(gt_index_path), "producer": binding(Path(__file__)),
               "gt_card_count": len(index), "image_count": len(grouped),
               "view_index": binding(output / "index.jsonl"), "no_truth_decisions": True}
    publish(output / "receipt.json", receipt)
    return receipt


def candidate_contexts(*, manifest_path: Path, output: Path,
                       gt_index_path: Path) -> dict[str, Any]:
    manifest = discovery.validate_manifest(manifest_path)
    require(output.resolve() == manifest_path.resolve().parent, "candidate review root/manifest")
    # Frozen discovery producer supplies request/hash/row/parser and four-policy
    # invariants; it also retains raw unlocatable parser drops in each packet.
    if not (output / "review-packets-v1/receipt.json").exists():
        discovery.build_review_packets(manifest_path=manifest_path, output=output,
                                       gt_index_path=gt_index_path)
    previous = read(output / "review-packets-v1/receipt.json")
    require(previous["manifest"] == binding(manifest_path)
            and previous["cohort_gt_visual_index"] == binding(gt_index_path)
            and previous["image_count"] == 11 and previous["request_count"] == 44,
            "frozen base candidate review packet provenance")
    destination = output / "review-packets-v2"
    require(not destination.exists(), "candidate bbox-context root occupied")
    destination.mkdir()
    packets = []
    raw_count = drawable = unlocatable = 0
    for image in previous["packets"]:
        image_id = image["image_id"]
        base = read(image["packet"]["path"])
        source = base["original_image"]
        require(binding(source["path"]) == source, "candidate source original image")
        with Image.open(source["path"]) as original:
            canvas = original.convert("RGB")
        policies = copy.deepcopy(base["policies"])
        folder = destination / f"image-{image_id:012d}"
        folder.mkdir()
        for policy in policies:
            for ordinal, proposal in enumerate(policy["proposals"]):
                raw_count += 1
                box = proposal["bbox_pixel_xyxy"]
                proposal["plain_crop_auxiliary"] = proposal.pop("crop_path", None)
                if box is None:
                    unlocatable += 1
                    proposal["visual_status"] = "raw_candidate_without_drawable_bbox"
                    proposal["individual_full_image_bbox_overlay"] = None
                    proposal["context_crop_with_bbox_and_id"] = None
                    proposal["original_image_reference"] = source
                    continue
                label = proposal["proposal_id"]
                filename = hashlib.sha256(f"{label}:{ordinal}".encode()).hexdigest()[:16]
                paths = annotated_bbox_views(canvas, box, label,
                                             folder / "individual-overlays" / f"{filename}.png",
                                             folder / "marked-context" / f"{filename}.png")
                proposal.update(paths, visual_status="bbox_visible_pending_owner_review",
                                original_image_reference=source)
                drawable += 1
        packet = {"schema": f"{SCHEMA}.candidate_packet", "status": "candidate_pending_native_visual_review",
                  "image_id": image_id, "original_image": source,
                  "corrected_gt_visual_refs": base["gt_visual_references"],
                  "frozen_proposal_packet": binding(image["packet"]["path"]), "policies": policies,
                  "boundary": "Individual overlays and marked context are viewing aids."
                              " Every raw parser drop remains visible; no object, class, or teacher is admitted."}
        publish(folder / "packet.json", packet)
        packets.append({"image_id": image_id, "packet": binding(folder / "packet.json"),
                        "policy_count": len(policies),
                        "candidate_count": sum(len(policy["proposals"]) for policy in policies)})
    require(raw_count == previous["raw_candidate_count"] and len(packets) == 11,
            "candidate rendering denominator")
    receipt = {"schema": f"{SCHEMA}.candidate_receipt", "status": "candidate_views_pending_native_review",
               "manifest": binding(manifest_path), "previous_review": binding(output / "review-packets-v1/receipt.json"),
               "corrected_gt_visual_index": binding(gt_index_path), "producer": binding(Path(__file__)),
               "image_count": len(packets), "request_count": 44,
               "raw_candidate_count": raw_count, "individual_bbox_overlay_count": drawable,
               "marked_bbox_context_count": drawable, "unlocatable_raw_candidate_count": unlocatable,
               "packets": packets, "no_truth_or_teacher_admissions": True}
    publish(destination / "receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("gt-contexts", "candidate-contexts"))
    parser.add_argument("--gt-review-index", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "gt-contexts":
        value = gt_contexts(gt_index_path=args.gt_review_index, output=args.output)
    else:
        require(args.manifest is not None, "candidate manifest required")
        value = candidate_contexts(manifest_path=args.manifest, output=args.output,
                                   gt_index_path=args.gt_review_index)
    print(json.dumps({"schema": value["schema"], "status": value["status"],
                      "image_count": value["image_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
