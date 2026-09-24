"""Render proposed native arrivals and fixed candidates on source images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from probes.training_set_completion.artifacts import literal_binding
from probes.training_set_completion.recurrence_first_arrivals.prepare import CENSUS, MATURE, _require
from src.data.geometry import parse_source_bbox_tokens


def render(selection: Path) -> list[dict]:
    proposal = json.loads(selection.read_text())
    out = []
    for family in proposal["families"]:
        source = family["source"]
        root = MATURE if source == "mature" else CENSUS
        source_row = family["source_bindings"]["image"]
        path = Path(source_row["path"])
        _require(literal_binding(path) == source_row, "image changed after selection")
        image = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(image)

        def box(values, color: str, label: str, width: int) -> None:
            xy = [round(value * (image.width if index % 2 == 0 else image.height) / 1000)
                  for index, value in enumerate(values)]
            draw.rectangle(xy, outline=color, width=width)
            draw.text((xy[0], max(0, xy[1] - 14)), label, fill=color)

        record_path = root / ("refined18.runtime.jsonl" if source == "mature" else "new128.runtime.jsonl")
        if source == "mature":
            panel = json.loads((MATURE / "panel.json").read_text())
            record = next(case["input_record"] for group in panel["groups"] for case in group["cases"]
                          if case["input_record"]["image_id"] == family["image_id"])
        else:
            record = next(json.loads(line) for line in record_path.read_text().splitlines()
                          if json.loads(line)["image_id"] == family["image_id"])
        for obj in record["objects"]:
            if obj["desc"] == family["description"]:
                values = parse_source_bbox_tokens(obj["bbox_2d"], field="bbox_2d")
                box(values, "#12ce57", str(obj["coco_ann_id"]), 2)
        for candidate in family["candidates"]:
            box(candidate["bbox"], "#00e8ff", f"N {candidate['owner_id']}", 3)
        by_index = {row["index"]: row for row in family["row_evidence"]}
        for ordinal, index in enumerate(family["arrival_row_indices"], 1):
            row = by_index[index]
            box(row["bbox"], ("#ff2222", "#ffae00", "#ec3aff")[ordinal - 1],
                f"arrival {ordinal} row {index}", 5)
        image.thumbnail((1200, 1000))
        destination = Path(family["visual_review_handles"][0])
        destination.parent.mkdir(parents=True, exist_ok=True)
        _require(not destination.exists(), f"visual output already exists: {destination}")
        image.save(destination)
        out.append(literal_binding(destination))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args()
    outputs = render(args.selection)
    print(json.dumps({"rendered": len(outputs), "outputs": outputs}, sort_keys=True))


if __name__ == "__main__":
    main()
