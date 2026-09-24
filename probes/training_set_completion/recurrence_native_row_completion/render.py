"""Render completed first-case boxes with annotation overlaps as proxies."""

import json
from pathlib import Path

from PIL import Image, ImageDraw

from probes.training_set_completion.recurrence_native_row_completion.run import (
    ADMISSION, OUT, UNIT, bind, require, write_new,
)
from probes.training_set_completion.recurrence_first_arrivals.prepare import MATURE
from src.data.geometry import iou_xyxy, parse_source_bbox_tokens


def main():
    dest = UNIT / "supporting/first-case-visual-proxies-v1.json"
    require(not dest.exists(), "visual proxy record exists")
    summary = json.loads((OUT / "readback.json").read_text())
    require(summary["status"] == "candidate_cold_readback_passed", "cold readback missing")
    admission = json.loads(ADMISSION.read_text())
    image_path = Path(admission["source_bindings"]["image"]["path"])
    require(bind(image_path)["sha256"] == admission["source_bindings"]["image"]["sha256"],
            "original image changed")
    panel_path = MATURE / "panel.json"
    panel = json.loads(panel_path.read_text())
    records = [c["input_record"] for g in panel["groups"] for c in g["cases"]
               if c["input_record"]["image_id"] == 313465]
    require(len(records) == 1, "annotation source image not unique")
    objects = sorted(
        ({"owner_id": int(obj["coco_ann_id"]),
          "bbox": list(parse_source_bbox_tokens(obj["bbox_2d"], field="bbox_2d"))}
         for obj in records[0]["objects"] if obj["desc"] == "bowl"),
        key=lambda x: x["owner_id"])
    require({x["owner_id"] for x in objects} == {716308, 715278},
            "reviewed bowl annotations changed")
    rendered = []
    for arm in summary["arms"]:
        ids = arm["emitted"]
        if arm["stop"] != "complete":
            rendered.append({"arm": arm["arm"], "category": arm["stop"], "box": None})
            continue
        require(len(ids) == 5 and ids[-1] == 151649 and
                all(151670 <= x <= 152670 for x in ids[:4]), "complete row syntax changed")
        box = [x - 151670 for x in ids[:4]]
        category = "complete_valid_geometry" if box[0] < box[2] and box[1] < box[3] else "invalid_geometry"
        overlaps = sorted(({"owner_id": x["owner_id"], "bbox": x["bbox"],
                            "iou": iou_xyxy(box, x["bbox"])} for x in objects),
                          key=lambda x: (-x["iou"], x["owner_id"]))
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        def pixels(b):
            return [round(v * (image.width if i % 2 == 0 else image.height) / 1000)
                    for i,v in enumerate(b)]
        for obj in objects:
            xy = pixels(obj["bbox"])
            draw.rectangle(xy, outline="#00e070", width=3)
            draw.text((xy[0], max(0,xy[1]-14)), str(obj["owner_id"]), fill="#00e070")
        xy = pixels(box)
        draw.rectangle(xy, outline="#ff3030", width=5)
        draw.text((xy[0], max(0,xy[1]-25)), arm["arm"], fill="#ff3030")
        path = OUT / f"overlay-{arm['arm']}.png"
        require(not path.exists(), "overlay exists")
        image.save(path)
        rendered.append({"arm": arm["arm"], "category": category, "box": box,
                         "same_class_annotation_overlaps": overlaps, "overlay": bind(path)})
    result = {"schema": "recurrence_native_row_completion.visual_proxies.v1",
              "status": "candidate_annotation_proxy_only",
              "admission": bind(ADMISSION), "readback": bind(OUT/"readback.json"),
              "annotation_panel": bind(panel_path), "original_image": bind(image_path),
              "reviewed_bowls": objects, "arms": rendered,
              "physical_owner_verdict": "UNKNOWN pending lead visual review"}
    write_new(dest, result)
    print(json.dumps({"status":result["status"],
                      "boxes":{x["arm"]:x["box"] for x in rendered},
                      "overlaps":{x["arm"]:[(y["owner_id"],round(y["iou"],6))
                                                for y in x.get("same_class_annotation_overlaps",[])]
                                  for x in rendered}}))


if __name__ == "__main__":
    main()
