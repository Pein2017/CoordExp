"""Render bounded saved-row overlays; row IDs are zero-based, geometry norm1000."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

from probes.training_set_completion.row_scoring import PAT


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def pixels(box, size):
    return [value * size[index % 2] / 1000 for index, value in enumerate(box)]


def render(selection_path, review_path, out):
    selection = json.loads(selection_path.read_text())
    reviews = json.loads(review_path.read_text())
    selected = {case["image_key"]: case for case in selection["selected"]}
    assert len(selected) <= 4 and set(reviews) == set(selected)
    out.mkdir(parents=True, exist_ok=True)
    result = []
    for key, review in reviews.items():
        case = selected[key]
        for name in ("raw", "image"):
            assert binding(case[name]["path"])["sha256"] == case[name]["sha256"], (key, name)
        raw = json.loads(Path(case["raw"]["path"]).read_text())["rows"][case["batch_index"]]
        assert int(raw["image_id"]) == case["image_id"]
        assert raw["row_id"] == f"coco2017_{key.split(':')[0]}_{case['image_id']:012d}"
        rows = [{"row_id": i, "description": m[1], "box_norm1000": list(map(int, m.groups()[1:]))}
                for i, m in enumerate(PAT.finditer(raw["text"]))]
        im = Image.open(case["image"]["path"]).convert("RGB")
        for row in rows:
            box = row["box_norm1000"]
            row["box_pixels"] = pixels(box, im.size)
            row["geometry_valid"] = 0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000
        stem = key.replace(":", "-")
        full = out / f"{stem}-full.png"
        im.save(full)
        figures = [str(full)]
        for group_index, group in enumerate(review["panels"]):
            crop = pixels(group["crop_norm1000"], im.size)
            crop = tuple(round(v) for v in crop)
            assert 0 <= crop[0] < crop[2] <= im.width and 0 <= crop[1] < crop[3] <= im.height
            base = im.crop(crop)
            scale = min(4, 650 / max(base.size))
            size = tuple(max(1, round(v * scale)) for v in base.size)
            base = base.resize(size, Image.Resampling.NEAREST)
            canvas = Image.new("RGB", (size[0] * (len(group["rows"]) + 1), size[1] + 50), "white")
            draw = ImageDraw.Draw(canvas)
            canvas.paste(base, (0, 50))
            draw.text((3, 3), f"{key} / original crop", fill="black")
            for j, row_id in enumerate(group["rows"], 1):
                row = rows[row_id]
                canvas.paste(base, (j * size[0], 50))
                box = pixels(row["box_norm1000"], im.size)
                box = [(v - crop[k % 2]) * scale + (j * size[0] if k % 2 == 0 else 50)
                       for k, v in enumerate(box)]
                valid = row["box_norm1000"][0] < row["box_norm1000"][2] and row["box_norm1000"][1] < row["box_norm1000"][3]
                if valid:
                    draw.rectangle(box, outline="red", width=2)
                else:
                    draw.line((box[0], box[1], box[2], box[3]), fill="red", width=3)
                    draw.ellipse((box[0]-3, box[1]-3, box[0]+3, box[1]+3), outline="red", width=2)
                draw.text((j * size[0]+3, 3), f"r{row_id} {row['description']}\n{row['box_norm1000']}" + (" INVALID" if not valid else ""), fill="black")
            figure = out / f"{stem}-panel-{group_index}.png"
            canvas.save(figure)
            figures.append(str(figure))
        inspected = sorted({i for group in review["panels"] for i in group["rows"]})
        result.append({"image_key": key, "raw": case["raw"], "image": case["image"],
                       "batch_index": case["batch_index"], "image_size": list(im.size),
                       "rows": [rows[i] for i in inspected], "judgments": review["judgments"],
                       "figures": [binding(p) for p in figures]})
    receipt = {"status": "candidate", "row_index_base": 0, "geometry_denominator": 1000,
               "selection": binding(selection_path), "review": binding(review_path),
               "source": binding(__file__), "cases": result}
    (out / "visual-evidence.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        assert pixels([0, 0, 1000, 1000], (1248, 832)) == [0, 0, 1248, 832]
        assert pixels([250, 250, 500, 500], (1248, 832)) == [312, 208, 624, 416]
        assert pixels([999, 0, 1000, 1000], (1000, 800))[0] == 999
        print("norm1000 geometry checks passed")
    else:
        if not all((args.selection, args.review, args.out)):
            parser.error("--selection, --review and --out are required")
        render(args.selection, args.review, args.out.resolve())
