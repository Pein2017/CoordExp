"""CPU projection of the frozen earliest numerical-return basket."""

import json
from pathlib import Path

from PIL import Image, ImageDraw
from transformers import AutoTokenizer

from probes.training_set_completion.recurrence_first_arrivals.prepare import BASE, _load, _source_bindings


ROOT = Path(__file__).resolve().parents[3]
UNIT = ROOT / "research/experiments/2026-09-25-recurrence-report-quality-feedback"
ATTRITION = ROOT / "research/experiments/2026-09-24-recurrence-grounded-window-transfer/supporting/numerical-return-attrition-v1.json"
VIEWS = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-25-recurrence-report-quality-feedback/cpu-review")


def main():
    frozen = json.loads(ATTRITION.read_text())
    assert len(frozen["images"]) == 273
    basket = [image for image in frozen["images"] if image["earliest_return"]]
    assert len(basket) == 29 and [x["global_source_rank"] for x in basket] == sorted(x["global_source_rank"] for x in basket)
    sources = _load(AutoTokenizer.from_pretrained(BASE, local_files_only=True))
    assert len(sources) == 273
    VIEWS.mkdir(parents=True, exist_ok=True)
    projection = []
    for entry in basket:
        item = sources[(entry["source"], entry["image_id"])]
        bound = _source_bindings(item)
        cell = item["cell"]
        j = entry["earliest_return"]["j"]
        i = entry["earliest_return"]["earliest_predecessor"]
        assert cell["group"] == entry["group"] and cell["batch_index"] == entry["batch_index"]
        assert item["source_rank"] == entry["source_rank"] and item["split"] == entry["split"]
        assert all(bound[k]["sha256"] == entry["source_bindings"][k]["sha256"] for k in ("raw", "trace", "runtime_receipt", "image"))
        assert item["rows"][j]["start"] == entry["returns"][0]["raw_span"][0]
        assert item["rows"][j]["end"] == entry["returns"][0]["raw_span"][1]
        rows = [{k: row[k] for k in ("index", "description", "values", "valid", "start", "end", "proxy_owner", "top_iou", "second_iou")}
                for row in item["rows"][:j + 1]]
        assert [row["index"] for row in rows] == list(range(j + 1))
        with Image.open(item["image_path"]) as original:
            image = original.convert("RGB")
        w, h = image.size
        draw = ImageDraw.Draw(image)
        for row in rows:
            x1, y1, x2, y2 = row["values"]
            if not row["valid"]:
                continue
            box = tuple(round(v * axis / 999) for v, axis in zip((x1, y1, x2, y2), (w, h, w, h)))
            color = "#ff3030" if row["index"] == j else "#0878ff" if row["index"] == i else "#e6d532"
            draw.rectangle(box, outline=color, width=4 if row["index"] in (i, j) else 2)
            draw.text((max(0, box[0]), max(0, box[1] - 13)), f"{row['index']}:{row['description']}", fill=color, stroke_width=2, stroke_fill="black")
        view = VIEWS / f"rank-{entry['global_source_rank']:03d}-{entry['image_id']}.png"
        image.save(view)
        projection.append({"global_source_rank": entry["global_source_rank"], "source": entry["source"],
                           "image_id": entry["image_id"], "group": entry["group"], "batch_index": entry["batch_index"],
                           "split": entry["split"], "j": j, "i": i, "rows": rows,
                           "image_path": item["image_path"], "view": str(view), "image_size": [w, h],
                           "source_stop": entry["native_stop"], "excluded": entry["excluded_from_new_picks"],
                           "source_bindings": entry["source_bindings"]})
    (UNIT / "supporting/review-projection.json").write_text(json.dumps({"count": len(projection), "cases": projection}, indent=2) + "\n")
    print("verified 29/273 source-ordered earliest boundaries and rendered 29 views")


if __name__ == "__main__":
    main()
