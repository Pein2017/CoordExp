"""Render exact-fixture coordinate hypotheses; never manufacture a parsed owner."""
from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

REPO = Path("/data/CoordExp/.worktrees/coordexp-infras")
ROOT = Path("/data/CoordExp/outputs/infra_base/optimization-20260912")
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from src.data.geometry import coord_bins_to_pixel_xyxy
from src.vis.normalization import load_visual_rows


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


composition_path = ROOT / "vllm-qualification-final-01/evidence/composition/composition-receipt.json"
composition = json.loads(composition_path.read_text())
fixture = composition["fixture_identity"]
run = ROOT / "infer/candidate-fixed-ddp-02-hf"
row = next(row for row in load_visual_rows(run).rows if row.row_id == fixture["row_id"])
raw = next(json.loads(line) for line in (run / "gt_vs_pred.jsonl").read_text().splitlines()
           if json.loads(line)["row_id"] == row.row_id)
image_plan = next(json.loads(line) for line in (run / "image_plan.jsonl").read_text().splitlines()
                  if json.loads(line)["row_id"] == row.row_id)
assert sha(row.image_path) == fixture["image_file_sha256"]
assert sha(ROOT / "inputs/infer-four.jsonl") == fixture["input_jsonl_sha256"]
assert image_plan["executed_media_sha256"] == fixture["executed_media_sha256"]
assert image_plan["logical_transform_id"] == "identity"
assert not row.pred and raw["valid_prediction_count"] == 0
image = Image.open(row.image_path).convert("RGB")
assert image.size == (row.image_width, row.image_height)
width, height = image.size
# Public coordinate conversion, with valid reference rectangles. These are
# hypothetical point locations, not model boxes or a repair of the malformed XML.
points = []
for y_bin in (733, 858):
    x_px, y_px, _, _ = coord_bins_to_pixel_xyxy(
        [733, y_bin, 999, 999], image_width=width,
        image_height=height, field="diagnostic.hypothetical_reference_rectangle",
    )
    contained = [obj.index for obj in row.gt
                 if obj.bbox_pixel_xyxy[0] <= x_px <= obj.bbox_pixel_xyxy[2]
                 and obj.bbox_pixel_xyxy[1] <= y_px <= obj.bbox_pixel_xyxy[3]]
    points.append({"x_bin": 733, "y_bin": y_bin, "x_pixel": x_px,
                   "y_pixel": y_px, "containing_gt_indices": contained})

fig = plt.figure(figsize=(14, 10.8), facecolor="#ffffff")
ax = fig.add_axes([0.035, 0.17, 0.93, 0.775])
ax.imshow(image, extent=(0, width, height, 0))
for obj in row.gt:
    x1, y1, x2, y2 = obj.bbox_pixel_xyxy
    ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False,
                           edgecolor="#b2f2bb", linewidth=1.1, alpha=.7))

colors = ("#00dcff", "#ff49b4")
for point, color in zip(points, colors):
    y = point["y_pixel"]
    ax.axhline(y, color="black", linewidth=3.6, alpha=.8)
    ax.axhline(y, color=color, linewidth=2)
    ax.scatter([point["x_pixel"]], [y], s=150, facecolors="none",
               edgecolors="black", linewidths=4, zorder=6)
    ax.scatter([point["x_pixel"]], [y], s=150, facecolors="none",
               edgecolors=color, linewidths=2, zorder=7)
    ax.text(18, y-10, f"y = {point['y_bin']}  ({y} px)", color=color,
            fontsize=12, weight="bold",
            bbox={"facecolor":"#111111", "edgecolor":"none", "alpha":.85, "pad":4})
ax.axvline(points[0]["x_pixel"], color="white", linestyle=(0,(4,5)), linewidth=1.3)
ax.text(points[0]["x_pixel"]-8, 35, "x = 733 (hypothesis only)", ha="right",
        color="white", fontsize=10,
        bbox={"facecolor":"#111111", "edgecolor":"none", "alpha":.8, "pad":4})
ax.annotate("", xy=(points[0]["x_pixel"]+38, points[0]["y_pixel"]),
            xytext=(points[0]["x_pixel"]+38, points[1]["y_pixel"]),
            arrowprops={"arrowstyle":"<->", "color":"white", "lw":1.6})
ax.text(points[0]["x_pixel"]+52, (points[0]["y_pixel"]+points[1]["y_pixel"])/2,
        "125 bins\n104 px", va="center", color="white", fontsize=10,
        bbox={"facecolor":"#111111", "edgecolor":"none", "alpha":.8, "pad":4})
ax.set_xlim(0, width)
ax.set_ylim(height, 0)
ax.set_xlabel("Pixel x", fontsize=10)
ax.set_ylabel("Pixel y", fontsize=10)
fig.suptitle("COCO 000000000139 | y-token 733 vs 858", y=.981,
             fontsize=18, weight="bold")
fig.text(.04, .105,
         "Cyan / magenta: hypothetical y positions of the differing token.  Pale green: all 20 GT boxes.",
         fontsize=11)
fig.text(.04, .073,
         "Circles hold x=733 fixed (first x1 token); both fall outside every annotated GT box.", fontsize=11)
fig.text(.04, .040,
         "Owner is UNASSIGNED: generated XML has three tokens per x1/y1, generic text, and no valid object box.",
         fontsize=11, weight="bold", color="#822b2b")
png = OUT / "coco139_coord733_vs858.png"
fig.savefig(png, dpi=150, facecolor=fig.get_facecolor())
plt.close(fig)

manifest = {
    "status": "rendered_geometry_hypotheses_owner_unassigned",
    "row_id": row.row_id,
    "image": {"path": str(row.image_path), "sha256": sha(row.image_path),
              "width": width, "height": height, "logical_transform": "identity"},
    "composition_receipt": {"path": str(composition_path), "sha256": sha(composition_path)},
    "input_jsonl": {"path": str(ROOT / "inputs/infer-four.jsonl"),
                    "sha256": fixture["input_jsonl_sha256"]},
    "gt_source": {"run_dir": str(run), "raw_sha256": sha(run / "gt_vs_pred.jsonl"),
                  "scored_sha256": sha(run / "gt_vs_pred_scored.jsonl"),
                  "use": "same-image ground truth only; batched HF text is not the composition continuation"},
    "generated_index_zero_based": 12,
    "generated_field": "first token in malformed triple-valued y1 attribute",
    "object_owner": None,
    "valid_parsed_prediction_count": 0,
    "coordinate_conversion": "src.data.geometry.coord_bins_to_pixel_xyxy; round(bin * extent / 1000)",
    "assumption": "Interpret first x1 token as x=733 and first y1 token as y; not a valid parse or object assignment",
    "points": points,
    "y_pixel_delta": points[1]["y_pixel"] - points[0]["y_pixel"],
    "gt": [{"index": obj.index, "description": obj.description,
            "object_id": raw["gt"][obj.index]["object_id"],
            "bbox_bins": list(obj.source_bbox), "bbox_pixel_xyxy": list(obj.bbox_pixel_xyxy)}
           for obj in row.gt],
    "script_sha256": sha(__file__),
    "output_png": str(png),
    "output_sha256": sha(png),
}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
print(json.dumps({"output_png": str(png), "points": points, "gt_count": len(row.gt),
                  "owner": None, "manifest": str(OUT / "manifest.json")}, indent=2))
