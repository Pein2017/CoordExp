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
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.vis.normalization import load_visual_rows
from src.vis.matching import iou_xyxy

arms = ["dynamic-hf-bf16", "materialized-hf-bf16"]
rows = [load_visual_rows(ROOT / "infer" / arm).rows[0] for arm in arms]
assert rows[0].gt_signature() == rows[1].gt_signature()
assert rows[0].row_id == "coco2017_val_000000000139"
raw = json.loads((ROOT / "infer" / arms[0] / "gt_vs_pred.jsonl").open().readline())
gt_index = next(i for i, obj in enumerate(raw["gt"]) if obj["object_id"] == "1667817")
gt = rows[0].gt[gt_index]
predictions = [max((p for p in row.pred if p.description == "vase"),
                   key=lambda p: iou_xyxy(gt.bbox_pixel_xyxy, p.bbox_pixel_xyxy)) for row in rows]
ious = [iou_xyxy(gt.bbox_pixel_xyxy, p.bbox_pixel_xyxy) for p in predictions]
assert all(value >= .5 for value in ious)
image = Image.open(rows[0].image_path).convert("RGB")
fig, axs = plt.subplots(1, 2, figsize=(7.5, 7), facecolor="white")
colors = ["#00c8ee", "#ed369b"]
for ax, pred, color, label, overlap in zip(axs, predictions, colors,
                                         ["Dynamic HF", "Merged BF16 HF"], ious):
    ax.imshow(image, extent=(0, image.width, image.height, 0))
    for box, c, linestyle, title in [(gt.bbox_pixel_xyxy, "#ffe100", "--", "GT vase"),
                                     (pred.bbox_pixel_xyxy, color, "-", label)]:
        x1, y1, x2, y2 = box
        ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False,
                               edgecolor=c, linewidth=2, linestyle=linestyle, label=title))
    ax.set_xlim(1015, 1195)
    ax.set_ylim(820, 560)
    ax.set_title(f"{label}\nIoU with GT = {overlap:.4f}", fontsize=12)
    ax.set_xlabel(f"coord bins: {list(pred.source_coord_bins)}", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
fig.suptitle("COCO139 | exact GT vase owner 1667817", fontsize=14, weight="bold")
fig.text(.5, .045, "GT bins: [858, 726, 915, 937] | yellow dashed box", ha="center", fontsize=10)
fig.text(.5, .017, "Mature step-2444, native prompt; different output from the earlier four-step smoke.",
         ha="center", fontsize=9)
fig.subplots_adjust(top=.88, bottom=.13, left=.07, right=.99, wspace=.25)
png = Path(__file__).with_name("coco139_vase_owner_comparison.png")
fig.savefig(png, dpi=160)
plt.close(fig)
evidence = {"row_id": rows[0].row_id, "gt_object_id": "1667817", "gt_bins": list(gt.source_bbox),
            "selection": "Each arm's same-class vase with greatest IoU to fixed GT annotation1667817",
            "arms": [{"arm": arm, "prediction_index": p.index, "bins": list(p.source_coord_bins),
                      "pixel_xyxy": list(p.bbox_pixel_xyxy), "iou_with_gt": value}
                     for arm, p, value in zip(arms, predictions, ious)],
            "paired_prediction_iou": iou_xyxy(*(p.bbox_pixel_xyxy for p in predictions)),
            "maximum_coordinate_bin_difference": max(abs(a-b) for a,b in zip(
                predictions[0].source_coord_bins, predictions[1].source_coord_bins)),
            "image_path": str(rows[0].image_path),
            "image_sha256": hashlib.sha256(rows[0].image_path.read_bytes()).hexdigest(),
            "png": str(png), "png_sha256": hashlib.sha256(png.read_bytes()).hexdigest()}
png.with_suffix(".json").write_text(json.dumps(evidence, indent=2)+"\n")
print(json.dumps(evidence, indent=2))
