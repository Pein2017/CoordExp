"""Render the bounded c01 late-suffix overlap counterexample from saved records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
PACKET = ROOT / "packet-v3.json"
RECORDS = ROOT / "smoke-02" / "records.jsonl"
OUT = ROOT / "smoke-02-successor-visualizations" / "h_plus_351017-c01-late-bottles.png"
MANIFEST = ROOT / "smoke-02-successor-visualizations" / "late-counterexample.json"
JOB_ID = "h_plus_351017-c01"
ORDERS = tuple(range(23, 30))
PAIR = (25, 26)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def iou(a: list[int], b: list[int]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / (area_a + area_b - intersection)


def main() -> int:
    packet = json.loads(PACKET.read_text())
    case = next(item for item in packet["cases"] if item["case_id"] == "351017")
    records = [json.loads(line) for line in RECORDS.read_text().splitlines() if line]
    record = next(item for item in records if item["job_id"] == JOB_ID)
    rows = {
        row["generated_order"]: row
        for row in record["parser_partition"]["valid_free_rows"]
        if row["generated_order"] in ORDERS
    }
    if tuple(sorted(rows)) != ORDERS:
        raise ValueError(f"missing requested orders: {sorted(rows)}")

    image = Image.open(case["source_case"]["image_path"]).convert("RGB")
    header = 92
    canvas = Image.new("RGB", (image.width, image.height + header), "white")
    canvas.paste(image, (0, header))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=18)
    draw.text((12, 8), "351017 c01 fresh late suffix: free orders 23-29", fill=(0, 0, 0), font=font)
    draw.text((12, 34), "Red orders 25/26: IoU below strict >.95 metric, but likely same bottle owner", fill=(180, 0, 0), font=font)
    draw.text((12, 60), "Forced table and native h omitted; no row is GT-certified", fill=(50, 50, 50), font=font)
    for order in ORDERS:
        row = rows[order]
        x1, y1, x2, y2 = row["bbox"]
        box = (x1, y1 + header, x2, y2 + header)
        color = (235, 0, 0) if order in PAIR else (0, 180, 50)
        draw.rectangle(box, outline=color, width=4)
        label = f"F{order} {row['description']}"
        label_y = max(header, y1 + header - 21)
        draw.rectangle((x1, label_y, x1 + 9 * len(label), label_y + 21), fill=(0, 0, 0))
        draw.text((x1 + 2, label_y + 1), label, fill=color, font=font)
    canvas.save(OUT)

    pair_iou = iou(rows[PAIR[0]]["bbox"], rows[PAIR[1]]["bbox"])
    payload = {
        "schema": "native_escape_witness.late_counterexample.v1",
        "packet": str(PACKET),
        "packet_sha256": sha(PACKET),
        "records": str(RECORDS),
        "records_sha256": sha(RECORDS),
        "job_id": JOB_ID,
        "selected_generated_orders": list(ORDERS),
        "rows": [rows[order] for order in ORDERS],
        "counterexample_pair": {
            "generated_orders": list(PAIR),
            "class_blind_iou": pair_iou,
            "strict_iou_gt_0_95": pair_iou > 0.95,
            "visual_interpretation": "likely_same_physical_bottle_owner_pending_root_adjudication",
        },
        "figure": str(OUT),
        "figure_sha256": sha(OUT),
        "producer": str(Path(__file__).resolve()),
        "producer_sha256": sha(Path(__file__)),
        "claim_boundary": (
            "This bounded overlay demonstrates why zero strict IoU>.95 recurrence is not "
            "duplicate-free owner enumeration. It does not change the frozen metric."
        ),
    }
    MANIFEST.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(MANIFEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
