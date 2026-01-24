#!/usr/bin/env python3
"""Filter CoordExp JSONL by a hard max number of objects per image.

This is a dataset-only operation (no model/tokenizer required) and is useful for:
- preventing extreme long assistant outputs on crowded images
- enforcing a strict max sequence length budget (e.g. vLLM max_model_len)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class Stats:
    images_seen: int = 0
    images_written: int = 0
    images_dropped: int = 0

    objects_seen: int = 0
    objects_written: int = 0
    objects_bbox: int = 0
    objects_poly: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            **self.__dict__,
            "poly_ratio": float(self.objects_poly / max(1, self.objects_written)),
            "bbox_ratio": float(self.objects_bbox / max(1, self.objects_written)),
        }


def _count_types(objects: List[Dict[str, Any]]) -> tuple[int, int]:
    polys = 0
    bboxes = 0
    for o in objects:
        if not isinstance(o, dict):
            continue
        if "poly" in o:
            polys += 1
        if "bbox_2d" in o:
            bboxes += 1
    return polys, bboxes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-objects", type=int, required=True)
    ap.add_argument("--stats-json", type=Path, default=None)
    args = ap.parse_args()

    max_objects = int(args.max_objects)
    if max_objects <= 0:
        raise SystemExit("--max-objects must be > 0")

    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = Stats()
    with (
        args.input.open("r", encoding="utf-8") as fin,
        out_path.open("w", encoding="utf-8") as fout,
    ):
        for line in fin:
            line = line.strip()
            if not line:
                continue
            stats.images_seen += 1
            rec = json.loads(line)
            objects = rec.get("objects") or []
            n = len(objects) if isinstance(objects, list) else 0
            stats.objects_seen += int(n)
            if n > max_objects:
                stats.images_dropped += 1
                continue
            stats.images_written += 1
            stats.objects_written += int(n)
            if isinstance(objects, list) and objects:
                polys, bboxes = _count_types(objects)
                stats.objects_poly += int(polys)
                stats.objects_bbox += int(bboxes)
            fout.write(line + "\n")

    print(json.dumps(stats.as_dict(), ensure_ascii=False, indent=2))
    if args.stats_json:
        args.stats_json.parent.mkdir(parents=True, exist_ok=True)
        args.stats_json.write_text(
            json.dumps(stats.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Wrote: {args.stats_json}")


if __name__ == "__main__":
    main()
