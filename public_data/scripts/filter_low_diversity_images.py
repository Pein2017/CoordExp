#!/usr/bin/env python3
"""Filter out whole images (records) with low semantic diversity + high repetition.

This script DOES NOT drop individual objects. It either keeps the record as-is
or removes it entirely, matching the user's preference to avoid ambiguity.

Filtering criteria (all must be satisfied to DROP):
  - n_objects >= --min_objects
  - n_unique_classes <= --max_unique

Optional additional gates:
  - --min_top1_ratio: also require the most frequent class ratio >= threshold
  - --max_effective_classes: also require exp(entropy) <= threshold

Notes:
- Class name normalization matches training/eval convention: `desc.split('/')[0]`
- This is meant to remove pathological cases (e.g. hundreds of pumpkins/chairs/fruits).

Example:
  PYTHONPATH=. /root/miniconda3/envs/ms/bin/python \\
    public_data/scripts/filter_low_diversity_images.py \\
    --input  public_data/lvis/rescale_32_768_poly_20/train.coord.jsonl \\
    --output public_data/lvis/rescale_32_768_poly_20/train.coord.filtered_u8_m200.jsonl \\
    --min_objects 200 \\
    --max_unique 8 \\
    --stats_json output/lvis_train_filter_low_diversity.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _stream_jsonl(path: str) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def _norm_desc(desc: Any) -> str:
    if not isinstance(desc, str):
        return str(desc)
    return desc.strip().lower().split("/")[0]


def _effective_classes(counts: Counter[str]) -> float:
    total = float(sum(counts.values()))
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = float(c) / total
        h -= p * math.log(p + 1e-12)
    return float(math.exp(h)) if h > 0 else float(len(counts))


@dataclass
class Decision:
    drop: bool
    n_objects: int
    n_unique: int
    top1_ratio: float
    effective_classes: float
    top_labels: List[Tuple[str, int]]


def decide_drop(
    record: Dict[str, Any],
    *,
    min_objects: int,
    max_unique: int,
    min_top1_ratio: Optional[float],
    max_effective_classes: Optional[float],
    topn: int = 5,
    hard_max_objects: Optional[int] = None,
) -> Decision:
    objs = record.get("objects") or []
    if not isinstance(objs, list) or not objs:
        return Decision(
            drop=False,
            n_objects=int(len(objs)) if isinstance(objs, list) else 0,
            n_unique=0,
            top1_ratio=0.0,
            effective_classes=0.0,
            top_labels=[],
        )

    counts: Counter[str] = Counter()
    for o in objs:
        if not isinstance(o, dict):
            continue
        counts[_norm_desc(o.get("desc", ""))] += 1

    n_objects = int(len(objs))
    n_unique = int(len([k for k in counts.keys() if k]))
    top = counts.most_common(topn)
    top1 = int(top[0][1]) if top else 0
    top1_ratio = float(top1 / n_objects) if n_objects else 0.0
    eff = float(_effective_classes(counts))

    # Two drop modes:
    # 1) hard cap on instance count (unconditional)
    # 2) low-diversity dense images (optionally gated by repetition metrics)
    drop_hard = bool(
        hard_max_objects is not None and n_objects >= int(hard_max_objects)
    )
    drop_diversity = bool(n_objects >= int(min_objects) and n_unique <= int(max_unique))

    drop = drop_hard
    if not drop and drop_diversity:
        drop = True
        if min_top1_ratio is not None:
            drop = drop and (top1_ratio >= float(min_top1_ratio))
        if max_effective_classes is not None:
            drop = drop and (eff <= float(max_effective_classes))

    return Decision(
        drop=drop,
        n_objects=n_objects,
        n_unique=n_unique,
        top1_ratio=top1_ratio,
        effective_classes=eff,
        top_labels=[(k, int(v)) for k, v in top],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--min_objects", type=int, default=200)
    ap.add_argument("--max_unique", type=int, default=8)
    ap.add_argument(
        "--hard_max_objects",
        type=int,
        default=None,
        help="Optional: drop any record with n_objects >= this (regardless of diversity).",
    )
    ap.add_argument(
        "--min_top1_ratio",
        type=float,
        default=None,
        help="Optional extra gate: require top1_ratio >= this to drop.",
    )
    ap.add_argument(
        "--max_effective_classes",
        type=float,
        default=None,
        help="Optional extra gate: require effective_classes <= this to drop.",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stats_json", type=str, default=None)
    args = ap.parse_args()

    scanned = 0
    kept = 0
    dropped = 0

    dropped_examples: List[Dict[str, Any]] = []
    kept_top_by_objects: List[Dict[str, Any]] = []

    def _maybe_add_top_by_objects(entry: Dict[str, Any], k: int = 20) -> None:
        nonlocal kept_top_by_objects
        kept_top_by_objects.append(entry)
        kept_top_by_objects = sorted(
            kept_top_by_objects, key=lambda x: int(x.get("n_objects", 0)), reverse=True
        )[:k]

    with open(args.output, "w", encoding="utf-8") as out_f:
        for line_no, rec in _stream_jsonl(args.input):
            scanned += 1
            if args.limit is not None and scanned > int(args.limit):
                break

            d = decide_drop(
                rec,
                min_objects=int(args.min_objects),
                max_unique=int(args.max_unique),
                min_top1_ratio=args.min_top1_ratio,
                max_effective_classes=args.max_effective_classes,
                hard_max_objects=args.hard_max_objects,
            )

            images = rec.get("images") or []
            img0 = images[0] if isinstance(images, list) and images else None

            if d.drop:
                dropped += 1
                if len(dropped_examples) < 20:
                    dropped_examples.append(
                        {
                            "line": int(line_no),
                            "image": img0,
                            "n_objects": int(d.n_objects),
                            "n_unique": int(d.n_unique),
                            "top1_ratio": float(d.top1_ratio),
                            "effective_classes": float(d.effective_classes),
                            "top_labels": list(d.top_labels),
                        }
                    )
                continue

            kept += 1
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            _maybe_add_top_by_objects(
                {
                    "line": int(line_no),
                    "image": img0,
                    "n_objects": int(d.n_objects),
                    "n_unique": int(d.n_unique),
                    "top1_ratio": float(d.top1_ratio),
                    "effective_classes": float(d.effective_classes),
                    "top_labels": list(d.top_labels),
                }
            )

    summary = {
        "config": {
            "input": args.input,
            "output": args.output,
            "min_objects": int(args.min_objects),
            "max_unique": int(args.max_unique),
            "hard_max_objects": args.hard_max_objects,
            "min_top1_ratio": args.min_top1_ratio,
            "max_effective_classes": args.max_effective_classes,
            "limit": args.limit,
        },
        "counts": {"scanned": int(scanned), "kept": int(kept), "dropped": int(dropped)},
        "dropped_examples": dropped_examples,
        "kept_top_by_objects": kept_top_by_objects,
    }

    print("Done")
    print(f"- scanned: {scanned}")
    print(f"- kept: {kept}")
    print(f"- dropped: {dropped}")
    if args.stats_json:
        with open(args.stats_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Wrote: {args.stats_json}")


if __name__ == "__main__":
    main()
