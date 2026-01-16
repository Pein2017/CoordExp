"""Analyze 'long' / dense samples for duplicated classes inside LVIS-style JSONLs.

Motivation:
Some images contain hundreds of nearly identical instances (e.g., fruits, chairs),
leading to extremely long GT assistant JSON and low semantic diversity.

This script computes per-record label distribution metrics and summarizes the
subset of "dense" records.

Example:
  PYTHONPATH=. /root/miniconda3/envs/ms/bin/python \
    scripts/analyze_repetitive_long_samples.py \
    --jsonl public_data/lvis/rescale_32_768_poly_20/train.coord.jsonl \
    --min_objects 200 \
    --out_json output/lvis_train_dense_dup_stats.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


def _stream_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def _norm_desc(desc: Any) -> str:
    if not isinstance(desc, str):
        return str(desc)
    # Match builder/evaluator behavior: normalize "a/b" -> "a"
    return desc.strip().lower().split("/")[0]


def _entropy_from_counts(counts: List[int]) -> float:
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = float(c) / total
        h -= p * math.log(p + 1e-12)
    return h


@dataclass
class RecordMetrics:
    line: int
    image: Any
    n_objects: int
    n_unique: int
    top1: int
    top1_ratio: float
    top5_ratio: float
    entropy: float
    effective_classes: float
    polys: int
    bboxes: int
    max_poly_points: int
    top_labels: List[Tuple[str, int]]


def _measure_record(line: int, record: Dict[str, Any], topn: int = 10) -> RecordMetrics:
    objs = record.get("objects") or []
    if not isinstance(objs, list):
        objs = []
    c: Counter[str] = Counter()
    polys = 0
    bboxes = 0
    max_pp = 0
    for o in objs:
        if not isinstance(o, dict):
            continue
        c[_norm_desc(o.get("desc", ""))] += 1
        if "poly" in o:
            polys += 1
            pp = o.get("poly_points")
            if pp is not None:
                try:
                    max_pp = max(max_pp, int(pp))
                except Exception:
                    pass
        if "bbox_2d" in o:
            bboxes += 1
    n = len(objs)
    top = c.most_common(topn)
    counts = [v for _, v in c.items() if _]
    n_unique = len([k for k, v in c.items() if k])
    top1 = top[0][1] if top else 0
    top5 = sum(v for _, v in top[:5]) if top else 0
    h = _entropy_from_counts(counts)
    eff = float(math.exp(h)) if h > 0 else float(n_unique)
    images = record.get("images") or []
    img0 = images[0] if isinstance(images, list) and images else None
    return RecordMetrics(
        line=int(line),
        image=img0,
        n_objects=int(n),
        n_unique=int(n_unique),
        top1=int(top1),
        top1_ratio=float(top1 / n) if n else 0.0,
        top5_ratio=float(top5 / n) if n else 0.0,
        entropy=float(h),
        effective_classes=float(eff),
        polys=int(polys),
        bboxes=int(bboxes),
        max_poly_points=int(max_pp),
        top_labels=[(k, int(v)) for k, v in top],
    )


def _pct(arr: np.ndarray, ps: List[float]) -> Dict[str, float]:
    return {f"p{p:g}": float(np.percentile(arr, p)) for p in ps}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=str, required=True)
    ap.add_argument("--min_objects", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--topk_examples", type=int, default=10)
    ap.add_argument("--out_json", type=str, default=None)
    args = ap.parse_args()

    min_objects = int(args.min_objects)
    topk = int(args.topk_examples)

    dense: List[RecordMetrics] = []
    seen = 0
    for line, rec in _stream_jsonl(args.jsonl):
        seen += 1
        if args.limit is not None and seen > int(args.limit):
            break
        m = _measure_record(line, rec, topn=10)
        if m.n_objects >= min_objects:
            dense.append(m)

    print(f"Scanned: {seen} records")
    print(f"Dense subset: n={len(dense)} where n_objects>={min_objects}")

    out: Dict[str, Any] = {
        "config": {
            "jsonl": args.jsonl,
            "min_objects": min_objects,
            "limit": args.limit,
        },
        "counts": {"scanned": int(seen), "dense": int(len(dense))},
    }

    if dense:
        n_obj = np.asarray([m.n_objects for m in dense], dtype=np.int32)
        n_unique = np.asarray([m.n_unique for m in dense], dtype=np.int32)
        top1r = np.asarray([m.top1_ratio for m in dense], dtype=np.float32)
        top5r = np.asarray([m.top5_ratio for m in dense], dtype=np.float32)
        eff = np.asarray([m.effective_classes for m in dense], dtype=np.float32)
        polys = np.asarray([m.polys for m in dense], dtype=np.int32)
        max_pp = np.asarray([m.max_poly_points for m in dense], dtype=np.int32)

        out["dense_summary"] = {
            "n_objects": {
                "min": int(n_obj.min()),
                "max": int(n_obj.max()),
                "percentiles": _pct(n_obj, [50, 90, 95, 99, 99.5, 99.9]),
            },
            "n_unique": {
                "min": int(n_unique.min()),
                "max": int(n_unique.max()),
                "percentiles": _pct(n_unique, [50, 90, 95, 99]),
            },
            "top1_ratio": {
                "min": float(top1r.min()),
                "max": float(top1r.max()),
                "percentiles": _pct(top1r, [50, 75, 90, 95, 99]),
            },
            "top5_ratio": {
                "min": float(top5r.min()),
                "max": float(top5r.max()),
                "percentiles": _pct(top5r, [50, 75, 90, 95, 99]),
            },
            "effective_classes": {
                "min": float(eff.min()),
                "max": float(eff.max()),
                "percentiles": _pct(eff, [50, 90, 95, 99]),
            },
            "polys": {
                "min": int(polys.min()),
                "max": int(polys.max()),
                "percentiles": _pct(polys, [50, 90, 95, 99]),
            },
            "max_poly_points": {
                "min": int(max_pp.min()),
                "max": int(max_pp.max()),
                "percentiles": _pct(max_pp, [50, 90, 95, 99]),
            },
        }

        # Example lists for manual inspection / policy tuning
        by_objects = sorted(dense, key=lambda m: m.n_objects, reverse=True)[:topk]
        by_top1 = sorted(dense, key=lambda m: (m.top1_ratio, m.n_objects), reverse=True)[
            :topk
        ]
        by_low_unique = sorted(dense, key=lambda m: (m.n_unique, -m.n_objects))[:topk]

        def pack(ms: List[RecordMetrics]) -> List[Dict[str, Any]]:
            return [
                {
                    "line": int(m.line),
                    "image": m.image,
                    "n_objects": int(m.n_objects),
                    "n_unique": int(m.n_unique),
                    "top1_ratio": float(m.top1_ratio),
                    "top_labels": list(m.top_labels),
                    "polys": int(m.polys),
                    "bboxes": int(m.bboxes),
                    "max_poly_points": int(m.max_poly_points),
                }
                for m in ms
            ]

        out["examples"] = {
            "top_by_objects": pack(by_objects),
            "top_by_top1_ratio": pack(by_top1),
            "top_by_low_unique": pack(by_low_unique),
        }

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Wrote: {args.out_json}")


if __name__ == "__main__":
    main()

