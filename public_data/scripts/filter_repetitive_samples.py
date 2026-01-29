#!/usr/bin/env python3
"""Filter/compact dense samples with many duplicated classes.

Goal:
Keep semantic richness (diverse classes) while avoiding pathological records
with hundreds of near-identical instances that explode GT token length.

Policy (default, configurable):
1) (Optional) Drop *purely repetitive* records:
   - n_objects >= drop_min_objects
   - n_unique <= drop_max_unique
   - top1_ratio >= drop_min_top1_ratio
2) Otherwise, compact the objects list:
   - Always keep at least 1 instance per class.
   - Cap per-class instances: cap_per_class
   - Cap total objects per record: max_total_objects
   - Prefer spatially diverse instances per class (grid bucketing on centers).
3) (Optional) Geometry simplification to reduce tokens:
   - For frequent classes (count >= poly_to_bbox_min_class_count), convert poly -> bbox_2d.

Examples:
  # Produce a filtered coord-token JSONL
  PYTHONPATH=. /root/miniconda3/envs/ms/bin/python \
    public_data/scripts/filter_repetitive_samples.py \
    --input  public_data/lvis/rescale_32_768_poly_20/train.coord.jsonl \
    --output public_data/lvis/rescale_32_768_poly_20/train.coord.filtered.jsonl \
    --max_total_objects 200 \
    --cap_per_class 40 \
    --drop_min_objects 400 \
    --drop_max_unique 2 \
    --drop_min_top1_ratio 0.99 \
    --poly_to_bbox_min_class_count 80 \
    --stats_json output/lvis_train_filter_stats.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

from src.coord_tokens.codec import int_to_token, tokens_to_ints
from src.datasets.utils import sort_objects_by_topleft


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


def _object_center(obj: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    # Works for coord tokens or numeric ints.
    if "bbox_2d" in obj and obj["bbox_2d"] is not None:
        vals = obj["bbox_2d"]
        if isinstance(vals, Sequence) and len(vals) >= 4:
            ints = tokens_to_ints(vals[:4], require_even=True)
            x1, y1, x2, y2 = ints[:4]
            return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    if "poly" in obj and obj["poly"] is not None:
        vals = obj["poly"]
        if isinstance(vals, Sequence) and len(vals) >= 2:
            ints = tokens_to_ints(vals, require_even=True)
            xs = ints[0::2]
            ys = ints[1::2]
            if xs and ys:
                return (float(sum(xs)) / len(xs), float(sum(ys)) / len(ys))
    return None


def _poly_to_bbox(obj: Dict[str, Any]) -> bool:
    poly = obj.get("poly")
    if poly is None or not isinstance(poly, Sequence) or len(poly) < 6:
        return False
    ints = tokens_to_ints(poly, require_even=True)
    xs = ints[0::2]
    ys = ints[1::2]
    if not xs or not ys:
        return False
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    obj["bbox_2d"] = [int_to_token(x1), int_to_token(y1), int_to_token(x2), int_to_token(y2)]
    if "poly" in obj:
        obj.pop("poly", None)
    obj.pop("poly_points", None)
    return True


def _select_spatially_diverse(
    indices: List[int],
    objects: List[Dict[str, Any]],
    *,
    cap: int,
    grid: int,
    rng: random.Random,
) -> List[int]:
    """Pick up to cap indices, preferring unique grid cells of object centers."""
    if cap <= 0 or not indices:
        return []
    if len(indices) <= cap:
        return list(indices)
    buckets: DefaultDict[Tuple[int, int], List[int]] = defaultdict(list)
    fallback: List[int] = []
    for idx in indices:
        c = _object_center(objects[idx])
        if c is None:
            fallback.append(idx)
            continue
        x, y = c
        gx = int(max(0, min(grid - 1, math.floor((x / 999.0) * grid))))
        gy = int(max(0, min(grid - 1, math.floor((y / 999.0) * grid))))
        buckets[(gx, gy)].append(idx)

    # deterministic but slightly randomized within cell (seeded rng)
    cells = list(buckets.keys())
    rng.shuffle(cells)
    selected: List[int] = []
    for cell in cells:
        cand = buckets[cell]
        rng.shuffle(cand)
        selected.append(cand[0])
        if len(selected) >= cap:
            return selected

    # Fill remaining from leftover in each cell, then fallback
    leftovers: List[int] = []
    for cell in cells:
        leftovers.extend(buckets[cell][1:])
    leftovers.extend(fallback)
    rng.shuffle(leftovers)
    for idx in leftovers:
        if idx in selected:
            continue
        selected.append(idx)
        if len(selected) >= cap:
            break
    return selected


@dataclass
class FilterDecision:
    dropped: bool
    original_objects: int
    new_objects: int
    n_unique: int
    top1_ratio: float
    poly_to_bbox: int


def filter_record(
    record: Dict[str, Any],
    *,
    max_total_objects: int,
    cap_per_class: int,
    grid: int,
    seed: int,
    drop_min_objects: int,
    drop_max_unique: int,
    drop_min_top1_ratio: float,
    poly_to_bbox_min_class_count: int,
) -> FilterDecision:
    objects = record.get("objects") or []
    if not isinstance(objects, list) or not objects:
        return FilterDecision(
            dropped=False,
            original_objects=int(len(objects)) if isinstance(objects, list) else 0,
            new_objects=int(len(objects)) if isinstance(objects, list) else 0,
            n_unique=0,
            top1_ratio=0.0,
            poly_to_bbox=0,
        )

    # Normalize class labels
    labels = [_norm_desc(o.get("desc", "")) if isinstance(o, dict) else "" for o in objects]
    counts = Counter([l for l in labels if l])
    n_obj = len(objects)
    n_unique = len(counts)
    top1 = counts.most_common(1)[0][1] if counts else 0
    top1_ratio = float(top1 / n_obj) if n_obj else 0.0

    # Drop purely repetitive / uninformative records (configurable)
    if (
        n_obj >= int(drop_min_objects)
        and n_unique <= int(drop_max_unique)
        and top1_ratio >= float(drop_min_top1_ratio)
    ):
        return FilterDecision(
            dropped=True,
            original_objects=n_obj,
            new_objects=0,
            n_unique=n_unique,
            top1_ratio=top1_ratio,
            poly_to_bbox=0,
        )

    rng = random.Random((seed ^ (n_obj * 1000003)) & 0xFFFFFFFF)

    # Optional: simplify poly->bbox for very frequent classes (reduces coord tokens a lot)
    poly_to_bbox = 0
    if poly_to_bbox_min_class_count > 0 and counts:
        frequent = {k for k, v in counts.items() if v >= int(poly_to_bbox_min_class_count)}
        if frequent:
            for obj, lab in zip(objects, labels):
                if not isinstance(obj, dict):
                    continue
                if lab in frequent and "poly" in obj and obj.get("poly") is not None:
                    if _poly_to_bbox(obj):
                        poly_to_bbox += 1

    # If not dense, keep as-is
    if n_obj <= max_total_objects and all(v <= cap_per_class for v in counts.values()):
        record["objects"] = sort_objects_by_topleft(objects)
        return FilterDecision(
            dropped=False,
            original_objects=n_obj,
            new_objects=n_obj,
            n_unique=n_unique,
            top1_ratio=top1_ratio,
            poly_to_bbox=poly_to_bbox,
        )

    # Group indices by label
    by_class: DefaultDict[str, List[int]] = defaultdict(list)
    for idx, lab in enumerate(labels):
        by_class[lab].append(idx)

    selected: List[int] = []

    # Phase 1: always keep at least 1 per class (preserve semantic coverage)
    for lab, idxs in sorted(by_class.items(), key=lambda kv: (len(kv[1]), kv[0])):
        keep = _select_spatially_diverse(
            idxs, objects, cap=1, grid=grid, rng=rng
        )
        selected.extend(keep)

    # Phase 2: allocate remaining budget, capping per class
    remaining_budget = max(0, int(max_total_objects) - len(selected))
    if remaining_budget > 0:
        # Iterate classes from most frequent to least to keep some density,
        # but still bounded by cap_per_class.
        classes = sorted(by_class.keys(), key=lambda k: len(by_class[k]), reverse=True)
        per_class_selected: DefaultDict[str, int] = defaultdict(int)
        for idx in selected:
            per_class_selected[labels[idx]] += 1

        # Round-robin add instances to balance spatial coverage.
        progress = True
        while remaining_budget > 0 and progress:
            progress = False
            for lab in classes:
                if remaining_budget <= 0:
                    break
                if per_class_selected[lab] >= int(cap_per_class):
                    continue
                idxs = by_class[lab]
                # pick next diverse instance set up to cap_per_class, then take one new.
                target = min(int(cap_per_class), per_class_selected[lab] + 1)
                cand = _select_spatially_diverse(
                    idxs, objects, cap=target, grid=grid, rng=rng
                )
                # cand contains the first target choices; add the newly available one if present.
                if len(cand) > per_class_selected[lab]:
                    new_idx = cand[per_class_selected[lab]]
                    if new_idx not in selected:
                        selected.append(new_idx)
                        per_class_selected[lab] += 1
                        remaining_budget -= 1
                        progress = True

    # Materialize + keep deterministic ordering
    keep_set = set(selected)
    filtered = [o for i, o in enumerate(objects) if i in keep_set]
    record["objects"] = sort_objects_by_topleft(filtered)

    return FilterDecision(
        dropped=False,
        original_objects=n_obj,
        new_objects=len(filtered),
        n_unique=n_unique,
        top1_ratio=top1_ratio,
        poly_to_bbox=poly_to_bbox,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--max_total_objects", type=int, default=200)
    ap.add_argument("--cap_per_class", type=int, default=40)
    ap.add_argument("--grid", type=int, default=10)

    ap.add_argument("--drop_min_objects", type=int, default=400)
    ap.add_argument("--drop_max_unique", type=int, default=1)
    ap.add_argument("--drop_min_top1_ratio", type=float, default=0.999)

    ap.add_argument(
        "--poly_to_bbox_min_class_count",
        type=int,
        default=0,
        help="If >0, convert poly->bbox_2d for classes with count >= this value (per record).",
    )

    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stats_json", type=str, default=None)
    args = ap.parse_args()

    kept = 0
    dropped = 0
    total_in = 0
    total_out = 0
    poly_to_bbox = 0

    examples_dropped: List[Dict[str, Any]] = []
    examples_compacted: List[Dict[str, Any]] = []

    with open(args.output, "w", encoding="utf-8") as out_f:
        for idx, (line_no, rec) in enumerate(_stream_jsonl(args.input), start=1):
            if args.limit is not None and idx > int(args.limit):
                break

            decision = filter_record(
                rec,
                max_total_objects=int(args.max_total_objects),
                cap_per_class=int(args.cap_per_class),
                grid=int(args.grid),
                seed=int(args.seed),
                drop_min_objects=int(args.drop_min_objects),
                drop_max_unique=int(args.drop_max_unique),
                drop_min_top1_ratio=float(args.drop_min_top1_ratio),
                poly_to_bbox_min_class_count=int(args.poly_to_bbox_min_class_count),
            )

            total_in += int(decision.original_objects)
            poly_to_bbox += int(decision.poly_to_bbox)

            if decision.dropped:
                dropped += 1
                if len(examples_dropped) < 10:
                    images = rec.get("images") or []
                    img0 = images[0] if isinstance(images, list) and images else None
                    examples_dropped.append(
                        {
                            "line": int(line_no),
                            "image": img0,
                            "objects": int(decision.original_objects),
                            "n_unique": int(decision.n_unique),
                            "top1_ratio": float(decision.top1_ratio),
                        }
                    )
                continue

            kept += 1
            total_out += int(decision.new_objects)
            if decision.new_objects < decision.original_objects and len(examples_compacted) < 10:
                images = rec.get("images") or []
                img0 = images[0] if isinstance(images, list) and images else None
                examples_compacted.append(
                    {
                        "line": int(line_no),
                        "image": img0,
                        "objects_before": int(decision.original_objects),
                        "objects_after": int(decision.new_objects),
                        "n_unique": int(decision.n_unique),
                        "top1_ratio": float(decision.top1_ratio),
                    }
                )

            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "config": {
            "input": args.input,
            "output": args.output,
            "seed": int(args.seed),
            "max_total_objects": int(args.max_total_objects),
            "cap_per_class": int(args.cap_per_class),
            "grid": int(args.grid),
            "drop_min_objects": int(args.drop_min_objects),
            "drop_max_unique": int(args.drop_max_unique),
            "drop_min_top1_ratio": float(args.drop_min_top1_ratio),
            "poly_to_bbox_min_class_count": int(args.poly_to_bbox_min_class_count),
            "limit": args.limit,
        },
        "records": {"kept": int(kept), "dropped": int(dropped)},
        "objects": {
            "total_in": int(total_in),
            "total_out": int(total_out),
            "reduction_ratio": float(1.0 - (total_out / total_in)) if total_in else 0.0,
        },
        "poly_to_bbox": int(poly_to_bbox),
        "examples_dropped": examples_dropped,
        "examples_compacted": examples_compacted,
    }

    print("Done")
    print(f"- kept records: {kept}")
    print(f"- dropped records: {dropped}")
    print(f"- objects: {total_in} -> {total_out} (reduction={summary['objects']['reduction_ratio']:.3f})")
    print(f"- poly->bbox conversions: {poly_to_bbox}")

    if args.stats_json:
        with open(args.stats_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Wrote: {args.stats_json}")


if __name__ == "__main__":
    main()

