#!/usr/bin/env python3
"""Collect high-confidence junk `desc` labels from a CoordExp JSONL.

This is VG-oriented (uses `public_data/vg/junk_descs.py`) and is meant to be
data-driven: scan your produced VG JSONL(s) and report how often these junk
labels appear, so you can decide whether to drop them.

Example:
  conda run -n ms python public_data/vg/collect_junk_descs.py \\
    --jsonl public_data/vg/raw/train.jsonl \\
    --top-k 50
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List

from public_data.vg.junk_descs import HIGH_CONF_JUNK_DESCS, normalize_desc_for_match


@dataclass
class JunkStats:
    total_records: int = 0
    total_objects: int = 0
    matched_objects: int = 0


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--jsonl",
        type=Path,
        action="append",
        required=True,
        help="Input JSONL to scan (repeatable)",
    )
    ap.add_argument("--top-k", type=int, default=100, help="Show top-K junk labels by count")
    ap.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (text table or JSON)",
    )
    ap.add_argument(
        "--max-examples",
        type=int,
        default=5,
        help="Store up to N example raw desc strings per normalized junk key",
    )
    args = ap.parse_args()

    counts: Counter[str] = Counter()
    examples: DefaultDict[str, List[str]] = defaultdict(list)
    stats = JunkStats()

    for p in args.jsonl:
        if not p.exists():
            raise FileNotFoundError(str(p))
        for rec in _iter_jsonl(p):
            stats.total_records += 1
            objs = rec.get("objects") or []
            if not isinstance(objs, list):
                continue
            for obj in objs:
                if not isinstance(obj, dict):
                    continue
                desc = obj.get("desc")
                if not isinstance(desc, str):
                    continue
                stats.total_objects += 1
                key = normalize_desc_for_match(desc)
                if key in HIGH_CONF_JUNK_DESCS:
                    stats.matched_objects += 1
                    counts[key] += 1
                    if len(examples[key]) < int(args.max_examples):
                        raw = desc.strip()
                        if raw and raw not in examples[key]:
                            examples[key].append(raw)

    items = counts.most_common(int(args.top_k))
    if args.format == "json":
        out: Dict[str, object] = {
            "stats": {
                "total_records": stats.total_records,
                "total_objects": stats.total_objects,
                "matched_objects": stats.matched_objects,
                "matched_ratio": (stats.matched_objects / stats.total_objects) if stats.total_objects else 0.0,
            },
            "junk_vocab": sorted(HIGH_CONF_JUNK_DESCS),
            "counts": [{"desc": k, "count": int(v), "examples": examples.get(k, [])} for k, v in items],
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    print("=== VG high-confidence junk desc report ===")
    print(f"jsonl_files: {[str(p) for p in args.jsonl]}")
    print(f"total_records: {stats.total_records}")
    print(f"total_objects: {stats.total_objects}")
    print(f"matched_objects: {stats.matched_objects}")
    ratio = (stats.matched_objects / stats.total_objects) if stats.total_objects else 0.0
    print(f"matched_ratio: {ratio:.6f}")
    print()
    print(f"Top {len(items)} junk desc labels (normalized):")
    for k, v in items:
        ex = examples.get(k, [])
        ex_str = f" examples={ex!r}" if ex else ""
        print(f"  {v:>10}  {k}{ex_str}")


if __name__ == "__main__":
    main()

