#!/usr/bin/env python3
"""Merge multiple CoordExp JSONL files into one, rewriting image paths.

Why this exists:
- CoordExp loads relative image paths relative to the JSONL's parent directory.
- If you simply `cat a/train.jsonl b/train.jsonl > merged.jsonl`, all relative
  image paths from the second dataset will break.

This script rewrites each record's `images` entries so they remain valid when
resolved against the *output* JSONL directory.

Supports two deterministic merge strategies:
- concat: write all of input_0, then input_1, ...
- round_robin: interleave 1 line at a time across inputs
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_num}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object at {path}:{line_num}, got {type(obj).__name__}")
            yield obj


def _rewrite_images(record: Dict[str, Any], *, input_dir: Path, output_dir: Path) -> Dict[str, Any]:
    images = record.get("images")
    if images is None:
        return record
    if not isinstance(images, list):
        raise ValueError("record['images'] must be a list when present")

    rewritten: List[str] = []
    for img in images:
        img_str = str(img)
        if os.path.isabs(img_str):
            abs_path = Path(img_str)
        else:
            abs_path = (input_dir / img_str).resolve()
        rel = os.path.relpath(str(abs_path), start=str(output_dir))
        rewritten.append(rel)

    out = dict(record)
    out["images"] = rewritten
    return out


def _round_robin_iters(iters: List[Iterator[Dict[str, Any]]]) -> Iterator[Dict[str, Any]]:
    active = list(iters)
    while active:
        next_active: List[Iterator[Dict[str, Any]]] = []
        for it in active:
            try:
                yield next(it)
                next_active.append(it)
            except StopIteration:
                continue
        active = next_active


def _iter_rewritten(path: Path, *, output_dir: Path) -> Iterator[Dict[str, Any]]:
    """Iterate a JSONL while rewriting its image paths for a target output dir.

    This helper avoids Python's late-binding gotcha with generator expressions
    inside loops.
    """
    input_dir = path.parent.resolve()
    for rec in _iter_jsonl(path):
        yield _rewrite_images(rec, input_dir=input_dir, output_dir=output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        required=True,
        help="Input JSONL paths to merge (order matters)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL path",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="concat",
        choices=["concat", "round_robin"],
        help="Merge strategy (default: concat)",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="Optional cap on number of records written (for smoke tests)",
    )
    args = parser.parse_args()

    out_path: Path = args.output
    out_dir = out_path.resolve().parent
    out_path.parent.mkdir(parents=True, exist_ok=True)

    inputs: List[Path] = [p.resolve() for p in args.inputs]
    for p in inputs:
        if not p.is_file():
            raise FileNotFoundError(p)

    # Build iterators that rewrite image paths on the fly.
    rewritten_iters: List[Iterator[Dict[str, Any]]] = []
    for p in inputs:
        rewritten_iters.append(_iter_rewritten(p, output_dir=out_dir))

    if args.strategy == "concat":
        merged_iter: Iterable[Dict[str, Any]] = (rec for it in rewritten_iters for rec in it)
    else:
        merged_iter = _round_robin_iters(rewritten_iters)

    written = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for rec in merged_iter:
            if args.max_lines is not None and written >= int(args.max_lines):
                break
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    print(f"[+] Wrote {written} records -> {out_path}")


if __name__ == "__main__":
    main()
