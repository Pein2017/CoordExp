#!/usr/bin/env python3
"""Merge multiple CoordExp JSONL files into one, rewriting image paths.

This script exists because CoordExp resolves relative image paths relative to
the JSONL's parent directory. If you simply do:

  cat a/train.jsonl b/train.jsonl > merged.jsonl

then all relative image paths from the second dataset will break.

Modes
-----
1) Direct merge (recommended for simple offline fusion):
   - Provide `--inputs ... --output ...`
   - Supports two deterministic strategies:
     - concat: write all of input_0, then input_1, ...
     - round_robin: interleave 1 line at a time across inputs

2) Fusion-config materialization (optional):
   - Provide `--fusion-config ... --output ...`
   - Delegates to `src.datasets.fusion` to build a fused JSONL.
   - This is useful when you already have a fusion config and want to snapshot
     the merged dataset for external tooling.
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
                raise ValueError(
                    f"Expected object at {path}:{line_num}, got {type(obj).__name__}"
                )
            yield obj


def _rewrite_images(
    record: Dict[str, Any], *, input_dir: Path, output_dir: Path
) -> Dict[str, Any]:
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
    """Iterate a JSONL while rewriting its image paths for a target output dir."""
    input_dir = path.parent.resolve()
    for rec in _iter_jsonl(path):
        yield _rewrite_images(rec, input_dir=input_dir, output_dir=output_dir)


def _merge_inputs(
    *,
    inputs: List[Path],
    output: Path,
    strategy: str,
    max_lines: Optional[int],
) -> None:
    out_path = output
    out_dir = out_path.resolve().parent
    out_path.parent.mkdir(parents=True, exist_ok=True)

    resolved_inputs = [p.resolve() for p in inputs]
    for p in resolved_inputs:
        if not p.is_file():
            raise FileNotFoundError(p)

    rewritten_iters: List[Iterator[Dict[str, Any]]] = []
    for p in resolved_inputs:
        rewritten_iters.append(_iter_rewritten(p, output_dir=out_dir))

    if strategy == "concat":
        merged_iter: Iterable[Dict[str, Any]] = (
            rec for it in rewritten_iters for rec in it
        )
    else:
        merged_iter = _round_robin_iters(rewritten_iters)

    written = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for rec in merged_iter:
            if max_lines is not None and written >= int(max_lines):
                break
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    print(f"[+] Wrote {written} records -> {out_path}")


def _materialize_fusion_config(
    *, fusion_config: str, output: Path, seed: int, shuffle: bool
) -> None:
    from src.datasets.fusion import FusionConfig, build_fused_jsonl

    cfg = FusionConfig.from_file(fusion_config)
    out = build_fused_jsonl(cfg, str(output), seed=int(seed), shuffle=bool(shuffle))
    print(str(out))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mx = parser.add_mutually_exclusive_group(required=True)

    # Direct JSONL merge mode
    mx.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        help="Input JSONL paths to merge (order matters)",
    )

    # Fusion-config materialization mode
    mx.add_argument(
        "--fusion-config",
        type=str,
        help="Path to fusion YAML/JSON (optional; uses src.datasets.fusion).",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL path",
    )

    # Direct merge options
    parser.add_argument(
        "--strategy",
        type=str,
        default="concat",
        choices=["concat", "round_robin"],
        help="Merge strategy for --inputs (default: concat)",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="Optional cap on number of records written (for smoke tests)",
    )

    # Fusion-config options
    parser.add_argument("--seed", type=int, default=2025, help="Shuffle/upsample seed.")
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable final shuffle of the merged records (fusion-config mode).",
    )

    args = parser.parse_args()
    if args.inputs is not None:
        _merge_inputs(
            inputs=list(args.inputs),
            output=args.output,
            strategy=str(args.strategy),
            max_lines=args.max_lines,
        )
        return

    _materialize_fusion_config(
        fusion_config=str(args.fusion_config),
        output=args.output,
        seed=int(args.seed),
        shuffle=not bool(args.no_shuffle),
    )


if __name__ == "__main__":
    main()

