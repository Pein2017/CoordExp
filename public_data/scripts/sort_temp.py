#!/usr/bin/env python3
"""Resort objects inside existing JSONL files (no image processing).

Use case: you already have processed LVIS JSONLs (pixel/raw, norm1000 ints, coord tokens)
and want to enforce the latest ordering rule without re-running smart-resize / image I/O.

What this script does per record:
1) Canonicalize `poly` vertices (drop closing duplicate; clockwise around centroid; start at top-most then left-most)
2) Sort `objects` top-to-bottom then left-to-right using bbox TL / poly first vertex (after canonicalization)

It preserves:
- record order (one output line per input line)
- object identity (same desc + same vertex set; only ordering changes)

Examples:
  # In-place overwrite all LVIS jsonls in a folder
  PYTHONPATH=. /root/miniconda3/envs/ms/bin/python public_data/scripts/sort_temp.py \\
    --inplace public_data/lvis/rescale_32_768_poly_20/*.jsonl

  # Safer: write to a new sibling file with suffix
  PYTHONPATH=. /root/miniconda3/envs/ms/bin/python public_data/scripts/sort_temp.py \\
    --suffix .sorted public_data/lvis/rescale_32_768_poly_20/val.coord.jsonl

  # Quick smoke (process only first 200 lines, no writes)
  PYTHONPATH=. /root/miniconda3/envs/ms/bin/python public_data/scripts/sort_temp.py \\
    --dry-run --limit 200 public_data/lvis/rescale_32_768_poly_20/val.raw.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

from public_data.converters.sorting import canonicalize_poly
from src.coord_tokens.codec import int_to_token, is_coord_token, token_to_int
from src.datasets.utils import sort_objects_by_topleft


def _flatten_points(value: Any) -> Optional[List[Any]]:
    """Flatten either `[x1,y1,...]` or `[[x1,y1], [x2,y2], ...]` into a flat list."""
    if value is None:
        return None
    if not isinstance(value, list):
        return None
    if not value:
        return []
    if isinstance(value[0], list):
        flat: List[Any] = []
        for p in value:
            if not isinstance(p, list) or len(p) < 2:
                return None
            flat.extend([p[0], p[1]])
        return flat
    return list(value)


def _detect_domain(path: Path) -> str:
    """Best-effort domain based on filename conventions used in this repo."""
    name = path.name.lower()
    if ".raw." in name or name.endswith(".raw.jsonl"):
        return "pixel"
    if ".coord." in name or name.endswith(".coord.jsonl"):
        return "coord_tokens"
    return "norm1000"


@dataclass
class Stats:
    records: int = 0
    objects: int = 0
    polys_seen: int = 0
    polys_canon: int = 0
    errors: int = 0


def _canonicalize_poly_any(poly_value: Any, *, domain: str, stats: Stats) -> Any:
    """Canonicalize polygon points for pixel/norm1000/tokens domains.

    Returns a new poly value in the *same representation* as the input domain:
    - pixel/norm1000: list[int]
    - coord_tokens: list[str] (tokens)
    """
    flat = _flatten_points(poly_value)
    if flat is None:
        return poly_value
    if len(flat) < 6 or len(flat) % 2 != 0:
        return poly_value

    stats.polys_seen += 1

    try:
        if domain == "coord_tokens":
            ints: List[int] = []
            for v in flat:
                if is_coord_token(v):
                    ints.append(token_to_int(str(v)))
                else:
                    ints.append(int(round(float(v))))
            canon = canonicalize_poly(ints)
            canon_ints = [int(round(v)) for v in canon]
            stats.polys_canon += 1
            return [int_to_token(v) for v in canon_ints]

        # pixel / norm1000: treat as numeric points
        nums = [float(v) for v in flat]
        canon = canonicalize_poly(nums)
        canon_ints = [int(round(v)) for v in canon]
        stats.polys_canon += 1
        return canon_ints
    except Exception:
        # Keep original on failure; count error for visibility.
        stats.errors += 1
        return poly_value


def process_record(
    record: MutableMapping[str, Any],
    *,
    domain: str,
    no_poly_canon: bool,
    stats: Stats,
) -> Dict[str, Any]:
    """Return a processed copy of one JSONL record."""
    out = dict(record)
    objects = out.get("objects") or []
    if not isinstance(objects, list) or not objects:
        return out

    new_objects: List[Dict[str, Any]] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        stats.objects += 1

        updated = dict(obj)
        if not no_poly_canon and updated.get("poly") is not None:
            updated["poly"] = _canonicalize_poly_any(
                updated.get("poly"), domain=domain, stats=stats
            )
            if "poly_points" in updated and isinstance(updated.get("poly"), list):
                updated["poly_points"] = len(updated["poly"]) // 2
        new_objects.append(updated)

    out["objects"] = sort_objects_by_topleft(new_objects)
    return out


def _iter_input_paths(paths: Sequence[str]) -> List[Path]:
    expanded: List[Path] = []
    for p in paths:
        expanded.append(Path(p))
    return expanded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resort objects inside JSONL files")
    parser.add_argument("inputs", nargs="+", help="Input JSONL paths (shell globs ok)")
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite input files in place (uses atomic temp + replace)",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="If not inplace, write to <input><suffix>.jsonl (e.g. .sorted). Default: ''",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional output directory (ignored when --inplace is set)",
    )
    parser.add_argument(
        "--domain",
        choices=["auto", "pixel", "norm1000", "coord_tokens"],
        default="auto",
        help="Coordinate domain. 'auto' uses filename heuristics. Default: auto",
    )
    parser.add_argument(
        "--no-poly-canon",
        action="store_true",
        help="Only sort objects; do not canonicalize polygon vertex order",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only first N lines per file (0 = no limit)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read/process but do not write outputs (prints stats only)",
    )
    parser.add_argument(
        "--backup-suffix",
        type=str,
        default="",
        help="When --inplace, also copy the original file to <file><backup_suffix> first (optional).",
    )
    return parser.parse_args()


def _output_path_for(input_path: Path, *, args: argparse.Namespace) -> Path:
    if args.inplace:
        return input_path
    if args.output_dir is not None:
        return (args.output_dir / input_path.name).resolve()
    if args.suffix:
        # Insert suffix before the .jsonl extension when present.
        if input_path.name.endswith(".jsonl"):
            return input_path.with_name(input_path.name[: -len(".jsonl")] + args.suffix + ".jsonl")
        return input_path.with_name(input_path.name + args.suffix)
    raise SystemExit("Specify --inplace or --suffix or --output-dir")


def process_file(input_path: Path, *, args: argparse.Namespace) -> Stats:
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))

    domain = _detect_domain(input_path) if args.domain == "auto" else args.domain

    stats = Stats()

    # In-place writes go to a temp file and then replace atomically.
    if args.dry_run:
        fout = None
        tmp_path = None
        out_path = None
    else:
        out_path = _output_path_for(input_path, args=args)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if args.inplace:
            tmp_path = out_path.with_name(out_path.name + f".tmp.{os.getpid()}")
            fout = tmp_path.open("w", encoding="utf-8")
        else:
            tmp_path = None
            fout = out_path.open("w", encoding="utf-8")

    try:
        with input_path.open("r", encoding="utf-8") as fin:
            for line_idx, line in enumerate(fin, 1):
                if args.limit and line_idx > args.limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    stats.errors += 1
                    continue

                stats.records += 1
                updated = process_record(
                    record,
                    domain=domain,
                    no_poly_canon=bool(args.no_poly_canon),
                    stats=stats,
                )

                if fout is not None:
                    fout.write(json.dumps(updated, ensure_ascii=False) + "\n")
    finally:
        if fout is not None:
            fout.close()

    if args.dry_run:
        return stats

    if args.inplace:
        if tmp_path is None:
            raise RuntimeError("inplace write requested but no temp file was created")
        # Optional backup (can be large; use only if you have disk headroom).
        if args.backup_suffix:
            backup_path = input_path.with_name(input_path.name + args.backup_suffix)
            shutil.copy2(input_path, backup_path)
        os.replace(tmp_path, input_path)

    return stats


def main() -> None:
    args = parse_args()
    paths = _iter_input_paths(args.inputs)
    if args.output_dir is not None and args.inplace:
        raise SystemExit("--output-dir cannot be used with --inplace")

    all_stats = Stats()
    for p in paths:
        if p.suffix != ".jsonl":
            print(f"[skip] {p} (not .jsonl)", file=sys.stderr)
            continue
        stats = process_file(p, args=args)
        all_stats.records += stats.records
        all_stats.objects += stats.objects
        all_stats.polys_seen += stats.polys_seen
        all_stats.polys_canon += stats.polys_canon
        all_stats.errors += stats.errors
        print(
            f"[ok] {p} records={stats.records} objs={stats.objects} "
            f"polys={stats.polys_seen} canon={stats.polys_canon} errors={stats.errors}"
        )

    print(
        f"[done] files={len(paths)} records={all_stats.records} objs={all_stats.objects} "
        f"polys={all_stats.polys_seen} canon={all_stats.polys_canon} errors={all_stats.errors}"
    )


if __name__ == "__main__":
    main()

