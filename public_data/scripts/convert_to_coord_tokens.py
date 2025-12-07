"""Convert geometry coords in JSONL to norm1000 coord tokens.

Usage:
  python public_data/scripts/convert_to_coord_tokens.py \
      --input public_data/lvis/rescale_32_768_poly_max_20/val.jsonl \
      --output public_data/lvis/rescale_32_768_poly_max_20/val.coord.jsonl \
      [--keys bbox_2d poly line] [--compact]

Notes:
- Input coords can be pixel-space; script normalizes to norm1000 using per-record width/height.
- Uses pixel bounds [0, width-1] / [0, height-1] so 0 → 0 and the max in-bounds pixel → 999 (no 1000 values).
- Leaves non-geometry fields unchanged.
- Emits geometry lists as JSON arrays of coord token strings, e.g.
  ["<|coord_1|>", "<|coord_2|>", "<|coord_3|>", "<|coord_4|>"]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence

from src.coord_tokens.codec import int_to_token, is_coord_token, token_to_int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert numeric coords to coord tokens in JSONL")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL path")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    parser.add_argument(
        "--keys",
        nargs="*",
        default=["bbox_2d", "poly", "line"],
        help="Geometry keys to convert (default: bbox_2d poly line)",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Emit compact JSON (no spaces after commas/colons)",
    )
    return parser.parse_args()


def convert_list(values: Sequence, width: float, height: float) -> List[str]:
    if width <= 0 or height <= 0:
        raise AssertionError(f"Invalid width/height: {width}, {height}")

    out: List[str] = []
    max_value = 999
    for idx, v in enumerate(values):
        if is_coord_token(v):
            value = token_to_int(str(v))
        else:
            is_x = idx % 2 == 0
            denom = float(width) if is_x else float(height)
            # Valid pixel domain is [0, denom-1]; clamp to that before scaling.
            denom_span = max(1e-6, denom - 1.0)
            v_float = max(0.0, min(float(v), denom - 1.0))
            value = int(round(v_float / denom_span * max_value))
        if not (0 <= value <= max_value):
            raise AssertionError(
                f"Normalized coord {value} out of [0, {max_value}] from value {v}"
            )
        out.append(int_to_token(value))
    return out


def convert_record(record: dict, keys: Iterable[str]) -> dict:
    objects = record.get("objects") or []
    width = record.get("width")
    height = record.get("height")
    if width is None or height is None:
        raise AssertionError("Record missing width/height for normalization")
    width = float(width)
    height = float(height)
    for obj in objects:
        for key in keys:
            if key in obj and obj[key] is not None:
                seq = obj[key]
                if not isinstance(seq, Sequence):
                    raise AssertionError(f"{key} must be a list, got {type(seq)}")
                obj[key] = convert_list(seq, width, height)
    return record


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    n_lines = 0
    with args.input.open("r", encoding="utf-8") as fin, args.output.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record = convert_record(record, args.keys)
            json.dump(record, fout, ensure_ascii=False, separators=(',', ':') if args.compact else None)
            fout.write("\n")
            n_lines += 1

    print(f"[+] Converted {n_lines} records -> {args.output}")


if __name__ == "__main__":
    main()
