"""Convert geometry coords in JSONL to norm1000 ints and/or coord tokens.

Offline pipeline support:
- Pixel coords → norm1000 ints (numeric JSONL)
- Pixel coords → norm1000 tokens (coord JSONL)
- Already-normalized ints/tokens → tokens without re-scaling (assume-normalized)

Examples:
  # Default: pixel → tokens (backward compatible)
  python public_data/scripts/convert_to_coord_tokens.py \
      --input public_data/lvis/rescale_32_768_poly_20/val.jsonl \
      --output-tokens public_data/lvis/rescale_32_768_poly_20/val.coord.jsonl

  # Pixel → norm ints and tokens in one pass
  python public_data/scripts/convert_to_coord_tokens.py \
      --input public_data/lvis/rescale_32_768_poly_20/val.jsonl \
      --output-norm public_data/lvis/rescale_32_768_poly_20/val.norm.jsonl \
      --output-tokens public_data/lvis/rescale_32_768_poly_20/val.coord.jsonl

  # Already-normalized ints/tokens → tokens only (no rescale)
  python public_data/scripts/convert_to_coord_tokens.py \
      --input public_data/lvis/rescale_32_768_poly_20/val.norm.jsonl \
      --assume-normalized \
      --output-tokens public_data/lvis/rescale_32_768_poly_20/val.coord.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence

from src.coord_tokens.codec import int_to_token, is_coord_token, token_to_int
from public_data.converters.sorting import canonicalize_poly, sort_objects_tlbr

MAX_VALUE = 999


def _scale_pixel_to_bin(value: float, denom: float) -> int:
    denom_f = float(denom)
    if denom_f <= 0:
        raise AssertionError(f"Invalid denom: {denom_f}")
    v = float(value)
    out = int(round(v / denom_f * 1000.0))
    if not (0 <= out <= 1000):
        raise AssertionError(f"Scaled coord {out} out of [0, 1000] from value {value}")
    return out


def convert_list(values: Sequence, *, width: float, height: float, assume_normalized: bool = False) -> List[str]:
    if width <= 0 or height <= 0:
        raise AssertionError(f"Invalid width/height: {width}, {height}")

    out: List[str] = []
    for idx, v in enumerate(values):
        if is_coord_token(v):
            value = token_to_int(str(v))
        else:
            if assume_normalized:
                value = int(round(float(v)))
            else:
                is_x = idx % 2 == 0
                denom = float(width) if is_x else float(height)
                value = _scale_pixel_to_bin(float(v), denom)

        if not (0 <= value <= MAX_VALUE):
            raise AssertionError(
                f"Normalized coord {value} out of [0, {MAX_VALUE}] from value {v}"
            )
        out.append(int_to_token(value))
    return out


def normalize_list(
    values: Sequence,
    width: float,
    height: float,
    *,
    assume_normalized: bool = False,
) -> List[int]:
    """Return norm1000 ints for a geometry sequence."""
    if width <= 0 or height <= 0:
        raise AssertionError(f"Invalid width/height: {width}, {height}")

    out: List[int] = []
    for idx, v in enumerate(values):
        # Handle existing coord tokens
        if is_coord_token(v):
            value = token_to_int(str(v))
        else:
            if assume_normalized:
                value = int(round(float(v)))
            else:
                is_x = idx % 2 == 0
                denom = float(width) if is_x else float(height)
                value = _scale_pixel_to_bin(float(v), denom)

        if not (0 <= value <= MAX_VALUE):
            raise AssertionError(
                f"Normalized coord {value} out of [0, {MAX_VALUE}] from value {v}"
            )
        out.append(value)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert coords to norm1000 ints and/or coord tokens in JSONL")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL path")
    # Backward-compatible default token output
    parser.add_argument(
        "--output",
        type=Path,
        help="(Deprecated) token output path; use --output-tokens",
    )
    parser.add_argument(
        "--output-tokens",
        type=Path,
        help="Output JSONL path for coord tokens",
    )
    parser.add_argument(
        "--output-norm",
        type=Path,
        help="Output JSONL path for normalized integer coords",
    )
    parser.add_argument(
        "--assume-normalized",
        action="store_true",
        help="Treat input coords as already in [0,999]; skip pixel rescale",
    )
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


def convert_record_to_ints(record: dict, keys: Iterable[str], *, assume_normalized: bool) -> dict:
    """Return a record with geometry normalized to norm1000 ints."""
    objects = record.get("objects") or []
    width = record.get("width")
    height = record.get("height")
    if width is None or height is None:
        raise AssertionError("Record missing width/height for normalization")
    width_f = float(width)
    height_f = float(height)
    for obj in objects:
        for key in keys:
            if key in obj and obj[key] is not None:
                seq = obj[key]
                if not isinstance(seq, Sequence):
                    raise AssertionError(f"{key} must be a list, got {type(seq)}")
                obj[key] = normalize_list(seq, width_f, height_f, assume_normalized=assume_normalized)
    return record


def convert_record_to_tokens(record: dict, keys: Iterable[str]) -> dict:
    """Return a record with geometry converted from ints to coord tokens."""
    objects = record.get("objects") or []
    for obj in objects:
        for key in keys:
            if key in obj and obj[key] is not None:
                seq = obj[key]
                if not isinstance(seq, Sequence):
                    raise AssertionError(f"{key} must be a list, got {type(seq)}")
                tokens = []
                for v in seq:
                    if is_coord_token(v):
                        tokens.append(str(v))
                    else:
                        tokens.append(int_to_token(int(v)))
                obj[key] = tokens
    return record


def _canonicalize_and_sort_objects_in_place(record: dict) -> dict:
    """Canonicalize polygon vertex ordering and sort objects (TLBR).

    This matches the CoordExp/Qwen3-VL prompt spec:
    - `poly`: vertices are canonicalized offline (clockwise around centroid;
      start from top-most then left-most vertex; drop duplicate closing point).
    - objects: sorted top-to-bottom then left-to-right using bbox TL / poly first vertex.

    Doing this in the *norm1000 integer* domain ensures that the ordering matches
    what we actually train/evaluate on (coord tokens / ints).
    """
    objects = record.get("objects") or []
    if not isinstance(objects, list) or not objects:
        return record

    processed = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        if obj.get("poly") is not None:
            poly = obj.get("poly")
            if not isinstance(poly, list):
                # Leave malformed objects untouched; validation happens elsewhere.
                processed.append(obj)
                continue
            try:
                canon = canonicalize_poly(poly)
                canon_ints = [int(round(v)) for v in canon]
            except Exception:
                processed.append(obj)
                continue

            updated = dict(obj)
            updated["poly"] = canon_ints
            if "poly_points" in updated:
                updated["poly_points"] = len(canon_ints) // 2
            processed.append(updated)
        else:
            processed.append(obj)

    record["objects"] = sort_objects_tlbr(processed)
    return record


def main() -> None:
    args = parse_args()
    # Resolve outputs with backward compatibility
    output_tokens = args.output_tokens or args.output
    output_norm = args.output_norm
    if output_tokens is None and output_norm is None:
        raise SystemExit("Specify at least one of --output-tokens or --output-norm (or legacy --output).")

    if output_tokens:
        output_tokens.parent.mkdir(parents=True, exist_ok=True)
    if output_norm:
        output_norm.parent.mkdir(parents=True, exist_ok=True)

    n_lines = 0
    compact_separators = (",", ":") if args.compact else None

    # Open outputs as needed
    fout_tokens = output_tokens.open("w", encoding="utf-8") if output_tokens else None
    fout_norm = output_norm.open("w", encoding="utf-8") if output_norm else None

    try:
        with args.input.open("r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                record_raw = json.loads(line)

                # Normalize to ints (always once)
                record_ints = convert_record_to_ints(
                    json.loads(json.dumps(record_raw)), args.keys, assume_normalized=args.assume_normalized
                )
                record_ints = _canonicalize_and_sort_objects_in_place(record_ints)

                # Write norm ints if requested
                if fout_norm:
                    json.dump(record_ints, fout_norm, ensure_ascii=False, separators=compact_separators)
                    fout_norm.write("\n")

                # Write tokens if requested
                if fout_tokens:
                    record_tokens = convert_record_to_tokens(
                        json.loads(json.dumps(record_ints)), args.keys
                    )
                    json.dump(record_tokens, fout_tokens, ensure_ascii=False, separators=compact_separators)
                    fout_tokens.write("\n")

                n_lines += 1
    finally:
        if fout_tokens:
            fout_tokens.close()
        if fout_norm:
            fout_norm.close()

    outputs = []
    if output_tokens:
        outputs.append(f"tokens: {output_tokens}")
    if output_norm:
        outputs.append(f"norm: {output_norm}")
    print(f"[+] Converted {n_lines} records -> {', '.join(outputs)}")


if __name__ == "__main__":
    main()
