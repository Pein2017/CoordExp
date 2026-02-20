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
import math
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

    # Mirror core coord math: bins are 0..999 and normalized by 999.
    out = int(round(v / denom_f * float(MAX_VALUE)))
    if not (0 <= out <= MAX_VALUE):
        raise AssertionError(
            f"Scaled coord {out} out of [0, {MAX_VALUE}] from value {value}"
        )
    return out


def _clamp_norm1000(v: int) -> int:
    return max(0, min(MAX_VALUE, int(v)))


def _normalize_bbox_2d_to_norm1000(
    values: Sequence,
    *,
    width: float,
    height: float,
    assume_normalized: bool,
) -> List[int]:
    """Normalize a bbox_2d into strict norm1000 ints [x1,y1,x2,y2].

    Two robustness goals:
    1) Prevent bbox collapse under rounding when we later convert to coord tokens.
       We do this by using floor for (x1,y1) and ceil for (x2,y2) in norm space.
    2) As a safety net, enforce strict validity (x2>x1 and y2>y1). If it's still
       invalid after normalization, nudge by 1 within [0,999].
    """
    if len(values) != 4:
        raise AssertionError(f"bbox_2d must have 4 numbers, got len={len(values)}")

    x1, y1, x2, y2 = (float(values[0]), float(values[1]), float(values[2]), float(values[3]))

    if assume_normalized:
        # Values are already in [0,999]-like space; preserve extents with floor/ceil.
        nx1 = int(math.floor(x1))
        ny1 = int(math.floor(y1))
        nx2 = int(math.ceil(x2))
        ny2 = int(math.ceil(y2))
    else:
        if width <= 0 or height <= 0:
            raise AssertionError(f"Invalid width/height: {width}, {height}")
        denom_x = max(1.0, float(width) - 1.0)
        denom_y = max(1.0, float(height) - 1.0)

        sx1 = x1 / denom_x * float(MAX_VALUE)
        sy1 = y1 / denom_y * float(MAX_VALUE)
        sx2 = x2 / denom_x * float(MAX_VALUE)
        sy2 = y2 / denom_y * float(MAX_VALUE)

        nx1 = int(math.floor(sx1))
        ny1 = int(math.floor(sy1))
        nx2 = int(math.ceil(sx2))
        ny2 = int(math.ceil(sy2))

    nx1 = _clamp_norm1000(nx1)
    ny1 = _clamp_norm1000(ny1)
    nx2 = _clamp_norm1000(nx2)
    ny2 = _clamp_norm1000(ny2)

    # Enforce strict validity; if it still collapses, nudge by 1.
    if nx2 <= nx1:
        if nx1 < MAX_VALUE:
            nx2 = nx1 + 1
        else:
            nx1 = max(0, nx1 - 1)
            nx2 = MAX_VALUE
    if ny2 <= ny1:
        if ny1 < MAX_VALUE:
            ny2 = ny1 + 1
        else:
            ny1 = max(0, ny1 - 1)
            ny2 = MAX_VALUE

    nx1 = _clamp_norm1000(nx1)
    ny1 = _clamp_norm1000(ny1)
    nx2 = _clamp_norm1000(nx2)
    ny2 = _clamp_norm1000(ny2)
    return [nx1, ny1, nx2, ny2]


def convert_list(values: Sequence, *, width: float, height: float, assume_normalized: bool = False) -> List[str]:
    if width <= 0 or height <= 0:
        raise AssertionError(f"Invalid width/height: {width}, {height}")

    denom_x = max(1.0, float(width) - 1.0)
    denom_y = max(1.0, float(height) - 1.0)

    out: List[str] = []
    for idx, v in enumerate(values):
        if is_coord_token(v):
            value = token_to_int(str(v))
        else:
            if assume_normalized:
                value = int(round(float(v)))
            else:
                is_x = idx % 2 == 0
                denom = denom_x if is_x else denom_y
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

    denom_x = max(1.0, float(width) - 1.0)
    denom_y = max(1.0, float(height) - 1.0)

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
                denom = denom_x if is_x else denom_y
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


def convert_record_to_ints(
    record: dict,
    keys: Iterable[str],
    *,
    assume_normalized: bool,
    stats: dict | None = None,
) -> dict:
    """Return a record with geometry normalized to norm1000 ints."""
    objects = record.get("objects") or []
    width = record.get("width")
    height = record.get("height")
    if width is None or height is None:
        raise AssertionError("Record missing width/height for normalization")
    width_f = float(width)
    height_f = float(height)
    keys_list = list(keys)

    # (2) Safety-net filtering: drop objects whose bbox_2d is irreparably invalid
    # after normalization. This should be extremely rare after the stricter
    # bbox normalization above, but keeps the pipeline "paper-ready".
    out_objects = []
    objects_seen = 0
    objects_written = 0
    objects_dropped_non_dict = 0
    objects_dropped_invalid_bbox = 0
    for obj in objects:
        objects_seen += 1
        if not isinstance(obj, dict):
            objects_dropped_non_dict += 1
            continue

        obj_out = obj
        for key in keys_list:
            if key not in obj_out or obj_out[key] is None:
                continue
            seq = obj_out[key]
            if not isinstance(seq, Sequence):
                raise AssertionError(f"{key} must be a list, got {type(seq)}")

            if key == "bbox_2d":
                bbox_ints = _normalize_bbox_2d_to_norm1000(
                    seq, width=width_f, height=height_f, assume_normalized=assume_normalized
                )
                obj_out[key] = bbox_ints
            else:
                obj_out[key] = normalize_list(seq, width_f, height_f, assume_normalized=assume_normalized)

        if "bbox_2d" in keys_list and obj_out.get("bbox_2d") is not None:
            b = obj_out["bbox_2d"]
            if isinstance(b, list) and len(b) == 4 and (int(b[2]) > int(b[0])) and (int(b[3]) > int(b[1])):
                out_objects.append(obj_out)
                objects_written += 1
            else:
                # Drop invalid bbox objects to keep downstream validator/train happy.
                objects_dropped_invalid_bbox += 1
                continue
        else:
            out_objects.append(obj_out)
            objects_written += 1

    record["objects"] = out_objects
    if stats is not None:
        stats["objects_seen"] = int(stats.get("objects_seen", 0)) + int(objects_seen)
        stats["objects_written"] = int(stats.get("objects_written", 0)) + int(objects_written)
        stats["objects_dropped_non_dict"] = int(
            stats.get("objects_dropped_non_dict", 0)
        ) + int(objects_dropped_non_dict)
        stats["objects_dropped_invalid_bbox"] = int(
            stats.get("objects_dropped_invalid_bbox", 0)
        ) + int(objects_dropped_invalid_bbox)
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

    Robustness note:
    - After normalization+rounding, a very thin/small polygon can collapse to a
      degenerate (zero-area) polygon in norm1000 space. When that happens, we
      deterministically fall back to `bbox_2d` derived from the polygon extent.
      This keeps the "single geometry key" invariant and avoids invalid GT.
    """

    def poly_area2(poly: List[int]) -> int:
        # Return twice the signed area (shoelace); 0 => degenerate.
        if len(poly) < 6 or len(poly) % 2 != 0:
            return 0
        pts = [(int(poly[i]), int(poly[i + 1])) for i in range(0, len(poly), 2)]
        if len(pts) >= 2 and pts[0] == pts[-1]:
            pts = pts[:-1]
        if len(pts) < 3:
            return 0
        a2 = 0
        for i in range(len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % len(pts)]
            a2 += x1 * y2 - x2 * y1
        return int(a2)

    def bbox_from_poly(poly: List[int]) -> List[int]:
        xs = poly[0::2]
        ys = poly[1::2]
        if not xs or not ys:
            return [0, 0, 1, 1]
        x1 = int(min(xs))
        y1 = int(min(ys))
        x2 = int(max(xs))
        y2 = int(max(ys))
        # Enforce strict bbox validity for downstream consumers.
        if x2 <= x1:
            if x1 < MAX_VALUE:
                x2 = x1 + 1
            else:
                x1 = max(0, x1 - 1)
        if y2 <= y1:
            if y1 < MAX_VALUE:
                y2 = y1 + 1
            else:
                y1 = max(0, y1 - 1)
        # Clamp to norm1000 domain.
        x1 = max(0, min(MAX_VALUE, x1))
        y1 = max(0, min(MAX_VALUE, y1))
        x2 = max(0, min(MAX_VALUE, x2))
        y2 = max(0, min(MAX_VALUE, y2))
        return [x1, y1, x2, y2]

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
            # If the canonicalized polygon collapses under rounding, fallback to bbox.
            if poly_area2(canon_ints) == 0:
                updated.pop("poly", None)
                updated.pop("poly_points", None)
                updated["bbox_2d"] = bbox_from_poly(canon_ints)
            else:
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
