#!/usr/bin/env python3
"""Select a deterministic image-disjoint pool for coordinate-boundary mining.

The selector is deliberately label-only.  It does not inspect model outputs or
choose a treatment arm.  It balances coarse object-count bands so the later
StateBank is not composed only of either sparse or crowded images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import copy
import os
from collections import defaultdict
from pathlib import Path


SCHEMA_VERSION = "coordinate_boundary_candidate_pool.v1"
BLIND_IMAGE_IDS = {
    1584, 2685, 4134, 5001, 6040, 7511,
    10707, 13348, 13923, 14038, 14439, 16228,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _band(count: int) -> str:
    if count <= 3:
        return "sparse_1_to_3"
    if count <= 7:
        return "medium_4_to_7"
    if count <= 15:
        return "dense_8_to_15"
    return "very_dense_16_plus"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--count", type=int, default=768)
    parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()
    if args.count <= 0 or args.count % 4:
        raise SystemExit("--count must be positive and divisible by four")
    source = args.input.expanduser().resolve(strict=True)
    rows: list[dict] = []
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            image_id = int(row["image_id"])
            objects = row.get("objects") or []
            if image_id in BLIND_IMAGE_IDS or not objects:
                continue
            rebased = copy.deepcopy(row)
            images = rebased.get("images") or []
            if len(images) != 1:
                raise SystemExit(f"image {image_id} must contain exactly one image path")
            declared = Path(str(images[0]))
            resolved_image = (
                declared if declared.is_absolute() else source.parent / declared
            ).resolve(strict=True)
            rebased["images"] = [
                os.path.relpath(resolved_image, args.output.expanduser().resolve().parent)
            ]
            rows.append(rebased)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[_band(len(row["objects"]))].append(row)
    rng = random.Random(args.seed)
    per_band = args.count // 4
    selected: list[dict] = []
    counts: dict[str, int] = {}
    for name in (
        "sparse_1_to_3", "medium_4_to_7", "dense_8_to_15", "very_dense_16_plus"
    ):
        candidates = groups[name]
        rng.shuffle(candidates)
        if len(candidates) < per_band:
            raise SystemExit(f"insufficient candidates in {name}: {len(candidates)} < {per_band}")
        chosen = candidates[:per_band]
        selected.extend(chosen)
        counts[name] = len(chosen)
    selected.sort(key=lambda row: int(row["image_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "source": str(source),
        "source_sha256": _sha256(source),
        "output": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
        "seed": args.seed,
        "selected_image_count": len(selected),
        "object_count_bands": counts,
        "blind_image_intersection": sorted(
            BLIND_IMAGE_IDS & {int(row["image_id"]) for row in selected}
        ),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
