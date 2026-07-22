#!/usr/bin/env python3
"""Split a frozen candidate pool before model rollouts.

This helper makes a deterministic, label-only train-candidate, development,
and held-out split.  It never examines a model artifact and writes selected
JavaScript Object Notation Lines (JSONL) rows byte-for-byte from the frozen
input pool.  The row order inside each output remains the original pool order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "label_only_candidate_pool_split.v1"
BAND_NAMES = (
    "sparse_1_to_3",
    "medium_4_to_7",
    "dense_8_to_15",
    "very_dense_16_plus",
)
OUTPUT_FILENAMES = {
    "train_candidate": "train-candidate.jsonl",
    "development": "development.jsonl",
    "heldout": "heldout.jsonl",
}
RECEIPT_FILENAME = "split-receipt.json"


@dataclass(frozen=True)
class PoolRow:
    """One validated source row retained exactly as it appeared in the pool."""

    image_id: str
    band: str
    raw_line: bytes
    has_relative_image_path: bool


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _canonical_image_id(value: Any, *, context: str) -> str:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{context} has a missing or unsupported image_id")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError(f"{context} has a missing or unsupported image_id")


def _band(object_count: int) -> str:
    if 1 <= object_count <= 3:
        return "sparse_1_to_3"
    if object_count <= 7:
        return "medium_4_to_7"
    if object_count <= 15:
        return "dense_8_to_15"
    return "very_dense_16_plus"


def _ordered_image_ids_sha256(image_ids: Sequence[str]) -> str:
    return _sha256_bytes("".join(f"{image_id}\n" for image_id in image_ids).encode())


def _read_pool(path: Path) -> list[PoolRow]:
    raw_lines = path.read_bytes().splitlines(keepends=True)
    if not raw_lines:
        raise ValueError("candidate pool must contain at least one JSONL row")

    rows: list[PoolRow] = []
    seen_image_ids: set[str] = set()
    for row_index, raw_line in enumerate(raw_lines, start=1):
        if not raw_line.strip():
            raise ValueError(f"candidate pool row {row_index} is blank")
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"candidate pool row {row_index} is not valid JSON") from exc
        if not isinstance(payload, Mapping):
            raise ValueError(f"candidate pool row {row_index} is not an object")
        image_id = _canonical_image_id(
            payload.get("image_id"), context=f"candidate pool row {row_index}"
        )
        if image_id in seen_image_ids:
            raise ValueError(f"candidate pool contains duplicate image_id: {image_id}")
        seen_image_ids.add(image_id)
        objects = payload.get("objects")
        if not isinstance(objects, list) or not objects:
            raise ValueError(f"candidate pool row {row_index} has empty objects")
        images = payload.get("images")
        has_relative_image_path = isinstance(images, list) and any(
            isinstance(image_path, str) and not Path(image_path).is_absolute()
            for image_path in images
        )
        rows.append(
            PoolRow(
                image_id=image_id,
                band=_band(len(objects)),
                raw_line=raw_line,
                has_relative_image_path=has_relative_image_path,
            )
        )
    return rows


def _per_band_allocation(count: int) -> dict[str, int]:
    """Allocate a requested split size deterministically across four bands."""

    quotient, remainder = divmod(count, len(BAND_NAMES))
    return {
        band: quotient + int(index < remainder)
        for index, band in enumerate(BAND_NAMES)
    }


def _require_nonnegative_counts(counts: Mapping[str, int]) -> None:
    for name, count in counts.items():
        if count < 0:
            raise ValueError(f"{name} count must be non-negative")


def _prepare_immutable_outputs(output_dir: Path) -> dict[str, Path]:
    output_paths = {
        name: output_dir / filename for name, filename in OUTPUT_FILENAMES.items()
    }
    output_paths["receipt"] = output_dir / RECEIPT_FILENAME
    for path in output_paths.values():
        if path.exists():
            raise FileExistsError(f"immutable output already exists: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_paths


def split_label_only_candidate_pool(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    train_count: int = 2048,
    development_count: int = 256,
    heldout_count: int = 128,
    seed: int = 19,
) -> Mapping[str, Any]:
    """Materialize deterministic, image-disjoint label-only split files."""

    counts = {
        "train_candidate": train_count,
        "development": development_count,
        "heldout": heldout_count,
    }
    _require_nonnegative_counts(counts)
    source = Path(input_path).expanduser().resolve(strict=True)
    destination = Path(output_dir).expanduser().resolve()
    rows = _read_pool(source)
    if destination != source.parent and any(
        row.has_relative_image_path for row in rows
    ):
        raise ValueError(
            "byte-preserving split outputs with relative image paths must be "
            "written beside the candidate pool"
        )
    requested_total = sum(counts.values())
    if requested_total != len(rows):
        raise ValueError(
            "requested split count total "
            f"{requested_total} does not match candidate pool count {len(rows)}"
        )
    output_paths = _prepare_immutable_outputs(destination)

    rows_by_band = {band: [] for band in BAND_NAMES}
    for row in rows:
        rows_by_band[row.band].append(row)

    requested_by_split = {
        name: _per_band_allocation(count) for name, count in counts.items()
    }
    assignments: dict[str, str] = {}
    random_generator = random.Random(seed)
    for band in BAND_NAMES:
        required = sum(requested_by_split[name][band] for name in counts)
        candidates = list(rows_by_band[band])
        if len(candidates) < required:
            raise ValueError(
                f"insufficient candidates in {band}: {len(candidates)} < {required}"
            )
        random_generator.shuffle(candidates)
        offset = 0
        for split_name in counts:
            next_offset = offset + requested_by_split[split_name][band]
            for row in candidates[offset:next_offset]:
                assignments[row.image_id] = split_name
            offset = next_offset

    selected_rows = {
        name: [row for row in rows if assignments.get(row.image_id) == name]
        for name in counts
    }
    for split_name, requested_count in counts.items():
        if len(selected_rows[split_name]) != requested_count:
            raise AssertionError(
                f"internal split count mismatch for {split_name}: "
                f"{len(selected_rows[split_name])} != {requested_count}"
            )

    output_documents: dict[str, dict[str, Any]] = {}
    for split_name, split_rows in selected_rows.items():
        output_bytes = b"".join(row.raw_line for row in split_rows)
        output_path = output_paths[split_name]
        output_path.open("xb").write(output_bytes)
        image_ids = [row.image_id for row in split_rows]
        output_documents[split_name] = {
            "path": str(output_path),
            "sha256": _sha256_bytes(output_bytes),
            "count": len(split_rows),
            "per_band_counts": {
                band: sum(row.band == band for row in split_rows) for band in BAND_NAMES
            },
            "ordered_image_ids_sha256": _ordered_image_ids_sha256(image_ids),
        }

    output_image_id_sets = {
        split_name: {row.image_id for row in split_rows}
        for split_name, split_rows in selected_rows.items()
    }
    source_image_ids = [row.image_id for row in rows]
    selected_image_ids = [row.image_id for row in rows if row.image_id in assignments]
    union_image_ids = set().union(*output_image_id_sets.values())
    expected_selected_set = set(selected_image_ids)
    pairwise_overlap_counts = {
        "train_candidate_development": len(
            output_image_id_sets["train_candidate"]
            & output_image_id_sets["development"]
        ),
        "train_candidate_heldout": len(
            output_image_id_sets["train_candidate"] & output_image_id_sets["heldout"]
        ),
        "development_heldout": len(
            output_image_id_sets["development"] & output_image_id_sets["heldout"]
        ),
    }
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "input": {
            "path": str(source),
            "sha256": _sha256_file(source),
            "count": len(rows),
            "ordered_image_ids_sha256": _ordered_image_ids_sha256(
                source_image_ids
            ),
        },
        "seed": seed,
        "requested_counts": counts,
        "requested_per_band_counts": requested_by_split,
        "outputs": output_documents,
        "proof": {
            "pairwise_overlap_counts": pairwise_overlap_counts,
            "zero_overlap": all(count == 0 for count in pairwise_overlap_counts.values()),
            "selected_image_count": len(selected_image_ids),
            "union_image_count": len(union_image_ids),
            "selected_ordered_image_ids_sha256": _ordered_image_ids_sha256(
                selected_image_ids
            ),
            "union_matches_selected": union_image_ids == expected_selected_set,
            "union_matches_input": union_image_ids == set(source_image_ids),
            "unselected_input_count": len(rows) - len(selected_image_ids),
        },
    }
    output_paths["receipt"].open("xb").write(_canonical_json_bytes(document))
    return document


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-count", type=int, default=2048)
    parser.add_argument("--development-count", type=int, default=256)
    parser.add_argument("--heldout-count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=19)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = split_label_only_candidate_pool(
        input_path=args.input,
        output_dir=args.output_dir,
        train_count=args.train_count,
        development_count=args.development_count,
        heldout_count=args.heldout_count,
        seed=args.seed,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
