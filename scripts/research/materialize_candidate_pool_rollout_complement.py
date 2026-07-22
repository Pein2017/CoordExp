#!/usr/bin/env python3
"""Materialize an immutable candidate-pool complement for rollout research.

The output keeps every selected candidate-pool JavaScript Object Notation Lines
row byte-for-byte and in the same order.  It is intentionally a small data
selection seam: it does not inspect annotations, execute a model, or alter a
row's image, prompt, or target serialization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROLLOUT_SCHEMA_VERSION = "current_seeded_sampled_rollouts.v1"
SCHEMA_VERSION = "candidate_pool_rollout_complement.v1"


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
    """Return an exact logical image identifier without accepting booleans/floats."""

    if isinstance(value, bool) or value is None:
        raise ValueError(f"{context} has a missing or unsupported image_id")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError(f"{context} has a missing or unsupported image_id")


def _ordered_image_ids_sha256(image_ids: Sequence[str]) -> str:
    return _sha256_bytes("".join(f"{image_id}\n" for image_id in image_ids).encode())


def _read_candidate_pool(path: Path) -> list[tuple[str, bytes, bool]]:
    rows: list[tuple[str, bytes, bool]] = []
    seen: set[str] = set()
    raw_rows = path.read_bytes().splitlines(keepends=True)
    if not raw_rows:
        raise ValueError("candidate pool must contain at least one JavaScript Object Notation Lines row")
    for row_index, raw_line in enumerate(raw_rows, start=1):
        if not raw_line.strip():
            raise ValueError(f"candidate pool row {row_index} is blank")
        if row_index != len(raw_rows) and not raw_line.endswith((b"\n", b"\r")):
            raise ValueError("candidate pool has an unterminated non-final row")
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"candidate pool row {row_index} is not valid JSON") from exc
        if not isinstance(payload, Mapping):
            raise ValueError(f"candidate pool row {row_index} is not an object")
        image_id = _canonical_image_id(
            payload.get("image_id"), context=f"candidate pool row {row_index}"
        )
        if image_id in seen:
            raise ValueError(f"candidate pool contains duplicate image_id: {image_id}")
        seen.add(image_id)
        images = payload.get("images")
        has_relative_image_path = isinstance(images, list) and any(
            isinstance(image_path, str) and not Path(image_path).is_absolute()
            for image_path in images
        )
        rows.append((image_id, raw_line, has_relative_image_path))
    return rows


def _read_existing_rollout_image_ids(path: Path) -> list[str]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("existing rollout artifact is not valid JSON") from exc
    if not isinstance(document, Mapping):
        raise ValueError("existing rollout artifact must be a JSON object")
    if document.get("schema_version") != ROLLOUT_SCHEMA_VERSION:
        raise ValueError(
            "existing rollout artifact schema_version must be "
            f"{ROLLOUT_SCHEMA_VERSION!r}"
        )
    rollouts = document.get("rollouts")
    if not isinstance(rollouts, list) or not rollouts:
        raise ValueError("existing rollout artifact must contain a non-empty rollouts list")

    image_ids: list[str] = []
    seen: set[str] = set()
    for rollout_index, rollout in enumerate(rollouts, start=1):
        if not isinstance(rollout, Mapping):
            raise ValueError(f"existing rollout {rollout_index} is not an object")
        image_id = _canonical_image_id(
            rollout.get("image_id"), context=f"existing rollout {rollout_index}"
        )
        if image_id in seen:
            raise ValueError(
                "existing rollout artifact contains duplicate image_id: "
                f"{image_id}; use a one-rollout-per-image artifact"
            )
        seen.add(image_id)
        image_ids.append(image_id)
    return image_ids


def _require_expected_count(
    *, actual: int, expected: int | None, label: str
) -> None:
    if expected is not None and actual != expected:
        raise ValueError(f"{label} is {actual}, expected {expected}")


def _prepare_immutable_outputs(output_jsonl: Path, receipt: Path) -> None:
    for path in (output_jsonl, receipt):
        if path.exists():
            raise FileExistsError(f"immutable output already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)


def materialize_candidate_pool_rollout_complement(
    *,
    candidate_pool: str | Path,
    existing_rollout_artifact: str | Path,
    output_jsonl: str | Path,
    receipt: str | Path,
    expected_pool_count: int | None = None,
    expected_existing_count: int | None = None,
    expected_complement_count: int | None = None,
) -> Mapping[str, Any]:
    """Write pool rows whose image identifiers do not appear in one rollout artifact."""

    candidate_pool_path = Path(candidate_pool).expanduser().resolve(strict=True)
    existing_rollout_path = Path(existing_rollout_artifact).expanduser().resolve(
        strict=True
    )
    output_path = Path(output_jsonl).expanduser().resolve()
    receipt_path = Path(receipt).expanduser().resolve()

    candidate_rows = _read_candidate_pool(candidate_pool_path)
    if output_path.parent != candidate_pool_path.parent and any(
        has_relative_image_path for _, _, has_relative_image_path in candidate_rows
    ):
        raise ValueError(
            "byte-preserving complement output with relative image paths must be "
            "written beside the candidate pool"
        )
    _prepare_immutable_outputs(output_path, receipt_path)
    existing_image_ids = _read_existing_rollout_image_ids(existing_rollout_path)
    candidate_image_ids = [image_id for image_id, _, _ in candidate_rows]
    candidate_set = set(candidate_image_ids)
    existing_set = set(existing_image_ids)
    missing_from_pool = sorted(existing_set - candidate_set)
    if missing_from_pool:
        raise ValueError(
            "existing rollout image_ids are not a subset of candidate pool: "
            + ", ".join(missing_from_pool)
        )

    complement_rows = [
        raw_line
        for image_id, raw_line, _ in candidate_rows
        if image_id not in existing_set
    ]
    complement_image_ids = [image_id for image_id in candidate_image_ids if image_id not in existing_set]
    _require_expected_count(
        actual=len(candidate_rows), expected=expected_pool_count, label="candidate pool count"
    )
    _require_expected_count(
        actual=len(existing_image_ids),
        expected=expected_existing_count,
        label="existing rollout image count",
    )
    _require_expected_count(
        actual=len(complement_rows),
        expected=expected_complement_count,
        label="complement count",
    )

    output_bytes = b"".join(complement_rows)
    output_path.open("xb").write(output_bytes)
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "sources": {
            "candidate_pool": {
                "path": str(candidate_pool_path),
                "sha256": _sha256_file(candidate_pool_path),
            },
            "existing_rollout_artifact": {
                "path": str(existing_rollout_path),
                "sha256": _sha256_file(existing_rollout_path),
            },
        },
        "counts": {
            "candidate_pool": len(candidate_rows),
            "existing_rollout_images": len(existing_image_ids),
            "complement": len(complement_rows),
        },
        "existing_ordered_image_ids_sha256": _ordered_image_ids_sha256(existing_image_ids),
        "complement_ordered_image_ids_sha256": _ordered_image_ids_sha256(
            complement_image_ids
        ),
        "output_jsonl": {
            "path": str(output_path),
            "output_row_sha256": _sha256_bytes(output_bytes),
        },
    }
    receipt_path.open("xb").write(_canonical_json_bytes(document))
    return document


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", required=True, type=Path)
    parser.add_argument("--existing-rollout-artifact", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--expected-pool-count", type=int)
    parser.add_argument("--expected-existing-count", type=int)
    parser.add_argument("--expected-complement-count", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = materialize_candidate_pool_rollout_complement(
        candidate_pool=args.candidate_pool,
        existing_rollout_artifact=args.existing_rollout_artifact,
        output_jsonl=args.output_jsonl,
        receipt=args.receipt,
        expected_pool_count=args.expected_pool_count,
        expected_existing_count=args.expected_existing_count,
        expected_complement_count=args.expected_complement_count,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
