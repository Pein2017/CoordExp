#!/usr/bin/env python3
"""Materialize the exact image cohort of a StateBank as inference JSONL.

The source candidate-pool rows are retained semantically, while relative image
paths are resolved to absolute paths so the derived panel can live beside the
experiment artifacts.  A receipt binds the input pool, StateBank manifest,
record file, ordered image identities, and output bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "state_bank_inference_panel.v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_line(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def materialize(
    *, candidate_pool: Path, state_bank_manifest: Path, output_root: Path
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"immutable output root already exists: {output_root}")
    candidate_pool = candidate_pool.expanduser().resolve(strict=True)
    state_bank_manifest = state_bank_manifest.expanduser().resolve(strict=True)
    manifest = json.loads(state_bank_manifest.read_text(encoding="utf-8"))
    records_path = state_bank_manifest.parent / "records.jsonl"
    expected_records_hash = str(manifest["records_sha256"])
    actual_records_hash = _sha256_file(records_path)
    if actual_records_hash != expected_records_hash:
        raise ValueError(
            f"StateBank records hash mismatch: {actual_records_hash} != {expected_records_hash}"
        )

    ordered_ids: list[str] = []
    for line_number, line in enumerate(records_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        image_id = str(int(row["image"]["image_id"]))
        if image_id in ordered_ids:
            raise ValueError(f"duplicate StateBank image_id at line {line_number}: {image_id}")
        ordered_ids.append(image_id)
    if len(ordered_ids) != int(manifest["record_count"]):
        raise ValueError("StateBank record count does not match manifest")

    wanted = set(ordered_ids)
    rows_by_id: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(candidate_pool.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        image_id = str(int(row["image_id"]))
        if image_id not in wanted:
            continue
        if image_id in rows_by_id:
            raise ValueError(f"duplicate candidate-pool image_id at line {line_number}: {image_id}")
        images = row.get("images")
        if not isinstance(images, list) or not images:
            raise ValueError(f"candidate-pool row omits images: {image_id}")
        resolved_images: list[str] = []
        for raw in images:
            path = Path(str(raw))
            resolved = path if path.is_absolute() else candidate_pool.parent / path
            absolute = resolved.resolve(strict=True)
            resolved_images.append(os.path.relpath(absolute, start=output_root))
        row["images"] = resolved_images
        rows_by_id[image_id] = row
    missing = [image_id for image_id in ordered_ids if image_id not in rows_by_id]
    if missing:
        raise ValueError(f"StateBank images missing from candidate pool: {missing}")

    output_root.mkdir(parents=True)
    output_path = output_root / "panel.coord.jsonl"
    receipt_path = output_root / "receipt.json"
    output_bytes = b"".join(_canonical_line(rows_by_id[image_id]) for image_id in ordered_ids)
    output_path.write_bytes(output_bytes)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "candidate_pool": {
            "path": str(candidate_pool),
            "sha256": _sha256_file(candidate_pool),
        },
        "state_bank": {
            "manifest_path": str(state_bank_manifest),
            "manifest_sha256": _sha256_file(state_bank_manifest),
            "bank_id": str(manifest["bank_id"]),
            "records_path": str(records_path.resolve()),
            "records_sha256": actual_records_hash,
        },
        "output": {
            "path": str(output_path.resolve()),
            "sha256": hashlib.sha256(output_bytes).hexdigest(),
            "image_count": len(ordered_ids),
            "ordered_image_ids": ordered_ids,
        },
    }
    receipt_path.write_bytes(_canonical_line(receipt))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--state-bank-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    receipt = materialize(
        candidate_pool=args.candidate_pool,
        state_bank_manifest=args.state_bank_manifest,
        output_root=args.output_root.expanduser().resolve(),
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
