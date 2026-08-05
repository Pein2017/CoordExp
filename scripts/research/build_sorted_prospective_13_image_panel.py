#!/usr/bin/env python3
"""Materialize the prospective 13-image panel-admission probe input.

This is a data-admission seam only.  It byte-preserves every line of the
frozen twelve-image human-refined panel and inserts exactly one additional
row -- COCO ``val2017`` image ``2299`` read from the training authority file
-- in numeric ``image_id`` order.  The inserted row references the canonical
on-disk image by a normalized relative path and a bound SHA-256; the JPEG
bytes are never copied into the output tree.  It does not run a model, score
a checkpoint, or change the completed twelve-image denominator used by prior
units.

Before publishing, every one of the 13 published rows' ``images`` references
is resolved and verified to exist against a staged directory at the exact
depth of the final output (``Path.resolve(strict=True)`` requires each
intermediate path component to physically exist, so this staging happens
inside the same atomic temp-directory publish used for the final commit).
The resulting per-image identity table (``image_id``, resolved path, byte
size, SHA-256) is sealed into the receipt alongside a table-level digest.
"""

from __future__ import annotations

import argparse
import bisect
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any


UNIT_ID = "2026-08-04-sorted-prospective-13-image-panel-admission"
SCHEMA_VERSION = "sorted_prospective_13_image_panel.v2"

ROOT = Path("/data/CoordExp")
OUTPUTS = ROOT / "outputs/research/qwen3-vl-dense-enumeration"

LEGACY_PANEL_PATH = (
    OUTPUTS
    / "2026-07-21-best-sampled-trajectory-positive-row-imitation-screen"
    / "evaluation-inputs/human-refined-12.coord.jsonl"
)
AUTHORITY_PATH = ROOT / "public_data/coco/rescale_32_1024_bbox_len12000/val.coord.jsonl"

UNIT_OUTPUT_ROOT = OUTPUTS / UNIT_ID
OUTPUT_JSONL_RELATIVE = "evaluation-inputs/human-refined-13.coord.jsonl"
RECEIPT_RELATIVE = "receipt.json"

LEGACY_PANEL_SHA256 = "cfe4f693133287f9e6c561fc094710c642f049aec3d50f44dea98b764ba2aa85"
AUTHORITY_SHA256 = "81d674070d4b588488a2cb911c09f765b63c0e6d035b50db27ee0a41ff2a1894"
TARGET_LINE_SHA256 = "ce19853c74a595f22cc183ce450e561f2da3216e54a1e499cfbca1be7e1c425b"
TARGET_IMAGE_SHA256 = "cd7199a37188c9ac6481520175866cbd78fa6fba35f4290bb0b60c78afcb2df3"

TARGET_IMAGE_ID = 2299
EXPECTED_LEGACY_LINE_COUNT = 12
EXPECTED_OWNER_TOTAL = 46
EXPECTED_PERSON_COUNT = 38
EXPECTED_TIE_COUNT = 8
EXPECTED_LEGACY_OWNER_TOTAL = 346
EXPECTED_FULL_PANEL_OWNER_TOTAL = EXPECTED_LEGACY_OWNER_TOTAL + EXPECTED_OWNER_TOTAL


class PanelContractError(ValueError):
    """Raised when a frozen input or panel invariant is not satisfied."""


@dataclass(frozen=True)
class SourcePaths:
    """All immutable inputs and expected identities for one panel build."""

    legacy_panel: Path = LEGACY_PANEL_PATH
    legacy_panel_sha256: str = LEGACY_PANEL_SHA256
    expected_legacy_line_count: int = EXPECTED_LEGACY_LINE_COUNT
    authority: Path = AUTHORITY_PATH
    authority_sha256: str = AUTHORITY_SHA256
    target_image_id: int = TARGET_IMAGE_ID
    target_line_sha256: str = TARGET_LINE_SHA256
    target_image_sha256: str = TARGET_IMAGE_SHA256
    expected_owner_total: int = EXPECTED_OWNER_TOTAL
    expected_person_count: int = EXPECTED_PERSON_COUNT
    expected_tie_count: int = EXPECTED_TIE_COUNT
    expected_legacy_owner_total: int = EXPECTED_LEGACY_OWNER_TOTAL
    expected_full_panel_owner_total: int = EXPECTED_FULL_PANEL_OWNER_TOTAL


DEFAULT_SOURCES = SourcePaths()


def canonical_json_bytes(value: Any) -> bytes:
    """The one JSON encoding used by all panel and receipt identities."""

    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_legacy_lines(sources: SourcePaths) -> tuple[bytes, list[bytes]]:
    legacy_panel = sources.legacy_panel.expanduser().resolve(strict=True)
    raw = legacy_panel.read_bytes()
    actual_sha256 = sha256_bytes(raw)
    if actual_sha256 != sources.legacy_panel_sha256:
        raise PanelContractError(
            f"legacy panel sha256 mismatch: {actual_sha256} != {sources.legacy_panel_sha256}"
        )
    lines = raw.splitlines(keepends=True)
    if len(lines) != sources.expected_legacy_line_count:
        raise PanelContractError(
            f"legacy panel line count mismatch: {len(lines)} != "
            f"{sources.expected_legacy_line_count}"
        )
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            raise PanelContractError(f"legacy panel line {index} is blank")
        if not line.endswith(b"\n"):
            raise PanelContractError(f"legacy panel line {index} is not newline-terminated")
    return raw, lines


def _legacy_image_ids(lines: Sequence[bytes]) -> list[int]:
    ids: list[int] = []
    for index, line in enumerate(lines, start=1):
        row = json.loads(line)
        image_id = row.get("image_id")
        if not isinstance(image_id, int) or isinstance(image_id, bool):
            raise PanelContractError(f"legacy panel line {index} has a non-integer image_id")
        ids.append(image_id)
    if ids != sorted(ids):
        raise PanelContractError("legacy panel image_ids are not sorted ascending")
    if len(set(ids)) != len(ids):
        raise PanelContractError("legacy panel contains a duplicate image_id")
    return ids


def _find_authority_row(sources: SourcePaths) -> tuple[Path, bytes, dict[str, Any], int]:
    authority = sources.authority.expanduser().resolve(strict=True)
    raw = authority.read_bytes()
    actual_sha256 = sha256_bytes(raw)
    if actual_sha256 != sources.authority_sha256:
        raise PanelContractError(
            f"authority file sha256 mismatch: {actual_sha256} != {sources.authority_sha256}"
        )
    matches: list[tuple[int, bytes, dict[str, Any]]] = []
    for index, line in enumerate(raw.splitlines(keepends=True), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("image_id") == sources.target_image_id:
            matches.append((index, line, row))
    if len(matches) != 1:
        raise PanelContractError(
            f"expected exactly one authority row for image_id={sources.target_image_id}, "
            f"found {len(matches)}"
        )
    line_number, raw_line, row = matches[0]
    actual_line_sha256 = sha256_bytes(raw_line)
    if actual_line_sha256 != sources.target_line_sha256:
        raise PanelContractError(
            f"target authority line sha256 mismatch: {actual_line_sha256} != "
            f"{sources.target_line_sha256}"
        )
    return authority, raw_line, row, line_number


def _aggregate_owner_counts(rows: Sequence[Mapping[str, Any]], *, context: str) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    total = 0
    for row in rows:
        objects = row.get("objects")
        if not isinstance(objects, list) or not objects:
            raise PanelContractError(f"{context} row has no objects (image_id={row.get('image_id')})")
        total += len(objects)
        counts.update(str(obj.get("desc")) for obj in objects)
    return {"total": total, "by_desc": dict(sorted(counts.items()))}


def _verify_target_owner_counts(row: Mapping[str, Any], sources: SourcePaths) -> dict[str, Any]:
    counts = _aggregate_owner_counts([row], context="target")
    total = counts["total"]
    person = counts["by_desc"].get("person", 0)
    tie = counts["by_desc"].get("tie", 0)
    if (
        total != sources.expected_owner_total
        or person != sources.expected_person_count
        or tie != sources.expected_tie_count
    ):
        raise PanelContractError(
            "target row owner-count mismatch: "
            f"total={total} person={person} tie={tie} (expected "
            f"total={sources.expected_owner_total} person={sources.expected_person_count} "
            f"tie={sources.expected_tie_count})"
        )
    return {"total": total, "person": person, "tie": tie, "by_desc": counts["by_desc"]}


def _verify_legacy_owner_counts(rows: Sequence[Mapping[str, Any]], sources: SourcePaths) -> dict[str, Any]:
    counts = _aggregate_owner_counts(rows, context="legacy panel")
    if counts["total"] != sources.expected_legacy_owner_total:
        raise PanelContractError(
            f"legacy panel object-count mismatch: total={counts['total']} "
            f"(expected {sources.expected_legacy_owner_total})"
        )
    return counts


def _verify_full_panel_owner_counts(
    rows: Sequence[Mapping[str, Any]],
    *,
    legacy_total: int,
    target_total: int,
    sources: SourcePaths,
) -> dict[str, Any]:
    counts = _aggregate_owner_counts(rows, context="full panel")
    if counts["total"] != sources.expected_full_panel_owner_total:
        raise PanelContractError(
            f"full-panel object-count mismatch: total={counts['total']} "
            f"(expected {sources.expected_full_panel_owner_total})"
        )
    if counts["total"] != legacy_total + target_total:
        raise PanelContractError(
            f"full-panel total {counts['total']} does not equal legacy total {legacy_total} "
            f"plus target total {target_total}"
        )
    return counts


def _normalize_target_row(
    row: Mapping[str, Any], *, authority: Path, eval_dir: Path, sources: SourcePaths
) -> tuple[dict[str, Any], dict[str, Any]]:
    images = row.get("images")
    if not isinstance(images, list) or len(images) != 1:
        raise PanelContractError("target row must reference exactly one image")
    resolved_image = (authority.parent / str(images[0])).resolve(strict=True)
    image_sha256 = sha256_file(resolved_image)
    if image_sha256 != sources.target_image_sha256:
        raise PanelContractError(
            f"target image sha256 mismatch: {image_sha256} != {sources.target_image_sha256}"
        )
    normalized_ref = os.path.relpath(resolved_image, start=eval_dir)
    normalized_row = dict(row)
    normalized_row["images"] = [normalized_ref]
    image_info = {
        "canonical_path": str(resolved_image),
        "sha256": image_sha256,
        "panel_relative_path": normalized_ref,
    }
    return normalized_row, image_info


def _resolve_and_verify_all_images(
    all_lines: Sequence[bytes], *, eval_dir: Path
) -> list[dict[str, Any]]:
    """Fail-closed resolution/existence/hash check for every published row.

    ``eval_dir`` must already physically exist (staged at the exact depth of
    the final published location) because ``Path.resolve(strict=True)``
    requires every intermediate path component to exist before it will
    collapse a ``..`` traversal.
    """

    manifest: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(all_lines, start=1):
        row = json.loads(raw_line)
        image_id = row.get("image_id")
        images = row.get("images")
        if not isinstance(images, list) or len(images) != 1:
            raise PanelContractError(
                f"panel row {line_number} (image_id={image_id}) must reference exactly one image"
            )
        raw_ref = str(images[0])
        try:
            resolved = (eval_dir / raw_ref).resolve(strict=True)
        except FileNotFoundError as exc:
            raise PanelContractError(
                f"panel row {line_number} (image_id={image_id}) image reference does not "
                f"resolve to an existing file: {raw_ref}"
            ) from exc
        if not resolved.is_file():
            raise PanelContractError(
                f"panel row {line_number} (image_id={image_id}) resolved image is not a "
                f"regular file: {resolved}"
            )
        manifest.append(
            {
                "line_number": line_number,
                "image_id": image_id,
                "panel_relative_path": raw_ref,
                "resolved_path": str(resolved),
                "byte_size": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )
    return manifest


def _commit_or_discard(
    output_root: Path, temp_dir: Path, files: Mapping[str, bytes]
) -> str:
    """Publish ``temp_dir`` as ``output_root``, or discard it if identical content exists.

    ``temp_dir`` is always fully staged (including the full-panel image
    verification) before this is called.  A pre-existing, byte-identical
    ``output_root`` makes this call a no-op republish; any foreign file or
    content divergence raises instead of silently overwriting.
    """

    if output_root.exists():
        try:
            if not output_root.is_dir():
                raise PanelContractError(
                    f"output path exists but is not a directory: {output_root}"
                )
            expected_names = set(files)
            existing_names = {
                str(path.relative_to(output_root))
                for path in output_root.rglob("*")
                if path.is_file()
            }
            if existing_names != expected_names:
                raise PanelContractError(
                    "output root already exists with a foreign or partial file set"
                )
            mismatched = [
                name
                for name, expected in files.items()
                if (output_root / name).read_bytes() != expected
            ]
            if mismatched:
                raise PanelContractError(
                    f"output root already exists with non-identical content: {mismatched}"
                )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        return "identical_existing_output"

    output_root.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temp_dir, output_root)
    return "created"


def build_panel(output_root: Path, *, sources: SourcePaths = DEFAULT_SOURCES) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    eval_dir = output_root / "evaluation-inputs"

    legacy_raw, legacy_lines = _read_legacy_lines(sources)
    legacy_ids = _legacy_image_ids(legacy_lines)
    if sources.target_image_id in legacy_ids:
        raise PanelContractError(
            f"target image_id {sources.target_image_id} is already present in the legacy panel"
        )
    legacy_rows = [json.loads(line) for line in legacy_lines]
    legacy_owner_counts = _verify_legacy_owner_counts(legacy_rows, sources)

    authority, target_raw_line, target_row, target_line_number = _find_authority_row(sources)
    target_owner_counts = _verify_target_owner_counts(target_row, sources)
    normalized_row, image_info = _normalize_target_row(
        target_row, authority=authority, eval_dir=eval_dir, sources=sources
    )
    target_line_bytes = canonical_json_bytes(normalized_row) + b"\n"

    insertion_index = bisect.bisect_left(legacy_ids, sources.target_image_id)
    all_lines = list(legacy_lines)
    all_lines.insert(insertion_index, target_line_bytes)
    ordered_ids = list(legacy_ids)
    ordered_ids.insert(insertion_index, sources.target_image_id)

    reconstructed_legacy = b"".join(
        line for index, line in enumerate(all_lines) if index != insertion_index
    )
    if reconstructed_legacy != legacy_raw:
        raise PanelContractError("legacy panel lines were not preserved byte-for-byte")

    full_panel_rows = list(legacy_rows)
    full_panel_rows.insert(insertion_index, normalized_row)
    full_panel_owner_counts = _verify_full_panel_owner_counts(
        full_panel_rows,
        legacy_total=legacy_owner_counts["total"],
        target_total=target_owner_counts["total"],
        sources=sources,
    )

    output_bytes = b"".join(all_lines)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent))
    try:
        staged_eval_dir = temp_dir / "evaluation-inputs"
        staged_eval_dir.mkdir(parents=True)
        (staged_eval_dir / "human-refined-13.coord.jsonl").write_bytes(output_bytes)

        images_manifest = _resolve_and_verify_all_images(all_lines, eval_dir=staged_eval_dir)
        target_manifest_entries = [
            entry for entry in images_manifest if entry["image_id"] == sources.target_image_id
        ]
        if len(target_manifest_entries) != 1:
            raise PanelContractError(
                f"expected exactly one images-manifest entry for target image_id="
                f"{sources.target_image_id}, found {len(target_manifest_entries)}"
            )
        target_manifest_entry = target_manifest_entries[0]
        if target_manifest_entry["sha256"] != sources.target_image_sha256:
            raise PanelContractError(
                "published target image sha256 mismatch: "
                f"{target_manifest_entry['sha256']} != {sources.target_image_sha256}"
            )

        receipt: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "unit_id": UNIT_ID,
            "scope": {
                "purpose": "prospective probe input only",
                "completed_denominator_unchanged": True,
                "pooled_13_image_summary_permitted": False,
                "required_reporting": (
                    "future results must report legacy-12 and image-2299 side-by-side "
                    "before any pooled 13-image summary"
                ),
            },
            "inputs": {
                "legacy_panel": {
                    "path": str(sources.legacy_panel.expanduser().resolve()),
                    "sha256": sources.legacy_panel_sha256,
                    "line_count": len(legacy_lines),
                },
                "authority": {
                    "path": str(authority),
                    "sha256": sources.authority_sha256,
                    "target_line_number": target_line_number,
                    "target_line_sha256": sources.target_line_sha256,
                },
                "target_image": {
                    "image_id": sources.target_image_id,
                    **image_info,
                },
            },
            "legacy_owner_counts": legacy_owner_counts,
            "target_owner_counts": target_owner_counts,
            "full_panel_owner_counts": full_panel_owner_counts,
            "images_manifest": {
                "sha256": sha256_json(images_manifest),
                "entries": images_manifest,
            },
            "output": {
                "jsonl_relative_path": OUTPUT_JSONL_RELATIVE,
                "jsonl_sha256": sha256_bytes(output_bytes),
                "line_count": len(all_lines),
                "ordered_image_ids": ordered_ids,
                "insertion_index": insertion_index,
                "legacy_lines_preserved_byte_for_byte": True,
            },
        }
        receipt["receipt_content_sha256"] = sha256_json(
            {key: value for key, value in receipt.items() if key != "receipt_content_sha256"}
        )
        receipt_bytes = canonical_json_bytes(receipt) + b"\n"
        (temp_dir / "receipt.json").write_bytes(receipt_bytes)

        files = {
            OUTPUT_JSONL_RELATIVE: output_bytes,
            RECEIPT_RELATIVE: receipt_bytes,
        }
        status = _commit_or_discard(output_root, temp_dir, files)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return {**receipt, "status": status, "output_root": str(output_root)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=UNIT_OUTPUT_ROOT)
    args = parser.parse_args()
    result = build_panel(args.output_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
