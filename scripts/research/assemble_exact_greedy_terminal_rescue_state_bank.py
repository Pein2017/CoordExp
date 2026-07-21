#!/usr/bin/env python3
"""Deterministically assemble exact-terminal rescue collector shards.

The collector intentionally writes paired rollout and review rows rather than
calling the immutable StateBank assembler.  This command is the small offline
join between those files.  It validates the collector receipt, verifies every
declared source checksum, binds all events to the exact reference checkpoint
and prompt contract, then delegates schema validation and immutable writing to
``src.rollout_calibration.assemble_state_bank``.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.config.fingerprint import sha256_file  # noqa: E402
from src.rollout_calibration import (  # noqa: E402
    assemble_state_bank,
    load_state_bank,
    load_state_bank_manifest_binding,
)


SCHEMA_VERSION = "exact_greedy_terminal_rescue_state_bank_assembler.v1"
DEFAULT_REFERENCE_MANIFEST = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-20-own-prefix-entity-transition-and-coordinate-boundary-calibration-training-screen/"
    "smoke-a-combined-v4/state-bank/manifest.json"
)
REQUIRED_FILES = ("rollout_rows.jsonl", "review_rows.jsonl", "collection-receipt.json")


class AssemblyError(ValueError):
    """Raised when collector evidence cannot be assembled safely."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AssemblyError(f"invalid JSON: {path}: {exc}") from exc


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load non-empty JSON object rows, preserving file order for joining."""

    resolved = Path(path).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    with resolved.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AssemblyError(f"invalid JSONL at {resolved}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise AssemblyError(f"JSONL row at {resolved}:{line_number} must be an object")
            rows.append(value)
    return rows


def _sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise AssemblyError(f"{field} must be a SHA-256 hex string")
    try:
        int(value, 16)
    except ValueError as exc:
        raise AssemblyError(f"{field} must be a SHA-256 hex string") from exc
    return value


def _event_id(row: Mapping[str, Any], *, field: str) -> str:
    value = row.get("event_id")
    if not isinstance(value, str) or not value:
        raise AssemblyError(f"{field}.event_id must be a non-empty string")
    return value


def _receipt_source_artifacts(receipt: Mapping[str, Any], *, root: Path) -> list[dict[str, str]]:
    declared = receipt.get("source_artifacts")
    if not isinstance(declared, Mapping):
        raise AssemblyError(f"{root}/collection-receipt.json.source_artifacts must be an object")
    result: list[dict[str, str]] = []
    for key in sorted(declared):
        item = declared[key]
        if not isinstance(item, Mapping):
            raise AssemblyError(f"source_artifacts[{key!r}] must be an object")
        path_value = item.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise AssemblyError(f"source_artifacts[{key!r}].path must be non-empty")
        declared_sha = _sha256(item.get("sha256"), field=f"source_artifacts[{key!r}].sha256")
        path = Path(path_value).expanduser().resolve()
        if not path.is_file():
            raise AssemblyError(f"declared source artifact does not exist: {path}")
        actual_sha = sha256_file(path)
        if actual_sha != declared_sha:
            raise AssemblyError(
                f"source artifact checksum mismatch for {key!r}: "
                f"expected {declared_sha}, got {actual_sha} ({path})"
            )
        result.append({"artifact_id": str(key), "sha256": actual_sha})
    return result


def _file_artifacts(root: Path, receipt: Mapping[str, Any], *, shard_index: int) -> list[dict[str, str]]:
    """Return deterministic hashes for receipt and collector rows as well as declared inputs."""

    artifacts: list[dict[str, str]] = []
    for name in ("collection-receipt.json", "rollout_rows.jsonl", "review_rows.jsonl"):
        path = root / name
        artifacts.append(
            {
                "artifact_id": f"collector-{shard_index:04d}-{name.replace('.', '-').replace('_', '-')}",
                "sha256": sha256_file(path),
            }
        )
    # Optional files are evidence too when present, but are not required for
    # assembly.  Including them makes the resulting bank receipt complete
    # without forcing older collectors to create new files.
    for name in ("scan.json", "sample_attempts.jsonl"):
        path = root / name
        if path.is_file():
            artifacts.append(
                {
                    "artifact_id": f"collector-{shard_index:04d}-{name.replace('.', '-').replace('_', '-')}",
                    "sha256": sha256_file(path),
                }
            )
    artifacts.extend(
        {
            "artifact_id": f"collector-{shard_index:04d}-{item['artifact_id']}",
            "sha256": item["sha256"],
        }
        for item in _receipt_source_artifacts(receipt, root=root)
    )
    return artifacts


def _validate_receipt(
    receipt: Mapping[str, Any],
    *,
    root: Path,
    reference_manifest: Path,
    source_checkpoint_id: str,
    rollout_count: int,
) -> list[dict[str, str]]:
    if receipt.get("schema_version") != "exact_greedy_terminal_rescue_collector.v1":
        raise AssemblyError(f"unsupported collector schema in {root}/collection-receipt.json")
    selected_count = receipt.get("selected_record_count")
    if not isinstance(selected_count, int) or selected_count < rollout_count:
        raise AssemblyError(
            f"collector selected_record_count is smaller than accepted rollout rows in {root}: "
            f"{selected_count} < {rollout_count}"
        )
    runtime = receipt.get("runtime")
    if not isinstance(runtime, Mapping):
        raise AssemblyError(f"collector receipt has no completed runtime section in {root}")
    if runtime.get("status") != "completed":
        raise AssemblyError(f"collector runtime is not completed in {root}")
    if runtime.get("accepted_event_count") != rollout_count:
        raise AssemblyError(
            f"collector accepted_event_count disagrees with rollout rows in {root}: "
            f"{runtime.get('accepted_event_count')} != {rollout_count}"
        )
    if runtime.get("checkpoint_id") != source_checkpoint_id:
        raise AssemblyError(f"collector checkpoint differs from reference in {root}")
    declared_reference = runtime.get("reference_state_bank_manifest")
    if declared_reference is not None and Path(str(declared_reference)).expanduser().resolve() != reference_manifest:
        raise AssemblyError(f"collector reference manifest differs from requested reference in {root}")
    declared_reference_sha = runtime.get("reference_state_bank_manifest_sha256")
    if declared_reference_sha is not None and declared_reference_sha != sha256_file(reference_manifest):
        raise AssemblyError(f"collector reference manifest checksum differs in {root}")
    return _file_artifacts(root, receipt, shard_index=0)


def merge_collector_shards(
    collector_dirs: Sequence[str | Path],
    *,
    reference_manifest: str | Path = DEFAULT_REFERENCE_MANIFEST,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]], Any]:
    """Validate and deterministically merge collector directories."""

    reference = Path(reference_manifest).expanduser().resolve(strict=True)
    binding = load_state_bank_manifest_binding(reference)
    roots = sorted({Path(item).expanduser().resolve() for item in collector_dirs}, key=str)
    if not roots:
        raise AssemblyError("at least one collector directory is required")
    rollout_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    source_artifacts: list[dict[str, str]] = [
        {"artifact_id": "reference-state-bank-manifest", "sha256": sha256_file(reference)}
    ]
    seen: set[str] = set()
    for shard_index, root in enumerate(roots):
        if not root.is_dir():
            raise AssemblyError(f"collector directory does not exist: {root}")
        for name in REQUIRED_FILES:
            if not (root / name).is_file():
                raise AssemblyError(f"collector directory is missing {name}: {root}")
        rollouts = load_jsonl(root / "rollout_rows.jsonl")
        reviews = load_jsonl(root / "review_rows.jsonl")
        receipt = _read_json(root / "collection-receipt.json")
        if not isinstance(receipt, Mapping):
            raise AssemblyError(f"collector receipt must be an object: {root}")
        artifacts = _file_artifacts(root, receipt, shard_index=shard_index)
        _validate_receipt(
            receipt,
            root=root,
            reference_manifest=reference,
            source_checkpoint_id=binding.source_checkpoint_id,
            rollout_count=len(rollouts),
        )
        if len(rollouts) != len(reviews):
            raise AssemblyError(f"rollout/review row count mismatch in {root}")
        for index, row in enumerate(rollouts):
            event_id = _event_id(row, field=f"{root}/rollout_rows[{index}]")
            if event_id in seen:
                raise AssemblyError(f"duplicate event_id across collector shards: {event_id}")
            seen.add(event_id)
            for candidate in row.get("candidates", ()):
                if not isinstance(candidate, Mapping):
                    raise AssemblyError(f"candidate must be an object: {event_id}")
                provenance = candidate.get("generation_provenance")
                if isinstance(provenance, Mapping) and provenance.get("checkpoint_id") != binding.source_checkpoint_id:
                    raise AssemblyError(f"candidate checkpoint differs from reference: {event_id}")
        review_ids: list[str] = []
        for index, row in enumerate(reviews):
            event_id = _event_id(row, field=f"{root}/review_rows[{index}]")
            review_ids.append(event_id)
        rollout_ids = [row["event_id"] for row in rollouts]
        if len(set(review_ids)) != len(review_ids):
            raise AssemblyError(f"duplicate event_id inside review rows: {root}")
        if set(review_ids) != set(rollout_ids):
            raise AssemblyError(f"rollout/review event identifiers differ in {root}")
        rollout_rows.extend(rollouts)
        review_rows.extend(reviews)
        source_artifacts.extend(artifacts)
    if len({item["artifact_id"] for item in source_artifacts}) != len(source_artifacts):
        raise AssemblyError("source artifact identifiers are not unique")
    order = sorted(range(len(rollout_rows)), key=lambda index: rollout_rows[index]["event_id"])
    rollout_rows = [rollout_rows[index] for index in order]
    review_by_id = {row["event_id"]: row for row in review_rows}
    review_rows = [review_by_id[row["event_id"]] for row in rollout_rows]
    return rollout_rows, review_rows, source_artifacts, binding


def assemble_collector_shards(
    collector_dirs: Sequence[str | Path],
    *,
    output_dir: str | Path,
    reference_manifest: str | Path = DEFAULT_REFERENCE_MANIFEST,
) -> dict[str, Any]:
    rollout_rows, review_rows, source_artifacts, binding = merge_collector_shards(
        collector_dirs, reference_manifest=reference_manifest
    )
    manifest = assemble_state_bank(
        output_dir=output_dir,
        rollout_rows=rollout_rows,
        review_rows=review_rows,
        source_checkpoint=binding.source_checkpoint,
        prompt_identity_sha256=binding.prompt_identity_sha256,
        source_artifacts=source_artifacts,
    )
    loaded = load_state_bank(
        Path(output_dir).resolve() / "manifest.json",
        expected_source_checkpoint=binding.source_checkpoint,
        expected_prompt_identity_sha256=binding.prompt_identity_sha256,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest": manifest.to_artifact_dict(),
        "validation": loaded.validation_receipt.to_artifact_dict(),
        "collector_count": len(tuple(collector_dirs)),
        "event_count": len(loaded.records),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collector-dir", action="append", required=True, dest="collector_dirs")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-manifest", default=str(DEFAULT_REFERENCE_MANIFEST))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = assemble_collector_shards(
        args.collector_dirs,
        output_dir=args.output_dir,
        reference_manifest=args.reference_manifest,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
