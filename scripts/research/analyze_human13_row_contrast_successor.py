#!/usr/bin/env python3
"""Analyze R1/R2 outputs against the immutable Human-13 Source baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.analyze_human13_k_union import (
    _analyze_outputs_unsafe,
    load_outputs,
)
from scripts.research.build_human13_k_union_manifest import (
    load_manifest,
    validate_manifest,
)
from scripts.research.human13_live_eval import source_outputs_from_manifest


def analyze_successor_outputs(
    *,
    manifest_path: str | Path,
    source_discovery_path: str | Path,
    output_paths: Sequence[str | Path],
    successor_arm_ids: Sequence[str],
    resolved_plan_sha256: str,
    resolved_config_sha256: str,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path).resolve(strict=True)
    manifest = load_manifest(manifest_file, require_full_panel=True)
    validate_manifest(manifest, require_full_panel=True)
    arms = tuple(dict.fromkeys(str(value) for value in successor_arm_ids))
    if not arms or any(not value for value in arms):
        raise ValueError("successor analysis requires declared arm identities")
    existing = {item.arm_id for item in manifest.arms}
    if existing.intersection(arms):
        raise ValueError("successor analysis arm collides with the sealed manifest")
    source_records = _source_records(source_discovery_path)
    source_outputs = source_outputs_from_manifest(
        manifest=manifest,
        manifest_sha256=_sha256_file(manifest_file),
        source_discovery_records=source_records,
        resolved_arm_plan_sha256=resolved_plan_sha256,
        resolved_config_sha256=resolved_config_sha256,
    )
    outputs = [dict(item) for item in source_outputs]
    for path in output_paths:
        outputs.extend(load_outputs(path))
    observed = {
        str(item.get("arm_id"))
        for item in outputs
        if item.get("arm_id") != "frozen_source"
    }
    if observed != set(arms):
        raise ValueError("successor output arms differ from the declared analysis view")
    result = _analyze_outputs_unsafe(manifest, outputs, allowed_arm_ids=arms)
    result["successor_analysis"] = {
        "schema_version": "human13_row_contrast_analysis.v1",
        "sealed_manifest_unchanged": True,
        "declared_successor_arms": list(arms),
        "source_discovery_sha256": _sha256_file(Path(source_discovery_path)),
        "output_sha256": {
            str(Path(path).resolve(strict=True)): _sha256_file(Path(path))
            for path in output_paths
        },
    }
    return result


def _source_records(path: str | Path) -> dict[int, Mapping[str, Any]]:
    records: dict[int, Mapping[str, Any]] = {}
    for line in (
        Path(path).resolve(strict=True).read_text(encoding="utf-8").splitlines()
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        image_id = value.get("image_id") if isinstance(value, Mapping) else None
        if (
            isinstance(image_id, bool)
            or not isinstance(image_id, int)
            or image_id in records
        ):
            raise ValueError("source discovery requires unique integer image IDs")
        records[image_id] = value
    return records


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).resolve(strict=True).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-discovery", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, action="append", required=True)
    parser.add_argument("--arm-id", action="append", required=True)
    parser.add_argument("--resolved-plan-sha256", required=True)
    parser.add_argument("--resolved-config-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite analysis: {args.output}")
    result = analyze_successor_outputs(
        manifest_path=args.manifest,
        source_discovery_path=args.source_discovery,
        output_paths=args.outputs,
        successor_arm_ids=args.arm_id,
        resolved_plan_sha256=args.resolved_plan_sha256,
        resolved_config_sha256=args.resolved_config_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["analyze_successor_outputs"]
