#!/usr/bin/env python
"""Attest one real non-identity HF transformed-media artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


def main() -> int:
    args = _parse_args()
    repo_root = Path.cwd().resolve()
    run_root = Path(args.run_root).resolve()
    source_image = Path(args.source_image).resolve()
    rows = _read_jsonl(run_root / "image_plan.jsonl")
    if len(rows) != 1:
        raise SystemExit("transformed-media probe requires exactly one image-plan row")
    row = rows[0]
    if row.get("logical_transform_id") != "hflip":
        raise SystemExit("transformed-media probe requires logical_transform_id=hflip")
    if row.get("backend_projection_evidence_kind") != "hf_executed_tensors":
        raise SystemExit("transformed-media probe lacks HF executed-tensor evidence")
    if row.get("status") != "ok" or row.get("error") is not None:
        raise SystemExit("transformed-media probe image-plan row is not successful")

    source_bytes = source_image.read_bytes()
    source_file_sha256 = hashlib.sha256(source_bytes).hexdigest()
    with Image.open(source_image) as image:
        original = image.convert("RGB")
    transformed = ImageOps.mirror(original)
    original_rgb_sha256 = _rgb8_sha256(original)
    transformed_rgb_sha256 = _rgb8_sha256(transformed)
    if row.get("image_content_sha256") != source_file_sha256:
        raise SystemExit("image-plan source-file identity does not match fixture bytes")
    if row.get("executed_media_sha256") != transformed_rgb_sha256:
        raise SystemExit("executed-media identity does not match independent hflip RGB8 hash")
    if transformed_rgb_sha256 == original_rgb_sha256:
        raise SystemExit("probe fixture is not asymmetric under hflip")
    if row.get("observed_image_grid_thw") != row.get("expected_image_grid_thw"):
        raise SystemExit("executed image grid does not match the shared plan")

    script_path = Path(__file__).resolve()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "passed",
        "run_root": _display_path(run_root, repo_root),
        "row_id": row["row_id"],
        "logical_transform_id": "hflip",
        "source_image": {
            "path": _display_path(source_image, repo_root),
            "file_sha256": source_file_sha256,
            "original_rgb8_sha256": original_rgb_sha256,
            "width": original.width,
            "height": original.height,
        },
        "executed_image": {
            "rgb8_sha256": transformed_rgb_sha256,
            "backend_projection_evidence_kind": row[
                "backend_projection_evidence_kind"
            ],
            "expected_image_grid_thw": row["expected_image_grid_thw"],
            "observed_image_grid_thw": row["observed_image_grid_thw"],
            "do_resize": row["do_resize"],
        },
        "artifact": {
            "path": _display_path(run_root / "image_plan.jsonl", repo_root),
            "sha256": _sha256_file(run_root / "image_plan.jsonl"),
            "row": row,
        },
        "source": {
            "verifier": {
                "path": _display_path(script_path, repo_root),
                "sha256": _sha256_file(script_path),
            },
            "files": {
                relative: _sha256_file(repo_root / relative)
                for relative in (
                    "src/inference/backend.py",
                    "src/inference/hf_backend.py",
                    "src/inference/pipeline.py",
                    "src/qwen/images.py",
                )
            },
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output.resolve())
    return 0


def _rgb8_sha256(image: Image.Image) -> str:
    digest = hashlib.sha256()
    digest.update(b"coordexp-rgb8-pixels-v1\0")
    digest.update(int(image.width).to_bytes(8, "big", signed=False))
    digest.update(int(image.height).to_bytes(8, "big", signed=False))
    digest.update(image.tobytes())
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--source-image", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
