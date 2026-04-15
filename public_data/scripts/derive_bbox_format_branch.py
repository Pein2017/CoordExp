#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from public_data.pipeline.naming import resolve_bbox_format_preset
from public_data.pipeline.types import SplitArtifactPaths
from public_data.pipeline.writer import build_split_artifact_paths, write_pipeline_manifest
from public_data.scripts.convert_to_coord_tokens import (
    _canonicalize_and_sort_objects_in_place,
    convert_record_to_ints,
    convert_record_to_tokens,
)
from src.common.geometry.bbox_parameterization import (
    CXCY_LOGW_LOGH_CONVERSION_VERSION,
    CXCY_LOGW_LOGH_SLOT_ORDER,
    CXCYWH_CONVERSION_VERSION,
    CXCYWH_SLOT_ORDER,
    cxcy_logw_logh_norm1000_to_xyxy_norm1000,
    cxcywh_norm1000_to_xyxy_norm1000,
    normalize_bbox_format,
    xyxy_norm1000_to_cxcy_logw_logh_bins,
    xyxy_norm1000_to_cxcywh_bins,
)
from src.coord_tokens.codec import is_coord_token, token_to_int


def _iter_split_paths(preset_dir: Path) -> list[tuple[str, Path, str]]:
    out: list[tuple[str, Path, str]] = []
    for split in ("train", "val"):
        raw_path = preset_dir / f"{split}.jsonl"
        coord_path = preset_dir / f"{split}.coord.jsonl"
        if raw_path.exists():
            out.append((split, raw_path, "raw"))
        elif coord_path.exists():
            out.append((split, coord_path, "coord"))
    if not out:
        raise FileNotFoundError(
            f"No canonical split JSONL found under {preset_dir}. Expected train.jsonl or train.coord.jsonl and optional val equivalents."
        )
    return out


def _assert_fresh_output_root(output_root: Path, splits: Iterable[str]) -> None:
    blocking: list[Path] = []
    for split in splits:
        for candidate in (
            output_root / f"{split}.jsonl",
            output_root / f"{split}.norm.jsonl",
            output_root / f"{split}.coord.jsonl",
        ):
            if candidate.exists():
                blocking.append(candidate)
    for candidate in (output_root / "images", output_root / "pipeline_manifest.json"):
        if candidate.exists():
            blocking.append(candidate)
    if not blocking:
        return
    preview = ", ".join(str(path) for path in blocking[:4])
    remainder = len(blocking) - 4
    if remainder > 0:
        preview = f"{preview}, ... (+{remainder} more)"
    raise RuntimeError(
        "bbox-format branch target is not fresh; refusing in-place overwrite. "
        f"Found existing artifacts: {preview}"
    )


def _validate_canonical_source_record(
    record: Mapping[str, object],
    *,
    source_path: Path,
    line_num: int,
) -> None:
    metadata = record.get("metadata")
    if isinstance(metadata, Mapping):
        prepared_format = metadata.get("prepared_bbox_format")
        if prepared_format not in (None, "", "xyxy"):
            raise ValueError(
                "Offline bbox-format derivation accepts canonical sources only; "
                f"found metadata.prepared_bbox_format={prepared_format!r} at {source_path}:{line_num}."
            )

    objects = record.get("objects")
    if not isinstance(objects, list):
        raise ValueError(f"Record objects must be a list at {source_path}:{line_num}.")
    for obj_idx, obj in enumerate(objects):
        if not isinstance(obj, Mapping):
            raise ValueError(
                f"Object objects[{obj_idx}] must be a mapping at {source_path}:{line_num}."
            )
        has_bbox = obj.get("bbox_2d") is not None
        has_poly = obj.get("poly") is not None
        if has_poly:
            raise ValueError(
                "Offline bbox-format derivation currently supports bbox_2d-only preset sources; "
                f"found poly geometry at {source_path}:{line_num} objects[{obj_idx}]."
            )
        if not has_bbox:
            raise ValueError(
                "Offline bbox-format derivation requires bbox_2d on every object for the first implementation surface; "
                f"missing bbox_2d at {source_path}:{line_num} objects[{obj_idx}]."
            )


def _prepared_bbox_slot_order(bbox_format: str) -> str:
    bbox_format_norm = normalize_bbox_format(bbox_format, path="bbox_format")
    if bbox_format_norm == "cxcy_logw_logh":
        return CXCY_LOGW_LOGH_SLOT_ORDER
    if bbox_format_norm == "cxcywh":
        return CXCYWH_SLOT_ORDER
    raise ValueError(f"Offline bbox-format derivation does not support {bbox_format!r}.")


def _prepared_bbox_conversion_version(bbox_format: str) -> int:
    bbox_format_norm = normalize_bbox_format(bbox_format, path="bbox_format")
    if bbox_format_norm == "cxcy_logw_logh":
        return CXCY_LOGW_LOGH_CONVERSION_VERSION
    if bbox_format_norm == "cxcywh":
        return CXCYWH_CONVERSION_VERSION
    raise ValueError(f"Offline bbox-format derivation does not support {bbox_format!r}.")


def _prepared_bbox_encoder(
    bbox_format: str,
) -> Callable[[Sequence[float | int]], list[int]]:
    bbox_format_norm = normalize_bbox_format(bbox_format, path="bbox_format")
    if bbox_format_norm == "cxcy_logw_logh":
        return xyxy_norm1000_to_cxcy_logw_logh_bins
    if bbox_format_norm == "cxcywh":
        return xyxy_norm1000_to_cxcywh_bins
    raise ValueError(f"Offline bbox-format derivation does not support {bbox_format!r}.")


def _prepared_bbox_decoder(
    bbox_format: str,
) -> Callable[[Sequence[float | int]], list[float]]:
    bbox_format_norm = normalize_bbox_format(bbox_format, path="bbox_format")
    if bbox_format_norm == "cxcy_logw_logh":
        return cxcy_logw_logh_norm1000_to_xyxy_norm1000
    if bbox_format_norm == "cxcywh":
        return cxcywh_norm1000_to_xyxy_norm1000
    raise ValueError(f"Offline bbox-format derivation does not support {bbox_format!r}.")


def _prepare_record_metadata(
    record: dict,
    *,
    source_path: Path,
    line_num: int,
    bbox_format: str,
    source_kind: str,
) -> None:
    metadata = dict(record.get("metadata") or {})
    metadata.update(
        {
            "prepared_bbox_format": bbox_format,
            "prepared_bbox_slot_order": _prepared_bbox_slot_order(bbox_format),
            "prepared_bbox_source_format": "xyxy",
            "prepared_bbox_conversion_version": _prepared_bbox_conversion_version(
                bbox_format
            ),
            "prepared_bbox_source_file": str(source_path),
            "prepared_bbox_source_record_index": int(line_num - 1),
            "prepared_bbox_source_surface": str(source_kind),
        }
    )
    record["metadata"] = metadata


def _normalize_local_image_rel_path(rel_path: str, *, source_path: Path) -> str:
    normalized = rel_path.replace("\\", "/")
    marker = "/images/"
    if normalized.startswith("images/"):
        return normalized
    if marker in normalized:
        return "images/" + normalized.split(marker, 1)[1]
    raise ValueError(
        "Offline bbox-format derivation requires images relative to a preset images/ root; "
        f"found unsupported image path {rel_path!r} at {source_path}."
    )


def _sort_bbox_only_objects_for_prepared_bbox_format(record: dict, *, bbox_format: str) -> None:
    objects = record.get("objects") or []
    if not isinstance(objects, list) or not objects:
        return
    decoder = _prepared_bbox_decoder(bbox_format)

    def _decoded_anchor(obj: Mapping[str, object]) -> tuple[float, float]:
        bbox = obj.get("bbox_2d")
        if not isinstance(bbox, list) or len(bbox) != 4:
            return (float("inf"), float("inf"))
        decoded = decoder([int(round(float(v))) for v in bbox])
        return (float(decoded[1]), float(decoded[0]))

    record["objects"] = sorted(list(objects), key=_decoded_anchor)


def _derive_split_records(
    *,
    source_path: Path,
    bbox_format: str,
    source_kind: str,
) -> tuple[list[dict], list[dict], dict[str, Path]]:
    numeric_rows: list[dict] = []
    coord_rows: list[dict] = []
    image_source_paths: dict[str, Path] = {}
    encode_bbox = _prepared_bbox_encoder(bbox_format)

    with source_path.open("r", encoding="utf-8") as fin:
        for line_num, raw_line in enumerate(fin, start=1):
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Expected object JSONL rows in {source_path}:{line_num}.")
            _validate_canonical_source_record(record, source_path=source_path, line_num=line_num)

            source_record = copy.deepcopy(record)
            if source_kind == "coord":
                objects = source_record.get("objects") or []
                for obj in objects:
                    if not isinstance(obj, dict):
                        continue
                    bbox = obj.get("bbox_2d")
                    if not isinstance(bbox, list):
                        continue
                    obj["bbox_2d"] = [
                        token_to_int(str(v)) if is_coord_token(v) else int(round(float(v)))
                        for v in bbox
                    ]

            numeric_record = convert_record_to_ints(
                source_record,
                ["bbox_2d"],
                assume_normalized=(source_kind == "coord"),
            )
            _canonicalize_and_sort_objects_in_place(numeric_record)
            images = numeric_record.get("images")
            if not isinstance(images, list) or not images or not isinstance(images[0], str):
                raise ValueError(
                    f"Derived bbox-format branch expects non-empty relative images list at {source_path}:{line_num}."
                )
            source_rel_path = images[0]
            local_rel_path = _normalize_local_image_rel_path(source_rel_path, source_path=source_path)
            source_image_path = (source_path.parent / source_rel_path).resolve()
            if not source_image_path.exists() or not source_image_path.is_file():
                raise FileNotFoundError(
                    "Offline bbox-format derivation could not resolve source image for derived branch: "
                    f"{source_image_path} (from {source_path}:{line_num} rel={source_rel_path!r})."
                )
            image_source_paths.setdefault(local_rel_path, source_image_path)
            images[0] = local_rel_path

            objects = numeric_record.get("objects") or []
            for obj in objects:
                bbox = obj.get("bbox_2d")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    raise ValueError(
                        f"Expected normalized bbox_2d quartet at {source_path}:{line_num}."
                    )
                obj["bbox_2d"] = encode_bbox(bbox)
            _sort_bbox_only_objects_for_prepared_bbox_format(
                numeric_record,
                bbox_format=bbox_format,
            )

            _prepare_record_metadata(
                numeric_record,
                source_path=source_path,
                line_num=line_num,
                bbox_format=bbox_format,
                source_kind=source_kind,
            )
            numeric_rows.append(numeric_record)

            coord_record = copy.deepcopy(numeric_record)
            convert_record_to_tokens(coord_record, ["bbox_2d"])
            coord_rows.append(coord_record)

    return numeric_rows, coord_rows, image_source_paths


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def _materialize_images_hardlinks(
    *,
    output_root: Path,
    image_source_paths: Mapping[str, Path],
) -> None:
    dst_images = output_root / "images"
    dst_images.mkdir(parents=True, exist_ok=True)
    try:
        dst_dev = dst_images.stat().st_dev
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Failed to stat derived bbox-format images directory: {dst_images}"
        ) from exc

    for rel_path, src_path in sorted(image_source_paths.items()):
        if not src_path.exists() or not src_path.is_file():
            raise FileNotFoundError(f"Missing preset image for derived branch: {src_path}")
        if src_path.stat().st_dev != dst_dev:
            raise RuntimeError(
                "Cannot hardlink bbox-format branch images across filesystems. "
                f"base={src_path} derived={dst_images}."
            )
        dst_path = output_root / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if dst_path.exists():
            continue
        os.link(src_path, dst_path)


def derive_bbox_format_branch(*, preset_dir: Path, bbox_format: str) -> Path:
    bbox_format_norm = normalize_bbox_format(bbox_format, path="bbox_format")
    if bbox_format_norm not in {"cxcy_logw_logh", "cxcywh"}:
        raise ValueError(
            "Unsupported bbox-format branch "
            f"{bbox_format!r}; expected one of ['cxcy_logw_logh', 'cxcywh']."
        )

    split_sources = _iter_split_paths(preset_dir)
    output_root = preset_dir.parent / resolve_bbox_format_preset(
        preset_dir.name,
        bbox_format_norm,
    )
    _assert_fresh_output_root(output_root, (split for split, _, _ in split_sources))

    split_artifacts: dict[str, SplitArtifactPaths] = {}
    image_source_paths: dict[str, Path] = {}
    for split, source_path, source_kind in split_sources:
        numeric_rows, coord_rows, split_image_sources = _derive_split_records(
            source_path=source_path,
            bbox_format=bbox_format_norm,
            source_kind=source_kind,
        )
        split_paths = build_split_artifact_paths(output_root, split)
        split_artifacts[split] = split_paths
        _write_jsonl(split_paths.raw, numeric_rows)
        _write_jsonl(split_paths.norm, numeric_rows)
        _write_jsonl(split_paths.coord, coord_rows)
        image_source_paths.update(split_image_sources)

    _materialize_images_hardlinks(
        output_root=output_root,
        image_source_paths=image_source_paths,
    )

    manifest_path = write_pipeline_manifest(
        preset_dir=output_root,
        dataset_id=preset_dir.parent.name,
        preset=output_root.name,
        max_objects=None,
        split_paths=split_artifacts,
        stage_stats={
            "bbox_format": {
                "source_preset_dir": str(preset_dir),
                "source_bbox_format": "xyxy",
                "derived_bbox_format": bbox_format_norm,
                "derived_bbox_slot_order": _prepared_bbox_slot_order(bbox_format_norm),
                "coord_token_norm_contract_version": 1,
                "numeric_split_contract": "norm1000_int_bbox_2d_same_lattice_as_coord",
                "bbox_format_conversion_version": _prepared_bbox_conversion_version(
                    bbox_format_norm
                ),
                "splits": {
                    split: {
                        "source": str(source_path),
                        "source_surface": str(source_kind),
                        "raw": str(split_artifacts[split].raw),
                        "norm": str(split_artifacts[split].norm),
                        "coord": str(split_artifacts[split].coord),
                    }
                    for split, source_path, source_kind in split_sources
                },
            }
        },
    )
    print(f"[bbox-format] manifest={manifest_path}")
    return output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive an offline bbox-format branch from a canonical preset."
    )
    parser.add_argument("--preset-dir", type=Path, required=True)
    parser.add_argument("--bbox-format", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    branch_root = derive_bbox_format_branch(
        preset_dir=args.preset_dir,
        bbox_format=args.bbox_format,
    )
    print(f"[bbox-format] branch_root={branch_root}")


if __name__ == "__main__":
    main()
