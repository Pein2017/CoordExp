"""Coverage-ledger preflight artifact writers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from src.training.coverage_ledger.sidecars import (
    CoverageLedgerObjectEntry,
    CoverageLedgerSidecar,
)
from src.training.coverage_ledger.visual_regions import VisualTokenRegion


SCHEMA_VERSION = "coverage_ledger_preflight_artifacts_v0"
REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class CoverageLedgerOverlayCandidate:
    sample_id: str
    row_index: int
    object_entry: CoverageLedgerObjectEntry
    visual_region: VisualTokenRegion
    render_image_path: Path


@dataclass(frozen=True, slots=True)
class CoverageLedgerPreflightArtifactInputs:
    output_root: Path
    source_jsonl_path: Path
    dataset_id: str
    split: str
    selected_row_indices: Sequence[int]
    selection_seed: int
    selection_algorithm: str
    template_id: str
    object_field_order: str
    tokenizer_id: str
    model_id: str
    processor_do_resize: bool
    image_grid_metadata_version: str
    sidecars: Sequence[CoverageLedgerSidecar]
    visual_regions_by_sample_id: Mapping[str, Sequence[VisualTokenRegion]]
    overlay_candidates: Sequence[CoverageLedgerOverlayCandidate]
    visual_grid_shape_by_sample_id: Mapping[str, tuple[int, int]]
    expected_sample_count: int = 128
    expected_overlay_count: int = 16


@dataclass(frozen=True, slots=True)
class CoverageLedgerPreflightArtifactResult:
    ledger_root: Path
    selected_samples_path: Path
    alignment_debug_path: Path
    overlay_index_path: Path
    overlay_paths: tuple[Path, ...]


def write_coverage_ledger_preflight_artifacts(
    inputs: CoverageLedgerPreflightArtifactInputs,
) -> CoverageLedgerPreflightArtifactResult:
    """Write the strict Task-10 ledger artifact tree."""

    sidecars = tuple(inputs.sidecars)
    overlay_candidates = tuple(inputs.overlay_candidates)
    if len(sidecars) != int(inputs.expected_sample_count):
        raise ValueError(
            "coverage ledger preflight requires exactly "
            f"{inputs.expected_sample_count} selected samples; got {len(sidecars)}"
        )
    if len(overlay_candidates) != int(inputs.expected_overlay_count):
        raise ValueError(
            "coverage ledger preflight requires exactly "
            f"{inputs.expected_overlay_count} overlay candidates; "
            f"got {len(overlay_candidates)}"
        )

    selected_row_indices = tuple(int(index) for index in inputs.selected_row_indices)
    if len(selected_row_indices) != len(sidecars):
        raise ValueError(
            "selected_row_indices length must match sidecars length; "
            f"got {len(selected_row_indices)} indices and {len(sidecars)} sidecars"
        )

    ledger_root = Path(inputs.output_root) / "ledger"
    overlays_root = ledger_root / "overlays"
    _reject_existing_nonempty_tree(ledger_root, label="ledger")
    _reject_existing_nonempty_tree(overlays_root, label="ledger/overlays")
    overlays_root.mkdir(parents=True, exist_ok=True)

    selected_samples_path = ledger_root / "selected_samples.json"
    alignment_debug_path = ledger_root / "alignment_debug.jsonl"
    overlay_index_path = overlays_root / "index.json"

    selected_samples = _selected_samples_payload(inputs, sidecars, selected_row_indices)
    selected_samples_path.write_text(
        json.dumps(selected_samples, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    alignment_lines = [
        json.dumps(
            _alignment_debug_record(
                sidecar,
                row_index=row_index,
                regions=inputs.visual_regions_by_sample_id[sidecar.sample_id],
            ),
            sort_keys=True,
        )
        for row_index, sidecar in zip(selected_row_indices, sidecars, strict=True)
    ]
    alignment_debug_path.write_text("\n".join(alignment_lines) + "\n", encoding="utf-8")

    overlay_paths, overlay_index = _write_overlays(
        overlay_candidates,
        sidecars_by_sample_id={sidecar.sample_id: sidecar for sidecar in sidecars},
        grid_shapes_by_sample_id=inputs.visual_grid_shape_by_sample_id,
        overlays_root=overlays_root,
        ledger_root=ledger_root,
        template_id=inputs.template_id,
    )
    overlay_index_path.write_text(
        json.dumps(overlay_index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return CoverageLedgerPreflightArtifactResult(
        ledger_root=ledger_root,
        selected_samples_path=selected_samples_path,
        alignment_debug_path=alignment_debug_path,
        overlay_index_path=overlay_index_path,
        overlay_paths=tuple(overlay_paths),
    )


def _selected_samples_payload(
    inputs: CoverageLedgerPreflightArtifactInputs,
    sidecars: tuple[CoverageLedgerSidecar, ...],
    selected_row_indices: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source_jsonl_path": str(Path(inputs.source_jsonl_path)),
        "source_jsonl_repo_path": _repo_relative_path(Path(inputs.source_jsonl_path)),
        "source_jsonl_resolved_path": str(Path(inputs.source_jsonl_path).resolve()),
        "source_jsonl_sha256": _sha256_file(Path(inputs.source_jsonl_path)),
        "dataset_id": inputs.dataset_id,
        "split": inputs.split,
        "selected_row_indices": list(selected_row_indices),
        "selected_sample_ids": [sidecar.sample_id for sidecar in sidecars],
        "selection_seed": int(inputs.selection_seed),
        "selection_algorithm": inputs.selection_algorithm,
        "template_id": inputs.template_id,
        "object_field_order": inputs.object_field_order,
        "tokenizer_id": inputs.tokenizer_id,
        "model_id": inputs.model_id,
        "processor_do_resize": bool(inputs.processor_do_resize),
        "image_grid_metadata_version": inputs.image_grid_metadata_version,
        "samples": [
            {
                "row_index": int(row_index),
                "sample_id": sidecar.sample_id,
                "object_count": len(sidecar.object_entries),
                "image_identity": sidecar.image_identity,
                "processed_width": int(sidecar.processed_width),
                "processed_height": int(sidecar.processed_height),
                "image_grid_thw": list(sidecar.image_grid_thw),
            }
            for row_index, sidecar in zip(selected_row_indices, sidecars, strict=True)
        ],
    }


def _alignment_debug_record(
    sidecar: CoverageLedgerSidecar,
    *,
    row_index: int,
    regions: Sequence[VisualTokenRegion],
) -> dict[str, Any]:
    frozen_regions = tuple(regions)
    if len(frozen_regions) != len(sidecar.object_entries):
        raise ValueError(
            "visual region count must match object entry count for "
            f"sample_id={sidecar.sample_id!r}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "row_index": int(row_index),
        "sample_id": sidecar.sample_id,
        "prompt_end_position": int(sidecar.prompt_end_position),
        "failure_status": "ok",
        "failure_reason": None,
        "objects": [
            _alignment_object_record(entry, region)
            for entry, region in zip(
                sidecar.object_entries,
                frozen_regions,
                strict=True,
            )
        ],
    }


def _alignment_object_record(
    entry: CoverageLedgerObjectEntry,
    region: VisualTokenRegion,
) -> dict[str, Any]:
    return {
        "object_instance_id": entry.object_instance_id,
        "source_object_index": int(entry.source_object_index),
        "emitted_order_index": int(entry.emitted_order_index),
        "image_index": int(entry.image_index),
        "object_ref_end_position": int(entry.object_ref_end_position),
        "box_start_position": int(entry.box_start_position),
        "coord_label_positions": list(entry.coord_label_positions),
        "box_end_position": int(entry.box_end_position),
        "bbox_norm1000_xyxy": list(entry.bbox_norm1000_xyxy),
        "mapped_visual_cells": _visual_region_payload(region),
    }


def _visual_region_payload(region: VisualTokenRegion) -> dict[str, Any]:
    return {
        "row_start": int(region.row_start),
        "row_end": int(region.row_end),
        "col_start": int(region.col_start),
        "col_end": int(region.col_end),
        "flattened_indices": list(region.flattened_indices),
    }


def _write_overlays(
    overlay_candidates: tuple[CoverageLedgerOverlayCandidate, ...],
    *,
    sidecars_by_sample_id: Mapping[str, CoverageLedgerSidecar],
    grid_shapes_by_sample_id: Mapping[str, tuple[int, int]],
    overlays_root: Path,
    ledger_root: Path,
    template_id: str,
) -> tuple[list[Path], dict[str, Any]]:
    overlay_paths: list[Path] = []
    index_entries: list[dict[str, Any]] = []
    for overlay_index, candidate in enumerate(overlay_candidates):
        sidecar = sidecars_by_sample_id[candidate.sample_id]
        grid_rows, grid_cols = grid_shapes_by_sample_id[candidate.sample_id]
        overlay_path = overlays_root / f"overlay_{overlay_index:04d}.png"
        _render_overlay(
            image_path=candidate.render_image_path,
            output_path=overlay_path,
            sidecar=sidecar,
            object_entry=candidate.object_entry,
            visual_region=candidate.visual_region,
            grid_rows=int(grid_rows),
            grid_cols=int(grid_cols),
            sample_id=candidate.sample_id,
            template_id=template_id,
        )
        overlay_paths.append(overlay_path)
        index_entries.append(
            {
                "path": overlay_path.relative_to(ledger_root).as_posix(),
                "sample_id": candidate.sample_id,
                "row_index": int(candidate.row_index),
                "object_instance_id": candidate.object_entry.object_instance_id,
                "source_object_index": int(candidate.object_entry.source_object_index),
                "emitted_order_index": int(candidate.object_entry.emitted_order_index),
                "template_id": template_id,
            }
        )
    return overlay_paths, {
        "schema_version": SCHEMA_VERSION,
        "template_id": template_id,
        "overlays": index_entries,
    }


def _render_overlay(
    *,
    image_path: Path,
    output_path: Path,
    sidecar: CoverageLedgerSidecar,
    object_entry: CoverageLedgerObjectEntry,
    visual_region: VisualTokenRegion,
    grid_rows: int,
    grid_cols: int,
    sample_id: str,
    template_id: str,
) -> None:
    if grid_rows <= 0 or grid_cols <= 0:
        raise ValueError("visual grid shape must be positive")
    base = Image.open(image_path).convert("RGBA")
    if base.size != (sidecar.processed_width, sidecar.processed_height):
        base = base.resize((sidecar.processed_width, sidecar.processed_height))
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    width, height = base.size
    cell_w = width / float(grid_cols)
    cell_h = height / float(grid_rows)
    for row in range(grid_rows + 1):
        y = round(row * cell_h)
        draw.line([(0, y), (width, y)], fill=(30, 144, 255, 90), width=1)
    for col in range(grid_cols + 1):
        x = round(col * cell_w)
        draw.line([(x, 0), (x, height)], fill=(30, 144, 255, 90), width=1)

    for row in range(visual_region.row_start, visual_region.row_end):
        for col in range(visual_region.col_start, visual_region.col_end):
            draw.rectangle(
                [
                    round(col * cell_w),
                    round(row * cell_h),
                    round((col + 1) * cell_w),
                    round((row + 1) * cell_h),
                ],
                fill=(30, 144, 255, 70),
                outline=(30, 144, 255, 180),
                width=1,
            )

    x1, y1, x2, y2 = object_entry.bbox_norm1000_xyxy
    bbox = [
        round(x1 / 1000.0 * width),
        round(y1 / 1000.0 * height),
        round(x2 / 1000.0 * width),
        round(y2 / 1000.0 * height),
    ]
    draw.rectangle(bbox, outline=(255, 48, 48, 255), width=3)
    label = (
        f"{sample_id} obj={object_entry.emitted_order_index} "
        f"src={object_entry.source_object_index} {template_id}"
    )
    text_box = [4, 4, min(width - 4, 4 + 7 * len(label)), 24]
    draw.rectangle(text_box, fill=(0, 0, 0, 170))
    draw.text((8, 8), label, fill=(255, 255, 255, 255))

    Image.alpha_composite(base, overlay).convert("RGB").save(output_path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative_path(path: Path) -> str | None:
    resolved = path.expanduser().resolve(strict=False)
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return None


def _reject_existing_nonempty_tree(path: Path, *, label: str) -> None:
    if not path.exists():
        return
    if not path.is_dir():
        raise ValueError(f"coverage ledger preflight {label} exists and is not a directory")
    if any(path.iterdir()):
        raise ValueError(
            f"coverage ledger preflight refuses to overwrite non-empty {label}: {path}"
        )


__all__ = [
    "CoverageLedgerOverlayCandidate",
    "CoverageLedgerPreflightArtifactInputs",
    "CoverageLedgerPreflightArtifactResult",
    "write_coverage_ledger_preflight_artifacts",
]
