"""Frozen Human13 panel and image-byte/owner identity, without K-union producers."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping
from src.data.geometry import coord_bins_to_pixel_xyxy, parse_source_bbox_tokens

PANEL_SHA256 = "5c6cc95965c6dd24d7f61f09a0c56edb71eb5a9a05664fa7d26269718f741f23"


PANEL_PATH = (
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-05-static-dynamic-owner-interface-crossover/inputs/"
    "human-refined-13.geo_sorted_xy.coord.jsonl"
)


EXPECTED_IMAGE_IDENTITIES = (
    (1584, "06b9d29a50b896f1bec14a267a57016723e54e205a1d1a40088237d95ce91206"),
    (2299, "cd7199a37188c9ac6481520175866cbd78fa6fba35f4290bb0b60c78afcb2df3"),
    (2685, "84514a8aed88aba07163aa5b1be5c6e0ee75da4351496cad062d1e355a261409"),
    (4134, "60dfd1369e0efa83dfa6c7d4035f4d9d66ca6ba9a0ec6b760349d0a0e30d7b34"),
    (5001, "faaecb19a8b681495f02e18493b8ae01c96d022767f25ded19f5cbec9d95cf31"),
    (6040, "585c27309e9130400849315b4bf49d99717f700507239f2bd4dc6fece3cbe894"),
    (7511, "2843c07959515a93d2791183c998462d76b34100185e57a258d8949f112296e7"),
    (10707, "eec1e22dc3ed6ff70d35dd771e05a20024264fdecaf5a187fdad616e6d20e5d6"),
    (13348, "14f222b4cbf6d90e60eb90eb72aebc0f83cebaf150b4c5999f107bb11294fc36"),
    (13923, "c5c32b9999259b6041e92815693b975f8ee2291b29a6577d983685b6d486796f"),
    (14038, "055f28bbd181590b7a7c4844bf488d7f1be752d39916e07034c279f5f387acd2"),
    (14439, "d09e1ef4ec3bcbfbe3e16a2f9b3ff92a743e4dc061eaca4add29f59787567f73"),
    (16228, "5aade4c6e9dbdbf3bd64072e813bae02975a342b84c312efc91983ac63300d49"),
)


@dataclass(frozen=True)
class OwnerInput:
    owner_id: str
    category: str
    bbox: tuple[float, float, float, float]
    source_object_index: int


@dataclass(frozen=True)
class FrozenPanelRow:
    image_id: int
    panel_row_sha256: str
    image_sha256: str
    owners: tuple[OwnerInput, ...]


def load_frozen_panel(path: str | Path = PANEL_PATH) -> tuple[FrozenPanelRow, ...]:
    """Load only the exact canonical panel and bind each row to image bytes."""

    panel_path = Path(path).expanduser().resolve(strict=True)
    payload = panel_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != PANEL_SHA256:
        raise ValueError("panel SHA-256 does not match the frozen Human-13 panel")
    raw_lines = tuple(line for line in payload.splitlines() if line.strip())
    if len(raw_lines) != len(EXPECTED_IMAGE_IDENTITIES):
        raise ValueError("frozen Human-13 panel must contain exactly thirteen rows")
    expected_images = dict(EXPECTED_IMAGE_IDENTITIES)
    rows: list[FrozenPanelRow] = []
    for row_index, raw_line in enumerate(raw_lines):
        document = json.loads(raw_line)
        if not isinstance(document, Mapping):
            raise ValueError(f"panel row {row_index} must be an object")
        image_id = int(document.get("image_id"))
        expected_image_id = EXPECTED_IMAGE_IDENTITIES[row_index][0]
        if image_id != expected_image_id:
            raise ValueError("panel row identities are not in canonical Human-13 order")
        image_refs = document.get("images")
        if not isinstance(image_refs, list) or len(image_refs) != 1:
            raise ValueError(f"panel row {image_id} must declare exactly one image")
        image_path = (panel_path.parent / str(image_refs[0])).resolve(strict=True)
        image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if image_sha256 != expected_images[image_id]:
            raise ValueError(f"image-content SHA-256 mismatch for image {image_id}")
        width = document.get("width")
        height = document.get("height")
        objects = document.get("objects")
        if not isinstance(width, int) or not isinstance(height, int):
            raise ValueError(f"panel row {image_id} has invalid dimensions")
        if not isinstance(objects, list):
            raise ValueError(f"panel row {image_id} has invalid objects")
        owners: list[OwnerInput] = []
        for object_index, obj in enumerate(objects):
            if not isinstance(obj, Mapping):
                raise ValueError(
                    f"panel row {image_id} object {object_index} is invalid"
                )
            bins = parse_source_bbox_tokens(
                obj.get("bbox_2d"), field=f"panel[{image_id}].objects[{object_index}]"
            )
            bbox = coord_bins_to_pixel_xyxy(
                bins,
                image_width=width,
                image_height=height,
                field=f"panel[{image_id}].objects[{object_index}]",
            )
            category = obj.get("category_name", obj.get("desc"))
            if not isinstance(category, str) or not category:
                raise ValueError(
                    f"panel row {image_id} object {object_index} has no category"
                )
            owners.append(
                OwnerInput(
                    owner_id=f"gt:{image_id}:{object_index}",
                    category=category,
                    bbox=tuple(float(item) for item in bbox),
                    source_object_index=object_index,
                )
            )
        rows.append(
            FrozenPanelRow(
                image_id=image_id,
                panel_row_sha256=hashlib.sha256(raw_line).hexdigest(),
                image_sha256=image_sha256,
                owners=tuple(owners),
            )
        )
    if sum(len(row.owners) for row in rows) != 392:
        raise ValueError("frozen Human-13 panel must contain exactly 392 owners")
    return tuple(rows)

