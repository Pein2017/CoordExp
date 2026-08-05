#!/usr/bin/env python3
"""CPU preparation for the sorted full-canvas visual-token-budget intervention.

This unit is a *scoring-only successor* to
``2026-08-03-sorted-owner-accessibility-phenotype-census``.  It asks one
question: does presenting the *same raw optical information* to Qwen at roughly
twice the merged visual-token density change which owners carry tested
localization support?

What changes
------------
Only the visual presentation of each image:

* the current pool is ``max_pixels = 32*32*1024`` (~1024 merged visual tokens);
* the treatment pool is ``max_pixels = 32*32*2048`` (~2x merged visual tokens,
  ~sqrt(2) linear scale), built **directly from raw COCO** with the same
  factor-32 maximize-under-cap geometry and the same PIL ``RGB -> LANCZOS ->
  save(default JPEG)`` materialization that produced the current pool.

Everything else is frozen: model/adapter/config identity, ``do_resize=False``,
the 0..999 normalized coordinate vocabulary, the sealed candidate bank and its
1x pixel-frame geometry, category query suffixes, native generated-prefix token
IDs and owner assignments.  Consequently the claim this unit can support is
"denser patch/token sampling over the same raw optical information", never "new
visual acuity".

What this script produces
-------------------------
A compact **sealed intervention overlay** -- not a duplicated plan.  The overlay
binds the predecessor plan receipt digest, the predecessor run root, the
intervention unit/arm identity, ``max_pixels``, per-image raw/current/treatment
media digests and dimensions, the treatment ``image_grid_thw`` and prompt token
IDs obtained from the **production frontend planning/prompt path** (never an
estimator), and a frozen query-group selection.

Before any of that is written the script proves, for all twelve images, that the
*current* 1024-pool media is byte-exactly reproducible from raw under this same
algorithm -- same dimensions and same ``executed_media_sha256``.  If it is not,
the treatment pool cannot be claimed to differ from the current pool only in
token density, and the script fails closed.

Nothing under ``public_data`` or the predecessor run root is ever modified.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import build_sorted_owner_accessibility_census_plan as planner  # noqa: E402

BASE_UNIT_ID = planner.UNIT_ID
BASE_PLAN_SCHEMA_VERSION = planner.PLAN_SCHEMA_VERSION

INTERVENTION_UNIT_ID = "2026-08-03-sorted-full-canvas-visual-token-budget-intervention"
OVERLAY_SCHEMA_VERSION = "sorted-full-canvas-visual-token-budget-intervention-overlay.v1"
RECEIPT_SCHEMA_VERSION = "sorted-full-canvas-visual-token-budget-intervention-prepare-receipt.v1"

#: The one treatment arm this unit defines.  Every capture, row and analysis
#: binds it, so two arms can never be pooled by accident.
ARM_ID = "full_canvas_max_pixels_32x32x2048"

#: The frozen 1x pool this unit is measured *against*.  Named as its own arm so
#: a baseline capture is never silently read as a treatment capture.
BASELINE_ARM_ID = "full_canvas_max_pixels_32x32x1024"

IMAGE_FACTOR = 32
BASELINE_MAX_PIXELS = 32 * 32 * 1024
TREATMENT_MAX_PIXELS = 32 * 32 * 2048
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_RATIO = 200

#: Restricted stratum.  Image 4134 is the confirmation image whose persistent
#: owners are far too many contexts wide to score exhaustively at 2x cost, so it
#: is captured on a declared restricted context union and reported descriptively
#: only.
RESTRICTED_IMAGE_ID = "4134"

DEFAULT_RAW_ROOT = Path("/data/CoordExp/public_data/coco/raw")
DEFAULT_INFER_CONFIG = (
    REPO_ROOT
    / "configs/coordexp_swift/infer"
    / "qwen3_vl_2b_desc_first_geo_sorted_step4887_human_refined12_hf_fp32_rp1p0.yaml"
)

OVERLAY_NAME = "overlay.json"
TREATMENT_PANEL_NAME = "treatment-panel.jsonl"
SELECTION_NAME = "query-group-selection.jsonl"
RECEIPT_NAME = "prepare-receipt.json"
MEDIA_DIRNAME = "media"

PERSISTENT_DISPOSITION = "persistent_no_tested_localization_support"
RESOLVED_DISPOSITION = "resolved_tested_localization_support"

sha256_json = planner.sha256_json
canonical_json_bytes = planner.canonical_json_bytes
sha256_file = planner.sha256_file

EXECUTED_SOURCE_SHA256 = planner.sha256_file(Path(__file__).resolve())


class PrepareContractError(RuntimeError):
    """Raised when a preparation precondition or invariant fails."""


# ---------------------------------------------------------------------------
# Aligned-dimension helper
# ---------------------------------------------------------------------------
#
# Provenance: this is a minimal, behaviour-identical copy of the pure
# dimension-solving part of ``src/datasets/preprocessors/resize.py``
# (``_round_by_factor`` / ``_ceil_by_factor`` / ``_floor_by_factor`` /
# ``_aligned_candidates`` / ``_maximize_aligned_size`` / ``smart_resize``), the
# module that produced the current ``public_data/coco/rescale_32_1024_bbox``
# pool.  It is copied rather than imported on purpose:
#
# * this worktree has no ``src/datasets`` package at all, so
#   ``public_data/scripts/rescale_jsonl.py`` cannot even be imported here; and
# * importing it out of another worktree at runtime would make this probe's
#   geometry depend on a checkout nobody pins.
#
# Only the *pure* dimension solver is copied.  Image and geometry
# materialization below re-implements nothing: it uses the same PIL calls in the
# same order.  ``test_prepare_sorted_full_canvas_token_budget_intervention.py``
# pins this copy against the twelve real current-pool dimensions, which is the
# behavioural equality that matters.


def _ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def _aligned_candidates(value: float, factor: int) -> set[int]:
    value_int = max(1, int(round(value)))
    floor_value = max(factor, _floor_by_factor(value_int, factor))
    ceil_value = max(factor, _ceil_by_factor(value_int, factor))
    return {
        candidate
        for candidate in {
            floor_value - factor,
            floor_value,
            ceil_value,
            ceil_value + factor,
        }
        if candidate >= factor
    }


def _maximize_aligned_size(
    *, height: int, width: int, factor: int, max_pixels: int
) -> tuple[int, int]:
    area = height * width
    scale = math.sqrt(max_pixels / area)
    ideal_h = height * scale
    ideal_w = width * scale
    aspect = float(width) / float(height)

    candidates: set[tuple[int, int]] = set()

    def _add_candidate(candidate_h: int, candidate_w: int) -> None:
        if candidate_h < factor or candidate_w < factor:
            return
        if candidate_h % factor != 0 or candidate_w % factor != 0:
            return
        if candidate_h * candidate_w > max_pixels:
            return
        candidates.add((candidate_h, candidate_w))

    for candidate_h in _aligned_candidates(ideal_h, factor):
        max_w_budget = max(factor, _floor_by_factor(max_pixels // candidate_h, factor))
        for candidate_w in _aligned_candidates(candidate_h * aspect, factor) | {
            max_w_budget,
            max_w_budget - factor,
        }:
            bounded_w = min(max_w_budget, candidate_w)
            if bounded_w >= factor:
                _add_candidate(candidate_h, bounded_w)

    for candidate_w in _aligned_candidates(ideal_w, factor):
        max_h_budget = max(factor, _floor_by_factor(max_pixels // candidate_w, factor))
        for candidate_h in _aligned_candidates(candidate_w / aspect, factor) | {
            max_h_budget,
            max_h_budget - factor,
        }:
            bounded_h = min(max_h_budget, candidate_h)
            if bounded_h >= factor:
                _add_candidate(bounded_h, candidate_w)

    floor_h = max(factor, _floor_by_factor(int(ideal_h), factor))
    floor_w = max(factor, _floor_by_factor(int(ideal_w), factor))
    _add_candidate(floor_h, floor_w)

    if not candidates:
        return factor, factor

    def _score(size: tuple[int, int]) -> tuple[float, int, float]:
        candidate_h, candidate_w = size
        ratio_error = abs(math.log((candidate_w / candidate_h) / aspect))
        ideal_error = abs(candidate_h - ideal_h) + abs(candidate_w - ideal_w)
        return (ratio_error, -(candidate_h * candidate_w), ideal_error)

    return min(candidates, key=_score)


def aligned_dimensions(
    *,
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int,
    max_ratio: int = MAX_RATIO,
) -> tuple[int, int]:
    """Return ``(height, width)``: the maximize-under-cap factor-aligned size."""

    height = int(height)
    width = int(width)
    factor = int(factor)
    min_pixels = int(min_pixels)
    max_pixels = int(max_pixels)

    if height <= 0 or width <= 0:
        raise PrepareContractError(f"height/width must be positive, got {(height, width)}")
    if factor <= 0:
        raise PrepareContractError(f"factor must be positive, got {factor}")
    if max_pixels <= 0 or min_pixels <= 0:
        raise PrepareContractError("pixel budgets must be positive")
    if max_pixels < min_pixels:
        raise PrepareContractError("max_pixels must be >= min_pixels")
    if max(height, width) / min(height, width) > max_ratio:
        raise PrepareContractError(f"absolute aspect ratio must be smaller than {max_ratio}")

    return _maximize_aligned_size(
        height=height, width=width, factor=factor, max_pixels=max_pixels
    )


def merged_visual_token_count(*, width: int, height: int, factor: int = IMAGE_FACTOR) -> int:
    """Merged visual tokens for a factor-aligned canvas (one token per factor block)."""

    if width % factor or height % factor:
        raise PrepareContractError(
            f"canvas {width}x{height} is not aligned to factor {factor}"
        )
    return (width // factor) * (height // factor)


# ---------------------------------------------------------------------------
# Media materialization (identical call order to the offline pool builder)
# ---------------------------------------------------------------------------


def rgb_pixel_sha256(image: Any) -> str:
    """The executed-media digest: canonical RGB8 dimensions and pixels.

    Mirrors ``src.qwen.images.rgb_image_sha256``.  Reproduced here so the CPU
    preparation does not need the inference stack merely to hash a file.
    """

    rgb = image if image.mode == "RGB" else image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(b"coordexp-rgb8-pixels-v1\0")
    digest.update(int(rgb.width).to_bytes(8, "big", signed=False))
    digest.update(int(rgb.height).to_bytes(8, "big", signed=False))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def render_pool_jpeg_bytes(raw_path: Path, *, width: int, height: int) -> bytes:
    """``open -> convert('RGB') -> resize(LANCZOS) -> save(default JPEG)``."""

    from PIL import Image

    with Image.open(raw_path) as handle:
        resized = handle.convert("RGB").resize(
            (int(width), int(height)), Image.Resampling.LANCZOS
        )
        buffer = io.BytesIO()
        resized.save(buffer, format="JPEG")
        return buffer.getvalue()


def decoded_media_identity(payload: bytes) -> dict[str, Any]:
    """Dimensions and executed-media digest of encoded JPEG bytes."""

    from PIL import Image

    with Image.open(io.BytesIO(payload)) as handle:
        rgb = handle.convert("RGB")
        return {
            "width": int(rgb.width),
            "height": int(rgb.height),
            "executed_media_sha256": rgb_pixel_sha256(rgb),
            "file_sha256": hashlib.sha256(payload).hexdigest(),
            "byte_count": len(payload),
        }


# ---------------------------------------------------------------------------
# Panel geometry
# ---------------------------------------------------------------------------


def _scale_and_clamp_box(
    box: Sequence[Any], *, sx: float, sy: float, width: int, height: int
) -> list[Any]:
    """Scale a pixel ``bbox_2d``/``poly`` and clamp it into the new canvas.

    Coordinate-token geometry (``"<|coord_206|>"`` and friends) is normalized
    0..999 and therefore resolution independent: it is passed through
    untouched, because rescaling it would move the box.
    """

    if any(isinstance(value, str) for value in box):
        if not all(isinstance(value, str) for value in box):
            raise PrepareContractError(
                "geometry mixes coordinate tokens and pixel values; refusing to scale it"
            )
        return list(box)

    scaled: list[float] = []
    for index, value in enumerate(box):
        factor = sx if index % 2 == 0 else sy
        scaled.append(float(value) * float(factor))
    clamped: list[int] = []
    for index, value in enumerate(scaled):
        bound = int(width) - 1 if index % 2 == 0 else int(height) - 1
        clamped.append(max(0, min(bound, int(round(value)))))

    if len(clamped) == 4:
        x1, y1, x2, y2 = clamped
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        if x2 <= x1:
            if x2 < width - 1:
                x2 = min(width - 1, x1 + 1)
            elif x1 > 0:
                x1 = max(0, x2 - 1)
        if y2 <= y1:
            if y2 < height - 1:
                y2 = min(height - 1, y1 + 1)
            elif y1 > 0:
                y1 = max(0, y2 - 1)
        return [x1, y1, x2, y2]
    return clamped


def derive_treatment_panel_row(
    row: Mapping[str, Any],
    *,
    treatment_width: int,
    treatment_height: int,
    relative_image_path: str,
) -> dict[str, Any]:
    """One treatment panel row: new canvas, new image path, scaled geometry."""

    source_width = int(row["width"])
    source_height = int(row["height"])
    if source_width <= 0 or source_height <= 0:
        raise PrepareContractError("panel row declares a non-positive canvas")
    sx = float(treatment_width) / float(source_width)
    sy = float(treatment_height) / float(source_height)

    objects: list[dict[str, Any]] = []
    for obj in row.get("objects") or ():
        updated = dict(obj)
        for key in ("bbox_2d", "poly"):
            if updated.get(key) is not None:
                updated[key] = _scale_and_clamp_box(
                    updated[key],
                    sx=sx,
                    sy=sy,
                    width=int(treatment_width),
                    height=int(treatment_height),
                )
        if updated.get("line") is not None or updated.get("line_points") is not None:
            raise PrepareContractError("line geometry is not supported by this probe")
        objects.append(updated)

    derived = dict(row)
    derived["images"] = [relative_image_path]
    derived["width"] = int(treatment_width)
    derived["height"] = int(treatment_height)
    derived["objects"] = objects
    return derived


# ---------------------------------------------------------------------------
# Sealed inputs
# ---------------------------------------------------------------------------


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not Path(path).is_file():
        raise PrepareContractError(f"{label} is missing at {path}")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not Path(path).is_file():
        raise PrepareContractError(f"{label} is missing at {path}")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise PrepareContractError(f"{label} line {index} is not valid JSON") from exc
    return rows


@dataclass(frozen=True)
class BasePlan:
    """The sealed predecessor plan, re-verified file by file."""

    plan_dir: Path
    receipt: dict[str, Any]
    images: dict[str, dict[str, Any]]
    owners: dict[str, dict[str, Any]]
    contexts: dict[str, dict[str, Any]]
    query_groups: dict[str, dict[str, Any]]
    native_sidecars: list[dict[str, Any]]

    @property
    def receipt_content_sha256(self) -> str:
        return str(self.receipt["receipt_content_sha256"])

    @property
    def capture_rules_sha256(self) -> str:
        return str(self.receipt["capture_rules_sha256"])


def load_base_plan(plan_dir: Path) -> BasePlan:
    """Load and re-verify the sealed predecessor plan, including every digest."""

    plan_dir = Path(plan_dir)
    receipt = _read_json(plan_dir / "receipt.json", "predecessor plan receipt")
    if receipt.get("schema_version") != BASE_PLAN_SCHEMA_VERSION:
        raise PrepareContractError(
            "predecessor plan receipt schema_version does not match the census plan schema"
        )
    if receipt.get("unit_id") != BASE_UNIT_ID:
        raise PrepareContractError("predecessor plan receipt belongs to another unit")
    reconstructed = sha256_json(
        {key: value for key, value in receipt.items() if key != "receipt_content_sha256"}
    )
    if reconstructed != receipt.get("receipt_content_sha256"):
        raise PrepareContractError("predecessor plan receipt does not reconstruct its own digest")

    declared = receipt.get("output_file_digests") or {}
    for name in planner.PLAN_FILE_NAMES:
        path = plan_dir / name
        if not path.is_file():
            raise PrepareContractError(f"predecessor plan is missing declared file {name!r}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != declared.get(name):
            raise PrepareContractError(
                f"predecessor plan file {name!r} does not match its sealed digest"
            )

    def index(name: str, key: str) -> dict[str, dict[str, Any]]:
        rows = _read_jsonl(plan_dir / name, name)
        indexed: dict[str, dict[str, Any]] = {}
        for row in rows:
            row_key = str(row[key])
            if row_key in indexed:
                raise PrepareContractError(f"{name} carries duplicate {key} {row_key!r}")
            indexed[row_key] = row
        return indexed

    return BasePlan(
        plan_dir=plan_dir,
        receipt=receipt,
        images=index("image-registry.jsonl", "image_id"),
        owners=index("owner-registry.jsonl", "gt_owner_id"),
        contexts=index("context-registry.jsonl", "context_id"),
        query_groups=index("query-group-registry.jsonl", "query_group_id"),
        native_sidecars=_read_jsonl(
            plan_dir / "native-sidecar-registry.jsonl", "native-sidecar-registry.jsonl"
        ),
    )


def load_panel(panel_jsonl: Path) -> dict[str, dict[str, Any]]:
    """The human-refined-12 panel, indexed by ``image_id``."""

    rows = _read_jsonl(Path(panel_jsonl), "human-refined-12 panel")
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        for key in ("image_id", "file_name", "width", "height", "images", "objects"):
            if key not in row:
                raise PrepareContractError(f"panel row is missing required field {key!r}")
        image_id = str(row["image_id"])
        if image_id in indexed:
            raise PrepareContractError(f"panel carries duplicate image_id {image_id!r}")
        indexed[image_id] = row
    return indexed


def validate_panel_against_plan(panel: Mapping[str, Any], plan: BasePlan) -> None:
    """The panel and the plan must describe the same twelve images and canvases."""

    if set(panel) != set(plan.images):
        raise PrepareContractError(
            "panel image set does not match the predecessor plan image registry"
        )
    for image_id, row in panel.items():
        registry = plan.images[image_id]
        if int(row["width"]) != int(registry["image_width"]) or int(row["height"]) != int(
            registry["image_height"]
        ):
            raise PrepareContractError(
                f"panel image {image_id!r} canvas does not match the plan image registry"
            )
        if str(row["file_name"]) != str(registry["file_name"]):
            raise PrepareContractError(
                f"panel image {image_id!r} file_name does not match the plan image registry"
            )


# ---------------------------------------------------------------------------
# Production frontend prompt/grid resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedPrompt:
    """One image's production prompt identity for one panel."""

    image_id: str
    image_path: str
    declared_width: int
    declared_height: int
    decoded_width: int
    decoded_height: int
    image_grid_thw: tuple[int, int, int]
    merged_visual_tokens: int
    image_content_sha256: str
    logical_transform_id: str
    prompt_token_ids: list[int]

    @property
    def prompt_token_ids_sha256(self) -> str:
        return sha256_json(self.prompt_token_ids)


def resolve_panel_prompts(
    panel_jsonl: Path, *, infer_config: Path, image_ids: Sequence[str] | None = None
) -> dict[str, ResolvedPrompt]:
    """Resolve prompts through the production frontend, never an estimator.

    Uses exactly the census scorer's resolution path -- resolved infer config ->
    ``assemble_frontend`` -> ``plan_image_batch`` -> ``build_prompt_record`` --
    but against a caller-supplied panel, so the same code answers both "what is
    the current prompt?" and "what is the treatment prompt?".

    CPU only: this opens the processor and tokenizer, never the model.
    """

    from src.config.fingerprint import sha256_json as config_sha256_json
    from src.config.inference import load_infer_config
    from src.data import load_raw_examples
    from src.inference.image_plan import plan_image_batch
    from src.inference.pipeline import _processor_config, _template_config
    from src.inference.prompt import build_prompt_record
    from src.inference.runtime import assemble_frontend

    resolved = load_infer_config(Path(infer_config).expanduser().resolve(strict=True))
    config = resolved.config
    if config.backend.type != "hf":
        raise PrepareContractError("this probe requires backend.type: hf")
    processor_config = _processor_config(config)
    if getattr(processor_config, "do_resize", False):
        raise PrepareContractError(
            "processor do_resize must stay False; the pool geometry is offline, and a "
            "processor-side resize would silently reinterpret the treatment canvas"
        )

    frontend = assemble_frontend(
        config,
        generation_config_fingerprint=config_sha256_json(
            config.generation.model_dump(mode="json")
        ),
    )
    raw_rows = load_raw_examples(Path(panel_jsonl).expanduser().resolve(strict=True))

    wanted = None if image_ids is None else {str(value) for value in image_ids}
    out: dict[str, ResolvedPrompt] = {}
    for row_index, raw in enumerate(raw_rows):
        source = raw.metadata.get("source")
        if not isinstance(source, Mapping) or source.get("image_id") is None:
            raise PrepareContractError(
                "panel row carries no metadata.source.image_id; the production loader "
                "could not bind it to a census image"
            )
        image_id = str(source["image_id"])
        if wanted is not None and image_id not in wanted:
            continue
        image_plan = plan_image_batch(
            [raw],
            components=frontend.qwen,
            processor_config=processor_config,
            row_indices=[row_index],
        ).rows[0]
        prompt_record = build_prompt_record(
            raw,
            _template_config(config),
            processor=frontend.qwen.processor,
            row_index=row_index,
            merged_visual_tokens=image_plan.merged_visual_tokens,
        )
        grid = tuple(int(value) for value in image_plan.expected_image_grid_thw)
        if len(grid) != 3:
            raise PrepareContractError(f"image {image_id!r} produced a non-3D image grid")
        out[image_id] = ResolvedPrompt(
            image_id=image_id,
            image_path=str(image_plan.image_path),
            declared_width=int(image_plan.declared_width),
            declared_height=int(image_plan.declared_height),
            decoded_width=int(image_plan.decoded_width),
            decoded_height=int(image_plan.decoded_height),
            image_grid_thw=(grid[0], grid[1], grid[2]),
            merged_visual_tokens=int(image_plan.merged_visual_tokens),
            image_content_sha256=str(image_plan.image_content_sha256),
            logical_transform_id=str(image_plan.logical_transform_id),
            prompt_token_ids=[
                int(value) for value in prompt_record.expected_executed_prompt_token_ids
            ],
        )
    if wanted is not None and set(out) != wanted:
        raise PrepareContractError(
            f"panel is missing images {sorted(wanted - set(out))!r}"
        )
    return out


# ---------------------------------------------------------------------------
# Query-group selection
# ---------------------------------------------------------------------------


def _u_block(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    block = summary.get("upper_bound_u")
    if not isinstance(block, Mapping):
        raise PrepareContractError(
            f"owner summary {summary.get('gt_owner_id')!r} carries no upper_bound_u block"
        )
    return block


def _view_context(block: Mapping[str, Any], view: str) -> str | None:
    projected = block.get(view)
    if not isinstance(projected, Mapping):
        return None
    context_id = projected.get("context_id")
    return None if context_id is None else str(context_id)


def due_context_by_owner(plan: BasePlan) -> dict[str, str]:
    """Deterministic due context of every native true positive.

    Mirrors the predecessor's sealed rule: the boundary index of the owner's
    unique native strict-match row, taken from the native sidecar registry.
    Owners without a unique mapping are simply absent, exactly as the
    predecessor records them as calibration exclusions.
    """

    row_index_by_pred: dict[str, int] = {}
    for row in plan.native_sidecars:
        pred_row_id = row.get("pred_row_id")
        if pred_row_id is None or "row_index" not in row:
            continue
        pred_row_id = str(pred_row_id)
        if pred_row_id in row_index_by_pred:
            raise PrepareContractError(
                f"native sidecar registry maps {pred_row_id!r} to two row indices"
            )
        row_index_by_pred[pred_row_id] = int(row["row_index"])

    mapping: dict[str, str] = {}
    for owner_id, owner in plan.owners.items():
        if not owner.get("native_true_positive"):
            continue
        matches = [str(value) for value in owner.get("native_strict_match_pred_row_ids", ())]
        if len(matches) != 1 or matches[0] not in row_index_by_pred:
            continue
        context_id = f"{owner['image_id']}:boundary-{row_index_by_pred[matches[0]]:03d}"
        if context_id in plan.contexts:
            mapping[owner_id] = context_id
    return mapping


def _admitted_group_id(
    plan: BasePlan, *, context_id: str, normalized_description: str
) -> str | None:
    group_id = f"{context_id}|{normalized_description}"
    group = plan.query_groups.get(group_id)
    if group is None or group.get("status") != "admitted":
        return None
    return group_id


def select_query_groups(
    plan: BasePlan, summaries: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Freeze the query groups this intervention will execute.

    Selection consumes only the *predecessor's* dispositions and frozen context
    views.  No treatment score exists yet, and none is ever consulted: this
    function is called before a single treatment forward pass.

    Cohorts
    -------
    ``persistent_outside_restricted``
        every non-loop native context of every category that contains a
        persistent owner, on the eleven images other than 4134.  A whole
        category is taken because support is a *category-field* estimand: a
        persistent owner's recovery has to be measurable wherever that category
        is queried, not only where the predecessor happened to find support.
    ``persistent_restricted``
        image 4134 only, on a declared restricted union: root + the owner's
        frozen U-frontier + its frozen U-best-diagnostic context.
    ``discovery_tp_calibration``
        discovery-half native true positives at their deterministic due
        contexts; this is what re-derives the treatment-specific q10 thresholds.
    ``confirmation_tp_retention``
        confirmation-half native true positives at due plus frozen U usable
        support contexts; the retention gate.
    ``resolved_retention``
        resolved-support owners at their frozen U usable support contexts, plus
        due and U-frontier where available.
    """

    by_owner = {str(row["gt_owner_id"]): row for row in summaries}
    due = due_context_by_owner(plan)

    selected: dict[str, set[str]] = {}
    cohorts: dict[str, list[str]] = {
        "persistent_outside_restricted": [],
        "persistent_restricted": [],
        "discovery_tp_calibration": [],
        "confirmation_tp_retention": [],
        "resolved_retention": [],
    }
    reasons: dict[str, set[str]] = {}

    def take(image_id: str, context_id: str, description: str, *, reason: str) -> None:
        group_id = _admitted_group_id(
            plan, context_id=context_id, normalized_description=description
        )
        if group_id is None:
            return
        selected.setdefault(image_id, set()).add(group_id)
        reasons.setdefault(group_id, set()).add(reason)

    # -- persistent owners, eleven images: whole category, non-loop contexts.
    persistent_categories: dict[str, set[str]] = {}
    for owner_id, summary in sorted(by_owner.items()):
        if str(summary.get("disposition")) != PERSISTENT_DISPOSITION:
            continue
        image_id = str(summary["image_id"])
        if image_id == RESTRICTED_IMAGE_ID:
            cohorts["persistent_restricted"].append(owner_id)
            continue
        cohorts["persistent_outside_restricted"].append(owner_id)
        persistent_categories.setdefault(image_id, set()).add(
            str(summary["normalized_description"])
        )

    for image_id, descriptions in sorted(persistent_categories.items()):
        for context in sorted(
            (
                row
                for row in plan.contexts.values()
                if str(row["image_id"]) == image_id
                and not bool(row["loop_marking"]["loop_tail"])
            ),
            key=lambda row: int(row["boundary_index"]),
        ):
            for description in sorted(descriptions):
                take(
                    image_id,
                    str(context["context_id"]),
                    description,
                    reason="persistent_outside_restricted",
                )

    # -- persistent owners on 4134: declared restricted union only.
    for owner_id in cohorts["persistent_restricted"]:
        summary = by_owner[owner_id]
        description = str(summary["normalized_description"])
        block = _u_block(summary)
        context_ids = {f"{RESTRICTED_IMAGE_ID}:boundary-000"}
        for view in ("primary_first_non_loop_minimal_abs_frontier", "diagnostic_best_all"):
            context_id = _view_context(block, view)
            if context_id is not None:
                context_ids.add(context_id)
        for context_id in sorted(context_ids):
            take(
                RESTRICTED_IMAGE_ID, context_id, description, reason="persistent_restricted"
            )

    # -- calibration and retention controls.
    for owner_id, summary in sorted(by_owner.items()):
        image_id = str(summary["image_id"])
        description = str(summary["normalized_description"])
        split = str(summary["split"])
        block = _u_block(summary)
        due_context = due.get(owner_id)

        if summary.get("native_true_positive"):
            if split == "discovery":
                cohorts["discovery_tp_calibration"].append(owner_id)
                if due_context is not None:
                    take(
                        image_id,
                        due_context,
                        description,
                        reason="discovery_tp_calibration",
                    )
            else:
                cohorts["confirmation_tp_retention"].append(owner_id)
                context_ids = set(
                    str(value) for value in block.get("usable_support_context_ids") or ()
                )
                if due_context is not None:
                    context_ids.add(due_context)
                for context_id in sorted(context_ids):
                    take(
                        image_id,
                        context_id,
                        description,
                        reason="confirmation_tp_retention",
                    )
            continue

        if str(summary.get("disposition")) == RESOLVED_DISPOSITION:
            cohorts["resolved_retention"].append(owner_id)
            context_ids = set(
                str(value) for value in block.get("usable_support_context_ids") or ()
            )
            if due_context is not None:
                context_ids.add(due_context)
            frontier = _view_context(block, "primary_first_non_loop_minimal_abs_frontier")
            if frontier is not None:
                context_ids.add(frontier)
            for context_id in sorted(context_ids):
                take(image_id, context_id, description, reason="resolved_retention")

    if not selected:
        raise PrepareContractError("query-group selection is empty; nothing would be scored")

    # Cost accounting.  The selection rule is fixed, but its cost is not obvious
    # from the rule, and a 2x-token capture is expensive: publish the breakdown
    # so trimming stays an explicit decision rather than a surprise.
    groups_by_reason: dict[str, int] = {}
    rows_by_reason: dict[str, int] = {}
    rows_by_image: dict[str, int] = {}
    for image_id, group_ids in selected.items():
        for group_id in group_ids:
            candidate_count = len(plan.query_groups[group_id]["candidate_ids"])
            rows_by_image[image_id] = rows_by_image.get(image_id, 0) + candidate_count
            for reason in reasons[group_id]:
                groups_by_reason[reason] = groups_by_reason.get(reason, 0) + 1
                rows_by_reason[reason] = rows_by_reason.get(reason, 0) + candidate_count

    return {
        "selection_policy": "frozen_before_any_treatment_score_never_score_selected",
        "restricted_image_id": RESTRICTED_IMAGE_ID,
        "cohort_owner_ids": {key: sorted(set(value)) for key, value in cohorts.items()},
        "cohort_owner_counts": {key: len(set(value)) for key, value in cohorts.items()},
        "query_group_ids_by_image": {
            image_id: sorted(group_ids) for image_id, group_ids in sorted(selected.items())
        },
        "query_group_count": sum(len(value) for value in selected.values()),
        "query_group_reasons": {
            group_id: sorted(value) for group_id, value in sorted(reasons.items())
        },
        "due_context_by_owner": dict(sorted(due.items())),
        "cost": {
            "candidate_score_rows": sum(rows_by_image.values()),
            "candidate_score_rows_by_image": dict(sorted(rows_by_image.items())),
            "query_groups_by_reason_overlapping": dict(sorted(groups_by_reason.items())),
            "candidate_score_rows_by_reason_overlapping": dict(sorted(rows_by_reason.items())),
            "reason_totals_overlap": "a group selected for two cohorts is counted under both",
        },
    }


# ---------------------------------------------------------------------------
# Overlay application
# ---------------------------------------------------------------------------


def overlay_content_sha256(overlay: Mapping[str, Any]) -> str:
    return sha256_json(
        {key: value for key, value in overlay.items() if key != "overlay_content_sha256"}
    )


def assert_overlay_seal(overlay: Mapping[str, Any]) -> None:
    """Fail closed unless the overlay reconstructs its own digest and identity."""

    if overlay.get("schema_version") != OVERLAY_SCHEMA_VERSION:
        raise PrepareContractError("overlay schema_version does not match this intervention")
    if overlay.get("intervention_unit_id") != INTERVENTION_UNIT_ID:
        raise PrepareContractError("overlay belongs to another intervention unit")
    if overlay.get("arm_id") != ARM_ID:
        raise PrepareContractError(
            f"overlay declares arm {overlay.get('arm_id')!r}; this unit executes {ARM_ID!r}"
        )
    if overlay.get("arm_id") == overlay.get("baseline_arm_id"):
        raise PrepareContractError("overlay treatment and baseline arms are the same arm")
    if overlay_content_sha256(overlay) != overlay.get("overlay_content_sha256"):
        raise PrepareContractError("overlay does not reconstruct its own digest; it was edited")


def load_overlay(path: Path) -> dict[str, Any]:
    overlay = _read_json(Path(path), "intervention overlay")
    assert_overlay_seal(overlay)
    return overlay


def assert_overlay_binds_plan(
    overlay: Mapping[str, Any],
    *,
    plan_receipt_content_sha256: str,
    capture_rules_sha256: str | None = None,
) -> None:
    base = overlay.get("base") or {}
    if str(base.get("plan_receipt_content_sha256")) != str(plan_receipt_content_sha256):
        raise PrepareContractError(
            "overlay was sealed against a different predecessor plan receipt digest"
        )
    if capture_rules_sha256 is not None and str(base.get("capture_rules_sha256")) != str(
        capture_rules_sha256
    ):
        raise PrepareContractError(
            "overlay was sealed against different predecessor capture rules"
        )


def _frozen_fingerprint(
    images: Mapping[str, Mapping[str, Any]],
    contexts: Mapping[str, Mapping[str, Any]],
    query_groups: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Everything the overlay must leave byte-identical."""

    return {
        "coordinate_token_ids": {
            image_id: row.get("coordinate_token_ids") for image_id, row in sorted(images.items())
        },
        "wrapper_token_ids": {
            image_id: row.get("wrapper_token_ids") for image_id, row in sorted(images.items())
        },
        "generated_prefix_token_ids": {
            context_id: list(row["generated_prefix_token_ids"])
            for context_id, row in sorted(contexts.items())
        },
        "generated_prefix_token_ids_sha256": {
            context_id: row["generated_prefix_token_ids_sha256"]
            for context_id, row in sorted(contexts.items())
        },
        "query_suffix_token_ids": {
            group_id: list(row["query_suffix_token_ids"])
            for group_id, row in sorted(query_groups.items())
        },
        "query_suffix_token_ids_sha256": {
            group_id: row["query_suffix_token_ids_sha256"]
            for group_id, row in sorted(query_groups.items())
        },
        "candidate_ids": {
            group_id: list(row["candidate_ids"])
            for group_id, row in sorted(query_groups.items())
        },
        "proposal_route_token_ids": {
            group_id: list(row.get("proposal_route_token_ids") or ())
            for group_id, row in sorted(query_groups.items())
        },
        "proposal_route_digest": {
            group_id: row.get("proposal_route_digest")
            for group_id, row in sorted(query_groups.items())
        },
    }


def apply_overlay(
    overlay: Mapping[str, Any],
    *,
    images: dict[str, Any],
    contexts: dict[str, Any],
    query_groups: dict[str, Any],
    categories: Mapping[str, Any],
    candidates: Mapping[str, Any] | None = None,
    owners: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Replace the image prompt prefix in memory and re-derive prefix identity.

    The generated prefix of every context is preserved byte for byte -- only the
    *image prompt* run in front of it changes, because only the visual token
    count changed.  Observed/query prefix digests and exact-prefix admission IDs
    are then recomputed from the new literal token sequences, exactly as the
    scorer and merge would re-derive them.

    Never writes anything: the caller owns a mutable copy of the registries.
    """

    assert_overlay_seal(overlay)
    before = _frozen_fingerprint(images, contexts, query_groups)
    candidate_fingerprint = (
        None
        if candidates is None
        else sha256_json(
            {
                candidate_id: {
                    "coord_token_ids": list(row["coord_token_ids"]),
                    "coord_token_ids_sha256": row.get("coord_token_ids_sha256"),
                    "strict_assignment_status": row.get("strict_assignment_status"),
                    "strict_assignment_gt_owner_id": row.get("strict_assignment_gt_owner_id"),
                    "ambiguity_owner_ids": list(row.get("ambiguity_owner_ids") or ()),
                }
                for candidate_id, row in sorted(candidates.items())
            }
        )
    )
    owner_fingerprint = (
        None
        if owners is None
        else sha256_json(
            {
                owner_id: {
                    "bbox_pixel_xyxy": list(row.get("bbox_pixel_xyxy") or ()),
                    "normalized_description": row.get("normalized_description"),
                    "native_true_positive": row.get("native_true_positive"),
                    "candidate_bank": row.get("candidate_bank"),
                }
                for owner_id, row in sorted(owners.items())
            }
        )
    )

    overlay_images = overlay.get("images") or {}
    if set(overlay_images) != set(images):
        raise PrepareContractError(
            "overlay image set does not match the plan image registry; refusing a partial "
            "presentation swap that would mix 1x and 2x media in one capture"
        )

    applied: dict[str, Any] = {}
    for image_id, block in sorted(overlay_images.items()):
        current = block["current"]
        treatment = block["treatment"]
        registry = images[image_id]

        # The overlay's "current" side must still describe the plan on disk;
        # otherwise this overlay belongs to another capture of the census.
        if str(registry["prompt_token_ids_sha256"]) != str(
            current["prompt_token_ids_sha256"]
        ):
            raise PrepareContractError(
                f"image {image_id!r}: overlay current prompt digest does not match the plan"
            )
        if str(registry["executed_media_sha256"]) != str(current["executed_media_sha256"]):
            raise PrepareContractError(
                f"image {image_id!r}: overlay current media digest does not match the plan"
            )

        treatment_prompt = [int(value) for value in treatment["prompt_token_ids"]]
        if sha256_json(treatment_prompt) != str(treatment["prompt_token_ids_sha256"]):
            raise PrepareContractError(
                f"image {image_id!r}: treatment prompt tokens do not reconstruct their digest"
            )
        if treatment_prompt == [int(value) for value in registry["prompt_token_ids"]]:
            raise PrepareContractError(
                f"image {image_id!r}: treatment prompt is identical to the current prompt; "
                "this would be a null intervention presented as a treatment"
            )

        registry["prompt_token_ids"] = list(treatment_prompt)
        registry["prompt_token_ids_sha256"] = str(treatment["prompt_token_ids_sha256"])
        registry["image_width"] = int(treatment["width"])
        registry["image_height"] = int(treatment["height"])
        registry["executed_media_sha256"] = str(treatment["executed_media_sha256"])

        image_contexts = [
            row for row in contexts.values() if str(row["image_id"]) == image_id
        ]
        for context in image_contexts:
            generated = [int(value) for value in context["generated_prefix_token_ids"]]
            observed = [*treatment_prompt, *generated]
            context["prompt_token_ids_sha256"] = str(treatment["prompt_token_ids_sha256"])
            context["observed_self_prefix_token_ids_sha256"] = sha256_json(observed)
            context["observed_self_prefix_token_count"] = len(observed)

        group_count = 0
        for group in query_groups.values():
            if str(group["image_id"]) != image_id:
                continue
            context = contexts[str(group["context_id"])]
            generated = [int(value) for value in context["generated_prefix_token_ids"]]
            observed = [*treatment_prompt, *generated]
            suffix = [int(value) for value in group["query_suffix_token_ids"]]
            query_prefix = [*observed, *suffix]
            observed_sha = sha256_json(observed)
            query_sha = sha256_json(query_prefix)
            category = categories[str(group["category_query_id"])]
            category_tokens = [int(value) for value in category["category_token_ids"]]

            group["observed_prefix_sha256"] = observed_sha
            group["observed_prefix_token_count"] = len(observed)
            group["query_prefix_sha256"] = query_sha
            group["query_prefix_token_count"] = len(query_prefix)
            group["admission_receipt_id"] = planner.admission_receipt_id(
                context_id=str(group["context_id"]),
                channel=planner.CHANNEL_QUERY_SUFFIX,
                prefix_sha256=query_sha,
            )
            group["proposal_boundary_gate_admission_receipt_id"] = planner.admission_receipt_id(
                context_id=str(group["context_id"]),
                channel=planner.CHANNEL_PROPOSAL_BOUNDARY_GATE,
                prefix_sha256=observed_sha,
            )
            group["proposal_route_admission_receipt_id"] = (
                planner.proposal_route_admission_receipt_id(
                    context_id=str(group["context_id"]),
                    observed_prefix_sha256=observed_sha,
                    category_token_ids=category_tokens,
                )
            )
            singleton = dict(group.get("singleton_group_key") or {})
            singleton["observed_prefix_sha256"] = observed_sha
            singleton["query_prefix_sha256"] = query_sha
            group["singleton_group_key"] = singleton
            group_count += 1

        applied[image_id] = {
            "treatment_prompt_token_count": len(treatment_prompt),
            "current_prompt_token_count": int(current["prompt_token_count"]),
            "context_count": len(image_contexts),
            "query_group_count": group_count,
        }

    after = _frozen_fingerprint(images, contexts, query_groups)
    if before != after:
        differing = sorted(key for key in before if before[key] != after[key])
        raise PrepareContractError(
            "overlay application changed frozen plan content "
            f"({differing!r}); candidate IDs, coordinate tokens, query suffixes and "
            "generated prefixes must survive a presentation swap untouched"
        )
    if candidates is not None:
        recomputed = sha256_json(
            {
                candidate_id: {
                    "coord_token_ids": list(row["coord_token_ids"]),
                    "coord_token_ids_sha256": row.get("coord_token_ids_sha256"),
                    "strict_assignment_status": row.get("strict_assignment_status"),
                    "strict_assignment_gt_owner_id": row.get("strict_assignment_gt_owner_id"),
                    "ambiguity_owner_ids": list(row.get("ambiguity_owner_ids") or ()),
                }
                for candidate_id, row in sorted(candidates.items())
            }
        )
        if recomputed != candidate_fingerprint:
            raise PrepareContractError(
                "overlay application changed the sealed candidate bank or its assignments"
            )
    if owners is not None:
        recomputed_owners = sha256_json(
            {
                owner_id: {
                    "bbox_pixel_xyxy": list(row.get("bbox_pixel_xyxy") or ()),
                    "normalized_description": row.get("normalized_description"),
                    "native_true_positive": row.get("native_true_positive"),
                    "candidate_bank": row.get("candidate_bank"),
                }
                for owner_id, row in sorted(owners.items())
            }
        )
        if recomputed_owners != owner_fingerprint:
            raise PrepareContractError(
                "overlay application changed owner geometry or bank accounting"
            )

    return {
        "arm_id": str(overlay["arm_id"]),
        "overlay_content_sha256": str(overlay["overlay_content_sha256"]),
        "max_pixels": int(overlay["max_pixels"]),
        "images": applied,
        "frozen_content_preserved": True,
    }


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------


def reproduce_current_pool(
    *, raw_path: Path, registry: Mapping[str, Any]
) -> dict[str, Any]:
    """Prove the current 1024-pool image is byte-exact reproducible from raw."""

    from PIL import Image

    with Image.open(raw_path) as handle:
        raw_width, raw_height = int(handle.width), int(handle.height)
    height, width = aligned_dimensions(
        height=raw_height, width=raw_width, max_pixels=BASELINE_MAX_PIXELS
    )
    payload = render_pool_jpeg_bytes(raw_path, width=width, height=height)
    identity = decoded_media_identity(payload)
    dimensions_match = width == int(registry["image_width"]) and height == int(
        registry["image_height"]
    )
    media_match = identity["executed_media_sha256"] == str(registry["executed_media_sha256"])
    return {
        "raw_width": raw_width,
        "raw_height": raw_height,
        "width": width,
        "height": height,
        "dimensions_match": dimensions_match,
        "executed_media_sha256": identity["executed_media_sha256"],
        "file_sha256": identity["file_sha256"],
        "executed_media_sha256_match": media_match,
        "reproduced": bool(dimensions_match and media_match),
        "merged_visual_tokens": merged_visual_token_count(width=width, height=height),
    }


def prepare_intervention(
    *,
    base_plan_dir: Path,
    predecessor_run_root: Path,
    panel_jsonl: Path,
    owner_summaries_jsonl: Path,
    raw_root: Path,
    output_root: Path,
    infer_config: Path,
    dry_run: bool = False,
    resolve_prompts: Any = None,
) -> dict[str, Any]:
    """Build and seal the intervention overlay; return the prepare receipt.

    ``dry_run`` stops after the 1x reproduction proof and the treatment
    dimension derivation, writes nothing, and returns a partial report.  The
    full path *must* materialize the treatment pool and panel, because the
    treatment ``image_grid_thw`` and prompt token IDs come from the production
    frontend reading the real media -- this probe never estimates a grid.
    """

    plan = load_base_plan(base_plan_dir)
    panel = load_panel(panel_jsonl)
    validate_panel_against_plan(panel, plan)
    summaries = _read_jsonl(Path(owner_summaries_jsonl), "predecessor owner summaries")
    if not summaries:
        raise PrepareContractError("predecessor owner summaries are empty")

    output_root = Path(output_root)
    media_root = output_root / MEDIA_DIRNAME
    treatment_panel_path = output_root / TREATMENT_PANEL_NAME

    # -- Step 1: prove the current pool reproduces from raw, for all twelve.
    reproduction: dict[str, Any] = {}
    raw_identity: dict[str, Any] = {}
    for image_id in sorted(plan.images):
        raw_path = Path(raw_root) / str(panel[image_id]["file_name"])
        if not raw_path.is_file():
            raise PrepareContractError(f"raw source image is missing at {raw_path}")
        reproduction[image_id] = reproduce_current_pool(
            raw_path=raw_path, registry=plan.images[image_id]
        )
        raw_identity[image_id] = {
            "path": str(raw_path),
            "sha256": sha256_file(raw_path),
            "width": reproduction[image_id]["raw_width"],
            "height": reproduction[image_id]["raw_height"],
        }
    failed = sorted(key for key, value in reproduction.items() if not value["reproduced"])
    if failed:
        raise PrepareContractError(
            f"the current 1024-pool media is not byte-exactly reproducible from raw for "
            f"images {failed!r}; the treatment pool cannot be claimed to differ from the "
            "current pool only in token density"
        )

    # -- Step 2: derive treatment dimensions and materialize the treatment pool.
    treatment_media: dict[str, Any] = {}
    panel_rows: list[dict[str, Any]] = []
    for image_id in sorted(plan.images, key=lambda value: int(value)):
        raw_path = Path(raw_identity[image_id]["path"])
        height, width = aligned_dimensions(
            height=raw_identity[image_id]["height"],
            width=raw_identity[image_id]["width"],
            max_pixels=TREATMENT_MAX_PIXELS,
        )
        relative = str(Path("media") / str(panel[image_id]["file_name"]))
        payload = render_pool_jpeg_bytes(raw_path, width=width, height=height)
        identity = decoded_media_identity(payload)
        if identity["width"] != width or identity["height"] != height:
            raise PrepareContractError(
                f"image {image_id!r}: materialized treatment media has unexpected dimensions"
            )
        if identity["executed_media_sha256"] == str(
            plan.images[image_id]["executed_media_sha256"]
        ):
            raise PrepareContractError(
                f"image {image_id!r}: treatment media is pixel-identical to the current pool"
            )
        treatment_media[image_id] = {
            "width": width,
            "height": height,
            "relative_path": relative,
            **identity,
            "merged_visual_tokens": merged_visual_token_count(width=width, height=height),
        }
        if not dry_run:
            destination = media_root / str(panel[image_id]["file_name"])
            _write_durable(destination, payload)
        panel_rows.append(
            derive_treatment_panel_row(
                panel[image_id],
                treatment_width=width,
                treatment_height=height,
                relative_image_path=relative,
            )
        )

    panel_bytes = b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
        for row in panel_rows
    )

    if dry_run:
        return {
            "dry_run": True,
            "current_pool_reproduced_from_raw": {
                image_id: value["reproduced"] for image_id, value in sorted(reproduction.items())
            },
            "treatment_dimensions": {
                image_id: {
                    "width": value["width"],
                    "height": value["height"],
                    "merged_visual_tokens": value["merged_visual_tokens"],
                }
                for image_id, value in sorted(treatment_media.items())
            },
            "current_merged_visual_tokens": {
                image_id: value["merged_visual_tokens"]
                for image_id, value in sorted(reproduction.items())
            },
            "treatment_panel_sha256": hashlib.sha256(panel_bytes).hexdigest(),
            "wrote_anything": False,
        }

    _write_durable(treatment_panel_path, panel_bytes)

    # -- Step 3: production prompts for both presentations.
    resolver = resolve_panel_prompts if resolve_prompts is None else resolve_prompts
    current_prompts = resolver(
        Path(panel_jsonl), infer_config=Path(infer_config), image_ids=sorted(plan.images)
    )
    treatment_prompts = resolver(
        treatment_panel_path, infer_config=Path(infer_config), image_ids=sorted(plan.images)
    )

    images_block: dict[str, Any] = {}
    for image_id in sorted(plan.images):
        registry = plan.images[image_id]
        current = current_prompts[image_id]
        treatment = treatment_prompts[image_id]
        media = treatment_media[image_id]
        if current.prompt_token_ids_sha256 != str(registry["prompt_token_ids_sha256"]):
            raise PrepareContractError(
                f"image {image_id!r}: the production frontend does not reproduce the sealed "
                "current prompt; the predecessor run and this checkout disagree"
            )
        if treatment.decoded_width != media["width"] or treatment.decoded_height != media[
            "height"
        ]:
            raise PrepareContractError(
                f"image {image_id!r}: treatment panel canvas and materialized media disagree"
            )
        if treatment.logical_transform_id != current.logical_transform_id:
            raise PrepareContractError(
                f"image {image_id!r}: treatment logical transform differs from the current "
                "presentation; only the pixel budget may change"
            )
        if treatment.merged_visual_tokens <= current.merged_visual_tokens:
            raise PrepareContractError(
                f"image {image_id!r}: treatment does not increase merged visual tokens"
            )
        images_block[image_id] = {
            "file_name": str(panel[image_id]["file_name"]),
            "raw": raw_identity[image_id],
            "current": {
                "arm_id": BASELINE_ARM_ID,
                "max_pixels": BASELINE_MAX_PIXELS,
                "width": int(registry["image_width"]),
                "height": int(registry["image_height"]),
                "executed_media_sha256": str(registry["executed_media_sha256"]),
                "file_sha256": str(current.image_content_sha256),
                "image_grid_thw": list(current.image_grid_thw),
                "merged_visual_tokens": current.merged_visual_tokens,
                "prompt_token_ids_sha256": current.prompt_token_ids_sha256,
                "prompt_token_count": len(current.prompt_token_ids),
                "reproduced_from_raw": reproduction[image_id],
            },
            "treatment": {
                "arm_id": ARM_ID,
                "max_pixels": TREATMENT_MAX_PIXELS,
                "width": media["width"],
                "height": media["height"],
                "media_relative_path": media["relative_path"],
                "executed_media_sha256": media["executed_media_sha256"],
                "file_sha256": media["file_sha256"],
                "image_grid_thw": list(treatment.image_grid_thw),
                "merged_visual_tokens": treatment.merged_visual_tokens,
                "prompt_token_ids": list(treatment.prompt_token_ids),
                "prompt_token_ids_sha256": treatment.prompt_token_ids_sha256,
                "prompt_token_count": len(treatment.prompt_token_ids),
                "merged_visual_token_ratio_vs_current": (
                    treatment.merged_visual_tokens / current.merged_visual_tokens
                ),
            },
        }

    selection = select_query_groups(plan, summaries)

    overlay: dict[str, Any] = {
        "schema_version": OVERLAY_SCHEMA_VERSION,
        "intervention_unit_id": INTERVENTION_UNIT_ID,
        "arm_id": ARM_ID,
        "baseline_arm_id": BASELINE_ARM_ID,
        "max_pixels": TREATMENT_MAX_PIXELS,
        "baseline_max_pixels": BASELINE_MAX_PIXELS,
        "image_factor": IMAGE_FACTOR,
        "claim_scope": "denser_patch_token_sampling_over_the_same_raw_optical_information",
        "claim_is_not": "new_visual_acuity_or_added_optical_information",
        "frozen": {
            "do_resize": False,
            "processor_pixel_knobs_added": False,
            "coordinate_bins": 1000,
            "candidate_bank": "frozen_from_predecessor_plan",
            "candidate_pixel_frame": "one_x_pixel_frame_unchanged",
            "generated_prefix_token_ids": "byte_identical",
            "query_suffix_token_ids": "byte_identical",
            "owner_assignments": "byte_identical",
            "model_identity": "sorted_step4887_hf_fp32_rp1p0",
        },
        "may_change": [
            "media_bytes",
            "media_dimensions",
            "image_grid_thw",
            "visual_image_pad_prompt_run",
            "prompt_and_prefix_digests",
            "exact_prefix_admission_ids",
            "arm_identity",
        ],
        "base": {
            "unit_id": BASE_UNIT_ID,
            "plan_schema_version": BASE_PLAN_SCHEMA_VERSION,
            "plan_dir": str(Path(base_plan_dir).resolve()),
            "plan_receipt_content_sha256": plan.receipt_content_sha256,
            "capture_rules_sha256": plan.capture_rules_sha256,
            "predecessor_run_root": str(Path(predecessor_run_root).resolve()),
            "panel_jsonl": str(Path(panel_jsonl).resolve()),
            "panel_sha256": sha256_file(Path(panel_jsonl)),
            "owner_summaries_jsonl": str(Path(owner_summaries_jsonl).resolve()),
            "owner_summaries_sha256": sha256_file(Path(owner_summaries_jsonl)),
            "infer_config": str(Path(infer_config).resolve()),
        },
        "treatment_panel": {
            "relative_path": TREATMENT_PANEL_NAME,
            "sha256": hashlib.sha256(panel_bytes).hexdigest(),
            "media_root_relative_path": MEDIA_DIRNAME,
        },
        "images": images_block,
        "query_group_selection": selection,
        "prepare_source_sha256": EXECUTED_SOURCE_SHA256,
    }
    overlay["overlay_content_sha256"] = overlay_content_sha256(overlay)
    assert_overlay_seal(overlay)

    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "intervention_unit_id": INTERVENTION_UNIT_ID,
        "arm_id": ARM_ID,
        "overlay_content_sha256": overlay["overlay_content_sha256"],
        "base_plan_receipt_content_sha256": plan.receipt_content_sha256,
        "predecessor_run_root": str(Path(predecessor_run_root).resolve()),
        "output_root": str(output_root.resolve()),
        "media_materialized": True,
        "current_pool_reproduced_from_raw": {
            image_id: value["reproduced"] for image_id, value in sorted(reproduction.items())
        },
        "image_count": len(images_block),
        "merged_visual_tokens": {
            image_id: {
                "current": block["current"]["merged_visual_tokens"],
                "treatment": block["treatment"]["merged_visual_tokens"],
                "ratio": block["treatment"]["merged_visual_token_ratio_vs_current"],
            }
            for image_id, block in sorted(images_block.items())
        },
        "total_merged_visual_tokens": {
            "current": sum(
                block["current"]["merged_visual_tokens"] for block in images_block.values()
            ),
            "treatment": sum(
                block["treatment"]["merged_visual_tokens"] for block in images_block.values()
            ),
        },
        "selection": {
            "query_group_count": selection["query_group_count"],
            "query_group_count_by_image": {
                image_id: len(group_ids)
                for image_id, group_ids in sorted(
                    selection["query_group_ids_by_image"].items()
                )
            },
            "cohort_owner_counts": selection["cohort_owner_counts"],
            "cost": selection["cost"],
        },
        "score_input_policy": {
            "candidate_selection_uses_treatment_scores": False,
            "query_group_selection_uses_treatment_scores": False,
            "reads_any_treatment_score_artifact": False,
        },
        "writes_outside_output_root": False,
        "prepare_source_sha256": EXECUTED_SOURCE_SHA256,
    }
    receipt["receipt_content_sha256"] = sha256_json(receipt)

    _write_durable(output_root / OVERLAY_NAME, canonical_json_bytes(overlay) + b"\n")
    _write_durable(
        output_root / SELECTION_NAME,
        b"".join(
            canonical_json_bytes(
                {
                    "row_kind": "intervention_selected_query_group",
                    "intervention_unit_id": INTERVENTION_UNIT_ID,
                    "arm_id": ARM_ID,
                    "image_id": image_id,
                    "query_group_id": group_id,
                    "selection_reasons": selection["query_group_reasons"][group_id],
                }
            )
            + b"\n"
            for image_id, group_ids in sorted(selection["query_group_ids_by_image"].items())
            for group_id in group_ids
        ),
    )
    _write_durable(output_root / RECEIPT_NAME, canonical_json_bytes(receipt) + b"\n")

    return {"receipt": receipt, "overlay": overlay, "treatment_panel_bytes": panel_bytes}


def _write_durable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--predecessor-run-root", type=Path, required=True)
    parser.add_argument(
        "--base-plan-dir",
        type=Path,
        default=None,
        help="defaults to <predecessor-run-root>/plan",
    )
    parser.add_argument(
        "--owner-summaries",
        type=Path,
        default=None,
        help="defaults to <predecessor-run-root>/phases/presentation/owner-summaries.jsonl",
    )
    parser.add_argument("--panel-jsonl", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--infer-config", type=Path, default=DEFAULT_INFER_CONFIG)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "prove the 1x reproduction and derive the treatment dimensions only; write "
            "nothing.  The full path materializes the treatment pool because the "
            "treatment grid and prompt come from the production frontend reading the "
            "real media, never from an estimator"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    run_root = Path(args.predecessor_run_root)
    base_plan_dir = args.base_plan_dir or (run_root / "plan")
    owner_summaries = args.owner_summaries or (
        run_root / "phases" / "presentation" / "owner-summaries.jsonl"
    )
    try:
        result = prepare_intervention(
            base_plan_dir=base_plan_dir,
            predecessor_run_root=run_root,
            panel_jsonl=args.panel_jsonl,
            owner_summaries_jsonl=owner_summaries,
            raw_root=args.raw_root,
            output_root=args.output_root,
            infer_config=args.infer_config,
            dry_run=bool(args.dry_run),
        )
    except PrepareContractError as exc:
        print(f"prepare error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result.get("receipt", result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
