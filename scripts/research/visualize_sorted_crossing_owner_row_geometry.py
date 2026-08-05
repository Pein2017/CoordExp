#!/usr/bin/env python3
"""Renderer for the sorted crossing owner-row geometric relation stratification
unit (``2026-08-04-sorted-crossing-owner-row-geometric-relation-stratification``).

Frozen unit::

    research/investigations/qwen3-vl-dense-enumeration/experiments/
    2026-08-04-sorted-crossing-owner-row-geometric-relation-stratification/
    {unit.md,tasks.md}

This module reads **only** the analysis directory published by
``analyze_sorted_crossing_owner_row_geometry.py``.  It verifies
``geometry-analysis-receipt.json``'s own self-seal, then every file digest that
receipt declares -- including ``geometry-visual-plan.json`` -- before a single
byte of the plan is parsed, and fails closed if the analysis directory carries
a file the receipt never sealed.

It draws exactly what the plan says and nothing else
------------------------------------------------------
The analyzer owns every geometric measure, every stratum label, every
materiality verdict and the three-way decision.  This renderer never recomputes
IoU, a centre distance, a relative coordinate delta, a geometric label, a rank
gap or a decision; it projects the plan's sealed norm-1000 boxes into the
sealed image extent for display only and prints the plan's own annotation
values verbatim.

Fail-closed preconditions
-------------------------
* receipt/plan schema drift, foreign ``unit_id``, or a broken receipt self-seal;
* any analysis file whose bytes differ from the sealed digest, or any extra file;
* a duplicate panel ID, output file name, point ID or box ID;
* an invalid norm-1000 box or crop window, or a non-finite scatter coordinate;
* a missing media file, a media file whose pixel size is not the sealed
  ``image_width``/``image_height``, or a ``file_name`` that escapes its root;
* incomplete visual coverage -- the plan's own panel/output-file counts, the
  arm x measure scatter grid, the per-arm point counts and greedy-pair panel
  count sealed by the receipt's denominators, and one crop panel for every
  material-negative scatter point.

Products (published together, atomically, create-or-identical)::

    scatter-<arm>-<measure>.png    one per plan scatter panel
    crop-<...>.png                 one per plan crop panel
    panel-specs.jsonl              the validated, rendered-verbatim panel specs
    visual-manifest.json           self-sealed input/output digests and IDs
    visual-index.md                human-readable panel index

``--specs-only`` publishes the specs, manifest and index without any PNG.  It
still resolves and verifies every media file, so a specs-only run proves the
same media preconditions a rendered run does.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import io
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, NoReturn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import analyze_sorted_crossing_owner_row_geometry as analyzer  # noqa: E402

UNIT_ID = analyzer.UNIT_ID
VISUAL_SCHEMA_VERSION = "sorted-crossing-owner-row-geometry-visual-spec.v1"
MANIFEST_SCHEMA_VERSION = "sorted-crossing-owner-row-geometry-visual-manifest.v1"

PANEL_SPECS_NAME = "panel-specs.jsonl"
MANIFEST_NAME = "visual-manifest.json"
INDEX_NAME = "visual-index.md"

#: The plan's crop panels for the twelve frozen greedy displacement pairs.
GREEDY_PAIR_ARM = "greedy_displacement_pair"
MATERIAL_NEGATIVE_PANEL_KIND = "material_negative_case"
GREEDY_PAIR_PANEL_KIND = "greedy_displacement_pair"

#: The two continuous geometry measures unit.md "Reports" item 5 names.
SCATTER_MEASURES: tuple[str, ...] = ("iou", "center_distance_normalized")

#: ``role -> the box roles that may appear beside it`` for each crop panel kind.
MATERIAL_NEGATIVE_ROLE_SETS: tuple[frozenset[str], ...] = (
    frozenset(("inserted_owner_c", "downstream_row_e")),
    frozenset(("inserted_owner_c", "downstream_row_f")),
)
GREEDY_PAIR_ROLE_SET = frozenset(("displacement_target", "displacer"))

#: Layout constants shared by the pure spec builders and the PIL renderers, so a
#: spec's declared ``dimensions`` always exactly matches the rendered PNG.
SCATTER_WIDTH = 760
SCATTER_HEIGHT = 560
SCATTER_MARGIN_LEFT = 92
SCATTER_MARGIN_RIGHT = 28
SCATTER_MARGIN_TOP = 64
SCATTER_MARGIN_BOTTOM = 80

CROP_MARGIN = 16
CROP_PANEL_WIDTH = 868
CROP_PANEL_MIN_HEIGHT = 180
CROP_PANEL_MAX_HEIGHT = 700
CROP_WIDTH = CROP_PANEL_WIDTH + CROP_MARGIN * 2
CROP_HEADER = 76
CROP_LINE_HEIGHT = 18

BACKGROUND_RGB = (18, 18, 20)
FOREGROUND_RGB = (232, 232, 238)
MUTED_RGB = (168, 168, 178)
AXIS_RGB = (120, 120, 130)
REFERENCE_RGB = (214, 96, 96)

_MISSING = object()


class VisualContractError(RuntimeError):
    """A precondition for rendering this unit's panels was not proven."""


def _fail(message: str) -> NoReturn:
    raise VisualContractError(message)


sha256_bytes = analyzer.sha256_bytes
sha256_json = analyzer.sha256_json
sha256_file = analyzer.sha256_file
canonical_json_bytes = analyzer.canonical_json_bytes


# ---------------------------------------------------------------------------
# 0. Small typed readers
# ---------------------------------------------------------------------------


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        _fail(f"{label} is missing at {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"{label} is unreadable at {path}: {exc}")
    if not isinstance(value, Mapping):
        _fail(f"{label} at {path} is not a JSON object")
    return dict(value)


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} is missing or is not an object")
    return value


def _require_list(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        _fail(f"{label} is missing or is not a list")
    return value


def _require_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{label} is missing or is not a nonempty string")
    return value


def _require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} is missing or is not a number")
    number = float(value)
    if not math.isfinite(number):
        _fail(f"{label} is not finite")
    return number


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        _fail(f"{label} is missing or is not a boolean")
    return value


def _require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{label} is missing or is not an integer")
    return int(value)


def _require_unique(values: Sequence[str], label: str) -> list[str]:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            _fail(f"{label} contains the duplicate ID {value!r}")
        seen.add(value)
    return list(values)


def _rgb(color: Any, label: str) -> tuple[int, int, int]:
    text = _require_str(color, label)
    if len(text) != 7 or not text.startswith("#"):
        _fail(f"{label} {text!r} is not a '#rrggbb' color")
    try:
        return (int(text[1:3], 16), int(text[3:5], 16), int(text[5:7], 16))
    except ValueError:
        _fail(f"{label} {text!r} is not a '#rrggbb' color")


# ---------------------------------------------------------------------------
# 1. Load and digest-verify the analysis directory
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Artifacts:
    analysis_dir: Path
    receipt: dict[str, Any]
    visual_plan: dict[str, Any]
    visual_plan_sha256: str
    receipt_content_sha256: str
    denominators: dict[str, Any]
    input_file_digests: dict[str, dict[str, Any]]


def load_artifacts(analysis_dir: Path) -> Artifacts:
    """Verify the receipt's self-seal, then every sealed digest, then parse the plan."""

    analysis_dir = Path(analysis_dir)
    if not analysis_dir.is_dir():
        _fail(f"analysis directory is missing at {analysis_dir}")

    receipt = _read_json(analysis_dir / analyzer.RECEIPT_NAME, analyzer.RECEIPT_NAME)
    if str(receipt.get("schema_version")) != analyzer.RECEIPT_SCHEMA_VERSION:
        _fail(
            f"{analyzer.RECEIPT_NAME} schema {receipt.get('schema_version')!r} is not "
            f"{analyzer.RECEIPT_SCHEMA_VERSION!r}"
        )
    if str(receipt.get("unit_id")) != UNIT_ID:
        _fail(f"{analyzer.RECEIPT_NAME} belongs to another unit")
    sealed_digest = receipt.get("receipt_content_sha256")
    reconstructed = sha256_json(
        {key: value for key, value in receipt.items() if key != "receipt_content_sha256"}
    )
    if reconstructed != sealed_digest:
        _fail(
            f"{analyzer.RECEIPT_NAME} does not reconstruct its own digest; it was edited "
            "after sealing"
        )

    declared = receipt.get("output_file_digests")
    if not isinstance(declared, Mapping) or not declared:
        _fail(f"{analyzer.RECEIPT_NAME} declares no output_file_digests")
    if analyzer.VISUAL_PLAN_NAME not in declared:
        _fail(f"{analyzer.RECEIPT_NAME} does not seal {analyzer.VISUAL_PLAN_NAME}")

    input_file_digests: dict[str, dict[str, Any]] = {}
    for name, entry in sorted(declared.items()):
        path = analysis_dir / str(name)
        if not path.is_file():
            _fail(f"analysis file {name!r} is missing at {path}")
        payload = path.read_bytes()
        observed = sha256_bytes(payload)
        if observed != str(_require_mapping(entry, f"sealed digest for {name!r}").get("sha256")):
            _fail(
                f"analysis file {name!r} does not match the digest sealed in "
                f"{analyzer.RECEIPT_NAME}; it is tampered or stale"
            )
        expected_size = entry.get("byte_size")
        if expected_size is not None and len(payload) != int(expected_size):
            _fail(
                f"analysis file {name!r} byte size does not match the size sealed in "
                f"{analyzer.RECEIPT_NAME}"
            )
        input_file_digests[str(name)] = {"byte_size": len(payload), "sha256": observed}

    receipt_payload = (analysis_dir / analyzer.RECEIPT_NAME).read_bytes()
    input_file_digests[analyzer.RECEIPT_NAME] = {
        "byte_size": len(receipt_payload),
        "sha256": sha256_bytes(receipt_payload),
    }

    expected_files = set(declared) | {analyzer.RECEIPT_NAME}
    present_files = {entry.name for entry in analysis_dir.iterdir() if entry.is_file()}
    unknown_files = sorted(present_files - expected_files)
    if unknown_files:
        _fail(
            f"analysis directory {analysis_dir} carries file(s) {unknown_files!r} that "
            f"{analyzer.RECEIPT_NAME} never sealed; the analyzer's output contract is closed"
        )

    visual_plan = _read_json(
        analysis_dir / analyzer.VISUAL_PLAN_NAME, analyzer.VISUAL_PLAN_NAME
    )
    if str(visual_plan.get("schema_version")) != analyzer.VISUAL_PLAN_SCHEMA_VERSION:
        _fail(
            f"{analyzer.VISUAL_PLAN_NAME} schema {visual_plan.get('schema_version')!r} is not "
            f"{analyzer.VISUAL_PLAN_SCHEMA_VERSION!r}"
        )
    if str(visual_plan.get("unit_id")) != UNIT_ID:
        _fail(f"{analyzer.VISUAL_PLAN_NAME} belongs to another unit")

    denominators = _require_mapping(
        receipt.get("denominators"), f"{analyzer.RECEIPT_NAME}.denominators"
    )

    return Artifacts(
        analysis_dir=analysis_dir,
        receipt=receipt,
        visual_plan=visual_plan,
        visual_plan_sha256=input_file_digests[analyzer.VISUAL_PLAN_NAME]["sha256"],
        receipt_content_sha256=str(sealed_digest),
        denominators=dict(denominators),
        input_file_digests=input_file_digests,
    )


# ---------------------------------------------------------------------------
# 2. Plan-level validation: coverage, uniqueness, palette
# ---------------------------------------------------------------------------


def validate_plan_coverage(artifacts: Artifacts) -> dict[str, Any]:
    """Prove the plan is internally complete and matches the receipt's denominators."""

    plan = artifacts.visual_plan
    palette = _require_mapping(plan.get("palette"), "visual plan palette")
    for role, color in sorted(palette.items()):
        _rgb(color, f"palette role {role!r}")

    scatter_panels = list(_require_list(plan.get("scatter_panels"), "scatter_panels"))
    crop_panels = list(_require_list(plan.get("crop_panels"), "crop_panels"))
    panels = [*scatter_panels, *crop_panels]
    if not panels:
        _fail("the visual plan names no panel at all")

    panel_ids = _require_unique(
        [_require_str(_require_mapping(p, "panel").get("panel_id"), "panel_id") for p in panels],
        "visual plan panel_id",
    )
    output_files = _require_unique(
        [_require_str(_require_mapping(p, "panel").get("output_file"), "output_file") for p in panels],
        "visual plan output_file",
    )
    for name in output_files:
        if Path(name).name != name or not name.endswith(".png"):
            _fail(f"visual plan output_file {name!r} is not a plain .png file name")

    declared_panel_count = _require_int(plan.get("panel_count"), "panel_count")
    if declared_panel_count != len(panels):
        _fail(
            f"visual plan declares panel_count {declared_panel_count} but names {len(panels)} "
            "panels; the plan's visual coverage is incomplete"
        )
    declared_outputs = [
        _require_str(name, "expected_output_files entry")
        for name in _require_list(plan.get("expected_output_files"), "expected_output_files")
    ]
    if sorted(declared_outputs) != sorted(output_files):
        _fail(
            "visual plan expected_output_files do not exactly cover the panel output files; "
            "the plan's visual coverage is incomplete"
        )

    # Scatter grid: every arm the analyzer binds, times both continuous measures.
    expected_scatter = {
        (arm, measure) for arm, _row_key, _label, _role in analyzer.ARMS for measure in SCATTER_MEASURES
    }
    observed_scatter = {
        (
            _require_str(_require_mapping(panel, "scatter panel").get("arm"), "scatter arm"),
            _require_str(
                _require_mapping(
                    _require_mapping(panel, "scatter panel").get("x_axis"), "scatter x_axis"
                ).get("field"),
                "scatter x_axis.field",
            ),
        )
        for panel in scatter_panels
    }
    if observed_scatter != expected_scatter:
        _fail(
            "visual plan scatter panels do not cover every arm x measure pair "
            f"(missing={sorted(expected_scatter - observed_scatter)!r}, "
            f"unexpected={sorted(observed_scatter - expected_scatter)!r})"
        )

    # Per-arm point counts must equal the owner-row denominators the receipt sealed.
    arm_row_count_key = {
        analyzer.PRIMARY_ARM: "primary_ce_row_count",
        analyzer.SENSITIVITY_ARM: "sensitivity_cf_row_count",
    }
    for panel in scatter_panels:
        arm = str(panel["arm"])
        key = arm_row_count_key.get(arm)
        if key is None:
            _fail(f"scatter panel {panel['panel_id']!r} carries unknown arm {arm!r}")
        sealed = _require_int(artifacts.denominators.get(key), f"denominators.{key}")
        points = _require_list(panel.get("points"), f"{panel['panel_id']} points")
        if len(points) != sealed:
            _fail(
                f"scatter panel {panel['panel_id']!r} plots {len(points)} points, not the "
                f"{sealed} owner rows the receipt sealed for {key}; visual coverage is incomplete"
            )

    material_negative_panels = [
        panel
        for panel in crop_panels
        if str(_require_mapping(panel, "crop panel").get("panel_kind"))
        == MATERIAL_NEGATIVE_PANEL_KIND
    ]
    greedy_panels = [
        panel
        for panel in crop_panels
        if str(_require_mapping(panel, "crop panel").get("panel_kind")) == GREEDY_PAIR_PANEL_KIND
    ]
    if len(material_negative_panels) + len(greedy_panels) != len(crop_panels):
        _fail("the visual plan carries a crop panel with an unknown panel_kind")

    declared_material = _require_int(
        plan.get("material_negative_case_count"), "material_negative_case_count"
    )
    if declared_material != len(material_negative_panels):
        _fail(
            f"visual plan declares {declared_material} material-negative cases but names "
            f"{len(material_negative_panels)} crop panels; visual coverage is incomplete"
        )
    declared_greedy = _require_int(plan.get("greedy_pair_panel_count"), "greedy_pair_panel_count")
    sealed_greedy = _require_int(
        artifacts.denominators.get("greedy_displacement_pair_count"),
        "denominators.greedy_displacement_pair_count",
    )
    if declared_greedy != len(greedy_panels) or declared_greedy != sealed_greedy:
        _fail(
            f"visual plan names {len(greedy_panels)} greedy-pair crop panels and declares "
            f"{declared_greedy}, but the receipt sealed {sealed_greedy} pairs; visual coverage "
            "is incomplete"
        )

    # Every material-negative scatter point needs its own crop panel.
    crop_panel_ids = set(
        str(_require_mapping(panel, "crop panel")["panel_id"]) for panel in crop_panels
    )
    required: set[str] = set()
    for panel in scatter_panels:
        for point in _require_list(panel.get("points"), f"{panel['panel_id']} points"):
            point = _require_mapping(point, "scatter point")
            if _require_bool(point.get("material_negative"), "point.material_negative"):
                required.add(f"crop:{panel['arm']}:{_require_str(point.get('gt_owner_id'), 'point.gt_owner_id')}")
    uncovered = sorted(required - crop_panel_ids)
    if uncovered:
        _fail(
            f"the visual plan marks material-negative rows with no crop panel: {uncovered!r}; "
            "visual coverage is incomplete"
        )

    return {
        "panel_ids": panel_ids,
        "output_files": output_files,
        "scatter_panel_count": len(scatter_panels),
        "crop_panel_count": len(crop_panels),
        "material_negative_case_count": len(material_negative_panels),
        "greedy_pair_panel_count": len(greedy_panels),
    }


# ---------------------------------------------------------------------------
# 3. Norm-1000 -> pixel projection (display only)
# ---------------------------------------------------------------------------


def project_norm1000_to_pixel(value: float, extent: int) -> float:
    """Display-only projection; no geometric measure in this unit depends on it."""

    if extent <= 0:
        _fail(f"cannot project into a nonpositive image extent {extent}")
    return float(value) * float(extent) / float(analyzer.CANVAS_BINS)


def _validate_box_bins(values: Any, label: str) -> tuple[int, int, int, int]:
    items = list(_require_list(values, label))
    if len(items) != 4:
        _fail(f"{label} is not a 4-tuple of norm-1000 bins")
    x1, y1, x2, y2 = (_require_int(item, f"{label} component") for item in items)
    _assert_bin_window(x1, y1, x2, y2, label)
    return x1, y1, x2, y2


def _assert_bin_window(x1: int, y1: int, x2: int, y2: int, label: str) -> None:
    for name, value in (("x1", x1), ("y1", y1), ("x2", x2), ("y2", y2)):
        if not 0 <= value <= analyzer.CANVAS_BINS:
            _fail(f"{label} {name}={value} is outside the norm-1000 canvas")
    if x2 <= x1 or y2 <= y1:
        _fail(f"{label} is not a valid box: it requires x2 > x1 and y2 > y1")


def _crop_window_bins(value: Any, label: str) -> tuple[int, int, int, int]:
    window = _require_mapping(value, label)
    x1, y1, x2, y2 = (
        _require_int(window.get(key), f"{label}.{key}") for key in ("x1", "y1", "x2", "y2")
    )
    _assert_bin_window(x1, y1, x2, y2, label)
    return x1, y1, x2, y2


def _pixel_window(
    window: tuple[int, int, int, int], *, width: int, height: int
) -> tuple[int, int, int, int]:
    x1 = int(math.floor(project_norm1000_to_pixel(window[0], width)))
    y1 = int(math.floor(project_norm1000_to_pixel(window[1], height)))
    x2 = int(math.ceil(project_norm1000_to_pixel(window[2], width)))
    y2 = int(math.ceil(project_norm1000_to_pixel(window[3], height)))
    x1, x2 = max(0, min(x1, width - 1)), min(width, max(x2, 1))
    y1, y2 = max(0, min(y1, height - 1)), min(height, max(y2, 1))
    if x2 <= x1 or y2 <= y1:
        _fail("the projected crop window collapsed to zero pixels")
    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# 4. Media resolution and verification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Media:
    image_id: str
    media_root: str
    media_root_source: str
    media_resolution_rule: str
    file_name: str
    path: Path
    sha256: str
    width: int
    height: int


#: The owning artifact declares a root and a root-relative ``file_name``, so the
#: direct join is the only rule that normally applies.  A superseded plan shape
#: declared a ``media_root`` that already ended with ``file_name``'s leading
#: directories; the one compatibility candidate below strips exactly that overlap.
#: Both candidates are always probed and *exactly one* must exist, so an ambiguous
#: layout fails closed instead of silently choosing a root.
RULE_DECLARED_ROOT_JOIN = "declared_root_join"
RULE_DECLARED_ROOT_OVERLAP_STRIPPED = "declared_root_overlap_stripped"


def _media_candidates(root: Path, relative: Path) -> list[tuple[str, Path]]:
    candidates: list[tuple[str, Path]] = [(RULE_DECLARED_ROOT_JOIN, root / relative)]
    root_parts, relative_parts = root.parts, relative.parts
    for overlap in range(len(relative_parts) - 1, 0, -1):
        if root_parts[-overlap:] == relative_parts[:overlap]:
            candidates.append(
                (RULE_DECLARED_ROOT_OVERLAP_STRIPPED, root.joinpath(*relative_parts[overlap:]))
            )
            break
    return candidates


def _resolve_media(
    block: Mapping[str, Any], *, image_root: Path | None, cache: dict[str, Media]
) -> Media:
    image_id = _require_str(block.get("image_id"), "crop panel image.image_id")
    file_name = _require_str(block.get("file_name"), f"image {image_id} file_name")
    width = _require_int(block.get("image_width"), f"image {image_id} image_width")
    height = _require_int(block.get("image_height"), f"image {image_id} image_height")
    if width <= 0 or height <= 0:
        _fail(f"image {image_id} declares a nonpositive extent {width}x{height}")
    plan_root = _require_str(block.get("media_root"), f"image {image_id} media_root")
    root = Path(image_root) if image_root is not None else Path(plan_root)
    source = "cli_image_root_override" if image_root is not None else "visual_plan_media_root"

    relative = Path(file_name)
    if relative.is_absolute() or any(part == ".." for part in relative.parts):
        _fail(f"image {image_id} file_name {file_name!r} escapes its media root")

    candidates = _media_candidates(root, relative)
    resolved = {str(path): rule for rule, path in candidates if path.is_file()}
    if not resolved:
        _fail(
            f"media file for image {image_id} is missing; no declared-root candidate exists "
            f"({[str(path) for _rule, path in candidates]!r})"
        )
    if len(resolved) > 1:
        _fail(
            f"media root for image {image_id} is ambiguous: {sorted(resolved)!r} both exist, so "
            "no single declared-root resolution is provable"
        )
    path_text, rule = next(iter(resolved.items()))
    path = Path(path_text)

    cached = cache.get(image_id)
    if cached is not None:
        if cached.path != path or cached.width != width or cached.height != height:
            _fail(f"image {image_id} is declared with two different media identities")
        return cached

    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as handle:
            size = handle.size
    except (OSError, UnidentifiedImageError) as exc:
        _fail(f"media file for image {image_id} is unreadable at {path}: {exc}")
    if tuple(size) != (width, height):
        _fail(
            f"media file for image {image_id} at {path} is {size[0]}x{size[1]} pixels, not the "
            f"sealed {width}x{height}"
        )

    media = Media(
        image_id=image_id,
        media_root=str(root),
        media_root_source=source,
        media_resolution_rule=rule,
        file_name=file_name,
        path=path,
        sha256=sha256_file(path),
        width=width,
        height=height,
    )
    cache[image_id] = media
    return media


# ---------------------------------------------------------------------------
# 5. Pure panel specs
# ---------------------------------------------------------------------------


def _scatter_dimensions() -> dict[str, int]:
    return {
        "width": SCATTER_WIDTH,
        "height": SCATTER_HEIGHT,
        "margin_left": SCATTER_MARGIN_LEFT,
        "margin_right": SCATTER_MARGIN_RIGHT,
        "margin_top": SCATTER_MARGIN_TOP,
        "margin_bottom": SCATTER_MARGIN_BOTTOM,
        "plot_width": SCATTER_WIDTH - SCATTER_MARGIN_LEFT - SCATTER_MARGIN_RIGHT,
        "plot_height": SCATTER_HEIGHT - SCATTER_MARGIN_TOP - SCATTER_MARGIN_BOTTOM,
    }


def _crop_dimensions(
    *, legend_count: int, annotation_count: int, placement: Mapping[str, Any]
) -> dict[str, int]:
    footer = CROP_MARGIN * 2 + CROP_LINE_HEIGHT * (legend_count + annotation_count + 2)
    panel_height = int(placement["panel_height"])
    return {
        "width": CROP_WIDTH,
        "height": CROP_HEADER + panel_height + footer,
        "header": CROP_HEADER,
        "panel_width": CROP_PANEL_WIDTH,
        "panel_height": panel_height,
        "footer": footer,
        "margin": CROP_MARGIN,
        "line_height": CROP_LINE_HEIGHT,
    }


def _panel_placement(window: Sequence[int]) -> dict[str, Any]:
    """Aspect-preserving placement of the pixel crop; the panel height follows the crop."""

    crop_width = int(window[2]) - int(window[0])
    crop_height = int(window[3]) - int(window[1])
    if crop_width <= 0 or crop_height <= 0:
        _fail("the projected crop window collapsed to zero pixels")
    scale = CROP_PANEL_WIDTH / crop_width
    if crop_height * scale > CROP_PANEL_MAX_HEIGHT:
        scale = CROP_PANEL_MAX_HEIGHT / crop_height
    draw_width = max(1, int(round(crop_width * scale)))
    draw_height = max(1, int(round(crop_height * scale)))
    panel_height = max(CROP_PANEL_MIN_HEIGHT, draw_height)
    return {
        "scale": scale,
        "draw_width": draw_width,
        "draw_height": draw_height,
        "panel_height": panel_height,
        "offset_x": CROP_MARGIN + (CROP_PANEL_WIDTH - draw_width) // 2,
        "offset_y": CROP_HEADER + (panel_height - draw_height) // 2,
        "aspect_preserved": True,
    }


def _axis_window(values: Sequence[float], references: Sequence[float]) -> dict[str, float]:
    pool = [*values, *references]
    if not pool:
        return {"min": 0.0, "max": 1.0}
    low, high = min(pool), max(pool)
    if high - low <= 0.0:
        pad = abs(high) * 0.5 if high != 0.0 else 0.5
        return {"min": low - pad, "max": high + pad}
    pad = (high - low) * 0.08
    return {"min": low - pad, "max": high + pad}


def build_scatter_spec(panel: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one plan scatter panel and freeze exactly what will be drawn."""

    panel_id = _require_str(panel.get("panel_id"), "scatter panel_id")
    x_axis = _require_mapping(panel.get("x_axis"), f"{panel_id} x_axis")
    y_axis = _require_mapping(panel.get("y_axis"), f"{panel_id} y_axis")
    measure = _require_str(x_axis.get("field"), f"{panel_id} x_axis.field")
    if measure not in SCATTER_MEASURES:
        _fail(f"{panel_id} plots unknown geometry measure {measure!r}")
    expected_id = f"scatter:{panel['arm']}:{measure}"
    if panel_id != expected_id:
        _fail(f"scatter panel_id {panel_id!r} does not match its arm/measure ({expected_id!r})")

    references: list[dict[str, Any]] = []
    for line in _require_list(panel.get("reference_lines"), f"{panel_id} reference_lines"):
        line = _require_mapping(line, f"{panel_id} reference line")
        axis = _require_str(line.get("axis"), f"{panel_id} reference line axis")
        if axis not in ("x", "y"):
            _fail(f"{panel_id} reference line names unknown axis {axis!r}")
        references.append(
            {
                "axis": axis,
                "value": _require_number(line.get("value"), f"{panel_id} reference line value"),
                "label": _require_str(line.get("label"), f"{panel_id} reference line label"),
            }
        )

    points: list[dict[str, Any]] = []
    for raw in _require_list(panel.get("points"), f"{panel_id} points"):
        raw = _require_mapping(raw, f"{panel_id} point")
        point_id = _require_str(raw.get("point_id"), f"{panel_id} point_id")
        owner_id = _require_str(raw.get("gt_owner_id"), f"{point_id} gt_owner_id")
        color = _require_str(raw.get("color"), f"{point_id} color")
        _rgb(color, f"{point_id} color")
        points.append(
            {
                "point_id": point_id,
                "image_id": _require_str(raw.get("image_id"), f"{point_id} image_id"),
                "gt_owner_id": owner_id,
                "x": _require_number(raw.get("x"), f"{point_id} x"),
                "y": _require_number(raw.get("y"), f"{point_id} y"),
                "geometric_label": _require_str(
                    raw.get("geometric_label"), f"{point_id} geometric_label"
                ),
                "description_axis": _require_str(
                    raw.get("description_axis"), f"{point_id} description_axis"
                ),
                "material_negative": _require_bool(
                    raw.get("material_negative"), f"{point_id} material_negative"
                ),
                "color": color,
                "display_label": owner_id.rsplit(":", 1)[-1],
            }
        )
    _require_unique([point["point_id"] for point in points], f"{panel_id} point_id")

    return {
        "schema_version": VISUAL_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "panel_id": panel_id,
        "panel_kind": "scatter",
        "output_file": _require_str(panel.get("output_file"), f"{panel_id} output_file"),
        "arm": str(panel["arm"]),
        "arm_role": _require_str(panel.get("arm_role"), f"{panel_id} arm_role"),
        "measure": measure,
        "x_axis": {
            "field": measure,
            "label": _require_str(x_axis.get("label"), f"{panel_id} x_axis.label"),
        },
        "y_axis": {
            "field": _require_str(y_axis.get("field"), f"{panel_id} y_axis.field"),
            "label": _require_str(y_axis.get("label"), f"{panel_id} y_axis.label"),
        },
        "reference_lines": references,
        "point_ids": [point["point_id"] for point in points],
        "points": points,
        "axis_windows": {
            "x": _axis_window(
                [point["x"] for point in points],
                [line["value"] for line in references if line["axis"] == "x"],
            ),
            "y": _axis_window(
                [point["y"] for point in points],
                [line["value"] for line in references if line["axis"] == "y"],
            ),
        },
        "value_source": "visual plan points only; no geometry or delta is recomputed here",
        "dimensions": _scatter_dimensions(),
    }


def build_crop_spec(
    panel: Mapping[str, Any], *, palette: Mapping[str, Any], image_root: Path | None, cache: dict[str, Media]
) -> dict[str, Any]:
    """Validate one plan crop panel, resolve its media, and freeze the overlay."""

    panel_id = _require_str(panel.get("panel_id"), "crop panel_id")
    panel_kind = _require_str(panel.get("panel_kind"), f"{panel_id} panel_kind")
    arm = _require_str(panel.get("arm"), f"{panel_id} arm")
    if panel_kind == GREEDY_PAIR_PANEL_KIND and arm != GREEDY_PAIR_ARM:
        _fail(f"{panel_id} is a greedy-pair panel but carries arm {arm!r}")
    if panel_kind == MATERIAL_NEGATIVE_PANEL_KIND and arm not in {
        entry[0] for entry in analyzer.ARMS
    }:
        _fail(f"{panel_id} is a material-negative panel but carries unknown arm {arm!r}")

    media = _resolve_media(
        _require_mapping(panel.get("image"), f"{panel_id} image"),
        image_root=image_root,
        cache=cache,
    )
    window = _crop_window_bins(
        panel.get("crop_window_norm1000_xyxy"), f"{panel_id} crop_window_norm1000_xyxy"
    )

    boxes: list[dict[str, Any]] = []
    for raw in _require_list(panel.get("boxes"), f"{panel_id} boxes"):
        raw = _require_mapping(raw, f"{panel_id} box")
        box_id = _require_str(raw.get("box_id"), f"{panel_id} box_id")
        role = _require_str(raw.get("role"), f"{box_id} role")
        color = _require_str(raw.get("color"), f"{box_id} color")
        _rgb(color, f"{box_id} color")
        if role not in palette:
            _fail(f"{box_id} carries role {role!r} which the plan palette does not define")
        if color != str(palette[role]):
            _fail(f"{box_id} color {color!r} is not the plan palette color for role {role!r}")
        bins = _validate_box_bins(raw.get("norm1000_xyxy"), f"{box_id} norm1000_xyxy")
        if not (
            window[0] <= bins[0] and window[1] <= bins[1]
            and bins[2] <= window[2] and bins[3] <= window[3]
        ):
            _fail(f"{box_id} lies outside the panel's own crop window")
        boxes.append(
            {
                "box_id": box_id,
                "role": role,
                "legend": _require_str(raw.get("legend"), f"{box_id} legend"),
                "norm1000_xyxy": list(bins),
                "color": color,
            }
        )
    if len(boxes) != 2:
        _fail(f"{panel_id} carries {len(boxes)} boxes; every panel of this unit renders a pair")
    _require_unique([box["box_id"] for box in boxes], f"{panel_id} box_id")

    roles = frozenset(box["role"] for box in boxes)
    if panel_kind == GREEDY_PAIR_PANEL_KIND:
        if roles != GREEDY_PAIR_ROLE_SET:
            _fail(f"{panel_id} does not pair a displacement target with its displacer")
    elif roles not in MATERIAL_NEGATIVE_ROLE_SETS:
        _fail(f"{panel_id} does not pair inserted owner C with its downstream row")
    if boxes[0]["color"] == boxes[1]["color"]:
        _fail(f"{panel_id} draws both roles in the same color; the roles would be indistinguishable")

    annotations = _require_mapping(panel.get("annotations"), f"{panel_id} annotations")
    annotation_lines = [
        {"key": key, "text": _format_annotation(annotations[key])} for key in sorted(annotations)
    ]
    pixel_window = _pixel_window(window, width=media.width, height=media.height)
    placement = _panel_placement(pixel_window)

    return {
        "schema_version": VISUAL_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "panel_id": panel_id,
        "panel_kind": "crop",
        "crop_kind": panel_kind,
        "output_file": _require_str(panel.get("output_file"), f"{panel_id} output_file"),
        "arm": arm,
        "arm_role": _require_str(panel.get("arm_role"), f"{panel_id} arm_role"),
        "image": {
            "image_id": media.image_id,
            "file_name": media.file_name,
            "media_root": media.media_root,
            "media_root_source": media.media_root_source,
            "media_resolution_rule": media.media_resolution_rule,
            "resolved_path": str(media.path),
            "media_file_sha256": media.sha256,
            "image_width": media.width,
            "image_height": media.height,
        },
        "crop_window_norm1000_xyxy": list(window),
        "crop_window_pixel_xyxy": list(pixel_window),
        "panel_placement": placement,
        "box_ids": [box["box_id"] for box in boxes],
        "boxes": boxes,
        "annotations": dict(annotations),
        "annotation_lines": annotation_lines,
        "value_source": (
            "visual plan boxes and annotations only; no geometry, label or decision is "
            "recomputed here"
        ),
        "projection": (
            "norm-1000 bins multiplied by the sealed image extent for display only"
        ),
        "dimensions": _crop_dimensions(
            legend_count=len(boxes),
            annotation_count=len(annotation_lines),
            placement=placement,
        ),
    }


def _format_annotation(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail("a plan annotation carries a non-finite number")
        return f"{value:.4f}"
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def build_panel_specs(
    artifacts: Artifacts, *, image_root: Path | None
) -> list[dict[str, Any]]:
    plan = artifacts.visual_plan
    palette = _require_mapping(plan.get("palette"), "visual plan palette")
    cache: dict[str, Media] = {}
    specs = [
        build_scatter_spec(_require_mapping(panel, "scatter panel"))
        for panel in _require_list(plan.get("scatter_panels"), "scatter_panels")
    ]
    specs.extend(
        build_crop_spec(
            _require_mapping(panel, "crop panel"),
            palette=palette,
            image_root=image_root,
            cache=cache,
        )
        for panel in _require_list(plan.get("crop_panels"), "crop_panels")
    )
    return specs


# ---------------------------------------------------------------------------
# 6. Rendering (PIL imported lazily so the specs stay importable without it)
# ---------------------------------------------------------------------------


def _font(size: int) -> Any:
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow older than 10.1 has no sized default font
        return ImageFont.load_default()


def render_scatter_png(spec: Mapping[str, Any]) -> bytes:
    from PIL import Image, ImageDraw

    dims = spec["dimensions"]
    width, height = int(dims["width"]), int(dims["height"])
    left, top = int(dims["margin_left"]), int(dims["margin_top"])
    plot_width, plot_height = int(dims["plot_width"]), int(dims["plot_height"])
    x_window, y_window = spec["axis_windows"]["x"], spec["axis_windows"]["y"]

    image = Image.new("RGB", (width, height), BACKGROUND_RGB)
    draw = ImageDraw.Draw(image)
    title_font, label_font = _font(14), _font(12)

    def px(value: float) -> float:
        span = float(x_window["max"]) - float(x_window["min"]) or 1.0
        return left + (float(value) - float(x_window["min"])) / span * plot_width

    def py(value: float) -> float:
        span = float(y_window["max"]) - float(y_window["min"]) or 1.0
        return top + plot_height - (float(value) - float(y_window["min"])) / span * plot_height

    draw.text((12, 10), f"{spec['panel_id']}  ({spec['arm_role']} arm)", fill=FOREGROUND_RGB, font=title_font)
    draw.text(
        (12, 30),
        f"{len(spec['points'])} owner rows; values are the analyzer's own, never recomputed",
        fill=MUTED_RGB,
        font=label_font,
    )
    draw.rectangle([left, top, left + plot_width, top + plot_height], outline=AXIS_RGB)

    for line in spec["reference_lines"]:
        value = float(line["value"])
        if line["axis"] == "y":
            y = py(value)
            draw.line([(left, y), (left + plot_width, y)], fill=REFERENCE_RGB, width=1)
            draw.text(
                (left + plot_width - 230, y - 14),
                f"{line['label']} = {value:.3f}",
                fill=REFERENCE_RGB,
                font=label_font,
            )
        else:
            x = px(value)
            draw.line([(x, top), (x, top + plot_height)], fill=REFERENCE_RGB, width=1)
            draw.text((x + 4, top + 4), f"{line['label']} = {value:.3f}", fill=REFERENCE_RGB, font=label_font)

    for point in spec["points"]:
        x, y = px(point["x"]), py(point["y"])
        color = _rgb(point["color"], "point color")
        radius = 5
        box = [x - radius, y - radius, x + radius, y + radius]
        if point["material_negative"]:
            draw.ellipse(box, fill=color, outline=FOREGROUND_RGB)
        else:
            draw.ellipse(box, outline=color, width=2)
        draw.text((x + radius + 2, y - 6), str(point["display_label"])[:18], fill=MUTED_RGB, font=label_font)

    for fraction in (0.0, 0.5, 1.0):
        x_value = float(x_window["min"]) + fraction * (float(x_window["max"]) - float(x_window["min"]))
        y_value = float(y_window["min"]) + fraction * (float(y_window["max"]) - float(y_window["min"]))
        draw.text((px(x_value) - 14, top + plot_height + 6), f"{x_value:.3f}", fill=MUTED_RGB, font=label_font)
        draw.text((6, py(y_value) - 6), f"{y_value:.3f}", fill=MUTED_RGB, font=label_font)

    draw.text(
        (left, top + plot_height + 28), str(spec["x_axis"]["label"]), fill=FOREGROUND_RGB, font=label_font
    )
    draw.text((12, top - 18), str(spec["y_axis"]["label"]), fill=FOREGROUND_RGB, font=label_font)
    draw.text(
        (left, height - 20),
        "filled marker = material_negative; hollow = nonmaterial",
        fill=MUTED_RGB,
        font=label_font,
    )

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def render_crop_png(spec: Mapping[str, Any]) -> bytes:
    from PIL import Image, ImageDraw

    dims = spec["dimensions"]
    width, height = int(dims["width"]), int(dims["height"])
    header, margin, line_height = (
        int(dims["header"]),
        int(dims["margin"]),
        int(dims["line_height"]),
    )
    window = [int(value) for value in spec["crop_window_pixel_xyxy"]]
    placement = spec["panel_placement"]
    scale = float(placement["scale"])
    offset_x, offset_y = int(placement["offset_x"]), int(placement["offset_y"])
    draw_width, draw_height = int(placement["draw_width"]), int(placement["draw_height"])

    figure = Image.new("RGB", (width, height), BACKGROUND_RGB)
    draw = ImageDraw.Draw(figure)
    title_font, label_font = _font(14), _font(12)

    with Image.open(spec["image"]["resolved_path"]) as handle:
        source = handle.convert("RGB")
    crop = source.crop((window[0], window[1], window[2], window[3]))
    figure.paste(
        crop.resize((draw_width, draw_height), Image.Resampling.LANCZOS), (offset_x, offset_y)
    )

    def projected(bins: Sequence[int]) -> list[float]:
        image_width = int(spec["image"]["image_width"])
        image_height = int(spec["image"]["image_height"])
        return [
            offset_x + (project_norm1000_to_pixel(bins[0], image_width) - window[0]) * scale,
            offset_y + (project_norm1000_to_pixel(bins[1], image_height) - window[1]) * scale,
            offset_x + (project_norm1000_to_pixel(bins[2], image_width) - window[0]) * scale,
            offset_y + (project_norm1000_to_pixel(bins[3], image_height) - window[1]) * scale,
        ]

    draw.text((12, 8), str(spec["panel_id"]), fill=FOREGROUND_RGB, font=title_font)
    draw.text(
        (12, 28),
        f"image {spec['image']['image_id']} | {spec['image']['file_name']} | "
        f"{spec['crop_kind']} | {spec['arm_role']}",
        fill=MUTED_RGB,
        font=label_font,
    )
    draw.text(
        (12, 46),
        "boxes are the analyzer's sealed norm-1000 geometry, projected for display only",
        fill=MUTED_RGB,
        font=label_font,
    )

    for index, box in enumerate(spec["boxes"]):
        color = _rgb(box["color"], f"{box['box_id']} color")
        rectangle = projected(box["norm1000_xyxy"])
        draw.rectangle(rectangle, outline=color, width=4 if index == 0 else 3)
        chip_x = min(max(float(offset_x), rectangle[0]), float(width - 120))
        chip_y = max(float(offset_y), rectangle[1] - 16.0)
        draw.rectangle([chip_x, chip_y, chip_x + 116, chip_y + 15], fill=color)
        draw.text((chip_x + 3, chip_y + 2), str(box["box_id"])[:20], fill=(12, 12, 14), font=label_font)

    y = header + int(dims["panel_height"]) + margin
    draw.text((12, y), "roles", fill=FOREGROUND_RGB, font=label_font)
    y += line_height
    for box in spec["boxes"]:
        color = _rgb(box["color"], f"{box['box_id']} color")
        draw.rectangle([12, y + 3, 26, y + 13], fill=color)
        draw.text((32, y), f"{box['box_id']} [{box['role']}] {box['legend']}"[:118], fill=FOREGROUND_RGB, font=label_font)
        y += line_height
    draw.text((12, y), "analyzer annotations (verbatim)", fill=FOREGROUND_RGB, font=label_font)
    y += line_height
    for line in spec["annotation_lines"]:
        draw.text((32, y), f"{line['key']} = {line['text']}"[:118], fill=MUTED_RGB, font=label_font)
        y += line_height

    buffer = io.BytesIO()
    figure.save(buffer, format="PNG")
    return buffer.getvalue()


def render_panel(spec: Mapping[str, Any]) -> bytes:
    if spec["panel_kind"] == "scatter":
        return render_scatter_png(spec)
    return render_crop_png(spec)


# ---------------------------------------------------------------------------
# 7. Manifest, index and create-or-identical publish
# ---------------------------------------------------------------------------


def build_manifest(
    *,
    artifacts: Artifacts,
    coverage: Mapping[str, Any],
    specs: Sequence[Mapping[str, Any]],
    rendered: Mapping[str, bytes],
    output_dir: Path,
    image_root: Path | None,
    specs_only: bool,
    output_files: Mapping[str, bytes],
) -> dict[str, Any]:
    panels: list[dict[str, Any]] = []
    for spec in specs:
        output_file = str(spec["output_file"])
        payload = rendered.get(output_file)
        panels.append(
            {
                "panel_id": str(spec["panel_id"]),
                "panel_kind": str(spec["panel_kind"]),
                "crop_kind": spec.get("crop_kind"),
                "arm": str(spec["arm"]),
                "arm_role": str(spec["arm_role"]),
                "image_id": (spec.get("image") or {}).get("image_id"),
                "media_file_sha256": (spec.get("image") or {}).get("media_file_sha256"),
                "media_resolved_path": (spec.get("image") or {}).get("resolved_path"),
                "media_resolution_rule": (spec.get("image") or {}).get("media_resolution_rule"),
                "point_ids": list(spec.get("point_ids", [])),
                "box_ids": list(spec.get("box_ids", [])),
                "spec_sha256": sha256_json(spec),
                "dimensions": dict(spec["dimensions"]),
                "output_relative_path": output_file if payload is not None else None,
                "rendered": payload is not None,
                "byte_size": None if payload is None else len(payload),
                "sha256": None if payload is None else sha256_bytes(payload),
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "visualizer_source_sha256": sha256_file(Path(__file__).resolve()),
        "analysis_dir": str(artifacts.analysis_dir),
        "output_dir": str(output_dir),
        "image_root_override": None if image_root is None else str(image_root),
        "specs_only": bool(specs_only),
        "authority": (
            "the analyzer's sealed visual plan; no geometry, stratum, materiality verdict or "
            "decision is recomputed here"
        ),
        "reads_model": False,
        "uses_gpu": False,
        "analysis_receipt_content_sha256": artifacts.receipt_content_sha256,
        "visual_plan_sha256": artifacts.visual_plan_sha256,
        "visual_plan_schema_version": analyzer.VISUAL_PLAN_SCHEMA_VERSION,
        "analysis_receipt_schema_version": analyzer.RECEIPT_SCHEMA_VERSION,
        "input_file_digests": {
            name: dict(entry) for name, entry in sorted(artifacts.input_file_digests.items())
        },
        "coverage": dict(coverage),
        "panel_count": len(panels),
        "panels": panels,
        "output_file_digests": {
            name: {"byte_size": len(payload), "sha256": sha256_bytes(payload)}
            for name, payload in sorted(output_files.items())
        },
        "publish_contract": "atomic_staging_directory_rename_or_byte_identical_no_op",
    }
    manifest["manifest_content_sha256"] = sha256_json(manifest)
    return manifest


def render_index_markdown(manifest: Mapping[str, Any]) -> str:
    lines = [
        "# Sorted crossing owner-row geometry panels",
        "",
        f"Unit: `{manifest['unit_id']}`",
        "",
        f"- analysis directory: `{manifest['analysis_dir']}`",
        f"- visual plan sha256: `{manifest['visual_plan_sha256']}`",
        f"- analysis receipt content sha256: `{manifest['analysis_receipt_content_sha256']}`",
        f"- specs only: `{str(bool(manifest['specs_only'])).lower()}`",
        f"- panels: `{manifest['panel_count']}`",
        "",
        "| panel_id | kind | arm | output | sha256 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for panel in manifest["panels"]:
        output = panel["output_relative_path"] or "(not rendered)"
        digest = panel["sha256"] or "-"
        kind = panel["crop_kind"] or panel["panel_kind"]
        lines.append(
            f"| `{panel['panel_id']}` | {kind} | {panel['arm']} | `{output}` | `{digest}` |"
        )
    lines.append("")
    lines.append("Every value drawn above is the analyzer's own; this renderer decides nothing.")
    lines.append("")
    return "\n".join(lines)


def publish_visualization(output_dir: Path, files: Mapping[str, bytes]) -> dict[str, Any]:
    """Create-or-identical publish: a rerun is a no-op, a drift fails closed."""

    output_dir = Path(output_dir)
    if output_dir.exists():
        present = sorted(entry.name for entry in output_dir.iterdir())
        missing = sorted(set(files) - set(present))
        unexpected = sorted(set(present) - set(files))
        differing = sorted(
            name
            for name in files
            if name in present and (output_dir / name).read_bytes() != files[name]
        )
        if not missing and not unexpected and not differing:
            return {
                "output_dir": str(output_dir),
                "published": False,
                "publish_mode": "no_op_identical_rerun",
                "file_names": sorted(files),
            }
        _fail(
            f"refusing to publish into existing visualization directory {output_dir}: it is not "
            f"a byte-identical rerun (missing={missing!r}, differing={differing!r}, "
            f"unexpected={unexpected!r}); the existing directory is left untouched"
        )
    staging = output_dir.parent / f"{output_dir.name}.staging-{os.getpid()}"
    if staging.exists():
        _fail(f"staging directory {staging} already exists")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True)
    published = False
    try:
        for name in sorted(files):
            (staging / name).write_bytes(files[name])
        os.rename(staging, output_dir)
        published = True
    finally:
        if not published:
            for child in sorted(staging.iterdir()):
                child.unlink()
            staging.rmdir()
    return {
        "output_dir": str(output_dir),
        "published": True,
        "publish_mode": "atomic_staging_directory_rename",
        "file_names": sorted(files),
    }


# ---------------------------------------------------------------------------
# 8. Orchestration and CLI
# ---------------------------------------------------------------------------


def build_output_files(
    artifacts: Artifacts,
    *,
    output_dir: Path,
    image_root: Path | None,
    specs_only: bool,
) -> dict[str, bytes]:
    coverage = validate_plan_coverage(artifacts)
    specs = build_panel_specs(artifacts, image_root=image_root)
    if [str(spec["panel_id"]) for spec in specs] != list(coverage["panel_ids"]):
        _fail("the validated panel specs do not cover every plan panel exactly once")

    rendered: dict[str, bytes] = {}
    if not specs_only:
        for spec in specs:
            rendered[str(spec["output_file"])] = render_panel(spec)
        if sorted(rendered) != sorted(coverage["output_files"]):
            _fail("the rendered panels do not cover the plan's expected output files")

    files: dict[str, bytes] = dict(rendered)
    files[PANEL_SPECS_NAME] = b"".join(canonical_json_bytes(spec) + b"\n" for spec in specs)
    manifest = build_manifest(
        artifacts=artifacts,
        coverage=coverage,
        specs=specs,
        rendered=rendered,
        output_dir=output_dir,
        image_root=image_root,
        specs_only=specs_only,
        output_files=files,
    )
    index_bytes = render_index_markdown(manifest).encode("utf-8")
    files[INDEX_NAME] = index_bytes
    manifest = build_manifest(
        artifacts=artifacts,
        coverage=coverage,
        specs=specs,
        rendered=rendered,
        output_dir=output_dir,
        image_root=image_root,
        specs_only=specs_only,
        output_files=files,
    )
    files[MANIFEST_NAME] = canonical_json_bytes(manifest) + b"\n"
    return files


def run_visualization(
    *,
    analysis_dir: Path,
    output_dir: Path,
    image_root: Path | None = None,
    specs_only: bool = False,
) -> dict[str, Any]:
    artifacts = load_artifacts(analysis_dir)
    files = build_output_files(
        artifacts, output_dir=output_dir, image_root=image_root, specs_only=specs_only
    )
    published = publish_visualization(output_dir, files)
    manifest = json.loads(files[MANIFEST_NAME].decode("utf-8"))
    return {
        "published": published,
        "specs_only": bool(specs_only),
        "panel_count": int(manifest["panel_count"]),
        "rendered_panel_count": sum(1 for panel in manifest["panels"] if panel["rendered"]),
        "manifest_content_sha256": str(manifest["manifest_content_sha256"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        required=True,
        help="Analysis directory published by analyze_sorted_crossing_owner_row_geometry.py",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help=(
            "Optional override for the media root; by default each panel's own sealed "
            "media_root and file_name are used"
        ),
    )
    parser.add_argument(
        "--specs-only",
        action="store_true",
        help="Validate everything and publish specs/manifest/index without rendering any PNG",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_visualization(
            analysis_dir=Path(args.analysis_dir),
            output_dir=Path(args.output_dir),
            image_root=None if args.image_root is None else Path(args.image_root),
            specs_only=bool(args.specs_only),
        )
    except (VisualContractError, analyzer.GeometryContractError) as exc:
        raise SystemExit(f"visualization contract violated: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
