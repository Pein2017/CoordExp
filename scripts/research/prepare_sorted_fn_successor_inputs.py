#!/usr/bin/env python3
"""CPU-only successor planner: FN mechanism registry -> production-v2 inputs.

Converts the frozen, immutable output of ``build_sorted_fn_mechanism_registry.py``
(an arbitrary-role owner/context registry) into the exact three artifacts
accepted, unchanged, by ``build_sorted_owner_basin_candidates.py``:

* ``owner-context-ledger.jsonl``
* ``landscape-decision-rules.json``
* ``candidate-bank-seeds.json``

plus a fourth, new artifact -- a score-independent fixed-budget ``L0``/``L1``
candidate lattice (``fixed-budget-candidates.jsonl``) consumed unchanged by
``reanalyze_sorted_fn_fixed_budget_controls.py`` -- and a companion
foil-geometry review file that gives the emitted background/scan foil seeds
real, dereferenceable provenance.

This module performs no model, tokenizer, or GPU load and no candidate
scoring.  The registry and the optional ``--canonical-description-registry``
supplement must not carry any score/logit/likelihood-derived field and are
rejected outright if they do.  An optional ``--predecessor-score-rows``
artifact is the one deliberate exception: real predecessor score-row files
(e.g. a merged ``landscape_scores.v1`` scoring run) legitimately carry
score/logit/repetition-penalty-policy payloads alongside candidate identity,
and are accepted, but purely for a diagnostic-only summary
(``summarize_predecessor_score_rows``): this planner never claims a
predecessor score is reusable from ``gt_owner_id`` + box-token geometry
alone, since raw likelihood is a function of the entire executed prefix and
real predecessor rows carry no literal source-identity field to verify that
safely.  No fixed-budget candidate row ever carries a
``predecessor_candidate_id`` or an overwritten ``source_digest``/
``raw_row_identity``; the successor scorer freshly scores every candidate.
Each row is passed through a strict, fixed identity-only projection
(``parse_predecessor_identity_row``) that reads only ``candidate_id``, the
complete box's coordinate tokens, ``gt_owner_id``, and optionally
``context_id``/``raw_row_identity``/``source_digest``/
``prefix_token_ids_sha256``; every score, logit, logprob, or
repetition-penalty policy value in that row is never read and never copied
into any planning output.  A row with no complete-box token field (e.g. a
partial conditional y1-scan-plan row) is skipped, not an error; a row that
carries box tokens but a missing or malformed identity field fails fast.
Every prefix
token id is copied literally from the registry's own ``prefix.token_ids``
(reconstructed, in turn, only from stored rollout arrays); this module never
decodes or re-tokenizes text.  The only exception is canonical-description
text, which the unchanged candidate builder requires pre-tokenized and
digest-bound per owner: this planner never tokenizes it itself, and instead
requires every needed description text to already be present in the supplied
rules-template's ``owner_canonical_descriptions`` map (keyed by description
text, not owner id, since tokenization is a pure function of the text) or in
an optional ``--canonical-description-registry`` supplement produced once,
offline, by the same frozen tokenizer.  A missing description text is a
fail-fast error, never a silent default.

The rules/ledger/bank-seeds documents are built by *specializing* an existing,
already-production-validated ``landscape-decision-rules.json`` template: the
entire model-invariant landscape policy (bank definitions, proposal measures,
spatial-clustering/shape rules, token registry, structural row wrapper,
score channels) is carried over verbatim, and only the per-owner canonical
descriptions and the foil-set membership are extended for the new roles.

Fixed-budget lattice ladder (frozen, non-uniform; see unit.md):

* ``L0``: 30 target candidates (within the 24-40 band) + 30 family-mirrored
  decoys, diagnostic/non-decision;
* ``L1``: exactly 256 target candidates (>= 64 in the strict IoU>=0.5 region
  by construction) + 256 family-mirrored decoys;
* ``L2``: not materialized by default (metadata/template only; a named
  successor decision is required to escalate).
* ``scalar_smoke``: used instead of the L0/L1 ladder for any owner-context
  whose cohort is ``strict_rescued`` (no dense predecessor reference exists
  for that control kind on this panel); dense-score reuse is refused for it
  by the downstream reanalysis consumer.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
from typing import Any

if __package__ in {None, ""}:  # Allow ``python scripts/research/...py``.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.build_sorted_fn_mechanism_registry import (  # noqa: E402
    load_owner_index,
    load_rollout_trajectories,
)
from scripts.research.build_sorted_owner_basin_candidates import (  # noqa: E402
    BANK_SEEDS_SCHEMA_VERSION,
    OFFICIAL_COCO_CATEGORY_IDS,
    OFFICIAL_COCO_NAMESPACE,
    OWNER_CONTEXT_LEDGER_SCHEMA_VERSION,
)
from scripts.research.prepare_sorted_owner_basin_inputs import _extent_grid  # noqa: E402
from scripts.research.sorted_owner_basin_landscape import (  # noqa: E402
    SEMANTIC_CORE_OUTER_EXECUTION_FIELDS,
    SEMANTIC_CORE_SCHEMA_VERSION,
    build_semantic_core_payload,
    semantic_core_payload as _validate_semantic_core_payload,
)
from scripts.research.sorted_owner_basin_landscape import (  # noqa: E402
    _canonical_json_digest as _semantic_core_json_digest,
)

SCHEMA_VERSION = "sorted-fn-successor-input-plan.v2"
FIXED_BUDGET_SCHEMA_VERSION = "sorted-fn-successor-fixed-budget-candidates.v2"
RECEIPT_SCHEMA_VERSION = "sorted-fn-successor-input-plan-receipt.v2"
FOIL_REVIEW_SCHEMA_VERSION = "sorted-fn-successor-foil-geometry-review.v1"
MECHANISM_DECISION_RULES_SCHEMA_VERSION = "sorted-fn-mechanism-decision-rules.v1"
UNIT_ID = "2026-08-02-sorted-false-negative-mechanism-decomposition"

# Fixed-budget candidate rows carry an explicit schema_version so a stale v1
# row (missing the neighborhood/reference/parent-binding fields introduced in
# v2) fails fast in the downstream reanalysis consumer instead of silently
# being read as if it had them.
_STALE_FIXED_BUDGET_SCHEMA_VERSIONS = frozenset({"sorted-fn-successor-fixed-budget-candidates.v1"})

COORDINATE_MIN = 0
COORDINATE_MAX = 999

_COHORT_CONTROL_KIND = {
    "strict_rescued": "strict_rescue",
    "greedy_strict_present": "strict_positive",
    "loose_only_b1": "loose_only",
}
_NO_DENSE_REFERENCE_COHORTS = frozenset({"strict_rescued"})

Box = tuple[int, int, int, int]


class SuccessorInputPlanError(ValueError):
    """Raised before any output when a proposed successor input is invalid."""


# --------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SuccessorInputPlanError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SuccessorInputPlanError(f"{label} must be an array")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise SuccessorInputPlanError(f"{label} must be a non-empty trimmed string")
    return value


def _resolved_file(path: str | Path, label: str) -> Path:
    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise SuccessorInputPlanError(f"{label} does not exist") from exc
    if not resolved.is_file():
        raise SuccessorInputPlanError(f"{label} must be a regular file")
    return resolved


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        return dict(_mapping(json.loads(path.read_text(encoding="utf-8")), label))
    except json.JSONDecodeError as exc:
        raise SuccessorInputPlanError(f"{label} is not valid JSON") from exc


_FORBIDDEN_FIELD_TOKENS = ("score", "logit", "logprob", "likelihood", "peak", "prominence")
_FORBIDDEN_FIELD_NAMES = frozenset(
    {"other_owner_margin", "localized_rank", "target_bank_score", "old_score", "reused_score"}
)


def _reject_score_derived_fields(value: Any, *, label: str) -> None:
    """Reject any score/logit/likelihood-derived field anywhere in an input.

    Applied only to the registry and the canonical-description-registry
    supplement, which must never carry a score. Predecessor score-row
    artifacts are validated separately by ``parse_predecessor_identity_row``,
    which reads a fixed identity-only field allowlist instead of scanning
    (and rejecting) the whole row, since those rows are expected to legitimately
    carry scores alongside identity.
    """

    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in _FORBIDDEN_FIELD_NAMES or any(
                token in lowered for token in _FORBIDDEN_FIELD_TOKENS
            ):
                raise SuccessorInputPlanError(
                    f"{label} contains forbidden score-derived field {key!r}; this "
                    "input must not carry model score/logit/likelihood fields"
                )
            _reject_score_derived_fields(item, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_score_derived_fields(item, label=f"{label}[{index}]")


# --------------------------------------------------------------------------
# Geometry (coordinate-bin space, 0..999, independent of image pixel size)
# --------------------------------------------------------------------------


def _clamp_bin(value: float) -> int:
    return max(COORDINATE_MIN, min(COORDINATE_MAX, round(value)))


def make_box(x1: float, y1: float, x2: float, y2: float) -> Box:
    bx1, by1, bx2, by2 = _clamp_bin(x1), _clamp_bin(y1), _clamp_bin(x2), _clamp_bin(y2)
    if bx2 <= bx1:
        bx2 = bx1 + 1 if bx1 < COORDINATE_MAX else COORDINATE_MAX
        bx1 = bx2 - 1
    if by2 <= by1:
        by2 = by1 + 1 if by1 < COORDINATE_MAX else COORDINATE_MAX
        by1 = by2 - 1
    return (bx1, by1, bx2, by2)


def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = iw * ih
    if intersection <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _translate_scale_box(gt: Box, *, dx_frac: float, dy_frac: float, scale: float) -> Box:
    x1, y1, x2, y2 = gt
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w, h = (x2 - x1) * scale, (y2 - y1) * scale
    dx, dy = dx_frac * (x2 - x1), dy_frac * (y2 - y1)
    return make_box(cx + dx - w / 2.0, cy + dy - h / 2.0, cx + dx + w / 2.0, cy + dy + h / 2.0)


def _linspace(lo: float, hi: float, n: int) -> list[float]:
    if n == 1:
        return [(lo + hi) / 2.0]
    step = (hi - lo) / (n - 1)
    return [lo + step * i for i in range(n)]


def classify_region(iou_value: float) -> str:
    if iou_value >= 0.5:
        return "target_strict"
    if iou_value > 0.0:
        return "target_halo"
    return "background"


_MICRO_COUNT_BY_RUNG = {"L0": 6, "L1": 64}


def _near_micro_boxes(anchor_box: Box, count: int) -> list[Box]:
    """Deterministic micro-perturbations of ``anchor_box`` (score-independent).

    Pure function of ``anchor_box`` and ``count`` alone. Local index 0 is
    *always* the literal, unperturbed ``anchor_box`` -- not a small
    perturbation that merely tends to round back to it: for large boxes, the
    old formula's fractional-bin offsets do not reliably round-trip to the
    exact integer box (verified against real production owners, e.g.
    gt:7511:2's 193x483 box has zero exact matches under pure perturbation),
    so this is a score-independent geometry guarantee, not a rounding
    accident. Indices ``1..count-1`` keep the original small
    offset/scale perturbation formula unchanged. This guarantees the
    ``near_gt_micro`` family always contains a literal-GT-box row regardless
    of box size, without adding or removing any row (``count`` is unchanged).
    Shared by the ``near_gt_micro`` (anchor: the target GT box) and
    ``near_other_micro`` (anchor: a bound same-description non-overlapping
    owner's box) families so both use byte-identical logic; the exact-anchor
    guarantee is harmless (merely redundant, never required) for
    ``near_other_micro``, whose own explicit trailing exact entry already
    covers that family's requirement.
    """

    boxes: list[Box] = [anchor_box]
    for i in range(1, count):
        dx = (((i * 0.6180339887) % 1.0) - 0.5) * 0.10
        dy = (((i * 0.4142135624) % 1.0) - 0.5) * 0.10
        scale = 1.0 + ((((i * 0.3660254038) % 1.0)) - 0.5) * 0.06
        boxes.append(_translate_scale_box(anchor_box, dx_frac=dx, dy_frac=dy, scale=scale))
    return boxes


def build_family_boxes(gt_box: Box, rung: str) -> list[tuple[str, Box]]:
    """Deterministic, score-independent target-family lattice for one rung.

    Every family is a pure function of ``gt_box`` (never of a candidate
    score). ``near_gt_micro`` uses offsets/scales small enough that, even for
    a one-bin-wide box, it collapses to the exact GT box (IoU 1.0, always
    strict); this guarantees the frozen L1 strict-region minimum (>= 64)
    regardless of box size.
    """

    if rung not in {"L0", "L1", "scalar_smoke"}:
        raise SuccessorInputPlanError(f"unrecognized rung {rung!r} for family-box generation")
    if rung == "scalar_smoke":
        rung = "L0"

    boxes: list[tuple[str, Box]] = []
    if rung == "L0":
        micro_count, fine_n, wide_n, scale_n, aspect_n, interior = 6, 0, 8, 6, 4, 2
    else:
        micro_count, fine_n, wide_n, scale_n, aspect_n, interior = 64, 7, 9, 15, 15, 32

    for micro_box in _near_micro_boxes(gt_box, micro_count):
        boxes.append(("near_gt_micro", micro_box))

    if fine_n:
        fine_fracs = _linspace(-0.15, 0.15, fine_n)
        for dx, dy in itertools.product(fine_fracs, fine_fracs):
            boxes.append(("translate_fine", _translate_scale_box(gt_box, dx_frac=dx, dy_frac=dy, scale=1.0)))
    else:
        # L0 has no dedicated translate_fine family; corner-anchored shifts
        # fill the small/near role instead.
        for dx, dy in ((-0.3, -0.3), (0.3, -0.3), (-0.3, 0.3), (0.3, 0.3)):
            boxes.append(("corner_shift", _translate_scale_box(gt_box, dx_frac=dx, dy_frac=dy, scale=1.0)))

    if rung == "L0":
        wide_pairs = [(-0.5, 0.0), (0.5, 0.0), (0.0, -0.5), (0.0, 0.5), (-0.35, -0.35), (0.35, 0.35), (-0.35, 0.35), (0.35, -0.35)]
    else:
        wide_fracs = _linspace(-0.5, 0.5, wide_n)
        wide_pairs = list(itertools.product(wide_fracs, wide_fracs))
    for dx, dy in wide_pairs:
        boxes.append(("translate_wide", _translate_scale_box(gt_box, dx_frac=dx, dy_frac=dy, scale=1.0)))

    for scale in _linspace(0.5, 1.5, scale_n):
        boxes.append(("scale", _translate_scale_box(gt_box, dx_frac=0.0, dy_frac=0.0, scale=scale)))

    for ws in _linspace(0.6, 1.4, aspect_n):
        hs = max(0.6, min(1.4, 2.0 - ws))
        x1, y1, x2, y2 = gt_box
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        w, h = (x2 - x1) * ws, (y2 - y1) * hs
        boxes.append(("aspect", make_box(cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)))

    x1, y1, x2, y2 = gt_box
    w, h = x2 - x1, y2 - y1
    if rung == "L0":
        interior_positions = [(0.35, 0.5), (0.65, 0.5)]
    else:
        interior_positions = list(
            itertools.product(_linspace(0.15, 0.85, 8), _linspace(0.15, 0.85, 4))
        )
    for fx, fy in interior_positions[:interior]:
        cx, cy = x1 + fx * w, y1 + fy * h
        boxes.append(
            (
                "interior_lattice",
                make_box(cx - 0.125 * w, cy - 0.125 * h, cx + 0.125 * w, cy + 0.125 * h),
            )
        )

    return boxes


_REFERENCE_MICRO_COUNT_BY_RUNG = {"L0": 6, "L1": 64}


def build_reference_family_boxes(other_owner_box: Box, rung: str) -> list[tuple[str, Box]]:
    """Score-independent ``near_other_micro`` lattice plus one exact reference box.

    A pure function of ``other_owner_box`` (the bound same-description,
    zero-overlap owner's own box) alone -- never of a candidate score, and
    never of the target owner's own geometry. Reuses the exact same
    micro-perturbation algorithm as ``near_gt_micro`` (:func:`_near_micro_boxes`)
    so both families are generated identically, just centered on a different
    anchor. Returns 7 boxes for ``L0``/``scalar_smoke`` (6 perturbed +
    1 exact) and 65 for ``L1`` (64 perturbed + 1 exact).
    """

    if rung not in {"L0", "L1", "scalar_smoke"}:
        raise SuccessorInputPlanError(f"unrecognized rung {rung!r} for reference-box generation")
    micro_count = _REFERENCE_MICRO_COUNT_BY_RUNG["L0" if rung == "scalar_smoke" else rung]
    boxes: list[tuple[str, Box]] = [
        ("near_other_micro", micro_box) for micro_box in _near_micro_boxes(other_owner_box, micro_count)
    ]
    boxes.append(("near_other_micro", other_owner_box))
    return boxes


def mirror_decoy_box(box: Box, gt_box: Box) -> Box:
    """An equal-size background decoy, deterministically placed clear of GT."""

    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    shift_x, shift_y = w * 3 + 60, h * 3 + 60
    for dx_sign, dy_sign in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        nx1 = _clamp_bin(x1 + dx_sign * shift_x)
        ny1 = _clamp_bin(y1 + dy_sign * shift_y)
        candidate = make_box(nx1, ny1, nx1 + w, ny1 + h)
        if iou(candidate, gt_box) == 0.0 and iou(candidate, box) == 0.0:
            return candidate
    raise SuccessorInputPlanError(
        "could not place a non-overlapping equal-size background decoy for "
        f"target box {box}; the GT box is too large relative to the coordinate range"
    )


# --------------------------------------------------------------------------
# Panel loading (Task-0 predecessor artifact: image size + coordinate-bin box)
# --------------------------------------------------------------------------


_COORD_TOKEN_RE = re.compile(r"^<\|coord_(\d+)\|>$")


def _coord_bin(value: Any, label: str) -> int:
    """Accept either a literal integer bin or the frozen ``<|coord_N|>`` token
    string; never a decode of free text or any other coordinate encoding."""

    if isinstance(value, str):
        match = _COORD_TOKEN_RE.fullmatch(value)
        if match is None:
            raise SuccessorInputPlanError(f"{label} is not an official coordinate token")
        value = int(match.group(1))
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 999:
        raise SuccessorInputPlanError(f"{label} must be a coordinate bin in 0..999")
    return value


def load_panel(path: Path) -> tuple[dict[str, tuple[int, int]], dict[tuple[str, int], list[int]]]:
    sizes: dict[str, tuple[int, int]] = {}
    boxes: dict[tuple[str, int], list[int]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = _mapping(json.loads(line), f"panel line {line_number}")
        image_id = str(row.get("image_id"))
        width, height = row.get("width"), row.get("height")
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            raise SuccessorInputPlanError(f"panel image {image_id!r} has invalid dimensions")
        sizes[image_id] = (width, height)
        for annotation_index, raw_object in enumerate(
            _sequence(row.get("objects"), f"panel image {image_id}.objects")
        ):
            obj = _mapping(raw_object, f"panel image {image_id} object {annotation_index}")
            bbox = _sequence(obj.get("bbox_2d"), f"panel image {image_id} object {annotation_index}.bbox_2d")
            if len(bbox) != 4:
                raise SuccessorInputPlanError(
                    f"panel image {image_id!r} object {annotation_index} has a malformed bbox_2d"
                )
            boxes[(image_id, annotation_index)] = [
                _coord_bin(v, f"panel image {image_id} object {annotation_index}.bbox_2d[{i}]")
                for i, v in enumerate(bbox)
            ]
    return sizes, boxes


# --------------------------------------------------------------------------
# Reference (near_other_micro) owner selection: nearest same-description,
# zero-overlap physical owner, chosen deterministically -- never scored.
# --------------------------------------------------------------------------

NO_SAME_DESCRIPTION_NON_OVERLAPPING_OWNER = "no_same_description_non_overlapping_owner"


def select_nearest_same_description_owner(
    *,
    target_gt_owner_id: str,
    target_image_id: str,
    target_box: Box,
    normalized_description: str,
    owner_index: Mapping[str, Mapping[str, Any]],
    panel_boxes: Mapping[tuple[str, int], Sequence[int]],
) -> dict[str, Any]:
    """Deterministically bind the nearest same-description, zero-overlap owner.

    Candidate owners: every *other* owner in the same ``image_id`` with the
    exact same ``normalized_description`` and zero geometric overlap with the
    target's own frozen panel bin-space box (never the pixel-space
    owner-ledger box, for consistency with every other geometry decision in
    this planner). Deterministic tie-break, in this exact order:

    1. minimal normalized center distance (Euclidean center distance divided
       by the square root of the target's own box area, removing scale
       dependence without needing the image's pixel dimensions);
    2. then minimal ``abs(log(area_ratio))`` (candidate area vs. target area);
    3. then minimal ``original_annotation_index`` (final deterministic
       tie-break; never arbitrary dict/iteration order).

    Returns ``{"status": "bound", "gt_owner_id", "box", "trace"}`` or
    ``{"status": "no_same_description_non_overlapping_owner"}`` -- the
    absence case is never a fabricated placeholder.
    """

    target_x1, target_y1, target_x2, target_y2 = target_box
    target_area = max(1, (target_x2 - target_x1) * (target_y2 - target_y1))
    scale = math.sqrt(target_area)
    target_center = ((target_x1 + target_x2) / 2.0, (target_y1 + target_y2) / 2.0)

    best_key: tuple[float, float, int] | None = None
    best: dict[str, Any] | None = None
    considered = 0
    for owner_id, owner_row in owner_index.items():
        if owner_id == target_gt_owner_id:
            continue
        if str(owner_row.get("image_id")) != target_image_id:
            continue
        if owner_row.get("normalized_description") != normalized_description:
            continue
        annotation_index = owner_row.get("original_annotation_index")
        if not isinstance(annotation_index, int) or isinstance(annotation_index, bool):
            continue
        candidate_box_bins = panel_boxes.get((target_image_id, annotation_index))
        if candidate_box_bins is None:
            continue
        candidate_box: Box = (
            candidate_box_bins[0],
            candidate_box_bins[1],
            candidate_box_bins[2],
            candidate_box_bins[3],
        )
        if iou(candidate_box, target_box) > 0.0:
            continue
        considered += 1
        cx1, cy1, cx2, cy2 = candidate_box
        candidate_area = max(1, (cx2 - cx1) * (cy2 - cy1))
        candidate_center = ((cx1 + cx2) / 2.0, (cy1 + cy2) / 2.0)
        center_distance = math.hypot(
            candidate_center[0] - target_center[0], candidate_center[1] - target_center[1]
        )
        normalized_center_distance = center_distance / scale
        abs_log_area_ratio = abs(math.log(candidate_area / target_area))
        key = (normalized_center_distance, abs_log_area_ratio, annotation_index)
        if best_key is None or key < best_key:
            best_key = key
            best = {
                "status": "bound",
                "gt_owner_id": owner_id,
                "box": candidate_box,
                "trace": {
                    "normalized_center_distance": normalized_center_distance,
                    "abs_log_area_ratio": abs_log_area_ratio,
                    "candidate_annotation_index": annotation_index,
                    "considered_owner_count": considered,
                },
            }
    if best is None:
        return {"status": NO_SAME_DESCRIPTION_NON_OVERLAPPING_OWNER}
    best["trace"]["considered_owner_count"] = considered
    return best


# --------------------------------------------------------------------------
# Canonical-description resolution (no tokenizer; text-keyed reuse only)
# --------------------------------------------------------------------------


def index_canonical_descriptions_by_text(rules_template: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    by_text: dict[str, dict[str, Any]] = {}
    for _owner_id, block in _mapping(
        rules_template.get("owner_canonical_descriptions"), "rules_template.owner_canonical_descriptions"
    ).items():
        block = _mapping(block, "owner_canonical_descriptions entry")
        by_text[str(block.get("text"))] = dict(block)
    return by_text


def resolve_canonical_descriptions(
    needed: Mapping[str, str], known_by_text: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Resolve ``{gt_owner_id: description_text}`` to full description blocks.

    Every needed text must already be present in ``known_by_text`` (merged
    from the rules template and any supplied supplement); this function never
    invokes a tokenizer. A missing text is a fail-fast error naming exactly
    which text is missing, never a silent default or re-tokenization.
    """

    missing = sorted({text for text in needed.values() if text not in known_by_text})
    if missing:
        raise SuccessorInputPlanError(
            "no pre-tokenized canonical description is available for text(s) "
            f"{missing}; supply --canonical-description-registry with these "
            "texts pre-tokenized by the frozen project tokenizer (this planner "
            "never tokenizes text itself)"
        )
    return {owner_id: dict(known_by_text[text]) for owner_id, text in needed.items()}


# --------------------------------------------------------------------------
# Registry role -> ledger row / cohort classification
# --------------------------------------------------------------------------


def _owner_cohort_index(registry: Mapping[str, Any]) -> dict[str, str]:
    index: dict[str, str] = {}
    for entry in _sequence(
        _mapping(registry.get("mechanism_cohort"), "registry.mechanism_cohort").get("targets"),
        "registry.mechanism_cohort.targets",
    ):
        entry = _mapping(entry, "mechanism_cohort target")
        index[str(entry.get("gt_owner_id"))] = str(entry.get("cohort"))
    for entry in _sequence(
        _mapping(registry.get("context_control_registry"), "registry.context_control_registry").get(
            "bound_non_targets"
        ),
        "registry.context_control_registry.bound_non_targets",
    ):
        entry = _mapping(entry, "bound non-target")
        index[str(entry.get("gt_owner_id"))] = str(entry.get("cohort"))
    return index


def _iter_all_roles(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    smoke = _mapping(registry.get("smoke"), "registry.smoke")
    roles = [dict(_mapping(role, "smoke role")) for role in _sequence(smoke.get("roles"), "registry.smoke.roles")]
    envelope = _mapping(smoke.get("null_pair_envelope"), "registry.smoke.null_pair_envelope")
    for pair in _sequence(envelope.get("pairs", []), "registry.smoke.null_pair_envelope.pairs"):
        pair = _mapping(pair, "null pair")
        for role in _sequence(pair.get("roles"), "null pair.roles"):
            roles.append(dict(_mapping(role, "null pair role")))
    role_ids = [role["role_id"] for role in roles]
    if len(role_ids) != len(set(role_ids)):
        raise SuccessorInputPlanError("registry declares duplicate role_id values")
    return roles


def _matched_control_group(role: Mapping[str, Any]) -> str:
    role_id = str(role["role_id"])
    if role_id.startswith("collision:") or role_id.startswith("null:"):
        parts = role_id.split(":")
        return ":".join(parts[:2])
    role_kind = str(role["role_kind"])
    if role_kind in {"first_skip_pre", "first_skip_post"}:
        return f"first_skip:{role['gt_owner_id']}"
    return str(role["gt_owner_id"])


def _source_pred_row_id(role: Mapping[str, Any]) -> str | None:
    for key in ("pred_row_id", "reference_pred_row_id", "successor_pred_row_id"):
        value = role.get(key)
        if value is not None:
            return str(value)
    return None


def resolve_context_tokens(
    role: Mapping[str, Any], trajectories: Mapping[tuple[str, str, int], Mapping[str, Any]]
) -> dict[str, Any]:
    """Split the registry's literal ``prefix.token_ids`` into prompt/self-prefix.

    The split point is the length of the declared trajectory's own stored
    ``prompt_token_ids`` array (an exact literal-array length lookup, never a
    decode or re-tokenization). Fails if the declared trajectory was not
    supplied, or if the registry's literal prefix does not start with that
    exact stored prompt.
    """

    trajectory_key_raw = _mapping(role.get("trajectory"), f"role {role['role_id']}.trajectory")
    key = (
        str(trajectory_key_raw["image_id"]),
        str(trajectory_key_raw["decode_mode"]),
        int(trajectory_key_raw["seed"]),
    )
    trajectory = trajectories.get(key)
    if trajectory is None:
        raise SuccessorInputPlanError(
            f"role {role['role_id']!r} declares trajectory {key!r}, but no matching "
            "rollout artifact was supplied; cannot establish the exact prompt/"
            "self-prefix boundary without decoding, refusing to infer"
        )
    prompt_token_ids = [int(v) for v in trajectory["prompt_token_ids"]]
    prefix = _mapping(role.get("prefix"), f"role {role['role_id']}.prefix")
    token_ids = [int(v) for v in _sequence(prefix.get("token_ids"), "role.prefix.token_ids")]
    if token_ids[: len(prompt_token_ids)] != prompt_token_ids:
        raise SuccessorInputPlanError(
            f"role {role['role_id']!r} literal prefix does not begin with its "
            "declared trajectory's exact stored prompt tokens; policy mismatch"
        )
    self_prefix_token_ids = token_ids[len(prompt_token_ids) :]
    return {
        "token_ids": token_ids,
        "token_ids_sha256": sha256_json(token_ids),
        "prompt_prefix_token_count": len(prompt_token_ids),
        "prompt_token_ids_sha256": sha256_json(prompt_token_ids),
        "self_prefix_generated_token_ids_sha256": sha256_json(self_prefix_token_ids),
        "split": {"prompt": [0, len(prompt_token_ids)], "self_prefix": [len(prompt_token_ids), len(token_ids)]},
        "copy_semantics": "literal_task6_model_input_token_ids_no_decode_or_retokenize",
    }


def build_ledger_row(
    *,
    role: Mapping[str, Any],
    owner_row: Mapping[str, Any],
    gt_box_bins: Sequence[int],
    image_size: tuple[int, int],
    canonical_description: Mapping[str, Any],
    context_tokens: Mapping[str, Any],
    coordinate_space: Mapping[str, Any],
    vocabulary_attestation: Mapping[str, str],
    token_registry: Mapping[str, Any],
    upstream_digests: Mapping[str, str],
    registry_digest: str,
    other_owner_reference_status: str,
) -> dict[str, Any]:
    gt_owner_id = str(role["gt_owner_id"])
    context_id = f"ctx:fn:{role['role_id']}"
    foreign_keys = {
        key: value
        for key, value in role.items()
        if key not in {"role_id", "role_kind", "gt_owner_id", "prefix", "trajectory", "provenance"}
    }
    source_pred_row_id = _source_pred_row_id(role)
    source_binding = {
        "registry_digest": registry_digest,
        "role_provenance": dict(_mapping(role.get("provenance"), "role.provenance")),
    }
    context_provenance = {
        "context_kind": str(role["role_kind"]),
        "context_status": "admitted",
        "eligibility": f"fn_mechanism:{role['role_kind']}",
        "source_pred_row_id": source_pred_row_id,
        "registry_id": str(role["role_id"]),
        "foreign_keys": foreign_keys,
        "source_binding": source_binding,
        "review_status": "sealed_context",
    }
    source_review_foreign_key_lineage = {
        "source_pred_row_id": source_pred_row_id,
        "review_status": "sealed_context",
        "registry_id": str(role["role_id"]),
        "foreign_keys": foreign_keys,
        "source_binding": source_binding,
    }
    panel_sha256 = _mapping(owner_row.get("source_digests"), "owner_row.source_digests").get("panel")
    width, height = image_size
    runtime_vocabulary_receipt = {
        "model_vocab_size": token_registry["model_vocab_size"],
        "token_registry_sha256": token_registry["registry_sha256"],
        "identity_receipt_digest": token_registry["identity_receipt_digest"],
        **token_registry["vocabulary_attestation"],
    }
    return {
        "schema_version": OWNER_CONTEXT_LEDGER_SCHEMA_VERSION,
        "owner_status": "resolved",
        "diagnostic_owner_id": f"diagnostic:{gt_owner_id}",
        "gt_owner_id": gt_owner_id,
        "context_id": context_id,
        "prompt_prefix_token_count": context_tokens["prompt_prefix_token_count"],
        "native_repetition_penalty_stratum": 1.0,
        "source_pred_row_id": source_pred_row_id,
        "image_id": str(owner_row["image_id"]),
        "image_identity": sha256_json({"image_id": str(owner_row["image_id"]), "panel_sha256": panel_sha256}),
        "image_size": {"width": width, "height": height},
        "ground_truth": {
            "box": [int(v) for v in gt_box_bins],
            "category": {
                "namespace": OFFICIAL_COCO_NAMESPACE,
                "category_id": int(owner_row["official_coco_category_id"]),
            },
        },
        "canonical_description": dict(canonical_description),
        "context_tokens": dict(context_tokens),
        "context_provenance": context_provenance,
        "source_review_foreign_key_lineage": source_review_foreign_key_lineage,
        "coordinate_space": dict(coordinate_space),
        "vocabulary_attestation": dict(vocabulary_attestation),
        "token_registry": dict(token_registry),
        "runtime_vocabulary_receipt": runtime_vocabulary_receipt,
        "upstream_digests": dict(upstream_digests),
        "execution_dedup_key": context_tokens["token_ids_sha256"],
        "other_owner_reference_status": other_owner_reference_status,
    }


# --------------------------------------------------------------------------
# Fixed-budget L0/L1 candidate lattice (own new artifact)
# --------------------------------------------------------------------------


def _box_to_tokens(box: Box, token_registry: Mapping[str, Any]) -> list[int]:
    coordinate = token_registry["coordinate_bin_to_token_id"]
    token_ids = coordinate["coordinate_bin_token_ids"]
    return [int(token_ids[bin_value]) for bin_value in box]


def resolve_rungs(cohort: str | None) -> tuple[bool, str | None, tuple[str, ...]]:
    """Rungs to materialize for one owner-context.

    ``strict_rescue`` (e.g. ``gt:7511:17``) now emits *both* ``scalar_smoke``
    (positive-direction-only; see ``mechanism-decision-rules.json``'s
    ``rung_quotas.scalar_smoke.claim_declaration``) *and* an unconditionally
    frozen ``L1`` -- L0 is still not materialized for this cohort, since it
    has no dense predecessor reference to validate against. L1 is the rung
    scored on marginal or negative evidence before any stop-rule-4 judgment
    (see unit.md stop rule 4).
    """

    if cohort in _NO_DENSE_REFERENCE_COHORTS:
        return True, _COHORT_CONTROL_KIND[cohort], ("scalar_smoke", "L1")
    if cohort in _COHORT_CONTROL_KIND:
        return True, _COHORT_CONTROL_KIND[cohort], ("L0", "L1")
    return False, None, ("L0", "L1")


def build_fixed_budget_candidates_for_context(
    *,
    owner_context_id: str,
    gt_owner_id: str,
    gt_box: Box,
    token_registry: Mapping[str, Any],
    is_control: bool,
    control_kind: str | None,
    matched_control_group: str,
    rung: str,
    mechanism_decision_rules_sha256: str,
    other_owner_reference: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Score-independent fixed-budget geometry only.

    No candidate row here ever claims a predecessor score is reusable from
    owner+box geometry alone: this planner does not carry a
    ``predecessor_candidate_id`` or overwrite ``source_digest``/
    ``raw_row_identity`` on any row, because the successor scorer freshly
    scores every fixed-budget candidate. See
    :func:`summarize_predecessor_score_rows` for the separate,
    diagnostic-only reporting of an optional ``--predecessor-score-rows``
    artifact.

    Every ``target``-population ``near_gt_micro`` row carries
    ``candidate_neighborhood_member=True`` and a stable
    ``candidate_neighborhood_id`` (``nbh:<gt_owner_id>:<rung>:<family>:<local_index>``)
    that is a pure function of the *target owner* + rung + family + local
    index -- never of ``owner_context_id`` -- so it is identical across every
    P/P+G/P+F role sharing the same target owner (see
    :func:`verify_neighborhood_consistency`). Every other row carries the
    field explicitly as ``False``/``None``, never merely omitted.

    Every row also carries ``mechanism_decision_rules_sha256``: the *parent*
    mechanism-decision-rules digest, frozen before any row here is generated
    (see :func:`build_mechanism_decision_rules_document`). This is a
    one-directional child-to-parent binding, not a circular reference: the
    parent rules document is complete and digested first, and never itself
    embeds a candidate-file digest.

    ``other_owner_reference``, when supplied (a bound nearest
    same-normalized-description, zero-overlap owner from
    :func:`select_nearest_same_description_owner`), additionally emits a
    ``population="reference"``, ``region="other_owner"``,
    ``family_id="near_other_micro"`` lattice (7 rows for
    ``L0``/``scalar_smoke``, 65 for ``L1``): not part of the target/decoy
    equal-count pairing.

    Exactly one ``near_gt_micro`` target row per context+rung is marked
    ``exact_gt_singleton_member=True`` (with a stable
    ``exact_gt_singleton_id``, common across P/P+G/P+F since it is a pure
    function of ``gt_owner_id`` + ``rung``): the row whose box is *literally*
    identical to the GT box (IoU 1.0). This marker is discovered by scanning
    every near_gt_micro target row's actual geometry and deterministically
    selecting the lowest local index among the exact matches -- it is never
    hardcoded to a fixed index, even though :func:`_near_micro_boxes`
    guarantees local index 0 is always the literal anchor box (a
    score-independent geometry guarantee: earlier small-perturbation-only
    formulas did not reliably round-trip to the exact integer box for larger
    real owners, e.g. a 193x483 box had zero exact matches, so this cannot be
    left to rounding chance). No new candidates are added; this only tags a
    subset of the existing near_gt_micro population. Every other row (every other near_gt_micro row,
    every other family, every decoy, every reference row) carries the field
    explicitly as ``False``/``None``. Plan construction fails if a
    context+rung's near_gt_micro population contains no exact-GT-box row at
    all, or if the deterministic selection does not resolve to exactly one.
    """

    rows: list[dict[str, Any]] = []
    near_gt_micro_target_rows: list[tuple[int, Box, dict[str, Any]]] = []
    for index, (family_id, target_box) in enumerate(build_family_boxes(gt_box, rung)):
        target_iou = iou(target_box, gt_box)
        target_region = classify_region(target_iou)
        target_tokens = _box_to_tokens(target_box, token_registry)
        target_candidate_id = f"cand:{owner_context_id}:{rung}:{family_id}:{index}:target"
        target_digest = sha256_json(
            {
                "owner_context_id": owner_context_id,
                "rung": rung,
                "family_id": family_id,
                "index": index,
                "population": "target",
                "box": list(target_box),
            }
        )
        is_neighborhood_member = family_id == "near_gt_micro"
        target_row: dict[str, Any] = {
            "schema_version": FIXED_BUDGET_SCHEMA_VERSION,
            "candidate_id": target_candidate_id,
            "coord_token_ids": target_tokens,
            "source_digest": target_digest,
            "owner_context_id": owner_context_id,
            "rung": rung,
            "region": target_region,
            "population": "target",
            "is_control": is_control,
            "control_kind": control_kind,
            "matched_control_group": matched_control_group,
            "family_id": family_id,
            "iou_to_target": target_iou,
            "candidate_neighborhood_member": is_neighborhood_member,
            "candidate_neighborhood_id": (
                f"nbh:{gt_owner_id}:{rung}:{family_id}:{index}" if is_neighborhood_member else None
            ),
            "mechanism_decision_rules_sha256": mechanism_decision_rules_sha256,
            "exact_gt_singleton_member": False,
            "exact_gt_singleton_id": None,
        }
        if is_neighborhood_member:
            near_gt_micro_target_rows.append((index, target_box, target_row))
        rows.append(target_row)

        decoy_box = mirror_decoy_box(target_box, gt_box)
        decoy_tokens = _box_to_tokens(decoy_box, token_registry)
        decoy_candidate_id = f"cand:{owner_context_id}:{rung}:{family_id}:{index}:decoy"
        decoy_digest = sha256_json(
            {
                "owner_context_id": owner_context_id,
                "rung": rung,
                "family_id": family_id,
                "index": index,
                "population": "decoy",
                "box": list(decoy_box),
            }
        )
        decoy_row: dict[str, Any] = {
            "schema_version": FIXED_BUDGET_SCHEMA_VERSION,
            "candidate_id": decoy_candidate_id,
            "coord_token_ids": decoy_tokens,
            "source_digest": decoy_digest,
            "owner_context_id": owner_context_id,
            "rung": rung,
            "region": "background",
            "population": "decoy",
            "is_control": is_control,
            "control_kind": control_kind,
            "matched_control_group": matched_control_group,
            "family_id": family_id,
            "iou_to_target": 0.0,
            "candidate_neighborhood_member": False,
            "candidate_neighborhood_id": None,
            "mechanism_decision_rules_sha256": mechanism_decision_rules_sha256,
            "exact_gt_singleton_member": False,
            "exact_gt_singleton_id": None,
        }
        rows.append(decoy_row)

    if other_owner_reference is not None:
        other_owner_box = other_owner_reference["box"]
        for index, (family_id, reference_box) in enumerate(build_reference_family_boxes(other_owner_box, rung)):
            reference_iou = iou(reference_box, gt_box)
            reference_tokens = _box_to_tokens(reference_box, token_registry)
            reference_candidate_id = f"cand:{owner_context_id}:{rung}:{family_id}:{index}:reference"
            reference_digest = sha256_json(
                {
                    "owner_context_id": owner_context_id,
                    "rung": rung,
                    "family_id": family_id,
                    "index": index,
                    "population": "reference",
                    "box": list(reference_box),
                }
            )
            rows.append(
                {
                    "schema_version": FIXED_BUDGET_SCHEMA_VERSION,
                    "candidate_id": reference_candidate_id,
                    "coord_token_ids": reference_tokens,
                    "source_digest": reference_digest,
                    "owner_context_id": owner_context_id,
                    "rung": rung,
                    "region": "other_owner",
                    "population": "reference",
                    "is_control": is_control,
                    "control_kind": control_kind,
                    "matched_control_group": matched_control_group,
                    "family_id": family_id,
                    "iou_to_target": reference_iou,
                    "candidate_neighborhood_member": False,
                    "candidate_neighborhood_id": None,
                    "mechanism_decision_rules_sha256": mechanism_decision_rules_sha256,
                    "exact_gt_singleton_member": False,
                    "exact_gt_singleton_id": None,
                    "other_owner_gt_owner_id": other_owner_reference["gt_owner_id"],
                    "other_owner_selection_trace": dict(other_owner_reference["trace"]),
                }
            )

    exact_matches = [
        (index, row) for index, box, row in near_gt_micro_target_rows if box == gt_box
    ]
    if not exact_matches:
        raise SuccessorInputPlanError(
            f"owner-context {owner_context_id!r} rung {rung!r}: no near_gt_micro target row "
            "is literally identical to the GT box; F3 (the exact-GT singleton statistic) "
            "requires at least one, and this planner never fabricates one by assuming a "
            "fixed local index"
        )
    singleton_index = min(index for index, _row in exact_matches)
    singleton_id = f"exact-gt:{gt_owner_id}:{rung}"
    selected_rows = [row for index, row in exact_matches if index == singleton_index]
    if len(selected_rows) != 1:
        raise SuccessorInputPlanError(
            f"owner-context {owner_context_id!r} rung {rung!r}: expected exactly one "
            f"exact_gt_singleton_member at local index {singleton_index}, found {len(selected_rows)}"
        )
    selected_rows[0]["exact_gt_singleton_member"] = True
    selected_rows[0]["exact_gt_singleton_id"] = singleton_id

    return rows


def verify_neighborhood_consistency(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Assert identical candidate-geometry digest multisets across P/P+G/P+F,
    and that the F3 exact-GT-singleton geometry/id agrees too.

    Groups fixed-budget rows by (``matched_control_group``, ``rung``) --
    ``matched_control_group`` is shared by every role in a collision or
    null-pair triple (``collision:<pair_id>`` / ``null:<pair_id>``) -- and,
    within any group spanning more than one ``owner_context_id`` (a genuine
    multi-role P/P+G/P+F triple, not a single-context owner), collects the
    exact-box-token multiset of ``near_gt_micro`` target rows per context.
    ``near_gt_micro`` geometry is a pure function of the target owner's own
    GT box (never of context/prefix), so every role in the same triple must
    produce byte-identical geometry; a mismatch is a hard failure, not a
    silent divergence, since it would mean two roles disagree about which
    physical owner (and therefore which GT box) they target.

    The same multi-context groups are also checked for the F3
    exact-GT-singleton marker: both ``exact_gt_singleton_id`` and the
    selected row's own box tokens must agree across every context in the
    group (the id is already a pure function of ``gt_owner_id`` + ``rung``,
    so this is belt-and-braces, not assumed from the id alone).

    Returns a summary for the receipt: how many multi-context groups were
    checked and their status.
    """

    near_gt_micro_groups: dict[tuple[str, str], dict[str, list[tuple[int, ...]]]] = {}
    singleton_groups: dict[tuple[str, str], dict[str, tuple[str, tuple[int, ...]]]] = {}
    for row in rows:
        if row.get("population") != "target" or row.get("family_id") != "near_gt_micro":
            continue
        group_key = (row["matched_control_group"], row["rung"])
        near_gt_micro_groups.setdefault(group_key, {}).setdefault(row["owner_context_id"], []).append(
            tuple(row["coord_token_ids"])
        )
        if row.get("exact_gt_singleton_member"):
            singleton_groups.setdefault(group_key, {})[row["owner_context_id"]] = (
                row["exact_gt_singleton_id"],
                tuple(row["coord_token_ids"]),
            )

    checked = 0
    for (matched_control_group, rung), by_context in near_gt_micro_groups.items():
        if len(by_context) < 2:
            continue
        checked += 1
        multisets = {context_id: sorted(boxes) for context_id, boxes in by_context.items()}
        reference_context, reference_multiset = next(iter(multisets.items()))
        for context_id, multiset in multisets.items():
            if multiset != reference_multiset:
                raise SuccessorInputPlanError(
                    f"matched_control_group {matched_control_group!r} rung {rung!r}: "
                    f"near_gt_micro candidate-geometry digest multiset for context "
                    f"{context_id!r} disagrees with context {reference_context!r}; "
                    "every role in a P/P+G/P+F triple must target byte-identical "
                    "GT-box geometry"
                )

        singleton_by_context = singleton_groups.get((matched_control_group, rung), {})
        missing_singleton = set(by_context) - set(singleton_by_context)
        if missing_singleton:
            raise SuccessorInputPlanError(
                f"matched_control_group {matched_control_group!r} rung {rung!r}: "
                f"context(s) {sorted(missing_singleton)} have no exact_gt_singleton_member "
                "row; every context in a multi-context group must have one"
            )
        reference_singleton_context, reference_singleton = next(iter(singleton_by_context.items()))
        for context_id, singleton in singleton_by_context.items():
            if singleton != reference_singleton:
                raise SuccessorInputPlanError(
                    f"matched_control_group {matched_control_group!r} rung {rung!r}: "
                    f"exact_gt_singleton (id, box) for context {context_id!r} disagrees "
                    f"with context {reference_singleton_context!r}; F3's exact-GT-singleton "
                    "must be identical across every role in a P/P+G/P+F triple"
                )
    return {"multi_context_groups_checked": checked, "status": "consistent"}


# Real predecessor score-row schemas name the *complete* box's coordinate
# tokens differently depending on pipeline; every one of these is a pure
# geometry/identity field, never a score. This deliberately excludes
# ``fixed_coord_token_ids``: that field names a *partial* (e.g. single-token,
# conditional y1-scan-plan) coordinate span, not a complete box, and a row
# carrying only it is not identity-bearing for exact box-candidate reuse.
_PREDECESSOR_BOX_TOKEN_KEYS = ("coord_token_ids", "box_tokens")


def _optional_nonempty_string(row: Mapping[str, Any], key: str, *, label: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise SuccessorInputPlanError(f"{label}.{key} must be a non-empty string when present")
    return value


def parse_predecessor_identity_row(row: Mapping[str, Any], *, index: int) -> dict[str, Any] | None:
    """Strict identity-only projection of one predecessor score row.

    Reads only candidate identity fields -- ``candidate_id``, the complete
    box's coordinate tokens, ``gt_owner_id``, and, when present,
    ``context_id``/``raw_row_identity``/``source_digest``/
    ``prefix_token_ids_sha256`` -- and never any score, logit, logprob, or
    repetition-penalty policy value, even though real predecessor rows (e.g.
    ``landscape_scores.v1``) legitimately carry those alongside identity in
    the same row. Fields outside this fixed allowlist are never read, so
    score leakage into candidate generation cannot happen structurally, not
    by pattern-matching keys.

    ``source_digest`` is read *only* as that literal field name: real
    ``landscape_scores.v1`` complete-box rows carry no ``source_digest`` and
    no ``raw_row_identity`` at all (verified by direct inspection), and
    ``prefix_token_ids_sha256``/``rule_digest`` are distinct fields -- a
    prefix identity and a rules-version identity, respectively -- that must
    never be substituted for a candidate's own source identity; see
    :func:`summarize_predecessor_score_rows`, which never treats a row
    lacking one as reuse-eligible (this planner does not bind reuse at all).

    A row that carries no complete-box token field (e.g. a conditional
    y1-scan plan row) is not identity-bearing for diagnostic purposes and is
    skipped, returning ``None``, not treated as an error. A row that *does*
    carry box tokens but is missing another required identity field, or
    carries a malformed one, is a hard failure: silently dropping or
    guessing it would risk misreporting the diagnostic summary.
    """

    label = f"predecessor score row {index}"
    tokens: Any = None
    for key in _PREDECESSOR_BOX_TOKEN_KEYS:
        if key in row:
            tokens = row[key]
            break
    if tokens is None:
        return None
    if (
        isinstance(tokens, (str, bytes))
        or not isinstance(tokens, Sequence)
        or not tokens
        or any(isinstance(token, bool) or not isinstance(token, int) for token in tokens)
    ):
        raise SuccessorInputPlanError(f"{label} has a malformed complete-box token field")
    candidate_id = row.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise SuccessorInputPlanError(f"{label} is missing a well-formed candidate_id")
    owner_id = row.get("gt_owner_id")
    if not isinstance(owner_id, str) or not owner_id.strip():
        raise SuccessorInputPlanError(f"{label} is missing a well-formed gt_owner_id")
    return {
        "candidate_id": candidate_id,
        "coord_token_ids": [int(token) for token in tokens],
        "gt_owner_id": owner_id,
        "context_id": _optional_nonempty_string(row, "context_id", label=label),
        "raw_row_identity": _optional_nonempty_string(row, "raw_row_identity", label=label),
        "source_digest": _optional_nonempty_string(row, "source_digest", label=label),
        "prefix_token_ids_sha256": _optional_nonempty_string(row, "prefix_token_ids_sha256", label=label),
    }


def summarize_predecessor_score_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Diagnostic-only summary of an optional ``--predecessor-score-rows`` file.

    This planner never claims a predecessor score is reusable from
    ``gt_owner_id`` + box-token geometry alone: raw model likelihood is a
    function of the entire executed prefix, and real predecessor score-row
    artifacts (e.g. a merged ``landscape_scores.v1`` scoring run) carry no
    literal ``source_digest`` and no ``raw_row_identity`` at all (verified
    by direct inspection of a real merged production scores file);
    ``prefix_token_ids_sha256``/``rule_digest`` are a *different* identity
    (which exact prefix was executed / which rules version was active) and
    are never substituted for a candidate's own source identity. No
    fixed-budget candidate row ever carries a ``predecessor_candidate_id``
    or an overwritten ``source_digest``/``raw_row_identity``: the successor
    scorer freshly scores every fixed-budget candidate.

    This function only reports, for transparency: how many rows carried a
    usable complete-box identity (built entirely from
    :func:`parse_predecessor_identity_row`'s strict identity-only
    projection, so no score/logit/policy value ever reaches this summary or
    any candidate generation); how many *distinct* (``gt_owner_id``, box
    tokens) identities were present *within the predecessor file itself*
    (this is not compared against the fixed-budget candidate bank at all,
    so it is not a claim of overlap with anything this planner generates);
    and how many of those rows were "fully qualified" (``context_id`` and a
    literal ``source_digest`` both present -- the minimum information a
    genuine reuse claim would need, though this contract never acts on it).
    ``valid_reuse_count`` is always ``0``.
    """

    rows_with_box_identity = 0
    predecessor_owner_box_identities: set[tuple[str, tuple[int, ...]]] = set()
    fully_qualified_identity_count = 0
    for position, row in enumerate(rows):
        identity = parse_predecessor_identity_row(row, index=position)
        if identity is None:
            continue
        rows_with_box_identity += 1
        predecessor_owner_box_identities.add((identity["gt_owner_id"], tuple(identity["coord_token_ids"])))
        if identity["context_id"] is not None and identity["source_digest"] is not None:
            fully_qualified_identity_count += 1
    return {
        "rows_with_box_identity": rows_with_box_identity,
        "distinct_predecessor_owner_box_identities": len(predecessor_owner_box_identities),
        "fully_qualified_identity_count": fully_qualified_identity_count,
        "valid_reuse_count": 0,
        "reuse_policy": (
            "diagnostic_only: no fixed-budget candidate may claim a reusable "
            "predecessor score from gt_owner_id+box-token geometry alone; the "
            "successor scorer freshly scores every candidate"
        ),
    }


# --------------------------------------------------------------------------
# Successor-owned mechanism-decision-rules.json (own new artifact)
#
# This is the *parent* in the DAG: frozen before any fixed-budget candidate
# row is generated, and read by every one of those rows via
# ``mechanism_decision_rules_sha256``. It therefore cannot itself embed a
# candidate-file digest (the candidates do not exist yet, and doing so would
# be a circular child-into-parent reference); the joint parent+child
# correspondence is instead established once, at the end, by the top-level
# run receipt, which is computed last and can safely reference both this
# document's digest and the (by-then-materialized) candidates file's digest
# side by side. This is the canonical, non-recursive parent-binding: parent
# (this document, self-contained) -> child (candidate rows, each carrying
# the parent's digest) -> receipt (the neutral third party binding both).
# --------------------------------------------------------------------------

DEFAULT_IOU_THRESHOLDS: tuple[float, ...] = (0.4, 0.5, 0.6)
DEFAULT_CALIBRATION_LOWER_QUANTILE = 0.10
DEFAULT_CALIBRATION_UPPER_QUANTILE = 0.90
DEFAULT_NEAR_MISS_UPPER_QUANTILE = 0.90
DEFAULT_SAMPLING_TEMPERATURE = 0.4
DEFAULT_SAMPLING_TOP_P = 0.95
DEFAULT_SAMPLING_REPETITION_PENALTY = 1.0


def build_mechanism_decision_rules_document(
    *,
    execution_landscape_decision_rules_sha256: str,
    fn_mechanism_registry_sha256: str,
    rules_template_sha256: str,
) -> dict[str, Any]:
    """Freeze the successor-owned ``mechanism-decision-rules.json``.

    Every field here is declarative policy (never a computed score); no
    model, tokenizer, or GPU is touched to build this document. See
    unit.md's "Frozen quantitative decision functional" and "Rough cost and
    recursive authority" sections for the frozen source text this
    transcribes into a machine-checkable contract.
    """

    content: dict[str, Any] = {
        "schema_version": MECHANISM_DECISION_RULES_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "geometry": {
            "iou_thresholds": list(DEFAULT_IOU_THRESHOLDS),
            "regions": ["target_strict", "target_halo", "other_owner", "background"],
            "target_strict_iou_min": 0.5,
        },
        "populations": {
            "target": "the exact target box and its score-independent family lattice",
            "decoy": "the equal-count, family-mirrored background/scan population",
            "reference": (
                "the near_other_micro lattice around a deterministically bound "
                "nearest same-normalized-description, zero-overlap owner; never "
                "part of the target/decoy equal-count pairing or the "
                "localized_rank union"
            ),
        },
        "statistics": {
            "target_peak": "the maximum score in the target strict region",
            "background_prominence": (
                "target_peak minus the maximum equal-count, equal-size "
                "background-region score"
            ),
            "other_owner_margin": (
                "target_peak minus the maximum registered same-description "
                "other-owner-region score (populated by the reference/"
                "near_other_micro lattice)"
            ),
            "localized_rank": (
                "the normalized rank of target_peak in the union of the "
                "equal-count target and decoy populations only; reference rows "
                "are excluded from this union"
            ),
            "target_bank_score": (
                "the proposal-weighted regional log-sum-exp, used only when its "
                "target and control proposal measures are comparable"
            ),
        },
        "calibration": {
            "quantile_algorithm": "type7",
            "lower_quantile": DEFAULT_CALIBRATION_LOWER_QUANTILE,
            "upper_quantile": DEFAULT_CALIBRATION_UPPER_QUANTILE,
            "degenerate_population_status": "unresolved",
            "declaration": (
                "a control population too small or degenerate for a well-defined "
                "type7 quantile at the frozen lower/upper bound never falls back "
                "to a pooled or nearby threshold; its status is unresolved"
            ),
        },
        "scalar_tolerance": {
            "source": "measured_scalar_full_reforward_numerical_tolerance",
            "declaration": (
                "every margin (background_prominence, other_owner_margin, "
                "selective_decline) must exceed the measured scalar batch-size-1 "
                "uncached FP32 full-reforward numerical tolerance before it is "
                "treated as decision-bearing"
            ),
        },
        "rung_quotas": {
            "L0": {"target_count_min": 24, "target_count_max": 40, "decoy_count": "equal_to_target"},
            "L1": {
                "target_count": 256,
                "target_strict_region_min": 64,
                "decoy_count": "equal_to_target",
                "reference_count": 65,
            },
            "L2": {
                "target_count": 1024,
                "decoy_count": "equal_to_target",
                "materialized_by_default": False,
                "escalation": (
                    "stratum-local: an L1 dense-reference-validation failure in one "
                    "matched_control_group, or a prospectively declared target "
                    "near-miss for that owner-context; a target-only null result "
                    "never self-authorizes L2"
                ),
            },
            "scalar_smoke": {
                "target_count": 30,
                "decoy_count": "equal_to_target",
                "reference_count": 7,
                "reserved_for": "control_kind == strict_rescue (no dense predecessor reference)",
                "claim_direction": "positive_only",
                "claim_declaration": (
                    "scalar_smoke can only ever support a positive claim (a usable "
                    "target strict-region peak exists); it never by itself "
                    "establishes an absence/negative claim. For strict_rescue "
                    "owner-contexts, the unconditionally frozen L1 rung -- scored "
                    "before any stop-rule-4 judgment -- is what carries marginal or "
                    "negative evidence"
                ),
            },
        },
        "near_miss": {
            "band": (
                "target prominence above the matched decoy 90th percentile but "
                "below the usable threshold"
            ),
            "effect": "permits exactly one L2 rescore for that owner-context only",
            "upper_quantile": DEFAULT_NEAR_MISS_UPPER_QUANTILE,
        },
        "collision": {
            "statistics": {
                "F1": "max score over the complete target_strict region (full target population)",
                "F2": "max score over near_gt_micro target rows that also fall in the target_strict region",
                "F3": (
                    "the score of the row explicitly marked "
                    "exact_gt_singleton_member=true -- the unique near_gt_micro target row "
                    "whose box is literally identical to the GT box (IoU 1.0), discovered by "
                    "scanning the family's actual geometry and deterministically selecting the "
                    "lowest local index among exact matches (never hardcoded to a fixed index, "
                    "even though local index 0 is guaranteed by construction to be the literal "
                    "anchor box, score-independently, for every box size)"
                ),
            },
            "selective_decline_formula": (
                "[target_peak(P + G) - target_peak(P)] - [target_peak(P + F) - target_peak(P)]"
            ),
            "null_envelope": {
                "minimum_witness_count": 3,
                "minimum_distinct_images_or_strata": 2,
                "witnesses": [
                    {
                        "witness_id": "wit-16228-46-a",
                        "role": "calibrating_null",
                        "trajectory": "sampled:21005:16228",
                        "target_gt_owner_id": "gt:16228:46",
                    },
                    {
                        "witness_id": "wit-5001-6-a",
                        "role": "mechanical_sign_join_check",
                        "trajectory": "sampled:21015:5001",
                        "target_gt_owner_id": "gt:5001:6",
                    },
                    {
                        "witness_id": "wit-4134-35-a",
                        "role": "calibrating_null",
                        "trajectory": "sampled:21008:4134",
                        "target_gt_owner_id": "gt:4134:35",
                    },
                ],
                "exceedance_rule": (
                    "the most negative observed null selective_decline, extended by "
                    "measured scalar numerical tolerance; a leave-one-out sign "
                    "readout is robustness reporting, never a replacement threshold"
                ),
            },
            "matched_strata": {
                "declared_gaps": [
                    "no witnessed null for the tiny-person image-7511 stratum",
                    "no bottle null for image 2685",
                ],
                "declaration": (
                    "a collision candidate in a stratum with a declared gap may "
                    "exceed the envelope only as collision-consistent diagnostic "
                    "evidence, never a collision verdict, until a stratum-matched "
                    "witnessed null exists"
                ),
            },
        },
        "neighborhood": {
            "id_rule": "nbh:<gt_owner_id>:<rung>:<family>:<local_index>",
            "member_field": "candidate_neighborhood_member",
            "id_field": "candidate_neighborhood_id",
            "eligible_family": "near_gt_micro",
            "eligible_population": "target",
            "digest_rule": (
                "the near_gt_micro target-row box-token multiset for a given "
                "matched_control_group and rung must be identical across every "
                "owner_context_id (every P/P+G/P+F role) in that group, since "
                "near_gt_micro geometry is a pure function of the target owner's "
                "own GT box, never of context/prefix; verified by "
                "verify_neighborhood_consistency and asserted in the plan receipt"
            ),
        },
        "exact_gt_singleton": {
            "member_field": "exact_gt_singleton_member",
            "id_field": "exact_gt_singleton_id",
            "id_rule": "exact-gt:<gt_owner_id>:<rung>",
            "eligible_family": "near_gt_micro",
            "eligible_population": "target",
            "selection_rule": (
                "the unique near_gt_micro target row whose box is literally identical "
                "to the GT box (IoU 1.0), discovered by scanning actual geometry and "
                "never hardcoded to a fixed index, even though local index 0 is "
                "guaranteed by construction (score-independently) to be the literal "
                "anchor box for every box size -- ties among exact matches (index 0 plus "
                "any perturbed index that also happens to round-trip exactly) are broken "
                "by the lowest local index"
            ),
            "failure_rule": (
                "plan construction fails if a context+rung's near_gt_micro population "
                "contains no exact-GT-box row, or if the deterministic selection does not "
                "resolve to exactly one member"
            ),
            "consistency_rule": (
                "exact_gt_singleton_id and the selected row's own box tokens must be "
                "identical across every owner_context_id (every P/P+G/P+F role) sharing "
                "a matched_control_group and rung; verified by "
                "verify_neighborhood_consistency and asserted in the plan receipt"
            ),
        },
        "behavior_admission": {
            "free_next_row_and_first_divergence_readout": (
                "release the model from the exact self-prefix to observe STOP, "
                "natural description choice, and route selection; the only "
                "readout that may support natural semantic drift"
            ),
            "canonical_description_conditioned_geometry_readout": (
                "force only the canonical description and box opener, then "
                "decode the box greedily; tests localization conditional on the "
                "supplied description and cannot by itself establish semantic drift"
            ),
        },
        "conditional_sampling": {
            "admitted_only_when": (
                "the likelihood landscape contains a usable target strict-region "
                "peak or multiple separated owner-localized peaks but greedy "
                "canonical-description-conditioned decoding does not release the "
                "target"
            ),
            "temperature": DEFAULT_SAMPLING_TEMPERATURE,
            "top_p": DEFAULT_SAMPLING_TOP_P,
            "repetition_penalty": DEFAULT_SAMPLING_REPETITION_PENALTY,
            "k_bound": "selected after the one-case smoke; frozen before execution",
            "null_semantics": "finite_sampling_null_is_absence_neutral",
        },
        "stop_bindings": {
            "source": "unit.md#stop-rules",
            "rule_4": (
                "the strict-rescue control's scalar_smoke exposes no "
                "positive-direction usable target support, and its "
                "unconditionally frozen L1 rung (scored before this stop "
                "judgment, on marginal or negative evidence) also fails to "
                "expose usable target support; scalar_smoke alone, being "
                "positive-direction-only, never establishes this stop by itself"
            ),
        },
        "invalidation_rule": (
            "any rule, tokenizer, model, runtime, control, foil, context, "
            "cohort, sampling, or upstream digest change voids all downstream "
            "scores, calibration, and eligibility receipts bound to this "
            "mechanism-decision-rules document"
        ),
        "superseded_execution_fields": {
            "localized_peak_classifier": "retired; regions are geometry-defined, never a label-first peak-shape classifier",
            "unbounded_dense_full_interior_bank": "retired; replaced by the frozen L0/L1/L2 budget ladder",
            "two_thousand_target_candidate_cap": "not a default rung; requires a named successor decision after L2",
        },
        "upstream_digests": {
            "execution_landscape_decision_rules_sha256": execution_landscape_decision_rules_sha256,
            "fn_mechanism_registry_sha256": fn_mechanism_registry_sha256,
            "rules_template_sha256": rules_template_sha256,
        },
    }
    return {**content, "self_digest": sha256_json(content)}


# --------------------------------------------------------------------------
# Rules / bank-seeds specialization
# --------------------------------------------------------------------------


def _foil_member_content_id(prefix: str, payload: Mapping[str, Any]) -> str:
    return f"{prefix}:sha256:{sha256_json(payload)}"


def build_new_foil_members(
    *,
    ledger_rows: Sequence[Mapping[str, Any]],
    gt_box_by_context: Mapping[str, Box],
    review_artifact_path: str,
    review_artifact_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Background+scan foil members for every new ledger row, plus their boxes.

    Returns ``(members, boxes_by_context)`` where ``boxes_by_context`` maps
    ``context_id -> {"background": box, "scan": box}`` so the same geometry
    is reused for the bank seeds, the fixed-budget lattice decoys, and the
    authored foil-geometry review file.
    """

    members: list[dict[str, Any]] = []
    boxes_by_context: dict[str, dict[str, Any]] = {}
    for row in ledger_rows:
        owner_id = row["gt_owner_id"]
        context_id = row["context_id"]
        gt_box = gt_box_by_context[context_id]
        background_box = mirror_decoy_box(gt_box, gt_box)
        scan_box = mirror_decoy_box(background_box, gt_box)
        boxes_by_context[context_id] = {"background": background_box, "scan": scan_box}
        for bank_name, box in (("background", background_box), ("scan", scan_box)):
            identity_id = f"foil-geometry:{owner_id}:{bank_name}:{context_id}"
            payload = {
                "bank_name": bank_name,
                "box": list(box),
                "context_id": context_id,
                "identity_id": identity_id,
                "identity_kind": "registered_geometry",
                "owner_id": owner_id,
            }
            provenance = {
                "artifact_path": review_artifact_path,
                "artifact_sha256": review_artifact_sha256,
                "source_row_id": f"foil-review:{context_id}:{bank_name}",
            }
            members.append(
                {
                    "foil_member_id": _foil_member_content_id("foil-member", payload),
                    "diagnostic_owner_id": row["diagnostic_owner_id"],
                    "context_id": context_id,
                    "bank_name": bank_name,
                    "source_id": _foil_member_content_id("candidate-source", payload),
                    "identity_kind": "registered_geometry",
                    "identity_id": identity_id,
                    "provenance": provenance,
                }
            )
    return members, boxes_by_context


def _reseal_semantic_core(document: dict[str, Any], *, rules_template: Mapping[str, Any]) -> None:
    """Preserve the landed ``semantic_core`` exactly when unchanged, or rebuild
    and self-verify it with the canonical predecessor functions when our
    owner-description/foil-set additions changed a semantic-core-owned field.

    ``build_semantic_core_payload`` and ``semantic_core_payload`` are the
    unowned, unchanged shared-core functions from
    ``sorted_owner_basin_landscape.py``; this never reimplements their
    digest or cross-binding logic.
    """

    if "semantic_core" not in rules_template:
        return
    template_core = _mapping(rules_template["semantic_core"], "rules_template.semantic_core")
    template_payload = dict(_mapping(template_core.get("payload"), "rules_template.semantic_core.payload"))

    new_payload = build_semantic_core_payload(document)
    if new_payload == template_payload:
        document["semantic_core"] = copy.deepcopy(dict(template_core))
        return

    rebuilt_core = {
        "schema_version": SEMANTIC_CORE_SCHEMA_VERSION,
        "payload": new_payload,
        "sha256": _semantic_core_json_digest(new_payload),
        "excluded_outer_execution_fields": list(SEMANTIC_CORE_OUTER_EXECUTION_FIELDS),
    }
    document["semantic_core"] = rebuilt_core
    try:
        verified_payload = _validate_semantic_core_payload(document)
    except ValueError as exc:
        raise SuccessorInputPlanError(
            f"rebuilt semantic_core failed self-verification against the canonical "
            f"predecessor validator: {exc}"
        ) from exc
    if dict(verified_payload) != new_payload:
        raise SuccessorInputPlanError(
            "rebuilt semantic_core payload does not round-trip through the canonical "
            "predecessor validator"
        )


def specialize_rules_document(
    *,
    rules_template: Mapping[str, Any],
    owner_canonical_descriptions: Mapping[str, Mapping[str, Any]],
    new_foil_members: Sequence[Mapping[str, Any]],
    upstream_digests: Mapping[str, str],
) -> dict[str, Any]:
    document: dict[str, Any] = copy.deepcopy(dict(rules_template))

    merged_descriptions = dict(
        _mapping(document.get("owner_canonical_descriptions"), "rules_template.owner_canonical_descriptions")
    )
    merged_descriptions.update({owner_id: dict(block) for owner_id, block in owner_canonical_descriptions.items()})
    document["owner_canonical_descriptions"] = merged_descriptions

    materializer: dict[str, Any] = dict(
        _mapping(document.get("candidate_materializer"), "rules_template.candidate_materializer")
    )
    template_foil_set = _mapping(materializer.get("foil_set"), "rules_template.candidate_materializer.foil_set")
    template_members = [
        dict(_mapping(m, "template foil member"))
        for m in _sequence(template_foil_set.get("members"), "template foil_set.members")
    ]
    all_members = template_members + [dict(m) for m in new_foil_members]
    seen_ids: set[str] = set()
    deduped_members: list[dict[str, Any]] = []
    for member in all_members:
        if member["foil_member_id"] in seen_ids:
            continue
        seen_ids.add(member["foil_member_id"])
        deduped_members.append(member)
    merged_foil_set = {
        "foil_set_id": template_foil_set["foil_set_id"],
        "members": deduped_members,
        "members_sha256": sha256_json(sorted(deduped_members, key=lambda item: item["foil_member_id"])),
    }
    materializer["foil_set"] = merged_foil_set
    materializer["upstream_digests"] = dict(upstream_digests)
    document["candidate_materializer"] = materializer

    # The scorer's semantic-core cross-check requires the global foil/
    # description registry to stay byte-identical to these growing copies.
    if "global_foil_role_and_description_registry" in document:
        global_registry = dict(
            _mapping(
                document["global_foil_role_and_description_registry"],
                "rules_template.global_foil_role_and_description_registry",
            )
        )
        global_registry["owner_canonical_descriptions"] = merged_descriptions
        global_registry["foil_set"] = merged_foil_set
        document["global_foil_role_and_description_registry"] = global_registry

    _reseal_semantic_core(document, rules_template=rules_template)
    return document


def build_bank_seeds_document(
    *,
    ledger_rows: Sequence[Mapping[str, Any]],
    gt_box_by_context: Mapping[str, Box],
    rules_document: Mapping[str, Any],
    rules_sha256: str,
    ledger_sha256: str,
    new_foil_members: Sequence[Mapping[str, Any]],
    foil_boxes_by_context: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    materializer = rules_document["candidate_materializer"]
    bank_roles = materializer["bank_roles"]
    target_bank_name = materializer["target_bank_name"]
    foil_members_by_key = {
        (m["diagnostic_owner_id"], m["context_id"], m["bank_name"]): m for m in new_foil_members
    }

    seeds: list[dict[str, Any]] = []
    for row in ledger_rows:
        gt_box = gt_box_by_context[row["context_id"]]
        for grid_member in _extent_grid(list(gt_box)):
            extent_id = grid_member["extent_id"]
            extent_submode = (
                "scale_aspect_perturbation"
                if extent_id in {"scale_0p80", "scale_1p20", "aspect_0p80", "aspect_1p20"}
                else extent_id
            )
            box = [0, 0, grid_member["width"], grid_member["height"]]
            target_payload = {
                "bank_name": target_bank_name,
                "box": box,
                "context_id": row["context_id"],
                "identity_id": row["gt_owner_id"],
                "identity_kind": "reviewed_physical_owner",
                "owner_id": row["gt_owner_id"],
                "extent_id": extent_id,
            }
            seeds.append(
                {
                    "bank_name": target_bank_name,
                    "box": box,
                    "context_id": row["context_id"],
                    "diagnostic_owner_id": row["diagnostic_owner_id"],
                    "expansion": "anchor_translate",
                    "extent_submode": extent_submode,
                    "extent_grid_member": grid_member,
                    "identity_id": row["gt_owner_id"],
                    "identity_kind": "reviewed_physical_owner",
                    "role_id": bank_roles[target_bank_name],
                    "source_id": _foil_member_content_id("target-extent", target_payload),
                }
            )
        for bank_name in ("background", "scan"):
            member = foil_members_by_key[(row["diagnostic_owner_id"], row["context_id"], bank_name)]
            box = foil_boxes_by_context[row["context_id"]][bank_name]
            seeds.append(
                {
                    "bank_name": bank_name,
                    "box": list(box),
                    "context_id": row["context_id"],
                    "diagnostic_owner_id": row["diagnostic_owner_id"],
                    "expansion": "exact",
                    "extent_submode": f"equal_size_{bank_name}" if bank_name == "background" else "sorted_scan",
                    "foil_member_id": member["foil_member_id"],
                    "foil_provenance": member["provenance"],
                    "identity_id": member["identity_id"],
                    "identity_kind": "registered_geometry",
                    "role_id": bank_roles[bank_name],
                    "source_id": member["source_id"],
                }
            )

    return {
        "schema_version": BANK_SEEDS_SCHEMA_VERSION,
        "landscape_decision_rules_sha256": rules_sha256,
        "owner_context_ledger_sha256": ledger_sha256,
        "foil_set_sha256": rules_document["candidate_materializer"]["foil_set"]["members_sha256"],
        "seeds": seeds,
    }


# --------------------------------------------------------------------------
# Top-level build
# --------------------------------------------------------------------------


def _write_create_or_identical(path: Path, encoded: bytes) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(encoded)
        return True
    except FileExistsError:
        if not path.is_file() or path.read_bytes() != encoded:
            raise SuccessorInputPlanError(
                f"{path} already exists with different content; refusing to overwrite"
            ) from None
        return False


def prepare_sorted_fn_successor_inputs(
    *,
    registry: str | Path,
    owner_ledger: str | Path,
    panel: str | Path,
    rules_template: str | Path,
    rollouts: Sequence[str | Path],
    canonical_description_registry: str | Path | None,
    predecessor_score_rows: str | Path | None,
    out_dir: str | Path,
) -> dict[str, Any]:
    registry_path = _resolved_file(registry, "registry")
    owner_ledger_path = _resolved_file(owner_ledger, "owner ledger")
    panel_path = _resolved_file(panel, "panel")
    rules_template_path = _resolved_file(rules_template, "rules template")
    rollout_paths = [_resolved_file(p, "rollout artifact") for p in rollouts]

    registry_document = _read_json(registry_path, "registry")
    _reject_score_derived_fields(registry_document, label="registry")
    rules_template_document = _read_json(rules_template_path, "rules template")

    owner_index = load_owner_index(owner_ledger_path)
    panel_sizes, panel_boxes = load_panel(panel_path)
    trajectories = load_rollout_trajectories(rollout_paths)
    cohort_by_owner = _owner_cohort_index(registry_document)

    canonical_description_registry_path: Path | None = None
    known_descriptions_by_text = index_canonical_descriptions_by_text(rules_template_document)
    if canonical_description_registry is not None:
        canonical_description_registry_path = _resolved_file(
            canonical_description_registry, "canonical description registry"
        )
        supplement = _read_json(canonical_description_registry_path, "canonical description registry")
        _reject_score_derived_fields(supplement, label="canonical description registry")
        for text, block in supplement.items():
            known_descriptions_by_text[text] = dict(_mapping(block, f"canonical description registry.{text}"))

    predecessor_reuse_summary: dict[str, Any] = {
        "rows_with_box_identity": 0,
        "distinct_predecessor_owner_box_identities": 0,
        "fully_qualified_identity_count": 0,
        "valid_reuse_count": 0,
        "reuse_policy": (
            "diagnostic_only: no --predecessor-score-rows was supplied"
        ),
    }
    if predecessor_score_rows is not None:
        predecessor_path = _resolved_file(predecessor_score_rows, "predecessor score rows")
        predecessor_rows = [
            json.loads(line)
            for line in predecessor_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        # Predecessor score rows are expected and permitted to carry
        # score/logit/policy fields alongside identity (that is their entire
        # purpose upstream); summarize_predecessor_score_rows only ever
        # reads a fixed identity-field allowlist via
        # parse_predecessor_identity_row and never scans or forwards the
        # rest of the row. This is diagnostic-only: no fixed-budget
        # candidate ever carries a predecessor_candidate_id or an
        # overwritten source_digest/raw_row_identity.
        predecessor_reuse_summary = summarize_predecessor_score_rows(predecessor_rows)

    roles = _iter_all_roles(registry_document)
    if not roles:
        raise SuccessorInputPlanError("registry declares no roles")

    upstream_digests = {
        "fn_mechanism_registry_sha256": registry_document["registry_digest"],
        "rules_template_sha256": sha256_file(rules_template_path),
    }
    if canonical_description_registry_path is not None:
        upstream_digests["canonical_description_registry_sha256"] = sha256_file(
            canonical_description_registry_path
        )
    token_registry = rules_template_document["token_registry"]
    coordinate_space = rules_template_document["candidate_materializer"]["coordinate_space"]
    vocabulary_attestation = token_registry["vocabulary_attestation"]

    needed_descriptions: dict[str, str] = {}
    ledger_rows: list[dict[str, Any]] = []
    gt_box_by_context: dict[str, Box] = {}
    role_meta_by_context: dict[str, dict[str, Any]] = {}

    for role in roles:
        gt_owner_id = str(role["gt_owner_id"])
        owner_row = owner_index.get(gt_owner_id)
        if owner_row is None:
            raise SuccessorInputPlanError(
                f"owner ledger is missing {gt_owner_id!r} referenced by role {role['role_id']!r}"
            )
        image_id = str(owner_row["image_id"])
        original_annotation_index = int(owner_row["original_annotation_index"])
        key = (image_id, original_annotation_index)
        if key not in panel_boxes:
            raise SuccessorInputPlanError(
                f"panel is missing the annotation for {gt_owner_id!r} ({key!r})"
            )
        if image_id not in panel_sizes:
            raise SuccessorInputPlanError(f"panel is missing image size for {image_id!r}")
        official_category_id = int(owner_row["official_coco_category_id"])
        if official_category_id not in OFFICIAL_COCO_CATEGORY_IDS:
            raise SuccessorInputPlanError(f"owner {gt_owner_id!r} has a non-official COCO category id")

        needed_descriptions[gt_owner_id] = str(owner_row["normalized_description"])
        context_tokens = resolve_context_tokens(role, trajectories)
        gt_box_bins = panel_boxes[key]
        gt_box: Box = (gt_box_bins[0], gt_box_bins[1], gt_box_bins[2], gt_box_bins[3])

        context_id = f"ctx:fn:{role['role_id']}"
        gt_box_by_context[context_id] = gt_box
        other_owner_selection = select_nearest_same_description_owner(
            target_gt_owner_id=gt_owner_id,
            target_image_id=image_id,
            target_box=gt_box,
            normalized_description=str(owner_row["normalized_description"]),
            owner_index=owner_index,
            panel_boxes=panel_boxes,
        )
        other_owner_reference_status = (
            f"bound:{other_owner_selection['gt_owner_id']}"
            if other_owner_selection["status"] == "bound"
            else NO_SAME_DESCRIPTION_NON_OVERLAPPING_OWNER
        )
        role_meta_by_context[context_id] = {
            "gt_owner_id": gt_owner_id,
            "matched_control_group": _matched_control_group(role),
            "cohort": cohort_by_owner.get(gt_owner_id),
            "other_owner_reference": (
                other_owner_selection if other_owner_selection["status"] == "bound" else None
            ),
        }
        ledger_rows.append(
            {
                "_role": role,
                "_owner_row": owner_row,
                "_gt_box_bins": gt_box_bins,
                "_image_size": panel_sizes[image_id],
                "_context_tokens": context_tokens,
                "_other_owner_reference_status": other_owner_reference_status,
                "gt_owner_id": gt_owner_id,
                "diagnostic_owner_id": f"diagnostic:{gt_owner_id}",
                "context_id": context_id,
            }
        )

    resolved_descriptions = resolve_canonical_descriptions(needed_descriptions, known_descriptions_by_text)

    final_ledger_rows: list[dict[str, Any]] = []
    for staged in ledger_rows:
        final_ledger_rows.append(
            build_ledger_row(
                role=staged["_role"],
                owner_row=staged["_owner_row"],
                gt_box_bins=staged["_gt_box_bins"],
                image_size=staged["_image_size"],
                canonical_description=resolved_descriptions[staged["gt_owner_id"]],
                context_tokens=staged["_context_tokens"],
                coordinate_space=coordinate_space,
                vocabulary_attestation=vocabulary_attestation,
                token_registry=token_registry,
                upstream_digests=upstream_digests,
                registry_digest=registry_document["registry_digest"],
                other_owner_reference_status=staged["_other_owner_reference_status"],
            )
        )
    final_ledger_rows.sort(key=lambda row: (row["diagnostic_owner_id"], row["context_id"]))

    out_dir_path = Path(out_dir).expanduser().resolve(strict=False)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    ledger_bytes = b"".join(canonical_json_bytes(row) + b"\n" for row in final_ledger_rows)
    ledger_path = out_dir_path / "owner-context-ledger.jsonl"
    _write_create_or_identical(ledger_path, ledger_bytes)
    ledger_sha256 = sha256_file(ledger_path)

    # A stable placeholder path/digest for the foil-geometry review file we
    # are about to author; the real digest is bound once the file is written.
    review_path = out_dir_path / "successor-foil-geometry-review.json"
    new_foil_members, foil_boxes_by_context = build_new_foil_members(
        ledger_rows=final_ledger_rows,
        gt_box_by_context=gt_box_by_context,
        review_artifact_path=review_path.name,
        review_artifact_sha256="0" * 64,
    )
    # Build the real review rows now that boxes are known, then bind the
    # foil-member provenance to this file's real digest below.
    reviewed_rows = []
    for row in final_ledger_rows:
        image_id = row["image_id"]
        for bank_name in ("background", "scan"):
            box = foil_boxes_by_context[row["context_id"]][bank_name]
            reviewed_rows.append(
                {
                    "source_row_id": f"foil-review:{row['context_id']}:{bank_name}",
                    "gt_owner_id": row["gt_owner_id"],
                    "image_id": image_id,
                    "bank_name": bank_name,
                    "box": list(box),
                    "extent_submode": f"equal_size_{bank_name}" if bank_name == "background" else "sorted_scan",
                }
            )
    review_document = {
        "schema_version": FOIL_REVIEW_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "status": "lead_reviewed_and_frozen_before_landscape_scoring",
        "anti_leakage_contract": {
            "landscape_scores_used_for_selection": False,
            "forced_continuation_scores_used_for_selection": False,
        },
        "reviewed_geometries": reviewed_rows,
    }
    review_bytes = canonical_json_bytes(review_document) + b"\n"
    _write_create_or_identical(review_path, review_bytes)
    review_sha256 = sha256_file(review_path)

    new_foil_members, foil_boxes_by_context = build_new_foil_members(
        ledger_rows=final_ledger_rows,
        gt_box_by_context=gt_box_by_context,
        review_artifact_path=review_path.name,
        review_artifact_sha256=review_sha256,
    )

    rules_document = specialize_rules_document(
        rules_template=rules_template_document,
        owner_canonical_descriptions=resolved_descriptions,
        new_foil_members=new_foil_members,
        upstream_digests=upstream_digests,
    )
    rules_bytes = canonical_json_bytes(rules_document) + b"\n"
    rules_path = out_dir_path / "landscape-decision-rules.json"
    _write_create_or_identical(rules_path, rules_bytes)
    rules_sha256 = sha256_file(rules_path)

    bank_seeds_document = build_bank_seeds_document(
        ledger_rows=final_ledger_rows,
        gt_box_by_context=gt_box_by_context,
        rules_document=rules_document,
        rules_sha256=rules_sha256,
        ledger_sha256=ledger_sha256,
        new_foil_members=new_foil_members,
        foil_boxes_by_context=foil_boxes_by_context,
    )
    bank_seeds_bytes = canonical_json_bytes(bank_seeds_document) + b"\n"
    bank_seeds_path = out_dir_path / "candidate-bank-seeds.json"
    _write_create_or_identical(bank_seeds_path, bank_seeds_bytes)

    # mechanism-decision-rules.json is the *parent*: frozen here, before any
    # fixed-budget candidate row is generated, and bound only to digests that
    # already exist at this point (the execution landscape-decision-rules.json
    # just written, the registry, and the rules template). It never embeds a
    # candidate-file digest -- see build_mechanism_decision_rules_document's
    # docstring for the non-recursive parent-binding design.
    mechanism_decision_rules_document = build_mechanism_decision_rules_document(
        execution_landscape_decision_rules_sha256=rules_sha256,
        fn_mechanism_registry_sha256=registry_document["registry_digest"],
        rules_template_sha256=sha256_file(rules_template_path),
    )
    mechanism_decision_rules_bytes = canonical_json_bytes(mechanism_decision_rules_document) + b"\n"
    mechanism_decision_rules_path = out_dir_path / "mechanism-decision-rules.json"
    _write_create_or_identical(mechanism_decision_rules_path, mechanism_decision_rules_bytes)
    mechanism_decision_rules_sha256 = sha256_file(mechanism_decision_rules_path)

    fixed_budget_rows: list[dict[str, Any]] = []
    for row in final_ledger_rows:
        meta = role_meta_by_context[row["context_id"]]
        is_control, control_kind, rungs = resolve_rungs(meta["cohort"])
        for rung in rungs:
            fixed_budget_rows.extend(
                build_fixed_budget_candidates_for_context(
                    owner_context_id=row["context_id"],
                    gt_owner_id=row["gt_owner_id"],
                    gt_box=gt_box_by_context[row["context_id"]],
                    token_registry=token_registry,
                    is_control=is_control,
                    control_kind=control_kind,
                    matched_control_group=meta["matched_control_group"],
                    rung=rung,
                    mechanism_decision_rules_sha256=mechanism_decision_rules_sha256,
                    other_owner_reference=meta["other_owner_reference"],
                )
            )
    fixed_budget_document_rows = fixed_budget_rows
    neighborhood_consistency = verify_neighborhood_consistency(fixed_budget_document_rows)
    fixed_budget_bytes = b"".join(
        canonical_json_bytes(row) + b"\n" for row in fixed_budget_document_rows
    )
    fixed_budget_path = out_dir_path / "fixed-budget-candidates.jsonl"
    _write_create_or_identical(fixed_budget_path, fixed_budget_bytes)

    receipt_content = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "sources": {
            "registry": {"path": str(registry_path), "sha256": sha256_file(registry_path)},
            "owner_ledger": {"path": str(owner_ledger_path), "sha256": sha256_file(owner_ledger_path)},
            "panel": {"path": str(panel_path), "sha256": sha256_file(panel_path)},
            "rules_template": {"path": str(rules_template_path), "sha256": sha256_file(rules_template_path)},
            "rollouts": [{"path": str(p), "sha256": sha256_file(p)} for p in rollout_paths],
            "canonical_description_registry": (
                {
                    "path": str(canonical_description_registry_path),
                    "sha256": sha256_file(canonical_description_registry_path),
                }
                if canonical_description_registry_path is not None
                else None
            ),
        },
        "outputs": {
            "owner_context_ledger": {"path": str(ledger_path), "sha256": sha256_file(ledger_path), "row_count": len(final_ledger_rows)},
            "landscape_decision_rules": {"path": str(rules_path), "sha256": rules_sha256},
            "candidate_bank_seeds": {"path": str(bank_seeds_path), "sha256": sha256_file(bank_seeds_path)},
            "fixed_budget_candidates": {
                "path": str(fixed_budget_path),
                "sha256": sha256_file(fixed_budget_path),
                "row_count": len(fixed_budget_document_rows),
            },
            "foil_geometry_review": {"path": str(review_path), "sha256": review_sha256},
            "mechanism_decision_rules": {
                "path": str(mechanism_decision_rules_path),
                "sha256": mechanism_decision_rules_sha256,
            },
        },
        "role_count": len(roles),
        "context_count": len(final_ledger_rows),
        "predecessor_reuse": predecessor_reuse_summary,
        "neighborhood_consistency": neighborhood_consistency,
    }
    receipt_document = {**receipt_content, "receipt_digest": sha256_json(receipt_content)}
    receipt_bytes = canonical_json_bytes(receipt_document) + b"\n"
    _write_create_or_identical(out_dir_path / "receipt.json", receipt_bytes)
    return receipt_document


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--owner-ledger", required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--rules-template", required=True)
    parser.add_argument("--rollout", action="append", required=True, dest="rollouts")
    parser.add_argument("--canonical-description-registry", default=None)
    parser.add_argument(
        "--predecessor-score-rows",
        default=None,
        help=(
            "optional real predecessor score-row artifact (e.g. a merged "
            "landscape_scores.v1 file); score/logit/policy payloads in it are "
            "legitimate and safely ignored (never read into any output). "
            "This is diagnostic-only: no fixed-budget candidate ever claims "
            "a reusable predecessor score from gt_owner_id+box-token "
            "geometry alone (real predecessor rows carry no literal "
            "source-identity field to verify that safely), so "
            "valid_reuse_count in the receipt is always 0; the successor "
            "scorer freshly scores every candidate"
        ),
    )
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    receipt = prepare_sorted_fn_successor_inputs(
        registry=args.registry,
        owner_ledger=args.owner_ledger,
        panel=args.panel,
        rules_template=args.rules_template,
        rollouts=args.rollouts,
        canonical_description_registry=args.canonical_description_registry,
        predecessor_score_rows=args.predecessor_score_rows,
        out_dir=args.out_dir,
    )
    print(
        json.dumps(
            {
                "out_dir": str(Path(args.out_dir).expanduser().resolve(strict=False)),
                "receipt_digest": receipt["receipt_digest"],
                "role_count": receipt["role_count"],
                "context_count": receipt["context_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
