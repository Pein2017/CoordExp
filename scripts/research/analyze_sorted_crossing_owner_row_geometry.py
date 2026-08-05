#!/usr/bin/env python3
"""Owner-row geometric relation stratification for the sorted crossing-boundary
catch-up arms.

Frozen unit::

    research/investigations/qwen3-vl-dense-enumeration/experiments/
    2026-08-04-sorted-crossing-owner-row-geometric-relation-stratification/
    {unit.md,tasks.md}

What this module does
---------------------
It reads **only** sealed products of the completed crossing-boundary unit
(``2026-08-03-sorted-crossing-boundary-owner-release-realization``) and, through
that plan's sealed lineage, the canonical owner ledger of the accessibility
census.  It emits one owner-level geometric census that answers exactly two
descriptive questions:

* are the primary ``P+C -> E`` coordinate changes concentrated in geometrically
  overlapping ``C``/``E`` pairs; and
* are the twelve frozen greedy displacers geometric neighbours of their targets.

Arms and geometry are bound one-to-one
--------------------------------------
``unit.md`` "Exact geometric fields": ``R=E`` for the primary ``P+C -> E`` arm
and ``R=F`` for the optional ``P+E+C -> F`` sensitivity arm; E geometry is never
reused for an F likelihood delta.  This is enforced, not asserted: for every
emitted row the sealed *scored* coordinate token span of that arm's secondary
merge row must equal the coordinate tokens of the sealed row whose geometry the
analyzer measured.  A crossed wiring fails closed before any output exists.

Every box is read from sealed norm-1000 coordinate **token ids**.  There is no
pixel-space round trip anywhere in the geometry: widths, heights, areas,
intersections, IoU, centre distance and the sorted-key ranks are all computed in
bin space, and the only place a pixel extent appears is the *display-only*
projection hint in the visualization plan.

The one aggregation rule
------------------------
``unit.md`` "Reports": raw likelihood deltas are never summarized by one
cross-image median, range or quantile.  Raw deltas therefore stay indexed by
image; only counts, exact owner rows and rank associations aggregate over
images.  :func:`assert_no_cross_image_raw_delta_pooling` fails closed if any
emitted summary key pools raw deltas outside a per-image or within-image scope.

What this module never does
---------------------------
It opens no model, uses no GPU, rescores no token, reads no branch it did not
join verbatim, changes no cutoff, and cannot alter the 26-owner denominator.
The three-way decision is evaluated only over the primary ``C``/``E`` arm, so
one owner never votes twice; the ``C``/``F`` arm is published as a named
sensitivity that can qualify a reading but never change the outcome.

Outputs (one analysis directory, published create-or-identical)::

    owner-geometry-rows.jsonl       26 primary C/E rows + 26 optional C/F rows
    displacement-pair-rows.jsonl    12 target/displacer rows
    geometry-summary.json           strata, per-image values, associations, decision
    geometry-report.md              the same content, rendered
    geometry-visual-plan.json       machine-readable plan/manifest for a renderer
    geometry-analysis-receipt.json  input/output digests and source hashes, self-sealed
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any, NoReturn

# ---------------------------------------------------------------------------
# 0. Frozen identities, thresholds and denominators
# ---------------------------------------------------------------------------

UNIT_ID = "2026-08-04-sorted-crossing-owner-row-geometric-relation-stratification"
SOURCE_UNIT_ID = "2026-08-03-sorted-crossing-boundary-owner-release-realization"
CENSUS_UNIT_ID = "2026-08-03-sorted-owner-accessibility-phenotype-census"

OWNER_ROW_SCHEMA_VERSION = "sorted-crossing-owner-row-geometry-row.v1"
PAIR_ROW_SCHEMA_VERSION = "sorted-crossing-owner-row-geometry-displacement-pair-row.v1"
SUMMARY_SCHEMA_VERSION = "sorted-crossing-owner-row-geometry-summary.v1"
VISUAL_PLAN_SCHEMA_VERSION = "sorted-crossing-owner-row-geometry-visual-plan.v1"
RECEIPT_SCHEMA_VERSION = "sorted-crossing-owner-row-geometry-analysis-receipt.v1"

OWNER_ROWS_NAME = "owner-geometry-rows.jsonl"
PAIR_ROWS_NAME = "displacement-pair-rows.jsonl"
SUMMARY_NAME = "geometry-summary.json"
REPORT_MD_NAME = "geometry-report.md"
VISUAL_PLAN_NAME = "geometry-visual-plan.json"
RECEIPT_NAME = "geometry-analysis-receipt.json"

# Sealed coordinate-token identity; cross-checked against every image registry row.
COORD_TOKEN_START = 151670
COORD_BIN_COUNT = 1000
CANVAS_BINS = 1000
CANVAS_DIAGONAL = float(CANVAS_BINS) * math.sqrt(2.0)

# Frozen cutoffs -- predeclared in unit.md and never changed after reading output.
HIGH_OVERLAP_MIN_IOU = 0.5
MATERIAL_NEGATIVE_MAX_NATS = -1.0
IDENTITY_SLIPPAGE_MIN_ROWS = 3
IDENTITY_SLIPPAGE_MIN_FRACTION = 0.75
SEPARATED_COMPETITION_MIN_ROWS = 3

PRIMARY_ARM = "p_plus_c_then_e"
SENSITIVITY_ARM = "p_plus_e_plus_c_then_f"
BENIGN_VARIANT = "benign_substitution_then_following_native_action"
CROSSING_COHORT = "u_bound_crossing_primary"
BENIGN_COHORT = "native_tp_replay_control"

#: ``(arm, sealed row key, row label, arm role)`` -- the one-to-one arm/geometry binding.
ARMS: tuple[tuple[str, str, str, str], ...] = (
    (PRIMARY_ARM, "e_row", "E", "primary"),
    (SENSITIVITY_ARM, "f_row", "F", "sensitivity"),
)

SEGMENTS: tuple[str, ...] = ("coordinates", "complete_row")

LABEL_HIGH_OVERLAP = "high_overlap"
LABEL_PARTIAL = "partial_overlap_or_center_containment"
LABEL_SEPARATED = "clearly_separated"
GEOMETRIC_LABELS: tuple[str, ...] = (LABEL_HIGH_OVERLAP, LABEL_PARTIAL, LABEL_SEPARATED)

DESCRIPTION_AXIS: tuple[str, ...] = ("same_description", "different_description")
E_STRATUM_AXIS: tuple[str, ...] = ("matched_e", "unmatched_e")
CHANGE_AXIS: tuple[str, ...] = ("material_negative", "nonmaterial")

OUTCOME_IDENTITY_SLIPPAGE = "identity_slippage_duplicate_suppression_eligible"
OUTCOME_SEPARATED_COMPETITION = "separated_competition_survives"
OUTCOME_INCONCLUSIVE = "inconclusive"
#: Fixed evaluation order: separated competition is tested first, so the identity
#: route can only be reached when fewer than three M_E rows are clearly separated.
DECISION_EXHAUSTIVE_ORDER: tuple[str, ...] = (
    OUTCOME_SEPARATED_COMPETITION,
    OUTCOME_IDENTITY_SLIPPAGE,
    OUTCOME_INCONCLUSIVE,
)
#: A non-inconclusive route needs qualifying rows from at least this many images,
#: because rows of one image share that image's single benign reference delta.
MIN_QUALIFYING_IMAGE_COUNT = 2

CLAIM_BOUNDARY = (
    "geometric association and owner identity ambiguity over frozen crossing rows; "
    "never a visual-tower absence, a causal insertion effect, eventual owner retention, "
    "free-rollout behavior, a final-set value, or a population estimate"
)

NOT_CLAIMED: tuple[str, ...] = (
    "no causal effect of the C insertion is claimed; every delta is a paired teacher-forced readout",
    "no eventual owner retention, natural stop or free-rollout behavior is claimed",
    "no final-set value and no population estimate is claimed",
    "no visual-tower absence is claimed",
    "image 2299 is not added to any completed denominator here",
)

#: Reserved for the prospective thirteen-image panel; carried forward, never a denominator.
RESERVED_NEXT_PANEL: Mapping[str, Any] = {
    "image_id": "2299",
    "role": "reserved_for_the_next_prospective_thirteen_image_panel",
    "enters_this_denominator": False,
    "gt_authority_path": (
        "/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/val.coord.jsonl"
    ),
    "annotation_file_sha256": (
        "81d674070d4b588488a2cb911c09f765b63c0e6d035b50db27ee0a41ff2a1894"
    ),
    "source_line_sha256": (
        "ce19853c74a595f22cc183ce450e561f2da3216e54a1e499cfbca1be7e1c425b"
    ),
    "image_sha256": "cd7199a37188c9ac6481520175866cbd78fa6fba35f4290bb0b60c78afcb2df3",
    "physical_owner_annotation_count": 46,
    "person_annotation_count": 38,
    "tie_annotation_count": 8,
    "person_positive_coco_annotation_id_count": 12,
    "person_manual_negative_id_count": 26,
    "verification_scope": (
        "declared by unit.md and carried forward verbatim; this analyzer opens no file "
        "outside the sealed crossing inputs and therefore does not re-verify it"
    ),
}

#: Declared only; this analyzer never opens media. A renderer resolves and verifies it.
DEFAULT_MEDIA_ROOT = "/data/CoordExp/public_data/coco/rescale_32_1024_bbox"

#: Stable per-role colours for every rendered panel.
ROLE_PALETTE: Mapping[str, str] = {
    "inserted_owner_c": "#1b6ca8",
    "downstream_row_e": "#d1495b",
    "downstream_row_f": "#e08e0b",
    "displacement_target": "#1b6ca8",
    "displacer": "#5f0f40",
}

#: Fraction of the union extent added on each side of a crop window.
CROP_MARGIN_FRACTION = 0.15
#: Minimum crop half-extent in bins, so a tiny pair still renders readably.
CROP_MIN_HALF_EXTENT_BINS = 40

_POOLED_STATISTIC_TOKENS: tuple[str, ...] = (
    "median",
    "mean",
    "quantile",
    "percentile",
    "range",
    "stdev",
    "variance",
    "iqr",
    "min",
    "max",
    "total",
    "sum",
)
_PER_IMAGE_SCOPE_MARKERS: tuple[str, ...] = ("per_image", "within_image", "by_image")

NO_CROSS_IMAGE_POOLING = (
    "raw likelihood deltas are never summarized by one cross-image median, range or "
    "quantile; only counts, exact owner rows and rank associations aggregate over images"
)

OPERATIONAL_DEFINITIONS: Mapping[str, str] = {
    "norm1000_box": (
        "x1,y1,x2,y2 recovered as coord_token_id - coordinate_token_start from the sealed "
        "row tokens; a valid box requires x2 > x1 and y2 > y1"
    ),
    "continuous_xyxy": (
        "widths, heights, areas, intersections and IoU use continuous xyxy geometry with no "
        "+1 endpoint convention"
    ),
    "center_distance_normalized": (
        "Euclidean distance between box centres in bin space divided by 1000*sqrt(2)"
    ),
    "signed_offsets": (
        "dx, dy are R-minus-C centre offsets and dw, dh are R-minus-C extent differences, "
        "each divided by 1000"
    ),
    "center_containment": "inclusive of the boundary: x1 <= cx <= x2 and y1 <= cy <= y2",
    "same_category_population": (
        "every canonical-ledger owner of that image carrying the same sealed normalized "
        "description as C; the official COCO category id is provenance beside it"
    ),
    "sorted_key_rank_gap": (
        "R's (y1,x1) key is inserted into the image's same-category physical-owner order, "
        "ordered by sealed norm-1000 (y1, x1); physical owners tie-break on their sealed "
        "gt_owner_id and R, carrying the stable synthetic tie key row:{variant}:{gt_owner_id}, "
        "sorts after every physical owner with the exact same (y1,x1); the gap is R's index "
        "minus C's index in that combined order"
    ),
    "pixel_sort_key_scope": (
        "the pixel-space owner_sort_key is provenance only; it is recorded beside each "
        "population and never mixed into the norm-1000 rank"
    ),
    "relative_coordinate_delta": (
        "crossing_coordinate_delta - same_image_benign_coordinate_delta, a descriptive "
        "image-referenced change and not a causal correction; the two deltas are summed over "
        "different scored rows -- the arm's own downstream row E or F, and that image's benign "
        "control's own downstream row -- each over its own four coordinate tokens"
    ),
    "material_negative": (
        f"relative_coordinate_delta <= {MATERIAL_NEGATIVE_MAX_NATS} nat over the same four "
        "coordinate tokens"
    ),
    "geometric_label": (
        "one mutually exclusive label per row: high_overlap (IoU >= 0.5), "
        "partial_overlap_or_center_containment (IoU > 0 or either centre inside the other, "
        "below the high-overlap cutoff), or clearly_separated"
    ),
    "any_overlap_or_center_containment": (
        "the inclusive predicate IoU > 0 or either centre inside the other; high_overlap "
        "rows satisfy it"
    ),
    "spearman_rho": (
        "tie-corrected Spearman rank association between a continuous geometry measure and "
        "relative_coordinate_delta; descriptive only, with leave-one-image-out coefficients"
    ),
}


class GeometryContractError(RuntimeError):
    """Raised when a sealed input or an emitted payload violates the frozen contract."""


def _fail(message: str) -> NoReturn:
    raise GeometryContractError(message)


# ---------------------------------------------------------------------------
# 1. Bytes, digests and fail-closed readers
# ---------------------------------------------------------------------------


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def sha256_bytes(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        _fail(f"{label} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _fail(f"{label} is not valid JSON ({path}): {exc}")
    if not isinstance(payload, dict):
        _fail(f"{label} must be a JSON object ({path})")
    return payload


def read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        _fail(f"{label} is missing: {path}")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            _fail(f"{label} line {index} is not valid JSON ({path}): {exc}")
        if not isinstance(row, dict):
            _fail(f"{label} line {index} must be a JSON object ({path})")
        rows.append(row)
    return rows


def assert_self_sealed(payload: Mapping[str, Any], *, digest_key: str, label: str) -> str:
    """A sealed receipt must reproduce its own content digest."""

    sealed = payload.get(digest_key)
    if not isinstance(sealed, str) or not sealed:
        _fail(f"{label} carries no {digest_key}")
    body = {key: value for key, value in payload.items() if key != digest_key}
    recomputed = sha256_json(body)
    if recomputed != sealed:
        _fail(
            f"{label} is not self-sealed: {digest_key}={sealed} but its content digests to "
            f"{recomputed}"
        )
    return sealed


def assert_file_digest(path: Path, *, expected: str, label: str) -> str:
    observed = sha256_file(path)
    if observed != expected:
        _fail(
            f"{label} digest mismatch for {path}: sealed {expected}, observed {observed}; "
            "the sealed inputs have drifted and nothing is emitted"
        )
    return observed


def _declared_digest(digests: Mapping[str, Any], name: str, *, label: str) -> str:
    entry = digests.get(name)
    if isinstance(entry, Mapping):
        value = entry.get("sha256")
    else:
        value = entry
    if not isinstance(value, str) or not value:
        _fail(f"{label} declares no sha256 for {name}")
    return value


def assert_finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} must be a finite number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        _fail(f"{label} must be finite, got {number!r}")
    return number


def assert_emitted_payload(payload: Any, *, label: str, path: str = "") -> None:
    """Every emitted number must be finite and every mapping key must be a string."""

    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if not isinstance(key, str):
                _fail(f"{label} has a non-string key at {path or '<root>'}: {key!r}")
            assert_emitted_payload(value, label=label, path=f"{path}.{key}" if path else key)
        return
    if isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            assert_emitted_payload(value, label=label, path=f"{path}[{index}]")
        return
    if isinstance(payload, float) and not math.isfinite(payload):
        _fail(f"{label} carries a nonfinite value at {path or '<root>'}")


def assert_no_cross_image_raw_delta_pooling(payload: Any, *, label: str, path: str = "") -> None:
    """Fail closed if a raw delta is pooled outside a per-image or within-image scope."""

    if isinstance(payload, Mapping):
        for key, value in payload.items():
            child = f"{path}.{key}" if path else key
            lowered = key.lower()
            if "delta" in lowered and any(
                token in lowered for token in _POOLED_STATISTIC_TOKENS
            ):
                scope = child.lower()
                if not any(marker in scope for marker in _PER_IMAGE_SCOPE_MARKERS):
                    _fail(
                        f"{label} pools raw deltas across images at {child}: "
                        f"{NO_CROSS_IMAGE_POOLING}"
                    )
            assert_no_cross_image_raw_delta_pooling(value, label=label, path=child)
        return
    if isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            assert_no_cross_image_raw_delta_pooling(
                value, label=label, path=f"{path}[{index}]"
            )


# ---------------------------------------------------------------------------
# 2. Norm-1000 geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Box:
    """A sealed norm-1000 box in continuous xyxy bin space."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> float:
        return float(self.x2 - self.x1)

    @property
    def height(self) -> float:
        return float(self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def contains_center_of(self, other: "Box") -> bool:
        cx, cy = other.center
        return self.x1 <= cx <= self.x2 and self.y1 <= cy <= self.y2

    def as_list(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]


def decode_box(
    coord_token_ids: Any,
    *,
    label: str,
    coord_token_start: int = COORD_TOKEN_START,
    bin_count: int = COORD_BIN_COUNT,
) -> Box:
    """Recover a norm-1000 box from sealed coordinate token ids, never from pixels."""

    if not isinstance(coord_token_ids, Sequence) or isinstance(coord_token_ids, (str, bytes)):
        _fail(f"{label} coordinate tokens must be a sequence, got {coord_token_ids!r}")
    tokens = list(coord_token_ids)
    if len(tokens) != 4:
        _fail(f"{label} must carry exactly four coordinate tokens, got {len(tokens)}")
    bins: list[int] = []
    stop = coord_token_start + bin_count - 1
    for position, token in enumerate(tokens):
        if isinstance(token, bool) or not isinstance(token, int):
            _fail(f"{label} coordinate token {position} must be an int, got {token!r}")
        if token < coord_token_start or token > stop:
            _fail(
                f"{label} coordinate token {position} is outside the sealed coordinate range "
                f"[{coord_token_start}, {stop}]: {token}"
            )
        bins.append(token - coord_token_start)
    box = Box(bins[0], bins[1], bins[2], bins[3])
    if box.x2 <= box.x1 or box.y2 <= box.y1:
        _fail(f"{label} is not a valid box: strict positive extents required, got {box.as_list()}")
    return box


def box_geometry(c_box: Box, r_box: Box, *, label: str) -> dict[str, Any]:
    """Every exact geometric field unit.md names, in continuous norm-1000 xyxy space."""

    inter_w = max(0.0, min(c_box.x2, r_box.x2) - max(c_box.x1, r_box.x1))
    inter_h = max(0.0, min(c_box.y2, r_box.y2) - max(c_box.y1, r_box.y1))
    intersection = inter_w * inter_h
    union = c_box.area + r_box.area - intersection
    if union <= 0.0:
        _fail(f"{label} has a nonpositive union area; boxes {c_box.as_list()} {r_box.as_list()}")
    iou = intersection / union

    c_cx, c_cy = c_box.center
    r_cx, r_cy = r_box.center
    center_distance_bins = math.hypot(r_cx - c_cx, r_cy - c_cy)

    c_center_inside_r = r_box.contains_center_of(c_box)
    r_center_inside_c = c_box.contains_center_of(r_box)
    high_overlap = iou >= HIGH_OVERLAP_MIN_IOU
    any_overlap = iou > 0.0 or c_center_inside_r or r_center_inside_c
    clearly_separated = iou == 0.0 and not c_center_inside_r and not r_center_inside_c
    if high_overlap:
        geometric_label = LABEL_HIGH_OVERLAP
    elif any_overlap:
        geometric_label = LABEL_PARTIAL
    else:
        geometric_label = LABEL_SEPARATED
    if any_overlap == clearly_separated:
        _fail(f"{label} produced contradictory geometric labels for {c_box} / {r_box}")

    area_ratio = r_box.area / c_box.area
    geometry: dict[str, Any] = {
        "c_box_norm1000_xyxy": c_box.as_list(),
        "r_box_norm1000_xyxy": r_box.as_list(),
        "c_area_bins2": c_box.area,
        "r_area_bins2": r_box.area,
        "intersection_area_bins2": intersection,
        "union_area_bins2": union,
        "iou": iou,
        "intersection_over_c_area": intersection / c_box.area,
        "intersection_over_r_area": intersection / r_box.area,
        "center_distance_bins": center_distance_bins,
        "center_distance_normalized": center_distance_bins / CANVAS_DIAGONAL,
        "dx": (r_cx - c_cx) / float(CANVAS_BINS),
        "dy": (r_cy - c_cy) / float(CANVAS_BINS),
        "dw": (r_box.width - c_box.width) / float(CANVAS_BINS),
        "dh": (r_box.height - c_box.height) / float(CANVAS_BINS),
        "area_ratio_r_over_c": area_ratio,
        "abs_log_area_ratio": abs(math.log(area_ratio)),
        "c_center_inside_r": c_center_inside_r,
        "r_center_inside_c": r_center_inside_c,
        "high_overlap": high_overlap,
        "any_overlap_or_center_containment": any_overlap,
        "clearly_separated": clearly_separated,
        "geometric_label": geometric_label,
    }
    for key, value in geometry.items():
        if isinstance(value, float):
            assert_finite(value, label=f"{label}.{key}")
    return geometry


# ---------------------------------------------------------------------------
# 3. Sealed input binding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DenominatorContract:
    """Frozen denominators. Never exposed on the CLI; tests may narrow it."""

    crossing_owner_count: int = 26
    greedy_pair_count: int = 12
    benign_control_count: int = 12
    secondary_owner_row_count: int = 38
    secondary_merged_row_count: int = 64

    @property
    def crossing_secondary_readout_count(self) -> int:
        return self.crossing_owner_count * len(ARMS)

    @property
    def expected_owner_geometry_row_count(self) -> int:
        return self.crossing_owner_count * len(ARMS)


FROZEN_DENOMINATORS = DenominatorContract()

COHORT_REGISTRY_NAME = "cohort-registry.jsonl"
PLAN_MANIFEST_NAME = "manifest.json"
PRIMARY_RECEIPT_NAME = "receipt.json"
PRIMARY_OWNER_ROWS_NAME = "owner-rows.jsonl"
SECONDARY_RECEIPT_NAME = "secondary-analysis-receipt.json"
SECONDARY_OWNER_ROWS_NAME = "secondary-owner-rows.jsonl"
SECONDARY_MERGE_RECEIPT_NAME = "secondary-merge-receipt.json"
SECONDARY_MERGED_ROWS_NAME = "secondary-compatibility-rows.jsonl"
CENSUS_OWNER_REGISTRY_NAME = "owner-registry.jsonl"
CENSUS_IMAGE_REGISTRY_NAME = "image-registry.jsonl"
CENSUS_RECEIPT_NAME = "receipt.json"


@dataclass(frozen=True)
class SealedInputs:
    """Every sealed row set, bound to its declared digest before anything is read."""

    plan_dir: Path
    plan_manifest: Mapping[str, Any]
    plan_manifest_content_sha256: str
    cohort_rows: tuple[Mapping[str, Any], ...]
    primary_rows: tuple[Mapping[str, Any], ...]
    primary_receipt_content_sha256: str
    secondary_rows: tuple[Mapping[str, Any], ...]
    secondary_receipt_content_sha256: str
    merged_rows: tuple[Mapping[str, Any], ...]
    merge_receipt_content_sha256: str
    census_run_root: Path
    census_receipt_content_sha256: str
    owner_registry: Mapping[str, Mapping[str, Any]]
    image_registry: Mapping[str, Mapping[str, Any]]
    input_file_sha256: Mapping[str, str]
    source_sha256: Mapping[str, str]


def _index_unique(
    rows: Iterable[Mapping[str, Any]], key: str, *, label: str
) -> dict[str, Mapping[str, Any]]:
    index: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            _fail(f"{label} row carries no {key}")
        if value in index:
            _fail(f"{label} carries a duplicate {key}: {value}")
        index[value] = row
    return index


def load_sealed_inputs(
    *,
    plan_dir: Path,
    primary_owner_rows: Path,
    secondary_owner_rows: Path,
    secondary_merged_dir: Path,
    census_plan_dir: Path | None = None,
    contract: DenominatorContract = FROZEN_DENOMINATORS,
) -> SealedInputs:
    """Bind every input digest and source hash before a single output field exists."""

    plan_dir = Path(plan_dir)
    manifest_path = plan_dir / PLAN_MANIFEST_NAME
    manifest = read_json(manifest_path, "crossing plan manifest")
    manifest_seal = assert_self_sealed(
        manifest, digest_key="manifest_content_sha256", label="crossing plan manifest"
    )
    if manifest.get("unit_id") != SOURCE_UNIT_ID:
        _fail(
            f"crossing plan manifest is for unit {manifest.get('unit_id')!r}, expected "
            f"{SOURCE_UNIT_ID!r}"
        )

    input_file_sha256: dict[str, str] = {}
    source_sha256: dict[str, str] = {}

    cohort_path = plan_dir / COHORT_REGISTRY_NAME
    manifest_digests = manifest.get("output_file_digests")
    if not isinstance(manifest_digests, Mapping):
        _fail("crossing plan manifest carries no output_file_digests")
    input_file_sha256[str(cohort_path)] = assert_file_digest(
        cohort_path,
        expected=_declared_digest(
            manifest_digests, COHORT_REGISTRY_NAME, label="crossing plan manifest"
        ),
        label="crossing cohort registry",
    )
    input_file_sha256[str(manifest_path)] = sha256_file(manifest_path)
    builder = manifest.get("builder_source")
    if isinstance(builder, Mapping) and isinstance(builder.get("sha256"), str):
        source_sha256["crossing_plan_builder"] = str(builder["sha256"])

    cohort_rows = read_jsonl(cohort_path, "crossing cohort registry")
    if len(cohort_rows) != contract.crossing_owner_count:
        _fail(
            f"crossing cohort registry holds {len(cohort_rows)} rows, expected "
            f"{contract.crossing_owner_count}"
        )

    # Primary owner rows, bound through their own analysis receipt.
    primary_owner_rows = Path(primary_owner_rows)
    primary_dir = primary_owner_rows.parent
    primary_receipt = read_json(primary_dir / PRIMARY_RECEIPT_NAME, "primary analysis receipt")
    primary_seal = assert_self_sealed(
        primary_receipt, digest_key="receipt_content_sha256", label="primary analysis receipt"
    )
    primary_digests = primary_receipt.get("output_file_digests")
    if not isinstance(primary_digests, Mapping):
        _fail("primary analysis receipt carries no output_file_digests")
    input_file_sha256[str(primary_owner_rows)] = assert_file_digest(
        primary_owner_rows,
        expected=_declared_digest(
            primary_digests, primary_owner_rows.name, label="primary analysis receipt"
        ),
        label="primary owner rows",
    )
    for key in ("analyzer_source_sha256", "merger_source_sha256", "scorer_source_sha256"):
        if isinstance(primary_receipt.get(key), str):
            source_sha256[f"primary_{key}"] = str(primary_receipt[key])
    primary_rows = read_jsonl(primary_owner_rows, "primary owner rows")

    # Secondary owner rows, bound through the secondary analysis receipt.
    secondary_owner_rows = Path(secondary_owner_rows)
    secondary_dir = secondary_owner_rows.parent
    secondary_receipt = read_json(
        secondary_dir / SECONDARY_RECEIPT_NAME, "secondary analysis receipt"
    )
    secondary_seal = assert_self_sealed(
        secondary_receipt,
        digest_key="receipt_content_sha256",
        label="secondary analysis receipt",
    )
    secondary_digests = secondary_receipt.get("output_file_digests")
    if not isinstance(secondary_digests, Mapping):
        _fail("secondary analysis receipt carries no output_file_digests")
    input_file_sha256[str(secondary_owner_rows)] = assert_file_digest(
        secondary_owner_rows,
        expected=_declared_digest(
            secondary_digests, secondary_owner_rows.name, label="secondary analysis receipt"
        ),
        label="secondary owner rows",
    )
    for key in ("analyzer_source_sha256", "merger_source_sha256", "secondary_scorer_source_sha256"):
        if isinstance(secondary_receipt.get(key), str):
            source_sha256[f"secondary_{key}"] = str(secondary_receipt[key])
    secondary_rows = read_jsonl(secondary_owner_rows, "secondary owner rows")
    if len(secondary_rows) != contract.secondary_owner_row_count:
        _fail(
            f"secondary owner rows hold {len(secondary_rows)} rows, expected "
            f"{contract.secondary_owner_row_count}"
        )

    # Secondary merge rows, for the exact scored token spans.
    secondary_merged_dir = Path(secondary_merged_dir)
    merge_receipt = read_json(
        secondary_merged_dir / SECONDARY_MERGE_RECEIPT_NAME, "secondary merge receipt"
    )
    merge_seal = assert_self_sealed(
        merge_receipt, digest_key="receipt_content_sha256", label="secondary merge receipt"
    )
    merge_digests = merge_receipt.get("output_file_digests")
    if not isinstance(merge_digests, Mapping):
        _fail("secondary merge receipt carries no output_file_digests")
    merged_rows_path = secondary_merged_dir / SECONDARY_MERGED_ROWS_NAME
    if not merged_rows_path.is_file():
        _fail(f"secondary merge rows are missing: {merged_rows_path}")
    input_file_sha256[str(merged_rows_path)] = assert_file_digest(
        merged_rows_path,
        expected=_declared_digest(
            merge_digests, SECONDARY_MERGED_ROWS_NAME, label="secondary merge receipt"
        ),
        label="secondary merge rows",
    )
    if isinstance(merge_receipt.get("merger_source_sha256"), str):
        source_sha256["secondary_merge_source_sha256"] = str(
            merge_receipt["merger_source_sha256"]
        )
    merged_rows = read_jsonl(merged_rows_path, "secondary merge rows")
    if len(merged_rows) != contract.secondary_merged_row_count:
        _fail(
            f"secondary merge rows hold {len(merged_rows)} rows, expected "
            f"{contract.secondary_merged_row_count}"
        )

    # The plan's sealed lineage to the canonical owner ledger.
    lineage = manifest.get("lineage")
    if not isinstance(lineage, Mapping):
        _fail("crossing plan manifest carries no lineage")
    if lineage.get("census_unit_id") != CENSUS_UNIT_ID:
        _fail(
            f"crossing plan lineage names census unit {lineage.get('census_unit_id')!r}, "
            f"expected {CENSUS_UNIT_ID!r}"
        )
    census_run_root = Path(str(lineage.get("census_run_root", "")))
    census_dir = Path(census_plan_dir) if census_plan_dir is not None else census_run_root / "plan"
    census_receipt = read_json(census_dir / CENSUS_RECEIPT_NAME, "census plan receipt")
    census_seal = assert_self_sealed(
        census_receipt, digest_key="receipt_content_sha256", label="census plan receipt"
    )
    declared_census_seal = lineage.get("census_plan_receipt_content_sha256")
    if census_seal != declared_census_seal:
        _fail(
            "census plan receipt does not match the crossing plan lineage: sealed "
            f"{declared_census_seal}, observed {census_seal}"
        )
    census_inputs = lineage.get("census_input_files")
    if not isinstance(census_inputs, Mapping):
        _fail("crossing plan lineage carries no census_input_files")

    census_files: dict[str, list[Mapping[str, Any]]] = {}
    for name in (CENSUS_OWNER_REGISTRY_NAME, CENSUS_IMAGE_REGISTRY_NAME):
        declared = census_inputs.get(f"plan/{name}")
        if not isinstance(declared, Mapping) or not isinstance(declared.get("sha256"), str):
            _fail(f"crossing plan lineage declares no digest for plan/{name}")
        path = census_dir / name
        input_file_sha256[str(path)] = assert_file_digest(
            path, expected=str(declared["sha256"]), label=f"census {name}"
        )
        census_files[name] = read_jsonl(path, f"census {name}")

    owner_registry = _index_unique(
        census_files[CENSUS_OWNER_REGISTRY_NAME], "gt_owner_id", label="census owner registry"
    )
    image_registry = _index_unique(
        census_files[CENSUS_IMAGE_REGISTRY_NAME], "image_id", label="census image registry"
    )
    for image_id, row in sorted(image_registry.items()):
        identity = row.get("coordinate_token_ids")
        if not isinstance(identity, Mapping):
            _fail(f"census image registry row {image_id} carries no coordinate_token_ids block")
        if (
            identity.get("start") != COORD_TOKEN_START
            or identity.get("bin_count") != COORD_BIN_COUNT
            or identity.get("end_inclusive") != COORD_TOKEN_START + COORD_BIN_COUNT - 1
        ):
            _fail(
                f"census image registry row {image_id} declares coordinate token identity "
                f"{dict(identity)!r}, which is not the frozen "
                f"[{COORD_TOKEN_START}, {COORD_TOKEN_START + COORD_BIN_COUNT - 1}] range"
            )

    return SealedInputs(
        plan_dir=plan_dir,
        plan_manifest=manifest,
        plan_manifest_content_sha256=manifest_seal,
        cohort_rows=tuple(cohort_rows),
        primary_rows=tuple(primary_rows),
        primary_receipt_content_sha256=primary_seal,
        secondary_rows=tuple(secondary_rows),
        secondary_receipt_content_sha256=secondary_seal,
        merged_rows=tuple(merged_rows),
        merge_receipt_content_sha256=merge_seal,
        census_run_root=census_run_root,
        census_receipt_content_sha256=census_seal,
        owner_registry=owner_registry,
        image_registry=image_registry,
        input_file_sha256=dict(sorted(input_file_sha256.items())),
        source_sha256=dict(sorted(source_sha256.items())),
    )


# ---------------------------------------------------------------------------
# 4. Frozen joins
# ---------------------------------------------------------------------------


def _owner_anchor_box(owner: Mapping[str, Any], *, label: str) -> Box:
    """The sealed exact-GT-anchor candidate is the owner's norm-1000 box."""

    bank = owner.get("candidate_bank")
    if not isinstance(bank, Mapping):
        _fail(f"{label} carries no candidate_bank")
    roles = bank.get("logical_roles")
    if not isinstance(roles, Sequence):
        _fail(f"{label} carries no logical_roles")
    anchors = [
        role
        for role in roles
        if isinstance(role, Mapping) and role.get("role") == "exact_gt_anchor"
    ]
    if len(anchors) != 1:
        _fail(f"{label} carries {len(anchors)} exact_gt_anchor roles, expected exactly one")
    return decode_box(anchors[0].get("coord_token_ids"), label=f"{label} exact_gt_anchor")


def build_owner_population(
    inputs: SealedInputs, *, image_id: str, normalized_description: str
) -> dict[str, Any]:
    """The image's same-description physical-owner order, in sealed norm-1000 (y1,x1).

    ``unit.md``: same-category means the same sealed normalized description within the
    image, and the rank is derived only from norm-1000 coordinate tokens.  The sealed
    pixel ``owner_sort_key`` is carried as provenance and never mixed into the rank.
    """

    members: list[dict[str, Any]] = []
    for owner_id, owner in inputs.owner_registry.items():
        if owner.get("image_id") != image_id:
            continue
        if owner.get("normalized_description") != normalized_description:
            continue
        box = _owner_anchor_box(owner, label=f"census owner {owner_id}")
        sort_key = owner.get("owner_sort_key")
        if not isinstance(sort_key, Sequence) or len(sort_key) != 2:
            _fail(f"census owner {owner_id} carries no two-element owner_sort_key")
        members.append(
            {
                "gt_owner_id": owner_id,
                "box": box,
                "sealed_pixel_sort_key": [int(sort_key[0]), int(sort_key[1])],
                "native_true_positive": bool(owner.get("native_true_positive")),
                "normalized_description": owner.get("normalized_description"),
                "official_coco_category_id": owner.get("official_coco_category_id"),
            }
        )
    if not members:
        _fail(
            f"census owner ledger holds no owner for image {image_id} description "
            f"{normalized_description!r}"
        )
    ordered = sorted(members, key=lambda item: (item["box"].y1, item["box"].x1, item["gt_owner_id"]))
    by_pixel = sorted(
        members,
        key=lambda item: (
            item["sealed_pixel_sort_key"][0],
            item["sealed_pixel_sort_key"][1],
            item["gt_owner_id"],
        ),
    )
    return {
        "image_id": image_id,
        "normalized_description": normalized_description,
        "members": ordered,
        "pixel_sort_key_order_agrees": (
            [item["gt_owner_id"] for item in ordered]
            == [item["gt_owner_id"] for item in by_pixel]
        ),
    }


def sorted_key_rank_gap(
    population: Mapping[str, Any], *, c_owner_id: str, r_box: Box, r_tie_key: str
) -> dict[str, Any]:
    """Insert R's (y1,x1) key into the owner order and compare it with C's rank.

    Physical owners sort by ``(y1, x1, gt_owner_id)``; the inserted row carries an
    explicit "after every equal-key owner" flag rather than relying on how its
    synthetic tie key happens to compare with owner ids.
    """

    members = population["members"]
    entries = [
        (item["box"].y1, item["box"].x1, 0, item["gt_owner_id"]) for item in members
    ]
    owner_ids = [entry[3] for entry in entries]
    if c_owner_id not in owner_ids:
        _fail(f"inserted owner {c_owner_id} is missing from its own same-category population")
    tie_with_owner = any(entry[0] == r_box.y1 and entry[1] == r_box.x1 for entry in entries)
    combined = sorted([*entries, (r_box.y1, r_box.x1, 1, r_tie_key)])
    c_rank = next(index for index, entry in enumerate(combined) if entry[3] == c_owner_id)
    r_rank = next(index for index, entry in enumerate(combined) if entry[3] == r_tie_key)
    return {
        "population_size": len(entries),
        "population_normalized_description": population["normalized_description"],
        "c_rank_in_population": owner_ids.index(c_owner_id),
        "c_rank_after_insertion": c_rank,
        "r_rank_after_insertion": r_rank,
        "sorted_key_rank_gap": r_rank - c_rank,
        "r_tie_key": r_tie_key,
        "rank_key_tie_with_physical_owner": tie_with_owner,
        "inserted_row_sorts_after_equal_key_owners": True,
        "pixel_sort_key_order_agrees": population["pixel_sort_key_order_agrees"],
        "pixel_sort_key_role": "provenance_only_never_mixed_into_this_rank",
    }


def _segment_delta(
    readout: Mapping[str, Any], segment: str, *, label: str
) -> dict[str, Any]:
    segments = readout.get("segments")
    if not isinstance(segments, Mapping):
        _fail(f"{label} carries no segments block")
    block = segments.get(segment)
    if not isinstance(block, Mapping):
        _fail(f"{label} carries no {segment} segment")
    if block.get("finite") is not True:
        _fail(f"{label} {segment} segment is not finite")
    return {
        "baseline_sum": assert_finite(block.get("baseline_sum"), label=f"{label}.{segment}"),
        "modified_sum": assert_finite(block.get("modified_sum"), label=f"{label}.{segment}"),
        "delta": assert_finite(block.get("delta"), label=f"{label}.{segment}"),
        "delta_token_mean": assert_finite(
            block.get("delta_token_mean"), label=f"{label}.{segment}"
        ),
        "token_count": int(block.get("token_count", 0)),
        "sign": int(block.get("sign", 0)),
    }


def build_row_mapping(
    inputs: SealedInputs, *, contract: DenominatorContract
) -> dict[str, Any]:
    """The 64 merge rows to 38 secondary owner rows mapping, verified not assumed."""

    per_cohort_variants: dict[str, set[str]] = {}
    per_owner_variants: dict[tuple[str, str], set[str]] = {}
    for row in inputs.merged_rows:
        cohort = str(row.get("cohort"))
        variant = str(row.get("variant"))
        owner_id = str(row.get("gt_owner_id"))
        per_cohort_variants.setdefault(cohort, set()).add(variant)
        key = (cohort, owner_id)
        if variant in per_owner_variants.setdefault(key, set()):
            _fail(f"secondary merge rows repeat variant {variant} for {cohort}/{owner_id}")
        per_owner_variants[key].add(variant)

    crossing_owners = sorted(
        owner for cohort, owner in per_owner_variants if cohort == CROSSING_COHORT
    )
    benign_owners = sorted(
        owner for cohort, owner in per_owner_variants if cohort == BENIGN_COHORT
    )
    if len(crossing_owners) != contract.crossing_owner_count:
        _fail(
            f"secondary merge rows cover {len(crossing_owners)} crossing owners, expected "
            f"{contract.crossing_owner_count}"
        )
    if len(benign_owners) != contract.benign_control_count:
        _fail(
            f"secondary merge rows cover {len(benign_owners)} benign controls, expected "
            f"{contract.benign_control_count}"
        )
    for owner_id in crossing_owners:
        variants = per_owner_variants[(CROSSING_COHORT, owner_id)]
        if variants != {PRIMARY_ARM, SENSITIVITY_ARM}:
            _fail(
                f"crossing owner {owner_id} carries secondary variants {sorted(variants)}, "
                f"expected both {PRIMARY_ARM!r} and {SENSITIVITY_ARM!r}"
            )
    for owner_id in benign_owners:
        variants = per_owner_variants[(BENIGN_COHORT, owner_id)]
        if variants != {BENIGN_VARIANT}:
            _fail(
                f"benign control {owner_id} carries secondary variants {sorted(variants)}, "
                f"expected only {BENIGN_VARIANT!r}"
            )
    expected_rows = contract.crossing_secondary_readout_count + contract.benign_control_count
    if expected_rows != len(inputs.merged_rows):
        _fail(
            f"secondary merge rows hold {len(inputs.merged_rows)} rows but the owner join "
            f"accounts for {expected_rows}"
        )
    return {
        "input_merge_row_count": len(inputs.merged_rows),
        "secondary_owner_row_count": len(inputs.secondary_rows),
        "crossing_owner_count": len(crossing_owners),
        "crossing_readouts_per_owner": len(ARMS),
        "benign_control_count": len(benign_owners),
        "benign_readouts_per_owner": 1,
        "transformation": (
            f"{len(inputs.merged_rows)} executed request rows become "
            f"{len(inputs.secondary_rows)} secondary owner rows "
            f"({len(crossing_owners)} crossing owners x {len(ARMS)} arms + "
            f"{len(benign_owners)} benign controls x 1), and this analyzer emits "
            f"{contract.expected_owner_geometry_row_count} owner-arm geometry rows"
        ),
        "owner_geometry_row_count": contract.expected_owner_geometry_row_count,
    }


def build_benign_reference(
    inputs: SealedInputs, *, contract: DenominatorContract
) -> dict[str, dict[str, Any]]:
    """Exactly one benign-substitution readout per image, joined never recomputed."""

    benign_merge: dict[str, Mapping[str, Any]] = {}
    for row in inputs.merged_rows:
        if row.get("cohort") != BENIGN_COHORT:
            continue
        benign_merge[str(row.get("gt_owner_id"))] = row

    reference: dict[str, dict[str, Any]] = {}
    for row in inputs.secondary_rows:
        if row.get("cohort") != BENIGN_COHORT:
            continue
        image_id = str(row.get("image_id"))
        if image_id in reference:
            _fail(f"image {image_id} carries more than one benign-substitution control")
        readouts = row.get("readouts")
        if not isinstance(readouts, Mapping) or BENIGN_VARIANT not in readouts:
            _fail(f"benign control {row.get('gt_owner_id')} carries no {BENIGN_VARIANT} readout")
        readout = readouts[BENIGN_VARIANT]
        owner_id = str(row.get("gt_owner_id"))
        label = f"benign control {owner_id}"
        merged = benign_merge.get(owner_id)
        if merged is None:
            _fail(f"benign control {owner_id} carries no secondary merge row")
        scored = merged.get("segments")
        if not isinstance(scored, Mapping) or not isinstance(scored.get("coordinates"), Mapping):
            _fail(f"benign control {owner_id} merge row carries no coordinate segment")
        scored_coordinates = list(scored["coordinates"].get("token_ids") or [])
        if len(scored_coordinates) != 4:
            _fail(
                f"benign control {owner_id} scores {len(scored_coordinates)} coordinate tokens, "
                "expected the same four-token span shape as a crossing arm"
            )
        if readout.get("request_id") != merged.get("request_id"):
            _fail(f"benign control {owner_id} readout and merge row name different requests")
        reference[image_id] = {
            "gt_owner_id": owner_id,
            "image_id": image_id,
            "variant": BENIGN_VARIANT,
            "request_id": readout.get("request_id"),
            "scored_token_count": int(readout.get("scored_token_count", 0)),
            "scored_coordinate_token_ids_sha256": sha256_json(scored_coordinates),
            "scored_row_identity": f"{BENIGN_COHORT}|{image_id}|{owner_id}|{BENIGN_VARIANT}",
            "segments": {
                segment: _segment_delta(readout, segment, label=label) for segment in SEGMENTS
            },
        }
    if len(reference) != contract.benign_control_count:
        _fail(
            f"benign reference covers {len(reference)} images, expected "
            f"{contract.benign_control_count}"
        )
    return reference


def build_owner_geometry_rows(
    inputs: SealedInputs, *, contract: DenominatorContract
) -> list[dict[str, Any]]:
    """One row per crossing owner per arm: C/E primary and C/F sensitivity."""

    cohort_index = _index_unique(inputs.cohort_rows, "gt_owner_id", label="crossing cohort registry")
    primary_index = _index_unique(
        (row for row in inputs.primary_rows if row.get("cohort") == CROSSING_COHORT),
        "gt_owner_id",
        label="primary crossing owner rows",
    )
    secondary_index = _index_unique(
        (row for row in inputs.secondary_rows if row.get("cohort") == CROSSING_COHORT),
        "gt_owner_id",
        label="secondary crossing owner rows",
    )
    merged_index: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in inputs.merged_rows:
        if row.get("cohort") != CROSSING_COHORT:
            continue
        merged_index[(str(row.get("gt_owner_id")), str(row.get("variant")))] = row

    if not (set(cohort_index) == set(primary_index) == set(secondary_index)):
        _fail(
            "the crossing owner join is not exact: cohort registry, primary rows and secondary "
            f"rows cover {len(cohort_index)}, {len(primary_index)} and {len(secondary_index)} "
            "owners respectively"
        )
    if len(cohort_index) != contract.crossing_owner_count:
        _fail(
            f"the crossing owner join covers {len(cohort_index)} owners, expected "
            f"{contract.crossing_owner_count}"
        )

    benign = build_benign_reference(inputs, contract=contract)
    population_cache: dict[tuple[str, str], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    for owner_id in sorted(cohort_index):
        cohort = cohort_index[owner_id]
        primary = primary_index[owner_id]
        secondary = secondary_index[owner_id]
        image_id = str(cohort.get("image_id"))
        if primary.get("image_id") != image_id or secondary.get("image_id") != image_id:
            _fail(f"owner {owner_id} carries inconsistent image ids across sealed inputs")
        if image_id not in benign:
            _fail(f"image {image_id} carries no benign-substitution reference")
        ledger = inputs.owner_registry.get(owner_id)
        if ledger is None:
            _fail(f"owner {owner_id} is missing from the canonical owner ledger")

        category_id = cohort.get("official_coco_category_id")
        if not isinstance(category_id, int):
            _fail(f"owner {owner_id} carries no official_coco_category_id")
        c_description = cohort.get("normalized_description")
        if c_description != ledger.get("normalized_description"):
            _fail(
                f"owner {owner_id} description disagrees with the canonical ledger: "
                f"{c_description!r} vs {ledger.get('normalized_description')!r}"
            )
        if ledger.get("official_coco_category_id") != category_id:
            _fail(f"owner {owner_id} category disagrees with the canonical ledger")

        inserted = cohort.get("inserted_clean_row_c")
        if not isinstance(inserted, Mapping):
            _fail(f"owner {owner_id} carries no inserted_clean_row_c")
        c_box = decode_box(inserted.get("coord_token_ids"), label=f"owner {owner_id} inserted C")
        if c_box.as_list() != _owner_anchor_box(ledger, label=f"census owner {owner_id}").as_list():
            _fail(
                f"owner {owner_id} inserted C box does not equal its sealed exact-GT-anchor box"
            )

        stratum = str(primary.get("stratum"))
        if stratum not in E_STRATUM_AXIS:
            _fail(f"owner {owner_id} carries an unknown stratum {stratum!r}")
        cache_key = (image_id, str(c_description))
        if cache_key not in population_cache:
            population_cache[cache_key] = build_owner_population(
                inputs, image_id=image_id, normalized_description=str(c_description)
            )
        population = population_cache[cache_key]

        readouts = secondary.get("readouts")
        if not isinstance(readouts, Mapping):
            _fail(f"owner {owner_id} carries no secondary readouts")

        for arm, row_key, row_label, arm_role in ARMS:
            sealed_row = cohort.get(row_key)
            if not isinstance(sealed_row, Mapping):
                _fail(f"owner {owner_id} carries no sealed {row_key}")
            if row_key == "f_row" and cohort.get("f_row_present") is not True:
                _fail(f"owner {owner_id} has no sealed F row but the sensitivity arm requires one")
            r_box = decode_box(
                sealed_row.get("coord_token_ids"), label=f"owner {owner_id} {row_label}"
            )

            readout = readouts.get(arm)
            if not isinstance(readout, Mapping):
                _fail(f"owner {owner_id} carries no {arm} readout")
            merged = merged_index.get((owner_id, arm))
            if merged is None:
                _fail(f"owner {owner_id} carries no {arm} secondary merge row")

            # The arm/geometry binding, enforced against the exact scored token span.
            scored = merged.get("segments")
            if not isinstance(scored, Mapping) or not isinstance(
                scored.get("coordinates"), Mapping
            ):
                _fail(f"owner {owner_id} {arm} merge row carries no coordinate segment")
            scored_tokens = list(scored["coordinates"].get("token_ids") or [])
            if len(scored_tokens) != 4:
                _fail(
                    f"owner {owner_id} {arm} scores {len(scored_tokens)} coordinate tokens, "
                    "expected four"
                )
            if scored_tokens != list(sealed_row.get("coord_token_ids") or []):
                _fail(
                    f"owner {owner_id} {arm} scored coordinate tokens do not equal the sealed "
                    f"{row_key} coordinate tokens; the arm would be measured against the wrong "
                    "geometry"
                )
            if list(merged.get("scored_token_ids") or []) != list(
                sealed_row.get("full_row_token_ids") or []
            ):
                _fail(
                    f"owner {owner_id} {arm} scored row tokens do not equal the sealed {row_key} "
                    "row tokens"
                )
            if readout.get("request_id") != merged.get("request_id"):
                _fail(f"owner {owner_id} {arm} readout and merge row name different requests")

            r_description = sealed_row.get("normalized_description")
            description_equal = c_description == r_description
            if row_key == "e_row":
                sealed_equal = cohort.get("same_description_as_e")
                if not isinstance(sealed_equal, bool):
                    _fail(f"owner {owner_id} carries no sealed same_description_as_e")
                if sealed_equal != description_equal:
                    _fail(
                        f"owner {owner_id} sealed same_description_as_e={sealed_equal} disagrees "
                        f"with the exact normalized descriptions {c_description!r}/"
                        f"{r_description!r}"
                    )
                description_equal_source = "sealed_same_description_as_e"
                description_equal = sealed_equal
            else:
                description_equal_source = "sealed_f_row_normalized_description"

            strict_status = sealed_row.get("strict_match_status")
            strict_owner = sealed_row.get("strict_match_gt_owner_id")
            if strict_status not in {"matched", "unmatched"}:
                _fail(f"owner {owner_id} {row_key} carries an unknown strict_match_status")
            if strict_status == "matched":
                if not isinstance(strict_owner, str) or strict_owner not in inputs.owner_registry:
                    _fail(
                        f"owner {owner_id} {row_key} matches {strict_owner!r}, which is missing "
                        "from the canonical owner ledger"
                    )
                matched_ledger = inputs.owner_registry[strict_owner]
                if matched_ledger.get("normalized_description") != r_description:
                    _fail(
                        f"owner {owner_id} {row_key} description {r_description!r} disagrees with "
                        f"its matched owner {strict_owner}"
                    )
            elif strict_owner is not None:
                _fail(f"owner {owner_id} {row_key} is unmatched but names owner {strict_owner!r}")
            if row_key == "e_row":
                expected_stratum = "matched_e" if strict_status == "matched" else "unmatched_e"
                if expected_stratum != stratum:
                    _fail(
                        f"owner {owner_id} sealed stratum {stratum!r} disagrees with the sealed E "
                        f"strict-match status {strict_status!r}"
                    )

            geometry = box_geometry(c_box, r_box, label=f"owner {owner_id} C/{row_label}")
            ranks = sorted_key_rank_gap(
                population,
                c_owner_id=owner_id,
                r_box=r_box,
                r_tie_key=f"row:{arm}:{owner_id}",
            )

            label = f"owner {owner_id} {arm}"
            segments = {
                segment: _segment_delta(readout, segment, label=label) for segment in SEGMENTS
            }
            benign_row = benign[image_id]
            # The two deltas are summed over different scored rows -- the arm's own
            # downstream row and this image's benign control's own downstream row.
            arm_row_identity = f"{CROSSING_COHORT}|{image_id}|{owner_id}|{arm}"
            if arm_row_identity == benign_row["scored_row_identity"]:
                _fail(
                    f"owner {owner_id} {arm} and its benign reference name the same scored row"
                )
            if readout.get("request_id") == benign_row["request_id"]:
                _fail(
                    f"owner {owner_id} {arm} and its benign reference name the same request"
                )
            relative: dict[str, Any] = {}
            for segment in SEGMENTS:
                crossing_delta = segments[segment]["delta"]
                benign_delta = benign_row["segments"][segment]["delta"]
                relative[segment] = {
                    "crossing_delta": crossing_delta,
                    "same_image_benign_delta": benign_delta,
                    "relative_delta": crossing_delta - benign_delta,
                }
            relative_coordinate_delta = relative["coordinates"]["relative_delta"]
            material_negative = relative_coordinate_delta <= MATERIAL_NEGATIVE_MAX_NATS

            displacement = primary.get("displacement")
            if not isinstance(displacement, Mapping):
                _fail(f"owner {owner_id} carries no displacement block")

            rows.append(
                {
                    "schema_version": OWNER_ROW_SCHEMA_VERSION,
                    "unit_id": UNIT_ID,
                    "row_kind": "crossing_owner_geometry_row",
                    "arm": arm,
                    "arm_role": arm_role,
                    "downstream_row_label": row_label,
                    "enters_primary_decision": arm == PRIMARY_ARM,
                    "image_id": image_id,
                    "gt_owner_id": owner_id,
                    "official_coco_category_id": category_id,
                    "c_normalized_description": c_description,
                    "r_normalized_description": r_description,
                    "description_equal": description_equal,
                    "description_equal_source": description_equal_source,
                    "description_axis": (
                        "same_description" if description_equal else "different_description"
                    ),
                    "r_strict_match_status": strict_status,
                    "r_strict_match_gt_owner_id": strict_owner,
                    "r_row_index": sealed_row.get("row_index"),
                    "r_pred_row_id": sealed_row.get("pred_row_id"),
                    "geometry": geometry,
                    "sorted_key_ranks": ranks,
                    "joined": {
                        "stratum": stratum,
                        "primary_branch": primary.get("primary_branch"),
                        "primary_branch_reasons": list(primary.get("primary_branch_reasons") or []),
                        "sensitivity_branch_l": primary.get("sensitivity_branch_l"),
                        "description_observability": primary.get("description_observability"),
                        "same_description_as_e": cohort.get("same_description_as_e"),
                        "interpretable": primary.get("interpretable"),
                        "quarantined": primary.get("quarantined"),
                        "tie_or_nonunique": primary.get("tie_or_nonunique"),
                        "displacement": {
                            "greedy_displaced": bool(displacement.get("greedy_displaced")),
                            "greedy_displaced_owner_id": displacement.get(
                                "greedy_displaced_owner_id"
                            ),
                            "likelihood_displaced": bool(displacement.get("likelihood_displaced")),
                            "likelihood_displaced_owner_id": displacement.get(
                                "likelihood_displaced_owner_id"
                            ),
                            "decoding_contradicted": bool(
                                displacement.get("decoding_contradicted")
                            ),
                        },
                    },
                    "likelihood": {
                        "request_id": readout.get("request_id"),
                        "scored_target_kind": readout.get("scored_target_kind"),
                        "scored_token_count": int(readout.get("scored_token_count", 0)),
                        "inserted_clean_row_c_token_count": int(
                            readout.get("inserted_clean_row_c_token_count", 0)
                        ),
                        "segments": segments,
                        "scored_row_identity": arm_row_identity,
                        "scored_coordinate_token_ids_sha256": sha256_json(scored_tokens),
                        "benign_reference": {
                            "gt_owner_id": benign_row["gt_owner_id"],
                            "request_id": benign_row["request_id"],
                            "scored_row_identity": benign_row["scored_row_identity"],
                            "scored_coordinate_token_ids_sha256": benign_row[
                                "scored_coordinate_token_ids_sha256"
                            ],
                            "scored_row_is_distinct_from_this_arm": True,
                        },
                        "benign_reference_gt_owner_id": benign_row["gt_owner_id"],
                        "relative": relative,
                    },
                    "relative_coordinate_delta": relative_coordinate_delta,
                    "relative_complete_row_delta": relative["complete_row"]["relative_delta"],
                    "material_negative": material_negative,
                    "change_axis": "material_negative" if material_negative else "nonmaterial",
                }
            )

    if len(rows) != contract.expected_owner_geometry_row_count:
        _fail(
            f"owner geometry rows hold {len(rows)} rows, expected "
            f"{contract.expected_owner_geometry_row_count}"
        )
    rows.sort(key=lambda row: (row["arm"], row["image_id"], row["gt_owner_id"]))
    return rows


def build_displacement_pair_rows(
    inputs: SealedInputs, *, contract: DenominatorContract
) -> list[dict[str, Any]]:
    """One row per frozen greedy displacement pair: target GT owner versus displacer."""

    cohort_index = _index_unique(inputs.cohort_rows, "gt_owner_id", label="crossing cohort registry")
    population_cache: dict[tuple[str, str], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    for primary in inputs.primary_rows:
        if primary.get("cohort") != CROSSING_COHORT:
            continue
        displacement = primary.get("displacement")
        if not isinstance(displacement, Mapping):
            _fail(f"owner {primary.get('gt_owner_id')} carries no displacement block")
        if not displacement.get("greedy_displaced"):
            continue
        target_id = str(primary.get("gt_owner_id"))
        displacer_id = displacement.get("greedy_displaced_owner_id")
        if not isinstance(displacer_id, str) or not displacer_id:
            _fail(f"owner {target_id} is greedy-displaced but names no displacer")
        if displacer_id == target_id:
            _fail(f"owner {target_id} names itself as its own greedy displacer")
        cohort = cohort_index.get(target_id)
        if cohort is None:
            _fail(f"greedy-displaced owner {target_id} is missing from the cohort registry")

        target_ledger = inputs.owner_registry.get(target_id)
        displacer_ledger = inputs.owner_registry.get(displacer_id)
        if target_ledger is None or displacer_ledger is None:
            _fail(
                f"greedy displacement pair {target_id}->{displacer_id} is missing from the "
                "canonical owner ledger"
            )
        image_id = str(target_ledger.get("image_id"))
        if displacer_ledger.get("image_id") != image_id:
            _fail(f"greedy displacement pair {target_id}->{displacer_id} spans two images")
        category_id = target_ledger.get("official_coco_category_id")
        if not isinstance(category_id, int):
            _fail(f"owner {target_id} carries no official_coco_category_id")

        target_box = _owner_anchor_box(target_ledger, label=f"census owner {target_id}")
        displacer_box = _owner_anchor_box(displacer_ledger, label=f"census owner {displacer_id}")
        geometry = box_geometry(
            target_box, displacer_box, label=f"pair {target_id}/{displacer_id}"
        )

        target_description = target_ledger.get("normalized_description")
        same_category = (
            displacer_ledger.get("normalized_description") == target_description
            and displacer_ledger.get("official_coco_category_id") == category_id
        )
        cache_key = (image_id, str(target_description))
        if cache_key not in population_cache:
            population_cache[cache_key] = build_owner_population(
                inputs, image_id=image_id, normalized_description=str(target_description)
            )
        population = population_cache[cache_key]
        owner_ids = [item["gt_owner_id"] for item in population["members"]]
        if target_id not in owner_ids:
            _fail(f"target {target_id} is missing from its own same-category population")
        if not same_category:
            _fail(
                f"greedy displacement pair {target_id}->{displacer_id} crosses categories; the "
                "same-category sorted-rank gap is undefined for it"
            )
        target_rank = owner_ids.index(target_id)
        displacer_rank = owner_ids.index(displacer_id)

        disposition = displacer_ledger.get("disposition_eligibility")
        if not isinstance(disposition, Mapping):
            _fail(f"displacer {displacer_id} carries no disposition_eligibility block")
        native_tp = bool(displacer_ledger.get("native_true_positive"))
        native_fn = bool(disposition.get("native_false_negative"))
        if native_tp == native_fn:
            _fail(
                f"displacer {displacer_id} carries a contradictory native disposition "
                f"(tp={native_tp}, fn={native_fn})"
            )

        rows.append(
            {
                "schema_version": PAIR_ROW_SCHEMA_VERSION,
                "unit_id": UNIT_ID,
                "row_kind": "greedy_displacement_pair_row",
                "image_id": image_id,
                "target_gt_owner_id": target_id,
                "displacer_gt_owner_id": displacer_id,
                "official_coco_category_id": category_id,
                "target_normalized_description": target_ledger.get("normalized_description"),
                "displacer_normalized_description": displacer_ledger.get(
                    "normalized_description"
                ),
                "description_equal": target_ledger.get("normalized_description")
                == displacer_ledger.get("normalized_description"),
                "same_category": same_category,
                "geometry": geometry,
                "sorted_key_ranks": {
                    "population_size": len(owner_ids),
                    "population_normalized_description": population["normalized_description"],
                    "target_rank": target_rank,
                    "displacer_rank": displacer_rank,
                    "sorted_key_rank_gap": displacer_rank - target_rank,
                    "pixel_sort_key_order_agrees": population["pixel_sort_key_order_agrees"],
                    "pixel_sort_key_role": "provenance_only_never_mixed_into_this_rank",
                },
                "displacer_native_disposition": {
                    "native_true_positive": native_tp,
                    "native_false_negative": native_fn,
                    "calibration_role": displacer_ledger.get("calibration_role"),
                    "greedy_eligibility_status": displacer_ledger.get(
                        "greedy_eligibility_status"
                    ),
                    "source": "frozen census owner ledger",
                },
                "joined": {
                    "primary_branch": primary.get("primary_branch"),
                    "likelihood_displaced": bool(displacement.get("likelihood_displaced")),
                    "likelihood_displaced_owner_id": displacement.get(
                        "likelihood_displaced_owner_id"
                    ),
                    "decoding_contradicted": bool(displacement.get("decoding_contradicted")),
                    "stratum": primary.get("stratum"),
                    "same_description_as_e": cohort.get("same_description_as_e"),
                },
                "claim_boundary": (
                    "descriptive compatibility check only; it can never change the frozen "
                    "three-way decision and implies no causality"
                ),
            }
        )

    if len(rows) != contract.greedy_pair_count:
        _fail(
            f"greedy displacement pairs hold {len(rows)} rows, expected "
            f"{contract.greedy_pair_count}"
        )
    rows.sort(key=lambda row: (row["image_id"], row["target_gt_owner_id"]))
    return rows


# ---------------------------------------------------------------------------
# 5. Strata, per-image values, associations and the frozen decision
# ---------------------------------------------------------------------------


def build_strata(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The full Cartesian grid, including zero cells, one block per arm."""

    strata: dict[str, Any] = {}
    for arm, _row_key, _label, arm_role in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        cells: list[dict[str, Any]] = []
        for description in DESCRIPTION_AXIS:
            for stratum in E_STRATUM_AXIS:
                for geometric_label in GEOMETRIC_LABELS:
                    for change in CHANGE_AXIS:
                        members = [
                            row["gt_owner_id"]
                            for row in arm_rows
                            if row["description_axis"] == description
                            and row["joined"]["stratum"] == stratum
                            and row["geometry"]["geometric_label"] == geometric_label
                            and row["change_axis"] == change
                        ]
                        cells.append(
                            {
                                "description": description,
                                "e_stratum": stratum,
                                "geometric_label": geometric_label,
                                "change": change,
                                "count": len(members),
                                "gt_owner_ids": sorted(members),
                            }
                        )
        covered = sum(cell["count"] for cell in cells)
        if covered != len(arm_rows):
            _fail(
                f"the Cartesian strata for arm {arm} cover {covered} rows but the arm holds "
                f"{len(arm_rows)}"
            )
        strata[arm] = {
            "arm_role": arm_role,
            "row_count": len(arm_rows),
            "axes": {
                "description": list(DESCRIPTION_AXIS),
                "e_stratum": list(E_STRATUM_AXIS),
                "geometric_label": list(GEOMETRIC_LABELS),
                "change": list(CHANGE_AXIS),
            },
            "e_stratum_scope": (
                "the sealed owner-level matched-E / unmatched-E stratum, held fixed across both "
                "arms so one owner is described the same way in each"
            ),
            "cells": cells,
        }
    return strata


def build_population_provenance(
    rows: Sequence[Mapping[str, Any]], pair_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Rank provenance: how each same-description population behaved, pixel key included."""

    tie_rows = [
        f"{row['arm']}:{row['gt_owner_id']}"
        for row in rows
        if row["sorted_key_ranks"]["rank_key_tie_with_physical_owner"]
    ]
    disagreeing = sorted(
        {
            f"{row['image_id']}|{row['sorted_key_ranks']['population_normalized_description']}"
            for row in [*rows, *pair_rows]
            if not row["sorted_key_ranks"]["pixel_sort_key_order_agrees"]
        }
    )
    return {
        "same_category_definition": OPERATIONAL_DEFINITIONS["same_category_population"],
        "rank_space": "sealed_norm1000_coordinate_tokens_only",
        "pixel_sort_key_role": "provenance_only_never_mixed_into_this_rank",
        "populations_where_pixel_order_disagrees": disagreeing,
        "rows_with_an_exact_key_tie_against_a_physical_owner": sorted(tie_rows),
        "inserted_row_tie_rule": "the inserted row sorts after every equal-key physical owner",
    }


def build_per_image(
    rows: Sequence[Mapping[str, Any]], benign: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Raw deltas stay indexed by image; only within-image summaries are computed."""

    per_image: dict[str, Any] = {}
    for image_id in sorted({row["image_id"] for row in rows}):
        arms_block: dict[str, Any] = {}
        for arm, _row_key, _label, _role in ARMS:
            image_rows = [
                row for row in rows if row["image_id"] == image_id and row["arm"] == arm
            ]
            values = [row["relative_coordinate_delta"] for row in image_rows]
            arms_block[arm] = {
                "owner_count": len(image_rows),
                "material_negative_count": sum(
                    1 for row in image_rows if row["material_negative"]
                ),
                "within_image_median_relative_coordinate_delta": (
                    statistics.median(values) if values else None
                ),
                "rows": [
                    {
                        "gt_owner_id": row["gt_owner_id"],
                        "crossing_coordinate_delta": row["likelihood"]["relative"]["coordinates"][
                            "crossing_delta"
                        ],
                        "relative_coordinate_delta": row["relative_coordinate_delta"],
                        "relative_complete_row_delta": row["relative_complete_row_delta"],
                        "iou": row["geometry"]["iou"],
                        "center_distance_normalized": row["geometry"][
                            "center_distance_normalized"
                        ],
                        "geometric_label": row["geometry"]["geometric_label"],
                        "description_axis": row["description_axis"],
                        "material_negative": row["material_negative"],
                    }
                    for row in sorted(image_rows, key=lambda item: item["gt_owner_id"])
                ],
            }
        reference = benign.get(image_id)
        if reference is None:
            _fail(f"image {image_id} carries no benign-substitution reference")
        per_image[image_id] = {
            "arms": arms_block,
            "benign_reference": {
                "gt_owner_id": reference["gt_owner_id"],
                "coordinate_delta": reference["segments"]["coordinates"]["delta"],
                "complete_row_delta": reference["segments"]["complete_row"]["delta"],
                "role": (
                    "one same-length replacement readout for this image, subtracted from this "
                    "image's crossing deltas only"
                ),
            },
        }
    return per_image


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        stop = position
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[position]]:
            stop += 1
        average = (position + stop) / 2.0 + 1.0
        for index in range(position, stop + 1):
            ranks[order[index]] = average
        position = stop + 1
    return ranks


def spearman_rho(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Tie-corrected Spearman rank association, or None when it is undefined."""

    if len(xs) != len(ys):
        _fail("spearman inputs must be the same length")
    if len(xs) < 3:
        return None
    rank_x = _average_ranks(xs)
    rank_y = _average_ranks(ys)
    mean_x = sum(rank_x) / len(rank_x)
    mean_y = sum(rank_y) / len(rank_y)
    covariance = sum((a - mean_x) * (b - mean_y) for a, b in zip(rank_x, rank_y))
    variance_x = sum((a - mean_x) ** 2 for a in rank_x)
    variance_y = sum((b - mean_y) ** 2 for b in rank_y)
    if variance_x <= 0.0 or variance_y <= 0.0:
        return None
    return covariance / math.sqrt(variance_x * variance_y)


def build_associations(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Descriptive rank associations with leave-one-image-out coefficients."""

    measures = {
        "iou": lambda row: row["geometry"]["iou"],
        "center_distance": lambda row: row["geometry"]["center_distance_normalized"],
    }
    associations: dict[str, Any] = {}
    for arm, _row_key, _label, arm_role in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        images = sorted({row["image_id"] for row in arm_rows})
        block: dict[str, Any] = {
            "arm_role": arm_role,
            "sample_size": len(arm_rows),
            "image_count": len(images),
        }
        for name, getter in measures.items():
            xs = [getter(row) for row in arm_rows]
            ys = [row["relative_coordinate_delta"] for row in arm_rows]
            leave_one_out: dict[str, Any] = {}
            for image_id in images:
                kept = [row for row in arm_rows if row["image_id"] != image_id]
                leave_one_out[image_id] = {
                    "held_out_image_id": image_id,
                    "sample_size": len(kept),
                    "spearman_rho": spearman_rho(
                        [getter(row) for row in kept],
                        [row["relative_coordinate_delta"] for row in kept],
                    ),
                }
            block[f"{name}_versus_relative_coordinate_delta"] = {
                "measure": name,
                "spearman_rho": spearman_rho(xs, ys),
                "sample_size": len(arm_rows),
                "tie_handling": "average_ranks",
                "association_role": "descriptive_only_never_causal",
                "leave_one_image_out": leave_one_out,
            }
        associations[arm] = block
    return associations


def _arm_decision_block(rows: Sequence[Mapping[str, Any]], *, arm: str) -> dict[str, Any]:
    arm_rows = [row for row in rows if row["arm"] == arm]
    material = [row for row in arm_rows if row["material_negative"]]
    qualifying = [
        row
        for row in material
        if row["description_equal"] and row["geometry"]["any_overlap_or_center_containment"]
    ]
    separated = [row for row in material if row["geometry"]["clearly_separated"]]
    separated_different_description = [
        row for row in separated if not row["description_equal"]
    ]
    fraction = len(qualifying) / len(material) if material else None

    identity_images = sorted({row["image_id"] for row in qualifying})
    separated_images = sorted({row["image_id"] for row in separated})

    # Separated competition is tested first; identity slippage can only be routed when
    # fewer than three M_E rows are clearly separated.
    separated_count_holds = len(separated) >= SEPARATED_COMPETITION_MIN_ROWS
    separated_spans_images = len(separated_images) >= MIN_QUALIFYING_IMAGE_COUNT
    separated_survives = separated_count_holds and separated_spans_images

    identity_core_holds = (
        len(material) >= IDENTITY_SLIPPAGE_MIN_ROWS
        and fraction is not None
        and fraction >= IDENTITY_SLIPPAGE_MIN_FRACTION
        and not separated_different_description
    )
    identity_spans_images = len(identity_images) >= MIN_QUALIFYING_IMAGE_COUNT
    identity_slippage = identity_core_holds and identity_spans_images and not separated_count_holds

    if separated_survives:
        outcome = OUTCOME_SEPARATED_COMPETITION
    elif identity_slippage:
        outcome = OUTCOME_IDENTITY_SLIPPAGE
    else:
        outcome = OUTCOME_INCONCLUSIVE
    return {
        "arm": arm,
        "row_count": len(arm_rows),
        "distinct_owner_count": len({row["gt_owner_id"] for row in arm_rows}),
        "material_negative_count": len(material),
        "material_negative_gt_owner_ids": sorted(row["gt_owner_id"] for row in material),
        "same_description_and_any_overlap_count": len(qualifying),
        "same_description_and_any_overlap_fraction": fraction,
        "same_description_and_any_overlap_image_ids": identity_images,
        "clearly_separated_count": len(separated),
        "clearly_separated_image_ids": separated_images,
        "clearly_separated_different_description_count": len(separated_different_description),
        "predicates": {
            "material_negative_count_at_least_minimum": len(material)
            >= IDENTITY_SLIPPAGE_MIN_ROWS,
            "same_description_and_any_overlap_fraction_at_least_minimum": (
                fraction is not None and fraction >= IDENTITY_SLIPPAGE_MIN_FRACTION
            ),
            "zero_material_negative_clearly_separated_different_description": (
                not separated_different_description
            ),
            "clearly_separated_count_at_least_minimum": separated_count_holds,
            "identity_qualifying_rows_span_two_images": identity_spans_images,
            "separated_qualifying_rows_span_two_images": separated_spans_images,
            "identity_route_blocked_by_separated_precedence": (
                identity_core_holds and identity_spans_images and separated_count_holds
            ),
            "identity_slippage_duplicate_suppression_eligible": identity_slippage,
            "separated_competition_survives": separated_survives,
        },
        "outcome": outcome,
    }


def build_decision(
    rows: Sequence[Mapping[str, Any]], pair_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """The frozen three-way rule, evaluated only over the primary C/E arm."""

    primary_block = _arm_decision_block(rows, arm=PRIMARY_ARM)
    sensitivity_block = _arm_decision_block(rows, arm=SENSITIVITY_ARM)
    sensitivity_block["role"] = (
        "named sensitivity over the optional C/F arm; it may qualify a reading but never "
        "changes the primary decision"
    )
    if primary_block["distinct_owner_count"] != primary_block["row_count"]:
        _fail("the primary decision arm would count an owner more than once")

    adjacent = [
        row
        for row in pair_rows
        if row["geometry"]["any_overlap_or_center_containment"]
    ]
    distant = [row for row in pair_rows if row["geometry"]["clearly_separated"]]
    if adjacent and len(adjacent) > len(distant):
        pair_reading = "identity_local"
    elif distant and len(distant) > len(adjacent):
        pair_reading = "distributed_same_category_reprioritization"
    else:
        pair_reading = "no_majority"

    return {
        "rule_id": "sorted-crossing-owner-row-geometry-three-way.v1",
        "exhaustive_order": list(DECISION_EXHAUSTIVE_ORDER),
        "cutoffs": {
            "high_overlap_min_iou": HIGH_OVERLAP_MIN_IOU,
            "material_negative_max_nats": MATERIAL_NEGATIVE_MAX_NATS,
            "identity_slippage_min_rows": IDENTITY_SLIPPAGE_MIN_ROWS,
            "identity_slippage_min_fraction": IDENTITY_SLIPPAGE_MIN_FRACTION,
            "separated_competition_min_rows": SEPARATED_COMPETITION_MIN_ROWS,
            "min_qualifying_image_count": MIN_QUALIFYING_IMAGE_COUNT,
            "frozen_before_analysis": True,
        },
        "route_precedence": (
            "separated competition is tested first; identity slippage may be routed only when "
            "fewer than three M_E rows are clearly separated"
        ),
        "image_span_requirement": (
            "a non-inconclusive route needs qualifying rows from at least "
            f"{MIN_QUALIFYING_IMAGE_COUNT} distinct images, because rows of one image share that "
            "image's single benign reference delta"
        ),
        "primary_arm": primary_block,
        "sensitivity_arm": sensitivity_block,
        "outcome": primary_block["outcome"],
        "pseudo_replication_guard": {
            "primary_decision_uses_only_the_primary_arm": True,
            "one_owner_one_vote": True,
            "sensitivity_can_never_change_the_outcome": True,
        },
        "greedy_pair_compatibility_check": {
            "pair_count": len(pair_rows),
            "any_overlap_or_center_containment_count": len(adjacent),
            "clearly_separated_count": len(distant),
            "reading": pair_reading,
            "authority": "descriptive only; it cannot change the three-way decision",
        },
    }


# ---------------------------------------------------------------------------
# 6. Visualization plan / manifest
# ---------------------------------------------------------------------------


def _crop_window(c_box: Box, r_box: Box) -> dict[str, int]:
    x1 = min(c_box.x1, r_box.x1)
    y1 = min(c_box.y1, r_box.y1)
    x2 = max(c_box.x2, r_box.x2)
    y2 = max(c_box.y2, r_box.y2)
    margin_x = max(CROP_MIN_HALF_EXTENT_BINS, int(round((x2 - x1) * CROP_MARGIN_FRACTION)))
    margin_y = max(CROP_MIN_HALF_EXTENT_BINS, int(round((y2 - y1) * CROP_MARGIN_FRACTION)))
    return {
        "x1": max(0, x1 - margin_x),
        "y1": max(0, y1 - margin_y),
        "x2": min(CANVAS_BINS, x2 + margin_x),
        "y2": min(CANVAS_BINS, y2 + margin_y),
    }


def build_visual_plan(
    inputs: SealedInputs,
    rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    *,
    media_root: str,
) -> dict[str, Any]:
    """A machine-readable plan a later renderer consumes; this module draws nothing."""

    def media(image_id: str) -> dict[str, Any]:
        registry = inputs.image_registry.get(image_id)
        if registry is None:
            _fail(f"image {image_id} is missing from the census image registry")
        return {
            "image_id": image_id,
            "media_root": media_root,
            "file_name": registry.get("file_name"),
            "image_width": registry.get("image_width"),
            "image_height": registry.get("image_height"),
            "executed_media_sha256": registry.get("executed_media_sha256"),
            "media_identity_note": (
                "executed_media_sha256 identifies the executed media tensor, not the file on "
                "disk; the renderer owns root resolution and verification"
            ),
        }

    scatter_panels: list[dict[str, Any]] = []
    for arm, _row_key, row_label, arm_role in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        for measure, accessor in (
            ("iou", lambda row: row["geometry"]["iou"]),
            ("center_distance_normalized", lambda row: row["geometry"]["center_distance_normalized"]),
        ):
            scatter_panels.append(
                {
                    "panel_id": f"scatter:{arm}:{measure}",
                    "output_file": f"scatter-{arm}-{measure}.png",
                    "arm": arm,
                    "arm_role": arm_role,
                    "x_axis": {"field": measure, "label": measure.replace("_", " ")},
                    "y_axis": {
                        "field": "relative_coordinate_delta",
                        "label": "relative coordinate delta (nat)",
                    },
                    "reference_lines": [
                        {
                            "axis": "y",
                            "value": MATERIAL_NEGATIVE_MAX_NATS,
                            "label": "material_negative cutoff",
                        }
                    ],
                    "points": [
                        {
                            "point_id": f"{arm}:{row['gt_owner_id']}",
                            "image_id": row["image_id"],
                            "gt_owner_id": row["gt_owner_id"],
                            "x": accessor(row),
                            "y": row["relative_coordinate_delta"],
                            "geometric_label": row["geometry"]["geometric_label"],
                            "description_axis": row["description_axis"],
                            "material_negative": row["material_negative"],
                            "color": ROLE_PALETTE[
                                "downstream_row_e" if row_label == "E" else "downstream_row_f"
                            ],
                        }
                        for row in sorted(
                            arm_rows, key=lambda item: (item["image_id"], item["gt_owner_id"])
                        )
                    ],
                }
            )

    crop_panels: list[dict[str, Any]] = []
    for row in rows:
        if not row["material_negative"]:
            continue
        c_box = Box(*row["geometry"]["c_box_norm1000_xyxy"])
        r_box = Box(*row["geometry"]["r_box_norm1000_xyxy"])
        label = row["downstream_row_label"]
        crop_panels.append(
            {
                "panel_id": f"crop:{row['arm']}:{row['gt_owner_id']}",
                "output_file": (
                    f"crop-{row['arm']}-{row['image_id']}-"
                    f"{row['gt_owner_id'].replace(':', '_')}.png"
                ),
                "panel_kind": "material_negative_case",
                "arm": row["arm"],
                "arm_role": row["arm_role"],
                "image": media(row["image_id"]),
                "crop_window_norm1000_xyxy": _crop_window(c_box, r_box),
                "boxes": [
                    {
                        "box_id": f"C:{row['gt_owner_id']}",
                        "role": "inserted_owner_c",
                        "legend": f"C {row['gt_owner_id']} ({row['c_normalized_description']})",
                        "norm1000_xyxy": c_box.as_list(),
                        "color": ROLE_PALETTE["inserted_owner_c"],
                    },
                    {
                        "box_id": f"{label}:{row['gt_owner_id']}",
                        "role": "downstream_row_e" if label == "E" else "downstream_row_f",
                        "legend": (
                            f"{label} row {row['r_row_index']} "
                            f"({row['r_normalized_description']}, {row['r_strict_match_status']})"
                        ),
                        "norm1000_xyxy": r_box.as_list(),
                        "color": ROLE_PALETTE[
                            "downstream_row_e" if label == "E" else "downstream_row_f"
                        ],
                    },
                ],
                "annotations": {
                    "iou": row["geometry"]["iou"],
                    "center_distance_normalized": row["geometry"]["center_distance_normalized"],
                    "relative_coordinate_delta": row["relative_coordinate_delta"],
                    "geometric_label": row["geometry"]["geometric_label"],
                    "description_axis": row["description_axis"],
                    "sorted_key_rank_gap": row["sorted_key_ranks"]["sorted_key_rank_gap"],
                },
            }
        )

    for pair in pair_rows:
        target_box = Box(*pair["geometry"]["c_box_norm1000_xyxy"])
        displacer_box = Box(*pair["geometry"]["r_box_norm1000_xyxy"])
        crop_panels.append(
            {
                "panel_id": f"crop:greedy_pair:{pair['target_gt_owner_id']}",
                "output_file": (
                    f"crop-greedy-pair-{pair['image_id']}-"
                    f"{pair['target_gt_owner_id'].replace(':', '_')}.png"
                ),
                "panel_kind": "greedy_displacement_pair",
                "arm": "greedy_displacement_pair",
                "arm_role": "descriptive_compatibility_check",
                "image": media(pair["image_id"]),
                "crop_window_norm1000_xyxy": _crop_window(target_box, displacer_box),
                "boxes": [
                    {
                        "box_id": f"target:{pair['target_gt_owner_id']}",
                        "role": "displacement_target",
                        "legend": (
                            f"target {pair['target_gt_owner_id']} "
                            f"({pair['target_normalized_description']})"
                        ),
                        "norm1000_xyxy": target_box.as_list(),
                        "color": ROLE_PALETTE["displacement_target"],
                    },
                    {
                        "box_id": f"displacer:{pair['displacer_gt_owner_id']}",
                        "role": "displacer",
                        "legend": (
                            f"displacer {pair['displacer_gt_owner_id']} "
                            f"({pair['displacer_normalized_description']})"
                        ),
                        "norm1000_xyxy": displacer_box.as_list(),
                        "color": ROLE_PALETTE["displacer"],
                    },
                ],
                "annotations": {
                    "iou": pair["geometry"]["iou"],
                    "center_distance_normalized": pair["geometry"][
                        "center_distance_normalized"
                    ],
                    "geometric_label": pair["geometry"]["geometric_label"],
                    "sorted_key_rank_gap": pair["sorted_key_ranks"]["sorted_key_rank_gap"],
                    "displacer_native_true_positive": pair["displacer_native_disposition"][
                        "native_true_positive"
                    ],
                },
            }
        )

    panels = [*scatter_panels, *crop_panels]
    output_files = [panel["output_file"] for panel in panels]
    if len(set(output_files)) != len(output_files):
        _fail("the visualization plan names a duplicate output file")
    return {
        "schema_version": VISUAL_PLAN_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "plan_role": (
            "machine-readable render plan and manifest; this analyzer draws nothing and opens "
            "no media"
        ),
        "palette": dict(ROLE_PALETTE),
        "render_projection": (
            "boxes and crop windows are norm-1000 bins; a renderer multiplies by the sealed "
            "image extent for display only, and no geometric measure in this unit depends on it"
        ),
        "material_negative_case_count": sum(1 for row in rows if row["material_negative"]),
        "greedy_pair_panel_count": len(pair_rows),
        "panel_count": len(panels),
        "expected_output_files": sorted(output_files),
        "scatter_panels": scatter_panels,
        "crop_panels": crop_panels,
    }


# ---------------------------------------------------------------------------
# 7. Summary and report
# ---------------------------------------------------------------------------


def build_summary(
    inputs: SealedInputs,
    rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    *,
    contract: DenominatorContract,
    row_mapping: Mapping[str, Any],
) -> dict[str, Any]:
    benign = build_benign_reference(inputs, contract=contract)
    summary: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "source_unit_id": SOURCE_UNIT_ID,
        "census_unit_id": CENSUS_UNIT_ID,
        "compute_scope": "cpu_only_no_model_no_gpu_no_rescoring",
        "denominators": {
            "crossing_owner_count": contract.crossing_owner_count,
            "primary_ce_row_count": sum(1 for row in rows if row["arm"] == PRIMARY_ARM),
            "sensitivity_cf_row_count": sum(1 for row in rows if row["arm"] == SENSITIVITY_ARM),
            "greedy_displacement_pair_count": len(pair_rows),
            "benign_control_count": len(benign),
            "image_count": len({row["image_id"] for row in rows}),
        },
        "row_mapping": dict(row_mapping),
        "arms": {
            arm: {
                "arm_role": arm_role,
                "downstream_row_label": row_label,
                "geometry_partner": row_key,
                "enters_primary_decision": arm == PRIMARY_ARM,
            }
            for arm, row_key, row_label, arm_role in ARMS
        },
        "sorted_key_populations": build_population_provenance(rows, pair_rows),
        "strata": build_strata(rows),
        "per_image": build_per_image(rows, benign),
        "associations": build_associations(rows),
        "decision": build_decision(rows, pair_rows),
        "greedy_displacement_pairs": {
            "pair_count": len(pair_rows),
            "rows": [
                {
                    "image_id": pair["image_id"],
                    "target_gt_owner_id": pair["target_gt_owner_id"],
                    "displacer_gt_owner_id": pair["displacer_gt_owner_id"],
                    "iou": pair["geometry"]["iou"],
                    "center_distance_normalized": pair["geometry"]["center_distance_normalized"],
                    "geometric_label": pair["geometry"]["geometric_label"],
                    "sorted_key_rank_gap": pair["sorted_key_ranks"]["sorted_key_rank_gap"],
                    "displacer_native_true_positive": pair["displacer_native_disposition"][
                        "native_true_positive"
                    ],
                    "description_equal": pair["description_equal"],
                }
                for pair in pair_rows
            ],
        },
        "reserved_next_panel": dict(RESERVED_NEXT_PANEL),
        "operational_definitions": dict(OPERATIONAL_DEFINITIONS),
        "cross_image_pooling": NO_CROSS_IMAGE_POOLING,
        "claim_boundary": CLAIM_BOUNDARY,
        "not_claimed": list(NOT_CLAIMED),
        "binding": {
            "plan_manifest_content_sha256": inputs.plan_manifest_content_sha256,
            "primary_analysis_receipt_content_sha256": inputs.primary_receipt_content_sha256,
            "secondary_analysis_receipt_content_sha256": inputs.secondary_receipt_content_sha256,
            "secondary_merge_receipt_content_sha256": inputs.merge_receipt_content_sha256,
            "census_plan_receipt_content_sha256": inputs.census_receipt_content_sha256,
            "census_run_root": str(inputs.census_run_root),
        },
    }
    assert_emitted_payload(summary, label="geometry summary")
    assert_no_cross_image_raw_delta_pooling(summary, label="geometry summary")
    return summary


def _fmt(value: Any, *, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(summary: Mapping[str, Any]) -> str:
    denominators = summary["denominators"]
    decision = summary["decision"]
    lines: list[str] = [
        "# Sorted crossing owner-row geometric relation stratification",
        "",
        f"Unit: `{summary['unit_id']}`",
        "",
        f"- source unit: `{summary['source_unit_id']}`",
        f"- canonical owner ledger: `{summary['census_unit_id']}` "
        f"(`{summary['binding']['census_plan_receipt_content_sha256']}`)",
        f"- crossing plan manifest: `{summary['binding']['plan_manifest_content_sha256']}`",
        f"- secondary merge receipt: "
        f"`{summary['binding']['secondary_merge_receipt_content_sha256']}`",
        f"- compute scope: {summary['compute_scope']}",
        "",
        "## Claim boundary",
        "",
        f"{summary['claim_boundary']}.",
        "",
    ]
    lines.extend(f"- {statement}" for statement in summary["not_claimed"])
    lines.extend(
        [
            "",
            "## How to read this",
            "",
            "Every geometry field is computed in sealed norm-1000 bin space; there is no "
            "pixel round trip. The primary `P+C -> E` arm measures C against E, and the "
            "optional `P+E+C -> F` arm measures C against F -- E geometry is never reused for "
            "an F delta, and the analyzer fails closed unless each arm's scored coordinate "
            "tokens equal the sealed tokens of the row it measured. Raw deltas stay indexed by "
            "image.",
            "",
            "## Denominators",
            "",
            f"- crossing owners: {denominators['crossing_owner_count']}",
            f"- primary C/E rows: {denominators['primary_ce_row_count']}",
            f"- optional C/F sensitivity rows: {denominators['sensitivity_cf_row_count']}",
            f"- greedy displacement pairs: {denominators['greedy_displacement_pair_count']}",
            f"- benign-substitution controls: {denominators['benign_control_count']} "
            f"over {denominators['image_count']} images",
            f"- row mapping: {summary['row_mapping']['transformation']}",
            "",
            "## Frozen decision",
            "",
            f"Outcome: **{decision['outcome']}** "
            f"(order: {', '.join(decision['exhaustive_order'])}).",
            "",
            f"Route precedence: {decision['route_precedence']}. "
            f"Image span: {decision['image_span_requirement']}.",
            "",
            "| arm | rows | material-negative | same-desc + overlap | fraction | "
            "identity images | clearly separated | separated images | "
            "separated + different desc | outcome |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for key in ("primary_arm", "sensitivity_arm"):
        block = decision[key]
        lines.append(
            f"| `{block['arm']}` | {block['row_count']} | {block['material_negative_count']} | "
            f"{block['same_description_and_any_overlap_count']} | "
            f"{_fmt(block['same_description_and_any_overlap_fraction'], digits=3)} | "
            f"{len(block['same_description_and_any_overlap_image_ids'])} | "
            f"{block['clearly_separated_count']} | "
            f"{len(block['clearly_separated_image_ids'])} | "
            f"{block['clearly_separated_different_description_count']} | {block['outcome']} |"
        )
    pair_check = decision["greedy_pair_compatibility_check"]
    lines.extend(
        [
            "",
            f"The optional C/F arm is a sensitivity: {decision['sensitivity_arm']['role']}.",
            "",
            f"Greedy displacement pairs ({pair_check['pair_count']}): "
            f"{pair_check['any_overlap_or_center_containment_count']} overlapping or "
            f"centre-contained, {pair_check['clearly_separated_count']} clearly separated -- "
            f"reading `{pair_check['reading']}`. {pair_check['authority']}.",
            "",
            "## Cartesian strata (nonzero cells)",
            "",
        ]
    )
    for arm, block in summary["strata"].items():
        lines.extend(
            [
                f"### `{arm}` ({block['arm_role']}, {block['row_count']} rows)",
                "",
                "| description | E stratum | geometric label | change | count |",
                "| --- | --- | --- | --- | ---: |",
            ]
        )
        nonzero = [cell for cell in block["cells"] if cell["count"]]
        for cell in nonzero:
            lines.append(
                f"| {cell['description']} | {cell['e_stratum']} | {cell['geometric_label']} | "
                f"{cell['change']} | {cell['count']} |"
            )
        if not nonzero:
            lines.append("| _none_ | | | | 0 |")
        lines.append("")
        lines.append(
            f"All {len(block['cells'])} Cartesian cells, including empty ones, are published in "
            f"`{SUMMARY_NAME}`."
        )
        lines.append("")

    lines.extend(["## Rank associations", "", "Descriptive only, never causal.", ""])
    for arm, block in summary["associations"].items():
        lines.append(f"### `{arm}` (n={block['sample_size']}, {block['image_count']} images)")
        lines.append("")
        lines.append("| measure | Spearman rho | leave-one-image-out range of rho |")
        lines.append("| --- | ---: | --- |")
        for measure in ("iou", "center_distance"):
            entry = block[f"{measure}_versus_relative_coordinate_delta"]
            loio = [
                value["spearman_rho"]
                for value in entry["leave_one_image_out"].values()
                if value["spearman_rho"] is not None
            ]
            spread = f"{_fmt(min(loio))} .. {_fmt(max(loio))}" if loio else "n/a"
            lines.append(f"| {measure} | {_fmt(entry['spearman_rho'])} | {spread} |")
        lines.append("")

    lines.extend(["## Per-image values", "", "Raw deltas are never pooled across images.", ""])
    for image_id, block in summary["per_image"].items():
        reference = block["benign_reference"]
        lines.append(
            f"- `{image_id}`: benign coordinate reference "
            f"{_fmt(reference['coordinate_delta'])} nat "
            f"(`{reference['gt_owner_id']}`); "
            + "; ".join(
                f"{arm} n={values['owner_count']}, material-negative "
                f"{values['material_negative_count']}, within-image median "
                f"{_fmt(values['within_image_median_relative_coordinate_delta'])}"
                for arm, values in block["arms"].items()
            )
        )
    lines.extend(["", "## Greedy displacement pairs", ""])
    lines.append(
        "| image | target | displacer | IoU | centre distance | label | rank gap | displacer TP |"
    )
    lines.append("| --- | --- | --- | ---: | ---: | --- | ---: | --- |")
    for pair in summary["greedy_displacement_pairs"]["rows"]:
        lines.append(
            f"| {pair['image_id']} | `{pair['target_gt_owner_id']}` | "
            f"`{pair['displacer_gt_owner_id']}` | {_fmt(pair['iou'])} | "
            f"{_fmt(pair['center_distance_normalized'])} | {pair['geometric_label']} | "
            f"{pair['sorted_key_rank_gap']} | {_fmt(pair['displacer_native_true_positive'])} |"
        )
    reserved = summary["reserved_next_panel"]
    lines.extend(
        [
            "",
            "## Reserved, not counted",
            "",
            f"Image `{reserved['image_id']}` is reserved for the next prospective "
            f"thirteen-image panel and enters no denominator here "
            f"({reserved['physical_owner_annotation_count']} physical-owner annotations: "
            f"{reserved['person_annotation_count']} person, {reserved['tie_annotation_count']} "
            f"tie). {reserved['verification_scope']}.",
            "",
            "## Operational definitions",
            "",
        ]
    )
    for name, text in sorted(summary["operational_definitions"].items()):
        lines.append(f"- **{name}**: {text}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 8. Run and publish
# ---------------------------------------------------------------------------


def run_analysis(
    *,
    plan_dir: Path,
    primary_owner_rows: Path,
    secondary_owner_rows: Path,
    secondary_merged_dir: Path,
    census_plan_dir: Path | None = None,
    media_root: str = DEFAULT_MEDIA_ROOT,
    contract: DenominatorContract = FROZEN_DENOMINATORS,
) -> dict[str, Any]:
    inputs = load_sealed_inputs(
        plan_dir=plan_dir,
        primary_owner_rows=primary_owner_rows,
        secondary_owner_rows=secondary_owner_rows,
        secondary_merged_dir=secondary_merged_dir,
        census_plan_dir=census_plan_dir,
        contract=contract,
    )
    # The 64 -> 38 secondary mapping is a structural contract; it is checked before any
    # geometry is measured, so a broken readout join never reaches an owner row.
    row_mapping = build_row_mapping(inputs, contract=contract)
    rows = build_owner_geometry_rows(inputs, contract=contract)
    pair_rows = build_displacement_pair_rows(inputs, contract=contract)
    assert_emitted_payload(rows, label="owner geometry rows")
    assert_emitted_payload(pair_rows, label="displacement pair rows")
    summary = build_summary(
        inputs, rows, pair_rows, contract=contract, row_mapping=row_mapping
    )
    visual_plan = build_visual_plan(inputs, rows, pair_rows, media_root=media_root)
    assert_emitted_payload(visual_plan, label="visual plan")
    return {
        "inputs": inputs,
        "owner_rows": rows,
        "pair_rows": pair_rows,
        "summary": summary,
        "visual_plan": visual_plan,
    }


def build_output_files(result: Mapping[str, Any]) -> dict[str, bytes]:
    """The deterministic, self-sealed analysis byte content."""

    inputs: SealedInputs = result["inputs"]
    summary = result["summary"]
    visual_plan = result["visual_plan"]

    owner_rows_bytes = b"".join(
        canonical_json_bytes(row) + b"\n" for row in result["owner_rows"]
    )
    pair_rows_bytes = b"".join(canonical_json_bytes(row) + b"\n" for row in result["pair_rows"])
    summary_bytes = (
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    report_bytes = render_markdown(summary).encode("utf-8")
    visual_plan_bytes = (
        json.dumps(visual_plan, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")

    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "source_unit_id": SOURCE_UNIT_ID,
        "census_unit_id": CENSUS_UNIT_ID,
        "analyzer_source_sha256": sha256_file(Path(__file__).resolve()),
        "input_file_sha256": dict(inputs.input_file_sha256),
        "upstream_source_sha256": dict(inputs.source_sha256),
        "binding": dict(summary["binding"]),
        "artifact_determinism": "no_wall_clock_or_host_specific_field_is_sealed",
        "denominators": dict(summary["denominators"]),
        "row_mapping": dict(summary["row_mapping"]),
        "decision_outcome": summary["decision"]["outcome"],
        "policy": {
            "inputs": (
                "the sealed crossing plan, primary and secondary analyses, the secondary merge "
                "and the canonical owner ledger reached through the plan's sealed lineage, and "
                "nothing else"
            ),
            "compute_scope": summary["compute_scope"],
            "geometry_space": "sealed_norm1000_coordinate_tokens_no_pixel_round_trip",
            "arm_geometry_binding": (
                "each arm's geometry partner is verified against that arm's exact scored "
                "coordinate token span"
            ),
            "primary_decision_scope": "primary C/E arm only; the C/F arm is a sensitivity",
            "cutoffs": dict(summary["decision"]["cutoffs"]),
            "cross_image_pooling": NO_CROSS_IMAGE_POOLING,
            "claim_boundary": CLAIM_BOUNDARY,
            "not_claimed": list(NOT_CLAIMED),
            "reserved_next_panel": dict(RESERVED_NEXT_PANEL),
        },
        "output_file_digests": {
            OWNER_ROWS_NAME: {
                "path": OWNER_ROWS_NAME,
                "byte_size": len(owner_rows_bytes),
                "row_count": len(result["owner_rows"]),
                "sha256": sha256_bytes(owner_rows_bytes),
            },
            PAIR_ROWS_NAME: {
                "path": PAIR_ROWS_NAME,
                "byte_size": len(pair_rows_bytes),
                "row_count": len(result["pair_rows"]),
                "sha256": sha256_bytes(pair_rows_bytes),
            },
            SUMMARY_NAME: {
                "path": SUMMARY_NAME,
                "byte_size": len(summary_bytes),
                "sha256": sha256_bytes(summary_bytes),
            },
            REPORT_MD_NAME: {
                "path": REPORT_MD_NAME,
                "byte_size": len(report_bytes),
                "sha256": sha256_bytes(report_bytes),
            },
            VISUAL_PLAN_NAME: {
                "path": VISUAL_PLAN_NAME,
                "byte_size": len(visual_plan_bytes),
                "sha256": sha256_bytes(visual_plan_bytes),
            },
        },
    }
    assert_emitted_payload(receipt, label="geometry analysis receipt")
    assert_no_cross_image_raw_delta_pooling(receipt, label="geometry analysis receipt")
    receipt["receipt_content_sha256"] = sha256_json(receipt)

    return {
        OWNER_ROWS_NAME: owner_rows_bytes,
        PAIR_ROWS_NAME: pair_rows_bytes,
        SUMMARY_NAME: summary_bytes,
        REPORT_MD_NAME: report_bytes,
        VISUAL_PLAN_NAME: visual_plan_bytes,
        RECEIPT_NAME: canonical_json_bytes(receipt) + b"\n",
    }


def publish_analysis(output_dir: Path, files: Mapping[str, bytes]) -> dict[str, Any]:
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
            f"refusing to publish into existing analysis directory {output_dir}: it is not a "
            f"byte-identical rerun (missing={missing!r}, differing={differing!r}, "
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-dir",
        type=Path,
        required=True,
        help="Sealed crossing plan directory holding manifest.json and cohort-registry.jsonl",
    )
    parser.add_argument(
        "--primary-owner-rows",
        type=Path,
        required=True,
        help="Sealed primary analysis owner-rows.jsonl (its receipt.json sits beside it)",
    )
    parser.add_argument(
        "--secondary-owner-rows",
        type=Path,
        required=True,
        help="Sealed secondary analysis secondary-owner-rows.jsonl",
    )
    parser.add_argument(
        "--secondary-merged-dir",
        type=Path,
        required=True,
        help="Sealed secondary merge directory, read for the exact scored token spans",
    )
    parser.add_argument(
        "--census-plan-dir",
        type=Path,
        default=None,
        help=(
            "Canonical owner ledger plan directory; defaults to the census run root the "
            "crossing plan lineage seals"
        ),
    )
    parser.add_argument(
        "--media-root",
        type=str,
        default=DEFAULT_MEDIA_ROOT,
        help="Recorded verbatim in the visualization plan; no media is opened here",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_analysis(
            plan_dir=args.plan_dir,
            primary_owner_rows=args.primary_owner_rows,
            secondary_owner_rows=args.secondary_owner_rows,
            secondary_merged_dir=args.secondary_merged_dir,
            census_plan_dir=args.census_plan_dir,
            media_root=args.media_root,
        )
        files = build_output_files(result)
        published = publish_analysis(Path(args.output_dir), files)
    except GeometryContractError as exc:
        raise SystemExit(f"geometry analysis contract violated: {exc}") from exc
    summary = result["summary"]
    print(
        json.dumps(
            {
                "analysis": published,
                "denominators": summary["denominators"],
                "decision_outcome": summary["decision"]["outcome"],
                "material_negative_primary_count": summary["decision"]["primary_arm"][
                    "material_negative_count"
                ],
                "material_negative_sensitivity_count": summary["decision"]["sensitivity_arm"][
                    "material_negative_count"
                ],
                "greedy_pair_reading": summary["decision"]["greedy_pair_compatibility_check"][
                    "reading"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
