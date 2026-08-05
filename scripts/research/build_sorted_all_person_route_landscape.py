#!/usr/bin/env python3
"""Build the CPU-only preflight plan for the all-person route landscape.

The unit deliberately separates the fixed candidate bank and exact-token
prefixes from later model work.  This script therefore never imports a model,
tokenizer, or torch.  It verifies the frozen source bytes, builds the shared
369-row bank once, joins every box to all 41 owners with the Task-0 strict
matcher's one-prediction semantics, and writes an immutable scoring plan.

The output directory is create-or-identical: validation finishes before any
output is written, then the complete directory is atomically published.  An
existing output may only be reused when every emitted byte is identical.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any


UNIT_ID = "2026-08-03-sorted-all-person-owner-relative-route-landscape"
SCHEMA_VERSION = "sorted-all-person-route-landscape-plan.v1"

UNIT_SHA256 = "6fb2af7019d776572cf4fcaa7666a31eb38fe42818422eeb9a57a1d7b0cf9109"
PANEL_SHA256 = "cfe4f693133287f9e6c561fc094710c642f049aec3d50f44dea98b764ba2aa85"
OWNER_LEDGER_SHA256 = "80357539069ac04c12522a211bb70a0bc6841b54a32627e993aeaa75574bd8a0"
DONOR_SHA256 = "9f123c036470a3f9545e917f78485ef60620908bbee240350c7f0d2b48d66037"
BEHAVIOR_SHA256 = "9c2e3c659f85a6add6c251578a494b1086a308a1c7ef65df5ac2a2eee7b3b26d"

ROOT = Path("/data/CoordExp")
OUTPUTS = ROOT / "outputs/research/qwen3-vl-dense-enumeration"
TASK0_ROOT = (
    OUTPUTS / "2026-08-01-sorted-owner-basin-task0-v2-global-ambiguity-neutral-final"
)
UNIT_PATH = (
    ROOT
    / ".worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration"
    / "experiments/2026-08-03-sorted-all-person-owner-relative-route-landscape/unit.md"
)
PANEL_PATH = (
    OUTPUTS
    / "2026-07-21-best-sampled-trajectory-positive-row-imitation-screen"
    / "evaluation-inputs/human-refined-12.coord.jsonl"
)
OWNER_LEDGER_PATH = TASK0_ROOT / "owner-ledger.jsonl"
PREDICTION_LEDGER_PATH = TASK0_ROOT / "prediction-row-ledger.jsonl"
MATCHER_PATH = TASK0_ROOT / "matcher-contract.json"
DONOR_PATH = (
    OUTPUTS
    / "2026-07-29-three-checkpoint-human-refined12-max3084/sorted/sampled/shard-1.json"
)
BEHAVIOR_PATH = (
    OUTPUTS
    / "2026-08-02-sorted-fn-mechanism-decomposition-smoke-v1"
    / "behavior-due-turn-gt7511-17-greedy-rp1p00-rp1p10-corrected-v2.json"
)

IMAGE_ID = "7511"
IMAGE_WIDTH = 1152
IMAGE_HEIGHT = 864
DESCRIPTION = "person"
PROMPT_SHA256 = "3fba18e878f1f98599328b0f65b1960b96e1ed64d33085cbb9f5859c23153fa7"
GENERATED_SHA256 = "41a4a8c0c7cda3837d5659954d372d930668d53fbc80d2413de4fb1d81b97b79"
EXECUTED_MEDIA_SHA256 = (
    "5df850d391307585569f39942c732e8769a8d18f3271c3ba8798bbfa509f8742"
)
IMAGE_SHA256 = "2843c07959515a93d2791183c998462d76b34100185e57a258d8949f112296e7"
TOKENIZER_IDENTITY_SHA256 = (
    "8edf6678e814c2d7417010effa42d99915f377b9e1d312a600aa9fa502241b27"
)

OBJECT_REF_START = 151646
OBJECT_REF_END = 151647
BOX_START = 151648
BOX_END = 151649
COORD_TOKEN_START = 151670
COORD_TOKEN_END = 152669
IOU_THRESHOLD = 0.5
MATCHER_EPSILON = 1e-12

EXPECTED_OWNER_IDS = tuple(f"gt:7511:{index}" for index in range(2, 43))
RETAINED_OWNER_IDS = frozenset({"gt:7511:2", "gt:7511:17", "gt:7511:22", "gt:7511:32"})

TRANSLATION_TRANSFORMS = (
    "translate_left",
    "translate_right",
    "translate_up",
    "translate_down",
    "translate_up_left",
    "translate_up_right",
    "translate_down_left",
    "translate_down_right",
)
EXTENT_TRANSFORMS = (
    "width_expand",
    "width_shrink",
    "height_expand",
    "height_shrink",
    "isotropic_expand",
    "isotropic_shrink",
    "width_expand_height_shrink",
    "width_shrink_height_expand",
)
CONTEXT_SPECS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("root", ()),
    ("self-due-gt2", (0,)),
    ("self-due-gt17", (0, 1, 2)),
    ("self-due-gt22", (0, 1, 2, 3)),
    ("self-due-gt32", (0, 1, 2, 3, 4, 5, 6)),
    ("skip-post-gt17", (0, 1, 2, 4)),
)
EXPECTED_DONOR_OWNER_IDS = {
    1: "gt:7511:2",
    3: "gt:7511:17",
    4: "gt:7511:22",
    7: "gt:7511:32",
}


class PlanContractError(ValueError):
    """Raised when a frozen input or plan invariant is not satisfied."""


@dataclass(frozen=True)
class SourcePaths:
    """All immutable inputs used by the CPU planner."""

    unit: Path = UNIT_PATH
    panel: Path = PANEL_PATH
    owner_ledger: Path = OWNER_LEDGER_PATH
    prediction_ledger: Path = PREDICTION_LEDGER_PATH
    matcher: Path = MATCHER_PATH
    donor: Path = DONOR_PATH
    behavior: Path = BEHAVIOR_PATH


DEFAULT_SOURCES = SourcePaths()


def canonical_json_bytes(value: Any) -> bytes:
    """The one JSON encoding used by all plan identities."""

    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PlanContractError(f"{label} is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PlanContractError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(raw, Mapping):
        raise PlanContractError(f"{label} must be a JSON object")
    return raw


def _read_jsonl(path: Path, label: str) -> list[Mapping[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise PlanContractError(f"{label} is missing: {path}") from exc
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise PlanContractError(f"{label} line {line_number} is blank")
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PlanContractError(f"{label} line {line_number} is not JSON") from exc
        if not isinstance(raw, Mapping):
            raise PlanContractError(f"{label} line {line_number} must be an object")
        rows.append(raw)
    if not rows:
        raise PlanContractError(f"{label} must contain at least one row")
    return rows


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise PlanContractError(f"{label} must be a non-empty trimmed string")
    return value


def _int_list(value: Any, label: str) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise PlanContractError(f"{label} must be an integer list")
    result: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise PlanContractError(f"{label}[{index}] must be a non-negative integer")
        result.append(item)
    return result


def _pixel_box(value: Any, label: str) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 4
    ):
        raise PlanContractError(f"{label} must be a four-value pixel xyxy box")
    result: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise PlanContractError(f"{label}[{index}] must be numeric")
        numeric = float(item)
        if not numeric.is_integer():
            raise PlanContractError(f"{label}[{index}] must be an integer pixel value")
        result.append(int(numeric))
    x1, y1, x2, y2 = result
    if not 0 <= x1 < x2 <= IMAGE_WIDTH or not 0 <= y1 < y2 <= IMAGE_HEIGHT:
        raise PlanContractError(f"{label} is outside the frozen image canvas")
    return x1, y1, x2, y2


def _assert_digest(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise PlanContractError(f"{label} SHA-256 does not match the frozen unit")
    return actual


def _source_digests(sources: SourcePaths) -> dict[str, str]:
    """Fail closed before reading mutable source semantics."""

    expected = {
        "unit": (sources.unit, UNIT_SHA256),
        "panel": (sources.panel, PANEL_SHA256),
        "owner_ledger": (sources.owner_ledger, OWNER_LEDGER_SHA256),
        "donor": (sources.donor, DONOR_SHA256),
        "behavior": (sources.behavior, BEHAVIOR_SHA256),
    }
    digests = {
        name: _assert_digest(path, digest, name)
        for name, (path, digest) in expected.items()
    }
    for name, path in {
        "prediction_ledger": sources.prediction_ledger,
        "matcher": sources.matcher,
    }.items():
        if not path.is_file():
            raise PlanContractError(f"{name} is missing: {path}")
        digests[name] = sha256_file(path)
    return digests


def _load_owners(path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path, "owner ledger")
    selected = [
        row
        for row in rows
        if str(row.get("image_id")) == IMAGE_ID
        and row.get("normalized_description") == DESCRIPTION
    ]
    selected.sort(key=lambda row: int(row.get("original_annotation_index", -1)))
    owner_ids = tuple(str(row.get("gt_owner_id")) for row in selected)
    if owner_ids != EXPECTED_OWNER_IDS:
        raise PlanContractError(
            "owner ledger does not contain exactly gt:7511:2 through gt:7511:42"
        )
    projection: list[dict[str, Any]] = []
    for expected_index, row in zip(range(2, 43), selected, strict=True):
        if row.get("schema_version") != "sorted-owner-basin-owner-ledger.v2":
            raise PlanContractError(
                "owner ledger must use the canonical Task-0 v2 schema"
            )
        if int(row.get("original_annotation_index", -1)) != expected_index:
            raise PlanContractError(
                "owner ledger ID and original annotation index disagree"
            )
        if row.get("official_coco_category_id") != 1:
            raise PlanContractError(
                "owner ledger person category is not official COCO category 1"
            )
        if row.get("ambiguity_receipt_ids") != []:
            raise PlanContractError(
                "a primary owner is ambiguity-neutral and cannot enter this unit"
            )
        box = _pixel_box(row.get("bbox_xyxy"), f"owner {row['gt_owner_id']} bbox_xyxy")
        projection.append(
            {
                "schema_version": SCHEMA_VERSION,
                "row_kind": "confirmed_person_owner",
                "gt_owner_id": row["gt_owner_id"],
                "original_annotation_index": expected_index,
                "image_id": IMAGE_ID,
                "normalized_description": DESCRIPTION,
                "official_coco_category_id": 1,
                "bbox_pixel_xyxy": list(box),
                "source_owner_ledger_schema_version": row["schema_version"],
                "source_owner_ledger_execution_receipt_content_sha256": row.get(
                    "execution_receipt_content_sha256"
                ),
            }
        )
    return projection


def _load_matcher(path: Path) -> Mapping[str, Any]:
    matcher = _read_json(path, "matcher contract")
    if matcher.get("schema_version") != "sorted-owner-basin-matcher.v2":
        raise PlanContractError(
            "matcher contract must be sorted-owner-basin-matcher.v2"
        )
    if matcher.get("iou_threshold") != IOU_THRESHOLD:
        raise PlanContractError("matcher contract IoU threshold is not the frozen 0.5")
    namespace = matcher.get("category_namespace")
    if not isinstance(namespace, Mapping) or namespace.get("strict_join") != (
        "normalized_description_or_predeclared_symmetric_alias_only"
    ):
        raise PlanContractError("matcher strict semantic join differs from Task-0")
    aliases = matcher.get("alias_table")
    if not isinstance(aliases, Mapping) or aliases.get("aliases") != []:
        raise PlanContractError(
            "this all-person unit requires the frozen empty alias table"
        )
    return matcher


def _load_prediction_rows(path: Path) -> dict[int, Mapping[str, Any]]:
    rows = _read_jsonl(path, "prediction row ledger")
    selected: dict[int, Mapping[str, Any]] = {}
    prefix = "pred:sorted:sampled:21010:7511:"
    for row in rows:
        row_id = row.get("pred_row_id")
        if not isinstance(row_id, str) or not row_id.startswith(prefix):
            continue
        try:
            index = int(row_id.rsplit(":", 1)[1])
        except ValueError as exc:
            raise PlanContractError(
                "sampled donor prediction ID has an invalid row suffix"
            ) from exc
        if index in selected:
            raise PlanContractError(
                "sampled donor prediction ledger has duplicate row IDs"
            )
        selected[index] = row
    if not all(index in selected for index in EXPECTED_DONOR_OWNER_IDS):
        raise PlanContractError(
            "prediction ledger lacks a required seed-21010 donor row"
        )
    return selected


def _round_to_bin(pixel: int, extent: int) -> int:
    """Production pixel-to-coordinate-token conversion, including V1 clamp."""

    result = int(round(pixel * 1000.0 / float(extent)))
    return max(0, min(999, result))


def pixel_to_bins(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    bins = (
        _round_to_bin(x1, IMAGE_WIDTH),
        _round_to_bin(y1, IMAGE_HEIGHT),
        _round_to_bin(x2, IMAGE_WIDTH),
        _round_to_bin(y2, IMAGE_HEIGHT),
    )
    if bins[0] >= bins[2] or bins[1] >= bins[3]:
        raise PlanContractError(
            "a candidate collapses after production pixel-to-token conversion"
        )
    return bins


def bins_to_pixel(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        int(round(x1 * IMAGE_WIDTH / 1000.0)),
        int(round(y1 * IMAGE_HEIGHT / 1000.0)),
        int(round(x2 * IMAGE_WIDTH / 1000.0)),
        int(round(y2 * IMAGE_HEIGHT / 1000.0)),
    )


def _coord_token_ids(bins: tuple[int, int, int, int]) -> list[int]:
    return [COORD_TOKEN_START + value for value in bins]


def _iou(left: Sequence[int], right: Sequence[int]) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(
        0.0, float(left[3]) - float(left[1])
    )
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(
        0.0, float(right[3]) - float(right[1])
    )
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def strict_assignment(
    pixel_box: tuple[int, int, int, int], owners: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Apply the canonical matcher to one prediction against all 41 owners.

    With one candidate, Task-0's maximum-cardinality/maximum-total-IoU face
    has a strict owner only when its best eligible edge is unique.  Equal
    best edges are ambiguity-neutral; lower-IoU eligible owners do not enter
    the ambiguity face and are retained as diagnostic membership only.
    """

    eligible: list[tuple[str, float]] = []
    all_receipts: list[dict[str, Any]] = []
    for owner in owners:
        owner_id = _string(owner.get("gt_owner_id"), "projected owner ID")
        owner_box = _pixel_box(owner.get("bbox_pixel_xyxy"), f"owner {owner_id} box")
        overlap = _iou(pixel_box, owner_box)
        all_receipts.append(
            {"gt_owner_id": owner_id, "intersection_over_union": overlap}
        )
        if overlap + MATCHER_EPSILON >= IOU_THRESHOLD:
            eligible.append((owner_id, overlap))
    all_receipts.sort(
        key=lambda item: (-item["intersection_over_union"], item["gt_owner_id"])
    )
    if not eligible:
        return {
            "strict_assignment_status": "unmatched",
            "strict_assignment_gt_owner_id": None,
            "ambiguity_owner_ids": [],
            "lower_bound_owner_ids": [],
            "upper_bound_owner_ids": [],
            "eligible_owner_iou_receipts": all_receipts,
        }
    maximum = max(value for _, value in eligible)
    optimal = sorted(
        owner_id
        for owner_id, value in eligible
        if abs(value - maximum) <= MATCHER_EPSILON
    )
    if len(optimal) != 1:
        return {
            "strict_assignment_status": "ambiguous_neutral",
            "strict_assignment_gt_owner_id": None,
            "ambiguity_owner_ids": optimal,
            "lower_bound_owner_ids": [],
            "upper_bound_owner_ids": optimal,
            "eligible_owner_iou_receipts": all_receipts,
        }
    return {
        "strict_assignment_status": "matched",
        "strict_assignment_gt_owner_id": optimal[0],
        "ambiguity_owner_ids": [],
        "lower_bound_owner_ids": optimal,
        "upper_bound_owner_ids": optimal,
        "eligible_owner_iou_receipts": all_receipts,
    }


def _translate(
    box: tuple[int, int, int, int], dx: int, dy: int, transform: str
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    x_offset = 0
    y_offset = 0
    if "left" in transform:
        x_offset = -dx
    if "right" in transform:
        x_offset = dx
    if transform.startswith("translate_up"):
        y_offset = -dy
    if transform.startswith("translate_down"):
        y_offset = dy
    new_x1 = min(max(0, x1 + x_offset), IMAGE_WIDTH - width)
    new_y1 = min(max(0, y1 + y_offset), IMAGE_HEIGHT - height)
    return new_x1, new_y1, new_x1 + width, new_y1 + height


def _extent(
    box: tuple[int, int, int, int], dx: int, dy: int, transform: str
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x_low = x1
    x_high = x2
    y_low = y1
    y_high = y2
    if transform in {"width_expand", "isotropic_expand", "width_expand_height_shrink"}:
        x_low -= dx
        x_high += dx
    if transform in {"width_shrink", "isotropic_shrink", "width_shrink_height_expand"}:
        x_low += dx
        x_high -= dx
    if transform in {"height_expand", "isotropic_expand", "width_shrink_height_expand"}:
        y_low -= dy
        y_high += dy
    if transform in {"height_shrink", "isotropic_shrink", "width_expand_height_shrink"}:
        y_low += dy
        y_high -= dy
    return (
        max(0, min(IMAGE_WIDTH, x_low)),
        max(0, min(IMAGE_HEIGHT, y_low)),
        max(0, min(IMAGE_WIDTH, x_high)),
        max(0, min(IMAGE_HEIGHT, y_high)),
    )


def _valid_pixel_box(box: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = box
    return 0 <= x1 < x2 <= IMAGE_WIDTH and 0 <= y1 < y2 <= IMAGE_HEIGHT


def _candidate_row(
    *,
    owner: Mapping[str, Any],
    source_box: tuple[int, int, int, int],
    transform: str,
    family: str,
    selection_ordinal: int,
    dx: int,
    dy: int,
    owners: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    bins = pixel_to_bins(source_box)
    coord_token_ids = _coord_token_ids(bins)
    decoded = bins_to_pixel(bins)
    assignment = strict_assignment(decoded, owners)
    owner_id = _string(owner.get("gt_owner_id"), "candidate owner")
    return {
        "schema_version": SCHEMA_VERSION,
        "row_kind": "primary_candidate",
        "candidate_id": f"primary:{owner_id}:{selection_ordinal:02d}:{transform}",
        "source": {
            "kind": "canonical_owner_ledger_gt_anchor",
            "gt_owner_id": owner_id,
        },
        "generator": {
            "gt_owner_id": owner_id,
            "policy": "fixed_size_aware_transform_order_without_scores",
        },
        "generator_gt_owner_id": owner_id,
        "candidate_family": family,
        "transform": transform,
        "selection_ordinal": selection_ordinal,
        "size_aware_offsets_pixels": {"dx": dx, "dy": dy},
        "source_bbox_pixel_xyxy": list(source_box),
        "coord_bins": list(bins),
        "coord_token_ids": coord_token_ids,
        "coord_token_ids_sha256": sha256_json(coord_token_ids),
        "decoded_bbox_pixel_xyxy": list(decoded),
        **assignment,
    }


def build_primary_candidates(
    owners: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Materialize exact plus four fixed-order translations and extents."""

    candidates: list[dict[str, Any]] = []
    for owner in owners:
        owner_box = _pixel_box(
            owner.get("bbox_pixel_xyxy"), f"owner {owner['gt_owner_id']} box"
        )
        x1, y1, x2, y2 = owner_box
        dx = max(1, int(round((x2 - x1) / 4.0)))
        dy = max(1, int(round((y2 - y1) / 4.0)))
        selected_tokens: set[tuple[int, int, int, int]] = set()
        exact = _candidate_row(
            owner=owner,
            source_box=owner_box,
            transform="exact_gt_anchor",
            family="exact",
            selection_ordinal=0,
            dx=dx,
            dy=dy,
            owners=owners,
        )
        selected_tokens.add(tuple(exact["coord_token_ids"]))
        candidates.append(exact)

        def select_family(
            transforms: Iterable[str], family: str, count: int, ordinal_start: int
        ) -> list[dict[str, Any]]:
            selected: list[dict[str, Any]] = []
            for transform in transforms:
                proposed = (
                    _translate(owner_box, dx, dy, transform)
                    if family == "translation"
                    else _extent(owner_box, dx, dy, transform)
                )
                if not _valid_pixel_box(proposed):
                    continue
                row = _candidate_row(
                    owner=owner,
                    source_box=proposed,
                    transform=transform,
                    family=family,
                    selection_ordinal=ordinal_start + len(selected),
                    dx=dx,
                    dy=dy,
                    owners=owners,
                )
                token_key = tuple(row["coord_token_ids"])
                if token_key in selected_tokens:
                    continue
                selected_tokens.add(token_key)
                selected.append(row)
                if len(selected) == count:
                    return selected
            raise PlanContractError(
                f"owner {owner['gt_owner_id']} cannot supply {count} valid {family} candidates"
            )

        candidates.extend(select_family(TRANSLATION_TRANSFORMS, "translation", 4, 1))
        candidates.extend(select_family(EXTENT_TRANSFORMS, "extent", 4, 5))
    if len(candidates) != 369:
        raise PlanContractError("primary candidate universe is not exactly 369 rows")
    if len({row["candidate_id"] for row in candidates}) != len(candidates):
        raise PlanContractError("primary candidate IDs are not unique")
    return candidates


def enforce_retention(candidates: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Apply the frozen five-of-nine unique-assignment preflight gate."""

    retained: dict[str, int] = {}
    for owner_id in sorted(
        RETAINED_OWNER_IDS, key=lambda item: int(item.rsplit(":", 1)[1])
    ):
        owner_rows = [
            row for row in candidates if row.get("generator_gt_owner_id") == owner_id
        ]
        if len(owner_rows) != 9:
            raise PlanContractError(
                f"retention owner {owner_id} does not have nine candidates"
            )
        count = sum(
            row.get("strict_assignment_status") == "matched"
            and row.get("strict_assignment_gt_owner_id") == owner_id
            for row in owner_rows
        )
        if count < 5:
            raise PlanContractError(
                f"retention preflight failed: {owner_id} has only {count}/9 uniquely assigned candidates"
            )
        retained[owner_id] = count
    return retained


def _split_donor_rows(tokens: Sequence[int]) -> list[list[int]]:
    starts = [index for index, token in enumerate(tokens) if token == OBJECT_REF_START]
    if not starts or starts[0] != 0:
        raise PlanContractError(
            "donor generated tokens do not begin with an object row"
        )
    ends = [*starts[1:], len(tokens)]
    rows = [list(tokens[start:end]) for start, end in zip(starts, ends, strict=True)]
    for index, row in enumerate(rows):
        if len(row) < 8 or row[0] != OBJECT_REF_START:
            raise PlanContractError(f"donor row {index} has an invalid object start")
        try:
            object_end = row.index(OBJECT_REF_END)
        except ValueError as exc:
            raise PlanContractError(f"donor row {index} lacks object_ref_end") from exc
        if object_end <= 1 or row[object_end + 1 : object_end + 2] != [BOX_START]:
            raise PlanContractError(
                f"donor row {index} has an invalid object/box boundary"
            )
        coordinate = row[object_end + 2 : object_end + 6]
        if len(coordinate) != 4 or any(
            token < COORD_TOKEN_START or token > COORD_TOKEN_END for token in coordinate
        ):
            raise PlanContractError(f"donor row {index} lacks four coordinate tokens")
        if row[object_end + 6 :] != [BOX_END]:
            raise PlanContractError(
                f"donor row {index} violates the closed one-box grammar"
            )
    return rows


def _row_coordinates(row: Sequence[int]) -> list[int]:
    object_end = row.index(OBJECT_REF_END)
    return list(row[object_end + 2 : object_end + 6])


def _load_donor(
    path: Path, prediction_rows: Mapping[int, Mapping[str, Any]]
) -> tuple[dict[str, Any], list[list[int]], list[Mapping[str, Any]]]:
    donor = _read_json(path, "seed-21010 donor rollout")
    rollouts = donor.get("rollouts")
    if not isinstance(rollouts, Sequence) or isinstance(rollouts, (str, bytes)):
        raise PlanContractError("donor has no rollout list")
    matching = [
        row
        for row in rollouts
        if isinstance(row, Mapping)
        and row.get("seed") == 21010
        and str(row.get("image_id")) == IMAGE_ID
    ]
    if len(matching) != 1:
        raise PlanContractError(
            "donor must contain exactly one seed-21010 image-7511 rollout"
        )
    rollout = matching[0]
    prompt = _int_list(rollout.get("prompt_token_ids"), "donor prompt_token_ids")
    generated = _int_list(
        rollout.get("generated_token_ids"), "donor generated_token_ids"
    )
    if (
        sha256_json(prompt) != PROMPT_SHA256
        or rollout.get("prompt_token_ids_sha256") != PROMPT_SHA256
    ):
        raise PlanContractError(
            "donor prompt tokens do not match the frozen prompt identity"
        )
    if (
        sha256_json(generated) != GENERATED_SHA256
        or rollout.get("generated_token_ids_sha256") != GENERATED_SHA256
    ):
        raise PlanContractError(
            "donor generated tokens do not match the frozen donor identity"
        )
    if rollout.get("executed_media_sha256") != EXECUTED_MEDIA_SHA256:
        raise PlanContractError(
            "donor executed media digest does not match the frozen identity"
        )
    metadata = donor.get("prompt_metadata")
    if not isinstance(metadata, Mapping):
        raise PlanContractError("donor lacks prompt metadata")
    image_metadata = metadata.get("coco2017_val_000000007511")
    if not isinstance(image_metadata, Mapping):
        raise PlanContractError("donor lacks image-7511 prompt metadata")
    if (
        image_metadata.get("width") != IMAGE_WIDTH
        or image_metadata.get("height") != IMAGE_HEIGHT
        or image_metadata.get("prompt_token_ids_sha256") != PROMPT_SHA256
        or image_metadata.get("image_sha256") != IMAGE_SHA256
    ):
        raise PlanContractError(
            "donor image/prompt dimensions or digest differs from the unit"
        )
    rows = _split_donor_rows(generated)
    predictions_object = rollout.get("predictions")
    if not isinstance(predictions_object, Mapping):
        raise PlanContractError("donor lacks parsed predictions")
    predictions = predictions_object.get("predictions")
    if not isinstance(predictions, Sequence) or isinstance(predictions, (str, bytes)):
        raise PlanContractError("donor parsed predictions is invalid")
    parsed = [item for item in predictions if isinstance(item, Mapping)]
    if len(parsed) != len(rows):
        raise PlanContractError(
            "donor parser row count does not match structural token rows"
        )
    for index, expected_owner in EXPECTED_DONOR_OWNER_IDS.items():
        source = prediction_rows[index]
        if (
            source.get("strict_match_status") != "matched"
            or source.get("strict_match_gt_owner_id") != expected_owner
        ):
            raise PlanContractError(
                f"donor row {index} does not have its frozen strict owner"
            )
        coord = _row_coordinates(rows[index])
        parsed_bins = _int_list(
            parsed[index].get("coord_bins"), f"donor parsed row {index} coord_bins"
        )
        if coord != [COORD_TOKEN_START + value for value in parsed_bins]:
            raise PlanContractError(
                f"donor row {index} parsed coordinates disagree with token IDs"
            )
        expected_raw = parsed[index].get("raw_span_sha256")
        if source.get("raw_span_sha256") != expected_raw:
            raise PlanContractError(
                f"donor row {index} source raw-span digest disagrees with ledger"
            )
    return dict(rollout), rows, parsed


def build_contexts(
    rollout: Mapping[str, Any],
    donor_rows: Sequence[Sequence[int]],
    parsed_donor_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    prompt = _int_list(rollout.get("prompt_token_ids"), "donor prompt_token_ids")
    contexts: list[dict[str, Any]] = []
    for context_id, row_indices in CONTEXT_SPECS:
        if any(index >= len(donor_rows) for index in row_indices):
            raise PlanContractError(
                f"context {context_id} references an absent donor row"
            )
        generated_prefix = [
            token for index in row_indices for token in donor_rows[index]
        ]
        full_prefix = [*prompt, *generated_prefix]
        contexts.append(
            {
                "schema_version": SCHEMA_VERSION,
                "row_kind": "admitted_context",
                "context_id": context_id,
                "image_id": IMAGE_ID,
                "donor_seed": 21010 if row_indices else None,
                "donor_row_indices": list(row_indices),
                "donor_rows": [
                    {
                        "row_index": index,
                        "generated_token_ids": list(donor_rows[index]),
                        "generated_token_ids_sha256": sha256_json(
                            list(donor_rows[index])
                        ),
                        "coordinate_token_ids": _row_coordinates(donor_rows[index]),
                        "raw_span_sha256": parsed_donor_rows[index].get(
                            "raw_span_sha256"
                        ),
                        "raw_span_text": parsed_donor_rows[index].get("raw_span_text"),
                    }
                    for index in row_indices
                ],
                "prompt_token_ids": prompt,
                "prompt_token_ids_sha256": sha256_json(prompt),
                "generated_prefix_token_ids": generated_prefix,
                "generated_prefix_token_ids_sha256": sha256_json(generated_prefix),
                "full_prefix_token_ids": full_prefix,
                "full_prefix_token_ids_sha256": sha256_json(full_prefix),
                "candidate_universe_join": "primary-candidates.jsonl:candidate_id",
                "teacher_forced_chosen_token_parity": {
                    "status": "required_not_cpu_verifiable",
                    "admission_requirement": "must_pass_before_model_scoring",
                    "claimed_pass": False,
                },
            }
        )
    if len(contexts) != 6 or {row["context_id"] for row in contexts} != {
        item[0] for item in CONTEXT_SPECS
    }:
        raise PlanContractError("context registry is not the six mandatory contexts")
    return contexts


def owner_swap_receipt(contexts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_id = {str(context["context_id"]): context for context in contexts}
    due = by_id["self-due-gt22"]
    skipped = by_id["skip-post-gt17"]
    due_rows = due["donor_rows"]
    skipped_rows = skipped["donor_rows"]
    if len(due_rows) != len(skipped_rows):
        raise PlanContractError("owner swap contexts have unequal donor row counts")
    due_shapes = [len(row["generated_token_ids"]) for row in due_rows]
    skipped_shapes = [len(row["generated_token_ids"]) for row in skipped_rows]
    grammar_shapes = [
        {
            "object_ref_start": row["generated_token_ids"][0] == OBJECT_REF_START,
            "object_ref_end": OBJECT_REF_END in row["generated_token_ids"],
            "box_start": BOX_START in row["generated_token_ids"],
            "box_end": row["generated_token_ids"][-1] == BOX_END,
            "coordinate_token_count": 4,
            "description_token_count": len(row["generated_token_ids"]) - 8,
        }
        for row in due_rows
    ]
    skipped_grammar_shapes = [
        {
            "object_ref_start": row["generated_token_ids"][0] == OBJECT_REF_START,
            "object_ref_end": OBJECT_REF_END in row["generated_token_ids"],
            "box_start": BOX_START in row["generated_token_ids"],
            "box_end": row["generated_token_ids"][-1] == BOX_END,
            "coordinate_token_count": 4,
            "description_token_count": len(row["generated_token_ids"]) - 8,
        }
        for row in skipped_rows
    ]
    due_total = len(due["generated_prefix_token_ids"])
    skipped_total = len(skipped["generated_prefix_token_ids"])
    if (
        due_shapes != skipped_shapes
        or grammar_shapes != skipped_grammar_shapes
        or due_total != skipped_total
    ):
        raise PlanContractError(
            "owner swap contexts fail grammar-token-shape or length parity"
        )
    return {
        "left_context_id": "self-due-gt22",
        "right_context_id": "skip-post-gt17",
        "equal_donor_row_count": True,
        "equal_row_grammar_token_shapes": True,
        "row_generated_token_counts": due_shapes,
        "row_grammar_token_shapes": grammar_shapes,
        "equal_total_generated_token_count": True,
        "total_generated_token_count": due_total,
        "left_prefix_token_ids_sha256": due["full_prefix_token_ids_sha256"],
        "right_prefix_token_ids_sha256": skipped["full_prefix_token_ids_sha256"],
        "prefixes_are_content_distinct": due["full_prefix_token_ids_sha256"]
        != skipped["full_prefix_token_ids_sha256"],
    }


def _sidecar_assignment(
    *,
    sidecar_id: str,
    source_kind: str,
    coord_token_ids: Sequence[int],
    raw_token_ids: Sequence[int],
    source_bbox: tuple[int, int, int, int],
    owners: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    source_ref: Mapping[str, Any],
) -> dict[str, Any]:
    if len(coord_token_ids) != 4 or any(
        token < COORD_TOKEN_START or token > COORD_TOKEN_END
        for token in coord_token_ids
    ):
        raise PlanContractError(
            f"sidecar {sidecar_id} has invalid coordinate token IDs"
        )
    bins = tuple(token - COORD_TOKEN_START for token in coord_token_ids)
    if bins[0] >= bins[2] or bins[1] >= bins[3]:
        raise PlanContractError(f"sidecar {sidecar_id} has a degenerate token box")
    decoded = bins_to_pixel(bins)
    if decoded != source_bbox:
        raise PlanContractError(
            f"sidecar {sidecar_id} source pixel box does not match its tokens"
        )
    assignment = strict_assignment(decoded, owners)
    members = [
        str(candidate["candidate_id"])
        for candidate in candidates
        if candidate.get("coord_token_ids") == list(coord_token_ids)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "row_kind": "realized_box_sidecar",
        "sidecar_id": sidecar_id,
        "source_kind": source_kind,
        "coord_token_ids": list(coord_token_ids),
        "coord_token_ids_sha256": sha256_json(list(coord_token_ids)),
        "raw_generated_token_ids": list(raw_token_ids),
        "raw_generated_token_ids_sha256": sha256_json(list(raw_token_ids)),
        "coord_bins": list(bins),
        "bbox_pixel_xyxy": list(decoded),
        "bank_member_candidate_ids": members,
        "requires_new_score_row": not bool(members),
        "excluded_from_primary_ranks": True,
        **assignment,
        "source": dict(source_ref),
    }


def build_sidecars(
    donor_rows: Sequence[Sequence[int]],
    parsed_donor_rows: Sequence[Mapping[str, Any]],
    behavior: Mapping[str, Any],
    owners: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    behavior_path: Path,
) -> list[dict[str, Any]]:
    sidecars: list[dict[str, Any]] = []
    for row_index, owner_id in EXPECTED_DONOR_OWNER_IDS.items():
        coord = _row_coordinates(donor_rows[row_index])
        source_box = _pixel_box(
            parsed_donor_rows[row_index].get("bbox"), f"donor sidecar row {row_index}"
        )
        row = _sidecar_assignment(
            sidecar_id=f"sidecar:seed-21010:row-{row_index}",
            source_kind="seed_21010_sampled_row",
            coord_token_ids=coord,
            raw_token_ids=donor_rows[row_index],
            source_bbox=source_box,
            owners=owners,
            candidates=candidates,
            source_ref={
                "path": str(DONOR_PATH),
                "donor_seed": 21010,
                "donor_row_index": row_index,
                "expected_strict_owner_id": owner_id,
                "raw_span_sha256": parsed_donor_rows[row_index].get("raw_span_sha256"),
            },
        )
        if (
            row["strict_assignment_status"] != "matched"
            or row["strict_assignment_gt_owner_id"] != owner_id
        ):
            raise PlanContractError(
                f"sidecar seed-21010 row {row_index} loses its frozen strict owner"
            )
        sidecars.append(row)

    views = behavior.get("policy_views")
    if not isinstance(views, Sequence) or isinstance(views, (str, bytes)):
        raise PlanContractError("corrected behavior source has no policy views")
    matching_views = [
        view
        for view in views
        if isinstance(view, Mapping)
        and view.get("policy_view_id") == "rp_1_00"
        and view.get("repetition_penalty") == 1.0
    ]
    if len(matching_views) != 1:
        raise PlanContractError("corrected behavior source has no unique rp1.00 view")
    roles = matching_views[0].get("roles")
    if not isinstance(roles, Sequence) or isinstance(roles, (str, bytes)):
        raise PlanContractError("corrected behavior source has no role list")
    matching_roles = [
        role
        for role in roles
        if isinstance(role, Mapping)
        and role.get("role_id") == "due_turn:gt:7511:17"
        and role.get("gt_owner_id") == "gt:7511:17"
    ]
    if len(matching_roles) != 1:
        raise PlanContractError(
            "corrected behavior source has no unique gt:7511:17 due role"
        )
    arms = matching_roles[0].get("arms")
    if not isinstance(arms, Mapping):
        raise PlanContractError("corrected behavior role has no arm registry")
    arm = arms.get("forced_description_greedy")
    if (
        not isinstance(arm, Mapping)
        or arm.get("producer", {}).get("repetition_penalty") != 1.0
    ):
        raise PlanContractError(
            "corrected greedy sidecar arm is missing or has a foreign repetition penalty"
        )
    intervention = arm.get("intervention")
    if not isinstance(intervention, Mapping) or not isinstance(
        intervention.get("row"), Mapping
    ):
        raise PlanContractError("corrected greedy sidecar intervention row is missing")
    raw = _int_list(
        intervention["row"].get("raw_generated_token_ids"),
        "corrected greedy raw tokens",
    )
    coord = _row_coordinates(raw)
    parsed = intervention["row"].get("parsed_predictions")
    if (
        not isinstance(parsed, Sequence)
        or isinstance(parsed, (str, bytes))
        or len(parsed) != 1
    ):
        raise PlanContractError(
            "corrected greedy sidecar does not contain exactly one parsed box"
        )
    if not isinstance(parsed[0], Mapping):
        raise PlanContractError("corrected greedy parsed sidecar box is invalid")
    source_box = _pixel_box(parsed[0].get("bbox"), "corrected greedy sidecar bbox")
    corrected = _sidecar_assignment(
        sidecar_id="sidecar:corrected-greedy:gt-7511-17:v2",
        source_kind="corrected_greedy_v2",
        coord_token_ids=coord,
        raw_token_ids=raw,
        source_bbox=source_box,
        owners=owners,
        candidates=candidates,
        source_ref={
            "path": str(behavior_path),
            "policy_view_id": "rp_1_00",
            "role_id": "due_turn:gt:7511:17",
            "arm_id": "forced_description_greedy",
            "expected_strict_owner_id": "gt:7511:17",
            "source_output_content_sha256": behavior.get("output_content_sha256"),
        },
    )
    if (
        corrected["strict_assignment_status"] != "matched"
        or corrected["strict_assignment_gt_owner_id"] != "gt:7511:17"
    ):
        raise PlanContractError(
            "corrected greedy sidecar loses its frozen gt:7511:17 owner"
        )
    sidecars.append(corrected)
    if len(sidecars) != 5 or len({row["sidecar_id"] for row in sidecars}) != 5:
        raise PlanContractError("sidecar registry is not exactly five frozen boxes")
    return sidecars


def sample_seed(role_id: str, sample_index: int) -> tuple[int, str]:
    payload = f"{UNIT_ID}|{role_id}|{sample_index}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF, digest.hex()


def build_sampling_seeds(contexts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for context in contexts:
        role_id = _string(context.get("context_id"), "sampling role ID")
        role_rows: list[dict[str, Any]] = []
        seen: set[int] = set()
        for sample_index in range(32):
            seed, digest = sample_seed(role_id, sample_index)
            if seed in seen:
                raise PlanContractError(f"sampling seed collision for role {role_id}")
            seen.add(seed)
            role_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "row_kind": "conditional_sampling_seed",
                    "role_id": role_id,
                    "sample_index": sample_index,
                    "seed": seed,
                    "utf8_sha256": digest,
                    "derivation": "first4_big_endian_and_0x7fffffff",
                    "sampling_policy": {
                        "temperature": 0.4,
                        "top_p": 0.95,
                        "repetition_penalty": 1.0,
                        "one_box_horizon": True,
                        "no_retry": True,
                    },
                }
            )
        if len(role_rows) != 32:
            raise PlanContractError("sampling role has a foreign K")
        rows.extend(role_rows)
    return rows


def build_scoring_requests(
    contexts: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    sidecars: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Join the shared bank to contexts without copying candidate geometry."""

    requests: list[dict[str, Any]] = []
    for context in contexts:
        context_id = str(context["context_id"])
        for candidate in candidates:
            requests.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "row_kind": "scoring_request",
                    "request_id": f"score:{context_id}:{candidate['candidate_id']}",
                    "request_kind": "primary",
                    "context_id": context_id,
                    "candidate_id": candidate["candidate_id"],
                    "candidate_source": "primary-candidates.jsonl",
                    "raw_complete_box_likelihood": True,
                    "repetition_penalty": 1.0,
                }
            )
        for sidecar in sidecars:
            if sidecar["requires_new_score_row"]:
                requests.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "row_kind": "scoring_request",
                        "request_id": f"score:{context_id}:{sidecar['sidecar_id']}",
                        "request_kind": "sidecar",
                        "context_id": context_id,
                        "sidecar_id": sidecar["sidecar_id"],
                        "candidate_source": "sidecars.jsonl",
                        "raw_complete_box_likelihood": True,
                        "repetition_penalty": 1.0,
                    }
                )
    exact_gt2 = next(
        row
        for row in candidates
        if row["generator_gt_owner_id"] == "gt:7511:2"
        and row["transform"] == "exact_gt_anchor"
    )
    for repeat_index in range(8):
        requests.append(
            {
                "schema_version": SCHEMA_VERSION,
                "row_kind": "scoring_request",
                "request_id": f"numerical-repeat:self-due-gt17:{repeat_index}",
                "request_kind": "numerical_repeat",
                "context_id": "self-due-gt17",
                "candidate_id": exact_gt2["candidate_id"],
                "repeat_index": repeat_index,
                "execution_requirement": "uncached_scalar_fp32_full_reforward",
                "raw_complete_box_likelihood": True,
                "repetition_penalty": 1.0,
            }
        )
    if len({row["request_id"] for row in requests}) != len(requests):
        raise PlanContractError("scoring request IDs are not unique")
    return requests


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def _receipt_bytes(receipt: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(receipt) + b"\n"


def _receipt_digest(receipt: Mapping[str, Any]) -> str:
    return sha256_json(
        {
            key: value
            for key, value in receipt.items()
            if key != "receipt_content_sha256"
        }
    )


def _materialize_output_bytes(
    *,
    owners: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    contexts: Sequence[Mapping[str, Any]],
    sidecars: Sequence[Mapping[str, Any]],
    seeds: Sequence[Mapping[str, Any]],
    requests: Sequence[Mapping[str, Any]],
    source_digests: Mapping[str, str],
    retention: Mapping[str, int],
    swap: Mapping[str, Any],
    matcher: Mapping[str, Any],
    behavior: Mapping[str, Any],
    sources: SourcePaths,
) -> dict[str, bytes]:
    files = {
        "owner-ledger.jsonl": _jsonl_bytes(owners),
        "primary-candidates.jsonl": _jsonl_bytes(candidates),
        "contexts.jsonl": _jsonl_bytes(contexts),
        "sidecars.jsonl": _jsonl_bytes(sidecars),
        "sampling-seeds.jsonl": _jsonl_bytes(seeds),
        "scoring-requests.jsonl": _jsonl_bytes(requests),
    }
    request_counts: dict[str, int] = {}
    for request in requests:
        kind = str(request["request_kind"])
        request_counts[kind] = request_counts.get(kind, 0) + 1
    code_path = Path(__file__).resolve()
    contract = behavior.get("contract")
    if not isinstance(contract, Mapping):
        raise PlanContractError("corrected behavior source lacks a runtime contract")
    runtime = contract.get("frozen_runtime_identity")
    if not isinstance(runtime, Mapping):
        raise PlanContractError(
            "corrected behavior source lacks frozen runtime identity"
        )
    tokenizer = runtime.get("tokenizer")
    if (
        not isinstance(tokenizer, Mapping)
        or tokenizer.get("identity_sha256") != TOKENIZER_IDENTITY_SHA256
    ):
        raise PlanContractError(
            "corrected behavior source does not bind the frozen tokenizer identity"
        )
    frozen_config = {
        "candidate_budget": {"exact": 1, "translation": 4, "extent": 4},
        "translation_order": list(TRANSLATION_TRANSFORMS),
        "extent_order": list(EXTENT_TRANSFORMS),
        "contexts": [
            {"context_id": context_id, "donor_row_indices": list(row_indices)}
            for context_id, row_indices in CONTEXT_SPECS
        ],
        "blocked_context_ids": ["skip-post-gt22"],
        "sampling": {
            "K": 32,
            "temperature": 0.4,
            "top_p": 0.95,
            "repetition_penalty": 1.0,
        },
        "numeric_repeats": {
            "context_id": "self-due-gt17",
            "owner_id": "gt:7511:2",
            "K": 8,
        },
    }
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "execution_surface": "deterministic_cpu_planner_no_model_no_tokenizer_no_gpu",
        "source_digests": dict(source_digests),
        "source_paths": {
            field: str(getattr(sources, field))
            for field in SourcePaths.__dataclass_fields__
        },
        "code": {"path": str(code_path), "sha256": sha256_file(code_path)},
        "frozen_config": frozen_config,
        "frozen_config_sha256": sha256_json(frozen_config),
        "frozen_runtime_identity": {
            "image_id": IMAGE_ID,
            "image_width": IMAGE_WIDTH,
            "image_height": IMAGE_HEIGHT,
            "canonical_description": DESCRIPTION,
            "prompt_token_ids_sha256": PROMPT_SHA256,
            "donor_generated_token_ids_sha256": GENERATED_SHA256,
            "executed_media_sha256": EXECUTED_MEDIA_SHA256,
            "image_sha256": IMAGE_SHA256,
            "tokenizer_identity_sha256": TOKENIZER_IDENTITY_SHA256,
            "coordinate_token_ids": {
                "start": COORD_TOKEN_START,
                "end": COORD_TOKEN_END,
            },
            "wrapper_token_ids": {
                "object_ref_start": OBJECT_REF_START,
                "object_ref_end": OBJECT_REF_END,
                "box_start": BOX_START,
                "box_end": BOX_END,
            },
            "pixel_to_coordinate_conversion": "round(pixel*1000/extent), clamp 0..999",
        },
        "strict_matcher": {
            "schema_version": matcher["schema_version"],
            "iou_threshold": matcher["iou_threshold"],
            "assignment_objective": matcher.get("assignment_objective"),
            "global_optimum_ambiguity": matcher.get("global_optimum_ambiguity"),
            "candidate_join_scope": "every candidate evaluated against all 41 projected owners",
        },
        "candidate_generation": {
            "exact_per_owner": 1,
            "translation_per_owner": 4,
            "extent_per_owner": 4,
            "translation_order": list(TRANSLATION_TRANSFORMS),
            "extent_order": list(EXTENT_TRANSFORMS),
            "primary_candidate_universe_sha256": sha256_json(list(candidates)),
            "retention_unique_assignment_counts": dict(retention),
            "retention_gate": "all target/control owners require at least 5 of 9",
        },
        "context_admission": {
            "admitted_context_ids": [context["context_id"] for context in contexts],
            "blocked_context_ids": ["skip-post-gt22"],
            "teacher_forced_chosen_token_parity": {
                "status": "required_not_cpu_verifiable",
                "claimed_pass": False,
                "must_pass_before_model_scoring": True,
            },
            "same_length_owner_swap": dict(swap),
        },
        "counts": {
            "owners": len(owners),
            "primary_candidates": len(candidates),
            "contexts": len(contexts),
            "sidecars": len(sidecars),
            "sampling_seeds": len(seeds),
            "scoring_requests": len(requests),
            "scoring_requests_by_kind": request_counts,
        },
        "output_file_digests": {
            name: hashlib.sha256(content).hexdigest() for name, content in files.items()
        },
    }
    receipt["receipt_content_sha256"] = _receipt_digest(receipt)
    files["receipt.json"] = _receipt_bytes(receipt)
    return files


def _commit_create_or_identical(output_dir: Path, files: Mapping[str, bytes]) -> str:
    expected_names = set(files)
    if output_dir.exists():
        if not output_dir.is_dir():
            raise PlanContractError(
                f"output path exists but is not a directory: {output_dir}"
            )
        existing_names = {path.name for path in output_dir.iterdir() if path.is_file()}
        if existing_names != expected_names:
            raise PlanContractError(
                "output directory already exists with a foreign or partial file set"
            )
        for name, expected in files.items():
            if (output_dir / name).read_bytes() != expected:
                raise PlanContractError(
                    "output directory already exists with non-identical content"
                )
        return "identical_existing_output"

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        for name, content in files.items():
            (temp_dir / name).write_bytes(content)
        os.replace(temp_dir, output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return "created"


def build_sorted_all_person_route_landscape(
    output_dir: str | Path, *, sources: SourcePaths = DEFAULT_SOURCES
) -> dict[str, Any]:
    """Validate and publish the complete deterministic CPU plan."""

    source_digests = _source_digests(sources)
    owners = _load_owners(sources.owner_ledger)
    matcher = _load_matcher(sources.matcher)
    prediction_rows = _load_prediction_rows(sources.prediction_ledger)
    rollout, donor_rows, parsed_donor_rows = _load_donor(sources.donor, prediction_rows)
    candidates = build_primary_candidates(owners)
    retention = enforce_retention(candidates)
    contexts = build_contexts(rollout, donor_rows, parsed_donor_rows)
    swap = owner_swap_receipt(contexts)
    behavior = _read_json(sources.behavior, "corrected greedy behavior")
    sidecars = build_sidecars(
        donor_rows,
        parsed_donor_rows,
        behavior,
        owners,
        candidates,
        sources.behavior,
    )
    seeds = build_sampling_seeds(contexts)
    requests = build_scoring_requests(contexts, candidates, sidecars)
    files = _materialize_output_bytes(
        owners=owners,
        candidates=candidates,
        contexts=contexts,
        sidecars=sidecars,
        seeds=seeds,
        requests=requests,
        source_digests=source_digests,
        retention=retention,
        swap=swap,
        matcher=matcher,
        behavior=behavior,
        sources=sources,
    )
    status = _commit_create_or_identical(Path(output_dir), files)
    return {
        "status": status,
        "output_dir": str(Path(output_dir)),
        "counts": {
            "owners": len(owners),
            "primary_candidates": len(candidates),
            "contexts": len(contexts),
            "sidecars": len(sidecars),
            "sampling_seeds": len(seeds),
            "scoring_requests": len(requests),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--unit-path", type=Path, default=UNIT_PATH)
    parser.add_argument("--panel-path", type=Path, default=PANEL_PATH)
    parser.add_argument("--owner-ledger-path", type=Path, default=OWNER_LEDGER_PATH)
    parser.add_argument(
        "--prediction-ledger-path", type=Path, default=PREDICTION_LEDGER_PATH
    )
    parser.add_argument("--matcher-path", type=Path, default=MATCHER_PATH)
    parser.add_argument("--donor-path", type=Path, default=DONOR_PATH)
    parser.add_argument("--behavior-path", type=Path, default=BEHAVIOR_PATH)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = build_sorted_all_person_route_landscape(
        args.output_dir,
        sources=SourcePaths(
            unit=args.unit_path,
            panel=args.panel_path,
            owner_ledger=args.owner_ledger_path,
            prediction_ledger=args.prediction_ledger_path,
            matcher=args.matcher_path,
            donor=args.donor_path,
            behavior=args.behavior_path,
        ),
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
