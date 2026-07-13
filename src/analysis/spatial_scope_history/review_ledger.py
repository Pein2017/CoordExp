"""Validate blinded reviews and assemble the sealed audit reference ledger."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Literal

from src.analysis.spatial_scope_history.cohort_ledger import canonical_json_text
from src.analysis.spatial_scope_history.metrics import _exact_maximum_flow_assignment
from src.eval.detection_categories import (
    COCO_80_CATEGORY_NAMESPACE_SHA256,
    COCO_80_EVALUATOR_LOCAL_CATEGORY_ID_BY_NAME,
    COCO_80_OFFICIAL_CATEGORY_ID_BY_NAME,
    normalize_coco_category_name,
)
from src.vis.matching import iou_xyxy


REVIEWER_LABEL_SCHEMA_VERSION = "dense-union-51.reviewer-label.v1"
ADJUDICATION_QUEUE_SCHEMA_VERSION = "dense-union-51.adjudication-queue.v1"
ANNOTATION_COHORT_SCHEMA_VERSION = (
    "spatial_scope_history.annotation_derived_cohort_materialization.v1"
)
REVIEW_PACKET_ID = "dense-union-51-image-only-review-v1"
REVIEWER_ROLES = ("reviewer-one", "reviewer-two")
PAIRING_INTERSECTION_OVER_UNION_THRESHOLD = 0.50
EXPECTED_REVIEW_QUEUE_ROWS = 102
EXPECTED_REVIEW_IMAGES = 51

ReviewerState = Literal["accepted", "ambiguous", "partial", "crowd", "out-of-scope"]
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REVIEWER_FIELDS = frozenset(
    {
        "schema_version",
        "packet_id",
        "review_identifier",
        "reviewer_role_identifier",
        "reviewer_local_object_identifier",
        "image_id",
        "normalized_category_name",
        "official_coco_category_id",
        "candidate_categories",
        "source_canvas_box_xyxy",
        "reviewer_state",
        "reason_code",
    }
)
_REVIEWER_REASON_BY_STATE = {
    "accepted": {"none", "occluded_but_boxable", "truncated_but_boxable"},
    "ambiguous": {"category_not_unique", "instance_not_separable"},
    "partial": {"boundary_not_reproducible", "instance_not_separable"},
    "crowd": {"instance_not_separable"},
    "out-of-scope": {"non_coco80"},
}
_STATE_ORDER = {
    "accepted": 0,
    "ambiguous": 1,
    "partial": 2,
    "crowd": 3,
    "out-of-scope": 4,
}


@dataclass(frozen=True)
class _ReviewContext:
    queue_rows: tuple[Mapping[str, Any], ...]
    labels_by_role: Mapping[str, tuple[Mapping[str, Any], ...]]
    official_rows: tuple[Mapping[str, Any], ...]
    crowd_rows: tuple[Mapping[str, Any], ...]
    image_order: Mapping[int, int]
    image_facts: Mapping[int, Mapping[str, Any]]
    packet_sha256: str
    ontology_sha256: str
    review_queue_sha256: str
    reviewer_label_sha256_by_role: Mapping[str, str]
    official_individual_ledger_sha256: str
    official_crowd_ledger_sha256: str


def build_adjudication_queue(
    *,
    review_queue_jsonl: bytes,
    reviewer_one_labels_jsonl: bytes,
    reviewer_two_labels_jsonl: bytes,
    official_individual_ledger_jsonl: bytes,
    official_crowd_ledger_jsonl: bytes,
    expected_review_queue_sha256: str,
    expected_packet_sha256: str,
    expected_ontology_sha256: str,
) -> bytes:
    """Validate blinded review inputs and return a decision-only queue."""

    context = _validate_review_inputs(
        review_queue_jsonl=review_queue_jsonl,
        reviewer_one_labels_jsonl=reviewer_one_labels_jsonl,
        reviewer_two_labels_jsonl=reviewer_two_labels_jsonl,
        official_individual_ledger_jsonl=official_individual_ledger_jsonl,
        official_crowd_ledger_jsonl=official_crowd_ledger_jsonl,
        expected_review_queue_sha256=expected_review_queue_sha256,
        expected_packet_sha256=expected_packet_sha256,
        expected_ontology_sha256=expected_ontology_sha256,
    )
    return _jsonl_bytes(_adjudication_queue_rows(context))


def _validate_review_inputs(
    *,
    review_queue_jsonl: bytes,
    reviewer_one_labels_jsonl: bytes,
    reviewer_two_labels_jsonl: bytes,
    official_individual_ledger_jsonl: bytes,
    official_crowd_ledger_jsonl: bytes,
    expected_review_queue_sha256: str,
    expected_packet_sha256: str,
    expected_ontology_sha256: str,
) -> _ReviewContext:
    for digest, field in (
        (expected_review_queue_sha256, "expected_review_queue_sha256"),
        (expected_packet_sha256, "expected_packet_sha256"),
        (expected_ontology_sha256, "expected_ontology_sha256"),
    ):
        _require_sha256(digest, field=field)
    if _sha256_bytes(review_queue_jsonl) != expected_review_queue_sha256:
        raise ValueError("review queue digest does not match the sealed expectation")
    queue_rows = tuple(
        _parse_canonical_jsonl(review_queue_jsonl, artifact_name="review queue")
    )
    if len(queue_rows) != EXPECTED_REVIEW_QUEUE_ROWS:
        raise ValueError(f"review queue must contain {EXPECTED_REVIEW_QUEUE_ROWS} rows")
    image_order, image_facts, queue_by_role = _validate_review_queue(
        queue_rows,
        expected_packet_sha256=expected_packet_sha256,
        expected_ontology_sha256=expected_ontology_sha256,
    )
    labels_by_role = {
        "reviewer-one": _validate_reviewer_labels(
            reviewer_one_labels_jsonl,
            role="reviewer-one",
            queue_by_review_identifier=queue_by_role["reviewer-one"],
            image_order=image_order,
        ),
        "reviewer-two": _validate_reviewer_labels(
            reviewer_two_labels_jsonl,
            role="reviewer-two",
            queue_by_review_identifier=queue_by_role["reviewer-two"],
            image_order=image_order,
        ),
    }
    official_rows = tuple(
        _parse_canonical_jsonl(
            official_individual_ledger_jsonl,
            artifact_name="official individual ledger",
        )
    )
    crowd_rows = tuple(
        _parse_canonical_jsonl(
            official_crowd_ledger_jsonl,
            artifact_name="official crowd ledger",
        )
    )
    _validate_official_rows(
        official_rows,
        image_facts=image_facts,
        image_order=image_order,
        expect_crowd=False,
    )
    _validate_official_rows(
        crowd_rows,
        image_facts=image_facts,
        image_order=image_order,
        expect_crowd=True,
    )
    return _ReviewContext(
        queue_rows=queue_rows,
        labels_by_role=labels_by_role,
        official_rows=official_rows,
        crowd_rows=crowd_rows,
        image_order=image_order,
        image_facts=image_facts,
        packet_sha256=expected_packet_sha256,
        ontology_sha256=expected_ontology_sha256,
        review_queue_sha256=expected_review_queue_sha256,
        reviewer_label_sha256_by_role={
            "reviewer-one": _sha256_bytes(reviewer_one_labels_jsonl),
            "reviewer-two": _sha256_bytes(reviewer_two_labels_jsonl),
        },
        official_individual_ledger_sha256=_sha256_bytes(
            official_individual_ledger_jsonl
        ),
        official_crowd_ledger_sha256=_sha256_bytes(official_crowd_ledger_jsonl),
    )


def _validate_review_queue(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_packet_sha256: str,
    expected_ontology_sha256: str,
) -> tuple[
    dict[int, int],
    dict[int, Mapping[str, Any]],
    dict[str, dict[str, Mapping[str, Any]]],
]:
    expected_fields = {
        "image_id",
        "image_path",
        "image_sha256",
        "ontology",
        "review_identifier",
        "reviewer_instruction_packet",
        "reviewer_role_slot",
        "schema_version",
        "source_image_height",
        "source_image_width",
    }
    queue_by_role: dict[str, dict[str, Mapping[str, Any]]] = {
        role: {} for role in REVIEWER_ROLES
    }
    image_order: dict[int, int] = {}
    image_facts: dict[int, Mapping[str, Any]] = {}
    seen_pair: set[tuple[int, str]] = set()
    prior_image_id: int | None = None
    for row_index, row in enumerate(rows):
        _require_exact_fields(row, expected_fields, record="review queue row")
        if row["schema_version"] != ANNOTATION_COHORT_SCHEMA_VERSION:
            raise ValueError("review queue schema version mismatch")
        role_slot = _require_mapping(row["reviewer_role_slot"], "reviewer_role_slot")
        role = role_slot.get("role_identifier")
        if role not in REVIEWER_ROLES:
            raise ValueError(f"unknown reviewer role: {role}")
        image_id = _require_int(row["image_id"], field="image_id", minimum=0)
        pair = (image_id, role)
        if pair in seen_pair:
            raise ValueError(f"duplicate review queue role assignment: {pair}")
        seen_pair.add(pair)
        review_identifier = _require_nonempty(
            row["review_identifier"], field="review_identifier"
        )
        expected_identifier = f"dense-union-51-review:{image_id}:{role}"
        if review_identifier != expected_identifier:
            raise ValueError("review identifier is not canonical")
        packet = _require_mapping(
            row["reviewer_instruction_packet"], "reviewer_instruction_packet"
        )
        ontology = _require_mapping(row["ontology"], "ontology")
        if packet.get("sha256") != expected_packet_sha256:
            raise ValueError("review queue packet digest drift")
        if ontology.get("sha256") != expected_ontology_sha256:
            raise ValueError("review queue ontology digest drift")
        image_path = Path(_require_nonempty(row["image_path"], field="image_path"))
        if not image_path.is_file():
            raise ValueError(f"review queue image is missing: {image_path}")
        image_sha256 = _require_sha256(row["image_sha256"], field="image_sha256")
        if _sha256_bytes(image_path.read_bytes()) != image_sha256:
            raise ValueError(f"review queue image digest drift: {image_path}")
        facts = {
            "image_id": image_id,
            "image_path": str(image_path),
            "image_sha256": image_sha256,
            "source_image_height": _require_int(
                row["source_image_height"], field="source_image_height", minimum=1
            ),
            "source_image_width": _require_int(
                row["source_image_width"], field="source_image_width", minimum=1
            ),
        }
        prior = image_facts.get(image_id)
        if prior is not None and prior != facts:
            raise ValueError("cross-role image facts disagree")
        image_facts[image_id] = facts
        if role == "reviewer-one":
            image_order[image_id] = len(image_order)
        queue_by_role[role][review_identifier] = row
        expected_role_for_position = REVIEWER_ROLES[row_index % len(REVIEWER_ROLES)]
        if role != expected_role_for_position:
            raise ValueError("review queue role order is not canonical")
        if role == "reviewer-one":
            prior_image_id = image_id
        elif prior_image_id != image_id:
            raise ValueError("review queue role rows for one image must be adjacent")
    if len(image_facts) != EXPECTED_REVIEW_IMAGES:
        raise ValueError(f"review queue must cover {EXPECTED_REVIEW_IMAGES} images")
    expected_pairs = {
        (image_id, role) for image_id in image_facts for role in REVIEWER_ROLES
    }
    if seen_pair != expected_pairs:
        raise ValueError("review queue does not assign both roles to every image")
    return image_order, image_facts, queue_by_role


def _validate_reviewer_labels(
    payload: bytes,
    *,
    role: str,
    queue_by_review_identifier: Mapping[str, Mapping[str, Any]],
    image_order: Mapping[int, int],
) -> tuple[Mapping[str, Any], ...]:
    rows = tuple(
        _parse_canonical_jsonl(payload, artifact_name=f"{role} label artifact")
    )
    seen_ids: set[str] = set()
    seen_review_ids: set[str] = set()
    prior_order_key: tuple[Any, ...] | None = None
    labels_by_image: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        _require_exact_fields(row, _REVIEWER_FIELDS, record="reviewer label")
        if "category_id" in row:
            raise ValueError("bare category_id is forbidden")
        if row["schema_version"] != REVIEWER_LABEL_SCHEMA_VERSION:
            raise ValueError("reviewer label schema version mismatch")
        if row["packet_id"] != REVIEW_PACKET_ID:
            raise ValueError("reviewer packet identifier mismatch")
        if row["reviewer_role_identifier"] != role:
            raise ValueError("cross-role reviewer-label leakage")
        review_identifier = _require_nonempty(
            row["review_identifier"], field="review_identifier"
        )
        queue_row = queue_by_review_identifier.get(review_identifier)
        if queue_row is None:
            raise ValueError("reviewer label is not assigned to this role")
        image_id = _require_int(row["image_id"], field="image_id", minimum=0)
        if image_id != queue_row["image_id"]:
            raise ValueError("reviewer label image differs from queue")
        local_id = _require_nonempty(
            row["reviewer_local_object_identifier"],
            field="reviewer_local_object_identifier",
        )
        if local_id in seen_ids:
            raise ValueError(f"duplicate reviewer label identifier: {local_id}")
        seen_ids.add(local_id)
        seen_review_ids.add(review_identifier)
        _validate_reviewer_category_and_state(row)
        _validate_box_for_image(
            row["source_canvas_box_xyxy"],
            state=row["reviewer_state"],
            reason=row["reason_code"],
            width=int(queue_row["source_image_width"]),
            height=int(queue_row["source_image_height"]),
        )
        labels_by_image[image_id].append(row)
    if seen_review_ids != set(queue_by_review_identifier):
        raise ValueError("reviewer artifact does not prove disposition of every image")
    canonical: list[Mapping[str, Any]] = []
    for image_id in sorted(labels_by_image, key=image_order.__getitem__):
        candidates = labels_by_image[image_id]
        sorted_candidates = sorted(candidates, key=_reviewer_sort_key)
        keys = [_reviewer_sort_key(row) for row in sorted_candidates]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate reviewer labels share a complete canonical key")
        for ordinal, row in enumerate(sorted_candidates, start=1):
            expected_id = f"{role}:{image_id}:{ordinal:04d}"
            if row["reviewer_local_object_identifier"] != expected_id:
                raise ValueError("reviewer local object identifier is not canonical")
            order_key = (image_order[image_id], expected_id)
            if prior_order_key is not None and order_key <= prior_order_key:
                raise ValueError("reviewer rows are not in canonical order")
            prior_order_key = order_key
            canonical.append(row)
    if tuple(canonical) != rows:
        raise ValueError("reviewer artifact row order is not canonical")
    return rows


def _validate_reviewer_category_and_state(row: Mapping[str, Any]) -> None:
    state = row["reviewer_state"]
    reason = row["reason_code"]
    if state not in _REVIEWER_REASON_BY_STATE:
        raise ValueError(f"unknown reviewer state: {state}")
    if reason not in _REVIEWER_REASON_BY_STATE[state]:
        raise ValueError(
            f"reviewer reason is incompatible with state: {state}/{reason}"
        )
    candidates = _validate_candidate_categories(row["candidate_categories"])
    name = row["normalized_category_name"]
    official_id = row["official_coco_category_id"]
    if state in {"accepted", "partial", "crowd"}:
        _validate_category_pair(name, official_id)
        if candidates:
            raise ValueError(
                f"{state} reviewer record cannot have candidate categories"
            )
    elif state == "ambiguous":
        if not candidates:
            raise ValueError("ambiguous reviewer record requires candidate categories")
        if len(candidates) == 1:
            if (name, official_id) != (
                candidates[0]["normalized_category_name"],
                candidates[0]["official_coco_category_id"],
            ):
                raise ValueError("single-candidate ambiguous category fields disagree")
        elif name is not None or official_id is not None:
            raise ValueError("multi-candidate ambiguous category fields must be null")
    elif name is not None or official_id is not None or candidates:
        raise ValueError("out-of-scope category fields must be null or empty")


def _validate_official_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    image_facts: Mapping[int, Mapping[str, Any]],
    image_order: Mapping[int, int],
    expect_crowd: bool,
) -> None:
    seen: set[str] = set()
    prior_order_key: tuple[int, str] | None = None
    for row in rows:
        if "category_id" in row:
            raise ValueError("bare category_id is forbidden in official ledgers")
        image_id = _require_int(row.get("image_id"), field="image_id", minimum=0)
        if image_id not in image_facts:
            raise ValueError(
                "official ledger contains an image outside the review queue"
            )
        if row.get("source_image_sha256") != image_facts[image_id]["image_sha256"]:
            raise ValueError("official ledger source image digest drift")
        iscrowd = _require_int(row.get("iscrowd"), field="iscrowd", minimum=0)
        if bool(iscrowd) is not expect_crowd or iscrowd not in {0, 1}:
            raise ValueError("official ledger crowd namespace mismatch")
        identifier = _require_nonempty(
            row.get("object_or_region_identifier"),
            field="object_or_region_identifier",
        )
        required_prefix = "coco-crowd:" if expect_crowd else "coco-ann:"
        if not identifier.startswith(required_prefix) or identifier in seen:
            raise ValueError("official object identifier is invalid or duplicated")
        seen.add(identifier)
        _validate_category_pair(
            row.get("normalized_category_name"),
            row.get("official_coco_category_id"),
        )
        expected_evaluator_id = COCO_80_EVALUATOR_LOCAL_CATEGORY_ID_BY_NAME[
            row["normalized_category_name"]
        ]
        if row.get("evaluator_local_category_id") != expected_evaluator_id:
            raise ValueError(
                "official and evaluator-local category identifiers disagree"
            )
        if row.get("coco_80_category_namespace_sha256") != (
            COCO_80_CATEGORY_NAMESPACE_SHA256
        ):
            raise ValueError("official ledger category namespace digest drift")
        geometry = _require_mapping(row.get("geometry"), "geometry")
        box = geometry.get("clipped_source_corners_xyxy")
        x1, y1, x2, y2 = _validate_numeric_box(box)
        facts = image_facts[image_id]
        if not (
            0 <= x1 < x2 <= int(facts["source_image_width"])
            and 0 <= y1 < y2 <= int(facts["source_image_height"])
        ):
            raise ValueError("official source-canvas box is outside image bounds")
        order_key = (image_order[image_id], identifier)
        if prior_order_key is not None and order_key <= prior_order_key:
            raise ValueError("official ledger rows are not in canonical order")
        prior_order_key = order_key


def _adjudication_queue_rows(context: _ReviewContext) -> list[Mapping[str, Any]]:
    labels_by_image_role: dict[int, dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: {role: [] for role in REVIEWER_ROLES}
    )
    for role, rows in context.labels_by_role.items():
        for row in rows:
            labels_by_image_role[int(row["image_id"])][role].append(row)
    official_by_image = _rows_by_image(context.official_rows)
    crowd_by_image = _rows_by_image(context.crowd_rows)
    result: list[Mapping[str, Any]] = []
    for image_id in sorted(context.image_order, key=context.image_order.__getitem__):
        one = labels_by_image_role[image_id]["reviewer-one"]
        two = labels_by_image_role[image_id]["reviewer-two"]
        pairs, unmatched_one, unmatched_two = _pair_reviewers(one, two)
        groups: list[dict[str, Any]] = []
        for left_index, right_index, overlap in pairs:
            groups.append(
                _reviewer_group(
                    [one[left_index], two[right_index]],
                    overlap=overlap,
                    official=official_by_image[image_id],
                    crowd=crowd_by_image[image_id],
                )
            )
        for row in [one[index] for index in unmatched_one] + [
            two[index] for index in unmatched_two
        ]:
            groups.append(
                _reviewer_group(
                    [row],
                    overlap=None,
                    official=official_by_image[image_id],
                    crowd=crowd_by_image[image_id],
                )
            )
        for row in official_by_image[image_id]:
            groups.append(
                _source_only_group(
                    group_kind="official-individual",
                    row=row,
                    labels=one + two,
                )
            )
        for row in crowd_by_image[image_id]:
            groups.append(
                _source_only_group(
                    group_kind="official-crowd-region",
                    row=row,
                    labels=one + two,
                )
            )
        groups.sort(key=_adjudication_group_sort_key)
        for ordinal, group in enumerate(groups, start=1):
            result.append(
                {
                    **group,
                    "adjudication_identifier": (
                        f"dense-union-51-adjudication:{image_id}:{ordinal:04d}"
                    ),
                    "category_namespace_sha256": COCO_80_CATEGORY_NAMESPACE_SHA256,
                    "image_id": image_id,
                    "image_sha256": context.image_facts[image_id]["image_sha256"],
                    "ontology_sha256": context.ontology_sha256,
                    "official_crowd_ledger_sha256": (
                        context.official_crowd_ledger_sha256
                    ),
                    "official_individual_ledger_sha256": (
                        context.official_individual_ledger_sha256
                    ),
                    "packet_id": REVIEW_PACKET_ID,
                    "packet_sha256": context.packet_sha256,
                    "review_queue_sha256": context.review_queue_sha256,
                    "reviewer_label_sha256_by_role": dict(
                        context.reviewer_label_sha256_by_role
                    ),
                    "schema_version": ADJUDICATION_QUEUE_SCHEMA_VERSION,
                }
            )
    return result


def _pair_reviewers(
    one: Sequence[Mapping[str, Any]], two: Sequence[Mapping[str, Any]]
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    candidate_rows: list[tuple[int, int, float]] = []
    for left_index, left in enumerate(one):
        left_category = _single_pairable_category(left)
        left_box = left["source_canvas_box_xyxy"]
        if left_category is None or left_box is None:
            continue
        for right_index, right in enumerate(two):
            if left_category != _single_pairable_category(right):
                continue
            right_box = right["source_canvas_box_xyxy"]
            if right_box is None:
                continue
            overlap = float(iou_xyxy(tuple(left_box), tuple(right_box)))
            if overlap >= PAIRING_INTERSECTION_OVER_UNION_THRESHOLD:
                candidate_rows.append((left_index, right_index, overlap))
    candidate_rows.sort(
        key=lambda row: (
            one[row[0]]["reviewer_local_object_identifier"],
            two[row[1]]["reviewer_local_object_identifier"],
        )
    )
    if candidate_rows:
        denominators = [
            overlap.as_integer_ratio()[1] for _, _, overlap in candidate_rows
        ]
        common_denominator = max(denominators)
        lexicographic_base = 1 << len(candidate_rows)
        benefits: list[int] = []
        for edge_index, (_, _, overlap) in enumerate(candidate_rows):
            numerator, denominator = overlap.as_integer_ratio()
            exact_iou_integer = numerator * (common_denominator // denominator)
            lexicographic_bit = 1 << (len(candidate_rows) - edge_index - 1)
            benefits.append(exact_iou_integer * lexicographic_base + lexicographic_bit)
        selected = _exact_maximum_flow_assignment(
            prediction_count=len(one),
            reference_count=len(two),
            candidate_rows=candidate_rows,
            benefits=benefits,
        )
    else:
        selected = frozenset()
    pairs = [candidate_rows[index] for index in sorted(selected)]
    matched_one = {row[0] for row in pairs}
    matched_two = {row[1] for row in pairs}
    return (
        pairs,
        [index for index in range(len(one)) if index not in matched_one],
        [index for index in range(len(two)) if index not in matched_two],
    )


def _reviewer_group(
    labels: Sequence[Mapping[str, Any]],
    *,
    overlap: float | None,
    official: Sequence[Mapping[str, Any]],
    crowd: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    category_names = sorted(
        {name for label in labels for name in _candidate_names(label)}
    )
    boxes = [
        {
            "source_canvas_box_xyxy": label["source_canvas_box_xyxy"],
            "source_record_identifier": label["reviewer_local_object_identifier"],
        }
        for label in labels
        if label["source_canvas_box_xyxy"] is not None
    ]
    linked_official = _overlapping_source_ids(labels, official)
    linked_crowd = _overlapping_source_ids(labels, crowd)
    return {
        "candidate_category_names": category_names,
        "candidate_source_canvas_boxes": boxes,
        "group_kind": "reviewer-proposal",
        "linked_official_crowd_region_identifiers": linked_crowd,
        "linked_official_object_identifiers": linked_official,
        "linked_reviewer_label_identifiers": sorted(
            label["reviewer_local_object_identifier"] for label in labels
        ),
        "reviewer_pair_intersection_over_union": overlap,
    }


def _source_only_group(
    *,
    group_kind: str,
    row: Mapping[str, Any],
    labels: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    identifier = row["object_or_region_identifier"]
    relevant_labels = _overlapping_label_ids(row, labels)
    is_crowd = group_kind == "official-crowd-region"
    return {
        "candidate_category_names": [row["normalized_category_name"]],
        "candidate_source_canvas_boxes": [
            {
                "source_canvas_box_xyxy": row["geometry"][
                    "clipped_source_corners_xyxy"
                ],
                "source_record_identifier": identifier,
            }
        ],
        "group_kind": group_kind,
        "linked_official_crowd_region_identifiers": [identifier] if is_crowd else [],
        "linked_official_object_identifiers": [] if is_crowd else [identifier],
        "linked_reviewer_label_identifiers": relevant_labels,
        "reviewer_pair_intersection_over_union": None,
    }


def _validate_candidate_categories(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("candidate categories must be a list")
    expected_fields = {"normalized_category_name", "official_coco_category_id"}
    result: list[Mapping[str, Any]] = []
    for item in value:
        mapping = _require_mapping(item, "candidate category")
        _require_exact_fields(mapping, expected_fields, record="candidate category")
        _validate_category_pair(
            mapping["normalized_category_name"], mapping["official_coco_category_id"]
        )
        result.append(mapping)
    expected = sorted(
        result,
        key=lambda item: (
            item["official_coco_category_id"],
            item["normalized_category_name"],
        ),
    )
    if result != expected or len({canonical_json_text(row) for row in result}) != len(
        result
    ):
        raise ValueError("candidate categories must be sorted and unique")
    return result


def _validate_category_pair(name: Any, official_id: Any) -> None:
    if not isinstance(name, str) or normalize_coco_category_name(name) != name:
        raise ValueError("category name must be canonical and normalized")
    expected = COCO_80_OFFICIAL_CATEGORY_ID_BY_NAME.get(name)
    if expected is None or not isinstance(official_id, int) or official_id != expected:
        raise ValueError("official category identifier and name disagree")


def _validate_box_for_image(
    value: Any,
    *,
    state: str,
    reason: str,
    width: int,
    height: int,
) -> None:
    if value is None:
        if state != "partial" or reason != "boundary_not_reproducible":
            raise ValueError(
                "null box is allowed only for boundary-unreproducible partial"
            )
        return
    box = _validate_numeric_box(value, require_integer=True)
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("source-canvas box is outside image bounds")


def _validate_numeric_box(
    value: Any, *, require_integer: bool = False
) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("source-canvas box must contain four coordinates")
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float)) for item in value
    ):
        raise ValueError("source-canvas box coordinates must be numeric")
    if require_integer and any(not isinstance(item, int) for item in value):
        raise ValueError("review source-canvas box coordinates must be integers")
    box = tuple(float(item) for item in value)
    if not (box[0] < box[2] and box[1] < box[3]):
        raise ValueError("source-canvas box must have positive extent")
    return box  # type: ignore[return-value]


def _reviewer_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    box = row["source_canvas_box_xyxy"]
    spatial = tuple(box) if box is not None else (float("inf"),) * 4
    official_id = row["official_coco_category_id"]
    return (
        0 if box is not None else 1,
        spatial[1],
        spatial[0],
        spatial[3],
        spatial[2],
        _STATE_ORDER[row["reviewer_state"]],
        official_id if official_id is not None else float("inf"),
        canonical_json_text(row["candidate_categories"]),
    )


def _single_pairable_category(row: Mapping[str, Any]) -> str | None:
    if row["reviewer_state"] in {"crowd", "out-of-scope"}:
        return None
    name = row["normalized_category_name"]
    return name if isinstance(name, str) else None


def _candidate_names(row: Mapping[str, Any]) -> set[str]:
    if row["candidate_categories"]:
        return {
            item["normalized_category_name"] for item in row["candidate_categories"]
        }
    name = row["normalized_category_name"]
    return {name} if isinstance(name, str) else set()


def _overlapping_source_ids(
    labels: Sequence[Mapping[str, Any]], sources: Sequence[Mapping[str, Any]]
) -> list[str]:
    result: set[str] = set()
    for label in labels:
        box = label["source_canvas_box_xyxy"]
        if box is None:
            continue
        names = _candidate_names(label)
        for source in sources:
            if source["normalized_category_name"] not in names:
                continue
            source_box = source["geometry"]["clipped_source_corners_xyxy"]
            if iou_xyxy(tuple(box), tuple(source_box)) >= 0.50:
                result.add(source["object_or_region_identifier"])
    return sorted(result)


def _overlapping_label_ids(
    source: Mapping[str, Any], labels: Sequence[Mapping[str, Any]]
) -> list[str]:
    result: list[str] = []
    source_box = source["geometry"]["clipped_source_corners_xyxy"]
    for label in labels:
        box = label["source_canvas_box_xyxy"]
        if (
            box is not None
            and source["normalized_category_name"] in _candidate_names(label)
            and iou_xyxy(tuple(box), tuple(source_box)) >= 0.50
        ):
            result.append(label["reviewer_local_object_identifier"])
    return sorted(result)


def _adjudication_group_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    kind_order = {
        "reviewer-proposal": 0,
        "official-individual": 1,
        "official-crowd-region": 2,
    }
    return (
        kind_order[row["group_kind"]],
        tuple(row["linked_reviewer_label_identifiers"]),
        tuple(row["linked_official_object_identifiers"]),
        tuple(row["linked_official_crowd_region_identifiers"]),
    )


def _rows_by_image(
    rows: Sequence[Mapping[str, Any]],
) -> dict[int, list[Mapping[str, Any]]]:
    result: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        result[int(row["image_id"])].append(row)
    for values in result.values():
        values.sort(key=lambda row: row["object_or_region_identifier"])
    return result


def _parse_canonical_jsonl(
    payload: bytes, *, artifact_name: str
) -> list[Mapping[str, Any]]:
    if not payload or not payload.endswith(b"\n"):
        raise ValueError(f"{artifact_name} must be nonempty and end with a newline")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{artifact_name} must use UTF-8") from exc
    rows: list[Mapping[str, Any]] = []
    for line in text.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{artifact_name} contains invalid JSON") from exc
        if not isinstance(value, Mapping) or line != canonical_json_text(value):
            raise ValueError(f"{artifact_name} must use canonical JSON object rows")
        rows.append(value)
    return rows


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return ("".join(canonical_json_text(row) + "\n" for row in rows)).encode("utf-8")


def _require_exact_fields(
    row: Mapping[str, Any], fields: set[str] | frozenset[str], *, record: str
) -> None:
    if set(row) != set(fields):
        raise ValueError(f"{record} fields differ: {sorted(set(row) ^ set(fields))}")


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _require_nonempty(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _require_int(value: Any, *, field: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
