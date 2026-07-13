#!/usr/bin/env python3
"""Materialize annotation-derived cohort inputs for spatial-scope/history research.

This command stops at the pre-review annotation boundary. It does not run a
model, synthesize reviewer labels, adjudicate proposals, create an
audit-augmented ledger, write the final ledger seal, or write metric run roots.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import uuid
from typing import Any

from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.analysis.spatial_scope_history.cohort_ledger import (  # noqa: E402
    CohortImageRecord,
    CohortLedger,
    canonical_json_text,
)
from src.eval.detection_categories import (  # noqa: E402
    COCO_80_CATEGORY_NAMESPACE_SHA256,
    COCO_80_EVALUATOR_LOCAL_CATEGORY_ID_BY_NAME,
    COCO_80_OFFICIAL_CATEGORY_ID_BY_NAME,
    coco_80_category_namespace_payload,
    normalize_coco_category_name,
)


SCHEMA_VERSION = "spatial_scope_history.annotation_derived_cohort_materialization.v1"
VALIDATION_200_COHORT_ID = "validation-200"
DENSE_UNION_51_COHORT_ID = "dense-union-51"
CALIBRATION_12_COHORT_ID = "sampling-calibration-12"
FOOD_AND_TABLEWARE_OFFICIAL_COCO_CATEGORY_IDS = frozenset(
    {44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61}
)
CALIBRATION_DENSE_IMAGE_IDS = (563648, 303713, 529148, 538236, 559099, 276434)
CALIBRATION_SPARSE_IMAGE_IDS = (30494, 269314, 546829, 42889, 79408, 61471)


@dataclass(frozen=True)
class CohortMaterializationExpectations:
    """Frozen source identities and annotation-derived cohort invariants."""

    model_facing_validation_sha256: str
    complete_validation_sha256: str
    raw_official_annotations_sha256: str
    validation_image_count: int
    complete_validation_image_count: int
    dense_image_count: int
    dense_noncrowd_annotation_count: int
    dense_person_annotation_count: int
    dense_food_and_tableware_annotation_count: int
    dense_crowd_annotation_count: int
    dense_ordered_image_identifiers_sha256: str
    calibration_ordered_stratum_identifiers_sha256: str


FROZEN_EXPECTATIONS = CohortMaterializationExpectations(
    model_facing_validation_sha256="9b524a8c20f03758e2e3939703ff35a1e35a1108fb095ac6e7af5a1539c8cfc4",
    complete_validation_sha256="18a3cad3b7ad847ecf39949fe751d963008dbb7707f796c742c3fa23ae3c8e8b",
    raw_official_annotations_sha256="e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f",
    validation_image_count=200,
    complete_validation_image_count=4951,
    dense_image_count=51,
    dense_noncrowd_annotation_count=860,
    dense_person_annotation_count=337,
    dense_food_and_tableware_annotation_count=200,
    dense_crowd_annotation_count=16,
    dense_ordered_image_identifiers_sha256="e7c6b59950cdef93b59638c1899179ee80319daeda712bfc171bd329978675c8",
    calibration_ordered_stratum_identifiers_sha256="fe11eb219531328cbd18189696925caf5b75c616f9ffd4eca2e5226911833d39",
)


@dataclass(frozen=True)
class SourceRow:
    """One source JavaScript Object Notation Lines row with byte identity."""

    image_id: int
    row_index: int
    raw_line: bytes
    payload: Mapping[str, Any]

    @property
    def row_sha256(self) -> str:
        return _sha256_bytes(self.raw_line)


@dataclass(frozen=True)
class RawImageFacts:
    """Official raw-image dimensions and annotations for one image."""

    image: Mapping[str, Any]
    annotations: tuple[Mapping[str, Any], ...]


def materialize_annotation_derived_cohorts(
    *,
    model_facing_validation_jsonl: Path,
    complete_validation_jsonl: Path,
    raw_official_annotations_json: Path,
    model_facing_image_root: Path,
    complete_validation_image_root: Path,
    output_root: Path,
    reviewer_instruction_packet_file: Path,
    reviewer_instruction_packet_sha256: str,
    ontology_file: Path,
    ontology_sha256: str,
    expectations: CohortMaterializationExpectations = FROZEN_EXPECTATIONS,
    allow_empty_temporary_test_root: bool = False,
) -> Mapping[str, Any]:
    """Build and atomically publish deterministic annotation-derived artifacts."""

    inputs = {
        "model_facing_validation_jsonl": _absolute_file(model_facing_validation_jsonl),
        "complete_validation_jsonl": _absolute_file(complete_validation_jsonl),
        "raw_official_annotations_json": _absolute_file(raw_official_annotations_json),
        "reviewer_instruction_packet_file": _absolute_file(
            reviewer_instruction_packet_file
        ),
        "ontology_file": _absolute_file(ontology_file),
    }
    image_roots = {
        "model_facing_image_root": _absolute_directory(model_facing_image_root),
        "complete_validation_image_root": _absolute_directory(
            complete_validation_image_root
        ),
    }
    output_root = _absolute_output_path(output_root)
    _prepare_output_destination(
        output_root,
        allow_empty_temporary_test_root=allow_empty_temporary_test_root,
    )

    observed_source_digests = {
        name: _sha256_file(path) for name, path in inputs.items()
    }
    _require_equal(
        observed_source_digests["model_facing_validation_jsonl"],
        expectations.model_facing_validation_sha256,
        "model-facing validation source Secure Hash Algorithm 256-bit digest",
    )
    _require_equal(
        observed_source_digests["complete_validation_jsonl"],
        expectations.complete_validation_sha256,
        "complete validation source Secure Hash Algorithm 256-bit digest",
    )
    _require_equal(
        observed_source_digests["raw_official_annotations_json"],
        expectations.raw_official_annotations_sha256,
        "raw official annotations Secure Hash Algorithm 256-bit digest",
    )
    _require_sha256_text(
        reviewer_instruction_packet_sha256,
        label="reviewer instruction packet digest",
    )
    _require_sha256_text(ontology_sha256, label="ontology digest")
    _require_equal(
        observed_source_digests["reviewer_instruction_packet_file"],
        reviewer_instruction_packet_sha256,
        "reviewer instruction packet Secure Hash Algorithm 256-bit digest",
    )
    _require_equal(
        observed_source_digests["ontology_file"],
        ontology_sha256,
        "ontology Secure Hash Algorithm 256-bit digest",
    )

    model_rows = _read_source_rows(inputs["model_facing_validation_jsonl"])
    complete_rows = _read_source_rows(inputs["complete_validation_jsonl"])
    _require_equal(
        len(model_rows), expectations.validation_image_count, "Validation-200 row count"
    )
    _require_equal(
        len(complete_rows),
        expectations.complete_validation_image_count,
        "complete validation row count",
    )
    _require_unique_ordered_rows(model_rows, label="Validation-200")
    _require_unique_ordered_rows(complete_rows, label="complete validation")
    _validate_model_facing_rows_against_complete(model_rows, complete_rows)

    raw_document = _read_json_object(inputs["raw_official_annotations_json"])
    raw_by_image, category_names = _index_raw_official_annotations(raw_document)
    _require_equal(len(category_names), 80, "Common Objects in Context category count")
    _validate_raw_category_registry(category_names)

    model_row_by_id = {row.image_id: row for row in model_rows}
    complete_row_by_id = {row.image_id: row for row in complete_rows}
    missing_raw = sorted(
        (set(model_row_by_id) | set(complete_row_by_id)) - set(raw_by_image)
    )
    if missing_raw:
        raise ValueError(
            f"raw official annotations miss image identifiers: {missing_raw}"
        )

    dense_image_ids = tuple(
        row.image_id
        for row in model_rows
        if _is_dense_union_image(raw_by_image[row.image_id].annotations)
    )
    dense_identifier_digest = _sha256_bytes(
        "".join(f"{image_id}\n" for image_id in dense_image_ids).encode("utf-8")
    )
    _require_equal(
        len(dense_image_ids), expectations.dense_image_count, "Dense-Union-51 count"
    )
    _require_equal(
        dense_identifier_digest,
        expectations.dense_ordered_image_identifiers_sha256,
        "Dense-Union-51 ordered image identifier digest",
    )

    calibration_entries = tuple(
        [("dense", image_id) for image_id in CALIBRATION_DENSE_IMAGE_IDS]
        + [("sparse", image_id) for image_id in CALIBRATION_SPARSE_IMAGE_IDS]
    )
    calibration_digest = _sha256_bytes(
        "".join(
            f"{stratum}:{image_id}\n" for stratum, image_id in calibration_entries
        ).encode("utf-8")
    )
    _require_equal(
        calibration_digest,
        expectations.calibration_ordered_stratum_identifiers_sha256,
        "Sampling Calibration-12 ordered stratum and image identifier digest",
    )
    if set(image_id for _, image_id in calibration_entries) & set(model_row_by_id):
        raise ValueError("Sampling Calibration-12 must be outside Validation-200")
    if len({image_id for _, image_id in calibration_entries}) != len(
        calibration_entries
    ):
        raise ValueError("Sampling Calibration-12 contains duplicate image identifiers")
    _validate_calibration_strata(calibration_entries, raw_by_image)

    dense_counts = _aggregate_counts(dense_image_ids, raw_by_image)
    _require_equal(
        dense_counts["noncrowd_annotated_object_count"],
        expectations.dense_noncrowd_annotation_count,
        "Dense-Union-51 non-crowd annotation count",
    )
    _require_equal(
        dense_counts["annotated_person_count"],
        expectations.dense_person_annotation_count,
        "Dense-Union-51 person annotation count",
    )
    _require_equal(
        dense_counts["annotated_food_and_tableware_count"],
        expectations.dense_food_and_tableware_annotation_count,
        "Dense-Union-51 food and tableware annotation count",
    )
    _require_equal(
        dense_counts["source_crowd_annotation_count"],
        expectations.dense_crowd_annotation_count,
        "Dense-Union-51 crowd annotation count",
    )

    validation_ledger = _build_cohort_ledger(
        cohort_id=VALIDATION_200_COHORT_ID,
        full_name="Validation-200",
        operational_meaning=(
            "The fixed first 200 processed validation images ordered by image identifier."
        ),
        ordered_rows=model_rows,
        memberships_by_image={
            image_id: tuple(
                sorted(
                    {
                        VALIDATION_200_COHORT_ID,
                        *(
                            (DENSE_UNION_51_COHORT_ID,)
                            if image_id in set(dense_image_ids)
                            else ()
                        ),
                    }
                )
            )
            for image_id in model_row_by_id
        },
        source_dataset_sha256=observed_source_digests["model_facing_validation_jsonl"],
        raw_annotation_sha256=observed_source_digests["raw_official_annotations_json"],
        image_root=image_roots["model_facing_image_root"],
        raw_by_image=raw_by_image,
    )
    dense_ledger = _build_cohort_ledger(
        cohort_id=DENSE_UNION_51_COHORT_ID,
        full_name="Dense-Union-51 — Annotation-Derived Dense Union of 51 Images",
        operational_meaning=(
            "The exhaustive Validation-200 subset satisfying the frozen raw official "
            "annotation density rule."
        ),
        ordered_rows=tuple(model_row_by_id[image_id] for image_id in dense_image_ids),
        memberships_by_image={
            image_id: (DENSE_UNION_51_COHORT_ID, VALIDATION_200_COHORT_ID)
            for image_id in dense_image_ids
        },
        source_dataset_sha256=observed_source_digests["model_facing_validation_jsonl"],
        raw_annotation_sha256=observed_source_digests["raw_official_annotations_json"],
        image_root=image_roots["model_facing_image_root"],
        raw_by_image=raw_by_image,
    )
    calibration_rows = tuple(
        complete_row_by_id[image_id] for _, image_id in calibration_entries
    )
    calibration_ledger = _build_cohort_ledger(
        cohort_id=CALIBRATION_12_COHORT_ID,
        full_name="Sampling Calibration-12",
        operational_meaning=(
            "Twelve non-metric-bearing processed validation images outside Validation-200, "
            "with six dense and six sparse images."
        ),
        ordered_rows=calibration_rows,
        memberships_by_image={
            image_id: (CALIBRATION_12_COHORT_ID, f"calibration-{stratum}")
            for stratum, image_id in calibration_entries
        },
        source_dataset_sha256=observed_source_digests["complete_validation_jsonl"],
        raw_annotation_sha256=observed_source_digests["raw_official_annotations_json"],
        image_root=image_roots["complete_validation_image_root"],
        raw_by_image=raw_by_image,
    )

    official_individual_rows = _official_annotation_ledger_rows(
        dense_ledger=dense_ledger,
        raw_by_image=raw_by_image,
        category_names=category_names,
        include_crowd=False,
        raw_annotation_sha256=observed_source_digests["raw_official_annotations_json"],
    )
    official_crowd_rows = _official_annotation_ledger_rows(
        dense_ledger=dense_ledger,
        raw_by_image=raw_by_image,
        category_names=category_names,
        include_crowd=True,
        raw_annotation_sha256=observed_source_digests["raw_official_annotations_json"],
    )
    review_queue_rows = _review_queue_rows(
        dense_ledger=dense_ledger,
        reviewer_instruction_packet_file=inputs["reviewer_instruction_packet_file"],
        reviewer_instruction_packet_sha256=reviewer_instruction_packet_sha256,
        ontology_file=inputs["ontology_file"],
        ontology_sha256=ontology_sha256,
    )
    artifacts: dict[str, bytes] = {
        "cohort-manifest.jsonl": validation_ledger.to_jsonl_bytes(),
        "dense-union-51-manifest.jsonl": dense_ledger.to_jsonl_bytes(),
        "sampling-calibration-12-manifest.jsonl": calibration_ledger.to_jsonl_bytes(),
        "official-individual-ledger.jsonl": _canonical_jsonl_bytes(
            official_individual_rows
        ),
        "official-crowd-ignore-ledger.jsonl": _canonical_jsonl_bytes(
            official_crowd_rows
        ),
        "review-queue.jsonl": _canonical_jsonl_bytes(review_queue_rows),
        "coco-80-category-namespace.json": canonical_json_text(
            coco_80_category_namespace_payload()
        ).encode("utf-8"),
    }
    summary = {
        "artifact_role": "annotation-derived cohort materialization summary",
        "calibration_ordered_stratum_identifiers_sha256": calibration_digest,
        "cohort_fingerprints": {
            CALIBRATION_12_COHORT_ID: calibration_ledger.fingerprint,
            DENSE_UNION_51_COHORT_ID: dense_ledger.fingerprint,
            VALIDATION_200_COHORT_ID: validation_ledger.fingerprint,
        },
        "counts": {
            "complete_validation_images": len(complete_rows),
            "dense_union_images": len(dense_image_ids),
            "sampling_calibration_images": len(calibration_entries),
            "validation_200_images": len(model_rows),
            **dense_counts,
        },
        "dense_ordered_image_identifiers_sha256": dense_identifier_digest,
        "coco_80_category_namespace_sha256": COCO_80_CATEGORY_NAMESPACE_SHA256,
        "name_registry": {
            "COCO-80": "Common Objects in Context 80-category ontology",
            "Dense-Union-51": "Annotation-Derived Dense Union of 51 Images",
            "JSON": "JavaScript Object Notation",
            "JSONL": "JavaScript Object Notation Lines",
            "RGB": "Red-Green-Blue color space",
            "SHA-256": "Secure Hash Algorithm 256-bit",
            "Validation-200": "The fixed first 200 processed validation images ordered by image identifier",
            "float32": "32-bit floating-point arithmetic",
            "iscrowd": "Common Objects in Context crowd-region flag",
            "xywh": "Raw width-height bounding-box format",
            "xyxy": "Corner bounding-box format",
        },
        "schema_version": SCHEMA_VERSION,
        "source_digests": observed_source_digests,
    }
    artifacts["annotation-derived-cohort-summary.json"] = _canonical_json_bytes(summary)
    artifact_digests = {
        name: _sha256_bytes(payload) for name, payload in artifacts.items()
    }
    seal = {
        "artifact_digests": artifact_digests,
        "artifact_role": "annotation-derived cohort seal; not the final ledger seal",
        "blocked_on": [
            "reviewer-one-labels.jsonl",
            "reviewer-two-labels.jsonl",
            "adjudication.jsonl",
            "audit-augmented-ledger.jsonl",
            "ledger-seal.json",
        ],
        "execution_readiness": "blocked",
        "materialization_status": "annotation-derived-boundary-complete",
        "schema_version": SCHEMA_VERSION,
        "source_digests": observed_source_digests,
    }
    artifacts["annotation-derived-cohort-seal.json"] = _canonical_json_bytes(seal)
    _publish_atomically(
        output_root,
        artifacts,
        replace_empty_temporary_test_root=allow_empty_temporary_test_root,
    )
    return summary


def _build_cohort_ledger(
    *,
    cohort_id: str,
    full_name: str,
    operational_meaning: str,
    ordered_rows: Sequence[SourceRow],
    memberships_by_image: Mapping[int, tuple[str, ...]],
    source_dataset_sha256: str,
    raw_annotation_sha256: str,
    image_root: Path,
    raw_by_image: Mapping[int, RawImageFacts],
) -> CohortLedger:
    records: list[CohortImageRecord] = []
    for frozen_order, row in enumerate(ordered_rows):
        source_width = _positive_int(row.payload.get("width"), field="source width")
        source_height = _positive_int(row.payload.get("height"), field="source height")
        raw_facts = raw_by_image[row.image_id]
        raw_width = _positive_int(raw_facts.image.get("width"), field="raw width")
        raw_height = _positive_int(raw_facts.image.get("height"), field="raw height")
        image_path = _resolve_image_path(row.payload, image_root=image_root)
        with Image.open(image_path) as image:
            if image.size != (source_width, source_height):
                raise ValueError(
                    f"source image dimensions drifted for {row.image_id}: "
                    f"declared={(source_width, source_height)} observed={image.size}"
                )
        counts = _annotation_counts(raw_facts.annotations)
        records.append(
            CohortImageRecord(
                image_id=row.image_id,
                frozen_order=frozen_order,
                source_row_index=row.row_index,
                image_path=str(image_path),
                image_sha256=_sha256_file(image_path),
                source_width=source_width,
                source_height=source_height,
                raw_width=raw_width,
                raw_height=raw_height,
                source_row_sha256=row.row_sha256,
                source_dataset_sha256=source_dataset_sha256,
                raw_annotation_sha256=raw_annotation_sha256,
                noncrowd_annotated_object_count=counts[
                    "noncrowd_annotated_object_count"
                ],
                annotated_person_count=counts["annotated_person_count"],
                annotated_food_tableware_count=counts[
                    "annotated_food_and_tableware_count"
                ],
                source_crowd_annotation_count=counts["source_crowd_annotation_count"],
                cohort_memberships=tuple(sorted(memberships_by_image[row.image_id])),
                density_tags=_density_tags(counts),
            )
        )
    return CohortLedger(
        cohort_id=cohort_id,
        full_name=full_name,
        operational_meaning=operational_meaning,
        records=tuple(records),
    )


def _read_source_rows(path: Path) -> tuple[SourceRow, ...]:
    rows: list[SourceRow] = []
    for row_index, raw_line in enumerate(path.read_bytes().splitlines(), start=0):
        if not raw_line.strip():
            raise ValueError(
                f"blank JavaScript Object Notation Lines row at index {row_index}"
            )
        payload = json.loads(raw_line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"source row {row_index} is not an object")
        rows.append(
            SourceRow(
                image_id=_image_id_from_row(payload),
                row_index=row_index,
                raw_line=raw_line,
                payload=payload,
            )
        )
    return tuple(rows)


def _validate_model_facing_rows_against_complete(
    model_rows: Sequence[SourceRow], complete_rows: Sequence[SourceRow]
) -> None:
    if tuple(row.image_id for row in model_rows) != tuple(
        row.image_id for row in complete_rows[: len(model_rows)]
    ):
        raise ValueError(
            "Validation-200 image order differs from complete validation prefix"
        )
    for model_row, complete_row in zip(
        model_rows, complete_rows[: len(model_rows)], strict=True
    ):
        if _row_semantic_signature(model_row.payload) != _row_semantic_signature(
            complete_row.payload
        ):
            raise ValueError(
                f"Validation-200 object contents drifted for image {model_row.image_id}"
            )


def _row_semantic_signature(row: Mapping[str, Any]) -> Any:
    return {
        "height": row.get("height"),
        "image_id": _image_id_from_row(row),
        "objects": row.get("objects"),
        "width": row.get("width"),
    }


def _index_raw_official_annotations(
    document: Mapping[str, Any],
) -> tuple[dict[int, RawImageFacts], dict[int, str]]:
    images = document.get("images")
    annotations = document.get("annotations")
    categories = document.get("categories")
    if (
        not isinstance(images, list)
        or not isinstance(annotations, list)
        or not isinstance(categories, list)
    ):
        raise ValueError("raw official annotation document has an invalid structure")
    image_by_id: dict[int, Mapping[str, Any]] = {}
    for image in images:
        if not isinstance(image, Mapping):
            raise ValueError("raw image record is not an object")
        image_id = _nonnegative_int(image.get("id"), field="raw image identifier")
        if image_id in image_by_id:
            raise ValueError(f"duplicate raw image identifier: {image_id}")
        image_by_id[image_id] = image
    category_names: dict[int, str] = {}
    for category in categories:
        if not isinstance(category, Mapping):
            raise ValueError("raw category record is not an object")
        official_coco_category_id = _positive_int(
            category.get("id"),
            field="official Common Objects in Context category identifier",
        )
        name = category.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"invalid category name for identifier {official_coco_category_id}"
            )
        if official_coco_category_id in category_names:
            raise ValueError(
                f"duplicate official category identifier: {official_coco_category_id}"
            )
        category_names[official_coco_category_id] = name
    annotations_by_image: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    seen_annotation_ids: set[int] = set()
    for annotation in annotations:
        if not isinstance(annotation, Mapping):
            raise ValueError("raw annotation record is not an object")
        annotation_id = _positive_int(
            annotation.get("id"), field="annotation identifier"
        )
        if annotation_id in seen_annotation_ids:
            raise ValueError(f"duplicate annotation identifier: {annotation_id}")
        seen_annotation_ids.add(annotation_id)
        image_id = _nonnegative_int(
            annotation.get("image_id"), field="annotation image identifier"
        )
        official_coco_category_id = _positive_int(
            annotation.get("category_id"),
            field="annotation official Common Objects in Context category identifier",
        )
        if (
            image_id not in image_by_id
            or official_coco_category_id not in category_names
        ):
            raise ValueError(f"orphan raw annotation identifier: {annotation_id}")
        annotations_by_image[image_id].append(annotation)
    return (
        {
            image_id: RawImageFacts(
                image=image,
                annotations=tuple(
                    sorted(
                        annotations_by_image[image_id],
                        key=lambda value: int(value["id"]),
                    )
                ),
            )
            for image_id, image in image_by_id.items()
        },
        category_names,
    )


def _validate_raw_category_registry(category_names: Mapping[int, str]) -> None:
    raw_official_by_name: dict[str, int] = {}
    for official_coco_category_id, raw_name in category_names.items():
        normalized_name = normalize_coco_category_name(raw_name)
        if normalized_name in raw_official_by_name:
            raise ValueError(
                f"raw category names collide after normalization: {normalized_name}"
            )
        raw_official_by_name[normalized_name] = official_coco_category_id
    if raw_official_by_name != dict(COCO_80_OFFICIAL_CATEGORY_ID_BY_NAME):
        raise ValueError(
            "raw official category registry differs from the canonical Common Objects "
            "in Context category namespace"
        )


def _official_annotation_ledger_rows(
    *,
    dense_ledger: CohortLedger,
    raw_by_image: Mapping[int, RawImageFacts],
    category_names: Mapping[int, str],
    include_crowd: bool,
    raw_annotation_sha256: str,
) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    frozen_image_order = {
        record.image_id: order for order, record in enumerate(dense_ledger.records)
    }
    for cohort_record in dense_ledger.records:
        facts = raw_by_image[cohort_record.image_id]
        for annotation in facts.annotations:
            iscrowd = int(annotation.get("iscrowd", 0))
            if iscrowd not in {0, 1}:
                raise ValueError(
                    f"invalid Common Objects in Context crowd flag: {iscrowd}"
                )
            if bool(iscrowd) is not include_crowd:
                continue
            official_coco_category_id = int(annotation["category_id"])
            normalized_category_name = normalize_coco_category_name(
                category_names[official_coco_category_id]
            )
            geometry = _raw_width_height_to_source_corners(
                raw_bounding_box=annotation.get("bbox"),
                raw_width=cohort_record.raw_width,
                raw_height=cohort_record.raw_height,
                source_width=cohort_record.source_width,
                source_height=cohort_record.source_height,
            )
            annotation_id = int(annotation["id"])
            rows.append(
                {
                    "annotation_id": annotation_id,
                    "annotation_source_sha256": raw_annotation_sha256,
                    "coco_80_category_namespace_sha256": (
                        COCO_80_CATEGORY_NAMESPACE_SHA256
                    ),
                    "evaluator_local_category_id": (
                        COCO_80_EVALUATOR_LOCAL_CATEGORY_ID_BY_NAME[
                            normalized_category_name
                        ]
                    ),
                    "normalized_category_name": normalized_category_name,
                    "official_coco_category_id": official_coco_category_id,
                    "raw_official_category_name": category_names[
                        official_coco_category_id
                    ],
                    "geometry": geometry,
                    "image_file_name": str(facts.image["file_name"]),
                    "image_id": cohort_record.image_id,
                    "iscrowd": iscrowd,
                    "object_or_region_identifier": (
                        f"coco-crowd:{annotation_id}"
                        if include_crowd
                        else f"coco-ann:{annotation_id}"
                    ),
                    "operational_status": "ignore-crowd-region"
                    if include_crowd
                    else "accepted-official-individual",
                    "schema_version": SCHEMA_VERSION,
                    "source_image_sha256": cohort_record.image_sha256,
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            frozen_image_order[int(row["image_id"])],
            str(row["object_or_region_identifier"]),
        ),
    )


def _raw_width_height_to_source_corners(
    *,
    raw_bounding_box: Any,
    raw_width: int,
    raw_height: int,
    source_width: int,
    source_height: int,
) -> Mapping[str, Any]:
    if not isinstance(raw_bounding_box, list) or len(raw_bounding_box) != 4:
        raise ValueError("raw bounding box must contain x, y, width, and height")
    raw_x, raw_y, raw_box_width, raw_box_height = (
        _float32(value) for value in raw_bounding_box
    )
    if raw_box_width <= 0 or raw_box_height <= 0:
        raise ValueError("raw bounding box width and height must be positive")
    raw_corners = (
        raw_x,
        raw_y,
        _float32(raw_x + raw_box_width),
        _float32(raw_y + raw_box_height),
    )
    horizontal_scale = _float32(source_width / raw_width)
    vertical_scale = _float32(source_height / raw_height)
    unrounded_source_corners = (
        _float32(raw_corners[0] * horizontal_scale),
        _float32(raw_corners[1] * vertical_scale),
        _float32(raw_corners[2] * horizontal_scale),
        _float32(raw_corners[3] * vertical_scale),
    )
    clipped_source_corners = (
        _float32(min(max(unrounded_source_corners[0], 0.0), source_width)),
        _float32(min(max(unrounded_source_corners[1], 0.0), source_height)),
        _float32(min(max(unrounded_source_corners[2], 0.0), source_width)),
        _float32(min(max(unrounded_source_corners[3], 0.0), source_height)),
    )
    round_trip_raw_corners = (
        _float32(unrounded_source_corners[0] / horizontal_scale),
        _float32(unrounded_source_corners[1] / vertical_scale),
        _float32(unrounded_source_corners[2] / horizontal_scale),
        _float32(unrounded_source_corners[3] / vertical_scale),
    )
    max_error = max(
        abs(float(observed) - float(expected))
        for observed, expected in zip(round_trip_raw_corners, raw_corners, strict=True)
    )
    tolerance = 1e-3
    if max_error > tolerance:
        raise ValueError(
            "raw-to-source coordinate round trip exceeded tolerance: "
            f"error={max_error} tolerance={tolerance}"
        )
    return {
        "arithmetic_precision": "32-bit floating-point arithmetic",
        "clipped_source_corners_xyxy": list(clipped_source_corners),
        "horizontal_scale_factor": horizontal_scale,
        "raw_bounding_box_format": "raw-width-height bounding-box format (xywh)",
        "raw_bounding_box_xywh": [
            raw_x,
            raw_y,
            raw_box_width,
            raw_box_height,
        ],
        "raw_corners_xyxy": list(raw_corners),
        "raw_image_height": raw_height,
        "raw_image_width": raw_width,
        "round_trip_receipt": {
            "comparison_corners": "unrounded source corners",
            "max_absolute_error_raw_pixels": max_error,
            "passed": True,
            "recovered_raw_corners_xyxy": list(round_trip_raw_corners),
            "tolerance_raw_pixels": tolerance,
        },
        "source_bounding_box_format": "corner bounding-box format (xyxy)",
        "source_image_height": source_height,
        "source_image_width": source_width,
        "unrounded_source_corners_xyxy": list(unrounded_source_corners),
        "vertical_scale_factor": vertical_scale,
        "visualization_integer_rounding_applied": False,
    }


def _review_queue_rows(
    *,
    dense_ledger: CohortLedger,
    reviewer_instruction_packet_file: Path,
    reviewer_instruction_packet_sha256: str,
    ontology_file: Path,
    ontology_sha256: str,
) -> list[Mapping[str, Any]]:
    role_slots = [
        {
            "full_name": "Independent Reviewer One",
            "operational_meaning": "First image-only visible-object enumeration pass",
            "role_identifier": "reviewer-one",
        },
        {
            "full_name": "Independent Reviewer Two",
            "operational_meaning": "Second image-only visible-object enumeration pass",
            "role_identifier": "reviewer-two",
        },
    ]
    return [
        {
            "image_id": record.image_id,
            "image_path": record.image_path,
            "image_sha256": record.image_sha256,
            "ontology": {
                "full_name": "Common Objects in Context 80-category ontology",
                "path": str(ontology_file),
                "sha256": ontology_sha256,
            },
            "review_identifier": (
                f"dense-union-51-review:{record.image_id}:{role_slot['role_identifier']}"
            ),
            "reviewer_instruction_packet": {
                "full_name": "Dense-Union-51 image-only reviewer instruction packet",
                "path": str(reviewer_instruction_packet_file),
                "sha256": reviewer_instruction_packet_sha256,
            },
            "reviewer_role_slot": role_slot,
            "schema_version": SCHEMA_VERSION,
            "source_image_height": record.source_height,
            "source_image_width": record.source_width,
        }
        for record in dense_ledger.records
        for role_slot in role_slots
    ]


def _annotation_counts(annotations: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    noncrowd = [
        annotation
        for annotation in annotations
        if int(annotation.get("iscrowd", 0)) == 0
    ]
    return {
        "annotated_food_and_tableware_count": sum(
            int(annotation["category_id"])
            in FOOD_AND_TABLEWARE_OFFICIAL_COCO_CATEGORY_IDS
            for annotation in noncrowd
        ),
        "annotated_person_count": sum(
            int(annotation["category_id"]) == 1 for annotation in noncrowd
        ),
        "noncrowd_annotated_object_count": len(noncrowd),
        "source_crowd_annotation_count": sum(
            int(annotation.get("iscrowd", 0)) == 1 for annotation in annotations
        ),
    }


def _aggregate_counts(
    image_ids: Sequence[int], raw_by_image: Mapping[int, RawImageFacts]
) -> dict[str, int]:
    totals: dict[str, int] = defaultdict(int)
    for image_id in image_ids:
        for name, value in _annotation_counts(
            raw_by_image[image_id].annotations
        ).items():
            totals[name] += value
    return dict(totals)


def _is_dense_union_image(annotations: Sequence[Mapping[str, Any]]) -> bool:
    counts = _annotation_counts(annotations)
    return (
        counts["noncrowd_annotated_object_count"] >= 12
        or counts["annotated_person_count"] >= 8
        or counts["annotated_food_and_tableware_count"] >= 7
    )


def _validate_calibration_strata(
    entries: Sequence[tuple[str, int]], raw_by_image: Mapping[int, RawImageFacts]
) -> None:
    for stratum, image_id in entries:
        if image_id not in raw_by_image:
            raise ValueError(
                f"Sampling Calibration-12 image is absent from raw source: {image_id}"
            )
        annotations = raw_by_image[image_id].annotations
        counts = _annotation_counts(annotations)
        if stratum == "dense" and not _is_dense_union_image(annotations):
            raise ValueError(
                f"dense calibration image does not satisfy dense rule: {image_id}"
            )
        if stratum == "sparse" and not (
            1 <= counts["noncrowd_annotated_object_count"] <= 3
            and counts["source_crowd_annotation_count"] == 0
        ):
            raise ValueError(
                f"sparse calibration image violates sparse rule: {image_id}"
            )


def _density_tags(counts: Mapping[str, int]) -> tuple[str, ...]:
    tags: list[str] = []
    if counts["noncrowd_annotated_object_count"] >= 12:
        tags.append("annotated-count-density")
    if counts["annotated_person_count"] >= 8:
        tags.append("annotated-person-density")
    if counts["annotated_food_and_tableware_count"] >= 7:
        tags.append("annotated-food-and-tableware-density")
    if counts["source_crowd_annotation_count"] > 0:
        tags.append("source-crowd-presence")
    return tuple(sorted(tags))


def _resolve_image_path(row: Mapping[str, Any], *, image_root: Path) -> Path:
    images = row.get("images")
    if (
        not isinstance(images, list)
        or len(images) != 1
        or not isinstance(images[0], str)
    ):
        raise ValueError("source row must contain exactly one image path")
    basename = Path(images[0]).name
    path = (image_root / basename).resolve()
    if not path.is_file():
        raise ValueError(f"source image does not exist: {path}")
    return path


def _image_id_from_row(row: Mapping[str, Any]) -> int:
    if row.get("image_id") is not None:
        return _nonnegative_int(row["image_id"], field="source image identifier")
    images = row.get("images")
    if isinstance(images, list) and len(images) == 1 and isinstance(images[0], str):
        try:
            return int(Path(images[0]).stem)
        except ValueError as exc:
            raise ValueError(
                f"cannot derive image identifier from {images[0]!r}"
            ) from exc
    raise ValueError("source row has no image identifier")


def _require_unique_ordered_rows(rows: Sequence[SourceRow], *, label: str) -> None:
    image_ids = [row.image_id for row in rows]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError(f"{label} contains duplicate image identifiers")
    if image_ids != sorted(image_ids):
        raise ValueError(f"{label} is not ordered by image identifier")


def _read_json_object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(
            f"JavaScript Object Notation document is not an object: {path}"
        )
    return value


def _canonical_jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return ("\n".join(canonical_json_text(row) for row in rows) + "\n").encode("utf-8")


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (canonical_json_text(value) + "\n").encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _absolute_file(path: Path) -> Path:
    if not path.is_absolute():
        raise ValueError(f"input path must be absolute: {path}")
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"input file does not exist: {resolved}")
    return resolved


def _absolute_directory(path: Path) -> Path:
    if not path.is_absolute():
        raise ValueError(f"image root must be absolute: {path}")
    resolved = path.resolve()
    if not resolved.is_dir():
        raise ValueError(f"image root does not exist: {resolved}")
    return resolved


def _absolute_output_path(path: Path) -> Path:
    if not path.is_absolute():
        raise ValueError(f"output root must be absolute: {path}")
    return path.resolve()


def _prepare_output_destination(
    output_root: Path, *, allow_empty_temporary_test_root: bool
) -> None:
    if not output_root.exists():
        return
    if not allow_empty_temporary_test_root:
        raise FileExistsError(
            f"refusing to overwrite existing output root: {output_root}"
        )
    if not (
        output_root.name.startswith("test-") or output_root.name.startswith("tmp-")
    ):
        raise ValueError("temporary test output root must start with 'test-' or 'tmp-'")
    if not output_root.is_dir() or any(output_root.iterdir()):
        raise FileExistsError(f"temporary test output root is not empty: {output_root}")


def _publish_atomically(
    output_root: Path,
    artifacts: Mapping[str, bytes],
    *,
    replace_empty_temporary_test_root: bool,
) -> None:
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.parent / f".{output_root.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        for relative_name, payload in sorted(artifacts.items()):
            path = staging / relative_name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        if output_root.exists() and not replace_empty_temporary_test_root:
            raise FileExistsError(
                f"output root appeared during materialization: {output_root}"
            )
        if output_root.exists():
            output_root.rmdir()
        os.replace(staging, output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _require_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise ValueError(
            f"{label} drifted: expected={expected!r} observed={observed!r}"
        )


def _require_sha256_text(value: str, *, label: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(
            f"{label} must be a lowercase Secure Hash Algorithm 256-bit digest"
        )


def _float32(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"32-bit floating-point value must be numeric: {value!r}")
    result = struct.unpack("!f", struct.pack("!f", float(value)))[0]
    if not (-float("inf") < result < float("inf")):
        raise ValueError(f"32-bit floating-point value must be finite: {value!r}")
    return result


def _nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer: {value!r}")
    return value


def _positive_int(value: Any, *, field: str) -> int:
    result = _nonnegative_int(value, field=field)
    if result == 0:
        raise ValueError(f"{field} must be positive")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-facing-validation-jsonl", type=Path, required=True)
    parser.add_argument("--complete-validation-jsonl", type=Path, required=True)
    parser.add_argument("--raw-official-annotations-json", type=Path, required=True)
    parser.add_argument("--model-facing-image-root", type=Path, required=True)
    parser.add_argument("--complete-validation-image-root", type=Path, required=True)
    parser.add_argument("--reviewer-instruction-packet-file", type=Path, required=True)
    parser.add_argument("--reviewer-instruction-packet-sha256", required=True)
    parser.add_argument("--ontology-file", type=Path, required=True)
    parser.add_argument("--ontology-sha256", required=True)
    parser.add_argument("--output-cohort-root", type=Path, required=True)
    parser.add_argument("--allow-empty-temporary-test-root", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    summary = materialize_annotation_derived_cohorts(
        model_facing_validation_jsonl=arguments.model_facing_validation_jsonl,
        complete_validation_jsonl=arguments.complete_validation_jsonl,
        raw_official_annotations_json=arguments.raw_official_annotations_json,
        model_facing_image_root=arguments.model_facing_image_root,
        complete_validation_image_root=arguments.complete_validation_image_root,
        output_root=arguments.output_cohort_root,
        reviewer_instruction_packet_file=arguments.reviewer_instruction_packet_file,
        reviewer_instruction_packet_sha256=arguments.reviewer_instruction_packet_sha256,
        ontology_file=arguments.ontology_file,
        ontology_sha256=arguments.ontology_sha256,
        allow_empty_temporary_test_root=arguments.allow_empty_temporary_test_root,
    )
    sys.stdout.write(canonical_json_text(summary) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
