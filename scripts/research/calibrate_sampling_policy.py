#!/usr/bin/env python3
"""Select sampling temperature from canonical executed calibration bundles only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.analysis.spatial_scope_history.calibration import (  # noqa: E402
    CALIBRATION_TERMINAL_BUNDLE_COLLECTION_SCHEMA_VERSION,
    CalibrationTerminalBundle,
    load_attested_sampling_policy_set,
    load_calibration_backend_attestation_binding,
    load_canonical_json,
    select_sampling_calibration_from_terminal_bundles,
    write_immutable_json,
)
from src.analysis.spatial_scope_history.cohort_ledger import (  # noqa: E402
    CohortLedger,
    sha256_file,
)
from src.analysis.spatial_scope_history.metrics import ReferenceObject  # noqa: E402
from src.data import load_raw_examples  # noqa: E402
from src.data.geometry import coord_bins_to_pixel_xyxy  # noqa: E402
from src.eval.detection_categories import (  # noqa: E402
    COCO_80_CATEGORY_NAMESPACE_SHA256,
    COCO_80_EVALUATOR_LOCAL_CATEGORY_ID_BY_NAME,
    COCO_80_OFFICIAL_CATEGORY_ID_BY_NAME,
    normalize_coco_category_name,
)


def _load_terminal_bundles(
    path: Path,
    *,
    calibration_cohort: CohortLedger,
    validation_cohort: CohortLedger,
    aggregate_artifact_sha256: str,
    aggregate_payload_fingerprint: str,
    reference_source_sha256: str,
) -> tuple[CalibrationTerminalBundle, ...]:
    payload = load_canonical_json(path.resolve())
    expected_keys = {
        "artifact_role",
        "attestation_aggregate_artifact_sha256",
        "attestation_aggregate_payload_fingerprint",
        "calibration_cohort_sha256",
        "calibration_reference_source_sha256",
        "metric_eligible",
        "schema_version",
        "terminal_bundles",
        "validation_cohort_sha256",
    }
    if set(payload) != expected_keys:
        raise RuntimeError("calibration terminal collection has an unsupported schema")
    expected = {
        "artifact_role": "canonical executed non-metric calibration terminal bundles",
        "attestation_aggregate_artifact_sha256": aggregate_artifact_sha256,
        "attestation_aggregate_payload_fingerprint": aggregate_payload_fingerprint,
        "calibration_cohort_sha256": calibration_cohort.fingerprint,
        "calibration_reference_source_sha256": reference_source_sha256,
        "metric_eligible": False,
        "schema_version": CALIBRATION_TERMINAL_BUNDLE_COLLECTION_SCHEMA_VERSION,
        "validation_cohort_sha256": validation_cohort.fingerprint,
    }
    mismatches = {
        field: {"expected": value, "observed": payload.get(field)}
        for field, value in expected.items()
        if payload.get(field) != value
    }
    rows = payload.get("terminal_bundles")
    if mismatches or not isinstance(rows, list):
        raise RuntimeError(
            f"calibration terminal collection identity mismatch: {mismatches}"
        )
    return tuple(CalibrationTerminalBundle.from_artifact_dict(row) for row in rows)


def _load_official_references(
    *,
    source_path: Path,
    cohort: CohortLedger,
) -> dict[int, tuple[ReferenceObject, ...]]:
    source = source_path.expanduser().resolve()
    source_sha256 = sha256_file(source)
    if {row.source_dataset_sha256 for row in cohort.records} != {source_sha256}:
        raise RuntimeError(
            "calibration reference source differs from the sealed cohort"
        )
    cohort_by_image_id = {row.image_id: row for row in cohort.records}
    references: dict[int, tuple[ReferenceObject, ...]] = {}
    for example in load_raw_examples(source):
        image_id = example.metadata.get("source", {}).get("image_id")
        if image_id not in cohort_by_image_id:
            continue
        record = cohort_by_image_id[image_id]
        if (
            example.source.row_number != record.source_row_index + 1
            or example.source.row_sha256 != record.source_row_sha256
            or sha256_file(example.image.path) != record.image_sha256
            or example.image.width != record.source_width
            or example.image.height != record.source_height
        ):
            raise RuntimeError(
                f"calibration reference row differs from sealed image {image_id}"
            )
        image_references: list[ReferenceObject] = []
        for object_record in example.objects:
            category = normalize_coco_category_name(object_record.description)
            source_metadata = object_record.metadata.get("source", {})
            image_references.append(
                ReferenceObject(
                    image_id=str(image_id),
                    ledger_scope="official_annotation",
                    reference_id=f"coco-ann:{source_metadata['coco_ann_id']}",
                    normalized_category_name=category,
                    evaluator_category_id=(
                        COCO_80_EVALUATOR_LOCAL_CATEGORY_ID_BY_NAME[category]
                    ),
                    official_coco_category_id=(
                        COCO_80_OFFICIAL_CATEGORY_ID_BY_NAME[category]
                    ),
                    category_namespace_sha256=COCO_80_CATEGORY_NAMESPACE_SHA256,
                    source_canvas_bbox_xyxy=tuple(
                        float(value)
                        for value in coord_bins_to_pixel_xyxy(
                            object_record.bbox,
                            image_width=record.source_width,
                            image_height=record.source_height,
                            field="calibration_reference.coordinate_bins",
                        )
                    ),
                    state="accepted",
                    provenance="official_annotation",
                )
            )
        references[image_id] = tuple(image_references)
    if set(references) != set(cohort_by_image_id):
        raise RuntimeError("calibration reference source omits sealed images")
    return references


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-cohort-manifest", type=Path, required=True)
    parser.add_argument("--validation-cohort-manifest", type=Path, required=True)
    parser.add_argument(
        "--sampled-runtime-attestation-aggregate", type=Path, required=True
    )
    parser.add_argument("--calibration-reference-source", type=Path, required=True)
    parser.add_argument("--terminal-bundle-collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    calibration_cohort = CohortLedger.from_jsonl_bytes(
        arguments.calibration_cohort_manifest.read_bytes()
    )
    validation_cohort = CohortLedger.from_jsonl_bytes(
        arguments.validation_cohort_manifest.read_bytes()
    )
    aggregate_path = arguments.sampled_runtime_attestation_aggregate.resolve()
    policy_set = load_attested_sampling_policy_set(aggregate_path)
    backend_attestation = load_calibration_backend_attestation_binding(aggregate_path)
    references = _load_official_references(
        source_path=arguments.calibration_reference_source,
        cohort=calibration_cohort,
    )
    bundles = _load_terminal_bundles(
        arguments.terminal_bundle_collection,
        calibration_cohort=calibration_cohort,
        validation_cohort=validation_cohort,
        aggregate_artifact_sha256=policy_set.aggregate_artifact_sha256,
        aggregate_payload_fingerprint=policy_set.aggregate_payload_fingerprint,
        reference_source_sha256=sha256_file(arguments.calibration_reference_source),
    )
    receipt = select_sampling_calibration_from_terminal_bundles(
        calibration_cohort=calibration_cohort,
        validation_cohort=validation_cohort,
        attested_policy_set=policy_set,
        backend_attestation=backend_attestation,
        terminal_bundles=bundles,
        official_reference_objects_by_image_id=references,
    )
    write_immutable_json(arguments.output, receipt.to_artifact_dict())
    print(json.dumps(receipt.to_artifact_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
