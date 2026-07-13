from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest

from scripts.research.materialize_spatial_scope_history_cohorts import (
    CALIBRATION_DENSE_IMAGE_IDS,
    CALIBRATION_SPARSE_IMAGE_IDS,
    CohortMaterializationExpectations,
    materialize_annotation_derived_cohorts,
)
from src.analysis.spatial_scope_history.cohort_ledger import (
    CohortLedger,
    canonical_json_text,
)
from src.eval.detection_categories import (
    COCO_80_CATEGORY_NAMESPACE_ENTRIES,
    COCO_80_CATEGORY_NAMESPACE_SHA256,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _annotation(
    *,
    annotation_id: int,
    image_id: int,
    category_id: int = 2,
    iscrowd: int = 0,
    bbox: list[float] | None = None,
) -> dict[str, object]:
    return {
        "area": 100.0,
        "bbox": bbox or [1.0, 2.0, 10.0, 10.0],
        "category_id": category_id,
        "id": annotation_id,
        "image_id": image_id,
        "iscrowd": iscrowd,
        "segmentation": [],
    }


def _make_fixture(tmp_path: Path) -> dict[str, object]:
    image_root = tmp_path / "images"
    image_root.mkdir()
    validation_ids = [1, 2, 3, 4]
    all_ids = sorted(
        validation_ids
        + list(CALIBRATION_DENSE_IMAGE_IDS)
        + list(CALIBRATION_SPARSE_IMAGE_IDS)
    )
    rows_by_id: dict[int, dict[str, object]] = {}
    raw_images: list[dict[str, object]] = []
    raw_annotations: list[dict[str, object]] = []
    next_annotation_id = 1
    for image_id in all_ids:
        filename = f"{image_id:012d}.jpg"
        Image.new("RGB", (32, 32), color=(image_id % 255, 20, 30)).save(
            image_root / filename
        )
        rows_by_id[image_id] = {
            "height": 32,
            "image_id": image_id,
            "images": [filename],
            # Deliberately constant: Dense-Union membership must not use processed count.
            "objects": [
                {
                    "bbox_2d": [
                        "<|coord_1|>",
                        "<|coord_1|>",
                        "<|coord_2|>",
                        "<|coord_2|>",
                    ],
                    "category_id": 2,
                    "category_name": "bicycle",
                    "coco_ann_id": next_annotation_id,
                    "desc": "bicycle",
                }
            ],
            "width": 32,
        }
        raw_images.append(
            {"file_name": filename, "height": 64, "id": image_id, "width": 64}
        )
        if image_id in {1, 3} or image_id in CALIBRATION_DENSE_IMAGE_IDS:
            count = 12
        else:
            count = 1
        for _ in range(count):
            raw_annotations.append(
                _annotation(annotation_id=next_annotation_id, image_id=image_id)
            )
            next_annotation_id += 1
    raw_annotations.append(
        _annotation(
            annotation_id=next_annotation_id,
            image_id=1,
            category_id=1,
            iscrowd=1,
            bbox=[-2.0, -4.0, 70.0, 80.0],
        )
    )

    model_path = tmp_path / "model.jsonl"
    complete_path = tmp_path / "complete.jsonl"
    raw_path = tmp_path / "raw.json"
    reviewer_packet_path = tmp_path / "reviewer-instructions.md"
    ontology_path = tmp_path / "coco-80-ontology.json"
    _write_jsonl(model_path, [rows_by_id[image_id] for image_id in validation_ids])
    _write_jsonl(complete_path, [rows_by_id[image_id] for image_id in all_ids])
    raw_path.write_text(
        json.dumps(
            {
                "annotations": raw_annotations,
                "categories": [
                    {
                        "id": entry.official_coco_category_id,
                        "name": entry.normalized_category_name,
                    }
                    for entry in COCO_80_CATEGORY_NAMESPACE_ENTRIES
                ],
                "images": raw_images,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    reviewer_packet_path.write_text(
        "# Dense-Union-51 image-only review instructions\n",
        encoding="utf-8",
    )
    ontology_path.write_text(
        canonical_json_text(
            {
                "full_name": "Common Objects in Context 80-category ontology",
                "categories": [
                    {
                        "normalized_category_name": entry.normalized_category_name,
                        "official_coco_category_id": entry.official_coco_category_id,
                    }
                    for entry in COCO_80_CATEGORY_NAMESPACE_ENTRIES
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dense_ids = (1, 3)
    dense_digest = hashlib.sha256(
        "".join(f"{image_id}\n" for image_id in dense_ids).encode("utf-8")
    ).hexdigest()
    calibration_digest = hashlib.sha256(
        "".join(
            [f"dense:{image_id}\n" for image_id in CALIBRATION_DENSE_IMAGE_IDS]
            + [f"sparse:{image_id}\n" for image_id in CALIBRATION_SPARSE_IMAGE_IDS]
        ).encode("utf-8")
    ).hexdigest()
    expectations = CohortMaterializationExpectations(
        model_facing_validation_sha256=_sha256(model_path),
        complete_validation_sha256=_sha256(complete_path),
        raw_official_annotations_sha256=_sha256(raw_path),
        validation_image_count=4,
        complete_validation_image_count=len(all_ids),
        dense_image_count=2,
        dense_noncrowd_annotation_count=24,
        dense_person_annotation_count=0,
        dense_food_and_tableware_annotation_count=0,
        dense_crowd_annotation_count=1,
        dense_ordered_image_identifiers_sha256=dense_digest,
        calibration_ordered_stratum_identifiers_sha256=calibration_digest,
    )
    return {
        "complete_validation_jsonl": complete_path,
        "complete_validation_image_root": image_root,
        "expectations": expectations,
        "model_facing_image_root": image_root,
        "model_facing_validation_jsonl": model_path,
        "raw_official_annotations_json": raw_path,
        "reviewer_instruction_packet_file": reviewer_packet_path,
        "reviewer_instruction_packet_sha256": _sha256(reviewer_packet_path),
        "ontology_file": ontology_path,
        "ontology_sha256": _sha256(ontology_path),
    }


def test_materialization_is_deterministic_raw_annotation_derived_and_sealed(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    first = tmp_path / "first-cohorts"
    second = tmp_path / "second-cohorts"

    summary = materialize_annotation_derived_cohorts(output_root=first, **fixture)
    materialize_annotation_derived_cohorts(output_root=second, **fixture)

    assert summary["counts"]["dense_union_images"] == 2
    assert sorted(path.name for path in first.iterdir()) == sorted(
        path.name for path in second.iterdir()
    )
    for first_path in first.iterdir():
        assert first_path.read_bytes() == (second / first_path.name).read_bytes()

    dense = CohortLedger.from_jsonl_bytes(
        (first / "dense-union-51-manifest.jsonl").read_bytes()
    )
    assert [record.image_id for record in dense.records] == [1, 3]
    assert all(record.noncrowd_annotated_object_count == 12 for record in dense.records)
    individual_rows = [
        json.loads(line)
        for line in (first / "official-individual-ledger.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    crowd_rows = [
        json.loads(line)
        for line in (first / "official-crowd-ignore-ledger.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(individual_rows) == 24
    assert len(crowd_rows) == 1
    assert set(individual_rows[0]) == {
        "annotation_id",
        "annotation_source_sha256",
        "coco_80_category_namespace_sha256",
        "evaluator_local_category_id",
        "geometry",
        "image_file_name",
        "image_id",
        "iscrowd",
        "object_or_region_identifier",
        "normalized_category_name",
        "official_coco_category_id",
        "operational_status",
        "raw_official_category_name",
        "schema_version",
        "source_image_sha256",
    }
    assert {row["iscrowd"] for row in individual_rows} == {0}
    assert crowd_rows[0]["iscrowd"] == 1
    assert crowd_rows[0]["operational_status"] == "ignore-crowd-region"
    assert crowd_rows[0]["geometry"]["unrounded_source_corners_xyxy"] == [
        -1.0,
        -2.0,
        34.0,
        38.0,
    ]
    assert crowd_rows[0]["geometry"]["clipped_source_corners_xyxy"] == [
        0.0,
        0.0,
        32.0,
        32.0,
    ]
    assert "category_id" not in individual_rows[0]
    assert individual_rows[0]["official_coco_category_id"] == 2
    assert individual_rows[0]["evaluator_local_category_id"] == 2
    assert individual_rows[0]["coco_80_category_namespace_sha256"] == (
        COCO_80_CATEGORY_NAMESPACE_SHA256
    )
    geometry = individual_rows[0]["geometry"]
    assert set(geometry) == {
        "arithmetic_precision",
        "clipped_source_corners_xyxy",
        "horizontal_scale_factor",
        "raw_bounding_box_format",
        "raw_bounding_box_xywh",
        "raw_corners_xyxy",
        "raw_image_height",
        "raw_image_width",
        "round_trip_receipt",
        "source_bounding_box_format",
        "source_image_height",
        "source_image_width",
        "unrounded_source_corners_xyxy",
        "vertical_scale_factor",
        "visualization_integer_rounding_applied",
    }
    assert geometry["horizontal_scale_factor"] == 0.5
    assert geometry["vertical_scale_factor"] == 0.5
    assert geometry["raw_corners_xyxy"] == [1.0, 2.0, 11.0, 12.0]
    assert geometry["unrounded_source_corners_xyxy"] == [0.5, 1.0, 5.5, 6.0]
    assert geometry["clipped_source_corners_xyxy"] == [0.5, 1.0, 5.5, 6.0]
    assert geometry["round_trip_receipt"]["passed"] is True
    assert geometry["visualization_integer_rounding_applied"] is False

    review_queue_text = (first / "review-queue.jsonl").read_text(encoding="utf-8")
    review_queue = [json.loads(line) for line in review_queue_text.splitlines()]
    assert len(review_queue) == 4
    assert set(review_queue[0]) == {
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
    assert {row["reviewer_role_slot"]["role_identifier"] for row in review_queue} == {
        "reviewer-one",
        "reviewer-two",
    }
    assert "bounding_box" not in review_queue_text
    assert "prediction" not in review_queue_text
    category_registry = json.loads(
        (first / "coco-80-category-namespace.json").read_text(encoding="utf-8")
    )
    assert category_registry[1] == {
        "evaluator_category_id": 2,
        "normalized_category_name": "bicycle",
        "official_coco_category_id": 2,
    }
    assert 12 not in {entry["official_coco_category_id"] for entry in category_registry}
    assert category_registry[11]["official_coco_category_id"] == 13
    assert _sha256(first / "coco-80-category-namespace.json") == (
        COCO_80_CATEGORY_NAMESPACE_SHA256
    )
    seal = json.loads(
        (first / "annotation-derived-cohort-seal.json").read_text(encoding="utf-8")
    )
    assert seal["artifact_role"].endswith("not the final ledger seal")
    assert seal["execution_readiness"] == "blocked"
    assert seal["blocked_on"] == [
        "reviewer-one-labels.jsonl",
        "reviewer-two-labels.jsonl",
        "adjudication.jsonl",
        "audit-augmented-ledger.jsonl",
        "ledger-seal.json",
    ]
    for name, expected_digest in seal["artifact_digests"].items():
        assert _sha256(first / name) == expected_digest


def test_official_ledgers_use_frozen_image_then_lexicographic_identifier_order(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    raw_path = fixture["raw_official_annotations_json"]
    raw_document = json.loads(raw_path.read_text(encoding="utf-8"))

    for replacement_identifier, annotation in enumerate(
        raw_document["annotations"], start=10_000
    ):
        annotation["id"] = replacement_identifier
    image_one_individuals = [
        annotation
        for annotation in raw_document["annotations"]
        if annotation["image_id"] == 1 and annotation["iscrowd"] == 0
    ]
    image_three_individuals = [
        annotation
        for annotation in raw_document["annotations"]
        if annotation["image_id"] == 3 and annotation["iscrowd"] == 0
    ]
    image_one_individuals[0]["id"] = 100
    image_one_individuals[1]["id"] = 20
    image_three_individuals[0]["id"] = 1

    existing_crowd = next(
        annotation
        for annotation in raw_document["annotations"]
        if annotation["image_id"] == 1 and annotation["iscrowd"] == 1
    )
    existing_crowd["id"] = 900_100
    raw_document["annotations"].extend(
        [
            _annotation(
                annotation_id=900_020,
                image_id=1,
                category_id=1,
                iscrowd=1,
            ),
            _annotation(
                annotation_id=5,
                image_id=3,
                category_id=1,
                iscrowd=1,
            ),
        ]
    )
    raw_path.write_text(json.dumps(raw_document, sort_keys=True), encoding="utf-8")
    fixture["expectations"] = CohortMaterializationExpectations(
        **{
            **fixture["expectations"].__dict__,
            "raw_official_annotations_sha256": _sha256(raw_path),
            "dense_crowd_annotation_count": 3,
        }
    )

    output = tmp_path / "ordered-official-ledgers"
    materialize_annotation_derived_cohorts(output_root=output, **fixture)

    individual_rows = [
        json.loads(line)
        for line in (output / "official-individual-ledger.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    crowd_rows = [
        json.loads(line)
        for line in (output / "official-crowd-ignore-ledger.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert [row["image_id"] for row in individual_rows] == [1] * 12 + [3] * 12
    assert [row["image_id"] for row in crowd_rows] == [1, 1, 3]
    for rows in (individual_rows, crowd_rows):
        for image_id in (1, 3):
            identifiers = [
                row["object_or_region_identifier"]
                for row in rows
                if row["image_id"] == image_id
            ]
            assert identifiers == sorted(identifiers)
    individual_identifiers = [
        row["object_or_region_identifier"] for row in individual_rows
    ]
    assert individual_identifiers.index("coco-ann:100") < individual_identifiers.index(
        "coco-ann:20"
    )
    assert crowd_rows[0]["object_or_region_identifier"] == "coco-crowd:900020"
    assert crowd_rows[1]["object_or_region_identifier"] == "coco-crowd:900100"
    assert crowd_rows[2]["object_or_region_identifier"] == "coco-crowd:5"


def test_materialization_fails_closed_on_digest_drift_without_output(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    fixture["expectations"] = CohortMaterializationExpectations(
        **{
            **fixture["expectations"].__dict__,
            "model_facing_validation_sha256": "0" * 64,
        }
    )
    output = tmp_path / "drift-cohorts"

    with pytest.raises(ValueError, match="digest.*drifted"):
        materialize_annotation_derived_cohorts(output_root=output, **fixture)

    assert not output.exists()


def test_materialization_rejects_reviewer_packet_digest_drift(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    fixture["reviewer_instruction_packet_sha256"] = "0" * 64
    output = tmp_path / "packet-drift-cohorts"

    with pytest.raises(ValueError, match="reviewer instruction packet.*drifted"):
        materialize_annotation_derived_cohorts(output_root=output, **fixture)

    assert not output.exists()


def test_materialization_rejects_raw_registry_namespace_drift(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    raw_path = fixture["raw_official_annotations_json"]
    raw_document = json.loads(raw_path.read_text(encoding="utf-8"))
    raw_document["categories"][0]["name"] = "not-person"
    raw_path.write_text(json.dumps(raw_document, sort_keys=True), encoding="utf-8")
    fixture["expectations"] = CohortMaterializationExpectations(
        **{
            **fixture["expectations"].__dict__,
            "raw_official_annotations_sha256": _sha256(raw_path),
        }
    )
    output = tmp_path / "category-drift-cohorts"

    with pytest.raises(ValueError, match="raw official category registry differs"):
        materialize_annotation_derived_cohorts(output_root=output, **fixture)

    assert not output.exists()


def test_materialization_refuses_overwrite_except_named_empty_test_root(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    output = tmp_path / "existing-cohorts"
    output.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        materialize_annotation_derived_cohorts(output_root=output, **fixture)

    test_output = tmp_path / "test-empty-cohorts"
    test_output.mkdir()
    materialize_annotation_derived_cohorts(
        output_root=test_output,
        allow_empty_temporary_test_root=True,
        **fixture,
    )
    assert (test_output / "annotation-derived-cohort-summary.json").is_file()


def test_written_single_document_artifacts_are_canonical_json(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    output = tmp_path / "canonical-cohorts"
    materialize_annotation_derived_cohorts(output_root=output, **fixture)

    for name in (
        "annotation-derived-cohort-summary.json",
        "annotation-derived-cohort-seal.json",
    ):
        text = (output / name).read_text(encoding="utf-8")
        assert text == canonical_json_text(json.loads(text)) + "\n"
