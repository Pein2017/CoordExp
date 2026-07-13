"""Pre-output review validation and deterministic adjudication-queue tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from src.analysis.spatial_scope_history.cohort_ledger import canonical_json_text
from src.analysis.spatial_scope_history import (
    ADJUDICATOR_DECISION_SCHEMA_VERSION,
    assemble_final_review_ledger,
)
from src.analysis.spatial_scope_history.review_ledger import (
    ANNOTATION_COHORT_SCHEMA_VERSION,
    REVIEWER_LABEL_SCHEMA_VERSION,
    REVIEW_PACKET_ID,
    build_adjudication_queue,
)
from src.eval.detection_categories import (
    COCO_80_CATEGORY_NAMESPACE_SHA256,
    COCO_80_EVALUATOR_LOCAL_CATEGORY_ID_BY_NAME,
    COCO_80_OFFICIAL_CATEGORY_ID_BY_NAME,
)


def test_review_validation_and_queue_cover_required_synthetic_cases(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    first = build_adjudication_queue(**fixture)
    second = build_adjudication_queue(**fixture)

    assert first == second
    changed_fixture = dict(fixture)
    changed_labels = _decode_jsonl(fixture["reviewer_one_labels_jsonl"])
    changed_labels[0]["reason_code"] = "occluded_but_boxable"
    changed_fixture["reviewer_one_labels_jsonl"] = _jsonl(changed_labels)
    assert build_adjudication_queue(**changed_fixture) != first
    rows = [json.loads(line) for line in first.decode("utf-8").splitlines()]
    reviewer_groups = [row for row in rows if row["group_kind"] == "reviewer-proposal"]
    assert all(
        row["review_queue_sha256"] == fixture["expected_review_queue_sha256"]
        for row in rows
    )
    assert all(
        set(row["reviewer_label_sha256_by_role"]) == {"reviewer-one", "reviewer-two"}
        for row in rows
    )
    assert any(
        len(row["linked_reviewer_label_identifiers"]) == 2 for row in reviewer_groups
    )
    assert any(
        len(row["linked_reviewer_label_identifiers"]) == 1 for row in reviewer_groups
    )
    disagreement_groups = [row for row in reviewer_groups if row["image_id"] == 3]
    assert {tuple(row["candidate_category_names"]) for row in disagreement_groups} == {
        ("cat",),
        ("dog",),
    }
    assert any("person" in row["candidate_category_names"] for row in reviewer_groups)
    assert any("chair" in row["candidate_category_names"] for row in reviewer_groups)
    assert any(
        row["linked_official_object_identifiers"] == ["coco-ann:6001"]
        for row in reviewer_groups
    )
    assert any(
        row["group_kind"] == "official-crowd-region"
        and row["linked_official_crowd_region_identifiers"] == ["coco-crowd:5001"]
        for row in rows
    )
    cardinality_groups = [row for row in reviewer_groups if row["image_id"] == 7]
    assert {
        tuple(row["linked_reviewer_label_identifiers"]) for row in cardinality_groups
    } == {
        ("reviewer-one:7:0001", "reviewer-two:7:0001"),
        ("reviewer-one:7:0002", "reviewer-two:7:0002"),
    }
    assert all(
        row["reviewer_pair_intersection_over_union"] is None
        for row in reviewer_groups
        if len(row["linked_reviewer_label_identifiers"]) == 1
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("cross-role", "cross-role"),
        ("official-id-swap", "official category identifier"),
        ("duplicate-label", "duplicate reviewer label identifier"),
        ("bare-category-id", "fields differ"),
        ("noncanonical", "canonical JSON"),
    ],
)
def test_review_validation_fails_closed(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    fixture = _fixture(tmp_path)
    rows = _decode_jsonl(fixture["reviewer_one_labels_jsonl"])
    if mutation == "cross-role":
        rows[0]["reviewer_role_identifier"] = "reviewer-two"
    elif mutation == "official-id-swap":
        rows[0]["official_coco_category_id"] = (
            COCO_80_EVALUATOR_LOCAL_CATEGORY_ID_BY_NAME[
                rows[0]["normalized_category_name"]
            ]
        )
        if (
            rows[0]["official_coco_category_id"]
            == COCO_80_OFFICIAL_CATEGORY_ID_BY_NAME[rows[0]["normalized_category_name"]]
        ):
            rows[0]["normalized_category_name"] = "stop sign"
            rows[0]["official_coco_category_id"] = 12
    elif mutation == "duplicate-label":
        rows.insert(1, dict(rows[0]))
    elif mutation == "bare-category-id":
        rows[0]["category_id"] = rows[0]["official_coco_category_id"]
    fixture["reviewer_one_labels_jsonl"] = _jsonl(rows)
    if mutation == "noncanonical":
        fixture["reviewer_one_labels_jsonl"] = fixture[
            "reviewer_one_labels_jsonl"
        ].replace(b'"candidate_categories":[]', b'"candidate_categories": [ ]', 1)

    with pytest.raises(ValueError, match=message):
        build_adjudication_queue(**fixture)


def test_review_validation_rejects_missing_queue_image(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    queue = _decode_jsonl(fixture["review_queue_jsonl"])
    Path(queue[0]["image_path"]).unlink()

    with pytest.raises(ValueError, match="image is missing"):
        build_adjudication_queue(**fixture)


def test_final_ledger_is_deterministic_and_preserves_provenance(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    queue = build_adjudication_queue(**fixture)
    decision_rows = _decode_jsonl(_decision_payload(queue))
    queue_by_id = {row["adjudication_identifier"]: row for row in _decode_jsonl(queue)}
    distinct = next(
        row
        for row in decision_rows
        if queue_by_id[row["adjudication_identifier"]]["image_id"] == 6
        and queue_by_id[row["adjudication_identifier"]]["group_kind"]
        == "reviewer-proposal"
    )
    distinct.update(
        {
            "decision_outcome": "accept-distinct-new-instance",
            "final_normalized_category_name": "person",
            "final_official_coco_category_id": 1,
            "final_source_canvas_box_xyxy": [50, 50, 70, 70],
            "provenance_decision": "adjudicator_override",
            "reason_code": "adjudicator_override",
        }
    )
    decisions = _jsonl(decision_rows)
    receipt = _correction_receipt(fixture)
    inputs = {
        **_final_inputs(fixture, queue, decisions),
        "pre_seal_ordering_correction_receipt_json": receipt,
        "expected_pre_seal_ordering_correction_receipt_sha256": _sha256(receipt),
    }

    first = assemble_final_review_ledger(**inputs)
    second = assemble_final_review_ledger(**inputs)
    later = assemble_final_review_ledger(
        **{
            **inputs,
            "seal_created_at": "2026-07-13T12:00:01Z",
            "earliest_permitted_metric_run_start": "2026-07-13T12:00:01Z",
        }
    )

    assert first == second
    assert first.adjudication_jsonl == later.adjudication_jsonl
    assert first.audit_augmented_ledger_jsonl == later.audit_augmented_ledger_jsonl
    assert first.ledger_seal_json != later.ledger_seal_json
    ledger = _decode_jsonl(first.audit_augmented_ledger_jsonl)
    assert any(row["provenance"] == "official_annotation" for row in ledger)
    assert any(row["provenance"] == "official_crowd_region" for row in ledger)
    assert all(
        row["adjudication_identifier"] is not None
        for row in ledger
        if row["provenance"] in {"official_annotation", "official_crowd_region"}
    )
    assert any(row["provenance"] == "reviewer_agreement" for row in ledger)
    assert any(
        row["decision_outcome"] == "accept-distinct-new-instance" for row in ledger
    )
    image_seven = [
        row
        for row in ledger
        if row["image_id"] == 7 and row["final_state"] == "accepted"
    ]
    assert [row["object_identifier"].rsplit(":", 1)[-1] for row in image_seven] == [
        "0001",
        "0002",
    ]
    assert [row["source_canvas_box_xyxy"][0] for row in image_seven] == [5, 60]
    seal = json.loads(first.ledger_seal_json)
    assert seal["artifact_digests"]["adjudication.jsonl"] == _sha256(decisions)
    assert seal["artifact_digests"]["audit-augmented-ledger.jsonl"] == _sha256(
        first.audit_augmented_ledger_jsonl
    )
    assert seal["counts"]["adjudication_decisions"] == len(_decode_jsonl(queue))
    assert seal["counts"]["audit_ledger_rows"] == len(ledger)
    assert seal["source_digests"][
        "pre_seal_ordering_correction_receipt_json"
    ] == _sha256(receipt)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("orphan", "orphan"),
        ("missing", "missing"),
        ("duplicate", "duplicate"),
        ("namespace", "source digest binding"),
        ("category", "official category identifier"),
    ],
)
def test_final_ledger_rejects_invalid_decisions(
    tmp_path: Path, mutation: str, message: str
) -> None:
    fixture = _fixture(tmp_path)
    queue = build_adjudication_queue(**fixture)
    rows = _decode_jsonl(_decision_payload(queue))
    if mutation == "orphan":
        rows[0]["adjudication_identifier"] = "dense-union-51-adjudication:999:0001"
    elif mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.insert(1, dict(rows[0]))
    elif mutation == "namespace":
        rows[0]["source_digest_binding"]["category_namespace_sha256"] = "f" * 64
    elif mutation == "category":
        target = next(
            row for row in rows if row["decision_outcome"] == "accept-reviewer-proposal"
        )
        target["final_official_coco_category_id"] = 90
    decisions = _jsonl(rows)

    with pytest.raises(ValueError, match=message):
        assemble_final_review_ledger(**_final_inputs(fixture, queue, decisions))


def test_queue_only_cli_binds_sources_and_refuses_overwrite(tmp_path: Path) -> None:
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()
    fixture = _fixture(fixture_root)
    receipt = _correction_receipt(fixture)
    files = {
        "review-queue": fixture["review_queue_jsonl"],
        "reviewer-one-labels": fixture["reviewer_one_labels_jsonl"],
        "reviewer-two-labels": fixture["reviewer_two_labels_jsonl"],
        "official-individual-ledger": fixture["official_individual_ledger_jsonl"],
        "official-crowd-ledger": fixture["official_crowd_ledger_jsonl"],
        "pre-seal-ordering-correction-receipt": receipt,
    }
    command = [
        sys.executable,
        "scripts/research/assemble_spatial_scope_history_review_ledger.py",
        "--build-queue-only",
    ]
    for name, payload in files.items():
        path = tmp_path / name
        path.write_bytes(payload)
        command.extend((f"--{name}", str(path)))
    output = tmp_path / "output"
    command.extend(
        (
            "--expected-review-queue-sha256",
            fixture["expected_review_queue_sha256"],
            "--expected-packet-sha256",
            fixture["expected_packet_sha256"],
            "--expected-ontology-sha256",
            fixture["expected_ontology_sha256"],
            "--expected-pre-seal-ordering-correction-receipt-sha256",
            _sha256(receipt),
            "--output-root",
            str(output),
        )
    )
    subprocess.run(command, check=True)
    queue = (output / "adjudication-queue.jsonl").read_bytes()
    summary = json.loads((output / "adjudication-queue-summary.json").read_bytes())
    assert summary["adjudication_queue_sha256"] == _sha256(queue)
    assert summary["adjudication_queue_row_count"] == len(queue.splitlines())
    second = subprocess.run(command, capture_output=True, text=True)
    assert second.returncode != 0
    assert "refusing to replace" in second.stderr


def _fixture(tmp_path: Path) -> dict[str, Any]:
    packet_sha = "1" * 64
    ontology_sha = "2" * 64
    queue: list[dict[str, Any]] = []
    reviewer_rows: dict[str, list[dict[str, Any]]] = {
        "reviewer-one": [],
        "reviewer-two": [],
    }
    for image_id in range(1, 52):
        image_path = tmp_path / f"{image_id}.bin"
        image_path.write_bytes(f"image-{image_id}".encode("ascii"))
        image_sha = _sha256(image_path.read_bytes())
        for role in ("reviewer-one", "reviewer-two"):
            queue.append(
                {
                    "image_id": image_id,
                    "image_path": str(image_path),
                    "image_sha256": image_sha,
                    "ontology": {
                        "full_name": "Common Objects in Context 80-category ontology",
                        "path": "ontology.json",
                        "sha256": ontology_sha,
                    },
                    "review_identifier": f"dense-union-51-review:{image_id}:{role}",
                    "reviewer_instruction_packet": {
                        "full_name": "Dense-Union-51 image-only reviewer instruction packet",
                        "path": "packet.md",
                        "sha256": packet_sha,
                    },
                    "reviewer_role_slot": {
                        "full_name": role,
                        "operational_meaning": "independent pass",
                        "role_identifier": role,
                    },
                    "schema_version": ANNOTATION_COHORT_SCHEMA_VERSION,
                    "source_image_height": 100,
                    "source_image_width": 100,
                }
            )
        reviewer_rows["reviewer-one"].extend(
            _labels_for_image(image_id, "reviewer-one")
        )
        reviewer_rows["reviewer-two"].extend(
            _labels_for_image(image_id, "reviewer-two")
        )
    official = [_official_row(image_id=6, annotation_id=6001, crowd=False)]
    crowd = [_official_row(image_id=5, annotation_id=5001, crowd=True)]
    queue_bytes = _jsonl(queue)
    return {
        "review_queue_jsonl": queue_bytes,
        "reviewer_one_labels_jsonl": _jsonl(reviewer_rows["reviewer-one"]),
        "reviewer_two_labels_jsonl": _jsonl(reviewer_rows["reviewer-two"]),
        "official_individual_ledger_jsonl": _jsonl(official),
        "official_crowd_ledger_jsonl": _jsonl(crowd),
        "expected_review_queue_sha256": _sha256(queue_bytes),
        "expected_packet_sha256": packet_sha,
        "expected_ontology_sha256": ontology_sha,
    }


def _decision_payload(queue_payload: bytes) -> bytes:
    rows: list[dict[str, Any]] = []
    image_seven_seen = 0
    for queue in _decode_jsonl(queue_payload):
        outcome = "accept-reviewer-proposal"
        state = "accepted"
        name = (
            queue["candidate_category_names"][0]
            if queue["candidate_category_names"]
            else None
        )
        category_id = (
            None if name is None else COCO_80_OFFICIAL_CATEGORY_ID_BY_NAME[name]
        )
        boxes = queue["candidate_source_canvas_boxes"]
        box = None if not boxes else boxes[0]["source_canvas_box_xyxy"]
        candidates: list[dict[str, Any]] = []
        if (
            queue["group_kind"] != "reviewer-proposal"
            or queue["linked_official_object_identifiers"]
            or queue["linked_official_crowd_region_identifiers"]
        ):
            outcome, state, name, category_id, box = (
                "accept-official-only",
                "accepted",
                None,
                None,
                None,
            )
            if queue["group_kind"] == "official-crowd-region":
                state = "crowd"
        elif not queue["candidate_category_names"]:
            outcome, state = "out-of-scope", "out-of-scope"
        elif len(queue["candidate_category_names"]) > 1:
            outcome, state, name, category_id = "ambiguous", "ambiguous", None, None
            candidates = [
                {
                    "normalized_category_name": candidate,
                    "official_coco_category_id": COCO_80_OFFICIAL_CATEGORY_ID_BY_NAME[
                        candidate
                    ],
                }
                for candidate in queue["candidate_category_names"]
            ]
        elif box is None:
            outcome, state = "partial", "partial"
        if queue["image_id"] == 7 and outcome == "accept-reviewer-proposal":
            image_seven_seen += 1
            box = [60, 60, 80, 80] if image_seven_seen == 1 else [5, 5, 25, 25]
        binding = {
            key: queue[key]
            for key in (
                "category_namespace_sha256",
                "ontology_sha256",
                "official_crowd_ledger_sha256",
                "official_individual_ledger_sha256",
                "packet_sha256",
                "review_queue_sha256",
                "reviewer_label_sha256_by_role",
            )
        }
        rows.append(
            {
                "adjudication_group_sha256": _sha256(
                    canonical_json_text(queue).encode("utf-8")
                ),
                "adjudication_identifier": queue["adjudication_identifier"],
                "adjudication_queue_sha256": _sha256(queue_payload),
                "adjudicator_identifier": "independent-adjudicator-one",
                "candidate_categories": candidates,
                "decision_outcome": outcome,
                "final_normalized_category_name": name,
                "final_official_coco_category_id": category_id,
                "final_source_canvas_box_xyxy": box,
                "final_state": state,
                "provenance_decision": "reviewer_agreement",
                "rationale": "Synthetic explicit adjudication decision.",
                "reason_code": "none",
                "schema_version": ADJUDICATOR_DECISION_SCHEMA_VERSION,
                "source_digest_binding": binding,
            }
        )
    return _jsonl(rows)


def _final_inputs(
    fixture: dict[str, Any], queue: bytes, decisions: bytes
) -> dict[str, Any]:
    return {
        **fixture,
        "adjudication_queue_jsonl": queue,
        "adjudicator_decisions_jsonl": decisions,
        "ledger_version": "dense-union-51-audit-v1",
        "seal_created_at": "2026-07-13T12:00:00Z",
        "earliest_permitted_metric_run_start": "2026-07-13T12:00:00Z",
    }


def _correction_receipt(fixture: dict[str, Any]) -> bytes:
    comparisons = []
    entries = {
        "official-individual-ledger.jsonl": (
            "changed_expected",
            "2b5b7356778c3383a499a80c7e383e86b239768f24b3031a3906f2e4e9e2f354",
            _sha256(fixture["official_individual_ledger_jsonl"]),
        ),
        "official-crowd-ignore-ledger.jsonl": (
            "unchanged",
            _sha256(fixture["official_crowd_ledger_jsonl"]),
            _sha256(fixture["official_crowd_ledger_jsonl"]),
        ),
        "review-queue.jsonl": (
            "unchanged",
            fixture["expected_review_queue_sha256"],
            fixture["expected_review_queue_sha256"],
        ),
        "reviewer-one-labels.jsonl": (
            "copied_byte_identical",
            _sha256(fixture["reviewer_one_labels_jsonl"]),
            _sha256(fixture["reviewer_one_labels_jsonl"]),
        ),
        "reviewer-two-labels.jsonl": (
            "copied_byte_identical",
            _sha256(fixture["reviewer_two_labels_jsonl"]),
            _sha256(fixture["reviewer_two_labels_jsonl"]),
        ),
    }
    for name, (classification, old_sha, new_sha) in entries.items():
        comparisons.append(
            {
                "classification": classification,
                "name": name,
                "new_sha256": new_sha,
                "old_sha256": old_sha,
            }
        )
    receipt = {
        "artifact_comparisons": comparisons,
        "operational_status": "pre-seal-ordering-correction-complete",
        "schema_version": "spatial_scope_history.pre_seal_ordering_correction.v1",
        "supersession_policy": (
            "readiness-v1 remains immutable provenance and must not become "
            "metric-bearing; readiness-v2 is the only candidate for final review "
            "ledger sealing."
        ),
    }
    return (canonical_json_text(receipt) + "\n").encode("utf-8")


def _labels_for_image(image_id: int, role: str) -> list[dict[str, Any]]:
    if image_id == 7:
        boxes = (
            ([10, 10, 20, 20], [14, 10, 24, 20])
            if role == "reviewer-one"
            else ([9, 10, 19, 20], [11, 10, 21, 20])
        )
        return [
            _label(image_id, role, box=box, ordinal=ordinal)
            for ordinal, box in enumerate(boxes, start=1)
        ]
    if image_id == 2 and role == "reviewer-two":
        return [
            _label(
                image_id,
                role,
                state="out-of-scope",
                name=None,
                box=[70, 70, 80, 80],
                reason="non_coco80",
            )
        ]
    if image_id == 3:
        return [_label(image_id, role, name="cat" if role == "reviewer-one" else "dog")]
    if image_id == 4 and role == "reviewer-one":
        return [_label(image_id, role, name="person", reason="occluded_but_boxable")]
    if image_id == 4:
        return [
            _label(
                image_id,
                role,
                state="partial",
                name="person",
                box=None,
                reason="boundary_not_reproducible",
            )
        ]
    if image_id == 5:
        return [
            _label(
                image_id,
                role,
                state="crowd",
                name="person",
                reason="instance_not_separable",
            )
        ]
    if image_id == 2:
        return [_label(image_id, role, name="chair", box=[20, 20, 30, 30])]
    return [_label(image_id, role, name="person")]


def _label(
    image_id: int,
    role: str,
    *,
    state: str = "accepted",
    name: str | None = "person",
    box: list[int] | None = None,
    reason: str = "none",
    ordinal: int = 1,
) -> dict[str, Any]:
    if box is None and not (
        state == "partial" and reason == "boundary_not_reproducible"
    ):
        box = [10, 10, 40, 40]
    official_id = None if name is None else COCO_80_OFFICIAL_CATEGORY_ID_BY_NAME[name]
    return {
        "candidate_categories": [],
        "image_id": image_id,
        "normalized_category_name": name,
        "official_coco_category_id": official_id,
        "packet_id": REVIEW_PACKET_ID,
        "reason_code": reason,
        "review_identifier": f"dense-union-51-review:{image_id}:{role}",
        "reviewer_local_object_identifier": f"{role}:{image_id}:{ordinal:04d}",
        "reviewer_role_identifier": role,
        "reviewer_state": state,
        "schema_version": REVIEWER_LABEL_SCHEMA_VERSION,
        "source_canvas_box_xyxy": box,
    }


def _official_row(*, image_id: int, annotation_id: int, crowd: bool) -> dict[str, Any]:
    name = "person"
    return {
        "coco_80_category_namespace_sha256": COCO_80_CATEGORY_NAMESPACE_SHA256,
        "evaluator_local_category_id": COCO_80_EVALUATOR_LOCAL_CATEGORY_ID_BY_NAME[
            name
        ],
        "geometry": {"clipped_source_corners_xyxy": [10.0, 10.0, 40.0, 40.0]},
        "image_id": image_id,
        "iscrowd": int(crowd),
        "normalized_category_name": name,
        "object_or_region_identifier": f"coco-{'crowd' if crowd else 'ann'}:{annotation_id}",
        "official_coco_category_id": COCO_80_OFFICIAL_CATEGORY_ID_BY_NAME[name],
        "source_image_sha256": _sha256(f"image-{image_id}".encode("ascii")),
    }


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return "".join(canonical_json_text(row) + "\n" for row in rows).encode("utf-8")


def _decode_jsonl(payload: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in payload.decode("utf-8").splitlines()]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
