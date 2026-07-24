from __future__ import annotations

# The retained version-one negative fixtures intentionally call superseded
# signatures; the active version-two tests below exercise the strict API.
# pyright: reportArgumentType=false, reportCallIssue=false

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from PIL import Image
import pytest

import scripts.research.analyze_trajectory_owner_set_admission_census as census
import scripts.research.assemble_trajectory_owner_set_adjudication_review as review
import scripts.research.select_trajectory_owner_set_adjudication_review_sample as selector
from scripts.research.analyze_individual_trajectory_union_support import match_prefix
from src.inference.backend import token_ids_sha256

canonical_json_text = review.canonical_json_text


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json(value: dict[str, Any]) -> bytes:
    return (canonical_json_text(value) + "\n").encode()


def _pretty_json(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return "".join(canonical_json_text(row) + "\n" for row in rows).encode()


def _decode_jsonl(payload: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in payload.decode().splitlines()]


def _stage_zero_fixture(
    root: Path, *, ordered_possible_image_ids: list[str]
) -> dict[str, Any]:
    root.mkdir()
    impossible_ids: list[str] = []
    possible_pool_path = root / "possible-pool.json"
    ordered_sha256 = review._ordered_image_ids_sha256(ordered_possible_image_ids)
    possible_pool_payload = _pretty_json(
        {
            "schema_version": review.STAGE_ZERO_SCHEMA_VERSION,
            "terminal_status": "completed",
            "population_scope": review.STAGE_ZERO_POPULATION_SCOPE,
            "pool_role": "possible",
            "count": len(ordered_possible_image_ids),
            "ordered_image_ids": ordered_possible_image_ids,
            "ordered_image_ids_sha256": ordered_sha256,
        }
    )
    possible_pool_path.write_bytes(possible_pool_payload)
    impossible_pool_payload = _pretty_json(
        {
            "schema_version": review.STAGE_ZERO_SCHEMA_VERSION,
            "terminal_status": "completed",
            "population_scope": review.STAGE_ZERO_POPULATION_SCOPE,
            "pool_role": "certified_impossible",
            "count": 0,
            "ordered_image_ids": impossible_ids,
            "ordered_image_ids_sha256": review._ordered_image_ids_sha256(
                impossible_ids
            ),
        }
    )
    (root / "impossible-pool.json").write_bytes(impossible_pool_payload)
    category_rows = [
        {"image_id": image_id} for image_id in ordered_possible_image_ids
    ]
    possibility_rows = [
        {"image_id": image_id, "possible": True}
        for image_id in ordered_possible_image_ids
    ]
    (root / "category-state.jsonl").write_bytes(_jsonl(category_rows))
    (root / "possibility-census.jsonl").write_bytes(_jsonl(possibility_rows))
    category_state_sha256 = _sha(canonical_json_text(category_rows).encode())
    possibility_census_sha256 = _sha(
        canonical_json_text(possibility_rows).encode()
    )
    (root / "summary.json").write_bytes(
        _pretty_json(
            {
                "population_count": len(ordered_possible_image_ids),
                "possible_pool_count": len(ordered_possible_image_ids),
                "impossible_pool_count": 0,
                "category_state_sha256": category_state_sha256,
                "possibility_census_sha256": possibility_census_sha256,
            }
        )
    )
    outputs = review._root_inventory(root)[0]
    receipt_path = root / "receipt.json"
    receipt_payload = _pretty_json(
        {
            "schema_version": review.STAGE_ZERO_SCHEMA_VERSION,
            "terminal_status": "completed",
            "population_scope": review.STAGE_ZERO_POPULATION_SCOPE,
            "outputs": outputs,
            "possible_pool_count": len(ordered_possible_image_ids),
            "ordered_possible_image_ids_sha256": ordered_sha256,
            "possible_pool_artifact_sha256": _sha(possible_pool_payload),
            "inputs": {
                "sampled_manifest_binding": {"fixture": "sampled"},
                "source_manifest_binding": {"fixture": "source"},
                "execution_model_identity_sha256": "1" * 64,
                "tokenizer_identity_sha256": "2" * 64,
            },
        }
    )
    receipt_path.write_bytes(receipt_payload)
    root_inventory, root_inventory_sha256 = review._root_inventory(root)
    assert root_inventory["receipt.json"]["sha256"] == _sha(receipt_payload)
    audit_path = root.parent / f"{root.name}-independent-audit.json"
    audit_payload = _pretty_json(
        {
            "schema_version": review.STAGE_ZERO_AUDIT_SCHEMA_VERSION,
            "terminal_status": "completed",
            "population_scope": review.STAGE_ZERO_POPULATION_SCOPE,
            "review_scope": review.STAGE_ZERO_AUDIT_REVIEW_SCOPE,
            "verdict": "approved",
            "stage_zero_receipt_sha256": _sha(receipt_payload),
            "stage_zero_root_inventory_sha256": root_inventory_sha256,
            "population_count": len(ordered_possible_image_ids),
            "possible_pool_count": len(ordered_possible_image_ids),
            "impossible_pool_count": 0,
            "ordered_possible_image_ids_sha256": ordered_sha256,
            "ordered_impossible_image_ids_sha256": (
                review._ordered_image_ids_sha256(impossible_ids)
            ),
            "category_state_sha256": category_state_sha256,
            "possibility_census_sha256": possibility_census_sha256,
            "replayed_possible_witness_count": len(ordered_possible_image_ids),
            "replayed_impossibility_certificate_count": 0,
            "full_certificate_replay": True,
        }
    )
    audit_path.write_bytes(audit_payload)
    return {
        "root": root,
        "root_inventory_sha256": root_inventory_sha256,
        "receipt": receipt_path,
        "receipt_sha256": _sha(receipt_payload),
        "audit": audit_path,
        "audit_sha256": _sha(audit_payload),
        "possible_pool": possible_pool_path,
        "possible_pool_sha256": _sha(possible_pool_payload),
    }


def _ontology_bytes() -> bytes:
    rows = [
        {
            "evaluator_category_id": evaluator_id,
            "normalized_category_name": name,
            "official_coco_category_id": review.COCO_80_OFFICIAL_CATEGORY_ID_BY_NAME[
                name
            ],
        }
        for name, evaluator_id in sorted(
            review.COCO_80_EVALUATOR_LOCAL_CATEGORY_ID_BY_NAME.items(),
            key=lambda item: item[1],
        )
    ]
    return (json.dumps(rows, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _selection(*, packet_sha: str, ontology_sha: str) -> bytes:
    image_ids = [str(index) for index in range(1, 33)]
    frozen = {
        name: {"path": f"/frozen/{name}", "sha256": str(index) * 64}
        for index, name in enumerate(
            (
                "reviewer_packet",
                "ontology_state_artifact",
                "adjudication_replay_implementation",
                "outcome_classifier_contract",
                "unit_contract",
            ),
            start=1,
        )
    }
    frozen["reviewer_packet"]["sha256"] = packet_sha
    frozen["ontology_state_artifact"]["sha256"] = ontology_sha
    frozen["adjudication_replay_implementation"]["sha256"] = _sha(
        Path(review.__file__).read_bytes()
    )
    fraction = {"numerator": 1, "denominator": 40, "decimal": "0.025"}
    value = {
        "schema_version": review.SELECTION_SCHEMA_VERSION,
        "terminal_status": "completed",
        "population_scope": "train_only",
        "randomization_domain": "trajectory-owner-set-adjudication-salvage-v1",
        "population_size_N": 248,
        "null_success_count_K": 248,
        "per_look_alpha": fraction,
        "cumulative_sample_sizes": [16, 32],
        "hypergeometric_cutoffs": [
            {
                "cumulative_sample_size": size,
                "largest_rejection_success_count": cutoff,
                "boundary_lower_tail_probability": {
                    "numerator": 1,
                    "denominator": 100,
                    "decimal": "0.01",
                },
            }
            for size, cutoff in ((16, 0), (32, 2))
        ],
        "ordered_pool_sha256": "a" * 64,
        "ordered_permutation_sha256": "b" * 64,
        "ordered_selected_image_ids": image_ids,
        "look_one_image_ids": image_ids[:16],
        "look_two_image_ids": image_ids[16:],
        "seed": {
            "seed_hex": "c" * 64,
            "seed_byte_count": 32,
            "seed_journal_path": "/frozen/seed-journal.json",
            "seed_journal_sha256": "d" * 64,
            "resumed_existing_journal": False,
            "preseed_binding_sha256": "e" * 64,
        },
        "input_bindings": {
            "stage_zero": {
                "root": "/frozen/stage-zero-v1",
                "receipt_path": "/frozen/stage-zero-v1/receipt.json",
                "receipt_sha256": "f" * 64,
                "audit_path": "/frozen/stage-zero-audit.json",
                "audit_sha256": "1" * 64,
                "possible_pool_path": "/frozen/stage-zero-v1/possible-pool.json",
                "possible_pool_file_sha256": "2" * 64,
                "possible_pool_count": 248,
                "ordered_possible_image_ids_sha256": "a" * 64,
                "independent_audit_verdict": "approved",
            },
            "frozen_review_contracts": frozen,
        },
    }
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()


def _legacy_queue_fixture(
    tmp_path: Path,
) -> tuple[review.ReviewQueueArtifacts, str, str]:
    image_root = tmp_path / "images"
    image_root.mkdir()
    pool_rows = []
    for index in range(1, 33):
        image_path = image_root / f"{index}.png"
        Image.new("RGB", (12, 10), (index, 0, 0)).save(image_path)
        pool_rows.append(
            {
                "image_id": index,
                "images": [str(image_path)],
                "width": 12,
                "height": 10,
                "metadata": {"split": "train"},
                "objects": [
                    {
                        "owner_id": 0,
                        "category": "cat",
                        "category_id": 17,
                        "bbox": [1, 1, 5, 5],
                    }
                ],
            }
        )
    pool = tmp_path / "pool.jsonl"
    pool.write_bytes(_jsonl(pool_rows))
    packet = tmp_path / "packet.md"
    packet.write_text("frozen packet\n", encoding="utf-8")
    ontology = tmp_path / "ontology.json"
    ontology.write_bytes(_ontology_bytes())
    selection = _selection(
        packet_sha=_sha(packet.read_bytes()), ontology_sha=_sha(ontology.read_bytes())
    )
    artifacts = review.build_review_queue(
        selection_json=selection,
        expected_selection_sha256=_sha(selection),
        candidate_pool_path=pool,
        expected_candidate_pool_sha256=_sha(pool.read_bytes()),
        packet_path=packet,
        expected_packet_sha256=_sha(packet.read_bytes()),
        ontology_path=ontology,
        expected_ontology_sha256=_sha(ontology.read_bytes()),
    )
    return artifacts, _sha(packet.read_bytes()), _sha(ontology.read_bytes())


def _label(
    role: str,
    image_id: str,
    ordinal: int,
    *,
    state: str = "accepted",
    reason: str = "none",
    name: str | None = "cat",
    official_id: int | None = 17,
    candidates: list[dict[str, Any]] | None = None,
    box: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "reviewer_local_object_identifier": f"{role}:{image_id}:{ordinal:04d}",
        "normalized_category_name": name,
        "official_coco_category_id": official_id,
        "candidate_categories": candidates or [],
        "source_canvas_box_xyxy": [1, 1, 5, 5] if box is None else box,
        "reviewer_state": state,
        "reason_code": reason,
    }


def _role_artifact(
    queue: bytes, role: str, packet_sha: str, ontology_sha: str
) -> bytes:
    queue_rows = [
        row for row in _decode_jsonl(queue) if row["reviewer_role_identifier"] == role
    ]
    queue_sha = _sha(queue)
    result = []
    for queue_row in queue_rows:
        image_id = queue_row["image_id"]
        labels: list[dict[str, Any]] = []
        if image_id == "1":
            labels = [_label(role, image_id, 1)]
        elif image_id == "2":
            labels = [
                _label(
                    role,
                    image_id,
                    1,
                    name="dog",
                    official_id=18,
                    box=[6, 5, 11, 9],
                )
            ]
        elif image_id == "3" and role == "reviewer-one":
            labels = [
                _label(
                    role,
                    image_id,
                    1,
                    state="ambiguous",
                    reason="category_not_unique",
                    name=None,
                    official_id=None,
                    candidates=[
                        {
                            "normalized_category_name": "cat",
                            "official_coco_category_id": 17,
                        },
                        {
                            "normalized_category_name": "dog",
                            "official_coco_category_id": 18,
                        },
                    ],
                )
            ]
        result.append(
            {
                "schema_version": review.REVIEWER_SCHEMA_BY_ROLE[role],
                "packet_id": review.PACKET_ID,
                "packet_sha256": packet_sha,
                "ontology_sha256": ontology_sha,
                "review_queue_sha256": queue_sha,
                "review_identifier": queue_row["review_identifier"],
                "reviewer_role_identifier": role,
                "image_id": image_id,
                "image_sha256": queue_row["image_sha256"],
                "source_image_width": queue_row["source_image_width"],
                "source_image_height": queue_row["source_image_height"],
                "image_disposition": "complete",
                "labels": labels,
            }
        )
    return _jsonl(result)


def _sealed_review_fixture(
    tmp_path: Path,
) -> tuple[review.ReviewQueueArtifacts, bytes, bytes, bytes, bytes, str, str]:
    queue, packet_sha, ontology_sha = _legacy_queue_fixture(tmp_path)
    one = _role_artifact(
        queue.review_queue_jsonl, "reviewer-one", packet_sha, ontology_sha
    )
    two = _role_artifact(
        queue.review_queue_jsonl, "reviewer-two", packet_sha, ontology_sha
    )
    common = {
        "review_queue_jsonl": queue.review_queue_jsonl,
        "expected_review_queue_sha256": _sha(queue.review_queue_jsonl),
        "expected_packet_sha256": packet_sha,
        "expected_ontology_sha256": ontology_sha,
    }
    one_seal = review.seal_role_artifact(
        role_artifact_jsonl=one,
        reviewer_role_identifier="reviewer-one",
        **common,
    )
    two_seal = review.seal_role_artifact(
        role_artifact_jsonl=two,
        reviewer_role_identifier="reviewer-two",
        **common,
    )
    return queue, one, one_seal, two, two_seal, packet_sha, ontology_sha


def _legacy_test_queue_is_train_only_and_reviewer_visible_fields_are_route_blind(
    tmp_path: Path,
) -> None:
    artifacts, _, _ = _legacy_queue_fixture(tmp_path)
    queue = _decode_jsonl(artifacts.review_queue_jsonl)

    assert len(queue) == 64
    assert [row["reviewer_role_identifier"] for row in queue[:2]] == [
        "reviewer-one",
        "reviewer-two",
    ]
    assert set(queue[0]) == set(review._QUEUE_FIELDS)
    assert not set(queue[0]) & review._FORBIDDEN_REVIEW_EVIDENCE_FIELDS
    assert (
        json.loads(artifacts.manifest_json)["official_owner_ledger_reviewer_visible"]
        is False
    )
    assert len(_decode_jsonl(artifacts.official_owner_ledger_jsonl)) == 32


def _legacy_test_preseed_contract_binds_packet_schema_and_source_hashes(
    tmp_path: Path,
) -> None:
    packet = tmp_path / "packet.md"
    packet.write_text("packet\n", encoding="utf-8")
    ontology = tmp_path / "ontology.json"
    ontology.write_bytes(_ontology_bytes())
    pool = tmp_path / "pool.jsonl"
    pool.write_text('{"image_id":1}\n', encoding="utf-8")

    payload = review.freeze_preseed_contract(
        packet_path=packet,
        expected_packet_sha256=_sha(packet.read_bytes()),
        ontology_path=ontology,
        expected_ontology_sha256=_sha(ontology.read_bytes()),
        candidate_pool_path=pool,
        expected_candidate_pool_sha256=_sha(pool.read_bytes()),
    )

    seal = json.loads(payload)
    assert seal["seed_generated"] is False
    assert seal["selection_materialized"] is False
    assert seal["schema_registry_sha256"] == _sha(
        canonical_json_text(review.SCHEMA_REGISTRY).encode()
    )
    assert seal["source_bindings"]["review_assembler"]["sha256"] == _sha(
        Path(review.__file__).read_bytes()
    )


def _legacy_test_role_validation_accepts_explicit_empty_dispositions_and_fails_closed(
    tmp_path: Path,
) -> None:
    artifacts, packet_sha, ontology_sha = _legacy_queue_fixture(tmp_path)
    one = _role_artifact(
        artifacts.review_queue_jsonl, "reviewer-one", packet_sha, ontology_sha
    )
    rows = review.validate_role_artifact(
        role_artifact_jsonl=one,
        reviewer_role_identifier="reviewer-one",
        review_queue_jsonl=artifacts.review_queue_jsonl,
        expected_review_queue_sha256=_sha(artifacts.review_queue_jsonl),
        expected_packet_sha256=packet_sha,
        expected_ontology_sha256=ontology_sha,
    )
    assert len(rows) == 32
    assert rows[3]["labels"] == []

    missing = _jsonl(_decode_jsonl(one)[:-1])
    with pytest.raises(ValueError, match="complete image disposition"):
        review.validate_role_artifact(
            role_artifact_jsonl=missing,
            reviewer_role_identifier="reviewer-one",
            review_queue_jsonl=artifacts.review_queue_jsonl,
            expected_review_queue_sha256=_sha(artifacts.review_queue_jsonl),
            expected_packet_sha256=packet_sha,
            expected_ontology_sha256=ontology_sha,
        )
    forbidden = _decode_jsonl(one)
    forbidden[0]["labels"][0]["route_id"] = "sample-00"
    with pytest.raises(ValueError, match="forbidden evidence"):
        review.validate_role_artifact(
            role_artifact_jsonl=_jsonl(forbidden),
            reviewer_role_identifier="reviewer-one",
            review_queue_jsonl=artifacts.review_queue_jsonl,
            expected_review_queue_sha256=_sha(artifacts.review_queue_jsonl),
            expected_packet_sha256=packet_sha,
            expected_ontology_sha256=ontology_sha,
        )

    image_path = Path(_decode_jsonl(artifacts.review_queue_jsonl)[0]["image_path"])
    Image.new("RGB", (12, 10), (255, 255, 255)).save(image_path)
    with pytest.raises(ValueError, match="path/hash drift"):
        review.validate_role_artifact(
            role_artifact_jsonl=one,
            reviewer_role_identifier="reviewer-one",
            review_queue_jsonl=artifacts.review_queue_jsonl,
            expected_review_queue_sha256=_sha(artifacts.review_queue_jsonl),
            expected_packet_sha256=packet_sha,
            expected_ontology_sha256=ontology_sha,
        )


def _legacy_test_two_seals_gate_add_only_ledger_and_preserve_uncertainty(
    tmp_path: Path,
) -> None:
    queue, one, one_seal, two, two_seal, packet_sha, ontology_sha = (
        _sealed_review_fixture(tmp_path)
    )
    bad_seal = json.loads(two_seal)
    bad_seal["role_artifact_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="seal binding drift"):
        review.build_adjudication_queue(
            review_queue_jsonl=queue.review_queue_jsonl,
            expected_review_queue_sha256=_sha(queue.review_queue_jsonl),
            reviewer_one_jsonl=one,
            reviewer_one_seal_json=one_seal,
            reviewer_two_jsonl=two,
            reviewer_two_seal_json=_json(bad_seal),
            official_owner_ledger_jsonl=queue.official_owner_ledger_jsonl,
            expected_official_owner_ledger_sha256=_sha(
                queue.official_owner_ledger_jsonl
            ),
            expected_packet_sha256=packet_sha,
            expected_ontology_sha256=ontology_sha,
        )

    adjudication = review.build_adjudication_queue(
        review_queue_jsonl=queue.review_queue_jsonl,
        expected_review_queue_sha256=_sha(queue.review_queue_jsonl),
        reviewer_one_jsonl=one,
        reviewer_one_seal_json=one_seal,
        reviewer_two_jsonl=two,
        reviewer_two_seal_json=two_seal,
        official_owner_ledger_jsonl=queue.official_owner_ledger_jsonl,
        expected_official_owner_ledger_sha256=_sha(queue.official_owner_ledger_jsonl),
        expected_packet_sha256=packet_sha,
        expected_ontology_sha256=ontology_sha,
    )
    decisions = []
    for row in _decode_jsonl(adjudication):
        image_id = row["image_id"]
        labels = {
            label["reviewer_local_object_identifier"]
            for role in review.REVIEWER_ROLES
            for label in row["reviewer_dispositions"][role]["labels"]
        }
        dispositions: list[dict[str, Any]] = []
        complete = True
        reasons: list[str] = []
        if image_id == "1":
            dispositions = [
                {
                    "linked_reviewer_label_identifiers": sorted(labels),
                    "adjudication_state": "official_duplicate",
                    "reason_code": "official_duplicate",
                    "linked_official_owner_ids": [
                        row["official_owners"][0]["owner_id"]
                    ],
                    "final_normalized_category_name": None,
                    "final_official_coco_category_id": None,
                    "final_source_canvas_box_xyxy": None,
                }
            ]
        elif image_id == "2":
            dispositions = [
                {
                    "linked_reviewer_label_identifiers": sorted(labels),
                    "adjudication_state": "accepted_missing_owner",
                    "reason_code": "reviewer_agreement",
                    "linked_official_owner_ids": [],
                    "final_normalized_category_name": "dog",
                    "final_official_coco_category_id": 18,
                    "final_source_canvas_box_xyxy": [6, 5, 11, 9],
                }
            ]
        elif image_id == "3":
            dispositions = [
                {
                    "linked_reviewer_label_identifiers": sorted(labels),
                    "adjudication_state": "ambiguous",
                    "reason_code": "category_not_unique",
                    "linked_official_owner_ids": [],
                    "final_normalized_category_name": None,
                    "final_official_coco_category_id": None,
                    "final_source_canvas_box_xyxy": None,
                }
            ]
            complete = False
            reasons = ["ambiguous_visible_evidence"]
        decisions.append(
            {
                "schema_version": review.ADJUDICATION_DECISION_SCHEMA_VERSION,
                "adjudication_identifier": row["adjudication_identifier"],
                "adjudication_queue_sha256": _sha(adjudication),
                "source_binding_sha256": _sha(canonical_json_text(row).encode()),
                "confirmed_official_owner_ids": sorted(
                    owner["owner_id"] for owner in row["official_owners"]
                ),
                "proposal_dispositions": dispositions,
                "owner_universe_complete": complete,
                "owner_universe_uncertainty_reasons": reasons,
            }
        )
    artifacts = review.assemble_owner_ledger(
        adjudication_queue_jsonl=adjudication,
        expected_adjudication_queue_sha256=_sha(adjudication),
        adjudicator_decisions_jsonl=_jsonl(decisions),
    )
    owners = _decode_jsonl(artifacts.owner_ledger_jsonl)
    official = _decode_jsonl(queue.official_owner_ledger_jsonl)
    assert [
        row for row in owners if row["owner_origin"] == "official_annotation"
    ] == official
    assert [row for row in owners if row["owner_origin"] == "review_addition"] == [
        {
            "schema_version": review.OFFICIAL_OWNER_SCHEMA_VERSION,
            "image_id": "2",
            "image_sha256": next(row for row in official if row["image_id"] == "2")[
                "image_sha256"
            ],
            "owner_id": "review-owner:2:0001",
            "owner_origin": "review_addition",
            "normalized_category_name": "dog",
            "official_coco_category_id": 18,
            "source_canvas_box_xyxy": [6, 5, 11, 9],
            "linked_reviewer_label_identifiers": [
                "reviewer-one:2:0001",
                "reviewer-two:2:0001",
            ],
        }
    ]
    uncertainty = {
        row["image_id"]: row
        for row in _decode_jsonl(artifacts.uncertainty_ledger_jsonl)
    }
    assert uncertainty["3"]["retained_uncertainty_axes"] == list(
        review.UNCERTAINTY_AXES
    )
    assert uncertainty["4"]["retained_uncertainty_axes"] == []


def _route(
    image_id: str, route_id: str, index: int, owners: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    decode = "greedy" if route_id == "source-b16" else "sampled"
    prediction = {
        "image_id": image_id,
        "trajectory_id": route_id,
        "decode_mode": decode,
        "seed": index,
        "generated_row_index": 0,
        "prediction_id": f"{route_id}:row-0",
        "category": "cat",
        "bbox": [0.0, 0.0, 10.0, 10.0],
        "raw": {"description": "cat", "bbox": [0, 0, 10, 10]},
    }
    assignment = match_prefix([prediction], owners, 16)
    assignment.update(
        {
            "malformed_row_count": 0,
            "harmful_row_count": 0,
            "row_counts": {
                "duplicate": 0,
                "malformed": 0,
                "unsupported_hallucination": 0,
                "semantic_error": 0,
                "unresolved": 0,
            },
        }
    )
    token_ids = [index + 10]
    token_hash = token_ids_sha256(token_ids)
    projection = {
        "status": "accepted_natural_end",
        "raw_generated_token_ids_sha256": f"{index + 1:064x}",
        "raw_generated_token_count": 1,
        "projected_token_ids_sha256": token_hash,
        "projected_token_count": 1,
    }
    row = {
        "image_id": image_id,
        "trajectory_id": route_id,
        "decode_mode": decode,
        "sample_index": None if route_id == "source-b16" else index,
        "seed": index,
        "stop_reason": "im_end",
        "generated_token_ids": token_ids,
        "generated_token_ids_sha256": token_hash,
        "_source_b16_provenance"
        if route_id == "source-b16"
        else "_sampled_b16_provenance": projection,
    }
    evidence = {
        "decode_mode": decode,
        "seed": index,
        "stop_reason": "im_end",
        "parser": {
            "parse_status": "accepted",
            "valid_prediction_count": 1,
            "reported_valid_prediction_count": 1,
            "dropped_prediction_count": 0,
            "dropped_predictions": [],
            "malformed_attempt_count": 0,
        },
    }
    return row, evidence, assignment


def _adapter() -> dict[str, Any]:
    image_id = "1"
    owners = [{"owner_id": "1:0", "category": "cat", "bbox": [0.0, 0.0, 10.0, 10.0]}]
    source_rows: dict[tuple[str, int], dict[str, Any]] = {}
    sampled_rows: dict[tuple[str, int], dict[str, Any]] = {}
    evidence: dict[str, dict[str, Any]] = {}
    assignments: dict[str, dict[str, Any]] = {}
    for index, route_id in enumerate(review.EXPECTED_ROUTE_IDS):
        row, route_evidence, assignment = _route(image_id, route_id, index, owners)
        evidence[route_id] = route_evidence
        assignments[route_id] = assignment
        if route_id == "source-b16":
            source_rows[(image_id, 0)] = row
        else:
            sampled_rows[(image_id, int(route_id[-2:]))] = row
    return {
        "source_rows": source_rows,
        "sampled_rows": sampled_rows,
        "reference_records": {
            image_id: {
                "image": {
                    "image_id": 1,
                    "path": "/synthetic/1.png",
                    "width": 10,
                    "height": 10,
                    "content_sha256": "9" * 64,
                }
            }
        },
        "image_results": {
            image_id: {
                "image_id": image_id,
                "trajectory_evidence": evidence,
                "owners": owners,
                "budgets": [
                    {
                        "budget": 16,
                        "trajectory_assignments": assignments,
                        "owner_sets": {
                            route_id: ["1:0"] for route_id in review.EXPECTED_ROUTE_IDS
                        },
                    }
                ],
            }
        },
    }


def _legacy_test_global_replay_noop_exactly_reproduces_census_and_preserves_identities() -> (
    None
):
    adapter = _adapter()
    original = census._analyze_image_candidates(
        "1", census._image_candidates("1", adapter, reverse_input=False)
    )
    owner_payload = _jsonl(
        [
            {
                "schema_version": review.OFFICIAL_OWNER_SCHEMA_VERSION,
                "image_id": "1",
                "image_sha256": "9" * 64,
                "owner_id": "1:0",
                "owner_origin": "official_annotation",
                "normalized_category_name": "cat",
                "official_coco_category_id": 17,
                "source_canvas_box_xyxy": [0.0, 0.0, 10.0, 10.0],
                "linked_reviewer_label_identifiers": [],
            }
        ]
    )
    seal = _json(
        {
            "schema_version": review.OWNER_LEDGER_SEAL_SCHEMA_VERSION,
            "adjudication_queue_sha256": "1" * 64,
            "adjudication_sha256": "2" * 64,
            "owner_ledger_sha256": _sha(owner_payload),
            "uncertainty_ledger_sha256": "3" * 64,
            "selected_image_count": 1,
            "owner_count": 1,
            "official_owner_count": 1,
            "added_owner_count": 0,
            "owner_universe_uncertain_image_count": 0,
            "add_only": True,
            "official_owner_identity_and_box_immutable": True,
            "route_blind": True,
        }
    )

    artifacts = review.replay_owner_ledger(
        adapter=adapter,
        ordered_image_ids=["1"],
        owner_ledger_jsonl=owner_payload,
        owner_ledger_seal_json=seal,
        original_census_records={"1": original},
    )

    replay = _decode_jsonl(artifacts.replay_records_jsonl)[0]
    assert replay["census_record"] == original
    assert replay["route_ids"] == list(review.EXPECTED_ROUTE_IDS)
    assert replay["route_count"] == 17
    assert replay["no_op_exact_reproduction"] is True
    assert (
        json.loads(artifacts.replay_receipt_json)["original_candidate_receipt_reused"]
        is True
    )


def _legacy_test_outcome_classifier_counts_all_retained_uncertainty_as_success() -> (
    None
):
    replay = _jsonl(
        [
            {
                "schema_version": review.REPLAY_RECORD_SCHEMA_VERSION,
                "image_id": image_id,
                "owner_ledger_sha256": "1" * 64,
                "official_owner_count": 1,
                "added_owner_count": 0,
                "route_count": 17,
                "route_ids": list(review.EXPECTED_ROUTE_IDS),
                "route_identity": {
                    route_id: {
                        "candidate_identity_sha256": "2" * 64,
                        "parser_evidence_sha256": "3" * 64,
                    }
                    for route_id in review.EXPECTED_ROUTE_IDS
                },
                "no_op_exact_reproduction": True,
                "census_record": {
                    "admission": {"primary_natural_alias_admitted": image_id == "1"},
                    "candidates": (
                        [
                            {
                                "candidate_id": "sample-00",
                                "unknown_count": 1,
                                "ambiguity_count": 0,
                            }
                        ]
                        if image_id == "3"
                        else []
                    ),
                },
            }
            for image_id in ("1", "2", "3", "4")
        ]
    )
    uncertainty = _jsonl(
        [
            {
                "schema_version": review.UNCERTAINTY_LEDGER_SCHEMA_VERSION,
                "image_id": "1",
                "owner_universe_complete": True,
                "owner_universe_uncertainty_reasons": [],
                "retained_uncertainty_axes": [],
                "unresolved_proposal_dispositions": [],
            },
            {
                "schema_version": review.UNCERTAINTY_LEDGER_SCHEMA_VERSION,
                "image_id": "2",
                "owner_universe_complete": False,
                "owner_universe_uncertainty_reasons": [
                    "ledger_completeness_not_established"
                ],
                "retained_uncertainty_axes": list(review.UNCERTAINTY_AXES),
                "unresolved_proposal_dispositions": [],
            },
            {
                "schema_version": review.UNCERTAINTY_LEDGER_SCHEMA_VERSION,
                "image_id": "3",
                "owner_universe_complete": True,
                "owner_universe_uncertainty_reasons": [],
                "retained_uncertainty_axes": [],
                "unresolved_proposal_dispositions": [],
            },
            {
                "schema_version": review.UNCERTAINTY_LEDGER_SCHEMA_VERSION,
                "image_id": "4",
                "owner_universe_complete": True,
                "owner_universe_uncertainty_reasons": [],
                "retained_uncertainty_axes": [],
                "unresolved_proposal_dispositions": [],
            },
        ]
    )

    outcomes = _decode_jsonl(
        review.classify_outcomes(
            replay_records_jsonl=replay,
            uncertainty_ledger_jsonl=uncertainty,
        )
    )

    assert [row["outcome"] for row in outcomes] == [
        "actual_admission",
        "potential_admission_unresolved",
        "potential_admission_unresolved",
        "definitive_non_admission",
    ]
    assert [row["statistical_success"] for row in outcomes] == [
        True,
        True,
        True,
        False,
    ]
    assert outcomes[2]["uncertain_candidate_ids"] == ["sample-00"]


# Version-two fixtures below replace the early abstract-selection fixtures above.
# Keeping the small role/route constructors shared makes the adversarial cases
# readable while exercising the complete pre-entropy and per-look chain.


def _subset_adapter(
    adapter: dict[str, Any], image_ids: tuple[str, ...] | list[str]
) -> dict[str, Any]:
    selected = set(image_ids)
    return {
        "source_rows": {
            key: value
            for key, value in adapter["source_rows"].items()
            if key[0] in selected
        },
        "sampled_rows": {
            key: value
            for key, value in adapter["sampled_rows"].items()
            if key[0] in selected
        },
        "reference_records": {
            key: value
            for key, value in adapter["reference_records"].items()
            if key in selected
        },
        "image_results": {
            key: value
            for key, value in adapter["image_results"].items()
            if key in selected
        },
    }


def _multi_adapter(image_ids: list[str], image_path: Path) -> dict[str, Any]:
    image_sha = _sha(image_path.read_bytes())
    merged: dict[str, Any] = {
        "source_rows": {},
        "sampled_rows": {},
        "reference_records": {},
        "image_results": {},
    }
    for image_id in image_ids:
        owners = [
            {
                "owner_id": f"{image_id}:0",
                "category": "cat",
                "category_id": 17,
                "bbox": [0.0, 0.0, 10.0, 10.0],
                "annotation_index": 0,
                "image_id": int(image_id),
            }
        ]
        evidence: dict[str, Any] = {}
        assignments: dict[str, Any] = {}
        for index, route_id in enumerate(review.EXPECTED_ROUTE_IDS):
            row, route_evidence, assignment = _route(image_id, route_id, index, owners)
            evidence[route_id] = route_evidence
            assignments[route_id] = assignment
            if route_id == "source-b16":
                merged["source_rows"][(image_id, 0)] = row
            else:
                merged["sampled_rows"][(image_id, int(route_id[-2:]))] = row
        merged["reference_records"][image_id] = {
            "image": {
                "image_id": int(image_id),
                "path": str(image_path.resolve()),
                "width": 12,
                "height": 10,
                "content_sha256": image_sha,
            }
        }
        merged["image_results"][image_id] = {
            "image_id": image_id,
            "trajectory_evidence": evidence,
            "owners": owners,
            "budgets": [
                {
                    "budget": 16,
                    "trajectory_assignments": assignments,
                    "owner_sets": {
                        route_id: [f"{image_id}:0"]
                        for route_id in review.EXPECTED_ROUTE_IDS
                    },
                }
            ],
        }
    return merged


def _make_experiment(tmp_path: Path) -> dict[str, Any]:
    image = tmp_path / "source.png"
    Image.new("RGB", (12, 10), (40, 50, 60)).save(image)
    source_panel = tmp_path / "source-panel"
    sampled_panel = tmp_path / "sampled-panel"
    source_panel.mkdir()
    sampled_panel.mkdir()
    (source_panel / "manifest.json").write_text("source\n", encoding="utf-8")
    (sampled_panel / "manifest.json").write_text("sampled\n", encoding="utf-8")
    packet = tmp_path / "packet.md"
    packet.write_text("frozen packet\n", encoding="utf-8")
    ontology = tmp_path / "ontology.json"
    ontology.write_bytes(_ontology_bytes())

    pool_ids = [str(index) for index in range(1, 249)]
    entropy_integer = 123456
    permutation_count = review._falling_factorial(
        len(pool_ids), review.SELECTION_COUNT
    )
    rank = entropy_integer % permutation_count
    remaining = pool_ids.copy()
    trace: list[dict[str, Any]] = []
    selected_ids: list[str] = []
    for position in range(review.SELECTION_COUNT):
        suffix_count = review._falling_factorial(
            len(pool_ids) - position - 1,
            review.SELECTION_COUNT - position - 1,
        )
        selected_index = rank // suffix_count
        next_rank = rank % suffix_count
        selected_id = remaining.pop(selected_index)
        trace.append(
            {
                "position": position,
                "remaining_count": len(remaining) + 1,
                "rank_before_decimal": str(rank),
                "suffix_count_decimal": str(suffix_count),
                "selected_remaining_index": selected_index,
                "selected_image_id": selected_id,
                "rank_after_decimal": str(next_rank),
            }
        )
        selected_ids.append(selected_id)
        rank = next_rank
    assert rank == 0
    adapter = _multi_adapter(selected_ids, image)

    pool_rows = [
        {
            "image_id": int(image_id),
            "images": [str(image.resolve())],
            "width": 12,
            "height": 10,
            "metadata": {"split": "train"},
            "objects": [
                {
                    "owner_id": 0,
                    "category": "cat",
                    "category_id": 17,
                    "bbox": [0.0, 0.0, 10.0, 10.0],
                }
            ],
        }
        for image_id in pool_ids
    ]
    pool = tmp_path / "candidate-pool.jsonl"
    pool.write_bytes(_jsonl(pool_rows))
    original_census: dict[str, dict[str, Any]] = {}
    for image_id in selected_ids:
        original_census[image_id] = census._analyze_image_candidates(
            image_id,
            census._image_candidates(image_id, adapter, reverse_input=False),
        )
    census_rows = [
        original_census.get(
            image_id,
            {
                "image_id": image_id,
                "admission": {"primary_natural_alias_admitted": False},
                "candidates": [],
            },
        )
        for image_id in pool_ids
    ]
    census_path = tmp_path / "image-census.jsonl"
    census_path.write_bytes(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in census_rows).encode()
    )
    contract_payload = review.freeze_preseed_contract(
        packet_path=packet,
        expected_packet_sha256=_sha(packet.read_bytes()),
        ontology_path=ontology,
        expected_ontology_sha256=_sha(ontology.read_bytes()),
        candidate_pool_path=pool,
        expected_candidate_pool_sha256=_sha(pool.read_bytes()),
        census_path=census_path,
        expected_census_sha256=_sha(census_path.read_bytes()),
        source_panel_root=source_panel,
        sampled_panel_root=sampled_panel,
    )
    contract_path = tmp_path / "preseed-contract.json"
    contract_path.write_bytes(contract_payload)

    image_sha = _sha(image.read_bytes())
    owner_semantics_by_id = {
        image_id: [
            {
                "owner_id": f"{image_id}:0",
                "source_category_name": "cat",
                "normalized_category_name": "cat",
                "source_category_id": 17,
                "official_coco_category_id": 17,
                "source_canvas_box_xyxy": [0.0, 0.0, 10.0, 10.0],
                "annotation_index": 0,
                "image_id": image_id,
            }
        ]
        for image_id in pool_ids
    }
    census_by_id = {str(row["image_id"]): row for row in census_rows}
    pool_by_id = {str(row["image_id"]): row for row in pool_rows}
    member_rows = []
    for image_id in pool_ids:
        replay_hash = (
            review._member_replay_input_sha256(
                adapter=adapter,
                image_id=image_id,
                census_record=census_by_id[image_id],
            )
            if image_id in set(selected_ids)
            else "a" * 64
        )
        member_rows.append(
            {
                "schema_version": review.MEMBER_MANIFEST_SCHEMA_VERSION,
                "image_id": image_id,
                "source_image_path": str(image.resolve()),
                "source_image_size_bytes": image.stat().st_size,
                "source_image_sha256": image_sha,
                "source_image_width": 12,
                "source_image_height": 10,
                "candidate_pool_record_sha256": _sha(
                    canonical_json_text(pool_by_id[image_id]).encode()
                ),
                "census_record_sha256": _sha(
                    canonical_json_text(census_by_id[image_id]).encode()
                ),
                "official_owner_record_sha256": _sha(
                    canonical_json_text(owner_semantics_by_id[image_id]).encode()
                ),
                "replay_input_sha256": replay_hash,
                "route_count": len(review.EXPECTED_ROUTE_IDS),
                "route_inventory_sha256": review._route_inventory_sha256(),
            }
        )
    member_payload = _jsonl(member_rows)
    member_path = tmp_path / "member-manifest.jsonl"
    member_path.write_bytes(member_payload)
    _, member_ids, semantic_hash = review._validate_member_manifest(
        member_payload, expected_sha256=_sha(member_payload)
    )
    stage = _stage_zero_fixture(
        tmp_path / "stage-zero", ordered_possible_image_ids=pool_ids
    )
    member_receipt = {
        "schema_version": review.MEMBER_MANIFEST_RECEIPT_SCHEMA_VERSION,
        "terminal_status": "completed_before_entropy",
        "built_before_entropy": True,
        "member_manifest_path": str(member_path.resolve()),
        "member_manifest_sha256": _sha(member_payload),
        "member_count": len(member_rows),
        "ordered_member_image_ids_sha256": review._ordered_image_ids_sha256(member_ids),
        "member_semantic_sha256": semantic_hash,
        "stage_zero_root": str(stage["root"].resolve()),
        "stage_zero_root_inventory_sha256": stage["root_inventory_sha256"],
        "stage_zero_receipt_path": str(stage["receipt"].resolve()),
        "stage_zero_receipt_sha256": stage["receipt_sha256"],
        "stage_zero_audit_path": str(stage["audit"].resolve()),
        "stage_zero_audit_sha256": stage["audit_sha256"],
        "stage_zero_audit_verdict": "approved",
        "stage_zero_possible_pool_path": str(stage["possible_pool"].resolve()),
        "stage_zero_possible_pool_sha256": stage["possible_pool_sha256"],
        "stage_zero_possible_pool_count": len(pool_ids),
        "stage_zero_ordered_possible_image_ids_sha256": (
            review._ordered_image_ids_sha256(pool_ids)
        ),
        "candidate_pool_path": str(pool.resolve()),
        "candidate_pool_sha256": _sha(pool.read_bytes()),
        "census_path": str(census_path.resolve()),
        "census_sha256": _sha(census_path.read_bytes()),
        "frozen_contract_path": str(contract_path.resolve()),
        "frozen_contract_sha256": _sha(contract_payload),
        "source_panel_root": str(source_panel.resolve()),
        "source_panel_manifest_set_sha256": review._directory_manifest(source_panel)[1],
        "sampled_panel_root": str(sampled_panel.resolve()),
        "sampled_panel_manifest_set_sha256": review._directory_manifest(sampled_panel)[
            1
        ],
        "route_count": len(review.EXPECTED_ROUTE_IDS),
        "route_inventory_sha256": review._route_inventory_sha256(),
        "adapter_inventory_sha256": review._adapter_inventory_sha256(
            {}, ordered_image_ids=pool_ids
        ),
    }
    member_receipt_payload = _json(member_receipt)
    member_receipt_path = tmp_path / "member-manifest-receipt.json"
    member_receipt_path.write_bytes(member_receipt_payload)

    selection_root = (tmp_path / "selection-final").resolve()
    selection_root.mkdir()
    selection_path = selection_root / "selection.json"

    stage_receipt = json.loads(stage["receipt"].read_bytes())
    stage_audit = json.loads(stage["audit"].read_bytes())
    stage_inventory = review._root_inventory(stage["root"])[0]
    selector_stage = selector.StageZeroBinding(
        root=stage["root"].resolve(),
        receipt=stage_receipt,
        receipt_sha256=stage["receipt_sha256"],
        inventory=stage_inventory,
        inventory_sha256=stage["root_inventory_sha256"],
        possible_image_ids=tuple(pool_ids),
        impossible_image_ids=(),
        ordered_pool_sha256=review._ordered_image_ids_sha256(pool_ids),
        audit_path=stage["audit"].resolve(),
        audit=stage_audit,
        audit_sha256=stage["audit_sha256"],
    )
    selector_members = [
        {
            "image_id": row["image_id"],
            "canonical_source_path": row["source_image_path"],
            "source_image_bytes": row["source_image_size_bytes"],
            "source_image_sha256": row["source_image_sha256"],
            "source_image_width": row["source_image_width"],
            "source_image_height": row["source_image_height"],
            "candidate_pool_record_sha256": row["candidate_pool_record_sha256"],
            "census_record_sha256": row["census_record_sha256"],
            "official_owner_record_sha256": row["official_owner_record_sha256"],
            "route_replay_input_record_sha256": row["replay_input_sha256"],
            "route_count": row["route_count"],
            "route_inventory_sha256": row["route_inventory_sha256"],
        }
        for row in member_rows
    ]
    stage_inputs = stage_receipt["inputs"]
    selector_member_bindings = {
        "candidate_pool_path": str(pool.resolve()),
        "candidate_pool_bytes": pool.stat().st_size,
        "candidate_pool_sha256": _sha(pool.read_bytes()),
        "census_path": str(census_path.resolve()),
        "census_bytes": census_path.stat().st_size,
        "census_sha256": _sha(census_path.read_bytes()),
        "sampled_manifest_binding_sha256": _sha(
            canonical_json_text(
                stage_inputs["sampled_manifest_binding"]
            ).encode()
        ),
        "source_manifest_binding_sha256": _sha(
            canonical_json_text(
                stage_inputs["source_manifest_binding"]
            ).encode()
        ),
        "execution_model_identity_sha256": stage_inputs[
            "execution_model_identity_sha256"
        ],
        "tokenizer_identity_sha256": stage_inputs[
            "tokenizer_identity_sha256"
        ],
        "stage_zero_receipt_sha256": stage["receipt_sha256"],
    }
    selector_member_manifest = {
        "schema_version": selector.MEMBER_MANIFEST_SCHEMA_VERSION,
        "population_scope": selector.STAGE_ZERO_POPULATION_SCOPE,
        "count": len(pool_ids),
        "ordered_image_ids": pool_ids,
        "ordered_image_ids_sha256": selector.sha256_json(pool_ids),
        "members": selector_members,
        "members_sha256": selector.sha256_json(selector_members),
        "input_bindings": selector_member_bindings,
        "input_bindings_sha256": selector.sha256_json(
            selector_member_bindings
        ),
    }
    selector.validate_member_manifest(selector_member_manifest, pool_ids)

    named_source_paths: dict[str, Path] = {
        "b16_drop_chronology_helper": (
            selector.B16_DROP_CHRONOLOGY_HELPER_PATH.resolve()
        ),
        "b16_drop_chronology_impact_review": (
            selector.B16_DROP_CHRONOLOGY_IMPACT_REVIEW_PATH.resolve()
        ),
        "reviewer_instruction_packet": packet.resolve(),
        "ontology_state_artifact": ontology.resolve(),
        "review_adjudication_replay_implementation": Path(
            review.__file__
        ).resolve(),
        "outcome_classifier_contract": contract_path.resolve(),
    }
    source_fixture_root = tmp_path / "selector-source-fixtures"
    source_fixture_root.mkdir()
    for label in selector.PRODUCTION_NAMED_SOURCE_LABELS:
        if label in named_source_paths:
            continue
        path = source_fixture_root / f"{label}.txt"
        path.write_text(f"{label}\n", encoding="utf-8")
        named_source_paths[label] = path.resolve()
    named_sources = [
        selector.NamedSource(label, named_source_paths[label], _sha(
            named_source_paths[label].read_bytes()
        ))
        for label in sorted(selector.PRODUCTION_NAMED_SOURCE_LABELS)
    ]
    source_manifest = selector.build_frozen_source_manifest(named_sources)

    claim = selector._journal_claim(
        output_root=selection_root,
        source_manifest=source_manifest,
        stage=selector_stage,
        member_manifest=selector_member_manifest,
    )
    entropy_bytes = entropy_integer.to_bytes(review.ENTROPY_BYTE_COUNT, "big")
    entropy_sha = _sha(entropy_bytes)
    journal_root = selector.journal_root_for(selection_root)
    journal_root.mkdir()
    claim_payload = _pretty_json(claim)
    (journal_root / "claim.json").write_bytes(claim_payload)
    (journal_root / "entropy.bin").write_bytes(entropy_bytes)
    (journal_root / "entropy-receipt.json").write_bytes(
        _pretty_json(
            {
                "byte_count": review.ENTROPY_BYTE_COUNT,
                "sha256": entropy_sha,
            }
        )
    )
    terminal = {
        "schema_version": selector.JOURNAL_TERMINAL_SCHEMA_VERSION,
        "terminal_status": "completed",
        "claim_sha256": selector.sha256_json(claim),
        "entropy_sha256": entropy_sha,
        "reason": None,
    }
    journal_payload = _pretty_json(terminal)
    (journal_root / "terminal.json").write_bytes(journal_payload)
    journal_state = selector.JournalState(
        root=journal_root,
        claim=claim,
        claim_file_sha256=_sha(claim_payload),
        entropy=entropy_bytes,
        resumed=False,
        terminal=terminal,
    )
    randomization = selector.randomization_from_entropy(pool_ids, entropy_bytes)
    assert randomization.accepted
    selection = selector._selection_record(
        state=journal_state,
        stage=selector_stage,
        source_manifest=source_manifest,
        member_manifest=selector_member_manifest,
        randomization=randomization,
    )
    selection_payload = _pretty_json(selection)
    selection_path.write_bytes(selection_payload)
    (selection_root / "frozen-source-manifest.json").write_bytes(
        _pretty_json(source_manifest)
    )
    (selection_root / "possible-pool-member-manifest.json").write_bytes(
        _pretty_json(selector_member_manifest)
    )
    snapshot_root = selection_root / "source-snapshots"
    snapshot_root.mkdir()
    source_entries = source_manifest["entries"]
    assert isinstance(source_entries, list)
    for entry in source_entries:
        assert isinstance(entry, dict)
        shutil.copyfile(
            Path(str(entry["path"])),
            selection_root / str(entry["snapshot_relative_path"]),
        )
    outputs = review._root_inventory(selection_root)[0]
    selection_receipt = selector._selection_receipt(
        output_root=selection_root,
        journal=journal_state,
        claim=claim,
        selection=selection,
        source_manifest=source_manifest,
        member_manifest=selector_member_manifest,
        outputs=outputs,
    )
    (selection_root / "receipt.json").write_bytes(
        _pretty_json(selection_receipt)
    )
    for immutable_root in (journal_root, selection_root):
        for path in immutable_root.rglob("*"):
            if path.is_file():
                path.chmod(0o444)
        for path in sorted(
            (path for path in immutable_root.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            path.chmod(0o555)
        immutable_root.chmod(0o555)
    journal_path = journal_root
    validated = review._validate_selection(
        selection_payload,
        expected_sha256=_sha(selection_payload),
        selection_path=selection_path,
        member_manifest_jsonl=member_payload,
        expected_member_manifest_sha256=_sha(member_payload),
        member_manifest_receipt_json=member_receipt_payload,
        expected_member_manifest_receipt_sha256=_sha(member_receipt_payload),
        expected_packet_sha256=_sha(packet.read_bytes()),
        expected_ontology_sha256=_sha(ontology.read_bytes()),
    )
    assert validated.ordered_selected_image_ids == tuple(selected_ids)
    return {
        "image": image,
        "source_panel": source_panel,
        "sampled_panel": sampled_panel,
        "packet": packet,
        "ontology": ontology,
        "pool": pool,
        "census": census_path,
        "contract": contract_path,
        "member": member_path,
        "member_payload": member_payload,
        "member_receipt": member_receipt_path,
        "member_receipt_payload": member_receipt_payload,
        "selection_root": selection_root,
        "selection": selection_path,
        "selection_payload": selection_payload,
        "journal": journal_path,
        "journal_payload": journal_payload,
        "selected_ids": selected_ids,
        "adapter": adapter,
    }


def _queue_for(
    experiment: dict[str, Any],
    *,
    look_id: str = "look_one",
    prior_decision: bytes | None = None,
    prior_receipt: bytes | None = None,
) -> review.ReviewQueueArtifacts:
    selected = (
        experiment["selected_ids"][:16]
        if look_id == "look_one"
        else experiment["selected_ids"][16:]
    )
    return review.build_review_queue(
        selection_json=experiment["selection_payload"],
        expected_selection_sha256=_sha(experiment["selection_payload"]),
        selection_path=experiment["selection"],
        member_manifest_jsonl=experiment["member_payload"],
        expected_member_manifest_sha256=_sha(experiment["member_payload"]),
        member_manifest_receipt_json=experiment["member_receipt_payload"],
        expected_member_manifest_receipt_sha256=_sha(
            experiment["member_receipt_payload"]
        ),
        adapter=_subset_adapter(experiment["adapter"], selected),
        candidate_pool_path=experiment["pool"],
        expected_candidate_pool_sha256=_sha(experiment["pool"].read_bytes()),
        census_path=experiment["census"],
        expected_census_path=experiment["census"],
        expected_census_sha256=_sha(experiment["census"].read_bytes()),
        packet_path=experiment["packet"],
        expected_packet_sha256=_sha(experiment["packet"].read_bytes()),
        ontology_path=experiment["ontology"],
        expected_ontology_sha256=_sha(experiment["ontology"].read_bytes()),
        look_id=look_id,
        prior_look_decision_json=prior_decision,
        prior_look_receipt_json=prior_receipt,
    )


def _queue_fixture(tmp_path: Path) -> tuple[review.ReviewQueueArtifacts, str, str]:
    experiment = _make_experiment(tmp_path)
    artifacts = _queue_for(experiment)
    return (
        artifacts,
        _sha(experiment["packet"].read_bytes()),
        _sha(experiment["ontology"].read_bytes()),
    )


def _empty_role(
    queue_payload: bytes, role: str, packet_sha: str, ontology_sha: str
) -> bytes:
    queue_rows = [
        row
        for row in _decode_jsonl(queue_payload)
        if row["reviewer_role_identifier"] == role
    ]
    return _jsonl(
        [
            {
                "schema_version": review.REVIEWER_SCHEMA_BY_ROLE[role],
                "packet_id": review.PACKET_ID,
                "packet_sha256": packet_sha,
                "ontology_sha256": ontology_sha,
                "review_queue_sha256": _sha(queue_payload),
                "review_identifier": row["review_identifier"],
                "reviewer_role_identifier": role,
                "image_id": row["image_id"],
                "image_sha256": row["image_sha256"],
                "source_image_width": row["source_image_width"],
                "source_image_height": row["source_image_height"],
                "image_disposition": "complete",
                "labels": [],
            }
            for row in queue_rows
        ]
    )


def _complete_look(
    experiment: dict[str, Any], *, added_owner_image_id: str | None = None
) -> dict[str, bytes]:
    queue = _queue_for(experiment)
    packet_sha = _sha(experiment["packet"].read_bytes())
    ontology_sha = _sha(experiment["ontology"].read_bytes())
    role_payloads = {
        role: _empty_role(queue.review_queue_jsonl, role, packet_sha, ontology_sha)
        for role in review.REVIEWER_ROLES
    }
    if added_owner_image_id is not None:
        if added_owner_image_id not in experiment["selected_ids"][:16]:
            raise ValueError("test addition must belong to Look One")
        for role in review.REVIEWER_ROLES:
            rows = _decode_jsonl(role_payloads[role])
            target = next(
                row for row in rows if row["image_id"] == added_owner_image_id
            )
            target["labels"] = [
                _label(
                    role,
                    added_owner_image_id,
                    1,
                    name="dog",
                    official_id=18,
                    box=[1, 1, 9, 9],
                )
            ]
            role_payloads[role] = _jsonl(rows)
    role_seals = {
        role: review.seal_role_artifact(
            role_artifact_jsonl=role_payloads[role],
            reviewer_role_identifier=role,
            review_queue_jsonl=queue.review_queue_jsonl,
            expected_review_queue_sha256=_sha(queue.review_queue_jsonl),
            expected_packet_sha256=packet_sha,
            expected_ontology_sha256=ontology_sha,
        )
        for role in review.REVIEWER_ROLES
    }
    adjudication_queue = review.build_adjudication_queue(
        review_queue_jsonl=queue.review_queue_jsonl,
        expected_review_queue_sha256=_sha(queue.review_queue_jsonl),
        reviewer_one_jsonl=role_payloads["reviewer-one"],
        reviewer_one_seal_json=role_seals["reviewer-one"],
        reviewer_two_jsonl=role_payloads["reviewer-two"],
        reviewer_two_seal_json=role_seals["reviewer-two"],
        official_owner_ledger_jsonl=queue.official_owner_ledger_jsonl,
        expected_official_owner_ledger_sha256=_sha(queue.official_owner_ledger_jsonl),
        expected_packet_sha256=packet_sha,
        expected_ontology_sha256=ontology_sha,
    )
    decision_rows = []
    for row in _decode_jsonl(adjudication_queue):
        is_addition = row["image_id"] == added_owner_image_id
        decision_rows.append(
            {
                "schema_version": review.ADJUDICATION_DECISION_SCHEMA_VERSION,
                "adjudication_identifier": row["adjudication_identifier"],
                "adjudication_queue_sha256": _sha(adjudication_queue),
                "source_binding_sha256": _sha(canonical_json_text(row).encode()),
                "confirmed_official_owner_ids": sorted(
                    owner["owner_id"] for owner in row["official_owners"]
                ),
                "proposal_dispositions": (
                    [
                        {
                            "linked_reviewer_label_identifiers": [
                                f"{role}:{added_owner_image_id}:0001"
                                for role in review.REVIEWER_ROLES
                            ],
                            "adjudication_state": "accepted_missing_owner",
                            "reason_code": "reviewer_agreement",
                            "linked_official_owner_ids": [],
                            "final_normalized_category_name": "dog",
                            "final_official_coco_category_id": 18,
                            "final_source_canvas_box_xyxy": [1, 1, 9, 9],
                        }
                    ]
                    if is_addition
                    else []
                ),
                "owner_universe_complete": is_addition,
                "owner_universe_uncertainty_reasons": (
                    [] if is_addition else ["ledger_completeness_not_established"]
                ),
            }
        )
    decisions = _jsonl(decision_rows)
    ledger = review.assemble_owner_ledger(
        adjudication_queue_jsonl=adjudication_queue,
        expected_adjudication_queue_sha256=_sha(adjudication_queue),
        adjudicator_decisions_jsonl=decisions,
        review_queue_manifest_json=queue.manifest_json,
        expected_review_queue_manifest_sha256=_sha(queue.manifest_json),
    )
    current_ids = experiment["selected_ids"][:16]
    replay = review.replay_owner_ledger(
        adapter=_subset_adapter(experiment["adapter"], current_ids),
        look_id="look_one",
        selection_json=experiment["selection_payload"],
        expected_selection_sha256=_sha(experiment["selection_payload"]),
        selection_path=experiment["selection"],
        member_manifest_jsonl=experiment["member_payload"],
        expected_member_manifest_sha256=_sha(experiment["member_payload"]),
        member_manifest_receipt_json=experiment["member_receipt_payload"],
        expected_member_manifest_receipt_sha256=_sha(
            experiment["member_receipt_payload"]
        ),
        candidate_pool_path=experiment["pool"],
        expected_candidate_pool_sha256=_sha(experiment["pool"].read_bytes()),
        census_path=experiment["census"],
        expected_census_path=experiment["census"],
        expected_census_sha256=_sha(experiment["census"].read_bytes()),
        owner_ledger_jsonl=ledger.owner_ledger_jsonl,
        owner_ledger_seal_json=ledger.owner_ledger_seal_json,
    )
    outcomes = review.classify_outcomes(
        replay_records_jsonl=replay.replay_records_jsonl,
        replay_receipt_json=replay.replay_receipt_json,
        uncertainty_ledger_jsonl=ledger.uncertainty_ledger_jsonl,
        owner_ledger_seal_json=ledger.owner_ledger_seal_json,
        look_id="look_one",
        ordered_image_ids=current_ids,
        expected_selection_sha256=_sha(experiment["selection_payload"]),
        expected_member_manifest_sha256=_sha(experiment["member_payload"]),
        expected_member_manifest_receipt_sha256=_sha(
            experiment["member_receipt_payload"]
        ),
        expected_candidate_pool_sha256=_sha(experiment["pool"].read_bytes()),
        expected_census_sha256=_sha(experiment["census"].read_bytes()),
    )
    decision = review.build_sequential_decision(
        look_id="look_one",
        outcomes_jsonl=outcomes,
        selection_json=experiment["selection_payload"],
        expected_selection_sha256=_sha(experiment["selection_payload"]),
        selection_path=experiment["selection"],
        member_manifest_jsonl=experiment["member_payload"],
        expected_member_manifest_sha256=_sha(experiment["member_payload"]),
        member_manifest_receipt_json=experiment["member_receipt_payload"],
        expected_member_manifest_receipt_sha256=_sha(
            experiment["member_receipt_payload"]
        ),
        expected_candidate_pool_sha256=_sha(experiment["pool"].read_bytes()),
        expected_census_sha256=_sha(experiment["census"].read_bytes()),
    )
    return {
        "selection.json": experiment["selection_payload"],
        "entropy-journal.json": experiment["journal_payload"],
        "member-manifest.jsonl": experiment["member_payload"],
        "member-manifest-receipt.json": experiment["member_receipt_payload"],
        "review-queue.jsonl": queue.review_queue_jsonl,
        "review-queue-manifest.json": queue.manifest_json,
        "official-owner-ledger.jsonl": queue.official_owner_ledger_jsonl,
        "reviewer-one.jsonl": role_payloads["reviewer-one"],
        "reviewer-one-seal.json": role_seals["reviewer-one"],
        "reviewer-two.jsonl": role_payloads["reviewer-two"],
        "reviewer-two-seal.json": role_seals["reviewer-two"],
        "adjudication-queue.jsonl": adjudication_queue,
        "adjudication.jsonl": decisions,
        "owner-ledger.jsonl": ledger.owner_ledger_jsonl,
        "uncertainty-ledger.jsonl": ledger.uncertainty_ledger_jsonl,
        "owner-ledger-seal.json": ledger.owner_ledger_seal_json,
        "replay-records.jsonl": replay.replay_records_jsonl,
        "replay-receipt.json": replay.replay_receipt_json,
        "outcomes.jsonl": outcomes,
        "sequential-decision.json": decision,
    }


def _finalize(
    experiment: dict[str, Any], bundle: dict[str, bytes], root: Path
) -> bytes:
    return review.finalize_look(
        adapter=_subset_adapter(
            experiment["adapter"], experiment["selected_ids"][:16]
        ),
        final_root=root,
        look_id="look_one",
        artifact_payloads=bundle,
        source_bindings={
            "packet": experiment["packet"],
            "ontology": experiment["ontology"],
            "candidate_pool": experiment["pool"],
            "census": experiment["census"],
            "frozen_contract": experiment["contract"],
            "source_panel": experiment["source_panel"],
            "sampled_panel": experiment["sampled_panel"],
        },
        selection_path=experiment["selection"],
        member_manifest_path=experiment["member"],
        member_manifest_receipt_path=experiment["member_receipt"],
        expected_selection_sha256=_sha(experiment["selection_payload"]),
        expected_member_manifest_sha256=_sha(experiment["member_payload"]),
        expected_member_manifest_receipt_sha256=_sha(
            experiment["member_receipt_payload"]
        ),
        expected_candidate_pool_sha256=_sha(experiment["pool"].read_bytes()),
        expected_census_sha256=_sha(experiment["census"].read_bytes()),
    )


def _unlock_tree(root: Path) -> None:
    if not root.exists():
        return
    root.chmod(0o755)
    for path in root.rglob("*"):
        if path.is_dir():
            path.chmod(0o755)
        else:
            path.chmod(0o644)


def test_queue_is_train_only_and_reviewer_visible_fields_are_route_blind(
    tmp_path: Path,
) -> None:
    experiment = _make_experiment(tmp_path)
    artifacts = _queue_for(experiment)
    queue_rows = _decode_jsonl(artifacts.review_queue_jsonl)
    manifest = json.loads(artifacts.manifest_json)

    assert len(queue_rows) == 32
    assert all(set(row) == set(review._QUEUE_FIELDS) for row in queue_rows)
    assert not any(
        set(row) & review._FORBIDDEN_REVIEW_EVIDENCE_FIELDS for row in queue_rows
    )
    visible_ids = [
        row["image_id"]
        for row in queue_rows
        if row["reviewer_role_identifier"] == "reviewer-one"
    ]
    assert visible_ids == sorted(experiment["selected_ids"][:16], key=int)
    assert manifest["private_sample_order_image_ids"] == experiment["selected_ids"][:16]
    assert manifest["look_identity_reviewer_visible"] is False
    assert all("look" not in key and "cutoff" not in key for key in queue_rows[0])


def test_actual_reviewer_packet_discloses_no_sequential_or_sample_information() -> None:
    packet_path = (
        Path(__file__).resolve().parents[2]
        / "research/investigations/qwen3-vl-dense-enumeration/experiments"
        / "2026-07-23-trajectory-owner-set-adjudication-salvage-gate"
        / "reviewer-instruction-packet-v1.md"
    )
    packet = packet_path.read_text(encoding="utf-8").lower()
    forbidden_fragments = (
        "look one",
        "look two",
        "look identity",
        "another look",
        "sample order",
        "sample position",
        "sample size",
        "cutoff",
        "hypergeometric",
        "sequential",
        "continue_to_look_two",
        "prior outcome",
        "cumulative success",
        "rejection boundary",
    )
    assert not [fragment for fragment in forbidden_fragments if fragment in packet]
    assert "any selection, ordering, grouping, statistical-analysis" in packet


def test_preseed_contract_binds_packet_schema_and_source_hashes(tmp_path: Path) -> None:
    experiment = _make_experiment(tmp_path)
    seal = json.loads(experiment["contract"].read_bytes())

    assert seal["seed_generated"] is False
    assert seal["selection_materialized"] is False
    assert seal["source_bindings"]["census"]["sha256"] == _sha(
        experiment["census"].read_bytes()
    )
    assert seal["source_bindings"]["sampled_panel"]["kind"] == "manifest_set"
    assert seal["source_bindings"]["review_assembler"]["sha256"] == _sha(
        Path(review.__file__).read_bytes()
    )


def test_role_validation_accepts_explicit_empty_dispositions_and_fails_closed(
    tmp_path: Path,
) -> None:
    experiment = _make_experiment(tmp_path)
    queue = _queue_for(experiment)
    packet_sha = _sha(experiment["packet"].read_bytes())
    ontology_sha = _sha(experiment["ontology"].read_bytes())
    artifact = _empty_role(
        queue.review_queue_jsonl, "reviewer-one", packet_sha, ontology_sha
    )
    rows = review.validate_role_artifact(
        role_artifact_jsonl=artifact,
        reviewer_role_identifier="reviewer-one",
        review_queue_jsonl=queue.review_queue_jsonl,
        expected_review_queue_sha256=_sha(queue.review_queue_jsonl),
        expected_packet_sha256=packet_sha,
        expected_ontology_sha256=ontology_sha,
    )
    assert len(rows) == 16
    assert all(row["labels"] == [] for row in rows)

    with pytest.raises(ValueError, match="complete image disposition"):
        review.validate_role_artifact(
            role_artifact_jsonl=_jsonl(_decode_jsonl(artifact)[:-1]),
            reviewer_role_identifier="reviewer-one",
            review_queue_jsonl=queue.review_queue_jsonl,
            expected_review_queue_sha256=_sha(queue.review_queue_jsonl),
            expected_packet_sha256=packet_sha,
            expected_ontology_sha256=ontology_sha,
        )


def test_two_seals_gate_add_only_ledger_and_preserve_uncertainty(
    tmp_path: Path,
) -> None:
    experiment = _make_experiment(tmp_path)
    bundle = _complete_look(experiment)
    seal = json.loads(bundle["owner-ledger-seal.json"])
    assert seal["selected_image_count"] == 16
    assert seal["official_owner_identity_and_box_immutable"] is True
    assert seal["look_id"] == "look_one"
    assert seal["selection_sha256"] == _sha(experiment["selection_payload"])
    assert len(_decode_jsonl(bundle["uncertainty-ledger.jsonl"])) == 16


def test_frozen_census_reads_spaced_bytes_before_decode_and_proves_set(
    tmp_path: Path,
) -> None:
    path = tmp_path / "spaced.jsonl"
    path.write_text('{"image_id": 2, "value": 1}\n{"image_id": 1, "value": 2}\n')
    selected = review.load_frozen_census_records(
        census_path=path,
        expected_census_path=path,
        expected_census_sha256=_sha(path.read_bytes()),
        selected_image_ids=["1", "2"],
    )
    assert tuple(selected.records) == ("1", "2")

    alias = tmp_path / "alias.jsonl"
    alias.write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="path drift"):
        review.load_frozen_census_records(
            census_path=alias,
            expected_census_path=path,
            expected_census_sha256=_sha(path.read_bytes()),
            selected_image_ids=["1"],
        )
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text('{"image_id": 1}\n{"image_id": 1}\n')
    with pytest.raises(ValueError, match="duplicates"):
        review.load_frozen_census_records(
            census_path=duplicate,
            expected_census_path=duplicate,
            expected_census_sha256=_sha(duplicate.read_bytes()),
            selected_image_ids=["1"],
        )


def test_selection_rejects_cutoff_unranking_and_source_drift(tmp_path: Path) -> None:
    experiment = _make_experiment(tmp_path)
    value = json.loads(experiment["selection_payload"])
    value["hypergeometric_design"][0]["largest_rejection_success_count"] -= 1
    tampered = _pretty_json(value)
    experiment["selection"].chmod(0o644)
    experiment["selection"].write_bytes(tampered)
    experiment["selection"].chmod(0o444)
    with pytest.raises(ValueError, match="terminal design contract drift"):
        review._validate_selection(
            tampered,
            expected_sha256=_sha(tampered),
            selection_path=experiment["selection"],
            member_manifest_jsonl=experiment["member_payload"],
            expected_member_manifest_sha256=_sha(experiment["member_payload"]),
            member_manifest_receipt_json=experiment["member_receipt_payload"],
            expected_member_manifest_receipt_sha256=_sha(
                experiment["member_receipt_payload"]
            ),
        )


def test_selection_uses_unit_m_r_lexicographic_unranking_and_one_entropy_block(
    tmp_path: Path,
) -> None:
    experiment = _make_experiment(tmp_path)
    selection = json.loads(experiment["selection_payload"])
    entropy = selection["entropy"]
    expected_m = review._falling_factorial(248, review.SELECTION_COUNT)
    assert entropy["permutation_count_M"] == expected_m
    assert entropy["entropy_integer_R"] == 123456
    assert entropy["accepted_initial_rank"] == 123456
    first = entropy["unranking_trace"][0]
    assert first["position"] == 0
    assert first["suffix_count"] == review._falling_factorial(247, 31)
    assert first["choice_index"] == 0
    assert first["selected_image_id"] == "1"

    journal_root = experiment["journal"]
    journal_root.chmod(0o755)
    redraw = journal_root / "entropy-redraw.bin"
    redraw.write_bytes(bytes(review.ENTROPY_BYTE_COUNT))
    redraw.chmod(0o444)
    journal_root.chmod(0o555)
    with pytest.raises(ValueError, match="journal inventory is not exact"):
        review._validate_selection(
            experiment["selection_payload"],
            expected_sha256=_sha(experiment["selection_payload"]),
            selection_path=experiment["selection"],
            member_manifest_jsonl=experiment["member_payload"],
            expected_member_manifest_sha256=_sha(experiment["member_payload"]),
            member_manifest_receipt_json=experiment["member_receipt_payload"],
            expected_member_manifest_receipt_sha256=_sha(
                experiment["member_receipt_payload"]
            ),
        )


def test_selection_rejects_claim_or_selector_member_that_postdates_entropy(
    tmp_path: Path,
) -> None:
    experiment = _make_experiment(tmp_path)
    claim_path = experiment["journal"] / "claim.json"
    terminal_path = experiment["journal"] / "terminal.json"
    claim = json.loads(claim_path.read_bytes())
    claim["member_records_sha256"] = "f" * 64
    terminal = json.loads(terminal_path.read_bytes())
    terminal["claim_sha256"] = _sha(canonical_json_text(claim).encode())
    claim_path.chmod(0o644)
    terminal_path.chmod(0o644)
    claim_path.write_bytes(_pretty_json(claim))
    terminal_path.write_bytes(_pretty_json(terminal))
    claim_path.chmod(0o444)
    terminal_path.chmod(0o444)
    with pytest.raises(ValueError, match="pre-entropy claim binding drift"):
        review._validate_selection(
            experiment["selection_payload"],
            expected_sha256=_sha(experiment["selection_payload"]),
            selection_path=experiment["selection"],
            member_manifest_jsonl=experiment["member_payload"],
            expected_member_manifest_sha256=_sha(experiment["member_payload"]),
            member_manifest_receipt_json=experiment["member_receipt_payload"],
            expected_member_manifest_receipt_sha256=_sha(
                experiment["member_receipt_payload"]
            ),
        )

    second_root = tmp_path / "second"
    second_root.mkdir()
    second = _make_experiment(second_root)
    selection = json.loads(second["selection_payload"])
    selector_manifest = selection["possible_pool_member_manifest"]
    selector_manifest["members"][0]["official_owner_record_sha256"] = "e" * 64
    selector_manifest["members_sha256"] = _sha(
        canonical_json_text(selector_manifest["members"]).encode()
    )
    selection["possible_pool_member_records_sha256"] = selector_manifest[
        "members_sha256"
    ]
    selection["possible_pool_member_manifest_sha256"] = _sha(
        canonical_json_text(selector_manifest).encode()
    )
    tampered = _pretty_json(selection)
    second["selection"].chmod(0o644)
    second["selection"].write_bytes(tampered)
    second["selection"].chmod(0o444)
    with pytest.raises(ValueError, match="pre-entropy member rows differ"):
        review._validate_selection(
            tampered,
            expected_sha256=_sha(tampered),
            selection_path=second["selection"],
            member_manifest_jsonl=second["member_payload"],
            expected_member_manifest_sha256=_sha(second["member_payload"]),
            member_manifest_receipt_json=second["member_receipt_payload"],
            expected_member_manifest_receipt_sha256=_sha(
                second["member_receipt_payload"]
            ),
        )


def test_global_replay_noop_exactly_reproduces_census_and_preserves_identities(
    tmp_path: Path,
) -> None:
    experiment = _make_experiment(tmp_path)
    bundle = _complete_look(experiment)
    replay_rows = _decode_jsonl(bundle["replay-records.jsonl"])
    frozen = review.load_frozen_census_records(
        census_path=experiment["census"],
        expected_census_path=experiment["census"],
        expected_census_sha256=_sha(experiment["census"].read_bytes()),
        selected_image_ids=experiment["selected_ids"][:16],
    )

    assert [row["census_record"] for row in replay_rows] == [
        frozen.records[image_id] for image_id in experiment["selected_ids"][:16]
    ]
    assert all(row["no_op_exact_reproduction"] for row in replay_rows)
    assert all(
        identity["original_row_identity_and_order_sha256"]
        == identity["replayed_row_identity_and_order_sha256"]
        for row in replay_rows
        for identity in row["route_identity"].values()
    )


def test_classifier_rejects_duplicates_and_mixed_chain_hashes(tmp_path: Path) -> None:
    experiment = _make_experiment(tmp_path)
    bundle = _complete_look(experiment)
    common = {
        "replay_receipt_json": bundle["replay-receipt.json"],
        "uncertainty_ledger_jsonl": bundle["uncertainty-ledger.jsonl"],
        "owner_ledger_seal_json": bundle["owner-ledger-seal.json"],
        "look_id": "look_one",
        "ordered_image_ids": experiment["selected_ids"][:16],
        "expected_selection_sha256": _sha(experiment["selection_payload"]),
        "expected_member_manifest_sha256": _sha(experiment["member_payload"]),
        "expected_member_manifest_receipt_sha256": _sha(
            experiment["member_receipt_payload"]
        ),
        "expected_candidate_pool_sha256": _sha(experiment["pool"].read_bytes()),
        "expected_census_sha256": _sha(experiment["census"].read_bytes()),
    }
    rows = _decode_jsonl(bundle["replay-records.jsonl"])
    with pytest.raises(ValueError, match="duplicate"):
        review.classify_outcomes(
            replay_records_jsonl=_jsonl([*rows, rows[0]]), **common
        )
    rows[1]["owner_ledger_seal_sha256"] = "0" * 64
    mixed = _jsonl(rows)
    receipt = json.loads(bundle["replay-receipt.json"])
    receipt["replay_records_sha256"] = _sha(mixed)
    with pytest.raises(ValueError, match="common chain"):
        review.classify_outcomes(
            replay_records_jsonl=mixed,
            **{**common, "replay_receipt_json": _json(receipt)},
        )


def test_outcome_classifier_counts_all_retained_uncertainty_as_success(
    tmp_path: Path,
) -> None:
    experiment = _make_experiment(tmp_path)
    bundle = _complete_look(experiment)
    outcomes = _decode_jsonl(
        review.classify_outcomes(
            replay_records_jsonl=bundle["replay-records.jsonl"],
            replay_receipt_json=bundle["replay-receipt.json"],
            uncertainty_ledger_jsonl=bundle["uncertainty-ledger.jsonl"],
            owner_ledger_seal_json=bundle["owner-ledger-seal.json"],
            look_id="look_one",
            ordered_image_ids=experiment["selected_ids"][:16],
            expected_selection_sha256=_sha(experiment["selection_payload"]),
            expected_member_manifest_sha256=_sha(experiment["member_payload"]),
            expected_member_manifest_receipt_sha256=_sha(
                experiment["member_receipt_payload"]
            ),
            expected_candidate_pool_sha256=_sha(experiment["pool"].read_bytes()),
            expected_census_sha256=_sha(experiment["census"].read_bytes()),
        )
    )
    assert outcomes[0]["outcome"] == "potential_admission_unresolved"
    assert outcomes[0]["statistical_success"] is True


def test_classifier_and_finalizer_reject_rehashed_corrupt_noop_replay(
    tmp_path: Path,
) -> None:
    experiment = _make_experiment(tmp_path)
    bundle = _complete_look(experiment)
    rows = _decode_jsonl(bundle["replay-records.jsonl"])
    assert rows[0]["added_owner_count"] == 0
    assert rows[0]["no_op_exact_reproduction"] is True
    rows[0]["census_record"]["admission"]["primary_natural_alias_admitted"] = (
        not rows[0]["census_record"]["admission"][
            "primary_natural_alias_admitted"
        ]
    )
    replay_payload = _jsonl(rows)
    receipt = json.loads(bundle["replay-receipt.json"])
    receipt["replay_records_sha256"] = _sha(replay_payload)
    replay_receipt = _json(receipt)
    common = {
        "replay_records_jsonl": replay_payload,
        "replay_receipt_json": replay_receipt,
        "uncertainty_ledger_jsonl": bundle["uncertainty-ledger.jsonl"],
        "owner_ledger_seal_json": bundle["owner-ledger-seal.json"],
        "look_id": "look_one",
        "ordered_image_ids": experiment["selected_ids"][:16],
        "expected_selection_sha256": _sha(experiment["selection_payload"]),
        "expected_member_manifest_sha256": _sha(experiment["member_payload"]),
        "expected_member_manifest_receipt_sha256": _sha(
            experiment["member_receipt_payload"]
        ),
        "expected_candidate_pool_sha256": _sha(experiment["pool"].read_bytes()),
        "expected_census_sha256": _sha(experiment["census"].read_bytes()),
    }
    with pytest.raises(ValueError, match="differs from frozen census"):
        review.classify_outcomes(**common)

    corrupt_bundle = {
        **bundle,
        "replay-records.jsonl": replay_payload,
        "replay-receipt.json": replay_receipt,
    }
    with pytest.raises(ValueError, match="global replay does not reproduce"):
        _finalize(experiment, corrupt_bundle, tmp_path / "corrupt-final")


def test_finalizer_recomputes_added_owner_replay_before_publication(
    tmp_path: Path,
) -> None:
    experiment = _make_experiment(tmp_path)
    added_owner_image_id = experiment["selected_ids"][0]
    bundle = _complete_look(
        experiment, added_owner_image_id=added_owner_image_id
    )
    authentic_rows = _decode_jsonl(bundle["replay-records.jsonl"])
    authentic_target = next(
        row for row in authentic_rows if row["image_id"] == added_owner_image_id
    )
    assert authentic_target["added_owner_count"] == 1
    assert authentic_target["no_op_exact_reproduction"] is False

    authentic_root = tmp_path / "authentic-added-owner-final"
    try:
        _finalize(experiment, bundle, authentic_root)
    finally:
        _unlock_tree(authentic_root)

    forged_rows = _decode_jsonl(bundle["replay-records.jsonl"])
    forged_target = next(
        row for row in forged_rows if row["image_id"] == added_owner_image_id
    )
    admission = forged_target["census_record"]["admission"]
    admission["primary_natural_alias_admitted"] = not admission[
        "primary_natural_alias_admitted"
    ]
    for candidate in forged_target["census_record"]["candidates"]:
        candidate["unknown_count"] = 0
        candidate["ambiguity_count"] = 0
    forged_target["route_identity"][review.EXPECTED_ROUTE_IDS[0]][
        "candidate_identity_sha256"
    ] = "0" * 64
    forged_replay = _jsonl(forged_rows)
    forged_receipt = json.loads(bundle["replay-receipt.json"])
    forged_receipt["replay_records_sha256"] = _sha(forged_replay)
    forged_receipt_payload = _json(forged_receipt)
    current_ids = experiment["selected_ids"][:16]
    forged_outcomes = review.classify_outcomes(
        replay_records_jsonl=forged_replay,
        replay_receipt_json=forged_receipt_payload,
        uncertainty_ledger_jsonl=bundle["uncertainty-ledger.jsonl"],
        owner_ledger_seal_json=bundle["owner-ledger-seal.json"],
        look_id="look_one",
        ordered_image_ids=current_ids,
        expected_selection_sha256=_sha(experiment["selection_payload"]),
        expected_member_manifest_sha256=_sha(experiment["member_payload"]),
        expected_member_manifest_receipt_sha256=_sha(
            experiment["member_receipt_payload"]
        ),
        expected_candidate_pool_sha256=_sha(experiment["pool"].read_bytes()),
        expected_census_sha256=_sha(experiment["census"].read_bytes()),
    )
    forged_decision = review.build_sequential_decision(
        look_id="look_one",
        outcomes_jsonl=forged_outcomes,
        selection_json=experiment["selection_payload"],
        expected_selection_sha256=_sha(experiment["selection_payload"]),
        selection_path=experiment["selection"],
        member_manifest_jsonl=experiment["member_payload"],
        expected_member_manifest_sha256=_sha(experiment["member_payload"]),
        member_manifest_receipt_json=experiment["member_receipt_payload"],
        expected_member_manifest_receipt_sha256=_sha(
            experiment["member_receipt_payload"]
        ),
        expected_candidate_pool_sha256=_sha(experiment["pool"].read_bytes()),
        expected_census_sha256=_sha(experiment["census"].read_bytes()),
    )
    assert forged_outcomes != bundle["outcomes.jsonl"]
    forged_bundle = {
        **bundle,
        "replay-records.jsonl": forged_replay,
        "replay-receipt.json": forged_receipt_payload,
        "outcomes.jsonl": forged_outcomes,
        "sequential-decision.json": forged_decision,
    }
    with pytest.raises(ValueError, match="global replay does not reproduce"):
        _finalize(experiment, forged_bundle, tmp_path / "forged-added-owner-final")


def test_finalize_is_atomic_immutable_and_authorizes_only_sealed_look_two(
    tmp_path: Path,
) -> None:
    experiment = _make_experiment(tmp_path)
    bundle = _complete_look(experiment)
    decision = json.loads(bundle["sequential-decision.json"])
    assert decision["decision"] == "continue_to_look_two"
    with pytest.raises(ValueError, match="finalized look-one authorization"):
        _queue_for(experiment, look_id="look_two_additional")

    final_root = tmp_path / "look-one-final"
    receipt = _finalize(experiment, bundle, final_root)
    try:
        validated = review._validate_final_look_receipt(
            receipt,
            expected_look_id="look_one",
            expected_selection_sha256=_sha(experiment["selection_payload"]),
            expected_member_manifest_sha256=_sha(experiment["member_payload"]),
        )
        assert validated["filesystem_immutable"] is True
        look_two = _queue_for(
            experiment,
            look_id="look_two_additional",
            prior_decision=bundle["sequential-decision.json"],
            prior_receipt=receipt,
        )
        assert json.loads(look_two.manifest_json)["look_id"] == "look_two_additional"
    finally:
        _unlock_tree(final_root)


def test_post_rename_failure_quarantines_invalid_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = _make_experiment(tmp_path)
    bundle = _complete_look(experiment)
    target = tmp_path / "must-not-survive"

    def fail_readback(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise ValueError("injected post-rename failure")

    monkeypatch.setattr(review, "_validate_final_look_receipt", fail_readback)
    with pytest.raises(ValueError, match="injected post-rename failure"):
        _finalize(experiment, bundle, target)
    assert not target.exists()
    assert not list(tmp_path.glob(".must-not-survive.staging-*"))
    quarantines = list(tmp_path.glob(".must-not-survive.quarantine-*"))
    assert len(quarantines) == 1
    _unlock_tree(quarantines[0])
    shutil.rmtree(quarantines[0])


def test_preentropy_member_builder_requires_complete_exact_adapter_inventory(
    tmp_path: Path,
) -> None:
    experiment = _make_experiment(tmp_path)
    image_ids = experiment["selected_ids"][:2]
    ordered_image_ids = sorted(image_ids, key=int)
    stage = _stage_zero_fixture(
        tmp_path / "two-member-stage-zero",
        ordered_possible_image_ids=ordered_image_ids,
    )
    pool = experiment["pool"]
    frozen = review.load_frozen_census_records(
        census_path=experiment["census"],
        expected_census_path=experiment["census"],
        expected_census_sha256=_sha(experiment["census"].read_bytes()),
        selected_image_ids=image_ids,
    )
    census_path = tmp_path / "two-member-census.jsonl"
    census_path.write_bytes(
        "".join(
            json.dumps(frozen.records[image_id], sort_keys=True) + "\n"
            for image_id in image_ids
        ).encode()
    )
    adapter = _subset_adapter(experiment["adapter"], image_ids)
    artifacts = review.build_preentropy_member_manifest(
        adapter=adapter,
        ordered_image_ids=ordered_image_ids,
        stage_zero_root=stage["root"],
        expected_stage_zero_root_inventory_sha256=stage[
            "root_inventory_sha256"
        ],
        stage_zero_receipt_path=stage["receipt"],
        expected_stage_zero_receipt_sha256=stage["receipt_sha256"],
        stage_zero_audit_path=stage["audit"],
        expected_stage_zero_audit_sha256=stage["audit_sha256"],
        possible_pool_path=stage["possible_pool"],
        expected_possible_pool_sha256=stage["possible_pool_sha256"],
        candidate_pool_path=pool,
        expected_candidate_pool_sha256=_sha(pool.read_bytes()),
        census_path=census_path,
        expected_census_path=census_path,
        expected_census_sha256=_sha(census_path.read_bytes()),
        intended_member_manifest_path=tmp_path / "out" / "member-manifest.jsonl",
        frozen_contract_path=experiment["contract"],
        expected_frozen_contract_sha256=_sha(experiment["contract"].read_bytes()),
        source_panel_root=experiment["source_panel"],
        sampled_panel_root=experiment["sampled_panel"],
    )
    assert len(_decode_jsonl(pool.read_bytes())) > len(ordered_image_ids)
    assert len(_decode_jsonl(artifacts.member_manifest_jsonl)) == 2
    member_receipt = json.loads(artifacts.member_manifest_receipt_json)
    assert member_receipt["built_before_entropy"]
    assert member_receipt["stage_zero_possible_pool_count"] == 2
    assert member_receipt["stage_zero_possible_pool_sha256"] == stage[
        "possible_pool_sha256"
    ]

    wrong_audit = json.loads(stage["audit"].read_bytes())
    wrong_audit["ordered_possible_image_ids_sha256"] = "0" * 64
    wrong_audit_path = tmp_path / "approved-wrong-stage-zero-audit.json"
    wrong_audit_path.write_bytes(_pretty_json(wrong_audit))
    with pytest.raises(ValueError, match="audit binding drift"):
        review._load_stage_zero_possible_pool(
            stage_zero_root=stage["root"],
            expected_stage_zero_root_inventory_sha256=stage[
                "root_inventory_sha256"
            ],
            stage_zero_receipt_path=stage["receipt"],
            expected_stage_zero_receipt_sha256=stage["receipt_sha256"],
            stage_zero_audit_path=wrong_audit_path,
            expected_stage_zero_audit_sha256=_sha(wrong_audit_path.read_bytes()),
            possible_pool_path=stage["possible_pool"],
            expected_possible_pool_sha256=stage["possible_pool_sha256"],
        )

    missing_pool = tmp_path / "candidate-pool-missing-member.jsonl"
    missing_pool.write_bytes(
        _jsonl(
            [
                row
                for row in _decode_jsonl(pool.read_bytes())
                if str(row["image_id"]) != ordered_image_ids[0]
            ]
        )
    )
    with pytest.raises(ValueError, match="not a candidate-pool subset"):
        review.build_preentropy_member_manifest(
            adapter=adapter,
            ordered_image_ids=ordered_image_ids,
            stage_zero_root=stage["root"],
            expected_stage_zero_root_inventory_sha256=stage[
                "root_inventory_sha256"
            ],
            stage_zero_receipt_path=stage["receipt"],
            expected_stage_zero_receipt_sha256=stage["receipt_sha256"],
            stage_zero_audit_path=stage["audit"],
            expected_stage_zero_audit_sha256=stage["audit_sha256"],
            possible_pool_path=stage["possible_pool"],
            expected_possible_pool_sha256=stage["possible_pool_sha256"],
            candidate_pool_path=missing_pool,
            expected_candidate_pool_sha256=_sha(missing_pool.read_bytes()),
            census_path=census_path,
            expected_census_path=census_path,
            expected_census_sha256=_sha(census_path.read_bytes()),
            intended_member_manifest_path=tmp_path / "missing" / "member.jsonl",
            frozen_contract_path=experiment["contract"],
            expected_frozen_contract_sha256=_sha(
                experiment["contract"].read_bytes()
            ),
            source_panel_root=experiment["source_panel"],
            sampled_panel_root=experiment["sampled_panel"],
        )

    extra_adapter = _subset_adapter(experiment["adapter"], image_ids)
    extra_id = experiment["selected_ids"][2]
    for field in ("reference_records", "image_results"):
        extra_adapter[field][extra_id] = experiment["adapter"][field][extra_id]
    extra_adapter["source_rows"][(extra_id, 0)] = experiment["adapter"]["source_rows"][
        (extra_id, 0)
    ]
    for index in range(16):
        extra_adapter["sampled_rows"][(extra_id, index)] = experiment["adapter"][
            "sampled_rows"
        ][(extra_id, index)]
    with pytest.raises(review.AssemblyError, match="exact image inventory"):
        review.build_preentropy_member_manifest(
            adapter=extra_adapter,
            ordered_image_ids=ordered_image_ids,
            stage_zero_root=stage["root"],
            expected_stage_zero_root_inventory_sha256=stage[
                "root_inventory_sha256"
            ],
            stage_zero_receipt_path=stage["receipt"],
            expected_stage_zero_receipt_sha256=stage["receipt_sha256"],
            stage_zero_audit_path=stage["audit"],
            expected_stage_zero_audit_sha256=stage["audit_sha256"],
            possible_pool_path=stage["possible_pool"],
            expected_possible_pool_sha256=stage["possible_pool_sha256"],
            candidate_pool_path=pool,
            expected_candidate_pool_sha256=_sha(pool.read_bytes()),
            census_path=census_path,
            expected_census_path=census_path,
            expected_census_sha256=_sha(census_path.read_bytes()),
            intended_member_manifest_path=tmp_path / "other" / "member.jsonl",
            frozen_contract_path=experiment["contract"],
            expected_frozen_contract_sha256=_sha(experiment["contract"].read_bytes()),
            source_panel_root=experiment["source_panel"],
            sampled_panel_root=experiment["sampled_panel"],
        )
