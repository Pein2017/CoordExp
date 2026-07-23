"""Focused contracts for explicit duplicate-review materialization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.research.assemble_duplicate_trajectory_state_banks import (
    SOURCE_ONLY,
    build_duplicate_trajectory_bank_drafts,
    state_bank_rows_from_duplicate_draft_bank,
)
from scripts.research.materialize_reviewed_duplication_training_inputs import (
    MaterializationError,
    build_reviewed_duplication_training_inputs,
    write_reviewed_duplication_training_inputs,
)
from src.data.geometry import validate_bbox_bins


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _tokens(index: int) -> list[int]:
    return [151646, 100 + index, 151647, 151648, 151670 + index, 151671 + index, 151690 + index, 151691 + index, 151649]


def _write(path: Path, value: object, *, jsonl: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if jsonl:
        path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in value), encoding="utf-8")
    else:
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _exact(tmp_path: Path) -> Path:
    prompt = [1, 151655, 151655, 2]
    generated = [token for index in range(4) for token in _tokens(index)]
    boxes = [[0, 0, 10, 10], [0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10]]
    return _write(
        tmp_path / "greedy.json",
        {
            "schema_version": "current_seeded_sampled_rollouts.v1",
            "config": {"decode_mode": "greedy", "temperature": 0.0, "top_p": 1.0, "repetition_penalty": 1.0},
            "model_identity": {"synthetic": "model"},
            "prompt_metadata": {"1": {"image_path": "/synthetic/1.jpg", "image_sha256": "b" * 64, "width": 32, "height": 32}},
            "rollouts": [
                {
                    "image_id": "1",
                    "decode_mode": "greedy",
                    "seed": 0,
                    "prompt_token_ids": prompt,
                    "prompt_token_ids_sha256": _hash(prompt),
                    "generated_token_ids": generated,
                    "generated_token_ids_sha256": _hash(generated),
                    "predictions": {"predictions": [{"generated_order": index, "object_span_id": f"span-{index}", "description": "person", "bbox": box} for index, box in enumerate(boxes)]},
                }
            ],
        },
    )


def _union(tmp_path: Path) -> Path:
    boxes = [[0, 0, 10, 10], [0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10]]
    receipts = [
        {"image_id": "1", "trajectory_id": "greedy", "decode_mode": "greedy", "generated_row_index": 0, "prediction_id": "span-0", "bbox": boxes[0], "category": "person", "entity_status": "verified_owner", "owner_id": "1:a", "owner_category": "person", "owner_bbox": boxes[0]},
        {"image_id": "1", "trajectory_id": "greedy", "decode_mode": "greedy", "generated_row_index": 1, "prediction_id": "span-1", "bbox": boxes[1], "category": "person", "entity_status": "duplicate", "candidate_owner_id": "1:a", "candidate_owner_iou": 1.0},
        {"image_id": "1", "trajectory_id": "greedy", "decode_mode": "greedy", "generated_row_index": 2, "prediction_id": "span-2", "bbox": boxes[2], "category": "person", "entity_status": "verified_owner", "owner_id": "1:b", "owner_category": "person", "owner_bbox": boxes[2]},
        {"image_id": "1", "trajectory_id": "greedy", "decode_mode": "greedy", "generated_row_index": 3, "prediction_id": "span-3", "bbox": boxes[3], "category": "person", "entity_status": "unmatched"},
    ]
    return _write(tmp_path / "union.json", {"schema_version": "individual_trajectory_union_support.v1", "image_results": [{"image_id": "1", "owners": [{"owner_id": "1:a", "category": "person", "bbox": boxes[0]}, {"owner_id": "1:b", "category": "person", "bbox": boxes[2]}], "budgets": [{"budget": 16, "row_assignment_receipts": receipts}]}]})


def _source_pairs(tmp_path: Path) -> tuple[Path, Path]:
    prompt = [1, 151655, 151655, 2]
    candidate = {"candidate_id": "source-candidate", "token_ids": _tokens(0), "token_ids_sha256": _hash(_tokens(0)), "generation_provenance": {"mode": "greedy", "seed": 0, "temperature": 0.0, "top_p": 1.0, "repetition_penalty": 1.0, "checkpoint_id": "a" * 64, "prompt_token_ids_sha256": _hash(prompt), "prefix_token_ids_sha256": _hash([])}}
    rollout = {"event_id": "source-1", "image": {"image_id": "1", "path": "/synthetic/1.jpg", "content_sha256": "b" * 64, "width": 32, "height": 32}, "executed_prompt_token_ids": prompt, "executed_prompt_token_ids_sha256": _hash(prompt), "image_pad_interval": [1, 3], "prefix_token_ids": [], "prefix_token_ids_sha256": _hash([]), "candidates": [candidate]}
    review = {"event_id": "source-1", "physical_entities": [{"entity_id": "1:a", "category": "person", "entity_trusted": True, "geometry_trusted": True, "reference_bbox": [0, 0, 10, 10]}], "review_provenance": {"event_family": "source_preservation"}}
    return _write(tmp_path / "source-rollout.jsonl", [rollout], jsonl=True), _write(tmp_path / "source-review.jsonl", [review], jsonl=True)


def _queue(tmp_path: Path, exact: Path) -> Path:
    exact_hash = _hashlib_file(exact)
    candidate = {
        "schema_version": "physical_owner_duplication_review_queue.v1",
        "candidate_id": "candidate-1",
        "candidate_status": "high_confidence_candidate_repeated_physical_owner",
        "not_a_training_label": True,
        "image_id": "1",
        "trajectory_id": "greedy",
        "physical_owner_id": "1:a",
        "earlier_accepted_owner_row": {"row_index": 0, "prediction_id": "span-0", "prediction_bbox": [0, 0, 10, 10]},
        "duplicate_rows": [{"row_index": 1, "prediction_id": "span-1", "prediction_bbox": [0, 0, 10, 10], "category": "person"}],
        "first_later_trusted_unseen_verified_owner_row": {"row_index": 2, "prediction_id": "span-2", "prediction_bbox": [20, 0, 30, 10], "owner_id": "1:b", "category": "person"},
        "exact_rollout": {"source_path": str(exact.resolve()), "source_sha256": exact_hash, "generated_token_ids_sha256": _hash([token for index in range(4) for token in _tokens(index)])},
    }
    return _write(tmp_path / "queue.jsonl", [candidate], jsonl=True)


def _hashlib_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(tmp_path: Path) -> dict[str, Path]:
    exact = _exact(tmp_path)
    union = _union(tmp_path)
    source_rollout, source_review = _source_pairs(tmp_path)
    queue = _queue(tmp_path, exact)
    decisions = _write(tmp_path / "decisions.jsonl", [{"candidate_id": "candidate-1", "decision": "approve", "reviewer": "main-agent", "comment": "Crop review approves the duplicate and immediate recovery.", "evidence": "synthetic reviewed case"}], jsonl=True)
    return {"exact": exact, "union": union, "source_rollout": source_rollout, "source_review": source_review, "queue": queue, "decisions": decisions}


def test_materializes_only_explicit_approval_into_assembler_inputs(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    result = build_reviewed_duplication_training_inputs(candidate_queue_jsonl=paths["queue"], approved_decisions_jsonl=paths["decisions"], union_support_json=paths["union"], greedy_json_paths=[paths["exact"]], source_rollout_jsonl=paths["source_rollout"], source_review_jsonl=paths["source_review"])

    assert len(result["rollout_trajectories"]) == 1
    assert [row["review_role"] for row in result["reviewed_owner_ledger"]] == ["accepted", "duplicate", "recovery", "neutral"]
    assert result["source_preservation_events"][0]["event_id"] == "source-1"
    assert result["receipt"]["approval_authority"] == "explicit approved-decisions JSONL only"
    source_pair_rollout = json.loads(paths["source_rollout"].read_text().splitlines()[0])
    source_pair_review = json.loads(paths["source_review"].read_text().splitlines()[0])
    source_event = result["source_preservation_events"][0]
    assert source_event["state_bank_rollout"] == source_pair_rollout
    assert source_event["state_bank_review"] == source_pair_review
    assert result["rollout_trajectories"][0]["physical_entities"] == source_pair_review["physical_entities"]
    drafts = build_duplicate_trajectory_bank_drafts(**{key: result[key] for key in ("rollout_trajectories", "reviewed_owner_ledger", "source_preservation_events")})
    assert drafts["allocation_receipt"]["admitted_burst_count"] == 1
    source_rollouts, source_reviews = state_bank_rows_from_duplicate_draft_bank(
        drafts["banks"][SOURCE_ONLY]
    )
    assert source_rollouts == [source_pair_rollout]
    assert source_reviews == [
        {**source_pair_review, "image_balanced_event_weight": 1.0}
    ]

    targets = write_reviewed_duplication_training_inputs(result, tmp_path / "out")
    assert set(targets) == {"rollout_trajectories", "reviewed_owner_ledger", "source_preservation_events", "receipt"}
    assert targets["reviewed_owner_ledger"].read_text(encoding="utf-8").count("\n") == 4


def test_rejects_nonapproved_or_nonimmediate_recovery_without_inference(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    _write(tmp_path / "reject.jsonl", [{"candidate_id": "candidate-1", "decision": "reject", "reviewer": "main-agent", "comment": "not approved"}], jsonl=True)
    with pytest.raises(MaterializationError, match="decision=approve"):
        build_reviewed_duplication_training_inputs(candidate_queue_jsonl=paths["queue"], approved_decisions_jsonl=tmp_path / "reject.jsonl", union_support_json=paths["union"], greedy_json_paths=[paths["exact"]], source_rollout_jsonl=paths["source_rollout"], source_review_jsonl=paths["source_review"])

    rows = [json.loads(line) for line in paths["queue"].read_text().splitlines()]
    rows[0]["first_later_trusted_unseen_verified_owner_row"]["row_index"] = 3
    rows[0]["first_later_trusted_unseen_verified_owner_row"]["prediction_id"] = "span-3"
    rows[0]["first_later_trusted_unseen_verified_owner_row"]["prediction_bbox"] = [40, 0, 50, 10]
    rows[0]["first_later_trusted_unseen_verified_owner_row"]["owner_id"] = "1:b"
    bad_queue = _write(tmp_path / "bad-queue.jsonl", rows, jsonl=True)
    with pytest.raises(MaterializationError, match="immediately follow"):
        build_reviewed_duplication_training_inputs(candidate_queue_jsonl=bad_queue, approved_decisions_jsonl=paths["decisions"], union_support_json=paths["union"], greedy_json_paths=[paths["exact"]], source_rollout_jsonl=paths["source_rollout"], source_review_jsonl=paths["source_review"])


def test_accepts_one_preassembled_immutable_source_event_without_replication(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    source_events = _write(
        tmp_path / "source-events.jsonl",
        [{"event_id": "preassembled-source-1", "image_id": "2", "image": {"image_id": "2", "path": "/synthetic/2.jpg"}, "opaque_source_receipt": {"immutable": True}}],
        jsonl=True,
    )
    result = build_reviewed_duplication_training_inputs(candidate_queue_jsonl=paths["queue"], approved_decisions_jsonl=paths["decisions"], union_support_json=paths["union"], greedy_json_paths=[paths["exact"]], source_events_jsonl=[source_events])

    assert result["source_preservation_events"][0]["immutable_source_kind"] == "preassembled_source_event"
    assert result["source_preservation_events"][0]["source_match_type"] == "global_fallback"
    assert result["rollout_trajectories"][0]["image"]["image_id"] == 1
    assert result["receipt"]["source_events_unique"] is True


def test_normalizes_fractional_absolute_pixel_union_boxes_for_global_source_fallback(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    exact = json.loads(paths["exact"].read_text())
    exact["prompt_metadata"]["1"].update({"width": 1152, "height": 640})
    _write(paths["exact"], exact)
    queue = [json.loads(line) for line in paths["queue"].read_text().splitlines()]
    queue[0]["exact_rollout"]["source_sha256"] = _hashlib_file(paths["exact"])
    _write(paths["queue"], queue, jsonl=True)
    union = json.loads(paths["union"].read_text())
    owners = union["image_results"][0]["owners"]
    owners[0]["bbox"] = [787.968, 128.0, 1000.1, 499.9]
    owners[1]["bbox"] = [-5.0, 0.0, 1200.0, 640.0]
    _write(paths["union"], union)
    source_events = _write(
        tmp_path / "source-events.jsonl",
        [
            {
                "event_id": "preassembled-source-2",
                "image_id": "2",
                "image": {"image_id": "2", "path": "/synthetic/2.jpg"},
                "opaque_source_receipt": {"immutable": True},
            }
        ],
        jsonl=True,
    )

    result = build_reviewed_duplication_training_inputs(
        candidate_queue_jsonl=paths["queue"],
        approved_decisions_jsonl=paths["decisions"],
        union_support_json=paths["union"],
        greedy_json_paths=[paths["exact"]],
        source_events_jsonl=[source_events],
    )

    entities = result["rollout_trajectories"][0]["physical_entities"]
    assert entities[0]["reference_bbox"] == [684, 200, 868, 781]
    assert entities[1]["reference_bbox"] == [0, 0, 999, 999]
    assert validate_bbox_bins(entities[0]["reference_bbox"], field="entity[0]") == (
        684,
        200,
        868,
        781,
    )
    assert "source_bbox=[787.968, 128.0, 1000.1, 499.9]" in entities[0]["comment"]
    assert "1152x640" in entities[0]["comment"]


def test_retains_nonverified_preburst_context_as_neutral_without_revoking_approval(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    exact = json.loads(paths["exact"].read_text())
    predictions = exact["rollouts"][0]["predictions"]["predictions"]
    predictions[2]["bbox"] = [0, 0, 10, 10]
    predictions[3]["bbox"] = [20, 0, 30, 10]
    _write(paths["exact"], exact)
    union = json.loads(paths["union"].read_text())
    receipts = union["image_results"][0]["budgets"][0]["row_assignment_receipts"]
    receipts[1] = {**receipts[1], "entity_status": "unmatched"}
    for key in ("candidate_owner_id", "candidate_owner_iou"):
        receipts[1].pop(key, None)
    receipts[2] = {**receipts[2], "entity_status": "duplicate", "bbox": [0, 0, 10, 10], "candidate_owner_id": "1:a", "candidate_owner_iou": 1.0}
    for key in ("owner_id", "owner_category", "owner_bbox"):
        receipts[2].pop(key, None)
    receipts[3] = {**receipts[3], "entity_status": "verified_owner", "bbox": [20, 0, 30, 10], "owner_id": "1:b", "owner_category": "person", "owner_bbox": [20, 0, 30, 10]}
    _write(paths["union"], union)
    queue = [json.loads(line) for line in paths["queue"].read_text().splitlines()]
    queue[0]["duplicate_rows"] = [{"row_index": 2, "prediction_id": "span-2", "prediction_bbox": [0, 0, 10, 10], "category": "person"}]
    queue[0]["first_later_trusted_unseen_verified_owner_row"] = {"row_index": 3, "prediction_id": "span-3", "prediction_bbox": [20, 0, 30, 10], "owner_id": "1:b", "category": "person"}
    queue[0]["exact_rollout"]["source_sha256"] = _hashlib_file(paths["exact"])
    _write(paths["queue"], queue, jsonl=True)

    result = build_reviewed_duplication_training_inputs(candidate_queue_jsonl=paths["queue"], approved_decisions_jsonl=paths["decisions"], union_support_json=paths["union"], greedy_json_paths=[paths["exact"]], source_rollout_jsonl=paths["source_rollout"], source_review_jsonl=paths["source_review"])
    assert [row["review_role"] for row in result["reviewed_owner_ledger"]] == ["accepted", "neutral", "duplicate", "recovery"]


def test_namespaces_same_local_trajectory_id_across_images_for_assembler(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)

    exact = json.loads(paths["exact"].read_text())
    second_rollout = json.loads(json.dumps(exact["rollouts"][0]))
    second_rollout["image_id"] = "2"
    exact["rollouts"].append(second_rollout)
    exact["prompt_metadata"]["2"] = {
        "image_path": "/synthetic/2.jpg",
        "image_sha256": "c" * 64,
        "width": 32,
        "height": 32,
    }
    _write(paths["exact"], exact)

    queue = [json.loads(line) for line in paths["queue"].read_text().splitlines()]
    queue[0]["exact_rollout"]["source_sha256"] = _hashlib_file(paths["exact"])
    second_candidate = json.loads(json.dumps(queue[0]))
    second_candidate.update(
        {
            "candidate_id": "candidate-2",
            "image_id": "2",
            "physical_owner_id": "2:a",
            "earlier_accepted_owner_row": {
                "row_index": 0,
                "prediction_id": "span-0",
                "prediction_bbox": [0, 0, 10, 10],
            },
            "first_later_trusted_unseen_verified_owner_row": {
                "row_index": 2,
                "prediction_id": "span-2",
                "prediction_bbox": [20, 0, 30, 10],
                "owner_id": "2:b",
                "category": "person",
            },
        }
    )
    queue.append(second_candidate)
    _write(paths["queue"], queue, jsonl=True)

    union = json.loads(paths["union"].read_text())
    second_image = json.loads(json.dumps(union["image_results"][0]))
    second_image["image_id"] = "2"
    for owner in second_image["owners"]:
        owner["owner_id"] = owner["owner_id"].replace("1:", "2:")
    for receipt in second_image["budgets"][0]["row_assignment_receipts"]:
        receipt["image_id"] = "2"
        for key in ("owner_id", "candidate_owner_id"):
            if key in receipt:
                receipt[key] = receipt[key].replace("1:", "2:")
    union["image_results"].append(second_image)
    _write(paths["union"], union)

    source_rollouts = [json.loads(line) for line in paths["source_rollout"].read_text().splitlines()]
    second_source_rollout = json.loads(json.dumps(source_rollouts[0]))
    second_source_rollout["event_id"] = "source-2"
    second_source_rollout["image"] = {
        **second_source_rollout["image"],
        "image_id": "2",
        "path": "/synthetic/2.jpg",
        "content_sha256": "c" * 64,
    }
    source_rollouts.append(second_source_rollout)
    _write(paths["source_rollout"], source_rollouts, jsonl=True)
    source_reviews = [json.loads(line) for line in paths["source_review"].read_text().splitlines()]
    second_source_review = json.loads(json.dumps(source_reviews[0]))
    second_source_review["event_id"] = "source-2"
    second_source_review["physical_entities"][0]["entity_id"] = "2:a"
    source_reviews.append(second_source_review)
    _write(paths["source_review"], source_reviews, jsonl=True)

    decisions = [json.loads(line) for line in paths["decisions"].read_text().splitlines()]
    decisions.append(
        {
            "candidate_id": "candidate-2",
            "decision": "approve",
            "reviewer": "main-agent",
            "comment": "Second image explicitly approved.",
        }
    )
    _write(paths["decisions"], decisions, jsonl=True)

    result = build_reviewed_duplication_training_inputs(
        candidate_queue_jsonl=paths["queue"],
        approved_decisions_jsonl=paths["decisions"],
        union_support_json=paths["union"],
        greedy_json_paths=[paths["exact"]],
        source_rollout_jsonl=paths["source_rollout"],
        source_review_jsonl=paths["source_review"],
    )

    assert [item["trajectory_id"] for item in result["rollout_trajectories"]] == ["1:greedy", "2:greedy"]
    assert {item["trajectory_id"] for item in result["reviewed_owner_ledger"]} == {"1:greedy", "2:greedy"}
    assert {item["trajectory_id"] for item in result["source_preservation_events"]} == {"1:greedy", "2:greedy"}
    assert {
        item["source_burst_key"]["trajectory_id"]
        for item in result["source_preservation_events"]
    } == {"1:greedy", "2:greedy"}
    assert {
        item["review_provenance"]["local_trajectory_id"]
        for item in result["reviewed_owner_ledger"]
    } == {"greedy"}
    assert {
        item["materialization_provenance"]["local_trajectory_id"]
        for item in result["rollout_trajectories"]
    } == {"greedy"}
    drafts = build_duplicate_trajectory_bank_drafts(
        **{
            key: result[key]
            for key in (
                "rollout_trajectories",
                "reviewed_owner_ledger",
                "source_preservation_events",
            )
        }
    )
    assert drafts["allocation_receipt"]["admitted_burst_count"] == 2
