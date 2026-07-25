from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from scripts.research import prepare_transition_phase0_candidate_scoring as phase0


def _row(description: list[int], coordinate_offset: int = 0) -> list[int]:
    scorer = phase0._canonical_scorer
    return [
        scorer.OBJECT_REF_START,
        *description,
        scorer.OBJECT_REF_END,
        scorer.BOX_START,
        *[
            scorer.COORDINATE_TOKEN_START + coordinate_offset + index
            for index in range(4)
        ],
        scorer.BOX_END,
    ]


def _candidate(
    *,
    event_id: str,
    candidate_id: str,
    token_ids: list[int],
    prompt_hash: str,
    prefix_hash: str,
    role: str = "positive",
    owner: str | None = "100157:567184",
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "coverage_status": "uncovered" if role == "positive" else "unknown",
        "entity_eligible": True,
        "entity_review_status": "trusted",
        "generation_provenance": {
            "checkpoint_id": "checkpoint-source",
            "mode": "sampled" if role == "positive" else "greedy",
            "prefix_token_ids_sha256": prefix_hash,
            "prompt_token_ids_sha256": prompt_hash,
            "repetition_penalty": 1.0,
            "seed": 1 if role == "positive" else 0,
            "temperature": 0.4 if role == "positive" else 0.0,
            "top_p": 0.95 if role == "positive" else 1.0,
        },
        "harmful_kind": None if role == "positive" else "premature_terminal",
        "physical_owner_id": owner,
        "role": role,
        "selected_sites": [{"candidate_token_offset": 0, "intended_token_type": "schema"}],
        "token_ids": token_ids,
        "token_ids_sha256": phase0.sha256_json(token_ids),
    }


def _record(
    event_id: str,
    *,
    image_id: int,
    description_tokens: list[int],
    aliases: int = 0,
) -> dict[str, Any]:
    prompt = [101, 102, image_id % 97]
    prefix = [phase0._canonical_scorer.OBJECT_REF_START, 700 + image_id % 17]
    prompt_hash = phase0.sha256_json(prompt)
    prefix_hash = phase0.sha256_json(prefix)
    positive_row = _row(description_tokens)
    common_id = f"{event_id}-positive-00-{phase0.sha256_json(positive_row)[:12]}"
    stop = _candidate(
        event_id=event_id,
        candidate_id=f"{event_id}-constructed-stop",
        token_ids=[phase0.STOP_TOKEN_ID],
        prompt_hash=prompt_hash,
        prefix_hash=prefix_hash,
        role="harmful",
        owner=None,
    )
    common = _candidate(
        event_id=event_id,
        candidate_id=common_id,
        token_ids=positive_row,
        prompt_hash=prompt_hash,
        prefix_hash=prefix_hash,
    )
    candidates = [stop, common]
    for alias_index in range(aliases):
        alias_row = _row([*description_tokens, 800 + alias_index], coordinate_offset=10)
        candidates.append(
            _candidate(
                event_id=event_id,
                candidate_id=(
                    f"{event_id}-positive-{alias_index + 1:02d}-"
                    f"{phase0.sha256_json(alias_row)[:12]}"
                ),
                token_ids=alias_row,
                prompt_hash=prompt_hash,
                prefix_hash=prefix_hash,
            )
        )
    return {
        "candidates": candidates,
        "event_id": event_id,
        "event_family": "entity_transition",
        "executed_prompt_token_ids": prompt,
        "executed_prompt_token_ids_sha256": prompt_hash,
        "image": {
            "content_sha256": f"image-sha-{image_id}",
            "height": 100,
            "image_id": image_id,
            "path": f"/tmp/{image_id}.jpg",
            "width": 120,
        },
        "physical_entities": [
            {
                "category": "person",
                "entity_id": "100157:567184",
                "entity_trusted": True,
                "geometry_trusted": True,
            }
        ],
        "prefix_covered_owner_proofs": [],
        "prefix_object_row_count": 1,
        "prefix_token_ids": prefix,
        "prefix_token_ids_sha256": prefix_hash,
        "review_provenance": {
            "constructed_stop": True,
            "constructed_stop_is_not_sampled_or_on_policy": True,
            "positive_alias_count": 1 + aliases,
        },
        "split": "train",
        "split_group_id": f"image:{image_id}",
    }


def _write_bank(path: Path, records: list[dict[str, Any]]) -> None:
    path.mkdir(parents=True)
    records_bytes = (
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for record in records
        )
    ).encode("utf-8")
    (path / "records.jsonl").write_bytes(records_bytes)
    manifest = {
        "bank_id": path.name,
        "blind_image_ids": [999],
        "event_family_counts": {"entity_transition": len(records)},
        "prompt_identity_sha256": "prompt-identity",
        "record_count": len(records),
        "records_file": "records.jsonl",
        "records_sha256": hashlib.sha256(records_bytes).hexdigest(),
        "schema_version": phase0.STATE_BANK_SCHEMA_VERSION,
        "source_checkpoint": {"tokenizer_sha256": "tokenizer"},
        "source_checkpoint_id": "checkpoint-source",
        "split_assignments": [
            {
                "image_content_sha256": record["image"]["content_sha256"],
                "image_id": record["image"]["image_id"],
                "split": "train",
                "split_group_id": record["split_group_id"],
            }
            for record in records
        ],
    }
    (path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _banks(
    tmp_path: Path,
    *,
    owner_mutator: Callable[[list[dict[str, Any]]], None] | None = None,
) -> tuple[Path, Path, dict[str, int]]:
    event_ids = [phase0.SMOKE_EVENT_ID, "event-second"]
    pair_records = [
        _record(event_ids[0], image_id=100157, description_tokens=[900]),
        _record(event_ids[1], image_id=200001, description_tokens=[901, 902]),
    ]
    owner_records = copy.deepcopy(pair_records)
    owner_records[1] = _record(
        event_ids[1], image_id=200001, description_tokens=[901, 902], aliases=1
    )
    if owner_mutator is not None:
        owner_mutator(owner_records)
    pair_path = tmp_path / "pairwise"
    owner_path = tmp_path / "owner-conditioned"
    _write_bank(pair_path, pair_records)
    _write_bank(owner_path, owner_records)
    expected = {
        "pairwise_event_count": 2,
        "owner_conditioned_event_count": 2,
        "common_singleton_action_count": 2,
        "pairwise_positive_action_count": 2,
        "owner_conditioned_positive_action_count": 3,
        "owner_conditioned_extra_alias_action_count": 1,
        "owner_conditioned_extra_alias_event_count": 1,
    }
    return pair_path, owner_path, expected


def _rehash_record_tokens(record: dict[str, Any], key: str, hash_key: str) -> None:
    new_hash = phase0.sha256_json(record[key])
    record[hash_key] = new_hash
    provenance_key = (
        "prompt_token_ids_sha256" if key == "executed_prompt_token_ids" else "prefix_token_ids_sha256"
    )
    for candidate in record["candidates"]:
        candidate["generation_provenance"][provenance_key] = new_hash


def test_one_event_lineage_smoke_emits_canonical_manifest_and_alias_audit(
    tmp_path: Path,
) -> None:
    pair_path, owner_path, expected = _banks(tmp_path)
    manifest, audit = phase0.build_common_projection(
        pair_path,
        owner_path,
        selected_event_ids=[phase0.SMOKE_EVENT_ID],
        expected_population=expected,
    )

    assert audit["population"]["common_singleton_action_count"] == 2
    assert audit["population"]["owner_conditioned_extra_alias_action_count"] == 1
    assert audit["selection"]["event_ids"] == [phase0.SMOKE_EVENT_ID]
    assert audit["selection"]["common_action_length_strata"] == {
        "9": 1,
        "10": 0,
        "11": 0,
    }
    event = audit["events"][0]
    assert event["constructed_stop_token_id"] == phase0.STOP_TOKEN_ID
    assert event["physical_owner_id"] == "100157:567184"
    assert manifest["projection_contract"]["common_singleton_only"] is True
    assert len(manifest["images"]) == 1
    boundary = manifest["images"][0]["boundaries"][0]
    assert boundary["boundary_id"] == phase0.SMOKE_EVENT_ID
    assert len(boundary["candidates"]) == 1
    phase0._canonical_scorer.validate_manifest(
        manifest, manifest_path=tmp_path / phase0.PREPARED_MANIFEST_NAME
    )


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda records: (
                records[0]["executed_prompt_token_ids"].append(999),
                _rehash_record_tokens(
                    records[0],
                    "executed_prompt_token_ids",
                    "executed_prompt_token_ids_sha256",
                ),
            ),
            "prompt mismatch",
        ),
        (
            lambda records: (
                records[0]["prefix_token_ids"].append(998),
                _rehash_record_tokens(
                    records[0], "prefix_token_ids", "prefix_token_ids_sha256"
                ),
            ),
            "prefix mismatch",
        ),
        (
            lambda records: records[0].update({"event_id": "different-event"}),
            "event mismatch",
        ),
        (
            lambda records: records[0]["candidates"][1].update(
                {
                    "token_ids": _row([777]),
                    "token_ids_sha256": phase0.sha256_json(_row([777])),
                }
            ),
            "common-action mismatch",
        ),
        (
            lambda records: records[0]["candidates"][0].update(
                {
                    "token_ids": [123],
                    "token_ids_sha256": phase0.sha256_json([123]),
                }
            ),
            "constructed-stop mismatch",
        ),
        (
            lambda records: records[0].update(
                {"prefix_token_ids_sha256": "0" * 64}
            ),
            "hash mismatch",
        ),
    ],
)
def test_projection_fails_closed_on_lineage_mismatch(
    tmp_path: Path,
    mutator: Callable[[list[dict[str, Any]]], None],
    message: str,
) -> None:
    pair_path, owner_path, expected = _banks(tmp_path, owner_mutator=mutator)
    with pytest.raises(ValueError, match=message):
        phase0.build_common_projection(
            pair_path, owner_path, expected_population=expected
        )


def _score(row: list[int], values: list[float]) -> dict[str, Any]:
    assert len(row) == len(values)
    phases = phase0._canonical_row_phases(row)
    score: dict[str, Any] = {
        "candidate_id": "candidate",
        "owner": "100157:567184",
        "row_token_ids": row,
        "token_count": len(row),
        "token_ids_sha256": phase0.sha256_json(row),
        "token_log_probabilities": values,
    }
    for phase_name, indices in phases.items():
        selected = [values[index] for index in indices]
        score[phase_name] = {
            "sum": sum(selected),
            "mean": sum(selected) / len(selected),
            "count": len(selected),
        }
    return score


@pytest.mark.parametrize(("description", "expected_schema_count"), [([11], 5), ([11, 12], 6), ([11, 12, 13], 7)])
def test_reducer_phase_counts_cover_lengths_9_10_11(
    description: list[int], expected_schema_count: int
) -> None:
    row = _row(description)
    reduced = phase0.reduce_candidate_score(_score(row, [-1.0] * len(row)))
    assert reduced["denominators"] == {
        "target_token_count": len(row),
        "schema_description_token_count": expected_schema_count,
        "coordinate_token_count": 4,
    }
    assert reduced["phase_counts"]["schema_description"] == expected_schema_count
    assert reduced["phase_counts"]["coordinate"] == 4
    assert reduced["phase_counts"]["full_row"] == len(row)


def test_equal_group_diagnostic_differs_from_token_weighted_mean() -> None:
    row = _row([11])
    # Five schema/description tokens at -1 and four coordinates at -3.
    coordinate_indices = {
        phase0._canonical_row_phases(row)[axis][0]
        for axis in ("x1", "y1", "x2", "y2")
    }
    values = [-3.0 if index in coordinate_indices else -1.0 for index in range(len(row))]
    reduced = phase0.reduce_candidate_score(_score(row, values))

    assert reduced["target_token_mean_log_probability"] == pytest.approx(-17.0 / 9.0)
    assert reduced["equal_group_mean_log_probability"] == pytest.approx(-2.0)
    assert reduced["equal_group_mean_log_probability"] != pytest.approx(
        reduced["target_token_mean_log_probability"]
    )


def test_aggregate_is_equal_event_weighted_and_keeps_declared_denominators() -> None:
    short = phase0.reduce_candidate_score(_score(_row([11]), [-1.0] * 9))
    long = phase0.reduce_candidate_score(_score(_row([11, 12, 13]), [-3.0] * 11))
    aggregate = phase0.aggregate_row_normalization([short, long])

    assert aggregate["aggregation"] == "equal_event_weighted_mean"
    assert aggregate["equal_event_weighted_means"][
        "target_token_mean_log_probability"
    ] == pytest.approx(-2.0)
    assert aggregate["denominators"] == {
        "event_count": 2,
        "target_token_count": 20,
        "schema_description_token_count": 12,
        "coordinate_token_count": 8,
    }


def test_receipt_reduction_keeps_stop_separate_and_emits_all_length_strata(
    tmp_path: Path,
) -> None:
    pair_path, owner_path, expected = _banks(tmp_path)
    manifest, audit = phase0.build_common_projection(
        pair_path,
        owner_path,
        selected_event_ids=[phase0.SMOKE_EVENT_ID],
        expected_population=expected,
    )
    audit_event = audit["events"][0]
    candidate_manifest = manifest["images"][0]["boundaries"][0]["candidates"][0]
    row = candidate_manifest["row"]["token_ids"]
    score = _score(row, [-1.0] * len(row))
    score["candidate_id"] = audit_event["candidate_id"]
    score["owner"] = audit_event["physical_owner_id"]
    receipt = {
        "schema_version": phase0._canonical_scorer.RECEIPT_SCHEMA_VERSION,
        "manifest": {"path": "/tmp/manifest.json", "sha256": audit["scorer_manifest"]["sha256"]},
        "runtime": {"repetition_penalty_processing": False},
        "images": [
            {
                "image_id": audit_event["image_id"],
                "boundaries": [
                    {
                        "boundary_id": audit_event["event_id"],
                        "prefix_token_ids_sha256": audit_event[
                            "full_model_prefix_token_ids_sha256"
                        ],
                        "terminal_boundary": {
                            "row_entry_token_id": phase0.OBJECT_REF_START,
                            "terminal_token_id": phase0.STOP_TOKEN_ID,
                            "row_entry_log_probability": -1.0,
                            "terminal_log_probability": -2.0,
                            "row_entry_minus_terminal": 1.0,
                        },
                        "candidate_scores": [score],
                    }
                ],
            }
        ],
    }

    reduced = phase0.reduce_score_receipts(audit, {"source": [receipt]})
    arm = reduced["arms"]["source"]
    assert set(arm["length_strata"]) == {"9", "10", "11"}
    assert arm["length_strata"]["9"]["denominators"]["event_count"] == 1
    assert arm["length_strata"]["10"]["denominators"]["event_count"] == 0
    assert arm["stop_boundary"]["equal_event_weighted_means"][
        "row_entry_minus_stop_log_probability"
    ] == pytest.approx(1.0)
    assert "stop_boundary" not in arm["row_normalization"]
    assert "probability_distribution" not in json.dumps(reduced)
