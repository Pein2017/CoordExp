from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.research import run_exact_prefix_sampled_rescue as rescue


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(*, owner: str, token_ids: list[int] | None = None) -> dict[str, object]:
    ids = token_ids or [10, 11]
    return {
        "mode": "sample",
        "seed": 7,
        "status": "success",
        "raw_generated_token_ids": ids,
        "raw_generated_token_ids_sha256": rescue.hash_prefix_token_ids(ids),
        "row_stop": {"stop_reason": "complete_row"},
        "parse_evidence": {"parse_status": "success"},
        "entity_matches": [
            {
                "prediction_index": 0,
                "status": "matched",
                "matched_entity_id": owner,
            }
        ],
        "strict_matched_owner_ids": [owner],
        "unmatched_or_ambiguous_prediction_indices": [],
    }


def _root_trace(*, image_id: str = "7816", terminal: bool = False) -> dict[str, object]:
    prefix = [1, 2]
    if terminal:
        row = {
            "row_index": 3,
            "input_prefix_token_ids": prefix,
            "input_prefix_token_ids_sha256": rescue.hash_prefix_token_ids(prefix),
            "prefix_token_ids": prefix,
            "prefix_token_ids_sha256": rescue.hash_prefix_token_ids(prefix),
            "status": "success",
            "raw_generated_token_ids": [30, 31],
            "raw_generated_token_ids_sha256": rescue.hash_prefix_token_ids([30, 31]),
            "row_stop": {"stop_reason": "terminal"},
            "parsed_predictions": [],
            "strict_matched_owner_ids": [],
            "unmatched_or_ambiguous_prediction_indices": [],
        }
        covered = ["covered"]
    else:
        row = _row(owner="covered", token_ids=[30, 31])
        row.update(
            {
                "row_index": 3,
                "input_prefix_token_ids": prefix,
                "input_prefix_token_ids_sha256": rescue.hash_prefix_token_ids(prefix),
                "prefix_token_ids": prefix,
                "prefix_token_ids_sha256": rescue.hash_prefix_token_ids(prefix),
                "covered_owner_ids_before_row": ["covered"],
            }
        )
        covered = ["covered"]
    row["covered_owner_ids_before_row"] = covered
    return {
        "schema_version": rescue.ROOT_TRACE_SCHEMA_VERSION,
        "phase": rescue.ROOT_TRACE_PHASE,
        "images": [
            {
                "image_id": image_id,
                "prompt": {},
                "runtime": {},
                "entity_ledger": [
                    {"entity_id": "covered", "verification": "verified"},
                    {"entity_id": "uncovered", "verification": "verified"},
                ],
                "extended_root_greedy": {"rows": [row]},
            }
        ],
    }


def test_seed_parser_is_ordered_unique_and_nonnegative() -> None:
    assert rescue.parse_seeds("11, 7,19") == (11, 7, 19)
    for invalid in ("", "1,", "1,1", "-1", "one"):
        with pytest.raises(rescue.RescueValidationError):
            rescue.parse_seeds(invalid)


def test_seed_range_is_inclusive_and_explicit_resolution_is_exclusive() -> None:
    assert rescue.parse_seed_range(11, 14) == (11, 12, 13, 14)
    assert rescue.parse_seed_range_spec("3:5") == (3, 4, 5)
    assert rescue.resolve_seeds(seed_start=11, seed_end=18) == tuple(range(11, 19))
    assert rescue.resolve_seeds(seeds="18,11,14") == (18, 11, 14)
    with pytest.raises(rescue.RescueValidationError, match="exactly one"):
        rescue.resolve_seeds(seeds="1", seed_range="1-2")
    with pytest.raises(rescue.RescueValidationError, match="together"):
        rescue.resolve_seeds(seed_start=1)


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"post_prefix_generated_token_budget": 17}, 17),
        ({"max_new_tokens": 23}, 23),
        ({"total_generated_token_budget": 512}, 512),
    ],
)
def test_row_token_budget_supports_all_frozen_source_schemas(
    config: dict[str, int], expected: int
) -> None:
    assert rescue.resolve_row_token_budget(config) == expected


@pytest.mark.parametrize("value", [None, 0, -1, True, "512"])
def test_row_token_budget_rejects_nonpositive_or_untyped_values(value: object) -> None:
    with pytest.raises(rescue.RescueValidationError, match="positive"):
        rescue.resolve_row_token_budget({"total_generated_token_budget": value})


def test_frozen_prefix_record_is_copied_and_hash_checked() -> None:
    ids = [1, 2, 3]
    row = _row(owner="covered")
    original = {
        "image_id": 987,
        "prefix_token_ids": ids,
        "prefix_token_ids_sha256": rescue.hash_prefix_token_ids(ids),
        "prefix_owner_ids": ["covered"],
        "frozen_greedy_row": row,
        "source_checkpoint": {"id": "checkpoint-a"},
        "source_config": {"fingerprint": "config-a"},
    }
    frozen = rescue.validate_prefix_record(original)
    ids.append(99)
    row["seed"] = 101
    assert frozen["prefix_token_ids"] == [1, 2, 3]
    assert frozen["frozen_greedy_row"]["seed"] == 7
    original["prefix_token_ids_sha256"] = "0" * 64
    with pytest.raises(rescue.RescueValidationError, match="token hash"):
        rescue.validate_prefix_record(original)


def test_frozen_prefix_record_allows_prefix_only_records() -> None:
    ids = [1, 2]
    record = {
        "prefix_token_ids": ids,
        "prefix_token_ids_sha256": rescue.hash_prefix_token_ids(ids),
        "prefix_owner_ids": ["covered"],
    }
    frozen = rescue.validate_prefix_record(record)
    assert frozen["prefix_token_ids"] == ids
    assert frozen["frozen_greedy_row"] is None


def _prefix_artifact(*, duplicate: bool = False) -> tuple[dict[str, object], str]:
    prefix = [1, 2, 3]
    frozen = _row(owner="covered")
    frozen["prefix_token_ids"] = prefix
    frozen["prefix_token_ids_sha256"] = rescue.hash_prefix_token_ids(prefix)
    evaluation = {
        "prefix": {
            "prefix_token_ids": prefix,
            "prefix_token_ids_sha256": rescue.hash_prefix_token_ids(prefix),
            "covered_entity_ids": ["from-prefix-record"],
            "trajectory_provenance": {"mode": "greedy", "seed": None},
        },
        "native_greedy_row": frozen,
        "classification": {"covered_owner_ids": ["wrong-source"]},
    }
    evaluations = [evaluation, dict(evaluation)] if duplicate else [evaluation]
    return (
        {"images": [{"image_id": "7816", "prompt": {}, "runtime": {}, "prefix_evaluations": evaluations}]},
        rescue.hash_prefix_token_ids(prefix),
    )


def test_local_prefix_artifact_selects_unique_prefix_and_uses_prefix_owners() -> None:
    artifact, prefix_hash = _prefix_artifact()
    selected = rescue.select_local_prefix_evaluation(artifact, prefix_hash=prefix_hash)
    assert selected["prefix_token_ids_sha256"] == prefix_hash
    assert selected["covered_owner_ids"] == ["from-prefix-record"]
    assert selected["trajectory_provenance"] == {"mode": "greedy", "seed": None}


def test_local_prefix_artifact_rejects_missing_or_duplicate_prefix() -> None:
    artifact, prefix_hash = _prefix_artifact()
    with pytest.raises(rescue.RescueValidationError, match="not found"):
        rescue.select_local_prefix_evaluation(artifact, prefix_hash="missing")
    duplicate, _ = _prefix_artifact(duplicate=True)
    with pytest.raises(rescue.RescueValidationError, match="not unique"):
        rescue.select_local_prefix_evaluation(duplicate, prefix_hash=prefix_hash)


def test_root_trace_selects_exact_duplicate_row_and_prefix() -> None:
    selected = rescue.select_root_trace_row(
        _root_trace(), row_index=3, harmful_kind="duplicate"
    )
    assert selected["image_id"] == "7816"
    assert selected["row_index"] == 3
    assert selected["prefix_token_ids"] == [1, 2]
    assert selected["covered_owner_ids"] == ["covered"]
    assert selected["frozen_greedy_row"]["raw_generated_token_ids"] == [30, 31]


def test_root_trace_rejects_blind_image() -> None:
    with pytest.raises(rescue.RescueValidationError, match="blind image"):
        rescue.select_root_trace_row(
            _root_trace(image_id="1584"), row_index=3, harmful_kind="duplicate"
        )


def test_root_trace_terminal_requires_verified_uncovered_owner() -> None:
    selected = rescue.select_root_trace_row(
        _root_trace(terminal=True), row_index=3, harmful_kind="premature_terminal"
    )
    receipt = rescue.validate_greedy_terminal(
        selected["frozen_greedy_row"],
        frozen=selected["frozen_greedy_row"],
        covered_owner_ids=selected["covered_owner_ids"],
        entity_ledger=selected["entity_ledger"],
    )
    assert receipt["verified_uncovered_owner_ids"] == ["uncovered"]
    bad = dict(selected["frozen_greedy_row"])
    bad["parsed_predictions"] = [{"description": "person"}]
    with pytest.raises(rescue.RescueValidationError, match="premature terminal"):
        rescue.validate_greedy_terminal(
            bad,
            frozen=selected["frozen_greedy_row"],
            covered_owner_ids=selected["covered_owner_ids"],
            entity_ledger=selected["entity_ledger"],
        )


def test_cli_requires_exactly_one_source_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "runner",
            "--stage6-artifact",
            "stage6.json",
            "--prefix-artifact",
            "prefix.json",
            "--seeds",
            "1",
            "--output",
            "out.json",
        ],
    )
    with pytest.raises(SystemExit):
        rescue._parse_args()


def test_cli_rejects_root_trace_with_stage6_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "runner",
            "--stage6-artifact",
            "stage6.json",
            "--root-trace",
            "root.json",
            "--row-index",
            "3",
            "--harmful-kind",
            "duplicate",
            "--seeds",
            "1",
            "--output",
            "out.json",
        ],
    )
    with pytest.raises(SystemExit):
        rescue._parse_args()


def test_stage6_arm_requires_exact_prefix_and_first_row_hash() -> None:
    prefix = [1, 2, 3]
    frozen_row = _row(owner="covered")
    stage6 = {
        "schema_version": rescue.STAGE6_SCHEMA_VERSION,
        "phase": rescue.STAGE6_PHASE,
        "config": {"model_dtype": "fp32"},
        "arms": {
            "arm": {
                "prefix_token_ids": prefix,
                "prefix_token_ids_sha256": rescue.hash_prefix_token_ids(prefix),
                "prefix_owner_ids": ["covered"],
                "continuation": {"rows": [frozen_row]},
            }
        },
    }
    result = rescue.validate_stage6_arm(stage6, arm_name="arm")
    assert result["prefix_token_ids"] == prefix
    assert result["covered_owner_ids"] == ["covered"]
    stage6["arms"]["arm"]["prefix_token_ids_sha256"] = "0" * 64
    with pytest.raises(rescue.RescueValidationError, match="prefix token hash"):
        rescue.validate_stage6_arm(stage6, arm_name="arm")


def test_static_identity_excludes_known_full_source_jsonl_drift(
    tmp_path: Path,
) -> None:
    files = {}
    for label in rescue.STATIC_IDENTITY_LABELS:
        path = tmp_path / label
        path.write_text(label, encoding="utf-8")
        files[label] = {"path": str(path), "sha256": _sha(path)}
    files["source_jsonl"] = {
        "path": str(tmp_path / "old.jsonl"),
        "sha256": "0" * 64,
    }
    result = rescue.validate_static_identity(
        {"frozen_file_identity": files},
        infer_config_path=Path(files["infer_config"]["path"]),
    )
    assert result["source_jsonl_policy"]["status"] == "selected_identity_only"
    assert result["source_jsonl_policy"]["frozen_sha256_not_enforced"] == "0" * 64


def test_static_identity_still_rejects_checkpoint_payload_drift(
    tmp_path: Path,
) -> None:
    files = {}
    for label in rescue.STATIC_IDENTITY_LABELS:
        path = tmp_path / label
        path.write_text(label, encoding="utf-8")
        files[label] = {"path": str(path), "sha256": _sha(path)}
    Path(files["adapter_model"]["path"]).write_text("drift", encoding="utf-8")
    with pytest.raises(rescue.RescueValidationError, match="adapter_model SHA-256"):
        rescue.validate_static_identity(
            {"frozen_file_identity": files},
            infer_config_path=Path(files["infer_config"]["path"]),
        )


def test_static_identity_accepts_stage_one_nested_identity(
    tmp_path: Path,
) -> None:
    files = {}
    for label in rescue.STATIC_IDENTITY_LABELS:
        path = tmp_path / label
        path.write_text(label, encoding="utf-8")
        files[label] = {"path": str(path), "sha256": _sha(path)}
    result = rescue.validate_static_identity(
        {"frozen_inputs": {"checkpoint_config_source_identity": files}},
        infer_config_path=Path(files["infer_config"]["path"]),
    )
    assert result["checkpoint_json"]["sha256"] == files["checkpoint_json"]["sha256"]


def test_greedy_parity_requires_one_strict_covered_duplicate() -> None:
    observed = _row(owner="covered")
    frozen = dict(observed)
    receipt = rescue.validate_greedy_duplicate(
        observed,
        frozen=frozen,
        covered_owner_ids=["covered"],
    )
    assert receipt["duplicate_owner_id"] == "covered"
    with pytest.raises(rescue.RescueValidationError, match="strict covered duplicate"):
        rescue.validate_greedy_duplicate(
            observed,
            frozen=frozen,
            covered_owner_ids=["different"],
        )


def test_sample_record_selects_only_clean_verified_uncovered_owner() -> None:
    row = _row(owner="uncovered")
    policy = rescue.build_producer_policy(
        mode="sampled",
        seed=7,
        checkpoint_id="checkpoint",
        prompt_token_ids_sha256=rescue.hash_prefix_token_ids([100]),
        prefix_token_ids_sha256=rescue.hash_prefix_token_ids([1, 2]),
    )
    record = rescue.build_sample_record(
        row,
        seed=7,
        producer_policy=policy,
        prompt_token_ids=[100],
        prefix_token_ids=[1, 2],
        covered_owner_ids=["covered"],
        uncovered_owner_ids=["uncovered"],
        entity_ledger=[
            {"entity_id": "covered", "verification": "verified"},
            {"entity_id": "uncovered", "verification": "verified"},
        ],
    )
    assert record["candidate_token_ids"] == [10, 11]
    assert record["verified_uncovered_owner_ids"] == ["uncovered"]
    assert record["selected_verified_uncovered_rescue"] is True


def test_sample_record_preserves_incomplete_seed_without_retry() -> None:
    row = _row(owner="uncovered")
    row["row_stop"] = {"stop_reason": "terminal"}
    policy = rescue.build_producer_policy(
        mode="sampled",
        seed=7,
        checkpoint_id="checkpoint",
        prompt_token_ids_sha256=rescue.hash_prefix_token_ids([100]),
        prefix_token_ids_sha256=rescue.hash_prefix_token_ids([1, 2]),
    )
    record = rescue.build_sample_record(
        row,
        seed=7,
        producer_policy=policy,
        prompt_token_ids=[100],
        prefix_token_ids=[1, 2],
        covered_owner_ids=["covered"],
        uncovered_owner_ids=["uncovered"],
        entity_ledger=[{"entity_id": "uncovered", "verification": "verified"}],
    )
    assert record["seed"] == 7
    assert record["completion_status"] == "incomplete"
    assert record["candidate_token_ids"] == [10, 11]
    assert record["selected_verified_uncovered_rescue"] is False
