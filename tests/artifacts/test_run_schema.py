"""Wave-4 tests for ``src.artifacts.run_schema``: pure strict-JSON schema
normalization/serialization, bounded detail/timestamp/number/lineage
validation, measurement payload construction, and logging-row normalization
moved verbatim out of ``src/artifacts/run_writer.py`` (design decision 10).

None of these functions perform filesystem I/O; ``RunWriter``'s own
exact-byte fixture suites (``tests/artifacts/test_run_artifacts.py`` and
``tests/training/test_orchestration_compatibility.py``'s
``exercise_characterized_run_writer``) are the byte-identity proof that the
move did not change behavior. These tests instead exact-compare the pure
functions directly, including edge cases the writer-level fixtures do not
individually isolate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.artifacts import run_schema
from src.common.errors import ArtifactContractError


# ---------------------------------------------------------------------------
# Strict JSON serialization
# ---------------------------------------------------------------------------


def test_strict_json_dumps_sorts_keys_and_is_ascii() -> None:
    encoded = run_schema._strict_json_dumps({"b": 1, "a": "café"})
    assert encoded == '{"a": "caf\\u00e9", "b": 1}'


def test_strict_json_dumps_compact_uses_tight_separators() -> None:
    encoded = run_schema._strict_json_dumps({"a": 1, "b": 2}, compact=True)
    assert encoded == '{"a":1,"b":2}'


def test_strict_json_dumps_rejects_nan_and_non_serializable() -> None:
    with pytest.raises(ArtifactContractError) as exc_info:
        run_schema._strict_json_dumps({"x": float("nan")})
    assert exc_info.value.code == "run_writer.not_json_serializable"

    with pytest.raises(ArtifactContractError):
        run_schema._strict_json_dumps({"x": object()})


def test_strict_mapping_deep_copies_and_validates_json_shape() -> None:
    source = {"nested": {"value": 1}}
    result = run_schema._strict_mapping(source, field="x")
    result["nested"]["value"] = 2
    assert source["nested"]["value"] == 1

    with pytest.raises(ArtifactContractError):
        run_schema._strict_mapping("not-a-mapping", field="x")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Non-finite normalization / rejection
# ---------------------------------------------------------------------------


def test_replace_non_finite_records_paths_and_nulls_values() -> None:
    fields: list[str] = []
    result = run_schema._replace_non_finite(
        {"a": float("inf"), "b": [1.0, float("nan")], "c": "ok"},
        path="",
        fields=fields,
    )
    assert result == {"a": None, "b": [1.0, None], "c": "ok"}
    assert sorted(fields) == ["a", "b[1]"]


def test_reject_non_finite_raises_when_a_float_remains_non_finite() -> None:
    with pytest.raises(ArtifactContractError) as exc_info:
        run_schema._reject_non_finite({"a": [float("inf")]})
    assert exc_info.value.code == "run_writer.non_finite_remains"

    run_schema._reject_non_finite({"a": [1.0, "ok", None]})  # does not raise


# ---------------------------------------------------------------------------
# Logging-row normalization
# ---------------------------------------------------------------------------


def test_normalize_logging_row_requires_split_and_positive_step() -> None:
    with pytest.raises(ArtifactContractError) as exc_info:
        run_schema._normalize_logging_row({"split": "bogus", "step": 1})
    assert exc_info.value.code == "run_writer.invalid_logging_split"

    with pytest.raises(ArtifactContractError) as exc_info:
        run_schema._normalize_logging_row({"split": "train", "step": 0})
    assert exc_info.value.code == "run_writer.invalid_logging_step"


def test_normalize_logging_row_declares_and_replaces_non_finite_fields() -> None:
    normalized = run_schema._normalize_logging_row(
        {"split": "train", "step": 1, "loss": float("nan")}
    )
    assert normalized["loss"] is None
    assert normalized["non_finite_fields"] == ["loss"]


def test_normalize_logging_row_rejects_malformed_declared_non_finite_fields() -> None:
    with pytest.raises(ArtifactContractError) as exc_info:
        run_schema._normalize_logging_row(
            {"split": "train", "step": 1, "non_finite_fields": "not-a-list"}
        )
    assert exc_info.value.code == "run_writer.invalid_non_finite_fields"


# ---------------------------------------------------------------------------
# Timestamp / number validation
# ---------------------------------------------------------------------------


def test_validate_timestamp_requires_nonempty_string() -> None:
    run_schema._validate_timestamp("2026-01-01T00:00:00Z", field="x")
    with pytest.raises(ArtifactContractError) as exc_info:
        run_schema._validate_timestamp("", field="x")
    assert exc_info.value.code == "run_writer.invalid_phase_timestamp"


def test_validate_nonnegative_finite_rejects_bool_negative_and_non_finite() -> None:
    assert run_schema._validate_nonnegative_finite(1, field="x") == 1.0
    for bad in (True, -1.0, float("inf"), float("nan"), "1"):
        with pytest.raises(ArtifactContractError) as exc_info:
            run_schema._validate_nonnegative_finite(bad, field="x")
        assert exc_info.value.code == "run_writer.invalid_phase_summary"


def test_validate_nonnegative_int_rejects_bool_negative_and_non_int() -> None:
    assert run_schema._validate_nonnegative_int(0, field="x") == 0
    for bad in (True, -1, 1.5, "1"):
        with pytest.raises(ArtifactContractError) as exc_info:
            run_schema._validate_nonnegative_int(bad, field="x")
        assert exc_info.value.code == "run_writer.invalid_measured_step_count"


def test_validate_phase_name_only_accepts_the_frozen_set() -> None:
    run_schema._validate_phase_name("first_optimizer_step")
    with pytest.raises(ArtifactContractError) as exc_info:
        run_schema._validate_phase_name("not_a_phase")
    assert exc_info.value.code == "run_writer.invalid_phase_name"


def test_unphased_failure_phase_is_a_member_of_the_frozen_phase_set() -> None:
    assert run_schema._UNPHASED_FAILURE_PHASE in run_schema._PHASE_NAMES


# ---------------------------------------------------------------------------
# Bounded detail validation
# ---------------------------------------------------------------------------


def test_attach_rank_phase_receipt_requires_resources_before_details() -> None:
    receipt: dict[str, object] = {}
    with pytest.raises(ArtifactContractError) as exc_info:
        run_schema._attach_rank_phase_receipt(
            receipt, rank_resources=None, rank_details={"0": {}}
        )
    assert exc_info.value.code == "run_writer.invalid_additive_receipt"


def test_attach_rank_phase_receipt_no_op_without_resources_or_details() -> None:
    receipt: dict[str, object] = {}
    run_schema._attach_rank_phase_receipt(
        receipt, rank_resources=None, rank_details=None
    )
    assert receipt == {}


def test_validate_bounded_rank_detail_value_rejects_excess_depth() -> None:
    nested: object = "leaf"
    for _ in range(8):
        nested = {"k": nested}
    with pytest.raises(ArtifactContractError) as exc_info:
        run_schema._validate_bounded_rank_detail_value(nested, depth=0)
    assert exc_info.value.code == "run_writer.invalid_additive_receipt"


def test_validate_bounded_rank_detail_value_rejects_non_ascii_and_long_strings() -> None:
    run_schema._validate_bounded_rank_detail_value("ok", depth=0)
    with pytest.raises(ArtifactContractError):
        run_schema._validate_bounded_rank_detail_value("café", depth=0)
    with pytest.raises(ArtifactContractError):
        run_schema._validate_bounded_rank_detail_value("x" * 257, depth=0)


# ---------------------------------------------------------------------------
# Lineage / continuation validation
# ---------------------------------------------------------------------------


def test_continuation_payload_defaults_when_lineage_is_absent() -> None:
    payload = run_schema._continuation_payload(
        run_id="run-1", segment_id="segment-1", lineage=None
    )
    assert payload == {
        "schema_version": 1,
        "segment_id": "segment-1",
        "continuation_index": 0,
        "parent": None,
    }


def _valid_lineage(**overrides: object) -> dict[str, object]:
    lineage: dict[str, object] = {
        "parent_run_id": "parent-run",
        "parent_segment_id": "parent-segment",
        "parent_continuation_index": 0,
        "continuation_index": 1,
        "parent_checkpoint_identity": {
            "resolved_path": str(Path("/tmp/checkpoints/step-1").resolve()),
            "checkpoint_step": 1,
            "training_state_manifest_file_sha256": "a" * 64,
            "training_state_aggregate_digest": "b" * 64,
        },
    }
    lineage.update(overrides)
    return lineage


def test_continuation_payload_admits_a_well_formed_lineage() -> None:
    payload = run_schema._continuation_payload(
        run_id="run-2", segment_id="segment-2", lineage=_valid_lineage()
    )
    assert payload["continuation_index"] == 1
    assert payload["parent"]["run_id"] == "parent-run"
    assert payload["parent"]["continuation_index"] == 0


def test_continuation_payload_rejects_self_referential_lineage() -> None:
    with pytest.raises(ArtifactContractError) as exc_info:
        run_schema._continuation_payload(
            run_id="parent-run",
            segment_id="segment-2",
            lineage=_valid_lineage(),
        )
    assert exc_info.value.code == "run_writer.invalid_continuation_lineage"


def test_continuation_payload_rejects_non_advancing_index() -> None:
    with pytest.raises(ArtifactContractError):
        run_schema._continuation_payload(
            run_id="run-2",
            segment_id="segment-2",
            lineage=_valid_lineage(continuation_index=5),
        )


def test_continuation_payload_rejects_incomplete_field_set() -> None:
    with pytest.raises(ArtifactContractError) as exc_info:
        run_schema._continuation_payload(
            run_id="run-2", segment_id="segment-2", lineage={"parent_run_id": "x"}
        )
    assert exc_info.value.code == "run_writer.invalid_continuation_lineage"


def test_is_lowercase_sha256_accepts_only_64_char_lowercase_hex() -> None:
    assert run_schema._is_lowercase_sha256("a" * 64) is True
    assert run_schema._is_lowercase_sha256("A" * 64) is False
    assert run_schema._is_lowercase_sha256("a" * 63) is False
    assert run_schema._is_lowercase_sha256(12345) is False


# ---------------------------------------------------------------------------
# Provenance / measurement payload construction
# ---------------------------------------------------------------------------


def test_provenance_payload_defaults_to_unavailable_when_absent() -> None:
    payload = run_schema._provenance_payload(None)
    assert payload["repository"]["state"] == "unavailable"
    assert payload["schema_version"] == 1


def test_provenance_payload_deep_copies_the_supplied_mapping() -> None:
    source = {"schema_version": 1, "repository": {"state": "clean"}}
    payload = run_schema._provenance_payload(source)
    payload["repository"]["state"] = "dirty"
    assert source["repository"]["state"] == "clean"


def test_measurement_payload_defaults_context_and_seeds_empty_state() -> None:
    payload = run_schema._measurement_payload(
        None, entry_started_at="2026-01-01T00:00:00Z"
    )
    assert payload["context"] == {
        "comparison_arm": "unclassified",
        "wall_clock_scope": "training_entry_to_terminal_artifact",
        "warmup_exclusion_steps": None,
    }
    assert payload["phases"] == {}
    assert payload["checkpoint_publication_events"] == []
    assert payload["entry_to_terminal"]["status"] == "running"


def test_measurement_payload_rejects_missing_entry_started_at() -> None:
    with pytest.raises(ArtifactContractError) as exc_info:
        run_schema._measurement_payload(None, entry_started_at="")
    assert exc_info.value.code == "run_writer.invalid_phase_timestamp"


# ---------------------------------------------------------------------------
# Checkpoint publication identity / progress validation
# ---------------------------------------------------------------------------


def test_checkpoint_publication_identity_admits_a_matching_identity(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "step-3"
    checkpoint_dir.mkdir()
    value = {
        "checkpoint_step": 3,
        "resolved_path": str(checkpoint_dir.resolve()),
        "training_state_manifest_file_sha256": "a" * 64,
        "training_state_aggregate_digest": "b" * 64,
    }
    identity = run_schema._checkpoint_publication_identity(
        value, step=3, checkpoint_dir=checkpoint_dir
    )
    assert identity == value


def test_checkpoint_publication_identity_rejects_step_mismatch(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "step-3"
    checkpoint_dir.mkdir()
    value = {
        "checkpoint_step": 4,
        "resolved_path": str(checkpoint_dir.resolve()),
        "training_state_manifest_file_sha256": "a" * 64,
        "training_state_aggregate_digest": "b" * 64,
    }
    with pytest.raises(ArtifactContractError) as exc_info:
        run_schema._checkpoint_publication_identity(
            value, step=3, checkpoint_dir=checkpoint_dir
        )
    assert exc_info.value.code == "run_writer.invalid_checkpoint_publication_event"


def test_checkpoint_committed_progress_admits_a_well_formed_payload() -> None:
    value = {
        "schema": "coordexp-swift-checkpoint-committed-progress",
        "schema_version": 1,
        "completed_steps": 5,
        "consumed_packs": 10,
        "optimizer_update_status": "applied",
        "finite_status": "finite",
    }
    progress = run_schema._checkpoint_committed_progress(value, step=5)
    assert progress == value


def test_checkpoint_committed_progress_rejects_step_disagreement() -> None:
    value = {
        "schema": "coordexp-swift-checkpoint-committed-progress",
        "schema_version": 1,
        "completed_steps": 4,
        "consumed_packs": 10,
        "optimizer_update_status": "applied",
        "finite_status": "finite",
    }
    with pytest.raises(ArtifactContractError) as exc_info:
        run_schema._checkpoint_committed_progress(value, step=5)
    assert exc_info.value.code == "run_writer.invalid_checkpoint_publication_event"
