"""Targeted CPU-only tests for score_sorted_all_person_route_landscape.py.

The real planner output (built once against the frozen production sources,
via ``build_sorted_all_person_route_landscape.py`` itself) is exercised with
a fully faked scoring backend -- no GPU, no HF backend session, no real
``transformers`` model -- end to end through plan loading, request
selection, scalar and batched scoring, shard-receipt construction, and
create-or-identical output.
"""

from __future__ import annotations

import json
import shutil
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import torch

from scripts.research import build_sorted_all_person_route_landscape as planner
from scripts.research import score_sorted_all_person_route_landscape as sut
from scripts.research import score_sorted_owner_basin_landscape as scorer

VOCAB_SIZE = 200_000


@pytest.fixture(scope="module")
def plan_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("plan") / "plan-fixture"
    planner.build_sorted_all_person_route_landscape(output)
    return output


def _deterministic_logits(token_ids: Sequence[int], vocab_size: int = VOCAB_SIZE) -> torch.Tensor:
    seed = zlib.crc32(json.dumps([int(v) for v in token_ids]).encode("utf-8")) & 0xFFFFFFFF
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(vocab_size, generator=generator)


def _fake_scalar_backend(context_id: str, full_prefix_token_ids: Sequence[int]) -> scorer.FullReforwardBackend:
    return scorer.FullReforwardBackend(
        root_prefix_token_ids=full_prefix_token_ids,
        full_reforward=lambda tokens: _deterministic_logits(tokens),
        context_id=context_id,
        group_id=f"fixture-group:{context_id}",
    )


def _fake_batchable_backend(
    context_id: str, full_prefix_token_ids: Sequence[int]
) -> tuple[scorer.FullReforwardBackend, dict[str, Any]]:
    admission = {
        "schema_version": "batched_full_reforward_parity.v1",
        "status": "not_requested",
        "requested_batch_size": 1,
        "effective_batch_size": 1,
    }
    return _fake_scalar_backend(context_id, full_prefix_token_ids), admission


def _fake_attestation() -> scorer.AttestationContext:
    return scorer.build_attestation_context(
        expected_vocab_size=VOCAB_SIZE,
        tokenizer_identity={"tokenizer": "fake-v1"},
        model_identity={"model": "fake-v1"},
        rule_digest="fake-rule-digest",
        runtime_receipt_id="fake-runtime-receipt",
    )


# ---------------------------------------------------------------------------
# Plan loading: tamper-evidence, schema/unit-id checks
# ---------------------------------------------------------------------------


def test_load_plan_accepts_the_real_sealed_planner_output(plan_dir: Path) -> None:
    plan = sut.load_plan(plan_dir)
    assert len(plan.contexts_by_id) == 6
    assert len(plan.candidates_by_id) == 369
    assert len(plan.sidecars_by_id) == 5
    assert len(plan.requests_by_id) == 2252


def test_load_plan_detects_tampered_candidate_bank(plan_dir: Path, tmp_path: Path) -> None:
    tampered = tmp_path / "tampered"
    shutil.copytree(plan_dir, tampered)
    path = tampered / "primary-candidates.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(sut.ScoreRunError, match="does not match the digest sealed"):
        sut.load_plan(tampered)


def test_load_plan_detects_tampered_receipt_content(plan_dir: Path, tmp_path: Path) -> None:
    tampered = tmp_path / "tampered-receipt"
    shutil.copytree(plan_dir, tampered)
    receipt_path = tampered / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["counts"]["scoring_requests"] = 999999
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(sut.ScoreRunError, match="receipt_content_sha256 does not reconstruct"):
        sut.load_plan(tampered)


def test_load_plan_rejects_foreign_schema_version(plan_dir: Path, tmp_path: Path) -> None:
    tampered = tmp_path / "foreign"
    shutil.copytree(plan_dir, tampered)
    receipt_path = tampered / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["schema_version"] = "foreign.v0"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(sut.ScoreRunError, match="schema_version"):
        sut.load_plan(tampered)


# ---------------------------------------------------------------------------
# Selection: context/kind filtering and shard partitioning
# ---------------------------------------------------------------------------


def test_select_requests_context_and_kind_filter(plan_dir: Path) -> None:
    plan = sut.load_plan(plan_dir)
    selected, selection = sut.select_requests(
        plan, include_context_ids=["self-due-gt17"], include_request_kinds=["numerical_repeat"]
    )
    assert len(selected) == 8
    assert {row["request_id"] for row in selected} == {
        f"numerical-repeat:self-due-gt17:{i}" for i in range(8)
    }
    assert selection["shard_request_count"] == 8


def test_select_requests_full_context_primary_and_sidecar_count(plan_dir: Path) -> None:
    plan = sut.load_plan(plan_dir)
    selected, _ = sut.select_requests(
        plan, include_context_ids=["self-due-gt17"], include_request_kinds=["primary", "sidecar"]
    )
    assert len(selected) == 374  # 369 primary + 5 nondup sidecars, per unit.md


def test_select_requests_shard_partition_is_exact_and_disjoint(plan_dir: Path) -> None:
    plan = sut.load_plan(plan_dir)
    full, _ = sut.select_requests(plan, include_context_ids=["root"], include_request_kinds=["primary"])
    union_ids: set[str] = set()
    for shard_index in range(4):
        shard, _ = sut.select_requests(
            plan,
            include_context_ids=["root"],
            include_request_kinds=["primary"],
            shard_index=shard_index,
            shard_count=4,
        )
        ids = {row["request_id"] for row in shard}
        assert not (ids & union_ids), "shards must be disjoint"
        union_ids |= ids
    assert union_ids == {row["request_id"] for row in full}


def test_select_requests_rejects_unknown_context(plan_dir: Path) -> None:
    plan = sut.load_plan(plan_dir)
    with pytest.raises(sut.ScoreRunError, match="absent from the plan"):
        sut.select_requests(plan, include_context_ids=["not-a-real-context"])


def test_select_requests_rejects_bad_shard_index(plan_dir: Path) -> None:
    plan = sut.load_plan(plan_dir)
    with pytest.raises(sut.ScoreRunError, match="shard-index"):
        sut.select_requests(plan, shard_index=2, shard_count=2)


# ---------------------------------------------------------------------------
# --include-request-id: strict selector for posthoc scalar rescoring
# ---------------------------------------------------------------------------


def test_select_requests_include_request_id_exact_subset(plan_dir: Path) -> None:
    plan = sut.load_plan(plan_dir)
    target_ids = [
        "numerical-repeat:self-due-gt17:0",
        "numerical-repeat:self-due-gt17:5",
    ]
    selected, selection = sut.select_requests(plan, include_request_ids=target_ids)
    assert {row["request_id"] for row in selected} == set(target_ids)
    assert selection["included_request_ids"] == sorted(target_ids)


def test_select_requests_include_request_id_validated_against_full_plan(plan_dir: Path) -> None:
    plan = sut.load_plan(plan_dir)
    with pytest.raises(sut.ScoreRunError, match="absent from the plan's full scoring-request domain"):
        sut.select_requests(plan, include_request_ids=["not-a-real-request-id"])


def test_select_requests_include_request_id_rejects_duplicates(plan_dir: Path) -> None:
    plan = sut.load_plan(plan_dir)
    dup_id = "numerical-repeat:self-due-gt17:0"
    with pytest.raises(sut.ScoreRunError, match="contains duplicates"):
        sut.select_requests(plan, include_request_ids=[dup_id, dup_id])


def test_select_requests_include_request_id_rejects_ids_excluded_by_other_filters(plan_dir: Path) -> None:
    plan = sut.load_plan(plan_dir)
    # This id is real (exists in the plan) but is a numerical_repeat row,
    # which the request-kind filter below excludes -- must fail loudly, not
    # silently drop.
    with pytest.raises(sut.ScoreRunError, match="excluded by"):
        sut.select_requests(
            plan,
            include_request_ids=["numerical-repeat:self-due-gt17:0"],
            include_request_kinds=["primary"],
        )


def test_run_binds_included_request_ids_in_selection_and_receipt(plan_dir: Path) -> None:
    target_id = "numerical-repeat:self-due-gt17:3"
    args = sut.build_parser().parse_args(
        [
            "--plan-dir",
            str(plan_dir),
            "--include-request-id",
            target_id,
            "--validate-contract-only",
        ]
    )
    result = sut.run(args)
    assert result["selection"]["included_request_ids"] == [target_id]
    assert result["selection"]["shard_request_ids"] == [target_id]


# ---------------------------------------------------------------------------
# --include-request-id-file: deterministic scalar-confirmation manifest
# ---------------------------------------------------------------------------


def test_include_request_id_file_plain_lines(plan_dir: Path, tmp_path: Path) -> None:
    target_ids = ["numerical-repeat:self-due-gt17:1", "numerical-repeat:self-due-gt17:6"]
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("\n".join(target_ids) + "\n", encoding="utf-8")

    args = sut.build_parser().parse_args(
        [
            "--plan-dir",
            str(plan_dir),
            "--include-request-id-file",
            str(manifest),
            "--validate-contract-only",
        ]
    )
    result = sut.run(args)
    assert result["selection"]["shard_request_ids"] == sorted(target_ids)
    file_binding = result["selection"]["included_request_id_file"]
    assert file_binding["path"] == str(manifest.resolve())
    assert file_binding["sha256"] == sut.sha256_file(manifest)


def test_include_request_id_file_jsonl_objects(plan_dir: Path, tmp_path: Path) -> None:
    target_id = "numerical-repeat:self-due-gt17:2"
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"request_id": target_id, "note": "top-boundary-row"}) + "\n", encoding="utf-8")

    args = sut.build_parser().parse_args(
        [
            "--plan-dir",
            str(plan_dir),
            "--include-request-id-file",
            str(manifest),
            "--validate-contract-only",
        ]
    )
    result = sut.run(args)
    assert result["selection"]["shard_request_ids"] == [target_id]


def test_include_request_id_file_combines_with_flag(plan_dir: Path, tmp_path: Path) -> None:
    flag_id = "numerical-repeat:self-due-gt17:0"
    file_id = "numerical-repeat:self-due-gt17:4"
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(file_id + "\n", encoding="utf-8")

    args = sut.build_parser().parse_args(
        [
            "--plan-dir",
            str(plan_dir),
            "--include-request-id",
            flag_id,
            "--include-request-id-file",
            str(manifest),
            "--validate-contract-only",
        ]
    )
    result = sut.run(args)
    assert result["selection"]["shard_request_ids"] == sorted([flag_id, file_id])


def test_include_request_id_file_rejects_duplicates_within_file(plan_dir: Path, tmp_path: Path) -> None:
    target_id = "numerical-repeat:self-due-gt17:0"
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(f"{target_id}\n{target_id}\n", encoding="utf-8")
    with pytest.raises(sut.ScoreRunError, match="duplicate request ids"):
        sut._load_request_id_file(manifest)


def test_include_request_id_file_rejects_blank_line(plan_dir: Path, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("numerical-repeat:self-due-gt17:0\n\n", encoding="utf-8")
    with pytest.raises(sut.ScoreRunError, match="blank"):
        sut._load_request_id_file(manifest)


def test_include_request_id_file_rejects_unknown_id(plan_dir: Path, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("not-a-real-request-id\n", encoding="utf-8")
    args = sut.build_parser().parse_args(
        [
            "--plan-dir",
            str(plan_dir),
            "--include-request-id-file",
            str(manifest),
            "--validate-contract-only",
        ]
    )
    with pytest.raises(sut.ScoreRunError, match="absent from the plan's full scoring-request domain"):
        sut.run(args)


def test_include_request_id_file_rejects_overlap_with_flag(plan_dir: Path, tmp_path: Path) -> None:
    target_id = "numerical-repeat:self-due-gt17:0"
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(target_id + "\n", encoding="utf-8")
    args = sut.build_parser().parse_args(
        [
            "--plan-dir",
            str(plan_dir),
            "--include-request-id",
            target_id,
            "--include-request-id-file",
            str(manifest),
            "--validate-contract-only",
        ]
    )
    with pytest.raises(sut.ScoreRunError, match="overlapping request ids"):
        sut.run(args)


# ---------------------------------------------------------------------------
# Shard-receipt code identity: import-time fingerprint, immune to a
# concurrent edit to this file after the process has already imported it.
# ---------------------------------------------------------------------------


def test_executed_source_sha256_matches_the_actual_on_disk_module() -> None:
    import hashlib

    on_disk = hashlib.sha256(Path(sut.__file__).read_bytes()).hexdigest()
    assert sut.EXECUTED_SOURCE_SHA256 == on_disk


def test_build_shard_receipt_binds_import_time_fingerprint_not_a_live_reread(
    plan_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Simulate a concurrent post-import edit via an injected snapshot helper.

    Rather than actually mutating this module's source file on disk (messy
    and order-dependent for other tests), monkeypatch the *live* re-read
    ``sha256_file`` uses at receipt-build time. ``EXECUTED_SOURCE_SHA256``
    was already captured into a plain module-level constant at import time,
    so it is completely unaffected by this monkeypatch -- exactly the
    property that fixes the 2026-08-03 code-identity race.
    """

    plan = sut.load_plan(plan_dir)
    selected, selection = sut.select_requests(
        plan, include_context_ids=["self-due-gt17"], include_request_kinds=["numerical_repeat"]
    )
    rows, scoring_meta = sut.score_selected_requests(
        plan,
        selected,
        open_scalar_backend=_fake_scalar_backend,
        open_batchable_backend=_fake_batchable_backend,
        attestation=_fake_attestation(),
    )

    drifted_hash = "f" * 64
    original_sha256_file = sut.sha256_file

    def _fake_sha256_file(path: Path) -> str:
        if Path(path).resolve() == Path(sut.__file__).resolve():
            return drifted_hash
        return original_sha256_file(path)

    monkeypatch.setattr(sut, "sha256_file", _fake_sha256_file)

    receipt = sut.build_shard_receipt(
        plan=plan,
        selection=selection,
        rows=rows,
        backend_admission={"selected_backend": "fake", **scoring_meta},
        source_identity={"model_identity_sha256": "fake", "tokenizer_identity_sha256": "fake"},
        environment={"note": "fake-cpu-smoke"},
    )
    code = receipt["code"]
    assert code["executed_source_sha256"] == sut.EXECUTED_SOURCE_SHA256
    assert code["executed_source_sha256"] != drifted_hash
    assert code["receipt_time_file_sha256"] == drifted_hash
    assert code["source_drift_detected"] is True
    # The back-compat "sha256" alias must bind the correct (import-time)
    # value, never the drifted end-of-job read that caused the race.
    assert code["sha256"] == sut.EXECUTED_SOURCE_SHA256


def test_build_shard_receipt_no_drift_when_file_unchanged(plan_dir: Path) -> None:
    plan = sut.load_plan(plan_dir)
    selected, selection = sut.select_requests(
        plan, include_context_ids=["self-due-gt17"], include_request_kinds=["numerical_repeat"]
    )
    rows, scoring_meta = sut.score_selected_requests(
        plan,
        selected,
        open_scalar_backend=_fake_scalar_backend,
        open_batchable_backend=_fake_batchable_backend,
        attestation=_fake_attestation(),
    )
    receipt = sut.build_shard_receipt(
        plan=plan,
        selection=selection,
        rows=rows,
        backend_admission={"selected_backend": "fake", **scoring_meta},
        source_identity={"model_identity_sha256": "fake", "tokenizer_identity_sha256": "fake"},
        environment={"note": "fake-cpu-smoke"},
    )
    assert receipt["code"]["source_drift_detected"] is False
    assert receipt["code"]["receipt_time_file_sha256"] == receipt["code"]["executed_source_sha256"]


# ---------------------------------------------------------------------------
# CPU-only fake-scorer smoke: full scalar scoring -> receipt -> create-or-identical
# ---------------------------------------------------------------------------


def test_score_selected_requests_fake_backend_end_to_end(plan_dir: Path, tmp_path: Path) -> None:
    plan = sut.load_plan(plan_dir)
    selected, selection = sut.select_requests(
        plan, include_context_ids=["self-due-gt17"], include_request_kinds=["numerical_repeat"]
    )

    rows, scoring_meta = sut.score_selected_requests(
        plan,
        selected,
        open_scalar_backend=_fake_scalar_backend,
        open_batchable_backend=_fake_batchable_backend,
        attestation=_fake_attestation(),
    )
    assert len(rows) == 8
    assert scoring_meta["contexts_scored"] == ["self-due-gt17"]
    for row in rows:
        assert row["request_kind"] == "numerical_repeat"
        assert row["excluded_from_primary_ranks"] is True
        assert row["primary_role"] is False
        assert isinstance(row["raw_model_logprob"]["complete_box_logprob_sum"], float)
        assert row["decision_bearing_channel"] == sut.DECISION_BEARING_CHANNEL

    receipt = sut.build_shard_receipt(
        plan=plan,
        selection=selection,
        rows=rows,
        backend_admission={"selected_backend": scorer.FULL_REFORWARD_SCORING_BACKEND, **scoring_meta},
        source_identity={"model_identity_sha256": "fake", "tokenizer_identity_sha256": "fake"},
        environment={"note": "fake-cpu-smoke"},
    )
    assert receipt["counts"]["rows"] == 8
    assert receipt["counts"]["rows_by_request_kind"] == {"numerical_repeat": 8}
    assert receipt["plan"]["receipt_content_sha256"] == plan.receipt["receipt_content_sha256"]

    output_dir = tmp_path / "shard"
    outcome = sut.write_shard(output_dir, rows=rows, receipt=receipt)
    assert outcome["scores_status"] == "created"
    assert outcome["receipt_status"] == "created"

    outcome_again = sut.write_shard(output_dir, rows=rows, receipt=receipt)
    assert outcome_again["scores_status"] == "identical_existing_output"
    assert outcome_again["receipt_status"] == "identical_existing_output"

    mutated_receipt = {**receipt, "counts": {**receipt["counts"], "rows": 999}}
    with pytest.raises(sut.ScoreRunError, match="already exists with different"):
        sut.write_shard(output_dir, rows=rows, receipt=mutated_receipt)

    forced = sut.write_shard(output_dir, rows=rows, receipt=mutated_receipt, force=True)
    assert forced["receipt_status"] == "overwritten_forced"


def test_sidecar_and_primary_rows_are_never_conflated_in_primary_role(plan_dir: Path) -> None:
    plan = sut.load_plan(plan_dir)
    selected, _ = sut.select_requests(
        plan, include_context_ids=["self-due-gt17"], include_request_kinds=["primary", "sidecar"]
    )
    rows, _ = sut.score_selected_requests(
        plan,
        selected,
        open_scalar_backend=_fake_scalar_backend,
        open_batchable_backend=_fake_batchable_backend,
        attestation=_fake_attestation(),
    )
    primary_rows = [row for row in rows if row["request_kind"] == "primary"]
    sidecar_rows = [row for row in rows if row["request_kind"] == "sidecar"]
    assert len(primary_rows) == 369
    assert len(sidecar_rows) == 5
    assert all(row["primary_role"] for row in primary_rows)
    assert not any(row["primary_role"] for row in sidecar_rows)
    assert all(row["excluded_from_primary_ranks"] for row in sidecar_rows)


def test_numerical_repeats_are_never_routed_through_the_batchable_backend(plan_dir: Path) -> None:
    """Even if repeats are selected alongside primary rows, they must use the scalar path."""

    plan = sut.load_plan(plan_dir)
    selected, _ = sut.select_requests(
        plan,
        include_context_ids=["self-due-gt17"],
        include_request_kinds=["primary", "numerical_repeat"],
    )
    batchable_calls: list[str] = []
    scalar_calls: list[str] = []

    def _tracking_batchable(context_id: str, full_prefix_token_ids: Sequence[int]):
        batchable_calls.append(context_id)
        return _fake_batchable_backend(context_id, full_prefix_token_ids)

    def _tracking_scalar(context_id: str, full_prefix_token_ids: Sequence[int]):
        scalar_calls.append(context_id)
        return _fake_scalar_backend(context_id, full_prefix_token_ids)

    rows, _ = sut.score_selected_requests(
        plan,
        selected,
        open_scalar_backend=_tracking_scalar,
        open_batchable_backend=_tracking_batchable,
        attestation=_fake_attestation(),
    )
    assert batchable_calls == ["self-due-gt17"]
    assert scalar_calls == ["self-due-gt17"]
    assert sum(1 for row in rows if row["request_kind"] == "numerical_repeat") == 8
    assert sum(1 for row in rows if row["request_kind"] == "primary") == 369


def test_batchable_backend_used_when_admitted(plan_dir: Path) -> None:
    """A passed batch admission is honored: the batchable backend scores every batchable row."""

    plan = sut.load_plan(plan_dir)
    selected, _ = sut.select_requests(
        plan, include_context_ids=["self-due-gt2"], include_request_kinds=["primary"]
    )

    def _admitted_batched_backend(context_id: str, full_prefix_token_ids: Sequence[int]):
        backend = scorer.FullReforwardBackend(
            root_prefix_token_ids=full_prefix_token_ids,
            full_reforward=lambda tokens: _deterministic_logits(tokens),
            batched_full_reforward=lambda rows: torch.stack([_deterministic_logits(row) for row in rows]),
            full_reforward_batch_size=8,
            context_id=context_id,
            group_id=f"fixture-batched-group:{context_id}",
        )
        admission = {
            "schema_version": "batched_full_reforward_parity.v1",
            "status": "passed",
            "requested_batch_size": 8,
            "effective_batch_size": 8,
        }
        return backend, admission

    rows, scoring_meta = sut.score_selected_requests(
        plan,
        selected,
        open_scalar_backend=_fake_scalar_backend,
        open_batchable_backend=_admitted_batched_backend,
        attestation=_fake_attestation(),
    )
    assert len(rows) == 369
    accounting = scoring_meta["per_context_accounting"][0]
    assert accounting["batch_admission"]["effective_batch_size"] == 8
    assert accounting["batch_admission"]["status"] == "passed"


# ---------------------------------------------------------------------------
# Executed-media digest singleton normalization (regression: a length-1
# per-request sequence must never be compared directly to the frozen scalar
# digest).
# ---------------------------------------------------------------------------


def test_normalized_executed_media_sha256_unwraps_singleton_sequence() -> None:
    normalized = sut._normalized_executed_media_sha256(["abc123"], expected_count=1)
    assert normalized == ("abc123",)


def test_normalized_executed_media_sha256_rejects_bare_scalar() -> None:
    with pytest.raises(sut.ScoreRunError, match="per-request sequence"):
        sut._normalized_executed_media_sha256("abc123", expected_count=1)


def test_normalized_executed_media_sha256_rejects_wrong_length() -> None:
    with pytest.raises(sut.ScoreRunError, match="length does not match"):
        sut._normalized_executed_media_sha256(["a", "b"], expected_count=1)


def test_normalized_executed_media_sha256_rejects_non_uniform_batch() -> None:
    with pytest.raises(sut.ScoreRunError, match="not uniform"):
        sut._normalized_executed_media_sha256(["a", "b"], expected_count=2)


# ---------------------------------------------------------------------------
# CLI-level contract-only validation (no GPU/torch import) + explicit failure
# modes
# ---------------------------------------------------------------------------


def test_run_validate_contract_only_never_touches_torch_or_gpu(plan_dir: Path) -> None:
    args = sut.build_parser().parse_args(
        [
            "--plan-dir",
            str(plan_dir),
            "--include-context-id",
            "self-due-gt17",
            "--include-request-kind",
            "numerical_repeat",
            "--validate-contract-only",
        ]
    )
    result = sut.run(args)
    assert result["runtime_execution_status"] == "contract_validated_no_gpu"
    assert result["selection"]["shard_request_count"] == 8


def test_run_rejects_non_positive_batch_size(plan_dir: Path) -> None:
    args = sut.build_parser().parse_args(
        [
            "--plan-dir",
            str(plan_dir),
            "--full-reforward-batch-size",
            "0",
            "--validate-contract-only",
        ]
    )
    with pytest.raises(sut.ScoreRunError, match="positive integer"):
        sut.run(args)
