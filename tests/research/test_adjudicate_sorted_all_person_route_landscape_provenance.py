"""Targeted CPU-only tests for
adjudicate_sorted_all_person_route_landscape_provenance.py.

Shards are produced with the real scorer pipeline against the real planner
output, using a fully faked scoring backend -- no GPU, no HF backend
session, no real ``transformers`` model. Heterogeneous code-identity is
simulated by mutating a real (v2-schema) shard receipt into a v1-shaped,
differently-hashed receipt, mirroring the documented 2026-08-03 race.
"""

from __future__ import annotations

import json
import shutil
import zlib
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch

from scripts.research import adjudicate_sorted_all_person_route_landscape_provenance as sut
from scripts.research import build_sorted_all_person_route_landscape as planner
from scripts.research import merge_sorted_all_person_route_landscape as merge_module
from scripts.research import score_sorted_all_person_route_landscape as score_module
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


def _fake_batchable_backend(context_id: str, full_prefix_token_ids: Sequence[int]):
    admission = {
        "schema_version": "batched_full_reforward_parity.v1",
        "status": "not_requested",
        "requested_batch_size": 1,
        "effective_batch_size": 1,
    }
    return _fake_scalar_backend(context_id, full_prefix_token_ids), admission


def _fake_attestation(*, rule_digest: str) -> scorer.AttestationContext:
    return scorer.build_attestation_context(
        expected_vocab_size=VOCAB_SIZE,
        tokenizer_identity={"tokenizer": "fake-v1"},
        model_identity={"model": "fake-v1"},
        rule_digest=rule_digest,
        runtime_receipt_id="fake-runtime-receipt",
    )


def _write_shard(
    plan_dir: Path,
    tmp_path: Path,
    name: str,
    *,
    include_context_ids: list[str] | None,
    include_request_kinds: list[str] | None = None,
    include_request_ids: list[str] | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
) -> Path:
    plan = score_module.load_plan(plan_dir)
    selected, selection = score_module.select_requests(
        plan,
        include_context_ids=include_context_ids,
        include_request_kinds=include_request_kinds,
        include_request_ids=include_request_ids,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    attestation = _fake_attestation(rule_digest=str(plan.receipt["receipt_content_sha256"]))
    rows, scoring_meta = score_module.score_selected_requests(
        plan,
        selected,
        open_scalar_backend=_fake_scalar_backend,
        open_batchable_backend=_fake_batchable_backend,
        attestation=attestation,
    )
    vocab_attestation = rows[0]["raw_model_logprob"]["vocab_attestation"]["x1"]
    receipt = score_module.build_shard_receipt(
        plan=plan,
        selection=selection,
        rows=rows,
        backend_admission={
            "selected_backend": scorer.FULL_REFORWARD_SCORING_BACKEND,
            "cache_enabled": False,
            "use_cache": False,
            **scoring_meta,
        },
        source_identity={
            "model_identity_sha256": vocab_attestation["model_identity_digest"],
            "tokenizer_identity_sha256": vocab_attestation["tokenizer_identity_digest"],
        },
        environment={"note": "fake-cpu-smoke"},
    )
    output_dir = tmp_path / name
    score_module.write_shard(output_dir, rows=rows, receipt=receipt)
    return output_dir


def _shard(shard_dir: Path) -> merge_module.ShardInput:
    return merge_module.ShardInput(
        scores_path=shard_dir / score_module.OUTPUT_JSONL_NAME,
        receipt_path=shard_dir / score_module.OUTPUT_RECEIPT_NAME,
    )


def _rewrite_as_legacy_v1_with_different_hash(shard_dir: Path) -> None:
    """Simulate a pre-fix (v1) receipt: only path/sha256, a *different* hash."""

    receipt_path = shard_dir / score_module.OUTPUT_RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["schema_version"] = sut.LEGACY_RECEIPT_SCHEMA_VERSION_V1
    receipt["code"] = {"path": receipt["code"]["path"], "sha256": "a" * 64}
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")


def _rewrite_single_row_and_receipt(
    shard_dir: Path,
    *,
    row_updates: dict[str, object] | None = None,
    receipt_updates: dict[str, object] | None = None,
) -> None:
    scores_path = shard_dir / score_module.OUTPUT_JSONL_NAME
    row = json.loads(scores_path.read_text(encoding="utf-8"))
    row.update(row_updates or {})
    scores_path.write_bytes(score_module.canonical_json_bytes(row) + b"\n")

    receipt_path = shard_dir / score_module.OUTPUT_RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.update(receipt_updates or {})
    if "request_id" in (row_updates or {}):
        request_id = str(row["request_id"])
        receipt["row_ids"] = [request_id]
        receipt["selection"]["shard_request_ids"] = [request_id]
        selection = receipt["selection"]
        selection["selection_sha256"] = score_module.sha256_json(
            {key: value for key, value in selection.items() if key != "selection_sha256"}
        )
    receipt["receipt_content_sha256"] = score_module.sha256_json(
        {key: value for key, value in receipt.items() if key != "receipt_content_sha256"}
    )
    receipt_path.write_bytes(score_module.canonical_json_bytes(receipt) + b"\n")


@pytest.fixture()
def heterogeneous_shard_pair(plan_dir: Path, tmp_path: Path) -> tuple[Path, Path]:
    """Two shards covering disjoint contexts: one real v2, one v1-shaped/different-hash."""

    v2_shard = _write_shard(
        plan_dir, tmp_path, "v2-shard", include_context_ids=["self-due-gt17"], include_request_kinds=["primary", "sidecar"]
    )
    v1_shard = _write_shard(
        plan_dir, tmp_path, "v1-shard", include_context_ids=["self-due-gt2"], include_request_kinds=["primary", "sidecar"]
    )
    _rewrite_as_legacy_v1_with_different_hash(v1_shard)
    return v2_shard, v1_shard


# ---------------------------------------------------------------------------
# Opt-in gate
# ---------------------------------------------------------------------------


def test_adjudicate_requires_explicit_acknowledgement(
    plan_dir: Path, heterogeneous_shard_pair: tuple[Path, Path], tmp_path: Path
) -> None:
    v2_shard, v1_shard = heterogeneous_shard_pair
    with pytest.raises(sut.ProvenanceAdjudicationError, match="discovery-only"):
        sut.adjudicate(
            plan_dir=plan_dir,
            shards=[_shard(v2_shard), _shard(v1_shard)],
            include_context_ids=["self-due-gt17", "self-due-gt2"],
            include_request_kinds=["primary", "sidecar"],
            independent_repeat_shard_dirs=[],
            output_dir=tmp_path / "salvage",
            acknowledge_discovery_only_salvage=False,
        )


def test_adjudicate_refuses_when_shards_are_already_uniform(plan_dir: Path, tmp_path: Path) -> None:
    shard_a = _write_shard(
        plan_dir, tmp_path, "uniform-a", include_context_ids=["self-due-gt17"], include_request_kinds=["primary", "sidecar"]
    )
    shard_b = _write_shard(
        plan_dir, tmp_path, "uniform-b", include_context_ids=["self-due-gt2"], include_request_kinds=["primary", "sidecar"]
    )
    with pytest.raises(sut.ProvenanceAdjudicationError, match="already code-uniform"):
        sut.adjudicate(
            plan_dir=plan_dir,
            shards=[_shard(shard_a), _shard(shard_b)],
            include_context_ids=["self-due-gt17", "self-due-gt2"],
            include_request_kinds=["primary", "sidecar"],
            independent_repeat_shard_dirs=[],
            output_dir=tmp_path / "salvage",
            acknowledge_discovery_only_salvage=True,
        )


# ---------------------------------------------------------------------------
# Success path: narrowed, discovery-only fields
# ---------------------------------------------------------------------------


def test_adjudicate_succeeds_and_emits_narrowed_discovery_only_fields(
    plan_dir: Path, heterogeneous_shard_pair: tuple[Path, Path], tmp_path: Path
) -> None:
    v2_shard, v1_shard = heterogeneous_shard_pair
    receipt = sut.adjudicate(
        plan_dir=plan_dir,
        shards=[_shard(v2_shard), _shard(v1_shard)],
        include_context_ids=["self-due-gt17", "self-due-gt2"],
        include_request_kinds=["primary", "sidecar"],
        independent_repeat_shard_dirs=[],
        output_dir=tmp_path / "salvage",
        acknowledge_discovery_only_salvage=True,
    )

    assert receipt["schema_version"] == sut.ADJUDICATION_SCHEMA_VERSION
    assert receipt["schema_version"] != merge_module.MERGE_SCHEMA_VERSION
    assert receipt["claim_scope"] == "descriptive_discovery_only"
    assert receipt["provenance_disposition"] == "post_capture_hash_race_adjudicated"
    assert receipt["uniform_executed_code"] == "unproven"
    assert receipt["scalar_batch_admission"] == "unresolved"
    assert receipt["causal_confirmation"] is False
    assert receipt["strict_merge_pass"] is False

    observed_hashes = receipt["rationale"]["observed_code_sha256"]
    assert len(observed_hashes) == 2
    observed_schemas = receipt["rationale"]["observed_receipt_schema_versions"]
    assert sut.LEGACY_RECEIPT_SCHEMA_VERSION_V1 in observed_schemas
    assert score_module.RECEIPT_SCHEMA_VERSION in observed_schemas
    assert "memo" in receipt["rationale"]["memo_reuse_caveat"].lower()

    shard_entries = {tuple(entry["context_ids_covered"]): entry for entry in receipt["shards"]}
    assert len(shard_entries) == 2
    for entry in receipt["shards"]:
        assert "code_identity" in entry
        assert entry["code_identity"]["sha256"]

    assert receipt["counts"]["rows"] == 748  # two contexts x (369 primary + 5 sidecar)
    assert receipt["independent_process_numerical_repeat_admission"]["status"] == "not_supplied"
    assert receipt["token_execution_parity_attestation"]["status"] == "passed"
    assert receipt["output_artifacts"]["salvaged_scores"]["status"] == "created"
    assert Path(receipt["output_artifacts"]["salvaged_scores"]["path"]).is_file()


def test_adjudicate_create_or_identical(
    plan_dir: Path, heterogeneous_shard_pair: tuple[Path, Path], tmp_path: Path
) -> None:
    v2_shard, v1_shard = heterogeneous_shard_pair
    kwargs = dict(
        plan_dir=plan_dir,
        shards=[_shard(v2_shard), _shard(v1_shard)],
        include_context_ids=["self-due-gt17", "self-due-gt2"],
        include_request_kinds=["primary", "sidecar"],
        independent_repeat_shard_dirs=[],
        output_dir=tmp_path / "salvage",
        acknowledge_discovery_only_salvage=True,
    )
    first = sut.adjudicate(**kwargs)
    assert first["output_artifacts"]["adjudication_receipt"]["status"] == "created"
    second = sut.adjudicate(**kwargs)
    assert second["output_artifacts"]["adjudication_receipt"]["status"] == "identical_existing_output"


# ---------------------------------------------------------------------------
# Never erases a genuine (unknown) mismatch
# ---------------------------------------------------------------------------


def test_adjudicate_rejects_content_disagreement_between_overlapping_shards(
    plan_dir: Path, tmp_path: Path
) -> None:
    shard_a = _write_shard(
        plan_dir, tmp_path, "overlap-a", include_context_ids=["self-due-gt17"], include_request_kinds=["primary", "sidecar"]
    )
    shard_b_dir = tmp_path / "overlap-b"
    shutil.copytree(shard_a, shard_b_dir)
    _rewrite_as_legacy_v1_with_different_hash(shard_b_dir)
    scores_path = shard_b_dir / score_module.OUTPUT_JSONL_NAME
    rows = [json.loads(line) for line in scores_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["raw_model_logprob"]["complete_box_logprob_sum"] += 1.0
    scores_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")

    # Reused directly from merge_module._reconcile_shards, so this surfaces
    # as MergeContractError, not ProvenanceAdjudicationError -- the shared
    # strict-content-disagreement gate is identical to strict merge's own.
    with pytest.raises(merge_module.MergeContractError, match="disagree"):
        sut.adjudicate(
            plan_dir=plan_dir,
            shards=[_shard(shard_a), _shard(shard_b_dir)],
            include_context_ids=["self-due-gt17"],
            include_request_kinds=["primary", "sidecar"],
            independent_repeat_shard_dirs=[],
            output_dir=tmp_path / "salvage",
            acknowledge_discovery_only_salvage=True,
        )


def test_adjudicate_rejects_coverage_gap(plan_dir: Path, heterogeneous_shard_pair: tuple[Path, Path], tmp_path: Path) -> None:
    v2_shard, _v1_shard = heterogeneous_shard_pair
    with pytest.raises(sut.ProvenanceAdjudicationError, match="already code-uniform|missing|coverage"):
        sut.adjudicate(
            plan_dir=plan_dir,
            shards=[_shard(v2_shard)],
            include_context_ids=["self-due-gt17", "self-due-gt2"],
            include_request_kinds=["primary", "sidecar"],
            independent_repeat_shard_dirs=[],
            output_dir=tmp_path / "salvage",
            acknowledge_discovery_only_salvage=True,
        )


def test_adjudicate_rejects_source_identity_divergence(
    plan_dir: Path, heterogeneous_shard_pair: tuple[Path, Path], tmp_path: Path
) -> None:
    v2_shard, v1_shard = heterogeneous_shard_pair
    receipt_path = v1_shard / score_module.OUTPUT_RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["source_identity"]["model_identity_sha256"] = "a-genuinely-different-checkpoint"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")

    with pytest.raises(sut.ProvenanceAdjudicationError, match="uniform source_identity"):
        sut.adjudicate(
            plan_dir=plan_dir,
            shards=[_shard(v2_shard), _shard(v1_shard)],
            include_context_ids=["self-due-gt17", "self-due-gt2"],
            include_request_kinds=["primary", "sidecar"],
            independent_repeat_shard_dirs=[],
            output_dir=tmp_path / "salvage",
            acknowledge_discovery_only_salvage=True,
        )


def test_adjudicate_rejects_score_digest_mutation(
    plan_dir: Path, heterogeneous_shard_pair: tuple[Path, Path], tmp_path: Path
) -> None:
    v2_shard, v1_shard = heterogeneous_shard_pair
    scores_path = v1_shard / score_module.OUTPUT_JSONL_NAME
    scores_path.write_text(scores_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    # Reused directly from score_module._read_jsonl (blank-line rejection),
    # so this surfaces as ScoreRunError, not ProvenanceAdjudicationError.
    with pytest.raises(score_module.ScoreRunError, match="blank"):
        sut.adjudicate(
            plan_dir=plan_dir,
            shards=[_shard(v2_shard), _shard(v1_shard)],
            include_context_ids=["self-due-gt17", "self-due-gt2"],
            include_request_kinds=["primary", "sidecar"],
            independent_repeat_shard_dirs=[],
            output_dir=tmp_path / "salvage",
            acknowledge_discovery_only_salvage=True,
        )


def test_adjudicate_rejects_foreign_receipt_schema_version(
    plan_dir: Path, heterogeneous_shard_pair: tuple[Path, Path], tmp_path: Path
) -> None:
    v2_shard, v1_shard = heterogeneous_shard_pair
    receipt_path = v1_shard / score_module.OUTPUT_RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["schema_version"] = "totally-unknown-schema.v9"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")

    with pytest.raises(sut.ProvenanceAdjudicationError, match="unknown mismatch"):
        sut.adjudicate(
            plan_dir=plan_dir,
            shards=[_shard(v2_shard), _shard(v1_shard)],
            include_context_ids=["self-due-gt17", "self-due-gt2"],
            include_request_kinds=["primary", "sidecar"],
            independent_repeat_shard_dirs=[],
            output_dir=tmp_path / "salvage",
            acknowledge_discovery_only_salvage=True,
        )


# ---------------------------------------------------------------------------
# Independent-process numerical-repeat evidence
# ---------------------------------------------------------------------------


@pytest.fixture()
def independent_repeat_shard_dirs(plan_dir: Path, tmp_path: Path) -> list[Path]:
    dirs: list[Path] = []
    for index in range(8):
        shard_dir = _write_shard(
            plan_dir,
            tmp_path,
            f"independent-{index}",
            include_context_ids=[merge_module.NUMERICAL_REPEAT_CONTEXT_ID],
            include_request_kinds=["numerical_repeat"],
            shard_index=index,
            shard_count=merge_module.NUMERICAL_REPEAT_COUNT,
        )
        dirs.append(shard_dir)
    return dirs


def test_independent_process_numerical_repeat_admission_zero_delta(independent_repeat_shard_dirs: list[Path]) -> None:
    admission = sut._independent_process_numerical_repeat_admission(independent_repeat_shard_dirs)  # noqa: SLF001
    assert admission["status"] == "computed"
    assert admission["evidence_class"] == "independent_process"
    assert admission["shard_count"] == 8
    assert admission["repeat_indices"] == list(range(8))
    assert admission["request_ids"] == [f"numerical-repeat:self-due-gt17:{index}" for index in range(8)]
    assert admission["delta"] == 0.0
    assert admission["epsilon"] == sut.UNIT_NUMERICAL_EPSILON_FLOOR


def test_independent_process_numerical_repeat_admission_not_supplied() -> None:
    assert sut._independent_process_numerical_repeat_admission([])["status"] == "not_supplied"  # noqa: SLF001


def test_independent_process_numerical_repeat_admission_rejects_duplicate_and_missing_ordinals(
    independent_repeat_shard_dirs: list[Path],
) -> None:
    dirs = independent_repeat_shard_dirs[:-1] + [independent_repeat_shard_dirs[0]]
    with pytest.raises(sut.ProvenanceAdjudicationError, match="repeat_index domain"):
        sut._independent_process_numerical_repeat_admission(dirs)  # noqa: SLF001


def test_independent_process_numerical_repeat_admission_rejects_foreign_ordinal(
    independent_repeat_shard_dirs: list[Path],
) -> None:
    foreign = independent_repeat_shard_dirs[-1]
    _rewrite_single_row_and_receipt(
        foreign,
        row_updates={"request_id": "numerical-repeat:self-due-gt17:8", "repeat_index": 8},
    )
    with pytest.raises(sut.ProvenanceAdjudicationError, match="frozen repeat_index domain"):
        sut._independent_process_numerical_repeat_admission(independent_repeat_shard_dirs)  # noqa: SLF001


def test_independent_process_numerical_repeat_admission_rejects_non_repeat_kind(
    independent_repeat_shard_dirs: list[Path],
) -> None:
    _rewrite_single_row_and_receipt(
        independent_repeat_shard_dirs[-1],
        row_updates={"request_kind": "primary"},
    )
    with pytest.raises(sut.ProvenanceAdjudicationError, match="request_kind"):
        sut._independent_process_numerical_repeat_admission(independent_repeat_shard_dirs)  # noqa: SLF001


def test_independent_process_numerical_repeat_admission_rejects_different_measurement(
    independent_repeat_shard_dirs: list[Path],
) -> None:
    _rewrite_single_row_and_receipt(
        independent_repeat_shard_dirs[-1],
        row_updates={"candidate_id": "foreign-candidate"},
    )
    with pytest.raises(sut.ProvenanceAdjudicationError, match="measurement identity"):
        sut._independent_process_numerical_repeat_admission(independent_repeat_shard_dirs)  # noqa: SLF001


def test_independent_process_numerical_repeat_admission_rejects_different_coord_tokens(
    independent_repeat_shard_dirs: list[Path],
) -> None:
    foreign_coord_token_ids = [1, 2, 3, 4]
    _rewrite_single_row_and_receipt(
        independent_repeat_shard_dirs[-1],
        row_updates={
            "coord_token_ids": foreign_coord_token_ids,
            "coord_token_ids_sha256": score_module.sha256_json(foreign_coord_token_ids),
        },
    )
    with pytest.raises(sut.ProvenanceAdjudicationError, match="measurement identity"):
        sut._independent_process_numerical_repeat_admission(independent_repeat_shard_dirs)  # noqa: SLF001


def test_independent_process_numerical_repeat_admission_rejects_different_source_identity(
    independent_repeat_shard_dirs: list[Path],
) -> None:
    receipt_path = independent_repeat_shard_dirs[-1] / score_module.OUTPUT_RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    source_identity = dict(receipt["source_identity"])
    source_identity["source_jsonl"] = {"path": "foreign.jsonl", "sha256": "foreign-source"}
    _rewrite_single_row_and_receipt(
        independent_repeat_shard_dirs[-1],
        receipt_updates={"source_identity": source_identity},
    )
    with pytest.raises(sut.ProvenanceAdjudicationError, match="source/model/plan/scoring identity"):
        sut._independent_process_numerical_repeat_admission(independent_repeat_shard_dirs)  # noqa: SLF001


def test_independent_process_numerical_repeat_admission_rejects_different_scoring_identity(
    independent_repeat_shard_dirs: list[Path],
) -> None:
    receipt_path = independent_repeat_shard_dirs[-1] / score_module.OUTPUT_RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    backend_admission = dict(receipt["scoring_backend_admission"])
    backend_admission["requested_batch_size"] = 2
    _rewrite_single_row_and_receipt(
        independent_repeat_shard_dirs[-1],
        receipt_updates={"scoring_backend_admission": backend_admission},
    )
    with pytest.raises(sut.ProvenanceAdjudicationError, match="source/model/plan/scoring identity"):
        sut._independent_process_numerical_repeat_admission(independent_repeat_shard_dirs)  # noqa: SLF001


def test_adjudicate_embeds_independent_process_evidence(
    plan_dir: Path,
    heterogeneous_shard_pair: tuple[Path, Path],
    independent_repeat_shard_dirs: list[Path],
    tmp_path: Path,
) -> None:
    v2_shard, v1_shard = heterogeneous_shard_pair
    receipt = sut.adjudicate(
        plan_dir=plan_dir,
        shards=[_shard(v2_shard), _shard(v1_shard)],
        include_context_ids=["self-due-gt17", "self-due-gt2"],
        include_request_kinds=["primary", "sidecar"],
        independent_repeat_shard_dirs=independent_repeat_shard_dirs,
        output_dir=tmp_path / "salvage",
        acknowledge_discovery_only_salvage=True,
    )
    independent = receipt["independent_process_numerical_repeat_admission"]
    assert independent["status"] == "computed"
    assert independent["delta"] == 0.0
    # Distinct from (never merged into) the in-plan-coverage repeat admission.
    assert receipt["numerical_repeat_admission"]["status"] == "not_included_in_this_merge"


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def test_resolve_shards_from_shard_dir_flag(heterogeneous_shard_pair: tuple[Path, Path]) -> None:
    v2_shard, v1_shard = heterogeneous_shard_pair
    args = sut.build_parser().parse_args(
        [
            "--plan-dir",
            "/nonexistent",
            "--shard-dir",
            str(v2_shard),
            "--shard-dir",
            str(v1_shard),
            "--output-dir",
            "/nonexistent-output",
            "--acknowledge-discovery-only-salvage",
        ]
    )
    shards = sut._resolve_shards(args)  # noqa: SLF001
    assert len(shards) == 2
    assert args.acknowledge_discovery_only_salvage is True
