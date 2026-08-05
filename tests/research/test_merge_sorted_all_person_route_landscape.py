"""Targeted CPU-only tests for merge_sorted_all_person_route_landscape.py.

Shards are produced with the real scorer's own pipeline
(``score_sorted_all_person_route_landscape.score_selected_requests`` /
``build_shard_receipt`` / ``write_shard``) against the real planner output,
using a fully faked scoring backend -- no GPU, no HF backend session, no
real ``transformers`` model.
"""

from __future__ import annotations

import json
import zlib
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch

from scripts.research import build_sorted_all_person_route_landscape as planner
from scripts.research import merge_sorted_all_person_route_landscape as sut
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


def _fake_attestation() -> scorer.AttestationContext:
    return scorer.build_attestation_context(
        expected_vocab_size=VOCAB_SIZE,
        tokenizer_identity={"tokenizer": "fake-v1"},
        model_identity={"model": "fake-v1"},
        rule_digest="fake-rule-digest",
        runtime_receipt_id="fake-runtime-receipt",
    )


def _write_shard(
    plan_dir: Path,
    tmp_path: Path,
    name: str,
    *,
    include_context_ids: list[str] | None,
    include_request_kinds: list[str] | None,
) -> Path:
    plan = score_module.load_plan(plan_dir)
    selected, selection = score_module.select_requests(
        plan, include_context_ids=include_context_ids, include_request_kinds=include_request_kinds
    )
    rows, scoring_meta = score_module.score_selected_requests(
        plan,
        selected,
        open_scalar_backend=_fake_scalar_backend,
        open_batchable_backend=_fake_batchable_backend,
        attestation=_fake_attestation(),
    )
    receipt = score_module.build_shard_receipt(
        plan=plan,
        selection=selection,
        rows=rows,
        backend_admission={"selected_backend": scorer.FULL_REFORWARD_SCORING_BACKEND, **scoring_meta},
        source_identity={
            "model_identity_sha256": "fake-model",
            "tokenizer_identity_sha256": "fake-tokenizer",
        },
        environment={"note": "fake-cpu-smoke"},
    )
    output_dir = tmp_path / name
    score_module.write_shard(output_dir, rows=rows, receipt=receipt)
    return output_dir


@pytest.fixture()
def repeats_shard_dir(plan_dir: Path, tmp_path: Path) -> Path:
    return _write_shard(
        plan_dir,
        tmp_path,
        "repeats-shard",
        include_context_ids=["self-due-gt17"],
        include_request_kinds=["numerical_repeat"],
    )


@pytest.fixture()
def primary_sidecar_shard_dir(plan_dir: Path, tmp_path: Path) -> Path:
    return _write_shard(
        plan_dir,
        tmp_path,
        "primary-sidecar-shard",
        include_context_ids=["self-due-gt17"],
        include_request_kinds=["primary", "sidecar"],
    )


def _shard(shard_dir: Path) -> sut.ShardInput:
    return sut.ShardInput(
        scores_path=shard_dir / score_module.OUTPUT_JSONL_NAME,
        receipt_path=shard_dir / score_module.OUTPUT_RECEIPT_NAME,
    )


# ---------------------------------------------------------------------------
# Exact partition: no missing/duplicate/foreign request ids
# ---------------------------------------------------------------------------


def test_merge_two_disjoint_shards_exact_partition_and_counts(
    plan_dir: Path, repeats_shard_dir: Path, primary_sidecar_shard_dir: Path, tmp_path: Path
) -> None:
    receipt = sut.merge_shards(
        plan_dir=plan_dir,
        shards=[_shard(repeats_shard_dir), _shard(primary_sidecar_shard_dir)],
        include_context_ids=["self-due-gt17"],
        include_request_kinds=None,
        output_dir=tmp_path / "merged",
    )
    assert receipt["counts"]["rows"] == 382  # 369 primary + 5 sidecar + 8 repeat
    assert receipt["counts"]["rows_by_request_kind"] == {"primary": 369, "sidecar": 5, "numerical_repeat": 8}
    assert receipt["covered_context_ids"] == ["self-due-gt17"]
    assert receipt["raw_capture_validity"]["status"] == "passed"
    assert receipt["output_artifacts"]["merged_scores"]["status"] == "created"
    assert Path(receipt["output_artifacts"]["merged_scores"]["path"]).is_file()


def test_merge_rejects_missing_requests(plan_dir: Path, repeats_shard_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(sut.MergeContractError, match="missing"):
        sut.merge_shards(
            plan_dir=plan_dir,
            shards=[_shard(repeats_shard_dir)],
            include_context_ids=["self-due-gt17"],
            include_request_kinds=None,
            output_dir=tmp_path / "merged",
        )


def test_merge_rejects_extra_requests_outside_declared_selection(
    plan_dir: Path, repeats_shard_dir: Path, primary_sidecar_shard_dir: Path, tmp_path: Path
) -> None:
    with pytest.raises(sut.MergeContractError, match="extra|missing"):
        sut.merge_shards(
            plan_dir=plan_dir,
            shards=[_shard(repeats_shard_dir), _shard(primary_sidecar_shard_dir)],
            include_context_ids=["self-due-gt17"],
            include_request_kinds=["numerical_repeat"],  # excludes the primary/sidecar shard's rows
            output_dir=tmp_path / "merged",
        )


def test_merge_rejects_disagreeing_overlap(
    plan_dir: Path, repeats_shard_dir: Path, tmp_path: Path
) -> None:
    tampered_dir = tmp_path / "tampered-repeats"
    import shutil

    shutil.copytree(repeats_shard_dir, tampered_dir)
    scores_path = tampered_dir / score_module.OUTPUT_JSONL_NAME
    rows = [json.loads(line) for line in scores_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["raw_model_logprob"]["complete_box_logprob_sum"] += 1.0
    scores_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")

    with pytest.raises(sut.MergeContractError, match="does not match the exact request-id set|disagree"):
        sut.merge_shards(
            plan_dir=plan_dir,
            shards=[_shard(repeats_shard_dir), _shard(tampered_dir)],
            include_context_ids=["self-due-gt17"],
            include_request_kinds=["numerical_repeat"],
            output_dir=tmp_path / "merged",
        )


def test_merge_allows_byte_identical_overlap(plan_dir: Path, repeats_shard_dir: Path, tmp_path: Path) -> None:
    receipt = sut.merge_shards(
        plan_dir=plan_dir,
        shards=[_shard(repeats_shard_dir), _shard(repeats_shard_dir)],
        include_context_ids=["self-due-gt17"],
        include_request_kinds=["numerical_repeat"],
        output_dir=tmp_path / "merged",
    )
    assert receipt["counts"]["rows"] == 8
    assert receipt["dedup_accounting"]["shard_count"] == 2
    assert receipt["dedup_accounting"]["union_request_count"] == 8


def test_merge_rejects_shard_bound_to_a_different_plan(
    plan_dir: Path, repeats_shard_dir: Path, tmp_path: Path
) -> None:
    # The planner is fully deterministic (create-or-identical against the
    # same frozen production sources), so two independently built plans are
    # byte-identical; simulate a genuinely foreign plan binding by tampering
    # the shard receipt's declared plan.receipt_content_sha256 directly.
    tampered_dir = tmp_path / "foreign-plan-binding"
    import shutil

    shutil.copytree(repeats_shard_dir, tampered_dir)
    receipt_path = tampered_dir / score_module.OUTPUT_RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["plan"]["receipt_content_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")

    with pytest.raises(sut.MergeContractError, match="different plan"):
        sut.merge_shards(
            plan_dir=plan_dir,
            shards=[_shard(tampered_dir)],
            include_context_ids=["self-due-gt17"],
            include_request_kinds=["numerical_repeat"],
            output_dir=tmp_path / "merged",
        )


# ---------------------------------------------------------------------------
# Batch parity receipts
# ---------------------------------------------------------------------------


def test_merge_rejects_bare_failed_batch_admission_with_no_fallback(
    plan_dir: Path, repeats_shard_dir: Path, tmp_path: Path
) -> None:
    tampered_dir = tmp_path / "tampered-batch"
    import shutil

    shutil.copytree(repeats_shard_dir, tampered_dir)
    receipt_path = tampered_dir / score_module.OUTPUT_RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["scoring_backend_admission"]["per_context_accounting"] = [
        {
            "context_id": "self-due-gt17",
            "batch_admission": {"status": "failed_scalar_fallback_required", "effective_batch_size": 1},
        }
    ]
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")

    with pytest.raises(sut.MergeContractError, match="no explicit recorded fallback"):
        sut.merge_shards(
            plan_dir=plan_dir,
            shards=[_shard(tampered_dir)],
            include_context_ids=["self-due-gt17"],
            include_request_kinds=["numerical_repeat"],
            output_dir=tmp_path / "merged",
        )


def test_merge_republishes_explicit_scalar_fallback(
    plan_dir: Path, repeats_shard_dir: Path, tmp_path: Path
) -> None:
    tampered_dir = tmp_path / "fallback-batch"
    import shutil

    shutil.copytree(repeats_shard_dir, tampered_dir)
    receipt_path = tampered_dir / score_module.OUTPUT_RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["scoring_backend_admission"]["per_context_accounting"] = [
        {
            "context_id": "self-due-gt17",
            "batch_admission": {
                "status": "failed_scalar_fallback_required",
                "effective_batch_size": 1,
                "fallback": "scalar_explicit",
            },
        }
    ]
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")

    merged = sut.merge_shards(
        plan_dir=plan_dir,
        shards=[_shard(tampered_dir)],
        include_context_ids=["self-due-gt17"],
        include_request_kinds=["numerical_repeat"],
        output_dir=tmp_path / "merged",
    )
    assert merged["batch_parity_receipts"][0]["batch_admission"]["fallback"] == "scalar_explicit"


# ---------------------------------------------------------------------------
# Numerical repeat epsilon (frozen formula, independently recomputed)
# ---------------------------------------------------------------------------


def test_numerical_repeat_admission_matches_frozen_formula_and_zero_delta(
    plan_dir: Path, repeats_shard_dir: Path, tmp_path: Path
) -> None:
    merged = sut.merge_shards(
        plan_dir=plan_dir,
        shards=[_shard(repeats_shard_dir)],
        include_context_ids=["self-due-gt17"],
        include_request_kinds=["numerical_repeat"],
        output_dir=tmp_path / "merged",
    )
    admission = merged["numerical_repeat_admission"]
    assert admission["status"] == "computed"
    assert admission["repeat_count"] == 8
    assert admission["context_id"] == "self-due-gt17"
    # A deterministic fake backend keyed purely on literal token content
    # produces byte-identical logits for every repeat (they share the exact
    # same context prefix and coordinate tokens), so delta is exactly zero
    # and epsilon collapses to the frozen floor.
    assert admission["delta"] == 0.0
    assert admission["epsilon"] == sut.EPSILON_FLOOR


def test_numerical_repeat_admission_not_included_when_absent(
    plan_dir: Path, primary_sidecar_shard_dir: Path, tmp_path: Path
) -> None:
    merged = sut.merge_shards(
        plan_dir=plan_dir,
        shards=[_shard(primary_sidecar_shard_dir)],
        include_context_ids=["self-due-gt17"],
        include_request_kinds=["primary", "sidecar"],
        output_dir=tmp_path / "merged",
    )
    assert merged["numerical_repeat_admission"]["status"] == "not_included_in_this_merge"


# ---------------------------------------------------------------------------
# Raw-capture validity vs. teacher-forced prefix parity residual
# ---------------------------------------------------------------------------


def test_residual_disclosure_never_claims_teacher_forced_parity_passed(
    plan_dir: Path, repeats_shard_dir: Path, tmp_path: Path
) -> None:
    merged = sut.merge_shards(
        plan_dir=plan_dir,
        shards=[_shard(repeats_shard_dir)],
        include_context_ids=["self-due-gt17"],
        include_request_kinds=["numerical_repeat"],
        output_dir=tmp_path / "merged",
    )
    residual = merged["residual_disclosures"]
    assert residual["status"] == "unverified_by_this_merge"
    per_context = residual["per_context_teacher_forced_chosen_token_parity"]
    assert "self-due-gt17" in per_context
    assert per_context["self-due-gt17"]["claimed_pass"] is False
    assert merged["raw_capture_validity"]["status"] == "passed"


# ---------------------------------------------------------------------------
# Create-or-identical + --force retry
# ---------------------------------------------------------------------------


def test_merge_create_or_identical_and_force_overwrite(
    plan_dir: Path, repeats_shard_dir: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "merged"
    first = sut.merge_shards(
        plan_dir=plan_dir,
        shards=[_shard(repeats_shard_dir)],
        include_context_ids=["self-due-gt17"],
        include_request_kinds=["numerical_repeat"],
        output_dir=output_dir,
    )
    assert first["output_artifacts"]["merged_receipt"]["status"] == "created"

    second = sut.merge_shards(
        plan_dir=plan_dir,
        shards=[_shard(repeats_shard_dir)],
        include_context_ids=["self-due-gt17"],
        include_request_kinds=["numerical_repeat"],
        output_dir=output_dir,
    )
    assert second["output_artifacts"]["merged_receipt"]["status"] == "identical_existing_output"

    # Mutate the sealed receipt on disk directly, then confirm a non-forced
    # re-merge refuses and a forced one overwrites.
    receipt_path = output_dir / sut.MERGED_RECEIPT_NAME
    tampered = json.loads(receipt_path.read_text(encoding="utf-8"))
    tampered["counts"]["rows"] = 999
    receipt_path.write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8")

    # _write_create_or_identical is reused directly from score_module, so its
    # failure surfaces as score_module.ScoreRunError, not MergeContractError.
    with pytest.raises(score_module.ScoreRunError, match="already exists with different"):
        sut.merge_shards(
            plan_dir=plan_dir,
            shards=[_shard(repeats_shard_dir)],
            include_context_ids=["self-due-gt17"],
            include_request_kinds=["numerical_repeat"],
            output_dir=output_dir,
        )

    forced = sut.merge_shards(
        plan_dir=plan_dir,
        shards=[_shard(repeats_shard_dir)],
        include_context_ids=["self-due-gt17"],
        include_request_kinds=["numerical_repeat"],
        output_dir=output_dir,
        force=True,
    )
    assert forced["output_artifacts"]["merged_receipt"]["status"] == "overwritten_forced"


# ---------------------------------------------------------------------------
# Shard/code/source identity uniformity
# ---------------------------------------------------------------------------


def test_merge_rejects_shards_with_differing_source_identity(
    plan_dir: Path, repeats_shard_dir: Path, tmp_path: Path
) -> None:
    tampered_dir = tmp_path / "different-source-identity"
    import shutil

    shutil.copytree(repeats_shard_dir, tampered_dir)
    receipt_path = tampered_dir / score_module.OUTPUT_RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["source_identity"]["model_identity_sha256"] = "a-different-model"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")

    with pytest.raises(sut.MergeContractError, match="uniform source_identity"):
        sut.merge_shards(
            plan_dir=plan_dir,
            shards=[_shard(repeats_shard_dir), _shard(tampered_dir)],
            include_context_ids=["self-due-gt17"],
            include_request_kinds=["numerical_repeat"],
            output_dir=tmp_path / "merged",
        )


def test_merge_rejects_shards_with_differing_code_hash(
    plan_dir: Path, repeats_shard_dir: Path, tmp_path: Path
) -> None:
    """Regression: the exact 2026-08-03 code-identity race scenario.

    Strict merge must still reject a mixed code.sha256 set outright -- this
    is never relaxed silently. Only the explicit, opt-in discovery-only
    salvage path (``adjudicate_sorted_all_person_route_landscape_provenance.py``)
    examines shards like this.
    """

    tampered_dir = tmp_path / "different-code-hash"
    import shutil

    shutil.copytree(repeats_shard_dir, tampered_dir)
    receipt_path = tampered_dir / score_module.OUTPUT_RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["code"]["sha256"] = "0" * 64
    receipt["code"]["executed_source_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")

    with pytest.raises(sut.MergeContractError, match="different scorer code"):
        sut.merge_shards(
            plan_dir=plan_dir,
            shards=[_shard(repeats_shard_dir), _shard(tampered_dir)],
            include_context_ids=["self-due-gt17"],
            include_request_kinds=["numerical_repeat"],
            output_dir=tmp_path / "merged",
        )


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def test_resolve_shards_from_shard_dir_flag(repeats_shard_dir: Path, primary_sidecar_shard_dir: Path) -> None:
    args = sut.build_parser().parse_args(
        [
            "--plan-dir",
            "/nonexistent",
            "--shard-dir",
            str(repeats_shard_dir),
            "--shard-dir",
            str(primary_sidecar_shard_dir),
            "--output-dir",
            "/nonexistent-output",
        ]
    )
    shards = sut._resolve_shards(args)
    assert len(shards) == 2
    assert shards[0].scores_path == repeats_shard_dir / score_module.OUTPUT_JSONL_NAME
