"""Targeted CPU-only tests for attest_sorted_all_person_route_landscape.py.

Builds a real merged surface (via the real scorer + real merger, both fed a
fully faked scoring backend) and attests it, then exercises the attestor's
own tamper/consistency checks directly on the sealed artifacts.
"""

from __future__ import annotations

import dataclasses
import json
import zlib
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch

from scripts.research import attest_sorted_all_person_route_landscape as sut
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


def _fake_attestation() -> scorer.AttestationContext:
    return scorer.build_attestation_context(
        expected_vocab_size=VOCAB_SIZE,
        tokenizer_identity={"tokenizer": "fake-v1"},
        model_identity={"model": "fake-v1"},
        rule_digest="fake-rule-digest",
        runtime_receipt_id="fake-runtime-receipt",
    )


def _write_shard(plan_dir: Path, tmp_path: Path, name: str) -> Path:
    plan = score_module.load_plan(plan_dir)
    selected, selection = score_module.select_requests(
        plan, include_context_ids=["self-due-gt17"], include_request_kinds=["numerical_repeat"]
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
        source_identity={"model_identity_sha256": "fake-model", "tokenizer_identity_sha256": "fake-tokenizer"},
        environment={"note": "fake-cpu-smoke"},
    )
    output_dir = tmp_path / name
    score_module.write_shard(output_dir, rows=rows, receipt=receipt)
    return output_dir


@pytest.fixture()
def merge_receipt_path(plan_dir: Path, tmp_path: Path) -> Path:
    shard_dir = _write_shard(plan_dir, tmp_path, "repeats-shard")
    shard = merge_module.ShardInput(
        scores_path=shard_dir / score_module.OUTPUT_JSONL_NAME,
        receipt_path=shard_dir / score_module.OUTPUT_RECEIPT_NAME,
    )
    merged_dir = tmp_path / "merged"
    merge_module.merge_shards(
        plan_dir=plan_dir,
        shards=[shard],
        include_context_ids=["self-due-gt17"],
        include_request_kinds=["numerical_repeat"],
        output_dir=merged_dir,
    )
    return merged_dir / merge_module.MERGED_RECEIPT_NAME


# ---------------------------------------------------------------------------
# Happy path: full attestation over a real merged surface
# ---------------------------------------------------------------------------


def test_attest_passes_and_binds_plan_identity(plan_dir: Path, merge_receipt_path: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "attestation"
    document = sut.attest(merge_receipt_path=merge_receipt_path, plan_dir=plan_dir, output_dir=output_dir)

    assert document["disposition"] == "accepted"
    assert document["scientific_conclusion"] is None
    assert document["plan"]["receipt_content_sha256"] == score_module.load_plan(plan_dir).receipt["receipt_content_sha256"]
    assert document["counts"]["rows"] == 8
    assert document["coverage_attestation"]["status"] == "passed"
    assert document["coverage_attestation"]["selection_sha256_matches"] is True
    assert document["decision_channel_attestation"]["status"] == "passed"
    assert document["primary_role_separation_attestation"]["numerical_repeat_rows"] == 8
    assert document["batch_parity_attestation"]["status"] == "passed"
    assert document["numerical_repeat_attestation"]["status"] == "computed"
    assert document["numerical_repeat_attestation"]["epsilon"] == merge_module.EPSILON_FLOOR
    assert document["teacher_forced_chosen_token_parity_residual"]["status"] == "unverified_by_this_merge"
    assert document["raw_capture_validity"]["status"] == "passed"
    token_execution = document["token_execution_parity_attestation"]
    assert token_execution["attestation_kind"] == "token_execution_parity"
    assert token_execution["status"] == "passed"
    assert token_execution["not_a_numerical_chosen_token_parity_claim"] is True
    assert token_execution["context_count"] == 1
    assert "self-due-gt17" in token_execution["per_context"]

    output_path = output_dir / sut.ATTESTATION_NAME
    assert output_path.is_file()

    # create-or-identical: re-attesting is idempotent.
    document_again = sut.attest(merge_receipt_path=merge_receipt_path, plan_dir=plan_dir, output_dir=output_dir)
    assert document_again == document


# ---------------------------------------------------------------------------
# token_execution_parity_attestation: literal prefix-arithmetic proof
# ---------------------------------------------------------------------------


def test_token_execution_parity_attestation_passes_over_all_six_contexts(plan_dir: Path) -> None:
    plan = score_module.load_plan(plan_dir)
    attestation = sut.token_execution_parity_attestation(plan)
    assert attestation["status"] == "passed"
    assert attestation["context_count"] == 6
    assert set(attestation["per_context"]) == set(plan.contexts_by_id)
    for context_id, entry in attestation["per_context"].items():
        context = plan.contexts_by_id[context_id]
        assert entry["root_prompt_length"] == len(context["prompt_token_ids"])
        assert entry["full_prefix_length"] == len(context["full_prefix_token_ids"])
        assert entry["reconstruction_status"] == "passed"


def test_token_execution_parity_attestation_restricts_to_requested_contexts(plan_dir: Path) -> None:
    plan = score_module.load_plan(plan_dir)
    attestation = sut.token_execution_parity_attestation(plan, context_ids=["root", "self-due-gt2"])
    assert attestation["context_count"] == 2
    assert set(attestation["per_context"]) == {"root", "self-due-gt2"}


def test_token_execution_parity_attestation_rejects_unknown_context(plan_dir: Path) -> None:
    plan = score_module.load_plan(plan_dir)
    with pytest.raises(sut.RunAttestationError, match="absent from the plan"):
        sut.token_execution_parity_attestation(plan, context_ids=["not-a-real-context"])


def test_token_execution_parity_attestation_detects_broken_reconstruction(plan_dir: Path) -> None:
    plan = score_module.load_plan(plan_dir)
    tampered_context = {**plan.contexts_by_id["root"], "generated_prefix_token_ids": [999999]}
    tampered_contexts_by_id = {**plan.contexts_by_id, "root": tampered_context}
    tampered_plan = dataclasses.replace(plan, contexts_by_id=tampered_contexts_by_id)
    with pytest.raises(sut.RunAttestationError, match="token execution parity violated"):
        sut.token_execution_parity_attestation(tampered_plan, context_ids=["root"])


def test_attest_token_execution_only_standalone_mode(plan_dir: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "token-execution-echo"
    document = sut.attest_token_execution_only(plan_dir=plan_dir, output_dir=output_dir)
    assert document["schema_version"] == sut.TOKEN_EXECUTION_ECHO_SCHEMA_VERSION
    assert document["disposition"] == "accepted"
    assert document["token_execution_parity_attestation"]["context_count"] == 6
    assert (output_dir / sut.TOKEN_EXECUTION_ECHO_NAME).is_file()


def test_attest_token_execution_only_cli_does_not_require_merge_receipt(plan_dir: Path, tmp_path: Path) -> None:
    args = sut.build_parser().parse_args(
        [
            "--plan-dir",
            str(plan_dir),
            "--output-dir",
            str(tmp_path / "cli-echo"),
            "--token-execution-only",
        ]
    )
    assert args.merge_receipt is None
    document = sut.attest_token_execution_only(
        plan_dir=args.plan_dir, output_dir=args.output_dir, include_context_ids=args.include_context_id
    )
    assert document["disposition"] == "accepted"


# ---------------------------------------------------------------------------
# Tamper detection
# ---------------------------------------------------------------------------


def test_attest_rejects_tampered_merge_receipt_content(
    plan_dir: Path, merge_receipt_path: Path, tmp_path: Path
) -> None:
    tampered = json.loads(merge_receipt_path.read_text(encoding="utf-8"))
    tampered["counts"]["rows"] = 999
    merge_receipt_path.write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8")
    with pytest.raises(sut.RunAttestationError, match="receipt_content_sha256 does not reconstruct"):
        sut.attest(merge_receipt_path=merge_receipt_path, plan_dir=plan_dir, output_dir=tmp_path / "attestation")


def test_attest_rejects_stale_merged_scores_file(plan_dir: Path, merge_receipt_path: Path, tmp_path: Path) -> None:
    merge_receipt = json.loads(merge_receipt_path.read_text(encoding="utf-8"))
    scores_path = Path(merge_receipt["output_artifacts"]["merged_scores"]["path"])
    scores_path.write_text(scores_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(sut.RunAttestationError, match="stale merge output"):
        sut.attest(merge_receipt_path=merge_receipt_path, plan_dir=plan_dir, output_dir=tmp_path / "attestation")


def test_attest_rejects_plan_binding_mismatch(
    plan_dir: Path, merge_receipt_path: Path, tmp_path_factory: pytest.TempPathFactory, tmp_path: Path
) -> None:
    other_plan_dir = tmp_path_factory.mktemp("other-plan") / "plan"
    planner.build_sorted_all_person_route_landscape(other_plan_dir)
    merge_receipt = json.loads(merge_receipt_path.read_text(encoding="utf-8"))
    merge_receipt["plan"]["receipt_content_sha256"] = "0" * 64
    # receipt_content_sha256 must still self-reconstruct for this to reach
    # the plan-binding check rather than failing earlier on tamper.
    reconstructed = score_module.sha256_json(
        {key: value for key, value in merge_receipt.items() if key != "receipt_content_sha256"}
    )
    merge_receipt["receipt_content_sha256"] = reconstructed
    tampered_path = tmp_path / "tampered-merge-receipt.json"
    tampered_path.write_text(json.dumps(merge_receipt, sort_keys=True), encoding="utf-8")

    with pytest.raises(sut.RunAttestationError, match="plan.receipt_content_sha256"):
        sut.attest(merge_receipt_path=tampered_path, plan_dir=other_plan_dir, output_dir=tmp_path / "attestation")


def test_attest_rejects_promoted_teacher_forced_parity_claim(
    plan_dir: Path, merge_receipt_path: Path, tmp_path: Path
) -> None:
    merge_receipt = json.loads(merge_receipt_path.read_text(encoding="utf-8"))
    merge_receipt["residual_disclosures"]["per_context_teacher_forced_chosen_token_parity"]["self-due-gt17"][
        "claimed_pass"
    ] = True
    reconstructed = score_module.sha256_json(
        {key: value for key, value in merge_receipt.items() if key != "receipt_content_sha256"}
    )
    merge_receipt["receipt_content_sha256"] = reconstructed
    tampered_path = tmp_path / "tampered-residual.json"
    tampered_path.write_text(json.dumps(merge_receipt, sort_keys=True), encoding="utf-8")

    with pytest.raises(sut.RunAttestationError, match="does not match the plan's own"):
        sut.attest(merge_receipt_path=tampered_path, plan_dir=plan_dir, output_dir=tmp_path / "attestation")


def test_attest_rejects_foreign_schema_version(plan_dir: Path, merge_receipt_path: Path, tmp_path: Path) -> None:
    merge_receipt = json.loads(merge_receipt_path.read_text(encoding="utf-8"))
    merge_receipt["schema_version"] = "foreign.v0"
    tampered_path = tmp_path / "foreign-schema.json"
    tampered_path.write_text(json.dumps(merge_receipt, sort_keys=True), encoding="utf-8")

    with pytest.raises(sut.RunAttestationError, match="schema_version"):
        sut.attest(merge_receipt_path=tampered_path, plan_dir=plan_dir, output_dir=tmp_path / "attestation")


def test_attest_rejects_counts_mismatch(plan_dir: Path, merge_receipt_path: Path, tmp_path: Path) -> None:
    merge_receipt = json.loads(merge_receipt_path.read_text(encoding="utf-8"))
    merge_receipt["counts"]["rows"] = 999
    reconstructed = score_module.sha256_json(
        {key: value for key, value in merge_receipt.items() if key != "receipt_content_sha256"}
    )
    merge_receipt["receipt_content_sha256"] = reconstructed
    tampered_path = tmp_path / "tampered-counts.json"
    tampered_path.write_text(json.dumps(merge_receipt, sort_keys=True), encoding="utf-8")

    with pytest.raises(sut.RunAttestationError, match="counts"):
        sut.attest(merge_receipt_path=tampered_path, plan_dir=plan_dir, output_dir=tmp_path / "attestation")
