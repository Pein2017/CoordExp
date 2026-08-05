"""Contract tests for the Sorted owner-basin shard merger."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import pytest


REPO_ROOT = Path("/data/CoordExp/.worktrees/research-probes")
MERGER_PATH = REPO_ROOT / "scripts/research/merge_sorted_owner_basin_landscape_shards.py"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import (  # noqa: E402
    build_sorted_owner_basin_candidates as candidate_builder,
)
from scripts.research import (  # noqa: E402
    score_sorted_owner_basin_landscape as scorer,
)
from scripts.research import (  # noqa: E402
    summarize_sorted_owner_basin_landscape as summarizer,
)


def _load_file_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


merger = _load_file_module("scratch_owner_basin_merger", MERGER_PATH)
builder_fixtures = _load_file_module(
    "candidate_builder_fixture_helpers",
    REPO_ROOT / "tests/research/test_build_sorted_owner_basin_candidates.py",
)


JsonDict = dict[str, Any]


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(candidate_builder.canonical_json_bytes(value) + b"\n")


def _write_jsonl(path: Path, rows: list[JsonDict]) -> None:
    path.write_bytes(
        b"".join(candidate_builder.canonical_json_bytes(row) + b"\n" for row in rows)
    )


def _four_context_inputs(tmp_path: Path) -> JsonDict:
    root = tmp_path / "authority"
    root.mkdir()
    rules = builder_fixtures._rules()  # noqa: SLF001
    base_ledger = builder_fixtures._ledger(rules)  # noqa: SLF001
    base_members = deepcopy(rules["candidate_materializer"]["foil_set"]["members"])
    base_context = base_ledger["context_id"]
    contexts = [f"{base_context}:shard-{index}" for index in range(4)]

    ledgers: list[JsonDict] = []
    members: list[JsonDict] = []
    for index, context_id in enumerate(contexts):
        ledger = deepcopy(base_ledger)
        ledger["context_id"] = context_id
        source_pred_row_id = f"pred-row-{index}"
        ledger["source_pred_row_id"] = source_pred_row_id
        token_ids = [1, 2, 30 + index]
        ledger["context_tokens"]["token_ids"] = token_ids
        ledger["context_tokens"]["token_ids_sha256"] = candidate_builder.sha256_json(
            token_ids
        )
        ledger["context_tokens"]["self_prefix_generated_token_ids_sha256"] = (
            candidate_builder.sha256_json(token_ids[2:])
        )
        provenance = ledger["context_provenance"]
        provenance["source_pred_row_id"] = source_pred_row_id
        provenance["registry_id"] = f"registry:ctx-{index}"
        provenance["foreign_keys"]["prediction_row"] = source_pred_row_id
        provenance["source_binding"]["source_row_id"] = f"context-row-{index}"
        ledger["source_review_foreign_key_lineage"] = {
            key: deepcopy(provenance[key])
            for key in (
                "source_pred_row_id",
                "review_status",
                "registry_id",
                "foreign_keys",
                "source_binding",
            )
        }
        ledgers.append(ledger)
        for original in base_members:
            member = deepcopy(original)
            suffix = f"-ctx-{index}"
            member["context_id"] = context_id
            member["foil_member_id"] += suffix
            member["source_id"] += suffix
            if member["identity_kind"] == "registered_geometry":
                member["identity_id"] += suffix
            member["provenance"]["source_row_id"] += suffix
            members.append(member)

    rules["candidate_materializer"]["foil_set"]["members"] = members
    rules["candidate_materializer"]["foil_set"]["members_sha256"] = (
        candidate_builder.sha256_json(
            sorted(members, key=lambda value: value["foil_member_id"])
        )
    )
    rules["scoring_contract"] = {
        "coordinate_token_id_start": rules["schema_tokens"][
            "coordinate_token_id_start"
        ],
        "coordinate_token_id_end_exclusive": rules["schema_tokens"][
            "coordinate_token_id_start"
        ]
        + 10,
        "foil_set_digests": {
            rules["candidate_materializer"]["foil_set"]["foil_set_id"]: rules[
                "candidate_materializer"
            ]["foil_set"]["members_sha256"]
        },
    }
    rules_path = root / "landscape-decision-rules.json"
    _write_json(rules_path, rules)
    ledger_path = root / "owner-context-ledger.jsonl"
    _write_jsonl(ledger_path, ledgers)

    seed_rows: list[JsonDict] = []
    for index, (context_id, ledger) in enumerate(zip(contexts, ledgers, strict=True)):
        base_seeds = builder_fixtures._seeds(  # noqa: SLF001
            rules_digest=candidate_builder.sha256_file(rules_path),
            ledger_digest=candidate_builder.sha256_file(ledger_path),
            rules=rules,
            score_value=0.0,
        )["seeds"]
        member_by_bank = {
            member["bank_name"]: member
            for member in members
            if member["context_id"] == context_id
        }
        for original in base_seeds:
            seed = deepcopy(original)
            seed["context_id"] = context_id
            bank = seed["bank_name"]
            if bank != "target":
                member = member_by_bank[bank]
                seed["source_id"] = member["source_id"]
                seed["identity_id"] = member["identity_id"]
                seed["foil_member_id"] = member["foil_member_id"]
                seed["foil_provenance"] = deepcopy(member["provenance"])
            else:
                seed["source_id"] = f"gt-template-{index}"
            seed_rows.append(seed)
    seeds = {
        "schema_version": candidate_builder.BANK_SEEDS_SCHEMA_VERSION,
        "landscape_decision_rules_sha256": candidate_builder.sha256_file(rules_path),
        "owner_context_ledger_sha256": candidate_builder.sha256_file(ledger_path),
        "foil_set_sha256": rules["candidate_materializer"]["foil_set"][
            "members_sha256"
        ],
        "seeds": seed_rows,
    }
    seeds_path = root / "bank-seeds.json"
    _write_json(seeds_path, seeds)
    candidates_path = root / "landscape-candidates.jsonl"
    candidate_receipt_path = root / "landscape-candidates-receipt.json"
    candidate_builder.build_sorted_owner_basin_candidates(
        owner_context_ledger=ledger_path,
        landscape_decision_rules=rules_path,
        bank_seeds=seeds_path,
        output_jsonl=candidates_path,
        receipt=candidate_receipt_path,
        expected_owner_context_ledger_sha256=candidate_builder.sha256_file(ledger_path),
        expected_landscape_decision_rules_sha256=candidate_builder.sha256_file(
            rules_path
        ),
        expected_bank_seeds_sha256=candidate_builder.sha256_file(seeds_path),
    )
    loaded_rules = scorer.load_decision_rules(rules_path)
    candidates = scorer.load_candidate_rows(candidates_path, rules=loaded_rules)
    ledger = scorer.load_owner_context_ledger(ledger_path, rules=loaded_rules)
    contract_validation = scorer.validate_v2_candidate_contract(
        candidates,
        owner_context_ledger=ledger,
        rules=loaded_rules,
    )
    return {
        "root": root,
        "rules_path": rules_path,
        "ledger_path": ledger_path,
        "candidates_path": candidates_path,
        "candidate_receipt_path": candidate_receipt_path,
        "rules": loaded_rules,
        "candidates": candidates,
        "ledger": ledger,
        "contract_validation": contract_validation,
        "contexts": contexts,
    }


def _common_score_row(candidate: Any, rules: Any) -> JsonDict:
    raw = candidate.raw_payload
    return {
        "schema_version": summarizer.SCORE_SCHEMA_VERSION,
        "candidate_id": candidate.candidate_id,
        "diagnostic_owner_id": candidate.diagnostic_owner_id,
        "gt_owner_id": candidate.gt_owner_id,
        "owner_status": "gt",
        "image_id": candidate.image_id,
        "image_identity": candidate.image_identity,
        "context_id": candidate.context_id,
        "landscape_surface": "restricted_gt_target",
        "role": candidate.role,
        "physical_owner_hint": candidate.physical_owner_hint,
        "review_status": "reviewed",
        "candidate_kind": candidate.candidate_kind,
        "foil_set_id": candidate.foil_set_id,
        "foil_set_digest": raw["foil_set_sha256"],
        "rule_digest": rules.rules_digest,
        "request_kind": candidate.request_kind,
        "native_repetition_penalty_stratum": candidate.native_repetition_penalty_stratum,
        "basin_id": None,
        "prefix_token_count": len(candidate.prefix_token_ids),
        "prefix_token_ids_sha256": candidate.prefix_token_ids_sha256,
        "proposal_verification": {
            "self_consistency": "passed",
            "pure_core_recomputation": "passed",
        },
        "upstream_adjudication": {"global_ambiguity_status": "clear"},
        "likelihood_channel_note": "raw is model likelihood; policy is auxiliary",
    }


def _score_row(candidate: Any, rules: Any) -> JsonDict:
    row = _common_score_row(candidate, rules)
    if candidate.record_type == "conditional_y1_score_plan":
        row.update(
            {
                "fixed_coord_token_ids": list(candidate.fixed_coord_token_ids),
                "scan_slot": "y1",
                "raw_bin_scan": {
                    "bin_logprobs": [
                        -1.0 - (entry["y1"] / 100.0)
                        for entry in candidate.complete_conditional_y1
                    ]
                },
                "auxiliary_policy_bin_scan": {},
            }
        )
        return row
    assert candidate.record_type == "complete_box_candidate"
    raw = candidate.raw_payload
    coordinate_tokens = list(candidate.coord_token_ids)
    row.update(
        {
            "coord_token_ids": coordinate_tokens,
            "coord_token_ids_sha256": scorer.sha256_json(coordinate_tokens),
            "raw_model_logprob": {
                "x1_logprob": -0.25,
                "y1_logprob": -0.25,
                "x2_logprob": -0.25,
                "y2_logprob": -0.25,
                "complete_box_logprob_sum": -1.0,
                "vocab_attestation": {
                    name: {
                        "vocab_size": rules.model_vocab_size,
                        "domain_digest": "fixture-vocabulary-domain",
                        "filtered": False,
                        "tokenizer_identity_digest": "fixture-tokenizer",
                        "model_identity_digest": "fixture-model",
                        "rule_digest": rules.rules_digest,
                        "runtime_receipt_id": "fixture-runtime",
                    }
                    for name in summarizer.COORDINATE_NAMES
                },
            },
            "auxiliary_policy_scores": {},
            "core_candidate": {
                "bank_name": raw["bank_name"],
                "source_id": raw["source_id"],
                "extent_submode": raw["extent_submode"],
                "proposal_measure_id": raw["proposal_measure_id"],
                "coordinate_bins": raw["coordinate_bin_values"],
                "role_id": raw["role_id"],
                "identity_kind": raw["identity_kind"],
                "identity_id": (
                    candidate.gt_owner_id
                    if raw.get("role_kind") == "target"
                    else raw["identity_id"]
                ),
                "geometry_identity": raw["geometry_identity"],
            },
        }
    )
    return row


def _bounded_null_receipt(request_id: str) -> JsonDict:
    receipt = {
        "schema_version": "sorted_owner_basin_free_tree_execution.v1",
        "surface": "free_coordinate_tree",
        "request_id": request_id,
        "status": "executed",
        "counts": {"complete_box_count": 0},
        "null_semantics": "bounded_free_search_null_is_non_evidence_for_absence",
    }
    receipt["receipt_sha256"] = scorer.sha256_json(receipt)
    return receipt


def _uncached_accounting(context_id: str) -> JsonDict:
    return {
        "scoring_backend": scorer.FULL_REFORWARD_SCORING_BACKEND,
        "context_id": context_id,
        "group_id": f"group:{context_id}",
        "root_prefix_length": 3,
        "root_calls": 1,
        "logical_token_step_requests": 1,
        "logical_token_step_requests_by_depth": {"1": 1},
        "actual_forward_calls": 1,
        "actual_forward_calls_by_depth": {"0": 1},
        "memo_hits_by_depth": {"1": 1},
        "memo_entries_by_depth": {"0": 1, "1": 0, "2": 0},
        "retained_relative_depths": [0, 1, 2],
    }


def _runtime_identity_admission() -> JsonDict:
    runtime_digest = "d" * 64
    projection = {"fixture": "exact", "precision": "float32"}
    return {
        "config": {
            "status": "passed",
            "model_dtype": "fp32",
            "source_jsonl": "fixture-source.jsonl",
            "binding_sha256": "c" * 64,
        },
        "preload": {
            "status": "passed",
            "identity_file_sha256": "1" * 64,
            "identity_receipt_digest": "2" * 64,
            "resolved_infer_fingerprint": "3" * 64,
            "source_panel_sha256": "4" * 64,
            "tokenizer_identity_sha256": "5" * 64,
            "model_identity_sha256": "6" * 64,
            "runtime_identity_sha256": runtime_digest,
            "admission_sha256": "7" * 64,
        },
        "postload": {
            "status": "passed",
            "projection_sha256": "8" * 64,
            "runtime_identity_sha256": runtime_digest,
            "observed_runtime_identity_sha256": runtime_digest,
            "expected_projection": projection,
            "observed_projection": projection,
        },
    }


def _make_shard(fixture: JsonDict, context_id: str, index: int) -> Any:
    root = fixture["root"] / f"shard-{index}"
    root.mkdir()
    selected, selection = scorer.select_candidate_contexts(
        fixture["candidates"], [context_id]
    )
    restricted = [
        candidate
        for candidate in selected
        if candidate.record_type != "free_coordinate_tree_root_request"
    ]
    rows = [_score_row(candidate, fixture["rules"]) for candidate in restricted]
    rows.sort(key=lambda row: row["candidate_id"])
    scores_path = root / merger.SCORE_NAME
    _write_jsonl(scores_path, rows)
    free_ids = sorted(
        candidate.candidate_id
        for candidate in selected
        if candidate.record_type == "free_coordinate_tree_root_request"
    )
    assert len(free_ids) == 1
    free_receipts = [_bounded_null_receipt(free_ids[0])]
    pure_core = scorer.require_production_pure_core(scorer._load_pure_core())  # noqa: SLF001
    attestations = scorer.build_conditional_y1_completeness_attestations(
        pure_core=pure_core,
        candidates=selected,
        scored_rows=rows,
        owner_context_ledger=fixture["ledger"],
        rules=fixture["rules"],
    )
    parity = {
        "status": "failed",
        "atol": scorer.CACHE_PARITY_ATOL,
        "rtol": scorer.CACHE_PARITY_RTOL,
    }
    admission = scorer.build_scoring_backend_admission(
        parity_gate=parity,
        selection=scorer.select_scoring_backend_from_parity(parity),
        score_row_count=len(rows),
        per_context_group_accounting=[_uncached_accounting(context_id)],
    )
    source_digests = {
        name: {
            "path": str(path),
            "sha256": scorer.sha256_file(path),
            "manifest_expected": scorer.sha256_file(path),
            "match": True,
        }
        for name, path in {
            "owner_context_ledger": fixture["ledger_path"],
            "decision_rules": fixture["rules_path"],
            "candidates": fixture["candidates_path"],
        }.items()
    }
    receipt: JsonDict = {
        "schema_version": summarizer.SCORE_RECEIPT_SCHEMA_VERSION,
        "unit_id": "fixture-owner-basin-shard",
        "command": ["fixture-scorer", "--include-context-id", context_id],
        "runtime_execution_status": scorer.LIVE_SCORING_PENDING_SEAL_STATUS,
        "source_digests": source_digests,
        "decision_rules": {
            "path": str(fixture["rules_path"]),
            "file_sha256": scorer.sha256_file(fixture["rules_path"]),
            "core_rule_digest": fixture["rules"].rules_digest,
        },
        "model_identity": {"model": "fixture-fp32"},
        "tokenizer_identity": {"tokenizer": "fixture"},
        "backend_session": {
            "backend": "hf",
            "effective_settings": {
                "device": "cuda",
                "observed_model_dtype": "float32",
                "use_cache": False,
            },
        },
        "environment": {
            "cuda_available": True,
            "device_name": "fixture A100",
            "python_version": "fixture",
            "torch_version": "fixture",
            "transformers_version": "fixture",
            "tf32": {"matmul_allow_tf32": False, "cudnn_allow_tf32": False},
        },
        "command_environment": {
            "cwd": str(REPO_ROOT),
            "python_executable": "python",
            "python_version": "fixture",
            "torch_version": "fixture",
            "transformers_version": "fixture",
        },
        "implementation_provenance": {
            "git_head": "a" * 40,
            "git_dirty": True,
            "git_dirty_diff_sha256": "b" * 64,
            "relevant_file_digests": {"fixture-scorer.py": "c" * 64},
        },
        "runtime_identity_admission": _runtime_identity_admission(),
        "numeric_reproduction_tolerance": 1e-6,
        "pure_core": {
            "present": True,
            "compatible_api": True,
            "path": "fixture",
            "note": "fixture",
        },
        "likelihood_channels": {
            "raw": "fp32 full vocabulary",
            "auxiliary_policy": ["rp_1_00", "rp_1_10"],
            "policy_is_not_a_model_likelihood": True,
            "subset_normalization_forbidden": True,
        },
        "execution_architecture": scorer.execution_architecture_for_admission(
            admission
        ),
        "candidate_count": len(rows),
        "owners_covered": 1,
        "runtime_receipt_id": f"runtime:{index}",
        "mandatory_cache_parity_gate": parity,
        "scoring_backend_admission": admission,
        "conditional_y1_attestations": attestations,
        "conditional_y1_completeness_contract": "full_core_v2_ConditionalY1CompletenessAttestation",
        "core_v2_full_vocabulary_attestation": {
            "contiguous_token_id_digest": "e" * 64,
            "expected_vocabulary_size": fixture["rules"].model_vocab_size,
            "model_identity": "6" * 64,
            "runtime_rule_digest": fixture["rules"].rules_digest,
            "tokenizer_identity": "5" * 64,
        },
        "contract_validation": fixture["contract_validation"],
        "context_selection": selection,
        "phase_a_freeze_admission": {
            "status": "not_applicable_non_sentinel",
            "sentinel_dependency_consumed": False,
            "structural_status": "test_fixture",
        },
        "surfaces": {
            "free_coordinate_tree": {
                "declared_request_ids": free_ids,
                "receipts": free_receipts,
                "null_semantics": "bounded_free_search_null_is_non_evidence_for_absence",
            },
            "restricted_candidate_bank": {
                "conditional_y1_attestation_count": len(attestations),
                "membership": "authoritative_candidate_builder_v2_pre_score_frozen",
            },
        },
    }
    receipt = scorer.seal_execution_receipt(receipt, scores_path=scores_path, rows=rows)
    receipt["output_artifacts"]["landscape_scores"]["path"] = str(scores_path)
    receipt_path = root / merger.RECEIPT_NAME
    _write_json(receipt_path, receipt)
    return merger.ShardInput(scores_path, receipt_path)


@pytest.fixture()
def landscape(tmp_path: Path) -> JsonDict:
    fixture = _four_context_inputs(tmp_path)
    fixture["shards"] = [
        _make_shard(fixture, context, index)
        for index, context in enumerate(fixture["contexts"])
    ]
    return fixture


def _merge(fixture: JsonDict, output_name: str = "merged", shards: Any = None) -> JsonDict:
    return merger.merge_shards(
        shards=fixture["shards"] if shards is None else shards,
        owner_context_ledger=fixture["ledger_path"],
        candidates_path=fixture["candidates_path"],
        candidate_receipt_path=fixture["candidate_receipt_path"],
        decision_rules_path=fixture["rules_path"],
        output_dir=fixture["root"] / output_name,
        repo_root=REPO_ROOT,
    )


def _load_shard(shard: Any) -> tuple[list[JsonDict], JsonDict]:
    rows = [json.loads(line) for line in shard.scores_path.read_text().splitlines()]
    return rows, json.loads(shard.receipt_path.read_text())


def _reseal_shard(
    shard: Any,
    rows: list[JsonDict],
    receipt: JsonDict,
    *,
    backend: str = "uncached",
) -> None:
    _write_jsonl(shard.scores_path, rows)
    context_id = receipt["context_selection"]["included_context_ids"][0]
    parity_status = "failed" if backend == "uncached" else "passed"
    parity = {
        "status": parity_status,
        "atol": scorer.CACHE_PARITY_ATOL,
        "rtol": scorer.CACHE_PARITY_RTOL,
    }
    if backend == "uncached":
        accounting = _uncached_accounting(context_id)
    else:
        accounting = {
            "scoring_backend": scorer.KV_CACHE_SCORING_BACKEND,
            "context_id": context_id,
            "group_id": f"group:{context_id}",
            "root_prefix_length": 3,
            "root_calls": 1,
            "logical_token_step_requests": 0,
            "logical_token_step_requests_by_depth": {},
            "actual_forward_calls": 1,
            "actual_forward_calls_by_depth": {"0": 1},
            "memo_hits_by_depth": {},
            "memo_entries_by_depth": {},
            "retained_relative_depths": [],
        }
    receipt["mandatory_cache_parity_gate"] = parity
    receipt["scoring_backend_admission"] = scorer.build_scoring_backend_admission(
        parity_gate=parity,
        selection=scorer.select_scoring_backend_from_parity(parity),
        score_row_count=len(rows),
        per_context_group_accounting=[accounting],
    )
    receipt["execution_architecture"] = scorer.execution_architecture_for_admission(
        receipt["scoring_backend_admission"]
    )
    receipt = scorer.seal_execution_receipt(receipt, scores_path=shard.scores_path, rows=rows)
    receipt["output_artifacts"]["landscape_scores"]["path"] = str(shard.scores_path)
    _write_json(shard.receipt_path, receipt)


def test_positive_four_context_bounded_null_is_accepted_end_to_end(
    landscape: JsonDict,
) -> None:
    result = _merge(landscape)
    output = landscape["root"] / "merged"
    receipt = json.loads((output / merger.RECEIPT_NAME).read_text())
    assert result["merge_acceptance"]["status"] == "passed"
    assert receipt["runtime_execution_status"] == merger.SEALED_LIVE_STATUS
    assert receipt["shard_merge"]["monolithic_runtime_claimed"] is False
    assert len(receipt["shard_merge"]["shards"]) == 4
    assert receipt["scoring_backend_admission"]["selected_backend"] == (
        scorer.FULL_REFORWARD_SCORING_BACKEND
    )
    assert receipt["scoring_backend_admission"]["use_cache"] is False
    summary, summary_receipt = summarizer.summarize(
        scores_path=output / merger.SCORE_NAME,
        score_receipt_path=output / merger.RECEIPT_NAME,
        rules_path=landscape["rules_path"],
    )
    assert summary_receipt["independent_reconstruction"] == "passed"
    assert len({entry["context_id"] for entry in summary["per_owner_context"]}) == 4


@pytest.mark.parametrize("case", ["missing", "duplicate", "non_singleton"])
def test_rejects_missing_duplicate_or_non_singleton_context(
    landscape: JsonDict, case: str
) -> None:
    shards = list(landscape["shards"])
    if case == "missing":
        shards.pop()
    elif case == "duplicate":
        shards[-1] = shards[0]
    else:
        rows, receipt = _load_shard(shards[0])
        receipt["context_selection"]["included_context_ids"] = landscape["contexts"][:2]
        _write_json(shards[0].receipt_path, receipt)
    with pytest.raises(merger.MergeContractError, match="exactly|one context|duplicate"):
        _merge(landscape, shards=shards)


def test_rejects_stale_selection_digest(landscape: JsonDict) -> None:
    rows, receipt = _load_shard(landscape["shards"][0])
    receipt["context_selection"]["selection_sha256"] = "0" * 64
    _write_json(landscape["shards"][0].receipt_path, receipt)
    with pytest.raises(merger.MergeContractError, match="selection"):
        _merge(landscape)


@pytest.mark.parametrize("case", ["digest", "count", "ledger_digest"])
def test_rejects_candidate_receipt_digest_or_count_drift(
    landscape: JsonDict, case: str
) -> None:
    path = landscape["candidate_receipt_path"]
    receipt = json.loads(path.read_text())
    if case == "digest":
        receipt["output_jsonl"]["sha256"] = "0" * 64
    elif case == "count":
        receipt["output_jsonl"]["row_count"] += 1
    else:
        receipt["inputs"]["owner_context_ledger_sha256"] = "0" * 64
    _write_json(path, receipt)
    with pytest.raises(merger.MergeContractError, match="candidate receipt"):
        _merge(landscape)


@pytest.mark.parametrize("case", ["missing", "extra"])
def test_rejects_missing_or_extra_restricted_candidate_id(
    landscape: JsonDict, case: str
) -> None:
    shard = landscape["shards"][0]
    rows, receipt = _load_shard(shard)
    complete_index = next(
        index for index, row in enumerate(rows) if row["request_kind"] == "complete_box"
    )
    if case == "missing":
        rows.pop(complete_index)
    else:
        extra = deepcopy(rows[complete_index])
        extra["candidate_id"] = "foreign-extra-candidate"
        extra["core_candidate"]["source_id"] = "foreign-extra-source"
        rows.append(extra)
    _reseal_shard(shard, rows, receipt)
    with pytest.raises(merger.MergeContractError, match="restricted score IDs"):
        _merge(landscape)


def _add_duplicate_free_row(shard: Any, duplicate_id: str) -> None:
    rows, receipt = _load_shard(shard)
    source = deepcopy(next(row for row in rows if row["request_kind"] == "complete_box"))
    request_id = receipt["surfaces"]["free_coordinate_tree"]["declared_request_ids"][0]
    source["candidate_id"] = duplicate_id
    source["landscape_surface"] = "canonical_description_free"
    source["review_status"] = "unreviewed"
    source["free_surface_execution"] = {"free_tree_request_id": request_id}
    rows.append(source)
    execution = receipt["surfaces"]["free_coordinate_tree"]["receipts"][0]
    execution.pop("receipt_sha256")
    execution["counts"]["complete_box_count"] = 1
    execution["receipt_sha256"] = scorer.sha256_json(execution)
    _reseal_shard(shard, rows, receipt)


def test_rejects_duplicate_merged_candidate(landscape: JsonDict) -> None:
    duplicate_id = "duplicate-free-score-candidate"
    _add_duplicate_free_row(landscape["shards"][0], duplicate_id)
    _add_duplicate_free_row(landscape["shards"][1], duplicate_id)
    with pytest.raises(merger.MergeContractError, match="duplicate merged candidate"):
        _merge(landscape)


@pytest.mark.parametrize("case", ["free_root", "y1"])
def test_rejects_free_root_or_y1_incompleteness(
    landscape: JsonDict, case: str
) -> None:
    shard = landscape["shards"][0]
    _, receipt = _load_shard(shard)
    if case == "free_root":
        receipt["surfaces"]["free_coordinate_tree"]["receipts"] = []
        receipt["landscape_surface_receipts"]["canonical_description_free"][
            "execution_receipts"
        ] = []
    else:
        receipt["conditional_y1_attestations"] = []
    _write_json(shard.receipt_path, receipt)
    with pytest.raises(merger.MergeContractError, match="summarizer rejected|incomplete"):
        _merge(landscape)


def test_rejects_mixed_parity_and_backend(landscape: JsonDict) -> None:
    shard = landscape["shards"][0]
    rows, receipt = _load_shard(shard)
    _reseal_shard(shard, rows, receipt, backend="cache")
    with pytest.raises(merger.MergeContractError, match="failed mandatory parity"):
        _merge(landscape)


def test_rejects_accounting_tamper(landscape: JsonDict) -> None:
    shard = landscape["shards"][0]
    _, receipt = _load_shard(shard)
    receipt["scoring_backend_admission"]["forward_accounting"]["aggregate"][
        "root_calls"
    ] += 1
    receipt["scoring_backend_admission"].pop("sha256")
    receipt["scoring_backend_admission"]["sha256"] = scorer.sha256_json(
        receipt["scoring_backend_admission"]
    )
    _write_json(shard.receipt_path, receipt)
    with pytest.raises(merger.MergeContractError, match="summarizer rejected"):
        _merge(landscape)


def test_rejects_conclusion_critical_identity_mismatch(landscape: JsonDict) -> None:
    shard = landscape["shards"][0]
    _, receipt = _load_shard(shard)
    receipt["model_identity"] = {"model": "different"}
    _write_json(shard.receipt_path, receipt)
    with pytest.raises(merger.MergeContractError, match="conclusion-critical"):
        _merge(landscape)


@pytest.mark.parametrize("case", ["score", "surface"])
def test_rejects_score_or_surface_tamper(landscape: JsonDict, case: str) -> None:
    shard = landscape["shards"][0]
    rows, receipt = _load_shard(shard)
    if case == "score":
        rows[0]["candidate_kind"] = "tampered"
        _write_jsonl(shard.scores_path, rows)
    else:
        receipt["landscape_surface_receipts"]["restricted_gt_target"][
            "score_rows_sha256"
        ] = "0" * 64
        _write_json(shard.receipt_path, receipt)
    with pytest.raises(merger.MergeContractError, match="summarizer rejected"):
        _merge(landscape)


def test_input_order_is_byte_deterministic(landscape: JsonDict) -> None:
    output = landscape["root"] / "deterministic"
    _merge(landscape, output_name="deterministic")
    first_scores = (output / merger.SCORE_NAME).read_bytes()
    first_receipt = (output / merger.RECEIPT_NAME).read_bytes()
    shutil.rmtree(output)
    _merge(
        landscape,
        output_name="deterministic",
        shards=list(reversed(landscape["shards"])),
    )
    assert (output / merger.SCORE_NAME).read_bytes() == first_scores
    assert (output / merger.RECEIPT_NAME).read_bytes() == first_receipt


def test_output_is_write_once(landscape: JsonDict) -> None:
    _merge(landscape, output_name="immutable")
    with pytest.raises(merger.MergeContractError, match="overwrite immutable"):
        _merge(landscape, output_name="immutable")
