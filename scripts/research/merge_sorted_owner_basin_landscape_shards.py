#!/usr/bin/env python3
"""Merge four sealed singleton-context Sorted owner-basin score shards.

This is a CPU-only artifact merger.  It does not score, reinterpret, or select
candidates.  It reconstructs the complete candidate-builder-v2 contract from
the authoritative ledger, candidate JSONL, candidate receipt, and decision
rules; independently admits each scorer shard through the landed summarizer;
then publishes one deterministic scorer-shaped score/receipt pair.

The merged receipt deliberately records a content-addressed ``shard_merge``
lineage.  It is not represented as a monolithic model runtime.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, NoReturn


SCORE_NAME = "landscape-scores.jsonl"
RECEIPT_NAME = "landscape-scores-receipt.json"
EXPECTED_SHARD_COUNT = 4
MERGE_SCHEMA_VERSION = "sorted_owner_basin_shard_merge.v1"
SEALED_LIVE_STATUS = "live_model_scoring_completed_artifacts_sealed"


class MergeContractError(RuntimeError):
    """A precondition for a conclusion-bearing shard merge was not proven."""


def _fail(message: str) -> NoReturn:
    raise MergeContractError(message)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be a JSON object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail(f"{label} must be a JSON array")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{label} must be a non-empty string")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"cannot read {path} as JSON: {exc}")
    return dict(_mapping(value, str(path)))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.resolve(strict=True).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    _fail(f"{path}:{line_number} is not valid JSON: {exc}")
                rows.append(dict(_mapping(value, f"{path}:{line_number}")))
    except OSError as exc:
        _fail(f"cannot read {path}: {exc}")
    return rows


@dataclass(frozen=True)
class RepoModules:
    root: Path
    scorer: Any
    summarizer: Any
    candidate_builder: Any


def load_repo_modules(repo_root: Path) -> RepoModules:
    root = repo_root.expanduser().resolve(strict=True)
    required = (
        root / "scripts/research/score_sorted_owner_basin_landscape.py",
        root / "scripts/research/summarize_sorted_owner_basin_landscape.py",
        root / "scripts/research/build_sorted_owner_basin_candidates.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        _fail(f"repository root lacks required Sorted owner-basin modules: {missing}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scripts.research import (  # noqa: PLC0415
        build_sorted_owner_basin_candidates as candidate_builder,
    )
    from scripts.research import (  # noqa: PLC0415
        score_sorted_owner_basin_landscape as scorer,
    )
    from scripts.research import (  # noqa: PLC0415
        summarize_sorted_owner_basin_landscape as summarizer,
    )

    return RepoModules(root, scorer, summarizer, candidate_builder)


@dataclass(frozen=True)
class ShardInput:
    scores_path: Path
    receipt_path: Path


@dataclass
class ValidatedShard:
    context_id: str
    rows: list[dict[str, Any]]
    receipt: dict[str, Any]
    scores_path: Path
    receipt_path: Path
    identity_projection: dict[str, Any]


def _candidate_receipt_counts(candidates: Sequence[Any]) -> dict[str, int]:
    complete = [row for row in candidates if row.record_type == "complete_box_candidate"]
    plans = [row for row in candidates if row.record_type == "conditional_y1_score_plan"]
    free = [
        row
        for row in candidates
        if row.record_type == "free_coordinate_tree_root_request"
    ]
    unique_boxes = {
        row.raw_payload.get("unique_box_id")
        for row in complete
        if row.raw_payload.get("unique_box_id") is not None
    }
    if len(unique_boxes) != len(complete):
        _fail("candidate JSONL does not preserve one unique_box_id per complete-box candidate")
    return {
        "owner_context_rows": len({(row.diagnostic_owner_id, row.context_id) for row in candidates}),
        "conditional_y1_score_plans": len(plans),
        "complete_box_candidates": len(complete),
        "free_coordinate_tree_root_requests": len(free),
        "unique_box_ids": len(unique_boxes),
    }


def validate_candidate_receipt(
    *,
    receipt_path: Path,
    candidates_path: Path,
    ledger_path: Path,
    rules_path: Path,
    candidates: Sequence[Any],
    ledger: Mapping[tuple[str, str], Any],
    rules: Any,
    candidate_builder: Any,
) -> dict[str, Any]:
    receipt = _read_json(receipt_path)
    if receipt.get("schema_version") != candidate_builder.SCHEMA_VERSION:
        _fail("candidate-builder receipt has an unexpected schema_version")
    if receipt.get("materializer") != "cpu_only_pre_score_candidate_membership":
        _fail("candidate receipt does not name the CPU-only pre-score materializer")
    inputs = _mapping(receipt.get("inputs"), "candidate receipt.inputs")
    expected_inputs = {
        "owner_context_ledger_sha256": sha256_file(ledger_path),
        "landscape_decision_rules_sha256": sha256_file(rules_path),
        "core_rule_digest": rules.rules_digest,
    }
    for name, expected in expected_inputs.items():
        if inputs.get(name) != expected:
            _fail(f"candidate receipt {name} is stale")
    output = _mapping(receipt.get("output_jsonl"), "candidate receipt.output_jsonl")
    if output.get("sha256") != sha256_file(candidates_path):
        _fail("candidate receipt output_jsonl digest is stale")
    if output.get("row_count") != len(candidates):
        _fail("candidate receipt output_jsonl count is stale")
    expected_counts = _candidate_receipt_counts(candidates)
    if dict(_mapping(receipt.get("counts"), "candidate receipt.counts")) != expected_counts:
        _fail("candidate receipt counts do not reconstruct from the full candidate JSONL")
    if expected_counts["owner_context_rows"] != len(ledger):
        _fail("candidate/receipt owner-context count differs from the authoritative ledger")
    membership = _mapping(
        receipt.get("candidate_membership"), "candidate receipt.candidate_membership"
    )
    if (
        membership.get("selection") != "frozen_pre_score_only"
        or membership.get("score_dependent_selection_executed") is not False
    ):
        _fail("candidate receipt does not freeze membership before scoring")
    free = _mapping(receipt.get("free_search"), "candidate receipt.free_search")
    if (
        free.get("executed") is not False
        or free.get("explicit_root_request_count")
        != expected_counts["free_coordinate_tree_root_requests"]
    ):
        _fail("candidate receipt free-root declaration does not reconstruct")
    return receipt


def _identity_projection(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Project conclusion-critical identity; omit only shard-bound provenance."""

    environment = dict(_mapping(receipt.get("environment"), "receipt.environment"))
    # A physical ordinal can legitimately differ when identical runtimes are
    # assigned to separate GPUs.  GPU model/type remains conclusion-critical.
    for key in ("cuda_device_index", "physical_cuda_device", "cuda_visible_devices"):
        environment.pop(key, None)
    source_digests = {
        str(name): _mapping(entry, f"source_digests.{name}").get("sha256")
        for name, entry in sorted(
            _mapping(receipt.get("source_digests"), "receipt.source_digests").items()
        )
    }
    return {
        "schema_version": receipt.get("schema_version"),
        "unit_id": receipt.get("unit_id"),
        "source_digests": source_digests,
        "decision_rules": receipt.get("decision_rules"),
        "model_identity": receipt.get("model_identity"),
        "tokenizer_identity": receipt.get("tokenizer_identity"),
        "backend_session": receipt.get("backend_session"),
        "environment": environment,
        "command_environment": receipt.get("command_environment"),
        "implementation_provenance": receipt.get("implementation_provenance"),
        "runtime_identity_admission": receipt.get("runtime_identity_admission"),
        "numeric_reproduction_tolerance": receipt.get("numeric_reproduction_tolerance"),
        "pure_core": receipt.get("pure_core"),
        "likelihood_channels": receipt.get("likelihood_channels"),
        "core_v2_full_vocabulary_attestation": receipt.get(
            "core_v2_full_vocabulary_attestation"
        ),
        "conditional_y1_completeness_contract": receipt.get(
            "conditional_y1_completeness_contract"
        ),
        "phase_a_freeze_admission": receipt.get("phase_a_freeze_admission"),
        "contract_validation": receipt.get("contract_validation"),
        "execution_architecture": receipt.get("execution_architecture"),
    }


def _require_expected_uncached_backend(receipt: Mapping[str, Any]) -> None:
    parity = _mapping(
        receipt.get("mandatory_cache_parity_gate"), "mandatory_cache_parity_gate"
    )
    admission = _mapping(
        receipt.get("scoring_backend_admission"), "scoring_backend_admission"
    )
    expected = {
        "parity_status": "failed",
        "selected_backend": "full_reforward_uncached",
        "atol": 1e-6,
        "rtol": 1e-5,
        "cache_enabled": False,
        "use_cache": False,
        "fallback_trigger": "mandatory_cache_parity_failed",
        "backend_mixing_detected": False,
        "cache_score_row_count": 0,
    }
    if parity.get("status") != "failed" or parity.get("atol") != 1e-6 or parity.get("rtol") != 1e-5:
        _fail("every shard must preserve the failed mandatory parity gate at 1e-6/1e-5")
    for field, expected_value in expected.items():
        if admission.get(field) != expected_value:
            _fail(f"shard scoring backend admission has incompatible {field}")
    if admission.get("uncached_score_row_count") != admission.get("score_row_count"):
        _fail("shard backend admission does not bind every score row to uncached reforward")


def _validate_shard(
    *,
    shard: ShardInput,
    modules: RepoModules,
    rules_path: Path,
    candidates: Sequence[Any],
    available_context_ids: Sequence[str],
) -> ValidatedShard:
    scores_path = shard.scores_path.expanduser().resolve(strict=True)
    receipt_path = shard.receipt_path.expanduser().resolve(strict=True)
    rows = _read_jsonl(scores_path)
    receipt = _read_json(receipt_path)
    if receipt.get("runtime_execution_status") != SEALED_LIVE_STATUS:
        _fail(f"shard {receipt_path} is not a sealed live scorer output")
    try:
        modules.summarizer.summarize(
            scores_path=scores_path,
            score_receipt_path=receipt_path,
            rules_path=rules_path,
        )
    except Exception as exc:
        raise MergeContractError(
            f"existing summarizer rejected shard {receipt_path}: {exc}"
        ) from exc

    output = _mapping(receipt.get("output_artifacts"), "shard output_artifacts")
    score_entry = _mapping(output.get("landscape_scores"), "shard landscape_scores")
    declared_path = Path(_string(score_entry.get("path"), "shard score path")).expanduser()
    if declared_path.resolve(strict=True) != scores_path:
        _fail("sealed shard receipt output path does not name its paired score artifact")

    selection = _mapping(receipt.get("context_selection"), "shard context_selection")
    included = list(_sequence(selection.get("included_context_ids"), "included contexts"))
    if len(included) != 1 or not isinstance(included[0], str):
        _fail("each shard must select exactly one context")
    context_id = included[0]
    _, reconstructed = modules.scorer.select_candidate_contexts(candidates, [context_id])
    if dict(selection) != reconstructed:
        _fail(f"shard {context_id} has a stale or non-canonical context selection digest")
    if list(selection.get("available_context_ids", ())) != list(available_context_ids):
        _fail(f"shard {context_id} did not validate the common full context set")
    if not rows or {row.get("context_id") for row in rows} != {context_id}:
        _fail(f"shard {context_id} score rows are empty or mix contexts")

    selected, _ = modules.scorer.select_candidate_contexts(candidates, [context_id])
    expected_restricted_ids = {
        candidate.candidate_id
        for candidate in selected
        if candidate.record_type != "free_coordinate_tree_root_request"
    }
    observed_restricted_ids = {
        _string(row.get("candidate_id"), "restricted score candidate ID")
        for row in rows
        if row.get("landscape_surface") == "restricted_gt_target"
    }
    if observed_restricted_ids != expected_restricted_ids:
        missing = sorted(expected_restricted_ids - observed_restricted_ids)
        extra = sorted(observed_restricted_ids - expected_restricted_ids)
        _fail(
            f"shard {context_id} restricted score IDs differ from the frozen selection; "
            f"missing={missing[:8]}, extra={extra[:8]}"
        )

    selected_free_ids = sorted(
        candidate.candidate_id
        for candidate in selected
        if candidate.record_type == "free_coordinate_tree_root_request"
    )
    surface_receipts = _mapping(
        receipt.get("landscape_surface_receipts"), "shard landscape_surface_receipts"
    )
    free_authority = _mapping(
        surface_receipts.get("canonical_description_free"),
        "shard canonical_description_free receipt",
    )
    if free_authority.get("declared_request_ids") != selected_free_ids:
        _fail(f"shard {context_id} free-root declaration differs from selected candidates")
    surfaces = _mapping(receipt.get("surfaces"), "shard surfaces")
    free_projection = _mapping(
        surfaces.get("free_coordinate_tree"), "shard free_coordinate_tree surface"
    )
    if (
        free_projection.get("declared_request_ids") != selected_free_ids
        or free_projection.get("receipts") != free_authority.get("execution_receipts")
    ):
        _fail(f"shard {context_id} free surface projections disagree")

    expected_attestation_keys = {
        (candidate.diagnostic_owner_id, candidate.context_id)
        for candidate in selected
        if candidate.record_type == "conditional_y1_score_plan"
    }
    attestations = [
        dict(_mapping(value, "conditional-y1 attestation"))
        for value in _sequence(
            receipt.get("conditional_y1_attestations"),
            "conditional_y1_attestations",
        )
    ]
    observed_attestation_keys = {
        (item.get("diagnostic_owner_id"), item.get("context_id")) for item in attestations
    }
    if observed_attestation_keys != expected_attestation_keys:
        _fail(f"shard {context_id} conditional-y1 attestations are incomplete")

    _require_expected_uncached_backend(receipt)
    accounting = _mapping(
        _mapping(receipt.get("scoring_backend_admission"), "backend admission").get(
            "forward_accounting"
        ),
        "forward_accounting",
    )
    for entry in _sequence(accounting.get("per_context_group"), "per-context accounting"):
        if _mapping(entry, "accounting entry").get("context_id") != context_id:
            _fail(f"shard {context_id} accounting contains a foreign context")

    return ValidatedShard(
        context_id=context_id,
        rows=rows,
        receipt=receipt,
        scores_path=scores_path,
        receipt_path=receipt_path,
        identity_projection=_identity_projection(receipt),
    )


def _source_entry(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "manifest_expected": sha256_file(path),
        "match": True,
    }


def _lineage_record(shard: ValidatedShard) -> dict[str, Any]:
    return {
        "context_id": shard.context_id,
        "scores": {
            "path": str(shard.scores_path),
            "sha256": sha256_file(shard.scores_path),
            "row_count": len(shard.rows),
        },
        "receipt": {
            "path": str(shard.receipt_path),
            "sha256": sha256_file(shard.receipt_path),
        },
        "context_selection": shard.receipt["context_selection"],
        "runtime_receipt_id": shard.receipt.get("runtime_receipt_id"),
        "command": shard.receipt.get("command"),
        "output_artifacts": shard.receipt.get("output_artifacts"),
        "mandatory_cache_parity_gate": shard.receipt["mandatory_cache_parity_gate"],
        "scoring_backend_admission_sha256": _mapping(
            shard.receipt.get("scoring_backend_admission"),
            "scoring_backend_admission",
        ).get("sha256"),
        "physical_cuda_provenance": {
            "environment": shard.receipt.get("environment"),
            "command_environment": shard.receipt.get("command_environment"),
        },
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(dict(row)) + "\n")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )


def _run_end_to_end_acceptance(
    *, modules: RepoModules, stage: Path, rules_path: Path
) -> dict[str, Any]:
    acceptance_dir = stage / ".summarizer-acceptance"
    command = [
        sys.executable,
        str(modules.root / "scripts/research/summarize_sorted_owner_basin_landscape.py"),
        "--scores",
        str(stage / SCORE_NAME),
        "--score-receipt",
        str(stage / RECEIPT_NAME),
        "--decision-rules",
        str(rules_path),
        "--output-dir",
        str(acceptance_dir),
    ]
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        command,
        cwd=modules.root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        _fail(
            "existing summarizer rejected the staged merged artifacts: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    acceptance_receipt = _read_json(acceptance_dir / "landscape-summary-receipt.json")
    if acceptance_receipt.get("independent_reconstruction") != "passed":
        _fail("staged summarizer acceptance did not independently reconstruct")
    shutil.rmtree(acceptance_dir)
    return {
        "status": "passed",
        "consumer": "scripts/research/summarize_sorted_owner_basin_landscape.py",
        "input_row_count": acceptance_receipt.get("input_row_count"),
        "owner_context_count": acceptance_receipt.get("owner_context_count"),
        "mandatory_cache_parity_gate": acceptance_receipt.get(
            "mandatory_cache_parity_gate"
        ),
    }


def merge_shards(
    *,
    shards: Sequence[ShardInput],
    owner_context_ledger: Path,
    candidates_path: Path,
    candidate_receipt_path: Path,
    decision_rules_path: Path,
    output_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    if len(shards) != EXPECTED_SHARD_COUNT:
        _fail(f"exactly {EXPECTED_SHARD_COUNT} shard score/receipt pairs are required")
    modules = load_repo_modules(repo_root)
    ledger_path = owner_context_ledger.expanduser().resolve(strict=True)
    candidates_path = candidates_path.expanduser().resolve(strict=True)
    candidate_receipt_path = candidate_receipt_path.expanduser().resolve(strict=True)
    rules_path = decision_rules_path.expanduser().resolve(strict=True)
    final_dir = output_dir.expanduser().resolve()
    if final_dir.exists():
        _fail(f"refusing to overwrite immutable output directory: {final_dir}")

    rules = modules.scorer.load_decision_rules(rules_path)
    candidates = modules.scorer.load_candidate_rows(candidates_path, rules=rules)
    ledger = modules.scorer.load_owner_context_ledger(ledger_path, rules=rules)
    try:
        contract_validation = modules.scorer.validate_v2_candidate_contract(
            candidates,
            owner_context_ledger=ledger,
            rules=rules,
        )
    except Exception as exc:
        raise MergeContractError(
            f"full v2 candidate contract reconstruction failed: {exc}"
        ) from exc
    candidate_receipt = validate_candidate_receipt(
        receipt_path=candidate_receipt_path,
        candidates_path=candidates_path,
        ledger_path=ledger_path,
        rules_path=rules_path,
        candidates=candidates,
        ledger=ledger,
        rules=rules,
        candidate_builder=modules.candidate_builder,
    )

    available_context_ids = sorted({candidate.context_id for candidate in candidates})
    if len(available_context_ids) != EXPECTED_SHARD_COUNT:
        _fail(
            "the authoritative candidate bank must expose exactly the frozen four contexts"
        )
    validated = [
        _validate_shard(
            shard=shard,
            modules=modules,
            rules_path=rules_path,
            candidates=candidates,
            available_context_ids=available_context_ids,
        )
        for shard in shards
    ]
    context_ids = [shard.context_id for shard in validated]
    if len(set(context_ids)) != EXPECTED_SHARD_COUNT:
        _fail("shards contain a missing or duplicate singleton context")
    if sorted(context_ids) != available_context_ids:
        _fail("shard context union differs from the frozen four emitted contexts")
    validated.sort(key=lambda shard: shard.context_id)

    identity_digests = {sha256_json(shard.identity_projection) for shard in validated}
    if len(identity_digests) != 1:
        _fail("shards differ in conclusion-critical source/runtime/implementation identity")

    rows = sorted(
        (dict(row) for shard in validated for row in shard.rows),
        key=lambda row: _string(row.get("candidate_id"), "score row candidate_id"),
    )
    candidate_ids = [str(row["candidate_id"]) for row in rows]
    if len(candidate_ids) != len(set(candidate_ids)):
        _fail("duplicate merged candidate_id across shards")

    _, merged_selection = modules.scorer.select_candidate_contexts(
        candidates, available_context_ids
    )
    attestations = sorted(
        (
            dict(_mapping(value, "conditional-y1 attestation"))
            for shard in validated
            for value in _sequence(
                shard.receipt.get("conditional_y1_attestations"),
                "conditional_y1_attestations",
            )
        ),
        key=lambda item: (
            str(item.get("context_id")),
            str(item.get("diagnostic_owner_id")),
            str(item.get("image_identity")),
            str(item.get("landscape_surface")),
        ),
    )
    attestation_keys = [
        (item.get("diagnostic_owner_id"), item.get("context_id")) for item in attestations
    ]
    if len(attestation_keys) != len(set(attestation_keys)):
        _fail("duplicate conditional-y1 attestation across shards")

    free_request_ids: list[str] = []
    free_execution_receipts: list[dict[str, Any]] = []
    accounting_entries: list[dict[str, Any]] = []
    for shard in validated:
        surface_receipts = _mapping(
            shard.receipt["landscape_surface_receipts"], "surface receipts"
        )
        free = _mapping(
            surface_receipts["canonical_description_free"], "free surface receipt"
        )
        free_request_ids.extend(str(value) for value in free["declared_request_ids"])
        free_execution_receipts.extend(
            dict(_mapping(value, "free execution receipt"))
            for value in free["execution_receipts"]
        )
        admission = _mapping(shard.receipt["scoring_backend_admission"], "admission")
        forward = _mapping(admission["forward_accounting"], "forward accounting")
        accounting_entries.extend(
            dict(_mapping(value, "accounting entry"))
            for value in forward["per_context_group"]
        )
    free_request_ids.sort()
    free_execution_receipts.sort(key=lambda value: str(value.get("request_id")))
    accounting_entries.sort(
        key=lambda value: (str(value.get("context_id")), str(value.get("group_id")))
    )

    parity_gate = {
        "status": "failed",
        "atol": modules.scorer.CACHE_PARITY_ATOL,
        "rtol": modules.scorer.CACHE_PARITY_RTOL,
        "aggregation": "all_child_gates_failed_same_frozen_tolerance",
    }
    backend_selection = modules.scorer.select_scoring_backend_from_parity(parity_gate)
    backend_admission = modules.scorer.build_scoring_backend_admission(
        parity_gate=parity_gate,
        selection=backend_selection,
        score_row_count=len(rows),
        per_context_group_accounting=accounting_entries,
    )

    lineage_payload = {
        "schema_version": MERGE_SCHEMA_VERSION,
        "execution_semantics": "four_independently_sealed_singleton_context_child_runtimes",
        "monolithic_runtime_claimed": False,
        "canonical_order": "ascending_context_id_then_candidate_id",
        "authoritative_inputs": {
            "owner_context_ledger_sha256": sha256_file(ledger_path),
            "candidates_sha256": sha256_file(candidates_path),
            "candidate_receipt_sha256": sha256_file(candidate_receipt_path),
            "decision_rules_sha256": sha256_file(rules_path),
        },
        "context_selection": merged_selection,
        "identity_projection_sha256": next(iter(identity_digests)),
        "shards": [_lineage_record(shard) for shard in validated],
    }
    shard_merge = {**lineage_payload, "sha256": sha256_json(lineage_payload)}
    merge_runtime_receipt_id = "shard-merge:sha256:" + shard_merge["sha256"]

    common = validated[0].receipt
    receipt: dict[str, Any] = {
        "schema_version": common["schema_version"],
        "unit_id": common.get("unit_id"),
        "command": [
            "merge_sorted_owner_basin_landscape_shards.py",
            "--four-sealed-singleton-context-shards",
        ],
        "runtime_execution_status": modules.scorer.LIVE_SCORING_PENDING_SEAL_STATUS,
        "source_digests": {
            "owner_context_ledger": _source_entry(ledger_path),
            "decision_rules": _source_entry(rules_path),
            "candidates": _source_entry(candidates_path),
            "candidate_receipt": _source_entry(candidate_receipt_path),
        },
        "decision_rules": {
            "path": str(rules_path),
            "file_sha256": sha256_file(rules_path),
            "core_rule_digest": rules.rules_digest,
        },
        "model_identity": common.get("model_identity"),
        "tokenizer_identity": common.get("tokenizer_identity"),
        "backend_session": common.get("backend_session"),
        "environment": common.get("environment"),
        "command_environment": common.get("command_environment"),
        "implementation_provenance": common.get("implementation_provenance"),
        "runtime_identity_admission": common.get("runtime_identity_admission"),
        "numeric_reproduction_tolerance": common.get(
            "numeric_reproduction_tolerance"
        ),
        "pure_core": common.get("pure_core"),
        "likelihood_channels": common.get("likelihood_channels"),
        "execution_architecture": modules.scorer.execution_architecture_for_admission(
            backend_admission
        ),
        "candidate_count": len(rows),
        "owners_covered": len({str(row.get("diagnostic_owner_id")) for row in rows}),
        "runtime_receipt_id": merge_runtime_receipt_id,
        "mandatory_cache_parity_gate": parity_gate,
        "scoring_backend_admission": backend_admission,
        "conditional_y1_attestations": attestations,
        "conditional_y1_completeness_contract": common.get(
            "conditional_y1_completeness_contract"
        ),
        "core_v2_full_vocabulary_attestation": common.get(
            "core_v2_full_vocabulary_attestation"
        ),
        "contract_validation": contract_validation,
        "context_selection": merged_selection,
        "phase_a_freeze_admission": common.get("phase_a_freeze_admission"),
        "surfaces": {
            "free_coordinate_tree": {
                "declared_request_ids": free_request_ids,
                "receipts": free_execution_receipts,
                "null_semantics": "bounded_free_search_null_is_non_evidence_for_absence",
            },
            "restricted_candidate_bank": {
                "conditional_y1_attestation_count": len(attestations),
                "membership": "authoritative_candidate_builder_v2_pre_score_frozen",
            },
        },
        "candidate_builder_receipt": {
            "path": str(candidate_receipt_path),
            "sha256": sha256_file(candidate_receipt_path),
            "schema_version": candidate_receipt.get("schema_version"),
        },
        "shard_merge": shard_merge,
    }

    final_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{final_dir.name}.merge-", dir=final_dir.parent)
    )
    try:
        staged_scores = stage / SCORE_NAME
        staged_receipt = stage / RECEIPT_NAME
        _write_jsonl(staged_scores, rows)
        receipt = modules.scorer.seal_execution_receipt(
            receipt,
            scores_path=staged_scores,
            rows=rows,
        )
        receipt["output_artifacts"]["landscape_scores"]["path"] = str(
            final_dir / SCORE_NAME
        )
        receipt["shard_merge"]["merged_output"] = {
            "path": str(final_dir / SCORE_NAME),
            "sha256": sha256_file(staged_scores),
            "row_count": len(rows),
        }
        resealed_lineage = dict(receipt["shard_merge"])
        resealed_lineage.pop("sha256")
        receipt["shard_merge"]["sha256"] = sha256_json(resealed_lineage)
        receipt["runtime_receipt_id"] = (
            "shard-merge:sha256:" + receipt["shard_merge"]["sha256"]
        )
        _write_json(staged_receipt, receipt)
        acceptance = _run_end_to_end_acceptance(
            modules=modules,
            stage=stage,
            rules_path=rules_path,
        )
        # The acceptance result is intentionally returned to the caller rather
        # than embedded after validation, which would invalidate the validated
        # receipt bytes.
        if final_dir.exists():
            _fail(f"output directory appeared during write-once staging: {final_dir}")
        stage.replace(final_dir)
        receipt["merge_acceptance"] = acceptance
        return receipt
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard",
        action="append",
        nargs=2,
        metavar=("SCORES", "RECEIPT"),
        default=[],
        help="repeat exactly four times with a score JSONL and its sealed receipt",
    )
    parser.add_argument(
        "--shard-dir",
        action="append",
        type=Path,
        default=[],
        help="alternative repeated shard directory containing canonical filenames",
    )
    parser.add_argument("--owner-context-ledger", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--candidate-receipt", type=Path, required=True)
    parser.add_argument("--decision-rules", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(
            os.environ.get(
                "COORDEXP_REPO_ROOT", "/data/CoordExp/.worktrees/research-probes"
            )
        ),
        help="read-only CoordExp repository root providing scorer/consumer modules",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    shard_inputs = [
        ShardInput(Path(scores), Path(receipt)) for scores, receipt in args.shard
    ]
    shard_inputs.extend(
        ShardInput(directory / SCORE_NAME, directory / RECEIPT_NAME)
        for directory in args.shard_dir
    )
    receipt = merge_shards(
        shards=shard_inputs,
        owner_context_ledger=args.owner_context_ledger,
        candidates_path=args.candidates,
        candidate_receipt_path=args.candidate_receipt,
        decision_rules_path=args.decision_rules,
        output_dir=args.output_dir,
        repo_root=args.repo_root,
    )
    print(
        json.dumps(
            {
                "rows": receipt["candidate_count"],
                "owners": receipt["owners_covered"],
                "contexts": len(receipt["context_selection"]["included_context_ids"]),
                "acceptance": receipt["merge_acceptance"]["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
