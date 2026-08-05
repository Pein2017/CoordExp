#!/usr/bin/env python3
"""CPU-only owner phenotype census for the sorted all-person route landscape.

The primary output is one row per confirmed owner.  It keeps continuous
geometry and score surfaces primary and deliberately does not turn post-hoc
phenotype groupings into admission gates or mechanism claims.

Inputs are the immutable planner directory plus either a strict list of sealed
score-shard directories or, once available, an attested merged score artifact.
Every planned request must appear exactly once.  Partial context coverage is
rejected so a directory that is still being populated cannot look analytical.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
from typing import Any, NoReturn


UNIT_ID = "2026-08-03-sorted-all-person-owner-relative-route-landscape"
PLAN_SCHEMA_VERSION = "sorted-all-person-route-landscape-plan.v1"
SCORE_SCHEMA_VERSION = "sorted-all-person-route-landscape-score.v1"
LEGACY_SCORE_RECEIPT_SCHEMA_VERSION = "sorted-all-person-route-landscape-score-receipt.v1"
SCORE_RECEIPT_SCHEMA_VERSION = "sorted-all-person-route-landscape-score-receipt.v2"
MERGE_SCHEMA_VERSION = "sorted-all-person-route-landscape-merge.v1"
ATTESTATION_SCHEMA_VERSION = "sorted-all-person-route-landscape-attestation.v1"
ANALYSIS_SCHEMA_VERSION = "sorted-all-person-route-landscape-analysis.v1"
OWNER_ROW_SCHEMA_VERSION = "sorted-all-person-route-landscape-owner-phenotype.v1"
RECEIPT_SCHEMA_VERSION = "sorted-all-person-route-landscape-analysis-receipt.v1"

SCORE_NAME = "route-landscape-scores.jsonl"
SCORE_RECEIPT_NAME = "route-landscape-scores-receipt.json"
ANALYSIS_NAME = "route-landscape-analysis.json"
OWNER_CENSUS_NAME = "owner-phenotype-census.jsonl"
SCALAR_CONFIRMATION_MANIFEST_NAME = "scalar-confirmation-manifest.jsonl"
RECEIPT_NAME = "route-landscape-analysis-receipt.json"
DECISION_CHANNEL = "raw_model_logprob.complete_box_logprob_sum"
FULL_REFORWARD_BACKEND = "full_reforward_uncached"
INPUT_MODES = frozenset({"strict_confirmation", "raw_capture_salvage"})

PLAN_FILE_NAMES = (
    "owner-ledger.jsonl",
    "primary-candidates.jsonl",
    "contexts.jsonl",
    "sidecars.jsonl",
    "sampling-seeds.jsonl",
    "scoring-requests.jsonl",
)
CONTEXT_IDS = (
    "root",
    "self-due-gt2",
    "self-due-gt17",
    "self-due-gt22",
    "self-due-gt32",
    "skip-post-gt17",
)
CALIBRATION_CONTEXT_BY_OWNER = {
    "gt:7511:2": "self-due-gt2",
    "gt:7511:17": "self-due-gt17",
    "gt:7511:22": "self-due-gt22",
    "gt:7511:32": "self-due-gt32",
}
PRIMARY_TARGETS = frozenset({"gt:7511:17", "gt:7511:22"})
_COORD_RE = re.compile(r"<\|coord_(\d+)\|>")


class AnalysisContractError(RuntimeError):
    """Raised when source, coverage, or numeric analysis admission fails."""


def _fail(message: str, **context: Any) -> NoReturn:
    if context:
        message = f"{message} | context: {json.dumps(context, sort_keys=True, default=str)}"
    raise AnalysisContractError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


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


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} must be a finite number", observed=value)
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{label} must be a finite number", observed=value)
    return result


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AnalysisContractError(f"{label} is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise AnalysisContractError(f"{label} is not valid JSON: {path}") from exc
    return dict(_mapping(value, label))


def _read_jsonl(path: Path, label: str, *, allow_empty: bool = False) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise AnalysisContractError(f"{label} is missing: {path}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line:
            _fail(f"{label} line {line_number} is blank")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AnalysisContractError(f"{label} line {line_number} is not JSON") from exc
        rows.append(dict(_mapping(value, f"{label} line {line_number}")))
    if not rows and not allow_empty:
        _fail(f"{label} must contain at least one row")
    return rows


def _validate_self_digest(document: Mapping[str, Any], label: str) -> str:
    observed = document.get("receipt_content_sha256")
    if not isinstance(observed, str):
        _fail(f"{label} lacks receipt_content_sha256")
    reconstructed = sha256_json(
        {key: value for key, value in document.items() if key != "receipt_content_sha256"}
    )
    if observed != reconstructed:
        _fail(f"{label} self digest does not reconstruct", observed=observed, expected=reconstructed)
    return observed


def _write_create_or_identical(path: Path, content: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            _fail(f"{path} already exists with different content; refusing to overwrite")
        return "identical_existing_output"
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temp.write_bytes(content)
    os.replace(temp, path)
    return "created"


def _unique_index(rows: Sequence[Mapping[str, Any]], key: str, label: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value or value in result:
            _fail(f"{label} has a missing or duplicate {key}", observed=value)
        result[value] = row
    return result


@dataclass(frozen=True)
class PlanBundle:
    plan_dir: Path
    receipt_path: Path
    receipt: Mapping[str, Any]
    owners: tuple[Mapping[str, Any], ...]
    owners_by_id: Mapping[str, Mapping[str, Any]]
    candidates: tuple[Mapping[str, Any], ...]
    candidates_by_id: Mapping[str, Mapping[str, Any]]
    contexts: tuple[Mapping[str, Any], ...]
    contexts_by_id: Mapping[str, Mapping[str, Any]]
    sidecars: tuple[Mapping[str, Any], ...]
    sidecars_by_id: Mapping[str, Mapping[str, Any]]
    requests: tuple[Mapping[str, Any], ...]
    requests_by_id: Mapping[str, Mapping[str, Any]]
    file_paths: Mapping[str, Path]


def load_plan(plan_dir: str | Path) -> PlanBundle:
    resolved = Path(plan_dir).expanduser().resolve(strict=True)
    receipt_path = resolved / "receipt.json"
    receipt = _read_json(receipt_path, "plan receipt")
    if receipt.get("schema_version") != PLAN_SCHEMA_VERSION or receipt.get("unit_id") != UNIT_ID:
        _fail("plan receipt schema/unit mismatch")
    _validate_self_digest(receipt, "plan receipt")

    source_paths = _mapping(receipt.get("source_paths"), "plan receipt.source_paths")
    source_digests = _mapping(receipt.get("source_digests"), "plan receipt.source_digests")
    if set(source_paths) != set(source_digests):
        _fail("plan source path/digest keys differ")
    for key, raw_path in source_paths.items():
        path = Path(str(raw_path)).expanduser().resolve(strict=True)
        if not path.is_file():
            _fail("plan source is not a file", source=key, path=str(path))
        observed = sha256_file(path)
        expected = source_digests.get(key)
        if observed != expected:
            _fail("plan source hash differs from the frozen receipt", source=key, observed=observed, expected=expected)

    output_digests = _mapping(receipt.get("output_file_digests"), "output_file_digests")
    file_paths: dict[str, Path] = {}
    for name in PLAN_FILE_NAMES:
        path = resolved / name
        if not path.is_file():
            _fail("plan is incomplete", missing=name)
        observed = sha256_file(path)
        if output_digests.get(name) != observed:
            _fail("plan file hash differs from the sealed receipt", file=name, observed=observed)
        file_paths[name] = path

    owners = tuple(_read_jsonl(file_paths["owner-ledger.jsonl"], "owner ledger"))
    candidates = tuple(_read_jsonl(file_paths["primary-candidates.jsonl"], "primary candidates"))
    contexts = tuple(_read_jsonl(file_paths["contexts.jsonl"], "contexts"))
    sidecars = tuple(_read_jsonl(file_paths["sidecars.jsonl"], "sidecars"))
    requests = tuple(_read_jsonl(file_paths["scoring-requests.jsonl"], "scoring requests"))
    owners_by_id = _unique_index(owners, "gt_owner_id", "owner ledger")
    candidates_by_id = _unique_index(candidates, "candidate_id", "primary candidates")
    contexts_by_id = _unique_index(contexts, "context_id", "contexts")
    sidecars_by_id = _unique_index(sidecars, "sidecar_id", "sidecars")
    requests_by_id = _unique_index(requests, "request_id", "scoring requests")

    expected_owner_ids = {f"gt:7511:{index}" for index in range(2, 43)}
    if set(owners_by_id) != expected_owner_ids or len(owners) != 41:
        _fail("owner ledger is not the frozen 41-owner census")
    if len(candidates) != 369:
        _fail("primary candidate universe must contain exactly 369 rows", observed=len(candidates))
    if set(contexts_by_id) != set(CONTEXT_IDS):
        _fail("admitted context registry differs from the frozen six contexts", observed=sorted(contexts_by_id))
    counts = _mapping(receipt.get("counts"), "plan receipt.counts")
    expected_counts = {
        "owners": len(owners),
        "primary_candidates": len(candidates),
        "contexts": len(contexts),
        "sidecars": len(sidecars),
        "scoring_requests": len(requests),
    }
    for key, observed in expected_counts.items():
        if counts.get(key) != observed:
            _fail("plan row count differs from receipt", field=key, observed=observed, expected=counts.get(key))
    return PlanBundle(
        plan_dir=resolved,
        receipt_path=receipt_path,
        receipt=receipt,
        owners=owners,
        owners_by_id=owners_by_id,
        candidates=candidates,
        candidates_by_id=candidates_by_id,
        contexts=contexts,
        contexts_by_id=contexts_by_id,
        sidecars=sidecars,
        sidecars_by_id=sidecars_by_id,
        requests=requests,
        requests_by_id=requests_by_id,
        file_paths=file_paths,
    )


@dataclass(frozen=True)
class ScoreBundle:
    rows_by_request_id: Mapping[str, Mapping[str, Any]]
    receipt_paths: tuple[Path, ...]
    score_paths: tuple[Path, ...]
    receipts: tuple[Mapping[str, Any], ...]
    source_identity_sha256: str
    input_mode: str
    scorer_code_lineage: tuple[Mapping[str, Any], ...]
    batch_score_error_bound_by_context: Mapping[str, float]
    legacy_rows_by_request_id: Mapping[str, Mapping[str, Any]] | None = None
    scalar_overlay_request_ids: frozenset[str] = frozenset()
    scalar_overlay_lineage: tuple[Mapping[str, Any], ...] = ()


def _validate_score_row(plan: PlanBundle, row: Mapping[str, Any]) -> None:
    if row.get("schema_version") != SCORE_SCHEMA_VERSION or row.get("unit_id") != UNIT_ID:
        _fail("score row schema/unit mismatch", request_id=row.get("request_id"))
    request_id = row.get("request_id")
    request = plan.requests_by_id.get(str(request_id))
    if request is None:
        _fail("score row request_id is absent from the plan", request_id=request_id)
    for key in ("request_kind", "context_id", "candidate_id", "sidecar_id", "repeat_index"):
        if row.get(key) != request.get(key):
            _fail("score row differs from its planned request", request_id=request_id, field=key)
    context = plan.contexts_by_id[str(request["context_id"])]
    if row.get("full_prefix_token_ids_sha256") != context.get("full_prefix_token_ids_sha256"):
        _fail("score row prefix digest differs from its planned context", request_id=request_id)
    if request["request_kind"] in {"primary", "numerical_repeat"}:
        source = plan.candidates_by_id[str(request["candidate_id"])]
    else:
        source = plan.sidecars_by_id[str(request["sidecar_id"])]
    coord_token_ids = [int(value) for value in _sequence(row.get("coord_token_ids"), "coord_token_ids")]
    if coord_token_ids != [int(value) for value in source["coord_token_ids"]]:
        _fail("score row coordinate tokens differ from the plan", request_id=request_id)
    if row.get("coord_token_ids_sha256") != sha256_json(coord_token_ids):
        _fail("score row coordinate-token digest does not reconstruct", request_id=request_id)
    request_kind = str(request["request_kind"])
    expected_primary = request_kind == "primary"
    if row.get("primary_role") is not expected_primary:
        _fail("score row primary_role is inconsistent with request_kind", request_id=request_id)
    if row.get("excluded_from_primary_ranks") is not (not expected_primary):
        _fail("score row rank-exclusion flag is inconsistent", request_id=request_id)
    if row.get("decision_bearing_channel") != DECISION_CHANNEL:
        _fail("score row decision channel differs from the frozen channel", request_id=request_id)
    if _finite(row.get("native_repetition_penalty_stratum"), "repetition penalty") != 1.0:
        _fail("score row repetition penalty stratum is not 1.0", request_id=request_id)
    raw = _mapping(row.get("raw_model_logprob"), "score row.raw_model_logprob")
    _finite(raw.get("complete_box_logprob_sum"), f"{request_id}.complete_box_logprob_sum")


def _validate_backend_receipt(receipt: Mapping[str, Any], label: str) -> None:
    backend = _mapping(receipt.get("scoring_backend_admission"), f"{label}.scoring_backend_admission")
    if backend.get("selected_backend") != FULL_REFORWARD_BACKEND:
        _fail("score shard did not use the frozen uncached full-reforward backend", label=label)
    if backend.get("cache_enabled") is not False or backend.get("use_cache") is not False:
        _fail("score shard enabled cache on the decision-bearing path", label=label)
    contexts_scored = _sequence(backend.get("contexts_scored"), f"{label}.contexts_scored")
    if not contexts_scored:
        _fail("score shard receipt has no scored contexts", label=label)


def _batch_score_error_bounds(
    receipts: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Return a conservative complete-box error bound from batch parity probes.

    The root distribution is scalar.  A complete box subsequently consumes
    three prefetched suffix distributions, so summing the three full
    coordinate-domain max-absolute differences bounds the chosen y1/x2/y2
    log-probability drift for the parity-probe path.
    """

    bounds = {context_id: 0.0 for context_id in CONTEXT_IDS}
    for receipt in receipts:
        backend = _mapping(receipt.get("scoring_backend_admission"), "scoring_backend_admission")
        for raw_entry in _sequence(
            backend.get("per_context_accounting"), "per_context_accounting"
        ):
            entry = _mapping(raw_entry, "per_context_accounting entry")
            context_id = str(entry.get("context_id"))
            if context_id not in bounds:
                _fail("batch accounting references an unknown context", context_id=context_id)
            admission = entry.get("batch_admission")
            if admission is None:
                continue
            admission = _mapping(admission, "batch_admission")
            status = admission.get("status")
            if status == "not_requested" or admission.get("fallback") == "scalar_explicit":
                bound = 0.0
            elif status == "passed":
                comparisons = _sequence(admission.get("comparisons"), "batch comparisons")
                if len(comparisons) != 3:
                    _fail("passed batch admission must carry three coordinate comparisons")
                bound = sum(
                    _finite(
                        _mapping(comparison, "batch comparison").get(
                            "coordinate_logprob_max_abs_diff"
                        ),
                        "coordinate_logprob_max_abs_diff",
                    )
                    for comparison in comparisons
                )
            else:
                _fail("batch admission neither passed nor recorded scalar fallback", admission=admission)
            bounds[context_id] = max(bounds[context_id], bound)
    return bounds


def load_score_shards(
    plan: PlanBundle,
    shard_dirs: Sequence[str | Path],
    *,
    input_mode: str = "strict_confirmation",
) -> ScoreBundle:
    if input_mode not in INPUT_MODES:
        _fail("unrecognized input mode", input_mode=input_mode)
    if not shard_dirs:
        _fail("at least one --score-shard-dir is required")
    all_rows: dict[str, Mapping[str, Any]] = {}
    receipt_paths: list[Path] = []
    score_paths: list[Path] = []
    receipts: list[Mapping[str, Any]] = []
    code_lineage: list[Mapping[str, Any]] = []
    source_identity_sha256: str | None = None
    plan_receipt_sha256 = sha256_file(plan.receipt_path)

    resolved_dirs: set[Path] = set()
    for raw_dir in shard_dirs:
        shard_dir = Path(raw_dir).expanduser().resolve(strict=True)
        if shard_dir in resolved_dirs:
            _fail("score shard directory was supplied more than once", path=str(shard_dir))
        resolved_dirs.add(shard_dir)
        score_path = shard_dir / SCORE_NAME
        receipt_path = shard_dir / SCORE_RECEIPT_NAME
        receipt = _read_json(receipt_path, f"score receipt {shard_dir}")
        label = str(receipt_path)
        observed_receipt_schema = receipt.get("schema_version")
        allowed_receipt_schemas = (
            {SCORE_RECEIPT_SCHEMA_VERSION}
            if input_mode == "strict_confirmation"
            else {LEGACY_SCORE_RECEIPT_SCHEMA_VERSION, SCORE_RECEIPT_SCHEMA_VERSION}
        )
        if (
            observed_receipt_schema not in allowed_receipt_schemas
            or receipt.get("row_schema_version") != SCORE_SCHEMA_VERSION
            or receipt.get("unit_id") != UNIT_ID
        ):
            _fail("score receipt schema/unit mismatch", path=label)
        _validate_self_digest(receipt, label)
        receipt_plan = _mapping(receipt.get("plan"), f"{label}.plan")
        if (
            receipt_plan.get("receipt_sha256") != plan_receipt_sha256
            or receipt_plan.get("receipt_content_sha256")
            != plan.receipt.get("receipt_content_sha256")
            or Path(str(receipt_plan.get("receipt_path"))).expanduser().resolve()
            != plan.receipt_path
        ):
            _fail("score receipt binds a different plan", path=label)
        code = _mapping(receipt.get("code"), f"{label}.code")
        code_path = Path(str(code.get("path"))).expanduser().resolve(strict=True)
        live_code_sha256 = sha256_file(code_path) if code_path.is_file() else None
        live_matches_declared = live_code_sha256 == code.get("sha256")
        if input_mode == "strict_confirmation" and not live_matches_declared:
            _fail("score receipt source hash is stale or mismatched", path=label)
        _validate_backend_receipt(receipt, label)
        identity_digest = sha256_json(_mapping(receipt.get("source_identity"), f"{label}.source_identity"))
        if source_identity_sha256 is None:
            source_identity_sha256 = identity_digest
        elif identity_digest != source_identity_sha256:
            _fail("score shards have different runtime source identities", path=label)

        rows = _read_jsonl(score_path, f"score rows {shard_dir}")
        row_ids: list[str] = []
        for row in rows:
            _validate_score_row(plan, row)
            request_id = str(row["request_id"])
            if request_id in all_rows:
                _fail("score request appears in more than one shard", request_id=request_id)
            all_rows[request_id] = row
            row_ids.append(request_id)
        declared_ids = sorted(str(value) for value in _sequence(receipt.get("row_ids"), f"{label}.row_ids"))
        if declared_ids != sorted(row_ids):
            _fail("score receipt row_ids differ from the score file", path=label)
        counts = _mapping(receipt.get("counts"), f"{label}.counts")
        if counts.get("rows") != len(rows):
            _fail("score receipt row count differs from the score file", path=label)
        selection = _mapping(receipt.get("selection"), f"{label}.selection")
        if "shard_request_ids" in selection and sorted(selection["shard_request_ids"]) != sorted(row_ids):
            _fail("score receipt selection differs from emitted rows", path=label)
        receipt_paths.append(receipt_path)
        score_paths.append(score_path)
        receipts.append(receipt)
        code_lineage.append(
            {
                "receipt_path": str(receipt_path),
                "declared_code_path": str(code_path),
                "declared_code_sha256": code.get("sha256"),
                "live_code_sha256": live_code_sha256,
                "live_matches_declared": live_matches_declared,
                "score_path": str(score_path),
                "score_sha256": sha256_file(score_path),
            }
        )

    expected = set(plan.requests_by_id)
    observed = set(all_rows)
    if observed != expected:
        _fail(
            "score shards do not provide exact complete plan coverage",
            missing_count=len(expected - observed),
            extra_count=len(observed - expected),
            missing_examples=sorted(expected - observed)[:10],
            extra_examples=sorted(observed - expected)[:10],
        )
    code_identities = {str(entry["declared_code_sha256"]) for entry in code_lineage}
    if input_mode == "strict_confirmation" and len(code_identities) != 1:
        _fail(
            "strict confirmation requires one uniform scorer code digest",
            observed=sorted(code_identities),
        )
    batch_bounds = _batch_score_error_bounds(receipts)
    return ScoreBundle(
        rows_by_request_id=all_rows,
        receipt_paths=tuple(receipt_paths),
        score_paths=tuple(score_paths),
        receipts=tuple(receipts),
        source_identity_sha256=source_identity_sha256 or "",
        input_mode=input_mode,
        scorer_code_lineage=tuple(code_lineage),
        batch_score_error_bound_by_context=batch_bounds,
    )


def _validate_scalar_v2_code_identity(
    receipt: Mapping[str, Any], label: str
) -> tuple[str, dict[str, Any]]:
    code = _mapping(receipt.get("code"), f"{label}.code")
    executed = code.get("executed_source_sha256")
    receipt_time = code.get("receipt_time_file_sha256")
    if not isinstance(executed, str) or len(executed) != 64:
        _fail("scalar receipt lacks a valid import-time executed_source_sha256", label=label)
    if code.get("sha256") != executed:
        _fail("scalar receipt code.sha256 is not the import-time executed source identity", label=label)
    if not isinstance(receipt_time, str) or len(receipt_time) != 64:
        _fail("scalar receipt lacks receipt_time_file_sha256", label=label)
    drift_detected = code.get("source_drift_detected")
    if not isinstance(drift_detected, bool) or drift_detected != (receipt_time != executed):
        _fail("scalar receipt source_drift_detected is inconsistent", label=label)
    return executed, {
        "executed_source_sha256": executed,
        "receipt_time_file_sha256": receipt_time,
        "source_drift_detected": drift_detected,
        "path": code.get("path"),
    }


def _validate_scalar_batch_size_one(receipt: Mapping[str, Any], label: str) -> None:
    backend = _mapping(
        receipt.get("scoring_backend_admission"), f"{label}.scoring_backend_admission"
    )
    if (
        backend.get("selected_backend") != FULL_REFORWARD_BACKEND
        or backend.get("cache_enabled") is not False
        or backend.get("use_cache") is not False
        or backend.get("requested_batch_size") != 1
    ):
        _fail("scalar receipt is not uncached full-reforward batch-size one", label=label)
    accounting = _sequence(
        backend.get("per_context_accounting"), f"{label}.per_context_accounting"
    )
    if not accounting:
        _fail("scalar receipt has no per-context accounting", label=label)
    for raw_entry in accounting:
        entry = _mapping(raw_entry, "scalar per-context accounting")
        admission = _mapping(entry.get("batch_admission"), "scalar batch_admission")
        if (
            admission.get("status") != "not_requested"
            or admission.get("requested_batch_size") != 1
            or admission.get("effective_batch_size") != 1
        ):
            _fail(
                "scalar overlay receipt contains a non-scalar batch admission",
                label=label,
                context_id=entry.get("context_id"),
            )


def apply_scalar_confirmation_overlay(
    plan: PlanBundle,
    legacy_scores: ScoreBundle,
    shard_dirs: Sequence[str | Path],
    *,
    manifest: Sequence[Mapping[str, Any]],
) -> ScoreBundle:
    if not shard_dirs:
        return legacy_scores
    manifest_ids = {str(row["request_id"]) for row in manifest}
    if len(manifest_ids) != len(manifest):
        _fail("analyzer-generated scalar confirmation manifest contains duplicate request IDs")
    overlay_rows: dict[str, Mapping[str, Any]] = {}
    lineage: list[Mapping[str, Any]] = []
    executed_code_identities: set[str] = set()
    plan_receipt_sha256 = sha256_file(plan.receipt_path)
    resolved_dirs: set[Path] = set()
    for raw_dir in shard_dirs:
        shard_dir = Path(raw_dir).expanduser().resolve(strict=True)
        if shard_dir in resolved_dirs:
            _fail("scalar confirmation shard directory was supplied twice", path=str(shard_dir))
        resolved_dirs.add(shard_dir)
        score_path = shard_dir / SCORE_NAME
        receipt_path = shard_dir / SCORE_RECEIPT_NAME
        receipt = _read_json(receipt_path, f"scalar confirmation receipt {receipt_path}")
        label = str(receipt_path)
        if (
            receipt.get("schema_version") != SCORE_RECEIPT_SCHEMA_VERSION
            or receipt.get("row_schema_version") != SCORE_SCHEMA_VERSION
            or receipt.get("unit_id") != UNIT_ID
        ):
            _fail("scalar confirmation receipt must use scorer receipt schema v2", path=label)
        _validate_self_digest(receipt, label)
        receipt_plan = _mapping(receipt.get("plan"), f"{label}.plan")
        if (
            receipt_plan.get("receipt_sha256") != plan_receipt_sha256
            or receipt_plan.get("receipt_content_sha256")
            != plan.receipt.get("receipt_content_sha256")
            or Path(str(receipt_plan.get("receipt_path"))).expanduser().resolve()
            != plan.receipt_path
        ):
            _fail("scalar confirmation receipt binds a different plan", path=label)
        source_digest = sha256_json(
            _mapping(receipt.get("source_identity"), f"{label}.source_identity")
        )
        if source_digest != legacy_scores.source_identity_sha256:
            _fail("scalar confirmation receipt binds a different runtime source identity", path=label)
        executed_code, code_lineage = _validate_scalar_v2_code_identity(receipt, label)
        executed_code_identities.add(executed_code)
        _validate_scalar_batch_size_one(receipt, label)

        rows = _read_jsonl(score_path, f"scalar confirmation rows {score_path}")
        row_ids: list[str] = []
        for row in rows:
            _validate_score_row(plan, row)
            request_id = str(row["request_id"])
            if request_id not in manifest_ids:
                _fail(
                    "scalar confirmation row is not a subset of the analyzer manifest",
                    request_id=request_id,
                )
            if request_id in overlay_rows:
                _fail("scalar confirmation request appears more than once", request_id=request_id)
            overlay_rows[request_id] = row
            row_ids.append(request_id)
        declared_ids = sorted(
            str(value) for value in _sequence(receipt.get("row_ids"), f"{label}.row_ids")
        )
        if declared_ids != sorted(row_ids):
            _fail("scalar confirmation receipt row_ids differ from its score file", path=label)
        counts = _mapping(receipt.get("counts"), f"{label}.counts")
        if counts.get("rows") != len(rows):
            _fail("scalar confirmation receipt row count differs from its score file", path=label)
        selection = _mapping(receipt.get("selection"), f"{label}.selection")
        if sorted(str(value) for value in selection.get("shard_request_ids", [])) != sorted(
            row_ids
        ):
            _fail("scalar confirmation selection differs from emitted rows", path=label)
        lineage.append(
            {
                "score_path": str(score_path),
                "score_sha256": sha256_file(score_path),
                "receipt_path": str(receipt_path),
                "receipt_sha256": sha256_file(receipt_path),
                "row_count": len(rows),
                "request_ids": sorted(row_ids),
                "code": code_lineage,
                "source_identity_sha256": source_digest,
            }
        )
    if len(executed_code_identities) != 1:
        _fail(
            "scalar confirmation shards do not share one import-time code identity",
            observed=sorted(executed_code_identities),
        )
    overlaid = dict(legacy_scores.rows_by_request_id)
    overlaid.update(overlay_rows)
    return ScoreBundle(
        rows_by_request_id=overlaid,
        receipt_paths=legacy_scores.receipt_paths,
        score_paths=legacy_scores.score_paths,
        receipts=legacy_scores.receipts,
        source_identity_sha256=legacy_scores.source_identity_sha256,
        input_mode=legacy_scores.input_mode,
        scorer_code_lineage=legacy_scores.scorer_code_lineage,
        batch_score_error_bound_by_context=legacy_scores.batch_score_error_bound_by_context,
        legacy_rows_by_request_id=(
            legacy_scores.legacy_rows_by_request_id or legacy_scores.rows_by_request_id
        ),
        scalar_overlay_request_ids=frozenset(overlay_rows),
        scalar_overlay_lineage=tuple(lineage),
    )


def load_attested_merged_scores(
    plan: PlanBundle,
    *,
    merged_scores_path: str | Path,
    merged_receipt_path: str | Path,
    run_attestation_path: str | Path,
    input_mode: str,
) -> ScoreBundle:
    if input_mode not in INPUT_MODES:
        _fail("unrecognized input mode", input_mode=input_mode)
    scores_path = Path(merged_scores_path).expanduser().resolve(strict=True)
    receipt_path = Path(merged_receipt_path).expanduser().resolve(strict=True)
    attestation_path = Path(run_attestation_path).expanduser().resolve(strict=True)
    receipt = _read_json(receipt_path, "merged score receipt")
    if receipt.get("schema_version") != MERGE_SCHEMA_VERSION or receipt.get("unit_id") != UNIT_ID:
        _fail("merged score receipt schema/unit mismatch")
    _validate_self_digest(receipt, "merged score receipt")
    plan_entry = _mapping(receipt.get("plan"), "merged score receipt.plan")
    if (
        plan_entry.get("receipt_sha256") != sha256_file(plan.receipt_path)
        or plan_entry.get("receipt_content_sha256") != plan.receipt.get("receipt_content_sha256")
    ):
        _fail("merged score receipt binds a different plan")
    output = _mapping(receipt.get("output_artifacts"), "merged output_artifacts")
    merged_entry = _mapping(output.get("merged_scores"), "merged output_artifacts.merged_scores")
    if (
        Path(str(merged_entry.get("path"))).expanduser().resolve() != scores_path
        or merged_entry.get("sha256") != sha256_file(scores_path)
    ):
        _fail("merged score artifact hash/path differs from its receipt")

    attestation = _read_json(attestation_path, "run attestation")
    if (
        attestation.get("schema_version") != ATTESTATION_SCHEMA_VERSION
        or attestation.get("unit_id") != UNIT_ID
        or attestation.get("disposition") != "accepted"
        or _mapping(attestation.get("raw_capture_validity"), "raw_capture_validity").get(
            "status"
        )
        != "passed"
    ):
        _fail("run attestation is not an accepted raw-capture attestation")
    attested_merge = _mapping(attestation.get("merge_receipt"), "attestation.merge_receipt")
    if (
        Path(str(attested_merge.get("path"))).expanduser().resolve() != receipt_path
        or attested_merge.get("sha256") != sha256_file(receipt_path)
    ):
        _fail("run attestation binds a different merge receipt")
    attested_scores = _mapping(attestation.get("merged_scores"), "attestation.merged_scores")
    if (
        Path(str(attested_scores.get("path"))).expanduser().resolve() != scores_path
        or attested_scores.get("sha256") != sha256_file(scores_path)
    ):
        _fail("run attestation binds a different merged scores artifact")

    rows = _read_jsonl(scores_path, "merged score rows")
    rows_by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        _validate_score_row(plan, row)
        request_id = str(row["request_id"])
        if request_id in rows_by_id:
            _fail("merged score artifact duplicates request_id", request_id=request_id)
        rows_by_id[request_id] = row
    if set(rows_by_id) != set(plan.requests_by_id):
        _fail("attested merged scores do not provide exact complete plan coverage")

    shard_lineage: list[Mapping[str, Any]] = []
    for raw_shard in _sequence(receipt.get("shards"), "merged score receipt.shards"):
        shard = _mapping(raw_shard, "merged shard lineage")
        score_ref = _mapping(shard.get("scores"), "merged shard scores ref")
        receipt_ref = _mapping(shard.get("receipt"), "merged shard receipt ref")
        source_score_path = Path(str(score_ref.get("path"))).expanduser().resolve(strict=True)
        source_receipt_path = Path(str(receipt_ref.get("path"))).expanduser().resolve(strict=True)
        if sha256_file(source_score_path) != score_ref.get("sha256"):
            _fail("merged shard lineage score hash is stale", path=str(source_score_path))
        if sha256_file(source_receipt_path) != receipt_ref.get("sha256"):
            _fail("merged shard lineage receipt hash is stale", path=str(source_receipt_path))
        shard_lineage.append(
            {
                "receipt_path": str(source_receipt_path),
                "declared_code_sha256": shard.get("code_sha256"),
                "score_path": str(source_score_path),
                "score_sha256": score_ref.get("sha256"),
                "merged_attested": True,
            }
        )
    code_digests = {str(row.get("declared_code_sha256")) for row in shard_lineage}
    if len(code_digests) != 1:
        _fail("attested merged input does not have one uniform scorer code digest")
    batch_entries = _sequence(receipt.get("batch_parity_receipts"), "batch_parity_receipts")
    synthetic_receipt = {
        "scoring_backend_admission": {
            "per_context_accounting": [
                {
                    "context_id": _mapping(entry, "batch parity entry").get("context_id"),
                    "batch_admission": _mapping(entry, "batch parity entry").get(
                        "batch_admission"
                    ),
                }
                for entry in batch_entries
            ]
        }
    }
    batch_bounds = _batch_score_error_bounds([synthetic_receipt])
    return ScoreBundle(
        rows_by_request_id=rows_by_id,
        receipt_paths=(receipt_path,),
        score_paths=(scores_path,),
        receipts=(receipt, attestation),
        source_identity_sha256=str(receipt.get("source_identity_sha256")),
        input_mode=input_mode,
        scorer_code_lineage=tuple(
            [
                *shard_lineage,
                {
                    "run_attestation_path": str(attestation_path),
                    "run_attestation_sha256": sha256_file(attestation_path),
                },
            ]
        ),
        batch_score_error_bound_by_context=batch_bounds,
    )


def score_value(row: Mapping[str, Any]) -> float:
    raw = _mapping(row.get("raw_model_logprob"), "raw_model_logprob")
    return _finite(raw.get("complete_box_logprob_sum"), "complete_box_logprob_sum")


def compute_epsilon(plan: PlanBundle, scores: ScoreBundle) -> dict[str, Any]:
    repeat_requests = sorted(
        (row for row in plan.requests if row.get("request_kind") == "numerical_repeat"),
        key=lambda row: int(row["repeat_index"]),
    )
    if len(repeat_requests) != 8 or [int(row["repeat_index"]) for row in repeat_requests] != list(range(8)):
        _fail("the frozen numerical repeat set must contain indices 0 through 7")
    expected_candidate = "primary:gt:7511:2:00:exact_gt_anchor"
    if any(
        row.get("context_id") != "self-due-gt17" or row.get("candidate_id") != expected_candidate
        for row in repeat_requests
    ):
        _fail("numerical repeat identity differs from the frozen target-blind probe")
    values = [score_value(scores.rows_by_request_id[str(row["request_id"])]) for row in repeat_requests]
    median = statistics.median(values)
    delta = max(abs(value - median) for value in values)
    epsilon = max(1e-6, 2.0 * delta)
    return {
        "candidate_id": expected_candidate,
        "context_id": "self-due-gt17",
        "repeat_count": 8,
        "repeat_scores": values,
        "median": median,
        "delta": delta,
        "epsilon": epsilon,
        "formula": "max(1e-6, 2 * max_i(abs(r_i - median(r))))",
        "execution_requirement": "uncached_scalar_fp32_full_reforward",
    }


def validate_independent_repeat_shards(
    plan: PlanBundle,
    shard_dirs: Sequence[str | Path],
    *,
    input_mode: str,
    expected_source_identity_sha256: str,
) -> dict[str, Any]:
    if len(shard_dirs) != 8:
        _fail(
            "independent numerical evidence must contain exactly eight one-process shard dirs",
            observed=len(shard_dirs),
        )
    plan_receipt_sha256 = sha256_file(plan.receipt_path)
    expected_requests = {
        str(row["request_id"]): row
        for row in plan.requests
        if row.get("request_kind") == "numerical_repeat"
    }
    observed: dict[str, float] = {}
    lineage: list[dict[str, Any]] = []
    source_digests: set[str] = set()
    for raw_dir in shard_dirs:
        shard_dir = Path(raw_dir).expanduser().resolve(strict=True)
        score_path = shard_dir / SCORE_NAME
        receipt_path = shard_dir / SCORE_RECEIPT_NAME
        receipt = _read_json(receipt_path, f"independent repeat receipt {receipt_path}")
        allowed_receipt_schemas = (
            {SCORE_RECEIPT_SCHEMA_VERSION}
            if input_mode == "strict_confirmation"
            else {LEGACY_SCORE_RECEIPT_SCHEMA_VERSION, SCORE_RECEIPT_SCHEMA_VERSION}
        )
        if (
            receipt.get("schema_version") not in allowed_receipt_schemas
            or receipt.get("row_schema_version") != SCORE_SCHEMA_VERSION
            or receipt.get("unit_id") != UNIT_ID
        ):
            _fail("independent repeat receipt schema/unit mismatch", path=str(receipt_path))
        _validate_self_digest(receipt, str(receipt_path))
        receipt_plan = _mapping(receipt.get("plan"), "independent repeat receipt.plan")
        if (
            receipt_plan.get("receipt_sha256") != plan_receipt_sha256
            or receipt_plan.get("receipt_content_sha256")
            != plan.receipt.get("receipt_content_sha256")
        ):
            _fail("independent repeat receipt binds a different plan", path=str(receipt_path))
        rows = _read_jsonl(score_path, f"independent repeat score {score_path}")
        if len(rows) != 1:
            _fail("each independent repeat process must emit exactly one row", path=str(score_path))
        row = rows[0]
        _validate_score_row(plan, row)
        request_id = str(row["request_id"])
        if request_id not in expected_requests or request_id in observed:
            _fail("independent repeat request coverage is foreign or duplicate", request_id=request_id)
        observed[request_id] = score_value(row)
        identity_digest = sha256_json(
            _mapping(receipt.get("source_identity"), "independent repeat source_identity")
        )
        source_digests.add(identity_digest)
        code = _mapping(receipt.get("code"), "independent repeat receipt.code")
        code_path = Path(str(code.get("path"))).expanduser().resolve(strict=True)
        live_code_sha256 = sha256_file(code_path)
        if input_mode == "strict_confirmation" and live_code_sha256 != code.get("sha256"):
            _fail("independent repeat scorer source hash is stale", path=str(receipt_path))
        lineage.append(
            {
                "request_id": request_id,
                "score_path": str(score_path),
                "score_sha256": sha256_file(score_path),
                "receipt_path": str(receipt_path),
                "receipt_sha256": sha256_file(receipt_path),
                "declared_code_sha256": code.get("sha256"),
                "live_code_sha256": live_code_sha256,
            }
        )
    if set(observed) != set(expected_requests):
        _fail("independent repeat shards do not cover repeat indices 0 through 7 exactly")
    if len(source_digests) != 1 or next(iter(source_digests)) != expected_source_identity_sha256:
        _fail("independent repeat runtime source identity differs from the main score capture")
    ordered_request_ids = sorted(
        observed,
        key=lambda request_id: int(expected_requests[request_id]["repeat_index"]),
    )
    values = [observed[request_id] for request_id in ordered_request_ids]
    median = statistics.median(values)
    delta = max(abs(value - median) for value in values)
    return {
        "status": "independent_process_evidence_bound",
        "process_count": 8,
        "request_ids": ordered_request_ids,
        "scores": values,
        "median": median,
        "independent_process_delta": delta,
        "epsilon": max(1e-6, 2.0 * delta),
        "formula": "max(1e-6, 2 * max_i(abs(r_i - median(r))))",
        "lineage": sorted(lineage, key=lambda row: str(row["request_id"])),
    }


def _sign(value: float, epsilon: float) -> str:
    if value > epsilon:
        return "positive"
    if value < -epsilon:
        return "negative"
    return "tied"


def _delta_sign(value: float, epsilon: float) -> str:
    if value > 2.0 * epsilon:
        return "positive"
    if value < -2.0 * epsilon:
        return "negative"
    return "tied"


def _competition_rank(owner_id: str, values: Mapping[str, float], epsilon: float) -> int:
    score = values[owner_id]
    return 1 + sum(value > score + epsilon for other, value in values.items() if other != owner_id)


def _tied_owner_ids(owner_id: str, values: Mapping[str, float], epsilon: float) -> list[str]:
    score = values[owner_id]
    return sorted(other for other, value in values.items() if abs(value - score) <= epsilon)


def _best_other(owner_id: str, values: Mapping[str, float], epsilon: float) -> tuple[float, list[str]]:
    others = {key: value for key, value in values.items() if key != owner_id}
    if not others:
        _fail("owner-relative margin requires at least one other eligible owner")
    best = max(others.values())
    return best, sorted(key for key, value in others.items() if abs(value - best) <= epsilon)


def _average_tolerance_rank(owner_id: str, values: Mapping[str, float], epsilon: float) -> float:
    score = values[owner_id]
    first = 1 + sum(value > score + epsilon for key, value in values.items() if key != owner_id)
    last = sum(value >= score - epsilon for value in values.values())
    return (first + last) / 2.0


def _average_exact_ranks(values: Sequence[float]) -> list[float]:
    result: list[float] = []
    for value in values:
        first = 1 + sum(other < value for other in values)
        last = sum(other <= value for other in values)
        result.append((first + last) / 2.0)
    return result


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    if denominator == 0.0:
        return None
    return sum(a * b for a, b in zip(left_centered, right_centered, strict=True)) / denominator


def _bbox(row: Mapping[str, Any], key: str = "bbox_pixel_xyxy") -> tuple[float, float, float, float]:
    values = _sequence(row.get(key), key)
    if len(values) != 4:
        _fail(f"{key} must contain four coordinates")
    x1, y1, x2, y2 = (_finite(value, key) for value in values)
    if not x1 < x2 or not y1 < y2:
        _fail(f"{key} is not a valid xyxy box", observed=values)
    return x1, y1, x2, y2


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / (left_area + right_area - intersection)


def owner_geometry(plan: PlanBundle) -> dict[str, dict[str, Any]]:
    boxes = {owner_id: _bbox(row) for owner_id, row in plan.owners_by_id.items()}
    result: dict[str, dict[str, Any]] = {}
    for owner_id, box in boxes.items():
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        diagonal = math.hypot(width, height)
        distances: list[tuple[str, float]] = []
        overlaps: list[tuple[str, float]] = []
        for other_id, other_box in boxes.items():
            if other_id == owner_id:
                continue
            other_center = (
                (other_box[0] + other_box[2]) / 2.0,
                (other_box[1] + other_box[3]) / 2.0,
            )
            distances.append((other_id, math.dist(center, other_center)))
            overlaps.append((other_id, _iou(box, other_box)))
        nearest_distance = min(distance for _, distance in distances)
        max_iou = max(value for _, value in overlaps)
        result[owner_id] = {
            "bbox_pixel_xyxy": list(box),
            "width_pixels": width,
            "height_pixels": height,
            "area_pixels2": width * height,
            "aspect_ratio_width_over_height": width / height,
            "center_pixel_xy": list(center),
            "diagonal_pixels": diagonal,
            "nearest_other_center_distance_pixels": nearest_distance,
            "nearest_other_center_distance_over_gt_diagonal": nearest_distance / diagonal,
            "neighbor_center_count_within_1x_gt_diagonal": sum(
                distance <= diagonal for _, distance in distances
            ),
            "neighbor_center_count_within_2x_gt_diagonal": sum(
                distance <= 2.0 * diagonal for _, distance in distances
            ),
            "overlap_other_owner_count": sum(value > 0.0 for _, value in overlaps),
            "strict_iou_0p5_other_owner_count": sum(value >= 0.5 for _, value in overlaps),
            "max_other_owner_iou": max_iou,
            "max_iou_other_owner_ids": sorted(
                other_id for other_id, value in overlaps if value == max_iou
            ),
            "sum_other_owner_iou": sum(value for _, value in overlaps),
        }
    return result


def _softmax_concentration(scores: Sequence[float]) -> dict[str, Any]:
    if not scores:
        return {
            "candidate_count": 0,
            "top_candidate_probability": None,
            "normalized_entropy": None,
            "effective_candidate_count": None,
            "top1_minus_top2_score": None,
        }
    maximum = max(scores)
    weights = [math.exp(value - maximum) for value in scores]
    total = sum(weights)
    probabilities = [value / total for value in weights]
    entropy = -sum(value * math.log(value) for value in probabilities if value > 0.0)
    ordered = sorted(scores, reverse=True)
    return {
        "candidate_count": len(scores),
        "top_candidate_probability": max(probabilities),
        "normalized_entropy": entropy / math.log(len(scores)) if len(scores) > 1 else 0.0,
        "effective_candidate_count": math.exp(entropy),
        "top1_minus_top2_score": ordered[0] - ordered[1] if len(ordered) > 1 else None,
    }


def _peak_surface(
    owner_id: str,
    candidate_rows: Sequence[tuple[Mapping[str, Any], float]],
    owner_box: Sequence[float],
    epsilon: float,
) -> dict[str, Any]:
    if not candidate_rows:
        return {
            "winning_candidate_ids": [],
            "winning_transforms": [],
            "winning_candidate_score": None,
            "winning_candidate_bbox_pixel_xyxy": None,
            "peak_center_offset_over_gt_width_height": None,
            "peak_width_height_area_ratio_to_gt": None,
            "concentration": _softmax_concentration([]),
        }
    maximum = max(score for _, score in candidate_rows)
    winners = [(candidate, score) for candidate, score in candidate_rows if abs(score - maximum) <= epsilon]
    representative = min(winners, key=lambda item: str(item[0]["candidate_id"]))[0]
    peak_box = _bbox(representative, "decoded_bbox_pixel_xyxy")
    gt_width, gt_height = owner_box[2] - owner_box[0], owner_box[3] - owner_box[1]
    gt_center = ((owner_box[0] + owner_box[2]) / 2.0, (owner_box[1] + owner_box[3]) / 2.0)
    peak_width, peak_height = peak_box[2] - peak_box[0], peak_box[3] - peak_box[1]
    peak_center = ((peak_box[0] + peak_box[2]) / 2.0, (peak_box[1] + peak_box[3]) / 2.0)
    return {
        "winning_candidate_ids": sorted(str(candidate["candidate_id"]) for candidate, _ in winners),
        "winning_transforms": sorted({str(candidate["transform"]) for candidate, _ in winners}),
        "winning_candidate_score": maximum,
        "winning_candidate_bbox_pixel_xyxy": list(peak_box),
        "peak_center_offset_over_gt_width_height": [
            (peak_center[0] - gt_center[0]) / gt_width,
            (peak_center[1] - gt_center[1]) / gt_height,
        ],
        "peak_width_height_area_ratio_to_gt": [
            peak_width / gt_width,
            peak_height / gt_height,
            (peak_width * peak_height) / (gt_width * gt_height),
        ],
        "concentration": _softmax_concentration([score for _, score in candidate_rows]),
        "owner_id": owner_id,
    }


def _bounded_outcome(rank: int, margin: float, epsilon: float) -> str:
    if rank == 1 and margin > epsilon:
        return "winner"
    if rank == 1 and abs(margin) <= epsilon:
        return "tied"
    return "not_winner"


def build_context_statistics(
    plan: PlanBundle,
    scores: ScoreBundle,
    epsilon: float,
) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    owner_ids = sorted(plan.owners_by_id, key=lambda value: int(value.rsplit(":", 1)[1]))
    for context_id in CONTEXT_IDS:
        candidate_scores: dict[str, float] = {}
        for candidate in plan.candidates:
            request_id = f"score:{context_id}:{candidate['candidate_id']}"
            candidate_scores[str(candidate["candidate_id"])] = score_value(
                scores.rows_by_request_id[request_id]
            )

        lower_candidates: dict[str, list[tuple[Mapping[str, Any], float]]] = {
            owner_id: [] for owner_id in owner_ids
        }
        upper_candidates: dict[str, list[tuple[Mapping[str, Any], float]]] = {
            owner_id: [] for owner_id in owner_ids
        }
        exact_scores: dict[str, float] = {}
        exact_candidate_ids: dict[str, str] = {}
        for candidate in plan.candidates:
            score = candidate_scores[str(candidate["candidate_id"])]
            for owner_id in _sequence(candidate.get("lower_bound_owner_ids"), "lower_bound_owner_ids"):
                if str(owner_id) in lower_candidates:
                    lower_candidates[str(owner_id)].append((candidate, score))
            for owner_id in _sequence(candidate.get("upper_bound_owner_ids"), "upper_bound_owner_ids"):
                if str(owner_id) in upper_candidates:
                    upper_candidates[str(owner_id)].append((candidate, score))
            generator_owner = str(candidate.get("generator_gt_owner_id"))
            if (
                candidate.get("candidate_family") == "exact"
                and candidate.get("strict_assignment_gt_owner_id") == generator_owner
                and list(candidate.get("lower_bound_owner_ids", [])) == [generator_owner]
            ):
                exact_scores[generator_owner] = score
                exact_candidate_ids[generator_owner] = str(candidate["candidate_id"])

        lower_values = {
            owner_id: max(score for _, score in rows)
            for owner_id, rows in lower_candidates.items()
            if rows
        }
        upper_values = {
            owner_id: max(score for _, score in rows)
            for owner_id, rows in upper_candidates.items()
            if rows
        }
        context_result: dict[str, dict[str, Any]] = {}
        for owner_id in owner_ids:
            exact_score = exact_scores.get(owner_id)
            exact: dict[str, Any]
            if exact_score is None:
                exact = {
                    "candidate_id": None,
                    "score": None,
                    "competition_rank": None,
                    "eligible_owner_count": len(exact_scores),
                    "tied_owner_ids": [],
                    "tied_owner_count": 0,
                }
            else:
                tied = _tied_owner_ids(owner_id, exact_scores, epsilon)
                exact = {
                    "candidate_id": exact_candidate_ids[owner_id],
                    "score": exact_score,
                    "competition_rank": _competition_rank(owner_id, exact_scores, epsilon),
                    "eligible_owner_count": len(exact_scores),
                    "tied_owner_ids": tied,
                    "tied_owner_count": len(tied),
                }

            lower = lower_values.get(owner_id)
            upper = upper_values.get(owner_id)
            if lower is None or upper is None:
                context_result[owner_id] = {
                    "context_id": context_id,
                    "exact_anchor": exact,
                    "neighborhood": {
                        "score": None,
                        "competition_rank": None,
                        "eligible_owner_count": len(lower_values),
                        "status": "neutral_no_uniquely_assigned_candidate",
                    },
                    "ambiguity_bounds": {
                        "lower_score": lower,
                        "upper_score": upper,
                        "status": "neutral_incomplete_bounds",
                    },
                    "peak": _peak_surface(
                        owner_id,
                        lower_candidates[owner_id],
                        _bbox(plan.owners_by_id[owner_id]),
                        epsilon,
                    ),
                }
                continue

            best_other, best_other_ids = _best_other(owner_id, lower_values, epsilon)
            margin = lower - best_other
            tied = _tied_owner_ids(owner_id, lower_values, epsilon)
            rank = _competition_rank(owner_id, lower_values, epsilon)
            other_upper = {key: value for key, value in upper_values.items() if key != owner_id}
            other_lower = {key: value for key, value in lower_values.items() if key != owner_id}
            lower_margin = lower - max(other_upper.values())
            upper_margin = upper - max(other_lower.values())
            best_rank = 1 + sum(value > upper + epsilon for value in lower_values.values())
            worst_rank = 1 + sum(value > lower + epsilon for value in upper_values.values())
            optimistic = _bounded_outcome(best_rank, upper_margin, epsilon)
            pessimistic = _bounded_outcome(worst_rank, lower_margin, epsilon)
            lower_sign, upper_sign = _sign(lower_margin, epsilon), _sign(upper_margin, epsilon)
            context_result[owner_id] = {
                "context_id": context_id,
                "exact_anchor": exact,
                "neighborhood": {
                    "score": lower,
                    "competition_rank": rank,
                    "eligible_owner_count": len(lower_values),
                    "tied_owner_ids": tied,
                    "tied_owner_count": len(tied),
                    "best_other_score": best_other,
                    "best_other_owner_ids": best_other_ids,
                    "margin": margin,
                    "margin_sign": _sign(margin, epsilon),
                    "point_outcome": _bounded_outcome(rank, margin, epsilon),
                },
                "ambiguity_bounds": {
                    "lower_score": lower,
                    "upper_score": upper,
                    "best_rank": best_rank,
                    "worst_rank": worst_rank,
                    "eligible_owner_count": len(lower_values),
                    "lower_margin": lower_margin,
                    "upper_margin": upper_margin,
                    "lower_margin_sign": lower_sign,
                    "upper_margin_sign": upper_sign,
                    "margin_sign_invariant": lower_sign == upper_sign,
                    "optimistic_outcome": optimistic,
                    "pessimistic_outcome": pessimistic,
                    "outcome_invariant": optimistic == pessimistic,
                    "invariant_outcome": optimistic if optimistic == pessimistic else None,
                    "status": "invariant" if optimistic == pessimistic and lower_sign == upper_sign else "ambiguity_sensitive",
                },
                "peak": _peak_surface(
                    owner_id,
                    lower_candidates[owner_id],
                    _bbox(plan.owners_by_id[owner_id]),
                    epsilon,
                ),
            }
        result[context_id] = context_result
    return result


def attach_sidecar_gaps(
    plan: PlanBundle,
    scores: ScoreBundle,
    contexts: dict[str, dict[str, dict[str, Any]]],
    epsilon: float,
) -> None:
    for context_id in CONTEXT_IDS:
        by_owner: dict[str, list[dict[str, Any]]] = {owner_id: [] for owner_id in plan.owners_by_id}
        for sidecar in plan.sidecars:
            owner_id = sidecar.get("strict_assignment_gt_owner_id")
            if owner_id not in by_owner:
                _fail("sidecar does not have one frozen strict owner", sidecar_id=sidecar.get("sidecar_id"))
            member_ids = [str(value) for value in sidecar.get("bank_member_candidate_ids", [])]
            if sidecar.get("requires_new_score_row"):
                request_id = f"score:{context_id}:{sidecar['sidecar_id']}"
                score = score_value(scores.rows_by_request_id[request_id])
                score_source = "dedicated_sidecar_score_row"
            else:
                if len(member_ids) != 1:
                    _fail("bank-member sidecar must bind exactly one primary candidate")
                request_id = f"score:{context_id}:{member_ids[0]}"
                score = score_value(scores.rows_by_request_id[request_id])
                score_source = "token_identical_primary_candidate"
            neighborhood_score = contexts[context_id][str(owner_id)]["neighborhood"].get("score")
            gap = score - neighborhood_score if neighborhood_score is not None else None
            by_owner[str(owner_id)].append(
                {
                    "sidecar_id": sidecar["sidecar_id"],
                    "score": score,
                    "score_source": score_source,
                    "bank_member_candidate_ids": member_ids,
                    "gap_above_primary_neighborhood_max": gap,
                    "finite_bank_undercoverage": gap is not None and gap > epsilon,
                }
            )
        for owner_id in plan.owners_by_id:
            entries = sorted(by_owner[owner_id], key=lambda row: str(row["sidecar_id"]))
            gaps = [float(row["gap_above_primary_neighborhood_max"]) for row in entries if row["gap_above_primary_neighborhood_max"] is not None]
            contexts[context_id][owner_id]["realized_sidecars"] = {
                "entries": entries,
                "maximum_gap": max(gaps) if gaps else None,
                "finite_bank_undercoverage": any(row["finite_bank_undercoverage"] for row in entries),
            }


def attach_score_overlay_provenance(
    plan: PlanBundle,
    scores: ScoreBundle,
    contexts: dict[str, dict[str, dict[str, Any]]],
    epsilon: float,
) -> None:
    scalar_ids = set(scores.scalar_overlay_request_ids)
    for context_id in CONTEXT_IDS:
        candidate_scores = {
            str(candidate["candidate_id"]): score_value(
                scores.rows_by_request_id[f"score:{context_id}:{candidate['candidate_id']}"]
            )
            for candidate in plan.candidates
        }
        for owner_id, cell in contexts[context_id].items():
            support_ids: set[str] = set()
            exact_candidate_id = cell["exact_anchor"].get("candidate_id")
            if exact_candidate_id is not None:
                support_ids.add(f"score:{context_id}:{exact_candidate_id}")
            for candidate_id in cell["peak"].get("winning_candidate_ids", []):
                support_ids.add(f"score:{context_id}:{candidate_id}")
            upper_score = cell["ambiguity_bounds"].get("upper_score")
            if upper_score is not None:
                for candidate in plan.candidates:
                    candidate_id = str(candidate["candidate_id"])
                    if (
                        owner_id in candidate.get("upper_bound_owner_ids", [])
                        and abs(candidate_scores[candidate_id] - upper_score) <= epsilon
                    ):
                        support_ids.add(f"score:{context_id}:{candidate_id}")
            scalar_support = sorted(support_ids & scalar_ids)
            legacy_support = sorted(support_ids - scalar_ids)
            if support_ids and not legacy_support:
                status = "decision_support_scalar_overlaid"
            elif scalar_support:
                status = "decision_support_mixed_scalar_and_legacy_batch"
            else:
                status = "decision_support_legacy_batch_only"
            cell["decision_score_provenance"] = {
                "status": status,
                "scalar_request_ids": scalar_support,
                "legacy_batch_request_ids": legacy_support,
            }


def _rank_confirmation_interval(
    owner_id: str,
    values: Mapping[str, float],
    *,
    score_error_bound: float,
    epsilon: float,
) -> tuple[int, int]:
    owner_score = values[owner_id]
    best_rank = 1 + sum(
        value - score_error_bound > owner_score + score_error_bound + epsilon
        for other_id, value in values.items()
        if other_id != owner_id
    )
    worst_rank = 1 + sum(
        value + score_error_bound > owner_score - score_error_bound + epsilon
        for other_id, value in values.items()
        if other_id != owner_id
    )
    return best_rank, worst_rank


def attach_batch_scalar_confirmation(
    plan: PlanBundle,
    contexts: dict[str, dict[str, dict[str, Any]]],
    scores: ScoreBundle,
    epsilon: float,
) -> dict[str, Any]:
    context_admission: dict[str, Any] = {}
    for context_id in CONTEXT_IDS:
        error = float(scores.batch_score_error_bound_by_context[context_id])
        exact_values = {
            owner_id: float(cell["exact_anchor"]["score"])
            for owner_id, cell in contexts[context_id].items()
            if cell["exact_anchor"].get("score") is not None
        }
        neighborhood_values = {
            owner_id: float(cell["neighborhood"]["score"])
            for owner_id, cell in contexts[context_id].items()
            if cell["neighborhood"].get("score") is not None
        }
        ambiguity_lower_values = {
            owner_id: float(cell["ambiguity_bounds"]["lower_score"])
            for owner_id, cell in contexts[context_id].items()
            if cell["ambiguity_bounds"].get("lower_score") is not None
        }
        ambiguity_upper_values = {
            owner_id: float(cell["ambiguity_bounds"]["upper_score"])
            for owner_id, cell in contexts[context_id].items()
            if cell["ambiguity_bounds"].get("upper_score") is not None
        }
        unknown_owner_ids: list[str] = []
        for owner_id in plan.owners_by_id:
            cell = contexts[context_id][owner_id]
            exact_confirmation: dict[str, Any]
            if owner_id in exact_values:
                exact_best, exact_worst = _rank_confirmation_interval(
                    owner_id,
                    exact_values,
                    score_error_bound=error,
                    epsilon=epsilon,
                )
                exact_confirmation = {
                    "best_possible_scalar_rank": exact_best,
                    "worst_possible_scalar_rank": exact_worst,
                    "status": (
                        "confirmed_within_batch_scalar_bound"
                        if exact_best == exact_worst
                        else "unknown_needs_scalar_confirmation"
                    ),
                }
            else:
                exact_confirmation = {"status": "neutral_no_exact_score"}

            neighborhood_confirmation: dict[str, Any]
            if owner_id in neighborhood_values:
                rank_best, rank_worst = _rank_confirmation_interval(
                    owner_id,
                    neighborhood_values,
                    score_error_bound=error,
                    epsilon=epsilon,
                )
                margin = float(cell["neighborhood"]["margin"])
                margin_low, margin_high = margin - 2.0 * error, margin + 2.0 * error
                signs = {_sign(margin_low, epsilon), _sign(margin_high, epsilon)}
                rank_confirmed = rank_best == rank_worst
                sign_confirmed = len(signs) == 1
                status = (
                    "confirmed_within_batch_scalar_bound"
                    if rank_confirmed and sign_confirmed
                    else "unknown_needs_scalar_confirmation"
                )
                neighborhood_confirmation = {
                    "best_possible_scalar_rank": rank_best,
                    "worst_possible_scalar_rank": rank_worst,
                    "scalar_margin_lower_bound": margin_low,
                    "scalar_margin_upper_bound": margin_high,
                    "margin_sign_possibilities": sorted(signs),
                    "status": status,
                }
                if status == "unknown_needs_scalar_confirmation":
                    unknown_owner_ids.append(owner_id)
            else:
                neighborhood_confirmation = {"status": "neutral_no_neighborhood_score"}
            cell["batch_scalar_confirmation"] = {
                "input_mode": scores.input_mode,
                "complete_box_score_error_bound": error,
                "epsilon": epsilon,
                "exact_anchor": exact_confirmation,
                "neighborhood_rank_and_margin": neighborhood_confirmation,
            }

            ambiguity = cell["ambiguity_bounds"]
            if owner_id in ambiguity_lower_values and owner_id in ambiguity_upper_values:
                lower_score = ambiguity_lower_values[owner_id]
                upper_score = ambiguity_upper_values[owner_id]
                best_rank_low = 1 + sum(
                    value - error > upper_score + error + epsilon
                    for value in ambiguity_lower_values.values()
                )
                best_rank_high = 1 + sum(
                    value + error > upper_score - error + epsilon
                    for value in ambiguity_lower_values.values()
                )
                worst_rank_low = 1 + sum(
                    value - error > lower_score + error + epsilon
                    for value in ambiguity_upper_values.values()
                )
                worst_rank_high = 1 + sum(
                    value + error > lower_score - error + epsilon
                    for value in ambiguity_upper_values.values()
                )
                lower_margin = float(ambiguity["lower_margin"])
                upper_margin = float(ambiguity["upper_margin"])
                ambiguity_confirmed = (
                    best_rank_low == best_rank_high
                    and worst_rank_low == worst_rank_high
                    and len(
                        {
                            _sign(lower_margin - 2.0 * error, epsilon),
                            _sign(lower_margin + 2.0 * error, epsilon),
                        }
                    )
                    == 1
                    and len(
                        {
                            _sign(upper_margin - 2.0 * error, epsilon),
                            _sign(upper_margin + 2.0 * error, epsilon),
                        }
                    )
                    == 1
                )
                ambiguity["batch_scalar_confirmation"] = {
                    "best_rank_lower_bound": best_rank_low,
                    "best_rank_upper_bound": best_rank_high,
                    "worst_rank_lower_bound": worst_rank_low,
                    "worst_rank_upper_bound": worst_rank_high,
                    "lower_margin_scalar_interval": [
                        lower_margin - 2.0 * error,
                        lower_margin + 2.0 * error,
                    ],
                    "upper_margin_scalar_interval": [
                        upper_margin - 2.0 * error,
                        upper_margin + 2.0 * error,
                    ],
                    "status": (
                        "confirmed_within_batch_scalar_bound"
                        if ambiguity_confirmed
                        else "unknown_needs_scalar_confirmation"
                    ),
                }

            peak = cell["peak"]
            top_gap = peak["concentration"].get("top1_minus_top2_score")
            peak["batch_scalar_confirmation"] = {
                "status": (
                    "confirmed_within_batch_scalar_bound"
                    if top_gap is None or top_gap > epsilon + 2.0 * error
                    else "unknown_needs_scalar_confirmation"
                ),
                "minimum_confirming_top1_minus_top2": epsilon + 2.0 * error,
            }
            for sidecar in cell["realized_sidecars"]["entries"]:
                gap = sidecar["gap_above_primary_neighborhood_max"]
                if gap is None:
                    sidecar["batch_scalar_confirmation"] = {"status": "neutral"}
                    continue
                lower, upper = gap - 2.0 * error, gap + 2.0 * error
                if lower > epsilon:
                    status = "undercoverage_confirmed_within_batch_scalar_bound"
                elif upper <= epsilon:
                    status = "no_undercoverage_confirmed_within_batch_scalar_bound"
                else:
                    status = "unknown_needs_scalar_confirmation"
                sidecar["batch_scalar_confirmation"] = {
                    "gap_lower_bound": lower,
                    "gap_upper_bound": upper,
                    "status": status,
                }
                sidecar["finite_bank_undercoverage_confirmed"] = (
                    True
                    if status == "undercoverage_confirmed_within_batch_scalar_bound"
                    else False
                    if status == "no_undercoverage_confirmed_within_batch_scalar_bound"
                    else None
                )
        context_admission[context_id] = {
            "complete_box_score_error_bound": error,
            "epsilon": epsilon,
            "batch_drift_exceeds_epsilon": error > epsilon,
            "unknown_needs_scalar_confirmation_owner_ids": sorted(unknown_owner_ids),
            "status": (
                "confirmed_within_frozen_epsilon"
                if error <= epsilon
                else "unresolved_needs_scalar_confirmation"
            ),
        }
    return context_admission


def _frontier_from_context(
    context: Mapping[str, Any], *, image_width: int, image_height: int
) -> dict[str, Any] | None:
    person_rows = [
        row
        for row in _sequence(context.get("donor_rows"), "context.donor_rows")
        if "<|object_ref_start|>person<|object_ref_end|>" in str(row.get("raw_span_text"))
    ]
    if not person_rows:
        return None
    row = person_rows[-1]
    bins = [int(value) for value in _COORD_RE.findall(str(row.get("raw_span_text")))]
    if len(bins) != 4:
        _fail("last donor person row does not contain four coordinate bins", context=context.get("context_id"))
    x1, y1, x2, y2 = (
        int(round(bins[0] * image_width / 1000.0)),
        int(round(bins[1] * image_height / 1000.0)),
        int(round(bins[2] * image_width / 1000.0)),
        int(round(bins[3] * image_height / 1000.0)),
    )
    return {
        "donor_row_index": int(row["row_index"]),
        "coord_bins": bins,
        "bbox_pixel_xyxy": [x1, y1, x2, y2],
        "sort_key_y1_x1": [y1, x1],
    }


def build_scan_statistics(
    plan: PlanBundle,
    contexts: dict[str, dict[str, dict[str, Any]]],
    epsilon: float,
) -> dict[str, dict[str, Any]]:
    runtime = _mapping(plan.receipt.get("frozen_runtime_identity"), "frozen_runtime_identity")
    image_width, image_height = int(runtime["image_width"]), int(runtime["image_height"])
    owner_order = sorted(
        plan.owners_by_id,
        key=lambda owner_id: (
            _bbox(plan.owners_by_id[owner_id])[1],
            _bbox(plan.owners_by_id[owner_id])[0],
            owner_id,
        ),
    )
    owner_pairs = [
        (_bbox(plan.owners_by_id[owner_id])[1], _bbox(plan.owners_by_id[owner_id])[0])
        for owner_id in owner_order
    ]
    result: dict[str, dict[str, Any]] = {}
    for context_id in CONTEXT_IDS:
        if context_id == "root":
            result[context_id] = {"status": "not_applicable_root_has_no_frontier"}
            continue
        frontier = _frontier_from_context(
            plan.contexts_by_id[context_id], image_width=image_width, image_height=image_height
        )
        if frontier is None:
            result[context_id] = {"status": "unavailable_no_complete_donor_person_row"}
            continue
        frontier_pair = tuple(frontier["sort_key_y1_x1"])
        insertion = bisect_right(owner_pairs, frontier_pair)
        distances: dict[str, dict[str, int]] = {}
        for owner_index, owner_id in enumerate(owner_order):
            signed = owner_index - insertion if owner_index < insertion else owner_index + 1 - insertion
            distances[owner_id] = {"signed_ordinal_distance": signed, "absolute_ordinal_distance": abs(signed)}
            contexts[context_id][owner_id]["scan_frontier_distance"] = distances[owner_id]
        values = {
            owner_id: contexts[context_id][owner_id]["neighborhood"]["score"]
            for owner_id in owner_order
            if contexts[context_id][owner_id]["neighborhood"].get("score") is not None
        }
        eligible = [owner_id for owner_id in owner_order if owner_id in values]
        score_ranks = [_average_tolerance_rank(owner_id, values, epsilon) for owner_id in eligible]
        distance_ranks = _average_exact_ranks(
            [float(distances[owner_id]["absolute_ordinal_distance"]) for owner_id in eligible]
        )
        result[context_id] = {
            "status": "descriptive_only",
            "frontier": frontier,
            "frontier_insertion_index_zero_based": insertion,
            "eligible_owner_count": len(eligible),
            "spearman_owner_score_rank_vs_absolute_scan_distance": _pearson(
                score_ranks, distance_ranks
            ),
            "tie_policy": "score ranks use epsilon-average occupied positions; distance ranks use exact average ties",
            "per_owner_distances": distances,
        }
    return result


def _context_delta(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    epsilon: float,
    *,
    before_score_error_bound: float = 0.0,
    after_score_error_bound: float = 0.0,
    before_decision_admission: str | None = None,
    after_decision_admission: str | None = None,
) -> dict[str, Any]:
    def delta(path: tuple[str, str]) -> float | int | None:
        left = _mapping(before.get(path[0]), path[0]).get(path[1])
        right = _mapping(after.get(path[0]), path[0]).get(path[1])
        if left is None or right is None:
            return None
        return right - left

    exact_delta = delta(("exact_anchor", "score"))
    neighborhood_delta = delta(("neighborhood", "score"))
    rank_delta = delta(("neighborhood", "competition_rank"))
    margin_delta = delta(("neighborhood", "margin"))
    lower_margin_delta = delta(("ambiguity_bounds", "lower_margin"))
    upper_margin_delta = delta(("ambiguity_bounds", "upper_margin"))
    score_delta_error = before_score_error_bound + after_score_error_bound
    margin_delta_error = 2.0 * score_delta_error
    legacy_batch_bound = {
        "score_delta_error_bound": score_delta_error,
        "margin_delta_error_bound": margin_delta_error,
        "neighborhood_max_delta_lower_bound": (
            neighborhood_delta - score_delta_error if neighborhood_delta is not None else None
        ),
        "neighborhood_max_delta_upper_bound": (
            neighborhood_delta + score_delta_error if neighborhood_delta is not None else None
        ),
        "owner_relative_margin_delta_lower_bound": (
            margin_delta - margin_delta_error if margin_delta is not None else None
        ),
        "owner_relative_margin_delta_upper_bound": (
            margin_delta + margin_delta_error if margin_delta is not None else None
        ),
        "status": (
            "confirmed_within_frozen_epsilon"
            if score_delta_error <= 2.0 * epsilon
            else "unknown_needs_scalar_confirmation_for_near_threshold_deltas"
        ),
    }
    scalar_confirmed = (
        before_decision_admission == "fully_scalar_confirmed_decision"
        and after_decision_admission == "fully_scalar_confirmed_decision"
    )
    batch_scalar_confirmation = (
        {
            "score_delta_error_bound": 0.0,
            "margin_delta_error_bound": 0.0,
            "neighborhood_max_delta_lower_bound": neighborhood_delta,
            "neighborhood_max_delta_upper_bound": neighborhood_delta,
            "owner_relative_margin_delta_lower_bound": margin_delta,
            "owner_relative_margin_delta_upper_bound": margin_delta,
            "status": "scalar_confirmed_delta",
            "legacy_batch_bound": legacy_batch_bound,
        }
        if scalar_confirmed
        else legacy_batch_bound
    )
    return {
        "exact_score_delta": exact_delta,
        "exact_score_delta_status_2epsilon": _delta_sign(exact_delta, epsilon) if exact_delta is not None else None,
        "neighborhood_max_delta": neighborhood_delta,
        "neighborhood_max_delta_status_2epsilon": _delta_sign(neighborhood_delta, epsilon) if neighborhood_delta is not None else None,
        "neighborhood_rank_delta": rank_delta,
        "owner_relative_margin_delta": margin_delta,
        "owner_relative_margin_delta_status_2epsilon": _delta_sign(margin_delta, epsilon) if margin_delta is not None else None,
        "ambiguity_lower_margin_delta": lower_margin_delta,
        "ambiguity_upper_margin_delta": upper_margin_delta,
        "batch_scalar_confirmation": batch_scalar_confirmation,
    }


def _prefix_present_confirmed_owners(plan: PlanBundle, context_id: str) -> set[str]:
    donor_indices = set(plan.contexts_by_id[context_id].get("donor_row_indices", []))
    present: set[str] = set()
    for sidecar in plan.sidecars:
        source = _mapping(sidecar.get("source"), "sidecar.source")
        row_index = source.get("donor_row_index")
        owner_id = sidecar.get("strict_assignment_gt_owner_id")
        if row_index in donor_indices and owner_id in plan.owners_by_id:
            present.add(str(owner_id))
    return present


def compute_selectivity(
    plan: PlanBundle,
    contexts: Mapping[str, Mapping[str, Mapping[str, Any]]],
    epsilon: float,
    *,
    score_error_bound_by_context: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    target = "gt:7511:17"
    due_id, post_id = "self-due-gt17", "skip-post-gt17"
    present = _prefix_present_confirmed_owners(plan, due_id) | _prefix_present_confirmed_owners(
        plan, post_id
    )
    controls: list[str] = []
    excluded: dict[str, str] = {}
    d_point: dict[str, float] = {}
    d_lower: dict[str, float] = {}
    d_upper: dict[str, float] = {}
    for owner_id in sorted(plan.owners_by_id):
        due = contexts[due_id][owner_id]
        post = contexts[post_id][owner_id]
        due_bounds = due["ambiguity_bounds"]
        post_bounds = post["ambiguity_bounds"]
        if due["neighborhood"].get("margin") is None or post["neighborhood"].get("margin") is None:
            excluded[owner_id] = "missing_admissible_margin"
            continue
        d_point[owner_id] = post["neighborhood"]["margin"] - due["neighborhood"]["margin"]
        d_lower[owner_id] = post_bounds["lower_margin"] - due_bounds["upper_margin"]
        d_upper[owner_id] = post_bounds["upper_margin"] - due_bounds["lower_margin"]
        if owner_id == target:
            continue
        if owner_id in present:
            excluded[owner_id] = "confirmed_owner_present_in_due_or_post_prefix"
        elif not due_bounds.get("outcome_invariant") or not post_bounds.get("outcome_invariant"):
            excluded[owner_id] = "ambiguity_bounds_do_not_admit_invariant_status"
        else:
            controls.append(owner_id)
    if target not in d_point or not controls:
        return {
            "target_owner_id": target,
            "status": "unresolved_missing_target_or_controls",
            "control_owner_ids": controls,
            "excluded_control_reasons": excluded,
        }
    point_threshold = min(d_point[owner_id] for owner_id in controls) - 2.0 * epsilon
    point_pass = d_point[target] < point_threshold
    invariant_pass = d_upper[target] < min(d_lower[owner_id] for owner_id in controls) - 2.0 * epsilon
    invariant_fail = d_lower[target] >= min(d_upper[owner_id] for owner_id in controls) - 2.0 * epsilon
    if invariant_pass:
        status = "prefix_content_sensitive_descriptive"
    elif invariant_fail:
        status = "not_target_selective_under_frozen_inequality"
    else:
        status = "ambiguity_sensitive_unresolved"
    score_error_bound_by_context = score_error_bound_by_context or {}
    batch_margin_d_error = 2.0 * (
        float(score_error_bound_by_context.get(due_id, 0.0))
        + float(score_error_bound_by_context.get(post_id, 0.0))
    )
    batch_invariant_pass = (
        d_upper[target] + batch_margin_d_error
        < min(d_lower[owner_id] - batch_margin_d_error for owner_id in controls) - 2.0 * epsilon
    )
    batch_invariant_fail = (
        d_lower[target] - batch_margin_d_error
        >= min(d_upper[owner_id] + batch_margin_d_error for owner_id in controls)
        - 2.0 * epsilon
    )
    if batch_invariant_pass:
        batch_status = "prefix_content_sensitive_descriptive_confirmed_within_batch_scalar_bound"
    elif batch_invariant_fail:
        batch_status = "not_selective_confirmed_within_batch_scalar_bound"
    else:
        batch_status = "unknown_needs_scalar_confirmation"
    return {
        "target_owner_id": target,
        "due_context_id": due_id,
        "post_context_id": post_id,
        "present_confirmed_owner_ids_excluded": sorted(present),
        "control_owner_ids": controls,
        "excluded_control_reasons": excluded,
        "D_by_owner": d_point,
        "D_lower_by_owner": d_lower,
        "D_upper_by_owner": d_upper,
        "target_D": d_point[target],
        "point_control_min_D_minus_2epsilon": point_threshold,
        "point_inequality_passed": point_pass,
        "ambiguity_invariant_inequality_passed": invariant_pass,
        "ambiguity_invariant_inequality_failed": invariant_fail,
        "status": status,
        "batch_scalar_confirmation": {
            "D_error_bound": batch_margin_d_error,
            "ambiguity_and_batch_invariant_inequality_passed": batch_invariant_pass,
            "ambiguity_and_batch_invariant_inequality_failed": batch_invariant_fail,
            "status": batch_status,
        },
        "claim_scope": "descriptive_shared_prefix_contrast_not_causal",
    }


def calibration_summary(
    contexts: Mapping[str, Mapping[str, Mapping[str, Any]]], epsilon: float
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for owner_id, context_id in CALIBRATION_CONTEXT_BY_OWNER.items():
        cell = contexts[context_id][owner_id]
        neighborhood = cell["neighborhood"]
        outcome = neighborhood.get("point_outcome")
        rows.append(
            {
                "owner_id": owner_id,
                "context_id": context_id,
                "rank": neighborhood.get("competition_rank"),
                "margin": neighborhood.get("margin"),
                "outcome": f"{outcome}_at_due" if outcome is not None else "neutral_at_due",
                "ambiguity_invariant": cell["ambiguity_bounds"].get("outcome_invariant"),
                "batch_scalar_confirmation": cell["batch_scalar_confirmation"][
                    "neighborhood_rank_and_margin"
                ]["status"],
            }
        )
    ranks = [row["rank"] for row in rows]
    if ranks == [1, 1, 1, 1]:
        label = "controls_strong"
    else:
        rank2_rows = [row for row in rows if row["rank"] == 2]
        rank1_count = sum(row["rank"] == 1 for row in rows)
        marginal = (
            rank1_count == 3
            and len(rank2_rows) == 1
            and rank2_rows[0]["margin"] is not None
            and -rank2_rows[0]["margin"] <= epsilon
        )
        label = "controls_marginal" if marginal else "controls_descriptive_only"
    return {
        "cells": rows,
        "label": label,
        "label_scope": "compact_descriptive_calibration_only_not_independent_replication",
    }


def load_native_greedy_join(paths: Sequence[str | Path] | None) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for raw_path in paths or ():
        path = Path(raw_path).expanduser().resolve(strict=True)
        rows = _read_jsonl(path, f"native greedy join {path}")
        for row in rows:
            context_id = row.get("context_id")
            owner_id = row.get("target_owner_id", row.get("gt_owner_id", row.get("owner_id")))
            if context_id not in CONTEXT_IDS or not isinstance(owner_id, str):
                _fail("native greedy join row lacks a recognized context/owner key", path=str(path))
            key = (str(context_id), owner_id)
            if key in result:
                _fail("native greedy join duplicates a context/owner cell", context_id=context_id, owner_id=owner_id)
            result[key] = {
                "source_path": str(path),
                "source_file_sha256": sha256_file(path),
                "source_row_sha256": sha256_json(row),
                "row": row,
                "interpretation": "joined_descriptive_behavior_only_does_not_change_landscape_rank",
            }
    return result


def build_owner_rows(
    plan: PlanBundle,
    contexts: dict[str, dict[str, dict[str, Any]]],
    geometry: Mapping[str, Mapping[str, Any]],
    selectivity: Mapping[str, Any],
    epsilon: float,
    native_greedy: Mapping[tuple[str, str], Mapping[str, Any]],
    score_error_bound_by_context: Mapping[str, float],
    scalar_overlay_admission: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    d_by_owner = _mapping(selectivity.get("D_by_owner", {}), "selectivity.D_by_owner")
    overlay_contexts = _mapping(
        scalar_overlay_admission.get("per_context", {}),
        "scalar_overlay_admission.per_context",
    )

    def context_admission(context_id: str) -> str | None:
        return _mapping(
            overlay_contexts.get(context_id, {}),
            f"scalar_overlay_admission.per_context.{context_id}",
        ).get("status")

    for owner_id in sorted(plan.owners_by_id, key=lambda value: int(value.rsplit(":", 1)[1])):
        context_rows = {context_id: contexts[context_id][owner_id] for context_id in CONTEXT_IDS}
        deltas_from_root = {
            context_id: _context_delta(
                context_rows["root"],
                context_rows[context_id],
                epsilon,
                before_score_error_bound=float(score_error_bound_by_context["root"]),
                after_score_error_bound=float(score_error_bound_by_context[context_id]),
                before_decision_admission=context_admission("root"),
                after_decision_admission=context_admission(context_id),
            )
            for context_id in CONTEXT_IDS
            if context_id != "root"
        }
        shared_swap = _context_delta(
            context_rows["self-due-gt22"],
            context_rows["skip-post-gt17"],
            epsilon,
            before_score_error_bound=float(score_error_bound_by_context["self-due-gt22"]),
            after_score_error_bound=float(score_error_bound_by_context["skip-post-gt17"]),
            before_decision_admission=context_admission("self-due-gt22"),
            after_decision_admission=context_admission("skip-post-gt17"),
        )
        roles: list[str] = []
        if owner_id in PRIMARY_TARGETS:
            roles.append("preregistered_primary_target")
        if owner_id in CALIBRATION_CONTEXT_BY_OWNER:
            roles.append("sampled_self_due_calibration_owner")
        if owner_id == "gt:7511:17":
            roles.append("positive_calibration_control_not_analysis_center")
        row = {
            "schema_version": OWNER_ROW_SCHEMA_VERSION,
            "unit_id": UNIT_ID,
            "owner_id": owner_id,
            "original_annotation_index": plan.owners_by_id[owner_id]["original_annotation_index"],
            "preregistered_roles": roles,
            "geometry_and_neighbor_context": dict(geometry[owner_id]),
            "contexts": context_rows,
            "context_deltas_from_root": deltas_from_root,
            "shared_context_contrasts": {
                "gt17_due_to_skip_post_D": d_by_owner.get(owner_id),
                "same_length_owner_swap_skip_post_gt17_minus_self_due_gt22": shared_swap,
                "limitation": (
                    "these contexts are shared interventions; self-due/post names do not make them "
                    "owner-specific due/post interventions for this owner"
                ),
            },
            "native_greedy_by_context": {
                context_id: native_greedy.get((context_id, owner_id)) for context_id in CONTEXT_IDS
            },
            "posthoc_discovery_cohort_labels": [],
            "posthoc_label_status": (
                "not_materialized; any hard phenotype cohort must be a separate discovery-only "
                "analysis and cannot become an admission gate or causal truth"
            ),
        }
        rows.append(row)
    return rows


def build_scalar_confirmation_manifest(
    plan: PlanBundle,
    scores: ScoreBundle,
    contexts: Mapping[str, Mapping[str, Mapping[str, Any]]],
    epsilon: float,
) -> list[dict[str, Any]]:
    selections: dict[tuple[str, str], set[str]] = {}

    def add(context_id: str, candidate_id: str, reason: str) -> None:
        request_id = f"score:{context_id}:{candidate_id}"
        if request_id not in scores.rows_by_request_id:
            _fail(
                "scalar confirmation manifest referenced an unscored request",
                request_id=request_id,
            )
        selections.setdefault((context_id, request_id), set()).add(reason)

    candidate_scores_by_context: dict[str, dict[str, float]] = {}
    lower_support_by_context: dict[str, dict[str, list[str]]] = {}
    upper_support_by_context: dict[str, dict[str, list[str]]] = {}
    for context_id in CONTEXT_IDS:
        candidate_scores = {
            str(candidate["candidate_id"]): score_value(
                scores.rows_by_request_id[f"score:{context_id}:{candidate['candidate_id']}"]
            )
            for candidate in plan.candidates
        }
        candidate_scores_by_context[context_id] = candidate_scores
        lower_support: dict[str, list[str]] = {owner_id: [] for owner_id in plan.owners_by_id}
        upper_support: dict[str, list[str]] = {owner_id: [] for owner_id in plan.owners_by_id}
        for owner_id in plan.owners_by_id:
            lower_candidates = [
                str(candidate["candidate_id"])
                for candidate in plan.candidates
                if owner_id in candidate.get("lower_bound_owner_ids", [])
            ]
            upper_candidates = [
                str(candidate["candidate_id"])
                for candidate in plan.candidates
                if owner_id in candidate.get("upper_bound_owner_ids", [])
            ]
            if lower_candidates:
                lower_max = max(candidate_scores[candidate_id] for candidate_id in lower_candidates)
                lower_support[owner_id] = sorted(
                    candidate_id
                    for candidate_id in lower_candidates
                    if abs(candidate_scores[candidate_id] - lower_max) <= epsilon
                )
            if upper_candidates:
                upper_max = max(candidate_scores[candidate_id] for candidate_id in upper_candidates)
                upper_support[owner_id] = sorted(
                    candidate_id
                    for candidate_id in upper_candidates
                    if abs(candidate_scores[candidate_id] - upper_max) <= epsilon
                )
        lower_support_by_context[context_id] = lower_support
        upper_support_by_context[context_id] = upper_support

        exact_ids = [
            str(candidate["candidate_id"])
            for candidate in plan.candidates
            if candidate.get("candidate_family") == "exact"
        ]
        if len(exact_ids) != 41:
            _fail("scalar confirmation manifest expected exactly 41 exact anchors per context")
        for candidate_id in exact_ids:
            add(context_id, candidate_id, "all_41_exact_anchors")
        for owner_id in sorted(plan.owners_by_id):
            for candidate_id in lower_support[owner_id]:
                add(context_id, candidate_id, f"owner_bank_winner:{owner_id}")
                add(context_id, candidate_id, f"global_rank_margin_tie_support:{owner_id}")
                add(context_id, candidate_id, f"ambiguity_lower_bound_support:{owner_id}")
            for candidate_id in upper_support[owner_id]:
                add(context_id, candidate_id, f"ambiguity_upper_bound_support:{owner_id}")

        error = float(scores.batch_score_error_bound_by_context[context_id])
        owner_values = {
            owner_id: float(contexts[context_id][owner_id]["neighborhood"]["score"])
            for owner_id in plan.owners_by_id
            if contexts[context_id][owner_id]["neighborhood"].get("score") is not None
        }
        for owner_id, owner_value in owner_values.items():
            confirmation = contexts[context_id][owner_id]["batch_scalar_confirmation"]
            status = confirmation["neighborhood_rank_and_margin"]["status"]
            if status == "unknown_needs_scalar_confirmation":
                near_owner_ids = [
                    other_id
                    for other_id, other_value in owner_values.items()
                    if abs(other_value - owner_value) <= epsilon + 2.0 * error
                ]
                for near_owner_id in near_owner_ids:
                    for candidate_id in lower_support[near_owner_id]:
                        add(
                            context_id,
                            candidate_id,
                            f"batch_bound_near_rank_or_margin_threshold:{owner_id}",
                        )
            peak_status = contexts[context_id][owner_id]["peak"][
                "batch_scalar_confirmation"
            ]["status"]
            if peak_status == "unknown_needs_scalar_confirmation":
                for candidate in plan.candidates:
                    candidate_id = str(candidate["candidate_id"])
                    if owner_id not in candidate.get("lower_bound_owner_ids", []):
                        continue
                    if owner_value - candidate_scores[candidate_id] <= epsilon + 2.0 * error:
                        add(
                            context_id,
                            candidate_id,
                            f"batch_bound_near_owner_peak_threshold:{owner_id}",
                        )
        for owner_id in plan.owners_by_id:
            for sidecar in contexts[context_id][owner_id]["realized_sidecars"]["entries"]:
                confirmation = sidecar.get("batch_scalar_confirmation", {})
                if confirmation.get("status") != "unknown_needs_scalar_confirmation":
                    continue
                sidecar_row = plan.sidecars_by_id[str(sidecar["sidecar_id"])]
                if sidecar_row.get("requires_new_score_row"):
                    request_id = f"score:{context_id}:{sidecar['sidecar_id']}"
                    selections.setdefault((context_id, request_id), set()).add(
                        f"batch_bound_near_sidecar_gap_threshold:{owner_id}"
                    )
                else:
                    for candidate_id in sidecar_row.get("bank_member_candidate_ids", []):
                        add(
                            context_id,
                            str(candidate_id),
                            f"batch_bound_near_sidecar_gap_threshold:{owner_id}",
                        )

    manifest: list[dict[str, Any]] = []
    for (context_id, request_id), reasons in sorted(selections.items()):
        request = plan.requests_by_id[request_id]
        manifest.append(
            {
                "schema_version": "sorted-all-person-route-landscape-scalar-confirmation-selection.v1",
                "unit_id": UNIT_ID,
                "context_id": context_id,
                "request_id": request_id,
                "request_kind": request["request_kind"],
                "candidate_id": request.get("candidate_id"),
                "sidecar_id": request.get("sidecar_id"),
                "observed_batched_score": score_value(scores.rows_by_request_id[request_id]),
                "complete_box_score_error_bound": scores.batch_score_error_bound_by_context[
                    context_id
                ],
                "selection_reasons": sorted(reasons),
                "execution_requirement": "uncached_scalar_fp32_full_reforward",
            }
        )
    return manifest


def build_scalar_overlay_admission(
    *,
    raw_manifest: Sequence[Mapping[str, Any]],
    recomputed_manifest: Sequence[Mapping[str, Any]],
    scores: ScoreBundle,
) -> dict[str, Any]:
    expected_ids = {str(row["request_id"]) for row in raw_manifest}
    recomputed_ids = {str(row["request_id"]) for row in recomputed_manifest}
    overlay_ids = set(scores.scalar_overlay_request_ids)
    if not overlay_ids:
        return {
            "status": "discovery_only_raw",
            "decision_admission": "discovery_only_raw",
            "expected_manifest_request_count": len(expected_ids),
            "scalar_overlay_request_count": 0,
            "missing_original_manifest_request_ids": sorted(expected_ids),
            "missing_recomputed_decision_request_ids": sorted(recomputed_ids),
            "scalar_vs_legacy_batch_drift": {"status": "not_available"},
            "per_context": {
                context_id: {
                    "status": "discovery_only_raw",
                    "expected_manifest_request_count": sum(
                        row["context_id"] == context_id for row in raw_manifest
                    ),
                    "scalar_overlay_request_count": 0,
                }
                for context_id in CONTEXT_IDS
            },
        }
    if not overlay_ids <= expected_ids:
        _fail("scalar overlay contains request IDs outside the original manifest")
    legacy_rows = scores.legacy_rows_by_request_id
    if legacy_rows is None:
        _fail("scalar overlay lacks preserved legacy rows for drift accounting")

    drift_rows: list[dict[str, Any]] = []
    for request_id in sorted(overlay_ids):
        scalar = score_value(scores.rows_by_request_id[request_id])
        legacy = score_value(legacy_rows[request_id])
        drift_rows.append(
            {
                "context_id": str(plan_request_context(request_id, scores.rows_by_request_id)),
                "request_id": request_id,
                "legacy_batch_score": legacy,
                "scalar_score": scalar,
                "scalar_minus_legacy_batch": scalar - legacy,
                "absolute_drift": abs(scalar - legacy),
            }
        )
    per_context: dict[str, Any] = {}
    for context_id in CONTEXT_IDS:
        expected_context = {
            str(row["request_id"]) for row in raw_manifest if row["context_id"] == context_id
        }
        recomputed_context = {
            str(row["request_id"])
            for row in recomputed_manifest
            if row["context_id"] == context_id
        }
        overlay_context = {
            request_id
            for request_id in overlay_ids
            if scores.rows_by_request_id[request_id]["context_id"] == context_id
        }
        original_complete = overlay_context == expected_context
        recomputed_closed = recomputed_context <= overlay_context
        if original_complete and recomputed_closed:
            status = "fully_scalar_confirmed_decision"
        elif overlay_context:
            status = "partial_scalar_overlay_bounded_decision"
        else:
            status = "discovery_only_raw"
        context_drifts = [
            row["scalar_minus_legacy_batch"]
            for row in drift_rows
            if row["context_id"] == context_id
        ]
        per_context[context_id] = {
            "status": status,
            "expected_manifest_request_count": len(expected_context),
            "scalar_overlay_request_count": len(overlay_context),
            "missing_original_manifest_request_ids": sorted(expected_context - overlay_context),
            "new_recomputed_decision_request_ids_without_scalar": sorted(
                recomputed_context - overlay_context
            ),
            "scalar_vs_legacy_batch_drift": _drift_summary(context_drifts),
        }
    fully_confirmed = all(
        row["status"] == "fully_scalar_confirmed_decision" for row in per_context.values()
    )
    status = (
        "fully_scalar_confirmed_decision"
        if fully_confirmed
        else "partial_scalar_overlay_bounded_decision"
    )
    all_drifts = [float(row["scalar_minus_legacy_batch"]) for row in drift_rows]
    return {
        "status": status,
        "decision_admission": status,
        "expected_manifest_request_count": len(expected_ids),
        "scalar_overlay_request_count": len(overlay_ids),
        "missing_original_manifest_request_ids": sorted(expected_ids - overlay_ids),
        "missing_recomputed_decision_request_ids": sorted(recomputed_ids - overlay_ids),
        "recomputed_decision_support_closed_under_scalar_overlay": recomputed_ids <= overlay_ids,
        "scalar_import_time_code_identity_sha256": (
            scores.scalar_overlay_lineage[0]["code"]["executed_source_sha256"]
        ),
        "scalar_overlay_lineage": list(scores.scalar_overlay_lineage),
        "scalar_vs_legacy_batch_drift": {
            **_drift_summary(all_drifts),
            "rows": drift_rows,
        },
        "per_context": per_context,
        "claim_boundary": (
            "fully_scalar_confirmed_decision refers only to numerical decision support; original raw "
            "artifacts remain bound lineage and runtime prefix parity remains separately unresolved"
        ),
    }


def promote_scalar_confirmed_decision_statuses(
    contexts: dict[str, dict[str, dict[str, Any]]],
    batch_scalar_admission: dict[str, Any],
    scalar_overlay_admission: Mapping[str, Any],
) -> None:
    per_context = _mapping(
        scalar_overlay_admission.get("per_context", {}),
        "scalar_overlay_admission.per_context",
    )

    def promote(confirmation: dict[str, Any]) -> None:
        legacy_batch_bound = dict(confirmation)
        legacy_status = str(confirmation["status"])
        confirmation["legacy_batch_bound"] = legacy_batch_bound
        confirmation["legacy_batch_bound_status"] = legacy_status
        confirmation["status"] = "scalar_confirmed_decision_support"

    for context_id in CONTEXT_IDS:
        overlay_context = _mapping(
            per_context.get(context_id, {}),
            f"scalar_overlay_admission.per_context.{context_id}",
        )
        if overlay_context.get("status") != "fully_scalar_confirmed_decision":
            continue
        context_admission = _mapping(
            batch_scalar_admission[context_id],
            f"batch_scalar_admission.{context_id}",
        )
        legacy_context_admission = dict(context_admission)
        context_admission["legacy_batch_bound"] = legacy_context_admission
        context_admission["legacy_batch_bound_status"] = context_admission["status"]
        context_admission["status"] = "scalar_confirmed_decision_support"
        context_admission["scalar_decision_support_error_bound"] = 0.0
        context_admission["unknown_needs_scalar_confirmation_owner_ids"] = []
        context_admission["scalar_decision_support_unknown_owner_ids"] = []
        for owner_id, cell in contexts[context_id].items():
            provenance = _mapping(
                cell.get("decision_score_provenance"),
                f"contexts.{context_id}.{owner_id}.decision_score_provenance",
            )
            if provenance.get("status") != "decision_support_scalar_overlaid":
                _fail(
                    "full scalar context contains legacy decision support",
                    context_id=context_id,
                    owner_id=owner_id,
                    provenance=provenance,
                )
            cell_confirmation = _mapping(
                cell.get("batch_scalar_confirmation"),
                f"contexts.{context_id}.{owner_id}.batch_scalar_confirmation",
            )
            promote(
                _mapping(
                    cell_confirmation.get("exact_anchor"),
                    f"contexts.{context_id}.{owner_id}.batch_scalar_confirmation.exact_anchor",
                )
            )
            promote(
                _mapping(
                    cell_confirmation.get("neighborhood_rank_and_margin"),
                    f"contexts.{context_id}.{owner_id}.batch_scalar_confirmation.neighborhood_rank_and_margin",
                )
            )
            promote(
                _mapping(
                    cell["ambiguity_bounds"].get("batch_scalar_confirmation"),
                    f"contexts.{context_id}.{owner_id}.ambiguity_bounds.batch_scalar_confirmation",
                )
            )
            promote(
                _mapping(
                    cell["peak"].get("batch_scalar_confirmation"),
                    f"contexts.{context_id}.{owner_id}.peak.batch_scalar_confirmation",
                )
            )


def promote_scalar_confirmed_selectivity(
    selectivity: dict[str, Any], scalar_overlay_admission: Mapping[str, Any]
) -> None:
    per_context = _mapping(
        scalar_overlay_admission.get("per_context", {}),
        "scalar_overlay_admission.per_context",
    )
    relevant_context_ids = ("self-due-gt17", "skip-post-gt17")
    if not all(
        _mapping(per_context.get(context_id, {}), context_id).get("status")
        == "fully_scalar_confirmed_decision"
        for context_id in relevant_context_ids
    ):
        return
    confirmation = selectivity.get("batch_scalar_confirmation")
    if not isinstance(confirmation, dict):
        return
    selectivity["batch_scalar_confirmation"] = {
        "D_error_bound": 0.0,
        "ambiguity_and_scalar_invariant_inequality_passed": selectivity[
            "ambiguity_invariant_inequality_passed"
        ],
        "ambiguity_and_scalar_invariant_inequality_failed": selectivity[
            "ambiguity_invariant_inequality_failed"
        ],
        "status": "scalar_confirmed_selectivity_decision_support",
        "legacy_batch_bound": confirmation,
    }


def plan_request_context(
    request_id: str, rows_by_request_id: Mapping[str, Mapping[str, Any]]
) -> str:
    return str(rows_by_request_id[request_id]["context_id"])


def _drift_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "max_absolute": None, "mean_signed": None, "median_signed": None}
    return {
        "count": len(values),
        "max_absolute": max(abs(value) for value in values),
        "mean_signed": statistics.fmean(values),
        "median_signed": statistics.median(values),
        "minimum_signed": min(values),
        "maximum_signed": max(values),
    }


def analyze(
    *,
    plan_dir: str | Path,
    score_shard_dirs: Sequence[str | Path] = (),
    native_greedy_jsonl: Sequence[str | Path] | None = None,
    input_mode: str = "strict_confirmation",
    independent_repeat_shard_dirs: Sequence[str | Path] | None = None,
    merged_scores_path: str | Path | None = None,
    merged_receipt_path: str | Path | None = None,
    run_attestation_path: str | Path | None = None,
    scalar_confirmation_shard_dirs: Sequence[str | Path] = (),
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    PlanBundle,
    ScoreBundle,
]:
    plan = load_plan(plan_dir)
    merged_args = (merged_scores_path, merged_receipt_path, run_attestation_path)
    if score_shard_dirs and any(value is not None for value in merged_args):
        _fail("choose either raw score shard dirs or the complete merged/attested input triple")
    if score_shard_dirs:
        scores = load_score_shards(plan, score_shard_dirs, input_mode=input_mode)
    elif all(value is not None for value in merged_args):
        scores = load_attested_merged_scores(
            plan,
            merged_scores_path=merged_scores_path,
            merged_receipt_path=merged_receipt_path,
            run_attestation_path=run_attestation_path,
            input_mode=input_mode,
        )
    else:
        _fail(
            "supply either one or more --score-shard-dir values or all of --merged-scores, "
            "--merged-receipt, and --run-attestation"
        )
    memoized_repeat_observation = compute_epsilon(plan, scores)
    if independent_repeat_shard_dirs:
        independent_repeat_evidence = validate_independent_repeat_shards(
            plan,
            independent_repeat_shard_dirs,
            input_mode=input_mode,
            expected_source_identity_sha256=scores.source_identity_sha256,
        )
        epsilon = float(independent_repeat_evidence["epsilon"])
        epsilon_receipt = {
            **memoized_repeat_observation,
            "epsilon": epsilon,
            "effective_source": "eight_independent_processes",
            "memoized_within_process_observation": memoized_repeat_observation,
            "independent_process_evidence": independent_repeat_evidence,
        }
    else:
        independent_repeat_evidence = None
        epsilon_receipt = {
            **memoized_repeat_observation,
            "effective_source": "single_process_repeat_rows_with_depth1_depth2_memo_reuse",
            "independent_process_evidence": None,
            "limitation": (
                "no --independent-repeat-shard-dir inputs were bound; exact independent-process "
                "reforward evidence remains a follow-up"
            ),
        }
        epsilon = float(epsilon_receipt["epsilon"])
    raw_scores = scores
    raw_contexts = build_context_statistics(plan, raw_scores, epsilon)
    attach_sidecar_gaps(plan, raw_scores, raw_contexts, epsilon)
    attach_score_overlay_provenance(plan, raw_scores, raw_contexts, epsilon)
    raw_batch_scalar_admission = attach_batch_scalar_confirmation(
        plan, raw_contexts, raw_scores, epsilon
    )
    raw_manifest = build_scalar_confirmation_manifest(
        plan, raw_scores, raw_contexts, epsilon
    )
    scores = apply_scalar_confirmation_overlay(
        plan,
        raw_scores,
        scalar_confirmation_shard_dirs,
        manifest=raw_manifest,
    )
    if scores.scalar_overlay_request_ids:
        contexts = build_context_statistics(plan, scores, epsilon)
        attach_sidecar_gaps(plan, scores, contexts, epsilon)
        attach_score_overlay_provenance(plan, scores, contexts, epsilon)
        batch_scalar_admission = attach_batch_scalar_confirmation(
            plan, contexts, scores, epsilon
        )
    else:
        contexts = raw_contexts
        batch_scalar_admission = raw_batch_scalar_admission
    scalar_confirmation_manifest = build_scalar_confirmation_manifest(
        plan, scores, contexts, epsilon
    )
    scalar_overlay_admission = build_scalar_overlay_admission(
        raw_manifest=raw_manifest,
        recomputed_manifest=scalar_confirmation_manifest,
        scores=scores,
    )
    promote_scalar_confirmed_decision_statuses(
        contexts,
        batch_scalar_admission,
        scalar_overlay_admission,
    )
    geometry = owner_geometry(plan)
    scans = build_scan_statistics(plan, contexts, epsilon)
    selectivity = compute_selectivity(
        plan,
        contexts,
        epsilon,
        score_error_bound_by_context=scores.batch_score_error_bound_by_context,
    )
    promote_scalar_confirmed_selectivity(selectivity, scalar_overlay_admission)
    calibration = calibration_summary(contexts, epsilon)
    native_greedy = load_native_greedy_join(native_greedy_jsonl)
    owner_rows = build_owner_rows(
        plan,
        contexts,
        geometry,
        selectivity,
        epsilon,
        native_greedy,
        scores.batch_score_error_bound_by_context,
        scalar_overlay_admission,
    )
    if len(owner_rows) != 41 or len({row["owner_id"] for row in owner_rows}) != 41:
        _fail("primary phenotype census did not produce exactly one row per owner")
    owner_rows_by_id = {str(row["owner_id"]): row for row in owner_rows}
    gt17_swap = owner_rows_by_id["gt:7511:17"]["shared_context_contrasts"][
        "same_length_owner_swap_skip_post_gt17_minus_self_due_gt22"
    ]
    gt17_lower = gt17_swap["ambiguity_lower_margin_delta"]
    gt17_batch_lower = gt17_swap["batch_scalar_confirmation"][
        "owner_relative_margin_delta_lower_bound"
    ]
    gt17_scalar_confirmed = (
        gt17_swap["batch_scalar_confirmation"]["status"] == "scalar_confirmed_delta"
    )
    if gt17_lower is None or gt17_batch_lower is None:
        gt17_coverage_compatible: bool | None = None
        gt17_coverage_status = "unresolved_missing_margin"
    elif gt17_batch_lower > 2.0 * epsilon:
        gt17_coverage_compatible = True
        gt17_coverage_status = (
            "descriptive_pattern_scalar_confirmed"
            if gt17_scalar_confirmed
            else "descriptive_pattern_confirmed_within_batch_scalar_bound"
        )
    elif (
        gt17_swap["batch_scalar_confirmation"][
            "owner_relative_margin_delta_upper_bound"
        ]
        <= 2.0 * epsilon
    ):
        gt17_coverage_compatible = False
        gt17_coverage_status = (
            "pattern_not_observed_scalar_confirmed"
            if gt17_scalar_confirmed
            else "pattern_not_observed_within_batch_scalar_bound"
        )
    else:
        gt17_coverage_compatible = None
        gt17_coverage_status = "unknown_needs_scalar_confirmation"
    global_batch_unresolved = any(
        value["status"] == "unresolved_needs_scalar_confirmation"
        for value in batch_scalar_admission.values()
    )
    if scalar_overlay_admission["decision_admission"] == "fully_scalar_confirmed_decision":
        claim_scope = "fully_scalar_confirmed_numerical_decision_with_raw_lineage"
    elif scalar_overlay_admission["decision_admission"] == "partial_scalar_overlay_bounded_decision":
        claim_scope = "partial_scalar_overlay_bounded_decision"
    else:
        claim_scope = (
            "descriptive_discovery_only"
            if input_mode == "raw_capture_salvage" or global_batch_unresolved
            else "frozen_descriptive_confirmation"
        )
    analysis = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "primary_product": {
            "name": OWNER_CENSUS_NAME,
            "row_schema_version": OWNER_ROW_SCHEMA_VERSION,
            "row_count": 41,
            "unit_of_analysis": "confirmed_physical_person_owner",
            "orientation": "continuous_data_driven_phenotype_census",
            "gt17_role": "positive_calibration_control_not_analysis_center",
        },
        "input_admission": {
            "mode": input_mode,
            "claim_scope": claim_scope,
            "raw_request_coverage": "exact_complete",
            "scorer_code_identity": (
                "provenance_race_preserved_not_uniform"
                if len(
                    {
                        str(entry.get("declared_code_sha256"))
                        for entry in scores.scorer_code_lineage
                        if entry.get("declared_code_sha256") is not None
                    }
                )
                != 1
                else "uniform"
            ),
            "scorer_code_lineage": list(scores.scorer_code_lineage),
            "batch_scalar_admission": (
                scalar_overlay_admission["decision_admission"]
                if scores.scalar_overlay_request_ids
                else "unresolved_needs_scalar_confirmation"
                if global_batch_unresolved
                else "confirmed_within_frozen_epsilon"
            ),
            "runtime_prefix_parity": "unresolved_cpu_not_attested",
            "runtime_prefix_parity_attestability": (
                "attestable only from a separate execution receipt that proves chosen-token parity; "
                "raw score coverage is not that receipt"
            ),
        },
        "numerical_tolerance": epsilon_receipt,
        "scalar_confirmation_overlay": scalar_overlay_admission,
        "batch_scalar_admission_by_context": batch_scalar_admission,
        "calibration_summary": calibration,
        "frozen_selectivity": selectivity,
        "same_length_owner_swap": {
            "left_context_id": "self-due-gt22",
            "right_context_id": "skip-post-gt17",
            "gt17": gt17_swap,
            "gt22": owner_rows_by_id["gt:7511:22"]["shared_context_contrasts"][
                "same_length_owner_swap_skip_post_gt17_minus_self_due_gt22"
            ],
            "gt17_coverage_compatible_owner_swap": gt17_coverage_compatible,
            "gt17_coverage_compatible_owner_swap_status": gt17_coverage_status,
            "claim_scope": "descriptive_only_owner_content_and_frontier_differ",
        },
        "scan_statistics": scans,
        "discovery_cohorts": {
            "status": "not_materialized",
            "policy": (
                "hard cohort labels are secondary, explicitly posthoc and discovery-only; they are "
                "not admission gates, causal truth, or substitutes for the continuous owner rows"
            ),
        },
        "posthoc_only_suggestions": [
            (
                "After scalar confirmation, explore owner phenotypes by clustering standardized continuous "
                "GT geometry, overlap/density, peak concentration, margin-delta, and sidecar-gap fields."
            ),
            (
                "Treat any cluster count, boundary, or name as posthoc discovery; report stability across "
                "scaling and clustering choices and never reuse it as an admission gate."
            ),
            (
                "Prioritize scalar rescoring of owners whose rank, margin sign, peak identity, sidecar gap, "
                "or selectivity status is marked unknown_needs_scalar_confirmation."
            ),
        ],
        "scalar_confirmation_manifest": {
            "name": SCALAR_CONFIRMATION_MANIFEST_NAME,
            "row_count": len(scalar_confirmation_manifest),
            "original_raw_manifest_row_count": len(raw_manifest),
            "ordering": "context_id_then_request_id",
            "deduplication": "one row per context/request_id with sorted selection_reasons",
            "minimum_surface": (
                "all 41 exact anchors per context, each owner bank winner, global rank/margin/tie and "
                "ambiguity-bound supports, plus every batch-bound near-threshold request"
            ),
        },
        "field_definitions": {
            "rank": "1 + count(other_score > owner_score + epsilon)",
            "tie": "abs(other_score - owner_score) <= epsilon",
            "margin": "owner neighborhood maximum minus best other confirmed-owner maximum",
            "context_delta_nonzero": "absolute delta > 2 * epsilon",
            "peak_concentration": (
                "softmax over unambiguous within-owner candidate log-likelihoods; reports top share, "
                "normalized entropy, effective candidate count, and top1-minus-top2 score"
            ),
            "neighbor_density": (
                "GT-center counts within one and two owner-specific GT diagonals; continuous geometry "
                "descriptors, not cohorts"
            ),
            "scan_distance": (
                "signed/absolute augmented-order distance after inserting the last complete donor person "
                "row into GT order sorted by (y1,x1,owner_id)"
            ),
        },
        "claim_boundaries": [
            (
                f"This analysis has claim_scope={claim_scope}. Raw-capture salvage preserves measurements "
                "for discovery but is not strict confirmation."
            ),
            (
                "All six contexts are shared prefix interventions. A context named self-due/post for one "
                "registered owner is not an owner-specific due/post intervention for every census row."
            ),
            (
                "The due-to-skip and same-length owner-swap contrasts jointly change prefix content and "
                "scan frontier and are descriptive, not causal coverage or route-state evidence."
            ),
            (
                "No prevalence, object-existence probability, negative localization/absence, natural "
                "final-set, training, or architecture claim is admitted."
            ),
            (
                "Sidecars diagnose finite-bank undercoverage only and never enter ranks, maxima, margins, "
                "selectivity, or ambiguity bounds."
            ),
        ],
        "receipt_limitations": [
            (
                "Scorer receipts enumerate row IDs but do not seal score-file bytes; this analyzer receipt "
                "therefore seals the exact consumed score-file hashes."
            ),
            (
                "Planner context parity is marked required_not_cpu_verifiable; this CPU analysis preserves "
                "that limitation and does not claim an independent teacher-forced parity attestation."
            ),
            *(
                []
                if independent_repeat_evidence is not None
                else [
                    (
                        "The eight plan repeat rows reused one process and depth-1/depth-2 memo entries; "
                        "supply eight --independent-repeat-shard-dir inputs to bind independent-process "
                        "epsilon evidence."
                    )
                ]
            ),
        ],
    }
    return analysis, owner_rows, scalar_confirmation_manifest, plan, scores


def write_outputs(
    output_dir: str | Path,
    *,
    analysis: Mapping[str, Any],
    owner_rows: Sequence[Mapping[str, Any]],
    scalar_confirmation_manifest: Sequence[Mapping[str, Any]],
    plan: PlanBundle,
    scores: ScoreBundle,
    native_greedy_jsonl: Sequence[str | Path] | None,
    independent_repeat_shard_dirs: Sequence[str | Path] | None,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    analysis_path = output / ANALYSIS_NAME
    census_path = output / OWNER_CENSUS_NAME
    scalar_manifest_path = output / SCALAR_CONFIRMATION_MANIFEST_NAME
    receipt_path = output / RECEIPT_NAME
    analysis_bytes = canonical_json_bytes(analysis) + b"\n"
    census_bytes = b"".join(canonical_json_bytes(row) + b"\n" for row in owner_rows)
    scalar_manifest_bytes = b"".join(
        canonical_json_bytes(row) + b"\n" for row in scalar_confirmation_manifest
    )
    analysis_status = _write_create_or_identical(analysis_path, analysis_bytes)
    census_status = _write_create_or_identical(census_path, census_bytes)
    scalar_manifest_status = _write_create_or_identical(
        scalar_manifest_path, scalar_manifest_bytes
    )
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "code": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "plan": {
            "receipt_path": str(plan.receipt_path),
            "receipt_sha256": sha256_file(plan.receipt_path),
            "receipt_content_sha256": plan.receipt["receipt_content_sha256"],
        },
        "score_inputs": [
            {
                "score_path": str(score_path),
                "score_sha256": sha256_file(score_path),
                "receipt_path": str(receipt_path_item),
                "receipt_sha256": sha256_file(receipt_path_item),
            }
            for score_path, receipt_path_item in zip(
                scores.score_paths, scores.receipt_paths, strict=True
            )
        ],
        "score_source_identity_sha256": scores.source_identity_sha256,
        "native_greedy_inputs": [
            {"path": str(Path(path).expanduser().resolve()), "sha256": sha256_file(Path(path).expanduser().resolve())}
            for path in native_greedy_jsonl or ()
        ],
        "independent_repeat_inputs": [
            {
                "directory": str(Path(path).expanduser().resolve()),
                "score_sha256": sha256_file(
                    Path(path).expanduser().resolve() / SCORE_NAME
                ),
                "receipt_sha256": sha256_file(
                    Path(path).expanduser().resolve() / SCORE_RECEIPT_NAME
                ),
            }
            for path in independent_repeat_shard_dirs or ()
        ],
        "scalar_confirmation_inputs": list(scores.scalar_overlay_lineage),
        "outputs": {
            ANALYSIS_NAME: {"sha256": sha256_file(analysis_path), "rows": 1},
            OWNER_CENSUS_NAME: {"sha256": sha256_file(census_path), "rows": len(owner_rows)},
            SCALAR_CONFIRMATION_MANIFEST_NAME: {
                "sha256": sha256_file(scalar_manifest_path),
                "rows": len(scalar_confirmation_manifest),
            },
        },
        "coverage": {
            "planned_request_count": len(plan.requests_by_id),
            "consumed_request_count": len(scores.rows_by_request_id),
            "exact_complete_coverage": set(plan.requests_by_id) == set(scores.rows_by_request_id),
            "owner_row_count": len(owner_rows),
            "scalar_overlay_request_count": len(scores.scalar_overlay_request_ids),
            "decision_admission": analysis["scalar_confirmation_overlay"][
                "decision_admission"
            ],
        },
    }
    receipt["receipt_content_sha256"] = sha256_json(receipt)
    receipt_status = _write_create_or_identical(
        receipt_path, canonical_json_bytes(receipt) + b"\n"
    )
    return {
        "analysis_path": str(analysis_path),
        "analysis_status": analysis_status,
        "owner_census_path": str(census_path),
        "owner_census_status": census_status,
        "scalar_confirmation_manifest_path": str(scalar_manifest_path),
        "scalar_confirmation_manifest_status": scalar_manifest_status,
        "scalar_confirmation_request_count": len(scalar_confirmation_manifest),
        "receipt_path": str(receipt_path),
        "receipt_status": receipt_status,
        "owner_count": len(owner_rows),
        "request_count": len(scores.rows_by_request_id),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument(
        "--score-shard-dir",
        type=Path,
        action="append",
        default=[],
        help=(
            "sealed scorer output directory; repeat for all six context shards and the scalar "
            "numerical-repeat shard (exact complete request coverage is mandatory)"
        ),
    )
    parser.add_argument("--merged-scores", type=Path, default=None)
    parser.add_argument("--merged-receipt", type=Path, default=None)
    parser.add_argument("--run-attestation", type=Path, default=None)
    parser.add_argument(
        "--scalar-confirmation-shard-dir",
        type=Path,
        action="append",
        default=[],
        help=(
            "optional scorer-receipt-v2 batch-size-one overlay directory; each row must be a unique "
            "subset of the analyzer-generated scalar confirmation manifest"
        ),
    )
    parser.add_argument(
        "--independent-repeat-shard-dir",
        type=Path,
        action="append",
        default=None,
        help=(
            "optional one-row scalar repeat process directory; when used, supply exactly eight "
            "dirs covering numerical repeat indices 0..7 (kept separate from exact plan coverage)"
        ),
    )
    parser.add_argument(
        "--native-greedy-jsonl",
        type=Path,
        action="append",
        default=None,
        help="optional normalized descriptive join keyed by context_id plus target/gt/owner_id",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--input-mode",
        choices=sorted(INPUT_MODES),
        default="strict_confirmation",
        help=(
            "strict_confirmation requires one live, uniform scorer source hash; "
            "raw_capture_salvage preserves a known capture-time code-hash race and limits claims to "
            "descriptive discovery"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    analysis, owner_rows, scalar_confirmation_manifest, plan, scores = analyze(
        plan_dir=args.plan_dir,
        score_shard_dirs=args.score_shard_dir,
        native_greedy_jsonl=args.native_greedy_jsonl,
        input_mode=args.input_mode,
        independent_repeat_shard_dirs=args.independent_repeat_shard_dir,
        merged_scores_path=args.merged_scores,
        merged_receipt_path=args.merged_receipt,
        run_attestation_path=args.run_attestation,
        scalar_confirmation_shard_dirs=args.scalar_confirmation_shard_dir,
    )
    result = write_outputs(
        args.output_dir,
        analysis=analysis,
        owner_rows=owner_rows,
        scalar_confirmation_manifest=scalar_confirmation_manifest,
        plan=plan,
        scores=scores,
        native_greedy_jsonl=args.native_greedy_jsonl,
        independent_repeat_shard_dirs=args.independent_repeat_shard_dir,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
