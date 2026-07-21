"""Strict data-parallel inference shard merge."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.common.errors import ArtifactContractError
from src.inference.artifacts import (
    IMAGE_PLAN_NAME,
    MANIFEST_NAME,
    PARSE_DIAGNOSTICS_NAME,
    PROVENANCE_NAME,
    RAW_NAME,
    SCORED_NAME,
    SUMMARY_NAME,
    TOKEN_TRACE_NAME,
    InferenceArtifactPaths,
    recompute_scores_from_artifacts,
    sha256_file,
    validate_scored_artifact_set,
    write_terminal_status_artifacts,
)
from src.inference.data_parallel import DataParallelPlan, RankShardPlan
from src.inference.scoring import PRED_SCORE_VERSION, SCORE_POLICY_FINGERPRINT


REQUIRED_MERGE_ARTIFACTS = (
    RAW_NAME,
    SCORED_NAME,
    PROVENANCE_NAME,
    TOKEN_TRACE_NAME,
    PARSE_DIAGNOSTICS_NAME,
    IMAGE_PLAN_NAME,
    SUMMARY_NAME,
    MANIFEST_NAME,
)
EVAL_OUTPUT_DIR_NAMES = ("eval_detection", "evaluation/detection")
EVAL_ARTIFACT_NAMES = (
    "metrics.json",
    "evaluation_receipt.json",
    "coco_gt.json",
    "coco_predictions.json",
)
SHARD_OWNED_IDENTITY_FIELDS = (
    "model_identity",
    "model_identity_fingerprint",
    "processor_identity",
    "processor_identity_fingerprint",
    "tokenizer_identity",
    "adapter_identity",
    "embedding_delta_identity",
    "backend_session",
    "likelihood_semantics",
    "execution_model_identity",
    "frontend_identity",
    "raw_model_logprob_status",
)
REQUIRED_NONEMPTY_SHARD_IDENTITY_OBJECTS = (
    "model_identity",
    "processor_identity",
    "tokenizer_identity",
    "backend_session",
    "likelihood_semantics",
    "frontend_identity",
    "generation_policy",
    "template_identity",
    "dataset_identity",
)
REQUIRED_NONEMPTY_SHARD_IDENTITY_STRINGS = (
    "model_identity_fingerprint",
    "processor_identity_fingerprint",
    "prompt_policy_fingerprint",
    "generation_config_fingerprint",
    "parser_policy",
    "score_policy_fingerprint",
    "backend",
    "backend_mode",
    "response_family",
    "raw_model_logprob_status",
)
REQUIRED_EXPLICIT_NULLABLE_SHARD_IDENTITY_FIELDS = (
    "adapter_identity",
    "embedding_delta_identity",
    "execution_model_identity",
)


@dataclass(frozen=True)
class ShardEvidence:
    rank: int
    shard_dir: Path
    artifact_hashes: dict[str, str]
    row_ids: list[str]
    row_indices: list[int]
    worker_status: str

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "shard_dir": self.shard_dir.as_posix(),
            "artifact_hashes": dict(sorted(self.artifact_hashes.items())),
            "row_ids": list(self.row_ids),
            "row_indices": list(self.row_indices),
            "worker_status": self.worker_status,
        }


def merge_shard_artifacts(
    *,
    output_dir: Path,
    shard_dirs: tuple[Path, ...] | list[Path],
    expected_row_ids: tuple[str, ...] | list[str],
    metadata: dict[str, Any],
    plan: DataParallelPlan,
    worker_statuses: dict[int, str] | None = None,
) -> InferenceArtifactPaths:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        return _merge_shard_artifacts(
            output_dir=output_dir,
            shard_dirs=tuple(Path(path) for path in shard_dirs),
            expected_row_ids=tuple(str(row_id) for row_id in expected_row_ids),
            metadata=dict(metadata),
            plan=plan,
            worker_statuses=dict(worker_statuses or {}),
        )
    except ArtifactContractError as exc:
        _write_merge_failure_terminal_status(
            output_dir=output_dir,
            metadata=metadata,
            error=exc,
        )
        raise


def merge_identity_vector(
    *,
    manifest: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Return the shard identity fields that must agree before merge."""

    return {
        "model_identity": manifest.get("model_identity")
        or provenance.get("model_identity"),
        "model_identity_fingerprint": manifest.get("model_identity_fingerprint")
        or provenance.get("model_identity_fingerprint"),
        "processor_identity": manifest.get("processor_identity")
        or provenance.get("processor_identity"),
        "processor_identity_fingerprint": manifest.get("processor_identity_fingerprint")
        or provenance.get("processor_identity_fingerprint"),
        "tokenizer_identity": manifest.get("tokenizer_identity"),
        "adapter_identity": manifest.get("adapter_identity"),
        "embedding_delta_identity": manifest.get("embedding_delta_identity"),
        "backend_session": _backend_session_semantic_identity(
            manifest.get("backend_session") or provenance.get("backend_session")
        ),
        "likelihood_semantics": manifest.get("likelihood_semantics")
        or provenance.get("likelihood_semantics"),
        "execution_model_identity": (
            manifest.get("execution_model_identity")
            if "execution_model_identity" in manifest
            else provenance.get("execution_model_identity")
        ),
        "frontend_identity": manifest.get("frontend_identity")
        or provenance.get("frontend_identity"),
        "raw_model_logprob_status": manifest.get("raw_model_logprob_status")
        or provenance.get("raw_model_logprob_status"),
        "template_identity": manifest.get("template_identity")
        or provenance.get("template_identity"),
        "prompt_policy_fingerprint": manifest.get("prompt_policy_fingerprint")
        or provenance.get("prompt_policy_fingerprint"),
        "generation_config_fingerprint": manifest.get("generation_config_fingerprint")
        or provenance.get("generation_config_fingerprint"),
        "generation_policy": manifest.get("generation_policy")
        or provenance.get("generation_policy"),
        "dataset_identity": manifest.get("dataset_identity"),
        "parser_policy": manifest.get("parser_policy") or provenance.get("parser_policy"),
        "score_policy_fingerprint": manifest.get("score_policy_fingerprint")
        or provenance.get("score_policy_fingerprint"),
        "backend": manifest.get("backend"),
        "backend_mode": manifest.get("backend_mode"),
        "response_family": manifest.get("response_family"),
    }


_DUPLICATED_SEMANTIC_SURFACE_FIELDS = (
    "adapter_identity",
    "embedding_delta_identity",
    "execution_model_identity",
    "frontend_identity",
    "generation_config_fingerprint",
    "generation_policy",
    "likelihood_semantics",
    "media_identity",
    "model_identity",
    "model_identity_fingerprint",
    "parallelism",
    "parser_policy",
    "processor_identity",
    "processor_identity_fingerprint",
    "prompt_policy_fingerprint",
    "raw_model_logprob_status",
    "score_policy_fingerprint",
    "template_identity",
    "tokenizer_identity",
)


def fingerprint_merge_identity_vector(vector: dict[str, Any]) -> str:
    return _sha256_json(vector)


def _merge_shard_artifacts(
    *,
    output_dir: Path,
    shard_dirs: tuple[Path, ...],
    expected_row_ids: tuple[str, ...],
    metadata: dict[str, Any],
    plan: DataParallelPlan,
    worker_statuses: dict[int, str],
) -> InferenceArtifactPaths:
    if not shard_dirs:
        raise ArtifactContractError(
            "merge requires at least one shard directory",
            code="merge.no_shards",
        )
    raw_rows: list[dict[str, Any]] = []
    scored_rows: list[dict[str, Any]] = []
    token_trace_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    image_plan_rows: list[dict[str, Any]] = []
    prompt_trace_rows: list[dict[str, Any]] = []
    raw_replay_trace_rows: list[dict[str, Any]] = []
    backend_sessions: list[tuple[int, dict[str, Any]]] = []
    shard_evidence: list[ShardEvidence] = []
    identity: dict[str, Any] | None = None
    expected_ranks = {rank_plan.rank for rank_plan in plan.ranks}
    seen_ranks: dict[int, Path] = {}

    for shard_dir in shard_dirs:
        shard = _load_validated_shard(
            shard_dir=shard_dir,
            plan=plan,
            worker_statuses=worker_statuses,
        )
        rank = _require_artifact_int(
            shard["rank"],
            artifact_name=MANIFEST_NAME,
            field="parallelism.worker.rank",
        )
        if rank not in expected_ranks:
            raise ArtifactContractError(
                "rank-local shard rank is not present in the controller plan",
                code="merge.unknown_rank",
                context={
                    "rank": rank,
                    "expected_ranks": sorted(expected_ranks),
                    "shard_dir": str(shard_dir),
                },
            )
        if rank in seen_ranks:
            raise ArtifactContractError(
                "multiple shard directories claim the same rank",
                code="merge.duplicate_rank",
                context={
                    "rank": rank,
                    "first_shard_dir": str(seen_ranks[rank]),
                    "duplicate_shard_dir": str(shard_dir),
                },
            )
        seen_ranks[rank] = shard_dir
        if identity is None:
            identity = shard["identity"]
        else:
            _require_identity_match(expected=identity, observed=shard["identity"])
        shard_raw_rows = shard["raw_rows"]
        raw_rows.extend(shard_raw_rows)
        scored_rows.extend(shard["scored_rows"])
        token_trace_rows.extend(shard["token_trace_rows"])
        diagnostic_rows.extend(shard["diagnostic_rows"])
        image_plan_rows.extend(shard["image_plan_rows"])
        prompt_trace_rows.extend(shard["prompt_trace"])
        raw_replay_trace_rows.extend(shard["raw_replay_trace"])
        backend_sessions.append((rank, shard["backend_session"]))
        shard_evidence.append(
            ShardEvidence(
                rank=rank,
                shard_dir=shard_dir,
                artifact_hashes=shard["artifact_hashes"],
                row_ids=[
                    _require_artifact_str(
                        row.get("row_id"),
                        artifact_name=RAW_NAME,
                        field="row_id",
                    )
                    for row in shard_raw_rows
                ],
                row_indices=[
                    _require_artifact_int(
                        row.get("row_index"),
                        artifact_name=RAW_NAME,
                        field="row_index",
                        row_id=str(row.get("row_id"))
                        if row.get("row_id") is not None
                        else None,
                    )
                    for row in shard_raw_rows
                ],
                worker_status=str(shard["worker_status"]),
            )
        )

    missing_ranks = sorted(expected_ranks - set(seen_ranks))
    if missing_ranks:
        raise ArtifactContractError(
            "controller plan ranks are missing shard artifacts",
            code="merge.missing_rank",
            context={"missing_ranks": missing_ranks},
        )
    if identity is not None:
        metadata = _metadata_with_shard_owned_identity(
            metadata=metadata,
            shard_identity=identity,
        )
        _require_controller_identity_match(
            shard_identity=identity,
            controller_identity=_controller_identity_vector(metadata),
        )

    expected_index_by_row_id = {row_id: index for index, row_id in enumerate(expected_row_ids)}
    raw_rows = _validate_and_sort_rows(
        rows=raw_rows,
        expected_row_ids=expected_row_ids,
        expected_index_by_row_id=expected_index_by_row_id,
        artifact_name=RAW_NAME,
    )
    scored_rows = _validate_and_sort_rows(
        rows=scored_rows,
        expected_row_ids=expected_row_ids,
        expected_index_by_row_id=expected_index_by_row_id,
        artifact_name=SCORED_NAME,
    )
    image_plan_rows = _validate_and_sort_rows(
        rows=image_plan_rows,
        expected_row_ids=expected_row_ids,
        expected_index_by_row_id=expected_index_by_row_id,
        artifact_name=IMAGE_PLAN_NAME,
    )
    _validate_raw_scored_image_identity(
        raw_rows=raw_rows,
        scored_rows=scored_rows,
        image_plan_rows=image_plan_rows,
    )
    _validate_trace_uniqueness(
        token_trace_rows,
        raw_model_logprob_status=metadata.get("raw_model_logprob_status"),
    )
    _validate_row_local_score_provenance(
        scored_rows,
        token_trace_rows=token_trace_rows,
    )
    token_trace_rows = _sort_sidecar_rows(
        token_trace_rows,
        expected_index_by_row_id=expected_index_by_row_id,
        artifact_name=TOKEN_TRACE_NAME,
    )
    diagnostic_rows = _sort_sidecar_rows(
        diagnostic_rows,
        expected_index_by_row_id=expected_index_by_row_id,
        artifact_name=PARSE_DIAGNOSTICS_NAME,
    )
    prompt_trace_rows = _validate_and_sort_runtime_trace(
        rows=prompt_trace_rows,
        expected_row_ids=expected_row_ids,
        expected_index_by_row_id=expected_index_by_row_id,
        trace_name="prompt_trace",
        required=True,
    )
    _validate_prompt_trace(prompt_trace_rows)
    raw_replay_required = (
        metadata.get("backend") == "vllm"
        and metadata.get("raw_model_logprob_status") == "available"
    )
    raw_replay_trace_rows = _validate_and_sort_runtime_trace(
        rows=raw_replay_trace_rows,
        expected_row_ids=expected_row_ids,
        expected_index_by_row_id=expected_index_by_row_id,
        trace_name="raw_replay_trace",
        required=raw_replay_required,
    )
    _validate_raw_replay_trace(raw_replay_trace_rows)
    metadata["prompt_trace"] = prompt_trace_rows
    metadata["raw_replay_trace"] = raw_replay_trace_rows
    metadata["backend_session"] = _merge_backend_sessions(
        semantic_session=dict(metadata.get("backend_session") or {}),
        ranked_shard_sessions=backend_sessions,
        raw_replay_trace=raw_replay_trace_rows,
        raw_replay_required=raw_replay_required,
    )

    paths = _paths(output_dir)
    staging_dir = Path(tempfile.mkdtemp(prefix=".merge-artifacts-", dir=output_dir))
    staged = _paths(staging_dir)
    try:
        _write_jsonl(staged.raw_jsonl, raw_rows)
        _write_jsonl(staged.scored_jsonl, scored_rows)
        _write_jsonl(staged.token_trace_jsonl, token_trace_rows)
        _write_jsonl(staged.parse_diagnostics_jsonl, diagnostic_rows)
        _write_jsonl(staged.image_plan_jsonl, image_plan_rows)

        raw_sha = sha256_file(staged.raw_jsonl)
        scored_sha = sha256_file(staged.scored_jsonl)
        merged_hashes = {
            RAW_NAME: raw_sha,
            SCORED_NAME: scored_sha,
            TOKEN_TRACE_NAME: sha256_file(staged.token_trace_jsonl),
            PARSE_DIAGNOSTICS_NAME: sha256_file(staged.parse_diagnostics_jsonl),
            IMAGE_PLAN_NAME: sha256_file(staged.image_plan_jsonl),
        }
        parallelism = _merged_parallelism(
            metadata=metadata,
            plan=plan,
            shard_evidence=shard_evidence,
            expected_row_ids=expected_row_ids,
            merged_hashes=merged_hashes,
        )
        provenance = _merged_provenance(
            metadata=metadata,
            raw_sha=raw_sha,
            scored_sha=scored_sha,
            expected_row_ids=expected_row_ids,
            parallelism=parallelism,
        )
        _write_json(staged.provenance_json, provenance)
        summary = _merged_summary(
            raw_rows=raw_rows,
            scored_rows=scored_rows,
            token_trace_rows=token_trace_rows,
            diagnostic_rows=diagnostic_rows,
            metadata=metadata,
        )
        _write_json(staged.summary_json, summary)
        manifest = _merged_manifest(
            metadata=metadata,
            summary=summary,
            parallelism=parallelism,
            identity_fingerprint=fingerprint_merge_identity_vector(identity or {}),
        )
        _write_json(staged.run_manifest_json, manifest)
        validate_scored_artifact_set(staging_dir)
        _validate_replay_scores(
            scored_jsonl=staged.scored_jsonl,
            token_trace_jsonl=staged.token_trace_jsonl,
        )
        _publish_staged_artifacts(staged=staged, final=paths)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
    validate_scored_artifact_set(output_dir)
    return paths


def _load_validated_shard(
    *,
    shard_dir: Path,
    plan: DataParallelPlan,
    worker_statuses: dict[int, str],
) -> dict[str, Any]:
    _require_shard_artifacts(shard_dir)
    manifest = _read_json(shard_dir / MANIFEST_NAME)
    provenance = _read_json(shard_dir / PROVENANCE_NAME)
    rank = _rank_from_manifest_or_path(manifest=manifest, shard_dir=shard_dir)
    summary_status = _summary_status(shard_dir)
    worker_status = str(worker_statuses.get(rank, summary_status))
    if summary_status != "completed":
        if worker_status != summary_status:
            raise ArtifactContractError(
                "worker receipt disagrees with failed shard summary",
                code="merge.worker_status_mismatch",
                context={
                    "rank": rank,
                    "summary_status": summary_status,
                    "worker_status": worker_status,
                },
            )
        raise ArtifactContractError(
            "cannot merge artifacts from a failed worker",
            code="merge.worker_failed",
            context={"rank": rank, "worker_status": summary_status},
        )
    if worker_status != summary_status:
        raise ArtifactContractError(
            "worker receipt disagrees with shard summary",
            code="merge.worker_status_mismatch",
            context={
                "rank": rank,
                "summary_status": summary_status,
                "worker_status": worker_status,
            },
        )
    if worker_status != "completed":
        raise ArtifactContractError(
            "cannot merge artifacts from a failed worker",
            code="merge.worker_failed",
            context={"rank": rank, "worker_status": worker_status},
        )
    _require_shard_plan_fingerprint(manifest=manifest, plan=plan, rank=rank)
    try:
        validate_scored_artifact_set(shard_dir)
    except ArtifactContractError as exc:
        raise ArtifactContractError(
            "rank-local shard artifact set is invalid",
            code="merge.invalid_shard_artifact_set",
            context={"shard_dir": str(shard_dir), "rank": rank, "cause_code": exc.code},
            cause=exc,
        ) from exc
    raw_rows = _read_jsonl(shard_dir / RAW_NAME)
    scored_rows = _read_jsonl(shard_dir / SCORED_NAME)
    token_trace_rows = _read_jsonl(shard_dir / TOKEN_TRACE_NAME)
    diagnostic_rows = _read_jsonl(shard_dir / PARSE_DIAGNOSTICS_NAME)
    image_plan_rows = _read_jsonl(shard_dir / IMAGE_PLAN_NAME)
    _validate_rank_assignment(
        manifest=manifest,
        plan=plan,
        rank=rank,
        raw_rows=raw_rows,
        scored_rows=scored_rows,
        image_plan_rows=image_plan_rows,
        token_trace_rows=token_trace_rows,
        diagnostic_rows=diagnostic_rows,
    )
    worker_metadata = _worker_metadata_from_manifest(
        manifest=manifest,
        rank=rank,
    )
    _validate_worker_metadata_matches_rank_plan(
        worker_metadata=worker_metadata,
        rank_plan=_rank_plan_for(plan=plan, rank=rank),
    )
    _require_matching_semantic_surfaces(
        manifest=manifest,
        provenance=provenance,
        rank=rank,
    )
    backend_session = _matching_backend_session_surface(
        manifest=manifest,
        provenance=provenance,
        rank=rank,
    )
    identity = merge_identity_vector(manifest=manifest, provenance=provenance)
    _require_complete_shard_identity(
        manifest=manifest,
        provenance=provenance,
        identity=identity,
        rank=rank,
    )
    prompt_trace = _matching_runtime_trace_surface(
        manifest=manifest,
        provenance=provenance,
        field="prompt_trace",
        rank=rank,
    )
    raw_replay_trace = _matching_runtime_trace_surface(
        manifest=manifest,
        provenance=provenance,
        field="raw_replay_trace",
        rank=rank,
    )
    rank_plan = _rank_plan_for(plan=plan, rank=rank)
    _require_runtime_trace_assignment(
        rows=prompt_trace,
        expected_row_ids=rank_plan.row_ids,
        field="prompt_trace",
        rank=rank,
        required=True,
    )
    raw_replay_required = (
        identity.get("backend") == "vllm"
        and identity.get("raw_model_logprob_status") == "available"
    )
    _require_runtime_trace_assignment(
        rows=raw_replay_trace,
        expected_row_ids=rank_plan.row_ids,
        field="raw_replay_trace",
        rank=rank,
        required=raw_replay_required,
    )
    _validate_shard_raw_replay_receipt(
        backend_session=backend_session,
        raw_replay_trace=raw_replay_trace,
        required=raw_replay_required,
        rank=rank,
    )
    return {
        "rank": rank,
        "worker_status": worker_status,
        "manifest": manifest,
        "provenance": provenance,
        "identity": identity,
        "artifact_hashes": {
            name: sha256_file(shard_dir / name) for name in REQUIRED_MERGE_ARTIFACTS
        },
        "raw_rows": raw_rows,
        "scored_rows": scored_rows,
        "token_trace_rows": _with_worker_metadata(
            token_trace_rows,
            worker_metadata=worker_metadata,
        ),
        "diagnostic_rows": _with_worker_metadata(
            diagnostic_rows,
            worker_metadata=worker_metadata,
        ),
        "image_plan_rows": image_plan_rows,
        "prompt_trace": prompt_trace,
        "raw_replay_trace": raw_replay_trace,
        "backend_session": backend_session,
    }


def _require_matching_semantic_surfaces(
    *,
    manifest: dict[str, Any],
    provenance: dict[str, Any],
    rank: int,
) -> None:
    missing = [
        field
        for field in _DUPLICATED_SEMANTIC_SURFACE_FIELDS
        if field not in manifest or field not in provenance
    ]
    if missing:
        raise ArtifactContractError(
            "rank-local semantic identity must exist in manifest and provenance",
            code="merge.semantic_surface_missing",
            context={"rank": rank, "fields": missing},
        )
    mismatched = [
        field
        for field in _DUPLICATED_SEMANTIC_SURFACE_FIELDS
        if manifest[field] != provenance[field]
    ]
    if mismatched:
        raise ArtifactContractError(
            "rank-local semantic identity disagrees between manifest and provenance",
            code="merge.semantic_surface_mismatch",
            context={"rank": rank, "fields": mismatched},
        )


def _require_complete_shard_identity(
    *,
    manifest: dict[str, Any],
    provenance: dict[str, Any],
    identity: dict[str, Any],
    rank: int,
) -> None:
    missing: list[str] = []
    invalid: list[str] = []
    for field in REQUIRED_NONEMPTY_SHARD_IDENTITY_OBJECTS:
        value = identity.get(field)
        if not isinstance(value, dict) or not value:
            missing.append(field)
    for field in REQUIRED_NONEMPTY_SHARD_IDENTITY_STRINGS:
        value = identity.get(field)
        if not isinstance(value, str) or not value.strip():
            missing.append(field)
    for field in REQUIRED_EXPLICIT_NULLABLE_SHARD_IDENTITY_FIELDS:
        if field not in manifest and field not in provenance:
            missing.append(field)

    backend_session = identity.get("backend_session")
    if isinstance(backend_session, dict) and backend_session:
        if backend_session.get("backend") != identity.get("backend"):
            invalid.append("backend_session.backend")
        if backend_session.get("backend_mode") != identity.get("backend_mode"):
            invalid.append("backend_session.backend_mode")
        if backend_session.get("response_family") != identity.get("response_family"):
            invalid.append("backend_session.response_family")
    likelihood = identity.get("likelihood_semantics")
    if isinstance(likelihood, dict) and likelihood:
        if likelihood.get("score_owned_channel") != "policy_logprob":
            invalid.append("likelihood_semantics.score_owned_channel")
        for field in ("policy", "raw"):
            value = likelihood.get(field)
            if not isinstance(value, str) or not value.strip():
                invalid.append(f"likelihood_semantics.{field}")
    raw_status = identity.get("raw_model_logprob_status")
    if raw_status not in {"disabled", "available"}:
        invalid.append("raw_model_logprob_status")
    if identity.get("backend") == "vllm":
        execution_identity = identity.get("execution_model_identity")
        if not isinstance(execution_identity, dict) or not execution_identity:
            invalid.append("execution_model_identity")

    if missing or invalid:
        raise ArtifactContractError(
            "rank-local shard is missing mandatory semantic identity evidence",
            code="merge.semantic_identity_incomplete",
            context={
                "rank": rank,
                "missing_fields": sorted(set(missing)),
                "invalid_fields": sorted(set(invalid)),
            },
        )


def _backend_session_semantic_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    session = _json_safe(value)
    effective_settings = session.get("effective_settings")
    if isinstance(effective_settings, dict):
        effective_settings.pop("performance", None)
        runtime_preflight = effective_settings.get("runtime_preflight")
        if isinstance(runtime_preflight, dict):
            runtime_preflight = dict(runtime_preflight)
            runtime_preflight.pop("live_decode", None)
            runtime_preflight.pop("cleanup", None)
            runtime_preflight.pop("process", None)
            runtime_preflight.pop("cuda", None)
            runtime_preflight.pop("rank_evidence", None)
            if str(runtime_preflight.get("status", "")).startswith("passed_"):
                runtime_preflight["status"] = "ready_for_engine_construction"
            effective_settings["runtime_preflight"] = runtime_preflight
        raw_replay = effective_settings.get("raw_replay")
        if isinstance(raw_replay, dict):
            raw_replay = dict(raw_replay)
            raw_replay.pop("request_count", None)
            raw_replay.pop("row_evidence_sha256", None)
            effective_settings["raw_replay"] = raw_replay
    return session


def _matching_runtime_trace_surface(
    *,
    manifest: dict[str, Any],
    provenance: dict[str, Any],
    field: str,
    rank: int,
) -> list[dict[str, Any]]:
    manifest_value = manifest.get(field, [])
    provenance_value = provenance.get(field, [])
    if manifest_value != provenance_value:
        raise ArtifactContractError(
            "rank-local runtime trace disagrees between manifest and provenance",
            code="merge.runtime_trace_surface_mismatch",
            context={"rank": rank, "field": field},
        )
    if not isinstance(manifest_value, list):
        raise ArtifactContractError(
            "rank-local runtime trace must be a JSON list",
            code="merge.invalid_runtime_trace",
            context={"rank": rank, "field": field},
        )
    rows: list[dict[str, Any]] = []
    for position, row in enumerate(manifest_value):
        if not isinstance(row, dict):
            raise ArtifactContractError(
                "rank-local runtime trace rows must be JSON objects",
                code="merge.invalid_runtime_trace",
                context={"rank": rank, "field": field, "row_position": position},
            )
        rows.append(dict(row))
    return rows


def _matching_backend_session_surface(
    *,
    manifest: dict[str, Any],
    provenance: dict[str, Any],
    rank: int,
) -> dict[str, Any]:
    manifest_value = manifest.get("backend_session")
    provenance_value = provenance.get("backend_session")
    if (
        not isinstance(manifest_value, dict)
        or not manifest_value
        or not isinstance(provenance_value, dict)
        or not provenance_value
    ):
        raise ArtifactContractError(
            "rank-local backend session must be present in manifest and provenance",
            code="merge.backend_session_surface_missing",
            context={"rank": rank},
        )
    if manifest_value != provenance_value:
        raise ArtifactContractError(
            "rank-local backend session disagrees between manifest and provenance",
            code="merge.backend_session_surface_mismatch",
            context={"rank": rank},
        )
    return dict(manifest_value)


def _require_runtime_trace_assignment(
    *,
    rows: list[dict[str, Any]],
    expected_row_ids: tuple[str, ...],
    field: str,
    rank: int,
    required: bool,
) -> None:
    if not rows and not required:
        return
    observed = [
        _require_artifact_str(
            row.get("row_id"),
            artifact_name=MANIFEST_NAME,
            field=f"{field}[{position}].row_id",
            row_position=position,
        )
        for position, row in enumerate(rows)
    ]
    if observed != list(expected_row_ids):
        raise ArtifactContractError(
            "rank-local runtime trace does not exactly cover its assigned rows",
            code="merge.runtime_trace_assignment_mismatch",
            context={
                "rank": rank,
                "field": field,
                "expected_row_ids": list(expected_row_ids),
                "observed_row_ids": observed,
            },
        )


def _validate_shard_raw_replay_receipt(
    *,
    backend_session: dict[str, Any],
    raw_replay_trace: list[dict[str, Any]],
    required: bool,
    rank: int,
) -> None:
    effective_settings = backend_session.get("effective_settings")
    raw_replay = (
        effective_settings.get("raw_replay")
        if isinstance(effective_settings, dict)
        else None
    )
    if not required:
        if raw_replay_trace:
            raise ArtifactContractError(
                "raw replay evidence is present when the backend contract does not require it",
                code="merge.unexpected_raw_replay_trace",
                context={"rank": rank},
            )
        return
    if not isinstance(raw_replay, dict) or raw_replay.get("status") != "completed":
        raise ArtifactContractError(
            "vLLM raw likelihood is missing its completed replay receipt",
            code="merge.raw_replay_receipt_missing",
            context={"rank": rank},
        )
    if raw_replay.get("request_count") != len(raw_replay_trace):
        raise ArtifactContractError(
            "vLLM raw replay request count disagrees with row evidence",
            code="merge.raw_replay_receipt_mismatch",
            context={
                "rank": rank,
                "expected": len(raw_replay_trace),
                "observed": raw_replay.get("request_count"),
            },
        )
    expected_hash = _sha256_json(_raw_replay_trace_mapping(raw_replay_trace))
    if raw_replay.get("row_evidence_sha256") != expected_hash:
        raise ArtifactContractError(
            "vLLM raw replay receipt hash disagrees with row evidence",
            code="merge.raw_replay_receipt_mismatch",
            context={
                "rank": rank,
                "expected": expected_hash,
                "observed": raw_replay.get("row_evidence_sha256"),
            },
        )


def _validate_and_sort_runtime_trace(
    *,
    rows: list[dict[str, Any]],
    expected_row_ids: tuple[str, ...],
    expected_index_by_row_id: dict[str, int],
    trace_name: str,
    required: bool,
) -> list[dict[str, Any]]:
    if not rows and not required:
        return []
    observed: dict[str, dict[str, Any]] = {}
    for position, row in enumerate(rows):
        row_id = _require_artifact_str(
            row.get("row_id"),
            artifact_name=MANIFEST_NAME,
            field=f"{trace_name}[{position}].row_id",
            row_position=position,
        )
        if row_id in observed:
            raise ArtifactContractError(
                "merged runtime trace contains duplicate row evidence",
                code="merge.duplicate_runtime_trace",
                context={"field": trace_name, "row_id": row_id},
            )
        if row_id not in expected_index_by_row_id:
            raise ArtifactContractError(
                "merged runtime trace contains an unknown row id",
                code="merge.unknown_runtime_trace_row",
                context={"field": trace_name, "row_id": row_id},
            )
        observed[row_id] = row
    missing = [row_id for row_id in expected_row_ids if row_id not in observed]
    if missing:
        raise ArtifactContractError(
            "merged runtime trace is missing row evidence",
            code="merge.missing_runtime_trace_row",
            context={"field": trace_name, "missing_row_ids": missing},
        )
    return [observed[row_id] for row_id in expected_row_ids]


def _validate_prompt_trace(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row_id = str(row["row_id"])
        if row.get("prompt_token_parity") != "verified":
            raise ArtifactContractError(
                "prompt trace is not parity verified",
                code="merge.prompt_trace_mismatch",
                context={"row_id": row_id},
            )
        for prefix in ("input", "expected_executed", "backend_executed"):
            count = row.get(f"{prefix}_prompt_token_count")
            if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                raise ArtifactContractError(
                    "prompt trace token count must be a positive integer",
                    code="merge.invalid_prompt_trace",
                    context={"row_id": row_id, "field": f"{prefix}_prompt_token_count"},
                )
            _require_sha256(
                row.get(f"{prefix}_prompt_token_ids_sha256"),
                code="merge.invalid_prompt_trace",
                context={"row_id": row_id, "field": f"{prefix}_prompt_token_ids_sha256"},
            )
        if (
            row["expected_executed_prompt_token_count"]
            != row["backend_executed_prompt_token_count"]
            or row["expected_executed_prompt_token_ids_sha256"]
            != row["backend_executed_prompt_token_ids_sha256"]
        ):
            raise ArtifactContractError(
                "prompt trace expected and backend-executed identities disagree",
                code="merge.prompt_trace_mismatch",
                context={"row_id": row_id},
            )


def _validate_raw_replay_trace(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row_id = str(row["row_id"])
        if row.get("status") != "verified":
            raise ArtifactContractError(
                "raw replay trace is not verified",
                code="merge.raw_replay_trace_mismatch",
                context={"row_id": row_id},
            )
        for prefix in ("prompt", "generated"):
            count = row.get(f"{prefix}_token_count")
            if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                raise ArtifactContractError(
                    "raw replay token count must be a positive integer",
                    code="merge.invalid_raw_replay_trace",
                    context={"row_id": row_id, "field": f"{prefix}_token_count"},
                )
            _require_sha256(
                row.get(f"{prefix}_token_ids_sha256"),
                code="merge.invalid_raw_replay_trace",
                context={"row_id": row_id, "field": f"{prefix}_token_ids_sha256"},
            )
        finish_reason = row.get("finish_reason")
        if not isinstance(finish_reason, str) or not finish_reason:
            raise ArtifactContractError(
                "raw replay finish reason must be a non-empty string",
                code="merge.invalid_raw_replay_trace",
                context={"row_id": row_id, "field": "finish_reason"},
            )
        native_stop_reason = row.get("native_stop_reason")
        if native_stop_reason is not None and not isinstance(native_stop_reason, str | int):
            raise ArtifactContractError(
                "raw replay native stop reason has an unsupported type",
                code="merge.invalid_raw_replay_trace",
                context={"row_id": row_id, "field": "native_stop_reason"},
            )


def _require_sha256(value: Any, *, code: str, context: dict[str, Any]) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ArtifactContractError(
            "runtime trace hash must be a lowercase SHA-256 digest",
            code=code,
            context=context,
        )


def _merge_backend_sessions(
    *,
    semantic_session: dict[str, Any],
    ranked_shard_sessions: list[tuple[int, dict[str, Any]]],
    raw_replay_trace: list[dict[str, Any]],
    raw_replay_required: bool,
) -> dict[str, Any]:
    shard_sessions = [session for _, session in ranked_shard_sessions]
    for _, session in ranked_shard_sessions:
        observed = _backend_session_semantic_identity(session)
        if observed != semantic_session:
            raise ArtifactContractError(
                "rank-local backend session disagrees with merged semantic identity",
                code="merge.identity_mismatch",
                context={"field": "backend_session", "expected": semantic_session, "observed": observed},
            )
    merged = _json_safe(semantic_session)
    performance = _merge_decode_performance(shard_sessions)
    operational = _merge_runtime_preflight(ranked_shard_sessions)
    if not raw_replay_required and performance is None and operational is None:
        return merged
    effective_settings = merged.get("effective_settings")
    if not isinstance(effective_settings, dict):
        raise ArtifactContractError(
            "merged backend session requires effective settings",
            code="merge.backend_session_effective_settings_missing",
        )
    if performance is not None:
        effective_settings["performance"] = performance
    if operational is not None:
        effective_settings["runtime_preflight"] = operational
    if not raw_replay_required:
        return merged
    raw_replay = effective_settings.get("raw_replay")
    if not isinstance(raw_replay, dict):
        raise ArtifactContractError(
            "vLLM raw replay requires semantic replay settings",
            code="merge.raw_replay_receipt_missing",
        )
    raw_replay["request_count"] = len(raw_replay_trace)
    raw_replay["row_evidence_sha256"] = _sha256_json(
        _raw_replay_trace_mapping(raw_replay_trace)
    )
    return merged


def _merge_runtime_preflight(
    ranked_shard_sessions: list[tuple[int, dict[str, Any]]],
) -> dict[str, Any] | None:
    observations: list[dict[str, Any]] = []
    semantic: dict[str, Any] | None = None
    missing = 0
    for rank, session in ranked_shard_sessions:
        settings = session.get("effective_settings")
        value = settings.get("runtime_preflight") if isinstance(settings, dict) else None
        if value is None:
            missing += 1
            continue
        if not isinstance(value, dict):
            raise ArtifactContractError(
                "rank-local runtime preflight must be a JSON object",
                code="merge.runtime_preflight_invalid",
                context={"rank": rank},
            )
        status = value.get("status")
        live_decode = value.get("live_decode")
        cleanup = value.get("cleanup")
        if (
            status != "passed_live_decode_and_cleanup"
            or not isinstance(live_decode, dict)
            or not isinstance(cleanup, dict)
            or cleanup.get("status") != "completed"
        ):
            raise ArtifactContractError(
                "rank-local runtime preflight is not live-decode and cleanup complete",
                code="merge.runtime_preflight_incomplete",
                context={"rank": rank, "status": status},
            )
        current_semantic = dict(value)
        current_semantic.pop("live_decode", None)
        current_semantic.pop("cleanup", None)
        process = current_semantic.pop("process", None)
        cuda = current_semantic.pop("cuda", None)
        current_semantic.pop("rank_evidence", None)
        current_semantic["status"] = "ready_for_engine_construction"
        if semantic is None:
            semantic = current_semantic
        elif current_semantic != semantic:
            raise ArtifactContractError(
                "rank-local runtime preflight settings disagree",
                code="merge.runtime_preflight_mismatch",
                context={"rank": rank},
            )
        observations.append(
            {
                "rank": rank,
                "process": process,
                "cuda": cuda,
                "live_decode": dict(live_decode),
                "cleanup": dict(cleanup),
            }
        )
    if not observations:
        return None
    if missing:
        raise ArtifactContractError(
            "rank-local runtime preflight is missing on some ranks",
            code="merge.runtime_preflight_missing",
            context={
                "missing_rank_count": missing,
                "rank_count": len(ranked_shard_sessions),
            },
        )
    assert semantic is not None
    return {
        **semantic,
        "status": "passed_all_rank_live_decode_and_cleanup",
        "rank_evidence": observations,
    }


def _merge_decode_performance(
    shard_sessions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    rows: list[dict[str, Any]] = []
    missing_count = 0
    for session in shard_sessions:
        settings = session.get("effective_settings")
        performance = settings.get("performance") if isinstance(settings, dict) else None
        if performance is None:
            missing_count += 1
            continue
        if not isinstance(performance, dict):
            raise ArtifactContractError(
                "rank-local decode performance must be a JSON object",
                code="merge.decode_performance_invalid",
            )
        rows.append(dict(performance))
    if not rows:
        return None
    if missing_count:
        raise ArtifactContractError(
            "rank-local decode performance is missing on some ranks",
            code="merge.decode_performance_missing",
            context={"missing_rank_count": missing_count, "rank_count": len(shard_sessions)},
        )

    request_count = sum(_performance_int(row, "request_count") for row in rows)
    generated_token_count = sum(
        _performance_int(row, "generated_token_count") for row in rows
    )
    wall_seconds = max(
        _performance_float(row, "decode_elapsed_seconds") for row in rows
    )
    allocated = _performance_optional_ints(rows, "peak_cuda_memory_allocated_bytes")
    reserved = _performance_optional_ints(rows, "peak_cuda_memory_reserved_bytes")
    return {
        "measurement_scope": "parallel_rank_backend_decode_capacity_estimate",
        "aggregation_assumption": "perfect_rank_overlap_using_max_rank_elapsed",
        "controller_wall_time_observed": False,
        "rank_count": len(rows),
        "request_count": request_count,
        "generated_token_count": generated_token_count,
        "parallel_decode_wall_seconds_max": wall_seconds,
        "requests_per_second": request_count / wall_seconds,
        "generated_tokens_per_second": generated_token_count / wall_seconds,
        "peak_cuda_memory_allocated_bytes_per_rank_max": (
            max(allocated) if allocated else None
        ),
        "peak_cuda_memory_reserved_bytes_per_rank_max": (
            max(reserved) if reserved else None
        ),
        "rank_measurements": rows,
    }


def _performance_int(row: dict[str, Any], field: str) -> int:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ArtifactContractError(
            "rank-local decode performance integer is invalid",
            code="merge.decode_performance_invalid",
            context={"field": field, "value": value},
        )
    return value


def _performance_float(row: dict[str, Any], field: str) -> float:
    value = row.get(field)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ArtifactContractError(
            "rank-local decode performance duration is invalid",
            code="merge.decode_performance_invalid",
            context={"field": field, "value": value},
        )
    return float(value)


def _performance_optional_ints(
    rows: list[dict[str, Any]], field: str
) -> list[int]:
    values: list[int] = []
    for row in rows:
        value = row.get(field)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ArtifactContractError(
                "rank-local decode performance peak memory is invalid",
                code="merge.decode_performance_invalid",
                context={"field": field, "value": value},
            )
        values.append(value)
    return values


def _raw_replay_trace_mapping(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        str(row["row_id"]): {
            key: value for key, value in row.items() if key != "row_id"
        }
        for row in rows
    }


def _require_shard_artifacts(shard_dir: Path) -> None:
    missing = [name for name in REQUIRED_MERGE_ARTIFACTS if not (shard_dir / name).is_file()]
    if missing:
        raise ArtifactContractError(
            "rank-local shard is missing required artifacts",
            code="merge.missing_shard_artifact",
            context={"shard_dir": str(shard_dir), "missing_artifacts": missing},
        )


def _worker_metadata_from_manifest(
    *,
    manifest: dict[str, Any],
    rank: int,
) -> dict[str, Any]:
    parallelism = manifest.get("parallelism")
    if not isinstance(parallelism, dict):
        raise ArtifactContractError(
            "rank-local shard manifest is missing parallelism metadata",
            code="merge.parallelism_metadata_missing",
            context={"rank": rank},
        )
    if not isinstance(parallelism.get("worker"), dict):
        raise ArtifactContractError(
            "rank-local shard manifest is missing worker metadata",
            code="merge.worker_metadata_missing",
            context={"rank": rank, "field": "parallelism.worker"},
        )
    worker = dict(parallelism["worker"])
    required = {
        "rank": worker.get("rank"),
        "world_size": worker.get("world_size"),
        "parent_visible_device_token": worker.get("parent_visible_device_token"),
        "worker_cuda_visible_devices": worker.get("worker_cuda_visible_devices"),
        "worker_logical_device": worker.get("worker_logical_device"),
        "cuda_device_count": worker.get("cuda_device_count"),
        "cuda_current_device": worker.get("cuda_current_device"),
        "model_first_parameter_device": worker.get("model_first_parameter_device"),
    }
    missing = [
        field
        for field, value in required.items()
        if value is None or (isinstance(value, str) and not value)
    ]
    if missing:
        raise ArtifactContractError(
            "rank-local shard manifest is missing required worker metadata",
            code="merge.worker_metadata_missing",
            context={"rank": rank, "missing_fields": missing},
        )
    return {
        "rank": _require_artifact_int(
            worker["rank"],
            artifact_name=MANIFEST_NAME,
            field="parallelism.worker.rank",
        ),
        "world_size": _require_artifact_int(
            worker["world_size"],
            artifact_name=MANIFEST_NAME,
            field="parallelism.worker.world_size",
            minimum=1,
        ),
        "assigned_parent_visible_device_token": _require_artifact_str(
            worker["parent_visible_device_token"],
            artifact_name=MANIFEST_NAME,
            field="parallelism.worker.parent_visible_device_token",
        ),
        "worker_cuda_visible_devices": _require_artifact_str(
            worker["worker_cuda_visible_devices"],
            artifact_name=MANIFEST_NAME,
            field="parallelism.worker.worker_cuda_visible_devices",
        ),
        "worker_logical_device": _require_artifact_str(
            worker["worker_logical_device"],
            artifact_name=MANIFEST_NAME,
            field="parallelism.worker.worker_logical_device",
        ),
        "cuda_device_count": _require_artifact_int(
            worker["cuda_device_count"],
            artifact_name=MANIFEST_NAME,
            field="parallelism.worker.cuda_device_count",
            minimum=0,
        ),
        "cuda_current_device": _require_artifact_int(
            worker["cuda_current_device"],
            artifact_name=MANIFEST_NAME,
            field="parallelism.worker.cuda_current_device",
            minimum=0,
        ),
        "model_first_parameter_device": _require_artifact_str(
            worker["model_first_parameter_device"],
            artifact_name=MANIFEST_NAME,
            field="parallelism.worker.model_first_parameter_device",
        ),
    }


def _with_worker_metadata(
    rows: list[dict[str, Any]],
    *,
    worker_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    merged_rows = []
    for row in rows:
        for key, expected in worker_metadata.items():
            if expected is None or key not in row:
                continue
            if row[key] != expected:
                raise ArtifactContractError(
                    "rank-local sidecar worker metadata disagrees with validated shard metadata",
                    code="merge.worker_metadata_mismatch",
                    context={
                        "row_id": row.get("row_id"),
                        "field": key,
                        "expected": expected,
                        "observed": row[key],
                    },
                )
        merged_rows.append(
            {
                **{
                    key: value
                    for key, value in worker_metadata.items()
                    if value is not None and key not in row
                },
                **row,
            }
        )
    return merged_rows


def _rank_from_manifest_or_path(*, manifest: dict[str, Any], shard_dir: Path) -> int:
    parallelism = manifest.get("parallelism")
    if isinstance(parallelism, dict):
        worker = parallelism.get("worker")
        if isinstance(worker, dict) and "rank" in worker:
            return _require_artifact_int(
                worker["rank"],
                artifact_name=MANIFEST_NAME,
                field="parallelism.worker.rank",
            )
    name = shard_dir.name
    if name.startswith("rank-"):
        try:
            return int(name.removeprefix("rank-"))
        except ValueError as exc:
            raise ArtifactContractError(
                "rank-local shard directory name has an invalid rank suffix",
                code="merge.invalid_integer_field",
                context={
                    "artifact": MANIFEST_NAME,
                    "field": "shard_dir.rank_suffix",
                    "shard_dir": str(shard_dir),
                    "observed_value": name,
                },
                cause=exc,
            ) from exc
    raise ArtifactContractError(
        "rank-local shard manifest does not expose rank identity",
        code="merge.rank_missing",
        context={"shard_dir": str(shard_dir)},
    )


def _summary_status(shard_dir: Path) -> str | None:
    summary = _read_json(shard_dir / SUMMARY_NAME)
    if "terminal_status" not in summary:
        raise ArtifactContractError(
            "rank-local shard summary is missing terminal status",
            code="merge.terminal_status_missing",
            context={"shard_dir": str(shard_dir)},
        )
    value = summary.get("terminal_status")
    if value is None:
        raise ArtifactContractError(
            "rank-local shard summary is missing terminal status",
            code="merge.terminal_status_missing",
            context={"shard_dir": str(shard_dir)},
        )
    return str(value)


def _require_shard_plan_fingerprint(
    *,
    manifest: dict[str, Any],
    plan: DataParallelPlan,
    rank: int,
) -> None:
    parallelism = manifest.get("parallelism")
    observed = None
    if isinstance(parallelism, dict):
        observed = parallelism.get("shard_plan_fingerprint")
        if observed is None:
            plan_payload = parallelism.get("plan")
            if isinstance(plan_payload, dict):
                observed = plan_payload.get("fingerprint")
    if observed is None:
        raise ArtifactContractError(
            "rank-local shard manifest is missing shard-plan fingerprint",
            code="merge.shard_plan_missing",
            context={"rank": rank, "expected": plan.fingerprint},
        )
    if observed is not None and observed != plan.fingerprint:
        raise ArtifactContractError(
            "rank-local shard plan fingerprint disagrees with controller plan",
            code="merge.shard_plan_mismatch",
            context={
                "rank": rank,
                "observed": observed,
                "expected": plan.fingerprint,
            },
        )


def _validate_rank_assignment(
    *,
    manifest: dict[str, Any],
    plan: DataParallelPlan,
    rank: int,
    raw_rows: list[dict[str, Any]],
    scored_rows: list[dict[str, Any]],
    image_plan_rows: list[dict[str, Any]],
    token_trace_rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
) -> None:
    rank_plan = _rank_plan_for(plan=plan, rank=rank)
    expected_row_ids = tuple(str(row_id) for row_id in rank_plan.row_ids)
    expected_row_indices = tuple(int(index) for index in rank_plan.row_indices)
    parallelism = manifest.get("parallelism")
    assignment = None
    if isinstance(parallelism, dict):
        assignment = parallelism.get("shard_assignment")
    if not isinstance(assignment, dict):
        raise ArtifactContractError(
            "rank-local shard manifest is missing shard assignment",
            code="merge.rank_assignment_missing",
            context={"rank": rank},
        )
    assigned_row_ids = _require_artifact_list(
        assignment.get("assigned_row_ids", []),
        artifact_name=MANIFEST_NAME,
        field="assigned_row_ids",
    )
    assigned_row_indices = _require_artifact_list(
        assignment.get("assigned_row_indices", []),
        artifact_name=MANIFEST_NAME,
        field="assigned_row_indices",
    )
    observed_manifest_ids = tuple(
        _require_artifact_str(
            row_id,
            artifact_name=MANIFEST_NAME,
            field=f"assigned_row_ids[{index}]",
        )
        for index, row_id in enumerate(assigned_row_ids)
    )
    observed_manifest_indices = tuple(
        _require_artifact_int(
            index,
            artifact_name=MANIFEST_NAME,
            field=f"assigned_row_indices[{offset}]",
        )
        for offset, index in enumerate(assigned_row_indices)
    )
    if (
        observed_manifest_ids != expected_row_ids
        or observed_manifest_indices != expected_row_indices
    ):
        raise ArtifactContractError(
            "rank-local shard assignment disagrees with controller rank plan",
            code="merge.rank_assignment_mismatch",
            context={
                "rank": rank,
                "expected_row_ids": list(expected_row_ids),
                "observed_row_ids": list(observed_manifest_ids),
                "expected_row_indices": list(expected_row_indices),
                "observed_row_indices": list(observed_manifest_indices),
                "source": "manifest",
            },
        )
    expected_identity = tuple(zip(expected_row_ids, expected_row_indices, strict=True))
    for artifact_name, rows in (
        (RAW_NAME, raw_rows),
        (SCORED_NAME, scored_rows),
        (IMAGE_PLAN_NAME, image_plan_rows),
    ):
        observed_identity = tuple(
            (
                _require_artifact_str(
                    row.get("row_id"),
                    artifact_name=artifact_name,
                    field="row_id",
                    row_position=row_position,
                ),
                _require_artifact_int(
                    row.get("row_index"),
                    artifact_name=artifact_name,
                    field="row_index",
                    row_id=str(row.get("row_id")) if row.get("row_id") is not None else None,
                    row_position=row_position,
                ),
            )
            for row_position, row in enumerate(rows)
        )
        if observed_identity != expected_identity:
            raise ArtifactContractError(
                "rank-local shard row artifacts disagree with controller rank plan",
                code="merge.rank_assignment_mismatch",
                context={
                    "rank": rank,
                    "artifact": artifact_name,
                    "expected": [list(item) for item in expected_identity],
                    "observed": [list(item) for item in observed_identity],
                    "source": "artifact_rows",
                },
            )
    allowed_row_ids = set(expected_row_ids)
    for artifact_name, rows in (
        (TOKEN_TRACE_NAME, token_trace_rows),
        (PARSE_DIAGNOSTICS_NAME, diagnostic_rows),
    ):
        unexpected = [
            str(row.get("row_id"))
            for row in rows
            if row.get("row_id") is not None and str(row.get("row_id")) not in allowed_row_ids
        ]
        if unexpected:
            raise ArtifactContractError(
                "rank-local sidecar rows disagree with controller rank plan",
                code="merge.rank_assignment_mismatch",
                context={
                    "rank": rank,
                    "artifact": artifact_name,
                    "unexpected_row_ids": unexpected,
                    "expected_row_ids": list(expected_row_ids),
                    "source": "sidecar_rows",
                },
            )


def _validate_worker_metadata_matches_rank_plan(
    *,
    worker_metadata: dict[str, Any],
    rank_plan: RankShardPlan,
) -> None:
    expected = {
        "rank": rank_plan.rank,
        "world_size": rank_plan.world_size,
        "assigned_parent_visible_device_token": rank_plan.parent_visible_device_token,
        "worker_cuda_visible_devices": rank_plan.parent_visible_device_token,
    }
    for field, expected_value in expected.items():
        observed = worker_metadata.get(field)
        if observed != expected_value:
            raise ArtifactContractError(
                "rank-local worker metadata disagrees with controller rank plan",
                code="merge.worker_metadata_mismatch",
                context={
                    "rank": rank_plan.rank,
                    "field": field,
                    "expected": expected_value,
                    "observed": observed,
                },
            )
    if worker_metadata.get("worker_logical_device") != "cuda:0":
        raise ArtifactContractError(
            "rank-local worker logical device must be cuda:0",
            code="merge.worker_metadata_mismatch",
            context={
                "rank": rank_plan.rank,
                "field": "worker_logical_device",
                "expected": "cuda:0",
                "observed": worker_metadata.get("worker_logical_device"),
            },
        )
    required_cuda_evidence = {
        "cuda_device_count": 1,
        "cuda_current_device": 0,
        "model_first_parameter_device": "cuda:0",
    }
    for field, expected_value in required_cuda_evidence.items():
        observed = worker_metadata.get(field)
        if field == "model_first_parameter_device" and observed == "cuda":
            observed = "cuda:0"
        if observed != expected_value:
            raise ArtifactContractError(
                "rank-local worker CUDA evidence violates one-device isolation",
                code="merge.worker_metadata_mismatch",
                context={
                    "rank": rank_plan.rank,
                    "field": field,
                    "expected": expected_value,
                    "observed": observed,
                },
            )


def _rank_plan_for(*, plan: DataParallelPlan, rank: int) -> RankShardPlan:
    for rank_plan in plan.ranks:
        if rank_plan.rank == rank:
            return rank_plan
    raise ArtifactContractError(
        "rank-local shard rank is not present in the controller plan",
        code="merge.unknown_rank",
        context={
            "rank": rank,
            "expected_ranks": [rank_plan.rank for rank_plan in plan.ranks],
        },
    )


def _require_identity_match(
    *,
    expected: dict[str, Any],
    observed: dict[str, Any],
) -> None:
    for field, expected_value in expected.items():
        observed_value = observed.get(field)
        if observed_value != expected_value:
            raise ArtifactContractError(
                "rank-local shard identity disagrees with prior shard",
                code="merge.identity_mismatch",
                context={
                    "field": field,
                    "expected": expected_value,
                    "observed": observed_value,
                },
            )


def _controller_identity_vector(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_identity": metadata.get("model_identity"),
        "model_identity_fingerprint": metadata.get("model_identity_fingerprint"),
        "processor_identity": metadata.get("processor_identity"),
        "processor_identity_fingerprint": metadata.get("processor_identity_fingerprint"),
        "tokenizer_identity": metadata.get("tokenizer_identity"),
        "adapter_identity": metadata.get("adapter_identity"),
        "embedding_delta_identity": metadata.get("embedding_delta_identity"),
        "backend_session": metadata.get("backend_session"),
        "likelihood_semantics": metadata.get("likelihood_semantics"),
        "execution_model_identity": metadata.get("execution_model_identity"),
        "frontend_identity": metadata.get("frontend_identity"),
        "raw_model_logprob_status": metadata.get("raw_model_logprob_status"),
        "template_identity": metadata.get("template_identity"),
        "prompt_policy_fingerprint": metadata.get("prompt_policy_fingerprint"),
        "generation_config_fingerprint": metadata.get("generation_config_fingerprint"),
        "generation_policy": metadata.get("generation_policy"),
        "dataset_identity": metadata.get("dataset_identity"),
        "parser_policy": metadata.get("parser_policy"),
        "score_policy_fingerprint": metadata.get(
            "score_policy_fingerprint",
            SCORE_POLICY_FINGERPRINT,
        ),
        "backend": metadata.get("backend"),
        "backend_mode": metadata.get("backend_mode"),
        "response_family": metadata.get("response_family"),
    }


def _require_controller_identity_match(
    *,
    shard_identity: dict[str, Any],
    controller_identity: dict[str, Any],
) -> None:
    for field, shard_value in shard_identity.items():
        controller_value = controller_identity.get(field)
        if controller_value != shard_value:
            raise ArtifactContractError(
                "controller merge metadata disagrees with validated shard identity",
                code="merge.controller_identity_mismatch",
                context={
                    "field": field,
                    "controller": controller_value,
                    "shard": shard_value,
                },
            )


def _metadata_with_shard_owned_identity(
    *,
    metadata: dict[str, Any],
    shard_identity: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(metadata)
    for field in SHARD_OWNED_IDENTITY_FIELDS:
        if _needs_shard_owned_identity(merged.get(field)):
            merged[field] = shard_identity.get(field)
    return merged


def _needs_shard_owned_identity(value: Any) -> bool:
    return value is None or value == {} or value == "" or value == "unknown-before-runtime"


def _field_context(
    *,
    artifact_name: str,
    field: str,
    row_id: str | None = None,
    row_position: int | None = None,
    pred_index: int | None = None,
    object_span_id: str | None = None,
    value: Any = None,
) -> dict[str, Any]:
    context: dict[str, Any] = {
        "artifact": artifact_name,
        "field": field,
        "observed_type": type(value).__name__,
    }
    if row_id is not None:
        context["row_id"] = row_id
    if row_position is not None:
        context["row_position"] = row_position
    if pred_index is not None:
        context["pred_index"] = pred_index
    if object_span_id is not None:
        context["object_span_id"] = object_span_id
    if isinstance(value, str | int | float | bool) or value is None:
        context["observed_value"] = value
    return context


def _require_artifact_dict(
    value: Any,
    *,
    artifact_name: str,
    field: str,
    row_id: str | None = None,
    row_position: int | None = None,
    pred_index: int | None = None,
    object_span_id: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ArtifactContractError(
            "merge artifact field must be a JSON object",
            code="merge.invalid_object_field",
            context=_field_context(
                artifact_name=artifact_name,
                field=field,
                row_id=row_id,
                row_position=row_position,
                pred_index=pred_index,
                object_span_id=object_span_id,
                value=value,
            ),
        )
    return value


def _require_artifact_list(
    value: Any,
    *,
    artifact_name: str,
    field: str,
    row_id: str | None = None,
    row_position: int | None = None,
    pred_index: int | None = None,
    object_span_id: str | None = None,
) -> list[Any]:
    if not isinstance(value, list):
        raise ArtifactContractError(
            "merge artifact field must be a JSON list",
            code="merge.invalid_list_field",
            context=_field_context(
                artifact_name=artifact_name,
                field=field,
                row_id=row_id,
                row_position=row_position,
                pred_index=pred_index,
                object_span_id=object_span_id,
                value=value,
            ),
        )
    return value


def _require_artifact_str(
    value: Any,
    *,
    artifact_name: str,
    field: str,
    row_id: str | None = None,
    row_position: int | None = None,
    pred_index: int | None = None,
    object_span_id: str | None = None,
) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactContractError(
            "merge artifact field must be a non-empty JSON string",
            code="merge.invalid_string_field",
            context=_field_context(
                artifact_name=artifact_name,
                field=field,
                row_id=row_id,
                row_position=row_position,
                pred_index=pred_index,
                object_span_id=object_span_id,
                value=value,
            ),
        )
    return value


def _require_artifact_int(
    value: Any,
    *,
    artifact_name: str,
    field: str,
    row_id: str | None = None,
    row_position: int | None = None,
    pred_index: int | None = None,
    object_span_id: str | None = None,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactContractError(
            "merge artifact field must be a JSON integer",
            code="merge.invalid_integer_field",
            context=_field_context(
                artifact_name=artifact_name,
                field=field,
                row_id=row_id,
                row_position=row_position,
                pred_index=pred_index,
                object_span_id=object_span_id,
                value=value,
            ),
        )
    if minimum is not None and value < minimum:
        raise ArtifactContractError(
            "merge artifact integer field is below the allowed minimum",
            code="merge.invalid_integer_field",
            context={
                **_field_context(
                    artifact_name=artifact_name,
                    field=field,
                    row_id=row_id,
                    row_position=row_position,
                    pred_index=pred_index,
                    object_span_id=object_span_id,
                    value=value,
                ),
                "minimum": minimum,
            },
        )
    return value


def _optional_artifact_int(
    value: Any,
    *,
    artifact_name: str,
    field: str,
    default: int,
    row_id: str | None = None,
    row_position: int | None = None,
) -> int:
    if value is None:
        return default
    return _require_artifact_int(
        value,
        artifact_name=artifact_name,
        field=field,
        row_id=row_id,
        row_position=row_position,
    )


def _require_artifact_float(
    value: Any,
    *,
    artifact_name: str,
    field: str,
    row_id: str | None = None,
    row_position: int | None = None,
    pred_index: int | None = None,
    object_span_id: str | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ArtifactContractError(
            "merge artifact field must be a JSON number",
            code="merge.invalid_float_field",
            context=_field_context(
                artifact_name=artifact_name,
                field=field,
                row_id=row_id,
                row_position=row_position,
                pred_index=pred_index,
                object_span_id=object_span_id,
                value=value,
            ),
        )
    value_float = float(value)
    if not math.isfinite(value_float) or (
        minimum is not None and value_float < minimum
    ) or (maximum is not None and value_float > maximum):
        raise ArtifactContractError(
            "merge artifact numeric field is outside the allowed finite range",
            code="merge.invalid_float_field",
            context={
                **_field_context(
                    artifact_name=artifact_name,
                    field=field,
                    row_id=row_id,
                    row_position=row_position,
                    pred_index=pred_index,
                    object_span_id=object_span_id,
                    value=value,
                ),
                "minimum": minimum,
                "maximum": maximum,
            },
        )
    return value_float


def _validate_and_sort_rows(
    *,
    rows: list[dict[str, Any]],
    expected_row_ids: tuple[str, ...],
    expected_index_by_row_id: dict[str, int],
    artifact_name: str,
) -> list[dict[str, Any]]:
    rows_by_id: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    unknown: list[str] = []
    for row_position, row in enumerate(rows):
        row_id = _require_artifact_str(
            row.get("row_id"),
            artifact_name=artifact_name,
            field="row_id",
            row_position=row_position,
        )
        if row_id in rows_by_id:
            duplicates.append(row_id)
        if row_id not in expected_index_by_row_id:
            unknown.append(row_id)
        rows_by_id[row_id] = row
    if duplicates:
        raise ArtifactContractError(
            "merged artifacts contain duplicate row ids",
            code="merge.duplicate_row",
            context={"artifact": artifact_name, "duplicate_row_ids": duplicates},
        )
    if unknown:
        raise ArtifactContractError(
            "merged artifacts contain rows outside the input identity set",
            code="merge.unknown_row",
            context={"artifact": artifact_name, "unknown_row_ids": unknown},
        )
    missing = [row_id for row_id in expected_row_ids if row_id not in rows_by_id]
    if missing:
        raise ArtifactContractError(
            "merged artifacts are missing input rows",
            code="merge.missing_row",
            context={"artifact": artifact_name, "missing_row_ids": missing},
        )
    sorted_rows: list[dict[str, Any]] = []
    for expected_index, row_id in enumerate(expected_row_ids):
        row = rows_by_id[row_id]
        observed_index = _require_artifact_int(
            row.get("row_index"),
            artifact_name=artifact_name,
            field="row_index",
            row_id=row_id,
        )
        if observed_index != expected_index:
            raise ArtifactContractError(
                "merged row identity does not match original row order",
                code="merge.row_order_mismatch",
                context={
                    "artifact": artifact_name,
                    "row_id": row_id,
                    "expected_row_index": expected_index,
                    "observed_row_index": observed_index,
                },
            )
        sorted_rows.append(row)
    return sorted_rows


def _validate_raw_scored_image_identity(
    *,
    raw_rows: list[dict[str, Any]],
    scored_rows: list[dict[str, Any]],
    image_plan_rows: list[dict[str, Any]],
) -> None:
    for index, (raw, scored, image_plan) in enumerate(
        zip(raw_rows, scored_rows, image_plan_rows, strict=True)
    ):
        expected_row_id = _require_artifact_str(
            raw.get("row_id"),
            artifact_name=RAW_NAME,
            field="row_id",
            row_position=index,
        )
        expected_row_index = _require_artifact_int(
            raw.get("row_index"),
            artifact_name=RAW_NAME,
            field="row_index",
            row_id=expected_row_id,
            row_position=index,
        )
        expected = (expected_row_id, expected_row_index)
        observed = (
            (
                _require_artifact_str(
                    scored.get("row_id"),
                    artifact_name=SCORED_NAME,
                    field="row_id",
                    row_position=index,
                ),
                _require_artifact_int(
                    scored.get("row_index"),
                    artifact_name=SCORED_NAME,
                    field="row_index",
                    row_id=str(scored.get("row_id")) if scored.get("row_id") is not None else None,
                    row_position=index,
                ),
            ),
            (
                _require_artifact_str(
                    image_plan.get("row_id"),
                    artifact_name=IMAGE_PLAN_NAME,
                    field="row_id",
                    row_position=index,
                ),
                _require_artifact_int(
                    image_plan.get("row_index"),
                    artifact_name=IMAGE_PLAN_NAME,
                    field="row_index",
                    row_id=str(image_plan.get("row_id"))
                    if image_plan.get("row_id") is not None
                    else None,
                    row_position=index,
                ),
            ),
        )
        if observed[0] != expected or observed[1] != expected:
            raise ArtifactContractError(
                "merged raw, scored, and image-plan rows disagree on row identity",
                code="merge.row_identity_mismatch",
                context={
                    "row_position": index,
                    "raw_identity": expected,
                    "scored_identity": observed[0],
                    "image_plan_identity": observed[1],
                },
            )


def _validate_row_local_score_provenance(
    scored_rows: list[dict[str, Any]],
    *,
    token_trace_rows: list[dict[str, Any]],
) -> None:
    replay_by_key = {
        (
            _require_artifact_str(
                row.get("row_id"),
                artifact_name=TOKEN_TRACE_NAME,
                field="row_id",
            ),
            _require_artifact_str(
                row.get("object_span_id"),
                artifact_name=TOKEN_TRACE_NAME,
                field="object_span_id",
            ),
        ): row
        for row in token_trace_rows
        if row.get("trace_type") == "selected_token_replay"
    }
    for row_position, row in enumerate(scored_rows):
        row_id = _require_artifact_str(
            row.get("row_id"),
            artifact_name=SCORED_NAME,
            field="row_id",
            row_position=row_position,
        )
        predictions = _require_artifact_list(
            row.get("pred", []),
            artifact_name=SCORED_NAME,
            field="pred",
            row_id=row_id,
            row_position=row_position,
        )
        for pred_index, pred_payload in enumerate(predictions):
            pred = _require_artifact_dict(
                pred_payload,
                artifact_name=SCORED_NAME,
                field=f"pred[{pred_index}]",
                row_id=row_id,
                row_position=row_position,
                pred_index=pred_index,
            )
            if pred.get("pred_score_version") != PRED_SCORE_VERSION:
                raise ArtifactContractError(
                    "merged scored prediction score version does not match score policy",
                    code="merge.row_score_version_mismatch",
                    context={
                        "row_id": row_id,
                        "pred_index": pred_index,
                        "observed": pred.get("pred_score_version"),
                        "expected": PRED_SCORE_VERSION,
                    },
                )
            _require_artifact_float(
                pred.get("score"),
                artifact_name=SCORED_NAME,
                field="score",
                row_id=row_id,
                row_position=row_position,
                pred_index=pred_index,
                minimum=0.0,
                maximum=1.0,
            )
            object_span_id = _require_artifact_str(
                pred.get("object_span_id"),
                artifact_name=SCORED_NAME,
                field="object_span_id",
                row_id=row_id,
                row_position=row_position,
                pred_index=pred_index,
            )
            source = pred.get("pred_score_source")
            if not isinstance(source, dict):
                raise ArtifactContractError(
                    "merged scored prediction lacks score source",
                    code="merge.row_score_provenance_missing",
                    context={"row_id": row_id, "pred_index": pred_index},
                )
            if str(source.get("row_id")) != row_id:
                raise ArtifactContractError(
                    "merged scored prediction score source row_id mismatches row",
                    code="merge.row_score_provenance_mismatch",
                    context={"row_id": row_id, "pred_index": pred_index},
                )
            if str(source.get("object_span_id")) != object_span_id:
                raise ArtifactContractError(
                    "merged scored prediction score source object span mismatches prediction",
                    code="merge.row_score_provenance_mismatch",
                    context={
                        "row_id": row_id,
                        "pred_index": pred_index,
                        "object_span_id": object_span_id,
                        "source_object_span_id": source.get("object_span_id"),
                    },
                )
            if source.get("score_policy_fingerprint") != SCORE_POLICY_FINGERPRINT:
                raise ArtifactContractError(
                    "merged scored prediction score source policy mismatches merged policy",
                    code="merge.row_score_policy_mismatch",
                    context={
                        "row_id": row_id,
                        "pred_index": pred_index,
                        "source_score_policy_fingerprint": source.get(
                            "score_policy_fingerprint"
                        ),
                        "expected": SCORE_POLICY_FINGERPRINT,
                    },
                )
            replay = replay_by_key.get((row_id, object_span_id))
            if replay is None:
                raise ArtifactContractError(
                    "selected-token replay row is missing",
                    code="artifacts.selected_replay_missing",
                    context={"row_id": row_id, "object_span_id": object_span_id},
                )
            _require_score_source_matches_replay(
                source=source,
                replay=replay,
                row_id=row_id,
                object_span_id=object_span_id,
                pred_index=pred_index,
            )


def _require_score_source_matches_replay(
    *,
    source: dict[str, Any],
    replay: dict[str, Any],
    row_id: str,
    object_span_id: str,
    pred_index: int,
) -> None:
    fields = (
        "generated_step_indices",
        "token_ids",
        "token_text",
        "selected_logprobs",
        "selected_count",
        "score_policy_fingerprint",
    )
    for field in fields:
        if source.get(field) != replay.get(field):
            raise ArtifactContractError(
                "merged scored prediction score source disagrees with replay evidence",
                code="merge.row_score_provenance_mismatch",
                context={
                    "row_id": row_id,
                    "object_span_id": object_span_id,
                    "pred_index": pred_index,
                    "field": field,
                    "source": source.get(field),
                    "replay": replay.get(field),
                },
            )


def _validate_trace_uniqueness(
    rows: list[dict[str, Any]],
    *,
    raw_model_logprob_status: Any,
) -> None:
    if raw_model_logprob_status not in {"available", "disabled"}:
        raise ArtifactContractError(
            "merged token trace requires an explicit raw-model likelihood status",
            code="merge.invalid_generated_likelihood",
            context={"raw_model_logprob_status": raw_model_logprob_status},
        )
    generated: set[tuple[str, int]] = set()
    replay: set[tuple[str, str]] = set()
    for row_position, row in enumerate(rows):
        trace_type = row.get("trace_type")
        if trace_type == "generated_token":
            row_id = _require_artifact_str(
                row.get("row_id"),
                artifact_name=TOKEN_TRACE_NAME,
                field="row_id",
                row_position=row_position,
            )
            generated_step_index = _require_artifact_int(
                row.get("generated_step_index"),
                artifact_name=TOKEN_TRACE_NAME,
                field="generated_step_index",
                row_id=row_id,
                row_position=row_position,
                minimum=0,
            )
            _validate_generated_token_likelihood(
                row,
                row_id=row_id,
                row_position=row_position,
                generated_step_index=generated_step_index,
                expected_raw_status=raw_model_logprob_status,
            )
            key = (row_id, generated_step_index)
            if key in generated:
                raise ArtifactContractError(
                    "merged token trace contains duplicate generated-token evidence",
                    code="merge.duplicate_generated_trace",
                    context={"row_id": key[0], "generated_step_index": key[1]},
                )
            generated.add(key)
        elif trace_type == "selected_token_replay":
            row_id, object_span_id = _validate_selected_replay_row_shape(
                row,
                row_position=row_position,
            )
            key = (row_id, object_span_id)
            if key in replay:
                raise ArtifactContractError(
                    "merged token trace contains duplicate replay evidence",
                    code="merge.duplicate_replay_trace",
                    context={"row_id": key[0], "object_span_id": key[1]},
                )
            replay.add(key)


def _validate_generated_token_likelihood(
    row: dict[str, Any],
    *,
    row_id: str,
    row_position: int,
    generated_step_index: int,
    expected_raw_status: str,
) -> None:
    context = {
        "artifact": TOKEN_TRACE_NAME,
        "row_id": row_id,
        "row_position": row_position,
        "generated_step_index": generated_step_index,
    }
    is_pad = row.get("is_pad")
    if not isinstance(is_pad, bool):
        raise ArtifactContractError(
            "generated-token padding evidence must be boolean",
            code="merge.invalid_generated_likelihood",
            context={**context, "field": "is_pad", "value": is_pad},
        )
    observed_raw_status = row.get("raw_model_logprob_status")
    if observed_raw_status != expected_raw_status:
        raise ArtifactContractError(
            "generated-token raw-model likelihood status disagrees with the shard contract",
            code="merge.invalid_generated_likelihood",
            context={
                **context,
                "field": "raw_model_logprob_status",
                "observed": observed_raw_status,
                "expected": expected_raw_status,
            },
        )
    policy_logprob = row.get("logprob")
    raw_model_logprob = row.get("raw_model_logprob")
    if is_pad:
        if policy_logprob is not None or raw_model_logprob is not None:
            raise ArtifactContractError(
                "padding trace rows must not carry likelihood evidence",
                code="merge.invalid_generated_likelihood",
                context={**context, "field": "logprob/raw_model_logprob"},
            )
        return
    _require_generated_logprob(
        policy_logprob,
        field="logprob",
        context=context,
    )
    if expected_raw_status == "available":
        _require_generated_logprob(
            raw_model_logprob,
            field="raw_model_logprob",
            context=context,
        )
    elif raw_model_logprob is not None:
        raise ArtifactContractError(
            "disabled raw-model likelihood channel must not carry values",
            code="merge.invalid_generated_likelihood",
            context={**context, "field": "raw_model_logprob", "value": raw_model_logprob},
        )


def _require_generated_logprob(
    value: Any,
    *,
    field: str,
    context: dict[str, Any],
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) > 0.0
    ):
        raise ArtifactContractError(
            "generated-token likelihood must be a finite non-positive number",
            code="merge.invalid_generated_likelihood",
            context={**context, "field": field, "value": value},
        )


def _validate_selected_replay_row_shape(
    row: dict[str, Any],
    *,
    row_position: int,
) -> tuple[str, str]:
    row_id = _require_artifact_str(
        row.get("row_id"),
        artifact_name=TOKEN_TRACE_NAME,
        field="row_id",
        row_position=row_position,
    )
    object_span_id = _require_artifact_str(
        row.get("object_span_id"),
        artifact_name=TOKEN_TRACE_NAME,
        field="object_span_id",
        row_id=row_id,
        row_position=row_position,
    )
    selected_count = _require_artifact_int(
        row.get("selected_count"),
        artifact_name=TOKEN_TRACE_NAME,
        field="selected_count",
        row_id=row_id,
        row_position=row_position,
        object_span_id=object_span_id,
        minimum=1,
    )
    generated_step_indices = _require_artifact_list(
        row.get("generated_step_indices"),
        artifact_name=TOKEN_TRACE_NAME,
        field="generated_step_indices",
        row_id=row_id,
        row_position=row_position,
        object_span_id=object_span_id,
    )
    token_ids = _require_artifact_list(
        row.get("token_ids"),
        artifact_name=TOKEN_TRACE_NAME,
        field="token_ids",
        row_id=row_id,
        row_position=row_position,
        object_span_id=object_span_id,
    )
    token_text = _require_artifact_list(
        row.get("token_text"),
        artifact_name=TOKEN_TRACE_NAME,
        field="token_text",
        row_id=row_id,
        row_position=row_position,
        object_span_id=object_span_id,
    )
    selected_logprobs = _require_artifact_list(
        row.get("selected_logprobs"),
        artifact_name=TOKEN_TRACE_NAME,
        field="selected_logprobs",
        row_id=row_id,
        row_position=row_position,
        object_span_id=object_span_id,
    )
    lengths = {
        "generated_step_indices": len(generated_step_indices),
        "token_ids": len(token_ids),
        "token_text": len(token_text),
        "selected_logprobs": len(selected_logprobs),
    }
    if any(length != selected_count for length in lengths.values()):
        raise ArtifactContractError(
            "selected-token replay field lengths must match selected_count",
            code="merge.replay_shape_mismatch",
            context={
                "artifact": TOKEN_TRACE_NAME,
                "row_id": row_id,
                "row_position": row_position,
                "object_span_id": object_span_id,
                "selected_count": selected_count,
                "lengths": lengths,
            },
        )
    for index, value in enumerate(generated_step_indices):
        _require_artifact_int(
            value,
            artifact_name=TOKEN_TRACE_NAME,
            field=f"generated_step_indices[{index}]",
            row_id=row_id,
            row_position=row_position,
            object_span_id=object_span_id,
            minimum=0,
        )
    for index, value in enumerate(token_ids):
        _require_artifact_int(
            value,
            artifact_name=TOKEN_TRACE_NAME,
            field=f"token_ids[{index}]",
            row_id=row_id,
            row_position=row_position,
            object_span_id=object_span_id,
            minimum=0,
        )
    for index, value in enumerate(token_text):
        _require_artifact_str(
            value,
            artifact_name=TOKEN_TRACE_NAME,
            field=f"token_text[{index}]",
            row_id=row_id,
            row_position=row_position,
            object_span_id=object_span_id,
        )
    for index, value in enumerate(selected_logprobs):
        _require_artifact_float(
            value,
            artifact_name=TOKEN_TRACE_NAME,
            field=f"selected_logprobs[{index}]",
            row_id=row_id,
            row_position=row_position,
            object_span_id=object_span_id,
        )
    return row_id, object_span_id


def _sort_sidecar_rows(
    rows: list[dict[str, Any]],
    *,
    expected_index_by_row_id: dict[str, int],
    artifact_name: str,
) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[int, str, int, str]:
        row_id = _require_artifact_str(
            row.get("row_id"),
            artifact_name=artifact_name,
            field="row_id",
        )
        return (
            expected_index_by_row_id.get(row_id, 10**12),
            str(row.get("trace_type") or row.get("diagnostic_type") or ""),
            _optional_artifact_int(
                row.get("generated_step_index"),
                artifact_name=artifact_name,
                field="generated_step_index",
                row_id=row_id,
                default=-1,
            ),
            str(row.get("object_span_id") or row.get("code") or ""),
        )

    return sorted(rows, key=key)


def _validate_replay_scores(*, scored_jsonl: Path, token_trace_jsonl: Path) -> None:
    try:
        recomputed = recompute_scores_from_artifacts(
            scored_jsonl=scored_jsonl,
            token_trace_jsonl=token_trace_jsonl,
        )
    except ArtifactContractError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactContractError(
            "merged replay score evidence is semantically malformed",
            code="merge.invalid_replay_field",
            context={
                "scored_jsonl": str(scored_jsonl),
                "token_trace_jsonl": str(token_trace_jsonl),
            },
            cause=exc,
        ) from exc
    scored_rows = _read_jsonl(scored_jsonl)
    for row_position, row in enumerate(scored_rows):
        row_id = _require_artifact_str(
            row.get("row_id"),
            artifact_name=SCORED_NAME,
            field="row_id",
            row_position=row_position,
        )
        predictions = _require_artifact_list(
            row.get("pred", []),
            artifact_name=SCORED_NAME,
            field="pred",
            row_id=row_id,
            row_position=row_position,
        )
        for pred_index, pred_payload in enumerate(predictions):
            pred = _require_artifact_dict(
                pred_payload,
                artifact_name=SCORED_NAME,
                field=f"pred[{pred_index}]",
                row_id=row_id,
                row_position=row_position,
                pred_index=pred_index,
            )
            object_span_id = _require_artifact_str(
                pred.get("object_span_id"),
                artifact_name=SCORED_NAME,
                field="object_span_id",
                row_id=row_id,
                row_position=row_position,
                pred_index=pred_index,
            )
            key = (row_id, object_span_id)
            observed = _require_artifact_float(
                pred.get("score"),
                artifact_name=SCORED_NAME,
                field="score",
                row_id=row_id,
                row_position=row_position,
                pred_index=pred_index,
                object_span_id=object_span_id,
                minimum=0.0,
                maximum=1.0,
            )
            expected = recomputed[key]
            if not math.isclose(observed, expected, rel_tol=1e-7, abs_tol=1e-8):
                raise ArtifactContractError(
                    "merged scored prediction score disagrees with replay evidence",
                    code="merge.score_replay_mismatch",
                    context={
                        "row_id": row_id,
                        "object_span_id": key[1],
                        "observed": observed,
                        "expected": expected,
                    },
                )


def _merged_parallelism(
    *,
    metadata: dict[str, Any],
    plan: DataParallelPlan,
    shard_evidence: list[ShardEvidence],
    expected_row_ids: tuple[str, ...],
    merged_hashes: dict[str, str],
) -> dict[str, Any]:
    base = dict(metadata.get("parallelism") or {})
    return {
        **base,
        "execution_mode": "controller_worker",
        "merge_status": "completed",
        "active_ranks": plan.active_ranks,
        "visible_cuda_tokens": list(plan.visible_cuda_tokens),
        "rank_to_device": {
            str(rank.rank): rank.parent_visible_device_token for rank in plan.ranks
        },
        "per_device_batch_size": plan.per_device_batch_size,
        "shard_plan_fingerprint": plan.fingerprint,
        "worker_exit_statuses": {
            str(shard.rank): shard.worker_status for shard in shard_evidence
        },
        "row_coverage": {
            "row_count": len(expected_row_ids),
            "row_ids": list(expected_row_ids),
            "row_ids_sha256": _sha256_json(list(expected_row_ids)),
        },
        "shards": [shard.to_artifact_dict() for shard in sorted(shard_evidence, key=lambda item: item.rank)],
        "merged_artifacts": dict(sorted(merged_hashes.items())),
    }


def _merged_provenance(
    *,
    metadata: dict[str, Any],
    raw_sha: str,
    scored_sha: str,
    expected_row_ids: tuple[str, ...],
    parallelism: dict[str, Any],
) -> dict[str, Any]:
    return {
        "artifact_schema_version": int(metadata.get("artifact_schema_version", 1)),
        "raw_artifact": {"path": RAW_NAME, "sha256": raw_sha},
        "scored_artifact": {"path": SCORED_NAME, "sha256": scored_sha},
        "detection_template_id": metadata["detection_template_id"],
        "prompt_policy_fingerprint": metadata["prompt_policy_fingerprint"],
        "decode_policy_fingerprint": metadata["generation_config_fingerprint"],
        "generation_config_fingerprint": metadata["generation_config_fingerprint"],
        "generation_policy": dict(metadata.get("generation_policy") or {}),
        "backend_session": dict(metadata.get("backend_session") or {}),
        "likelihood_semantics": dict(metadata.get("likelihood_semantics") or {}),
        "execution_model_identity": metadata.get("execution_model_identity"),
        "frontend_identity": dict(metadata.get("frontend_identity") or {}),
        "raw_model_logprob_status": metadata.get("raw_model_logprob_status"),
        "prompt_trace": list(metadata.get("prompt_trace") or []),
        "raw_replay_trace": list(metadata.get("raw_replay_trace") or []),
        "parallelism": parallelism,
        "model_identity": dict(metadata.get("model_identity") or {}),
        "model_identity_fingerprint": metadata["model_identity_fingerprint"],
        "processor_identity": dict(metadata.get("processor_identity") or {}),
        "processor_identity_fingerprint": metadata["processor_identity_fingerprint"],
        "tokenizer_identity": dict(metadata.get("tokenizer_identity") or {}),
        "adapter_identity": metadata.get("adapter_identity"),
        "embedding_delta_identity": metadata.get("embedding_delta_identity"),
        "template_identity": metadata["template_identity"],
        "parser_policy": metadata["parser_policy"],
        "score_policy_fingerprint": SCORE_POLICY_FINGERPRINT,
        "row_binding": {
            "row_count": len(expected_row_ids),
            "row_ids_sha256": _sha256_json(list(expected_row_ids)),
        },
    }


def _merged_summary(
    *,
    raw_rows: list[dict[str, Any]],
    scored_rows: list[dict[str, Any]],
    token_trace_rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    decode_stop_reasons: dict[str, int] = {}
    for row in raw_rows:
        reason = str(row.get("decode_stop_reason", ""))
        decode_stop_reasons[reason] = decode_stop_reasons.get(reason, 0) + 1
    parser_failure_count = sum(
        1
        for row in raw_rows
        if row.get("parse_status") not in {"accepted", "accepted_with_drops"}
    )
    score_failure_count = sum(
        1 for row in diagnostic_rows if row.get("diagnostic_type") == "scoring"
    )
    summary = {
        "terminal_status": "completed",
        "row_count": len(scored_rows),
        "raw_row_count": len(raw_rows),
        "scored_row_count": len(scored_rows),
        "scoreable_prediction_count": sum(
            len(row.get("pred", [])) for row in scored_rows
        ),
        "diagnostic_row_count": len(diagnostic_rows),
        "trace_row_count": len(token_trace_rows),
        "scored_artifact_materialized": True,
        "benchmark_eligible": bool(metadata.get("benchmark_eligible", False)),
        "generation_policy": dict(metadata.get("generation_policy") or {}),
        "likelihood_semantics": dict(metadata.get("likelihood_semantics") or {}),
        "raw_model_logprob_status": metadata.get("raw_model_logprob_status"),
        "decode_success_count": len(raw_rows),
        "parser_failure_count": parser_failure_count,
        "dropped_prediction_count": sum(
            int(row.get("dropped_prediction_count", 0)) for row in raw_rows
        ),
        "truncated_decode_count": int(decode_stop_reasons.get("length", 0)),
        "decode_stop_reasons": dict(sorted(decode_stop_reasons.items())),
        "image_validation_failure_count": 0,
        "score_failure_count": score_failure_count,
    }
    session = metadata.get("backend_session")
    settings = session.get("effective_settings") if isinstance(session, dict) else None
    performance = settings.get("performance") if isinstance(settings, dict) else None
    if isinstance(performance, dict):
        summary["performance"] = dict(performance)
    return summary


def _merged_manifest(
    *,
    metadata: dict[str, Any],
    summary: dict[str, Any],
    parallelism: dict[str, Any],
    identity_fingerprint: str,
) -> dict[str, Any]:
    return {
        "artifact_schema_version": int(metadata.get("artifact_schema_version", 1)),
        "artifacts": {
            "gt_vs_pred": RAW_NAME,
            "gt_vs_pred_scored": SCORED_NAME,
            "gt_vs_pred_scored_provenance": PROVENANCE_NAME,
            "pred_token_trace": TOKEN_TRACE_NAME,
            "parse_diagnostics": PARSE_DIAGNOSTICS_NAME,
            "image_plan": IMAGE_PLAN_NAME,
            "summary": SUMMARY_NAME,
        },
        "resolved_config_fingerprints": metadata.get("resolved_config_fingerprints", {}),
        "model_identity": metadata.get("model_identity", {}),
        "model_identity_fingerprint": metadata["model_identity_fingerprint"],
        "processor_identity": metadata.get("processor_identity", {}),
        "processor_identity_fingerprint": metadata["processor_identity_fingerprint"],
        "tokenizer_identity": metadata.get("tokenizer_identity", {}),
        "adapter_identity": metadata.get("adapter_identity"),
        "embedding_delta_identity": metadata.get("embedding_delta_identity"),
        "backend": metadata["backend"],
        "backend_mode": metadata["backend_mode"],
        "response_family": metadata["response_family"],
        "dataset_identity": metadata["dataset_identity"],
        "generation_config_fingerprint": metadata["generation_config_fingerprint"],
        "generation_policy": dict(metadata.get("generation_policy") or {}),
        "backend_session": dict(metadata.get("backend_session") or {}),
        "likelihood_semantics": dict(metadata.get("likelihood_semantics") or {}),
        "execution_model_identity": metadata.get("execution_model_identity"),
        "frontend_identity": dict(metadata.get("frontend_identity") or {}),
        "raw_model_logprob_status": metadata.get("raw_model_logprob_status"),
        "prompt_trace": list(metadata.get("prompt_trace") or []),
        "raw_replay_trace": list(metadata.get("raw_replay_trace") or []),
        "parallelism": parallelism,
        "merge_identity_fingerprint": identity_fingerprint,
        "score_policy_fingerprint": SCORE_POLICY_FINGERPRINT,
        "trace_scoring_status": "scored",
        "prompt_policy_fingerprint": metadata["prompt_policy_fingerprint"],
        "template_identity": metadata["template_identity"],
        "parser_policy": metadata["parser_policy"],
        "evaluator_consumer_status": str(
            metadata.get("evaluator_consumer_status", "available_not_run")
        ),
        "scored_artifact_materialized": bool(summary["scored_artifact_materialized"]),
        "benchmark_eligible": bool(summary["benchmark_eligible"]),
        "terminal_status": summary.get("terminal_status", "completed"),
    }


def _write_merge_failure_terminal_status(
    *,
    output_dir: Path,
    metadata: dict[str, Any],
    error: ArtifactContractError,
) -> None:
    _remove_benchmark_artifacts(output_dir)
    failure_metadata = dict(metadata)
    parallelism = dict(failure_metadata.get("parallelism") or {})
    parallelism["merge_status"] = "failed"
    failure_metadata["parallelism"] = parallelism
    write_terminal_status_artifacts(
        output_dir=output_dir,
        metadata=failure_metadata,
        summary={
            "terminal_status": "failed",
            "failure_class": "merge_failure",
            "merge_failure_count": 1,
            "error": {
                "code": error.code,
                "message": error.message,
                "context": error.context,
            },
        },
    )


def _remove_benchmark_artifacts(output_dir: Path) -> None:
    for name in (
        RAW_NAME,
        SCORED_NAME,
        PROVENANCE_NAME,
        TOKEN_TRACE_NAME,
        PARSE_DIAGNOSTICS_NAME,
        IMAGE_PLAN_NAME,
    ):
        path = output_dir / name
        if path.exists():
            path.unlink()
    for name in EVAL_ARTIFACT_NAMES:
        path = output_dir / name
        if path.exists():
            path.unlink()
    for name in EVAL_OUTPUT_DIR_NAMES:
        path = output_dir / name
        if path.exists():
            shutil.rmtree(path)


def _paths(output_dir: Path) -> InferenceArtifactPaths:
    return InferenceArtifactPaths(
        output_dir=output_dir,
        raw_jsonl=output_dir / RAW_NAME,
        scored_jsonl=output_dir / SCORED_NAME,
        provenance_json=output_dir / PROVENANCE_NAME,
        token_trace_jsonl=output_dir / TOKEN_TRACE_NAME,
        parse_diagnostics_jsonl=output_dir / PARSE_DIAGNOSTICS_NAME,
        image_plan_jsonl=output_dir / IMAGE_PLAN_NAME,
        summary_json=output_dir / SUMMARY_NAME,
        run_manifest_json=output_dir / MANIFEST_NAME,
    )


def _publish_staged_artifacts(*, staged: InferenceArtifactPaths, final: InferenceArtifactPaths) -> None:
    pairs = [
        (staged.raw_jsonl, final.raw_jsonl),
        (staged.scored_jsonl, final.scored_jsonl),
        (staged.provenance_json, final.provenance_json),
        (staged.token_trace_jsonl, final.token_trace_jsonl),
        (staged.parse_diagnostics_jsonl, final.parse_diagnostics_jsonl),
        (staged.image_plan_jsonl, final.image_plan_jsonl),
        (staged.summary_json, final.summary_json),
        (staged.run_manifest_json, final.run_manifest_json),
    ]
    backup_dir = Path(tempfile.mkdtemp(prefix=".merge-artifact-backup-", dir=final.output_dir))
    backups: dict[Path, Path | None] = {}
    replaced: list[Path] = []
    try:
        for _, final_path in pairs:
            if final_path.exists():
                backup_path = backup_dir / final_path.name
                os.replace(final_path, backup_path)
                backups[final_path] = backup_path
            else:
                backups[final_path] = None
        for staged_path, final_path in pairs:
            os.replace(staged_path, final_path)
            replaced.append(final_path)
    except BaseException as exc:
        for _, final_path in pairs:
            try:
                if final_path.exists():
                    final_path.unlink()
            except OSError:
                pass
        for final_path, backup_path in backups.items():
            if backup_path is None:
                continue
            try:
                shutil.move(str(backup_path), str(final_path))
            except OSError:
                pass
        if isinstance(exc, OSError):
            raise ArtifactContractError(
                "failed to publish complete merged inference artifact set",
                code="merge.publish_failed",
                context={"failed_after": [path.name for path in replaced]},
                cause=exc,
            ) from exc
        raise
    finally:
        shutil.rmtree(backup_dir, ignore_errors=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactContractError(
            "merge artifact JSON is malformed",
            code="merge.json_decode",
            context={
                "path": str(path),
                "line": exc.lineno,
                "column": exc.colno,
            },
            cause=exc,
        ) from exc
    except OSError as exc:
        raise ArtifactContractError(
            "failed to read merge artifact JSON",
            code="merge.artifact_read_failed",
            context={"path": str(path)},
            cause=exc,
        ) from exc
    if not isinstance(payload, dict):
        raise ArtifactContractError(
            "merge artifact JSON must be an object",
            code="merge.json_shape",
            context={"path": str(path), "type": type(payload).__name__},
        )
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ArtifactContractError(
            "failed to read merge artifact JSONL",
            code="merge.artifact_read_failed",
            context={"path": str(path)},
            cause=exc,
        ) from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ArtifactContractError(
                "merge artifact JSONL row is malformed",
                code="merge.json_decode",
                context={
                    "path": str(path),
                    "line": line_number,
                    "column": exc.colno,
                },
                cause=exc,
            ) from exc
        if not isinstance(row, dict):
            raise ArtifactContractError(
                "merge artifact JSONL row must be an object",
                code="merge.json_shape",
                context={
                    "path": str(path),
                    "line": line_number,
                    "type": type(row).__name__,
                },
            )
        rows.append(row)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    _json_safe(row),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str, allow_nan=False))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
