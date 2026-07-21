#!/usr/bin/env python3
"""Run a real, single-GPU vLLM concurrency qualification shard.

This probe intentionally bypasses only the not-yet-promoted ``max_num_seqs``
value.  It still validates the current production qualification envelope at
its accepted concurrency, then delegates the actual run to the production
frontend, pipeline, artifact writer, and vLLM backend session.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from importlib import metadata
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


RECEIPT_SCHEMA_VERSION = 1
RECEIPT_VERSION = "coordexp-swift-vllm-concurrency-qualification-v1"


@dataclass(frozen=True)
class ProbeDependencies:
    """Injectable seams used only to unit-test the qualification harness."""

    load_config: Callable[[Path], Any]
    load_rows: Callable[[str], Sequence[Any]]
    resolve_execution_model: Callable[[Any], Mapping[str, Any] | None]
    validate_execution_model: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    write_resolved_config_artifacts: Callable[[Any, Path], Any]
    write_execution_model_artifact: Callable[..., Any]
    run_shard: Callable[..., int]
    validate_artifact_set: Callable[[Path], None]
    session_opener: Callable[[Any], Any]
    inspect_process_cuda_binding: Callable[[], Mapping[str, Any]]
    package_version: Callable[[str], str]


class _SessionWithReceipt:
    """Delegate to the real session while exposing probe qualification scope."""

    def __init__(self, session: Any, receipt: Any) -> None:
        self._session = session
        self._runtime_qualification = dict(
            receipt.effective_settings["runtime_qualification"]
        )

    @property
    def receipt(self) -> Any:
        live_receipt = self._session.receipt
        return replace(
            live_receipt,
            effective_settings={
                **dict(live_receipt.effective_settings),
                "runtime_qualification": self._runtime_qualification,
            },
        )

    def decode(self, requests: Sequence[Any]) -> Sequence[Any]:
        return self._session.decode(requests)

    def close(self) -> None:
        self._session.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = run_probe(
        config_path=args.config,
        output_dir=args.output_dir,
        receipt_path=args.receipt,
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt": str(args.receipt.expanduser().resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


def run_probe(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    receipt_path: str | Path,
    dependencies: ProbeDependencies | None = None,
) -> dict[str, Any]:
    """Execute one direct rank-local shard and publish a passed receipt last."""

    deps = dependencies or _production_dependencies()
    config_path = Path(config_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    receipt_path = Path(receipt_path).expanduser().resolve()
    _validate_destination_paths(output_dir=output_dir, receipt_path=receipt_path)

    resolved = deps.load_config(config_path)
    config = resolved.config
    _validate_probe_config(config)
    config_sources = _validated_config_sources(resolved)
    input_jsonl = Path(config.data.input_jsonl).expanduser().resolve()
    input_sha256_before = _sha256_file(input_jsonl)
    all_rows = tuple(deps.load_rows(str(input_jsonl)))
    batch_size = int(config.generation.batch_size)
    if len(all_rows) < batch_size:
        _fail(
            "the concurrency source must contain at least generation.batch_size rows",
            code="vllm_concurrency.input_row_count",
            context={
                "row_count": len(all_rows),
                "generation.batch_size": batch_size,
            },
        )
    rows = all_rows[:batch_size]
    request_ids = tuple(str(row.example_id) for row in rows)
    if len(set(request_ids)) != len(request_ids):
        _fail(
            "the concurrency fixture must contain unique request ids",
            code="vllm_concurrency.request_ids",
            context={"request_ids": list(request_ids)},
        )

    process_cuda_binding = dict(deps.inspect_process_cuda_binding())
    execution_model = deps.resolve_execution_model(resolved)
    if not isinstance(execution_model, Mapping):
        _fail(
            "the vLLM concurrency probe requires a production execution-model receipt",
            code="vllm_concurrency.execution_model_missing",
        )
    validated_execution_model = dict(deps.validate_execution_model(execution_model))

    output_dir.mkdir(parents=True, exist_ok=False)
    deps.write_resolved_config_artifacts(resolved, output_dir)
    deps.write_execution_model_artifact(
        run_dir=output_dir,
        execution_model=validated_execution_model,
    )
    deps.run_shard(
        resolved=resolved,
        output_dir=output_dir,
        row_indices=tuple(range(batch_size)),
        worker_metadata=_rank_local_worker_metadata(
            process_cuda_binding=process_cuda_binding,
            batch_size=batch_size,
            request_ids=request_ids,
        ),
        session_opener=deps.session_opener,
        execution_model=validated_execution_model,
    )

    deps.validate_artifact_set(output_dir)
    post_execution_model = dict(
        deps.validate_execution_model(validated_execution_model)
    )
    if post_execution_model != validated_execution_model:
        _fail(
            "execution-model identity changed during the concurrency probe",
            code="vllm_concurrency.execution_model_drift",
        )
    if _validated_config_sources(resolved) != config_sources:
        _fail(
            "resolved config sources changed during the concurrency probe",
            code="vllm_concurrency.config_source_drift",
        )
    input_sha256_after = _sha256_file(input_jsonl)
    if input_sha256_after != input_sha256_before:
        _fail(
            "input JSONL changed during the concurrency probe",
            code="vllm_concurrency.input_source_drift",
            context={
                "before": input_sha256_before,
                "after": input_sha256_after,
            },
        )

    receipt = build_passed_receipt(
        resolved=resolved,
        output_dir=output_dir,
        execution_model=validated_execution_model,
        expected_request_ids=request_ids,
        config_sources=config_sources,
        input_jsonl=input_jsonl,
        input_jsonl_sha256=input_sha256_after,
        input_row_count=len(all_rows),
        process_cuda_binding=process_cuda_binding,
        vllm_version=deps.package_version("vllm"),
        probe_source_sha256=_sha256_file(Path(__file__).resolve()),
    )
    _atomic_write_json(receipt_path, receipt)
    return receipt


def build_passed_receipt(
    *,
    resolved: Any,
    output_dir: Path,
    execution_model: Mapping[str, Any],
    expected_request_ids: Sequence[str],
    config_sources: Sequence[Mapping[str, str]],
    input_jsonl: Path,
    input_jsonl_sha256: str,
    input_row_count: int,
    process_cuda_binding: Mapping[str, Any],
    vllm_version: str,
    probe_source_sha256: str,
) -> dict[str, Any]:
    """Construct deterministic evidence from a terminal production artifact set."""

    manifest = _read_json(output_dir / "run_manifest.json")
    summary = _read_json(output_dir / "summary.json")
    raw_rows = _read_jsonl(output_dir / "gt_vs_pred.jsonl")
    trace_rows = _read_jsonl(output_dir / "pred_token_trace.jsonl")
    image_plan_rows = _read_jsonl(output_dir / "image_plan.jsonl")

    if summary.get("terminal_status") != "completed" or not bool(
        manifest.get("scored_artifact_materialized")
    ):
        _fail(
            "concurrency receipt requires terminal completed inference artifacts",
            code="vllm_concurrency.artifacts_not_completed",
        )
    if manifest.get("backend") != "vllm":
        _fail(
            "concurrency artifacts were not produced by the vLLM backend",
            code="vllm_concurrency.artifact_backend",
            context={"backend": manifest.get("backend")},
        )

    request_ids = [str(row.get("row_id")) for row in raw_rows]
    expected_ids = [str(value) for value in expected_request_ids]
    if request_ids != expected_ids:
        _fail(
            "artifact request order differs from the submitted fixture order",
            code="vllm_concurrency.request_order",
            context={"expected": expected_ids, "observed": request_ids},
        )
    if int(summary.get("row_count", -1)) != len(expected_ids):
        _fail(
            "summary row count differs from the submitted fixture",
            code="vllm_concurrency.summary_row_count",
        )

    backend_session = _require_mapping(
        manifest.get("backend_session"),
        field="run_manifest.backend_session",
    )
    observed_version = str(backend_session.get("backend_version") or "")
    if observed_version != vllm_version:
        _fail(
            "artifact vLLM version differs from the installed package version",
            code="vllm_concurrency.vllm_version",
            context={"artifact": observed_version, "installed": vllm_version},
        )
    effective_settings = _require_mapping(
        backend_session.get("effective_settings"),
        field="run_manifest.backend_session.effective_settings",
    )
    engine_kwargs = dict(
        _require_mapping(
            effective_settings.get("engine_kwargs"),
            field="run_manifest.backend_session.effective_settings.engine_kwargs",
        )
    )
    prompt_trace = manifest.get("prompt_trace")
    if not isinstance(prompt_trace, list):
        _fail(
            "manifest prompt trace is missing",
            code="vllm_concurrency.prompt_trace",
        )
    prompt_trace = [
        dict(_require_mapping(row, field="run_manifest.prompt_trace[]"))
        for row in prompt_trace
    ]
    if [str(row.get("row_id")) for row in prompt_trace] != expected_ids:
        _fail(
            "manifest prompt trace differs from request order",
            code="vllm_concurrency.prompt_trace",
        )
    max_num_seqs = engine_kwargs.get("max_num_seqs")
    if max_num_seqs != len(expected_ids) or max_num_seqs <= 0:
        _fail(
            "engine max_num_seqs does not match the concurrent request count",
            code="vllm_concurrency.max_num_seqs",
            context={
                "max_num_seqs": max_num_seqs,
                "request_count": len(expected_ids),
            },
        )

    artifact_execution_model = backend_session.get("execution_model_identity")
    if artifact_execution_model != execution_model:
        _fail(
            "backend artifacts do not bind the validated execution-model receipt",
            code="vllm_concurrency.execution_model_artifact_mismatch",
        )

    raw_status = str(summary.get("raw_model_logprob_status") or "")
    if raw_status not in {"available", "disabled"} or raw_status != str(
        manifest.get("raw_model_logprob_status") or ""
    ):
        _fail(
            "raw-likelihood status is missing or inconsistent",
            code="vllm_concurrency.raw_likelihood_status",
        )
    raw_replay_trace = manifest.get("raw_replay_trace")
    if not isinstance(raw_replay_trace, list):
        _fail(
            "manifest raw replay trace is missing",
            code="vllm_concurrency.raw_replay_trace",
        )
    raw_replay_trace = [
        dict(_require_mapping(row, field="run_manifest.raw_replay_trace[]"))
        for row in raw_replay_trace
    ]
    if raw_status == "available":
        if [str(row.get("row_id")) for row in raw_replay_trace] != expected_ids:
            _fail(
                "manifest raw replay trace differs from request order",
                code="vllm_concurrency.raw_replay_trace",
            )
        raw_replay_settings = _require_mapping(
            effective_settings.get("raw_replay"),
            field="run_manifest.backend_session.effective_settings.raw_replay",
        )
        if raw_replay_settings.get("status") != "completed":
            _fail(
                "raw replay receipt is not completed",
                code="vllm_concurrency.raw_replay_receipt",
            )
        renewal = _require_mapping(
            raw_replay_settings.get("qualification"),
            field=(
                "run_manifest.backend_session.effective_settings.raw_replay."
                "qualification"
            ),
        )
        if renewal.get("status") != "qualification_probe_under_renewal":
            _fail(
                "raw replay did not use the qualification-renewal boundary",
                code="vllm_concurrency.raw_replay_receipt",
            )
        raw_replay_settings = {
            **dict(raw_replay_settings),
            "qualification": {
                **dict(renewal),
                "status": "passed",
                "evidence": "executed_by_this_receipt",
            },
        }
        raw_replay_by_id = {
            str(row["row_id"]): {
                key: value for key, value in row.items() if key != "row_id"
            }
            for row in raw_replay_trace
        }
        if (
            raw_replay_settings.get("request_count") != len(expected_ids)
            or raw_replay_settings.get("row_evidence_sha256")
            != _sha256_json(raw_replay_by_id)
        ):
            _fail(
                "raw replay receipt disagrees with row evidence",
                code="vllm_concurrency.raw_replay_receipt",
            )
    else:
        raw_replay_settings = {}
        if raw_replay_trace:
            _fail(
                "disabled raw replay unexpectedly emitted row evidence",
                code="vllm_concurrency.raw_replay_trace",
            )

    generated_by_id: dict[str, list[dict[str, Any]]] = {
        request_id: [] for request_id in expected_ids
    }
    for row in trace_rows:
        if row.get("trace_type") != "generated_token":
            continue
        request_id = str(row.get("row_id"))
        if request_id not in generated_by_id:
            _fail(
                "token trace contains an unknown request id",
                code="vllm_concurrency.trace_request_id",
                context={"request_id": request_id},
            )
        generated_by_id[request_id].append(row)

    image_plan_by_id = {str(row.get("row_id")): row for row in image_plan_rows}
    if (
        len(image_plan_rows) != len(expected_ids)
        or list(image_plan_by_id) != expected_ids
    ):
        _fail(
            "image-plan evidence differs from the submitted request order",
            code="vllm_concurrency.image_plan_order",
        )

    requests: list[dict[str, Any]] = []
    for raw_row in raw_rows:
        request_id = str(raw_row["row_id"])
        generated_rows = generated_by_id[request_id]
        steps = [int(row.get("generated_step_index", -1)) for row in generated_rows]
        if not generated_rows or steps != list(range(len(generated_rows))):
            _fail(
                "generated token evidence is missing or non-contiguous",
                code="vllm_concurrency.generated_token_evidence",
                context={"request_id": request_id, "steps": steps},
            )
        stop_reason = str(raw_row.get("decode_stop_reason") or "")
        if not stop_reason:
            _fail(
                "raw artifact is missing a terminal stop reason",
                code="vllm_concurrency.stop_reason",
                context={"request_id": request_id},
            )
        image_plan = image_plan_by_id[request_id]
        requests.append(
            {
                "request_id": request_id,
                "generated_token_ids": [int(row["token_id"]) for row in generated_rows],
                "stop_reason": stop_reason,
                "prompt_image_plan_evidence": {
                    "image_plan_row_sha256": _sha256_json(image_plan),
                    "image_content_sha256": image_plan.get("image_content_sha256"),
                    "executed_media_sha256": image_plan.get("executed_media_sha256"),
                    "expected_image_grid_thw": image_plan.get(
                        "expected_image_grid_thw"
                    ),
                    "backend_prompt_token_count": image_plan.get(
                        "backend_prompt_token_count"
                    ),
                    "backend_image_placeholder_ranges": image_plan.get(
                        "backend_image_placeholder_ranges"
                    ),
                },
            }
        )

    runtime_qualification = _require_mapping(
        effective_settings.get("runtime_qualification"),
        field="run_manifest.backend_session.effective_settings.runtime_qualification",
    )
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "version": RECEIPT_VERSION,
        "status": "passed",
        "config": {
            "entry_path": str(resolved.entry_config_path),
            "entry_repo_relative_path": _repo_relative_path_or_none(
                Path(resolved.entry_config_path)
            ),
            "fingerprint": str(resolved.fingerprint),
            "sources": [dict(source) for source in config_sources],
        },
        "input_source": {
            "path": str(input_jsonl),
            "sha256": input_jsonl_sha256,
            "row_count": input_row_count,
            "selected_prefix_row_count": len(expected_ids),
        },
        "vllm_version": vllm_version,
        "max_num_seqs": max_num_seqs,
        "request_count": len(expected_ids),
        "request_ids": expected_ids,
        "requests": requests,
        "prompt_trace": {
            "rows": prompt_trace,
            "row_evidence_sha256": _sha256_json(prompt_trace),
        },
        "raw_replay": {
            "settings": dict(raw_replay_settings),
            "rows": raw_replay_trace,
            "row_evidence_sha256": _sha256_json(raw_replay_trace),
        },
        "backend_session_engine_kwargs": engine_kwargs,
        "runtime_qualification": dict(runtime_qualification),
        "execution_model": {
            "mode": execution_model.get("mode"),
            "composition_key": execution_model.get("composition_key"),
            "snapshot_fingerprint": execution_model.get("snapshot_fingerprint"),
            "receipt_fingerprint": execution_model.get("receipt_fingerprint"),
            "source_base_snapshot_fingerprint": (
                execution_model.get("source_identity", {})
                .get("base", {})
                .get("fingerprint")
            ),
        },
        "raw_model_logprob_status": raw_status,
        "artifacts": _artifact_hashes(output_dir),
        "process_cuda_binding": dict(process_cuda_binding),
        "probe_source_sha256": probe_source_sha256,
    }


def _production_dependencies() -> ProbeDependencies:
    from src.config.inference import load_infer_config
    from src.config.writer import write_resolved_config_artifacts
    from src.data import load_raw_examples
    from src.inference.artifacts import validate_scored_artifact_set
    from src.inference.execution_model import validate_execution_model_receipt
    from src.inference.pipeline import (
        _resolve_execution_model_for_run,
        _write_execution_model_artifact,
        run_shard,
    )

    return ProbeDependencies(
        load_config=load_infer_config,
        load_rows=load_raw_examples,
        resolve_execution_model=_resolve_execution_model_for_run,
        validate_execution_model=validate_execution_model_receipt,
        write_resolved_config_artifacts=write_resolved_config_artifacts,
        write_execution_model_artifact=_write_execution_model_artifact,
        run_shard=run_shard,
        validate_artifact_set=validate_scored_artifact_set,
        session_opener=_open_probe_vllm_session,
        inspect_process_cuda_binding=_inspect_process_cuda_binding,
        package_version=metadata.version,
    )


def _open_probe_vllm_session(launch: Any) -> Any:
    """Open the real engine while preserving every already-qualified check."""

    from src.inference.vllm_backend import (
        _engine_kwargs,
        _vllm_options,
        open_vllm_backend_session,
    )
    from src.inference.vllm_qualification import (
        QUALIFIED_MAX_NUM_SEQS,
        validate_vllm_runtime_qualification,
    )

    actual_engine_kwargs = _engine_kwargs(launch, options=_vllm_options(launch))
    if 1 not in QUALIFIED_MAX_NUM_SEQS:
        _fail(
            "production vLLM qualification has no accepted single-sequence baseline",
            code="vllm_concurrency.qualification_baseline",
        )
    # Renewal must not depend on the concurrency receipt it is replacing.
    baseline_max_num_seqs = 1
    baseline_engine_kwargs = {
        **actual_engine_kwargs,
        "max_num_seqs": baseline_max_num_seqs,
    }
    baseline = validate_vllm_runtime_qualification(
        launch=launch,
        engine_kwargs=baseline_engine_kwargs,
    )
    session = open_vllm_backend_session(
        launch,
        engine_factory=_real_default_engine_factory,
        raw_replay_qualifier=lambda current_launch, processor_identity: {
            "status": "qualification_probe_under_renewal",
            "probe_source_sha256": _sha256_file(Path(__file__).resolve()),
            "source_base_snapshot_fingerprint": (
                current_launch.execution_model_identity.get("source_identity", {})
                .get("base", {})
                .get("fingerprint")
            ),
            "processor_source_sha256": processor_identity.get("source_sha256"),
        },
    )
    qualification = {
        "status": "production_baseline_passed_concurrency_under_probe",
        "baseline": baseline,
        "qualified_baseline_max_num_seqs": baseline_max_num_seqs,
        "probed_max_num_seqs": actual_engine_kwargs["max_num_seqs"],
        "override_scope": "max_num_seqs_only",
        "engine_factory": (
            "explicit_wrapper_around_src.inference.vllm_backend._default_engine_factory"
        ),
    }
    effective_settings = {
        **dict(session.receipt.effective_settings),
        "runtime_qualification": qualification,
    }
    receipt = replace(session.receipt, effective_settings=effective_settings)
    receipt.validate_for_launch(launch)
    return _SessionWithReceipt(session, receipt)


def _real_default_engine_factory(kwargs: Mapping[str, object]) -> Any:
    from src.inference.vllm_backend import _default_engine_factory

    return _default_engine_factory(kwargs)


def _validate_probe_config(config: Any) -> None:
    if config.backend.type != "vllm":
        _fail(
            "vLLM concurrency qualification requires backend.type: vllm",
            code="vllm_concurrency.backend",
            context={"backend.type": config.backend.type},
        )
    if int(config.generation.batch_size) <= 0:
        _fail(
            "vLLM runtime qualification requires a positive generation.batch_size",
            code="vllm_concurrency.batch_size",
            context={"generation.batch_size": config.generation.batch_size},
        )
    if config.debug.smoke is not True or config.debug.dry_run is True:
        _fail(
            "vLLM concurrency qualification requires debug.smoke: true and a real run",
            code="vllm_concurrency.smoke",
            context={
                "debug.smoke": config.debug.smoke,
                "debug.dry_run": config.debug.dry_run,
            },
        )


def _validate_destination_paths(*, output_dir: Path, receipt_path: Path) -> None:
    if output_dir.exists():
        _fail(
            "probe output directory must not already exist",
            code="vllm_concurrency.output_exists",
            context={"output_dir": str(output_dir)},
        )
    if receipt_path.exists():
        _fail(
            "probe receipt must not already exist",
            code="vllm_concurrency.receipt_exists",
            context={"receipt": str(receipt_path)},
        )
    try:
        receipt_path.relative_to(output_dir)
    except ValueError:
        return
    _fail(
        "probe receipt must be outside the inference artifact directory",
        code="vllm_concurrency.receipt_inside_output",
    )


def _validated_config_sources(resolved: Any) -> tuple[dict[str, object], ...]:
    sources: list[dict[str, object]] = []
    for source in resolved.sources:
        path = Path(source.path).expanduser().resolve()
        expected = str(source.sha256)
        observed = _sha256_file(path)
        if observed != expected:
            _fail(
                "resolved config source hash no longer matches",
                code="vllm_concurrency.config_source_drift",
                context={
                    "path": str(path),
                    "expected": expected,
                    "observed": observed,
                },
            )
        sources.append(
            {
                "path": str(path),
                "repo_relative_path": _repo_relative_path_or_none(path),
                "sha256": observed,
            }
        )
    if not sources:
        _fail(
            "resolved config did not retain source hashes",
            code="vllm_concurrency.config_sources_missing",
        )
    return tuple(sources)


def _repo_relative_path_or_none(path: Path) -> str | None:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        # External configs are valid probe inputs, but their receipts cannot
        # become reusable runtime qualification evidence.
        return None


def _inspect_process_cuda_binding() -> dict[str, Any]:
    import torch

    from src.inference.data_parallel import resolve_visible_cuda_tokens

    visible_tokens = resolve_visible_cuda_tokens()
    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count())
    current_device = int(torch.cuda.current_device()) if device_count > 0 else None
    if (
        len(visible_tokens) != 1
        or not cuda_available
        or device_count != 1
        or current_device != 0
    ):
        _fail(
            "vLLM concurrency qualification requires exactly one visible logical GPU",
            code="vllm_concurrency.cuda_binding",
            context={
                "visible_cuda_tokens": list(visible_tokens),
                "cuda_available": cuda_available,
                "cuda_device_count": device_count,
                "cuda_current_device": current_device,
            },
        )
    return {
        "process": {"pid": os.getpid(), "ppid": os.getppid()},
        "cuda": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "visible_cuda_tokens": list(visible_tokens),
            "cuda_available": cuda_available,
            "cuda_device_count": device_count,
            "cuda_current_device": current_device,
            "logical_device": "cuda:0",
            "device_name": str(torch.cuda.get_device_name(0)),
            "torch_cuda_version": torch.version.cuda,
        },
    }


def _rank_local_worker_metadata(
    *,
    process_cuda_binding: Mapping[str, Any],
    batch_size: int,
    request_ids: Sequence[str],
) -> dict[str, Any]:
    cuda = _require_mapping(
        process_cuda_binding.get("cuda"),
        field="process_cuda_binding.cuda",
    )
    process = _require_mapping(
        process_cuda_binding.get("process"),
        field="process_cuda_binding.process",
    )
    visible_tokens = list(cuda.get("visible_cuda_tokens") or [])
    return {
        "rank": 0,
        "world_size": 1,
        "parent_visible_device_token": visible_tokens[0],
        "worker_cuda_visible_devices": cuda.get("CUDA_VISIBLE_DEVICES"),
        "worker_logical_device": "cuda:0",
        "cuda_device_count": cuda.get("cuda_device_count"),
        "cuda_current_device": cuda.get("cuda_current_device"),
        "model_first_parameter_device": "cuda:0",
        "per_device_batch_size": batch_size,
        "batch_ids": [0],
        "assigned_request_ids": list(request_ids),
        "pid": process.get("pid"),
        "ppid": process.get("ppid"),
        "probe_process_mode": "fresh_direct_rank_local_shard",
    }


def _artifact_hashes(output_dir: Path) -> list[dict[str, Any]]:
    files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    if not files:
        _fail(
            "completed probe output contains no artifacts",
            code="vllm_concurrency.artifacts_missing",
        )
    return [
        {
            "relative_path": path.relative_to(output_dir).as_posix(),
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in files
    ]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(
            "required probe artifact is unavailable or malformed",
            code="vllm_concurrency.artifact_read",
            context={"path": str(path), "error": type(exc).__name__},
        )
    if not isinstance(value, dict):
        _fail(
            "required probe artifact must be a JSON object",
            code="vllm_concurrency.artifact_shape",
            context={"path": str(path)},
        )
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        values = [json.loads(line) for line in lines]
    except (OSError, json.JSONDecodeError) as exc:
        _fail(
            "required probe JSONL artifact is unavailable or malformed",
            code="vllm_concurrency.artifact_read",
            context={"path": str(path), "error": type(exc).__name__},
        )
    if any(not isinstance(value, dict) for value in values):
        _fail(
            "required probe JSONL artifact must contain JSON objects",
            code="vllm_concurrency.artifact_shape",
            context={"path": str(path)},
        )
    return values


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(
            "probe evidence field must be a mapping",
            code="vllm_concurrency.evidence_shape",
            context={"field": field},
        )
    return value


def _sha256_file(path: Path) -> str:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError as exc:
        _fail(
            "required probe source is unavailable",
            code="vllm_concurrency.source_unavailable",
            context={"path": str(path), "error": type(exc).__name__},
        )


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    published = False
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        published = True
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        if published:
            path.unlink(missing_ok=True)
        raise


def _fail(
    message: str,
    *,
    code: str,
    context: Mapping[str, Any] | None = None,
) -> None:
    from src.common.errors import RuntimeContractError

    raise RuntimeContractError(message, code=code, context=dict(context or {}))


if __name__ == "__main__":
    raise SystemExit(main())
