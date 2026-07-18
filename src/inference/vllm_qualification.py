"""Fail-closed binding to the executed vLLM runtime qualification receipt."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path
from typing import Any

from src.common.errors import RuntimeContractError
from src.inference.backend import BackendLaunch


_REPO_ROOT = Path(__file__).resolve().parents[2]
_RECEIPT_ROOT = Path(__file__).resolve().with_name("qualification_receipts")
QUALIFICATION_RECEIPT = _RECEIPT_ROOT / "vllm-0.14.1-qualification.json"
APPLICATION_QUALIFICATION_RECEIPT = _RECEIPT_ROOT / (
    "vllm-0.14.1-application-sources.json"
)
CONCURRENCY_QUALIFICATION_RECEIPT = QUALIFICATION_RECEIPT.with_name(
    "vllm-0.14.1-concurrency4.json"
)
FORCED_REPLAY_SINGLE_QUALIFICATION_RECEIPT = QUALIFICATION_RECEIPT.with_name(
    "vllm-0.14.1-forced-replay1.json"
)
FP32_QUALIFICATION_RECEIPT = _RECEIPT_ROOT / "vllm-0.14.1-fp32-qualification.json"
FP32_CONCURRENCY_QUALIFICATION_RECEIPT = _RECEIPT_ROOT / (
    "vllm-0.14.1-fp32-concurrency4.json"
)
FP32_FORCED_REPLAY_SINGLE_QUALIFICATION_RECEIPT = _RECEIPT_ROOT / (
    "vllm-0.14.1-fp32-forced-replay1.json"
)
_QUALIFICATION_RECEIPTS = {
    "bf16": QUALIFICATION_RECEIPT,
    "fp32": FP32_QUALIFICATION_RECEIPT,
}
_CONCURRENCY_QUALIFICATION_RECEIPTS = {
    "bf16": CONCURRENCY_QUALIFICATION_RECEIPT,
    "fp32": FP32_CONCURRENCY_QUALIFICATION_RECEIPT,
}
_FORCED_REPLAY_QUALIFICATION_RECEIPTS = {
    ("bf16", 1): FORCED_REPLAY_SINGLE_QUALIFICATION_RECEIPT,
    ("bf16", 4): CONCURRENCY_QUALIFICATION_RECEIPT,
    ("fp32", 1): FP32_FORCED_REPLAY_SINGLE_QUALIFICATION_RECEIPT,
    ("fp32", 4): FP32_CONCURRENCY_QUALIFICATION_RECEIPT,
}
QUALIFICATION_PROBE_RELATIVE_PATH = Path(
    "scripts/probes/coordexp_swift/vllm_qualification.py"
)
QUALIFICATION_PROBE = _REPO_ROOT / QUALIFICATION_PROBE_RELATIVE_PATH
QUALIFIED_MAX_NUM_SEQS = {1, 4}
APPLICATION_EXECUTION_SOURCE_PATHS = (
    "src/adapters/dora.py",
    "src/config/inference.py",
    "src/inference/artifacts.py",
    "src/inference/backend.py",
    "src/inference/data_parallel.py",
    "src/inference/execution_model.py",
    "src/inference/execution_model_composition.py",
    "src/inference/hf_backend.py",
    "src/inference/image_plan.py",
    "src/inference/merge.py",
    "src/inference/model_assets.py",
    "src/inference/parsing.py",
    "src/inference/pipeline.py",
    "src/inference/prompt.py",
    "src/inference/runtime.py",
    "src/inference/scoring.py",
    "src/inference/vllm_backend.py",
    "src/inference/vllm_forced_replay.py",
    "src/inference/vllm_qualification.py",
    "src/inference/worker.py",
    "src/qwen/images.py",
    "src/qwen/runtime_loading.py",
    "src/qwen/special_token_embeddings.py",
)


def validate_vllm_runtime_qualification(
    *,
    launch: BackendLaunch,
    engine_kwargs: Mapping[str, object],
    receipt_path: str | Path | None = None,
    concurrency_receipt_path: str | Path | None = None,
    application_receipt_path: str | Path | None = None,
) -> dict[str, object]:
    path = _select_qualification_receipt(
        launch=launch,
        explicit_path=receipt_path,
        receipts=_QUALIFICATION_RECEIPTS,
        kind="runtime",
    )
    concurrency_path = _select_qualification_receipt(
        launch=launch,
        explicit_path=concurrency_receipt_path,
        receipts=_CONCURRENCY_QUALIFICATION_RECEIPTS,
        kind="concurrency",
    )
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeContractError(
            "vLLM runtime qualification receipt is unavailable",
            code="vllm_backend.qualification_receipt",
            context={"path": str(path)},
            cause=exc,
        ) from exc
    if not isinstance(payload, dict) or payload.get("status") != "passed":
        _fail("status", payload.get("status") if isinstance(payload, dict) else None)
    if payload.get("candidate_version") != "0.14.1":
        _fail("candidate_version", payload.get("candidate_version"))
    _validate_qualification_probe(payload.get("probe"))

    execution = launch.execution_model_identity
    if not isinstance(execution, Mapping):
        _fail("execution_model_identity", None)
    source = execution.get("source_identity")
    base = source.get("base") if isinstance(source, Mapping) else None
    base_fingerprint = base.get("fingerprint") if isinstance(base, Mapping) else None
    scope = payload.get("qualification_scope")
    qualified_base = (
        scope.get("source_base_snapshot_fingerprint")
        if isinstance(scope, Mapping)
        else None
    )
    if base_fingerprint != qualified_base:
        raise RuntimeContractError(
            "execution model does not derive from the runtime-qualified base",
            code="vllm_backend.qualification_model_family",
            context={
                "expected_base_fingerprint": qualified_base,
                "observed_base_fingerprint": base_fingerprint,
            },
        )
    if execution.get("mode") == "materialized" and not isinstance(
        execution.get("composition_fidelity"),
        Mapping,
    ):
        _fail("execution_model_identity.composition_fidelity", None)

    engine = payload.get("engine")
    policy = (
        engine.get("qualification_argument_policy")
        if isinstance(engine, Mapping)
        else None
    )
    invariants = (
        policy.get("semantic_invariants") if isinstance(policy, Mapping) else None
    )
    if not isinstance(invariants, Mapping):
        _fail("engine.qualification_argument_policy.semantic_invariants", invariants)
    for field in (
        "tensor_parallel_size",
        "data_parallel_size",
        "dtype",
        "logprobs_mode",
        "generation_config",
        "limit_mm_per_prompt",
        "mm_processor_kwargs",
    ):
        if engine_kwargs.get(field) != invariants.get(field):
            raise RuntimeContractError(
                "vLLM engine argument differs from its qualified semantic envelope",
                code="vllm_backend.qualification_engine_argument",
                context={
                    "field": field,
                    "expected": invariants.get(field),
                    "observed": engine_kwargs.get(field),
                },
            )
    exact = policy.get("qualified_exact_values")
    if not isinstance(exact, Mapping):
        _fail("engine.qualification_argument_policy.qualified_exact_values", exact)
    for field in ("gpu_memory_utilization", "max_model_len"):
        if engine_kwargs.get(field) != exact.get(field):
            raise RuntimeContractError(
                "vLLM engine argument differs from its qualified value",
                code="vllm_backend.qualification_engine_argument",
                context={
                    "field": field,
                    "expected": exact.get(field),
                    "observed": engine_kwargs.get(field),
                },
            )
    max_num_seqs = engine_kwargs.get("max_num_seqs")
    if max_num_seqs not in QUALIFIED_MAX_NUM_SEQS:
        raise RuntimeContractError(
            "vLLM batch concurrency has no accepted executed qualification",
            code="vllm_backend.qualification_max_num_seqs",
            context={
                "observed": max_num_seqs,
                "qualified": sorted(QUALIFIED_MAX_NUM_SEQS),
            },
        )

    _validate_source_identities(payload.get("installed_sources"))
    _validate_loaded_runtime_sources(payload.get("loaded_runtime_sources"))
    application_qualification = _validate_application_sources(
        _application_receipt_path(
            baseline_receipt_path=path,
            explicit_path=application_receipt_path,
        )
    )
    result = {
        "status": "passed",
        "receipt_path": str(path),
        "receipt_sha256": hashlib.sha256(raw).hexdigest(),
        "candidate_version": payload["candidate_version"],
        "model_dtype": launch.model_dtype,
        "source_base_snapshot_fingerprint": qualified_base,
        "max_num_seqs": max_num_seqs,
        "application_qualification": application_qualification,
    }
    if max_num_seqs != 1:
        result["concurrency_qualification"] = _validate_concurrency_qualification(
            launch=launch,
            engine_kwargs=engine_kwargs,
            baseline_receipt_sha256=result["receipt_sha256"],
            receipt_path=concurrency_path,
        )
    return result


def validate_vllm_forced_replay_qualification(
    *,
    launch: BackendLaunch,
    processor_identity: Mapping[str, object],
    receipt_path: str | Path | None = None,
    baseline_receipt_path: str | Path | None = None,
) -> dict[str, object]:
    """Bind raw generation replay to the executed forced-decode probe."""

    selected_path = receipt_path or _FORCED_REPLAY_QUALIFICATION_RECEIPTS.get(
        (launch.model_dtype, launch.batch_size)
    )
    if selected_path is None:
        raise RuntimeContractError(
            "vLLM raw replay concurrency has no executed qualification",
            code="vllm_backend.raw_replay_qualification_max_num_seqs",
            context={
                "observed": launch.batch_size,
                "qualified": sorted(
                    batch
                    for dtype, batch in _FORCED_REPLAY_QUALIFICATION_RECEIPTS
                    if dtype == launch.model_dtype
                ),
                "model_dtype": launch.model_dtype,
            },
        )
    path = Path(selected_path).resolve()
    baseline_path = _select_qualification_receipt(
        launch=launch,
        explicit_path=baseline_receipt_path,
        receipts=_QUALIFICATION_RECEIPTS,
        kind="runtime",
    )
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
        baseline_sha256 = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeContractError(
            "vLLM forced-replay qualification receipt is unavailable",
            code="vllm_backend.raw_replay_qualification_receipt",
            context={"path": str(path)},
            cause=exc,
        ) from exc
    if not isinstance(payload, Mapping) or payload.get("status") != "passed":
        _fail_raw_replay(
            "status", payload.get("status") if isinstance(payload, Mapping) else None
        )
    if payload.get("version") != "coordexp-swift-vllm-concurrency-qualification-v1":
        _fail_raw_replay("version", payload.get("version"))
    if payload.get("vllm_version") != "0.14.1":
        _fail_raw_replay("vllm_version", payload.get("vllm_version"))
    if (
        payload.get("max_num_seqs") != launch.batch_size
        or payload.get("request_count") != launch.batch_size
    ):
        _fail_raw_replay(
            "max_num_seqs",
            {
                "launch": launch.batch_size,
                "receipt": payload.get("max_num_seqs"),
                "request_count": payload.get("request_count"),
            },
        )

    runtime = payload.get("runtime_qualification")
    baseline = runtime.get("baseline") if isinstance(runtime, Mapping) else None
    if not isinstance(baseline, Mapping) or baseline.get("receipt_sha256") != baseline_sha256:
        _fail_raw_replay(
            "runtime_qualification.baseline.receipt_sha256",
            baseline.get("receipt_sha256") if isinstance(baseline, Mapping) else None,
        )

    probe_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "probes"
        / "coordexp_swift"
        / "vllm_concurrency.py"
    )
    observed_probe_sha256 = _sha256_file(probe_path)
    if payload.get("probe_source_sha256") != observed_probe_sha256:
        raise RuntimeContractError(
            "vLLM forced-replay qualification probe has drifted",
            code="vllm_backend.raw_replay_qualification_source_drift",
            context={
                "path": str(probe_path),
                "expected": payload.get("probe_source_sha256"),
                "observed": observed_probe_sha256,
            },
        )
    _validate_config_sources(payload.get("config"))

    execution = launch.execution_model_identity
    source = execution.get("source_identity") if isinstance(execution, Mapping) else None
    base = source.get("base") if isinstance(source, Mapping) else None
    base_fingerprint = base.get("fingerprint") if isinstance(base, Mapping) else None
    receipt_execution = payload.get("execution_model")
    qualified_base = (
        receipt_execution.get("source_base_snapshot_fingerprint")
        if isinstance(receipt_execution, Mapping)
        else None
    )
    if not isinstance(qualified_base, str) or base_fingerprint != qualified_base:
        raise RuntimeContractError(
            "execution model does not derive from the forced-replay-qualified base",
            code="vllm_backend.raw_replay_qualification_model_family",
            context={
                "expected_base_fingerprint": qualified_base,
                "observed_base_fingerprint": base_fingerprint,
            },
        )

    replay = payload.get("raw_replay")
    settings = replay.get("settings") if isinstance(replay, Mapping) else None
    rows = replay.get("rows") if isinstance(replay, Mapping) else None
    if not isinstance(settings, Mapping) or settings.get("status") != "completed":
        _fail_raw_replay("raw_replay.settings.status", settings)
    if settings.get("logprobs_mode") != "raw_logprobs":
        _fail_raw_replay(
            "raw_replay.settings.logprobs_mode", settings.get("logprobs_mode")
        )
    if settings.get("max_num_seqs") != launch.batch_size:
        _fail_raw_replay(
            "raw_replay.settings.max_num_seqs",
            settings.get("max_num_seqs"),
        )
    if not isinstance(rows, list) or not rows:
        _fail_raw_replay("raw_replay.rows", rows)
    if replay.get("row_evidence_sha256") != _sha256_json(rows):
        _fail_raw_replay(
            "raw_replay.row_evidence_sha256", replay.get("row_evidence_sha256")
        )
    row_map: dict[str, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or row.get("status") != "verified":
            _fail_raw_replay("raw_replay.rows[].status", row)
        row_id = row.get("row_id")
        if not isinstance(row_id, str) or not row_id or row_id in row_map:
            _fail_raw_replay("raw_replay.rows[].row_id", row_id)
        row_map[row_id] = {
            key: value for key, value in row.items() if key != "row_id"
        }
    if (
        settings.get("request_count") != len(rows)
        or settings.get("row_evidence_sha256") != _sha256_json(row_map)
    ):
        _fail_raw_replay("raw_replay.settings.row_evidence_sha256", settings)

    qualified_processor = settings.get("forced_logits_processor")
    for field in ("module", "qualname", "source_path", "source_sha256"):
        qualified = (
            qualified_processor.get(field)
            if isinstance(qualified_processor, Mapping)
            else None
        )
        if processor_identity.get(field) != qualified:
            raise RuntimeContractError(
                "forced-replay processor differs from executed qualification",
                code="vllm_backend.raw_replay_qualification_processor_drift",
                context={
                    "field": field,
                    "expected": qualified,
                    "observed": processor_identity.get(field),
                },
            )
    return {
        "status": "passed",
        "receipt_path": str(path),
        "receipt_sha256": hashlib.sha256(raw).hexdigest(),
        "candidate_version": payload["vllm_version"],
        "model_dtype": launch.model_dtype,
        "source_base_snapshot_fingerprint": qualified_base,
        "processor_source_sha256": processor_identity["source_sha256"],
        "qualified_request_count": len(rows),
    }


def _select_qualification_receipt(
    *,
    launch: BackendLaunch,
    explicit_path: str | Path | None,
    receipts: Mapping[str, Path],
    kind: str,
) -> Path:
    if explicit_path is not None:
        return Path(explicit_path).resolve()
    path = receipts.get(launch.model_dtype)
    if path is None:
        raise RuntimeContractError(
            "vLLM model dtype has no executed qualification receipt",
            code="vllm_backend.qualification_dtype",
            context={
                "kind": kind,
                "observed": launch.model_dtype,
                "qualified": sorted(receipts),
            },
        )
    return path.resolve()


def _validate_concurrency_qualification(
    *,
    launch: BackendLaunch,
    engine_kwargs: Mapping[str, object],
    baseline_receipt_sha256: object,
    receipt_path: str | Path,
) -> dict[str, object]:
    path = Path(receipt_path).resolve()
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeContractError(
            "vLLM concurrency qualification receipt is unavailable",
            code="vllm_backend.concurrency_qualification_receipt",
            context={"path": str(path)},
            cause=exc,
        ) from exc
    if not isinstance(payload, Mapping) or payload.get("status") != "passed":
        _fail_concurrency("status", payload.get("status") if isinstance(payload, Mapping) else None)
    if payload.get("version") != "coordexp-swift-vllm-concurrency-qualification-v1":
        _fail_concurrency("version", payload.get("version"))
    if payload.get("vllm_version") != "0.14.1":
        _fail_concurrency("vllm_version", payload.get("vllm_version"))

    observed_max = engine_kwargs.get("max_num_seqs")
    if payload.get("max_num_seqs") != observed_max:
        _fail_concurrency("max_num_seqs", payload.get("max_num_seqs"))
    if payload.get("request_count") != observed_max:
        _fail_concurrency("request_count", payload.get("request_count"))
    requests = payload.get("requests")
    if not isinstance(requests, list) or len(requests) != observed_max:
        _fail_concurrency("requests", requests)

    qualified_engine = payload.get("backend_session_engine_kwargs")
    if not isinstance(qualified_engine, Mapping):
        _fail_concurrency("backend_session_engine_kwargs", qualified_engine)
    for field in (
        "tensor_parallel_size",
        "data_parallel_size",
        "dtype",
        "logprobs_mode",
        "generation_config",
        "limit_mm_per_prompt",
        "mm_processor_kwargs",
        "gpu_memory_utilization",
        "max_model_len",
        "max_num_seqs",
    ):
        if qualified_engine.get(field) != engine_kwargs.get(field):
            raise RuntimeContractError(
                "vLLM engine argument differs from its concurrency qualification",
                code="vllm_backend.concurrency_qualification_engine_argument",
                context={
                    "field": field,
                    "expected": qualified_engine.get(field),
                    "observed": engine_kwargs.get(field),
                },
            )

    runtime = payload.get("runtime_qualification")
    baseline = runtime.get("baseline") if isinstance(runtime, Mapping) else None
    if not isinstance(baseline, Mapping):
        _fail_concurrency("runtime_qualification.baseline", baseline)
    if baseline.get("receipt_sha256") != baseline_receipt_sha256:
        _fail_concurrency(
            "runtime_qualification.baseline.receipt_sha256",
            baseline.get("receipt_sha256"),
        )

    execution = launch.execution_model_identity
    receipt_execution = payload.get("execution_model")
    if not isinstance(execution, Mapping) or not isinstance(receipt_execution, Mapping):
        _fail_concurrency("execution_model", receipt_execution)
    source = execution.get("source_identity")
    base = source.get("base") if isinstance(source, Mapping) else None
    observed_base = base.get("fingerprint") if isinstance(base, Mapping) else None
    expected_base = receipt_execution.get("source_base_snapshot_fingerprint")
    if observed_base != expected_base:
        raise RuntimeContractError(
            "execution model differs from the concurrency-qualified base family",
            code="vllm_backend.concurrency_qualification_execution_model",
            context={
                "expected_source_base_snapshot_fingerprint": expected_base,
                "observed_source_base_snapshot_fingerprint": observed_base,
            },
        )

    probe_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "probes"
        / "coordexp_swift"
        / "vllm_concurrency.py"
    )
    observed_probe_sha256 = _sha256_file(probe_path)
    if payload.get("probe_source_sha256") != observed_probe_sha256:
        raise RuntimeContractError(
            "vLLM concurrency probe source differs from qualification evidence",
            code="vllm_backend.concurrency_qualification_source_drift",
            context={
                "path": str(probe_path),
                "expected": payload.get("probe_source_sha256"),
                "observed": observed_probe_sha256,
            },
        )
    _validate_config_sources(payload.get("config"))
    return {
        "status": "passed",
        "receipt_path": str(path),
        "receipt_sha256": hashlib.sha256(raw).hexdigest(),
        "max_num_seqs": observed_max,
    }


def _validate_config_sources(value: object) -> None:
    sources = value.get("sources") if isinstance(value, Mapping) else None
    if not isinstance(sources, list) or not sources:
        _fail_concurrency("config.sources", sources)
    mismatches: list[dict[str, object]] = []
    for source in sources:
        if not isinstance(source, Mapping):
            mismatches.append({"reason": "not_mapping"})
            continue
        path = source.get("path")
        relative_path = source.get("repo_relative_path")
        expected = source.get("sha256")
        observed = (
            _sha256_file(_repo_source_path(relative_path))
            if isinstance(relative_path, str)
            else None
        )
        if not isinstance(expected, str) or observed != expected:
            mismatches.append(
                {
                    "path": path,
                    "repo_relative_path": relative_path,
                    "expected": expected,
                    "observed": observed,
                }
            )
    if mismatches:
        raise RuntimeContractError(
            "vLLM concurrency qualification config sources have drifted",
            code="vllm_backend.concurrency_qualification_source_drift",
            context={"mismatches": mismatches[:8]},
        )


def _validate_source_identities(value: object) -> None:
    if not isinstance(value, Mapping) or not value:
        _fail("installed_sources", value)
    mismatches: list[dict[str, object]] = []
    for name, identity in value.items():
        if not isinstance(identity, Mapping):
            mismatches.append({"name": name, "reason": "not_mapping"})
            continue
        path = identity.get("path")
        package = identity.get("package")
        relative_path = identity.get("relative_path")
        expected = identity.get("sha256")
        if (
            not isinstance(package, str)
            or not isinstance(relative_path, str)
            or not isinstance(expected, str)
        ):
            mismatches.append({"name": name, "reason": "missing_identity"})
            continue
        observed = _sha256_file(
            _installed_package_source_path(
                package=package,
                relative_path=relative_path,
            )
        )
        if observed != expected:
            mismatches.append(
                {
                    "name": name,
                    "path": path,
                    "package": package,
                    "relative_path": relative_path,
                    "expected": expected,
                    "observed": observed,
                }
            )
    if mismatches:
        raise RuntimeContractError(
            "installed vLLM runtime sources differ from qualification evidence",
            code="vllm_backend.qualification_source_drift",
            context={"mismatches": mismatches[:8]},
        )


def _validate_loaded_runtime_sources(value: object) -> None:
    if not isinstance(value, Mapping):
        _fail("loaded_runtime_sources", value)
    files = value.get("files")
    if not isinstance(files, list) or not files:
        _fail("loaded_runtime_sources.files", files)
    if value.get("file_count") != len(files):
        _fail("loaded_runtime_sources.file_count", value.get("file_count"))

    mismatches: list[dict[str, object]] = []
    fingerprint_payload: list[dict[str, object]] = []
    observed_relative_paths: set[str] = set()
    for source in files:
        if not isinstance(source, Mapping):
            mismatches.append({"reason": "not_mapping"})
            continue
        path = source.get("path")
        expected = source.get("sha256")
        package = source.get("package")
        relative_path = source.get("relative_path")
        modules = source.get("modules")
        if (
            not isinstance(path, str)
            or not isinstance(expected, str)
            or not isinstance(package, str)
            or not isinstance(relative_path, str)
            or not isinstance(modules, list)
            or any(not isinstance(module, str) for module in modules)
        ):
            mismatches.append({"path": path, "reason": "missing_identity"})
            continue
        observed_relative_paths.add(relative_path)
        observed = _sha256_file(
            _installed_package_source_path(
                package=package,
                relative_path=relative_path,
            )
        )
        if observed != expected:
            mismatches.append(
                {"path": path, "expected": expected, "observed": observed}
            )
        fingerprint_payload.append(
            {
                "package": package,
                "relative_path": relative_path,
                "sha256": expected,
                "modules": modules,
            }
        )

    required_paths = value.get("required_paths")
    if not isinstance(required_paths, list) or any(
        not isinstance(path, str) for path in required_paths
    ):
        _fail("loaded_runtime_sources.required_paths", required_paths)
    missing_required = sorted(set(required_paths) - observed_relative_paths)
    if missing_required or value.get("required_paths_present") is not True:
        mismatches.append(
            {"reason": "required_paths_missing", "paths": missing_required}
        )
    expected_fingerprint = value.get("fingerprint")
    observed_fingerprint = _sha256_json(fingerprint_payload)
    if expected_fingerprint != observed_fingerprint:
        mismatches.append(
            {
                "reason": "fingerprint_mismatch",
                "expected": expected_fingerprint,
                "observed": observed_fingerprint,
            }
        )
    if mismatches:
        raise RuntimeContractError(
            "loaded vLLM runtime sources differ from qualification evidence",
            code="vllm_backend.qualification_loaded_source_drift",
            context={"mismatches": mismatches[:8]},
        )


def _application_receipt_path(
    *,
    baseline_receipt_path: Path,
    explicit_path: str | Path | None,
) -> Path:
    if explicit_path is not None:
        return Path(explicit_path).expanduser().resolve()
    if baseline_receipt_path.resolve() in {
        path.resolve() for path in _QUALIFICATION_RECEIPTS.values()
    }:
        return APPLICATION_QUALIFICATION_RECEIPT.resolve()
    return baseline_receipt_path.with_name("vllm-0.14.1-application-sources.json")


def _validate_application_sources(path: Path) -> dict[str, object]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeContractError(
            "CoordExp vLLM application-source qualification is unavailable",
            code="vllm_backend.application_qualification_receipt",
            context={"path": str(path)},
            cause=exc,
        ) from exc
    if (
        not isinstance(payload, Mapping)
        or payload.get("status") != "passed"
        or payload.get("version") != "coordexp-swift-vllm-application-sources-v1"
    ):
        raise RuntimeContractError(
            "CoordExp vLLM application-source qualification is invalid",
            code="vllm_backend.application_qualification_receipt",
            context={"path": str(path)},
        )
    sources = payload.get("source_sha256")
    if not isinstance(sources, Mapping) or not sources:
        raise RuntimeContractError(
            "CoordExp vLLM application-source qualification has no sources",
            code="vllm_backend.application_qualification_receipt",
            context={"path": str(path)},
        )
    expected_paths = set(APPLICATION_EXECUTION_SOURCE_PATHS)
    observed_paths = set(sources)
    if observed_paths != expected_paths:
        raise RuntimeContractError(
            "CoordExp vLLM application-source qualification has incomplete owners",
            code="vllm_backend.application_qualification_receipt",
            context={
                "path": str(path),
                "missing_paths": sorted(expected_paths - observed_paths),
                "unexpected_paths": sorted(observed_paths - expected_paths),
            },
        )
    mismatches: list[dict[str, object]] = []
    for relative_path, expected in sources.items():
        if not isinstance(relative_path, str) or not isinstance(expected, str):
            mismatches.append({"path": relative_path, "reason": "invalid_identity"})
            continue
        source_path = _repo_source_path(relative_path)
        observed = _sha256_file(source_path)
        if observed != expected:
            mismatches.append(
                {
                    "path": relative_path,
                    "expected": expected,
                    "observed": observed,
                }
            )
    if mismatches:
        raise RuntimeContractError(
            "CoordExp vLLM execution sources differ from qualification evidence",
            code="vllm_backend.application_qualification_source_drift",
            context={"mismatches": mismatches[:8]},
        )
    return {
        "status": "passed",
        "receipt_path": str(path),
        "receipt_sha256": hashlib.sha256(raw).hexdigest(),
        "source_count": len(sources),
    }


def _validate_qualification_probe(probe: object) -> None:
    if not isinstance(probe, Mapping):
        _fail("probe", probe)
    recorded_path = probe.get("path")
    recorded_relative_path = probe.get("repo_relative_path")
    recorded_sha256 = probe.get("sha256")
    expected_relative_path = QUALIFICATION_PROBE_RELATIVE_PATH.as_posix()
    expected_path = _repo_source_path(expected_relative_path)
    if recorded_relative_path != expected_relative_path:
        raise RuntimeContractError(
            "vLLM qualification receipt points to a different probe",
            code="vllm_backend.qualification_probe_drift",
            context={
                "expected_path": str(expected_path),
                "recorded_path": recorded_path,
                "expected_repo_relative_path": expected_relative_path,
                "recorded_repo_relative_path": recorded_relative_path,
            },
        )
    observed_sha256 = _sha256_file(expected_path)
    if not isinstance(recorded_sha256, str) or observed_sha256 != recorded_sha256:
        raise RuntimeContractError(
            "vLLM qualification probe source differs from its receipt",
            code="vllm_backend.qualification_probe_drift",
            context={
                "path": str(expected_path),
                "expected_sha256": recorded_sha256,
                "observed_sha256": observed_sha256,
            },
        )


def _repo_source_path(relative_path: str) -> Path:
    candidate = (_REPO_ROOT / relative_path).resolve()
    try:
        candidate.relative_to(_REPO_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeContractError(
            "qualification source escapes the active repository",
            code="vllm_backend.qualification_source_path",
            context={"repo_relative_path": relative_path},
            cause=exc,
        ) from exc
    return candidate


def _installed_package_source_path(*, package: str, relative_path: str) -> Path:
    top_level_package = relative_path.split("/", 1)[0]
    if top_level_package != package:
        raise RuntimeContractError(
            "qualification package-relative source identity is inconsistent",
            code="vllm_backend.qualification_source_path",
            context={"package": package, "relative_path": relative_path},
        )
    try:
        distribution = metadata.distribution(package)
    except metadata.PackageNotFoundError as exc:
        raise RuntimeContractError(
            "qualified package source cannot be resolved in this environment",
            code="vllm_backend.qualification_source_path",
            context={"package": package, "relative_path": relative_path},
            cause=exc,
        ) from exc
    installation_root = Path(distribution.locate_file("")).resolve()
    candidate = Path(distribution.locate_file(relative_path)).resolve()
    try:
        candidate.relative_to(installation_root)
    except ValueError as exc:
        raise RuntimeContractError(
            "qualification package-relative source escapes its installation root",
            code="vllm_backend.qualification_source_path",
            context={"package": package, "relative_path": relative_path},
            cause=exc,
        ) from exc
    return candidate


def _sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _sha256_json(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _fail(field: str, value: object) -> None:
    raise RuntimeContractError(
        "vLLM runtime qualification receipt is invalid",
        code="vllm_backend.qualification_receipt",
        context={"field": field, "value": value},
    )


def _fail_concurrency(field: str, value: object) -> None:
    raise RuntimeContractError(
        "vLLM concurrency qualification receipt is invalid",
        code="vllm_backend.concurrency_qualification_receipt",
        context={"field": field, "value": value},
    )


def _fail_raw_replay(field: str, value: object) -> None:
    raise RuntimeContractError(
        "vLLM forced-replay qualification receipt is invalid",
        code="vllm_backend.raw_replay_qualification_invalid",
        context={"field": field, "value": value},
    )
