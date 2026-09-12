"""Produce and atomically admit bounded BF16 vLLM qualification receipts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Literal, NoReturn, TypeGuard, cast

from src.common.errors import RuntimeContractError


QualificationKind = Literal["composition", "runtime", "concurrency", "forced_replay"]

SCHEMA_VERSION = 4
CONTRACT_VERSION = "coordexp-vllm-bf16-qualification-v4"
EXPECTED_RECEIPT_FILENAMES: dict[QualificationKind, str] = {
    "composition": "vllm-bf16-composition.json",
    "runtime": "vllm-bf16-runtime-seq1.json",
    "concurrency": "vllm-bf16-concurrency-seq4.json",
    "forced_replay": "vllm-bf16-forced-replay-seq1.json",
}
MODE_SPECS: tuple[tuple[QualificationKind, int], ...] = (
    ("composition", 1),
    ("runtime", 1),
    ("concurrency", 4),
    ("forced_replay", 1),
)
DEFAULT_ADMISSION_ROOT = Path(__file__).resolve().with_name("qualification_receipts")
MAX_RECEIPT_BYTES = 64 * 1024
MAX_EVIDENCE_ARTIFACTS = 32
MAX_EVIDENCE_ARTIFACT_BYTES = 16 * 1024 * 1024
GPU_MEMORY_TOLERANCE_MIB = 64
GPU_MEMORY_SETTLE_ATTEMPTS = 20
GPU_MEMORY_SETTLE_INTERVAL_SECONDS = 0.25
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_PATHS = (
    "src/qualify_vllm.py",
    *tuple(
        path.relative_to(_REPO_ROOT).as_posix()
        for root in (
            "src/adapters",
            "src/common",
            "src/config",
            "src/data",
            "src/inference",
            "src/qwen",
        )
        for path in sorted((_REPO_ROOT / root).glob("*.py"))
    ),
)
_RUNTIME_PACKAGES = ("vllm", "torch", "transformers", "peft", "qwen-vl-utils")
_RUNTIME_MODULES = (
    ("vllm", "vllm"),
    ("vllm", "vllm.engine.arg_utils"),
    ("vllm", "vllm.multimodal.processing"),
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("peft", "peft"),
)


@dataclass(frozen=True)
class ChildSpec:
    kind: QualificationKind
    max_num_seqs: int
    config_path: Path
    output_root: Path
    evidence_dir: Path


@dataclass(frozen=True)
class ChildExecution:
    returncode: int
    evidence: Mapping[str, object] | None
    process: Mapping[str, object]


@dataclass(frozen=True)
class QualificationDependencies:
    build_identity: Callable[[Path], Mapping[str, object]]
    run_child: Callable[[ChildSpec], ChildExecution]


def produce(
    *,
    config_path: str | Path,
    output_root: str | Path,
    dependencies: QualificationDependencies | None = None,
) -> dict[str, object]:
    """Run the four isolated qualification modes and publish passed receipts last."""

    config = Path(config_path).expanduser().resolve()
    root = Path(output_root).expanduser().resolve()
    if root.exists():
        _fail(
            "qualification output root must be absent",
            code="vllm_qualification.output_root_exists",
            context={"path": str(root)},
        )
    deps = dependencies or _production_dependencies()
    identity = _validate_identity(deps.build_identity(config))
    if identity["model"]["target_dtype"] != "bf16":
        _fail(
            "vLLM qualification is BF16-only",
            code="vllm_qualification.bf16_required",
            context={"target_dtype": identity["model"]["target_dtype"]},
        )

    root.mkdir(parents=True, exist_ok=False)
    published: dict[str, str] = {}
    for kind, max_num_seqs in MODE_SPECS:
        spec = ChildSpec(
            kind=kind,
            max_num_seqs=max_num_seqs,
            config_path=config,
            output_root=root,
            evidence_dir=root / "evidence" / kind,
        )
        execution = deps.run_child(spec)
        _validate_child_process(execution, kind=kind)
        if execution.returncode != 0:
            _fail(
                "qualification child exited nonzero",
                code="vllm_qualification.child_failed",
                context={"kind": kind, "returncode": execution.returncode},
            )
        evidence = _validate_mode_evidence(
            kind=kind,
            max_num_seqs=max_num_seqs,
            value=execution.evidence,
            expected_snapshot_fingerprint=identity["model"]["snapshot_fingerprint"],
        )
        artifacts = _artifact_manifest(spec.evidence_dir, root=root)
        receipt = _build_receipt(
            kind=kind,
            max_num_seqs=max_num_seqs,
            identity=identity,
            evidence=evidence,
            process=dict(execution.process),
            artifacts=artifacts,
        )
        filename = EXPECTED_RECEIPT_FILENAMES[kind]
        path = root / filename
        _atomic_write_json(path, receipt)
        published[kind] = str(path)
    return {"status": "passed", "output_root": str(root), "receipts": published}


def admit(
    *,
    config_path: str | Path,
    receipts_root: str | Path,
    target_root: str | Path = DEFAULT_ADMISSION_ROOT,
    dependencies: QualificationDependencies | None = None,
) -> dict[str, object]:
    """Validate a whole external set, then install only its known receipts."""

    config = Path(config_path).expanduser().resolve()
    source = Path(receipts_root).expanduser().resolve()
    target = Path(target_root).expanduser().resolve()
    deps = dependencies or _production_dependencies()
    identity = _validate_identity(deps.build_identity(config))
    receipts = validate_receipt_set(
        receipt_root=source,
        identity=identity,
        require_artifacts=True,
    )

    expected_names = set(EXPECTED_RECEIPT_FILENAMES.values())
    if target.exists():
        if not target.is_dir():
            _admission_drift(target, "target is not a directory")
        present = {path.name for path in target.iterdir()}
        if present == expected_names and all(
            (target / name).read_bytes() == (source / name).read_bytes()
            for name in expected_names
        ):
            return {"status": "identical", "target_root": str(target)}
        if present:
            _admission_drift(target, "existing receipt set is partial or different")

    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.admit-", dir=target.parent))
    try:
        for name in sorted(expected_names):
            destination = stage / name
            shutil.copyfile(source / name, destination)
            _fsync_file(destination)
        _fsync_directory(stage)
        if target.exists():
            target.rmdir()
        os.replace(stage, target)
        _fsync_directory(target.parent)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {
        "status": "installed",
        "target_root": str(target),
        "receipt_digests": {
            kind: receipt["digest"] for kind, receipt in receipts.items()
        },
    }


def validate_receipt_set(
    *,
    receipt_root: str | Path,
    identity: Mapping[str, object],
    require_artifacts: bool,
) -> dict[str, dict[str, Any]]:
    root = Path(receipt_root).expanduser().resolve()
    if not root.is_dir():
        _fail(
            "vLLM qualification receipt set is unavailable",
            code="vllm_backend.qualification_receipt",
            context={"path": str(root)},
        )
    current = _validate_identity(identity)
    receipts: dict[str, dict[str, Any]] = {}
    declared_artifacts: set[str] = set()
    for kind, max_num_seqs in MODE_SPECS:
        path = root / EXPECTED_RECEIPT_FILENAMES[kind]
        try:
            raw = path.read_bytes()
            payload = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            _fail(
                "vLLM qualification receipt is unavailable or malformed",
                code="vllm_backend.qualification_receipt",
                context={"path": str(path)},
                cause=exc,
            )
        if len(raw) >= MAX_RECEIPT_BYTES:
            _fail(
                "vLLM qualification receipt exceeds the bounded payload limit",
                code="vllm_backend.qualification_receipt_unbounded",
                context={"path": str(path), "bytes": len(raw)},
            )
        receipt = _validate_receipt(
            payload,
            kind=kind,
            max_num_seqs=max_num_seqs,
            identity=current,
        )
        for artifact in receipt["artifacts"]:
            relative = str(artifact["path"])
            if relative in declared_artifacts:
                _fail(
                    "qualification evidence artifact is declared twice",
                    code="vllm_backend.qualification_artifact",
                    context={"path": relative},
                )
            declared_artifacts.add(relative)
            if require_artifacts:
                evidence_path = _contained_path(root, relative)
                if (
                    not evidence_path.is_file()
                    or evidence_path.stat().st_size != artifact["bytes"]
                    or _sha256_file(evidence_path) != artifact["sha256"]
                ):
                    _fail(
                        "qualification evidence artifact differs from its receipt",
                        code="vllm_backend.qualification_artifact",
                        context={"path": str(evidence_path)},
                    )
        receipts[kind] = receipt

    if require_artifacts:
        expected_files = set(EXPECTED_RECEIPT_FILENAMES.values()) | declared_artifacts
        observed_files = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        }
        if observed_files != expected_files:
            _fail(
                "qualification root contains missing or extra files",
                code="vllm_backend.qualification_receipt_set",
                context={
                    "missing": sorted(expected_files - observed_files),
                    "extra": sorted(observed_files - expected_files),
                },
            )
    return receipts


def build_qualification_identity(config_path: str | Path) -> dict[str, object]:
    """Build the current source/config/model/runtime identity without GPU work."""

    from src.config.inference import load_infer_config
    from src.inference.execution_model import resolve_execution_model

    config = Path(config_path).expanduser().resolve()
    resolved = load_infer_config(config)
    if resolved.config.backend.type != "vllm":
        _fail(
            "qualification config must select the vLLM backend",
            code="vllm_qualification.backend_required",
        )
    if resolved.config.model.dtype != "bf16":
        _fail(
            "vLLM qualification is BF16-only",
            code="vllm_qualification.bf16_required",
            context={"target_dtype": resolved.config.model.dtype},
        )
    execution_model = resolve_execution_model(
        base_model_path=resolved.config.model.base_model,
        target_dtype="bf16",
        adapter_path=(None if resolved.config.adapter is None else resolved.config.adapter.path),
        adapter_name=("default" if resolved.config.adapter is None else resolved.config.adapter.name),
        embedding_delta_path=(
            None if resolved.config.embedding_delta is None else resolved.config.embedding_delta.path
        ),
        _skip_existing_composition_fidelity=True,
    )
    source_identity = _build_source_identity()
    source_fingerprints = execution_model["source_identity"]
    launch_contracts = _build_config_launch_contracts(
        config=resolved.config,
        execution_model=execution_model,
    )
    identity: dict[str, object] = {
        "source": source_identity,
        "config": {
            "entry_path": str(resolved.entry_config_path),
            "fingerprint": resolved.fingerprint,
            "sources": [
                {"path": str(source.path), "sha256": source.sha256}
                for source in resolved.sources
            ],
            "launch_contracts": launch_contracts,
        },
        "model": {
            "mode": execution_model["mode"],
            "composition_key": execution_model["composition_key"],
            "snapshot_fingerprint": execution_model["snapshot_fingerprint"],
            "receipt_fingerprint": execution_model["receipt_fingerprint"],
            "source_fingerprints": {
                "base": source_fingerprints["base"]["fingerprint"],
                "adapter": _optional_fingerprint(source_fingerprints.get("adapter")),
                "embedding_delta": _optional_fingerprint(
                    source_fingerprints.get("embedding_delta")
                ),
            },
            "target_dtype": execution_model["target_dtype"],
        },
        "runtime": {
            "python": ".".join(str(item) for item in sys.version_info[:3]),
            "packages": {package: metadata.version(package) for package in _RUNTIME_PACKAGES},
            "loaded_sources": _runtime_source_identity(),
        },
    }
    return _validate_identity(identity)


def _production_dependencies() -> QualificationDependencies:
    return QualificationDependencies(
        build_identity=build_qualification_identity,
        run_child=_run_subprocess_child,
    )


def _build_source_identity() -> dict[str, object]:
    files = [
        {"path": relative, "sha256": _sha256_file(_REPO_ROOT / relative)}
        for relative in _SOURCE_PATHS
    ]
    return {"files": files, "fingerprint": _sha256_json(files)}


def _build_config_launch_contracts(
    *,
    config: Any,
    execution_model: Mapping[str, object],
) -> dict[str, object]:
    from src.config.fingerprint import sha256_json
    from src.inference.runtime import prepare_backend_launch
    from src.inference.vllm_backend import _engine_kwargs, _vllm_options

    contracts: dict[str, object] = {}
    for max_num_seqs in (1, 4):
        generation = config.generation.model_copy(update={"batch_size": max_num_seqs})
        mode_config = config.model_copy(update={"generation": generation})
        launch = prepare_backend_launch(
            mode_config,
            generation_config_fingerprint=sha256_json(
                generation.model_dump(mode="json")
            ),
            execution_model=execution_model,
        )
        engine_kwargs = _engine_kwargs(launch, options=_vllm_options(launch))
        contracts[f"seq{max_num_seqs}"] = build_launch_contract(
            launch=launch,
            engine_kwargs=engine_kwargs,
        )
    return contracts


def build_launch_contract(
    *,
    launch: Any,
    engine_kwargs: Mapping[str, object],
) -> dict[str, object]:
    """Return the bounded executable launch identity shared by producer and validator."""

    payload = {
        "backend": launch.backend,
        "model_path": str(Path(launch.model_path).expanduser().resolve()),
        "model_dtype": launch.model_dtype,
        "batch_size": launch.batch_size,
        "generation_config_fingerprint": launch.generation_config_fingerprint,
        "backend_options": json.loads(json.dumps(launch.backend_options, sort_keys=True)),
        "engine_kwargs": json.loads(json.dumps(engine_kwargs, sort_keys=True)),
    }
    return {"payload": payload, "fingerprint": _sha256_json(payload)}


def _run_subprocess_child(spec: ChildSpec) -> ChildExecution:
    spec.evidence_dir.parent.mkdir(parents=True, exist_ok=True)
    result_path = spec.output_root / f".{spec.kind}-child-result.json"
    command = [
        sys.executable,
        "-m",
        "src.qualify_vllm",
        "_child",
        "--kind",
        spec.kind,
        "--max-num-seqs",
        str(spec.max_num_seqs),
        "--config",
        str(spec.config_path),
        "--evidence-dir",
        str(spec.evidence_dir),
        "--result",
        str(result_path),
    ]
    visible_gpu_selector = _visible_physical_gpu_selector()
    gpu_memory_before = _gpu_memory_used_mib(visible_gpu_selector)
    if gpu_memory_before is None:
        _fail(
            "selected-GPU memory is unavailable before qualification",
            code="vllm_qualification.gpu_memory_measurement",
            context={"selector": visible_gpu_selector, "phase": "before"},
        )
    child = subprocess.Popen(
        command,
        cwd=_REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = child.communicate()
    except BaseException:
        _terminate_process_group(child.pid)
        try:
            child.wait(timeout=2)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=2)
        raise
    returncode = int(child.returncode)
    process_group_id = child.pid
    members_before_cleanup = _process_group_members(process_group_id)
    if members_before_cleanup:
        _terminate_process_group(process_group_id)
    members_after_cleanup = _process_group_members(process_group_id)
    group_gpu_memory_after = _gpu_memory_for_pids_mib(
        members_after_cleanup,
        selector=visible_gpu_selector,
    )
    gpu_memory_after = _settle_gpu_memory_used_mib(
        selector=visible_gpu_selector,
        before_mib=gpu_memory_before,
        tolerance_mib=GPU_MEMORY_TOLERANCE_MIB,
    )
    memory_returned = (
        gpu_memory_after is not None
        and gpu_memory_after <= gpu_memory_before + GPU_MEMORY_TOLERANCE_MIB
    )
    evidence: Mapping[str, object] | None = None
    child_process: dict[str, object] = {
        "worker_pid": child.pid,
        "worker_returncode": returncode,
        "worker_pid_alive_after_exit": Path(f"/proc/{child.pid}").exists(),
        "owned_children_after": [],
        "process_group_id": process_group_id,
        "process_group_members_after": members_after_cleanup,
        "process_group_termination_required": bool(members_before_cleanup),
        "gpu_process_group_memory_after_mib": group_gpu_memory_after,
        "visible_gpu_selector": visible_gpu_selector,
        "gpu_memory_returned_to_baseline": memory_returned,
        "gpu_memory_before_mib": gpu_memory_before,
        "gpu_memory_after_mib": gpu_memory_after,
        "gpu_memory_tolerance_mib": GPU_MEMORY_TOLERANCE_MIB,
        "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr.encode()).hexdigest(),
    }
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result_path.unlink()
        evidence_value = result.get("evidence")
        if isinstance(evidence_value, Mapping):
            evidence = dict(evidence_value)
        process_value = result.get("process")
        if isinstance(process_value, Mapping):
            child_process.update(process_value)
        owned = child_process.get("owned_children_after")
        child_process.update(
            worker_pid=child.pid,
            worker_returncode=returncode,
            worker_pid_alive_after_exit=Path(f"/proc/{child.pid}").exists(),
            owned_children_after=(
                [pid for pid in owned if Path(f"/proc/{pid}").exists()]
                if isinstance(owned, list)
                else owned
            ),
            process_group_id=process_group_id,
            process_group_members_after=members_after_cleanup,
            process_group_termination_required=bool(members_before_cleanup),
            gpu_process_group_memory_after_mib=group_gpu_memory_after,
            visible_gpu_selector=visible_gpu_selector,
            gpu_memory_returned_to_baseline=memory_returned,
            gpu_memory_before_mib=gpu_memory_before,
            gpu_memory_after_mib=gpu_memory_after,
            gpu_memory_tolerance_mib=GPU_MEMORY_TOLERANCE_MIB,
        )
        if evidence is not None:
            cleanup = evidence.get("cleanup")
            if isinstance(cleanup, Mapping):
                evidence = {
                    **dict(evidence),
                    "cleanup": {
                        **dict(cleanup),
                        "worker_pid_alive_after_exit": child_process[
                            "worker_pid_alive_after_exit"
                        ],
                        "owned_children_after": child_process[
                            "owned_children_after"
                        ],
                        "gpu_memory_returned_to_baseline": memory_returned,
                    },
                }
    return ChildExecution(
        returncode=returncode,
        evidence=evidence,
        process=child_process,
    )


def _process_group_members(process_group_id: int) -> list[int]:
    if process_group_id <= 1:
        return []
    members: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            remainder = (entry / "stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
            state = remainder[0]
            process_group = int(remainder[2])
        except (OSError, IndexError, ValueError):
            continue
        if process_group == process_group_id and state != "Z":
            members.append(int(entry.name))
    return sorted(members)


def _terminate_process_group(process_group_id: int) -> None:
    if process_group_id <= 1:
        return
    for sig, timeout in ((signal.SIGTERM, 1.0), (signal.SIGKILL, 1.0)):
        if not _process_group_members(process_group_id):
            return
        try:
            os.killpg(process_group_id, sig)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not _process_group_members(process_group_id):
                return
            time.sleep(0.02)


def _gpu_memory_for_pids_mib(pids: Sequence[int], *, selector: str) -> int:
    owned = set(pids)
    if not owned:
        return 0
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={selector}",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return -1
    total = 0
    try:
        for line in result.stdout.splitlines():
            pid, memory = (part.strip() for part in line.split(",", 1))
            if int(pid) in owned:
                total += int(memory)
    except ValueError:
        return -1
    return total


def _visible_physical_gpu_selector() -> str:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    selector = "" if raw is None else raw.strip()
    is_index = selector.isdecimal()
    is_gpu_uuid = (
        selector.startswith("GPU-")
        and len(selector) > 4
        and all(character.isalnum() or character == "-" for character in selector)
    )
    if raw != selector or not (is_index or is_gpu_uuid):
        _fail(
            "qualification requires one explicit physical CUDA_VISIBLE_DEVICES selector",
            code="vllm_qualification.cuda_visible_devices",
            context={"CUDA_VISIBLE_DEVICES": raw},
        )
    return selector


def _gpu_memory_used_mib(selector: str) -> int | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={selector}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    try:
        values = [
            int(line.strip()) for line in result.stdout.splitlines() if line.strip()
        ]
    except ValueError:
        return None
    return values[0] if len(values) == 1 else None


def _settle_gpu_memory_used_mib(
    *,
    selector: str,
    before_mib: int,
    tolerance_mib: int,
    attempts: int = GPU_MEMORY_SETTLE_ATTEMPTS,
    interval_seconds: float = GPU_MEMORY_SETTLE_INTERVAL_SECONDS,
    read_memory: Callable[[str], int | None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> int | None:
    if attempts <= 0 or interval_seconds < 0:
        raise ValueError("GPU memory settle bounds must be positive")
    memory_reader = _gpu_memory_used_mib if read_memory is None else read_memory
    last_measured: int | None = None
    for attempt in range(attempts):
        measured = memory_reader(selector)
        if measured is not None:
            last_measured = measured
            if measured <= before_mib + tolerance_mib:
                return measured
        if attempt + 1 < attempts:
            sleep(interval_seconds)
    return last_measured


def _build_receipt(
    *,
    kind: QualificationKind,
    max_num_seqs: int,
    identity: Mapping[str, object],
    evidence: Mapping[str, object],
    process: Mapping[str, object],
    artifacts: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "status": "passed",
        "kind": kind,
        "model_dtype": "bf16",
        "max_num_seqs": max_num_seqs,
        "identity": json.loads(json.dumps(identity, sort_keys=True)),
        "evidence": dict(evidence),
        "process": dict(process),
        "artifacts": [dict(item) for item in artifacts],
    }
    receipt["digest"] = _receipt_digest(receipt)
    encoded = _json_bytes(receipt)
    if len(encoded) >= MAX_RECEIPT_BYTES:
        _fail(
            "qualification receipt exceeds the bounded payload limit",
            code="vllm_qualification.receipt_unbounded",
            context={"kind": kind, "bytes": len(encoded)},
        )
    return receipt


def _validate_receipt(
    value: object,
    *,
    kind: QualificationKind,
    max_num_seqs: int,
    identity: Mapping[str, object],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _receipt_failure(kind, "receipt", type(value).__name__)
    payload = dict(value)
    required = {
        "schema_version",
        "contract_version",
        "status",
        "kind",
        "model_dtype",
        "max_num_seqs",
        "identity",
        "evidence",
        "process",
        "artifacts",
        "digest",
    }
    if set(payload) != required:
        _receipt_failure(kind, "fields", sorted(payload))
    expected_scalars = {
        "schema_version": SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "status": "passed",
        "kind": kind,
        "model_dtype": "bf16",
        "max_num_seqs": max_num_seqs,
    }
    for field, expected in expected_scalars.items():
        if payload.get(field) != expected:
            _receipt_failure(kind, field, payload.get(field))
    if payload.get("identity") != identity:
        _fail(
            "vLLM qualification receipt identity has drifted",
            code="vllm_backend.qualification_identity_drift",
            context={"kind": kind},
        )
    _validate_mode_evidence(
        kind=kind,
        max_num_seqs=max_num_seqs,
        value=payload.get("evidence"),
        expected_snapshot_fingerprint=cast(Mapping[str, object], identity["model"])["snapshot_fingerprint"],
    )
    _validate_process_mapping(payload.get("process"), kind=kind)
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        _receipt_failure(kind, "artifacts", artifacts)
    if len(artifacts) > MAX_EVIDENCE_ARTIFACTS:
        _receipt_failure(kind, "artifacts", len(artifacts))
    for artifact in artifacts:
        _validate_artifact_entry(artifact, kind=kind)
    if payload.get("digest") != _receipt_digest(payload):
        _receipt_failure(kind, "digest", payload.get("digest"))
    return payload


def _validate_mode_evidence(
    *,
    kind: QualificationKind,
    max_num_seqs: int,
    value: object,
    expected_snapshot_fingerprint: object,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _receipt_failure(kind, "evidence", type(value).__name__)
    evidence = dict(value)
    common = {"status", "cleanup"}
    expected_fields: dict[QualificationKind, set[str]] = {
        "composition": common
        | {
            "composition_digest",
            "prompt_ids_equal",
            "selected_rows_equal",
            "greedy_ids_equal",
            "bounded_generation",
            "full_vocab",
            "selected_vocab",
        },
        "runtime": common
        | {
            "max_num_seqs",
            "request_count",
            "completed_request_count",
            "generated_token_count",
            "finite_non_positive_policy_logprobs",
        },
        "concurrency": common
        | {
            "max_num_seqs",
            "request_count",
            "completed_request_count",
            "ordered_request_ids_sha256",
            "generated_token_count",
            "finite_non_positive_policy_logprobs",
        },
        "forced_replay": common
        | {
            "max_num_seqs",
            "request_count",
            "completed_request_count",
            "aligned_token_count",
            "processor_source_sha256",
            "finite_non_positive_raw_logprobs",
            "raw_logprob_min",
            "raw_logprob_max",
        },
    }
    if set(evidence) != expected_fields[kind] or evidence.get("status") != "passed":
        _receipt_failure(kind, "evidence.fields/status", sorted(evidence))
    _validate_cleanup(evidence.get("cleanup"), kind=kind)
    if kind == "composition":
        _require_sha256(evidence.get("composition_digest"), kind, "composition_digest")
        for field in ("prompt_ids_equal", "selected_rows_equal"):
            if evidence.get(field) is not True:
                _receipt_failure(kind, field, evidence.get(field))
        from src.inference.execution_model_composition import (
            validate_bounded_generation_evidence,
        )

        bounded = evidence.get("bounded_generation")
        if not isinstance(bounded, Mapping):
            _receipt_failure(kind, "bounded_generation", "mapping_required")
        _require_sha256(
            expected_snapshot_fingerprint,
            kind,
            "bounded_generation.snapshot_fingerprint",
        )
        checked = validate_bounded_generation_evidence(
            bounded,
            expected_snapshot_fingerprint=cast(str, expected_snapshot_fingerprint),
        )
        if checked["accepted"] is not True:
            _receipt_failure(kind, "bounded_generation.accepted", False)
        greedy_equal = (
            checked["dynamic_generated_ids"] == checked["materialized_generated_ids"]
        )
        if (
            not isinstance(evidence.get("greedy_ids_equal"), bool)
            or evidence["greedy_ids_equal"] != greedy_equal
        ):
            _receipt_failure(kind, "greedy_ids_equal", "bounded_evidence_mismatch")
        _validate_numeric_summary(evidence.get("full_vocab"), kind, "full_vocab")
        _validate_numeric_summary(evidence.get("selected_vocab"), kind, "selected_vocab")
        return evidence
    expected_requests = 4 if kind == "concurrency" else 1
    if (
        evidence.get("max_num_seqs") != max_num_seqs
        or evidence.get("request_count") != expected_requests
        or evidence.get("completed_request_count") != expected_requests
    ):
        _receipt_failure(kind, "request_counts", evidence)
    if kind in ("runtime", "concurrency"):
        if not _positive_int(evidence.get("generated_token_count")):
            _receipt_failure(kind, "generated_token_count", evidence.get("generated_token_count"))
        if evidence.get("finite_non_positive_policy_logprobs") is not True:
            _receipt_failure(kind, "finite_non_positive_policy_logprobs", False)
        if kind == "concurrency":
            _require_sha256(
                evidence.get("ordered_request_ids_sha256"),
                kind,
                "ordered_request_ids_sha256",
            )
        return evidence
    if not _positive_int(evidence.get("aligned_token_count")):
        _receipt_failure(kind, "aligned_token_count", evidence.get("aligned_token_count"))
    _require_sha256(evidence.get("processor_source_sha256"), kind, "processor_source_sha256")
    if evidence.get("finite_non_positive_raw_logprobs") is not True:
        _receipt_failure(kind, "finite_non_positive_raw_logprobs", False)
    raw_min = evidence.get("raw_logprob_min")
    raw_max = evidence.get("raw_logprob_max")
    if not _finite_number(raw_min) or not _finite_number(raw_max) or raw_min > raw_max or raw_max > 0:
        _receipt_failure(kind, "raw_logprob_range", [raw_min, raw_max])
    return evidence


def _validate_cleanup(value: object, *, kind: str) -> None:
    expected = {
        "shutdown_completed": True,
        "worker_pid_alive_after_exit": False,
        "owned_children_after": [],
        "gpu_memory_returned_to_baseline": True,
    }
    if value != expected:
        _receipt_failure(kind, "cleanup", value)


def _validate_child_process(execution: ChildExecution, *, kind: str) -> None:
    _validate_process_mapping(execution.process, kind=kind)
    if execution.process.get("worker_returncode") != execution.returncode:
        _receipt_failure(kind, "process.worker_returncode", execution.process)


def _validate_process_mapping(value: object, *, kind: str) -> None:
    if not isinstance(value, Mapping):
        _receipt_failure(kind, "process", value)
    measurement_fields = {
        "visible_gpu_selector",
        "gpu_memory_before_mib",
        "gpu_memory_after_mib",
        "gpu_memory_tolerance_mib",
    }
    if not measurement_fields.issubset(value):
        _receipt_failure(kind, "process.gpu_memory_measurement", sorted(value))
    required = {
        "worker_pid",
        "worker_returncode",
        "worker_pid_alive_after_exit",
        "owned_children_after",
        "process_group_id",
        "process_group_members_after",
        "process_group_termination_required",
        "gpu_process_group_memory_after_mib",
        "gpu_memory_returned_to_baseline",
        *measurement_fields,
    }
    optional = {
        "stdout_sha256",
        "stderr_sha256",
    }
    if not required.issubset(value) or set(value).difference(required | optional):
        _receipt_failure(kind, "process.fields", sorted(value))
    selector = value.get("visible_gpu_selector")
    before = value.get("gpu_memory_before_mib")
    after = value.get("gpu_memory_after_mib")
    tolerance = value.get("gpu_memory_tolerance_mib")
    valid_selector = isinstance(selector, str) and (
        selector.isdecimal()
        or (
            selector.startswith("GPU-")
            and len(selector) > 4
            and all(character.isalnum() or character == "-" for character in selector)
        )
    )
    if (
        not valid_selector
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in (before, after, tolerance)
        )
        or tolerance != GPU_MEMORY_TOLERANCE_MIB
    ):
        _receipt_failure(
            kind,
            "process.gpu_memory_measurement",
            {
                "visible_gpu_selector": selector,
                "before": before,
                "after": after,
                "tolerance": tolerance,
            },
        )
    before_mib = cast(int, before)
    after_mib = cast(int, after)
    tolerance_mib = cast(int, tolerance)
    expected_memory_return = after_mib <= before_mib + tolerance_mib
    if value.get("gpu_memory_returned_to_baseline") is not expected_memory_return:
        _receipt_failure(
            kind,
            "process.gpu_memory_consistency",
            {
                "claimed": value.get("gpu_memory_returned_to_baseline"),
                "computed": expected_memory_return,
                "before": before,
                "after": after,
                "tolerance": tolerance,
            },
        )
    if (
        value.get("worker_pid_alive_after_exit") is not False
        or value.get("owned_children_after") != []
        or value.get("process_group_members_after") != []
        or value.get("process_group_termination_required") is not False
        or value.get("gpu_process_group_memory_after_mib") != 0
        or value.get("gpu_memory_returned_to_baseline") is not True
    ):
        _receipt_failure(kind, "process.cleanup", value)
    for field in ("stdout_sha256", "stderr_sha256"):
        if field in value:
            _require_sha256(value[field], kind, f"process.{field}")


def _artifact_manifest(directory: Path, *, root: Path) -> list[dict[str, object]]:
    if not directory.is_dir():
        _fail(
            "qualification child did not create its evidence directory",
            code="vllm_qualification.child_partial",
            context={"path": str(directory)},
        )
    paths = sorted(path for path in directory.rglob("*") if path.is_file())
    if not paths or len(paths) > MAX_EVIDENCE_ARTIFACTS:
        _fail(
            "qualification child evidence artifact count is invalid",
            code="vllm_qualification.evidence_artifacts",
            context={"count": len(paths)},
        )
    result: list[dict[str, object]] = []
    for path in paths:
        size = path.stat().st_size
        if size > MAX_EVIDENCE_ARTIFACT_BYTES:
            _fail(
                "qualification evidence artifact exceeds its size bound",
                code="vllm_qualification.evidence_artifact_unbounded",
                context={"path": str(path), "bytes": size},
            )
        result.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": size,
                "sha256": _sha256_file(path),
            }
        )
    return result


def _validate_identity(value: Mapping[str, object]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"source", "config", "model", "runtime"}:
        _fail("qualification identity is malformed", code="vllm_qualification.identity")
    payload = json.loads(json.dumps(value, sort_keys=True))
    for name in ("source", "config", "model", "runtime"):
        if not isinstance(payload[name], dict):
            _fail("qualification identity is malformed", code="vllm_qualification.identity")
    _require_sha256(payload["source"].get("fingerprint"), "identity", "source.fingerprint")
    _require_sha256(payload["config"].get("fingerprint"), "identity", "config.fingerprint")
    launch_contracts = payload["config"].get("launch_contracts")
    if not isinstance(launch_contracts, dict) or set(launch_contracts) != {"seq1", "seq4"}:
        _fail("qualification launch contracts are malformed", code="vllm_qualification.identity")
    for mode, contract in launch_contracts.items():
        if not isinstance(contract, dict) or set(contract) != {"payload", "fingerprint"}:
            _fail("qualification launch contract is malformed", code="vllm_qualification.identity")
        if contract["fingerprint"] != _sha256_json(contract["payload"]):
            _fail(
                "qualification launch contract fingerprint is malformed",
                code="vllm_qualification.identity",
                context={"mode": mode},
            )
    for field in ("composition_key", "snapshot_fingerprint", "receipt_fingerprint"):
        _require_sha256(payload["model"].get(field), "identity", f"model.{field}")
    if payload["model"].get("target_dtype") not in ("bf16", "fp16", "fp32"):
        _fail("qualification model dtype is malformed", code="vllm_qualification.identity")
    if not isinstance(payload["runtime"].get("packages"), dict):
        _fail("qualification runtime identity is malformed", code="vllm_qualification.identity")
    return payload


def _runtime_source_identity() -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for package, module in _RUNTIME_MODULES:
        spec = importlib.util.find_spec(module)
        origin = None if spec is None else spec.origin
        if not origin or not Path(origin).is_file():
            _fail(
                "qualification runtime module source is unavailable",
                code="vllm_qualification.runtime_source",
                context={"module": module},
            )
        package_spec = importlib.util.find_spec(package)
        package_root = Path(package_spec.origin).parent if package_spec and package_spec.origin else Path(origin).parent
        result.append(
            {
                "package": package,
                "relative_path": Path(origin).resolve().relative_to(package_root.resolve().parent).as_posix(),
                "sha256": _sha256_file(Path(origin)),
            }
        )
    return result


def _validate_numeric_summary(value: object, kind: str, field: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "allclose",
        "max_abs_diff",
        "max_rel_diff",
        "compared_value_count",
    }:
        _receipt_failure(kind, field, value)
    if (
        not isinstance(value.get("allclose"), bool)
        or not _finite_number(value.get("max_abs_diff"))
        or not _finite_number(value.get("max_rel_diff"))
        or not _positive_int(value.get("compared_value_count"))
    ):
        _receipt_failure(kind, field, value)
    if value["max_abs_diff"] < 0 or value["max_rel_diff"] < 0:
        _receipt_failure(kind, field, "negative_difference")


def _validate_artifact_entry(value: object, *, kind: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {"path", "bytes", "sha256"}:
        _receipt_failure(kind, "artifact", value)
    relative = value.get("path")
    if (
        not isinstance(relative, str)
        or not relative.startswith(f"evidence/{kind}/")
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or isinstance(value.get("bytes"), bool)
        or not isinstance(value.get("bytes"), int)
        or value["bytes"] < 0
        or value["bytes"] > MAX_EVIDENCE_ARTIFACT_BYTES
    ):
        _receipt_failure(kind, "artifact", value)
    _require_sha256(value.get("sha256"), kind, "artifact.sha256")


def _receipt_digest(receipt: Mapping[str, object]) -> str:
    return _sha256_json({key: value for key, value in receipt.items() if key != "digest"})


def _atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _contained_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        _fail(
            "qualification artifact path escapes its root",
            code="vllm_backend.qualification_artifact",
            context={"path": relative},
            cause=exc,
        )
    return path


def _optional_fingerprint(value: object) -> str | None:
    return value.get("fingerprint") if isinstance(value, Mapping) else None


def _positive_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _finite_number(value: object) -> TypeGuard[int | float]:
    import math

    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _require_sha256(value: object, kind: str, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        _receipt_failure(kind, field, value)
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _admission_drift(path: Path, reason: str) -> NoReturn:
    _fail(
        "admission target is not absent or identical",
        code="vllm_qualification.admission_target_drift",
        context={"path": str(path), "reason": reason},
    )


def _receipt_failure(kind: str, field: str, value: object) -> NoReturn:
    _fail(
        "vLLM qualification receipt is incompatible",
        code="vllm_backend.qualification_receipt_incompatible",
        context={"kind": kind, "field": field, "value": value},
    )


def _fail(
    message: str,
    *,
    code: str,
    context: Mapping[str, object] | None = None,
    cause: BaseException | None = None,
) -> NoReturn:
    raise RuntimeContractError(message, code=code, context=context, cause=cause)


__all__ = [
    "CONTRACT_VERSION",
    "DEFAULT_ADMISSION_ROOT",
    "EXPECTED_RECEIPT_FILENAMES",
    "MODE_SPECS",
    "SCHEMA_VERSION",
    "ChildExecution",
    "ChildSpec",
    "QualificationDependencies",
    "admit",
    "build_qualification_identity",
    "produce",
    "validate_receipt_set",
]
