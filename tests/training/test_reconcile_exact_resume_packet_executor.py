from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO_ROOT
    / "scripts/probes/coordexp_swift/reconcile_exact_resume_packet_executor.py"
)
ORDER = [
    "setup",
    "success.uninterrupted_control",
    "success.resumed_child",
    "rank_failure",
    "interruption",
    "verification",
]
OUTER_SCHEMA = "coordexp-swift-reconcile-resume-probe-outer-terminal-receipt-v1"
INNER_SCHEMA = "coordexp-swift-reconcile-resume-probe-terminal-receipt-v1"
REVIEW_SCHEMA = "coordexp-swift-reconcile-resume-probe-pre-cost-review-v1"


class _FinishedProcess:
    def __init__(self, returncode: int) -> None:
        self.pid = 4000
        self.starttime = 50
        self.process_group_id = 4000
        self.returncode = returncode

    def poll(self) -> int:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _signed(body: dict[str, Any], field: str = "receipt_payload_sha256") -> dict[str, Any]:
    return {**body, field: _sha256(_canonical(body))}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value) + b"\n")


@pytest.fixture(scope="module")
def executor() -> ModuleType:
    assert SCRIPT.is_file(), "packet executor is not implemented yet"
    spec = importlib.util.spec_from_file_location(
        "reconcile_exact_resume_packet_executor", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _init_repo(root: Path) -> str:
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "packet-executor@example.invalid")
    _git(root, "config", "user.name", "Packet Executor Test")
    (root / "tracked.txt").write_text("clean\n", encoding="utf-8")
    _git(root, "add", "tracked.txt")
    _git(root, "commit", "-qm", "fixture")
    return _git(root, "rev-parse", "HEAD")


def _process_rows(*, include_rank1: bool = True, reused: bool = False) -> list[dict[str, Any]]:
    rows = [
        {
            "pid": 4100,
            "ppid": 4000,
            "starttime": 100,
            "rank": 0,
            "rss_bytes": 16,
        },
    ]
    if include_rank1:
        rows.append(
            {
                "pid": 4101,
                "ppid": 4000,
                "starttime": 100,
                "rank": 1,
                "rss_bytes": 18,
            }
        )
    if reused:
        rows.append(
            {
                "pid": 4100,
                "ppid": 4000,
                "starttime": 101,
                "rank": 0,
                "rss_bytes": 24,
            }
        )
    return rows


def _gpu_rows(*, include_rank1: bool = True) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "kind": "device",
            "physical_index": "0",
            "gpu_uuid": "GPU-a",
            "memory_used_bytes": 0,
        },
        {
            "kind": "device",
            "physical_index": "1",
            "gpu_uuid": "GPU-b",
            "memory_used_bytes": 0,
        },
        {
            "kind": "process",
            "gpu_uuid": "GPU-a",
            "pid": 4100,
            "starttime": 100,
            "rank": 0,
            "gpu_memory_bytes": 32,
        },
    ]
    if include_rank1:
        rows.append(
            {
                "kind": "process",
                "gpu_uuid": "GPU-b",
                "pid": 4101,
                "starttime": 100,
                "rank": 1,
                "gpu_memory_bytes": 36,
            }
        )
    return rows


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    repo = tmp_path / "repo"
    commit = _init_repo(repo)
    monkeypatch.chdir(repo)
    packet = repo / "packet.md"
    packet.write_text("frozen packet\n", encoding="utf-8")
    packet_sha = _sha256(packet.read_bytes())
    artifact_root = repo / "artifact"
    marker = repo / "attempt-marker.json"
    outer = repo / "outer-terminal-receipt.json"
    review = repo / "pre-cost-review.json"
    cache_root = repo / ".artifact.pack-cache"
    cache_receipt = repo / ".artifact.pack-cache-receipt.json"
    inner = artifact_root / "terminal-receipt.json"
    config_paths = {
        role: artifact_root / "configs" / f"{role}.yaml"
        for role in ("uninterrupted_control", "resumed_parent", "resumed_child")
    }
    config_bytes = {role: f"role: {role}\n".encode() for role in config_paths}
    commands = {name: ["packet-fake", name] for name in ORDER}
    resource_bounds = {
        name: {
            "wall_time_seconds": 10.0,
            "max_cpu_rss_bytes_per_rank": 1024,
            "max_gpu_memory_bytes_per_rank": 1024 if name.startswith("success.") else 0,
            "max_new_artifact_bytes": 1024 * 1024,
            "required_cpu_ranks": (
                [0, 1]
                if name
                in {
                    "success.uninterrupted_control",
                    "success.resumed_child",
                    "rank_failure",
                    "interruption",
                }
                else []
            ),
            "required_gpu_ranks": [0, 1] if name.startswith("success.") else [],
        }
        for name in ORDER
    }
    contract = {
        "command_order": ORDER,
        "marker_creation": "O_EXCL",
        "stop_on_first_failure": True,
        "retry_count": 0,
        "required_resource_observations": [
            "wall_time",
            "cpu_rss_per_rank",
            "gpu_memory_per_rank",
            "artifact_bytes",
        ],
        "required_outer_receipt_bindings": [
            "implementation_commit",
            "manifest_sha256",
            "packet_sha256",
            "marker_sha256",
            "argv_observations",
            "resource_maxima",
            "pre_cost_review",
            "launcher_identity",
            "runtime_identity",
            "command_process_groups",
            "artifact_tree_summaries",
            "cleanup_outcomes",
            "stop_outcome",
        ],
        "required_inner_receipt_bindings": [
            "schema",
            "status",
            "commit",
            "artifact_root",
            "world_size",
            "receipt_payload_sha256",
        ],
        "bindings": {
            "packet_path": str(packet),
            "packet_sha256": packet_sha,
        },
        "pre_cost_review": {
            "path": str(review),
            "schema": REVIEW_SCHEMA,
            "required_status": "READY",
            "packet_author_identity": "packet-author",
        },
        "targets": {
            "artifact_root": str(artifact_root),
            "attempt_marker_path": str(marker),
            "terminal_receipt_path": str(outer),
            "must_be_absent": [
                str(artifact_root),
                str(cache_root),
                str(cache_receipt),
                str(marker),
                str(outer),
            ],
        },
        "preflight": {
            "filesystem_path": str(repo),
            "required_free_disk_bytes": 1,
            "gpu_devices": [
                {"physical_index": "0", "uuid": "GPU-a"},
                {"physical_index": "1", "uuid": "GPU-b"},
            ],
            "max_gpu_occupancy_bytes": 1024,
        },
        "resource_bounds": resource_bounds,
        "total_bounds": {
            "wall_time_seconds": 60.0,
            "max_new_artifact_bytes": 8 * 1024 * 1024,
        },
        "setup_validation": {
            "prepare_receipt_path": str(artifact_root / "prepare-receipt.json"),
            "pack_cache_root": str(cache_root),
            "pack_cache_receipt_path": str(cache_receipt),
        },
        "inner_receipt": {
            "path": str(inner),
            "schema": INNER_SCHEMA,
            "required_status": "verified",
        },
        "artifact_tree_summary": {
            "roots": [str(artifact_root), str(cache_root), str(cache_receipt)],
            "max_entries": 100,
            "max_depth": 8,
            "max_path_bytes": 512,
            "max_total_bytes": 8 * 1024 * 1024,
        },
    }
    manifest = {
        "schema": "coordexp-swift-reconcile-resume-probe-command-manifest-v2",
        "implementation_commit": commit,
        "cwd": str(repo),
        "world_size": 2,
        "arms": ["success", "rank_failure", "interruption"],
        "setup_command": commands["setup"],
        "commands": {
            "success": {
                "uninterrupted_control": commands["success.uninterrupted_control"],
                "resumed_child": commands["success.resumed_child"],
            },
            "rank_failure": commands["rank_failure"],
            "interruption": commands["interruption"],
        },
        "config_files": {
            "success": {
                "uninterrupted_control": {
                    "path": str(config_paths["uninterrupted_control"]),
                    "expected_sha256": _sha256(config_bytes["uninterrupted_control"]),
                },
                "resumed_child": {
                    "setup_parent": {
                        "path": str(config_paths["resumed_parent"]),
                        "expected_sha256": _sha256(config_bytes["resumed_parent"]),
                    },
                    "resumed_child": {
                        "path": str(config_paths["resumed_child"]),
                        "expected_sha256": _sha256(config_bytes["resumed_child"]),
                    },
                },
            },
            "rank_failure": None,
            "interruption": None,
        },
        "artifact_root": str(artifact_root),
        "verification_command": commands["verification"],
        "authorization_status": "READY",
        "execution_contract": contract,
    }
    manifest_path = repo / "manifest.json"
    _write_json(manifest_path, manifest)
    manifest_sha = _sha256(manifest_path.read_bytes())
    review_body = {
        "schema": REVIEW_SCHEMA,
        "status": "READY",
        "implementation_commit": commit,
        "manifest_sha256": manifest_sha,
        "packet_sha256": packet_sha,
        "reviewer_identity": "independent-reviewer",
    }
    _write_json(review, _signed(review_body))
    review.chmod(0o444)

    def write_setup(*, tamper: str | None = None) -> None:
        artifact_root.mkdir()
        (artifact_root / "configs").mkdir()
        for role, path in config_paths.items():
            path.write_bytes(config_bytes[role])
        cache_root.mkdir()
        (cache_root / "manifest.json").write_text("{}\n", encoding="utf-8")
        cache_body = {
            "terminal_status": "completed",
            "result": {"resolved_config_fingerprint": "cache-fingerprint"},
        }
        _write_json(cache_receipt, _signed(cache_body, field="receipt_sha256"))
        prepare_body = {
            "schema": "coordexp-swift-reconcile-resume-probe-prepare-receipt-v1",
            "status": "prepared",
            "commit": commit,
            "world_size": 2,
            "artifact_root": str(artifact_root),
            "base_config": {"path": str(repo / "base.yaml"), "file_sha256": "0" * 64},
            "configs": {
                role: {
                    "path": str(config_paths[role]),
                    "file_sha256": _sha256(config_bytes[role]),
                    "resolved_config_fingerprint": f"fingerprint-{role}",
                }
                for role in config_paths
            },
            "pack_cache": {
                "root": str(cache_root),
                "status": "prepared",
                "receipt_path": str(cache_receipt),
                "receipt_sha256": _sha256(cache_receipt.read_bytes()),
                "prepared_for_role": "uninterrupted_control",
                "shared_with_roles": ["resumed_parent", "resumed_child"],
                "resolved_config_fingerprint": "cache-fingerprint",
            },
        }
        prepare = _signed(prepare_body)
        if tamper == "prepare_digest":
            prepare["status"] = "tampered"
        elif tamper == "config":
            config_paths["resumed_child"].write_text("tampered\n", encoding="utf-8")
        elif tamper == "cache":
            cache_receipt.write_text("{}\n", encoding="utf-8")
        _write_json(artifact_root / "prepare-receipt.json", prepare)

    def write_inner(*, status: str = "verified", valid_digest: bool = True) -> None:
        body = {
            "schema": INNER_SCHEMA,
            "status": status,
            "commit": commit,
            "artifact_root": str(artifact_root),
            "world_size": 2,
            "required_comparisons": ["boundary", "post_update"],
            "comparison_policy": "fixture",
            "missing_inputs": [],
            "bounded_mismatches": [],
            "input_file_sha256": {},
        }
        receipt = _signed(body)
        if not valid_digest:
            receipt["receipt_payload_sha256"] = "f" * 64
        _write_json(inner, receipt)

    def launch(
        argv: list[str],
        *,
        cwd: Path,
        fail_name: str | None = None,
        setup_tamper: str | None = None,
        inner_status: str = "verified",
        valid_inner_digest: bool = True,
        artifact_bytes: int = 0,
    ) -> _FinishedProcess:
        del cwd
        name = argv[1]
        if name == "setup":
            write_setup(tamper=setup_tamper)
        elif artifact_bytes:
            (artifact_root / f"{name}.bin").write_bytes(b"x" * artifact_bytes)
        if name == "verification" and fail_name != "missing_inner":
            write_inner(status=inner_status, valid_digest=valid_inner_digest)
        return _FinishedProcess(returncode=1 if name == fail_name else 0)

    return {
        "repo": repo,
        "commit": commit,
        "packet": packet,
        "packet_sha": packet_sha,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha": manifest_sha,
        "review": review,
        "artifact_root": artifact_root,
        "marker": marker,
        "outer": outer,
        "cache_root": cache_root,
        "cache_receipt": cache_receipt,
        "inner": inner,
        "launch": launch,
        "write_setup": write_setup,
        "write_inner": write_inner,
    }


def _refresh_manifest_and_review(case: dict[str, Any]) -> None:
    _write_json(case["manifest_path"], case["manifest"])
    case["manifest_sha"] = _sha256(case["manifest_path"].read_bytes())
    _rewrite_review(case, manifest_sha256=case["manifest_sha"])


def _rewrite_review(case: dict[str, Any], **updates: Any) -> None:
    review = json.loads(case["review"].read_text(encoding="utf-8"))
    review.pop("receipt_payload_sha256", None)
    review.update(updates)
    case["review"].chmod(0o644)
    _write_json(case["review"], _signed(review))
    case["review"].chmod(0o444)


def _execute(
    executor: ModuleType,
    case: dict[str, Any],
    *,
    launch: Callable[..., Any] | None = None,
    gpu_sampler: Callable[[], list[dict[str, Any]]] | None = None,
    process_sampler: Callable[[], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    return executor.execute(
        manifest_path=case["manifest_path"],
        packet_path=case["packet"],
        expected_manifest_sha256=case["manifest_sha"],
        expected_packet_sha256=case["packet_sha"],
        attempt_marker_path=case["marker"],
        terminal_receipt_path=case["outer"],
        launch=launch or case["launch"],
        gpu_sampler=gpu_sampler or (lambda: _gpu_rows()),
        process_sampler=process_sampler or (lambda: _process_rows()),
    )


def _launch_sleep(cwd: Path, launched: list[subprocess.Popen[Any]]) -> subprocess.Popen[Any]:
    process = subprocess.Popen(["/bin/sleep", "30"], cwd=cwd, start_new_session=True)
    launched.append(process)
    return process


@pytest.mark.parametrize("drift", ["schema", "manifest_digest", "packet_digest", "head", "target"])
def test_schema_digest_head_and_target_drift_fail_before_marker(
    executor: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    case = _fixture(tmp_path, monkeypatch)
    if drift == "schema":
        case["manifest"]["schema"] = "v1"
        _write_json(case["manifest_path"], case["manifest"])
        case["manifest_sha"] = _sha256(case["manifest_path"].read_bytes())
    elif drift == "manifest_digest":
        case["manifest_sha"] = "0" * 64
    elif drift == "packet_digest":
        case["packet_sha"] = "0" * 64
    elif drift == "head":
        case["manifest"]["implementation_commit"] = "0" * 40
        _write_json(case["manifest_path"], case["manifest"])
        case["manifest_sha"] = _sha256(case["manifest_path"].read_bytes())
    else:
        case["artifact_root"].mkdir()
    calls: list[list[str]] = []

    with pytest.raises(executor.PacketExecutorError):
        _execute(executor, case, launch=lambda argv, **_: calls.append(argv))

    assert calls == []
    assert not case["marker"].exists()
    assert not case["outer"].exists()


def test_tracked_diff_fails_before_marker(
    executor: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _fixture(tmp_path, monkeypatch)
    (case["repo"] / "tracked.txt").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(executor.PacketExecutorError, match="tracked"):
        _execute(executor, case)

    assert not case["marker"].exists()


def test_manifest_self_ready_without_independent_review_fails_before_marker(
    executor: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _fixture(tmp_path, monkeypatch)
    case["review"].unlink()

    with pytest.raises(executor.PacketExecutorError, match="review"):
        _execute(executor, case)

    assert not case["marker"].exists()
    assert not case["outer"].exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "HOLD"),
        ("implementation_commit", "0" * 40),
        ("manifest_sha256", "0" * 64),
        ("packet_sha256", "0" * 64),
        ("reviewer_identity", "packet-author"),
    ],
)
def test_stale_hold_or_non_independent_review_fails_before_marker(
    executor: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    case = _fixture(tmp_path, monkeypatch)
    _rewrite_review(case, **{field: value})

    with pytest.raises(executor.PacketExecutorError, match="review"):
        _execute(executor, case)

    assert not case["marker"].exists()


def test_review_digest_mismatch_fails_before_marker(
    executor: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _fixture(tmp_path, monkeypatch)
    review = json.loads(case["review"].read_text(encoding="utf-8"))
    review["receipt_payload_sha256"] = "f" * 64
    case["review"].chmod(0o644)
    _write_json(case["review"], review)
    case["review"].chmod(0o444)

    with pytest.raises(executor.PacketExecutorError, match="review"):
        _execute(executor, case)

    assert not case["marker"].exists()


def test_exact_signed_independent_ready_review_is_bound_in_outer_receipt(
    executor: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _fixture(tmp_path, monkeypatch)

    receipt = _execute(executor, case)

    assert receipt["status"] == "verified"
    assert receipt["pre_cost_review"]["path"] == str(case["review"])
    assert receipt["pre_cost_review"]["payload"]["status"] == "READY"
    assert (
        receipt["pre_cost_review"]["payload"]["reviewer_identity"]
        == "independent-reviewer"
    )


def test_existing_marker_is_terminal_and_launches_nothing(
    executor: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _fixture(tmp_path, monkeypatch)
    case["marker"].write_text("owned\n", encoding="utf-8")
    calls: list[list[str]] = []

    with pytest.raises(executor.PacketExecutorError):
        _execute(executor, case, launch=lambda argv, **_: calls.append(argv))

    assert calls == []
    assert case["marker"].read_text(encoding="utf-8") == "owned\n"
    assert not case["outer"].exists()


def test_runs_exact_six_commands_once_in_order_and_verifies(
    executor: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _fixture(tmp_path, monkeypatch)
    calls: list[str] = []

    def launch(argv: list[str], **kwargs: Any) -> Any:
        calls.append(argv[1])
        return case["launch"](argv, **kwargs)

    receipt = _execute(executor, case, launch=launch)

    assert calls == ORDER
    assert receipt["schema"] == OUTER_SCHEMA
    assert receipt["status"] == "verified"
    assert receipt["attempted_order"] == ORDER
    assert receipt["completed_order"] == ORDER
    assert [row["name"] for row in receipt["argv_observations"]] == ORDER
    body = {key: value for key, value in receipt.items() if key != "receipt_payload_sha256"}
    assert receipt["receipt_payload_sha256"] == _sha256(_canonical(body))
    assert receipt["inner_receipt"]["status"] == "verified"


@pytest.mark.parametrize("failure_index", [0, 2, 4])
def test_failure_n_blocks_n_plus_one_and_never_retries(
    executor: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_index: int,
) -> None:
    case = _fixture(tmp_path, monkeypatch)
    calls: list[str] = []
    failed = ORDER[failure_index]

    def launch(argv: list[str], **kwargs: Any) -> Any:
        calls.append(argv[1])
        return case["launch"](argv, fail_name=failed, **kwargs)

    receipt = _execute(executor, case, launch=launch)

    assert calls == ORDER[: failure_index + 1]
    assert calls.count(failed) == 1
    assert receipt["status"] == "stopped"
    assert receipt["stop_outcome"]["command"] == failed
    assert case["outer"].is_file()


def test_launch_exception_stops_once_and_records_failed_argv(
    executor: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _fixture(tmp_path, monkeypatch)
    calls: list[str] = []

    def launch(argv: list[str], **kwargs: Any) -> Any:
        calls.append(argv[1])
        if argv[1] == "success.uninterrupted_control":
            raise RuntimeError("injected launch failure")
        return case["launch"](argv, **kwargs)

    receipt = _execute(executor, case, launch=launch)

    assert calls == ORDER[:2]
    assert [row["name"] for row in receipt["argv_observations"]] == ORDER[:2]
    assert receipt["status"] == "stopped"
    assert receipt["stop_outcome"]["code"] == "packet_executor.launch_exception"


def test_non_popen_launch_result_is_rejected_and_stops_next_command(
    executor: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _fixture(tmp_path, monkeypatch)
    calls: list[str] = []

    def launch(argv: list[str], **_kwargs: Any) -> Any:
        calls.append(argv[1])
        return {"returncode": 0}

    receipt = _execute(executor, case, launch=launch)

    assert calls == ["setup"]
    assert receipt["status"] == "stopped"
    assert receipt["stop_outcome"]["code"] == "packet_executor.launch_protocol"


@pytest.mark.parametrize("failing_sampler", ["process", "gpu"])
def test_real_sleep_sampler_exception_is_cleaned_before_terminal_receipt(
    executor: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failing_sampler: str,
) -> None:
    case = _fixture(tmp_path, monkeypatch)
    launched: list[subprocess.Popen[Any]] = []
    process_calls = 0
    gpu_calls = 0

    def launch(_argv: list[str], *, cwd: Path) -> subprocess.Popen[Any]:
        return _launch_sleep(cwd, launched)

    def process_sampler() -> list[dict[str, Any]]:
        nonlocal process_calls
        process_calls += 1
        if failing_sampler == "process" and process_calls >= 2:
            raise RuntimeError("injected process sampler failure")
        return executor._default_process_sampler()

    def gpu_sampler() -> list[dict[str, Any]]:
        nonlocal gpu_calls
        gpu_calls += 1
        if failing_sampler == "gpu" and gpu_calls >= 3:
            raise RuntimeError("injected GPU sampler failure")
        return _gpu_rows()

    try:
        receipt = _execute(
            executor,
            case,
            launch=launch,
            process_sampler=process_sampler,
            gpu_sampler=gpu_sampler,
        )
        assert launched
        assert launched[0].poll() is not None
        assert receipt["status"] == "stopped"
        assert receipt["commands"][0]["cleanup"]["status"] == "confirmed_absent"
        assert case["outer"].stat().st_mtime_ns >= case["marker"].stat().st_mtime_ns
    finally:
        for process in launched:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)


def test_real_sleep_timeout_is_cleaned_and_reaped_before_stopped_receipt(
    executor: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _fixture(tmp_path, monkeypatch)
    case["manifest"]["execution_contract"]["resource_bounds"]["setup"][
        "wall_time_seconds"
    ] = 0.01
    _refresh_manifest_and_review(case)
    launched: list[subprocess.Popen[Any]] = []

    def launch(_argv: list[str], *, cwd: Path) -> subprocess.Popen[Any]:
        return _launch_sleep(cwd, launched)

    try:
        receipt = _execute(executor, case, launch=launch)
        assert launched[0].poll() is not None
        assert receipt["status"] == "stopped"
        assert receipt["stop_outcome"]["code"] == "packet_executor.wall_timeout"
        assert receipt["commands"][0]["cleanup"]["reaped"] is True
        assert receipt["commands"][0]["cleanup"]["status"] == "confirmed_absent"
    finally:
        for process in launched:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)


def test_cleanup_absence_failure_is_explicit_terminal_failure(
    executor: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        executor, "_process_group_absent", lambda **_kwargs: False, raising=False
    )

    receipt = _execute(executor, case)

    assert receipt["status"] == "stopped"
    assert receipt["stop_outcome"]["code"] == "packet_executor.cleanup_failed"
    assert receipt["commands"][0]["cleanup"]["status"] == "failed"


def test_post_launch_identity_failure_still_terminates_and_reaps_process(
    executor: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _fixture(tmp_path, monkeypatch)
    process = _FinishedProcess(0)
    del process.starttime
    terminated: list[bool] = []

    def terminate() -> None:
        terminated.append(True)
        process.returncode = -15

    process.returncode = None
    process.terminate = terminate  # type: ignore[method-assign]

    receipt = _execute(executor, case, launch=lambda *_args, **_kwargs: process)

    assert terminated == [True]
    assert process.poll() is not None
    assert receipt["status"] == "stopped"
    assert receipt["commands"][0]["cleanup"]["status"] == "failed"
    assert receipt["stop_outcome"]["code"] == "packet_executor.cleanup_failed"


@pytest.mark.parametrize("tamper", ["prepare_digest", "config", "cache"])
def test_setup_mismatch_stops_before_first_gpu_command(
    executor: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    case = _fixture(tmp_path, monkeypatch)
    calls: list[str] = []

    def launch(argv: list[str], **kwargs: Any) -> Any:
        calls.append(argv[1])
        return case["launch"](argv, setup_tamper=tamper, **kwargs)

    receipt = _execute(executor, case, launch=launch)

    assert calls == ["setup"]
    assert receipt["status"] == "stopped"
    assert receipt["stop_outcome"]["code"].startswith("packet_executor.setup_")


@pytest.mark.parametrize("bound", ["free_disk", "gpu_occupancy", "wall", "rss", "gpu", "artifact"])
def test_resource_bound_failure_stops_later_commands(
    executor: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bound: str,
) -> None:
    case = _fixture(tmp_path, monkeypatch)
    contract = case["manifest"]["execution_contract"]

    def gpu_sampler() -> list[dict[str, Any]]:
        return _gpu_rows()

    def process_sampler() -> list[dict[str, Any]]:
        return _process_rows()

    launch_kwargs: dict[str, Any] = {}
    if bound == "free_disk":
        contract["preflight"]["required_free_disk_bytes"] = 10**30
    elif bound == "gpu_occupancy":
        contract["preflight"]["max_gpu_occupancy_bytes"] = 0

        def gpu_sampler() -> list[dict[str, Any]]:
            return [
                {
                    "kind": "device",
                    "physical_index": "0",
                    "gpu_uuid": "GPU-a",
                    "memory_used_bytes": 1,
                },
                {
                    "kind": "device",
                    "physical_index": "1",
                    "gpu_uuid": "GPU-b",
                    "memory_used_bytes": 0,
                },
            ]
    elif bound == "wall":
        contract["resource_bounds"]["setup"]["wall_time_seconds"] = 0.0
    elif bound == "rss":
        contract["resource_bounds"]["success.uninterrupted_control"][
            "max_cpu_rss_bytes_per_rank"
        ] = 1
    elif bound == "gpu":
        contract["resource_bounds"]["success.uninterrupted_control"][
            "max_gpu_memory_bytes_per_rank"
        ] = 1
    else:
        contract["resource_bounds"]["success.uninterrupted_control"][
            "max_new_artifact_bytes"
        ] = 1
        launch_kwargs["artifact_bytes"] = 2
    _refresh_manifest_and_review(case)
    calls: list[str] = []

    def launch(argv: list[str], **kwargs: Any) -> Any:
        calls.append(argv[1])
        return case["launch"](argv, **launch_kwargs, **kwargs)

    receipt = _execute(
        executor,
        case,
        launch=launch,
        gpu_sampler=gpu_sampler,
        process_sampler=process_sampler,
    )

    assert receipt["status"] == "stopped"
    if bound in {"free_disk", "gpu_occupancy"}:
        assert calls == []
    elif bound == "wall":
        assert calls == ["setup"]
    else:
        assert calls == ["setup", "success.uninterrupted_control"]


def test_pid_reuse_is_keyed_by_pid_and_starttime(
    executor: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _fixture(tmp_path, monkeypatch)
    receipt = _execute(
        executor,
        case,
        process_sampler=lambda: _process_rows(reused=True),
    )

    identities = {
        (row["pid"], row["starttime"])
        for command in receipt["commands"]
        for row in command["process_observations"]
    }
    assert (4100, 100) in identities
    assert (4100, 101) in identities
    assert receipt["resource_maxima"]["cpu_rss_bytes_per_rank"]["0"] == 24


def test_physical_index_uuid_pair_drift_fails_before_marker(
    executor: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _fixture(tmp_path, monkeypatch)
    swapped = _gpu_rows()
    swapped[0]["physical_index"] = "1"
    swapped[1]["physical_index"] = "0"

    with pytest.raises(executor.PacketExecutorError, match="GPU"):
        _execute(executor, case, gpu_sampler=lambda: swapped)

    assert not case["marker"].exists()


def test_foreign_rank_tagged_gpu_rows_do_not_satisfy_current_command(
    executor: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _fixture(tmp_path, monkeypatch)
    foreign = _gpu_rows()
    for row in foreign:
        if row["kind"] == "process":
            row["pid"] += 9000

    receipt = _execute(executor, case, gpu_sampler=lambda: foreign)

    assert receipt["status"] == "stopped"
    assert receipt["stop_outcome"]["code"] == "packet_executor.missing_gpu_rank"


@pytest.mark.parametrize(
    "command",
    [
        "success.uninterrupted_control",
        "success.resumed_child",
        "rank_failure",
        "interruption",
    ],
)
def test_missing_required_cpu_rank_is_command_specific(
    executor: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    case = _fixture(tmp_path, monkeypatch)
    active = ""

    def launch(argv: list[str], **kwargs: Any) -> Any:
        nonlocal active
        active = argv[1]
        return case["launch"](argv, **kwargs)

    def process_sampler() -> list[dict[str, Any]]:
        return _process_rows(include_rank1=active != command)

    receipt = _execute(executor, case, launch=launch, process_sampler=process_sampler)

    assert receipt["status"] == "stopped"
    assert receipt["stop_outcome"]["command"] == command
    assert receipt["stop_outcome"]["code"] == "packet_executor.missing_process_rank"


def test_outer_receipt_binds_launcher_runtime_groups_and_bounded_artifact_summaries(
    executor: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _fixture(tmp_path, monkeypatch)

    receipt = _execute(executor, case)

    assert receipt["launcher_identity"]["qualname"]
    assert receipt["runtime_identity"]["python_executable"]
    assert receipt["artifact_tree"]["initial"]["roots"]
    assert receipt["artifact_tree"]["final"]["inventory_sha256"]
    assert all(row["process_group"]["leader_pid"] for row in receipt["commands"])
    assert all(row["cleanup"]["status"] == "confirmed_absent" for row in receipt["commands"])


@pytest.mark.parametrize("missing", ["process", "gpu"])
def test_missing_required_rank_measurement_stops(
    executor: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
) -> None:
    case = _fixture(tmp_path, monkeypatch)
    receipt = _execute(
        executor,
        case,
        process_sampler=(
            (lambda: _process_rows(include_rank1=False))
            if missing == "process"
            else (lambda: _process_rows())
        ),
        gpu_sampler=(
            (lambda: _gpu_rows(include_rank1=False))
            if missing == "gpu"
            else (lambda: _gpu_rows())
        ),
    )

    assert receipt["status"] == "stopped"
    assert receipt["attempted_order"] == ["setup", "success.uninterrupted_control"]
    assert receipt["stop_outcome"]["code"] == f"packet_executor.missing_{missing}_rank"


@pytest.mark.parametrize("inner", ["missing_inner", "failed", "bad_digest"])
def test_verifier_zero_without_valid_signed_verified_inner_receipt_stops(
    executor: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inner: str,
) -> None:
    case = _fixture(tmp_path, monkeypatch)

    def launch(argv: list[str], **kwargs: Any) -> Any:
        return case["launch"](
            argv,
            fail_name="missing_inner" if inner == "missing_inner" else None,
            inner_status="failed" if inner == "failed" else "verified",
            valid_inner_digest=inner != "bad_digest",
            **kwargs,
        )

    receipt = _execute(executor, case, launch=launch)

    assert receipt["commands"][-1]["returncode"] == 0
    assert receipt["status"] == "stopped"
    assert receipt["stop_outcome"]["code"].startswith("packet_executor.inner_")


def test_terminal_receipt_is_immutable_and_marker_is_signed(
    executor: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _fixture(tmp_path, monkeypatch)
    first = _execute(executor, case)
    receipt_bytes = case["outer"].read_bytes()
    marker = json.loads(case["marker"].read_text(encoding="utf-8"))
    marker_body = {
        key: value for key, value in marker.items() if key != "receipt_payload_sha256"
    }

    with pytest.raises(executor.PacketExecutorError):
        _execute(executor, case)

    assert first["status"] == "verified"
    assert case["outer"].read_bytes() == receipt_bytes
    assert marker["receipt_payload_sha256"] == _sha256(_canonical(marker_body))


def test_cli_execute_parses_exact_public_flags(
    executor: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def fake_execute(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"status": "verified"}

    monkeypatch.setattr(executor, "execute", fake_execute)
    result = executor.main(
        [
            "execute",
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--packet",
            str(tmp_path / "packet.md"),
            "--manifest-sha256",
            "a" * 64,
            "--packet-sha256",
            "b" * 64,
            "--attempt-marker",
            str(tmp_path / "marker.json"),
            "--terminal-receipt",
            str(tmp_path / "receipt.json"),
        ]
    )

    assert result == 0
    assert captured["manifest_path"] == tmp_path / "manifest.json"
    assert captured["packet_path"] == tmp_path / "packet.md"
    assert captured["attempt_marker_path"] == tmp_path / "marker.json"
    assert captured["terminal_receipt_path"] == tmp_path / "receipt.json"
