from __future__ import annotations

import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from scripts.probes.coordexp_swift import wave7_exact_resume_postrun as postrun


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _signed(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    value["receipt_payload_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value) + b"\n")
    return value


def _binding(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "receipt_payload_sha256": payload["receipt_payload_sha256"],
        "schema": payload["schema"],
        "status": payload["status"],
    }


def _receipts(tmp_path: Path) -> tuple[Path, Path, Path]:
    r5 = (tmp_path / "r5").resolve()
    child = r5 / "runs/resume_child"
    checkpoint = child / "checkpoints/step-5"
    checkpoint.mkdir(parents=True)
    identity = {
        "schema": "coordexp-swift-inference-checkpoint-payload-publication",
        "schema_version": 2,
        "manifest_relative_path": "inference_payload_manifest.json",
        "manifest_file_sha256": "a" * 64,
        "aggregate_digest": "b" * 64,
    }
    final_path = r5 / "final-comparison.json"
    final = _signed(
        final_path,
        {
            "schema": postrun.FINAL_COMPARISON_SCHEMA,
            "status": "passed",
            "mismatches": [],
            "runs": {
                "resume_child": {
                    "path": str(child),
                    "run_id": "resume-child",
                    "segment_id": "resume-child",
                    "hashes": {},
                }
            },
            "event_validation": {
                "resume_child": {
                    "event_versions": [2],
                    "events": [
                        {
                            "inference_payload_identity": identity,
                            "committed_progress": {
                                "schema": "coordexp-swift-committed-training-progress",
                                "schema_version": 1,
                                "completed_steps": 5,
                                "consumed_packs": 10,
                                "optimizer_update_status": "applied",
                                "finite_status": "finite",
                            },
                        }
                    ],
                }
            },
        },
    )
    sequence_path = r5 / "sequence-receipt.json"
    _signed(
        sequence_path,
        {
            "schema": postrun.SEQUENCE_SCHEMA,
            "status": "passed",
            "failure": None,
            "final_receipt": {
                **_binding(final_path, final),
                "validated": True,
                "mismatches": [],
            },
        },
    )
    return r5, sequence_path, final_path


@pytest.mark.parametrize("mutation", ("digest", "schema", "status", "duplicate"))
def test_receipt_authentication_rejects_drift(tmp_path: Path, mutation: str) -> None:
    r5, sequence_path, final_path = _receipts(tmp_path)
    if mutation == "duplicate":
        sequence_path.write_text('{"schema":"a","schema":"b"}\n', encoding="utf-8")
    else:
        value = json.loads(sequence_path.read_text(encoding="utf-8"))
        if mutation == "digest":
            value["receipt_payload_sha256"] = "0" * 64
        elif mutation == "schema":
            value["schema"] = "legacy"
        else:
            value["status"] = "failed"
        sequence_path.write_bytes(_canonical(value) + b"\n")

    with pytest.raises(postrun.Wave7PostrunError):
        postrun.authenticate_inputs(
            r5_root=r5,
            sequence_receipt=sequence_path,
            final_comparison=final_path,
        )


def test_sequence_schema_and_final_binding_match_live_v6_producer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    r5, sequence_path, final_path = _receipts(tmp_path)
    monkeypatch.setattr(
        postrun,
        "admit_inference_checkpoint_payload_identity",
        lambda _path, identity: dict(identity),
    )

    admitted = postrun.authenticate_inputs(
        r5_root=r5,
        sequence_receipt=sequence_path,
        final_comparison=final_path,
    )

    assert postrun.SEQUENCE_SCHEMA == (
        "coordexp-swift-wave7-exact-resume-sequence-receipt-v6"
    )
    assert set(admitted["sequence"]["final_receipt"]) == {
        "path",
        "file_sha256",
        "receipt_payload_sha256",
        "schema",
        "status",
        "mismatches",
        "validated",
    }
    assert (
        admitted["sequence_file_sha256"]
        == hashlib.sha256(sequence_path.read_bytes()).hexdigest()
    )
    assert (
        admitted["final_file_sha256"]
        == hashlib.sha256(final_path.read_bytes()).hexdigest()
    )


def test_proc_stat_parent_parser_handles_spaces_and_parentheses_in_comm() -> None:
    encoded = "123 (worker name) with ) chars) S 456 789 0 0 0"

    assert postrun._proc_stat_parent_pid(encoded) == 456


@pytest.mark.parametrize(
    "mutation",
    (
        "path",
        "file_sha256",
        "digest",
        "digest_field_name",
        "schema",
        "status",
        "mismatches",
        "validated",
        "missing",
        "unexpected",
    ),
)
def test_final_receipt_binding_rejects_each_producer_field_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    r5, sequence_path, final_path = _receipts(tmp_path)
    sequence = json.loads(sequence_path.read_text(encoding="utf-8"))
    unsigned = dict(sequence)
    unsigned.pop("receipt_payload_sha256")
    binding = unsigned["final_receipt"]
    if mutation == "path":
        binding["path"] = str(r5 / "other-final.json")
    elif mutation == "file_sha256":
        binding["file_sha256"] = "f" * 64
    elif mutation == "digest":
        binding["receipt_payload_sha256"] = "0" * 64
    elif mutation == "digest_field_name":
        binding["payload_sha256"] = binding.pop("receipt_payload_sha256")
    elif mutation == "schema":
        binding["schema"] = "legacy"
    elif mutation == "status":
        binding["status"] = "failed"
    elif mutation == "mismatches":
        binding["mismatches"] = [{"code": "drift"}]
    elif mutation == "validated":
        binding["validated"] = False
    elif mutation == "missing":
        binding.pop("validated")
    else:
        binding["unexpected"] = True
    _signed(sequence_path, unsigned)
    monkeypatch.setattr(
        postrun,
        "admit_inference_checkpoint_payload_identity",
        lambda _path, identity: dict(identity),
    )

    with pytest.raises(postrun.Wave7PostrunError) as caught:
        postrun.authenticate_inputs(
            r5_root=r5,
            sequence_receipt=sequence_path,
            final_comparison=final_path,
        )
    assert caught.value.code == "wave7_postrun.receipt_binding"


def test_authentication_rejects_non_latest_child_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    r5, sequence_path, final_path = _receipts(tmp_path)
    final = json.loads(final_path.read_text(encoding="utf-8"))
    unsigned = dict(final)
    unsigned.pop("receipt_payload_sha256")
    unsigned["event_validation"]["resume_child"]["events"].append(
        copy.deepcopy(unsigned["event_validation"]["resume_child"]["events"][0])
    )
    unsigned["event_validation"]["resume_child"]["events"][-1]["committed_progress"][
        "completed_steps"
    ] = 6
    final = _signed(final_path, unsigned)
    sequence = json.loads(sequence_path.read_text(encoding="utf-8"))
    sequence_unsigned = dict(sequence)
    sequence_unsigned.pop("receipt_payload_sha256")
    sequence_unsigned["final_receipt"] = {
        **_binding(final_path, final),
        "validated": True,
        "mismatches": [],
    }
    _signed(sequence_path, sequence_unsigned)

    with pytest.raises(postrun.Wave7PostrunError) as caught:
        postrun.authenticate_inputs(
            r5_root=r5,
            sequence_receipt=sequence_path,
            final_comparison=final_path,
        )
    assert caught.value.code == "wave7_postrun.checkpoint_selection"


@pytest.mark.parametrize(
    "symlink_component",
    ("runs", "resume_child", "checkpoints", "step-5"),
)
def test_authentication_rejects_each_symlinked_child_checkpoint_ancestor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    symlink_component: str,
) -> None:
    r5, sequence_path, final_path = _receipts(tmp_path)
    components = {
        "runs": r5 / "runs",
        "resume_child": r5 / "runs/resume_child",
        "checkpoints": r5 / "runs/resume_child/checkpoints",
        "step-5": r5 / "runs/resume_child/checkpoints/step-5",
    }
    link = components[symlink_component]
    outside = tmp_path / f"outside-{symlink_component}"
    link.rename(outside)
    link.symlink_to(outside, target_is_directory=True)

    final = json.loads(final_path.read_text(encoding="utf-8"))
    final_unsigned = dict(final)
    final_unsigned.pop("receipt_payload_sha256")
    final_unsigned["runs"]["resume_child"]["path"] = str(
        (r5 / "runs/resume_child").resolve()
    )
    final = _signed(final_path, final_unsigned)
    sequence = json.loads(sequence_path.read_text(encoding="utf-8"))
    sequence_unsigned = dict(sequence)
    sequence_unsigned.pop("receipt_payload_sha256")
    sequence_unsigned["final_receipt"] = {
        **_binding(final_path, final),
        "validated": True,
        "mismatches": [],
    }
    _signed(sequence_path, sequence_unsigned)
    monkeypatch.setattr(
        postrun,
        "admit_inference_checkpoint_payload_identity",
        lambda _path, identity: dict(identity),
    )

    with pytest.raises(postrun.Wave7PostrunError) as caught:
        postrun.authenticate_inputs(
            r5_root=r5,
            sequence_receipt=sequence_path,
            final_comparison=final_path,
        )
    assert caught.value.code == "wave7_postrun.path_topology"


def test_derived_config_is_fully_materialized_and_training_aligned(
    tmp_path: Path,
) -> None:
    r5, sequence_path, final_path = _receipts(tmp_path)
    checkpoint = r5 / "runs/resume_child/checkpoints/step-5"
    value = postrun.build_derived_config(r5_root=r5, checkpoint_dir=checkpoint)

    assert value["schema_version"] == 1
    assert value["run"] == {
        "name": "final-child-infer",
        "artifact_root": str((r5 / "postrun").resolve()),
        "output_dir": "final-child-infer",
        "collision_policy": "fail",
    }
    assert value["model"]["dtype"] == "bf16"
    assert value["model"]["processor"] == {"do_resize": False}
    assert value["backend"] == {
        "type": "hf",
        "hf": {
            "attn_implementation": "flash_attention_2",
            "patch_embed_linearization": "enabled",
        },
    }
    assert value["template"]["object_ordering"] == "geo_sorted"
    assert "COCO-80 class list" in value["template"]["prompt"]["user"]
    assert value["generation"] == {
        "batch_size": 1,
        "max_new_tokens": 512,
        "temperature": 0.0,
        "top_p": 1.0,
        "n": 1,
        "repetition_penalty": 1.0,
    }
    assert value["scoring"] == {"enabled": True}
    assert value["artifacts"] == {
        "write_token_trace": True,
        "write_parse_diagnostics": True,
        "include_raw_model_logprob": False,
    }
    assert value["debug"] == {"smoke": True, "dry_run": False}
    assert value["adapter"]["path"] == str(checkpoint / "adapter")
    assert value["embedding_delta"]["path"] == str(
        checkpoint / "special_token_embeddings"
    )


def test_derived_yaml_resolves_without_implicit_semantic_fields(tmp_path: Path) -> None:
    r5, _, _ = _receipts(tmp_path)
    checkpoint = r5 / "runs/resume_child/checkpoints/step-5"
    value = postrun.build_derived_config(r5_root=r5, checkpoint_dir=checkpoint)
    identity = postrun._write_config(r5 / "postrun-config.yaml", value)

    assert identity["config"] == value
    assert identity["resolved_fingerprint"]


def _sample(
    *, used: tuple[int, ...], compute: tuple[tuple[str, int], ...] = ()
) -> dict[str, Any]:
    return {
        "gpu_inventory": [
            {
                "index": index,
                "gpu_uuid": f"GPU-{index}",
                "memory_total_mib": 81_920,
                "memory_used_mib": memory,
                "memory_headroom_mib": 81_920 - memory,
                "utilization_gpu_percent": 0,
            }
            for index, memory in enumerate(used)
        ],
        "compute_inventory": [
            {"gpu_uuid": uuid, "driver_pid": pid} for uuid, pid in compute
        ],
    }


def test_gpu_selection_uses_lowest_stable_qualifying_physical_index() -> None:
    first = _sample(used=(60_000, 20_000, 10_000), compute=(("GPU-1", 101),))
    second = _sample(used=(59_000, 21_000, 11_000), compute=(("GPU-1", 101),))
    selected = postrun.select_gpu(first, second)
    assert selected == {
        "physical_index": 1,
        "gpu_uuid": "GPU-1",
        "preexisting_compute_inventory": [{"gpu_uuid": "GPU-1", "driver_pid": 101}],
    }


@pytest.mark.parametrize("mutation", ("uuid", "process", "no_capacity"))
def test_gpu_selection_rejects_unstable_or_unqualified_samples(mutation: str) -> None:
    first = _sample(used=(20_000, 30_000), compute=(("GPU-0", 100),))
    second = copy.deepcopy(first)
    if mutation == "uuid":
        second["gpu_inventory"][0]["gpu_uuid"] = "GPU-replaced"
    elif mutation == "process":
        second["compute_inventory"][0]["driver_pid"] = 101
    else:
        for sample in (first, second):
            for row in sample["gpu_inventory"]:
                row["memory_used_mib"] = 60_000
                row["memory_headroom_mib"] = 21_920
    with pytest.raises(postrun.Wave7PostrunError):
        postrun.select_gpu(first, second)


@pytest.mark.parametrize(
    ("stream", "byte_count"),
    (("stdout", 1024 * 1024 + 1), ("stderr", 64 * 1024 + 1)),
)
def test_bounded_nvidia_csv_rejects_large_stdout_and_stderr(
    stream: str, byte_count: int
) -> None:
    program = (
        "import sys; "
        f"sys.{stream}.buffer.write(b'x' * {byte_count}); "
        f"sys.{stream}.buffer.flush()"
    )

    with pytest.raises(postrun.Wave7PostrunError) as caught:
        postrun._bounded_nvidia_csv(
            [sys.executable, "-c", program], owner="synthetic GPU inventory"
        )

    assert caught.value.code == "wave7_postrun.gpu_output_oversize"
    assert caught.value.context == {
        "stream": stream,
        "total_bytes": byte_count,
        "maximum_bytes": 1024 * 1024 if stream == "stdout" else 64 * 1024,
    }


@pytest.mark.parametrize("observation", ("wrong_uuid", "wrong_ancestry", "absent"))
def test_gpu_observation_rejects_wrong_uuid_pid_ancestry_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
    observation: str,
) -> None:
    class Process:
        pid = 123

        def poll(self) -> None:
            return None

    compute = {
        "wrong_uuid": [{"gpu_uuid": "GPU-1", "driver_pid": 124}],
        "wrong_ancestry": [{"gpu_uuid": "GPU-0", "driver_pid": 999}],
        "absent": [],
    }[observation]
    monkeypatch.setattr(
        postrun,
        "_gpu_sample",
        lambda: {"gpu_inventory": [], "compute_inventory": compute},
    )
    monkeypatch.setattr(postrun, "_descendant_pids", lambda _pid: {123, 124})
    monotonic = iter((0.0, 0.0, 121.0))
    monkeypatch.setattr(postrun.time, "monotonic", lambda: next(monotonic))
    monkeypatch.setattr(postrun.time, "sleep", lambda _: None)

    with pytest.raises(postrun.Wave7PostrunError) as caught:
        postrun._wait_for_gpu_observation(
            process=Process(),
            selection={
                "physical_index": 0,
                "gpu_uuid": "GPU-0",
                "preexisting_compute_inventory": [],
            },
        )
    assert caught.value.code == "wave7_postrun.gpu_observation"


def test_gpu_cleanup_rejects_surviving_owned_driver_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    survivor = {"gpu_uuid": "GPU-0", "driver_pid": 777}
    monkeypatch.setattr(
        postrun,
        "_gpu_sample",
        lambda: {"gpu_inventory": [], "compute_inventory": [survivor]},
    )

    with pytest.raises(postrun.Wave7PostrunError) as caught:
        postrun._assert_gpu_cleanup(
            observed=[survivor],
            selection={
                "physical_index": 0,
                "gpu_uuid": "GPU-0",
                "preexisting_compute_inventory": [],
            },
        )
    assert caught.value.code == "wave7_postrun.gpu_cleanup"


def test_process_group_cleanup_fails_closed_when_owned_group_survives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals: list[tuple[int, int]] = []

    class Process:
        pid = 4321

        def poll(self) -> None:
            return None

        def wait(self, timeout: float) -> int:
            raise subprocess.TimeoutExpired("infer", timeout)

    monkeypatch.setattr(
        postrun.os,
        "killpg",
        lambda pid, signum: signals.append((pid, int(signum))),
    )

    with pytest.raises(postrun.Wave7PostrunError) as caught:
        postrun._terminate_process_group(Process(), term_grace=0.0, kill_grace=0.0)
    assert caught.value.code == "wave7_postrun.process_cleanup"
    assert signals == [
        (4321, 0),
        (4321, int(postrun.signal.SIGTERM)),
        (4321, 0),
        (4321, int(postrun.signal.SIGKILL)),
        (4321, 0),
    ]


def test_process_group_cleanup_rejects_surviving_group_after_leader_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals: list[tuple[int, int]] = []

    class Process:
        pid = 4322

        def poll(self) -> int:
            return 0

        def wait(self, timeout: float) -> int:
            return 0

    monkeypatch.setattr(
        postrun.os,
        "killpg",
        lambda pid, signum: signals.append((pid, int(signum))),
    )

    with pytest.raises(postrun.Wave7PostrunError) as caught:
        postrun._terminate_process_group(Process(), term_grace=0.0, kill_grace=0.0)
    assert caught.value.code == "wave7_postrun.process_cleanup"
    assert signals == [
        (4322, 0),
        (4322, int(postrun.signal.SIGTERM)),
        (4322, 0),
        (4322, int(postrun.signal.SIGKILL)),
        (4322, 0),
    ]


def test_process_group_cleanup_terminates_actual_owned_group() -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        shell=False,
    )
    try:
        assert postrun._process_group_exists(process.pid)
        postrun._terminate_process_group(process, term_grace=2.0, kill_grace=2.0)
        assert process.poll() is not None
        assert not postrun._process_group_exists(process.pid)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, postrun.signal.SIGKILL)
            process.wait(timeout=2.0)


def test_launch_contract_uses_exact_argv_env_shell_false_and_no_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []

    class Process:
        pid = 1234
        returncode = 0
        stdout = io.BytesIO()
        stderr = io.BytesIO()

        def wait(self, timeout: float) -> int:
            return 0

        def poll(self) -> int:
            return 0

    def popen(argv: list[str], **kwargs: Any) -> Process:
        calls.append({"argv": argv, **kwargs})
        return Process()

    monkeypatch.setattr(postrun.subprocess, "Popen", popen)
    monkeypatch.setattr(
        postrun, "_wait_for_gpu_observation", lambda **_: [{"driver_pid": 777}]
    )
    monkeypatch.setattr(postrun, "_assert_gpu_cleanup", lambda **_: [])
    monkeypatch.setattr(postrun, "_terminate_process_group", lambda *_a, **_k: None)
    config = (tmp_path / "config.yaml").resolve()
    config.write_text("schema_version: 1\n", encoding="utf-8")
    result = postrun.launch_inference(
        config_path=config,
        selection={
            "physical_index": 3,
            "gpu_uuid": "GPU-3",
            "preexisting_compute_inventory": [],
        },
        timeout_seconds=10.0,
    )
    assert result["return_code"] == 0
    assert len(calls) == 1
    call = calls[0]
    assert call["argv"] == [sys.executable, "-m", "src.infer", "--config", str(config)]
    assert call["shell"] is False
    assert call["start_new_session"] is True
    assert call["stdin"] is subprocess.DEVNULL
    assert call["stdout"] is subprocess.PIPE
    assert call["stderr"] is subprocess.PIPE
    assert call["cwd"] == postrun.REPO_ROOT
    assert call["env"]["CUDA_VISIBLE_DEVICES"] == "3"
    assert call["env"]["FLASH_ATTENTION_DETERMINISTIC"] == "1"
    assert call["env"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_launch_rejects_child_nonzero_and_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("schema_version: 1\n", encoding="utf-8")
    selection = {
        "physical_index": 0,
        "gpu_uuid": "GPU-0",
        "preexisting_compute_inventory": [],
    }

    class Process:
        pid = 99

        def __init__(self, timeout: bool) -> None:
            self.timeout = timeout
            self.returncode = None
            self.stdout = io.BytesIO()
            self.stderr = io.BytesIO()

        def wait(self, timeout: float) -> int:
            if self.timeout:
                raise subprocess.TimeoutExpired("infer", timeout)
            self.returncode = 9
            return 9

        def poll(self) -> int | None:
            return self.returncode

    monkeypatch.setattr(
        postrun, "_wait_for_gpu_observation", lambda **_: [{"driver_pid": 777}]
    )
    monkeypatch.setattr(postrun, "_assert_gpu_cleanup", lambda **_: [])
    monkeypatch.setattr(
        postrun, "_terminate_process_group", lambda *_args, **_kwargs: None
    )
    for timeout in (False, True):
        monkeypatch.setattr(
            postrun.subprocess, "Popen", lambda *_a, **_k: Process(timeout)
        )
        with pytest.raises(postrun.Wave7PostrunError):
            postrun.launch_inference(
                config_path=config,
                selection=selection,
                timeout_seconds=0.1,
            )


@pytest.mark.parametrize(
    ("outcome", "expected_code"),
    (
        ("nonzero", "wave7_postrun.child_nonzero"),
        ("timeout", "wave7_postrun.child_timeout"),
        ("cleanup", "wave7_postrun.process_cleanup"),
    ),
)
def test_launch_failure_preserves_bounded_diagnostics_and_exact_process_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
    expected_code: str,
) -> None:
    calls: list[dict[str, Any]] = []
    stdout = b"stdout-prefix\n" + b"o" * 70_000
    stderr = b"stderr-prefix\n" + b"e" * 70_000

    class Process:
        pid = 5100
        returncode: int | None = None

        def __init__(self) -> None:
            self.stdout = io.BytesIO(stdout)
            self.stderr = io.BytesIO(stderr)

        def wait(self, timeout: float) -> int:
            if outcome == "timeout":
                raise subprocess.TimeoutExpired("infer", timeout)
            self.returncode = 17 if outcome == "nonzero" else 0
            return self.returncode

        def poll(self) -> int | None:
            return self.returncode

    def popen(argv: list[str], **kwargs: Any) -> Process:
        calls.append({"argv": argv, **kwargs})
        return Process()

    monkeypatch.setattr(postrun.subprocess, "Popen", popen)
    monkeypatch.setattr(
        postrun, "_wait_for_gpu_observation", lambda **_: [{"driver_pid": 777}]
    )
    monkeypatch.setattr(postrun, "_assert_gpu_cleanup", lambda **_: [])

    def cleanup(*_args: Any, **_kwargs: Any) -> None:
        if outcome == "cleanup":
            raise postrun.Wave7PostrunError(
                "surviving process group",
                code="wave7_postrun.process_cleanup",
                context={"remaining_process_group_id": 5100},
            )

    monkeypatch.setattr(postrun, "_terminate_process_group", cleanup)
    config = tmp_path / "config.yaml"
    config.write_text("schema_version: 1\n", encoding="utf-8")

    with pytest.raises(postrun.Wave7PostrunError) as caught:
        postrun.launch_inference(
            config_path=config,
            selection={
                "physical_index": 0,
                "gpu_uuid": "GPU-0",
                "preexisting_compute_inventory": [],
            },
            timeout_seconds=0.1,
        )

    assert caught.value.code == expected_code
    evidence = caught.value.context
    assert evidence["return_code"] == (
        17 if outcome == "nonzero" else 0 if outcome == "cleanup" else None
    )
    assert evidence["timed_out"] is (outcome == "timeout")
    assert evidence["stdout"]["truncated"] is True
    assert evidence["stderr"]["truncated"] is True
    assert evidence["stdout"]["total_bytes"] == len(stdout)
    assert evidence["stderr"]["total_bytes"] == len(stderr)
    assert len(evidence["stdout"]["tail"].encode("utf-8")) <= 65_536
    assert len(evidence["stderr"]["tail"].encode("utf-8")) <= 65_536
    if outcome == "cleanup":
        assert evidence["cleanup_errors"] == [
            {
                "code": "wave7_postrun.process_cleanup",
                "message": "surviving process group",
                "context": {"remaining_process_group_id": 5100},
            }
        ]
    else:
        assert evidence["cleanup_errors"] == []
    call = calls[0]
    assert call["start_new_session"] is True
    assert call["stdin"] is subprocess.DEVNULL
    assert call["stdout"] is subprocess.PIPE
    assert call["stderr"] is subprocess.PIPE
    assert call["shell"] is False


def test_launch_cleanup_failure_takes_precedence_without_losing_child_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Process:
        pid = 5200
        returncode: int | None = None
        stdout = io.BytesIO(b"stdout")
        stderr = io.BytesIO(b"stderr")

        def wait(self, timeout: float) -> int:
            self.returncode = 17
            return self.returncode

        def poll(self) -> int | None:
            return self.returncode

    monkeypatch.setattr(postrun.subprocess, "Popen", lambda *_a, **_k: Process())
    monkeypatch.setattr(
        postrun, "_wait_for_gpu_observation", lambda **_: [{"driver_pid": 777}]
    )
    monkeypatch.setattr(postrun, "_assert_gpu_cleanup", lambda **_: [])
    monkeypatch.setattr(
        postrun,
        "_terminate_process_group",
        lambda *_a, **_k: (_ for _ in ()).throw(
            postrun.Wave7PostrunError(
                "surviving process group",
                code="wave7_postrun.process_cleanup",
                context={"remaining_process_group_id": 5200},
            )
        ),
    )
    config = tmp_path / "config.yaml"
    config.write_text("schema_version: 1\n", encoding="utf-8")

    with pytest.raises(postrun.Wave7PostrunError) as caught:
        postrun.launch_inference(
            config_path=config,
            selection={
                "physical_index": 0,
                "gpu_uuid": "GPU-0",
                "preexisting_compute_inventory": [],
            },
            timeout_seconds=1.0,
        )

    assert caught.value.code == "wave7_postrun.process_cleanup"
    assert caught.value.context["primary_failure"] == {
        "code": "wave7_postrun.child_nonzero",
        "message": "fresh inference child exited nonzero",
        "context": {"return_code": 17},
    }
    assert caught.value.context["cleanup_errors"] == [
        {
            "code": "wave7_postrun.process_cleanup",
            "message": "surviving process group",
            "context": {"remaining_process_group_id": 5200},
        }
    ]


def test_descriptive_timing_is_not_part_of_semantic_projection() -> None:
    left = {"status": "passed", "duration_seconds": 1.0, "checked_at": "a"}
    right = {"status": "passed", "duration_seconds": 99.0, "checked_at": "b"}
    assert postrun.semantic_projection(left) == postrun.semantic_projection(right)


def test_signed_publication_is_absent_only_and_reloads_exactly(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    payload = {"schema": postrun.RECEIPT_SCHEMA, "status": "passed", "value": 1}
    signed = postrun._publish_signed(output, payload)

    assert json.loads(output.read_text(encoding="utf-8")) == signed
    with pytest.raises(Exception):
        postrun._publish_signed(output, payload)


def test_final_publication_failure_writes_only_signed_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = tmp_path / "receipt.json"
    sidecar = tmp_path / "receipt.publication-failure.json"
    marker = tmp_path / "marker.json"
    marker.write_text("{}\n", encoding="utf-8")
    real_publish = postrun._publish_signed
    calls = 0

    def publish(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("simulated receipt publication failure")
        return real_publish(path, payload)

    monkeypatch.setattr(postrun, "_publish_signed", publish)
    with pytest.raises(postrun.Wave7PostrunError) as caught:
        postrun._publish_final_or_sidecar(
            receipt_path=receipt,
            sidecar_path=sidecar,
            marker_path=marker,
            payload={"schema": postrun.RECEIPT_SCHEMA, "status": "passed"},
        )
    assert caught.value.code == "wave7_postrun.publication_failure"
    assert not receipt.exists()
    published = json.loads(sidecar.read_text(encoding="utf-8"))
    assert published["schema"] == postrun.PUBLICATION_FAILURE_SCHEMA
    assert published["status"] == "failed"


def _write_output_fixture(
    root: Path,
    *,
    config: dict[str, Any],
    adapter: dict[str, Any],
    delta: dict[str, Any],
) -> None:
    (root / "configs").mkdir(parents=True)
    fixture = json.loads(postrun.FIXTURE.read_text(encoding="utf-8"))
    rows = [
        {
            "row_id": postrun.EXPECTED_ROW_ID,
            "row_index": 0,
            "example_id": postrun.EXPECTED_ROW_ID,
            "image_path": str(
                (postrun.FIXTURE.parent / fixture["image"]["path"]).resolve()
            ),
            "image_width": 1248,
            "image_height": 832,
            "gt": fixture["objects"],
            "pred": [],
        }
    ]
    for name in (postrun.RAW_NAME, postrun.SCORED_NAME):
        (root / name).write_bytes(_canonical(rows[0]) + b"\n")
    for name in (postrun.TOKEN_TRACE_NAME, postrun.PARSE_DIAGNOSTICS_NAME):
        (root / name).write_bytes(b"")
    (root / postrun.IMAGE_PLAN_NAME).write_bytes(
        _canonical({"row_id": postrun.EXPECTED_ROW_ID, "row_index": 0}) + b"\n"
    )
    (root / postrun.PROVENANCE_NAME).write_bytes(
        _canonical(
            {
                "raw_artifact": {
                    "sha256": hashlib.sha256(
                        (root / postrun.RAW_NAME).read_bytes()
                    ).hexdigest()
                },
                "scored_artifact": {
                    "sha256": hashlib.sha256(
                        (root / postrun.SCORED_NAME).read_bytes()
                    ).hexdigest()
                },
            }
        )
        + b"\n"
    )
    (root / postrun.SUMMARY_NAME).write_bytes(
        _canonical(
            {
                "row_count": 1,
                "raw_row_count": 1,
                "scored_row_count": 1,
                "scored_artifact_materialized": True,
            }
        )
        + b"\n"
    )
    (root / postrun.MANIFEST_NAME).write_bytes(
        _canonical(
            {
                "backend": "hf",
                "backend_mode": "generate",
                "model_identity": {"family": "base-plus-adapter-plus-delta"},
                "adapter_identity": {"loaded": adapter},
                "embedding_delta_identity": {"loaded": delta},
            }
        )
        + b"\n"
    )
    (root / "configs/resolved.json").write_bytes(_canonical({"config": config}) + b"\n")
    (root / "configs/resolved.yaml").write_text("config: {}\n", encoding="utf-8")


@pytest.mark.parametrize(
    ("mutation", "code"),
    (
        ("inventory", "wave7_postrun.output_inventory"),
        ("count", "wave7_postrun.output_rows"),
        ("row_id", "wave7_postrun.output_rows"),
        ("backend", "wave7_postrun.backend_identity"),
        ("adapter", "wave7_postrun.runtime_payload_identity"),
        ("config", "wave7_postrun.config_semantics"),
    ),
)
def test_output_attestation_rejects_contract_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    code: str,
) -> None:
    output = tmp_path / "output"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config = {"schema_version": 1, "semantic": "expected"}
    adapter = {"root": str(checkpoint / "adapter"), "fingerprint": "adapter"}
    delta = {
        "root": str(checkpoint / "special_token_embeddings"),
        "fingerprint": "delta",
    }
    _write_output_fixture(output, config=config, adapter=adapter, delta=delta)
    checkpoint_manifest = {
        "adapter": {
            "inspector_identity": {
                key: value for key, value in adapter.items() if key != "root"
            }
        },
        "special_token_embedding_delta": {
            "inspector_identity": {
                key: value for key, value in delta.items() if key != "root"
            }
        },
    }
    monkeypatch.setattr(postrun, "validate_scored_artifact_set", lambda _: None)
    monkeypatch.setattr(
        postrun, "inspect_dora_adapter_payload", lambda *_args, **_kwargs: adapter
    )
    monkeypatch.setattr(
        postrun,
        "inspect_special_token_embedding_delta_payload",
        lambda *_args, **_kwargs: delta,
    )
    monkeypatch.setattr(
        postrun,
        "load_inference_checkpoint_payload_manifest",
        lambda _: checkpoint_manifest,
    )
    if mutation == "inventory":
        (output / "unexpected.txt").write_text("x", encoding="utf-8")
    elif mutation == "count":
        summary = json.loads((output / postrun.SUMMARY_NAME).read_text())
        summary["row_count"] = 2
        (output / postrun.SUMMARY_NAME).write_bytes(_canonical(summary) + b"\n")
    elif mutation == "row_id":
        (output / postrun.RAW_NAME).write_bytes(
            _canonical({"row_id": "wrong", "pred": []}) + b"\n"
        )
    elif mutation == "backend":
        manifest = json.loads((output / postrun.MANIFEST_NAME).read_text())
        manifest["backend_mode"] = "offline_generate"
        (output / postrun.MANIFEST_NAME).write_bytes(_canonical(manifest) + b"\n")
    elif mutation == "adapter":
        manifest = json.loads((output / postrun.MANIFEST_NAME).read_text())
        manifest["adapter_identity"] = {"fingerprint": "mutated"}
        (output / postrun.MANIFEST_NAME).write_bytes(_canonical(manifest) + b"\n")
    else:
        config = {"schema_version": 1, "semantic": "different"}

    with pytest.raises(postrun.Wave7PostrunError) as caught:
        postrun.validate_outputs(
            output_dir=output,
            config_value=config,
            checkpoint_dir=checkpoint,
        )
    assert caught.value.code == code


@pytest.mark.parametrize("artifact", (postrun.RAW_NAME, postrun.SCORED_NAME))
@pytest.mark.parametrize(
    ("field", "mutated"),
    (
        ("example_id", "wrong-example"),
        ("row_index", 1),
        ("image_path", "/wrong/image.jpg"),
        ("image_width", 1247),
        ("image_height", 831),
        ("gt", []),
    ),
)
def test_output_attestation_rejects_each_fixture_identity_projection_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
    field: str,
    mutated: Any,
) -> None:
    output = tmp_path / "output"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config = {"schema_version": 1, "semantic": "expected"}
    adapter = {"root": str(checkpoint / "adapter"), "fingerprint": "adapter"}
    delta = {
        "root": str(checkpoint / "special_token_embeddings"),
        "fingerprint": "delta",
    }
    _write_output_fixture(output, config=config, adapter=adapter, delta=delta)
    row = json.loads((output / artifact).read_text(encoding="utf-8"))
    row[field] = mutated
    (output / artifact).write_bytes(_canonical(row) + b"\n")
    monkeypatch.setattr(postrun, "validate_scored_artifact_set", lambda _: None)

    with pytest.raises(postrun.Wave7PostrunError) as caught:
        postrun.validate_outputs(
            output_dir=output,
            config_value=config,
            checkpoint_dir=checkpoint,
        )
    assert caught.value.code == "wave7_postrun.output_rows"


@pytest.mark.parametrize(
    "mutation",
    ("metadata", "tensor_path", "tensor_shape", "source_dtype"),
)
def test_embedding_delta_runtime_identity_rejects_direct_mutation(
    tmp_path: Path,
    mutation: str,
) -> None:
    root = (tmp_path / "special_token_embeddings").resolve()
    semantic = {
        "semantics": "selected_token_additive_delta",
        "tensor_key": "embed_delta",
        "tensor_shape": [1004, 2048],
        "tensor_dtype": "bfloat16",
        "token_strings": ["<|coord_0|>"],
        "token_ids": [100],
        "base_model_path": str(postrun.BASE_MODEL),
        "base_config_sha256": "a" * 64,
        "tokenizer_sha256": "b" * 64,
        "tie_word_embeddings": True,
    }
    delta = {
        "kind": "special_token_embedding_delta",
        "version": "v1",
        "root": str(root),
        "file_count": 2,
        "files": [],
        "semantic_identity": semantic,
        "fingerprint": "c" * 64,
    }
    runtime = {
        "status": "loaded",
        "identity": {
            "status": "validated",
            "delta_path": str(root),
            "metadata_path": str(root / "special_token_embeddings.json"),
            "metadata": copy.deepcopy(semantic),
            "base_model_path": str(postrun.BASE_MODEL),
        },
        "load": {
            "loaded": True,
            "tensor_path": str(root / "special_token_embeddings.safetensors"),
            "metadata_path": str(root / "special_token_embeddings.json"),
            "tensor_shape": [1004, 2048],
            "source_tensor_dtype": "bfloat16",
        },
    }
    assert postrun._runtime_delta_matches_inspector(runtime, delta)
    if mutation == "metadata":
        runtime["identity"]["metadata"]["tokenizer_sha256"] = "d" * 64
    elif mutation == "tensor_path":
        runtime["load"]["tensor_path"] = str(root / "other.safetensors")
    elif mutation == "tensor_shape":
        runtime["load"]["tensor_shape"] = [1003, 2048]
    else:
        runtime["load"]["source_tensor_dtype"] = "float32"

    assert not postrun._runtime_delta_matches_inspector(runtime, delta)


@pytest.mark.parametrize("mutation_phase", ("none", "before", "after"))
def test_run_readmits_payload_immediately_before_and_after_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation_phase: str,
) -> None:
    r5 = (tmp_path / "r5").resolve()
    checkpoint = r5 / "runs/resume_child/checkpoints/step-5"
    checkpoint.mkdir(parents=True)
    sequence_path = r5 / "sequence.json"
    final_path = r5 / "final.json"
    sequence_path.write_text("{}\n", encoding="utf-8")
    final_path.write_text("{}\n", encoding="utf-8")
    identity = {"aggregate_digest": "a" * 64}
    receipt_stub = {
        "schema": "stub",
        "status": "passed",
        "receipt_payload_sha256": "b" * 64,
    }
    monkeypatch.setattr(
        postrun,
        "authenticate_inputs",
        lambda **_: {
            "r5_root": r5,
            "sequence_path": sequence_path,
            "sequence": receipt_stub,
            "sequence_file_sha256": hashlib.sha256(
                sequence_path.read_bytes()
            ).hexdigest(),
            "final_path": final_path,
            "final": receipt_stub,
            "final_file_sha256": hashlib.sha256(final_path.read_bytes()).hexdigest(),
            "checkpoint_dir": checkpoint,
            "inference_payload_identity": identity,
        },
    )

    def write_config(path: Path, value: dict[str, Any]) -> dict[str, Any]:
        path.write_text("schema_version: 1\n", encoding="utf-8")
        return {"path": str(path), "file_sha256": "c" * 64, "config": value}

    monkeypatch.setattr(postrun, "_write_config", write_config)
    monkeypatch.setattr(
        postrun,
        "_stable_gpu_selection",
        lambda: (
            {
                "physical_index": 0,
                "gpu_uuid": "GPU-0",
                "preexisting_compute_inventory": [],
            },
            [],
        ),
    )
    mutated = False

    real_publish = postrun._publish_signed

    def publish(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal mutated
        result = real_publish(path, payload)
        if path.name == "inference-attempt-marker.json" and mutation_phase == "before":
            mutated = True
        return result

    monkeypatch.setattr(postrun, "_publish_signed", publish)

    def launch(**_: Any) -> dict[str, Any]:
        nonlocal mutated
        mutated = mutation_phase == "after"
        return {"return_code": 0, "attempt_count": 1}

    monkeypatch.setattr(postrun, "launch_inference", launch)
    monkeypatch.setattr(postrun, "validate_outputs", lambda **_: {"status": "passed"})
    admissions: list[bool] = []

    def admit(*_: Any, **__: Any) -> dict[str, Any]:
        assert (r5 / "postrun/inference-attempt-marker.json").is_file()
        admissions.append(mutated)
        if mutated:
            raise RuntimeError("payload mutated")
        return identity

    monkeypatch.setattr(postrun, "admit_inference_checkpoint_payload_identity", admit)
    output = r5 / "postrun/postrun-receipt.json"
    result = postrun.run(
        r5_root=r5,
        sequence_receipt=sequence_path,
        final_comparison=final_path,
        output=output,
    )
    expected_admissions = {
        "none": [False, False],
        "before": [True],
        "after": [False, True],
    }
    assert admissions == expected_admissions[mutation_phase]
    assert result == (0 if mutation_phase == "none" else 2)
    assert output.exists() is (mutation_phase == "none")


@pytest.mark.parametrize("mutation_phase", ("before", "after"))
def test_run_rejects_authenticated_receipt_drift_around_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation_phase: str,
) -> None:
    r5 = (tmp_path / "r5").resolve()
    checkpoint = r5 / "runs/resume_child/checkpoints/step-5"
    checkpoint.mkdir(parents=True)
    sequence_path = r5 / "sequence.json"
    final_path = r5 / "final.json"
    sequence_path.write_text("sequence-v1\n", encoding="utf-8")
    final_path.write_text("final-v1\n", encoding="utf-8")
    identity = {"aggregate_digest": "a" * 64}
    receipt_stub = {
        "schema": "stub",
        "status": "passed",
        "receipt_payload_sha256": "b" * 64,
    }
    monkeypatch.setattr(
        postrun,
        "authenticate_inputs",
        lambda **_: {
            "r5_root": r5,
            "sequence_path": sequence_path,
            "sequence": receipt_stub,
            "sequence_file_sha256": hashlib.sha256(
                sequence_path.read_bytes()
            ).hexdigest(),
            "final_path": final_path,
            "final": receipt_stub,
            "final_file_sha256": hashlib.sha256(final_path.read_bytes()).hexdigest(),
            "checkpoint_dir": checkpoint,
            "inference_payload_identity": identity,
        },
    )

    def write_config(path: Path, value: dict[str, Any]) -> dict[str, Any]:
        path.write_text("schema_version: 1\n", encoding="utf-8")
        return {"path": str(path), "file_sha256": "c" * 64, "config": value}

    monkeypatch.setattr(postrun, "_write_config", write_config)
    monkeypatch.setattr(
        postrun,
        "_stable_gpu_selection",
        lambda: (
            {
                "physical_index": 0,
                "gpu_uuid": "GPU-0",
                "preexisting_compute_inventory": [],
            },
            [],
        ),
    )
    real_publish = postrun._publish_signed

    def publish(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
        result = real_publish(path, payload)
        if path.name == "inference-attempt-marker.json" and mutation_phase == "before":
            sequence_path.write_text("sequence-v2\n", encoding="utf-8")
        return result

    monkeypatch.setattr(postrun, "_publish_signed", publish)
    launch_calls = 0

    def launch(**_: Any) -> dict[str, Any]:
        nonlocal launch_calls
        launch_calls += 1
        if mutation_phase == "after":
            final_path.write_text("final-v2\n", encoding="utf-8")
        return {"return_code": 0, "attempt_count": 1}

    monkeypatch.setattr(postrun, "launch_inference", launch)
    monkeypatch.setattr(
        postrun,
        "admit_inference_checkpoint_payload_identity",
        lambda *_args, **_kwargs: identity,
    )
    monkeypatch.setattr(
        postrun,
        "validate_outputs",
        lambda **_: pytest.fail("drift must fail before output attestation"),
    )
    output = r5 / "postrun/postrun-receipt.json"

    assert (
        postrun.run(
            r5_root=r5,
            sequence_receipt=sequence_path,
            final_comparison=final_path,
            output=output,
        )
        == 2
    )
    assert launch_calls == (0 if mutation_phase == "before" else 1)
    assert not output.exists()
    sidecar = json.loads(
        (r5 / "postrun/postrun-receipt.publication-failure.json").read_text(
            encoding="utf-8"
        )
    )
    assert sidecar["failure"]["code"] == "wave7_postrun.identity_drift"


def test_run_publishes_signed_failure_sidecar_with_launch_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r5 = (tmp_path / "r5").resolve()
    checkpoint = r5 / "runs/resume_child/checkpoints/step-5"
    checkpoint.mkdir(parents=True)
    sequence_path = r5 / "sequence.json"
    final_path = r5 / "final.json"
    sequence_path.write_text("{}\n", encoding="utf-8")
    final_path.write_text("{}\n", encoding="utf-8")
    identity = {"aggregate_digest": "a" * 64}
    receipt_stub = {
        "schema": "stub",
        "status": "passed",
        "receipt_payload_sha256": "b" * 64,
    }
    monkeypatch.setattr(
        postrun,
        "authenticate_inputs",
        lambda **_: {
            "r5_root": r5,
            "sequence_path": sequence_path,
            "sequence": receipt_stub,
            "sequence_file_sha256": hashlib.sha256(
                sequence_path.read_bytes()
            ).hexdigest(),
            "final_path": final_path,
            "final": receipt_stub,
            "final_file_sha256": hashlib.sha256(final_path.read_bytes()).hexdigest(),
            "checkpoint_dir": checkpoint,
            "inference_payload_identity": identity,
        },
    )

    def write_config(path: Path, value: dict[str, Any]) -> dict[str, Any]:
        path.write_text("schema_version: 1\n", encoding="utf-8")
        return {"path": str(path), "file_sha256": "c" * 64, "config": value}

    monkeypatch.setattr(postrun, "_write_config", write_config)
    monkeypatch.setattr(
        postrun,
        "_stable_gpu_selection",
        lambda: (
            {
                "physical_index": 0,
                "gpu_uuid": "GPU-0",
                "preexisting_compute_inventory": [],
            },
            [],
        ),
    )
    monkeypatch.setattr(
        postrun,
        "admit_inference_checkpoint_payload_identity",
        lambda *_args, **_kwargs: identity,
    )
    diagnostics = {
        "return_code": 17,
        "post_cleanup_return_code": 17,
        "timed_out": False,
        "timeout_seconds": 0.1,
        "stdout": {
            "tail": "stdout-tail",
            "total_bytes": 11,
            "truncated": False,
            "tail_cap_bytes": 65_536,
        },
        "stderr": {
            "tail": "stderr-tail",
            "total_bytes": 11,
            "truncated": False,
            "tail_cap_bytes": 65_536,
        },
        "cleanup_errors": [],
    }
    monkeypatch.setattr(
        postrun,
        "launch_inference",
        lambda **_: (_ for _ in ()).throw(
            postrun.Wave7PostrunError(
                "fresh inference child exited nonzero",
                code="wave7_postrun.child_nonzero",
                context=diagnostics,
            )
        ),
    )
    output = r5 / "postrun/postrun-receipt.json"

    assert (
        postrun.run(
            r5_root=r5,
            sequence_receipt=sequence_path,
            final_comparison=final_path,
            output=output,
            timeout_seconds=0.1,
        )
        == 2
    )
    sidecar_path = r5 / "postrun/postrun-receipt.publication-failure.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    unsigned = dict(sidecar)
    observed_digest = unsigned.pop("receipt_payload_sha256")
    assert sidecar["schema"] == "coordexp-swift-wave7-exact-resume-postrun-failure-v1"
    assert sidecar["status"] == "failed"
    assert sidecar["failure"] == {
        "code": "wave7_postrun.child_nonzero",
        "message": "fresh inference child exited nonzero",
        "context": diagnostics,
    }
    assert observed_digest == hashlib.sha256(_canonical(unsigned)).hexdigest()
    assert not output.exists()
