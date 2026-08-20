from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import ModuleType
from typing import Any

import pytest
import yaml

from src.config.loader import load_train_config


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/probes/coordexp_swift/wave7_exact_resume_sequence.py"
TRAIN_SCRIPT = REPO_ROOT / "src/train.py"
INTERRUPT_SCRIPT = (
    REPO_ROOT / "scripts/probes/coordexp_swift/wave7_exact_resume_interrupt.py"
)
COMPARE_V2_SCRIPT = (
    REPO_ROOT / "scripts/probes/coordexp_swift/wave7_exact_resume_compare_v2.py"
)
REQUEST_PRODUCER_SCRIPT = (
    REPO_ROOT / "scripts/probes/coordexp_swift/wave7_exact_resume_request.py"
)
INPUT_ATTESTATION_SOURCE = REPO_ROOT / "src/training/input_attestation.py"
PYTHON_EXECUTABLE = Path(sys.executable).resolve()
ACCELERATE_EXECUTABLE = Path(shutil.which("accelerate") or "").resolve()
BASE_TRAIN_CONFIG = (
    REPO_ROOT / "configs/coordexp_swift/smoke/"
    "qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_llm_12000_"
    "accelerate8_ebs24_2step_warmup0p1_eval_patchproof.yaml"
)
IMMUTABLE_R5_SEQUENCE_RECEIPT = (
    REPO_ROOT / "outputs/probes/coordexp_swift/wave7_exact_resume/2026-08-11-r5/"
    "sequence-receipt.json"
)
IMMUTABLE_R6_PREFLIGHT_ROOT = (
    REPO_ROOT / "outputs/probes/coordexp_swift/wave7_exact_resume/2026-08-11-r6/"
    "determinism-preflight"
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_signed_json(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    signed = dict(payload)
    signed["receipt_payload_sha256"] = _sha256(_canonical(payload))
    path.write_bytes(_canonical(signed) + b"\n")
    return signed


def _write_digest_json(
    path: Path, payload: dict[str, Any], *, digest_field: str
) -> dict[str, Any]:
    signed = dict(payload)
    signed[digest_field] = _sha256(_canonical(payload))
    path.write_bytes(_canonical(signed) + b"\n")
    return signed


def _binding(path: Path, receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "file_sha256": _sha256(path.read_bytes()),
        "payload_sha256": receipt["receipt_payload_sha256"],
        "schema": receipt["schema"],
        "status": receipt["status"],
    }


def _payload_binding(
    path: Path, payload: dict[str, Any], *, digest_field: str
) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "file_sha256": _sha256(path.read_bytes()),
        "payload_sha256": payload[digest_field],
        "schema": payload["schema"],
        "status": payload["status"],
    }


def _source_binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path.read_bytes()),
    }


def _fake_process_record(**overrides: Any) -> dict[str, Any]:
    record = {
        "pid": 12345,
        "pgid": 12345,
        "term_sent": False,
        "kill_sent": False,
        "reaped": True,
        "captured_process_graph": [],
        "remaining_pids": [],
        "remaining_pgids": [],
        "signal_events": [],
        "graph_overflow": False,
        "observed_process_count": 0,
        "stdout_tail": "",
        "stdout_truncated": False,
        "stderr_tail": "",
        "stderr_truncated": False,
    }
    record.update(overrides)
    return record


@pytest.fixture
def controller() -> ModuleType:
    spec = importlib.util.spec_from_file_location("wave7_exact_resume_sequence", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def request_producer() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "wave7_exact_resume_request_for_sequence_test", REQUEST_PRODUCER_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_current_namespace_is_authorized_core_4_with_exact_derived_paths(controller):
    sequence_root = (
        REPO_ROOT / "outputs/probes/coordexp_swift/wave7_exact_resume/2026-08-12-core-4"
    ).resolve()
    cache_root = (
        REPO_ROOT
        / "outputs/probes/coordexp_swift/private_v3_cache/2026-08-12-wave7-core-4"
    ).resolve()

    assert controller.R7_SEQUENCE_ROOT == sequence_root
    assert controller.R7_PRIVATE_CACHE_ROOT == cache_root
    assert controller.R7_REQUEST_PATH == sequence_root / "request-v6.json"
    assert controller.R7_PLAN_PATH == sequence_root / "sequence-plan-v6.json"
    assert controller._expected_r7_run_roots() == {
        role: sequence_root / "runs" / role for role in controller.RUN_ROLES
    }
    assert controller._expected_r7_targets() == {
        "sequence_marker": sequence_root / "sequence-marker.json",
        "sequence_receipt": sequence_root / "sequence-receipt.json",
        "publication_failure_sidecar": (
            sequence_root / "sequence-receipt.publication-failure.json"
        ),
        "interruption_marker": (
            sequence_root / "interrupted-parent-attempt-marker.json"
        ),
        "interruption_receipt": (
            sequence_root / "interrupted-parent-termination-receipt.json"
        ),
        "pre_child_receipt": sequence_root / "pre-child-receipt.json",
        "final_receipt": sequence_root / "exact-resume-comparison-receipt-v2.json",
    }


def test_command_contract_hashes_all_resolved_config_sources_in_exact_order(
    controller: ModuleType, tmp_path: Path
) -> None:
    overlay = tmp_path / "interrupted-parent.yaml"
    overlay.write_text(
        "\n".join(
            (
                "schema_version: 1",
                f"extends: {BASE_TRAIN_CONFIG.resolve()}",
                "run:",
                "  name: interrupted_parent",
                f"  artifact_root: {tmp_path / 'runs'}",
                "  collision_policy: fail",
                "",
            )
        ),
        encoding="utf-8",
    )
    config_paths = {role: overlay.resolve() for role in controller.RUN_ROLES}
    roots = {
        role: (tmp_path / "runs" / role).resolve() for role in controller.RUN_ROLES
    }
    targets = {
        name: (tmp_path / f"{name}.json").resolve()
        for name in controller.TARGET_NAMES
    }

    commands = controller._expected_commands(
        config_paths=config_paths,
        roots=roots,
        targets=targets,
        policy={
            "phase_timeout_seconds": {"interrupted_parent": 600},
            "term_grace_seconds": 30,
            "kill_grace_seconds": 30,
            "gpu_baseline_stability_seconds": 2,
        },
        provenance_sha256="f" * 64,
    )

    outer = commands["interrupted_parent"]
    launcher_boundary = outer.index("--")
    outer_config_values = [
        outer[index + 1]
        for index, value in enumerate(outer[:launcher_boundary])
        if value == "--config"
    ]
    resolved = load_train_config(overlay)
    assert outer_config_values == [str(source.path) for source in resolved.sources]


def _write_fake_phase_runner(tmp_path: Path) -> Path:
    path = tmp_path / "fake_phase.py"
    path.write_text(
        """
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import time

def canonical(value):
    return json.dumps(
        value, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

def signed(path, payload):
    value = dict(payload)
    value["receipt_payload_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    path.write_bytes(canonical(value) + b"\\n")

phase, log_text, marker_text, *args = sys.argv[1:]
log = Path(log_text)
marker = Path(marker_text)
assert marker.exists(), "sequence marker must predate every phase"
with log.open("a", encoding="utf-8") as handle:
    handle.write(phase + "\\n")

if phase == "sleep":
    time.sleep(30)
if phase in {"uninterrupted", "interrupted_parent", "resume_child"}:
    Path(args[0]).mkdir(parents=True)
    if phase == "interrupted_parent":
        signed(
            Path(args[1]),
            {"schema": "wave7-interrupt-marker-test-v1", "status": "passed"},
        )
        signed(
            Path(args[2]),
            {"schema": "wave7-interrupt-receipt-test-v1", "status": "passed"},
        )
elif phase == "pre_child":
    signed(
        Path(args[0]),
        {
            "schema": "coordexp-swift-wave7-pre-child-gate-v1",
            "status": "passed",
            "child_launch_authorized": True,
            "mismatches": [],
        },
    )
elif phase == "verify_pre_child":
    receipt = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    if receipt["receipt_payload_sha256"] != args[1]:
        raise SystemExit(19)
elif phase == "final_compare":
    signed(
        Path(args[0]),
        {
            "schema": "coordexp-swift-wave7-exact-resume-comparison-v2",
            "status": "passed",
            "mismatches": [],
        },
    )
elif phase == "fail":
    raise SystemExit(23)
else:
    raise AssertionError(phase)
""".lstrip(),
        encoding="utf-8",
    )
    return path


def _request_fixture(
    controller: ModuleType, tmp_path: Path
) -> tuple[Path, Path, dict[str, Any]]:
    controller.R7_SEQUENCE_ROOT = tmp_path.resolve()
    controller.R7_PRIVATE_CACHE_ROOT = (tmp_path / "cache").resolve()
    controller.R7_REQUEST_PATH = (tmp_path / "request-v6.json").resolve()
    controller.R7_PLAN_PATH = (tmp_path / "sequence-plan-v6.json").resolve()
    (tmp_path / "runs").mkdir()
    fake = _write_fake_phase_runner(tmp_path)
    log = tmp_path / "phases.log"
    marker = tmp_path / "sequence-marker.json"
    terminal = tmp_path / "sequence-receipt.json"
    sidecar = tmp_path / "sequence-receipt.publication-failure.json"
    run_roots = {
        role: str((tmp_path / "runs" / role).resolve()) for role in controller.RUN_ROLES
    }
    targets = {
        "sequence_marker": str(marker.resolve()),
        "sequence_receipt": str(terminal.resolve()),
        "publication_failure_sidecar": str(sidecar.resolve()),
        "interruption_marker": str(
            (tmp_path / "interrupted-parent-attempt-marker.json").resolve()
        ),
        "interruption_receipt": str(
            (tmp_path / "interrupted-parent-termination-receipt.json").resolve()
        ),
        "pre_child_receipt": str((tmp_path / "pre-child-receipt.json").resolve()),
        "final_receipt": str(
            (tmp_path / "exact-resume-comparison-receipt-v2.json").resolve()
        ),
    }

    amendment_path = tmp_path / "amendment.json"
    authority_path = controller.AMENDMENT_AUTHORITY_PATH
    authority_text = authority_path.read_text(encoding="utf-8")
    controller.FROZEN_AMENDMENT_AUTHORITY_FILE_SHA256 = _sha256(
        authority_path.read_bytes()
    )
    section_start = authority_text.index(controller.AMENDMENT_SECTION_ANCHOR)
    next_heading = authority_text.find(
        "\n### ", section_start + len(controller.AMENDMENT_SECTION_ANCHOR)
    )
    authority_section = (
        authority_text[section_start:]
        if next_heading < 0
        else authority_text[section_start:next_heading]
    )
    amendment = _write_digest_json(
        amendment_path,
        {
            "schema": "coordexp-swift-wave7-r7-amendment-v4",
            "status": "approved",
            "effective_date": "2026-08-12",
            "section_anchor": controller.AMENDMENT_SECTION_ANCHOR,
            "scope": copy.deepcopy(controller.AMENDMENT_SCOPE),
            "authority": {
                "path": str(authority_path),
                "file_sha256": controller.FROZEN_AMENDMENT_AUTHORITY_FILE_SHA256,
                "section_sha256": _sha256(authority_section.encode("utf-8")),
            },
        },
        digest_field="amendment_sha256",
    )
    r4_path = tmp_path / "r4-failure.json"
    r4 = _write_signed_json(
        r4_path,
        {
            "schema": "coordexp-swift-wave7-exact-resume-comparison-v1",
            "status": "failed",
            "mismatches": [{"code": "legacy-r4"}],
        },
    )
    r4_binding = _binding(r4_path, r4)
    controller.FROZEN_R4_FAILURE_FILE_SHA256 = r4_binding["file_sha256"]
    controller.FROZEN_R4_FAILURE_PAYLOAD_SHA256 = r4_binding["payload_sha256"]
    predecessor_path = IMMUTABLE_R5_SEQUENCE_RECEIPT
    predecessor_receipt = json.loads(predecessor_path.read_text(encoding="utf-8"))
    predecessor_binding = _binding(predecessor_path, predecessor_receipt)
    r6_plan_path = IMMUTABLE_R6_PREFLIGHT_ROOT / "plan.json"
    r6_marker_path = IMMUTABLE_R6_PREFLIGHT_ROOT / "attempt-start-marker.json"
    r6_terminal_path = IMMUTABLE_R6_PREFLIGHT_ROOT / "terminal-receipt.json"
    r6_plan = json.loads(r6_plan_path.read_text(encoding="utf-8"))
    r6_marker = json.loads(r6_marker_path.read_text(encoding="utf-8"))
    r6_terminal = json.loads(r6_terminal_path.read_text(encoding="utf-8"))
    predecessor_preflight_failure = {
        "historical_non_executable": True,
        "plan": _payload_binding(
            r6_plan_path, r6_plan, digest_field="plan_payload_sha256"
        ),
        "attempt_marker": _binding(r6_marker_path, r6_marker),
        "terminal_receipt": _binding(r6_terminal_path, r6_terminal),
    }
    determinism_marker_path = tmp_path / "determinism-attempt-marker.json"
    determinism_marker = _write_signed_json(
        determinism_marker_path,
        {
            "schema": "coordexp-swift-wave7-determinism-preflight-attempt-start-marker-v4",
            "status": "started",
        },
    )

    def determinism_process(launch_id: str, pid: int) -> dict[str, Any]:
        member = {
            "pid": pid,
            "state": "Z",
            "parent_pid": 1,
            "process_group_id": pid,
            "session_id": pid,
            "start_time_ticks": pid * 10,
        }
        return {
            "launch_id": launch_id,
            "pid": pid,
            "returncode": 0,
            "termination": "exited",
            "process_scope": {
                "session_id": pid,
                "leader": {"pid": pid, "start_time_ticks": pid * 10},
                "observed_members": [member],
                "term_sent": False,
                "kill_sent": False,
                "remaining_members": [],
                "leader_reaped": True,
                "cleanup_failure": None,
                "cleanup_verified": True,
            },
            "cleanup_verified": True,
            "stdout_tail": "",
            "stderr_tail": "",
        }

    determinism_path = tmp_path / "determinism.json"
    determinism = _write_signed_json(
        determinism_path,
        {
            "schema": "coordexp-swift-wave7-r5-determinism-preflight-v4",
            "status": "passed",
            "world_size": 8,
            "launch_count": 2,
            "mismatches": [],
            "plan_sha256": "d" * 64,
            "attempt_marker": _binding(determinism_marker_path, determinism_marker),
            "processes": [
                determinism_process("launch-a", 9_001),
                determinism_process("launch-b", 9_002),
            ],
            "rank_receipts": [{"rank": rank} for rank in range(16)],
            "comparisons": {
                "launch_digest_equal": True,
                "device_mapping_equal": True,
                "native_mapping_equal": True,
                "rank_count_per_launch": 8,
                "mismatch_ranks": [],
                "launch_a_aggregate_sha256": "e" * 64,
                "launch_b_aggregate_sha256": "e" * 64,
            },
            "gpu_shared_occupancy_sweeps": [
                {
                    "phase": phase,
                    "samples": [
                        {
                            "sample_index": 0,
                            "sample_monotonic_ns": index * 3_000_000_000
                            + 10_000_000_000,
                        },
                        {
                            "sample_index": 1,
                            "sample_monotonic_ns": index * 3_000_000_000
                            + 12_000_000_000,
                        },
                    ],
                    "device_inventory_matches_baseline": True,
                    "observed_compute_processes": [],
                    "new_compute_processes": [],
                    "admitted": True,
                }
                for index, phase in enumerate(
                    ("post-launch-a", "post-launch-b", "preterminal")
                )
            ],
            "cleanup": {
                "all_process_groups_exited": True,
                "bounded": True,
            },
            "claim_scope": dict(controller.DETERMINISM_CLAIM_SCOPE),
        },
    )
    determinism_plan_path = tmp_path / "determinism-plan.json"
    determinism_plan = _write_digest_json(
        determinism_plan_path,
        {
            "schema": controller.DETERMINISM_PREFLIGHT_PLAN_SCHEMA,
            "status": "prepared",
        },
        digest_field="plan_payload_sha256",
    )
    determinism_plan_binding = {
        "path": str(determinism_plan_path.resolve()),
        "file_sha256": _sha256(determinism_plan_path.read_bytes()),
        "payload_sha256": determinism_plan["plan_payload_sha256"],
        "schema": determinism_plan["schema"],
        "status": determinism_plan["status"],
    }

    def validate_preflight_fixture(
        *, plan_binding: dict[str, Any], terminal_binding: dict[str, Any]
    ) -> dict[str, Any]:
        assert plan_binding == determinism_plan_binding
        assert (
            _sha256(Path(plan_binding["path"]).read_bytes())
            == plan_binding["file_sha256"]
        )
        plan_payload = json.loads(Path(plan_binding["path"]).read_text())
        assert plan_payload["plan_payload_sha256"] == plan_binding["payload_sha256"]
        terminal_payload = controller._verify_receipt_binding(
            terminal_binding, owner="8-rank determinism preflight"
        )
        controller._validate_determinism_preflight_payload(terminal_payload)
        return terminal_payload

    controller._validate_determinism_preflight_evidence = validate_preflight_fixture
    provenance = {
        "schema_version": 1,
        "repository": {
            "test_fixture": "wave7-sequence",
            "execution_relevant_digest": {
                "status": "available",
                "value": "8" * 64,
            },
        },
        "dependencies": {},
        "runtime": {},
    }
    pinned_runtime_baseline = {
        "schema_version": 3,
        "baseline_sha256": controller.execution_provenance.PINNED_RUNTIME_BASELINE_SHA256,
        "attention_backend": "flash_attention_2",
        "admitted": True,
        "mismatches": [],
        "reference_only": {},
    }
    controller._collect_live_execution_provenance = lambda: copy.deepcopy(provenance)
    controller._require_pinned_runtime_baseline = lambda _: copy.deepcopy(
        pinned_runtime_baseline
    )
    provenance_source = Path(controller.execution_provenance.__file__).resolve()
    native_reference_path = tmp_path / "native-reference.json"
    native_reference = _write_digest_json(
        native_reference_path,
        {
            "schema": controller.NATIVE_REFERENCE_SCHEMA,
            "created_at": "2026-08-11T00:00:00+00:00",
            "repository_root": str(REPO_ROOT),
            "source_identity": {
                "path": str(provenance_source),
                "sha256": _sha256(provenance_source.read_bytes()),
            },
            "attention_backend": "flash_attention_2",
            "pinned_runtime_baseline": {
                "schema_version": 3,
                "baseline_sha256": controller.execution_provenance.PINNED_RUNTIME_BASELINE_SHA256,
                "admitted": True,
            },
            "cuda": {"status": "available"},
            "mapped_native_execution": {
                "schema_version": 1,
                "cuda_initialized": True,
                "admitted": True,
                "mismatches": [],
            },
            "terminal_status": "passed",
        },
        digest_field="receipt_sha256",
    )
    native_reference_binding = {
        "path": str(native_reference_path.resolve()),
        "file_sha256": _sha256(native_reference_path.read_bytes()),
        "payload_sha256": native_reference["receipt_sha256"],
        "schema": controller.NATIVE_REFERENCE_SCHEMA,
        "status": "passed",
    }
    late_native_expectations: dict[str, dict[str, Any]] = {}
    for component, distribution in (
        ("flash_attention_2", "flash-attn"),
        ("libcublas", "nvidia-cublas-cu12"),
        ("libnccl", "nvidia-nccl-cu12"),
    ):
        library = tmp_path / f"lib-{component}.so"
        library.write_bytes(f"{component}-test-identity".encode("utf-8"))
        late_native_expectations[component] = {
            "distribution": distribution,
            "soname": library.name,
            "path": str(library.resolve()),
            "size_bytes": library.stat().st_size,
            "file_sha256": _sha256(library.read_bytes()),
        }
    runtime_path = tmp_path / "runtime.json"
    runtime = _write_signed_json(
        runtime_path,
        {
            "schema": controller.RUNTIME_ADMISSION_SCHEMA,
            "status": "passed",
            "model_loaded": False,
            "cuda_initialized": False,
            "repository_root": str(REPO_ROOT),
            "source_identity": {
                "path": str(provenance_source),
                "file_sha256": _sha256(provenance_source.read_bytes()),
            },
            "native_reference": native_reference_binding,
            "attention_backend": "flash_attention_2",
            "provenance": provenance,
            "pinned_runtime_baseline": pinned_runtime_baseline,
            "late_native_expectations": late_native_expectations,
        },
    )
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    cache_preparation_receipt = tmp_path / "cache-preparation-receipt.json"
    cache_preparation_receipt.write_text("cache-preparation-v1\n", encoding="utf-8")
    cache_native_files: dict[str, Path] = {}
    for split in ("train", "eval.forward"):
        split_slug = split.replace(".", "-")
        manifest = cache_root / f"{split_slug}-manifest.json"
        chunk = cache_root / f"{split_slug}-chunk.pkl"
        manifest.write_text(f"{split}-manifest-v1\n", encoding="utf-8")
        chunk.write_bytes(f"{split}-chunk-v1".encode("utf-8"))
        cache_native_files[f"{split_slug}_manifest"] = manifest
        cache_native_files[f"{split_slug}_chunk"] = chunk
    model_root = tmp_path / "model"
    model_root.mkdir()
    model_native_files = {
        "index": model_root / "model.safetensors.index.json",
        "shard": model_root / "model-00001-of-00001.safetensors",
        "standalone": model_root / "adapter_model.safetensors",
    }
    for name, path in model_native_files.items():
        path.write_bytes(f"{name}-v1".encode("utf-8"))
    base_config = load_train_config(BASE_TRAIN_CONFIG)
    assert base_config.entry_config_path == BASE_TRAIN_CONFIG.resolve()
    configs: dict[str, str] = {}
    for role in run_roots:
        config = tmp_path / f"{role}.yaml"
        payload = copy.deepcopy(base_config.config_dict)
        payload["run"] = {
            "name": role,
            "artifact_root": str((tmp_path / "runs").resolve()),
            "output_dir": Path(run_roots[role]).name,
            "collision_policy": "fail",
        }
        payload["training"]["max_steps"] = 5
        payload["training"]["forward_input_provider_mode"] = "synchronous"
        payload["runtime"] = {
            "seed": 17,
            "determinism": {"mode": "strict_cuda_replay_v1"},
        }
        payload["eval"]["forward"] = {
            "every_fraction": None,
            "steps": [3],
        }
        payload["checkpoint"] = {
            "every_fraction": None,
            "steps": [3, 5],
            "save_final": True,
        }
        payload["resume"] = {
            "mode": "exact_same_world_size",
            "checkpoint_dir": (
                str(Path(run_roots["interrupted_parent"]) / "checkpoints" / "step-3")
                if role == "resume_child"
                else None
            ),
        }
        config.write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )
        loaded = load_train_config(config).config
        assert loaded.run.name == role
        assert loaded.resume.mode == "exact_same_world_size"
        configs[role] = str(config.resolve())

    def native_file_identity(path: Path) -> dict[str, Any]:
        return {
            "path": str(path.resolve()),
            "file_sha256": _sha256(path.read_bytes()),
        }

    cache_attestation_path = tmp_path / "cache-input-attestation.json"
    cache_body = {
        "schema": controller.CACHE_ATTESTATION_SCHEMA,
        "status": "passed",
        "model_loaded": False,
        "config_identity": {"paths": configs},
        "cache_root": str(cache_root.resolve()),
        "max_cache_payload_bytes": 1_048_576,
        "measured_payload_bytes": sum(
            path.stat().st_size
            for key, path in cache_native_files.items()
            if key.endswith("_chunk")
        ),
        "preparation_duration_seconds": 0.01,
        "preparation_receipt": native_file_identity(cache_preparation_receipt),
        "splits": {
            split: {
                "manifest": native_file_identity(
                    cache_native_files[f"{split.replace('.', '-')}_manifest"]
                ),
                "chunks": [
                    native_file_identity(
                        cache_native_files[f"{split.replace('.', '-')}_chunk"]
                    )
                ],
                "materialization": {"strategy": "fork_process_pool", "workers": 2},
            }
            for split in ("train", "eval.forward")
        },
    }
    cache_attestation = _write_digest_json(
        cache_attestation_path,
        cache_body,
        digest_field="attestation_sha256",
    )
    model_attestation_path = tmp_path / "model-input-attestation.json"
    model_body = {
        "schema": controller.MODEL_ATTESTATION_SCHEMA,
        "status": "passed",
        "model_loaded": False,
        "config_identity": {"paths": configs},
        "model_root": str(model_root.resolve()),
        "weight_hash_execution_policy": {
            "schema": "coordexp-swift-base-model-weight-hash-execution-policy-v1",
            "strategy": "thread_pool_file_sha256",
            "resolved_workers": 2,
            "payload_file_count": 3,
        },
        "base_model_weight_identity": {
            name: native_file_identity(path)
            for name, path in model_native_files.items()
        },
    }
    model_attestation = _write_digest_json(
        model_attestation_path,
        model_body,
        digest_field="attestation_sha256",
    )

    def validate_test_attestations(
        *,
        cache_attestation: dict[str, Any],
        model_attestation: dict[str, Any],
        config_paths: dict[str, str],
        validate_cache_payloads: bool,
        rehash_model_weights: bool,
        max_cache_payload_bytes: int,
    ) -> dict[str, Any]:
        assert validate_cache_payloads is True
        assert rehash_model_weights is True
        assert {role: str(path) for role, path in config_paths.items()} == configs
        assert max_cache_payload_bytes == 1_048_576
        assert cache_attestation["config_identity"] == {"paths": configs}
        assert model_attestation["config_identity"] == {"paths": configs}
        assert model_attestation["weight_hash_execution_policy"] == {
            "schema": "coordexp-swift-base-model-weight-hash-execution-policy-v1",
            "strategy": "thread_pool_file_sha256",
            "resolved_workers": 2,
            "payload_file_count": 3,
        }
        for identity in (cache_attestation["preparation_receipt"],):
            assert (
                _sha256(Path(identity["path"]).read_bytes()) == identity["file_sha256"]
            )
        for split in ("train", "eval.forward"):
            split_row = cache_attestation["splits"][split]
            for identity in (split_row["manifest"], *split_row["chunks"]):
                assert (
                    _sha256(Path(identity["path"]).read_bytes())
                    == identity["file_sha256"]
                )
        for identity in model_attestation["base_model_weight_identity"].values():
            assert (
                _sha256(Path(identity["path"]).read_bytes()) == identity["file_sha256"]
            )
        return {
            "status": "passed",
            "cache_attestation_sha256": cache_attestation["attestation_sha256"],
            "model_attestation_sha256": model_attestation["attestation_sha256"],
            "cache_payloads_validated": True,
            "model_weights_rehashed": True,
            "measured_cache_payload_bytes": cache_attestation["measured_payload_bytes"],
            "materialization_policy": {
                split: {"strategy": "fork_process_pool", "workers": 2}
                for split in ("train", "eval.forward")
            },
            "weight_hash_execution_policy": model_attestation[
                "weight_hash_execution_policy"
            ],
        }

    controller.validate_training_input_attestations = validate_test_attestations

    def train_command(role: str, port: int) -> list[str]:
        return [
            str(ACCELERATE_EXECUTABLE),
            "launch",
            "--multi_gpu",
            "--num_processes",
            "8",
            "--main_process_port",
            str(port),
            "--module",
            "src.train",
            "--config",
            configs[role],
        ]

    uninterrupted_command = train_command("uninterrupted", 29681)
    parent_train_command = train_command("interrupted_parent", 29682)
    child_train_command = train_command("resume_child", 29683)
    interrupted_config_args = [
        item
        for source in load_train_config(Path(configs["interrupted_parent"])).sources
        for item in ("--config", str(source.path))
    ]
    interrupt_sha256 = _sha256(INTERRUPT_SCRIPT.read_bytes())
    commands = {
        "uninterrupted": uninterrupted_command,
        "interrupted_parent": [
            str(PYTHON_EXECUTABLE),
            str(INTERRUPT_SCRIPT.resolve()),
            "--parent-run-dir",
            run_roots["interrupted_parent"],
            "--expected-checkpoint-step",
            "3",
            "--marker",
            targets["interruption_marker"],
            "--receipt",
            targets["interruption_receipt"],
            "--timeout-seconds",
            "5.0",
            "--term-grace-seconds",
            "0.25",
            "--kill-grace-seconds",
            "0.25",
            "--stability-seconds",
            "2.0",
            "--poll-seconds",
            "0.01",
            "--source",
            str(INTERRUPT_SCRIPT.resolve()),
            "--source",
            str(TRAIN_SCRIPT.resolve()),
            "--source",
            str(
                (
                    REPO_ROOT
                    / "scripts/probes/coordexp_swift/wave7_exact_resume_compare.py"
                ).resolve()
            ),
            "--source",
            str(COMPARE_V2_SCRIPT.resolve()),
            *interrupted_config_args,
            "--",
            *parent_train_command,
        ],
        "pre_child": [
            str(PYTHON_EXECUTABLE),
            str(COMPARE_V2_SCRIPT.resolve()),
            "pre-child",
            "--uninterrupted-run-dir",
            run_roots["uninterrupted"],
            "--interrupted-parent-run-dir",
            run_roots["interrupted_parent"],
            "--interruption-marker",
            targets["interruption_marker"],
            "--termination-receipt",
            targets["interruption_receipt"],
            "--output",
            targets["pre_child_receipt"],
            "--expected-interrupt-source-sha256",
            interrupt_sha256,
            "--expected-source-sha256",
            controller.FROZEN_V2_COMPARATOR_SHA256,
            "--expected-provenance-sha256",
            provenance["repository"]["execution_relevant_digest"]["value"],
        ],
        "verify_pre_child": [
            str(PYTHON_EXECUTABLE),
            str(COMPARE_V2_SCRIPT.resolve()),
            "verify-pre-child",
            "--receipt",
            targets["pre_child_receipt"],
            "--expected-payload-sha256",
            controller.PRE_CHILD_DIGEST_PLACEHOLDER,
        ],
        "resume_child": [
            str(PYTHON_EXECUTABLE),
            str(SCRIPT.resolve()),
            "launch-child",
            "--receipt",
            targets["pre_child_receipt"],
            "--expected-payload-sha256",
            controller.PRE_CHILD_DIGEST_PLACEHOLDER,
            "--config",
            configs["resume_child"],
            "--run-root",
            run_roots["resume_child"],
            "--parent-run-root",
            run_roots["interrupted_parent"],
            "--",
            *child_train_command,
        ],
        "final_compare": [
            str(PYTHON_EXECUTABLE),
            str(COMPARE_V2_SCRIPT.resolve()),
            "compare",
            "--uninterrupted-run-dir",
            run_roots["uninterrupted"],
            "--interrupted-parent-run-dir",
            run_roots["interrupted_parent"],
            "--resume-child-run-dir",
            run_roots["resume_child"],
            "--interruption-marker",
            targets["interruption_marker"],
            "--termination-receipt",
            targets["interruption_receipt"],
            "--output",
            targets["final_receipt"],
            "--expected-interrupt-source-sha256",
            interrupt_sha256,
            "--expected-source-sha256",
            controller.FROZEN_V2_COMPARATOR_SHA256,
            "--expected-provenance-sha256",
            provenance["repository"]["execution_relevant_digest"]["value"],
        ],
    }

    request_producer_source = tmp_path / "wave7_exact_resume_request.py"
    request_producer_source.write_bytes(REQUEST_PRODUCER_SCRIPT.read_bytes())
    input_attestation_source = tmp_path / "input_attestation.py"
    input_attestation_source.write_bytes(INPUT_ATTESTATION_SOURCE.read_bytes())
    model_weight_identity_source = tmp_path / "parity.py"
    model_weight_identity_source.write_bytes(
        controller.MODEL_WEIGHT_IDENTITY_SOURCE.read_bytes()
    )
    controller.REQUEST_PRODUCER_SOURCE = request_producer_source.resolve()
    controller.FROZEN_REQUEST_PRODUCER_SOURCE_SHA256 = _sha256(
        request_producer_source.read_bytes()
    )
    controller.INPUT_ATTESTATION_SOURCE = input_attestation_source.resolve()
    controller.MODEL_WEIGHT_IDENTITY_SOURCE = model_weight_identity_source.resolve()
    source_paths = [
        fake.resolve(),
        PYTHON_EXECUTABLE,
        ACCELERATE_EXECUTABLE,
        TRAIN_SCRIPT.resolve(),
        INTERRUPT_SCRIPT.resolve(),
        COMPARE_V2_SCRIPT.resolve(),
        (
            REPO_ROOT / "scripts/probes/coordexp_swift/wave7_exact_resume_compare.py"
        ).resolve(),
        (
            REPO_ROOT / "scripts/probes/coordexp_swift/wave7_determinism_preflight.py"
        ).resolve(),
        SCRIPT.resolve(),
        request_producer_source.resolve(),
        input_attestation_source.resolve(),
        model_weight_identity_source.resolve(),
    ]
    request_payload = {
        "schema": controller.REQUEST_SCHEMA,
        "status": "prepared",
        "amendment": _payload_binding(
            amendment_path, amendment, digest_field="amendment_sha256"
        ),
        "legacy_r4_failure": r4_binding,
        "predecessor_sequence_failure": predecessor_binding,
        "predecessor_preflight_failure": predecessor_preflight_failure,
        "determinism_preflight_plan": determinism_plan_binding,
        "determinism_preflight": _binding(determinism_path, determinism),
        "runtime_receipt": _binding(runtime_path, runtime),
        "cache_input_attestation": _payload_binding(
            cache_attestation_path,
            cache_attestation,
            digest_field="attestation_sha256",
        ),
        "model_input_attestation": _payload_binding(
            model_attestation_path,
            model_attestation,
            digest_field="attestation_sha256",
        ),
        "source_inventory": [_source_binding(path) for path in sorted(source_paths)],
        "config_paths": configs,
        "provenance_sha256": _sha256(_canonical(provenance)),
        "environment": {
            "COORDEXP_SWIFT_PACK_CACHE_ROOT": str(cache_root.resolve()),
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "FLASH_ATTENTION_DETERMINISTIC": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        },
        "run_roots": run_roots,
        "targets": targets,
        "commands": commands,
        "oracles": {
            "pre_child": {
                "schema": "coordexp-swift-wave7-pre-child-gate-v1",
                "status": "passed",
                "child_launch_authorized": True,
                "mismatches": [],
            },
            "final": {
                "schema": "coordexp-swift-wave7-exact-resume-comparison-v2",
                "status": "passed",
                "mismatches": [],
            },
        },
        "policy": {
            "phase_order": list(controller.PHASE_ORDER),
            "gpu_phases": list(controller.GPU_PHASES),
            "gpu_admission_mode": "shared_preexisting_subset_v1",
            "gpu_memory_total_mib": 81_920,
            "gpu_memory_used_ceiling_mib": 49_152,
            "gpu_memory_headroom_floor_mib": 32_768,
            "gpu_post_cleanup_stability_seconds": 2.0,
            "no_retry": True,
            "max_attempts_per_phase": 1,
            "preserve_failed_artifacts": True,
            "phase_timeout_seconds": {phase: 5.0 for phase in controller.PHASE_ORDER},
            "term_grace_seconds": 0.25,
            "kill_grace_seconds": 0.25,
            "gpu_baseline_stability_seconds": 2.0,
            "max_total_wall_seconds": 30.0,
            "max_total_gpu_device_seconds": 120.0,
        },
    }
    request = dict(request_payload)
    request["request_payload_sha256"] = _sha256(_canonical(request_payload))
    request_path = controller.R7_REQUEST_PATH
    request_path.write_bytes(_canonical(request) + b"\n")
    plan_path = controller.R7_PLAN_PATH

    def simulated_run_phase(
        phase: str,
        argv: list[str],
        **_: Any,
    ) -> tuple[int, float, dict[str, Any]]:
        assert marker.exists(), "sequence marker must predate every phase"
        with log.open("a", encoding="utf-8") as handle:
            handle.write(phase + "\n")
        if phase in {"uninterrupted", "interrupted_parent", "resume_child"}:
            if phase == "resume_child":
                receipt_index = argv.index("--receipt") + 1
                digest_index = argv.index("--expected-payload-sha256") + 1
                controller._verify_child_gate(
                    Path(argv[receipt_index]), argv[digest_index]
                )
            Path(run_roots[phase]).mkdir(parents=True)
            if phase == "interrupted_parent":
                _write_signed_json(
                    Path(targets["interruption_marker"]),
                    {"schema": "wave7-interrupt-marker-test-v1", "status": "passed"},
                )
                _write_signed_json(
                    Path(targets["interruption_receipt"]),
                    {
                        "schema": "wave7-interrupt-receipt-test-v1",
                        "status": "passed",
                    },
                )
        elif phase == "pre_child":
            _write_signed_json(
                Path(targets["pre_child_receipt"]),
                {
                    "schema": controller.PRE_CHILD_RECEIPT_SCHEMA,
                    "status": "passed",
                    "child_launch_authorized": True,
                    "mismatches": [],
                },
            )
        elif phase == "verify_pre_child":
            receipt = json.loads(
                Path(targets["pre_child_receipt"]).read_text(encoding="utf-8")
            )
            digest = argv[argv.index("--expected-payload-sha256") + 1]
            assert receipt["receipt_payload_sha256"] == digest
        elif phase == "final_compare":
            _write_signed_json(
                Path(targets["final_receipt"]),
                {
                    "schema": controller.FINAL_RECEIPT_SCHEMA,
                    "status": "passed",
                    "mismatches": [],
                },
            )
        else:
            raise AssertionError(phase)
        return (
            0,
            0.001,
            _fake_process_record(),
        )

    controller._run_phase = simulated_run_phase
    return (
        request_path,
        plan_path,
        {
            "log": log,
            "marker": marker,
            "terminal": terminal,
            "sidecar": sidecar,
            "targets": targets,
            "configs": configs,
            "request": request,
            "predecessor_path": predecessor_path,
            "predecessor_binding": predecessor_binding,
            "predecessor_preflight_failure": predecessor_preflight_failure,
            "provenance": provenance,
            "determinism_path": determinism_path,
            "determinism_plan_path": determinism_plan_path,
            "simulated_run_phase": simulated_run_phase,
            "cache_attestation_path": cache_attestation_path,
            "model_attestation_path": model_attestation_path,
            "cache_native_files": cache_native_files,
            "cache_preparation_receipt": cache_preparation_receipt,
            "model_native_files": model_native_files,
            "request_producer_source": request_producer_source,
            "input_attestation_source": input_attestation_source,
            "model_weight_identity_source": model_weight_identity_source,
        },
    )


def _set_required_environment(
    monkeypatch: pytest.MonkeyPatch, request: dict[str, Any]
) -> None:
    for key, value in request["environment"].items():
        monkeypatch.setenv(key, value)


def _nvidia_gpu_csv(
    *, memory_total_mib: int = 81_920, memory_used_mib: int = 12_345
) -> str:
    return "".join(
        f"{index}, GPU-{index}, {memory_total_mib}, "
        f"{memory_used_mib + index}, {index * 3}\n"
        for index in range(8)
    )


def _shared_gpu_baseline(controller: ModuleType) -> dict[str, Any]:
    gpu_inventory = [
        {
            "index": index,
            "gpu_uuid": f"GPU-{index}",
            "memory_total_mib": 81_920,
            "memory_used_mib": 12_345 + index,
            "memory_headroom_mib": 81_920 - (12_345 + index),
            "utilization_gpu_percent": index * 3,
        }
        for index in range(8)
    ]
    compute_inventory = [
        {"gpu_uuid": "GPU-0", "driver_pid": 101},
        {"gpu_uuid": "GPU-3", "driver_pid": 303},
    ]
    samples = [
        {
            "checked_at": f"2026-08-11T00:00:0{offset}+00:00",
            "gpu_inventory": copy.deepcopy(gpu_inventory),
            "compute_inventory": copy.deepcopy(compute_inventory),
        }
        for offset in range(2)
    ]
    return {
        "mode": controller.SHARED_GPU_ADMISSION_MODE,
        "stability_seconds": 2.0,
        "observed_stability_seconds": 2.0,
        "sample_monotonic_ns": [10_000_000_000, 12_000_000_000],
        "memory_total_mib": controller.SHARED_GPU_MEMORY_TOTAL_MIB,
        "memory_used_ceiling_mib": controller.SHARED_GPU_MEMORY_USED_CEILING_MIB,
        "memory_headroom_floor_mib": controller.SHARED_GPU_MEMORY_HEADROOM_FLOOR_MIB,
        "device_inventory": [
            {"index": index, "gpu_uuid": f"GPU-{index}"} for index in range(8)
        ],
        "compute_inventory": compute_inventory,
        "samples": samples,
    }


def _shared_gpu_subset_sample(
    controller: ModuleType, *, include_new_row: bool = False
) -> dict[str, Any]:
    sample = copy.deepcopy(_shared_gpu_baseline(controller)["samples"][1])
    sample["checked_at"] = "2026-08-11T00:00:09+00:00"
    sample["compute_inventory"] = [{"gpu_uuid": "GPU-0", "driver_pid": 101}]
    if include_new_row:
        sample["compute_inventory"].append({"gpu_uuid": "GPU-7", "driver_pid": 707})
    return sample


def _shared_gpu_recovery(
    controller: ModuleType, *, phase: str = "post-uninterrupted"
) -> dict[str, Any]:
    samples = [
        _shared_gpu_subset_sample(controller),
        _shared_gpu_subset_sample(controller),
    ]
    samples[0]["checked_at"] = "2026-08-11T00:00:09+00:00"
    samples[1]["checked_at"] = "2026-08-11T00:00:11+00:00"
    return {
        "phase": phase,
        "stability_seconds": 2.0,
        "sample_monotonic_ns": [10_000_000_000, 12_000_000_000],
        "samples": samples,
    }


def _install_static_shared_gpu_attestation(
    controller: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        controller,
        "_attest_shared_gpu_baseline",
        lambda **_: _shared_gpu_baseline(controller),
    )
    monkeypatch.setattr(
        controller,
        "_assert_shared_gpu_subset",
        lambda **_: _shared_gpu_subset_sample(controller),
    )
    monkeypatch.setattr(
        controller,
        "_attest_shared_gpu_recovery",
        lambda **kwargs: _shared_gpu_recovery(controller, phase=kwargs["phase"]),
    )


def _install_nvidia_samples(
    controller: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    *,
    compute_samples: list[str],
    memory_used_mib: int = 12_345,
    gpu_samples: list[str] | None = None,
) -> list[list[str]]:
    commands: list[list[str]] = []
    outputs: list[str] = []
    if gpu_samples is None:
        gpu_samples = [
            _nvidia_gpu_csv(memory_used_mib=memory_used_mib) for _ in compute_samples
        ]
    assert len(gpu_samples) == len(compute_samples)
    for gpu_csv, compute_csv in zip(gpu_samples, compute_samples, strict=True):
        outputs.extend((gpu_csv, compute_csv))

    def run(
        command: list[str],
        *,
        check: bool,
        stdout: Any,
        stderr: Any,
        timeout: float,
    ) -> subprocess.CompletedProcess[bytes]:
        del check, stderr, timeout
        commands.append(command)
        assert outputs, "unexpected nvidia-smi sample"
        stdout.write(outputs.pop(0).encode("utf-8"))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(controller.subprocess, "run", run)
    monotonic_values = iter((10_000_000_000, 12_000_000_000))
    monkeypatch.setattr(controller.time, "monotonic_ns", lambda: next(monotonic_values))
    monkeypatch.setattr(controller.time, "sleep", lambda _: None)
    return commands


@pytest.mark.parametrize("stream", ("stdout", "stderr"))
def test_nvidia_csv_rejects_each_oversized_stream_with_typed_bound(
    controller: ModuleType,
    stream: str,
) -> None:
    descriptor = 1 if stream == "stdout" else 2
    observed_bytes = controller.MAX_GPU_CSV_BYTES + 1
    program = f"import os; os.write({descriptor}, b'x' * {observed_bytes})"

    with pytest.raises(controller.Wave7SequenceError) as exc_info:
        controller._bounded_nvidia_csv(
            [sys.executable, "-c", program], owner="bounded test inventory"
        )

    assert exc_info.value.code == "wave7_sequence.gpu_inventory"
    assert exc_info.value.context == {
        "stream": stream,
        "maximum_bytes": controller.MAX_GPU_CSV_BYTES,
        "observed_bytes": observed_bytes,
    }


def test_shared_gpu_baseline_accepts_stable_nonempty_compute_inventory(
    controller: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands = _install_nvidia_samples(
        controller,
        monkeypatch,
        compute_samples=["GPU-0, 101\nGPU-3, 303\n"] * 2,
    )
    attest = getattr(controller, "_attest_shared_gpu_baseline", None)
    assert callable(attest), "shared-GPU baseline attestor is required"
    baseline = attest(
        stability_seconds=2.0,
        memory_total_mib=81_920,
        memory_used_ceiling_mib=49_152,
        memory_headroom_floor_mib=32_768,
    )

    assert baseline == {
        "mode": "shared_preexisting_subset_v1",
        "stability_seconds": 2.0,
        "observed_stability_seconds": baseline["observed_stability_seconds"],
        "sample_monotonic_ns": [10_000_000_000, 12_000_000_000],
        "memory_total_mib": 81_920,
        "memory_used_ceiling_mib": 49_152,
        "memory_headroom_floor_mib": 32_768,
        "device_inventory": [
            {"index": index, "gpu_uuid": f"GPU-{index}"} for index in range(8)
        ],
        "compute_inventory": [
            {"gpu_uuid": "GPU-0", "driver_pid": 101},
            {"gpu_uuid": "GPU-3", "driver_pid": 303},
        ],
        "samples": baseline["samples"],
    }
    assert len(baseline["samples"]) == 2
    assert baseline["observed_stability_seconds"] == 2.0
    assert (
        commands
        == [
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
        ]
        * 2
    )


@pytest.mark.parametrize("drift", ("compute", "device_uuid"))
def test_shared_gpu_baseline_rejects_changing_inventory(
    controller: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    second_gpu_csv = _nvidia_gpu_csv()
    second_compute_csv = "GPU-0, 101\n"
    if drift == "compute":
        second_compute_csv = "GPU-0, 102\n"
    else:
        second_gpu_csv = second_gpu_csv.replace("GPU-7", "GPU-replaced")
    _install_nvidia_samples(
        controller,
        monkeypatch,
        compute_samples=["GPU-0, 101\n", second_compute_csv],
        gpu_samples=[_nvidia_gpu_csv(), second_gpu_csv],
    )
    attest = getattr(controller, "_attest_shared_gpu_baseline", None)
    assert callable(attest), "shared-GPU baseline attestor is required"

    with pytest.raises(controller.Wave7SequenceError) as exc_info:
        attest(
            stability_seconds=2.0,
            memory_total_mib=81_920,
            memory_used_ceiling_mib=49_152,
            memory_headroom_floor_mib=32_768,
        )

    assert exc_info.value.code == "wave7_sequence.gpu_baseline"


def test_shared_gpu_baseline_treats_utilization_as_observational(
    controller: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    second_gpu_csv = "".join(
        f"{index}, GPU-{index}, 81920, {20_000 + index}, {100 - index}\n"
        for index in range(8)
    )
    _install_nvidia_samples(
        controller,
        monkeypatch,
        compute_samples=["GPU-0, 101\n"] * 2,
        gpu_samples=[_nvidia_gpu_csv(), second_gpu_csv],
    )

    baseline = controller._attest_shared_gpu_baseline(
        stability_seconds=2.0,
        memory_total_mib=81_920,
        memory_used_ceiling_mib=49_152,
        memory_headroom_floor_mib=32_768,
    )

    assert (
        baseline["samples"][0]["gpu_inventory"]
        != baseline["samples"][1]["gpu_inventory"]
    )
    assert baseline["compute_inventory"] == [{"gpu_uuid": "GPU-0", "driver_pid": 101}]


@pytest.mark.parametrize("observed_seconds", (1.999, 0.01))
def test_shared_gpu_baseline_rejects_less_than_hard_two_second_measurement(
    controller: ModuleType, observed_seconds: float
) -> None:
    baseline = _shared_gpu_baseline(controller)
    baseline["sample_monotonic_ns"][1] = baseline["sample_monotonic_ns"][0] + int(
        observed_seconds * 1_000_000_000
    )
    baseline["observed_stability_seconds"] = observed_seconds

    with pytest.raises(controller.Wave7SequenceError) as exc_info:
        controller._validate_shared_gpu_baseline(
            baseline, expected_stability_seconds=2.0
        )

    assert exc_info.value.code == "wave7_sequence.gpu_baseline"


def test_shared_gpu_sample_rejects_memory_above_fixed_ceiling(
    controller: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_nvidia_samples(
        controller,
        monkeypatch,
        compute_samples=["GPU-0, 101\n"],
        memory_used_mib=49_153,
    )

    with pytest.raises(controller.Wave7SequenceError) as exc_info:
        controller._gpu_compute_sample(
            memory_total_mib=81_920,
            memory_used_ceiling_mib=49_152,
            memory_headroom_floor_mib=32_768,
        )

    assert exc_info.value.code == "wave7_sequence.gpu_memory_ceiling"


@pytest.mark.parametrize(
    ("observed", "accepted"),
    (
        ("GPU-0, 101\n", True),
        ("GPU-0, 101\nGPU-7, 707\n", False),
    ),
)
def test_shared_gpu_phase_inventory_allows_only_baseline_subsets(
    controller: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    observed: str,
    accepted: bool,
) -> None:
    _install_nvidia_samples(
        controller,
        monkeypatch,
        compute_samples=["GPU-0, 101\nGPU-3, 303\n"] * 2 + [observed],
    )
    attest = getattr(controller, "_attest_shared_gpu_baseline", None)
    assert callable(attest), "shared-GPU baseline attestor is required"
    baseline = attest(
        stability_seconds=2.0,
        memory_total_mib=81_920,
        memory_used_ceiling_mib=49_152,
        memory_headroom_floor_mib=32_768,
    )
    assert_subset = getattr(controller, "_assert_shared_gpu_subset", None)
    assert callable(assert_subset), "shared-GPU subset gate is required"

    if accepted:
        sample = assert_subset(baseline=baseline)
        assert sample["compute_inventory"] == [{"gpu_uuid": "GPU-0", "driver_pid": 101}]
    else:
        with pytest.raises(controller.Wave7SequenceError) as exc_info:
            assert_subset(baseline=baseline)
        assert exc_info.value.code == "wave7_sequence.gpu_added_process"


def test_shared_gpu_post_cleanup_recovery_collects_two_samples_two_seconds_apart(
    controller: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = [
        _shared_gpu_subset_sample(controller),
        _shared_gpu_subset_sample(controller),
    ]
    calls = 0

    def subset(**_: Any) -> dict[str, Any]:
        nonlocal calls
        sample = copy.deepcopy(samples[calls])
        calls += 1
        return sample

    monotonic_values = iter((10_000_000_000, 12_000_000_000))
    sleeps: list[float] = []
    monkeypatch.setattr(controller, "_assert_shared_gpu_subset", subset)
    monkeypatch.setattr(controller.time, "monotonic_ns", lambda: next(monotonic_values))
    monkeypatch.setattr(controller.time, "sleep", sleeps.append)
    attest = getattr(controller, "_attest_shared_gpu_recovery", None)
    assert callable(attest), "post-cleanup shared-GPU recovery attestor is required"

    recovery = attest(
        baseline=_shared_gpu_baseline(controller),
        stability_seconds=2.0,
        phase="post-uninterrupted",
    )

    assert recovery == {
        "phase": "post-uninterrupted",
        "stability_seconds": 2.0,
        "sample_monotonic_ns": [10_000_000_000, 12_000_000_000],
        "samples": samples,
    }
    assert calls == 2
    assert sleeps == [2.0]


@pytest.mark.parametrize(
    "mutation",
    ("zero_samples", "one_sample", "three_samples", "one_second", "new_row"),
)
def test_shared_gpu_post_cleanup_recovery_validator_rejects_incomplete_sweep(
    controller: ModuleType,
    mutation: str,
) -> None:
    recovery = _shared_gpu_recovery(controller)
    if mutation == "zero_samples":
        recovery["samples"] = []
    elif mutation == "one_sample":
        recovery["samples"] = recovery["samples"][:1]
    elif mutation == "three_samples":
        recovery["samples"].append(copy.deepcopy(recovery["samples"][1]))
    else:
        if mutation == "one_second":
            recovery["sample_monotonic_ns"] = [10_000_000_000, 11_000_000_000]
        else:
            recovery["samples"][1]["compute_inventory"].append(
                {"gpu_uuid": "GPU-7", "driver_pid": 707}
            )
    validate = getattr(controller, "_validate_shared_gpu_recovery", None)
    assert callable(validate), "post-cleanup shared-GPU recovery validator is required"

    with pytest.raises(controller.Wave7SequenceError) as exc_info:
        validate(
            recovery,
            expected_stability_seconds=2.0,
            expected_phase="post-uninterrupted",
            baseline=_shared_gpu_baseline(controller),
        )

    assert exc_info.value.code == "wave7_sequence.gpu_recovery"


def test_frozen_comparator_and_predecessor_failure_identities(
    controller: ModuleType,
) -> None:
    # DECLARED RE-PIN (add-coordexp-swift-training-observability, task 5.2);
    # previous seals `dfbb4d63...` (v1) and `2e3b6ea6...` (v2).  Both
    # comparators now classify the new timing/throughput/allocator/
    # availability row fields as non-semantic observations by exact name.
    assert controller.FROZEN_V1_COMPARATOR_SHA256 == (
        "4698eeca1439bf65d55d524acf9d780f6078251756e792258172d3c79147c8d2"
    )
    assert controller.FROZEN_V2_COMPARATOR_SHA256 == (
        "aaac6aeecc825773e1b5d93827a61474581dd8fa894f400713719c2628cada66"
    )
    assert _sha256(controller.REQUIRED_SOURCE_PATHS[0].read_bytes()) == (
        controller.FROZEN_V1_COMPARATOR_SHA256
    )
    assert _sha256(controller.REQUIRED_SOURCE_PATHS[1].read_bytes()) == (
        controller.FROZEN_V2_COMPARATOR_SHA256
    )
    assert controller.FROZEN_R4_FAILURE_FILE_SHA256 == (
        "1d56e8139500ac09e62f57b2d2ce074401bc7596a64bee3b3915ff5036575e5b"
    )
    assert controller.FROZEN_R4_FAILURE_PAYLOAD_SHA256 == (
        "13a39f43f1e504f48a2e70a8037d5ea7ccfd919003ffa1f89417e1b592973f4a"
    )
    assert controller.FROZEN_R5_FAILURE_FILE_SHA256 == (
        "d5244fb6d96b24b1cfac4438628ae141496f7357c286dff5454c19c9f39731b1"
    )
    assert controller.FROZEN_R5_FAILURE_PAYLOAD_SHA256 == (
        "f560cbd7bfeac8e25f031021d9d0053c3df0d69961d52f1b854754ce4f4e7656"
    )
    assert controller.FROZEN_R6_PREFLIGHT_PLAN_FILE_SHA256 == (
        "2d5d5108c07e02ff964f5b3e6e4e42ce84e58bd0c17a8310dd7e2f3e6147e6ec"
    )
    assert controller.FROZEN_R6_PREFLIGHT_PLAN_PAYLOAD_SHA256 == (
        "9b5c820c5f2c366c994a15732b99498b4072d1d552d9a7048c6afb65e0532a8f"
    )
    assert controller.FROZEN_R6_PREFLIGHT_MARKER_FILE_SHA256 == (
        "f19a894ac3113aecc3a126dab7e3327389518c9e12e34d7806ad401adf239ce9"
    )
    assert controller.FROZEN_R6_PREFLIGHT_MARKER_PAYLOAD_SHA256 == (
        "967fc87daabc19dd82d9d5a0f2567a2700da71bf9cb3773f904da63488585d7e"
    )
    assert controller.FROZEN_R6_PREFLIGHT_FAILURE_FILE_SHA256 == (
        "eaf20c86a3fa95d7fa6533faaf7fb0e9b8fe938aa6f62d2ba725c3c604a64784"
    )
    assert controller.FROZEN_R6_PREFLIGHT_FAILURE_PAYLOAD_SHA256 == (
        "4b400e094c494099b611e85d78b2156ce53b0fe372e12de6ff780c493b5770ab"
    )
    assert controller.FROZEN_DETERMINISM_PREFLIGHT_SOURCE_SHA256 == (
        "52d6e2a3b709ede73f5794eefdbfd6c57997714af29f5329e936b8953492346c"
    )
    assert _sha256(controller.REQUIRED_SOURCE_PATHS[3].read_bytes()) == (
        controller.FROZEN_DETERMINISM_PREFLIGHT_SOURCE_SHA256
    )
    assert _sha256(controller.REQUEST_PRODUCER_SOURCE.read_bytes()) == (
        controller.FROZEN_REQUEST_PRODUCER_SOURCE_SHA256
    )
    assert _sha256(controller.INPUT_ATTESTATION_SOURCE.read_bytes()) == (
        controller.FROZEN_INPUT_ATTESTATION_SOURCE_SHA256
    )
    assert _sha256(controller.MODEL_WEIGHT_IDENTITY_SOURCE.read_bytes()) == (
        controller.FROZEN_PARITY_SOURCE_SHA256
    )


def test_shared_gpu_sequence_schema_and_claim_scope_are_bounded(
    controller: ModuleType,
) -> None:
    assert controller.REQUEST_SCHEMA.endswith("request-v6")
    assert controller.PLAN_SCHEMA.endswith("plan-v6")
    assert controller.MARKER_SCHEMA.endswith("marker-v6")
    assert controller.RECEIPT_SCHEMA.endswith("receipt-v6")
    assert controller.AMENDMENT_SCHEMA == "coordexp-swift-wave7-r7-amendment-v4"
    assert controller.SEQUENCE_CLAIM_SCOPE == {
        "gpu_occupancy": "shared_preexisting_compute_processes_allowed",
        "establishes": [
            "correctness_evidence",
            "artifact_evidence",
            "exact_resume_correctness_evidence",
        ],
        "does_not_establish": [
            "throughput",
            "performance",
            "resource_capacity_or_promotion",
        ],
    }
    assert all(
        "performance" not in claim
        for claim in controller.SEQUENCE_CLAIM_SCOPE["establishes"]
    )


def test_train_command_uses_accelerate_module_entry_and_keeps_file_owner(
    controller: ModuleType,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "train.yaml"

    command = controller._train_command(config_path, port=29_681)

    assert command == [
        str(controller.ACCELERATE_EXECUTABLE),
        "launch",
        "--multi_gpu",
        "--num_processes",
        "8",
        "--main_process_port",
        "29681",
        "--module",
        "src.train",
        "--config",
        str(config_path),
    ]
    assert str(controller.TRAIN_ENTRYPOINT) not in command
    assert controller.TRAIN_ENTRYPOINT in controller.REQUIRED_SOURCE_PATHS


def test_public_train_module_reaches_argument_parser_without_pythonpath() -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [sys.executable, "-m", "src.train", "--help"],
        cwd=REPO_ROOT,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30.0,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--config" in completed.stdout


def test_prepare_rejects_historical_request_v4_before_any_publication(
    controller: ModuleType,
    tmp_path: Path,
) -> None:
    request_path, plan_path, paths = _request_fixture(controller, tmp_path)
    request = paths["request"]
    request["schema"] = "coordexp-swift-wave7-exact-resume-sequence-request-v4"
    unsigned = dict(request)
    unsigned.pop("request_payload_sha256")
    request["request_payload_sha256"] = _sha256(_canonical(unsigned))
    request_path.write_bytes(_canonical(request) + b"\n")

    assert controller.prepare(request_path, plan_path) == 2
    assert not plan_path.exists()
    assert not paths["marker"].exists()
    assert all(not Path(path).exists() for path in request["run_roots"].values())


@pytest.mark.parametrize(
    "mutation",
    (
        "cache_chunk",
        "cache_manifest",
        "cache_preparation_receipt",
        "model_index",
        "model_shard",
        "model_standalone_weight",
        "cache_attestation_joint_resign",
        "model_attestation_joint_resign",
        "model_hash_policy_joint_resign",
        "request_producer_source",
        "input_attestation_source",
        "model_weight_identity_source",
    ),
)
def test_prepare_rejects_native_input_or_source_drift_before_plan_marker_or_run(
    controller: ModuleType,
    tmp_path: Path,
    mutation: str,
) -> None:
    request_path, plan_path, paths = _request_fixture(controller, tmp_path)
    request = paths["request"]
    if mutation == "cache_chunk":
        paths["cache_native_files"]["train_chunk"].write_bytes(b"mutated-chunk")
    elif mutation == "cache_manifest":
        paths["cache_native_files"]["eval-forward_manifest"].write_text(
            "mutated-manifest\n", encoding="utf-8"
        )
    elif mutation == "cache_preparation_receipt":
        paths["cache_preparation_receipt"].write_text(
            "mutated-preparation\n", encoding="utf-8"
        )
    elif mutation == "model_index":
        paths["model_native_files"]["index"].write_bytes(b"mutated-index")
    elif mutation == "model_shard":
        paths["model_native_files"]["shard"].write_bytes(b"mutated-shard")
    elif mutation == "model_standalone_weight":
        paths["model_native_files"]["standalone"].write_bytes(b"mutated-weight")
    elif mutation in {
        "cache_attestation_joint_resign",
        "model_attestation_joint_resign",
        "model_hash_policy_joint_resign",
    }:
        owner = mutation.split("_", maxsplit=1)[0]
        attestation_path = paths[f"{owner}_attestation_path"]
        attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
        attestation.pop("attestation_sha256")
        if owner == "cache":
            attestation["splits"]["train"]["chunks"][0]["file_sha256"] = "f" * 64
        elif mutation == "model_attestation_joint_resign":
            attestation["base_model_weight_identity"]["shard"]["file_sha256"] = "f" * 64
        else:
            attestation["weight_hash_execution_policy"]["resolved_workers"] = 1
        attestation["attestation_sha256"] = _sha256(_canonical(attestation))
        attestation_path.write_bytes(_canonical(attestation) + b"\n")
        request[f"{owner}_input_attestation"] = _payload_binding(
            attestation_path, attestation, digest_field="attestation_sha256"
        )
        unsigned_request = dict(request)
        unsigned_request.pop("request_payload_sha256")
        request["request_payload_sha256"] = _sha256(_canonical(unsigned_request))
        request_path.write_bytes(_canonical(request) + b"\n")
    else:
        paths[mutation].write_text("source drift\n", encoding="utf-8")

    assert controller.prepare(request_path, plan_path) == 2
    assert not plan_path.exists()
    assert not paths["marker"].exists()
    assert all(not Path(path).exists() for path in request["run_roots"].values())


@pytest.mark.parametrize(
    "mutation",
    (
        "arbitrary_role_yaml",
        "legacy_determinism",
        "wrong_role",
        "wrong_run_root",
        "wrong_resume_mode",
        "wrong_checkpoint_dir",
        "wrong_cadence",
        "semantic_drift",
    ),
)
def test_prepare_rejects_non_exact_role_configs_before_plan_or_marker(
    controller: ModuleType,
    tmp_path: Path,
    mutation: str,
) -> None:
    request_path, plan_path, paths = _request_fixture(controller, tmp_path)
    role = {
        "arbitrary_role_yaml": "uninterrupted",
        "legacy_determinism": "uninterrupted",
        "wrong_role": "uninterrupted",
        "wrong_run_root": "interrupted_parent",
        "wrong_resume_mode": "resume_child",
        "wrong_checkpoint_dir": "resume_child",
        "wrong_cadence": "interrupted_parent",
        "semantic_drift": "resume_child",
    }[mutation]
    config_path = Path(paths["configs"][role])
    original = config_path.read_text(encoding="utf-8")
    if mutation == "arbitrary_role_yaml":
        mutated = f"schema_version: 1\nrole: {role}\n"
    else:
        payload = yaml.safe_load(original)
        if mutation == "legacy_determinism":
            payload["runtime"]["determinism"]["mode"] = "legacy"
        elif mutation == "wrong_role":
            payload["run"]["name"] = "wrong-role"
        elif mutation == "wrong_run_root":
            payload["run"]["artifact_root"] = str(tmp_path / "wrong-root")
        elif mutation == "wrong_resume_mode":
            payload["resume"] = {"mode": "disabled", "checkpoint_dir": None}
        elif mutation == "wrong_checkpoint_dir":
            payload["resume"]["checkpoint_dir"] = str(tmp_path / "wrong-step-3")
        elif mutation == "wrong_cadence":
            payload["checkpoint"]["steps"] = [3]
        else:
            payload["runtime"]["seed"] = 18
        mutated = yaml.safe_dump(payload, sort_keys=False)
    assert mutated != original
    config_path.write_text(mutated, encoding="utf-8")

    assert controller.prepare(request_path, plan_path) == 2

    assert not plan_path.exists()
    assert not paths["marker"].exists()
    assert all(not Path(path).exists() for path in paths["targets"].values())
    assert all(
        not Path(path).exists() for path in paths["request"]["run_roots"].values()
    )


def test_prepare_rejects_arbitrary_valid_provenance_digest_before_plan_or_marker(
    controller: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    request_path, plan_path, paths = _request_fixture(controller, tmp_path)
    request = paths["request"]
    assert request["provenance_sha256"] != "f" * 64
    request["provenance_sha256"] = "f" * 64
    unsigned = dict(request)
    unsigned.pop("request_payload_sha256")
    request["request_payload_sha256"] = _sha256(_canonical(unsigned))
    request_path.write_bytes(_canonical(request) + b"\n")

    assert controller.prepare(request_path, plan_path) == 2

    assert "wave7_sequence.runtime_provenance" in capsys.readouterr().err
    assert not plan_path.exists()
    assert all(not Path(path).exists() for path in paths["targets"].values())
    assert all(not Path(path).exists() for path in request["run_roots"].values())


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("gpu_admission_mode", "idle_only"),
        ("gpu_memory_total_mib", 80_000),
        ("gpu_memory_used_ceiling_mib", 65_536),
        ("gpu_memory_used_ceiling_mib", 49_152.0),
        ("gpu_memory_headroom_floor_mib", 16_384),
    ),
)
def test_prepare_rejects_permissive_shared_gpu_policy_override(
    controller: ModuleType,
    tmp_path: Path,
    field: str,
    value: Any,
) -> None:
    request_path, plan_path, paths = _request_fixture(controller, tmp_path)
    request = paths["request"]
    request["policy"][field] = value
    unsigned = dict(request)
    unsigned.pop("request_payload_sha256")
    request["request_payload_sha256"] = _sha256(_canonical(unsigned))
    request_path.write_bytes(_canonical(request) + b"\n")

    assert controller.prepare(request_path, plan_path) == 2

    assert not plan_path.exists()
    assert not paths["marker"].exists()
    assert all(not Path(path).exists() for path in request["run_roots"].values())


def test_policy_accepts_exact_r7_authority_boundaries(
    controller: ModuleType,
    tmp_path: Path,
) -> None:
    _, _, paths = _request_fixture(controller, tmp_path)
    policy = copy.deepcopy(paths["request"]["policy"])
    policy["phase_timeout_seconds"] = {phase: 600.0 for phase in controller.PHASE_ORDER}
    policy["max_total_wall_seconds"] = 2400.0
    policy["max_total_gpu_device_seconds"] = 14400.0

    normalized = controller._validate_policy(policy)

    assert normalized["phase_timeout_seconds"] == policy["phase_timeout_seconds"]
    assert normalized["max_total_wall_seconds"] == 2400.0
    assert normalized["max_total_gpu_device_seconds"] == 14400.0


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("phase_timeout_seconds", 600.000000001),
        ("max_total_wall_seconds", 2400.000000001),
        ("max_total_gpu_device_seconds", 14400.000000001),
    ),
)
def test_prepare_rejects_r7_authority_cost_overages_before_plan_or_marker(
    controller: ModuleType,
    tmp_path: Path,
    field: str,
    value: float,
) -> None:
    request_path, plan_path, paths = _request_fixture(controller, tmp_path)
    request = paths["request"]
    if field == "phase_timeout_seconds":
        request["policy"][field]["uninterrupted"] = value
    else:
        request["policy"][field] = value
    unsigned = dict(request)
    unsigned.pop("request_payload_sha256")
    request["request_payload_sha256"] = _sha256(_canonical(unsigned))
    request_path.write_bytes(_canonical(request) + b"\n")

    assert controller.prepare(request_path, plan_path) == 2
    assert not plan_path.exists()
    assert not paths["marker"].exists()


@pytest.mark.parametrize("surface", ("request", "run_root", "cache", "external"))
def test_prepare_rejects_noncanonical_r7_namespace_before_plan_or_marker(
    controller: ModuleType,
    tmp_path: Path,
    surface: str,
) -> None:
    request_path, plan_path, paths = _request_fixture(controller, tmp_path)
    request = paths["request"]
    prepared_path = request_path
    if surface == "request":
        prepared_path = tmp_path / "request-v6-alias.json"
        prepared_path.write_bytes(request_path.read_bytes())
    elif surface == "run_root":
        request["run_roots"]["uninterrupted"] = str(
            (tmp_path / "runs" / "uninterrupted-alias").resolve()
        )
    elif surface == "cache":
        alias_cache = tmp_path / "cache-alias"
        alias_cache.mkdir()
        cache_path = paths["cache_attestation_path"]
        cache_payload = json.loads(cache_path.read_text(encoding="utf-8"))
        cache_payload.pop("attestation_sha256")
        cache_payload["cache_root"] = str(alias_cache.resolve())
        signed_cache = _write_digest_json(
            cache_path, cache_payload, digest_field="attestation_sha256"
        )
        request["cache_input_attestation"] = _payload_binding(
            cache_path, signed_cache, digest_field="attestation_sha256"
        )
        request["environment"]["COORDEXP_SWIFT_PACK_CACHE_ROOT"] = str(
            alias_cache.resolve()
        )
    else:
        runtime_binding = request["runtime_receipt"]
        alias_runtime = tmp_path.parent / f"{tmp_path.name}-runtime.json"
        alias_runtime.write_bytes(Path(runtime_binding["path"]).read_bytes())
        runtime_binding["path"] = str(alias_runtime.resolve())
    if surface != "request":
        unsigned = dict(request)
        unsigned.pop("request_payload_sha256")
        request["request_payload_sha256"] = _sha256(_canonical(unsigned))
        request_path.write_bytes(_canonical(request) + b"\n")

    assert controller.prepare(prepared_path, plan_path) == 2
    assert not plan_path.exists()
    assert not paths["marker"].exists()


def test_execute_rejects_noncanonical_plan_alias_before_marker(
    controller: ModuleType,
    tmp_path: Path,
) -> None:
    request_path, plan_path, paths = _request_fixture(controller, tmp_path)
    assert controller.prepare(request_path, plan_path) == 0
    alias_plan = tmp_path / "sequence-plan-v6-alias.json"
    alias_plan.write_bytes(plan_path.read_bytes())

    assert controller.execute(alias_plan) == 2
    assert not paths["marker"].exists()


def _tree_file_snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_verify_plan_cli_exactly_revalidates_without_writes(
    controller: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path, plan_path, paths = _request_fixture(controller, tmp_path)
    _set_required_environment(monkeypatch, paths["request"])
    assert controller.prepare(request_path, plan_path) == 0
    expected_file_sha256 = _sha256(plan_path.read_bytes())
    before = _tree_file_snapshot(tmp_path)

    assert (
        controller.main(
            [
                "verify-plan",
                "--plan",
                str(plan_path),
                "--expected-file-sha256",
                expected_file_sha256,
            ]
        )
        == 0
    )

    assert _tree_file_snapshot(tmp_path) == before
    assert not paths["marker"].exists()
    assert not paths["terminal"].exists()
    assert all(
        not Path(path).exists() for path in paths["request"]["run_roots"].values()
    )


@pytest.mark.parametrize("mutation", ("wrong_hash", "request_drift", "collision"))
def test_verify_plan_cli_rejects_drift_or_collision_without_publication(
    controller: ModuleType,
    tmp_path: Path,
    mutation: str,
) -> None:
    request_path, plan_path, paths = _request_fixture(controller, tmp_path)
    assert controller.prepare(request_path, plan_path) == 0
    expected_file_sha256 = _sha256(plan_path.read_bytes())
    if mutation == "wrong_hash":
        expected_file_sha256 = "0" * 64
    elif mutation == "request_drift":
        request_path.write_bytes(request_path.read_bytes() + b" ")
    else:
        paths["marker"].write_text("collision\n", encoding="utf-8")
    before = _tree_file_snapshot(tmp_path)

    assert (
        controller.main(
            [
                "verify-plan",
                "--plan",
                str(plan_path),
                "--expected-file-sha256",
                expected_file_sha256,
            ]
        )
        == 2
    )

    assert _tree_file_snapshot(tmp_path) == before
    assert not paths["terminal"].exists()
    assert all(
        not Path(path).exists() for path in paths["request"]["run_roots"].values()
    )


def test_prepare_rejects_equivalent_copied_amendment_authority(
    controller: ModuleType,
    tmp_path: Path,
) -> None:
    request_path, plan_path, paths = _request_fixture(controller, tmp_path)
    request = paths["request"]
    copied_authority = tmp_path / "measurement-plan-copy.md"
    copied_authority.write_bytes(controller.AMENDMENT_AUTHORITY_PATH.read_bytes())
    amendment_path = Path(request["amendment"]["path"])
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    amendment.pop("amendment_sha256")
    amendment["authority"]["path"] = str(copied_authority.resolve())
    signed = _write_digest_json(
        amendment_path, amendment, digest_field="amendment_sha256"
    )
    request["amendment"] = _payload_binding(
        amendment_path, signed, digest_field="amendment_sha256"
    )
    unsigned = dict(request)
    unsigned.pop("request_payload_sha256")
    request["request_payload_sha256"] = _sha256(_canonical(unsigned))
    request_path.write_bytes(_canonical(request) + b"\n")

    assert controller.prepare(request_path, plan_path) == 2
    assert not plan_path.exists()
    assert not paths["marker"].exists()


@pytest.mark.parametrize(
    "mutation",
    (
        "cleanup_summary",
        "process_cleanup",
        "cleanup_survivor",
        "postlaunch_added_row",
        "postlaunch_zero_samples",
        "postlaunch_one_sample",
        "postlaunch_one_second",
        "native_mapping_mismatch",
    ),
)
def test_prepare_rejects_non_fail_closed_v4_preflight_cleanup(
    controller: ModuleType,
    tmp_path: Path,
    mutation: str,
) -> None:
    request_path, plan_path, paths = _request_fixture(controller, tmp_path)
    determinism_path = paths["determinism_path"]
    signed = json.loads(determinism_path.read_text(encoding="utf-8"))
    signed.pop("receipt_payload_sha256")
    if mutation == "cleanup_summary":
        signed["cleanup"]["all_process_groups_exited"] = False
    elif mutation == "process_cleanup":
        signed["processes"][0]["cleanup_verified"] = False
    elif mutation == "cleanup_survivor":
        process = signed["processes"][0]
        scope = process["process_scope"]
        scope["remaining_members"] = [copy.deepcopy(scope["observed_members"][0])]
        scope["cleanup_failure"] = "surviving_members_after_sigkill"
        scope["cleanup_verified"] = False
        process["cleanup_verified"] = False
    elif mutation == "postlaunch_added_row":
        sweep = signed["gpu_shared_occupancy_sweeps"][0]
        sweep["new_compute_processes"] = [{"gpu_uuid": "GPU-7", "pid": 707}]
        sweep["admitted"] = False
    elif mutation == "postlaunch_zero_samples":
        signed["gpu_shared_occupancy_sweeps"][0]["samples"] = []
    elif mutation == "postlaunch_one_sample":
        sweep = signed["gpu_shared_occupancy_sweeps"][0]
        sweep["samples"] = sweep["samples"][:1]
    elif mutation == "postlaunch_one_second":
        sweep = signed["gpu_shared_occupancy_sweeps"][0]
        sweep["samples"][1]["sample_monotonic_ns"] = (
            sweep["samples"][0]["sample_monotonic_ns"] + 1_000_000_000
        )
    else:
        signed["comparisons"]["native_mapping_equal"] = False
    rewritten = _write_signed_json(determinism_path, signed)
    request = paths["request"]
    request["determinism_preflight"] = _binding(determinism_path, rewritten)
    unsigned = dict(request)
    unsigned.pop("request_payload_sha256")
    request["request_payload_sha256"] = _sha256(_canonical(unsigned))
    request_path.write_bytes(_canonical(request) + b"\n")

    assert controller.prepare(request_path, plan_path) == 2

    assert not plan_path.exists()
    assert not paths["marker"].exists()
    assert all(not Path(path).exists() for path in request["run_roots"].values())


@pytest.mark.parametrize(
    "mutation",
    (
        "plan_source",
        "marker_baseline",
        "rank_workload",
        "rank_native",
        "sweep_inventory",
        "file_binding",
    ),
)
def test_preflight_production_verifier_rejection_is_sequence_fail_closed(
    controller: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    plan_path = tmp_path / "preflight-plan.json"
    terminal_path = tmp_path / "preflight-terminal.json"
    plan_path.write_text("{}\n", encoding="utf-8")
    terminal_path.write_text("{}\n", encoding="utf-8")
    plan_binding = {
        "path": str(plan_path.resolve()),
        "file_sha256": _sha256(plan_path.read_bytes()),
        "payload_sha256": "a" * 64,
        "schema": controller.DETERMINISM_PREFLIGHT_PLAN_SCHEMA,
        "status": "prepared",
    }
    terminal_binding = {
        "path": str(terminal_path.resolve()),
        "file_sha256": _sha256(terminal_path.read_bytes()),
        "payload_sha256": "b" * 64,
        "schema": controller.DETERMINISM_PREFLIGHT_RECEIPT_SCHEMA,
        "status": "passed",
    }
    monkeypatch.setattr(controller.determinism_preflight, "_load_json", lambda _: {})

    def validate_plan(*_: Any, **__: Any) -> dict[str, Any]:
        if mutation == "plan_source":
            raise RuntimeError("production plan source rejection")
        return {"plan_payload_sha256": "a" * 64}

    def verify_receipt(*_: Any, **__: Any) -> dict[str, Any]:
        raise RuntimeError(f"production {mutation} rejection")

    monkeypatch.setattr(
        controller.determinism_preflight, "_validate_plan", validate_plan
    )
    monkeypatch.setattr(
        controller.determinism_preflight, "verify_receipt", verify_receipt
    )
    if mutation == "file_binding":
        terminal_binding["file_sha256"] = "c" * 64

    with pytest.raises(controller.Wave7SequenceError) as exc_info:
        controller._validate_determinism_preflight_evidence(
            plan_binding=plan_binding,
            terminal_binding=terminal_binding,
        )

    assert exc_info.value.code == "wave7_sequence.determinism_preflight"


@pytest.mark.parametrize(
    "mutation",
    (
        "substitute_executable",
        "script_entrypoint",
        "missing_config",
        "swap_gpu_roles",
        "wrong_controller",
        "wrong_subcommand",
        "missing_child_gate",
    ),
)
def test_prepare_rejects_non_production_command_contract_before_marker(
    controller: ModuleType,
    tmp_path: Path,
    mutation: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    request_path, plan_path, paths = _request_fixture(controller, tmp_path)
    request = paths["request"]
    commands = request["commands"]
    if mutation == "substitute_executable":
        commands["uninterrupted"][0] = str(PYTHON_EXECUTABLE)
    elif mutation == "script_entrypoint":
        module_index = commands["uninterrupted"].index("--module")
        commands["uninterrupted"][module_index : module_index + 2] = [
            str(TRAIN_SCRIPT.resolve())
        ]
    elif mutation == "missing_config":
        del commands["uninterrupted"][-2:]
    elif mutation == "swap_gpu_roles":
        separator = commands["resume_child"].index("--")
        commands["uninterrupted"] = commands["resume_child"][separator + 1 :]
    elif mutation == "wrong_controller":
        commands["pre_child"][1] = str(INTERRUPT_SCRIPT.resolve())
    elif mutation == "wrong_subcommand":
        commands["pre_child"][2] = "compare"
    else:
        index = commands["resume_child"].index("--expected-payload-sha256")
        del commands["resume_child"][index : index + 2]
    unsigned = dict(request)
    unsigned.pop("request_payload_sha256")
    request["request_payload_sha256"] = _sha256(_canonical(unsigned))
    request_path.write_bytes(_canonical(request) + b"\n")

    assert controller.prepare(request_path, plan_path) == 2

    assert "wave7_sequence." in capsys.readouterr().err
    assert not plan_path.exists()
    assert not paths["marker"].exists()
    assert all(not Path(path).exists() for path in request["run_roots"].values())


def test_real_producer_shaped_request_v6_is_accepted_with_module_commands(
    controller: ModuleType,
    request_producer: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path, plan_path, paths = _request_fixture(controller, tmp_path)
    fixture_request = paths["request"]
    bindings_by_path = {
        binding["path"]: binding
        for binding in (
            fixture_request["legacy_r4_failure"],
            fixture_request["determinism_preflight_plan"],
            fixture_request["determinism_preflight"],
            fixture_request["runtime_receipt"],
        )
    }

    def bind_external(path: str, **_: Any) -> dict[str, Any]:
        return copy.deepcopy(bindings_by_path[str(Path(path).resolve())])

    monkeypatch.setattr(request_producer, "_bind_external_receipt", bind_external)
    monkeypatch.setattr(
        request_producer,
        "_bind_predecessor_preflight_failure",
        lambda *_: copy.deepcopy(fixture_request["predecessor_preflight_failure"]),
    )
    monkeypatch.setattr(
        request_producer,
        "_source_inventory",
        lambda: copy.deepcopy(fixture_request["source_inventory"]),
    )
    monkeypatch.setattr(
        request_producer,
        "R7_CONFIG_PATHS",
        {
            role: Path(fixture_request["config_paths"][role])
            for role in controller.RUN_ROLES
        },
    )
    monkeypatch.setattr(
        request_producer,
        "R7_RUN_ROOTS",
        {
            role: Path(fixture_request["run_roots"][role])
            for role in controller.RUN_ROLES
        },
    )
    monkeypatch.setattr(
        request_producer,
        "R7_TARGETS",
        {name: Path(path) for name, path in fixture_request["targets"].items()},
    )
    monkeypatch.setattr(
        request_producer,
        "R7_CACHE_ROOT",
        Path(fixture_request["environment"]["COORDEXP_SWIFT_PACK_CACHE_ROOT"]),
    )
    monkeypatch.setattr(
        request_producer,
        "R7_CACHE_PREPARATION_RECEIPT",
        paths["cache_preparation_receipt"],
    )
    monkeypatch.setattr(
        request_producer,
        "R7_RUNTIME_RECEIPT",
        Path(fixture_request["runtime_receipt"]["path"]),
    )
    monkeypatch.setattr(
        request_producer,
        "R7_DETERMINISM_PREFLIGHT_PLAN",
        Path(fixture_request["determinism_preflight_plan"]["path"]),
    )
    monkeypatch.setattr(
        request_producer,
        "R7_DETERMINISM_PREFLIGHT_RECEIPT",
        Path(fixture_request["determinism_preflight"]["path"]),
    )
    args = argparse.Namespace(
        **{
            f"{role}_config": fixture_request["config_paths"][role]
            for role in controller.RUN_ROLES
        },
        **{
            f"{role}_run_root": fixture_request["run_roots"][role]
            for role in controller.RUN_ROLES
        },
        **fixture_request["targets"],
        legacy_r4_failure=fixture_request["legacy_r4_failure"]["path"],
        predecessor_r5_failure=fixture_request["predecessor_sequence_failure"]["path"],
        predecessor_r6_preflight_plan=fixture_request["predecessor_preflight_failure"][
            "plan"
        ]["path"],
        predecessor_r6_preflight_marker=fixture_request[
            "predecessor_preflight_failure"
        ]["attempt_marker"]["path"],
        predecessor_r6_preflight_terminal=fixture_request[
            "predecessor_preflight_failure"
        ]["terminal_receipt"]["path"],
        determinism_preflight_plan=fixture_request["determinism_preflight_plan"][
            "path"
        ],
        determinism_preflight_receipt=fixture_request["determinism_preflight"]["path"],
        runtime_receipt=fixture_request["runtime_receipt"]["path"],
        cache_preparation_receipt=str(paths["cache_preparation_receipt"]),
        cache_root=fixture_request["environment"]["COORDEXP_SWIFT_PACK_CACHE_ROOT"],
        phase_timeout_seconds=5.0,
        term_grace_seconds=0.25,
        kill_grace_seconds=0.25,
        gpu_baseline_stability_seconds=2.0,
        gpu_post_cleanup_stability_seconds=2.0,
        max_total_wall_seconds=30.0,
        max_total_gpu_device_seconds=120.0,
    )
    amendment_path = Path(fixture_request["amendment"]["path"])
    cache_path = Path(fixture_request["cache_input_attestation"]["path"])
    model_path = Path(fixture_request["model_input_attestation"]["path"])
    produced = request_producer._request_payload(
        args,
        amendment=json.loads(amendment_path.read_text(encoding="utf-8")),
        cache=json.loads(cache_path.read_text(encoding="utf-8")),
        model=json.loads(model_path.read_text(encoding="utf-8")),
        leaf_paths={
            "amendment": amendment_path,
            "cache": cache_path,
            "model": model_path,
        },
    )

    assert produced["commands"] == fixture_request["commands"]
    assert produced["commands"]["uninterrupted"][-4:] == [
        "--module",
        "src.train",
        "--config",
        fixture_request["config_paths"]["uninterrupted"],
    ]
    request_path.write_bytes(_canonical(produced) + b"\n")

    assert controller.prepare(request_path, plan_path) == 0
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert (
        plan["predecessor_sequence_failure"] == produced["predecessor_sequence_failure"]
    )
    assert (
        plan["predecessor_preflight_failure"]
        == produced["predecessor_preflight_failure"]
    )


def test_prepare_and_run_execute_exact_sequence_once(
    controller: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path, plan_path, paths = _request_fixture(controller, tmp_path)
    _set_required_environment(monkeypatch, paths["request"])
    baseline_marker_states: list[bool] = []
    subset_marker_states: list[bool] = []
    recovery_marker_states: list[bool] = []
    baseline = _shared_gpu_baseline(controller)

    def attest_baseline(**_: Any) -> dict[str, Any]:
        baseline_marker_states.append(paths["marker"].exists())
        return copy.deepcopy(baseline)

    def attest_subset(**_: Any) -> dict[str, Any]:
        subset_marker_states.append(paths["marker"].exists())
        return _shared_gpu_subset_sample(controller)

    def attest_recovery(**kwargs: Any) -> dict[str, Any]:
        recovery_marker_states.append(paths["marker"].exists())
        return _shared_gpu_recovery(controller, phase=kwargs["phase"])

    monkeypatch.setattr(controller, "_attest_shared_gpu_baseline", attest_baseline)
    monkeypatch.setattr(controller, "_assert_shared_gpu_subset", attest_subset)
    monkeypatch.setattr(controller, "_attest_shared_gpu_recovery", attest_recovery)

    assert controller.prepare(request_path, plan_path) == 0
    original_plan = plan_path.read_bytes()
    assert controller.prepare(request_path, plan_path) == 2
    assert plan_path.read_bytes() == original_plan
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["schema"] == controller.PLAN_SCHEMA
    assert plan["request"]["file_sha256"] == _sha256(request_path.read_bytes())
    assert (
        plan["request"]["payload_sha256"] == paths["request"]["request_payload_sha256"]
    )
    assert (
        plan["predecessor_sequence_failure"]
        == paths["request"]["predecessor_sequence_failure"]
    )
    assert plan["controller_source"]["sha256"] == _sha256(SCRIPT.read_bytes())
    assert (
        plan["input_attestations"]["cache"]["payload"]["attestation_sha256"]
        == (paths["request"]["cache_input_attestation"]["payload_sha256"])
    )
    assert (
        plan["input_attestations"]["model"]["payload"]["attestation_sha256"]
        == (paths["request"]["model_input_attestation"]["payload_sha256"])
    )
    assert plan["input_attestations"]["validation"] == {
        "status": "passed",
        "cache_attestation_sha256": paths["request"]["cache_input_attestation"][
            "payload_sha256"
        ],
        "model_attestation_sha256": paths["request"]["model_input_attestation"][
            "payload_sha256"
        ],
        "cache_payloads_validated": True,
        "model_weights_rehashed": True,
        "measured_cache_payload_bytes": plan["input_attestations"]["cache"]["payload"][
            "measured_payload_bytes"
        ],
        "materialization_policy": {
            split: {"strategy": "fork_process_pool", "workers": 2}
            for split in ("train", "eval.forward")
        },
        "weight_hash_execution_policy": plan["input_attestations"]["model"]["payload"][
            "weight_hash_execution_policy"
        ],
    }
    assert plan["policy"]["gpu_admission_mode"] == "shared_preexisting_subset_v1"
    assert plan["policy"]["gpu_memory_total_mib"] == 81_920
    assert plan["policy"]["gpu_memory_used_ceiling_mib"] == 49_152
    assert plan["policy"]["gpu_memory_headroom_floor_mib"] == 32_768
    assert plan["policy"]["gpu_post_cleanup_stability_seconds"] == 2.0
    semantic_projections = []
    for role in controller.RUN_ROLES:
        resolved = plan["resolved_configs"][role]
        run = resolved["config_projection"]["run"]
        resume = resolved["config_projection"]["resume"]
        assert run == {
            "name": role,
            "artifact_root": str((tmp_path / "runs").resolve()),
            "output_dir": Path(paths["request"]["run_roots"][role]).name,
            "collision_policy": "fail",
        }
        assert resume["mode"] == "exact_same_world_size"
        assert resume["checkpoint_dir"] == (
            str(
                Path(paths["request"]["run_roots"]["interrupted_parent"])
                / "checkpoints"
                / "step-3"
            )
            if role == "resume_child"
            else None
        )
        assert set(resolved["config_projection"]) - set(
            resolved["semantic_projection"]
        ) == {"run", "resume"}
        assert set(resolved["semantic_projection"]) == set(
            resolved["config_projection"]
        ) - {"run", "resume"}
        semantic_projections.append(resolved["semantic_projection"])
    assert semantic_projections[1:] == semantic_projections[:1] * 2

    original_revalidate = controller._revalidate_plan_inputs
    revalidated: list[str] = []

    def counted_revalidate(*args: Any, **kwargs: Any) -> None:
        revalidated.append("called")
        original_revalidate(*args, **kwargs)

    monkeypatch.setattr(controller, "_revalidate_plan_inputs", counted_revalidate)

    assert controller.execute(plan_path) == 0

    assert paths["log"].read_text(encoding="utf-8").splitlines() == list(
        controller.PHASE_ORDER
    )
    marker = json.loads(paths["marker"].read_text(encoding="utf-8"))
    terminal = json.loads(paths["terminal"].read_text(encoding="utf-8"))
    assert marker["schema"] == controller.MARKER_SCHEMA
    assert marker["gpu_admission"] == baseline
    assert marker["claim_scope"] == controller.SEQUENCE_CLAIM_SCOPE
    assert marker["identities"]["input_attestations"] == plan["input_attestations"]
    assert terminal["schema"] == controller.RECEIPT_SCHEMA
    assert terminal["status"] == "passed"
    assert terminal["failure"] is None
    assert [item["status"] for item in terminal["phase_records"]] == ["passed"] * len(
        controller.PHASE_ORDER
    )
    assert terminal["pre_child_receipt"]["validated"] is True
    assert terminal["final_receipt"]["validated"] is True
    assert terminal["cost"]["gpu_device_count"] == 8
    assert terminal["cost"]["gpu_device_seconds"] >= 0.0
    assert terminal["claim_scope"] == controller.SEQUENCE_CLAIM_SCOPE
    assert terminal["input_attestations"] == plan["input_attestations"]
    assert terminal["final_gpu_recovery"]["phase"] == "preterminal"
    for phase in controller.PHASE_ORDER:
        phase_record = next(
            record for record in terminal["phase_records"] if record["phase"] == phase
        )
        assert phase_record["gpu_after_cleanup"] is not None
        assert phase_record["gpu_after_cleanup"]["phase"] == f"post-{phase}"
        assert (phase_record["gpu_before_launch"] is not None) == (
            phase in controller.GPU_PHASES
        )
    assert not paths["sidecar"].exists()
    assert baseline_marker_states == [False]
    assert subset_marker_states == [True] * 3
    assert recovery_marker_states == [True] * 7
    assert (
        len(revalidated) == 9
    )  # premarker, six phases, pre/post-recovery terminal gates

    assert controller.execute(plan_path) == 2
    assert paths["log"].read_text(encoding="utf-8").splitlines() == list(
        controller.PHASE_ORDER
    )


def test_full_revalidation_precedes_adjacent_gpu_samples_and_phase_launches(
    controller: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path, plan_path, paths = _request_fixture(controller, tmp_path)
    _set_required_environment(monkeypatch, paths["request"])
    original_revalidate = controller._revalidate_plan_inputs
    original_run = paths["simulated_run_phase"]
    events: list[str] = []

    def revalidate(*args: Any, **kwargs: Any) -> None:
        events.append("revalidate")
        original_revalidate(*args, **kwargs)

    def baseline(**_: Any) -> dict[str, Any]:
        events.append("baseline")
        return _shared_gpu_baseline(controller)

    def subset(**_: Any) -> dict[str, Any]:
        events.append("subset")
        return _shared_gpu_subset_sample(controller)

    def run(
        phase: str, argv: list[str], **kwargs: Any
    ) -> tuple[int, float, dict[str, Any]]:
        events.append(f"run:{phase}")
        return original_run(phase, argv, **kwargs)

    monkeypatch.setattr(controller, "_revalidate_plan_inputs", revalidate)
    monkeypatch.setattr(controller, "_attest_shared_gpu_baseline", baseline)
    monkeypatch.setattr(controller, "_assert_shared_gpu_subset", subset)
    monkeypatch.setattr(
        controller,
        "_attest_shared_gpu_recovery",
        lambda **kwargs: _shared_gpu_recovery(controller, phase=kwargs["phase"]),
    )
    monkeypatch.setattr(controller, "_run_phase", run)
    assert controller.prepare(request_path, plan_path) == 0
    assert controller.execute(plan_path) == 0

    assert events.index("revalidate") < events.index("baseline")
    for phase in controller.PHASE_ORDER:
        launch_index = events.index(f"run:{phase}")
        revalidate_index = max(
            index
            for index, event in enumerate(events[:launch_index])
            if event == "revalidate"
        )
        if phase in controller.GPU_PHASES:
            subset_index = max(
                index
                for index, event in enumerate(events[:launch_index])
                if event == "subset"
            )
            assert revalidate_index < subset_index < launch_index
            assert "revalidate" not in events[subset_index + 1 : launch_index]


def test_predecessor_is_live_revalidated_at_every_publication_and_phase_gate(
    controller: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path, plan_path, paths = _request_fixture(controller, tmp_path)
    _set_required_environment(monkeypatch, paths["request"])
    _install_static_shared_gpu_attestation(controller, monkeypatch)
    original_validate = controller._validate_predecessor_sequence_failure
    original_validate_amendment = controller._validate_r7_amendment_binding
    original_publish = controller._publish_terminal_or_sidecar
    publishing_terminal = False
    events: dict[str, list[dict[str, Any]]] = {
        "predecessor": [],
        "amendment": [],
    }

    def phase_count() -> int:
        if not paths["log"].exists():
            return 0
        return len(paths["log"].read_text(encoding="utf-8").splitlines())

    def record(owner: str) -> None:
        events[owner].append(
            dict(
                plan_exists=plan_path.exists(),
                marker_exists=paths["marker"].exists(),
                terminal_exists=paths["terminal"].exists(),
                phase_count=phase_count(),
                publishing_terminal=publishing_terminal,
            )
        )

    def validate(value: dict[str, Any]) -> dict[str, Any]:
        record("predecessor")
        return original_validate(value)

    def validate_amendment(value: dict[str, Any]) -> dict[str, Any]:
        record("amendment")
        return original_validate_amendment(value)

    def publish(*args: Any, **kwargs: Any) -> bool:
        nonlocal publishing_terminal
        publishing_terminal = True
        try:
            return original_publish(*args, **kwargs)
        finally:
            publishing_terminal = False

    monkeypatch.setattr(controller, "_validate_predecessor_sequence_failure", validate)
    monkeypatch.setattr(
        controller, "_validate_r7_amendment_binding", validate_amendment
    )
    monkeypatch.setattr(controller, "_publish_terminal_or_sidecar", publish)

    assert controller.prepare(request_path, plan_path) == 0
    assert all(
        any(not event["plan_exists"] for event in owner_events)
        for owner_events in events.values()
    )
    for owner_events in events.values():
        owner_events.clear()

    assert controller.execute(plan_path) == 0

    for owner, owner_events in events.items():
        assert any(not event["marker_exists"] for event in owner_events), owner
        for completed_phases in range(len(controller.PHASE_ORDER)):
            assert any(
                event["marker_exists"] and event["phase_count"] == completed_phases
                for event in owner_events
            ), (owner, completed_phases)
        assert any(
            event["publishing_terminal"]
            and event["phase_count"] == len(controller.PHASE_ORDER)
            and not event["terminal_exists"]
            for event in owner_events
        ), owner


def test_gpu_drift_during_premarker_revalidation_is_sampled_before_marker(
    controller: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path, plan_path, paths = _request_fixture(controller, tmp_path)
    _set_required_environment(monkeypatch, paths["request"])
    original_revalidate = controller._revalidate_plan_inputs
    drifted = False

    def revalidate(*args: Any, **kwargs: Any) -> None:
        nonlocal drifted
        original_revalidate(*args, **kwargs)
        if not paths["marker"].exists():
            drifted = True

    def baseline(**_: Any) -> dict[str, Any]:
        if drifted:
            raise controller.Wave7SequenceError(
                "GPU drift appeared during input revalidation",
                code="wave7_sequence.gpu_baseline",
            )
        return _shared_gpu_baseline(controller)

    monkeypatch.setattr(controller, "_revalidate_plan_inputs", revalidate)
    monkeypatch.setattr(controller, "_attest_shared_gpu_baseline", baseline)
    _install_static_shared_gpu_attestation(controller, monkeypatch)
    monkeypatch.setattr(controller, "_attest_shared_gpu_baseline", baseline)
    assert controller.prepare(request_path, plan_path) == 0

    assert controller.execute(plan_path) == 2
    assert not paths["marker"].exists()
    assert not paths["log"].exists()


def test_passed_terminal_validator_independently_rejects_incomplete_evidence(
    controller: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path, plan_path, paths = _request_fixture(controller, tmp_path)
    _set_required_environment(monkeypatch, paths["request"])
    _install_static_shared_gpu_attestation(controller, monkeypatch)
    assert controller.prepare(request_path, plan_path) == 0
    assert controller.execute(plan_path) == 0
    terminal = json.loads(paths["terminal"].read_text(encoding="utf-8"))
    terminal.pop("receipt_payload_sha256")

    mutations: list[dict[str, Any]] = []
    attempt = copy.deepcopy(terminal)
    attempt["phase_records"][0]["attempt_count"] = 0
    mutations.append(attempt)
    survivor = copy.deepcopy(terminal)
    survivor["phase_records"][0]["process"]["reaped"] = False
    mutations.append(survivor)
    recovery = copy.deepcopy(terminal)
    recovery["phase_records"][0]["gpu_after_cleanup"]["samples"] = []
    mutations.append(recovery)
    cost = copy.deepcopy(terminal)
    cost["cost"]["wall_seconds"] = cost["cost"]["wall_seconds_cap"] + 1
    mutations.append(cost)
    comparison = copy.deepcopy(terminal)
    comparison["pre_child_receipt"]["validated"] = False
    mutations.append(comparison)
    expected_gpu = terminal["cost"]["gpu_device_seconds"]
    for mutated_gpu in (0.0, expected_gpu / 2, expected_gpu + 1.0, float("nan")):
        gpu_cost = copy.deepcopy(terminal)
        gpu_cost["cost"]["gpu_device_seconds"] = mutated_gpu
        mutations.append(gpu_cost)
    cap_drift = copy.deepcopy(terminal)
    cap_drift["cost"]["gpu_device_seconds_cap"] += 1.0
    mutations.append(cap_drift)
    wall_below_sum = copy.deepcopy(terminal)
    wall_below_sum["cost"]["wall_seconds"] = 0.0
    mutations.append(wall_below_sum)

    for mutation in mutations:
        with pytest.raises(controller.Wave7SequenceError) as exc_info:
            controller._validate_terminal_payload(mutation)
        assert exc_info.value.code == "wave7_sequence.terminal_schema"


@pytest.mark.parametrize(
    ("mutation", "value"),
    (
        ("missing_stdout_tail", None),
        ("extra_field", None),
        ("stdout_tail", "x" * (64 * 1024 + 1)),
        ("stdout_truncated", "false"),
        ("stderr_tail", 7),
        ("stderr_truncated", 0),
    ),
)
def test_terminal_authenticates_exact_bounded_process_diagnostics(
    controller: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    value: Any,
) -> None:
    request_path, plan_path, paths = _request_fixture(controller, tmp_path)
    _set_required_environment(monkeypatch, paths["request"])
    _install_static_shared_gpu_attestation(controller, monkeypatch)
    assert controller.prepare(request_path, plan_path) == 0
    assert controller.execute(plan_path) == 0
    terminal = json.loads(paths["terminal"].read_text(encoding="utf-8"))
    terminal.pop("receipt_payload_sha256")
    first_process = terminal["phase_records"][0]["process"]
    first_cleanup = terminal["bounded_cleanup"]["records"][0]
    diagnostic_fields = {
        "stdout_tail",
        "stdout_truncated",
        "stderr_tail",
        "stderr_truncated",
    }
    assert diagnostic_fields.issubset(first_process)
    assert {name: first_cleanup[name] for name in diagnostic_fields} == {
        name: first_process[name] for name in diagnostic_fields
    }

    if mutation == "missing_stdout_tail":
        first_process.pop("stdout_tail")
        first_cleanup.pop("stdout_tail")
    elif mutation == "extra_field":
        first_process["diagnostic_extra"] = True
        first_cleanup["diagnostic_extra"] = True
    else:
        first_process[mutation] = value
        first_cleanup[mutation] = value

    with pytest.raises(controller.Wave7SequenceError) as exc_info:
        controller._validate_terminal_payload(terminal)
    assert exc_info.value.code == "wave7_sequence.terminal_schema"


def test_terminal_rejects_cleanup_graph_and_signal_event_row_mutations(
    controller: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path, plan_path, paths = _request_fixture(controller, tmp_path)
    _set_required_environment(monkeypatch, paths["request"])
    _install_static_shared_gpu_attestation(controller, monkeypatch)
    assert controller.prepare(request_path, plan_path) == 0
    assert controller.execute(plan_path) == 0
    terminal = json.loads(paths["terminal"].read_text(encoding="utf-8"))
    terminal.pop("receipt_payload_sha256")
    process = terminal["phase_records"][0]["process"]
    graph_row = {
        "pid": 12345,
        "parent_pid": 1,
        "process_group_id": 12345,
        "session_id": 12345,
        "state": "R",
        "start_time_ticks": 999,
        "depth": 0,
    }
    group_event = {
        "kind": "signal_group_sigterm",
        "pgid": 12345,
        "depth": 0,
        "outcome": "sent",
    }
    pid_event = {
        "kind": "signal_pid_sigterm",
        "pid": 12345,
        "depth": 0,
        "outcome": "sent",
    }
    process["captured_process_graph"] = [graph_row]
    process["observed_process_count"] = 1
    process["signal_events"] = [group_event, pid_event]
    terminal["bounded_cleanup"]["records"][0] = {
        "phase": "uninterrupted",
        **copy.deepcopy(process),
    }
    controller._validate_terminal_payload(terminal)

    mutations: dict[str, dict[str, Any]] = {}
    for name in (
        "graph_missing_field",
        "graph_extra_field",
        "graph_bool_pid",
        "graph_duplicate_pid",
        "graph_unsorted",
        "event_missing_field",
        "event_extra_field",
        "event_bad_kind",
        "event_bool_target",
        "event_bad_outcome",
        "event_count_overflow",
    ):
        mutation = copy.deepcopy(terminal)
        mutated_process = mutation["phase_records"][0]["process"]
        if name == "graph_missing_field":
            mutated_process["captured_process_graph"][0].pop("state")
        elif name == "graph_extra_field":
            mutated_process["captured_process_graph"][0]["extra"] = 1
        elif name == "graph_bool_pid":
            mutated_process["captured_process_graph"][0]["pid"] = True
        elif name == "graph_duplicate_pid":
            mutated_process["captured_process_graph"].append(copy.deepcopy(graph_row))
            mutated_process["observed_process_count"] = 2
        elif name == "graph_unsorted":
            earlier = {**graph_row, "pid": 12344, "process_group_id": 12344}
            mutated_process["captured_process_graph"].append(earlier)
            mutated_process["observed_process_count"] = 2
        elif name == "event_missing_field":
            mutated_process["signal_events"][0].pop("outcome")
        elif name == "event_extra_field":
            mutated_process["signal_events"][0]["extra"] = 1
        elif name == "event_bad_kind":
            mutated_process["signal_events"][0]["kind"] = "signal_group_sigint"
        elif name == "event_bool_target":
            mutated_process["signal_events"][0]["pgid"] = True
        elif name == "event_bad_outcome":
            mutated_process["signal_events"][1]["outcome"] = "ignored"
        else:
            mutated_process["signal_events"] = [
                copy.deepcopy(pid_event)
                for _ in range(controller.MAX_PROCESS_GRAPH * 4 + 1)
            ]
        mutation["bounded_cleanup"]["records"][0] = {
            "phase": "uninterrupted",
            **copy.deepcopy(mutated_process),
        }
        mutations[name] = mutation

    for name, mutation in mutations.items():
        with pytest.raises(controller.Wave7SequenceError) as exc_info:
            controller._validate_terminal_payload(mutation)
        assert exc_info.value.code == "wave7_sequence.terminal_schema", name


def test_immutable_r5_v4_terminal_is_not_reinterpreted_or_rewritten(
    controller: ModuleType,
) -> None:
    receipt_path = IMMUTABLE_R5_SEQUENCE_RECEIPT
    before = receipt_path.read_bytes()
    assert _sha256(before) == (
        "d5244fb6d96b24b1cfac4438628ae141496f7357c286dff5454c19c9f39731b1"
    )
    receipt = json.loads(before)
    receipt.pop("receipt_payload_sha256")

    with pytest.raises(controller.Wave7SequenceError) as exc_info:
        controller._validate_terminal_payload(receipt)

    assert exc_info.value.code == "wave7_sequence.terminal_schema"
    assert receipt_path.read_bytes() == before


def test_dedicated_predecessor_reader_accepts_only_the_immutable_r5_failure(
    controller: ModuleType,
) -> None:
    receipt = json.loads(IMMUTABLE_R5_SEQUENCE_RECEIPT.read_text(encoding="utf-8"))
    binding = _binding(IMMUTABLE_R5_SEQUENCE_RECEIPT, receipt)

    observed = controller._validate_predecessor_sequence_failure(binding)

    assert observed == receipt
    assert observed["schema"].endswith("receipt-v4")
    assert observed["status"] == "failed"


@pytest.mark.parametrize("binding_field", ("file_sha256", "payload_sha256"))
def test_dedicated_predecessor_reader_rejects_binding_digest_mutations(
    controller: ModuleType,
    binding_field: str,
) -> None:
    receipt = json.loads(IMMUTABLE_R5_SEQUENCE_RECEIPT.read_text(encoding="utf-8"))
    binding = _binding(IMMUTABLE_R5_SEQUENCE_RECEIPT, receipt)
    binding[binding_field] = "0" * 64

    with pytest.raises(controller.Wave7SequenceError) as exc_info:
        controller._validate_predecessor_sequence_failure(binding)

    assert exc_info.value.code == "wave7_sequence.predecessor_r5"


def test_dedicated_predecessor_reader_rejects_byte_identical_alternate_path(
    controller: ModuleType,
    tmp_path: Path,
) -> None:
    alternate = tmp_path / "copied-r5-receipt.json"
    alternate.write_bytes(IMMUTABLE_R5_SEQUENCE_RECEIPT.read_bytes())
    receipt = json.loads(alternate.read_text(encoding="utf-8"))

    with pytest.raises(controller.Wave7SequenceError) as exc_info:
        controller._validate_predecessor_sequence_failure(_binding(alternate, receipt))

    assert exc_info.value.code == "wave7_sequence.predecessor_r5"


@pytest.mark.parametrize(
    "mutation",
    (
        "schema",
        "status",
        "failure_phase",
        "failure_code",
        "first_return_code",
        "first_attempt_count",
        "later_phase_state",
        "plan_file",
        "plan_payload",
        "marker_file",
        "marker_payload",
        "bounded_cleanup",
        "final_recovery",
        "wall_cost",
        "gpu_cost",
    ),
)
def test_dedicated_predecessor_reader_rejects_resigned_projection_mutations(
    controller: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    payload = json.loads(IMMUTABLE_R5_SEQUENCE_RECEIPT.read_text(encoding="utf-8"))
    payload.pop("receipt_payload_sha256")
    if mutation == "schema":
        payload["schema"] = controller.RECEIPT_SCHEMA
    elif mutation == "status":
        payload["status"] = "passed"
    elif mutation == "failure_phase":
        payload["failure"]["phase"] = "interrupted_parent"
    elif mutation == "failure_code":
        payload["failure"]["code"] = "wave7_sequence.launch"
    elif mutation == "first_return_code":
        payload["phase_records"][0]["return_code"] = 2
    elif mutation == "first_attempt_count":
        payload["phase_records"][0]["attempt_count"] = 0
    elif mutation == "later_phase_state":
        payload["phase_records"][1]["status"] = "failed"
    elif mutation == "plan_file":
        payload["plan"]["file_sha256"] = "0" * 64
    elif mutation == "plan_payload":
        payload["plan"]["payload_sha256"] = "0" * 64
    elif mutation == "marker_file":
        payload["marker"]["file_sha256"] = "0" * 64
    elif mutation == "marker_payload":
        payload["marker"]["payload_sha256"] = "0" * 64
    elif mutation == "bounded_cleanup":
        payload["bounded_cleanup"]["records"][0]["reaped"] = False
    elif mutation == "final_recovery":
        payload["final_gpu_recovery"]["samples"].pop()
    elif mutation == "wall_cost":
        payload["cost"]["wall_seconds"] = 51.0
    else:
        payload["cost"]["gpu_device_seconds"] = 160.0

    mutated_path = tmp_path / f"r5-{mutation}.json"
    signed = _write_signed_json(mutated_path, payload)
    binding = _binding(mutated_path, signed)
    monkeypatch.setattr(
        controller, "FROZEN_R5_FAILURE_FILE_SHA256", binding["file_sha256"]
    )
    monkeypatch.setattr(
        controller, "FROZEN_R5_FAILURE_PAYLOAD_SHA256", binding["payload_sha256"]
    )
    monkeypatch.setattr(controller, "FROZEN_R5_FAILURE_PATH", mutated_path.resolve())

    with pytest.raises(controller.Wave7SequenceError) as exc_info:
        controller._validate_predecessor_sequence_failure(binding)

    assert exc_info.value.code == "wave7_sequence.predecessor_r5", mutation


def test_dedicated_r6_preflight_reader_accepts_only_immutable_failed_chain(
    controller: ModuleType,
) -> None:
    plan_path = IMMUTABLE_R6_PREFLIGHT_ROOT / "plan.json"
    marker_path = IMMUTABLE_R6_PREFLIGHT_ROOT / "attempt-start-marker.json"
    terminal_path = IMMUTABLE_R6_PREFLIGHT_ROOT / "terminal-receipt.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    binding = {
        "historical_non_executable": True,
        "plan": _payload_binding(plan_path, plan, digest_field="plan_payload_sha256"),
        "attempt_marker": _binding(marker_path, marker),
        "terminal_receipt": _binding(terminal_path, terminal),
    }

    observed = controller._validate_predecessor_preflight_failure(binding)

    assert observed == {
        "plan": plan,
        "attempt_marker": marker,
        "terminal_receipt": terminal,
    }
    assert terminal["status"] == "failed"
    assert terminal["mismatches"] == ["KeyboardInterrupt"]
    assert terminal["launch_count"] == 0
    assert terminal["rank_receipts"] == []
    assert terminal["comparisons"] is None


@pytest.mark.parametrize(
    ("owner", "field", "value"),
    (
        ("root", "historical_non_executable", False),
        ("plan", "file_sha256", "0" * 64),
        ("attempt_marker", "payload_sha256", "0" * 64),
        ("terminal_receipt", "status", "passed"),
    ),
)
def test_dedicated_r6_preflight_reader_rejects_substitution(
    controller: ModuleType,
    owner: str,
    field: str,
    value: Any,
) -> None:
    plan_path = IMMUTABLE_R6_PREFLIGHT_ROOT / "plan.json"
    marker_path = IMMUTABLE_R6_PREFLIGHT_ROOT / "attempt-start-marker.json"
    terminal_path = IMMUTABLE_R6_PREFLIGHT_ROOT / "terminal-receipt.json"
    binding = {
        "historical_non_executable": True,
        "plan": _payload_binding(
            plan_path,
            json.loads(plan_path.read_text(encoding="utf-8")),
            digest_field="plan_payload_sha256",
        ),
        "attempt_marker": _binding(
            marker_path, json.loads(marker_path.read_text(encoding="utf-8"))
        ),
        "terminal_receipt": _binding(
            terminal_path, json.loads(terminal_path.read_text(encoding="utf-8"))
        ),
    }
    if owner == "root":
        binding[field] = value
    else:
        binding[owner][field] = value

    with pytest.raises(controller.Wave7SequenceError) as exc_info:
        controller._validate_predecessor_preflight_failure(binding)

    assert exc_info.value.code == "wave7_sequence.predecessor_r6_preflight"


@pytest.mark.parametrize(
    "failure_mode",
    ("setup_runtime_error", "setup_keyboard_interrupt", "finalize_runtime_error"),
)
def test_post_popen_failures_reap_and_publish_exact_fallback_diagnostics(
    controller: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    real_run_phase = controller._run_phase
    request_path, plan_path, paths = _request_fixture(controller, tmp_path)
    _set_required_environment(monkeypatch, paths["request"])
    _install_static_shared_gpu_attestation(controller, monkeypatch)
    monkeypatch.setattr(controller, "_run_phase", real_run_phase)
    original_popen = subprocess.Popen
    processes: list[subprocess.Popen[bytes]] = []
    program = (
        "import os, time; "
        "os.write(1, b'capture-stdout\\n'); "
        "os.write(2, b'capture-stderr\\n'); "
        + ("time.sleep(30)" if failure_mode.startswith("setup_") else "None")
    )

    def cpu_only_popen(argv: list[str], **kwargs: Any) -> subprocess.Popen[bytes]:
        del argv
        process = original_popen([sys.executable, "-c", program], **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(controller.subprocess, "Popen", cpu_only_popen)
    if failure_mode.startswith("setup_"):

        def fail_setup(process: subprocess.Popen[bytes]) -> None:
            assert process.poll() is None
            if failure_mode == "setup_keyboard_interrupt":
                raise KeyboardInterrupt("injected output setup interrupt")
            raise RuntimeError("injected output setup failure")

        monkeypatch.setattr(controller, "_start_phase_output_capture", fail_setup)
    else:

        def fail_finalize(*args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise RuntimeError("injected output finalization failure")

        monkeypatch.setattr(controller, "_attach_phase_diagnostics", fail_finalize)

    assert controller.prepare(request_path, plan_path) == 0
    try:
        return_code = controller.execute(plan_path)
        survivors = [process.pid for process in processes if process.poll() is None]
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5.0)

    assert return_code == 1
    assert survivors == []
    assert paths["terminal"].is_file()
    assert not paths["sidecar"].exists()
    terminal = json.loads(paths["terminal"].read_text(encoding="utf-8"))
    assert terminal["schema"] == controller.RECEIPT_SCHEMA
    assert terminal["status"] == "failed"
    assert terminal["failure"]["phase"] == "uninterrupted"
    assert terminal["failure"]["code"] == "wave7_sequence.output_capture"
    first = terminal["phase_records"][0]
    process_record = first["process"]
    assert process_record is not None
    assert process_record["reaped"] is True
    assert process_record["remaining_pids"] == []
    assert process_record["remaining_pgids"] == []
    assert {
        name: process_record[name]
        for name in (
            "stdout_tail",
            "stdout_truncated",
            "stderr_tail",
            "stderr_truncated",
        )
    } == {
        "stdout_tail": "",
        "stdout_truncated": True,
        "stderr_tail": "",
        "stderr_truncated": True,
    }
    assert terminal["bounded_cleanup"]["records"] == [
        {"phase": "uninterrupted", **process_record}
    ]
    unsigned = dict(terminal)
    unsigned.pop("receipt_payload_sha256")
    controller._validate_terminal_payload(unsigned)

    mutation = copy.deepcopy(unsigned)
    mutation["phase_records"][0]["process"].pop("stdout_tail")
    mutation["bounded_cleanup"]["records"][0].pop("stdout_tail")
    with pytest.raises(controller.Wave7SequenceError) as exc_info:
        controller._validate_terminal_payload(mutation)
    assert exc_info.value.code == "wave7_sequence.terminal_schema"


def test_terminal_validator_reopens_live_comparison_receipt_bindings(
    controller: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path, plan_path, paths = _request_fixture(controller, tmp_path)
    _set_required_environment(monkeypatch, paths["request"])
    _install_static_shared_gpu_attestation(controller, monkeypatch)
    assert controller.prepare(request_path, plan_path) == 0
    assert controller.execute(plan_path) == 0
    terminal = json.loads(paths["terminal"].read_text(encoding="utf-8"))
    terminal.pop("receipt_payload_sha256")

    for binding_name in ("pre_child_receipt", "final_receipt"):
        receipt_path = Path(terminal[binding_name]["path"])
        original_bytes = receipt_path.read_bytes()
        replacement_target = receipt_path.with_name(f"{receipt_path.name}.replacement")
        replacement_target.write_bytes(original_bytes)
        for mutation in ("replacement", "tamper", "missing", "directory", "symlink"):
            if receipt_path.is_symlink() or receipt_path.is_file():
                receipt_path.unlink()
            elif receipt_path.is_dir():
                receipt_path.rmdir()
            receipt_path.write_bytes(original_bytes)
            if mutation == "replacement":
                payload = json.loads(receipt_path.read_text(encoding="utf-8"))
                payload.pop("receipt_payload_sha256")
                payload["replacement_generation"] = 2
                _write_signed_json(receipt_path, payload)
            elif mutation == "tamper":
                receipt_path.write_text("{}\n", encoding="utf-8")
            elif mutation == "missing":
                receipt_path.unlink()
            elif mutation == "directory":
                receipt_path.unlink()
                receipt_path.mkdir()
            else:
                receipt_path.unlink()
                receipt_path.symlink_to(replacement_target)

            with pytest.raises(controller.Wave7SequenceError) as exc_info:
                controller._validate_terminal_payload(terminal)
            assert exc_info.value.code == "wave7_sequence.terminal_schema"

        if receipt_path.is_symlink() or receipt_path.is_file():
            receipt_path.unlink()
        elif receipt_path.is_dir():
            receipt_path.rmdir()
        receipt_path.write_bytes(original_bytes)


def test_terminal_validator_rejects_launched_command_digest_substitutions(
    controller: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path, plan_path, paths = _request_fixture(controller, tmp_path)
    _set_required_environment(monkeypatch, paths["request"])
    _install_static_shared_gpu_attestation(controller, monkeypatch)
    assert controller.prepare(request_path, plan_path) == 0
    assert controller.execute(plan_path) == 0
    terminal = json.loads(paths["terminal"].read_text(encoding="utf-8"))
    terminal.pop("receipt_payload_sha256")

    mutations: list[dict[str, Any]] = []
    arbitrary = copy.deepcopy(terminal)
    arbitrary["phase_records"][0]["launched_command_sha256"] = "f" * 64
    mutations.append(arbitrary)
    wrong_argv = copy.deepcopy(terminal)
    wrong_argv["phase_records"][1]["launched_command_sha256"] = _sha256(
        _canonical(["wrong", "argv"])
    )
    mutations.append(wrong_argv)
    swapped = copy.deepcopy(terminal)
    (
        swapped["phase_records"][0]["launched_command_sha256"],
        swapped["phase_records"][1]["launched_command_sha256"],
    ) = (
        swapped["phase_records"][1]["launched_command_sha256"],
        swapped["phase_records"][0]["launched_command_sha256"],
    )
    mutations.append(swapped)

    for mutation in mutations:
        with pytest.raises(controller.Wave7SequenceError) as exc_info:
            controller._validate_terminal_payload(mutation)
        assert exc_info.value.code == "wave7_sequence.terminal_schema"


def test_terminal_publication_rechecks_live_final_receipt_binding(
    controller: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path, plan_path, paths = _request_fixture(controller, tmp_path)
    _set_required_environment(monkeypatch, paths["request"])
    _install_static_shared_gpu_attestation(controller, monkeypatch)
    assert controller.prepare(request_path, plan_path) == 0
    original_publish = controller._publish_terminal_or_sidecar

    def mutate_before_publication(
        terminal_path: Path,
        sidecar_path: Path,
        payload: dict[str, Any],
        *,
        marker_identity: dict[str, Any],
    ) -> bool:
        if payload["status"] == "passed":
            final_path = Path(payload["final_receipt"]["path"])
            replacement = json.loads(final_path.read_text(encoding="utf-8"))
            replacement.pop("receipt_payload_sha256")
            replacement["replacement_generation"] = 2
            _write_signed_json(final_path, replacement)
        return original_publish(
            terminal_path,
            sidecar_path,
            payload,
            marker_identity=marker_identity,
        )

    monkeypatch.setattr(
        controller, "_publish_terminal_or_sidecar", mutate_before_publication
    )

    assert controller.execute(plan_path) == 2
    assert not paths["terminal"].exists()
    assert paths["sidecar"].is_file()


def test_post_recovery_live_receipt_drift_marks_preterminal_before_publication(
    controller: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path, plan_path, paths = _request_fixture(controller, tmp_path)
    _set_required_environment(monkeypatch, paths["request"])
    _install_static_shared_gpu_attestation(controller, monkeypatch)
    original_recovery = controller._attest_shared_gpu_recovery
    original_publish = controller._publish_terminal_or_sidecar
    published_payloads: list[dict[str, Any]] = []

    def recover(**kwargs: Any) -> dict[str, Any]:
        result = original_recovery(**kwargs)
        if kwargs["phase"] == "preterminal":
            final_path = Path(paths["targets"]["final_receipt"])
            replacement = json.loads(final_path.read_text(encoding="utf-8"))
            replacement.pop("receipt_payload_sha256")
            replacement["replacement_generation"] = 2
            _write_signed_json(final_path, replacement)
        return result

    def capture_publication(
        terminal_path: Path,
        sidecar_path: Path,
        payload: dict[str, Any],
        *,
        marker_identity: dict[str, Any],
    ) -> bool:
        published_payloads.append(copy.deepcopy(payload))
        return original_publish(
            terminal_path,
            sidecar_path,
            payload,
            marker_identity=marker_identity,
        )

    monkeypatch.setattr(controller, "_attest_shared_gpu_recovery", recover)
    monkeypatch.setattr(controller, "_publish_terminal_or_sidecar", capture_publication)
    assert controller.prepare(request_path, plan_path) == 0

    assert controller.execute(plan_path) == 2
    assert published_payloads[0]["status"] == "failed"
    assert published_payloads[0]["failure"]["phase"] == "preterminal"
    assert not paths["terminal"].exists()
    assert paths["sidecar"].is_file()


def test_failed_terminal_rechecks_nonnull_live_pre_child_binding(
    controller: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path, plan_path, paths = _request_fixture(controller, tmp_path)
    _set_required_environment(monkeypatch, paths["request"])
    _install_static_shared_gpu_attestation(controller, monkeypatch)
    simulated = paths["simulated_run_phase"]

    def fail_resume_child(
        phase: str, argv: list[str], **kwargs: Any
    ) -> tuple[int, float, dict[str, Any]]:
        if phase != "resume_child":
            return simulated(phase, argv, **kwargs)
        return (
            23,
            0.001,
            _fake_process_record(),
        )

    monkeypatch.setattr(controller, "_run_phase", fail_resume_child)
    assert controller.prepare(request_path, plan_path) == 0
    assert controller.execute(plan_path) == 1
    terminal = json.loads(paths["terminal"].read_text(encoding="utf-8"))
    terminal.pop("receipt_payload_sha256")
    assert terminal["status"] == "failed"
    assert terminal["pre_child_receipt"] is not None
    pre_child_path = Path(terminal["pre_child_receipt"]["path"])
    replacement = json.loads(pre_child_path.read_text(encoding="utf-8"))
    replacement.pop("receipt_payload_sha256")
    replacement["replacement_generation"] = 2
    _write_signed_json(pre_child_path, replacement)

    with pytest.raises(controller.Wave7SequenceError) as exc_info:
        controller._validate_terminal_payload(terminal)
    assert exc_info.value.code == "wave7_sequence.terminal_schema"


@pytest.mark.parametrize(
    "mutation",
    (
        "null_failure",
        "mismatched_failure_phase",
        "mismatched_failure_code",
        "all_phases_passed",
        "multiple_failed_phases",
        "malformed_later_phase",
        "cleanup_inconsistency",
        "recovery_inconsistency",
    ),
)
def test_failed_terminal_validator_rejects_state_machine_mutations(
    controller: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    request_path, plan_path, paths = _request_fixture(controller, tmp_path)
    _set_required_environment(monkeypatch, paths["request"])
    _install_static_shared_gpu_attestation(controller, monkeypatch)
    simulated = paths["simulated_run_phase"]

    def fail_pre_child(
        phase: str, argv: list[str], **kwargs: Any
    ) -> tuple[int, float, dict[str, Any]]:
        return (
            (
                23,
                0.001,
                _fake_process_record(),
            )
            if phase == "pre_child"
            else simulated(phase, argv, **kwargs)
        )

    monkeypatch.setattr(controller, "_run_phase", fail_pre_child)
    assert controller.prepare(request_path, plan_path) == 0
    assert controller.execute(plan_path) == 1
    failed = json.loads(paths["terminal"].read_text(encoding="utf-8"))
    failed.pop("receipt_payload_sha256")
    controller._validate_terminal_payload(failed)

    mutated = copy.deepcopy(failed)
    if mutation == "null_failure":
        mutated["failure"] = None
    elif mutation == "mismatched_failure_phase":
        mutated["failure"]["phase"] = "resume_child"
    elif mutation == "mismatched_failure_code":
        mutated["failure"]["code"] = "wave7_sequence.timeout"
    elif mutation == "all_phases_passed":
        passed_root = tmp_path / "passed"
        passed_root.mkdir()
        passed_request, passed_plan, passed_paths = _request_fixture(
            controller, passed_root
        )
        _set_required_environment(monkeypatch, passed_paths["request"])
        _install_static_shared_gpu_attestation(controller, monkeypatch)
        assert controller.prepare(passed_request, passed_plan) == 0
        assert controller.execute(passed_plan) == 0
        mutated = json.loads(passed_paths["terminal"].read_text(encoding="utf-8"))
        mutated.pop("receipt_payload_sha256")
        mutated["status"] = "failed"
        mutated["failure"] = copy.deepcopy(failed["failure"])
    elif mutation == "multiple_failed_phases":
        mutated["phase_records"][3]["status"] = "failed"
    elif mutation == "malformed_later_phase":
        mutated["phase_records"][3]["started_at"] = "unexpected"
    elif mutation == "cleanup_inconsistency":
        mutated["bounded_cleanup"]["records"].pop()
    else:
        mutated["final_gpu_recovery"] = {"phase": "preterminal"}

    with pytest.raises(controller.Wave7SequenceError) as exc_info:
        controller._validate_terminal_payload(mutated)
    assert exc_info.value.code == "wave7_sequence.terminal_schema"


def test_failed_phase_still_records_decision_bearing_preterminal_recovery_failure(
    controller: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path, plan_path, paths = _request_fixture(controller, tmp_path)
    _set_required_environment(monkeypatch, paths["request"])
    baseline = _shared_gpu_baseline(controller)
    monkeypatch.setattr(
        controller, "_attest_shared_gpu_baseline", lambda **_: copy.deepcopy(baseline)
    )
    monkeypatch.setattr(
        controller,
        "_assert_shared_gpu_subset",
        lambda **_: _shared_gpu_subset_sample(controller),
    )
    recovery_phases: list[str] = []

    def recover(**kwargs: Any) -> dict[str, Any]:
        phase = kwargs["phase"]
        recovery_phases.append(phase)
        if phase == "preterminal":
            raise controller.Wave7SequenceError(
                "new row during failed preterminal sampling",
                code="wave7_sequence.gpu_added_process",
            )
        return _shared_gpu_recovery(controller, phase=phase)

    monkeypatch.setattr(controller, "_attest_shared_gpu_recovery", recover)
    simulated = paths["simulated_run_phase"]

    def fail_pre_child(
        phase: str, argv: list[str], **kwargs: Any
    ) -> tuple[int, float, dict[str, Any]]:
        if phase != "pre_child":
            return simulated(phase, argv, **kwargs)
        return (
            23,
            0.001,
            _fake_process_record(),
        )

    monkeypatch.setattr(controller, "_run_phase", fail_pre_child)
    assert controller.prepare(request_path, plan_path) == 0

    assert controller.execute(plan_path) == 1

    terminal = json.loads(paths["terminal"].read_text(encoding="utf-8"))
    assert terminal["status"] == "failed"
    assert terminal["failure"]["phase"] == "pre_child"
    assert terminal["failure"]["code"] == "wave7_sequence.phase_failed"
    recovery = terminal["final_gpu_recovery"]
    assert recovery["phase"] == "preterminal"
    assert recovery["status"] == "failed"
    assert recovery["failure"]["phase"] == "preterminal"
    assert recovery["failure"]["code"] == "wave7_sequence.gpu_added_process"
    assert recovery_phases[-1] == "preterminal"
    assert not paths["sidecar"].exists()


@pytest.mark.parametrize(
    ("primary_mode", "expected_code", "expected_return_code"),
    (
        ("nonzero", "wave7_sequence.phase_failed", 17),
        ("exception", "wave7_sequence.timeout", None),
    ),
)
def test_phase_recovery_failure_is_secondary_to_primary_phase_failure(
    controller: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    primary_mode: str,
    expected_code: str,
    expected_return_code: int | None,
) -> None:
    request_path, plan_path, paths = _request_fixture(controller, tmp_path)
    _set_required_environment(monkeypatch, paths["request"])
    _install_static_shared_gpu_attestation(controller, monkeypatch)

    def recover(**kwargs: Any) -> dict[str, Any]:
        if kwargs["phase"] == "post-uninterrupted":
            raise controller.Wave7SequenceError(
                "injected post-cleanup recovery failure",
                code="wave7_sequence.gpu_recovery",
            )
        return _shared_gpu_recovery(controller, phase=kwargs["phase"])

    def fail_first_phase(
        phase: str, argv: list[str], **kwargs: Any
    ) -> tuple[int, float, dict[str, Any]]:
        del argv, kwargs
        assert phase == "uninterrupted"
        if primary_mode == "exception":
            raise controller.Wave7SequenceError(
                "injected primary timeout",
                code="wave7_sequence.timeout",
                context={
                    "duration_seconds": 0.001,
                    "cleanup": _fake_process_record(),
                },
            )
        return 17, 0.001, _fake_process_record()

    monkeypatch.setattr(controller, "_attest_shared_gpu_recovery", recover)
    monkeypatch.setattr(controller, "_run_phase", fail_first_phase)
    assert controller.prepare(request_path, plan_path) == 0

    assert controller.execute(plan_path) == 1

    assert paths["terminal"].is_file()
    assert not paths["sidecar"].exists()
    terminal = json.loads(paths["terminal"].read_text(encoding="utf-8"))
    assert terminal["schema"] == controller.RECEIPT_SCHEMA
    assert terminal["status"] == "failed"
    assert terminal["failure"]["phase"] == "uninterrupted"
    assert terminal["failure"]["code"] == expected_code
    first = terminal["phase_records"][0]
    assert first["status"] == "failed"
    assert first["return_code"] == expected_return_code
    secondary = first["gpu_after_cleanup"]
    assert secondary["phase"] == "post-uninterrupted"
    assert secondary["status"] == "failed"
    assert secondary["failure"]["phase"] == "post-uninterrupted"
    assert secondary["failure"]["code"] == "wave7_sequence.gpu_recovery"
    assert [record["status"] for record in terminal["phase_records"][1:]] == [
        "not_started"
    ] * 5
    unsigned = dict(terminal)
    unsigned.pop("receipt_payload_sha256")
    controller._validate_terminal_payload(unsigned)

    mutation = copy.deepcopy(unsigned)
    mutation["phase_records"][0]["gpu_after_cleanup"]["failure"]["phase"] = (
        "preterminal"
    )
    with pytest.raises(controller.Wave7SequenceError) as exc_info:
        controller._validate_terminal_payload(mutation)
    assert exc_info.value.code == "wave7_sequence.terminal_schema"


@pytest.mark.parametrize(
    ("reject_position", "expected_log"),
    (
        ("before", []),
        ("after", ["uninterrupted"]),
    ),
)
def test_new_gpu_row_before_or_after_gpu_phase_stops_sequence(
    controller: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reject_position: str,
    expected_log: list[str],
) -> None:
    request_path, plan_path, paths = _request_fixture(controller, tmp_path)
    _set_required_environment(monkeypatch, paths["request"])
    monkeypatch.setattr(
        controller,
        "_attest_shared_gpu_baseline",
        lambda **_: _shared_gpu_baseline(controller),
    )

    def subset_gate(**_: Any) -> dict[str, Any]:
        if reject_position == "before":
            raise controller.Wave7SequenceError(
                "new compute row",
                code="wave7_sequence.gpu_added_process",
            )
        return _shared_gpu_subset_sample(controller)

    def recovery_gate(**kwargs: Any) -> dict[str, Any]:
        if reject_position == "after":
            raise controller.Wave7SequenceError(
                "new compute row",
                code="wave7_sequence.gpu_added_process",
            )
        return _shared_gpu_recovery(controller, phase=kwargs["phase"])

    monkeypatch.setattr(controller, "_assert_shared_gpu_subset", subset_gate)
    monkeypatch.setattr(controller, "_attest_shared_gpu_recovery", recovery_gate)
    assert controller.prepare(request_path, plan_path) == 0

    assert controller.execute(plan_path) == 1

    observed_log = (
        paths["log"].read_text(encoding="utf-8").splitlines()
        if paths["log"].exists()
        else []
    )
    assert observed_log == expected_log
    terminal = json.loads(paths["terminal"].read_text(encoding="utf-8"))
    assert terminal["status"] == "failed"
    assert terminal["failure"]["phase"] == "uninterrupted"
    assert terminal["failure"]["code"] == "wave7_sequence.gpu_added_process"


def test_marker_schema_substitution_is_rejected_before_phase_launch(
    controller: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path, plan_path, paths = _request_fixture(controller, tmp_path)
    _set_required_environment(monkeypatch, paths["request"])
    _install_static_shared_gpu_attestation(controller, monkeypatch)
    assert controller.prepare(request_path, plan_path) == 0
    original_write = controller.write_strict_json_atomic

    def substitute_marker(path: Path, payload: dict[str, Any], **kwargs: Any) -> None:
        original_write(path, payload, **kwargs)
        if path == paths["marker"]:
            unsigned = dict(payload)
            unsigned.pop("receipt_payload_sha256")
            unsigned["schema"] = controller.RECEIPT_SCHEMA
            _write_signed_json(path, unsigned)

    monkeypatch.setattr(controller, "write_strict_json_atomic", substitute_marker)

    assert controller.execute(plan_path) == 1

    assert not paths["log"].exists()
    terminal = json.loads(paths["terminal"].read_text(encoding="utf-8"))
    assert terminal["status"] == "failed"
    assert terminal["failure"]["code"] == "wave7_sequence.marker_schema"


def test_terminal_schema_substitution_uses_failure_sidecar(
    controller: ModuleType,
    tmp_path: Path,
) -> None:
    request_path, plan_path, paths = _request_fixture(controller, tmp_path)
    assert controller.prepare(request_path, plan_path) == 0
    plan, plan_identity = controller._load_plan(plan_path)
    marker_identity = {
        "path": str(paths["marker"]),
        "file_sha256": "a" * 64,
        "payload_sha256": "b" * 64,
        "schema": controller.MARKER_SCHEMA,
    }
    terminal = controller._emergency_terminal(
        plan,
        plan_identity,
        marker_identity,
        controller.Wave7SequenceError("injected", code="wave7_sequence.test"),
    )
    terminal["schema"] = controller.MARKER_SCHEMA

    assert not controller._publish_terminal_or_sidecar(
        paths["terminal"],
        paths["sidecar"],
        terminal,
        marker_identity=marker_identity,
    )

    assert not paths["terminal"].exists()
    sidecar = json.loads(paths["sidecar"].read_text(encoding="utf-8"))
    assert sidecar["schema"] == controller.PUBLICATION_FAILURE_SCHEMA
    assert "terminal receipt schema" in sidecar["error"]


def test_historical_v4_plan_is_non_executable_and_does_not_publish_marker(
    controller: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path, plan_path, paths = _request_fixture(controller, tmp_path)
    _set_required_environment(monkeypatch, paths["request"])
    assert controller.prepare(request_path, plan_path) == 0
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["schema"] = "coordexp-swift-wave7-exact-resume-sequence-plan-v4"
    unsigned = dict(plan)
    unsigned.pop("plan_payload_sha256")
    plan["plan_payload_sha256"] = _sha256(_canonical(unsigned))
    plan_path.write_bytes(_canonical(plan) + b"\n")

    assert controller.execute(plan_path) == 2

    assert not paths["marker"].exists()
    assert not paths["terminal"].exists()


def test_phase_failure_stops_all_later_arms_and_publishes_one_terminal(
    controller: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path, plan_path, paths = _request_fixture(controller, tmp_path)
    request = paths["request"]
    _set_required_environment(monkeypatch, request)
    _install_static_shared_gpu_attestation(controller, monkeypatch)
    simulated = paths["simulated_run_phase"]

    def fail_pre_child(
        phase: str, argv: list[str], **kwargs: Any
    ) -> tuple[int, float, dict[str, Any]]:
        if phase != "pre_child":
            return simulated(phase, argv, **kwargs)
        with paths["log"].open("a", encoding="utf-8") as handle:
            handle.write("fail\n")
        return (
            23,
            0.001,
            _fake_process_record(),
        )

    monkeypatch.setattr(controller, "_run_phase", fail_pre_child)
    assert controller.prepare(request_path, plan_path) == 0

    assert controller.execute(plan_path) == 1

    assert paths["log"].read_text(encoding="utf-8").splitlines() == [
        "uninterrupted",
        "interrupted_parent",
        "fail",
    ]
    terminal = json.loads(paths["terminal"].read_text(encoding="utf-8"))
    assert terminal["status"] == "failed"
    assert terminal["failure"]["phase"] == "pre_child"
    assert [item["status"] for item in terminal["phase_records"]] == [
        "passed",
        "passed",
        "failed",
        "not_started",
        "not_started",
        "not_started",
    ]
    assert not paths["sidecar"].exists()


@pytest.mark.parametrize(
    ("mutation", "error_code"),
    (
        ("existing_root", "wave7_sequence.target_exists"),
        ("determinism_drift", "wave7_sequence.identity_drift"),
        ("wrong_environment", "wave7_sequence.environment"),
        ("changing_gpu_baseline", "wave7_sequence.gpu_baseline"),
    ),
)
def test_pre_marker_fail_closed_checks_launch_nothing(
    controller: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mutation: str,
    error_code: str,
) -> None:
    request_path, plan_path, paths = _request_fixture(controller, tmp_path)
    request = paths["request"]
    _set_required_environment(monkeypatch, request)
    _install_static_shared_gpu_attestation(controller, monkeypatch)
    assert controller.prepare(request_path, plan_path) == 0
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if mutation == "existing_root":
        Path(plan["run_roots"]["uninterrupted"]).mkdir()
    elif mutation == "determinism_drift":
        Path(plan["determinism_preflight"]["path"]).write_text("{}\n", encoding="utf-8")
    elif mutation == "wrong_environment":
        monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", "0")
    else:

        def reject_changing_baseline(**_: Any) -> dict[str, Any]:
            raise controller.Wave7SequenceError(
                "changing baseline", code="wave7_sequence.gpu_baseline"
            )

        monkeypatch.setattr(
            controller, "_attest_shared_gpu_baseline", reject_changing_baseline
        )

    assert controller.execute(plan_path) == 2

    assert error_code in capsys.readouterr().err
    assert not paths["log"].exists()
    assert not paths["marker"].exists()
    assert not paths["terminal"].exists()
    assert not paths["sidecar"].exists()


@pytest.mark.parametrize("owner", ("cache", "model"))
def test_native_input_drift_after_plan_fails_before_marker_or_phase_launch(
    controller: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    owner: str,
) -> None:
    request_path, plan_path, paths = _request_fixture(controller, tmp_path)
    request = paths["request"]
    _set_required_environment(monkeypatch, request)
    _install_static_shared_gpu_attestation(controller, monkeypatch)
    assert controller.prepare(request_path, plan_path) == 0
    if owner == "cache":
        paths["cache_native_files"]["eval-forward_chunk"].write_bytes(
            b"post-plan-cache-drift"
        )
    else:
        paths["model_native_files"]["standalone"].write_bytes(b"post-plan-model-drift")

    assert controller.execute(plan_path) == 2
    assert not paths["log"].exists()
    assert not paths["marker"].exists()
    assert not paths["terminal"].exists()
    assert not paths["sidecar"].exists()
    assert all(not Path(path).exists() for path in request["run_roots"].values())


def test_identity_drift_between_gpu_phases_stops_before_second_gpu_launch(
    controller: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path, plan_path, paths = _request_fixture(controller, tmp_path)
    request = paths["request"]
    config_path = Path(paths["configs"]["uninterrupted"])
    simulated = paths["simulated_run_phase"]

    def mutate_after_uninterrupted(
        phase: str, argv: list[str], **kwargs: Any
    ) -> tuple[int, float, dict[str, Any]]:
        result = simulated(phase, argv, **kwargs)
        if phase == "uninterrupted":
            config_path.write_text("drift: true\n", encoding="utf-8")
        return result

    monkeypatch.setattr(controller, "_run_phase", mutate_after_uninterrupted)
    _set_required_environment(monkeypatch, request)
    _install_static_shared_gpu_attestation(controller, monkeypatch)
    assert controller.prepare(request_path, plan_path) == 0

    assert controller.execute(plan_path) == 1

    assert paths["log"].read_text(encoding="utf-8").splitlines() == ["uninterrupted"]
    terminal = json.loads(paths["terminal"].read_text(encoding="utf-8"))
    assert terminal["failure"]["phase"] == "interrupted_parent"
    assert terminal["failure"]["code"] == "wave7_sequence.identity_drift"


def test_native_cache_drift_between_phases_fails_terminally_before_next_launch(
    controller: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path, plan_path, paths = _request_fixture(controller, tmp_path)
    request = paths["request"]
    chunk_path = paths["cache_native_files"]["train_chunk"]
    simulated = paths["simulated_run_phase"]

    def mutate_after_uninterrupted(
        phase: str, argv: list[str], **kwargs: Any
    ) -> tuple[int, float, dict[str, Any]]:
        result = simulated(phase, argv, **kwargs)
        if phase == "uninterrupted":
            chunk_path.write_bytes(b"live-cache-drift")
        return result

    monkeypatch.setattr(controller, "_run_phase", mutate_after_uninterrupted)
    _set_required_environment(monkeypatch, request)
    _install_static_shared_gpu_attestation(controller, monkeypatch)
    assert controller.prepare(request_path, plan_path) == 0

    assert controller.execute(plan_path) == 1
    assert paths["log"].read_text(encoding="utf-8").splitlines() == ["uninterrupted"]
    terminal = json.loads(paths["terminal"].read_text(encoding="utf-8"))
    assert terminal["failure"]["phase"] == "interrupted_parent"
    assert terminal["failure"]["code"] == "wave7_sequence.input_attestation"
    assert not Path(request["run_roots"]["interrupted_parent"]).exists()
    assert not Path(request["run_roots"]["resume_child"]).exists()


def test_live_provenance_drift_stops_before_second_gpu_launch(
    controller: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path, plan_path, paths = _request_fixture(controller, tmp_path)
    request = paths["request"]
    _set_required_environment(monkeypatch, request)
    _install_static_shared_gpu_attestation(controller, monkeypatch)
    assert controller.prepare(request_path, plan_path) == 0
    admitted = copy.deepcopy(paths["provenance"])
    calls = 0

    def collect_live() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        observed = copy.deepcopy(admitted)
        if calls >= 3:
            observed["runtime"]["drift"] = True
        return observed

    monkeypatch.setattr(controller, "_collect_live_execution_provenance", collect_live)

    assert controller.execute(plan_path) == 1

    assert paths["log"].read_text(encoding="utf-8").splitlines() == ["uninterrupted"]
    terminal = json.loads(paths["terminal"].read_text(encoding="utf-8"))
    assert terminal["failure"]["phase"] == "interrupted_parent"
    assert terminal["failure"]["code"] == "wave7_sequence.runtime_provenance"
    assert calls == 4  # includes mandatory post-recovery preterminal revalidation


def test_mutated_gate_between_verifier_and_child_publishes_no_child_root(
    controller: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path, plan_path, paths = _request_fixture(controller, tmp_path)
    request = paths["request"]
    _set_required_environment(monkeypatch, request)
    _install_static_shared_gpu_attestation(controller, monkeypatch)
    simulated = paths["simulated_run_phase"]

    def mutate_after_verifier(
        phase: str, argv: list[str], **kwargs: Any
    ) -> tuple[int, float, dict[str, Any]]:
        try:
            result = simulated(phase, argv, **kwargs)
        except controller.Wave7SequenceError as exc:
            cleanup = _fake_process_record()
            raise controller.Wave7SequenceError(
                str(exc),
                code=exc.code,
                context={"duration_seconds": 0.001, "cleanup": cleanup},
            ) from exc
        if phase == "verify_pre_child":
            Path(paths["targets"]["pre_child_receipt"]).write_text(
                "{}\n", encoding="utf-8"
            )
        return result

    monkeypatch.setattr(controller, "_run_phase", mutate_after_verifier)
    assert controller.prepare(request_path, plan_path) == 0

    assert controller.execute(plan_path) == 2

    child_root = Path(request["run_roots"]["resume_child"])
    assert not child_root.exists()
    assert not paths["terminal"].exists()
    sidecar = json.loads(paths["sidecar"].read_text(encoding="utf-8"))
    assert sidecar["schema"] == controller.PUBLICATION_FAILURE_SCHEMA
    assert "pre-child" in sidecar["error"]


def test_launch_child_wrapper_rechecks_gate_before_exec_or_run_root(
    controller: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, paths = _request_fixture(controller, tmp_path)
    request = paths["request"]
    parent_root = Path(request["run_roots"]["interrupted_parent"])
    (parent_root / "checkpoints" / "step-3").mkdir(parents=True)
    receipt_path = Path(paths["targets"]["pre_child_receipt"])
    receipt = _write_signed_json(
        receipt_path,
        {
            "schema": controller.PRE_CHILD_RECEIPT_SCHEMA,
            "status": "passed",
            "child_launch_authorized": True,
            "mismatches": [],
        },
    )
    expected_digest = receipt["receipt_payload_sha256"]
    receipt_path.write_text("{}\n", encoding="utf-8")
    exec_calls: list[list[str]] = []

    def forbidden_execve(
        executable: str, argv: list[str], environment: dict[str, str]
    ) -> None:
        del executable, environment
        exec_calls.append(argv)
        raise AssertionError("mutated gate must not transfer to training")

    monkeypatch.setattr(controller.os, "execve", forbidden_execve)
    child_command = request["commands"]["resume_child"]
    separator = child_command.index("--")

    assert (
        controller.launch_child(
            receipt_path=receipt_path,
            expected_payload_sha256=expected_digest,
            config_path=Path(paths["configs"]["resume_child"]),
            run_root=Path(request["run_roots"]["resume_child"]),
            parent_run_root=parent_root,
            launcher=child_command[separator + 1 :],
        )
        == 1
    )

    assert exec_calls == []
    assert not Path(request["run_roots"]["resume_child"]).exists()


def test_phase_timeout_budget_includes_remaining_gpu_device_seconds(
    controller: ModuleType,
) -> None:
    policy = {
        "phase_timeout_seconds": {phase: 10.0 for phase in controller.PHASE_ORDER},
        "max_total_wall_seconds": 7.0,
        "max_total_gpu_device_seconds": 40.0,
    }

    assert controller._phase_timeout_budget(
        phase="uninterrupted",
        policy=policy,
        wall_elapsed=1.0,
        gpu_device_seconds=16.0,
    ) == pytest.approx(3.0)
    assert controller._phase_timeout_budget(
        phase="pre_child",
        policy=policy,
        wall_elapsed=1.0,
        gpu_device_seconds=39.0,
    ) == pytest.approx(0.125)

    with pytest.raises(controller.Wave7SequenceError) as exc_info:
        controller._phase_timeout_budget(
            phase="resume_child",
            policy=policy,
            wall_elapsed=1.0,
            gpu_device_seconds=40.0,
        )
    assert exc_info.value.code == "wave7_sequence.cost_cap"


def test_gpu_budget_is_recomputed_at_each_launch_and_blocks_zero_remaining(
    controller: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path, plan_path, paths = _request_fixture(controller, tmp_path)
    request = paths["request"]
    request["policy"]["max_total_gpu_device_seconds"] = 8.0
    unsigned = dict(request)
    unsigned.pop("request_payload_sha256")
    request["request_payload_sha256"] = _sha256(_canonical(unsigned))
    request_path.write_bytes(_canonical(request) + b"\n")
    _set_required_environment(monkeypatch, request)
    _install_static_shared_gpu_attestation(controller, monkeypatch)
    simulated = paths["simulated_run_phase"]
    launched: list[tuple[str, float]] = []
    controller_clock = [0.0]
    monkeypatch.setattr(controller.time, "monotonic", lambda: controller_clock[0])

    def record_budget(
        phase: str, argv: list[str], *, timeout: float, **kwargs: Any
    ) -> tuple[int, float, dict[str, Any]]:
        launched.append((phase, timeout))
        return_code, _, cleanup = simulated(phase, argv, timeout=timeout, **kwargs)
        duration = 0.75 if phase == "uninterrupted" else 0.25
        controller_clock[0] += duration
        return return_code, duration, cleanup

    monkeypatch.setattr(controller, "_run_phase", record_budget)
    assert controller.prepare(request_path, plan_path) == 0

    assert controller.execute(plan_path) == 1

    assert launched[0][0] == "uninterrupted"
    assert launched[0][1] == pytest.approx(1.0)
    assert launched[1][0] == "interrupted_parent"
    assert launched[1][1] == pytest.approx(0.25)
    assert "resume_child" not in {phase for phase, _ in launched}
    terminal = json.loads(paths["terminal"].read_text(encoding="utf-8"))
    assert terminal["failure"]["phase"] == "pre_child"
    assert terminal["failure"]["code"] == "wave7_sequence.cost_cap"


def test_phase_revalidation_exhausts_wall_budget_before_process_launch(
    controller: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path, plan_path, paths = _request_fixture(controller, tmp_path)
    request = paths["request"]
    request["policy"]["max_total_wall_seconds"] = 1.0
    request["policy"]["max_total_gpu_device_seconds"] = 8.0
    unsigned = dict(request)
    unsigned.pop("request_payload_sha256")
    request["request_payload_sha256"] = _sha256(_canonical(unsigned))
    request_path.write_bytes(_canonical(request) + b"\n")
    _set_required_environment(monkeypatch, request)
    _install_static_shared_gpu_attestation(controller, monkeypatch)
    original_revalidate = controller._revalidate_plan_inputs
    controller_clock = [0.0]
    launched: list[str] = []
    monkeypatch.setattr(controller.time, "monotonic", lambda: controller_clock[0])

    def consume_remaining_wall_budget(*args: Any, **kwargs: Any) -> None:
        original_revalidate(*args, **kwargs)
        if paths["marker"].exists():
            controller_clock[0] = 1.0

    def record_launch(
        phase: str, argv: list[str], **kwargs: Any
    ) -> tuple[int, float, dict[str, Any]]:
        launched.append(phase)
        return paths["simulated_run_phase"](phase, argv, **kwargs)

    monkeypatch.setattr(
        controller, "_revalidate_plan_inputs", consume_remaining_wall_budget
    )
    monkeypatch.setattr(controller, "_run_phase", record_launch)
    assert controller.prepare(request_path, plan_path) == 0

    assert controller.execute(plan_path) == 1
    assert launched == []
    terminal = json.loads(paths["terminal"].read_text(encoding="utf-8"))
    assert terminal["failure"]["phase"] == "uninterrupted"
    assert terminal["failure"]["code"] == "wave7_sequence.cost_cap"
    assert terminal["phase_records"][0]["attempt_count"] == 0


def test_terminal_publication_failure_uses_only_failure_sidecar(
    controller: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path, plan_path, paths = _request_fixture(controller, tmp_path)
    request = paths["request"]
    _set_required_environment(monkeypatch, request)
    _install_static_shared_gpu_attestation(controller, monkeypatch)
    assert controller.prepare(request_path, plan_path) == 0
    original = controller.write_strict_json_atomic

    def fail_terminal(path: Path, payload: dict[str, Any], **kwargs: Any) -> None:
        if path == paths["terminal"]:
            raise OSError("injected terminal publication failure")
        original(path, payload, **kwargs)

    monkeypatch.setattr(controller, "write_strict_json_atomic", fail_terminal)

    assert controller.execute(plan_path) == 2

    assert not paths["terminal"].exists()
    sidecar = json.loads(paths["sidecar"].read_text(encoding="utf-8"))
    assert sidecar["schema"] == controller.PUBLICATION_FAILURE_SCHEMA
    assert sidecar["terminal_receipt_path"] == str(paths["terminal"])
    assert sidecar["error_type"] == "OSError"


def test_timeout_terms_kills_and_reaps_only_started_process_group(
    controller: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path, plan_path, paths = _request_fixture(controller, tmp_path)
    request = paths["request"]
    request["policy"]["phase_timeout_seconds"]["uninterrupted"] = 0.1
    request["commands"]["interrupted_parent"][
        request["commands"]["interrupted_parent"].index("--timeout-seconds") + 1
    ] = "5.0"
    unsigned = dict(request)
    unsigned.pop("request_payload_sha256")
    request["request_payload_sha256"] = _sha256(_canonical(unsigned))
    request_path.write_bytes(_canonical(request) + b"\n")
    _set_required_environment(monkeypatch, request)
    _install_static_shared_gpu_attestation(controller, monkeypatch)

    def timeout_phase(
        phase: str, argv: list[str], **kwargs: Any
    ) -> tuple[int, float, dict[str, Any]]:
        with paths["log"].open("a", encoding="utf-8") as handle:
            handle.write("sleep\n")
        cleanup = _fake_process_record(term_sent=True)
        raise controller.Wave7SequenceError(
            "phase exceeded its fixed timeout",
            code="wave7_sequence.timeout",
            context={"phase": phase, "duration_seconds": 0.001, "cleanup": cleanup},
        )

    monkeypatch.setattr(controller, "_run_phase", timeout_phase)
    assert controller.prepare(request_path, plan_path) == 0

    assert controller.execute(plan_path) == 1

    assert paths["log"].read_text(encoding="utf-8").splitlines() == ["sleep"]
    terminal = json.loads(paths["terminal"].read_text(encoding="utf-8"))
    first = terminal["phase_records"][0]
    assert first["status"] == "failed"
    assert first["process"]["term_sent"] is True
    assert first["process"]["reaped"] is True
    assert first["duration_seconds"] == 0.001
    assert terminal["cost"]["gpu_device_seconds"] == 0.008
    assert terminal["bounded_cleanup"]["artifacts_deleted"] is False
    assert [row["status"] for row in terminal["phase_records"][1:]] == [
        "not_started"
    ] * 5
    unsigned_terminal = dict(terminal)
    unsigned_terminal.pop("receipt_payload_sha256")
    controller._validate_terminal_payload(unsigned_terminal)
    failed_mutations: list[dict[str, Any]] = []
    for mutated_gpu in (0.0, 0.004, 0.009, float("nan")):
        mutation = copy.deepcopy(unsigned_terminal)
        mutation["cost"]["gpu_device_seconds"] = mutated_gpu
        failed_mutations.append(mutation)
    cap_drift = copy.deepcopy(unsigned_terminal)
    cap_drift["cost"]["wall_seconds_cap"] += 1.0
    failed_mutations.append(cap_drift)
    wall_below_sum = copy.deepcopy(unsigned_terminal)
    wall_below_sum["cost"]["wall_seconds"] = 0.0005
    failed_mutations.append(wall_below_sum)
    for mutation in failed_mutations:
        with pytest.raises(controller.Wave7SequenceError) as exc_info:
            controller._validate_terminal_payload(mutation)
        assert exc_info.value.code == "wave7_sequence.terminal_schema"


@pytest.mark.parametrize("return_code", (0, 17))
def test_run_phase_captures_non_utf8_output_for_every_return_code(
    controller: ModuleType,
    return_code: int,
) -> None:
    program = (
        "import os; "
        "os.write(1, b'stdout-before-\\xff-after\\n'); "
        "os.write(2, b'stderr-before-\\xfe-after\\n'); "
        f"raise SystemExit({return_code})"
    )

    observed_return_code, _, process = controller._run_phase(
        "diagnostic-return-code",
        [sys.executable, "-c", program],
        environment={},
        timeout=5.0,
        term_grace=0.2,
        kill_grace=0.2,
    )

    assert observed_return_code == return_code
    assert process["stdout_tail"] == "stdout-before-\ufffd-after\n"
    assert process["stdout_truncated"] is False
    assert process["stderr_tail"] == "stderr-before-\ufffd-after\n"
    assert process["stderr_truncated"] is False


def test_run_phase_concurrently_caps_stdout_and_stderr_to_exact_tails(
    controller: ModuleType,
) -> None:
    cap = 64 * 1024
    stdout_suffix = b"stdout-exact-tail\n"
    stderr_suffix = b"stderr-exact-tail\n"
    program = (
        "import os; "
        f"os.write(1, b'a' * {cap * 2} + {stdout_suffix!r}); "
        f"os.write(2, b'b' * {cap * 2} + {stderr_suffix!r})"
    )

    return_code, _, process = controller._run_phase(
        "diagnostic-cap",
        [sys.executable, "-c", program],
        environment={},
        timeout=5.0,
        term_grace=0.2,
        kill_grace=0.2,
    )

    assert return_code == 0
    assert process["stdout_tail"].endswith(stdout_suffix.decode("ascii"))
    assert process["stdout_truncated"] is True
    assert len(process["stdout_tail"].encode("utf-8")) == cap
    assert process["stderr_tail"].endswith(stderr_suffix.decode("ascii"))
    assert process["stderr_truncated"] is True
    assert len(process["stderr_tail"].encode("utf-8")) == cap


def test_run_phase_timeout_retains_output_in_reaped_cleanup(
    controller: ModuleType,
) -> None:
    program = (
        "import os, time; "
        "os.write(1, b'timeout-stdout\\n'); "
        "os.write(2, b'timeout-stderr\\n'); "
        "time.sleep(30)"
    )

    with pytest.raises(controller.Wave7SequenceError) as exc_info:
        controller._run_phase(
            "diagnostic-timeout",
            [sys.executable, "-c", program],
            environment={},
            timeout=0.2,
            term_grace=0.2,
            kill_grace=0.2,
        )

    assert exc_info.value.code == "wave7_sequence.timeout"
    cleanup = exc_info.value.context["cleanup"]
    assert cleanup["reaped"] is True
    assert cleanup["stdout_tail"] == "timeout-stdout\n"
    assert cleanup["stdout_truncated"] is False
    assert cleanup["stderr_tail"] == "timeout-stderr\n"
    assert cleanup["stderr_truncated"] is False


def test_rc0_leader_with_live_adopted_child_is_cleaned_and_rejected(
    controller: ModuleType, tmp_path: Path
) -> None:
    child_pid_path = tmp_path / "adopted-child.pid"
    program = (
        "import pathlib, subprocess, sys; "
        "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid))"
    )

    with pytest.raises(controller.Wave7SequenceError) as exc_info:
        controller._run_phase(
            "rc0-child-survivor",
            [sys.executable, "-c", program],
            environment={},
            timeout=5.0,
            term_grace=0.2,
            kill_grace=0.2,
        )

    assert exc_info.value.code == "wave7_sequence.cleanup"
    cleanup = exc_info.value.context["cleanup"]
    assert cleanup["reaped"] is True
    assert cleanup["remaining_pids"] == []
    assert cleanup["remaining_pgids"] == []
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    assert not Path(f"/proc/{child_pid}").exists()


def test_cpu_signal_interrupt_reaps_process_group_before_late_write(
    tmp_path: Path,
) -> None:
    started = tmp_path / "child-started"
    late = tmp_path / "late-write"
    result_path = tmp_path / "cleanup.json"
    child = tmp_path / "signal-child.py"
    child.write_text(
        """
from pathlib import Path
import signal
import sys
import time

signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path(sys.argv[1]).write_text("started\\n", encoding="utf-8")
time.sleep(0.8)
Path(sys.argv[2]).write_text("late\\n", encoding="utf-8")
""".lstrip(),
        encoding="utf-8",
    )
    helper = tmp_path / "signal-helper.py"
    helper.write_text(
        f"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys

spec = importlib.util.spec_from_file_location("signal_sequence", {str(SCRIPT)!r})
module = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(module)
try:
    module._run_phase(
        "signal-probe",
        [sys.executable, {str(child)!r}, {str(started)!r}, {str(late)!r}],
        environment={{}},
        timeout=5.0,
        term_grace=0.1,
        kill_grace=1.0,
    )
except module.Wave7SequenceError as exc:
    Path({str(result_path)!r}).write_text(
        json.dumps({{"code": exc.code, "context": exc.context}}) + "\\n",
        encoding="utf-8",
    )
""".lstrip(),
        encoding="utf-8",
    )
    process = subprocess.Popen(
        [sys.executable, str(helper)],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    deadline = time.monotonic() + 15.0
    while not started.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert started.exists()

    process.send_signal(signal.SIGINT)
    stdout, stderr = process.communicate(timeout=10.0)

    assert process.returncode == 0, (stdout, stderr)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["code"] == "wave7_sequence.interrupted"
    assert result["context"]["cleanup"]["kill_sent"] is True
    assert result["context"]["cleanup"]["reaped"] is True
    time.sleep(0.9)
    assert not late.exists()


def test_cpu_signal_interrupt_reaps_detached_nested_session_before_terminal(
    tmp_path: Path,
) -> None:
    started = tmp_path / "nested-started.json"
    late = tmp_path / "nested-late-write"
    result_path = tmp_path / "nested-cleanup.json"
    worker = tmp_path / "nested-worker.py"
    worker.write_text(
        """
from pathlib import Path
import signal
import sys
import time

signal.signal(signal.SIGTERM, signal.SIG_IGN)
time.sleep(1.2)
Path(sys.argv[1]).write_text("late\\n", encoding="utf-8")
""".lstrip(),
        encoding="utf-8",
    )
    nested = tmp_path / "nested-session.py"
    nested.write_text(
        f"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

signal.signal(signal.SIGTERM, signal.SIG_IGN)
worker = subprocess.Popen([sys.executable, {str(worker)!r}, {str(late)!r}])
Path({str(started)!r}).write_text(
    json.dumps({{
        "nested_pid": os.getpid(),
        "nested_pgid": os.getpgrp(),
        "worker_pid": worker.pid,
    }}) + "\\n",
    encoding="utf-8",
)
time.sleep(5.0)
""".lstrip(),
        encoding="utf-8",
    )
    outer = tmp_path / "outer-controller.py"
    outer.write_text(
        f"""
import signal
import subprocess
import sys
import time

signal.signal(signal.SIGTERM, signal.SIG_IGN)
subprocess.Popen([sys.executable, {str(nested)!r}], start_new_session=True)
time.sleep(5.0)
""".lstrip(),
        encoding="utf-8",
    )
    helper = tmp_path / "nested-helper.py"
    helper.write_text(
        f"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys

spec = importlib.util.spec_from_file_location("nested_sequence", {str(SCRIPT)!r})
module = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(module)
try:
    module._run_phase(
        "nested-signal-probe",
        [sys.executable, {str(outer)!r}],
        environment={{}},
        timeout=5.0,
        term_grace=0.1,
        kill_grace=1.0,
    )
except module.Wave7SequenceError as exc:
    Path({str(result_path)!r}).write_text(
        json.dumps({{"code": exc.code, "context": exc.context}}) + "\\n",
        encoding="utf-8",
    )
""".lstrip(),
        encoding="utf-8",
    )
    process = subprocess.Popen(
        [sys.executable, str(helper)],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    deadline = time.monotonic() + 15.0
    while not started.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert started.exists()
    nested_identity = json.loads(started.read_text(encoding="utf-8"))

    process.send_signal(signal.SIGINT)
    stdout, stderr = process.communicate(timeout=10.0)

    assert process.returncode == 0, (stdout, stderr)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    cleanup = result["context"]["cleanup"]
    assert result["code"] == "wave7_sequence.interrupted"
    assert cleanup["kill_sent"] is True
    assert cleanup["reaped"] is True
    assert cleanup["remaining_pids"] == []
    assert cleanup["remaining_pgids"] == []
    captured_pids = {row["pid"] for row in cleanup["captured_process_graph"]}
    assert nested_identity["nested_pid"] in captured_pids
    assert nested_identity["worker_pid"] in captured_pids
    assert not Path(f"/proc/{nested_identity['nested_pid']}").exists()
    assert not Path(f"/proc/{nested_identity['worker_pid']}").exists()
    with pytest.raises(ProcessLookupError):
        os.killpg(nested_identity["nested_pgid"], 0)
    time.sleep(1.3)
    assert not late.exists()
