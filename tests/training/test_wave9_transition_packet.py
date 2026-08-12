from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/probes/coordexp_swift/wave9_transition_packet.py"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _signed(value: dict[str, object], field: str) -> dict[str, object]:
    return {**value, field: _sha256(_canonical(value))}


def _write_canonical(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value) + b"\n")


@pytest.fixture
def validator():
    spec = importlib.util.spec_from_file_location(
        "wave9_transition_packet_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _binding(tmp_path: Path, name: str, schema: str) -> dict[str, object]:
    path = tmp_path / "evidence" / f"{name}.json"
    receipt = _signed(
        {
            "schema": schema,
            "status": "passed",
            "identity": _sha256(name.encode("ascii")),
        },
        "receipt_payload_sha256",
    )
    _write_canonical(path, receipt)
    return {
        "path": str(path.resolve()),
        "file_sha256": _sha256(path.read_bytes()),
        "payload_sha256": receipt["receipt_payload_sha256"],
        "schema": schema,
        "status": "passed",
        "current": True,
    }


def _sealed_packet(validator, tmp_path: Path) -> dict[str, object]:
    config = tmp_path / "configs" / "prod.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("training:\n  world_size: 8\n", encoding="utf-8")
    cache_root = tmp_path / "production-cache"
    cache_root.mkdir(parents=True)
    train_fp = "1" * 64
    eval_fp = "2" * 64
    receipt_target = tmp_path / "receipts" / "cache-build.json"
    run_root = tmp_path / "wave9-run"

    packet: dict[str, object] = {
        "schema": validator.PACKET_SCHEMA,
        "lifecycle": "sealed",
        "missing_inputs": [],
        "gates": {
            "wave7": _binding(
                tmp_path, "wave7", "coordexp-swift-wave7-gate-receipt-v1"
            ),
            "wave8": _binding(
                tmp_path, "wave8", "coordexp-swift-wave8-gate-receipt-v1"
            ),
            "findings": {"p0": [], "p1": []},
        },
        "production": {
            "config": {
                "path": str(config.resolve()),
                "file_sha256": _sha256(config.read_bytes()),
                "resolved_fingerprint": "3" * 64,
            },
            "policy": {
                "packing_policy": "source_order_next_fit",
                "input_provider": "synchronous",
                "cache_version": "coordexp-swift-pack-cache-v3",
                "dependency_change": "none",
                "world_size": 8,
            },
            "identities": {
                "source": {
                    **_binding(
                        tmp_path,
                        "source",
                        "coordexp-swift-executed-source-identity-v1",
                    ),
                    "repository_commit": "4" * 40,
                    "dirty": True,
                    "dirty_sha256": "5" * 64,
                },
                "provenance": _binding(
                    tmp_path,
                    "provenance",
                    "coordexp-swift-executed-provenance-v1",
                ),
                "runtime": _binding(
                    tmp_path, "runtime", "coordexp-swift-runtime-identity-v3"
                ),
                "determinants": _binding(
                    tmp_path,
                    "determinants",
                    "coordexp-swift-cache-determinant-identity-v3",
                ),
                "model": _binding(
                    tmp_path, "model", "coordexp-swift-model-identity-v1"
                ),
            },
        },
        "cache": {
            "root_env": {
                "name": "COORDEXP_SWIFT_PACK_CACHE_ROOT",
                "value": str(cache_root.resolve()),
            },
            "splits": {
                "train": {
                    "target": str(
                        (cache_root / "coordexp-swift-pack-cache-v3" / train_fp)
                    ),
                    "fingerprint": train_fp,
                },
                "eval.forward": {
                    "target": str(
                        (cache_root / "coordexp-swift-pack-cache-v3" / eval_fp)
                    ),
                    "fingerprint": eval_fp,
                },
            },
            "build": {
                "argv": [
                    str(Path(sys.executable).resolve()),
                    "-m",
                    "src.prepare_train_cache",
                    "--config",
                    str(config.resolve()),
                    "--receipt",
                    str(receipt_target.resolve()),
                ],
                "env": {
                    "COORDEXP_SWIFT_PACK_CACHE_ROOT": str(cache_root.resolve()),
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                    "FLASH_ATTENTION_DETERMINISTIC": "1",
                    "OMP_NUM_THREADS": "16",
                },
                "payload_validation": "payloads",
                "receipt_target": str(receipt_target.resolve()),
            },
            "materialization": {
                "strategy": "fork_process_pool",
                "workers": 16,
                "resolved_planner_workers": 1,
            },
        },
        "budgets": {
            name: {
                "expected": expected,
                "hard": hard,
                "unit": unit,
                "evidence": _binding(
                    tmp_path,
                    f"budget-{name}",
                    f"coordexp-swift-wave9-{name}-budget-evidence-v1",
                ),
            }
            for name, expected, hard, unit in (
                ("cpu", 100.0, 200.0, "cpu_core_seconds"),
                ("io", 1_000_000, 2_000_000, "bytes"),
                ("time", 60.0, 120.0, "seconds"),
                ("storage", 2_000_000, 4_000_000, "bytes"),
                ("gpu", 4_800.0, 9_600.0, "gpu_device_seconds"),
                ("host", 8_000_000, 16_000_000, "bytes"),
                ("artifact", 1_000_000, 2_000_000, "bytes"),
            )
        },
        "rollback": {
            "bindings": [
                _binding(
                    tmp_path,
                    "rollback",
                    "coordexp-swift-wave9-rollback-binding-v1",
                )
            ],
            "route": "restore_previous_config_and_reuse_immutable_prior_cache",
            "immutable": True,
            "delete_old_caches": False,
            "mutate_old_caches": False,
        },
        "launch": {
            "argv": [
                "/opt/conda/envs/ms/bin/accelerate",
                "launch",
                "--multi_gpu",
                "--num_processes",
                "8",
                "--module",
                "src.train",
                "--config",
                str(config.resolve()),
            ],
            "env": {
                "COORDEXP_SWIFT_PACK_CACHE_ROOT": str(cache_root.resolve()),
                "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
                "FLASH_ATTENTION_DETERMINISTIC": "1",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            },
            "devices": list(range(8)),
            "targets": {
                "run_root": str(run_root.resolve()),
                "checkpoint_root": str((run_root / "checkpoints").resolve()),
                "evaluation_root": str((run_root / "evaluation").resolve()),
                "resume_root": str((run_root / "resume").resolve()),
            },
            "attempt_marker": str((tmp_path / "wave9-attempt.json").resolve()),
            "terminal_receipt": str((tmp_path / "wave9-terminal.json").resolve()),
            "retry": {"allowed": False, "automatic": False, "max_attempts": 1},
        },
    }
    return _signed(packet, "packet_payload_sha256")


def _resign(packet: dict[str, object]) -> None:
    packet.pop("packet_payload_sha256", None)
    packet["packet_payload_sha256"] = _sha256(_canonical(packet))


def test_sealed_packet_validates_to_a_deterministic_executable_projection(
    validator, tmp_path: Path
) -> None:
    packet = _sealed_packet(validator, tmp_path)

    first = validator.validate_transition_packet(packet)
    reordered = {key: packet[key] for key in reversed(packet)}
    second = validator.validate_transition_packet(reordered)

    assert first == second
    assert first["schema"] == validator.PROJECTION_SCHEMA
    assert first["lifecycle"] == "sealed"
    assert first["executable"] is True
    assert first["missing_inputs"] == []
    assert first["packet_payload_sha256"] == packet["packet_payload_sha256"]
    assert first["policy"] == packet["production"]["policy"]
    projection_sha256 = first.pop("projection_sha256")
    assert projection_sha256 == _sha256(_canonical(first))


def test_draft_requires_exact_missing_inputs_and_is_never_executable(
    validator, tmp_path: Path
) -> None:
    packet = _sealed_packet(validator, tmp_path)
    packet["lifecycle"] = "draft"
    del packet["gates"]["wave8"]
    packet["missing_inputs"] = ["/gates/wave8"]
    _resign(packet)

    projection = validator.validate_transition_packet(packet)

    assert projection["lifecycle"] == "draft"
    assert projection["executable"] is False
    assert projection["missing_inputs"] == ["/gates/wave8"]

    packet["missing_inputs"] = []
    _resign(packet)
    with pytest.raises(validator.TransitionPacketError, match="missing_inputs"):
        validator.validate_transition_packet(packet)


def test_minimal_draft_names_every_missing_sealed_input_literal(validator) -> None:
    missing = sorted(
        [
            "/gates/findings",
            "/gates/wave7",
            "/gates/wave8",
            "/production/config",
            "/production/policy",
            "/production/identities/source",
            "/production/identities/provenance",
            "/production/identities/runtime",
            "/production/identities/determinants",
            "/production/identities/model",
            "/cache/root_env",
            "/cache/splits/train",
            "/cache/splits/eval.forward",
            "/cache/build",
            "/cache/materialization",
            "/budgets/cpu",
            "/budgets/io",
            "/budgets/time",
            "/budgets/storage",
            "/budgets/gpu",
            "/budgets/host",
            "/budgets/artifact",
            "/rollback/bindings",
            "/rollback/route",
            "/rollback/immutable",
            "/rollback/delete_old_caches",
            "/rollback/mutate_old_caches",
            "/launch/argv",
            "/launch/env",
            "/launch/devices",
            "/launch/targets",
            "/launch/attempt_marker",
            "/launch/terminal_receipt",
            "/launch/retry",
        ]
    )
    packet = _signed(
        {
            "schema": validator.PACKET_SCHEMA,
            "lifecycle": "draft",
            "missing_inputs": missing,
        },
        "packet_payload_sha256",
    )

    projection = validator.validate_transition_packet(packet)

    assert projection["executable"] is False
    assert projection["missing_inputs"] == missing


@pytest.mark.parametrize(
    "encoded",
    (
        '{"schema":"a","schema":"b"}\n',
        '{"schema":"a","value":NaN}\n',
        '{"schema":"a","value":Infinity}\n',
    ),
)
def test_file_loader_rejects_duplicate_keys_and_nonfinite_json(
    validator, tmp_path: Path, encoded: str
) -> None:
    path = tmp_path / "bad.json"
    path.write_text(encoded, encoding="utf-8")
    with pytest.raises(validator.TransitionPacketError, match="strict"):
        validator.load_transition_packet(path)


def test_file_loader_requires_canonical_signed_bytes(validator, tmp_path: Path) -> None:
    packet = _sealed_packet(validator, tmp_path)
    path = tmp_path / "packet.json"
    path.write_text(json.dumps(packet, indent=2), encoding="utf-8")
    with pytest.raises(validator.TransitionPacketError, match="canonical"):
        validator.load_transition_packet(path)


def test_packet_signature_and_bound_file_mutation_fail_closed(
    validator, tmp_path: Path
) -> None:
    packet = _sealed_packet(validator, tmp_path)
    packet["production"]["policy"]["world_size"] = 4
    with pytest.raises(validator.TransitionPacketError, match="payload signature"):
        validator.validate_transition_packet(packet)

    packet = _sealed_packet(validator, tmp_path / "bound")
    gate = Path(packet["gates"]["wave7"]["path"])
    gate.write_bytes(gate.read_bytes() + b" ")
    with pytest.raises(validator.TransitionPacketError, match="file_sha256"):
        validator.validate_transition_packet(packet)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    (
        (("gates", "wave7", "status"), "failed", "passed"),
        (("gates", "wave8", "current"), False, "current"),
        (("production", "policy", "packing_policy"), "window_binpack", "policy"),
        (("production", "policy", "input_provider"), "depth_one", "policy"),
        (("production", "policy", "dependency_change"), "upgrade", "policy"),
        (("production", "policy", "world_size"), 4, "policy"),
        (("cache", "build", "payload_validation"), "manifest", "payloads"),
        (("cache", "materialization", "strategy"), "serial", "materialization"),
        (("cache", "materialization", "workers"), 1, "workers"),
        (("rollback", "delete_old_caches"), True, "old cache"),
        (("launch", "retry", "allowed"), True, "retry"),
    ),
)
def test_sealed_packet_rejects_gate_policy_materialization_and_retry_drift(
    validator,
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    packet = _sealed_packet(validator, tmp_path)
    target = packet
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    _resign(packet)
    with pytest.raises(validator.TransitionPacketError, match=message):
        validator.validate_transition_packet(packet)


def test_gate_rejects_legacy_schema_even_when_resigned(
    validator, tmp_path: Path
) -> None:
    packet = _sealed_packet(validator, tmp_path)
    gate = packet["gates"]["wave7"]
    gate["schema"] = "coordexp-swift-wave7-legacy-gate-v0"
    _resign(packet)
    with pytest.raises(validator.TransitionPacketError, match="legacy"):
        validator.validate_transition_packet(packet)


def test_cache_targets_must_be_absent_canonical_v3_paths(
    validator, tmp_path: Path
) -> None:
    packet = _sealed_packet(validator, tmp_path)
    target = Path(packet["cache"]["splits"]["train"]["target"])
    target.mkdir(parents=True)
    with pytest.raises(validator.TransitionPacketError, match="absent"):
        validator.validate_transition_packet(packet)

    packet = _sealed_packet(validator, tmp_path / "wrong")
    packet["cache"]["splits"]["train"]["target"] = str(
        (tmp_path / "wrong" / "elsewhere" / ("1" * 64)).resolve()
    )
    _resign(packet)
    with pytest.raises(validator.TransitionPacketError, match="canonical"):
        validator.validate_transition_packet(packet)


def test_symlinked_config_and_nonregular_evidence_are_rejected(
    validator, tmp_path: Path
) -> None:
    packet = _sealed_packet(validator, tmp_path)
    original = Path(packet["production"]["config"]["path"])
    link = tmp_path / "config-link.yaml"
    link.symlink_to(original)
    packet["production"]["config"]["path"] = str(link.absolute())
    _resign(packet)
    with pytest.raises(validator.TransitionPacketError, match="symlink"):
        validator.validate_transition_packet(packet)

    packet = _sealed_packet(validator, tmp_path / "directory")
    binding = packet["production"]["identities"]["runtime"]
    evidence = Path(binding["path"])
    evidence.unlink()
    evidence.mkdir()
    _resign(packet)
    with pytest.raises(validator.TransitionPacketError, match="regular file"):
        validator.validate_transition_packet(packet)


@pytest.mark.parametrize("bad", (None, "TBD", "{FINGERPRINT}", "<path>"))
def test_nulls_and_placeholders_are_rejected(
    validator, tmp_path: Path, bad: object
) -> None:
    packet = _sealed_packet(validator, tmp_path)
    packet["rollback"]["route"] = bad
    _resign(packet)
    with pytest.raises(validator.TransitionPacketError, match="null|placeholder"):
        validator.validate_transition_packet(packet)


def test_build_environment_is_allowlisted_and_binds_explicit_cache_root(
    validator, tmp_path: Path
) -> None:
    packet = _sealed_packet(validator, tmp_path)
    packet["cache"]["build"]["env"]["SECRET_TOKEN"] = "do-not-copy"
    _resign(packet)
    with pytest.raises(validator.TransitionPacketError, match="allowlist"):
        validator.validate_transition_packet(packet)

    packet = _sealed_packet(validator, tmp_path / "root")
    del packet["cache"]["build"]["env"]["COORDEXP_SWIFT_PACK_CACHE_ROOT"]
    _resign(packet)
    with pytest.raises(validator.TransitionPacketError, match="PACK_CACHE_ROOT"):
        validator.validate_transition_packet(packet)


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("FLASH_ATTENTION_DETERMINISTIC", None),
        ("FLASH_ATTENTION_DETERMINISTIC", "0"),
        ("CUBLAS_WORKSPACE_CONFIG", None),
        ("CUBLAS_WORKSPACE_CONFIG", ":16:8"),
    ),
)
def test_cache_build_requires_exact_strict_determinism_environment(
    validator,
    tmp_path: Path,
    name: str,
    value: str | None,
) -> None:
    packet = _sealed_packet(validator, tmp_path)
    if value is None:
        del packet["cache"]["build"]["env"][name]
    else:
        packet["cache"]["build"]["env"][name] = value
    _resign(packet)

    with pytest.raises(validator.TransitionPacketError, match="environment"):
        validator.validate_transition_packet(packet)


def test_each_budget_requires_evidence_and_hard_headroom(
    validator, tmp_path: Path
) -> None:
    packet = _sealed_packet(validator, tmp_path)
    packet["budgets"]["gpu"]["hard"] = 1.0
    _resign(packet)
    with pytest.raises(validator.TransitionPacketError, match="hard"):
        validator.validate_transition_packet(packet)

    packet = _sealed_packet(validator, tmp_path / "missing")
    del packet["budgets"]["artifact"]
    _resign(packet)
    with pytest.raises(validator.TransitionPacketError, match="budgets"):
        validator.validate_transition_packet(packet)


def test_launch_contract_binds_devices_absent_targets_and_strict_environment(
    validator, tmp_path: Path
) -> None:
    packet = _sealed_packet(validator, tmp_path)
    packet["launch"]["devices"] = list(range(7))
    _resign(packet)
    with pytest.raises(validator.TransitionPacketError, match="devices"):
        validator.validate_transition_packet(packet)

    packet = _sealed_packet(validator, tmp_path / "target")
    Path(packet["launch"]["attempt_marker"]).write_text("occupied")
    with pytest.raises(validator.TransitionPacketError, match="absent"):
        validator.validate_transition_packet(packet)

    packet = _sealed_packet(validator, tmp_path / "env")
    packet["launch"]["env"]["FLASH_ATTENTION_DETERMINISTIC"] = "0"
    _resign(packet)
    with pytest.raises(validator.TransitionPacketError, match="environment"):
        validator.validate_transition_packet(packet)


def test_validate_cli_emits_projection_and_has_no_author_or_seal_surface(
    validator, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    packet = _sealed_packet(validator, tmp_path)
    packet_path = tmp_path / "packet.json"
    _write_canonical(packet_path, packet)

    assert validator.main(["validate", "--packet", str(packet_path)]) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["schema"] == validator.PROJECTION_SCHEMA
    assert emitted["executable"] is True

    parser = validator._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["author"])
    with pytest.raises(SystemExit):
        parser.parse_args(["seal"])
