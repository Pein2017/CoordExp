from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/probes/coordexp_swift/wave7_exact_resume_request.py"
IMMUTABLE_R5_SEQUENCE_RECEIPT = (
    REPO_ROOT / "outputs/probes/coordexp_swift/wave7_exact_resume/2026-08-11-r5/"
    "sequence-receipt.json"
)
IMMUTABLE_R6_PREFLIGHT_ROOT = (
    REPO_ROOT / "outputs/probes/coordexp_swift/wave7_exact_resume/2026-08-11-r6/"
    "determinism-preflight"
)
IMMUTABLE_R6_PREFLIGHT_PLAN = IMMUTABLE_R6_PREFLIGHT_ROOT / "plan.json"
IMMUTABLE_R6_PREFLIGHT_MARKER = (
    IMMUTABLE_R6_PREFLIGHT_ROOT / "attempt-start-marker.json"
)
IMMUTABLE_R6_PREFLIGHT_TERMINAL = IMMUTABLE_R6_PREFLIGHT_ROOT / "terminal-receipt.json"
AMENDMENT_AUTHORITY = (
    REPO_ROOT
    / "openspec/changes/archive/2026-08-12-harden-optimize-coordexp-swift-training-infrastructure/"
    "measurement-plan.md"
)
BASE_CONFIG = (
    REPO_ROOT
    / "configs/coordexp_swift/smoke/"
    "qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_llm_12000_"
    "accelerate8_ebs24_2step_warmup0p1_eval_patchproof.yaml"
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_signed_json(path: Path, payload: dict[str, object]) -> dict[str, object]:
    signed = dict(payload)
    signed["receipt_payload_sha256"] = _sha256(_canonical(payload))
    path.write_bytes(_canonical(signed) + b"\n")
    return signed


@pytest.fixture
def producer():
    spec = importlib.util.spec_from_file_location(
        "wave7_exact_resume_request_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parser_has_no_arbitrary_command_surface(producer):
    parser = producer._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["author", "--command", "/bin/sh"])


def test_r7_schemas_and_predecessor_cli_are_exact(producer):
    assert producer.AMENDMENT_SCHEMA == "coordexp-swift-wave7-r7-amendment-v4"
    assert producer.REQUEST_SCHEMA == (
        "coordexp-swift-wave7-exact-resume-sequence-request-v6"
    )
    assert producer.LEAF_NAMES == {
        "amendment": "amendment-v4.json",
        "cache": "cache-input-attestation.json",
        "model": "model-input-attestation.json",
        "request": "request-v6.json",
    }
    parser = producer._build_parser()
    author = parser._subparsers._group_actions[0].choices["author"]
    destinations = {action.dest for action in author._actions}
    assert {
        "predecessor_r5_failure",
        "predecessor_r6_preflight_plan",
        "predecessor_r6_preflight_marker",
        "predecessor_r6_preflight_terminal",
    } <= destinations
    assert producer.AMENDMENT_SCOPE == {
        "sequence": "one_fresh_r7_one_shot_successor",
        "predecessor": "immutable_r4_r5_and_r6_failures_historical_non_executable",
        "gpu_admission": "shared_preexisting_subset_v1",
        "claim_scope": (
            "correctness_plumbing_numerical_artifact_interruption_exact_resume_only"
        ),
        "performance_promotion": False,
        "automatic_r8": False,
    }


def test_current_namespace_is_authorized_core_4_with_exact_derived_paths(producer):
    sequence_root = (
        REPO_ROOT / "outputs/probes/coordexp_swift/wave7_exact_resume/2026-08-12-core-4"
    ).resolve()
    cache_root = (
        REPO_ROOT
        / "outputs/probes/coordexp_swift/private_v3_cache/2026-08-12-wave7-core-4"
    ).resolve()

    assert producer.R7_SEQUENCE_ROOT == sequence_root
    assert producer.R7_CACHE_ROOT == cache_root
    assert producer.R7_CONFIG_PATHS == {
        "uninterrupted": sequence_root / "configs/uninterrupted.yaml",
        "interrupted_parent": sequence_root / "configs/interrupted-parent.yaml",
        "resume_child": sequence_root / "configs/resume-child.yaml",
    }
    assert producer.R7_RUN_ROOTS == {
        role: sequence_root / "runs" / role for role in producer.RUN_ROLES
    }
    assert producer.R7_CACHE_PREPARATION_RECEIPT == (
        cache_root / "preparation-receipt.json"
    )
    assert (
        producer.R7_RUNTIME_RECEIPT == sequence_root / "runtime/runtime-admission.json"
    )
    assert producer.R7_DETERMINISM_PREFLIGHT_PLAN == (
        sequence_root / "determinism-preflight/plan.json"
    )
    assert producer.R7_DETERMINISM_PREFLIGHT_RECEIPT == (
        sequence_root / "determinism-preflight/terminal-receipt.json"
    )


def test_authority_section_drift_is_rejected(
    producer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    authority = tmp_path / "measurement-plan.md"
    authority.write_text("### 2026-08-11: wrong heading\n\nnot approved\n")
    monkeypatch.setattr(producer, "AMENDMENT_AUTHORITY_PATH", authority.resolve())
    monkeypatch.setattr(
        producer,
        "FROZEN_AMENDMENT_AUTHORITY_FILE_SHA256",
        _sha256(authority.read_bytes()),
    )
    with pytest.raises(producer.RequestAuthoringError, match="anchor"):
        producer._build_amendment_payload(authority)


@pytest.mark.parametrize(
    "phrase",
    (
        "The user explicitly authorized exactly one fresh Wave 7 `r7` one-shot successor.",
        "This is not an r6 retry",
        "no retry, root switch, or automatic Wave 7 `r8` successor",
    ),
)
def test_authority_requires_each_exact_r7_phrase(
    producer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phrase: str
):
    authority = tmp_path / "measurement-plan.md"
    section = "\n".join(
        (
            producer.AMENDMENT_SECTION_ANCHOR,
            "",
            "The user explicitly authorized exactly one fresh Wave 7 `r7` one-shot successor.",
            "This is not an r6 retry",
            "no retry, root switch, or automatic Wave 7 `r8` successor",
        )
    )
    authority.write_text(
        section.replace(phrase, "authority phrase removed"), encoding="utf-8"
    )
    monkeypatch.setattr(producer, "AMENDMENT_AUTHORITY_PATH", authority.resolve())
    monkeypatch.setattr(
        producer,
        "FROZEN_AMENDMENT_AUTHORITY_FILE_SHA256",
        _sha256(authority.read_bytes()),
    )
    authority_text = authority.read_text(encoding="utf-8")
    monkeypatch.setattr(
        producer,
        "FROZEN_AMENDMENT_AUTHORITY_SECTION_SHA256",
        _sha256(authority_text.encode("utf-8")),
    )
    with pytest.raises(producer.RequestAuthoringError, match="approval text"):
        producer._build_amendment_payload(authority)


def test_amendment_binds_final_canonical_openspec_authority(producer):
    if producer.FROZEN_AMENDMENT_AUTHORITY_FILE_SHA256 == "PENDING":
        pytest.skip("final r7 authority hash is not frozen yet")
    assert producer.FROZEN_AMENDMENT_AUTHORITY_FILE_SHA256 == (
        "6b088bc3ba7befcca26963b08ef493ce67a14dbfa190158b4ec270a410332730"
    )
    assert producer.FROZEN_AMENDMENT_AUTHORITY_SECTION_SHA256 == (
        "6b94dafab01c0845e32a807a1d2b8dc5d31e677d5ae81e448a3110a7dd05a491"
    )
    amendment = producer._build_amendment_payload(AMENDMENT_AUTHORITY)

    assert amendment["authority"] == {
        "path": str(AMENDMENT_AUTHORITY.resolve()),
        "file_sha256": producer.FROZEN_AMENDMENT_AUTHORITY_FILE_SHA256,
        "section_sha256": producer.FROZEN_AMENDMENT_AUTHORITY_SECTION_SHA256,
    }


def test_amendment_rejects_noncanonical_authority_copy(producer, tmp_path: Path):
    copied = tmp_path / "measurement-plan.md"
    copied.write_bytes(AMENDMENT_AUTHORITY.read_bytes())

    with pytest.raises(producer.RequestAuthoringError, match="canonical"):
        producer._build_amendment_payload(copied)


def test_publish_order_and_request_is_last_commit_point(
    producer, tmp_path: Path, monkeypatch
):
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(producer, "R7_SEQUENCE_ROOT", output.resolve())
    payloads = {
        "amendment": {"schema": producer.AMENDMENT_SCHEMA, "status": "approved"},
        "cache": {"schema": producer.CACHE_ATTESTATION_SCHEMA, "status": "passed"},
        "model": {"schema": producer.MODEL_ATTESTATION_SCHEMA, "status": "passed"},
        "request": {"schema": producer.REQUEST_SCHEMA, "status": "prepared"},
    }
    observed: list[str] = []
    real = producer.write_strict_json_atomic
    monkeypatch.setattr(
        producer,
        "write_strict_json_atomic",
        lambda path, payload, **kwargs: observed.append(Path(path).name)
        or real(path, payload, **kwargs),
    )
    paths = producer._publish_bundle(output_root=output, payloads=payloads)
    assert observed == [
        "amendment-v4.json",
        "cache-input-attestation.json",
        "model-input-attestation.json",
        "request-v6.json",
    ]
    assert paths["request"].name == "request-v6.json"


@pytest.mark.parametrize("failure_kind", ("directory_open", "directory_fsync"))
def test_request_post_link_durability_failure_returns_committed_bundle(
    producer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
):
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(producer, "R7_SEQUENCE_ROOT", output.resolve())
    payloads = {
        "amendment": {"schema": producer.AMENDMENT_SCHEMA, "status": "approved"},
        "cache": {"schema": producer.CACHE_ATTESTATION_SCHEMA, "status": "passed"},
        "model": {"schema": producer.MODEL_ATTESTATION_SCHEMA, "status": "passed"},
        "request": {"schema": producer.REQUEST_SCHEMA, "status": "prepared"},
    }
    parity = sys.modules[producer.write_strict_json_atomic.__module__]
    if failure_kind == "directory_open":
        real_open = parity.os.open
        directory_open_count = 0

        def fail_request_directory_open(path, flags):
            nonlocal directory_open_count
            if Path(path).resolve() == output.resolve():
                directory_open_count += 1
                if directory_open_count == 4:
                    raise OSError("injected request directory open failure")
            return real_open(path, flags)

        monkeypatch.setattr(parity.os, "open", fail_request_directory_open)
    else:
        real_fsync = parity.os.fsync
        directory_fsync_count = 0

        def fail_request_directory_fsync(descriptor):
            nonlocal directory_fsync_count
            descriptor_path = Path(f"/proc/self/fd/{descriptor}").resolve()
            if descriptor_path == output.resolve():
                directory_fsync_count += 1
                if directory_fsync_count == 4:
                    raise OSError("injected request directory fsync failure")
            return real_fsync(descriptor)

        monkeypatch.setattr(parity.os, "fsync", fail_request_directory_fsync)

    paths = producer._publish_bundle(output_root=output, payloads=payloads)

    assert paths["request"] == output / "request-v6.json"
    assert (
        producer._strict_json(paths["request"], owner="committed request")
        == payloads["request"]
    )
    assert set(path.name for path in output.iterdir()) == {
        "amendment-v4.json",
        "cache-input-attestation.json",
        "model-input-attestation.json",
        "request-v6.json",
    }


def test_request_pre_link_failure_raises_without_request_commit(
    producer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(producer, "R7_SEQUENCE_ROOT", output.resolve())
    payloads = {
        "amendment": {"schema": producer.AMENDMENT_SCHEMA, "status": "approved"},
        "cache": {"schema": producer.CACHE_ATTESTATION_SCHEMA, "status": "passed"},
        "model": {"schema": producer.MODEL_ATTESTATION_SCHEMA, "status": "passed"},
        "request": {"schema": producer.REQUEST_SCHEMA, "status": "prepared"},
    }
    parity = sys.modules[producer.write_strict_json_atomic.__module__]
    real_link = parity.os.link

    def fail_request_link(source, target):
        if Path(target).name == "request-v6.json":
            raise OSError("injected request pre-link failure")
        return real_link(source, target)

    monkeypatch.setattr(parity.os, "link", fail_request_link)

    with pytest.raises(OSError, match="request pre-link failure"):
        producer._publish_bundle(output_root=output, payloads=payloads)

    assert not (output / "request-v6.json").exists()
    assert {
        path.name for path in output.iterdir() if not path.name.startswith(".")
    } == {
        "amendment-v4.json",
        "cache-input-attestation.json",
        "model-input-attestation.json",
    }


@pytest.mark.parametrize(
    "encoded",
    (
        '{"schema":"first","schema":"second"}',
        '{"outer":{"status":"passed","status":"failed"}}',
    ),
)
def test_strict_json_rejects_duplicate_object_keys(
    producer, tmp_path: Path, encoded: str
):
    path = tmp_path / "duplicate.json"
    path.write_text(encoded, encoding="utf-8")

    with pytest.raises(producer.RequestAuthoringError, match="strict UTF-8 JSON"):
        producer._strict_json(path, owner="duplicate fixture")


def test_preexisting_leaf_rejects_before_any_publication(
    producer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(producer, "R7_SEQUENCE_ROOT", output.resolve())
    (output / "model-input-attestation.json").write_text("{}\n")
    with pytest.raises(Exception):
        producer._publish_bundle(
            output_root=output,
            payloads={
                "amendment": {},
                "cache": {},
                "model": {},
                "request": {},
            },
        )
    assert not (output / "amendment-v4.json").exists()
    assert not (output / "cache-input-attestation.json").exists()


def test_source_inventory_binds_model_weight_hash_helper_owner(producer):
    inventory = producer._source_inventory()
    by_path = {row["path"]: row for row in inventory}
    owner_path = str((REPO_ROOT / "src/artifacts/identity.py").resolve())
    assert by_path[owner_path]["sha256"] == (
        "ae155a6fb2cb613f217a5873093ebd8e8fe19d5bd0503bc5abde653bab96c816"
    )


def test_train_command_uses_module_mode_without_pythonpath_dependency(
    producer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(producer.shutil, "which", lambda _: "/opt/ms/bin/accelerate")
    config = tmp_path / "train.yaml"

    assert producer._train_command(config, port=29681) == [
        "/opt/ms/bin/accelerate",
        "launch",
        "--multi_gpu",
        "--num_processes",
        "8",
        "--main_process_port",
        "29681",
        "--module",
        "src.train",
        "--config",
        str(config.resolve()),
    ]


def test_interrupted_command_hashes_all_resolved_config_sources_in_exact_order(
    producer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(producer.shutil, "which", lambda _: "/opt/ms/bin/accelerate")
    overlay = tmp_path / "interrupted-parent.yaml"
    overlay.write_text(
        "\n".join(
            (
                "schema_version: 1",
                f"extends: {BASE_CONFIG.resolve()}",
                "run:",
                "  name: interrupted_parent",
                f"  artifact_root: {tmp_path / 'runs'}",
                "  collision_policy: fail",
                "",
            )
        ),
        encoding="utf-8",
    )
    configs = {role: overlay.resolve() for role in producer.RUN_ROLES}
    roots = {
        role: (tmp_path / "runs" / role).resolve() for role in producer.RUN_ROLES
    }
    targets = {
        name: (tmp_path / f"{name}.json").resolve() for name in producer.R7_TARGETS
    }

    commands = producer._expected_commands(
        configs=configs,
        roots=roots,
        targets=targets,
        phase_timeout_seconds=600,
        term_grace_seconds=30,
        kill_grace_seconds=30,
        baseline_stability_seconds=2,
        provenance_sha256="f" * 64,
    )

    outer = commands["interrupted_parent"]
    launcher_boundary = outer.index("--")
    outer_config_values = [
        outer[index + 1]
        for index, value in enumerate(outer[:launcher_boundary])
        if value == "--config"
    ]
    resolved = producer.load_train_config(overlay)
    assert outer_config_values == [str(source.path) for source in resolved.sources]


def test_exact_immutable_r5_failure_is_bound_as_historical_evidence(producer):
    before = IMMUTABLE_R5_SEQUENCE_RECEIPT.read_bytes()

    binding = producer._bind_predecessor_sequence_failure(IMMUTABLE_R5_SEQUENCE_RECEIPT)

    assert binding == {
        "path": str(IMMUTABLE_R5_SEQUENCE_RECEIPT.resolve()),
        "file_sha256": (
            "d5244fb6d96b24b1cfac4438628ae141496f7357c286dff5454c19c9f39731b1"
        ),
        "payload_sha256": (
            "f560cbd7bfeac8e25f031021d9d0053c3df0d69961d52f1b854754ce4f4e7656"
        ),
        "schema": "coordexp-swift-wave7-exact-resume-sequence-receipt-v4",
        "status": "failed",
    }
    assert IMMUTABLE_R5_SEQUENCE_RECEIPT.read_bytes() == before


@pytest.mark.parametrize("alias_kind", ("copy", "symlink", "alternate_path"))
def test_predecessor_r5_rejects_noncanonical_file_aliases(
    producer, tmp_path: Path, alias_kind: str
):
    alias = tmp_path / (
        "renamed-terminal.json"
        if alias_kind == "alternate_path"
        else "sequence-receipt.json"
    )
    if alias_kind == "copy":
        alias.write_bytes(IMMUTABLE_R5_SEQUENCE_RECEIPT.read_bytes())
    elif alias_kind == "symlink":
        alias.symlink_to(IMMUTABLE_R5_SEQUENCE_RECEIPT)
    else:
        alias.write_bytes(IMMUTABLE_R5_SEQUENCE_RECEIPT.read_bytes())

    with pytest.raises(producer.RequestAuthoringError, match="canonical"):
        producer._bind_predecessor_sequence_failure(alias)


def test_exact_immutable_r6_preflight_failure_is_bound_as_historical_evidence(
    producer,
):
    before = {
        path: path.read_bytes()
        for path in (
            IMMUTABLE_R6_PREFLIGHT_PLAN,
            IMMUTABLE_R6_PREFLIGHT_MARKER,
            IMMUTABLE_R6_PREFLIGHT_TERMINAL,
        )
    }

    binding = producer._bind_predecessor_preflight_failure(
        IMMUTABLE_R6_PREFLIGHT_PLAN,
        IMMUTABLE_R6_PREFLIGHT_MARKER,
        IMMUTABLE_R6_PREFLIGHT_TERMINAL,
    )

    assert binding == {
        "historical_non_executable": True,
        "plan": {
            "path": str(IMMUTABLE_R6_PREFLIGHT_PLAN.resolve()),
            "file_sha256": (
                "2d5d5108c07e02ff964f5b3e6e4e42ce84e58bd0c17a8310dd7e2f3e6147e6ec"
            ),
            "payload_sha256": (
                "9b5c820c5f2c366c994a15732b99498b4072d1d552d9a7048c6afb65e0532a8f"
            ),
            "schema": "coordexp-swift-wave7-determinism-preflight-plan-v4",
            "status": "prepared",
        },
        "attempt_marker": {
            "path": str(IMMUTABLE_R6_PREFLIGHT_MARKER.resolve()),
            "file_sha256": (
                "f19a894ac3113aecc3a126dab7e3327389518c9e12e34d7806ad401adf239ce9"
            ),
            "payload_sha256": (
                "967fc87daabc19dd82d9d5a0f2567a2700da71bf9cb3773f904da63488585d7e"
            ),
            "schema": (
                "coordexp-swift-wave7-determinism-preflight-attempt-start-marker-v4"
            ),
            "status": "started",
        },
        "terminal_receipt": {
            "path": str(IMMUTABLE_R6_PREFLIGHT_TERMINAL.resolve()),
            "file_sha256": (
                "eaf20c86a3fa95d7fa6533faaf7fb0e9b8fe938aa6f62d2ba725c3c604a64784"
            ),
            "payload_sha256": (
                "4b400e094c494099b611e85d78b2156ce53b0fe372e12de6ff780c493b5770ab"
            ),
            "schema": "coordexp-swift-wave7-r5-determinism-preflight-v4",
            "status": "failed",
        },
    }
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("leaf", ("plan", "marker", "terminal"))
def test_predecessor_r6_preflight_rejects_noncanonical_aliases(
    producer, tmp_path: Path, leaf: str
):
    paths = {
        "plan": IMMUTABLE_R6_PREFLIGHT_PLAN,
        "marker": IMMUTABLE_R6_PREFLIGHT_MARKER,
        "terminal": IMMUTABLE_R6_PREFLIGHT_TERMINAL,
    }
    aliased = dict(paths)
    alias = tmp_path / paths[leaf].name
    alias.write_bytes(paths[leaf].read_bytes())
    aliased[leaf] = alias

    with pytest.raises(producer.RequestAuthoringError, match="canonical"):
        producer._bind_predecessor_preflight_failure(
            aliased["plan"], aliased["marker"], aliased["terminal"]
        )


@pytest.mark.parametrize(
    ("leaf", "field", "value"),
    (
        ("plan", "status", "passed"),
        ("marker", "status", "prepared"),
        ("terminal", "status", "passed"),
        ("terminal", "launch_count", 1),
        ("terminal", "mismatches", []),
    ),
)
def test_resigned_r6_preflight_projection_mutations_are_rejected(
    producer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    leaf: str,
    field: str,
    value: object,
):
    source = {
        "plan": IMMUTABLE_R6_PREFLIGHT_PLAN,
        "marker": IMMUTABLE_R6_PREFLIGHT_MARKER,
        "terminal": IMMUTABLE_R6_PREFLIGHT_TERMINAL,
    }[leaf]
    payload = json.loads(source.read_text(encoding="utf-8"))
    digest_field = "plan_payload_sha256" if leaf == "plan" else "receipt_payload_sha256"
    payload.pop(digest_field)
    payload[field] = value
    payload[digest_field] = _sha256(_canonical(payload))
    mutated = tmp_path / source.name
    mutated.write_bytes(_canonical(payload) + b"\n")
    path_name = {
        "plan": "FROZEN_R6_PREFLIGHT_PLAN_PATH",
        "marker": "FROZEN_R6_PREFLIGHT_MARKER_PATH",
        "terminal": "FROZEN_R6_PREFLIGHT_TERMINAL_PATH",
    }[leaf]
    file_name = path_name.replace("_PATH", "_FILE_SHA256")
    monkeypatch.setattr(producer, path_name, mutated.resolve())
    monkeypatch.setattr(producer, file_name, _sha256(mutated.read_bytes()))
    paths = {
        "plan": IMMUTABLE_R6_PREFLIGHT_PLAN,
        "marker": IMMUTABLE_R6_PREFLIGHT_MARKER,
        "terminal": IMMUTABLE_R6_PREFLIGHT_TERMINAL,
    }
    paths[leaf] = mutated

    with pytest.raises(producer.RequestAuthoringError, match="predecessor r6"):
        producer._bind_predecessor_preflight_failure(
            paths["plan"], paths["marker"], paths["terminal"]
        )


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
def test_resigned_r5_projection_mutations_are_rejected(
    producer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
):
    payload = json.loads(IMMUTABLE_R5_SEQUENCE_RECEIPT.read_text(encoding="utf-8"))
    payload.pop("receipt_payload_sha256")
    if mutation == "schema":
        payload["schema"] = producer.REQUEST_SCHEMA
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

    path = tmp_path / f"r5-{mutation}.json"
    signed = _write_signed_json(path, payload)
    monkeypatch.setattr(
        producer, "FROZEN_R5_FAILURE_PATH", path.resolve(), raising=False
    )
    monkeypatch.setattr(
        producer, "FROZEN_R5_FAILURE_FILE_SHA256", _sha256(path.read_bytes())
    )
    monkeypatch.setattr(
        producer,
        "FROZEN_R5_FAILURE_PAYLOAD_SHA256",
        signed["receipt_payload_sha256"],
    )

    with pytest.raises(producer.RequestAuthoringError, match="predecessor r5"):
        producer._bind_predecessor_sequence_failure(path)


def test_request_v6_binds_execution_relevant_provenance_for_comparators(
    producer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(producer, "_validate_configs_and_roots", lambda *_: None)
    monkeypatch.setattr(producer, "_validate_path_topology", lambda *_: None)
    monkeypatch.setattr(producer, "_validate_r7_request_namespace", lambda *_: None)
    monkeypatch.setattr(
        producer,
        "_bind_external_receipt",
        lambda *_, owner, **__: {
            "path": f"/{owner}.json",
            "file_sha256": "1" * 64,
            "payload_sha256": "2" * 64,
            "schema": owner,
            "status": "passed",
        },
    )
    predecessor = {
        "path": "/r5.json",
        "file_sha256": "3" * 64,
        "payload_sha256": "4" * 64,
        "schema": "coordexp-swift-wave7-exact-resume-sequence-receipt-v4",
        "status": "failed",
    }
    monkeypatch.setattr(
        producer, "_bind_predecessor_sequence_failure", lambda _: predecessor
    )
    predecessor_preflight = {
        "historical_non_executable": True,
        "plan": {"path": "/r6/plan.json"},
        "attempt_marker": {"path": "/r6/marker.json"},
        "terminal_receipt": {"path": "/r6/terminal.json"},
    }
    monkeypatch.setattr(
        producer,
        "_bind_predecessor_preflight_failure",
        lambda *_: predecessor_preflight,
    )
    execution_digest = "8" * 64
    provenance = {
        "repository": {
            "execution_relevant_digest": {
                "status": "available",
                "value": execution_digest,
            }
        }
    }
    monkeypatch.setattr(
        producer,
        "_strict_json",
        lambda *_, **__: {
            "model_loaded": False,
            "cuda_initialized": False,
            "provenance": provenance,
        },
    )
    monkeypatch.setattr(producer, "_source_inventory", lambda: [])
    command_kwargs = {}

    def capture_commands(**kwargs):
        command_kwargs.update(kwargs)
        return {}

    monkeypatch.setattr(producer, "_expected_commands", capture_commands)
    args = argparse.Namespace(
        uninterrupted_config="/uninterrupted.yaml",
        interrupted_parent_config="/parent.yaml",
        resume_child_config="/child.yaml",
        uninterrupted_run_root="/r7/uninterrupted",
        interrupted_parent_run_root="/r7/parent",
        resume_child_run_root="/r7/child",
        sequence_marker="/r7/marker.json",
        sequence_receipt="/r7/receipt.json",
        publication_failure_sidecar="/r7/sidecar.json",
        interruption_marker="/r7/interruption-marker.json",
        interruption_receipt="/r7/interruption-receipt.json",
        pre_child_receipt="/r7/pre-child.json",
        final_receipt="/r7/final.json",
        legacy_r4_failure="/r4.json",
        predecessor_r5_failure="/r5.json",
        predecessor_r6_preflight_plan="/r6/preflight-plan.json",
        predecessor_r6_preflight_marker="/r6/preflight-marker.json",
        predecessor_r6_preflight_terminal="/r6/preflight-terminal.json",
        determinism_preflight_plan="/preflight-plan.json",
        determinism_preflight_receipt="/preflight.json",
        runtime_receipt="/runtime.json",
        cache_root="/r7/cache",
        phase_timeout_seconds=600,
        term_grace_seconds=10,
        kill_grace_seconds=10,
        gpu_baseline_stability_seconds=2,
        gpu_post_cleanup_stability_seconds=2,
        max_total_wall_seconds=2400,
        max_total_gpu_device_seconds=14400,
    )
    amendment = {
        "schema": producer.AMENDMENT_SCHEMA,
        "status": "approved",
        "amendment_sha256": "5" * 64,
    }
    cache = {
        "schema": producer.CACHE_ATTESTATION_SCHEMA,
        "status": "passed",
        "attestation_sha256": "6" * 64,
    }
    model = {
        "schema": producer.MODEL_ATTESTATION_SCHEMA,
        "status": "passed",
        "attestation_sha256": "7" * 64,
    }
    leaves = {
        "amendment": tmp_path / "amendment-v4.json",
        "cache": tmp_path / "cache.json",
        "model": tmp_path / "model.json",
    }

    request = producer._request_payload(
        args, amendment=amendment, cache=cache, model=model, leaf_paths=leaves
    )

    assert command_kwargs["provenance_sha256"] == execution_digest
    assert request["predecessor_sequence_failure"] == predecessor
    assert request["predecessor_preflight_failure"] == predecessor_preflight
    assert set(request) == {
        "schema",
        "status",
        "amendment",
        "legacy_r4_failure",
        "predecessor_sequence_failure",
        "predecessor_preflight_failure",
        "determinism_preflight_plan",
        "determinism_preflight",
        "runtime_receipt",
        "cache_input_attestation",
        "model_input_attestation",
        "source_inventory",
        "config_paths",
        "provenance_sha256",
        "environment",
        "run_roots",
        "targets",
        "commands",
        "oracles",
        "policy",
        "request_payload_sha256",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("phase_timeout_seconds", 600.001),
        ("phase_timeout_seconds", 601),
        ("max_total_wall_seconds", 2400.001),
        ("max_total_wall_seconds", 2401),
        ("max_total_gpu_device_seconds", 14400.001),
        ("max_total_gpu_device_seconds", 14401),
    ),
)
def test_r7_cost_authority_rejects_excess(producer, field: str, value: float):
    args = argparse.Namespace(
        phase_timeout_seconds=600,
        term_grace_seconds=10,
        kill_grace_seconds=10,
        gpu_baseline_stability_seconds=2,
        gpu_post_cleanup_stability_seconds=2,
        max_total_wall_seconds=2400,
        max_total_gpu_device_seconds=14400,
    )
    setattr(args, field, value)

    with pytest.raises(producer.RequestAuthoringError, match="r7 authority"):
        producer._validate_authorized_costs(args)


def test_r7_cost_authority_accepts_exact_boundaries(producer):
    args = argparse.Namespace(
        phase_timeout_seconds=600,
        term_grace_seconds=10,
        kill_grace_seconds=10,
        gpu_baseline_stability_seconds=2,
        gpu_post_cleanup_stability_seconds=2,
        max_total_wall_seconds=2400,
        max_total_gpu_device_seconds=14400,
    )

    producer._validate_authorized_costs(args)

    assert args.phase_timeout_seconds == 600.0
    assert args.max_total_wall_seconds == 2400.0
    assert args.max_total_gpu_device_seconds == 14400.0


def _exact_r7_namespace(producer):
    configs = dict(producer.R7_CONFIG_PATHS)
    roots = dict(producer.R7_RUN_ROOTS)
    targets = dict(producer.R7_TARGETS)
    args = argparse.Namespace(
        cache_root=str(producer.R7_CACHE_ROOT),
        cache_preparation_receipt=str(producer.R7_CACHE_PREPARATION_RECEIPT),
        determinism_preflight_plan=str(producer.R7_DETERMINISM_PREFLIGHT_PLAN),
        determinism_preflight_receipt=str(producer.R7_DETERMINISM_PREFLIGHT_RECEIPT),
        runtime_receipt=str(producer.R7_RUNTIME_RECEIPT),
    )
    for role, path in configs.items():
        setattr(args, f"{role}_config", str(path))
    for role, path in roots.items():
        setattr(args, f"{role}_run_root", str(path))
    for name, path in targets.items():
        setattr(args, name, str(path))
    return args, configs, roots, targets


def test_exact_fresh_r7_namespace_is_accepted(producer):
    args, configs, roots, targets = _exact_r7_namespace(producer)

    producer._validate_r7_request_namespace(args, configs, roots, targets)


@pytest.mark.parametrize(
    "mutation",
    (
        "r6_cache_root",
        "alternate_date_config",
        "aliased_config",
        "r6_run_root",
        "aliased_target",
        "r6_runtime",
        "r6_preflight_plan",
        "r6_preflight_receipt",
        "alternate_cache_receipt",
    ),
)
def test_r6_or_alternate_namespace_is_rejected_before_request_authoring(
    producer, mutation: str
):
    args, configs, roots, targets = _exact_r7_namespace(producer)
    if mutation == "r6_cache_root":
        args.cache_root = str(producer.R7_CACHE_ROOT).replace(
            "wave7-core-4", "wave7-r6"
        )
    elif mutation == "alternate_date_config":
        configs["uninterrupted"] = Path(
            str(configs["uninterrupted"]).replace("2026-08-12-core-4", "2026-08-11-r7")
        )
    elif mutation == "aliased_config":
        configs["resume_child"] = producer.R7_SEQUENCE_ROOT / "configs/child.yaml"
    elif mutation == "r6_run_root":
        roots["interrupted_parent"] = Path(
            str(roots["interrupted_parent"]).replace(
                "2026-08-12-core-4", "2026-08-11-r6"
            )
        )
    elif mutation == "aliased_target":
        targets["sequence_marker"] = producer.R7_SEQUENCE_ROOT / "attempt.json"
    elif mutation == "r6_runtime":
        args.runtime_receipt = str(producer.R7_RUNTIME_RECEIPT).replace(
            "2026-08-12-core-4", "2026-08-11-r6"
        )
    elif mutation == "r6_preflight_plan":
        args.determinism_preflight_plan = str(
            producer.R7_DETERMINISM_PREFLIGHT_PLAN
        ).replace("2026-08-12-core-4", "2026-08-11-r6")
    elif mutation == "r6_preflight_receipt":
        args.determinism_preflight_receipt = str(
            producer.R7_DETERMINISM_PREFLIGHT_RECEIPT
        ).replace("2026-08-12-core-4", "2026-08-11-r6")
    else:
        args.cache_preparation_receipt = str(
            producer.R7_CACHE_ROOT / "alternate-preparation.json"
        )

    with pytest.raises(producer.RequestAuthoringError, match="fresh r7 namespace"):
        producer._validate_r7_request_namespace(args, configs, roots, targets)


def test_publication_rejects_alternate_output_root_before_writing(
    producer, tmp_path: Path
):
    alternate = tmp_path / "2026-08-11-r6"
    alternate.mkdir()

    with pytest.raises(producer.RequestAuthoringError, match="fresh r7 namespace"):
        producer._publish_bundle(
            output_root=alternate,
            payloads={name: {} for name in producer.LEAF_NAMES},
        )

    assert list(alternate.iterdir()) == []
