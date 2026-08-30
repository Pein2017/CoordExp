from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import MappingProxyType

import pytest

import src.artifacts as artifact_package
import src.artifacts.research_probe_admission as admission_module
from src.artifacts.evidence_journal import (
    JOURNAL_SCHEMA_VERSION,
    ExecutionEvidenceJournal,
)
from src.artifacts.json_values import json_sha256, load_canonical_json
from src.artifacts.research_probe_admission import (
    AbsoluteExecutableBinding,
    DirectoryTreeBinding,
    RegularFileBinding,
    ResearchProbeAdmission,
    ResearchProbeAdmissionError,
    ReservedOutputPath,
    ResolvedDataFileBinding,
    StageEvidence,
    StrictValueBinding,
    TargetTreeBinding,
    capture_target_tree_binding,
    capture_binding_manifest,
    revalidate_target_tree_binding,
    revalidate_binding_manifest,
)
from src.common.errors import ArtifactContractError
from src.inference.execution_context import (
    prepare_execution_context_payload,
    publish_execution_context_artifact,
    validate_execution_context_identity,
    verify_local_execution_context_artifact,
)


def _write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _fixture_manifest(tmp_path: Path):
    producer = _write(tmp_path / "source" / "producer.py", "producer-v1\n")
    validator = _write(tmp_path / "source" / "validator.py", "validator-v1\n")
    runtime = tmp_path / "runtime"
    _write(runtime / "z" / "tail.txt", "tail\n")
    _write(runtime / "a.txt", "head\n")
    data_base = tmp_path / "data"
    image = _write(data_base / "images" / "one.bin", "pixels\n")
    executable = _write(tmp_path / "bin" / "python", "#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    manifest = capture_binding_manifest(
        [
            RegularFileBinding("producer", producer),
            RegularFileBinding("validator", validator),
            DirectoryTreeBinding("runtime", runtime),
            ResolvedDataFileBinding("image", Path("images/one.bin"), data_base),
            AbsoluteExecutableBinding("python", executable.resolve()),
            StrictValueBinding("policy", {"mode": "bounded", "count": 1}),
        ]
    )
    return manifest, producer, validator, runtime, image


def _run_git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _clean_target_worktree(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "target"
    root.mkdir(parents=True)
    _run_git(root, "init", "--initial-branch=main")
    _run_git(root, "config", "user.email", "fixture@example.invalid")
    _run_git(root, "config", "user.name", "Fixture")
    source = _write(root / "source.py", "source-v1\n")
    config = _write(tmp_path / "runtime.json", '{"version":1}\n')
    _run_git(root, "add", "source.py")
    _run_git(root, "commit", "-m", "fixture")
    return root, source, config


def _reserved(tmp_path: Path) -> tuple[ReservedOutputPath, ...]:
    return (
        ReservedOutputPath("cpu", (tmp_path / "reserved" / "cpu").resolve()),
        ReservedOutputPath("vertical", (tmp_path / "reserved" / "vertical").resolve()),
    )


def _cpu_assertions() -> dict[str, bool]:
    return {
        "production_entrypoint_resolved": True,
        "consumer_validator_ran": True,
        "model_free_finalizer_ran": True,
        "downstream_validator_accepted": True,
        "model_loaded": False,
        "gpu_used": False,
    }


def _vertical_assertions() -> dict[str, bool | int]:
    return {
        "production_entrypoint_executed": True,
        "model_runtime_loaded": True,
        "durable_work_item_count": 1,
        "terminal_finalizer_completed": True,
        "downstream_validator_accepted": True,
    }


def _evidence(output: Path, assertions: dict[str, object]) -> StageEvidence:
    return StageEvidence(
        producer_binding="producer",
        validator_binding="validator",
        output_files=(RegularFileBinding("receipt", output),),
        assertions=assertions,
        detail={"consumer": "fixture"},
    )


def _append_both(admission: ResearchProbeAdmission, tmp_path: Path) -> None:
    attempt = admission.start_attempt()
    cpu_output = _write(tmp_path / "reserved" / "cpu" / "receipt.json", '{"ok":true}\n')
    admission.append_stage(
        stage="cpu_preflight",
        evidence=_evidence(cpu_output, _cpu_assertions()),
        attempt_id=attempt,
    )
    vertical_output = _write(
        tmp_path / "reserved" / "vertical" / "receipt.json", '{"durable":1}\n'
    )
    admission.append_stage(
        stage="vertical_smoke",
        evidence=_evidence(vertical_output, _vertical_assertions()),
        attempt_id=attempt,
    )


def test_capture_all_binding_kinds_and_deterministic_full_tree(tmp_path: Path) -> None:
    manifest, _, _, runtime, image = _fixture_manifest(tmp_path)
    mapping = manifest.to_mapping()

    assert [item["kind"] for item in mapping["bindings"]] == [
        "regular_file",
        "regular_file",
        "directory_tree",
        "resolved_data_file",
        "absolute_executable",
        "strict_value",
    ]
    tree = mapping["bindings"][2]
    assert [item["relative_path"] for item in tree["files"]] == [
        "a.txt",
        "z/tail.txt",
    ]
    assert all(len(item["sha256"]) == 64 for item in tree["files"])
    assert len(tree["inventory_sha256"]) == 64
    resolved = mapping["bindings"][3]
    assert resolved["declared_path"] == "images/one.bin"
    assert resolved["base_directory"] == str(image.parent.parent.resolve())
    assert resolved["resolved_file"]["path"] == str(image.resolve())
    assert manifest == capture_binding_manifest(
        [
            RegularFileBinding("producer", tmp_path / "source" / "producer.py"),
            RegularFileBinding("validator", tmp_path / "source" / "validator.py"),
            DirectoryTreeBinding("runtime", runtime),
            ResolvedDataFileBinding("image", "images/one.bin", tmp_path / "data"),
            AbsoluteExecutableBinding(
                "python", (tmp_path / "bin" / "python").resolve()
            ),
            StrictValueBinding("policy", {"count": 1, "mode": "bounded"}),
        ]
    )
    assert isinstance(manifest.bindings[0], MappingProxyType)
    with pytest.raises(TypeError):
        manifest.bindings[0]["sha256"] = "changed"  # type: ignore[index]


@pytest.mark.parametrize(
    "bad_request",
    [
        lambda root: AbsoluteExecutableBinding("python", Path("python")),
        lambda root: AbsoluteExecutableBinding(
            "python", _write(root / "not-executable", "x").resolve()
        ),
        lambda root: RegularFileBinding("file", (root / "file-link")),
        lambda root: DirectoryTreeBinding("tree", root / "tree"),
    ],
)
def test_path_binding_failures_are_typed_and_fail_closed(
    tmp_path: Path, bad_request
) -> None:
    target = _write(tmp_path / "target", "target")
    (tmp_path / "file-link").symlink_to(target)
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "link").symlink_to(target)

    with pytest.raises(ResearchProbeAdmissionError):
        capture_binding_manifest([bad_request(tmp_path)])


def test_strict_live_objects_duplicates_and_context_fail_before_root(
    tmp_path: Path,
) -> None:
    with pytest.raises(ResearchProbeAdmissionError):
        StrictValueBinding("callback", lambda: None)

    file_path = _write(tmp_path / "source.py", "x")
    with pytest.raises(ResearchProbeAdmissionError) as duplicate:
        capture_binding_manifest(
            [
                RegularFileBinding("same", file_path),
                StrictValueBinding("same", 1),
            ]
        )
    assert duplicate.value.code == "admission.duplicate_binding_name"

    manifest = capture_binding_manifest([RegularFileBinding("source", file_path)])
    root = tmp_path / "admission"
    with pytest.raises(ResearchProbeAdmissionError):
        ResearchProbeAdmission.create(
            root=root,
            admission_id="case",
            bindings=manifest,
            reserved_output_paths=_reserved(tmp_path),
            context={"callback": lambda: None},
        )
    assert not root.exists()

    invalid_contexts = (
        {Path("non-string-key"): 1},
        {"tuple-is-not-json": (1,)},
    )
    for index, invalid_context in enumerate(invalid_contexts):
        invalid_root = tmp_path / f"invalid-context-{index}"
        with pytest.raises(ResearchProbeAdmissionError) as invalid:
            ResearchProbeAdmission.create(
                root=invalid_root,
                admission_id="case",
                bindings=manifest,
                reserved_output_paths=_reserved(tmp_path),
                context=invalid_context,
            )
        assert invalid.value.code == "admission.invalid_strict_value"
        assert not invalid_root.exists()


def test_live_revalidation_detects_every_file_and_tree_drift(tmp_path: Path) -> None:
    manifest, producer, _, runtime, _ = _fixture_manifest(tmp_path)
    revalidate_binding_manifest(manifest)

    producer.write_text("producer-v2\n", encoding="utf-8")
    with pytest.raises(ResearchProbeAdmissionError) as file_drift:
        revalidate_binding_manifest(manifest)
    assert file_drift.value.code == "admission.binding_drift"

    producer.write_text("producer-v1\n", encoding="utf-8")
    _write(runtime / "new.txt", "new")
    with pytest.raises(ResearchProbeAdmissionError) as tree_drift:
        revalidate_binding_manifest(manifest)
    assert tree_drift.value.code == "admission.binding_drift"


def test_target_tree_capture_revalidation_and_typed_dirty_rejections(
    tmp_path: Path,
) -> None:
    root, source, runtime = _clean_target_worktree(tmp_path)
    manifest = capture_binding_manifest(
        (
            RegularFileBinding("target_source", source),
            RegularFileBinding("runtime_config", runtime),
        )
    )
    target = capture_target_tree_binding(
        TargetTreeBinding(
            root=root,
            effective_binding_names=("target_source", "runtime_config"),
        ),
        manifest,
    )
    mapping = target.to_mapping()
    assert mapping["root"] == str(root.resolve())
    assert mapping["commit"] == _run_git(root, "rev-parse", "HEAD")
    assert mapping["clean"] is True
    assert [entry["name"] for entry in mapping["effective_inputs"]] == [
        "target_source",
        "runtime_config",
    ]
    revalidate_target_tree_binding(target)

    runtime.write_text('{"version":2}\n', encoding="utf-8")
    with pytest.raises(ResearchProbeAdmissionError) as effective_drift:
        revalidate_target_tree_binding(target)
    assert effective_drift.value.code == "admission.target_tree_drift"

    runtime.write_text('{"version":1}\n', encoding="utf-8")
    _run_git(root, "commit", "--allow-empty", "-m", "new-head")
    with pytest.raises(ResearchProbeAdmissionError) as commit_drift:
        revalidate_target_tree_binding(target)
    assert commit_drift.value.code == "admission.target_tree_drift"


def test_admission_persists_target_tree_as_outer_journal_identity(
    tmp_path: Path,
) -> None:
    target_root, source, runtime = _clean_target_worktree(tmp_path / "fixture")
    manifest = capture_binding_manifest(
        (
            RegularFileBinding("producer", source),
            RegularFileBinding("validator", runtime),
        )
    )
    target = TargetTreeBinding(
        root=target_root,
        effective_binding_names=("producer", "validator"),
    )
    root = tmp_path / "admission"
    admission = ResearchProbeAdmission.create(
        root=root,
        admission_id="target-outer",
        bindings=manifest,
        target_tree=target,
        reserved_output_paths=_reserved(tmp_path),
        context={"fixture": "target-tree"},
    )
    admission.close()

    plan = load_canonical_json(root / "journal" / "plan.json")
    target_mapping = plan["execution_identity"]["target_tree"]
    assert target_mapping["root"] == str(target_root.resolve())
    assert (
        target_mapping["effective_inputs"]
        == capture_target_tree_binding(target, manifest).to_mapping()[
            "effective_inputs"
        ]
    )

    reopened = ResearchProbeAdmission.open(
        root=root,
        admission_id="target-outer",
        bindings=manifest,
        target_tree=target,
        reserved_output_paths=_reserved(tmp_path),
        context={"fixture": "target-tree"},
    )
    reopened.revalidate_target_tree()
    reopened.close()


def test_inspect_does_not_revalidate_live_target_tree_after_completion(
    tmp_path: Path,
) -> None:
    target_root, source, runtime = _clean_target_worktree(tmp_path / "fixture")
    manifest = capture_binding_manifest(
        (
            RegularFileBinding("producer", source),
            RegularFileBinding("validator", runtime),
        )
    )
    target = TargetTreeBinding(
        root=target_root,
        effective_binding_names=("producer", "validator"),
    )
    root = tmp_path / "admission"
    admission = ResearchProbeAdmission.create(
        root=root,
        admission_id="target-inspect",
        bindings=manifest,
        target_tree=target,
        reserved_output_paths=_reserved(tmp_path),
        context={"fixture": "persisted-inspection"},
    )
    _append_both(admission, tmp_path)
    admission.finalize()
    admission.close()

    _write(target_root / "unrelated-untracked.py", "unrelated\n")

    inspection = ResearchProbeAdmission.inspect(root)
    assert inspection.completed_stages == ("cpu_preflight", "vertical_smoke")
    assert inspection.mechanically_admitted is True


@pytest.mark.parametrize(
    "mutate",
    (
        lambda root, source: (
            source.write_text("staged\n", encoding="utf-8"),
            _run_git(root, "add", "source.py"),
        ),
        lambda root, source: source.write_text("modified\n", encoding="utf-8"),
        lambda root, source: source.unlink(),
        lambda root, source: _run_git(root, "mv", "source.py", "renamed.py"),
        lambda root, source: _write(root / "untracked.py", "untracked\n"),
    ),
)
def test_target_tree_capture_rejects_each_dirty_status(tmp_path: Path, mutate) -> None:
    root, source, runtime = _clean_target_worktree(tmp_path)
    manifest = capture_binding_manifest(
        (
            RegularFileBinding("target_source", source),
            RegularFileBinding("runtime_config", runtime),
        )
    )
    mutate(root, source)

    with pytest.raises(ResearchProbeAdmissionError) as rejected:
        capture_target_tree_binding(
            TargetTreeBinding(
                root=root,
                effective_binding_names=("target_source", "runtime_config"),
            ),
            manifest,
        )
    assert rejected.value.code == "admission.target_tree_dirty"


def test_target_tree_capture_rejects_nonworktree_symlink_and_conflict(
    tmp_path: Path,
) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    source = _write(plain / "source.py", "source\n")
    runtime = _write(tmp_path / "runtime.json", "{}\n")
    manifest = capture_binding_manifest(
        (
            RegularFileBinding("target_source", source),
            RegularFileBinding("runtime_config", runtime),
        )
    )
    with pytest.raises(ResearchProbeAdmissionError) as nonworktree:
        capture_target_tree_binding(
            TargetTreeBinding(plain, ("target_source", "runtime_config")), manifest
        )
    assert nonworktree.value.code == "admission.target_tree_not_worktree"

    with pytest.raises(ResearchProbeAdmissionError) as unresolved:
        capture_target_tree_binding(
            TargetTreeBinding(
                tmp_path / "missing-target",
                ("target_source", "runtime_config"),
            ),
            manifest,
        )
    assert unresolved.value.code == "admission.target_tree_not_worktree"

    root, source, runtime = _clean_target_worktree(tmp_path / "symlink")
    link = tmp_path / "target-link"
    link.symlink_to(root, target_is_directory=True)
    manifest = capture_binding_manifest(
        (
            RegularFileBinding("target_source", source),
            RegularFileBinding("runtime_config", runtime),
        )
    )
    with pytest.raises(ResearchProbeAdmissionError) as symlink:
        capture_target_tree_binding(
            TargetTreeBinding(link, ("target_source", "runtime_config")), manifest
        )
    assert symlink.value.code == "admission.target_tree_symlink"

    root, source, runtime = _clean_target_worktree(tmp_path / "conflict")
    _run_git(root, "checkout", "-b", "other")
    source.write_text("other\n", encoding="utf-8")
    _run_git(root, "commit", "-am", "other")
    _run_git(root, "checkout", "main")
    source.write_text("main\n", encoding="utf-8")
    _run_git(root, "commit", "-am", "main")
    merged = subprocess.run(
        ["git", "-C", str(root), "merge", "other"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert merged.returncode != 0
    manifest = capture_binding_manifest(
        (
            RegularFileBinding("target_source", source),
            RegularFileBinding("runtime_config", runtime),
        )
    )
    with pytest.raises(ResearchProbeAdmissionError) as conflict:
        capture_target_tree_binding(
            TargetTreeBinding(root, ("target_source", "runtime_config")), manifest
        )
    assert conflict.value.code == "admission.target_tree_conflicted"


def test_reserved_paths_are_absolute_unique_absent_and_never_cleaned(
    tmp_path: Path,
) -> None:
    manifest, *_ = _fixture_manifest(tmp_path)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    root = tmp_path / "admission"

    with pytest.raises(ResearchProbeAdmissionError) as error:
        ResearchProbeAdmission.create(
            root=root,
            admission_id="case",
            bindings=manifest,
            reserved_output_paths=(ReservedOutputPath("occupied", occupied.resolve()),),
            context={},
        )
    assert error.value.code == "admission.reserved_path_occupied"
    assert occupied.is_dir()
    assert not root.exists()

    with pytest.raises(ResearchProbeAdmissionError):
        ResearchProbeAdmission.create(
            root=root,
            admission_id="case",
            bindings=manifest,
            reserved_output_paths=(ReservedOutputPath("relative", Path("relative")),),
            context={},
        )
    assert not root.exists()


def test_create_publishes_exact_two_stage_journal_without_occupying_reserved(
    tmp_path: Path,
) -> None:
    manifest, *_ = _fixture_manifest(tmp_path)
    reserved = _reserved(tmp_path)
    root = tmp_path / "admission"
    with ResearchProbeAdmission.create(
        root=root,
        admission_id="case",
        bindings=manifest,
        reserved_output_paths=reserved,
        context={"purpose": "test"},
    ) as admission:
        snapshot = ExecutionEvidenceJournal.inspect(root / "journal")
        assert snapshot.expected_work_item_ids == (
            "cpu_preflight",
            "vertical_smoke",
        )
        assert admission.missing_stages == snapshot.expected_work_item_ids
        plan = load_canonical_json(root / "journal" / "plan.json")
        assert plan["journal_schema_version"] == 1
    assert all(not request.path.exists() for request in reserved)


def test_stage_order_fixed_assertions_and_binding_closure_fail_before_append(
    tmp_path: Path,
) -> None:
    manifest, *_ = _fixture_manifest(tmp_path)
    root = tmp_path / "admission"
    output = _write(tmp_path / "output.json", "{}")
    with ResearchProbeAdmission.create(
        root=root,
        admission_id="case",
        bindings=manifest,
        reserved_output_paths=_reserved(tmp_path),
        context={},
    ) as admission:
        attempt = admission.start_attempt()
        with pytest.raises(ResearchProbeAdmissionError) as order:
            admission.append_stage(
                stage="vertical_smoke",
                evidence=_evidence(output, _vertical_assertions()),
                attempt_id=attempt,
            )
        assert order.value.code == "admission.invalid_stage_order"

        invalid = _cpu_assertions()
        invalid["gpu_used"] = True
        with pytest.raises(ResearchProbeAdmissionError) as assertions:
            admission.append_stage(
                stage="cpu_preflight",
                evidence=_evidence(output, invalid),
                attempt_id=attempt,
            )
        assert assertions.value.code == "admission.invalid_stage_assertions"

        missing_binding = StageEvidence(
            producer_binding="helper",
            validator_binding="validator",
            output_files=(RegularFileBinding("output", output),),
            assertions=_cpu_assertions(),
            detail={},
        )
        with pytest.raises(ResearchProbeAdmissionError) as binding:
            admission.append_stage(
                stage="cpu_preflight",
                evidence=missing_binding,
                attempt_id=attempt,
            )
        assert binding.value.code == "admission.missing_stage_binding"
        assert (
            ExecutionEvidenceJournal.inspect(root / "journal").completed_work_item_ids
            == ()
        )


def test_preexisting_foreign_input_cannot_satisfy_stage_output_closure(
    tmp_path: Path,
) -> None:
    manifest, producer, *_ = _fixture_manifest(tmp_path)
    root = tmp_path / "admission"
    with ResearchProbeAdmission.create(
        root=root,
        admission_id="case",
        bindings=manifest,
        reserved_output_paths=_reserved(tmp_path),
        context={},
    ) as admission:
        attempt = admission.start_attempt()
        with pytest.raises(ResearchProbeAdmissionError) as foreign_output:
            admission.append_stage(
                stage="cpu_preflight",
                evidence=_evidence(producer, _cpu_assertions()),
                attempt_id=attempt,
            )
        assert foreign_output.value.code == ("admission.output_outside_reserved_paths")
        assert (
            ExecutionEvidenceJournal.inspect(root / "journal").completed_work_item_ids
            == ()
        )


def test_cpu_stage_survives_vertical_failure_and_exact_continuation(
    tmp_path: Path,
) -> None:
    manifest, *_ = _fixture_manifest(tmp_path)
    reserved = _reserved(tmp_path)
    root = tmp_path / "admission"
    admission = ResearchProbeAdmission.create(
        root=root,
        admission_id="case",
        bindings=manifest,
        reserved_output_paths=reserved,
        context={"purpose": "continuation"},
    )
    cpu_output = _write(reserved[0].path, '{"ok":true}')
    first_attempt = admission.start_attempt()
    admission.append_stage(
        stage="cpu_preflight",
        evidence=_evidence(cpu_output, _cpu_assertions()),
        attempt_id=first_attempt,
    )
    admission.record_attempt_failure(
        attempt_id=first_attempt,
        failure_code="vertical.process_exit",
        failure_message="vertical process exited",
    )
    admission.close()

    reserved[1].path.mkdir(parents=True)
    vertical_output = _write(reserved[1].path / "receipt.json", '{"durable":1}')
    continued = ResearchProbeAdmission.open(
        root=root,
        admission_id="case",
        bindings=manifest,
        reserved_output_paths=reserved,
        context={"purpose": "continuation"},
    )
    assert continued.completed_stages == ("cpu_preflight",)
    second_attempt = continued.start_attempt()
    assert second_attempt != first_attempt
    continued.append_stage(
        stage="vertical_smoke",
        evidence=_evidence(vertical_output, _vertical_assertions()),
        attempt_id=second_attempt,
    )
    continued.close()
    assert ExecutionEvidenceJournal.inspect(
        root / "journal"
    ).completed_work_item_ids == (
        "cpu_preflight",
        "vertical_smoke",
    )

    with pytest.raises((ResearchProbeAdmissionError, ArtifactContractError)):
        ResearchProbeAdmission.open(
            root=root,
            admission_id="different-case",
            bindings=manifest,
            reserved_output_paths=reserved,
            context={"purpose": "continuation"},
        )


def test_input_drift_between_stages_rejects_vertical_without_losing_cpu(
    tmp_path: Path,
) -> None:
    manifest, producer, *_ = _fixture_manifest(tmp_path)
    root = tmp_path / "admission"
    admission = ResearchProbeAdmission.create(
        root=root,
        admission_id="case",
        bindings=manifest,
        reserved_output_paths=_reserved(tmp_path),
        context={},
    )
    attempt = admission.start_attempt()
    admission.append_stage(
        stage="cpu_preflight",
        evidence=_evidence(
            _write(tmp_path / "reserved" / "cpu" / "receipt.json", "{}"),
            _cpu_assertions(),
        ),
        attempt_id=attempt,
    )
    producer.write_text("drift", encoding="utf-8")
    with pytest.raises(ResearchProbeAdmissionError):
        admission.append_stage(
            stage="vertical_smoke",
            evidence=_evidence(
                _write(tmp_path / "reserved" / "vertical" / "receipt.json", "{}"),
                _vertical_assertions(),
            ),
            attempt_id=attempt,
        )
    admission.close()
    assert ExecutionEvidenceJournal.inspect(
        root / "journal"
    ).completed_work_item_ids == ("cpu_preflight",)


def test_finalize_is_mechanics_only_idempotent_and_fresh_process_recoverable(
    tmp_path: Path,
) -> None:
    manifest, *_ = _fixture_manifest(tmp_path)
    reserved = _reserved(tmp_path)
    root = tmp_path / "admission"
    admission = ResearchProbeAdmission.create(
        root=root,
        admission_id="case",
        bindings=manifest,
        reserved_output_paths=reserved,
        context={"purpose": "final"},
    )
    _append_both(admission, tmp_path)
    first = admission.finalize()
    encoded = first.read_bytes()
    assert admission.finalize() == first
    assert first.read_bytes() == encoded
    admission.close()

    reopened = ResearchProbeAdmission.open(
        root=root,
        admission_id="case",
        bindings=manifest,
        reserved_output_paths=reserved,
        context={"purpose": "final"},
    )
    assert reopened.finalize().read_bytes() == encoded
    receipt = json.loads(encoded)
    assert receipt["status"] == "mechanically_admitted"
    assert receipt["claim_boundary"]["mechanics_only"] is True
    assert receipt["claim_boundary"]["scientific_validity"] is False
    assert receipt["claim_boundary"]["launch_authorized"] is False
    assert "attempt_id" not in encoded.decode("utf-8")
    inspection = ResearchProbeAdmission.inspect(root)
    assert inspection.journal_terminal is True
    assert inspection.mechanically_admitted is True


def test_finalize_accepts_publish_uncertainty_only_for_identical_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, *_ = _fixture_manifest(tmp_path)
    root = tmp_path / "admission"
    admission = ResearchProbeAdmission.create(
        root=root,
        admission_id="case",
        bindings=manifest,
        reserved_output_paths=_reserved(tmp_path),
        context={},
    )
    _append_both(admission, tmp_path)
    original_publish = admission_module.publish_json_exclusive

    def publish_then_report_failure(path: Path, value):
        result = original_publish(path, value)
        if path.name == "admission.json":
            raise ArtifactContractError(
                "uncertain publication",
                code="artifact.publish_failed",
                context={"published_before_failure": True},
            )
        return result

    monkeypatch.setattr(
        admission_module, "publish_json_exclusive", publish_then_report_failure
    )
    assert admission.finalize() == root / "admission.json"
    assert load_canonical_json(root / "admission.json")["status"] == (
        "mechanically_admitted"
    )


def test_output_drift_prevents_final_receipt_but_preserves_terminal_records(
    tmp_path: Path,
) -> None:
    manifest, *_ = _fixture_manifest(tmp_path)
    root = tmp_path / "admission"
    admission = ResearchProbeAdmission.create(
        root=root,
        admission_id="case",
        bindings=manifest,
        reserved_output_paths=_reserved(tmp_path),
        context={},
    )
    _append_both(admission, tmp_path)
    (tmp_path / "reserved" / "vertical" / "receipt.json").write_text(
        "drift", encoding="utf-8"
    )
    with pytest.raises(ResearchProbeAdmissionError) as drift:
        admission.finalize()
    assert drift.value.code == "admission.output_drift"
    assert not (root / "admission.json").exists()
    assert ExecutionEvidenceJournal.inspect(root / "journal").terminal is None


def test_package_exports_only_the_intentional_top_level_admission_surface() -> None:
    expected_admission_exports = {
        "AbsoluteExecutableBinding",
        "AdmissionInspection",
        "BindingManifest",
        "DirectoryTreeBinding",
        "RegularFileBinding",
        "ResearchProbeAdmission",
        "ResearchProbeAdmissionError",
        "ReservedOutputPath",
        "ResolvedDataFileBinding",
        "StageEvidence",
        "StrictValueBinding",
        "TargetTreeBinding",
        "TargetTreeIdentity",
        "capture_binding_manifest",
        "capture_target_tree_binding",
        "revalidate_binding_manifest",
        "revalidate_target_tree_binding",
    }
    module_admission_names = set(admission_module.__all__)
    assert module_admission_names.intersection(artifact_package.__all__) == (
        expected_admission_exports
    )
    for internal_name in {
        "ADMISSION_SCHEMA_VERSION",
        "MECHANICS_STATEMENT",
        "STAGES",
    }:
        assert internal_name not in artifact_package.__all__
        assert not hasattr(artifact_package, internal_name)


def test_nested_journal_schema_and_terminal_receipt_remain_version_one(
    tmp_path: Path,
) -> None:
    root = tmp_path / "journal-residue"
    journal = ExecutionEvidenceJournal.create(
        root=root,
        execution_id="residue",
        execution_identity={"kind": "mechanics-residue"},
        expected_work_item_ids=("one",),
        context={"scope": "schema"},
    )
    attempt = journal.start_attempt()
    record_path = journal.append_record(
        work_item_id="one",
        payload={"opaque": True},
        attempt_id=attempt,
    )
    terminal_path = journal.finalize()
    journal.close()

    plan = load_canonical_json(root / "plan.json")
    record = load_canonical_json(record_path)
    terminal = load_canonical_json(terminal_path)
    assert JOURNAL_SCHEMA_VERSION == 1
    assert {
        plan["journal_schema_version"],
        record["journal_schema_version"],
        terminal["journal_schema_version"],
    } == {1}
    assert terminal["status"] == "completed"
    assert terminal["record_digests"] == [record["content_sha256"]]
    assert terminal["record_digest_aggregate"] == json_sha256(
        terminal["record_digests"]
    )
    assert set(terminal) == {
        "journal_schema_version",
        "execution_id",
        "plan_fingerprint",
        "record_digests",
        "record_digest_aggregate",
        "status",
        "content_sha256",
    }


def test_absent_inference_execution_context_remains_side_effect_free(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "absent-execution-context"
    payload = prepare_execution_context_payload(execution_context=None)

    assert payload is None
    assert validate_execution_context_identity(None) is None
    assert publish_execution_context_artifact(run_dir=run_dir, payload=payload) is None
    verify_local_execution_context_artifact(output_dir=run_dir, identity=None)
    assert not run_dir.exists()


def test_schema_owned_admission_payload_and_receipt_avoid_consumer_authority(
    tmp_path: Path,
) -> None:
    manifest, *_ = _fixture_manifest(tmp_path)
    root = tmp_path / "admission"
    admission = ResearchProbeAdmission.create(
        root=root,
        admission_id="residue",
        bindings=manifest,
        reserved_output_paths=_reserved(tmp_path),
        context={"purpose": "mechanics"},
    )
    _append_both(admission, tmp_path)
    receipt_path = admission.finalize()
    admission.close()

    diagnostics = ExecutionEvidenceJournal.inspect_diagnostics(root / "journal")
    payload_schema_keys = set()
    for record in diagnostics.records:
        payload_schema_keys.update(record.payload)
        payload_schema_keys.update(record.payload["claim_boundary"])
    receipt = load_canonical_json(receipt_path)
    receipt_schema_keys = set(receipt)
    receipt_schema_keys.update(receipt["claim_boundary"])
    consumer_or_authority_vocabulary = {
        "approved",
        "approval",
        "arm",
        "cohort",
        "endpoint",
        "estimand",
        "intervention",
        "matched",
        "outcome",
        "owner",
        "qualified",
        "qualification",
        "ready_to_launch",
        "stop_rule",
        "tau",
        "threshold",
        "unmatched",
    }
    assert not (
        payload_schema_keys.intersection(consumer_or_authority_vocabulary)
        or receipt_schema_keys.intersection(consumer_or_authority_vocabulary)
    )
    assert receipt["status"] == "mechanically_admitted"
    authority_flags = {
        key: value
        for key, value in receipt["claim_boundary"].items()
        if key.endswith("_authorized")
    }
    assert authority_flags
    assert set(authority_flags.values()) == {False}
