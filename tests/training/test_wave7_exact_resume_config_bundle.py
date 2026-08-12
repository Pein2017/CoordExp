from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import yaml

from src.config.fingerprint import sha256_json
from src.config.loader import load_train_config


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/probes/coordexp_swift/wave7_exact_resume_config_bundle.py"
BASE_CONFIG = (
    REPO_ROOT / "configs/coordexp_swift/smoke/"
    "qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_llm_12000_"
    "accelerate8_ebs24_2step_warmup0p1_eval_patchproof.yaml"
)


@pytest.fixture
def producer():
    spec = importlib.util.spec_from_file_location(
        "wave7_exact_resume_config_bundle_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _expected_raw(role: str, root: Path) -> dict[str, object]:
    checkpoint = (
        str(root / "runs/interrupted_parent/checkpoints/step-3")
        if role == "resume_child"
        else None
    )
    return {
        "schema_version": 1,
        "extends": str(BASE_CONFIG.resolve()),
        "run": {
            "name": role,
            "artifact_root": str(root / "runs"),
            "collision_policy": "fail",
        },
        "training": {
            "max_steps": 5,
            "forward_input_provider_mode": "synchronous",
        },
        "runtime": {
            "seed": 17,
            "determinism": {"mode": "strict_cuda_replay_v1"},
        },
        "eval": {"forward": {"steps": [3]}},
        "checkpoint": {
            "steps": [3, 5],
            "save_final": True,
        },
        "resume": {
            "mode": "exact_same_world_size",
            "checkpoint_dir": checkpoint,
        },
    }


def _author_in_tmp(producer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "2026-08-12-core-1"
    monkeypatch.setattr(producer, "CANONICAL_OUTPUT_ROOT", root)
    receipt_path = producer.author(BASE_CONFIG, root)
    return root, receipt_path


def test_canonical_output_root_is_authorized_core_4_namespace(producer):
    assert (
        producer.CANONICAL_OUTPUT_ROOT
        == (
            REPO_ROOT
            / "outputs/probes/coordexp_swift/wave7_exact_resume/2026-08-12-core-4"
        ).resolve()
    )


def test_author_publishes_exact_strictly_loadable_bundle(
    producer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root, receipt_path = _author_in_tmp(producer, tmp_path, monkeypatch)

    assert receipt_path == root / "config-bundle-receipt.json"
    assert {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    } == {
        "configs/uninterrupted.yaml",
        "configs/interrupted-parent.yaml",
        "configs/resume-child.yaml",
        "config-bundle-receipt.json",
    }
    resolved_by_role = {}
    filenames = {
        "uninterrupted": "uninterrupted.yaml",
        "interrupted_parent": "interrupted-parent.yaml",
        "resume_child": "resume-child.yaml",
    }
    for role, filename in filenames.items():
        path = root / "configs" / filename
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert raw == _expected_raw(role, root)
        resolved_by_role[role] = load_train_config(path)

    projections = []
    for resolved in resolved_by_role.values():
        projection = dict(resolved.config_dict)
        del projection["run"]
        del projection["resume"]
        projections.append(projection)
    assert projections[0] == projections[1] == projections[2]

    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    assert receipt_bytes == _canonical(receipt) + b"\n"
    signed_body = dict(receipt)
    observed_digest = signed_body.pop("receipt_payload_sha256")
    assert observed_digest == _sha256_bytes(_canonical(signed_body))
    assert set(signed_body) == {
        "schema",
        "status",
        "base_config",
        "output_root",
        "configs",
        "semantic_projection_sha256",
    }
    assert signed_body["schema"] == "coordexp-swift-wave7-r7-config-bundle-v1"
    assert signed_body["status"] == "passed"
    assert signed_body["base_config"] == {
        "path": str(BASE_CONFIG.resolve()),
        "file_sha256": (
            "43f46f3df390cea3e6fe117654d9d82822a448a06e71b36d83d46c5a4d22f313"
        ),
    }
    assert signed_body["output_root"] == str(root)
    assert signed_body["semantic_projection_sha256"] == sha256_json(projections[0])
    for role, filename in filenames.items():
        path = root / "configs" / filename
        assert signed_body["configs"][role] == {
            "path": str(path),
            "file_sha256": _sha256_bytes(path.read_bytes()),
            "resolved_config_fingerprint": resolved_by_role[role].fingerprint,
        }


def test_parser_exposes_only_author_without_arbitrary_write_surface(producer):
    parser = producer._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["author", "--payload", "{}"])


def test_producer_has_no_model_or_cuda_import_surface():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported_roots = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )

    assert imported_roots.isdisjoint(
        {"accelerate", "safetensors", "torch", "transformers"}
    )


@pytest.mark.parametrize("kind", ("copied", "symlink"))
def test_noncanonical_base_is_rejected(
    producer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
):
    copied = tmp_path / "base.yaml"
    if kind == "copied":
        copied.write_bytes(BASE_CONFIG.read_bytes())
    else:
        copied.symlink_to(BASE_CONFIG)
    root = tmp_path / "2026-08-12-core-1"
    monkeypatch.setattr(producer, "CANONICAL_OUTPUT_ROOT", root)

    with pytest.raises(producer.ConfigBundleError, match="canonical base"):
        producer.author(copied, root)

    assert not root.exists()


@pytest.mark.parametrize("name", ("2026-08-11-r6", "2026-08-12-r6", "r7"))
def test_noncanonical_output_root_is_rejected(producer, tmp_path: Path, name: str):
    root = tmp_path / name

    with pytest.raises(producer.ConfigBundleError, match="canonical r7 root"):
        producer.author(BASE_CONFIG, root)

    assert not root.exists()


@pytest.mark.parametrize("occupied_kind", ("directory", "file", "symlink"))
def test_existing_or_symlinked_root_is_never_replaced(
    producer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    occupied_kind: str,
):
    root = tmp_path / "2026-08-12-core-1"
    monkeypatch.setattr(producer, "CANONICAL_OUTPUT_ROOT", root)
    if occupied_kind == "directory":
        root.mkdir()
        (root / "foreign.txt").write_text("keep", encoding="utf-8")
    elif occupied_kind == "file":
        root.write_text("keep", encoding="utf-8")
    else:
        foreign = tmp_path / "foreign"
        foreign.mkdir()
        root.symlink_to(foreign, target_is_directory=True)
    before = root.lstat()

    with pytest.raises(producer.ConfigBundleError, match="absent"):
        producer.author(BASE_CONFIG, root)

    assert root.lstat() == before
    if occupied_kind == "directory":
        assert (root / "foreign.txt").read_text(encoding="utf-8") == "keep"


def test_symlinked_output_ancestor_is_rejected(
    producer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    alias_parent = tmp_path / "alias"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    root = alias_parent / "2026-08-12-core-1"
    monkeypatch.setattr(producer, "CANONICAL_OUTPUT_ROOT", root)

    with pytest.raises(producer.ConfigBundleError, match="symlink"):
        producer.author(BASE_CONFIG, root)

    assert not (real_parent / "2026-08-12-core-1").exists()


@pytest.mark.parametrize("mutation", ("max_steps", "child_checkpoint"))
def test_resolved_semantic_or_child_checkpoint_drift_prevents_publication(
    producer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
):
    root = tmp_path / "2026-08-12-core-1"
    monkeypatch.setattr(producer, "CANONICAL_OUTPUT_ROOT", root)
    real_payload = producer._config_payload

    def mutated_payload(role: str, output_root: Path):
        payload = real_payload(role, output_root)
        if mutation == "max_steps" and role == "uninterrupted":
            payload["training"]["max_steps"] = 4
        if mutation == "child_checkpoint" and role == "resume_child":
            payload["resume"]["checkpoint_dir"] = str(
                output_root / "runs/interrupted_parent/checkpoints/step-4"
            )
        return payload

    monkeypatch.setattr(producer, "_config_payload", mutated_payload)

    with pytest.raises(producer.ConfigBundleError, match="semantics"):
        producer.author(BASE_CONFIG, root)

    assert not root.exists()
    assert not any(
        path.name.startswith(".2026-08-12-core-1.stage-") for path in tmp_path.iterdir()
    )


def test_atomic_publication_race_never_authorizes_foreign_root(
    producer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "2026-08-12-core-1"
    monkeypatch.setattr(producer, "CANONICAL_OUTPUT_ROOT", root)

    def collide(stage: Path, target: Path) -> None:
        del stage
        target.mkdir()
        raise FileExistsError("injected publication race")

    monkeypatch.setattr(producer, "_install_directory_no_replace", collide)

    with pytest.raises(producer.ConfigBundleError, match="publication"):
        producer.author(BASE_CONFIG, root)

    assert root.is_dir()
    assert list(root.iterdir()) == []
    assert not any(
        path.name.startswith(".2026-08-12-core-1.stage-") for path in tmp_path.iterdir()
    )


def test_post_commit_parent_fsync_failure_keeps_committed_bundle_successful(
    producer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "2026-08-12-core-1"
    monkeypatch.setattr(producer, "CANONICAL_OUTPUT_ROOT", root)
    real_fsync_directory = producer._fsync_directory

    def fail_only_parent_after_commit(path: Path) -> None:
        if path == root.parent:
            raise OSError("injected post-rename parent fsync failure")
        real_fsync_directory(path)

    monkeypatch.setattr(producer, "_fsync_directory", fail_only_parent_after_commit)

    receipt_path = producer.author(BASE_CONFIG, root)

    assert receipt_path == root / "config-bundle-receipt.json"
    assert receipt_path.is_file()
    receipt = json.loads(receipt_path.read_bytes())
    signed_body = dict(receipt)
    observed_digest = signed_body.pop("receipt_payload_sha256")
    assert observed_digest == _sha256_bytes(_canonical(signed_body))
    assert receipt["schema"] == "coordexp-swift-wave7-r7-config-bundle-v1"
    assert receipt["status"] == "passed"
    assert {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    } == {
        "configs/uninterrupted.yaml",
        "configs/interrupted-parent.yaml",
        "configs/resume-child.yaml",
        "config-bundle-receipt.json",
    }
