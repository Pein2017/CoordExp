from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts.probes.label_studio_coco_refinement import materialize_full as probe
from src.label_studio_coco_refinement.models import RefinementRuntimeLayout


def _minimal_live_repository(tmp_path: Path) -> tuple[Path, RefinementRuntimeLayout]:
    repository = tmp_path / "repo"
    layout = RefinementRuntimeLayout.under_repository(repository)
    source = layout.selected_source("train")
    source.parent.mkdir(parents=True)
    source.write_bytes(b'{"source":"immutable"}\n')
    image_root = layout.image_root
    image_root.mkdir(parents=True)
    split_root = layout.split_root("train")
    split_root.mkdir(parents=True)
    layout.images_link("train").symlink_to(image_root, target_is_directory=True)
    layout.working_norm("train").write_bytes(b'{"working":"live"}\n')
    (split_root / ".commit.lock").touch()
    (split_root / ".queue.lock").touch()
    (split_root / "queue.jsonl").touch()
    (split_root / "journal.jsonl").touch()
    (split_root / "task_index.json").write_text('{"entries":[]}\n')
    layout.project_manifest("train").write_text(
        json.dumps(
            {
                "split": "train",
                "generation": 0,
                "managed_image_link": str(layout.images_link("train")),
                "sentinel": {"must": "survive"},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return repository, layout


def _fake_operator_receipt(
    fake_repo_root: Path, split: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    layout = RefinementRuntimeLayout.under_repository(fake_repo_root)
    output = layout.working_coord(split)  # type: ignore[arg-type]
    output.write_bytes(b'{"coord":"materialized"}\n')
    output_hash = probe.sha256_file(output)
    receipt = {
        "schema_version": "test-operator-v1",
        "code": "label_studio.working_coord_materialized",
        "materialization": {
            "split": split,
            "row_count": 1,
            "object_count": 2,
            "loader_row_count": 1,
            "destination_sha256": output_hash,
        },
        "loader_attestation": {
            "seam": "src.data.iter_raw_examples",
            "row_count": 1,
            "status": "passed",
        },
    }
    probe._atomic_write_json(  # noqa: SLF001
        layout.split_root(split) / probe.OPERATOR_RECEIPT_NAME,  # type: ignore[arg-type]
        receipt,
    )
    return receipt, {"wall_seconds": 0.25, "peak_rss_bytes": 4096, "returncode": 0}


def test_snapshot_rebinds_only_copied_manifest_and_links_public_data(
    tmp_path: Path,
) -> None:
    repository, live_layout = _minimal_live_repository(tmp_path)
    probe_root = repository / "outputs/materialize-full-test"
    probe_root.mkdir(parents=True)
    live_manifest_bytes = live_layout.project_manifest("train").read_bytes()

    snapshot_layout, receipt = probe._snapshot_live_split(  # noqa: SLF001
        repo_root=repository, probe_root=probe_root, split="train"
    )

    live_manifest = json.loads(live_manifest_bytes)
    copied_manifest = json.loads(
        snapshot_layout.project_manifest("train").read_text(encoding="utf-8")
    )
    expected = dict(live_manifest)
    expected["managed_image_link"] = str(snapshot_layout.images_link("train"))
    assert copied_manifest == expected
    assert live_layout.project_manifest("train").read_bytes() == live_manifest_bytes
    assert receipt["manifest_rebind"]["changed_fields"] == ["managed_image_link"]
    assert receipt["locks"] == {
        "mode": "shared",
        "order": [".commit.lock", ".queue.lock"],
        "held_only_during_snapshot": True,
        "hold_seconds": receipt["locks"]["hold_seconds"],
    }
    assert (snapshot_layout.repository_root / "public_data").is_symlink()
    assert (snapshot_layout.repository_root / "public_data").resolve() == (
        repository / "public_data"
    ).resolve()
    assert snapshot_layout.working_norm("train").read_bytes() == b'{"working":"live"}\n'
    assert receipt["copy_strategy"]["working_mode"] in {"cow_clone", "copy"}
    assert (
        receipt["copy_strategy"]["cow_clone_file_count"]
        + receipt["copy_strategy"]["copy_file_count"]
        == receipt["snapshot_regular_file_count"]
    )


def test_probe_root_must_be_fresh_inside_repository_and_git_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    candidate = repository / "outputs/fresh-probe"
    monkeypatch.setattr(probe, "_git_path_is_ignored", lambda *_: False)
    with pytest.raises(probe.ProbeError, match="not ignored"):
        probe._validate_probe_root(repository, candidate)  # noqa: SLF001

    monkeypatch.setattr(probe, "_git_path_is_ignored", lambda *_: True)
    assert probe._validate_probe_root(repository, candidate) == candidate  # noqa: SLF001
    with pytest.raises(probe.ProbeError, match="inside"):
        probe._validate_probe_root(repository, tmp_path / "outside")  # noqa: SLF001
    candidate.parent.mkdir(parents=True)
    candidate.mkdir()
    with pytest.raises(probe.ProbeError, match="already exists"):
        probe._validate_probe_root(repository, candidate)  # noqa: SLF001


def test_run_probe_records_hashes_counts_loader_and_child_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, live_layout = _minimal_live_repository(tmp_path)
    probe_root = repository / "outputs/materialize-full-run"
    source_before = live_layout.selected_source("train").read_bytes()
    working_before = live_layout.working_norm("train").read_bytes()
    monkeypatch.setattr(probe, "_git_path_is_ignored", lambda *_: True)
    monkeypatch.setattr(probe, "_run_materializer_child", _fake_operator_receipt)

    artifact = probe.run_probe(
        repo_root=repository, probe_root=probe_root, split="train"
    )

    assert json.loads((probe_root / probe.PROBE_RECEIPT_NAME).read_text()) == artifact
    assert artifact["performance"] == {
        "wall_seconds": 0.25,
        "peak_rss_bytes": 4096,
        "returncode": 0,
    }
    assert artifact["result"]["row_count"] == 1
    assert artifact["result"]["object_count"] == 2
    assert artifact["result"]["loader_attestation"] == {
        "seam": "src.data.iter_raw_examples",
        "row_count": 1,
        "status": "passed",
    }
    assert artifact["immutability"]["all_unchanged"] is True
    assert artifact["immutability"]["source_code_unchanged"] is True
    assert artifact["immutability"]["cloned_working"]["sha256"] == hashlib.sha256(
        working_before
    ).hexdigest()
    assert artifact["snapshot"]["retained"] is False
    assert artifact["immutability"]["cloned_working"]["retained"] is False
    assert artifact["result"]["destination"]["retained"] is False
    assert artifact["result"]["operator_receipt_path_retained"] is False
    assert not (probe_root / "repository").exists()
    assert not list(probe_root.glob(".repository.snapshot-*"))
    assert not Path(artifact["result"]["destination"]["path"]).exists()
    assert not Path(artifact["result"]["operator_receipt_path"]).exists()
    assert live_layout.selected_source("train").read_bytes() == source_before
    assert live_layout.working_norm("train").read_bytes() == working_before
    assert artifact["credential_material_recorded"] is False
    assert set(artifact["code_identity"]["source_files"]) == {
        str(path) for path in probe.CODE_PATHS
    }


def test_run_probe_fails_if_live_working_bytes_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, live_layout = _minimal_live_repository(tmp_path)
    probe_root = repository / "outputs/materialize-full-mutated"
    monkeypatch.setattr(probe, "_git_path_is_ignored", lambda *_: True)

    def mutate_live(
        fake_repo_root: Path, split: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        live_layout.working_norm("train").write_bytes(b"changed\n")
        return _fake_operator_receipt(fake_repo_root, split)

    monkeypatch.setattr(probe, "_run_materializer_child", mutate_live)

    with pytest.raises(probe.ProbeError, match="changed immutable/live bytes"):
        probe.run_probe(repo_root=repository, probe_root=probe_root, split="train")
    assert not (probe_root / probe.PROBE_RECEIPT_NAME).exists()
    assert not (probe_root / "repository").exists()
    assert not list(probe_root.glob(".repository.snapshot-*"))


def test_retain_snapshot_opt_in_keeps_successful_repository_and_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, _live_layout = _minimal_live_repository(tmp_path)
    probe_root = repository / "outputs/materialize-full-retained"
    monkeypatch.setattr(probe, "_git_path_is_ignored", lambda *_: True)
    monkeypatch.setattr(probe, "_run_materializer_child", _fake_operator_receipt)

    artifact = probe.run_probe(
        repo_root=repository,
        probe_root=probe_root,
        split="train",
        retain_snapshot=True,
    )

    assert artifact["snapshot"]["retained"] is True
    assert artifact["immutability"]["cloned_working"]["retained"] is True
    assert artifact["result"]["destination"]["retained"] is True
    assert artifact["result"]["operator_receipt_path_retained"] is True
    assert (probe_root / "repository").is_dir()
    assert Path(artifact["result"]["destination"]["path"]).is_file()
    assert Path(artifact["result"]["operator_receipt_path"]).is_file()


def test_failed_run_cleans_snapshot_even_when_retention_was_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, _live_layout = _minimal_live_repository(tmp_path)
    probe_root = repository / "outputs/materialize-full-failed-retained"
    monkeypatch.setattr(probe, "_git_path_is_ignored", lambda *_: True)

    def fail_child(*, fake_repo_root: Path, split: str) -> Any:
        raise probe.ProbeError(f"synthetic child failure: {fake_repo_root}:{split}")

    monkeypatch.setattr(probe, "_run_materializer_child", fail_child)

    with pytest.raises(probe.ProbeError, match="synthetic child failure"):
        probe.run_probe(
            repo_root=repository,
            probe_root=probe_root,
            split="train",
            retain_snapshot=True,
        )
    assert not (probe_root / probe.PROBE_RECEIPT_NAME).exists()
    assert not (probe_root / "repository").exists()
    assert not list(probe_root.glob(".repository.snapshot-*"))


def test_child_runner_uses_current_interpreter_and_records_peak_rss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child_script = tmp_path / "operator.py"
    child_script.write_text(
        "import json, sys\n"
        "assert sys.executable == " + repr(sys.executable) + "\n"
        "payload = bytearray(1024 * 1024)\n"
        "print(json.dumps({'ok': True, 'bytes': len(payload)}))\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(probe, "OPERATOR_SCRIPT", child_script)

    receipt, performance = probe._run_materializer_child(  # noqa: SLF001
        fake_repo_root=tmp_path / "fake-repository",
        split="train",
    )

    assert receipt == {"ok": True, "bytes": 1024 * 1024}
    assert performance["returncode"] == 0
    assert performance["peak_rss_bytes"] > 0
    assert performance["wall_seconds"] > 0
