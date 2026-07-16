from __future__ import annotations

import hashlib
import json
import stat
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from scripts.probes.label_studio_coco_refinement import async_batch as probe


EXPECTED_CONTRACTS = {
    "DraftSaveReceipt": [
        "project_id",
        "task_id",
        "annotation_id",
        "draft_id",
        "annotation_revision",
        "draft_updated_at",
        "semantic_hash",
        "result_hash",
        "durable",
    ],
    "CommitRequest": [
        "commit_id",
        "split",
        "image_id",
        "project_id",
        "task_id",
        "annotation_id",
        "draft_id",
        "annotation_revision",
        "draft_updated_at",
        "semantic_hash",
        "result_hash",
        "base_row_hash",
        "observed_generation",
        "regions",
        "draft_save",
        "inference_receipts",
    ],
    "BatchMember": ["source_row_index", "request"],
    "BatchRequest": [
        "batch_id",
        "split",
        "current_user_id",
        "base_generation",
        "members",
    ],
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _artifact_identity(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    info = resolved.stat()
    return {
        "path": str(path),
        "resolved": str(resolved),
        "device": info.st_dev,
        "inode": info.st_ino,
        "size_bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "sha256": _sha256_file(resolved),
    }


def _git_output(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(probe.REPO_ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _code_identity() -> dict[str, Any]:
    script = Path(probe.__file__).resolve()
    store = Path(probe.store_module.__file__).resolve()
    return {
        "git_head": _git_output("rev-parse", "HEAD"),
        "store": {
            "path": str(store),
            "sha256": _sha256_file(store),
            "git_blob_sha1": _git_output("hash-object", str(store)),
        },
        "script": {
            "path": str(script),
            "sha256": _sha256_file(script),
            "git_blob_sha1": _git_output("hash-object", str(script)),
        },
    }


def _row(image_id: int) -> dict[str, Any]:
    return {
        "images": [f"../rescale_32_1024_bbox/images/train2017/{image_id:012d}.jpg"],
        "objects": [
            {
                "bbox_2d": [10, 20, 30, 40],
                "desc": "cat",
                "category_id": 17,
                "category_name": "cat",
                "coco_ann_id": image_id * 100 + 1,
            }
        ],
        "width": 640,
        "height": 480,
        "image_id": image_id,
        "file_name": f"images/train2017/{image_id:012d}.jpg",
        "metadata": {"source": "coco2017", "split": "train"},
    }


def _initial_working_line(image_id: int) -> bytes:
    row = _row(image_id)
    row["images"] = [row["file_name"]]
    return (_canonical_json(row) + "\n").encode()


@pytest.fixture
def tiny_source(tmp_path: Path) -> tuple[Path, Path, dict[Path, bytes]]:
    source_dir = tmp_path / "public_data/coco/rescale_32_1024_bbox_len12000"
    image_root = tmp_path / "public_data/coco/rescale_32_1024_bbox/images"
    image_dir = image_root / "train2017"
    source_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    image_bytes: dict[Path, bytes] = {}
    for image_id in range(1, 6):
        image = image_dir / f"{image_id:012d}.jpg"
        payload = f"image-{image_id}".encode()
        image.write_bytes(payload)
        image_bytes[image] = payload
    source = source_dir / "train.norm.jsonl"
    source.write_text(
        "".join(_canonical_json(_row(image_id)) + "\n" for image_id in range(1, 6)),
        encoding="utf-8",
    )
    return source, image_root, image_bytes


@pytest.mark.parametrize("rows", [5, None], ids=["slice", "full-source"])
def test_probe_receipt_replays_current_contract_and_exact_publication(
    tiny_source: tuple[Path, Path, dict[Path, bytes]],
    rows: int | None,
) -> None:
    source, image_root, image_bytes = tiny_source
    source_before = _file_identity(source.resolve())
    images_before = {
        str(path.resolve()): _file_identity(path.resolve()) for path in image_bytes
    }
    code_before = _code_identity()
    outputs = probe.REPO_ROOT / "outputs"
    outputs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="async-batch-test-", dir=outputs) as parent:
        output_root = Path(parent) / "run"
        returned = probe.run_probe(
            output_root=output_root,
            members=2,
            rows=rows,
            source=source,
            image_root=image_root,
        )
        receipt_path = output_root / "receipt.json"
        persisted = json.loads(receipt_path.read_text(encoding="utf-8"))

        assert returned == persisted
        assert persisted["receipt_path"] == str(receipt_path)
        assert persisted["contracts"] == EXPECTED_CONTRACTS
        assert persisted["configuration"]["full_source"] is (rows is None)
        assert persisted["configuration"]["rows"] == 5
        assert persisted["bootstrap"]["task_count"] == 5
        assert persisted["code_identity_unchanged"] is True
        assert not list(output_root.glob(".receipt.json.*"))

        split_dir = output_root / "runtime/train"
        artifact_paths = {
            "queue": split_dir / "queue.jsonl",
            "journal": split_dir / "journal.jsonl",
            "manifest": split_dir / "project.json",
            "working": split_dir / "working.norm.jsonl",
        }
        assert persisted["final_artifacts"] == {
            name: _artifact_identity(path) for name, path in artifact_paths.items()
        }

        manifest = json.loads(artifact_paths["manifest"].read_text(encoding="utf-8"))
        working_lines = artifact_paths["working"].read_bytes().splitlines(keepends=True)
        assert manifest == persisted["final_manifest"]
        assert manifest["generation"] == persisted["batch"]["base_generation"] + 1
        assert (
            manifest["generation"] == persisted["batch"]["final_status"]["generation"]
        )
        assert manifest["working_sha256"] == _sha256_file(artifact_paths["working"])
        assert manifest["working_line_count"] == len(working_lines) == 5
        assert persisted["batch"]["generation_increment"] == 1

        initial_lines = [_initial_working_line(image_id) for image_id in range(1, 6)]
        independently_changed = [
            index
            for index, (before, after) in enumerate(zip(initial_lines, working_lines))
            if before != after
        ]
        member_indices = persisted["batch"]["member_source_row_indices"]
        sentinel_index = persisted["batch"]["sentinel_source_row_index"]
        assert independently_changed == member_indices == [0, 1]
        assert working_lines[sentinel_index] == initial_lines[sentinel_index]
        assert all(
            working_lines[index] == initial_lines[index]
            for index in set(range(5)) - set(member_indices)
        )

        queue_records = [
            json.loads(line)
            for line in artifact_paths["queue"].read_text(encoding="utf-8").splitlines()
        ]
        enqueue = next(
            record for record in queue_records if record["kind"] == "enqueue"
        )
        payload = enqueue["payload"]
        assert payload["current_user_id"] == "async-batch-probe"
        assert persisted["batch"]["current_user_id"] == payload["current_user_id"]
        queue_members = {
            member["source_row_index"]: member["request"]
            for member in payload["members"]
        }
        receipt_members = {
            member["source_row_index"]: member
            for member in persisted["batch"]["members"]
        }
        assigned_ids: list[int] = []
        for index in member_indices:
            request = queue_members[index]
            member = receipt_members[index]
            assert member["image_id"] == request["image_id"]
            assert member["draft_updated_at"] == request["draft_updated_at"]
            assert member["result_hash"] == request["result_hash"]
            assert request["result_hash"] == _sha256_json(request["regions"])
            assert request["draft_updated_at"].endswith("Z")
            datetime.fromisoformat(request["draft_updated_at"].replace("Z", "+00:00"))
            handshake_fields = (
                "project_id",
                "task_id",
                "annotation_id",
                "draft_id",
                "annotation_revision",
                "draft_updated_at",
                "semantic_hash",
                "result_hash",
            )
            assert {name: request[name] for name in handshake_fields} == {
                name: request["draft_save"][name] for name in handshake_fields
            }
            assigned = member["negative_coco_ann_id"]
            assigned_ids.append(assigned)
            final_row = json.loads(working_lines[index])
            added = [
                obj
                for obj in final_row["objects"]
                if obj.get("metadata", {}).get("human_note")
                == "current async batch probe"
            ]
            assert len(added) == 1
            assert added[0]["coco_ann_id"] == assigned
        assert len(set(assigned_ids)) == 2
        assert all(type(value) is int and value < 0 for value in assigned_ids)

        source_after = _file_identity(source.resolve())
        assert source_before == source_after
        assert persisted["source_identity_before"] == source_before
        assert persisted["source_identity_after"] == source_after
        identity_keys = {
            "path",
            "resolved",
            "device",
            "inode",
            "size_bytes",
            "mtime_ns",
            "sha256",
        }
        for before, after in zip(
            persisted["sampled_images_before"],
            persisted["sampled_images_after"],
        ):
            before_identity = {key: before[key] for key in identity_keys}
            after_identity = {key: after[key] for key in identity_keys}
            assert before_identity == images_before[before["resolved"]]
            assert after_identity == _file_identity(Path(after["resolved"]))
            assert before_identity == after_identity
        assert all(path.read_bytes() == value for path, value in image_bytes.items())

        code_after = _code_identity()
        assert code_before == code_after
        assert persisted["code_identity_before"] == code_before
        assert persisted["code_identity_after"] == code_after
        assert persisted["authority"] == {
            "batch_verify_calls": 1,
            "fallback_verify_calls": 0,
            "inference_resolver_calls": [],
        }


def test_atomic_receipt_writer_replaces_and_fsyncs_file_and_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_path = tmp_path / "receipt.json"
    receipt = {"receipt_path": str(receipt_path), "value": 7}
    real_replace = probe.os.replace
    real_fsync = probe.os.fsync
    real_path_open = Path.open
    replace_calls: list[tuple[Path, Path]] = []
    fsync_targets: list[str] = []

    def guarded_path_open(
        self: Path, mode: str = "r", *args: Any, **kwargs: Any
    ) -> Any:
        if self == receipt_path and any(flag in mode for flag in "wax+"):
            raise AssertionError("direct receipt write forbidden")
        return real_path_open(self, mode, *args, **kwargs)

    def replace_spy(source: str | Path, destination: str | Path) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        assert source_path.parent == destination_path.parent == tmp_path
        replace_calls.append((source_path, destination_path))
        real_replace(source, destination)

    def fsync_spy(descriptor: int) -> None:
        mode = probe.os.fstat(descriptor).st_mode
        fsync_targets.append("directory" if stat.S_ISDIR(mode) else "file")
        real_fsync(descriptor)

    monkeypatch.setattr(Path, "open", guarded_path_open)
    monkeypatch.setattr(probe.os, "replace", replace_spy)
    monkeypatch.setattr(probe.os, "fsync", fsync_spy)

    assert probe._write_receipt_atomic(tmp_path, receipt) == receipt_path
    assert len(replace_calls) == 1
    assert replace_calls[0][1] == receipt_path
    assert fsync_targets == ["file", "directory"]
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == receipt
    assert not replace_calls[0][0].exists()
    with pytest.raises(AssertionError, match="direct receipt write forbidden"):
        receipt_path.write_text("forbidden", encoding="utf-8")


def test_parser_defaults_rows_full_source_and_mutual_exclusion() -> None:
    parser = probe._parser()
    default = parser.parse_args(["--output-root", "outputs/probe-default"])
    assert default.rows == 1000
    assert default.members == 10

    bounded = parser.parse_args(
        ["--rows", "7", "--members", "2", "--output-root", "outputs/probe-rows"]
    )
    assert bounded.rows == 7
    assert bounded.members == 2

    full = parser.parse_args(["--full-source", "--output-root", "outputs/probe-full"])
    assert full.rows is None

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--rows",
                "7",
                "--full-source",
                "--output-root",
                "outputs/probe-conflict",
            ]
        )


def test_probe_refuses_existing_outside_and_nonignored_output_roots(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="already exists"):
        probe._validate_output_root(tmp_path)

    with pytest.raises(ValueError, match="must stay inside"):
        probe._validate_output_root(tmp_path / "does-not-exist")

    nonignored = (
        probe.REPO_ROOT
        / "scripts/probes/label_studio_coco_refinement/not-an-output-root"
    )
    assert not nonignored.exists()
    with pytest.raises(ValueError, match="not ignored by git"):
        probe._validate_output_root(nonignored)
