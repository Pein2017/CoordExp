from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from src.coco_refinement.bootstrap import (
    BootstrapContractError,
    BootstrapSourceContract,
    _native_region,
    bootstrap_workspace,
    resume_workspace,
    resolve_indexed_image,
)
from src.coco_refinement.canonical import canonicalize_objects
from src.label_studio_coco_refinement import project as legacy_project
from src.label_studio_coco_refinement.store import (
    ManifestDriftError,
    canonical_json,
    sha256_file,
)


def _row(image_id: int) -> dict[str, object]:
    return {
        "images": [f"../rescale_32_1024_bbox/images/train2017/{image_id:012d}.jpg"],
        "objects": [
            {
                "bbox_2d": [10, 20, 300, 400],
                "desc": "person",
                "category_id": 1,
                "category_name": "person",
                "coco_ann_id": image_id * 100,
            }
        ],
        "width": 32,
        "height": 24,
        "image_id": image_id,
        "file_name": f"images/train2017/{image_id:012d}.jpg",
        "metadata": {"source": "coco2017", "split": "train"},
    }


def _fixture_contract(
    tmp_path: Path,
) -> tuple[BootstrapSourceContract, tuple[Path, ...]]:
    source_dir = tmp_path / "public_data/coco/rescale_32_1024_bbox_len12000"
    image_dir = tmp_path / "public_data/coco/rescale_32_1024_bbox/images/train2017"
    source_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    images: list[Path] = []
    for image_id, color in ((1, "red"), (2, "blue")):
        path = image_dir / f"{image_id:012d}.jpg"
        Image.new("RGB", (32, 24), color=color).save(path, format="JPEG")
        images.append(path)
    source = source_dir / "train.norm.jsonl"
    source.write_text(
        "".join(canonical_json(_row(image_id)) + "\n" for image_id in (1, 2)),
        encoding="utf-8",
    )
    return (
        BootstrapSourceContract(
            split="train",
            source_path=source,
            image_root=image_dir.parent,
            expected_source_sha256=sha256_file(source),
            expected_row_count=2,
        ),
        tuple(images),
    )


def test_bounded_bootstrap_creates_complete_compact_index_and_restarts_without_legacy_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, images = _fixture_contract(tmp_path)
    source_before = contract.source_path.read_bytes()
    image_bytes_before = tuple(path.read_bytes() for path in images)

    def reject_legacy_sidecar(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("legacy task-index sidecar must not be loaded")

    monkeypatch.setattr(
        legacy_project, "load_canonical_task_index_receipt", reject_legacy_sidecar
    )
    runtime_root = tmp_path / "outputs/coco_refinement/test"
    created = bootstrap_workspace(
        tmp_path,
        runtime_root=runtime_root,
        source_contracts=(contract,),
    )

    split = created.splits["train"]
    assert split.store_created is True
    assert len(split.tasks) == 2
    assert created.repository.count_tasks(project_id="coco-refinement:train") == 2
    assert created.repository.count_drafts(project_id="coco-refinement:train") == 0
    assert created.repository.list_task_identities(
        project_id="coco-refinement:train"
    ) == tuple(task.identity for task in split.tasks)
    assert [task.image_locator for task in split.tasks] == [
        "train2017/000000000001.jpg",
        "train2017/000000000002.jpg",
    ]
    assert all(task.current_generation == 0 for task in split.tasks)
    assert all(len(task.base_row_hash) == 64 for task in split.tasks)
    assert all(len(task.committed_result_hash) == 64 for task in split.tasks)
    managed_link = runtime_root / "train/images"
    assert managed_link.is_symlink()
    assert managed_link.resolve(strict=True) == contract.image_root.resolve(strict=True)
    assert contract.source_path.read_bytes() == source_before
    assert tuple(path.read_bytes() for path in images) == image_bytes_before

    with sqlite3.connect(runtime_root / "state.sqlite3") as connection:
        task_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(tasks)")
        }
        assert "objects_json" not in task_columns
        assert connection.execute("SELECT COUNT(*) FROM drafts").fetchone()[0] == 0

    restarted = bootstrap_workspace(
        tmp_path,
        runtime_root=runtime_root,
        source_contracts=(contract,),
    )
    assert restarted.splits["train"].store_created is False
    assert restarted.splits["train"].tasks == split.tasks
    assert restarted.repository.count_drafts() == 0
    assert resolve_indexed_image(split.tasks[0], contract.image_root) == images[0]


def test_source_or_store_manifest_drift_fails_closed(tmp_path: Path) -> None:
    contract, _ = _fixture_contract(tmp_path)
    runtime_root = tmp_path / "runtime"
    bootstrap_workspace(
        tmp_path, runtime_root=runtime_root, source_contracts=(contract,)
    )

    original_source = contract.source_path.read_bytes()
    contract.source_path.write_bytes(original_source + b"\n")
    with pytest.raises(ManifestDriftError, match="source_sha256"):
        bootstrap_workspace(
            tmp_path, runtime_root=runtime_root, source_contracts=(contract,)
        )
    contract.source_path.write_bytes(original_source)

    manifest_path = runtime_root / "train/project.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["storage"]["storage_id"] = "drifted"
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    with pytest.raises(ManifestDriftError, match="storage"):
        bootstrap_workspace(
            tmp_path, runtime_root=runtime_root, source_contracts=(contract,)
        )


def test_existing_workspace_resume_preserves_bootstrap_identity_after_export_drift(
    tmp_path: Path,
) -> None:
    contract, _ = _fixture_contract(tmp_path)
    runtime_root = tmp_path / "runtime"
    created = bootstrap_workspace(
        tmp_path, runtime_root=runtime_root, source_contracts=(contract,)
    )
    with sqlite3.connect(runtime_root / "state.sqlite3") as connection:
        project_before = connection.execute(
            "SELECT * FROM projects WHERE project_id = 'coco-refinement:train'"
        ).fetchone()
        tasks_before = connection.execute(
            "SELECT * FROM tasks WHERE project_id = 'coco-refinement:train' "
            "ORDER BY source_row_index"
        ).fetchall()
    contract.source_path.write_bytes(contract.source_path.read_bytes() + b"\n")

    resumed = resume_workspace(
        tmp_path,
        runtime_root=runtime_root,
        source_contracts=(contract,),
        repository=created.repository,
        annotation_verifier=type("Verifier", (), {"verify": lambda self, _identity: False})(),
        inference_receipt_resolver=type(
            "Resolver", (), {"resolve": lambda self, _receipt_id: None}
        )(),
    )

    assert resumed.splits["train"].store_created is False
    with sqlite3.connect(runtime_root / "state.sqlite3") as connection:
        assert connection.execute(
            "SELECT * FROM projects WHERE project_id = 'coco-refinement:train'"
        ).fetchone() == project_before
        assert connection.execute(
            "SELECT * FROM tasks WHERE project_id = 'coco-refinement:train' "
            "ORDER BY source_row_index"
        ).fetchall() == tasks_before


def test_indexed_image_rejects_traversal_descendant_symlink_hash_and_dimension_drift(
    tmp_path: Path,
) -> None:
    contract, images = _fixture_contract(tmp_path)
    result = bootstrap_workspace(
        tmp_path,
        runtime_root=tmp_path / "runtime",
        source_contracts=(contract,),
    )
    task = result.splits["train"].tasks[0]

    traversal_task = replace(task)
    # Simulate a corrupted persistence/caller boundary; the repository's
    # constructor independently rejects this locator during ordinary use.
    object.__setattr__(traversal_task, "image_locator", "train2017/../outside.jpg")
    with pytest.raises(BootstrapContractError) as traversal:
        resolve_indexed_image(traversal_task, contract.image_root)
    assert traversal.value.code == "coco_refinement.image_locator"

    symlink = images[0].with_name("symlink.jpg")
    symlink.symlink_to(images[0])
    with pytest.raises(BootstrapContractError) as linked:
        resolve_indexed_image(
            replace(task, image_locator="train2017/symlink.jpg"),
            contract.image_root,
        )
    assert linked.value.code == "coco_refinement.image_symlink"

    original = images[0].read_bytes()
    images[0].write_bytes(images[1].read_bytes())
    with pytest.raises(BootstrapContractError) as hashed:
        resolve_indexed_image(task, contract.image_root)
    assert hashed.value.code == "coco_refinement.image_hash"
    images[0].write_bytes(original)

    wrong_dimensions = replace(task, image_width=31)
    with pytest.raises(BootstrapContractError) as dimensioned:
        resolve_indexed_image(wrong_dimensions, contract.image_root)
    assert dimensioned.value.code == "coco_refinement.image_dimensions"


def test_native_projection_preserves_committed_roi_provenance() -> None:
    metadata = {
        "inference_origin": True,
        "receipt_id": "receipt-1",
        "request_id": "request-1",
        "result_id": "result-1",
        "draft_revision": "3",
    }
    region = _native_region(
        {
            "region_key": "roi:receipt-1:result-1",
            "bbox_2d": [1, 2, 30, 40],
            "category_name": "person",
            "category_id": 1,
            "coco_ann_id": -1,
            "metadata": metadata,
            "desc": "person",
        }
    )

    canonical = canonicalize_objects([region], split="train")

    assert canonical.to_json_regions()[0]["metadata"] == metadata
