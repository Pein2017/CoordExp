from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from src.coco_refinement.dataset_publisher import (
    RECEIPT_NAME,
    TRANSACTION_NAME,
    CommittedGenerationPublisher,
    DatasetPublishError,
    TokenBudgetValidation,
)
from src.common.errors import DataContractError
from src.data import iter_raw_examples


class _PassingTokenValidator:
    def validate(
        self,
        coord_jsonl: Path,
        *,
        expected_row_count: int,
    ) -> TokenBudgetValidation:
        assert sum(1 for _ in iter_raw_examples(coord_jsonl)) == expected_row_count
        return TokenBudgetValidation(
            row_count=expected_row_count,
            max_total_tokens=12000,
            max_total_tokens_seen=1500,
            training_config_path=Path("/test/config.yaml"),
            training_config_sha256="a" * 64,
            training_config_fingerprint="b" * 64,
            tokenizer_sha256="c" * 64,
        )


class _RejectingTokenValidator:
    def validate(
        self,
        coord_jsonl: Path,
        *,
        expected_row_count: int,
    ) -> TokenBudgetValidation:
        raise DatasetPublishError(
            "synthetic row exceeds 12000 tokens",
            code="test.over_budget",
            context={"row": expected_row_count, "path": str(coord_jsonl)},
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _working_row(*, split: str, image_id: int = 2299) -> dict[str, object]:
    locator = f"images/{split}2017/{image_id:012d}.jpg"
    return {
        "file_name": locator,
        "height": 1056,
        "image_id": image_id,
        "images": [locator],
        "metadata": {"source": "coco2017", "split": split},
        "objects": [
            {
                "bbox_2d": [10, 20, 30, 40],
                "desc": "person",
                "category_id": 1,
                "category_name": "person",
                "coco_ann_id": -1,
            },
            {
                "bbox_2d": [50, 60, 90, 100],
                "desc": "tie",
                "category_id": 32,
                "category_name": "tie",
                "coco_ann_id": 99,
                "metadata": {"origin": "human"},
            },
        ],
        "width": 960,
    }


def _prepare_repository(tmp_path: Path) -> tuple[Path, Path]:
    repository = tmp_path / "repo"
    target_root = repository / "public_data/coco/rescale_32_1024_bbox_len12000"
    image_root = repository / "public_data/coco/rescale_32_1024_bbox/images"
    runtime_root = repository / "outputs/coco_refinement/test"
    target_root.mkdir(parents=True)
    image_root.mkdir(parents=True)
    runtime_root.mkdir(parents=True)
    for split in ("train", "val"):
        image = image_root / f"{split}2017/000000002299.jpg"
        image.parent.mkdir(parents=True)
        image.write_bytes(f"shared-{split}".encode())
        (target_root / f"{split}.norm.jsonl").write_bytes(
            f"old-{split}-norm\n".encode()
        )
        (target_root / f"{split}.coord.jsonl").write_bytes(
            f"old-{split}-coord\n".encode()
        )
        split_root = runtime_root / split
        split_root.mkdir()
        (split_root / ".commit.lock").touch()
        (split_root / "journal.jsonl").write_text("", encoding="utf-8")
        working = split_root / "working.norm.jsonl"
        working.write_text(
            json.dumps(_working_row(split=split), separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        (split_root / "project.json").write_text(
            json.dumps(
                {
                    "split": split,
                    "generation": 3,
                    "task_count": 1,
                    "working_sha256": _sha256(working),
                }
            )
            + "\n",
            encoding="utf-8",
        )
    return repository, runtime_root


def _publisher(
    repository: Path,
    runtime_root: Path,
    *,
    validator: object | None = None,
    fault_injector=None,
) -> CommittedGenerationPublisher:
    return CommittedGenerationPublisher(
        repository_root=repository,
        runtime_root=runtime_root,
        split="val",
        token_budget_validator=validator or _PassingTokenValidator(),
        fault_injector=fault_injector,
    )


def test_publish_replaces_only_selected_split_and_reuses_shared_images(
    tmp_path: Path,
) -> None:
    repository, runtime_root = _prepare_repository(tmp_path)
    target_root = repository / "public_data/coco/rescale_32_1024_bbox_len12000"
    train_before = {
        suffix: (target_root / f"train.{suffix}.jsonl").read_bytes()
        for suffix in ("norm", "coord")
    }

    receipt = _publisher(repository, runtime_root).publish()

    norm_path = target_root / "val.norm.jsonl"
    coord_path = target_root / "val.coord.jsonl"
    norm = json.loads(norm_path.read_text(encoding="utf-8"))
    coord = json.loads(coord_path.read_text(encoding="utf-8"))
    expected_image = "../rescale_32_1024_bbox/images/val2017/000000002299.jpg"
    assert norm["images"] == [expected_image]
    assert coord["images"] == [expected_image]
    assert norm["objects"][0]["bbox_2d"] == [10, 20, 30, 40]
    assert coord["objects"][0]["bbox_2d"] == [
        "<|coord_10|>",
        "<|coord_20|>",
        "<|coord_30|>",
        "<|coord_40|>",
    ]
    assert coord["objects"][1]["metadata"] == {"origin": "human"}
    assert sum(1 for _ in iter_raw_examples(coord_path)) == 1
    assert receipt.generation == 3
    assert receipt.row_count == 1
    assert receipt.object_count == 2
    assert receipt.negative_object_count == 1
    assert receipt.target_norm_sha256 == _sha256(norm_path)
    assert receipt.target_coord_sha256 == _sha256(coord_path)
    receipt_payload = json.loads(
        (runtime_root / "val" / RECEIPT_NAME).read_text(encoding="utf-8")
    )
    assert receipt_payload["images"]["copied"] is False
    assert receipt_payload["token_budget"]["max_total_tokens"] == 12000
    assert receipt_payload["schema_version"] == 2
    assert receipt_payload["identity_authority"] == {
        "journal_path": str(runtime_root / "val/journal.jsonl"),
        "journal_sha256": _sha256(runtime_root / "val/journal.jsonl"),
        "negative_object_count": 1,
        "status": "passed",
    }
    assert not (target_root / "images").exists()
    for suffix in ("norm", "coord"):
        assert (target_root / f"train.{suffix}.jsonl").read_bytes() == train_before[suffix]


def test_over_budget_failure_preserves_both_training_files(tmp_path: Path) -> None:
    repository, runtime_root = _prepare_repository(tmp_path)
    target_root = repository / "public_data/coco/rescale_32_1024_bbox_len12000"
    before = {
        suffix: (target_root / f"val.{suffix}.jsonl").read_bytes()
        for suffix in ("norm", "coord")
    }

    with pytest.raises(DatasetPublishError, match="exceeds 12000"):
        _publisher(
            repository,
            runtime_root,
            validator=_RejectingTokenValidator(),
        ).publish()

    for suffix in ("norm", "coord"):
        assert (target_root / f"val.{suffix}.jsonl").read_bytes() == before[suffix]
    assert not (runtime_root / "val" / RECEIPT_NAME).exists()
    assert not (runtime_root / "val" / TRANSACTION_NAME).exists()


def test_failure_between_pair_replacements_rolls_back_outputs_and_receipt(
    tmp_path: Path,
) -> None:
    repository, runtime_root = _prepare_repository(tmp_path)
    target_root = repository / "public_data/coco/rescale_32_1024_bbox_len12000"
    receipt_path = runtime_root / "val" / RECEIPT_NAME
    receipt_path.write_bytes(b'old-receipt\n')
    before = {
        "norm": (target_root / "val.norm.jsonl").read_bytes(),
        "coord": (target_root / "val.coord.jsonl").read_bytes(),
        "receipt": receipt_path.read_bytes(),
    }

    def fail(stage: str) -> None:
        if stage == "after_norm_replace":
            raise RuntimeError("synthetic replacement failure")

    with pytest.raises(RuntimeError, match="synthetic replacement failure"):
        _publisher(repository, runtime_root, fault_injector=fail).publish()

    assert (target_root / "val.norm.jsonl").read_bytes() == before["norm"]
    assert (target_root / "val.coord.jsonl").read_bytes() == before["coord"]
    assert receipt_path.read_bytes() == before["receipt"]
    assert not (runtime_root / "val" / TRANSACTION_NAME).exists()
    assert not list(target_root.glob(".*.rollback-*"))


def test_stale_working_hash_fails_before_candidate_publication(tmp_path: Path) -> None:
    repository, runtime_root = _prepare_repository(tmp_path)
    manifest_path = runtime_root / "val/project.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["working_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(DatasetPublishError, match="does not match"):
        _publisher(repository, runtime_root).publish()


def test_reversed_xyxy_is_rejected_without_touching_targets(tmp_path: Path) -> None:
    repository, runtime_root = _prepare_repository(tmp_path)
    working_path = runtime_root / "val/working.norm.jsonl"
    row = _working_row(split="val")
    row["objects"][0]["bbox_2d"] = [30, 40, 10, 20]  # type: ignore[index]
    working_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    manifest_path = runtime_root / "val/project.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["working_sha256"] = _sha256(working_path)
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(DataContractError, match="x1,y1,x2,y2"):
        _publisher(repository, runtime_root).publish()


def test_prepared_transaction_is_rolled_back_before_next_publish(tmp_path: Path) -> None:
    repository, runtime_root = _prepare_repository(tmp_path)
    target_root = repository / "public_data/coco/rescale_32_1024_bbox_len12000"
    split_root = runtime_root / "val"
    norm = target_root / "val.norm.jsonl"
    coord = target_root / "val.coord.jsonl"
    receipt = split_root / RECEIPT_NAME
    receipt.write_bytes(b"old-receipt\n")
    transaction_id = "interrupted"
    backups = [
        target_root / f".val.norm.rollback-{transaction_id}",
        target_root / f".val.coord.rollback-{transaction_id}",
        split_root / f".{RECEIPT_NAME}.rollback-{transaction_id}",
    ]
    for source, backup in zip((norm, coord, receipt), backups, strict=True):
        os.link(source, backup)
    partial = target_root / ".partial-new-norm"
    partial.write_bytes(b"partial-new-norm\n")
    os.replace(partial, norm)
    state = {
        "schema_version": 1,
        "transaction_id": transaction_id,
        "status": "prepared",
        "split": "val",
        "targets": [
            {"path": str(path), "backup": str(backup), "existed": True}
            for path, backup in zip((norm, coord, receipt), backups, strict=True)
        ],
        "candidates": [],
    }
    (split_root / TRANSACTION_NAME).write_text(json.dumps(state) + "\n")

    published = _publisher(repository, runtime_root).publish()

    assert published.generation == 3
    assert not (split_root / TRANSACTION_NAME).exists()
    assert not any(path.exists() for path in backups)
