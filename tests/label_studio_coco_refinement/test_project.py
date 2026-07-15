from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

import src.label_studio_coco_refinement.project as project_module
from src.label_studio_coco_refinement.categories import COCO80_REGISTRY
from src.label_studio_coco_refinement.geometry import norm1000_bbox_to_label_studio_xywh
from src.label_studio_coco_refinement.label_config import (
    build_label_config,
    label_config_fingerprint,
)
from src.label_studio_coco_refinement.project import (
    BootstrapAction,
    CanonicalJsonArrayFingerprint,
    FrozenTaskImport,
    LiveProjectAttestation,
    LiveTaskSetAttestation,
    ManifestDriftError,
    ProjectContractError,
    RUNTIME_ROOT,
    SHARED_IMAGE_ROOT,
    SOURCE_CONTRACTS,
    RuntimeLayout,
    Split,
    TaskIdentity,
    assert_manifest_matches,
    build_instance_bootstrap_manifest,
    build_split_project_plan,
    compare_manifests,
    fingerprint_json,
    inspect_source,
    local_files_image_locator,
    make_task_import_payload,
    managed_image_link_plan,
    plan_instance_bootstrap,
    resolve_working_image,
    sha256_file,
    validate_authoritative_task_payload,
    validate_immutable_row_fields,
    validate_source_row,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _row(split: Split, *, image_id: int = 9) -> dict:
    subdirectory = "train2017" if split is Split.TRAIN else "val2017"
    file_name = f"images/{subdirectory}/{image_id:012d}.jpg"
    return {
        "images": [f"../rescale_32_1024_bbox/{file_name}"],
        "objects": [
            {
                "bbox_2d": [0, 10, 500, 999],
                "desc": "person",
                "category_id": 1,
                "category_name": "person",
                "coco_ann_id": 123,
            }
        ],
        "width": 1152,
        "height": 864,
        "image_id": image_id,
        "file_name": file_name,
        "metadata": {"source": "coco2017", "split": split.value},
    }


class _AttestingAdapter:
    def __init__(
        self,
        states: dict[Split, LiveProjectAttestation | None],
        *,
        bootstrap_manifest: dict[str, Any] | None = None,
    ) -> None:
        self.states = states
        self.bootstrap_manifest = bootstrap_manifest
        self.calls: list[Split] = []

    def attest_project(self, desired) -> LiveProjectAttestation | None:
        self.calls.append(desired.split)
        return self.states.get(desired.split)

    def attest_bootstrap_manifest(self) -> dict[str, Any] | None:
        return self.bootstrap_manifest


class _SavedManifestOnlyAdapter:
    def __init__(
        self,
        manifests: dict[Split, dict[str, Any]],
        bootstrap_manifest: dict[str, Any],
    ) -> None:
        self.manifests = manifests
        self.bootstrap_manifest = bootstrap_manifest

    def attest_project(self, desired) -> dict[str, Any] | None:
        return self.manifests.get(desired.split)

    def attest_bootstrap_manifest(self) -> dict[str, Any]:
        return self.bootstrap_manifest


def _install_small_source(
    tmp_path: Path,
    split: Split,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    row = _row(split, image_id=9 if split is Split.TRAIN else 139)
    return _install_source_rows(tmp_path, split, monkeypatch, [row])


def _install_source_rows(
    tmp_path: Path,
    split: Split,
    monkeypatch: pytest.MonkeyPatch,
    rows: list[dict[str, Any]],
) -> Path:
    contract = project_module.SOURCE_CONTRACTS[split]
    source_path = tmp_path / Path(contract.relative_path)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = b"".join(
        (json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )
    source_path.write_bytes(encoded)
    for row in rows:
        image_path = (
            tmp_path / "public_data/coco/rescale_32_1024_bbox" / row["file_name"]
        )
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"fixture-image-bytes")
    contracts = dict(project_module.SOURCE_CONTRACTS)
    contracts[split] = replace(
        contract,
        sha256=hashlib.sha256(encoded).hexdigest(),
        row_count=len(rows),
        box_count=sum(len(row["objects"]) for row in rows),
    )
    monkeypatch.setattr(
        project_module,
        "SOURCE_CONTRACTS",
        MappingProxyType(contracts),
    )
    return source_path


def _small_plan(tmp_path: Path, split: Split, monkeypatch: pytest.MonkeyPatch):
    source_path = _install_small_source(tmp_path, split, monkeypatch)
    return _build_plan(tmp_path, split, source_path)


def _build_plan(tmp_path: Path, split: Split, source_path: Path):
    config = build_label_config()
    return build_split_project_plan(
        source_path,
        repo_root=tmp_path,
        split=split,
        vendor_revision="pinned-vendor-revision",
        label_config=config,
        label_config_fingerprint=label_config_fingerprint(config),
        registry=COCO80_REGISTRY,
        bbox_converter=norm1000_bbox_to_label_studio_xywh,
    )


def _live_attestation(
    plan,
    *,
    project_id: int,
    observed_task_count: int | None = None,
) -> LiveProjectAttestation:
    observed = (
        plan.task_manifest.task_count
        if observed_task_count is None
        else observed_task_count
    )
    task_set = LiveTaskSetAttestation(
        expected_task_manifest_fingerprint=plan.task_manifest.fingerprint,
        observed_task_count=observed,
        missing_task_count=plan.task_manifest.task_count - observed,
        content_fingerprint=fingerprint_json(
            {"project_id": project_id, "observed_task_count": observed}
        ),
    )
    return LiveProjectAttestation(
        split=plan.split,
        project_id=project_id,
        project_identity=plan.manifest.project_identity,
        saved_manifest=plan.manifest.to_dict(),
        vendor_revision=plan.manifest.vendor_revision,
        label_config=plan.label_config,
        controls=plan.controls,
        storage_manifest=plan.storage_manifest,
        managed_link_is_symlink=True,
        managed_link_resolved_target=plan.storage_manifest.managed_link_target,
        task_set=task_set,
    )


def _adapter_for(
    train,
    val,
    states: dict[Split, LiveProjectAttestation | None],
) -> _AttestingAdapter:
    manifest = build_instance_bootstrap_manifest({Split.TRAIN: train, Split.VAL: val})
    return _AttestingAdapter(states, bootstrap_manifest=manifest.to_dict())


def _resign_task_index_with_mutation(receipt, record_index: int, field: str):
    records = [
        json.loads(line)
        for line in Path(receipt.path).read_bytes().splitlines()
    ]
    records[record_index][field] = True
    previous_hash = None
    encoded_records = []
    for index, record in enumerate(records):
        body = {key: value for key, value in record.items() if key != "record_hash"}
        if index:
            body["previous_record_hash"] = previous_hash
        record_hash = fingerprint_json(body)
        previous_hash = record_hash
        encoded_records.append(
            project_module._canonical_json_bytes({**body, "record_hash": record_hash})
            + b"\n"
        )
    encoded = b"".join(encoded_records)
    digest = hashlib.sha256(encoded).hexdigest()
    path = Path(receipt.path).with_name(
        project_module._task_index_sidecar_name(
            converter_fingerprint=receipt.converter_fingerprint,
            content_sha256=digest,
        )
    )
    path.write_bytes(encoded)
    path.chmod(0o444)
    return replace(receipt, path=str(path), sha256=digest)


def test_exact_source_and_runtime_contracts() -> None:
    assert SOURCE_CONTRACTS[Split.TRAIN].relative_path.as_posix() == (
        "public_data/coco/rescale_32_1024_bbox_len12000/train.norm.jsonl"
    )
    assert SOURCE_CONTRACTS[Split.VAL].relative_path.as_posix() == (
        "public_data/coco/rescale_32_1024_bbox_len12000/val.norm.jsonl"
    )
    assert SOURCE_CONTRACTS[Split.TRAIN].row_count == 117_266
    assert SOURCE_CONTRACTS[Split.TRAIN].box_count == 849_947
    assert SOURCE_CONTRACTS[Split.VAL].row_count == 4_952
    assert SOURCE_CONTRACTS[Split.VAL].box_count == 36_335
    assert SOURCE_CONTRACTS[Split.TRAIN].sha256 == (
        "d64edc553bdc4d725cb9c3a504f369a9799e8bdde20c8fef0787a09cec33c16a"
    )
    assert SOURCE_CONTRACTS[Split.VAL].sha256 == (
        "a34afb33c567690f56fa3704e213cfc00dc3c000f018d2bb89a7f60139efd795"
    )

    layout = RuntimeLayout.for_repo(REPO_ROOT)
    assert layout.root == REPO_ROOT / Path(RUNTIME_ROOT)
    assert layout.image_root == REPO_ROOT / Path(SHARED_IMAGE_ROOT)
    assert layout.label_studio_state == layout.root / "label-studio" / "state"
    assert layout.for_split(Split.TRAIN).working_norm_jsonl == (
        layout.root / "train" / "working.norm.jsonl"
    )
    assert layout.for_split(Split.TRAIN).task_index_json == (
        layout.root / "train" / "task_index.json"
    )
    assert (
        layout.for_split(Split.VAL).queue_jsonl == layout.root / "val" / "queue.jsonl"
    )
    assert (
        layout.for_split(Split.VAL).project_manifest
        == layout.root / "val" / "project.json"
    )
    assert managed_image_link_plan(layout, Split.TRAIN).target_path == layout.image_root


def test_selected_validation_source_hash_schema_counts_and_identities() -> None:
    contract = SOURCE_CONTRACTS[Split.VAL]
    inspection = inspect_source(
        contract.path(REPO_ROOT),
        repo_root=REPO_ROOT,
        split=Split.VAL,
        registry=COCO80_REGISTRY,
    )
    assert inspection.sha256 == contract.sha256
    assert inspection.row_count == contract.row_count
    assert inspection.box_count == contract.box_count
    assert len(inspection.task_identity_fingerprint) == 64


def test_source_row_identity_schema_and_immutable_fields() -> None:
    source = _row(Split.TRAIN)
    assert validate_source_row(
        source, split=Split.TRAIN, registry=COCO80_REGISTRY
    ) == TaskIdentity(Split.TRAIN, 9)
    assert TaskIdentity(Split.TRAIN, 9) != TaskIdentity(Split.VAL, 9)

    working = deepcopy(source)
    working["images"] = [working["file_name"]]
    working["objects"][0]["bbox_2d"] = [1, 20, 400, 900]
    validate_immutable_row_fields(source, working, split=Split.TRAIN)

    working["width"] += 1
    with pytest.raises(ProjectContractError, match="immutable row field drift: width"):
        validate_immutable_row_fields(source, working, split=Split.TRAIN)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.update(extra=True),
        lambda row: row["metadata"].update(split="val"),
        lambda row: row["objects"][0].update(category_id=2),
        lambda row: row["objects"][0].update(bbox_2d=[0, 10, 0, 20]),
        lambda row: row.update(images=["../../outside.jpg"]),
    ],
)
def test_source_row_rejects_schema_identity_geometry_category_and_locator_drift(
    mutation,
) -> None:
    row = _row(Split.TRAIN)
    mutation(row)
    with pytest.raises(ProjectContractError):
        validate_source_row(row, split=Split.TRAIN, registry=COCO80_REGISTRY)


def test_source_inspection_fails_closed_on_missing_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = _install_small_source(tmp_path, Split.TRAIN, monkeypatch)
    image_path = (
        tmp_path
        / "public_data/coco/rescale_32_1024_bbox/images/train2017/000000000009.jpg"
    )
    image_path.unlink()

    with pytest.raises(ProjectContractError, match="source image missing at line 1"):
        inspect_source(
            source_path,
            repo_root=tmp_path,
            split=Split.TRAIN,
            registry=COCO80_REGISTRY,
        )


def test_source_inspection_rejects_duplicate_task_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duplicate = _row(Split.TRAIN, image_id=9)
    source_path = _install_source_rows(
        tmp_path,
        Split.TRAIN,
        monkeypatch,
        [duplicate, deepcopy(duplicate)],
    )

    with pytest.raises(ProjectContractError, match="duplicate task identity train:9"):
        inspect_source(
            source_path,
            repo_root=tmp_path,
            split=Split.TRAIN,
            registry=COCO80_REGISTRY,
        )


def test_working_image_resolution_is_allowlisted() -> None:
    layout = RuntimeLayout.for_repo(REPO_ROOT)
    resolved = resolve_working_image(
        "images/train2017/000000000009.jpg",
        layout=layout,
        split=Split.TRAIN,
    )
    assert resolved == layout.image_root / "train2017" / "000000000009.jpg"

    for locator in (
        "images/val2017/000000000139.jpg",
        "images/train2017/../val2017/000000000139.jpg",
        "/data/CoordExp/public_data/coco/image.jpg",
    ):
        with pytest.raises(ProjectContractError):
            resolve_working_image(locator, layout=layout, split=Split.TRAIN)


def test_task_import_has_one_editable_annotation_and_never_predictions() -> None:
    payload = make_task_import_payload(
        _row(Split.TRAIN),
        split=Split.TRAIN,
        source_line=1,
        registry=COCO80_REGISTRY,
        bbox_converter=norm1000_bbox_to_label_studio_xywh,
    )
    assert payload["data"]["coordexp_task_key"] == "train:9"
    assert payload["data"]["image"] == "/data/local-files/?d=train2017/000000000009.jpg"
    assert payload["data"]["image"] == local_files_image_locator(
        _row(Split.TRAIN)["file_name"],
        split=Split.TRAIN,
    )
    assert "predictions" not in payload
    assert len(payload["annotations"]) == 1
    assert payload["annotations"][0]["ground_truth"] is False
    result = payload["annotations"][0]["result"][0]
    assert result["meta"]["coordexp_region_key"] == "train:coco:123"
    assert result["value"]["rotation"] == 0
    assert result["image_rotation"] == 0
    assert result["value"]["rectanglelabels"] == ["person"]

    rotated = deepcopy(payload)
    rotated["annotations"][0]["result"][0]["value"]["rotation"] = 0.5
    with pytest.raises(ProjectContractError, match="rotation must be zero"):
        validate_authoritative_task_payload(rotated, registry=COCO80_REGISTRY)
    with_predictions = deepcopy(payload)
    with_predictions["predictions"] = []
    with pytest.raises(ProjectContractError, match="never contain predictions"):
        validate_authoritative_task_payload(with_predictions, registry=COCO80_REGISTRY)

    ground_truth = deepcopy(payload)
    ground_truth["annotations"][0]["ground_truth"] = True
    with pytest.raises(ProjectContractError, match="ground_truth must be false"):
        validate_authoritative_task_payload(ground_truth, registry=COCO80_REGISTRY)

    identity_drift = deepcopy(payload)
    identity_drift["annotations"][0]["result"][0]["id"] = "train:coco:999"
    with pytest.raises(ProjectContractError, match="hidden region identity"):
        validate_authoritative_task_payload(identity_drift, registry=COCO80_REGISTRY)

    image_drift = deepcopy(payload)
    image_drift["data"]["image"] = "/data/local-files/?d=train2017/000000000025.jpg"
    with pytest.raises(ProjectContractError, match="image locator is not canonical"):
        validate_authoritative_task_payload(image_drift, registry=COCO80_REGISTRY)


def test_frozen_task_import_detaches_every_nested_original_reference() -> None:
    row = _row(Split.TRAIN)
    payload = make_task_import_payload(
        row,
        split=Split.TRAIN,
        source_line=1,
        registry=COCO80_REGISTRY,
        bbox_converter=norm1000_bbox_to_label_studio_xywh,
    )
    frozen = FrozenTaskImport.freeze(payload)
    stable_bytes = frozen.canonical_json

    row["objects"][0]["bbox_2d"][0] = 111
    payload["annotations"][0]["ground_truth"] = True
    payload["annotations"][0]["result"][0]["meta"]["last_committed_bbox"][0] = 222

    assert frozen.canonical_json == stable_bytes
    assert b'"ground_truth":false' in frozen.canonical_json
    assert b'"last_committed_bbox":[0,10,500,999]' in frozen.canonical_json


def test_adapter_send_payloads_are_defensive_copies_of_immutable_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = _small_plan(tmp_path, Split.TRAIN, monkeypatch)
    val = _small_plan(tmp_path, Split.VAL, monkeypatch)
    plan = plan_instance_bootstrap(
        {Split.TRAIN: train, Split.VAL: val},
        _AttestingAdapter({Split.TRAIN: None, Split.VAL: None}),
    )
    action = plan.projects[0]
    stable_sidecar = Path(train.task_manifest.task_index.path).read_bytes()
    stable_manifest = train.manifest.to_dict()

    first_send = next(action.iter_task_import_chunks(1))
    first_send[0]["annotations"][0]["ground_truth"] = True
    first_send[0]["annotations"][0]["result"][0]["meta"][
        "last_committed_bbox"
    ][0] = 333
    second_send = next(action.iter_task_import_chunks(1))

    assert second_send[0]["annotations"][0]["ground_truth"] is False
    assert second_send[0]["annotations"][0]["result"][0]["meta"][
        "last_committed_bbox"
    ] == [0, 10, 500, 999]
    assert first_send is not second_send
    assert first_send[0] is not second_send[0]
    assert Path(train.task_manifest.task_index.path).read_bytes() == stable_sidecar
    assert train.manifest.to_dict() == stable_manifest
    assert not hasattr(train, "task_imports")
    assert not hasattr(action, "missing_task_imports")


def test_task_index_tamper_fails_before_live_adapter_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = _small_plan(tmp_path, Split.TRAIN, monkeypatch)
    val = _small_plan(tmp_path, Split.VAL, monkeypatch)
    sidecar = Path(train.task_manifest.task_index.path)
    lines = sidecar.read_bytes().splitlines(keepends=True)
    record = json.loads(lines[1])
    record["entry"]["task_data_fingerprint"] = "0" * 64
    lines[1] = (
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    sidecar.chmod(0o644)
    sidecar.write_bytes(b"".join(lines))
    sidecar.chmod(0o444)
    adapter = _AttestingAdapter({Split.TRAIN: None, Split.VAL: None})

    with pytest.raises(ProjectContractError, match="sidecar hash drift"):
        plan_instance_bootstrap(
            {Split.TRAIN: train, Split.VAL: val},
            adapter,
        )
    assert adapter.calls == []


def test_streamed_task_fingerprints_exactly_match_legacy_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _install_source_rows(
        tmp_path,
        Split.TRAIN,
        monkeypatch,
        [_row(Split.TRAIN, image_id=9), _row(Split.TRAIN, image_id=25)],
    )
    plan = _build_plan(tmp_path, Split.TRAIN, source)
    entries = list(plan.task_manifest.iter_entries())
    legacy_body = {
        "split": Split.TRAIN.value,
        "task_count": len(entries),
        "entries": [entry.to_dict() for entry in entries],
    }
    assert plan.task_manifest.fingerprint == fingerprint_json(legacy_body)
    assert plan.task_manifest.identity_fingerprint == fingerprint_json(
        sorted(entry.identity.key for entry in entries)
    )
    assert plan.source_inspection.task_identity_fingerprint == fingerprint_json(
        [entry.identity.key for entry in entries]
    )
    assert (
        plan.source_inspection.task_identity_fingerprint
        != plan.task_manifest.identity_fingerprint
    )


def test_project_task_storage_fingerprints_and_bootstrap_are_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = _small_plan(tmp_path, Split.TRAIN, monkeypatch)
    val = _small_plan(tmp_path, Split.VAL, monkeypatch)
    repeated = _small_plan(tmp_path, Split.TRAIN, monkeypatch)

    assert train.task_manifest.fingerprint == repeated.task_manifest.fingerprint
    assert train.task_manifest.task_index == repeated.task_manifest.task_index
    assert Path(train.task_manifest.task_index.path).is_file()
    assert train.storage_manifest.fingerprint == repeated.storage_manifest.fingerprint
    assert train.manifest.fingerprint == repeated.manifest.fingerprint
    assert train.source_inspection.fingerprint == repeated.source_inspection.fingerprint
    assert train.controls.authoritative_annotations_per_task == 1
    assert not train.controls.allow_alternate_annotations
    assert not train.controls.allow_annotation_deletion
    assert not train.controls.show_native_submit
    assert not train.controls.show_native_skip
    assert train.controls.allow_region_crud
    assert train.storage_manifest.document_root == str(
        RuntimeLayout.for_repo(tmp_path).image_root
    )
    assert (
        train.storage_manifest.storage_identity != val.storage_manifest.storage_identity
    )
    assert train.manifest.project_identity != val.manifest.project_identity

    create = plan_instance_bootstrap(
        {Split.TRAIN: train, Split.VAL: val},
        _AttestingAdapter({Split.TRAIN: None, Split.VAL: None}),
    )
    assert [item.action for item in create.projects] == [
        BootstrapAction.CREATE,
        BootstrapAction.CREATE,
    ]
    assert [item.missing_task_count for item in create.projects] == [1, 1]
    assert create.manifest == build_instance_bootstrap_manifest(
        {Split.TRAIN: train, Split.VAL: val}
    )
    resume_after_manifest = plan_instance_bootstrap(
        {Split.TRAIN: train, Split.VAL: val},
        _AttestingAdapter(
            {Split.TRAIN: None, Split.VAL: None},
            bootstrap_manifest=create.manifest.to_dict(),
        ),
    )
    assert [item.action for item in resume_after_manifest.projects] == [
        BootstrapAction.CREATE,
        BootstrapAction.CREATE,
    ]
    live_train = _live_attestation(train, project_id=1)
    live_val = _live_attestation(val, project_id=2)
    adapter = _adapter_for(
        train,
        val,
        {Split.TRAIN: live_train, Split.VAL: live_val},
    )
    reuse = plan_instance_bootstrap(
        {Split.TRAIN: train, Split.VAL: val},
        adapter,
    )
    assert [item.action for item in reuse.projects] == [
        BootstrapAction.REUSE,
        BootstrapAction.REUSE,
    ]
    assert [item.missing_task_count for item in reuse.projects] == [0, 0]
    assert adapter.calls == [Split.TRAIN, Split.VAL]
    assert [item.live_attestation_fingerprint for item in reuse.projects] == [
        live_train.fingerprint,
        live_val.fingerprint,
    ]


def test_warm_build_reuses_attested_sidecar_without_rewriting_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _install_small_source(tmp_path, Split.TRAIN, monkeypatch)
    first = _build_plan(tmp_path, Split.TRAIN, source)
    sidecar = Path(first.task_manifest.task_index.path)
    anchor = project_module._task_index_anchor_path(
        sidecar.parent,
        converter_fingerprint=first.task_manifest.task_index.converter_fingerprint,
        build_key=first.task_manifest.task_index.build_key,
    )
    before = (
        sidecar.stat().st_ino,
        sidecar.stat().st_mtime_ns,
        anchor.stat().st_ino,
        anchor.stat().st_mtime_ns,
    )

    monkeypatch.setattr(
        project_module,
        "_build_task_index_from_source",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("warm build attempted to reconstruct the sidecar")
        ),
    )
    repeated = _build_plan(tmp_path, Split.TRAIN, source)

    assert repeated.task_manifest.task_index == first.task_manifest.task_index
    assert before == (
        sidecar.stat().st_ino,
        sidecar.stat().st_mtime_ns,
        anchor.stat().st_ino,
        anchor.stat().st_mtime_ns,
    )


def test_warm_build_manifest_and_reuse_plan_hash_each_file_once_without_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_source = _install_small_source(tmp_path, Split.TRAIN, monkeypatch)
    val_source = _install_small_source(tmp_path, Split.VAL, monkeypatch)
    cold_train = _build_plan(tmp_path, Split.TRAIN, train_source)
    cold_val = _build_plan(tmp_path, Split.VAL, val_source)
    sidecars = {
        Path(cold_train.task_manifest.task_index.path),
        Path(cold_val.task_manifest.task_index.path),
    }
    sources = {train_source, val_source}
    before = {
        path: (path.stat().st_ino, path.stat().st_mtime_ns, path.stat().st_size)
        for path in sidecars
    }
    hash_completions: dict[Path, int] = {}

    def hook(path: Path, stage: str, _byte_count: int) -> None:
        if stage == "hash_complete":
            resolved = path.resolve()
            hash_completions[resolved] = hash_completions.get(resolved, 0) + 1

    project_module._clear_byte_attestation_cache()
    monkeypatch.setattr(project_module, "_STABLE_FILE_TEST_HOOK", hook)
    monkeypatch.setattr(
        project_module,
        "_iter_task_index_records",
        lambda _receipt: (_ for _ in ()).throw(
            AssertionError("warm REUSE parsed task-index records")
        ),
    )

    train = _build_plan(tmp_path, Split.TRAIN, train_source)
    val = _build_plan(tmp_path, Split.VAL, val_source)
    adapter = _adapter_for(
        train,
        val,
        {
            Split.TRAIN: _live_attestation(train, project_id=1),
            Split.VAL: _live_attestation(val, project_id=2),
        },
    )
    planned = plan_instance_bootstrap(
        {Split.TRAIN: train, Split.VAL: val},
        adapter,
    )

    assert [item.action for item in planned.projects] == [
        BootstrapAction.REUSE,
        BootstrapAction.REUSE,
    ]
    assert {path: hash_completions[path.resolve()] for path in sidecars} == {
        path: 1 for path in sidecars
    }
    assert {path: hash_completions[path.resolve()] for path in sources} == {
        path: 1 for path in sources
    }
    assert before == {
        path: (path.stat().st_ino, path.stat().st_mtime_ns, path.stat().st_size)
        for path in sidecars
    }


def test_every_build_force_hashes_exact_source_even_when_memoized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _install_small_source(tmp_path, Split.TRAIN, monkeypatch)
    _build_plan(tmp_path, Split.TRAIN, source)
    source_hashes = 0

    def hook(path: Path, stage: str, _byte_count: int) -> None:
        nonlocal source_hashes
        if path == source and stage == "hash_complete":
            source_hashes += 1

    project_module._clear_byte_attestation_cache()
    monkeypatch.setattr(project_module, "_STABLE_FILE_TEST_HOOK", hook)
    _build_plan(tmp_path, Split.TRAIN, source)
    _build_plan(tmp_path, Split.TRAIN, source)

    assert source_hashes == 2


def test_source_symlink_is_rejected_even_when_target_bytes_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _install_small_source(tmp_path, Split.TRAIN, monkeypatch)
    backing = source.with_suffix(".backing")
    source.rename(backing)
    source.symlink_to(backing)

    with pytest.raises(ProjectContractError, match="regular non-symlink"):
        _build_plan(tmp_path, Split.TRAIN, source)


def test_source_stat_race_during_hash_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _install_small_source(tmp_path, Split.TRAIN, monkeypatch)
    changed = False

    def hook(path: Path, stage: str, _byte_count: int) -> None:
        nonlocal changed
        if path == source and stage == "hash_chunk" and not changed:
            changed = True
            metadata = path.stat()
            os.utime(
                path,
                ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 1),
            )

    monkeypatch.setattr(project_module, "_STABLE_FILE_TEST_HOOK", hook)
    with pytest.raises(ProjectContractError, match="changed during attestation"):
        _build_plan(tmp_path, Split.TRAIN, source)


def test_cold_source_stream_rejects_inode_swap_before_index_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _install_source_rows(
        tmp_path,
        Split.TRAIN,
        monkeypatch,
        [_row(Split.TRAIN, image_id=9), _row(Split.TRAIN, image_id=25)],
    )
    replacement = source.with_suffix(".replacement")
    replacement.write_bytes(source.read_bytes())
    original = source.with_suffix(".opened")
    changed = False

    def hook(path: Path, stage: str, record_count: int) -> None:
        nonlocal changed
        if (
            path == source
            and stage == "source_record_complete"
            and record_count == 1
            and not changed
        ):
            changed = True
            source.rename(original)
            replacement.rename(source)

    monkeypatch.setattr(project_module, "_STABLE_FILE_TEST_HOOK", hook)
    with pytest.raises(ProjectContractError, match="changed during attestation"):
        _build_plan(tmp_path, Split.TRAIN, source)

    index_root = (
        RuntimeLayout.for_repo(tmp_path).for_split(Split.TRAIN).root
        / project_module.TASK_INDEX_DIRECTORY_NAME
    )
    assert changed
    assert list(index_root.glob("task-index-v*.jsonl")) == []
    assert list(index_root.glob("build-v*.json")) == []
    assert list(index_root.glob(".builder-*")) == []
    assert list(index_root.glob(".identity-sort-*")) == []


def test_sidecar_stat_race_during_hash_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _install_small_source(tmp_path, Split.TRAIN, monkeypatch)
    project = _build_plan(tmp_path, Split.TRAIN, source)
    sidecar = Path(project.task_manifest.task_index.path)
    changed = False

    def hook(path: Path, stage: str, _byte_count: int) -> None:
        nonlocal changed
        if path == sidecar and stage == "hash_chunk" and not changed:
            changed = True
            metadata = path.stat()
            os.utime(
                path,
                ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 1),
            )

    project_module._clear_byte_attestation_cache()
    monkeypatch.setattr(project_module, "_STABLE_FILE_TEST_HOOK", hook)
    with pytest.raises(ProjectContractError, match="changed during attestation"):
        _build_plan(tmp_path, Split.TRAIN, source)


def test_sidecar_same_size_tamper_with_restored_mtime_fails_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _install_small_source(tmp_path, Split.TRAIN, monkeypatch)
    project = _build_plan(tmp_path, Split.TRAIN, source)
    sidecar = Path(project.task_manifest.task_index.path)
    metadata = sidecar.stat()
    tampered = bytearray(sidecar.read_bytes())
    offset = next(
        index for index, value in enumerate(tampered) if value not in {10, 13}
    )
    tampered[offset] = ord("X") if tampered[offset] != ord("X") else ord("Y")
    sidecar.chmod(0o644)
    sidecar.write_bytes(tampered)
    os.utime(sidecar, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
    sidecar.chmod(0o444)

    project_module._clear_byte_attestation_cache()
    with pytest.raises(ProjectContractError, match="sidecar hash drift"):
        _build_plan(tmp_path, Split.TRAIN, source)


def test_sidecar_same_bytes_stat_change_forces_rehash_and_is_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _install_small_source(tmp_path, Split.TRAIN, monkeypatch)
    project = _build_plan(tmp_path, Split.TRAIN, source)
    sidecar = Path(project.task_manifest.task_index.path)
    sidecar_hashes = 0

    def hook(path: Path, stage: str, _byte_count: int) -> None:
        nonlocal sidecar_hashes
        if path == sidecar and stage == "hash_complete":
            sidecar_hashes += 1

    project_module._clear_byte_attestation_cache()
    monkeypatch.setattr(project_module, "_STABLE_FILE_TEST_HOOK", hook)
    _build_plan(tmp_path, Split.TRAIN, source)
    metadata = sidecar.stat()
    os.utime(
        sidecar,
        ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 1),
    )
    _build_plan(tmp_path, Split.TRAIN, source)

    assert sidecar_hashes == 2


def test_sidecar_memo_hit_restats_and_rejects_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _install_small_source(tmp_path, Split.TRAIN, monkeypatch)
    project = _build_plan(tmp_path, Split.TRAIN, source)
    sidecar = Path(project.task_manifest.task_index.path)
    project_module._clear_byte_attestation_cache()
    _build_plan(tmp_path, Split.TRAIN, source)
    changed = False

    def hook(path: Path, stage: str, _byte_count: int) -> None:
        nonlocal changed
        if path == sidecar and stage == "memo_hit" and not changed:
            changed = True
            metadata = path.stat()
            os.utime(
                path,
                ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 1),
            )

    monkeypatch.setattr(project_module, "_STABLE_FILE_TEST_HOOK", hook)
    with pytest.raises(ProjectContractError, match="changed during attestation"):
        _build_plan(tmp_path, Split.TRAIN, source)


def test_warm_anchor_requires_exact_read_only_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _install_small_source(tmp_path, Split.TRAIN, monkeypatch)
    project = _build_plan(tmp_path, Split.TRAIN, source)
    receipt = project.task_manifest.task_index
    anchor = project_module._task_index_anchor_path(
        Path(receipt.path).parent,
        converter_fingerprint=receipt.converter_fingerprint,
        build_key=receipt.build_key,
    )
    anchor.chmod(0o644)

    with pytest.raises(ProjectContractError, match="mode drift"):
        _build_plan(tmp_path, Split.TRAIN, source)


def test_warm_anchor_schema_and_hash_read_are_stable_and_strict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _install_small_source(tmp_path, Split.TRAIN, monkeypatch)
    project = _build_plan(tmp_path, Split.TRAIN, source)
    receipt = project.task_manifest.task_index
    anchor = project_module._task_index_anchor_path(
        Path(receipt.path).parent,
        converter_fingerprint=receipt.converter_fingerprint,
        build_key=receipt.build_key,
    )
    payload = json.loads(anchor.read_bytes())
    payload["schema_version"] = 1
    anchor.chmod(0o644)
    anchor.write_bytes(project_module._canonical_json_bytes(payload) + b"\n")
    anchor.chmod(0o444)

    with pytest.raises(ProjectContractError, match="must be at least 2"):
        _build_plan(tmp_path, Split.TRAIN, source)


def test_warm_anchor_read_race_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _install_small_source(tmp_path, Split.TRAIN, monkeypatch)
    project = _build_plan(tmp_path, Split.TRAIN, source)
    receipt = project.task_manifest.task_index
    anchor = project_module._task_index_anchor_path(
        Path(receipt.path).parent,
        converter_fingerprint=receipt.converter_fingerprint,
        build_key=receipt.build_key,
    )
    changed = False

    def hook(path: Path, stage: str, _byte_count: int) -> None:
        nonlocal changed
        if path == anchor and stage == "read_complete" and not changed:
            changed = True
            metadata = path.stat()
            os.utime(
                path,
                ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 1),
            )

    monkeypatch.setattr(project_module, "_STABLE_FILE_TEST_HOOK", hook)
    with pytest.raises(ProjectContractError, match="changed during attestation"):
        _build_plan(tmp_path, Split.TRAIN, source)


def test_warm_sidecar_symlink_is_rejected_even_when_target_bytes_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _install_small_source(tmp_path, Split.TRAIN, monkeypatch)
    project = _build_plan(tmp_path, Split.TRAIN, source)
    sidecar = Path(project.task_manifest.task_index.path)
    backing = sidecar.with_suffix(".backing")
    sidecar.rename(backing)
    sidecar.symlink_to(backing.name)

    project_module._clear_byte_attestation_cache()
    with pytest.raises(ProjectContractError, match="regular non-symlink"):
        _build_plan(tmp_path, Split.TRAIN, source)


def test_explicit_full_validation_rechecks_every_indexed_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _install_small_source(tmp_path, Split.TRAIN, monkeypatch)
    project = _build_plan(tmp_path, Split.TRAIN, source)
    image = (
        tmp_path
        / "public_data/coco/rescale_32_1024_bbox"
        / _row(Split.TRAIN)["file_name"]
    )
    image.unlink()

    with pytest.raises(ProjectContractError, match="task-index image missing"):
        project.task_manifest.task_index.validate_for_repo(tmp_path)


def test_lost_task_create_response_reconciles_by_stable_source_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_source = _install_source_rows(
        tmp_path,
        Split.TRAIN,
        monkeypatch,
        [_row(Split.TRAIN, image_id=9), _row(Split.TRAIN, image_id=25)],
    )
    val_source = _install_small_source(tmp_path, Split.VAL, monkeypatch)
    train = _build_plan(tmp_path, Split.TRAIN, train_source)
    val = _build_plan(tmp_path, Split.VAL, val_source)
    full_train = _live_attestation(train, project_id=1)
    partial_train = _live_attestation(
        train,
        project_id=1,
        observed_task_count=1,
    )
    live_val = _live_attestation(val, project_id=2)

    retry = plan_instance_bootstrap(
        {Split.TRAIN: train, Split.VAL: val},
        _adapter_for(
            train,
            val,
            {Split.TRAIN: partial_train, Split.VAL: live_val},
        ),
    )
    train_retry = retry.projects[0]
    assert train_retry.action is BootstrapAction.RECONCILE
    assert train_retry.observed_task_count == 1
    assert train_retry.missing_task_count == 1
    assert [
        task["data"]["coordexp_task_key"]
        for chunk in train_retry.iter_task_import_chunks(1)
        for task in chunk
    ] == ["train:9", "train:25"]

    reconciled = plan_instance_bootstrap(
        {Split.TRAIN: train, Split.VAL: val},
        _adapter_for(
            train,
            val,
            {Split.TRAIN: full_train, Split.VAL: live_val},
        ),
    )
    assert reconciled.projects[0].action is BootstrapAction.REUSE
    assert reconciled.projects[0].missing_task_count == 0


def test_bootstrap_rejects_instance_manifest_and_cross_split_project_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = _small_plan(tmp_path, Split.TRAIN, monkeypatch)
    val = _small_plan(tmp_path, Split.VAL, monkeypatch)
    live_train = _live_attestation(train, project_id=1)
    live_val = _live_attestation(val, project_id=2)
    manifest = build_instance_bootstrap_manifest(
        {Split.TRAIN: train, Split.VAL: val}
    ).to_dict()
    manifest["local_files_document_root"] = str(tmp_path / "outside")

    with pytest.raises(ManifestDriftError) as error:
        plan_instance_bootstrap(
            {Split.TRAIN: train, Split.VAL: val},
            _AttestingAdapter(
                {Split.TRAIN: live_train, Split.VAL: live_val},
                bootstrap_manifest=manifest,
            ),
        )
    assert "local_files_document_root" in error.value.mismatches

    with pytest.raises(ManifestDriftError, match="cross-split duplicate"):
        plan_instance_bootstrap(
            {Split.TRAIN: train, Split.VAL: val},
            _adapter_for(
                train,
                val,
                {
                    Split.TRAIN: live_train,
                    Split.VAL: _live_attestation(val, project_id=1),
                },
            ),
        )


def test_manifest_comparison_fails_closed_and_reports_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _small_plan(tmp_path, Split.TRAIN, monkeypatch)
    observed = plan.manifest.to_dict()
    observed["source_sha256"] = "0" * 64
    observed["unexpected"] = True

    mismatches = compare_manifests(plan.manifest, observed)
    assert "source_sha256" in mismatches
    assert "unexpected (unexpected)" in mismatches
    with pytest.raises(ManifestDriftError) as error:
        assert_manifest_matches(plan.manifest, observed)
    assert error.value.mismatches == mismatches


def test_project_plan_rejects_source_mutation_that_preserves_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = _install_small_source(tmp_path, Split.TRAIN, monkeypatch)
    config = build_label_config()
    accepted = build_split_project_plan(
        source_path,
        repo_root=tmp_path,
        split=Split.TRAIN,
        vendor_revision="pinned-vendor-revision",
        label_config=config,
        label_config_fingerprint=label_config_fingerprint(config),
    )
    original_hash = accepted.source_inspection.sha256

    mutated = _row(Split.TRAIN)
    mutated["objects"][0]["bbox_2d"] = [1, 10, 501, 999]
    source_path.write_text(
        json.dumps(mutated, ensure_ascii=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ProjectContractError, match="source hash drift"):
        build_split_project_plan(
            source_path,
            repo_root=tmp_path,
            split=Split.TRAIN,
            vendor_revision="pinned-vendor-revision",
            label_config=config,
            label_config_fingerprint=label_config_fingerprint(config),
        )
    assert sha256_file(source_path) != original_hash
    index_root = (
        RuntimeLayout.for_repo(tmp_path).for_split(Split.TRAIN).root
        / project_module.TASK_INDEX_DIRECTORY_NAME
    )
    assert list(index_root.glob(".builder-*")) == []
    assert list(index_root.glob(".identity-sort-*")) == []


def test_saved_manifest_without_live_attestation_can_never_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = _small_plan(tmp_path, Split.TRAIN, monkeypatch)
    val = _small_plan(tmp_path, Split.VAL, monkeypatch)
    adapter = _SavedManifestOnlyAdapter(
        {
            Split.TRAIN: train.manifest.to_dict(),
            Split.VAL: val.manifest.to_dict(),
        },
        build_instance_bootstrap_manifest(
            {Split.TRAIN: train, Split.VAL: val}
        ).to_dict(),
    )

    with pytest.raises(ProjectContractError, match="live project attestation"):
        plan_instance_bootstrap({Split.TRAIN: train, Split.VAL: val}, adapter)


def test_reuse_rejects_task_set_manifest_and_count_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = _small_plan(tmp_path, Split.TRAIN, monkeypatch)
    val = _small_plan(tmp_path, Split.VAL, monkeypatch)
    live_train = _live_attestation(train, project_id=1)
    wrong_manifest = replace(
        live_train.task_set,
        expected_task_manifest_fingerprint="0" * 64,
    )
    drifted = replace(live_train, task_set=wrong_manifest)

    with pytest.raises(ManifestDriftError) as error:
        plan_instance_bootstrap(
            {Split.TRAIN: train, Split.VAL: val},
            _adapter_for(
                train,
                val,
                {
                    Split.TRAIN: drifted,
                    Split.VAL: _live_attestation(val, project_id=2),
                },
            ),
        )
    assert (
        "live.task_set.expected_task_manifest_fingerprint"
        in error.value.mismatches
    )

    wrong_count = replace(
        live_train.task_set,
        observed_task_count=0,
        missing_task_count=0,
    )
    with pytest.raises(ManifestDriftError) as error:
        plan_instance_bootstrap(
            {Split.TRAIN: train, Split.VAL: val},
            _adapter_for(
                train,
                val,
                {
                    Split.TRAIN: replace(live_train, task_set=wrong_count),
                    Split.VAL: _live_attestation(val, project_id=2),
                },
            ),
        )
    assert "live.task_set.task_count" in error.value.mismatches


def test_task_import_iteration_is_restartable_and_chunk_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [_row(Split.TRAIN, image_id=100 + index) for index in range(7)]
    source = _install_source_rows(tmp_path, Split.TRAIN, monkeypatch, rows)
    project = _build_plan(tmp_path, Split.TRAIN, source)
    first = list(project.iter_task_import_chunks(3))
    second = list(project.iter_task_import_chunks(3))
    assert [len(chunk) for chunk in first] == [3, 3, 1]
    assert [
        payload["data"]["coordexp_task_key"]
        for chunk in first
        for payload in chunk
    ] == [
        payload["data"]["coordexp_task_key"]
        for chunk in second
        for payload in chunk
    ]
    assert not hasattr(project.task_manifest, "entries")


def test_task_import_iteration_keeps_explicit_full_record_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _small_plan(tmp_path, Split.TRAIN, monkeypatch)
    original = project_module._iter_task_index_records
    opened_descriptors: list[tuple[int, int]] = []

    def counted(receipt, *, handle):
        opened_descriptors.append(
            (handle.fileno(), os.fstat(handle.fileno()).st_ino)
        )
        yield from original(receipt, handle=handle)

    monkeypatch.setattr(project_module, "_iter_task_index_records", counted)
    chunks = list(project.iter_task_import_chunks(1))

    assert len(chunks) == 1
    assert len(opened_descriptors) == 2
    assert opened_descriptors[0] == opened_descriptors[1]


def test_task_import_rejects_path_replacement_between_full_passes_before_yield(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _small_plan(tmp_path, Split.TRAIN, monkeypatch)
    receipt = project.task_manifest.task_index
    sidecar = Path(receipt.path)
    replacement = sidecar.with_name("replacement.jsonl")
    replacement.write_bytes(sidecar.read_bytes())
    replacement.chmod(0o444)
    validated_inode = sidecar.stat().st_ino
    opened = sidecar.with_name("validated-open.jsonl")
    changed = False

    def hook(path: Path, stage: str, _byte_count: int) -> None:
        nonlocal changed
        if path == sidecar and stage == "task_index_validation_complete":
            changed = True
            sidecar.rename(opened)
            replacement.rename(sidecar)

    monkeypatch.setattr(project_module, "_STABLE_FILE_TEST_HOOK", hook)
    records = receipt.iter_records()
    with pytest.raises(ProjectContractError, match="changed during attestation"):
        next(records)

    assert changed
    assert opened.stat().st_ino == validated_inode
    assert sidecar.stat().st_ino != validated_inode


def test_task_import_final_recheck_rejects_metadata_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _small_plan(tmp_path, Split.TRAIN, monkeypatch)
    receipt = project.task_manifest.task_index
    sidecar = Path(receipt.path)
    changed = False

    def hook(path: Path, stage: str, _byte_count: int) -> None:
        nonlocal changed
        if path == sidecar and stage == "task_index_import_complete":
            changed = True
            metadata = sidecar.stat()
            os.utime(
                sidecar,
                ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 1),
            )

    monkeypatch.setattr(project_module, "_STABLE_FILE_TEST_HOOK", hook)
    with pytest.raises(ProjectContractError, match="changed during attestation"):
        list(receipt.iter_records())

    assert changed


def test_published_task_index_is_the_only_post_plan_import_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _install_source_rows(
        tmp_path,
        Split.TRAIN,
        monkeypatch,
        [_row(Split.TRAIN, image_id=9), _row(Split.TRAIN, image_id=25)],
    )
    project = _build_plan(tmp_path, Split.TRAIN, source)
    source.unlink()

    keys = [
        payload["data"]["coordexp_task_key"]
        for chunk in project.iter_task_import_chunks(1)
        for payload in chunk
    ]
    assert keys == ["train:9", "train:25"]


def test_custom_first_cannot_seed_default_converter_anchor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _install_small_source(tmp_path, Split.TRAIN, monkeypatch)
    config = build_label_config()
    custom = build_split_project_plan(
        source,
        repo_root=tmp_path,
        split=Split.TRAIN,
        vendor_revision="pinned-vendor-revision",
        label_config=config,
        label_config_fingerprint=label_config_fingerprint(config),
        registry=COCO80_REGISTRY,
        bbox_converter=lambda _bbox: (1.0, 1.0, 50.0, 50.0),
    )
    index_root = Path(custom.task_manifest.task_index.path).parent
    assert custom.task_manifest.task_index.converter_fingerprint == (
        project_module.UNFINGERPRINTED_CUSTOM_BBOX_CONVERTER_FINGERPRINT
    )
    assert list(index_root.glob("build-v*.json")) == []

    default = _build_plan(tmp_path, Split.TRAIN, source)
    assert default.task_manifest.task_index.converter_fingerprint == (
        project_module.CANONICAL_BBOX_CONVERTER_FINGERPRINT
    )
    assert default.task_manifest.task_index != custom.task_manifest.task_index
    custom_payload = next(custom.iter_task_import_chunks(1))[0]
    default_payload = next(default.iter_task_import_chunks(1))[0]
    assert custom_payload["annotations"][0]["result"][0]["value"]["x"] == 1.0
    assert default_payload["annotations"][0]["result"][0]["value"]["x"] == 0.0
    assert len(list(index_root.glob("build-v*.json"))) == 1


def test_default_first_custom_converter_never_reuses_default_anchor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _install_small_source(tmp_path, Split.TRAIN, monkeypatch)
    default = _build_plan(tmp_path, Split.TRAIN, source)
    config = build_label_config()
    custom = build_split_project_plan(
        source,
        repo_root=tmp_path,
        split=Split.TRAIN,
        vendor_revision="pinned-vendor-revision",
        label_config=config,
        label_config_fingerprint=label_config_fingerprint(config),
        registry=COCO80_REGISTRY,
        bbox_converter=lambda _bbox: (1.0, 1.0, 50.0, 50.0),
    )

    assert custom.task_manifest.task_index != default.task_manifest.task_index
    assert custom.task_manifest.task_index.converter_fingerprint == (
        project_module.UNFINGERPRINTED_CUSTOM_BBOX_CONVERTER_FINGERPRINT
    )
    assert next(custom.iter_task_import_chunks(1))[0]["annotations"][0]["result"][
        0
    ]["value"]["x"] == 1.0
    assert _build_plan(tmp_path, Split.TRAIN, source).task_manifest.task_index == (
        default.task_manifest.task_index
    )


def test_same_explicit_custom_converter_fingerprint_and_output_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _install_small_source(tmp_path, Split.TRAIN, monkeypatch)
    config = build_label_config()
    converter_fingerprint = fingerprint_json(
        {"bbox_converter_semantics": "test-fixed-geometry-v1"}
    )

    def custom_converter(_bbox):
        return (1.0, 1.0, 50.0, 50.0)

    def build_custom():
        return build_split_project_plan(
            source,
            repo_root=tmp_path,
            split=Split.TRAIN,
            vendor_revision="pinned-vendor-revision",
            label_config=config,
            label_config_fingerprint=label_config_fingerprint(config),
            registry=COCO80_REGISTRY,
            bbox_converter=custom_converter,
            converter_fingerprint=converter_fingerprint,
        )

    first = build_custom()
    first_inode = Path(first.task_manifest.task_index.path).stat().st_ino
    repeated = build_custom()

    assert repeated.task_manifest.task_index == first.task_manifest.task_index
    assert Path(repeated.task_manifest.task_index.path).stat().st_ino == first_inode
    assert repeated.task_manifest.task_index.converter_fingerprint == (
        converter_fingerprint
    )
    assert converter_fingerprint in Path(first.task_manifest.task_index.path).name
    index_root = Path(first.task_manifest.task_index.path).parent
    assert len(list(index_root.glob("build-v*.json"))) == 1


def test_same_explicit_custom_fingerprint_with_different_output_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _install_small_source(tmp_path, Split.TRAIN, monkeypatch)
    config = build_label_config()
    converter_fingerprint = fingerprint_json(
        {"bbox_converter_semantics": "test-fixed-geometry-v1"}
    )

    def build_custom(value: float):
        return build_split_project_plan(
            source,
            repo_root=tmp_path,
            split=Split.TRAIN,
            vendor_revision="pinned-vendor-revision",
            label_config=config,
            label_config_fingerprint=label_config_fingerprint(config),
            registry=COCO80_REGISTRY,
            bbox_converter=lambda _bbox: (value, value, 50.0, 50.0),
            converter_fingerprint=converter_fingerprint,
        )

    build_custom(1.0)

    with pytest.raises(ProjectContractError, match="build receipt conflict"):
        build_custom(2.0)


def test_converter_fingerprint_reserved_markers_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _install_small_source(tmp_path, Split.TRAIN, monkeypatch)
    config = build_label_config()
    common = {
        "repo_root": tmp_path,
        "split": Split.TRAIN,
        "vendor_revision": "pinned-vendor-revision",
        "label_config": config,
        "label_config_fingerprint": label_config_fingerprint(config),
        "registry": COCO80_REGISTRY,
    }

    with pytest.raises(ProjectContractError, match="canonical semantics"):
        build_split_project_plan(
            source,
            bbox_converter=norm1000_bbox_to_label_studio_xywh,
            converter_fingerprint=fingerprint_json({"wrong": "default"}),
            **common,
        )
    with pytest.raises(ProjectContractError, match="reserved semantics marker"):
        build_split_project_plan(
            source,
            bbox_converter=lambda _bbox: (1.0, 1.0, 50.0, 50.0),
            converter_fingerprint=(
                project_module.CANONICAL_BBOX_CONVERTER_FINGERPRINT
            ),
            **common,
        )


def test_concurrent_same_content_builders_converge_on_one_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _install_source_rows(
        tmp_path,
        Split.TRAIN,
        monkeypatch,
        [_row(Split.TRAIN, image_id=9), _row(Split.TRAIN, image_id=25)],
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        plans = list(
            executor.map(
                lambda _index: _build_plan(tmp_path, Split.TRAIN, source),
                range(2),
            )
        )

    receipts = [plan.task_manifest.task_index for plan in plans]
    assert receipts[0] == receipts[1]
    index_root = Path(receipts[0].path).parent
    assert len(list(index_root.glob("task-index-v*.jsonl"))) == 1
    assert len(list(index_root.glob("build-v*.json"))) == 1
    assert list(index_root.glob(".builder-*")) == []
    assert list(index_root.glob(".identity-sort-*")) == []


def test_immutable_index_and_anchor_fsync_after_read_only_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_fchmod = os.fchmod
    original_fsync = os.fsync
    events: list[tuple[str, int, int]] = []

    def tracked_fchmod(descriptor: int, mode: int) -> None:
        original_fchmod(descriptor, mode)
        metadata = os.fstat(descriptor)
        events.append(("fchmod", metadata.st_ino, metadata.st_mode & 0o777))

    def tracked_fsync(descriptor: int) -> None:
        metadata = os.fstat(descriptor)
        events.append(("fsync", metadata.st_ino, metadata.st_mode & 0o777))
        original_fsync(descriptor)

    monkeypatch.setattr(os, "fchmod", tracked_fchmod)
    monkeypatch.setattr(os, "fsync", tracked_fsync)
    project = _small_plan(tmp_path, Split.TRAIN, monkeypatch)
    receipt = project.task_manifest.task_index
    sidecar = Path(receipt.path)
    anchor = project_module._task_index_anchor_path(
        sidecar.parent,
        converter_fingerprint=receipt.converter_fingerprint,
        build_key=receipt.build_key,
    )

    for published in (sidecar, anchor):
        inode = published.stat().st_ino
        inode_events = [event for event in events if event[1] == inode]
        fchmod_index = next(
            index
            for index, event in enumerate(inode_events)
            if event == ("fchmod", inode, 0o444)
        )
        assert any(
            index > fchmod_index and event == ("fsync", inode, 0o444)
            for index, event in enumerate(inode_events)
        )


def test_task_index_rejects_bytes_after_trailer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _small_plan(tmp_path, Split.TRAIN, monkeypatch)
    sidecar = Path(project.task_manifest.task_index.path)
    sidecar.chmod(0o644)
    with sidecar.open("ab") as handle:
        handle.write(b"{}\n")
    sidecar.chmod(0o444)

    with pytest.raises(ProjectContractError, match="after its trailer"):
        project.task_manifest.task_index.validate()


@pytest.mark.parametrize(
    ("record_index", "field"),
    [(0, "schema_version"), (1, "sequence"), (-1, "task_count")],
)
def test_task_index_rejects_resigned_boolean_numeric_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_index: int,
    field: str,
) -> None:
    project = _small_plan(tmp_path, Split.TRAIN, monkeypatch)
    forged = _resign_task_index_with_mutation(
        project.task_manifest.task_index,
        record_index,
        field,
    )

    with pytest.raises(ProjectContractError, match="must be an integer"):
        forged.validate()


def test_incremental_json_array_fingerprint_matches_materialized_reference() -> None:
    payloads = [{"b": 2, "a": 1}, [3, 4], "done"]
    streamed = CanonicalJsonArrayFingerprint()
    for payload in payloads:
        streamed.add(payload)
    assert streamed.count == len(payloads)
    assert streamed.fingerprint == fingerprint_json(payloads)


def test_byte_attestation_memo_is_bounded_and_thread_safe() -> None:
    memo = project_module._ByteAttestationMemo(max_entries=4)

    def remember(index: int):
        identity = project_module._StableFileIdentity(
            resolved_path=f"/tmp/task-index-{index}",
            device=1,
            inode=index + 1,
            size=100 + index,
            mtime_ns=index,
            ctime_ns=index,
            mode=0o444,
            owner_uid=os.geteuid(),
        )
        key = project_module._ByteAttestationKey(
            file=identity,
            expected_sha256=f"{index:064x}",
            schema_contract="task-index-v2:test",
        )
        memo.remember(key)
        return key

    with ThreadPoolExecutor(max_workers=8) as executor:
        keys = list(executor.map(remember, range(64)))

    assert memo.size == memo.max_entries == 4
    assert not memo.contains(keys[0])
    assert memo.contains(keys[-1])


def test_identity_external_sort_caps_retained_runs_and_fan_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(project_module, "TASK_INDEX_IDENTITY_SORT_CHUNK_SIZE", 1)
    builder = project_module._IdentityFingerprintBuilder(tmp_path)
    keys = [f"train:{index}" for index in range(5_000, 0, -1)]
    maximum_retained_runs = 0
    for key in keys:
        builder.add(key)
        maximum_retained_runs = max(
            maximum_retained_runs,
            sum(len(paths) for paths in builder._levels.values()),
        )
    count, observed = builder.finish()

    assert count == len(keys)
    assert observed == fingerprint_json(sorted(keys))
    assert maximum_retained_runs <= (
        project_module.TASK_INDEX_SORT_FAN_IN
        * project_module.TASK_INDEX_MAX_SORT_LEVELS
    )
    assert list(tmp_path.iterdir()) == []


def test_reuse_rejects_mutated_live_storage_and_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = _small_plan(tmp_path, Split.TRAIN, monkeypatch)
    val = _small_plan(tmp_path, Split.VAL, monkeypatch)
    live_train = _live_attestation(train, project_id=1)
    drifted_storage = replace(
        live_train.storage_manifest,
        storage_identity="local-files:wrong-project",
    )
    drifted = replace(
        live_train,
        storage_manifest=drifted_storage,
        managed_link_is_symlink=False,
        managed_link_resolved_target=str(tmp_path / "outside"),
    )

    with pytest.raises(ManifestDriftError) as error:
        plan_instance_bootstrap(
            {Split.TRAIN: train, Split.VAL: val},
            _adapter_for(
                train,
                val,
                {
                    Split.TRAIN: drifted,
                    Split.VAL: _live_attestation(val, project_id=2),
                },
            ),
        )
    assert {
        "live.storage_manifest_fingerprint",
        "live.managed_link_is_symlink",
        "live.managed_link_resolved_target",
    }.issubset(error.value.mismatches)


@pytest.mark.parametrize(
    ("mutation", "mismatch"),
    [
        (lambda live: replace(live, vendor_revision="wrong"), "live.vendor_revision"),
        (
            lambda live: replace(
                live,
                label_config=live.label_config.replace(
                    'canRotate="false"', 'canRotate="true"'
                ),
            ),
            "live.label_config_fingerprint",
        ),
        (
            lambda live: replace(
                live,
                controls=replace(live.controls, show_native_skip=True),
            ),
            "live.project_capabilities",
        ),
    ],
)
def test_reuse_rejects_mutated_live_vendor_config_and_capabilities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation,
    mismatch: str,
) -> None:
    train = _small_plan(tmp_path, Split.TRAIN, monkeypatch)
    val = _small_plan(tmp_path, Split.VAL, monkeypatch)
    drifted = mutation(_live_attestation(train, project_id=1))

    with pytest.raises(ManifestDriftError) as error:
        plan_instance_bootstrap(
            {Split.TRAIN: train, Split.VAL: val},
            _adapter_for(
                train,
                val,
                {
                    Split.TRAIN: drifted,
                    Split.VAL: _live_attestation(val, project_id=2),
                },
            ),
        )
    assert mismatch in error.value.mismatches


def test_fingerprints_and_file_hashes_are_content_based(tmp_path: Path) -> None:
    assert fingerprint_json({"b": 2, "a": 1}) == fingerprint_json({"a": 1, "b": 2})
    payload = tmp_path / "payload"
    payload.write_bytes(b"coordexp")
    assert (
        sha256_file(payload)
        == "34dcf3a7f3833341119a22215d5634290f4419753b8cb47cc767d56344ae9a58"
    )
