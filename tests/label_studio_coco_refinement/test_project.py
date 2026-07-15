from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
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
    FrozenTaskImport,
    LiveProjectAttestation,
    LiveTaskAttestation,
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

    def attest_project(self, split: Split) -> LiveProjectAttestation | None:
        self.calls.append(split)
        return self.states.get(split)

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

    def attest_project(self, split: Split) -> dict[str, Any] | None:
        return self.manifests.get(split)

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


def _live_attestation(plan, *, project_id: int) -> LiveProjectAttestation:
    tasks = tuple(
        LiveTaskAttestation(
            identity=entry.identity,
            source_line=entry.source_line,
            image_locator=entry.label_studio_image_locator,
            task_data_fingerprint=entry.task_data_fingerprint,
            task_id=10_000 + index,
            annotation_count=1,
            authoritative_annotation_id=20_000 + index,
            authoritative_annotation_revision=1,
            authoritative_annotation_fingerprint=(
                entry.authoritative_annotation_fingerprint
            ),
            authoritative_annotation_ground_truth=False,
            alternate_annotation_count=0,
            prediction_count=0,
        )
        for index, entry in enumerate(plan.task_manifest.entries)
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
        tasks=tasks,
    )


def _adapter_for(
    train,
    val,
    states: dict[Split, LiveProjectAttestation | None],
) -> _AttestingAdapter:
    manifest = build_instance_bootstrap_manifest({Split.TRAIN: train, Split.VAL: val})
    return _AttestingAdapter(states, bootstrap_manifest=manifest.to_dict())


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
    stable_task_bytes = train.task_imports[0].canonical_json
    stable_manifest = train.manifest.to_dict()

    with pytest.raises(TypeError):
        train.task_imports[0].canonical_json[0] = 0  # type: ignore[index]

    first_send = action.task_imports_for_adapter_send()
    first_send[0]["annotations"][0]["ground_truth"] = True
    first_send[0]["annotations"][0]["result"][0]["meta"]["last_committed_bbox"][0] = 333
    second_send = action.task_imports_for_adapter_send()

    assert second_send[0]["annotations"][0]["ground_truth"] is False
    assert second_send[0]["annotations"][0]["result"][0]["meta"][
        "last_committed_bbox"
    ] == [0, 10, 500, 999]
    assert first_send is not second_send
    assert first_send[0] is not second_send[0]
    assert train.task_imports[0].canonical_json == stable_task_bytes
    assert train.manifest.to_dict() == stable_manifest


@pytest.mark.parametrize(
    ("fingerprint_field", "message"),
    [
        ("task_data_fingerprint", "task data fingerprint"),
        (
            "authoritative_annotation_fingerprint",
            "authoritative annotation fingerprint",
        ),
    ],
)
def test_altered_task_fingerprint_fails_before_live_adapter_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fingerprint_field: str,
    message: str,
) -> None:
    train = _small_plan(tmp_path, Split.TRAIN, monkeypatch)
    val = _small_plan(tmp_path, Split.VAL, monkeypatch)
    entry = replace(
        train.task_manifest.entries[0],
        **{fingerprint_field: "0" * 64},
    )
    task_manifest = replace(train.task_manifest, entries=(entry,))
    tampered_train = replace(
        train,
        task_manifest=task_manifest,
        manifest=replace(
            train.manifest,
            task_manifest_fingerprint=task_manifest.fingerprint,
        ),
    )
    adapter = _AttestingAdapter({Split.TRAIN: None, Split.VAL: None})

    with pytest.raises(ProjectContractError, match=message):
        plan_instance_bootstrap(
            {Split.TRAIN: tampered_train, Split.VAL: val},
            adapter,
        )
    assert adapter.calls == []


def test_project_task_storage_fingerprints_and_bootstrap_are_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = _small_plan(tmp_path, Split.TRAIN, monkeypatch)
    val = _small_plan(tmp_path, Split.VAL, monkeypatch)
    repeated = _small_plan(tmp_path, Split.TRAIN, monkeypatch)

    assert train.task_manifest.fingerprint == repeated.task_manifest.fingerprint
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
    assert [len(item.missing_task_imports) for item in create.projects] == [1, 1]
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
    assert [item.missing_task_imports for item in reuse.projects] == [(), ()]
    assert adapter.calls == [Split.TRAIN, Split.VAL]
    assert [item.live_attestation_fingerprint for item in reuse.projects] == [
        live_train.fingerprint,
        live_val.fingerprint,
    ]


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
    partial_train = replace(full_train, tasks=(full_train.tasks[0],))
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
    assert train_retry.reused_task_identities == (TaskIdentity(Split.TRAIN, 9),)
    assert [
        task["data"]["coordexp_task_key"]
        for task in train_retry.task_imports_for_adapter_send()
    ] == ["train:25"]

    reconciled = plan_instance_bootstrap(
        {Split.TRAIN: train, Split.VAL: val},
        _adapter_for(
            train,
            val,
            {Split.TRAIN: full_train, Split.VAL: live_val},
        ),
    )
    assert reconciled.projects[0].action is BootstrapAction.REUSE
    assert reconciled.projects[0].missing_task_imports == ()


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


def test_reuse_rejects_mutated_live_task_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = _small_plan(tmp_path, Split.TRAIN, monkeypatch)
    val = _small_plan(tmp_path, Split.VAL, monkeypatch)
    live_train = _live_attestation(train, project_id=1)
    wrong_task = replace(
        live_train.tasks[0],
        identity=TaskIdentity(Split.TRAIN, 999),
    )
    drifted = replace(live_train, tasks=(wrong_task,))

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
    assert "live.tasks[train:999].identity (unexpected)" in error.value.mismatches


def test_reconcile_rejects_duplicate_live_task_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = _small_plan(tmp_path, Split.TRAIN, monkeypatch)
    val = _small_plan(tmp_path, Split.VAL, monkeypatch)
    live_train = _live_attestation(train, project_id=1)
    duplicate = replace(live_train, tasks=(live_train.tasks[0], live_train.tasks[0]))

    with pytest.raises(ManifestDriftError) as error:
        plan_instance_bootstrap(
            {Split.TRAIN: train, Split.VAL: val},
            _adapter_for(
                train,
                val,
                {
                    Split.TRAIN: duplicate,
                    Split.VAL: _live_attestation(val, project_id=2),
                },
            ),
        )
    assert any("identity (duplicate)" in item for item in error.value.mismatches)


@pytest.mark.parametrize(
    ("field", "value", "expected_mismatch"),
    [
        ("annotation_count", 2, ".annotation_count"),
        ("alternate_annotation_count", 1, ".alternate_annotation_count"),
        ("prediction_count", 1, ".prediction_count"),
        ("authoritative_annotation_id", "", ".authoritative_annotation_id"),
        ("authoritative_annotation_revision", "", ".authoritative_annotation_revision"),
        (
            "authoritative_annotation_ground_truth",
            True,
            ".authoritative_annotation_ground_truth",
        ),
        ("source_line", 2, ".source_line"),
        ("image_locator", "/data/local-files/?d=train2017/wrong.jpg", ".image_locator"),
        ("task_data_fingerprint", "0" * 64, ".task_data_fingerprint"),
        (
            "authoritative_annotation_fingerprint",
            "0" * 64,
            ".authoritative_annotation_fingerprint",
        ),
    ],
)
def test_reuse_rejects_mutated_live_annotation_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
    expected_mismatch: str,
) -> None:
    train = _small_plan(tmp_path, Split.TRAIN, monkeypatch)
    val = _small_plan(tmp_path, Split.VAL, monkeypatch)
    live_train = _live_attestation(train, project_id=1)
    task = replace(live_train.tasks[0], **{field: value})

    with pytest.raises(ManifestDriftError) as error:
        plan_instance_bootstrap(
            {Split.TRAIN: train, Split.VAL: val},
            _adapter_for(
                train,
                val,
                {
                    Split.TRAIN: replace(live_train, tasks=(task,)),
                    Split.VAL: _live_attestation(val, project_id=2),
                },
            ),
        )
    assert any(expected_mismatch in mismatch for mismatch in error.value.mismatches)


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
