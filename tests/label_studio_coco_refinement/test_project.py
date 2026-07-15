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
    build_split_project_plan,
    compare_manifests,
    fingerprint_json,
    inspect_source,
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
    def __init__(self, states: dict[Split, LiveProjectAttestation | None]) -> None:
        self.states = states
        self.calls: list[Split] = []

    def attest_project(self, split: Split) -> LiveProjectAttestation | None:
        self.calls.append(split)
        return self.states.get(split)


class _SavedManifestOnlyAdapter:
    def __init__(self, manifests: dict[Split, dict[str, Any]]) -> None:
        self.manifests = manifests

    def attest_project(self, split: Split) -> dict[str, Any] | None:
        return self.manifests.get(split)


def _install_small_source(
    tmp_path: Path,
    split: Split,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    row = _row(split, image_id=9 if split is Split.TRAIN else 139)
    contract = project_module.SOURCE_CONTRACTS[split]
    source_path = tmp_path / Path(contract.relative_path)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n").encode()
    source_path.write_bytes(encoded)
    contracts = dict(project_module.SOURCE_CONTRACTS)
    contracts[split] = replace(
        contract,
        sha256=hashlib.sha256(encoded).hexdigest(),
        row_count=1,
        box_count=1,
    )
    monkeypatch.setattr(
        project_module,
        "SOURCE_CONTRACTS",
        MappingProxyType(contracts),
    )
    return source_path


def _small_plan(tmp_path: Path, split: Split, monkeypatch: pytest.MonkeyPatch):
    source_path = _install_small_source(tmp_path, split, monkeypatch)
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
            task_id=10_000 + index,
            annotation_count=1,
            authoritative_annotation_id=20_000 + index,
            authoritative_annotation_revision=1,
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
    assert len(SOURCE_CONTRACTS[Split.TRAIN].sha256) == 64
    assert len(SOURCE_CONTRACTS[Split.VAL].sha256) == 64

    layout = RuntimeLayout.for_repo(REPO_ROOT)
    assert layout.root == REPO_ROOT / Path(RUNTIME_ROOT)
    assert layout.image_root == REPO_ROOT / Path(SHARED_IMAGE_ROOT)
    assert layout.label_studio_state == layout.root / "label-studio" / "state"
    assert layout.for_split(Split.TRAIN).working_norm_jsonl == (
        layout.root / "train" / "working.norm.jsonl"
    )
    assert layout.for_split(Split.VAL).project_manifest == layout.root / "val" / "project.json"
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
def test_source_row_rejects_schema_identity_geometry_category_and_locator_drift(mutation) -> None:
    row = _row(Split.TRAIN)
    mutation(row)
    with pytest.raises(ProjectContractError):
        validate_source_row(row, split=Split.TRAIN, registry=COCO80_REGISTRY)


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
    assert "predictions" not in payload
    assert len(payload["annotations"]) == 1
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

    create = plan_instance_bootstrap(
        {Split.TRAIN: train, Split.VAL: val},
        _AttestingAdapter({Split.TRAIN: None, Split.VAL: None}),
    )
    assert [item.action for item in create.projects] == [
        BootstrapAction.CREATE,
        BootstrapAction.CREATE,
    ]
    live_train = _live_attestation(train, project_id=1)
    live_val = _live_attestation(val, project_id=2)
    adapter = _AttestingAdapter({Split.TRAIN: live_train, Split.VAL: live_val})
    reuse = plan_instance_bootstrap(
        {Split.TRAIN: train, Split.VAL: val},
        adapter,
    )
    assert [item.action for item in reuse.projects] == [
        BootstrapAction.REUSE,
        BootstrapAction.REUSE,
    ]
    assert adapter.calls == [Split.TRAIN, Split.VAL]
    assert [item.live_attestation_fingerprint for item in reuse.projects] == [
        live_train.fingerprint,
        live_val.fingerprint,
    ]


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
        }
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
            _AttestingAdapter(
                {
                    Split.TRAIN: drifted,
                    Split.VAL: _live_attestation(val, project_id=2),
                }
            ),
        )
    assert "live.task_identity_fingerprint" in error.value.mismatches


@pytest.mark.parametrize(
    ("field", "value", "expected_mismatch"),
    [
        ("annotation_count", 2, ".annotation_count"),
        ("alternate_annotation_count", 1, ".alternate_annotation_count"),
        ("prediction_count", 1, ".prediction_count"),
        ("authoritative_annotation_id", "", ".authoritative_annotation_id"),
        ("authoritative_annotation_revision", "", ".authoritative_annotation_revision"),
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
            _AttestingAdapter(
                {
                    Split.TRAIN: replace(live_train, tasks=(task,)),
                    Split.VAL: _live_attestation(val, project_id=2),
                }
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
            _AttestingAdapter(
                {
                    Split.TRAIN: drifted,
                    Split.VAL: _live_attestation(val, project_id=2),
                }
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
                label_config=live.label_config.replace('canRotate="false"', 'canRotate="true"'),
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
            _AttestingAdapter(
                {
                    Split.TRAIN: drifted,
                    Split.VAL: _live_attestation(val, project_id=2),
                }
            ),
        )
    assert mismatch in error.value.mismatches


def test_fingerprints_and_file_hashes_are_content_based(tmp_path: Path) -> None:
    assert fingerprint_json({"b": 2, "a": 1}) == fingerprint_json({"a": 1, "b": 2})
    payload = tmp_path / "payload"
    payload.write_bytes(b"coordexp")
    assert sha256_file(payload) == "34dcf3a7f3833341119a22215d5634290f4419753b8cb47cc767d56344ae9a58"
