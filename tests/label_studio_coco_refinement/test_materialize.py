from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterator

import pytest

import src.label_studio_coco_refinement.materialize as materialize_module
from src.common.errors import DataContractError
from src.data import load_raw_examples
from src.label_studio_coco_refinement.categories import COCO80_REGISTRY
from src.label_studio_coco_refinement.materialize import (
    CommittedGenerationReceipt,
    SourceTaskIdentityReceipt,
    WorkingCoordMaterializer,
)
from src.label_studio_coco_refinement.models import (
    ObjectIdentity,
    OrderedWorkingObject,
    RefinementRuntimeLayout,
    WorkingObject,
    stable_top_left_order,
)
from src.label_studio_coco_refinement.store import (
    BatchMember,
    BatchRequest,
    BootstrapSpec,
    CommitRequest,
    DraftSaveReceipt,
    InjectedCrash,
    RecoveryError,
    WorkingDatasetStore,
    semantic_hash,
    sha256_json,
)


class RecordingLock:
    def __init__(self) -> None:
        self.active = False
        self.acquisitions = 0

    @contextmanager
    def __call__(self) -> Iterator[None]:
        assert not self.active
        self.active = True
        self.acquisitions += 1
        try:
            yield
        finally:
            self.active = False


class FakeSourceIdentityResolver:
    def __init__(self, receipts: list[SourceTaskIdentityReceipt]) -> None:
        self._receipts = {
            (receipt.split, receipt.image_id): receipt for receipt in receipts
        }
        self.calls: list[tuple[str, int]] = []

    def resolve(self, *, split: str, image_id: int) -> SourceTaskIdentityReceipt:
        self.calls.append((split, image_id))
        try:
            return self._receipts[(split, image_id)]
        except KeyError as exc:
            raise LookupError(f"unknown source task: {split}:{image_id}") from exc


class AcceptingAnnotationVerifier:
    def verify(self, identity: Any) -> bool:
        return True


class EmptyInferenceReceiptResolver:
    def resolve(self, receipt_id: str) -> None:
        return None


class CrashAt:
    def __init__(self, boundary: str) -> None:
        self.boundary = boundary

    def __call__(self, boundary: str) -> None:
        if boundary == self.boundary:
            raise InjectedCrash(boundary)


def test_materializer_is_layout_split_bound_atomic_and_loader_compatible(
    tmp_path: Path,
) -> None:
    layout, source, output, generation, lock, identities = _project(
        tmp_path,
        split="train",
    )
    source_before = source.read_bytes()

    receipt = WorkingCoordMaterializer._for_test(
        layout,
        "train",
        exclusive_lock=lock,
        source_identity_resolver=identities,
    ).materialize(generation)

    payload = json.loads(output.read_text())
    assert payload["objects"][0]["bbox_2d"] == [
        "<|coord_1|>",
        "<|coord_46|>",
        "<|coord_691|>",
        "<|coord_941|>",
    ]
    assert payload["objects"][1]["coco_ann_id"] == -1
    assert payload["objects"][1]["metadata"] == {"receipt": "r1"}
    assert source.read_bytes() == source_before
    assert receipt.split == "train"
    assert receipt.generation == generation.generation
    assert receipt.source_path == layout.working_norm("train")
    assert receipt.destination_path == layout.working_coord("train")
    assert receipt.row_count == 1
    assert receipt.object_count == 2
    assert receipt.source_sha256 == hashlib.sha256(source_before).hexdigest()
    assert receipt.destination_sha256 == hashlib.sha256(output.read_bytes()).hexdigest()
    assert lock.acquisitions == 1
    assert lock.active is False
    assert identities.calls == [("train", 34)]

    (loaded,) = load_raw_examples(output)
    assert [obj.object_id for obj in loaded.objects] == ["589229", "-1"]
    assert [obj.bbox for obj in loaded.objects] == [
        (1, 46, 691, 941),
        (20, 50, 700, 950),
    ]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda row: row["objects"][0].update({"category_id": 1}),
        lambda row: row.update({"objects": []}),
        lambda row: row["objects"][0].update({"unexpected": True}),
        lambda row: row["objects"][0].update({"bbox_2d": [1, 1, 1, 2]}),
    ],
)
def test_invalid_working_row_preserves_source_and_prior_output(
    tmp_path: Path,
    mutate,
) -> None:
    row = _row(split="train")
    mutate(row)
    layout, source, output, generation, lock, identities = _project(
        tmp_path,
        split="train",
        rows=[row],
    )
    source_before = source.read_bytes()
    output_before = output.read_bytes()

    with pytest.raises(DataContractError):
        WorkingCoordMaterializer._for_test(
            layout,
            "train",
            exclusive_lock=lock,
            source_identity_resolver=identities,
        ).materialize(generation)

    assert source.read_bytes() == source_before
    assert output.read_bytes() == output_before
    assert not list(output.parent.glob(".working.coord.jsonl.*.tmp"))


@pytest.mark.parametrize("target", ["input", "output"])
def test_canonical_selected_source_cannot_be_input_or_output_target(
    tmp_path: Path,
    target: str,
) -> None:
    layout, source, output, generation, lock, identities = _project(
        tmp_path,
        split="train",
    )
    canonical_source = layout.selected_source("train")
    canonical_source.parent.mkdir(parents=True, exist_ok=True)
    canonical_source.write_bytes(b"immutable-canonical-source\n")
    if target == "input":
        source.unlink()
        source.symlink_to(canonical_source)
    else:
        output.unlink()
        output.symlink_to(canonical_source)
    source_before = source.read_bytes()
    output_before = output.read_bytes()
    canonical_before = canonical_source.read_bytes()

    with pytest.raises(DataContractError, match="canonical selected source|symlink"):
        WorkingCoordMaterializer._for_test(
            layout,
            "train",
            exclusive_lock=lock,
            source_identity_resolver=identities,
        ).materialize(generation)

    assert source.read_bytes() == source_before
    assert output.read_bytes() == output_before
    assert canonical_source.read_bytes() == canonical_before


def test_cross_split_working_locator_is_rejected_without_mutation(
    tmp_path: Path,
) -> None:
    row = _row(split="train")
    row["images"] = ["images/val2017/000000000034.jpg"]
    row["file_name"] = "images/val2017/000000000034.jpg"
    layout, source, output, generation, lock, identities = _project(
        tmp_path,
        split="train",
        rows=[row],
    )
    cross_split_image = layout.image_root / "val2017" / "000000000034.jpg"
    cross_split_image.parent.mkdir(parents=True, exist_ok=True)
    cross_split_image.write_bytes(b"cross-split-image")
    source_before = source.read_bytes()
    output_before = output.read_bytes()

    with pytest.raises(DataContractError, match="train2017"):
        WorkingCoordMaterializer._for_test(
            layout,
            "train",
            exclusive_lock=lock,
            source_identity_resolver=identities,
        ).materialize(generation)

    assert source.read_bytes() == source_before
    assert output.read_bytes() == output_before


@pytest.mark.parametrize(
    "mismatch",
    ["generation", "working_sha256", "task_count", "task_manifest_hash"],
)
def test_committed_manifest_mismatch_is_stale_and_atomic(
    tmp_path: Path,
    mismatch: str,
) -> None:
    layout, source, output, generation, lock, identities = _project(
        tmp_path,
        split="val",
    )
    manifest = json.loads(layout.project_manifest("val").read_text())
    if mismatch == "generation":
        manifest["generation"] += 1
    elif mismatch == "working_sha256":
        manifest["working_sha256"] = "0" * 64
    elif mismatch == "task_count":
        manifest["task_count"] += 1
    else:
        manifest["task_manifest_hash"] = "0" * 64
    _write_manifest(layout, "val", manifest)
    source_before = source.read_bytes()
    output_before = output.read_bytes()

    with pytest.raises(DataContractError, match="committed generation receipt"):
        WorkingCoordMaterializer._for_test(
            layout,
            "val",
            exclusive_lock=lock,
            source_identity_resolver=identities,
        ).materialize(generation)

    assert source.read_bytes() == source_before
    assert output.read_bytes() == output_before


def test_manifest_is_reverified_before_replace_under_the_injected_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, source, output, generation, lock, identities = _project(
        tmp_path,
        split="train",
    )
    source_before = source.read_bytes()
    output_before = output.read_bytes()
    original = materialize_module._read_committed_generation
    calls = 0

    def stale_on_second_read(*args, **kwargs):
        nonlocal calls
        calls += 1
        assert lock.active
        if calls == 2:
            manifest = json.loads(layout.project_manifest("train").read_text())
            manifest["generation"] += 1
            _write_manifest(layout, "train", manifest)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        materialize_module,
        "_read_committed_generation",
        stale_on_second_read,
    )

    with pytest.raises(DataContractError, match="committed generation receipt"):
        WorkingCoordMaterializer._for_test(
            layout,
            "train",
            exclusive_lock=lock,
            source_identity_resolver=identities,
        ).materialize(generation)

    assert calls == 2
    assert source.read_bytes() == source_before
    assert output.read_bytes() == output_before


def test_managed_image_link_must_resolve_to_exact_allowlisted_root(
    tmp_path: Path,
) -> None:
    layout, source, output, generation, lock, identities = _project(
        tmp_path,
        split="train",
    )
    other_root = layout.repository_root / "other-images"
    other_image = other_root / "train2017" / "000000000034.jpg"
    other_image.parent.mkdir(parents=True)
    other_image.write_bytes(b"wrong-root")
    layout.images_link("train").unlink()
    layout.images_link("train").symlink_to(other_root, target_is_directory=True)
    source_before = source.read_bytes()
    output_before = output.read_bytes()

    with pytest.raises(DataContractError, match="allowlisted shared image root"):
        WorkingCoordMaterializer._for_test(
            layout,
            "train",
            exclusive_lock=lock,
            source_identity_resolver=identities,
        ).materialize(generation)

    assert source.read_bytes() == source_before
    assert output.read_bytes() == output_before


@pytest.mark.parametrize(
    "mutation",
    [
        "same_split_image_substitution",
        "width",
        "height",
        "metadata_source",
        "metadata_split",
        "metadata_identity",
    ],
)
def test_row_identity_drift_is_rejected_against_source_task_receipt(
    tmp_path: Path,
    mutation: str,
) -> None:
    expected = _row(split="train")
    row = deepcopy(expected)
    if mutation == "same_split_image_substitution":
        row["images"] = ["images/train2017/000000000035.jpg"]
        row["file_name"] = "images/train2017/000000000035.jpg"
    elif mutation == "width":
        row["width"] += 1
    elif mutation == "height":
        row["height"] += 1
    elif mutation == "metadata_source":
        row["metadata"]["source"] = "not-coco2017"
    elif mutation == "metadata_split":
        row["metadata"]["split"] = "val"
    else:
        row["metadata"]["source_identity"] = "forged"

    layout, source, output, generation, lock, identities = _project(
        tmp_path,
        split="train",
        rows=[row],
        identity_rows=[expected],
    )
    substituted = layout.image_root / "train2017" / "000000000035.jpg"
    substituted.write_bytes(b"existing-substitution")
    source_before = source.read_bytes()
    output_before = output.read_bytes()

    with pytest.raises(DataContractError, match="source task identity|canonical"):
        WorkingCoordMaterializer._for_test(
            layout,
            "train",
            exclusive_lock=lock,
            source_identity_resolver=identities,
        ).materialize(generation)

    assert source.read_bytes() == source_before
    assert output.read_bytes() == output_before


def test_unknown_source_task_identity_is_rejected_without_replacement(
    tmp_path: Path,
) -> None:
    layout, source, output, generation, lock, identities = _project(
        tmp_path,
        split="train",
    )
    identities._receipts.clear()
    source_before = source.read_bytes()
    output_before = output.read_bytes()

    with pytest.raises(DataContractError, match="source task identity resolver"):
        WorkingCoordMaterializer._for_test(
            layout,
            "train",
            exclusive_lock=lock,
            source_identity_resolver=identities,
        ).materialize(generation)

    assert source.read_bytes() == source_before
    assert output.read_bytes() == output_before


@pytest.mark.parametrize("receipt_source_line", [1, 2])
def test_coherent_same_split_task_substitution_is_rejected_by_committed_inventory(
    tmp_path: Path,
    receipt_source_line: int,
) -> None:
    expected = _row(split="train", image_id=34)
    substitute = _row(split="train", image_id=35)
    layout, source, output, generation, lock, identities = _project(
        tmp_path,
        split="train",
        rows=[substitute],
        identity_rows=[expected],
    )
    substitute_image = layout.image_root / "train2017" / "000000000035.jpg"
    substitute_image.write_bytes(b"coherent-substitute-image")
    substitute_receipt = SourceTaskIdentityReceipt.capture(
        split="train",
        image_id=35,
        file_name=substitute["file_name"],
        width=substitute["width"],
        height=substitute["height"],
        metadata=substitute["metadata"],
        source_line=receipt_source_line,
        task_row_fingerprint=_json_fingerprint(substitute),
        task_manifest_hash=generation.task_manifest_hash,
        image_sha256=_file_sha256(substitute_image),
    )
    identities._receipts[("train", 35)] = substitute_receipt
    source_before = source.read_bytes()
    output_before = output.read_bytes()

    with pytest.raises(DataContractError, match="source line|task manifest inventory"):
        WorkingCoordMaterializer._for_test(
            layout,
            "train",
            exclusive_lock=lock,
            source_identity_resolver=identities,
        ).materialize(generation)

    assert identities.calls == [("train", 35)]
    assert source.read_bytes() == source_before
    assert output.read_bytes() == output_before


def test_in_place_image_byte_drift_is_rejected_without_replacement(
    tmp_path: Path,
) -> None:
    layout, source, output, generation, lock, identities = _project(
        tmp_path,
        split="train",
    )
    image = layout.image_root / "train2017" / "000000000034.jpg"
    image.write_bytes(b"mutated-after-source-task-attestation")
    source_before = source.read_bytes()
    output_before = output.read_bytes()

    with pytest.raises(DataContractError, match="image content"):
        WorkingCoordMaterializer._for_test(
            layout,
            "train",
            exclusive_lock=lock,
            source_identity_resolver=identities,
        ).materialize(generation)

    assert source.read_bytes() == source_before
    assert output.read_bytes() == output_before


def test_image_bytes_are_reverified_immediately_before_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, source, output, generation, lock, identities = _project(
        tmp_path,
        split="train",
    )
    image = layout.image_root / "train2017" / "000000000034.jpg"
    source_before = source.read_bytes()
    output_before = output.read_bytes()
    original = materialize_module._read_committed_generation
    calls = 0

    def mutate_image_on_second_manifest_read(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            image.write_bytes(b"mutated-between-candidate-and-replace")
        return original(*args, **kwargs)

    monkeypatch.setattr(
        materialize_module,
        "_read_committed_generation",
        mutate_image_on_second_manifest_read,
    )

    with pytest.raises(DataContractError, match="image content changed"):
        WorkingCoordMaterializer._for_test(
            layout,
            "train",
            exclusive_lock=lock,
            source_identity_resolver=identities,
        ).materialize(generation)

    assert calls == 2
    assert source.read_bytes() == source_before
    assert output.read_bytes() == output_before


def test_public_constructor_rejects_raw_callable_lock(tmp_path: Path) -> None:
    layout, _, _, _, lock, identities = _project(tmp_path, split="train")

    with pytest.raises((TypeError, DataContractError)):
        WorkingCoordMaterializer(
            layout,
            "train",
            exclusive_lock=lock,  # type: ignore[call-arg]
            source_identity_resolver=identities,
        )


def test_public_constructor_rejects_recording_and_private_store_locks(
    tmp_path: Path,
) -> None:
    layout, store, _, _, identities = _store_project(tmp_path)

    for invalid_store in (RecordingLock(), store._exclusive_lock):
        with pytest.raises(DataContractError, match="WorkingDatasetStore"):
            WorkingCoordMaterializer(
                layout,
                "train",
                store=invalid_store,  # type: ignore[arg-type]
                source_identity_resolver=identities,
            )


def test_public_constructor_rejects_replaced_generation_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, store, _, _, identities = _store_project(tmp_path)
    monkeypatch.setattr(store, "committed_generation_guard", RecordingLock())

    with pytest.raises(
        DataContractError, match="exact bound committed generation guard"
    ):
        WorkingCoordMaterializer(
            layout,
            "train",
            store=store,
            source_identity_resolver=identities,
        )


@pytest.mark.parametrize("transaction", ["legacy", "batch"])
def test_public_store_guard_blocks_manifest_published_unfinished_transaction(
    tmp_path: Path,
    transaction: str,
) -> None:
    layout, worker, peer, output, identities = _store_project(tmp_path)
    request = _store_commit_request(worker)
    worker._fault_injector = CrashAt("manifest_replaced")

    with pytest.raises(InjectedCrash):
        if transaction == "legacy":
            worker.commit(request)
        else:
            worker.enqueue_batch(
                BatchRequest(
                    batch_id="materialize-interrupted",
                    split="train",
                    base_generation=request.observed_generation,
                    members=(BatchMember(source_row_index=0, request=request),),
                )
            )
            worker.process_next_batch()

    generation = _store_generation_receipt(peer)
    assert generation.generation == 1
    output_before = output.read_bytes()
    materializer = WorkingCoordMaterializer(
        layout,
        "train",
        store=peer,
        source_identity_resolver=identities,
    )

    with pytest.raises(RecoveryError, match="requires recovery"):
        materializer.materialize(generation)

    assert output.read_bytes() == output_before
    assert identities.calls == []

    peer.recover()
    receipt = materializer.materialize(generation)

    assert receipt.generation == 1
    assert receipt.destination_path == output
    assert output.read_bytes() != output_before
    assert identities.calls == [("train", 34)]


def test_stable_top_left_order_preserves_prior_rank_and_creation_ties() -> None:
    known_late = _ordered("known-late", 2, prior_rank=4, bbox=(10, 10, 20, 20))
    new_first = _ordered("new-first", -1, creation_ordinal=1, bbox=(10, 10, 30, 30))
    known_early = _ordered("known-early", 1, prior_rank=2, bbox=(10, 10, 40, 40))
    geometrically_first = _ordered("top", 3, prior_rank=8, bbox=(900, 1, 999, 10))

    ordered = stable_top_left_order(
        [new_first, known_late, geometrically_first, known_early]
    )

    assert [item.identity.region_key for item in ordered] == [
        "top",
        "known-early",
        "known-late",
        "new-first",
    ]


def test_runtime_layout_is_exact_dedicated_and_split_scoped(tmp_path: Path) -> None:
    repository = (tmp_path / "repo").resolve()
    layout = RefinementRuntimeLayout.under_repository(repository)

    assert layout.repository_root == repository
    assert layout.root == repository / "outputs" / "label_studio_coco_refinement" / (
        "rescale_32_1024_bbox_len12000"
    )
    assert (
        layout.image_root == repository / "public_data/coco/rescale_32_1024_bbox/images"
    )
    assert layout.label_studio_state == layout.root / "label-studio" / "state"
    assert layout.working_norm("train") == layout.root / "train" / "working.norm.jsonl"
    assert layout.working_coord("val") == layout.root / "val" / "working.coord.jsonl"
    assert layout.commit_lock("train") == layout.root / "train" / ".commit.lock"
    assert layout.images_link("train") == layout.root / "train" / "images"
    with pytest.raises(DataContractError):
        layout.working_norm("test")  # type: ignore[arg-type]


def _store_project(
    tmp_path: Path,
) -> tuple[
    RefinementRuntimeLayout,
    WorkingDatasetStore,
    WorkingDatasetStore,
    Path,
    FakeSourceIdentityResolver,
]:
    repository = (tmp_path / "repo").resolve()
    layout = RefinementRuntimeLayout.under_repository(repository)
    image = layout.image_root / "train2017" / "000000000034.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"store-fixture-image")
    source_dir = repository / "public_data" / "coco" / "rescale_32_1024_bbox_len12000"
    source_dir.mkdir(parents=True)
    source = source_dir / "train.norm.jsonl"
    source_row = _row(split="train")
    source_row["images"] = ["../rescale_32_1024_bbox/images/train2017/000000000034.jpg"]
    source_row["objects"][1]["coco_ann_id"] = 589230
    source.write_text(_canonical_json(source_row) + "\n")
    verifier = AcceptingAnnotationVerifier()
    inference_resolver = EmptyInferenceReceiptResolver()
    result = WorkingDatasetStore.bootstrap(
        BootstrapSpec(
            split="train",
            source_path=source,
            runtime_root=layout.root,
            image_root=layout.image_root,
            expected_source_sha256=_file_sha256(source),
            project_id="project-train",
            storage_id="storage-train",
            adapter_version="adapter-v1",
            vendor_revision="label-studio-rev",
            registry_fingerprint=COCO80_REGISTRY.fingerprint,
            label_config_fingerprint="label-config-v1",
        ),
        annotation_verifier=verifier,
        inference_receipt_resolver=inference_resolver,
    )
    worker = result.store
    peer = WorkingDatasetStore(
        worker.split_dir,
        annotation_verifier=verifier,
        inference_receipt_resolver=inference_resolver,
    )
    output = layout.working_coord("train")
    output.write_bytes(b'{"prior":"valid"}\n')
    manifest = json.loads(worker.manifest_path.read_text())
    working_row = json.loads(worker.working_path.read_text())
    identities = FakeSourceIdentityResolver(
        [
            SourceTaskIdentityReceipt.capture(
                split="train",
                image_id=34,
                file_name=working_row["file_name"],
                width=working_row["width"],
                height=working_row["height"],
                metadata=working_row["metadata"],
                source_line=1,
                task_row_fingerprint=_json_fingerprint(working_row),
                task_manifest_hash=manifest["task_manifest_hash"],
                image_sha256=_file_sha256(image),
            )
        ]
    )
    return layout, worker, peer, output, identities


def _store_commit_request(store: WorkingDatasetStore) -> CommitRequest:
    restored = store.restore_draft(34)
    identity_by_object_id = {
        object_id: region_key
        for region_key, object_id in restored.region_id_mapping.items()
    }
    regions = []
    for obj in restored.row["objects"]:
        region = deepcopy(obj)
        region["region_key"] = identity_by_object_id[obj["coco_ann_id"]]
        regions.append(region)
    regions[0]["bbox_2d"] = [2, 46, 691, 941]
    projection_hash = semantic_hash(regions)
    draft_save = DraftSaveReceipt(
        project_id="project-train",
        task_id="train:34",
        annotation_id="annotation-34",
        draft_id="draft-34",
        annotation_revision="annotation-v7",
        draft_updated_at="2026-07-15T00:00:07Z",
        semantic_hash=projection_hash,
        result_hash=sha256_json(regions),
    )
    return CommitRequest(
        commit_id="materialize-interrupted:member:34",
        split="train",
        image_id=34,
        project_id="project-train",
        task_id="train:34",
        annotation_id="annotation-34",
        draft_id="draft-34",
        annotation_revision="annotation-v7",
        draft_updated_at="2026-07-15T00:00:07Z",
        semantic_hash=projection_hash,
        result_hash=sha256_json(regions),
        base_row_hash=restored.row_hash,
        observed_generation=restored.generation,
        regions=regions,
        draft_save=draft_save,
    )


def _store_generation_receipt(
    store: WorkingDatasetStore,
) -> CommittedGenerationReceipt:
    manifest = json.loads(store.manifest_path.read_text())
    return CommittedGenerationReceipt(
        split=manifest["split"],
        generation=manifest["generation"],
        working_sha256=manifest["working_sha256"],
        task_count=manifest["task_count"],
        task_manifest_hash=manifest["task_manifest_hash"],
    )


def _project(
    tmp_path: Path,
    *,
    split: str,
    rows: list[dict] | None = None,
    identity_rows: list[dict] | None = None,
) -> tuple[
    RefinementRuntimeLayout,
    Path,
    Path,
    CommittedGenerationReceipt,
    RecordingLock,
    FakeSourceIdentityResolver,
]:
    layout = RefinementRuntimeLayout.under_repository(tmp_path / "repo")
    split_root = layout.split_root(split)  # type: ignore[arg-type]
    split_root.mkdir(parents=True)
    image_subdir = layout.image_root / f"{split}2017"
    image_subdir.mkdir(parents=True)
    image = image_subdir / "000000000034.jpg"
    image.write_bytes(b"fixture-image")
    layout.images_link(split).symlink_to(layout.image_root, target_is_directory=True)  # type: ignore[arg-type]
    source = layout.working_norm(split)  # type: ignore[arg-type]
    output = layout.working_coord(split)  # type: ignore[arg-type]
    _write_jsonl(source, rows if rows is not None else [_row(split=split)])
    output.write_bytes(b'{"prior":"valid"}\n')
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    authoritative_rows = (
        identity_rows if identity_rows is not None else [_row(split=split)]
    )
    task_manifest_hash = _task_manifest_hash(split, authoritative_rows)
    generation = CommittedGenerationReceipt(
        split=split,  # type: ignore[arg-type]
        generation=7,
        working_sha256=source_hash,
        task_count=len(authoritative_rows),
        task_manifest_hash=task_manifest_hash,
    )
    _write_manifest(
        layout,
        split,
        {
            "split": split,
            "generation": generation.generation,
            "working_sha256": generation.working_sha256,
            "task_count": generation.task_count,
            "task_manifest_hash": generation.task_manifest_hash,
            "image_root": str(layout.image_root),
            "document_root": str(layout.image_root),
            "managed_image_link": str(layout.images_link(split)),  # type: ignore[arg-type]
        },
    )
    identities = FakeSourceIdentityResolver(
        [
            SourceTaskIdentityReceipt.capture(
                split=split,
                image_id=row["image_id"],
                file_name=row["file_name"],
                width=row["width"],
                height=row["height"],
                metadata=row["metadata"],
                source_line=source_line,
                task_row_fingerprint=_json_fingerprint(row),
                task_manifest_hash=task_manifest_hash,
                image_sha256=_file_sha256(
                    layout.image_root / f"{split}2017" / f"{row['image_id']:012d}.jpg"
                ),
            )
            for source_line, row in enumerate(authoritative_rows, start=1)
        ]
    )
    return layout, source, output, generation, RecordingLock(), identities


def _write_manifest(
    layout: RefinementRuntimeLayout,
    split: str,
    manifest: dict,
) -> None:
    layout.project_manifest(split).write_text(  # type: ignore[arg-type]
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    )


def _ordered(
    region_key: str,
    coco_ann_id: int,
    *,
    bbox: tuple[int, int, int, int],
    prior_rank: int | None = None,
    creation_ordinal: int | None = None,
) -> OrderedWorkingObject:
    return OrderedWorkingObject(
        identity=ObjectIdentity(
            region_key=region_key,
            coco_ann_id=coco_ann_id,
            prior_rank=prior_rank,
            creation_ordinal=creation_ordinal,
        ),
        object=WorkingObject(
            bbox_2d=bbox,
            desc="person",
            category_id=1,
            category_name="person",
            coco_ann_id=coco_ann_id,
        ),
    )


def _row(*, split: str, image_id: int = 34) -> dict:
    file_name = f"images/{split}2017/{image_id:012d}.jpg"
    return {
        "images": [file_name],
        "objects": [
            _object((1, 46, 691, 941), "zebra", 24, 589229),
            _object((20, 50, 700, 950), "person", 1, -1, metadata={"receipt": "r1"}),
        ],
        "width": 1248,
        "height": 832,
        "image_id": image_id,
        "file_name": file_name,
        "metadata": {"source": "coco2017", "split": split},
    }


def _object(
    bbox: tuple[int, int, int, int],
    name: str,
    category_id: int,
    coco_ann_id: int,
    *,
    metadata: dict | None = None,
) -> dict:
    payload = {
        "bbox_2d": list(bbox),
        "desc": name,
        "category_id": category_id,
        "category_name": name,
        "coco_ann_id": coco_ann_id,
    }
    if metadata is not None:
        payload["metadata"] = metadata
    return payload


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _json_fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _task_manifest_hash(split: str, rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for source_line, row in enumerate(rows, start=1):
        seed_identity = {
            "task_id": f"{split}:{row['image_id']}",
            "image_id": row["image_id"],
            "row_hash": _json_fingerprint(row),
            "source_line": source_line,
        }
        digest.update((_canonical_json(seed_identity) + "\n").encode("utf-8"))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
