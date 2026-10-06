"""CPU caller-to-consumer checks using disposable datasets and real persistence."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from fastapi.testclient import TestClient
from PIL import Image

from scripts import materialize_label_studio_coco_refinement as materializer_cli
from scripts import run_coco_refinement as standalone_cli
from src.coco_refinement.bootstrap import BootstrapSourceContract
from src.coco_refinement.http_security import CSRF_HEADER
from src.coco_refinement.preflight import SUPPORTED_DEPENDENCIES
from src.coco_refinement.runtime import create_standalone_runtime
from src.data import iter_raw_examples
from src.label_studio_coco_refinement import project
from src.label_studio_coco_refinement.draft_adapter import (
    canonicalize_label_studio_draft,
)
from src.label_studio_coco_refinement.geometry import norm1000_bbox_to_label_studio_xywh
from src.label_studio_coco_refinement.label_config import (
    build_label_config,
    label_config_fingerprint,
)
from src.label_studio_coco_refinement.models import RefinementRuntimeLayout
from src.label_studio_coco_refinement.runtime import (
    AuthenticatedPrincipal,
    AuthoritativeDraftSnapshot,
    DraftCatalogCapture,
    RefinementRuntime,
)
from src.working_dataset.categories import COCO80_REGISTRY
from src.working_dataset.codec import canonical_json, sha256_file
from src.working_dataset.records import BatchStatus, BootstrapSpec, StoreBusyError
from src.working_dataset.store import WorkingDatasetStore


def _source(root: Path, split: str) -> BootstrapSourceContract:
    source = root / f"public_data/coco/rescale_32_1024_bbox_len12000/{split}.norm.jsonl"
    images = root / "public_data/coco/rescale_32_1024_bbox/images"
    image = images / f"{split}2017/000000000001.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (32, 24), "red").save(image)
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        canonical_json(
            {
                "images": [
                    f"../rescale_32_1024_bbox/images/{split}2017/000000000001.jpg"
                ],
                "objects": [
                    {
                        "bbox_2d": [10, 20, 300, 400],
                        "desc": "person",
                        "category_name": "person",
                        "category_id": 1,
                        "coco_ann_id": 100,
                    }
                ],
                "width": 32,
                "height": 24,
                "image_id": 1,
                "file_name": f"images/{split}2017/000000000001.jpg",
                "metadata": {"source": "coco2017", "split": split},
            }
        )
        + "\n"
    )
    return BootstrapSourceContract(
        split=split,
        source_path=source,
        image_root=images,
        expected_source_sha256=sha256_file(source),
        expected_row_count=1,
    )


def _wait(predicate):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            if predicate():
                return
        except StoreBusyError:
            pass
        time.sleep(0.01)
    raise AssertionError("publication did not reach its terminal boundary")


def test_standalone_main_serves_real_sqlite_draft_and_dataset_commit(
    tmp_path, monkeypatch
):
    contracts = tuple(_source(tmp_path, split) for split in ("train", "val"))
    before = [value.source_path.read_bytes() for value in contracts]
    runtimes = []

    def runtime_factory(repo_root, **kwargs):
        # The selected annotation package has no tokenizer/model qualification.
        # Dataset publisher transaction behavior has its own temporary-file tests.
        kwargs.pop("terminal_publisher_factory")
        runtime = create_standalone_runtime(
            repo_root,
            **kwargs,
            source_contracts=contracts,
            environment={},
            version_resolver=SUPPORTED_DEPENDENCIES.__getitem__,
            poll_interval=0.005,
        )
        runtimes.append(runtime)
        return runtime

    class InProcessServer:
        def __init__(self, config):
            self.app = config.app

        def run(self):
            runtime = runtimes[-1]
            store = runtime.stores["train"]
            assert type(store) is WorkingDatasetStore
            task = runtime.workspace.splits["train"].tasks[0]
            origin = "http://127.0.0.1:19172"
            with TestClient(self.app, base_url=origin) as client:
                csrf = client.get("/api/session").json()["csrf_token"]
                headers = {"Origin": origin, CSRF_HEADER: csrf}
                saved = client.put(
                    "/api/splits/train/tasks/train:1/draft",
                    headers=headers,
                    json={
                        "mutation_id": "native-entry-save",
                        "expected_revision": 0,
                        "expected_generation": 0,
                        "expected_base_row_hash": task.base_row_hash,
                        "objects": [
                            {
                                "region_key": "train:coco:100",
                                "coco_ann_id": 100,
                                "bbox_2d": [11, 20, 300, 400],
                                "category_name": "person",
                                "category_id": 1,
                            }
                        ],
                    },
                )
                assert saved.status_code == 200, saved.text
                assert store.state_snapshot().generation == 0
                endpoint = "/api/splits/train/commits"
                queued = client.post(
                    endpoint, headers=headers, json={"batch_id": "native-entry"}
                )
                assert queued.status_code == 202, queued.text
                _wait(
                    lambda: client.get(endpoint + "/native-entry").json().get("status")
                    == "succeeded"
                )
                retry = client.post(
                    endpoint, headers=headers, json={"batch_id": "native-entry"}
                )
                assert retry.status_code == 200, retry.text
                assert retry.json()["status"] == "succeeded"
                state = runtime.workspace.repository.get_task_state(
                    task.project_id, task.task_id
                )
                assert state.current_generation == 1 and state.draft is None
                assert store.restore_draft(1).row["objects"][0]["bbox_2d"] == [
                    11,
                    20,
                    300,
                    400,
                ]
                assert runtime.stores["val"].state_snapshot().generation == 0
                assert (
                    client.get("/api/splits/train/tasks/train:1/draft").status_code
                    == 200
                )

    run_server = standalone_cli.run_server

    def run_with_temporary_inputs(**kwargs):
        return run_server(
            **kwargs, runtime_factory=runtime_factory, server_factory=InProcessServer
        )

    monkeypatch.setattr(standalone_cli, "run_server", run_with_temporary_inputs)
    assert (
        standalone_cli.main(["--repo-root", str(tmp_path), "--startup-timeout", "5"])
        == 0
    )
    assert runtimes[0].lock_held is False
    assert [value.source_path.read_bytes() for value in contracts] == before
    records = [
        json.loads(line)
        for line in runtimes[0].stores["train"].journal_path.read_text().splitlines()
    ]
    assert sum(row["kind"] == "batch_terminal" for row in records) == 1


@pytest.mark.parametrize("vendor_api", [False, True], ids=["direct-core", "vendor-api"])
def test_label_studio_draft_runtime_and_materialize_main_share_authority(
    tmp_path, monkeypatch, capsys, vendor_api
):
    contract = _source(tmp_path, "train")
    source_before = contract.source_path.read_bytes()
    selected = project.Split.TRAIN
    contracts = dict(project.SOURCE_CONTRACTS)
    contracts[selected] = replace(
        contracts[selected],
        sha256=contract.expected_source_sha256,
        row_count=1,
        box_count=1,
    )
    monkeypatch.setattr(project, "SOURCE_CONTRACTS", contracts)
    label_config = build_label_config()
    plan = project.build_split_project_plan(
        contract.source_path,
        repo_root=tmp_path,
        split=selected,
        vendor_revision="temporary-entrypoint-test",
        label_config=label_config,
        label_config_fingerprint=label_config_fingerprint(label_config),
        registry=COCO80_REGISTRY,
    )
    layout = RefinementRuntimeLayout.under_repository(tmp_path)

    class Catalog:
        capture_count = 0
        snapshot = None

        def verify(self, identity):
            return all(
                getattr(identity, name) == getattr(self.snapshot, name)
                for name in identity.__dataclass_fields__
            )

        def resolve(self, receipt_id):
            raise AssertionError("this fixture has no inference receipts")

        def capture_current_user_drafts(self, request):
            self.capture_count += 1
            assert request.principal.user_id == "reviewer"
            return DraftCatalogCapture(
                "train", "ls-train", "reviewer", 0, (self.snapshot,)
            )

    catalog = Catalog()
    if vendor_api:
        # The separately versioned Django application retains this real interface.
        from src.label_studio_coco_refinement.store import WorkingDatasetStore as VendorStore

        bootstrap = VendorStore.bootstrap
        binding_options = {}
    else:
        bootstrap = WorkingDatasetStore.bootstrap
        binding_options = {"task_index_binder": project.bind_bootstrap_task_index_fingerprints}
    store = bootstrap(
        BootstrapSpec(
            split="train",
            source_path=contract.source_path,
            runtime_root=layout.root,
            image_root=contract.image_root,
            expected_source_sha256=contract.expected_source_sha256,
            project_id="ls-train",
            storage_id="ls-images",
            adapter_version="entrypoint-test",
            vendor_revision="temporary-entrypoint-test",
            registry_fingerprint=COCO80_REGISTRY.fingerprint,
            label_config_fingerprint=plan.manifest.label_config_fingerprint,
            extra_fingerprints={"task_manifest": plan.task_manifest.fingerprint},
        ),
        annotation_verifier=catalog,
        inference_receipt_resolver=catalog,
        **binding_options,
    ).store
    assert isinstance(store, WorkingDatasetStore)
    assert type(store)._publish_batch_candidate is WorkingDatasetStore._publish_batch_candidate
    assert type(store).recover is WorkingDatasetStore.recover
    x, y, width, height = norm1000_bbox_to_label_studio_xywh((11, 20, 300, 400))
    canonical = canonicalize_label_studio_draft(
        [
            {
                "id": "train:coco:100",
                "type": "rectanglelabels",
                "from_name": "bbox",
                "to_name": "image",
                "original_width": 32,
                "original_height": 24,
                "image_rotation": 0,
                "value": {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "rotation": 0,
                    "rectanglelabels": ["person"],
                },
                "meta": {
                    "coordexp_region_key": "train:coco:100",
                    "coco_ann_id": 100,
                    "last_committed_bbox": [11, 20, 300, 400],
                },
            }
        ],
        split="train",
        image_id=1,
        image_width=32,
        image_height=24,
    )
    catalog.snapshot = AuthoritativeDraftSnapshot(
        split="train",
        project_id="ls-train",
        image_id=1,
        task_id="train:1",
        annotation_id="annotation-1",
        draft_id="draft-1",
        annotation_revision="revision-1",
        draft_updated_at="2026-10-06T00:00:00Z",
        semantic_hash=canonical.semantic_hash,
        result_hash=canonical.result_hash,
        base_row_hash=store.restore_draft(1).row_hash,
        observed_generation=0,
        regions=canonical.to_json_regions(),
    )
    runtime = RefinementRuntime(
        catalog=catalog, stores={"train": store}, project_ids={"train": "ls-train"}
    )
    principal = AuthenticatedPrincipal("reviewer", authenticated=True)
    queued = runtime.capture_and_enqueue(
        split="train", batch_id="ls-entry", principal=principal
    )
    assert queued.status == "queued" and store.state_snapshot().generation == 0
    runtime.start_workers()
    try:
        _wait(
            lambda: store.get_batch_status("ls-entry").status is BatchStatus.SUCCEEDED
        )
        assert (
            runtime.capture_and_enqueue(
                split="train", batch_id="ls-entry", principal=principal
            ).status
            == "succeeded"
        )
        assert catalog.capture_count == 1
    finally:
        runtime.stop_workers(timeout=5)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "materialize_label_studio_coco_refinement.py",
            "--repo-root",
            str(tmp_path),
            "--split",
            "train",
        ],
    )
    journal_before = store.journal_path.read_bytes()
    assert materializer_cli.main() == 0
    first = json.loads(capsys.readouterr().out)
    assert first["materialization"]["generation"] == 1
    assert first["materialization"]["row_count"] == 1
    assert len(tuple(iter_raw_examples(layout.working_coord("train")))) == 1
    output_before = layout.working_coord("train").read_bytes()
    assert materializer_cli.main() == 0
    assert json.loads(capsys.readouterr().out)["materialization"]["generation"] == 1
    assert layout.working_coord("train").read_bytes() == output_before
    assert store.journal_path.read_bytes() == journal_before
    assert contract.source_path.read_bytes() == source_before
