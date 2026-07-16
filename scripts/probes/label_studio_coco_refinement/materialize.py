#!/usr/bin/env python3
"""Persistent one-row real-fixture probe for the coord materializer."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import iter_raw_examples  # noqa: E402
from src.data.examples import thaw_json  # noqa: E402
from src.config.models import (  # noqa: E402
    DataAugmentationConfig,
    DataConfig,
    DatasetSplitConfig,
    RuntimeConfig,
    TemplateConfig,
    TemplatePromptConfig,
)
from src.label_studio_coco_refinement.categories import (  # noqa: E402
    COCO80_REGISTRY,
)
from src.label_studio_coco_refinement.materialize import (  # noqa: E402
    OPERATOR_RECEIPT_NAME,
    ProjectTaskIndexSourceIdentityResolver,
    WorkingCoordMaterializer,
)
from src.label_studio_coco_refinement.models import (  # noqa: E402
    RefinementRuntimeLayout,
)
from src.label_studio_coco_refinement.store import (  # noqa: E402
    BootstrapSpec,
    CommitRequest,
    DraftSaveReceipt,
    WorkingDatasetStore,
    semantic_hash,
    sha256_file,
    sha256_json,
)
import src.training.pipeline as training_pipeline  # noqa: E402

FIXTURE = (
    REPO_ROOT
    / "tests/fixtures/label_studio_coco_refinement/train.representative.norm.jsonl"
)
CANONICAL_IMAGE_ROOT = REPO_ROOT / "public_data/coco/rescale_32_1024_bbox/images"


class _AcceptingAnnotationVerifier:
    def verify(self, identity: Any) -> bool:
        return True


class _UnavailableInferenceResolver:
    def resolve(self, receipt_id: str) -> None:
        raise RuntimeError("probe materialization must not consult inference")


def _validate_output_root(value: Path) -> Path:
    candidate = value if value.is_absolute() else REPO_ROOT / value
    candidate = candidate.resolve(strict=False)
    allowed = (
        REPO_ROOT / "outputs/label_studio_coco_refinement/materializer-probes"
    ).resolve(strict=False)
    try:
        candidate.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(f"output root must be below {allowed}") from exc
    if os.path.lexists(candidate):
        raise ValueError(f"output root already exists: {candidate}")
    check = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "check-ignore",
            "-q",
            "--",
            str(candidate.relative_to(REPO_ROOT)),
        ],
        check=False,
    )
    if check.returncode != 0:
        raise ValueError(f"output root is not ignored by git: {candidate}")
    return candidate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a persistent real-fixture materializer probe."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help=(
            "New path below outputs/label_studio_coco_refinement/materializer-probes/."
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    output_root = _validate_output_root(args.output_root)
    repository = output_root / "repository"
    layout = RefinementRuntimeLayout.under_repository(repository)
    selected_source = layout.selected_source("train")
    selected_source.parent.mkdir(parents=True)
    fixture_bytes = FIXTURE.read_bytes()
    selected_source.write_bytes(fixture_bytes)
    layout.image_root.parent.mkdir(parents=True)
    layout.image_root.symlink_to(CANONICAL_IMAGE_ROOT, target_is_directory=True)

    verifier = _AcceptingAnnotationVerifier()
    inference = _UnavailableInferenceResolver()
    bootstrap = WorkingDatasetStore.bootstrap(
        BootstrapSpec(
            split="train",
            source_path=selected_source,
            runtime_root=layout.root,
            image_root=layout.image_root,
            expected_source_sha256=hashlib.sha256(fixture_bytes).hexdigest(),
            project_id="materializer-probe-train",
            storage_id="materializer-probe-storage-train",
            adapter_version="materializer-probe-v1",
            vendor_revision="bounded-real-fixture",
            registry_fingerprint=COCO80_REGISTRY.fingerprint,
            label_config_fingerprint="bounded-real-fixture",
        ),
        annotation_verifier=verifier,
        inference_receipt_resolver=inference,
    )
    restored = bootstrap.store.restore_draft(34)
    region_key_by_id = {
        object_id: region_key
        for region_key, object_id in restored.region_id_mapping.items()
    }
    regions: list[dict[str, Any]] = []
    for source_object in restored.row["objects"]:
        region = copy.deepcopy(source_object)
        region["region_key"] = region_key_by_id[source_object["coco_ann_id"]]
        regions.append(region)
    for ordinal, marker in enumerate(("first", "second"), start=1):
        regions.append(
            {
                "region_key": f"drawn:materializer-probe:{marker}",
                "bbox_2d": [20, 50, 700, 950],
                "desc": "person",
                "category_name": "person",
                "category_id": 1,
                "creation_ordinal": ordinal,
                "metadata": {"probe_tie": marker},
            }
        )
    projection_hash = semantic_hash(regions)
    draft_save = DraftSaveReceipt(
        project_id="materializer-probe-train",
        task_id="train:34",
        annotation_id="probe-annotation-34",
        draft_id="probe-draft-34",
        annotation_revision="probe-v1",
        draft_updated_at="2026-07-16T00:00:00Z",
        semantic_hash=projection_hash,
        result_hash=sha256_json(regions),
    )
    committed = bootstrap.store.commit(
        CommitRequest(
            commit_id="materializer-probe:34",
            split="train",
            image_id=34,
            project_id="materializer-probe-train",
            task_id="train:34",
            annotation_id="probe-annotation-34",
            draft_id="probe-draft-34",
            annotation_revision="probe-v1",
            draft_updated_at="2026-07-16T00:00:00Z",
            semantic_hash=projection_hash,
            result_hash=sha256_json(regions),
            base_row_hash=restored.row_hash,
            observed_generation=restored.generation,
            regions=regions,
            draft_save=draft_save,
        )
    )
    if [item["coco_ann_id"] for item in committed.committed_row["objects"]] != [
        589229,
        -1,
        -2,
    ]:
        raise AssertionError("probe commit did not preserve stable tied-object order")
    resolver = ProjectTaskIndexSourceIdentityResolver._for_bounded_probe(
        layout,
        "train",
        store=bootstrap.store,
    )
    receipt = WorkingCoordMaterializer(
        layout,
        "train",
        store=bootstrap.store,
        source_identity_resolver=resolver,
    ).materialize_current_generation()
    output = layout.working_coord("train")
    loaded = tuple(iter_raw_examples(output))
    if len(loaded) != 1 or not loaded[0].example_id.endswith("_000000000034"):
        raise AssertionError("current loader did not recover representative image 34")
    raw = loaded[0]
    if (
        [item.object_id for item in raw.objects] != ["589229", "-1", "-2"]
        or [item.description for item in raw.objects] != ["zebra", "person", "person"]
        or (raw.image.width, raw.image.height) != (1248, 832)
        or thaw_json(raw.objects[0].metadata)["source"]["category_id"] != 24
        or [
            thaw_json(item.metadata)["source"].get("metadata", {}).get("probe_tie")
            for item in raw.objects[1:]
        ]
        != ["first", "second"]
    ):
        raise AssertionError("current loader compatibility contract drift")

    dataset = DatasetSplitConfig(path=str(output), sample_limit=1)
    training_config = SimpleNamespace(
        data=DataConfig(train=dataset, augmentation=DataAugmentationConfig()),
        runtime=RuntimeConfig(seed=17),
        template=TemplateConfig(
            object_field_order="desc_first",
            object_ordering="source_order",
            assistant_format="object_box_closed",
            prompt=TemplatePromptConfig(
                system=None,
                user="Describe each requested object with its bounding box.",
            ),
        ),
    )
    training_result = training_pipeline._materialize_raw_examples_for_dataset(
        training_config,
        dataset,
        split="train",
    )
    if [item.object_id for item in training_result.examples[0].objects] != [
        "589229",
        "-1",
        "-2",
    ]:
        raise AssertionError("current training data path changed object identity/order")
    artifact = {
        "code": "label_studio.materializer_probe_passed",
        "fixture": {
            "path": str(FIXTURE),
            "sha256": sha256_file(FIXTURE),
        },
        "canonical_image_root": {
            "path": str(CANONICAL_IMAGE_ROOT),
            "resolved": str(layout.image_root.resolve(strict=True)),
        },
        "materializer_receipt": receipt.to_artifact_dict(),
        "durable_receipt_path": str(layout.split_root("train") / OPERATOR_RECEIPT_NAME),
        "output": {
            "path": str(output),
            "sha256": sha256_file(output),
            "row_count": len(loaded),
            "example_ids": [example.example_id for example in loaded],
            "object_ids": [item.object_id for item in raw.objects],
            "category_ids": [
                thaw_json(item.metadata)["source"]["category_id"]
                for item in raw.objects
            ],
            "image_resolution": [raw.image.width, raw.image.height],
            "training_example_count": len(training_result.examples),
        },
    }
    probe_receipt = output_root / "probe.json"
    probe_receipt.write_text(
        json.dumps(
            artifact,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(str(probe_receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
