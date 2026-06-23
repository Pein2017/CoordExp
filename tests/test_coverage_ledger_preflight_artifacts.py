from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest
from PIL import Image

from src.config.loader import ConfigLoader
from src.detection.dataset import DetectionTrainingDataset
from src.training.coverage_ledger.artifacts import (
    CoverageLedgerOverlayCandidate,
    CoverageLedgerPreflightArtifactInputs,
    write_coverage_ledger_preflight_artifacts,
)
from src.training.coverage_ledger.preflight import (
    assert_smoke_config_diff_allowed,
    build_preflight_swift_template,
    _require_existing_file,
    _require_existing_path,
    _resolve_local_model_path,
    resolve_overlay_render_image_path,
)
from src.training.coverage_ledger.sidecars import (
    CoverageLedgerObjectEntry,
    CoverageLedgerSidecar,
)
from src.training.sidecars import TrainingSidecars
from src.training.coverage_ledger.visual_regions import VisualTokenRegion
from test_detection_training_dataset import FakeSwiftTemplate, _raw_row, _write_jsonl


REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_ROOT = REPO_ROOT / "configs/stage1/detection_teacher_forcing/smoke"
BASELINE_CONFIG = SMOKE_ROOT / "coverage_ledger_closed_hard_sft_128_baseline.yaml"
LEDGER_CONFIG = SMOKE_ROOT / "coverage_ledger_closed_hard_sft_128.yaml"

REQUIRED_MANIFEST_KEYS = {
    "schema_version",
    "source_jsonl_path",
    "source_jsonl_repo_path",
    "source_jsonl_resolved_path",
    "source_jsonl_sha256",
    "dataset_id",
    "split",
    "selected_row_indices",
    "selected_sample_ids",
    "selection_seed",
    "selection_algorithm",
    "template_id",
    "object_field_order",
    "tokenizer_id",
    "model_id",
    "processor_do_resize",
    "image_grid_metadata_version",
    "samples",
}

REQUIRED_SAMPLE_KEYS = {
    "row_index",
    "sample_id",
    "object_count",
    "image_identity",
    "processed_width",
    "processed_height",
    "image_grid_thw",
}


def _make_sidecar(index: int, image_path: Path) -> CoverageLedgerSidecar:
    entry = CoverageLedgerObjectEntry(
        object_instance_id=f"image-{index}:ann-{index}:src-0",
        source_object_index=0,
        emitted_order_index=0,
        image_index=0,
        bbox_norm1000_xyxy=(125, 150, 625, 700),
        object_ref_end_position=20 + index,
        box_start_position=21 + index,
        coord_label_positions=(22 + index, 23 + index, 24 + index, 25 + index),
        box_end_position=26 + index,
    )
    return CoverageLedgerSidecar(
        sample_id=f"sample-{index:03d}",
        prompt_end_position=19 + index,
        object_entries=(entry,),
        image_grid_thw=(1, 8, 8),
        processed_width=80,
        processed_height=80,
        image_identity=str(image_path),
    )


def _make_artifact_inputs(tmp_path: Path) -> CoverageLedgerPreflightArtifactInputs:
    source_jsonl = tmp_path / "train.coord.jsonl"
    source_jsonl.write_text(
        "".join(json.dumps({"row": index}) + "\n" for index in range(128)),
        encoding="utf-8",
    )
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    sidecars: list[CoverageLedgerSidecar] = []
    regions_by_sample_id: dict[str, tuple[VisualTokenRegion, ...]] = {}
    overlay_candidates: list[CoverageLedgerOverlayCandidate] = []
    for index in range(128):
        image_path = image_dir / f"{index:03d}.png"
        Image.new(
            "RGB",
            (80, 80),
            color=(240 - index % 50, 245 - index % 40, 250 - index % 30),
        ).save(image_path)
        sidecar = _make_sidecar(index, image_path)
        sidecars.append(sidecar)
        region = VisualTokenRegion(
            row_start=1,
            row_end=4,
            col_start=1,
            col_end=4,
            flattened_indices=(5, 6, 7, 9, 10, 11, 13, 14, 15),
        )
        regions_by_sample_id[sidecar.sample_id] = (region,)
        if index < 16:
            overlay_candidates.append(
                CoverageLedgerOverlayCandidate(
                    sample_id=sidecar.sample_id,
                    row_index=index,
                    object_entry=sidecar.object_entries[0],
                    visual_region=region,
                    render_image_path=image_path,
                )
            )
    return CoverageLedgerPreflightArtifactInputs(
        output_root=tmp_path / "coverage_ledger_preflight_smoke",
        source_jsonl_path=source_jsonl,
        dataset_id="unit-detection",
        split="train",
        selected_row_indices=tuple(range(128)),
        selection_seed=20260623,
        selection_algorithm="seeded_random_without_replacement_v0",
        template_id="compact_object_box_closed",
        object_field_order="desc_first",
        tokenizer_id="unit-tokenizer",
        model_id="unit-model",
        processor_do_resize=False,
        image_grid_metadata_version="qwen3_vl_image_grid_thw_v0",
        sidecars=tuple(sidecars),
        visual_regions_by_sample_id=regions_by_sample_id,
        overlay_candidates=tuple(overlay_candidates),
        visual_grid_shape_by_sample_id={
            sidecar.sample_id: (4, 4) for sidecar in sidecars
        },
        expected_sample_count=128,
        expected_overlay_count=16,
    )


def test_artifact_writer_materializes_manifest_alignment_jsonl_and_overlay_index(
    tmp_path: Path,
) -> None:
    result = write_coverage_ledger_preflight_artifacts(_make_artifact_inputs(tmp_path))
    ledger_root = tmp_path / "coverage_ledger_preflight_smoke" / "ledger"

    assert result.ledger_root == ledger_root
    assert (ledger_root / "selected_samples.json").is_file()
    assert (ledger_root / "alignment_debug.jsonl").is_file()
    assert (ledger_root / "overlays").is_dir()
    assert (ledger_root / "overlays/index.json").is_file()

    manifest = json.loads((ledger_root / "selected_samples.json").read_text())
    assert REQUIRED_MANIFEST_KEYS <= set(manifest)
    assert manifest["source_jsonl_repo_path"] is None
    assert manifest["source_jsonl_resolved_path"] == str(
        (tmp_path / "train.coord.jsonl").resolve()
    )
    assert manifest["source_jsonl_sha256"]
    assert manifest["selected_row_indices"] == list(range(128))
    assert manifest["selected_sample_ids"] == [
        f"sample-{index:03d}" for index in range(128)
    ]
    assert manifest["selection_seed"] == 20260623
    assert manifest["selection_algorithm"] == "seeded_random_without_replacement_v0"
    assert manifest["template_id"] == "compact_object_box_closed"
    assert manifest["object_field_order"] == "desc_first"
    assert manifest["processor_do_resize"] is False
    assert len(manifest["samples"]) == 128
    assert REQUIRED_SAMPLE_KEYS <= set(manifest["samples"][0])

    alignment_lines = (ledger_root / "alignment_debug.jsonl").read_text().splitlines()
    assert len(alignment_lines) == 128
    first_alignment = json.loads(alignment_lines[0])
    assert first_alignment["failure_status"] == "ok"
    assert first_alignment["prompt_end_position"] == 19
    assert first_alignment["objects"][0]["object_ref_end_position"] == 20
    assert first_alignment["objects"][0]["box_start_position"] == 21
    assert first_alignment["objects"][0]["coord_label_positions"] == [22, 23, 24, 25]
    assert first_alignment["objects"][0]["box_end_position"] == 26
    assert first_alignment["objects"][0]["bbox_norm1000_xyxy"] == [125, 150, 625, 700]
    assert first_alignment["objects"][0]["mapped_visual_cells"] == {
        "row_start": 1,
        "row_end": 4,
        "col_start": 1,
        "col_end": 4,
        "flattened_indices": [5, 6, 7, 9, 10, 11, 13, 14, 15],
    }
    assert {json.loads(line)["failure_status"] for line in alignment_lines} == {"ok"}

    overlay_index = json.loads((ledger_root / "overlays/index.json").read_text())
    assert len(overlay_index["overlays"]) == 16
    overlay_paths = [ledger_root / item["path"] for item in overlay_index["overlays"]]
    assert all(path.is_file() for path in overlay_paths)
    assert all(path.suffix == ".png" for path in overlay_paths)
    assert overlay_index["overlays"][0]["sample_id"] == "sample-000"
    assert overlay_index["overlays"][0]["source_object_index"] == 0
    assert overlay_index["overlays"][0]["emitted_order_index"] == 0
    assert overlay_index["overlays"][0]["template_id"] == "compact_object_box_closed"


def test_artifact_writer_rejects_stale_overlay_files(tmp_path: Path) -> None:
    inputs = _make_artifact_inputs(tmp_path)
    stale_overlay = inputs.output_root / "ledger" / "overlays" / "stale.png"
    stale_overlay.parent.mkdir(parents=True)
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(stale_overlay)

    with pytest.raises(ValueError, match="stale|non-empty|overlays"):
        write_coverage_ledger_preflight_artifacts(inputs)


def test_preflight_enforces_smoke_config_diff_allowlist() -> None:
    baseline = ConfigLoader.load_materialized_training_config(str(BASELINE_CONFIG))
    ledger = ConfigLoader.load_materialized_training_config(str(LEDGER_CONFIG))

    assert_smoke_config_diff_allowed(baseline, ledger)


def test_preflight_swift_template_builder_uses_current_swift_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}

    class _Template:
        def normalize_bbox(self, inputs: Any) -> None:
            raise AssertionError("coord adapter did not patch normalize_bbox")

    processor = object()
    template = _Template()

    def _fake_get_model_processor(*args: Any, **kwargs: Any) -> tuple[None, object]:
        calls["model_processor"] = (args, kwargs)
        return None, processor

    def _fake_get_template(*args: Any, **kwargs: Any) -> _Template:
        calls["template"] = (args, kwargs)
        assert args == ()
        assert kwargs["processor"] is processor
        return template

    monkeypatch.setattr(
        "swift.model.get_model_processor",
        _fake_get_model_processor,
    )
    monkeypatch.setattr("swift.template.get_template", _fake_get_template)

    training_config = SimpleNamespace(
        model={
            "model": "local-model",
            "torch_dtype": "float32",
        },
        template={
            "template": "qwen3_vl",
            "max_length": 4096,
            "truncation_strategy": "raise",
            "max_pixels": 1048576,
        },
    )

    result = build_preflight_swift_template(
        training_config,  # type: ignore[arg-type]
        system_prompt="system",
    )

    assert result is template
    assert calls["model_processor"][0] == ("local-model",)
    assert calls["model_processor"][1]["load_model"] is False
    assert calls["model_processor"][1]["download_model"] is False
    assert calls["template"][1]["template_type"] == "qwen3_vl"
    assert getattr(template, "_coord_tokens_skip_norm") is True


def test_preflight_prerequisite_guards_fail_before_swift_download(
    tmp_path: Path,
) -> None:
    missing_jsonl = tmp_path / "missing.coord.jsonl"
    with pytest.raises(FileNotFoundError, match="training JSONL"):
        _require_existing_file(missing_jsonl, "training JSONL")

    missing_model = tmp_path / "model-cache"
    with pytest.raises(FileNotFoundError, match="model cache"):
        _require_existing_path(missing_model, "model cache")

    assert _resolve_local_model_path("model_cache/models/local") == (
        REPO_ROOT / "model_cache/models/local"
    ).resolve(strict=False)
    assert _resolve_local_model_path("Qwen/Qwen3-VL-2B-Instruct") is None


class _ImageGridSwiftTemplate(FakeSwiftTemplate):
    def encode(
        self,
        payload: Mapping[str, Any],
        *,
        return_length: bool,
        do_resize: bool | None = None,
    ) -> dict[str, Any]:
        encoded = super().encode(payload, return_length=return_length)
        encoded["image_grid_thw"] = (1, 8, 8)
        return encoded


def test_preflight_overlay_render_path_uses_resolved_dataset_image_path(
    tmp_path: Path,
) -> None:
    row = _raw_row()
    row["file_name"] = "images/train2017/example.jpg"
    row["images"] = ["images/train2017/example.jpg"]
    jsonl_path = tmp_path / "train.coord.jsonl"
    _write_jsonl(jsonl_path, [row])
    image_path = tmp_path / "image-root/images/train2017/example.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (640, 480), color=(245, 245, 245)).save(image_path)

    dataset = DetectionTrainingDataset.from_jsonl(
        jsonl_path,
        swift_template=_ImageGridSwiftTemplate(),
        image_root=tmp_path / "image-root",
        detection_template_id="compact_object_box_closed",
        mode="random_order_sft",
        object_ordering="sorted",
        user_prompt="Detect every object.",
        system_prompt="You are a detector.",
        seed=20260623,
        state_weighting="uniform_permutation",
        normalization="semantic_image_bucket_balanced",
        object_field_order="desc_first",
        teacher_forcing_profile="hard_sft",
        teacher_forcing_rollin_base_seed=17,
        coverage_ledger_enabled=True,
        dataset_name="unit",
    )

    sample = dataset[0]
    training_sidecars = sample["training_sidecars"]
    assert isinstance(training_sidecars, TrainingSidecars)
    sidecar = training_sidecars.supervision.payloads[0]
    assert isinstance(sidecar, CoverageLedgerSidecar)
    assert sidecar.image_identity == "images/train2017/example.jpg"

    assert resolve_overlay_render_image_path(sample) == image_path.resolve()
