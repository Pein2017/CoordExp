from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from src.config.loader import ConfigLoader
from src.training.coverage_ledger.artifacts import (
    CoverageLedgerOverlayCandidate,
    CoverageLedgerPreflightArtifactInputs,
    write_coverage_ledger_preflight_artifacts,
)
from src.training.coverage_ledger.preflight import assert_smoke_config_diff_allowed
from src.training.coverage_ledger.sidecars import (
    CoverageLedgerObjectEntry,
    CoverageLedgerSidecar,
)
from src.training.coverage_ledger.visual_regions import VisualTokenRegion


REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_ROOT = REPO_ROOT / "configs/stage1/detection_teacher_forcing/smoke"
BASELINE_CONFIG = SMOKE_ROOT / "coverage_ledger_closed_hard_sft_128_baseline.yaml"
LEDGER_CONFIG = SMOKE_ROOT / "coverage_ledger_closed_hard_sft_128.yaml"

REQUIRED_MANIFEST_KEYS = {
    "schema_version",
    "source_jsonl_path",
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
                    image_path=image_path,
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


def test_preflight_enforces_smoke_config_diff_allowlist() -> None:
    baseline = ConfigLoader.load_materialized_training_config(str(BASELINE_CONFIG))
    ledger = ConfigLoader.load_materialized_training_config(str(LEDGER_CONFIG))

    assert_smoke_config_diff_allowed(baseline, ledger)
