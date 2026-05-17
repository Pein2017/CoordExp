from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from public_data.pipeline import stages
from public_data.pipeline.planner import PipelinePlanner
from public_data.pipeline.types import PipelineConfig, PipelineState, SplitArtifactPaths
from public_data.scripts import convert_to_coord_tokens


def test_convert_to_coord_tokens_help_marks_legacy_debug_surface() -> None:
    script = Path("public_data/scripts/convert_to_coord_tokens.py")

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": "."},
        text=True,
    )

    module_text = convert_to_coord_tokens.__doc__ or ""
    combined = f"{module_text}\n{result.stdout}".lower()

    assert "legacy/debug" in combined
    assert "not the canonical phase 1 public_data/coco/views writer" in combined
    assert "public_data/scripts/build_coco_views.py" in combined


def test_planned_canonical_coco_view_factory_route_is_documented_without_import() -> None:
    assert (
        getattr(stages, "CANONICAL_COCO_VIEW_FACTORY_PATH", None)
        == "public_data/scripts/build_coco_views.py"
    )
    scope = getattr(stages, "LEGACY_SHARED_PRESET_STAGE_SCOPE", "")
    assert "coco" in scope.lower()
    assert "shared/legacy" in scope.lower()


def test_shared_pipeline_stage_defaults_remain_legacy_coord_token_preset_outputs(
    tmp_path: Path,
) -> None:
    planner = PipelinePlanner()

    coord_stage_names_by_dataset: dict[str, list[str]] = {}
    full_stage_names_by_dataset: dict[str, list[str]] = {}
    for dataset_id in ("coco", "lvis", "vg", "vg_ref"):
        state = _build_state(tmp_path=tmp_path, dataset_id=dataset_id)
        coord_stage_names_by_dataset[dataset_id] = [
            stage.name
            for stage in planner._build_stages(
                state=state,
                mode="coord",
                validate_raw=True,
                validate_preset=True,
            )
        ]
        full_stage_names_by_dataset[dataset_id] = [
            stage.name
            for stage in planner._build_stages(
                state=state,
                mode="full",
                validate_raw=True,
                validate_preset=True,
            )
        ]

    assert set(coord_stage_names_by_dataset) == {"coco", "lvis", "vg", "vg_ref"}
    assert len({tuple(names) for names in coord_stage_names_by_dataset.values()}) == 1
    assert len({tuple(names) for names in full_stage_names_by_dataset.values()}) == 1

    assert coord_stage_names_by_dataset["coco"] == [
        "structural_preflight",
        "max_objects_filter",
        "structural_preflight",
        "normalize_norm1000",
        "structural_preflight",
        "coord_tokens",
    ]
    assert full_stage_names_by_dataset["coco"] == [
        "structural_preflight",
        "rescale",
        "structural_preflight",
        "normalize_norm1000",
        "structural_preflight",
        "coord_tokens",
    ]
    for stage_names in (
        *coord_stage_names_by_dataset.values(),
        *full_stage_names_by_dataset.values(),
    ):
        assert all("view" not in stage_name for stage_name in stage_names)


def _build_state(*, tmp_path: Path, dataset_id: str) -> PipelineState:
    dataset_dir = tmp_path / "public_data" / dataset_id
    preset_dir = dataset_dir / "rescale_32_768_bbox"
    split_artifacts = {
        split: SplitArtifactPaths(
            split=split,
            raw=preset_dir / f"{split}.jsonl",
            norm=preset_dir / f"{split}.norm.jsonl",
            coord=preset_dir / f"{split}.coord.jsonl",
            filter_stats=preset_dir / f"{split}.filter_stats.json",
        )
        for split in ("train", "val")
    }

    return PipelineState(
        config=PipelineConfig(
            dataset_id=dataset_id,
            dataset_dir=dataset_dir,
            raw_dir=dataset_dir / "raw",
            preset="rescale_32_768_bbox",
            num_workers=1,
            run_validation_stage=False,
        ),
        effective_preset="rescale_32_768_bbox",
        base_preset="rescale_32_768_bbox",
        base_preset_dir=preset_dir,
        is_derived_preset=False,
        preset_dir=preset_dir,
        split_inputs={},
        split_artifacts=split_artifacts,
    )
