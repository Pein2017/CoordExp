from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path
from typing import Dict, Literal

from .adapters import AdapterRegistry, build_default_registry
from .naming import resolve_effective_preset
from .stages import (
    CoordTokenStage,
    LegacyAliasStage,
    MaxObjectsFilterStage,
    NormalizeStage,
    PipelineStage,
    RescaleStage,
    StructuralPreflightStage,
    ValidationStage,
)
from .types import PipelineConfig, PipelineResult, PipelineState
from .writer import build_split_artifact_paths, write_pipeline_manifest

PipelineMode = Literal["rescale", "coord", "validate", "full"]


class PipelinePlanner:
    def __init__(self, *, registry: AdapterRegistry | None = None) -> None:
        self.registry = registry or build_default_registry()

    def run(
        self,
        *,
        config: PipelineConfig,
        mode: PipelineMode,
        validate_raw: bool = True,
        validate_preset: bool = True,
    ) -> PipelineResult:
        needs_raw_inputs = mode in ("rescale", "full") or (mode == "validate" and validate_raw)
        state = self._build_state(
            config=config,
            mode=mode,
            needs_raw_inputs=needs_raw_inputs,
            validate_preset=validate_preset,
        )
        stages = self._build_stages(
            state=state,
            mode=mode,
            validate_raw=validate_raw,
            validate_preset=validate_preset,
        )
        for stage in stages:
            stage.run(state)

        if mode != "validate":
            manifest_path = write_pipeline_manifest(
                preset_dir=state.preset_dir,
                dataset_id=state.config.dataset_id,
                preset=state.effective_preset,
                max_objects=state.config.max_objects,
                split_paths=state.split_artifacts,
                stage_stats=state.stage_stats,
            )
            state.stage_stats["manifest"] = {"path": str(manifest_path)}

        return PipelineResult(
            dataset_id=state.config.dataset_id,
            preset=state.effective_preset,
            preset_dir=state.preset_dir,
            split_artifacts=state.split_artifacts,
            stage_stats=state.stage_stats,
        )

    def _build_state(
        self,
        *,
        config: PipelineConfig,
        mode: PipelineMode,
        needs_raw_inputs: bool,
        validate_preset: bool,
    ) -> PipelineState:
        adapter = self.registry.get(config.dataset_id)
        effective_preset = resolve_effective_preset(
            config.dataset_dir,
            config.preset,
            config.max_objects,
        )
        preset_dir = config.dataset_dir / effective_preset

        split_inputs: Dict[str, Path] = {}
        if needs_raw_inputs:
            split_inputs = adapter.split_input_paths(config.raw_dir)
            split_names = sorted(split_inputs.keys())
        else:
            split_names = self._detect_splits_from_preset(
                preset_dir=preset_dir,
                base_preset_dir=config.dataset_dir / config.preset,
            )

        split_artifacts = {
            split: build_split_artifact_paths(preset_dir, split)
            for split in split_names
        }

        state = PipelineState(
            config=replace(config, preset=effective_preset),
            effective_preset=effective_preset,
            preset_dir=preset_dir,
            split_inputs=split_inputs,
            split_artifacts=split_artifacts,
        )

        if mode == "coord" or (mode == "validate" and validate_preset):
            self._prepare_preset_sources_for_coord_or_validate(state=state, base_preset=config.preset)
        return state

    def _detect_splits_from_preset(self, *, preset_dir: Path, base_preset_dir: Path) -> list[str]:
        split_names: list[str] = []
        for split in ("train", "val"):
            candidates = [
                preset_dir / f"{split}.raw.jsonl",
                preset_dir / f"{split}.jsonl",
                base_preset_dir / f"{split}.raw.jsonl",
                base_preset_dir / f"{split}.jsonl",
            ]
            if any(p.exists() for p in candidates):
                split_names.append(split)

        if not split_names:
            raise FileNotFoundError(
                "Could not infer preset splits. Expected at least train.raw.jsonl or train.jsonl under "
                f"{preset_dir} (or base preset {base_preset_dir})."
            )
        return split_names

    def _prepare_preset_sources_for_coord_or_validate(self, *, state: PipelineState, base_preset: str) -> None:
        preset_dir = state.preset_dir
        base_preset_dir = state.config.dataset_dir / base_preset
        if not preset_dir.exists() and base_preset_dir.exists() and base_preset != state.effective_preset:
            preset_dir.mkdir(parents=True, exist_ok=True)
            self._copy_images_if_needed(src_dir=base_preset_dir, dst_dir=preset_dir)

        for split in sorted(state.split_artifacts.keys()):
            paths = state.split_artifacts[split]
            candidate_sources: list[Path] = []

            if paths.raw.exists():
                candidate_sources.append(paths.raw)
            if paths.legacy_raw_alias.exists():
                candidate_sources.append(paths.legacy_raw_alias)

            if base_preset != state.effective_preset:
                legacy_base = state.config.dataset_dir / base_preset
                base_raw = legacy_base / f"{split}.raw.jsonl"
                base_alias = legacy_base / f"{split}.jsonl"
                if base_raw.exists():
                    candidate_sources.append(base_raw)
                if base_alias.exists():
                    candidate_sources.append(base_alias)
                if (base_raw.exists() or base_alias.exists()) and not (paths.raw.exists() or paths.legacy_raw_alias.exists()):
                    self._copy_images_if_needed(src_dir=legacy_base, dst_dir=state.preset_dir)

            src = next((p for p in candidate_sources if p.exists()), None)
            if src is None:
                raise FileNotFoundError(
                    f"Missing preset raw input for split '{split}'. Expected one of: "
                    f"{paths.raw}, {paths.legacy_raw_alias}"
                )

            if src != paths.raw:
                paths.raw.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, paths.raw)
            state.split_raw_sources[split] = paths.raw

    @staticmethod
    def _copy_images_if_needed(*, src_dir: Path, dst_dir: Path) -> None:
        src_images = src_dir / "images"
        dst_images = dst_dir / "images"
        if not src_images.exists() or dst_images.exists():
            return
        shutil.copytree(src_images, dst_images)

    def _build_stages(
        self,
        *,
        state: PipelineState,
        mode: PipelineMode,
        validate_raw: bool,
        validate_preset: bool,
    ) -> list[PipelineStage]:
        stages: list[PipelineStage] = []

        if mode == "rescale":
            stages.extend(
                [
                    StructuralPreflightStage(source="raw_input"),
                    RescaleStage(),
                    MaxObjectsFilterStage(),
                    StructuralPreflightStage(source="raw_output"),
                    LegacyAliasStage(),
                ]
            )
            if state.config.run_validation_stage:
                stages.append(
                    ValidationStage(
                        include_raw=False,
                        include_preset=True,
                        include_norm=False,
                        include_coord=False,
                        skip_image_check=state.config.skip_image_check,
                    )
                )
            return stages

        if mode == "coord":
            stages.extend(
                [
                    StructuralPreflightStage(source="raw_output"),
                    MaxObjectsFilterStage(),
                    StructuralPreflightStage(source="raw_output"),
                    NormalizeStage(),
                    StructuralPreflightStage(source="norm_output"),
                    CoordTokenStage(),
                    LegacyAliasStage(),
                ]
            )
            if state.config.run_validation_stage:
                stages.append(
                    ValidationStage(
                        include_raw=False,
                        include_preset=True,
                        include_norm=True,
                        include_coord=True,
                        skip_image_check=state.config.skip_image_check,
                    )
                )
            return stages

        if mode == "full":
            stages.extend(
                [
                    StructuralPreflightStage(source="raw_input"),
                    RescaleStage(),
                    MaxObjectsFilterStage(),
                    StructuralPreflightStage(source="raw_output"),
                    NormalizeStage(),
                    StructuralPreflightStage(source="norm_output"),
                    CoordTokenStage(),
                    LegacyAliasStage(),
                ]
            )
            if state.config.run_validation_stage:
                stages.append(
                    ValidationStage(
                        include_raw=True,
                        include_preset=True,
                        include_norm=True,
                        include_coord=True,
                        skip_image_check=state.config.skip_image_check,
                    )
                )
            return stages

        if mode == "validate":
            if validate_raw:
                stages.append(StructuralPreflightStage(source="raw_input"))
            if validate_preset:
                stages.append(StructuralPreflightStage(source="raw_output"))
            stages.append(
                ValidationStage(
                    include_raw=validate_raw,
                    include_preset=validate_preset,
                    include_norm=validate_preset,
                    include_coord=validate_preset,
                    skip_image_check=state.config.skip_image_check,
                )
            )
            return stages

        raise ValueError(f"Unsupported pipeline mode: {mode}")
