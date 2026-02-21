from __future__ import annotations

import json
import os
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, Literal, Set

from .adapters import AdapterRegistry, build_default_registry
from .naming import resolve_effective_preset
from .stages import (
    CoordTokenStage,
    DerivedImagesHardlinkStage,
    MaxObjectsFilterStage,
    NormalizeStage,
    PipelineStage,
    RescaleStage,
    StructuralPreflightStage,
    ValidationStage,
)
from .types import PipelineConfig, PipelineResult, PipelineState, SplitArtifactPaths
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
        if config.max_objects is not None:
            max_n = int(config.max_objects)
            if max_n <= 0:
                raise ValueError("max_objects must be > 0 when provided")
            if mode != "coord":
                raise ValueError(
                    "max_objects is only supported for mode 'coord'. "
                    "Run rescale/full first, then coord with max_objects."
                )

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
        effective_preset = resolve_effective_preset(config.preset, config.max_objects)
        base_preset = config.preset
        base_preset_dir = config.dataset_dir / base_preset
        is_derived_preset = base_preset != effective_preset
        preset_dir = config.dataset_dir / effective_preset

        split_inputs: Dict[str, Path] = {}
        if needs_raw_inputs:
            split_inputs = adapter.split_input_paths(config.raw_dir)
            split_names = sorted(split_inputs.keys())
        else:
            split_names = self._detect_splits_from_preset(
                preset_dir=preset_dir,
                base_preset_dir=base_preset_dir,
            )

        split_artifacts = {
            split: build_split_artifact_paths(preset_dir, split)
            for split in split_names
        }

        state = PipelineState(
            config=replace(config, preset=effective_preset),
            effective_preset=effective_preset,
            base_preset=base_preset,
            base_preset_dir=base_preset_dir,
            is_derived_preset=is_derived_preset,
            preset_dir=preset_dir,
            split_inputs=split_inputs,
            split_artifacts=split_artifacts,
        )

        if mode == "coord" or (mode == "validate" and validate_preset):
            self._prepare_preset_sources_for_coord_or_validate(state=state, base_preset=base_preset)
        return state

    def _detect_splits_from_preset(self, *, preset_dir: Path, base_preset_dir: Path) -> list[str]:
        split_names: list[str] = []
        for split in ("train", "val"):
            candidates = [
                preset_dir / f"{split}.jsonl",
                base_preset_dir / f"{split}.jsonl",
            ]
            if any(p.exists() for p in candidates):
                split_names.append(split)

        if not split_names:
            raise FileNotFoundError(
                "Could not infer preset splits. Expected at least train.jsonl under "
                f"{preset_dir} (or base preset {base_preset_dir})."
            )
        return split_names

    def _prepare_preset_sources_for_coord_or_validate(self, *, state: PipelineState, base_preset: str) -> None:
        preset_dir = state.preset_dir
        base_preset_dir = state.config.dataset_dir / base_preset
        is_derived_preset = base_preset != state.effective_preset

        if is_derived_preset:
            if not base_preset_dir.exists():
                raise FileNotFoundError(
                    "Derived preset requested but base preset directory is missing: "
                    f"{base_preset_dir} (effective={state.effective_preset})"
                )
            preset_dir.mkdir(parents=True, exist_ok=True)

        for split in sorted(state.split_artifacts.keys()):
            paths = state.split_artifacts[split]
            candidate_sources: list[Path] = []

            if paths.raw.exists():
                candidate_sources.append(paths.raw)

            if is_derived_preset:
                base_raw = base_preset_dir / f"{split}.jsonl"
                if base_raw.exists():
                    candidate_sources.append(base_raw)

            src = next((p for p in candidate_sources if p.exists()), None)
            if src is None:
                expected = [str(paths.raw)]
                if is_derived_preset:
                    expected.append(str(base_preset_dir / f"{split}.jsonl"))
                raise FileNotFoundError(
                    f"Missing preset raw input for split '{split}'. Expected one of: "
                    + ", ".join(expected)
                )

            if src != paths.raw:
                paths.raw.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, paths.raw)
            state.split_raw_sources[split] = paths.raw


    @staticmethod
    def _iter_required_image_paths(jsonl_paths: Iterable[Path]) -> Set[Path]:
        required: Set[Path] = set()
        for jsonl_path in jsonl_paths:
            with jsonl_path.open("r", encoding="utf-8") as fin:
                for line_num, raw_line in enumerate(fin, start=1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError(
                            "Derived preset image materialization requires valid JSONL. "
                            f"Failed to parse {jsonl_path}:{line_num} ({exc})."
                        ) from exc

                    images = row.get("images")
                    if not isinstance(images, list) or not images or not isinstance(images[0], str):
                        raise RuntimeError(
                            "Derived preset image materialization expects JSONL records with "
                            f"non-empty images list; got invalid record at {jsonl_path}:{line_num}."
                        )

                    rel_path = Path(images[0])
                    if rel_path.is_absolute() or ".." in rel_path.parts:
                        raise RuntimeError(
                            "Derived preset image path must be a safe relative path under preset dir: "
                            f"{images[0]!r} at {jsonl_path}:{line_num}."
                        )
                    if not rel_path.parts or rel_path.parts[0] != "images":
                        raise RuntimeError(
                            "Derived preset image path must keep stable 'images/...'-style layout: "
                            f"{images[0]!r} at {jsonl_path}:{line_num}."
                        )
                    required.add(rel_path)
        return required

    @classmethod
    def _materialize_derived_images_hardlinks(
        cls,
        *,
        base_preset_dir: Path,
        derived_preset_dir: Path,
        split_paths: Dict[str, SplitArtifactPaths],
    ) -> None:
        base_images = base_preset_dir / "images"
        derived_images = derived_preset_dir / "images"

        if not base_images.exists() or not base_images.is_dir() or base_images.is_symlink():
            raise RuntimeError(
                "Base preset images directory must exist as a real directory before creating derived preset: "
                f"{base_images}"
            )

        derived_preset_dir.mkdir(parents=True, exist_ok=True)
        if derived_images.exists():
            if derived_images.is_symlink() or derived_images.is_file() or not derived_images.is_dir():
                raise RuntimeError(
                    "Derived preset images path must be a real directory (not symlink/file): "
                    f"{derived_images}"
                )
        else:
            derived_images.mkdir(parents=True, exist_ok=True)

        try:
            base_dev = base_images.stat().st_dev
            derived_dev = derived_images.stat().st_dev
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Failed to stat base/derived images directories for hardlink precheck."
            ) from exc
        if base_dev != derived_dev:
            raise RuntimeError(
                "Cannot hardlink derived preset images across filesystems. "
                f"base={base_images} (st_dev={base_dev}) derived={derived_images} (st_dev={derived_dev})."
            )

        image_rel_paths = cls._iter_required_image_paths(
            jsonl_paths=[split_paths[split].raw for split in sorted(split_paths.keys())]
        )

        for rel_path in sorted(image_rel_paths):
            src_path = base_preset_dir / rel_path
            dst_path = derived_preset_dir / rel_path

            if not src_path.exists() or not src_path.is_file() or src_path.is_symlink():
                raise RuntimeError(
                    "Missing or invalid base preset image required for derived preset: "
                    f"{src_path}. Rebuild/repair base preset first."
                )

            dst_path.parent.mkdir(parents=True, exist_ok=True)

            if dst_path.exists():
                if dst_path.is_symlink() or dst_path.is_dir():
                    raise RuntimeError(
                        "Derived preset image destination must be a file path, not symlink/dir: "
                        f"{dst_path}"
                    )
                src_stat = src_path.stat()
                dst_stat = dst_path.stat()
                if src_stat.st_ino == dst_stat.st_ino and src_stat.st_dev == dst_stat.st_dev:
                    continue
                raise RuntimeError(
                    "Derived preset image already exists but is not the same hardlink as base image: "
                    f"dst={dst_path} src={src_path}."
                )

            try:
                os.link(src_path, dst_path)
            except OSError as exc:
                raise RuntimeError(
                    "Failed to materialize derived preset image hardlink (no byte-copy fallback): "
                    f"src={src_path} dst={dst_path} ({exc})."
                ) from exc

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
                    StructuralPreflightStage(source="raw_output"),
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
            stages.append(StructuralPreflightStage(source="raw_output"))
            stages.append(MaxObjectsFilterStage())
            if state.is_derived_preset:
                stages.append(
                    DerivedImagesHardlinkStage(
                        materialize_fn=self._materialize_derived_images_hardlinks,
                    )
                )
            stages.extend(
                [
                    StructuralPreflightStage(source="raw_output"),
                    NormalizeStage(),
                    StructuralPreflightStage(source="norm_output"),
                    CoordTokenStage(),
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
                    StructuralPreflightStage(source="raw_output"),
                    NormalizeStage(),
                    StructuralPreflightStage(source="norm_output"),
                    CoordTokenStage(),
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
