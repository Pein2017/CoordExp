#!/usr/bin/env python3
"""Build Phase 1 canonical COCO public-data annotation views."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence

from public_data.scripts.build_coco_length_budget_artifacts import (
    CompactFullTokenBudgetEstimator,
    LengthEstimator,
    TokenBudgetBreakdown,
)
from public_data.scripts.convert_to_coord_tokens import (
    _canonicalize_and_sort_objects_in_place,
    convert_record_to_ints,
)
from public_data.view_contracts import (
    ASSISTANT_COORDINATE_RENDERING_QWEN_COORD_TOKENS,
    COORDINATE_CHART_XYXY,
    COORDINATE_RANGE_NORM1000,
    COORDINATE_SPACE_NORM1000,
    COORDINATE_STORAGE_INTEGER,
    IMAGE_PATH_SEMANTICS_IMAGE_STORE_RELATIVE,
    PROXY_ANNOTATION_POLICY_ALL_PROXY,
    SCHEMA_VERSION_V1,
    write_image_store_metadata,
    write_view_metadata,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(
    "configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2.yaml"
)
DEFAULT_MODEL = Path("model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp")
DEFAULT_SOURCE_PRESET = Path("public_data/coco/rescale_32_1024_bbox")
DEFAULT_IMAGE_STORE_ROOT = Path("public_data/coco/images/res-1024")
DEFAULT_VIEWS_ROOT = Path("public_data/coco/views")
DEFAULT_LEGACY_LENGTH_BUDGET_SOURCE = Path(
    "public_data/coco/rescale_32_1024_bbox_len12000"
)
DEFAULT_LEGACY_MAX_OBJECTS_SOURCE = Path("public_data/coco/rescale_32_1024_bbox_max60")
DEFAULT_PROXY_SOURCE = Path("public_data/coco/rescale_32_1024_bbox_lvis_proxy_len12000")
DEFAULT_VIEWS = (
    "coco80/full",
    "coco80/len-12000",
    "coco80/max-60",
    "coco80-lvis-proxy/len-12000",
)
SUPPORTED_IMAGE_STORE_MODES = frozenset(
    {"copy", "hardlink", "reflink", "reuse-existing"}
)
PHASE1_REJECTED_IMAGE_STORE_MODE = "move"
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})


@dataclass(frozen=True)
class CocoViewFactoryConfig:
    """Configuration for the canonical COCO view factory.

    :param repo_root: Repository root used for repo-relative metadata paths.
    :param source_preset: Existing COCO 1024 pixel-space source preset root.
    :param image_store_root: Canonical shared image-store root.
    :param views_root: Canonical COCO views root.
    :param legacy_length_budget_source: Historical length-budget source root.
    :param legacy_max_objects_source: Historical max-60 source root.
    :param proxy_source: Existing all-proxy source root for the research view.
    :param splits: Dataset splits to build.
    :param views: View names to build.
    :param max_total_tokens: Length budget used by ``len-*`` views.
    :param image_store_mode: Explicit image-store adoption mode.
    :param reuse_existing_image_store: Whether a non-empty target is reusable.
    :param dry_run: Whether to validate and report without artifact writes.
    :param dry_run_report: Optional dry-run report path.
    :param config_path: Latest compact detection config for real estimator use.
    :param model_path: Local Qwen tokenizer/processor path for real estimator use.
    """

    repo_root: Path
    source_preset: Path
    image_store_root: Path
    views_root: Path
    legacy_length_budget_source: Path | None
    legacy_max_objects_source: Path | None
    proxy_source: Path | None
    splits: tuple[str, ...]
    views: tuple[str, ...]
    max_total_tokens: int
    image_store_mode: str
    reuse_existing_image_store: bool
    dry_run: bool = False
    dry_run_report: Path | None = None
    config_path: Path | None = None
    model_path: Path | None = None

    def __post_init__(self) -> None:
        """Validate immutable factory options."""

        # checking phase-1 image-store mode
        if self.image_store_mode == PHASE1_REJECTED_IMAGE_STORE_MODE:
            raise ValueError("image-store mode 'move' is unavailable in Phase 1")
        if self.image_store_mode not in SUPPORTED_IMAGE_STORE_MODES:
            allowed = ", ".join(sorted(SUPPORTED_IMAGE_STORE_MODES))
            raise ValueError(f"image_store_mode must be one of: {allowed}")

        # checking required build dimensions
        if not self.splits:
            raise ValueError("splits must not be empty")
        if not self.views:
            raise ValueError("views must not be empty")
        if self.max_total_tokens <= 0:
            raise ValueError("max_total_tokens must be positive")

    @property
    def source_image_dir(self) -> Path:
        """Source image directory under the legacy COCO preset."""

        return self.source_preset / "images"

    @property
    def target_image_dir(self) -> Path:
        """Canonical image directory under the shared image store."""

        return self.image_store_root / "images"

    def view_root(self, view_name: str) -> Path:
        """Return the filesystem root for ``view_name``."""

        return self.views_root / Path(*Path(view_name).parts)


class ImageStoreAdopter:
    """Adopter for the canonical shared COCO image store."""

    def __init__(self, config: CocoViewFactoryConfig) -> None:
        self._config = config
        self._manifest_builder = ViewManifestPayloadBuilder(config)

    def prepare(self) -> dict[str, Any]:
        """Prepare the image store or report what would happen.

        :returns: Summary of source/target counts and mode.
        """

        # validating source and target state
        if not self._config.source_image_dir.is_dir():
            raise FileNotFoundError(
                f"source image directory does not exist: {self._config.source_image_dir}"
            )
        source_images = _list_image_files(self._config.source_image_dir)
        target_images = _list_image_files(self._config.target_image_dir)
        self._validate_target_is_reusable(target_images)

        # preparing dry-run report without touching artifacts
        if self._config.dry_run:
            report = self._build_dry_run_report(source_images, target_images)
            if self._config.dry_run_report is not None:
                self._config.dry_run_report.parent.mkdir(parents=True, exist_ok=True)
                self._config.dry_run_report.write_text(
                    json.dumps(report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            return report

        # adopting files according to the explicit mode
        if self._config.image_store_mode in {"copy", "hardlink", "reflink"}:
            self._copy_image_tree()
            target_images = _list_image_files(self._config.target_image_dir)
        elif self._config.image_store_mode == "reuse-existing":
            if not self._config.target_image_dir.exists():
                raise FileNotFoundError(
                    "reuse-existing image-store mode requires an existing "
                    f"target image directory: {self._config.target_image_dir}"
                )

        # writing the image-store contract
        metadata = self._manifest_builder.build_image_store_metadata()
        write_image_store_metadata(self._config.image_store_root / "meta.json", metadata)

        return {
            "image_store_mode": self._config.image_store_mode,
            "source_image_count": len(source_images),
            "target_image_count": len(target_images),
            "image_store_root": str(self._config.image_store_root),
        }

    def _validate_target_is_reusable(self, target_images: Sequence[Path]) -> None:
        """Fail if an existing target would be overwritten accidentally."""

        if not self._config.image_store_root.exists():
            return
        if not any(self._config.image_store_root.iterdir()):
            return
        if self._config.reuse_existing_image_store:
            return
        if self._config.image_store_mode == "reuse-existing":
            return
        raise FileExistsError(
            "Target image store exists and is non-empty. Use "
            "--reuse-existing-image-store or --image-store-mode reuse-existing: "
            f"{self._config.image_store_root}"
        )

    def _copy_image_tree(self) -> None:
        """Adopt the source image tree into the canonical store."""

        if not self._config.source_image_dir.is_dir():
            raise FileNotFoundError(
                f"source image directory does not exist: {self._config.source_image_dir}"
            )

        self._config.target_image_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            self._config.source_image_dir,
            self._config.target_image_dir,
            copy_function=self._copy_file,
            dirs_exist_ok=True,
        )

    def _copy_file(self, source: str, destination: str) -> str:
        """Copy one file according to the configured adoption mode."""

        if self._config.image_store_mode == "reflink":
            subprocess.run(
                ["cp", "--reflink=always", source, destination],
                check=True,
            )
            shutil.copystat(source, destination)
            return destination

        if self._config.image_store_mode == "hardlink":
            os.link(source, destination)
            return destination

        return shutil.copy2(source, destination)

    def _build_dry_run_report(
        self,
        source_images: Sequence[Path],
        target_images: Sequence[Path],
    ) -> dict[str, Any]:
        """Return the dry-run report payload."""

        # sampling old-to-new image references
        sample_checks = []
        for source_path in source_images[:5]:
            relative = source_path.relative_to(self._config.source_image_dir)
            new_ref = Path("images") / relative
            sample_checks.append(
                {
                    "old_path": str(source_path),
                    "old_exists": source_path.is_file(),
                    "new_ref": new_ref.as_posix(),
                    "new_path": str(self._config.image_store_root / new_ref),
                }
            )

        # estimating byte footprint
        disk_bytes = sum(path.stat().st_size for path in source_images if path.is_file())
        return {
            "dry_run": True,
            "image_store_mode": self._config.image_store_mode,
            "source_image_root": str(self._config.source_image_dir),
            "target_image_root": str(self._config.target_image_dir),
            "source_image_count": len(source_images),
            "target_image_count": len(target_images),
            "estimated_copy_bytes": int(disk_bytes),
            "sample_resolution_checks": sample_checks,
            "planned_outputs": {
                "image_store_meta": str(self._config.image_store_root / "meta.json"),
                "views": [str(self._config.view_root(view)) for view in self._config.views],
            },
            "rollback_notes": (
                "No files were copied or moved in dry-run mode. If a later real run "
                "is interrupted, remove only the new canonical image-store/view roots; "
                "old sibling roots are never deleted by this factory."
            ),
        }


class Norm1000ViewWriter:
    """Writer for canonical norm1000 integer annotation views."""

    def __init__(self, *, config: CocoViewFactoryConfig) -> None:
        self._config = config
        self._manifest_builder = ViewManifestPayloadBuilder(config)

    def write_view(
        self,
        *,
        source_root: Path,
        view_name: str,
        view_root: Path,
        sample_policy: Mapping[str, Any] | None,
        source_suffix: str = ".jsonl",
        assume_normalized: bool = False,
    ) -> dict[str, Any]:
        """Write a canonical view from source JSONL rows.

        :param source_root: Source artifact root.
        :param view_name: Logical view name recorded in metadata.
        :param view_root: Output root for the view.
        :param sample_policy: Optional sample-policy metadata.
        :param source_suffix: Per-split source file suffix.
        :param assume_normalized: Whether source geometry is already norm1000-like.
        :returns: Summary for the written view.
        """

        # collecting split outputs
        primary_jsonl: dict[str, str] = {}
        summary = _empty_summary()
        for split in self._config.splits:
            source_jsonl = source_root / f"{split}{source_suffix}"
            output_jsonl = view_root / f"{split}.jsonl"
            split_summary = self._write_split(
                source_jsonl=source_jsonl,
                output_jsonl=output_jsonl,
                split=split,
                assume_normalized=assume_normalized,
            )
            primary_jsonl[split] = output_jsonl.name
            _merge_summary(summary, split_summary)

        # writing view metadata
        if not self._config.dry_run:
            metadata = self._manifest_builder.build_view_metadata(
                view_name=view_name,
                primary_jsonl=primary_jsonl,
                summary=summary,
                sample_policy=sample_policy,
            )
            write_view_metadata(view_root / "meta.json", metadata)

        return summary

    def _write_split(
        self,
        *,
        source_jsonl: Path,
        output_jsonl: Path,
        split: str,
        assume_normalized: bool,
    ) -> dict[str, int]:
        """Write one canonical split JSONL."""

        if not source_jsonl.is_file():
            raise FileNotFoundError(f"source JSONL does not exist: {source_jsonl}")
        if self._config.dry_run:
            return _summarize_source_jsonl(source_jsonl)

        if not self._config.dry_run:
            output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        split_summary = _empty_summary()
        with (
            source_jsonl.open("r", encoding="utf-8") as source,
            output_jsonl.open("w", encoding="utf-8") as output,
        ):
            for line in source:
                stripped = line.strip()
                if not stripped:
                    continue
                row = normalize_view_record(
                    json.loads(stripped),
                    split=split,
                    source_image_dir=self._config.source_image_dir,
                    assume_normalized=assume_normalized,
                )
                _write_jsonl_record(output, row)
                _observe_written_row(split_summary, row)
        return split_summary


class LengthBudgetViewBuilder:
    """Builder for length-budget filtered COCO views."""

    def __init__(
        self,
        *,
        config: CocoViewFactoryConfig,
        estimator: LengthEstimator,
        stats_writer: "ViewStatsWriter",
        manifest_builder: "ViewManifestPayloadBuilder",
    ) -> None:
        self._config = config
        self._estimator = estimator
        self._stats_writer = stats_writer
        self._manifest_builder = manifest_builder

    def build(
        self,
        *,
        source_view_root: Path,
        view_name: str,
        max_total_tokens: int,
    ) -> dict[str, Any]:
        """Build a child view by filtering source rows by total token budget."""

        # filtering every configured split
        view_root = self._config.view_root(view_name)
        primary_jsonl: dict[str, str] = {}
        length_stats: dict[str, Mapping[str, str]] = {}
        summary = _empty_summary()
        for split in self._config.splits:
            split_summary, stats_ref = self._build_split(
                source_jsonl=source_view_root / f"{split}.jsonl",
                output_jsonl=view_root / f"{split}.jsonl",
                split=split,
                max_total_tokens=max_total_tokens,
            )
            primary_jsonl[split] = f"{split}.jsonl"
            length_stats[split] = stats_ref
            _merge_summary(summary, split_summary)

        # writing metadata after stats are available
        if not self._config.dry_run:
            metadata = self._manifest_builder.build_view_metadata(
                view_name=view_name,
                primary_jsonl=primary_jsonl,
                summary=summary,
                sample_policy=_length_budget_sample_policy(max_total_tokens),
                length_budget_scope=_length_budget_scope(),
                length_budget_template_id="compact-detection-v1",
                length_stats=length_stats,
            )
            write_view_metadata(view_root / "meta.json", metadata)

        return summary

    def _build_split(
        self,
        *,
        source_jsonl: Path,
        output_jsonl: Path,
        split: str,
        max_total_tokens: int,
    ) -> tuple[dict[str, Any], Mapping[str, str]]:
        """Build one length-filtered split."""

        if not source_jsonl.is_file():
            raise FileNotFoundError(f"source view JSONL does not exist: {source_jsonl}")

        if not self._config.dry_run:
            output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        stats = SplitLengthStats(
            split=split,
            source_jsonl=_safe_artifact_reference_path(
                source_jsonl,
                repo_root=self._config.repo_root,
            ),
        )
        summary = _empty_summary()
        output_handle = None
        try:
            if not self._config.dry_run:
                output_handle = output_jsonl.open("w", encoding="utf-8")
            with source_jsonl.open("r", encoding="utf-8") as source:
                for line in source:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    row = json.loads(stripped)
                    breakdown = self._estimator.measure(row)
                    stats.observe(row, breakdown)
                    if breakdown.total_tokens > max_total_tokens:
                        stats.drop(row, breakdown)
                        continue
                    if output_handle is not None:
                        _write_jsonl_record(output_handle, row)
                    stats.keep(row, breakdown)
                    _observe_written_row(summary, row)
        finally:
            if output_handle is not None:
                output_handle.close()

        stats_ref = self._stats_writer.write_length_stats(
            view_root=output_jsonl.parent,
            split=split,
            stats=stats.as_dict(
                output_jsonl=_safe_artifact_reference_path(
                    output_jsonl,
                    repo_root=self._config.repo_root,
                )
            ),
            dry_run=self._config.dry_run,
        )
        return summary, stats_ref


class LegacyMaxObjectsViewBuilder:
    """Builder for historical max-object membership views."""

    def __init__(
        self,
        *,
        config: CocoViewFactoryConfig,
        writer: Norm1000ViewWriter,
    ) -> None:
        self._config = config
        self._writer = writer

    def build(self, *, view_name: str, max_objects: int) -> dict[str, Any]:
        """Build the legacy membership view without applying a new admission cap."""

        # resolving historical source files
        if self._config.legacy_max_objects_source is None:
            raise ValueError("legacy_max_objects_source is required for max-* views")
        source_suffix = _source_suffix_for_root(self._config.legacy_max_objects_source)

        # copying existing membership into canonical view format
        return self._writer.write_view(
            source_root=self._config.legacy_max_objects_source,
            view_name=view_name,
            view_root=self._config.view_root(view_name),
            sample_policy={
                "type": "max_objects_legacy",
                "max_objects": int(max_objects),
                "membership_source": _safe_artifact_reference_path(
                    self._config.legacy_max_objects_source,
                    repo_root=self._config.repo_root,
                ),
            },
            source_suffix=source_suffix,
            assume_normalized=True,
        )


class AllProxyResearchViewBuilder:
    """Builder for the Phase 1 all-proxy research view."""

    def __init__(
        self,
        *,
        config: CocoViewFactoryConfig,
        estimator: LengthEstimator,
        stats_writer: "ViewStatsWriter",
        manifest_builder: "ViewManifestPayloadBuilder",
    ) -> None:
        self._config = config
        self._estimator = estimator
        self._stats_writer = stats_writer
        self._manifest_builder = manifest_builder

    def build(self, *, view_name: str, max_total_tokens: int) -> dict[str, Any]:
        """Build the all-proxy view and filter after annotation policy."""

        # resolving proxy source files
        if self._config.proxy_source is None:
            raise ValueError("proxy_source is required for all-proxy views")
        source_suffix = _source_suffix_for_root(self._config.proxy_source)
        view_root = self._config.view_root(view_name)

        # filtering proxy-annotated rows
        primary_jsonl: dict[str, str] = {}
        length_stats: dict[str, Mapping[str, str]] = {}
        summary = _empty_summary()
        for split in self._config.splits:
            split_summary, stats_ref = self._build_split(
                source_jsonl=self._config.proxy_source / f"{split}{source_suffix}",
                output_jsonl=view_root / f"{split}.jsonl",
                split=split,
                max_total_tokens=max_total_tokens,
            )
            primary_jsonl[split] = f"{split}.jsonl"
            length_stats[split] = stats_ref
            _merge_summary(summary, split_summary)

        # writing proxy view metadata
        if not self._config.dry_run:
            metadata = self._manifest_builder.build_view_metadata(
                view_name=view_name,
                primary_jsonl=primary_jsonl,
                summary=summary,
                sample_policy={
                    **_length_budget_sample_policy(max_total_tokens),
                    "applied_after_annotation_policy": True,
                },
                length_budget_scope=_length_budget_scope(),
                length_budget_template_id="compact-detection-v1",
                length_stats=length_stats,
                annotation_policy=PROXY_ANNOTATION_POLICY_ALL_PROXY,
                parent_view="coco80/full",
                proxy_policy={
                    "source_artifacts": [
                        {
                            "kind": "legacy_all_proxy_jsonl",
                            "path": _safe_artifact_reference_path(
                                self._config.proxy_source,
                                repo_root=self._config.repo_root,
                            ),
                        }
                    ]
                },
            )
            write_view_metadata(view_root / "meta.json", metadata)

        return summary

    def _build_split(
        self,
        *,
        source_jsonl: Path,
        output_jsonl: Path,
        split: str,
        max_total_tokens: int,
    ) -> tuple[dict[str, Any], Mapping[str, str]]:
        """Build one proxy split."""

        if not source_jsonl.is_file():
            raise FileNotFoundError(f"proxy source JSONL does not exist: {source_jsonl}")

        if not self._config.dry_run:
            output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        stats = SplitLengthStats(
            split=split,
            source_jsonl=_safe_artifact_reference_path(
                source_jsonl,
                repo_root=self._config.repo_root,
            ),
        )
        summary = _empty_summary()
        output_handle = None
        try:
            if not self._config.dry_run:
                output_handle = output_jsonl.open("w", encoding="utf-8")
            with source_jsonl.open("r", encoding="utf-8") as source:
                for line in source:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    row = normalize_view_record(
                        json.loads(stripped),
                        split=split,
                        source_image_dir=self._config.source_image_dir,
                        assume_normalized=True,
                    )
                    row = attach_object_supervision(row)
                    breakdown = self._estimator.measure(row)
                    stats.observe(row, breakdown)
                    if breakdown.total_tokens > max_total_tokens:
                        stats.drop(row, breakdown)
                        continue
                    if output_handle is not None:
                        _write_jsonl_record(output_handle, row)
                    stats.keep(row, breakdown)
                    _observe_written_row(summary, row)
        finally:
            if output_handle is not None:
                output_handle.close()

        stats_ref = self._stats_writer.write_length_stats(
            view_root=output_jsonl.parent,
            split=split,
            stats=stats.as_dict(
                output_jsonl=_safe_artifact_reference_path(
                    output_jsonl,
                    repo_root=self._config.repo_root,
                )
            ),
            dry_run=self._config.dry_run,
        )
        return summary, stats_ref


class DryRunViewPlanner:
    """Planner for annotation views that are not materialized in dry-run mode."""

    def __init__(self, config: CocoViewFactoryConfig) -> None:
        self._config = config

    def plan_view(
        self,
        *,
        source_root: Path,
        view_name: str,
        source_suffix: str = ".jsonl",
        parent_view: str | None = None,
        sample_policy: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a dry-run summary for a view from source JSONL files.

        :param source_root: Existing source root used for planning.
        :param view_name: Logical view that would be materialized.
        :param source_suffix: Per-split source file suffix to validate.
        :param parent_view: Optional logical parent that would be consumed in a real run.
        :param sample_policy: Optional sample policy planned for the view.
        :returns: Dry-run planning summary without writing artifacts.
        """

        # validating source rows and accumulating summary counters
        summary = _empty_summary()
        source_jsonl: dict[str, str] = {}
        planned_primary_jsonl: dict[str, str] = {}
        for split in self._config.splits:
            split_source = source_root / f"{split}{source_suffix}"
            if not split_source.is_file():
                raise FileNotFoundError(
                    f"dry-run source JSONL does not exist: {split_source}"
                )
            source_jsonl[split] = str(split_source)
            planned_primary_jsonl[split] = f"{split}.jsonl"
            _merge_summary(summary, _summarize_source_jsonl(split_source))

        # describing the non-materialized outputs
        return {
            **summary,
            "dry_run": True,
            "materialization": "skipped",
            "source_root": str(source_root),
            "source_jsonl": source_jsonl,
            "planned_view_root": str(self._config.view_root(view_name)),
            "planned_primary_jsonl": planned_primary_jsonl,
            "parent_view": parent_view,
            "sample_policy": dict(sample_policy) if sample_policy is not None else None,
        }


class ViewStatsWriter:
    """Writer for split-level length stats."""

    def write_length_stats(
        self,
        *,
        view_root: Path,
        split: str,
        stats: Mapping[str, Any],
        dry_run: bool = False,
    ) -> Mapping[str, str]:
        """Write a length stats JSON file and return its metadata reference."""

        filename = f"{split}.length_stats.json"
        stats_path = view_root / filename
        if dry_run:
            return {"filename": filename, "sha256": "0" * 64}

        stats_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(stats, indent=2, sort_keys=True) + "\n"
        stats_path.write_text(payload, encoding="utf-8")
        return {"filename": filename, "sha256": _sha256_bytes(payload.encode("utf-8"))}


@dataclass(frozen=True)
class SourceComparisonSource:
    """Source artifact selected for deterministic view comparison.

    :param root: Source artifact root used for split JSONL reads.
    :param suffix: Per-split source suffix.
    :param mode: Comparison mode controlling expected membership deltas.
    :param notes: Deterministic notes explaining intentional source selection.
    """

    root: Path
    suffix: str
    mode: str
    notes: tuple[str, ...] = ()


@dataclass
class SourceComparisonRow:
    """Compact comparable source or generated row descriptor."""

    image_id: Any
    image_refs: tuple[str, ...]
    object_count: int
    proxy_object_count: int


class SourceComparisonWriter:
    """Writer for deterministic source-membership comparison artifacts."""

    def __init__(self, config: CocoViewFactoryConfig) -> None:
        self._config = config

    def write_for_view(self, view_name: str) -> Mapping[str, str]:
        """Write and attach ``source_comparison.json`` for ``view_name``.

        :raises ValueError: If unexpected membership, image, or object drift exists.
        :returns: Metadata reference for the written comparison artifact.
        """

        # resolving comparison roots and target files
        source = self._source_for_view(view_name)
        view_root = self._config.view_root(view_name)
        comparison_path = view_root / "source_comparison.json"

        # comparing every configured split
        split_payloads: dict[str, Any] = {}
        unexpected_deltas: list[dict[str, Any]] = []
        expected_deltas: list[dict[str, Any]] = [
            {
                "type": "coordinate_storage_changed",
                "detail": "generated view stores norm1000 integer coordinates",
            },
            {
                "type": "image_refs_rebased",
                "detail": "image references are compared after normalization to images/...",
            },
        ]
        for note in source.notes:
            expected_deltas.append({"type": "source_selection_note", "detail": note})

        for split in self._config.splits:
            split_payload = self._compare_split(
                split=split,
                source=source,
                view_root=view_root,
            )
            split_payloads[split] = split_payload
            unexpected_deltas.extend(split_payload["unexpected_deltas"])
            expected_deltas.extend(split_payload["expected_intentional_deltas"])

        # writing deterministic payload before failing on drift
        payload = {
            "schema_version": 1,
            "kind": "source_comparison",
            "dataset": "coco",
            "view": view_name,
            "generated_view_path": _safe_artifact_reference_path(
                view_root,
                repo_root=self._config.repo_root,
            ),
            "source_artifact_path": _safe_artifact_reference_path(
                source.root,
                repo_root=self._config.repo_root,
            ),
            "source_suffix": source.suffix,
            "source_mode": source.mode,
            "comparison_policy": {
                "identity_key": "image_id",
                "checks": [
                    "row_counts",
                    "image_ids",
                    "image_refs_normalized_to_images_prefix",
                    "object_counts",
                    (
                        "length_budget_inclusion_exclusion_counts_and_membership_"
                        "when_available"
                    ),
                    "lvis_proxy_object_counts_when_applicable",
                ],
                "allowed_intentional_deltas": [
                    "coordinate_storage_changed",
                    "image_refs_rebased",
                    "verified_length_budget_membership_filter_when_no_legacy_source",
                ],
            },
            "code_version": _code_version(self._config.repo_root),
            "splits": split_payloads,
            "expected_intentional_deltas": _dedupe_dicts(expected_deltas),
            "unexpected_deltas": unexpected_deltas,
        }
        ref = self._write_payload(comparison_path, payload)

        if unexpected_deltas:
            raise ValueError(
                "unexpected source comparison deltas for "
                f"{view_name}: {json.dumps(unexpected_deltas, sort_keys=True)}"
            )

        self._attach_to_metadata(view_root=view_root, ref=ref)
        return ref

    def _source_for_view(self, view_name: str) -> SourceComparisonSource:
        """Return the deterministic comparison source for one view."""

        if view_name == "coco80/full":
            return SourceComparisonSource(
                root=self._config.source_preset,
                suffix=".jsonl",
                mode="exact_membership",
            )

        if view_name == "coco80/len-12000":
            legacy_root = self._config.legacy_length_budget_source
            if legacy_root is not None and legacy_root.exists():
                return SourceComparisonSource(
                    root=legacy_root,
                    suffix=_source_suffix_for_root(legacy_root),
                    mode="exact_membership",
                )
            return SourceComparisonSource(
                root=self._config.source_preset,
                suffix=".jsonl",
                mode="length_budget_subset",
                notes=(
                    "legacy length-budget source unavailable; comparing generated "
                    "view membership as a length-budget subset of source_preset",
                ),
            )

        if view_name == "coco80/max-60":
            if self._config.legacy_max_objects_source is None:
                raise ValueError("legacy_max_objects_source is required for max-* views")
            return SourceComparisonSource(
                root=self._config.legacy_max_objects_source,
                suffix=_source_suffix_for_root(self._config.legacy_max_objects_source),
                mode="exact_membership",
            )

        if view_name == "coco80-lvis-proxy/len-12000":
            if self._config.proxy_source is None:
                raise ValueError("proxy_source is required for all-proxy views")
            return SourceComparisonSource(
                root=self._config.proxy_source,
                suffix=_source_suffix_for_root(self._config.proxy_source),
                mode="exact_membership",
            )

        raise ValueError(f"unsupported Phase 1 view: {view_name}")

    def _compare_split(
        self,
        *,
        split: str,
        source: SourceComparisonSource,
        view_root: Path,
    ) -> dict[str, Any]:
        """Compare one split and return a deterministic payload."""

        # loading comparable row descriptors
        source_jsonl = source.root / f"{split}{source.suffix}"
        generated_jsonl = view_root / f"{split}.jsonl"
        if not source_jsonl.is_file():
            raise FileNotFoundError(f"comparison source JSONL does not exist: {source_jsonl}")
        if not generated_jsonl.is_file():
            raise FileNotFoundError(
                f"comparison generated JSONL does not exist: {generated_jsonl}"
            )

        source_rows = self._load_rows(source_jsonl)
        generated_rows = self._load_rows(generated_jsonl)
        source_by_id, source_duplicates = _rows_by_image_id(source_rows)
        generated_by_id, generated_duplicates = _rows_by_image_id(generated_rows)

        # checking deterministic membership and per-row invariants
        source_ids = set(source_by_id)
        generated_ids = set(generated_by_id)
        missing_ids = sorted(source_ids - generated_ids, key=str)
        extra_ids = sorted(generated_ids - source_ids, key=str)
        unexpected_deltas: list[dict[str, Any]] = []
        expected_deltas: list[dict[str, Any]] = []

        if source_duplicates:
            unexpected_deltas.append(
                {
                    "type": "duplicate_source_image_ids",
                    "split": split,
                    "image_ids": source_duplicates,
                }
            )
        if generated_duplicates:
            unexpected_deltas.append(
                {
                    "type": "duplicate_generated_image_ids",
                    "split": split,
                    "image_ids": generated_duplicates,
                }
            )

        length_stats = _load_length_stats(view_root, split)
        if source.mode == "length_budget_subset":
            expected_deltas.append(
                {
                    "type": "length_budget_membership_filter",
                    "split": split,
                    "excluded_from_generated_count": len(missing_ids),
                }
            )
            if extra_ids:
                unexpected_deltas.append(
                    {
                        "type": "generated_image_ids_not_in_source",
                        "split": split,
                        "image_ids": extra_ids,
                    }
                )
            self._compare_length_stats(
                split=split,
                source_count=len(source_rows),
                generated_count=len(generated_rows),
                excluded_count=len(missing_ids),
                generated_ids=generated_ids,
                missing_ids=set(missing_ids),
                length_stats=length_stats,
                unexpected_deltas=unexpected_deltas,
            )
        else:
            if len(source_rows) != len(generated_rows):
                unexpected_deltas.append(
                    {
                        "type": "row_count_mismatch",
                        "split": split,
                        "source": len(source_rows),
                        "generated": len(generated_rows),
                    }
                )
            if missing_ids:
                unexpected_deltas.append(
                    {
                        "type": "missing_generated_image_ids",
                        "split": split,
                        "image_ids": missing_ids,
                    }
                )
            if extra_ids:
                unexpected_deltas.append(
                    {
                        "type": "extra_generated_image_ids",
                        "split": split,
                        "image_ids": extra_ids,
                    }
                )

        for image_id in sorted(source_ids & generated_ids, key=str):
            source_row = source_by_id[image_id]
            generated_row = generated_by_id[image_id]
            if source_row.image_refs != generated_row.image_refs:
                unexpected_deltas.append(
                    {
                        "type": "image_refs_mismatch",
                        "split": split,
                        "image_id": image_id,
                        "source": list(source_row.image_refs),
                        "generated": list(generated_row.image_refs),
                    }
                )
            if source_row.object_count != generated_row.object_count:
                unexpected_deltas.append(
                    {
                        "type": "object_count_mismatch",
                        "split": split,
                        "image_id": image_id,
                        "source": source_row.object_count,
                        "generated": generated_row.object_count,
                    }
                )
            if source_row.proxy_object_count != generated_row.proxy_object_count:
                unexpected_deltas.append(
                    {
                        "type": "proxy_object_count_mismatch",
                        "split": split,
                        "image_id": image_id,
                        "source": source_row.proxy_object_count,
                        "generated": generated_row.proxy_object_count,
                    }
                )

        return {
            "source_jsonl": _safe_artifact_reference_path(
                source_jsonl,
                repo_root=self._config.repo_root,
            ),
            "generated_jsonl": _safe_artifact_reference_path(
                generated_jsonl,
                repo_root=self._config.repo_root,
            ),
            "source_records": len(source_rows),
            "generated_records": len(generated_rows),
            "source_object_count": sum(row.object_count for row in source_rows),
            "generated_object_count": sum(row.object_count for row in generated_rows),
            "source_proxy_object_count": sum(
                row.proxy_object_count for row in source_rows
            ),
            "generated_proxy_object_count": sum(
                row.proxy_object_count for row in generated_rows
            ),
            "matching_image_id_count": len(source_ids & generated_ids),
            "missing_from_generated_count": len(missing_ids),
            "extra_in_generated_count": len(extra_ids),
            "length_stats": length_stats,
            "expected_intentional_deltas": expected_deltas,
            "unexpected_deltas": unexpected_deltas,
        }

    def _load_rows(self, path: Path) -> list[SourceComparisonRow]:
        """Load comparable descriptors from a JSONL file."""

        rows: list[SourceComparisonRow] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                rows.append(self._row_descriptor(record))
        return rows

    def _row_descriptor(self, record: Mapping[str, Any]) -> SourceComparisonRow:
        """Return the comparison descriptor for one record."""

        objects = [obj for obj in record.get("objects") or [] if isinstance(obj, Mapping)]
        image_refs = _normalize_image_refs(
            _record_image_refs(record),
            source_image_dir=self._config.source_image_dir,
        )
        supervision = record.get("metadata", {}).get("supervision", {})
        object_supervision = (
            supervision.get("object_supervision", {})
            if isinstance(supervision, Mapping)
            else {}
        )
        proxy_candidate_ids = {
            str(obj.get("object_id", index))
            for index, obj in enumerate(objects)
            if _has_lvis_proxy_evidence(obj)
        }
        if isinstance(object_supervision, Mapping):
            proxy_candidate_ids.update(
                str(object_id)
                for object_id, snapshot in object_supervision.items()
                if _supervision_marks_lvis_proxy_candidate(snapshot)
            )

        return SourceComparisonRow(
            image_id=record.get("image_id"),
            image_refs=tuple(image_refs),
            object_count=len(objects),
            proxy_object_count=len(proxy_candidate_ids),
        )

    def _compare_length_stats(
        self,
        *,
        split: str,
        source_count: int,
        generated_count: int,
        excluded_count: int,
        generated_ids: set[Any],
        missing_ids: set[Any],
        length_stats: Mapping[str, Any] | None,
        unexpected_deltas: list[dict[str, Any]],
    ) -> None:
        """Check length-stat inclusion/exclusion counts when available."""

        if length_stats is None:
            unexpected_deltas.append(
                {
                    "type": "length_budget_subset_membership_unverifiable",
                    "split": split,
                    "detail": (
                        "legacy length-budget source is unavailable and "
                        "generated membership cannot be verified without "
                        "split length stats"
                    ),
                }
            )
            return

        stats_seen = int(length_stats.get("records_seen", -1))
        stats_written = int(length_stats.get("records_written", -1))
        stats_dropped = int(length_stats.get("records_dropped", -1))
        if stats_seen != source_count:
            unexpected_deltas.append(
                {
                    "type": "length_stats_records_seen_mismatch",
                    "split": split,
                    "stats": stats_seen,
                    "source": source_count,
                }
            )
        if stats_written != generated_count:
            unexpected_deltas.append(
                {
                    "type": "length_stats_records_written_mismatch",
                    "split": split,
                    "stats": stats_written,
                    "generated": generated_count,
                }
            )
        if stats_dropped != excluded_count:
            unexpected_deltas.append(
                {
                    "type": "length_stats_records_dropped_mismatch",
                    "split": split,
                    "stats": stats_dropped,
                    "excluded": excluded_count,
                }
            )

        kept_ids = _optional_image_id_set(length_stats, field="kept_image_ids")
        if kept_ids is None:
            unexpected_deltas.append(
                {
                    "type": "length_budget_subset_membership_unverifiable",
                    "split": split,
                    "detail": (
                        "legacy length-budget source is unavailable and "
                        "length stats do not contain kept_image_ids"
                    ),
                }
            )
        elif kept_ids != generated_ids:
            unexpected_deltas.append(
                {
                    "type": "length_budget_kept_image_ids_mismatch",
                    "split": split,
                    "stats_only": _sample_sorted_ids(kept_ids - generated_ids),
                    "generated_only": _sample_sorted_ids(generated_ids - kept_ids),
                }
            )

        dropped_ids = _optional_image_id_set(length_stats, field="dropped_image_ids")
        if dropped_ids is not None and dropped_ids != missing_ids:
            unexpected_deltas.append(
                {
                    "type": "length_budget_dropped_image_ids_mismatch",
                    "split": split,
                    "stats_only": _sample_sorted_ids(dropped_ids - missing_ids),
                    "source_missing_only": _sample_sorted_ids(missing_ids - dropped_ids),
                }
            )

    def _write_payload(
        self,
        comparison_path: Path,
        payload: Mapping[str, Any],
    ) -> Mapping[str, str]:
        """Write the deterministic comparison JSON and return its metadata ref."""

        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        comparison_path.write_text(serialized, encoding="utf-8")
        return {
            "filename": comparison_path.name,
            "sha256": _sha256_bytes(serialized.encode("utf-8")),
        }

    def _attach_to_metadata(
        self,
        *,
        view_root: Path,
        ref: Mapping[str, str],
    ) -> None:
        """Attach the comparison reference under ``summary`` in view metadata."""

        meta_path = view_root / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"view metadata does not exist: {meta_path}")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        summary = metadata.setdefault("summary", {})
        if not isinstance(summary, dict):
            raise ValueError(f"view metadata summary is not an object: {meta_path}")
        summary["source_comparison"] = dict(ref)
        write_view_metadata(meta_path, metadata)


class ViewManifestPayloadBuilder:
    """Builder for image-store and annotation-view metadata payloads."""

    def __init__(self, config: CocoViewFactoryConfig) -> None:
        self._config = config

    def build_image_store_metadata(self) -> dict[str, Any]:
        """Return image-store metadata using the Phase 1 contract."""

        return {
            "schema_version": SCHEMA_VERSION_V1,
            "kind": "image_store",
            "dataset": "coco",
            "image_store": "res-1024",
            "image_path_semantics": IMAGE_PATH_SEMANTICS_IMAGE_STORE_RELATIVE,
            "max_pixels": 1024 * 1024,
            "visual_token_budget": 1024,
            "image_factor": 32,
            "image_root": _repo_relative_or_abs(
                self._config.image_store_root,
                repo_root=self._config.repo_root,
            ),
            "splits": list(self._config.splits),
        }

    def build_view_metadata(
        self,
        *,
        view_name: str,
        primary_jsonl: Mapping[str, str],
        summary: Mapping[str, Any],
        sample_policy: Mapping[str, Any] | None = None,
        length_budget_scope: Mapping[str, Any] | None = None,
        length_budget_template_id: str | None = None,
        length_stats: Mapping[str, Mapping[str, str]] | None = None,
        annotation_policy: str | None = None,
        parent_view: str | None = None,
        proxy_policy: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return annotation-view metadata using the Phase 1 contract."""

        metadata: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION_V1,
            "kind": "annotation_view",
            "dataset": "coco",
            "view": view_name,
            "image_store": _repo_relative_or_abs(
                self._config.image_store_root,
                repo_root=self._config.repo_root,
            ),
            "path_anchor": "repo_root",
            "image_path_semantics": IMAGE_PATH_SEMANTICS_IMAGE_STORE_RELATIVE,
            "coordinate_space": COORDINATE_SPACE_NORM1000,
            "coordinate_storage": COORDINATE_STORAGE_INTEGER,
            "coordinate_range": list(COORDINATE_RANGE_NORM1000),
            "coordinate_chart": COORDINATE_CHART_XYXY,
            "assistant_coordinate_rendering": (
                ASSISTANT_COORDINATE_RENDERING_QWEN_COORD_TOKENS
            ),
            "primary_jsonl": dict(primary_jsonl),
            "summary": dict(summary),
        }

        # adding optional view contract fields
        if sample_policy is not None:
            metadata["sample_policy"] = dict(sample_policy)
        if length_budget_scope is not None:
            metadata["length_budget_scope"] = dict(length_budget_scope)
        if length_budget_template_id is not None:
            metadata["length_budget_template_id"] = length_budget_template_id
        if length_stats is not None:
            metadata["length_stats"] = {
                str(split): dict(ref) for split, ref in length_stats.items()
            }
        if annotation_policy is not None:
            metadata["annotation_policy"] = annotation_policy
        if parent_view is not None:
            metadata["parent_view"] = parent_view
        if proxy_policy is not None:
            metadata["proxy_policy"] = dict(proxy_policy)
        return metadata


@dataclass
class SplitLengthStats:
    """Length-filter statistics for one split."""

    split: str
    source_jsonl: str
    records_seen: int = 0
    records_written: int = 0
    records_dropped: int = 0
    objects_seen: int = 0
    objects_written: int = 0
    max_total_tokens_seen: int = 0
    max_total_tokens_written: int = 0
    lengths_written: list[int] | None = None
    kept_image_ids: list[Any] | None = None
    dropped_image_ids: list[Any] | None = None
    length_over_budget_examples: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        """Initialize mutable containers."""

        if self.lengths_written is None:
            self.lengths_written = []
        if self.kept_image_ids is None:
            self.kept_image_ids = []
        if self.dropped_image_ids is None:
            self.dropped_image_ids = []
        if self.length_over_budget_examples is None:
            self.length_over_budget_examples = []

    def observe(
        self,
        record: Mapping[str, Any],
        breakdown: TokenBudgetBreakdown,
    ) -> None:
        """Record one source sample before filtering."""

        self.records_seen += 1
        self.objects_seen += int(breakdown.object_count)
        self.max_total_tokens_seen = max(
            self.max_total_tokens_seen,
            int(breakdown.total_tokens),
        )

    def keep(
        self,
        record: Mapping[str, Any],
        breakdown: TokenBudgetBreakdown,
    ) -> None:
        """Record one kept sample."""

        self.records_written += 1
        self.objects_written += int(breakdown.object_count)
        self.max_total_tokens_written = max(
            self.max_total_tokens_written,
            int(breakdown.total_tokens),
        )
        assert self.lengths_written is not None
        self.lengths_written.append(int(breakdown.total_tokens))
        assert self.kept_image_ids is not None
        self.kept_image_ids.append(record.get("image_id"))

    def drop(
        self,
        record: Mapping[str, Any],
        breakdown: TokenBudgetBreakdown,
    ) -> None:
        """Record one over-budget sample."""

        self.records_dropped += 1
        assert self.dropped_image_ids is not None
        self.dropped_image_ids.append(record.get("image_id"))
        assert self.length_over_budget_examples is not None
        if len(self.length_over_budget_examples) < 20:
            self.length_over_budget_examples.append(
                {
                    "image_id": record.get("image_id"),
                    "file_name": record.get("file_name"),
                    "total_tokens": int(breakdown.total_tokens),
                    "object_count": int(breakdown.object_count),
                }
            )

    def as_dict(self, *, output_jsonl: str) -> dict[str, Any]:
        """Return a JSON-serializable stats payload."""

        assert self.lengths_written is not None
        assert self.kept_image_ids is not None
        assert self.dropped_image_ids is not None
        assert self.length_over_budget_examples is not None
        return {
            "split": self.split,
            "source_jsonl": self.source_jsonl,
            "output_jsonl": output_jsonl,
            "records_seen": self.records_seen,
            "records_written": self.records_written,
            "records_dropped": self.records_dropped,
            "objects_seen": self.objects_seen,
            "objects_written": self.objects_written,
            "max_total_tokens_seen": self.max_total_tokens_seen,
            "max_total_tokens_written": self.max_total_tokens_written,
            "lengths_written": _summarize_ints(self.lengths_written),
            "kept_image_ids": self.kept_image_ids,
            "dropped_image_ids": self.dropped_image_ids,
            "length_over_budget_examples": self.length_over_budget_examples,
        }


def normalize_view_record(
    record: Mapping[str, Any],
    *,
    split: str,
    source_image_dir: Path,
    assume_normalized: bool,
) -> dict[str, Any]:
    """Return a canonical norm1000 integer view record."""

    # normalizing geometry in the norm1000 integer domain
    normalized = convert_record_to_ints(
        copy.deepcopy(dict(record)),
        ("bbox_2d",),
        assume_normalized=assume_normalized,
    )
    normalized = _canonicalize_and_sort_objects_in_place(normalized)

    # normalizing paths, metadata, and object ids
    normalized["images"] = _normalize_image_refs(
        normalized.get("images") or (),
        source_image_dir=source_image_dir,
    )
    _ensure_metadata(normalized, split=split)
    _ensure_object_ids(normalized)
    return normalized


def attach_object_supervision(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a row with object supervision keyed by object id."""

    # copying and locating metadata containers
    out = copy.deepcopy(dict(record))
    metadata = out.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
        out["metadata"] = metadata
    supervision = metadata.setdefault("supervision", {})
    if not isinstance(supervision, dict):
        supervision = {}
        metadata["supervision"] = supervision

    # constructing object-level supervision snapshots
    object_supervision: dict[str, dict[str, Any]] = {}
    for obj in out.get("objects") or []:
        if not isinstance(obj, dict):
            continue
        object_id = obj.get("object_id")
        if not isinstance(object_id, str) or object_id == "":
            continue
        snapshot = _object_supervision_snapshot(obj)
        object_supervision[object_id] = snapshot
    supervision["object_supervision"] = object_supervision
    return out


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the COCO view factory."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--source-preset", type=Path, default=DEFAULT_SOURCE_PRESET)
    parser.add_argument("--image-store-root", type=Path, default=DEFAULT_IMAGE_STORE_ROOT)
    parser.add_argument("--views-root", type=Path, default=DEFAULT_VIEWS_ROOT)
    parser.add_argument(
        "--legacy-length-budget-source",
        type=Path,
        default=DEFAULT_LEGACY_LENGTH_BUDGET_SOURCE,
    )
    parser.add_argument(
        "--legacy-max-objects-source",
        type=Path,
        default=DEFAULT_LEGACY_MAX_OBJECTS_SOURCE,
    )
    parser.add_argument("--proxy-source", type=Path, default=DEFAULT_PROXY_SOURCE)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--views", nargs="+", default=list(DEFAULT_VIEWS))
    parser.add_argument("--max-total-tokens", type=int, default=12000)
    parser.add_argument(
        "--image-store-mode",
        default=None,
        choices=sorted(SUPPORTED_IMAGE_STORE_MODES | {PHASE1_REJECTED_IMAGE_STORE_MODE}),
        help=(
            "Explicit image-store adoption mode. Phase 1 rejects move; use copy, "
            "hardlink, reflink, or reuse-existing. Required unless "
            "--comparison-only is set."
        ),
    )
    parser.add_argument("--reuse-existing-image-store", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-report", type=Path)
    parser.add_argument(
        "--comparison-only",
        action="store_true",
        help=(
            "Write source_comparison.json and patch existing view metadata without "
            "rebuilding image or annotation artifacts."
        ),
    )
    args = parser.parse_args(argv)
    if args.image_store_mode == PHASE1_REJECTED_IMAGE_STORE_MODE:
        parser.error("--image-store-mode move is unavailable in Phase 1")
    if args.image_store_mode is None:
        if args.comparison_only:
            args.image_store_mode = "reuse-existing"
        else:
            parser.error("--image-store-mode is required unless --comparison-only is set")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """Run the COCO view factory CLI."""

    args = parse_args(argv)
    config = CocoViewFactoryConfig(
        repo_root=REPO_ROOT,
        source_preset=_resolve_repo_path(args.source_preset),
        image_store_root=_resolve_repo_path(args.image_store_root),
        views_root=_resolve_repo_path(args.views_root),
        legacy_length_budget_source=_resolve_repo_path(args.legacy_length_budget_source)
        if args.legacy_length_budget_source is not None
        else None,
        legacy_max_objects_source=_resolve_repo_path(args.legacy_max_objects_source)
        if args.legacy_max_objects_source is not None
        else None,
        proxy_source=_resolve_repo_path(args.proxy_source)
        if args.proxy_source is not None
        else None,
        splits=tuple(str(split) for split in args.splits),
        views=tuple(str(view) for view in args.views),
        max_total_tokens=int(args.max_total_tokens),
        image_store_mode=str(args.image_store_mode),
        reuse_existing_image_store=bool(args.reuse_existing_image_store),
        dry_run=bool(args.dry_run),
        dry_run_report=args.dry_run_report,
        config_path=_resolve_repo_path(args.config),
        model_path=_resolve_repo_path(args.model_path),
    )

    # optionally attaching comparison artifacts to already materialized views
    if bool(args.comparison_only):
        comparison_writer = SourceComparisonWriter(config)
        summary = {
            view: {"source_comparison": dict(comparison_writer.write_for_view(view))}
            for view in config.views
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    # preparing image store before annotation views
    summary: dict[str, Any] = {"image_store": ImageStoreAdopter(config).prepare()}
    writer = Norm1000ViewWriter(config=config)
    stats_writer = ViewStatsWriter()
    manifest_builder = ViewManifestPayloadBuilder(config)
    dry_run_planner = DryRunViewPlanner(config)
    comparison_writer = SourceComparisonWriter(config)

    # loading real estimator only when requested views need it
    estimator: LengthEstimator | None = None
    if not config.dry_run and any("len-" in view for view in config.views):
        assert config.config_path is not None
        assert config.model_path is not None
        estimator = CompactFullTokenBudgetEstimator.from_config(
            config.config_path,
            config.model_path,
        )

    # building requested views
    for view in config.views:
        if view == "coco80/full":
            summary[view] = writer.write_view(
                source_root=config.source_preset,
                view_name=view,
                view_root=config.view_root(view),
                sample_policy=None,
            )
        elif view == "coco80/len-12000":
            if config.dry_run:
                summary[view] = dry_run_planner.plan_view(
                    source_root=config.source_preset,
                    view_name=view,
                    source_suffix=".jsonl",
                    parent_view="coco80/full",
                    sample_policy=_length_budget_sample_policy(config.max_total_tokens),
                )
                continue
            if estimator is None:
                raise RuntimeError("length estimator was not initialized")
            summary[view] = LengthBudgetViewBuilder(
                config=config,
                estimator=estimator,
                stats_writer=stats_writer,
                manifest_builder=manifest_builder,
            ).build(
                source_view_root=config.view_root("coco80/full"),
                view_name=view,
                max_total_tokens=config.max_total_tokens,
            )
        elif view == "coco80/max-60":
            summary[view] = LegacyMaxObjectsViewBuilder(
                config=config,
                writer=writer,
            ).build(view_name=view, max_objects=60)
        elif view == "coco80-lvis-proxy/len-12000":
            if config.dry_run:
                if config.proxy_source is None:
                    raise ValueError("proxy_source is required for all-proxy views")
                summary[view] = dry_run_planner.plan_view(
                    source_root=config.proxy_source,
                    view_name=view,
                    source_suffix=_source_suffix_for_root(config.proxy_source),
                    parent_view="coco80/full",
                    sample_policy={
                        **_length_budget_sample_policy(config.max_total_tokens),
                        "applied_after_annotation_policy": True,
                    },
                )
                continue
            if estimator is None:
                raise RuntimeError("length estimator was not initialized")
            summary[view] = AllProxyResearchViewBuilder(
                config=config,
                estimator=estimator,
                stats_writer=stats_writer,
                manifest_builder=manifest_builder,
            ).build(view_name=view, max_total_tokens=config.max_total_tokens)
        else:
            raise ValueError(f"unsupported Phase 1 view: {view}")

        if not config.dry_run:
            summary[view]["source_comparison"] = dict(
                comparison_writer.write_for_view(view)
            )

    print(json.dumps(summary, indent=2, sort_keys=True))


def _normalize_image_refs(
    images: Iterable[Any],
    *,
    source_image_dir: Path,
) -> list[str]:
    """Return canonical image-store-relative references."""

    refs: list[str] = []
    for image in images:
        raw = Path(str(image))
        if raw.is_absolute():
            refs.append(_image_ref_from_absolute_path(raw))
            continue

        if raw.parts and raw.parts[0] == "images":
            refs.append(raw.as_posix())
            continue

        source_candidate = (source_image_dir / raw).resolve(strict=False)
        refs.append(_image_ref_from_absolute_path(source_candidate))

    if not refs:
        raise ValueError("record must include at least one image reference")
    return refs


def _record_image_refs(record: Mapping[str, Any]) -> Iterable[Any]:
    """Return image refs from ``images`` with ``file_name`` as a legacy fallback."""

    images = record.get("images") or ()
    if images:
        return images
    file_name = record.get("file_name")
    if file_name is None or file_name == "":
        return ()
    return (file_name,)


def _image_ref_from_absolute_path(path: Path) -> str:
    """Return ``images/...`` from an absolute path containing an images segment."""

    parts = path.parts
    if "images" not in parts:
        raise ValueError(f"cannot derive image-store-relative ref from {path}")
    image_index = len(parts) - 1 - parts[::-1].index("images")
    return Path(*parts[image_index:]).as_posix()


def _ensure_metadata(record: dict[str, Any], *, split: str) -> None:
    """Ensure metadata contains minimal current detection fields."""

    metadata = record.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
        record["metadata"] = metadata
    metadata.setdefault("source", "coco")
    metadata.setdefault("split", split)


def _ensure_object_ids(record: dict[str, Any]) -> None:
    """Ensure every object has a stable object id."""

    image_key = record.get("image_id") or record.get("file_name") or "image"
    for index, obj in enumerate(record.get("objects") or []):
        if not isinstance(obj, dict):
            continue
        object_id = obj.get("object_id")
        if not isinstance(object_id, str) or object_id == "":
            obj["object_id"] = f"{image_key}:{index}"


def _object_supervision_snapshot(obj: Mapping[str, Any]) -> dict[str, Any]:
    """Return the supervision sidecar snapshot for one rendered object."""

    source_role = obj.get("source_role") or obj.get("role")
    if not isinstance(source_role, str) or source_role == "":
        source_role = (
            "lvis_proxy_candidate"
            if _has_lvis_proxy_evidence(obj)
            else "coco_ground_truth"
        )

    snapshot: dict[str, Any] = {"source_role": source_role}
    for key in (
        "relation",
        "source",
        "proxy_source",
        "category_id",
        "category_name",
        "lvis_ann_id",
        "lvis_category_id",
        "lvis_category_name",
        "coco_category_id",
        "coco_category_name",
        "coordinate_weight",
        "regression_weight",
        "hard_bbox_supervision",
    ):
        if key in obj:
            snapshot[key] = copy.deepcopy(obj[key])

    # defaulting direct bbox supervision off for inferred proxy candidates
    if source_role == "lvis_proxy_candidate":
        snapshot.setdefault("coordinate_weight", 0.0)
        snapshot.setdefault("regression_weight", 0.0)
        snapshot.setdefault("hard_bbox_supervision", False)
    return snapshot


def _has_lvis_proxy_evidence(obj: Mapping[str, Any]) -> bool:
    """Return whether an object carries robust LVIS proxy evidence."""

    # honoring established proxy markers
    if obj.get("is_proxy") is True:
        return True
    if obj.get("source_role") == "lvis_proxy_candidate":
        return True
    if obj.get("role") == "lvis_proxy_candidate":
        return True

    # detecting real-source LVIS proxy fields
    if obj.get("proxy_source") == "lvis":
        return True
    if obj.get("source") == "lvis":
        return True
    for key in ("lvis_ann_id", "lvis_category_id", "lvis_category_name"):
        value = obj.get(key)
        if value is not None and value != "":
            return True
    return False


def _supervision_marks_lvis_proxy_candidate(snapshot: Any) -> bool:
    """Return whether a supervision snapshot marks an LVIS proxy candidate."""

    if not isinstance(snapshot, Mapping):
        return False
    if snapshot.get("source_role") == "lvis_proxy_candidate":
        return True
    return _has_lvis_proxy_evidence(snapshot)


def _source_suffix_for_root(source_root: Path) -> str:
    """Return the preferred normalized source suffix for a legacy root."""

    for split in ("train", "val"):
        if (source_root / f"{split}.norm.jsonl").is_file():
            return ".norm.jsonl"
        if (source_root / f"{split}.coord.jsonl").is_file():
            return ".coord.jsonl"
    raise FileNotFoundError(
        f"source root does not contain *.norm.jsonl or *.coord.jsonl: {source_root}"
    )


def _rows_by_image_id(
    rows: Sequence[SourceComparisonRow],
) -> tuple[dict[Any, SourceComparisonRow], list[Any]]:
    """Return rows keyed by image id plus any duplicate keys."""

    by_id: dict[Any, SourceComparisonRow] = {}
    duplicates: list[Any] = []
    for row in rows:
        if row.image_id in by_id:
            duplicates.append(row.image_id)
            continue
        by_id[row.image_id] = row
    return by_id, sorted(duplicates, key=str)


def _load_length_stats(view_root: Path, split: str) -> Mapping[str, Any] | None:
    """Load split length stats when present."""

    stats_path = view_root / f"{split}.length_stats.json"
    if not stats_path.is_file():
        return None
    return json.loads(stats_path.read_text(encoding="utf-8"))


def _optional_image_id_set(
    length_stats: Mapping[str, Any],
    *,
    field: str,
) -> set[Any] | None:
    """Return an optional image-id set from split length stats."""

    if field not in length_stats:
        return None

    values = length_stats[field]
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"length_stats.{field} must be a list when present")

    return set(values)


def _sample_sorted_ids(values: Iterable[Any], *, limit: int = 20) -> list[Any]:
    """Return a bounded deterministic sample of image ids."""

    return sorted(values, key=str)[:limit]


def _code_version(repo_root: Path) -> Mapping[str, Any]:
    """Return cheap deterministic code-version metadata."""

    git_head: str | None = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        git_head = result.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        git_head = None

    return {
        "script": "public_data/scripts/build_coco_views.py",
        "git_head": git_head,
    }


def _dedupe_dicts(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return dictionaries with deterministic duplicate removal."""

    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in items:
        key = json.dumps(item, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(dict(item))
    return deduped


def _length_budget_sample_policy(max_total_tokens: int) -> dict[str, Any]:
    """Return the standard length-budget sample policy."""

    return {
        "type": "length_budget",
        "max_total_tokens": int(max_total_tokens),
        "budget_includes": [
            "image_patch_tokens",
            "system_prompt_tokens",
            "user_prompt_tokens",
            "assistant_response_tokens",
        ],
    }


def _length_budget_scope() -> dict[str, Any]:
    """Return the standard rendered-family length scope."""

    return {
        "rendered_families": ["objects"],
        "excluded_sidecars": ["metadata.supervision.support_objects"],
    }


def _empty_summary() -> dict[str, int]:
    """Return an empty view summary."""

    return {
        "records": 0,
        "rendered_object_count": 0,
        "object_supervision_count": 0,
        "rendered_proxy_candidate_count": 0,
        "support_sidecar_count": 0,
    }


def _merge_summary(target: dict[str, int], source: Mapping[str, Any]) -> None:
    """Merge numeric summary counters."""

    for key, value in source.items():
        if isinstance(value, int):
            target[key] = int(target.get(key, 0)) + int(value)


def _observe_written_row(summary: dict[str, int], row: Mapping[str, Any]) -> None:
    """Update summary counters from one written row."""

    objects = [obj for obj in row.get("objects") or [] if isinstance(obj, Mapping)]
    summary["records"] += 1
    summary["rendered_object_count"] += len(objects)

    supervision = row.get("metadata", {}).get("supervision", {})
    object_supervision = (
        supervision.get("object_supervision", {})
        if isinstance(supervision, Mapping)
        else {}
    )
    if isinstance(object_supervision, Mapping):
        summary["object_supervision_count"] += len(object_supervision)
    proxy_candidate_ids = {
        str(obj.get("object_id"))
        for obj in objects
        if isinstance(obj.get("object_id"), str) and _has_lvis_proxy_evidence(obj)
    }
    if isinstance(object_supervision, Mapping):
        proxy_candidate_ids.update(
            str(object_id)
            for object_id, snapshot in object_supervision.items()
            if isinstance(object_id, str)
            and _supervision_marks_lvis_proxy_candidate(snapshot)
        )
    summary["rendered_proxy_candidate_count"] += len(proxy_candidate_ids)


def _summarize_source_jsonl(source_jsonl: Path) -> dict[str, int]:
    """Return summary counters for dry-run source rows."""

    summary = _empty_summary()
    with source_jsonl.open("r", encoding="utf-8") as source:
        for line in source:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            _observe_written_row(summary, row)
    return summary


def _summarize_ints(values: Sequence[int]) -> dict[str, Any]:
    """Return a compact integer distribution summary."""

    if not values:
        return {"count": 0}
    ordered = sorted(int(value) for value in values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": mean(ordered),
        "median": median(ordered),
    }


def _list_image_files(root: Path) -> list[Path]:
    """Return sorted image-like files under ``root``."""

    if not root.exists():
        return []
    if not root.is_dir():
        raise NotADirectoryError(f"image root is not a directory: {root}")
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _write_jsonl_record(handle: Any, record: Mapping[str, Any]) -> None:
    """Write one compact JSONL record."""

    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256_bytes(payload: bytes) -> str:
    """Return the SHA256 hash for ``payload``."""

    return hashlib.sha256(payload).hexdigest()


def _repo_relative_or_abs(path: Path, *, repo_root: Path) -> str:
    """Return a repo-relative POSIX path when possible."""

    resolved_path = path.resolve(strict=False)
    resolved_root = repo_root.resolve(strict=False)
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError:
        return str(path)


def _safe_artifact_reference_path(path: Path, *, repo_root: Path) -> str:
    """Return a safe logical artifact path for provenance metadata."""

    relative_or_abs = _repo_relative_or_abs(path, repo_root=repo_root)
    if not PurePosixPath(relative_or_abs).is_absolute():
        return relative_or_abs

    parts = path.parts
    if "public_data" in parts:
        public_data_index = parts.index("public_data")
        return Path(*parts[public_data_index:]).as_posix()

    return relative_or_abs


def _resolve_repo_path(path: Path) -> Path:
    """Resolve a CLI path relative to the repository root."""

    if path.is_absolute():
        return path
    return REPO_ROOT / path


if __name__ == "__main__":
    main()
