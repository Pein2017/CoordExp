#!/usr/bin/env python3
"""Build Phase 1 canonical COCO public-data annotation views."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
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
DEFAULT_LEGACY_MAX_OBJECTS_SOURCE = Path("public_data/coco/rescale_32_1024_bbox_max60")
DEFAULT_PROXY_SOURCE = Path("public_data/coco/rescale_32_1024_bbox_lvis_proxy_len12000")
DEFAULT_VIEWS = (
    "coco80/full",
    "coco80/len-12000",
    "coco80/max-60",
    "coco80-lvis-proxy/len-12000",
)
SUPPORTED_IMAGE_STORE_MODES = frozenset({"copy", "reflink", "reuse-existing"})
PHASE1_REJECTED_IMAGE_STORE_MODE = "move"
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})


@dataclass(frozen=True)
class CocoViewFactoryConfig:
    """Configuration for the canonical COCO view factory.

    :param repo_root: Repository root used for repo-relative metadata paths.
    :param source_preset: Existing COCO 1024 pixel-space source preset root.
    :param image_store_root: Canonical shared image-store root.
    :param views_root: Canonical COCO views root.
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
        if self._config.image_store_mode in {"copy", "reflink"}:
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
        """Copy or reflink the source image tree into the canonical store."""

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
        stats = SplitLengthStats(split=split, source_jsonl=str(source_jsonl))
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
            stats=stats.as_dict(output_jsonl=str(output_jsonl)),
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
                "membership_source": str(self._config.legacy_max_objects_source),
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
                            "path": _repo_relative_or_abs(
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
        stats = SplitLengthStats(split=split, source_jsonl=str(source_jsonl))
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
            stats=stats.as_dict(output_jsonl=str(output_jsonl)),
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
            "image_factor": 28,
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
    length_over_budget_examples: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        """Initialize mutable containers."""

        if self.lengths_written is None:
            self.lengths_written = []
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

    def drop(
        self,
        record: Mapping[str, Any],
        breakdown: TokenBudgetBreakdown,
    ) -> None:
        """Record one over-budget sample."""

        self.records_dropped += 1
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
        required=True,
        choices=sorted(SUPPORTED_IMAGE_STORE_MODES | {PHASE1_REJECTED_IMAGE_STORE_MODE}),
        help=(
            "Explicit image-store adoption mode. Phase 1 rejects move; use copy, "
            "reflink, or reuse-existing."
        ),
    )
    parser.add_argument("--reuse-existing-image-store", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-report", type=Path)
    args = parser.parse_args(argv)
    if args.image_store_mode == PHASE1_REJECTED_IMAGE_STORE_MODE:
        parser.error("--image-store-mode move is unavailable in Phase 1")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """Run the COCO view factory CLI."""

    args = parse_args(argv)
    config = CocoViewFactoryConfig(
        repo_root=REPO_ROOT,
        source_preset=_resolve_repo_path(args.source_preset),
        image_store_root=_resolve_repo_path(args.image_store_root),
        views_root=_resolve_repo_path(args.views_root),
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

    # preparing image store before annotation views
    summary: dict[str, Any] = {"image_store": ImageStoreAdopter(config).prepare()}
    writer = Norm1000ViewWriter(config=config)
    stats_writer = ViewStatsWriter()
    manifest_builder = ViewManifestPayloadBuilder(config)
    dry_run_planner = DryRunViewPlanner(config)

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

    source_role = obj.get("source_role") or obj.get("role") or obj.get("source")
    if not isinstance(source_role, str) or source_role == "":
        source_role = "lvis_proxy_candidate" if obj.get("is_proxy") else "coco_ground_truth"

    snapshot: dict[str, Any] = {"source_role": source_role}
    for key in (
        "relation",
        "source",
        "category_id",
        "lvis_category_id",
        "coco_category_id",
        "coordinate_weight",
        "regression_weight",
        "hard_bbox_supervision",
    ):
        if key in obj:
            snapshot[key] = copy.deepcopy(obj[key])
    return snapshot


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
    summary["rendered_proxy_candidate_count"] += sum(
        1
        for obj in objects
        if obj.get("source_role") == "lvis_proxy_candidate"
        or obj.get("is_proxy") is True
    )


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


def _resolve_repo_path(path: Path) -> Path:
    """Resolve a CLI path relative to the repository root."""

    if path.is_absolute():
        return path
    return REPO_ROOT / path


if __name__ == "__main__":
    main()
