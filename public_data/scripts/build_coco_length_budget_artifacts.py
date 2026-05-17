#!/usr/bin/env python3
"""Build COCO compact-full datasets filtered by total token budget.

The factory derives COCO 1024 artifacts from the existing resized pixel JSONL,
then filters whole image records by the same lightweight compact-full token
accounting used in analysis:

``text/chat tokens - image placeholders + post-merge image tokens``.

The optional LVIS-proxy branch augments the length-filtered COCO coord JSONL
with LVIS-derived proxy objects, then applies the same final length filter to
the augmented rows.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, MutableMapping, Protocol, Sequence

from transformers import AutoProcessor, AutoTokenizer

from public_data.converters.sorting import sort_objects_tlbr
from public_data.scripts.convert_to_coord_tokens import (
    _canonicalize_and_sort_objects_in_place,
    convert_record_to_ints,
    convert_record_to_tokens,
)
from src.analysis.coco_lvis_missing_objects import (
    AnalysisConfig,
    ProxyAugmentConfig,
    export_augmented_coco_with_lvis_proxies,
    run_coco_lvis_projection_analysis,
)
from src.common.detection_chat import build_detection_chat_messages
from src.common.detection_sequence import render_compact_detection_sequence
from src.config.loader import ConfigLoader
from src.config.schema import LatestDetectionTrainingConfig
from src.coord_tokens.codec import int_to_token, token_to_int
from src.detection.runtime import resolve_latest_detection_prompts


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(
    "configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2.yaml"
)
DEFAULT_MODEL = Path("model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp")
DEFAULT_SOURCE_PRESET = Path("public_data/coco/rescale_32_1024_bbox")
DEFAULT_COCO_OUTPUT = Path("public_data/coco/rescale_32_1024_bbox_len12000")
DEFAULT_PROXY_OUTPUT = Path("public_data/coco/rescale_32_1024_bbox_lvis_proxy_len12000")
DEFAULT_MAPPING_CSV = Path(
    "openspec/changes/add-lvis-coco-proxy-supervision/artifacts/"
    "determined_proxy_mappings_val2017.csv"
)
DEFAULT_PROJECTION_ROOT = Path("temp/coco_lvis_projection_length_budget")
DEFAULT_OLD_MAX60_DIR = Path("public_data/coco/rescale_32_1024_bbox_max60")
DEFAULT_OLD_PROXY_DIR = Path("public_data/coco/rescale_32_1024_bbox_max60_lvis_proxy")


class LengthEstimator(Protocol):
    """Object capable of measuring compact-full token budget components."""

    def measure(self, record: Mapping[str, Any]) -> "TokenBudgetBreakdown":
        """Return total-token accounting for ``record``."""


@dataclass(frozen=True)
class TokenBudgetBreakdown:
    """Token accounting for one model-facing compact-full sample."""

    total_tokens: int
    text_tokens_without_image_placeholders: int
    image_patch_tokens: int
    image_placeholders: int
    assistant_tokens: int
    object_count: int


@dataclass
class SplitBuildStats:
    """Per-split filter and length summary."""

    split: str
    source_jsonl: str
    output_jsonl: str
    records_seen: int = 0
    records_written: int = 0
    records_dropped: int = 0
    objects_seen: int = 0
    objects_written: int = 0
    dense_records_seen: int = 0
    dense_records_written: int = 0
    max_total_tokens_seen: int = 0
    max_total_tokens_written: int = 0
    lengths_written: list[int] = field(default_factory=list)
    length_over_budget_examples: list[dict[str, Any]] = field(default_factory=list)
    dense_recovered_examples: list[dict[str, Any]] = field(default_factory=list)

    def observe(self, record: Mapping[str, Any], breakdown: TokenBudgetBreakdown) -> None:
        """Record one source sample before the keep/drop decision."""

        self.records_seen += 1
        self.objects_seen += int(breakdown.object_count)
        self.max_total_tokens_seen = max(
            self.max_total_tokens_seen, int(breakdown.total_tokens)
        )
        if breakdown.object_count > 60:
            self.dense_records_seen += 1

    def keep(self, record: Mapping[str, Any], breakdown: TokenBudgetBreakdown) -> None:
        """Record one sample that passed the length budget."""

        self.records_written += 1
        self.objects_written += int(breakdown.object_count)
        self.lengths_written.append(int(breakdown.total_tokens))
        self.max_total_tokens_written = max(
            self.max_total_tokens_written, int(breakdown.total_tokens)
        )
        if breakdown.object_count > 60:
            self.dense_records_written += 1
            if len(self.dense_recovered_examples) < 20:
                self.dense_recovered_examples.append(
                    _record_example(record, breakdown=breakdown)
                )

    def drop(self, record: Mapping[str, Any], breakdown: TokenBudgetBreakdown) -> None:
        """Record one sample that exceeded the length budget."""

        self.records_dropped += 1
        if len(self.length_over_budget_examples) < 20:
            self.length_over_budget_examples.append(
                _record_example(record, breakdown=breakdown)
            )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable stats payload."""

        lengths = sorted(self.lengths_written)
        return {
            "split": self.split,
            "source_jsonl": self.source_jsonl,
            "output_jsonl": self.output_jsonl,
            "records_seen": self.records_seen,
            "records_written": self.records_written,
            "records_dropped": self.records_dropped,
            "objects_seen": self.objects_seen,
            "objects_written": self.objects_written,
            "dense_records_seen_object_count_gt_60": self.dense_records_seen,
            "dense_records_written_object_count_gt_60": self.dense_records_written,
            "max_total_tokens_seen": self.max_total_tokens_seen,
            "max_total_tokens_written": self.max_total_tokens_written,
            "lengths_written": _summarize_ints(lengths),
            "length_over_budget_examples": self.length_over_budget_examples,
            "dense_recovered_examples": self.dense_recovered_examples,
        }


@dataclass(frozen=True)
class BuildPaths:
    """Path set for one prepared artifact root."""

    root: Path
    split: str

    @property
    def jsonl(self) -> Path:
        return self.root / f"{self.split}.jsonl"

    @property
    def norm(self) -> Path:
        return self.root / f"{self.split}.norm.jsonl"

    @property
    def coord(self) -> Path:
        return self.root / f"{self.split}.coord.jsonl"

    @property
    def stats(self) -> Path:
        return self.root / f"{self.split}.length_budget_stats.json"


@dataclass(frozen=True)
class FactoryConfig:
    """Top-level build options for COCO length-budget artifacts."""

    config_path: Path
    model_path: Path
    source_preset: Path
    coco_output: Path
    proxy_output: Path
    projection_root: Path
    mapping_csv: Path
    max_total_tokens: int
    splits: tuple[str, ...]
    build_lvis_proxy: bool
    build_projections_if_missing: bool
    force: bool
    old_max60_dir: Path | None
    old_proxy_dir: Path | None


class CompactFullTokenBudgetEstimator:
    """Compact-full length estimator backed by the production tokenizer."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        processor: Any,
        system_prompt: str,
        user_prompt: str,
    ) -> None:
        self._tokenizer = tokenizer
        self._system_prompt = system_prompt
        self._user_prompt = user_prompt
        self._image_token_id = int(tokenizer.convert_tokens_to_ids("<|image_pad|>"))

        image_processor = processor.image_processor
        patch_size = int(getattr(image_processor, "patch_size", 16))
        merge_size = int(getattr(image_processor, "merge_size", 2))
        self._vision_stride = int(patch_size * merge_size)

    @classmethod
    def from_config(cls, config_path: Path, model_path: Path) -> "CompactFullTokenBudgetEstimator":
        """Build an estimator from the latest compact-full training config."""

        raw_cfg = ConfigLoader.load_yaml_with_extends(str(config_path))
        latest_cfg = LatestDetectionTrainingConfig.from_mapping(raw_cfg)
        system_prompt, user_prompt = resolve_latest_detection_prompts(latest_cfg)

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            use_fast=False,
            local_files_only=True,
        )
        processor = AutoProcessor.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            local_files_only=True,
        )
        return cls(
            tokenizer=tokenizer,
            processor=processor,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    def measure(self, record: Mapping[str, Any]) -> TokenBudgetBreakdown:
        """Return total compact-full tokens for one coord-token record."""

        assistant_text = render_compact_detection_sequence(
            {"objects": _model_facing_objects(record)},
            detection_sequence_format="compact_full",
        )
        messages = build_detection_chat_messages(
            system_prompt=self._system_prompt,
            user_prompt=self._user_prompt,
            images=tuple(str(image) for image in (record.get("images") or ())),
            assistant_text=assistant_text,
        )

        chat_ids = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors=None,
        )
        input_ids = [int(token_id) for token_id in chat_ids]
        image_placeholders = int(input_ids.count(self._image_token_id))
        text_without_placeholders = int(len(input_ids) - image_placeholders)
        image_tokens = self._image_patch_tokens(record)
        assistant_token_count = len(
            self._tokenizer(assistant_text, add_special_tokens=False)["input_ids"]
        )

        return TokenBudgetBreakdown(
            total_tokens=int(text_without_placeholders + image_tokens),
            text_tokens_without_image_placeholders=text_without_placeholders,
            image_patch_tokens=int(image_tokens),
            image_placeholders=image_placeholders,
            assistant_tokens=int(assistant_token_count),
            object_count=len(record.get("objects") or []),
        )

    def _image_patch_tokens(self, record: Mapping[str, Any]) -> int:
        """Return post-merge Qwen3-VL visual tokens for all row images."""

        width = int(record.get("width") or 1)
        height = int(record.get("height") or 1)
        image_count = max(1, len(record.get("images") or []))
        per_image = math.ceil(height / self._vision_stride) * math.ceil(
            width / self._vision_stride
        )
        return int(image_count * per_image)


class CoordTripletConverter:
    """Converter from pixel COCO records into aligned pixel/norm/coord triplets."""

    def convert(self, record: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Return ``(pixel, norm1000, coord-token)`` records."""

        pixel_record = copy.deepcopy(dict(record))
        if isinstance(pixel_record.get("objects"), list):
            pixel_record["objects"] = sort_objects_tlbr(pixel_record["objects"])

        norm_record = convert_record_to_ints(
            copy.deepcopy(dict(record)),
            ("bbox_2d",),
            assume_normalized=False,
        )
        norm_record = _canonicalize_and_sort_objects_in_place(norm_record)

        coord_record = convert_record_to_tokens(copy.deepcopy(norm_record), ("bbox_2d",))
        return pixel_record, norm_record, coord_record


class CocoLengthBudgetBuilder:
    """Builder for the base COCO compact-full length-budget artifact."""

    def __init__(
        self,
        *,
        estimator: LengthEstimator,
        converter: CoordTripletConverter,
        max_total_tokens: int,
    ) -> None:
        self._estimator = estimator
        self._converter = converter
        self._max_total_tokens = int(max_total_tokens)

    def build_split(
        self,
        *,
        source_jsonl: Path,
        shared_image_root: Path,
        output_root: Path,
        split: str,
    ) -> SplitBuildStats:
        """Build one COCO split and return filter stats."""

        paths = BuildPaths(output_root, split)
        stats = SplitBuildStats(
            split=split,
            source_jsonl=str(source_jsonl),
            output_jsonl=str(paths.coord),
        )
        _ensure_parent(paths.coord)
        _ensure_parent(paths.norm)
        _ensure_parent(paths.jsonl)

        with (
            source_jsonl.open("r", encoding="utf-8") as src,
            paths.jsonl.open("w", encoding="utf-8") as pixel_out,
            paths.norm.open("w", encoding="utf-8") as norm_out,
            paths.coord.open("w", encoding="utf-8") as coord_out,
        ):
            for line in src:
                stripped = line.strip()
                if not stripped:
                    continue
                pixel_record, norm_record, coord_record = self._converter.convert(
                    json.loads(stripped)
                )
                pixel_record = _with_shared_image_refs(
                    record=pixel_record,
                    source_image_root=shared_image_root,
                    output_root=output_root,
                )
                norm_record = _with_shared_image_refs(
                    record=norm_record,
                    source_image_root=shared_image_root,
                    output_root=output_root,
                )
                coord_record = _with_shared_image_refs(
                    record=coord_record,
                    source_image_root=shared_image_root,
                    output_root=output_root,
                )
                breakdown = self._estimator.measure(coord_record)
                stats.observe(coord_record, breakdown)
                if breakdown.total_tokens > self._max_total_tokens:
                    stats.drop(coord_record, breakdown)
                    continue

                _write_jsonl_record(pixel_out, pixel_record)
                _write_jsonl_record(norm_out, norm_record)
                _write_jsonl_record(coord_out, coord_record)
                stats.keep(coord_record, breakdown)

        paths.stats.write_text(
            json.dumps(stats.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return stats


class ProxyLengthBudgetBuilder:
    """Builder for LVIS-proxy coord/norm artifacts under the same token budget."""

    def __init__(
        self,
        *,
        estimator: LengthEstimator,
        max_total_tokens: int,
    ) -> None:
        self._estimator = estimator
        self._max_total_tokens = int(max_total_tokens)

    def build_split(
        self,
        *,
        base_coord_jsonl: Path,
        output_root: Path,
        projection_dir: Path,
        mapping_csv: Path,
        raw_coco_annotation: Path,
        split: str,
    ) -> SplitBuildStats:
        """Build one LVIS-proxy split and return final length-filter stats."""

        paths = BuildPaths(output_root, split)
        candidate_coord = output_root / f"{split}.candidate.coord.jsonl"
        candidate_summary = output_root / f"{split}.candidate.proxy_summary.json"
        _ensure_parent(candidate_coord)

        result = export_augmented_coco_with_lvis_proxies(
            base_jsonl_path=base_coord_jsonl,
            projection_dir=projection_dir,
            determined_mapping_csv_path=mapping_csv,
            output_jsonl_path=candidate_coord,
            raw_coco_annotation_paths=(raw_coco_annotation,),
            config=ProxyAugmentConfig(),
        )
        candidate_summary.write_text(
            json.dumps(result.summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        stats = self._filter_candidate(
            candidate_coord=candidate_coord,
            paths=paths,
            split=split,
        )
        proxy_summary = dict(result.summary)
        proxy_summary["length_budget_filter"] = stats.as_dict()
        (output_root / f"{split}.proxy_summary.json").write_text(
            json.dumps(proxy_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        candidate_coord.unlink()
        candidate_summary.unlink()
        return stats

    def _filter_candidate(
        self,
        *,
        candidate_coord: Path,
        paths: BuildPaths,
        split: str,
    ) -> SplitBuildStats:
        """Filter augmented candidate rows and emit coord/norm sidecars."""

        stats = SplitBuildStats(
            split=split,
            source_jsonl=str(candidate_coord),
            output_jsonl=str(paths.coord),
        )
        _ensure_parent(paths.coord)
        _ensure_parent(paths.norm)
        with (
            candidate_coord.open("r", encoding="utf-8") as src,
            paths.coord.open("w", encoding="utf-8") as coord_out,
            paths.norm.open("w", encoding="utf-8") as norm_out,
        ):
            for line in src:
                stripped = line.strip()
                if not stripped:
                    continue
                coord_record = json.loads(stripped)
                breakdown = self._estimator.measure(coord_record)
                stats.observe(coord_record, breakdown)
                if breakdown.total_tokens > self._max_total_tokens:
                    stats.drop(coord_record, breakdown)
                    continue
                _write_jsonl_record(coord_out, coord_record)
                _write_jsonl_record(norm_out, _coord_record_to_norm(coord_record))
                stats.keep(coord_record, breakdown)

        paths.stats.write_text(
            json.dumps(stats.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return stats


class ProvenanceManifestWriter:
    """Writer for Git-tracked public-data provenance manifests."""

    def __init__(self, *, repo_root: Path) -> None:
        self._repo_root = repo_root

    def write(
        self,
        *,
        relative_path: Path,
        producer_script: Path,
        command: str,
        inputs: list[dict[str, str]],
        key_params: dict[str, Any],
        notes: str,
    ) -> Path:
        """Write one manifest under ``manifests/public_data_provenance``."""

        manifest_path = (
            self._repo_root
            / "manifests"
            / "public_data_provenance"
            / Path(*relative_path.parts[1:]).with_suffix(".json")
        )
        payload = {
            "schema_version": 1,
            "relative_path": str(relative_path),
            "producer_script": str(producer_script),
            "working_dir": ".",
            "command": command,
            "inputs": inputs,
            "key_params": key_params,
            "checksums": _jsonl_checksums(self._repo_root / relative_path),
            "code_ref": {
                "git_commit": _git_head(),
                "git_dirty_allowed": True,
            },
            "generated_at_utc": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "notes": notes,
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--source-preset", type=Path, default=DEFAULT_SOURCE_PRESET)
    parser.add_argument("--coco-output", type=Path, default=DEFAULT_COCO_OUTPUT)
    parser.add_argument("--proxy-output", type=Path, default=DEFAULT_PROXY_OUTPUT)
    parser.add_argument("--projection-root", type=Path, default=DEFAULT_PROJECTION_ROOT)
    parser.add_argument("--mapping-csv", type=Path, default=DEFAULT_MAPPING_CSV)
    parser.add_argument("--max-total-tokens", type=int, default=12000)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--build-lvis-proxy", action="store_true")
    parser.add_argument("--skip-projection-build", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--old-max60-dir", type=Path, default=DEFAULT_OLD_MAX60_DIR)
    parser.add_argument("--old-proxy-dir", type=Path, default=DEFAULT_OLD_PROXY_DIR)
    return parser.parse_args()


def main() -> None:
    """Build requested length-budget artifacts and manifests."""

    args = parse_args()
    cfg = FactoryConfig(
        config_path=_resolve_repo_path(args.config),
        model_path=_resolve_repo_path(args.model_path),
        source_preset=_resolve_repo_path(args.source_preset),
        coco_output=_resolve_repo_path(args.coco_output),
        proxy_output=_resolve_repo_path(args.proxy_output),
        projection_root=_resolve_repo_path(args.projection_root),
        mapping_csv=_resolve_repo_path(args.mapping_csv),
        max_total_tokens=int(args.max_total_tokens),
        splits=tuple(str(split) for split in args.splits),
        build_lvis_proxy=bool(args.build_lvis_proxy),
        build_projections_if_missing=not bool(args.skip_projection_build),
        force=bool(args.force),
        old_max60_dir=_resolve_repo_path(args.old_max60_dir)
        if args.old_max60_dir is not None
        else None,
        old_proxy_dir=_resolve_repo_path(args.old_proxy_dir)
        if args.old_proxy_dir is not None
        else None,
    )

    _prepare_output_dir(cfg.coco_output, force=cfg.force)
    if cfg.build_lvis_proxy:
        _prepare_output_dir(cfg.proxy_output, force=cfg.force)

    estimator = CompactFullTokenBudgetEstimator.from_config(
        cfg.config_path, cfg.model_path
    )
    coco_builder = CocoLengthBudgetBuilder(
        estimator=estimator,
        converter=CoordTripletConverter(),
        max_total_tokens=cfg.max_total_tokens,
    )
    proxy_builder = ProxyLengthBudgetBuilder(
        estimator=estimator,
        max_total_tokens=cfg.max_total_tokens,
    )

    coco_stats = []
    for split in cfg.splits:
        stats = coco_builder.build_split(
            source_jsonl=cfg.source_preset / f"{split}.jsonl",
            shared_image_root=cfg.source_preset,
            output_root=cfg.coco_output,
            split=split,
        )
        coco_stats.append(stats)

    _write_pipeline_manifest(
        cfg.coco_output,
        artifact_id=f"coco-{cfg.max_total_tokens // 1000}k",
        source_preset=cfg.source_preset,
        stats=coco_stats,
        config=cfg,
        lvis_proxy=False,
    )

    proxy_stats = []
    if cfg.build_lvis_proxy:
        for split in cfg.splits:
            projection_dir = _ensure_projection(cfg, split=split)
            stats = proxy_builder.build_split(
                base_coord_jsonl=cfg.coco_output / f"{split}.coord.jsonl",
                output_root=cfg.proxy_output,
                projection_dir=projection_dir,
                mapping_csv=cfg.mapping_csv,
                raw_coco_annotation=_raw_coco_annotation(split),
                split=split,
            )
            proxy_stats.append(stats)
        _write_pipeline_manifest(
            cfg.proxy_output,
            artifact_id=f"coco-lvis_proxy-{cfg.max_total_tokens // 1000}k",
            source_preset=cfg.coco_output,
            stats=proxy_stats,
            config=cfg,
            lvis_proxy=True,
        )

    manifest_writer = ProvenanceManifestWriter(repo_root=REPO_ROOT)
    coco_rel = _repo_relative(cfg.coco_output)
    manifest_writer.write(
        relative_path=coco_rel,
        producer_script=Path("public_data/scripts/build_coco_length_budget_artifacts.py"),
        command=_repro_command(cfg, include_proxy=False),
        inputs=[
            {
                "kind": "processed_base",
                "path": str(_repo_relative(cfg.source_preset)),
                "notes": "COCO 1024 resized pixel-space base preset.",
            },
            {
                "kind": "training_config",
                "path": str(_repo_relative(cfg.config_path)),
                "notes": "Latest compact-full prompt/model/template contract.",
            },
        ],
        key_params=_manifest_key_params(cfg, lvis_proxy=False),
        notes="COCO 1024 compact-full dataset filtered by total 12k token budget.",
    )
    if cfg.build_lvis_proxy:
        proxy_rel = _repo_relative(cfg.proxy_output)
        manifest_writer.write(
            relative_path=proxy_rel,
            producer_script=Path("public_data/scripts/build_coco_length_budget_artifacts.py"),
            command=_repro_command(cfg, include_proxy=True),
            inputs=[
                {
                    "kind": "processed_base",
                    "path": str(coco_rel),
                    "notes": "Length-filtered COCO coord-token base branch.",
                },
                {
                    "kind": "semantic_mapping",
                    "path": str(_repo_relative(cfg.mapping_csv)),
                    "notes": "Versioned LVIS-to-COCO proxy mapping decisions.",
                },
                {
                    "kind": "raw_annotations",
                    "path": "public_data/coco/raw/annotations",
                    "notes": "COCO instance annotations used for original image dimensions.",
                },
                {
                    "kind": "raw_annotations",
                    "path": "public_data/lvis/raw/annotations",
                    "notes": "LVIS instance annotations used to build projection artifacts.",
                },
            ],
            key_params=_manifest_key_params(cfg, lvis_proxy=True),
            notes=(
                "COCO 1024 LVIS-proxy compact-full dataset filtered by final total "
                "12k token budget after proxy augmentation."
            ),
        )

    summary = {
        "coco": _summary_payload(coco_stats, old_dir=cfg.old_max60_dir),
        "lvis_proxy": _summary_payload(proxy_stats, old_dir=cfg.old_proxy_dir)
        if proxy_stats
        else None,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def _prepare_output_dir(path: Path, *, force: bool) -> None:
    """Create or replace an output directory."""

    if path.exists() and force:
        shutil.rmtree(path)
    elif path.exists():
        existing = sorted(p.name for p in path.iterdir())
        if existing:
            raise FileExistsError(
                f"Output directory is not empty: {path}. Use --force to rebuild."
            )
    path.mkdir(parents=True, exist_ok=True)


def _ensure_projection(config: FactoryConfig, *, split: str) -> Path:
    """Return a projection directory, building it when requested and missing."""

    split_tag = f"{split}2017"
    projection_dir = config.projection_root / split_tag
    recovered = projection_dir / "recovered_coco80_instances.jsonl"
    if recovered.is_file():
        return projection_dir
    if not config.build_projections_if_missing:
        raise FileNotFoundError(
            f"Missing projection artifact: {recovered}. "
            "Rerun without --skip-projection-build to regenerate it."
        )

    projection_dir.mkdir(parents=True, exist_ok=True)
    run_coco_lvis_projection_analysis(
        output_dir=projection_dir,
        coco_annotation_paths=(_raw_coco_annotation(split),),
        lvis_annotation_paths=(_raw_lvis_annotation(split),),
        config=AnalysisConfig(allowed_coco_image_splits=(split_tag,)),
    )
    return projection_dir


def _raw_coco_annotation(split: str) -> Path:
    """Return the raw COCO instances JSON for a split."""

    return REPO_ROOT / "public_data" / "coco" / "raw" / "annotations" / f"instances_{split}2017.json"


def _raw_lvis_annotation(split: str) -> Path:
    """Return the raw LVIS annotation JSON for a split."""

    return REPO_ROOT / "public_data" / "lvis" / "raw" / "annotations" / f"lvis_v1_{split}.json"


def _model_facing_objects(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return compact renderer object view, dropping non-model metadata keys.

    The legacy builder measured coord-token JSONLs. Canonical Phase 1 views store
    norm1000 integer boxes, so normalize either surface to Qwen coord-token text
    before rendering and tokenization.
    """

    objects = record.get("objects") or []
    model_objects: list[dict[str, Any]] = []
    for obj in objects:
        if not isinstance(obj, Mapping):
            continue
        model_objects.append(
            {
                "desc": obj.get("desc"),
                "bbox_2d": _model_facing_bbox(obj.get("bbox_2d")),
            }
        )
    return model_objects


def _model_facing_bbox(value: Any) -> Any:
    """Return a coord-token bbox for compact assistant rendering."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return value

    tokens: list[str] = []
    for component in value:
        if isinstance(component, str):
            tokens.append(component)
        else:
            tokens.append(int_to_token(int(component)))
    return tokens


def _coord_record_to_norm(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a norm1000 copy of a coord-token record."""

    out = copy.deepcopy(dict(record))
    for obj in out.get("objects") or []:
        if not isinstance(obj, MutableMapping):
            continue
        bbox = obj.get("bbox_2d")
        if isinstance(bbox, Sequence) and not isinstance(bbox, (str, bytes)):
            obj["bbox_2d"] = [
                token_to_int(str(value)) if isinstance(value, str) else int(value)
                for value in bbox
            ]
    return out


def _with_shared_image_refs(
    *,
    record: Mapping[str, Any],
    source_image_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Return a record whose image paths point at the shared image root."""

    out = copy.deepcopy(dict(record))
    rewritten_images: list[str] = []
    for image in record.get("images") or []:
        rel = Path(str(image))
        if rel.is_absolute() or ".." in rel.parts:
            image_abs = (output_root / rel).resolve(strict=False)
        else:
            image_abs = source_image_root / rel
        rewritten_images.append(os.path.relpath(image_abs, start=output_root))
    out["images"] = rewritten_images
    return out


def _write_pipeline_manifest(
    output_root: Path,
    *,
    artifact_id: str,
    source_preset: Path,
    stats: Sequence[SplitBuildStats],
    config: FactoryConfig,
    lvis_proxy: bool,
) -> None:
    """Write an artifact-local pipeline manifest."""

    payload = {
        "artifact_id": artifact_id,
        "dataset_id": "coco",
        "relative_path": str(_repo_relative(output_root)),
        "source_preset": str(_repo_relative(source_preset)),
        "shared_image_root": str(_repo_relative(config.source_preset / "images")),
        "max_total_tokens": int(config.max_total_tokens),
        "tokenizer": str(_repo_relative(config.model_path)),
        "training_config": str(_repo_relative(config.config_path)),
        "template": "compact_full",
        "length_formula": (
            "len(chat_template_ids) - count(<|image_pad|>) + "
            "sum(ceil(height/32) * ceil(width/32))"
        ),
        "includes_lvis_proxy": bool(lvis_proxy),
        "splits": {item.split: item.as_dict() for item in stats},
    }
    (output_root / "pipeline_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _jsonl_checksums(root: Path) -> dict[str, Any]:
    """Return JSONL-only checksum payload for a materialized artifact root."""

    files = []
    aggregate_lines = []
    for path in sorted(root.glob("*.jsonl")):
        rel = _repo_relative(path)
        sha = _sha256(path)
        records = _count_nonempty_lines(path)
        size = path.stat().st_size
        entry = {
            "path": str(rel),
            "sha256": sha,
            "size_bytes": int(size),
            "records": int(records),
        }
        files.append(entry)
        aggregate_lines.append(f"{rel} {sha} {size} {records}\n")
    aggregate = hashlib.sha256("".join(aggregate_lines).encode("utf-8")).hexdigest()
    return {
        "scope": "jsonl_training_samples_only",
        "algorithm": "sha256",
        "files": files,
        "aggregate_sha256": aggregate,
    }


def _manifest_key_params(config: FactoryConfig, *, lvis_proxy: bool) -> dict[str, Any]:
    """Return common manifest key parameters."""

    return {
        "max_total_tokens": int(config.max_total_tokens),
        "token_budget_components": [
            "image_patch_tokens_post_merge",
            "system_prompt_tokens",
            "user_chat_tokens",
            "assistant_compact_full_detection_tokens",
        ],
        "tokenizer": str(_repo_relative(config.model_path)),
        "training_config": str(_repo_relative(config.config_path)),
        "template": "compact_full",
        "image_storage_policy": "share_same_resolution_image_root",
        "shared_image_root": str(_repo_relative(config.source_preset / "images")),
        "image_factor": 32,
        "max_pixels": 1048576,
        "max_objects": None,
        "lvis_proxy": bool(lvis_proxy),
        "routine_sync_policy": "regenerate_from_raw_plus_manifest",
    }


def _repro_command(config: FactoryConfig, *, include_proxy: bool) -> str:
    """Return a single repo-root command that reproduces the artifact."""

    command = (
        "PYTHONPATH=. conda run -n ms python "
        "public_data/scripts/build_coco_length_budget_artifacts.py "
        f"--config {_repo_relative(config.config_path)} "
        f"--model-path {_repo_relative(config.model_path)} "
        f"--source-preset {_repo_relative(config.source_preset)} "
        f"--coco-output {_repo_relative(config.coco_output)} "
        f"--proxy-output {_repo_relative(config.proxy_output)} "
        f"--projection-root {_repo_relative(config.projection_root)} "
        f"--mapping-csv {_repo_relative(config.mapping_csv)} "
        f"--max-total-tokens {config.max_total_tokens} "
        f"--splits {' '.join(config.splits)} "
        "--force"
    )
    if include_proxy:
        command += " --build-lvis-proxy"
    return command


def _summary_payload(
    stats: Sequence[SplitBuildStats],
    *,
    old_dir: Path | None,
) -> dict[str, Any]:
    """Return aggregate summary with old-artifact comparison when available."""

    payload = {
        "splits": {item.split: item.as_dict() for item in stats},
        "totals": {
            "records_seen": sum(item.records_seen for item in stats),
            "records_written": sum(item.records_written for item in stats),
            "records_dropped": sum(item.records_dropped for item in stats),
            "dense_records_written_object_count_gt_60": sum(
                item.dense_records_written for item in stats
            ),
        },
    }
    if old_dir is not None and old_dir.exists():
        payload["old_artifact_records"] = {
            split: _count_nonempty_lines(old_dir / f"{split}.coord.jsonl")
            for split in payload["splits"]
            if (old_dir / f"{split}.coord.jsonl").is_file()
        }
    return payload


def _record_example(
    record: Mapping[str, Any],
    *,
    breakdown: TokenBudgetBreakdown,
) -> dict[str, Any]:
    """Return compact row identity for stats examples."""

    return {
        "image_id": record.get("image_id"),
        "file_name": record.get("file_name"),
        "object_count": int(breakdown.object_count),
        "total_tokens": int(breakdown.total_tokens),
        "assistant_tokens": int(breakdown.assistant_tokens),
        "image_patch_tokens": int(breakdown.image_patch_tokens),
    }


def _summarize_ints(values: Sequence[int]) -> dict[str, Any]:
    """Return robust integer summary statistics."""

    if not values:
        return {"count": 0}
    sorted_values = sorted(int(value) for value in values)
    return {
        "count": len(sorted_values),
        "min": sorted_values[0],
        "mean": float(mean(sorted_values)),
        "median": float(median(sorted_values)),
        "p95": sorted_values[int((len(sorted_values) - 1) * 0.95)],
        "p99": sorted_values[int((len(sorted_values) - 1) * 0.99)],
        "max": sorted_values[-1],
    }


def _write_jsonl_record(handle: Any, record: Mapping[str, Any]) -> None:
    """Write one JSONL record."""

    handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def _ensure_parent(path: Path) -> None:
    """Create the parent directory for a file path."""

    path.parent.mkdir(parents=True, exist_ok=True)


def _sha256(path: Path) -> str:
    """Return file SHA256 hex digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_nonempty_lines(path: Path) -> int:
    """Return non-empty JSONL line count."""

    count = 0
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _repo_relative(path: Path) -> Path:
    """Return a path relative to the repository root."""

    absolute_path = path if path.is_absolute() else REPO_ROOT / path
    return absolute_path.absolute().relative_to(REPO_ROOT.absolute())


def _resolve_repo_path(path: Path) -> Path:
    """Resolve a repo-relative or absolute path."""

    return path if path.is_absolute() else REPO_ROOT / path


def _git_head() -> str:
    """Return the current Git HEAD SHA when available."""

    head_path = REPO_ROOT / ".git" / "HEAD"
    if not head_path.is_file():
        return "unknown"
    head = head_path.read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        ref_path = REPO_ROOT / ".git" / head.removeprefix("ref: ").strip()
        if ref_path.is_file():
            return ref_path.read_text(encoding="utf-8").strip()
    return head


if __name__ == "__main__":
    main()
