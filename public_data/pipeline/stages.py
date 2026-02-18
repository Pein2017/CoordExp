from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, List

from public_data.scripts.convert_to_coord_tokens import (
    _canonicalize_and_sort_objects_in_place,
    convert_record_to_ints,
    convert_record_to_tokens,
)
from public_data.scripts.rescale_jsonl import run_smart_resize
from public_data.scripts.validate_jsonl import JSONLValidator
from src.datasets.preprocessors.resize import SmartResizeParams

from .structural import run_structural_preflight
from .types import PipelineState
from .writer import ensure_legacy_raw_alias


class PipelineStage(ABC):
    name: str

    @abstractmethod
    def run(self, state: PipelineState) -> None:
        raise NotImplementedError


class StructuralPreflightStage(PipelineStage):
    name = "structural_preflight"

    def __init__(self, *, source: str) -> None:
        self._source = source

    def _paths(self, state: PipelineState) -> list[Path]:
        out: list[Path] = []
        for split in sorted(state.split_artifacts.keys()):
            if self._source == "raw_input":
                p = state.split_inputs.get(split)
            elif self._source == "raw_output":
                p = state.split_artifacts[split].raw
            elif self._source == "norm_output":
                p = state.split_artifacts[split].norm
            elif self._source == "coord_output":
                p = state.split_artifacts[split].coord
            else:
                raise ValueError(f"Unknown preflight source: {self._source}")
            if p is not None and p.exists():
                out.append(p)
        return out

    def run(self, state: PipelineState) -> None:
        stats = run_structural_preflight(self._paths(state))
        state.stage_stats[self.name] = {
            "source": self._source,
            **stats,
        }


class RescaleStage(PipelineStage):
    name = "rescale"

    def run(self, state: PipelineState) -> None:
        cfg = state.config
        params = SmartResizeParams(
            max_pixels=int(cfg.max_pixels),
            min_pixels=int(cfg.min_pixels),
            image_factor=int(cfg.image_factor),
        )
        state.preset_dir.mkdir(parents=True, exist_ok=True)

        lines_written: Dict[str, int] = {}
        for split in sorted(state.split_artifacts.keys()):
            inp = state.split_inputs[split]
            outp = state.split_artifacts[split].raw
            run_smart_resize(
                input_jsonl=inp,
                output_jsonl=outp,
                images_output_dir=state.preset_dir,
                params=params,
                relative_image_paths=bool(cfg.relative_images),
                images_root_override=None,
                num_workers=max(1, int(cfg.num_workers)),
            )
            state.split_raw_sources[split] = outp
            lines_written[split] = _count_lines(outp)

        state.stage_stats[self.name] = {
            "image_factor": cfg.image_factor,
            "max_pixels": cfg.max_pixels,
            "min_pixels": cfg.min_pixels,
            "num_workers": cfg.num_workers,
            "records": lines_written,
        }


class MaxObjectsFilterStage(PipelineStage):
    name = "max_objects_filter"

    def run(self, state: PipelineState) -> None:
        max_objects = state.config.max_objects
        if max_objects is None:
            state.stage_stats[self.name] = {"enabled": False}
            return

        n_max = int(max_objects)
        if n_max <= 0:
            raise ValueError("max_objects must be > 0")

        per_split = {}
        for split in sorted(state.split_artifacts.keys()):
            src = state.split_raw_sources.get(split) or state.split_artifacts[split].raw
            dst = state.split_artifacts[split].raw

            if src == dst:
                tmp = dst.with_name(dst.name + ".tmp")
                stats = _filter_records_drop_only(src, tmp, n_max)
                tmp.replace(dst)
            else:
                stats = _filter_records_drop_only(src, dst, n_max)

            state.split_raw_sources[split] = dst
            state.split_artifacts[split].filter_stats.write_text(
                json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            per_split[split] = stats

        state.stage_stats[self.name] = {
            "enabled": True,
            "max_objects": n_max,
            "splits": per_split,
        }


class NormalizeStage(PipelineStage):
    name = "normalize_norm1000"

    def run(self, state: PipelineState) -> None:
        stats = {}
        for split in sorted(state.split_artifacts.keys()):
            src = state.split_raw_sources.get(split) or state.split_artifacts[split].raw
            outp = state.split_artifacts[split].norm
            n = _write_norm_jsonl(
                input_path=src,
                output_path=outp,
                assume_normalized=bool(state.config.assume_normalized),
                compact=bool(state.config.compact_json),
            )
            stats[split] = n

        state.stage_stats[self.name] = {
            "assume_normalized": bool(state.config.assume_normalized),
            "records": stats,
        }


class CoordTokenStage(PipelineStage):
    name = "coord_tokens"

    def run(self, state: PipelineState) -> None:
        stats = {}
        for split in sorted(state.split_artifacts.keys()):
            src = state.split_artifacts[split].norm
            outp = state.split_artifacts[split].coord
            n = _write_coord_jsonl(
                input_path=src,
                output_path=outp,
                compact=bool(state.config.compact_json),
            )
            stats[split] = n

        state.stage_stats[self.name] = {"records": stats}


class LegacyAliasStage(PipelineStage):
    name = "legacy_alias"

    def run(self, state: PipelineState) -> None:
        for split in sorted(state.split_artifacts.keys()):
            ensure_legacy_raw_alias(state.split_artifacts[split])
        state.stage_stats[self.name] = {"enabled": True}


class ValidationStage(PipelineStage):
    name = "validation"

    def __init__(
        self,
        *,
        include_raw: bool,
        include_preset: bool,
        include_norm: bool,
        include_coord: bool,
        skip_image_check: bool,
    ) -> None:
        self.include_raw = include_raw
        self.include_preset = include_preset
        self.include_norm = include_norm
        self.include_coord = include_coord
        self.skip_image_check = skip_image_check

    def _targets(self, state: PipelineState) -> tuple[list[Path], list[Path]]:
        targets: list[Path] = []
        missing: list[Path] = []

        for split in sorted(state.split_artifacts.keys()):
            split_paths = state.split_artifacts[split]
            if self.include_raw and state.split_inputs.get(split) is not None:
                targets.append(state.split_inputs[split])
            if self.include_preset:
                targets.append(split_paths.raw)
            if self.include_norm:
                targets.append(split_paths.norm)
            if self.include_coord:
                targets.append(split_paths.coord)

        dedup: list[Path] = []
        seen: set[Path] = set()
        for p in targets:
            if p in seen:
                continue
            seen.add(p)
            if p.exists():
                dedup.append(p)
            else:
                missing.append(p)
        return dedup, missing

    def run(self, state: PipelineState) -> None:
        target_paths, missing = self._targets(state)
        if missing:
            raise FileNotFoundError("Validation stage missing expected artifacts: " + ", ".join(str(p) for p in missing))

        failed: list[str] = []
        for path in target_paths:
            validator = JSONLValidator(check_images=not self.skip_image_check, verbose=False)
            if not validator.validate_file(str(path)):
                failed.append(str(path))
        if failed:
            raise RuntimeError("Validation stage failed for: " + ", ".join(failed))

        state.stage_stats[self.name] = {
            "files": [str(p) for p in target_paths],
            "skip_image_check": self.skip_image_check,
        }


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _filter_records_drop_only(input_path: Path, output_path: Path, max_objects: int) -> dict:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("max-object filter cannot read/write in-place; use a temp output path")

    stats = {
        "images_seen": 0,
        "images_written": 0,
        "images_dropped": 0,
        "objects_seen": 0,
        "objects_written": 0,
        "objects_poly": 0,
        "objects_bbox": 0,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            stats["images_seen"] += 1
            objects = rec.get("objects") or []
            n = len(objects) if isinstance(objects, list) else 0
            stats["objects_seen"] += int(n)
            if n > max_objects:
                stats["images_dropped"] += 1
                continue
            stats["images_written"] += 1
            stats["objects_written"] += int(n)
            if isinstance(objects, list):
                for obj in objects:
                    if not isinstance(obj, dict):
                        continue
                    if obj.get("poly") is not None:
                        stats["objects_poly"] += 1
                    elif obj.get("bbox_2d") is not None:
                        stats["objects_bbox"] += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return stats


def _write_norm_jsonl(*, input_path: Path, output_path: Path, assume_normalized: bool, compact: bool) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    compact_separators = (",", ":") if compact else None
    n = 0
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record_raw = json.loads(line)
            record_ints = convert_record_to_ints(
                json.loads(json.dumps(record_raw)),
                ["bbox_2d", "poly"],
                assume_normalized=assume_normalized,
            )
            record_ints = _canonicalize_and_sort_objects_in_place(record_ints)
            json.dump(record_ints, fout, ensure_ascii=False, separators=compact_separators)
            fout.write("\n")
            n += 1
    return n


def _write_coord_jsonl(*, input_path: Path, output_path: Path, compact: bool) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    compact_separators = (",", ":") if compact else None
    n = 0
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record_norm = json.loads(line)
            record_tokens = convert_record_to_tokens(
                json.loads(json.dumps(record_norm)),
                ["bbox_2d", "poly"],
            )
            json.dump(record_tokens, fout, ensure_ascii=False, separators=compact_separators)
            fout.write("\n")
            n += 1
    return n
