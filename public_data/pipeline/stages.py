from __future__ import annotations

import json
from abc import ABC, abstractmethod
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

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


def _assert_rescale_target_is_fresh(state: PipelineState) -> None:
    preset_dir = state.preset_dir
    if not preset_dir.exists():
        return

    blocking_paths: list[Path] = []
    images_dir = preset_dir / "images"
    manifest_path = preset_dir / "pipeline_manifest.json"

    if images_dir.exists():
        blocking_paths.append(images_dir)
    if manifest_path.exists():
        blocking_paths.append(manifest_path)

    for split in sorted(state.split_artifacts.keys()):
        paths = state.split_artifacts[split]
        for artifact_path in (paths.raw, paths.norm, paths.coord, paths.filter_stats):
            if artifact_path.exists():
                blocking_paths.append(artifact_path)

    if not blocking_paths:
        return

    preview = ", ".join(str(path) for path in blocking_paths[:4])
    remainder = len(blocking_paths) - 4
    if remainder > 0:
        preview = f"{preview}, ... (+{remainder} more)"

    raise RuntimeError(
        "Rescale target preset is not fresh; refusing in-place overwrite. "
        "Use a new preset name or delete the entire preset directory before rerunning. "
        f"Found existing artifacts: {preview}"
    )


class RescaleStage(PipelineStage):
    name = "rescale"

    def run(self, state: PipelineState) -> None:
        _assert_rescale_target_is_fresh(state)

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

            workers = max(1, int(state.config.num_workers))
            if src == dst:
                tmp = dst.with_name(dst.name + ".tmp")
                stats = _filter_records_drop_only(src, tmp, n_max, num_workers=workers)
                tmp.replace(dst)
            else:
                stats = _filter_records_drop_only(src, dst, n_max, num_workers=workers)

            state.split_raw_sources[split] = dst
            state.split_artifacts[split].filter_stats.write_text(
                json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            per_split[split] = stats

        state.stage_stats[self.name] = {
            "enabled": True,
            "max_objects": n_max,
            "num_workers": int(state.config.num_workers),
            "splits": per_split,
        }


class DerivedImagesHardlinkStage(PipelineStage):
    name = "derived_image_hardlinks"

    def __init__(self, *, materialize_fn: Callable[..., None]) -> None:
        self._materialize_fn = materialize_fn

    def run(self, state: PipelineState) -> None:
        if not state.is_derived_preset:
            state.stage_stats[self.name] = {"enabled": False}
            return

        self._materialize_fn(
            base_preset_dir=state.base_preset_dir,
            derived_preset_dir=state.preset_dir,
            split_paths=state.split_artifacts,
        )
        state.stage_stats[self.name] = {
            "enabled": True,
            "base_preset_dir": str(state.base_preset_dir),
            "derived_preset_dir": str(state.preset_dir),
        }


class NormalizeStage(PipelineStage):
    name = "normalize_norm1000"

    def run(self, state: PipelineState) -> None:
        record_counts: dict[str, int] = {}
        object_counts: dict[str, dict] = {}
        for split in sorted(state.split_artifacts.keys()):
            src = state.split_raw_sources.get(split) or state.split_artifacts[split].raw
            outp = state.split_artifacts[split].norm
            per_split_objects: dict = {}
            n = _write_norm_jsonl(
                input_path=src,
                output_path=outp,
                assume_normalized=bool(state.config.assume_normalized),
                compact=bool(state.config.compact_json),
                stats=per_split_objects,
                num_workers=max(1, int(state.config.num_workers)),
            )
            record_counts[split] = n
            object_counts[split] = per_split_objects

        state.stage_stats[self.name] = {
            "assume_normalized": bool(state.config.assume_normalized),
            "num_workers": int(state.config.num_workers),
            "records": record_counts,
            "objects": object_counts,
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
                num_workers=max(1, int(state.config.num_workers)),
            )
            stats[split] = n

        state.stage_stats[self.name] = {
            "num_workers": int(state.config.num_workers),
            "records": stats,
        }



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
            raise FileNotFoundError(
                "Validation stage missing expected artifacts: "
                + ", ".join(str(p) for p in missing)
            )

        cfg = state.config
        check_images = not self.skip_image_check

        raw_inputs: set[Path] = {p for p in state.split_inputs.values() if p is not None}

        # Avoid opening every image multiple times: only do the expensive size check on the
        # most downstream preset artifact we emit for each split.
        preset_size_check_targets: set[Path] = set()
        for split in sorted(state.split_artifacts.keys()):
            paths = state.split_artifacts[split]
            if self.include_coord:
                preset_size_check_targets.add(paths.coord)
            elif self.include_norm:
                preset_size_check_targets.add(paths.norm)
            elif self.include_preset:
                preset_size_check_targets.add(paths.raw)

        # Preset image roots must always be real directories. This is required for
        # derived presets that reuse images via hardlinks and prevents symlink hazards.
        if self.include_preset or self.include_norm or self.include_coord:
            images_dir = state.preset_dir / "images"
            if images_dir.exists():
                if images_dir.is_symlink() or images_dir.is_file() or not images_dir.is_dir():
                    raise RuntimeError(
                        "Preset images dir must be a real directory (not symlink/file): "
                        f"{images_dir} (preset={state.effective_preset})"
                    )
            elif state.config.max_objects is not None:
                raise RuntimeError(
                    "Derived preset is missing images/ directory required for hardlink materialization: "
                    f"{images_dir} (preset={state.effective_preset})"
                )

        raw_image_open_check_n = 64
        preset_image_open_check_n = 64

        failed: list[str] = []
        for path in target_paths:
            is_raw_input = path in raw_inputs
            expected_max_pixels: Optional[int] = None
            expected_multiple_of: Optional[int] = None
            if not is_raw_input:
                expected_max_pixels = int(cfg.max_pixels)
                expected_multiple_of = int(cfg.image_factor)

            check_image_sizes = False
            if check_images:
                if is_raw_input:
                    check_image_sizes = True
                elif path in preset_size_check_targets:
                    check_image_sizes = True

            image_check_mode = "none"
            if check_images:
                image_check_mode = "open" if check_image_sizes else "exists"

            image_check_n = 0
            if check_image_sizes:
                image_check_n = (
                    raw_image_open_check_n
                    if is_raw_input
                    else preset_image_open_check_n
                )

            validator = JSONLValidator(
                check_images=check_images,
                verbose=False,
                expected_max_pixels=expected_max_pixels,
                expected_multiple_of=expected_multiple_of,
                check_image_sizes=check_image_sizes,
                image_check_mode=image_check_mode,
                image_check_n=image_check_n,
                enforce_rescale_images_real_dir=(
                    self.include_preset or self.include_norm or self.include_coord
                ),
            )
            if not validator.validate_file(str(path)):
                failed.append(str(path))

        if failed:
            raise RuntimeError("Validation stage failed for: " + ", ".join(failed))

        state.stage_stats[self.name] = {
            "files": [str(p) for p in target_paths],
            "skip_image_check": self.skip_image_check,
            "max_pixels": int(cfg.max_pixels),
            "image_factor": int(cfg.image_factor),
            "size_check_files": sorted(str(p) for p in preset_size_check_targets),
            "raw_image_open_check_n": raw_image_open_check_n,
            "preset_image_open_check_n": preset_image_open_check_n,
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


def _normalize_worker_count(num_workers: int) -> int:
    return max(1, min(int(num_workers), cpu_count()))


def _iter_nonempty_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as fin:
        for raw in fin:
            line = raw.strip()
            if line:
                yield line


def _parallel_imap(
    *,
    payloads: Iterable[Any],
    worker_fn,
    num_workers: int,
    chunksize: int = 64,
) -> Iterable[Any]:
    workers = _normalize_worker_count(num_workers)
    if workers == 1:
        for payload in payloads:
            yield worker_fn(payload)
        return

    with Pool(processes=workers) as pool:
        yield from pool.imap(worker_fn, payloads, chunksize=max(1, int(chunksize)))


def _merge_int_stats(dst: dict[str, int], src: dict[str, int]) -> None:
    for k, v in src.items():
        dst[k] = int(dst.get(k, 0)) + int(v)


def _filter_record_drop_worker(payload: Tuple[str, int]) -> Tuple[bool, str, dict[str, int]]:
    line, max_objects = payload
    rec = json.loads(line)

    stats: dict[str, int] = {
        "images_seen": 1,
        "images_written": 0,
        "images_dropped": 0,
        "objects_seen": 0,
        "objects_written": 0,
        "objects_poly": 0,
        "objects_bbox": 0,
    }

    objects = rec.get("objects") or []
    n = len(objects) if isinstance(objects, list) else 0
    stats["objects_seen"] = int(n)
    if n > int(max_objects):
        stats["images_dropped"] = 1
        return (False, "", stats)

    stats["images_written"] = 1
    stats["objects_written"] = int(n)
    if isinstance(objects, list):
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            if obj.get("poly") is not None:
                stats["objects_poly"] += 1
            elif obj.get("bbox_2d") is not None:
                stats["objects_bbox"] += 1

    return (True, json.dumps(rec, ensure_ascii=False), stats)


def _norm_record_worker(payload: Tuple[str, bool, bool]) -> Tuple[str, dict[str, int]]:
    line, assume_normalized, compact = payload
    record_raw = json.loads(line)
    local_stats: dict[str, int] = {}
    record_ints = convert_record_to_ints(
        json.loads(json.dumps(record_raw)),
        ["bbox_2d", "poly"],
        assume_normalized=bool(assume_normalized),
        stats=local_stats,
    )
    record_ints = _canonicalize_and_sort_objects_in_place(record_ints)
    compact_separators = (",", ":") if compact else None
    out_line = json.dumps(record_ints, ensure_ascii=False, separators=compact_separators)
    return (out_line, local_stats)


def _coord_record_worker(payload: Tuple[str, bool]) -> str:
    line, compact = payload
    record_norm = json.loads(line)
    record_tokens = convert_record_to_tokens(
        json.loads(json.dumps(record_norm)),
        ["bbox_2d", "poly"],
    )
    compact_separators = (",", ":") if compact else None
    return json.dumps(record_tokens, ensure_ascii=False, separators=compact_separators)


def _filter_records_drop_only(
    input_path: Path,
    output_path: Path,
    max_objects: int,
    *,
    num_workers: int = 1,
) -> dict:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("max-object filter cannot read/write in-place; use a temp output path")

    stats: dict[str, int] = {
        "images_seen": 0,
        "images_written": 0,
        "images_dropped": 0,
        "objects_seen": 0,
        "objects_written": 0,
        "objects_poly": 0,
        "objects_bbox": 0,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payloads = ((line, int(max_objects)) for line in _iter_nonempty_lines(input_path))

    with output_path.open("w", encoding="utf-8") as fout:
        for keep, out_line, record_stats in _parallel_imap(
            payloads=payloads,
            worker_fn=_filter_record_drop_worker,
            num_workers=num_workers,
            chunksize=128,
        ):
            _merge_int_stats(stats, record_stats)
            if keep:
                fout.write(out_line + "\n")
    return stats


def _write_norm_jsonl(
    *,
    input_path: Path,
    output_path: Path,
    assume_normalized: bool,
    compact: bool,
    stats: Optional[dict] = None,
    num_workers: int = 1,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0

    payloads = (
        (line, bool(assume_normalized), bool(compact))
        for line in _iter_nonempty_lines(input_path)
    )

    with output_path.open("w", encoding="utf-8") as fout:
        for out_line, local_stats in _parallel_imap(
            payloads=payloads,
            worker_fn=_norm_record_worker,
            num_workers=num_workers,
            chunksize=64,
        ):
            fout.write(out_line + "\n")
            n += 1
            if stats is not None:
                _merge_int_stats(stats, local_stats)
    return n


def _write_coord_jsonl(
    *,
    input_path: Path,
    output_path: Path,
    compact: bool,
    num_workers: int = 1,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0

    payloads = ((line, bool(compact)) for line in _iter_nonempty_lines(input_path))
    with output_path.open("w", encoding="utf-8") as fout:
        for out_line in _parallel_imap(
            payloads=payloads,
            worker_fn=_coord_record_worker,
            num_workers=num_workers,
            chunksize=64,
        ):
            fout.write(out_line + "\n")
            n += 1
    return n
