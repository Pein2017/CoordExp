from __future__ import annotations

import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import yaml
from PIL import Image

from src.analysis.rollout_fn_factor_study import (
    HFStudyRunner,
    ResolvedCheckpoint,
    RolloutGeneration,
    _serialize_objects_to_prefix_text,
)
from src.analysis.rollout_parity import collect_stage2_parity_gt_vs_pred
from src.analysis.unmatched_proposal_verifier import (
    EvalConfig,
    TeacherForcedScorer,
    _bbox_iou_xyxy,
    _build_bbox_gt_object,
    _build_closed_container_text,
    _extract_dataset_gt_bbox,
    _load_image,
    _mask_image,
    _norm1000_to_pixel,
    _pixel_to_norm1000,
    _resolve_existing_input_path,
    _run_detection_eval,
    REPO_ROOT,
    resolve_checkpoint_path,
    resolve_prompt_controls_for_checkpoint,
)
from src.common.semantic_desc import normalize_desc
from src.infer.engine import GenerationConfig
from src.trainers.rollout_matching.parsing import (
    find_desc_value_token_positions_by_span,
    parse_rollout_for_matching,
)


_VALID_STAGES = ("subset", "decode", "prefix_probe", "counterfactual", "report")
_VALID_DIRECTIONS = ("left", "right", "up", "down")


@dataclass(frozen=True)
class RunConfig:
    name: str
    output_dir: str
    stages: Tuple[str, ...]


@dataclass(frozen=True)
class SubsetConfig:
    dataset_jsonl: str
    monitor_dump_paths: Tuple[str, ...]
    root_image_dir: Optional[str]
    max_samples: int
    min_fp_count: int
    require_selection: Optional[str]
    require_small_object: bool
    small_object_max_area_frac: float


@dataclass(frozen=True)
class CheckpointConfig:
    path: str
    alias: str
    prompt_variant: Optional[str]
    object_field_order: Optional[str]


@dataclass(frozen=True)
class DecodeConfig:
    temperatures: Tuple[float, ...]
    top_p: float
    max_new_tokens: int
    repetition_penalty: float
    batch_size: int
    tensor_parallel_size: int
    gpu_memory_utilization: float
    max_model_len: int
    max_num_seqs: int
    enforce_eager: bool
    seed: int
    local_radius_px: float
    local_radius_scale: float


@dataclass(frozen=True)
class PrefixProbeConfig:
    max_samples: int
    prefix_lengths: Tuple[int, ...]
    jitter_pixels: Tuple[int, ...]
    directions: Tuple[str, ...]
    device: str
    temperature: float
    top_p: float
    max_new_tokens: int
    repetition_penalty: float
    seed: int


@dataclass(frozen=True)
class CounterfactualConfig:
    max_samples: int
    jitter_pixels: Tuple[int, ...]
    directions: Tuple[str, ...]
    device: str
    attn_implementation: str
    mask_fill: int


@dataclass(frozen=True)
class EvalConfig:
    semantic_model: str
    semantic_threshold: float
    semantic_device: str
    semantic_batch_size: int
    num_workers: int
    f1ish_iou_thrs: Tuple[float, ...]
    f1ish_pred_scope: str
    use_segm: bool

@dataclass(frozen=True)
class StudyConfig:
    run: RunConfig
    subset: SubsetConfig
    checkpoint: CheckpointConfig
    decode: DecodeConfig
    prefix_probe: PrefixProbeConfig
    counterfactual: CounterfactualConfig
    eval: EvalConfig


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping YAML at {path}")
    return data


def _ensure_tuple_str(raw: Any, *, default: Sequence[str]) -> Tuple[str, ...]:
    if raw is None:
        return tuple(str(v) for v in default)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("Expected sequence of strings")
    return tuple(str(v) for v in raw if str(v).strip())


def _ensure_tuple_float(raw: Any, *, default: Sequence[float]) -> Tuple[float, ...]:
    if raw is None:
        return tuple(float(v) for v in default)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("Expected sequence of floats")
    return tuple(float(v) for v in raw)


def _ensure_tuple_int(raw: Any, *, default: Sequence[int]) -> Tuple[int, ...]:
    if raw is None:
        return tuple(int(v) for v in default)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("Expected sequence of ints")
    return tuple(int(v) for v in raw)


def load_study_config(path: Path) -> StudyConfig:
    raw = _load_yaml(path)
    run_raw = raw.get("run") or {}
    subset_raw = raw.get("subset") or {}
    checkpoint_raw = raw.get("checkpoint") or {}
    decode_raw = raw.get("decode") or {}
    prefix_raw = raw.get("prefix_probe") or {}
    counterfactual_raw = raw.get("counterfactual") or {}
    eval_raw = raw.get("eval") or {}

    run = RunConfig(
        name=str(run_raw.get("name") or "small-object-duplication-diagnostics").strip(),
        output_dir=str(run_raw.get("output_dir") or "output/analysis").strip(),
        stages=_ensure_tuple_str(run_raw.get("stages"), default=_VALID_STAGES),
    )
    invalid_stages = [stage for stage in run.stages if stage not in _VALID_STAGES]
    if invalid_stages:
        raise ValueError(f"Unsupported stages: {invalid_stages}")

    dataset_jsonl = str(subset_raw.get("dataset_jsonl") or "").strip()
    monitor_dump_paths = _ensure_tuple_str(
        subset_raw.get("monitor_dump_paths"),
        default=(),
    )
    if not dataset_jsonl:
        raise ValueError("subset.dataset_jsonl is required")
    if not monitor_dump_paths:
        raise ValueError("subset.monitor_dump_paths is required")
    subset = SubsetConfig(
        dataset_jsonl=dataset_jsonl,
        monitor_dump_paths=monitor_dump_paths,
        root_image_dir=(
            str(subset_raw["root_image_dir"]).strip()
            if subset_raw.get("root_image_dir") is not None
            else None
        ),
        max_samples=int(subset_raw.get("max_samples", 12)),
        min_fp_count=int(subset_raw.get("min_fp_count", 20)),
        require_selection=(
            str(subset_raw["require_selection"]).strip()
            if subset_raw.get("require_selection") is not None
            else "suspicious_duplication"
        ),
        require_small_object=bool(subset_raw.get("require_small_object", True)),
        small_object_max_area_frac=float(
            subset_raw.get("small_object_max_area_frac", 0.01)
        ),
    )

    checkpoint_path = str(checkpoint_raw.get("path") or "").strip()
    if not checkpoint_path:
        raise ValueError("checkpoint.path is required")
    checkpoint = CheckpointConfig(
        path=checkpoint_path,
        alias=str(
            checkpoint_raw.get("alias") or Path(checkpoint_path).name or "checkpoint"
        ).strip(),
        prompt_variant=(
            str(checkpoint_raw["prompt_variant"]).strip()
            if checkpoint_raw.get("prompt_variant") is not None
            else None
        ),
        object_field_order=(
            str(checkpoint_raw["object_field_order"]).strip()
            if checkpoint_raw.get("object_field_order") is not None
            else None
        ),
    )

    decode = DecodeConfig(
        temperatures=_ensure_tuple_float(
            decode_raw.get("temperatures"),
            default=(0.0, 0.1, 0.3, 0.7),
        ),
        top_p=float(decode_raw.get("top_p", 0.95)),
        max_new_tokens=int(decode_raw.get("max_new_tokens", 3084)),
        repetition_penalty=float(decode_raw.get("repetition_penalty", 1.05)),
        batch_size=int(decode_raw.get("batch_size", 8)),
        tensor_parallel_size=int(decode_raw.get("tensor_parallel_size", 1)),
        gpu_memory_utilization=float(decode_raw.get("gpu_memory_utilization", 0.8)),
        max_model_len=int(decode_raw.get("max_model_len", 14000)),
        max_num_seqs=int(decode_raw.get("max_num_seqs", 64)),
        enforce_eager=bool(decode_raw.get("enforce_eager", True)),
        seed=int(decode_raw.get("seed", 42)),
        local_radius_px=float(decode_raw.get("local_radius_px", 24.0)),
        local_radius_scale=float(decode_raw.get("local_radius_scale", 1.5)),
    )

    prefix_directions = _ensure_tuple_str(
        prefix_raw.get("directions"),
        default=("right", "down"),
    )
    if any(direction not in _VALID_DIRECTIONS for direction in prefix_directions):
        raise ValueError("prefix_probe.directions contains unsupported values")
    prefix_probe = PrefixProbeConfig(
        max_samples=int(prefix_raw.get("max_samples", 4)),
        prefix_lengths=_ensure_tuple_int(prefix_raw.get("prefix_lengths"), default=(1, 2)),
        jitter_pixels=_ensure_tuple_int(prefix_raw.get("jitter_pixels"), default=(8,)),
        directions=prefix_directions,
        device=str(prefix_raw.get("device", "cuda:0")).strip(),
        temperature=float(prefix_raw.get("temperature", 0.0)),
        top_p=float(prefix_raw.get("top_p", 1.0)),
        max_new_tokens=int(prefix_raw.get("max_new_tokens", 512)),
        repetition_penalty=float(prefix_raw.get("repetition_penalty", 1.0)),
        seed=int(prefix_raw.get("seed", 13)),
    )

    counterfactual_directions = _ensure_tuple_str(
        counterfactual_raw.get("directions"),
        default=("left", "right", "up", "down"),
    )
    if any(direction not in _VALID_DIRECTIONS for direction in counterfactual_directions):
        raise ValueError("counterfactual.directions contains unsupported values")
    counterfactual = CounterfactualConfig(
        max_samples=int(counterfactual_raw.get("max_samples", 6)),
        jitter_pixels=_ensure_tuple_int(
            counterfactual_raw.get("jitter_pixels"),
            default=(4, 8, 16),
        ),
        directions=counterfactual_directions,
        device=str(counterfactual_raw.get("device", "cuda:0")).strip(),
        attn_implementation=str(
            counterfactual_raw.get("attn_implementation", "auto")
        ).strip(),
        mask_fill=int(counterfactual_raw.get("mask_fill", 127)),
    )

    eval_cfg = EvalConfig(
        semantic_model=str(
            eval_raw.get("semantic_model") or "model_cache/all-MiniLM-L6-v2-local"
        ).strip(),
        semantic_threshold=float(eval_raw.get("semantic_threshold", 0.5)),
        semantic_device=str(eval_raw.get("semantic_device", "cuda:0")).strip(),
        semantic_batch_size=int(eval_raw.get("semantic_batch_size", 64)),
        num_workers=int(eval_raw.get("num_workers", 8)),
        f1ish_iou_thrs=_ensure_tuple_float(
            eval_raw.get("f1ish_iou_thrs"),
            default=(0.3, 0.5),
        ),
        f1ish_pred_scope=str(eval_raw.get("f1ish_pred_scope", "all")).strip(),
        use_segm=bool(eval_raw.get("use_segm", False)),
    )

    return StudyConfig(
        run=run,
        subset=subset,
        checkpoint=checkpoint,
        decode=decode,
        prefix_probe=prefix_probe,
        counterfactual=counterfactual,
        eval=eval_cfg,
    )


def _stage_enabled(config: StudyConfig, stage_name: str) -> bool:
    return stage_name in set(config.run.stages)


def _normalize_monitor_stats(sample: Mapping[str, Any]) -> Dict[str, Any]:
    stats = sample.get("stats")
    duplication = sample.get("duplication")
    if not isinstance(stats, Mapping):
        stats = {}
    if not isinstance(duplication, Mapping):
        duplication = {}
    return {
        "stats": dict(stats),
        "duplication": dict(duplication),
    }


def monitor_sort_key(sample: Mapping[str, Any]) -> Tuple[int, int, int, int, float, float, int, int]:
    normalized = _normalize_monitor_stats(sample)
    duplication = normalized["duplication"]
    stats = normalized["stats"]
    return (
        int(duplication.get("duplicates", 0) or 0),
        int(duplication.get("near_iou90_pairs_same_desc_count", 0) or 0),
        int(duplication.get("duplicate_bursts", 0) or 0),
        int(duplication.get("near_iou90_pairs_any_desc_count", 0) or 0),
        float(duplication.get("saturation_rate", 0.0) or 0.0),
        float(duplication.get("max_desc_count", 0.0) or 0.0),
        int(stats.get("fp_count", 0) or 0),
        int(stats.get("raw_valid_pred_objects", 0) or 0),
    )


def _iter_monitor_rows(
    *,
    monitor_path: Path,
    required_selection: Optional[str],
    min_fp_count: int,
) -> Iterable[Dict[str, Any]]:
    payload = json.loads(monitor_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return []
    meta = payload.get("meta")
    if not isinstance(meta, Mapping):
        meta = {}
    selection = str(meta.get("selection") or "").strip()
    if required_selection and selection != required_selection:
        return []
    samples = payload.get("samples")
    if not isinstance(samples, list):
        return []
    out: List[Dict[str, Any]] = []
    for rank, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            continue
        base_idx = sample.get("base_idx")
        try:
            base_idx_int = int(base_idx)
        except (TypeError, ValueError):
            continue
        stats = dict((sample.get("stats") or {})) if isinstance(sample.get("stats"), Mapping) else {}
        duplication = (
            dict((sample.get("duplication") or {}))
            if isinstance(sample.get("duplication"), Mapping)
            else {}
        )
        fp_count = int(stats.get("fp_count", 0) or 0)
        if fp_count < int(min_fp_count):
            continue
        out.append(
            {
                "base_idx": base_idx_int,
                "monitor_path": str(monitor_path),
                "rank_in_dump": int(rank),
                "image_id": sample.get("image_id"),
                "images": list(sample.get("images") or [])
                if isinstance(sample.get("images"), list)
                else [],
                "width": sample.get("width"),
                "height": sample.get("height"),
                "stats": stats,
                "duplication": duplication,
            }
        )
    return out


def _load_jsonl_records_by_indices(
    *,
    jsonl_path: Path,
    indices: Sequence[int],
) -> Dict[int, Dict[str, Any]]:
    wanted = {int(idx) for idx in indices}
    found: Dict[int, Dict[str, Any]] = {}
    if not wanted:
        return found
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if int(line_idx) not in wanted:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Non-object record at {jsonl_path}:{line_idx}")
            found[int(line_idx)] = record
            if len(found) >= len(wanted):
                break
    missing = sorted(wanted.difference(found))
    if missing:
        raise IndexError(f"Missing indices from {jsonl_path}: {missing[:16]}")
    return found


def _object_area_frac(
    *,
    bbox_norm1000: Sequence[int],
) -> float:
    if len(bbox_norm1000) != 4:
        return 0.0
    x1, y1, x2, y2 = [int(v) for v in bbox_norm1000]
    width = max(0.0, float(x2 - x1) / 1000.0)
    height = max(0.0, float(y2 - y1) / 1000.0)
    return float(width * height)


def count_small_gt_objects(
    record: Mapping[str, Any],
    *,
    max_area_frac: float,
) -> int:
    total = 0
    for obj in list(record.get("objects") or []):
        if not isinstance(obj, Mapping):
            continue
        bbox_norm = _extract_dataset_gt_bbox(obj)
        if bbox_norm is None:
            continue
        if _object_area_frac(bbox_norm1000=bbox_norm) <= float(max_area_frac):
            total += 1
    return int(total)


def materialize_monitor_subset(
    config: StudyConfig,
    *,
    run_dir: Path,
) -> Tuple[Path, List[Dict[str, Any]], Dict[str, Any]]:
    dataset_path = _resolve_existing_input_path(config.subset.dataset_jsonl)
    root_image_dir = (
        _resolve_existing_input_path(config.subset.root_image_dir)
        if config.subset.root_image_dir
        else dataset_path.parent.resolve()
    )
    monitor_paths = [
        _resolve_existing_input_path(path_raw)
        for path_raw in config.subset.monitor_dump_paths
    ]
    best_by_base_idx: Dict[int, Dict[str, Any]] = {}
    for monitor_path in monitor_paths:
        for row in _iter_monitor_rows(
            monitor_path=monitor_path,
            required_selection=config.subset.require_selection,
            min_fp_count=config.subset.min_fp_count,
        ):
            base_idx = int(row["base_idx"])
            current = best_by_base_idx.get(base_idx)
            if current is None or monitor_sort_key(row) > monitor_sort_key(current):
                best_by_base_idx[base_idx] = row
    selected_rows = sorted(
        best_by_base_idx.values(),
        key=monitor_sort_key,
        reverse=True,
    )
    record_map = _load_jsonl_records_by_indices(
        jsonl_path=dataset_path,
        indices=[int(row["base_idx"]) for row in selected_rows],
    )
    materialized_rows: List[Dict[str, Any]] = []
    subset_records: List[Dict[str, Any]] = []
    for row in selected_rows:
        base_idx = int(row["base_idx"])
        record = dict(record_map[base_idx])
        small_gt_count = count_small_gt_objects(
            record,
            max_area_frac=config.subset.small_object_max_area_frac,
        )
        if config.subset.require_small_object and small_gt_count <= 0:
            continue
        images = list(record.get("images") or [])
        materialized_rows.append(
            {
                **row,
                "small_gt_count": int(small_gt_count),
                "image": str(images[0]) if images else "",
                "gt_count": int(len(record.get("objects") or [])),
            }
        )
        subset_records.append(record)
        if len(subset_records) >= int(config.subset.max_samples):
            break

    subset_dir = run_dir / "subset"
    subset_dir.mkdir(parents=True, exist_ok=True)
    subset_path = subset_dir / "monitor_subset.coord.jsonl"
    with subset_path.open("w", encoding="utf-8") as f:
        for record in subset_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    meta = {
        "dataset_jsonl": str(dataset_path),
        "root_image_dir": str(root_image_dir),
        "monitor_dump_paths": [str(path) for path in monitor_paths],
        "selected_count": int(len(subset_records)),
        "rows": materialized_rows,
    }
    (subset_dir / "monitor_subset.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return subset_path, subset_records, meta


def _entropy_from_counts(counts: Sequence[int]) -> float:
    total = float(sum(int(v) for v in counts if int(v) > 0))
    if total <= 0.0:
        return 0.0
    acc = 0.0
    for value in counts:
        v = int(value)
        if v <= 0:
            continue
        p = float(v) / total
        acc -= p * math.log(p)
    return float(acc)


def _extract_bbox_px(
    obj: Mapping[str, Any],
    *,
    width: int,
    height: int,
) -> Optional[List[float]]:
    bbox_raw = obj.get("bbox")
    if isinstance(bbox_raw, list) and len(bbox_raw) == 4:
        try:
            return [float(v) for v in bbox_raw]
        except (TypeError, ValueError):
            return None
    bbox_raw = obj.get("bbox_2d")
    if isinstance(bbox_raw, list) and len(bbox_raw) == 4:
        try:
            values = [float(v) for v in bbox_raw]
        except (TypeError, ValueError):
            return None
        if any(value > 1000.0 or value < 0.0 for value in values):
            return values
        return [float(v) for v in _norm1000_to_pixel(values, width, height)]
    points = obj.get("points")
    obj_type = str(obj.get("type") or "").strip().lower()
    if obj_type == "bbox_2d" and isinstance(points, list):
        if len(points) == 4:
            try:
                return [float(v) for v in points]
            except (TypeError, ValueError):
                return None
        if len(points) >= 8:
            try:
                xs = [float(points[idx]) for idx in range(0, min(len(points), 8), 2)]
                ys = [float(points[idx]) for idx in range(1, min(len(points), 8), 2)]
            except (TypeError, ValueError):
                return None
            return [min(xs), min(ys), max(xs), max(ys)]
    return None


def _bbox_center_and_scale(bbox_px: Sequence[float]) -> Tuple[float, float, float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in bbox_px]
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = x1 + (0.5 * w)
    cy = y1 + (0.5 * h)
    return cx, cy, w, h, max(w, h)


def _center_distance(bbox_a: Sequence[float], bbox_b: Sequence[float]) -> float:
    ax, ay, *_rest_a = _bbox_center_and_scale(bbox_a)
    bx, by, *_rest_b = _bbox_center_and_scale(bbox_b)
    return float(math.hypot(ax - bx, ay - by))


def _pairwise_duplicate_metrics(
    preds: Sequence[Mapping[str, Any]],
    *,
    width: int,
    height: int,
    local_radius_px: float,
    local_radius_scale: float,
) -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = []
    for pred_idx, pred in enumerate(preds):
        if not isinstance(pred, Mapping):
            continue
        desc = normalize_desc(str(pred.get("desc") or ""))
        bbox_px = _extract_bbox_px(pred, width=width, height=height)
        if not desc or bbox_px is None:
            continue
        entries.append(
            {
                "pred_idx": int(pred_idx),
                "desc": desc,
                "bbox_px": [float(v) for v in bbox_px],
            }
        )
    counts: Dict[str, Any] = {}
    desc_counter: Dict[str, int] = {}
    graph: Dict[int, set[int]] = {}
    for entry in entries:
        desc_counter[entry["desc"]] = int(desc_counter.get(entry["desc"], 0) + 1)
        graph[int(entry["pred_idx"])] = set()
    for idx, left in enumerate(entries):
        bbox_left = left["bbox_px"]
        desc_left = left["desc"]
        _cx_l, _cy_l, _w_l, _h_l, scale_l = _bbox_center_and_scale(bbox_left)
        for right in entries[idx + 1 :]:
            if desc_left != right["desc"]:
                continue
            bbox_right = right["bbox_px"]
            _cx_r, _cy_r, _w_r, _h_r, scale_r = _bbox_center_and_scale(bbox_right)
            iou = float(_bbox_iou_xyxy(bbox_left, bbox_right))
            if iou >= 0.5:
                counts["same_desc_iou50_pairs"] = int(
                    counts.get("same_desc_iou50_pairs", 0) + 1
                )
            if iou >= 0.7:
                counts["same_desc_iou70_pairs"] = int(
                    counts.get("same_desc_iou70_pairs", 0) + 1
                )
            if iou >= 0.9:
                counts["same_desc_iou90_pairs"] = int(
                    counts.get("same_desc_iou90_pairs", 0) + 1
                )
            radius = max(float(local_radius_px), float(local_radius_scale) * max(scale_l, scale_r))
            if _center_distance(bbox_left, bbox_right) <= radius:
                counts["same_desc_local_pairs"] = int(
                    counts.get("same_desc_local_pairs", 0) + 1
                )
                graph[int(left["pred_idx"])].add(int(right["pred_idx"]))
                graph[int(right["pred_idx"])].add(int(left["pred_idx"]))
    max_cluster = 0
    visited: set[int] = set()
    for node in list(graph.keys()):
        if node in visited:
            continue
        stack = [node]
        size = 0
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            size += 1
            stack.extend(int(v) for v in graph.get(current, set()) if int(v) not in visited)
        max_cluster = max(max_cluster, size)
    counts["max_desc_count"] = int(max(desc_counter.values()) if desc_counter else 0)
    counts["desc_entropy"] = float(_entropy_from_counts(desc_counter.values()))
    counts["unique_desc_count"] = int(len(desc_counter))
    counts["max_local_same_desc_cluster_size"] = int(max_cluster)
    return counts


def _gt_area_frac(
    gt_obj: Mapping[str, Any],
) -> float:
    bbox_norm = _extract_dataset_gt_bbox(gt_obj)
    if bbox_norm is None:
        return 0.0
    return float(_object_area_frac(bbox_norm1000=bbox_norm))


def _first_small_match_spillover(
    *,
    preds: Sequence[Mapping[str, Any]],
    gts: Sequence[Mapping[str, Any]],
    match_row: Mapping[str, Any],
    width: int,
    height: int,
    small_object_max_area_frac: float,
    local_radius_px: float,
    local_radius_scale: float,
) -> Dict[str, Any]:
    matches = list(match_row.get("matches") or [])
    if not matches:
        return {
            "first_small_match_pred_idx": None,
            "first_small_match_gt_idx": None,
            "first_small_match_desc": None,
            "first_small_match_spill_count": 0,
        }
    anchor_match: Optional[Mapping[str, Any]] = None
    for match in sorted(matches, key=lambda row: int(row.get("pred_idx", 10**9))):
        gt_idx = int(match.get("gt_idx", -1))
        if gt_idx < 0 or gt_idx >= len(gts):
            continue
        if _gt_area_frac(gts[gt_idx]) <= float(small_object_max_area_frac):
            anchor_match = match
            break
    if anchor_match is None:
        return {
            "first_small_match_pred_idx": None,
            "first_small_match_gt_idx": None,
            "first_small_match_desc": None,
            "first_small_match_spill_count": 0,
        }
    pred_idx = int(anchor_match.get("pred_idx", -1))
    gt_idx = int(anchor_match.get("gt_idx", -1))
    if pred_idx < 0 or pred_idx >= len(preds):
        return {
            "first_small_match_pred_idx": None,
            "first_small_match_gt_idx": None,
            "first_small_match_desc": None,
            "first_small_match_spill_count": 0,
        }
    anchor_pred = preds[pred_idx]
    anchor_bbox = _extract_bbox_px(anchor_pred, width=width, height=height)
    anchor_desc = normalize_desc(str(anchor_pred.get("desc") or ""))
    if anchor_bbox is None or not anchor_desc:
        return {
            "first_small_match_pred_idx": None,
            "first_small_match_gt_idx": None,
            "first_small_match_desc": None,
            "first_small_match_spill_count": 0,
        }
    _cx, _cy, _w, _h, scale = _bbox_center_and_scale(anchor_bbox)
    radius = max(float(local_radius_px), float(local_radius_scale) * scale)
    spill_count = 0
    for later_pred in preds[pred_idx + 1 :]:
        if normalize_desc(str(later_pred.get("desc") or "")) != anchor_desc:
            continue
        later_bbox = _extract_bbox_px(later_pred, width=width, height=height)
        if later_bbox is None:
            continue
        if _center_distance(anchor_bbox, later_bbox) <= radius:
            spill_count += 1
    return {
        "first_small_match_pred_idx": int(pred_idx),
        "first_small_match_gt_idx": int(gt_idx),
        "first_small_match_desc": anchor_desc,
        "first_small_match_spill_count": int(spill_count),
    }


def summarize_decode_run(
    *,
    gt_vs_pred_path: Path,
    matches_path: Path,
    subset_meta: Mapping[str, Any],
    config: StudyConfig,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    gt_rows = [
        json.loads(line)
        for line in gt_vs_pred_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    match_rows = [
        json.loads(line)
        for line in matches_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    meta_rows = list(subset_meta.get("rows") or [])
    per_sample: List[Dict[str, Any]] = []
    for sample_idx, gt_row in enumerate(gt_rows):
        match_row = match_rows[sample_idx] if sample_idx < len(match_rows) else {}
        meta_row = meta_rows[sample_idx] if sample_idx < len(meta_rows) else {}
        preds = list(gt_row.get("pred") or [])
        gts = list(gt_row.get("gt") or [])
        width = int(gt_row.get("width") or 0)
        height = int(gt_row.get("height") or 0)
        matched = int(len(match_row.get("matches") or []))
        fp_count = int(len(match_row.get("unmatched_pred_indices") or []))
        fn_count = int(len(match_row.get("unmatched_gt_indices") or []))
        precision = float(matched / max(1, matched + fp_count))
        recall = float(matched / max(1, matched + fn_count))
        duplicate_metrics = _pairwise_duplicate_metrics(
            preds,
            width=width,
            height=height,
            local_radius_px=config.decode.local_radius_px,
            local_radius_scale=config.decode.local_radius_scale,
        )
        spillover_metrics = _first_small_match_spillover(
            preds=preds,
            gts=gts,
            match_row=match_row,
            width=width,
            height=height,
            small_object_max_area_frac=config.subset.small_object_max_area_frac,
            local_radius_px=config.decode.local_radius_px,
            local_radius_scale=config.decode.local_radius_scale,
        )
        desc_counts: Dict[str, int] = {}
        for pred in preds:
            desc = normalize_desc(str(pred.get("desc") or ""))
            if desc:
                desc_counts[desc] = int(desc_counts.get(desc, 0) + 1)
        top_desc = max(desc_counts.items(), key=lambda item: item[1])[0] if desc_counts else None
        sample_out = {
            "sample_index": int(sample_idx),
            "base_idx": meta_row.get("base_idx"),
            "image": meta_row.get("image") or gt_row.get("image"),
            "gt_count": int(len(gts)),
            "pred_count": int(len(preds)),
            "matched": int(matched),
            "fp_count": int(fp_count),
            "fn_count": int(fn_count),
            "precision": float(precision),
            "recall": float(recall),
            "top_desc": top_desc,
            **duplicate_metrics,
            **spillover_metrics,
        }
        per_sample.append(sample_out)
    summary = {
        "num_samples": int(len(per_sample)),
        "mean_pred_count": float(mean(row["pred_count"] for row in per_sample))
        if per_sample
        else 0.0,
        "mean_fp_count": float(mean(row["fp_count"] for row in per_sample))
        if per_sample
        else 0.0,
        "mean_precision": float(mean(row["precision"] for row in per_sample))
        if per_sample
        else 0.0,
        "mean_recall": float(mean(row["recall"] for row in per_sample))
        if per_sample
        else 0.0,
        "mean_same_desc_local_pairs": float(
            mean(row.get("same_desc_local_pairs", 0) for row in per_sample)
        )
        if per_sample
        else 0.0,
        "mean_same_desc_iou90_pairs": float(
            mean(row.get("same_desc_iou90_pairs", 0) for row in per_sample)
        )
        if per_sample
        else 0.0,
        "spillover_rate": float(
            sum(
                1
                for row in per_sample
                if int(row.get("first_small_match_spill_count", 0) or 0) > 0
            )
            / max(1, len(per_sample))
        ),
    }
    return per_sample, summary


def _temp_slug(value: float) -> str:
    text = f"{float(value):.2f}".rstrip("0").rstrip(".")
    return f"temp_{text.replace('-', 'm').replace('.', 'p')}"


def _resolved_checkpoint(
    *,
    config: StudyConfig,
) -> Tuple[Path, str, str, ResolvedCheckpoint]:
    checkpoint_path, path_source = resolve_checkpoint_path(config.checkpoint.path)
    prompt_variant, object_field_order, prompt_source = resolve_prompt_controls_for_checkpoint(
        checkpoint_path,
        default_prompt_variant="coco_80",
        default_object_field_order="desc_first",
        override_prompt_variant=config.checkpoint.prompt_variant,
        override_object_field_order=config.checkpoint.object_field_order,
    )
    resolved = ResolvedCheckpoint(
        alias=config.checkpoint.alias,
        path=checkpoint_path,
        resolve_source=path_source,
        artifact_kind="executable_checkpoint",
        fingerprint="diagnostic",
        prompt_variant=str(prompt_variant),
        object_field_order=str(object_field_order),
        prompt_control_source=str(prompt_source),
        provenance_sidecars={},
    )
    return checkpoint_path, str(prompt_variant), str(object_field_order), resolved


def run_decode_temperature_sweep(
    config: StudyConfig,
    *,
    run_dir: Path,
    subset_path: Path,
    subset_records: Sequence[Mapping[str, Any]],
    subset_meta: Mapping[str, Any],
    root_image_dir: Path,
) -> Dict[str, Any]:
    checkpoint_path, prompt_variant, object_field_order, _resolved = _resolved_checkpoint(
        config=config
    )
    decode_dir = run_dir / "decode"
    decode_dir.mkdir(parents=True, exist_ok=True)
    aggregate_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    for temperature in config.decode.temperatures:
        temp_dir = decode_dir / _temp_slug(temperature)
        temp_dir.mkdir(parents=True, exist_ok=True)
        gt_vs_pred_path = temp_dir / "gt_vs_pred.jsonl"
        trace_path = temp_dir / "pred_token_trace.jsonl"
        summary_path = temp_dir / "summary.json"
        collect_stage2_parity_gt_vs_pred(
            jsonl_path=subset_path,
            records=subset_records,
            root_image_dir=root_image_dir,
            checkpoint_path=checkpoint_path,
            prompt_variant=prompt_variant,
            object_field_order=object_field_order,
            temperature=float(temperature),
            top_p=float(config.decode.top_p),
            repetition_penalty=float(config.decode.repetition_penalty),
            max_new_tokens=int(config.decode.max_new_tokens),
            batch_size=int(config.decode.batch_size),
            tensor_parallel_size=int(config.decode.tensor_parallel_size),
            gpu_memory_utilization=float(config.decode.gpu_memory_utilization),
            max_model_len=int(config.decode.max_model_len),
            max_num_seqs=int(config.decode.max_num_seqs),
            enforce_eager=bool(config.decode.enforce_eager),
            seed=int(config.decode.seed),
            out_path=gt_vs_pred_path,
            pred_token_trace_path=trace_path,
            summary_path=summary_path,
        )
        _run_detection_eval(
            pred_path=gt_vs_pred_path,
            eval_cfg=config.eval,
            out_dir=temp_dir / "eval",
        )
        per_sample, summary = summarize_decode_run(
            gt_vs_pred_path=gt_vs_pred_path,
            matches_path=temp_dir / "eval" / "matches.jsonl",
            subset_meta=subset_meta,
            config=config,
        )
        for row in per_sample:
            aggregate_rows.append({"temperature": float(temperature), **row})
        summary_out = {"temperature": float(temperature), **summary}
        summaries.append(summary_out)
        (temp_dir / "per_sample_metrics.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in per_sample),
            encoding="utf-8",
        )
        (temp_dir / "duplication_summary.json").write_text(
            json.dumps(summary_out, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    manifest = {
        "decode_dir": str(decode_dir),
        "summaries": summaries,
        "summary_by_temperature": {
            _temp_slug(row["temperature"]): row for row in summaries
        },
    }
    (decode_dir / "aggregate_metrics.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in aggregate_rows),
        encoding="utf-8",
    )
    (decode_dir / "decode_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _direction_delta(direction: str, pixels: int) -> Tuple[int, int]:
    if direction == "left":
        return -int(pixels), 0
    if direction == "right":
        return int(pixels), 0
    if direction == "up":
        return 0, -int(pixels)
    if direction == "down":
        return 0, int(pixels)
    raise ValueError(f"Unsupported direction: {direction}")


def shift_bbox_xyxy(
    bbox_px: Sequence[float],
    *,
    dx: int,
    dy: int,
    width: int,
    height: int,
) -> List[int]:
    x1, y1, x2, y2 = [float(v) for v in bbox_px]
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    nx1 = min(max(0.0, x1 + float(dx)), max(0.0, float(width) - box_w))
    ny1 = min(max(0.0, y1 + float(dy)), max(0.0, float(height) - box_h))
    nx2 = nx1 + box_w
    ny2 = ny1 + box_h
    return [int(round(nx1)), int(round(ny1)), int(round(nx2)), int(round(ny2))]


def find_small_object_anchor_index(
    record: Mapping[str, Any],
    *,
    max_area_frac: float,
) -> Optional[int]:
    objects = list(record.get("objects") or [])
    for idx, obj in enumerate(objects[:-1]):
        if not isinstance(obj, Mapping):
            continue
        if _gt_area_frac(obj) <= float(max_area_frac):
            return int(idx)
    return None


def _serialize_gt_prefix(
    *,
    gt_objects: Sequence[Mapping[str, Any]],
    prefix_length: int,
    width: int,
    height: int,
    object_field_order: str,
) -> Optional[str]:
    if prefix_length <= 0:
        return None
    prefix_objects = list(gt_objects[:prefix_length])
    prefix_text, _compact = _serialize_objects_to_prefix_text(
        prefix_objects,
        width=width,
        height=height,
        object_field_order=object_field_order,
    )
    return prefix_text


def _serialize_pred_prefix(
    *,
    pred_objects: Sequence[Mapping[str, Any]],
    prefix_length: int,
    width: int,
    height: int,
    object_field_order: str,
) -> Optional[str]:
    if prefix_length <= 0 or len(pred_objects) < prefix_length:
        return None
    prefix_text, _compact = _serialize_objects_to_prefix_text(
        list(pred_objects[:prefix_length]),
        width=width,
        height=height,
        object_field_order=object_field_order,
    )
    return prefix_text


def _jitter_last_prefix_object(
    *,
    pred_objects: Sequence[Mapping[str, Any]],
    prefix_length: int,
    width: int,
    height: int,
    object_field_order: str,
    direction: str,
    pixels: int,
) -> Optional[str]:
    if prefix_length <= 0 or len(pred_objects) < prefix_length:
        return None
    prefix_copy = [dict(obj) for obj in list(pred_objects[:prefix_length])]
    last_bbox = _extract_bbox_px(prefix_copy[-1], width=width, height=height)
    if last_bbox is None:
        return None
    dx, dy = _direction_delta(direction, pixels)
    jitter_bbox = shift_bbox_xyxy(
        last_bbox,
        dx=dx,
        dy=dy,
        width=width,
        height=height,
    )
    prefix_copy[-1]["bbox"] = jitter_bbox
    prefix_copy[-1]["bbox_2d"] = jitter_bbox
    prefix_text, _compact = _serialize_objects_to_prefix_text(
        prefix_copy,
        width=width,
        height=height,
        object_field_order=object_field_order,
    )
    return prefix_text


def _extract_continuation_metrics(
    *,
    generation: RolloutGeneration,
    preds: Sequence[Mapping[str, Any]],
    width: int,
    height: int,
    prefix_last_desc: Optional[str],
    config: StudyConfig,
) -> Dict[str, Any]:
    duplicate_metrics = _pairwise_duplicate_metrics(
        preds,
        width=width,
        height=height,
        local_radius_px=config.decode.local_radius_px,
        local_radius_scale=config.decode.local_radius_scale,
    )
    first_desc = None
    if preds:
        first_desc = normalize_desc(str((preds[0] or {}).get("desc") or ""))
    continuation_same_as_prefix_last = 0
    if prefix_last_desc:
        continuation_same_as_prefix_last = int(
            sum(
                1
                for pred in preds
                if normalize_desc(str(pred.get("desc") or "")) == prefix_last_desc
            )
        )
    return {
        "generated_token_count": int(generation.generated_token_count),
        "finish_reason": str(generation.finish_reason),
        "continuation_pred_count": int(len(preds)),
        "continuation_first_desc": first_desc,
        "continuation_same_as_prefix_last_count": int(continuation_same_as_prefix_last),
        **duplicate_metrics,
    }


def run_prefix_probe(
    config: StudyConfig,
    *,
    run_dir: Path,
    subset_path: Path,
    subset_records: Sequence[Mapping[str, Any]],
    root_image_dir: Path,
) -> Dict[str, Any]:
    checkpoint_path, prompt_variant, object_field_order, resolved_checkpoint = _resolved_checkpoint(
        config=config
    )
    del checkpoint_path, prompt_variant
    baseline_path = (
        run_dir
        / "decode"
        / _temp_slug(0.0)
        / "gt_vs_pred.jsonl"
    )
    baseline_rows = _load_jsonl(baseline_path)
    if not baseline_rows:
        raise FileNotFoundError(
            f"Prefix probe requires greedy baseline outputs at {baseline_path}"
        )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    runner = HFStudyRunner(
        checkpoint=resolved_checkpoint,
        device=config.prefix_probe.device,
        image_root=root_image_dir,
    )
    prefix_dir = run_dir / "prefix_probe"
    prefix_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    eligible = 0
    for sample_idx, (record, baseline_row) in enumerate(zip(subset_records, baseline_rows)):
        if eligible >= int(config.prefix_probe.max_samples):
            break
        width = int(record.get("width") or baseline_row.get("width") or 0)
        height = int(record.get("height") or baseline_row.get("height") or 0)
        loaded = _load_image(record, root_image_dir=root_image_dir, subset_path=subset_path)
        baseline_preds = list(baseline_row.get("pred") or [])
        gt_objects = list(record.get("objects") or [])
        if not baseline_preds and not gt_objects:
            continue
        eligible += 1
        for prefix_length in config.prefix_probe.prefix_lengths:
            conditions: List[Tuple[str, Optional[str], Optional[str]]] = []
            self_prefix = _serialize_pred_prefix(
                pred_objects=baseline_preds,
                prefix_length=int(prefix_length),
                width=width,
                height=height,
                object_field_order=object_field_order,
            )
            if self_prefix is not None:
                last_desc = normalize_desc(
                    str((baseline_preds[prefix_length - 1] or {}).get("desc") or "")
                )
                conditions.append(("self_prefix", self_prefix, last_desc))
                for pixels in config.prefix_probe.jitter_pixels:
                    for direction in config.prefix_probe.directions:
                        jittered = _jitter_last_prefix_object(
                            pred_objects=baseline_preds,
                            prefix_length=int(prefix_length),
                            width=width,
                            height=height,
                            object_field_order=object_field_order,
                            direction=direction,
                            pixels=int(pixels),
                        )
                        if jittered is not None:
                            conditions.append(
                                (
                                    f"self_prefix_jitter_{direction}_{int(pixels)}",
                                    jittered,
                                    last_desc,
                                )
                            )
            gt_prefix = _serialize_gt_prefix(
                gt_objects=gt_objects,
                prefix_length=int(prefix_length),
                width=width,
                height=height,
                object_field_order=object_field_order,
            )
            if gt_prefix is not None:
                gt_last_desc = normalize_desc(
                    str((gt_objects[prefix_length - 1] or {}).get("desc") or "")
                )
                conditions.append(("gt_prefix", gt_prefix, gt_last_desc))

            gen_cfg = GenerationConfig(
                temperature=float(config.prefix_probe.temperature),
                top_p=float(config.prefix_probe.top_p),
                max_new_tokens=int(config.prefix_probe.max_new_tokens),
                repetition_penalty=float(config.prefix_probe.repetition_penalty),
                batch_size=1,
                seed=int(config.prefix_probe.seed),
            )
            seen_conditions: set[str] = set()
            for condition_name, prefix_text, prefix_last_desc in conditions:
                if prefix_text is None or condition_name in seen_conditions:
                    continue
                seen_conditions.add(condition_name)
                generation = runner.generate_with_prefix(
                    image=loaded.image,
                    prefix_text=prefix_text,
                    gen_cfg=gen_cfg,
                )
                parsed_preds, pred_errors, _raw_json, _special_tokens, _ends = runner.parse_prediction(
                    raw_text=generation.raw_text,
                    width=width,
                    height=height,
                )
                rows.append(
                    {
                        "sample_index": int(sample_idx),
                        "image": str((record.get("images") or [""])[0]),
                        "prefix_length": int(prefix_length),
                        "condition": condition_name,
                        "pred_errors": list(pred_errors),
                        **_extract_continuation_metrics(
                            generation=generation,
                            preds=parsed_preds,
                            width=width,
                            height=height,
                            prefix_last_desc=prefix_last_desc,
                            config=config,
                        ),
                    }
                )
    (prefix_dir / "prefix_probe_rows.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary: Dict[str, Dict[str, float]] = {}
    for condition in sorted({str(row["condition"]) for row in rows}):
        group = [row for row in rows if str(row["condition"]) == condition]
        summary[condition] = {
            "num_rows": float(len(group)),
            "mean_continuation_pred_count": float(
                mean(int(row["continuation_pred_count"]) for row in group)
            )
            if group
            else 0.0,
            "mean_same_as_prefix_last": float(
                mean(int(row["continuation_same_as_prefix_last_count"]) for row in group)
            )
            if group
            else 0.0,
            "mean_same_desc_local_pairs": float(
                mean(int(row.get("same_desc_local_pairs", 0)) for row in group)
            )
            if group
            else 0.0,
        }
    manifest = {
        "prefix_probe_dir": str(prefix_dir),
        "summary_by_condition": summary,
    }
    (prefix_dir / "prefix_probe_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    del runner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return manifest


def _score_positions(
    *,
    scorer: TeacherForcedScorer,
    prepared_full_text: str,
    full_input_ids: Sequence[int],
    image: Image.Image,
    positions_by_name: Mapping[str, Sequence[int]],
) -> Dict[str, float]:
    model_inputs = scorer.processor(
        text=[prepared_full_text],
        images=[image],
        return_tensors="pt",
        padding=True,
    )
    model_inputs = {
        key: value.to(scorer.device) if isinstance(value, torch.Tensor) else value
        for key, value in model_inputs.items()
    }
    with torch.inference_mode():
        outputs = scorer.model(**model_inputs, use_cache=False)
    logits = getattr(outputs, "logits", None)
    input_ids = model_inputs.get("input_ids")
    if not isinstance(logits, torch.Tensor) or not isinstance(input_ids, torch.Tensor):
        raise RuntimeError("teacher-forced scorer did not return logits/input_ids")
    padded_len = int(input_ids.shape[1])
    seq_len = int(len(full_input_ids))
    pad_offset = int(padded_len - seq_len)
    observed_ids = input_ids[0, pad_offset:].detach().cpu().tolist()
    if [int(v) for v in observed_ids] != [int(v) for v in full_input_ids]:
        raise RuntimeError("assistant_span_build_failed")
    out: Dict[str, float] = {}
    for name, positions in positions_by_name.items():
        values: List[float] = []
        for pos in positions:
            abs_pos = int(pad_offset + int(pos))
            if abs_pos <= 0 or abs_pos >= int(input_ids.shape[1]):
                continue
            prev_logits = logits[0, abs_pos - 1].float()
            target_id = int(input_ids[0, abs_pos].item())
            target_logit = float(prev_logits[target_id].detach().cpu().item())
            log_norm = float(torch.logsumexp(prev_logits, dim=-1).detach().cpu().item())
            value = float(target_logit - log_norm)
            if math.isfinite(value):
                values.append(value)
        out[name] = float(mean(values)) if values else float("nan")
    return out


def _candidate_token_positions(
    *,
    scorer: TeacherForcedScorer,
    assistant_text: str,
    object_field_order: str,
) -> Tuple[List[int], List[int], List[int]]:
    assistant_ids = scorer.tokenizer.encode(assistant_text, add_special_tokens=False)
    desc_spans = find_desc_value_token_positions_by_span(
        tokenizer=scorer.tokenizer,
        token_ids=assistant_ids,
    )
    parsed = parse_rollout_for_matching(
        tokenizer=scorer.tokenizer,
        response_token_ids=[int(tok) for tok in assistant_ids],
        object_field_order=object_field_order,
    )
    if not desc_spans or not parsed.valid_objects:
        raise ValueError("candidate_span_build_failed")
    candidate_desc_positions = [int(v) for v in desc_spans[-1]]
    candidate_coord_positions = [
        int(v) for v in parsed.valid_objects[-1].coord_token_indices
    ]
    full_positions = sorted(
        {
            int(v)
            for v in [*candidate_desc_positions, *candidate_coord_positions]
        }
    )
    return candidate_desc_positions, candidate_coord_positions, full_positions


def _build_candidate_row(
    *,
    scorer: TeacherForcedScorer,
    image: Image.Image,
    prefix_objects: Sequence[Mapping[str, Any]],
    candidate_object: Mapping[str, Any],
    width: int,
    height: int,
    object_field_order: str,
    prompt_variant: str,
    anchor_mask_bbox_px: Optional[Sequence[float]],
    candidate_mask_bbox_px: Optional[Sequence[float]],
    mask_fill: int,
) -> Dict[str, float]:
    all_objects = [*prefix_objects, candidate_object]
    gt_objects = []
    for idx, obj in enumerate(all_objects):
        bbox_px = _extract_bbox_px(obj, width=width, height=height)
        if bbox_px is None:
            raise ValueError("candidate_bbox_missing")
        bbox_norm = _pixel_to_norm1000(bbox_px, width, height)
        if bbox_norm is None:
            raise ValueError("candidate_bbox_norm_failed")
        gt_objects.append(_build_bbox_gt_object(idx, str(obj.get("desc") or ""), bbox_norm))
    assistant_text = _build_closed_container_text(
        gt_objects,
        object_field_order=object_field_order,
    )
    (
        candidate_desc_positions,
        candidate_coord_positions,
        candidate_full_positions,
    ) = _candidate_token_positions(
        scorer=scorer,
        assistant_text=assistant_text,
        object_field_order=object_field_order,
    )
    prepared = scorer.prepare_example(
        image=image,
        assistant_text=assistant_text,
        desc_positions_rel=[],
        prompt_variant=prompt_variant,
        object_field_order=object_field_order,
    )
    original_scores = _score_positions(
        scorer=scorer,
        prepared_full_text=prepared.full_text,
        full_input_ids=prepared.full_input_ids,
        image=image,
        positions_by_name={
            "desc_score": candidate_desc_positions,
            "coord_score": candidate_coord_positions,
            "full_score": candidate_full_positions,
        },
    )
    out = dict(original_scores)
    if anchor_mask_bbox_px is not None:
        anchor_masked = _mask_image(
            image,
            anchor_mask_bbox_px,
            fill=int(mask_fill),
        )
        masked_scores = _score_positions(
            scorer=scorer,
            prepared_full_text=prepared.full_text,
            full_input_ids=prepared.full_input_ids,
            image=anchor_masked,
            positions_by_name={"anchor_mask_full_score": candidate_full_positions},
        )
        out.update(masked_scores)
    if candidate_mask_bbox_px is not None:
        candidate_masked = _mask_image(
            image,
            candidate_mask_bbox_px,
            fill=int(mask_fill),
        )
        masked_scores = _score_positions(
            scorer=scorer,
            prepared_full_text=prepared.full_text,
            full_input_ids=prepared.full_input_ids,
            image=candidate_masked,
            positions_by_name={"candidate_mask_full_score": candidate_full_positions},
        )
        out.update(masked_scores)
    return out


def _record_gt_objects_with_bbox(
    record: Mapping[str, Any],
    *,
    width: int,
    height: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for obj in list(record.get("objects") or []):
        if not isinstance(obj, Mapping):
            continue
        desc = str(obj.get("desc") or "").strip()
        bbox_norm = _extract_dataset_gt_bbox(obj)
        if not desc or bbox_norm is None:
            continue
        bbox_px = _norm1000_to_pixel(bbox_norm, width, height)
        out.append(
            {
                "desc": desc,
                "bbox": [int(v) for v in bbox_px],
                "bbox_2d": [int(v) for v in bbox_px],
            }
        )
    return out


def run_counterfactual_probe(
    config: StudyConfig,
    *,
    run_dir: Path,
    subset_path: Path,
    subset_records: Sequence[Mapping[str, Any]],
    root_image_dir: Path,
) -> Dict[str, Any]:
    checkpoint_path, prompt_variant, object_field_order, _resolved = _resolved_checkpoint(
        config=config
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    scorer = TeacherForcedScorer(
        checkpoint_path=checkpoint_path,
        device=config.counterfactual.device,
        attn_implementation=config.counterfactual.attn_implementation,
    )
    counterfactual_dir = run_dir / "counterfactual"
    counterfactual_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    used = 0
    for sample_idx, record in enumerate(subset_records):
        if used >= int(config.counterfactual.max_samples):
            break
        width = int(record.get("width") or 0)
        height = int(record.get("height") or 0)
        gt_objects = _record_gt_objects_with_bbox(record, width=width, height=height)
        anchor_idx = find_small_object_anchor_index(
            record,
            max_area_frac=config.subset.small_object_max_area_frac,
        )
        if anchor_idx is None or anchor_idx + 1 >= len(gt_objects):
            continue
        used += 1
        loaded = _load_image(record, root_image_dir=root_image_dir, subset_path=subset_path)
        prefix_objects = list(gt_objects[: anchor_idx + 1])
        anchor_object = prefix_objects[-1]
        gt_next_object = dict(gt_objects[anchor_idx + 1])
        anchor_bbox = _extract_bbox_px(anchor_object, width=width, height=height)
        gt_next_bbox = _extract_bbox_px(gt_next_object, width=width, height=height)
        if anchor_bbox is None or gt_next_bbox is None:
            continue
        candidate_specs: List[Tuple[str, Dict[str, Any]]] = [("gt_next", gt_next_object)]
        candidate_specs.append(("exact_duplicate", dict(anchor_object)))
        for pixels in config.counterfactual.jitter_pixels:
            for direction in config.counterfactual.directions:
                dx, dy = _direction_delta(direction, pixels)
                shifted = shift_bbox_xyxy(
                    anchor_bbox,
                    dx=dx,
                    dy=dy,
                    width=width,
                    height=height,
                )
                candidate_specs.append(
                    (
                        f"duplicate_jitter_{direction}_{int(pixels)}",
                        {
                            "desc": str(anchor_object["desc"]),
                            "bbox": shifted,
                            "bbox_2d": shifted,
                        },
                    )
                )
        for candidate_kind, candidate_object in candidate_specs:
            candidate_bbox = _extract_bbox_px(candidate_object, width=width, height=height)
            if candidate_bbox is None:
                continue
            scores = _build_candidate_row(
                scorer=scorer,
                image=loaded.image,
                prefix_objects=prefix_objects,
                candidate_object=candidate_object,
                width=width,
                height=height,
                object_field_order=object_field_order,
                prompt_variant=prompt_variant,
                anchor_mask_bbox_px=anchor_bbox,
                candidate_mask_bbox_px=candidate_bbox,
                mask_fill=config.counterfactual.mask_fill,
            )
            rows.append(
                {
                    "sample_index": int(sample_idx),
                    "image": str((record.get("images") or [""])[0]),
                    "anchor_gt_idx": int(anchor_idx),
                    "candidate_kind": candidate_kind,
                    "anchor_desc": str(anchor_object["desc"]),
                    "candidate_desc": str(candidate_object["desc"]),
                    **scores,
                }
            )
    (counterfactual_dir / "counterfactual_rows.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary_by_sample: Dict[int, Dict[str, Any]] = {}
    for sample_idx in sorted({int(row["sample_index"]) for row in rows}):
        sample_rows = [row for row in rows if int(row["sample_index"]) == sample_idx]
        gt_rows = [row for row in sample_rows if str(row["candidate_kind"]) == "gt_next"]
        dup_rows = [row for row in sample_rows if str(row["candidate_kind"]) != "gt_next"]
        if not gt_rows or not dup_rows:
            continue
        gt_row = gt_rows[0]
        best_dup = max(
            dup_rows,
            key=lambda row: float(row.get("full_score", float("-inf"))),
        )
        summary_by_sample[int(sample_idx)] = {
            "image": gt_row.get("image"),
            "gt_next_full_score": float(gt_row.get("full_score", float("nan"))),
            "best_duplicate_full_score": float(best_dup.get("full_score", float("nan"))),
            "full_margin_gt_minus_dup": float(
                float(gt_row.get("full_score", float("nan")))
                - float(best_dup.get("full_score", float("nan")))
            ),
            "best_duplicate_kind": str(best_dup.get("candidate_kind")),
            "gt_anchor_mask_full_score": float(
                gt_row.get("anchor_mask_full_score", float("nan"))
            ),
            "dup_anchor_mask_full_score": float(
                best_dup.get("anchor_mask_full_score", float("nan"))
            ),
        }
    manifest = {
        "counterfactual_dir": str(counterfactual_dir),
        "summary_by_sample": summary_by_sample,
        "gt_beats_duplicate_rate": float(
            sum(
                1
                for row in summary_by_sample.values()
                if float(row["full_margin_gt_minus_dup"]) > 0.0
            )
            / max(1, len(summary_by_sample))
        ),
    }
    (counterfactual_dir / "counterfactual_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    del scorer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return manifest


def write_report(
    *,
    run_dir: Path,
    subset_meta: Mapping[str, Any],
    decode_manifest: Optional[Mapping[str, Any]],
    prefix_manifest: Optional[Mapping[str, Any]],
    counterfactual_manifest: Optional[Mapping[str, Any]],
) -> Path:
    lines: List[str] = []
    lines.append("# Small Object Duplication Diagnostics")
    lines.append("")
    lines.append("## Subset")
    lines.append(
        f"- selected monitor samples: {int(subset_meta.get('selected_count', 0) or 0)}"
    )
    rows = list(subset_meta.get("rows") or [])
    if rows:
        top = rows[:5]
        for row in top:
            lines.append(
                "- "
                f"base_idx={row.get('base_idx')} image={row.get('image')} "
                f"fp_count={((row.get('stats') or {}).get('fp_count'))} "
                f"max_desc_count={((row.get('duplication') or {}).get('max_desc_count'))}"
            )
    if decode_manifest is not None:
        lines.append("")
        lines.append("## Decode Sweep")
        for summary in list(decode_manifest.get("summaries") or []):
            lines.append(
                "- "
                f"temp={summary.get('temperature')}: "
                f"mean_fp={summary.get('mean_fp_count'):.2f}, "
                f"mean_local_pairs={summary.get('mean_same_desc_local_pairs'):.2f}, "
                f"spillover_rate={summary.get('spillover_rate'):.2f}"
            )
    if prefix_manifest is not None:
        lines.append("")
        lines.append("## Prefix Probe")
        for condition, summary in sorted(
            dict(prefix_manifest.get("summary_by_condition") or {}).items()
        ):
            lines.append(
                "- "
                f"{condition}: "
                f"mean_continuation={summary.get('mean_continuation_pred_count', 0.0):.2f}, "
                f"mean_repeat_prefix_desc={summary.get('mean_same_as_prefix_last', 0.0):.2f}, "
                f"mean_local_pairs={summary.get('mean_same_desc_local_pairs', 0.0):.2f}"
            )
    if counterfactual_manifest is not None:
        lines.append("")
        lines.append("## Counterfactual")
        lines.append(
            "- "
            f"gt_next beats best duplicate rate: "
            f"{float(counterfactual_manifest.get('gt_beats_duplicate_rate', 0.0)):.2f}"
        )
        for sample_idx, summary in sorted(
            dict(counterfactual_manifest.get("summary_by_sample") or {}).items()
        )[:5]:
            lines.append(
                "- "
                f"sample={sample_idx} image={summary.get('image')} "
                f"margin={float(summary.get('full_margin_gt_minus_dup', 0.0)):.4f} "
                f"best_dup={summary.get('best_duplicate_kind')}"
            )
    report_path = run_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run_study(config_path: Path) -> Dict[str, Any]:
    config = load_study_config(config_path)
    run_dir = (REPO_ROOT / config.run.output_dir / config.run.name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    subset_path, subset_records, subset_meta = materialize_monitor_subset(
        config,
        run_dir=run_dir,
    )
    root_image_dir = Path(str(subset_meta["root_image_dir"])).resolve()

    decode_manifest: Optional[Dict[str, Any]] = None
    if _stage_enabled(config, "decode"):
        decode_manifest = run_decode_temperature_sweep(
            config,
            run_dir=run_dir,
            subset_path=subset_path,
            subset_records=subset_records,
            subset_meta=subset_meta,
            root_image_dir=root_image_dir,
        )

    prefix_manifest: Optional[Dict[str, Any]] = None
    if _stage_enabled(config, "prefix_probe"):
        prefix_manifest = run_prefix_probe(
            config,
            run_dir=run_dir,
            subset_path=subset_path,
            subset_records=subset_records,
            root_image_dir=root_image_dir,
        )

    counterfactual_manifest: Optional[Dict[str, Any]] = None
    if _stage_enabled(config, "counterfactual"):
        counterfactual_manifest = run_counterfactual_probe(
            config,
            run_dir=run_dir,
            subset_path=subset_path,
            subset_records=subset_records,
            root_image_dir=root_image_dir,
        )

    report_path = write_report(
        run_dir=run_dir,
        subset_meta=subset_meta,
        decode_manifest=decode_manifest,
        prefix_manifest=prefix_manifest,
        counterfactual_manifest=counterfactual_manifest,
    )
    manifest = {
        "config_path": str(config_path.resolve()),
        "run_dir": str(run_dir),
        "subset_path": str(subset_path),
        "report_path": str(report_path),
        "decode_manifest": decode_manifest,
        "prefix_manifest": prefix_manifest,
        "counterfactual_manifest": counterfactual_manifest,
    }
    (run_dir / "study_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "StudyConfig",
    "count_small_gt_objects",
    "find_small_object_anchor_index",
    "load_study_config",
    "materialize_monitor_subset",
    "monitor_sort_key",
    "run_study",
    "shift_bbox_xyxy",
]
