from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
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
    _close_prefix_rollout_text,
    _serialize_objects_to_prefix_text,
)
from src.analysis.unmatched_proposal_verifier import (
    TeacherForcedScorer,
    _bbox_iou_xyxy,
    _find_subsequence,
    _norm1000_to_pixel,
    _pixel_to_norm1000,
    resolve_checkpoint_path,
    resolve_prompt_controls_for_checkpoint,
)
from src.common.object_field_order import (
    build_object_payload,
    normalize_object_field_order,
)
from src.config.prompts import resolve_dense_prompt_variant_key
from src.infer.engine import GenerationConfig
from src.utils.assistant_json import dumps_coordjson

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMON_REPO_ROOT = REPO_ROOT.parent.parent if REPO_ROOT.parent.name == ".worktrees" else REPO_ROOT
_DEFAULT_STAGES = ("cohort", "decode", "prefix", "score")
_DEFAULT_JITTER_OFFSETS = (
    (-2, 0),
    (2, 0),
    (0, -2),
    (0, 2),
    (-4, 0),
    (4, 0),
    (0, -4),
    (0, 4),
)


@dataclass(frozen=True)
class RunConfig:
    name: str
    output_dir: str
    stages: Tuple[str, ...]


@dataclass(frozen=True)
class MonitorDumpConfig:
    paths: Tuple[str, ...]
    top_duplication_cases: int
    top_control_cases: int
    min_gt_objects: int
    min_pred_objects: int
    crowded_min_gt_objects: int
    small_area_max_ratio: float
    duplicate_iou_threshold: float
    center_radius_scale: float
    control_max_duplication_score: float


@dataclass(frozen=True)
class CheckpointConfig:
    alias: str
    path: str
    prompt_variant: Optional[str]
    object_field_order: Optional[str]


@dataclass(frozen=True)
class ExecutionConfig:
    device: str
    cuda_visible_devices: Optional[str]
    decode_batch_size: int
    score_batch_size: int


@dataclass(frozen=True)
class DecodeConfig:
    temperatures: Tuple[float, ...]
    top_p: float
    max_new_tokens: int
    repetition_penalty: float
    sample_seeds: Tuple[int, ...]


@dataclass(frozen=True)
class PrefixConfig:
    max_cases: int
    focus_policy: str
    sources: Tuple[str, ...]
    jitter_offsets: Tuple[Tuple[int, int], ...]
    match_iou_threshold: float


@dataclass(frozen=True)
class ScoringConfig:
    device: str
    attn_implementation: str
    max_cases: int
    match_iou_threshold: float
    max_remaining_gt_candidates: int
    duplicate_jitter_offsets: Tuple[Tuple[int, int], ...]
    include_close_candidate: bool


@dataclass(frozen=True)
class StudyConfig:
    run: RunConfig
    monitor_dumps: MonitorDumpConfig
    checkpoint: CheckpointConfig
    execution: ExecutionConfig
    decode: DecodeConfig
    prefix: PrefixConfig
    scoring: ScoringConfig


def _load_yaml(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"Expected YAML mapping at {path}")
    return raw


def _ensure_tuple_str(
    value: Any, *, default: Sequence[str], path: str
) -> Tuple[str, ...]:
    if value is None:
        return tuple(str(item) for item in default)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{path} must be a sequence of strings")
    out = tuple(str(item).strip() for item in value if str(item).strip())
    if not out:
        raise ValueError(f"{path} must not be empty")
    return out


def _ensure_tuple_int(
    value: Any, *, default: Sequence[int], path: str
) -> Tuple[int, ...]:
    if value is None:
        return tuple(int(item) for item in default)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{path} must be a sequence of ints")
    out = tuple(int(item) for item in value)
    if not out:
        raise ValueError(f"{path} must not be empty")
    return out


def _ensure_tuple_float(
    value: Any, *, default: Sequence[float], path: str
) -> Tuple[float, ...]:
    if value is None:
        return tuple(float(item) for item in default)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{path} must be a sequence of floats")
    out = tuple(float(item) for item in value)
    if not out:
        raise ValueError(f"{path} must not be empty")
    return out


def _ensure_offsets(
    value: Any,
    *,
    default: Sequence[Sequence[int]],
    path: str,
) -> Tuple[Tuple[int, int], ...]:
    raw = value if value is not None else default
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"{path} must be a sequence of [dx, dy] pairs")
    out: List[Tuple[int, int]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)):
            raise ValueError(f"{path}[{idx}] must be a two-item sequence")
        if len(item) != 2:
            raise ValueError(f"{path}[{idx}] must contain exactly two ints")
        out.append((int(item[0]), int(item[1])))
    if not out:
        raise ValueError(f"{path} must not be empty")
    return tuple(out)


def _resolve_path(path_raw: str) -> Path:
    raw = Path(path_raw)
    if raw.is_absolute():
        return raw
    candidates = [
        REPO_ROOT / raw,
        COMMON_REPO_ROOT / raw,
        Path.cwd() / raw,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return COMMON_REPO_ROOT / raw


def load_study_config(path: Path) -> StudyConfig:
    raw = _load_yaml(path)
    run_raw = raw.get("run") or {}
    monitor_raw = raw.get("monitor_dumps") or {}
    checkpoint_raw = raw.get("checkpoint") or {}
    execution_raw = raw.get("execution") or {}
    decode_raw = raw.get("decode") or {}
    prefix_raw = raw.get("prefix") or {}
    scoring_raw = raw.get("scoring") or {}

    monitor_paths = _ensure_tuple_str(
        monitor_raw.get("paths"),
        default=(),
        path="monitor_dumps.paths",
    )
    if not monitor_paths:
        raise ValueError("monitor_dumps.paths must define at least one file or directory")
    checkpoint_path_raw = str(checkpoint_raw.get("path") or "").strip()
    if not checkpoint_path_raw:
        raise ValueError("checkpoint.path is required")

    run = RunConfig(
        name=str(run_raw.get("name") or "small-object-duplication-study").strip(),
        output_dir=str(run_raw.get("output_dir") or "output/analysis").strip(),
        stages=_ensure_tuple_str(
            run_raw.get("stages"),
            default=_DEFAULT_STAGES,
            path="run.stages",
        ),
    )
    monitor_dumps = MonitorDumpConfig(
        paths=monitor_paths,
        top_duplication_cases=int(monitor_raw.get("top_duplication_cases", 12)),
        top_control_cases=int(monitor_raw.get("top_control_cases", 12)),
        min_gt_objects=int(monitor_raw.get("min_gt_objects", 4)),
        min_pred_objects=int(monitor_raw.get("min_pred_objects", 4)),
        crowded_min_gt_objects=int(monitor_raw.get("crowded_min_gt_objects", 8)),
        small_area_max_ratio=float(monitor_raw.get("small_area_max_ratio", 0.003)),
        duplicate_iou_threshold=float(monitor_raw.get("duplicate_iou_threshold", 0.7)),
        center_radius_scale=float(monitor_raw.get("center_radius_scale", 0.8)),
        control_max_duplication_score=float(
            monitor_raw.get("control_max_duplication_score", 0.0)
        ),
    )
    checkpoint = CheckpointConfig(
        alias=str(
            checkpoint_raw.get("alias")
            or Path(checkpoint_path_raw).name
        ).strip(),
        path=checkpoint_path_raw,
        prompt_variant=(
            str(checkpoint_raw["prompt_variant"]).strip()
            if checkpoint_raw.get("prompt_variant") is not None
            else None
        ),
        object_field_order=(
            normalize_object_field_order(
                str(checkpoint_raw["object_field_order"]),
                path="checkpoint.object_field_order",
            )
            if checkpoint_raw.get("object_field_order") is not None
            else None
        ),
    )
    execution = ExecutionConfig(
        device=str(execution_raw.get("device") or "cuda:0").strip(),
        cuda_visible_devices=(
            str(execution_raw["cuda_visible_devices"]).strip()
            if execution_raw.get("cuda_visible_devices") is not None
            else None
        ),
        decode_batch_size=int(execution_raw.get("decode_batch_size", 4)),
        score_batch_size=int(execution_raw.get("score_batch_size", 4)),
    )
    decode = DecodeConfig(
        temperatures=_ensure_tuple_float(
            decode_raw.get("temperatures"),
            default=(0.0, 0.01, 0.05, 0.1, 0.2),
            path="decode.temperatures",
        ),
        top_p=float(decode_raw.get("top_p", 0.95)),
        max_new_tokens=int(decode_raw.get("max_new_tokens", 3084)),
        repetition_penalty=float(decode_raw.get("repetition_penalty", 1.05)),
        sample_seeds=_ensure_tuple_int(
            decode_raw.get("sample_seeds"),
            default=(11, 12, 13),
            path="decode.sample_seeds",
        ),
    )
    prefix = PrefixConfig(
        max_cases=int(prefix_raw.get("max_cases", 8)),
        focus_policy=str(
            prefix_raw.get("focus_policy") or "earliest_matched_small_or_first_matched"
        ).strip(),
        sources=_ensure_tuple_str(
            prefix_raw.get("sources"),
            default=("pred", "gt"),
            path="prefix.sources",
        ),
        jitter_offsets=_ensure_offsets(
            prefix_raw.get("jitter_offsets"),
            default=_DEFAULT_JITTER_OFFSETS,
            path="prefix.jitter_offsets",
        ),
        match_iou_threshold=float(prefix_raw.get("match_iou_threshold", 0.5)),
    )
    scoring = ScoringConfig(
        device=str(scoring_raw.get("device") or execution.device).strip(),
        attn_implementation=str(scoring_raw.get("attn_implementation") or "auto").strip(),
        max_cases=int(scoring_raw.get("max_cases", 8)),
        match_iou_threshold=float(scoring_raw.get("match_iou_threshold", 0.5)),
        max_remaining_gt_candidates=int(
            scoring_raw.get("max_remaining_gt_candidates", 5)
        ),
        duplicate_jitter_offsets=_ensure_offsets(
            scoring_raw.get("duplicate_jitter_offsets"),
            default=_DEFAULT_JITTER_OFFSETS,
            path="scoring.duplicate_jitter_offsets",
        ),
        include_close_candidate=bool(scoring_raw.get("include_close_candidate", True)),
    )
    return StudyConfig(
        run=run,
        monitor_dumps=monitor_dumps,
        checkpoint=checkpoint,
        execution=execution,
        decode=decode,
        prefix=prefix,
        scoring=scoring,
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _chunked(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    chunk_size = max(1, int(size))
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def _iter_monitor_paths(paths: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    for raw in paths:
        resolved = _resolve_path(str(raw))
        if resolved.is_dir():
            out.extend(sorted(resolved.glob("step_*.json")))
        elif resolved.is_file():
            out.append(resolved)
        else:
            raise FileNotFoundError(f"Monitor dump path does not exist: {raw}")
    if not out:
        raise ValueError("No monitor dump files were found")
    return out


def _normalize_desc(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _extract_image_path(sample: Mapping[str, Any]) -> Optional[str]:
    for message in sample.get("messages") or []:
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, Mapping):
                    continue
                if str(item.get("type") or "").strip().lower() == "image":
                    image_path = str(item.get("image") or "").strip()
                    if image_path:
                        return image_path
        elif isinstance(content, Mapping):
            image_path = str(content.get("image") or "").strip()
            if image_path:
                return image_path
    return None


def _bbox_from_points(points: Sequence[float]) -> Optional[List[float]]:
    if len(points) < 4:
        return None
    xs = [float(v) for v in points[0::2]]
    ys = [float(v) for v in points[1::2]]
    if not xs or not ys:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def _object_bbox_norm1000(
    obj: Mapping[str, Any],
    *,
    width: int,
    height: int,
) -> Optional[List[int]]:
    bbox_raw = obj.get("bbox_2d")
    if isinstance(bbox_raw, list) and len(bbox_raw) == 4:
        parsed: List[int] = []
        for value in bbox_raw:
            text = str(value).strip()
            text = text.replace("<|coord_", "").replace("|>", "")
            try:
                parsed.append(int(text))
            except ValueError:
                parsed = []
                break
        if len(parsed) == 4:
            coord_mode = str(obj.get("coord_mode") or "").strip().lower()
            if coord_mode == "pixel" or any(int(v) < 0 or int(v) > 999 for v in parsed):
                return _pixel_to_norm1000(parsed, width, height)
            return [max(0, min(999, int(v))) for v in parsed]
    points_norm = obj.get("points_norm1000")
    if isinstance(points_norm, list) and len(points_norm) == 4:
        return [max(0, min(999, int(v))) for v in points_norm]
    points = obj.get("points")
    if isinstance(points, list) and len(points) >= 4:
        bbox_px = _bbox_from_points(points)
        if bbox_px is not None:
            return _pixel_to_norm1000(bbox_px, width, height)
    return None


def _canonicalize_objects(
    objects: Sequence[Mapping[str, Any]],
    *,
    width: int,
    height: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, obj in enumerate(objects):
        if not isinstance(obj, Mapping):
            continue
        desc = str(obj.get("desc") or "").strip()
        bbox_norm1000 = _object_bbox_norm1000(obj, width=width, height=height)
        if not desc or bbox_norm1000 is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox_norm1000]
        width_bins = max(0, x2 - x1)
        height_bins = max(0, y2 - y1)
        area_bins = float(width_bins * height_bins)
        area_ratio = area_bins / 1_000_000.0
        center_x = float(x1 + x2) / 2.0
        center_y = float(y1 + y2) / 2.0
        out.append(
            {
                "index": int(idx),
                "desc": desc,
                "desc_norm": _normalize_desc(desc),
                "bbox_norm1000": [int(x1), int(y1), int(x2), int(y2)],
                "bbox_pixel": _norm1000_to_pixel(bbox_norm1000, width, height),
                "area_bins": float(area_bins),
                "area_ratio": float(area_ratio),
                "center_norm1000": [float(center_x), float(center_y)],
                "anchor": [float(y1), float(x1)],
            }
        )
    return out


def _pair_is_duplicate_like(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    *,
    iou_threshold: float,
    center_radius_scale: float,
) -> bool:
    if str(a.get("desc_norm") or "") != str(b.get("desc_norm") or ""):
        return False
    iou = _bbox_iou_xyxy(a["bbox_norm1000"], b["bbox_norm1000"])
    if float(iou) >= float(iou_threshold):
        return True
    ax, ay = [float(v) for v in a["center_norm1000"]]
    bx, by = [float(v) for v in b["center_norm1000"]]
    dist = math.hypot(ax - bx, ay - by)
    ref = math.sqrt(max(1.0, min(float(a["area_bins"]), float(b["area_bins"]))))
    return bool(dist <= float(center_radius_scale) * ref)


def _principal_linearity(points: Sequence[Sequence[float]]) -> Optional[float]:
    if len(points) < 2:
        return None
    xs = [float(item[0]) for item in points]
    ys = [float(item[1]) for item in points]
    mean_x = sum(xs) / float(len(xs))
    mean_y = sum(ys) / float(len(ys))
    sxx = sum((x - mean_x) ** 2 for x in xs)
    syy = sum((y - mean_y) ** 2 for y in ys)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    trace = sxx + syy
    if trace <= 0.0:
        return None
    det_term = max(0.0, ((sxx - syy) ** 2) + (4.0 * sxy * sxy))
    lambda_max = 0.5 * (trace + math.sqrt(det_term))
    return float(lambda_max / trace)


def _duplicate_metrics(
    objects: Sequence[Mapping[str, Any]],
    *,
    small_area_max_ratio: float,
    duplicate_iou_threshold: float,
    center_radius_scale: float,
) -> Dict[str, Any]:
    desc_counter = Counter(str(obj.get("desc_norm") or "") for obj in objects)
    total = float(sum(desc_counter.values()))
    entropy = 0.0
    if total > 0.0:
        for count in desc_counter.values():
            p = float(count) / total
            entropy -= p * math.log(max(p, 1e-12))

    adjacency: Dict[int, set[int]] = {int(obj["index"]): set() for obj in objects}
    same_desc_pairs = 0
    small_same_desc_pairs = 0
    for idx, a in enumerate(objects):
        for b in objects[idx + 1 :]:
            if not _pair_is_duplicate_like(
                a,
                b,
                iou_threshold=duplicate_iou_threshold,
                center_radius_scale=center_radius_scale,
            ):
                continue
            same_desc_pairs += 1
            adjacency[int(a["index"])].add(int(b["index"]))
            adjacency[int(b["index"])].add(int(a["index"]))
            if (
                float(a["area_ratio"]) <= float(small_area_max_ratio)
                and float(b["area_ratio"]) <= float(small_area_max_ratio)
            ):
                small_same_desc_pairs += 1

    clusters: List[List[int]] = []
    seen: set[int] = set()
    for node in adjacency:
        if node in seen or not adjacency[node]:
            continue
        queue = [node]
        seen.add(node)
        cluster: List[int] = []
        while queue:
            cur = queue.pop()
            cluster.append(cur)
            for nxt in adjacency[cur]:
                if nxt in seen:
                    continue
                seen.add(nxt)
                queue.append(nxt)
        clusters.append(sorted(cluster))

    by_index = {int(obj["index"]): obj for obj in objects}
    largest_cluster = max(clusters, key=len, default=[])
    largest_points = [
        by_index[idx]["center_norm1000"]
        for idx in sorted(
            largest_cluster,
            key=lambda idx: tuple(by_index[idx]["anchor"]),
        )
        if idx in by_index
    ]
    step_distances: List[float] = []
    for prev, cur in zip(largest_points, largest_points[1:]):
        step_distances.append(math.hypot(float(cur[0]) - float(prev[0]), float(cur[1]) - float(prev[1])))

    small_cluster_sizes: List[int] = []
    for cluster in clusters:
        if all(
            float(by_index[idx]["area_ratio"]) <= float(small_area_max_ratio)
            for idx in cluster
            if idx in by_index
        ):
            small_cluster_sizes.append(len(cluster))

    return {
        "duplicate_like_pair_count": int(same_desc_pairs),
        "small_duplicate_like_pair_count": int(small_same_desc_pairs),
        "duplicate_like_cluster_count": int(len(clusters)),
        "duplicate_like_max_cluster_size": int(max((len(cluster) for cluster in clusters), default=1)),
        "small_duplicate_like_max_cluster_size": int(max(small_cluster_sizes, default=1)),
        "largest_cluster_linearity": _principal_linearity(largest_points),
        "largest_cluster_mean_step_distance": (
            float(mean(step_distances)) if step_distances else None
        ),
        "desc_entropy": float(entropy),
    }


def _case_key(global_step: int, sample_index: int, sample_id: Any) -> str:
    return f"step_{int(global_step):06d}_sample_{int(sample_index):03d}_{int(sample_id)}"


def _extract_focus_match(
    sample: Mapping[str, Any],
    *,
    pred_objects: Sequence[Mapping[str, Any]],
    gt_objects: Sequence[Mapping[str, Any]],
    small_area_max_ratio: float,
    match_iou_threshold: float,
    focus_policy: str,
) -> Optional[Dict[str, Any]]:
    match_details = list(((sample.get("match") or {}).get("matched_pair_details") or []))
    candidates: List[Dict[str, Any]] = []
    if match_details:
        for item in match_details:
            pred_i = int(item.get("pred_i", item.get("pred_index", -1)) or -1)
            gt_i = int(item.get("gt_i", item.get("gt_index", -1)) or -1)
            if pred_i < 0 or pred_i >= len(pred_objects) or gt_i < 0 or gt_i >= len(gt_objects):
                continue
            pred = pred_objects[pred_i]
            gt = gt_objects[gt_i]
            candidates.append(
                {
                    "pred_i": int(pred_i),
                    "gt_i": int(gt_i),
                    "bbox_iou_norm1000": float(
                        item.get("bbox_iou_norm1000")
                        or _bbox_iou_xyxy(pred["bbox_norm1000"], gt["bbox_norm1000"])
                    ),
                    "pred_obj": pred,
                    "gt_obj": gt,
                }
            )
    else:
        used_gt: set[int] = set()
        for pred in pred_objects:
            best_gt: Optional[Mapping[str, Any]] = None
            best_gt_idx = -1
            best_iou = 0.0
            for gt in gt_objects:
                gt_idx = int(gt["index"])
                if gt_idx in used_gt:
                    continue
                if str(pred.get("desc_norm") or "") != str(gt.get("desc_norm") or ""):
                    continue
                iou = _bbox_iou_xyxy(pred["bbox_norm1000"], gt["bbox_norm1000"])
                if iou < float(match_iou_threshold) or iou <= best_iou:
                    continue
                best_gt = gt
                best_gt_idx = gt_idx
                best_iou = float(iou)
            if best_gt is None:
                continue
            used_gt.add(int(best_gt_idx))
            candidates.append(
                {
                    "pred_i": int(pred["index"]),
                    "gt_i": int(best_gt_idx),
                    "bbox_iou_norm1000": float(best_iou),
                    "pred_obj": pred,
                    "gt_obj": best_gt,
                }
            )
    if not candidates:
        return None
    ordered = sorted(candidates, key=lambda row: int(row["pred_i"]))
    if focus_policy == "earliest_matched_small_or_first_matched":
        small = [
            row
            for row in ordered
            if float(row["pred_obj"]["area_ratio"]) <= float(small_area_max_ratio)
        ]
        if small:
            return small[0]
    return ordered[0]


def _build_case_row(
    *,
    monitor_path: Path,
    global_step: int,
    sample_index: int,
    sample: Mapping[str, Any],
    cfg: StudyConfig,
) -> Optional[Dict[str, Any]]:
    width = int(sample.get("width") or 0)
    height = int(sample.get("height") or 0)
    if width <= 0 or height <= 0:
        return None
    pred_objects = _canonicalize_objects(sample.get("pred") or [], width=width, height=height)
    gt_objects = _canonicalize_objects(sample.get("gt") or [], width=width, height=height)
    if len(pred_objects) < int(cfg.monitor_dumps.min_pred_objects) or len(gt_objects) < int(
        cfg.monitor_dumps.min_gt_objects
    ):
        return None
    image_path = _extract_image_path(sample)
    if not image_path:
        return None
    dup_local = _duplicate_metrics(
        pred_objects,
        small_area_max_ratio=cfg.monitor_dumps.small_area_max_ratio,
        duplicate_iou_threshold=cfg.monitor_dumps.duplicate_iou_threshold,
        center_radius_scale=cfg.monitor_dumps.center_radius_scale,
    )
    dup_existing = sample.get("duplication") or {}
    stats = sample.get("stats") or {}
    focus_match = _extract_focus_match(
        sample,
        pred_objects=pred_objects,
        gt_objects=gt_objects,
        small_area_max_ratio=cfg.monitor_dumps.small_area_max_ratio,
        match_iou_threshold=cfg.prefix.match_iou_threshold,
        focus_policy=cfg.prefix.focus_policy,
    )
    fp_excess = max(0, int(len(pred_objects) - len(gt_objects)))
    duplication_score = (
        4.0 * float(dup_existing.get("near_iou90_pairs_same_desc_count", 0.0) or 0.0)
        + 10.0 * float(dup_existing.get("duplicate_bursts", 0.0) or 0.0)
        + 6.0 * float(dup_local["small_duplicate_like_pair_count"])
        + 2.0 * max(0.0, float(dup_local["small_duplicate_like_max_cluster_size"]) - 1.0)
        + 0.25 * float(fp_excess)
    )
    case_key = _case_key(
        global_step=global_step,
        sample_index=sample_index,
        sample_id=sample.get("sample_id") or sample.get("image_id") or sample_index,
    )
    small_gt_count = sum(
        1
        for obj in gt_objects
        if float(obj["area_ratio"]) <= float(cfg.monitor_dumps.small_area_max_ratio)
    )
    row = {
        "case_key": case_key,
        "monitor_path": str(monitor_path),
        "global_step": int(global_step),
        "sample_index": int(sample_index),
        "sample_id": int(sample.get("sample_id") or sample.get("image_id") or sample_index),
        "base_idx": int(sample.get("base_idx") or -1),
        "image_path": str(image_path),
        "width": int(width),
        "height": int(height),
        "gt_count": int(len(gt_objects)),
        "pred_count": int(len(pred_objects)),
        "small_gt_count": int(small_gt_count),
        "duplication_score": float(duplication_score),
        "existing_duplication": {
            "duplicates": int(dup_existing.get("duplicates", 0) or 0),
            "duplicate_bursts": int(dup_existing.get("duplicate_bursts", 0) or 0),
            "near_iou90_pairs_same_desc_count": float(
                dup_existing.get("near_iou90_pairs_same_desc_count", 0.0) or 0.0
            ),
            "near_iou90_pairs_any_desc_count": float(
                dup_existing.get("near_iou90_pairs_any_desc_count", 0.0) or 0.0
            ),
            "saturation_rate": float(dup_existing.get("saturation_rate", 0.0) or 0.0),
            "max_desc_count": float(dup_existing.get("max_desc_count", 0.0) or 0.0),
        },
        "local_duplicate_metrics": dup_local,
        "stats": {
            "matched": int(stats.get("matched", 0) or 0),
            "fp_count": int(stats.get("fp_count", 0) or 0),
            "fn_count": int(stats.get("fn_count", 0) or 0),
            "precision": float(stats.get("precision", 0.0) or 0.0),
            "recall": float(stats.get("recall", 0.0) or 0.0),
            "f1": float(stats.get("f1", 0.0) or 0.0),
            "duplicate_burst_unlikelihood_boundary_count": int(
                stats.get("duplicate_burst_unlikelihood_boundary_count", 0) or 0
            ),
        },
        "triage": {
            "shielded_anchor_indices": list(
                ((sample.get("triage") or {}).get("shielded_anchor_indices") or [])
            ),
            "dead_anchor_indices": list(
                ((sample.get("triage") or {}).get("dead_anchor_indices") or [])
            ),
            "pseudo_positive_anchor_indices": list(
                ((sample.get("triage") or {}).get("pseudo_positive_anchor_indices") or [])
            ),
            "recovered_gt_indices": list(
                ((sample.get("triage") or {}).get("recovered_gt_indices") or [])
            ),
        },
        "focus_match": focus_match,
        "gt_objects": gt_objects,
        "pred_objects": pred_objects,
    }
    return row


def _select_cohorts(
    rows: Sequence[Mapping[str, Any]],
    *,
    cfg: StudyConfig,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    duplication_candidates = [
        dict(row)
        for row in rows
        if float(row.get("duplication_score", 0.0) or 0.0) > 0.0
    ]
    duplication_cases = sorted(
        duplication_candidates,
        key=lambda row: (
            -float(row["duplication_score"]),
            -int(row["small_gt_count"]),
            -int(row["gt_count"]),
            str(row["case_key"]),
        ),
    )[: int(cfg.monitor_dumps.top_duplication_cases)]
    for row in duplication_cases:
        row["cohort_bucket"] = "duplication_case"

    control_candidates = [
        dict(row)
        for row in rows
        if int(row["gt_count"]) >= int(cfg.monitor_dumps.crowded_min_gt_objects)
        and float(row["duplication_score"]) <= float(cfg.monitor_dumps.control_max_duplication_score)
    ]
    controls = sorted(
        control_candidates,
        key=lambda row: (
            -int(row["gt_count"]),
            -int(row["small_gt_count"]),
            float(row["duplication_score"]),
            str(row["case_key"]),
        ),
    )[: int(cfg.monitor_dumps.top_control_cases)]
    for row in controls:
        row["cohort_bucket"] = "crowded_control"
    return duplication_cases, controls


def _stage_cohort(cfg: StudyConfig, *, run_dir: Path) -> Dict[str, Any]:
    monitor_paths = _iter_monitor_paths(cfg.monitor_dumps.paths)
    all_rows: List[Dict[str, Any]] = []
    for monitor_path in monitor_paths:
        payload = json.loads(monitor_path.read_text(encoding="utf-8"))
        samples = payload.get("samples") or []
        global_step = int(payload.get("global_step") or 0)
        for sample_index, sample in enumerate(samples):
            if not isinstance(sample, Mapping):
                continue
            row = _build_case_row(
                monitor_path=monitor_path,
                global_step=global_step,
                sample_index=sample_index,
                sample=sample,
                cfg=cfg,
            )
            if row is not None:
                all_rows.append(row)
    duplication_cases, controls = _select_cohorts(all_rows, cfg=cfg)
    summary = {
        "monitor_paths": [str(path) for path in monitor_paths],
        "n_all_samples": int(len(all_rows)),
        "n_duplication_cases": int(len(duplication_cases)),
        "n_controls": int(len(controls)),
        "duplication_case_keys": [row["case_key"] for row in duplication_cases],
        "control_case_keys": [row["case_key"] for row in controls],
    }
    cohort_dir = run_dir / "cohort"
    _write_jsonl(
        cohort_dir / "all_samples.jsonl",
        [
            {
                "case_key": row["case_key"],
                "global_step": row["global_step"],
                "sample_index": row["sample_index"],
                "image_path": row["image_path"],
                "gt_count": row["gt_count"],
                "pred_count": row["pred_count"],
                "duplication_score": row["duplication_score"],
                "existing_duplication": row["existing_duplication"],
                "local_duplicate_metrics": row["local_duplicate_metrics"],
                "stats": row["stats"],
            }
            for row in all_rows
        ],
    )
    _write_jsonl(cohort_dir / "duplication_cases.jsonl", duplication_cases)
    _write_jsonl(cohort_dir / "crowded_controls.jsonl", controls)
    _write_json(cohort_dir / "summary.json", summary)
    return {
        "all_rows": all_rows,
        "duplication_cases": duplication_cases,
        "controls": controls,
        "summary": summary,
    }


def _set_cuda_visible_devices(value: Optional[str]) -> None:
    if value is None:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(value)


def _resolve_checkpoint(cfg: StudyConfig) -> Tuple[ResolvedCheckpoint, Dict[str, str]]:
    checkpoint_path, resolve_source = resolve_checkpoint_path(cfg.checkpoint.path)
    prompt_variant, object_field_order, prompt_source = resolve_prompt_controls_for_checkpoint(
        checkpoint_path,
        default_prompt_variant="coco_80",
        default_object_field_order="desc_first",
        override_prompt_variant=cfg.checkpoint.prompt_variant,
        override_object_field_order=cfg.checkpoint.object_field_order,
    )
    resolved = ResolvedCheckpoint(
        alias=cfg.checkpoint.alias,
        path=checkpoint_path,
        resolve_source=str(resolve_source),
        artifact_kind="executable_checkpoint",
        fingerprint=str(checkpoint_path.name),
        prompt_variant=resolve_dense_prompt_variant_key(prompt_variant),
        object_field_order=normalize_object_field_order(
            object_field_order,
            path="resolved_checkpoint.object_field_order",
        ),
        prompt_control_source=str(prompt_source),
        provenance_sidecars={},
    )
    return resolved, {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_resolve_source": str(resolve_source),
        "prompt_control_source": str(prompt_source),
        "prompt_variant": resolved.prompt_variant,
        "object_field_order": resolved.object_field_order,
    }


def _load_case_image(row: Mapping[str, Any]) -> Image.Image:
    return Image.open(str(row["image_path"])).convert("RGB")


def _decode_settings(cfg: StudyConfig) -> List[Tuple[float, int]]:
    settings: List[Tuple[float, int]] = []
    for temperature in cfg.decode.temperatures:
        if float(temperature) <= 0.0:
            settings.append((float(temperature), int(cfg.decode.sample_seeds[0])))
        else:
            for seed in cfg.decode.sample_seeds:
                settings.append((float(temperature), int(seed)))
    return settings


def _run_generation_batch(
    runner: HFStudyRunner,
    *,
    rows: Sequence[Mapping[str, Any]],
    gen_cfg: GenerationConfig,
    prefix_texts: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    images = [_load_case_image(row) for row in rows]
    try:
        if prefix_texts is None:
            generations = runner.generate_image_only_batch(images=images, gen_cfg=gen_cfg)
        else:
            generations = runner.generate_with_prefix_batch(
                images=images,
                prefix_texts=prefix_texts,
                gen_cfg=gen_cfg,
            )
    finally:
        for image in images:
            image.close()
    results: List[Dict[str, Any]] = []
    for row, generation in zip(rows, generations):
        pred_objects, pred_errors, raw_json, raw_special_tokens, raw_ends_with_im_end = (
            runner.parse_prediction(
                raw_text=str(generation.raw_text),
                width=int(row["width"]),
                height=int(row["height"]),
            )
        )
        canonical_pred = _canonicalize_objects(
            pred_objects,
            width=int(row["width"]),
            height=int(row["height"]),
        )
        results.append(
            {
                "case_key": row["case_key"],
                "raw_text": str(generation.raw_text),
                "generated_token_count": int(generation.generated_token_count),
                "finish_reason": str(generation.finish_reason),
                "eos_reached": bool(generation.eos_reached),
                "pred_count": int(len(canonical_pred)),
                "pred_objects": canonical_pred,
                "pred_errors": list(pred_errors),
                "raw_json_present": bool(raw_json is not None),
                "raw_special_token_count": int(len(raw_special_tokens)),
                "raw_ends_with_im_end": bool(raw_ends_with_im_end),
            }
        )
    return results


def _duplicate_metrics_from_cfg(
    cfg: StudyConfig,
    objects: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    return _duplicate_metrics(
        objects,
        small_area_max_ratio=cfg.monitor_dumps.small_area_max_ratio,
        duplicate_iou_threshold=cfg.monitor_dumps.duplicate_iou_threshold,
        center_radius_scale=cfg.monitor_dumps.center_radius_scale,
    )


def _stage_decode(
    cfg: StudyConfig,
    *,
    run_dir: Path,
    selected_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    _set_cuda_visible_devices(cfg.execution.cuda_visible_devices)
    resolved_checkpoint, checkpoint_meta = _resolve_checkpoint(cfg)
    runner = HFStudyRunner(
        checkpoint=resolved_checkpoint,
        device=cfg.execution.device,
        image_root=COMMON_REPO_ROOT,
    )
    result_rows: List[Dict[str, Any]] = []
    for temperature, seed in _decode_settings(cfg):
        gen_cfg = GenerationConfig(
            temperature=float(temperature),
            top_p=float(cfg.decode.top_p),
            max_new_tokens=int(cfg.decode.max_new_tokens),
            repetition_penalty=float(cfg.decode.repetition_penalty),
            batch_size=int(cfg.execution.decode_batch_size),
            seed=int(seed),
        )
        for chunk in _chunked(list(selected_rows), cfg.execution.decode_batch_size):
            batch_rows = _run_generation_batch(runner, rows=chunk, gen_cfg=gen_cfg)
            for row, generated in zip(chunk, batch_rows):
                result_rows.append(
                    {
                        "case_key": row["case_key"],
                        "cohort_bucket": row.get("cohort_bucket"),
                        "temperature": float(temperature),
                        "seed": int(seed),
                        **generated,
                        "duplicate_metrics": _duplicate_metrics_from_cfg(
                            cfg, generated["pred_objects"]
                        ),
                    }
                )
    decode_dir = run_dir / "decode"
    _write_jsonl(decode_dir / "results.jsonl", result_rows)
    summary_rows: Dict[Tuple[str, float], List[Mapping[str, Any]]] = defaultdict(list)
    for row in result_rows:
        summary_rows[(str(row.get("cohort_bucket") or "unknown"), float(row["temperature"]))].append(row)
    summary = {
        "checkpoint": checkpoint_meta,
        "by_bucket_temperature": [
            {
                "cohort_bucket": bucket,
                "temperature": float(temp),
                "count": int(len(items)),
                "mean_pred_count": float(mean(item["pred_count"] for item in items)),
                "mean_small_duplicate_like_pair_count": float(
                    mean(
                        item["duplicate_metrics"]["small_duplicate_like_pair_count"]
                        for item in items
                    )
                ),
                "mean_duplicate_like_max_cluster_size": float(
                    mean(
                        item["duplicate_metrics"]["duplicate_like_max_cluster_size"]
                        for item in items
                    )
                ),
            }
            for (bucket, temp), items in sorted(summary_rows.items())
        ],
    }
    _write_json(decode_dir / "summary.json", summary)
    return {"results": result_rows, "summary": summary}


def _shift_bbox_norm1000(
    bbox_norm1000: Sequence[int], *, dx: int, dy: int
) -> Optional[List[int]]:
    if len(bbox_norm1000) != 4:
        return None
    x1, y1, x2, y2 = [int(v) for v in bbox_norm1000]
    shifted = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]
    clipped = [max(0, min(999, int(v))) for v in shifted]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return clipped


def _prefix_object(
    obj: Mapping[str, Any], *, bbox_norm1000: Sequence[int]
) -> Dict[str, Any]:
    return {
        "desc": str(obj["desc"]),
        "bbox_2d": [int(v) for v in bbox_norm1000],
    }


def _build_prefix_conditions(
    case_row: Mapping[str, Any],
    *,
    object_field_order: str,
    cfg: StudyConfig,
) -> List[Dict[str, Any]]:
    focus = case_row.get("focus_match")
    if not isinstance(focus, Mapping):
        return []
    width = int(case_row["width"])
    height = int(case_row["height"])
    pred_obj = focus["pred_obj"]
    gt_obj = focus["gt_obj"]
    conditions: List[Dict[str, Any]] = []
    if "pred" in cfg.prefix.sources:
        base_prefix_text, _ = _serialize_objects_to_prefix_text(
            [_prefix_object(pred_obj, bbox_norm1000=pred_obj["bbox_norm1000"])],
            width=width,
            height=height,
            object_field_order=object_field_order,
        )
        conditions.append(
            {
                "condition": "first_correct_pred",
                "prefix_source": "pred",
                "prefix_text": base_prefix_text,
                "focus_pred_i": int(focus["pred_i"]),
                "focus_gt_i": int(focus["gt_i"]),
            }
        )
        for dx, dy in cfg.prefix.jitter_offsets:
            shifted = _shift_bbox_norm1000(pred_obj["bbox_norm1000"], dx=dx, dy=dy)
            if shifted is None:
                continue
            prefix_text, _ = _serialize_objects_to_prefix_text(
                [_prefix_object(pred_obj, bbox_norm1000=shifted)],
                width=width,
                height=height,
                object_field_order=object_field_order,
            )
            conditions.append(
                {
                    "condition": f"first_correct_pred_jitter_dx{int(dx)}_dy{int(dy)}",
                    "prefix_source": "pred",
                    "prefix_text": prefix_text,
                    "focus_pred_i": int(focus["pred_i"]),
                    "focus_gt_i": int(focus["gt_i"]),
                }
            )
    if "gt" in cfg.prefix.sources:
        prefix_text, _ = _serialize_objects_to_prefix_text(
            [_prefix_object(gt_obj, bbox_norm1000=gt_obj["bbox_norm1000"])],
            width=width,
            height=height,
            object_field_order=object_field_order,
        )
        conditions.append(
            {
                "condition": "first_correct_gt",
                "prefix_source": "gt",
                "prefix_text": prefix_text,
                "focus_pred_i": int(focus["pred_i"]),
                "focus_gt_i": int(focus["gt_i"]),
            }
        )
    return conditions


def _stage_prefix(
    cfg: StudyConfig,
    *,
    run_dir: Path,
    duplication_cases: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    _set_cuda_visible_devices(cfg.execution.cuda_visible_devices)
    resolved_checkpoint, checkpoint_meta = _resolve_checkpoint(cfg)
    runner = HFStudyRunner(
        checkpoint=resolved_checkpoint,
        device=cfg.execution.device,
        image_root=COMMON_REPO_ROOT,
    )
    selected_cases = list(duplication_cases)[: int(cfg.prefix.max_cases)]
    requests: List[Dict[str, Any]] = []
    for row in selected_cases:
        for condition in _build_prefix_conditions(
            row,
            object_field_order=resolved_checkpoint.object_field_order,
            cfg=cfg,
        ):
            payload = dict(row)
            payload.update(condition)
            requests.append(payload)
    result_rows: List[Dict[str, Any]] = []
    for temperature, seed in _decode_settings(cfg):
        gen_cfg = GenerationConfig(
            temperature=float(temperature),
            top_p=float(cfg.decode.top_p),
            max_new_tokens=int(cfg.decode.max_new_tokens),
            repetition_penalty=float(cfg.decode.repetition_penalty),
            batch_size=int(cfg.execution.decode_batch_size),
            seed=int(seed),
        )
        for chunk in _chunked(requests, cfg.execution.decode_batch_size):
            generated_rows = _run_generation_batch(
                runner,
                rows=chunk,
                gen_cfg=gen_cfg,
                prefix_texts=[str(item["prefix_text"]) for item in chunk],
            )
            for request, generated in zip(chunk, generated_rows):
                result_rows.append(
                    {
                        "case_key": request["case_key"],
                        "condition": request["condition"],
                        "prefix_source": request["prefix_source"],
                        "temperature": float(temperature),
                        "seed": int(seed),
                        "focus_pred_i": int(request["focus_pred_i"]),
                        "focus_gt_i": int(request["focus_gt_i"]),
                        **generated,
                        "duplicate_metrics": _duplicate_metrics_from_cfg(
                            cfg, generated["pred_objects"]
                        ),
                    }
                )
    prefix_dir = run_dir / "prefix"
    _write_jsonl(prefix_dir / "results.jsonl", result_rows)
    by_condition: Dict[Tuple[str, float], List[Mapping[str, Any]]] = defaultdict(list)
    for row in result_rows:
        by_condition[(str(row["condition"]), float(row["temperature"]))].append(row)
    summary = {
        "checkpoint": checkpoint_meta,
        "by_condition_temperature": [
            {
                "condition": condition,
                "temperature": float(temp),
                "count": int(len(items)),
                "mean_pred_count": float(mean(item["pred_count"] for item in items)),
                "mean_small_duplicate_like_pair_count": float(
                    mean(
                        item["duplicate_metrics"]["small_duplicate_like_pair_count"]
                        for item in items
                    )
                ),
                "mean_duplicate_like_max_cluster_size": float(
                    mean(
                        item["duplicate_metrics"]["duplicate_like_max_cluster_size"]
                        for item in items
                    )
                ),
            }
            for (condition, temp), items in sorted(by_condition.items())
        ],
    }
    _write_json(prefix_dir / "summary.json", summary)
    return {"results": result_rows, "summary": summary}


def _object_entry_text(
    obj: Mapping[str, Any], *, object_field_order: str
) -> str:
    payload = build_object_payload(
        desc=str(obj["desc"]),
        geometry_key="bbox_2d",
        geometry_value=[f"<|coord_{int(v)}|>" for v in obj["bbox_norm1000"]],
        object_field_order=object_field_order,
    )
    return dumps_coordjson(payload)


def _candidate_rel_positions(
    tokenizer: Any,
    *,
    assistant_text: str,
    prefix_text: str,
    candidate_text: str,
    desc: Optional[str],
    bbox_norm1000: Optional[Sequence[int]],
) -> Dict[str, Tuple[int, ...]]:
    def _positions_from_span(
        offsets: Sequence[Sequence[int]],
        *,
        start_char: int,
        end_char: int,
    ) -> Tuple[int, ...]:
        out: List[int] = []
        for idx, span in enumerate(offsets):
            if len(span) != 2:
                continue
            token_start = int(span[0])
            token_end = int(span[1])
            if token_end <= int(start_char) or token_start >= int(end_char):
                continue
            out.append(int(idx))
        return tuple(out)

    try:
        encoded = tokenizer(
            str(assistant_text),
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = encoded.get("offset_mapping") or []
    except TypeError:
        offsets = []
    candidate_char_start = str(assistant_text).find(
        str(candidate_text),
        max(0, len(str(prefix_text)) - 4),
    )
    if candidate_char_start < 0:
        raise ValueError("candidate span not found in assistant text")
    candidate_char_end = int(candidate_char_start + len(str(candidate_text)))

    if offsets:
        positions: Dict[str, Tuple[int, ...]] = {
            "full": _positions_from_span(
                offsets,
                start_char=int(candidate_char_start),
                end_char=int(candidate_char_end),
            )
        }
        if not positions["full"]:
            raise ValueError("candidate token span is empty")
        if desc:
            desc_text = f'"{str(desc)}"'
            desc_char_start = str(assistant_text).find(
                desc_text,
                int(candidate_char_start),
                int(candidate_char_end),
            )
            if desc_char_start >= 0:
                desc_positions = _positions_from_span(
                    offsets,
                    start_char=int(desc_char_start),
                    end_char=int(desc_char_start + len(desc_text)),
                )
                if desc_positions:
                    positions["desc"] = desc_positions
        if bbox_norm1000 is not None:
            coord_positions: List[int] = []
            cursor = int(candidate_char_start)
            for value in bbox_norm1000:
                coord_literal = f"<|coord_{int(value)}|>"
                coord_char_start = str(assistant_text).find(
                    coord_literal,
                    cursor,
                    int(candidate_char_end),
                )
                if coord_char_start < 0:
                    coord_positions = []
                    break
                coord_token_positions = _positions_from_span(
                    offsets,
                    start_char=int(coord_char_start),
                    end_char=int(coord_char_start + len(coord_literal)),
                )
                if not coord_token_positions:
                    coord_positions = []
                    break
                coord_positions.extend(int(pos) for pos in coord_token_positions)
                cursor = int(coord_char_start + len(coord_literal))
            if coord_positions:
                positions["coord"] = tuple(coord_positions)
        return positions

    assistant_ids = tokenizer.encode(str(assistant_text), add_special_tokens=False)
    prefix_ids = tokenizer.encode(str(prefix_text), add_special_tokens=False)
    candidate_ids = tokenizer.encode(str(candidate_text), add_special_tokens=False)
    candidate_start = _find_subsequence(
        assistant_ids,
        candidate_ids,
        start_hint=max(0, len(prefix_ids) - 4),
    )
    if candidate_start is None:
        raise ValueError("candidate span not found in assistant text")
    candidate_end = int(candidate_start + len(candidate_ids))
    positions = {"full": tuple(range(int(candidate_start), int(candidate_end)))}
    if desc:
        desc_ids = tokenizer.encode(f'"{str(desc)}"', add_special_tokens=False)
        desc_start = _find_subsequence(
            assistant_ids,
            desc_ids,
            start_hint=int(candidate_start),
        )
        if desc_start is not None and int(desc_start + len(desc_ids)) <= int(candidate_end):
            positions["desc"] = tuple(
                range(int(desc_start), int(desc_start + len(desc_ids)))
            )
    if bbox_norm1000 is not None:
        coord_positions = []
        cursor = int(candidate_start)
        for value in bbox_norm1000:
            coord_ids = tokenizer.encode(
                f"<|coord_{int(value)}|>",
                add_special_tokens=False,
            )
            coord_start = _find_subsequence(
                assistant_ids,
                coord_ids,
                start_hint=cursor,
            )
            if coord_start is None or int(coord_start + len(coord_ids)) > int(candidate_end):
                coord_positions = []
                break
            coord_positions.extend(
                int(pos) for pos in range(int(coord_start), int(coord_start + len(coord_ids)))
            )
            cursor = int(coord_start + len(coord_ids))
        if coord_positions:
            positions["coord"] = tuple(coord_positions)
    return positions


def _build_scoring_candidates(
    case_row: Mapping[str, Any],
    *,
    tokenizer: Any,
    object_field_order: str,
    cfg: StudyConfig,
) -> List[Dict[str, Any]]:
    focus = case_row.get("focus_match")
    if not isinstance(focus, Mapping):
        return []
    width = int(case_row["width"])
    height = int(case_row["height"])
    focus_pred = focus["pred_obj"]
    gt_objects = list(case_row["gt_objects"])
    focus_gt_i = int(focus["gt_i"])
    prefix_text, _ = _serialize_objects_to_prefix_text(
        [_prefix_object(focus_pred, bbox_norm1000=focus_pred["bbox_norm1000"])],
        width=width,
        height=height,
        object_field_order=object_field_order,
    )
    candidates: List[Dict[str, Any]] = []
    if cfg.scoring.include_close_candidate:
        assistant_text = str(prefix_text[:-2] + "]}") if str(prefix_text).endswith(", ") else str(prefix_text) + "]}"
        rel_positions = _candidate_rel_positions(
            tokenizer,
            assistant_text=assistant_text,
            prefix_text=str(prefix_text[:-2]) if str(prefix_text).endswith(", ") else str(prefix_text),
            candidate_text="]}",
            desc=None,
            bbox_norm1000=None,
        )
        candidates.append(
            {
                "family": "close",
                "label": "close_container",
                "assistant_text": assistant_text,
                "rel_positions": rel_positions,
                "candidate_desc": None,
                "candidate_bbox_norm1000": None,
                "focus_pred_i": int(focus["pred_i"]),
                "focus_gt_i": int(focus_gt_i),
            }
        )
    remaining_gts = [
        obj
        for obj in gt_objects[int(focus_gt_i) + 1 :]
        if int(obj["index"]) != int(focus_gt_i)
    ]
    if not remaining_gts:
        remaining_gts = [
            obj for obj in gt_objects if int(obj["index"]) != int(focus_gt_i)
        ]
    for gt_obj in remaining_gts[: int(cfg.scoring.max_remaining_gt_candidates)]:
        candidate_text = _object_entry_text(gt_obj, object_field_order=object_field_order)
        assistant_text = _close_prefix_rollout_text(
            prefix_text,
            candidate_text,
            object_field_order=object_field_order,
        )
        rel_positions = _candidate_rel_positions(
            tokenizer,
            assistant_text=assistant_text,
            prefix_text=prefix_text,
            candidate_text=candidate_text,
            desc=str(gt_obj["desc"]),
            bbox_norm1000=gt_obj["bbox_norm1000"],
        )
        candidates.append(
            {
                "family": "remaining_gt",
                "label": f"gt_idx_{int(gt_obj['index'])}",
                "assistant_text": assistant_text,
                "rel_positions": rel_positions,
                "candidate_desc": str(gt_obj["desc"]),
                "candidate_bbox_norm1000": list(gt_obj["bbox_norm1000"]),
                "focus_pred_i": int(focus["pred_i"]),
                "focus_gt_i": int(focus_gt_i),
            }
        )
    for dx, dy in cfg.scoring.duplicate_jitter_offsets:
        shifted = _shift_bbox_norm1000(focus_pred["bbox_norm1000"], dx=dx, dy=dy)
        if shifted is None:
            continue
        duplicate_obj = _prefix_object(focus_pred, bbox_norm1000=shifted)
        duplicate_obj["bbox_norm1000"] = list(shifted)
        candidate_text = _object_entry_text(duplicate_obj, object_field_order=object_field_order)
        assistant_text = _close_prefix_rollout_text(
            prefix_text,
            candidate_text,
            object_field_order=object_field_order,
        )
        rel_positions = _candidate_rel_positions(
            tokenizer,
            assistant_text=assistant_text,
            prefix_text=prefix_text,
            candidate_text=candidate_text,
            desc=str(duplicate_obj["desc"]),
            bbox_norm1000=shifted,
        )
        candidates.append(
            {
                "family": "duplicate_jitter",
                "label": f"dup_dx{int(dx)}_dy{int(dy)}",
                "assistant_text": assistant_text,
                "rel_positions": rel_positions,
                "candidate_desc": str(duplicate_obj["desc"]),
                "candidate_bbox_norm1000": list(shifted),
                "focus_pred_i": int(focus["pred_i"]),
                "focus_gt_i": int(focus_gt_i),
            }
        )
    return candidates


def _score_positions_from_logits(
    *,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    batch_idx: int,
    padded_len: int,
    full_input_ids: Sequence[int],
    positions: Sequence[int],
) -> Optional[float]:
    seq_len = int(len(full_input_ids))
    pad_offset = int(padded_len - seq_len)
    observed_ids = input_ids[batch_idx, pad_offset:].detach().cpu().tolist()
    if [int(v) for v in observed_ids] != [int(v) for v in full_input_ids]:
        raise RuntimeError("assistant_span_build_failed")
    values: List[float] = []
    for pos in positions:
        abs_pos = int(pad_offset + int(pos))
        if abs_pos <= 0 or abs_pos >= int(input_ids.shape[1]):
            continue
        prev_logits = logits[batch_idx, abs_pos - 1].float()
        target_id = int(input_ids[batch_idx, abs_pos].item())
        target_logit = float(prev_logits[target_id].detach().cpu().item())
        log_norm = float(torch.logsumexp(prev_logits, dim=-1).detach().cpu().item())
        value = float(target_logit - log_norm)
        if math.isfinite(value):
            values.append(value)
    if not values:
        return None
    return float(mean(values))


def _score_candidate_batch(
    scorer: TeacherForcedScorer,
    *,
    prepared_rows: Sequence[Mapping[str, Any]],
    images: Sequence[Image.Image],
) -> List[Dict[str, Optional[float]]]:
    model_inputs = scorer.processor(
        text=[row["prepared"].full_text for row in prepared_rows],
        images=list(images),
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
    if logits.shape[:2] != input_ids.shape[:2]:
        raise RuntimeError("teacher-forced scorer requires unsliced logits")
    padded_len = int(input_ids.shape[1])
    scored: List[Dict[str, Optional[float]]] = []
    for batch_idx, row in enumerate(prepared_rows):
        prepared = row["prepared"]
        scored.append(
            {
                key: _score_positions_from_logits(
                    logits=logits,
                    input_ids=input_ids,
                    batch_idx=batch_idx,
                    padded_len=padded_len,
                    full_input_ids=prepared.full_input_ids,
                    positions=positions,
                )
                for key, positions in row["abs_positions"].items()
            }
        )
    return scored


def _stage_score(
    cfg: StudyConfig,
    *,
    run_dir: Path,
    duplication_cases: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    _set_cuda_visible_devices(cfg.execution.cuda_visible_devices)
    resolved_checkpoint, checkpoint_meta = _resolve_checkpoint(cfg)
    scorer = TeacherForcedScorer(
        checkpoint_path=resolved_checkpoint.path,
        device=cfg.scoring.device,
        attn_implementation=cfg.scoring.attn_implementation,
    )
    selected_cases = list(duplication_cases)[: int(cfg.scoring.max_cases)]
    candidate_rows: List[Dict[str, Any]] = []
    prepared_rows: List[Dict[str, Any]] = []
    images: List[Image.Image] = []
    for case_row in selected_cases:
        case_image = _load_case_image(case_row)
        try:
            candidates = _build_scoring_candidates(
                case_row,
                tokenizer=scorer.tokenizer,
                object_field_order=resolved_checkpoint.object_field_order,
                cfg=cfg,
            )
            for candidate in candidates:
                rel_positions = dict(candidate["rel_positions"])
                full_positions = list(rel_positions["full"])
                prepared = scorer.prepare_example(
                    image=case_image,
                    assistant_text=str(candidate["assistant_text"]),
                    desc_positions_rel=full_positions,
                    prompt_variant=resolved_checkpoint.prompt_variant,
                    object_field_order=resolved_checkpoint.object_field_order,
                )
                full_start = int(prepared.desc_positions[0]) - int(full_positions[0])
                abs_positions = {
                    key: tuple(int(full_start + int(pos)) for pos in positions)
                    for key, positions in rel_positions.items()
                }
                prepared_rows.append({"prepared": prepared, "abs_positions": abs_positions})
                candidate_rows.append(
                    {
                        "case_key": case_row["case_key"],
                        "family": candidate["family"],
                        "label": candidate["label"],
                        "focus_pred_i": candidate["focus_pred_i"],
                        "focus_gt_i": candidate["focus_gt_i"],
                        "candidate_desc": candidate["candidate_desc"],
                        "candidate_bbox_norm1000": candidate["candidate_bbox_norm1000"],
                    }
                )
                images.append(case_image.copy())
        finally:
            case_image.close()
    scored_rows: List[Dict[str, Any]] = []
    for batch_candidates, batch_prepared, batch_images in zip(
        _chunked(candidate_rows, cfg.execution.score_batch_size),
        _chunked(prepared_rows, cfg.execution.score_batch_size),
        _chunked(images, cfg.execution.score_batch_size),
    ):
        try:
            score_dicts = _score_candidate_batch(
                scorer,
                prepared_rows=batch_prepared,
                images=batch_images,
            )
            for candidate, score_dict in zip(batch_candidates, score_dicts):
                scored_rows.append(
                    {
                        **candidate,
                        "score_full": score_dict.get("full"),
                        "score_desc": score_dict.get("desc"),
                        "score_coord": score_dict.get("coord"),
                    }
                )
        finally:
            for image in batch_images:
                image.close()
    score_dir = run_dir / "score"
    _write_jsonl(score_dir / "results.jsonl", scored_rows)
    by_case: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        by_case[str(row["case_key"])].append(row)
    summary_rows: List[Dict[str, Any]] = []
    for case_key, items in sorted(by_case.items()):
        best_duplicate = max(
            (
                float(row["score_full"])
                for row in items
                if row["family"] == "duplicate_jitter" and row.get("score_full") is not None
            ),
            default=None,
        )
        best_gt = max(
            (
                float(row["score_full"])
                for row in items
                if row["family"] == "remaining_gt" and row.get("score_full") is not None
            ),
            default=None,
        )
        close_score = next(
            (
                float(row["score_full"])
                for row in items
                if row["family"] == "close" and row.get("score_full") is not None
            ),
            None,
        )
        summary_rows.append(
            {
                "case_key": case_key,
                "best_duplicate_full": best_duplicate,
                "best_remaining_gt_full": best_gt,
                "close_full": close_score,
                "margin_gt_minus_duplicate": (
                    None
                    if best_gt is None or best_duplicate is None
                    else float(best_gt - best_duplicate)
                ),
                "margin_close_minus_duplicate": (
                    None
                    if close_score is None or best_duplicate is None
                    else float(close_score - best_duplicate)
                ),
            }
        )
    summary = {"checkpoint": checkpoint_meta, "case_summaries": summary_rows}
    _write_json(score_dir / "summary.json", summary)
    return {"results": scored_rows, "summary": summary}


def run_study(config_path: Path) -> Dict[str, Any]:
    cfg = load_study_config(config_path)
    run_dir = _resolve_path(cfg.run.output_dir) / str(cfg.run.name)
    run_dir.mkdir(parents=True, exist_ok=True)
    result: Dict[str, Any] = {
        "config_path": str(config_path),
        "run_dir": str(run_dir),
    }
    cohort_bundle = _stage_cohort(cfg, run_dir=run_dir)
    result["cohort"] = cohort_bundle["summary"]
    selected_rows = list(cohort_bundle["duplication_cases"]) + list(cohort_bundle["controls"])
    if "decode" in cfg.run.stages and selected_rows:
        result["decode"] = _stage_decode(
            cfg,
            run_dir=run_dir,
            selected_rows=selected_rows,
        )["summary"]
    if "prefix" in cfg.run.stages and cohort_bundle["duplication_cases"]:
        result["prefix"] = _stage_prefix(
            cfg,
            run_dir=run_dir,
            duplication_cases=cohort_bundle["duplication_cases"],
        )["summary"]
    if "score" in cfg.run.stages and cohort_bundle["duplication_cases"]:
        result["score"] = _stage_score(
            cfg,
            run_dir=run_dir,
            duplication_cases=cohort_bundle["duplication_cases"],
        )["summary"]
    _write_json(run_dir / "summary.json", result)
    return result


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Offline small-object duplication diagnostics for a fixed CoordExp checkpoint"
    )
    parser.add_argument("--config", required=True, help="Path to the study YAML config")
    args = parser.parse_args(list(argv) if argv is not None else None)
    return run_study(Path(str(args.config)))


def run_small_object_duplication_study(config_path: str) -> Dict[str, Any]:
    return run_study(Path(config_path))


if __name__ == "__main__":
    main()
