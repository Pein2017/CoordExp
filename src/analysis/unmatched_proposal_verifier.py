from __future__ import annotations

import csv
import json
import math
import os
import random
import re
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import torch
import yaml
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from src.common.object_field_order import (
    build_object_payload,
    normalize_object_field_order,
)
from src.common.paths import resolve_image_path_strict
from src.common.semantic_desc import normalize_desc
from src.config.prompts import get_template_prompts
from src.infer.pipeline import run_pipeline
from src.trainers.rollout_matching.contracts import GTObject
from src.trainers.rollout_matching.parsing import (
    find_desc_value_token_positions_by_span,
)
from src.utils.assistant_json import dumps_coordjson


REPO_ROOT = Path(__file__).resolve().parents[2]
_NON_ALNUM_RE = re.compile(r"[^a-zA-Z0-9]+")
_DEFAULT_TOP_K = (10, 25, 50)
_PRIMARY_IOU_THRESHOLD = 0.5


def _common_repo_root() -> Path:
    git_entry = REPO_ROOT / ".git"
    if git_entry.is_file():
        raw = git_entry.read_text(encoding="utf-8").strip()
        if raw.startswith("gitdir:"):
            git_dir = Path(raw.split(":", 1)[1].strip())
            if not git_dir.is_absolute():
                git_dir = (REPO_ROOT / git_dir).resolve()
            if git_dir.parent.name == "worktrees":
                return git_dir.parents[1].parent
    return REPO_ROOT


COMMON_REPO_ROOT = _common_repo_root()


@dataclass(frozen=True)
class CheckpointSpec:
    path: str
    name: str
    prompt_variant: Optional[str] = None
    object_field_order: Optional[str] = None


@dataclass(frozen=True)
class DatasetConfig:
    jsonl_path: str
    sample_count: int = 200
    seed: int = 42


@dataclass(frozen=True)
class CollectionConfig:
    backend_mode: str = "local"
    device: str = "cuda:0"
    cuda_visible_devices: Optional[str] = None
    temperature: float = 0.1
    top_p: float = 0.9
    max_new_tokens: int = 3084
    repetition_penalty: float = 1.1
    batch_size: int = 16
    seed: int = 42
    gpu_memory_utilization: float = 0.9
    gpu_memory_utilization_fallbacks: Tuple[float, ...] = ()
    tensor_parallel_size: int = 1
    max_model_len: int = 12000
    reuse_existing: bool = True


@dataclass(frozen=True)
class EvalConfig:
    semantic_model: str = "model_cache/all-MiniLM-L6-v2-local"
    semantic_threshold: float = 0.5
    semantic_device: str = "cuda:0"
    semantic_batch_size: int = 64
    num_workers: int = 8
    f1ish_iou_thrs: Tuple[float, ...] = (_PRIMARY_IOU_THRESHOLD,)
    f1ish_pred_scope: str = "all"
    use_segm: bool = False


@dataclass(frozen=True)
class ScoringConfig:
    device: str = "cuda:0"
    attn_implementation: str = "auto"
    gt_batch_size: int = 8
    masked_batch_size: int = 8
    rollout_counterfactual_scope: str = "all"
    mask_fill: int = 127


@dataclass(frozen=True)
class PromptConfig:
    prompt_variant: str = "coco_80"
    object_field_order: str = "desc_first"


@dataclass(frozen=True)
class RunConfig:
    name: str
    output_dir: str
    root_image_dir: Optional[str] = None


@dataclass(frozen=True)
class ReportConfig:
    histogram_bins: int = 20
    top_k_values: Tuple[int, ...] = _DEFAULT_TOP_K
    audit_pack_top_n: int = 24


@dataclass(frozen=True)
class StudyConfig:
    run: RunConfig
    dataset: DatasetConfig
    collection: CollectionConfig
    eval: EvalConfig
    scoring: ScoringConfig
    prompts: PromptConfig
    report: ReportConfig
    checkpoints: Tuple[CheckpointSpec, ...]


@dataclass(frozen=True)
class PreparedExample:
    full_text: str
    assistant_text: str
    desc_positions: List[int]
    full_input_ids: List[int]


@dataclass(frozen=True)
class LoadedImage:
    image_path: str
    image: Image.Image
    width: int
    height: int


def _slugify(value: str) -> str:
    stripped = _NON_ALNUM_RE.sub("-", str(value).strip()).strip("-").lower()
    return stripped or "item"


def _load_yaml(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, Mapping):
        raise ValueError(f"Expected YAML mapping at {path}")
    return obj


def _ensure_tuple_floats(values: Any) -> Tuple[float, ...]:
    if values is None:
        return (_PRIMARY_IOU_THRESHOLD,)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("Expected a sequence of numeric values")
    out: List[float] = []
    for value in values:
        out.append(float(value))
    return tuple(out)


def _ensure_tuple_ints(values: Any, *, default: Sequence[int]) -> Tuple[int, ...]:
    if values is None:
        return tuple(int(v) for v in default)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("Expected a sequence of integer values")
    return tuple(int(v) for v in values)


def load_study_config(path: Path) -> StudyConfig:
    raw = _load_yaml(path)

    run_raw = raw.get("run") or {}
    dataset_raw = raw.get("dataset") or {}
    collection_raw = raw.get("collection") or {}
    eval_raw = raw.get("eval") or {}
    scoring_raw = raw.get("scoring") or {}
    prompts_raw = raw.get("prompts") or {}
    report_raw = raw.get("report") or {}
    checkpoints_raw = raw.get("checkpoints") or []

    if not isinstance(checkpoints_raw, Sequence) or not checkpoints_raw:
        raise ValueError("study config must define a non-empty checkpoints list")

    checkpoints: List[CheckpointSpec] = []
    for idx, ckpt_raw in enumerate(checkpoints_raw):
        if not isinstance(ckpt_raw, Mapping):
            raise ValueError(f"checkpoint[{idx}] must be a mapping")
        ckpt_path = str(ckpt_raw.get("path") or "").strip()
        if not ckpt_path:
            raise ValueError(f"checkpoint[{idx}].path is required")
        ckpt_name = str(ckpt_raw.get("name") or Path(ckpt_path).name).strip()
        checkpoints.append(
            CheckpointSpec(
                path=ckpt_path,
                name=ckpt_name,
                prompt_variant=(
                    str(ckpt_raw["prompt_variant"]).strip()
                    if ckpt_raw.get("prompt_variant") is not None
                    else None
                ),
                object_field_order=(
                    normalize_object_field_order(
                        str(ckpt_raw["object_field_order"]),
                        path=f"checkpoints[{idx}].object_field_order",
                    )
                    if ckpt_raw.get("object_field_order") is not None
                    else None
                ),
            )
        )

    run = RunConfig(
        name=str(run_raw.get("name") or "unmatched-proposal-verifier").strip(),
        output_dir=str(run_raw.get("output_dir") or "output/analysis").strip(),
        root_image_dir=(
            str(run_raw["root_image_dir"]).strip()
            if run_raw.get("root_image_dir") is not None
            else None
        ),
    )
    dataset = DatasetConfig(
        jsonl_path=str(dataset_raw.get("jsonl_path") or "").strip(),
        sample_count=int(dataset_raw.get("sample_count", 200)),
        seed=int(dataset_raw.get("seed", 42)),
    )
    if not dataset.jsonl_path:
        raise ValueError("dataset.jsonl_path is required")
    collection = CollectionConfig(
        backend_mode=str(collection_raw.get("backend_mode", "local")).strip().lower(),
        device=str(collection_raw.get("device", "cuda:0")).strip(),
        cuda_visible_devices=(
            str(collection_raw.get("cuda_visible_devices")).strip()
            if collection_raw.get("cuda_visible_devices") is not None
            else None
        ),
        temperature=float(collection_raw.get("temperature", 0.1)),
        top_p=float(collection_raw.get("top_p", 0.9)),
        max_new_tokens=int(collection_raw.get("max_new_tokens", 3084)),
        repetition_penalty=float(collection_raw.get("repetition_penalty", 1.1)),
        batch_size=int(collection_raw.get("batch_size", 16)),
        seed=int(collection_raw.get("seed", 42)),
        gpu_memory_utilization=float(collection_raw.get("gpu_memory_utilization", 0.9)),
        gpu_memory_utilization_fallbacks=(
            _ensure_tuple_floats(collection_raw.get("gpu_memory_utilization_fallbacks"))
            if collection_raw.get("gpu_memory_utilization_fallbacks") is not None
            else ()
        ),
        tensor_parallel_size=int(collection_raw.get("tensor_parallel_size", 1)),
        max_model_len=int(collection_raw.get("max_model_len", 12000)),
        reuse_existing=bool(collection_raw.get("reuse_existing", True)),
    )
    eval_cfg = EvalConfig(
        semantic_model=str(
            eval_raw.get("semantic_model", "model_cache/all-MiniLM-L6-v2-local")
        ),
        semantic_threshold=float(eval_raw.get("semantic_threshold", 0.5)),
        semantic_device=str(eval_raw.get("semantic_device", "cuda:0")),
        semantic_batch_size=int(eval_raw.get("semantic_batch_size", 64)),
        num_workers=int(eval_raw.get("num_workers", 8)),
        f1ish_iou_thrs=_ensure_tuple_floats(eval_raw.get("f1ish_iou_thrs")),
        f1ish_pred_scope=str(eval_raw.get("f1ish_pred_scope", "all")).strip().lower(),
        use_segm=bool(eval_raw.get("use_segm", False)),
    )
    scoring = ScoringConfig(
        device=str(scoring_raw.get("device", "cuda:0")),
        attn_implementation=str(scoring_raw.get("attn_implementation", "auto")),
        gt_batch_size=int(scoring_raw.get("gt_batch_size", 8)),
        masked_batch_size=int(scoring_raw.get("masked_batch_size", 8)),
        rollout_counterfactual_scope=str(
            scoring_raw.get("rollout_counterfactual_scope", "all")
        ).strip(),
        mask_fill=int(scoring_raw.get("mask_fill", 127)),
    )
    prompts = PromptConfig(
        prompt_variant=str(prompts_raw.get("prompt_variant", "coco_80")).strip(),
        object_field_order=normalize_object_field_order(
            str(prompts_raw.get("object_field_order", "desc_first")),
            path="prompts.object_field_order",
        ),
    )
    report = ReportConfig(
        histogram_bins=int(report_raw.get("histogram_bins", 20)),
        top_k_values=_ensure_tuple_ints(
            report_raw.get("top_k_values"),
            default=_DEFAULT_TOP_K,
        ),
        audit_pack_top_n=int(report_raw.get("audit_pack_top_n", 24)),
    )
    return StudyConfig(
        run=run,
        dataset=dataset,
        collection=collection,
        eval=eval_cfg,
        scoring=scoring,
        prompts=prompts,
        report=report,
        checkpoints=tuple(checkpoints),
    )


def _iter_jsonl(path: Path) -> Iterator[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            if isinstance(obj, Mapping):
                yield obj


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _reservoir_sample_records(
    input_path: Path, num_samples: int, seed: int
) -> Tuple[List[Tuple[int, str, Mapping[str, Any]]], int]:
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")
    rng = random.Random(seed)
    reservoir: List[Tuple[int, str, Mapping[str, Any]]] = []
    total_nonempty = 0
    with input_path.open("r", encoding="utf-8") as f:
        for line_idx, raw_line in enumerate(f):
            line = raw_line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, Mapping):
                continue
            total_nonempty += 1
            item = (int(line_idx), line, obj)
            if len(reservoir) < num_samples:
                reservoir.append(item)
                continue
            j = rng.randrange(total_nonempty)
            if j < num_samples:
                reservoir[j] = item
    if total_nonempty < num_samples:
        raise ValueError(
            f"Requested {num_samples} samples from {input_path}, found only {total_nonempty}"
        )
    rng.shuffle(reservoir)
    return reservoir, total_nonempty


def _load_subset_records(path: Path) -> List[Mapping[str, Any]]:
    return list(_iter_jsonl(path))


def _resolve_existing_input_path(path_raw: str) -> Path:
    raw = Path(str(path_raw))
    if raw.is_absolute():
        return raw.absolute()
    for root in (REPO_ROOT, COMMON_REPO_ROOT):
        candidate = (root / raw).absolute()
        if candidate.exists():
            return candidate
    return (COMMON_REPO_ROOT / raw).absolute()


def materialize_subset(
    config: StudyConfig, run_dir: Path
) -> Tuple[Path, List[Mapping[str, Any]], Dict[str, Any]]:
    subset_dir = run_dir / "subset"
    subset_path = subset_dir / "sampled.coord.jsonl"
    meta_path = subset_dir / "sampled.coord.jsonl.meta.json"
    source_path = _resolve_existing_input_path(config.dataset.jsonl_path)
    root_image_dir = (
        _resolve_existing_input_path(config.run.root_image_dir)
        if config.run.root_image_dir
        else source_path.parent.resolve()
    )

    if subset_path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            str(meta.get("input_path")) == str(source_path)
            and int(meta.get("num_samples", -1)) == int(config.dataset.sample_count)
            and int(meta.get("seed", -1)) == int(config.dataset.seed)
            and str(meta.get("root_image_dir")) == str(root_image_dir)
            and int(meta.get("input_mtime_ns", -1))
            == int(source_path.stat().st_mtime_ns)
        ):
            return subset_path, _load_subset_records(subset_path), meta

    sampled, total_nonempty = _reservoir_sample_records(
        source_path,
        num_samples=int(config.dataset.sample_count),
        seed=int(config.dataset.seed),
    )
    subset_dir.mkdir(parents=True, exist_ok=True)
    with subset_path.open("w", encoding="utf-8") as f:
        for _line_idx, raw_line, _obj in sampled:
            f.write(raw_line)
            f.write("\n")
    sampled_indices = [int(line_idx) for line_idx, _raw_line, _obj in sampled]
    sampled_images = [
        str(((obj.get("images") or [None])[0] or obj.get("image") or ""))
        for _line_idx, _raw_line, obj in sampled
    ]
    meta = {
        "input_path": str(source_path),
        "output_path": str(subset_path),
        "num_samples": int(config.dataset.sample_count),
        "seed": int(config.dataset.seed),
        "total_nonempty_lines_seen": int(total_nonempty),
        "sampled_source_line_indices": sampled_indices,
        "sampled_images": sampled_images,
        "root_image_dir": str(root_image_dir),
        "input_mtime_ns": int(source_path.stat().st_mtime_ns),
    }
    _write_json(meta_path, meta)
    return subset_path, _load_subset_records(subset_path), meta


def _candidate_checkpoint_paths(path_raw: str) -> List[Path]:
    raw = Path(path_raw)
    candidates: List[Path] = []
    if raw.is_absolute():
        candidates.append(raw.absolute())
    else:
        candidates.append((REPO_ROOT / raw).absolute())
        candidates.append((COMMON_REPO_ROOT / raw).absolute())
    candidates.append((COMMON_REPO_ROOT / "result" / raw.name).absolute())
    return candidates


def resolve_checkpoint_path(path_raw: str) -> Tuple[Path, str]:
    for idx, candidate in enumerate(_candidate_checkpoint_paths(path_raw)):
        if candidate.exists():
            if Path(path_raw).is_absolute():
                source = "config_path_absolute"
            elif idx == 0:
                source = "config_path_worktree"
            elif idx == 1:
                source = "config_path_common_root"
            else:
                source = "result_basename_fallback"
            return candidate, source
    raise FileNotFoundError(
        f"Could not resolve checkpoint path from {path_raw!r}; checked {[str(p) for p in _candidate_checkpoint_paths(path_raw)]}"
    )


def _extract_prompt_controls_from_mapping(
    raw: Mapping[str, Any],
) -> Tuple[Optional[str], Optional[str]]:
    prompt_variant: Optional[str] = None
    object_field_order: Optional[str] = None

    infer_raw = raw.get("infer")
    if isinstance(infer_raw, Mapping):
        if infer_raw.get("prompt_variant") is not None:
            prompt_variant = str(infer_raw.get("prompt_variant")).strip()
        if infer_raw.get("object_field_order") is not None:
            object_field_order = normalize_object_field_order(
                str(infer_raw.get("object_field_order")),
                path="resolved.infer.object_field_order",
            )

    cfg_raw = raw.get("cfg")
    if isinstance(cfg_raw, Mapping):
        infer_cfg = cfg_raw.get("infer")
        if isinstance(infer_cfg, Mapping):
            if prompt_variant is None and infer_cfg.get("prompt_variant") is not None:
                prompt_variant = str(infer_cfg.get("prompt_variant")).strip()
            if (
                object_field_order is None
                and infer_cfg.get("object_field_order") is not None
            ):
                object_field_order = normalize_object_field_order(
                    str(infer_cfg.get("object_field_order")),
                    path="resolved.cfg.infer.object_field_order",
                )
    return prompt_variant, object_field_order


def resolve_prompt_controls_for_checkpoint(
    checkpoint_path: Path,
    *,
    default_prompt_variant: str,
    default_object_field_order: str,
    override_prompt_variant: Optional[str],
    override_object_field_order: Optional[str],
) -> Tuple[str, str, str]:
    if override_prompt_variant is not None and override_object_field_order is not None:
        return (
            override_prompt_variant,
            override_object_field_order,
            "checkpoint_override",
        )

    candidates = [
        checkpoint_path / "resolved_config.json",
        checkpoint_path / "run_metadata.json",
        checkpoint_path.parent / "resolved_config.json",
        checkpoint_path.parent / "run_metadata.json",
    ]
    pointer = checkpoint_path / "resolved_config.path"
    if pointer.is_file():
        try:
            pointed = Path(pointer.read_text(encoding="utf-8").strip())
            candidates.insert(0, pointed)
        except OSError:
            pass

    found_prompt = override_prompt_variant
    found_order = override_object_field_order
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            raw = json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(raw, Mapping):
            continue
        prompt_variant, object_field_order = _extract_prompt_controls_from_mapping(raw)
        if found_prompt is None and prompt_variant:
            found_prompt = prompt_variant
        if found_order is None and object_field_order:
            found_order = object_field_order
        if found_prompt is not None and found_order is not None:
            return found_prompt, found_order, str(candidate)

    return (
        found_prompt or default_prompt_variant,
        found_order or default_object_field_order,
        "study_default",
    )


def _sanitize_bbox_xyxy(
    box: Sequence[float], width: int, height: int
) -> Optional[List[float]]:
    if len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except (TypeError, ValueError):
        return None
    x1 = min(max(x1, 0.0), float(width))
    x2 = min(max(x2, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    y2 = min(max(y2, 0.0), float(height))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _bbox_iou_xyxy(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    if len(box_a) != 4 or len(box_b) != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def _coord_token_to_int(value: Any) -> Optional[int]:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("<|coord_") and stripped.endswith("|>"):
            middle = stripped[len("<|coord_") : -2]
            if middle.isdigit():
                return int(middle)
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _parse_norm1000_bbox(values: Any) -> Optional[List[int]]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return None
    out: List[int] = []
    for value in values:
        parsed = _coord_token_to_int(value)
        if parsed is None:
            return None
        out.append(int(parsed))
    if len(out) != 4:
        return None
    if out[2] <= out[0] or out[3] <= out[1]:
        return None
    return out


def _norm1000_to_pixel(box: Sequence[int], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = [int(v) for v in box]
    return [
        float(x1) * float(width) / 1000.0,
        float(y1) * float(height) / 1000.0,
        float(x2) * float(width) / 1000.0,
        float(y2) * float(height) / 1000.0,
    ]


def _pixel_to_norm1000(
    box: Sequence[float], width: int, height: int
) -> Optional[List[int]]:
    if width <= 0 or height <= 0:
        return None
    clean = _sanitize_bbox_xyxy(box, width, height)
    if clean is None:
        return None
    x1, y1, x2, y2 = clean
    bins = [
        int(round((x1 / float(width)) * 1000.0)),
        int(round((y1 / float(height)) * 1000.0)),
        int(round((x2 / float(width)) * 1000.0)),
        int(round((y2 / float(height)) * 1000.0)),
    ]
    bins = [min(999, max(0, int(v))) for v in bins]
    if bins[2] <= bins[0] or bins[3] <= bins[1]:
        return None
    return bins


def _extract_dataset_gt_bbox(obj: Mapping[str, Any]) -> Optional[List[int]]:
    for key in ("bbox_2d", "bbox"):
        bbox = _parse_norm1000_bbox(obj.get(key))
        if bbox is not None:
            return bbox
    points = obj.get("points")
    bbox = _parse_norm1000_bbox(points)
    if bbox is not None:
        return bbox
    return None


def _extract_pixel_bbox(
    obj: Mapping[str, Any], width: int, height: int
) -> Optional[List[float]]:
    for key in ("bbox", "bbox_2d"):
        bbox_raw = obj.get(key)
        if isinstance(bbox_raw, Sequence) and not isinstance(bbox_raw, (str, bytes)):
            bbox = _sanitize_bbox_xyxy(bbox_raw, width, height)
            if bbox is not None:
                return bbox
    points_raw = obj.get("points")
    if isinstance(points_raw, Sequence) and not isinstance(points_raw, (str, bytes)):
        bbox = _sanitize_bbox_xyxy(points_raw, width, height)
        if bbox is not None:
            return bbox
    return None


def _build_bbox_gt_object(
    index: int, desc: str, bbox_norm1000: Sequence[int]
) -> GTObject:
    return GTObject(
        index=int(index),
        geom_type="bbox_2d",
        points_norm1000=[int(v) for v in bbox_norm1000],
        desc=str(desc),
    )


def _build_closed_container_text(
    objects: Sequence[GTObject],
    *,
    object_field_order: str,
) -> str:
    payload: Dict[str, Any] = {"objects": []}
    rendered = payload["objects"]
    if not isinstance(rendered, list):
        raise RuntimeError("internal error: objects payload is not a list")
    for obj in objects:
        rendered.append(
            build_object_payload(
                desc=str(obj.desc),
                geometry_key="bbox_2d",
                geometry_value=[f"<|coord_{int(v)}|>" for v in obj.points_norm1000],
                object_field_order=object_field_order,
            )
        )
    return dumps_coordjson(payload)


def _extract_rollout_objects_for_scoring(
    pred_objects: Sequence[Mapping[str, Any]],
    *,
    width: int,
    height: int,
) -> Tuple[Optional[List[GTObject]], Optional[str]]:
    out: List[GTObject] = []
    for pred_idx, obj in enumerate(pred_objects):
        desc = str(obj.get("desc") or "").strip()
        if not desc:
            return None, "sequence_canonicalization_failed"
        bbox_px = _extract_pixel_bbox(obj, width, height)
        if bbox_px is None:
            return None, "sequence_canonicalization_failed"
        bbox_norm = _pixel_to_norm1000(bbox_px, width, height)
        if bbox_norm is None:
            return None, "sequence_canonicalization_failed"
        out.append(_build_bbox_gt_object(pred_idx, desc, bbox_norm))
    return out, None


def _dedup_rows(
    rows: Sequence[Mapping[str, Any]], *, keys: Sequence[str]
) -> List[Dict[str, Any]]:
    seen: set[Tuple[Any, ...]] = set()
    out: List[Dict[str, Any]] = []
    for row in rows:
        normalized_key: List[Any] = []
        for key_name in keys:
            value = row.get(key_name)
            if isinstance(value, list):
                normalized_key.append(tuple(value))
            elif isinstance(value, dict):
                normalized_key.append(tuple(sorted(value.items())))
            else:
                normalized_key.append(value)
        key = tuple(normalized_key)
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(row))
    return out


def build_gt_and_negative_tables(
    subset_records: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    positives: List[Dict[str, Any]] = []
    negatives: List[Dict[str, Any]] = []
    for image_idx, record in enumerate(subset_records):
        width = int(record.get("width") or 0)
        height = int(record.get("height") or 0)
        image_rel = str(
            ((record.get("images") or [None])[0] or record.get("image") or "")
        )
        gt_rows: List[Dict[str, Any]] = []
        for gt_idx, obj in enumerate(record.get("objects") or []):
            if not isinstance(obj, Mapping):
                continue
            desc = str(obj.get("desc") or "").strip()
            bbox_norm = _extract_dataset_gt_bbox(obj)
            if not desc or bbox_norm is None:
                continue
            bbox_px = _norm1000_to_pixel(bbox_norm, width, height)
            row = {
                "image_idx": int(image_idx),
                "image_path": image_rel,
                "width": int(width),
                "height": int(height),
                "gt_idx": int(gt_idx),
                "desc": desc,
                "bbox_norm1000": list(bbox_norm),
                "bbox_pixel": list(bbox_px),
                "label": 1,
                "row_family": "gt_positive",
                "source_gt_idx": int(gt_idx),
                "source_image_idx": int(image_idx),
            }
            positives.append(dict(row))
            gt_rows.append(row)

        for row in gt_rows:
            bbox = [int(v) for v in row["bbox_norm1000"]]
            dx = max(60, int(round(0.35 * float(bbox[2] - bbox[0]))))
            dy = max(60, int(round(0.35 * float(bbox[3] - bbox[1]))))
            shifted = [
                min(999, bbox[0] + dx),
                min(999, bbox[1] + dy),
                min(999, bbox[2] + dx),
                min(999, bbox[3] + dy),
            ]
            if shifted[2] > shifted[0] and shifted[3] > shifted[1]:
                shifted_iou = _bbox_iou_xyxy(
                    _norm1000_to_pixel(bbox, width, height),
                    _norm1000_to_pixel(shifted, width, height),
                )
                if shifted_iou < 0.2:
                    negatives.append(
                        {
                            **row,
                            "label": 0,
                            "row_family": "same_desc_wrong_location_jitter",
                            "bbox_norm1000": shifted,
                            "bbox_pixel": _norm1000_to_pixel(shifted, width, height),
                        }
                    )

        if len(gt_rows) >= 2:
            for src, donor in zip(gt_rows, gt_rows[1:] + gt_rows[:1]):
                negatives.append(
                    {
                        **src,
                        "label": 0,
                        "row_family": "desc_box_cross_swap",
                        "bbox_norm1000": list(donor["bbox_norm1000"]),
                        "bbox_pixel": list(donor["bbox_pixel"]),
                        "source_gt_idx": int(src["gt_idx"]),
                        "donor_gt_idx": int(donor["gt_idx"]),
                    }
                )

        desc_groups: Dict[str, List[Dict[str, Any]]] = {}
        for row in gt_rows:
            desc_groups.setdefault(normalize_desc(str(row["desc"])), []).append(row)
        for grouped in desc_groups.values():
            if len(grouped) < 2:
                continue
            for src, donor in zip(grouped, grouped[1:] + grouped[:1]):
                negatives.append(
                    {
                        **src,
                        "label": 0,
                        "row_family": "same_class_wrong_location",
                        "bbox_norm1000": list(donor["bbox_norm1000"]),
                        "bbox_pixel": list(donor["bbox_pixel"]),
                        "source_gt_idx": int(src["gt_idx"]),
                        "donor_gt_idx": int(donor["gt_idx"]),
                    }
                )

    negatives = _dedup_rows(
        negatives,
        keys=("image_idx", "desc", "row_family", "bbox_norm1000"),
    )
    return positives, negatives


def _build_pipeline_yaml(
    *,
    output_root: Path,
    subset_path: Path,
    root_image_dir: Path,
    checkpoint_name: str,
    checkpoint_path: Path,
    prompt_variant: str,
    object_field_order: str,
    collection: CollectionConfig,
    eval_cfg: EvalConfig,
) -> Dict[str, Any]:
    backend_mode = str(collection.backend_mode).lower()
    backend_type = "hf" if backend_mode == "hf" else "vllm"
    return {
        "run": {
            "name": checkpoint_name,
            "output_dir": str(output_root),
            "root_image_dir": str(root_image_dir),
        },
        "stages": {"infer": True, "eval": True, "vis": False},
        "infer": {
            "gt_jsonl": str(subset_path),
            "model_checkpoint": str(checkpoint_path),
            "prompt_variant": prompt_variant,
            "object_field_order": object_field_order,
            "mode": "auto",
            "pred_coord_mode": "auto",
            "backend": (
                {
                    "type": "hf",
                }
                if backend_type == "hf"
                else {
                    "type": "vllm",
                    "mode": str(collection.backend_mode),
                    "auto_launch": True,
                    "base_url": "http://127.0.0.1:8000",
                    "server_options": {
                        "vllm_tensor_parallel_size": int(collection.tensor_parallel_size),
                        "vllm_gpu_memory_utilization": float(
                            collection.gpu_memory_utilization
                        ),
                        "vllm_max_model_len": int(collection.max_model_len),
                    },
                }
            ),
            "generation": {
                "temperature": float(collection.temperature),
                "top_p": float(collection.top_p),
                "max_new_tokens": int(collection.max_new_tokens),
                "repetition_penalty": float(collection.repetition_penalty),
                "batch_size": int(collection.batch_size),
                "seed": int(collection.seed),
            },
            "device": str(collection.device),
            "limit": 0,
            "detect_samples": 128,
        },
        "eval": {
            "output_dir": None,
            "metrics": "f1ish",
            "use_segm": bool(eval_cfg.use_segm),
            "overlay": False,
            "overlay_k": 12,
            "num_workers": int(eval_cfg.num_workers),
            "semantic_model": str(
                _resolve_existing_input_path(eval_cfg.semantic_model)
                if str(eval_cfg.semantic_model).strip().startswith("model_cache/")
                else eval_cfg.semantic_model
            ),
            "semantic_threshold": float(eval_cfg.semantic_threshold),
            "semantic_device": str(eval_cfg.semantic_device),
            "semantic_batch_size": int(eval_cfg.semantic_batch_size),
            "f1ish_iou_thrs": [float(v) for v in eval_cfg.f1ish_iou_thrs],
            "f1ish_pred_scope": str(eval_cfg.f1ish_pred_scope),
        },
    }


@contextmanager
def _temporary_cuda_visible_devices(cuda_visible_devices: Optional[str]):
    if cuda_visible_devices is None or not str(cuda_visible_devices).strip():
        yield
        return
    key = "CUDA_VISIBLE_DEVICES"
    old_value = os.environ.get(key)
    os.environ[key] = str(cuda_visible_devices).strip()
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old_value


def _local_vllm_utilization_candidates(collection: CollectionConfig) -> List[float]:
    values: List[float] = [float(collection.gpu_memory_utilization)]
    for value in collection.gpu_memory_utilization_fallbacks:
        fvalue = float(value)
        if fvalue not in values:
            values.append(fvalue)
    return values


def _collection_fingerprint(
    *,
    subset_path: Path,
    root_image_dir: Path,
    checkpoint_path: Path,
    prompt_variant: str,
    object_field_order: str,
    collection: CollectionConfig,
    eval_cfg: EvalConfig,
) -> Dict[str, Any]:
    return {
        "subset_path": str(subset_path),
        "root_image_dir": str(root_image_dir),
        "checkpoint_path": str(checkpoint_path),
        "prompt_variant": str(prompt_variant),
        "object_field_order": str(object_field_order),
        "backend_mode": str(collection.backend_mode),
        "device": str(collection.device),
        "cuda_visible_devices": collection.cuda_visible_devices,
        "temperature": float(collection.temperature),
        "top_p": float(collection.top_p),
        "max_new_tokens": int(collection.max_new_tokens),
        "repetition_penalty": float(collection.repetition_penalty),
        "batch_size": int(collection.batch_size),
        "seed": int(collection.seed),
        "gpu_memory_utilization": float(collection.gpu_memory_utilization),
        "gpu_memory_utilization_fallbacks": [
            float(v) for v in collection.gpu_memory_utilization_fallbacks
        ],
        "tensor_parallel_size": int(collection.tensor_parallel_size),
        "max_model_len": int(collection.max_model_len),
        "semantic_model": str(eval_cfg.semantic_model),
        "semantic_threshold": float(eval_cfg.semantic_threshold),
        "semantic_device": str(eval_cfg.semantic_device),
        "semantic_batch_size": int(eval_cfg.semantic_batch_size),
        "num_workers": int(eval_cfg.num_workers),
        "f1ish_iou_thrs": [float(v) for v in eval_cfg.f1ish_iou_thrs],
        "f1ish_pred_scope": str(eval_cfg.f1ish_pred_scope),
        "use_segm": bool(eval_cfg.use_segm),
    }


def run_collection_for_checkpoint(
    config: StudyConfig,
    *,
    subset_path: Path,
    root_image_dir: Path,
    checkpoint: CheckpointSpec,
    run_dir: Path,
) -> Dict[str, Any]:
    resolved_checkpoint_path, checkpoint_path_source = resolve_checkpoint_path(
        checkpoint.path
    )
    prompt_variant, object_field_order, prompt_source = (
        resolve_prompt_controls_for_checkpoint(
            resolved_checkpoint_path,
            default_prompt_variant=config.prompts.prompt_variant,
            default_object_field_order=config.prompts.object_field_order,
            override_prompt_variant=checkpoint.prompt_variant,
            override_object_field_order=checkpoint.object_field_order,
        )
    )

    output_root = run_dir.parent
    pipeline_cfg = _build_pipeline_yaml(
        output_root=output_root,
        subset_path=subset_path,
        root_image_dir=root_image_dir,
        checkpoint_name=run_dir.name,
        checkpoint_path=resolved_checkpoint_path,
        prompt_variant=prompt_variant,
        object_field_order=object_field_order,
        collection=config.collection,
        eval_cfg=config.eval,
    )

    pipeline_cfg_path = run_dir / "pipeline_config.yaml"
    pipeline_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline_cfg_path.write_text(
        yaml.safe_dump(pipeline_cfg, sort_keys=False),
        encoding="utf-8",
    )

    gt_vs_pred_path = run_dir / "gt_vs_pred.jsonl"
    matches_path = run_dir / "eval" / "matches.jsonl"
    selected_gpu_utilization = float(config.collection.gpu_memory_utilization)
    collection_error: Optional[Exception] = None
    fingerprint = _collection_fingerprint(
        subset_path=subset_path,
        root_image_dir=root_image_dir,
        checkpoint_path=resolved_checkpoint_path,
        prompt_variant=prompt_variant,
        object_field_order=object_field_order,
        collection=config.collection,
        eval_cfg=config.eval,
    )
    manifest_path = run_dir / "checkpoint_manifest.json"
    prior_manifest: Dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded_manifest, dict):
                prior_manifest = loaded_manifest
        except json.JSONDecodeError:
            prior_manifest = {}
    if (
        config.collection.reuse_existing
        and gt_vs_pred_path.is_file()
        and matches_path.is_file()
        and prior_manifest.get("collection_fingerprint") == fingerprint
    ):
        status = "reused"
    else:
        status = "collected"
        candidates = _local_vllm_utilization_candidates(config.collection)
        if str(config.collection.backend_mode).lower() == "local":
            for util in candidates:
                selected_gpu_utilization = float(util)
                (
                    pipeline_cfg["infer"]["backend"]["server_options"][
                        "vllm_gpu_memory_utilization"
                    ]
                ) = float(util)
                pipeline_cfg_path.write_text(
                    yaml.safe_dump(pipeline_cfg, sort_keys=False),
                    encoding="utf-8",
                )
                try:
                    with _temporary_cuda_visible_devices(
                        config.collection.cuda_visible_devices
                    ):
                        run_pipeline(config_path=pipeline_cfg_path)
                    collection_error = None
                    break
                except (RuntimeError, ValueError) as exc:
                    collection_error = exc
                    if float(util) == float(candidates[-1]):
                        raise
            if collection_error:
                raise RuntimeError(str(collection_error))
        else:
            pipeline_cfg_path.write_text(
                yaml.safe_dump(pipeline_cfg, sort_keys=False),
                encoding="utf-8",
            )
            with _temporary_cuda_visible_devices(config.collection.cuda_visible_devices):
                run_pipeline(config_path=pipeline_cfg_path)

    manifest = {
        "checkpoint_name": checkpoint.name,
        "checkpoint_path_raw": checkpoint.path,
        "checkpoint_path_resolved": str(resolved_checkpoint_path),
        "checkpoint_path_source": checkpoint_path_source,
        "prompt_variant": prompt_variant,
        "object_field_order": object_field_order,
        "prompt_control_source": prompt_source,
        "semantic_model": config.eval.semantic_model,
        "collection_status": status,
        "pipeline_config_path": str(pipeline_cfg_path),
        "collection_device": str(config.collection.device),
        "collection_cuda_visible_devices": config.collection.cuda_visible_devices,
        "collection_gpu_memory_utilization_requested": float(
            config.collection.gpu_memory_utilization
        ),
        "collection_gpu_memory_utilization_used": float(selected_gpu_utilization),
        "collection_fingerprint": fingerprint,
    }
    _write_json(manifest_path, manifest)
    return manifest


def _nearest_gt_metadata(
    pred_bbox: Sequence[float],
    gt_objects: Sequence[Mapping[str, Any]],
    *,
    width: int,
    height: int,
) -> Dict[str, Any]:
    best_iou = -1.0
    best_idx = None
    best_desc = None
    for gt_idx, gt in enumerate(gt_objects):
        if not isinstance(gt, Mapping):
            continue
        gt_bbox = _extract_pixel_bbox(gt, width, height)
        if gt_bbox is None:
            continue
        iou = _bbox_iou_xyxy(pred_bbox, gt_bbox)
        if iou > best_iou:
            best_iou = float(iou)
            best_idx = int(gt_idx)
            best_desc = str(gt.get("desc") or "")
    return {
        "nearest_gt_idx": best_idx,
        "nearest_gt_desc": best_desc,
        "nearest_gt_iou": float(best_iou) if best_iou >= 0.0 else None,
    }


def build_rollout_proposal_table(
    *,
    checkpoint_name: str,
    run_dir: Path,
) -> List[Dict[str, Any]]:
    gt_vs_pred_rows = list(_iter_jsonl(run_dir / "gt_vs_pred.jsonl"))
    matches_by_image = {
        int(row.get("image_id")): row
        for row in _iter_jsonl(run_dir / "eval" / "matches.jsonl")
    }
    rows: List[Dict[str, Any]] = []
    for image_idx, gt_pred_row in enumerate(gt_vs_pred_rows):
        width = int(gt_pred_row.get("width") or 0)
        height = int(gt_pred_row.get("height") or 0)
        preds = list(gt_pred_row.get("pred") or [])
        gts = list(gt_pred_row.get("gt") or [])
        match_row = matches_by_image.get(int(image_idx), {})
        matched_indices = {
            int(m["pred_idx"])
            for m in (match_row.get("matches") or [])
            if isinstance(m, Mapping) and m.get("pred_idx") is not None
        }
        unmatched_indices = {
            int(v)
            for v in (match_row.get("unmatched_pred_indices") or [])
            if v is not None
        }
        ignored_indices = {
            int(v)
            for v in (match_row.get("ignored_pred_indices") or [])
            if v is not None
        }

        image_rows: List[Dict[str, Any]] = []
        for pred_idx, pred in enumerate(preds):
            if not isinstance(pred, Mapping):
                continue
            desc = str(pred.get("desc") or "").strip()
            bbox_px = _extract_pixel_bbox(pred, width, height)
            bbox_norm = (
                _pixel_to_norm1000(bbox_px, width, height)
                if bbox_px is not None
                else None
            )
            nearest = (
                _nearest_gt_metadata(bbox_px, gts, width=width, height=height)
                if bbox_px is not None
                else {
                    "nearest_gt_idx": None,
                    "nearest_gt_desc": None,
                    "nearest_gt_iou": None,
                }
            )
            if pred_idx in matched_indices:
                match_status = "matched"
            elif pred_idx in unmatched_indices:
                match_status = "unmatched"
            elif pred_idx in ignored_indices:
                match_status = "ignored"
            else:
                match_status = "unknown"
            row = {
                "checkpoint": checkpoint_name,
                "image_idx": int(image_idx),
                "image_path": str(gt_pred_row.get("image") or ""),
                "width": int(width),
                "height": int(height),
                "proposal_index": int(pred_idx),
                "desc": desc,
                "bbox_pixel": list(bbox_px) if bbox_px is not None else None,
                "bbox_norm1000": list(bbox_norm) if bbox_norm is not None else None,
                "match_status": match_status,
                "is_matched": int(match_status == "matched"),
                "is_unmatched": int(match_status == "unmatched"),
                "is_ignored": int(match_status == "ignored"),
                "pred_count": int(len(preds)),
                **nearest,
                "scoring_status": "pending",
                "failure_reason": None,
                "commitment": None,
                "masked_commitment": None,
                "counterfactual": None,
                "combined_linear": None,
            }
            rows.append(row)
            image_rows.append(row)

        for i, row_i in enumerate(image_rows):
            row_i["duplicate_like_same_desc_iou90"] = 0
            row_i["duplicate_like_any_desc_iou90"] = 0
            bbox_i = row_i.get("bbox_norm1000")
            if not isinstance(bbox_i, list):
                continue
            for j in range(i + 1, len(image_rows)):
                row_j = image_rows[j]
                bbox_j = row_j.get("bbox_norm1000")
                if not isinstance(bbox_j, list):
                    continue
                iou = _bbox_iou_xyxy(
                    _norm1000_to_pixel(bbox_i, width, height),
                    _norm1000_to_pixel(bbox_j, width, height),
                )
                if iou < 0.90:
                    continue
                row_i["duplicate_like_any_desc_iou90"] = 1
                row_j["duplicate_like_any_desc_iou90"] = 1
                if normalize_desc(str(row_i["desc"])) == normalize_desc(
                    str(row_j["desc"])
                ):
                    row_i["duplicate_like_same_desc_iou90"] = 1
                    row_j["duplicate_like_same_desc_iou90"] = 1
    return rows


class TeacherForcedScorer:
    def __init__(
        self,
        *,
        checkpoint_path: Path,
        device: str,
        attn_implementation: str = "auto",
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.attn_implementation = attn_implementation
        self.model = self._load_model()
        self.processor = AutoProcessor.from_pretrained(
            str(checkpoint_path), trust_remote_code=True
        )
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("AutoProcessor did not expose a tokenizer")
        try:
            setattr(tokenizer, "padding_side", "left")
            if getattr(tokenizer, "pad_token_id", None) is None:
                eos_token_id = getattr(tokenizer, "eos_token_id", None)
                if eos_token_id is None:
                    raise ValueError("tokenizer.eos_token_id is required")
                setattr(tokenizer, "pad_token_id", eos_token_id)
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "Failed to configure tokenizer padding for scorer"
            ) from exc
        self.tokenizer = tokenizer

    def _load_model(self) -> Qwen3VLForConditionalGeneration:
        requested = str(self.attn_implementation or "auto").strip().lower()
        if not requested or requested == "auto":
            if "cuda" in str(self.device).lower() and torch.cuda.is_available():
                requested = "flash_attention_2"
            else:
                requested = "sdpa"
        candidates: List[str] = []
        for cand in (requested, "flash_attention_2", "sdpa", "eager"):
            value = str(cand).strip().lower()
            if value and value not in candidates:
                candidates.append(value)
        last_exc: Exception | None = None
        for cand in candidates:
            try:
                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    str(self.checkpoint_path),
                    torch_dtype=torch.bfloat16,
                    attn_implementation=cand,
                )
                model = model.to(self.device)
                model.eval()
                return model
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                last_exc = exc
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        raise RuntimeError(
            f"Failed to load HF scorer model for {self.checkpoint_path}"
        ) from last_exc

    def build_messages(
        self,
        *,
        image: Image.Image,
        assistant_text: str,
        prompt_variant: str,
        object_field_order: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        system_prompt, user_prompt = get_template_prompts(
            ordering="sorted",
            coord_mode="coord_tokens",
            prompt_variant=prompt_variant,
            object_field_order=object_field_order,
        )
        prompt_messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image", "image": image},
                ],
            },
        ]
        full_messages = prompt_messages + [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            }
        ]
        return prompt_messages, full_messages

    def prepare_example(
        self,
        *,
        image: Image.Image,
        assistant_text: str,
        desc_positions_rel: Sequence[int],
        prompt_variant: str,
        object_field_order: str,
    ) -> PreparedExample:
        prompt_messages, full_messages = self.build_messages(
            image=image,
            assistant_text=assistant_text,
            prompt_variant=prompt_variant,
            object_field_order=object_field_order,
        )
        prompt_text = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = self.processor.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_inputs = self.processor(
            text=[prompt_text],
            images=[image],
            return_tensors="pt",
            padding=False,
        )
        full_inputs = self.processor(
            text=[full_text],
            images=[image],
            return_tensors="pt",
            padding=False,
        )
        prompt_ids = prompt_inputs["input_ids"][0].tolist()
        full_ids = full_inputs["input_ids"][0].tolist()
        assistant_ids = self.tokenizer.encode(assistant_text, add_special_tokens=False)
        start = _find_subsequence(full_ids, assistant_ids, start_hint=len(prompt_ids))
        if start is None:
            raise ValueError("assistant_span_build_failed")
        desc_positions = [int(start + pos) for pos in desc_positions_rel]
        return PreparedExample(
            full_text=str(full_text),
            assistant_text=str(assistant_text),
            desc_positions=desc_positions,
            full_input_ids=[int(v) for v in full_ids],
        )

    def score_prepared_batch(
        self,
        *,
        examples: Sequence[PreparedExample],
        images: Sequence[Image.Image],
    ) -> List[float]:
        if len(examples) != len(images):
            raise ValueError("examples/images length mismatch")
        if not examples:
            return []
        model_inputs = self.processor(
            text=[ex.full_text for ex in examples],
            images=list(images),
            return_tensors="pt",
            padding=True,
        )
        model_inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in model_inputs.items()
        }
        with torch.inference_mode():
            outputs = self.model(**model_inputs, use_cache=False)
        logits = getattr(outputs, "logits", None)
        if not isinstance(logits, torch.Tensor):
            raise RuntimeError("teacher-forced scorer did not return logits")
        input_ids = model_inputs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor):
            raise RuntimeError("teacher-forced scorer missing input_ids")
        if logits.shape[:2] != input_ids.shape[:2]:
            raise RuntimeError("teacher-forced scorer requires unsliced logits")
        scores: List[float] = []
        padded_len = int(input_ids.shape[1])
        for batch_idx, ex in enumerate(examples):
            seq_len = int(len(ex.full_input_ids))
            pad_offset = int(padded_len - seq_len)
            observed_ids = input_ids[batch_idx, pad_offset:].detach().cpu().tolist()
            if [int(v) for v in observed_ids] != [int(v) for v in ex.full_input_ids]:
                raise RuntimeError("assistant_span_build_failed")
            scores.append(
                self._score_positions_for_batch_row(
                    logits=logits,
                    input_ids=input_ids,
                    batch_idx=batch_idx,
                    padded_len=padded_len,
                    full_input_ids=ex.full_input_ids,
                    desc_positions=ex.desc_positions,
                )
            )
        return scores

    def _score_positions_for_batch_row(
        self,
        *,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        batch_idx: int,
        padded_len: int,
        full_input_ids: Sequence[int],
        desc_positions: Sequence[int],
    ) -> float:
        seq_len = int(len(full_input_ids))
        pad_offset = int(padded_len - seq_len)
        observed_ids = input_ids[batch_idx, pad_offset:].detach().cpu().tolist()
        if [int(v) for v in observed_ids] != [int(v) for v in full_input_ids]:
            raise RuntimeError("assistant_span_build_failed")
        values: List[float] = []
        for pos in desc_positions:
            abs_pos = int(pad_offset + pos)
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
            raise ValueError("missing_desc_span")
        return float(mean(values))

    def score_prepared_multi_positions(
        self,
        *,
        prepared: PreparedExample,
        image: Image.Image,
        desc_positions_list: Sequence[Sequence[int]],
    ) -> List[float]:
        model_inputs = self.processor(
            text=[prepared.full_text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )
        model_inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in model_inputs.items()
        }
        with torch.inference_mode():
            outputs = self.model(**model_inputs, use_cache=False)
        logits = getattr(outputs, "logits", None)
        if not isinstance(logits, torch.Tensor):
            raise RuntimeError("teacher-forced scorer did not return logits")
        input_ids = model_inputs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor):
            raise RuntimeError("teacher-forced scorer missing input_ids")
        if logits.shape[:2] != input_ids.shape[:2]:
            raise RuntimeError("teacher-forced scorer requires unsliced logits")
        padded_len = int(input_ids.shape[1])
        return [
            self._score_positions_for_batch_row(
                logits=logits,
                input_ids=input_ids,
                batch_idx=0,
                padded_len=padded_len,
                full_input_ids=prepared.full_input_ids,
                desc_positions=list(desc_positions),
            )
            for desc_positions in desc_positions_list
        ]


def _find_subsequence(
    haystack: Sequence[int],
    needle: Sequence[int],
    *,
    start_hint: int = 0,
) -> Optional[int]:
    if not needle:
        return int(start_hint)
    max_start = int(len(haystack) - len(needle))
    for start in range(max(0, int(start_hint)), max_start + 1):
        if list(haystack[start : start + len(needle)]) == list(needle):
            return int(start)
    return None


def _load_image(
    record: Mapping[str, Any], *, root_image_dir: Path, subset_path: Path
) -> LoadedImage:
    image_rel = str(((record.get("images") or [None])[0] or record.get("image") or ""))
    resolved = resolve_image_path_strict(
        image_rel,
        jsonl_dir=subset_path.parent,
        root_image_dir=root_image_dir,
    )
    if resolved is None:
        raise FileNotFoundError(f"Unable to resolve image path {image_rel!r}")
    image = Image.open(resolved).convert("RGB")
    return LoadedImage(
        image_path=str(image_rel),
        image=image,
        width=int(record.get("width") or image.width),
        height=int(record.get("height") or image.height),
    )


def _mask_image(
    image: Image.Image, bbox_px: Sequence[float], *, fill: int
) -> Image.Image:
    masked = image.copy()
    draw = ImageDraw.Draw(masked)
    x1, y1, x2, y2 = [float(v) for v in bbox_px]
    draw.rectangle([x1, y1, x2, y2], fill=(fill, fill, fill))
    return masked


def score_gt_table(
    *,
    rows: Sequence[Dict[str, Any]],
    scorer: TeacherForcedScorer,
    root_image_dir: Path,
    subset_path: Path,
    prompt_variant: str,
    object_field_order: str,
    batch_size: int,
    mask_fill: int,
) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    prepared_rows: List[Tuple[Dict[str, Any], PreparedExample, Image.Image]] = []
    for row in rows:
        row_out = dict(row)
        row_out.update(
            {
                "scoring_status": "failed",
                "failure_reason": None,
                "commitment": None,
                "masked_commitment": None,
                "counterfactual": None,
                "combined_linear": None,
            }
        )
        bbox_norm = row.get("bbox_norm1000")
        if not isinstance(bbox_norm, list) or len(bbox_norm) != 4:
            row_out["failure_reason"] = "degenerate_bbox"
            scored.append(row_out)
            continue
        obj = _build_bbox_gt_object(
            int(row.get("source_gt_idx", 0)), str(row.get("desc") or ""), bbox_norm
        )
        assistant_text = _build_closed_container_text(
            [obj],
            object_field_order=object_field_order,
        )
        desc_spans = find_desc_value_token_positions_by_span(
            tokenizer=scorer.tokenizer,
            token_ids=scorer.tokenizer.encode(assistant_text, add_special_tokens=False),
        )
        if len(desc_spans) != 1 or not desc_spans[0]:
            row_out["failure_reason"] = "missing_desc_span"
            scored.append(row_out)
            continue
        try:
            loaded = _load_image(
                {
                    "images": [row.get("image_path")],
                    "width": row.get("width"),
                    "height": row.get("height"),
                },
                root_image_dir=root_image_dir,
                subset_path=subset_path,
            )
            prepared = scorer.prepare_example(
                image=loaded.image,
                assistant_text=assistant_text,
                desc_positions_rel=desc_spans[0],
                prompt_variant=prompt_variant,
                object_field_order=object_field_order,
            )
            prepared_rows.append((row_out, prepared, loaded.image))
        except ValueError as exc:
            row_out["failure_reason"] = str(exc)
            scored.append(row_out)
    for start in range(0, len(prepared_rows), max(1, int(batch_size))):
        chunk = prepared_rows[start : start + max(1, int(batch_size))]
        try:
            commitments = scorer.score_prepared_batch(
                examples=[prepared for _row, prepared, _image in chunk],
                images=[image for _row, _prepared, image in chunk],
            )
        except ValueError as exc:
            for row_out, _prepared, _image in chunk:
                row_out["failure_reason"] = str(exc)
                scored.append(row_out)
            continue
        masked_images: List[Image.Image] = []
        for row_out, _prepared, image in chunk:
            bbox_px = row_out.get("bbox_pixel")
            if not isinstance(bbox_px, list) or len(bbox_px) != 4:
                masked_images.append(image)
                continue
            masked_images.append(_mask_image(image, bbox_px, fill=mask_fill))
        try:
            masked_commitments = scorer.score_prepared_batch(
                examples=[prepared for _row, prepared, _image in chunk],
                images=masked_images,
            )
        except ValueError as exc:
            for row_out, _prepared, _image in chunk:
                row_out["failure_reason"] = str(exc)
                scored.append(row_out)
            continue
        for (row_out, _prepared, _image), commitment, masked_commitment in zip(
            chunk, commitments, masked_commitments
        ):
            row_out["scoring_status"] = "ok"
            row_out["commitment"] = float(commitment)
            row_out["masked_commitment"] = float(masked_commitment)
            row_out["counterfactual"] = float(commitment - masked_commitment)
            row_out["combined_linear"] = float(
                row_out["commitment"] + row_out["counterfactual"]
            )
            scored.append(row_out)
    return scored


def score_rollout_proposals(
    *,
    proposal_rows: Sequence[Dict[str, Any]],
    subset_records: Sequence[Mapping[str, Any]],
    scorer: TeacherForcedScorer,
    root_image_dir: Path,
    subset_path: Path,
    prompt_variant: str,
    object_field_order: str,
    mask_fill: int,
    masked_batch_size: int,
    rollout_counterfactual_scope: str,
) -> List[Dict[str, Any]]:
    by_image: Dict[int, List[Dict[str, Any]]] = {}
    for row in proposal_rows:
        by_image.setdefault(int(row["image_idx"]), []).append(dict(row))

    scored_rows: List[Dict[str, Any]] = []
    for image_idx in sorted(by_image):
        rows = by_image[image_idx]
        record = subset_records[int(image_idx)]
        try:
            loaded = _load_image(
                record, root_image_dir=root_image_dir, subset_path=subset_path
            )
        except FileNotFoundError:
            for row in rows:
                row["scoring_status"] = "failed"
                row["failure_reason"] = "assistant_span_build_failed"
                scored_rows.append(row)
            continue

        gt_pred_ordered = sorted(rows, key=lambda row: int(row["proposal_index"]))
        pred_payload_objects: List[Mapping[str, Any]] = []
        for row in gt_pred_ordered:
            pred_payload_objects.append(
                {
                    "desc": row.get("desc"),
                    "bbox": row.get("bbox_pixel"),
                }
            )
        rollout_objects, failure = _extract_rollout_objects_for_scoring(
            pred_payload_objects,
            width=loaded.width,
            height=loaded.height,
        )
        if rollout_objects is None or failure is not None:
            for row in rows:
                row["scoring_status"] = "failed"
                row["failure_reason"] = failure or "sequence_canonicalization_failed"
                scored_rows.append(row)
            continue

        assistant_text = _build_closed_container_text(
            rollout_objects,
            object_field_order=object_field_order,
        )
        assistant_ids = scorer.tokenizer.encode(
            assistant_text, add_special_tokens=False
        )
        desc_spans = find_desc_value_token_positions_by_span(
            tokenizer=scorer.tokenizer,
            token_ids=assistant_ids,
        )
        if len(desc_spans) != len(rows):
            for row in rows:
                row["scoring_status"] = "failed"
                row["failure_reason"] = "sequence_canonicalization_failed"
                scored_rows.append(row)
            continue

        try:
            prepared = scorer.prepare_example(
                image=loaded.image,
                assistant_text=assistant_text,
                desc_positions_rel=[],
                prompt_variant=prompt_variant,
                object_field_order=object_field_order,
            )
        except ValueError as exc:
            for row in rows:
                row["scoring_status"] = "failed"
                row["failure_reason"] = str(exc)
                scored_rows.append(row)
            continue

        assistant_start = _find_subsequence(
            prepared.full_input_ids, assistant_ids, start_hint=0
        )
        if assistant_start is None:
            for row in rows:
                row["scoring_status"] = "failed"
                row["failure_reason"] = "assistant_span_build_failed"
                scored_rows.append(row)
            continue

        try:
            baseline_desc_positions = [
                [int(assistant_start + rel) for rel in desc_span]
                for desc_span in desc_spans
            ]
            baseline_scores = scorer.score_prepared_multi_positions(
                prepared=prepared,
                image=loaded.image,
                desc_positions_list=baseline_desc_positions,
            )
        except ValueError as exc:
            for row in rows:
                row["scoring_status"] = "failed"
                row["failure_reason"] = str(exc)
                scored_rows.append(row)
            continue

        masked_examples: List[PreparedExample] = []
        masked_images: List[Image.Image] = []
        masked_targets: List[Dict[str, Any]] = []
        masked_scores_by_proposal: Dict[int, Optional[float]] = {}
        for row, baseline_score, desc_positions in zip(
            rows, baseline_scores, baseline_desc_positions
        ):
            row["commitment"] = float(baseline_score)
            row["scoring_status"] = "ok"
            scope = str(rollout_counterfactual_scope or "all").strip().lower()
            if scope not in {"all", "unmatched_only"}:
                scope = "all"
            if scope == "unmatched_only" and str(row.get("match_status")) != "unmatched":
                continue
            bbox_px = row.get("bbox_pixel")
            if not isinstance(bbox_px, list):
                row["failure_reason"] = "degenerate_bbox"
                row["scoring_status"] = "failed"
                continue
            masked_targets.append(row)
            masked_examples.append(
                PreparedExample(
                    full_text=prepared.full_text,
                    assistant_text=prepared.assistant_text,
                    desc_positions=list(desc_positions),
                    full_input_ids=list(prepared.full_input_ids),
                )
            )
            masked_images.append(_mask_image(loaded.image, bbox_px, fill=mask_fill))

        masked_commitments: List[Optional[float]] = [None] * len(masked_targets)
        for start in range(0, len(masked_targets), max(1, int(masked_batch_size))):
            end = start + max(1, int(masked_batch_size))
            chunk_examples = masked_examples[start:end]
            chunk_images = masked_images[start:end]
            try:
                chunk_scores = scorer.score_prepared_batch(
                    examples=chunk_examples,
                    images=chunk_images,
                )
            except ValueError as exc:
                for row in masked_targets[start:end]:
                    row["scoring_status"] = "failed"
                    row["failure_reason"] = str(exc)
                continue
            for chunk_idx, score in enumerate(chunk_scores, start=start):
                masked_commitments[chunk_idx] = float(score)
        for row, masked in zip(masked_targets, masked_commitments):
            masked_scores_by_proposal[int(row["proposal_index"])] = (
                float(masked) if masked is not None else None
            )

        for row in rows:
            if row.get("commitment") is None:
                scored_rows.append(row)
                continue
            scope = str(rollout_counterfactual_scope or "all").strip().lower()
            if scope == "unmatched_only" and str(row.get("match_status")) != "unmatched":
                scored_rows.append(row)
                continue
            masked = masked_scores_by_proposal.get(int(row["proposal_index"]))
            if masked is None:
                row["scoring_status"] = "failed"
                row["failure_reason"] = row.get("failure_reason") or "nonfinite_logprob"
                scored_rows.append(row)
                continue
            row["masked_commitment"] = float(masked)
            row["counterfactual"] = float(row["commitment"] - masked)
            row["combined_linear"] = float(row["commitment"] + row["counterfactual"])
            scored_rows.append(row)
    return scored_rows


def _binary_metrics(
    rows: Sequence[Mapping[str, Any]], *, label_key: str, score_key: str
) -> Dict[str, Any]:
    pairs: List[Tuple[float, int]] = []
    excluded: Dict[str, int] = {}
    for row in rows:
        if row.get("scoring_status") != "ok":
            reason = str(row.get("failure_reason") or "unknown")
            excluded[reason] = int(excluded.get(reason, 0) + 1)
            continue
        score_raw = row.get(score_key)
        label_raw = row.get(label_key)
        if score_raw is None or label_raw is None:
            continue
        score = float(score_raw)
        label = int(label_raw)
        if math.isfinite(score) and label in {0, 1}:
            pairs.append((score, label))
    positives = sum(1 for _score, label in pairs if label == 1)
    negatives = sum(1 for _score, label in pairs if label == 0)
    out = {
        "count": int(len(pairs)),
        "positives": int(positives),
        "negatives": int(negatives),
        "excluded_by_reason": excluded,
        "auroc": None,
        "auprc": None,
    }
    if positives <= 0 or negatives <= 0:
        return out
    sorted_pairs = sorted(pairs, key=lambda item: item[0], reverse=True)
    out["auroc"] = _compute_auroc(
        sorted_pairs, positives=positives, negatives=negatives
    )
    out["auprc"] = _compute_auprc(sorted_pairs, positives=positives)
    return out


def _compute_auroc(
    sorted_pairs: Sequence[Tuple[float, int]], *, positives: int, negatives: int
) -> float:
    tp = 0
    fp = 0
    prev_score: Optional[float] = None
    roc_points: List[Tuple[float, float]] = [(0.0, 0.0)]
    for score, label in sorted_pairs:
        if prev_score is not None and score != prev_score:
            roc_points.append(
                (float(fp) / float(negatives), float(tp) / float(positives))
            )
        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score
    roc_points.append((float(fp) / float(negatives), float(tp) / float(positives)))
    area = 0.0
    for (x0, y0), (x1, y1) in zip(roc_points, roc_points[1:]):
        area += (x1 - x0) * (y0 + y1) * 0.5
    return float(area)


def _compute_auprc(
    sorted_pairs: Sequence[Tuple[float, int]], *, positives: int
) -> float:
    tp = 0
    fp = 0
    prev_recall = 0.0
    area = 0.0
    for _score, label in sorted_pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        precision = float(tp) / float(max(1, tp + fp))
        recall = float(tp) / float(positives)
        area += (recall - prev_recall) * precision
        prev_recall = recall
    return float(area)


def _histogram(
    rows: Sequence[Mapping[str, Any]], *, score_key: str, label_key: str, bins: int
) -> Dict[str, Any]:
    values = [
        (float(row[score_key]), int(row[label_key]))
        for row in rows
        if row.get("scoring_status") == "ok"
        and row.get(score_key) is not None
        and row.get(label_key) in {0, 1}
        and math.isfinite(float(row[score_key]))
    ]
    if not values:
        return {"bins": []}
    scores = [score for score, _label in values]
    lo = min(scores)
    hi = max(scores)
    if math.isclose(lo, hi):
        hi = lo + 1e-6
    width = (hi - lo) / float(max(1, bins))
    payload: List[Dict[str, Any]] = []
    for idx in range(max(1, bins)):
        left = lo + width * idx
        right = hi if idx == bins - 1 else (left + width)
        pos = 0
        neg = 0
        for score, label in values:
            in_bin = (
                (left <= score <= right) if idx == bins - 1 else (left <= score < right)
            )
            if not in_bin:
                continue
            if label == 1:
                pos += 1
            else:
                neg += 1
        payload.append(
            {
                "left": float(left),
                "right": float(right),
                "positive_count": int(pos),
                "negative_count": int(neg),
            }
        )
    return {"bins": payload}


def _top_k_unmatched_proxy_stats(
    rows: Sequence[Mapping[str, Any]],
    *,
    score_key: str,
    top_k_values: Sequence[int],
) -> Dict[str, Any]:
    unmatched = [
        row
        for row in rows
        if row.get("scoring_status") == "ok"
        and row.get("match_status") == "unmatched"
        and row.get(score_key) is not None
    ]
    unmatched.sort(key=lambda row: float(row[score_key]), reverse=True)
    payload: Dict[str, Any] = {}
    for k in top_k_values:
        head = unmatched[: int(k)]
        if not head:
            payload[str(int(k))] = {"count": 0}
            continue
        iou03 = sum(1 for row in head if float(row.get("nearest_gt_iou") or 0.0) >= 0.3)
        iou05 = sum(1 for row in head if float(row.get("nearest_gt_iou") or 0.0) >= 0.5)
        payload[str(int(k))] = {
            "count": int(len(head)),
            "nearest_gt_iou_ge_0.3_rate": float(iou03) / float(len(head)),
            "nearest_gt_iou_ge_0.5_rate": float(iou05) / float(len(head)),
        }
    return payload


def summarize_checkpoint(
    *,
    checkpoint_name: str,
    gt_rows: Sequence[Mapping[str, Any]],
    proposal_rows: Sequence[Mapping[str, Any]],
    histogram_bins: int,
    top_k_values: Sequence[int],
) -> Dict[str, Any]:
    gt_summary: Dict[str, Any] = {}
    for score_key in ("commitment", "counterfactual", "combined_linear"):
        gt_summary[score_key] = {
            **_binary_metrics(gt_rows, label_key="label", score_key=score_key),
            "histogram": _histogram(
                gt_rows,
                score_key=score_key,
                label_key="label",
                bins=histogram_bins,
            ),
        }

    rollout_summary: Dict[str, Any] = {}
    for score_key in ("commitment", "counterfactual", "combined_linear"):
        rollout_rows = []
        for row in proposal_rows:
            row2 = dict(row)
            if row.get("match_status") == "matched":
                row2["label"] = 1
            elif row.get("match_status") == "unmatched":
                row2["label"] = 0
            else:
                continue
            rollout_rows.append(row2)
        rollout_summary[score_key] = {
            **_binary_metrics(rollout_rows, label_key="label", score_key=score_key),
            "top_k_unmatched_proxy_stats": _top_k_unmatched_proxy_stats(
                proposal_rows,
                score_key=score_key,
                top_k_values=top_k_values,
            ),
        }

    correlation_rows = [
        row
        for row in proposal_rows
        if row.get("scoring_status") == "ok"
        and row.get("commitment") is not None
        and row.get("counterfactual") is not None
    ]
    commitment_vals = [float(row["commitment"]) for row in correlation_rows]
    counter_vals = [float(row["counterfactual"]) for row in correlation_rows]
    corr = _pearson(commitment_vals, counter_vals) if commitment_vals else None
    return {
        "checkpoint": checkpoint_name,
        "gt_vs_hard_negative": gt_summary,
        "matched_vs_unmatched": rollout_summary,
        "commitment_counterfactual_correlation": corr,
        "calibration_skipped_reason": "v1 study reports raw log-probability proxies only",
    }


def write_unmatched_audit_pack(
    *,
    proposal_rows: Sequence[Mapping[str, Any]],
    subset_records: Sequence[Mapping[str, Any]],
    root_image_dir: Path,
    subset_path: Path,
    out_dir: Path,
    top_n: int,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = [
        row
        for row in proposal_rows
        if row.get("scoring_status") == "ok"
        and row.get("match_status") == "unmatched"
        and row.get("combined_linear") is not None
    ]
    candidates.sort(key=lambda row: float(row["combined_linear"]), reverse=True)
    selected = candidates[: max(0, int(top_n))]
    index_rows: List[Dict[str, Any]] = []

    for rank, row in enumerate(selected):
        image_idx = int(row.get("image_idx") or 0)
        if image_idx < 0 or image_idx >= len(subset_records):
            continue
        try:
            loaded = _load_image(
                subset_records[image_idx],
                root_image_dir=root_image_dir,
                subset_path=subset_path,
            )
        except FileNotFoundError:
            continue
        bbox = row.get("bbox_pixel")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        clean_bbox = _sanitize_bbox_xyxy(bbox, loaded.width, loaded.height)
        if clean_bbox is None:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in clean_bbox]
        pad_x = max(8, int(round(0.1 * float(x2 - x1))))
        pad_y = max(8, int(round(0.1 * float(y2 - y1))))
        crop_box = (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(loaded.width, x2 + pad_x),
            min(loaded.height, y2 + pad_y),
        )
        crop = loaded.image.crop(crop_box)
        overlay = loaded.image.copy()
        draw = ImageDraw.Draw(overlay)
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=4)

        stem = f"{rank:03d}_img{image_idx:04d}_prop{int(row.get('proposal_index') or 0):03d}"
        crop_rel = f"crops/{stem}.png"
        overlay_rel = f"overlays/{stem}.png"
        (out_dir / "crops").mkdir(parents=True, exist_ok=True)
        (out_dir / "overlays").mkdir(parents=True, exist_ok=True)
        crop.save(out_dir / crop_rel)
        overlay.save(out_dir / overlay_rel)
        index_rows.append(
            {
                **dict(row),
                "audit_rank": int(rank),
                "crop_path": crop_rel,
                "overlay_path": overlay_rel,
            }
        )

    _write_jsonl(out_dir / "index.jsonl", index_rows)
    _write_csv(out_dir / "index.csv", index_rows)
    return {
        "count": int(len(index_rows)),
        "index_jsonl": str(out_dir / "index.jsonl"),
        "index_csv": str(out_dir / "index.csv"),
    }


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = float(mean(xs))
    y_mean = float(mean(ys))
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    if den_x <= 0.0 or den_y <= 0.0:
        return None
    return float(num / (den_x * den_y))


def _render_report(
    *,
    config: StudyConfig,
    subset_meta: Mapping[str, Any],
    manifests: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append("# Unmatched Proposal Verifier Study")
    lines.append("")
    lines.append("## Setup")
    lines.append(
        f"- subset: `{subset_meta.get('output_path')}` from `{subset_meta.get('input_path')}`"
    )
    lines.append(f"- sample_count: `{subset_meta.get('num_samples')}`")
    lines.append(f"- seed: `{subset_meta.get('seed')}`")
    lines.append(f"- root_image_dir: `{subset_meta.get('root_image_dir')}`")
    lines.append("")
    lines.append("## Collection")
    lines.append(f"- collection backend mode: `{config.collection.backend_mode}`")
    lines.append(f"- temperature: `{config.collection.temperature}`")
    lines.append(f"- repetition_penalty: `{config.collection.repetition_penalty}`")
    lines.append(f"- infer.generation.batch_size: `{config.collection.batch_size}`")
    if str(config.collection.backend_mode).lower() != "hf":
        lines.append(
            f"- infer.backend.server_options.vllm_gpu_memory_utilization: `{config.collection.gpu_memory_utilization}`"
        )
    lines.append("")
    lines.append("## Checkpoints")
    for manifest in manifests:
        lines.append(
            "- `{name}` -> `{path}` (prompt_variant=`{prompt}`, object_field_order=`{order}`, source=`{source}`)".format(
                name=manifest.get("checkpoint_name"),
                path=manifest.get("checkpoint_path_resolved"),
                prompt=manifest.get("prompt_variant"),
                order=manifest.get("object_field_order"),
                source=manifest.get("prompt_control_source"),
            )
        )
    lines.append("")
    lines.append("## Proxy Definitions")
    lines.append(
        "- commitment: desc-only average teacher-forced log-probability on the original image"
    )
    lines.append(
        "- counterfactual: commitment(original) - commitment(masked bbox image)"
    )
    lines.append("- combined_linear: commitment + counterfactual")
    lines.append("")
    lines.append("## Results")
    for summary in summaries:
        lines.append(f"### {summary.get('checkpoint')}")
        gt_summary = summary.get("gt_vs_hard_negative") or {}
        rollout_summary = summary.get("matched_vs_unmatched") or {}
        for score_key in ("commitment", "counterfactual", "combined_linear"):
            gt_metrics = gt_summary.get(score_key) or {}
            rollout_metrics = rollout_summary.get(score_key) or {}
            lines.append(
                "- `{score}` GT-vs-hard-neg: AUROC=`{auroc}` AUPRC=`{auprc}` counted=`{count}`".format(
                    score=score_key,
                    auroc=_fmt_metric(gt_metrics.get("auroc")),
                    auprc=_fmt_metric(gt_metrics.get("auprc")),
                    count=gt_metrics.get("count"),
                )
            )
            lines.append(
                "- `{score}` matched-vs-unmatched: AUROC=`{auroc}` AUPRC=`{auprc}` counted=`{count}`".format(
                    score=score_key,
                    auroc=_fmt_metric(rollout_metrics.get("auroc")),
                    auprc=_fmt_metric(rollout_metrics.get("auprc")),
                    count=rollout_metrics.get("count"),
                )
            )
        lines.append(
            "- commitment/counterfactual correlation: `{}`".format(
                _fmt_metric(summary.get("commitment_counterfactual_correlation"))
            )
        )
        lines.append(
            "- calibration: skipped (`{}`)".format(
                summary.get("calibration_skipped_reason")
            )
        )
        audit_pack = summary.get("audit_pack") or {}
        lines.append(
            "- audit pack: count=`{count}` index=`{path}`".format(
                count=audit_pack.get("count"),
                path=audit_pack.get("index_jsonl"),
            )
        )
        lines.append("")

    strongest_single = _best_single_proxy(summaries)
    strongest_any = _best_proxy(summaries)
    combined_helped = _combined_helped(summaries)
    stable = _signal_stable(summaries, strongest_single)
    ready = _pseudo_label_ready(summaries)
    lines.append("## Recommendation")
    lines.append(f"- strongest single proxy: `{strongest_single}`")
    lines.append(
        f"- does commitment + counterfactual materially outperform either single proxy? `{'yes' if combined_helped else 'no'}`"
    )
    lines.append(
        f"- is the signal stable across checkpoints? `{'yes' if stable else 'mixed'}`"
    )
    lines.append(
        f"- is the proxy good enough for soft pseudo-label promotion? `{'yes' if ready else 'not yet'}`"
    )
    lines.append("- main observed failure modes:")
    lines.append(
        f"  scoring drift / exclusions: `{json.dumps(_merge_excluded_reasons(summaries), ensure_ascii=False)}`"
    )
    lines.append(
        "- commitment can remain high on visually plausible wrong-location boxes; counterfactual is the intended corrective signal."
    )
    lines.append(
        f"- best overall proxy on the current summary tables: `{strongest_any}`."
    )
    return "\n".join(lines).rstrip() + "\n"


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(numeric):
        return "NA"
    return f"{numeric:.4f}"


def _metric_means(
    summaries: Sequence[Mapping[str, Any]],
    *,
    group_key: str = "gt_vs_hard_negative",
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for proxy in ("commitment", "counterfactual", "combined_linear"):
        values: List[float] = []
        for summary in summaries:
            metric = (
                ((summary.get(group_key) or {}).get(proxy) or {}).get("auprc")
            )
            if metric is None:
                continue
            values.append(float(metric))
        if values:
            out[proxy] = float(mean(values))
    return out


def _best_single_proxy(summaries: Sequence[Mapping[str, Any]]) -> str:
    means = _metric_means(summaries)
    if not means:
        return "NA"
    single = {k: v for k, v in means.items() if k in {"commitment", "counterfactual"}}
    if not single:
        return "NA"
    return max(single.items(), key=lambda item: item[1])[0]


def _best_proxy(summaries: Sequence[Mapping[str, Any]]) -> str:
    means = _metric_means(summaries)
    if not means:
        return "NA"
    return max(means.items(), key=lambda item: item[1])[0]


def _combined_helped(summaries: Sequence[Mapping[str, Any]]) -> bool:
    means = _metric_means(summaries)
    best_single = _best_single_proxy(summaries)
    if best_single == "NA":
        return False
    return float(means.get("combined_linear", float("-inf"))) >= float(
        means.get(best_single, float("-inf"))
    ) + 0.01


def _signal_stable(summaries: Sequence[Mapping[str, Any]], proxy: str) -> bool:
    values: List[float] = []
    for summary in summaries:
        metric = (
            ((summary.get("gt_vs_hard_negative") or {}).get(proxy) or {}).get("auroc")
        )
        if metric is None:
            continue
        values.append(float(metric))
    if len(values) < 2:
        return False
    return max(values) - min(values) <= 0.05


def _pseudo_label_ready(summaries: Sequence[Mapping[str, Any]]) -> bool:
    best = _best_proxy(summaries)
    if best == "NA":
        return False
    gt_vals: List[float] = []
    rollout_vals: List[float] = []
    for summary in summaries:
        gt_metric = (
            ((summary.get("gt_vs_hard_negative") or {}).get(best) or {}).get("auroc")
        )
        rollout_metric = (
            ((summary.get("matched_vs_unmatched") or {}).get(best) or {}).get("auroc")
        )
        if gt_metric is not None:
            gt_vals.append(float(gt_metric))
        if rollout_metric is not None:
            rollout_vals.append(float(rollout_metric))
    if not gt_vals or not rollout_vals:
        return False
    return float(mean(gt_vals)) >= 0.8 and float(mean(rollout_vals)) >= 0.6


def _merge_excluded_reasons(summaries: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    merged: Dict[str, int] = {}
    for summary in summaries:
        for group_key in ("gt_vs_hard_negative", "matched_vs_unmatched"):
            group = summary.get(group_key) or {}
            for proxy in ("commitment", "counterfactual", "combined_linear"):
                excluded = ((group.get(proxy) or {}).get("excluded_by_reason") or {})
                for reason, count in excluded.items():
                    merged[str(reason)] = int(merged.get(str(reason), 0) + int(count))
    return merged


def run_study(config: StudyConfig) -> Dict[str, Any]:
    run_dir = (REPO_ROOT / config.run.output_dir / config.run.name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    subset_path, subset_records, subset_meta = materialize_subset(config, run_dir)
    root_image_dir = Path(str(subset_meta["root_image_dir"])).resolve()

    gt_positives, gt_negatives = build_gt_and_negative_tables(subset_records)
    gt_rows = [*gt_positives, *gt_negatives]
    _write_jsonl(run_dir / "gt" / "gt_positives.jsonl", gt_positives)
    _write_csv(run_dir / "gt" / "gt_positives.csv", gt_positives)
    _write_jsonl(run_dir / "gt" / "gt_hard_negatives.jsonl", gt_negatives)
    _write_csv(run_dir / "gt" / "gt_hard_negatives.csv", gt_negatives)

    manifests: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    for checkpoint in config.checkpoints:
        ckpt_slug = _slugify(checkpoint.name)
        checkpoint_run_dir = run_dir / "checkpoints" / ckpt_slug
        manifest = run_collection_for_checkpoint(
            config,
            subset_path=subset_path,
            root_image_dir=root_image_dir,
            checkpoint=checkpoint,
            run_dir=checkpoint_run_dir,
        )
        manifests.append(manifest)
        proposal_rows = build_rollout_proposal_table(
            checkpoint_name=checkpoint.name,
            run_dir=checkpoint_run_dir,
        )
        _write_jsonl(checkpoint_run_dir / "proposal_table.jsonl", proposal_rows)
        _write_csv(checkpoint_run_dir / "proposal_table.csv", proposal_rows)

        scorer = TeacherForcedScorer(
            checkpoint_path=Path(str(manifest["checkpoint_path_resolved"])),
            device=config.scoring.device,
            attn_implementation=config.scoring.attn_implementation,
        )
        gt_scored = score_gt_table(
            rows=gt_rows,
            scorer=scorer,
            root_image_dir=root_image_dir,
            subset_path=subset_path,
            prompt_variant=str(manifest["prompt_variant"]),
            object_field_order=str(manifest["object_field_order"]),
            batch_size=config.scoring.gt_batch_size,
            mask_fill=config.scoring.mask_fill,
        )
        _write_jsonl(checkpoint_run_dir / "gt_proxy_scores.jsonl", gt_scored)
        _write_csv(checkpoint_run_dir / "gt_proxy_scores.csv", gt_scored)

        proposal_scored = score_rollout_proposals(
            proposal_rows=proposal_rows,
            subset_records=subset_records,
            scorer=scorer,
            root_image_dir=root_image_dir,
            subset_path=subset_path,
            prompt_variant=str(manifest["prompt_variant"]),
            object_field_order=str(manifest["object_field_order"]),
            mask_fill=config.scoring.mask_fill,
            masked_batch_size=config.scoring.masked_batch_size,
            rollout_counterfactual_scope=config.scoring.rollout_counterfactual_scope,
        )
        _write_jsonl(
            checkpoint_run_dir / "proposal_proxy_scores.jsonl", proposal_scored
        )
        _write_csv(checkpoint_run_dir / "proposal_proxy_scores.csv", proposal_scored)
        audit_pack = write_unmatched_audit_pack(
            proposal_rows=proposal_scored,
            subset_records=subset_records,
            root_image_dir=root_image_dir,
            subset_path=subset_path,
            out_dir=checkpoint_run_dir / "audit_pack",
            top_n=config.report.audit_pack_top_n,
        )

        summary = summarize_checkpoint(
            checkpoint_name=checkpoint.name,
            gt_rows=gt_scored,
            proposal_rows=proposal_scored,
            histogram_bins=config.report.histogram_bins,
            top_k_values=config.report.top_k_values,
        )
        summary["audit_pack"] = audit_pack
        summaries.append(summary)
        _write_json(checkpoint_run_dir / "summary.json", summary)
        del scorer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report_text = _render_report(
        config=config,
        subset_meta=subset_meta,
        manifests=manifests,
        summaries=summaries,
    )
    report_path = run_dir / "report.md"
    report_path.write_text(report_text, encoding="utf-8")

    study_manifest = {
        "run_dir": str(run_dir),
        "subset": subset_meta,
        "checkpoints": manifests,
        "report_path": str(report_path),
    }
    _write_json(run_dir / "study_manifest.json", study_manifest)
    return study_manifest


def main(config_path: str) -> Dict[str, Any]:
    config = load_study_config(Path(config_path))
    return run_study(config)


def run_unmatched_proposal_verifier_study(
    *,
    config_path: Path,
    smoke: bool = False,
    limit_checkpoints: Optional[int] = None,
) -> Dict[str, Any]:
    config = load_study_config(Path(config_path))
    if smoke:
        config = replace(
            config,
            dataset=replace(
                config.dataset,
                sample_count=min(int(config.dataset.sample_count), 8),
            ),
        )
        if limit_checkpoints is None:
            limit_checkpoints = 1
    if limit_checkpoints is not None:
        config = replace(
            config,
            checkpoints=tuple(config.checkpoints[: max(0, int(limit_checkpoints))]),
        )
    return run_study(config)
