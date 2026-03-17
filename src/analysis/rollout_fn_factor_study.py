from __future__ import annotations

import csv
import hashlib
import json
import multiprocessing as mp
import os
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import yaml
from PIL import Image

from src.analysis.unmatched_proposal_verifier import (
    _pixel_to_norm1000,
    build_rollout_proposal_table,
    resolve_checkpoint_path,
    resolve_prompt_controls_for_checkpoint,
)
from src.common.geometry.coord_utils import bbox_from_points
from src.common.object_field_order import build_object_payload, normalize_object_field_order
from src.config.prompts import get_template_prompts, resolve_dense_prompt_variant_key
from src.eval.detection import EvalOptions, evaluate_and_save
from src.eval.oracle_k import _build_match_index
from src.infer.engine import GenerationConfig, InferenceConfig, InferenceEngine
from src.trainers.rollout_matching.parsing import _is_append_ready_prefix
from src.utils.coordjson_transpiler import CoordJSONValidationError, parse_coordjson
from src.utils.assistant_json import dumps_coordjson
from src.vis import materialize_gt_vs_pred_vis_resource, render_gt_vs_pred_review


REPO_ROOT = Path(__file__).resolve().parents[2]
_PRIMARY_IOU_THR = 0.5
_SECONDARY_IOU_THR = 0.3
_IM_END_TOKEN = "<|im_end|>"
_DEFAULT_STAGES = (
    "bootstrap",
    "subset",
    "baseline",
    "sampling",
    "prefix",
    "stress",
    "length",
    "report",
)
_VALID_PREFIX_MODES = {
    "image_only",
    "oracle_gt_prefix_train_order",
    "oracle_gt_prefix_random_order",
    "self_prefix",
    "switched_prefix",
    "broken_prefix",
}
_VALID_MUTATIONS = {"delete", "adjacent_swap", "insert"}
_HEALTH_INVALID_REASON_PRECEDENCE = (
    "parse_invalid",
    "invalid_rollout",
    "truncation_anomaly",
    "too_few_nonempty",
    "too_few_predictions",
    "too_duplicate_like",
)


@dataclass(frozen=True)
class CheckpointSpec:
    alias: str
    path: str
    artifact_kind: str = "executable_checkpoint"
    prompt_variant: Optional[str] = None
    object_field_order: Optional[str] = None
    provenance_sidecars: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SplitSpec:
    name: str
    jsonl_path: str


@dataclass(frozen=True)
class RunConfig:
    name: str
    output_dir: str
    stages: Tuple[str, ...] = _DEFAULT_STAGES


@dataclass(frozen=True)
class PromptConfig:
    prompt_variant: str = "coco_80"
    object_field_order: str = "desc_first"
    do_resize: bool = False


@dataclass(frozen=True)
class BootstrapConfig:
    candidate_pool_strategy: str = "top_gt_count"
    candidate_pool_limit: int = 512
    hard32_size: int = 32
    hard16_size: int = 16


@dataclass(frozen=True)
class DecodeConfig:
    temperature: float
    top_p: float
    max_new_tokens: int
    repetition_penalty: float
    batch_size: int


@dataclass(frozen=True)
class SamplingConfig:
    k: int = 8
    temperature: float = 0.6
    top_p: float = 0.95
    max_new_tokens: int = 3084
    repetition_penalty: float = 1.0
    batch_size: int = 1
    seed: int = 42


@dataclass(frozen=True)
class PrefixConfig:
    lengths: Tuple[int, ...] = (1, 2, 4, 8)
    random_seed: int = 17


@dataclass(frozen=True)
class StressConfig:
    prefix_length: int = 4
    mutations: Tuple[str, ...] = ("delete", "adjacent_swap", "insert")


@dataclass(frozen=True)
class LengthConfig:
    extended_max_new_tokens: int = 4096


@dataclass(frozen=True)
class EvalStudyConfig:
    metrics: str = "f1ish"
    f1ish_iou_thrs: Tuple[float, ...] = (_SECONDARY_IOU_THR, _PRIMARY_IOU_THR)
    f1ish_pred_scope: str = "all"
    semantic_model: str = "model_cache/all-MiniLM-L6-v2-local"
    semantic_threshold: float = 0.5
    semantic_device: str = "cuda:0"
    semantic_batch_size: int = 64
    num_workers: int = 8
    overlay_top_n_per_bucket: int = 8


@dataclass(frozen=True)
class HealthGateConfig:
    min_parse_valid_rate: float = 1.0
    min_nonempty_rate: float = 1.0
    min_pred_count_total: int = 1
    max_duplicate_like_rate: float = 0.35


@dataclass(frozen=True)
class ExecutionConfig:
    gpu_ids: Tuple[int, ...] = (0, 1, 2, 3)
    reuse_existing: bool = True
    start_method: str = "spawn"


@dataclass(frozen=True)
class ReportConfig:
    review_top_n_per_bucket: int = 8
    bucket_order: Tuple[str, ...] = (
        "deterministic_hit",
        "decode_selection_miss",
        "prefix_sensitive_miss",
        "length_bias_miss",
        "persistent_unrecovered",
    )


@dataclass(frozen=True)
class StudyConfig:
    run: RunConfig
    prompts: PromptConfig
    bootstrap: BootstrapConfig
    baseline_decode: DecodeConfig
    sampling: SamplingConfig
    prefix: PrefixConfig
    stress: StressConfig
    length: LengthConfig
    eval: EvalStudyConfig
    health_gate: HealthGateConfig
    execution: ExecutionConfig
    report: ReportConfig
    checkpoints: Tuple[CheckpointSpec, ...]
    splits: Tuple[SplitSpec, ...]


@dataclass(frozen=True)
class ResolvedCheckpoint:
    alias: str
    path: Path
    resolve_source: str
    artifact_kind: str
    fingerprint: str
    prompt_variant: str
    object_field_order: str
    prompt_control_source: str
    provenance_sidecars: Dict[str, str]


@dataclass(frozen=True)
class ResolvedSplit:
    name: str
    jsonl_path: Path
    image_root: Path


@dataclass(frozen=True)
class ImageRecord:
    record_idx: int
    source_image_id: int
    image_rel: str
    file_name: str
    width: int
    height: int
    objects: List[Dict[str, Any]]
    raw_record: Dict[str, Any]


@dataclass(frozen=True)
class PrefixPlan:
    prefix_mode: str
    requested_prefix_length: int
    actual_prefix_length: int
    prefix_text: str
    prefix_pred_objects: List[Dict[str, Any]]
    prefix_source_checkpoint: Optional[str]
    prefix_ordering_rule: str
    prefix_hash: str
    prefix_seed: Optional[int] = None
    prefix_permutation: Optional[List[int]] = None
    mutation_type: Optional[str] = None
    continuation_pred_start_index: int = 0


@dataclass(frozen=True)
class LogicalCell:
    stage: str
    dataset_split: str
    subset_name: str
    checkpoint_alias: str
    factor_family: str
    logical_cell_id: str
    decode_family: str
    prefix_mode: str
    prefix_length: int
    max_new_tokens: int
    temperature: float
    top_p: float
    repetition_penalty: float
    sample_k: int
    sample_seed_base: int
    prefix_source_checkpoint: Optional[str] = None
    mutation_type: Optional[str] = None


@dataclass(frozen=True)
class RolloutGeneration:
    raw_text: str
    generated_token_ids: List[int]
    generated_token_text: List[str]
    generated_token_count: int
    prompt_token_count: int
    eos_reached: bool
    finish_reason: str


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"study config must be a mapping, got {type(raw).__name__}")
    return raw


def _ensure_tuple_str(value: Any, *, default: Sequence[str]) -> Tuple[str, ...]:
    if value is None:
        return tuple(str(v) for v in default)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("expected sequence of strings")
    return tuple(str(v).strip() for v in value if str(v).strip())


def _ensure_tuple_int(value: Any, *, default: Sequence[int]) -> Tuple[int, ...]:
    if value is None:
        return tuple(int(v) for v in default)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("expected sequence of ints")
    return tuple(int(v) for v in value)


def _ensure_tuple_float(value: Any, *, default: Sequence[float]) -> Tuple[float, ...]:
    if value is None:
        return tuple(float(v) for v in default)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("expected sequence of floats")
    return tuple(float(v) for v in value)


def _sha256_json(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _slugify(value: str) -> str:
    out = []
    prev_dash = False
    for ch in str(value).strip().lower():
        if ch.isalnum():
            out.append(ch)
            prev_dash = False
            continue
        if not prev_dash:
            out.append("-")
            prev_dash = True
    slug = "".join(out).strip("-")
    return slug or "item"


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                yield payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(str(key))
                seen.add(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _stage_enabled(config: StudyConfig, stage: str) -> bool:
    return str(stage).strip().lower() in {str(s).strip().lower() for s in config.run.stages}


def _resolve_image_root(jsonl_path: Path) -> Path:
    return jsonl_path.parent.resolve()


def _checkpoint_fingerprint(path: Path) -> str:
    candidates = [
        path / "config.json",
        path / "generation_config.json",
        path / "model.safetensors.index.json",
        path / "tokenizer_config.json",
        path / "chat_template.jinja",
    ]
    payload: Dict[str, Any] = {"path": str(path)}
    for candidate in candidates:
        if candidate.is_file():
            stat = candidate.stat()
            payload[candidate.name] = {
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
            try:
                payload[f"{candidate.name}:sha256"] = hashlib.sha256(
                    candidate.read_bytes()
                ).hexdigest()
            except OSError:
                payload[f"{candidate.name}:sha256"] = "read_error"
    return _sha256_json(payload)


def _prompt_hash(prompt_variant: str, object_field_order: str) -> str:
    system_prompt, user_prompt = get_template_prompts(
        ordering="sorted",
        coord_mode="coord_tokens",
        prompt_variant=prompt_variant,
        object_field_order=object_field_order,
    )
    return _sha256_json(
        {
            "prompt_variant": prompt_variant,
            "object_field_order": object_field_order,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "do_resize": False,
        }
    )


def load_study_config(path: Path) -> StudyConfig:
    raw = _load_yaml(path)
    run_raw = raw.get("run") or {}
    prompts_raw = raw.get("prompts") or {}
    bootstrap_raw = raw.get("bootstrap") or {}
    baseline_raw = raw.get("baseline_decode") or {}
    sampling_raw = raw.get("sampling") or {}
    prefix_raw = raw.get("prefix") or {}
    stress_raw = raw.get("stress") or {}
    length_raw = raw.get("length") or {}
    eval_raw = raw.get("eval") or {}
    health_raw = raw.get("health_gate") or {}
    execution_raw = raw.get("execution") or {}
    report_raw = raw.get("report") or {}
    checkpoints_raw = raw.get("checkpoints") or []
    splits_raw = raw.get("splits") or {}

    if not checkpoints_raw:
        raise ValueError("study config must define checkpoints")
    if not isinstance(checkpoints_raw, Sequence):
        raise ValueError("checkpoints must be a sequence")
    checkpoints: List[CheckpointSpec] = []
    for idx, item in enumerate(checkpoints_raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"checkpoint[{idx}] must be a mapping")
        alias = str(item.get("alias") or item.get("name") or "").strip()
        path_raw = str(item.get("path") or "").strip()
        if not alias or not path_raw:
            raise ValueError(f"checkpoint[{idx}] requires alias and path")
        checkpoints.append(
            CheckpointSpec(
                alias=alias,
                path=path_raw,
                artifact_kind=str(item.get("artifact_kind") or "executable_checkpoint").strip(),
                prompt_variant=(
                    str(item["prompt_variant"]).strip()
                    if item.get("prompt_variant") is not None
                    else None
                ),
                object_field_order=(
                    normalize_object_field_order(
                        str(item["object_field_order"]),
                        path=f"checkpoints[{idx}].object_field_order",
                    )
                    if item.get("object_field_order") is not None
                    else None
                ),
                provenance_sidecars={
                    str(k): str(v)
                    for k, v in dict(item.get("provenance_sidecars") or {}).items()
                    if str(k).strip() and str(v).strip()
                },
            )
        )

    if not isinstance(splits_raw, Mapping) or not splits_raw:
        raise ValueError("study config must define split mappings")
    splits: List[SplitSpec] = []
    for name, split_raw in splits_raw.items():
        if not isinstance(split_raw, Mapping):
            raise ValueError(f"split {name!r} must be a mapping")
        jsonl_path = str(split_raw.get("jsonl_path") or "").strip()
        if not jsonl_path:
            raise ValueError(f"split {name!r} missing jsonl_path")
        splits.append(SplitSpec(name=str(name), jsonl_path=jsonl_path))

    run = RunConfig(
        name=str(run_raw.get("name") or "rollout-fn-factor-study").strip(),
        output_dir=str(run_raw.get("output_dir") or "output/analysis").strip(),
        stages=_ensure_tuple_str(run_raw.get("stages"), default=_DEFAULT_STAGES),
    )
    prompts = PromptConfig(
        prompt_variant=resolve_dense_prompt_variant_key(
            str(prompts_raw.get("prompt_variant") or "coco_80")
        ),
        object_field_order=normalize_object_field_order(
            str(prompts_raw.get("object_field_order") or "desc_first"),
            path="prompts.object_field_order",
        ),
        do_resize=bool(prompts_raw.get("do_resize", False)),
    )
    if prompts.do_resize:
        raise ValueError("rollout FN-factor study requires do_resize=false parity")
    bootstrap = BootstrapConfig(
        candidate_pool_strategy=str(
            bootstrap_raw.get("candidate_pool_strategy") or "top_gt_count"
        ).strip(),
        candidate_pool_limit=int(bootstrap_raw.get("candidate_pool_limit", 512)),
        hard32_size=int(bootstrap_raw.get("hard32_size", 32)),
        hard16_size=int(bootstrap_raw.get("hard16_size", 16)),
    )
    baseline_decode = DecodeConfig(
        temperature=float(baseline_raw.get("temperature", 0.0)),
        top_p=float(baseline_raw.get("top_p", 1.0)),
        max_new_tokens=int(baseline_raw.get("max_new_tokens", 3084)),
        repetition_penalty=float(baseline_raw.get("repetition_penalty", 1.05)),
        batch_size=int(baseline_raw.get("batch_size", 4)),
    )
    sampling = SamplingConfig(
        k=int(sampling_raw.get("k", 8)),
        temperature=float(sampling_raw.get("temperature", 0.6)),
        top_p=float(sampling_raw.get("top_p", 0.95)),
        max_new_tokens=int(sampling_raw.get("max_new_tokens", 3084)),
        repetition_penalty=float(sampling_raw.get("repetition_penalty", 1.0)),
        batch_size=int(sampling_raw.get("batch_size", 1)),
        seed=int(sampling_raw.get("seed", 42)),
    )
    prefix = PrefixConfig(
        lengths=_ensure_tuple_int(prefix_raw.get("lengths"), default=(1, 2, 4, 8)),
        random_seed=int(prefix_raw.get("random_seed", 17)),
    )
    stress = StressConfig(
        prefix_length=int(stress_raw.get("prefix_length", 4)),
        mutations=_ensure_tuple_str(
            stress_raw.get("mutations"),
            default=("delete", "adjacent_swap", "insert"),
        ),
    )
    for mutation in stress.mutations:
        if mutation not in _VALID_MUTATIONS:
            raise ValueError(f"unsupported broken-prefix mutation: {mutation}")
    length = LengthConfig(
        extended_max_new_tokens=int(length_raw.get("extended_max_new_tokens", 4096))
    )
    eval_cfg = EvalStudyConfig(
        metrics=str(eval_raw.get("metrics") or "f1ish").strip(),
        f1ish_iou_thrs=_ensure_tuple_float(
            eval_raw.get("f1ish_iou_thrs"), default=(_SECONDARY_IOU_THR, _PRIMARY_IOU_THR)
        ),
        f1ish_pred_scope=str(eval_raw.get("f1ish_pred_scope") or "all").strip(),
        semantic_model=str(
            eval_raw.get("semantic_model") or "model_cache/all-MiniLM-L6-v2-local"
        ).strip(),
        semantic_threshold=float(eval_raw.get("semantic_threshold", 0.5)),
        semantic_device=str(eval_raw.get("semantic_device") or "cuda:0").strip(),
        semantic_batch_size=int(eval_raw.get("semantic_batch_size", 64)),
        num_workers=int(eval_raw.get("num_workers", 8)),
        overlay_top_n_per_bucket=int(eval_raw.get("overlay_top_n_per_bucket", 8)),
    )
    health_gate = HealthGateConfig(
        min_parse_valid_rate=float(health_raw.get("min_parse_valid_rate", 1.0)),
        min_nonempty_rate=float(health_raw.get("min_nonempty_rate", 1.0)),
        min_pred_count_total=int(health_raw.get("min_pred_count_total", 1)),
        max_duplicate_like_rate=float(health_raw.get("max_duplicate_like_rate", 0.35)),
    )
    execution = ExecutionConfig(
        gpu_ids=_ensure_tuple_int(execution_raw.get("gpu_ids"), default=(0, 1, 2, 3)),
        reuse_existing=bool(execution_raw.get("reuse_existing", True)),
        start_method=str(execution_raw.get("start_method") or "spawn").strip(),
    )
    report = ReportConfig(
        review_top_n_per_bucket=int(report_raw.get("review_top_n_per_bucket", 8))
    )
    if bootstrap.hard16_size > bootstrap.hard32_size:
        raise ValueError("Hard-16 must be a subset of Hard-32")
    return StudyConfig(
        run=run,
        prompts=prompts,
        bootstrap=bootstrap,
        baseline_decode=baseline_decode,
        sampling=sampling,
        prefix=prefix,
        stress=stress,
        length=length,
        eval=eval_cfg,
        health_gate=health_gate,
        execution=execution,
        report=report,
        checkpoints=tuple(checkpoints),
        splits=tuple(splits),
    )


def _resolve_study_inputs(config: StudyConfig) -> Tuple[Dict[str, ResolvedCheckpoint], Dict[str, ResolvedSplit]]:
    resolved_checkpoints: Dict[str, ResolvedCheckpoint] = {}
    for ckpt in config.checkpoints:
        if ckpt.artifact_kind == "reference_only":
            raise ValueError(
                f"checkpoint alias {ckpt.alias!r} is marked reference_only and cannot be executed"
            )
        checkpoint_path, resolve_source = resolve_checkpoint_path(ckpt.path)
        prompt_variant, object_field_order, prompt_source = resolve_prompt_controls_for_checkpoint(
            checkpoint_path,
            default_prompt_variant=config.prompts.prompt_variant,
            default_object_field_order=config.prompts.object_field_order,
            override_prompt_variant=ckpt.prompt_variant,
            override_object_field_order=ckpt.object_field_order,
        )
        resolved_checkpoints[ckpt.alias] = ResolvedCheckpoint(
            alias=ckpt.alias,
            path=checkpoint_path.resolve(),
            resolve_source=resolve_source,
            artifact_kind=ckpt.artifact_kind,
            fingerprint=_checkpoint_fingerprint(checkpoint_path.resolve()),
            prompt_variant=prompt_variant,
            object_field_order=object_field_order,
            prompt_control_source=prompt_source,
            provenance_sidecars={k: str((REPO_ROOT / v).resolve()) for k, v in ckpt.provenance_sidecars.items()},
        )
    resolved_splits: Dict[str, ResolvedSplit] = {}
    for split in config.splits:
        jsonl_path = (REPO_ROOT / split.jsonl_path).resolve()
        if not jsonl_path.is_file():
            raise FileNotFoundError(f"split {split.name!r} jsonl not found: {jsonl_path}")
        resolved_splits[split.name] = ResolvedSplit(
            name=split.name,
            jsonl_path=jsonl_path,
            image_root=_resolve_image_root(jsonl_path),
        )
    return resolved_checkpoints, resolved_splits


def _read_split_records(split: ResolvedSplit) -> List[ImageRecord]:
    records: List[ImageRecord] = []
    for idx, raw in enumerate(_iter_jsonl(split.jsonl_path)):
        image_list = list(raw.get("images") or [])
        if len(image_list) != 1:
            continue
        image_rel = str(image_list[0])
        width = int(raw.get("width") or 0)
        height = int(raw.get("height") or 0)
        if width <= 0 or height <= 0:
            continue
        objects = list(raw.get("objects") or [])
        source_image_id = int(raw.get("image_id") or idx)
        file_name = str(raw.get("file_name") or image_rel)
        records.append(
            ImageRecord(
                record_idx=idx,
                source_image_id=source_image_id,
                image_rel=image_rel,
                file_name=file_name,
                width=width,
                height=height,
                objects=objects,
                raw_record=dict(raw),
            )
        )
    return records


def _select_candidate_pool(records: Sequence[ImageRecord], config: BootstrapConfig) -> List[ImageRecord]:
    if config.candidate_pool_strategy != "top_gt_count":
        raise ValueError(
            f"unsupported candidate_pool_strategy={config.candidate_pool_strategy!r}"
        )
    ordered = sorted(
        records,
        key=lambda row: (-len(row.objects), row.source_image_id, row.record_idx),
    )
    return list(ordered[: int(config.candidate_pool_limit)])


def _subset_row(record: ImageRecord, *, subset_index: int) -> Dict[str, Any]:
    return {
        "index": int(subset_index),
        "images": [record.image_rel],
        "width": int(record.width),
        "height": int(record.height),
        "image_id": int(record.source_image_id),
        "file_name": record.file_name,
        "objects": list(record.objects),
        "metadata": dict(record.raw_record.get("metadata") or {}),
    }


def _compact_gt_objects(record: ImageRecord) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for obj in record.objects:
        if not isinstance(obj, Mapping):
            continue
        bbox_raw = obj.get("bbox_2d")
        if not isinstance(bbox_raw, list):
            continue
        try:
            bins = [int(str(v).replace("<|coord_", "").replace("|>", "")) for v in bbox_raw]
        except ValueError:
            continue
        if len(bins) != 4:
            continue
        width = int(record.width)
        height = int(record.height)
        x1 = (bins[0] / 1000.0) * width
        y1 = (bins[1] / 1000.0) * height
        x2 = (bins[2] / 1000.0) * width
        y2 = (bins[3] / 1000.0) * height
        out.append(
            {
                "type": "bbox_2d",
                "points": [x1, y1, x2, y1, x2, y2, x1, y2],
                "desc": str(obj.get("desc") or "").strip(),
            }
        )
    return out


def _canonicalize_eval_objects(objects: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    canonical: List[Dict[str, Any]] = []
    for index, obj in enumerate(objects):
        if not isinstance(obj, Mapping):
            continue
        desc = str(obj.get("desc") or "").strip()
        bbox_xyxy: Optional[List[int]] = None
        bbox_raw = obj.get("bbox_2d")
        if isinstance(bbox_raw, list) and len(bbox_raw) == 4:
            try:
                bbox_xyxy = [int(round(float(v))) for v in bbox_raw]
            except (TypeError, ValueError):
                bbox_xyxy = None
        if bbox_xyxy is None:
            points = obj.get("points")
            if isinstance(points, list) and len(points) >= 4:
                try:
                    x1, y1, x2, y2 = bbox_from_points(points)
                    bbox_xyxy = [
                        int(round(float(x1))),
                        int(round(float(y1))),
                        int(round(float(x2))),
                        int(round(float(y2))),
                    ]
                except (TypeError, ValueError):
                    bbox_xyxy = None
        if bbox_xyxy is None:
            continue
        canonical.append(
            {
                "index": int(index),
                "desc": desc,
                "bbox_2d": bbox_xyxy,
                "coord_mode": "pixel",
            }
        )
    return canonical


def _serialize_objects_to_prefix_text(
    objects: Sequence[Mapping[str, Any]],
    *,
    width: int,
    height: int,
    object_field_order: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    prefix_text = dumps_coordjson({"objects": []})
    if not prefix_text.endswith("]}"):
        raise ValueError("unexpected canonical CoordJSON rendering")
    prefix_text = prefix_text[:-2]
    compact_objects: List[Dict[str, Any]] = []
    for obj in objects:
        desc = str(obj.get("desc") or "").strip()
        if not desc:
            continue
        geometry_key: Optional[str] = None
        geometry_value: Optional[List[str]] = None
        compact_obj: Optional[Dict[str, Any]] = None
        bbox_raw = obj.get("bbox_2d")
        if isinstance(bbox_raw, list):
            try:
                parsed_bbox = [
                    int(str(v).replace("<|coord_", "").replace("|>", ""))
                    for v in bbox_raw
                ]
            except ValueError:
                parsed_bbox = []
            coord_mode = str(obj.get("coord_mode") or "").strip().lower()
            bins: Optional[List[int]] = None
            if len(parsed_bbox) == 4:
                if coord_mode == "pixel" or any(
                    int(v) < 0 or int(v) > 999 for v in parsed_bbox
                ):
                    bins = _pixel_to_norm1000(parsed_bbox, width, height)
                else:
                    bins = [int(v) for v in parsed_bbox]
            if bins is not None and len(bins) == 4:
                geometry_key = "bbox_2d"
                geometry_value = [f"<|coord_{int(v)}|>" for v in bins]
                x1 = (bins[0] / 1000.0) * width
                y1 = (bins[1] / 1000.0) * height
                x2 = (bins[2] / 1000.0) * width
                y2 = (bins[3] / 1000.0) * height
                compact_obj = {
                    "type": "bbox_2d",
                    "points": [x1, y1, x2, y1, x2, y2, x1, y2],
                    "desc": desc,
                }
        if geometry_key is None:
            obj_type = str(obj.get("type") or "").strip()
            points = obj.get("points")
            if obj_type == "bbox_2d" and isinstance(points, list) and len(points) >= 4:
                x1, y1, x2, y2 = bbox_from_points(points)
                bins = _pixel_to_norm1000([x1, y1, x2, y2], width, height)
                if bins is not None:
                    geometry_key = "bbox_2d"
                    geometry_value = [f"<|coord_{int(v)}|>" for v in bins]
                    compact_obj = {
                        "type": "bbox_2d",
                        "points": [x1, y1, x2, y1, x2, y2, x1, y2],
                        "desc": desc,
                    }
            elif obj_type == "poly" and isinstance(points, list) and len(points) >= 6:
                bins: List[str] = []
                for idx, point in enumerate(points):
                    denom = width if idx % 2 == 0 else height
                    val = int(max(0, min(999, round((float(point) / float(denom)) * 1000.0))))
                    bins.append(f"<|coord_{val}|>")
                geometry_key = "poly"
                geometry_value = bins
                compact_obj = {"type": "poly", "points": list(points), "desc": desc}
        if geometry_key is None or geometry_value is None or compact_obj is None:
            continue
        payload = build_object_payload(
            desc=desc,
            geometry_key=geometry_key,
            geometry_value=geometry_value,
            object_field_order=object_field_order,
        )
        entry_text = dumps_coordjson(payload)
        if not prefix_text.endswith("["):
            prefix_text = prefix_text + ", "
        prefix_text = prefix_text + entry_text
        compact_objects.append(compact_obj)
    if compact_objects and not prefix_text.endswith(", "):
        prefix_text = prefix_text + ", "
    if not _is_append_ready_prefix(prefix_text):
        raise ValueError("prefix text must remain append-ready after serialization")
    return prefix_text, compact_objects


def _prefix_hash(prefix_text: str) -> str:
    return hashlib.sha256(prefix_text.encode("utf-8")).hexdigest()


def _sample_seed(base_seed: int, *, sample_idx: int) -> int:
    return int(base_seed + sample_idx)


def _close_prefix_rollout_text(
    prefix_text: str,
    continuation_text: str,
    *,
    object_field_order: str,
) -> str:
    normalized_continuation = str(continuation_text)
    if str(prefix_text).rstrip().endswith(","):
        stripped_continuation = normalized_continuation.lstrip()
        if stripped_continuation.startswith(","):
            normalized_continuation = stripped_continuation[1:].lstrip()
    full_text = str(prefix_text) + normalized_continuation
    stripped = full_text.rstrip()
    trailing_ws = full_text[len(stripped) :]
    trailer = ""
    if stripped.endswith(_IM_END_TOKEN):
        stripped = stripped[: -len(_IM_END_TOKEN)].rstrip()
        trailer = _IM_END_TOKEN
    try:
        parse_coordjson(
            stripped,
            mode="strict",
            object_field_order=object_field_order,
        )
    except CoordJSONValidationError:
        pass
    else:
        return stripped + trailer + trailing_ws
    candidate = stripped
    repaired_trailing_ws = trailing_ws
    if candidate.endswith(","):
        candidate = candidate[:-1].rstrip()
        repaired_trailing_ws = ""
    if _is_append_ready_prefix(candidate):
        return candidate + "]}" + trailer + repaired_trailing_ws
    return full_text


class HFStudyRunner:
    def __init__(
        self,
        *,
        checkpoint: ResolvedCheckpoint,
        device: str,
        image_root: Path,
    ) -> None:
        cfg = InferenceConfig(
            gt_jsonl="unused.jsonl",
            model_checkpoint=str(checkpoint.path),
            mode="coord",
            prompt_variant=checkpoint.prompt_variant,
            object_field_order=checkpoint.object_field_order,
            out_path="unused.jsonl",
            device=device,
            backend_type="hf",
            root_image_dir=str(image_root),
        )
        gen_cfg = GenerationConfig(
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=128,
            repetition_penalty=1.0,
            batch_size=1,
            seed=0,
        )
        self.engine = InferenceEngine(cfg, gen_cfg)
        self.engine.load_model()
        self.tokenizer = self.engine.processor.tokenizer if self.engine.processor else None
        if self.tokenizer is None:
            raise RuntimeError("HFStudyRunner requires tokenizer")

    def _generate_sequences(
        self,
        *,
        prompt_texts: Sequence[str],
        images: Sequence[Image.Image],
        gen_cfg: GenerationConfig,
    ) -> List[RolloutGeneration]:
        assert self.engine.processor is not None and self.engine.model is not None
        model_inputs = self.engine.processor(
            text=list(prompt_texts),
            images=list(images),
            return_tensors="pt",
            padding=True,
        )
        model_inputs = {k: v.to(self.engine.cfg.device) for k, v in model_inputs.items()}
        self.engine.gen_cfg = gen_cfg
        self.engine._seed()
        gen_kwargs = dict(
            max_new_tokens=gen_cfg.max_new_tokens,
            do_sample=gen_cfg.temperature > 0,
            temperature=max(1e-4, gen_cfg.temperature),
            top_p=gen_cfg.top_p,
            use_cache=True,
        )
        if gen_cfg.repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = gen_cfg.repetition_penalty
        with torch.inference_mode():
            gen_outputs = self.engine.model.generate(
                **model_inputs,
                **gen_kwargs,
                return_dict_in_generate=True,
                output_scores=False,
            )
        sequences = gen_outputs.sequences
        if "attention_mask" in model_inputs and torch.is_tensor(model_inputs["attention_mask"]):
            prompt_lengths = (
                model_inputs["attention_mask"].sum(dim=1).detach().cpu().tolist()
            )
        else:
            prompt_lengths = [int(model_inputs["input_ids"].shape[1])] * int(sequences.shape[0])
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        out: List[RolloutGeneration] = []
        for row_idx in range(int(sequences.shape[0])):
            prompt_len = int(prompt_lengths[row_idx])
            generated_ids = sequences[row_idx, prompt_len:].detach().cpu().tolist()
            generated_token_text = (
                self.tokenizer.batch_decode(
                    [[int(tok)] for tok in generated_ids],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                if generated_ids
                else []
            )
            finish_reason = "unknown"
            eos_reached = False
            if generated_ids and eos_token_id is not None and int(generated_ids[-1]) == int(eos_token_id):
                eos_reached = True
                finish_reason = "eos"
            elif len(generated_ids) >= int(gen_cfg.max_new_tokens):
                finish_reason = "length"
            raw_text = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            out.append(
                RolloutGeneration(
                    raw_text=raw_text,
                    generated_token_ids=[int(v) for v in generated_ids],
                    generated_token_text=list(generated_token_text),
                    generated_token_count=int(len(generated_ids)),
                    prompt_token_count=prompt_len,
                    eos_reached=bool(eos_reached),
                    finish_reason=finish_reason,
                )
            )
        return out

    def generate_image_only_batch(
        self,
        *,
        images: Sequence[Image.Image],
        gen_cfg: GenerationConfig,
    ) -> List[RolloutGeneration]:
        assert self.engine.processor is not None
        messages = [self.engine._build_messages(image) for image in images]
        prompt_texts = [
            self.engine.processor.apply_chat_template(
                message, add_generation_prompt=True, tokenize=False
            )
            for message in messages
        ]
        return self._generate_sequences(
            prompt_texts=prompt_texts,
            images=images,
            gen_cfg=gen_cfg,
        )

    def generate_with_prefix(
        self,
        *,
        image: Image.Image,
        prefix_text: str,
        gen_cfg: GenerationConfig,
    ) -> RolloutGeneration:
        return self.generate_with_prefix_batch(
            images=[image],
            prefix_texts=[prefix_text],
            gen_cfg=gen_cfg,
        )[0]

    def generate_with_prefix_batch(
        self,
        *,
        images: Sequence[Image.Image],
        prefix_texts: Sequence[str],
        gen_cfg: GenerationConfig,
    ) -> List[RolloutGeneration]:
        assert self.engine.processor is not None
        if len(images) != len(prefix_texts):
            raise ValueError(
                f"images/prefix_texts length mismatch: {len(images)} vs {len(prefix_texts)}"
            )
        prefill_texts: List[str] = []
        for image, prefix_text in zip(images, prefix_texts):
            prompt_messages = self.engine._build_messages(image)
            prefill_messages = prompt_messages + [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": str(prefix_text)}],
                }
            ]
            prefill_texts.append(
                self.engine.processor.apply_chat_template(
                    prefill_messages,
                    add_generation_prompt=False,
                    continue_final_message=True,
                    tokenize=False,
                )
            )
        return self._generate_sequences(
            prompt_texts=prefill_texts,
            images=images,
            gen_cfg=gen_cfg,
        )

    def parse_prediction(
        self,
        *,
        raw_text: str,
        width: int,
        height: int,
    ) -> Tuple[List[Dict[str, Any]], List[str], Any, List[str], bool]:
        pred_errors: List[str] = []
        pred = self.engine._process_pred(
            raw_text,
            width=width,
            height=height,
            errors=pred_errors,
        )
        pred = self.engine._compact_objects(pred)
        raw_special_tokens = []
        raw_output_json = None
        raw_ends_with_im_end = raw_text.endswith("<|im_end|>")
        try:
            from src.common.formatting import extract_special_tokens, load_prediction_dict

            raw_special_tokens = extract_special_tokens(raw_text, preserve_duplicates=True)
            raw_output_json = load_prediction_dict(raw_text)
        except Exception:
            raw_special_tokens = []
            raw_output_json = None
        return pred, pred_errors, raw_output_json, raw_special_tokens, raw_ends_with_im_end


def _record_to_gt_pred_row(
    *,
    subset_index: int,
    record: ImageRecord,
    split_jsonl_path: Path,
    gt_objects: Sequence[Mapping[str, Any]],
    pred_objects: Sequence[Mapping[str, Any]],
    raw_output_json: Any,
    raw_special_tokens: Sequence[str],
    raw_ends_with_im_end: bool,
    error_codes: Sequence[str],
    finish_reason: str,
    eos_reached: bool,
    generated_token_count: int,
    prompt_token_count: int,
    logical_cell: LogicalCell,
    execution_shard_id: str,
    sample_idx: int,
    prefix_plan: Optional[PrefixPlan],
    prefix_only_salvaged: bool = False,
) -> Dict[str, Any]:
    return {
        "index": int(subset_index),
        "record_idx": int(subset_index),
        "source_image_id": int(record.source_image_id),
        "image": record.image_rel,
        "images": [record.image_rel],
        "file_name": record.file_name,
        "width": int(record.width),
        "height": int(record.height),
        "mode": "coord",
        "coord_mode": "pixel",
        "provenance": {"source_jsonl_dir": str(split_jsonl_path.parent.resolve())},
        "gt": _canonicalize_eval_objects(gt_objects),
        "pred": _canonicalize_eval_objects(pred_objects),
        "raw_output_json": raw_output_json,
        "raw_special_tokens": list(raw_special_tokens),
        "raw_ends_with_im_end": bool(raw_ends_with_im_end),
        "errors": list(error_codes),
        "error_entries": [{"code": str(code)} for code in error_codes],
        "stage": logical_cell.stage,
        "logical_cell_id": logical_cell.logical_cell_id,
        "execution_shard_id": execution_shard_id,
        "dataset_split": logical_cell.dataset_split,
        "subset_name": logical_cell.subset_name,
        "checkpoint_alias": logical_cell.checkpoint_alias,
        "factor_family": logical_cell.factor_family,
        "prefix_mode": logical_cell.prefix_mode,
        "prefix_length_requested": int(logical_cell.prefix_length),
        "sample_idx": int(sample_idx),
        "generated_token_count": int(generated_token_count),
        "prompt_token_count": int(prompt_token_count),
        "finish_reason": finish_reason,
        "eos_reached": bool(eos_reached),
        "max_new_tokens": int(logical_cell.max_new_tokens),
        "temperature": float(logical_cell.temperature),
        "top_p": float(logical_cell.top_p),
        "repetition_penalty": float(logical_cell.repetition_penalty),
        "sample_seed_base": int(logical_cell.sample_seed_base),
        "prefix_pred_count": int(prefix_plan.actual_prefix_length if prefix_plan else 0),
        "continuation_pred_start_index": int(
            prefix_plan.continuation_pred_start_index if prefix_plan else 0
        ),
        "prefix_source_checkpoint": (
            prefix_plan.prefix_source_checkpoint if prefix_plan is not None else None
        ),
        "prefix_ordering_rule": (
            prefix_plan.prefix_ordering_rule if prefix_plan is not None else "image_only"
        ),
        "prefix_hash": prefix_plan.prefix_hash if prefix_plan is not None else None,
        "prefix_seed": prefix_plan.prefix_seed if prefix_plan is not None else None,
        "prefix_permutation": (
            list(prefix_plan.prefix_permutation) if prefix_plan and prefix_plan.prefix_permutation else None
        ),
        "mutation_type": prefix_plan.mutation_type if prefix_plan is not None else None,
        "prefix_only_salvaged": bool(prefix_only_salvaged),
    }


def _candidate_subset_dir(run_dir: Path, split_name: str) -> Path:
    return run_dir / "bootstrap" / split_name


def _hard_subset_dir(run_dir: Path, split_name: str, subset_name: str) -> Path:
    return run_dir / "subsets" / split_name / subset_name


def _cells_root(run_dir: Path, stage: str) -> Path:
    return run_dir / "cells" / stage


def _cell_dir(run_dir: Path, cell: LogicalCell) -> Path:
    return _cells_root(run_dir, cell.stage) / cell.logical_cell_id


def _sample_dir(run_dir: Path, cell: LogicalCell, sample_idx: int) -> Path:
    return _cell_dir(run_dir, cell) / "samples" / f"sample_{int(sample_idx):03d}"


def _shard_dir(run_dir: Path, cell: LogicalCell, execution_shard_id: str, sample_idx: int) -> Path:
    return (
        _cell_dir(run_dir, cell)
        / "shards"
        / execution_shard_id
        / f"sample_{int(sample_idx):03d}"
    )


def _build_eval_options(config: StudyConfig, *, output_dir: Path) -> EvalOptions:
    return EvalOptions(
        metrics=config.eval.metrics,
        strict_parse=False,
        use_segm=False,
        iou_types=("bbox",),
        f1ish_iou_thrs=list(config.eval.f1ish_iou_thrs),
        f1ish_pred_scope=config.eval.f1ish_pred_scope,
        output_dir=output_dir,
        overlay=False,
        num_workers=config.eval.num_workers,
        semantic_model=config.eval.semantic_model,
        semantic_threshold=config.eval.semantic_threshold,
        semantic_device=config.eval.semantic_device,
        semantic_batch_size=config.eval.semantic_batch_size,
    )


def _thr_key(thr: float) -> str:
    return f"{float(thr):.2f}"


def _load_match_indexes_for_run(
    eval_dir: Path,
    *,
    thresholds: Sequence[float],
) -> Dict[str, Dict[str, Dict[Tuple[int, int], Dict[str, Any]]]]:
    out: Dict[str, Dict[str, Dict[Tuple[int, int], Dict[str, Any]]]] = {}
    primary_key = _thr_key(_PRIMARY_IOU_THR)
    for thr in thresholds:
        key = _thr_key(thr)
        if key == primary_key:
            matches_path = eval_dir / "matches.jsonl"
        else:
            matches_path = eval_dir / f"matches@{key}.jsonl"
        rows = list(_iter_jsonl(matches_path)) if matches_path.is_file() else []
        loc_index, full_index = _build_match_index(rows)
        out[key] = {"loc": loc_index, "full": full_index}
    return out


def _load_gt_rows(gt_vs_pred_path: Path) -> List[Dict[str, Any]]:
    return list(_iter_jsonl(gt_vs_pred_path))


def _validate_cell_alignment(sample_paths: Sequence[Path]) -> None:
    if not sample_paths:
        return
    baseline = _load_gt_rows(sample_paths[0])
    for other_path in sample_paths[1:]:
        rows = _load_gt_rows(other_path)
        if len(rows) != len(baseline):
            raise ValueError(
                f"alignment failure: {other_path} has {len(rows)} rows, expected {len(baseline)}"
            )
        for left, right in zip(baseline, rows):
            if int(left.get("index", -1)) != int(right.get("index", -2)):
                raise ValueError(f"alignment failure: row index mismatch in {other_path}")
            if str(left.get("file_name") or "") != str(right.get("file_name") or ""):
                raise ValueError(f"alignment failure: file_name mismatch in {other_path}")
            left_gt = list(left.get("gt") or [])
            right_gt = list(right.get("gt") or [])
            if left_gt != right_gt:
                raise ValueError(f"alignment failure: gt mismatch in {other_path}")


def _build_rollout_health(
    *,
    gt_vs_pred_path: Path,
    proposal_rows: Sequence[Mapping[str, Any]],
    gate: HealthGateConfig,
) -> Dict[str, Any]:
    gt_rows = list(_iter_jsonl(gt_vs_pred_path))
    num_images = int(len(gt_rows))
    pred_counts = [int(len(row.get("pred") or [])) for row in gt_rows]
    invalid_rollout_count = 0
    parse_invalid_count = 0
    truncation_anomaly_count = 0
    parser_failure_counts: Dict[str, int] = {}
    for row in gt_rows:
        errors = [str(error).strip() for error in (row.get("errors") or []) if str(error).strip()]
        pred_count = int(len(row.get("pred") or []))
        if errors:
            parse_invalid_count += 1
            invalid_rollout_count += 1
        elif row.get("raw_output_json") is None and pred_count <= 0:
            invalid_rollout_count += 1
        if str(row.get("finish_reason") or "") == "length" and errors:
            truncation_anomaly_count += 1
        for error in errors:
            parser_failure_counts[error] = int(parser_failure_counts.get(error, 0) + 1)
    parse_valid_rate = (
        float(num_images - parse_invalid_count) / float(num_images) if num_images else 0.0
    )
    nonempty_pred_rate = (
        float(sum(1 for count in pred_counts if count > 0)) / float(num_images)
        if num_images
        else 0.0
    )
    pred_count_total = int(sum(pred_counts))
    duplicate_like_rate = (
        float(
            sum(int(row.get("duplicate_like_any_desc_iou90") or 0) for row in proposal_rows)
        )
        / float(len(proposal_rows))
        if proposal_rows
        else 0.0
    )
    invalid_reason = None
    if parse_valid_rate < float(gate.min_parse_valid_rate):
        invalid_reason = "parse_invalid"
    elif invalid_rollout_count > 0:
        invalid_reason = "invalid_rollout"
    elif truncation_anomaly_count > 0:
        invalid_reason = "truncation_anomaly"
    elif nonempty_pred_rate < float(gate.min_nonempty_rate):
        invalid_reason = "too_few_nonempty"
    elif pred_count_total < int(gate.min_pred_count_total):
        invalid_reason = "too_few_predictions"
    elif duplicate_like_rate > float(gate.max_duplicate_like_rate):
        invalid_reason = "too_duplicate_like"
    return {
        "num_images": num_images,
        "parse_valid_rate": float(parse_valid_rate),
        "invalid_rollout_count": int(invalid_rollout_count),
        "nonempty_pred_rate": float(nonempty_pred_rate),
        "pred_count_total": int(pred_count_total),
        "duplicate_like_rate": float(duplicate_like_rate),
        "truncation_anomaly_count": int(truncation_anomaly_count),
        "parser_failure_counts": parser_failure_counts,
        "rollout_health_valid": invalid_reason is None,
        "rollout_health_invalid_reason": invalid_reason,
        "gate_thresholds": asdict(gate),
    }


def _aggregate_sample_health(sample_artifacts: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    health_rows = [dict(sample.get("rollout_health") or {}) for sample in sample_artifacts]
    if not health_rows:
        return {
            "num_samples": 0,
            "rollout_health_valid": False,
            "rollout_health_invalid_reason": "invalid_rollout",
        }
    invalid_reasons = [
        str(row.get("rollout_health_invalid_reason"))
        for row in health_rows
        if row.get("rollout_health_invalid_reason")
    ]
    invalid_reason = None
    if invalid_reasons:
        order = {reason: idx for idx, reason in enumerate(_HEALTH_INVALID_REASON_PRECEDENCE)}
        invalid_reason = min(invalid_reasons, key=lambda reason: order.get(reason, len(order)))
    return {
        "num_samples": int(len(health_rows)),
        "num_images": int(sum(int(row.get("num_images", 0)) for row in health_rows)),
        "parse_valid_rate_mean": float(mean(float(row.get("parse_valid_rate", 0.0)) for row in health_rows)),
        "invalid_rollout_count_total": int(
            sum(int(row.get("invalid_rollout_count", 0)) for row in health_rows)
        ),
        "nonempty_pred_rate_mean": float(
            mean(float(row.get("nonempty_pred_rate", 0.0)) for row in health_rows)
        ),
        "pred_count_total": int(sum(int(row.get("pred_count_total", 0)) for row in health_rows)),
        "duplicate_like_rate_mean": float(
            mean(float(row.get("duplicate_like_rate", 0.0)) for row in health_rows)
        ),
        "truncation_anomaly_count_total": int(
            sum(int(row.get("truncation_anomaly_count", 0)) for row in health_rows)
        ),
        "rollout_health_valid": invalid_reason is None,
        "rollout_health_invalid_reason": invalid_reason,
    }


def _continuation_only_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        prefix_pred_count = int(row.get("prefix_pred_count") or 0)
        preds = list(row.get("pred") or [])
        row_out = dict(row)
        continuation_preds: List[Dict[str, Any]] = []
        for new_index, pred in enumerate(preds[prefix_pred_count:]):
            pred_out = dict(pred)
            pred_out["index"] = int(new_index)
            continuation_preds.append(pred_out)
        row_out["pred"] = continuation_preds
        row_out.pop("matching", None)
        row_out.pop("match", None)
        row_out["prefix_pred_count"] = 0
        row_out["continuation_pred_start_index"] = 0
        out.append(row_out)
    return out


def _sample_seed_schedule(cell: LogicalCell) -> List[int]:
    return [_sample_seed(cell.sample_seed_base, sample_idx=idx) for idx in range(cell.sample_k)]


def _logical_cell_manifest(cell: LogicalCell) -> Dict[str, Any]:
    return {
        "stage": cell.stage,
        "dataset_split": cell.dataset_split,
        "subset_name": cell.subset_name,
        "checkpoint_alias": cell.checkpoint_alias,
        "factor_family": cell.factor_family,
        "logical_cell_id": cell.logical_cell_id,
        "decode_family": cell.decode_family,
        "prefix_mode": cell.prefix_mode,
        "prefix_length": cell.prefix_length,
        "temperature": cell.temperature,
        "top_p": cell.top_p,
        "repetition_penalty": cell.repetition_penalty,
        "max_new_tokens": cell.max_new_tokens,
        "sample_k": cell.sample_k,
        "sample_seed_base": cell.sample_seed_base,
        "sample_seed_schedule": _sample_seed_schedule(cell),
        "prefix_source_checkpoint": cell.prefix_source_checkpoint,
        "mutation_type": cell.mutation_type,
    }


def _logical_cell_from_payload(payload: Mapping[str, Any]) -> LogicalCell:
    allowed = {item.name for item in fields(LogicalCell)}
    cell_kwargs = {key: value for key, value in payload.items() if key in allowed}
    return LogicalCell(**cell_kwargs)


def _gen_cfg_for_cell(cell: LogicalCell, *, sample_idx: int, batch_size: int) -> GenerationConfig:
    return GenerationConfig(
        temperature=float(cell.temperature),
        top_p=float(cell.top_p),
        max_new_tokens=int(cell.max_new_tokens),
        repetition_penalty=float(cell.repetition_penalty),
        batch_size=int(batch_size),
        seed=_sample_seed(cell.sample_seed_base, sample_idx=sample_idx),
    )


def _chunked(seq: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(seq), max(1, int(size))):
        yield seq[start : start + max(1, int(size))]


def _load_image(runner: HFStudyRunner, *, split_jsonl_path: Path, record: ImageRecord) -> Image.Image:
    path = runner.engine._resolve_image_path(split_jsonl_path, record.image_rel)
    return Image.open(path).convert("RGB")


def _run_worker_task(
    *,
    runner: HFStudyRunner,
    split_jsonl_path: Path,
    cell: LogicalCell,
    execution_shard_id: str,
    shard_records: Sequence[Tuple[int, ImageRecord]],
    shard_dir_base: Path,
    prefix_plans: Mapping[int, PrefixPlan],
    baseline_batch_size: int,
) -> None:
    shard_dir_base.mkdir(parents=True, exist_ok=True)
    _write_json(shard_dir_base / "shard_manifest.json", {"execution_shard_id": execution_shard_id})
    gt_cache: Dict[int, List[Dict[str, Any]]] = {
        subset_index: _compact_gt_objects(record) for subset_index, record in shard_records
    }

    for sample_idx in range(cell.sample_k):
        out_rows: List[Dict[str, Any]] = []
        out_dir = shard_dir_base / f"sample_{sample_idx:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        gen_cfg = _gen_cfg_for_cell(
            cell,
            sample_idx=sample_idx,
            batch_size=baseline_batch_size,
        )
        if cell.prefix_mode == "image_only":
            for batch in _chunked(list(shard_records), size=max(1, int(gen_cfg.batch_size))):
                images = [
                    _load_image(runner, split_jsonl_path=split_jsonl_path, record=record)
                    for _, record in batch
                ]
                gens = runner.generate_image_only_batch(images=images, gen_cfg=gen_cfg)
                for (subset_index, record), gen in zip(batch, gens):
                    pred, errors, raw_output_json, raw_special_tokens, raw_ends = runner.parse_prediction(
                        raw_text=gen.raw_text,
                        width=record.width,
                        height=record.height,
                    )
                    out_rows.append(
                        _record_to_gt_pred_row(
                            subset_index=subset_index,
                            record=record,
                            split_jsonl_path=split_jsonl_path,
                            gt_objects=gt_cache[subset_index],
                            pred_objects=pred,
                            raw_output_json=raw_output_json,
                            raw_special_tokens=raw_special_tokens,
                            raw_ends_with_im_end=raw_ends,
                            error_codes=errors,
                            finish_reason=gen.finish_reason,
                            eos_reached=gen.eos_reached,
                            generated_token_count=gen.generated_token_count,
                            prompt_token_count=gen.prompt_token_count,
                            logical_cell=cell,
                            execution_shard_id=execution_shard_id,
                            sample_idx=sample_idx,
                            prefix_plan=None,
                            prefix_only_salvaged=False,
                        )
                    )
        else:
            for batch in _chunked(list(shard_records), size=max(1, int(gen_cfg.batch_size))):
                batch_plans: List[PrefixPlan] = []
                batch_images: List[Image.Image] = []
                for subset_index, record in batch:
                    plan = prefix_plans.get(subset_index)
                    if plan is None:
                        raise KeyError(
                            f"missing prefix plan for cell={cell.logical_cell_id} record={subset_index}"
                        )
                    batch_plans.append(plan)
                    batch_images.append(
                        _load_image(runner, split_jsonl_path=split_jsonl_path, record=record)
                    )
                gens = runner.generate_with_prefix_batch(
                    images=batch_images,
                    prefix_texts=[plan.prefix_text for plan in batch_plans],
                    gen_cfg=gen_cfg,
                )
                for (subset_index, record), plan, gen in zip(batch, batch_plans, gens):
                    full_text = _close_prefix_rollout_text(
                        str(plan.prefix_text),
                        str(gen.raw_text),
                        object_field_order=runner.engine.object_field_order,
                    )
                    pred, errors, raw_output_json, raw_special_tokens, raw_ends = (
                        runner.parse_prediction(
                            raw_text=full_text,
                            width=record.width,
                            height=record.height,
                        )
                    )
                    prefix_only_salvaged = False
                    if errors and int(plan.actual_prefix_length) > 0:
                        prefix_only_text = _close_prefix_rollout_text(
                            str(plan.prefix_text),
                            "",
                            object_field_order=runner.engine.object_field_order,
                        )
                        (
                            prefix_only_pred,
                            prefix_only_errors,
                            prefix_only_raw_output_json,
                            prefix_only_raw_special_tokens,
                            prefix_only_raw_ends,
                        ) = runner.parse_prediction(
                            raw_text=prefix_only_text,
                            width=record.width,
                            height=record.height,
                        )
                        if prefix_only_pred and not prefix_only_errors:
                            pred = prefix_only_pred
                            errors = []
                            raw_output_json = prefix_only_raw_output_json
                            raw_special_tokens = prefix_only_raw_special_tokens
                            raw_ends = prefix_only_raw_ends
                            prefix_only_salvaged = True
                    out_rows.append(
                        _record_to_gt_pred_row(
                            subset_index=subset_index,
                            record=record,
                            split_jsonl_path=split_jsonl_path,
                            gt_objects=gt_cache[subset_index],
                            pred_objects=pred,
                            raw_output_json=raw_output_json,
                            raw_special_tokens=raw_special_tokens,
                            raw_ends_with_im_end=raw_ends,
                            error_codes=errors,
                            finish_reason=gen.finish_reason,
                            eos_reached=gen.eos_reached,
                            generated_token_count=gen.generated_token_count,
                            prompt_token_count=gen.prompt_token_count,
                            logical_cell=cell,
                            execution_shard_id=execution_shard_id,
                            sample_idx=sample_idx,
                            prefix_plan=plan,
                            prefix_only_salvaged=prefix_only_salvaged,
                        )
                    )
        out_rows.sort(key=lambda row: int(row["index"]))
        _write_jsonl(out_dir / "gt_vs_pred.jsonl", out_rows)


def _worker_main(
    *,
    gpu_id: int,
    checkpoint: ResolvedCheckpoint,
    split_image_root: Path,
    split_jsonl_path: Path,
    tasks: Sequence[Dict[str, Any]],
    baseline_batch_size: int,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    runner = HFStudyRunner(
        checkpoint=checkpoint,
        device="cuda:0",
        image_root=split_image_root,
    )
    for task in tasks:
        cell = _logical_cell_from_payload(task["cell"])
        shard_records = [
            (
                int(item["subset_index"]),
                ImageRecord(
                    record_idx=int(item["record_idx"]),
                    source_image_id=int(item["source_image_id"]),
                    image_rel=str(item["image_rel"]),
                    file_name=str(item["file_name"]),
                    width=int(item["width"]),
                    height=int(item["height"]),
                    objects=list(item["objects"]),
                    raw_record=dict(item["raw_record"]),
                ),
            )
            for item in task["records"]
        ]
        prefix_plans = {
            int(subset_index): PrefixPlan(**payload)
            for subset_index, payload in dict(task.get("prefix_plans") or {}).items()
        }
        _run_worker_task(
            runner=runner,
            split_jsonl_path=split_jsonl_path,
            cell=cell,
            execution_shard_id=str(task["execution_shard_id"]),
            shard_records=shard_records,
            shard_dir_base=Path(str(task["shard_dir"])),
            prefix_plans=prefix_plans,
            baseline_batch_size=baseline_batch_size,
        )


def _make_logical_cell_id(
    *,
    stage: str,
    split_name: str,
    subset_name: str,
    checkpoint_alias: str,
    factor_family: str,
    prefix_mode: str,
    prefix_length: int,
    decode_family: str,
    max_new_tokens: int,
    sample_k: int,
    mutation_type: Optional[str] = None,
    prefix_source_checkpoint: Optional[str] = None,
) -> str:
    parts = [
        stage,
        split_name,
        subset_name,
        checkpoint_alias,
        factor_family,
        prefix_mode,
        f"n{int(prefix_length)}",
        decode_family,
        f"len{int(max_new_tokens)}",
        f"k{int(sample_k)}",
    ]
    if prefix_source_checkpoint:
        parts.append(f"src-{prefix_source_checkpoint}")
    if mutation_type:
        parts.append(f"mut-{mutation_type}")
    return _slugify("__".join(parts))


def _build_baseline_cells(config: StudyConfig) -> List[LogicalCell]:
    cells: List[LogicalCell] = []
    for split in config.splits:
        for checkpoint in config.checkpoints:
            cells.append(
                LogicalCell(
                    stage="baseline",
                    dataset_split=split.name,
                    subset_name="Hard-32",
                    checkpoint_alias=checkpoint.alias,
                    factor_family="deterministic_baseline",
                    logical_cell_id=_make_logical_cell_id(
                        stage="baseline",
                        split_name=split.name,
                        subset_name="Hard-32",
                        checkpoint_alias=checkpoint.alias,
                        factor_family="deterministic_baseline",
                        prefix_mode="image_only",
                        prefix_length=0,
                        decode_family="greedy",
                        max_new_tokens=config.baseline_decode.max_new_tokens,
                        sample_k=1,
                    ),
                    decode_family="greedy",
                    prefix_mode="image_only",
                    prefix_length=0,
                    max_new_tokens=config.baseline_decode.max_new_tokens,
                    temperature=config.baseline_decode.temperature,
                    top_p=config.baseline_decode.top_p,
                    repetition_penalty=config.baseline_decode.repetition_penalty,
                    sample_k=1,
                    sample_seed_base=0,
                )
            )
    return cells


def _build_bootstrap_cells(config: StudyConfig) -> List[LogicalCell]:
    cells: List[LogicalCell] = []
    for split in config.splits:
        for checkpoint in config.checkpoints:
            cells.append(
                LogicalCell(
                    stage="bootstrap",
                    dataset_split=split.name,
                    subset_name="candidate_pool",
                    checkpoint_alias=checkpoint.alias,
                    factor_family="bootstrap_selector",
                    logical_cell_id=_make_logical_cell_id(
                        stage="bootstrap",
                        split_name=split.name,
                        subset_name="candidate_pool",
                        checkpoint_alias=checkpoint.alias,
                        factor_family="bootstrap_selector",
                        prefix_mode="image_only",
                        prefix_length=0,
                        decode_family="greedy",
                        max_new_tokens=config.baseline_decode.max_new_tokens,
                        sample_k=1,
                    ),
                    decode_family="greedy",
                    prefix_mode="image_only",
                    prefix_length=0,
                    max_new_tokens=config.baseline_decode.max_new_tokens,
                    temperature=config.baseline_decode.temperature,
                    top_p=config.baseline_decode.top_p,
                    repetition_penalty=config.baseline_decode.repetition_penalty,
                    sample_k=1,
                    sample_seed_base=0,
                )
            )
    return cells


def _build_sampling_cells(config: StudyConfig) -> List[LogicalCell]:
    cells: List[LogicalCell] = []
    for split in config.splits:
        for checkpoint in config.checkpoints:
            cells.append(
                LogicalCell(
                    stage="sampling",
                    dataset_split=split.name,
                    subset_name="Hard-32",
                    checkpoint_alias=checkpoint.alias,
                    factor_family="image_only_sampling",
                    logical_cell_id=_make_logical_cell_id(
                        stage="sampling",
                        split_name=split.name,
                        subset_name="Hard-32",
                        checkpoint_alias=checkpoint.alias,
                        factor_family="image_only_sampling",
                        prefix_mode="image_only",
                        prefix_length=0,
                        decode_family="sample",
                        max_new_tokens=config.sampling.max_new_tokens,
                        sample_k=config.sampling.k,
                    ),
                    decode_family="sample",
                    prefix_mode="image_only",
                    prefix_length=0,
                    max_new_tokens=config.sampling.max_new_tokens,
                    temperature=config.sampling.temperature,
                    top_p=config.sampling.top_p,
                    repetition_penalty=config.sampling.repetition_penalty,
                    sample_k=config.sampling.k,
                    sample_seed_base=config.sampling.seed,
                )
            )
    return cells


def _build_prefix_cells(config: StudyConfig) -> List[LogicalCell]:
    cells: List[LogicalCell] = []
    for split in config.splits:
        for checkpoint in config.checkpoints:
            for prefix_mode in (
                "oracle_gt_prefix_train_order",
                "oracle_gt_prefix_random_order",
                "self_prefix",
            ):
                for prefix_length in config.prefix.lengths:
                    cells.append(
                        LogicalCell(
                            stage="prefix",
                            dataset_split=split.name,
                            subset_name="Hard-32",
                            checkpoint_alias=checkpoint.alias,
                            factor_family="prefix_order",
                            logical_cell_id=_make_logical_cell_id(
                                stage="prefix",
                                split_name=split.name,
                                subset_name="Hard-32",
                                checkpoint_alias=checkpoint.alias,
                                factor_family="prefix_order",
                                prefix_mode=prefix_mode,
                                prefix_length=prefix_length,
                                decode_family="greedy",
                                max_new_tokens=config.baseline_decode.max_new_tokens,
                                sample_k=1,
                            ),
                            decode_family="greedy",
                            prefix_mode=prefix_mode,
                            prefix_length=int(prefix_length),
                            max_new_tokens=config.baseline_decode.max_new_tokens,
                            temperature=config.baseline_decode.temperature,
                            top_p=config.baseline_decode.top_p,
                            repetition_penalty=config.baseline_decode.repetition_penalty,
                            sample_k=1,
                            sample_seed_base=0,
                        )
                    )
    return cells


def _build_stress_cells(config: StudyConfig) -> List[LogicalCell]:
    cells: List[LogicalCell] = []
    aliases = [checkpoint.alias for checkpoint in config.checkpoints]
    for split in config.splits:
        for checkpoint in config.checkpoints:
            source_alias = next(alias for alias in aliases if alias != checkpoint.alias)
            cells.append(
                LogicalCell(
                    stage="stress",
                    dataset_split=split.name,
                    subset_name="Hard-32",
                    checkpoint_alias=checkpoint.alias,
                    factor_family="switched_prefix",
                    logical_cell_id=_make_logical_cell_id(
                        stage="stress",
                        split_name=split.name,
                        subset_name="Hard-32",
                        checkpoint_alias=checkpoint.alias,
                        factor_family="switched_prefix",
                        prefix_mode="switched_prefix",
                        prefix_length=config.stress.prefix_length,
                        decode_family="greedy",
                        max_new_tokens=config.baseline_decode.max_new_tokens,
                        sample_k=1,
                        prefix_source_checkpoint=source_alias,
                    ),
                    decode_family="greedy",
                    prefix_mode="switched_prefix",
                    prefix_length=config.stress.prefix_length,
                    max_new_tokens=config.baseline_decode.max_new_tokens,
                    temperature=config.baseline_decode.temperature,
                    top_p=config.baseline_decode.top_p,
                    repetition_penalty=config.baseline_decode.repetition_penalty,
                    sample_k=1,
                    sample_seed_base=0,
                    prefix_source_checkpoint=source_alias,
                )
            )
            for mutation in config.stress.mutations:
                cells.append(
                    LogicalCell(
                        stage="stress",
                        dataset_split=split.name,
                        subset_name="Hard-32",
                        checkpoint_alias=checkpoint.alias,
                        factor_family="broken_prefix",
                        logical_cell_id=_make_logical_cell_id(
                            stage="stress",
                            split_name=split.name,
                            subset_name="Hard-32",
                            checkpoint_alias=checkpoint.alias,
                            factor_family="broken_prefix",
                            prefix_mode="broken_prefix",
                            prefix_length=config.stress.prefix_length,
                            decode_family="greedy",
                            max_new_tokens=config.baseline_decode.max_new_tokens,
                            sample_k=1,
                            mutation_type=mutation,
                        ),
                        decode_family="greedy",
                        prefix_mode="broken_prefix",
                        prefix_length=config.stress.prefix_length,
                        max_new_tokens=config.baseline_decode.max_new_tokens,
                        temperature=config.baseline_decode.temperature,
                        top_p=config.baseline_decode.top_p,
                        repetition_penalty=config.baseline_decode.repetition_penalty,
                        sample_k=1,
                        sample_seed_base=0,
                        mutation_type=mutation,
                    )
                )
    return cells


def _build_length_cells(config: StudyConfig) -> List[LogicalCell]:
    cells: List[LogicalCell] = []
    for split in config.splits:
        for checkpoint in config.checkpoints:
            cells.append(
                LogicalCell(
                    stage="length",
                    dataset_split=split.name,
                    subset_name="Hard-32",
                    checkpoint_alias=checkpoint.alias,
                    factor_family="extended_length",
                    logical_cell_id=_make_logical_cell_id(
                        stage="length",
                        split_name=split.name,
                        subset_name="Hard-32",
                        checkpoint_alias=checkpoint.alias,
                        factor_family="extended_length",
                        prefix_mode="image_only",
                        prefix_length=0,
                        decode_family="greedy",
                        max_new_tokens=config.length.extended_max_new_tokens,
                        sample_k=1,
                    ),
                    decode_family="greedy",
                    prefix_mode="image_only",
                    prefix_length=0,
                    max_new_tokens=config.length.extended_max_new_tokens,
                    temperature=config.baseline_decode.temperature,
                    top_p=config.baseline_decode.top_p,
                    repetition_penalty=config.baseline_decode.repetition_penalty,
                    sample_k=1,
                    sample_seed_base=0,
                )
            )
    return cells


def _subset_records_by_name(records: Sequence[ImageRecord], subset_manifest: Mapping[str, Any]) -> List[Tuple[int, ImageRecord]]:
    ordered_ids = [int(item["source_image_id"]) for item in subset_manifest.get("images") or []]
    by_image_id = {int(record.source_image_id): record for record in records}
    return [(idx, by_image_id[image_id]) for idx, image_id in enumerate(ordered_ids)]


def _baseline_rows_by_checkpoint(
    baseline_cells: Sequence[LogicalCell],
    *,
    run_dir: Path,
) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    out: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for cell in baseline_cells:
        sample_rows = list(_iter_jsonl(_sample_dir(run_dir, cell, 0) / "gt_vs_pred.jsonl"))
        out[(cell.dataset_split, cell.checkpoint_alias)] = sample_rows
    return out


def _pick_gt_prefix_objects(
    record: ImageRecord,
    *,
    prefix_length: int,
) -> List[Dict[str, Any]]:
    return [dict(obj) for obj in list(record.objects)[: int(prefix_length)]]


def _randomize_prefix_objects(
    objects: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    indices = list(range(len(objects)))
    rng = random.Random(int(seed))
    rng.shuffle(indices)
    return ([dict(objects[idx]) for idx in indices], [int(idx) for idx in indices])


def _prefix_plan_for_mode(
    *,
    cell: LogicalCell,
    record: ImageRecord,
    checkpoint_rows_by_image_id: Mapping[Tuple[str, str], Mapping[int, Mapping[str, Any]]],
    object_field_order: str,
    random_seed: int,
) -> PrefixPlan:
    if cell.prefix_mode not in _VALID_PREFIX_MODES:
        raise ValueError(f"unsupported prefix_mode={cell.prefix_mode!r}")
    requested_n = int(cell.prefix_length)
    if cell.prefix_mode == "image_only":
        return PrefixPlan(
            prefix_mode="image_only",
            requested_prefix_length=0,
            actual_prefix_length=0,
            prefix_text=dumps_coordjson({"objects": []})[:-2],
            prefix_pred_objects=[],
            prefix_source_checkpoint=None,
            prefix_ordering_rule="image_only",
            prefix_hash=_prefix_hash(dumps_coordjson({"objects": []})[:-2]),
            continuation_pred_start_index=0,
        )

    if cell.prefix_mode == "oracle_gt_prefix_train_order":
        objects = _pick_gt_prefix_objects(record, prefix_length=requested_n)
        prefix_text, compact_objects = _serialize_objects_to_prefix_text(
            objects,
            width=record.width,
            height=record.height,
            object_field_order=object_field_order,
        )
        return PrefixPlan(
            prefix_mode=cell.prefix_mode,
            requested_prefix_length=requested_n,
            actual_prefix_length=len(compact_objects),
            prefix_text=prefix_text,
            prefix_pred_objects=compact_objects,
            prefix_source_checkpoint=None,
            prefix_ordering_rule="train_order",
            prefix_hash=_prefix_hash(prefix_text),
            continuation_pred_start_index=len(compact_objects),
        )

    if cell.prefix_mode == "oracle_gt_prefix_random_order":
        objects = _pick_gt_prefix_objects(record, prefix_length=requested_n)
        seed = int(random_seed + record.source_image_id + (requested_n * 1009))
        shuffled, permutation = _randomize_prefix_objects(objects, seed=seed)
        prefix_text, compact_objects = _serialize_objects_to_prefix_text(
            shuffled,
            width=record.width,
            height=record.height,
            object_field_order=object_field_order,
        )
        return PrefixPlan(
            prefix_mode=cell.prefix_mode,
            requested_prefix_length=requested_n,
            actual_prefix_length=len(compact_objects),
            prefix_text=prefix_text,
            prefix_pred_objects=compact_objects,
            prefix_source_checkpoint=None,
            prefix_ordering_rule="random_fixed_seed",
            prefix_hash=_prefix_hash(prefix_text),
            prefix_seed=seed,
            prefix_permutation=permutation,
            continuation_pred_start_index=len(compact_objects),
        )

    source_alias = cell.checkpoint_alias
    if cell.prefix_mode == "switched_prefix":
        if cell.prefix_source_checkpoint is None:
            raise ValueError("switched prefix requires source checkpoint")
        source_alias = cell.prefix_source_checkpoint
    source_row = dict(
        checkpoint_rows_by_image_id[(cell.dataset_split, source_alias)][
            int(record.source_image_id)
        ]
    )
    source_pred = list(source_row.get("pred") or [])[:requested_n]
    prefix_objects = [dict(obj) for obj in source_pred]
    if cell.prefix_mode == "broken_prefix":
        prefix_objects = _mutate_prefix_objects(
            prefix_objects=prefix_objects,
            record=record,
            mutation_type=str(cell.mutation_type or "delete"),
        )
    prefix_text, compact_objects = _serialize_objects_to_prefix_text(
        prefix_objects,
        width=record.width,
        height=record.height,
        object_field_order=object_field_order,
    )
    ordering_rule = "rollout_prefix_order"
    if cell.prefix_mode == "switched_prefix":
        ordering_rule = "switched_rollout_prefix_order"
    if cell.prefix_mode == "broken_prefix":
        ordering_rule = f"broken_rollout_prefix_order:{cell.mutation_type}"
    return PrefixPlan(
        prefix_mode=cell.prefix_mode,
        requested_prefix_length=requested_n,
        actual_prefix_length=len(compact_objects),
        prefix_text=prefix_text,
        prefix_pred_objects=compact_objects,
        prefix_source_checkpoint=source_alias,
        prefix_ordering_rule=ordering_rule,
        prefix_hash=_prefix_hash(prefix_text),
        mutation_type=cell.mutation_type,
        continuation_pred_start_index=len(compact_objects),
    )


def _mutate_prefix_objects(
    *,
    prefix_objects: Sequence[Mapping[str, Any]],
    record: ImageRecord,
    mutation_type: str,
) -> List[Dict[str, Any]]:
    objects = [dict(obj) for obj in prefix_objects]
    if mutation_type == "delete":
        if objects:
            return objects[:-1]
        return []
    if mutation_type == "adjacent_swap":
        if len(objects) >= 2:
            objects[0], objects[1] = objects[1], objects[0]
        return objects
    if mutation_type == "insert":
        candidates = list(record.objects)[len(objects) : len(objects) + 1]
        if candidates:
            inserted = dict(candidates[0])
            return [inserted, *objects]
        if objects:
            return [dict(objects[0]), *objects]
        return objects
    raise ValueError(f"unsupported mutation_type={mutation_type!r}")


def _build_prefix_plans_for_cell(
    *,
    cell: LogicalCell,
    subset_records: Sequence[Tuple[int, ImageRecord]],
    checkpoint_rows: Mapping[Tuple[str, str], Sequence[Mapping[str, Any]]],
    object_field_order: str,
    random_seed: int,
) -> Dict[int, PrefixPlan]:
    checkpoint_rows_by_image_id = {
        key: {
            int(row.get("source_image_id") or -1): row
            for row in rows
            if int(row.get("source_image_id") or -1) >= 0
        }
        for key, rows in checkpoint_rows.items()
    }
    return {
        int(subset_index): _prefix_plan_for_mode(
            cell=cell,
            record=record,
            checkpoint_rows_by_image_id=checkpoint_rows_by_image_id,
            object_field_order=object_field_order,
            random_seed=random_seed,
        )
        for subset_index, record in subset_records
    }


def _worker_assignments(
    config: StudyConfig,
) -> Dict[Tuple[str, str], int]:
    pairs = [(split.name, checkpoint.alias) for split in config.splits for checkpoint in config.checkpoints]
    gpu_ids = list(config.execution.gpu_ids)
    if not gpu_ids:
        raise ValueError("study requires at least one GPU id")
    return {pair: int(gpu_ids[idx % len(gpu_ids)]) for idx, pair in enumerate(pairs)}


def _task_exists(run_dir: Path, cell: LogicalCell, execution_shard_id: str) -> bool:
    for sample_idx in range(cell.sample_k):
        if not (_shard_dir(run_dir, cell, execution_shard_id, sample_idx) / "gt_vs_pred.jsonl").is_file():
            return False
    return True


def _run_cells(
    *,
    config: StudyConfig,
    run_dir: Path,
    cells: Sequence[LogicalCell],
    subset_records_by_split: Mapping[str, Sequence[Tuple[int, ImageRecord]]],
    resolved_checkpoints: Mapping[str, ResolvedCheckpoint],
    resolved_splits: Mapping[str, ResolvedSplit],
    prefix_plans_by_cell: Optional[Mapping[str, Mapping[int, PrefixPlan]]] = None,
) -> List[Dict[str, Any]]:
    assignments = _worker_assignments(config)
    tasks_by_worker: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    manifests: List[Dict[str, Any]] = []
    for cell in cells:
        pair = (cell.dataset_split, cell.checkpoint_alias)
        gpu_id = assignments[pair]
        execution_shard_id = f"gpu{gpu_id:02d}"
        if config.execution.reuse_existing and _task_exists(run_dir, cell, execution_shard_id):
            manifests.append(
                {
                    **_logical_cell_manifest(cell),
                    "execution_shard_id": execution_shard_id,
                    "gpu_id": gpu_id,
                    "reused_existing": True,
                }
            )
            continue
        prefix_plans = {
            int(idx): asdict(plan)
            for idx, plan in dict((prefix_plans_by_cell or {}).get(cell.logical_cell_id) or {}).items()
        }
        subset_records = list(subset_records_by_split[cell.dataset_split])
        tasks_by_worker[pair].append(
            {
                "cell": _logical_cell_manifest(cell),
                "execution_shard_id": execution_shard_id,
                "shard_dir": str(_cell_dir(run_dir, cell) / "shards" / execution_shard_id),
                "records": [
                    {
                        "subset_index": int(subset_index),
                        "record_idx": int(record.record_idx),
                        "source_image_id": int(record.source_image_id),
                        "image_rel": record.image_rel,
                        "file_name": record.file_name,
                        "width": int(record.width),
                        "height": int(record.height),
                        "objects": list(record.objects),
                        "raw_record": dict(record.raw_record),
                    }
                    for subset_index, record in subset_records
                ],
                "prefix_plans": prefix_plans,
            }
        )
        manifests.append(
            {
                **_logical_cell_manifest(cell),
                "execution_shard_id": execution_shard_id,
                "gpu_id": gpu_id,
                "reused_existing": False,
            }
        )

    ctx = mp.get_context(config.execution.start_method)
    procs: List[mp.Process] = []
    for (split_name, checkpoint_alias), tasks in tasks_by_worker.items():
        if not tasks:
            continue
        gpu_id = assignments[(split_name, checkpoint_alias)]
        resolved_checkpoint = resolved_checkpoints[checkpoint_alias]
        resolved_split = resolved_splits[split_name]
        proc = ctx.Process(
            target=_worker_main,
            kwargs={
                "gpu_id": gpu_id,
                "checkpoint": resolved_checkpoint,
                "split_image_root": resolved_split.image_root,
                "split_jsonl_path": resolved_split.jsonl_path,
                "tasks": tasks,
                "baseline_batch_size": config.baseline_decode.batch_size,
            },
        )
        proc.start()
        procs.append(proc)
    failures: List[str] = []
    for proc in procs:
        proc.join()
        if proc.exitcode != 0:
            failures.append(f"pid={proc.pid} exitcode={proc.exitcode}")
    if failures:
        raise RuntimeError(f"worker execution failed: {failures}")
    return manifests


def _merge_and_eval_cell(
    *,
    config: StudyConfig,
    run_dir: Path,
    cell: LogicalCell,
    execution_shard_id: str,
) -> Dict[str, Any]:
    cell_dir = _cell_dir(run_dir, cell)
    cell_dir.mkdir(parents=True, exist_ok=True)
    sample_artifacts: List[Dict[str, Any]] = []
    sample_paths: List[Path] = []
    for sample_idx in range(cell.sample_k):
        shard_path = _shard_dir(run_dir, cell, execution_shard_id, sample_idx) / "gt_vs_pred.jsonl"
        if not shard_path.is_file():
            raise FileNotFoundError(f"missing shard output for {cell.logical_cell_id}: {shard_path}")
        rows = list(_iter_jsonl(shard_path))
        rows.sort(key=lambda row: int(row.get("index", -1)))
        sample_dir = _sample_dir(run_dir, cell, sample_idx)
        sample_dir.mkdir(parents=True, exist_ok=True)
        gt_vs_pred_path = sample_dir / "gt_vs_pred.jsonl"
        _write_jsonl(gt_vs_pred_path, rows)
        eval_dir = sample_dir / "eval"
        eval_summary = evaluate_and_save(
            gt_vs_pred_path,
            options=_build_eval_options(config, output_dir=eval_dir),
        )
        proposal_rows = build_rollout_proposal_table(
            checkpoint_name=cell.checkpoint_alias,
            run_dir=sample_dir,
            temperature=float(cell.temperature),
        )
        _write_jsonl(sample_dir / "proposal_table.jsonl", proposal_rows)
        health = _build_rollout_health(
            gt_vs_pred_path=gt_vs_pred_path,
            proposal_rows=proposal_rows,
            gate=config.health_gate,
        )
        continuation_eval_dir = None
        if cell.prefix_mode != "image_only":
            continuation_rows = _continuation_only_rows(rows)
            continuation_path = sample_dir / "gt_vs_pred_continuation.jsonl"
            _write_jsonl(continuation_path, continuation_rows)
            continuation_eval_dir = sample_dir / "eval_continuation"
            evaluate_and_save(
                continuation_path,
                options=_build_eval_options(config, output_dir=continuation_eval_dir),
            )
        sample_artifacts.append(
            {
                "sample_idx": sample_idx,
                "sample_dir": str(sample_dir),
                "gt_vs_pred_jsonl": str(gt_vs_pred_path),
                "eval_dir": str(eval_dir),
                "eval_summary": eval_summary,
                "rollout_health": health,
                "continuation_eval_dir": str(continuation_eval_dir) if continuation_eval_dir else None,
            }
        )
        sample_paths.append(gt_vs_pred_path)
    _validate_cell_alignment(sample_paths)
    health_summary = _aggregate_sample_health(sample_artifacts)
    cell_manifest = {
        **_logical_cell_manifest(cell),
        "execution_shard_id": execution_shard_id,
        "rollout_health": health_summary,
        "rollout_health_valid": bool(health_summary.get("rollout_health_valid")),
        "rollout_health_invalid_reason": health_summary.get("rollout_health_invalid_reason"),
        "samples": sample_artifacts,
    }
    _write_json(cell_dir / "cell_manifest.json", cell_manifest)
    return cell_manifest


def _run_stage_cells(
    *,
    config: StudyConfig,
    run_dir: Path,
    stage_name: str,
    cells: Sequence[LogicalCell],
    subset_records_by_split: Mapping[str, Sequence[Tuple[int, ImageRecord]]],
    resolved_checkpoints: Mapping[str, ResolvedCheckpoint],
    resolved_splits: Mapping[str, ResolvedSplit],
    prefix_plans_by_cell: Optional[Mapping[str, Mapping[int, PrefixPlan]]] = None,
) -> List[Dict[str, Any]]:
    if not cells:
        return []
    stage_dir = run_dir / f"{stage_name}_stage"
    stage_dir.mkdir(parents=True, exist_ok=True)
    execution_manifests = _run_cells(
        config=config,
        run_dir=run_dir,
        cells=cells,
        subset_records_by_split=subset_records_by_split,
        resolved_checkpoints=resolved_checkpoints,
        resolved_splits=resolved_splits,
        prefix_plans_by_cell=prefix_plans_by_cell,
    )
    cell_manifests: List[Dict[str, Any]] = []
    for manifest in execution_manifests:
        cell = LogicalCell(
            stage=str(manifest["stage"]),
            dataset_split=str(manifest["dataset_split"]),
            subset_name=str(manifest["subset_name"]),
            checkpoint_alias=str(manifest["checkpoint_alias"]),
            factor_family=str(manifest["factor_family"]),
            logical_cell_id=str(manifest["logical_cell_id"]),
            decode_family=str(manifest["decode_family"]),
            prefix_mode=str(manifest["prefix_mode"]),
            prefix_length=int(manifest["prefix_length"]),
            max_new_tokens=int(manifest["max_new_tokens"]),
            temperature=float(manifest["temperature"]),
            top_p=float(manifest["top_p"]),
            repetition_penalty=float(manifest["repetition_penalty"]),
            sample_k=int(manifest["sample_k"]),
            sample_seed_base=int(manifest["sample_seed_base"]),
            prefix_source_checkpoint=manifest.get("prefix_source_checkpoint"),
            mutation_type=manifest.get("mutation_type"),
        )
        cell_manifests.append(
            _merge_and_eval_cell(
                config=config,
                run_dir=run_dir,
                cell=cell,
                execution_shard_id=str(manifest["execution_shard_id"]),
            )
        )
    _write_json(stage_dir / "stage_manifest.json", {"stage": stage_name, "cells": cell_manifests})
    return cell_manifests


def _bootstrap_image_health_reason(row: Mapping[str, Any]) -> Optional[str]:
    errors = [str(error).strip() for error in (row.get("errors") or []) if str(error).strip()]
    if errors:
        return "parse_invalid"
    if row.get("raw_output_json") is None and not row.get("pred"):
        return "invalid_rollout"
    return None


def _bootstrap_rank_for_split(
    *,
    split_name: str,
    records: Sequence[Tuple[int, ImageRecord]],
    bootstrap_cell_manifests: Sequence[Mapping[str, Any]],
    config: StudyConfig,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    rows_by_checkpoint: Dict[str, List[Dict[str, Any]]] = {}
    matches_by_checkpoint: Dict[str, List[Dict[str, Any]]] = {}
    for manifest in bootstrap_cell_manifests:
        if str(manifest["dataset_split"]) != split_name:
            continue
        checkpoint_alias = str(manifest["checkpoint_alias"])
        sample_dir = Path(str(manifest["samples"][0]["sample_dir"]))
        rows_by_checkpoint[checkpoint_alias] = list(_iter_jsonl(sample_dir / "gt_vs_pred.jsonl"))
        matches_by_checkpoint[checkpoint_alias] = list(_iter_jsonl(sample_dir / "eval" / "matches.jsonl"))

    checkpoint_aliases = sorted(rows_by_checkpoint.keys())
    if len(checkpoint_aliases) != 2:
        raise ValueError(f"bootstrap ranking expects 2 checkpoints, got {checkpoint_aliases}")

    match_indexes = {
        alias: {
            int(row.get("image_id", idx)): row
            for idx, row in enumerate(matches_by_checkpoint[alias])
        }
        for alias in checkpoint_aliases
    }
    ranked_rows: List[Dict[str, Any]] = []
    for subset_index, record in records:
        exclude_reasons: List[str] = []
        unresolved_counts: List[int] = []
        unmatched_pred_counts: List[int] = []
        for alias in checkpoint_aliases:
            row = rows_by_checkpoint[alias][subset_index]
            health_reason = _bootstrap_image_health_reason(row)
            if health_reason is not None:
                exclude_reasons.append(f"{alias}:{health_reason}")
            match_row = match_indexes[alias].get(subset_index, {})
            unresolved_counts.append(len(match_row.get("unmatched_gt_indices") or []))
            unmatched_pred_counts.append(len(match_row.get("unmatched_pred_indices") or []))
        ranked_rows.append(
            {
                "dataset_split": split_name,
                "subset_index": int(subset_index),
                "source_image_id": int(record.source_image_id),
                "file_name": record.file_name,
                "gt_count": int(len(record.objects)),
                "mean_unresolved_gt_count": float(mean(unresolved_counts)),
                "mean_unmatched_pred_count": float(mean(unmatched_pred_counts)),
                "exclude_reason": ",".join(exclude_reasons) if exclude_reasons else None,
            }
        )
    ranked_rows.sort(
        key=lambda row: (
            row.get("exclude_reason") is not None,
            -float(row["mean_unresolved_gt_count"]),
            -float(row["mean_unmatched_pred_count"]),
            -int(row["gt_count"]),
            int(row["source_image_id"]),
        )
    )
    valid_rows = [row for row in ranked_rows if row.get("exclude_reason") is None]
    hard32_rows = valid_rows[: int(config.bootstrap.hard32_size)]
    hard16_rows = hard32_rows[: int(config.bootstrap.hard16_size)]
    hard32_manifest = {
        "dataset_split": split_name,
        "subset_name": "Hard-32",
        "selection_strategy": "bootstrap_ranked_top_prefix",
        "bootstrap_ranking_tuple": [
            "desc_mean_unresolved_gt_count",
            "desc_mean_unmatched_pred_count",
            "desc_gt_count",
            "asc_source_image_id",
        ],
        "images": hard32_rows,
    }
    hard16_manifest = {
        "dataset_split": split_name,
        "subset_name": "Hard-16",
        "selection_strategy": "bootstrap_ranked_top_prefix",
        "parent_subset_name": "Hard-32",
        "images": hard16_rows,
    }
    return ranked_rows, hard32_manifest, hard16_manifest


def _materialize_subset_artifacts(
    *,
    run_dir: Path,
    split_name: str,
    records: Sequence[ImageRecord],
    hard32_manifest: Mapping[str, Any],
    hard16_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    by_image_id = {int(record.source_image_id): record for record in records}
    subset_outputs: Dict[str, Any] = {}
    for manifest in (hard32_manifest, hard16_manifest):
        subset_name = str(manifest["subset_name"])
        subset_dir = _hard_subset_dir(run_dir, split_name, subset_name)
        rows = [
            _subset_row(by_image_id[int(item["source_image_id"])], subset_index=idx)
            for idx, item in enumerate(manifest.get("images") or [])
        ]
        subset_jsonl = subset_dir / "subset.coord.jsonl"
        _write_jsonl(subset_jsonl, rows)
        _write_json(subset_dir / "subset_manifest.json", dict(manifest))
        subset_outputs[subset_name] = {
            "subset_jsonl": str(subset_jsonl),
            "subset_manifest": str(subset_dir / "subset_manifest.json"),
            "num_images": len(rows),
        }
    return subset_outputs


def _sample_manifest_lookup(cell_manifests: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, str, str, str], Mapping[str, Any]]:
    return {
        (
            str(manifest["stage"]),
            str(manifest["dataset_split"]),
            str(manifest["checkpoint_alias"]),
            str(manifest["logical_cell_id"]),
        ): manifest
        for manifest in cell_manifests
    }


def _load_gt_index_rows(cell_manifest: Mapping[str, Any], sample_idx: int = 0) -> List[Dict[str, Any]]:
    sample_dir = Path(str((cell_manifest.get("samples") or [])[sample_idx]["sample_dir"]))
    return list(_iter_jsonl(sample_dir / "gt_vs_pred.jsonl"))


def _all_gt_keys(rows: Sequence[Mapping[str, Any]]) -> List[Tuple[int, int]]:
    keys: List[Tuple[int, int]] = []
    for row in rows:
        record_idx = int(row.get("index", -1))
        for gt_idx, _ in enumerate(row.get("gt") or []):
            keys.append((record_idx, int(gt_idx)))
    return keys


def _union_hit_stats(
    *,
    sample_dirs: Sequence[Path],
    use_continuation: bool,
    thresholds: Sequence[float],
) -> Dict[str, Any]:
    if not sample_dirs:
        return {"thresholds": {}}
    gt_paths = [
        (
            sample_dir / "gt_vs_pred.jsonl"
            if not use_continuation
            else sample_dir / "gt_vs_pred_continuation.jsonl"
        )
        for sample_dir in sample_dirs
    ]
    _validate_cell_alignment(gt_paths if not use_continuation else [sample_dirs[0] / "gt_vs_pred.jsonl"])
    base_rows = list(_iter_jsonl(sample_dirs[0] / "gt_vs_pred.jsonl"))
    all_keys = _all_gt_keys(base_rows)
    output: Dict[str, Any] = {"thresholds": {}}
    for thr in thresholds:
        key = _thr_key(thr)
        full_indexes: List[Dict[Tuple[int, int], Dict[str, Any]]] = []
        loc_indexes: List[Dict[Tuple[int, int], Dict[str, Any]]] = []
        for sample_dir in sample_dirs:
            eval_dir = (
                sample_dir / "eval_continuation" if use_continuation else sample_dir / "eval"
            )
            match_index = _load_match_indexes_for_run(eval_dir, thresholds=thresholds)[key]
            loc_indexes.append(match_index["loc"])
            full_indexes.append(match_index["full"])
        per_gt_rows: List[Dict[str, Any]] = []
        for record_idx, gt_idx in all_keys:
            loc_count = sum(int((record_idx, gt_idx) in loc) for loc in loc_indexes)
            full_count = sum(int((record_idx, gt_idx) in full) for full in full_indexes)
            k = len(sample_dirs)
            if full_count <= 0:
                frequency_bucket = "never_hit"
            elif full_count >= k:
                frequency_bucket = "always_hit"
            elif full_count * 2 < k:
                frequency_bucket = "rare_hit"
            else:
                frequency_bucket = "often_hit"
            per_gt_rows.append(
                {
                    "record_idx": int(record_idx),
                    "gt_idx": int(gt_idx),
                    "hit_count_loc": int(loc_count),
                    "hit_count_full": int(full_count),
                    "hit_fraction_loc": float(loc_count / float(k)),
                    "hit_fraction_full": float(full_count / float(k)),
                    "ever_hit_loc": bool(loc_count > 0),
                    "ever_hit_full": bool(full_count > 0),
                    "frequency_bucket": frequency_bucket,
                }
            )
        output["thresholds"][key] = {
            "per_gt_rows": per_gt_rows,
            "union_recall_full": float(
                sum(1 for row in per_gt_rows if bool(row["ever_hit_full"])) / float(len(per_gt_rows))
            )
            if per_gt_rows
            else 0.0,
        }
    return output


def _baseline_summary_rows(
    *,
    baseline_manifests: Sequence[Mapping[str, Any]],
    subset_name: str,
    subset_prefix_size: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for manifest in baseline_manifests:
        sample = dict((manifest.get("samples") or [])[0] or {})
        eval_summary = dict(sample.get("eval_summary") or {})
        metrics = dict(eval_summary.get("metrics") or {})
        gt_rows = list(_iter_jsonl(Path(str(sample["gt_vs_pred_jsonl"]))))
        gt_object_count = sum(len(row.get("gt") or []) for row in gt_rows[:subset_prefix_size])
        rows.append(
            {
                "dataset_split": str(manifest["dataset_split"]),
                "subset_name": subset_name,
                "checkpoint_alias": str(manifest["checkpoint_alias"]),
                "mAP": metrics.get("mAP"),
                "f1_full_micro@0.50": metrics.get("f1ish@0.50_f1_full_micro"),
                "precision_full_micro@0.50": metrics.get("f1ish@0.50_precision_full_micro"),
                "recall_full_micro@0.50": metrics.get("f1ish@0.50_recall_full_micro"),
                "gt_object_count": int(gt_object_count),
                "rollout_health_valid": bool(sample.get("rollout_health", {}).get("rollout_health_valid")),
                "rollout_health_invalid_reason": sample.get("rollout_health", {}).get(
                    "rollout_health_invalid_reason"
                ),
            }
        )
    return rows


def _quality_flags_for_key(
    *,
    key: Tuple[int, int],
    baseline_indexes_050: Dict[str, Dict[Tuple[int, int], Dict[str, Any]]],
    baseline_indexes_030: Dict[str, Dict[Tuple[int, int], Dict[str, Any]]],
) -> Dict[str, bool]:
    full_050 = baseline_indexes_050["full"].get(key)
    loc_050 = baseline_indexes_050["loc"].get(key)
    full_030 = baseline_indexes_030["full"].get(key)
    annotation_mismatch_candidate = full_050 is None and full_030 is not None
    semantic_confusion_candidate = loc_050 is not None and full_050 is None
    return {
        "annotation_mismatch_candidate": bool(annotation_mismatch_candidate),
        "semantic_confusion_candidate": bool(semantic_confusion_candidate),
    }


def _image_metadata(rows: Sequence[Mapping[str, Any]]) -> Dict[int, Dict[str, Any]]:
    return {
        int(row.get("index", -1)): {
            "record_idx": int(row.get("index", -1)),
            "source_image_id": int(row.get("source_image_id", -1)),
            "file_name": row.get("file_name"),
            "gt_count": len(row.get("gt") or []),
        }
        for row in rows
    }


def _build_recovery_table(
    *,
    split_name: str,
    checkpoint_alias: str,
    baseline_manifest: Mapping[str, Any],
    sampling_manifest: Mapping[str, Any],
    prefix_manifests: Sequence[Mapping[str, Any]],
    length_manifest: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    baseline_sample_dir = Path(str((baseline_manifest.get("samples") or [])[0]["sample_dir"]))
    baseline_rows = list(_iter_jsonl(baseline_sample_dir / "gt_vs_pred.jsonl"))
    metadata_by_record = _image_metadata(baseline_rows)
    baseline_indexes = _load_match_indexes_for_run(
        baseline_sample_dir / "eval",
        thresholds=(_SECONDARY_IOU_THR, _PRIMARY_IOU_THR),
    )
    sampling_sample_dirs = [
        Path(str(sample["sample_dir"])) for sample in (sampling_manifest.get("samples") or [])
    ]
    sampling_union = _union_hit_stats(
        sample_dirs=sampling_sample_dirs,
        use_continuation=False,
        thresholds=(_SECONDARY_IOU_THR, _PRIMARY_IOU_THR),
    )
    prefix_union_by_mode: Dict[str, Dict[str, Any]] = {}
    for manifest in prefix_manifests:
        sample_dirs = [Path(str(sample["sample_dir"])) for sample in (manifest.get("samples") or [])]
        prefix_union_by_mode[str(manifest["logical_cell_id"])] = _union_hit_stats(
            sample_dirs=sample_dirs,
            use_continuation=True,
            thresholds=(_SECONDARY_IOU_THR, _PRIMARY_IOU_THR),
        )
    length_sample_dir = Path(str((length_manifest.get("samples") or [])[0]["sample_dir"]))
    length_indexes = _load_match_indexes_for_run(
        length_sample_dir / "eval",
        thresholds=(_SECONDARY_IOU_THR, _PRIMARY_IOU_THR),
    )
    baseline_full_050 = baseline_indexes[_thr_key(_PRIMARY_IOU_THR)]["full"]
    baseline_loc_050 = baseline_indexes[_thr_key(_PRIMARY_IOU_THR)]["loc"]
    baseline_full_030 = baseline_indexes[_thr_key(_SECONDARY_IOU_THR)]["full"]
    baseline_loc_030 = baseline_indexes[_thr_key(_SECONDARY_IOU_THR)]["loc"]
    sampling_rows_050 = {
        (int(row["record_idx"]), int(row["gt_idx"])): row
        for row in sampling_union["thresholds"][_thr_key(_PRIMARY_IOU_THR)]["per_gt_rows"]
    }
    prefix_rows_050_by_cell = {
        cell_id: {
            (int(row["record_idx"]), int(row["gt_idx"])): row
            for row in payload["thresholds"][_thr_key(_PRIMARY_IOU_THR)]["per_gt_rows"]
        }
        for cell_id, payload in prefix_union_by_mode.items()
    }
    length_full_050 = length_indexes[_thr_key(_PRIMARY_IOU_THR)]["full"]
    recovery_rows: List[Dict[str, Any]] = []
    for record_idx, meta in sorted(metadata_by_record.items()):
        gt_count = int(meta["gt_count"])
        for gt_idx in range(gt_count):
            key = (int(record_idx), int(gt_idx))
            deterministic_hit = key in baseline_full_050
            sampling_row = sampling_rows_050.get(
                key,
                {
                    "ever_hit_full": False,
                    "hit_fraction_full": 0.0,
                    "frequency_bucket": "never_hit",
                },
            )
            prefix_recoveries = [
                cell_id
                for cell_id, rows_by_key in prefix_rows_050_by_cell.items()
                if bool(rows_by_key.get(key, {}).get("ever_hit_full"))
            ]
            length_recovered = key in length_full_050
            if deterministic_hit:
                status = "deterministic_hit"
            elif bool(sampling_row.get("ever_hit_full")):
                status = "decode_selection_miss"
            elif prefix_recoveries:
                status = "prefix_sensitive_miss"
            elif length_recovered:
                status = "length_bias_miss"
            else:
                status = "persistent_unrecovered"
            quality_flags = _quality_flags_for_key(
                key=key,
                baseline_indexes_050={"loc": baseline_loc_050, "full": baseline_full_050},
                baseline_indexes_030={"loc": baseline_loc_030, "full": baseline_full_030},
            )
            recovery_rows.append(
                {
                    "dataset_split": split_name,
                    "checkpoint_alias": checkpoint_alias,
                    "record_idx": int(record_idx),
                    "gt_idx": int(gt_idx),
                    "source_image_id": int(meta["source_image_id"]),
                    "file_name": meta["file_name"],
                    "status": status,
                    "deterministic_hit": bool(deterministic_hit),
                    "sampling_hit_fraction": float(sampling_row.get("hit_fraction_full", 0.0)),
                    "sampling_frequency_bucket": sampling_row.get("frequency_bucket", "never_hit"),
                    "prefix_recovered_by_cells": prefix_recoveries,
                    "length_recovered": bool(length_recovered),
                    **quality_flags,
                }
            )
    summary = {
        "dataset_split": split_name,
        "checkpoint_alias": checkpoint_alias,
        "status_counts": {
            status: int(sum(1 for row in recovery_rows if row["status"] == status))
            for status in (
                "deterministic_hit",
                "decode_selection_miss",
                "prefix_sensitive_miss",
                "length_bias_miss",
                "persistent_unrecovered",
            )
        },
    }
    return recovery_rows, summary


def _load_stage_manifest(run_dir: Path, stage_name: str) -> List[Dict[str, Any]]:
    path = run_dir / f"{stage_name}_stage" / "stage_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("cells") or [])


def _load_stage_manifest_if_present(run_dir: Path, stage_name: str) -> List[Dict[str, Any]]:
    path = run_dir / f"{stage_name}_stage" / "stage_manifest.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("cells") or [])


def _subset_manifest_for_split(run_dir: Path, split_name: str, subset_name: str) -> Dict[str, Any]:
    path = _hard_subset_dir(run_dir, split_name, subset_name) / "subset_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _subset_prefix_filter(rows: Sequence[Mapping[str, Any]], prefix_size: int) -> List[Dict[str, Any]]:
    return [dict(row) for row in rows if int(row.get("record_idx", row.get("index", -1))) < int(prefix_size)]


def _render_report(
    *,
    config: StudyConfig,
    resolved_manifest: Mapping[str, Any],
    bootstrap_rankings: Mapping[str, Sequence[Mapping[str, Any]]],
    recovery_payload: Mapping[Tuple[str, str], Tuple[Sequence[Mapping[str, Any]], Mapping[str, Any]]],
    baseline_manifests: Sequence[Mapping[str, Any]],
    prefix_manifests: Sequence[Mapping[str, Any]],
    stress_manifests: Sequence[Mapping[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append(f"# Rollout FN-Factor Study: {config.run.name}")
    lines.append("")
    lines.append("## Fixed Inputs")
    lines.append("")
    for checkpoint in resolved_manifest.get("checkpoints") or []:
        lines.append(
            f"- `{checkpoint['alias']}`: `{checkpoint['path']}` "
            f"(prompt=`{checkpoint['prompt_variant']}`, field_order=`{checkpoint['object_field_order']}`)"
        )
    for split in resolved_manifest.get("splits") or []:
        lines.append(f"- `{split['name']}`: `{split['jsonl_path']}`")
    lines.append("")
    lines.append("## Bootstrap Hard-Case Cohorts")
    lines.append("")
    for split_name, rows in bootstrap_rankings.items():
        top_rows = list(rows[:5])
        lines.append(f"### {split_name}")
        for row in top_rows:
            lines.append(
                f"- image_id `{row['source_image_id']}` `{row['file_name']}`: "
                f"mean_unresolved_gt={row['mean_unresolved_gt_count']:.2f}, "
                f"mean_unmatched_pred={row['mean_unmatched_pred_count']:.2f}, gt_count={row['gt_count']}"
            )
        lines.append("")
    lines.append("## Recovery Waterfall")
    lines.append("")
    for split_name in sorted({key[0] for key in recovery_payload.keys()}):
        lines.append(f"### {split_name}")
        for checkpoint_alias in sorted({key[1] for key in recovery_payload.keys() if key[0] == split_name}):
            rows, summary = recovery_payload[(split_name, checkpoint_alias)]
            hard32_rows = list(rows)
            hard16_rows = _subset_prefix_filter(rows, int(config.bootstrap.hard16_size))
            for subset_name, subset_rows in (("Hard-16", hard16_rows), ("Hard-32", hard32_rows)):
                counts = {
                    status: int(sum(1 for row in subset_rows if row["status"] == status))
                    for status in config.report.bucket_order
                }
                total = len(subset_rows)
                lines.append(f"- `{checkpoint_alias}` / `{subset_name}`:")
                lines.append(
                    f"  total_gt={total}, deterministic_hit={counts['deterministic_hit']}, "
                    f"decode_selection_miss={counts['decode_selection_miss']}, "
                    f"prefix_sensitive_miss={counts['prefix_sensitive_miss']}, "
                    f"length_bias_miss={counts['length_bias_miss']}, "
                    f"persistent_unrecovered={counts['persistent_unrecovered']}"
                )
                mismatch_count = sum(
                    1 for row in subset_rows if bool(row.get("annotation_mismatch_candidate"))
                )
                confusion_count = sum(
                    1 for row in subset_rows if bool(row.get("semantic_confusion_candidate"))
                )
                lines.append(
                    f"  annotation_mismatch_candidate={mismatch_count}, semantic_confusion_candidate={confusion_count}"
                )
        lines.append("")
    lines.append("## Prefix Findings")
    lines.append("")
    prefix_by_group: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for manifest in prefix_manifests:
        prefix_by_group[(str(manifest["dataset_split"]), str(manifest["checkpoint_alias"]))].append(
            manifest
        )
    for key, manifests in sorted(prefix_by_group.items()):
        split_name, checkpoint_alias = key
        lines.append(f"### {split_name} / {checkpoint_alias}")
        for manifest in sorted(
            manifests,
            key=lambda item: (str(item["prefix_mode"]), int(item["prefix_length"])),
        ):
            sample = dict((manifest.get("samples") or [])[0] or {})
            health = sample.get("rollout_health") or {}
            lines.append(
                f"- `{manifest['prefix_mode']}` n={manifest['prefix_length']}: "
                f"health_valid={health.get('rollout_health_valid')}, "
                f"invalid_reason={health.get('rollout_health_invalid_reason')}"
            )
        lines.append("")
    if stress_manifests:
        lines.append("## Stress Findings")
        lines.append("")
        for manifest in stress_manifests:
            sample = dict((manifest.get("samples") or [])[0] or {})
            health = sample.get("rollout_health") or {}
            lines.append(
                f"- `{manifest['dataset_split']}` / `{manifest['checkpoint_alias']}` / "
                f"`{manifest['factor_family']}` / `{manifest['prefix_mode']}`"
                f"{' / ' + str(manifest['mutation_type']) if manifest.get('mutation_type') else ''}: "
                f"health_valid={health.get('rollout_health_valid')} "
                f"reason={health.get('rollout_health_invalid_reason')}"
            )
        lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- `decode_selection_miss` means same-prompt union-of-K recovered the object before any prefix/length intervention."
    )
    lines.append(
        "- `prefix_sensitive_miss` is scored from continuation-only recovery; injected prefix objects do not count."
    )
    lines.append(
        "- `length_bias_miss` means default deterministic/image-only rollout missed the object but the extended-length control recovered it."
    )
    lines.append(
        "- `persistent_unrecovered` should be read as unrecovered under tested interventions, not proven incapacity."
    )
    return "\n".join(lines).strip() + "\n"


def _write_recovery_outputs(
    *,
    run_dir: Path,
    recovery_payload: Mapping[Tuple[str, str], Tuple[Sequence[Mapping[str, Any]], Mapping[str, Any]]],
) -> Dict[str, Any]:
    out_dir = run_dir / "recovery"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {"tables": []}
    for (split_name, checkpoint_alias), (rows, summary) in sorted(recovery_payload.items()):
        slug = _slugify(f"{split_name}-{checkpoint_alias}")
        jsonl_path = out_dir / f"{slug}.jsonl"
        summary_path = out_dir / f"{slug}.summary.json"
        _write_jsonl(jsonl_path, rows)
        _write_json(summary_path, summary)
        manifest["tables"].append(
            {
                "dataset_split": split_name,
                "checkpoint_alias": checkpoint_alias,
                "rows_jsonl": str(jsonl_path),
                "summary_json": str(summary_path),
            }
        )
    _write_json(out_dir / "recovery_manifest.json", manifest)
    return manifest


def _load_metrics_json(sample_dir: Path, *, continuation: bool = False) -> Dict[str, Any]:
    eval_dir = sample_dir / ("eval_continuation" if continuation else "eval")
    metrics_path = eval_dir / "metrics.json"
    if not metrics_path.is_file():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _materialize_review_queue(
    *,
    config: StudyConfig,
    run_dir: Path,
    recovery_payload: Mapping[Tuple[str, str], Tuple[Sequence[Mapping[str, Any]], Mapping[str, Any]]],
    baseline_manifests: Sequence[Mapping[str, Any]],
    resolved_splits: Mapping[str, ResolvedSplit],
) -> Dict[str, Any]:
    baseline_lookup = {
        (str(manifest["dataset_split"]), str(manifest["checkpoint_alias"])): manifest
        for manifest in baseline_manifests
    }
    queue_rows: List[Dict[str, Any]] = []
    for (split_name, checkpoint_alias), (rows, _summary) in recovery_payload.items():
        baseline_manifest = baseline_lookup[(split_name, checkpoint_alias)]
        baseline_sample_dir = Path(str((baseline_manifest.get("samples") or [])[0]["sample_dir"]))
        split_image_root = resolved_splits[split_name].image_root
        vis_resource = materialize_gt_vs_pred_vis_resource(
            baseline_sample_dir / "gt_vs_pred.jsonl",
            source_kind="rollout_fn_factor_study",
        )
        overlay_dir = baseline_sample_dir / "overlays"
        render_gt_vs_pred_review(
            vis_resource,
            out_dir=overlay_dir,
            limit=32,
            root_image_dir=split_image_root,
            root_source=f"split:{split_name}",
            record_order="error_first",
        )
        for bucket in config.report.bucket_order:
            bucket_rows = [row for row in rows if str(row.get("status")) == bucket]
            bucket_rows = bucket_rows[: int(config.report.review_top_n_per_bucket)]
            for row in bucket_rows:
                queue_rows.append(
                    {
                        **dict(row),
                        "baseline_logical_cell_id": baseline_manifest["logical_cell_id"],
                        "baseline_sample_dir": str(baseline_sample_dir),
                        "baseline_overlay_dir": str(overlay_dir),
                    }
                )
    queue_path = run_dir / "review_queue.jsonl"
    _write_jsonl(queue_path, queue_rows)
    return {"review_queue_jsonl": str(queue_path), "count": len(queue_rows)}


def run_study(config_path: Path) -> Dict[str, Any]:
    config = load_study_config(config_path)
    resolved_checkpoints, resolved_splits = _resolve_study_inputs(config)
    run_dir = (REPO_ROOT / config.run.output_dir / config.run.name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_manifest = {
        "config_path": str(config_path.resolve()),
        "run_dir": str(run_dir),
        "checkpoints": [
            {
                "alias": checkpoint.alias,
                "path": str(checkpoint.path),
                "artifact_kind": checkpoint.artifact_kind,
                "fingerprint": checkpoint.fingerprint,
                "prompt_variant": checkpoint.prompt_variant,
                "object_field_order": checkpoint.object_field_order,
                "prompt_hash": _prompt_hash(checkpoint.prompt_variant, checkpoint.object_field_order),
                "prompt_control_source": checkpoint.prompt_control_source,
                "provenance_sidecars": checkpoint.provenance_sidecars,
            }
            for checkpoint in resolved_checkpoints.values()
        ],
        "splits": [
            {
                "name": split.name,
                "jsonl_path": str(split.jsonl_path),
                "image_root": str(split.image_root),
            }
            for split in resolved_splits.values()
        ],
        "prompt_contract": {
            "prompt_variant": config.prompts.prompt_variant,
            "object_field_order": config.prompts.object_field_order,
            "do_resize": False,
        },
        "eval": asdict(config.eval),
        "health_gate": asdict(config.health_gate),
        "execution": asdict(config.execution),
    }
    _write_json(run_dir / "resolved_manifest.json", resolved_manifest)

    all_split_records = {
        split_name: _read_split_records(split)
        for split_name, split in resolved_splits.items()
    }

    bootstrap_rankings: Dict[str, Sequence[Mapping[str, Any]]] = {}
    hard32_manifests: Dict[str, Dict[str, Any]] = {}
    hard16_manifests: Dict[str, Dict[str, Any]] = {}
    bootstrap_cell_manifests: List[Dict[str, Any]]
    if _stage_enabled(config, "bootstrap"):
        candidate_records_by_split = {
            split_name: list(enumerate(_select_candidate_pool(records, config.bootstrap)))
            for split_name, records in all_split_records.items()
        }
        for split_name, subset_records in candidate_records_by_split.items():
            subset_dir = _candidate_subset_dir(run_dir, split_name)
            subset_rows = [_subset_row(record, subset_index=idx) for idx, (_orig, record) in enumerate(subset_records)]
            _write_jsonl(subset_dir / "candidate_pool.coord.jsonl", subset_rows)
            _write_json(
                subset_dir / "bootstrap_manifest.json",
                {
                    "dataset_split": split_name,
                    "candidate_pool_strategy": config.bootstrap.candidate_pool_strategy,
                    "candidate_pool_limit": config.bootstrap.candidate_pool_limit,
                    "candidate_pool_jsonl": str(subset_dir / "candidate_pool.coord.jsonl"),
                },
            )
        bootstrap_cells = _build_bootstrap_cells(config)
        bootstrap_cell_manifests = _run_stage_cells(
            config=config,
            run_dir=run_dir,
            stage_name="bootstrap",
            cells=bootstrap_cells,
            subset_records_by_split=candidate_records_by_split,
            resolved_checkpoints=resolved_checkpoints,
            resolved_splits=resolved_splits,
        )
        for split_name, subset_records in candidate_records_by_split.items():
            ranked_rows, hard32_manifest, hard16_manifest = _bootstrap_rank_for_split(
                split_name=split_name,
                records=subset_records,
                bootstrap_cell_manifests=bootstrap_cell_manifests,
                config=config,
            )
            bootstrap_rankings[split_name] = ranked_rows
            hard32_manifests[split_name] = hard32_manifest
            hard16_manifests[split_name] = hard16_manifest
            subset_outputs = _materialize_subset_artifacts(
                run_dir=run_dir,
                split_name=split_name,
                records=all_split_records[split_name],
                hard32_manifest=hard32_manifest,
                hard16_manifest=hard16_manifest,
            )
            _write_json(
                _candidate_subset_dir(run_dir, split_name) / "bootstrap_ranked.json",
                {"ranked_rows": ranked_rows, "subset_outputs": subset_outputs},
            )
    else:
        bootstrap_cell_manifests = _load_stage_manifest(run_dir, "bootstrap")
        for split_name in resolved_splits:
            ranked_payload = json.loads(
                (_candidate_subset_dir(run_dir, split_name) / "bootstrap_ranked.json").read_text(
                    encoding="utf-8"
                )
            )
            bootstrap_rankings[split_name] = list(ranked_payload.get("ranked_rows") or [])
            hard32_manifests[split_name] = _subset_manifest_for_split(run_dir, split_name, "Hard-32")
            hard16_manifests[split_name] = _subset_manifest_for_split(run_dir, split_name, "Hard-16")

    hard32_subset_records = {
        split_name: _subset_records_by_name(all_split_records[split_name], manifest)
        for split_name, manifest in hard32_manifests.items()
    }
    _write_json(
        run_dir / "subset_manifest.json",
        {
            "Hard-32": hard32_manifests,
            "Hard-16": hard16_manifests,
        },
    )

    needs_baseline = any(
        _stage_enabled(config, stage_name)
        for stage_name in ("baseline", "sampling", "prefix", "stress", "length", "report")
    )
    if needs_baseline and _stage_enabled(config, "baseline"):
        baseline_manifests = _run_stage_cells(
            config=config,
            run_dir=run_dir,
            stage_name="baseline",
            cells=_build_baseline_cells(config),
            subset_records_by_split=hard32_subset_records,
            resolved_checkpoints=resolved_checkpoints,
            resolved_splits=resolved_splits,
        )
    elif needs_baseline:
        baseline_manifests = _load_stage_manifest(run_dir, "baseline")
    else:
        baseline_manifests = []

    needs_sampling = any(
        _stage_enabled(config, stage_name) for stage_name in ("sampling", "report")
    )
    if needs_sampling and _stage_enabled(config, "sampling"):
        sampling_manifests = _run_stage_cells(
            config=config,
            run_dir=run_dir,
            stage_name="sampling",
            cells=_build_sampling_cells(config),
            subset_records_by_split=hard32_subset_records,
            resolved_checkpoints=resolved_checkpoints,
            resolved_splits=resolved_splits,
        )
    elif needs_sampling:
        sampling_manifests = _load_stage_manifest(run_dir, "sampling")
    else:
        sampling_manifests = []

    baseline_rows_lookup = (
        _baseline_rows_by_checkpoint(
            [
                LogicalCell(
                    stage=str(manifest["stage"]),
                    dataset_split=str(manifest["dataset_split"]),
                    subset_name=str(manifest["subset_name"]),
                    checkpoint_alias=str(manifest["checkpoint_alias"]),
                    factor_family=str(manifest["factor_family"]),
                    logical_cell_id=str(manifest["logical_cell_id"]),
                    decode_family=str(manifest["decode_family"]),
                    prefix_mode=str(manifest["prefix_mode"]),
                    prefix_length=int(manifest["prefix_length"]),
                    max_new_tokens=int(manifest["max_new_tokens"]),
                    temperature=float(manifest["temperature"]),
                    top_p=float(manifest["top_p"]),
                    repetition_penalty=float(manifest["repetition_penalty"]),
                    sample_k=int(manifest["sample_k"]),
                    sample_seed_base=int(manifest["sample_seed_base"]),
                    prefix_source_checkpoint=manifest.get("prefix_source_checkpoint"),
                    mutation_type=manifest.get("mutation_type"),
                )
                for manifest in baseline_manifests
            ],
            run_dir=run_dir,
        )
        if baseline_manifests
        else {}
    )

    prefix_plans_by_cell = {
        cell.logical_cell_id: _build_prefix_plans_for_cell(
            cell=cell,
            subset_records=hard32_subset_records[cell.dataset_split],
            checkpoint_rows=baseline_rows_lookup,
            object_field_order=resolved_checkpoints[cell.checkpoint_alias].object_field_order,
            random_seed=config.prefix.random_seed,
        )
        for cell in _build_prefix_cells(config)
    }
    needs_prefix = any(_stage_enabled(config, stage_name) for stage_name in ("prefix", "report"))
    if needs_prefix and _stage_enabled(config, "prefix"):
        prefix_manifests = _run_stage_cells(
            config=config,
            run_dir=run_dir,
            stage_name="prefix",
            cells=_build_prefix_cells(config),
            subset_records_by_split=hard32_subset_records,
            resolved_checkpoints=resolved_checkpoints,
            resolved_splits=resolved_splits,
            prefix_plans_by_cell=prefix_plans_by_cell,
        )
    elif needs_prefix:
        prefix_manifests = _load_stage_manifest_if_present(run_dir, "prefix")
    else:
        prefix_manifests = []

    stress_cells = _build_stress_cells(config)
    stress_prefix_plans = {
        cell.logical_cell_id: _build_prefix_plans_for_cell(
            cell=cell,
            subset_records=hard32_subset_records[cell.dataset_split],
            checkpoint_rows=baseline_rows_lookup,
            object_field_order=resolved_checkpoints[cell.checkpoint_alias].object_field_order,
            random_seed=config.prefix.random_seed,
        )
        for cell in stress_cells
    }
    needs_stress = any(_stage_enabled(config, stage_name) for stage_name in ("stress", "report"))
    if needs_stress and _stage_enabled(config, "stress"):
        stress_manifests = _run_stage_cells(
            config=config,
            run_dir=run_dir,
            stage_name="stress",
            cells=stress_cells,
            subset_records_by_split=hard32_subset_records,
            resolved_checkpoints=resolved_checkpoints,
            resolved_splits=resolved_splits,
            prefix_plans_by_cell=stress_prefix_plans,
        )
    elif needs_stress:
        stress_manifests = _load_stage_manifest_if_present(run_dir, "stress")
    else:
        stress_manifests = []

    needs_length = any(_stage_enabled(config, stage_name) for stage_name in ("length", "report"))
    if needs_length and _stage_enabled(config, "length"):
        length_manifests = _run_stage_cells(
            config=config,
            run_dir=run_dir,
            stage_name="length",
            cells=_build_length_cells(config),
            subset_records_by_split=hard32_subset_records,
            resolved_checkpoints=resolved_checkpoints,
            resolved_splits=resolved_splits,
        )
    elif needs_length:
        length_manifests = _load_stage_manifest(run_dir, "length")
    else:
        length_manifests = []

    report_manifest: Optional[Dict[str, Any]] = None
    if _stage_enabled(config, "report"):
        baseline_by_key = {
            (str(manifest["dataset_split"]), str(manifest["checkpoint_alias"])): manifest
            for manifest in baseline_manifests
        }
        sampling_by_key = {
            (str(manifest["dataset_split"]), str(manifest["checkpoint_alias"])): manifest
            for manifest in sampling_manifests
        }
        length_by_key = {
            (str(manifest["dataset_split"]), str(manifest["checkpoint_alias"])): manifest
            for manifest in length_manifests
        }
        prefix_by_key: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
        for manifest in prefix_manifests:
            prefix_by_key[(str(manifest["dataset_split"]), str(manifest["checkpoint_alias"]))].append(
                manifest
            )

        recovery_payload: Dict[
            Tuple[str, str], Tuple[Sequence[Mapping[str, Any]], Mapping[str, Any]]
        ] = {}
        for split_name in resolved_splits:
            for checkpoint_alias in resolved_checkpoints:
                key = (split_name, checkpoint_alias)
                recovery_payload[key] = _build_recovery_table(
                    split_name=split_name,
                    checkpoint_alias=checkpoint_alias,
                    baseline_manifest=baseline_by_key[key],
                    sampling_manifest=sampling_by_key[key],
                    prefix_manifests=prefix_by_key[key],
                    length_manifest=length_by_key[key],
                )

        recovery_manifest = _write_recovery_outputs(run_dir=run_dir, recovery_payload=recovery_payload)
        review_manifest = _materialize_review_queue(
            config=config,
            run_dir=run_dir,
            recovery_payload=recovery_payload,
            baseline_manifests=baseline_manifests,
            resolved_splits=resolved_splits,
        )
        report_text = _render_report(
            config=config,
            resolved_manifest=resolved_manifest,
            bootstrap_rankings=bootstrap_rankings,
            recovery_payload=recovery_payload,
            baseline_manifests=baseline_manifests,
            prefix_manifests=prefix_manifests,
            stress_manifests=stress_manifests,
        )
        report_path = run_dir / "report.md"
        report_path.write_text(report_text, encoding="utf-8")
        report_manifest = {
            "report_path": str(report_path),
            "recovery_manifest": recovery_manifest,
            "review_manifest": review_manifest,
        }
        _write_json(run_dir / "report_manifest.json", report_manifest)

    study_manifest = {
        "run_dir": str(run_dir),
        "resolved_manifest_path": str(run_dir / "resolved_manifest.json"),
        "subset_manifest_path": str(run_dir / "subset_manifest.json"),
        "bootstrap_stage_manifest": str(run_dir / "bootstrap_stage" / "stage_manifest.json"),
        "baseline_stage_manifest": str(run_dir / "baseline_stage" / "stage_manifest.json"),
        "sampling_stage_manifest": str(run_dir / "sampling_stage" / "stage_manifest.json"),
        "prefix_stage_manifest": str(run_dir / "prefix_stage" / "stage_manifest.json"),
        "stress_stage_manifest": str(run_dir / "stress_stage" / "stage_manifest.json"),
        "length_stage_manifest": str(run_dir / "length_stage" / "stage_manifest.json"),
        "report_manifest_path": str(run_dir / "report_manifest.json")
        if report_manifest is not None
        else None,
    }
    _write_json(run_dir / "study_manifest.json", study_manifest)
    return study_manifest


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to the study YAML config.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = run_study(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def run_rollout_fn_factor_study(config_path: Path) -> Dict[str, Any]:
    return run_study(config_path)


if __name__ == "__main__":
    main()
