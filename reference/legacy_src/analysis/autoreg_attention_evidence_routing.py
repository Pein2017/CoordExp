from __future__ import annotations

import json
import re
import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from src.datasets.geometry import box_iou_xyxy, valid_xyxy_box


ATTENTION_STAGES = (
    "select_cases",
    "feasibility",
    "attention_atlas",
    "merge",
    "report",
)

ATTENTION_QUERY_ROLES = (
    "final_generated_prefix_state",
    "row_start",
    "desc_end",
    "box_start",
    "pre_x1",
)

ATTENTION_TEACHER_FORCED_QUERY_ROLES = (
    "desc_end",
    "box_start",
    "pre_x1",
    "post_x1",
    "post_y1",
)

MERGE_JSONL_FILES = (
    "selected_cases.jsonl",
    "candidate_region_rows.jsonl",
    "feasibility_rows.jsonl",
    "attention_region_rows.jsonl",
    "decision_context_rows.jsonl",
)

_COORD_TOKEN_RE = re.compile(r"^<\|coord_(\d+)\|>$")


@dataclass(frozen=True)
class AttentionPaths:
    artifact_root: Path
    checkpoint: Path
    dataset_jsonl: Path
    lane_a_rollout_root: Path
    lane_c_per_case: Path
    lane_c_study_config: Path
    lane_d_selected_cases: Path
    self_rollout_root: Path


@dataclass(frozen=True)
class AttentionSelectionConfig:
    sample_limit: int
    max_cases: int
    prefer_prefix_mode: str
    min_remaining_gt: int
    case_source: str = "lane_d_self_prefix_remaining_gt"
    scope_label: str = "val200_self_rollout"


@dataclass(frozen=True)
class AttentionRegionConfig:
    context_expansion_norm1000: int
    duplicate_iou_threshold: float = 0.95


@dataclass(frozen=True)
class AttentionExecutionConfig:
    batch_size: int
    attn_implementation: str
    torch_dtype: str
    max_feasibility_cases: int


@dataclass(frozen=True)
class AttentionConfig:
    config_path: Path
    paths: AttentionPaths
    selection: AttentionSelectionConfig
    regions: AttentionRegionConfig
    execution: AttentionExecutionConfig


def attention_shard_label(shard_index: int, num_shards: int) -> str:
    return f"shard_{int(shard_index):03d}-of-{int(num_shards):03d}"


def normalize_attention_shard(
    *, shard_index: int | None, num_shards: int | None
) -> tuple[int | None, int | None, str | None]:
    if shard_index is None and num_shards is None:
        return None, None, None
    if shard_index is None or num_shards is None:
        raise ValueError("shard_index and num_shards must be provided together")
    shard_count = int(num_shards)
    shard_i = int(shard_index)
    if shard_count <= 0:
        raise ValueError("num_shards must be positive")
    if shard_i < 0 or shard_i >= shard_count:
        raise ValueError("shard_index must be in [0, num_shards)")
    return shard_i, shard_count, attention_shard_label(shard_i, shard_count)


def attention_record_selected(
    source_line_idx: int,
    *,
    shard_index: int | None,
    num_shards: int | None,
) -> bool:
    idx = int(source_line_idx)
    if idx < 0:
        return False
    shard_index, num_shards, _ = normalize_attention_shard(
        shard_index=shard_index,
        num_shards=num_shards,
    )
    if shard_index is None or num_shards is None:
        return True
    return idx % num_shards == shard_index


def _desc(obj: Mapping[str, Any]) -> str:
    return str(obj.get("desc", obj.get("category", ""))).strip()


def _box(obj: Mapping[str, Any]) -> Sequence[object]:
    box = obj.get("bbox_xyxy", obj.get("bbox_2d", obj.get("bbox", ())))
    tokens = getattr(box, "tokens", None)
    if tokens is not None:
        box = tokens
    if not isinstance(box, Sequence) or isinstance(box, (str, bytes)):
        return ()
    return box


def _coord_value(value: object) -> float:
    if isinstance(value, str):
        match = _COORD_TOKEN_RE.fullmatch(value.strip())
        if match is not None:
            return float(int(match.group(1)))
    return float(value)


def _coerce_xyxy_box(box: Any) -> list[float] | None:
    tokens = getattr(box, "tokens", None)
    if tokens is not None:
        box = tokens
    if isinstance(box, (str, bytes)) or not isinstance(box, Sequence) or len(box) != 4:
        return None
    try:
        out = [_coord_value(value) for value in box]
    except (TypeError, ValueError):
        return None
    if not valid_xyxy_box(out):
        return None
    return out


def classify_extra_prediction(
    prediction: Mapping[str, Any],
    previous_predictions: Sequence[Mapping[str, Any]],
    *,
    duplicate_iou_threshold: float = 0.95,
) -> str:
    pred_box = _coerce_xyxy_box(_box(prediction))
    if pred_box is None:
        return "format_or_geometry_invalid_extra"
    pred_desc = _desc(prediction)
    for previous in previous_predictions:
        prev_box = _coerce_xyxy_box(_box(previous))
        if pred_desc and pred_desc == _desc(previous):
            if prev_box is not None and box_iou_xyxy(pred_box, prev_box) > float(
                duplicate_iou_threshold
            ):
                return "same_desc_iou_gt_0p95_duplicate"
    return "other_extra_prediction"


def _required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"config.{key} must be a mapping")
    return value


def _required_path(payload: Mapping[str, Any], key: str, *, base: Path) -> Path:
    raw = payload.get(key)
    if raw is None or str(raw).strip() == "":
        raise ValueError(f"missing required path: {key}")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path


def _required_int(payload: Mapping[str, Any], key: str, *, minimum: int) -> int:
    if key not in payload:
        raise ValueError(f"missing required integer: {key}")
    value = int(payload.get(key))
    if value < minimum:
        raise ValueError(f"{key} must be >= {minimum}")
    return value


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = str(payload.get(key, "")).strip()
    if not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def load_attention_config(path: Path | str) -> AttentionConfig:
    config_path = Path(path).expanduser()
    repo_root = Path(__file__).resolve().parents[2]
    base = Path("/") if config_path.is_absolute() else repo_root
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"attention config must be a mapping: {config_path}")
    paths = _required_mapping(payload, "paths")
    selection = _required_mapping(payload, "selection")
    regions = _required_mapping(payload, "regions")
    execution = _required_mapping(payload, "execution")
    return AttentionConfig(
        config_path=config_path,
        paths=AttentionPaths(
            artifact_root=_required_path(paths, "artifact_root", base=base),
            checkpoint=_required_path(paths, "checkpoint", base=base),
            dataset_jsonl=_required_path(paths, "dataset_jsonl", base=base),
            lane_a_rollout_root=_required_path(paths, "lane_a_rollout_root", base=base),
            lane_c_per_case=_required_path(paths, "lane_c_per_case", base=base),
            lane_c_study_config=_required_path(paths, "lane_c_study_config", base=base),
            lane_d_selected_cases=_required_path(paths, "lane_d_selected_cases", base=base),
            self_rollout_root=_required_path(paths, "self_rollout_root", base=base),
        ),
        selection=AttentionSelectionConfig(
            sample_limit=_required_int(selection, "sample_limit", minimum=1),
            max_cases=_required_int(selection, "max_cases", minimum=1),
            prefer_prefix_mode=_required_str(selection, "prefer_prefix_mode"),
            min_remaining_gt=_required_int(selection, "min_remaining_gt", minimum=0),
            case_source=str(
                selection.get("case_source", "lane_d_self_prefix_remaining_gt")
            ).strip(),
            scope_label=str(selection.get("scope_label", "val200_self_rollout")).strip(),
        ),
        regions=AttentionRegionConfig(
            context_expansion_norm1000=_required_int(
                regions,
                "context_expansion_norm1000",
                minimum=0,
            ),
            duplicate_iou_threshold=float(regions.get("duplicate_iou_threshold", 0.95)),
        ),
        execution=AttentionExecutionConfig(
            batch_size=_required_int(execution, "batch_size", minimum=1),
            attn_implementation=_required_str(execution, "attn_implementation"),
            torch_dtype=_required_str(execution, "torch_dtype"),
            max_feasibility_cases=_required_int(
                execution,
                "max_feasibility_cases",
                minimum=1,
            ),
        ),
    )


def build_attention_dry_run_plan(
    config: AttentionConfig,
    *,
    stages: Sequence[str],
    shard_index: int | None,
    num_shards: int | None,
) -> dict[str, Any]:
    shard_index, num_shards, shard_label = normalize_attention_shard(
        shard_index=shard_index,
        num_shards=num_shards,
    )
    unknown = [stage for stage in stages if stage not in ATTENTION_STAGES]
    if unknown:
        raise ValueError(f"unknown attention stage(s): {unknown}")
    return {
        "artifact_root": str(config.paths.artifact_root),
        "stages": list(stages),
        "shard_index": shard_index,
        "num_shards": num_shards,
        "shard_label": shard_label,
        "sample_limit": config.selection.sample_limit,
        "max_cases": config.selection.max_cases,
        "case_source": config.selection.case_source,
        "scope_label": config.selection.scope_label,
        "attn_implementation": config.execution.attn_implementation,
    }


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def attention_config_expected_merge_metadata(config: AttentionConfig) -> dict[str, Any]:
    return {
        "analysis_name": "autoreg_attention_evidence_routing",
        "schema_version": 1,
        "config_path": str(config.config_path),
        "config_sha256": _sha256_path(config.config_path),
        "checkpoint": str(config.paths.checkpoint),
        "dataset_jsonl": str(config.paths.dataset_jsonl),
        "artifact_root": str(config.paths.artifact_root),
        "case_source": config.selection.case_source,
        "scope_label": config.selection.scope_label,
        "attn_implementation": config.execution.attn_implementation,
        "torch_dtype": config.execution.torch_dtype,
    }


def write_attention_shards_manifest(
    config: AttentionConfig,
    num_shards: int,
) -> dict[str, Any]:
    shard_count = int(num_shards)
    if shard_count <= 0:
        raise ValueError("num_shards must be positive")
    manifest = {
        **attention_config_expected_merge_metadata(config),
        "num_shards": shard_count,
        "expected_shards": shard_count,
        "shard_labels": [
            attention_shard_label(index, shard_count) for index in range(shard_count)
        ],
    }
    _write_json(config.paths.artifact_root / "shards_manifest.json", manifest)
    return manifest


def build_attention_lane_c_config(config: AttentionConfig) -> Any:
    from src.analysis.hard_ce_coord_logit_locality import load_study_config

    lane_c_config = load_study_config(config.paths.lane_c_study_config)
    return replace(
        lane_c_config,
        paths=replace(
            lane_c_config.paths,
            dataset_jsonl=config.paths.dataset_jsonl,
            artifact_root=config.paths.artifact_root,
            self_rollout_root=config.paths.self_rollout_root,
            lane_a_rollout_root=config.paths.lane_a_rollout_root,
        ),
        model=replace(
            lane_c_config.model,
            attn_implementation=config.execution.attn_implementation,
            torch_dtype=config.execution.torch_dtype,
        ),
    )


def prepare_attention_teacher_forced_lane_c_examples(
    lane_c_config: Any,
    *,
    model_handle: Any,
    limit: int | None,
    shard_index: int | None,
    num_shards: int | None,
) -> tuple[list[Any], dict[str, Any]]:
    """Build only Lane-C teacher-forced forced-continuation examples.

    This intentionally does not call `prepare_lane_c_x1_basin_examples`, because
    that helper also constructs self-prefix examples from rollout sidecars.
    """

    from src.analysis.hard_ce_coord_logit_locality import (
        _lane_c_build_forward_example,
        _lane_c_gt_bins_by_source_index,
        _lane_c_teacher_prefix_state,
        _ordering_plan,
        _render_normalized_object,
        _resolve_prompts,
        normalize_detection_row,
        parse_raw_detection_row,
        resolve_coord_token_ids,
        select_lane_c_intended_target_gt_idx,
    )

    rows = _read_jsonl(lane_c_config.paths.dataset_jsonl)
    active_limit = min(
        lane_c_config.execution.sample_limit if limit is None else int(limit),
        len(rows),
    )
    scope_label = f"limit={active_limit}"
    system_prompt, user_prompt = _resolve_prompts(lane_c_config)
    vocab = resolve_coord_token_ids(model_handle.tokenizer)
    examples: list[Any] = []
    skipped: Counter[str] = Counter()
    selected_record_count = 0
    for row_index in range(active_limit):
        if not attention_record_selected(
            row_index,
            shard_index=shard_index,
            num_shards=num_shards,
        ):
            continue
        selected_record_count += 1
        raw = parse_raw_detection_row(rows[row_index])
        normalized = normalize_detection_row(
            raw,
            object_ordering=_ordering_plan(lane_c_config, row_index=row_index),
        )
        gt_count = len(normalized.objects)
        if gt_count <= 0:
            skipped["empty_gt"] += 1
            continue
        teacher_order_gt_indices = tuple(
            int(obj.source_object_index) for obj in normalized.objects
        )
        objects_by_source = {
            int(obj.source_object_index): obj for obj in normalized.objects
        }
        gt_bins_by_index = _lane_c_gt_bins_by_source_index(normalized)
        for depth in range(gt_count):
            prefix_state = _lane_c_teacher_prefix_state(
                teacher_order_gt_indices,
                depth=depth,
                gt_count=gt_count,
            )
            selection = select_lane_c_intended_target_gt_idx(
                teacher_order_gt_indices,
                prefix_state,
            )
            example = _lane_c_build_forward_example(
                lane_c_config,
                model_handle=model_handle,
                raw=raw,
                normalized=normalized,
                source_line_idx=row_index,
                prefix_text="\n".join(
                    _render_normalized_object(obj)
                    for obj in normalized.objects[:depth]
                ),
                prefix_mode="teacher_forced",
                prefix_depth=depth,
                prefix_state=prefix_state,
                selection=selection,
                objects_by_source=objects_by_source,
                gt_bins_by_index=gt_bins_by_index,
                scope_label=scope_label,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                vocab=vocab,
            )
            if example is None:
                skipped["teacher_forced_unrenderable"] += 1
            else:
                examples.append(example)
    return examples, {
        "selected_record_count": selected_record_count,
        "skipped_counts": dict(sorted(skipped.items())),
        "planned_example_count": len(examples),
        "prefix_modes": ["teacher_forced"],
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=True, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _clip_box_norm1000(box: Sequence[object]) -> list[int]:
    coerced = _coerce_xyxy_box(box)
    if coerced is None:
        raise ValueError(f"invalid xyxy box: {box!r}")
    x1, y1, x2, y2 = [int(round(float(value))) for value in coerced]
    return [
        max(0, min(999, x1)),
        max(0, min(999, y1)),
        max(0, min(999, x2)),
        max(0, min(999, y2)),
    ]


def _expanded_context_box(box: Sequence[object], expansion: int) -> list[int]:
    x1, y1, x2, y2 = _clip_box_norm1000(box)
    return [
        max(0, x1 - int(expansion)),
        max(0, y1 - int(expansion)),
        min(999, x2 + int(expansion)),
        min(999, y2 + int(expansion)),
    ]


def select_attention_cases_from_rows(
    lane_d_cases: Sequence[Mapping[str, Any]],
    *,
    shard_index: int | None,
    num_shards: int | None,
    max_cases: int,
    prefer_prefix_mode: str,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in lane_d_cases:
        source_line_idx = int(row["source_line_idx"])
        if not attention_record_selected(
            source_line_idx,
            shard_index=shard_index,
            num_shards=num_shards,
        ):
            continue
        if str(row.get("prefix_mode")) != prefer_prefix_mode:
            continue
        case_id = str(row.get("case_id", ""))
        if not case_id or case_id in seen:
            continue
        if row.get("intended_target_gt_idx") is None:
            continue
        seen.add(case_id)
        out = dict(row)
        out["attention_case_family"] = "missed_gt_evidence_routing"
        out["selection_policy"] = "lane_d_self_prefix_remaining_gt"
        selected.append(out)
        if len(selected) >= int(max_cases):
            break
    return selected


def select_teacher_forced_anchor_cases_from_dataset_rows(
    dataset_rows: Sequence[Mapping[str, Any]],
    *,
    lane_c_config: Any,
    shard_index: int | None,
    num_shards: int | None,
    max_cases: int,
    scope_label: str,
) -> list[dict[str, Any]]:
    if lane_c_config is not None:
        from src.analysis.hard_ce_coord_logit_locality import (
            _lane_c_teacher_prefix_state,
            _ordering_plan,
            normalize_detection_row,
            parse_raw_detection_row,
            select_lane_c_intended_target_gt_idx,
        )
    else:
        _lane_c_teacher_prefix_state = None
        _ordering_plan = None
        normalize_detection_row = None
        parse_raw_detection_row = None
        select_lane_c_intended_target_gt_idx = None

    selected: list[dict[str, Any]] = []
    for source_line_idx, row in enumerate(dataset_rows):
        if not attention_record_selected(
            source_line_idx,
            shard_index=shard_index,
            num_shards=num_shards,
        ):
            continue
        if lane_c_config is None:
            objects = row.get("objects")
            if not isinstance(objects, Sequence) or isinstance(objects, (str, bytes)):
                continue
            teacher_order_gt_indices = tuple(range(len(objects)))
            objects_by_source = {
                gt_idx: obj for gt_idx, obj in enumerate(objects) if isinstance(obj, Mapping)
            }
        else:
            raw = parse_raw_detection_row(row)
            normalized = normalize_detection_row(
                raw,
                object_ordering=_ordering_plan(lane_c_config, row_index=source_line_idx),
            )
            teacher_order_gt_indices = tuple(
                int(obj.source_object_index) for obj in normalized.objects
            )
            objects_by_source = {
                int(obj.source_object_index): {
                    "desc": str(obj.desc),
                    "bbox_2d": list(getattr(obj.bbox_2d, "tokens", ())),
                }
                for obj in normalized.objects
            }
        gt_count = len(teacher_order_gt_indices)
        for depth in range(gt_count):
            if lane_c_config is None:
                gt_idx = teacher_order_gt_indices[depth]
            else:
                prefix_state = _lane_c_teacher_prefix_state(
                    teacher_order_gt_indices,
                    depth=depth,
                    gt_count=gt_count,
                )
                selection = select_lane_c_intended_target_gt_idx(
                    teacher_order_gt_indices,
                    prefix_state,
                )
                gt_idx = selection.intended_target_gt_idx
            if gt_idx is None or int(gt_idx) not in objects_by_source:
                continue
            obj = objects_by_source[int(gt_idx)]
            if _coerce_xyxy_box(_box(obj)) is None:
                continue
            selected.append(
                {
                    "case_id": (
                        f"row{source_line_idx}:teacher_forced:"
                        f"depth{depth}:gt{int(gt_idx)}"
                    ),
                    "source_line_idx": source_line_idx,
                    "prefix_mode": "teacher_forced",
                    "prefix_depth": depth,
                    "prefix_quality": "gt_prefix",
                    "intended_target_gt_idx": int(gt_idx),
                    "target_desc": _desc(obj),
                    "attention_case_family": "train_teacher_forced_anchor",
                    "selection_policy": "teacher_forced_all_gt_objects",
                    "scope_label": scope_label,
                }
            )
            if len(selected) >= int(max_cases):
                return selected
    return selected


def build_candidate_region_rows(
    selected_case: Mapping[str, Any],
    dataset_row: Mapping[str, Any],
    *,
    context_expansion_norm1000: int,
    shard_index: int,
    num_shards: int,
) -> list[dict[str, Any]]:
    source_line_idx = int(selected_case["source_line_idx"])
    target_gt_idx = int(selected_case["intended_target_gt_idx"])
    objects = dataset_row.get("objects")
    if not isinstance(objects, Sequence) or isinstance(objects, (str, bytes)):
        raise ValueError("dataset row missing objects")
    if target_gt_idx < 0 or target_gt_idx >= len(objects):
        raise ValueError(f"target_gt_idx out of range: {target_gt_idx}")
    target = objects[target_gt_idx]
    if not isinstance(target, Mapping):
        raise ValueError("target object must be a mapping")
    target_desc = _desc(target)
    target_box = _clip_box_norm1000(_box(target))
    shard_label = attention_shard_label(shard_index, num_shards)
    base = {
        "case_id": selected_case["case_id"],
        "source_line_idx": source_line_idx,
        "prefix_mode": selected_case["prefix_mode"],
        "prefix_depth": selected_case["prefix_depth"],
        "prefix_quality": selected_case["prefix_quality"],
        "target_gt_idx": target_gt_idx,
        "target_desc": selected_case.get("target_desc", target_desc),
        "shard_index": shard_index,
        "num_shards": num_shards,
        "shard_label": shard_label,
    }
    rows: list[dict[str, Any]] = [
        {
            **base,
            "region_kind": "target_gt",
            "gt_idx": target_gt_idx,
            "desc": target_desc,
            "bbox_xyxy": target_box,
        },
        {
            **base,
            "region_kind": "context_ring",
            "gt_idx": target_gt_idx,
            "desc": target_desc,
            "bbox_xyxy": _expanded_context_box(target_box, context_expansion_norm1000),
            "exclude_bbox_xyxy": target_box,
        },
    ]
    for gt_idx, obj in enumerate(objects):
        if gt_idx == target_gt_idx or not isinstance(obj, Mapping):
            continue
        if target_desc and _desc(obj) == target_desc:
            rows.append(
                {
                    **base,
                    "region_kind": "same_desc_gt",
                    "gt_idx": gt_idx,
                    "desc": _desc(obj),
                    "bbox_xyxy": _clip_box_norm1000(_box(obj)),
                }
            )
    rows.append(
        {
            **base,
            "region_kind": "far_background",
            "gt_idx": None,
            "desc": None,
            "bbox_xyxy": [0, 0, 999, 999],
            "exclude_bbox_xyxy": target_box,
        }
    )
    return rows


def materialize_attention_select_cases_shard(
    config: AttentionConfig,
    *,
    shard_index: int,
    num_shards: int,
) -> dict[str, Any]:
    shard_index, num_shards, shard_label = normalize_attention_shard(
        shard_index=shard_index,
        num_shards=num_shards,
    )
    assert shard_index is not None and num_shards is not None and shard_label is not None
    dataset_rows = _read_jsonl(config.paths.dataset_jsonl)
    if config.selection.case_source == "teacher_forced_anchor":
        lane_c_config = build_attention_lane_c_config(config)
        selected = select_teacher_forced_anchor_cases_from_dataset_rows(
            dataset_rows[: config.selection.sample_limit],
            lane_c_config=lane_c_config,
            shard_index=shard_index,
            num_shards=num_shards,
            max_cases=config.selection.max_cases,
            scope_label=config.selection.scope_label,
        )
        selection_policy = "teacher_forced_all_gt_objects"
    else:
        lane_d_cases = _read_jsonl(config.paths.lane_d_selected_cases)
        selected = select_attention_cases_from_rows(
            lane_d_cases,
            shard_index=shard_index,
            num_shards=num_shards,
            max_cases=config.selection.max_cases,
            prefer_prefix_mode=config.selection.prefer_prefix_mode,
        )
        selection_policy = "lane_d_self_prefix_remaining_gt"
    candidate_rows: list[dict[str, Any]] = []
    for case in selected:
        candidate_rows.extend(
            build_candidate_region_rows(
                case,
                dataset_rows[int(case["source_line_idx"])],
                context_expansion_norm1000=config.regions.context_expansion_norm1000,
                shard_index=shard_index,
                num_shards=num_shards,
            )
        )
    shard_dir = config.paths.artifact_root / "shards" / shard_label
    manifest = write_attention_shards_manifest(config, num_shards)
    _write_jsonl(shard_dir / "selected_cases.jsonl", selected)
    _write_jsonl(shard_dir / "candidate_region_rows.jsonl", candidate_rows)
    for filename in (
        "feasibility_rows.jsonl",
        "attention_region_rows.jsonl",
        "decision_context_rows.jsonl",
    ):
        path = shard_dir / filename
        if not path.exists():
            _write_jsonl(path, [])
    summary = {
        "analysis_name": "autoreg_attention_evidence_routing",
        "schema_version": 1,
        "stage": "attention_evidence_routing",
        "stages_completed": ["select_cases"],
        "artifact_root": str(config.paths.artifact_root),
        "shard_index": shard_index,
        "num_shards": num_shards,
        "shard_label": shard_label,
        "row_counts": {
            "selected_cases": len(selected),
            "candidate_region_rows": len(candidate_rows),
        },
        "case_source": config.selection.case_source,
        "scope_label": config.selection.scope_label,
        "config_sha256": manifest["config_sha256"],
        "selection_policy": selection_policy,
        "duplicate_policy": "same_desc_iou_gt_0p95",
        "output_paths": {
            "selected_cases": str(shard_dir / "selected_cases.jsonl"),
            "candidate_region_rows": str(shard_dir / "candidate_region_rows.jsonl"),
            "summary": str(shard_dir / "summary.json"),
        },
    }
    _write_json(shard_dir / "summary.json", summary)
    return summary


def find_visual_token_spans(
    input_ids: Sequence[int],
    *,
    image_token_id: int,
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, token_id in enumerate(input_ids):
        if int(token_id) == int(image_token_id):
            if start is None:
                start = index
        elif start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(input_ids)))
    return spans


def _point_in_box(x: float, y: float, box: Sequence[object]) -> bool:
    coerced = _coerce_xyxy_box(box)
    if coerced is None:
        return False
    x1, y1, x2, y2 = [float(value) for value in coerced]
    return x1 <= x <= x2 and y1 <= y <= y2


def build_patch_region_membership(
    *,
    visual_token_start: int,
    grid_h: int,
    grid_w: int,
    region_rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[int]]:
    membership: dict[str, list[int]] = {str(row["region_kind"]): [] for row in region_rows}
    for row in range(int(grid_h)):
        for col in range(int(grid_w)):
            token_index = int(visual_token_start) + row * int(grid_w) + col
            x = (col + 0.5) * 1000.0 / float(grid_w)
            y = (row + 0.5) * 1000.0 / float(grid_h)
            for region in region_rows:
                kind = str(region["region_kind"])
                if _point_in_box(x, y, region.get("bbox_xyxy", ())):
                    exclude = region.get("exclude_bbox_xyxy")
                    if (
                        isinstance(exclude, Sequence)
                        and not isinstance(exclude, (str, bytes))
                        and _point_in_box(x, y, exclude)
                    ):
                        continue
                    membership.setdefault(kind, []).append(token_index)
    return membership


def _processor_merge_size(processor: Any) -> int:
    image_processor = getattr(processor, "image_processor", None)
    merge_size = getattr(image_processor, "merge_size", 1)
    try:
        parsed = int(merge_size)
    except (TypeError, ValueError):
        parsed = 1
    return max(1, parsed)


def _visual_grid_from_thw(
    grid_t: int,
    grid_h: int,
    grid_w: int,
    *,
    merge_size: int,
) -> tuple[int, int, int, int]:
    if grid_h % merge_size != 0 or grid_w % merge_size != 0:
        raise RuntimeError(
            "image_grid_thw spatial dimensions are not divisible by merge_size"
        )
    visual_grid_h = grid_h // merge_size
    visual_grid_w = grid_w // merge_size
    expected_visual_tokens = int(grid_t) * visual_grid_h * visual_grid_w
    return expected_visual_tokens, int(grid_t), visual_grid_h, visual_grid_w


def aggregate_attention_for_query(
    attention: Any,
    *,
    batch_idx: int,
    query_index: int,
    layer_index: int,
    role: str,
    region_membership: Mapping[str, Sequence[int]],
    base_row: Mapping[str, Any],
) -> list[dict[str, Any]]:
    import torch

    if not isinstance(attention, torch.Tensor):
        raise TypeError("attention must be a torch.Tensor")
    if attention.ndim != 4:
        raise ValueError(
            f"expected attention shape [batch, heads, query, key], got {tuple(attention.shape)}"
        )
    rows: list[dict[str, Any]] = []
    head_count = int(attention.shape[1])
    key_count = int(attention.shape[3])
    for head in range(head_count):
        query_vector = attention[batch_idx, head, query_index].detach().float().cpu()
        denom = float(query_vector.sum().item())
        for region_kind, indices in region_membership.items():
            valid_indices = [int(index) for index in indices if 0 <= int(index) < key_count]
            mass = float(query_vector[valid_indices].sum().item()) if valid_indices else 0.0
            rows.append(
                {
                    **dict(base_row),
                    "layer": int(layer_index),
                    "head": int(head),
                    "role": role,
                    "query_index": int(query_index),
                    "region_kind": str(region_kind),
                    "region_token_count": len(valid_indices),
                    "attention_mass": mass,
                    "attention_mass_normalized": 0.0 if denom <= 0.0 else mass / denom,
                }
            )
    return rows


def _selected_case_query_roles(config: AttentionConfig) -> tuple[str, ...]:
    if config.selection.case_source == "teacher_forced_anchor":
        return ATTENTION_TEACHER_FORCED_QUERY_ROLES
    return ATTENTION_QUERY_ROLES


def _prepare_attention_pairs(
    config: AttentionConfig,
    *,
    model_handle: Any,
    selected: Sequence[Mapping[str, Any]],
    shard_index: int,
    num_shards: int,
) -> tuple[list[tuple[Mapping[str, Any], Any]], Any]:
    from src.analysis.autoreg_hidden_state_probe import filter_lane_d_examples_for_selected_cases
    from src.analysis.hard_ce_coord_logit_locality import prepare_lane_c_x1_basin_examples

    lane_c_config = build_attention_lane_c_config(config)
    if config.selection.case_source == "teacher_forced_anchor":
        examples, _ = prepare_attention_teacher_forced_lane_c_examples(
            lane_c_config,
            model_handle=model_handle,
            limit=config.selection.sample_limit,
            shard_index=shard_index,
            num_shards=num_shards,
        )
    else:
        examples, _ = prepare_lane_c_x1_basin_examples(
            lane_c_config,
            model_handle=model_handle,
            limit=config.selection.sample_limit,
            shard_index=shard_index,
            num_shards=num_shards,
        )
    return filter_lane_d_examples_for_selected_cases(examples, selected), lane_c_config


def _open_rgb(path: Path) -> Any:
    from PIL import Image

    return Image.open(path).convert("RGB")


def _forward_attention_case(
    *,
    config: AttentionConfig,
    model_handle: Any,
    selected_case: Mapping[str, Any],
    example: Any,
    candidate_rows: Sequence[Mapping[str, Any]],
    shard_index: int,
    num_shards: int,
    shard_label: str,
    atlas: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    from src.analysis.autoreg_hidden_state_probe import (
        build_lane_d_position_inventory_for_prepared_example,
    )
    from src.analysis.hard_ce_coord_logit_locality import _batch_pad_offset, _model_device

    image_token_id = model_handle.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if image_token_id is None or int(image_token_id) < 0:
        raise RuntimeError("tokenizer cannot resolve <|image_pad|>")
    inputs = model_handle.processor(
        text=[example.full_text],
        images=[_open_rgb(example.image_path)],
        return_tensors="pt",
        padding=True,
    )
    inputs = {
        key: value.to(_model_device(model_handle.model)) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }
    with torch.inference_mode():
        outputs = model_handle.model(
            **inputs,
            use_cache=False,
            output_attentions=True,
        )
    attentions = getattr(outputs, "attentions", None)
    if not isinstance(attentions, tuple) or not attentions:
        raise RuntimeError("model forward did not return attentions")
    input_ids_tensor = inputs.get("input_ids")
    if not isinstance(input_ids_tensor, torch.Tensor):
        raise RuntimeError("processor output missing input_ids")
    input_ids = input_ids_tensor[0].detach().cpu().tolist()
    visual_spans = find_visual_token_spans(input_ids, image_token_id=int(image_token_id))
    image_grid_thw = inputs.get("image_grid_thw")
    if not isinstance(image_grid_thw, torch.Tensor):
        raise RuntimeError("processor output missing image_grid_thw")
    grid_t, grid_h, grid_w = [int(value) for value in image_grid_thw[0].detach().cpu().tolist()]
    merge_size = _processor_merge_size(model_handle.processor)
    expected_visual_tokens, visual_grid_t, visual_grid_h, visual_grid_w = _visual_grid_from_thw(
        grid_t,
        grid_h,
        grid_w,
        merge_size=merge_size,
    )
    if len(visual_spans) != 1:
        raise RuntimeError(f"expected exactly one visual token span, got {len(visual_spans)}")
    visual_span = visual_spans[0]
    if visual_span[1] - visual_span[0] != expected_visual_tokens:
        raise RuntimeError("visual token span does not match image_grid_thw")
    inventory = build_lane_d_position_inventory_for_prepared_example(
        example,
        selected_case,
        model_handle.tokenizer,
        shard_index=shard_index,
        num_shards=num_shards,
        shard_label=shard_label,
    )
    pad_offset = _batch_pad_offset(
        input_ids=input_ids_tensor,
        batch_idx=0,
        expected_ids=example.full_input_ids,
    )
    base = {
        "case_id": selected_case["case_id"],
        "source_line_idx": selected_case["source_line_idx"],
        "prefix_mode": selected_case["prefix_mode"],
        "prefix_depth": selected_case["prefix_depth"],
        "prefix_quality": selected_case["prefix_quality"],
        "target_gt_idx": selected_case.get("intended_target_gt_idx"),
        "target_desc": selected_case.get("target_desc"),
        "x1_target_rank": selected_case.get("x1_target_rank"),
        "x1_top_peak_attribution": selected_case.get("x1_top_peak_attribution"),
        "scope_label": selected_case.get("scope_label", config.selection.scope_label),
        "shard_index": shard_index,
        "num_shards": num_shards,
        "shard_label": shard_label,
    }
    decision_row = {
        **base,
        "image_token_id": int(image_token_id),
        "visual_span_start": visual_span[0],
        "visual_span_end": visual_span[1],
        "grid_t": grid_t,
        "grid_h": grid_h,
        "grid_w": grid_w,
        "merge_size": merge_size,
        "visual_grid_t": visual_grid_t,
        "visual_grid_h": visual_grid_h,
        "visual_grid_w": visual_grid_w,
        "attention_layer_count": len(attentions),
        "attention_shape_0": list(attentions[0].shape),
        "inventory_role_count": len(inventory),
        "pad_offset": int(pad_offset),
        "candidate_region_count": len(candidate_rows),
        "query_roles": list(_selected_case_query_roles(config)),
    }
    if not atlas:
        return [], decision_row

    membership = build_patch_region_membership(
        visual_token_start=visual_span[0],
        grid_h=visual_grid_h,
        grid_w=visual_grid_w,
        region_rows=candidate_rows,
    )
    inv_by_role = {str(row["role"]): row for row in inventory}
    attention_rows: list[dict[str, Any]] = []
    for role in _selected_case_query_roles(config):
        inv = inv_by_role.get(role)
        if inv is None:
            continue
        hidden_index = inv.get("prediction_token_index")
        if hidden_index is None:
            hidden_index = inv.get("absolute_token_index")
        query_index = int(pad_offset) + int(hidden_index)
        for layer_index, attention in enumerate(attentions):
            attention_rows.extend(
                aggregate_attention_for_query(
                    attention,
                    batch_idx=0,
                    query_index=query_index,
                    layer_index=layer_index,
                    role=role,
                    region_membership=membership,
                    base_row=base,
                )
            )
    return attention_rows, decision_row


def materialize_attention_feasibility_shard(
    config: AttentionConfig,
    *,
    shard_index: int,
    num_shards: int,
) -> dict[str, Any]:
    from src.analysis.hard_ce_coord_logit_locality import load_model_handle

    shard_index, num_shards, shard_label = normalize_attention_shard(
        shard_index=shard_index,
        num_shards=num_shards,
    )
    assert shard_index is not None and num_shards is not None and shard_label is not None
    shard_dir = config.paths.artifact_root / "shards" / shard_label
    selected = _read_jsonl(shard_dir / "selected_cases.jsonl")[
        : config.execution.max_feasibility_cases
    ]
    candidate_rows_all = _read_jsonl(shard_dir / "candidate_region_rows.jsonl")
    candidate_by_case: dict[str, list[dict[str, Any]]] = {}
    for row in candidate_rows_all:
        candidate_by_case.setdefault(str(row["case_id"]), []).append(row)
    lane_c_config = build_attention_lane_c_config(config)
    model_handle = load_model_handle(lane_c_config)
    pairs, _ = _prepare_attention_pairs(
        config,
        model_handle=model_handle,
        selected=selected,
        shard_index=shard_index,
        num_shards=num_shards,
    )
    rows: list[dict[str, Any]] = []
    for selected_case, example in pairs:
        _, decision_row = _forward_attention_case(
            config=config,
            model_handle=model_handle,
            selected_case=selected_case,
            example=example,
            candidate_rows=candidate_by_case.get(str(selected_case["case_id"]), []),
            shard_index=shard_index,
            num_shards=num_shards,
            shard_label=shard_label,
            atlas=False,
        )
        rows.append(decision_row)
    _write_jsonl(shard_dir / "feasibility_rows.jsonl", rows)
    summary = {
        "analysis_name": "autoreg_attention_evidence_routing",
        "schema_version": 1,
        "stage": "attention_evidence_routing",
        "stages_completed": ["select_cases", "feasibility"],
        "artifact_root": str(config.paths.artifact_root),
        "shard_index": shard_index,
        "num_shards": num_shards,
        "shard_label": shard_label,
        "row_counts": {"feasibility_rows": len(rows)},
        "runtime_kind": "attention_feasibility_forward",
        "case_source": config.selection.case_source,
        "scope_label": config.selection.scope_label,
        "output_paths": {
            "feasibility_rows": str(shard_dir / "feasibility_rows.jsonl"),
            "summary": str(shard_dir / "summary.json"),
        },
    }
    _write_json(shard_dir / "summary.json", summary)
    return summary


def materialize_attention_atlas_shard(
    config: AttentionConfig,
    *,
    shard_index: int,
    num_shards: int,
) -> dict[str, Any]:
    from src.analysis.hard_ce_coord_logit_locality import load_model_handle

    shard_index, num_shards, shard_label = normalize_attention_shard(
        shard_index=shard_index,
        num_shards=num_shards,
    )
    assert shard_index is not None and num_shards is not None and shard_label is not None
    shard_dir = config.paths.artifact_root / "shards" / shard_label
    selected = _read_jsonl(shard_dir / "selected_cases.jsonl")
    candidate_rows_all = _read_jsonl(shard_dir / "candidate_region_rows.jsonl")
    candidate_by_case: dict[str, list[dict[str, Any]]] = {}
    for row in candidate_rows_all:
        candidate_by_case.setdefault(str(row["case_id"]), []).append(row)
    lane_c_config = build_attention_lane_c_config(config)
    model_handle = load_model_handle(lane_c_config)
    pairs, _ = _prepare_attention_pairs(
        config,
        model_handle=model_handle,
        selected=selected,
        shard_index=shard_index,
        num_shards=num_shards,
    )
    attention_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    for selected_case, example in pairs:
        case_attention_rows, decision_row = _forward_attention_case(
            config=config,
            model_handle=model_handle,
            selected_case=selected_case,
            example=example,
            candidate_rows=candidate_by_case.get(str(selected_case["case_id"]), []),
            shard_index=shard_index,
            num_shards=num_shards,
            shard_label=shard_label,
            atlas=True,
        )
        attention_rows.extend(case_attention_rows)
        decision_rows.append(decision_row)
    _write_jsonl(shard_dir / "attention_region_rows.jsonl", attention_rows)
    _write_jsonl(shard_dir / "decision_context_rows.jsonl", decision_rows)
    summary = {
        "analysis_name": "autoreg_attention_evidence_routing",
        "schema_version": 1,
        "stage": "attention_evidence_routing",
        "stages_completed": ["select_cases", "attention_atlas"],
        "artifact_root": str(config.paths.artifact_root),
        "shard_index": shard_index,
        "num_shards": num_shards,
        "shard_label": shard_label,
        "row_counts": {
            "selected_cases": len(selected),
            "candidate_region_rows": len(candidate_rows_all),
            "attention_region_rows": len(attention_rows),
            "decision_context_rows": len(decision_rows),
        },
        "runtime_kind": "attention_atlas_forward",
        "case_source": config.selection.case_source,
        "scope_label": config.selection.scope_label,
        "output_paths": {
            "attention_region_rows": str(shard_dir / "attention_region_rows.jsonl"),
            "decision_context_rows": str(shard_dir / "decision_context_rows.jsonl"),
            "summary": str(shard_dir / "summary.json"),
        },
    }
    _write_json(shard_dir / "summary.json", summary)
    return summary


def _jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def merge_attention_shards(root: Path, *, expected_shards: int) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    shard_labels = [attention_shard_label(index, expected_shards) for index in range(expected_shards)]
    manifest_path = root / "shards_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8") or "{}")
        if int(manifest.get("expected_shards", -1)) != int(expected_shards):
            raise ValueError(
                "shards_manifest expected_shards mismatch: "
                f"{manifest.get('expected_shards')} != {expected_shards}"
            )
        manifest_labels = list(manifest.get("shard_labels") or [])
        if manifest_labels != shard_labels:
            raise ValueError("shards_manifest shard_labels mismatch")
    else:
        manifest = {}
    row_counts: dict[str, int] = {}
    missing: list[str] = []
    shard_summaries: list[dict[str, Any]] = []
    for filename in MERGE_JSONL_FILES:
        out_path = root / filename
        with out_path.open("w", encoding="utf-8") as out:
            for label in shard_labels:
                in_path = root / "shards" / label / filename
                if not in_path.exists():
                    missing.append(str(in_path))
                    continue
                text = in_path.read_text(encoding="utf-8")
                if text and not text.endswith("\n"):
                    text += "\n"
                out.write(text)
        row_counts[filename.removesuffix(".jsonl")] = _jsonl_count(out_path)
    for label in shard_labels:
        summary_path = root / "shards" / label / "summary.json"
        if not summary_path.exists():
            missing.append(str(summary_path))
            continue
        shard_summaries.append(json.loads(summary_path.read_text(encoding="utf-8") or "{}"))
    if missing:
        raise FileNotFoundError(f"missing shard outputs: {missing}")
    summary = {
        "analysis_name": "autoreg_attention_evidence_routing",
        "schema_version": 1,
        "stage": "attention_evidence_routing",
        "artifact_root": str(root),
        "expected_shards": expected_shards,
        "merged_shards": shard_labels,
        "manifest": manifest,
        "row_counts": row_counts,
        "shard_summaries": shard_summaries,
        "duplicate_policy": "same_desc_iou_gt_0p95",
        "causal_status": "not_in_scope_for_this_plan",
        "output_paths": {
            filename.removesuffix(".jsonl"): str(root / filename)
            for filename in MERGE_JSONL_FILES
        }
        | {
            "summary": str(root / "summary.json"),
            "merge_summary": str(root / "merge_summary.json"),
            "report": str(root / "report.md"),
        },
    }
    _write_json(root / "merge_summary.json", summary)
    _write_json(root / "summary.json", summary)
    return summary


def summarize_attention_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts_by_role: Counter[str] = Counter()
    target_mass_by_role: dict[str, list[float]] = {}
    background_mass_by_role: dict[str, list[float]] = {}
    for row in rows:
        role = str(row.get("role", "unknown"))
        counts_by_role[role] += 1
        kind = str(row.get("region_kind", ""))
        mass = float(row.get("attention_mass_normalized", row.get("attention_mass", 0.0)))
        if kind == "target_gt":
            target_mass_by_role.setdefault(role, []).append(mass)
        if kind == "far_background":
            background_mass_by_role.setdefault(role, []).append(mass)
    return {
        "row_count": len(rows),
        "counts_by_role": dict(sorted(counts_by_role.items())),
        "target_gt_mass_mean_by_role": {
            role: sum(values) / len(values) for role, values in sorted(target_mass_by_role.items())
        },
        "far_background_mass_mean_by_role": {
            role: sum(values) / len(values)
            for role, values in sorted(background_mass_by_role.items())
        },
    }


def write_attention_report(root: Path) -> Path:
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    attention_rows_path = root / "attention_region_rows.jsonl"
    attention_summary = (
        summarize_attention_rows(_read_jsonl(attention_rows_path))
        if attention_rows_path.exists()
        else {"row_count": 0}
    )
    report = root / "report.md"
    report.write_text(
        "\n".join(
            [
                "# Attention Evidence Routing Report",
                "",
                "## Scope",
                "",
                "- checkpoint: checkpoint-3664",
                "- evidence_scope: observational_attention_only",
                "- duplicate_policy: same-desc IoU > 0.95",
                "- causal_status: not_in_scope_for_this_plan",
                "",
                "## Row Counts",
                "",
                "```json",
                json.dumps(summary.get("row_counts", {}), sort_keys=True, indent=2),
                "```",
                "",
                "## Attention Summary",
                "",
                "```json",
                json.dumps(attention_summary, sort_keys=True, indent=2),
                "```",
                "",
                "## Interpretation Bounds",
                "",
                "- Attention rows are mechanism candidates, not causal proof.",
                "- Forced continuation is not used as evidence of solved recall.",
                "- Other extra predictions are soft/neutral under partial COCO labels.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return report
