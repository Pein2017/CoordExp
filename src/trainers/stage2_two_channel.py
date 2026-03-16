import contextlib
import json
import math
import os
import time
import logging
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, ClassVar, Deque, Dict, List, Literal, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch

from swift.llm import MaxLengthError
from swift.trainers.rlhf_trainer.utils import replace_assistant_response_with_ids

from src.common.semantic_desc import normalize_desc
from src.common.object_field_order import build_object_payload
from src.utils.assistant_json import dumps_coordjson

from .stage2_rollout_aligned import RolloutMatchingSFTTrainer
from .rollout_matching.contracts import GTObject
from .rollout_matching.matching import (
    associate_one_to_one_max_iou,
    hungarian_match_maskiou,
)
from .rollout_matching.parsing import (
    decode_pieces,
    find_desc_value_char_spans,
    find_desc_value_token_positions,
    parse_rollout_for_matching,
    points_from_coord_tokens,
    serialize_append_fragment,
)
from .rollout_matching.telemetry import PendingTrainRolloutLog
from .monitoring.loss_gradient_monitor import (
    build_stage2_two_channel_coord_monitor_terms,
    get_loss_gradient_monitor,
)
from ..common.geometry.coord_utils import decode_coord

from .stage2_two_channel.executors import Stage2ABChannelExecutorsMixin
from .stage2_two_channel.scheduler import Stage2ABSchedulerMixin
from .teacher_forcing.contracts import TeacherForcingContext, PipelineModuleSpec
from .teacher_forcing.forwards import (
    assert_unsliced_logits,
    prepare_forward_inputs,
    run_no_cache_forward,
)
from .teacher_forcing.geometry import (
    bbox_smoothl1_ciou_loss as _tf_bbox_smoothl1_ciou_loss,
    expectation_decode_coords as _tf_expectation_decode_coords,
)
from .teacher_forcing.objective_pipeline import run_teacher_forcing_pipeline
from .teacher_forcing.rollout_masks import build_rollout_subset_masks
from .teacher_forcing.rollout_meta import (
    bbox_groups_from_token_ids as _tf_bbox_groups_from_token_ids,
    matched_prefix_structure_positions as _tf_matched_prefix_structure_positions,
    tail_closure_positions as _tf_tail_closure_positions,
)
from .teacher_forcing.token_types import build_token_type_masks


logger = logging.getLogger(__name__)


@dataclass
class _PendingStage2Log:
    """Accumulate Stage-2 AB logs across micro-batches for one optimizer step.

    Stage2-AB step-budgeted packing can run multiple packed forwards/backwards per
    optimizer step. To keep loss component telemetry interpretable, we support a
    segment-weighted mean when callers provide `stage2/_log_weight` (typically the
    number of segments/samples in the packed forward).
    """

    n_micro: int = 0
    weight_sum: float = 0.0
    gradmon_weight_sum: float = 0.0
    sums: Dict[str, float] = field(default_factory=dict)

    _COUNTER_SUFFIXES: ClassVar[tuple[str, ...]] = (
        "_total",
        "_count",
        "_sum",
        "_num",
        "_den",
    )

    @staticmethod
    def _is_counter_like_key(key: str) -> bool:
        if key in {"dup/max_desc_count"}:
            return False
        if key in {
            "stage2/raw_rollouts",
            "stage2/invalid_rollout",
            "stage2_ab/channel_b/invalid_rollout",
            "stage2/drop_poly",
            "stage2/drop_unknown",
            "stage2/drop_bbox_invalid",
            "rollout/parse_truncated",
            "rollout/_parse_truncated_num",
            "rollout/_parse_truncated_den",
            "stage2_ab/channel_b/closure_supervision/N_drop",
        }:
            return True

        if key.startswith("stage2_ab/channel_b/strict_drop/reason/"):
            return True
        if key.startswith("stage2_ab/") and "/N_" in key:
            return True

        return key.endswith(_PendingStage2Log._COUNTER_SUFFIXES)

    @staticmethod
    def _is_mean_like_key(key: str) -> bool:
        return (
            key.startswith("loss/")
            or key.startswith("gradmon/")
            or key in {"dup/max_desc_count", "dup/saturation_rate"}
            or key.startswith("stage2/channel_")
            or key.startswith("rollout/")
            or key.startswith("coord_diag/")
            or key == "stage2_ab/b_ratio_realized"
            or (key.startswith("stage2_ab/") and "/is_" in key)
        )

    def add(self, metrics: Mapping[str, float]) -> None:
        self.n_micro += 1

        weight = 1.0
        if isinstance(metrics, Mapping) and "stage2/_log_weight" in metrics:
            try:
                weight = float(metrics.get("stage2/_log_weight") or 0.0)
            except (TypeError, ValueError):
                weight = 1.0

        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(
                "stage2/_log_weight must be a positive finite float; "
                f"got {weight!r}"
            )

        self.weight_sum += float(weight)
        saw_gradmon = False

        for k, v in metrics.items():
            ks = str(k)
            if ks == "stage2/_log_weight":
                continue

            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue

            if ks == "rollout/parse_truncated_rate":
                # Always derived from numerator/denominator when rollout counters exist.
                continue

            if self._is_counter_like_key(ks):
                # Counters: sum across micro-packs (never weighted).
                self.sums[ks] = float(self.sums.get(ks, 0.0)) + float(fv)
                continue

            if self._is_mean_like_key(ks):
                if ks.startswith("gradmon/"):
                    saw_gradmon = True
                # Means: segment-weighted when stage2/_log_weight is provided.
                self.sums[ks] = float(self.sums.get(ks, 0.0)) + float(fv) * float(weight)
            else:
                # Unknown/non-contract keys default to sum semantics.
                self.sums[ks] = float(self.sums.get(ks, 0.0)) + float(fv)

        if saw_gradmon:
            self.gradmon_weight_sum += float(weight)

    def finalize(self, *, drop_internal: bool = True) -> Dict[str, float]:
        if self.n_micro <= 0:
            return {}

        denom = float(self.weight_sum) if float(self.weight_sum) > 0.0 else float(self.n_micro)

        out: Dict[str, float] = {}
        for k, v in self.sums.items():
            if k == "rollout/parse_truncated_rate":
                # Always derived from numerator/denominator when rollout counters exist.
                continue

            if self._is_counter_like_key(k):
                out[k] = float(v)
                continue

            if self._is_mean_like_key(k):
                if k.startswith("gradmon/"):
                    gradmon_denom = (
                        float(self.gradmon_weight_sum)
                        if float(self.gradmon_weight_sum) > 0.0
                        else float(denom)
                    )
                    out[k] = float(v) / float(gradmon_denom)
                else:
                    out[k] = float(v) / float(denom)
            else:
                out[k] = float(v)

        # Internal: used for DDP-weighted mean reduction.
        out["stage2/_log_weight_total"] = float(denom)
        if float(self.gradmon_weight_sum) > 0.0:
            out["gradmon/_log_weight_total"] = float(self.gradmon_weight_sum)

        trunc_num_key = "rollout/_parse_truncated_num"
        trunc_den_key = "rollout/_parse_truncated_den"
        has_trunc_inputs = any(
            k in out
            for k in {
                trunc_num_key,
                trunc_den_key,
                "rollout/parse_truncated",
                "stage2/raw_rollouts",
            }
        )
        if has_trunc_inputs:
            trunc_num = float(
                out.get(
                    trunc_num_key,
                    out.get("rollout/parse_truncated", 0.0),
                )
            )
            trunc_den = float(
                out.get(
                    trunc_den_key,
                    out.get("stage2/raw_rollouts", 0.0),
                )
            )
            out["rollout/parse_truncated_rate"] = (
                float(trunc_num / trunc_den) if trunc_den > 0.0 else 0.0
            )

        if drop_internal:
            out.pop(trunc_num_key, None)
            out.pop(trunc_den_key, None)
            out.pop("stage2/_log_weight_total", None)
            out.pop("gradmon/_log_weight_total", None)

        return out


def _stage2_snapshot_key(metric_key: str) -> str | None:
    key = str(metric_key)

    if (
        key.startswith(("loss/A1_", "loss/A2_"))
        or key.startswith(("coord_diag/A1/", "coord_diag/A2/"))
        or key.startswith("time/channel_a_")
        or key == "stage2/channel_a"
    ):
        return key

    if (
        key.startswith("loss/B_")
        or key.startswith("coord_diag/B/")
        or key.startswith("stage2_ab/channel_b/")
        or key.startswith("dup/")
        or key.startswith("train/triage/")
        or key.startswith("diag/dead_anchor/")
        or key.startswith("rollout/")
        or key.startswith("time/rollout_")
        or key
        in {
            "stage2/channel_b",
            "stage2/raw_rollouts",
            "stage2/invalid_rollout",
            "stage2/drop_poly",
            "stage2/drop_unknown",
            "stage2/drop_bbox_invalid",
        }
    ):
        return key

    return None


def _merge_stage2_metric_snapshots(
    snapshots: MutableMapping[str, float],
    metrics: Mapping[str, Any],
) -> Dict[str, float]:
    for key_raw, value in metrics.items():
        snapshot_key = _stage2_snapshot_key(str(key_raw))
        if snapshot_key is None:
            continue
        try:
            snapshots[snapshot_key] = float(value)
        except (TypeError, ValueError):
            continue
    return {
        str(key): float(value)
        for key, value in sorted(snapshots.items(), key=lambda item: item[0])
    }




def _expectation_decode_coords(
    *,
    coord_logits: torch.Tensor,
    temperature: float,
    mode: str = "exp",
) -> torch.Tensor:
    return _tf_expectation_decode_coords(
        coord_logits=coord_logits,
        temperature=float(temperature),
        mode=str(mode or "exp"),
    )


def _bbox_smoothl1_ciou_loss(
    *,
    pred_xyxy: torch.Tensor,
    gt_xyxy: torch.Tensor,
    eps: float = 1e-7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    out = _tf_bbox_smoothl1_ciou_loss(
        pred_xyxy=pred_xyxy,
        gt_xyxy=gt_xyxy,
        eps=float(eps),
    )
    return out.smoothl1, out.ciou


def _find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> int:
    """Return start index of the last occurrence of needle in haystack."""
    if not needle:
        raise ValueError("needle is empty")
    if len(needle) > len(haystack):
        raise ValueError("needle longer than haystack")
    last = -1
    n = len(needle)
    for i in range(0, len(haystack) - n + 1):
        if list(haystack[i : i + n]) == list(needle):
            last = i
    if last < 0:
        raise ValueError(
            "unable to locate assistant token ids inside encoded input_ids"
        )
    return int(last)


def _percentile(xs: Sequence[float], q: float) -> float:
    """Small percentile helper for rollout telemetry without adding numpy here."""

    vals = sorted(float(x) for x in xs)
    if not vals:
        return 0.0
    if len(vals) == 1:
        return float(vals[0])

    rank = (float(len(vals)) - 1.0) * (float(q) / 100.0)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(vals[lo])
    frac = float(rank - float(lo))
    return float((1.0 - frac) * float(vals[lo]) + frac * float(vals[hi]))


def _coerce_bbox_bins(values: Any) -> Optional[List[int]]:
    if not isinstance(values, Sequence):
        return None
    if len(values) != 4:
        return None
    out: List[int] = []
    for v in values:
        k = decode_coord(v)
        if k is None:
            return None
        out.append(int(k))
    x1, y1, x2, y2 = out
    if x2 < x1 or y2 < y1:
        return None
    return out


def _extract_gt_bboxonly(sample: Mapping[str, Any]) -> List[GTObject]:
    payload = sample.get("assistant_payload")
    if not isinstance(payload, Mapping):
        raise ValueError("stage2-ab requires assistant_payload in each sample")

    objects = payload.get("objects")
    if not isinstance(objects, Sequence):
        raise ValueError("assistant_payload must contain top-level 'objects' list")

    objs: List[GTObject] = []
    for idx, entry in enumerate(objects):
        if not isinstance(entry, Mapping):
            raise ValueError(f"assistant_payload.objects[{int(idx)}] must be a mapping")

        # Enforce bbox-only v1 on GT: exactly one geometry field, and it must be bbox_2d.
        # Any other geometry key (including poly) must fail fast.
        geom_keys = sorted(
            {
                str(k)
                for k, v in entry.items()
                if str(k) != "desc" and v is not None
            }
        )
        if geom_keys != ["bbox_2d"]:
            raise ValueError(
                "bbox-only v1 requires each GT object to contain exactly one geometry field 'bbox_2d' "
                f"(no poly/other geometry keys); got geometry keys={geom_keys} for objects[{int(idx)}]"
            )

        pts = _coerce_bbox_bins(entry.get("bbox_2d"))
        if pts is None:
            raise ValueError(
                f"invalid bbox_2d for objects[{int(idx)}]; expected 4 bins in [0,999] and ordered xyxy"
            )

        desc = entry.get("desc", "")
        objs.append(
            GTObject(
                index=int(idx),
                geom_type="bbox_2d",
                points_norm1000=list(pts),
                desc=str(desc) if desc is not None else "",
            )
        )

    if not objs:
        raise ValueError("no valid GT objects found in assistant_payload")
    return objs


def _sample_monitor_images(sample: Mapping[str, Any]) -> List[str]:
    images_raw = sample.get("images")
    images: List[str] = []
    if isinstance(images_raw, Sequence) and not isinstance(images_raw, (str, bytes)):
        for value in images_raw:
            if isinstance(value, str) and value.strip():
                images.append(str(value))
    if images:
        return images

    image_one = sample.get("image")
    if isinstance(image_one, str) and image_one.strip():
        return [str(image_one)]
    return []


def _sample_monitor_image_id(sample: Mapping[str, Any]) -> Any:
    for key in ("image_id", "sample_id", "base_idx"):
        value = sample.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _serialize_monitor_object(obj: GTObject) -> Dict[str, Any]:
    return {
        "index": int(obj.index),
        "geom_type": str(obj.geom_type),
        "points_norm1000": [int(x) for x in obj.points_norm1000],
        "desc": str(obj.desc),
    }


def _build_stage2_train_monitor_record(
    *,
    sample: Mapping[str, Any],
    gts: Sequence[GTObject],
    preds: Sequence[GTObject],
    object_field_order: str,
) -> Dict[str, Any]:
    images = _sample_monitor_images(sample)

    width = sample.get("width")
    height = sample.get("height")
    try:
        width = int(width) if width is not None else None
    except (TypeError, ValueError):
        width = None
    try:
        height = int(height) if height is not None else None
    except (TypeError, ValueError):
        height = None
    if width is None:
        width = 1000
    if height is None:
        height = 1000

    def _geometry_key(value: Any) -> str:
        key = str(value or "").strip().lower()
        if key == "bbox":
            return "bbox_2d"
        return key

    gt_payload = [
        build_object_payload(
            desc=str(obj.desc),
            geometry_key=_geometry_key(obj.geom_type),
            geometry_value=[int(x) for x in obj.points_norm1000],
            object_field_order=object_field_order,
        )
        for obj in gts
    ]
    pred_payload = [
        build_object_payload(
            desc=str(obj.desc),
            geometry_key=_geometry_key(obj.geom_type),
            geometry_value=[int(x) for x in obj.points_norm1000],
            object_field_order=object_field_order,
        )
        for obj in preds
    ]

    return {
        "image_id": _sample_monitor_image_id(sample),
        "images": images,
        "width": int(width),
        "height": int(height),
        "gt": gt_payload,
        "pred": pred_payload,
    }


def _build_teacher_forced_payload(
    *, gt_objects: Sequence[GTObject], object_field_order: str
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"objects": []}
    objects = payload["objects"]
    if not isinstance(objects, list):
        raise RuntimeError("internal error: objects payload is not a list")

    for obj in gt_objects:
        if obj.geom_type == "bbox_2d":
            geometry_key = "bbox_2d"
            geometry_value = [f"<|coord_{int(v)}|>" for v in obj.points_norm1000]
        elif obj.geom_type == "poly":
            points = obj.points_norm1000
            geometry_key = "poly"
            geometry_value = [f"<|coord_{int(v)}|>" for v in points]
        else:
            raise ValueError(f"unsupported geometry type for stage2 payload: {obj.geom_type!r}")

        objects.append(
            build_object_payload(
                desc=str(obj.desc),
                geometry_key=geometry_key,
                geometry_value=geometry_value,
                object_field_order=object_field_order,
            )
        )
    return payload


def _stage2_ab_tail_closure_positions(
    *,
    tokenizer: Any,
    assistant_span_ids: Sequence[int],
    prefix_len: int,
) -> List[int]:
    return _tf_tail_closure_positions(
        tokenizer=tokenizer,
        assistant_span_ids=assistant_span_ids,
        prefix_len=int(prefix_len),
    )


def _bbox_groups_from_token_ids(
    *,
    token_ids: Sequence[int],
    coord_id_set: set[int],
    gt_objs: Sequence[GTObject],
) -> List[List[int]]:
    return _tf_bbox_groups_from_token_ids(
        token_ids=token_ids,
        coord_id_set=coord_id_set,
        gt_objs=gt_objs,
    )


def _matched_prefix_structure_positions(
    *,
    tokenizer: Any,
    prefix_token_ids: Sequence[int],
    prefix_text: str,
    matched_pred_objects: Sequence[Any],
) -> List[int]:
    return _tf_matched_prefix_structure_positions(
        tokenizer=tokenizer,
        prefix_token_ids=prefix_token_ids,
        prefix_text=prefix_text,
        matched_pred_objects=matched_pred_objects,
    )


@dataclass(frozen=True)
class _ValueSpanObject:
    value_span: Tuple[int, int]


@dataclass(frozen=True)
class _CanonicalPrefixData:
    prefix_text: str
    prefix_token_ids: List[int]
    boundary_prefix_texts: List[str]
    object_value_spans: List[Tuple[int, int]]


def _bbox_iou_norm1000_xyxy(box_a: Sequence[int], box_b: Sequence[int]) -> float:
    if len(box_a) != 4 or len(box_b) != 4:
        return 0.0
    try:
        ax1, ay1, ax2, ay2 = [int(v) for v in box_a]
        bx1, by1, bx2, by2 = [int(v) for v in box_b]
    except (TypeError, ValueError):
        return 0.0

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = float(inter_w * inter_h)
    area_a = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
    area_b = float(max(0, bx2 - bx1) * max(0, by2 - by1))
    union = float(area_a + area_b - inter)
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def _serialize_gt_object_entry(
    *,
    obj: GTObject,
    object_field_order: str,
) -> str:
    if str(obj.geom_type) != "bbox_2d":
        raise ValueError(
            f"Channel-B clean-prefix v1 only supports bbox_2d objects; got {obj.geom_type!r}"
        )
    if len(obj.points_norm1000) != 4:
        raise ValueError(
            "Channel-B clean-prefix v1 requires bbox_2d objects with four coord bins"
        )

    payload = build_object_payload(
        desc=str(obj.desc),
        geometry_key="bbox_2d",
        geometry_value=[f"<|coord_{int(v)}|>" for v in obj.points_norm1000],
        object_field_order=object_field_order,
    )
    return dumps_coordjson(payload)


def _build_canonical_prefix_text_data(
    *,
    objects: Sequence[GTObject],
    object_field_order: str,
) -> Tuple[str, List[str], List[Tuple[int, int]]]:
    empty_container = dumps_coordjson({"objects": []})
    if not str(empty_container).endswith("]}"):
        raise ValueError(
            "unexpected canonical CoordJSON container rendering for empty objects list"
        )

    prefix_text = str(empty_container[:-2])
    boundary_prefix_texts: List[str] = [str(prefix_text)]
    object_value_spans: List[Tuple[int, int]] = []

    for obj in objects:
        entry_text = _serialize_gt_object_entry(
            obj=obj,
            object_field_order=object_field_order,
        )
        if not prefix_text.endswith("["):
            prefix_text = prefix_text + ", "
        start = int(len(prefix_text))
        prefix_text = prefix_text + str(entry_text)
        object_value_spans.append((start, int(len(prefix_text))))
        boundary_prefix_texts.append(str(prefix_text))

    return prefix_text, boundary_prefix_texts, object_value_spans


def _build_canonical_prefix_data(
    *,
    tokenizer: Any,
    objects: Sequence[GTObject],
    object_field_order: str,
) -> _CanonicalPrefixData:
    prefix_text, boundary_prefix_texts, object_value_spans = (
        _build_canonical_prefix_text_data(
            objects=objects,
            object_field_order=object_field_order,
        )
    )
    prefix_token_ids = [
        int(t) for t in tokenizer.encode(prefix_text, add_special_tokens=False)
    ]
    return _CanonicalPrefixData(
        prefix_text=str(prefix_text),
        prefix_token_ids=prefix_token_ids,
        boundary_prefix_texts=[str(t) for t in boundary_prefix_texts],
        object_value_spans=[tuple(span) for span in object_value_spans],
    )


def _build_canonical_closed_container_text(
    *,
    objects: Sequence[GTObject],
    object_field_order: str,
) -> str:
    prefix_text, _boundary_prefix_texts, _value_spans = _build_canonical_prefix_text_data(
        objects=objects,
        object_field_order=object_field_order,
    )
    return str(prefix_text) + "]}"


def _token_piece_char_spans(
    *,
    tokenizer: Any,
    token_ids: Sequence[int],
 ) -> List[Tuple[int, int]]:
    pieces = decode_pieces(tokenizer, token_ids)
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for piece in pieces:
        start = int(cursor)
        cursor += int(len(piece))
        spans.append((start, int(cursor)))
    return spans


def _first_safe_token_index_from_char_cut(
    *,
    tokenizer: Any,
    token_ids: Sequence[int],
    cut_char_pos: int,
) -> int:
    if cut_char_pos <= 0 or not token_ids:
        return 0

    # Keep any token that starts before the clean boundary in the prefix/context.
    # This avoids retokenizing token-internal char cuts into synthetic positions that
    # do not exist in the actual teacher-forced target tokenization.
    for idx, (start, _end) in enumerate(
        _token_piece_char_spans(tokenizer=tokenizer, token_ids=token_ids)
    ):
        if int(start) >= int(cut_char_pos):
            return int(idx)
    return int(len(token_ids))


def _compute_duplicate_diagnostics(
    parsed_bbox_objects_raw: Sequence[GTObject],
) -> Dict[str, float]:
    norm_descs = [normalize_desc(obj.desc) for obj in parsed_bbox_objects_raw]
    max_desc_count = 0
    if norm_descs:
        counts = Counter(norm_descs)
        max_desc_count = int(max(counts.values()))

    saturated = 0
    for obj in parsed_bbox_objects_raw:
        if any(int(v) in {0, 999} for v in obj.points_norm1000):
            saturated += 1

    near_same_desc = 0
    near_any_desc = 0
    for i, obj_i in enumerate(parsed_bbox_objects_raw):
        for j in range(int(i + 1), int(len(parsed_bbox_objects_raw))):
            obj_j = parsed_bbox_objects_raw[j]
            iou = _bbox_iou_norm1000_xyxy(obj_i.points_norm1000, obj_j.points_norm1000)
            if iou < 0.90:
                continue
            near_any_desc += 1
            if norm_descs[i] == norm_descs[j]:
                near_same_desc += 1

    n_raw = int(len(parsed_bbox_objects_raw))
    saturation_rate = (float(saturated) / float(n_raw)) if n_raw > 0 else 0.0
    return {
        "dup/max_desc_count": float(max_desc_count),
        "dup/saturation_rate": float(saturation_rate),
        "dup/near_iou90_pairs_same_desc_count": float(near_same_desc),
        "dup/near_iou90_pairs_any_desc_count": float(near_any_desc),
    }


def _sequential_dedup_bbox_objects(
    *,
    parsed_bbox_objects_raw: Sequence[GTObject],
    duplicate_iou_threshold: float,
) -> Tuple[List[GTObject], Dict[int, List[GTObject]]]:
    accepted_objects_clean: List[GTObject] = []
    accepted_norm_descs: List[str] = []
    duplicate_bursts_by_boundary: Dict[int, List[GTObject]] = {}

    for obj in parsed_bbox_objects_raw:
        boundary = int(len(accepted_objects_clean))
        obj_norm_desc = normalize_desc(obj.desc)
        is_duplicate = False

        for accepted, accepted_norm_desc in zip(
            accepted_objects_clean,
            accepted_norm_descs,
        ):
            if obj_norm_desc != accepted_norm_desc:
                continue
            if (
                _bbox_iou_norm1000_xyxy(
                    obj.points_norm1000,
                    accepted.points_norm1000,
                )
                < float(duplicate_iou_threshold)
            ):
                continue
            duplicate_bursts_by_boundary.setdefault(boundary, []).append(obj)
            is_duplicate = True
            break

        if is_duplicate:
            continue

        accepted_objects_clean.append(obj)
        accepted_norm_descs.append(obj_norm_desc)

    return accepted_objects_clean, duplicate_bursts_by_boundary


def _build_dead_anchor_suppression_targets(
    *,
    tokenizer: Any,
    y_train_ids: Sequence[int],
    clean_target_text: str,
    accepted_objects_clean: Sequence[GTObject],
    fn_objects: Sequence[GTObject],
    duplicate_bursts_by_boundary: Mapping[int, Sequence[GTObject]],
    boundary_prefix_texts: Sequence[str],
    object_field_order: str,
) -> Tuple[List[Dict[str, int]], int, int]:
    targets_by_boundary_token: Dict[Tuple[int, int], Dict[str, int]] = {}
    skipped_no_divergence = 0

    y_train_ids_list = [int(t) for t in y_train_ids]
    clean_target_text_s = str(clean_target_text)

    for boundary, duplicates in sorted(duplicate_bursts_by_boundary.items()):
        boundary_i = int(boundary)
        if boundary_i < 0 or boundary_i >= len(boundary_prefix_texts):
            raise ValueError(
                f"duplicate burst boundary is outside clean-prefix range: {boundary_i}"
            )

        boundary_prefix_text = str(boundary_prefix_texts[boundary_i])
        if not clean_target_text_s.startswith(boundary_prefix_text):
            raise ValueError(
                "clean teacher-forced target does not share the declared boundary prefix"
            )

        boundary_char_pos = int(len(boundary_prefix_text))
        clean_boundary_token_idx = _first_safe_token_index_from_char_cut(
            tokenizer=tokenizer,
            token_ids=y_train_ids_list,
            cut_char_pos=boundary_char_pos,
        )

        for dup in duplicates:
            duplicate_target_text = _build_canonical_closed_container_text(
                objects=(
                    list(accepted_objects_clean[:boundary_i])
                    + [dup]
                    + list(accepted_objects_clean[boundary_i:])
                    + list(fn_objects)
                ),
                object_field_order=object_field_order,
            )
            if not duplicate_target_text.startswith(boundary_prefix_text):
                raise ValueError(
                    "duplicate continuation does not preserve the declared clean boundary prefix"
                )

            duplicate_target_ids = [
                int(t)
                for t in tokenizer.encode(
                    duplicate_target_text,
                    add_special_tokens=False,
                )
            ]
            duplicate_boundary_token_idx = _first_safe_token_index_from_char_cut(
                tokenizer=tokenizer,
                token_ids=duplicate_target_ids,
                cut_char_pos=boundary_char_pos,
            )

            clean_pos = int(clean_boundary_token_idx)
            duplicate_pos = int(duplicate_boundary_token_idx)
            while (
                clean_pos < len(y_train_ids_list)
                and duplicate_pos < len(duplicate_target_ids)
                and y_train_ids_list[clean_pos] == duplicate_target_ids[duplicate_pos]
            ):
                clean_pos += 1
                duplicate_pos += 1

            if clean_pos >= len(y_train_ids_list) or duplicate_pos >= len(
                duplicate_target_ids
            ):
                skipped_no_divergence += 1
                continue

            rel_pos = int(clean_pos)
            bad_token_id = int(duplicate_target_ids[duplicate_pos])
            candidate = {
                "boundary": int(boundary_i),
                "rel_pos": int(rel_pos),
                "token_id": int(bad_token_id),
            }
            key = (int(boundary_i), int(bad_token_id))
            existing = targets_by_boundary_token.get(key)
            if existing is None or int(candidate["rel_pos"]) < int(existing["rel_pos"]):
                targets_by_boundary_token[key] = candidate

    targets = sorted(
        targets_by_boundary_token.values(),
        key=lambda item: (int(item["boundary"]), int(item["rel_pos"]), int(item["token_id"])),
    )
    dead_anchor_suppression_boundary_count = len({int(item["boundary"]) for item in targets})
    return targets, int(dead_anchor_suppression_boundary_count), int(skipped_no_divergence)


def _desc_tail_positions_and_weights(
    *,
    tokenizer: Any,
    token_ids: Sequence[int],
    object_weights: Sequence[float],
) -> Tuple[List[int], List[float]]:
    ids = [int(t) for t in token_ids]
    if not ids:
        return [], []

    pieces = decode_pieces(tokenizer, ids)
    text = "".join(pieces)
    desc_spans = find_desc_value_char_spans(text)
    if not desc_spans:
        return [], []
    if len(desc_spans) != len(object_weights):
        raise ValueError(
            "Channel-B FN desc spans do not align with fn_object_weights: "
            f"spans={len(desc_spans)} weights={len(object_weights)}"
        )

    token_spans = _token_piece_char_spans(tokenizer=tokenizer, token_ids=ids)
    positions: List[int] = []
    weights: List[float] = []
    for (start_char, end_char), weight in zip(desc_spans, object_weights):
        for token_i, (token_start, token_end) in enumerate(token_spans):
            if int(token_start) < int(end_char) and int(token_end) > int(start_char):
                positions.append(int(token_i))
                weights.append(float(weight))

    return positions, weights


class Stage2ABTrainingTrainer(
    Stage2ABSchedulerMixin,
    Stage2ABChannelExecutorsMixin,
    RolloutMatchingSFTTrainer,
):
    """Stage-2 AB trainer: Channel-A iterative soft self-context + Channel-B rollout matching.

    This is bbox-only v1.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Stage2-AB step-budgeted packing can execute a variable number of packed
        # forwards/backwards per optimizer step. All loss/metrics should be locally
        # normalized, with global averaging performed later at the log/aggregation
        # boundary (one reduction per optimizer step).
        #
        # In particular, `average_tokens_across_devices=True` may trigger distributed
        # collectives during loss computation (e.g. coord_soft_ce_w1 denom reductions),
        # which can deadlock under DDP when per-rank pack counts differ.
        args_obj = getattr(self, "args", None)
        if args_obj is not None and hasattr(args_obj, "average_tokens_across_devices"):
            try:
                prev_avg = bool(getattr(args_obj, "average_tokens_across_devices", False))
            except (TypeError, ValueError):
                prev_avg = False
            if bool(prev_avg):
                try:
                    setattr(args_obj, "average_tokens_across_devices", False)
                except (AttributeError, TypeError, ValueError):
                    pass
                else:
                    logger.warning(
                        "Stage2-AB: forcing args.average_tokens_across_devices=false by default; "
                        "loss/metrics are locally normalized and globally reduced at the step boundary."
                    )

        # Disable per-forward dataset-metric key sync collectives; Stage2-AB performs
        # key union/reduction via pending-log aggregation.
        setattr(self, "_coordexp_disable_dataset_metric_key_sync", True)

        self._stage2_pending_train_logs: Dict[int, _PendingStage2Log] = {}
        self._stage2_metric_snapshots: Dict[str, float] = {}
        self._stage2_channel_override: Optional[str] = None

        # Keep per-channel packing buffers so mixed schedules never pack A/B segments together.
        # (Packing uses a shared carry buffer in RolloutMatchingSFTTrainer.)
        self._stage2_post_rollout_segments: Dict[
            str, List[Tuple[Dict[str, Any], Dict[str, Any], int]]
        ] = {"A": [], "B": []}

        # Channel-B step-budgeted mode state: accumulate raw samples across micro-steps
        # and execute rollout+packing+learning only on the final micro-step.
        self._stage2_b_step_gs: Optional[int] = None
        self._stage2_b_step_micro: int = 0
        self._stage2_b_step_raw: List[Mapping[str, Any]] = []

        # Channel-A step-budgeted packing state: accumulate raw samples across micro-steps
        # and execute packing+learning only on the final micro-step of the accumulation window.
        self._stage2_a_step_gs: Optional[int] = None
        self._stage2_a_step_micro: int = 0
        self._stage2_a_step_raw: List[Mapping[str, Any]] = []

        # Rolling realized B-step ratio diagnostics (optimizer-step granularity).
        self._stage2_ab_realized_last_gs: Optional[int] = None
        self._stage2_ab_realized_recent: Deque[int] = deque(maxlen=200)
        self._stage2_train_monitor_pending_gs: Optional[int] = None
        self._stage2_train_monitor_candidates: List[Dict[str, Any]] = []
        self._stage2_train_monitor_b_step_count: int = 0
        self._stage2_train_monitor_dump_last_step: Optional[int] = None
        self._stage2_train_monitor_dump_count: int = 0
        self._stage2_train_monitor_dump_written_step: Optional[int] = None

    def _merge_rollout_matching_batch_metrics(
        self, batch: MutableMapping[str, Any], metrics: Mapping[str, Any]
    ) -> None:
        """Merge rollout-matching batch metrics onto an existing batch.

        Some Stage2-AB code paths attach base rollout/decode metadata during batch
        preparation and then add additional telemetry later.
        Treat `_rollout_matching_batch_metrics` as merge-only to avoid accidental
        overwrites.
        """
        if not isinstance(batch, MutableMapping):
            raise TypeError("batch must be a MutableMapping")
        if not isinstance(metrics, Mapping):
            raise TypeError("metrics must be a Mapping")

        existing = batch.get("_rollout_matching_batch_metrics")
        out: Dict[str, Any] = dict(existing) if isinstance(existing, Mapping) else {}
        for k, v in metrics.items():
            out[str(k)] = v
        batch["_rollout_matching_batch_metrics"] = out

    def _stage2_reset_train_monitor_dump(self, *, global_step: int) -> None:
        gs = int(global_step)
        pending_gs = getattr(self, "_stage2_train_monitor_pending_gs", None)
        if pending_gs is None or int(pending_gs) != gs:
            b_step_count = int(
                getattr(self, "_stage2_train_monitor_b_step_count", 0) or 0
            )
            self._stage2_train_monitor_b_step_count = b_step_count + 1
        self._stage2_train_monitor_pending_gs = gs
        self._stage2_train_monitor_candidates = []

    def _stage2_train_monitor_step_allowed(self, *, global_step: int) -> bool:
        cfg = self._train_monitor_dump_cfg()
        if not bool(cfg.get("enabled", False)):
            return False
        if (
            bool(cfg.get("only_world_process_zero", True))
            and not self._is_main_process()
        ):
            return False

        max_events = int(cfg.get("max_events", 20) or 0)
        train_dump_count = int(
            getattr(self, "_stage2_train_monitor_dump_count", 0) or 0
        )
        if max_events > 0 and train_dump_count >= max_events:
            return False

        gs = int(global_step)
        last_step = getattr(self, "_stage2_train_monitor_dump_last_step", None)
        if last_step is not None and int(last_step) == gs:
            return False

        args_obj = getattr(self, "args", None)
        dump_first = bool(
            cfg.get(
                "dump_first_step",
                bool(getattr(args_obj, "logging_first_step", False)),
            )
        )

        every_channel_b = cfg.get("every_channel_b_steps", None)
        if every_channel_b is not None:
            every_channel_b = max(1, int(every_channel_b))
            b_step_count = int(
                getattr(self, "_stage2_train_monitor_b_step_count", 0) or 0
            )
            if b_step_count <= 0:
                return False
            if dump_first and b_step_count == 1:
                return True
            return (b_step_count % every_channel_b) == 0

        every = cfg.get("every_steps", None)
        if every is None:
            every = int(getattr(args_obj, "logging_steps", 1) or 1)
        every = max(1, int(every))

        if gs == 0 and not dump_first:
            return False
        if gs % every != 0:
            return False
        return True

    @staticmethod
    def _stage2_train_monitor_sort_key(
        sample: Mapping[str, Any],
    ) -> Tuple[int, int, int, int, float, float, int, int]:
        duplication = sample.get("duplication")
        if not isinstance(duplication, Mapping):
            duplication = {}
        stats = sample.get("stats")
        if not isinstance(stats, Mapping):
            stats = {}
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

    def _stage2_note_train_monitor_candidate(
        self, *, global_step: int, sample: Mapping[str, Any]
    ) -> None:
        if not self._stage2_train_monitor_step_allowed(global_step=global_step):
            return

        pending_gs = getattr(self, "_stage2_train_monitor_pending_gs", None)
        if pending_gs is None or int(pending_gs) != int(global_step):
            self._stage2_reset_train_monitor_dump(global_step=global_step)

        candidates = getattr(self, "_stage2_train_monitor_candidates", None)
        if not isinstance(candidates, list):
            candidates = []
            setattr(self, "_stage2_train_monitor_candidates", candidates)
        candidates.append(dict(sample))

    def _stage2_flush_train_monitor_dump(self, *, global_step: int) -> None:
        pending_gs = getattr(self, "_stage2_train_monitor_pending_gs", None)
        candidates = getattr(self, "_stage2_train_monitor_candidates", None)
        if (
            pending_gs is None
            or int(pending_gs) != int(global_step)
            or not isinstance(candidates, list)
            or not candidates
        ):
            return

        try:
            if not self._stage2_train_monitor_step_allowed(global_step=global_step):
                return

            dump_cfg = self._train_monitor_dump_cfg()
            dump_max_samples = max(1, int(dump_cfg.get("max_samples", 1) or 1))
            selected = sorted(
                (dict(sample) for sample in candidates),
                key=self._stage2_train_monitor_sort_key,
                reverse=True,
            )[:dump_max_samples]

            payload = {
                "kind": "train_monitor_dump",
                "global_step": int(global_step),
                "epoch": float(
                    getattr(getattr(self, "state", None), "epoch", 0.0) or 0.0
                ),
                "time": float(time.time()),
                "meta": {
                    "phase": "train",
                    "stage2_channel": "B",
                    "selection": "suspicious_duplication",
                    "candidate_count": int(len(candidates)),
                    "rollout_backend": str(self._rollout_backend()),
                    "decode_mode": str(self._cfg("decode_mode", "greedy")),
                },
                "metrics": {},
                "samples": selected,
            }
            self._stage2_train_monitor_dump_last_step = int(global_step)
            self._write_monitor_dump(global_step=int(global_step), payload=payload)
            self._stage2_train_monitor_dump_count = int(
                getattr(self, "_stage2_train_monitor_dump_count", 0) or 0
            ) + 1
            self._stage2_train_monitor_dump_written_step = int(global_step)
            logger.info(
                "stage2-ab wrote train monitor dump for global_step=%s with %s candidate samples",
                int(global_step),
                int(len(selected)),
            )
        finally:
            self._stage2_reset_train_monitor_dump(global_step=global_step)

    def train(self, *args, **kwargs):
        """Run training and ensure rollout resources shut down cleanly."""
        try:
            return super().train(*args, **kwargs)
        finally:
            # Best-effort: close HTTP sessions and communicator if server mode was used.
            try:
                self._shutdown_vllm_server_client(
                    close_communicator=True,
                    close_sessions=True,
                )
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
                logger.warning("Failed to shutdown vLLM server client: %s", exc)

            # Best-effort: release colocate engine before Python finalization to
            # reduce allocator teardown races in sleep-mode runs.
            try:
                self._shutdown_vllm_colocate_engine(wake_before_release=False)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
                logger.warning("Failed to shutdown vLLM colocate engine: %s", exc)

            # Ensure process-group teardown is explicit.
            try:
                import torch.distributed as dist

                if dist.is_available() and dist.is_initialized():
                    dist.destroy_process_group()
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
                logger.warning("Failed to destroy distributed process group: %s", exc)

    @contextlib.contextmanager
    def _hf_sampling_seed_context(
        self, *, seed_base: int, backend: str, do_sample: bool
    ):
        """Context manager: seed HF stochastic rollout generation without perturbing training RNG.

        - If backend!=hf or do_sample==False, yields False and does nothing.
        - Otherwise saves python/numpy/torch(+cuda) RNG state, seeds deterministically,
          yields True, then restores the saved RNG states.

        This satisfies the stage2-ab spec requirement (seed stochastic HF rollouts)
        while avoiding unintended effects on later training randomness.
        """
        if str(backend).lower() != "hf" or not bool(do_sample):
            yield False
            return

        sb = int(seed_base)

        import random

        py_state = random.getstate()

        np_state = None
        np_module = None
        try:
            import numpy as np
        except ImportError:
            np = None  # type: ignore[assignment]
        else:
            np_module = np
            np_state = np.random.get_state()

        torch_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        try:
            from transformers.trainer_utils import set_seed
        except ImportError:
            random.seed(sb)
            if np_state is not None and np_module is not None:
                np_module.random.seed(sb)
            torch.manual_seed(sb)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(sb)
        else:
            set_seed(sb)

        try:
            yield True
        finally:
            random.setstate(py_state)
            if np_state is not None and np_module is not None:
                np_module.random.set_state(np_state)
            torch.set_rng_state(torch_state)
            if cuda_state is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(cuda_state)

    # Stage-2 AB scheduler helpers live in `src/trainers/stage2_ab/scheduler.py`.

    def _dist_info(self) -> Tuple[int, int, Any]:
        rank = 0
        world_size = 1
        try:
            import torch.distributed as dist
        except ImportError:
            dist = None  # type: ignore[assignment]

        if dist is not None and dist.is_available() and dist.is_initialized():
            world_size = int(dist.get_world_size())
            rank = int(dist.get_rank())

        return int(rank), int(world_size), dist

    def _reduce_stage2_pending_metrics_global(
        self, metrics: Mapping[str, Any]
    ) -> Dict[str, float]:
        reduced: Dict[str, float] = {}
        for k, v in metrics.items():
            try:
                reduced[str(k)] = float(v)
            except (TypeError, ValueError):
                continue

        reduced.pop("rollout/parse_truncated_rate", None)

        trunc_num_key = "rollout/_parse_truncated_num"
        trunc_den_key = "rollout/_parse_truncated_den"
        gradmon_weight_key = "gradmon/_log_weight_total"
        has_rollout_parse_inputs = any(
            key in reduced
            for key in {
                trunc_num_key,
                trunc_den_key,
                "rollout/parse_truncated",
                "stage2/raw_rollouts",
            }
        )
        if has_rollout_parse_inputs:
            reduced.setdefault(
                trunc_num_key,
                float(reduced.get("rollout/parse_truncated", 0.0)),
            )
            reduced.setdefault(
                trunc_den_key,
                float(reduced.get("stage2/raw_rollouts", 0.0)),
            )

        rank, world_size, dist = self._dist_info()
        metric_keys: List[str] = sorted(str(k) for k in reduced.keys())

        if dist is not None and int(world_size) > 1:
            gathered_keys: List[Any] = [None] * int(world_size)
            dist.all_gather_object(gathered_keys, metric_keys)

            merged_keys: Dict[str, None] = {}
            for item in gathered_keys:
                if not isinstance(item, (list, tuple)):
                    raise RuntimeError(
                        "stage2-ab metric key sync produced non-list keys "
                        f"(rank={int(rank)}/{int(world_size)} type={type(item).__name__})"
                    )
                for key_raw in item:
                    key = str(key_raw)
                    merged_keys[key] = None
                    reduced.setdefault(key, 0.0)

            metric_keys = sorted(merged_keys.keys())

        sum_key_set: set[str] = set()
        max_key_set: set[str] = set()
        mean_key_set: set[str] = set()

        for key in metric_keys:
            if key.startswith("time/") or key in {
                "rollout/backend_hf",
                "rollout/backend_vllm",
                "rollout/decode_mode_greedy",
                "rollout/decode_mode_beam",
                "rollout/hf_seeded_global",
                "rollout/do_sample",
            }:
                max_key_set.add(key)
                continue

            if key == "stage2/_log_weight_total":
                sum_key_set.add(key)
                continue
            if key == gradmon_weight_key:
                sum_key_set.add(key)
                continue

            if key in {
                "stage2/raw_rollouts",
                "stage2/invalid_rollout",
                "stage2_ab/channel_b/invalid_rollout",
                "stage2/drop_poly",
                "stage2/drop_unknown",
                "stage2/drop_bbox_invalid",
                "rollout/parse_truncated",
                trunc_num_key,
                trunc_den_key,
            }:
                sum_key_set.add(key)
                continue

            if key.startswith("stage2_ab/channel_b/strict_drop/reason/"):
                sum_key_set.add(key)
                continue

            if key.startswith("stage2_ab/") and "/N_" in key:
                sum_key_set.add(key)
                continue

            if key.endswith(("_total", "_count", "_sum", "_num", "_den")):
                sum_key_set.add(key)
                continue

            mean_key_set.add(key)

        # Stable key ordering for all-reduce tensors (important for reproducibility).
        sum_priority = [
            gradmon_weight_key,
            trunc_num_key,
            trunc_den_key,
            "stage2/raw_rollouts",
            "stage2_ab/channel_b/invalid_rollout",
            "rollout/parse_truncated",
        ]
        sum_keys: List[str] = []
        for k in sum_priority:
            if k in sum_key_set:
                sum_keys.append(k)
                sum_key_set.remove(k)
        sum_keys.extend(sorted(sum_key_set))

        max_keys: List[str] = sorted(max_key_set)

        mean_keys: List[str] = sorted(mean_key_set)
        if dist is not None and int(world_size) > 1:
            try:
                device = torch.device("cpu")
                try:
                    model = getattr(self, "model", None)
                    if model is not None and hasattr(model, "device"):
                        device = model.device
                    elif model is not None:
                        device = next(model.parameters()).device
                except (AttributeError, RuntimeError, StopIteration, TypeError):
                    device = torch.device("cpu")

                def _all_reduce(keys: List[str], op: Any) -> None:
                    if not keys:
                        return
                    values = torch.tensor(
                        [float(reduced[k]) for k in keys],
                        dtype=torch.float64,
                        device=device,
                    )
                    dist.all_reduce(values, op=op)
                    for idx, key in enumerate(keys):
                        reduced[key] = float(values[idx].item())

                weight_key = "stage2/_log_weight_total"
                local_weight_total = float(reduced.get(weight_key, 0.0))
                local_gradmon_weight_total = float(reduced.get(gradmon_weight_key, 0.0))

                if weight_key in reduced and mean_keys:
                    # Convert local means into local numerators before reduction.
                    for key in mean_keys:
                        if key.startswith("gradmon/"):
                            continue
                        reduced[key] = float(reduced.get(key, 0.0)) * float(local_weight_total)
                if gradmon_weight_key in reduced and mean_keys:
                    for key in mean_keys:
                        if not key.startswith("gradmon/"):
                            continue
                        reduced[key] = float(reduced.get(key, 0.0)) * float(
                            local_gradmon_weight_total
                        )

                _all_reduce(sum_keys + mean_keys, dist.ReduceOp.SUM)
                _all_reduce(max_keys, dist.ReduceOp.MAX)

                if weight_key in reduced and mean_keys:
                    global_weight_total = float(reduced.get(weight_key, 0.0))
                    if global_weight_total > 0.0:
                        for key in mean_keys:
                            if key.startswith("gradmon/"):
                                continue
                            reduced[key] = float(reduced.get(key, 0.0) / global_weight_total)
                    else:
                        for key in mean_keys:
                            if key.startswith("gradmon/"):
                                continue
                            reduced[key] = 0.0
                if gradmon_weight_key in reduced and mean_keys:
                    global_gradmon_weight_total = float(reduced.get(gradmon_weight_key, 0.0))
                    if global_gradmon_weight_total > 0.0:
                        for key in mean_keys:
                            if not key.startswith("gradmon/"):
                                continue
                            reduced[key] = float(
                                reduced.get(key, 0.0) / global_gradmon_weight_total
                            )
                    else:
                        for key in mean_keys:
                            if not key.startswith("gradmon/"):
                                continue
                            reduced[key] = 0.0
                elif weight_key not in reduced:
                    scale = float(world_size)
                    if scale > 0.0:
                        for key in mean_keys:
                            if key.startswith("gradmon/"):
                                continue
                            reduced[key] = float(reduced[key] / scale)
            except (AttributeError, RuntimeError) as exc:
                raise RuntimeError(
                    "stage2-ab metric all-reduce failed (DDP is initialized); "
                    f"rank={int(rank)}/{int(world_size)}"
                ) from exc

        if has_rollout_parse_inputs:
            trunc_num = float(
                reduced.get(
                    trunc_num_key,
                    reduced.get("rollout/parse_truncated", 0.0),
                )
            )
            trunc_den = float(
                reduced.get(
                    trunc_den_key,
                    reduced.get("stage2/raw_rollouts", 0.0),
                )
            )
            reduced["rollout/parse_truncated_rate"] = (
                float(trunc_num / trunc_den) if trunc_den > 0.0 else 0.0
            )
        reduced.pop(trunc_num_key, None)
        reduced.pop(trunc_den_key, None)
        reduced.pop("stage2/_log_weight_total", None)
        reduced.pop(gradmon_weight_key, None)

        return reduced

    # Stage-2 AB channel executors live in `src/trainers/stage2_ab/executors.py`.

    def log(self, logs: Dict[str, float]) -> None:
        if (
            isinstance(logs, dict)
            and "loss" in logs
            and not any(str(k).startswith("eval_") for k in logs.keys())
        ):
            step = int(getattr(getattr(self, "state", None), "global_step", 0) or 0)
            pending = self._stage2_pending_train_logs.get(step)
            snapshot_state = getattr(self, "_stage2_metric_snapshots", None)
            if not isinstance(snapshot_state, dict):
                snapshot_state = {}
                setattr(self, "_stage2_metric_snapshots", snapshot_state)

            # IMPORTANT (DDP contract): stage2-ab log() merges buffered per-step
            # debug metrics via distributed collectives.
            #
            # It must be invoked on *every* rank when torch.distributed is
            # initialized. Do not make Trainer.log() rank-0-only, and do not gate
            # this path on is_main_process()/rank==0, otherwise rank 0 will enter
            # all_reduce() while other ranks skip it and the job will deadlock.
            self._ddp_assert_all_ranks_true_or_raise(
                where="stage2-ab train log",
                local_true=pending is not None,
                global_step=step,
            )
            pending = self._stage2_pending_train_logs.pop(step, None)
            snapshot_logs: Dict[str, float] = {}
            if pending is not None:
                reduced = self._reduce_stage2_pending_metrics_global(
                    pending.finalize(drop_internal=False)
                )
                reduced.pop("rollout/_parse_truncated_num", None)
                reduced.pop("rollout/_parse_truncated_den", None)
                logs.update(reduced)
                snapshot_logs = _merge_stage2_metric_snapshots(
                    snapshot_state,
                    reduced,
                )
            elif snapshot_state:
                snapshot_logs = {
                    str(key): float(value)
                    for key, value in sorted(
                        snapshot_state.items(), key=lambda item: item[0]
                    )
                }
            if snapshot_logs:
                logs.update(snapshot_logs)
            logs.update(
                self._reduce_stage_wallclock_metrics_global(
                    self._stage_wallclock_metrics_local()
                )
            )
        return super().log(logs)

    def training_step(self, model, inputs, *args, **kwargs):
        # When using identity collator, `inputs` is a list of raw samples.
        if not isinstance(inputs, list):
            with self._track_stage_wallclock("sft"):
                return super(RolloutMatchingSFTTrainer, self).training_step(
                    model, inputs, *args, **kwargs
                )

        if not inputs:
            rank, world_size, _dist = self._dist_info()
            gs = int(getattr(getattr(self, "state", None), "global_step", 0) or 0)
            raise ValueError(
                "stage2-ab training_step received an empty raw batch "
                f"(global_step={int(gs)} rank={int(rank)}/{int(world_size)}). "
                "This is unsafe under DDP because other ranks may execute forward/backward while this rank does not. "
                "Mitigations: ensure dataset length >= world_size and verify your sampler/drop_last settings."
            )

        self._validate_rollout_matching_cfg()

        gs = int(getattr(getattr(self, "state", None), "global_step", 0) or 0)
        ch_sched = self._stage2_channel_for_step(gs)

        # No async actor-learner gating: realized step kind matches schedule deterministically.
        self._stage2_record_realized_step(
            global_step=gs,
            executed_b=bool(ch_sched == "B"),
        )

        # Channel-B: single step-budgeted pathway.
        if ch_sched == "B":
            return self._stage2_training_step_b_step_mode(model, inputs, global_step=gs)

        # Channel-A: keep existing behavior (step-budgeted packing when enabled).
        prev = self._stage2_channel_override
        self._stage2_channel_override = "A"
        try:
            packing_enabled = bool(self._packing_enabled())
            if packing_enabled:
                return self._stage2_training_step_a_step_mode(
                    model, inputs, global_step=gs
                )
            with self._track_stage_wallclock("sft"):
                prepared = self._prepare_batch_inputs(inputs)
                return super(RolloutMatchingSFTTrainer, self).training_step(
                    model, prepared, *args, **kwargs
                )
        finally:
            self._stage2_channel_override = prev



    def _prepare_window_packed_batches(
        self,
        *,
        window_raw_micro_batches: List[List[Mapping[str, Any]]],
        global_step: int,
    ) -> List[Dict[str, Any]]:
        """Build packed prepared batches for one full accumulation window.

        Stage2-AB specialization: batch rollout generation across the whole window so
        rollout request batch size can be > learner microbatch size.

        - Learner microbatch stays small (typically 1 packed sequence per backward).
        - Channel-B rollouts can be generated for the whole window in one pass.
        - Channel-A teacher-forced targets are encoded for the whole window in one pass.
        """

        gas = int(len(window_raw_micro_batches))
        if gas <= 0:
            raise ValueError("window_raw_micro_batches is empty")

        ch = self._stage2_channel_for_step(int(global_step))
        channel: Literal["A", "B"] = "A" if ch == "A" else "B"

        # Flatten window samples (preserve micro-step order).
        flat_inputs: List[Mapping[str, Any]] = []
        for mb in window_raw_micro_batches:
            flat_inputs.extend(list(mb))

        # Build post-rollout segments for the whole window in one pass.
        if channel == "A":
            segs, bm_total = self._prepare_batch_inputs_a(
                flat_inputs, _segments_only=True
            )
        else:
            segs, bm_total = self._prepare_batch_inputs_b(
                flat_inputs, _segments_only=True
            )

        # Schedule segments into exactly `gas` micro-packs.
        t_pack0 = time.perf_counter()
        packs: List[List[Tuple[Dict[str, Any], Dict[str, Any], int]]]
        window_pack_metrics: Dict[str, float] = {}
        fallback_used = False
        pack_metrics_list: Optional[List[Dict[str, float]]] = None
        try:
            packs, window_pack_metrics = self._schedule_post_rollout_packs_window(
                window_segments=segs,
                gas=gas,
            )
        except (RuntimeError, ValueError) as exc:
            # Fallback: greedy carry packing using the channel-local buffer. This is
            # less strict than window scheduling (it can borrow segments across micro-steps)
            # and avoids a hard failure when a window produces too few/infeasible segments.
            fallback_used = True
            logger.warning(
                "stage2-ab window packing failed (channel=%s); falling back to greedy carry packing for this window: %s",
                channel,
                exc,
            )
            self._stage2_append_post_rollout_segments(channel=channel, segments=segs)

            packs = []
            pack_metrics_list = []
            for _ in range(int(gas)):
                selected, pm = self._stage2_pop_post_rollout_pack(channel=channel)
                packs.append(selected)
                pack_metrics_list.append(dict(pm) if isinstance(pm, Mapping) else {})

        t_pack_s = float(time.perf_counter() - t_pack0)

        template = self.template
        from swift.llm import to_device

        prepared: List[Dict[str, Any]] = []
        packing_length = int(self._packing_length())

        # Put aggregated rollout/match metrics on the first micro-step only to avoid
        # spamming the log with duplicated window totals.
        for i, selected in enumerate(packs):
            if not selected:
                raise ValueError(
                    "window post-rollout packing produced an empty micro-pack. "
                    "This indicates the window did not produce enough post-rollout segments."
                )

            with self._template_packing_enabled():
                packed = template.data_collator([enc for enc, _, _ in selected])
            batch = to_device(packed, self.model.device)
            self._assert_single_packed_forward(batch, where="stage2_ab/packed_forward")
            batch["_rollout_matching_meta"] = [m for _, m, _ in selected]

            bm: Dict[str, float] = {}
            if i == 0 and isinstance(bm_total, Mapping):
                bm.update(
                    {
                        k: float(v)
                        for k, v in bm_total.items()
                        if isinstance(v, (int, float))
                    }
                )
                if not fallback_used:
                    bm.update(window_pack_metrics)
                bm["time/post_rollout_pack_s"] = float(t_pack_s)
                bm["packing/post_rollout_scope_window"] = 1.0
                bm["packing/post_rollout_scope_window_fallback"] = float(
                    1.0 if fallback_used else 0.0
                )
            else:
                bm["time/post_rollout_pack_s"] = 0.0
                bm["packing/post_rollout_scope_window"] = 0.0
                bm["packing/post_rollout_scope_window_fallback"] = 0.0

            sel_total = int(sum(int(sl) for _, _, sl in selected))
            fill = (
                float(sel_total) / float(packing_length) if packing_length > 0 else 0.0
            )

            buf_size = 0.0
            if fallback_used and pack_metrics_list is not None and i < len(pack_metrics_list):
                try:
                    buf_size = float(pack_metrics_list[i].get("packing/post_rollout_buffer", 0.0))
                except (TypeError, ValueError):
                    buf_size = 0.0

            bm.update(
                {
                    "packing/post_rollout_fill": float(fill),
                    "packing/post_rollout_selected_total_len": float(sel_total),
                    "packing/post_rollout_segments": float(len(selected)),
                    "packing/post_rollout_buffer": float(buf_size if fallback_used else 0.0),
                }
            )

            self._merge_rollout_matching_batch_metrics(batch, bm)
            batch["_stage2_ab_channel"] = channel
            prepared.append(batch)

        return prepared

    def _prepare_batch_inputs(
        self,
        inputs: List[Mapping[str, Any]],
        _segments_only: bool = False,
    ) -> Any:
        ch = self._stage2_channel_for_step(
            int(getattr(getattr(self, "state", None), "global_step", 0) or 0)
        )
        if ch == "A":
            return self._prepare_batch_inputs_a(inputs, _segments_only=_segments_only)
        return self._prepare_batch_inputs_b(inputs, _segments_only=_segments_only)

    def _prepare_batch_inputs_a(
        self, inputs: List[Mapping[str, Any]], *, _segments_only: bool
    ) -> Any:
        template = self.template
        tok = template.tokenizer

        coord_token_ids = self._get_coord_token_ids()
        coord_id_set = set(int(i) for i in coord_token_ids if int(i) >= 0)

        packing_enabled = self._packing_enabled()
        if packing_enabled and not self._packing_drop_last():
            raise ValueError(
                "stage2-ab packing uses carry-only mode and requires training.packing_drop_last: true"
            )
        if packing_enabled and self._packing_buffer_cap() <= 0:
            raise ValueError(
                "training.packing_buffer must be a positive int when packing is enabled"
            )
        if packing_enabled and self._packing_length() <= 0:
            raise ValueError(
                "packing is enabled but no valid packing_length/template.max_length is set (check global_max_length)"
            )

        encoded_batch: List[Dict[str, Any]] = []
        meta_unpacked: List[Dict[str, Any]] = []
        segments: List[Tuple[Dict[str, Any], Dict[str, Any], int]] = []

        t_encode_s = 0.0

        for sample in inputs:
            if "messages" not in sample:
                raise ValueError("stage2-ab requires 'messages' in dataset samples")

            gts = _extract_gt_bboxonly(sample)

            payload = _build_teacher_forced_payload(
                gt_objects=gts,
                object_field_order=self._object_field_order(),
            )
            assistant_text = dumps_coordjson(payload)
            y_train_ids = tok.encode(assistant_text, add_special_tokens=False)

            # Desc spans for CE weighting (relative to tail ids).
            tail_desc_pos = find_desc_value_token_positions(
                tokenizer=tok, token_ids=y_train_ids
            )

            # Bbox groups (relative positions in y_train_ids); convert to full positions after we know prompt_len.
            rel_groups = _bbox_groups_from_token_ids(
                token_ids=y_train_ids, coord_id_set=coord_id_set, gt_objs=gts
            )

            t_enc0 = time.perf_counter()
            data_for_encode = dict(sample)
            messages = json.loads(json.dumps(sample["messages"]))
            has_assistant = any(
                isinstance(m, dict) and m.get("role") == "assistant" for m in messages
            )

            if has_assistant:
                data_for_encode["messages"] = replace_assistant_response_with_ids(
                    messages, y_train_ids
                )
            else:
                data_for_encode["messages"] = list(messages) + [
                    {"role": "assistant", "content": y_train_ids}
                ]
            try:
                with self._template_train_mode():
                    encoded = template.encode(data_for_encode, return_length=True)
            except MaxLengthError as e:
                max_len = getattr(template, "max_length", None)
                raise MaxLengthError(
                    "stage2-ab teacher-forced encode exceeded max_length="
                    f"{max_len}. SFT forbids truncation; pre-filter long samples or increase global_max_length."
                ) from e
            t_encode_s += time.perf_counter() - t_enc0

            encoded_len = self._extract_encoded_len(encoded)

            enc_ids = encoded.get("input_ids")
            if not isinstance(enc_ids, Sequence):
                raise ValueError("template.encode did not return sequence input_ids")
            enc_ids_list = [int(x) for x in enc_ids]

            # Robustly locate assistant span (prompt_len) even under truncation.
            prompt_len: Optional[int] = None
            train_len_eff: Optional[int] = None
            labels = encoded.get("labels")
            if isinstance(labels, Sequence):
                try:
                    lbl = [int(x) for x in labels]
                    first = next((i for i, x in enumerate(lbl) if int(x) != -100), None)
                    if first is not None:
                        last = max(i for i, x in enumerate(lbl) if int(x) != -100)
                        prompt_len = int(first)
                        train_len_eff = int(last - first + 1)
                except (TypeError, ValueError):
                    prompt_len = None
                    train_len_eff = None

            if prompt_len is None:
                prompt_len = _find_subsequence(enc_ids_list, y_train_ids)
                train_len_eff = int(len(y_train_ids))

            prompt_len = int(prompt_len)
            train_len_eff = int(train_len_eff or 0)
            max_train_len = int(encoded_len) - int(prompt_len)
            if max_train_len < int(len(y_train_ids)):
                raise ValueError(
                    "stage2-ab SFT forbids truncation: teacher-forced assistant span was truncated. "
                    f"encoded_len={int(encoded_len)} prompt_len={int(prompt_len)} max_train_len={int(max_train_len)} y_train_len={int(len(y_train_ids))}"
                )
            train_len_eff = max(0, min(train_len_eff, int(max_train_len)))
            if train_len_eff < int(len(y_train_ids)):
                raise ValueError(
                    "stage2-ab teacher-forced labels span shorter than assistant token ids (possible masking). "
                    f"train_len={int(train_len_eff)} y_train_len={int(len(y_train_ids))}"
                )

            prompt_ids = enc_ids_list[:prompt_len]

            tail_desc_pos_eff: List[int] = []
            for rel in tail_desc_pos:
                try:
                    rel_i = int(rel)
                except (TypeError, ValueError):
                    continue
                if 0 <= rel_i < train_len_eff:
                    tail_desc_pos_eff.append(rel_i)

            bbox_groups_fn: List[Dict[str, Any]] = []
            for obj, rel_pos in zip(gts, rel_groups):
                try:
                    rel_pos_int = [int(p) for p in rel_pos]
                except (TypeError, ValueError):
                    continue
                if len(rel_pos_int) != 4:
                    continue
                if any(p < 0 or p >= train_len_eff for p in rel_pos_int):
                    raise ValueError(
                        "stage2-ab bbox group pos out of range in teacher-forced encode (possible truncation). "
                        f"rel_pos={rel_pos_int} train_len={int(train_len_eff)} y_train_len={int(len(y_train_ids))}"
                    )
                abs_pos = [int(prompt_len + p) for p in rel_pos_int]
                if any(p >= int(encoded_len) for p in abs_pos):
                    raise ValueError(
                        "stage2-ab bbox group absolute pos exceeds encoded_len (possible truncation/misalignment). "
                        f"abs_pos={abs_pos} encoded_len={int(encoded_len)} prompt_len={int(prompt_len)}"
                    )
                bbox_groups_fn.append(
                    {
                        "pos": abs_pos,
                        "gt_bins": list(obj.points_norm1000),
                    }
                )

            meta_entry: Dict[str, Any] = {
                "stage2_channel": "A",
                "prompt_len": int(prompt_len),
                "prompt_ids": prompt_ids,
                "rollout_len": int(0),
                "prefix_len": int(0),
                "train_len": int(train_len_eff),
                "encoded_len": int(encoded_len),
                "decode_mode": "none",
                "parse_dropped_invalid": int(0),
                "parse_dropped_ambiguous": int(0),
                "parse_truncated": bool(False),
                "valid_pred_objects": int(0),
                "matched_for_supervision": int(0),
                "matched_maskiou_sum": float(0.0),
                "matched_maskiou_count": int(0),
                "gt_objects": int(len(gts)),
                "fn_count": int(0),
                "gating_rejections": int(0),
                "excluded_from_supervision": int(0),
                "prefix_coord_pos": [],
                "prefix_coord_target_bins": [],
                "tail_ignore_pos": [],
                "tail_desc_pos": tail_desc_pos_eff,
                "bbox_groups_prefix": [],
                "bbox_groups_fn": bbox_groups_fn,
            }

            segments.append((encoded, meta_entry, int(encoded_len)))
            if not packing_enabled:
                encoded_batch.append(encoded)
                meta_unpacked.append(meta_entry)

        from swift.llm import to_device

        batch_metrics: Dict[str, float] = {
            "stage2/channel_a": float(1.0),
            "stage2/channel_b": float(0.0),
            "time/channel_a_teacher_encode_s": float(t_encode_s),
        }

        if bool(_segments_only):
            return segments, batch_metrics

        if packing_enabled:
            self._stage2_append_post_rollout_segments(channel="A", segments=segments)

            t_pack0 = time.perf_counter()
            selected, pack_metrics = self._stage2_pop_post_rollout_pack(channel="A")
            with self._template_packing_enabled():
                packed = template.data_collator([enc for enc, _, _ in selected])
            batch = to_device(packed, self.model.device)
            self._assert_single_packed_forward(batch, where="stage2_ab/packed_forward")
            batch["_rollout_matching_meta"] = [m for _, m, _ in selected]

            batch_metrics.update(pack_metrics)
            batch_metrics["time/channel_a_pack_s"] = float(
                time.perf_counter() - t_pack0
            )
            self._merge_rollout_matching_batch_metrics(batch, batch_metrics)
            batch["_stage2_ab_channel"] = "A"
            return batch

        with self._template_packing_disabled():
            batch = to_device(template.data_collator(encoded_batch), self.model.device)
        batch["_rollout_matching_meta"] = meta_unpacked
        self._merge_rollout_matching_batch_metrics(batch, batch_metrics)
        batch["_stage2_ab_channel"] = "A"
        return batch

    def _prepare_batch_inputs_b(
        self, inputs: List[Mapping[str, Any]], *, _segments_only: bool
    ) -> Any:
        template = self.template
        tok = template.tokenizer

        coord_token_ids = self._get_coord_token_ids()
        coord_id_set = set(int(i) for i in coord_token_ids if int(i) >= 0)
        coord_id_to_bin = self._coord_id_map()

        gate_thr = float(self._cfg("maskiou_gate", 0.3))
        match_top_k = int(self._cfg("candidate_top_k", 10))
        mask_res = int(self._cfg("maskiou_resolution", 256))

        fp_cost = float(self._cfg("fp_cost", 1.0))
        fn_cost = float(self._cfg("fn_cost", 1.0))

        # Stage-2 AB contract: FN append is always enabled in Channel-B.
        duplicate_iou_threshold_raw = self._ab_channel_b_get(
            "duplicate_iou_threshold",
            0.90,
        )
        duplicate_iou_threshold = float(
            0.90 if duplicate_iou_threshold_raw is None else duplicate_iou_threshold_raw
        )
        unlabeled_consistent_iou_threshold = float(
            self._ab_channel_b_get("triage_posterior.unlabeled_consistent_iou_threshold", 0.85)
        )
        recovered_ground_truth_weight_multiplier = float(
            self._ab_channel_b_get("triage_posterior.recovered_ground_truth_weight_multiplier", 2.0)
        )
        explorer_temperature = float(
            self._ab_channel_b_get("triage_posterior.explorer_temperature", 0.7)
        )
        explorer_top_p = float(self._ab_channel_b_get("triage_posterior.explorer_top_p", 1.0))
        explorer_top_k = int(self._ab_channel_b_get("triage_posterior.explorer_top_k", -1))


        packing_enabled = self._packing_enabled()
        if packing_enabled and not self._packing_drop_last():
            raise ValueError(
                "stage2-ab post-rollout packing uses carry-only mode and requires training.packing_drop_last: true"
            )
        if packing_enabled and self._packing_buffer_cap() <= 0:
            raise ValueError(
                "training.packing_buffer must be a positive int when packing is enabled"
            )
        if packing_enabled and self._packing_length() <= 0:
            raise ValueError(
                "packing is enabled but no valid packing_length/template.max_length is set (check global_max_length)"
            )

        gs = int(getattr(getattr(self, "state", None), "global_step", 0) or 0)
        monitor_step = int(gs + 1)
        object_field_order = self._object_field_order()
        track_monitor_candidates = bool(
            self._stage2_train_monitor_step_allowed(global_step=monitor_step)
        )
        seed_base = int(self._derive_rollout_seed_base(global_step=gs))

        backend = self._rollout_backend()
        anchor_decode_request = self._resolve_rollout_decode_request(
            decode_override={
                "decode_mode": "greedy",
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
            }
        )
        explorer_decode_request = self._resolve_rollout_decode_request(
            decode_override={
                "decode_mode": "sampling",
                "temperature": float(explorer_temperature),
                "top_p": float(explorer_top_p),
                "top_k": int(explorer_top_k),
            }
        )
        max_new_tokens = int(anchor_decode_request.max_new_tokens)
        num_beams = int(anchor_decode_request.num_beams)
        repetition_penalty = float(anchor_decode_request.repetition_penalty)
        do_sample = bool(float(explorer_decode_request.temperature) > 0.0)

        inputs_for_rollout = self._prepare_samples_for_rollout(
            inputs,
            rollout_backend=backend,
        )

        def _rollout_many_with_decode_override(
            chunk: Sequence[Mapping[str, Any]],
            *,
            decode_override: Mapping[str, Any],
            request_index_offset: int,
        ) -> Any:
            candidates = [
                {
                    "decode_override": decode_override,
                    "request_index_offset": int(request_index_offset),
                },
                {"decode_override": decode_override},
                {"request_index_offset": int(request_index_offset)},
                {},
            ]
            last_exc: Optional[TypeError] = None
            for kwargs in candidates:
                try:
                    return self._rollout_many(chunk, **kwargs)
                except TypeError as exc:
                    text = str(exc)
                    if "unexpected keyword argument" not in text:
                        raise
                    last_exc = exc
            if last_exc is not None:
                raise last_exc
            raise AssertionError("unreachable")

        with self._hf_sampling_seed_context(
            seed_base=seed_base, backend=backend, do_sample=do_sample
        ) as seeded:
            hf_seeded_global = float(1.0 if seeded else 0.0)

            t_gen0 = time.perf_counter()

            # Derived rollout request chunk size per learner rank.
            #
            # `channel_b_decode_batch_size` is defined as a per-rollout-GPU cap per
            # generation call during train-step Channel-B rollouts. For vLLM server mode
            # (data-parallel replicas), we derive the per-rank chunk size so the per-GPU
            # cap holds when all learner ranks run concurrently.
            rollout_infer_bs = int(self._rollout_decode_batch_size_per_rank())
            rollout_infer_bs = max(1, int(rollout_infer_bs))
            if int(len(inputs_for_rollout)) > 0:
                rollout_infer_bs = min(
                    int(rollout_infer_bs), int(len(inputs_for_rollout))
                )

            anchor_rollout_results = []
            explorer_rollout_results = []
            for off in range(0, int(len(inputs_for_rollout)), int(rollout_infer_bs)):
                chunk = inputs_for_rollout[int(off) : int(off + rollout_infer_bs)]
                if not chunk:
                    continue

                anchor_chunk_results = _rollout_many_with_decode_override(
                    chunk,
                    decode_override={
                        "decode_mode": str(anchor_decode_request.decode_mode),
                        "temperature": float(anchor_decode_request.temperature),
                        "top_p": float(anchor_decode_request.top_p),
                        "top_k": int(anchor_decode_request.top_k),
                    },
                    request_index_offset=int(off),
                )
                explorer_chunk_results = _rollout_many_with_decode_override(
                    chunk,
                    decode_override={
                        "decode_mode": str(explorer_decode_request.decode_mode),
                        "temperature": float(explorer_decode_request.temperature),
                        "top_p": float(explorer_decode_request.top_p),
                        "top_k": int(explorer_decode_request.top_k),
                    },
                    request_index_offset=int(off),
                )
                anchor_rollout_results.extend(anchor_chunk_results)
                explorer_rollout_results.extend(explorer_chunk_results)

            if len(anchor_rollout_results) != len(inputs_for_rollout):
                raise RuntimeError(
                    "anchor rollout backend returned unexpected number of results"
                )
            if len(explorer_rollout_results) != len(inputs_for_rollout):
                raise RuntimeError(
                    "explorer rollout backend returned unexpected number of results"
                )
            t_gen_s = time.perf_counter() - t_gen0

        encoded_batch: List[Dict[str, Any]] = []
        meta_unpacked: List[Dict[str, Any]] = []
        segments: List[Tuple[Dict[str, Any], Dict[str, Any], int]] = []

        t_parse_match_s = 0.0
        t_encode_s = 0.0

        drop_poly_total = 0
        drop_unknown_total = 0
        drop_bbox_invalid_total = 0
        invalid_rollout_total = 0
        parse_truncated_total = 0
        closure_supervision_drop_total = 0
        prompt_tok_mismatch_total = 0

        strict_valid_pred_total = 0
        strict_drop_invalid_total = 0
        strict_drop_by_reason_total: Dict[str, int] = {}
        dup_max_desc_count_sum = 0.0
        dup_saturation_rate_sum = 0.0
        dup_near_same_desc_pairs_total = 0
        dup_near_any_desc_pairs_total = 0
        dup_raw_bbox_valid_total = 0
        dup_clean_accepted_total = 0
        dup_duplicates_total = 0
        dup_duplicate_bursts_total = 0
        dup_ul_boundaries_total = 0
        dup_dead_anchor_suppression_skipped_no_divergence_total = 0
        dup_metric_samples = 0
        triage_anchor_gt_backed_total = 0
        triage_shielded_anchor_total = 0
        triage_dead_anchor_total = 0
        triage_dead_explorer_total = 0
        triage_recovered_gt_total = 0
        triage_recovered_gt_den_total = 0
        triage_dead_anchor_den_total = 0
        triage_dead_explorer_den_total = 0
        matched_for_supervision_total = 0

        anchor_pred_objects_total = 0
        anchor_valid_pred_objects_total = 0
        anchor_parse_truncated_total = 0
        anchor_gen_new_token_lens: List[int] = []
        anchor_near_same_desc_pairs_total = 0
        anchor_near_any_desc_pairs_total = 0

        explorer_pred_objects_total = 0
        explorer_valid_pred_objects_total = 0
        explorer_parse_truncated_total = 0
        explorer_gen_new_token_lens: List[int] = []
        explorer_near_same_desc_pairs_total = 0
        explorer_near_any_desc_pairs_total = 0

        def _build_rollout_view(
            *,
            gts: Sequence[GTObject],
            rollout_result: Tuple[List[int], str, str, List[int]],
        ) -> Dict[str, Any]:
            resp_ids, _resp_text, rollout_decode_mode, prompt_ids = rollout_result
            resp_ids_local = [int(t) for t in resp_ids]
            eos_id = getattr(tok, "eos_token_id", None)
            if int(max_new_tokens) > 0 and int(len(resp_ids_local)) >= int(max_new_tokens):
                try:
                    eos = int(eos_id) if eos_id is not None else -1
                except (TypeError, ValueError):
                    eos = -1
                if eos >= 0 and (not resp_ids_local or int(resp_ids_local[-1]) != eos):
                    resp_ids_local.append(int(eos))

            parse = parse_rollout_for_matching(
                tokenizer=tok,
                response_token_ids=resp_ids_local,
                object_field_order=self._object_field_order(),
            )
            invalid_rollout = int(
                1 if bool(getattr(parse, "invalid_rollout", False)) else 0
            )

            drop_reasons: Dict[str, int] = {}
            raw = getattr(parse, "dropped_invalid_by_reason", None)
            if isinstance(raw, Mapping):
                for k, v in raw.items():
                    try:
                        drop_reasons[str(k)] = int(v)
                    except (TypeError, ValueError):
                        continue

            drop_poly = 0
            drop_unknown = 0
            drop_bbox_invalid = 0
            parsed_bbox_objects_raw: List[GTObject] = []
            for pobj in list(parse.valid_objects):
                if pobj.geom_type != "bbox_2d":
                    if pobj.geom_type == "poly":
                        drop_poly += 1
                    else:
                        drop_unknown += 1
                    continue

                pts = points_from_coord_tokens(
                    response_token_ids=parse.response_token_ids,
                    coord_token_indices=pobj.coord_token_indices,
                    coord_id_to_bin=coord_id_to_bin,
                )
                if pts is None or len(pts) != 4:
                    drop_bbox_invalid += 1
                    continue
                try:
                    x1, y1, x2, y2 = [int(x) for x in pts]
                except (TypeError, ValueError):
                    drop_bbox_invalid += 1
                    continue
                if x2 < x1 or y2 < y1:
                    drop_bbox_invalid += 1
                    continue

                parsed_bbox_objects_raw.append(
                    GTObject(
                        index=int(pobj.index),
                        geom_type="bbox_2d",
                        points_norm1000=[x1, y1, x2, y2],
                        desc=str(pobj.desc),
                    )
                )

            if drop_poly:
                drop_reasons["poly_unsupported"] = int(
                    drop_reasons.get("poly_unsupported", 0)
                ) + int(drop_poly)
            if drop_unknown:
                drop_reasons["unknown_geom"] = int(
                    drop_reasons.get("unknown_geom", 0)
                ) + int(drop_unknown)
            if drop_bbox_invalid:
                drop_reasons["bbox_invalid"] = int(
                    drop_reasons.get("bbox_invalid", 0)
                ) + int(drop_bbox_invalid)

            n_valid_pred = int(len(parsed_bbox_objects_raw))
            n_drop_invalid = (
                int(getattr(parse, "dropped_invalid", 0) or 0)
                + int(getattr(parse, "dropped_ambiguous", 0) or 0)
                + int(drop_poly)
                + int(drop_unknown)
                + int(drop_bbox_invalid)
            )

            accepted_objects_clean, duplicate_bursts_by_boundary = (
                _sequential_dedup_bbox_objects(
                    parsed_bbox_objects_raw=parsed_bbox_objects_raw,
                    duplicate_iou_threshold=duplicate_iou_threshold,
                )
            )
            duplicate_metrics = _compute_duplicate_diagnostics(parsed_bbox_objects_raw)
            match = hungarian_match_maskiou(
                preds=accepted_objects_clean,
                gts=gts,
                top_k=match_top_k,
                gate_threshold=gate_thr,
                mask_resolution=mask_res,
                fp_cost=fp_cost,
                fn_cost=fn_cost,
            )

            return {
                "prompt_ids": [int(t) for t in prompt_ids],
                "decode_mode": str(rollout_decode_mode),
                "pred_objects": int(len(parse.valid_objects)),
                "parse_truncated": int(
                    1 if bool(getattr(parse, "truncated", False)) else 0
                ),
                "gen_new_tokens": int(len(parse.response_token_ids)),
                "parse": parse,
                "invalid_rollout": int(invalid_rollout),
                "drop_reasons": drop_reasons,
                "drop_poly": int(drop_poly),
                "drop_unknown": int(drop_unknown),
                "drop_bbox_invalid": int(drop_bbox_invalid),
                "parsed_bbox_objects_raw": parsed_bbox_objects_raw,
                "n_valid_pred": int(n_valid_pred),
                "n_drop_invalid": int(n_drop_invalid),
                "accepted_objects_clean": accepted_objects_clean,
                "duplicate_bursts_by_boundary": duplicate_bursts_by_boundary,
                "duplicate_metrics": duplicate_metrics,
                "duplicate_count": int(
                    sum(
                        len(burst)
                        for burst in duplicate_bursts_by_boundary.values()
                    )
                ),
                "duplicate_burst_count": int(len(duplicate_bursts_by_boundary)),
                "match": match,
            }

        for sample, anchor_rollout, explorer_rollout in zip(
            inputs_for_rollout,
            anchor_rollout_results,
            explorer_rollout_results,
        ):
            if "messages" not in sample:
                raise ValueError("stage2-ab requires 'messages' in dataset samples")

            t_pm0 = time.perf_counter()
            gts = _extract_gt_bboxonly(sample)
            anchor_view = _build_rollout_view(
                gts=gts,
                rollout_result=anchor_rollout,
            )
            explorer_view = _build_rollout_view(
                gts=gts,
                rollout_result=explorer_rollout,
            )

            parse = anchor_view["parse"]
            invalid_rollout = int(anchor_view["invalid_rollout"])
            parse_truncated_total += int(1 if bool(parse.truncated) else 0)

            drop_reasons = dict(anchor_view["drop_reasons"])
            drop_poly = int(anchor_view["drop_poly"])
            drop_unknown = int(anchor_view["drop_unknown"])
            drop_bbox_invalid = int(anchor_view["drop_bbox_invalid"])
            parsed_bbox_objects_raw = list(anchor_view["parsed_bbox_objects_raw"])
            n_valid_pred = int(anchor_view["n_valid_pred"])
            n_drop_invalid = int(anchor_view["n_drop_invalid"])
            accepted_objects_clean = list(anchor_view["accepted_objects_clean"])
            duplicate_bursts_by_boundary = {
                int(boundary): list(burst)
                for boundary, burst in dict(
                    anchor_view["duplicate_bursts_by_boundary"]
                ).items()
            }
            duplicate_metrics = dict(anchor_view["duplicate_metrics"])
            duplicate_count = int(anchor_view["duplicate_count"])
            duplicate_burst_count = int(anchor_view["duplicate_burst_count"])
            match = anchor_view["match"]
            prompt_ids = list(anchor_view["prompt_ids"])
            decode_mode = str(anchor_view["decode_mode"])

            dup_max_desc_count = float(
                duplicate_metrics.get("dup/max_desc_count", 0.0) or 0.0
            )
            dup_saturation_rate = float(
                duplicate_metrics.get("dup/saturation_rate", 0.0) or 0.0
            )
            dup_near_same_desc_pairs = int(
                duplicate_metrics.get("dup/near_iou90_pairs_same_desc_count", 0.0)
                or 0.0
            )
            dup_near_any_desc_pairs = int(
                duplicate_metrics.get("dup/near_iou90_pairs_any_desc_count", 0.0)
                or 0.0
            )

            drop_poly_total += int(drop_poly)
            drop_unknown_total += int(drop_unknown)
            drop_bbox_invalid_total += int(drop_bbox_invalid)

            strict_valid_pred_total += int(n_valid_pred)
            strict_drop_invalid_total += int(n_drop_invalid)
            for rk, rv in drop_reasons.items():
                try:
                    rvi = int(rv)
                except (TypeError, ValueError):
                    continue
                if rvi <= 0:
                    continue
                strict_drop_by_reason_total[str(rk)] = int(
                    strict_drop_by_reason_total.get(str(rk), 0)
                ) + int(rvi)

            dup_max_desc_count_sum += float(dup_max_desc_count)
            dup_saturation_rate_sum += float(dup_saturation_rate)
            dup_near_same_desc_pairs_total += int(dup_near_same_desc_pairs)
            dup_near_any_desc_pairs_total += int(dup_near_any_desc_pairs)
            dup_raw_bbox_valid_total += int(len(parsed_bbox_objects_raw))
            dup_clean_accepted_total += int(len(accepted_objects_clean))
            dup_duplicates_total += int(duplicate_count)
            dup_duplicate_bursts_total += int(duplicate_burst_count)
            dup_metric_samples += 1

            anchor_match_by_pred = {
                int(pred_i): int(gt_i)
                for pred_i, gt_i in match.matched_pairs
                if 0 <= int(pred_i) < len(accepted_objects_clean)
                and 0 <= int(gt_i) < len(gts)
            }
            explorer_match = explorer_view["match"]
            explorer_accepted_objects_clean = list(
                explorer_view["accepted_objects_clean"]
            )
            explorer_match_by_pred = {
                int(pred_i): int(gt_i)
                for pred_i, gt_i in explorer_match.matched_pairs
                if 0 <= int(pred_i) < len(explorer_accepted_objects_clean)
                and 0 <= int(gt_i) < len(gts)
            }
            anchor_gt_indices = set(int(gt_i) for gt_i in anchor_match_by_pred.values())
            explorer_gt_indices = set(
                int(gt_i) for gt_i in explorer_match_by_pred.values()
            )
            anchor_pred_objects_total += int(anchor_view["pred_objects"])
            anchor_valid_pred_objects_total += int(anchor_view["n_valid_pred"])
            anchor_parse_truncated_total += int(anchor_view["parse_truncated"])
            anchor_gen_new_token_lens.append(int(anchor_view["gen_new_tokens"]))
            anchor_near_same_desc_pairs_total += int(
                anchor_view["duplicate_metrics"].get(
                    "dup/near_iou90_pairs_same_desc_count", 0.0
                )
                or 0.0
            )
            anchor_near_any_desc_pairs_total += int(
                anchor_view["duplicate_metrics"].get(
                    "dup/near_iou90_pairs_any_desc_count", 0.0
                )
                or 0.0
            )

            explorer_pred_objects_total += int(explorer_view["pred_objects"])
            explorer_valid_pred_objects_total += int(explorer_view["n_valid_pred"])
            explorer_parse_truncated_total += int(explorer_view["parse_truncated"])
            explorer_gen_new_token_lens.append(int(explorer_view["gen_new_tokens"]))
            explorer_near_same_desc_pairs_total += int(
                explorer_view["duplicate_metrics"].get(
                    "dup/near_iou90_pairs_same_desc_count", 0.0
                )
                or 0.0
            )
            explorer_near_any_desc_pairs_total += int(
                explorer_view["duplicate_metrics"].get(
                    "dup/near_iou90_pairs_any_desc_count", 0.0
                )
                or 0.0
            )

            association_pairs = associate_one_to_one_max_iou(
                anchors=accepted_objects_clean,
                explorers=explorer_accepted_objects_clean,
                min_iou=float(unlabeled_consistent_iou_threshold),
            )
            anchor_to_explorer = {
                int(anchor_i): int(explorer_i)
                for anchor_i, explorer_i in association_pairs
            }

            anchor_gt_backed_indices = sorted(anchor_match_by_pred.keys())
            shielded_anchor_indices: set[int] = set()
            dead_anchor_indices: set[int] = set()
            for anchor_i, anchor_obj in enumerate(accepted_objects_clean):
                if int(anchor_i) in anchor_match_by_pred:
                    continue
                explorer_i = anchor_to_explorer.get(int(anchor_i))
                if (
                    explorer_i is not None
                    and explorer_i not in explorer_match_by_pred
                ):
                    conflicts_gt_backed = any(
                        _bbox_iou_norm1000_xyxy(
                            anchor_obj.points_norm1000,
                            accepted_objects_clean[int(gt_anchor_i)].points_norm1000,
                        )
                        >= float(unlabeled_consistent_iou_threshold)
                        for gt_anchor_i in anchor_gt_backed_indices
                    )
                    if not conflicts_gt_backed:
                        shielded_anchor_indices.add(int(anchor_i))
                        continue
                dead_anchor_indices.add(int(anchor_i))

            recovered_gt_indices = sorted(
                int(gt_i) for gt_i in (explorer_gt_indices - anchor_gt_indices)
            )
            dead_explorer_indices = sorted(
                int(explorer_i)
                for explorer_i in range(len(explorer_accepted_objects_clean))
                if int(explorer_i) not in explorer_match_by_pred
            )

            triage_anchor_gt_backed_total += int(len(anchor_gt_backed_indices))
            triage_shielded_anchor_total += int(len(shielded_anchor_indices))
            triage_dead_anchor_total += int(len(dead_anchor_indices))
            triage_dead_explorer_total += int(len(dead_explorer_indices))
            triage_recovered_gt_total += int(len(recovered_gt_indices))
            triage_dead_anchor_den_total += int(len(accepted_objects_clean))
            triage_dead_explorer_den_total += int(
                len(explorer_accepted_objects_clean)
            )

            prefix_bbox_groups: List[Dict[str, Any]] = []
            fn_bbox_groups: List[Dict[str, Any]] = []
            prefix_pos: List[int] = []
            prefix_bins: List[int] = []
            matched_gt_for_supervision: set[int] = set()

            prefix_struct_pos: List[int] = []

            fn_count_for_meta = 0

            kept_anchor_objects: List[GTObject] = []
            kept_anchor_new_index_by_old: Dict[int, int] = {}
            dead_anchor_bursts_by_boundary: Dict[int, List[GTObject]] = {}
            kept_anchor_count = 0
            for anchor_i, anchor_obj in enumerate(accepted_objects_clean):
                if int(anchor_i) in dead_anchor_indices:
                    dead_anchor_bursts_by_boundary.setdefault(
                        int(kept_anchor_count), []
                    ).append(anchor_obj)
                    continue
                kept_anchor_new_index_by_old[int(anchor_i)] = int(len(kept_anchor_objects))
                kept_anchor_objects.append(anchor_obj)
                kept_anchor_count += 1

            clean_prefix = _build_canonical_prefix_data(
                tokenizer=tok,
                objects=kept_anchor_objects,
                object_field_order=object_field_order,
            )
            prefix_len_raw_local = int(len(clean_prefix.prefix_token_ids))

            prefix_coord_positions_all = [
                int(i)
                for i, tok_id in enumerate(clean_prefix.prefix_token_ids)
                if int(tok_id) in coord_id_set
            ]
            expected_prefix_coord_slots = int(len(kept_anchor_objects) * 4)
            if len(prefix_coord_positions_all) != expected_prefix_coord_slots:
                raise ValueError(
                    "clean-prefix canonical serialization produced an unexpected number "
                    "of coord tokens for accepted Channel-B bbox objects: "
                    f"got={len(prefix_coord_positions_all)} expected={expected_prefix_coord_slots}"
                )

            matched_clean_indices: List[int] = []
            for pred_i, gt_i in sorted(match.matched_pairs, key=lambda item: int(item[0])):
                if pred_i < 0 or pred_i >= len(accepted_objects_clean):
                    continue
                if gt_i < 0 or gt_i >= len(gts):
                    continue
                if int(pred_i) not in kept_anchor_new_index_by_old:
                    continue

                matched_gt_for_supervision.add(int(gt_i))
                kept_pred_i = int(kept_anchor_new_index_by_old[int(pred_i)])
                matched_clean_indices.append(int(kept_pred_i))

                coord_group = prefix_coord_positions_all[
                    int(kept_pred_i) * 4 : int(kept_pred_i + 1) * 4
                ]
                if len(coord_group) != 4:
                    raise ValueError(
                        "clean-prefix Channel-B expected exactly four coord slots per bbox object"
                    )

                gt_bins = list(gts[gt_i].points_norm1000)
                prefix_bbox_groups.append(
                    {
                        "pos": [int(len(prompt_ids) + int(p)) for p in coord_group],
                        "gt_bins": gt_bins,
                    }
                )
                for local_idx, tbin in zip(coord_group, gt_bins):
                    prefix_pos.append(int(local_idx))
                    prefix_bins.append(int(tbin))

            matched_prefix_objects = [
                _ValueSpanObject(value_span=clean_prefix.object_value_spans[int(i)])
                for i in matched_clean_indices
                if 0 <= int(i) < len(clean_prefix.object_value_spans)
            ]
            prefix_struct_pos = _matched_prefix_structure_positions(
                tokenizer=tok,
                prefix_token_ids=clean_prefix.prefix_token_ids,
                prefix_text=clean_prefix.prefix_text,
                matched_pred_objects=matched_prefix_objects,
            )

            fn_gt_indices_final = [
                i for i in range(len(gts)) if i not in matched_gt_for_supervision
            ]
            matched_for_supervision_total += int(len(matched_gt_for_supervision))
            fn_objs = [gts[i] for i in fn_gt_indices_final]
            fn_object_weights = [
                float(recovered_ground_truth_weight_multiplier)
                if int(gt_i) in recovered_gt_indices
                else 1.0
                for gt_i in fn_gt_indices_final
            ]
            fn_count_for_meta = int(len(fn_objs))
            triage_recovered_gt_den_total += int(len(fn_gt_indices_final))

            append_text = serialize_append_fragment(
                fn_objects=fn_objs,
                prefix_text=clean_prefix.prefix_text,
                object_field_order=object_field_order,
            )
            append_ids = tok.encode(append_text, add_special_tokens=False)

            tail_desc_pos, tail_desc_weights = _desc_tail_positions_and_weights(
                tokenizer=tok,
                token_ids=append_ids,
                object_weights=fn_object_weights,
            )

            y_train_ids = list(clean_prefix.prefix_token_ids) + [
                int(t) for t in append_ids
            ]

            clean_target_text = str(clean_prefix.prefix_text) + str(append_text)
            dead_anchor_suppression_targets, dead_anchor_suppression_boundary_count, dead_anchor_suppression_skipped_no_divergence = (
                _build_dead_anchor_suppression_targets(
                    tokenizer=tok,
                    y_train_ids=y_train_ids,
                    clean_target_text=clean_target_text,
                    accepted_objects_clean=kept_anchor_objects,
                    fn_objects=fn_objs,
                    duplicate_bursts_by_boundary=dead_anchor_bursts_by_boundary,
                    boundary_prefix_texts=clean_prefix.boundary_prefix_texts,
                    object_field_order=object_field_order,
                )
            )
            dup_ul_boundaries_total += int(dead_anchor_suppression_boundary_count)
            dup_dead_anchor_suppression_skipped_no_divergence_total += int(dead_anchor_suppression_skipped_no_divergence)

            if track_monitor_candidates and (
                duplicate_count > 0
                or dup_near_same_desc_pairs > 0
                or dup_near_any_desc_pairs > 0
                or dead_anchor_indices
                or recovered_gt_indices
                or shielded_anchor_indices
            ):
                duplicate_bursts_dump = {
                    str(boundary): [
                        _serialize_monitor_object(obj) for obj in burst
                    ]
                    for boundary, burst in sorted(duplicate_bursts_by_boundary.items())
                }
                match_details: List[Dict[str, Any]] = []
                for pred_i, gt_i in match.matched_pairs:
                    if pred_i < 0 or pred_i >= len(accepted_objects_clean):
                        continue
                    if gt_i < 0 or gt_i >= len(gts):
                        continue
                    match_details.append(
                        {
                            "pred_i": int(pred_i),
                            "gt_i": int(gt_i),
                            "bbox_iou_norm1000": float(
                                _bbox_iou_norm1000_xyxy(
                                    accepted_objects_clean[pred_i].points_norm1000,
                                    gts[gt_i].points_norm1000,
                                )
                            ),
                            "pred_index": int(accepted_objects_clean[pred_i].index),
                            "gt_index": int(gts[gt_i].index),
                            "pred_desc": str(accepted_objects_clean[pred_i].desc),
                            "gt_desc": str(gts[gt_i].desc),
                        }
                    )

                clean_pred_n = float(len(accepted_objects_clean))
                gt_n = float(len(gts))
                matched_n = float(len(match.matched_pairs))
                prec_local = (matched_n / clean_pred_n) if clean_pred_n > 0 else 0.0
                rec_local = (matched_n / gt_n) if gt_n > 0 else 0.0
                f1_local = (
                    (2.0 * prec_local * rec_local / (prec_local + rec_local))
                    if (prec_local + rec_local) > 0.0
                    else 0.0
                )

                monitor_record = _build_stage2_train_monitor_record(
                    sample=sample,
                    gts=gts,
                    preds=parsed_bbox_objects_raw,
                    object_field_order=object_field_order,
                )
                self._stage2_note_train_monitor_candidate(
                    global_step=monitor_step,
                    sample={
                        "sample_id": sample.get("sample_id"),
                        "base_idx": sample.get("base_idx"),
                        "image_id": monitor_record.get("image_id"),
                        "image": sample.get("image"),
                        "images": monitor_record.get("images"),
                        "width": monitor_record.get("width"),
                        "height": monitor_record.get("height"),
                        "messages": sample.get("messages"),
                        "rollout_text": str(getattr(parse, "response_text", "") or ""),
                        "anchor_rollout_text": str(
                            getattr(anchor_view["parse"], "response_text", "") or ""
                        ),
                        "explorer_rollout_text": str(
                            getattr(explorer_view["parse"], "response_text", "") or ""
                        ),
                        "prefix_text": str(clean_prefix.prefix_text),
                        "append_text": str(append_text),
                        "train_text": str(clean_target_text),
                        "gt": monitor_record.get("gt"),
                        "pred": monitor_record.get("pred"),
                        "gt_objects": [
                            _serialize_monitor_object(obj) for obj in gts
                        ],
                        "pred_objects": [
                            _serialize_monitor_object(obj)
                            for obj in parsed_bbox_objects_raw
                        ],
                        "match": {
                            "match_domain": "clean_accepted_vs_gt",
                            "matched_pairs": [
                                [int(pred_i), int(gt_i)]
                                for pred_i, gt_i in match.matched_pairs
                            ],
                            "matched_pair_details": match_details,
                            "fn_gt_indices": [
                                int(idx) for idx in match.fn_gt_indices
                            ],
                            "fp_pred_indices": [
                                int(idx) for idx in match.fp_pred_indices
                            ],
                            "gating_rejections": int(match.gating_rejections),
                        },
                        "duplication": {
                            "raw_bbox_valid": int(len(parsed_bbox_objects_raw)),
                            "clean_accepted": int(len(accepted_objects_clean)),
                            "duplicates": int(duplicate_count),
                            "duplicate_bursts": int(duplicate_burst_count),
                            "near_iou90_pairs_same_desc_count": int(
                                dup_near_same_desc_pairs
                            ),
                            "near_iou90_pairs_any_desc_count": int(
                                dup_near_any_desc_pairs
                            ),
                            "max_desc_count": float(dup_max_desc_count),
                            "saturation_rate": float(dup_saturation_rate),
                            "clean_accepted_objects": [
                                _serialize_monitor_object(obj)
                                for obj in accepted_objects_clean
                            ],
                            "duplicate_bursts_by_boundary": duplicate_bursts_dump,
                        },
                        "explorer": {
                            "clean_accepted_objects": [
                                _serialize_monitor_object(obj)
                                for obj in explorer_accepted_objects_clean
                            ],
                            "matched_pairs": [
                                [int(pred_i), int(gt_i)]
                                for pred_i, gt_i in explorer_match.matched_pairs
                            ],
                            "fn_gt_indices": [
                                int(idx) for idx in explorer_match.fn_gt_indices
                            ],
                            "fp_pred_indices": [
                                int(idx) for idx in explorer_match.fp_pred_indices
                            ],
                        },
                        "triage": {
                            "association_pairs": [
                                [int(anchor_i), int(explorer_i)]
                                for anchor_i, explorer_i in association_pairs
                            ],
                            "anchor_gt_backed_indices": [
                                int(idx) for idx in anchor_gt_backed_indices
                            ],
                            "shielded_anchor_indices": [
                                int(idx) for idx in sorted(shielded_anchor_indices)
                            ],
                            "dead_anchor_indices": [
                                int(idx) for idx in sorted(dead_anchor_indices)
                            ],
                            "dead_explorer_indices": [
                                int(idx) for idx in dead_explorer_indices
                            ],
                            "recovered_gt_indices": [
                                int(idx) for idx in recovered_gt_indices
                            ],
                            "fn_object_weights": [
                                float(weight) for weight in fn_object_weights
                            ],
                        },
                        "stats": {
                            "decode_mode": str(decode_mode),
                            "anchor_decode_mode": str(anchor_view["decode_mode"]),
                            "explorer_decode_mode": str(
                                explorer_view["decode_mode"]
                            ),
                            "parse_invalid_rollout": bool(invalid_rollout),
                            "parse_dropped_invalid": int(parse.dropped_invalid),
                            "parse_dropped_ambiguous": int(parse.dropped_ambiguous),
                            "parse_truncated": bool(parse.truncated),
                            "raw_valid_pred_objects": int(len(parsed_bbox_objects_raw)),
                            "clean_accepted_pred_objects": int(
                                len(accepted_objects_clean)
                            ),
                            "gt_objects": int(len(gts)),
                            "matched": int(len(match.matched_pairs)),
                            "fp_count": int(len(match.fp_pred_indices)),
                            "fn_count": int(len(match.fn_gt_indices)),
                            "precision": float(prec_local),
                            "recall": float(rec_local),
                            "f1": float(f1_local),
                            "dead_anchor_suppression_boundary_count": int(dead_anchor_suppression_boundary_count),
                            "dead_anchor_suppression_skipped_no_divergence": int(
                                dead_anchor_suppression_skipped_no_divergence
                            ),
                        },
                    },
                )

            # FN bbox groups in the appended tail.
            rel_groups = _bbox_groups_from_token_ids(
                token_ids=append_ids, coord_id_set=coord_id_set, gt_objs=fn_objs
            )
            for obj, rel_pos, obj_weight in zip(fn_objs, rel_groups, fn_object_weights):
                fn_bbox_groups.append(
                    {
                        "pos": [
                            int(
                                len(prompt_ids)
                                + int(prefix_len_raw_local)
                                + int(p)
                            )
                            for p in rel_pos
                        ],
                        "gt_bins": list(obj.points_norm1000),
                        "weight": float(obj_weight),
                    }
                )

            t_parse_match_s += time.perf_counter() - t_pm0

            # Teacher-forced encode.
            t_enc0 = time.perf_counter()
            data_for_encode = dict(sample)
            messages = json.loads(json.dumps(sample["messages"]))
            has_assistant = any(
                isinstance(m, dict) and m.get("role") == "assistant" for m in messages
            )

            if has_assistant:
                data_for_encode["messages"] = replace_assistant_response_with_ids(
                    messages, y_train_ids
                )
            else:
                data_for_encode["messages"] = list(messages) + [
                    {"role": "assistant", "content": y_train_ids}
                ]
            try:
                with self._template_train_mode():
                    encoded = template.encode(data_for_encode, return_length=True)
            except MaxLengthError as e:
                max_len = getattr(template, "max_length", None)
                raise MaxLengthError(
                    "stage2-ab teacher-forced encode exceeded max_length="
                    f"{max_len}. SFT forbids truncation; pre-filter long samples or increase global_max_length."
                ) from e
            t_encode_s += time.perf_counter() - t_enc0

            encoded_len = self._extract_encoded_len(encoded)

            enc_ids = encoded.get("input_ids")
            if not isinstance(enc_ids, Sequence):
                raise ValueError("template.encode did not return sequence input_ids")
            enc_ids_list = [int(x) for x in enc_ids]

            # Robustly locate assistant span (prompt_len) even under truncation.
            prompt_len: Optional[int] = None
            train_len_eff: Optional[int] = None
            labels = encoded.get("labels")
            if isinstance(labels, Sequence):
                try:
                    lbl = [int(x) for x in labels]
                    first = next((i for i, x in enumerate(lbl) if int(x) != -100), None)
                    if first is not None:
                        last = max(i for i, x in enumerate(lbl) if int(x) != -100)
                        prompt_len = int(first)
                        train_len_eff = int(last - first + 1)
                except (TypeError, ValueError):
                    prompt_len = None
                    train_len_eff = None

            if prompt_len is None:
                prompt_len = _find_subsequence(enc_ids_list, y_train_ids)
                train_len_eff = int(len(y_train_ids))

            prompt_len = int(prompt_len)
            train_len_eff = int(train_len_eff or 0)
            train_len_eff = max(
                0, min(train_len_eff, int(encoded_len) - int(prompt_len))
            )

            prefix_len_raw = int(prefix_len_raw_local)
            prefix_len_eff = int(min(prefix_len_raw, train_len_eff))

            if int(encoded_len) <= int(prompt_len):
                raise ValueError(
                    "teacher-forced encode produced no assistant span: "
                    f"prompt_len={int(prompt_len)} encoded_len={int(encoded_len)} train_len={int(train_len_eff)}"
                )

            # Sanity: prompt prefix must exactly match the server-provided prompt_token_ids.
            # Without this, coord-position offsets can silently drift and corrupt supervision.
            #
            # In practice, vLLM server-mode can occasionally return prompt_token_ids that do not
            # byte-for-byte match the local teacher-forced encoding (most commonly within the
            # vision token region, e.g. extra `<|image_pad|>` padding). When this happens we do
            # NOT have a safe way to reconcile offsets, so we drop the sample for this step.
            if isinstance(prompt_ids, list) and prompt_ids:
                prompt_ids_int = [int(t) for t in prompt_ids]
                teacher_prefix = enc_ids_list[: len(prompt_ids_int)]
                if teacher_prefix != prompt_ids_int:
                    prompt_tok_mismatch_total += 1
                    mismatch_at = next(
                        (
                            i
                            for i, (a, b) in enumerate(
                                zip(teacher_prefix, prompt_ids_int)
                            )
                            if int(a) != int(b)
                        ),
                        None,
                    )
                    lo = max(0, int(mismatch_at or 0) - 3)
                    hi = min(int(len(prompt_ids_int)), int(mismatch_at or 0) + 4)
                    rank, _world, _dist = self._dist_info()
                    if int(rank) == 0:
                        logger.warning(
                            "stage2-ab Channel-B prompt tokenization mismatch; dropping sample. "
                            "mismatch_at=%s teacher_ids=%s server_ids=%s",
                            int(mismatch_at) if mismatch_at is not None else None,
                            teacher_prefix[lo:hi],
                            prompt_ids_int[lo:hi],
                        )
                    continue
                if int(prompt_len) != int(len(prompt_ids_int)):
                    prompt_tok_mismatch_total += 1
                    rank, _world, _dist = self._dist_info()
                    if int(rank) == 0:
                        logger.warning(
                            "stage2-ab Channel-B prompt_len mismatch vs server prompt_token_ids length; "
                            "dropping sample. prompt_len=%s server_prompt_len=%s",
                            int(prompt_len),
                            int(len(prompt_ids_int)),
                        )
                    continue


            prompt_ids_local = enc_ids_list[:prompt_len]
            delta_prompt = int(prompt_len) - int(len(prompt_ids))

            # Shift bbox groups based on local prompt_len, and drop any out-of-range groups.
            def _shift_groups(
                groups: Sequence[Mapping[str, Any]], *, lower: int, upper: int
            ) -> List[Dict[str, Any]]:
                out: List[Dict[str, Any]] = []
                for g in groups:
                    if not isinstance(g, Mapping):
                        continue
                    pos = g.get("pos")
                    gb = g.get("gt_bins")
                    weight_raw = g.get("weight", 1.0)
                    if not isinstance(pos, Sequence) or not isinstance(gb, Sequence):
                        continue
                    if len(pos) != 4 or len(gb) != 4:
                        continue
                    try:
                        pos_i = [int(p) for p in pos]
                        gb_i = [int(x) for x in gb]
                        weight_i = float(weight_raw)
                    except (TypeError, ValueError):
                        continue
                    pos_i = [int(p + delta_prompt) for p in pos_i]
                    if any(p < int(lower) or p >= int(upper) for p in pos_i):
                        raise ValueError(
                            "stage2-ab bbox group pos escaped expected span after prompt shift (possible truncation/misalignment). "
                            f"pos={pos_i} span=[{int(lower)},{int(upper)}) delta_prompt={int(delta_prompt)}"
                        )
                    if any(p >= int(encoded_len) for p in pos_i):
                        raise ValueError(
                            "stage2-ab bbox group pos exceeds encoded_len after prompt shift (possible truncation/misalignment). "
                            f"pos={pos_i} encoded_len={int(encoded_len)}"
                        )
                    out.append(
                        {
                            "pos": pos_i,
                            "gt_bins": gb_i,
                            "weight": float(weight_i),
                        }
                    )
                return out

            bbox_groups_prefix = _shift_groups(
                prefix_bbox_groups,
                lower=int(prompt_len),
                upper=int(prompt_len + prefix_len_eff),
            )
            bbox_groups_fn = _shift_groups(
                fn_bbox_groups,
                lower=int(prompt_len + prefix_len_eff),
                upper=int(prompt_len + train_len_eff),
            )

            # Tail desc spans for CE weighting (relative to tail ids).
            tail_desc_pos_eff: List[int] = []
            tail_desc_weights_eff: List[float] = []
            tail_cap = max(0, int(train_len_eff) - int(prefix_len_eff))

            # Stop/closure supervision is active (no stop-neutral masking). If
            # deterministic closure-marker resolution fails, keep the sample and
            # fall back to the normal FN-tail supervision path instead of dropping it.
            tail_ignore_pos_eff: List[int] = []
            assistant_span_ids = enc_ids_list[
                int(prompt_len) : int(prompt_len) + int(train_len_eff)
            ]
            try:
                tail_closure_pos_eff = _stage2_ab_tail_closure_positions(
                    tokenizer=tok,
                    assistant_span_ids=assistant_span_ids,
                    prefix_len=int(prefix_len_eff),
                )
            except ValueError:
                closure_supervision_drop_total += 1
                tail_closure_pos_eff = []

            for rel, weight in zip(tail_desc_pos, tail_desc_weights):
                try:
                    rel_i = int(rel)
                    weight_f = float(weight)
                except (TypeError, ValueError):
                    continue
                if 0 <= rel_i < tail_cap:
                    tail_desc_pos_eff.append(rel_i)
                    tail_desc_weights_eff.append(weight_f)

            invalid_rollout_total += int(invalid_rollout)

            meta_entry: Dict[str, Any] = {
                "stage2_channel": "B",
                "stage2_invalid_rollout": int(invalid_rollout),
                "rollout_seed_base": int(seed_base),
                "prompt_len": int(prompt_len),
                "prompt_ids": prompt_ids_local,
                "rollout_len": int(len(parse.response_token_ids)),
                "prefix_len": int(prefix_len_eff),
                "train_len": int(train_len_eff),
                "encoded_len": int(encoded_len),
                "decode_mode": str(decode_mode),
                "parse_dropped_invalid": int(parse.dropped_invalid),
                "parse_dropped_ambiguous": int(parse.dropped_ambiguous),
                "parse_truncated": bool(parse.truncated),
                "drop_invalid_total": int(n_drop_invalid),
                "valid_pred_objects": int(len(parsed_bbox_objects_raw)),
                "matched_for_supervision": int(len(matched_gt_for_supervision)),
                "matched_maskiou_sum": float(match.matched_maskiou_sum),
                "matched_maskiou_count": int(match.matched_maskiou_count),
                "gt_objects": int(len(gts)),
                "fn_count": int(fn_count_for_meta),
                "gating_rejections": int(match.gating_rejections),
                "excluded_from_supervision": int(0),
                "prefix_coord_pos": prefix_pos,
                "prefix_coord_target_bins": prefix_bins,
                "prefix_struct_pos": [int(p) for p in prefix_struct_pos],
                "tail_closure_pos": [int(p) for p in tail_closure_pos_eff],
                "tail_ignore_pos": tail_ignore_pos_eff,
                "tail_desc_pos": tail_desc_pos_eff,
                "tail_desc_weights": [float(w) for w in tail_desc_weights_eff],
                "fn_object_weights": [float(w) for w in fn_object_weights],
                "bbox_groups_prefix": bbox_groups_prefix,
                "bbox_groups_fn": bbox_groups_fn,
                "anchor_decode_mode": str(anchor_view["decode_mode"]),
                "explorer_decode_mode": str(explorer_view["decode_mode"]),
                "anchor_gt_backed_indices": [
                    int(idx) for idx in anchor_gt_backed_indices
                ],
                "shielded_anchor_indices": [
                    int(idx) for idx in sorted(shielded_anchor_indices)
                ],
                "dead_anchor_indices": [
                    int(idx) for idx in sorted(dead_anchor_indices)
                ],
                "dead_explorer_indices": [
                    int(idx) for idx in dead_explorer_indices
                ],
                "recovered_gt_indices": [
                    int(idx) for idx in recovered_gt_indices
                ],
                "dead_anchor_suppression_targets": [
                    {
                        "boundary": int(item["boundary"]),
                        "rel_pos": int(item["rel_pos"]),
                        "token_id": int(item["token_id"]),
                    }
                    for item in dead_anchor_suppression_targets
                ],
                "dead_anchor_suppression_boundary_count": int(
                    dead_anchor_suppression_boundary_count
                ),
                "dead_anchor_suppression_skipped_no_divergence": int(
                    dead_anchor_suppression_skipped_no_divergence
                ),
            }

            segments.append((encoded, meta_entry, int(encoded_len)))
            if not packing_enabled:
                encoded_batch.append(encoded)
                meta_unpacked.append(meta_entry)

        from swift.llm import to_device

        batch_metrics: Dict[str, float] = {
            "stage2/channel_a": float(0.0),
            "stage2/channel_b": float(1.0),
            "stage2/invalid_rollout": float(invalid_rollout_total),
            "stage2_ab/channel_b/invalid_rollout": float(invalid_rollout_total),
            "stage2_ab/channel_b/closure_supervision/N_drop": float(
                closure_supervision_drop_total
            ),
            "stage2_ab/channel_b/prompt_tok_mismatch": float(prompt_tok_mismatch_total),
            "stage2_ab/channel_b/prompt_tok_mismatch_rate": float(
                (float(prompt_tok_mismatch_total) / float(len(anchor_rollout_results)))
                if len(anchor_rollout_results) > 0
                else 0.0
            ),
            "stage2_ab/channel_b/_prompt_tok_mismatch_num": float(prompt_tok_mismatch_total),
            "stage2_ab/channel_b/_prompt_tok_mismatch_den": float(len(anchor_rollout_results)),
            "stage2/drop_poly": float(drop_poly_total),
            "stage2/drop_unknown": float(drop_unknown_total),
            "stage2/drop_bbox_invalid": float(drop_bbox_invalid_total),
            "rollout/seed_base": float(seed_base),
            "rollout/backend_hf": float(1.0 if backend == "hf" else 0.0),
            "rollout/backend_vllm": float(1.0 if backend == "vllm" else 0.0),
            "rollout/decode_mode_greedy": float(
                1.0 if anchor_decode_request.decode_mode == "greedy" else 0.0
            ),
            "rollout/decode_mode_beam": float(
                1.0 if anchor_decode_request.decode_mode == "beam" else 0.0
            ),
            "rollout/hf_seeded_global": float(hf_seeded_global),
            "rollout/temperature": float(anchor_decode_request.temperature),
            "rollout/top_p": float(anchor_decode_request.top_p),
            "rollout/top_k": float(anchor_decode_request.top_k),
            "rollout/do_sample": float(1.0 if do_sample else 0.0),
            "rollout/max_new_tokens": float(max_new_tokens),
            "rollout/num_beams": float(num_beams),
            "rollout/repetition_penalty": float(repetition_penalty),
            "rollout/anchor_temperature": float(anchor_decode_request.temperature),
            "rollout/anchor_top_p": float(anchor_decode_request.top_p),
            "rollout/anchor_top_k": float(anchor_decode_request.top_k),
            "rollout/explorer_temperature": float(explorer_decode_request.temperature),
            "rollout/explorer_top_p": float(explorer_decode_request.top_p),
            "rollout/explorer_top_k": float(explorer_decode_request.top_k),
            # Per-policy rollout split over the current Channel-B raw rollout window.
            "rollout/anchor/pred_objects": float(anchor_pred_objects_total),
            "rollout/anchor/valid_pred_objects": float(anchor_valid_pred_objects_total),
            "rollout/anchor/parse_truncated_rate": float(
                float(anchor_parse_truncated_total) / float(len(anchor_rollout_results))
                if len(anchor_rollout_results) > 0
                else 0.0
            ),
            "rollout/anchor/gen_new_tokens_mean": float(
                sum(anchor_gen_new_token_lens) / len(anchor_gen_new_token_lens)
                if anchor_gen_new_token_lens
                else 0.0
            ),
            "rollout/anchor/gen_new_tokens_p90": float(
                _percentile(anchor_gen_new_token_lens, 90.0)
            ),
            "rollout/anchor/near_iou90_any": float(anchor_near_any_desc_pairs_total),
            "rollout/anchor/near_iou90_same": float(anchor_near_same_desc_pairs_total),
            "rollout/explorer/pred_objects": float(explorer_pred_objects_total),
            "rollout/explorer/valid_pred_objects": float(
                explorer_valid_pred_objects_total
            ),
            "rollout/explorer/parse_truncated_rate": float(
                float(explorer_parse_truncated_total)
                / float(len(explorer_rollout_results))
                if len(explorer_rollout_results) > 0
                else 0.0
            ),
            "rollout/explorer/gen_new_tokens_mean": float(
                sum(explorer_gen_new_token_lens) / len(explorer_gen_new_token_lens)
                if explorer_gen_new_token_lens
                else 0.0
            ),
            "rollout/explorer/gen_new_tokens_p90": float(
                _percentile(explorer_gen_new_token_lens, 90.0)
            ),
            "rollout/explorer/near_iou90_any": float(
                explorer_near_any_desc_pairs_total
            ),
            "rollout/explorer/near_iou90_same": float(
                explorer_near_same_desc_pairs_total
            ),
            "rollout/explorer/temperature": float(explorer_decode_request.temperature),
            "rollout/explorer/do_sample": float(1.0 if do_sample else 0.0),
            "rollout/explorer/top_p": float(explorer_decode_request.top_p),
            "rollout/explorer/top_k": float(explorer_decode_request.top_k),
            "rollout/parse_truncated": float(parse_truncated_total),
            "rollout/parse_truncated_rate": float(
                (float(parse_truncated_total) / float(len(anchor_rollout_results)))
                if len(anchor_rollout_results) > 0
                else 0.0
            ),
            "rollout/_parse_truncated_num": float(parse_truncated_total),
            "rollout/_parse_truncated_den": float(len(anchor_rollout_results)),
            "dup/max_desc_count": float(
                dup_max_desc_count_sum / float(dup_metric_samples)
                if dup_metric_samples > 0
                else 0.0
            ),
            "dup/saturation_rate": float(
                dup_saturation_rate_sum / float(dup_metric_samples)
                if dup_metric_samples > 0
                else 0.0
            ),
            "dup/near_iou90_pairs_same_desc_count": float(
                dup_near_same_desc_pairs_total
            ),
            "dup/near_iou90_pairs_any_desc_count": float(
                dup_near_any_desc_pairs_total
            ),
            "stage2_ab/channel_b/dup/N_raw_bbox_valid": float(dup_raw_bbox_valid_total),
            "stage2_ab/channel_b/dup/N_clean_accepted": float(
                dup_clean_accepted_total
            ),
            "stage2_ab/channel_b/dup/N_duplicates": float(dup_duplicates_total),
            "stage2_ab/channel_b/dup/N_duplicate_bursts": float(
                dup_duplicate_bursts_total
            ),
            "stage2_ab/channel_b/dup/N_ul_boundaries": float(
                dup_ul_boundaries_total
            ),
            "stage2_ab/channel_b/dup/N_dead_anchor_suppression_skipped_no_divergence": float(
                dup_dead_anchor_suppression_skipped_no_divergence_total
            ),
            "train/triage/gt_backed_count": float(
                triage_anchor_gt_backed_total
            ),
            "train/triage/unlabeled_consistent_count": float(
                triage_shielded_anchor_total
            ),
            "train/triage/dead_anchor_count": float(
                triage_dead_anchor_total
            ),
            "train/triage/explorer_only_dead_count": float(
                triage_dead_explorer_total
            ),
            "train/triage/recovered_ground_truth_count": float(
                triage_recovered_gt_total
            ),
            "train/triage/recovered_ground_truth_rate_num": float(
                triage_recovered_gt_total
            ),
            "train/triage/recovered_ground_truth_rate_den": float(
                triage_recovered_gt_den_total
            ),
            "train/triage/recovered_ground_truth_rate": float(
                float(triage_recovered_gt_total)
                / float(triage_recovered_gt_den_total)
                if triage_recovered_gt_den_total > 0
                else 0.0
            ),
            "train/triage/dead_anchor_rate_num": float(
                triage_dead_anchor_total
            ),
            "train/triage/dead_anchor_rate_den": float(
                triage_dead_anchor_den_total
            ),
            "train/triage/dead_anchor_rate": float(
                float(triage_dead_anchor_total) / float(triage_dead_anchor_den_total)
                if triage_dead_anchor_den_total > 0
                else 0.0
            ),
            "train/triage/explorer_only_dead_rate_num": float(
                triage_dead_explorer_total
            ),
            "train/triage/explorer_only_dead_rate_den": float(
                triage_dead_explorer_den_total
            ),
            "train/triage/explorer_only_dead_rate": float(
                float(triage_dead_explorer_total)
                / float(triage_dead_explorer_den_total)
                if triage_dead_explorer_den_total > 0
                else 0.0
            ),
            # Direct supervision-efficiency gauge for strict-valid predictions.
            "rollout/matched_for_supervision_count": float(
                matched_for_supervision_total
            ),
            "rollout/matched_for_supervision_over_valid_pred": float(
                float(matched_for_supervision_total) / float(strict_valid_pred_total)
                if strict_valid_pred_total > 0
                else 0.0
            ),
            "time/rollout_generate_s": float(t_gen_s),
            "time/rollout_parse_match_s": float(t_parse_match_s),
            "time/rollout_teacher_encode_s": float(t_encode_s),
        }

        batch_metrics["stage2_ab/channel_b/strict_drop/N_valid_pred"] = float(
            strict_valid_pred_total
        )
        batch_metrics["stage2_ab/channel_b/strict_drop/N_drop_invalid"] = float(
            strict_drop_invalid_total
        )
        for rk, rv in strict_drop_by_reason_total.items():
            try:
                rvi = int(rv)
            except (TypeError, ValueError):
                continue
            if rvi <= 0:
                continue
            batch_metrics[f"stage2_ab/channel_b/strict_drop/reason/{str(rk)}"] = float(
                rvi
            )

        if bool(_segments_only):
            return segments, batch_metrics

        if packing_enabled:
            self._stage2_append_post_rollout_segments(channel="B", segments=segments)

            t_pack0 = time.perf_counter()
            selected, pack_metrics = self._stage2_pop_post_rollout_pack(channel="B")
            with self._template_packing_enabled():
                packed = template.data_collator([enc for enc, _, _ in selected])
            batch = to_device(packed, self.model.device)
            self._assert_single_packed_forward(batch, where="stage2_ab/packed_forward")
            batch["_rollout_matching_meta"] = [m for _, m, _ in selected]

            batch_metrics.update(pack_metrics)
            batch_metrics["time/post_rollout_pack_s"] = float(
                time.perf_counter() - t_pack0
            )
            self._merge_rollout_matching_batch_metrics(batch, batch_metrics)
            batch["_stage2_ab_channel"] = "B"
            return batch

        if not encoded_batch:
            raise ValueError(
                "stage2-ab Channel-B produced no usable segments (all samples were skipped/dropped); "
                "this usually indicates prompt-token mismatch or another strict sample-level gating failure."
            )

        with self._template_packing_disabled():
            batch = to_device(template.data_collator(encoded_batch), self.model.device)
        batch["_rollout_matching_meta"] = meta_unpacked
        self._merge_rollout_matching_batch_metrics(batch, batch_metrics)
        batch["_stage2_ab_channel"] = "B"
        return batch

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        channel = inputs.pop("_stage2_ab_channel", None)
        if channel not in {"A", "B"}:
            channel = "B"

        meta = inputs.pop("_rollout_matching_meta", None)
        if not isinstance(meta, list):
            raise ValueError("stage2-ab trainer requires _rollout_matching_meta")

        batch_metrics = inputs.pop("_rollout_matching_batch_metrics", None)

        input_ids = inputs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor):
            raise ValueError("stage2-ab compute_loss requires input_ids tensor")

        coord_token_ids = self._get_coord_token_ids()
        coord_id_set = set(int(i) for i in coord_token_ids if int(i) >= 0)
        coord_ids_t = torch.tensor(
            coord_token_ids, device=input_ids.device, dtype=torch.long
        )

        # Config.
        n_softctx_iter = int(self._ab_get("n_softctx_iter", 2) or 2)
        n_softctx_iter = max(1, n_softctx_iter)

        softctx_grad_mode = str(
            self._ab_get("softctx_grad_mode", "unroll") or "unroll"
        ).strip().lower()
        if softctx_grad_mode not in {"unroll", "em_detach"}:
            raise ValueError(
                "stage2_ab.softctx_grad_mode must be one of {'unroll','em_detach'}"
            )

        coord_ctx_embed_mode = str(
            self._ab_get("coord_ctx_embed_mode", "st") or "st"
        ).strip().lower()
        if coord_ctx_embed_mode not in {"soft", "st", "hard"}:
            raise ValueError(
                "stage2_ab.coord_ctx_embed_mode must be one of {'soft','st','hard'}"
            )

        coord_decode_mode = str(
            self._ab_get("coord_decode_mode", "exp") or "exp"
        ).strip().lower()
        if coord_decode_mode not in {"exp", "st"}:
            raise ValueError(
                "stage2_ab.coord_decode_mode must be one of {'exp','st'}"
            )

        pipeline_manifest = getattr(self, "stage2_pipeline_manifest", None)
        objective_specs = (
            pipeline_manifest.get("objective", [])
            if isinstance(pipeline_manifest, Mapping)
            else []
        )
        diagnostic_specs = (
            pipeline_manifest.get("diagnostics", [])
            if isinstance(pipeline_manifest, Mapping)
            else []
        )

        if not objective_specs:
            raise ValueError(
                "stage2-ab trainer requires stage2_ab.pipeline.objective (pipeline-only contract). "
                "Ensure `sft.py` injected stage2_pipeline_manifest from your config."
            )

        def _module_weight(
            specs: Sequence[Mapping[str, Any]],
            *,
            name: str,
            channel_name: str,
            default: float,
        ) -> float:
            for spec in specs:
                if not isinstance(spec, Mapping):
                    continue
                if str(spec.get("name", "") or "").strip() != name:
                    continue
                enabled = bool(spec.get("enabled", True))
                channels_raw = spec.get("channels", ["A", "B"])
                channels: set[str] = set()
                if isinstance(channels_raw, Sequence) and not isinstance(
                    channels_raw, (str, bytes)
                ):
                    for ch in channels_raw:
                        channels.add(str(ch).strip().upper())
                if not channels:
                    channels = {"A", "B"}
                if (not enabled) or (str(channel_name).upper() not in channels):
                    return 0.0
                try:
                    w = float(spec.get("weight", 1.0) or 0.0)
                except (TypeError, ValueError):
                    w = 0.0
                return max(0.0, w)
            return float(default)

        def _module_config(
            specs: Sequence[Mapping[str, Any]],
            *,
            name: str,
        ) -> Mapping[str, Any]:
            for spec in specs:
                if not isinstance(spec, Mapping):
                    continue
                if str(spec.get("name", "") or "").strip() != name:
                    continue
                cfg = spec.get("config", {})
                if isinstance(cfg, Mapping):
                    return cfg
                return {}
            return {}

        def _module_application(
            specs: Sequence[Mapping[str, Any]],
            *,
            name: str,
        ) -> Mapping[str, Any]:
            for spec in specs:
                if not isinstance(spec, Mapping):
                    continue
                if str(spec.get("name", "") or "").strip() != name:
                    continue
                app = spec.get("application", {})
                if isinstance(app, Mapping):
                    return app
                return {}
            return {}

        token_ce_module_w = _module_weight(
            objective_specs,
            name="token_ce",
            channel_name=channel,
            default=0.0,
        )
        bbox_geo_module_w = _module_weight(
            objective_specs,
            name="bbox_geo",
            channel_name=channel,
            default=0.0,
        )
        bbox_size_aux_module_w = _module_weight(
            objective_specs,
            name="bbox_size_aux",
            channel_name=channel,
            default=0.0,
        )
        dead_anchor_suppression_module_w = _module_weight(
            objective_specs,
            name="loss_dead_anchor_suppression",
            channel_name=channel,
            default=0.0,
        )
        coord_reg_module_w = _module_weight(
            objective_specs,
            name="coord_reg",
            channel_name=channel,
            default=0.0,
        )
        coord_diag_enabled = (
            _module_weight(
                diagnostic_specs,
                name="coord_diag",
                channel_name=channel,
                default=0.0,
            )
            > 0.0
        )

        token_module_cfg = _module_config(objective_specs, name="token_ce")
        bbox_module_cfg = _module_config(objective_specs, name="bbox_geo")
        bbox_size_aux_module_cfg = _module_config(objective_specs, name="bbox_size_aux")
        coord_module_cfg = _module_config(objective_specs, name="coord_reg")
        token_module_app = _module_application(objective_specs, name="token_ce")
        bbox_module_app = _module_application(objective_specs, name="bbox_geo")
        bbox_size_aux_module_app = _module_application(objective_specs, name="bbox_size_aux")
        coord_module_app = _module_application(objective_specs, name="coord_reg")

        token_cfg = token_module_cfg if isinstance(token_module_cfg, Mapping) else {}
        bbox_cfg = bbox_module_cfg if isinstance(bbox_module_cfg, Mapping) else {}
        bbox_size_aux_cfg = (
            bbox_size_aux_module_cfg
            if isinstance(bbox_size_aux_module_cfg, Mapping)
            else {}
        )
        coord_cfg = coord_module_cfg if isinstance(coord_module_cfg, Mapping) else {}
        token_preset = str(token_module_app.get("preset", "") or "").strip()
        bbox_geo_preset = str(bbox_module_app.get("preset", "") or "").strip()
        bbox_size_aux_preset = str(
            bbox_size_aux_module_app.get("preset", "") or ""
        ).strip()
        coord_reg_preset = str(coord_module_app.get("preset", "") or "").strip()

        def _token_ce_targets(preset: str) -> tuple[bool, bool]:
            if str(channel).upper() != "A":
                return False, False
            if preset == "anchor_text_plus_final_struct":
                return True, int(n_softctx_iter) > 1
            if preset == "anchor_text_only":
                return True, False
            if preset == "rollout_text_only":
                return False, False
            raise ValueError(f"Unsupported token_ce application preset: {preset!r}")

        def _coord_targets(preset: str) -> tuple[bool, bool]:
            if str(channel).upper() != "A":
                return False, False
            if preset == "anchor_if_single_iter_else_final":
                return int(n_softctx_iter) <= 1, int(n_softctx_iter) > 1
            if preset == "anchor_only":
                return True, False
            if preset == "final_only":
                return False, True
            if preset == "anchor_and_final":
                return True, True
            raise ValueError(f"Unsupported coord/bbox application preset: {preset!r}")

        def _cfg_float(
            cfg: Mapping[str, Any],
            *,
            keys: Sequence[str],
            default: float,
            min_value: Optional[float] = None,
        ) -> float:
            value = float(default)
            for key in keys:
                if key in cfg and cfg.get(key) is not None:
                    try:
                        value = float(cfg.get(key))
                    except (TypeError, ValueError):
                        value = float(default)
                    break
            if min_value is not None:
                value = max(float(min_value), value)
            return float(value)

        token_desc_ce_weight = _cfg_float(
            token_cfg,
            keys=("desc_ce_weight",),
            default=1.0,
            min_value=0.0,
        )

        self_context_struct_ce_weight = _cfg_float(
            token_cfg,
            keys=("struct_ce_weight",),
            default=0.1,
            min_value=0.0,
        )
        fn_desc_ce_weight = _cfg_float(
            token_cfg,
            keys=("rollout_fn_desc_weight",),
            default=token_desc_ce_weight,
            min_value=0.0,
        )
        bbox_smoothl1_w = _cfg_float(
            bbox_cfg,
            keys=("smoothl1_weight",),
            default=1.0,
            min_value=0.0,
        )
        bbox_ciou_w = _cfg_float(
            bbox_cfg,
            keys=("ciou_weight",),
            default=1.0,
            min_value=0.0,
        )
        bbox_log_wh_w = _cfg_float(
            bbox_size_aux_cfg,
            keys=("log_wh_weight",),
            default=0.0,
            min_value=0.0,
        )
        bbox_oversize_w = _cfg_float(
            bbox_size_aux_cfg,
            keys=("oversize_penalty_weight",),
            default=0.0,
            min_value=0.0,
        )

        # Optional coord-distribution losses/regularizers (multi-peak stability).
        coord_ce_w = _cfg_float(
            coord_cfg,
            keys=("coord_ce_weight",),
            default=0.0,
            min_value=0.0,
        )
        coord_el1_w = _cfg_float(
            coord_cfg,
            keys=("coord_el1_weight",),
            default=0.0,
            min_value=0.0,
        )
        coord_ehuber_w = _cfg_float(
            coord_cfg,
            keys=("coord_ehuber_weight",),
            default=0.0,
            min_value=0.0,
        )
        # Entropy regularizer (sign controls direction): +w increases entropy, -w sharpens.
        coord_entropy_w = _cfg_float(
            coord_cfg,
            keys=("coord_entropy_weight",),
            default=0.0,
        )

        # Coord-vocab gate: encourage coord slots to place probability mass on coord tokens
        # rather than arbitrary text/number tokens (prevents "wrong_arity" rollouts).
        coord_gate_w = _cfg_float(
            coord_cfg,
            keys=("coord_gate_weight",),
            default=0.0,
            min_value=0.0,
        )

        text_gate_w = _cfg_float(
            coord_cfg,
            keys=("text_gate_weight",),
            default=0.0,
            min_value=0.0,
        )

        temperature = float(self._ab_get("softctx_temperature", 1.0) or 1.0)
        if temperature <= 0:
            raise ValueError(f"softctx_temperature must be > 0; got {temperature}")

        coord_soft_ce_w = _cfg_float(
            coord_cfg,
            keys=("soft_ce_weight",),
            default=0.0,
            min_value=0.0,
        )
        coord_w1_w = _cfg_float(
            coord_cfg,
            keys=("w1_weight",),
            default=0.0,
            min_value=0.0,
        )
        run_a1_text, run_a2_text = _token_ce_targets(token_preset)
        run_a1_bbox_geo, run_a2_bbox_geo = _coord_targets(bbox_geo_preset)
        run_a1_bbox_size_aux, run_a2_bbox_size_aux = _coord_targets(
            bbox_size_aux_preset
        )
        run_a1_coord_reg, run_a2_coord_reg = _coord_targets(coord_reg_preset)

        # Always compute logits; do not rely on model.loss.
        ignored_keys = {
            "labels",
            "compute_loss_func",
            "loss_scale",
            "text_position_ids",
            "channel",
            "logits_to_keep",
        }
        packing_enabled = bool(self._packing_enabled())
        core_model, inputs_for_model, model_type = prepare_forward_inputs(
            model=model,
            inputs=inputs,
            ignored_keys=tuple(ignored_keys),
            packing_enabled=packing_enabled,
            where="stage2-ab",
        )

        # Debug-only: validate packing boundaries against meta to catch leakage/misalignment.
        debug_asserts = str(os.environ.get("COORDEXP_DEBUG_STAGE2_ASSERTS", "") or "").strip()
        if packing_enabled and debug_asserts not in {"", "0"}:
            cu = inputs.get("cu_seq_lens_q")
            if not isinstance(cu, torch.Tensor) or cu.ndim != 1:
                raise ValueError(
                    "COORDEXP_DEBUG_STAGE2_ASSERTS=1: expected cu_seq_lens_q[segments+1] for packed batch"
                )
            cu_list = [int(x) for x in cu.detach().cpu().tolist()]
            seqlen_in = int(input_ids.shape[1])
            if not cu_list or cu_list[0] != 0 or cu_list[-1] != seqlen_in:
                raise ValueError(
                    f"Invalid cu_seq_lens_q boundaries: {cu_list} (expected 0..{seqlen_in})"
                )
            if any(b <= a for a, b in zip(cu_list, cu_list[1:])):
                raise ValueError(
                    f"cu_seq_lens_q must be strictly increasing; got {cu_list}"
                )
            if len(meta) != len(cu_list) - 1:
                raise ValueError(
                    f"meta segments ({len(meta)}) must match cu_seq_lens_q-1 ({len(cu_list)-1})"
                )
            for i, m in enumerate(meta):
                exp = int(m.get("encoded_len") or 0)
                got = int(cu_list[i + 1] - cu_list[i])
                if exp != got:
                    raise ValueError(
                        f"packed segment {i} length mismatch: meta.encoded_len={exp} vs cu_seq_lens_q diff={got}"
                    )
                seg_chan = m.get("stage2_channel")
                if seg_chan is not None and str(seg_chan) != str(channel):
                    raise ValueError(
                        f"packed segment {i} has stage2_channel={seg_chan!r} but batch channel={channel!r}"
                    )

        t_fwd0 = time.perf_counter()
        outputs = None
        logits_a1: Optional[torch.Tensor] = None

        if channel == "A":
            embed = core_model.get_input_embeddings()
            if embed is None:
                raise ValueError("core_model.get_input_embeddings() returned None")

            # Build coord-slot update indices (segment-aware for packing).
            def _collect_coord_slots() -> Tuple[torch.Tensor, torch.Tensor]:
                bsz, seqlen = input_ids.shape
                b_list: List[int] = []
                p_list: List[int] = []
                if len(meta) == bsz:
                    for b in range(bsz):
                        m = meta[b]
                        prompt_len = int(m.get("prompt_len", 0))
                        train_len = int(m.get("train_len", 0))
                        start = max(1, prompt_len)
                        end = min(seqlen, prompt_len + train_len)
                        for p in range(start, end):
                            if int(input_ids[b, p].item()) in coord_id_set:
                                b_list.append(int(b))
                                p_list.append(int(p))
                    return (
                        torch.tensor(b_list, device=input_ids.device, dtype=torch.long),
                        torch.tensor(p_list, device=input_ids.device, dtype=torch.long),
                    )

                if bsz != 1:
                    raise ValueError("packed-mode meta requires bsz==1")
                offset = 0
                for seg in meta:
                    encoded_len = int(seg.get("encoded_len") or 0)
                    if encoded_len <= 0:
                        raise ValueError("packed-mode segment missing encoded_len")
                    prompt_len = int(seg.get("prompt_len", 0))
                    train_len = int(seg.get("train_len", 0))
                    seg_start = offset
                    seg_end = offset + encoded_len
                    start = max(seg_start + 1, seg_start + prompt_len)
                    end = min(seg_end, seg_start + prompt_len + train_len)
                    for p in range(start, end):
                        if int(input_ids[0, p].item()) in coord_id_set:
                            b_list.append(0)
                            p_list.append(int(p))
                    offset += encoded_len
                return (
                    torch.tensor(b_list, device=input_ids.device, dtype=torch.long),
                    torch.tensor(p_list, device=input_ids.device, dtype=torch.long),
                )

            b_slots, p_slots = _collect_coord_slots()
            coord_table = embed(coord_ids_t)
            coord_table_f = coord_table.float()

            logits_prev: Optional[torch.Tensor] = None
            for it in range(int(n_softctx_iter)):
                detach_exp_emb = bool(softctx_grad_mode == "em_detach")
                if softctx_grad_mode == "unroll":
                    ctx = torch.enable_grad()
                else:
                    last_it = int(n_softctx_iter) - 1
                    # Channel-A anchors CE to the A1 (it==0) logits. If we run A1 under
                    # torch.no_grad(), CE has no gradient and the model quickly drifts in
                    # structure/format tokens (e.g., emitting bbox_3d/bbox_d keys).
                    grad_on = bool(it == 0 or it == last_it)
                    ctx = torch.enable_grad() if grad_on else torch.no_grad()

                with ctx:
                    base_embeds = embed(input_ids)
                    embeds = base_embeds
                    if it > 0 and logits_prev is not None and p_slots.numel() > 0:
                        logit_pos = (p_slots - 1).clamp(min=0)
                        logits_next = logits_prev[b_slots, logit_pos]
                        coord_logits = logits_next.index_select(dim=-1, index=coord_ids_t)
                        probs = torch.softmax(coord_logits.float() / temperature, dim=-1)
                        soft_emb = probs @ coord_table_f
                        hard_ids = coord_logits.argmax(dim=-1)
                        hard_emb = coord_table_f.index_select(0, hard_ids)
                        if coord_ctx_embed_mode == "soft":
                            exp_emb = soft_emb
                        elif coord_ctx_embed_mode == "hard":
                            exp_emb = hard_emb
                        else:
                            exp_emb = hard_emb + (soft_emb - soft_emb.detach())
                        exp_emb = exp_emb.to(base_embeds.dtype)
                        if detach_exp_emb:
                            exp_emb = exp_emb.detach()
                        embeds = base_embeds.clone()
                        embeds[b_slots, p_slots] = exp_emb

                    fwd_inputs = dict(inputs_for_model)
                    fwd_inputs.pop("input_ids", None)
                    fwd_inputs["inputs_embeds"] = embeds
                    out = run_no_cache_forward(model=model, inputs_for_model=fwd_inputs)
                    if not hasattr(out, "logits") or out.logits is None:
                        raise ValueError("model did not return logits")
                    if getattr(out, "past_key_values", None) is not None:
                        raise ValueError("past_key_values must be None for Channel-A")

                    logits_prev = out.logits
                    outputs = out
                    if it == 0:
                        logits_a1 = out.logits

        else:
            outputs = run_no_cache_forward(model=model, inputs_for_model=inputs_for_model)
            logits_a1 = getattr(outputs, "logits", None) if outputs is not None else None

        t_fwd_s = time.perf_counter() - t_fwd0

        if outputs is None or outputs.logits is None:
            raise ValueError("model did not return logits")
        logits = outputs.logits
        assert_unsliced_logits(
            logits=logits,
            input_ids=input_ids,
            where="stage2-ab training",
        )

        if channel == "A":
            if logits_a1 is None:
                raise ValueError("Channel-A did not capture A1 logits for CE anchoring")
            if logits_a1.shape != logits.shape:
                raise ValueError("Channel-A A1 logits shape mismatch vs final logits")
            logits_ce = logits_a1
        else:
            logits_ce = logits

        bsz, seq_len, vocab = logits.shape

        # ------------------------------------------------------------------
        # Teacher-forcing objective via the unified module pipeline.
        #
        # Channel-A runs two contexts:
        #   - A1: registry_context=gt (token_ce anchor only; geo/coord_reg are disabled)
        #   - A2: registry_context=self_context (token_ce struct-only stabilizer + geo + coord_reg)
        #
        # Channel-B runs one context:
        #   - B: registry_context=rollout (rollout-context token_ce + geo + coord_reg; FP-neutral)
        # ------------------------------------------------------------------

        token_type_masks = build_token_type_masks(
            input_ids=input_ids,
            meta=meta,
            coord_id_set=coord_id_set,
            channel=channel,
        )
        rollout_subset_masks = build_rollout_subset_masks(
            input_ids=input_ids,
            meta=meta,
            coord_id_set=coord_id_set,
        )

        tf_context = TeacherForcingContext(
            channel=str(channel),
            registry_context=("self_context" if channel == "A" else "rollout"),
            input_ids=input_ids,
            logits=logits,
            logits_ce=logits,
            meta=meta,
            coord_token_ids=coord_token_ids,
            temperature=float(temperature),
            decode_mode=str(coord_decode_mode),
            token_type_masks=token_type_masks,
            rollout_subset_masks=rollout_subset_masks,
            extra={},
        )

        warn_once = getattr(self, "_tf_diag_warn_once", None)
        if not isinstance(warn_once, set):
            warn_once = set()
            setattr(self, "_tf_diag_warn_once", warn_once)

        def _spec_name(spec: Mapping[str, Any]) -> str:
            return str(spec.get("name", "") or "").strip()

        def _filter_channel_a_gt_specs(
            specs: Sequence[Mapping[str, Any]],
        ) -> list[Mapping[str, Any]]:
            if not run_a1_text:
                return []
            return [
                spec
                for spec in list(specs or [])
                if isinstance(spec, Mapping) and _spec_name(spec) == "token_ce"
            ]

        def _filter_channel_a_ctx_specs(
            specs: Sequence[Mapping[str, Any]],
        ) -> list[Mapping[str, Any]]:
            out: list[Mapping[str, Any]] = []
            for spec in list(specs or []):
                if not isinstance(spec, Mapping):
                    continue
                name = _spec_name(spec)
                if name == "token_ce":
                    if run_a2_text:
                        out.append(spec)
                    continue
                if name == "bbox_geo" and run_a2_bbox_geo:
                    out.append(spec)
                    continue
                if name == "bbox_size_aux" and run_a2_bbox_size_aux:
                    out.append(spec)
                    continue
                if name == "coord_reg" and run_a2_coord_reg:
                    out.append(spec)
                    continue
                if name == "loss_dead_anchor_suppression":
                    out.append(spec)
            return out

        def _filter_channel_a_a1_coord_specs(
            specs: Sequence[Mapping[str, Any]],
        ) -> list[Mapping[str, Any]]:
            out: list[Mapping[str, Any]] = []
            for spec in list(specs or []):
                if not isinstance(spec, Mapping):
                    continue
                name = _spec_name(spec)
                if name == "bbox_geo" and run_a1_bbox_geo:
                    out.append(spec)
                    continue
                if name == "bbox_size_aux" and run_a1_bbox_size_aux:
                    out.append(spec)
                    continue
                if name == "coord_reg" and run_a1_coord_reg:
                    out.append(spec)
            return out

        objective_specs_ctx = (
            _filter_channel_a_ctx_specs(objective_specs)
            if channel == "A"
            else list(objective_specs or [])
        )

        pipeline_ctx_result = run_teacher_forcing_pipeline(
            context=tf_context,
            objective_specs=objective_specs_ctx,
            diagnostics_specs=diagnostic_specs,
            initial_state=None,
            warn_once_cache=warn_once,
        )
        pipeline_metrics_ctx = dict(pipeline_ctx_result.metrics)
        pipeline_ctx_total_loss = pipeline_ctx_result.total_loss

        pipeline_metrics_gt: Dict[str, float] = {}
        a1_bbox_state: Optional[Mapping[str, Any]] = None
        a1_coord_state: Optional[Mapping[str, Any]] = None
        pipeline_metrics_a1: Dict[str, float] = {}
        pipeline_a1_total_loss = logits_ce.new_tensor(0.0)
        pipeline_gt_total_loss = logits_ce.new_tensor(0.0)
        if channel == "A":
            objective_specs_gt = _filter_channel_a_gt_specs(objective_specs)
            token_ctx_gt = TeacherForcingContext(
                channel="A",
                registry_context="gt",
                input_ids=input_ids,
                logits=logits_ce,
                logits_ce=logits_ce,
                meta=meta,
                coord_token_ids=coord_token_ids,
                temperature=float(temperature),
                decode_mode=str(coord_decode_mode),
                extra={},
            )
            if objective_specs_gt:
                pipeline_gt = run_teacher_forcing_pipeline(
                    context=token_ctx_gt,
                    objective_specs=objective_specs_gt,
                    diagnostics_specs=diagnostic_specs,
                    initial_state=None,
                    warn_once_cache=warn_once,
                )
                pipeline_metrics_gt = dict(pipeline_gt.metrics)
                pipeline_gt_total_loss = pipeline_gt.total_loss
                del pipeline_gt

            objective_specs_a1 = _filter_channel_a_a1_coord_specs(objective_specs)
            if objective_specs_a1:
                ctx_a1_obj = TeacherForcingContext(
                    channel="A",
                    registry_context="a1",
                    input_ids=input_ids,
                    logits=logits_ce,
                    logits_ce=logits_ce,
                    meta=meta,
                    coord_token_ids=coord_token_ids,
                    temperature=float(temperature),
                    decode_mode=str(coord_decode_mode),
                    token_type_masks=token_type_masks,
                    rollout_subset_masks=rollout_subset_masks,
                    extra={},
                )
                pipeline_a1 = run_teacher_forcing_pipeline(
                    context=ctx_a1_obj,
                    objective_specs=objective_specs_a1,
                    diagnostics_specs=[],
                    initial_state=None,
                    warn_once_cache=warn_once,
                )
                pipeline_metrics_a1 = dict(pipeline_a1.metrics)
                pipeline_a1_total_loss = pipeline_a1.total_loss
                a1_bbox_state = dict(pipeline_a1.state or {})
                a1_coord_state = dict(pipeline_a1.state or {})
                del pipeline_a1

            total = pipeline_gt_total_loss + pipeline_ctx_total_loss + pipeline_a1_total_loss
        else:
            total = pipeline_ctx_total_loss

        from src.metrics.reporter import best_effort_value

        monitor = get_loss_gradient_monitor(self)
        gradmon_metrics = {}
        if monitor is not None:
            gradmon_metrics = best_effort_value(
                self,
                name="loss_gradient_monitor",
                fn=lambda: monitor.measure(
                    model=model,
                    loss_terms=build_stage2_two_channel_coord_monitor_terms(
                        channel=channel,
                        pipeline_result=pipeline_ctx_result,
                        objective_specs=objective_specs_ctx,
                        bbox_module_weight=float(bbox_geo_module_w),
                        bbox_size_aux_module_weight=float(bbox_size_aux_module_w),
                        coord_module_weight=float(coord_reg_module_w),
                        a1_bbox_state=a1_bbox_state,
                        a1_coord_state=a1_coord_state,
                    ),
                ),
                default={},
            )

        # Buffer Stage-2 logs to merge into post-optimizer-step train log line.
        try:
            step = int(getattr(getattr(self, "state", None), "global_step", 0) or 0)
            target_step = step + 1

            pending2 = self._stage2_pending_train_logs.get(target_step)
            if pending2 is None:
                pending2 = _PendingStage2Log()
                self._stage2_pending_train_logs[target_step] = pending2
            # Counters should be summed across micro-batches; objective/monitor scalars averaged.
            stage2_logs: Dict[str, float] = {}

            if channel == "A":
                if float(token_ce_module_w) != 0.0 and run_a1_text:
                    token_struct = float(
                        pipeline_metrics_gt.get("loss/token_ce_struct", 0.0) or 0.0
                    )
                    token_desc = float(
                        pipeline_metrics_gt.get("loss/token_ce_desc", 0.0) or 0.0
                    )

                    stage2_logs["loss/A1_text/struct_ce"] = float(
                        float(token_ce_module_w) * float(token_struct)
                    )
                    if float(token_desc_ce_weight) != 0.0:
                        stage2_logs["loss/A1_text/desc_ce"] = float(
                            float(token_ce_module_w) * float(token_desc)
                        )

                    fmt_weight = (
                        float(self_context_struct_ce_weight)
                        if int(n_softctx_iter) > 1 and run_a2_text
                        else 0.0
                    )
                    if float(fmt_weight) != 0.0:
                        token_self_struct = float(
                            pipeline_metrics_ctx.get("loss/token_ce_struct", 0.0) or 0.0
                        )
                        stage2_logs["loss/A2_text/struct_ce"] = float(
                            float(token_ce_module_w)
                            * float(fmt_weight)
                            * float(token_self_struct)
                        )

                if pipeline_metrics_a1:
                    if float(bbox_geo_module_w) != 0.0 and run_a1_bbox_geo:
                        smoothl1_a1 = float(
                            pipeline_metrics_a1.get("loss/bbox_smoothl1", 0.0) or 0.0
                        )
                        ciou_a1 = float(
                            pipeline_metrics_a1.get("loss/bbox_ciou", 0.0) or 0.0
                        )
                        if float(bbox_smoothl1_w) != 0.0:
                            stage2_logs["loss/A1_coord/bbox_smoothl1"] = float(
                                float(bbox_geo_module_w)
                                * float(bbox_smoothl1_w)
                                * float(smoothl1_a1)
                            )
                        if float(bbox_ciou_w) != 0.0:
                            stage2_logs["loss/A1_coord/bbox_ciou"] = float(
                                float(bbox_geo_module_w)
                                * float(bbox_ciou_w)
                                * float(ciou_a1)
                            )

                    if float(bbox_size_aux_module_w) != 0.0 and run_a1_bbox_size_aux:
                        def _emit_a1_bbox_size(term: str, weight: float, raw_key: str) -> None:
                            if float(weight) == 0.0:
                                return
                            value = float(pipeline_metrics_a1.get(raw_key, 0.0) or 0.0)
                            stage2_logs[f"loss/A1_coord/{term}"] = float(
                                float(bbox_size_aux_module_w)
                                * float(weight)
                                * float(value)
                            )

                        _emit_a1_bbox_size("bbox_log_wh", bbox_log_wh_w, "loss/bbox_log_wh")
                        _emit_a1_bbox_size(
                            "bbox_oversize", bbox_oversize_w, "loss/bbox_oversize"
                        )

                    if float(coord_reg_module_w) != 0.0 and run_a1_coord_reg:
                        def _emit_a1(term: str, weight: float, raw_key: str) -> None:
                            if float(weight) == 0.0:
                                return
                            value = float(pipeline_metrics_a1.get(raw_key, 0.0) or 0.0)
                            stage2_logs[f"loss/A1_coord/{term}"] = float(
                                float(coord_reg_module_w) * float(weight) * float(value)
                            )

                        _emit_a1("coord_token_ce", coord_ce_w, "loss/coord_token_ce")
                        _emit_a1("coord_soft_ce", coord_soft_ce_w, "loss/coord_soft_ce")
                        _emit_a1("coord_w1", coord_w1_w, "loss/coord_w1")
                        _emit_a1("coord_el1", coord_el1_w, "loss/coord_el1")
                        _emit_a1("coord_ehuber", coord_ehuber_w, "loss/coord_ehuber")
                        _emit_a1("coord_entropy", coord_entropy_w, "loss/coord_entropy")
                        _emit_a1("coord_gate", coord_gate_w, "loss/coord_gate")
                        _emit_a1("text_gate", text_gate_w, "loss/text_gate")

                    stage2_logs["loss/A1_coord/total"] = float(
                        pipeline_a1_total_loss.detach().cpu().item()
                    )

                if float(bbox_geo_module_w) != 0.0 and run_a2_bbox_geo:
                    smoothl1 = float(
                        pipeline_metrics_ctx.get("loss/bbox_smoothl1", 0.0) or 0.0
                    )
                    ciou = float(
                        pipeline_metrics_ctx.get("loss/bbox_ciou", 0.0) or 0.0
                    )
                    if float(bbox_smoothl1_w) != 0.0:
                        stage2_logs["loss/A2_coord/bbox_smoothl1"] = float(
                            float(bbox_geo_module_w)
                            * float(bbox_smoothl1_w)
                            * float(smoothl1)
                        )
                    if float(bbox_ciou_w) != 0.0:
                        stage2_logs["loss/A2_coord/bbox_ciou"] = float(
                            float(bbox_geo_module_w)
                            * float(bbox_ciou_w)
                            * float(ciou)
                        )

                if float(bbox_size_aux_module_w) != 0.0 and run_a2_bbox_size_aux:
                    def _emit_a2_bbox_size(term: str, weight: float, raw_key: str) -> None:
                        if float(weight) == 0.0:
                            return
                        value = float(pipeline_metrics_ctx.get(raw_key, 0.0) or 0.0)
                        stage2_logs[f"loss/A2_coord/{term}"] = float(
                            float(bbox_size_aux_module_w) * float(weight) * float(value)
                        )

                    _emit_a2_bbox_size("bbox_log_wh", bbox_log_wh_w, "loss/bbox_log_wh")
                    _emit_a2_bbox_size("bbox_oversize", bbox_oversize_w, "loss/bbox_oversize")

                if float(coord_reg_module_w) != 0.0 and run_a2_coord_reg:
                    def _emit_a2(term: str, weight: float, raw_key: str) -> None:
                        if float(weight) == 0.0:
                            return
                        value = float(pipeline_metrics_ctx.get(raw_key, 0.0) or 0.0)
                        stage2_logs[f"loss/A2_coord/{term}"] = float(
                            float(coord_reg_module_w) * float(weight) * float(value)
                        )

                    _emit_a2("coord_token_ce", coord_ce_w, "loss/coord_token_ce")
                    _emit_a2("coord_soft_ce", coord_soft_ce_w, "loss/coord_soft_ce")
                    _emit_a2("coord_w1", coord_w1_w, "loss/coord_w1")
                    _emit_a2("coord_el1", coord_el1_w, "loss/coord_el1")
                    _emit_a2("coord_ehuber", coord_ehuber_w, "loss/coord_ehuber")
                    _emit_a2("coord_entropy", coord_entropy_w, "loss/coord_entropy")
                    _emit_a2("coord_gate", coord_gate_w, "loss/coord_gate")
                    _emit_a2("text_gate", text_gate_w, "loss/text_gate")
            else:
                if float(token_ce_module_w) != 0.0:
                    token_struct = float(
                        pipeline_metrics_ctx.get("loss/token_ce_struct", 0.0) or 0.0
                    )
                    token_desc = float(
                        pipeline_metrics_ctx.get("loss/token_ce_desc", 0.0) or 0.0
                    )

                    stage2_logs["loss/B_rollout_text/struct_ce"] = float(
                        float(token_ce_module_w) * float(token_struct)
                    )
                    if float(fn_desc_ce_weight) != 0.0:
                        stage2_logs["loss/B_rollout_text/desc_ce"] = float(
                            float(token_ce_module_w) * float(token_desc)
                        )

                if float(dead_anchor_suppression_module_w) != 0.0:
                    dead_anchor_loss = float(
                        pipeline_metrics_ctx.get(
                            "train/optimization/loss_dead_anchor_suppression", 0.0
                        )
                        or 0.0
                    )
                    dead_anchor_num_terms = float(
                        pipeline_metrics_ctx.get(
                            "train/triage/dead_anchor_suppression_target_count", 0.0
                        )
                        or 0.0
                    )
                    dead_anchor_num_boundaries = float(
                        pipeline_metrics_ctx.get(
                            "train/triage/dead_anchor_suppression_boundary_count", 0.0
                        )
                        or 0.0
                    )
                    stage2_logs["train/optimization/loss_dead_anchor_suppression"] = float(
                        dead_anchor_loss
                    )
                    # Compatibility alias so dead-anchor UL is visible beside the
                    # other Channel-B text losses under loss/B_rollout_text/*.
                    stage2_logs["loss/B_rollout_text/dead_anchor_suppression"] = float(
                        dead_anchor_loss
                    )
                    # Counts explain whether low UL is "healthy" or simply "few terms."
                    stage2_logs["diag/dead_anchor/num_terms"] = float(
                        dead_anchor_num_terms
                    )
                    stage2_logs["diag/dead_anchor/num_ul_boundaries"] = float(
                        dead_anchor_num_boundaries
                    )
                    stage2_logs["diag/dead_anchor/loss_per_term"] = float(
                        dead_anchor_loss if dead_anchor_num_terms > 0.0 else 0.0
                    )

                if float(bbox_geo_module_w) != 0.0:
                    smoothl1 = float(
                        pipeline_metrics_ctx.get("loss/bbox_smoothl1", 0.0) or 0.0
                    )
                    ciou = float(
                        pipeline_metrics_ctx.get("loss/bbox_ciou", 0.0) or 0.0
                    )
                    if float(bbox_smoothl1_w) != 0.0:
                        stage2_logs["loss/B_coord/bbox_smoothl1"] = float(
                            float(bbox_geo_module_w)
                            * float(bbox_smoothl1_w)
                            * float(smoothl1)
                        )
                    if float(bbox_ciou_w) != 0.0:
                        stage2_logs["loss/B_coord/bbox_ciou"] = float(
                            float(bbox_geo_module_w) * float(bbox_ciou_w) * float(ciou)
                        )

                if float(bbox_size_aux_module_w) != 0.0:
                    def _emit_b_bbox_size(term: str, weight: float, raw_key: str) -> None:
                        if float(weight) == 0.0:
                            return
                        value = float(pipeline_metrics_ctx.get(raw_key, 0.0) or 0.0)
                        stage2_logs[f"loss/B_coord/{term}"] = float(
                            float(bbox_size_aux_module_w) * float(weight) * float(value)
                        )

                    _emit_b_bbox_size("bbox_log_wh", bbox_log_wh_w, "loss/bbox_log_wh")
                    _emit_b_bbox_size("bbox_oversize", bbox_oversize_w, "loss/bbox_oversize")

                if float(coord_reg_module_w) != 0.0:
                    def _emit_b(term: str, weight: float, raw_key: str) -> None:
                        if float(weight) == 0.0:
                            return
                        value = float(pipeline_metrics_ctx.get(raw_key, 0.0) or 0.0)
                        stage2_logs[f"loss/B_coord/{term}"] = float(
                            float(coord_reg_module_w) * float(weight) * float(value)
                        )

                    _emit_b("coord_token_ce", coord_ce_w, "loss/coord_token_ce")
                    _emit_b("coord_soft_ce", coord_soft_ce_w, "loss/coord_soft_ce")
                    _emit_b("coord_w1", coord_w1_w, "loss/coord_w1")
                    _emit_b("coord_el1", coord_el1_w, "loss/coord_el1")
                    _emit_b("coord_ehuber", coord_ehuber_w, "loss/coord_ehuber")
                    _emit_b("coord_entropy", coord_entropy_w, "loss/coord_entropy")
                    _emit_b("coord_gate", coord_gate_w, "loss/coord_gate")
                    _emit_b("text_gate", text_gate_w, "loss/text_gate")

            if isinstance(gradmon_metrics, Mapping):
                for k, v in gradmon_metrics.items():
                    try:
                        stage2_logs[str(k)] = float(v)
                    except (TypeError, ValueError):
                        continue

            pack_segments = int(len(meta))
            if pack_segments <= 0:
                raise ValueError(
                    "stage2-ab compute_loss requires non-empty _rollout_matching_meta per packed forward"
                )
            stage2_logs["stage2/_log_weight"] = float(pack_segments)

            if coord_diag_enabled:
                def _emit_coord_diag(prefix: str, metrics: Mapping[str, float]) -> None:
                    for k, v in metrics.items():
                        ks = str(k)
                        if not ks.startswith("coord_diag/"):
                            continue
                        suffix = ks[len("coord_diag/") :]
                        stage2_logs[f"coord_diag/{prefix}/{suffix}"] = float(v)

                if channel == "A":
                    # A1: diagnostics from the GT-anchor logits (it==0).
                    try:
                        with torch.no_grad():
                            from .teacher_forcing.modules import (
                                run_bbox_geo_module,
                                run_coord_diag_module,
                            )

                            ctx_a1 = TeacherForcingContext(
                                channel="A",
                                registry_context="self_context",
                                input_ids=input_ids,
                                logits=logits_ce,
                                logits_ce=logits_ce,
                                meta=meta,
                                coord_token_ids=coord_token_ids,
                                temperature=float(temperature),
                                decode_mode=str(coord_decode_mode),
                                token_type_masks=token_type_masks,
                                rollout_subset_masks=rollout_subset_masks,
                                extra={},
                            )
                            bbox_spec = PipelineModuleSpec(
                                name="bbox_geo",
                                enabled=True,
                                weight=0.0,
                                channels=("A",),
                                config=dict(bbox_cfg),
                            )
                            diag_spec = PipelineModuleSpec(
                                name="coord_diag",
                                enabled=True,
                                weight=0.0,
                                channels=("A",),
                                config={},
                            )

                            bbox_out = run_bbox_geo_module(
                                context=ctx_a1,
                                spec=bbox_spec,
                            )
                            diag_out = run_coord_diag_module(
                                context=ctx_a1,
                                spec=diag_spec,
                                state=bbox_out.state,
                            )
                            _emit_coord_diag("A1", diag_out.metrics)
                    except Exception:
                        key = "coord_diag/A1_failed"
                        if isinstance(warn_once, set) and key not in warn_once:
                            logger.warning(
                                "Skipping Channel-A A1 coord diagnostics after helper failure.",
                                exc_info=True,
                            )
                            warn_once.add(key)

                    # A2: diagnostics from final self-context logits (only meaningful if we ran >1 iter).
                    if int(n_softctx_iter) > 1:
                        _emit_coord_diag("A2", pipeline_metrics_ctx)
                else:
                    _emit_coord_diag("B", pipeline_metrics_ctx)

            b_ratio_cfg = float(self._ab_schedule_b_ratio())
            if 0.0 < b_ratio_cfg < 1.0:
                stage2_logs["stage2/channel_a"] = float(1.0 if channel == "A" else 0.0)
                stage2_logs["stage2/channel_b"] = float(1.0 if channel == "B" else 0.0)
                stage2_logs["stage2_ab/b_ratio_realized"] = float(
                    self._stage2_b_ratio_realized()
                )

            if channel == "B" and isinstance(batch_metrics, Mapping):
                raw_rollouts = 0.0
                try:
                    raw_rollouts = float(
                        batch_metrics.get("stage2/raw_rollouts", 0.0) or 0.0
                    )
                except (TypeError, ValueError):
                    raw_rollouts = 0.0
                ran_rollout = bool(raw_rollouts > 0.0)

                for k in (
                    "stage2/raw_rollouts",
                    "stage2/invalid_rollout",
                    "stage2_ab/channel_b/invalid_rollout",
                    "stage2/drop_poly",
                    "stage2/drop_unknown",
                    "stage2/drop_bbox_invalid",
                ):
                    if k in batch_metrics:
                        stage2_logs[k] = float(batch_metrics.get(k) or 0.0)

                if ran_rollout:
                    for k in (
                        "rollout/seed_base",
                        "rollout/parse_truncated",
                    ):
                        if k in batch_metrics:
                            stage2_logs[k] = float(batch_metrics.get(k) or 0.0)

                for k, v in batch_metrics.items():
                    key = str(k)
                    if (
                        key.startswith("stage2_ab/")
                        or key.startswith("dup/")
                        or key.startswith("train/triage/")
                        or key.startswith("diag/dead_anchor/")
                        or key.startswith("rollout/anchor/")
                        or key.startswith("rollout/explorer/")
                        or key
                        in {
                            "rollout/matched_for_supervision_count",
                            "rollout/matched_for_supervision_over_valid_pred",
                            "rollout/valid_pred_objects_total",
                        }
                    ):
                        stage2_logs[key] = float(v or 0.0)

            if isinstance(batch_metrics, Mapping):
                for k, v in batch_metrics.items():
                    if not str(k).startswith("time/"):
                        continue
                    fv = float(v or 0.0)
                    if str(k) == "time/mask_build_s" and fv == 0.0:
                        continue
                    stage2_logs[str(k)] = fv

            pending2.add(stage2_logs)
        except (AttributeError, KeyError, TypeError, ValueError):
            raise

        # Also feed the base rollout-matching pending log so its timing/packing/buffer plots stay intact.
        try:
            pending = self._rm_pending_train_logs.get(
                int(getattr(getattr(self, "state", None), "global_step", 0) or 0) + 1
            )
            if pending is None:
                pending = PendingTrainRolloutLog()
                self._rm_pending_train_logs[
                    int(getattr(getattr(self, "state", None), "global_step", 0) or 0)
                    + 1
                ] = pending
            pending.add_micro(
                meta=meta,
                objective_atoms=None,
                gradmon_metrics=None,
                time_forward_s=float(t_fwd_s),
                time_mask_build_s=float(0.0),
                batch_metrics=batch_metrics
                if isinstance(batch_metrics, Mapping)
                else None,
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            raise

        return (total, outputs) if return_outputs else total

# Canonical alias for forward-looking callsites.
Stage2TwoChannelTrainer = Stage2ABTrainingTrainer

__all__ = ["Stage2ABTrainingTrainer", "Stage2TwoChannelTrainer"]
