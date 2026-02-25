import contextlib
import json
import math
import os
import time
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, ClassVar, Deque, Dict, List, Literal, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from swift.llm import MaxLengthError
from swift.trainers.rlhf_trainer.utils import replace_assistant_response_with_ids

from src.common.object_field_order import build_object_payload
from src.utils.assistant_json import dumps_coordjson

from .stage2_rollout_aligned import RolloutMatchingSFTTrainer
from .rollout_matching.contracts import GTObject
from .rollout_matching.matching import hungarian_match_maskiou
from .rollout_matching.parsing import (
    decode_pieces,
    find_desc_value_char_spans,
    find_desc_value_token_positions,
    parse_rollout_for_matching,
    points_from_coord_tokens,
    serialize_append_fragment,
)
from .rollout_matching.telemetry import PendingTrainRolloutLog
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
                # Means: segment-weighted when stage2/_log_weight is provided.
                self.sums[ks] = float(self.sums.get(ks, 0.0)) + float(fv) * float(weight)
            else:
                # Unknown/non-contract keys default to sum semantics.
                self.sums[ks] = float(self.sums.get(ks, 0.0)) + float(fv)

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
                out[k] = float(v) / float(denom)
            else:
                out[k] = float(v)

        # Internal: used for DDP-weighted mean reduction.
        out["stage2/_log_weight_total"] = float(denom)

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

        return out




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
        self._stage2_pending_train_logs: Dict[int, _PendingStage2Log] = {}
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

    def train(self, *args, **kwargs):
        """Run training and ensure rollout server clients shut down cleanly."""
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
            try:
                world_size = int(dist.get_world_size())
            except RuntimeError:
                world_size = 1
            try:
                rank = int(dist.get_rank())
            except RuntimeError:
                rank = 0

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
            try:
                gathered_keys: List[Any] = [None] * int(world_size)
                dist.all_gather_object(gathered_keys, metric_keys)

                merged_keys: Dict[str, None] = {}
                for item in gathered_keys:
                    if not isinstance(item, (list, tuple)):
                        continue
                    for key_raw in item:
                        key = str(key_raw)
                        merged_keys[key] = None
                        reduced.setdefault(key, 0.0)

                metric_keys = sorted(merged_keys.keys())
            except (AttributeError, RuntimeError) as exc:
                if int(rank) == 0:
                    logger.warning(
                        "stage2-ab metric key sync failed; proceeding without key union: %r",
                        exc,
                    )

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

                if weight_key in reduced and mean_keys:
                    # Convert local means into local numerators before reduction.
                    for key in mean_keys:
                        reduced[key] = float(reduced.get(key, 0.0)) * float(local_weight_total)

                _all_reduce(sum_keys + mean_keys, dist.ReduceOp.SUM)
                _all_reduce(max_keys, dist.ReduceOp.MAX)

                if weight_key in reduced and mean_keys:
                    global_weight_total = float(reduced.get(weight_key, 0.0))
                    if global_weight_total > 0.0:
                        for key in mean_keys:
                            reduced[key] = float(reduced.get(key, 0.0) / global_weight_total)
                    else:
                        for key in mean_keys:
                            reduced[key] = 0.0
                else:
                    scale = float(world_size)
                    if scale > 0.0:
                        for key in mean_keys:
                            reduced[key] = float(reduced[key] / scale)
            except (AttributeError, RuntimeError) as exc:
                if int(rank) == 0:
                    logger.warning(
                        "stage2-ab metric all-reduce failed; falling back to local metrics: %r",
                        exc,
                    )

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
            if pending is not None:
                reduced = self._reduce_stage2_pending_metrics_global(
                    pending.finalize(drop_internal=False)
                )
                reduced.pop("rollout/_parse_truncated_num", None)
                reduced.pop("rollout/_parse_truncated_den", None)
                logs.update(reduced)
        return super().log(logs)

    def training_step(self, model, inputs, *args, **kwargs):
        # When using identity collator, `inputs` is a list of raw samples.
        if not isinstance(inputs, list):
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
            prepared = self._prepare_batch_inputs(inputs)
        finally:
            self._stage2_channel_override = prev

        return super(RolloutMatchingSFTTrainer, self).training_step(
            model, prepared, *args, **kwargs
        )



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

        # Optional weak correction when strict-drop removes invalid predicted instances.
        drop_invalid_struct_ce_multiplier = float(
            self._ab_channel_b_get("drop_invalid_struct_ce_multiplier", 1.0) or 1.0
        )
        drop_invalid_struct_ce_multiplier = float(
            max(1.0, min(4.0, drop_invalid_struct_ce_multiplier))
        )


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
        seed_base = int(self._derive_rollout_seed_base(global_step=gs))

        backend = self._rollout_backend()
        decode_mode = str(self._cfg("decode_mode", "greedy")).lower()
        max_new_tokens = int(self._cfg("max_new_tokens", 512))
        num_beams = int(self._cfg("num_beams", 1))
        temperature, top_p, decode_top_k = self._decoding_params()
        repetition_penalty = float(self._cfg("repetition_penalty", 1.0) or 1.0)
        do_sample = bool(float(temperature) > 0.0)

        with self._hf_sampling_seed_context(
            seed_base=seed_base, backend=backend, do_sample=do_sample
        ) as seeded:
            hf_seeded_global = float(1.0 if seeded else 0.0)

            t_gen0 = time.perf_counter()

            # Derived rollout request chunk size per learner rank.
            #
            # `decode_batch_size` is defined as a per-rollout-GPU cap per generation call.
            # For vLLM server mode (data-parallel replicas), we derive the per-rank chunk
            # size so the per-GPU cap holds when all learner ranks run concurrently.
            rollout_infer_bs = int(self._rollout_decode_batch_size_per_rank())
            rollout_infer_bs = max(1, int(rollout_infer_bs))
            if int(len(inputs)) > 0:
                rollout_infer_bs = min(int(rollout_infer_bs), int(len(inputs)))

            rollout_results = []
            for off in range(0, int(len(inputs)), int(rollout_infer_bs)):
                chunk = inputs[int(off) : int(off + rollout_infer_bs)]
                if not chunk:
                    continue

                chunk_results = self._rollout_many(chunk)
                rollout_results.extend(chunk_results)

            if len(rollout_results) != len(inputs):
                raise RuntimeError(
                    "rollout backend returned unexpected number of results"
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

        strict_valid_pred_total = 0
        strict_drop_invalid_total = 0
        strict_drop_by_reason_total: Dict[str, int] = {}

        for sample, (resp_ids, _resp_text, decode_mode, prompt_ids) in zip(
            inputs,
            rollout_results,
        ):
            if "messages" not in sample:
                raise ValueError("stage2-ab requires 'messages' in dataset samples")

            # If generation stopped due to max_new_tokens, it may lack EOS.
            # For Channel-B rollout this is acceptable; append EOS to make downstream parsing deterministic.
            resp_ids_local = [int(t) for t in resp_ids]
            eos_id = getattr(tok, "eos_token_id", None)
            if int(max_new_tokens) > 0 and int(len(resp_ids_local)) >= int(max_new_tokens):
                try:
                    eos = int(eos_id) if eos_id is not None else -1
                except (TypeError, ValueError):
                    eos = -1
                if eos >= 0 and (not resp_ids_local or int(resp_ids_local[-1]) != eos):
                    resp_ids_local.append(int(eos))

            t_pm0 = time.perf_counter()
            parse = parse_rollout_for_matching(
                tokenizer=tok,
                response_token_ids=resp_ids_local,
                object_field_order=self._object_field_order(),
            )
            invalid_rollout = int(
                1 if bool(getattr(parse, "invalid_rollout", False)) else 0
            )

            parse_truncated_total += int(1 if bool(parse.truncated) else 0)

            gts = _extract_gt_bboxonly(sample)

            # Filter preds to bbox-only and accumulate strict-drop diagnostics.
            drop_reasons: Dict[str, int] = {}
            try:
                raw = getattr(parse, "dropped_invalid_by_reason", None)
                if isinstance(raw, Mapping):
                    for k, v in raw.items():
                        try:
                            drop_reasons[str(k)] = int(v)
                        except (TypeError, ValueError):
                            continue
            except Exception:
                raise

            drop_poly = 0
            drop_unknown = 0
            drop_bbox_invalid = 0

            pred_meta = []
            preds: List[GTObject] = []
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

                pred_meta.append(pobj)
                preds.append(
                    GTObject(
                        index=int(pobj.index),
                        geom_type="bbox_2d",
                        points_norm1000=[x1, y1, x2, y2],
                        desc="",
                    )
                )

            drop_poly_total += int(drop_poly)
            drop_unknown_total += int(drop_unknown)
            drop_bbox_invalid_total += int(drop_bbox_invalid)

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

            n_valid_pred = int(len(preds))
            n_drop_invalid = (
                int(getattr(parse, "dropped_invalid", 0) or 0)
                + int(getattr(parse, "dropped_ambiguous", 0) or 0)
                + int(drop_poly)
                + int(drop_unknown)
                + int(drop_bbox_invalid)
            )

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

            match = hungarian_match_maskiou(
                preds=preds,
                gts=gts,
                top_k=match_top_k,
                gate_threshold=gate_thr,
                mask_resolution=mask_res,
                fp_cost=fp_cost,
                fn_cost=fn_cost,
            )

            prefix_bbox_groups: List[Dict[str, Any]] = []
            fn_bbox_groups: List[Dict[str, Any]] = []
            prefix_pos: List[int] = []
            prefix_bins: List[int] = []
            matched_gt_for_supervision: set[int] = set()

            prefix_len_raw_local = int(len(parse.prefix_token_ids))

            prefix_struct_pos: List[int] = []

            fn_count_for_meta = 0

            matched_pred_indices: set[int] = set()
            for pred_i, gt_i in match.matched_pairs:
                if pred_i < 0 or pred_i >= len(pred_meta):
                    continue
                if gt_i < 0 or gt_i >= len(gts):
                    continue
                pobj = pred_meta[pred_i]
                if len(pobj.coord_token_indices) != 4:
                    continue
                matched_gt_for_supervision.add(int(gt_i))
                matched_pred_indices.add(int(pred_i))
                gt_bins = list(gts[gt_i].points_norm1000)
                pos_seg = [
                    int(len(prompt_ids) + int(p)) for p in pobj.coord_token_indices
                ]
                prefix_bbox_groups.append({"pos": pos_seg, "gt_bins": gt_bins})
                for local_idx, tbin in zip(pobj.coord_token_indices, gt_bins):
                    prefix_pos.append(int(local_idx))
                    prefix_bins.append(int(tbin))

            matched_pred_objects = [
                pred_meta[int(i)]
                for i in sorted(matched_pred_indices)
                if 0 <= int(i) < len(pred_meta)
            ]
            prefix_struct_pos = _matched_prefix_structure_positions(
                tokenizer=tok,
                prefix_token_ids=parse.prefix_token_ids,
                prefix_text=parse.prefix_text,
                matched_pred_objects=matched_pred_objects,
            )

            fn_gt_indices_final = [
                i for i in range(len(gts)) if i not in matched_gt_for_supervision
            ]
            fn_objs = [gts[i] for i in fn_gt_indices_final]
            fn_count_for_meta = int(len(fn_objs))

            append_text = serialize_append_fragment(
                fn_objects=fn_objs,
                prefix_text=parse.prefix_text,
                object_field_order=self._object_field_order(),
            )
            append_ids = tok.encode(append_text, add_special_tokens=False)

            tail_desc_pos = find_desc_value_token_positions(
                tokenizer=tok, token_ids=append_ids
            )

            y_train_ids = list(parse.prefix_token_ids) + [
                int(t) for t in append_ids
            ]

            # FN bbox groups in the appended tail.
            rel_groups = _bbox_groups_from_token_ids(
                token_ids=append_ids, coord_id_set=coord_id_set, gt_objs=fn_objs
            )
            for obj, rel_pos in zip(fn_objs, rel_groups):
                fn_bbox_groups.append(
                    {
                        "pos": [
                            int(
                                len(prompt_ids)
                                + int(len(parse.prefix_token_ids))
                                + int(p)
                            )
                            for p in rel_pos
                        ],
                        "gt_bins": list(obj.points_norm1000),
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
            if isinstance(prompt_ids, list) and prompt_ids:
                prompt_ids_int = [int(t) for t in prompt_ids]
                teacher_prefix = enc_ids_list[: len(prompt_ids_int)]
                if teacher_prefix != prompt_ids_int:
                    raise ValueError(
                        "prompt tokenization mismatch between generation and teacher-forced encoding"
                    )
                if int(prompt_len) != int(len(prompt_ids_int)):
                    raise ValueError(
                        "stage2-ab prompt_len mismatch vs server prompt_token_ids length; "
                        f"prompt_len={int(prompt_len)} server_prompt_len={int(len(prompt_ids_int))}"
                    )

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
                    if not isinstance(pos, Sequence) or not isinstance(gb, Sequence):
                        continue
                    if len(pos) != 4 or len(gb) != 4:
                        continue
                    try:
                        pos_i = [int(p) for p in pos]
                        gb_i = [int(x) for x in gb]
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
                    out.append({"pos": pos_i, "gt_bins": gb_i})
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
            tail_cap = max(0, int(train_len_eff) - int(prefix_len_eff))

            # Stop/closure supervision is active (no stop-neutral masking), but we still
            # require deterministic closure-marker resolution for contract validation.
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
                invalid_rollout_total += 1
                continue

            for rel in tail_desc_pos:
                try:
                    rel_i = int(rel)
                except (TypeError, ValueError):
                    continue
                if 0 <= rel_i < tail_cap:
                    tail_desc_pos_eff.append(rel_i)

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
                "drop_invalid_struct_ce_multiplier": float(
                    drop_invalid_struct_ce_multiplier
                ),
                "valid_pred_objects": int(len(preds)),
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
                "bbox_groups_prefix": bbox_groups_prefix,
                "bbox_groups_fn": bbox_groups_fn,
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
            "stage2/drop_poly": float(drop_poly_total),
            "stage2/drop_unknown": float(drop_unknown_total),
            "stage2/drop_bbox_invalid": float(drop_bbox_invalid_total),
            "rollout/seed_base": float(seed_base),
            "rollout/backend_hf": float(1.0 if backend == "hf" else 0.0),
            "rollout/backend_vllm": float(1.0 if backend == "vllm" else 0.0),
            "rollout/decode_mode_greedy": float(
                1.0 if decode_mode == "greedy" else 0.0
            ),
            "rollout/decode_mode_beam": float(1.0 if decode_mode == "beam" else 0.0),
            "rollout/hf_seeded_global": float(hf_seeded_global),
            "rollout/temperature": float(temperature),
            "rollout/top_p": float(top_p),
            "rollout/top_k": float(decode_top_k),
            "rollout/do_sample": float(1.0 if do_sample else 0.0),
            "rollout/max_new_tokens": float(max_new_tokens),
            "rollout/num_beams": float(num_beams),
            "rollout/repetition_penalty": float(repetition_penalty),
            "rollout/parse_truncated": float(parse_truncated_total),
            "rollout/parse_truncated_rate": float(
                (float(parse_truncated_total) / float(len(rollout_results)))
                if len(rollout_results) > 0
                else 0.0
            ),
            "rollout/_parse_truncated_num": float(parse_truncated_total),
            "rollout/_parse_truncated_den": float(len(rollout_results)),
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
                "this can happen when closure-marker resolution fails due to truncation/misalignment."
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
        coord_module_cfg = _module_config(objective_specs, name="coord_reg")

        token_cfg = token_module_cfg if isinstance(token_module_cfg, Mapping) else {}
        bbox_cfg = bbox_module_cfg if isinstance(bbox_module_cfg, Mapping) else {}
        coord_cfg = coord_module_cfg if isinstance(coord_module_cfg, Mapping) else {}

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
            keys=("self_context_struct_ce_weight",),
            default=0.1,
            min_value=0.0,
        )
        matched_prefix_struct_ce_weight = _cfg_float(
            token_cfg,
            keys=("rollout_matched_prefix_struct_weight",),
            default=1.0,
            min_value=0.0,
        )
        fn_desc_ce_weight = _cfg_float(
            token_cfg,
            keys=("rollout_fn_desc_weight",),
            default=token_desc_ce_weight,
            min_value=0.0,
        )
        rollout_drop_invalid_struct_ce_multiplier_cfg = _cfg_float(
            token_cfg,
            keys=("rollout_drop_invalid_struct_ce_multiplier",),
            default=1.0,
            min_value=1.0,
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
        coord_huber_delta = _cfg_float(
            coord_cfg,
            keys=("coord_huber_delta",),
            default=0.001,
            min_value=1e-6,
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

        if channel == "A":
            coord_soft_ce_w = _cfg_float(
                coord_cfg,
                keys=("self_context_soft_ce_weight", "soft_ce_weight"),
                default=0.0,
                min_value=0.0,
            )
        else:
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

        objective_specs_ctx = objective_specs
        if channel == "A" and int(n_softctx_iter) <= 1:
            # No iterative self-context updates => no self-context CE stabilizer.
            #
            # Keep Channel-A token CE anchored to the gt forward only.
            objective_specs_ctx = []
            for spec in list(objective_specs or []):
                if not isinstance(spec, Mapping):
                    continue
                name = str(spec.get("name", "") or "").strip()
                if name != "token_ce":
                    objective_specs_ctx.append(spec)
                    continue
                spec2 = dict(spec)
                cfg_raw = spec2.get("config", {})
                cfg2 = dict(cfg_raw) if isinstance(cfg_raw, Mapping) else {}
                cfg2["self_context_struct_ce_weight"] = 0.0
                spec2["config"] = cfg2
                objective_specs_ctx.append(spec2)

        pipeline_ctx = run_teacher_forcing_pipeline(
            context=tf_context,
            objective_specs=objective_specs_ctx,
            diagnostics_specs=diagnostic_specs,
            initial_state=None,
            warn_once_cache=warn_once,
        )
        pipeline_metrics_ctx = dict(pipeline_ctx.metrics)

        pipeline_metrics_gt: Dict[str, float] = {}
        if channel == "A":
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
            pipeline_gt = run_teacher_forcing_pipeline(
                context=token_ctx_gt,
                objective_specs=objective_specs,
                diagnostics_specs=diagnostic_specs,
                initial_state=None,
                warn_once_cache=warn_once,
            )
            pipeline_metrics_gt = dict(pipeline_gt.metrics)

            # Optional A1 coord/geometric anchors (small weights).
            #
            # Motivation: A1 (GT-prefix) logits are used to bootstrap A2 self-context.
            # Adding a weak A1 anchor can reduce self-context noise and stabilize bbox loss.
            a1_bbox_obj = logits_ce.new_tensor(0.0)
            a1_coord_obj = logits_ce.new_tensor(0.0)
            a1_bbox_metrics: Dict[str, float] = {}
            a1_coord_metrics: Dict[str, float] = {}

            try:
                a1_smoothl1_w = float(bbox_cfg.get("a1_smoothl1_weight", 0.0) or 0.0)
                a1_ciou_w = float(bbox_cfg.get("a1_ciou_weight", 0.0) or 0.0)
            except (TypeError, ValueError):
                a1_smoothl1_w = 0.0
                a1_ciou_w = 0.0

            try:
                a1_soft_ce_w = float(coord_cfg.get("a1_soft_ce_weight", 0.0) or 0.0)
                a1_w1_w = float(coord_cfg.get("a1_w1_weight", 0.0) or 0.0)
            except (TypeError, ValueError):
                a1_soft_ce_w = 0.0
                a1_w1_w = 0.0

            if (
                (float(bbox_geo_module_w) != 0.0 or float(coord_reg_module_w) != 0.0)
                and (
                    float(a1_smoothl1_w) != 0.0
                    or float(a1_ciou_w) != 0.0
                    or float(a1_soft_ce_w) != 0.0
                    or float(a1_w1_w) != 0.0
                )
            ):
                from .teacher_forcing.modules import run_bbox_geo_module, run_coord_reg_module

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

                bbox_spec_a1 = PipelineModuleSpec(
                    name="bbox_geo",
                    enabled=True,
                    weight=1.0,
                    channels=("A",),
                    config={
                        "smoothl1_weight": float(max(float(a1_smoothl1_w), 0.0)),
                        "ciou_weight": float(max(float(a1_ciou_w), 0.0)),
                    },
                )
                bbox_out_a1 = run_bbox_geo_module(context=ctx_a1_obj, spec=bbox_spec_a1)
                a1_bbox_obj = bbox_out_a1.loss * float(bbox_geo_module_w)
                a1_bbox_metrics = {
                    str(k): float(v) for k, v in dict(bbox_out_a1.metrics or {}).items()
                }

                if float(a1_soft_ce_w) != 0.0 or float(a1_w1_w) != 0.0:
                    coord_spec_a1 = PipelineModuleSpec(
                        name="coord_reg",
                        enabled=True,
                        weight=1.0,
                        channels=("A",),
                        config={
                            # A1 uses SoftCE/W1 only by default (no hard CE, no gates).
                            "coord_ce_weight": 0.0,
                            "coord_gate_weight": 0.0,
                            "text_gate_weight": 0.0,
                            "soft_ce_weight": float(max(float(a1_soft_ce_w), 0.0)),
                            "w1_weight": float(max(float(a1_w1_w), 0.0)),
                            # Keep distribution target settings aligned with A2 unless overridden.
                            "temperature": coord_cfg.get("temperature", float(temperature)),
                            "target_sigma": coord_cfg.get("target_sigma", 2.0),
                            "target_truncate": coord_cfg.get("target_truncate", None),
                        },
                    )
                    coord_out_a1 = run_coord_reg_module(
                        context=ctx_a1_obj,
                        spec=coord_spec_a1,
                        state=bbox_out_a1.state,
                    )
                    a1_coord_obj = coord_out_a1.loss * float(coord_reg_module_w)
                    a1_coord_metrics = {
                        str(k): float(v)
                        for k, v in dict(coord_out_a1.metrics or {}).items()
                    }

            total = pipeline_gt.total_loss + pipeline_ctx.total_loss + a1_bbox_obj + a1_coord_obj
        else:
            total = pipeline_ctx.total_loss

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
                if float(token_ce_module_w) != 0.0:
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
                        if int(n_softctx_iter) > 1
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

                # Optional A1 coord/geo anchors.
                # NOTE: these are computed from A1 logits but under registry_context='a1'
                # (not 'gt'), so bbox_geo/coord_reg modules can run.
                if (
                    float(a1_smoothl1_w) != 0.0
                    or float(a1_ciou_w) != 0.0
                    or float(a1_soft_ce_w) != 0.0
                    or float(a1_w1_w) != 0.0
                ):
                    if float(bbox_geo_module_w) != 0.0 and (
                        float(a1_smoothl1_w) != 0.0 or float(a1_ciou_w) != 0.0
                    ):
                        smoothl1_a1 = float(
                            a1_bbox_metrics.get("loss/bbox_smoothl1", 0.0) or 0.0
                        )
                        ciou_a1 = float(a1_bbox_metrics.get("loss/bbox_ciou", 0.0) or 0.0)
                        if float(a1_smoothl1_w) != 0.0:
                            stage2_logs["loss/A1_coord/bbox_smoothl1"] = float(
                                float(bbox_geo_module_w)
                                * float(a1_smoothl1_w)
                                * float(smoothl1_a1)
                            )
                        if float(a1_ciou_w) != 0.0:
                            stage2_logs["loss/A1_coord/bbox_ciou"] = float(
                                float(bbox_geo_module_w)
                                * float(a1_ciou_w)
                                * float(ciou_a1)
                            )

                    if float(coord_reg_module_w) != 0.0 and (
                        float(a1_soft_ce_w) != 0.0 or float(a1_w1_w) != 0.0
                    ):
                        soft_ce_a1 = float(
                            a1_coord_metrics.get("loss/coord_soft_ce", 0.0) or 0.0
                        )
                        w1_a1 = float(a1_coord_metrics.get("loss/coord_w1", 0.0) or 0.0)
                        if float(a1_soft_ce_w) != 0.0:
                            stage2_logs["loss/A1_coord/coord_soft_ce"] = float(
                                float(coord_reg_module_w)
                                * float(a1_soft_ce_w)
                                * float(soft_ce_a1)
                            )
                        if float(a1_w1_w) != 0.0:
                            stage2_logs["loss/A1_coord/coord_w1"] = float(
                                float(coord_reg_module_w)
                                * float(a1_w1_w)
                                * float(w1_a1)
                            )

                    stage2_logs["loss/A1_coord/total"] = float(
                        float(a1_bbox_obj + a1_coord_obj).detach().cpu().item()
                    )

                if float(bbox_geo_module_w) != 0.0:
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

                if float(coord_reg_module_w) != 0.0:
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
                    if str(k).startswith("stage2_ab/"):
                        stage2_logs[str(k)] = float(v or 0.0)

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
