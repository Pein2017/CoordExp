"""Shared Stage-2 rollout runtime helpers.

This module coordinates Stage-2 rollout batches, backend lifecycle hooks,
evaluation rollout artifacts, and post-rollout packing helpers.  Shared prompt,
decode, backend-adapter, and trace contracts live under :mod:`src.infer`.
"""

from __future__ import annotations

import gc
import json
import math
import os
import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import torch
import torch.nn.functional as F
from transformers.trainer_utils import SaveStrategy
from swift.trainers import Seq2SeqTrainer
try:
    from swift.rlhf_trainers.utils import (
        get_gather_if_zero3_context,
        replace_assistant_response_with_ids,
    )
except ImportError:
    from swift.trainers.rlhf_trainer.utils import (
        get_gather_if_zero3_context,
        replace_assistant_response_with_ids,
    )
from swift.utils import get_logger

from src.common.object_field_order import (
    normalize_object_field_order,
    normalize_object_ordering,
)
from src.common.detection_sequence import (
    COORDJSON_FORMAT,
    normalize_detection_sequence_format,
)
from src.common.geometry import normalize_bbox_format
from src.config.prompts import (
    resolve_dense_prompt_variant_key,
)
from src.infer.artifacts import (
    build_stage2_rollout_eval_artifact_record,
    confidence_options_from_eval_config,
    score_stage2_confidence_eval_record,
)
from src.infer.parsing import (
    diagnostic_parser_result,
    parse_stage2_detection_rollout_predictions,
)
from src.infer.runtime import (
    build_decode_request_from_rollout_owner,
    build_decode_request_from_rollout_matching_config,
    current_rollout_context_from_owner,
    effective_rollout_backend_from_owner,
    normalize_rollout_backend_value,
    rollout_decode_batch_size_from_owner,
    vllm_mode_from_rollout_owner,
)
from src.training.stage2.rollout_codec import (
    CompactFullRolloutCodec,
    resolve_stage2_rollout_template_policy,
)
from src.coord_tokens.codec import (
    get_coord_token_ids,
)
from src.utils.metric_key_lookup import (
    metric_lookup_candidates,
    metric_name_matches_key,
    resolve_metric_value,
    stage2_eval_metric_key,
)
from .stage2_coordination import (
    ddp_assert_all_ranks_true_or_raise,
    reduce_metric_payload_global,
    resolve_rollout_log_metric_spec,
)
from src.infer.rollout_dispatch import (
    rollout_many,
    rollout_many_traced,
)
from src.infer.backend_vllm_engine import (
    best_effort_cleanup_vllm_sleep_mode_pools,
    best_effort_fix_vllm_nccl_allocator_atexit_order,
    best_effort_patch_vllm_cumem_sleep_no_empty_cache,
    ensure_vllm_engine,
    maybe_eval_vllm_colocate_window,
    sleep_vllm_engine,
    shutdown_vllm_colocate_engine,
    validate_vllm_eval_lifecycle_preflight,
    vllm_raw_engine_or_raise,
    vllm_reinit_each_eval,
    vllm_sleep_level,
    vllm_sleep_mode_enabled,
    wake_vllm_engine,
)
from src.infer.backend_vllm_server import (
    allocate_weighted_counts_with_caps as _allocate_weighted_counts_with_caps,
    contiguous_chunk_slices as _contiguous_chunk_slices,
    contiguous_weighted_chunk_slices as _contiguous_weighted_chunk_slices,
    effective_vllm_server_sync_mode,
    ensure_vllm_server_client,
    ensure_vllm_server_communicator_rank0,
    per_server_rank_request_caps as _per_server_rank_request_caps,
    rollout_decode_batch_size_per_rank,
    sync_vllm_server_rollout_model_if_needed,
    shutdown_vllm_server_client,
    vllm_server_cfg,
    vllm_server_specs,
    vllm_server_timeouts,
    vllm_server_world_sizes,
)
from .rollout_aligned_evaluator import (
    build_eval_detection_record as _build_eval_detection_record,
    build_eval_detection_record_confidence_postop_input as _build_eval_detection_record_confidence_postop_input,
    extract_eval_gt_objects as _extract_gt_objects,
    finalize_rollout_aligned_evaluation,
)
from .rollout_aligned_targets import (
    build_labels_and_coord_targets_for_batch,
    build_labels_and_coord_targets_for_sample,
)

from .rollout_matching.contracts import (
    GTObject,
)
from .rollout_matching.matching import _mask_iou_norm1000, greedy_match_iou
from .rollout_matching.packing import (
    DropRemainderAccumulationWindow as _DropRemainderAccumulationWindow,
)
from .rollout_matching.parsing import (
    parse_rollout_for_matching,
    points_from_coord_tokens as _points_from_coord_tokens,
    serialize_append_fragment as _serialize_append_fragment,
)
from .rollout_matching.telemetry import (
    PendingTrainRolloutLog as _PendingTrainRolloutLog,
)
logger = get_logger()


def _build_stage2_eval_invalid_artifact_record(
    *,
    eval_record_index: int,
    sample: Mapping[str, Any],
    parser_artifact_metadata: Mapping[str, Any],
    reason: str,
    message: str | None = None,
) -> Dict[str, Any]:
    images_raw = sample.get("images")
    images = list(images_raw) if isinstance(images_raw, list) else []
    metadata = sample.get("metadata")
    return {
        "index": int(eval_record_index),
        "sample_id": sample.get("sample_id"),
        "base_idx": sample.get("base_idx"),
        "image": sample.get("image"),
        "images": images,
        "width": sample.get("width"),
        "height": sample.get("height"),
        "image_id": sample.get("image_id"),
        "metadata": dict(metadata) if isinstance(metadata, Mapping) else None,
        "parser_result": dict(parser_artifact_metadata),
        "official_eval_invalid_reason": str(reason),
        "official_eval_invalid_message": str(message or reason),
    }


def _sinkhorn_barycentric_targets(
    *,
    pred_points: np.ndarray,  # [N,2] in norm1000
    gt_points: np.ndarray,  # [M,2] in norm1000
    epsilon: float,
    iters: int,
    cost: Literal["l1", "l2"],
) -> np.ndarray:
    """Compute barycentric-projected GT targets for each pred point via Sinkhorn OT."""
    if pred_points.size == 0 or gt_points.size == 0:
        return pred_points.copy()
    eps = float(epsilon)
    if not math.isfinite(eps) or eps <= 0:
        eps = 1.0
    n_iter = max(1, int(iters))

    p = torch.tensor(pred_points, dtype=torch.float32)
    g = torch.tensor(gt_points, dtype=torch.float32)
    if cost == "l1":
        c = torch.cdist(p, g, p=1)
    else:
        c = torch.cdist(p, g, p=2)

    # Uniform marginals.
    n = p.shape[0]
    m = g.shape[0]
    a = torch.full((n,), 1.0 / float(n), dtype=torch.float32)
    b = torch.full((m,), 1.0 / float(m), dtype=torch.float32)

    k = torch.exp((-c / eps).clamp(min=-50.0, max=50.0))
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    for _ in range(n_iter):
        kv = k @ v
        kv = torch.where(kv > 0, kv, torch.ones_like(kv))
        u = a / kv
        ku = k.t() @ u
        ku = torch.where(ku > 0, ku, torch.ones_like(ku))
        v = b / ku

    t = (u[:, None] * k) * v[None, :]
    row_sum = t.sum(dim=1, keepdim=True)
    row_sum = torch.where(row_sum > 0, row_sum, torch.ones_like(row_sum))
    w = t / row_sum
    g_hat = w @ g
    return g_hat.detach().cpu().numpy()


def _build_labels_and_coord_targets_for_sample(
    *,
    input_ids_1d: torch.Tensor,  # [T]
    prompt_len: int,
    prefix_len: int,
    train_len: int,
    coord_id_set: set[int],
    coord_id_to_bin: Mapping[int, int],
    prefix_coord_pos: Sequence[int],
    prefix_coord_target_bins: Sequence[int],
    tail_ignore_pos: Optional[Sequence[int]] = None,
    prefix_struct_pos: Optional[Sequence[int]] = None,
    tail_desc_pos: Optional[Sequence[int]] = None,
    tail_closure_pos: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, List[int], List[int], List[bool]]:
    return build_labels_and_coord_targets_for_sample(
        input_ids_1d=input_ids_1d,
        prompt_len=prompt_len,
        prefix_len=prefix_len,
        train_len=train_len,
        coord_id_set=coord_id_set,
        coord_id_to_bin=coord_id_to_bin,
        prefix_coord_pos=prefix_coord_pos,
        prefix_coord_target_bins=prefix_coord_target_bins,
        tail_ignore_pos=tail_ignore_pos,
        prefix_struct_pos=prefix_struct_pos,
        tail_desc_pos=tail_desc_pos,
        tail_closure_pos=tail_closure_pos,
    )


def _build_labels_and_coord_targets_for_batch(
    *,
    input_ids: torch.Tensor,  # [B, T]
    meta: List[Mapping[str, Any]],
    coord_id_set: set[int],
    coord_id_to_bin: Mapping[int, int],
) -> Tuple[torch.Tensor, List[int], List[int], List[int], List[bool]]:
    return build_labels_and_coord_targets_for_batch(
        input_ids=input_ids,
        meta=meta,
        coord_id_set=coord_id_set,
        coord_id_to_bin=coord_id_to_bin,
    )



class Stage2RolloutRuntime(Seq2SeqTrainer):
    """Shared Stage-2 rollout runtime base.

    This class owns rollout generation, vLLM/server dispatch, dynamic
    post-rollout packing, evaluation artifact support, and rollout observability.
    It is not a public trainer variant; concrete Stage-2 trainers provide the
    training-step and objective semantics.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._coord_token_ids: Optional[List[int]] = None
        self._coord_id_to_bin: Optional[Dict[int, int]] = None
        self._debug_dump_count: int = 0
        # Rank-local carry buffer for dynamic post-rollout packing (stage_2 only).
        # Each entry is (encoded, meta, encoded_len).
        self._post_rollout_segments: List[
            Tuple[Dict[str, Any], Dict[str, Any], int]
        ] = []

        # vLLM rollout backend state (lazy init).
        self._vllm_engine: Any = None
        self._vllm_tp_group: Any = None
        self._vllm_tp_size: int = 1
        self._vllm_last_loaded_step: int = -1

        # vLLM server-mode rollout backend state (lazy init).
        self._vllm_server_client: Any = None
        self._vllm_server_client_lock = threading.Lock()
        self._vllm_server_comm_inited: bool = False
        self._vllm_server_last_synced_step: int = -1
        self._vllm_server_debug_dump_count: int = 0
        self._vllm_server_debug_last_step: Optional[int] = None
        self._vllm_server_last_logged_step: int = -1
        self._vllm_server_force_full_sync: bool = False

        # Buffered training logs: accumulate across micro-batches and merge into the step log.
        # Keyed by the *post-optimizer* global_step (HF logs after increment).
        self._rm_pending_train_logs: Dict[int, _PendingTrainRolloutLog] = {}

        # Qualitative train dumps (rank0 only): rollout vs GT vs training target.
        self._monitor_dump_last_step: Optional[int] = None
        self._monitor_dump_count: int = 0
        # Qualitative eval dumps use a separate cadence and budget namespace.
        self._eval_monitor_dump_eval_index: int = 0
        self._eval_monitor_dump_last_eval: Optional[int] = None
        self._eval_monitor_dump_count: int = 0

        # Optional semantic desc monitoring (lazy init; metrics only).
        self._desc_semantic_encoder: Any = None
        self._desc_semantic_encoder_sig: Any = None

        # Eval-only vLLM rollout window lifecycle state.
        self._eval_vllm_window_active: bool = False
        self._vllm_eval_lifecycle_preflight_done: bool = False

        # Mutable config injected by src/sft.py after construction.
        self.rollout_matching_cfg: Mapping[str, Any] = {}
        self._stage_wallclock_totals_s: Dict[str, float] = {
            "sft": 0.0,
            "rollout": 0.0,
        }

    def _coordexp_checkpoint_runtime_state(self) -> Dict[str, Any]:
        return {
            "post_rollout_segments": list(self._post_rollout_segments),
            "rm_pending_train_logs": {
                int(step): asdict(pending)
                for step, pending in self._rm_pending_train_logs.items()
            },
            "monitor_dump_last_step": self._monitor_dump_last_step,
            "monitor_dump_count": int(self._monitor_dump_count),
            "eval_monitor_dump_eval_index": int(self._eval_monitor_dump_eval_index),
            "eval_monitor_dump_last_eval": self._eval_monitor_dump_last_eval,
            "eval_monitor_dump_count": int(self._eval_monitor_dump_count),
        }

    def _coordexp_restore_checkpoint_runtime_state(
        self, payload: Mapping[str, Any]
    ) -> None:
        if not isinstance(payload, Mapping):
            raise TypeError(
                "Stage2RolloutRuntime checkpoint runtime state must be a Mapping"
            )

        post_rollout_segments = payload.get("post_rollout_segments")
        if isinstance(post_rollout_segments, list):
            self._post_rollout_segments = list(post_rollout_segments)

        pending_logs_raw = payload.get("rm_pending_train_logs")
        if isinstance(pending_logs_raw, Mapping):
            restored_pending: Dict[int, _PendingTrainRolloutLog] = {}
            for step_raw, pending_payload in pending_logs_raw.items():
                if not isinstance(pending_payload, Mapping):
                    continue
                restored_pending[int(step_raw)] = _PendingTrainRolloutLog(
                    **dict(pending_payload)
                )
            self._rm_pending_train_logs = restored_pending

        self._monitor_dump_last_step = payload.get("monitor_dump_last_step")
        self._monitor_dump_count = int(payload.get("monitor_dump_count", 0) or 0)
        self._eval_monitor_dump_eval_index = int(
            payload.get("eval_monitor_dump_eval_index", 0) or 0
        )
        self._eval_monitor_dump_last_eval = payload.get("eval_monitor_dump_last_eval")
        self._eval_monitor_dump_count = int(
            payload.get("eval_monitor_dump_count", 0) or 0
        )
        self._stage_wallclock_lock = threading.Lock()

    def _ensure_stage_wallclock_state(self) -> None:
        totals = getattr(self, "_stage_wallclock_totals_s", None)
        if not isinstance(totals, dict):
            totals = {"sft": 0.0, "rollout": 0.0}
            setattr(self, "_stage_wallclock_totals_s", totals)
        for key in ("sft", "rollout"):
            try:
                totals[key] = float(totals.get(key, 0.0) or 0.0)
            except (AttributeError, TypeError, ValueError):
                totals[key] = 0.0

        lock = getattr(self, "_stage_wallclock_lock", None)
        if lock is None or not hasattr(lock, "acquire") or not hasattr(lock, "release"):
            setattr(self, "_stage_wallclock_lock", threading.Lock())

    def _record_stage_wallclock_span(
        self,
        *,
        stage: Literal["sft", "rollout"],
        start_ts: float,
        end_ts: Optional[float] = None,
    ) -> float:
        if stage not in {"sft", "rollout"}:
            raise ValueError(f"unknown stage wallclock timer: {stage!r}")

        self._ensure_stage_wallclock_state()

        stop_ts = float(time.perf_counter() if end_ts is None else end_ts)
        delta = float(stop_ts - float(start_ts))
        if (not math.isfinite(delta)) or delta <= 0.0:
            return 0.0

        totals = getattr(self, "_stage_wallclock_totals_s")
        lock = getattr(self, "_stage_wallclock_lock")
        with lock:
            totals[str(stage)] = float(totals.get(str(stage), 0.0) or 0.0) + float(delta)
        return float(delta)

    @contextmanager
    def _track_stage_wallclock(self, stage: Literal["sft", "rollout"]):
        start_ts = time.perf_counter()
        try:
            yield
        finally:
            self._record_stage_wallclock_span(stage=stage, start_ts=float(start_ts))

    def _stage_wallclock_metrics_local(self) -> Dict[str, float]:
        self._ensure_stage_wallclock_state()
        totals = getattr(self, "_stage_wallclock_totals_s")
        return {
            "time/sft_total_time": float(totals.get("sft", 0.0) or 0.0),
            "time/rollout_total_time": float(totals.get("rollout", 0.0) or 0.0),
        }

    def _reduce_stage_wallclock_metrics_global(
        self, metrics: Mapping[str, Any]
    ) -> Dict[str, float]:
        reduced: Dict[str, float] = {}
        for k, v in metrics.items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(fv):
                continue
            reduced[str(k)] = float(fv)

        if not reduced:
            return {}

        try:
            import torch.distributed as dist
        except (AttributeError, RuntimeError, TypeError, ValueError):
            dist = None  # type: ignore[assignment]

        rank = 0
        world_size = 1
        if dist is not None and dist.is_available() and dist.is_initialized():
            try:
                world_size = int(dist.get_world_size())
            except (TypeError, ValueError, RuntimeError):
                world_size = 1
            try:
                rank = int(dist.get_rank())
            except (TypeError, ValueError, RuntimeError):
                rank = 0

        metric_keys = sorted(reduced.keys())
        if (
            dist is not None
            and dist.is_available()
            and dist.is_initialized()
            and int(world_size) > 1
            and metric_keys
        ):
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

                values = torch.tensor(
                    [float(reduced[k]) for k in metric_keys],
                    dtype=torch.float64,
                    device=device,
                )
                dist.all_reduce(values, op=dist.ReduceOp.MAX)
                for idx, key in enumerate(metric_keys):
                    reduced[key] = float(values[idx].item())
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    "stage wallclock metric all-reduce failed (DDP is initialized); "
                    f"rank={int(rank)}/{int(world_size)}"
                ) from exc

        return reduced

    def _merge_rollout_matching_batch_metrics(
        self, batch: MutableMapping[str, Any], metrics: Mapping[str, Any]
    ) -> None:
        """Merge rollout-matching batch metrics onto an existing batch.

        Treat `_rollout_matching_batch_metrics` as merge-only so that later pipeline
        stages (packing, async prefetch, post-processing) can add telemetry without
        losing base rollout/decode metrics.
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

    # ------------------------ config helpers ------------------------ #
    def _cfg(self, key: str, default: Any) -> Any:
        cfg = getattr(self, "rollout_matching_cfg", None)
        if not isinstance(cfg, Mapping):
            return default
        return cfg.get(str(key), default)

    @staticmethod
    def _normalize_rollout_backend_value(
        raw: Any,
        *,
        key_path: str,
        allow_none: bool,
    ) -> Optional[Literal["hf", "vllm"]]:
        return normalize_rollout_backend_value(
            raw,
            key_path=key_path,
            allow_none=allow_none,
        )

    def _effective_rollout_backend(
        self, *, context: Literal["train", "eval"] = "train"
    ) -> Literal["hf", "vllm"]:
        return effective_rollout_backend_from_owner(self, context=context)

    def _object_field_order(self) -> Literal["desc_first", "geometry_first"]:
        raw = getattr(self, "object_field_order", None)
        if raw is None:
            raw = self._cfg("object_field_order", "desc_first")
        return normalize_object_field_order(raw, path="custom.object_field_order")

    def _object_ordering(self) -> Literal["sorted", "random"]:
        return normalize_object_ordering(
            self._cfg("object_ordering", "sorted"),
            path="custom.object_ordering",
        )

    def _eval_prompt_variant(self) -> Optional[str]:
        raw = self._cfg("eval_prompt_variant", None)
        if raw is None:
            return None
        if not isinstance(raw, str):
            raise TypeError(
                "rollout_matching.eval_prompt_variant must be a string when provided"
            )
        key = raw.strip()
        if not key:
            return None
        return resolve_dense_prompt_variant_key(key)

    def _training_prompt_variant(self) -> Optional[str]:
        raw = self._cfg("prompt_variant", None)
        if raw is None:
            return None
        if not isinstance(raw, str):
            raise TypeError(
                "rollout_matching.prompt_variant must be a string when provided"
            )
        key = raw.strip()
        if not key:
            return None
        return resolve_dense_prompt_variant_key(key)

    def _detection_sequence_format(self) -> str:
        return normalize_detection_sequence_format(
            self._cfg("detection_sequence_format", COORDJSON_FORMAT)
        )

    def _eval_rollout_template_policy(self):
        resolver = getattr(self, "_resolve_stage2_rollout_template_policy", None)
        if callable(resolver):
            return resolver()
        return resolve_stage2_rollout_template_policy(
            self._detection_sequence_format()
        )

    def _eval_detection_cfg(self) -> Mapping[str, Any]:
        default_cfg: Dict[str, Any] = {
            "enabled": True,
            "metrics": "coco",
            "score_mode": "constant",
            "constant_score": 1.0,
            "pred_score_source": "eval_rollout_constant",
            "pred_score_version": 2,
        }
        raw = self._cfg("eval_detection", default_cfg)
        if raw is None:
            return dict(default_cfg)
        if not isinstance(raw, Mapping):
            raise TypeError("rollout_matching.eval_detection must be a mapping")
        merged = dict(default_cfg)
        merged.update(dict(raw))
        return merged

    def _eval_decode_override(self, *, has_token_trace: bool) -> Optional[Dict[str, Any]]:
        """Resolve eval-only decode overrides without relaxing shared runtime validation."""

        cfg = getattr(self, "rollout_matching_cfg", {}) or {}
        if not isinstance(cfg, Mapping):
            return None
        dec = cfg.get("decoding", {}) or {}
        if not isinstance(dec, Mapping):
            dec = {}

        try:
            temperature = float(dec.get("temperature", 0.0) or 0.0)
        except (TypeError, ValueError):
            temperature = 0.0
        decode_mode = str(cfg.get("decode_mode", "") or "").strip().lower()

        # Confidence post-op needs generated-token logprobs from a deterministic
        # rollout. Train-time rollout-correction configs may still declare
        # decode_mode=sampling because per-attempt temperature overrides drive
        # the explorer rollouts; eval should use the canonical greedy pass.
        if bool(has_token_trace) or (decode_mode == "sampling" and temperature <= 0.0):
            return {
                "decode_mode": "greedy",
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
            }
        return None

    def _write_eval_phase_trace(
        self,
        *,
        phase: str,
        global_step: int,
        eval_index: int,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        """Write a small eval-progress breadcrumb for long vLLM/DDP runs."""

        args_obj = getattr(self, "args", None)
        output_dir = str(getattr(args_obj, "output_dir", ".") or ".")
        if not output_dir.strip():
            output_dir = "."

        safe_phase = "".join(
            ch if (str(ch).isalnum() or ch in ("-", "_")) else "_"
            for ch in str(phase)
        ).strip("_")
        if not safe_phase:
            safe_phase = "unknown_phase"

        rank = 0
        world_size = 1
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                rank = int(dist.get_rank())
                world_size = int(dist.get_world_size())
        except (RuntimeError, TypeError, ValueError):
            rank = 0
            world_size = 1

        out_dir = os.path.join(output_dir, "monitor_dumps", "eval_phase_trace")
        record = {
            "kind": "eval_phase_trace",
            "global_step": int(global_step),
            "eval_index": int(eval_index),
            "epoch": float(
                getattr(getattr(self, "state", None), "epoch", 0.0) or 0.0
            ),
            "time": float(time.time()),
            "meta": {
                "phase": "eval",
                "stage2_surface": "rollout_correction",
                "eval_phase": str(phase),
                "rank": int(rank),
                "world_size": int(world_size),
            },
            "payload": dict(payload) if isinstance(payload, Mapping) else {},
        }

        try:
            os.makedirs(out_dir, exist_ok=True)
            trace_path = os.path.join(
                out_dir,
                (
                    f"eval_{int(eval_index):04d}_step_{int(global_step):06d}_"
                    f"rank{int(rank):02d}_{safe_phase}.json"
                ),
            )
            with open(trace_path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=True, indent=2)
        except (OSError, TypeError, ValueError) as exc:
            logger.warning(
                "Failed to write rollout-correction eval phase trace for "
                "global_step=%s eval_index=%s phase=%s rank=%s/%s: %r",
                int(global_step),
                int(eval_index),
                str(phase),
                int(rank),
                int(world_size),
                exc,
            )

    def _validate_rollout_matching_cfg(self) -> None:
        cfg = getattr(self, "rollout_matching_cfg", None)
        if cfg is None:
            return
        if not isinstance(cfg, Mapping):
            raise TypeError(
                "rollout_matching_cfg must be a mapping (injected from rollout_matching)"
            )
        bbox_format = normalize_bbox_format(
            cfg.get("bbox_format", "xyxy"),
            path="rollout_matching.bbox_format",
        )
        if bbox_format != "xyxy":
            raise ValueError(
                "rollout_matching.bbox_format must remain 'xyxy' for stage-2 trainers "
                "until target construction is updated for alternate bbox parameterizations."
            )

        removed = [
            k
            for k in (
                # Older top-level decoding knobs.
                "temperature",
                "top_p",
                "top_k",
                # Removed buffer reuse.
                "rollout_buffer",
                # Legacy batching knobs (replaced by explicit per-context decode batch sizes).
                "decode_batch_size",
                "rollout_generate_batch_size",
                "rollout_infer_batch_size",
                "channel_b_decode_batch_size",
                # Removed packing-scope knob.
                "post_rollout_pack_scope",
            )
            if k in cfg
        ]
        if removed:
            rendered: List[str] = []
            for k in removed:
                if k in {
                    "decode_batch_size",
                    "rollout_generate_batch_size",
                    "rollout_infer_batch_size",
                    "channel_b_decode_batch_size",
                }:
                    rendered.append(
                        "rollout_matching."
                        f"{k} (use rollout_matching.rollout_decode_batch_size / rollout_matching.eval_decode_batch_size)"
                    )
                elif k == "post_rollout_pack_scope":
                    rendered.append(
                        f"rollout_matching.{k} (remove; micro-scope packing is standard)"
                    )
                else:
                    rendered.append(f"rollout_matching.{k}")

            legacy_s = ", ".join(rendered)
            raise ValueError(
                "Legacy rollout-matching keys have been removed: "
                f"{legacy_s}. (No backward compatibility.)"
            )

        # Validate explicit per-context decode batch-size knobs.
        rollout_decode_bs_raw = cfg.get("rollout_decode_batch_size", None)
        if rollout_decode_bs_raw is None:
            raise ValueError(
                "rollout_matching.rollout_decode_batch_size must be provided explicitly"
            )
        try:
            rollout_decode_bs = int(rollout_decode_bs_raw)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "rollout_matching.rollout_decode_batch_size must be an int"
            ) from exc
        if rollout_decode_bs <= 0:
            raise ValueError(
                "rollout_matching.rollout_decode_batch_size must be > 0"
            )

        eval_decode_bs_raw = cfg.get("eval_decode_batch_size", None)
        if eval_decode_bs_raw is None:
            raise ValueError(
                "rollout_matching.eval_decode_batch_size must be provided explicitly"
            )
        try:
            eval_decode_bs = int(eval_decode_bs_raw)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "rollout_matching.eval_decode_batch_size must be an int"
            ) from exc
        if eval_decode_bs <= 0:
            raise ValueError(
                "rollout_matching.eval_decode_batch_size must be > 0"
            )

        dec = cfg.get("decoding", None)
        if dec is None:
            dec = {}
        if not isinstance(dec, Mapping):
            raise TypeError(
                "rollout_matching.decoding must be a mapping when provided"
            )

        # Validate decoding ranges (robust defaults).
        try:
            temperature = float(dec.get("temperature", 0.0) or 0.0)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "rollout_matching.decoding.temperature must be a float"
            ) from exc
        if temperature < 0.0:
            raise ValueError(
                "rollout_matching.decoding.temperature must be >= 0"
            )

        try:
            top_p = float(
                dec.get("top_p", 1.0) if dec.get("top_p", None) is not None else 1.0
            )
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "rollout_matching.decoding.top_p must be a float"
            ) from exc
        if not (0.0 < top_p <= 1.0):
            raise ValueError(
                "rollout_matching.decoding.top_p must be in (0, 1]"
            )

        top_k_raw = dec.get("top_k", -1)
        try:
            top_k = int(top_k_raw)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "rollout_matching.decoding.top_k must be an int"
            ) from exc
        if top_k != -1 and top_k < 1:
            raise ValueError(
                "rollout_matching.decoding.top_k must be -1 (disabled) or >= 1"
            )

        train_backend = self._effective_rollout_backend(context="train")
        eval_backend = self._effective_rollout_backend(context="eval")
        vllm_cfg = cfg.get("vllm", {})
        if vllm_cfg is None:
            vllm_cfg = {}
        if not isinstance(vllm_cfg, Mapping):
            raise TypeError("rollout_matching.vllm must be a mapping when provided")
        if train_backend == "vllm" or eval_backend == "vllm":
            enable_lora = bool(vllm_cfg.get("enable_lora", False))
            vllm_mode = str(vllm_cfg.get("mode", "") or "").strip().lower()
            if enable_lora and vllm_mode not in {"server", ""}:
                raise ValueError(
                    "rollout_matching.vllm.enable_lora=true is currently supported "
                    "only for rollout_matching.vllm.mode=server."
                )
            sync_raw = vllm_cfg.get("sync", {}) or {}
            if sync_raw is not None and not isinstance(sync_raw, Mapping):
                raise TypeError("rollout_matching.vllm.sync must be a mapping")
            sync_mode = (
                str(sync_raw.get("mode", "full") or "full").strip().lower()
                if isinstance(sync_raw, Mapping)
                else "full"
            )
            if sync_mode not in {"full", "adapter"}:
                raise ValueError(
                    "rollout_matching.vllm.sync.mode must be one of {'full', 'adapter'}."
                )
            if sync_mode != "adapter":
                raise ValueError(
                    "vLLM rollouts require official adapter sync: set "
                    "rollout_matching.vllm.sync.mode=adapter."
                )
            if not enable_lora:
                raise ValueError(
                    "vLLM rollouts require official adapter sync: set "
                    "rollout_matching.vllm.enable_lora=true."
                )
            if enable_lora and sync_mode != "adapter":
                raise ValueError(
                    "rollout_matching.vllm.enable_lora=true requires "
                    "rollout_matching.vllm.sync.mode=adapter."
                )
            if sync_mode == "adapter" and not enable_lora:
                raise ValueError(
                    "rollout_matching.vllm.sync.mode=adapter requires "
                    "rollout_matching.vllm.enable_lora=true."
                )

        # Legacy sleep-mode lifecycle is removed.
        if bool(vllm_cfg.get("enable_sleep_mode", False)):
            raise ValueError(
                "rollout_matching.vllm.enable_sleep_mode is no longer supported."
            )

        reinit_each_eval_raw = vllm_cfg.get("reinit_each_eval", False)
        if not isinstance(reinit_each_eval_raw, bool):
            raise TypeError(
                "rollout_matching.vllm.reinit_each_eval must be a bool"
            )
        if bool(reinit_each_eval_raw):
            if eval_backend != "vllm":
                raise ValueError(
                    "rollout_matching.vllm.reinit_each_eval requires eval_rollout_backend=vllm."
                )
            mode_raw = str(vllm_cfg.get("mode", "colocate") or "colocate").strip().lower()
            if mode_raw != "colocate":
                raise ValueError(
                    "rollout_matching.vllm.reinit_each_eval requires rollout_matching.vllm.mode=colocate."
                )

        sleep_level_raw = vllm_cfg.get("sleep_level", 0)
        try:
            sleep_level = int(0 if sleep_level_raw is None else sleep_level_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "rollout_matching.vllm.sleep_level must be an int"
            ) from exc
        if sleep_level != 0:
            raise ValueError(
                "rollout_matching.vllm.sleep_level is no longer supported (must be 0)."
            )

        for prompt_variant_key in ("prompt_variant", "eval_prompt_variant"):
            prompt_variant_raw = cfg.get(prompt_variant_key, None)
            if prompt_variant_raw is None:
                continue
            if not isinstance(prompt_variant_raw, str):
                raise TypeError(
                    f"rollout_matching.{prompt_variant_key} must be a string when provided"
                )
            if prompt_variant_raw.strip():
                resolve_dense_prompt_variant_key(prompt_variant_raw.strip())

        pipeline_raw = cfg.get("pipeline", None)
        if pipeline_raw is not None:
            raise ValueError(
                "rollout_matching.pipeline has been removed. "
                "Use stage2_rollout_correction.pipeline with custom.trainer_variant=stage2_rollout_correction instead."
            )

        eval_det_raw = cfg.get("eval_detection", None)
        if eval_det_raw is not None:
            if not isinstance(eval_det_raw, Mapping):
                raise TypeError("rollout_matching.eval_detection must be a mapping")
            metrics_mode = str(
                eval_det_raw.get("metrics", "coco") or "coco"
            ).strip().lower()
            if metrics_mode not in {"coco", "lvis", "f1ish", "both"}:
                raise ValueError(
                    "rollout_matching.eval_detection.metrics must be one of {'coco', 'lvis', 'f1ish', 'both'}"
                )
            score_mode = str(
                eval_det_raw.get("score_mode", "constant") or "constant"
            ).strip().lower()
            if score_mode not in {"constant", "confidence_postop"}:
                raise ValueError(
                    "rollout_matching.eval_detection.score_mode must be one of {'constant', 'confidence_postop'}"
                )
            try:
                score = float(eval_det_raw.get("constant_score", 1.0) or 1.0)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "rollout_matching.eval_detection.constant_score must be numeric"
                ) from exc
            if score < 0.0 or score > 1.0:
                raise ValueError(
                    "rollout_matching.eval_detection.constant_score must satisfy 0.0 <= score <= 1.0"
                )
            score_source = str(
                eval_det_raw.get("pred_score_source", "") or ""
            ).strip()
            if not score_source:
                raise ValueError(
                    "rollout_matching.eval_detection.pred_score_source must be non-empty"
                )
            try:
                int(eval_det_raw.get("pred_score_version", 2))
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "rollout_matching.eval_detection.pred_score_version must be int-compatible"
                ) from exc

    def _monitor_dump_cfg(self) -> Mapping[str, Any]:
        return self._train_monitor_dump_cfg()

    def _train_monitor_dump_cfg(self) -> Mapping[str, Any]:
        cfg = self._cfg("train_monitor_dump", None)
        if cfg is None:
            cfg = self._cfg("monitor_dump", {}) or {}
        return cfg if isinstance(cfg, Mapping) else {}

    def _eval_monitor_dump_cfg(self) -> Mapping[str, Any]:
        cfg = self._cfg("eval_monitor_dump", None)
        if cfg is None:
            cfg = self._cfg("monitor_dump", {}) or {}
        return cfg if isinstance(cfg, Mapping) else {}

    def _monitor_dump_cfg_for_payload(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        kind = str(payload.get("kind", "") or "").strip().lower()
        if kind == "eval_monitor_dump":
            return self._eval_monitor_dump_cfg()
        return self._train_monitor_dump_cfg()

    def _desc_monitor_cfg(self) -> Mapping[str, Any]:
        cfg = self._cfg("desc_monitor", {}) or {}
        return cfg if isinstance(cfg, Mapping) else {}

    def _get_desc_semantic_encoder(self, cfg: Mapping[str, Any]) -> Any:
        """Return a cached semantic encoder instance, or None if disabled/unavailable."""

        mode = str(cfg.get("mode", "semantic") or "semantic").strip().lower()
        if mode not in {"semantic", "both"}:
            return None

        try:
            from src.metrics.semantic_desc import SemanticDescEncoder
        except (ImportError, OSError, RuntimeError) as exc:
            warned = bool(
                getattr(self, "_coordexp_desc_semantic_import_warned", False)
            )
            if not warned:
                logger.warning(
                    "Semantic desc encoder disabled (import failed): %r",
                    exc,
                )
                setattr(self, "_coordexp_desc_semantic_import_warned", True)
            return None

        model_name = str(
            cfg.get("semantic_model", "sentence-transformers/all-MiniLM-L6-v2")
        )
        device = str(cfg.get("semantic_device", "cpu"))
        batch_size = int(cfg.get("semantic_batch_size", 64) or 64)
        max_length = int(cfg.get("semantic_max_length", 64) or 64)

        sig = (model_name, device, batch_size, max_length)
        enc = getattr(self, "_desc_semantic_encoder", None)
        enc_sig = getattr(self, "_desc_semantic_encoder_sig", None)
        if enc is not None and enc_sig == sig:
            return enc

        enc = SemanticDescEncoder(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
        setattr(self, "_desc_semantic_encoder", enc)
        setattr(self, "_desc_semantic_encoder_sig", sig)
        return enc

    def _is_main_process(self) -> bool:
        acc = getattr(self, "accelerator", None)
        if acc is not None and hasattr(acc, "is_main_process"):
            try:
                return bool(acc.is_main_process)
            except (TypeError, ValueError):
                raise
        return bool(getattr(self, "is_world_process_zero", False))

    def _monitor_dump_step_allowed(self, *, global_step: int) -> bool:
        cfg = self._train_monitor_dump_cfg()
        if not bool(cfg.get("enabled", False)):
            return False
        if (
            bool(cfg.get("only_world_process_zero", True))
            and not self._is_main_process()
        ):
            return False

        max_events = int(cfg.get("max_events", 20) or 0)
        monitor_dump_count = int(getattr(self, "_monitor_dump_count", 0) or 0)
        if max_events > 0 and monitor_dump_count >= max_events:
            return False

        gs = int(global_step)
        last_step = getattr(self, "_monitor_dump_last_step", None)
        if last_step is not None and int(last_step) == gs:
            return False
        return True

    def _should_monitor_dump(self, *, global_step: int) -> bool:
        if not self._monitor_dump_step_allowed(global_step=global_step):
            return False

        cfg = self._train_monitor_dump_cfg()
        gs = int(global_step)

        every = cfg.get("every_steps", None)
        if every is None:
            args_obj = getattr(self, "args", None)
            every = int(getattr(args_obj, "logging_steps", 1) or 1)
        every = max(1, int(every))

        args_obj = getattr(self, "args", None)
        dump_first = bool(
            cfg.get(
                "dump_first_step",
                bool(getattr(args_obj, "logging_first_step", False)),
            )
        )
        if gs == 0 and not dump_first:
            return False
        if gs % every != 0:
            return False
        return True

    def _should_eval_monitor_dump(self, *, global_step: int, eval_index: int) -> bool:
        cfg = self._eval_monitor_dump_cfg()
        if not bool(cfg.get("enabled", False)):
            return False
        if (
            bool(cfg.get("only_world_process_zero", True))
            and not self._is_main_process()
        ):
            return False

        max_events = int(cfg.get("max_events", 20) or 0)
        eval_dump_count = int(getattr(self, "_eval_monitor_dump_count", 0) or 0)
        if max_events > 0 and eval_dump_count >= max_events:
            return False

        last_eval = getattr(self, "_eval_monitor_dump_last_eval", None)
        if last_eval is not None and int(last_eval) == int(eval_index):
            return False

        every_evals_raw = cfg.get("every_evals", 1)
        try:
            every_evals = int(every_evals_raw) if every_evals_raw is not None else 1
        except Exception:
            every_evals = 1
        every_evals = max(1, int(every_evals))
        if int(eval_index) % every_evals != 0:
            return False
        return True

    @staticmethod
    def _clip_text(text: Any, *, max_chars: int) -> str:
        s = ""
        try:
            s = str(text)
        except Exception:
            s = ""
        if max_chars <= 0:
            return s
        if len(s) <= max_chars:
            return s
        return s[:max_chars] + "...<truncated>"


    def _dump_warn_once(self, key: str, message: str, *args: object) -> None:
        warned = getattr(self, "_coordexp_dump_warned_once", None)
        if not isinstance(warned, set):
            warned = set()
            setattr(self, "_coordexp_dump_warned_once", warned)
        if key in warned:
            return
        warned.add(key)
        logger.warning(message, *args)

    def _submit_dump_write(
        self,
        *,
        kind: str,
        async_write: bool,
        max_pending_writes: int,
        fn: Any,
    ) -> None:
        """Best-effort dump writer.

        Dumps are diagnostics only and must never crash training.

        When `async_write=true`, I/O happens in a single-thread executor and is
        bounded by `max_pending_writes` to prevent unbounded memory growth.
        """

        if not async_write:
            try:
                fn()
            except Exception as exc:
                self._dump_warn_once(
                    f"{kind}_write_failed_sync",
                    "%s dump write failed: %r",
                    kind,
                    exc,
                )
            return

        try:
            max_pending_writes = max(1, int(max_pending_writes))
        except Exception:
            max_pending_writes = 2

        executor = getattr(self, "_coordexp_dump_executor", None)
        pending = getattr(self, "_coordexp_dump_futures", None)
        if executor is None or pending is None:
            from collections import deque
            from concurrent.futures import ThreadPoolExecutor

            executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="coordexp_dump_io"
            )
            pending = deque()
            setattr(self, "_coordexp_dump_executor", executor)
            setattr(self, "_coordexp_dump_futures", pending)

        # Drop completed futures to bound memory.
        try:
            while pending and pending[0].done():
                pending.popleft()
        except Exception:
            from collections import deque

            pending = deque()
            setattr(self, "_coordexp_dump_futures", pending)

        try:
            pending_len = int(len(pending))
        except Exception:
            pending_len = 0

        if pending_len >= max_pending_writes:
            self._dump_warn_once(
                f"{kind}_queue_full",
                "Skipping %s dump write: async queue full (pending=%s, max_pending_writes=%s).",
                kind,
                pending_len,
                max_pending_writes,
            )
            return

        def _run() -> None:
            try:
                fn()
            except Exception as exc:
                # Dumps are diagnostics only; never crash training.
                logger.warning("%s dump write failed (async): %r", kind, exc)

        submitted = False
        try:
            fut = executor.submit(_run)
            pending.append(fut)
            submitted = True
        except Exception as exc:
            self._dump_warn_once(
                f"{kind}_submit_failed",
                "Failed to submit %s dump write (async): %r",
                kind,
                exc,
            )
            submitted = False

        if not submitted:
            return

    @staticmethod
    def _ascii_safe_text(text: str) -> str:
        # Keep dumps ASCII by escaping non-ASCII characters, but preserve newlines.
        out: List[str] = []
        for ch in text:
            if ord(ch) < 128:
                out.append(ch)
            else:
                out.append("\\u%04x" % ord(ch))
        return "".join(out)

    def _write_monitor_dump(
        self, *, global_step: int, payload: Mapping[str, Any]
    ) -> None:
        cfg = self._monitor_dump_cfg_for_payload(payload)
        out_dir = cfg.get("out_dir")
        if not isinstance(out_dir, str) or not out_dir.strip():
            out_dir = os.path.join(
                str(getattr(self.args, "output_dir", ".")), "monitor_dumps"
            )

        # Monitor dumps are diagnostic artifacts only. Do not crash training on
        # I/O errors (e.g. intermittent filesystem issues, full disks). Emit a
        # warning and continue.
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError as exc:
            logger.warning("Failed to create monitor dump dir %s: %r", out_dir, exc)
            return

        min_free_gb_raw = cfg.get("min_free_gb", 2.0)
        try:
            min_free_gb = float(min_free_gb_raw) if min_free_gb_raw is not None else 0.0
        except Exception:
            min_free_gb = 2.0
        min_free_gb = max(0.0, float(min_free_gb))
        if min_free_gb > 0:
            try:
                import shutil

                usage = shutil.disk_usage(out_dir)
                free_gb = float(usage.free) / float(1024**3)
                if free_gb < min_free_gb:
                    self._dump_warn_once(
                        "monitor_dump_low_disk",
                        "Skipping monitor dump write: free disk %.2f GB < min_free_gb=%.2f at %s.",
                        free_gb,
                        min_free_gb,
                        out_dir,
                    )
                    return
            except Exception as exc:
                self._dump_warn_once(
                    "monitor_dump_disk_check_failed",
                    "Failed to check disk usage for monitor dump dir %s: %r",
                    out_dir,
                    exc,
                )

        async_write = bool(cfg.get("async_write", True))
        max_pending_raw = cfg.get("max_pending_writes", 2)
        try:
            max_pending_writes = (
                int(max_pending_raw) if max_pending_raw is not None else 2
            )
        except Exception:
            max_pending_writes = 2
        max_pending_writes = max(1, int(max_pending_writes))

        write_markdown = bool(cfg.get("write_markdown", True))

        def _write() -> None:
            # One file per optimizer step by default (easy to inspect while training).
            step_path = os.path.join(out_dir, f"step_{int(global_step):06d}.json")
            try:
                with open(step_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=True, indent=2)
            except (OSError, TypeError, ValueError) as exc:
                logger.warning(
                    "Failed to write monitor dump json %s: %r", step_path, exc
                )
                return

            if write_markdown:
                md_path = os.path.join(out_dir, f"step_{int(global_step):06d}.md")
                try:
                    md = self._format_monitor_dump_markdown(payload)
                    with open(md_path, "w", encoding="utf-8") as f:
                        f.write(md)
                except Exception as exc:
                    logger.warning(
                        "Failed to write monitor dump markdown %s: %r", md_path, exc
                    )

        self._submit_dump_write(
            kind="monitor_dump",
            async_write=async_write,
            max_pending_writes=max_pending_writes,
            fn=_write,
        )

    def _format_monitor_dump_markdown(self, payload: Mapping[str, Any]) -> str:
        # Human-readable dump; keep it ASCII-safe to avoid surprising tooling issues.
        kind = str(payload.get("kind", "") or "").strip().lower()
        if kind == "train_monitor_dump":
            max_chars = 0
        else:
            cfg = self._monitor_dump_cfg_for_payload(payload)
            max_chars_raw = cfg.get("max_text_chars", 4000)
            try:
                # Contract: <=0 disables clipping (full text).
                max_chars = int(max_chars_raw) if max_chars_raw is not None else 4000
            except Exception:
                max_chars = 4000
            max_chars = max(0, int(max_chars))

        def _j(obj: Any) -> str:
            try:
                return json.dumps(obj, ensure_ascii=True, indent=2)
            except (TypeError, ValueError):
                return "{}"

        lines: List[str] = []
        gs = payload.get("global_step")
        lines.append(f"# Rollout-Matching Monitor Dump (global_step={gs})\n")
        meta = payload.get("meta") or {}
        lines.append("## Meta\n")
        lines.append("```json\n" + _j(meta) + "\n```\n")

        samples = (
            payload.get("samples") if isinstance(payload.get("samples"), list) else []
        )
        for i, s in enumerate(samples):
            if not isinstance(s, Mapping):
                continue
            lines.append(f"## Sample {i}\n")
            sid = s.get("sample_id")
            bidx = s.get("base_idx")
            image_id = s.get("image_id")
            img = s.get("image") or s.get("images")
            lines.append(f"- sample_id: `{sid}`\n")
            lines.append(f"- base_idx: `{bidx}`\n")
            lines.append(f"- image_id: `{image_id}`\n")
            lines.append(f"- image(s): `{img}`\n\n")

            lines.append("### Messages\n")
            lines.append("```json\n" + _j(s.get("messages")) + "\n```\n")

            lines.append("### Rollout (raw)\n")
            lines.append(
                "```text\n"
                + self._ascii_safe_text(
                    self._clip_text(s.get("rollout_text"), max_chars=max_chars)
                )
                + "\n```\n"
            )
            lines.append("### Prefix Used (append-ready)\n")
            lines.append(
                "```text\n"
                + self._ascii_safe_text(
                    self._clip_text(s.get("prefix_text"), max_chars=max_chars)
                )
                + "\n```\n"
            )
            lines.append("### Training Target (prefix + FN append)\n")
            lines.append(
                "```text\n"
                + self._ascii_safe_text(
                    self._clip_text(s.get("train_text"), max_chars=max_chars)
                )
                + "\n```\n"
            )

            gt_payload = s.get("gt")
            if gt_payload is None:
                gt_payload = s.get("gt_objects")
            pred_payload = s.get("pred")
            if pred_payload is None:
                pred_payload = s.get("pred_objects")

            lines.append("### GT\n")
            lines.append("```json\n" + _j(gt_payload) + "\n```\n")
            lines.append("### Pred\n")
            lines.append("```json\n" + _j(pred_payload) + "\n```\n")
            if s.get("duplication") is not None:
                lines.append("### Duplication\n")
                lines.append("```json\n" + _j(s.get("duplication")) + "\n```\n")
            lines.append("### Match\n")
            lines.append("```json\n" + _j(s.get("match")) + "\n```\n")
            lines.append("### Stats\n")
            lines.append("```json\n" + _j(s.get("stats")) + "\n```\n")

        return "".join(lines)

    def _offload_settings(self) -> Tuple[bool, bool, bool]:
        cfg_raw = self._cfg("offload", {}) or {}
        if cfg_raw is None:
            cfg_raw = {}
        if not isinstance(cfg_raw, Mapping):
            raise ValueError("rollout_matching.offload must be a mapping")

        enabled = bool(cfg_raw.get("enabled", False))
        offload_model_raw = cfg_raw.get("offload_model", None)
        offload_optimizer_raw = cfg_raw.get("offload_optimizer", None)
        if enabled:
            offload_model = (
                True if offload_model_raw is None else bool(offload_model_raw)
            )
            offload_optimizer = (
                True
                if offload_optimizer_raw is None
                else bool(offload_optimizer_raw)
            )
        else:
            offload_model = (
                False if offload_model_raw is None else bool(offload_model_raw)
            )
            offload_optimizer = (
                False
                if offload_optimizer_raw is None
                else bool(offload_optimizer_raw)
            )
        return enabled, offload_model, offload_optimizer

    @contextmanager
    def _maybe_rollout_offload_context(
        self,
        *,
        rollout_backend: Optional[Literal["hf", "vllm"]] = None,
        force_enable: bool = False,
        force_offload_model: bool = False,
        force_offload_optimizer: bool = False,
        require_cuda_drain: bool = False,
    ):
        """Offload training state during colocate vLLM rollout generation.

        Fail-fast when offload is requested but not safe to apply.

        Args:
            rollout_backend: Explicit rollout backend override for gating.
            force_enable: Force-enable offload for this context.
            force_offload_model: Force model parameters/buffers CPU offload.
            force_offload_optimizer: Force optimizer state CPU offload.
            require_cuda_drain: Use stricter CUDA drain (sync + cache drain) at
                transition boundaries.
        """

        enabled, offload_model, offload_optimizer = self._offload_settings()
        if bool(force_enable):
            enabled = True
        if bool(force_offload_model):
            offload_model = True
        if bool(force_offload_optimizer):
            offload_optimizer = True

        # CoordExp policy: user-configurable offload is disabled. Offload is only
        # permitted in internal forced handoff windows (e.g. eval-time vLLM colocate).
        if not bool(force_enable):
            yield
            return

        if not enabled or (not offload_model and not offload_optimizer):
            yield
            return

        backend = (
            rollout_backend
            if rollout_backend is not None
            else self._effective_rollout_backend(context="train")
        )
        is_vllm_colocate = bool(
            backend == "vllm" and self._vllm_mode() == "colocate"
        )

        # Model offload is only safe/needed for colocate vLLM windows.
        if offload_model and not is_vllm_colocate:
            offload_model = False

        # Optimizer-only offload is also useful for HF rollout generation in rollout-correction
        # to avoid transient memory additive peaks (train state + decode cache).
        if not is_vllm_colocate and not offload_optimizer:
            yield
            return

        # Fail-fast on known-incompatible setups.
        if bool(getattr(self, "is_deepspeed_enabled", False)):
            raise RuntimeError(
                "rollout offload is not supported with DeepSpeed/ZeRO in this trainer. "
                "Mitigations: disable rollout_matching.offload, switch rollout_backend=hf, "
                "or disable DeepSpeed."
            )

        train_device = getattr(getattr(self, "accelerator", None), "device", None)
        if train_device is None:
            train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = getattr(
            getattr(self, "accelerator", None), "unwrap_model", lambda x: x
        )(self.model)
        opt = getattr(self, "optimizer", None)

        @torch.no_grad()
        def _offload_model_to_cpu(m) -> None:
            params = getattr(m, "parameters", None)
            buffers = getattr(m, "buffers", None)
            if params is None or buffers is None:
                return

            cpu = torch.device("cpu")
            for p in params():
                p.data = p.data.to(cpu, non_blocking=True)
            for b in buffers():
                b.data = b.data.to(cpu, non_blocking=True)

        @torch.no_grad()
        def _load_model_to_device(m) -> None:
            params = getattr(m, "parameters", None)
            buffers = getattr(m, "buffers", None)
            if params is None or buffers is None:
                return

            for p in params():
                p.data = p.data.to(train_device, non_blocking=True)
            for b in buffers():
                b.data = b.data.to(train_device, non_blocking=True)

        @torch.no_grad()
        def _offload_opt_to_cpu(o) -> None:
            if o is None or not getattr(o, "state", None):
                return
            for pg in o.param_groups:
                for p in pg.get("params", []):
                    st = o.state.get(p)
                    if not isinstance(st, dict):
                        continue
                    for k, v in list(st.items()):
                        if isinstance(v, torch.Tensor):
                            st[k] = v.to(torch.device("cpu"), non_blocking=True)

        @torch.no_grad()
        def _load_opt_to_device(o) -> None:
            if o is None or not getattr(o, "state", None):
                return
            for pg in o.param_groups:
                for p in pg.get("params", []):
                    st = o.state.get(p)
                    if not isinstance(st, dict):
                        continue
                    for k, v in list(st.items()):
                        if isinstance(v, torch.Tensor):
                            st[k] = v.to(train_device, non_blocking=True)

        try:
            if offload_model:
                _offload_model_to_cpu(model)
            if offload_optimizer:
                _offload_opt_to_cpu(opt)
            self._cuda_memory_drain(synchronize=bool(require_cuda_drain))
            yield
        finally:
            if offload_model:
                _load_model_to_device(model)
            if offload_optimizer:
                _load_opt_to_device(opt)
            self._cuda_memory_drain(synchronize=bool(require_cuda_drain))

    @staticmethod
    def _cuda_memory_drain(*, synchronize: bool = False) -> None:
        """Best-effort CUDA allocator drain at lifecycle transitions."""
        if not torch.cuda.is_available():
            return

        try:
            if bool(synchronize):
                torch.cuda.synchronize()
        except (AssertionError, RuntimeError):
            pass

        try:
            torch.cuda.empty_cache()
        except (AssertionError, RuntimeError):
            pass

        try:
            ipc_collect = getattr(torch.cuda, "ipc_collect", None)
            if callable(ipc_collect):
                ipc_collect()
        except (AssertionError, RuntimeError):
            pass

        try:
            import gc

            gc.collect()
        except RuntimeError:
            pass

    def _rollout_backend(self) -> Literal["hf", "vllm"]:
        return self._effective_rollout_backend(context="train")

    def _current_rollout_context(self) -> Literal["train", "eval"]:
        return current_rollout_context_from_owner(self)

    def _vllm_mode(self) -> Literal["colocate", "server"]:
        return vllm_mode_from_rollout_owner(self)


    def _vllm_sleep_mode_enabled(self) -> bool:
        return vllm_sleep_mode_enabled(self)


    def _vllm_reinit_each_eval(self) -> bool:
        return vllm_reinit_each_eval(self)

    def _vllm_sleep_level(self, *, default: int = 0) -> int:
        return vllm_sleep_level(self, default=default)

    def _validate_vllm_eval_lifecycle_preflight(self) -> None:
        validate_vllm_eval_lifecycle_preflight()

    @staticmethod
    def _vllm_raw_engine_or_raise(engine_wrapper: Any) -> Any:
        return vllm_raw_engine_or_raise(engine_wrapper)

    @classmethod
    def _wake_vllm_engine(cls, engine_wrapper: Any) -> None:
        wake_vllm_engine(engine_wrapper)

    @classmethod
    def _sleep_vllm_engine(cls, engine_wrapper: Any, *, level: int) -> None:
        sleep_vllm_engine(engine_wrapper, level=int(level))


    @staticmethod
    def _best_effort_fix_vllm_nccl_allocator_atexit_order() -> None:
        best_effort_fix_vllm_nccl_allocator_atexit_order()


    @staticmethod
    def _best_effort_patch_vllm_cumem_sleep_no_empty_cache() -> None:
        best_effort_patch_vllm_cumem_sleep_no_empty_cache()


    @staticmethod
    def _best_effort_cleanup_vllm_sleep_mode_pools() -> None:
        best_effort_cleanup_vllm_sleep_mode_pools()

    @contextmanager
    def _maybe_eval_vllm_colocate_window(
        self,
        *,
        rollout_backend: Literal["hf", "vllm"],
    ):
        with maybe_eval_vllm_colocate_window(
            owner=self,
            rollout_backend=rollout_backend,
        ):
            yield

    def _vllm_server_cfg(self) -> Mapping[str, Any]:
        return vllm_server_cfg(self)

    def _vllm_server_specs(self) -> List[Dict[str, Any]]:
        return vllm_server_specs(self)

    def _vllm_server_timeouts(self) -> Tuple[float, Optional[float]]:
        return vllm_server_timeouts(owner=self, logger=logger)

    def _vllm_server_world_sizes(self) -> List[int]:
        return vllm_server_world_sizes(owner=self, logger=logger)


    def _rollout_decode_batch_size_per_rank(
        self,
        *,
        rollout_backend: Optional[Literal["hf", "vllm"]] = None,
        rollout_context: Literal["train", "eval"] = "train",
    ) -> int:
        return rollout_decode_batch_size_per_rank(
            owner=self,
            rollout_backend=rollout_backend,
            rollout_context=rollout_context,
            logger=logger,
        )

    def _vllm_server_sync_cfg(self) -> str:
        return effective_vllm_server_sync_mode(self)

    @staticmethod
    def _normalize_rollout_seed_int32(seed: int) -> int:
        """Normalize a rollout seed into a non-zero signed int32 range.

        ms-swift rollout server code treats `RequestConfig.seed` as truthy/falsey
        (e.g. `if request_config.seed:`). A seed value of 0 is therefore
        semantically equivalent to "unset" and can silently disable seeding.

        To keep rollouts deterministic across backends and ms-swift versions, we
        canonicalize the seed into:
          [1, 2^31 - 1]
        """

        s = int(seed) & 0x7FFFFFFF
        return 1 if s == 0 else s

    def _derive_rollout_seed_base(self, *, global_step: int) -> int:
        """Deterministic seed base for rollouts.

        Contract: per-request seeds are derived deterministically from:
        - `training.seed` (HF TrainingArguments.seed)
        - `global_step` (optimizer-step index)
        - within-batch sample index
        """
        base = int(getattr(getattr(self, "args", None), "seed", 0) or 0)
        gs = int(global_step)
        # Keep in non-zero signed int32 range for compatibility with ms-swift rollouts.
        return self._normalize_rollout_seed_int32(int(base + gs * 1000003))

    def _decode_batch_size(
        self,
        *,
        context: Literal["train", "eval"] = "train",
    ) -> int:
        return rollout_decode_batch_size_from_owner(self, context=context)

    def _packing_enabled(self) -> bool:
        return bool(self._cfg("packing_enabled", False))

    def _packing_length(self) -> int:
        try:
            return int(self._cfg("packing_length", 0) or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError("packing_length must be an int") from exc

    def _assert_single_packed_forward(
        self, batch: Mapping[str, Any], *, where: str
    ) -> None:
        input_ids = batch.get("input_ids") if isinstance(batch, Mapping) else None
        if not isinstance(input_ids, torch.Tensor):
            return
        if input_ids.ndim != 2:
            raise ValueError(
                f"{where}: expected input_ids with shape [B, T], got {tuple(input_ids.shape)}"
            )
        bsz, seq_len = input_ids.shape
        if int(bsz) != 1:
            raise ValueError(
                f"{where}: packing must produce exactly one packed sequence per forward pass (batch_size=1), got batch_size={int(bsz)}"
            )
        max_len = 0
        try:
            max_len = int(self._packing_length() or 0)
        except (TypeError, ValueError):
            max_len = 0
        if int(max_len) > 0 and int(seq_len) > int(max_len):
            raise ValueError(
                f"{where}: packed seq_len={int(seq_len)} exceeds packing_length/global_max_length={int(max_len)}"
            )

    def _packing_buffer_cap(self) -> int:
        try:
            return int(self._cfg("packing_buffer", 0) or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError("packing_buffer must be an int") from exc

    def _packing_min_fill_ratio(self) -> float:
        try:
            v = float(self._cfg("packing_min_fill_ratio", 0.65))
        except (TypeError, ValueError) as exc:
            raise ValueError("packing_min_fill_ratio must be a float") from exc
        if not (0 < v <= 1):
            raise ValueError("packing_min_fill_ratio must be in (0, 1]")
        return float(v)

    def _packing_drop_last(self) -> bool:
        return bool(self._cfg("packing_drop_last", True))


    @staticmethod
    def _extract_encoded_len(encoded: Mapping[str, Any]) -> int:
        length = encoded.get("length")
        if isinstance(length, int) and length > 0:
            return int(length)
        input_ids = encoded.get("input_ids")
        if input_ids is not None and hasattr(input_ids, "__len__"):
            try:
                n = int(len(input_ids))
                if n > 0:
                    return n
            except (TypeError, ValueError):
                raise
        raise ValueError("encoded sample is missing a valid length/input_ids")

    @contextmanager
    def _template_state_context(
        self,
        *,
        packing: Optional[bool] = None,
        padding_free: Optional[bool] = None,
        mode: Optional[str] = None,
    ):
        template = self.template

        lock = getattr(self, "_template_toggle_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._template_toggle_lock = lock

        tls = getattr(self, "_template_toggle_tls", None)
        if tls is None:
            tls = threading.local()
            self._template_toggle_tls = tls

        depth = int(getattr(tls, "depth", 0) or 0)
        acquired = False
        if depth == 0:
            lock.acquire()
            acquired = True
            tls.stack = []

        tls.depth = depth + 1
        stack = getattr(tls, "stack", None)
        if stack is None:
            stack = []
            tls.stack = stack

        old_padding_free = getattr(template, "padding_free", False)
        old_packing = getattr(template, "packing", False)
        old_mode = getattr(template, "mode", None)
        stack.append((old_padding_free, old_packing, old_mode))

        try:
            if packing is not None:
                try:
                    template.packing = bool(packing)
                except (TypeError, ValueError):
                    raise
            if padding_free is not None:
                try:
                    template.padding_free = bool(padding_free)
                except (TypeError, ValueError):
                    raise
            if (
                mode is not None
                and old_mode is not None
                and hasattr(template, "set_mode")
            ):
                try:
                    if str(old_mode) != str(mode):
                        template.set_mode(str(mode))
                except (TypeError, ValueError):
                    raise

            yield
        finally:
            try:
                old_padding_free, old_packing, old_mode = stack.pop()
            except (TypeError, ValueError):
                old_padding_free, old_packing, old_mode = False, False, None

            if old_mode is not None and hasattr(template, "set_mode"):
                try:
                    template.set_mode(old_mode)
                except (TypeError, ValueError):
                    raise
            try:
                template.padding_free = old_padding_free
            except (TypeError, ValueError):
                raise
            try:
                template.packing = old_packing
            except (TypeError, ValueError):
                raise

            try:
                tls.depth = int(getattr(tls, "depth", 1) or 1) - 1
            except (TypeError, ValueError):
                tls.depth = 0

            if int(getattr(tls, "depth", 0) or 0) <= 0:
                tls.depth = 0
                try:
                    tls.stack = []
                except (TypeError, ValueError):
                    raise
                if acquired:
                    lock.release()

    @contextmanager
    def _template_packing_disabled(self):
        """Temporarily disable ms-swift template packing/padding-free flags."""
        with self._template_state_context(packing=False, padding_free=False):
            yield

    @contextmanager
    def _template_train_mode(self):
        """Temporarily force template mode to `train` for teacher-forced encoding.

        Some runners may keep the template in a non-training mode (e.g. `pt`), which
        would strip assistant responses from messages. Stage_2 needs the assistant span
        present in `input_ids` for masking/loss construction.
        """
        with self._template_state_context(mode="train"):
            yield

    @contextmanager
    def _template_packing_enabled(self):
        """Temporarily enable ms-swift template packing/padding-free flags."""
        with self._template_state_context(packing=True, padding_free=True):
            yield

    def _maybe_debug_dump_parse_failure(
        self,
        *,
        sample: Mapping[str, Any],
        response_text: str,
        prefix_text: str,
        dropped_invalid: int,
        dropped_ambiguous: int,
        truncated: bool,
        decode_mode: str,
    ) -> None:
        if not bool(self._cfg("debug_dump_parse_failures", False)):
            return
        max_dumps = int(self._cfg("debug_dump_max", 3))
        if max_dumps <= 0 or self._debug_dump_count >= max_dumps:
            return
        if dropped_invalid <= 0 and dropped_ambiguous <= 0 and not truncated:
            return

        self._debug_dump_count += 1
        images = (
            sample.get("images") if isinstance(sample.get("images"), list) else None
        )
        image = sample.get("image") if isinstance(sample.get("image"), str) else None
        tag = f"images={images!r}" if images else f"image={image!r}"

        def _clip(text: str, n: int = 600) -> str:
            t = text.replace("\n", "\\n")
            if len(t) <= n:
                return t
            return t[:n] + "...<truncated>"

        logger.warning(
            "rollout debug dump #%s (mode=%s %s): dropped_invalid=%s dropped_ambiguous=%s truncated=%s raw=%s prefix=%s",
            self._debug_dump_count,
            decode_mode,
            tag,
            dropped_invalid,
            dropped_ambiguous,
            truncated,
            _clip(response_text),
            _clip(prefix_text),
        )

    def _get_coord_token_ids(self) -> List[int]:
        if self._coord_token_ids is not None:
            return self._coord_token_ids
        tok = getattr(getattr(self, "template", None), "tokenizer", None)
        if tok is None:
            return []
        ids = get_coord_token_ids(tok, validate=True)
        self._coord_token_ids = [int(i) for i in ids]
        self._coord_id_to_bin = {int(tok_id): int(i) for i, tok_id in enumerate(ids)}
        return self._coord_token_ids

    def _coord_id_map(self) -> Dict[int, int]:
        if self._coord_id_to_bin is None:
            _ = self._get_coord_token_ids()
        return self._coord_id_to_bin or {}

    # ------------------------ rollout + batch prep ------------------------ #
    # ---- rollout backends -------------------------------------------------
    def _ensure_vllm_engine(self) -> Any:
        """Initialize a colocated vLLM engine (lazy) through the infer backend."""
        return ensure_vllm_engine(owner=self, logger=logger)

    def _sync_vllm_rollout_model_if_needed(self) -> None:
        raise RuntimeError(
            "Colocate vLLM sync has been removed from CoordExp. "
            "Use rollout_matching.vllm.mode=server with official adapter sync."
        )

    def _sync_vllm_lora_if_needed(self) -> None:
        raise RuntimeError(
            "Colocate adapter-only vLLM sync is not wired in CoordExp. "
            "Use rollout_matching.vllm.mode=server for official ms-swift "
            "adapter sync."
        )

    def _effective_vllm_server_sync_mode(self) -> str:
        return self._vllm_server_sync_cfg()

    def _ensure_vllm_server_client(self) -> Any:
        """Create an ms-swift vLLM server client (lazy).

        Important:
        - HTTP `/infer/` does NOT require the NCCL communicator.
        - The NCCL communicator is only required for in-memory weight sync.
        - Under a multi-process learner (`torchrun`, `world_size>1`), communicator init
          MUST be rank0-only.

        Thread safety:
        - Stage2-AB async actor-learner may call this from a background prefetch thread.
          Guard creation so we never build multiple clients / init communicator twice.
        """
        return ensure_vllm_server_client(owner=self, logger=logger)

    def _ensure_vllm_server_communicator_rank0(self, client: Any) -> None:
        """Initialize vLLM server NCCL communicator (rank0-only under DDP)."""
        ensure_vllm_server_communicator_rank0(owner=self, client=client)

    def _shutdown_vllm_server_client(
        self, *, close_communicator: bool = True, close_sessions: bool = True
    ) -> None:
        """Best-effort cleanup for vLLM server client resources.

        This is called during trainer shutdown to reduce teardown races between
        background rollout workers and process-exit cleanup.
        """
        shutdown_vllm_server_client(
            owner=self,
            logger=logger,
            close_communicator=bool(close_communicator),
            close_sessions=bool(close_sessions),
        )

    def _shutdown_vllm_colocate_engine(
        self, *, wake_before_release: bool = True
    ) -> None:
        """Best-effort cleanup for colocate vLLM engine resources."""
        shutdown_vllm_colocate_engine(
            owner=self,
            logger=logger,
            wake_before_release=bool(wake_before_release),
        )

    def _vllm_server_infer_guard(self):
        """Optional hook for staging safe vLLM server inference.

        Stage2-AB async actor-learner may override this to prevent HTTP `/infer/` calls
        from racing with rank0 weight sync.
        """
        return nullcontext()

    def _maybe_debug_dump_vllm_server_rollouts(
        self,
        *,
        global_step: int,
        seed_base: int,
        infer_requests: Sequence[Mapping[str, Any]],
        outputs: Sequence[Tuple[List[int], str, str, List[int]]],
        samples: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> None:
        """Optional raw rollout dump for diagnosing vLLM server-mode formatting.

        Controlled via:
          rollout_matching.vllm.server.debug_dump:
            enabled: true
            every_steps: 10           # defaults to args.logging_steps
            dump_first_step: false    # defaults to args.logging_first_step
            only_world_process_zero: true
            max_events: 3
            max_samples: 1
            max_chars: 4000           # <=0 disables clipping (full raw text)
            async_write: true         # non-blocking I/O (single-thread executor)
            max_pending_writes: 2     # bounds in-flight async tasks
            min_free_gb: 2.0          # skip dumps when disk is low
            out_dir: null             # defaults to <output_dir>/vllm_server_debug

        Notes:
        - In DDP, the default is rank0-only dumping to avoid I/O storms.
        - If only_world_process_zero=false, dumps go to per-rank subdirectories.
        - Payload is intentionally minimal for human review: GT text + rollout text.
        """

        scfg_raw = None
        try:
            scfg_raw = self._vllm_server_cfg()
        except Exception as exc:
            # Debug dumps are best-effort diagnostics; never crash training.
            self._dump_warn_once(
                "vllm_server_debug_dump_cfg_failed",
                "Failed to read vLLM server config for debug dump: %r",
                exc,
            )
            scfg_raw = None

        if scfg_raw is None:
            return

        debug_raw = (
            scfg_raw.get("debug_dump", {}) if isinstance(scfg_raw, Mapping) else {}
        )
        if not isinstance(debug_raw, Mapping) or not bool(
            debug_raw.get("enabled", False)
        ):
            return

        only_main = bool(debug_raw.get("only_world_process_zero", True))
        if only_main and not self._is_main_process():
            return

        max_events = int(debug_raw.get("max_events", 3) or 0)
        if max_events > 0 and int(self._vllm_server_debug_dump_count) >= int(
            max_events
        ):
            return

        gs = int(global_step)
        if (
            self._vllm_server_debug_last_step is not None
            and int(self._vllm_server_debug_last_step) == gs
        ):
            return

        every = debug_raw.get("every_steps", None)
        if every is None:
            every = int(getattr(self.args, "logging_steps", 1) or 1)
        every = max(1, int(every))

        dump_first = bool(
            debug_raw.get(
                "dump_first_step", bool(getattr(self.args, "logging_first_step", False))
            )
        )
        if gs == 0 and not dump_first:
            return
        if gs % every != 0:
            return

        out_dir = debug_raw.get("out_dir")
        if not isinstance(out_dir, str) or not out_dir.strip():
            out_dir = os.path.join(
                str(getattr(self.args, "output_dir", ".")), "vllm_server_debug"
            )

        rank = 0
        world_size = 1
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                rank = int(dist.get_rank())
                world_size = int(dist.get_world_size())
        except (TypeError, ValueError):
            rank = 0
            world_size = 1

        # DDP-safe output naming: if dumping from multiple ranks, isolate paths.
        if (not only_main) and int(world_size) > 1:
            out_dir = os.path.join(str(out_dir), f"rank_{int(rank)}")

        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError as exc:
            self._dump_warn_once(
                "vllm_server_debug_dump_makedirs_failed",
                "Failed to create vLLM server debug dump dir %s: %r",
                out_dir,
                exc,
            )
            return

        min_free_gb_raw = debug_raw.get("min_free_gb", 2.0)
        try:
            min_free_gb = float(min_free_gb_raw) if min_free_gb_raw is not None else 0.0
        except Exception:
            min_free_gb = 2.0
        min_free_gb = max(0.0, float(min_free_gb))
        if min_free_gb > 0:
            try:
                import shutil

                usage = shutil.disk_usage(out_dir)
                free_gb = float(usage.free) / float(1024**3)
                if free_gb < min_free_gb:
                    self._dump_warn_once(
                        "vllm_server_debug_dump_low_disk",
                        "Skipping vLLM server debug dump: free disk %.2f GB < min_free_gb=%.2f at %s.",
                        free_gb,
                        min_free_gb,
                        out_dir,
                    )
                    return
            except Exception as exc:
                self._dump_warn_once(
                    "vllm_server_debug_dump_disk_check_failed",
                    "Failed to check disk usage for vLLM server debug dump dir %s: %r",
                    out_dir,
                    exc,
                )

        async_write = bool(debug_raw.get("async_write", True))
        max_pending_raw = debug_raw.get("max_pending_writes", 2)
        try:
            max_pending_writes = (
                int(max_pending_raw) if max_pending_raw is not None else 2
            )
        except Exception:
            max_pending_writes = 2
        max_pending_writes = max(1, int(max_pending_writes))

        self._vllm_server_debug_last_step = int(gs)
        self._vllm_server_debug_dump_count += 1
        event = int(self._vllm_server_debug_dump_count)

        max_samples = int(debug_raw.get("max_samples", 1) or 1)
        max_chars_raw = debug_raw.get("max_chars", 4000)
        try:
            max_chars = int(max_chars_raw) if max_chars_raw is not None else 0
        except (TypeError, ValueError):
            max_chars = 4000

        def _content_to_text(content: Any) -> str:
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: List[str] = []
                for item in content:
                    if isinstance(item, Mapping):
                        # OpenAI-style multimodal content item.
                        if item.get("type") == "text" and item.get("text") is not None:
                            parts.append(str(item.get("text")))
                        elif item.get("text") is not None:
                            parts.append(str(item.get("text")))
                    elif item is not None:
                        parts.append(str(item))
                return "\n".join(parts)
            if content is None:
                return ""
            text = ""
            try:
                s = str(content)
            except Exception:
                s = ""
            return s

        def _extract_gt_text(sample_obj: Any) -> str:
            if not isinstance(sample_obj, Mapping):
                return ""
            messages = sample_obj.get("messages")
            if not isinstance(messages, list):
                return ""
            for m in reversed(messages):
                if not isinstance(m, Mapping):
                    continue
                if str(m.get("role", "")).lower() != "assistant":
                    continue
                return _content_to_text(m.get("content"))
            return ""

        sample_list: List[Mapping[str, Any]] = []
        if isinstance(samples, Sequence):
            for s in samples:
                if isinstance(s, Mapping):
                    sample_list.append(s)
                else:
                    sample_list.append({})

        samples_dump: List[Dict[str, Any]] = []
        for i, (_, out) in enumerate(zip(infer_requests, outputs)):
            if i >= max_samples:
                break

            resp_text = ""
            try:
                if isinstance(out, (list, tuple)) and len(out) > 1:
                    resp_text = str(out[1])
            except Exception:
                resp_text = ""
            sample_obj = sample_list[i] if i < len(sample_list) else {}

            gt_text_raw = _extract_gt_text(sample_obj)
            gt_text = self._ascii_safe_text(
                self._clip_text(gt_text_raw, max_chars=max_chars)
            )
            rollout_text = self._ascii_safe_text(
                self._clip_text(resp_text, max_chars=max_chars)
            )

            samples_dump.append(
                {
                    "gt_text": gt_text,
                    "rollout_text": rollout_text,
                }
            )

        payload = {
            "global_step": int(gs),
            "samples": samples_dump,
        }

        path = os.path.join(
            out_dir,
            f"step_{int(gs):06d}_event_{int(event):03d}.json",
        )

        def _write() -> None:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=True, indent=2)
                logger.warning(
                    "vLLM server debug dump wrote %s (samples=%s)",
                    path,
                    len(samples_dump),
                )
            except Exception as exc:
                logger.warning("Failed to write vLLM server debug dump %s: %r", path, exc)

        self._submit_dump_write(
            kind="vllm_server_debug_dump",
            async_write=async_write,
            max_pending_writes=max_pending_writes,
            fn=_write,
        )

    def _sync_vllm_server_rollout_model_if_needed(self) -> None:
        """Sync weights/adapters to rollout server for vLLM server mode.

        DDP safety (when torch.distributed is initialized):
        - Rank0-only communicator init + weight push.
        - Strict ordering: barrier -> rank0 sync -> barrier.
        - All ranks must take the same control-flow to avoid deadlocks.

        NOTE: This sync is intended for the synchronous rollout path.
        Async actor-learner should coordinate server sync at safe boundaries and
        avoid invoking DDP collectives from background prefetch threads.
        """
        sync_vllm_server_rollout_model_if_needed(owner=self)

    def _sync_vllm_server_adapter(self, client: Any) -> None:
        from src.infer.backend_vllm_server import sync_vllm_server_adapter

        sync_vllm_server_adapter(
            owner=self,
            client=client,
            logger=logger,
        )

    @torch.no_grad()
    def _rollout_many(
        self,
        samples: Sequence[Mapping[str, Any]],
        *,
        prompt_variant_override: Optional[str] = None,
        rollout_backend: Optional[Literal["hf", "vllm"]] = None,
        decode_override: Optional[Mapping[str, Any]] = None,
        request_index_offset: int = 0,
    ) -> List[Tuple[List[int], str, str, List[int]]]:
        return rollout_many(
            owner=self,
            samples=samples,
            prompt_variant_override=prompt_variant_override,
            rollout_backend=rollout_backend,
            decode_override=decode_override,
            request_index_offset=int(request_index_offset),
        )

    def _append_post_rollout_segments(
        self, segments: Sequence[Tuple[Dict[str, Any], Dict[str, Any], int]]
    ) -> None:
        """Append newly produced post-rollout segments to the rank-local buffer.

        Safety: fail-fast if any single segment exceeds packing_length, at insertion time.
        """
        packing_length = int(self._packing_length())
        if packing_length <= 0:
            raise ValueError("packing is enabled but packing_length is invalid")

        seg_list = segments if isinstance(segments, list) else list(segments)

        for _, _, seg_len in seg_list:
            sl = int(seg_len)
            if sl > packing_length:
                raise ValueError(
                    f"post-rollout packing cannot fit a single segment: encoded_len={sl} > packing_length={packing_length}. "
                    "Mitigations: increase global_max_length/template.max_length, reduce max_new_tokens, or disable packing."
                )

        cap = int(self._packing_buffer_cap())
        if cap > 0:
            new_size = len(self._post_rollout_segments) + len(seg_list)
            if new_size > cap:
                raise ValueError(
                    "post-rollout packing buffer overflow: "
                    f"buffer_size={new_size} > packing_buffer={cap}. "
                    "Mitigations: reduce per_device_train_batch_size, increase training.packing_buffer, "
                    "or enable multi-pack-per-step in a future change."
                )

        self._post_rollout_segments.extend(seg_list)

    @staticmethod
    def _select_post_rollout_segment_indices(
        encoded_lens: Sequence[int],
        packing_length: int,
        *,
        min_fill_ratio: Optional[float] = None,
    ) -> List[int]:
        """Select segment indices for one packed forward pass.

        Input `encoded_lens` is in insertion order (index 0 is oldest). Output indices
        are in insertion order, MUST include the oldest segment, and total length MUST
        be <= packing_length.

        Selection always uses pool-aware packing:
        - compute FIFO-greedy baseline,
        - compute a stage-1-like constant-volume pool plan (oldest pinned),
        - pick the plan with better pool-level packing score.

        `min_fill_ratio` acts as a soft fill target used during pool-level scoring.
        """
        packing_length = int(packing_length)
        if packing_length <= 0:
            raise ValueError("packing_length must be positive")
        if not encoded_lens:
            return []

        try:
            import binpacking
        except ImportError as exc:
            raise ImportError(
                "binpacking is required for stage-2 post-rollout packing selection; "
                "install `binpacking` or disable `training.packing`."
            ) from exc

        lens = [int(x) for x in encoded_lens]
        oldest_len = int(lens[0])
        if oldest_len > packing_length:
            raise ValueError(
                f"post-rollout packing cannot fit a single segment: encoded_len={oldest_len} > packing_length={packing_length}. "
                "Mitigations: increase global_max_length/template.max_length, reduce max_new_tokens, or disable packing."
            )
        if oldest_len <= 0:
            raise ValueError("oldest post-rollout segment has non-positive encoded_len")

        for sl in lens:
            if int(sl) > packing_length:
                raise ValueError(
                    f"post-rollout packing buffer contains an oversized segment: encoded_len={int(sl)} > packing_length={packing_length}."
                )

        # 1) FIFO-greedy baseline (must include oldest; stable insertion-order scan).
        baseline: List[int] = [0]
        used = int(oldest_len)
        for i in range(1, len(lens)):
            sl = int(lens[i])
            if sl <= 0:
                continue
            if used + sl <= packing_length:
                baseline.append(int(i))
                used += sl
        baseline = sorted(int(i) for i in baseline)

        # Pool-aware mode: allow selecting a shorter current pack when it improves the
        # remaining pool packing (reduces underfilled remainder packs).
        target_len = 0
        if min_fill_ratio is not None:
            try:
                mfr = float(min_fill_ratio)
            except (TypeError, ValueError):
                mfr = 0.0
            if math.isfinite(mfr) and mfr > 0.0:
                mfr = min(1.0, max(0.0, mfr))
                target_len = int(math.ceil(float(mfr) * float(packing_length)))

        def _bin_total(idxs: Sequence[int]) -> int:
            return int(sum(int(lens[i]) for i in idxs))

        def _bins_from_indices(indices: Sequence[int]) -> List[List[int]]:
            if not indices:
                return []
            items: List[Tuple[int, int]] = []
            for i in indices:
                ii = int(i)
                sl = int(lens[ii])
                if sl <= 0:
                    continue
                if sl > packing_length:
                    continue
                items.append((ii, sl))
            if not items:
                return []

            raw_bins = binpacking.to_constant_volume(
                items,
                packing_length,
                weight_pos=1,
            )

            out: List[List[int]] = []
            for b in raw_bins:
                idxs = sorted(int(idx) for idx, _ in b)
                if idxs:
                    out.append(idxs)
            return out

        def _count_underfilled(bins: Sequence[Sequence[int]]) -> int:
            if target_len <= 0:
                return 0
            c = 0
            for b in bins:
                if _bin_total(b) < int(target_len):
                    c += 1
            return int(c)

        def _min_fill(bins: Sequence[Sequence[int]]) -> float:
            if not bins:
                return 1.0
            mf = 1.0
            for b in bins:
                tot = float(_bin_total(b))
                mf = min(mf, tot / float(packing_length))
            return float(mf)

        def _rebalance_bins(
            bins: List[List[int]],
            *,
            frozen: Optional[set[int]] = None,
        ) -> List[List[int]]:
            if target_len <= 0:
                return bins
            if len(bins) < 2:
                return bins

            frozen_set: set[int] = set(frozen or set())

            for _pass in range(8):
                totals = [_bin_total(b) for b in bins]
                moved_any = False

                order = sorted(
                    range(len(bins)),
                    key=lambda bi: (totals[bi], bins[bi][0] if bins[bi] else 10**9, bi),
                )

                for recv_i in order:
                    recv = bins[recv_i]
                    if not recv:
                        continue
                    recv_total = int(totals[recv_i])
                    if recv_total >= int(target_len):
                        continue

                    recv_cap = int(packing_length - recv_total)
                    if recv_cap <= 0:
                        continue

                    deficit = int(target_len - recv_total)

                    best: Optional[Tuple[int, int, int, int, int]] = None
                    # (category, secondary, item_len, item_idx, donor_i) smaller is better

                    for donor_i, donor in enumerate(bins):
                        if donor_i == recv_i:
                            continue
                        if len(donor) <= 1:
                            continue
                        donor_total = int(totals[donor_i])
                        if donor_total <= int(target_len):
                            continue

                        for item_idx in donor:
                            ii = int(item_idx)
                            if ii in frozen_set:
                                continue
                            item_len = int(lens[ii])
                            if item_len <= 0:
                                continue
                            if item_len > recv_cap:
                                continue
                            if int(donor_total - item_len) < int(target_len):
                                continue

                            if item_len <= deficit:
                                key = (0, -item_len, item_len, ii, donor_i)
                            else:
                                key = (1, item_len, item_len, ii, donor_i)

                            if best is None or key < best:
                                best = key

                    if best is None:
                        continue

                    _cat, _sec, item_len, item_idx, donor_i = best

                    donor = bins[int(donor_i)]
                    if int(item_idx) not in donor:
                        continue
                    if len(donor) <= 1:
                        continue

                    donor_total = int(_bin_total(donor))
                    recv_total = int(_bin_total(recv))
                    if int(donor_total - item_len) < int(target_len):
                        continue
                    if int(recv_total + item_len) > int(packing_length):
                        continue

                    donor.remove(int(item_idx))
                    recv.append(int(item_idx))
                    donor.sort()
                    recv.sort()

                    totals[int(donor_i)] = int(donor_total - item_len)
                    totals[int(recv_i)] = int(recv_total + item_len)
                    moved_any = True

                if not moved_any:
                    break

            return bins

        def _score(
            *,
            selection: List[int],
            remaining_bins: List[List[int]],
        ) -> Tuple[int, int, int, float, Tuple[int, ...]]:
            tot = int(sum(int(lens[i]) for i in selection))
            fill = float(tot) / float(packing_length) if packing_length > 0 else 0.0
            underfilled = 0
            if target_len > 0 and tot < int(target_len):
                underfilled += 1
            underfilled += int(_count_underfilled(remaining_bins))

            n_bins = 1 + int(len(remaining_bins))
            min_fill = min(
                float(fill),
                float(_min_fill(remaining_bins)) if remaining_bins else float(fill),
            )
            return (
                int(n_bins),
                int(underfilled),
                -int(tot),
                -float(min_fill),
                tuple(int(i) for i in selection),
            )

        def _best_fill_oldest_pinned_selection() -> List[int]:
            residual_cap = int(packing_length - oldest_len)
            if residual_cap <= 0:
                return [0]

            # exact best-fill subset for the current pack, with stable
            # lexicographic tie-breaks over insertion-order indices.
            dp: Dict[int, Tuple[int, ...]] = {0: tuple()}
            for i in range(1, len(lens)):
                sl = int(lens[i])
                if sl <= 0 or sl > residual_cap:
                    continue

                updates: Dict[int, Tuple[int, ...]] = {}
                for used, idxs in dp.items():
                    new_used = int(used + sl)
                    if new_used > residual_cap:
                        continue

                    new_idxs = tuple(list(idxs) + [int(i)])
                    incumbent = dp.get(new_used)
                    pending = updates.get(new_used)
                    best_existing = incumbent if incumbent is not None else pending
                    if best_existing is None or new_idxs < best_existing:
                        updates[new_used] = new_idxs

                for used, idxs in updates.items():
                    incumbent = dp.get(int(used))
                    if incumbent is None or idxs < incumbent:
                        dp[int(used)] = idxs

            best_used = max(dp.keys())
            return [0, *list(dp[int(best_used)])]

        baseline_set = set(baseline)
        baseline_remaining_idx = [i for i in range(len(lens)) if i not in baseline_set]
        baseline_remaining_bins = _bins_from_indices(baseline_remaining_idx)
        baseline_remaining_bins = _rebalance_bins(baseline_remaining_bins)
        baseline_score = _score(
            selection=baseline,
            remaining_bins=baseline_remaining_bins,
        )

        all_idx = list(range(len(lens)))
        bins_full = _bins_from_indices(all_idx)
        if not bins_full:
            return baseline
        bins_full = _rebalance_bins(bins_full, frozen={0})

        oldest_bin: Optional[List[int]] = None
        rest_bins: List[List[int]] = []
        for b in bins_full:
            if 0 in b and oldest_bin is None:
                oldest_bin = list(b)
            else:
                rest_bins.append(list(b))
        if oldest_bin is None:
            return baseline

        oldest_bin = sorted(int(i) for i in oldest_bin)
        if not oldest_bin or oldest_bin[0] != 0:
            return baseline

        smart_score = _score(selection=oldest_bin, remaining_bins=rest_bins)

        best_fill = _best_fill_oldest_pinned_selection()
        best_fill_set = set(best_fill)
        best_fill_remaining_idx = [
            i for i in range(len(lens)) if i not in best_fill_set
        ]
        best_fill_remaining_bins = _bins_from_indices(best_fill_remaining_idx)
        best_fill_remaining_bins = _rebalance_bins(best_fill_remaining_bins)
        best_fill_score = _score(
            selection=best_fill,
            remaining_bins=best_fill_remaining_bins,
        )

        candidates = (
            (baseline_score, baseline),
            (smart_score, oldest_bin),
            (best_fill_score, best_fill),
        )
        return list(min(candidates, key=lambda item: item[0])[1])

    def _pop_post_rollout_pack(
        self,
    ) -> Tuple[List[Tuple[Dict[str, Any], Dict[str, Any], int]], Dict[str, float]]:
        """Select and remove a subset of buffered segments for one packed forward pass (carry-only).

        Returns (selected_segments, packing_metrics). Packing metrics are emitted into the main
        training log line (merged with loss) to avoid per-micro-batch log spam.
        """
        packing_length = int(self._packing_length())
        if packing_length <= 0:
            raise ValueError("packing is enabled but packing_length is invalid")
        if not self._post_rollout_segments:
            raise ValueError(
                "packing is enabled but no post-rollout segments are available"
            )

        encoded_lens = [int(seg_len) for _, _, seg_len in self._post_rollout_segments]
        selected_idx = self._select_post_rollout_segment_indices(
            encoded_lens,
            packing_length,
            min_fill_ratio=self._packing_min_fill_ratio(),
        )
        if not selected_idx:
            raise AssertionError("post-rollout packing selected an empty segment set")
        total_len = int(sum(encoded_lens[i] for i in selected_idx))

        selected = [self._post_rollout_segments[i] for i in selected_idx]
        for i in reversed(selected_idx):
            self._post_rollout_segments.pop(i)

        fill = float(total_len) / float(packing_length) if packing_length > 0 else 0.0
        target = float(self._packing_min_fill_ratio())

        # Expose last-pack stats for adaptive raw batching.
        try:
            self._rm_last_pack_fill = float(fill)
            self._rm_last_pack_segments = int(len(selected))
            self._rm_last_pack_buffer_after = int(len(self._post_rollout_segments))
        except (TypeError, ValueError):
            raise

        if fill < target:
            logger.warning(
                "post-rollout packing underfilled: fill=%.3f target=%.3f segments=%s buffer=%s",
                fill,
                target,
                len(selected),
                len(self._post_rollout_segments),
            )

        pack_metrics: Dict[str, float] = {
            "packing/post_rollout_fill": float(fill),
            "packing/post_rollout_selected_total_len": float(total_len),
            "packing/post_rollout_segments": float(len(selected)),
            "packing/post_rollout_buffer": float(len(self._post_rollout_segments)),
        }

        # Update a running average segment length estimate for adaptive raw batching.
        try:
            seg_count = int(len(selected))
            if seg_count > 0:
                avg = float(total_len) / float(seg_count)
                prev = float(getattr(self, "_rm_avg_segment_len", 0.0) or 0.0)

                # If we're underfilled *and* the buffer emptied, we were supply-limited.
                # Update aggressively downward so the next raw batch is larger.
                supply_limited = bool(
                    fill < target and len(self._post_rollout_segments) == 0
                )

                alpha = 0.2
                if supply_limited:
                    alpha = 0.5

                ema = (
                    float(avg)
                    if prev <= 0
                    else float((1.0 - alpha) * prev + alpha * avg)
                )
                if supply_limited:
                    ema = min(float(ema), float(avg))

                self._rm_avg_segment_len = float(ema)
                pack_metrics["packing/avg_segment_len_last"] = float(avg)
                pack_metrics["packing/avg_segment_len_ema"] = float(ema)
        except (TypeError, ValueError):
            raise

        return selected, pack_metrics

    def _reduce_train_rollout_log_payload_global(
        self, payload: Mapping[str, Any]
    ) -> Dict[str, float]:
        reduced: Dict[str, float] = {}
        for k, v in payload.items():
            try:
                reduced[str(k)] = float(v)
            except (TypeError, ValueError):
                continue

        trunc_num_key = "rollout/_parse_truncated_num"
        trunc_den_key = "rollout/_parse_truncated_den"
        sample_total_key = "train/samples_total"

        reduced.pop("rollout/parse_truncated_rate", None)
        has_parse_inputs = any(
            key in reduced
            for key in {
                trunc_num_key,
                trunc_den_key,
                "rollout/parse_truncated",
                sample_total_key,
            }
        ) and (
            trunc_num_key in reduced
            or trunc_den_key in reduced
            or "rollout/parse_truncated" in reduced
        )
        if has_parse_inputs:
            reduced.setdefault(
                trunc_num_key,
                float(reduced.get("rollout/parse_truncated", 0.0)),
            )
            reduced.setdefault(
                trunc_den_key,
                float(reduced.get(sample_total_key, 0.0)),
            )

        sample_total_local = float(reduced.get(sample_total_key, 0.0))
        if (
            "rollout/matched_maskiou_mean" in reduced
            and "rollout/matched_maskiou_count" in reduced
        ):
            reduced["rollout/_matched_maskiou_sum"] = float(
                float(reduced.get("rollout/matched_maskiou_mean", 0.0))
                * float(reduced.get("rollout/matched_maskiou_count", 0.0))
            )
        if sample_total_local > 0.0:
            if "rollout/sample_valid_pred_rate" in reduced:
                reduced["rollout/_sample_valid_pred_num"] = float(
                    float(reduced.get("rollout/sample_valid_pred_rate", 0.0))
                    * sample_total_local
                )
            if "rollout/sample_any_match_rate" in reduced:
                reduced["rollout/_sample_any_match_num"] = float(
                    float(reduced.get("rollout/sample_any_match_rate", 0.0))
                    * sample_total_local
                )
        if (
            "rollout/desc_exact_acc_on_matched" in reduced
            and "rollout/desc_pairs_total" in reduced
        ):
            reduced["rollout/_desc_exact_ok"] = float(
                float(reduced.get("rollout/desc_exact_acc_on_matched", 0.0))
                * float(reduced.get("rollout/desc_pairs_total", 0.0))
            )
        if (
            "rollout/desc_sem_acc_on_matched" in reduced
            and "rollout/desc_pairs_total" in reduced
        ):
            reduced["rollout/_desc_sem_ok"] = float(
                float(reduced.get("rollout/desc_sem_acc_on_matched", 0.0))
                * float(reduced.get("rollout/desc_pairs_total", 0.0))
            )

        # Loss scalars are mean-like per-segment metrics. Reduce them with sample weights
        # so ranks with different numbers of packed segments (or zero) don't skew the
        # global average under DDP.
        #
        # Contract: Any scalar under `loss/*` is mean-like and should be averaged across
        # ranks using `train/samples_total` as a weight (not an unweighted rank mean).
        loss_weight = float(sample_total_local)
        loss_total_keys: Dict[str, str] = {}
        loss_mean_keys = [
            str(k)
            for k in list(reduced.keys())
            if str(k).startswith("loss/")
            and not str(k).endswith("_total")
            and not str(k).endswith("_sum")
            and not str(k).endswith("_count")
            and not str(k).endswith("_num")
            and not str(k).endswith("_den")
        ]
        for key in loss_mean_keys:
            total_key = f"{key}_total"
            loss_total_keys[total_key] = key
            reduced[total_key] = float(float(reduced.get(key, 0.0)) * loss_weight)
            reduced.pop(key, None)

        gradmon_weight_key = "gradmon/_log_weight_total"

        reduced = reduce_metric_payload_global(
            self,
            reduced,
            resolver=resolve_rollout_log_metric_spec,
            error_prefix="rollout metric",
        )

        sample_total = float(reduced.get(sample_total_key, 0.0))

        for key_total, key_out in loss_total_keys.items():
            if key_total in reduced:
                reduced[key_out] = (
                    float(reduced.get(key_total, 0.0) / sample_total)
                    if sample_total > 0.0
                    else 0.0
                )

        if has_parse_inputs:
            trunc_num = float(reduced.get(trunc_num_key, 0.0))
            trunc_den = float(reduced.get(trunc_den_key, sample_total))
            reduced["rollout/parse_truncated_rate"] = (
                float(trunc_num / trunc_den) if trunc_den > 0.0 else 0.0
            )

        new_tok_total = float(reduced.get("rollout/gen_new_tokens_total", 0.0))
        if "rollout/gen_new_tokens_mean" in reduced:
            reduced["rollout/gen_new_tokens_mean"] = (
                float(new_tok_total / sample_total) if sample_total > 0.0 else 0.0
            )

        rollout_gen_s = float(reduced.get("time/rollout_generate_s", 0.0))
        if "rollout/gen_tokens_per_s" in reduced:
            reduced["rollout/gen_tokens_per_s"] = (
                float(new_tok_total / rollout_gen_s) if rollout_gen_s > 0.0 else 0.0
            )

        pred_total = float(reduced.get("rollout/valid_pred_objects_total", 0.0))
        gt_total = float(reduced.get("rollout/gt_objects_total", 0.0))
        matched_total = float(reduced.get("rollout/matched_for_supervision", 0.0))

        if any(str(k).startswith("rollout/") for k in reduced.keys()):
            precision = (matched_total / pred_total) if pred_total > 0.0 else 0.0
            recall = (matched_total / gt_total) if gt_total > 0.0 else 0.0
            f1 = (
                (2.0 * precision * recall / (precision + recall))
                if (precision + recall) > 0.0
                else 0.0
            )
            reduced["rollout/precision"] = float(precision)
            reduced["rollout/recall"] = float(recall)
            reduced["rollout/f1"] = float(f1)

        if "rollout/excluded_rate" in reduced:
            excluded_total = float(
                reduced.get("rollout/excluded_from_supervision", 0.0)
            )
            denom = float(matched_total + excluded_total)
            reduced["rollout/excluded_rate"] = (
                float(excluded_total / denom) if denom > 0.0 else 0.0
            )

        if "rollout/prefix_coord_targets_per_matched" in reduced:
            prefix_total = float(reduced.get("rollout/prefix_coord_targets_total", 0.0))
            reduced["rollout/prefix_coord_targets_per_matched"] = (
                float(prefix_total / matched_total) if matched_total > 0.0 else 0.0
            )

        if "rollout/gating_rejection_rate" in reduced:
            top_k = int(self._cfg("candidate_top_k", 10))
            denom = float(pred_total * float(max(1, top_k)))
            reduced["rollout/gating_rejection_rate"] = (
                float(reduced.get("rollout/gating_rejections", 0.0) / denom)
                if denom > 0.0
                else 0.0
            )

        if sample_total > 0.0:
            for key_total, key_out in (
                ("rollout/gt_objects_total", "rollout/gt_per_sample"),
                ("rollout/valid_pred_objects_total", "rollout/pred_per_sample"),
                ("rollout/fp_total", "rollout/fp_per_sample"),
                ("rollout/fn_total", "rollout/fn_per_sample"),
            ):
                if key_out in reduced or key_total in reduced:
                    reduced[key_out] = float(reduced.get(key_total, 0.0) / sample_total)

        if (
            "rollout/parse_obj_valid_frac" in reduced
            or "rollout/parse_obj_drop_frac" in reduced
            or "rollout/parse_obj_total" in reduced
        ):
            dropped_invalid_total = float(
                reduced.get("rollout/parse_dropped_invalid", 0.0)
            )
            dropped_amb_total = float(
                reduced.get("rollout/parse_dropped_ambiguous", 0.0)
            )
            obj_total = pred_total + dropped_invalid_total + dropped_amb_total
            reduced["rollout/parse_obj_total"] = float(obj_total)
            reduced["rollout/parse_obj_valid_frac"] = (
                float(pred_total / obj_total) if obj_total > 0.0 else 0.0
            )
            reduced["rollout/parse_obj_drop_frac"] = (
                float((dropped_invalid_total + dropped_amb_total) / obj_total)
                if obj_total > 0.0
                else 0.0
            )

        if (
            "rollout/matched_maskiou_mean" in reduced
            or "rollout/_matched_maskiou_sum" in reduced
            or "rollout/matched_maskiou_count" in reduced
        ):
            iou_sum = float(reduced.get("rollout/_matched_maskiou_sum", 0.0))
            iou_cnt = float(reduced.get("rollout/matched_maskiou_count", 0.0))
            reduced["rollout/matched_maskiou_mean"] = (
                float(iou_sum / iou_cnt) if iou_cnt > 0.0 else 0.0
            )

        if "rollout/sample_valid_pred_rate" in reduced:
            valid_pred_num = float(reduced.get("rollout/_sample_valid_pred_num", 0.0))
            reduced["rollout/sample_valid_pred_rate"] = (
                float(valid_pred_num / sample_total) if sample_total > 0.0 else 0.0
            )
        if "rollout/sample_any_match_rate" in reduced:
            any_match_num = float(reduced.get("rollout/_sample_any_match_num", 0.0))
            reduced["rollout/sample_any_match_rate"] = (
                float(any_match_num / sample_total) if sample_total > 0.0 else 0.0
            )

        if "rollout/desc_exact_acc_on_matched" in reduced:
            pairs_total = float(reduced.get("rollout/desc_pairs_total", 0.0))
            exact_ok = float(reduced.get("rollout/_desc_exact_ok", 0.0))
            reduced["rollout/desc_exact_acc_on_matched"] = (
                float(exact_ok / pairs_total) if pairs_total > 0.0 else 1.0
            )
        if "rollout/desc_sem_acc_on_matched" in reduced:
            pairs_total = float(reduced.get("rollout/desc_pairs_total", 0.0))
            sem_ok = float(reduced.get("rollout/_desc_sem_ok", 0.0))
            reduced["rollout/desc_sem_acc_on_matched"] = (
                float(sem_ok / pairs_total) if pairs_total > 0.0 else 1.0
            )
        if "rollout/desc_sem_sim_mean" in reduced:
            sim_sum = float(reduced.get("rollout/desc_sem_sim_sum", 0.0))
            sim_cnt = float(reduced.get("rollout/desc_sem_sim_count", 0.0))
            reduced["rollout/desc_sem_sim_mean"] = (
                float(sim_sum / sim_cnt) if sim_cnt > 0.0 else 0.0
            )

        for key in (
            trunc_num_key,
            trunc_den_key,
            *sorted(loss_total_keys.keys()),
            gradmon_weight_key,
            "rollout/_matched_maskiou_sum",
            "rollout/_sample_valid_pred_num",
            "rollout/_sample_any_match_num",
            "rollout/_desc_exact_ok",
            "rollout/_desc_sem_ok",
        ):
            reduced.pop(key, None)
        return reduced

    def _ddp_assert_all_ranks_true_or_raise(
        self,
        *,
        where: str,
        local_true: bool,
        global_step: int,
    ) -> None:
        ddp_assert_all_ranks_true_or_raise(
            self,
            where=where,
            local_true=local_true,
            global_step=global_step,
        )

    def log(self, logs: Dict[str, float]) -> None:
        """Merge buffered rollout-matching metrics into the main train log record.

        HF/Swift logs `loss` after the optimizer step (global_step already incremented).
        Our rollout metrics are computed inside `compute_loss` (before the increment),
        so we buffer them keyed by `global_step + 1` and merge here.

        This keeps one scalar per step per tag in TensorBoard (clean plots) and reduces
        `logging.jsonl` fragmentation.
        """

        try:
            if (
                isinstance(logs, dict)
                and "loss" in logs
                and not any(str(k).startswith("eval_") for k in logs.keys())
            ):
                step = int(getattr(getattr(self, "state", None), "global_step", 0) or 0)
                pending = self._rm_pending_train_logs.get(step)

                # IMPORTANT (DDP contract): this log() override performs distributed
                # collectives to merge buffered per-step metrics.
                #
                # It must be invoked on *every* rank when torch.distributed is
                # initialized. Do not make Trainer.log() rank-0-only, and do not gate
                # this path on is_main_process()/rank==0, otherwise rank 0 will enter
                # all_reduce() while other ranks skip it and the job will deadlock.
                self._ddp_assert_all_ranks_true_or_raise(
                    where="rollout-matching train log",
                    local_true=pending is not None,
                    global_step=step,
                )
                pending = self._rm_pending_train_logs.pop(step, None)
                if pending is not None:
                    payload = self._build_train_rollout_log_payload(pending)
                    payload = self._reduce_train_rollout_log_payload_global(payload)
                    sample_step = int(
                        round(float(payload.get("train/samples_total", 0.0)))
                    )
                    try:
                        self._rm_train_samples_seen = (
                            int(getattr(self, "_rm_train_samples_seen", 0) or 0)
                            + sample_step
                        )
                    except (TypeError, ValueError):
                        self._rm_train_samples_seen = sample_step

                    logs.update(payload)
                    logs["train/samples_seen"] = float(
                        getattr(self, "_rm_train_samples_seen", 0) or 0
                    )
                logs.update(
                    self._reduce_stage_wallclock_metrics_global(
                        self._stage_wallclock_metrics_local()
                    )
                )
        except (TypeError, ValueError):
            raise

        return super().log(logs)

    def _build_rollout_metrics_from_meta(
        self, meta: List[Mapping[str, Any]]
    ) -> Dict[str, float]:
        """Compute step-level rollout metrics from slim meta dicts."""

        n_samples = float(len(meta))

        rollout_active = any(
            int(m.get("rollout_len", 0)) > 0
            or str(m.get("decode_mode", "none") or "none").strip().lower()
            != "none"
            for m in meta
        )
        if not rollout_active:
            return {}

        gt_total = float(sum(int(m.get("gt_objects", 0)) for m in meta))
        matched_total = float(
            sum(int(m.get("matched_for_supervision", 0)) for m in meta)
        )
        pred_total = float(sum(int(m.get("valid_pred_objects", 0)) for m in meta))
        excluded_total = float(
            sum(int(m.get("excluded_from_supervision", 0)) for m in meta)
        )

        # Sample-level rates (helps detect systematic parse failures).
        n_samples_valid_pred = float(
            sum(1 for m in meta if int(m.get("valid_pred_objects", 0)) > 0)
        )
        n_samples_any_match = float(
            sum(1 for m in meta if int(m.get("matched_for_supervision", 0)) > 0)
        )
        sample_valid_pred_rate = (
            (n_samples_valid_pred / n_samples) if n_samples > 0 else 0.0
        )
        sample_any_match_rate = (
            (n_samples_any_match / n_samples) if n_samples > 0 else 0.0
        )

        fp_total = max(0.0, pred_total - matched_total)
        fn_total = max(0.0, gt_total - matched_total)
        precision = (matched_total / pred_total) if pred_total > 0 else 0.0
        recall = (matched_total / gt_total) if gt_total > 0 else 0.0
        f1 = (
            (2.0 * precision * recall / (precision + recall))
            if (precision + recall) > 0.0
            else 0.0
        )

        dropped_invalid_total = float(
            sum(int(m.get("parse_dropped_invalid", 0)) for m in meta)
        )
        dropped_ambiguous_total = float(
            sum(int(m.get("parse_dropped_ambiguous", 0)) for m in meta)
        )
        obj_total = pred_total + dropped_invalid_total + dropped_ambiguous_total
        obj_valid_frac = (pred_total / obj_total) if obj_total > 0 else 0.0
        obj_drop_frac = (
            ((dropped_invalid_total + dropped_ambiguous_total) / obj_total)
            if obj_total > 0
            else 0.0
        )

        trunc_samples = float(sum(1 for m in meta if m.get("parse_truncated")))
        trunc_rate = (trunc_samples / n_samples) if n_samples > 0 else 0.0

        gate_rejections_total = float(
            sum(int(m.get("gating_rejections", 0)) for m in meta)
        )
        top_k = int(self._cfg("candidate_top_k", 10))
        gate_rejection_rate = (
            (gate_rejections_total / (pred_total * float(max(1, top_k))))
            if pred_total > 0
            else 0.0
        )

        matched_iou_sum = float(
            sum(float(m.get("matched_maskiou_sum", 0.0)) for m in meta)
        )
        matched_iou_count = float(
            sum(int(m.get("matched_maskiou_count", 0)) for m in meta)
        )
        matched_iou_mean = (
            (matched_iou_sum / matched_iou_count) if matched_iou_count > 0 else 0.0
        )

        # Supervision coverage diagnostics.
        excluded_rate = (
            (excluded_total / (matched_total + excluded_total))
            if (matched_total + excluded_total) > 0
            else 0.0
        )
        prefix_targets_total = float(
            sum(len(m.get("prefix_coord_target_bins") or []) for m in meta)
        )
        prefix_targets_per_matched = (
            (prefix_targets_total / matched_total) if matched_total > 0 else 0.0
        )
        tail_ignore_total = float(
            sum(len(m.get("tail_ignore_pos") or []) for m in meta)
        )
        append_len_total = float(
            sum(
                max(0, int(m.get("train_len", 0)) - int(m.get("prefix_len", 0)))
                for m in meta
            )
        )
        tail_ignore_frac = (
            (tail_ignore_total / append_len_total) if append_len_total > 0 else 0.0
        )

        # Length stats (prompt/prefix/train/encoded) help diagnose truncation/packing behavior.
        def _int_list(key: str) -> List[int]:
            xs: List[int] = []
            for m in meta:
                try:
                    xs.append(int(m.get(key, 0)))
                except (TypeError, ValueError):
                    continue
            return xs

        prompt_lens = _int_list("prompt_len")
        prefix_lens = _int_list("prefix_len")
        train_lens = _int_list("train_len")
        encoded_lens = _int_list("encoded_len")
        rollout_lens = _int_list("rollout_len")
        append_lens: List[int] = []
        for m in meta:
            try:
                append_lens.append(
                    int(m.get("train_len", 0)) - int(m.get("prefix_len", 0))
                )
            except (TypeError, ValueError):
                continue

        def _mean(xs: List[int]) -> float:
            return float(sum(xs) / len(xs)) if xs else 0.0

        def _p(xs: List[int], q: float) -> float:
            if not xs:
                return 0.0
            arr = np.asarray(xs, dtype=np.float64)
            return float(np.percentile(arr, float(q)))

        payload: Dict[str, float] = {
            "rollout/parse_dropped_invalid": float(dropped_invalid_total),
            "rollout/parse_dropped_ambiguous": float(dropped_ambiguous_total),
            "rollout/parse_truncated": float(trunc_samples),
            "rollout/parse_truncated_rate": float(trunc_rate),
            "rollout/parse_obj_total": float(obj_total),
            "rollout/parse_obj_valid_frac": float(obj_valid_frac),
            "rollout/parse_obj_drop_frac": float(obj_drop_frac),
            "rollout/sample_valid_pred_rate": float(sample_valid_pred_rate),
            "rollout/sample_any_match_rate": float(sample_any_match_rate),
            "rollout/fn_appended_total": float(sum(int(m.get("fn_count", 0)) for m in meta)),
            "rollout/gating_rejections": float(gate_rejections_total),
            "rollout/gating_rejection_rate": float(gate_rejection_rate),
            "rollout/valid_pred_objects_total": float(pred_total),
            "rollout/gt_objects_total": float(gt_total),
            "rollout/precision": float(precision),
            "rollout/recall": float(recall),
            "rollout/f1": float(f1),
            "rollout/fp_total": float(fp_total),
            "rollout/fn_total": float(fn_total),
            "rollout/gt_per_sample": float(gt_total / n_samples)
            if n_samples > 0
            else 0.0,
            "rollout/pred_per_sample": float(pred_total / n_samples)
            if n_samples > 0
            else 0.0,
            "rollout/fp_per_sample": float(fp_total / n_samples)
            if n_samples > 0
            else 0.0,
            "rollout/fn_per_sample": float(fn_total / n_samples)
            if n_samples > 0
            else 0.0,
            "rollout/matched_maskiou_mean": float(matched_iou_mean),
            "rollout/matched_maskiou_count": float(matched_iou_count),
            "rollout/excluded_rate": float(excluded_rate),
            "rollout/prefix_coord_targets_total": float(prefix_targets_total),
            "rollout/prefix_coord_targets_per_matched": float(
                prefix_targets_per_matched
            ),
            "rollout/tail_ignore_frac": float(tail_ignore_frac),
            "rollout/prompt_len_mean": float(_mean(prompt_lens)),
            "rollout/prompt_len_p90": float(_p(prompt_lens, 90)),
            "rollout/prefix_len_mean": float(_mean(prefix_lens)),
            "rollout/rollout_len_mean": float(_mean(rollout_lens)),
            "rollout/rollout_len_p90": float(_p(rollout_lens, 90)),
            "rollout/train_len_mean": float(_mean(train_lens)),
            "rollout/train_len_p90": float(_p(train_lens, 90)),
            "rollout/append_len_mean": float(_mean(append_lens)),
            "rollout/append_len_p90": float(_p(append_lens, 90)),
            "rollout/encoded_len_mean": float(_mean(encoded_lens)),
            "rollout/encoded_len_p90": float(_p(encoded_lens, 90)),
            "rollout/decode_non_beam_count": float(
                sum(1 for m in meta if str(m.get("decode_mode", "")).lower() != "beam")
            ),
            "rollout/decode_beam_count": float(
                sum(1 for m in meta if str(m.get("decode_mode", "")).lower() == "beam")
            ),
            "rollout/matched_for_supervision": float(matched_total),
            "rollout/excluded_from_supervision": float(excluded_total),
        }

        try:
            decode_request = build_decode_request_from_rollout_owner(self)
            temperature = float(decode_request.temperature)
            top_p = float(decode_request.top_p)
            top_k = int(decode_request.top_k)
            do_sample = str(decode_request.decode_mode) == "sampling"
            payload["rollout/do_sample"] = float(1.0 if do_sample else 0.0)
            payload["rollout/temperature"] = float(temperature)
            payload["rollout/top_p"] = float(top_p)
            payload["rollout/top_k"] = float(top_k)
        except (TypeError, ValueError):
            decode_modes = [str(m.get("decode_mode", "")).lower() for m in meta]
            temperatures: List[float] = []
            top_ps: List[float] = []
            top_ks: List[float] = []
            for m in meta:
                try:
                    temperatures.append(float(m.get("rollout_temperature")))
                except (TypeError, ValueError):
                    pass
                try:
                    top_ps.append(float(m.get("rollout_top_p")))
                except (TypeError, ValueError):
                    pass
                try:
                    top_ks.append(float(m.get("rollout_top_k")))
                except (TypeError, ValueError):
                    pass
            do_sample = any(mode == "sampling" for mode in decode_modes) or any(
                float(value) > 0.0 for value in temperatures
            )
            payload["rollout/do_sample"] = float(1.0 if do_sample else 0.0)
            if temperatures:
                payload["rollout/temperature"] = float(_mean(temperatures))
            if top_ps:
                payload["rollout/top_p"] = float(_mean(top_ps))
            if top_ks:
                payload["rollout/top_k"] = float(_mean(top_ks))

        # Desc monitor outputs (matched pairs only).
        try:
            if any(bool(m.get("desc_monitor_ran", False)) for m in meta):
                pairs_total = float(
                    sum(int(m.get("desc_pairs_total", 0)) for m in meta)
                )
                exact_ok_total = float(
                    sum(int(m.get("desc_exact_ok", 0)) for m in meta)
                )
                exact_acc = (exact_ok_total / pairs_total) if pairs_total > 0 else 1.0
                payload["rollout/desc_pairs_total"] = float(pairs_total)
                payload["rollout/desc_exact_acc_on_matched"] = float(exact_acc)

                sem_enabled_total = float(
                    sum(int(m.get("desc_sem_enabled", 0)) for m in meta)
                )
                payload["rollout/desc_sem_enabled"] = float(
                    1.0 if sem_enabled_total > 0 else 0.0
                )
                if sem_enabled_total > 0:
                    sem_ok_total = float(
                        sum(int(m.get("desc_sem_ok", 0)) for m in meta)
                    )
                    sem_acc = (sem_ok_total / pairs_total) if pairs_total > 0 else 1.0
                    payload["rollout/desc_sem_acc_on_matched"] = float(sem_acc)

                    sim_sum_total = float(
                        sum(float(m.get("desc_sem_sim_sum", 0.0)) for m in meta)
                    )
                    sim_count_total = float(
                        sum(int(m.get("desc_sem_sim_count", 0)) for m in meta)
                    )
                    if sim_count_total > 0:
                        payload["rollout/desc_sem_sim_mean"] = float(
                            sim_sum_total / sim_count_total
                        )
                        payload["rollout/desc_sem_sim_count"] = float(sim_count_total)
        except (TypeError, ValueError):
            raise

        return payload

    def _build_train_rollout_log_payload(
        self, pending: _PendingTrainRolloutLog
    ) -> Dict[str, float]:
        payload: Dict[str, float] = {}

        sample_total = float(len(pending.meta))
        payload["train/samples_total"] = float(sample_total)
        payload["train/micro_steps"] = float(pending.n_micro)
        payload["train/samples_per_micro"] = (
            float(sample_total / float(pending.n_micro)) if pending.n_micro > 0 else 0.0
        )


        if sample_total > 0.0:
            if float(getattr(pending, "loss_weight_sum", 0.0)) > 0.0:
                denom = float(pending.loss_weight_sum)
                atom_keys = sorted(
                    set(getattr(pending, "objective_atom_weighted_sum", {}).keys())
                    | set(getattr(pending, "objective_atom_sum", {}).keys())
                )
                for key in atom_keys:
                    payload[str(key)] = float(
                        float(getattr(pending, "objective_atom_weighted_sum", {}).get(key, 0.0))
                        / denom
                    )
            elif pending.n_micro > 0:
                atom_keys = sorted(
                    set(getattr(pending, "objective_atom_sum", {}).keys())
                    | set(getattr(pending, "objective_atom_weighted_sum", {}).keys())
                )
                denom = float(pending.n_micro)
                for key in atom_keys:
                    payload[str(key)] = float(
                        float(getattr(pending, "objective_atom_sum", {}).get(key, 0.0))
                        / denom
                    )
        if float(getattr(pending, "gradmon_weight_sum", 0.0)) > 0.0:
            gradmon_denom = float(getattr(pending, "gradmon_weight_sum", 0.0))
            for key in sorted(getattr(pending, "gradmon_weighted_sum", {}).keys()):
                payload[str(key)] = float(
                    float(getattr(pending, "gradmon_weighted_sum", {}).get(key, 0.0))
                    / gradmon_denom
                )
            payload["gradmon/_log_weight_total"] = float(gradmon_denom)

        payload["time/forward_s"] = float(pending.time_forward_s)
        if float(pending.time_mask_build_s) > 0.0:
            payload["time/mask_build_s"] = float(pending.time_mask_build_s)
        if float(getattr(pending, "time_gradmon_s", 0.0)) > 0.0:
            payload["time/gradmon_s"] = float(getattr(pending, "time_gradmon_s", 0.0))

        # Rollout pipeline timings are only meaningful when we actually ran a rollout.
        ran_rollout = bool(
            float(pending.time_rollout_generate_s) > 0.0
            or float(pending.time_rollout_parse_match_s) > 0.0
            or float(pending.time_rollout_teacher_encode_s) > 0.0
        )
        if ran_rollout:
            payload.update(self._build_rollout_metrics_from_meta(pending.meta))
            if "rollout/parse_truncated" in payload:
                payload["rollout/_parse_truncated_num"] = float(
                    payload.get("rollout/parse_truncated", 0.0)
                )
                payload["rollout/_parse_truncated_den"] = float(sample_total)

            payload["time/rollout_generate_s"] = float(pending.time_rollout_generate_s)
            payload["time/rollout_parse_match_s"] = float(pending.time_rollout_parse_match_s)
            payload["time/rollout_teacher_encode_s"] = float(
                pending.time_rollout_teacher_encode_s
            )
        if pending.time_post_rollout_pack_s > 0:
            payload["time/post_rollout_pack_s"] = float(pending.time_post_rollout_pack_s)

        if pending.packing_count > 0:
            payload["packing/post_rollout_fill"] = float(
                pending.packing_fill_sum / float(pending.packing_count)
            )
            payload["packing/post_rollout_selected_total_len"] = float(
                pending.packing_selected_total_len_sum / float(pending.packing_count)
            )
            payload["packing/post_rollout_segments"] = float(
                pending.packing_segments_sum / float(pending.packing_count)
            )
            payload["packing/post_rollout_buffer"] = float(pending.packing_buffer_last)

        # Generation-length stats are only meaningful when we actually ran a rollout this step.
        if pending.time_rollout_generate_s > 0:
            rollout_lens = [int(m.get("rollout_len", 0)) for m in pending.meta]

            def _p(xs: List[int], q: float) -> float:
                if not xs:
                    return 0.0
                arr = np.asarray(xs, dtype=np.float64)
                return float(np.percentile(arr, float(q)))

            new_tok_total = float(sum(int(x) for x in rollout_lens))
            new_tok_mean = (
                float(new_tok_total / len(rollout_lens)) if rollout_lens else 0.0
            )
            payload["rollout/gen_new_tokens_total"] = float(new_tok_total)
            payload["rollout/gen_new_tokens_mean"] = float(new_tok_mean)
            payload["rollout/gen_new_tokens_p90"] = float(_p(rollout_lens, 90))
            payload["rollout/gen_new_tokens_p99"] = float(_p(rollout_lens, 99))
            payload["rollout/gen_tokens_per_s"] = float(
                (new_tok_total / float(pending.time_rollout_generate_s))
                if pending.time_rollout_generate_s > 0
                else 0.0
            )

        return payload

    def get_train_dataloader(self):
        dl = super().get_train_dataloader()

        gas = int(getattr(self.args, "gradient_accumulation_steps", 1) or 1)

        # Drop the final partial accumulation window when requested.
        #
        # This keeps optimizer-step semantics consistent for step-budgeted Stage-2 runs
        # (fixed samples per step) and avoids a trailing underfull/no-op step.
        try:
            drop_last = bool(getattr(self.args, "dataloader_drop_last", False))
        except (TypeError, ValueError):
            drop_last = False
        if self._packing_enabled() and drop_last and int(gas) > 1:
            dl = _DropRemainderAccumulationWindow(dl, gas=gas)

        return dl

    def _determine_best_metric(self, metrics, trial):
        """Resolve Stage-2 metric aliases before best-checkpoint selection.

        `transformers.Trainer` looks up `eval_<metric_for_best_model>`, but this
        trainer emits grouped eval keys like `eval/detection/mAP`.
        """

        is_new_best_metric = False

        metric_name = getattr(self.args, "metric_for_best_model", None)
        if metric_name is None:
            return is_new_best_metric

        resolved_metric = resolve_metric_value(metrics, str(metric_name))
        if resolved_metric is None:
            metric_candidates = list(metric_lookup_candidates(str(metric_name)))
            display_metric = (
                metric_candidates[1]
                if len(metric_candidates) > 1
                and metric_candidates[0] == str(metric_name).strip()
                else (metric_candidates[0] if metric_candidates else str(metric_name))
            )
            raise KeyError(
                f"The `metric_for_best_model` training argument is set to '{display_metric}', which is not found in the evaluation metrics. "
                f"Tried aliases: {metric_candidates}. The available evaluation metrics are: {list(metrics.keys())}. "
                "Consider changing the `metric_for_best_model` via the TrainingArguments."
            )

        _resolved_metric_key, metric_value = resolved_metric
        operator = np.greater if self.args.greater_is_better else np.less

        if self.state.best_metric is None:
            self.state.best_metric = (
                float("-inf") if self.args.greater_is_better else float("inf")
            )

        if operator(metric_value, self.state.best_metric):
            self.state.best_metric = metric_value

            if self.args.save_strategy in [SaveStrategy.STEPS, SaveStrategy.EPOCH]:
                self.state.best_global_step = self.state.global_step

            is_new_best_metric = True

        return is_new_best_metric

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ):
        """Production-style evaluator: rollout -> parse -> greedy IoU match.

        This intentionally skips teacher-forced encoding and loss computation to keep eval
        fast and reflective of real rollout performance on unseen data.
        """

        was_training = bool(getattr(self.model, "training", False))
        self.model.eval()

        t0 = time.perf_counter()
        dl = self.get_eval_dataloader(eval_dataset)

        template = self.template
        tok = template.tokenizer

        gate_thr = float(self._cfg("maskiou_gate", 0.3))
        top_k = int(self._cfg("candidate_top_k", 10))
        mask_res = int(self._cfg("maskiou_resolution", 256))
        fp_cost = float(self._cfg("fp_cost", 1.0))
        fn_cost = float(self._cfg("fn_cost", 1.0))
        object_field_order = self._object_field_order()
        eval_rollout_template_policy = self._eval_rollout_template_policy()

        eval_prompt_variant = self._eval_prompt_variant()
        eval_detection_cfg = self._eval_detection_cfg()
        eval_detection_enabled = bool(eval_detection_cfg.get("enabled", True))
        eval_detection_score_mode = str(
            eval_detection_cfg.get("score_mode", "constant") or "constant"
        ).strip().lower()
        eval_detection_const_score = float(
            eval_detection_cfg.get("constant_score", 1.0) or 1.0
        )
        eval_detection_score_source = str(
            eval_detection_cfg.get("pred_score_source", "eval_rollout_constant")
            or "eval_rollout_constant"
        ).strip()
        eval_detection_score_version = int(
            eval_detection_cfg.get("pred_score_version", 2) or 2
        )

        eval_detection_use_confidence_postop = bool(
            eval_detection_enabled
            and eval_detection_score_mode in {"confidence_postop", "confidence"}
        )
        eval_rollout_backend = self._effective_rollout_backend(context="eval")
        eval_vllm_mode = self._vllm_mode() if eval_rollout_backend == "vllm" else "n/a"
        logger.info(
            "Starting evaluate(): resolved_eval_rollout_backend=%s vllm_mode=%s",
            eval_rollout_backend,
            eval_vllm_mode,
        )

        confidence_postop_opts = None
        if eval_detection_use_confidence_postop:
            if eval_rollout_backend not in {"hf", "vllm"}:
                raise ValueError(
                    "rollout_matching.eval_detection.score_mode=confidence_postop requires "
                    "effective eval rollout backend in {'hf', 'vllm'} "
                    "(token logprob traces must be available)."
                )

            # Validate early so failures are consistent across ranks.
            confidence_postop_opts = confidence_options_from_eval_config(
                eval_detection_cfg.get("confidence", None)
            )

        # Optional semantic desc monitoring (metrics only).
        desc_cfg = self._desc_monitor_cfg()
        desc_enabled = isinstance(desc_cfg, Mapping) and bool(
            desc_cfg.get("enabled", False)
        )
        desc_mode = str(desc_cfg.get("mode", "semantic") or "semantic").strip().lower()
        desc_thr = float(desc_cfg.get("semantic_threshold", 0.6) or 0.6)
        desc_max_pairs = int(desc_cfg.get("max_pairs", 0) or 0)

        try:
            from src.metrics.semantic_desc import normalize_desc
        except (TypeError, ValueError):
            normalize_desc = None  # type: ignore[assignment]

        sem_loaded_local = 0.0
        sem_encoder = None
        if desc_enabled and desc_mode in {"semantic", "both"}:
            try:
                sem_encoder = self._get_desc_semantic_encoder(desc_cfg)
            except (TypeError, ValueError):
                sem_encoder = None
            if sem_encoder is not None:
                # Probe once so failures are detected consistently across ranks.
                try:
                    _ = sem_encoder.encode_norm_texts(["__probe__"])
                    sem_loaded_local = 1.0
                except (TypeError, ValueError):
                    sem_encoder = None
                    sem_loaded_local = 0.0

        n_samples = 0.0
        gt_total = 0.0
        pred_total = 0.0
        matched_total = 0.0
        fp_total = 0.0
        fn_total = 0.0
        gating_rejections_total = 0.0
        dropped_invalid_total = 0.0
        dropped_ambiguous_total = 0.0
        trunc_samples = 0.0
        matched_iou_sum = 0.0
        matched_iou_count = 0.0
        n_samples_valid_pred = 0.0
        n_samples_any_match = 0.0

        # Desc monitor accumulators (matched pairs only).
        desc_pairs_total = 0.0
        desc_exact_ok_total = 0.0
        desc_sem_ok_total = 0.0
        desc_sem_sim_sum_total = 0.0
        desc_sem_sim_count_total = 0.0

        n_steps = 0.0
        eval_detection_records_local: List[Dict[str, Any]] = []
        eval_rollout_artifacts_local: List[Dict[str, Any]] = []
        eval_record_counter_local = 0
        vllm_decode_error_count_local = 0.0

        # Optional qualitative monitor dumps during eval (rank0 only).
        gs = int(getattr(getattr(self, "state", None), "global_step", 0) or 0)
        eval_dump_index = int(getattr(self, "_eval_monitor_dump_eval_index", 0) or 0) + 1
        self._eval_monitor_dump_eval_index = int(eval_dump_index)
        do_dump = False
        dump_cfg = self._eval_monitor_dump_cfg()
        dump_max_samples = 0
        dump_max_chars = 0
        dump_fail_samples: List[Dict[str, Any]] = []
        dump_other_samples: List[Dict[str, Any]] = []
        if self._should_eval_monitor_dump(global_step=gs, eval_index=eval_dump_index):
            do_dump = True
            dump_max_samples = max(1, int(dump_cfg.get("max_samples", 1) or 1))
            dump_max_chars_raw = dump_cfg.get("max_text_chars", 4000)
            try:
                dump_max_chars = (
                    int(dump_max_chars_raw) if dump_max_chars_raw is not None else 4000
                )
            except Exception:
                dump_max_chars = 4000
            dump_max_chars = max(0, int(dump_max_chars))
            # Mark early to avoid duplicate dumps in the same eval invocation.
            self._eval_monitor_dump_last_eval = int(eval_dump_index)

        self._write_eval_phase_trace(
            phase="start",
            global_step=int(gs),
            eval_index=int(eval_dump_index),
            payload={
                "eval_rollout_backend": str(eval_rollout_backend),
                "eval_vllm_mode": str(eval_vllm_mode),
                "eval_detection_enabled": bool(eval_detection_enabled),
                "eval_detection_score_mode": str(eval_detection_score_mode),
                "has_token_trace": bool(eval_detection_use_confidence_postop),
                "eval_prompt_variant": str(eval_prompt_variant),
            },
        )

        with torch.no_grad(), self._maybe_eval_vllm_colocate_window(
            rollout_backend=eval_rollout_backend
        ):
            for batch in dl:
                # For rollout-matching, we expect identity_data_collator to yield a
                # list[dict] of raw samples (with `messages` + GT geometry).
                if not isinstance(batch, list):
                    raise ValueError(
                        "rollout-matching evaluator expects eval batches as list[dict]; "
                        f"got {type(batch).__name__}"
                    )
                if not batch:
                    continue

                n_steps += 1.0

                has_token_trace = bool(eval_detection_use_confidence_postop)
                eval_decode_override = self._eval_decode_override(
                    has_token_trace=has_token_trace
                )
                batch_index = int(n_steps)
                self._write_eval_phase_trace(
                    phase=f"before_rollout_batch_{batch_index:04d}",
                    global_step=int(gs),
                    eval_index=int(eval_dump_index),
                    payload={
                        "batch_index": int(batch_index),
                        "batch_size": int(len(batch)),
                        "has_token_trace": bool(has_token_trace),
                        "eval_rollout_backend": str(eval_rollout_backend),
                        "eval_vllm_mode": str(eval_vllm_mode),
                        "decode_override": (
                            dict(eval_decode_override)
                            if isinstance(eval_decode_override, Mapping)
                            else None
                        ),
                    },
                )
                sample_rollouts: List[Tuple[Mapping[str, Any], Any]] = []
                if has_token_trace:
                    try:
                        rollout_results = rollout_many_traced(
                            owner=self,
                            samples=batch,
                            prompt_variant_override=eval_prompt_variant,
                            rollout_backend=eval_rollout_backend,
                            decode_override=eval_decode_override,
                        )
                    except Exception as exc:
                        self._write_eval_phase_trace(
                            phase=f"rollout_batch_{batch_index:04d}_exception",
                            global_step=int(gs),
                            eval_index=int(eval_dump_index),
                            payload={
                                "batch_index": int(batch_index),
                                "batch_size": int(len(batch)),
                                "has_token_trace": bool(has_token_trace),
                                "error_type": str(exc.__class__.__name__),
                                "error": str(exc),
                            },
                        )
                        raise
                    if len(rollout_results) != len(batch):
                        raise RuntimeError(
                            "rollout backend returned unexpected number of results"
                        )
                    rollout_token_lengths = [len(r[0]) for r in rollout_results]
                    self._write_eval_phase_trace(
                        phase=f"after_rollout_batch_{batch_index:04d}",
                        global_step=int(gs),
                        eval_index=int(eval_dump_index),
                        payload={
                            "batch_index": int(batch_index),
                            "batch_size": int(len(batch)),
                            "result_count": int(len(rollout_results)),
                            "response_token_lengths": rollout_token_lengths,
                            "response_token_length_max": (
                                int(max(rollout_token_lengths))
                                if rollout_token_lengths
                                else 0
                            ),
                        },
                    )
                    sample_rollouts = list(zip(batch, rollout_results))
                else:
                    try:
                        rollout_results = self._rollout_many(
                            batch,
                            prompt_variant_override=eval_prompt_variant,
                            rollout_backend=eval_rollout_backend,
                            decode_override=eval_decode_override,
                        )
                        if len(rollout_results) != len(batch):
                            raise RuntimeError(
                                "rollout backend returned unexpected number of results"
                            )
                        rollout_token_lengths = [len(r[0]) for r in rollout_results]
                        self._write_eval_phase_trace(
                            phase=f"after_rollout_batch_{batch_index:04d}",
                            global_step=int(gs),
                            eval_index=int(eval_dump_index),
                            payload={
                                "batch_index": int(batch_index),
                                "batch_size": int(len(batch)),
                                "result_count": int(len(rollout_results)),
                                "response_token_lengths": rollout_token_lengths,
                                "response_token_length_max": (
                                    int(max(rollout_token_lengths))
                                    if rollout_token_lengths
                                    else 0
                                ),
                            },
                        )
                        sample_rollouts = list(zip(batch, rollout_results))
                    except Exception as batch_exc:
                        self._write_eval_phase_trace(
                            phase=f"rollout_batch_{batch_index:04d}_exception",
                            global_step=int(gs),
                            eval_index=int(eval_dump_index),
                            payload={
                                "batch_index": int(batch_index),
                                "batch_size": int(len(batch)),
                                "has_token_trace": bool(has_token_trace),
                                "error_type": str(batch_exc.__class__.__name__),
                                "error": str(batch_exc),
                            },
                        )
                        if eval_rollout_backend != "vllm":
                            raise
                        sample_rollouts = []
                        for sample_idx, sample in enumerate(batch):
                            try:
                                rollout_one = self._rollout_many(
                                    [sample],
                                    prompt_variant_override=eval_prompt_variant,
                                    rollout_backend=eval_rollout_backend,
                                    decode_override=eval_decode_override,
                                )
                                if len(rollout_one) != 1:
                                    raise RuntimeError(
                                        "rollout backend returned unexpected number of results"
                                    )
                                sample_rollouts.append((sample, rollout_one[0]))
                            except Exception as sample_exc:
                                vllm_decode_error_count_local += 1.0
                                logger.warning(
                                    "Eval vLLM decode failed for sample_idx=%s; skipping sample. "
                                    "error=%s: %s",
                                    int(sample_idx),
                                    sample_exc.__class__.__name__,
                                    sample_exc,
                                )
                        if not sample_rollouts:
                            raise RuntimeError(
                                "Eval vLLM rollout failed for all samples in a batch; "
                                "aborting evaluation. "
                                f"batch_error={batch_exc.__class__.__name__}: {batch_exc}"
                            ) from batch_exc

                for sample, rollout in sample_rollouts:
                    if has_token_trace:
                        (
                            resp_ids,
                            raw_resp_text,
                            _decode_mode,
                            _prompt_ids,
                            token_logprobs,
                            generated_token_text,
                        ) = rollout
                    else:
                        resp_ids, raw_resp_text, _decode_mode, _prompt_ids = rollout
                        token_logprobs = None
                        generated_token_text = None
                    n_samples += 1.0

                    try:
                        parsed_rollout = parse_stage2_detection_rollout_predictions(
                            tokenizer=tok,
                            response_token_ids=resp_ids,
                            response_text=str(raw_resp_text or ""),
                            rollout_template_policy=eval_rollout_template_policy,
                            object_field_order=object_field_order,
                            coord_id_to_bin=self._coord_id_map(),
                            gt_object_factory=GTObject,
                            compact_rollout_codec_factory=CompactFullRolloutCodec,
                            parse_rollout_for_matching_fn=parse_rollout_for_matching,
                            points_from_coord_tokens_fn=_points_from_coord_tokens,
                        )
                    except Exception as exc:
                        if not eval_detection_enabled:
                            raise
                        eval_record_index = int(eval_record_counter_local)
                        eval_record_counter_local += 1
                        parser_artifact_metadata = diagnostic_parser_result(
                            predictions=(),
                            parser_id="stage2_eval_parse_exception",
                            errors=(exc.__class__.__name__,),
                            diagnostics={
                                "exception_type": exc.__class__.__name__,
                                "exception_message": str(exc),
                            },
                            salvage_recovered=False,
                        ).to_artifact_metadata()
                        eval_rollout_artifacts_local.append(
                            _build_stage2_eval_invalid_artifact_record(
                                eval_record_index=int(eval_record_index),
                                sample=sample,
                                parser_artifact_metadata=parser_artifact_metadata,
                                reason="parser_exception_not_metric_bearing",
                                message=(
                                    "Stage-2 official eval parser raised before "
                                    f"metric-bearing output: {exc.__class__.__name__}: {exc}"
                                ),
                            )
                        )
                        continue
                    parse = parsed_rollout.parse
                    pred_meta: List[Any] = list(parsed_rollout.pred_meta)
                    preds: List[GTObject] = list(parsed_rollout.preds)
                    pred_objs_dump: List[Dict[str, Any]] = [
                        dict(obj) for obj in parsed_rollout.pred_objects_dump
                    ]

                    dropped_invalid_total += float(parse.dropped_invalid)
                    dropped_ambiguous_total += float(parse.dropped_ambiguous)
                    trunc_samples += 1.0 if bool(parse.truncated) else 0.0

                    gts = _extract_gt_objects(sample)
                    eval_error_codes: List[str] = []
                    eval_error_entries: List[Dict[str, str]] = []
                    if bool(getattr(parse, "invalid_rollout", False)):
                        eval_error_codes.append("invalid_rollout")
                        eval_error_entries.append(
                            {
                                "code": "invalid_rollout",
                                "message": "Rollout parsing marked this sample invalid.",
                                "stage": "eval_rollout_parse",
                            }
                        )
                    if int(getattr(parse, "dropped_invalid", 0) or 0) > 0:
                        eval_error_codes.append("dropped_invalid_objects")
                        eval_error_entries.append(
                            {
                                "code": "dropped_invalid_objects",
                                "message": (
                                    "One or more predicted objects were dropped as invalid "
                                    "during rollout parsing."
                                ),
                                "stage": "eval_rollout_parse",
                            }
                        )
                    if int(getattr(parse, "dropped_ambiguous", 0) or 0) > 0:
                        eval_error_codes.append("dropped_ambiguous_objects")
                        eval_error_entries.append(
                            {
                                "code": "dropped_ambiguous_objects",
                                "message": (
                                    "One or more predicted objects were dropped as ambiguous "
                                    "during rollout parsing."
                                ),
                                "stage": "eval_rollout_parse",
                            }
                        )
                    if bool(getattr(parse, "truncated", False)):
                        eval_error_codes.append("truncated_rollout")
                        eval_error_entries.append(
                            {
                                "code": "truncated_rollout",
                                "message": "Rollout output was truncated before full completion.",
                                "stage": "eval_rollout_parse",
                            }
                        )
                    gt_total += float(len(gts))
                    pred_total += float(len(preds))
                    if len(preds) > 0:
                        n_samples_valid_pred += 1.0

                    match = greedy_match_iou(
                        preds=preds,
                        gts=gts,
                        gate_threshold=gate_thr,
                    )

                    matched = float(len(match.matched_pairs))
                    matched_total += matched
                    fp_total += float(len(match.fp_pred_indices))
                    fn_total += float(len(match.fn_gt_indices))
                    gating_rejections_total += float(match.gating_rejections)
                    matched_iou_sum += float(match.matched_maskiou_sum)
                    matched_iou_count += float(match.matched_maskiou_count)
                    if matched > 0:
                        n_samples_any_match += 1.0

                    if eval_detection_enabled:
                        eval_record_index = int(eval_record_counter_local)
                        eval_record_counter_local += 1
                        parser_result = parsed_rollout.parser_result
                        parser_artifact_metadata = (
                            parser_result.to_artifact_metadata()
                        )
                        if (
                            not bool(parser_result.metric_bearing)
                            or parser_result.parser_policy != "strict"
                            or bool(parser_result.salvage_recovered)
                        ):
                            eval_rollout_artifacts_local.append(
                                _build_stage2_eval_invalid_artifact_record(
                                    eval_record_index=int(eval_record_index),
                                    sample=sample,
                                    parser_artifact_metadata=parser_artifact_metadata,
                                    reason="parser_result_not_metric_bearing",
                                    message=(
                                        "Stage-2 official eval requires strict "
                                        "metric-bearing parser output."
                                    ),
                                )
                            )
                            continue

                        confidence_objects_payload: List[Dict[str, Any]] = []
                        try:
                            base_eval_record = _build_eval_detection_record_confidence_postop_input(
                                sample=sample,
                                gts=gts,
                                preds=preds,
                                pred_meta=pred_meta,
                                object_field_order=object_field_order,
                                record_index=eval_record_index,
                                raw_text=raw_resp_text,
                                error_codes=eval_error_codes,
                                error_entries=eval_error_entries,
                            )
                        except ValueError as exc:
                            eval_rollout_artifacts_local.append(
                                _build_stage2_eval_invalid_artifact_record(
                                    eval_record_index=int(eval_record_index),
                                    sample=sample,
                                    parser_artifact_metadata=parser_artifact_metadata,
                                    reason="source_identity_or_dimensions_invalid",
                                    message=str(exc),
                                )
                            )
                            continue
                        scored_eval_record: Dict[str, Any] | None = None

                        if eval_detection_use_confidence_postop:
                            (
                                scored_eval_record,
                                confidence_objects_payload,
                            ) = score_stage2_confidence_eval_record(
                                line_idx=int(eval_record_index),
                                base_eval_record=base_eval_record,
                                parse=parse,
                                token_logprobs=token_logprobs,
                                generated_token_text=generated_token_text,
                                confidence_postop_opts=confidence_postop_opts,
                            )
                            eval_detection_records_local.append(scored_eval_record)
                        else:
                            if eval_detection_score_mode != "constant":
                                raise ValueError(
                                    "rollout_matching.eval_detection.score_mode must be 'constant' "
                                    "or 'confidence_postop'"
                                )

                            scored_eval_record = _build_eval_detection_record(
                                sample=sample,
                                gts=gts,
                                preds=preds,
                                pred_meta=pred_meta,
                                object_field_order=object_field_order,
                                record_index=eval_record_index,
                                pred_score_source=eval_detection_score_source,
                                pred_score_version=eval_detection_score_version,
                                score_mode=eval_detection_score_mode,
                                constant_score=eval_detection_const_score,
                                raw_text=raw_resp_text,
                                error_codes=eval_error_codes,
                                error_entries=eval_error_entries,
                            )
                            eval_detection_records_local.append(scored_eval_record)

                        if scored_eval_record is not None:
                            artifact_record = (
                                build_stage2_rollout_eval_artifact_record(
                                    eval_record_index=int(eval_record_index),
                                    sample=sample,
                                    base_eval_record=base_eval_record,
                                    scored_eval_record=scored_eval_record,
                                    response_token_ids=resp_ids,
                                    prompt_token_ids=_prompt_ids,
                                    decode_mode=str(_decode_mode),
                                    response_text=str(raw_resp_text or ""),
                                    generated_token_text=generated_token_text,
                                    token_logprobs=token_logprobs,
                                    parse=parse,
                                    pred_objects_dump=pred_objs_dump,
                                    eval_error_codes=eval_error_codes,
                                    eval_error_entries=eval_error_entries,
                                    match=match,
                                    confidence_objects_payload=confidence_objects_payload,
                                )
                            )
                            artifact_record["parser_result"] = dict(
                                parser_artifact_metadata
                            )
                            parse_payload = artifact_record.get("parse")
                            if isinstance(parse_payload, MutableMapping):
                                parse_payload["parser_result"] = dict(
                                    parser_artifact_metadata
                                )
                            eval_rollout_artifacts_local.append(artifact_record)

                    if do_dump and (
                        len(dump_fail_samples) < dump_max_samples
                        or len(dump_other_samples) < dump_max_samples
                    ):
                        is_failure = (
                            bool(getattr(parse, "invalid_rollout", False))
                            or int(getattr(parse, "dropped_invalid", 0) or 0) > 0
                            or int(getattr(parse, "dropped_ambiguous", 0) or 0) > 0
                            or bool(getattr(parse, "truncated", False))
                            or len(preds) == 0
                        )
                        target = dump_fail_samples if is_failure else dump_other_samples
                        if len(target) < dump_max_samples:
                            gt_objs_dump = [
                                {
                                    "index": int(o.index),
                                    "geom_type": str(o.geom_type),
                                    "points_norm1000": list(o.points_norm1000),
                                    "desc": str(o.desc),
                                }
                                for o in gts
                            ]

                            pred_n = float(len(preds))
                            gt_n = float(len(gts))
                            matched_n = float(len(match.matched_pairs))
                            prec_local = (matched_n / pred_n) if pred_n > 0 else 0.0
                            rec_local = (matched_n / gt_n) if gt_n > 0 else 0.0
                            f1_local = (
                                (2.0 * prec_local * rec_local / (prec_local + rec_local))
                                if (prec_local + rec_local) > 0.0
                                else 0.0
                            )

                            drop_by_reason: Dict[str, int] = {}
                            try:
                                raw = getattr(parse, "dropped_invalid_by_reason", None)
                                if isinstance(raw, Mapping):
                                    for k, v in raw.items():
                                        try:
                                            drop_by_reason[str(k)] = int(v)
                                        except (TypeError, ValueError):
                                            continue
                            except Exception:
                                drop_by_reason = {}

                            target.append(
                                {
                                    "sample_id": sample.get("sample_id"),
                                    "base_idx": sample.get("base_idx"),
                                    "image": sample.get("image"),
                                    "images": sample.get("images"),
                                    "width": sample.get("width"),
                                    "height": sample.get("height"),
                                    "messages": sample.get("messages"),
                                    "rollout_text": self._clip_text(
                                        parse.response_text, max_chars=dump_max_chars
                                    ),
                                    "prefix_text": self._clip_text(
                                        parse.prefix_text, max_chars=dump_max_chars
                                    ),
                                    "gt_objects": gt_objs_dump,
                                    "pred_objects": pred_objs_dump,
                                    "match": {
                                        "matched_pairs": list(match.matched_pairs),
                                        "fn_gt_indices": list(match.fn_gt_indices),
                                        "fp_pred_indices": list(match.fp_pred_indices),
                                        "gating_rejections": int(match.gating_rejections),
                                    },
                                    "stats": {
                                        "decode_mode": str(_decode_mode),
                                        "parse_invalid_rollout": bool(
                                            getattr(parse, "invalid_rollout", False)
                                        ),
                                        "parse_dropped_invalid": int(
                                            getattr(parse, "dropped_invalid", 0) or 0
                                        ),
                                        "parse_dropped_ambiguous": int(
                                            getattr(parse, "dropped_ambiguous", 0) or 0
                                        ),
                                        "parse_truncated": bool(
                                            getattr(parse, "truncated", False)
                                        ),
                                        "parse_dropped_invalid_by_reason": drop_by_reason,
                                        "valid_pred_objects": int(len(preds)),
                                        "gt_objects": int(len(gts)),
                                        "matched": int(len(match.matched_pairs)),
                                        "precision": float(prec_local),
                                        "recall": float(rec_local),
                                        "f1": float(f1_local),
                                    },
                                }
                            )

                    # Optional desc semantic monitor on matched pairs.
                    if desc_enabled and match.matched_pairs:
                        pairs = list(match.matched_pairs)
                        if desc_max_pairs > 0 and len(pairs) > desc_max_pairs:
                            pairs = pairs[:desc_max_pairs]

                        uniq: set[str] = set()
                        norm_pairs: List[Tuple[str, str, bool]] = []
                        for pred_i, gt_i in pairs:
                            if pred_i < 0 or pred_i >= len(pred_meta):
                                continue
                            if gt_i < 0 or gt_i >= len(gts):
                                continue
                            pred_desc_raw = str(
                                getattr(pred_meta[pred_i], "desc", "") or ""
                            )
                            gt_desc_raw = str(getattr(gts[gt_i], "desc", "") or "")
                            if normalize_desc is None:
                                p = pred_desc_raw.strip().lower()
                                g = gt_desc_raw.strip().lower()
                            else:
                                p = normalize_desc(pred_desc_raw)
                                g = normalize_desc(gt_desc_raw)
                            exact_ok = bool(p) and (p == g)
                            if exact_ok:
                                desc_exact_ok_total += 1.0
                            if p and g:
                                norm_pairs.append((p, g, bool(exact_ok)))
                                uniq.add(p)
                                uniq.add(g)

                        desc_pairs_total += float(len(norm_pairs))

                        if (
                            sem_loaded_local > 0.0
                            and sem_encoder is not None
                            and norm_pairs
                        ):
                            try:
                                emb = sem_encoder.encode_norm_texts(sorted(uniq))
                            except (TypeError, ValueError):
                                emb = {}
                                sem_encoder = None
                                sem_loaded_local = 0.0

                            if sem_loaded_local > 0.0 and sem_encoder is not None:
                                for p, g, exact_ok in norm_pairs:
                                    pv = emb.get(p)
                                    gv = emb.get(g)
                                    if pv is None or gv is None:
                                        ok = bool(exact_ok)
                                        sim = None
                                    else:
                                        sim = float(np.dot(pv, gv))
                                        ok = bool(exact_ok or sim >= desc_thr)
                                    if ok:
                                        desc_sem_ok_total += 1.0
                                    if sim is not None:
                                        desc_sem_sim_sum_total += float(sim)
                                        desc_sem_sim_count_total += 1.0

        self._write_eval_phase_trace(
            phase="before_finalize",
            global_step=int(gs),
            eval_index=int(eval_dump_index),
            payload={
                "n_steps": float(n_steps),
                "n_samples": float(n_samples),
                "eval_detection_records_local": int(len(eval_detection_records_local)),
                "eval_rollout_artifacts_local": int(len(eval_rollout_artifacts_local)),
                "dropped_invalid_total": float(dropped_invalid_total),
                "trunc_samples": float(trunc_samples),
                "vllm_decode_error_count_local": float(vllm_decode_error_count_local),
            },
        )

        metrics = finalize_rollout_aligned_evaluation(
            owner=self,
            logger=logger,
            metric_key_prefix=str(metric_key_prefix),
            eval_prompt_variant=eval_prompt_variant,
            eval_detection_enabled=eval_detection_enabled,
            eval_detection_use_confidence_postop=eval_detection_use_confidence_postop,
            eval_detection_score_mode=str(eval_detection_score_mode),
            eval_detection_score_version=int(eval_detection_score_version),
            eval_detection_score_source=str(eval_detection_score_source),
            eval_detection_cfg=eval_detection_cfg,
            desc_enabled=desc_enabled,
            n_samples=float(n_samples),
            gt_total=float(gt_total),
            pred_total=float(pred_total),
            matched_total=float(matched_total),
            fp_total=float(fp_total),
            fn_total=float(fn_total),
            gating_rejections_total=float(gating_rejections_total),
            dropped_invalid_total=float(dropped_invalid_total),
            dropped_ambiguous_total=float(dropped_ambiguous_total),
            trunc_samples=float(trunc_samples),
            matched_iou_sum=float(matched_iou_sum),
            matched_iou_count=float(matched_iou_count),
            n_samples_valid_pred=float(n_samples_valid_pred),
            n_samples_any_match=float(n_samples_any_match),
            n_steps=float(n_steps),
            desc_pairs_total=float(desc_pairs_total),
            desc_exact_ok_total=float(desc_exact_ok_total),
            desc_sem_ok_total=float(desc_sem_ok_total),
            desc_sem_sim_sum_total=float(desc_sem_sim_sum_total),
            desc_sem_sim_count_total=float(desc_sem_sim_count_total),
            sem_loaded_local=float(sem_loaded_local),
            vllm_decode_error_count_local=float(vllm_decode_error_count_local),
            runtime_local_s=float(time.perf_counter() - t0),
            eval_detection_records_local=list(eval_detection_records_local),
            eval_rollout_artifacts_local=list(eval_rollout_artifacts_local),
            do_dump=bool(do_dump),
            dump_fail_samples=list(dump_fail_samples),
            dump_other_samples=list(dump_other_samples),
            dump_max_samples=int(dump_max_samples),
            gs=int(gs),
            eval_rollout_backend=str(eval_rollout_backend),
            eval_vllm_mode=str(eval_vllm_mode),
            top_k=int(top_k),
            gate_thr=float(gate_thr),
            mask_res=int(mask_res),
            fp_cost=float(fp_cost),
            fn_cost=float(fn_cost),
            was_training=bool(was_training),
            metric_name_matches_key_fn=metric_name_matches_key,
            stage2_eval_metric_key_fn=stage2_eval_metric_key,
        )
        self._write_eval_phase_trace(
            phase="after_finalize",
            global_step=int(gs),
            eval_index=int(eval_dump_index),
            payload={
                "metric_count": int(len(metrics)) if isinstance(metrics, Mapping) else 0,
                "has_detection_map": bool(
                    isinstance(metrics, Mapping)
                    and f"{metric_key_prefix}/detection/mAP" in metrics
                ),
            },
        )
        return metrics

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool = False,
        ignore_keys: Optional[List[str]] = None,
    ):
        # Handle the case where inputs is a list of raw samples during evaluation.
        # Concrete Stage-2 trainers own the raw-sample-to-forward-batch conversion.
        if isinstance(inputs, list):
            inputs = self._prepare_batch_inputs(inputs)

        # Call the parent prediction_step with properly formatted inputs.
        return super().prediction_step(
            model=model,
            inputs=inputs,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
        )

    def _prepare_batch_inputs(
        self,
        inputs: List[Mapping[str, Any]],
        _segments_only: bool = False,
    ) -> Any:
        raise NotImplementedError(
            "Stage2RolloutRuntime is a shared runtime base; concrete Stage-2 "
            "trainers must implement raw-batch preparation."
        )

__all__ = ["Stage2RolloutRuntime"]
