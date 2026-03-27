from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable, Dict, List, Literal, Mapping, MutableMapping

import torch


LocalMetricMode = Literal["sum", "weighted_mean"]
DDPMetricMode = Literal["sum", "max", "weighted_mean", "world_mean"]

STAGE2_SNAPSHOT_PREFIX = "snapshot/"


@dataclass(frozen=True)
class MetricSpec:
    local_mode: LocalMetricMode
    ddp_mode: DDPMetricMode
    ddp_weight_key: str | None = None


@dataclass(frozen=True)
class Stage2DDPPhaseConfig:
    monitor_enabled: bool
    final_sync_timeout_s: float
    monitor_group_timeout_s: float


def _clamp_stage2_ddp_phase_timeout(timeout_s: float) -> float:
    return float(max(30.0, min(3600.0, float(timeout_s))))


def resolve_stage2_ab_ddp_phase_config(
    owner: Any,
    *,
    ddp_world_size: int,
) -> Stage2DDPPhaseConfig:
    ddp_phase_timeout_raw = owner._ab_channel_b_get("ddp_phase_timeout_s", None)
    if ddp_phase_timeout_raw is None:
        ddp_phase_monitor_enabled = True
        ddp_phase_timeout_s = 120.0
    else:
        try:
            ddp_phase_timeout_s = float(ddp_phase_timeout_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "stage2_ab.channel_b.ddp_phase_timeout_s must be a float/int when set"
            ) from exc

        if float(ddp_phase_timeout_s) <= 0.0:
            if int(ddp_world_size) > 1:
                raise ValueError(
                    "stage2_ab.channel_b.ddp_phase_timeout_s must be > 0 under DDP "
                    "(coordination barriers must be bounded to prevent deadlocks)."
                )
            ddp_phase_monitor_enabled = False
            ddp_phase_timeout_s = 0.0
        else:
            ddp_phase_monitor_enabled = True
            ddp_phase_timeout_s = _clamp_stage2_ddp_phase_timeout(
                float(ddp_phase_timeout_s)
            )

    return Stage2DDPPhaseConfig(
        monitor_enabled=bool(ddp_phase_monitor_enabled),
        final_sync_timeout_s=float(ddp_phase_timeout_s),
        monitor_group_timeout_s=float(ddp_phase_timeout_s),
    )


def resolve_stage2_prepare_barrier_timeout(
    *,
    final_sync_timeout_s: float,
    producer_wait_timeout_s: float,
) -> float:
    return float(
        max(
            float(final_sync_timeout_s),
            float(producer_wait_timeout_s),
        )
    )


def prime_stage2_ddp_monitor_group(
    owner: Any,
    *,
    dist: Any,
    rank: int,
    world_size: int,
    config: Stage2DDPPhaseConfig,
    logger: Any | None = None,
    warning_prefix: str = "stage2-ab DDP phase monitor disabled (gloo group init failed)",
) -> None:
    if (
        dist is None
        or (not dist.is_available())
        or (not dist.is_initialized())
        or int(world_size) <= 1
        or (not bool(config.monitor_enabled))
        or (not hasattr(dist, "monitored_barrier"))
    ):
        return

    group = getattr(owner, "_stage2_ab_ddp_monitor_group", None)
    if group is not None:
        return

    try:
        group = dist.new_group(
            backend="gloo",
            timeout=timedelta(
                seconds=_clamp_stage2_ddp_phase_timeout(
                    float(config.monitor_group_timeout_s)
                )
            ),
        )
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        warned = bool(getattr(owner, "_stage2_ab_ddp_monitor_group_warned", False))
        if int(rank) == 0 and not warned and logger is not None:
            logger.warning("%s: %r", str(warning_prefix), exc)
            setattr(owner, "_stage2_ab_ddp_monitor_group_warned", True)
        setattr(owner, "_stage2_ab_ddp_monitor_group", False)
        return

    setattr(owner, "_stage2_ab_ddp_monitor_group", group)


def run_stage2_ddp_monitored_barrier(
    owner: Any,
    *,
    dist: Any,
    phase: str,
    rank: int,
    world_size: int,
    timeout_s: float,
    config: Stage2DDPPhaseConfig,
    missing_barrier_label: str,
    timeout_error_label: str,
    timeout_error_hint: str,
) -> None:
    if int(world_size) <= 1:
        return

    if not bool(config.monitor_enabled):
        raise RuntimeError(
            "stage2-ab DDP phase monitor is disabled under DDP. "
            "Coordination barriers must be bounded to prevent deadlocks. "
            "Set stage2_ab.channel_b.ddp_phase_timeout_s to a positive value."
        )

    if not hasattr(dist, "monitored_barrier"):
        raise RuntimeError(
            f"{str(missing_barrier_label)} "
            f"(phase={str(phase)} rank={int(rank)}/{int(world_size)})."
        )

    group = getattr(owner, "_stage2_ab_ddp_monitor_group", None)
    if group is None:
        try:
            group = dist.new_group(
                backend="gloo",
                timeout=timedelta(
                    seconds=_clamp_stage2_ddp_phase_timeout(
                        float(config.monitor_group_timeout_s)
                    )
                ),
            )
        except Exception as exc:
            raise RuntimeError(
                "stage2-ab DDP monitored barrier requested but gloo group init failed; "
                f"phase={str(phase)} rank={int(rank)}/{int(world_size)} "
                f"timeout_s={float(config.monitor_group_timeout_s):.1f}."
            ) from exc
        setattr(owner, "_stage2_ab_ddp_monitor_group", group)

    if group is False:
        raise RuntimeError(
            "stage2-ab internal error: DDP monitor group is disabled under DDP; "
            "this is unsafe because unbounded barriers can deadlock"
        )

    local_timeout_s = _clamp_stage2_ddp_phase_timeout(float(timeout_s))
    try:
        try:
            dist.monitored_barrier(
                group=group,
                timeout=timedelta(seconds=float(local_timeout_s)),
                wait_all_ranks=True,
            )
        except TypeError:
            dist.monitored_barrier(
                group=group,
                timeout=timedelta(seconds=float(local_timeout_s)),
            )
    except Exception as exc:
        raise RuntimeError(
            f"{str(timeout_error_label)}; "
            f"phase={str(phase)} rank={int(rank)}/{int(world_size)} "
            f"timeout_s={float(local_timeout_s):.1f}. "
            f"{str(timeout_error_hint)}"
        ) from exc


def _resolve_stage2_collective_monitor_config(
    owner: Any,
    *,
    world_size: int,
) -> Stage2DDPPhaseConfig:
    get_ab_cfg = getattr(owner, "_ab_channel_b_get", None)
    if callable(get_ab_cfg):
        return resolve_stage2_ab_ddp_phase_config(owner, ddp_world_size=int(world_size))
    return Stage2DDPPhaseConfig(
        monitor_enabled=bool(int(world_size) > 1),
        final_sync_timeout_s=120.0,
        monitor_group_timeout_s=120.0,
    )


def _run_stage2_collective_entry_barrier(
    owner: Any,
    *,
    dist: Any,
    rank: int,
    world_size: int,
    where: str,
) -> None:
    if dist is None or int(world_size) <= 1:
        return
    if not (hasattr(dist, "monitored_barrier") and hasattr(dist, "new_group")):
        return

    config = _resolve_stage2_collective_monitor_config(owner, world_size=int(world_size))
    if not bool(config.monitor_enabled):
        return

    group = getattr(owner, "_stage2_shared_collective_monitor_group", None)
    if group is None:
        try:
            group = dist.new_group(
                backend="gloo",
                timeout=timedelta(
                    seconds=_clamp_stage2_ddp_phase_timeout(
                        float(config.monitor_group_timeout_s)
                    )
                ),
            )
        except Exception:
            return
        setattr(owner, "_stage2_shared_collective_monitor_group", group)

    if group is False:
        return

    timeout_s = _clamp_stage2_ddp_phase_timeout(float(config.final_sync_timeout_s))
    try:
        try:
            dist.monitored_barrier(
                group=group,
                timeout=timedelta(seconds=float(timeout_s)),
                wait_all_ranks=True,
            )
        except TypeError:
            dist.monitored_barrier(
                group=group,
                timeout=timedelta(seconds=float(timeout_s)),
            )
    except Exception as exc:
        raise RuntimeError(
            "stage2 shared collective entry barrier timed out; "
            f"where={str(where)} rank={int(rank)}/{int(world_size)} "
            f"timeout_s={float(timeout_s):.1f}. "
            "At least one rank never entered the shared metric/readiness collective."
        ) from exc


def stage2_snapshot_metric_key(metric_key: str) -> str | None:
    key = str(metric_key)

    if (
        key.startswith(("loss/text/", "loss/coord/"))
        or (key.startswith("coord_diag/") and not key.startswith("coord_diag/B/"))
        or key.startswith("time/channel_a_")
        or key == "stage2/channel_a"
    ):
        return f"{STAGE2_SNAPSHOT_PREFIX}{key}"

    if (
        key.startswith("loss/B_")
        or key.startswith("coord_diag/B/")
        or key.startswith("stage2_ab/channel_b/")
        or key.startswith("dup/")
        or key.startswith("train/triage/")
        or key.startswith("diag/duplicate_burst/")
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
        return f"{STAGE2_SNAPSHOT_PREFIX}{key}"

    return None


def stage2_snapshot_source_key(snapshot_key: str) -> str:
    key = str(snapshot_key)
    if key.startswith(STAGE2_SNAPSHOT_PREFIX):
        return key[len(STAGE2_SNAPSHOT_PREFIX) :]
    return key


def merge_stage2_metric_snapshots(
    snapshots: MutableMapping[str, float],
    metrics: Mapping[str, Any],
) -> Dict[str, float]:
    for key_raw, value in metrics.items():
        snapshot_key = stage2_snapshot_metric_key(str(key_raw))
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


def build_stage2_snapshot_logs(
    snapshots: Mapping[str, float],
    current_metrics: Mapping[str, Any] | None = None,
) -> Dict[str, float]:
    current = {str(key) for key in (current_metrics or {}).keys()}
    return {
        str(key): float(value)
        for key, value in sorted(snapshots.items(), key=lambda item: item[0])
        if stage2_snapshot_source_key(str(key)) not in current
    }


def resolve_stage2_ab_metric_spec(key: str) -> MetricSpec:
    key = str(key)
    gradmon_weight_key = "gradmon/_log_weight_total"
    stage2_weight_key = "stage2/_log_weight_total"

    if key.startswith("time/"):
        return MetricSpec(local_mode="sum", ddp_mode="max")

    if key in {
        "rollout/backend_hf",
        "rollout/backend_vllm",
        "rollout/decode_mode_greedy",
        "rollout/decode_mode_beam",
        "rollout/hf_seeded_global",
        "rollout/do_sample",
    }:
        return MetricSpec(local_mode="weighted_mean", ddp_mode="max")

    if key in {stage2_weight_key, gradmon_weight_key}:
        return MetricSpec(local_mode="sum", ddp_mode="sum")

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
        return MetricSpec(local_mode="sum", ddp_mode="sum")

    if key.startswith("stage2_ab/channel_b/strict_drop/reason/"):
        return MetricSpec(local_mode="sum", ddp_mode="sum")

    if key.startswith("stage2_ab/") and "/N_" in key:
        return MetricSpec(local_mode="sum", ddp_mode="sum")

    if key in {"dup/max_desc_count", "dup/saturation_rate"}:
        return MetricSpec(
            local_mode="weighted_mean",
            ddp_mode="weighted_mean",
            ddp_weight_key=stage2_weight_key,
        )

    if key.endswith(("_total", "_count", "_sum", "_num", "_den")):
        return MetricSpec(local_mode="sum", ddp_mode="sum")

    if key.startswith("gradmon/"):
        return MetricSpec(
            local_mode="weighted_mean",
            ddp_mode="weighted_mean",
            ddp_weight_key=gradmon_weight_key,
        )

    if (
        key.startswith("loss/")
        or key.startswith("train/optimization/")
        or key in {"dup/max_desc_count", "dup/saturation_rate"}
        or key.startswith("stage2/channel_")
        or key.startswith("rollout/")
        or key.startswith("coord_diag/")
        or key == "stage2_ab/b_ratio_realized"
        or (key.startswith("stage2_ab/") and "/is_" in key)
    ):
        return MetricSpec(
            local_mode="weighted_mean",
            ddp_mode="weighted_mean",
            ddp_weight_key=stage2_weight_key,
        )

    return MetricSpec(local_mode="sum", ddp_mode="sum")


def resolve_rollout_log_metric_spec(key: str) -> MetricSpec:
    key = str(key)
    gradmon_weight_key = "gradmon/_log_weight_total"

    if key.startswith("time/"):
        return MetricSpec(local_mode="sum", ddp_mode="max")

    if key in {
        "rollout/backend_hf",
        "rollout/backend_vllm",
        "rollout/decode_mode_greedy",
        "rollout/decode_mode_beam",
        "rollout/hf_seeded_global",
        "rollout/do_sample",
        "rollout/desc_sem_enabled",
        "rollout/gen_new_tokens_p90",
        "rollout/gen_new_tokens_p99",
    }:
        return MetricSpec(local_mode="sum", ddp_mode="max")

    if key == gradmon_weight_key:
        return MetricSpec(local_mode="sum", ddp_mode="sum")

    if key in {
        "train/samples_total",
        "rollout/parse_truncated",
        "rollout/parse_dropped_invalid",
        "rollout/parse_dropped_ambiguous",
        "rollout/valid_pred_objects_total",
        "rollout/gt_objects_total",
        "rollout/matched_for_supervision",
        "rollout/excluded_from_supervision",
        "rollout/fp_total",
        "rollout/fn_total",
        "rollout/gating_rejections",
        "rollout/fn_appended_total",
        "rollout/prefix_coord_targets_total",
        "rollout/matched_maskiou_count",
        "rollout/desc_pairs_total",
        "rollout/desc_sem_sim_sum",
        "rollout/desc_sem_sim_count",
        "rollout/_matched_maskiou_sum",
        "rollout/_sample_valid_pred_num",
        "rollout/_sample_any_match_num",
        "rollout/_desc_exact_ok",
        "rollout/_desc_sem_ok",
        "rollout/decode_non_beam_count",
        "rollout/decode_beam_count",
        "rollout/gen_new_tokens_total",
        "rollout/_parse_truncated_num",
        "rollout/_parse_truncated_den",
    } or key.endswith("_total"):
        return MetricSpec(local_mode="sum", ddp_mode="sum")

    if key.startswith("gradmon/"):
        return MetricSpec(
            local_mode="weighted_mean",
            ddp_mode="weighted_mean",
            ddp_weight_key=gradmon_weight_key,
        )

    return MetricSpec(local_mode="weighted_mean", ddp_mode="world_mean")


def ddp_assert_all_ranks_true_or_raise(
    owner: Any,
    *,
    where: str,
    local_true: bool,
    global_step: int,
) -> None:
    rank, world_size, dist = _resolve_distributed_context(owner)
    if dist is None:
        return
    if int(world_size) <= 1:
        return

    _run_stage2_collective_entry_barrier(
        owner,
        dist=dist,
        rank=int(rank),
        world_size=int(world_size),
        where=str(where),
    )

    backend = ""
    try:
        backend = str(dist.get_backend()).lower()
    except Exception:
        backend = ""

    reduce_device = torch.device("cpu")
    if backend == "nccl" and torch.cuda.is_available():
        try:
            model = getattr(owner, "model", None)
            if model is not None and hasattr(model, "device"):
                reduce_device = model.device
            elif model is not None:
                reduce_device = next(model.parameters()).device
        except Exception:
            reduce_device = torch.device("cpu")

    flag = torch.tensor(
        [1 if bool(local_true) else 0],
        device=reduce_device,
        dtype=torch.int32,
    )
    dist.all_reduce(flag, op=dist.ReduceOp.SUM)
    ready_sum = int(flag.item())

    if int(ready_sum) != int(world_size):
        raise RuntimeError(
            f"{where}: pending-metrics readiness mismatch under DDP "
            f"(global_step={int(global_step)} rank={int(rank)}/{int(world_size)}, "
            f"local_ready={int(bool(local_true))} ready_sum={int(ready_sum)})."
        )


def _resolve_distributed_context(owner: Any) -> tuple[int, int, Any]:
    dist_info = getattr(owner, "_dist_info", None)
    if callable(dist_info):
        try:
            rank, world_size, dist = dist_info()
            return int(rank), int(world_size), dist
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass

    try:
        import torch.distributed as dist
    except (ImportError, TypeError, ValueError):
        return 0, 1, None

    if not (dist.is_available() and dist.is_initialized()):
        return 0, 1, None

    try:
        world_size = int(dist.get_world_size())
    except (TypeError, ValueError, RuntimeError):
        world_size = 1
    try:
        rank = int(dist.get_rank())
    except (TypeError, ValueError, RuntimeError):
        rank = 0
    return int(rank), int(world_size), dist


def _append_metric_key(keys: List[str], seen: set[str], key: str) -> None:
    key = str(key)
    if key in seen:
        return
    seen.add(key)
    keys.append(key)


def reduce_metric_payload_global(
    owner: Any,
    metrics: Mapping[str, Any],
    *,
    resolver: Callable[[str], MetricSpec],
    error_prefix: str,
) -> Dict[str, float]:
    reduced: Dict[str, float] = {}
    for key_raw, value in metrics.items():
        try:
            reduced[str(key_raw)] = float(value)
        except (TypeError, ValueError):
            continue

    rank, world_size, dist = _resolve_distributed_context(owner)

    metric_keys: List[str] = [str(key) for key in reduced.keys()]
    if dist is not None and int(world_size) > 1:
        _run_stage2_collective_entry_barrier(
            owner,
            dist=dist,
            rank=int(rank),
            world_size=int(world_size),
            where=f"{str(error_prefix)} key sync",
        )
        gathered_keys: List[Any] = [None] * int(world_size)
        dist.all_gather_object(gathered_keys, metric_keys)

        merged_keys = list(metric_keys)
        merged_key_set = set(metric_keys)
        for item in gathered_keys:
            if not isinstance(item, (list, tuple)):
                raise RuntimeError(
                    f"{error_prefix} key sync produced non-list keys "
                    f"(rank={int(rank)}/{int(world_size)} type={type(item).__name__})"
                )
            for key_raw in item:
                key = str(key_raw)
                if key not in merged_key_set:
                    merged_key_set.add(key)
                    merged_keys.append(key)
                reduced.setdefault(key, 0.0)
        metric_keys = merged_keys

    sum_keys: List[str] = []
    sum_seen: set[str] = set()
    max_keys: List[str] = []
    max_seen: set[str] = set()
    world_mean_keys: List[str] = []
    world_mean_seen: set[str] = set()
    weighted_groups: Dict[str, List[str]] = {}
    weighted_group_seen: Dict[str, set[str]] = {}

    for key in metric_keys:
        spec = resolver(str(key))
        if spec.ddp_mode == "sum":
            _append_metric_key(sum_keys, sum_seen, key)
            continue
        if spec.ddp_mode == "max":
            _append_metric_key(max_keys, max_seen, key)
            continue
        if spec.ddp_mode == "world_mean":
            _append_metric_key(world_mean_keys, world_mean_seen, key)
            continue
        if spec.ddp_mode == "weighted_mean":
            if not spec.ddp_weight_key:
                raise RuntimeError(
                    f"{error_prefix} metric {key!r} is missing a DDP weight key"
                )
            weight_key = str(spec.ddp_weight_key)
            weighted_groups.setdefault(weight_key, [])
            weighted_group_seen.setdefault(weight_key, set())
            _append_metric_key(
                weighted_groups[weight_key], weighted_group_seen[weight_key], key
            )
            _append_metric_key(sum_keys, sum_seen, weight_key)
            continue
        raise RuntimeError(f"{error_prefix} metric {key!r} has unsupported DDP mode")

    if dist is None or int(world_size) <= 1:
        return reduced

    try:
        _run_stage2_collective_entry_barrier(
            owner,
            dist=dist,
            rank=int(rank),
            world_size=int(world_size),
            where=f"{str(error_prefix)} all-reduce",
        )
        device = torch.device("cpu")
        try:
            model = getattr(owner, "model", None)
            if model is not None and hasattr(model, "device"):
                device = model.device
            elif model is not None:
                device = next(model.parameters()).device
        except (AttributeError, RuntimeError, StopIteration, TypeError, ValueError):
            device = torch.device("cpu")

        for weight_key, keys in weighted_groups.items():
            local_weight = float(reduced.get(weight_key, 0.0))
            for key in keys:
                reduced[key] = float(reduced.get(key, 0.0)) * float(local_weight)

        sum_like_keys: List[str] = []
        sum_like_seen: set[str] = set()
        for key in sum_keys:
            _append_metric_key(sum_like_keys, sum_like_seen, key)
        for key in world_mean_keys:
            _append_metric_key(sum_like_keys, sum_like_seen, key)
        for keys in weighted_groups.values():
            for key in keys:
                _append_metric_key(sum_like_keys, sum_like_seen, key)
        max_reduce_keys = list(max_keys)

        def _all_reduce(keys: List[str], op: Any) -> None:
            if not keys:
                return
            values = torch.tensor(
                [float(reduced.get(key, 0.0)) for key in keys],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(values, op=op)
            for index, key in enumerate(keys):
                reduced[key] = float(values[index].item())

        _all_reduce(sum_like_keys, dist.ReduceOp.SUM)
        _all_reduce(max_reduce_keys, dist.ReduceOp.MAX)

        scale = float(world_size)
        if scale > 0.0:
            for key in world_mean_keys:
                reduced[key] = float(reduced.get(key, 0.0) / scale)

        for weight_key, keys in weighted_groups.items():
            global_weight = float(reduced.get(weight_key, 0.0))
            for key in keys:
                reduced[key] = (
                    float(reduced.get(key, 0.0) / global_weight)
                    if global_weight > 0.0
                    else 0.0
                )
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{error_prefix} all-reduce failed (DDP is initialized); "
            f"rank={int(rank)}/{int(world_size)}"
        ) from exc

    return reduced


__all__ = [
    "MetricSpec",
    "Stage2DDPPhaseConfig",
    "STAGE2_SNAPSHOT_PREFIX",
    "build_stage2_snapshot_logs",
    "ddp_assert_all_ranks_true_or_raise",
    "merge_stage2_metric_snapshots",
    "prime_stage2_ddp_monitor_group",
    "reduce_metric_payload_global",
    "resolve_stage2_ab_ddp_phase_config",
    "resolve_stage2_prepare_barrier_timeout",
    "resolve_rollout_log_metric_spec",
    "resolve_stage2_ab_metric_spec",
    "run_stage2_ddp_monitored_barrier",
    "stage2_snapshot_metric_key",
    "stage2_snapshot_source_key",
]
