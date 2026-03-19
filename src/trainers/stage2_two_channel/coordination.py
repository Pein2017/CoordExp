from datetime import timedelta
from typing import Any, Dict, Mapping, Tuple


def run_stage2_ab_ddp_monitored_barrier(
    *,
    owner: Any,
    dist: Any,
    phase: str,
    rank: int,
    world_size: int,
    timeout_s: float,
    monitor_group_timeout_s: float,
) -> None:
    if int(world_size) <= 1:
        return

    if not hasattr(dist, "monitored_barrier"):
        raise RuntimeError(
            "torch.distributed.monitored_barrier is required for bounded stage2-ab DDP barriers "
            f"(phase={str(phase)} rank={int(rank)}/{int(world_size)})."
        )

    group = getattr(owner, "_stage2_ab_ddp_monitor_group", None)
    if group is None:
        try:
            group = dist.new_group(
                backend="gloo",
                timeout=timedelta(seconds=float(monitor_group_timeout_s)),
            )
        except Exception as exc:
            raise RuntimeError(
                "stage2-ab monitored barrier requested but gloo group init failed; "
                f"phase={str(phase)} rank={int(rank)}/{int(world_size)} "
                f"timeout_s={float(monitor_group_timeout_s):.1f}."
            ) from exc
        setattr(owner, "_stage2_ab_ddp_monitor_group", group)

    if group is False:
        raise RuntimeError(
            "stage2-ab internal error: DDP monitor group is disabled under DDP; "
            "this is unsafe because unbounded barriers can deadlock"
        )

    local_timeout_s = float(max(30.0, min(3600.0, float(timeout_s))))

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
            "stage2-ab DDP barrier timed out; "
            f"phase={str(phase)} rank={int(rank)}/{int(world_size)} "
            f"timeout_s={float(local_timeout_s):.1f}. "
            "This indicates cross-rank stage skew or a deadlock."
        ) from exc


def resolve_channel_b_timeouts(
    *,
    owner: Any,
    ddp_world_size: int,
) -> Tuple[float, bool, float, float]:
    wait_timeout_cfg = owner._ab_channel_b_get("producer_wait_timeout_s", None)
    if wait_timeout_cfg is None:
        producer_wait_timeout_s = 0.0
    else:
        try:
            producer_wait_timeout_s = float(wait_timeout_cfg)
        except Exception as exc:
            raise ValueError(
                "stage2_ab.channel_b.producer_wait_timeout_s must be a float/int when set"
            ) from exc
    if producer_wait_timeout_s <= 0.0:
        try:
            conn_timeout_s, infer_timeout_s = owner._vllm_server_timeouts()  # type: ignore[attr-defined]
            base_timeout = (
                float(infer_timeout_s)
                if infer_timeout_s is not None
                else float(conn_timeout_s)
            )
            producer_wait_timeout_s = max(120.0, float(base_timeout) * 2.0)
        except Exception:
            producer_wait_timeout_s = 300.0

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
            ddp_phase_timeout_s = float(max(30.0, min(3600.0, ddp_phase_timeout_s)))

    ddp_phase_final_sync_timeout_s = float(ddp_phase_timeout_s)
    ddp_monitor_group_timeout_s = float(ddp_phase_timeout_s)
    return (
        float(producer_wait_timeout_s),
        bool(ddp_phase_monitor_enabled),
        float(ddp_phase_final_sync_timeout_s),
        float(ddp_monitor_group_timeout_s),
    )


def split_rollout_metrics(
    metrics: Mapping[str, Any],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    rollout_static: Dict[str, float] = {}
    step_totals: Dict[str, float] = {}
    for k, v in metrics.items():
        ks = str(k)
        try:
            fv = float(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if ks.startswith("rollout/"):
            rollout_static[ks] = float(fv)
        else:
            step_totals[ks] = float(fv)
    return rollout_static, step_totals


def accumulate_channel_b_producer_item(
    *,
    segs: list[tuple[dict[str, Any], dict[str, Any], int]],
    metrics: Mapping[str, Any],
    raw_n: int,
    rollout_static: Dict[str, float],
    pending_totals: Dict[str, float],
    seen_segments: int,
    seen_raw: int,
    buf_total_len: int,
) -> Tuple[int, int, int]:
    seen_segments += int(len(segs))
    seen_raw += int(raw_n)
    buf_total_len += int(sum(int(sl) for _, _, sl in segs))

    r_static, step_tot = split_rollout_metrics(metrics)
    if not rollout_static:
        rollout_static.update(r_static)
    else:
        for k, v in r_static.items():
            rollout_static.setdefault(k, float(v))

    for k, v in step_tot.items():
        pending_totals[str(k)] = float(pending_totals.get(str(k), 0.0)) + float(v)

    return int(seen_segments), int(seen_raw), int(buf_total_len)


__all__ = [
    "accumulate_channel_b_producer_item",
    "run_stage2_ab_ddp_monitored_barrier",
    "resolve_channel_b_timeouts",
    "split_rollout_metrics",
]
