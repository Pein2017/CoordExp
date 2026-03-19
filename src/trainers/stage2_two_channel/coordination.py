from datetime import timedelta
from typing import Any


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


__all__ = ["run_stage2_ab_ddp_monitored_barrier"]
