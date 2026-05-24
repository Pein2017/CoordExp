import contextlib
import queue
import threading
import time
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch

from ..stage2_coordination import (
    Stage2DDPPhaseConfig,
    resolve_stage2_ab_ddp_phase_config,
    run_stage2_ddp_monitored_barrier,
)
from .pack_schedule import Stage2PackSchedule, Stage2PackSlot


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
    run_stage2_ddp_monitored_barrier(
        owner,
        dist=dist,
        phase=phase,
        rank=rank,
        world_size=world_size,
        timeout_s=timeout_s,
        config=Stage2DDPPhaseConfig(
            monitor_enabled=True,
            final_sync_timeout_s=float(timeout_s),
            monitor_group_timeout_s=float(monitor_group_timeout_s),
        ),
        missing_barrier_label=(
            "torch.distributed.monitored_barrier is required for bounded stage2-ab DDP barriers"
        ),
        timeout_error_label="stage2-ab DDP barrier timed out",
        timeout_error_hint="This indicates cross-rank stage skew or a deadlock.",
    )


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

    phase_config = resolve_stage2_ab_ddp_phase_config(
        owner,
        ddp_world_size=int(ddp_world_size),
    )
    return (
        float(producer_wait_timeout_s),
        bool(phase_config.monitor_enabled),
        float(phase_config.final_sync_timeout_s),
        float(phase_config.monitor_group_timeout_s),
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


def consume_channel_b_queue_item(
    *,
    owner: Any,
    item: Any,
    rollout_static: Dict[str, float],
    pending_totals: Dict[str, float],
    seen_segments: int,
    seen_raw: int,
    buf_total_len: int,
) -> Tuple[int, int, int]:
    segs, metrics, raw_n = item
    if not isinstance(segs, list):
        raise TypeError("producer returned non-list segments")
    if not isinstance(metrics, Mapping):
        metrics = {}
    raw_n = int(raw_n)

    owner._stage2_append_post_rollout_segments(channel="B", segments=segs)
    return accumulate_channel_b_producer_item(
        segs=segs,
        metrics=metrics,
        raw_n=int(raw_n),
        rollout_static=rollout_static,
        pending_totals=pending_totals,
        seen_segments=int(seen_segments),
        seen_raw=int(seen_raw),
        buf_total_len=int(buf_total_len),
    )


def prepare_channel_b_pipeline_pack_step(
    *,
    owner: Any,
    selected: Sequence[tuple[dict[str, Any], dict[str, Any], int]],
    pending_totals: Dict[str, float],
    seen_raw: int,
    total_segments_target: int,
    ddp_phase_final_sync_timeout_s: float,
    ddp_phase_barrier_fn: Any,
) -> Tuple[Dict[str, float], bool]:
    step_totals_pack = dict(pending_totals)
    is_last_pack = (int(seen_raw) >= int(total_segments_target)) and (
        not bool(owner._stage2_post_rollout_buffer(channel="B"))
    )
    if bool(is_last_pack):
        ddp_phase_barrier_fn(
            "channel_b_pipeline_before_final_sync_backward",
            timeout_s=float(ddp_phase_final_sync_timeout_s),
        )
    return step_totals_pack, bool(is_last_pack)


def finalize_channel_b_pipeline_step(
    *,
    thread_obj: Any,
    owner: Any,
    target_log_step: int,
    producer_exc: Sequence[Exception],
    total_segments_target: int,
    seen_raw: int,
    seen_segments: int,
    loss_total: Any,
) -> Any:
    thread_obj.join(timeout=5.0)
    if thread_obj.is_alive():
        raise RuntimeError(
            "stage2-ab Channel-B producer thread did not terminate cleanly after pipeline step"
        )

    owner._stage2_flush_train_monitor_dump(global_step=target_log_step)

    if producer_exc:
        raise producer_exc[0]

    if int(total_segments_target) > 0 and int(seen_raw) != int(total_segments_target):
        raise ValueError(
            "stage2-ab Channel-B pipeline produced unexpected raw-rollout count: "
            f"seen_raw={int(seen_raw)} target={int(total_segments_target)}"
        )

    # `total_segments_target` is the raw-rollout budget. Residual-trie and
    # self-prefix objectives may legitimately emit multiple post-rollout
    # teacher-forcing segments per raw rollout, so `seen_segments` is
    # telemetry, not a cardinality guard.

    if loss_total is None:
        raise AssertionError("stage2-ab Channel-B pipelined step produced no packs")
    return loss_total


def run_channel_b_train_one_pack(
    *,
    owner: Any,
    model: Any,
    selected: Sequence[tuple[dict[str, Any], dict[str, Any], int]],
    pack_metrics: Mapping[str, float],
    rollout_static: Mapping[str, float],
    step_totals: Mapping[str, float],
    total_segments_target: int,
    sync_gradients: bool,
    dist: Any,
    ddp_rank: int,
    ddp_world_size: int,
    shadow_zero_loss: bool = False,
) -> torch.Tensor:
    from swift.llm import to_device

    with owner._template_packing_enabled():
        packed = owner.template.data_collator([enc for enc, _, _ in selected])
    batch = to_device(packed, owner.model.device)
    owner._assert_single_packed_forward(batch, where="stage2_ab/packed_forward")
    batch["_rollout_matching_meta"] = [m for _, m, _ in selected]

    bm: Dict[str, float] = {}
    bm.update({str(k): float(v) for k, v in rollout_static.items()})
    bm.update({str(k): float(v) for k, v in step_totals.items()})
    bm.update({str(k): float(v) for k, v in pack_metrics.items()})

    owner._merge_rollout_matching_batch_metrics(batch, bm)
    batch["_stage2_ab_channel"] = "B"
    if bool(shadow_zero_loss):
        batch["_stage2_ab_shadow_pack"] = True

    pack_segments = int(len(selected))
    weight = (
        0.0
        if bool(shadow_zero_loss)
        else float(pack_segments) / float(total_segments_target)
    )

    cm = contextlib.nullcontext()
    if not bool(sync_gradients):
        acc = getattr(owner, "accelerator", None)
        if acc is not None and hasattr(acc, "no_sync"):
            cm = acc.no_sync(model)
        else:
            no_sync = getattr(model, "no_sync", None)
            if callable(no_sync):
                cm = model.no_sync()

    with cm:
        loss_cm = getattr(owner, "compute_loss_context_manager", None)
        loss_ctx = loss_cm() if callable(loss_cm) else contextlib.nullcontext()
        prev_gradmon_sync = getattr(owner, "_loss_gradient_monitor_sync_gradients", None)
        setattr(owner, "_loss_gradient_monitor_sync_gradients", bool(sync_gradients))
        try:
            with loss_ctx:
                with owner._stage2_ab_disable_average_tokens_across_devices_for_packed_step(
                    dist=dist,
                    ddp_rank=int(ddp_rank),
                    ddp_world_size=int(ddp_world_size),
                    where=f"stage2_ab/channel_{str(batch.get('_stage2_ab_channel', '?'))}/train_one_pack",
                ):
                    loss = owner.compute_loss(model, batch)
        finally:
            if prev_gradmon_sync is None:
                try:
                    delattr(owner, "_loss_gradient_monitor_sync_gradients")
                except AttributeError:
                    pass
            else:
                setattr(owner, "_loss_gradient_monitor_sync_gradients", prev_gradmon_sync)
        if not isinstance(loss, torch.Tensor):
            raise TypeError("compute_loss must return a torch.Tensor")

        loss_scaled = loss * float(weight)

        acc = getattr(owner, "accelerator", None)
        if acc is not None and hasattr(acc, "backward"):
            acc.backward(loss_scaled)
        else:
            loss_scaled.backward()

    return loss.detach() * float(weight)


def _channel_b_pack_count_gather_device(*, owner: Any, model: Any, dist: Any) -> torch.device:
    backend = None
    get_backend = getattr(dist, "get_backend", None)
    if callable(get_backend):
        try:
            backend = str(get_backend()).lower()
        except Exception:
            backend = None
    if backend == "gloo":
        return torch.device("cpu")
    if backend == "nccl":
        for candidate in (
            getattr(getattr(owner, "model", None), "device", None),
            getattr(model, "device", None),
        ):
            if candidate is None:
                continue
            try:
                device = torch.device(candidate)
            except Exception:
                continue
            if device.type == "cuda":
                return device
        parameters = getattr(model, "parameters", None)
        if callable(parameters):
            try:
                first_param = next(parameters())
                device = torch.device(first_param.device)
                if device.type == "cuda":
                    return device
            except Exception:
                pass
        if torch.cuda.is_available():
            try:
                return torch.device("cuda", torch.cuda.current_device())
            except Exception:
                pass
        raise RuntimeError(
            "stage2-ab DDP pack-count gather with NCCL requires a CUDA device"
        )

    for candidate in (
        getattr(getattr(owner, "model", None), "device", None),
        getattr(model, "device", None),
    ):
        if candidate is None:
            continue
        try:
            return torch.device(candidate)
        except Exception:
            continue

    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            first_param = next(parameters())
            return torch.device(first_param.device)
        except Exception:
            pass
    return torch.device("cpu")


def gather_channel_b_local_pack_counts(
    *,
    owner: Any,
    model: Any,
    local_pack_count: int,
    dist: Any,
    ddp_world_size: int,
) -> list[int]:
    """Gather local post-rollout pack counts so every rank runs the same slot plan."""
    local_pack_count = int(local_pack_count)
    if int(ddp_world_size) <= 1:
        return [local_pack_count]
    if dist is None:
        return [local_pack_count]
    is_available = getattr(dist, "is_available", None)
    if callable(is_available) and not bool(is_available()):
        return [local_pack_count]
    is_initialized = getattr(dist, "is_initialized", None)
    if callable(is_initialized) and not bool(is_initialized()):
        return [local_pack_count]
    all_gather_object = getattr(dist, "all_gather_object", None)
    if callable(all_gather_object):
        gathered_obj: list[Any] = [None] * int(ddp_world_size)
        all_gather_object(gathered_obj, int(local_pack_count))
        counts = [int(item) for item in gathered_obj]
        if len(counts) != int(ddp_world_size):
            raise RuntimeError(
                "stage2-ab Channel-B DDP pack scheduling gathered unexpected world size: "
                f"counts={len(counts)} world_size={int(ddp_world_size)}"
            )
        return counts

    all_gather = getattr(dist, "all_gather", None)
    if not callable(all_gather):
        raise RuntimeError(
            "stage2-ab Channel-B DDP pack scheduling requires torch.distributed.all_gather_object or all_gather"
        )

    device = _channel_b_pack_count_gather_device(owner=owner, model=model, dist=dist)
    local = torch.tensor([local_pack_count], dtype=torch.long, device=device)
    gathered = [torch.zeros_like(local) for _ in range(int(ddp_world_size))]
    all_gather(gathered, local)
    counts = [int(item.item()) for item in gathered]
    if len(counts) != int(ddp_world_size):
        raise RuntimeError(
            "stage2-ab Channel-B DDP pack scheduling gathered unexpected world size: "
            f"counts={len(counts)} world_size={int(ddp_world_size)}"
        )
    return counts


def build_channel_b_pack_schedule(
    *,
    owner: Any,
    model: Any,
    local_pack_count: int,
    dist: Any,
    ddp_world_size: int,
) -> Stage2PackSchedule:
    pack_counts = gather_channel_b_local_pack_counts(
        owner=owner,
        model=model,
        local_pack_count=int(local_pack_count),
        dist=dist,
        ddp_world_size=int(ddp_world_size),
    )
    return Stage2PackSchedule.from_rank_pack_counts(
        local_pack_count=int(local_pack_count),
        rank_pack_counts=pack_counts,
    )


def run_channel_b_nonpipeline_learning_loop(
    *,
    owner: Any,
    model: Any,
    segments: list[tuple[dict[str, Any], dict[str, Any], int]],
    batch_metrics: Mapping[str, Any],
    target_log_step: int,
    total_segments_target: int,
    ddp_phase_prepare_timeout_s: float,
    ddp_phase_final_sync_timeout_s: float,
    ddp_phase_barrier_fn: Any,
    dist: Any,
    ddp_rank: int,
    ddp_world_size: int,
) -> torch.Tensor:
    trace_fn = getattr(owner, "_stage2_record_ddp_phase_trace", None)

    def _trace(phase: str, *, extra: Mapping[str, Any] | None = None) -> None:
        if not callable(trace_fn):
            return
        trace_payload = {
            "segment_count": int(len(segments)) if isinstance(segments, list) else 0,
            "total_segments_target": int(total_segments_target),
            "channel_b_buffer_size": int(
                len(owner._stage2_post_rollout_buffer(channel="B"))
            ),
        }
        if isinstance(extra, Mapping):
            trace_payload.update(dict(extra))
        trace_fn(
            global_step=int(target_log_step),
            phase=str(phase),
            rank=int(ddp_rank),
            world_size=int(ddp_world_size),
            payload=trace_payload,
        )

    _trace("channel_b_non_pipeline_before_flush")
    owner._stage2_flush_train_monitor_dump(global_step=target_log_step)
    if not isinstance(segments, list):
        raise ValueError(
            "stage2-ab Channel-B step mode expected post-rollout segments as a list; "
            f"got {type(segments).__name__}"
        )

    batch_metrics = dict(batch_metrics) if isinstance(batch_metrics, Mapping) else {}
    batch_metrics["stage2_ab/channel_b/train_monitor_dump_written"] = float(
        1.0
        if int(getattr(owner, "_stage2_train_monitor_dump_written_step", -1) or -1)
        == int(target_log_step)
        else 0.0
    )
    _trace(
        "channel_b_non_pipeline_after_flush",
        extra={
            "train_monitor_dump_written": float(
                batch_metrics["stage2_ab/channel_b/train_monitor_dump_written"]
            )
        },
    )
    rollout_static, step_totals = split_rollout_metrics(batch_metrics)
    step_totals["stage2/raw_rollouts"] = float(total_segments_target)

    owner._stage2_append_post_rollout_segments(channel="B", segments=segments)
    _trace("channel_b_non_pipeline_after_append")
    local_packs: list[
        tuple[Sequence[tuple[dict[str, Any], dict[str, Any], int]], dict[str, float]]
    ] = []
    while owner._stage2_post_rollout_buffer(channel="B"):
        t_pack0 = time.perf_counter()
        selected, pack_metrics = owner._stage2_pop_post_rollout_pack(channel="B")
        pack_metrics = dict(pack_metrics)
        pack_metrics["time/post_rollout_pack_s"] = float(time.perf_counter() - t_pack0)
        local_packs.append((selected, pack_metrics))

    _trace("channel_b_non_pipeline_before_prepare_barrier")
    # This barrier sits after the full rank-local rollout/parse/prepare/pack path.
    # Use the rollout wait budget rather than the shorter final-sync timeout so
    # healthy but imbalanced ranks do not trip a false DDP deadlock.
    ddp_phase_barrier_fn(
        "channel_b_non_pipeline_after_prepare",
        timeout_s=float(ddp_phase_prepare_timeout_s),
    )
    _trace("channel_b_non_pipeline_after_prepare_barrier")

    schedule = build_channel_b_pack_schedule(
        owner=owner,
        model=model,
        local_pack_count=int(len(local_packs)),
        dist=dist,
        ddp_world_size=int(ddp_world_size),
    )
    if not schedule.has_global_packs:
        raise ValueError(
            "stage2-ab Channel-B step mode produced zero post-rollout packs on all ranks; "
            "check rollout parsing, fallback policy, and target construction"
        )
    _trace(
        "channel_b_non_pipeline_after_pack_schedule",
        extra=schedule.trace_payload(),
    )
    if (
        schedule.has_global_packs
        and schedule.rank_pack_counts
        and int(min(schedule.rank_pack_counts)) <= 0
    ):
        raise RuntimeError(
            "stage2-ab Channel-B DDP pack schedule found a rank with zero local packs "
            "while at least one peer rank has trainable packs. This is a target-construction "
            "or strict-drop issue; ensure fallback target construction yields at least one "
            "segment per rank before entering DDP learner slots. "
            f"rank_pack_counts={list(schedule.rank_pack_counts)}"
        )

    loss_total = None
    first_real_pack = True
    for slot in schedule.slots:
        with owner._stage2_stage_wallclock_ctx("sft"):
            assert isinstance(slot, Stage2PackSlot)
            sync_gradients = bool(slot.sync_gradients)
            if slot.is_empty:
                _trace(
                    "channel_b_non_pipeline_empty_pack_slot",
                    extra={
                        "slot_index": int(slot.slot_index),
                        "global_slot_count": int(schedule.global_slot_count),
                        "local_pack_count": int(schedule.local_pack_count),
                        "sync_gradients": float(bool(sync_gradients)),
                    },
                )
                if bool(sync_gradients):
                    _trace(
                        "channel_b_non_pipeline_before_final_sync_backward_barrier",
                        extra={
                            "slot_index": int(slot.slot_index),
                            "global_slot_count": int(schedule.global_slot_count),
                            "local_pack_count": int(schedule.local_pack_count),
                            "shadow_zero_loss": 1.0,
                        },
                    )
                    ddp_phase_barrier_fn(
                        "channel_b_non_pipeline_before_final_sync_backward",
                        timeout_s=float(ddp_phase_final_sync_timeout_s),
                    )
                    _trace(
                        "channel_b_non_pipeline_after_final_sync_backward_barrier",
                        extra={
                            "slot_index": int(slot.slot_index),
                            "global_slot_count": int(schedule.global_slot_count),
                            "local_pack_count": int(schedule.local_pack_count),
                            "shadow_zero_loss": 1.0,
                        },
                    )
                shadow_selected, shadow_pack_metrics = local_packs[0]
                shadow_metrics = dict(shadow_pack_metrics)
                shadow_metrics["packing/post_rollout_local_pack_count"] = float(
                    schedule.local_pack_count
                )
                shadow_metrics["packing/post_rollout_global_slot_count"] = float(
                    schedule.global_slot_count
                )
                shadow_metrics["packing/post_rollout_empty_slot_count"] = float(
                    schedule.empty_slot_count
                )
                shadow_metrics["packing/post_rollout_slot_index"] = float(slot.slot_index)
                shadow_metrics["packing/post_rollout_slot_is_final_sync"] = float(
                    1.0 if bool(sync_gradients) else 0.0
                )
                shadow_metrics["packing/post_rollout_slot_sync_gradients"] = float(
                    1.0 if bool(sync_gradients) else 0.0
                )
                shadow_metrics["packing/post_rollout_shadow_slot"] = 1.0
                loss_pack = run_channel_b_train_one_pack(
                    owner=owner,
                    model=model,
                    selected=shadow_selected,
                    pack_metrics=shadow_metrics,
                    rollout_static=rollout_static,
                    step_totals={},
                    total_segments_target=int(total_segments_target),
                    sync_gradients=bool(sync_gradients),
                    dist=dist,
                    ddp_rank=int(ddp_rank),
                    ddp_world_size=int(ddp_world_size),
                    shadow_zero_loss=True,
                )
                loss_total = loss_pack if loss_total is None else (loss_total + loss_pack)
                continue

            selected, pack_metrics = local_packs[int(slot.local_pack_index)]
            pack_metrics = dict(pack_metrics)
            pack_metrics["packing/post_rollout_local_pack_count"] = float(
                schedule.local_pack_count
            )
            pack_metrics["packing/post_rollout_global_slot_count"] = float(
                schedule.global_slot_count
            )
            pack_metrics["packing/post_rollout_empty_slot_count"] = float(
                schedule.empty_slot_count
            )
            pack_metrics["packing/post_rollout_slot_index"] = float(slot.slot_index)
            pack_metrics["packing/post_rollout_slot_is_final_sync"] = float(
                1.0 if bool(sync_gradients) else 0.0
            )
            pack_metrics["packing/post_rollout_slot_sync_gradients"] = float(
                1.0 if bool(sync_gradients) else 0.0
            )

            step_totals_pack = step_totals if first_real_pack else {}
            if bool(sync_gradients):
                _trace(
                    "channel_b_non_pipeline_before_final_sync_backward_barrier",
                    extra={
                        "selected_pack_size": int(len(selected)),
                        "slot_index": int(slot.slot_index),
                        "global_slot_count": int(schedule.global_slot_count),
                        "local_pack_count": int(schedule.local_pack_count),
                    },
                )
                ddp_phase_barrier_fn(
                    "channel_b_non_pipeline_before_final_sync_backward",
                    timeout_s=float(ddp_phase_final_sync_timeout_s),
                )
                _trace(
                    "channel_b_non_pipeline_after_final_sync_backward_barrier",
                    extra={
                        "selected_pack_size": int(len(selected)),
                        "slot_index": int(slot.slot_index),
                        "global_slot_count": int(schedule.global_slot_count),
                        "local_pack_count": int(schedule.local_pack_count),
                    },
                )
            loss_pack = run_channel_b_train_one_pack(
                owner=owner,
                model=model,
                selected=selected,
                pack_metrics=pack_metrics,
                rollout_static=rollout_static,
                step_totals=step_totals_pack,
                total_segments_target=int(total_segments_target),
                sync_gradients=bool(sync_gradients),
                dist=dist,
                ddp_rank=int(ddp_rank),
                ddp_world_size=int(ddp_world_size),
            )

        loss_total = loss_pack if loss_total is None else (loss_total + loss_pack)
        first_real_pack = False

    if loss_total is None:
        raise AssertionError("stage2-ab Channel-B step mode produced no packs")
    return loss_total


def run_channel_b_pipeline_learning_loop(
    *,
    owner: Any,
    model: Any,
    raw_samples: Sequence[Mapping[str, Any]],
    rollout_decode_bs: int,
    producer_wait_timeout_s: float,
    packing_length: int,
    target_fill: float,
    total_segments_target: int,
    target_log_step: int,
    ddp_phase_final_sync_timeout_s: float,
    ddp_phase_barrier_fn: Any,
    dist: Any,
    ddp_rank: int,
    ddp_world_size: int,
) -> torch.Tensor:
    q: queue.Queue = queue.Queue(maxsize=1)
    producer_exc: list[Exception] = []

    def _producer() -> None:
        run_channel_b_pipeline_producer(
            owner=owner,
            raw_samples=raw_samples,
            rollout_decode_bs=int(rollout_decode_bs),
            queue_obj=q,
            producer_exc=producer_exc,
        )

    th = threading.Thread(target=_producer, daemon=True)
    th.start()

    rollout_static: Dict[str, float] = {}
    pending_totals: Dict[str, float] = {
        "stage2/raw_rollouts": float(total_segments_target)
    }

    buf_total_len = 0
    seen_segments = 0
    seen_raw = 0
    producer_done = False

    prefill_target_len = int(max(1, int(packing_length)))
    loss_total = None

    while (not producer_done) or owner._stage2_post_rollout_buffer(channel="B"):
        while (not producer_done) and (buf_total_len < int(prefill_target_len)):
            try:
                item = q.get(timeout=float(producer_wait_timeout_s))
            except queue.Empty as exc:
                producer_alive = bool(th.is_alive())
                pending_buf = int(len(owner._stage2_post_rollout_buffer(channel="B")))
                raise RuntimeError(
                    "stage2-ab Channel-B pipeline stalled while waiting for producer output; "
                    f"waited={float(producer_wait_timeout_s):.1f}s "
                    f"seen_raw={int(seen_raw)}/{int(total_segments_target)} "
                    f"seen_segments={int(seen_segments)} "
                    f"buf_total_len={int(buf_total_len)} pending_buf={int(pending_buf)} "
                    f"producer_done={bool(producer_done)} producer_alive={bool(producer_alive)} "
                    f"rollout_decode_batch_size={int(rollout_decode_bs)} "
                    f"packing_length={int(packing_length)} target_fill={float(target_fill):.3f} "
                    f"prefill_target_len={int(prefill_target_len)}."
                ) from exc
            if item is None:
                producer_done = True
                break

            seen_segments, seen_raw, buf_total_len = consume_channel_b_queue_item(
                owner=owner,
                item=item,
                rollout_static=rollout_static,
                pending_totals=pending_totals,
                seen_segments=int(seen_segments),
                seen_raw=int(seen_raw),
                buf_total_len=int(buf_total_len),
            )

        if not owner._stage2_post_rollout_buffer(channel="B"):
            continue

        with owner._stage2_stage_wallclock_ctx("sft"):
            t_pack0 = time.perf_counter()
            selected, pack_metrics = owner._stage2_pop_post_rollout_pack(channel="B")
            buf_total_len -= int(sum(int(sl) for _, _, sl in selected))

            pack_metrics = dict(pack_metrics)
            pack_metrics["time/post_rollout_pack_s"] = float(time.perf_counter() - t_pack0)

            step_totals_pack, is_last_pack = prepare_channel_b_pipeline_pack_step(
                owner=owner,
                selected=selected,
                pending_totals=pending_totals,
                seen_raw=int(seen_raw),
                total_segments_target=int(total_segments_target),
                ddp_phase_final_sync_timeout_s=float(ddp_phase_final_sync_timeout_s),
                ddp_phase_barrier_fn=ddp_phase_barrier_fn,
            )
            pending_totals = {}
            loss_pack = run_channel_b_train_one_pack(
                owner=owner,
                model=model,
                selected=selected,
                pack_metrics=pack_metrics,
                rollout_static=rollout_static,
                step_totals=step_totals_pack,
                total_segments_target=int(total_segments_target),
                sync_gradients=bool(is_last_pack),
                dist=dist,
                ddp_rank=int(ddp_rank),
                ddp_world_size=int(ddp_world_size),
            )
        loss_total = loss_pack if loss_total is None else (loss_total + loss_pack)

    return finalize_channel_b_pipeline_step(
        thread_obj=th,
        owner=owner,
        target_log_step=int(target_log_step),
        producer_exc=producer_exc,
        total_segments_target=int(total_segments_target),
        seen_raw=int(seen_raw),
        seen_segments=int(seen_segments),
        loss_total=loss_total,
    )


def run_channel_b_pipeline_producer(
    *,
    owner: Any,
    raw_samples: Sequence[Mapping[str, Any]],
    rollout_decode_bs: int,
    queue_obj: Any,
    producer_exc: list[Exception],
) -> None:
    prev_skip = bool(getattr(owner, "_stage2_skip_vllm_server_sync", False))
    setattr(owner, "_stage2_skip_vllm_server_sync", True)
    try:
        for off in range(0, int(len(raw_samples)), int(rollout_decode_bs)):
            chunk = list(raw_samples[int(off) : int(off + rollout_decode_bs)])
            if not chunk:
                continue
            with owner._stage2_stage_wallclock_ctx("rollout"):
                segs, m = owner._prepare_batch_inputs_b(chunk, _segments_only=True)
            raw_n = int(len(chunk))
            queue_obj.put((segs, dict(m) if isinstance(m, Mapping) else {}, raw_n))
    except (
        AttributeError,
        IndexError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        producer_exc.append(exc)
    finally:
        setattr(owner, "_stage2_skip_vllm_server_sync", prev_skip)
        while True:
            try:
                queue_obj.put(None, timeout=1.0)
                break
            except Exception:
                continue


def accumulate_step_mode_microbatches(
    *,
    owner: Any,
    gs_attr: str,
    micro_attr: str,
    raw_attr: str,
    raw_micro_batch: Sequence[Mapping[str, Any]],
    global_step: int,
) -> Tuple[bool, list[Mapping[str, Any]]]:
    gs = int(global_step)
    if getattr(owner, gs_attr) is None or int(getattr(owner, gs_attr)) != gs:
        setattr(owner, gs_attr, int(gs))
        setattr(owner, micro_attr, 0)
        setattr(owner, raw_attr, [])

    setattr(owner, micro_attr, int(getattr(owner, micro_attr)) + 1)
    getattr(owner, raw_attr).extend(list(raw_micro_batch))

    try:
        gas = int(getattr(owner.args, "gradient_accumulation_steps", 1) or 1)
    except (AttributeError, TypeError, ValueError):
        gas = 1
    gas = max(1, int(gas))

    if int(getattr(owner, micro_attr)) < int(gas):
        return False, []

    raw_all = list(getattr(owner, raw_attr))
    setattr(owner, raw_attr, [])
    setattr(owner, micro_attr, 0)
    return True, raw_all


__all__ = [
    "accumulate_channel_b_producer_item",
    "accumulate_step_mode_microbatches",
    "build_channel_b_pack_schedule",
    "consume_channel_b_queue_item",
    "finalize_channel_b_pipeline_step",
    "gather_channel_b_local_pack_counts",
    "prepare_channel_b_pipeline_pack_step",
    "run_channel_b_nonpipeline_learning_loop",
    "run_channel_b_pipeline_learning_loop",
    "run_channel_b_train_one_pack",
    "run_stage2_ab_ddp_monitored_barrier",
    "run_channel_b_pipeline_producer",
    "resolve_channel_b_timeouts",
    "split_rollout_metrics",
]
