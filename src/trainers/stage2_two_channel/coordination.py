import contextlib
from datetime import timedelta
import queue
import threading
import time
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch


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

    if int(total_segments_target) > 0 and int(seen_segments) > int(total_segments_target):
        raise ValueError(
            "stage2-ab Channel-B pipeline produced too many segments: "
            f"seen_segments={int(seen_segments)} target={int(total_segments_target)}"
        )

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

    pack_segments = int(len(selected))
    weight = float(pack_segments) / float(total_segments_target)

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


def run_channel_b_nonpipeline_learning_loop(
    *,
    owner: Any,
    model: Any,
    segments: list[tuple[dict[str, Any], dict[str, Any], int]],
    batch_metrics: Mapping[str, Any],
    target_log_step: int,
    total_segments_target: int,
    ddp_phase_final_sync_timeout_s: float,
    ddp_phase_barrier_fn: Any,
    dist: Any,
    ddp_rank: int,
    ddp_world_size: int,
) -> torch.Tensor:
    owner._stage2_flush_train_monitor_dump(global_step=target_log_step)
    if not isinstance(segments, list) or not segments:
        raise ValueError(
            "stage2-ab Channel-B step mode produced no post-rollout segments; "
            "check rollout parsing / dataset contract"
        )

    batch_metrics = dict(batch_metrics) if isinstance(batch_metrics, Mapping) else {}
    batch_metrics["stage2_ab/channel_b/train_monitor_dump_written"] = float(
        1.0
        if int(getattr(owner, "_stage2_train_monitor_dump_written_step", -1) or -1)
        == int(target_log_step)
        else 0.0
    )
    rollout_static, step_totals = split_rollout_metrics(batch_metrics)
    step_totals["stage2/raw_rollouts"] = float(total_segments_target)

    owner._stage2_append_post_rollout_segments(channel="B", segments=segments)
    ddp_phase_barrier_fn("channel_b_non_pipeline_after_prepare")

    loss_total = None
    first_pack = True
    while owner._stage2_post_rollout_buffer(channel="B"):
        with owner._stage2_stage_wallclock_ctx("sft"):
            t_pack0 = time.perf_counter()
            selected, pack_metrics = owner._stage2_pop_post_rollout_pack(channel="B")
            pack_metrics = dict(pack_metrics)
            pack_metrics["time/post_rollout_pack_s"] = float(time.perf_counter() - t_pack0)

            step_totals_pack = step_totals if first_pack else {}
            sync_gradients = not bool(owner._stage2_post_rollout_buffer(channel="B"))
            if bool(sync_gradients):
                ddp_phase_barrier_fn(
                    "channel_b_non_pipeline_before_final_sync_backward",
                    timeout_s=float(ddp_phase_final_sync_timeout_s),
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
        first_pack = False

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
    "consume_channel_b_queue_item",
    "finalize_channel_b_pipeline_step",
    "prepare_channel_b_pipeline_pack_step",
    "run_channel_b_nonpipeline_learning_loop",
    "run_channel_b_pipeline_learning_loop",
    "run_channel_b_train_one_pack",
    "run_stage2_ab_ddp_monitored_barrier",
    "run_channel_b_pipeline_producer",
    "resolve_channel_b_timeouts",
    "split_rollout_metrics",
]
