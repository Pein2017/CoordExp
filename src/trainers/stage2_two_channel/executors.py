"""Stage-2 AB per-channel execution helpers.

This module contains methods that execute Channel-A / Channel-B work, including
step-budgeted training modes and per-channel post-rollout packing buffers.

The mixin methods are designed to operate on a partially-initialized trainer
instance (some unit tests construct the trainer via `__new__`).
"""

from __future__ import annotations

import contextlib
from datetime import timedelta
import logging
import queue
import threading
import time
from typing import Any, Dict, List, Literal, Mapping, Sequence, Tuple

import torch

from .coordination import (
    accumulate_channel_b_producer_item,
    accumulate_step_mode_microbatches,
    consume_channel_b_queue_item,
    finalize_channel_b_pipeline_step,
    prepare_channel_b_pipeline_pack_step,
    resolve_channel_b_timeouts,
    run_channel_b_nonpipeline_learning_loop,
    run_channel_b_pipeline_learning_loop,
    run_channel_b_train_one_pack,
    run_stage2_ab_ddp_monitored_barrier,
    run_channel_b_pipeline_producer,
    split_rollout_metrics,
)

logger = logging.getLogger(__name__)


class Stage2ABChannelExecutorsMixin:
    def _stage2_stage_wallclock_ctx(self, stage: str):
        track = getattr(self, "_track_stage_wallclock", None)
        if callable(track):
            return track(str(stage))
        return contextlib.nullcontext()

    def _stage2_reset_train_monitor_dump(self, *, global_step: int) -> None:
        """Best-effort hook for subclasses that collect suspicious train dumps.

        The main Stage-2 trainer overrides this with real buffering logic. Keep a
        no-op default here so executor-level tests and lightweight subclasses can
        reuse the mixin without implementing the monitoring helpers.
        """

        return None

    def _stage2_flush_train_monitor_dump(self, *, global_step: int) -> None:
        """Best-effort hook for subclasses that collect suspicious train dumps."""

        return None

    def _stage2_post_rollout_channel(self, channel: str) -> Literal["A", "B"]:
        s = str(channel).strip().upper()
        return "A" if s == "A" else "B"

    @contextlib.contextmanager
    def _stage2_ab_disable_average_tokens_across_devices_for_packed_step(
        self,
        *,
        dist: Any,
        ddp_rank: int,
        ddp_world_size: int,
        where: str,
    ):
        """Temporarily disable `average_tokens_across_devices` during packed per-step loops.

        Stage2-AB step-budgeted packing can execute a variable number of per-pack
        forward/backward passes per optimizer step, and the *pack count may differ
        across ranks*. When `TrainingArguments.average_tokens_across_devices=True`,
        some loss terms (e.g. coord_soft_ce_w1) perform distributed collectives
        inside loss computation. If ranks call those collectives a different number
        of times, training will deadlock.

        We force-disable token-averaging for the duration of a single packed forward
        to ensure no per-pack collectives are executed under DDP.
        """

        args = getattr(self, "args", None)
        prev = None
        changed = False

        if (
            args is not None
            and hasattr(args, "average_tokens_across_devices")
            and dist is not None
            and hasattr(dist, "is_available")
            and hasattr(dist, "is_initialized")
            and callable(dist.is_available)
            and callable(dist.is_initialized)
            and bool(dist.is_available())
            and bool(dist.is_initialized())
            and int(ddp_world_size) > 1
        ):
            try:
                prev = bool(getattr(args, "average_tokens_across_devices", False))
            except (TypeError, ValueError):
                prev = None

            if bool(prev):
                try:
                    setattr(args, "average_tokens_across_devices", False)
                    changed = True
                except (AttributeError, TypeError, ValueError):
                    changed = False
                else:
                    warned = bool(
                        getattr(self, "_stage2_ab_avg_tokens_override_warned", False)
                    )
                    if (not warned) and int(ddp_rank) == 0:
                        logger.warning(
                            "Stage2-AB: forcing args.average_tokens_across_devices=false during packed per-step execution "
                            "(world_size=%s where=%s) to avoid per-pack collective deadlocks when pack counts differ across ranks.",
                            int(ddp_world_size),
                            str(where),
                        )
                        setattr(self, "_stage2_ab_avg_tokens_override_warned", True)

        try:
            yield
        finally:
            if (
                bool(changed)
                and args is not None
                and prev is not None
                and hasattr(args, "average_tokens_across_devices")
            ):
                try:
                    setattr(args, "average_tokens_across_devices", bool(prev))
                except (AttributeError, TypeError, ValueError):
                    pass

    def _stage2_post_rollout_buffer(
        self, *, channel: str
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any], int]]:
        ch = self._stage2_post_rollout_channel(channel)
        buf_map = getattr(self, "_stage2_post_rollout_segments", None)
        if not isinstance(buf_map, dict):
            buf_map = {"A": [], "B": []}
            self._stage2_post_rollout_segments = buf_map  # type: ignore[attr-defined]
        buf = buf_map.get(ch)
        if not isinstance(buf, list):
            buf = []
            buf_map[ch] = buf
        return buf

    def _stage2_append_post_rollout_segments(
        self,
        *,
        channel: str,
        segments: Sequence[Tuple[Dict[str, Any], Dict[str, Any], int]],
    ) -> None:
        """Append newly produced segments to the channel-local packing buffer."""
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
            buf_map = getattr(self, "_stage2_post_rollout_segments", None)
            if not isinstance(buf_map, dict):
                buf_map = {"A": [], "B": []}
                self._stage2_post_rollout_segments = buf_map  # type: ignore[attr-defined]
            size_a = len(buf_map.get("A") or [])
            size_b = len(buf_map.get("B") or [])
            new_size = int(size_a) + int(size_b) + int(len(seg_list))
            if new_size > cap:
                raise ValueError(
                    "post-rollout packing buffer overflow: "
                    f"buffer_size={new_size} > packing_buffer={cap}. "
                    "Mitigations: reduce rollout_matching.channel_b_decode_batch_size, increase training.packing_buffer, "
                    "or disable packing."
                )

        self._stage2_post_rollout_buffer(channel=channel).extend(seg_list)

    def _stage2_pop_post_rollout_pack(
        self,
        *,
        channel: str,
    ) -> Tuple[List[Tuple[Dict[str, Any], Dict[str, Any], int]], Dict[str, float]]:
        """Select and remove segments for one packed forward pass (channel-local)."""
        packing_length = int(self._packing_length())
        if packing_length <= 0:
            raise ValueError("packing is enabled but packing_length is invalid")

        buf = self._stage2_post_rollout_buffer(channel=channel)
        if not buf:
            raise ValueError(
                f"packing is enabled but no post-rollout segments are available for channel {channel!r}"
            )

        encoded_lens = [int(seg_len) for _, _, seg_len in buf]
        selected_idx = self._select_post_rollout_segment_indices(
            encoded_lens,
            packing_length,
            min_fill_ratio=self._packing_min_fill_ratio(),
        )
        if not selected_idx:
            raise AssertionError("post-rollout packing selected an empty segment set")
        total_len = int(sum(encoded_lens[i] for i in selected_idx))

        selected = [buf[i] for i in selected_idx]
        for i in reversed(selected_idx):
            buf.pop(int(i))

        fill = float(total_len) / float(packing_length) if packing_length > 0 else 0.0
        target = float(self._packing_min_fill_ratio())
        if fill < target:
            logger.warning(
                "post-rollout packing underfilled (channel=%s): fill=%.3f target=%.3f segments=%s buffer=%s",
                self._stage2_post_rollout_channel(channel),
                fill,
                target,
                len(selected),
                len(buf),
            )

        pack_metrics: Dict[str, float] = {
            "packing/post_rollout_fill": float(fill),
            "packing/post_rollout_selected_total_len": float(total_len),
            "packing/post_rollout_segments": float(len(selected)),
            "packing/post_rollout_buffer": float(len(buf)),
        }
        return selected, pack_metrics

    def _stage2_ab_ddp_monitored_barrier(
        self,
        *,
        dist: Any,
        phase: str,
        rank: int,
        world_size: int,
        timeout_s: float,
        monitor_group_timeout_s: float,
    ) -> None:
        run_stage2_ab_ddp_monitored_barrier(
            owner=self,
            dist=dist,
            phase=phase,
            rank=rank,
            world_size=world_size,
            timeout_s=timeout_s,
            monitor_group_timeout_s=monitor_group_timeout_s,
        )

    def _stage2_a_step_budgeted_train(
        self,
        model,
        *,
        raw_samples: List[Mapping[str, Any]],
        global_step: int,
    ) -> torch.Tensor:
        """Run one Channel-A optimizer step worth of work from a raw sample batch.

        Step-budgeted semantics:
        - build teacher-forced segments for the raw batch
        - pack into a variable number of packed sequences (<= packing_length)
        - run forward/backward once per pack and accumulate gradients
        - outer Trainer performs the single optimizer.step()
        """
        if not raw_samples:
            raise ValueError(
                "stage2-ab Channel-A step mode requires non-empty raw_samples"
            )

        with self._stage2_stage_wallclock_ctx("sft"):
            # Ensure dropout/BN behavior is correct even when we bypass the base Trainer.training_step.
            model.train()

            packing_enabled = bool(self._packing_enabled())
            if not packing_enabled:
                raise ValueError(
                    "stage2-ab Channel-A step mode requires training.packing=true "
                    "(learner microbatch=1 under global_max_length)."
                )

            # Step-budgeted mode: do NOT carry segments across optimizer steps.
            buf = self._stage2_post_rollout_buffer(channel="A")
            if buf:
                raise ValueError(
                    "stage2-ab Channel-A step mode requires an empty post-rollout buffer at step start; "
                    "disable carry across steps or investigate unexpected leftovers"
                )

            total_segments_target = int(len(raw_samples))
            if total_segments_target <= 0:
                raise AssertionError("unexpected empty raw_samples")

            from swift.llm import to_device

            def _train_one_pack(
                *,
                selected: List[Tuple[Dict[str, Any], Dict[str, Any], int]],
                pack_metrics: Mapping[str, float],
                step_totals: Mapping[str, float],
                sync_gradients: bool,
            ) -> torch.Tensor:
                t_collate0 = time.perf_counter()
                with self._template_packing_enabled():
                    packed = self.template.data_collator([enc for enc, _, _ in selected])
                t_collate_s = float(time.perf_counter() - t_collate0)
                batch = to_device(packed, self.model.device)
                self._assert_single_packed_forward(batch, where="stage2_ab/packed_forward")
                batch["_rollout_matching_meta"] = [m for _, m, _ in selected]

                bm: Dict[str, float] = {}
                # Attach step-level totals (time/*, packing settings, etc) sparingly.
                bm.update({str(k): float(v) for k, v in step_totals.items()})
                bm.update({str(k): float(v) for k, v in pack_metrics.items()})
                bm["time/channel_a_collate_s"] = float(t_collate_s)

                self._merge_rollout_matching_batch_metrics(batch, bm)
                batch["_stage2_ab_channel"] = "A"

                pack_segments = int(len(selected))
                weight = float(pack_segments) / float(total_segments_target)
                cm = contextlib.nullcontext()
                if not bool(sync_gradients):
                    acc = getattr(self, "accelerator", None)
                    if acc is not None and hasattr(acc, "no_sync"):
                        cm = acc.no_sync(model)
                    else:
                        no_sync = getattr(model, "no_sync", None)
                        if callable(no_sync):
                            cm = model.no_sync()

                with cm:
                    loss_cm = getattr(self, "compute_loss_context_manager", None)
                    loss_ctx = loss_cm() if callable(loss_cm) else contextlib.nullcontext()
                    prev_gradmon_sync = getattr(
                        self, "_loss_gradient_monitor_sync_gradients", None
                    )
                    setattr(
                        self,
                        "_loss_gradient_monitor_sync_gradients",
                        bool(sync_gradients),
                    )
                    t_compute0 = time.perf_counter()
                    try:
                        with loss_ctx:
                            with self._stage2_ab_disable_average_tokens_across_devices_for_packed_step(
                                dist=dist,
                                ddp_rank=int(ddp_rank),
                                ddp_world_size=int(ddp_world_size),
                                where=f"stage2_ab/channel_{str(batch.get('_stage2_ab_channel', '?'))}/train_one_pack",
                            ):
                                loss = self.compute_loss(model, batch)
                    finally:
                        if prev_gradmon_sync is None:
                            try:
                                delattr(self, "_loss_gradient_monitor_sync_gradients")
                            except AttributeError:
                                pass
                        else:
                            setattr(
                                self,
                                "_loss_gradient_monitor_sync_gradients",
                                prev_gradmon_sync,
                            )
                    t_compute_s = float(time.perf_counter() - t_compute0)
                    if not isinstance(loss, torch.Tensor):
                        raise TypeError("compute_loss must return a torch.Tensor")

                    loss_scaled = loss * float(weight)

                    acc = getattr(self, "accelerator", None)
                    t_backward0 = time.perf_counter()
                    if acc is not None and hasattr(acc, "backward"):
                        acc.backward(loss_scaled)
                    else:
                        loss_scaled.backward()
                    t_backward_s = float(time.perf_counter() - t_backward0)

                loss_value = float(loss.detach().float().cpu().item())
                return loss.detach() * float(weight)

            t_segments0 = time.perf_counter()
            segments, batch_metrics = self._prepare_batch_inputs_a(
                list(raw_samples), _segments_only=True
            )
            t_segments_s = float(time.perf_counter() - t_segments0)
            if not isinstance(segments, list) or not segments:
                raise ValueError(
                    "stage2-ab Channel-A step mode produced no segments; check dataset contract"
                )

            step_totals = dict(batch_metrics) if isinstance(batch_metrics, Mapping) else {}
            step_totals["time/channel_a_prepare_segments_s"] = float(t_segments_s)
            self._stage2_append_post_rollout_segments(channel="A", segments=segments)

            try:
                import torch.distributed as dist
            except (AttributeError, RuntimeError, TypeError, ValueError):
                dist = None  # type: ignore[assignment]

            ddp_rank = 0
            ddp_world_size = 1
            if dist is not None and dist.is_available() and dist.is_initialized():
                ddp_world_size = max(1, int(dist.get_world_size()))

            loss_total = None
            first_pack = True
            while self._stage2_post_rollout_buffer(channel="A"):
                t_pack0 = time.perf_counter()
                selected, pack_metrics = self._stage2_pop_post_rollout_pack(channel="A")
                pack_metrics = dict(pack_metrics)
                pack_metrics["time/post_rollout_pack_s"] = float(
                    time.perf_counter() - t_pack0
                )

                step_totals_pack = step_totals if first_pack else {}
                sync_gradients = not bool(self._stage2_post_rollout_buffer(channel="A"))
                if (
                    bool(sync_gradients)
                    and dist is not None
                    and dist.is_available()
                    and dist.is_initialized()
                    and int(ddp_world_size) > 1
                ):
                    # Align ranks on the final (sync) backward to avoid DDP no_sync skew deadlocks
                    # when per-rank pack counts differ.
                    timeout_s = 120.0
                    ddp_phase_timeout_raw = self._ab_channel_b_get("ddp_phase_timeout_s", None)
                    if ddp_phase_timeout_raw is not None:
                        try:
                            timeout_s = float(ddp_phase_timeout_raw)
                        except (TypeError, ValueError) as exc:
                            raise ValueError(
                                "stage2_ab.channel_b.ddp_phase_timeout_s must be a float/int when set"
                            ) from exc

                        if float(timeout_s) <= 0.0:
                            timeout_s = 0.0

                    if float(timeout_s) > 0.0:
                        timeout_s = float(max(30.0, min(3600.0, float(timeout_s))))
                        self._stage2_ab_ddp_monitored_barrier(
                            dist=dist,
                            phase="stage2-ab Channel-A final-sync backward",
                            rank=int(dist.get_rank()),
                            world_size=int(ddp_world_size),
                            timeout_s=float(timeout_s),
                            monitor_group_timeout_s=float(timeout_s),
                        )
                loss_pack = _train_one_pack(
                    selected=selected,
                    pack_metrics=pack_metrics,
                    step_totals=step_totals_pack,
                    sync_gradients=sync_gradients,
                )

                loss_total = loss_pack if loss_total is None else (loss_total + loss_pack)
                first_pack = False

            if loss_total is None:
                raise AssertionError("stage2-ab Channel-A step mode produced no packs")
            return loss_total

    def _stage2_channel_b_pipeline_enabled(
        self,
        *,
        backend: str,
        mode: str,
    ) -> bool:
        if (
            str(backend).strip().lower() != "vllm"
            or str(mode).strip().lower() != "server"
        ):
            return False

        rank = 0
        world_size = 1
        dist_info_fn = getattr(self, "_dist_info", None)
        if callable(dist_info_fn):
            try:
                rank_raw, world_raw, _dist = dist_info_fn()
                rank = int(rank_raw)
                world_size = max(1, int(world_raw))
            except (TypeError, ValueError):
                rank = 0
                world_size = 1
        else:
            try:
                import torch.distributed as dist

                if dist.is_available() and dist.is_initialized():
                    rank = int(dist.get_rank())
                    world_size = max(1, int(dist.get_world_size()))
            except (AttributeError, RuntimeError, TypeError, ValueError):
                rank = 0
                world_size = 1

        if int(world_size) > 1:
            warned = bool(getattr(self, "_stage2_channel_b_pipeline_ddp_warned", False))
            if (not warned) and int(rank) == 0:
                logger.warning(
                    "stage2-ab Channel-B async rollout pipeline is disabled under DDP "
                    "(world_size=%s) to prevent cross-rank sync deadlocks; "
                    "falling back to non-pipelined step execution.",
                    int(world_size),
                )
                setattr(self, "_stage2_channel_b_pipeline_ddp_warned", True)
            return False

        return True

    def _stage2_b_step_budgeted_train(
        self,
        model,
        *,
        raw_samples: List[Mapping[str, Any]],
        global_step: int,
    ) -> torch.Tensor:
        """Run one Channel-B optimizer step worth of work from a raw rollout batch.

        This method is intentionally factored out so unit tests can monkeypatch it.

        Step-budgeted semantics:
        - build post-rollout segments for the raw batch
        - pack into a variable number of packed sequences (<= packing_length)
        - run forward/backward once per pack and accumulate gradients
        - outer Trainer performs the single optimizer.step()
        """
        if not raw_samples:
            raise ValueError(
                "stage2-ab Channel-B step mode requires non-empty raw_samples"
            )

        target_log_step = int(global_step + 1)
        self._stage2_reset_train_monitor_dump(global_step=target_log_step)

        # Ensure dropout/BN behavior is correct even when we bypass the base Trainer.training_step.
        model.train()

        packing_enabled = bool(self._packing_enabled())
        if not packing_enabled:
            raise ValueError(
                "stage2-ab Channel-B step mode currently requires training.packing=true "
                "(learner microbatch=1 under global_max_length)."
            )

        backend = str(getattr(self, "_rollout_backend", lambda: "")()).strip().lower()
        mode = str(getattr(self, "_vllm_mode", lambda: "")()).strip().lower()
        enable_pipeline = bool(
            self._stage2_channel_b_pipeline_enabled(
                backend=backend,
                mode=mode,
            )
        )

        rollout_decode_bs = int(self._rollout_decode_batch_size_per_rank())
        rollout_decode_bs = max(1, int(rollout_decode_bs))

        packing_length = int(self._packing_length())
        target_fill = float(self._packing_min_fill_ratio())

        try:
            import torch.distributed as dist
        except (AttributeError, RuntimeError, TypeError, ValueError):
            dist = None  # type: ignore[assignment]

        ddp_rank = 0
        ddp_world_size = 1
        if dist is not None and dist.is_available() and dist.is_initialized():
            ddp_rank = int(dist.get_rank())
            ddp_world_size = max(1, int(dist.get_world_size()))

        (
            producer_wait_timeout_s,
            ddp_phase_monitor_enabled,
            ddp_phase_final_sync_timeout_s,
            ddp_monitor_group_timeout_s,
        ) = resolve_channel_b_timeouts(
            owner=self,
            ddp_world_size=int(ddp_world_size),
        )

        # Eagerly initialize the optional gloo monitor group at a safe synchronized
        # boundary (start of Channel-B step) so later monitored barriers can time out
        # even if a rank stalls before reaching the first barrier. Lazy init inside
        # `_ddp_phase_barrier` can itself hang waiting for the missing rank.
        if (
            dist is not None
            and dist.is_available()
            and dist.is_initialized()
            and int(ddp_world_size) > 1
            and bool(ddp_phase_monitor_enabled)
            and hasattr(dist, "monitored_barrier")
        ):
            group = getattr(self, "_stage2_ab_ddp_monitor_group", None)
            if group is None:
                try:
                    init_timeout_s = float(max(30.0, min(120.0, ddp_phase_timeout_s)))
                    group = dist.new_group(
                        backend="gloo",
                        timeout=timedelta(seconds=float(init_timeout_s)),
                    )
                except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                    warned = bool(
                        getattr(self, "_stage2_ab_ddp_monitor_group_warned", False)
                    )
                    if int(ddp_rank) == 0 and not warned:
                        logger.warning(
                            "stage2-ab DDP phase monitor disabled (gloo group init failed): %r",
                            exc,
                        )
                        setattr(self, "_stage2_ab_ddp_monitor_group_warned", True)
                    setattr(self, "_stage2_ab_ddp_monitor_group", False)
                else:
                    setattr(self, "_stage2_ab_ddp_monitor_group", group)

        def _ddp_phase_barrier(phase: str, *, timeout_s: float | None = None) -> None:
            if (
                dist is None
                or (not dist.is_available())
                or (not dist.is_initialized())
                or int(ddp_world_size) <= 1
            ):
                return

            if not bool(ddp_phase_monitor_enabled):
                raise RuntimeError(
                    "stage2-ab DDP phase monitor is disabled under DDP. "
                    "Coordination barriers must be bounded to prevent deadlocks. "
                    "Set stage2_ab.channel_b.ddp_phase_timeout_s to a positive value."
                )

            if not hasattr(dist, "monitored_barrier"):
                raise RuntimeError(
                    "torch.distributed.monitored_barrier is required for bounded stage2-ab DDP phase barriers "
                    f"(phase={str(phase)} rank={int(ddp_rank)}/{int(ddp_world_size)})."
                )

            group = getattr(self, "_stage2_ab_ddp_monitor_group", None)
            if group is None:
                try:
                    group = dist.new_group(
                        backend="gloo",
                        timeout=timedelta(seconds=float(ddp_monitor_group_timeout_s)),
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "stage2-ab DDP phase monitored barrier requested but gloo group init failed; "
                        f"rank={int(ddp_rank)}/{int(ddp_world_size)} "
                        f"timeout_s={float(ddp_monitor_group_timeout_s):.1f}. "
                        "This is unsafe because falling back to an unbounded barrier can deadlock."
                    ) from exc
                setattr(self, "_stage2_ab_ddp_monitor_group", group)

            if group is False:
                raise RuntimeError(
                    "stage2-ab internal error: DDP monitor group is disabled under DDP; "
                    "this is unsafe because unbounded barriers can deadlock"
                )

            local_timeout_s = (
                float(ddp_phase_final_sync_timeout_s)
                if timeout_s is None
                else float(timeout_s)
            )
            local_timeout_s = float(max(30.0, min(3600.0, local_timeout_s)))

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
                    "stage2-ab Channel-B DDP phase barrier timed out; "
                    f"phase={str(phase)} rank={int(ddp_rank)}/{int(ddp_world_size)} "
                    f"timeout_s={float(local_timeout_s):.1f}. "
                    "This indicates a cross-rank stage skew or deadlock after rollout."
                ) from exc

        # Step-budgeted mode: do NOT carry segments across optimizer steps.
        buf = self._stage2_post_rollout_buffer(channel="B")
        if buf:
            raise ValueError(
                "stage2-ab Channel-B step mode requires an empty post-rollout buffer at step start; "
                "disable carry across steps or investigate unexpected leftovers"
            )

        total_segments_target = int(len(raw_samples))
        if total_segments_target <= 0:
            raise AssertionError("unexpected empty raw_samples")

        if not enable_pipeline:
            with self._stage2_stage_wallclock_ctx("rollout"):
                segments, batch_metrics = self._prepare_batch_inputs_b(
                    list(raw_samples), _segments_only=True
                )
            trace_fn = getattr(self, "_stage2_record_ddp_phase_trace", None)
            if callable(trace_fn):
                trace_fn(
                    global_step=int(target_log_step),
                    phase="channel_b_prepare_return",
                    rank=int(ddp_rank),
                    world_size=int(ddp_world_size),
                    payload={
                        "segment_count": int(len(segments)) if isinstance(segments, list) else 0,
                        "total_segments_target": int(total_segments_target),
                    },
                )
            return run_channel_b_nonpipeline_learning_loop(
                owner=self,
                model=model,
                segments=segments,
                batch_metrics=batch_metrics,
                target_log_step=int(target_log_step),
                total_segments_target=int(total_segments_target),
                ddp_phase_final_sync_timeout_s=float(ddp_phase_final_sync_timeout_s),
                ddp_phase_barrier_fn=_ddp_phase_barrier,
                dist=dist,
                ddp_rank=int(ddp_rank),
                ddp_world_size=int(ddp_world_size),
            )

        # Pipelined mode: produce segments in small decode micro-batches while the learner
        # consumes packed sequences. A bounded queue prevents unbounded rollout pooling.
        #
        # IMPORTANT: vLLM server sync uses DDP collectives/barriers and is not thread-safe.
        # Perform sync once on the main thread, then force the producer thread to skip sync.
        sync_fn = getattr(self, "_sync_vllm_server_rollout_model_if_needed", None)
        if callable(sync_fn):
            sync_fn()

        return run_channel_b_pipeline_learning_loop(
            owner=self,
            model=model,
            raw_samples=raw_samples,
            rollout_decode_bs=int(rollout_decode_bs),
            producer_wait_timeout_s=float(producer_wait_timeout_s),
            packing_length=int(packing_length),
            target_fill=float(target_fill),
            total_segments_target=int(total_segments_target),
            target_log_step=int(target_log_step),
            ddp_phase_final_sync_timeout_s=float(ddp_phase_final_sync_timeout_s),
            ddp_phase_barrier_fn=_ddp_phase_barrier,
            dist=dist,
            ddp_rank=int(ddp_rank),
            ddp_world_size=int(ddp_world_size),
        )

    def _stage2_training_step_a_step_mode(
        self,
        model,
        raw_micro_batch: List[Mapping[str, Any]],
        *,
        global_step: int,
    ) -> torch.Tensor:
        """Channel-A step-budgeted training_step shim.

        Collect raw samples across micro-steps and execute the full packing+learning loop
        only on the final micro-step of the accumulation window.

        This keeps one optimizer update per optimizer step, while still allowing packing
        multiple samples per backward under global_max_length.
        """
        gs = int(global_step)
        ready, raw_all = accumulate_step_mode_microbatches(
            owner=self,
            gs_attr="_stage2_a_step_gs",
            micro_attr="_stage2_a_step_micro",
            raw_attr="_stage2_a_step_raw",
            raw_micro_batch=raw_micro_batch,
            global_step=int(gs),
        )
        if not bool(ready):
            return torch.tensor(0.0, device=self.model.device)

        return self._stage2_a_step_budgeted_train(
            model, raw_samples=raw_all, global_step=gs
        )

    def _stage2_training_step_b_step_mode(
        self,
        model,
        raw_micro_batch: List[Mapping[str, Any]],
        *,
        global_step: int,
    ) -> torch.Tensor:
        """Channel-B step-budgeted training_step shim.

        Collect raw samples across micro-steps and execute the full Channel-B loop only
        on the final micro-step of the accumulation window.
        """
        gs = int(global_step)
        ready, raw_all = accumulate_step_mode_microbatches(
            owner=self,
            gs_attr="_stage2_b_step_gs",
            micro_attr="_stage2_b_step_micro",
            raw_attr="_stage2_b_step_raw",
            raw_micro_batch=raw_micro_batch,
            global_step=int(gs),
        )
        if not bool(ready):
            return torch.tensor(0.0, device=self.model.device)

        # Validate expected raw sample count (best-effort; may differ under drop_last/resume).
        try:
            target_global = int(self._stage2_b_rollouts_per_step())
            target_local = (
                int(self._stage2_b_rollouts_per_rank()) if target_global > 0 else 0
            )
        except (AttributeError, TypeError, ValueError):
            target_global = 0
            target_local = 0

        if target_local > 0:
            if len(raw_all) < target_local:
                raise ValueError(
                    "stage2-ab Channel-B step mode collected fewer raw samples than expected on this rank: "
                    f"{len(raw_all)} < {target_local} (expected global_raw={target_global}). "
                    "Mitigations: set training.dataloader_drop_last=true, and ensure training.effective_batch_size is divisible by per_device_train_batch_size*world_size (so gradient_accumulation_steps is an integer)."
                )
            if len(raw_all) > target_local:
                logger.warning(
                    "stage2-ab Channel-B step mode collected more raw samples than expected on this rank; "
                    "dropping extras to honor effective_batch_size-derived raw budget: %s > %s (global=%s)",
                    len(raw_all),
                    target_local,
                    target_global,
                )
                raw_all = list(raw_all[:target_local])

        return self._stage2_b_step_budgeted_train(
            model, raw_samples=raw_all, global_step=gs
        )
