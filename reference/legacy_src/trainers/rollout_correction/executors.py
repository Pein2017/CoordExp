"""Stage-2 rollout-correction execution helpers.

This module contains methods that execute rollout-correction work, including
step-budgeted training modes and post-rollout packing buffers.

The mixin methods are designed to operate on a partially-initialized trainer
instance (some unit tests construct the trainer via `__new__`).
"""

from __future__ import annotations

import contextlib
import logging
import time
from typing import Any, Dict, List, Literal, Mapping, Sequence, Tuple

import torch

from ..stage2_coordination import (
    Stage2DDPPhaseConfig,
    prime_stage2_ddp_monitor_group,
    resolve_rollout_correction_ddp_phase_config,
    resolve_stage2_prepare_barrier_timeout,
)
from .coordination import (
    accumulate_step_mode_microbatches,
    build_rollout_correction_pack_schedule,
    resolve_rollout_correction_timeouts,
    run_rollout_correction_nonpipeline_learning_loop,
    run_rollout_correction_pipeline_learning_loop,
    run_rollout_correction_ddp_monitored_barrier,
)

logger = logging.getLogger(__name__)


class RolloutCorrectionExecutorsMixin:
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

    def _stage2_post_rollout_channel(self, channel: str) -> Literal["rollout_correction"]:
        if str(channel).strip() != "rollout_correction":
            raise ValueError("stage2_rollout_correction post-rollout buffer has no A/B channels")
        return "rollout_correction"

    @contextlib.contextmanager
    def _rollout_correction_disable_average_tokens_across_devices_for_packed_step(
        self,
        *,
        dist: Any,
        ddp_rank: int,
        ddp_world_size: int,
        where: str,
    ):
        """Temporarily disable `average_tokens_across_devices` during packed per-step loops.

        Stage-2 rollout-correction packing can execute a variable number of per-pack
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
                        getattr(self, "_stage2_rollout_correction_avg_tokens_override_warned", False)
                    )
                    if (not warned) and int(ddp_rank) == 0:
                        logger.warning(
                            "stage2_rollout_correction: forcing args.average_tokens_across_devices=false during packed per-step execution "
                            "(world_size=%s where=%s) to avoid per-pack collective deadlocks when pack counts differ across ranks.",
                            int(ddp_world_size),
                            str(where),
                        )
                        setattr(self, "_stage2_rollout_correction_avg_tokens_override_warned", True)

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
            buf_map = {"rollout_correction": []}
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
                buf_map = {"rollout_correction": []}
                self._stage2_post_rollout_segments = buf_map  # type: ignore[attr-defined]
            current_size = len(buf_map.get("rollout_correction") or [])
            new_size = int(current_size) + int(len(seg_list))
            if new_size > cap:
                raise ValueError(
                    "post-rollout packing buffer overflow: "
                    f"buffer_size={new_size} > packing_buffer={cap}. "
                    "Mitigations: reduce rollout_matching.rollout_decode_batch_size, increase training.packing_buffer, "
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

    def _rollout_correction_ddp_monitored_barrier(
        self,
        *,
        dist: Any,
        phase: str,
        rank: int,
        world_size: int,
        timeout_s: float,
        monitor_group_timeout_s: float,
    ) -> None:
        run_rollout_correction_ddp_monitored_barrier(
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
        raise ValueError(
            "The obsolete stage-a teacher-forced training path has been removed; "
            "use stage2_rollout_correction rollout-prefix + GT-correction training."
        )

    def _stage2_rollout_correction_pipeline_enabled(
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
            warned = bool(getattr(self, "_stage2_rollout_correction_pipeline_ddp_warned", False))
            if (not warned) and int(rank) == 0:
                logger.warning(
                    "stage2_rollout_correction async rollout pipeline is disabled under DDP "
                    "(world_size=%s) to prevent cross-rank sync deadlocks; "
                    "falling back to non-pipelined step execution.",
                    int(world_size),
                )
                setattr(self, "_stage2_rollout_correction_pipeline_ddp_warned", True)
            return False

        return True

    def _stage2_rollout_correction_step_budgeted_train(
        self,
        model,
        *,
        raw_samples: List[Mapping[str, Any]],
        global_step: int,
    ) -> torch.Tensor:
        """Run one rollout-correction optimizer step from a raw rollout batch.

        This method is intentionally factored out so unit tests can monkeypatch it.

        Step-budgeted semantics:
        - build post-rollout segments for the raw batch
        - pack into a variable number of packed sequences (<= packing_length)
        - run forward/backward once per pack and accumulate gradients
        - outer Trainer performs the single optimizer.step()
        """
        if not raw_samples:
            raise ValueError(
                "stage2_rollout_correction step mode requires non-empty raw_samples"
            )

        target_log_step = int(global_step + 1)
        self._stage2_reset_train_monitor_dump(global_step=target_log_step)

        # Ensure dropout/BN behavior is correct even when we bypass the base Trainer.training_step.
        model.train()

        packing_enabled = bool(self._packing_enabled())
        if not packing_enabled:
            raise ValueError(
                "stage2_rollout_correction step mode currently requires training.packing=true "
                "(learner microbatch=1 under global_max_length)."
            )

        backend = str(getattr(self, "_rollout_backend", lambda: "")()).strip().lower()
        mode = str(getattr(self, "_vllm_mode", lambda: "")()).strip().lower()
        enable_pipeline = bool(
            self._stage2_rollout_correction_pipeline_enabled(
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
        ) = resolve_rollout_correction_timeouts(
            owner=self,
            ddp_world_size=int(ddp_world_size),
        )
        # The non-pipelined "after prepare" barrier covers full rank-local
        # rollout/parse/prepare work, which can skew far more than the short
        # final-sync backward barrier. Reuse the rollout wait budget there.
        ddp_phase_prepare_timeout_s = resolve_stage2_prepare_barrier_timeout(
            final_sync_timeout_s=float(ddp_phase_final_sync_timeout_s),
            producer_wait_timeout_s=float(producer_wait_timeout_s),
        )
        phase_config = Stage2DDPPhaseConfig(
            monitor_enabled=bool(ddp_phase_monitor_enabled),
            final_sync_timeout_s=float(ddp_phase_final_sync_timeout_s),
            monitor_group_timeout_s=float(
                max(
                    float(ddp_monitor_group_timeout_s),
                    float(ddp_phase_prepare_timeout_s),
                )
            ),
        )

        # Eagerly initialize the optional gloo monitor group at a safe synchronized
        # boundary (start of rollout-correction step) so later monitored barriers can time out
        # even if a rank stalls before reaching the first barrier.
        prime_stage2_ddp_monitor_group(
            self,
            dist=dist,
            rank=int(ddp_rank),
            world_size=int(ddp_world_size),
            config=phase_config,
            logger=logger,
        )

        def _ddp_phase_barrier(phase: str, *, timeout_s: float | None = None) -> None:
            if (
                dist is None
                or (not dist.is_available())
                or (not dist.is_initialized())
                or int(ddp_world_size) <= 1
            ):
                return

            local_timeout_s = (
                float(ddp_phase_final_sync_timeout_s)
                if timeout_s is None
                else float(timeout_s)
            )
            run_rollout_correction_ddp_monitored_barrier(
                owner=self,
                dist=dist,
                phase=phase,
                rank=int(ddp_rank),
                world_size=int(ddp_world_size),
                timeout_s=float(local_timeout_s),
                monitor_group_timeout_s=float(phase_config.monitor_group_timeout_s),
            )

        # Step-budgeted mode: do NOT carry segments across optimizer steps.
        buf = self._stage2_post_rollout_buffer(channel="rollout_correction")
        if buf:
            raise ValueError(
                "stage2_rollout_correction step mode requires an empty post-rollout buffer at step start; "
                "disable carry across steps or investigate unexpected leftovers"
            )

        total_segments_target = int(len(raw_samples))
        if total_segments_target <= 0:
            raise AssertionError("unexpected empty raw_samples")

        if not enable_pipeline:
            with self._stage2_stage_wallclock_ctx("rollout"):
                segments, batch_metrics = self._prepare_rollout_correction_inputs(
                    list(raw_samples), _segments_only=True
                )
            trace_fn = getattr(self, "_stage2_record_ddp_phase_trace", None)
            if callable(trace_fn):
                trace_fn(
                    global_step=int(target_log_step),
                    phase="rollout_correction_prepare_return",
                    rank=int(ddp_rank),
                    world_size=int(ddp_world_size),
                    payload={
                        "segment_count": int(len(segments)) if isinstance(segments, list) else 0,
                        "total_segments_target": int(total_segments_target),
                    },
                )
            return run_rollout_correction_nonpipeline_learning_loop(
                owner=self,
                model=model,
                segments=segments,
                batch_metrics=batch_metrics,
                target_log_step=int(target_log_step),
                total_segments_target=int(total_segments_target),
                ddp_phase_prepare_timeout_s=float(ddp_phase_prepare_timeout_s),
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

        return run_rollout_correction_pipeline_learning_loop(
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
        raise ValueError(
            "The obsolete stage-a teacher-forced training path has been removed; "
            "use stage2_rollout_correction rollout-prefix + GT-correction training."
        )

    def _stage2_training_step_b_step_mode(
        self,
        model,
        raw_micro_batch: List[Mapping[str, Any]],
        *,
        global_step: int,
    ) -> torch.Tensor:
        """Rollout-correction step-budgeted training_step shim.

        Collect raw samples across micro-steps and execute the full correction loop only
        on the final micro-step of the accumulation window.
        """
        gs = int(global_step)
        ready, raw_all = accumulate_step_mode_microbatches(
            owner=self,
            gs_attr="_stage2_rollout_correction_step_gs",
            micro_attr="_stage2_rollout_correction_step_micro",
            raw_attr="_stage2_rollout_correction_step_raw",
            raw_micro_batch=raw_micro_batch,
            global_step=int(gs),
        )
        if not bool(ready):
            return torch.tensor(0.0, device=self.model.device)

        # Validate expected raw sample count (best-effort; may differ under drop_last/resume).
        try:
            target_global = int(self._stage2_rollouts_per_step())
            target_local = (
                int(self._stage2_rollouts_per_rank()) if target_global > 0 else 0
            )
        except (AttributeError, TypeError, ValueError):
            target_global = 0
            target_local = 0

        if target_local > 0:
            if len(raw_all) < target_local:
                raise ValueError(
                    "stage2_rollout_correction step mode collected fewer raw samples than expected on this rank: "
                    f"{len(raw_all)} < {target_local} (expected global_raw={target_global}). "
                    "Mitigations: set training.dataloader_drop_last=true, and ensure training.effective_batch_size is divisible by per_device_train_batch_size*world_size (so gradient_accumulation_steps is an integer)."
                )
            if len(raw_all) > target_local:
                logger.warning(
                    "stage2_rollout_correction step mode collected more raw samples than expected on this rank; "
                    "dropping extras to honor effective_batch_size-derived raw budget: %s > %s (global=%s)",
                    len(raw_all),
                    target_local,
                    target_global,
                )
                raw_all = list(raw_all[:target_local])

        return self._stage2_rollout_correction_step_budgeted_train(
            model, raw_samples=raw_all, global_step=gs
        )
