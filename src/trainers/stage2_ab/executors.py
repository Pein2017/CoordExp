"""Stage-2 AB per-channel execution helpers.

This module contains methods that execute Channel-A / Channel-B work, including
step-budgeted training modes and per-channel post-rollout packing buffers.

The mixin methods are designed to operate on a partially-initialized trainer
instance (some unit tests construct the trainer via `__new__`).
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
from typing import Any, Dict, List, Literal, Mapping, Sequence, Tuple

import torch


logger = logging.getLogger(__name__)


class Stage2ABChannelExecutorsMixin:
    def _stage2_post_rollout_channel(self, channel: str) -> Literal["A", "B"]:
        s = str(channel).strip().upper()
        return "A" if s == "A" else "B"

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
                    "Mitigations: reduce rollout_matching.decode_batch_size, increase training.packing_buffer, "
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

        # Ensure dropout/BN behavior is correct even when we bypass the base Trainer.training_step.
        model.train()

        packing_enabled = False
        try:
            packing_enabled = bool(self._packing_enabled())
        except Exception:
            packing_enabled = False
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
            with self._template_packing_enabled():
                packed = self.template.data_collator([enc for enc, _, _ in selected])
            batch = to_device(packed, self.model.device)
            self._assert_single_packed_forward(batch, where="stage2_ab/packed_forward")
            batch["_rollout_matching_meta"] = [m for _, m, _ in selected]

            bm: Dict[str, float] = {}
            # Attach step-level totals (time/*, packing settings, etc) sparingly.
            bm.update({str(k): float(v) for k, v in step_totals.items()})
            bm.update({str(k): float(v) for k, v in pack_metrics.items()})

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
                with loss_ctx:
                    loss = self.compute_loss(model, batch)
                if not isinstance(loss, torch.Tensor):
                    raise TypeError("compute_loss must return a torch.Tensor")

                loss_scaled = loss * float(weight)

                acc = getattr(self, "accelerator", None)
                if acc is not None and hasattr(acc, "backward"):
                    acc.backward(loss_scaled)
                else:
                    loss_scaled.backward()

            return loss.detach() * float(weight)

        segments, batch_metrics = self._prepare_batch_inputs_a(
            list(raw_samples), _segments_only=True
        )
        if not isinstance(segments, list) or not segments:
            raise ValueError(
                "stage2-ab Channel-A step mode produced no segments; check dataset contract"
            )

        step_totals = dict(batch_metrics) if isinstance(batch_metrics, Mapping) else {}

        self._stage2_append_post_rollout_segments(channel="A", segments=segments)

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

        # Ensure dropout/BN behavior is correct even when we bypass the base Trainer.training_step.
        model.train()

        packing_enabled = False
        try:
            packing_enabled = bool(self._packing_enabled())
        except Exception:
            packing_enabled = False
        if not packing_enabled:
            raise ValueError(
                "stage2-ab Channel-B step mode currently requires training.packing=true "
                "(learner microbatch=1 under global_max_length)."
            )

        backend = str(getattr(self, "_rollout_backend", lambda: "")()).strip().lower()
        mode = str(getattr(self, "_vllm_mode", lambda: "")()).strip().lower()
        enable_pipeline = bool(backend == "vllm" and mode == "server")

        rollout_decode_bs = int(self._rollout_decode_batch_size_per_rank())
        rollout_decode_bs = max(1, int(rollout_decode_bs))

        packing_length = int(self._packing_length())
        target_fill = float(self._packing_min_fill_ratio())

        def _split_metrics(metrics: Mapping[str, Any]) -> Tuple[Dict[str, float], Dict[str, float]]:
            rollout_static: Dict[str, float] = {}
            step_totals: Dict[str, float] = {}
            for k, v in metrics.items():
                ks = str(k)
                try:
                    fv = float(v)  # type: ignore[arg-type]
                except Exception:
                    continue
                if ks.startswith("rollout/"):
                    rollout_static[ks] = float(fv)
                else:
                    step_totals[ks] = float(fv)
            return rollout_static, step_totals

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

        from swift.llm import to_device

        def _train_one_pack(
            *,
            selected: List[Tuple[Dict[str, Any], Dict[str, Any], int]],
            pack_metrics: Mapping[str, float],
            rollout_static: Mapping[str, float],
            step_totals: Mapping[str, float],
            sync_gradients: bool,
        ) -> torch.Tensor:
            with self._template_packing_enabled():
                packed = self.template.data_collator([enc for enc, _, _ in selected])
            batch = to_device(packed, self.model.device)
            self._assert_single_packed_forward(batch, where="stage2_ab/packed_forward")
            batch["_rollout_matching_meta"] = [m for _, m, _ in selected]

            bm: Dict[str, float] = {}
            # rollout/* keys are averaged in Stage2 pending logs; include on EVERY micro-pack.
            bm.update({str(k): float(v) for k, v in rollout_static.items()})
            # step-level totals (time/*, stage2/* counters, etc) can be attached sparsely.
            bm.update({str(k): float(v) for k, v in step_totals.items()})
            bm.update({str(k): float(v) for k, v in pack_metrics.items()})

            self._merge_rollout_matching_batch_metrics(batch, bm)
            batch["_stage2_ab_channel"] = "B"

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
                with loss_ctx:
                    loss = self.compute_loss(model, batch)
                if not isinstance(loss, torch.Tensor):
                    raise TypeError("compute_loss must return a torch.Tensor")

                loss_scaled = loss * float(weight)

                acc = getattr(self, "accelerator", None)
                if acc is not None and hasattr(acc, "backward"):
                    acc.backward(loss_scaled)
                else:
                    loss_scaled.backward()

            return loss.detach() * float(weight)

        if not enable_pipeline:
            segments, batch_metrics = self._prepare_batch_inputs_b(
                list(raw_samples), _segments_only=True
            )
            if not isinstance(segments, list) or not segments:
                raise ValueError(
                    "stage2-ab Channel-B step mode produced no post-rollout segments; "
                    "check rollout parsing / dataset contract"
                )

            batch_metrics = (
                dict(batch_metrics) if isinstance(batch_metrics, Mapping) else {}
            )
            rollout_static, step_totals = _split_metrics(batch_metrics)
            step_totals["stage2/raw_rollouts"] = float(total_segments_target)

            self._stage2_append_post_rollout_segments(channel="B", segments=segments)

            loss_total = None
            first_pack = True
            while self._stage2_post_rollout_buffer(channel="B"):
                t_pack0 = time.perf_counter()
                selected, pack_metrics = self._stage2_pop_post_rollout_pack(channel="B")
                pack_metrics = dict(pack_metrics)
                pack_metrics["time/post_rollout_pack_s"] = float(
                    time.perf_counter() - t_pack0
                )

                step_totals_pack = step_totals if first_pack else {}
                sync_gradients = not bool(self._stage2_post_rollout_buffer(channel="B"))
                loss_pack = _train_one_pack(
                    selected=selected,
                    pack_metrics=pack_metrics,
                    rollout_static=rollout_static,
                    step_totals=step_totals_pack,
                    sync_gradients=sync_gradients,
                )

                loss_total = (
                    loss_pack if loss_total is None else (loss_total + loss_pack)
                )
                first_pack = False

            if loss_total is None:
                raise AssertionError("stage2-ab Channel-B step mode produced no packs")
            return loss_total

        # Pipelined mode: produce segments in small decode micro-batches while the learner
        # consumes packed sequences. A bounded queue prevents unbounded rollout pooling.
        #
        # IMPORTANT: vLLM server sync uses DDP collectives/barriers and is not thread-safe.
        # Perform sync once on the main thread, then force the producer thread to skip sync.
        sync_fn = getattr(self, "_sync_vllm_server_rollout_model_if_needed", None)
        if callable(sync_fn):
            sync_fn()

        q: queue.Queue = queue.Queue(maxsize=1)
        producer_exc: List[BaseException] = []

        def _producer() -> None:
            prev_skip = bool(getattr(self, "_stage2_skip_vllm_server_sync", False))
            setattr(self, "_stage2_skip_vllm_server_sync", True)
            try:
                for off in range(0, int(len(raw_samples)), int(rollout_decode_bs)):
                    chunk = list(raw_samples[int(off) : int(off + rollout_decode_bs)])
                    if not chunk:
                        continue
                    segs, m = self._prepare_batch_inputs_b(chunk, _segments_only=True)
                    q.put((segs, dict(m) if isinstance(m, Mapping) else {}))
            except BaseException as exc:
                producer_exc.append(exc)
            finally:
                setattr(self, "_stage2_skip_vllm_server_sync", prev_skip)
                q.put(None)

        th = threading.Thread(target=_producer, daemon=True)
        th.start()

        rollout_static: Dict[str, float] = {}
        pending_totals: Dict[str, float] = {
            "stage2/raw_rollouts": float(total_segments_target)
        }

        buf_total_len = 0
        seen_segments = 0
        producer_done = False

        loss_total = None

        while (not producer_done) or self._stage2_post_rollout_buffer(channel="B"):
            # Fill the buffer until we can build a reasonably full pack, or until producer finishes.
            while (not producer_done) and (
                buf_total_len < float(target_fill) * float(packing_length)
            ):
                item = q.get()
                if item is None:
                    producer_done = True
                    break

                segs, metrics = item
                if not isinstance(segs, list):
                    raise TypeError("producer returned non-list segments")
                if not isinstance(metrics, Mapping):
                    metrics = {}

                seen_segments += int(len(segs))
                buf_total_len += int(sum(int(sl) for _, _, sl in segs))

                self._stage2_append_post_rollout_segments(channel="B", segments=segs)

                r_static, step_tot = _split_metrics(metrics)
                if not rollout_static:
                    rollout_static.update(r_static)
                else:
                    # Keep the first-seen rollout/* values (should be constant per step).
                    for k, v in r_static.items():
                        rollout_static.setdefault(k, float(v))

                # Accumulate step-level totals to be attached to the next trained pack.
                for k, v in step_tot.items():
                    pending_totals[str(k)] = float(
                        pending_totals.get(str(k), 0.0)
                    ) + float(v)

            # Train one pack if available.
            if not self._stage2_post_rollout_buffer(channel="B"):
                continue

            t_pack0 = time.perf_counter()
            selected, pack_metrics = self._stage2_pop_post_rollout_pack(channel="B")
            buf_total_len -= int(sum(int(sl) for _, _, sl in selected))

            pack_metrics = dict(pack_metrics)
            pack_metrics["time/post_rollout_pack_s"] = float(
                time.perf_counter() - t_pack0
            )

            # Attach accumulated totals to this pack, then reset.
            step_totals_pack = dict(pending_totals)
            pending_totals = {}

            is_last_pack = (seen_segments >= total_segments_target) and (
                not bool(self._stage2_post_rollout_buffer(channel="B"))
            )
            loss_pack = _train_one_pack(
                selected=selected,
                pack_metrics=pack_metrics,
                rollout_static=rollout_static,
                step_totals=step_totals_pack,
                sync_gradients=bool(is_last_pack),
            )
            loss_total = loss_pack if loss_total is None else (loss_total + loss_pack)

        th.join()

        if producer_exc:
            # Re-raise the first producer exception.
            raise producer_exc[0]

        if total_segments_target > 0 and seen_segments != total_segments_target:
            raise ValueError(
                "stage2-ab Channel-B pipeline produced unexpected segment count: "
                f"seen={seen_segments} target={total_segments_target}"
            )

        if loss_total is None:
            raise AssertionError("stage2-ab Channel-B pipelined step produced no packs")
        return loss_total

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
        if self._stage2_a_step_gs is None or int(self._stage2_a_step_gs) != gs:
            self._stage2_a_step_gs = int(gs)
            self._stage2_a_step_micro = 0
            self._stage2_a_step_raw = []

        self._stage2_a_step_micro += 1
        self._stage2_a_step_raw.extend(list(raw_micro_batch))

        try:
            gas = int(getattr(self.args, "gradient_accumulation_steps", 1) or 1)
        except Exception:
            gas = 1
        gas = max(1, int(gas))

        # Execute only on the final micro-step so the outer Trainer performs exactly one
        # optimizer.step() after we have accumulated gradients for all packs.
        if int(self._stage2_a_step_micro) < int(gas):
            return torch.tensor(0.0, device=self.model.device)

        # Reset state eagerly so exceptions do not poison the next step.
        raw_all = list(self._stage2_a_step_raw)
        self._stage2_a_step_raw = []
        self._stage2_a_step_micro = 0

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
        if self._stage2_b_step_gs is None or int(self._stage2_b_step_gs) != gs:
            self._stage2_b_step_gs = int(gs)
            self._stage2_b_step_micro = 0
            self._stage2_b_step_raw = []

        self._stage2_b_step_micro += 1
        self._stage2_b_step_raw.extend(list(raw_micro_batch))

        try:
            gas = int(getattr(self.args, "gradient_accumulation_steps", 1) or 1)
        except Exception:
            gas = 1
        gas = max(1, int(gas))

        # Execute only on the final micro-step so the outer Trainer performs exactly one
        # optimizer.step() after we have accumulated gradients for all packs.
        if int(self._stage2_b_step_micro) < int(gas):
            return torch.tensor(0.0, device=self.model.device)

        # Reset state eagerly so exceptions do not poison the next step.
        raw_all = list(self._stage2_b_step_raw)
        self._stage2_b_step_raw = []
        self._stage2_b_step_micro = 0

        # Validate expected raw sample count (best-effort; may differ under drop_last/resume).
        try:
            target_global = int(self._stage2_b_rollouts_per_step())
            target_local = (
                int(self._stage2_b_rollouts_per_rank()) if target_global > 0 else 0
            )
        except Exception:
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
