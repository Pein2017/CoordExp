"""Stage-2 AB async actor/learner queue manager.

This module contains the async Channel-B readiness queue + version-window gating.

The mixin methods are designed to operate on a partially-initialized trainer
instance (some unit tests construct the trainer via `__new__`).
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Mapping, Optional

import torch


logger = logging.getLogger(__name__)


@dataclass
class Stage2AsyncReadyPack:
    ver: int
    batch: Dict[str, Any]


class Stage2ABAsyncQueueManagerMixin:
    def _stage2_async_cfg(self) -> Mapping[str, Any]:
        raw = self._ab_channel_b_cfg().get("async")
        return raw if isinstance(raw, Mapping) else {}

    def _stage2_async_get_int(self, key: str, default: int) -> int:
        cfg = self._stage2_async_cfg()
        raw = cfg.get(key, default)
        try:
            v = int(raw)
        except Exception:
            v = int(default)
        return int(v)

    def _stage2_async_queue_limit(self) -> int:
        return max(1, int(self._stage2_async_get_int("queue_limit", 8)))

    def _stage2_async_version_window(self) -> int:
        return max(0, int(self._stage2_async_get_int("version_window", 2)))

    def _stage2_async_sync_every_steps(self) -> int:
        return max(1, int(self._stage2_async_get_int("sync_every_steps", 1)))

    def _stage2_async_prefetch_target_packs(self) -> int:
        limit = int(self._stage2_async_queue_limit())
        target = max(1, int(self._stage2_async_get_int("prefetch_target_packs", 2)))
        return int(min(limit, target))

    def _vllm_server_infer_guard(self):
        # Only used for async actor-learner; other modes keep the default behavior.
        try:
            if self._stage2_b_step_mode() != "async":
                return contextlib.nullcontext()
        except Exception:
            return contextlib.nullcontext()

        cv = getattr(self, "_stage2_async_infer_cv", None)
        if cv is None:
            return contextlib.nullcontext()

        @contextlib.contextmanager
        def _cm():
            with cv:
                while bool(getattr(self, "_stage2_async_sync_in_progress", False)):
                    cv.wait(timeout=0.1)

                # Snapshot the version observed for the rollout inference that follows.
                # The async prefetch thread reads this to tag ready packs with the
                # correct policy version and enforce version-pure packs.
                try:
                    self._stage2_async_last_infer_ver = int(
                        getattr(self, "_stage2_async_ver", 0) or 0
                    )
                except Exception:
                    self._stage2_async_last_infer_ver = 0

                try:
                    self._stage2_async_infer_inflight += 1
                except Exception:
                    self._stage2_async_infer_inflight = 1
            try:
                yield
            finally:
                with cv:
                    try:
                        self._stage2_async_infer_inflight -= 1
                    except Exception:
                        self._stage2_async_infer_inflight = 0
                    if int(getattr(self, "_stage2_async_infer_inflight", 0) or 0) < 0:
                        self._stage2_async_infer_inflight = 0
                    cv.notify_all()

        return _cm()

    def _stage2_async_pause_infer_for_sync(self) -> None:
        cv = getattr(self, "_stage2_async_infer_cv", None)
        if cv is None:
            return
        with cv:
            self._stage2_async_sync_in_progress = True
            while int(getattr(self, "_stage2_async_infer_inflight", 0) or 0) > 0:
                cv.wait(timeout=0.1)

    def _stage2_async_resume_infer_after_sync(self) -> None:
        cv = getattr(self, "_stage2_async_infer_cv", None)
        if cv is None:
            return
        with cv:
            self._stage2_async_sync_in_progress = False
            cv.notify_all()

    def _stage2_async_validate_requirements(self) -> None:
        backend = (
            str(getattr(self, "_rollout_backend", lambda: "")()).strip().lower()
        )
        mode = str(getattr(self, "_vllm_mode", lambda: "")()).strip().lower()

        if backend != "vllm" or mode != "server":
            raise ValueError(
                "stage2_ab.channel_b.mode='async' requires server rollouts with dedicated GPUs. "
                "Set custom.extra.rollout_matching.rollout_backend=vllm and custom.extra.rollout_matching.vllm.mode=server. "
                f"Got rollout_backend={backend!r}, vllm.mode={mode!r}."
            )

        vcfg = None
        try:
            vcfg = getattr(self, "rollout_matching_cfg", None)
        except Exception:
            vcfg = None

        # Enforce full sync for async mode (robust default and required under multi-GPU learners).
        try:
            vllm_cfg = (vcfg or {}).get("vllm") if isinstance(vcfg, Mapping) else None
        except Exception:
            vllm_cfg = None

        sync_mode = "full"
        if isinstance(vllm_cfg, Mapping):
            sync_raw = vllm_cfg.get("sync")
            if isinstance(sync_raw, Mapping):
                sync_mode = (
                    str(sync_raw.get("mode", "full") or "full").strip().lower()
                )

        if sync_mode != "full":
            raise ValueError(
                "stage2_ab.channel_b.mode='async' requires custom.extra.rollout_matching.vllm.sync.mode=full "
                f"(robust, DDP-safe sync). Got sync.mode={sync_mode!r}."
            )

        # Async mode relies on one packed fwd/bwd per micro-step; require window packing.
        if not bool(self._packing_enabled()):
            raise ValueError(
                "stage2_ab.channel_b.mode='async' requires training.packing=true (post-rollout packing)."
            )
        if str(self._post_rollout_pack_scope()).strip().lower() != "window":
            raise ValueError(
                "stage2_ab.channel_b.mode='async' requires custom.extra.rollout_matching.post_rollout_pack_scope='window' "
                "so each micro-step runs exactly one packed forward/backward per rank."
            )

        _rank, world_size, _dist = self._dist_info()
        if int(world_size) > 1 and self._stage2_b_step_mode() == "step":
            raise ValueError(
                "stage2_ab.channel_b.mode='step' is not supported under multi-GPU learner (DDP) (world_size>1). "
                "Use channel_b.mode='async' (recommended) or channel_b.mode='micro'."
            )

        # Async feasibility requires each rank to have >= GAS packs of the same eligible version.
        try:
            gas = int(getattr(self.args, "gradient_accumulation_steps", 1) or 1)
        except Exception:
            gas = 1
        gas = max(1, int(gas))

        qlim = int(self._stage2_async_queue_limit())
        target = int(self._stage2_async_prefetch_target_packs())
        if int(qlim) < int(gas):
            raise ValueError(
                "stage2_ab.channel_b.async.queue_limit must be >= gradient_accumulation_steps "
                f"(queue_limit={int(qlim)}, gas={int(gas)}). Otherwise Channel-B can never pass the feasibility gate."
            )
        if int(target) < int(gas):
            raise ValueError(
                "stage2_ab.channel_b.async.prefetch_target_packs must be >= gradient_accumulation_steps "
                f"(prefetch_target_packs={int(target)}, gas={int(gas)}). Otherwise Channel-B will starve and always fall back to A."
            )

    def _stage2_async_ready_depth_locked(self) -> int:
        return int(len(self._stage2_async_ready))

    def _stage2_async_prune_stale_locked(self) -> None:
        """Drop stale packs from the ready queue.

        NOTE: Do not assume the deque is monotonic by `ver`. The async prefetch loop
        can (rarely) append older-version packs after newer ones if current-version
        rollout preparation fails and only older buffered segments remain.
        """

        cur_ver = int(getattr(self, "_stage2_async_ver", 0) or 0)
        window = int(self._stage2_async_version_window())
        min_ver = int(cur_ver - window)

        if not self._stage2_async_ready:
            return

        kept: Deque[Stage2AsyncReadyPack] = deque()
        dropped = 0
        for p in self._stage2_async_ready:
            if int(p.ver) >= int(min_ver):
                kept.append(p)
            else:
                dropped += 1

        if dropped <= 0:
            return

        self._stage2_async_ready.clear()
        self._stage2_async_ready.extend(kept)
        self._stage2_async_drop_stale_total += int(dropped)

    def _stage2_async_ready_depth_fresh(self) -> int:
        with self._stage2_async_ready_lock:
            self._stage2_async_prune_stale_locked()
            return self._stage2_async_ready_depth_locked()

    def _stage2_async_push_ready(self, pack: Stage2AsyncReadyPack) -> None:
        with self._stage2_async_ready_lock:
            self._stage2_async_prune_stale_locked()
            limit = int(self._stage2_async_queue_limit())
            while (
                int(len(self._stage2_async_ready)) >= int(limit)
                and self._stage2_async_ready
            ):
                self._stage2_async_ready.popleft()
                self._stage2_async_drop_oldest_total += 1
            self._stage2_async_ready.append(pack)

    def _stage2_async_pop_ready(self) -> Stage2AsyncReadyPack:
        with self._stage2_async_ready_lock:
            self._stage2_async_prune_stale_locked()
            if not self._stage2_async_ready:
                cur_ver = int(getattr(self, "_stage2_async_ver", 0) or 0)
                window = int(self._stage2_async_version_window())
                raise RuntimeError(
                    "stage2-ab async ready-pack queue is empty "
                    f"(cur_ver={cur_ver}, window={window})"
                )
            return self._stage2_async_ready.popleft()

    def _stage2_async_pop_ready_for_ver(self, *, ver: int) -> Stage2AsyncReadyPack:
        """Pop the oldest ready pack matching `ver`.

        This enforces per-optimizer-step version purity under async Channel-B.
        """
        target_ver = int(ver)
        with self._stage2_async_ready_lock:
            self._stage2_async_prune_stale_locked()
            if not self._stage2_async_ready:
                cur_ver = int(getattr(self, "_stage2_async_ver", 0) or 0)
                window = int(self._stage2_async_version_window())
                raise RuntimeError(
                    "stage2-ab async ready-pack queue is empty "
                    f"(requested_ver={target_ver}, cur_ver={cur_ver}, window={window})"
                )

            out: Optional[Stage2AsyncReadyPack] = None
            new_q: Deque[Stage2AsyncReadyPack] = deque()
            while self._stage2_async_ready:
                item = self._stage2_async_ready.popleft()
                if out is None and int(getattr(item, "ver", 0) or 0) == target_ver:
                    out = item
                    break
                new_q.append(item)

            # Preserve relative order of the remaining items.
            while self._stage2_async_ready:
                new_q.append(self._stage2_async_ready.popleft())
            self._stage2_async_ready = new_q

            if out is None:
                cur_ver = int(getattr(self, "_stage2_async_ver", 0) or 0)
                window = int(self._stage2_async_version_window())
                min_ver = int(cur_ver - window)

                by_ver: Dict[int, int] = {}
                for p in new_q:
                    try:
                        pv = int(getattr(p, "ver", 0) or 0)
                    except Exception:
                        pv = 0
                    by_ver[int(pv)] = int(by_ver.get(int(pv), 0)) + 1

                raise RuntimeError(
                    "stage2-ab async ready-pack queue has no eligible pack "
                    f"for ver={target_ver} (cur_ver={cur_ver}, window={window}, "
                    f"min_ver={min_ver}, queue_depth={len(new_q)}, by_ver={by_ver})"
                )
            return out

    def _stage2_async_build_raw_dataloader(self):
        """Build a num_workers=0 dataloader for async prefetch (thread-safe)."""
        ds = getattr(self, "train_dataset", None)
        if ds is None:
            raise RuntimeError("trainer.train_dataset is required for async prefetch")

        rank, world_size, dist = self._dist_info()

        seed = 0
        try:
            seed = int(getattr(self.args, "seed", 0) or 0)
        except Exception:
            seed = 0

        try:
            from torch.utils.data import DataLoader
        except Exception as exc:
            raise RuntimeError("torch.utils.data.DataLoader is required") from exc

        sampler = None
        shuffle = True
        if (
            dist is not None
            and dist.is_available()
            and dist.is_initialized()
            and int(world_size) > 1
        ):
            try:
                from torch.utils.data.distributed import DistributedSampler

                sampler = DistributedSampler(
                    ds,
                    num_replicas=int(world_size),
                    rank=int(rank),
                    shuffle=True,
                    seed=int(seed),
                    drop_last=False,
                )
                shuffle = False
            except Exception as exc:
                raise RuntimeError(
                    "Failed to create DistributedSampler for async prefetch"
                ) from exc

        def _identity_collate(batch):
            return batch

        dl = DataLoader(
            ds,
            batch_size=1,
            shuffle=bool(shuffle),
            sampler=sampler,
            collate_fn=_identity_collate,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )
        return dl

    def _stage2_async_next_raw_sample(self) -> Mapping[str, Any]:
        if self._stage2_async_raw_dataloader is None:
            self._stage2_async_raw_dataloader = self._stage2_async_build_raw_dataloader()
            self._stage2_async_raw_iter = iter(self._stage2_async_raw_dataloader)
            self._stage2_async_raw_epoch = 0

        try:
            batch = next(self._stage2_async_raw_iter)
        except StopIteration:
            self._stage2_async_raw_epoch += 1
            dl = self._stage2_async_raw_dataloader
            sampler = getattr(dl, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                try:
                    sampler.set_epoch(int(self._stage2_async_raw_epoch))
                except Exception:
                    pass
            self._stage2_async_raw_iter = iter(dl)
            batch = next(self._stage2_async_raw_iter)

        if not isinstance(batch, list) or len(batch) != 1:
            raise RuntimeError(
                "async prefetch expected batch_size=1 identity-collated list"
            )
        sample = batch[0]
        if not isinstance(sample, Mapping):
            raise RuntimeError("async prefetch expected dataset samples to be mappings")
        return sample

    def _stage2_async_prefetch_loop(self) -> None:
        local_segments: List[Tuple[Dict[str, Any], Dict[str, Any], int, int]] = []
        bm_by_ver: Dict[int, Mapping[str, Any]] = {}
        logged_error = False

        def _count_segments_for_ver(ver: int) -> int:
            return sum(
                1
                for _enc, _meta, _sl, _ver in local_segments
                if int(_ver) == int(ver)
            )

        while not self._stage2_async_stop.is_set():
            try:
                try:
                    self._stage2_async_prefetch_iter_total += 1
                    self._stage2_async_prefetch_state = 1
                    self._stage2_async_prefetch_last_progress_ts = float(time.time())
                except Exception:
                    pass
                target = int(self._stage2_async_prefetch_target_packs())
                depth = int(self._stage2_async_ready_depth_fresh())
                if depth >= target:
                    try:
                        self._stage2_async_prefetch_state = 10
                        self._stage2_async_prefetch_last_progress_ts = float(time.time())
                    except Exception:
                        pass
                    time.sleep(0.02)
                    continue

                try:
                    self._stage2_async_prefetch_state = 2
                    self._stage2_async_prefetch_last_progress_ts = float(time.time())
                except Exception:
                    pass

                # Fill segments buffer if needed.
                rollout_gen_bs = 1
                try:
                    rollout_gen_bs = int(self._cfg("rollout_generate_batch_size", 1) or 1)
                except Exception:
                    rollout_gen_bs = 1
                rollout_gen_bs = max(1, int(rollout_gen_bs))

                cur_ver = int(getattr(self, "_stage2_async_ver", 0) or 0)

                while _count_segments_for_ver(cur_ver) < max(1, rollout_gen_bs):
                    try:
                        self._stage2_async_prefetch_state = 3
                        self._stage2_async_prefetch_last_progress_ts = float(time.time())
                    except Exception:
                        pass

                    raw_samples: List[Mapping[str, Any]] = []
                    for _ in range(int(rollout_gen_bs)):
                        raw_samples.append(self._stage2_async_next_raw_sample())

                    # Prevent server sync (DDP collectives) from being triggered in the background thread.
                    setattr(self, "_stage2_async_skip_vllm_server_sync", True)
                    try:
                        with self._vllm_server_infer_guard():
                            ver_used = int(getattr(self, "_stage2_async_ver", 0) or 0)

                            try:
                                self._stage2_async_prefetch_state = 4
                                self._stage2_async_prefetch_last_progress_ts = float(time.time())
                            except Exception:
                                pass
                            segs, bm = self._prepare_batch_inputs_b(
                                raw_samples,
                                _segments_only=True,
                            )
                            try:
                                self._stage2_async_prefetch_state = 5
                                self._stage2_async_prefetch_last_progress_ts = float(time.time())
                            except Exception:
                                pass

                        if isinstance(bm, Mapping):
                            bm_by_ver[int(ver_used)] = bm
                    finally:
                        setattr(self, "_stage2_async_skip_vllm_server_sync", False)

                    if not isinstance(segs, list) or not segs:
                        break

                    for enc, meta, sl in list(segs):
                        local_segments.append((enc, meta, int(sl), ver_used))

                    # If the version changed while we were generating, refresh our target.
                    cur_ver = int(getattr(self, "_stage2_async_ver", 0) or 0)

                if not local_segments:
                    try:
                        self._stage2_async_prefetch_state = 11
                        self._stage2_async_prefetch_last_progress_ts = float(time.time())
                    except Exception:
                        pass
                    time.sleep(0.02)
                    continue

                # Choose a single version to build the next pack from (version-pure packs).
                cur_ver = int(getattr(self, "_stage2_async_ver", 0) or 0)

                # Prune local stale segments beyond the freshness window so we never build
                # an out-of-window pack and don't retain unbounded version tails.
                window = int(self._stage2_async_version_window())
                min_ver = int(cur_ver - window)
                if window >= 0 and local_segments:
                    local_segments = [
                        s
                        for s in local_segments
                        if int(s[3]) >= int(min_ver)
                    ]
                    for vv in list(bm_by_ver.keys()):
                        if int(vv) < int(min_ver):
                            bm_by_ver.pop(int(vv), None)

                if not local_segments:
                    try:
                        self._stage2_async_prefetch_state = 11
                        self._stage2_async_prefetch_last_progress_ts = float(time.time())
                    except Exception:
                        pass
                    time.sleep(0.02)
                    continue

                if any(
                    int(v) == int(cur_ver) for _enc, _meta, _sl, v in local_segments
                ):
                    pack_ver = int(cur_ver)
                else:
                    pack_ver = int(
                        max(int(v) for _enc, _meta, _sl, v in local_segments)
                    )

                packing_length = int(self._packing_length())
                candidate_idx = [
                    i
                    for i, (_enc, _meta, _sl, v) in enumerate(local_segments)
                    if int(v) == int(pack_ver)
                ]
                encoded_lens = [int(local_segments[i][2]) for i in candidate_idx]
                selected_rel = self._select_post_rollout_segment_indices(
                    encoded_lens,
                    packing_length,
                )
                if not selected_rel:
                    raise RuntimeError("async prefetch selected an empty pack")

                selected_idx = [int(candidate_idx[i]) for i in selected_rel]
                selected = [local_segments[i] for i in selected_idx]
                for i in sorted(selected_idx, reverse=True):
                    local_segments.pop(int(i))

                try:
                    self._stage2_async_prefetch_state = 6
                    self._stage2_async_prefetch_last_progress_ts = float(time.time())
                except Exception:
                    pass

                # Collate on CPU; the learner moves to GPU in training_step.
                with self._template_packing_enabled():
                    packed = self.template.data_collator(
                        [enc for enc, _, _, _ in selected]
                    )

                batch: Dict[str, Any] = dict(packed)
                self._assert_single_packed_forward(
                    batch,
                    where="stage2_ab/async_prefetch",
                )
                batch["_rollout_matching_meta"] = [m for _, m, _, _ in selected]
                batch["_stage2_ab_channel"] = "B"

                # Attach async telemetry for this pack (merge base rollout/match metrics).
                bm_base = bm_by_ver.get(int(pack_ver), {})
                bm2: Dict[str, Any] = (
                    dict(bm_base) if isinstance(bm_base, Mapping) else {}
                )
                bm2["stage2_ab/async/ver"] = float(pack_ver)

                # Emit packing telemetry so B steps are comparable to Channel-A window packing.
                try:
                    sel_total = int(sum(int(sl) for _enc, _meta, sl, _v in selected))
                    fill = (
                        float(sel_total) / float(packing_length)
                        if packing_length > 0
                        else 0.0
                    )
                    buf_same_ver = int(
                        sum(
                            1
                            for _enc, _meta, _sl, v in local_segments
                            if int(v) == int(pack_ver)
                        )
                    )
                    bm2.update(
                        {
                            "packing/post_rollout_fill": float(fill),
                            "packing/post_rollout_selected_total_len": float(sel_total),
                            "packing/post_rollout_segments": float(len(selected)),
                            "packing/post_rollout_buffer": float(buf_same_ver),
                        }
                    )
                except Exception:
                    pass

                try:
                    bm2["stage2_ab/async/queue_target"] = float(target)
                    bm2["stage2_ab/async/queue_depth"] = float(
                        self._stage2_async_ready_depth_fresh()
                    )
                    bm2["stage2_ab/async/drop_stale_total"] = float(
                        self._stage2_async_drop_stale_total
                    )
                    bm2["stage2_ab/async/drop_oldest_total"] = float(
                        self._stage2_async_drop_oldest_total
                    )
                except Exception:
                    pass
                self._merge_rollout_matching_batch_metrics(batch, bm2)

                self._stage2_async_push_ready(
                    Stage2AsyncReadyPack(ver=int(pack_ver), batch=batch)
                )
                try:
                    self._stage2_async_prefetch_state = 7
                    self._stage2_async_prefetch_last_progress_ts = float(time.time())
                except Exception:
                    pass
                try:
                    self._stage2_async_prefetch_success_total += 1
                except Exception:
                    pass
            except Exception as exc:
                try:
                    self._stage2_async_prefetch_state = 99
                    self._stage2_async_prefetch_last_progress_ts = float(time.time())
                except Exception:
                    pass

                # During interpreter teardown (e.g. torchrun shutdown), background threads may
                # see "cannot schedule new futures after interpreter shutdown" and similar.
                # Avoid logging/retrying in that state; just exit the daemon thread.
                try:
                    import sys

                    if bool(getattr(sys, "is_finalizing", lambda: False)()):
                        return
                except Exception:
                    pass
                if bool(self._stage2_async_stop.is_set()):
                    return

                try:
                    self._stage2_async_prefetch_fail_total += 1
                except Exception:
                    pass

                if not logged_error:
                    logger.exception("Stage2-AB async prefetch failed (will retry)")
                    logged_error = True
                time.sleep(0.1)

    def _stage2_async_ensure_prefetch_thread(self) -> None:
        if self._stage2_async_prefetch_thread is not None:
            return
        th = threading.Thread(
            target=self._stage2_async_prefetch_loop,
            name="stage2_ab_async_prefetch",
            daemon=True,
        )
        self._stage2_async_prefetch_thread = th
        th.start()

    def _stage2_async_should_sync(self, global_step: int) -> bool:
        every = int(self._stage2_async_sync_every_steps())
        if every <= 1:
            return True
        last = getattr(self, "_stage2_async_last_synced_gs", None)
        if last is None:
            return True
        return int(global_step) - int(last) >= int(every)

    def _stage2_async_maybe_sync_server(self, global_step: int) -> None:
        rank, world_size, dist = self._dist_info()

        do_sync = bool(self._stage2_async_should_sync(int(global_step)))
        if (
            dist is not None
            and dist.is_available()
            and dist.is_initialized()
            and int(world_size) > 1
        ):
            flag = [bool(do_sync)] if int(rank) == 0 else [False]
            dist.broadcast_object_list(flag, src=0)
            do_sync = bool(flag[0])

        if not bool(do_sync):
            return

        # Fence background HTTP infer against rank0 sync.
        cur_ver = int(getattr(self, "_stage2_async_ver", 0) or 0)
        self._stage2_async_pause_infer_for_sync()
        try:
            # All ranks participate in sync barriers (DDP safety). Rank0 does the weight push.
            try:
                self._sync_vllm_server_rollout_model_if_needed()
            except Exception as exc:
                raise RuntimeError(
                    "stage2-ab async rollout-model sync failed "
                    f"(global_step={int(global_step)}, cur_ver={cur_ver}, "
                    f"rank={int(rank)}, world_size={int(world_size)})"
                ) from exc
        finally:
            self._stage2_async_resume_infer_after_sync()

        # Advance ver (rank0 authority under DDP).
        if (
            dist is not None
            and dist.is_available()
            and dist.is_initialized()
            and int(world_size) > 1
        ):
            ver = [int(getattr(self, "_stage2_async_ver", 0) or 0)]
            if int(rank) == 0:
                ver = [int(ver[0]) + 1]
                self._stage2_async_last_synced_gs = int(global_step)
            dist.broadcast_object_list(ver, src=0)
            self._stage2_async_ver = int(ver[0])
        else:
            self._stage2_async_ver = int(getattr(self, "_stage2_async_ver", 0) or 0) + 1
            self._stage2_async_last_synced_gs = int(global_step)

    def _stage2_async_decide_step_kind(self, *, global_step: int, policy_wants_b: bool) -> str:
        rank, world_size, dist = self._dist_info()
        gas = 1
        try:
            gas = int(getattr(self.args, "gradient_accumulation_steps", 1) or 1)
        except Exception:
            gas = 1
        gas = max(1, int(gas))

        cur_ver = int(getattr(self, "_stage2_async_ver", 0) or 0)
        window = int(self._stage2_async_version_window())

        step_kind = "A"
        step_ver: Optional[int] = None

        if bool(policy_wants_b):
            # Version-pure feasibility gate: require >= GAS packs of a single eligible `ver`.
            for cand_ver in range(int(cur_ver), int(cur_ver) - int(window) - 1, -1):
                local_count = 0
                with self._stage2_async_ready_lock:
                    self._stage2_async_prune_stale_locked()
                    for p in self._stage2_async_ready:
                        try:
                            if int(getattr(p, "ver", 0) or 0) == int(cand_ver):
                                local_count += 1
                        except Exception:
                            continue

                min_count = int(local_count)
                if (
                    dist is not None
                    and dist.is_available()
                    and dist.is_initialized()
                    and int(world_size) > 1
                ):
                    t = torch.tensor(
                        [int(local_count)],
                        device=self.model.device,
                        dtype=torch.int64,
                    )
                    dist.all_reduce(t, op=dist.ReduceOp.MIN)
                    min_count = int(t.detach().cpu().item())

                if int(min_count) >= int(gas):
                    step_kind = "B"
                    step_ver = int(cand_ver)
                    break

        # Rank0 authority: broadcast final step decision.
        if (
            dist is not None
            and dist.is_available()
            and dist.is_initialized()
            and int(world_size) > 1
        ):
            obj = (
                [{"kind": str(step_kind), "ver": step_ver}] if int(rank) == 0 else [{}]
            )
            dist.broadcast_object_list(obj, src=0)
            payload = obj[0] if isinstance(obj[0], Mapping) else {}
            step_kind = str(payload.get("kind", "A"))
            v = payload.get("ver", None)
            try:
                step_ver = int(v) if v is not None else None
            except Exception:
                step_ver = None

        self._stage2_async_step_ver = step_ver
        return "B" if step_kind == "B" else "A"
