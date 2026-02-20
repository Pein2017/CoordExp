"""Stage-2 AB schedule helpers.

This module keeps deterministic channel selection and lightweight realized-ratio
tracking separate from trainer execution details.

The mixin methods are designed to operate on a partially-initialized trainer
instance (some unit tests construct the trainer via `__new__`).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Mapping, Literal


class Stage2ABSchedulerMixin:
    def _ab_cfg(self) -> Mapping[str, Any]:
        cfg = getattr(self, "stage2_ab_cfg", None)
        return cfg if isinstance(cfg, Mapping) else {}

    def _ab_get(self, key: str, default: Any) -> Any:
        cfg = self._ab_cfg()
        if key in cfg:
            return cfg[key]
        return default

    def _ab_schedule_b_ratio(self) -> float:
        cfg = self._ab_cfg()
        sched = cfg.get("schedule")
        if not isinstance(sched, Mapping):
            raise ValueError(
                "stage2_ab.schedule must be a mapping (missing typed stage2_ab config?)"
            )
        if "pattern" in sched:
            raise ValueError(
                "stage2_ab.schedule.pattern is not supported; use stage2_ab.schedule.b_ratio"
            )
        if "b_ratio" not in sched:
            raise ValueError(
                "stage2_ab.schedule.b_ratio must be provided (float in [0,1])"
            )
        raw = sched.get("b_ratio")
        try:
            b_ratio = float(raw)
        except Exception as exc:
            raise TypeError(
                "stage2_ab.schedule.b_ratio must be a float in [0,1]"
            ) from exc
        if not (0.0 <= b_ratio <= 1.0):
            raise ValueError(
                f"stage2_ab.schedule.b_ratio must be in [0,1], got {b_ratio!r}"
            )
        return b_ratio

    def _ab_channel_b_cfg(self) -> Mapping[str, Any]:
        cfg = self._ab_cfg()
        raw = cfg.get("channel_b")
        out = raw if isinstance(raw, Mapping) else {}
        if "stop_neutral" in out:
            raise ValueError(
                "stage2_ab.channel_b.stop_neutral is unsupported in the current contract; "
                "remove legacy stop-neutral keys from config."
            )
        return out

    def _ab_channel_b_get(self, key: str, default: Any) -> Any:
        cfg = self._ab_channel_b_cfg()
        if key in cfg:
            return cfg[key]
        return default

    def _stage2_b_rollouts_per_step(self) -> int:
        # Single source of truth for raw-rollout budgeting: training.effective_batch_size.
        #
        # For Stage2-AB we require loader-side exact divisibility, so the realized
        # global effective batch equals the user-requested effective_batch_size.
        try:
            per_device = int(getattr(self.args, "per_device_train_batch_size", 1) or 1)
        except Exception:
            per_device = 1
        try:
            world_size = int(getattr(self.args, "world_size", 1) or 1)
        except Exception:
            world_size = 1
        try:
            gas = int(getattr(self.args, "gradient_accumulation_steps", 1) or 1)
        except Exception:
            gas = 1

        per_device = max(1, int(per_device))
        world_size = max(1, int(world_size))
        gas = max(1, int(gas))
        return max(1, int(per_device) * int(world_size) * int(gas))

    def _stage2_b_rollouts_per_rank(self) -> int:
        """Per-train-rank raw rollouts for this optimizer step.

        The global raw rollout budget for Channel-B is `training.effective_batch_size`
        (enforced exactly for Stage2-AB). This helper splits the global target across
        ranks deterministically so that per-rank targets sum to the global budget.
        """
        total = int(self._stage2_b_rollouts_per_step())
        try:
            world_size = int(getattr(self.args, "world_size", 1) or 1)
        except Exception:
            world_size = 1
        world_size = max(1, int(world_size))

        try:
            rank = int(getattr(self.args, "process_index", 0) or 0)
        except Exception:
            rank = 0
        rank = max(0, int(rank))

        if total < world_size:
            raise ValueError(
                "training.effective_batch_size must be >= learner world_size so every train rank has at least one raw rollout. "
                f"Got effective_batch_size={total}, world_size={world_size}."
            )

        base, rem = divmod(total, world_size)
        return int(base + (1 if rank < rem else 0))

    def _stage2_record_realized_step(self, *, global_step: int, executed_b: bool) -> None:
        """Track realized B-step ratio once per optimizer step."""
        gs = int(global_step)
        last = getattr(self, "_stage2_ab_realized_last_gs", None)
        if last is not None and int(last) == gs:
            return
        hist = getattr(self, "_stage2_ab_realized_recent", None)
        if hist is None:
            hist = deque(maxlen=200)
            setattr(self, "_stage2_ab_realized_recent", hist)
        self._stage2_ab_realized_last_gs = gs
        hist.append(1 if bool(executed_b) else 0)

    def _stage2_b_ratio_realized(self) -> float:
        hist = getattr(self, "_stage2_ab_realized_recent", None)
        if not hist:
            return 0.0
        try:
            return float(sum(int(x) for x in hist)) / float(len(hist))
        except (TypeError, ValueError, ZeroDivisionError):
            return 0.0

    def _stage2_channel_for_step(self, global_step: int) -> Literal["A", "B"]:
        if isinstance(
            getattr(self, "_stage2_channel_override", None),
            str,
        ) and getattr(self, "_stage2_channel_override", None) in {
            "A",
            "B",
        }:
            return "A" if getattr(self, "_stage2_channel_override") == "A" else "B"

        b_ratio = float(self._ab_schedule_b_ratio())
        s = int(global_step)

        if b_ratio <= 0.0:
            return "A"
        if b_ratio >= 1.0:
            return "B"

        # Bresenham-style deterministic ratio schedule: select B iff the running
        # floor count increases at this step.
        a = math.floor(float(s + 1) * float(b_ratio))
        b = math.floor(float(s) * float(b_ratio))
        return "B" if a > b else "A"

