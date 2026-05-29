"""Stage-2 rollout-correction runtime budget helpers.

The mixin methods are designed to operate on a partially-initialized trainer
instance (some unit tests construct the trainer via `__new__`).
"""

from __future__ import annotations

from collections import deque
from typing import Any, Mapping


class RolloutCorrectionRuntimeBudgetMixin:
    def _rollout_correction_cfg(self) -> Mapping[str, Any]:
        cfg = getattr(self, "stage2_rollout_correction_cfg", None)
        return cfg if isinstance(cfg, Mapping) else {}

    def _rollout_correction_get(self, key: str, default: Any) -> Any:
        cfg = self._rollout_correction_cfg()
        if key in cfg:
            return cfg[key]
        return default

    def _rollout_correction_runtime_cfg(self) -> Mapping[str, Any]:
        cfg = self._rollout_correction_cfg()
        raw = cfg.get("correction")
        out = raw if isinstance(raw, Mapping) else {}
        if "stop_neutral" in out:
            raise ValueError(
                "stage2_rollout_correction.correction.stop_neutral is unsupported; "
                "remove legacy stop-neutral keys from config."
            )
        return out

    def _rollout_correction_runtime_get(self, key: str, default: Any) -> Any:
        cfg = self._rollout_correction_runtime_cfg()
        if key in cfg:
            return cfg[key]
        if not isinstance(key, str) or "." not in key:
            return default
        cur: Any = cfg
        for part in key.split("."):
            if not part or not isinstance(cur, Mapping) or part not in cur:
                return default
            cur = cur[part]
        return cur

    # Internal compatibility while rollout-correction target construction is still
    # hosted in the old module path.
    def _rollout_correction_cfg_get(self, key: str, default: Any) -> Any:
        return self._rollout_correction_runtime_get(key, default)

    def _stage2_rollouts_per_step(self) -> int:
        # Single source of truth for raw-rollout budgeting: training.effective_batch_size.
        #
        # For rollout-correction we require loader-side exact divisibility, so the realized
        # global effective batch equals the user-requested effective_batch_size.
        try:
            per_device = int(getattr(self.args, "per_device_train_batch_size", 1) or 1)
        except (TypeError, ValueError):
            per_device = 1
        try:
            world_size = int(getattr(self.args, "world_size", 1) or 1)
        except (TypeError, ValueError):
            world_size = 1
        try:
            gas = int(getattr(self.args, "gradient_accumulation_steps", 1) or 1)
        except (TypeError, ValueError):
            gas = 1

        per_device = max(1, int(per_device))
        world_size = max(1, int(world_size))
        gas = max(1, int(gas))
        return max(1, int(per_device) * int(world_size) * int(gas))

    def _stage2_rollouts_per_rank(self) -> int:
        """Per-train-rank raw rollouts for this optimizer step.

        The global raw rollout budget is `training.effective_batch_size`
        (enforced exactly for rollout-correction). This helper splits the global target across
        ranks deterministically so that per-rank targets sum to the global budget.
        """
        total = int(self._stage2_rollouts_per_step())
        try:
            world_size = int(getattr(self.args, "world_size", 1) or 1)
        except (TypeError, ValueError):
            world_size = 1
        world_size = max(1, int(world_size))

        try:
            rank = int(getattr(self.args, "process_index", 0) or 0)
        except (TypeError, ValueError):
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
        """Track optimizer steps with rollout-correction work."""
        gs = int(global_step)
        last = getattr(self, "_stage2_rollout_correction_realized_last_gs", None)
        if last is not None and int(last) == gs:
            return
        hist = getattr(self, "_stage2_rollout_correction_realized_recent", None)
        if hist is None:
            hist = deque(maxlen=200)
            setattr(self, "_stage2_rollout_correction_realized_recent", hist)
        self._stage2_rollout_correction_realized_last_gs = gs
        hist.append(1 if bool(executed_b) else 0)

    def _stage2_rollout_correction_realized(self) -> float:
        hist = getattr(self, "_stage2_rollout_correction_realized_recent", None)
        if not hist:
            return 0.0
        try:
            return float(sum(int(x) for x in hist)) / float(len(hist))
        except (TypeError, ValueError, ZeroDivisionError):
            return 0.0
