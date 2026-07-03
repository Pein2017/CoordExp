"""Runtime seed control for CoordExp-Swift training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from transformers.trainer_utils import set_seed


@dataclass(frozen=True)
class TrainingSeedReceipt:
    seed: int
    phase: str
    deterministic_algorithms: bool

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "phase": self.phase,
            "helper": "transformers.trainer_utils.set_seed",
            "deterministic_algorithms": self.deterministic_algorithms,
            "surfaces": [
                "python.random",
                "numpy",
                "torch.cpu",
                "torch.cuda_all",
                "torch.other_supported_devices",
            ],
            "applied_before": _applied_before(self.phase),
        }


def seed_training_runtime(
    seed: int,
    *,
    deterministic: bool = False,
    phase: str = "pipeline_assembly",
) -> TrainingSeedReceipt:
    set_seed(int(seed), deterministic=deterministic)
    return TrainingSeedReceipt(
        seed=int(seed),
        phase=str(phase),
        deterministic_algorithms=bool(deterministic),
    )


def _applied_before(phase: str) -> list[str]:
    if phase == "pipeline_assembly":
        return [
            "qwen_model_load",
            "adapter_setup",
            "special_token_embedding_setup",
            "optimizer_setup",
            "runtime_setup",
        ]
    if phase == "runtime_setup_reapplied":
        return [
            "model_device_move",
            "backend_prepare",
            "first_forward",
        ]
    return []


__all__ = ["TrainingSeedReceipt", "seed_training_runtime"]
