"""Runtime seed control for CoordExp-Swift training."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import torch
from transformers.trainer_utils import set_seed

from src.common.errors import RuntimeContractError


_STRICT_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "FLASH_ATTENTION_DETERMINISTIC": "1",
}
_STRICT_INITIAL_PHASES = frozenset({"pack_cache_preparation", "pipeline_entry"})


@dataclass(frozen=True)
class TrainingSeedReceipt:
    seed: int
    phase: str
    determinism_mode: str
    cuda_initialized: bool

    def to_policy_identity_dict(self) -> dict[str, Any]:
        strict = self.determinism_mode == "strict_cuda_replay_v1"
        return {
            "schema_version": 1,
            "mode": self.determinism_mode,
            "seed": self.seed,
            "deterministic_algorithms": {
                "enabled": True if strict else False,
                "managed": strict,
                "warn_only": False if strict else None,
            },
            "cudnn": {
                "benchmark": False if strict else None,
                "deterministic": True if strict else None,
                "managed": strict,
            },
            "environment": dict(_STRICT_ENVIRONMENT) if strict else {},
        }

    def to_artifact_dict(self) -> dict[str, Any]:
        policy = self.to_policy_identity_dict()
        return {
            **policy,
            "seed": self.seed,
            "phase": self.phase,
            "helper": "transformers.trainer_utils.set_seed",
            "required_environment": dict(_STRICT_ENVIRONMENT)
            if self.determinism_mode == "strict_cuda_replay_v1"
            else {},
            "observed_environment": {
                name: os.environ.get(name) for name in sorted(_STRICT_ENVIRONMENT)
            }
            if self.determinism_mode == "strict_cuda_replay_v1"
            else {},
            "cuda_initialized": self.cuda_initialized,
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
    determinism_mode: str = "legacy",
    phase: str = "pipeline_assembly",
) -> TrainingSeedReceipt:
    if determinism_mode not in {"legacy", "strict_cuda_replay_v1"}:
        raise RuntimeContractError(
            "runtime determinism mode is unsupported",
            code="runtime.determinism_mode_unsupported",
            context={"mode": determinism_mode},
        )
    phase = str(phase)
    cuda_initialized = bool(torch.cuda.is_initialized())
    if determinism_mode == "strict_cuda_replay_v1":
        _apply_strict_determinism_policy(
            phase=phase,
            cuda_initialized=cuda_initialized,
        )
    set_seed(int(seed), deterministic=False)
    return TrainingSeedReceipt(
        seed=int(seed),
        phase=phase,
        determinism_mode=determinism_mode,
        cuda_initialized=cuda_initialized,
    )


def _apply_strict_determinism_policy(
    *,
    phase: str,
    cuda_initialized: bool,
) -> None:
    conflicts = {
        name: {"expected": expected, "observed": os.environ.get(name)}
        for name, expected in sorted(_STRICT_ENVIRONMENT.items())
        if os.environ.get(name) != expected
    }
    if conflicts:
        raise RuntimeContractError(
            "strict runtime determinism environment conflicts with the resolved policy",
            code="runtime.determinism_environment_conflict",
            context={"conflicts": conflicts},
        )
    if phase == "runtime_setup_reapplied":
        _validate_strict_policy()
        return
    if phase not in _STRICT_INITIAL_PHASES:
        raise RuntimeContractError(
            "strict runtime determinism must use an admitted application phase",
            code="runtime.determinism_phase_unsupported",
            context={"phase": phase},
        )
    if cuda_initialized:
        raise RuntimeContractError(
            "strict runtime determinism must be established before CUDA initialization",
            code="runtime.determinism_cuda_already_initialized",
            context={"phase": phase},
        )

    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    _validate_strict_policy()


def _validate_strict_policy() -> None:
    observed_environment = {
        name: os.environ.get(name) for name in sorted(_STRICT_ENVIRONMENT)
    }
    drift: dict[str, Any] = {}
    if observed_environment != _STRICT_ENVIRONMENT:
        drift["environment"] = {
            "expected": dict(_STRICT_ENVIRONMENT),
            "observed": observed_environment,
        }
    deterministic_enabled = bool(torch.are_deterministic_algorithms_enabled())
    deterministic_warn_only = bool(
        torch.is_deterministic_algorithms_warn_only_enabled()
    )
    if not deterministic_enabled or deterministic_warn_only:
        drift["deterministic_algorithms"] = {
            "enabled": deterministic_enabled,
            "warn_only": deterministic_warn_only,
        }
    cudnn_deterministic = bool(torch.backends.cudnn.deterministic)
    cudnn_benchmark = bool(torch.backends.cudnn.benchmark)
    if not cudnn_deterministic or cudnn_benchmark:
        drift["cudnn"] = {
            "benchmark": cudnn_benchmark,
            "deterministic": cudnn_deterministic,
        }
    if drift:
        raise RuntimeContractError(
            "strict runtime determinism changed before runtime setup reapplication",
            code="runtime.determinism_policy_drift",
            context={"drift": drift},
        )


def _applied_before(phase: str) -> list[str]:
    if phase == "pipeline_entry":
        return [
            "cache_preflight",
            "accelerator_setup",
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
