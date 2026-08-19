"""Narrow owner for the cached micro-step runtime determinant projection.

``PACKING_CACHE_DETERMINANT_OWNERS`` binds every determinant to the complete
source that can change its semantic content.  The runtime configuration the
production micro-step constructor serializes into every cached payload used to
be owned by the whole training assembly module, which made unrelated
orchestration edits look like cache-identity changes.

This module owns exactly that projection and nothing else.  It must not import
the training facade, the cache workflow, the training session, or the low-level
cache serializer.
"""

from __future__ import annotations

from typing import Any

from src.config.models import TrainConfig


def micro_step_runtime_config_identity(config: TrainConfig) -> dict[str, Any]:
    """Return the exact runtime fields serialized into every cached micro-step.

    The three keys, their order, and their derivation are the protected
    contract: they mirror the production ``SupervisedMicroStep`` constructor
    arguments that are decided by configuration rather than by data.
    """

    return {
        "fa2_model_dtype": config.training.precision,
        "capture_fa2_branch": config.model.fa2_branch_proof == "every_forward",
        "require_fa2_branch_proof": (config.model.fa2_branch_proof == "every_forward"),
    }


__all__ = ["micro_step_runtime_config_identity"]
