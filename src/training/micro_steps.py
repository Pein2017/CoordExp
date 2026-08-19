"""Canonical owner of the cached supervised micro-step record.

Design decision 4 of ``decompose-coordexp-swift-training-orchestration`` gives
``SupervisedMicroStep`` and its schema identity one narrow owner.  The record's
fields, order, annotations, defaults, and frozen status are a protected cache
contract: they are hashed into the packing-cache ``micro_step_schema``
determinant and pickled into every published chunk.

``src.training.supervised_trainer`` and ``src.training`` re-export the record so
existing supported and historical imports keep resolving.  New cache payloads
pickle under the canonical module path ``src.training.micro_steps``; old
immutable payloads keep their historical ``src.training.supervised_trainer``
path and stay readable through the restricted unpickler's explicit allowlist.
Old and new payload bytes are therefore intentionally different and must never
be characterized as byte-identical.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import Any

import torch

from src.losses.vocab import TokenVocabularyGroups
from src.supervision import TokenSequence


@dataclass(frozen=True)
class SupervisedMicroStep:
    pack: Any
    encoded_examples: Sequence[Any]
    position_inputs: Any
    token_sequence: TokenSequence | Any
    vocab_groups: TokenVocabularyGroups | Any
    metadata: Mapping[str, Any] | None = None
    forward_device: torch.device | str | None = None
    expected_vocab_size: int | None = None
    extra_model_kwargs: Mapping[str, Any] | None = None
    fa2_branch_evidence: Mapping[str, Any] | None = None
    fa2_model_dtype: str | None = None
    capture_fa2_branch: bool = False
    require_fa2_branch_proof: bool = False
    fa2_branch_proof_policy: str | None = None


def supervised_micro_step_schema_identity() -> dict[str, Any]:
    """Return the exact serialized-payload schema of ``SupervisedMicroStep``."""

    schema_fields: list[dict[str, Any]] = []
    for schema_field in fields(SupervisedMicroStep):
        has_default = schema_field.default is not MISSING
        schema_fields.append(
            {
                "name": schema_field.name,
                "annotation": str(schema_field.type),
                "has_default": has_default,
                "default": (
                    _json_identity_value(schema_field.default) if has_default else None
                ),
            }
        )
    return {
        "class": "SupervisedMicroStep",
        "frozen": bool(SupervisedMicroStep.__dataclass_params__.frozen),
        "fields": schema_fields,
    }


def _json_identity_value(value: Any) -> Any:
    """Project one field default with the cache's exact coercion contract.

    This mirrors ``src.training.pack_cache._json_identity_value`` rule for rule.
    It is duplicated rather than imported because ``pack_cache`` imports this
    module; the frozen ``micro_step_schema`` determinant is the proof that the
    two projections stay equal.
    """

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_identity_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        return sorted((_json_identity_value(item) for item in value), key=str)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_identity_value(item) for item in value]
    return str(value)


__all__ = [
    "SupervisedMicroStep",
    "supervised_micro_step_schema_identity",
]
