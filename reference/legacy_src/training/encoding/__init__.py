"""Training encoding contract facades.

The facade intentionally stays lazy so lightweight imports such as
``src.training.encoding.model_inputs`` do not load detection/tokenization owners.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.training.encoding.example import EncodedTrainingExample
    from src.training.encoding.model_inputs import (
        BackendKeyDisposition,
        ModelInputBundle,
    )
    from src.training.encoding.view import (
        CoordinateSlot,
        EncodedDetectionView,
        EncodedObjectEntry,
    )

_LAZY_EXPORTS = {
    "BackendKeyDisposition": "src.training.encoding.model_inputs",
    "CoordinateSlot": "src.training.encoding.view",
    "EncodedDetectionView": "src.training.encoding.view",
    "EncodedObjectEntry": "src.training.encoding.view",
    "EncodedTrainingExample": "src.training.encoding.example",
    "ModelInputBundle": "src.training.encoding.model_inputs",
    "backend_key_registry": "src.training.encoding.model_inputs",
    "classify_backend_key": "src.training.encoding.model_inputs",
}


def __getattr__(name: str) -> Any:
    """Resolve facade exports on first access."""

    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value

__all__ = [
    "BackendKeyDisposition",
    "CoordinateSlot",
    "EncodedDetectionView",
    "EncodedObjectEntry",
    "EncodedTrainingExample",
    "ModelInputBundle",
    "backend_key_registry",
    "classify_backend_key",
]
