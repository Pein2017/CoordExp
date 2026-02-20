from __future__ import annotations

from dataclasses import dataclass
from collections.abc import MutableMapping
from typing import Any


# NOTE: These keys are emitted by collators and consumed by trainer-side diagnostics.
# They MUST NOT be forwarded into model(**inputs).
DATASET_LABELS_KEY = "dataset_labels"
DATASET_SEGMENTS_KEY = "dataset_segments"
PACK_NUM_SAMPLES_KEY = "pack_num_samples"
TOKEN_TYPES_KEY = "token_types"
INSTABILITY_META_JSON_KEY = "instability_meta_json"

BATCH_EXTRAS_KEYS: tuple[str, ...] = (
    DATASET_LABELS_KEY,
    DATASET_SEGMENTS_KEY,
    PACK_NUM_SAMPLES_KEY,
    TOKEN_TYPES_KEY,
    INSTABILITY_META_JSON_KEY,
)


@dataclass
class BatchExtras:
    """Auxiliary batch fields produced by collators.

    These are debug/diagnostic-only fields. Trainer mixins should pop them from the
    input dict before calling model(**inputs), but keep them available for logging.
    """

    dataset_labels: Any = None
    dataset_segments: Any = None
    pack_num_samples: Any = None
    token_types: Any = None
    instability_meta_json: Any = None


_STASH_ATTR = "_coordexp_batch_extras"


def pop_batch_extras(inputs: MutableMapping[str, Any]) -> BatchExtras:
    """Remove known batch-extras keys from `inputs` and return them."""

    return BatchExtras(
        dataset_labels=inputs.pop(DATASET_LABELS_KEY, None),
        dataset_segments=inputs.pop(DATASET_SEGMENTS_KEY, None),
        pack_num_samples=inputs.pop(PACK_NUM_SAMPLES_KEY, None),
        token_types=inputs.pop(TOKEN_TYPES_KEY, None),
        instability_meta_json=inputs.pop(INSTABILITY_META_JSON_KEY, None),
    )


def stash_batch_extras(trainer: Any, extras: BatchExtras) -> None:
    """Attach extras to the trainer instance for later mixins."""

    try:
        setattr(trainer, _STASH_ATTR, extras)
    except (AttributeError, TypeError):
        # Best-effort only; never block training.
        return

    # Back-compat: historically, pack_num_samples was stashed under this name.
    try:
        setattr(trainer, "_coordexp_pack_num_samples", extras.pack_num_samples)
    except (AttributeError, TypeError):
        return


def pop_and_stash_batch_extras(trainer: Any, inputs: Any) -> BatchExtras:
    """Pop extras from inputs (if dict-like), stash on trainer, and return them."""

    if not isinstance(inputs, MutableMapping):
        extras = BatchExtras()
        stash_batch_extras(trainer, extras)
        return extras

    extras = pop_batch_extras(inputs)
    stash_batch_extras(trainer, extras)
    return extras


def maybe_pop_and_stash_batch_extras(trainer: Any, inputs: Any) -> BatchExtras:
    """Get the already-stashed extras, or pop+stash them once.

    This is safe to call from multiple mixins without clobbering the stash.
    """

    existing = getattr(trainer, _STASH_ATTR, None)
    if isinstance(existing, BatchExtras):
        return existing
    return pop_and_stash_batch_extras(trainer, inputs)


def get_stashed_batch_extras(trainer: Any) -> BatchExtras:
    extras = getattr(trainer, _STASH_ATTR, None)
    return extras if isinstance(extras, BatchExtras) else BatchExtras()
