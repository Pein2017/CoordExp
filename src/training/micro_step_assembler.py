"""Narrow owner for the assembly of every serialized cached micro-step.

``PACKING_CACHE_DETERMINANT_OWNERS`` binds every determinant to the complete
source that can change its semantic content.  The per-pack composition that
turns realized stage outputs plus configuration into a ``SupervisedMicroStep``
-- the metadata dictionary, the position and token-sequence construction calls,
and the config/identity-derived constructor arguments -- used to live inside
the cache workflow module, which is not a determinant owner: an edit there
would have changed cached payload bytes while still hitting an old semantic
fingerprint.

This module owns exactly that composition and nothing else, so its source hash
is a fingerprint determinant.  Upstream stage materialization (augmentation,
encoding, pack planning, supervision building) and the fail-closed
orchestration around it stay with the cache workflow.

It must not import the training facade, the cache workflow, the training
session, or the low-level cache serializer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from src.packing.planner import PackedSequence
from src.qwen.positions import build_qwen_position_inputs
from src.supervision.tokens import build_token_sequence_from_packed_supervision
from src.training.micro_steps import SupervisedMicroStep


def assemble_micro_steps(
    config: Any,
    components: Any,
    vocab_groups: Any,
    *,
    split: str,
    packs: Sequence[PackedSequence],
    encoded_examples: Sequence[Any],
    augmentation_receipt: Any,
    pack_plan_receipt: Mapping[str, Any],
    fragment_by_pack: Mapping[int, str],
    token_atoms_by_pack: Mapping[int, Sequence[Any]],
) -> tuple[SupervisedMicroStep, ...]:
    """Compose one serialized micro-step per planned pack.

    Every field this function decides enters the committed cache payload: the
    metadata keys and their order, the realized position inputs and token
    sequence, and the four constructor arguments derived from configuration and
    resolved token identity.  Changing any of them changes cached payloads and
    MUST change the semantic cache fingerprint through this module's source
    hash.
    """

    image_token_id = _image_token_id(components)
    micro_steps: list[SupervisedMicroStep] = []
    for pack in packs:
        pack_examples = _encoded_examples_for_pack(pack, encoded_examples)
        position_inputs = build_qwen_position_inputs(
            pack,
            pack_examples,
            image_token_id=image_token_id,
        )
        token_sequence = build_token_sequence_from_packed_supervision(
            pack,
            token_atoms_by_pack.get(pack.pack_index, ()),
        )
        micro_steps.append(
            SupervisedMicroStep(
                pack=pack,
                encoded_examples=pack_examples,
                position_inputs=position_inputs,
                token_sequence=token_sequence,
                vocab_groups=vocab_groups,
                metadata={
                    "split": split,
                    "pack_id": pack.pack_index,
                    "example_ids": [segment.example_id for segment in pack.segments],
                    "augmentation_receipt": augmentation_receipt,
                    "pack_plan": {
                        **pack_plan_receipt,
                        "fragment_sha256": fragment_by_pack[pack.pack_index],
                    },
                },
                expected_vocab_size=components.token_identity.tokenizer_vocab_size,
                fa2_model_dtype=config.training.precision,
                capture_fa2_branch=config.model.fa2_branch_proof == "every_forward",
                require_fa2_branch_proof=config.model.fa2_branch_proof
                == "every_forward",
            )
        )
    return tuple(micro_steps)


def _encoded_examples_for_pack(
    pack: PackedSequence,
    encoded_examples: Sequence[Any],
) -> tuple[Any, ...]:
    examples_by_id = {
        str(getattr(example, "example_id")): example for example in encoded_examples
    }
    return tuple(examples_by_id[segment.example_id] for segment in pack.segments)


def _image_token_id(components: Any) -> int | None:
    tokenizer = getattr(components, "tokenizer", None)
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        return None
    token_id = convert("<|image_pad|>")
    return None if token_id is None else int(token_id)


__all__ = ["assemble_micro_steps"]
