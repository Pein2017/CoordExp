"""Byte-level characterization of the assembled cached micro-step payload.

Frozen-contract guard over the production micro-step assembly path: identical
inputs must serialize to identical payload bytes, in the representation
``write_micro_step_cache`` commits per chunk (``pickle`` at
``pickle.HIGHEST_PROTOCOL``).

One canonicalization is unavoidable.  A realized ``QwenPositionInputs`` carries
``torch`` tensors, and a tensor pickles through
``torch.storage._load_from_bytes`` whose legacy storage key is derived from the
storage address, so two pickles of *equal* tensors differ in bytes.  The
pickler below therefore reduces tensors to ``(dtype, shape, values)`` and
leaves every other object to the ordinary protocol.  That keeps the guard over
the complete payload -- metadata composition, pack, encoded examples, token
sequence, vocabulary groups, realized position values, and the four
config/identity-derived constructor fields -- while removing the only
address-dependent bytes.

The golden constants below are a payload-identity contract, not an incidental
snapshot.  Re-pinning them declares that the realized cache payload changed and
that every cache published by the previous assembler is stale; do that only
under an explicit payload-identity decision, never to make a refactor green.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import pickle
from types import SimpleNamespace
from typing import Any

import pytest
import torch


#: sha256 and byte length of the canonically serialized micro-step tuple
#: assembled from the fixture below.
GOLDEN_PAYLOAD_SHA256 = (
    "45c07ba45eaf902c61a86fc82c74b0468d0f3845488b5ed9b17e1b34a24286cd"
)
GOLDEN_PAYLOAD_BYTE_LENGTH = 2924

IMAGE_PAD_TOKEN_ID = 151655


@dataclass(frozen=True)
class FakeImagePlan:
    merge_size: int


@dataclass(frozen=True)
class FakeImageEncoding:
    image_grid_thw: tuple[int, int, int]
    merged_visual_tokens: int
    plan: FakeImagePlan


@dataclass(frozen=True)
class FakeEncodedExample:
    """Minimal encoded example the real planner and position owner accept."""

    example_id: str
    input_ids: tuple[int, ...]
    image_pad_physical_start: int
    image_pad_physical_end: int
    image_grid_thw: tuple[int, int, int]
    merge_size: int
    supervised_token_spans: tuple[object, ...] = ()

    @property
    def image_token_count(self) -> int:
        return self.image_pad_physical_end - self.image_pad_physical_start

    @property
    def image_encoding(self) -> FakeImageEncoding:
        return FakeImageEncoding(
            image_grid_thw=self.image_grid_thw,
            merged_visual_tokens=self.image_token_count,
            plan=FakeImagePlan(merge_size=self.merge_size),
        )


ENCODED_EXAMPLE = FakeEncodedExample(
    example_id="characterization-example",
    input_ids=(
        10,
        11,
        12,
        IMAGE_PAD_TOKEN_ID,
        IMAGE_PAD_TOKEN_ID,
        IMAGE_PAD_TOKEN_ID,
        IMAGE_PAD_TOKEN_ID,
        IMAGE_PAD_TOKEN_ID,
        IMAGE_PAD_TOKEN_ID,
        13,
        14,
    ),
    image_pad_physical_start=3,
    image_pad_physical_end=9,
    image_grid_thw=(1, 4, 6),
    merge_size=2,
)
AUGMENTATION_RECEIPT = {"policy": "characterization-noop"}


def _canonical_tensor(
    dtype: str, shape: tuple[int, ...], values: list[Any]
) -> tuple[str, tuple[int, ...], list[Any]]:
    """Address-free stand-in for a tensor inside the canonical payload."""

    return (dtype, shape, values)


class _CanonicalPickler(pickle.Pickler):
    def reducer_override(self, obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return (
                _canonical_tensor,
                (
                    str(obj.dtype),
                    tuple(int(size) for size in obj.shape),
                    obj.detach().cpu().flatten().tolist(),
                ),
            )
        return NotImplemented


def _canonical_payload_bytes(micro_steps: tuple[Any, ...]) -> bytes:
    buffer = io.BytesIO()
    _CanonicalPickler(buffer, protocol=pickle.HIGHEST_PROTOCOL).dump(tuple(micro_steps))
    return buffer.getvalue()


def _config() -> SimpleNamespace:
    from src.config.models import PackingConfig

    return SimpleNamespace(
        training=SimpleNamespace(precision="bf16"),
        model=SimpleNamespace(fa2_branch_proof="every_forward"),
        packing=PackingConfig(global_max_length=16),
    )


def _components() -> SimpleNamespace:
    return SimpleNamespace(
        token_identity=SimpleNamespace(tokenizer_vocab_size=16),
        tokenizer=SimpleNamespace(
            convert_tokens_to_ids=lambda _token: IMAGE_PAD_TOKEN_ID
        ),
    )


def _vocab_groups() -> Any:
    from src.losses.vocab import TokenVocabularyGroups

    return TokenVocabularyGroups(
        vocab_size=16,
        desc_text=(0, 1, 2, 3),
        schema=(4,),
        coordinate=(5,),
        eos=(6,),
        blocked=(7,),
    )


def _assemble(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, ...]:
    """Run the production assembly path over controlled upstream materializers.

    Only the two source-reading stages are replaced.  Pack planning,
    supervision, position construction, the metadata composition, and every
    config/identity-derived constructor argument are the real production
    owners.
    """

    import src.training.cache_workflow as cache_workflow
    from src.augmentation.processor import AugmentationMaterializationResult

    augmentation = AugmentationMaterializationResult(
        examples=(SimpleNamespace(example_id=ENCODED_EXAMPLE.example_id),),
        receipt=dict(AUGMENTATION_RECEIPT),
    )
    monkeypatch.setattr(
        cache_workflow,
        "_materialize_raw_examples_for_dataset",
        lambda *_args, **_kwargs: augmentation,
    )
    monkeypatch.setattr(
        cache_workflow,
        "_build_encoded_examples_for_dataset",
        lambda *_args, **_kwargs: (ENCODED_EXAMPLE,),
    )
    return cache_workflow._build_micro_steps_for_dataset(
        _config(),
        _components(),
        _vocab_groups(),
        dataset=SimpleNamespace(path="unused-by-controlled-materializer.jsonl"),
        split="train",
        materialization_workers=1,
    )


def test_assembled_micro_step_payload_bytes_are_frozen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _canonical_payload_bytes(_assemble(monkeypatch))

    assert (hashlib.sha256(payload).hexdigest(), len(payload)) == (
        GOLDEN_PAYLOAD_SHA256,
        GOLDEN_PAYLOAD_BYTE_LENGTH,
    )


def test_assembled_micro_step_payload_bytes_are_reproducible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _canonical_payload_bytes(_assemble(monkeypatch))
    second = _canonical_payload_bytes(_assemble(monkeypatch))

    assert first == second


def test_assembled_micro_step_composition_is_the_declared_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Readable mirror of the byte guard, so a break says what moved."""

    (micro_step,) = _assemble(monkeypatch)

    assert list(micro_step.metadata) == [
        "split",
        "pack_id",
        "example_ids",
        "augmentation_receipt",
        "pack_plan",
    ]
    assert micro_step.metadata["split"] == "train"
    assert micro_step.metadata["pack_id"] == 0
    assert micro_step.metadata["example_ids"] == [ENCODED_EXAMPLE.example_id]
    assert micro_step.metadata["augmentation_receipt"] == AUGMENTATION_RECEIPT
    pack_plan = micro_step.metadata["pack_plan"]
    assert pack_plan["fragment_sha256"] == pack_plan["plan_sha256"]
    assert micro_step.expected_vocab_size == 16
    assert micro_step.fa2_model_dtype == "bf16"
    assert micro_step.capture_fa2_branch is True
    assert micro_step.require_fa2_branch_proof is True
    assert micro_step.encoded_examples == (ENCODED_EXAMPLE,)
    assert micro_step.vocab_groups == _vocab_groups()


def test_assembled_position_inputs_use_the_resolved_image_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The image-token argument the assembler resolves from components."""

    from src.qwen.positions import build_qwen_position_inputs

    (micro_step,) = _assemble(monkeypatch)
    reference = build_qwen_position_inputs(
        micro_step.pack,
        micro_step.encoded_examples,
        image_token_id=IMAGE_PAD_TOKEN_ID,
    )

    assert micro_step.position_inputs.to_artifact_dict() == (
        reference.to_artifact_dict()
    )
    assert micro_step.position_inputs.position_ids.equal(reference.position_ids)
