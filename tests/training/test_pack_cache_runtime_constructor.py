from __future__ import annotations

import pickle
from types import SimpleNamespace
from typing import Any

import pytest

import src.training.cache_workflow as cache_workflow
import src.training.micro_step_assembler as micro_step_assembler
from src.augmentation.processor import AugmentationMaterializationResult
from src.config.models import PackingConfig
from src.losses.vocab import TokenVocabularyGroups


_RUNTIME_FIELD_CASES = (
    (
        "bf16",
        "first_micro_step",
        {
            "fa2_model_dtype": "bf16",
            "capture_fa2_branch": False,
            "require_fa2_branch_proof": False,
        },
    ),
    (
        "fp16",
        "first_micro_step",
        {
            "fa2_model_dtype": "fp16",
            "capture_fa2_branch": False,
            "require_fa2_branch_proof": False,
        },
    ),
    (
        "bf16",
        "every_forward",
        {
            "fa2_model_dtype": "bf16",
            "capture_fa2_branch": True,
            "require_fa2_branch_proof": True,
        },
    ),
    (
        "fp16",
        "every_forward",
        {
            "fa2_model_dtype": "fp16",
            "capture_fa2_branch": True,
            "require_fa2_branch_proof": True,
        },
    ),
)


def _config(*, precision: str, fa2_branch_proof: str) -> SimpleNamespace:
    return SimpleNamespace(
        training=SimpleNamespace(precision=precision),
        model=SimpleNamespace(fa2_branch_proof=fa2_branch_proof),
        packing=PackingConfig(global_max_length=8),
    )


def _components() -> SimpleNamespace:
    return SimpleNamespace(
        token_identity=SimpleNamespace(tokenizer_vocab_size=16),
        tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda _token: 7),
    )


def _vocab_groups() -> TokenVocabularyGroups:
    return TokenVocabularyGroups(
        vocab_size=16,
        desc_text=(0, 1, 2, 3),
        schema=(4,),
        coordinate=(5,),
        eos=(6,),
        blocked=(7,),
    )


@pytest.mark.parametrize(
    ("precision", "fa2_branch_proof", "expected_runtime_determinants"),
    _RUNTIME_FIELD_CASES,
)
def test_production_micro_step_constructor_serializes_runtime_determinants(
    monkeypatch: pytest.MonkeyPatch,
    precision: str,
    fa2_branch_proof: str,
    expected_runtime_determinants: dict[str, Any],
) -> None:
    """Attest the production constructor independently of its determinant registry."""

    encoded_example = SimpleNamespace(
        example_id="runtime-constructor-example",
        input_ids=(7, 1, 6),
        supervised_token_spans=(),
    )
    augmentation = AugmentationMaterializationResult(
        examples=(SimpleNamespace(example_id=encoded_example.example_id),),
        receipt={"policy": "test-noop"},
    )

    monkeypatch.setattr(
        cache_workflow,
        "_materialize_raw_examples_for_dataset",
        lambda *_args, **_kwargs: augmentation,
    )
    monkeypatch.setattr(
        cache_workflow,
        "_build_encoded_examples_for_dataset",
        lambda *_args, **_kwargs: (encoded_example,),
    )
    # Position construction now lives with the micro-step assembler owner; the
    # substitution follows the call, the attested constructor fields do not.
    monkeypatch.setattr(
        micro_step_assembler,
        "build_qwen_position_inputs",
        lambda pack, _examples, **_kwargs: {"pack_index": pack.pack_index},
    )

    constructed = cache_workflow._build_micro_steps_for_dataset(
        _config(precision=precision, fa2_branch_proof=fa2_branch_proof),
        _components(),
        _vocab_groups(),
        dataset=SimpleNamespace(path="unused-by-controlled-materializer.jsonl"),
        split="train",
        materialization_workers=1,
    )

    # Packing-cache chunks use the same highest-protocol pickle representation.
    # Keep the expected values literal here: importing the determinant builder or
    # registry would let the test reproduce the same bug on both sides.
    serialized = pickle.dumps(constructed, protocol=pickle.HIGHEST_PROTOCOL)
    (restored,) = pickle.loads(serialized)  # noqa: S301 - local trusted test payload
    observed_runtime_determinants = {
        "fa2_model_dtype": restored.fa2_model_dtype,
        "capture_fa2_branch": restored.capture_fa2_branch,
        "require_fa2_branch_proof": restored.require_fa2_branch_proof,
    }

    assert observed_runtime_determinants == expected_runtime_determinants
