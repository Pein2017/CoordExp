from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch

from src.common.errors import QwenForwardContractError
from src.packing.planner import plan_packed_sequences
from src.qwen.fa2 import (
    Fa2VarlenPlan,
    build_fa2_varlen_plan,
    validate_fa2_varlen_branch_evidence,
)
from src.qwen.forward import build_qwen_forward_inputs, run_qwen_forward
from src.qwen.positions import build_qwen_position_inputs


IMAGE_TOKEN_ID = 151655


def test_fa2_varlen_plan_derives_boundaries_from_packed_segments() -> None:
    pack = plan_packed_sequences(
        (
            FakeEncodedExample(
                "ex-0",
                (10, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID),
            ),
            FakeEncodedExample(
                "ex-1",
                (20, 21, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID),
            ),
            FakeEncodedExample(
                "ex-2",
                (30, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID),
            ),
        ),
        global_max_length=20,
    )[0]

    plan = build_fa2_varlen_plan(pack)

    assert plan.segment_boundaries == (0, 5, 11, 16)
    assert plan.segment_lengths == (5, 6, 5)
    assert plan.max_length_q == 6
    assert plan.max_length_k == 6
    assert plan.attention_mask is None
    assert plan.cu_seq_lens_q.tolist() == [0, 5, 11, 16]
    assert plan.cu_seq_lens_q.dtype == torch.int32
    assert torch.equal(plan.cu_seq_lens_q, plan.cu_seq_lens_k)
    assert plan.to_artifact_dict()["branch_evidence_required"] is True


def test_fa2_varlen_plan_rejects_ordinary_attention_mask() -> None:
    example = FakeEncodedExample(
        "ex-0",
        (10, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID),
    )
    pack = plan_packed_sequences((example,), global_max_length=10)[0]

    with pytest.raises(QwenForwardContractError) as exc_info:
        build_fa2_varlen_plan(pack, attention_mask=torch.ones((1, pack.length), dtype=torch.long))

    assert exc_info.value.code == "qwen.fa2_attention_mask"


def test_fa2_branch_evidence_accepts_padding_free_varlen() -> None:
    pack = plan_packed_sequences(_fake_examples(), global_max_length=32)[0]
    plan = build_fa2_varlen_plan(pack)

    proof = validate_fa2_varlen_branch_evidence(
        plan,
        _branch_evidence(plan),
        resolved_attention_implementation="flash_attention_2",
        model_dtype="torch.bfloat16",
    )

    assert proof.observed_branch == "padding_free_varlen"
    assert proof.cu_seq_lens_q == plan.segment_boundaries
    assert proof.max_length_q == plan.max_length_q
    assert proof.branch_evidence_from_explicit_varlen_kwargs is True
    assert proof.to_artifact_dict()["status"] == "pass"


def test_fa2_branch_evidence_accepts_config_precision_spellings() -> None:
    pack = plan_packed_sequences(_fake_examples(), global_max_length=32)[0]
    plan = build_fa2_varlen_plan(pack)

    proof = validate_fa2_varlen_branch_evidence(
        plan,
        _branch_evidence(plan),
        resolved_attention_implementation="flash_attention_2",
        model_dtype="bf16",
    )

    assert proof.model_dtype == "bf16"


@pytest.mark.parametrize(
    ("mutator", "code"),
    [
        (
            lambda evidence, plan: evidence | {"observed_branch": "ordinary_flash"},
            "qwen.fa2_branch",
        ),
        (
            lambda evidence, plan: evidence | {"cu_seq_lens_q": (0, plan.segment_boundaries[-1])},
            "qwen.fa2_cu_seq_lens",
        ),
        (
            lambda evidence, plan: evidence | {"max_length_q": plan.max_length_q + 1},
            "qwen.fa2_max_length",
        ),
        (
            lambda evidence, plan: evidence | {"attention_mask": [[1, 1]]},
            "qwen.fa2_attention_mask",
        ),
        (
            lambda evidence, plan: evidence
            | {"branch_evidence_from_explicit_varlen_kwargs": False},
            "qwen.fa2_branch_evidence",
        ),
        (
            lambda evidence, plan: evidence | {"observed_call": None},
            "qwen.fa2_observed_call",
        ),
        (
            lambda evidence, plan: evidence
            | {"observed_call": {"cu_seqlens_q": (0, plan.segment_boundaries[-1])}},
            "qwen.fa2_observed_call",
        ),
        (lambda evidence, plan: evidence | {"flash_fn_called": True}, "qwen.fa2_branch"),
    ],
)
def test_fa2_branch_evidence_rejects_non_varlen_or_mismatched_calls(
    mutator: Any,
    code: str,
) -> None:
    pack = plan_packed_sequences(_fake_examples(), global_max_length=32)[0]
    plan = build_fa2_varlen_plan(pack)
    evidence = mutator(_branch_evidence(plan), plan)

    with pytest.raises(QwenForwardContractError) as exc_info:
        validate_fa2_varlen_branch_evidence(
            plan,
            evidence,
            resolved_attention_implementation="flash_attention_2",
            model_dtype="torch.bfloat16",
        )

    assert exc_info.value.code == code


def test_qwen_forward_inputs_include_fa2_varlen_kwargs_and_receipt() -> None:
    examples = _fake_examples()
    pack = plan_packed_sequences(examples, global_max_length=32)[0]
    positions = build_qwen_position_inputs(pack, examples)

    forward_inputs = build_qwen_forward_inputs(pack, examples, positions)

    kwargs = forward_inputs.to_model_kwargs()
    assert kwargs["attention_mask"] is None
    assert kwargs["cu_seq_lens_q"].tolist() == [0, 11, 20]
    assert kwargs["cu_seq_lens_k"].tolist() == [0, 11, 20]
    assert kwargs["max_length_q"] == 11
    assert kwargs["max_length_k"] == 11
    artifact = forward_inputs.to_artifact_dict()
    assert artifact["fa2_varlen"]["segment_boundaries"] == [0, 11, 20]
    assert artifact["fa2_varlen"]["branch_evidence_required"] is True


def test_qwen_forward_inputs_reject_supplied_fa2_plan_that_collapses_segments() -> None:
    examples = _fake_examples()
    pack = plan_packed_sequences(examples, global_max_length=32)[0]
    positions = build_qwen_position_inputs(pack, examples)
    wrong_plan = Fa2VarlenPlan(
        segment_boundaries=(0, pack.length),
        segment_lengths=(pack.length,),
        cu_seq_lens_q=torch.tensor((0, pack.length), dtype=torch.int32),
        cu_seq_lens_k=torch.tensor((0, pack.length), dtype=torch.int32),
        max_length_q=pack.length,
        max_length_k=pack.length,
        attention_mask=None,
    )

    with pytest.raises(QwenForwardContractError) as exc_info:
        build_qwen_forward_inputs(
            pack,
            examples,
            positions,
            fa2_varlen_plan=wrong_plan,
        )

    assert exc_info.value.code == "qwen.fa2_plan_boundaries"


def test_qwen_forward_runner_rejects_attention_mask_override_before_model_call() -> None:
    examples = _fake_examples()
    pack = plan_packed_sequences(examples, global_max_length=32)[0]
    positions = build_qwen_position_inputs(pack, examples)
    forward_inputs = build_qwen_forward_inputs(pack, examples, positions)
    model = FakeQwenModel(vocab_size=17)

    with pytest.raises(QwenForwardContractError) as exc_info:
        run_qwen_forward(
            model,
            forward_inputs,
            expected_vocab_size=17,
            extra_model_kwargs={"attention_mask": torch.ones((1, pack.length), dtype=torch.long)},
        )

    assert exc_info.value.code == "qwen.forward_attention_mask"
    assert model.calls == 0


def test_qwen_forward_runner_attaches_validated_fa2_branch_evidence() -> None:
    examples = _fake_examples()
    pack = plan_packed_sequences(examples, global_max_length=32)[0]
    positions = build_qwen_position_inputs(pack, examples)
    forward_inputs = build_qwen_forward_inputs(pack, examples, positions)

    result = run_qwen_forward(
        FakeQwenModel(vocab_size=17),
        forward_inputs,
        expected_vocab_size=17,
        fa2_branch_evidence=_branch_evidence(forward_inputs.fa2_varlen_plan),
    )

    artifact = result.receipt.to_artifact_dict()
    assert artifact["fa2_varlen"]["proof"]["observed_branch"] == "padding_free_varlen"
    assert artifact["fa2_varlen"]["proof"]["cu_seq_lens_q"] == [0, 11, 20]
    assert artifact["fa2_varlen"]["proof"]["status"] == "pass"


def test_qwen_forward_runner_captures_real_lazy_imported_fa2_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformers.modeling_flash_attention_utils as flash_utils

    examples = _fake_examples()
    pack = plan_packed_sequences(examples, global_max_length=32)[0]
    positions = build_qwen_position_inputs(pack, examples)
    forward_inputs = build_qwen_forward_inputs(pack, examples, positions)

    monkeypatch.setattr(
        flash_utils,
        "lazy_import_flash_attention",
        fake_lazy_import_flash_attention,
    )
    result = run_qwen_forward(
        FakeQwenModelWithFa2Call(vocab_size=17),
        forward_inputs,
        expected_vocab_size=17,
        capture_fa2_branch=True,
        require_fa2_branch_proof=True,
    )

    artifact = result.receipt.to_artifact_dict()
    proof = artifact["fa2_varlen"]["proof"]
    assert proof["status"] == "pass"
    assert proof["observed_branch"] == "padding_free_varlen"
    assert proof["cu_seq_lens_q"] == [0, 11, 20]
    assert proof["max_length_q"] == 11
    assert proof["flash_varlen_fn_called"] is True
    assert proof["flash_fn_called"] is False
    assert proof["observed_call"]["cu_seqlens_q"] == [0, 11, 20]


def test_qwen_forward_runner_can_require_captured_fa2_branch_proof() -> None:
    examples = _fake_examples()
    pack = plan_packed_sequences(examples, global_max_length=32)[0]
    positions = build_qwen_position_inputs(pack, examples)
    forward_inputs = build_qwen_forward_inputs(pack, examples, positions)

    with pytest.raises(QwenForwardContractError) as exc_info:
        run_qwen_forward(
            FakeQwenModel(vocab_size=17),
            forward_inputs,
            expected_vocab_size=17,
            capture_fa2_branch=True,
            require_fa2_branch_proof=True,
        )

    assert exc_info.value.code == "qwen.fa2_branch_evidence_missing"


def _fake_examples() -> tuple["FakeEncodedExample", ...]:
    return (
        FakeEncodedExample(
            "ex-0",
            (10, 11, 12, *([IMAGE_TOKEN_ID] * 6), 13, 14),
        ),
        FakeEncodedExample(
            "ex-1",
            (20, 21, *([IMAGE_TOKEN_ID] * 6), 22),
        ),
    )


def _branch_evidence(plan: Any) -> dict[str, Any]:
    return {
        "observed_branch": "padding_free_varlen",
        "attention_mask": None,
        "cu_seq_lens_q": plan.segment_boundaries,
        "cu_seq_lens_k": plan.segment_boundaries,
        "max_length_q": plan.max_length_q,
        "max_length_k": plan.max_length_k,
        "branch_evidence_from_explicit_varlen_kwargs": True,
        "flash_fn_called": False,
        "flash_varlen_fn_called": True,
        "pad_fn_called": False,
        "unpad_fn_called": False,
        "observed_call": {
            "cu_seqlens_q": plan.segment_boundaries,
            "cu_seqlens_k": plan.segment_boundaries,
            "max_seqlen_q": plan.max_length_q,
            "max_seqlen_k": plan.max_length_k,
        },
    }


@dataclass(frozen=True)
class FakeEncodedExample:
    example_id: str
    input_ids: tuple[int, ...]

    @property
    def image_pad_physical_start(self) -> int:
        return self.input_ids.index(IMAGE_TOKEN_ID)

    @property
    def image_pad_physical_end(self) -> int:
        cursor = self.image_pad_physical_start
        while cursor < len(self.input_ids) and self.input_ids[cursor] == IMAGE_TOKEN_ID:
            cursor += 1
        return cursor

    @property
    def image_encoding(self) -> "FakeImageEncoding":
        return FakeImageEncoding(
            image_grid_thw=(1, 4, 6),
            merged_visual_tokens=6,
            pixel_values=torch.ones((24, 8), dtype=torch.float32),
            plan=FakeImagePlan(merge_size=2),
        )


@dataclass(frozen=True)
class FakeImageEncoding:
    image_grid_thw: tuple[int, int, int]
    merged_visual_tokens: int
    pixel_values: torch.Tensor
    plan: "FakeImagePlan"


@dataclass(frozen=True)
class FakeImagePlan:
    merge_size: int


class FakeQwenModel:
    def __init__(self, *, vocab_size: int) -> None:
        self.config = type(
            "Config",
            (),
            {"text_config": type("Text", (), {"vocab_size": vocab_size})()},
        )()
        self.dtype = torch.bfloat16
        self.calls = 0

    def __call__(self, **kwargs: Any) -> Any:
        self.calls += 1
        seq_length = int(kwargs["input_ids"].shape[1])
        return type(
            "Output",
            (),
            {
                "logits": torch.zeros((1, seq_length, self.config.text_config.vocab_size)),
                "loss": None,
                "past_key_values": None,
                "rope_deltas": None,
            },
        )()


class FakeQwenModelWithFa2Call(FakeQwenModel):
    def __call__(self, **kwargs: Any) -> Any:
        import transformers.modeling_flash_attention_utils as flash_utils

        (flash_fn, flash_varlen_fn, _pad_fn, _unpad_fn), _process = (
            flash_utils.lazy_import_flash_attention("flash_attention_2")
        )
        del flash_fn
        seq_length = int(kwargs["input_ids"].shape[1])
        q = torch.zeros((seq_length, 1, 4), dtype=torch.bfloat16)
        flash_varlen_fn(
            q,
            q,
            q,
            cu_seqlens_q=kwargs["cu_seq_lens_q"],
            cu_seqlens_k=kwargs["cu_seq_lens_k"],
            max_seqlen_q=kwargs["max_length_q"],
            max_seqlen_k=kwargs["max_length_k"],
        )
        return super().__call__(**kwargs)


def fake_lazy_import_flash_attention(implementation: str | None = None) -> tuple[Any, Any]:
    del implementation

    def fake_flash_fn(q: torch.Tensor, *_args: Any, **_kwargs: Any) -> torch.Tensor:
        return q

    def fake_flash_varlen_fn(
        q: torch.Tensor,
        _k: torch.Tensor,
        _v: torch.Tensor,
        *_args: Any,
        **_kwargs: Any,
    ) -> torch.Tensor:
        return q

    def fake_pad_fn(q: torch.Tensor, *_args: Any, **_kwargs: Any) -> torch.Tensor:
        return q

    def fake_unpad_fn(q: torch.Tensor, *_args: Any, **_kwargs: Any) -> torch.Tensor:
        return q

    def fake_process_flash_kwargs_fn(**_kwargs: Any) -> dict[str, Any]:
        return {}

    return (
        (fake_flash_fn, fake_flash_varlen_fn, fake_pad_fn, fake_unpad_fn),
        fake_process_flash_kwargs_fn,
    )
