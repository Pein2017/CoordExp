from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from src.common.errors import QwenForwardContractError
from src.packing.planner import plan_packed_sequences
from src.qwen.fa2 import (
    AttentionEventKind,
    Fa2AttentionProofEvidence,
    Fa2VarlenPlan,
    build_fa2_varlen_plan,
    capture_fa2_varlen_branch,
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
                (
                    20,
                    21,
                    IMAGE_TOKEN_ID,
                    IMAGE_TOKEN_ID,
                    IMAGE_TOKEN_ID,
                    IMAGE_TOKEN_ID,
                ),
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
        build_fa2_varlen_plan(
            pack, attention_mask=torch.ones((1, pack.length), dtype=torch.long)
        )

    assert exc_info.value.code == "qwen.fa2_attention_mask"


def test_fa2_all_text_layers_accept_exact_padding_free_varlen(
    fake_attention_runtime: None,
) -> None:
    plan = _fake_plan()
    model = FakeTopologicalQwenModel(vocab_size=17, layer_count=3)
    evidence = _capture_evidence(model, plan)

    proof = validate_fa2_varlen_branch_evidence(
        plan,
        evidence,
        model_dtype="torch.bfloat16",
        expected_device="cpu",
    )

    assert proof.observed_branch == "padding_free_varlen"
    assert proof.cu_seq_lens_q == plan.segment_boundaries
    assert len(proof.text_layer_events) == 3
    assert [event.layer_idx for event in proof.text_layer_events] == [0, 1, 2]
    assert all(
        event.flash_varlen_fn_call_count == 1 for event in proof.text_layer_events
    )
    artifact = proof.to_artifact_dict()
    assert artifact["status"] == "pass"
    assert artifact["expected_text_layer_count"] == 3
    assert artifact["observed_text_layer_count"] == 3


def test_fa2_topology_resolution_is_wrapper_prefix_agnostic(
    fake_attention_runtime: None,
) -> None:
    plan = _fake_plan()
    wrapped = OpaqueQwenWrapper(FakeTopologicalQwenModel(vocab_size=17))
    evidence = _capture_evidence(wrapped, plan)

    proof = validate_fa2_varlen_branch_evidence(
        plan,
        evidence,
        model_dtype="torch.bfloat16",
        expected_device="cpu",
    )

    assert proof.topology.text_model_name == "opaque.language_model"
    assert proof.topology.text_layers[0].module_name.startswith(
        "opaque.language_model.layers."
    )


@pytest.mark.parametrize("mutation", ["config_count", "layer_idx"])
def test_fa2_topology_rejects_config_count_or_noncontiguous_layer_idx(
    fake_attention_runtime: None,
    mutation: str,
) -> None:
    model = FakeTopologicalQwenModel(vocab_size=17)
    if mutation == "config_count":
        model.language_model.config.num_hidden_layers += 1
    else:
        model.language_model.layers[1].self_attn.layer_idx = 7

    with pytest.raises(QwenForwardContractError) as exc_info:
        with capture_fa2_varlen_branch(model):
            pass

    assert exc_info.value.code == "qwen.fa2_topology"


def test_fa2_branch_evidence_accepts_config_precision_spellings(
    fake_attention_runtime: None,
) -> None:
    plan = _fake_plan()
    evidence = _capture_evidence(FakeTopologicalQwenModel(vocab_size=17), plan)

    proof = validate_fa2_varlen_branch_evidence(
        plan, evidence, model_dtype="bf16", expected_device="cpu"
    )

    assert proof.model_dtype == "bf16"


def test_fa2_legacy_aggregate_evidence_fails_closed() -> None:
    plan = _fake_plan()

    with pytest.raises(QwenForwardContractError) as exc_info:
        validate_fa2_varlen_branch_evidence(
            plan,
            {"observed_branch": "padding_free_varlen"},  # type: ignore[arg-type]
            model_dtype="torch.bfloat16",
            expected_device="cpu",
        )

    assert exc_info.value.code == "qwen.fa2_legacy_evidence"


@pytest.mark.parametrize(
    ("call_layers", "code"),
    [
        ((0,), "qwen.fa2_text_layer_coverage"),
        ((0, 1), "qwen.fa2_text_layer_coverage"),
        ((0, 1, 1, 2), "qwen.fa2_text_layer_coverage"),
    ],
)
def test_fa2_text_layer_proof_rejects_one_matching_missing_or_duplicate_calls(
    fake_attention_runtime: None,
    call_layers: tuple[int, ...],
    code: str,
) -> None:
    plan = _fake_plan()
    evidence = _capture_evidence(
        FakeTopologicalQwenModel(
            vocab_size=17,
            layer_count=3,
            call_layers=call_layers,
        ),
        plan,
    )

    with pytest.raises(QwenForwardContractError) as exc_info:
        validate_fa2_varlen_branch_evidence(
            plan,
            evidence,
            model_dtype="torch.bfloat16",
            expected_device="cpu",
        )

    assert exc_info.value.code == code


def test_fa2_text_layer_proof_rejects_one_mismatched_boundary(
    fake_attention_runtime: None,
) -> None:
    plan = _fake_plan()
    evidence = _capture_evidence(
        FakeTopologicalQwenModel(vocab_size=17, mismatch_boundary_layer=1),
        plan,
    )

    with pytest.raises(QwenForwardContractError) as exc_info:
        validate_fa2_varlen_branch_evidence(
            plan,
            evidence,
            model_dtype="torch.bfloat16",
            expected_device="cpu",
        )

    assert exc_info.value.code == "qwen.fa2_cu_seq_lens"
    assert "language_model.layers.1.self_attn#1" in str(exc_info.value)


@pytest.mark.parametrize(
    ("configured_backend", "registry_key"),
    [("sdpa", "sdpa"), ("flash_attention_2", "sdpa")],
)
def test_fa2_text_layer_proof_rejects_non_fa_config_or_registry_backend(
    fake_attention_runtime: None,
    configured_backend: str,
    registry_key: str,
) -> None:
    plan = _fake_plan()
    evidence = _capture_evidence(
        FakeTopologicalQwenModel(
            vocab_size=17,
            configured_backend=configured_backend,
            registry_key=registry_key,
        ),
        plan,
    )

    with pytest.raises(QwenForwardContractError) as exc_info:
        validate_fa2_varlen_branch_evidence(
            plan,
            evidence,
            model_dtype="torch.bfloat16",
            expected_device="cpu",
        )

    assert exc_info.value.code == "qwen.fa2_attention_implementation"


def test_fa2_vision_calls_are_typed_but_never_inflate_text_coverage(
    fake_attention_runtime: None,
) -> None:
    plan = _fake_plan()
    evidence = _capture_evidence(
        FakeTopologicalQwenModel(
            vocab_size=17,
            layer_count=2,
            call_layers=(0,),
            include_vision=True,
        ),
        plan,
    )

    assert [event.kind for event in evidence.events] == [
        AttentionEventKind.TEXT,
        AttentionEventKind.VISION,
    ]
    with pytest.raises(QwenForwardContractError) as exc_info:
        validate_fa2_varlen_branch_evidence(
            plan,
            evidence,
            model_dtype="torch.bfloat16",
            expected_device="cpu",
        )

    assert exc_info.value.code == "qwen.fa2_text_layer_coverage"
    assert exc_info.value.context["vision_event_count"] == 1


def test_fa2_unrelated_registry_event_fails_closed(
    fake_attention_runtime: None,
) -> None:
    plan = _fake_plan()
    evidence = _capture_evidence(
        FakeTopologicalQwenModel(
            vocab_size=17,
            include_unrelated=True,
        ),
        plan,
    )

    with pytest.raises(QwenForwardContractError) as exc_info:
        validate_fa2_varlen_branch_evidence(
            plan,
            evidence,
            model_dtype="torch.bfloat16",
            expected_device="cpu",
        )

    assert exc_info.value.code == "qwen.fa2_unrelated_attention"


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


def test_qwen_forward_receipt_marks_disabled_fa2_proof_policy_without_proof() -> None:
    examples = _fake_examples()
    pack = plan_packed_sequences(examples, global_max_length=32)[0]
    positions = build_qwen_position_inputs(pack, examples)
    forward_inputs = build_qwen_forward_inputs(
        pack,
        examples,
        positions,
        fa2_branch_proof_policy="disabled",
    )

    kwargs = forward_inputs.to_model_kwargs()
    assert kwargs["attention_mask"] is None
    assert kwargs["cu_seq_lens_q"].tolist() == [0, 11, 20]
    assert kwargs["cu_seq_lens_k"].tolist() == [0, 11, 20]
    result = run_qwen_forward(
        FakeQwenModel(vocab_size=17),
        forward_inputs,
        expected_vocab_size=17,
        capture_fa2_branch=False,
        require_fa2_branch_proof=False,
    )

    artifact = result.receipt.to_artifact_dict()
    assert artifact["fa2_varlen"]["branch_proof_policy"] == "disabled"
    assert artifact["fa2_varlen"]["proof"] is None
    assert artifact["fa2_varlen"]["segment_boundaries"] == [0, 11, 20]


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


def test_qwen_forward_runner_rejects_attention_mask_override_before_model_call() -> (
    None
):
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
            extra_model_kwargs={
                "attention_mask": torch.ones((1, pack.length), dtype=torch.long)
            },
        )

    assert exc_info.value.code == "qwen.forward_attention_mask"
    assert model.calls == 0


def test_qwen_forward_runner_attaches_all_layer_fa2_branch_evidence(
    fake_attention_runtime: None,
) -> None:
    examples = _fake_examples()
    pack = plan_packed_sequences(examples, global_max_length=32)[0]
    positions = build_qwen_position_inputs(pack, examples)
    forward_inputs = build_qwen_forward_inputs(pack, examples, positions)

    model = FakeTopologicalQwenModel(vocab_size=17, layer_count=3)
    result = run_qwen_forward(
        model,
        forward_inputs,
        expected_vocab_size=17,
        capture_fa2_branch=True,
        require_fa2_branch_proof=True,
    )

    artifact = result.receipt.to_artifact_dict()
    assert artifact["fa2_varlen"]["proof"]["observed_branch"] == "padding_free_varlen"
    assert artifact["fa2_varlen"]["proof"]["cu_seq_lens_q"] == [0, 11, 20]
    assert artifact["fa2_varlen"]["proof"]["status"] == "pass"
    assert artifact["fa2_varlen"]["proof"]["observed_text_layer_count"] == 3
    assert model.calls == 1


def test_fa2_capture_restores_registry_and_lazy_import_and_rejects_nesting(
    fake_attention_runtime: None,
) -> None:
    import transformers.modeling_flash_attention_utils as flash_utils
    from transformers import modeling_utils

    plan = _fake_plan()
    model = FakeTopologicalQwenModel(vocab_size=17)
    registry = modeling_utils.ALL_ATTENTION_FUNCTIONS
    original_lazy_import = flash_utils.lazy_import_flash_attention
    original_local_mapping = dict(registry._local_mapping)

    with capture_fa2_varlen_branch(model) as capture:
        with pytest.raises(QwenForwardContractError) as nested_exc:
            with capture_fa2_varlen_branch(model):
                pass
        assert nested_exc.value.code == "qwen.fa2_capture_nested"
        model(**_model_kwargs(plan))
        with pytest.raises(QwenForwardContractError) as early_exc:
            capture.evidence_for_plan(plan)
        assert early_exc.value.code == "qwen.fa2_capture_unrestored"

    evidence = capture.evidence_for_plan(plan)
    assert evidence.instrumentation_restored is True
    assert flash_utils.lazy_import_flash_attention is original_lazy_import
    assert registry._local_mapping.keys() == original_local_mapping.keys()
    assert all(
        registry._local_mapping[key] is original_local_mapping[key]
        for key in original_local_mapping
    )


def test_fa2_capture_restores_after_registry_wrapper_install_failure(
    fake_attention_runtime: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.qwen.fa2 as fa2_module
    import transformers.modeling_flash_attention_utils as flash_utils
    from transformers import modeling_utils

    model = FakeTopologicalQwenModel(vocab_size=17)
    registry = modeling_utils.ALL_ATTENTION_FUNCTIONS
    original_lazy_import = flash_utils.lazy_import_flash_attention
    original_local_mapping = dict(registry._local_mapping)
    original_callable_identity = fa2_module._callable_identity
    calls = 0

    def fail_during_install(value: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected wrapper install failure")
        return original_callable_identity(value)

    monkeypatch.setattr(fa2_module, "_callable_identity", fail_during_install)
    with pytest.raises(RuntimeError, match="injected wrapper install failure"):
        with capture_fa2_varlen_branch(model):
            pass

    assert flash_utils.lazy_import_flash_attention is original_lazy_import
    assert registry._local_mapping.keys() == original_local_mapping.keys()
    assert all(
        registry._local_mapping[key] is original_local_mapping[key]
        for key in original_local_mapping
    )


def test_fa2_text_layer_proof_rejects_local_override_even_with_canonical_callable(
    fake_attention_runtime: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transformers import modeling_utils
    from transformers.integrations.flash_attention import flash_attention_forward

    monkeypatch.setitem(
        modeling_utils.ALL_ATTENTION_FUNCTIONS._local_mapping,
        "flash_attention_2",
        flash_attention_forward,
    )
    plan = _fake_plan()
    evidence = _capture_evidence(FakeTopologicalQwenModel(vocab_size=17), plan)

    with pytest.raises(QwenForwardContractError) as exc_info:
        validate_fa2_varlen_branch_evidence(
            plan,
            evidence,
            model_dtype="torch.bfloat16",
            expected_device="cpu",
        )

    assert exc_info.value.code == "qwen.fa2_registry_callable"
    assert exc_info.value.context["registry_had_local_override"] is True


def test_fa2_text_layer_proof_binds_forward_device(
    fake_attention_runtime: None,
) -> None:
    plan = _fake_plan()
    evidence = _capture_evidence(FakeTopologicalQwenModel(vocab_size=17), plan)

    with pytest.raises(QwenForwardContractError) as exc_info:
        validate_fa2_varlen_branch_evidence(
            plan,
            evidence,
            model_dtype="torch.bfloat16",
            expected_device="cuda:0",
        )

    assert exc_info.value.code == "qwen.fa2_device"


def test_qwen_forward_runner_required_capture_rejects_model_without_qwen_topology() -> (
    None
):
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

    assert exc_info.value.code == "qwen.fa2_topology"


def test_qwen_forward_runner_rejects_legacy_supplied_evidence_when_proof_is_required() -> (
    None
):
    examples = _fake_examples()
    pack = plan_packed_sequences(examples, global_max_length=32)[0]
    positions = build_qwen_position_inputs(pack, examples)
    forward_inputs = build_qwen_forward_inputs(pack, examples, positions)

    with pytest.raises(QwenForwardContractError) as exc_info:
        run_qwen_forward(
            FakeQwenModel(vocab_size=17),
            forward_inputs,
            expected_vocab_size=17,
            fa2_branch_evidence={"stale": True},
            require_fa2_branch_proof=True,
        )

    assert exc_info.value.code == "qwen.fa2_legacy_evidence"


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


def _fake_plan() -> Fa2VarlenPlan:
    pack = plan_packed_sequences(_fake_examples(), global_max_length=32)[0]
    return build_fa2_varlen_plan(pack)


def _model_kwargs(plan: Fa2VarlenPlan) -> dict[str, Any]:
    return {
        "input_ids": torch.zeros((1, plan.segment_boundaries[-1]), dtype=torch.long),
        **plan.to_model_kwargs(),
    }


def _capture_evidence(
    model: nn.Module,
    plan: Fa2VarlenPlan,
) -> Fa2AttentionProofEvidence:
    with capture_fa2_varlen_branch(model) as capture:
        model(**_model_kwargs(plan))
    return capture.evidence_for_plan(plan)


@pytest.fixture
def fake_attention_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    import transformers.modeling_flash_attention_utils as flash_utils
    from transformers import modeling_utils

    monkeypatch.setattr(
        flash_utils,
        "lazy_import_flash_attention",
        fake_lazy_import_flash_attention,
    )
    monkeypatch.delitem(
        modeling_utils.ALL_ATTENTION_FUNCTIONS._local_mapping,
        "flash_attention_2",
        raising=False,
    )
    monkeypatch.setitem(
        modeling_utils.ALL_ATTENTION_FUNCTIONS._local_mapping,
        "sdpa",
        fake_registry_attention,
    )


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
                "logits": torch.zeros(
                    (1, seq_length, self.config.text_config.vocab_size)
                ),
                "loss": None,
                "past_key_values": None,
                "rope_deltas": None,
            },
        )()


class FakeTopologicalQwenModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        layer_count: int = 2,
        call_layers: tuple[int, ...] | None = None,
        configured_backend: str = "flash_attention_2",
        registry_key: str | None = None,
        mismatch_boundary_layer: int | None = None,
        include_vision: bool = False,
        include_unrelated: bool = False,
    ) -> None:
        super().__init__()
        from transformers.models.qwen3_vl.configuration_qwen3_vl import (
            Qwen3VLTextConfig,
        )
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLTextModel,
            Qwen3VLVisionAttention,
        )

        text_config = Qwen3VLTextConfig(
            vocab_size=vocab_size,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=layer_count,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 0]},
        )
        self.language_model = Qwen3VLTextModel(text_config).to(torch.bfloat16)
        self.language_model.config._attn_implementation = configured_backend
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(vocab_size=vocab_size)
        )
        self.call_layers = (
            tuple(range(layer_count)) if call_layers is None else call_layers
        )
        self.registry_key = configured_backend if registry_key is None else registry_key
        self.mismatch_boundary_layer = mismatch_boundary_layer
        self.include_vision = include_vision
        self.include_unrelated = include_unrelated
        self.calls = 0

        vision_attention = object.__new__(Qwen3VLVisionAttention)
        nn.Module.__init__(vision_attention)
        vision_attention.config = SimpleNamespace(
            _attn_implementation="flash_attention_2"
        )
        vision_attention.is_causal = False
        self.vision_attention = vision_attention
        self.unrelated_attention = nn.Identity()
        self.unrelated_attention.config = SimpleNamespace(
            _attn_implementation="flash_attention_2"
        )
        self.unrelated_attention.is_causal = True

    @property
    def dtype(self) -> torch.dtype:
        return torch.bfloat16

    def forward(self, **kwargs: Any) -> Any:
        from transformers import modeling_utils

        self.calls += 1
        sequence_length = int(kwargs["input_ids"].shape[1])
        q = torch.zeros((1, 2, sequence_length, 4), dtype=torch.bfloat16)
        for layer_idx in self.call_layers:
            boundaries_q = kwargs["cu_seq_lens_q"]
            if layer_idx == self.mismatch_boundary_layer:
                boundaries_q = torch.tensor((0, sequence_length), dtype=torch.int32)
            modeling_utils.ALL_ATTENTION_FUNCTIONS[self.registry_key](
                self.language_model.layers[layer_idx].self_attn,
                q,
                q,
                q,
                None,
                cu_seq_lens_q=boundaries_q,
                cu_seq_lens_k=kwargs["cu_seq_lens_k"],
                max_length_q=kwargs["max_length_q"],
                max_length_k=kwargs["max_length_k"],
            )
        if self.include_vision:
            modeling_utils.ALL_ATTENTION_FUNCTIONS["flash_attention_2"](
                self.vision_attention,
                q,
                q,
                q,
                None,
                cu_seq_lens_q=kwargs["cu_seq_lens_q"],
                cu_seq_lens_k=kwargs["cu_seq_lens_k"],
                max_length_q=kwargs["max_length_q"],
                max_length_k=kwargs["max_length_k"],
            )
        if self.include_unrelated:
            modeling_utils.ALL_ATTENTION_FUNCTIONS["flash_attention_2"](
                self.unrelated_attention,
                q,
                q,
                q,
                None,
                cu_seq_lens_q=kwargs["cu_seq_lens_q"],
                cu_seq_lens_k=kwargs["cu_seq_lens_k"],
                max_length_q=kwargs["max_length_q"],
                max_length_k=kwargs["max_length_k"],
            )
        return SimpleNamespace(
            logits=torch.zeros(
                (1, sequence_length, self.config.text_config.vocab_size)
            ),
            loss=None,
            past_key_values=None,
            rope_deltas=None,
        )


class OpaqueQwenWrapper(nn.Module):
    def __init__(self, model: FakeTopologicalQwenModel) -> None:
        super().__init__()
        self.opaque = model

    def forward(self, **kwargs: Any) -> Any:
        return self.opaque(**kwargs)


def fake_registry_attention(
    _module: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    _attention_mask: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    import transformers.modeling_flash_attention_utils as flash_utils

    (_flash_fn, flash_varlen_fn, _pad_fn, _unpad_fn), _process = (
        flash_utils.lazy_import_flash_attention("flash_attention_2")
    )
    sequence_length = int(q.shape[2])
    reshaped_q = q.transpose(1, 2).reshape(sequence_length, q.shape[1], q.shape[3])
    flash_varlen_fn(
        reshaped_q,
        reshaped_q if k is q else k,
        reshaped_q if v is q else v,
        cu_seqlens_q=kwargs["cu_seq_lens_q"],
        cu_seqlens_k=kwargs["cu_seq_lens_k"],
        max_seqlen_q=kwargs["max_length_q"],
        max_seqlen_k=kwargs["max_length_k"],
    )
    return q, None


def fake_lazy_import_flash_attention(
    implementation: str | None = None,
) -> tuple[Any, Any]:
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
