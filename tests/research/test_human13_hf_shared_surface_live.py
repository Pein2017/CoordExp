from __future__ import annotations

from dataclasses import dataclass
import hashlib
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from scripts.research.human13_hf_shared_surface import (
    GradientReplayGroup,
    SampledHFGroup,
    causal_history_sha256,
    plan_image1584_k16,
)
from scripts.research.human13_live_model import Human13LiveAssembly


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


class TinyTokenizer:
    pad_token_id = 0

    def __len__(self) -> int:
        return 8

    def convert_tokens_to_ids(self, token: str) -> int:
        if token != "<|im_end|>":
            raise AssertionError(f"unexpected token lookup: {token}")
        return 3


class TinyCausalModel(torch.nn.Module):
    """CPU causal fake whose logits depend only on the causal position."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(8, dtype=torch.bfloat16))
        self.config = SimpleNamespace(
            _attn_implementation="flash_attention_2", use_cache=False
        )
        self.forward_calls: list[dict[str, Any]] = []
        self.return_cache = False
        self.return_nonfinite = False
        self.raise_forward = False

    def forward(self, **kwargs: Any) -> SimpleNamespace:
        call = {
            "input_ids": kwargs["input_ids"].detach().cpu().clone(),
            "attention_mask": kwargs["attention_mask"].detach().cpu().clone(),
            "use_cache": kwargs.get("use_cache"),
            "grad_enabled": torch.is_grad_enabled(),
            "training": self.training,
            "model_object_id": id(self),
            "dtype": str(self.weight.dtype),
            "backend": self.config._attn_implementation,
        }
        self.forward_calls.append(call)
        if self.raise_forward:
            raise RuntimeError("injected causal-forward failure")
        input_ids = kwargs["input_ids"]
        batch, length = input_ids.shape
        rows = []
        for position in range(length):
            logits = self.weight.reshape(1, -1).expand(batch, -1)
            if position >= 3:
                stop_bias = torch.zeros_like(logits)
                stop_bias[:, 3] = 40.0
                logits = logits + stop_bias
            rows.append(logits)
        # Accelerate's native BF16 wrapper converts prepared-model outputs to
        # fp32 while parameters and autocast compute remain BF16.
        output = torch.stack(rows, dim=1).float()
        if self.return_nonfinite:
            output = output.clone()
            output[0, -1, 0] = float("nan")
        return SimpleNamespace(
            logits=output,
            past_key_values=(object(),) if self.return_cache else None,
        )

    def get_rope_index(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
        video_grid_thw: torch.Tensor | None,
        *,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        del image_grid_thw, video_grid_thw, attention_mask
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        return positions.view(1, 1, -1).expand(3, input_ids.shape[0], -1), None


@dataclass
class TinySkeleton:
    example_id: str = "coco2017_val_000000001584"
    human13_image_id: int = 1584
    input_ids: tuple[int, ...] = (5, 6, 7, 4)
    prompt_token_count: int = 3
    image_encoding: Any = None

    def __post_init__(self) -> None:
        if self.image_encoding is None:
            self.image_encoding = SimpleNamespace(
                pixel_values=torch.ones((1, 2), dtype=torch.bfloat16),
                image_grid_thw=(1, 1, 1),
                plan=SimpleNamespace(image_content_sha256=_digest("image-1584")),
            )


class SurfaceReceipt:
    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "phase": "before_first_backward",
            "frozen_towers": ["language", "vision", "aligner"],
            "trainable_towers": ["adapter.language"],
            "optimizer_groups": [{"group_name": "adapter.language"}],
            "unmatched_trainable_names": [],
            "exact_surface_groups": {
                "trainable_language_dora": {"parameter_count": 1},
                "frozen_selected_token_delta": {"parameter_count": 1},
                "frozen_vision": {"parameter_count": 1},
                "frozen_aligner": {"parameter_count": 1},
            },
        }


def _assembly(
    model: TinyCausalModel | None = None,
    tokenizer: TinyTokenizer | None = None,
) -> Human13LiveAssembly:
    live_model = model or TinyCausalModel()
    live_tokenizer = tokenizer or TinyTokenizer()
    source = SimpleNamespace(
        checkpoint_path="/source/step-2444",
        adapter_sha256=_digest("source-adapter"),
        special_embedding_sha256=_digest("source-embedding-delta"),
    )
    plan = SimpleNamespace(
        source=source,
        mixed_precision="bf16",
        attn_implementation="flash_attention_2",
        adapter_target_towers=("language",),
        freeze_special_token_delta=True,
    )
    validation = SimpleNamespace(
        base_config_sha256=_digest("base-config"),
        tokenizer_sha256=_digest("tokenizer"),
        adapter_tensor_sha256=source.adapter_sha256,
        special_embedding_tensor_sha256=source.special_embedding_sha256,
    )
    components = SimpleNamespace(
        tokenizer=live_tokenizer,
        tokenizer_sha256=validation.tokenizer_sha256,
        processor=SimpleNamespace(name="tiny-processor"),
    )
    return Human13LiveAssembly(
        plan=plan,
        validation=validation,
        components=components,
        accelerator=SimpleNamespace(free_memory=lambda: None),
        model=live_model,
        adapter_result=SimpleNamespace(model=live_model),
        special_token_result=SimpleNamespace(
            model=live_model,
            shared_embed_delta=SimpleNamespace(requires_grad=False),
        ),
        optimizer=object(),
        scheduler=object(),
        optimizer_group_plan=object(),
        trainable_surface_receipt=SurfaceReceipt(),
        runtime=SimpleNamespace(model=live_model, world_size=1),
        memory_saver_receipt={"use_cache_disabled": True},
    )


def _open() -> tuple[Any, Human13LiveAssembly, TinySkeleton]:
    from scripts.research.human13_hf_shared_surface_live import (
        open_hf_shared_surface,
    )

    assembly = _assembly()
    skeleton = TinySkeleton()
    return (
        open_hf_shared_surface(plan_image1584_k16(), assembly, skeleton),
        assembly,
        skeleton,
    )


def test_stepwise_sampling_preserves_survivor_order_and_rng_history_receipts() -> None:
    session, assembly, _skeleton = _open()
    model = assembly.model

    group = session.sample_group((35001, 35002, 35003, 35004))

    assert type(group) is SampledHFGroup
    assert tuple(request.seed for request in group.requests) == (
        35001,
        35002,
        35003,
        35004,
    )
    assert [len(request.tokens) for request in group.requests] == [1, 2, 2, 2]
    assert group.requests[0].stop_reason == "im_end"
    assert group.active_batch_steps[0].active_request_ids == tuple(
        request.request_id for request in group.requests
    )
    assert group.active_batch_steps[1].active_request_ids == tuple(
        request.request_id for request in group.requests[1:]
    )
    assert group.active_batch_steps[1].rng_before_sha256 == (
        group.active_batch_steps[0].rng_after_sha256
    )
    assert [call["input_ids"].shape for call in model.forward_calls] == [
        torch.Size([4, 3]),
        torch.Size([3, 4]),
    ]
    assert all(call["use_cache"] is False for call in model.forward_calls)
    assert all(call["grad_enabled"] is False for call in model.forward_calls)
    assert all(call["training"] is False for call in model.forward_calls)
    assert all(call["model_object_id"] == id(model) for call in model.forward_calls)
    for request in group.requests:
        generated: list[int] = []
        for token in request.tokens:
            assert token.history_sha256 == causal_history_sha256(
                request.prompt_history_sha256, tuple(generated)
            )
            generated.append(token.chosen_token_id)


def test_all_four_groups_cover_the_frozen_k16_seed_plan() -> None:
    session, _assembly_value, _skeleton = _open()
    plan = plan_image1584_k16()

    groups = tuple(session.sample_group(seeds) for seeds in plan.seed_groups)

    assert tuple(group.group_index for group in groups) == (0, 1, 2, 3)
    assert tuple(request.seed for group in groups for request in group.requests) == tuple(
        range(35001, 35017)
    )
    receipt = session.resource_receipt
    assert receipt.sample_forward_count == 8
    assert receipt.replay_forward_count == 0
    assert receipt.no_cache_forward_count == receipt.total_forward_count == 8


def test_vectorized_replay_uses_same_eval_model_with_grad_and_causal_gathers() -> None:
    session, assembly, _skeleton = _open()
    model = assembly.model
    sampled = session.sample_group((35001, 35002, 35003, 35004))

    replay = session.replay_group(sampled)

    assert type(replay) is GradientReplayGroup
    replay_call = model.forward_calls[-1]
    assert replay_call["input_ids"].shape == torch.Size([4, 5])
    assert replay_call["attention_mask"].tolist() == [
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ]
    assert replay_call["use_cache"] is False
    assert replay_call["grad_enabled"] is True
    assert replay_call["training"] is False
    assert replay_call["model_object_id"] == id(model)
    assert tuple(gather.causal_logit_index for gather in replay.causal_gathers) == (
        2,
        2,
        3,
        2,
        3,
        2,
        3,
    )
    assert replay.parity.max_abs_error == 0.0
    assert replay.parity.mean_abs_error == 0.0
    assert session.live_replay_tensor_count == 7
    assert session.resource_receipt.live_replay_group_count == 1
    assert all(isinstance(token.processed_logp, float) for token in replay.replayed_tokens)


def test_resource_receipt_binds_identity_processor_forwards_and_cleanup() -> None:
    session, assembly, _skeleton = _open()
    sampled = session.sample_group((35001, 35002, 35003, 35004))
    replay = session.replay_group(sampled)

    opened = session.resource_receipt
    assert opened.identity.model_object_id == id(assembly.model)
    assert opened.parameter_state_sha256 == opened.identity.parameter_state_sha256
    assert opened.processor_order == (
        "repetition_penalty",
        "temperature",
        "top_p",
    )
    assert opened.sample_forward_count == 2
    assert opened.replay_forward_count == 1
    assert opened.no_cache_forward_count == opened.total_forward_count == 3
    assert opened.observed_logits_dtype == "float32"
    assert opened.cleanup_state == "open"
    assert opened.latest_replay_group_sha256 == replay.content_sha256

    closed = session.close()

    assert closed.cleanup_state == "closed"
    assert closed.cleanup_call_count == 1
    assert closed.live_replay_group_count == 0
    assert closed.retained_live_resource_count == 0
    assert session.live_replay_tensor_count == 0
    with pytest.raises(RuntimeError, match="already closed"):
        session.close()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("model", "model object"),
        ("parameter", "parameter state"),
        ("mode", "eval mode"),
        ("dtype", "bfloat16"),
        ("backend", "flash_attention_2"),
        ("checkpoint", "checkpoint payload"),
        ("adapter", "adapter"),
        ("delta", "embedding delta"),
        ("prompt", "prompt"),
        ("image", "image"),
        ("tokenizer", "tokenizer"),
        ("tokenizer_hash", "tokenizer"),
        ("processor", "processor"),
    ),
)
def test_replay_fails_closed_on_surface_or_lineage_drift(
    mutation: str, message: str
) -> None:
    session, assembly, skeleton = _open()
    sampled = session.sample_group((35001, 35002, 35003, 35004))
    if mutation == "model":
        session._model = TinyCausalModel()  # noqa: SLF001 - adversarial substitution
    elif mutation == "parameter":
        with torch.no_grad():
            assembly.model.weight.add_(1)
    elif mutation == "mode":
        assembly.model.train()
    elif mutation == "dtype":
        assembly.model.weight.data = assembly.model.weight.data.float()
    elif mutation == "backend":
        assembly.model.config._attn_implementation = "sdpa"
    elif mutation == "checkpoint":
        assembly.plan.source.checkpoint_path = "/substituted/checkpoint"
    elif mutation == "adapter":
        assembly.plan.source.adapter_sha256 = _digest("different-adapter")
    elif mutation == "delta":
        assembly.plan.source.special_embedding_sha256 = _digest("different-delta")
    elif mutation == "prompt":
        skeleton.input_ids = (5, 6, 1, 4)
    elif mutation == "image":
        skeleton.image_encoding.pixel_values.add_(1)
    elif mutation == "tokenizer":
        assembly.components.tokenizer = TinyTokenizer()
    elif mutation == "tokenizer_hash":
        assembly.components.tokenizer_sha256 = _digest("different-tokenizer")
    elif mutation == "processor":
        assembly.components.processor = SimpleNamespace(name="substitute")
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(mutation)

    with pytest.raises(RuntimeError, match=message):
        session.replay_group(sampled)

    assert session.resource_receipt.cleanup_state == "closed"
    assert session.live_replay_tensor_count == 0


@pytest.mark.parametrize(
    ("fault", "message"),
    (
        ("cache", "cache"),
        ("nonfinite", "finite"),
        ("forward", "injected causal-forward failure"),
    ),
)
def test_sampling_forward_faults_are_terminal_without_retry(
    fault: str, message: str
) -> None:
    session, assembly, _skeleton = _open()
    model = assembly.model
    if fault == "cache":
        model.return_cache = True
    elif fault == "nonfinite":
        model.return_nonfinite = True
    else:
        model.raise_forward = True

    with pytest.raises((RuntimeError, ValueError), match=message):
        session.sample_group((35001, 35002, 35003, 35004))

    assert len(model.forward_calls) == 1
    assert session.resource_receipt.cleanup_state == "closed"


def test_wrong_group_padding_or_history_cannot_cross_replay_boundary() -> None:
    first, _first_assembly, _first_skeleton = _open()
    second, _second_assembly, _second_skeleton = _open()
    sampled = first.sample_group((35001, 35002, 35003, 35004))

    with pytest.raises(RuntimeError, match="shared surface identity"):
        second.replay_group(sampled)

    assert second.resource_receipt.cleanup_state == "closed"


def test_context_exception_closes_session_and_drops_live_graphs() -> None:
    session, _assembly_value, _skeleton = _open()

    with pytest.raises(RuntimeError, match="body failed"):
        with session:
            sampled = session.sample_group((35001, 35002, 35003, 35004))
            session.replay_group(sampled)
            raise RuntimeError("body failed")

    receipt = session.resource_receipt
    assert receipt.cleanup_state == "closed"
    assert receipt.cleanup_reason == "failed"
    assert receipt.live_replay_group_count == 0


def test_open_rejects_nonfrozen_plan_or_noncanonical_skeleton_before_forward() -> None:
    from scripts.research.human13_hf_shared_surface_live import (
        open_hf_shared_surface,
    )

    assembly = _assembly()
    with pytest.raises(RuntimeError, match="image 1584"):
        open_hf_shared_surface(
            plan_image1584_k16(),
            assembly,
            TinySkeleton(human13_image_id=2299),
        )
    assert assembly.model.forward_calls == []
