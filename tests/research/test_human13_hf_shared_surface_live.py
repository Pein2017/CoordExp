from __future__ import annotations

import copy
from dataclasses import dataclass
from dataclasses import replace
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from scripts.research.human13_hf_shared_surface import (
    GradientReplayGroup,
    SampledHFGroup,
    causal_history_sha256,
    plan_image1584_k16,
)
from scripts.research import human13_live_model as live_model
from scripts.research.human13_live_model import Human13LiveAssembly
from src.artifacts.json_values import json_sha256


CONFIG_ROOT = Path("configs/coordexp_swift/research/human13_k_union")


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

    def __init__(self, *, stop_position: int = 3) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(8, dtype=torch.bfloat16))
        self.visual = torch.nn.Module()
        self.visual.register_parameter(
            "weight",
            torch.nn.Parameter(
                torch.zeros(1, dtype=torch.bfloat16), requires_grad=False
            ),
        )
        self.visual.merger = torch.nn.Module()
        self.visual.merger.register_parameter(
            "weight",
            torch.nn.Parameter(
                torch.zeros(1, dtype=torch.bfloat16), requires_grad=False
            ),
        )
        self.shared_embed_delta = torch.nn.Parameter(
            torch.zeros(1, dtype=torch.bfloat16), requires_grad=False
        )
        self.config = SimpleNamespace(
            _attn_implementation="flash_attention_2", use_cache=False
        )
        self.stop_position = stop_position
        self.forward_calls: list[dict[str, Any]] = []
        self.return_cache = False
        self.return_nonfinite = False
        self.raise_forward = False
        self.raise_zero_grad = False
        self.zero_grad_calls = 0

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.zero_grad_calls += 1
        if self.raise_zero_grad:
            raise RuntimeError("injected zero_grad failure")
        super().zero_grad(set_to_none=set_to_none)

    def forward(self, **kwargs: Any) -> SimpleNamespace:
        call = {
            "input_ids": kwargs["input_ids"].detach().cpu().clone(),
            "attention_mask": kwargs["attention_mask"].detach().cpu().clone(),
            "use_cache": kwargs.get("use_cache"),
            "logits_to_keep": kwargs.get("logits_to_keep"),
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
            if position >= self.stop_position:
                stop_bias = torch.zeros_like(logits)
                stop_bias[:, 3] = 40.0
                logits = logits + stop_bias
            rows.append(logits)
        # Accelerate's native BF16 wrapper converts prepared-model outputs to
        # fp32 while parameters and autocast compute remain BF16.
        output = torch.stack(rows, dim=1).float()
        kept = kwargs.get("logits_to_keep", 0)
        if isinstance(kept, torch.Tensor):
            output = output[:, kept.detach().cpu().tolist(), :]
        elif isinstance(kept, int) and kept > 0:
            output = output[:, -kept:, :]
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
                "trainable_language_dora": {
                    "parameter_count": 1,
                    "parameter_names": ["weight"],
                },
                "frozen_selected_token_delta": {
                    "parameter_count": 1,
                    "parameter_names": ["shared_embed_delta"],
                },
                "frozen_vision": {
                    "parameter_count": 1,
                    "parameter_names": ["visual.weight"],
                },
                "frozen_aligner": {
                    "parameter_count": 1,
                    "parameter_names": ["visual.merger.weight"],
                },
            },
        }


class TinyOptimizer:
    state: dict[str, object] = {}

    def __init__(
        self, parameter: torch.nn.Parameter, *, learning_rate: float
    ) -> None:
        self.param_groups = [
            {
                "name": "adapter.language",
                "lr": learning_rate,
                "weight_decay": 0.0,
                "params": [parameter],
            }
        ]
        self.defaults = {"betas": (0.9, 0.999), "eps": 1.0e-8}


class TinyAccelerator:
    num_processes = 1
    process_index = 0
    mixed_precision = "bf16"
    gradient_accumulation_steps = 1
    distributed_type = SimpleNamespace(name="NO")
    device = "cuda:0"

    def __init__(self) -> None:
        self.free_memory_calls = 0
        self.raise_free_memory = False

    def free_memory(self) -> None:
        self.free_memory_calls += 1
        if self.raise_free_memory:
            raise RuntimeError("injected free_memory failure")


class TinyAssemblyBackend:
    def __init__(
        self,
        model: TinyCausalModel,
        tokenizer: TinyTokenizer,
        plan: live_model.Human13LiveModelPlan,
    ) -> None:
        self.model = model
        self.accelerator = TinyAccelerator()
        self.components = SimpleNamespace(
            model=model,
            base_model_path=Path(live_model.SOURCE_BASE_MODEL_PATH),
            base_config_sha256=live_model.SOURCE_BASE_CONFIG_SHA256,
            tokenizer_sha256=live_model.SOURCE_TOKENIZER_SHA256,
            tokenizer=tokenizer,
            processor=SimpleNamespace(name="tiny-processor"),
        )
        self.adapter_result = SimpleNamespace(
            model=model,
            receipt=SimpleNamespace(
                adapter_name="default",
                warm_start={
                    "source_adapter_tensor_sha256": live_model.SOURCE_ADAPTER_SHA256
                },
            ),
        )
        self.special_result = SimpleNamespace(
            model=model,
            shared_embed_delta=model.shared_embed_delta,
            receipt=SimpleNamespace(name="selected-delta"),
            loaded_tensor_sha256=live_model.SOURCE_SPECIAL_EMBEDDING_SHA256,
        )
        self.optimizer = TinyOptimizer(
            model.weight, learning_rate=plan.learning_rate
        )
        self.scheduler = object()
        self.group_plan = object()

    def create_accelerator(self, plan: live_model.Human13LiveModelPlan) -> Any:
        del plan
        return self.accelerator

    def validate_accelerator(
        self, accelerator: Any, plan: live_model.Human13LiveModelPlan
    ) -> None:
        del accelerator, plan

    def load_qwen(self, plan: live_model.Human13LiveModelPlan) -> Any:
        del plan
        return self.components

    def warm_start_language_dora(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return self.adapter_result

    def load_and_freeze_special_token_delta(
        self, *args: Any, **kwargs: Any
    ) -> Any:
        del args, kwargs
        return self.special_result

    def enable_memory_savers(self, model: Any) -> dict[str, bool]:
        del model
        return {"use_cache_disabled": True}

    def build_optimizer(self, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        del args, kwargs
        return self.optimizer, self.scheduler, self.group_plan

    def build_trainable_surface_receipt(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return SurfaceReceipt()

    def build_runtime(self, **kwargs: Any) -> Any:
        return SimpleNamespace(
            model=kwargs["model"],
            optimizer=kwargs["optimizer"],
            scheduler=kwargs["scheduler"],
            world_size=1,
        )


def _assembly(
    model: TinyCausalModel | None = None,
    tokenizer: TinyTokenizer | None = None,
    plan: live_model.Human13LiveModelPlan | None = None,
) -> Human13LiveAssembly:
    causal_model = model or TinyCausalModel()
    live_tokenizer = tokenizer or TinyTokenizer()
    effective_plan = plan or live_model.build_human13_all_hf_vertical_source_plan()
    backend = TinyAssemblyBackend(causal_model, live_tokenizer, effective_plan)
    return live_model.assemble_human13_live_model(
        effective_plan,
        pack_count=1,
        repo_root=Path.cwd(),
        backend=cast(live_model.Human13AssemblyBackend, backend),
    )


def _open(
    model: TinyCausalModel | None = None,
) -> tuple[Any, Human13LiveAssembly, TinySkeleton]:
    from scripts.research.human13_hf_shared_surface_live import (
        open_hf_shared_surface,
    )

    assembly = _assembly(model=model)
    skeleton = TinySkeleton()
    return (
        open_hf_shared_surface(plan_image1584_k16(), assembly, skeleton),
        assembly,
        skeleton,
    )


def _complete_k16(
    session: Any,
) -> tuple[tuple[SampledHFGroup, ...], tuple[GradientReplayGroup, ...]]:
    groups = tuple(
        session.sample_group(seeds) for seeds in plan_image1584_k16().seed_groups
    )
    replays = tuple(session.replay_group(group) for group in groups)
    return groups, replays


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

    groups, _replays = _complete_k16(session)

    assert tuple(group.group_index for group in groups) == (0, 1, 2, 3)
    assert tuple(request.seed for group in groups for request in group.requests) == tuple(
        range(35001, 35017)
    )
    receipt = session.close()
    assert receipt.sample_forward_count == 8
    assert receipt.replay_forward_count == 8
    assert receipt.no_cache_forward_count == receipt.total_forward_count == 16


def test_vectorized_replay_uses_same_eval_model_with_grad_and_causal_gathers() -> None:
    session, assembly, _skeleton = _open()
    model = assembly.model
    sampled = session.sample_group((35001, 35002, 35003, 35004))

    replay = session.replay_group(sampled)

    assert type(replay) is GradientReplayGroup
    replay_call = model.forward_calls[-1]
    assert replay_call["input_ids"].shape == torch.Size([3, 4])
    assert isinstance(replay_call["logits_to_keep"], torch.Tensor)
    assert replay_call["logits_to_keep"].tolist() == [3]
    assert replay_call["attention_mask"].tolist() == [
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
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
    assert all(isinstance(token.processed_logp, float) for token in replay.replayed_tokens)


def test_replay_reconstructs_each_sampler_step_length_for_fa2_parity() -> None:
    session, assembly, _skeleton = _open(model=TinyCausalModel(stop_position=6))
    model = assembly.model

    sampled = session.sample_group((35001, 35002, 35003, 35004))
    session.replay_group(sampled)

    sample_calls = model.forward_calls[: len(sampled.active_batch_steps)]
    replay_calls = model.forward_calls[len(sampled.active_batch_steps) :]
    assert len(sample_calls) == len(replay_calls) == len(sampled.active_batch_steps)
    assert [call["input_ids"].shape for call in replay_calls] == [
        call["input_ids"].shape for call in sample_calls
    ]
    assert [call["logits_to_keep"].tolist() for call in replay_calls] == [
        call["logits_to_keep"].tolist() for call in sample_calls
    ]


def test_grad_replay_checkpoints_each_step_before_accumulating_graphs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = 0
    import torch.utils.checkpoint as checkpoint_utils

    original = checkpoint_utils.checkpoint

    def wrapped(function: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal entered
        entered += 1
        assert kwargs.get("use_reentrant") is False
        return original(function, *args, **kwargs)

    monkeypatch.setattr(
        checkpoint_utils,
        "checkpoint",
        wrapped,
    )
    session, _assembly_value, _skeleton = _open()
    sampled = session.sample_group((35001, 35002, 35003, 35004))
    session.replay_group(sampled)
    assert entered == len(sampled.active_batch_steps)


def test_resource_receipt_binds_identity_processor_forwards_and_cleanup() -> None:
    session, assembly, _skeleton = _open()
    _sampled, replays = _complete_k16(session)
    replay = replays[-1]

    closed = session.close()
    assert closed.identity.model_object_id == id(assembly.model)
    assert closed.parameter_state_sha256 == closed.identity.parameter_state_sha256
    assert closed.processor_order == (
        "repetition_penalty",
        "temperature",
        "top_p",
    )
    assert closed.sample_forward_count == 8
    assert closed.replay_forward_count == 8
    assert closed.no_cache_forward_count == closed.total_forward_count == 16
    assert closed.observed_logits_dtype == "float32"
    assert closed.latest_replay_group_sha256 == replay.content_sha256

    assert closed.cleanup_state == "closed"
    assert closed.cleanup_call_count == 1
    assert closed.retained_graph_count == 0
    assert closed.session_held_reference_count == 0
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
        ("runtime", "runtime aliases"),
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
        object.__setattr__(
            assembly.plan.source, "checkpoint_path", "/substituted/checkpoint"
        )
    elif mutation == "adapter":
        object.__setattr__(
            assembly.plan.source, "adapter_sha256", _digest("different-adapter")
        )
    elif mutation == "delta":
        object.__setattr__(
            assembly.plan.source,
            "special_embedding_sha256",
            _digest("different-delta"),
        )
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
    elif mutation == "runtime":
        assembly.runtime.model = TinyCausalModel()
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


def test_replay_forward_fault_preserves_primary_and_seals_attempt_count() -> None:
    session, assembly, _skeleton = _open()
    sampled = session.sample_group(plan_image1584_k16().seed_groups[0])
    assembly.model.raise_forward = True

    with pytest.raises(RuntimeError, match="injected causal-forward failure"):
        session.replay_group(sampled)

    receipt = session.resource_receipt
    assert receipt.cleanup_reason == "failed"
    assert receipt.replay_forward_count == 1
    assert receipt.replay_group_sha256s == ()
    assert receipt.cleanup_call_count == 1
    assert receipt.session_held_reference_count == receipt.retained_graph_count == 0


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
    assert receipt.retained_graph_count == 0


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


def test_resource_receipt_is_sealed_and_rejects_incomplete_completion() -> None:
    from scripts.research.human13_hf_shared_surface_live import (
        SharedSurfaceResourceReceipt,
    )

    session, _assembly_value, _skeleton = _open()

    with pytest.raises(RuntimeError, match="four sampled and replayed groups"):
        session.close()

    failed = session.resource_receipt
    restored = SharedSurfaceResourceReceipt.from_dict(failed.to_dict())
    assert restored.to_dict() == failed.to_dict()
    with pytest.raises((RuntimeError, ValueError), match="admitted"):
        replace(failed, sample_forward_count=999).to_dict()
    with pytest.raises((RuntimeError, ValueError), match="admitted"):
        copy.copy(failed).to_dict()


def test_open_rejects_unadmitted_or_wrong_language_surface() -> None:
    from scripts.research.human13_hf_shared_surface_live import (
        open_hf_shared_surface,
    )

    assembly = _assembly()
    assembly.trainable_surface_receipt.to_artifact_dict = lambda: {
        **SurfaceReceipt().to_artifact_dict(),
        "trainable_towers": ["adapter.language", "adapter.vision"],
        "unmatched_trainable_names": ["visual.trainable"],
    }
    with pytest.raises(RuntimeError, match="admitted|trainable-surface|language"):
        open_hf_shared_surface(plan_image1584_k16(), assembly, TinySkeleton())


@pytest.mark.parametrize(
    "mutation",
    (
        "vision_trainable",
        "floating_delta",
        "validation_hash",
        "validation_delta_hash",
        "loaded_hash",
        "loaded_delta_hash",
    ),
)
def test_open_requires_exact_builder_issued_source_surface(mutation: str) -> None:
    from scripts.research.human13_hf_shared_surface_live import open_hf_shared_surface

    assembly = _assembly()
    if mutation == "vision_trainable":
        assembly.model.register_parameter(
            "vision_weight",
            torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16)),
        )
    elif mutation == "floating_delta":
        assembly.special_token_result.shared_embed_delta.requires_grad_(True)
    elif mutation == "validation_hash":
        assembly = replace(
            assembly,
            validation=replace(
                assembly.validation,
                adapter_tensor_sha256=_digest("wrong-loaded-adapter"),
            ),
        )
    elif mutation == "validation_delta_hash":
        assembly = replace(
            assembly,
            validation=replace(
                assembly.validation,
                special_embedding_tensor_sha256=_digest("wrong-loaded-delta"),
            ),
        )
    elif mutation == "loaded_hash":
        assembly.adapter_result.receipt.warm_start[
            "source_adapter_tensor_sha256"
        ] = _digest("wrong-adapter-result")
    else:
        assembly.special_token_result.loaded_tensor_sha256 = _digest(
            "wrong-delta-result"
        )

    with pytest.raises(
        RuntimeError, match="admitted|trainable|delta|validation|parameter"
    ):
        open_hf_shared_surface(plan_image1584_k16(), assembly, TinySkeleton())
    assert assembly.model.forward_calls == []


@pytest.mark.parametrize("parameter_name", ("weight", "shared_embed_delta"))
def test_open_rejects_parameter_content_mutated_after_builder_seal(
    parameter_name: str,
) -> None:
    from scripts.research.human13_hf_shared_surface_live import open_hf_shared_surface

    assembly = _assembly()
    parameter = dict(assembly.model.named_parameters())[parameter_name]
    with torch.no_grad():
        parameter.add_(1)

    with pytest.raises(RuntimeError, match="builder|parameter|Source"):
        open_hf_shared_surface(plan_image1584_k16(), assembly, TinySkeleton())
    assert assembly.model.forward_calls == []


def test_open_rejects_trainability_mutated_after_builder_seal() -> None:
    from scripts.research.human13_hf_shared_surface_live import open_hf_shared_surface

    assembly = _assembly()
    assembly.model.weight.requires_grad_(False)

    with pytest.raises(RuntimeError, match="builder|parameter|trainable"):
        open_hf_shared_surface(plan_image1584_k16(), assembly, TinySkeleton())
    assert assembly.model.forward_calls == []


def test_open_rejects_predecessor_a1_instead_of_owning_vertical_plan() -> None:
    from scripts.research.human13_hf_shared_surface_live import open_hf_shared_surface

    predecessor = _assembly(
        plan=live_model.build_human13_live_model_plan(CONFIG_ROOT / "03_a1.yaml")
    )
    with pytest.raises(RuntimeError, match="all-HF|vertical"):
        open_hf_shared_surface(
            plan_image1584_k16(), predecessor, TinySkeleton()
        )


def test_open_rejects_wrong_source_plan_and_nonbuilder_assembly() -> None:
    from scripts.research.human13_hf_shared_surface_live import open_hf_shared_surface

    assembly = _assembly()
    wrong_plan = replace(assembly.plan, unit_id=live_model.SUCCESSOR_UNIT_ID, arm_id="R1")
    for invalid in (replace(assembly, plan=wrong_plan), replace(assembly)):
        with pytest.raises(RuntimeError, match="admitted|Source"):
            open_hf_shared_surface(plan_image1584_k16(), invalid, TinySkeleton())
        assert invalid.model.forward_calls == []


def test_resource_receipt_rejects_outer_and_semantic_rehash_tamper() -> None:
    from scripts.research.human13_hf_shared_surface_live import (
        SharedSurfaceResourceReceipt,
    )

    session, _assembly_value, _skeleton = _open()
    with pytest.raises(RuntimeError):
        session.close()
    payload = session.resource_receipt.to_dict()
    outer_tamper = dict(payload)
    outer_tamper["content_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="content hash"):
        SharedSurfaceResourceReceipt.from_dict(outer_tamper)

    semantic_tamper = dict(payload)
    semantic_tamper["sample_forward_count"] = 999
    semantic_tamper["content_sha256"] = json_sha256(
        {key: value for key, value in semantic_tamper.items() if key != "content_sha256"}
    )
    with pytest.raises(RuntimeError, match="lifecycle values"):
        SharedSurfaceResourceReceipt.from_dict(semantic_tamper)


def test_completed_receipt_serializes_nested_task1_lineage() -> None:
    from scripts.research.human13_hf_shared_surface_live import (
        SharedSurfaceResourceReceipt,
    )

    session, _assembly_value, _skeleton = _open()
    _complete_k16(session)
    receipt = session.close()

    payload = receipt.to_dict()
    assert len(payload["sampled_groups"]) == 4
    assert len(payload["replay_groups"]) == 4
    restored = SharedSurfaceResourceReceipt.from_dict(payload)
    assert restored.to_dict() == payload


def test_canonical_rehash_cannot_fabricate_completed_k16_without_nested_groups() -> None:
    from scripts.research.human13_hf_shared_surface_live import (
        SharedSurfaceResourceReceipt,
    )

    session, _assembly_value, _skeleton = _open()
    with pytest.raises(RuntimeError):
        session.close()
    payload = session.resource_receipt.to_dict()
    payload["sampled_seed_groups"] = [
        list(group) for group in plan_image1584_k16().seed_groups
    ]
    payload["sampled_group_sha256s"] = [_digest(f"sample-{index}") for index in range(4)]
    payload["replay_group_sha256s"] = [_digest(f"replay-{index}") for index in range(4)]
    payload["sample_forward_count"] = 0
    payload["replay_forward_count"] = 4
    payload["total_forward_count"] = 4
    payload["no_cache_forward_count"] = 4
    payload["latest_replay_group_sha256"] = payload["replay_group_sha256s"][-1]
    payload["cleanup_reason"] = "completed"
    payload["content_sha256"] = json_sha256(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )

    with pytest.raises(RuntimeError, match="nested|lineage|completed"):
        SharedSurfaceResourceReceipt.from_dict(payload)


def test_nested_task1_tamper_and_derived_hash_rehash_are_rejected() -> None:
    from scripts.research.human13_hf_shared_surface_live import (
        SharedSurfaceResourceReceipt,
    )

    session, _assembly_value, _skeleton = _open()
    _complete_k16(session)
    payload = session.close().to_dict()

    nested_tamper = copy.deepcopy(payload)
    first_group = nested_tamper["sampled_groups"][0]
    first_group["group_index"] = 1
    first_group["content_sha256"] = json_sha256(
        {
            key: value
            for key, value in first_group.items()
            if key != "content_sha256"
        }
    )
    nested_tamper["content_sha256"] = json_sha256(
        {
            key: value
            for key, value in nested_tamper.items()
            if key != "content_sha256"
        }
    )
    with pytest.raises((RuntimeError, ValueError), match="seed|group|lineage"):
        SharedSurfaceResourceReceipt.from_dict(nested_tamper)

    derived_tamper = copy.deepcopy(payload)
    derived_tamper["sampled_group_sha256s"][0] = _digest("forged-derived-hash")
    derived_tamper["content_sha256"] = json_sha256(
        {
            key: value
            for key, value in derived_tamper.items()
            if key != "content_sha256"
        }
    )
    with pytest.raises(RuntimeError, match="derive|lineage"):
        SharedSurfaceResourceReceipt.from_dict(derived_tamper)


def test_close_rejects_one_group_missing_replay_and_wrong_order() -> None:
    one, _assembly_value, _skeleton = _open()
    one.sample_group(plan_image1584_k16().seed_groups[0])
    with pytest.raises(RuntimeError, match="four sampled and replayed groups"):
        one.close()
    assert one.resource_receipt.cleanup_reason == "failed"

    missing, _assembly_value, _skeleton = _open()
    for seeds in plan_image1584_k16().seed_groups:
        missing.sample_group(seeds)
    with pytest.raises(RuntimeError, match="four sampled and replayed groups"):
        missing.close()
    assert len(missing.resource_receipt.sampled_group_sha256s) == 4
    assert missing.resource_receipt.replay_group_sha256s == ()

    wrong, _assembly_value, _skeleton = _open()
    with pytest.raises(RuntimeError, match="frozen K16 order"):
        wrong.sample_group(plan_image1584_k16().seed_groups[1])
    assert wrong.resource_receipt.cleanup_reason == "failed"

    reordered, _assembly_value, _skeleton = _open()
    for seeds in plan_image1584_k16().seed_groups:
        reordered.sample_group(seeds)
    reordered._sampled_groups.reverse()  # noqa: SLF001 - adversarial close input
    with pytest.raises(RuntimeError, match="four sampled and replayed groups"):
        reordered.close()
    assert reordered.resource_receipt.cleanup_reason == "failed"


def test_valid_k16_close_clears_only_session_references() -> None:
    session, assembly, _skeleton = _open()
    caller_model = assembly.model
    _complete_k16(session)

    receipt = session.close()

    assert receipt.cleanup_reason == "completed"
    assert len(receipt.sampled_group_sha256s) == 4
    assert len(receipt.replay_group_sha256s) == 4
    assert receipt.sampled_seed_groups == plan_image1584_k16().seed_groups
    assert receipt.session_held_reference_count == 0
    assert receipt.assembly_ownership == "borrowed_external"
    assert receipt.caller_release_claim == "not_claimed"
    assert assembly.model is caller_model
    assert session._retained_live_resource_count() == 0  # noqa: SLF001
    assert session._sampled_groups == []  # noqa: SLF001
    assert session._replayed_groups == []  # noqa: SLF001


@pytest.mark.parametrize("failure", ("zero_grad", "free_memory"))
def test_explicit_close_cleanup_failure_is_terminal_and_not_retried(
    failure: str,
) -> None:
    session, assembly, _skeleton = _open()
    _complete_k16(session)
    if failure == "zero_grad":
        assembly.model.raise_zero_grad = True
    else:
        assembly.accelerator.raise_free_memory = True

    with pytest.raises(RuntimeError, match="cleanup failed"):
        session.close()

    receipt = session.resource_receipt
    assert receipt.cleanup_reason == "failed"
    assert receipt.cleanup_call_count == 1
    assert receipt.session_held_reference_count == receipt.retained_graph_count == 0
    assert len(receipt.cleanup_failures) == 1
    assert assembly.model.zero_grad_calls == 1
    assert assembly.accelerator.free_memory_calls == 1
    with pytest.raises(RuntimeError, match="already closed"):
        session.close()
    assert assembly.model.zero_grad_calls == 1
    assert assembly.accelerator.free_memory_calls == 1


def test_open_failure_cleanup_preserves_primary_and_attempts_every_cleanup() -> None:
    from scripts.research.human13_hf_shared_surface_live import open_hf_shared_surface

    assembly = _assembly()
    assembly.model.raise_zero_grad = True
    assembly.accelerator.raise_free_memory = True

    with pytest.raises(RuntimeError, match="image 1584") as failure:
        open_hf_shared_surface(
            plan_image1584_k16(),
            assembly,
            TinySkeleton(human13_image_id=2299),
        )

    assert assembly.model.zero_grad_calls == 1
    assert assembly.accelerator.free_memory_calls == 1
    assert len(getattr(failure.value, "__notes__", ())) == 2
    assert assembly.runtime.model is assembly.model


def test_context_primary_exception_survives_cleanup_subfailures() -> None:
    session, assembly, _skeleton = _open()
    assembly.model.raise_zero_grad = True
    assembly.accelerator.raise_free_memory = True

    with pytest.raises(RuntimeError, match="body remains primary"):
        with session:
            sampled = session.sample_group(plan_image1584_k16().seed_groups[0])
            session.replay_group(sampled)
            raise RuntimeError("body remains primary")

    receipt = session.resource_receipt
    assert receipt.cleanup_reason == "failed"
    assert len(receipt.cleanup_failures) == 2
    assert receipt.cleanup_call_count == 1
    assert receipt.session_held_reference_count == receipt.retained_graph_count == 0
    assert assembly.model.zero_grad_calls == 1
    assert assembly.accelerator.free_memory_calls == 1
