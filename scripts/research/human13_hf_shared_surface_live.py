"""Live-call plumbing for the Human-13 all-HF shared surface.

This experiment-local owner performs no loading or publication.  It consumes an
already admitted :class:`Human13LiveAssembly`, keeps one BF16/FA2 model object
in eval mode, and delegates serialized sampling/replay admission to the Task-1
contracts in ``human13_hf_shared_surface``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from collections.abc import Callable, Iterable
from typing import Any, Literal, Mapping, cast
from weakref import ReferenceType, ref

import torch

from scripts.research.human13_adamw_proposal_preservation import (
    ParameterLayout,
    parameter_state_sha256,
)
from scripts.research.human13_hf_shared_surface import (
    GradientReplayGroup,
    HFActiveBatchStep,
    HFReplayCausalGather,
    HFSharedSurfaceIdentity,
    HFSharedSurfacePlan,
    SampledHFGroup,
    SampledHFRequest,
    SampledHFToken,
    admit_gradient_replay,
    admit_sampled_group,
    causal_history_sha256,
    plan_image1584_k16,
)
from scripts.research import human13_live_model as live_model
from scripts.research.human13_live_model import Human13LiveAssembly
from src.artifacts.json_values import json_sha256
from src.inference.hf_backend import _derive_qwen_position_ids


_PROCESSOR_ORDER = ("repetition_penalty", "temperature", "top_p")
_RESOURCE_SEALS: dict[int, tuple[ReferenceType[object], str]] = {}


class HFSharedSurfaceLiveError(RuntimeError):
    """The live shared surface no longer matches its admitted identity."""


@dataclass(frozen=True)
class SharedSurfaceResourceReceipt:
    """Sealed terminal lifecycle evidence; it owns no live object or tensor."""

    identity: HFSharedSurfaceIdentity
    model_object_id: int
    parameter_state_sha256: str
    processor_object_id: int
    tokenizer_object_id: int
    processor_order: tuple[str, str, str]
    observed_logits_dtype: Literal["float32"]
    sampled_seed_groups: tuple[tuple[int, int, int, int], ...]
    sampled_group_sha256s: tuple[str, ...]
    replay_group_sha256s: tuple[str, ...]
    sample_forward_count: int
    replay_forward_count: int
    total_forward_count: int
    no_cache_forward_count: int
    retained_graph_count: Literal[0]
    latest_replay_group_sha256: str | None
    session_held_reference_count: Literal[0]
    assembly_ownership: Literal["borrowed_external"]
    caller_release_claim: Literal["not_claimed"]
    cleanup_state: Literal["closed"]
    cleanup_reason: Literal["completed", "failed"]
    cleanup_failures: tuple[str, ...]
    cleanup_call_count: Literal[1]

    def _payload(self) -> dict[str, object]:
        return {
            "identity": self.identity.to_dict(),
            "model_object_id": self.model_object_id,
            "parameter_state_sha256": self.parameter_state_sha256,
            "processor_object_id": self.processor_object_id,
            "tokenizer_object_id": self.tokenizer_object_id,
            "processor_order": list(self.processor_order),
            "observed_logits_dtype": self.observed_logits_dtype,
            "sampled_seed_groups": [list(group) for group in self.sampled_seed_groups],
            "sampled_group_sha256s": list(self.sampled_group_sha256s),
            "replay_group_sha256s": list(self.replay_group_sha256s),
            "sample_forward_count": self.sample_forward_count,
            "replay_forward_count": self.replay_forward_count,
            "total_forward_count": self.total_forward_count,
            "no_cache_forward_count": self.no_cache_forward_count,
            "retained_graph_count": self.retained_graph_count,
            "latest_replay_group_sha256": self.latest_replay_group_sha256,
            "session_held_reference_count": self.session_held_reference_count,
            "assembly_ownership": self.assembly_ownership,
            "caller_release_claim": self.caller_release_claim,
            "cleanup_state": self.cleanup_state,
            "cleanup_reason": self.cleanup_reason,
            "cleanup_failures": list(self.cleanup_failures),
            "cleanup_call_count": self.cleanup_call_count,
        }

    @property
    def content_sha256(self) -> str:
        _require_resource_receipt(self)
        return json_sha256(self._payload())

    def to_dict(self) -> dict[str, object]:
        _require_resource_receipt(self)
        return self._payload() | {"content_sha256": self.content_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SharedSurfaceResourceReceipt:
        expected = set(cls.__dataclass_fields__) | {"content_sha256"}
        if set(value) != expected or not isinstance(value.get("identity"), Mapping):
            raise HFSharedSurfaceLiveError(
                "resource receipt fields differ from canonical schema"
            )
        receipt = cls(
            identity=HFSharedSurfaceIdentity.from_dict(value["identity"]),
            model_object_id=value["model_object_id"],
            parameter_state_sha256=value["parameter_state_sha256"],
            processor_object_id=value["processor_object_id"],
            tokenizer_object_id=value["tokenizer_object_id"],
            processor_order=tuple(value["processor_order"]),
            observed_logits_dtype=value["observed_logits_dtype"],
            sampled_seed_groups=tuple(
                tuple(group) for group in value["sampled_seed_groups"]
            ),
            sampled_group_sha256s=tuple(value["sampled_group_sha256s"]),
            replay_group_sha256s=tuple(value["replay_group_sha256s"]),
            sample_forward_count=value["sample_forward_count"],
            replay_forward_count=value["replay_forward_count"],
            total_forward_count=value["total_forward_count"],
            no_cache_forward_count=value["no_cache_forward_count"],
            retained_graph_count=value["retained_graph_count"],
            latest_replay_group_sha256=value["latest_replay_group_sha256"],
            session_held_reference_count=value["session_held_reference_count"],
            assembly_ownership=value["assembly_ownership"],
            caller_release_claim=value["caller_release_claim"],
            cleanup_state=value["cleanup_state"],
            cleanup_reason=value["cleanup_reason"],
            cleanup_failures=tuple(value["cleanup_failures"]),
            cleanup_call_count=value["cleanup_call_count"],
        )
        _admit_resource_receipt(receipt)
        if value["content_sha256"] != receipt.content_sha256:
            raise HFSharedSurfaceLiveError("resource receipt content hash differs")
        return receipt


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise HFSharedSurfaceLiveError(f"{label} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise HFSharedSurfaceLiveError(f"{label} must be a SHA-256 digest") from exc
    return value


def _validate_resource_receipt(receipt: SharedSurfaceResourceReceipt) -> None:
    if type(receipt.identity) is not HFSharedSurfaceIdentity:
        raise HFSharedSurfaceLiveError("resource receipt identity is not exact")
    for label, values in (
        ("sampled group", receipt.sampled_group_sha256s),
        ("replay group", receipt.replay_group_sha256s),
    ):
        for value in values:
            _digest(value, label=label)
    counts = (
        receipt.sample_forward_count,
        receipt.replay_forward_count,
        receipt.total_forward_count,
        receipt.no_cache_forward_count,
    )
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
        raise HFSharedSurfaceLiveError("resource receipt forward counts are invalid")
    if (
        receipt.processor_order != _PROCESSOR_ORDER
        or receipt.observed_logits_dtype != "float32"
        or receipt.parameter_state_sha256 != receipt.identity.parameter_state_sha256
        or receipt.model_object_id != receipt.identity.model_object_id
        or receipt.total_forward_count
        != receipt.sample_forward_count + receipt.replay_forward_count
        or receipt.no_cache_forward_count != receipt.total_forward_count
        or receipt.replay_forward_count < len(receipt.replay_group_sha256s)
        or receipt.replay_forward_count > len(receipt.replay_group_sha256s) + 1
        or len(receipt.sampled_seed_groups) != len(receipt.sampled_group_sha256s)
        or len(receipt.replay_group_sha256s) > len(receipt.sampled_group_sha256s)
        or receipt.retained_graph_count != 0
        or receipt.session_held_reference_count != 0
        or receipt.assembly_ownership != "borrowed_external"
        or receipt.caller_release_claim != "not_claimed"
        or receipt.cleanup_state != "closed"
        or receipt.cleanup_call_count != 1
    ):
        raise HFSharedSurfaceLiveError("resource receipt lifecycle values differ")
    if (
        len(set(receipt.sampled_seed_groups)) != len(receipt.sampled_seed_groups)
        or any(
            group not in plan_image1584_k16().seed_groups
            for group in receipt.sampled_seed_groups
        )
    ):
        raise HFSharedSurfaceLiveError("resource receipt seed coverage differs")
    expected_latest = (
        receipt.replay_group_sha256s[-1] if receipt.replay_group_sha256s else None
    )
    if receipt.latest_replay_group_sha256 != expected_latest:
        raise HFSharedSurfaceLiveError("resource receipt replay lineage differs")
    completed = receipt.cleanup_reason == "completed"
    if completed and (
        receipt.sampled_seed_groups != plan_image1584_k16().seed_groups
        or len(receipt.sampled_group_sha256s) != 4
        or len(receipt.replay_group_sha256s) != 4
        or receipt.replay_forward_count != 4
        or receipt.cleanup_failures
    ):
        raise HFSharedSurfaceLiveError(
            "completed resource receipt requires four sampled and replayed groups"
        )
    if receipt.cleanup_reason not in ("completed", "failed"):
        raise HFSharedSurfaceLiveError("resource receipt terminal reason differs")
    if any(not isinstance(item, str) or not item for item in receipt.cleanup_failures):
        raise HFSharedSurfaceLiveError("resource receipt cleanup failures are invalid")


def _admit_resource_receipt(
    receipt: SharedSurfaceResourceReceipt,
) -> SharedSurfaceResourceReceipt:
    _validate_resource_receipt(receipt)
    fingerprint = json_sha256(receipt._payload())
    identity = id(receipt)

    def cleanup(_: ReferenceType[object], *, key: int = identity) -> None:
        _RESOURCE_SEALS.pop(key, None)

    _RESOURCE_SEALS[identity] = (ref(receipt, cleanup), fingerprint)
    return receipt


def _require_resource_receipt(receipt: SharedSurfaceResourceReceipt) -> None:
    entry = _RESOURCE_SEALS.get(id(receipt))
    if (
        entry is None
        or entry[0]() is not receipt
        or entry[1] != json_sha256(receipt._payload())
    ):
        raise HFSharedSurfaceLiveError(
            "resource receipt was not admitted through the lifecycle choke point"
        )


def _tensor_sha256(tensor: torch.Tensor, *, label: str) -> str:
    if not isinstance(tensor, torch.Tensor):
        raise HFSharedSurfaceLiveError(f"{label} must be a tensor")
    value = tensor.detach().cpu().contiguous()
    hasher = hashlib.sha256()
    hasher.update(str(value.dtype).encode())
    hasher.update(b"\0")
    hasher.update(repr(tuple(int(size) for size in value.shape)).encode())
    hasher.update(b"\0")
    hasher.update(value.view(torch.uint8).numpy().tobytes())
    return hasher.hexdigest()


def _prompt_tokens(skeleton: Any) -> tuple[int, ...]:
    values = getattr(skeleton, "input_ids", None)
    count = getattr(skeleton, "prompt_token_count", None)
    if (
        not isinstance(values, tuple)
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        or count >= len(values)
        or any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in values)
    ):
        raise HFSharedSurfaceLiveError("image-1584 skeleton prompt is malformed")
    return tuple(values[:count])


def _image_id(skeleton: Any) -> int:
    value = getattr(skeleton, "human13_image_id", None)
    if value is None:
        example_id = getattr(skeleton, "example_id", None)
        if example_id == "coco2017_val_000000001584":
            value = 1584
    if value != 1584:
        raise HFSharedSurfaceLiveError("shared-surface skeleton must be image 1584")
    return 1584


def _materialized_image(skeleton: Any) -> tuple[torch.Tensor, tuple[int, int, int]]:
    encoding = getattr(skeleton, "image_encoding", None)
    if encoding is None:
        raise HFSharedSurfaceLiveError("image-1584 skeleton lacks image encoding")
    pixels = getattr(encoding, "pixel_values", None)
    if not isinstance(pixels, torch.Tensor):
        try:
            from src.qwen import materialize_qwen_image_encoding

            encoding = materialize_qwen_image_encoding(encoding)
        except Exception as exc:
            raise HFSharedSurfaceLiveError(
                "image-1584 skeleton image materialization failed"
            ) from exc
        pixels = getattr(encoding, "pixel_values", None)
    grid = getattr(encoding, "image_grid_thw", None)
    if (
        not isinstance(pixels, torch.Tensor)
        or pixels.ndim != 2
        or not isinstance(grid, tuple)
        or len(grid) != 3
        or any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in grid)
    ):
        raise HFSharedSurfaceLiveError("image-1584 skeleton image payload is malformed")
    return pixels, grid


def _image_sha256(skeleton: Any) -> str:
    pixels, grid = _materialized_image(skeleton)
    encoding = skeleton.image_encoding
    plan = getattr(encoding, "plan", None)
    content = getattr(plan, "image_content_sha256", None)
    if not isinstance(content, str) or len(content) != 64:
        raise HFSharedSurfaceLiveError("image-1584 skeleton image identity is missing")
    return json_sha256(
        {
            "image_id": _image_id(skeleton),
            "image_content_sha256": content,
            "pixel_values_sha256": _tensor_sha256(pixels, label="pixel values"),
            "image_grid_thw": list(grid),
        }
    )


def _named_parameters(model: Any) -> tuple[tuple[str, torch.Tensor], ...]:
    method = getattr(model, "named_parameters", None)
    if not callable(method):
        raise HFSharedSurfaceLiveError("shared-surface model lacks named parameters")
    typed_method = cast(
        Callable[[], Iterable[tuple[str, torch.Tensor]]], method
    )
    values = tuple(typed_method())
    if not values or any(
        not isinstance(item, tuple)
        or len(item) != 2
        or not isinstance(item[0], str)
        or not isinstance(item[1], torch.Tensor)
        for item in values
    ):
        raise HFSharedSurfaceLiveError("shared-surface parameter layout is malformed")
    return values


def _parameter_versions(
    values: tuple[tuple[str, torch.Tensor], ...],
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            name,
            id(parameter),
            tuple(int(size) for size in parameter.shape),
            str(parameter.dtype),
            bool(parameter.requires_grad),
            int(getattr(parameter, "_version", -1)),
        )
        for name, parameter in values
    )


def _trainable_state(
    values: tuple[tuple[str, torch.Tensor], ...],
) -> tuple[ParameterLayout, str]:
    trainable = tuple((name, value) for name, value in values if value.requires_grad)
    if not trainable:
        raise HFSharedSurfaceLiveError(
            "shared-surface model lacks trainable language DoRA parameters"
        )
    try:
        layout = ParameterLayout.from_named_parameters(trainable)
        return layout, parameter_state_sha256(trainable, layout)
    except Exception as exc:
        raise HFSharedSurfaceLiveError("shared-surface parameter state is invalid") from exc


def _observed_attention_backends(model: Any) -> tuple[str, ...]:
    pending = [model]
    seen: set[int] = set()
    values: list[str] = []
    while pending:
        owner = pending.pop()
        if owner is None or id(owner) in seen:
            continue
        seen.add(id(owner))
        config = getattr(owner, "config", None)
        observed = getattr(config, "_attn_implementation", None)
        if isinstance(observed, str):
            values.append(observed)
        for name in ("model", "base_model", "module"):
            nested = getattr(owner, name, None)
            if nested is not None and nested is not owner:
                pending.append(nested)
    return tuple(values)


def _rng_sha256(generators: tuple[torch.Generator, ...]) -> str:
    return json_sha256(
        [
            {
                "index": index,
                "state_sha256": _tensor_sha256(
                    generator.get_state(), label="RNG state"
                ),
            }
            for index, generator in enumerate(generators)
        ]
    )


def _checkpoint_payload_sha256(assembly: Human13LiveAssembly) -> str:
    source = assembly.plan.source
    validation = assembly.validation
    return json_sha256(
        {
            "checkpoint_path": getattr(source, "checkpoint_path", None),
            "base_config_sha256": getattr(validation, "base_config_sha256", None),
            "adapter_sha256": getattr(source, "adapter_sha256", None),
            "embedding_delta_sha256": getattr(
                source, "special_embedding_sha256", None
            ),
        }
    )


def _require_exact_source_assembly(assembly: Human13LiveAssembly) -> None:
    """Re-admit builder provenance before publishing a shared-surface identity."""

    try:
        live_model.validate_human13_live_assembly_values(assembly)
    except live_model.Human13LiveModelError as exc:
        raise HFSharedSurfaceLiveError(str(exc)) from exc
    if assembly.plan.unit_id != live_model.UNIT_ID or assembly.plan.arm_id != "A1":
        raise HFSharedSurfaceLiveError(
            "shared-surface assembly must be the exact Human-13 Source A1 plan"
        )
    validation = assembly.validation
    source = assembly.plan.source
    if (
        type(validation) is not live_model.Human13PlanValidationReceipt
        or validation.arm_id != assembly.plan.arm_id
        or validation.schema_version != live_model.VALIDATION_SCHEMA_VERSION
        or validation.adapter_tensor_sha256 != source.adapter_sha256
        or validation.special_embedding_tensor_sha256
        != source.special_embedding_sha256
    ):
        raise HFSharedSurfaceLiveError(
            "shared-surface loaded adapter/delta validation hashes differ"
        )
    if (
        getattr(
            getattr(assembly.special_token_result, "shared_embed_delta", None),
            "requires_grad",
            None,
        )
        is not False
    ):
        raise HFSharedSurfaceLiveError(
            "shared-surface selected-token delta must remain frozen"
        )
    surface = assembly.trainable_surface_receipt.to_artifact_dict()
    exact = surface.get("exact_surface_groups")
    language = (
        exact.get("trainable_language_dora") if isinstance(exact, Mapping) else None
    )
    expected_trainable_names = (
        tuple(language.get("parameter_names", ()))
        if isinstance(language, Mapping)
        else ()
    )
    actual_trainable_names = tuple(
        name
        for name, parameter in _named_parameters(assembly.model)
        if parameter.requires_grad
    )
    if (
        not expected_trainable_names
        or actual_trainable_names != expected_trainable_names
    ):
        raise HFSharedSurfaceLiveError(
            "shared-surface actual trainables must be exact language DoRA only"
        )
    if (
        getattr(assembly.runtime, "model", None) is not assembly.model
        or getattr(assembly.runtime, "optimizer", None) is not assembly.optimizer
        or getattr(assembly.runtime, "scheduler", None) is not assembly.scheduler
    ):
        raise HFSharedSurfaceLiveError(
            "shared-surface runtime substituted prepared model/state aliases"
        )


def _apply_policy(
    input_ids: torch.Tensor,
    logits: torch.Tensor,
    plan: HFSharedSurfacePlan,
) -> torch.Tensor:
    """Use the HF generation processors in the frozen Task-1 order."""

    from transformers.generation.logits_process import (
        RepetitionPenaltyLogitsProcessor,
        TemperatureLogitsWarper,
        TopPLogitsWarper,
    )

    scores = RepetitionPenaltyLogitsProcessor(
        penalty=plan.policy.repetition_penalty
    )(input_ids, logits)  # pyright: ignore[reportArgumentType] - HF stub aliases
    scores = TemperatureLogitsWarper(temperature=plan.policy.temperature)(
        input_ids, scores  # pyright: ignore[reportArgumentType] - HF stub aliases
    )
    return TopPLogitsWarper(top_p=plan.policy.top_p)(
        input_ids, scores  # pyright: ignore[reportArgumentType] - HF stub aliases
    )


class HFSharedSurfaceSession:
    """One terminal-on-failure, no-cache sampling and replay lifecycle."""

    def __init__(
        self,
        plan: HFSharedSurfacePlan,
        assembly: Human13LiveAssembly,
        skeleton: Any,
    ) -> None:
        if type(plan) is not HFSharedSurfacePlan or plan != plan_image1584_k16():
            raise HFSharedSurfaceLiveError(
                "shared-surface session requires the frozen image-1584 K16 plan"
            )
        if type(assembly) is not Human13LiveAssembly:
            raise HFSharedSurfaceLiveError(
                "shared-surface session requires Human13LiveAssembly"
            )
        _require_exact_source_assembly(assembly)
        if (
            getattr(assembly.plan, "mixed_precision", None) != "bf16"
            or getattr(assembly.plan, "attn_implementation", None)
            != "flash_attention_2"
            or getattr(assembly.plan, "adapter_target_towers", None) != ("language",)
            or getattr(assembly.plan, "freeze_special_token_delta", None) is not True
        ):
            raise HFSharedSurfaceLiveError(
                "shared-surface assembly must be BF16/flash_attention_2 language DoRA"
            )
        _image_id(skeleton)
        self._plan = plan
        self._assembly = assembly
        self._skeleton = skeleton
        self._model = assembly.model
        self._expected_model = assembly.model
        self._components = assembly.components
        self._tokenizer = getattr(assembly.components, "tokenizer", None)
        self._processor = getattr(assembly.components, "processor", None)
        if self._tokenizer is None or self._processor is None:
            raise HFSharedSurfaceLiveError(
                "shared-surface assembly lacks tokenizer or processor"
            )
        if getattr(assembly.runtime, "model", None) is not self._model:
            raise HFSharedSurfaceLiveError(
                "shared-surface runtime substituted the live model object"
            )
        self._model.eval()
        self._model_object_id = id(self._model)
        self._tokenizer_object_id = id(self._tokenizer)
        self._processor_object_id = id(self._processor)
        self._parameters = _named_parameters(self._model)
        self._parameter_versions = _parameter_versions(self._parameters)
        self._parameter_layout, parameter_sha256 = _trainable_state(self._parameters)
        self._prompt = _prompt_tokens(skeleton)
        prompt_sha256 = json_sha256(list(self._prompt))
        image_sha256 = _image_sha256(skeleton)
        validation = assembly.validation
        tokenizer_sha256 = getattr(assembly.components, "tokenizer_sha256", None)
        if (
            not isinstance(tokenizer_sha256, str)
            or tokenizer_sha256 != getattr(validation, "tokenizer_sha256", None)
        ):
            raise HFSharedSurfaceLiveError("shared-surface tokenizer identity drifted")
        source = assembly.plan.source
        checkpoint_payload_sha256 = _checkpoint_payload_sha256(assembly)
        self._identity = HFSharedSurfaceIdentity(
            checkpoint_payload_sha256=checkpoint_payload_sha256,
            model_object_id=id(self._model),
            parameter_state_sha256=parameter_sha256,
            adapter_sha256=source.adapter_sha256,
            embedding_delta_sha256=source.special_embedding_sha256,
            dtype="bfloat16",
            attention_backend="flash_attention_2",
            model_mode="eval",
            tokenizer_sha256=tokenizer_sha256,
            prompt_sha256=prompt_sha256,
            image_sha256=image_sha256,
            use_cache=False,
        )
        self._sample_forward_count = 0
        self._replay_forward_count = 0
        self._no_cache_forward_count = 0
        self._sampled_groups: list[
            tuple[tuple[int, int, int, int], str]
        ] = []
        self._replayed_groups: list[tuple[str, str]] = []
        self._live_replay_tensors: dict[str, torch.Tensor] = {}
        self._latest_replay_group_sha256: str | None = None
        self._closed = False
        self._cleanup_reason: Literal["completed", "failed"] | None = None
        self._cleanup_call_count = 0
        self._terminal_receipt: SharedSurfaceResourceReceipt | None = None
        self._require_invariants()

    @property
    def resource_receipt(self) -> SharedSurfaceResourceReceipt:
        if self._terminal_receipt is None:
            raise HFSharedSurfaceLiveError(
                "resource receipt is available only after terminal cleanup"
            )
        return self._terminal_receipt

    def _build_terminal_receipt(
        self,
        *,
        reason: Literal["completed", "failed"],
        cleanup_failures: tuple[str, ...],
    ) -> SharedSurfaceResourceReceipt:
        return _admit_resource_receipt(SharedSurfaceResourceReceipt(
            identity=self._identity,
            model_object_id=self._model_object_id,
            parameter_state_sha256=self._identity.parameter_state_sha256,
            processor_object_id=self._processor_object_id,
            tokenizer_object_id=self._tokenizer_object_id,
            processor_order=_PROCESSOR_ORDER,
            observed_logits_dtype="float32",
            sampled_seed_groups=tuple(
                seeds for seeds, _sha256 in self._sampled_groups
            ),
            sampled_group_sha256s=tuple(
                sha256 for _seeds, sha256 in self._sampled_groups
            ),
            replay_group_sha256s=tuple(
                replay_sha256 for _sampled_sha256, replay_sha256 in self._replayed_groups
            ),
            sample_forward_count=self._sample_forward_count,
            replay_forward_count=self._replay_forward_count,
            total_forward_count=(
                self._sample_forward_count + self._replay_forward_count
            ),
            no_cache_forward_count=self._no_cache_forward_count,
            retained_graph_count=0,
            latest_replay_group_sha256=self._latest_replay_group_sha256,
            session_held_reference_count=0,
            assembly_ownership="borrowed_external",
            caller_release_claim="not_claimed",
            cleanup_state="closed",
            cleanup_reason=reason,
            cleanup_failures=cleanup_failures,
            cleanup_call_count=1,
        ))

    @property
    def live_replay_tensor_count(self) -> int:
        return sum(int(value.numel()) for value in self._live_replay_tensors.values())

    def _retained_live_resource_count(self) -> int:
        resources = (
            self._model,
            self._expected_model,
            self._assembly,
            self._components,
            self._skeleton,
            self._tokenizer,
            self._processor,
            self._parameters,
        )
        return sum(value is not None and value != () for value in resources)

    def __enter__(self) -> HFSharedSurfaceSession:
        self._require_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> Literal[False]:
        del exc_type, traceback
        if not self._closed:
            if exc is not None:
                self._close_internal("failed", primary_exception=exc)
            else:
                self.close()
        return False

    def _require_open(self) -> None:
        if self._closed:
            raise HFSharedSurfaceLiveError("shared-surface session is already closed")

    def _require_invariants(self, *, verify_parameter_values: bool = True) -> None:
        self._require_open()
        model = self._model
        assembly = self._assembly
        components = self._components
        skeleton = self._skeleton
        tokenizer = self._tokenizer
        processor = self._processor
        if (
            model is None
            or assembly is None
            or components is None
            or skeleton is None
            or tokenizer is None
            or processor is None
        ):
            raise HFSharedSurfaceLiveError(
                "shared-surface session live references are unavailable"
            )
        if (
            model is not self._expected_model
            or id(model) != self._identity.model_object_id
        ):
            raise HFSharedSurfaceLiveError("shared-surface model object was substituted")
        if model.training:
            raise HFSharedSurfaceLiveError("shared-surface model must remain in eval mode")
        current = _named_parameters(model)
        dtypes = {str(parameter.dtype) for _name, parameter in current}
        if dtypes != {"torch.bfloat16"}:
            raise HFSharedSurfaceLiveError(
                "shared-surface parameters must remain exclusively bfloat16"
            )
        if _parameter_versions(current) != self._parameter_versions:
            raise HFSharedSurfaceLiveError("shared-surface parameter state changed")
        if verify_parameter_values:
            _layout, state_sha256 = _trainable_state(current)
            if state_sha256 != self._identity.parameter_state_sha256:
                raise HFSharedSurfaceLiveError("shared-surface parameter state changed")
        backends = _observed_attention_backends(model)
        if not backends or any(value != "flash_attention_2" for value in backends):
            raise HFSharedSurfaceLiveError(
                "shared-surface attention backend must remain flash_attention_2"
            )
        if any(
            getattr(getattr(owner, "config", None), "use_cache", False) is not False
            for owner in (model,)
        ):
            raise HFSharedSurfaceLiveError("shared-surface cache configuration drifted")
        if getattr(components, "tokenizer", None) is not tokenizer:
            raise HFSharedSurfaceLiveError("shared-surface tokenizer object was substituted")
        if (
            getattr(components, "tokenizer_sha256", None)
            != self._identity.tokenizer_sha256
            or getattr(assembly.validation, "tokenizer_sha256", None)
            != self._identity.tokenizer_sha256
        ):
            raise HFSharedSurfaceLiveError("shared-surface tokenizer identity drifted")
        if getattr(components, "processor", None) is not processor:
            raise HFSharedSurfaceLiveError("shared-surface processor object was substituted")
        if (
            getattr(assembly.runtime, "model", None) is not model
            or getattr(assembly.runtime, "optimizer", None) is not assembly.optimizer
            or getattr(assembly.runtime, "scheduler", None) is not assembly.scheduler
        ):
            raise HFSharedSurfaceLiveError(
                "shared-surface prepared runtime aliases were substituted"
            )
        source = assembly.plan.source
        if getattr(source, "adapter_sha256", None) != self._identity.adapter_sha256:
            raise HFSharedSurfaceLiveError("shared-surface adapter identity drifted")
        if (
            getattr(source, "special_embedding_sha256", None)
            != self._identity.embedding_delta_sha256
        ):
            raise HFSharedSurfaceLiveError(
                "shared-surface embedding delta identity drifted"
            )
        if (
            _checkpoint_payload_sha256(assembly)
            != self._identity.checkpoint_payload_sha256
        ):
            raise HFSharedSurfaceLiveError(
                "shared-surface checkpoint payload identity drifted"
            )
        if _prompt_tokens(skeleton) != self._prompt:
            raise HFSharedSurfaceLiveError("shared-surface prompt identity drifted")
        if _image_sha256(skeleton) != self._identity.image_sha256:
            raise HFSharedSurfaceLiveError("shared-surface image identity drifted")
        if verify_parameter_values:
            try:
                live_model.validate_human13_live_assembly_values(assembly)
            except live_model.Human13LiveModelError as exc:
                raise HFSharedSurfaceLiveError(str(exc)) from exc

    def _forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        retain_grad: bool,
    ) -> torch.Tensor:
        self._require_invariants(verify_parameter_values=False)
        model = self._model
        tokenizer = self._tokenizer
        skeleton = self._skeleton
        if model is None or tokenizer is None or skeleton is None:
            raise HFSharedSurfaceLiveError(
                "shared-surface forward references are unavailable"
            )
        first_parameter = next(iter(model.parameters()))
        device = first_parameter.device
        input_ids = input_ids.to(device=device, dtype=torch.long)
        attention_mask = attention_mask.to(device=device, dtype=torch.long)
        pixels, grid = _materialized_image(skeleton)
        batch = int(input_ids.shape[0])
        pixel_values = pixels.repeat((batch, 1)).to(
            device=device, dtype=first_parameter.dtype
        )
        image_grid_thw = torch.tensor(
            [grid] * batch, dtype=torch.long, device=device
        )
        position_ids = _derive_qwen_position_ids(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
        )
        if retain_grad:
            self._replay_forward_count += 1
        else:
            self._sample_forward_count += 1
        self._no_cache_forward_count += 1
        context = torch.enable_grad() if retain_grad else torch.inference_mode()
        with context:
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                use_cache=False,
                return_dict=True,
                logits_to_keep=0,
            )
        if getattr(output, "past_key_values", None) is not None:
            raise HFSharedSurfaceLiveError(
                "shared-surface forward returned forbidden cache state"
            )
        logits = getattr(output, "logits", None)
        if (
            not isinstance(logits, torch.Tensor)
            or logits.ndim != 3
            or tuple(logits.shape[:2]) != tuple(input_ids.shape)
            or int(logits.shape[2]) != len(tokenizer)
        ):
            raise HFSharedSurfaceLiveError(
                "shared-surface causal logits have the wrong shape"
            )
        if logits.dtype != torch.float32:
            raise HFSharedSurfaceLiveError(
                "prepared BF16 shared-surface logits must be converted to float32"
            )
        if not bool(torch.isfinite(logits).all().item()):
            raise HFSharedSurfaceLiveError(
                "shared-surface causal logits must be finite"
            )
        self._require_invariants(verify_parameter_values=False)
        return logits

    def sample_group(self, seeds: tuple[int, ...]) -> SampledHFGroup:
        try:
            self._require_open()
            self._require_invariants()
            if seeds not in self._plan.seed_groups:
                raise HFSharedSurfaceLiveError(
                    "sample group seeds differ from the frozen image-1584 K16 plan"
                )
            group_index = self._plan.seed_groups.index(seeds)
            if group_index != len(self._sampled_groups):
                raise HFSharedSurfaceLiveError(
                    "sample groups must follow the frozen K16 order exactly"
                )
            model = self._model
            tokenizer = self._tokenizer
            if model is None or tokenizer is None:
                raise HFSharedSurfaceLiveError(
                    "shared-surface sampling references are unavailable"
                )
            device = next(iter(model.parameters())).device
            generators = tuple(
                torch.Generator(device=device).manual_seed(seed) for seed in seeds
            )
            request_ids = tuple(f"image-1584:seed-{seed}" for seed in seeds)
            generated: list[list[int]] = [[] for _ in seeds]
            token_rows: list[list[SampledHFToken]] = [[] for _ in seeds]
            stopped = [False for _ in seeds]
            steps: list[HFActiveBatchStep] = []
            stop_token_id = tokenizer.convert_tokens_to_ids(
                self._plan.policy.stop_token
            )
            if (
                isinstance(stop_token_id, bool)
                or not isinstance(stop_token_id, int)
                or stop_token_id < 0
                or stop_token_id >= len(tokenizer)
            ):
                raise HFSharedSurfaceLiveError(
                    "shared-surface tokenizer lacks the exact im_end stop token"
                )
            for token_index in range(self._plan.policy.max_new_tokens):
                active = tuple(index for index, value in enumerate(stopped) if not value)
                if not active:
                    break
                histories = tuple(
                    self._prompt + tuple(generated[index]) for index in active
                )
                if len({len(history) for history in histories}) != 1:
                    raise HFSharedSurfaceLiveError(
                        "active full-history batch lost equal causal lengths"
                    )
                input_ids = torch.tensor(histories, dtype=torch.long, device=device)
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)
                rng_before = _rng_sha256(generators)
                logits = self._forward(
                    input_ids, attention_mask, retain_grad=False
                )[:, -1, :]
                processed = _apply_policy(input_ids, logits, self._plan)
                logps = torch.log_softmax(processed, dim=-1)
                chosen: list[int] = []
                for row, request_index in enumerate(active):
                    probability = torch.softmax(processed[row], dim=-1)
                    token_id = int(
                        torch.multinomial(
                            probability,
                            num_samples=1,
                            generator=generators[request_index],
                        ).item()
                    )
                    chosen.append(token_id)
                    history_sha256 = causal_history_sha256(
                        self._identity.prompt_sha256,
                        tuple(generated[request_index]),
                    )
                    token_rows[request_index].append(
                        SampledHFToken(
                            request_id=request_ids[request_index],
                            token_index=token_index,
                            history_sha256=history_sha256,
                            chosen_token_id=token_id,
                            raw_chosen_logit=float(
                                logits[row, token_id].detach().float().item()
                            ),
                            processed_logp=float(
                                logps[row, token_id].detach().float().item()
                            ),
                            causal_logit_index=len(histories[row]) - 1,
                        )
                    )
                rng_after = _rng_sha256(generators)
                steps.append(
                    HFActiveBatchStep(
                        token_index=token_index,
                        active_request_ids=tuple(request_ids[index] for index in active),
                        active_history_sha256s=tuple(
                            token_rows[index][-1].history_sha256 for index in active
                        ),
                        batch_shape=(
                            int(input_ids.shape[0]),
                            int(input_ids.shape[1]),
                        ),
                        rng_before_sha256=rng_before,
                        rng_after_sha256=rng_after,
                    )
                )
                for request_index, token_id in zip(active, chosen, strict=True):
                    generated[request_index].append(token_id)
                    if token_id == stop_token_id:
                        stopped[request_index] = True
            if not all(stopped):
                if any(
                    not stopped[index]
                    and len(tokens) != self._plan.policy.max_new_tokens
                    for index, tokens in enumerate(token_rows)
                ):
                    raise HFSharedSurfaceLiveError(
                        "shared-surface sampler ended before im_end or the token cap"
                    )
            requests = tuple(
                SampledHFRequest(
                    request_id=request_ids[index],
                    image_id=self._plan.image_id,
                    seed=seed,
                    prompt_history_sha256=self._identity.prompt_sha256,
                    tokens=tuple(token_rows[index]),
                    processor_order=_PROCESSOR_ORDER,
                    stop_reason="im_end" if stopped[index] else "cap",
                    use_cache=False,
                )
                for index, seed in enumerate(seeds)
            )
            group = admit_sampled_group(
                plan=self._plan,
                group_index=group_index,
                expected_identity=self._identity,
                identity=self._identity,
                policy=self._plan.policy,
                requests=requests,
                active_batch_steps=tuple(steps),
            )
            self._require_invariants()
            self._sampled_groups.append(
                (cast(tuple[int, int, int, int], seeds), group.content_sha256)
            )
            return group
        except Exception as exc:
            if not self._closed:
                self._close_internal("failed", primary_exception=exc)
            raise

    def replay_group(self, group: SampledHFGroup) -> GradientReplayGroup:
        try:
            self._require_open()
            self._require_invariants()
            if type(group) is not SampledHFGroup or group.identity != self._identity:
                raise HFSharedSurfaceLiveError(
                    "replay group differs from the shared surface identity"
                )
            if group.plan != self._plan or group.policy != self._plan.policy:
                raise HFSharedSurfaceLiveError(
                    "replay group differs from the frozen shared-surface plan"
                )
            if group.group_index >= len(self._sampled_groups):
                raise HFSharedSurfaceLiveError(
                    "replay group was not sampled by this live session"
                )
            expected_replay_index = len(self._replayed_groups)
            if (
                group.group_index != expected_replay_index
                or self._sampled_groups[expected_replay_index][1]
                != group.content_sha256
            ):
                raise HFSharedSurfaceLiveError(
                    "replay groups must follow sampled K16 order exactly"
                )
            histories = tuple(
                self._prompt
                + tuple(token.chosen_token_id for token in request.tokens)
                for request in group.requests
            )
            maximum = max(len(history) for history in histories)
            pad_token_id = getattr(self._tokenizer, "pad_token_id", None)
            if (
                isinstance(pad_token_id, bool)
                or not isinstance(pad_token_id, int)
                or pad_token_id < 0
            ):
                raise HFSharedSurfaceLiveError(
                    "shared-surface tokenizer lacks a padding token"
                )
            input_ids = torch.full(
                (len(histories), maximum), pad_token_id, dtype=torch.long
            )
            attention_mask = torch.zeros_like(input_ids)
            for row, history in enumerate(histories):
                input_ids[row, : len(history)] = torch.tensor(history, dtype=torch.long)
                attention_mask[row, : len(history)] = 1
            logits = self._forward(input_ids, attention_mask, retain_grad=True)
            row_indexes: list[int] = []
            positions: list[int] = []
            chosen_ids: list[int] = []
            processor_histories: list[tuple[int, ...]] = []
            sampled_tokens: list[SampledHFToken] = []
            for row, request in enumerate(group.requests):
                for token in request.tokens:
                    expected_position = len(self._prompt) - 1 + token.token_index
                    if token.causal_logit_index != expected_position:
                        raise HFSharedSurfaceLiveError(
                            "sampled token has the wrong causal position"
                        )
                    row_indexes.append(row)
                    positions.append(expected_position)
                    chosen_ids.append(token.chosen_token_id)
                    processor_histories.append(
                        self._prompt
                        + tuple(
                            item.chosen_token_id
                            for item in request.tokens[: token.token_index]
                        )
                    )
                    sampled_tokens.append(token)
            device = logits.device
            row_tensor = torch.tensor(row_indexes, dtype=torch.long, device=device)
            position_tensor = torch.tensor(positions, dtype=torch.long, device=device)
            causal_logits = logits[row_tensor, position_tensor]
            history_width = max(len(history) for history in processor_histories)
            processor_input_ids = torch.full(
                (len(processor_histories), history_width),
                pad_token_id,
                dtype=torch.long,
                device=device,
            )
            for row, history in enumerate(processor_histories):
                processor_input_ids[row, : len(history)] = torch.tensor(
                    history, dtype=torch.long, device=device
                )
            processed = _apply_policy(processor_input_ids, causal_logits, self._plan)
            logps = torch.log_softmax(processed, dim=-1)
            chosen_tensor = torch.tensor(chosen_ids, dtype=torch.long, device=device)
            chosen_raw = causal_logits.gather(1, chosen_tensor[:, None]).squeeze(1)
            chosen_logps = logps.gather(1, chosen_tensor[:, None]).squeeze(1)
            if not bool(torch.isfinite(chosen_logps).all().item()):
                raise HFSharedSurfaceLiveError(
                    "replayed chosen log probabilities must be finite"
                )
            replayed = tuple(
                SampledHFToken(
                    request_id=sampled.request_id,
                    token_index=sampled.token_index,
                    history_sha256=sampled.history_sha256,
                    chosen_token_id=sampled.chosen_token_id,
                    raw_chosen_logit=float(chosen_raw[index].detach().float().item()),
                    processed_logp=float(chosen_logps[index].detach().float().item()),
                    causal_logit_index=sampled.causal_logit_index,
                )
                for index, sampled in enumerate(sampled_tokens)
            )
            gathers = tuple(
                HFReplayCausalGather(
                    request_id=sampled.request_id,
                    token_index=sampled.token_index,
                    history_sha256=sampled.history_sha256,
                    chosen_token_id=sampled.chosen_token_id,
                    causal_logit_index=sampled.causal_logit_index,
                )
                for sampled in sampled_tokens
            )
            replay = admit_gradient_replay(
                sampled_group=group,
                replay_identity=self._identity,
                replayed_tokens=replayed,
                replay_processor_order=_PROCESSOR_ORDER,
                causal_gathers=gathers,
            )
            self._require_invariants()
            self._live_replay_tensors[replay.content_sha256] = chosen_logps
            self._latest_replay_group_sha256 = replay.content_sha256
            self._replayed_groups.append(
                (group.content_sha256, replay.content_sha256)
            )
            return replay
        except Exception as exc:
            if not self._closed:
                self._close_internal("failed", primary_exception=exc)
            raise

    def _close_internal(
        self,
        reason: Literal["completed", "failed"],
        *,
        primary_exception: BaseException | None = None,
    ) -> SharedSurfaceResourceReceipt:
        if self._closed:
            raise HFSharedSurfaceLiveError("shared-surface session is already closed")
        self._cleanup_call_count += 1
        self._cleanup_reason = reason
        model = self._expected_model
        accelerator = getattr(self._assembly, "accelerator", None)
        cleanup_failures: list[str] = []
        try:
            zero_grad = getattr(model, "zero_grad", None)
            if callable(zero_grad):
                try:
                    zero_grad(set_to_none=True)
                except Exception as exc:  # cleanup must continue
                    cleanup_failures.append(f"zero_grad: {type(exc).__name__}: {exc}")
            free_memory = getattr(accelerator, "free_memory", None)
            if callable(free_memory):
                try:
                    free_memory()
                except Exception as exc:  # cleanup must continue
                    cleanup_failures.append(f"free_memory: {type(exc).__name__}: {exc}")
        finally:
            self._live_replay_tensors.clear()
            self._closed = True
            self._model = None
            self._expected_model = None
            self._assembly = None
            self._components = None
            self._skeleton = None
            self._tokenizer = None
            self._processor = None
            self._parameters = ()
            self._parameter_layout = None
            self._prompt = ()
            terminal_reason: Literal["completed", "failed"] = (
                "failed" if cleanup_failures or reason == "failed" else "completed"
            )
            self._terminal_receipt = self._build_terminal_receipt(
                reason=terminal_reason,
                cleanup_failures=tuple(cleanup_failures),
            )
        if cleanup_failures and primary_exception is None:
            raise HFSharedSurfaceLiveError(
                "shared-surface cleanup failed: " + "; ".join(cleanup_failures)
            )
        return self.resource_receipt

    def close(self) -> SharedSurfaceResourceReceipt:
        self._require_open()
        complete = (
            tuple(seeds for seeds, _sha256 in self._sampled_groups)
            == self._plan.seed_groups
            and len(self._sampled_groups) == 4
            and len(self._replayed_groups) == 4
        )
        if not complete:
            error = HFSharedSurfaceLiveError(
                "completed close requires four sampled and replayed groups"
            )
            self._close_internal("failed", primary_exception=error)
            raise error
        return self._close_internal("completed")


def open_hf_shared_surface(
    plan: HFSharedSurfacePlan,
    assembly: Human13LiveAssembly,
    skeleton: Any,
) -> HFSharedSurfaceSession:
    """Admit an already loaded assembly; never load or launch one here."""

    try:
        return HFSharedSurfaceSession(plan, assembly, skeleton)
    except Exception as primary:
        model = getattr(assembly, "model", None)
        zero_grad = getattr(model, "zero_grad", None)
        cleanup_failures: list[str] = []
        if callable(zero_grad):
            try:
                zero_grad(set_to_none=True)
            except Exception as exc:
                cleanup_failures.append(f"zero_grad: {type(exc).__name__}: {exc}")
        accelerator = getattr(assembly, "accelerator", None)
        free_memory = getattr(accelerator, "free_memory", None)
        if callable(free_memory):
            try:
                free_memory()
            except Exception as exc:
                cleanup_failures.append(f"free_memory: {type(exc).__name__}: {exc}")
        for failure in cleanup_failures:
            primary.add_note(f"shared-surface open cleanup failure: {failure}")
        raise


__all__ = [
    "HFSharedSurfaceLiveError",
    "HFSharedSurfaceSession",
    "SharedSurfaceResourceReceipt",
    "open_hf_shared_surface",
]
