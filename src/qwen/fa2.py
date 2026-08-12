"""FlashAttention 2 varlen segment-isolation contracts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import hashlib
import inspect
import json
import threading
from typing import Any

import torch

from src.common.errors import QwenForwardContractError
from src.packing.planner import PackedSequence


PADDING_FREE_VARLEN_BRANCH = "padding_free_varlen"
_ACCEPTED_FA2_BACKEND = "flash_attention_2"
_ACCEPTED_MODEL_DTYPES = frozenset(
    {
        "torch.bfloat16",
        "torch.float16",
        "bfloat16",
        "float16",
        "bf16",
        "fp16",
    }
)
_CAPTURE_LOCK = threading.Lock()


class AttentionEventKind(str, Enum):
    TEXT = "text"
    VISION = "vision"
    UNRELATED = "unrelated"


@dataclass(frozen=True)
class AttentionCallableIdentity:
    module: str
    qualname: str
    source_file: str | None
    source_first_line: int | None
    content_sha256: str

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "qualname": self.qualname,
            "source_file": self.source_file,
            "source_first_line": self.source_first_line,
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True)
class TensorExecutionIdentity:
    shape: tuple[int, ...]
    dtype: str
    device: str

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "device": self.device,
        }


@dataclass(frozen=True)
class QwenTextLayerIdentity:
    topology_id: str
    module_name: str
    module_class: str
    layer_idx: int
    configured_backend: str

    @property
    def identity(self) -> str:
        return f"{self.module_name}#{self.layer_idx}"

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "topology_id": self.topology_id,
            "identity": self.identity,
            "module_name": self.module_name,
            "module_class": self.module_class,
            "layer_idx": self.layer_idx,
            "configured_backend": self.configured_backend,
        }


@dataclass(frozen=True)
class QwenAttentionTopology:
    topology_id: str
    text_model_name: str
    text_model_class: str
    configured_text_layer_count: int
    text_layers: tuple[QwenTextLayerIdentity, ...]

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "topology_id": self.topology_id,
            "text_model_name": self.text_model_name,
            "text_model_class": self.text_model_class,
            "configured_text_layer_count": self.configured_text_layer_count,
            "text_layers": [layer.to_artifact_dict() for layer in self.text_layers],
        }


@dataclass(frozen=True)
class AttentionProofEvent:
    kind: AttentionEventKind
    topology_id: str
    module_name: str
    module_class: str
    layer_idx: int | None
    configured_backend: str | None
    registry_key: str
    registry_had_local_override: bool
    resolved_callable: AttentionCallableIdentity
    attention_mask: Any
    cu_seq_lens_q: tuple[int, ...] | None
    cu_seq_lens_k: tuple[int, ...] | None
    max_length_q: int | None
    max_length_k: int | None
    query: TensorExecutionIdentity | None
    key: TensorExecutionIdentity | None
    value: TensorExecutionIdentity | None
    completion_status: str
    lazy_import_implementations: tuple[str | None, ...]
    flash_fn_call_count: int
    flash_varlen_fn_call_count: int
    pad_fn_call_count: int
    unpad_fn_call_count: int
    varlen_calls: tuple[Mapping[str, Any], ...]

    @property
    def identity(self) -> str:
        layer = "none" if self.layer_idx is None else str(self.layer_idx)
        return f"{self.module_name}#{layer}"

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "identity": self.identity,
            "topology_id": self.topology_id,
            "module_name": self.module_name,
            "module_class": self.module_class,
            "layer_idx": self.layer_idx,
            "configured_backend": self.configured_backend,
            "registry_key": self.registry_key,
            "registry_had_local_override": self.registry_had_local_override,
            "resolved_callable": self.resolved_callable.to_artifact_dict(),
            "attention_mask": _artifact_value(self.attention_mask),
            "cu_seq_lens_q": (
                None if self.cu_seq_lens_q is None else list(self.cu_seq_lens_q)
            ),
            "cu_seq_lens_k": (
                None if self.cu_seq_lens_k is None else list(self.cu_seq_lens_k)
            ),
            "max_length_q": self.max_length_q,
            "max_length_k": self.max_length_k,
            "query": None if self.query is None else self.query.to_artifact_dict(),
            "key": None if self.key is None else self.key.to_artifact_dict(),
            "value": None if self.value is None else self.value.to_artifact_dict(),
            "completion_status": self.completion_status,
            "lazy_import_implementations": list(self.lazy_import_implementations),
            "flash_fn_call_count": self.flash_fn_call_count,
            "flash_varlen_fn_call_count": self.flash_varlen_fn_call_count,
            "pad_fn_call_count": self.pad_fn_call_count,
            "unpad_fn_call_count": self.unpad_fn_call_count,
            "varlen_calls": [_artifact_value(call) for call in self.varlen_calls],
        }


@dataclass(frozen=True)
class Fa2AttentionProofEvidence:
    topology: QwenAttentionTopology
    events: tuple[AttentionProofEvent, ...]
    registry_keys: tuple[str, ...]
    instrumentation_restored: bool
    orphan_flash_fn_call_count: int
    orphan_flash_varlen_fn_call_count: int
    orphan_pad_fn_call_count: int
    orphan_unpad_fn_call_count: int

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "topology": self.topology.to_artifact_dict(),
            "events": [event.to_artifact_dict() for event in self.events],
            "registry_keys": list(self.registry_keys),
            "instrumentation_restored": self.instrumentation_restored,
            "orphan_low_level_calls": {
                "flash": self.orphan_flash_fn_call_count,
                "flash_varlen": self.orphan_flash_varlen_fn_call_count,
                "pad": self.orphan_pad_fn_call_count,
                "unpad": self.orphan_unpad_fn_call_count,
            },
        }


@dataclass(frozen=True)
class Fa2VarlenPlan:
    segment_boundaries: tuple[int, ...]
    segment_lengths: tuple[int, ...]
    cu_seq_lens_q: torch.Tensor
    cu_seq_lens_k: torch.Tensor
    max_length_q: int
    max_length_k: int
    attention_mask: None
    branch_evidence_required: bool = True

    @property
    def segment_count(self) -> int:
        return len(self.segment_lengths)

    def to_model_kwargs(self) -> dict[str, Any]:
        return {
            "attention_mask": None,
            "cu_seq_lens_q": self.cu_seq_lens_q,
            "cu_seq_lens_k": self.cu_seq_lens_k,
            "max_length_q": self.max_length_q,
            "max_length_k": self.max_length_k,
        }

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "segment_count": self.segment_count,
            "segment_boundaries": list(self.segment_boundaries),
            "segment_lengths": list(self.segment_lengths),
            "cumulative_sequence_lengths": list(self.segment_boundaries),
            "cu_seq_lens_q": _tensor_int_list(self.cu_seq_lens_q),
            "cu_seq_lens_k": _tensor_int_list(self.cu_seq_lens_k),
            "max_length_q": self.max_length_q,
            "max_length_k": self.max_length_k,
            "attention_mask": None,
            "branch_evidence_required": self.branch_evidence_required,
        }


@dataclass(frozen=True)
class Fa2VarlenBranchProof:
    observed_branch: str
    segment_boundaries: tuple[int, ...]
    cu_seq_lens_q: tuple[int, ...]
    cu_seq_lens_k: tuple[int, ...]
    max_length_q: int
    max_length_k: int
    resolved_attention_implementation: str
    model_dtype: str
    branch_evidence_from_explicit_varlen_kwargs: bool
    flash_fn_called: bool
    flash_varlen_fn_called: bool
    pad_fn_called: bool
    unpad_fn_called: bool
    observed_call: Mapping[str, Any]
    topology: QwenAttentionTopology
    text_layer_events: tuple[AttentionProofEvent, ...]
    vision_attention_events: tuple[AttentionProofEvent, ...]
    unrelated_attention_events: tuple[AttentionProofEvent, ...]

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "status": "pass",
            "observed_branch": self.observed_branch,
            "segment_boundaries": list(self.segment_boundaries),
            "cu_seq_lens_q": list(self.cu_seq_lens_q),
            "cu_seq_lens_k": list(self.cu_seq_lens_k),
            "max_length_q": self.max_length_q,
            "max_length_k": self.max_length_k,
            "resolved_attention_implementation": self.resolved_attention_implementation,
            "model_dtype": self.model_dtype,
            "branch_evidence_from_explicit_varlen_kwargs": (
                self.branch_evidence_from_explicit_varlen_kwargs
            ),
            "flash_fn_called": self.flash_fn_called,
            "flash_varlen_fn_called": self.flash_varlen_fn_called,
            "pad_fn_called": self.pad_fn_called,
            "unpad_fn_called": self.unpad_fn_called,
            "observed_call": _artifact_value(self.observed_call),
            "topology": self.topology.to_artifact_dict(),
            "expected_text_layer_count": len(self.topology.text_layers),
            "observed_text_layer_count": len(self.text_layer_events),
            "text_layer_events": [
                event.to_artifact_dict() for event in self.text_layer_events
            ],
            "vision_attention_events": [
                event.to_artifact_dict() for event in self.vision_attention_events
            ],
            "unrelated_attention_events": [
                event.to_artifact_dict() for event in self.unrelated_attention_events
            ],
        }


@dataclass
class Fa2VarlenBranchCapture:
    topology: QwenAttentionTopology
    _expected_text_by_object_id: Mapping[int, QwenTextLayerIdentity]
    _module_names_by_object_id: Mapping[int, str]
    _qwen_text_attention_class: type[Any]
    _qwen_vision_attention_class: type[Any]
    _events: list[AttentionProofEvent]
    _active_event_stack: list[dict[str, Any]]
    _registry_keys: tuple[str, ...]
    _orphan_low_level_counts: dict[str, int]
    _instrumentation_restored: bool = False

    def evidence_for_plan(self, plan: Fa2VarlenPlan) -> Fa2AttentionProofEvidence:
        if not isinstance(plan, Fa2VarlenPlan):
            raise QwenForwardContractError(
                "FA2 capture evidence requires a Fa2VarlenPlan",
                code="qwen.fa2_plan_type",
                context={"value_type": type(plan).__name__},
            )
        if not self._instrumentation_restored:
            raise QwenForwardContractError(
                "FA2 capture evidence is valid only after instrumentation restoration",
                code="qwen.fa2_capture_unrestored",
                context={"topology_id": self.topology.topology_id},
            )
        return Fa2AttentionProofEvidence(
            topology=self.topology,
            events=tuple(self._events),
            registry_keys=self._registry_keys,
            instrumentation_restored=True,
            orphan_flash_fn_call_count=self._orphan_low_level_counts["flash"],
            orphan_flash_varlen_fn_call_count=self._orphan_low_level_counts[
                "flash_varlen"
            ],
            orphan_pad_fn_call_count=self._orphan_low_level_counts["pad"],
            orphan_unpad_fn_call_count=self._orphan_low_level_counts["unpad"],
        )

    def _event_classification(
        self,
        module: Any,
    ) -> tuple[AttentionEventKind, str, str, int | None, str | None]:
        expected = self._expected_text_by_object_id.get(id(module))
        if expected is not None:
            return (
                AttentionEventKind.TEXT,
                expected.module_name,
                expected.module_class,
                expected.layer_idx,
                expected.configured_backend,
            )
        module_name = self._module_names_by_object_id.get(
            id(module), f"<unresolved:{type(module).__qualname__}>"
        )
        module_class = _qualified_class_name(module)
        configured_backend = _configured_backend(module)
        layer_idx = _optional_int(getattr(module, "layer_idx", None))
        if isinstance(module, self._qwen_text_attention_class):
            return (
                AttentionEventKind.TEXT,
                module_name,
                module_class,
                layer_idx,
                configured_backend,
            )
        if isinstance(module, self._qwen_vision_attention_class):
            return (
                AttentionEventKind.VISION,
                module_name,
                module_class,
                layer_idx,
                configured_backend,
            )
        return (
            AttentionEventKind.UNRELATED,
            module_name,
            module_class,
            layer_idx,
            configured_backend,
        )

    def _record_low_level_call(
        self,
        call_kind: str,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> None:
        if not self._active_event_stack:
            self._orphan_low_level_counts[call_kind] += 1
            return
        event = self._active_event_stack[-1]
        event[f"{call_kind}_fn_call_count"] += 1
        if call_kind == "flash_varlen":
            event["varlen_calls"].append(_varlen_flash_call_artifact(args, kwargs))


@contextmanager
def capture_fa2_varlen_branch(model: Any) -> Any:
    if not _CAPTURE_LOCK.acquire(blocking=False):
        raise QwenForwardContractError(
            "nested or concurrent FA2 branch capture is not supported",
            code="qwen.fa2_capture_nested",
            context={},
        )
    try:
        try:
            import transformers.modeling_flash_attention_utils as flash_utils
            from transformers import modeling_utils
        except ImportError as exc:
            raise QwenForwardContractError(
                "FA2 branch capture requires transformers attention utilities",
                code="qwen.fa2_capture_unavailable",
                cause=exc,
            ) from exc

        resolved_topology = _resolve_qwen_attention_topology(model)
        registry = modeling_utils.ALL_ATTENTION_FUNCTIONS
        local_mapping = getattr(registry, "_local_mapping", None)
        if not isinstance(local_mapping, dict):
            raise QwenForwardContractError(
                "Transformers attention registry does not expose restorable local overrides",
                code="qwen.fa2_registry_unavailable",
                context={"registry_type": type(registry).__name__},
            )
        original_local_mapping = dict(local_mapping)
        registry_keys = tuple(sorted(str(key) for key in registry.keys()))
        capture = Fa2VarlenBranchCapture(
            topology=resolved_topology.topology,
            _expected_text_by_object_id=resolved_topology.expected_text_by_object_id,
            _module_names_by_object_id=resolved_topology.module_names_by_object_id,
            _qwen_text_attention_class=resolved_topology.qwen_text_attention_class,
            _qwen_vision_attention_class=resolved_topology.qwen_vision_attention_class,
            _events=[],
            _active_event_stack=[],
            _registry_keys=registry_keys,
            _orphan_low_level_counts={
                "flash": 0,
                "flash_varlen": 0,
                "pad": 0,
                "unpad": 0,
            },
        )
        original_lazy_import = flash_utils.lazy_import_flash_attention

        def capturing_lazy_import_flash_attention(
            implementation: str | None = None,
            *args: Any,
            **kwargs: Any,
        ) -> tuple[Any, Any]:
            if capture._active_event_stack:
                capture._active_event_stack[-1]["lazy_import_implementations"].append(
                    implementation
                )
            (flash_fn, flash_varlen_fn, pad_fn, unpad_fn), process_flash_kwargs_fn = (
                original_lazy_import(implementation, *args, **kwargs)
            )

            def wrapped_flash_fn(*flash_args: Any, **flash_kwargs: Any) -> Any:
                capture._record_low_level_call("flash", flash_args, flash_kwargs)
                return flash_fn(*flash_args, **flash_kwargs)

            def wrapped_flash_varlen_fn(*flash_args: Any, **flash_kwargs: Any) -> Any:
                capture._record_low_level_call("flash_varlen", flash_args, flash_kwargs)
                return flash_varlen_fn(*flash_args, **flash_kwargs)

            def wrapped_pad_fn(*pad_args: Any, **pad_kwargs: Any) -> Any:
                capture._record_low_level_call("pad", pad_args, pad_kwargs)
                return pad_fn(*pad_args, **pad_kwargs)

            def wrapped_unpad_fn(*unpad_args: Any, **unpad_kwargs: Any) -> Any:
                capture._record_low_level_call("unpad", unpad_args, unpad_kwargs)
                return unpad_fn(*unpad_args, **unpad_kwargs)

            return (
                (
                    wrapped_flash_fn,
                    wrapped_flash_varlen_fn,
                    wrapped_pad_fn,
                    wrapped_unpad_fn,
                ),
                process_flash_kwargs_fn,
            )

        try:
            flash_utils.lazy_import_flash_attention = (
                capturing_lazy_import_flash_attention
            )
            for registry_key in registry_keys:
                resolved_callable = registry[registry_key]
                registry[registry_key] = _registry_attention_wrapper(
                    capture,
                    registry_key=registry_key,
                    registry_had_local_override=registry_key in original_local_mapping,
                    resolved_callable=resolved_callable,
                )
            yield capture
        finally:
            try:
                flash_utils.lazy_import_flash_attention = original_lazy_import
            finally:
                local_mapping.clear()
                local_mapping.update(original_local_mapping)
            if not _mapping_identity_equal(local_mapping, original_local_mapping):
                raise QwenForwardContractError(
                    "FA2 capture failed to restore attention registry overrides",
                    code="qwen.fa2_capture_restore",
                    context={"registry_keys": list(registry_keys)},
                )
            capture._instrumentation_restored = True
    finally:
        _CAPTURE_LOCK.release()


@dataclass(frozen=True)
class _ResolvedAttentionTopology:
    topology: QwenAttentionTopology
    expected_text_by_object_id: Mapping[int, QwenTextLayerIdentity]
    module_names_by_object_id: Mapping[int, str]
    qwen_text_attention_class: type[Any]
    qwen_vision_attention_class: type[Any]


def _resolve_qwen_attention_topology(model: Any) -> _ResolvedAttentionTopology:
    try:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLTextAttention,
            Qwen3VLTextModel,
            Qwen3VLVisionAttention,
        )
    except ImportError as exc:
        raise QwenForwardContractError(
            "Qwen topology proof requires installed Qwen3-VL model classes",
            code="qwen.fa2_topology_unavailable",
            cause=exc,
        ) from exc

    named_modules_fn = getattr(model, "named_modules", None)
    if not callable(named_modules_fn):
        raise QwenForwardContractError(
            "FA2 proof requires a loaded module topology",
            code="qwen.fa2_topology",
            context={"model_type": type(model).__name__},
        )
    named_modules = tuple(named_modules_fn())
    module_names_by_object_id = {
        id(module): str(name) for name, module in named_modules
    }
    text_models = [
        (str(name), module)
        for name, module in named_modules
        if isinstance(module, Qwen3VLTextModel)
    ]
    if len(text_models) != 1:
        raise QwenForwardContractError(
            "FA2 proof requires exactly one Qwen3VLTextModel in the loaded topology",
            code="qwen.fa2_topology",
            context={
                "observed_text_model_count": len(text_models),
                "observed_text_model_names": [name for name, _ in text_models[:16]],
            },
        )
    text_model_name, text_model = text_models[0]
    layers = tuple(getattr(text_model, "layers", ()))
    configured_count = _optional_int(
        getattr(getattr(text_model, "config", None), "num_hidden_layers", None)
    )
    if (
        configured_count is None
        or configured_count <= 0
        or configured_count != len(layers)
    ):
        raise QwenForwardContractError(
            "Qwen text topology must match config.num_hidden_layers",
            code="qwen.fa2_topology",
            context={
                "configured_text_layer_count": configured_count,
                "observed_text_layer_count": len(layers),
                "text_model_name": text_model_name,
            },
        )

    unresolved_layers: list[dict[str, Any]] = []
    raw_layers: list[dict[str, Any]] = []
    for expected_idx, layer in enumerate(layers):
        attention = getattr(layer, "self_attn", None)
        module_name = module_names_by_object_id.get(id(attention))
        layer_idx = _optional_int(getattr(attention, "layer_idx", None))
        if (
            not isinstance(attention, Qwen3VLTextAttention)
            or module_name is None
            or layer_idx != expected_idx
        ):
            unresolved_layers.append(
                {
                    "expected_layer_idx": expected_idx,
                    "observed_layer_idx": layer_idx,
                    "module_name": module_name,
                    "module_class": _qualified_class_name(attention),
                }
            )
            continue
        raw_layers.append(
            {
                "module_name": module_name,
                "module_class": _qualified_class_name(attention),
                "layer_idx": expected_idx,
                "configured_backend": _configured_backend(attention),
                "module": attention,
            }
        )
    if unresolved_layers:
        raise QwenForwardContractError(
            "Qwen text topology has missing, non-Qwen, or non-contiguous self-attention layers",
            code="qwen.fa2_topology",
            context={"offending_layers": unresolved_layers[:32]},
        )

    topology_payload = {
        "text_model_class": _qualified_class_name(text_model),
        "configured_text_layer_count": configured_count,
        "text_layers": [
            {
                "module_name": layer["module_name"],
                "module_class": layer["module_class"],
                "layer_idx": layer["layer_idx"],
                "configured_backend": layer["configured_backend"],
            }
            for layer in raw_layers
        ],
    }
    topology_id = hashlib.sha256(
        json.dumps(
            topology_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    text_layer_identities = tuple(
        QwenTextLayerIdentity(
            topology_id=topology_id,
            module_name=str(layer["module_name"]),
            module_class=str(layer["module_class"]),
            layer_idx=int(layer["layer_idx"]),
            configured_backend=str(layer["configured_backend"]),
        )
        for layer in raw_layers
    )
    topology = QwenAttentionTopology(
        topology_id=topology_id,
        text_model_name=text_model_name,
        text_model_class=_qualified_class_name(text_model),
        configured_text_layer_count=configured_count,
        text_layers=text_layer_identities,
    )
    return _ResolvedAttentionTopology(
        topology=topology,
        expected_text_by_object_id={
            id(layer["module"]): identity
            for layer, identity in zip(raw_layers, text_layer_identities, strict=True)
        },
        module_names_by_object_id=module_names_by_object_id,
        qwen_text_attention_class=Qwen3VLTextAttention,
        qwen_vision_attention_class=Qwen3VLVisionAttention,
    )


def _registry_attention_wrapper(
    capture: Fa2VarlenBranchCapture,
    *,
    registry_key: str,
    registry_had_local_override: bool,
    resolved_callable: Callable[..., Any],
) -> Callable[..., Any]:
    callable_identity = _callable_identity(resolved_callable)

    def wrapped_attention(*args: Any, **kwargs: Any) -> Any:
        module = args[0] if args else kwargs.get("module")
        query = args[1] if len(args) > 1 else kwargs.get("query")
        key = args[2] if len(args) > 2 else kwargs.get("key")
        value = args[3] if len(args) > 3 else kwargs.get("value")
        attention_mask = args[4] if len(args) > 4 else kwargs.get("attention_mask")
        kind, module_name, module_class, layer_idx, configured_backend = (
            capture._event_classification(module)
        )
        event: dict[str, Any] = {
            "kind": kind,
            "topology_id": capture.topology.topology_id,
            "module_name": module_name,
            "module_class": module_class,
            "layer_idx": layer_idx,
            "configured_backend": configured_backend,
            "registry_key": registry_key,
            "registry_had_local_override": registry_had_local_override,
            "resolved_callable": callable_identity,
            "attention_mask": (
                None if attention_mask is None else _shape_dtype_device(attention_mask)
            ),
            "cu_seq_lens_q": _tensor_int_tuple_or_none(kwargs.get("cu_seq_lens_q")),
            "cu_seq_lens_k": _tensor_int_tuple_or_none(kwargs.get("cu_seq_lens_k")),
            "max_length_q": _optional_int(kwargs.get("max_length_q")),
            "max_length_k": _optional_int(kwargs.get("max_length_k")),
            "query": _tensor_execution_identity(query),
            "key": _tensor_execution_identity(key),
            "value": _tensor_execution_identity(value),
            "completion_status": "failed",
            "lazy_import_implementations": [],
            "flash_fn_call_count": 0,
            "flash_varlen_fn_call_count": 0,
            "pad_fn_call_count": 0,
            "unpad_fn_call_count": 0,
            "varlen_calls": [],
        }
        capture._active_event_stack.append(event)
        try:
            result = resolved_callable(*args, **kwargs)
            event["completion_status"] = "completed"
            return result
        finally:
            active = capture._active_event_stack.pop()
            if active is not event:
                raise QwenForwardContractError(
                    "FA2 attention event stack was corrupted",
                    code="qwen.fa2_capture_state",
                    context={"registry_key": registry_key},
                )
            capture._events.append(_freeze_attention_event(event))

    return wrapped_attention


def _freeze_attention_event(event: Mapping[str, Any]) -> AttentionProofEvent:
    return AttentionProofEvent(
        kind=event["kind"],
        topology_id=str(event["topology_id"]),
        module_name=str(event["module_name"]),
        module_class=str(event["module_class"]),
        layer_idx=event["layer_idx"],
        configured_backend=event["configured_backend"],
        registry_key=str(event["registry_key"]),
        registry_had_local_override=bool(event["registry_had_local_override"]),
        resolved_callable=event["resolved_callable"],
        attention_mask=event["attention_mask"],
        cu_seq_lens_q=event["cu_seq_lens_q"],
        cu_seq_lens_k=event["cu_seq_lens_k"],
        max_length_q=event["max_length_q"],
        max_length_k=event["max_length_k"],
        query=event["query"],
        key=event["key"],
        value=event["value"],
        completion_status=str(event["completion_status"]),
        lazy_import_implementations=tuple(event["lazy_import_implementations"]),
        flash_fn_call_count=int(event["flash_fn_call_count"]),
        flash_varlen_fn_call_count=int(event["flash_varlen_fn_call_count"]),
        pad_fn_call_count=int(event["pad_fn_call_count"]),
        unpad_fn_call_count=int(event["unpad_fn_call_count"]),
        varlen_calls=tuple(event["varlen_calls"]),
    )


def build_fa2_varlen_plan(
    pack: PackedSequence,
    *,
    attention_mask: Any = None,
    device: torch.device | str | None = None,
) -> Fa2VarlenPlan:
    if attention_mask is not None:
        raise QwenForwardContractError(
            "packed FA2 segment isolation must not rely on an ordinary attention_mask",
            code="qwen.fa2_attention_mask",
            context={"attention_mask_type": type(attention_mask).__name__},
        )
    if not isinstance(pack, PackedSequence):
        raise QwenForwardContractError(
            "FA2 varlen planning requires a PackedSequence",
            code="qwen.fa2_pack_type",
            context={"value_type": type(pack).__name__},
        )
    boundaries = _segment_boundaries(pack)
    lengths = tuple(
        boundaries[index + 1] - boundaries[index]
        for index in range(len(boundaries) - 1)
    )
    if not lengths:
        raise QwenForwardContractError(
            "FA2 varlen planning requires at least one packed segment",
            code="qwen.fa2_empty_pack",
            context={"pack_index": pack.pack_index},
        )
    torch_device = torch.device("cpu") if device is None else torch.device(device)
    cu_seq_lens = torch.tensor(boundaries, dtype=torch.int32, device=torch_device)
    max_length = max(lengths)
    return Fa2VarlenPlan(
        segment_boundaries=boundaries,
        segment_lengths=lengths,
        cu_seq_lens_q=cu_seq_lens,
        cu_seq_lens_k=cu_seq_lens.clone(),
        max_length_q=max_length,
        max_length_k=max_length,
        attention_mask=None,
    )


def validate_fa2_varlen_plan_matches_pack(
    plan: Fa2VarlenPlan,
    pack: PackedSequence,
) -> None:
    if not isinstance(plan, Fa2VarlenPlan):
        raise QwenForwardContractError(
            "FA2 varlen plan validation requires a Fa2VarlenPlan",
            code="qwen.fa2_plan_type",
            context={"value_type": type(plan).__name__},
        )
    if not isinstance(pack, PackedSequence):
        raise QwenForwardContractError(
            "FA2 varlen plan validation requires a PackedSequence",
            code="qwen.fa2_pack_type",
            context={"value_type": type(pack).__name__},
        )
    expected_boundaries = _segment_boundaries(pack)
    expected_lengths = tuple(
        expected_boundaries[index + 1] - expected_boundaries[index]
        for index in range(len(expected_boundaries) - 1)
    )
    if plan.segment_boundaries != expected_boundaries:
        raise QwenForwardContractError(
            "FA2 varlen plan boundaries must match packed segment boundaries",
            code="qwen.fa2_plan_boundaries",
            context={
                "expected_boundaries": list(expected_boundaries),
                "plan_boundaries": list(plan.segment_boundaries),
            },
        )
    if tuple(_tensor_int_list(plan.cu_seq_lens_q)) != expected_boundaries:
        raise QwenForwardContractError(
            "FA2 plan cu_seq_lens_q must match packed segment boundaries",
            code="qwen.fa2_cu_seq_lens",
            context={
                "expected_boundaries": list(expected_boundaries),
                "cu_seq_lens_q": _tensor_int_list(plan.cu_seq_lens_q),
            },
        )
    if tuple(_tensor_int_list(plan.cu_seq_lens_k)) != expected_boundaries:
        raise QwenForwardContractError(
            "FA2 plan cu_seq_lens_k must match packed segment boundaries",
            code="qwen.fa2_cu_seq_lens",
            context={
                "expected_boundaries": list(expected_boundaries),
                "cu_seq_lens_k": _tensor_int_list(plan.cu_seq_lens_k),
            },
        )
    expected_max_length = max(expected_lengths)
    if (
        plan.max_length_q != expected_max_length
        or plan.max_length_k != expected_max_length
    ):
        raise QwenForwardContractError(
            "FA2 plan max lengths must match packed segment lengths",
            code="qwen.fa2_max_length",
            context={
                "expected_max_length": expected_max_length,
                "max_length_q": plan.max_length_q,
                "max_length_k": plan.max_length_k,
            },
        )
    if plan.attention_mask is not None:
        raise QwenForwardContractError(
            "FA2 plan must not carry an ordinary attention_mask",
            code="qwen.fa2_attention_mask",
            context={"attention_mask": _artifact_value(plan.attention_mask)},
        )


def validate_fa2_varlen_branch_evidence(
    plan: Fa2VarlenPlan,
    evidence: Fa2AttentionProofEvidence,
    *,
    model_dtype: str,
    expected_device: torch.device | str,
) -> Fa2VarlenBranchProof:
    if not isinstance(plan, Fa2VarlenPlan):
        raise QwenForwardContractError(
            "FA2 branch evidence validation requires a Fa2VarlenPlan",
            code="qwen.fa2_plan_type",
            context={"value_type": type(plan).__name__},
        )
    if not isinstance(evidence, Fa2AttentionProofEvidence):
        raise QwenForwardContractError(
            "legacy aggregate FA2 evidence cannot prove every Qwen text layer",
            code="qwen.fa2_legacy_evidence",
            context={"evidence_type": type(evidence).__name__},
        )
    if not evidence.instrumentation_restored:
        raise QwenForwardContractError(
            "FA2 evidence is invalid until all instrumentation is restored",
            code="qwen.fa2_capture_unrestored",
            context={"topology_id": evidence.topology.topology_id},
        )
    if model_dtype not in _ACCEPTED_MODEL_DTYPES:
        raise QwenForwardContractError(
            "FA2 branch proof requires bf16 or fp16 model dtype",
            code="qwen.fa2_dtype",
            context={"model_dtype": model_dtype},
        )

    topology = evidence.topology
    expected_identities = tuple(layer.identity for layer in topology.text_layers)
    if (
        topology.configured_text_layer_count != len(topology.text_layers)
        or tuple(layer.layer_idx for layer in topology.text_layers)
        != tuple(range(topology.configured_text_layer_count))
        or len(set(expected_identities)) != len(expected_identities)
    ):
        raise QwenForwardContractError(
            "FA2 evidence contains an invalid expected Qwen text topology",
            code="qwen.fa2_topology",
            context={
                "configured_text_layer_count": topology.configured_text_layer_count,
                "expected_identities": list(expected_identities[:64]),
            },
        )
    non_fa_expected = [
        layer.identity
        for layer in topology.text_layers
        if layer.configured_backend != _ACCEPTED_FA2_BACKEND
    ]
    if non_fa_expected:
        raise QwenForwardContractError(
            "Qwen text topology is not configured for flash_attention_2",
            code="qwen.fa2_attention_implementation",
            context={"offending_identities": non_fa_expected[:64]},
        )

    text_events = tuple(
        event for event in evidence.events if event.kind is AttentionEventKind.TEXT
    )
    vision_events = tuple(
        event for event in evidence.events if event.kind is AttentionEventKind.VISION
    )
    unrelated_events = tuple(
        event for event in evidence.events if event.kind is AttentionEventKind.UNRELATED
    )
    observed_counts = Counter(event.identity for event in text_events)
    expected_counts = Counter(expected_identities)
    missing = sorted((expected_counts - observed_counts).elements())
    extra = sorted((observed_counts - expected_counts).elements())
    duplicates = sorted(
        identity for identity, count in observed_counts.items() if count > 1
    )
    if missing or extra or duplicates or len(text_events) != len(expected_identities):
        raise QwenForwardContractError(
            "FA2 text-layer proof requires one exact event per resolved layer",
            code="qwen.fa2_text_layer_coverage",
            context={
                "expected_text_layer_count": len(expected_identities),
                "observed_text_layer_count": len(text_events),
                "missing_identities": missing[:64],
                "extra_identities": extra[:64],
                "duplicate_identities": duplicates[:64],
                "vision_event_count": len(vision_events),
                "unrelated_event_count": len(unrelated_events),
            },
        )
    if unrelated_events:
        raise QwenForwardContractError(
            "unrelated attention registry events are not allowed in Qwen FA2 proof mode",
            code="qwen.fa2_unrelated_attention",
            context={
                "unrelated_identities": [
                    event.identity for event in unrelated_events[:64]
                ]
            },
        )
    orphan_counts = {
        "flash": evidence.orphan_flash_fn_call_count,
        "flash_varlen": evidence.orphan_flash_varlen_fn_call_count,
        "pad": evidence.orphan_pad_fn_call_count,
        "unpad": evidence.orphan_unpad_fn_call_count,
    }
    if any(orphan_counts.values()):
        raise QwenForwardContractError(
            "FA2 proof observed low-level attention calls outside a registry event",
            code="qwen.fa2_branch",
            context={"orphan_low_level_calls": orphan_counts},
        )

    expected_by_identity = {layer.identity: layer for layer in topology.text_layers}
    approved_callable = _approved_fa2_callable_identity()
    dtype_devices: set[tuple[str, str]] = set()
    for event in text_events:
        expected = expected_by_identity[event.identity]
        if event.topology_id != topology.topology_id:
            raise QwenForwardContractError(
                "FA2 event topology identity does not match the resolved model",
                code="qwen.fa2_topology",
                context={"offending_identity": event.identity},
            )
        if (
            event.configured_backend != expected.configured_backend
            or event.registry_key != expected.configured_backend
            or event.registry_key != _ACCEPTED_FA2_BACKEND
        ):
            raise QwenForwardContractError(
                "Qwen text layer used a registry backend different from its FA2 config",
                code="qwen.fa2_attention_implementation",
                context={
                    "offending_identity": event.identity,
                    "configured_backend": event.configured_backend,
                    "registry_key": event.registry_key,
                },
            )
        if (
            event.registry_had_local_override
            or event.resolved_callable != approved_callable
        ):
            raise QwenForwardContractError(
                "Qwen FA2 proof requires the approved installed Transformers integration callable",
                code="qwen.fa2_registry_callable",
                context={
                    "offending_identity": event.identity,
                    "registry_had_local_override": event.registry_had_local_override,
                    "expected_callable": approved_callable.to_artifact_dict(),
                    "observed_callable": event.resolved_callable.to_artifact_dict(),
                },
            )
        if event.completion_status != "completed":
            raise QwenForwardContractError(
                "Qwen text attention event did not complete",
                code="qwen.fa2_branch",
                context={"offending_identity": event.identity},
            )
        if event.attention_mask is not None:
            raise QwenForwardContractError(
                "FA2 padding-free varlen proof requires attention_mask=None",
                code="qwen.fa2_attention_mask",
                context={"offending_identity": event.identity},
            )
        if (
            event.cu_seq_lens_q != plan.segment_boundaries
            or event.cu_seq_lens_k != plan.segment_boundaries
        ):
            raise QwenForwardContractError(
                "FA2 text-layer boundaries must match the packed plan",
                code="qwen.fa2_cu_seq_lens",
                context={
                    "offending_identity": event.identity,
                    "expected_boundaries": list(plan.segment_boundaries),
                    "observed_cu_seq_lens_q": _artifact_value(event.cu_seq_lens_q),
                    "observed_cu_seq_lens_k": _artifact_value(event.cu_seq_lens_k),
                },
            )
        if (
            event.max_length_q != plan.max_length_q
            or event.max_length_k != plan.max_length_k
        ):
            raise QwenForwardContractError(
                "FA2 text-layer max lengths must match the packed plan",
                code="qwen.fa2_max_length",
                context={
                    "offending_identity": event.identity,
                    "expected_max_length_q": plan.max_length_q,
                    "expected_max_length_k": plan.max_length_k,
                    "observed_max_length_q": event.max_length_q,
                    "observed_max_length_k": event.max_length_k,
                },
            )
        low_level_counts = {
            "flash": event.flash_fn_call_count,
            "flash_varlen": event.flash_varlen_fn_call_count,
            "pad": event.pad_fn_call_count,
            "unpad": event.unpad_fn_call_count,
        }
        if low_level_counts != {"flash": 0, "flash_varlen": 1, "pad": 0, "unpad": 0}:
            raise QwenForwardContractError(
                "FA2 text layer must execute exactly one native varlen call and no fallback",
                code="qwen.fa2_branch",
                context={
                    "offending_identity": event.identity,
                    "low_level_call_counts": low_level_counts,
                },
            )
        if len(event.varlen_calls) != 1:
            raise QwenForwardContractError(
                "FA2 text layer must record exactly one native varlen call",
                code="qwen.fa2_observed_call",
                context={"offending_identity": event.identity},
            )
        _validate_observed_call_matches_plan(event.varlen_calls[0], plan)
        low_level_dtype_devices = {
            (
                str(event.varlen_calls[0][name].get("dtype")),
                str(event.varlen_calls[0][name].get("device")),
            )
            for name in ("q", "k", "v")
            if isinstance(event.varlen_calls[0].get(name), Mapping)
        }
        if len(low_level_dtype_devices) != 1:
            raise QwenForwardContractError(
                "FA2 native varlen q/k/v dtype and device must agree",
                code="qwen.fa2_dtype_device",
                context={"offending_identity": event.identity},
            )
        tensors = (event.query, event.key, event.value)
        if any(tensor is None for tensor in tensors):
            raise QwenForwardContractError(
                "FA2 text-layer proof requires q/k/v tensor identities",
                code="qwen.fa2_dtype",
                context={"offending_identity": event.identity},
            )
        concrete_tensors = tuple(tensor for tensor in tensors if tensor is not None)
        event_dtype_devices = {
            (tensor.dtype, tensor.device) for tensor in concrete_tensors
        }
        if len(event_dtype_devices) != 1:
            raise QwenForwardContractError(
                "FA2 q/k/v dtype and device must agree within each text layer",
                code="qwen.fa2_dtype_device",
                context={"offending_identity": event.identity},
            )
        if event_dtype_devices != low_level_dtype_devices:
            raise QwenForwardContractError(
                "FA2 registry and native varlen q/k/v dtype/device evidence must agree",
                code="qwen.fa2_dtype_device",
                context={"offending_identity": event.identity},
            )
        dtype_devices.update(event_dtype_devices)

    canonical_model_dtype = _canonical_dtype(model_dtype)
    if len(dtype_devices) != 1:
        raise QwenForwardContractError(
            "FA2 q/k/v dtype and device must agree across all text layers",
            code="qwen.fa2_dtype_device",
            context={"observed_dtype_devices": sorted(dtype_devices)},
        )
    observed_dtype, observed_device = next(iter(dtype_devices))
    if _canonical_dtype(observed_dtype) != canonical_model_dtype:
        raise QwenForwardContractError(
            "FA2 text-layer q/k/v dtype must match the accepted model dtype",
            code="qwen.fa2_dtype",
            context={
                "model_dtype": model_dtype,
                "observed_dtype": observed_dtype,
            },
        )
    normalized_expected_device = str(torch.device(expected_device))
    if observed_device != normalized_expected_device:
        raise QwenForwardContractError(
            "FA2 text-layer q/k/v device must match the forward/model device",
            code="qwen.fa2_device",
            context={
                "expected_device": normalized_expected_device,
                "observed_device": observed_device,
            },
        )

    observed_call = text_events[0].varlen_calls[0]
    flash_fn_called = any(event.flash_fn_call_count for event in text_events)
    flash_varlen_fn_called = any(
        event.flash_varlen_fn_call_count for event in text_events
    )
    pad_fn_called = any(event.pad_fn_call_count for event in text_events)
    unpad_fn_called = any(event.unpad_fn_call_count for event in text_events)
    if not isinstance(observed_call, Mapping):
        raise QwenForwardContractError(
            "FA2 branch proof requires observed varlen call kwargs",
            code="qwen.fa2_observed_call",
            context={"observed_call": _artifact_value(observed_call)},
        )
    return Fa2VarlenBranchProof(
        observed_branch=PADDING_FREE_VARLEN_BRANCH,
        segment_boundaries=plan.segment_boundaries,
        cu_seq_lens_q=plan.segment_boundaries,
        cu_seq_lens_k=plan.segment_boundaries,
        max_length_q=plan.max_length_q,
        max_length_k=plan.max_length_k,
        resolved_attention_implementation=_ACCEPTED_FA2_BACKEND,
        model_dtype=model_dtype,
        branch_evidence_from_explicit_varlen_kwargs=True,
        flash_fn_called=flash_fn_called,
        flash_varlen_fn_called=flash_varlen_fn_called,
        pad_fn_called=pad_fn_called,
        unpad_fn_called=unpad_fn_called,
        observed_call=observed_call,
        topology=topology,
        text_layer_events=text_events,
        vision_attention_events=vision_events,
        unrelated_attention_events=unrelated_events,
    )


def _segment_boundaries(pack: PackedSequence) -> tuple[int, ...]:
    boundaries = [0]
    for segment in pack.segments:
        if segment.start != boundaries[-1] or segment.end <= segment.start:
            raise QwenForwardContractError(
                "PackedSegment boundaries must be contiguous for FA2 varlen planning",
                code="qwen.fa2_segment_boundaries",
                context={
                    "pack_index": pack.pack_index,
                    "segment_index": segment.segment_index,
                    "start": segment.start,
                    "end": segment.end,
                    "expected_start": boundaries[-1],
                },
            )
        boundaries.append(segment.end)
    if boundaries[-1] != pack.length:
        raise QwenForwardContractError(
            "FA2 segment boundaries must end at pack length",
            code="qwen.fa2_segment_boundaries",
            context={"pack_length": pack.length, "last_boundary": boundaries[-1]},
        )
    return tuple(boundaries)


def _validate_observed_call_matches_plan(
    observed_call: Mapping[str, Any],
    plan: Fa2VarlenPlan,
) -> None:
    observed_cu_seq_lens_q = _observed_call_int_tuple(
        observed_call,
        ("cu_seqlens_q", "cu_seq_lens_q"),
    )
    observed_cu_seq_lens_k = _observed_call_int_tuple(
        observed_call,
        ("cu_seqlens_k", "cu_seq_lens_k"),
    )
    observed_max_length_q = _observed_call_int_value(
        observed_call,
        ("max_seqlen_q", "max_length_q"),
    )
    observed_max_length_k = _observed_call_int_value(
        observed_call,
        ("max_seqlen_k", "max_length_k"),
    )
    if (
        observed_cu_seq_lens_q != plan.segment_boundaries
        or observed_cu_seq_lens_k != plan.segment_boundaries
        or observed_max_length_q != plan.max_length_q
        or observed_max_length_k != plan.max_length_k
    ):
        raise QwenForwardContractError(
            "observed FA2 varlen call kwargs must match the packed plan",
            code="qwen.fa2_observed_call",
            context={
                "segment_boundaries": list(plan.segment_boundaries),
                "observed_cu_seq_lens_q": list(observed_cu_seq_lens_q),
                "observed_cu_seq_lens_k": list(observed_cu_seq_lens_k),
                "expected_max_length_q": plan.max_length_q,
                "expected_max_length_k": plan.max_length_k,
                "observed_max_length_q": observed_max_length_q,
                "observed_max_length_k": observed_max_length_k,
            },
        )


def _observed_call_int_tuple(
    observed_call: Mapping[str, Any],
    keys: tuple[str, ...],
) -> tuple[int, ...]:
    value = _observed_call_value(observed_call, keys)
    return _int_tuple(value, code="qwen.fa2_observed_call")


def _observed_call_int_value(
    observed_call: Mapping[str, Any], keys: tuple[str, ...]
) -> int:
    value = _observed_call_value(observed_call, keys)
    return _int_value(value, code="qwen.fa2_observed_call")


def _observed_call_value(
    observed_call: Mapping[str, Any], keys: tuple[str, ...]
) -> Any:
    for key in keys:
        if key in observed_call:
            return observed_call[key]
    raise QwenForwardContractError(
        "observed FA2 varlen call is missing required kwargs",
        code="qwen.fa2_observed_call",
        context={"missing_any_of": list(keys), "observed_keys": sorted(observed_call)},
    )


def _tensor_int_list(value: torch.Tensor) -> list[int]:
    return [int(item) for item in value.detach().cpu().tolist()]


def _tensor_int_list_or_none(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return _tensor_int_list(value)
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError):
        return None


def _tensor_int_tuple_or_none(value: Any) -> tuple[int, ...] | None:
    values = _tensor_int_list_or_none(value)
    return None if values is None else tuple(values)


def _tensor_execution_identity(value: Any) -> TensorExecutionIdentity | None:
    if not isinstance(value, torch.Tensor):
        return None
    return TensorExecutionIdentity(
        shape=tuple(int(item) for item in value.shape),
        dtype=str(value.dtype),
        device=str(value.device),
    )


def _configured_backend(module: Any) -> str | None:
    value = getattr(getattr(module, "config", None), "_attn_implementation", None)
    return None if value is None else str(value)


def _qualified_class_name(value: Any) -> str:
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _canonical_dtype(value: str) -> str:
    return {
        "torch.bfloat16": "bfloat16",
        "bfloat16": "bfloat16",
        "bf16": "bfloat16",
        "torch.float16": "float16",
        "float16": "float16",
        "fp16": "float16",
    }.get(value, value)


def _mapping_identity_equal(
    observed: Mapping[Any, Any], expected: Mapping[Any, Any]
) -> bool:
    return observed.keys() == expected.keys() and all(
        observed[key] is expected[key] for key in expected
    )


def _callable_identity(value: Callable[..., Any]) -> AttentionCallableIdentity:
    unwrapped = inspect.unwrap(value)
    module = str(getattr(unwrapped, "__module__", type(unwrapped).__module__))
    qualname = str(
        getattr(
            unwrapped,
            "__qualname__",
            getattr(unwrapped, "__name__", type(unwrapped).__qualname__),
        )
    )
    try:
        source_file = inspect.getsourcefile(unwrapped)
    except (OSError, TypeError):
        source_file = None
    try:
        source_lines, source_first_line = inspect.getsourcelines(unwrapped)
        content = "".join(source_lines).encode("utf-8")
    except (OSError, TypeError):
        source_first_line = None
        code = getattr(unwrapped, "__code__", None)
        if code is not None:
            content = bytes(code.co_code)
        else:
            content = f"{module}:{qualname}:{type(unwrapped).__qualname__}".encode(
                "utf-8"
            )
    return AttentionCallableIdentity(
        module=module,
        qualname=qualname,
        source_file=None if source_file is None else str(source_file),
        source_first_line=source_first_line,
        content_sha256=hashlib.sha256(content).hexdigest(),
    )


def _approved_fa2_callable_identity() -> AttentionCallableIdentity:
    try:
        from transformers.integrations.flash_attention import flash_attention_forward
    except ImportError as exc:
        raise QwenForwardContractError(
            "approved Transformers FA2 integration callable is unavailable",
            code="qwen.fa2_registry_callable",
            cause=exc,
        ) from exc
    return _callable_identity(flash_attention_forward)


def _shape_dtype_device(value: Any) -> dict[str, Any]:
    if not isinstance(value, torch.Tensor):
        return {"type": type(value).__name__}
    return {
        "shape": [int(item) for item in value.shape],
        "dtype": str(value.dtype),
        "device": str(value.device),
    }


def _varlen_flash_call_artifact(
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    cu_seqlens_q = kwargs.get("cu_seqlens_q")
    cu_seqlens_k = kwargs.get("cu_seqlens_k")
    max_seqlen_q = kwargs.get("max_seqlen_q")
    max_seqlen_k = kwargs.get("max_seqlen_k")
    if cu_seqlens_q is None and len(args) > 3:
        cu_seqlens_q = args[3]
    if cu_seqlens_k is None and len(args) > 4:
        cu_seqlens_k = args[4]
    if max_seqlen_q is None and len(args) > 5:
        max_seqlen_q = args[5]
    if max_seqlen_k is None and len(args) > 6:
        max_seqlen_k = args[6]
    return {
        "q": _shape_dtype_device(args[0] if len(args) > 0 else None),
        "k": _shape_dtype_device(args[1] if len(args) > 1 else None),
        "v": _shape_dtype_device(args[2] if len(args) > 2 else None),
        "cu_seqlens_q": _tensor_int_list_or_none(cu_seqlens_q),
        "cu_seqlens_k": _tensor_int_list_or_none(cu_seqlens_k),
        "max_seqlen_q": None if max_seqlen_q is None else int(max_seqlen_q),
        "max_seqlen_k": None if max_seqlen_k is None else int(max_seqlen_k),
        "flash_kwargs": {
            str(key): _artifact_value(value)
            for key, value in kwargs.items()
            if key
            not in {
                "cu_seqlens_q",
                "cu_seqlens_k",
                "max_seqlen_q",
                "max_seqlen_k",
            }
        },
    }


def _ordinary_flash_call_artifact(
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "q": _shape_dtype_device(args[0] if args else None),
        "flash_kwargs": {
            str(key): _artifact_value(value) for key, value in kwargs.items()
        },
    }


def _int_tuple(value: Any, *, code: str) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise QwenForwardContractError(
            "FA2 evidence value must be an integer sequence",
            code=code,
            context={"value": _artifact_value(value)},
            cause=exc,
        ) from exc


def _int_value(value: Any, *, code: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise QwenForwardContractError(
            "FA2 evidence value must be an integer",
            code=code,
            context={"value": _artifact_value(value)},
            cause=exc,
        ) from exc


def _artifact_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _artifact_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_artifact_value(item) for item in value]
    return value


__all__ = [
    "AttentionCallableIdentity",
    "AttentionEventKind",
    "AttentionProofEvent",
    "Fa2AttentionProofEvidence",
    "PADDING_FREE_VARLEN_BRANCH",
    "Fa2VarlenBranchCapture",
    "Fa2VarlenBranchProof",
    "Fa2VarlenPlan",
    "QwenAttentionTopology",
    "QwenTextLayerIdentity",
    "TensorExecutionIdentity",
    "build_fa2_varlen_plan",
    "capture_fa2_varlen_branch",
    "validate_fa2_varlen_branch_evidence",
    "validate_fa2_varlen_plan_matches_pack",
]
