"""Typed, value-free autograd graph-owner attribution for the Human-13 probe."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, Mapping

import torch

from src.artifacts.json_values import json_sha256


GRAPH_OWNER_SCHEMA = "human13_graph_owner_attribution.v1"
GRAPH_INPUT_ATTESTATION_SCHEMA = "human13_graph_input_attestation.v1"
MAX_FOREIGN_LEAF_DETAILS = 8

GraphInputRole = Literal[
    "input_ids",
    "attention_mask",
    "position_ids",
    "pixel_values",
    "image_grid_thw",
    "logits_to_keep",
    "task2_replay_logprob",
    "task3_compiler_logit",
    "objective",
    "trajectory_component",
    "compiler_component",
]
GraphOwnerDisposition = Literal[
    "admitted_model_graph",
    "admitted_non_model_input",
    "detached_or_no_grad",
    "foreign_parameter",
    "unregistered_trainable_input",
    "wrong_model_object",
    "receipt_tensor_mismatch",
    "stale_or_rebuilt_graph",
]

_GRAPH_INPUT_ROLES = frozenset(GraphInputRole.__args__)
_GRAPH_OWNER_DISPOSITIONS = frozenset(GraphOwnerDisposition.__args__)


def _type_name(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _exact_bool(value: object, *, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field} must be boolean")
    return value


@dataclass(frozen=True)
class GraphLeafFingerprint:
    object_id: int
    concrete_type: str
    registered_name: str | None
    registered: bool
    requires_grad: bool
    device: str
    dtype: str
    shape: tuple[int, ...]
    model_object_id: int
    input_role: GraphInputRole

    def __post_init__(self) -> None:
        if (
            isinstance(self.object_id, bool)
            or not isinstance(self.object_id, int)
            or self.object_id <= 0
            or isinstance(self.model_object_id, bool)
            or not isinstance(self.model_object_id, int)
            or self.model_object_id <= 0
        ):
            raise ValueError("graph leaf object identity is invalid")
        if not self.concrete_type or not self.device or not self.dtype:
            raise ValueError("graph leaf type/storage metadata is incomplete")
        if self.registered != (self.registered_name is not None):
            raise ValueError("graph leaf registration metadata differs")
        if self.registered_name == "":
            raise ValueError("registered graph leaf name is empty")
        if type(self.requires_grad) is not bool:
            raise ValueError("graph leaf requires-grad flag is not boolean")
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in self.shape
        ):
            raise ValueError("graph leaf shape is invalid")
        _input_role(self.input_role)

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "concrete_type": self.concrete_type,
            "registered_name": self.registered_name,
            "registered": self.registered,
            "requires_grad": self.requires_grad,
            "device": self.device,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "model_object_id": self.model_object_id,
            "input_role": self.input_role,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GraphLeafFingerprint:
        return cls(
            object_id=int(value["object_id"]),
            concrete_type=str(value["concrete_type"]),
            registered_name=(
                None
                if value.get("registered_name") is None
                else str(value["registered_name"])
            ),
            registered=_exact_bool(value["registered"], field="leaf.registered"),
            requires_grad=_exact_bool(
                value["requires_grad"], field="leaf.requires_grad"
            ),
            device=str(value["device"]),
            dtype=str(value["dtype"]),
            shape=tuple(int(item) for item in value["shape"]),
            model_object_id=int(value["model_object_id"]),
            input_role=_input_role(value["input_role"]),
        )


@dataclass(frozen=True)
class GraphOwnerAttributionReceipt:
    model_object_id: int
    tensor_object_id: int
    tensor_concrete_type: str
    tensor_requires_grad: bool
    tensor_device: str
    tensor_dtype: str
    tensor_shape: tuple[int, ...]
    input_role: GraphInputRole
    leaves: tuple[GraphLeafFingerprint, ...]
    foreign_leaves: tuple[GraphLeafFingerprint, ...]
    foreign_leaf_count: int
    admitted: bool
    disposition: GraphOwnerDisposition
    schema_version: Literal["human13_graph_owner_attribution.v1"] = (
        GRAPH_OWNER_SCHEMA
    )

    def __post_init__(self) -> None:
        if self.schema_version != GRAPH_OWNER_SCHEMA:
            raise ValueError("graph-owner receipt schema differs")
        if (
            isinstance(self.model_object_id, bool)
            or not isinstance(self.model_object_id, int)
            or self.model_object_id <= 0
            or isinstance(self.tensor_object_id, bool)
            or not isinstance(self.tensor_object_id, int)
            or self.tensor_object_id <= 0
        ):
            raise ValueError("graph-owner object identity is invalid")
        if not self.tensor_concrete_type or not self.tensor_device or not self.tensor_dtype:
            raise ValueError("graph-owner tensor metadata is incomplete")
        if type(self.tensor_requires_grad) is not bool or type(self.admitted) is not bool:
            raise ValueError("graph-owner boolean metadata is invalid")
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in self.tensor_shape
        ):
            raise ValueError("graph-owner tensor shape is invalid")
        _input_role(self.input_role)
        if self.disposition not in _GRAPH_OWNER_DISPOSITIONS:
            raise ValueError("graph-owner disposition is invalid")
        if any(type(leaf) is not GraphLeafFingerprint for leaf in self.leaves):
            raise ValueError("graph-owner leaf fingerprint type differs")
        if len({leaf.object_id for leaf in self.leaves}) != len(self.leaves):
            raise ValueError("graph-owner leaf identities are duplicated")
        if any(
            leaf.model_object_id != self.model_object_id
            or leaf.input_role != self.input_role
            for leaf in self.leaves
        ):
            raise ValueError("graph-owner leaf lineage differs")
        leaf_ids = frozenset(leaf.object_id for leaf in self.leaves)
        if (
            len(self.foreign_leaves) > MAX_FOREIGN_LEAF_DETAILS
            or isinstance(self.foreign_leaf_count, bool)
            or not isinstance(self.foreign_leaf_count, int)
            or self.foreign_leaf_count < len(self.foreign_leaves)
            or (
                self.foreign_leaf_count > len(self.foreign_leaves)
                and len(self.foreign_leaves) != MAX_FOREIGN_LEAF_DETAILS
            )
            or any(
                type(leaf) is not GraphLeafFingerprint
                or leaf.object_id not in leaf_ids
                for leaf in self.foreign_leaves
            )
        ):
            raise ValueError("graph-owner foreign leaf evidence is invalid")
        if self.admitted != self.disposition.startswith("admitted_"):
            raise ValueError("graph-owner admission/disposition differs")
        if self.disposition == "admitted_model_graph" and (
            not self.tensor_requires_grad
            or not self.leaves
            or self.foreign_leaves
            or self.foreign_leaf_count != 0
        ):
            raise ValueError("admitted model graph lacks exact registered leaves")
        if self.disposition == "admitted_non_model_input" and (
            self.tensor_requires_grad
            or self.leaves
            or self.foreign_leaves
            or self.foreign_leaf_count != 0
        ):
            raise ValueError("admitted non-model input carries graph ownership")
        if self.disposition in (
            "foreign_parameter",
            "unregistered_trainable_input",
        ) and self.foreign_leaf_count == 0:
            raise ValueError("foreign graph disposition lacks foreign leaves")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_object_id": self.model_object_id,
            "tensor_object_id": self.tensor_object_id,
            "tensor_concrete_type": self.tensor_concrete_type,
            "tensor_requires_grad": self.tensor_requires_grad,
            "tensor_device": self.tensor_device,
            "tensor_dtype": self.tensor_dtype,
            "tensor_shape": list(self.tensor_shape),
            "input_role": self.input_role,
            "leaves": [leaf.to_dict() for leaf in self.leaves],
            "foreign_leaves": [leaf.to_dict() for leaf in self.foreign_leaves],
            "foreign_leaf_count": self.foreign_leaf_count,
            "foreign_leaf_details_truncated": self.foreign_leaf_count
            > len(self.foreign_leaves),
            "admitted": self.admitted,
            "disposition": self.disposition,
        }

    @property
    def content_sha256(self) -> str:
        return json_sha256(self._payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["content_sha256"] = self.content_sha256
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GraphOwnerAttributionReceipt:
        if value.get("schema_version") != GRAPH_OWNER_SCHEMA:
            raise ValueError("graph-owner receipt schema differs")
        result = cls(
            model_object_id=int(value["model_object_id"]),
            tensor_object_id=int(value["tensor_object_id"]),
            tensor_concrete_type=str(value["tensor_concrete_type"]),
            tensor_requires_grad=_exact_bool(
                value["tensor_requires_grad"],
                field="tensor_requires_grad",
            ),
            tensor_device=str(value["tensor_device"]),
            tensor_dtype=str(value["tensor_dtype"]),
            tensor_shape=tuple(int(item) for item in value["tensor_shape"]),
            input_role=_input_role(value["input_role"]),
            leaves=tuple(
                GraphLeafFingerprint.from_dict(item) for item in value["leaves"]
            ),
            foreign_leaves=tuple(
                GraphLeafFingerprint.from_dict(item)
                for item in value["foreign_leaves"]
            ),
            foreign_leaf_count=int(value["foreign_leaf_count"]),
            admitted=_exact_bool(value["admitted"], field="admitted"),
            disposition=str(value["disposition"]),  # type: ignore[arg-type]
        )
        if value.get("foreign_leaf_count") != result.foreign_leaf_count:
            raise ValueError("graph-owner foreign leaf count differs")
        if value.get("foreign_leaf_details_truncated") != (
            result.foreign_leaf_count > len(result.foreign_leaves)
        ):
            raise ValueError("graph-owner foreign leaf truncation flag differs")
        if value.get("content_sha256") != result.content_sha256:
            raise ValueError("graph-owner receipt content hash differs")
        return result


@dataclass(frozen=True)
class GraphInputAttestationReceipt:
    model_object_id: int
    input_receipts: tuple[GraphOwnerAttributionReceipt, ...]
    admitted: bool = True
    schema_version: Literal["human13_graph_input_attestation.v1"] = (
        GRAPH_INPUT_ATTESTATION_SCHEMA
    )

    def __post_init__(self) -> None:
        if self.schema_version != GRAPH_INPUT_ATTESTATION_SCHEMA:
            raise ValueError("graph input attestation schema differs")
        if type(self.admitted) is not bool or self.admitted is not True:
            raise ValueError("graph input attestation must be admitted")
        if not self.input_receipts or any(
            type(receipt) is not GraphOwnerAttributionReceipt
            or receipt.model_object_id != self.model_object_id
            or receipt.disposition != "admitted_non_model_input"
            for receipt in self.input_receipts
        ):
            raise ValueError("graph input attestation receipt lineage differs")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_object_id": self.model_object_id,
            "input_receipts": [receipt.to_dict() for receipt in self.input_receipts],
            "input_receipt_sha256s": [
                receipt.content_sha256 for receipt in self.input_receipts
            ],
            "admitted": self.admitted,
        }

    @property
    def content_sha256(self) -> str:
        return json_sha256(self._payload())

    def to_artifact_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["content_sha256"] = self.content_sha256
        return payload

    @classmethod
    def from_artifact_dict(
        cls, value: Mapping[str, Any]
    ) -> GraphInputAttestationReceipt:
        if value.get("schema_version") != GRAPH_INPUT_ATTESTATION_SCHEMA:
            raise ValueError("graph input attestation schema differs")
        receipts = tuple(
            GraphOwnerAttributionReceipt.from_dict(item)
            for item in value.get("input_receipts", ())
        )
        if tuple(value.get("input_receipt_sha256s", ())) != tuple(
            receipt.content_sha256 for receipt in receipts
        ):
            raise ValueError("graph input attestation receipt lineage differs")
        result = cls(
            model_object_id=int(value["model_object_id"]),
            input_receipts=receipts,
            admitted=_exact_bool(value["admitted"], field="admitted"),
        )
        if value.get("content_sha256") != result.content_sha256:
            raise ValueError("graph input attestation content hash differs")
        return result


class GraphOwnerAttributionError(ValueError):
    def __init__(self, receipt: GraphOwnerAttributionReceipt) -> None:
        self.receipt = receipt
        foreign = ",".join(
            f"{leaf.object_id}:{leaf.concrete_type}:{leaf.registered_name or 'unregistered'}"
            for leaf in receipt.foreign_leaves[:MAX_FOREIGN_LEAF_DETAILS]
        )
        super().__init__(
            f"{receipt.disposition}: graph owner attribution; "
            f"input_role={receipt.input_role}; "
            f"graph_owner_receipt_sha256={receipt.content_sha256}; "
            f"foreign_leaves=[{foreign}]"
        )


def _input_role(value: object) -> GraphInputRole:
    if not isinstance(value, str) or value not in _GRAPH_INPUT_ROLES:
        raise ValueError("graph-owner input role is outside the bounded vocabulary")
    return value  # type: ignore[return-value]


def _reachable_autograd_leaves(value: torch.Tensor) -> tuple[torch.Tensor, ...]:
    if not value.requires_grad:
        return ()
    if value.is_leaf:
        return (value,)
    pending = [value.grad_fn]
    visited: set[object] = set()
    leaves: dict[int, torch.Tensor] = {}
    while pending:
        node = pending.pop()
        if node is None or node in visited:
            continue
        visited.add(node)
        variable = getattr(node, "variable", None)
        if isinstance(variable, torch.Tensor) and variable.requires_grad:
            leaves[id(variable)] = variable
        pending.extend(next_node for next_node, _ in node.next_functions)
    return tuple(leaves[key] for key in sorted(leaves))


def _leaf_fingerprint(
    leaf: torch.Tensor,
    *,
    model: torch.nn.Module,
    registered_names: Mapping[int, str],
    input_role: GraphInputRole,
) -> GraphLeafFingerprint:
    registered_name = registered_names.get(id(leaf))
    return GraphLeafFingerprint(
        object_id=id(leaf),
        concrete_type=_type_name(leaf),
        registered_name=registered_name,
        registered=registered_name is not None,
        requires_grad=bool(leaf.requires_grad),
        device=str(leaf.device),
        dtype=str(leaf.dtype),
        shape=tuple(int(item) for item in leaf.shape),
        model_object_id=id(model),
        input_role=input_role,
    )


def attribute_model_graph(
    value: torch.Tensor,
    *,
    model: torch.nn.Module,
    input_role: GraphInputRole,
) -> GraphOwnerAttributionReceipt:
    role = _input_role(input_role)
    registered_parameters = {
        id(parameter): (name, parameter)
        for name, parameter in model.named_parameters()
    }
    names = {object_id: item[0] for object_id, item in registered_parameters.items()}
    leaves = _reachable_autograd_leaves(value)
    fingerprints = tuple(
        _leaf_fingerprint(
            leaf,
            model=model,
            registered_names=names,
            input_role=role,
        )
        for leaf in leaves
    )
    all_foreign = tuple(
        leaf
        for leaf, fingerprint in zip(leaves, fingerprints, strict=True)
        if (
            not fingerprint.registered
            or not isinstance(leaf, torch.nn.Parameter)
            or registered_parameters[id(leaf)][1] is not leaf
            or not leaf.requires_grad
        )
    )
    foreign = all_foreign[:MAX_FOREIGN_LEAF_DETAILS]
    foreign_ids = frozenset(id(leaf) for leaf in foreign)
    if not value.requires_grad or not leaves:
        disposition: GraphOwnerDisposition = "detached_or_no_grad"
    elif any(isinstance(leaf, torch.nn.Parameter) for leaf in all_foreign):
        disposition = "foreign_parameter"
    elif all_foreign:
        disposition = "unregistered_trainable_input"
    else:
        disposition = "admitted_model_graph"
    receipt = GraphOwnerAttributionReceipt(
        model_object_id=id(model),
        tensor_object_id=id(value),
        tensor_concrete_type=_type_name(value),
        tensor_requires_grad=bool(value.requires_grad),
        tensor_device=str(value.device),
        tensor_dtype=str(value.dtype),
        tensor_shape=tuple(int(item) for item in value.shape),
        input_role=role,
        leaves=fingerprints,
        foreign_leaves=tuple(
            fingerprint
            for leaf, fingerprint in zip(leaves, fingerprints, strict=True)
            if id(leaf) in foreign_ids
        ),
        foreign_leaf_count=len(all_foreign),
        admitted=disposition == "admitted_model_graph",
        disposition=disposition,
    )
    if not receipt.admitted:
        raise GraphOwnerAttributionError(receipt)
    return receipt


def attest_non_model_input(
    value: torch.Tensor,
    *,
    model: torch.nn.Module,
    input_role: GraphInputRole,
) -> GraphOwnerAttributionReceipt:
    role = _input_role(input_role)
    if value.requires_grad:
        registered_names = {
            id(parameter): name for name, parameter in model.named_parameters()
        }
        leaf = _leaf_fingerprint(
            value,
            model=model,
            registered_names=registered_names,
            input_role=role,
        )
        disposition: GraphOwnerDisposition = (
            "foreign_parameter"
            if isinstance(value, torch.nn.Parameter)
            else "unregistered_trainable_input"
        )
        receipt = GraphOwnerAttributionReceipt(
            model_object_id=id(model),
            tensor_object_id=id(value),
            tensor_concrete_type=_type_name(value),
            tensor_requires_grad=True,
            tensor_device=str(value.device),
            tensor_dtype=str(value.dtype),
            tensor_shape=tuple(int(item) for item in value.shape),
            input_role=role,
            leaves=(leaf,),
            foreign_leaves=(leaf,),
            foreign_leaf_count=1,
            admitted=False,
            disposition=disposition,
        )
        raise GraphOwnerAttributionError(receipt)
    return GraphOwnerAttributionReceipt(
        model_object_id=id(model),
        tensor_object_id=id(value),
        tensor_concrete_type=_type_name(value),
        tensor_requires_grad=False,
        tensor_device=str(value.device),
        tensor_dtype=str(value.dtype),
        tensor_shape=tuple(int(item) for item in value.shape),
        input_role=role,
        leaves=(),
        foreign_leaves=(),
        foreign_leaf_count=0,
        admitted=True,
        disposition="admitted_non_model_input",
    )


def validate_expected_model_graph(
    value: torch.Tensor,
    *,
    model: torch.nn.Module,
    input_role: GraphInputRole,
    expected: GraphOwnerAttributionReceipt | object,
) -> GraphOwnerAttributionReceipt:
    role = _input_role(input_role)
    if (
        type(expected) is GraphOwnerAttributionReceipt
        and expected.model_object_id != id(model)
    ):
        mismatch = replace(
            expected,
            model_object_id=id(model),
            leaves=tuple(
                replace(leaf, model_object_id=id(model)) for leaf in expected.leaves
            ),
            foreign_leaves=tuple(
                replace(leaf, model_object_id=id(model))
                for leaf in expected.foreign_leaves
            ),
            admitted=False,
            disposition="wrong_model_object",
        )
        raise GraphOwnerAttributionError(mismatch)
    observed = attribute_model_graph(value, model=model, input_role=role)
    if type(expected) is not GraphOwnerAttributionReceipt:
        mismatch = GraphOwnerAttributionReceipt(
            **{
                **observed.__dict__,
                "admitted": False,
                "disposition": "receipt_tensor_mismatch",
            }
        )
        raise GraphOwnerAttributionError(mismatch)
    expected_receipt = expected
    GraphOwnerAttributionReceipt.from_dict(expected_receipt.to_dict())
    if expected_receipt.tensor_object_id != id(value):
        mismatch = GraphOwnerAttributionReceipt(
            **{
                **observed.__dict__,
                "admitted": False,
                "disposition": "stale_or_rebuilt_graph",
            }
        )
        raise GraphOwnerAttributionError(mismatch)
    if expected_receipt.content_sha256 != observed.content_sha256:
        mismatch = GraphOwnerAttributionReceipt(
            **{
                **observed.__dict__,
                "admitted": False,
                "disposition": "receipt_tensor_mismatch",
            }
        )
        raise GraphOwnerAttributionError(mismatch)
    return observed


__all__ = [
    "GRAPH_INPUT_ATTESTATION_SCHEMA",
    "GRAPH_OWNER_SCHEMA",
    "GraphInputAttestationReceipt",
    "GraphLeafFingerprint",
    "GraphOwnerAttributionError",
    "GraphOwnerAttributionReceipt",
    "attest_non_model_input",
    "attribute_model_graph",
    "validate_expected_model_graph",
]
