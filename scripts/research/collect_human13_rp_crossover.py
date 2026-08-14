#!/usr/bin/env python3
"""Fresh, receipted RP-aware Human-13 K16 crossover acquisition.

The historical discovery planner is deliberately not reused: it seals
``top_p=.95``.  This collector admits only immutable native request/output
receipts returned by an injected vLLM boundary; a plan is never allowed to
overwrite native output identity.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from types import MappingProxyType
from typing import Any, cast

from scripts.research.build_human13_k_union_manifest import EXPECTED_IMAGE_IDENTITIES
from scripts.research.human13_k_trajectory_contracts import (
    AcquisitionGroup,
    ArtifactIdentity,
    CompleteTrajectoryEvidence,
    GeneratedTokenEvidence,
    PolicyContract,
    ReplayTolerance,
)


REQUESTS_PER_BATCH = 4
BATCHES_PER_IMAGE = 4
REQUESTS_PER_IMAGE = REQUESTS_PER_BATCH * BATCHES_PER_IMAGE
NATURAL_STOP_TOKEN_ID = 151645
MAX_NEW_TOKENS = 512
PROCESSOR_ORDER = ("repetition_penalty", "temperature", "log_softmax")
QUALIFICATION_SEEDS = tuple(range(30001, 30017))
MATRIX_SEED_GROUPS = {
    "matrix_a": tuple(range(31001, 31017)),
    "matrix_b": tuple(range(32001, 32017)),
    "matrix_c": tuple(range(33001, 33017)),
}
_SEED_GROUPS = {"qualification": QUALIFICATION_SEEDS, **MATRIX_SEED_GROUPS}
_SAMPLING_BASE = {
    "n": 1,
    "temperature": 0.4,
    "top_p": 1.0,
    "top_k": None,
    "max_new_tokens": MAX_NEW_TOKENS,
    "stop_token_ids": (NATURAL_STOP_TOKEN_ID,),
    "ignore_eos": False,
}
_NATIVE_SAMPLING_BASE = {**_SAMPLING_BASE, "top_k": 0, "logprobs": 1}
_DIGEST = set("0123456789abcdef")


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _digest(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value) - _DIGEST:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _mapping(value: object, *, field: str, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{field} schema differs from the canonical receipt")
    return value


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    if not token_ids or any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in token_ids):
        raise ValueError("prompt token ids must be nonempty nonnegative integers")
    return _sha256({"token_ids": list(token_ids)})


@dataclass(frozen=True)
class AcquisitionRequest:
    image_id: int
    seed_group_id: str
    seed: int
    physical_batch_index: int
    request_order_in_batch: int
    request_id: str
    sampling: Mapping[str, object]


@dataclass(frozen=True)
class AcquisitionBatch:
    image_id: int
    seed_group_id: str
    batch_index: int
    requests: tuple[AcquisitionRequest, ...]


@dataclass(frozen=True)
class AcquisitionGroupPlan:
    image_id: int
    seed_group_id: str
    repetition_penalty: float
    batches: tuple[AcquisitionBatch, ...]

    @property
    def requests(self) -> tuple[AcquisitionRequest, ...]:
        return tuple(request for batch in self.batches for request in batch.requests)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "human13_rp_crossover_plan.v1",
            "image_id": self.image_id,
            "seed_group_id": self.seed_group_id,
            "repetition_penalty": self.repetition_penalty,
            "batches": [
                {"batch_index": batch.batch_index, "requests": [
                    {
                        "image_id": request.image_id,
                        "seed_group_id": request.seed_group_id,
                        "seed": request.seed,
                        "physical_batch_index": request.physical_batch_index,
                        "request_order_in_batch": request.request_order_in_batch,
                        "request_id": request.request_id,
                        "sampling": dict(request.sampling),
                    }
                    for request in batch.requests
                ]}
                for batch in self.batches
            ],
        }

    @property
    def content_sha256(self) -> str:
        return _sha256(self.to_dict())


@dataclass(frozen=True)
class NativeRequestReceipt:
    """Content-addressed actual native request submitted at the vLLM boundary."""

    request_id: str
    seed: int
    physical_batch_index: int
    request_order_in_batch: int
    sampling_params: Mapping[str, object]
    prompt_token_ids_sha256: str
    model_id: str
    model_identity_sha256: str
    session_identity_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("native request receipt requires a request identity")
        for field in ("seed", "physical_batch_index", "request_order_in_batch"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"native request receipt {field} must be nonnegative")
        if not isinstance(self.sampling_params, Mapping):
            raise ValueError("native request receipt sampling parameters must be a mapping")
        sampling_params = dict(self.sampling_params)
        stop_token_ids = sampling_params.get("stop_token_ids")
        if isinstance(stop_token_ids, Sequence) and not isinstance(stop_token_ids, (str, bytes)):
            sampling_params["stop_token_ids"] = tuple(stop_token_ids)
        object.__setattr__(self, "sampling_params", MappingProxyType(sampling_params))
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError("native request receipt model identity must be nonempty")
        for field in ("prompt_token_ids_sha256", "model_identity_sha256", "session_identity_sha256"):
            object.__setattr__(self, field, _digest(getattr(self, field), field=field))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "human13_native_request_receipt.v1",
            "request_id": self.request_id,
            "seed": self.seed,
            "physical_batch_index": self.physical_batch_index,
            "request_order_in_batch": self.request_order_in_batch,
            "sampling_params": dict(self.sampling_params),
            "prompt_token_ids_sha256": self.prompt_token_ids_sha256,
            "model_id": self.model_id,
            "model_identity_sha256": self.model_identity_sha256,
            "session_identity_sha256": self.session_identity_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> "NativeRequestReceipt":
        item = _mapping(value, field="native request receipt", keys={
            "schema_version", "request_id", "seed", "physical_batch_index", "request_order_in_batch",
            "sampling_params", "prompt_token_ids_sha256", "model_id", "model_identity_sha256", "session_identity_sha256",
        })
        payload: dict[str, Any] = dict(item)
        if payload["schema_version"] != "human13_native_request_receipt.v1":
            raise ValueError("native request receipt schema version differs")
        if not isinstance(payload["sampling_params"], Mapping):
            raise ValueError("native request receipt sampling parameters are missing")
        return cls(request_id=str(payload["request_id"]), seed=int(payload["seed"]),
            physical_batch_index=int(payload["physical_batch_index"]), request_order_in_batch=int(payload["request_order_in_batch"]),
            sampling_params=dict(payload["sampling_params"]), prompt_token_ids_sha256=str(payload["prompt_token_ids_sha256"]),
            model_id=str(payload["model_id"]), model_identity_sha256=str(payload["model_identity_sha256"]),
            session_identity_sha256=str(payload["session_identity_sha256"]))

    @property
    def content_sha256(self) -> str:
        return _sha256(self.to_dict())


@dataclass(frozen=True)
class NativeOutputReceipt:
    """Content-addressed native output tied to its actual submitted request."""

    native_request_receipt_sha256: str
    request_id: str
    seed: int
    physical_batch_index: int
    request_order_in_batch: int
    prompt_token_ids: tuple[int, ...]
    source_sha256: str
    manifest_sha256: str
    model_id: str
    tokenizer_id: str
    processor_id: str
    processor_order: tuple[str, ...]
    sampler_backend_id: str
    generated_token_ids: tuple[int, ...]
    processed_logprobs: tuple[float, ...]
    terminal_kind: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "native_request_receipt_sha256", _digest(self.native_request_receipt_sha256, field="native_request_receipt_sha256"))
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("native output receipt requires a request identity")
        for field in ("seed", "physical_batch_index", "request_order_in_batch"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"native output receipt {field} must be nonnegative")
        object.__setattr__(self, "prompt_token_ids", _token_ids(self.prompt_token_ids, field="prompt_token_ids", nonempty=True))
        generated = _token_ids(self.generated_token_ids, field="generated_token_ids", nonempty=True)
        scores = _scores(self.processed_logprobs)
        if len(generated) != len(scores):
            raise ValueError("native output receipt requires complete chosen-token log probabilities")
        object.__setattr__(self, "generated_token_ids", generated)
        object.__setattr__(self, "processed_logprobs", scores)
        for field in ("source_sha256", "manifest_sha256"):
            object.__setattr__(self, field, _digest(getattr(self, field), field=field))
        for field in ("model_id", "tokenizer_id", "processor_id", "sampler_backend_id"):
            if not isinstance(getattr(self, field), str) or not getattr(self, field):
                raise ValueError(f"native output receipt {field} must be nonempty")
        if self.processor_order != PROCESSOR_ORDER:
            raise ValueError("native output receipt processor order differs from the sealed policy")
        if self.terminal_kind == "natural_stop":
            if generated[-1] != NATURAL_STOP_TOKEN_ID:
                raise ValueError("natural stop must end with im_end")
        elif self.terminal_kind == "cap_stop":
            if len(generated) != MAX_NEW_TOKENS or NATURAL_STOP_TOKEN_ID in generated:
                raise ValueError("cap stop must retain all 512 tokens before im_end")
        else:
            raise ValueError("native terminal kind must be natural_stop or cap_stop")

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": "human13_native_output_receipt.v1", **asdict(self)}

    @classmethod
    def from_dict(cls, value: object) -> "NativeOutputReceipt":
        item = _mapping(value, field="native output receipt", keys={
            "schema_version", "native_request_receipt_sha256", "request_id", "seed", "physical_batch_index",
            "request_order_in_batch", "prompt_token_ids", "source_sha256", "manifest_sha256", "model_id",
            "tokenizer_id", "processor_id", "processor_order", "sampler_backend_id", "generated_token_ids",
            "processed_logprobs", "terminal_kind",
        })
        payload: dict[str, Any] = dict(item)
        if payload["schema_version"] != "human13_native_output_receipt.v1":
            raise ValueError("native output receipt schema version differs")
        for field in ("prompt_token_ids", "processor_order", "generated_token_ids", "processed_logprobs"):
            if not isinstance(payload[field], list):
                raise ValueError(f"native output receipt {field} must be a JSON list")
        return cls(native_request_receipt_sha256=str(payload["native_request_receipt_sha256"]), request_id=str(payload["request_id"]),
            seed=int(payload["seed"]), physical_batch_index=int(payload["physical_batch_index"]),
            request_order_in_batch=int(payload["request_order_in_batch"]), prompt_token_ids=tuple(payload["prompt_token_ids"]),
            source_sha256=str(payload["source_sha256"]), manifest_sha256=str(payload["manifest_sha256"]), model_id=str(payload["model_id"]),
            tokenizer_id=str(payload["tokenizer_id"]), processor_id=str(payload["processor_id"]), processor_order=tuple(payload["processor_order"]),
            sampler_backend_id=str(payload["sampler_backend_id"]), generated_token_ids=tuple(payload["generated_token_ids"]),
            processed_logprobs=tuple(payload["processed_logprobs"]), terminal_kind=str(payload["terminal_kind"]))

    @property
    def content_sha256(self) -> str:
        return _sha256(self.to_dict())


@dataclass(frozen=True)
class NativeBatchReceipt:
    """One physical native submission; executor validation establishes plan binding."""

    requests: tuple[NativeRequestReceipt, ...]
    outputs: tuple[NativeOutputReceipt, ...]

    def __post_init__(self) -> None:
        requests, outputs = tuple(self.requests), tuple(self.outputs)
        if len(requests) != REQUESTS_PER_BATCH or len(outputs) != REQUESTS_PER_BATCH:
            raise ValueError("native batch receipt requires exactly four requests and outputs")
        if any(type(item) is not NativeRequestReceipt for item in requests) or any(type(item) is not NativeOutputReceipt for item in outputs):
            raise ValueError("native batch receipt requires sealed native receipt types")
        object.__setattr__(self, "requests", requests)
        object.__setattr__(self, "outputs", outputs)

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": "human13_native_batch_receipt.v1", "requests": [item.to_dict() for item in self.requests], "outputs": [item.to_dict() for item in self.outputs]}

    @classmethod
    def from_dict(cls, value: object) -> "NativeBatchReceipt":
        item = _mapping(value, field="native batch receipt", keys={"schema_version", "requests", "outputs"})
        if item["schema_version"] != "human13_native_batch_receipt.v1":
            raise ValueError("native batch receipt schema version differs")
        if not isinstance(item["requests"], list) or not isinstance(item["outputs"], list):
            raise ValueError("native batch receipt members must be JSON lists")
        return cls(requests=tuple(NativeRequestReceipt.from_dict(value) for value in item["requests"]),
            outputs=tuple(NativeOutputReceipt.from_dict(value) for value in item["outputs"]))

    @property
    def content_sha256(self) -> str:
        return _sha256(self.to_dict())


@dataclass(frozen=True)
class NativeReceiptsArtifact:
    """The canonical native preimages required to independently reconstruct K16."""

    plan_sha256: str
    plan_request_ids: tuple[str, ...]
    batch_receipts: tuple[NativeBatchReceipt, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_request_ids", tuple(self.plan_request_ids))
        object.__setattr__(self, "batch_receipts", tuple(self.batch_receipts))
        object.__setattr__(self, "plan_sha256", _digest(self.plan_sha256, field="plan_sha256"))
        if len(self.plan_request_ids) != REQUESTS_PER_IMAGE or len(set(self.plan_request_ids)) != REQUESTS_PER_IMAGE:
            raise ValueError("native receipt artifact requires exact K16 request ordering")
        if len(self.batch_receipts) != BATCHES_PER_IMAGE or any(type(item) is not NativeBatchReceipt for item in self.batch_receipts):
            raise ValueError("native receipt artifact requires four canonical native batches")

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": "human13_rp_crossover_native_receipts.v1", "plan_sha256": self.plan_sha256,
            "plan_request_ids": list(self.plan_request_ids), "batch_receipts": [item.to_dict() for item in self.batch_receipts]}

    @classmethod
    def from_dict(cls, value: object) -> "NativeReceiptsArtifact":
        item = _mapping(value, field="native receipt artifact", keys={"schema_version", "plan_sha256", "plan_request_ids", "batch_receipts"})
        if item["schema_version"] != "human13_rp_crossover_native_receipts.v1":
            raise ValueError("native receipt artifact schema version differs")
        if not isinstance(item["plan_request_ids"], list) or not isinstance(item["batch_receipts"], list):
            raise ValueError("native receipt artifact lists are missing")
        artifact = cls(plan_sha256=str(item["plan_sha256"]), plan_request_ids=tuple(item["plan_request_ids"]),
            batch_receipts=tuple(NativeBatchReceipt.from_dict(value) for value in item["batch_receipts"]))
        if json.loads(_canonical_bytes(artifact.to_dict())) != dict(item):
            raise ValueError("native receipt artifact is not canonical")
        return artifact

    @property
    def content_sha256(self) -> str:
        return _sha256(self.to_dict())


@dataclass(frozen=True)
class AcquisitionExecution:
    """Receipted K16 group plus immutable plan/native lineage retained for publication."""

    plan: AcquisitionGroupPlan
    plan_sha256: str
    plan_request_ids: tuple[str, ...]
    native_batch_receipts: tuple[NativeBatchReceipt, ...]
    group: AcquisitionGroup

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_request_ids", tuple(self.plan_request_ids))
        object.__setattr__(self, "native_batch_receipts", tuple(self.native_batch_receipts))
        _validate_plan(self.plan)
        object.__setattr__(self, "plan_sha256", _digest(self.plan_sha256, field="plan_sha256"))
        if self.plan_sha256 != self.plan.content_sha256:
            raise ValueError("execution plan SHA differs from its sealed plan")
        if len(self.plan_request_ids) != REQUESTS_PER_IMAGE or len(set(self.plan_request_ids)) != REQUESTS_PER_IMAGE:
            raise ValueError("execution requires the exact ordered K16 request identities")
        if self.plan_request_ids != tuple(item.request_id for item in self.plan.requests):
            raise ValueError("execution request ordering differs from its sealed plan")
        if len(self.native_batch_receipts) != BATCHES_PER_IMAGE or any(type(item) is not NativeBatchReceipt for item in self.native_batch_receipts):
            raise ValueError("execution requires four native batch receipts")
        if type(self.group) is not AcquisitionGroup:
            raise ValueError("execution requires a sealed acquisition group")
        _validate_receipts_against_plan(plan=self.plan, receipts=self.native_batch_receipts)
        expected = _group_from_native_receipts(plan=self.plan, receipts=self.native_batch_receipts)
        if expected.content_sha256 != self.group.content_sha256:
            raise ValueError("execution group differs from the canonical native group")

    @property
    def native_receipts_artifact(self) -> NativeReceiptsArtifact:
        return NativeReceiptsArtifact(plan_sha256=self.plan_sha256, plan_request_ids=self.plan_request_ids,
            batch_receipts=self.native_batch_receipts)


@dataclass(frozen=True)
class AdmittedPublication:
    """All five canonical publication artifacts after cross-artifact admission."""

    binding: PublicationBinding
    execution: AcquisitionExecution
    replayed_group: AcquisitionGroup
    parity_receipt: Any

    def __post_init__(self) -> None:
        _validate_admitted_publication(
            binding=self.binding,
            execution=self.execution,
            replayed_group=self.replayed_group,
            parity_receipt=self.parity_receipt,
        )


@dataclass(frozen=True)
class PackedRawLogits:
    """Ephemeral compact replay input; full vocabulary rows remain tensors."""

    request_ids: tuple[str, ...]
    token_indices: tuple[int, ...]
    logits: Any

    def __post_init__(self) -> None:
        import torch

        if not isinstance(self.logits, torch.Tensor) or self.logits.ndim != 2:
            raise ValueError("packed raw logits must be a two-dimensional tensor")
        if len(self.request_ids) != len(self.token_indices) or len(self.request_ids) != self.logits.shape[0]:
            raise ValueError("packed raw logits rows do not cover the declared causal positions")
        if not self.request_ids or any(not isinstance(value, str) or not value for value in self.request_ids):
            raise ValueError("packed raw logits require nonempty request identities")
        if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in self.token_indices):
            raise ValueError("packed raw logits token indexes must be nonnegative integers")
        if not bool(torch.isfinite(self.logits).all().item()):
            raise ValueError("packed raw logits must be finite")


@dataclass(frozen=True)
class PublicationBinding:
    plan_sha256: str
    plan_request_ids: tuple[str, ...]
    native_batch_receipt_sha256s: tuple[str, ...]
    native_receipts_filename: str
    native_receipts_sha256: str
    sampled_group_sha256: str
    replayed_group_sha256: str
    parity_receipt_sha256: str
    tolerance_sha256: str
    token_count: int

    def __post_init__(self) -> None:
        for field in ("plan_sha256", "native_receipts_sha256", "sampled_group_sha256", "replayed_group_sha256", "parity_receipt_sha256", "tolerance_sha256"):
            object.__setattr__(self, field, _digest(getattr(self, field), field=field))
        if len(self.plan_request_ids) != REQUESTS_PER_IMAGE or len(set(self.plan_request_ids)) != REQUESTS_PER_IMAGE:
            raise ValueError("publication binding requires exact K16 request ordering")
        if len(self.native_batch_receipt_sha256s) != BATCHES_PER_IMAGE:
            raise ValueError("publication binding requires four native batch receipts")
        if any(_digest(value, field="native batch receipt SHA-256") != value for value in self.native_batch_receipt_sha256s):
            raise ValueError("publication binding native receipt digest differs")
        if self.native_receipts_filename != "native-receipts.json":
            raise ValueError("publication binding native receipt filename differs")
        if isinstance(self.token_count, bool) or not isinstance(self.token_count, int) or self.token_count <= 0:
            raise ValueError("publication binding token count must be positive")

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": "human13_rp_crossover_publication_binding.v1", **asdict(self)}

    @classmethod
    def from_dict(cls, value: object) -> "PublicationBinding":
        item = _mapping(value, field="publication binding", keys={
            "schema_version", "plan_sha256", "plan_request_ids", "native_batch_receipt_sha256s",
            "native_receipts_filename", "native_receipts_sha256", "sampled_group_sha256", "replayed_group_sha256",
            "parity_receipt_sha256", "tolerance_sha256", "token_count",
        })
        payload: dict[str, Any] = dict(item)
        if payload["schema_version"] != "human13_rp_crossover_publication_binding.v1":
            raise ValueError("publication binding schema version differs")
        for field in ("plan_request_ids", "native_batch_receipt_sha256s"):
            if not isinstance(payload[field], list):
                raise ValueError(f"publication binding {field} must be a JSON list")
        return cls(plan_sha256=str(payload["plan_sha256"]), plan_request_ids=tuple(payload["plan_request_ids"]),
            native_batch_receipt_sha256s=tuple(payload["native_batch_receipt_sha256s"]), native_receipts_filename=str(payload["native_receipts_filename"]),
            native_receipts_sha256=str(payload["native_receipts_sha256"]), sampled_group_sha256=str(payload["sampled_group_sha256"]),
            replayed_group_sha256=str(payload["replayed_group_sha256"]), parity_receipt_sha256=str(payload["parity_receipt_sha256"]),
            tolerance_sha256=str(payload["tolerance_sha256"]), token_count=int(payload["token_count"]))


def _validate_admitted_publication(*, binding: object, execution: object, replayed_group: object,
        parity_receipt: object) -> None:
    """Admit one complete publication only when every artifact names the same group."""
    from scripts.research.human13_rp_policy import AcquisitionGroupParityReceipt

    if type(binding) is not PublicationBinding or type(execution) is not AcquisitionExecution:
        raise ValueError("admitted publication requires sealed binding and execution")
    if type(replayed_group) is not AcquisitionGroup or type(parity_receipt) is not AcquisitionGroupParityReceipt:
        raise ValueError("admitted publication requires exact replayed group and parity receipt")

    expected_request_ids = tuple(request.request_id for request in execution.plan.requests)
    if execution.plan_sha256 != execution.plan.content_sha256 or execution.plan_request_ids != expected_request_ids:
        raise ValueError("admitted publication execution plan identity differs from its canonical plan")
    canonical_sampled = _group_from_native_receipts(
        plan=execution.plan, receipts=execution.native_batch_receipts
    )
    if canonical_sampled.content_sha256 != execution.group.content_sha256:
        raise ValueError("admitted publication sampled group differs from canonical native receipts")
    native = execution.native_receipts_artifact
    if (
        native.plan_sha256,
        native.plan_request_ids,
        native.batch_receipts,
    ) != (
        execution.plan_sha256,
        execution.plan_request_ids,
        execution.native_batch_receipts,
    ):
        raise ValueError("admitted publication native receipt artifact differs from execution lineage")
    if (
        binding.plan_sha256,
        binding.plan_request_ids,
        binding.native_batch_receipt_sha256s,
        binding.native_receipts_sha256,
        binding.sampled_group_sha256,
    ) != (
        execution.plan_sha256,
        execution.plan_request_ids,
        tuple(item.content_sha256 for item in execution.native_batch_receipts),
        native.content_sha256,
        execution.group.content_sha256,
    ):
        raise ValueError("publication binding differs from the sealed execution/native lineage")
    if binding.replayed_group_sha256 != replayed_group.content_sha256:
        raise ValueError("publication binding replayed group differs from its replayed artifact")

    canonical_receipt = AcquisitionGroupParityReceipt.from_dict(parity_receipt.to_dict())
    if canonical_receipt != parity_receipt or canonical_receipt.content_sha256 != parity_receipt.content_sha256:
        raise ValueError("parity receipt canonical serialization or invariants differ")
    token_count = sum(len(item.generated_tokens) for item in execution.group.trajectories)
    tolerance_sha256 = ReplayTolerance().content_sha256
    if (
        binding.parity_receipt_sha256,
        binding.tolerance_sha256,
        binding.token_count,
    ) != (
        parity_receipt.content_sha256,
        tolerance_sha256,
        token_count,
    ):
        raise ValueError("publication binding differs from parity receipt parameters")
    if (
        parity_receipt.sampled_group_sha256,
        parity_receipt.replayed_group_sha256,
        parity_receipt.tolerance_sha256,
        parity_receipt.token_count,
    ) != (
        execution.group.content_sha256,
        replayed_group.content_sha256,
        tolerance_sha256,
        token_count,
    ):
        raise ValueError("parity receipt differs from the exact sampled or replayed group")
    if set(parity_receipt.request_ids) != set(execution.plan_request_ids):
        raise ValueError("parity receipt request identities differ from the exact K16 plan")


def _token_ids(value: Sequence[object] | tuple[int, ...], *, field: str, nonempty: bool) -> tuple[int, ...]:
    tokens = tuple(value)
    if (nonempty and not tokens) or any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in tokens):
        raise ValueError(f"{field} must contain {'nonempty ' if nonempty else ''}nonnegative token ids")
    return tuple(cast(int, token) for token in tokens)


def _scores(value: Sequence[object] | tuple[float, ...]) -> tuple[float, ...]:
    if any(isinstance(score, bool) or not isinstance(score, (int, float)) for score in value):
        raise ValueError("native output receipt processed log probabilities must be numeric")
    scores = tuple(float(cast(int | float, score)) for score in value)
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("native output receipt processed log probabilities must be finite")
    return scores


def _request_id(*, image_id: int, seed_group_id: str, seed: int) -> str:
    return f"human13:{image_id}:rp-crossover:{seed_group_id}:{seed}"


def _seed_group(seed_group_id: str) -> tuple[int, ...]:
    try:
        return _SEED_GROUPS[seed_group_id]
    except KeyError as exc:
        raise ValueError("seed group is not frozen for Human-13 RP crossover") from exc


def _validate_plan(plan: AcquisitionGroupPlan) -> None:
    if type(plan) is not AcquisitionGroupPlan:
        raise ValueError("acquisition plan must be sealed")
    if plan.image_id not in tuple(image_id for image_id, _ in EXPECTED_IMAGE_IDENTITIES):
        raise ValueError("acquisition image is outside canonical Human-13 panel")
    if plan.repetition_penalty not in (1.0, 1.10):
        raise ValueError("training repetition penalty must be exactly 1.0 or 1.10")
    seeds = _seed_group(plan.seed_group_id)
    if len(plan.batches) != BATCHES_PER_IMAGE or [batch.batch_index for batch in plan.batches] != list(range(BATCHES_PER_IMAGE)):
        raise ValueError("each acquisition image requires four ordered physical batches")
    for batch in plan.batches:
        if type(batch) is not AcquisitionBatch or batch.image_id != plan.image_id or batch.seed_group_id != plan.seed_group_id:
            raise ValueError("acquisition batch lineage differs from its sealed plan")
        if any(type(request) is not AcquisitionRequest for request in batch.requests):
            raise ValueError("acquisition plan requires sealed request records")
    requests = plan.requests
    if len(requests) != REQUESTS_PER_IMAGE or any(len(batch.requests) != REQUESTS_PER_BATCH for batch in plan.batches):
        raise ValueError("each image requires four batches of four requests")
    if tuple(request.seed for request in requests) != seeds or len({request.seed for request in requests}) != REQUESTS_PER_IMAGE:
        raise ValueError("acquisition seed ordering differs from its frozen group")
    if len({request.request_id for request in requests}) != REQUESTS_PER_IMAGE:
        raise ValueError("acquisition group contains duplicate request identities")
    expected_sampling = {**_SAMPLING_BASE, "repetition_penalty": plan.repetition_penalty}
    for index, request in enumerate(requests):
        if (request.image_id, request.seed_group_id, request.physical_batch_index, request.request_order_in_batch, request.request_id, dict(request.sampling)) != (
            plan.image_id, plan.seed_group_id, index // 4, index % 4,
            _request_id(image_id=plan.image_id, seed_group_id=plan.seed_group_id, seed=request.seed), expected_sampling,
        ):
            raise ValueError("acquisition request differs from the sealed batch-four policy")


def plan_acquisition_group(*, image_id: int, repetition_penalty: float, seed_group_id: str) -> AcquisitionGroupPlan:
    seeds = _seed_group(seed_group_id)
    policy = MappingProxyType({**_SAMPLING_BASE, "repetition_penalty": float(repetition_penalty)})
    plan = AcquisitionGroupPlan(
        image_id=image_id, seed_group_id=seed_group_id, repetition_penalty=float(repetition_penalty),
        batches=tuple(
            AcquisitionBatch(image_id=image_id, seed_group_id=seed_group_id, batch_index=batch_index, requests=tuple(
                AcquisitionRequest(image_id=image_id, seed_group_id=seed_group_id, seed=seed, physical_batch_index=batch_index,
                    request_order_in_batch=order, request_id=_request_id(image_id=image_id, seed_group_id=seed_group_id, seed=seed), sampling=policy)
                for order, seed in enumerate(seeds[batch_index * 4:(batch_index + 1) * 4])
            )) for batch_index in range(BATCHES_PER_IMAGE)
        ),
    )
    _validate_plan(plan)
    return plan


def plan_panel_acquisition(*, repetition_penalty: float, seed_group_id: str) -> tuple[AcquisitionGroupPlan, ...]:
    plans = tuple(plan_acquisition_group(image_id=image_id, repetition_penalty=repetition_penalty, seed_group_id=seed_group_id) for image_id, _ in EXPECTED_IMAGE_IDENTITIES)
    if len(plans) != 13:
        raise ValueError("canonical Human-13 panel must contain exactly thirteen images")
    return plans


def dry_run_plan() -> dict[str, object]:
    return {
        "schema_version": "human13_rp_crossover_acquisition.v2", "status": "plan_only",
        "panel_image_count": 13, "group_physical_batch_count": BATCHES_PER_IMAGE,
        "group_request_count": REQUESTS_PER_IMAGE, "panel_physical_batch_count": 52,
        "panel_request_count": 208,
        "seed_groups": {key: list(value) for key, value in _SEED_GROUPS.items()},
        "sampling": {**_SAMPLING_BASE, "repetition_penalty": "explicit: 1.0 or 1.10"},
        "actions": {"model_imports": 0, "model_loads": 0, "engine_opens": 0, "gpu_allocations": 0, "artifact_writes": 0},
    }


def _request_repetition_penalty(request: AcquisitionRequest) -> float:
    value = request.sampling["repetition_penalty"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("request repetition penalty must be numeric")
    return float(value)


def expected_native_sampling_evidence(request: AcquisitionRequest) -> dict[str, object]:
    return {**_NATIVE_SAMPLING_BASE, "seed": request.seed, "repetition_penalty": _request_repetition_penalty(request)}


def native_sampling_evidence(params: object) -> dict[str, object]:
    fields = ("n", "seed", "temperature", "top_p", "top_k", "repetition_penalty", "max_tokens", "logprobs", "stop_token_ids", "ignore_eos")
    values = {field: getattr(params, field) for field in fields}
    if not isinstance(values["stop_token_ids"], Sequence) or isinstance(values["stop_token_ids"], (str, bytes)):
        raise ValueError("native SamplingParams stop token ids are missing")
    return {
        "n": values["n"], "seed": values["seed"], "temperature": values["temperature"], "top_p": values["top_p"],
        "top_k": values["top_k"], "repetition_penalty": values["repetition_penalty"], "max_new_tokens": values["max_tokens"],
        "logprobs": values["logprobs"], "stop_token_ids": tuple(values["stop_token_ids"]), "ignore_eos": values["ignore_eos"],
    }


def vllm_sampling_params(request: AcquisitionRequest) -> Any:
    """Import vLLM only for an explicit injected live execution."""
    from vllm import SamplingParams
    return SamplingParams(n=1, seed=request.seed, temperature=0.4, top_p=1.0, top_k=0,
        repetition_penalty=_request_repetition_penalty(request), max_tokens=MAX_NEW_TOKENS, logprobs=1,
        stop_token_ids=[NATURAL_STOP_TOKEN_ID], ignore_eos=False, detokenize=True,
        skip_special_tokens=False, spaces_between_special_tokens=True)


def _validate_receipt_batch(*, plan_batch: AcquisitionBatch, receipt: NativeBatchReceipt) -> None:
    if type(receipt) is not NativeBatchReceipt:
        raise ValueError("executor must return an immutable NativeBatchReceipt")
    for request, native_request, output in zip(plan_batch.requests, receipt.requests, receipt.outputs, strict=True):
        if native_request.request_id != request.request_id:
            raise ValueError("native request identity differs from the sealed plan")
        if output.request_id != request.request_id:
            raise ValueError("native output ordering differs from the sealed plan")
        if native_request.seed != request.seed or output.seed != request.seed:
            raise ValueError("native seed differs from the sealed plan")
        if native_request.physical_batch_index != request.physical_batch_index or output.physical_batch_index != request.physical_batch_index:
            raise ValueError("native physical batch differs from the sealed plan")
        if native_request.request_order_in_batch != request.request_order_in_batch or output.request_order_in_batch != request.request_order_in_batch:
            raise ValueError("native request order differs from the sealed plan")
        if dict(native_request.sampling_params) != expected_native_sampling_evidence(request):
            raise ValueError("native sampling parameters differ from the sealed plan")
        if output.native_request_receipt_sha256 != native_request.content_sha256:
            raise ValueError("native output lineage differs from its actual request receipt")
        if output.model_id != native_request.model_id:
            raise ValueError("native output model identity differs from its actual request receipt")
        if token_ids_sha256(output.prompt_token_ids) != native_request.prompt_token_ids_sha256:
            raise ValueError("native output prompt differs from its actual request receipt")


def _validate_native_batch(*, plan_batch: AcquisitionBatch, params: tuple[Any, ...], receipt: NativeBatchReceipt) -> None:
    if len(params) != REQUESTS_PER_BATCH:
        raise ValueError("native batch parameters must contain exactly four requests")
    _validate_receipt_batch(plan_batch=plan_batch, receipt=receipt)
    for request, parameter in zip(plan_batch.requests, params, strict=True):
        if native_sampling_evidence(parameter) != expected_native_sampling_evidence(request):
            raise ValueError("native SamplingParams differ from the sealed plan")


def _validate_execution_surface(receipts: Sequence[NativeBatchReceipt]) -> None:
    requests = tuple(item for receipt in receipts for item in receipt.requests)
    outputs = tuple(item for receipt in receipts for item in receipt.outputs)
    if len({item.prompt_token_ids_sha256 for item in requests}) != 1:
        raise ValueError("native prompt identity differs within one acquisition group")
    if len({item.model_identity_sha256 for item in requests}) != 1 or len({item.session_identity_sha256 for item in requests}) != 1:
        raise ValueError("native model or session identity differs within one acquisition group")
    if len({(item.source_sha256, item.manifest_sha256, item.model_id, item.tokenizer_id, item.processor_id, item.sampler_backend_id) for item in outputs}) != 1:
        raise ValueError("native output model or processor identity differs within one acquisition group")


def _validate_receipts_against_plan(*, plan: AcquisitionGroupPlan, receipts: Sequence[NativeBatchReceipt]) -> None:
    if len(receipts) != BATCHES_PER_IMAGE:
        raise ValueError("execution requires four native batch receipts")
    for batch, receipt in zip(plan.batches, receipts, strict=True):
        _validate_receipt_batch(plan_batch=batch, receipt=receipt)
    _validate_execution_surface(receipts)


def _trajectory_from_native(*, request: NativeRequestReceipt, output: NativeOutputReceipt, repetition_penalty: float) -> CompleteTrajectoryEvidence:
    identity = ArtifactIdentity(source_sha256=output.source_sha256, manifest_sha256=output.manifest_sha256,
        request_id=output.request_id, model_id=output.model_id, tokenizer_id=output.tokenizer_id,
        processor_id=output.processor_id, prompt_token_ids=output.prompt_token_ids, generated_token_ids=output.generated_token_ids)
    contract = PolicyContract(identity=identity, repetition_penalty=repetition_penalty, temperature=0.4,
        natural_stop_token_id=NATURAL_STOP_TOKEN_ID, max_new_tokens=MAX_NEW_TOKENS, processor_order=PROCESSOR_ORDER,
        sampler_backend_id=output.sampler_backend_id, top_p=1.0, top_k=None, n=1)
    tokens = tuple(GeneratedTokenEvidence(identity=identity, policy_contract_sha256=contract.content_sha256,
        token_index=index, history_token_ids=(*output.prompt_token_ids, *output.generated_token_ids[:index]),
        chosen_token_id=token, processed_logprob=score)
        for index, (token, score) in enumerate(zip(output.generated_token_ids, output.processed_logprobs, strict=True)))
    return CompleteTrajectoryEvidence(identity=identity, policy_contract=contract, generated_tokens=tokens, terminal_kind=output.terminal_kind)


def _group_from_native_receipts(*, plan: AcquisitionGroupPlan, receipts: Sequence[NativeBatchReceipt]) -> AcquisitionGroup:
    trajectories = tuple(_trajectory_from_native(request=request, output=output, repetition_penalty=plan.repetition_penalty)
        for receipt in receipts for request, output in zip(receipt.requests, receipt.outputs, strict=True))
    return AcquisitionGroup(identity=trajectories[0].identity, policy_contract=trajectories[0].policy_contract,
        trajectories=trajectories, seed_group_id=plan.seed_group_id)


def execute_acquisition_group(*, plan: AcquisitionGroupPlan, execute_batch: Callable[[AcquisitionBatch, tuple[Any, ...]], NativeBatchReceipt]) -> AcquisitionExecution:
    """Validate actual native receipts in order; no caller freshness string is admitted."""
    _validate_plan(plan)
    receipts: list[NativeBatchReceipt] = []
    for batch in plan.batches:
        params = tuple(vllm_sampling_params(request) for request in batch.requests)
        receipt = execute_batch(batch, params)
        _validate_native_batch(plan_batch=batch, params=params, receipt=receipt)
        receipts.append(receipt)
    _validate_receipts_against_plan(plan=plan, receipts=receipts)
    group = _group_from_native_receipts(plan=plan, receipts=receipts)
    return AcquisitionExecution(plan=plan, plan_sha256=plan.content_sha256, plan_request_ids=tuple(item.request_id for item in plan.requests),
        native_batch_receipts=tuple(receipts), group=group)


def replay_acquisition_group(*, sampled: AcquisitionGroup, packed: PackedRawLogits) -> tuple[AcquisitionGroup, Any]:
    from scripts.research.human13_rp_policy import processed_policy_logprobs, validate_acquisition_group_replay
    expected = tuple((trajectory.identity.request_id, token.token_index) for trajectory in sampled.trajectories for token in trajectory.generated_tokens)
    if tuple(zip(packed.request_ids, packed.token_indices, strict=True)) != expected:
        raise ValueError("packed raw logits must cover generated causal positions in sealed request order")
    rows = iter(packed.logits.unbind(0))
    replayed_trajectories = []
    for trajectory in sampled.trajectories:
        tokens = tuple(GeneratedTokenEvidence(identity=trajectory.identity, policy_contract_sha256=trajectory.policy_contract.content_sha256,
            token_index=token.token_index, history_token_ids=token.history_token_ids, chosen_token_id=token.chosen_token_id,
            processed_logprob=float(processed_policy_logprobs(next(rows), token, trajectory.policy_contract)[token.chosen_token_id].item()))
            for token in trajectory.generated_tokens)
        replayed_trajectories.append(CompleteTrajectoryEvidence(identity=trajectory.identity, policy_contract=trajectory.policy_contract, generated_tokens=tokens, terminal_kind=trajectory.terminal_kind))
    replayed = AcquisitionGroup(identity=sampled.identity, policy_contract=sampled.policy_contract,
        trajectories=tuple(replayed_trajectories), seed_group_id=sampled.seed_group_id)
    return replayed, validate_acquisition_group_replay(sampled, replayed, ReplayTolerance())


def _publication_binding(*, execution: AcquisitionExecution, replayed: AcquisitionGroup, replay_receipt: object) -> PublicationBinding:
    from scripts.research.human13_rp_policy import AcquisitionGroupParityReceipt
    if type(replay_receipt) is not AcquisitionGroupParityReceipt:
        raise ValueError("publication requires exact AcquisitionGroupParityReceipt")
    sampled = execution.group
    token_count = sum(len(item.generated_tokens) for item in sampled.trajectories)
    native_preimage = execution.native_receipts_artifact
    return PublicationBinding(plan_sha256=execution.plan_sha256, plan_request_ids=execution.plan_request_ids,
        native_batch_receipt_sha256s=tuple(item.content_sha256 for item in execution.native_batch_receipts),
        native_receipts_filename="native-receipts.json", native_receipts_sha256=native_preimage.content_sha256,
        sampled_group_sha256=sampled.content_sha256, replayed_group_sha256=replayed.content_sha256,
        parity_receipt_sha256=replay_receipt.content_sha256, tolerance_sha256=replay_receipt.tolerance_sha256,
        token_count=token_count)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_acquisition_group(*, output_root: str | Path, execution: AcquisitionExecution, replayed: AcquisitionGroup,
        replay_receipt: object, write_bytes: Callable[[Path, bytes], None] | None = None) -> Path:
    """Atomically publish only a typed, canonical, plan-bound parity result."""
    if type(execution) is not AcquisitionExecution or type(replayed) is not AcquisitionGroup:
        raise ValueError("publication requires sealed execution and replayed acquisition groups")
    binding = _publication_binding(execution=execution, replayed=replayed, replay_receipt=replay_receipt)
    admitted = AdmittedPublication(
        binding=binding,
        execution=execution,
        replayed_group=replayed,
        parity_receipt=replay_receipt,
    )
    native_preimage = execution.native_receipts_artifact
    target = Path(output_root).resolve()
    staging = target.with_name(f".{target.name}.staging")
    if target.exists() or staging.exists():
        raise FileExistsError("refusing to overwrite an acquisition artifact")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    writer = write_bytes or (lambda path, payload: path.write_bytes(payload))
    payloads = (
        (staging / "acquisition-group.json", _canonical_bytes(execution.group.to_dict())),
        (staging / "native-receipts.json", _canonical_bytes(native_preimage.to_dict())),
        (staging / "replayed-group.json", _canonical_bytes(admitted.replayed_group.to_dict())),
        (staging / "replay-receipt.json", _canonical_bytes(admitted.parity_receipt.to_dict())),
        (staging / "publication-binding.json", _canonical_bytes(binding.to_dict())),
    )
    try:
        for path, payload in payloads:
            writer(path, payload)
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        _fsync_directory(staging)
        os.replace(staging, target)
        _fsync_directory(target.parent)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return target


def _load_json(path: Path, *, field: str) -> Mapping[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {field}: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field} is unreadable") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _canonical_record(value: object, record: Mapping[str, object], *, field: str) -> None:
    if json.loads(_canonical_bytes(value)) != dict(record):
        raise ValueError(f"{field} is not canonical")


def load_published_acquisition(output_root: str | Path, *, plan: AcquisitionGroupPlan) -> AdmittedPublication:
    """Reload and cross-admit every canonical artifact in a published group."""
    from scripts.research.human13_rp_policy import AcquisitionGroupParityReceipt

    root = Path(output_root).resolve()
    binding_record = _load_json(root / "publication-binding.json", field="publication binding")
    binding = PublicationBinding.from_dict(binding_record)
    _canonical_record(binding.to_dict(), binding_record, field="publication binding")
    native_record = _load_json(root / binding.native_receipts_filename, field="native receipt artifact")
    native = NativeReceiptsArtifact.from_dict(native_record)
    sampled_record = _load_json(root / "acquisition-group.json", field="sampled acquisition group")
    try:
        sampled = AcquisitionGroup.from_dict(sampled_record)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("sampled acquisition group is invalid") from exc
    _canonical_record(sampled.to_dict(), sampled_record, field="sampled acquisition group")
    execution = AcquisitionExecution(plan=plan, plan_sha256=native.plan_sha256, plan_request_ids=native.plan_request_ids,
        native_batch_receipts=native.batch_receipts, group=sampled)
    replayed_record = _load_json(root / "replayed-group.json", field="replayed acquisition group")
    try:
        replayed = AcquisitionGroup.from_dict(replayed_record)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("replayed acquisition group is invalid") from exc
    _canonical_record(replayed.to_dict(), replayed_record, field="replayed acquisition group")
    parity_record = _load_json(root / "replay-receipt.json", field="parity receipt")
    try:
        parity_receipt = AcquisitionGroupParityReceipt.from_dict(parity_record)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("parity receipt is invalid") from exc
    _canonical_record(parity_receipt.to_dict(), parity_record, field="parity receipt")
    return AdmittedPublication(binding=binding, execution=execution, replayed_group=replayed, parity_receipt=parity_receipt)


__all__ = [
    "AcquisitionBatch", "AcquisitionExecution", "AcquisitionGroupPlan", "AcquisitionRequest", "AdmittedPublication", "MATRIX_SEED_GROUPS",
    "NativeBatchReceipt", "NativeOutputReceipt", "NativeReceiptsArtifact", "NativeRequestReceipt", "PackedRawLogits", "PublicationBinding",
    "QUALIFICATION_SEEDS", "dry_run_plan", "execute_acquisition_group", "expected_native_sampling_evidence",
    "native_sampling_evidence", "plan_acquisition_group", "plan_panel_acquisition", "publish_acquisition_group",
    "load_published_acquisition", "replay_acquisition_group", "token_ids_sha256", "vllm_sampling_params",
]
