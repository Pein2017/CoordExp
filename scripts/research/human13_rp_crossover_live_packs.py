#!/usr/bin/env python3
"""Live packed materialization of admitted Human-13 RP-crossover trajectories.

This module is the join between Task-2 admitted sampled trajectories and the
existing no-padding Qwen FA2/MRoPE compact-logit forward.  It owns no model,
optimizer, checkpoint, matcher, or trainer: it plans one isolated causal
segment per sampled trajectory, reuses the experiment's pack planner, requests
only the causal rows that precede each chosen token, and emits

* the exact ephemeral ``PackedRawLogits`` consumed by
  ``collect_human13_rp_crossover.replay_acquisition_group``;
* gradient-carrying chosen-token processed policy log probabilities in the
  form ``human13_trajectory_credit.trajectory_score_function_numerator``
  already accepts; and
* the packed compiler rows ``human13_greedy_compiler.bind_packed_compiler_logits``
  already accepts.

Importing this module is runtime-free.  Torch, the pack planner, and the Qwen
forward seam are imported lazily inside the functions that need them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any


SCHEMA_VERSION = "human13_rp_crossover_live_packs.v1"
SEGMENT_ROLE = "h1_independent"
SEGMENT_PREFIX = "human13rp"
FA2_BRANCH_PROOF_POLICY = "human13_rp_crossover_live_packs"
PROCESSED_TRANSFORM_TOLERANCE_NATS = 1e-5


class LivePackContractError(ValueError):
    """Raised when live pack materialization cannot be bound fail-closed."""


def _sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Typed plan records
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CompilerSegmentRequest:
    """One Source-boundary compiler segment co-planned with the trajectories."""

    site_id: str
    image_id: int
    segment_id: str
    token_ids: tuple[int, ...]
    local_causal_position: int

    def __post_init__(self) -> None:
        for field in ("site_id", "segment_id"):
            if not isinstance(getattr(self, field), str) or not getattr(self, field):
                raise LivePackContractError(
                    f"compiler segment {field} must be nonempty"
                )
        if isinstance(self.image_id, bool) or not isinstance(self.image_id, int):
            raise LivePackContractError("compiler segment image_id must be an integer")
        token_ids = tuple(self.token_ids)
        if not token_ids or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in token_ids
        ):
            raise LivePackContractError(
                "compiler segment requires nonnegative integer token ids"
            )
        object.__setattr__(self, "token_ids", token_ids)
        if (
            isinstance(self.local_causal_position, bool)
            or not isinstance(self.local_causal_position, int)
            or not 0 <= self.local_causal_position < len(token_ids)
        ):
            raise LivePackContractError(
                "compiler segment causal position is outside its own segment"
            )


@dataclass(frozen=True)
class TrajectoryRowBinding:
    """The local causal mapping for one sampled trajectory."""

    request_id: str
    segment_id: str
    image_id: int
    prompt_token_count: int
    generated_token_count: int
    prompt_token_sha256: str
    generated_token_sha256: str
    local_causal_positions: tuple[int, ...]
    chosen_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class PhysicalRowBinding:
    """One requested compact row bound to its physical packed position."""

    segment_id: str
    pack_index: int
    local_causal_position: int
    packed_causal_position: int
    request_id: str | None = None
    token_index: int | None = None
    site_id: str | None = None


@dataclass(frozen=True)
class PackForwardRequest:
    """The exact compact-logit request and no-padding evidence for one pack."""

    pack_index: int
    pack_length: int
    segment_ids: tuple[str, ...]
    segment_boundaries: tuple[int, ...]
    cu_seq_lens: tuple[int, ...]
    mrope_reset_points: tuple[int, ...]
    max_segment_length: int
    compact_positions: tuple[int, ...]


@dataclass(frozen=True)
class LivePackPlan:
    """A runtime-free physical mapping for one image's K trajectories."""

    image_id: int
    seed_group_id: str
    repetition_penalty: float
    acquisition_group_sha256: str
    plan_sha256: str
    prompt_token_count: int
    prompt_token_sha256: str
    request_order: tuple[str, ...]
    trajectory_bindings: tuple[TrajectoryRowBinding, ...]
    compiler_requests: tuple[CompilerSegmentRequest, ...]
    row_bindings: tuple[PhysicalRowBinding, ...]
    compiler_row_bindings: tuple[PhysicalRowBinding, ...]
    pack_requests: tuple[PackForwardRequest, ...]
    scored_token_indices: tuple[int, ...] | None
    publication: Any
    packed_plan: Any

    @property
    def requested_row_count(self) -> int:
        return len(self.row_bindings)


@dataclass(frozen=True)
class CompilerPackedRow:
    """Exactly the arguments ``bind_packed_compiler_logits`` requires."""

    site_id: str
    pack_index: int
    logits_position_ids: tuple[int, ...]
    raw_logits: Any


@dataclass(frozen=True)
class LivePackReceipt:
    """Immutable measured evidence for one image's packed materialization."""

    schema_version: str
    image_id: int
    seed_group_id: str
    repetition_penalty: float
    acquisition_group_sha256: str
    plan_sha256: str
    prompt_token_sha256: str
    pack_count: int
    forward_count: int
    packed_token_count: int
    logical_token_count: int
    requested_row_count: int
    compiler_row_count: int
    vocab_size: int
    sealed_row_bytes: int
    compact_row_bytes: int
    max_pack_length: int
    segment_count: int
    pack_requests: tuple[PackForwardRequest, ...]
    row_bindings: tuple[PhysicalRowBinding, ...]
    compiler_row_bindings: tuple[PhysicalRowBinding, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["pack_requests"] = [asdict(item) for item in self.pack_requests]
        payload["row_bindings"] = [asdict(item) for item in self.row_bindings]
        payload["compiler_row_bindings"] = [
            asdict(item) for item in self.compiler_row_bindings
        ]
        return {**payload, "content_sha256": _sha256(payload)}

    @property
    def content_sha256(self) -> str:
        payload = self.to_dict()
        payload.pop("content_sha256")
        return _sha256(payload)


class MaterializedImagePacks:
    """One image's ephemeral tensor-backed materialization.

    The full-vocabulary rows live only until :meth:`release`; nothing here is
    converted to Python floats and no panel-wide ``[K, L, V]`` object exists.
    """

    def __init__(
        self,
        *,
        plan: LivePackPlan,
        packed_raw_logits: Any,
        policy_logprobs: Mapping[str, Any],
        compiler_rows: Mapping[str, CompilerPackedRow],
        receipt: LivePackReceipt,
    ) -> None:
        self._plan = plan
        self._packed_raw_logits = packed_raw_logits
        self._policy_logprobs = dict(policy_logprobs)
        self._compiler_rows = dict(compiler_rows)
        self._receipt = receipt
        self._released = False

    def _live(self, value: Any) -> Any:
        if self._released:
            raise LivePackContractError(
                "materialized live packs were released; tensors are not retained"
            )
        return value

    @property
    def plan(self) -> LivePackPlan:
        return self._plan

    @property
    def receipt(self) -> LivePackReceipt:
        return self._receipt

    @property
    def released(self) -> bool:
        return self._released

    @property
    def packed_raw_logits(self) -> Any:
        return self._live(self._packed_raw_logits)

    @property
    def policy_logprobs(self) -> Mapping[str, Any]:
        return self._live(self._policy_logprobs)

    @property
    def compiler_rows(self) -> Mapping[str, CompilerPackedRow]:
        return self._live(self._compiler_rows)

    @property
    def scored_token_indices(self) -> tuple[int, ...] | None:
        return self._plan.scored_token_indices

    def release(self) -> None:
        """Drop every tensor reference; the receipt stays available."""

        self._packed_raw_logits = None
        self._policy_logprobs = {}
        self._compiler_rows = {}
        self._released = True


@dataclass(frozen=True)
class StreamingObjectiveStep:
    """One image/pack's unnormalized live objective and graph release owner."""

    image_id: int
    trajectory_numerator: Any
    compiler_numerator: Any | None
    release: Callable[[], None]


@dataclass(frozen=True)
class IncrementalBackwardReceipt:
    """Exact denominator and graph-lifetime evidence for streamed backward."""

    image_ids: tuple[int, ...]
    trajectory_denominator: int
    compiler_image_denominator: int | None
    backward_count: int
    released_graph_count: int


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------


def _skeleton_prompt(skeleton: Any, *, image_id: int) -> tuple[int, ...]:
    count = getattr(skeleton, "prompt_token_count", None)
    input_ids = getattr(skeleton, "input_ids", None)
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        or not isinstance(input_ids, tuple)
        or len(input_ids) < count
    ):
        raise LivePackContractError(
            f"image {image_id} requires a canonical processor skeleton"
        )
    prompt = tuple(int(token) for token in input_ids[:count])
    if any(token < 0 for token in prompt):
        raise LivePackContractError(f"image {image_id} skeleton prompt has a bad token")
    return prompt


def _pack_request(pack: Any) -> tuple[PackForwardRequest, dict[str, int]]:
    """Bind one physical pack's no-padding FA2/MRoPE evidence."""

    segments = tuple(pack.pack.segments)
    if not segments:
        raise LivePackContractError("a physical pack must carry at least one segment")
    boundaries = [0]
    starts: dict[str, int] = {}
    for segment in segments:
        if segment.start != boundaries[-1]:
            raise LivePackContractError(
                "packed segments must tile the pack without padding"
            )
        if segment.example_id in starts:
            raise LivePackContractError("packed segment identities must be unique")
        starts[segment.example_id] = int(segment.start)
        boundaries.append(int(segment.end))
    expected = tuple(boundaries)
    if int(pack.pack.length) != expected[-1]:
        raise LivePackContractError("packed length differs from its segment tiling")
    if tuple(int(item) for item in pack.fa2_varlen_plan.segment_boundaries) != expected:
        raise LivePackContractError(
            "FA2 cu_seqlens differ from the packed segment boundaries"
        )
    if tuple(int(item) for item in pack.position_inputs.segment_boundaries) != expected:
        raise LivePackContractError(
            "MRoPE segment boundaries differ from the packed segment boundaries"
        )
    if int(pack.position_inputs.pack_index) != int(pack.pack.pack_index):
        raise LivePackContractError("MRoPE position inputs name a different pack")
    reset_points = tuple(int(segment.start) for segment in segments)
    if tuple(int(item) for item in pack.position_inputs.reset_points) != reset_points:
        raise LivePackContractError("MRoPE resets differ from the segment starts")
    request = PackForwardRequest(
        pack_index=int(pack.pack.pack_index),
        pack_length=int(pack.pack.length),
        segment_ids=tuple(segment.example_id for segment in segments),
        segment_boundaries=expected,
        cu_seq_lens=expected,
        mrope_reset_points=reset_points,
        max_segment_length=int(pack.position_inputs.max_segment_length),
        compact_positions=(),
    )
    return request, starts


def _scored_token_indices(
    ledger: Any, plan_image_id: int, bindings: Sequence[TrajectoryRowBinding]
) -> tuple[int, ...]:
    from scripts.research.human13_trajectory_credit import TrajectoryCreditLedger

    if type(ledger) is not TrajectoryCreditLedger:
        raise LivePackContractError(
            "credit ledger must be an exact TrajectoryCreditLedger"
        )
    counts = [
        sum(
            1
            for trajectory in image.trajectories
            for row in trajectory.rows
            for token in row.tokens
            if token.scored
        )
        for image in ledger.images
    ]
    matches = [
        index
        for index, image in enumerate(ledger.images)
        if image.image_id == plan_image_id
    ]
    if len(matches) != 1:
        raise LivePackContractError("credit ledger does not own this exact image once")
    index = matches[0]
    image = ledger.images[index]
    by_request = {binding.request_id: binding for binding in bindings}
    if tuple(item.request_id for item in image.trajectories) != tuple(
        binding.request_id for binding in bindings
    ):
        raise LivePackContractError(
            "credit ledger request order differs from the sealed acquisition order"
        )
    for trajectory in image.trajectories:
        binding = by_request[trajectory.request_id]
        if trajectory.token_count != binding.generated_token_count:
            raise LivePackContractError(
                "credit ledger token count differs from the sealed generated history"
            )
        for row in trajectory.rows:
            for token in row.tokens:
                if not 0 <= token.token_index < binding.generated_token_count:
                    raise LivePackContractError(
                        "credit ledger names a token outside the sealed history"
                    )
    start = sum(counts[:index])
    return tuple(range(start, start + counts[index]))


def _bind_compiler_requests(
    requests: Sequence[CompilerSegmentRequest],
    *,
    image_id: int,
    prompt: tuple[int, ...],
    repetition_penalty: float,
    ledger: Any,
) -> tuple[CompilerSegmentRequest, ...]:
    from scripts.research.collect_human13_rp_crossover import token_ids_sha256

    checked = tuple(requests)
    for request in checked:
        if type(request) is not CompilerSegmentRequest:
            raise LivePackContractError(
                "compiler segments must be typed CompilerSegmentRequest records"
            )
        if request.image_id != image_id:
            raise LivePackContractError("compiler segment names a different image")
        if request.token_ids[: len(prompt)] != prompt:
            raise LivePackContractError(
                "compiler segment prompt differs from the canonical image prompt"
            )
    if len({item.site_id for item in checked}) != len(checked):
        raise LivePackContractError("compiler site identities must be unique")
    if len({item.segment_id for item in checked}) != len(checked):
        raise LivePackContractError("compiler segment identities must be unique")
    if ledger is None:
        return checked

    from scripts.research.human13_greedy_compiler import CompilerLedger

    if type(ledger) is not CompilerLedger:
        raise LivePackContractError("compiler ledger must be an exact CompilerLedger")
    if ledger.repetition_penalty != repetition_penalty:
        raise LivePackContractError(
            "compiler ledger RP differs from this training acquisition contract"
        )
    sites = {
        image.site.site_id: image.site
        for image in ledger.images
        if image.site is not None
    }
    for request in checked:
        site = sites.get(request.site_id)
        if site is None or site.image_id != image_id:
            raise LivePackContractError(
                "compiler segment is absent from the admitted compiler ledger"
            )
        if (
            site.packed_segment_id != request.segment_id
            or site.local_causal_position != request.local_causal_position
            or site.prompt_token_count != len(prompt)
        ):
            raise LivePackContractError(
                "compiler segment mapping differs from its admitted ledger site"
            )
        prefix_end = site.prompt_token_count + site.source_prefix_token_count
        if (
            len(request.token_ids) < prefix_end
            or token_ids_sha256(request.token_ids[: site.prompt_token_count])
            != site.prompt_token_sha256
            or token_ids_sha256(request.token_ids[site.prompt_token_count : prefix_end])
            != site.source_prefix_token_sha256
        ):
            raise LivePackContractError(
                "compiler segment tokens differ from the admitted Source boundary"
            )
    return checked


def plan_live_packs(
    *,
    publication: Any,
    skeleton: Any,
    compiler_segments: Sequence[CompilerSegmentRequest] = (),
    credit_ledger: Any = None,
    compiler_ledger: Any = None,
    global_max_length: int | None = None,
) -> LivePackPlan:
    """Bind one image's admitted trajectories to one physical packed mapping."""

    from scripts.research.collect_human13_rp_crossover import (
        AdmittedPublication,
        token_ids_sha256,
    )
    from scripts.research.human13_live_census import _clone_skeleton
    from scripts.research.run_human13_k_union_overfit import (
        GLOBAL_MAX_LENGTH,
        LogicalPanelSegment,
        plan_panel_packs,
    )

    if type(publication) is not AdmittedPublication:
        raise LivePackContractError(
            "live packs require one exact Task2 AdmittedPublication"
        )
    execution = publication.execution
    group = execution.group
    image_id = int(execution.plan.image_id)
    prompt = _skeleton_prompt(skeleton, image_id=image_id)
    request_order = tuple(item.identity.request_id for item in group.trajectories)
    if request_order != execution.plan_request_ids:
        raise LivePackContractError(
            "acquisition group order differs from the sealed plan request order"
        )

    bindings: list[TrajectoryRowBinding] = []
    segments: list[Any] = []
    for trajectory in group.trajectories:
        identity = trajectory.identity
        if identity.prompt_token_ids != prompt:
            raise LivePackContractError(
                "sealed trajectory prompt differs from the canonical image prompt"
            )
        generated = identity.generated_token_ids
        segment_id = f"{SEGMENT_PREFIX}:{image_id}:{identity.request_id}"
        segments.append(
            LogicalPanelSegment(
                segment_id=segment_id,
                image_id=image_id,
                role=SEGMENT_ROLE,
                encoded_example=_clone_skeleton(
                    skeleton,
                    segment_id=segment_id,
                    image_id=image_id,
                    input_ids=(*prompt, *generated),
                ),
            )
        )
        bindings.append(
            TrajectoryRowBinding(
                request_id=identity.request_id,
                segment_id=segment_id,
                image_id=image_id,
                prompt_token_count=len(prompt),
                generated_token_count=len(generated),
                prompt_token_sha256=token_ids_sha256(prompt),
                generated_token_sha256=token_ids_sha256(generated),
                # the row that predicts token ``t`` is the one before it
                local_causal_positions=tuple(
                    len(prompt) - 1 + index for index in range(len(generated))
                ),
                chosen_token_ids=generated,
            )
        )

    compiler_requests = _bind_compiler_requests(
        compiler_segments,
        image_id=image_id,
        prompt=prompt,
        repetition_penalty=float(execution.plan.repetition_penalty),
        ledger=compiler_ledger,
    )
    known = {binding.segment_id for binding in bindings}
    for request in compiler_requests:
        if request.segment_id in known:
            raise LivePackContractError(
                "compiler segment identity collides with a trajectory segment"
            )
        segments.append(
            LogicalPanelSegment(
                segment_id=request.segment_id,
                image_id=image_id,
                role=SEGMENT_ROLE,
                encoded_example=_clone_skeleton(
                    skeleton,
                    segment_id=request.segment_id,
                    image_id=image_id,
                    input_ids=request.token_ids,
                ),
            )
        )

    limit = GLOBAL_MAX_LENGTH if global_max_length is None else int(global_max_length)
    try:
        packed_plan = plan_panel_packs(segments, global_max_length=limit)
    except (ValueError, AssertionError) as error:
        raise LivePackContractError(f"live pack planning failed: {error}") from error

    requests_by_pack: dict[int, PackForwardRequest] = {}
    starts: dict[str, tuple[int, int]] = {}
    for pack in packed_plan.packs:
        request, pack_starts = _pack_request(pack)
        requests_by_pack[request.pack_index] = request
        for segment_id, start in pack_starts.items():
            starts[segment_id] = (request.pack_index, start)

    row_bindings: list[PhysicalRowBinding] = []
    for binding in bindings:
        pack_index, start = starts[binding.segment_id]
        for token_index, local in enumerate(binding.local_causal_positions):
            row_bindings.append(
                PhysicalRowBinding(
                    segment_id=binding.segment_id,
                    pack_index=pack_index,
                    local_causal_position=local,
                    packed_causal_position=start + local,
                    request_id=binding.request_id,
                    token_index=token_index,
                )
            )
    compiler_row_bindings = tuple(
        PhysicalRowBinding(
            segment_id=request.segment_id,
            pack_index=starts[request.segment_id][0],
            local_causal_position=request.local_causal_position,
            packed_causal_position=starts[request.segment_id][1]
            + request.local_causal_position,
            site_id=request.site_id,
        )
        for request in compiler_requests
    )

    physical = [
        (row.pack_index, row.packed_causal_position)
        for row in (*row_bindings, *compiler_row_bindings)
    ]
    if len(set(physical)) != len(
        {
            (
                row.pack_index,
                row.packed_causal_position,
                row.segment_id,
                row.token_index,
                row.site_id,
            )
            for row in (*row_bindings, *compiler_row_bindings)
        }
    ):
        raise LivePackContractError("packed causal positions collide across segments")
    by_pack: dict[int, set[int]] = {index: set() for index in requests_by_pack}
    for pack_index, position in physical:
        if not 0 <= position < requests_by_pack[pack_index].pack_length:
            raise LivePackContractError("requested row falls outside its physical pack")
        by_pack[pack_index].add(position)
    pack_requests = tuple(
        PackForwardRequest(
            **{
                **asdict(requests_by_pack[index]),
                "compact_positions": tuple(sorted(by_pack[index])),
            }
        )
        for index in sorted(requests_by_pack)
    )
    if any(not item.compact_positions for item in pack_requests):
        raise LivePackContractError("every planned pack must request at least one row")

    scored_token_indices = (
        None
        if credit_ledger is None
        else _scored_token_indices(credit_ledger, image_id, bindings)
    )
    if credit_ledger is not None:
        image = next(item for item in credit_ledger.images if item.image_id == image_id)
        if image.acquisition_group_sha256 != publication.replayed_group.content_sha256:
            raise LivePackContractError(
                "credit ledger acquisition group differs from this publication"
            )
        if credit_ledger.training_repetition_penalty is not None and (
            credit_ledger.training_repetition_penalty
            != execution.plan.repetition_penalty
            or credit_ledger.seed_group_id != execution.plan.seed_group_id
        ):
            raise LivePackContractError(
                "credit ledger RP/seed lineage differs from this acquisition plan"
            )

    return LivePackPlan(
        image_id=image_id,
        seed_group_id=execution.plan.seed_group_id,
        repetition_penalty=float(execution.plan.repetition_penalty),
        acquisition_group_sha256=group.content_sha256,
        plan_sha256=execution.plan_sha256,
        prompt_token_count=len(prompt),
        prompt_token_sha256=token_ids_sha256(prompt),
        request_order=request_order,
        trajectory_bindings=tuple(bindings),
        compiler_requests=compiler_requests,
        row_bindings=tuple(row_bindings),
        compiler_row_bindings=compiler_row_bindings,
        pack_requests=pack_requests,
        scored_token_indices=scored_token_indices,
        publication=publication,
        packed_plan=packed_plan,
    )


# --------------------------------------------------------------------------
# Forward seam and materialization
# --------------------------------------------------------------------------


def default_live_packed_forward(
    model: Any,
    runtime: Any,
    tokenizer: Any,
    packed: Any,
    positions: tuple[int, ...],
) -> Any:
    """Run the existing Qwen no-padding FA2/MRoPE compact-logit forward.

    Unlike the no-update census forward this path keeps autograd enabled: the
    same rows carry both the replay evidence and the score-function gradient.
    """

    import src.qwen.forward as qwen_forward
    from src.qwen.fa2 import build_fa2_varlen_plan

    device = getattr(getattr(runtime, "accelerator", None), "device", None)
    if device is None:
        raise LivePackContractError(
            "live pack runtime does not expose accelerator.device"
        )
    inputs = qwen_forward.build_qwen_forward_inputs(
        packed.pack,
        packed.encoded_examples,
        packed.position_inputs,
        fa2_varlen_plan=build_fa2_varlen_plan(packed.pack, device=device),
        logits_to_keep_positions=positions,
        device=device,
        fa2_branch_proof_policy=FA2_BRANCH_PROOF_POLICY,
    )
    return qwen_forward.run_qwen_forward(
        model,
        inputs,
        expected_vocab_size=_tokenizer_vocab_size(tokenizer),
        capture_fa2_branch=True,
        require_fa2_branch_proof=True,
    )


def _tokenizer_vocab_size(tokenizer: Any) -> int:
    try:
        value = len(tokenizer)
    except (TypeError, AttributeError) as error:
        raise LivePackContractError(
            "live pack tokenizer does not expose its full vocabulary"
        ) from error
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise LivePackContractError("live pack tokenizer vocabulary is invalid")
    return value


def _compact_rows(
    result: Any, *, request: PackForwardRequest, expected_vocab_size: int
) -> tuple[Any, dict[int, int]]:
    import torch

    logits = getattr(result, "logits", None)
    positions = getattr(result, "logits_position_ids", None)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or logits.shape[0] != 1:
        raise LivePackContractError(
            f"pack {request.pack_index} forward must return logits with shape [1,N,V]"
        )
    if int(logits.shape[2]) != expected_vocab_size:
        raise LivePackContractError(
            f"pack {request.pack_index} forward returned a different vocabulary"
        )
    if positions is None:
        raise LivePackContractError(
            f"pack {request.pack_index} forward omitted its compact position coverage"
        )
    returned = tuple(int(item) for item in positions)
    if (
        len(returned) != int(logits.shape[1])
        or len(set(returned)) != len(returned)
        or set(returned) != set(request.compact_positions)
    ):
        raise LivePackContractError(
            f"pack {request.pack_index} compact logits position coverage differs"
        )
    rows = logits[0]
    if not bool(torch.isfinite(rows.detach()).all().item()):
        raise LivePackContractError(
            f"pack {request.pack_index} compact logits must be finite"
        )
    return rows, {position: index for index, position in enumerate(returned)}


def _sealed_row_identity(row: PhysicalRowBinding) -> tuple[str, int]:
    if row.request_id is None or row.token_index is None:
        raise LivePackContractError("a sealed trajectory row lost its token identity")
    return row.request_id, row.token_index


def _compiler_site_id(row: PhysicalRowBinding) -> str:
    if row.site_id is None:
        raise LivePackContractError("a compiler row lost its site identity")
    return row.site_id


def _processed_chosen_logprob(row: Any, token: Any, contract: Any) -> Any:
    """Differentiable mirror of the sealed replay processor order.

    ``repetition_penalty`` then ``temperature`` then ``log_softmax``, with the
    history read from the sealed token evidence and the repetition processor
    applied once per token type.
    """

    import torch

    logits = row.to(dtype=torch.float32)
    history = token.history_token_ids
    if contract.repetition_penalty != 1.0 and history:
        indices = torch.tensor(
            sorted(set(history)), device=logits.device, dtype=torch.long
        )
        selected = logits.index_select(0, indices)
        penalized = torch.where(
            selected < 0,
            selected * contract.repetition_penalty,
            selected / contract.repetition_penalty,
        )
        logits = logits.scatter(0, indices, penalized)
    logits = logits / contract.temperature
    return torch.log_softmax(logits, dim=-1)[token.chosen_token_id]


def materialize_live_packs(
    *,
    plan: LivePackPlan,
    expected_vocab_size: int,
    model: Any = None,
    runtime: Any = None,
    tokenizer: Any = None,
    packed_forward: Callable[..., Any] | None = None,
    verify_processed_transform: bool = True,
) -> MaterializedImagePacks:
    """Forward every planned pack once and emit the sealed ephemeral rows."""

    import torch

    from scripts.research.collect_human13_rp_crossover import PackedRawLogits
    from scripts.research.human13_rp_policy import processed_policy_logprobs

    if type(plan) is not LivePackPlan:
        raise LivePackContractError("materialization requires a typed LivePackPlan")
    if (
        isinstance(expected_vocab_size, bool)
        or not isinstance(expected_vocab_size, int)
        or expected_vocab_size < 2
    ):
        raise LivePackContractError(
            "expected_vocab_size must be a real vocabulary size"
        )
    forward = packed_forward or default_live_packed_forward
    packs_by_index = {
        int(pack.pack.pack_index): pack for pack in plan.packed_plan.packs
    }

    compact: dict[int, tuple[Any, dict[int, int]]] = {}
    compact_row_bytes = 0
    forward_count = 0
    for request in plan.pack_requests:
        result = forward(
            model,
            runtime,
            tokenizer,
            packs_by_index[request.pack_index],
            request.compact_positions,
        )
        forward_count += 1
        rows, row_by_position = _compact_rows(
            result, request=request, expected_vocab_size=expected_vocab_size
        )
        compact[request.pack_index] = (rows, row_by_position)
        compact_row_bytes += int(rows.numel()) * int(rows.element_size())

    trajectories = {
        item.identity.request_id: item
        for item in plan.publication.execution.group.trajectories
    }
    chunks: list[Any] = []
    policy_logprobs: dict[str, Any] = {}
    cursor = 0
    for binding in plan.trajectory_bindings:
        rows, row_by_position = compact[plan.row_bindings[cursor].pack_index]
        selected = plan.row_bindings[cursor : cursor + binding.generated_token_count]
        cursor += binding.generated_token_count
        if tuple(_sealed_row_identity(row) for row in selected) != tuple(
            (binding.request_id, index)
            for index in range(binding.generated_token_count)
        ):
            raise LivePackContractError(
                "sealed row order differs from its trajectory token order"
            )
        index = torch.tensor(
            [row_by_position[row.packed_causal_position] for row in selected],
            dtype=torch.long,
            device=rows.device,
        )
        gathered = rows.index_select(0, index)
        chunks.append(gathered)
        trajectory = trajectories[binding.request_id]
        contract = trajectory.policy_contract
        values = []
        for offset, token in enumerate(trajectory.generated_tokens):
            value = _processed_chosen_logprob(gathered[offset], token, contract)
            if verify_processed_transform:
                reference = processed_policy_logprobs(
                    gathered[offset].detach(), token, contract
                )[token.chosen_token_id]
                if not bool(
                    torch.isclose(
                        value.detach(),
                        reference,
                        atol=PROCESSED_TRANSFORM_TOLERANCE_NATS,
                        rtol=0.0,
                    ).item()
                ):
                    raise LivePackContractError(
                        "live processed policy log probability differs from the "
                        "sealed replay processor"
                    )
            values.append(value)
        policy_logprobs[binding.request_id] = torch.stack(values)

    sealed = torch.cat(chunks, dim=0)
    sealed_identities = tuple(_sealed_row_identity(row) for row in plan.row_bindings)
    packed_raw_logits = PackedRawLogits(
        request_ids=tuple(request_id for request_id, _ in sealed_identities),
        token_indices=tuple(token_index for _, token_index in sealed_identities),
        logits=sealed.detach(),
    )
    compiler_rows = {
        _compiler_site_id(row): CompilerPackedRow(
            site_id=_compiler_site_id(row),
            pack_index=row.pack_index,
            logits_position_ids=(row.packed_causal_position,),
            raw_logits=compact[row.pack_index][0].index_select(
                0,
                torch.tensor(
                    [compact[row.pack_index][1][row.packed_causal_position]],
                    dtype=torch.long,
                    device=compact[row.pack_index][0].device,
                ),
            ),
        )
        for row in plan.compiler_row_bindings
    }

    receipt = LivePackReceipt(
        schema_version=SCHEMA_VERSION,
        image_id=plan.image_id,
        seed_group_id=plan.seed_group_id,
        repetition_penalty=plan.repetition_penalty,
        acquisition_group_sha256=plan.acquisition_group_sha256,
        plan_sha256=plan.plan_sha256,
        prompt_token_sha256=plan.prompt_token_sha256,
        pack_count=len(plan.pack_requests),
        forward_count=forward_count,
        packed_token_count=sum(item.pack_length for item in plan.pack_requests),
        logical_token_count=sum(
            item.encoded_length for item in plan.packed_plan.logical_segments
        ),
        requested_row_count=len(plan.row_bindings),
        compiler_row_count=len(plan.compiler_row_bindings),
        vocab_size=expected_vocab_size,
        sealed_row_bytes=int(sealed.numel()) * int(sealed.element_size()),
        compact_row_bytes=compact_row_bytes,
        max_pack_length=max(item.pack_length for item in plan.pack_requests),
        segment_count=len(plan.packed_plan.logical_segments),
        pack_requests=plan.pack_requests,
        row_bindings=plan.row_bindings,
        compiler_row_bindings=plan.compiler_row_bindings,
    )
    return MaterializedImagePacks(
        plan=plan,
        packed_raw_logits=packed_raw_logits,
        policy_logprobs=policy_logprobs,
        compiler_rows=compiler_rows,
        receipt=receipt,
    )


def stream_panel_live_packs(
    *,
    acquisition: Any,
    skeletons: Mapping[int, Any],
    expected_vocab_size: int,
    credit_ledger: Any = None,
    compiler_ledger: Any = None,
    compiler_segments_by_image: Mapping[int, Sequence[CompilerSegmentRequest]]
    | None = None,
    model: Any = None,
    runtime: Any = None,
    tokenizer: Any = None,
    packed_forward: Callable[..., Any] | None = None,
    global_max_length: int | None = None,
    verify_processed_transform: bool = True,
) -> Iterator[MaterializedImagePacks]:
    """Yield one image's materialization at a time, releasing the previous one."""

    from scripts.research.human13_trajectory_credit import (
        TrajectoryCreditPanelAcquisition,
    )

    if type(acquisition) is not TrajectoryCreditPanelAcquisition:
        raise LivePackContractError(
            "panel streaming requires a typed TrajectoryCreditPanelAcquisition"
        )
    if credit_ledger is not None and (
        getattr(credit_ledger, "acquisition_sha256", None) != acquisition.content_sha256
    ):
        raise LivePackContractError(
            "credit ledger acquisition lineage differs from this panel"
        )
    if compiler_ledger is not None:
        if compiler_ledger.acquisition_sha256 != acquisition.content_sha256:
            raise LivePackContractError(
                "compiler ledger acquisition lineage differs from this panel"
            )
        if credit_ledger is not None and (
            compiler_ledger.trajectory_credit_sha256 != credit_ledger.content_sha256
        ):
            raise LivePackContractError(
                "compiler ledger Task3 lineage differs from this credit ledger"
            )
    by_image = dict(compiler_segments_by_image or {})
    previous: MaterializedImagePacks | None = None
    try:
        for publication in acquisition.publications:
            image_id = int(publication.execution.plan.image_id)
            if image_id not in skeletons:
                raise LivePackContractError(
                    f"image {image_id} has no canonical processor skeleton"
                )
            if previous is not None:
                previous.release()
            plan = plan_live_packs(
                publication=publication,
                skeleton=skeletons[image_id],
                compiler_segments=tuple(by_image.get(image_id, ())),
                credit_ledger=credit_ledger,
                compiler_ledger=compiler_ledger,
                global_max_length=global_max_length,
            )
            previous = materialize_live_packs(
                plan=plan,
                expected_vocab_size=expected_vocab_size,
                model=model,
                runtime=runtime,
                tokenizer=tokenizer,
                packed_forward=packed_forward,
                verify_processed_transform=verify_processed_transform,
            )
            yield previous
    finally:
        # an abandoned or failed stream still releases its live tensors
        if previous is not None:
            previous.release()


def backward_incremental_objectives(
    steps: Iterator[StreamingObjectiveStep],
    *,
    trajectory_denominator: int,
    compiler_image_denominator: int | None,
    include_compiler: bool,
    backward: Callable[[Any], None] | None = None,
) -> IncrementalBackwardReceipt:
    """Backward each unnormalized image/pack numerator and release immediately.

    No numerator sequence is materialized or stacked.  The global ``N*K`` and
    image-mean compiler denominators are applied to every additive numerator,
    which is algebraically identical to dividing once after a global sum while
    bounding the live autograd graph to one yielded step.
    """

    import torch

    if (
        isinstance(trajectory_denominator, bool)
        or not isinstance(trajectory_denominator, int)
        or trajectory_denominator <= 0
    ):
        raise LivePackContractError("trajectory denominator must be positive")
    if include_compiler:
        if (
            isinstance(compiler_image_denominator, bool)
            or not isinstance(compiler_image_denominator, int)
            or compiler_image_denominator <= 0
        ):
            raise LivePackContractError("compiler image denominator must be positive")
    elif compiler_image_denominator is not None:
        raise LivePackContractError(
            "trajectory-only streaming cannot carry a compiler denominator"
        )

    image_ids: list[int] = []
    backward_count = 0
    released_graph_count = 0
    for step in steps:
        if not isinstance(step, StreamingObjectiveStep):
            raise LivePackContractError("stream must yield StreamingObjectiveStep")
        if step.image_id in image_ids:
            raise LivePackContractError("streaming objective duplicated an image")
        if not callable(step.release):
            raise LivePackContractError("streaming objective requires a release owner")
        trajectory = step.trajectory_numerator
        compiler = step.compiler_numerator
        if not torch.is_tensor(trajectory) or trajectory.numel() != 1:
            raise LivePackContractError(
                "trajectory numerator must be one scalar tensor"
            )
        if include_compiler and (
            not torch.is_tensor(compiler) or compiler.numel() != 1
        ):
            raise LivePackContractError("compiler numerator must be one scalar tensor")
        if not include_compiler and compiler is not None:
            raise LivePackContractError(
                "trajectory-only step must not retain a compiler numerator"
            )

        loss = trajectory.reshape(()) / trajectory_denominator
        if include_compiler:
            assert compiler is not None and compiler_image_denominator is not None
            loss = loss + compiler.reshape(()) / compiler_image_denominator
        release = step.release
        image_id = step.image_id
        try:
            if backward is None:
                loss.backward()
            else:
                backward(loss)
            backward_count += 1
            image_ids.append(image_id)
        finally:
            # Delete the sole local graph-bearing references before advancing
            # the iterator to the next image.
            del loss, trajectory, compiler, step
            release()
            released_graph_count += 1

    if not image_ids:
        raise LivePackContractError("incremental backward requires at least one step")
    return IncrementalBackwardReceipt(
        image_ids=tuple(image_ids),
        trajectory_denominator=trajectory_denominator,
        compiler_image_denominator=compiler_image_denominator,
        backward_count=backward_count,
        released_graph_count=released_graph_count,
    )


def combine_trajectory_numerators(numerators: Sequence[Any], ledger: Any) -> Any:
    """Sum unnormalized numerators and apply the one logical ``N*K`` denominator."""

    import torch

    from scripts.research.human13_trajectory_credit import TrajectoryCreditLedger

    if type(ledger) is not TrajectoryCreditLedger:
        raise LivePackContractError(
            "the global denominator requires an exact TrajectoryCreditLedger"
        )
    terms = tuple(numerators)
    if not terms or any(not torch.is_tensor(item) for item in terms):
        raise LivePackContractError("trajectory numerators must be nonempty tensors")
    return torch.stack([item.reshape(()) for item in terms]).sum() / (
        ledger.logical_denominator
    )


__all__ = [
    "CompilerPackedRow",
    "CompilerSegmentRequest",
    "FA2_BRANCH_PROOF_POLICY",
    "LivePackContractError",
    "LivePackPlan",
    "LivePackReceipt",
    "IncrementalBackwardReceipt",
    "MaterializedImagePacks",
    "StreamingObjectiveStep",
    "backward_incremental_objectives",
    "PackForwardRequest",
    "PhysicalRowBinding",
    "SCHEMA_VERSION",
    "SEGMENT_ROLE",
    "TrajectoryRowBinding",
    "combine_trajectory_numerators",
    "default_live_packed_forward",
    "materialize_live_packs",
    "plan_live_packs",
    "stream_panel_live_packs",
]
