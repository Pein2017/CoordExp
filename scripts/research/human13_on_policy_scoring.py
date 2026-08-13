"""Current-frontier candidate scoring over packed-prefilter and exact HF surfaces.

The module is deliberately runtime-injected: importing it does not load a model or
touch a GPU.  Every native alias is encoded as an independent causal segment after
the exact natural pre-stop prefix.  The packed BF16/FA2 surface only bounds HF work;
the exact HF fp32/SDPA surface owns all scientific ranking decisions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any

import torch

from scripts.research.build_human13_on_policy_frontier import (
    FrontierImage,
    natural_pre_stop_prefix,
)
from scripts.research.human13_frontier_selection import (
    CandidatePath,
    CandidateScore,
    PackedCandidateScore,
    packed_prefilter,
    score_candidate_tensor,
    score_packed_candidate_tensor,
    shortlist_candidates,
)
from scripts.research.human13_live_census import (
    _clone_skeleton,
    _default_packed_forward,
    _position_rows,
    _vocab_size,
)
from scripts.research.run_human13_k_union_overfit import (
    GLOBAL_MAX_LENGTH,
    LogicalPanelSegment,
    PackedPanelMicroStep,
    PackedPanelPlan,
    plan_panel_packs,
)
from src.inference.backend import token_ids_sha256


PackedForward = Callable[[Any, Any, Any, PackedPanelMicroStep, tuple[int, ...]], Any]


@dataclass(frozen=True)
class CandidateSegmentBinding:
    path: CandidatePath
    segment_id: str
    local_causal_positions: tuple[int, ...]
    prompt_token_sha256: str
    natural_pre_stop_prefix_token_sha256: str
    candidate_token_sha256: str


@dataclass(frozen=True)
class NaturalPrefixReceipt:
    image_id: int
    raw_natural_token_sha256: str
    natural_pre_stop_prefix_token_sha256: str
    raw_natural_token_count: int
    natural_pre_stop_token_count: int
    terminal_token_index: int | None
    malformed_row_count: int


@dataclass(frozen=True)
class PreparedCandidateScoring:
    packed_plan: PackedPanelPlan
    bindings: tuple[CandidateSegmentBinding, ...]
    prefix_receipts: tuple[NaturalPrefixReceipt, ...]


@dataclass(frozen=True)
class TensorArtifactReceipt:
    shape: tuple[int, int]
    dtype: str
    positions: tuple[int, ...]
    sha256: str


@dataclass(frozen=True)
class PackedCandidateReceipt:
    path: CandidatePath
    segment_id: str
    local_causal_positions: tuple[int, ...]
    packed_causal_positions: tuple[int, ...]
    packed_tensor_artifact: TensorArtifactReceipt
    score: PackedCandidateScore


@dataclass(frozen=True)
class CrossSurfaceCandidateReceipt:
    path: CandidatePath
    segment_id: str
    packed_causal_positions: tuple[int, ...]
    hf_causal_positions: tuple[int, ...]
    packed_tensor_artifact: TensorArtifactReceipt
    hf_tensor_artifact: TensorArtifactReceipt
    score: CandidateScore


@dataclass(frozen=True)
class OnPolicyCandidateScoringResult:
    prepared: PreparedCandidateScoring
    packed_receipts: tuple[PackedCandidateReceipt, ...]
    cross_surface_receipts: tuple[CrossSurfaceCandidateReceipt, ...]
    shortlist_by_image: Mapping[int, tuple[CandidateScore, ...]]
    receipt: Mapping[str, int | str]


def _prompt(skeleton: Any, *, image_id: int) -> tuple[int, ...]:
    count = getattr(skeleton, "prompt_token_count", None)
    input_ids = getattr(skeleton, "input_ids", None)
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        or not isinstance(input_ids, tuple)
        or len(input_ids) < count
    ):
        raise ValueError(f"image {image_id} has an invalid processor prompt skeleton")
    prompt = tuple(int(token_id) for token_id in input_ids[:count])
    if any(token_id < 0 for token_id in prompt):
        raise ValueError(f"image {image_id} processor prompt has a negative token")
    return prompt


def _candidate_key(path: CandidatePath) -> tuple[int, str, str]:
    return path.image_id, path.owner_id, path.alias_id


def prepare_on_policy_candidate_scoring(
    *,
    frontier_images: Mapping[int, FrontierImage],
    prompt_skeletons: Mapping[int, Any],
    global_max_length: int = GLOBAL_MAX_LENGTH,
) -> PreparedCandidateScoring:
    """Materialize one isolated, zero-padding-packable segment per native alias."""

    if set(frontier_images) != set(prompt_skeletons):
        raise ValueError("frontier and processor-skeleton image identities differ")
    logical: list[LogicalPanelSegment] = []
    bindings: list[CandidateSegmentBinding] = []
    prefix_receipts: list[NaturalPrefixReceipt] = []
    seen_segments: set[str] = set()
    seen_paths: set[tuple[int, str, str]] = set()

    for image_id in sorted(frontier_images):
        image = frontier_images[image_id]
        if image.image_id != image_id:
            raise ValueError("frontier mapping key differs from its image identity")
        prompt = _prompt(prompt_skeletons[image_id], image_id=image_id)
        natural_prefix = natural_pre_stop_prefix(image)
        prefix_receipt = NaturalPrefixReceipt(
            image_id=image_id,
            raw_natural_token_sha256=token_ids_sha256(image.generated_token_ids),
            natural_pre_stop_prefix_token_sha256=token_ids_sha256(natural_prefix),
            raw_natural_token_count=len(image.generated_token_ids),
            natural_pre_stop_token_count=len(natural_prefix),
            terminal_token_index=image.terminal_token_index,
            malformed_row_count=image.malformed_row_count,
        )
        prefix_receipts.append(prefix_receipt)
        uncovered = set(image.uncovered_h_owner_ids)
        for alias in sorted(
            image.candidate_aliases,
            key=lambda item: (
                item.owner_id,
                item.row_id,
                item.trajectory_id,
                item.seed,
            ),
        ):
            if alias.owner_id not in uncovered:
                raise ValueError(
                    "candidate alias is outside the current uncovered H frontier"
                )
            if not alias.token_ids or any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                for token_id in alias.token_ids
            ):
                raise ValueError(
                    "candidate alias must contain nonnegative integer tokens"
                )
            path = CandidatePath(
                image_id=image_id,
                owner_id=alias.owner_id,
                alias_id=alias.row_id,
                token_ids=tuple(alias.token_ids),
            )
            key = _candidate_key(path)
            if key in seen_paths:
                raise ValueError("candidate path identities are not unique")
            seen_paths.add(key)
            segment_id = f"on-policy-score:{image_id}:{alias.owner_id}:{alias.row_id}"
            if segment_id in seen_segments:
                raise ValueError("candidate segment identities are not unique")
            seen_segments.add(segment_id)
            row_start = len(prompt) + len(natural_prefix)
            local_positions = tuple(
                row_start + offset - 1 for offset in range(len(path.token_ids))
            )
            input_ids = (*prompt, *natural_prefix, *path.token_ids)
            if any(
                position < len(prompt) - 1
                or position >= len(input_ids) - 1
                or input_ids[position + 1] != target
                for position, target in zip(
                    local_positions, path.token_ids, strict=True
                )
            ):
                raise ValueError(
                    "candidate causal positions differ from exact row targets"
                )
            encoded = _clone_skeleton(
                prompt_skeletons[image_id],
                segment_id=segment_id,
                image_id=image_id,
                input_ids=input_ids,
            )
            object.__setattr__(encoded, "human13_candidate_path", path)
            object.__setattr__(encoded, "human13_candidate_positions", local_positions)
            logical.append(
                LogicalPanelSegment(
                    segment_id=segment_id,
                    image_id=image_id,
                    role="h1_independent",
                    encoded_example=encoded,
                )
            )
            bindings.append(
                CandidateSegmentBinding(
                    path=path,
                    segment_id=segment_id,
                    local_causal_positions=local_positions,
                    prompt_token_sha256=token_ids_sha256(prompt),
                    natural_pre_stop_prefix_token_sha256=(
                        prefix_receipt.natural_pre_stop_prefix_token_sha256
                    ),
                    candidate_token_sha256=token_ids_sha256(path.token_ids),
                )
            )

    packed_plan = (
        plan_panel_packs(logical, global_max_length=global_max_length)
        if logical
        else PackedPanelPlan((), (), global_max_length)
    )
    if tuple(sorted(item.segment_id for item in logical)) != tuple(
        sorted(item.segment_id for item in packed_plan.logical_segments)
    ):
        raise ValueError("packed plan does not cover every isolated candidate segment")
    if any(
        pack.pack.length
        != sum(segment.end - segment.start for segment in pack.pack.segments)
        or pack.pack.to_artifact_dict().get("padding_tokens") != 0
        for pack in packed_plan.packs
    ):
        raise ValueError("candidate packing introduced padding or position gaps")
    return PreparedCandidateScoring(
        packed_plan=packed_plan,
        bindings=tuple(bindings),
        prefix_receipts=tuple(prefix_receipts),
    )


def _tensor_artifact(
    logits: torch.Tensor, *, positions: tuple[int, ...]
) -> TensorArtifactReceipt:
    if logits.ndim != 2 or int(logits.shape[0]) != len(positions) or not positions:
        raise ValueError("candidate tensor artifact shape or positions differ")
    materialized = logits.detach().cpu().contiguous()
    shape = (int(materialized.shape[0]), int(materialized.shape[1]))
    dtype = str(materialized.dtype)
    header = json.dumps(
        {"dtype": dtype, "positions": positions, "shape": shape},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(materialized.view(torch.uint8).numpy().tobytes())
    return TensorArtifactReceipt(shape, dtype, positions, digest.hexdigest())


def _capture_packed_candidates(
    *,
    prepared: PreparedCandidateScoring,
    packed_model: Any,
    packed_runtime: Any,
    tokenizer: Any,
    packed_forward: PackedForward,
) -> tuple[PackedCandidateReceipt, ...]:
    by_segment = {binding.segment_id: binding for binding in prepared.bindings}
    if len(by_segment) != len(prepared.bindings):
        raise ValueError("candidate scoring bindings are not unique")
    receipts: list[PackedCandidateReceipt] = []
    for packed in prepared.packed_plan.packs:
        resolved: list[tuple[CandidateSegmentBinding, tuple[int, ...]]] = []
        for segment in packed.pack.segments:
            binding = by_segment.get(segment.example_id)
            if binding is None:
                raise ValueError("packed candidate segment has no causal binding")
            positions = tuple(
                segment.start + position for position in binding.local_causal_positions
            )
            if any(
                position < segment.start or position >= segment.end - 1
                for position in positions
            ):
                raise ValueError("packed candidate causal position escapes its segment")
            resolved.append((binding, positions))
        requested = tuple(
            sorted(
                {position for _binding, positions in resolved for position in positions}
            )
        )
        if not requested:
            raise ValueError("packed candidate micro-step has no causal positions")
        output = packed_forward(
            packed_model,
            packed_runtime,
            tokenizer,
            packed,
            requested,
        )
        rows = _position_rows(
            output,
            expected_positions=requested,
            expected_vocab_size=_vocab_size(tokenizer),
            label="packed candidate",
        )
        for binding, positions in resolved:
            candidate_logits = torch.stack(
                tuple(rows[position] for position in positions)
            )
            artifact = _tensor_artifact(candidate_logits, positions=positions)
            score = score_packed_candidate_tensor(
                binding.path,
                candidate_logits,
                expected_vocab_size=_vocab_size(tokenizer),
            )
            receipts.append(
                PackedCandidateReceipt(
                    path=binding.path,
                    segment_id=binding.segment_id,
                    local_causal_positions=binding.local_causal_positions,
                    packed_causal_positions=positions,
                    packed_tensor_artifact=artifact,
                    score=score,
                )
            )
    if {_candidate_key(item.path) for item in receipts} != {
        _candidate_key(binding.path) for binding in prepared.bindings
    } or len(receipts) != len(prepared.bindings):
        raise ValueError("packed candidate evidence coverage differs")
    return tuple(sorted(receipts, key=lambda item: _candidate_key(item.path)))


def score_on_policy_frontier_candidates(
    *,
    frontier_images: Mapping[int, FrontierImage],
    prompt_skeletons: Mapping[int, Any],
    packed_model: Any,
    packed_runtime: Any,
    tokenizer: Any,
    hf_scorer: Any,
    packed_forward: PackedForward | None = None,
    aliases_per_owner: int = 2,
    shortlist_limit: int = 4,
    global_max_length: int = GLOBAL_MAX_LENGTH,
) -> OnPolicyCandidateScoringResult:
    """Score the current frontier while keeping HF as the only decision surface."""

    if getattr(hf_scorer, "model_dtype", None) != "torch.float32":
        raise ValueError("candidate HF scorer must be exact fp32")
    if getattr(hf_scorer, "attention_implementation", None) != "sdpa":
        raise ValueError("candidate HF scorer must use SDPA")
    prepared = prepare_on_policy_candidate_scoring(
        frontier_images=frontier_images,
        prompt_skeletons=prompt_skeletons,
        global_max_length=global_max_length,
    )
    vocab_size = _vocab_size(tokenizer)
    if any(
        token_id >= vocab_size
        for binding in prepared.bindings
        for token_id in binding.path.token_ids
    ):
        raise ValueError(
            "candidate target token is outside the full tokenizer vocabulary"
        )
    packed_receipts = _capture_packed_candidates(
        prepared=prepared,
        packed_model=packed_model,
        packed_runtime=packed_runtime,
        tokenizer=tokenizer,
        packed_forward=packed_forward or _default_packed_forward,
    )
    packed_by_key = {_candidate_key(item.path): item for item in packed_receipts}
    bindings = {binding.segment_id: binding for binding in prepared.bindings}
    encoded_by_segment = {
        segment.segment_id: segment.encoded_example
        for segment in prepared.packed_plan.logical_segments
    }

    survivors: list[PackedCandidateReceipt] = []
    for image_id in sorted(frontier_images):
        image_scores = tuple(
            item.score for item in packed_receipts if item.path.image_id == image_id
        )
        kept = packed_prefilter(image_scores, aliases_per_owner=aliases_per_owner)
        survivors.extend(packed_by_key[_candidate_key(score.path)] for score in kept)

    cross_surface: list[CrossSurfaceCandidateReceipt] = []
    for packed_receipt in survivors:
        binding = bindings[packed_receipt.segment_id]
        encoded = encoded_by_segment[packed_receipt.segment_id]
        hf_output = hf_scorer.score_causal_logits(
            encoded,
            binding.local_causal_positions,
        )
        hf_rows = _position_rows(
            hf_output,
            expected_positions=binding.local_causal_positions,
            expected_vocab_size=_vocab_size(tokenizer),
            label="HF candidate",
            require_fp32=True,
        )
        hf_logits = torch.stack(
            tuple(hf_rows[position] for position in binding.local_causal_positions)
        )
        hf_artifact = _tensor_artifact(
            hf_logits, positions=binding.local_causal_positions
        )
        score = score_candidate_tensor(
            binding.path,
            packed=packed_receipt.score,
            hf_logits=hf_logits,
            expected_vocab_size=_vocab_size(tokenizer),
        )
        cross_surface.append(
            CrossSurfaceCandidateReceipt(
                path=binding.path,
                segment_id=binding.segment_id,
                packed_causal_positions=packed_receipt.packed_causal_positions,
                hf_causal_positions=binding.local_causal_positions,
                packed_tensor_artifact=packed_receipt.packed_tensor_artifact,
                hf_tensor_artifact=hf_artifact,
                score=score,
            )
        )

    all_scores = tuple(item.score for item in cross_surface)
    globally_shortlisted = (
        shortlist_candidates(all_scores, limit=shortlist_limit) if all_scores else ()
    )
    shortlist_by_image: dict[int, tuple[CandidateScore, ...]] = {
        image_id: tuple(
            score for score in globally_shortlisted if score.path.image_id == image_id
        )
        for image_id in sorted(frontier_images)
    }
    shortlisted_owner_count = sum(len(scores) for scores in shortlist_by_image.values())
    return OnPolicyCandidateScoringResult(
        prepared=prepared,
        packed_receipts=packed_receipts,
        cross_surface_receipts=tuple(cross_surface),
        shortlist_by_image=MappingProxyType(shortlist_by_image),
        receipt=MappingProxyType(
            {
                "schema_version": "human13_on_policy_candidate_scoring.v1",
                "packed_surface": "packed_bf16_fa2",
                "decision_surface": "hf_fp32_sdpa",
                "image_count": len(frontier_images),
                "logical_segment_count": len(prepared.bindings),
                "physical_pack_count": len(prepared.packed_plan.packs),
                "packed_forward_count": len(prepared.packed_plan.packs),
                "packed_candidate_count": len(packed_receipts),
                "hf_forward_count": len(cross_surface),
                "hf_candidate_count": len(cross_surface),
                "shortlisted_owner_count": shortlisted_owner_count,
                "aliases_per_owner": aliases_per_owner,
                "shortlist_limit": shortlist_limit,
                "padding_tokens": 0,
            }
        ),
    )


__all__ = [
    "CandidateSegmentBinding",
    "CrossSurfaceCandidateReceipt",
    "NaturalPrefixReceipt",
    "OnPolicyCandidateScoringResult",
    "PackedCandidateReceipt",
    "PreparedCandidateScoring",
    "TensorArtifactReceipt",
    "prepare_on_policy_candidate_scoring",
    "score_on_policy_frontier_candidates",
]
