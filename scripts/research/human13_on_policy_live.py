"""Experiment-local packed payload and gate for the Human-13 on-policy successor."""

from __future__ import annotations

from dataclasses import dataclass, is_dataclass, replace
import hashlib
from types import SimpleNamespace
from typing import Any, Literal, Mapping, Sequence, cast

import torch

from scripts.research import run_human13_k_union_overfit as base
from scripts.research.build_human13_on_policy_frontier import FrontierImage
from scripts.research.human13_frontier_selection import CandidateScore
from src.losses.human13_k_union import (
    first_bottleneck_argmax_hinge,
    image_balanced_duplicate_token_contrast,
    owner_mean_masked_row_cross_entropy,
    rectangle_valid_argmax_hinge,
)
from src.losses.normalizers import SegmentBalancedDenominator
from src.losses.runner import LossBundle, LossTermResult
from src.supervision import build_token_sequence_from_packed_supervision
from src.training.supervised_trainer import SupervisedMicroStep


ArmId = Literal["O-Full-Safe", "O-First-Safe"]
Objective = Literal["full_row_ce", "first_bottleneck", "rectangle", "duplicate_pair"]

_COORD_TOKEN_START = 151670
_COORD_TOKEN_END_EXCLUSIVE = 152670


@dataclass(frozen=True)
class OnPolicyBinding:
    objective: Objective
    unit_id: str
    owner_id: str
    token_indices: tuple[int, ...]
    target_token_ids: tuple[int, ...]
    required_margin: float | None = None
    lower_coordinate_token_ids: tuple[int, ...] = ()
    duplicate_token_id: int | None = None


@dataclass(frozen=True)
class OnPolicyLossSite:
    objective: Objective
    unit_id: str
    owner_id: str
    image_id: int
    segment_id: str
    logits_positions: tuple[int, ...]
    target_token_ids: tuple[int, ...]
    required_margin: float | None = None
    lower_coordinate_token_ids: tuple[int, ...] = ()
    duplicate_token_id: int | None = None


@dataclass(frozen=True)
class PrefixReceipt:
    image_id: int
    raw_natural_token_sha256: str
    training_prefix_token_sha256: str
    raw_natural_token_count: int
    training_prefix_token_count: int
    removed_duplicate_row_orders: tuple[int, ...]
    terminal_removed: bool


@dataclass(frozen=True)
class MaterializedOnPolicySegments:
    arm_id: ArmId
    segments: tuple[base.LogicalPanelSegment, ...]
    prefix_receipts: tuple[PrefixReceipt, ...]

    def preflight(self, global_max_length: int = base.GLOBAL_MAX_LENGTH) -> None:
        if not self.segments:
            raise ValueError("on-policy materialization emitted no segments")
        for segment in self.segments:
            if segment.encoded_length > global_max_length:
                raise ValueError(
                    f"on-policy segment {segment.segment_id} exceeds {global_max_length}"
                )


@dataclass(frozen=True)
class OnPolicyDenominators:
    positive_owners: int
    rectangle_rows: int
    duplicate_images: int
    duplicate_events_by_image: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class OnPolicyLossConfig:
    required_margin: float
    rectangle_margin: float
    duplicate_margin: float


@dataclass(frozen=True)
class OnPolicyPayload:
    arm_id: ArmId
    selected_segments: tuple[base.LogicalPanelSegment, ...]
    packed_plan: base.PackedPanelPlan
    sites_by_pack: dict[int, tuple[OnPolicyLossSite, ...]]
    micro_steps: tuple[SupervisedMicroStep, ...]
    denominators: OnPolicyDenominators
    prefix_receipts: tuple[PrefixReceipt, ...]
    loss_config: OnPolicyLossConfig


@dataclass(frozen=True)
class OnPolicyPackContext:
    logits: torch.Tensor
    logits_position_ids: tuple[int, ...] | None
    sites: tuple[OnPolicyLossSite, ...]


@dataclass(frozen=True)
class OnPolicyLossPlan:
    denominators: OnPolicyDenominators
    micro_step_count: int
    loss_config: OnPolicyLossConfig


@dataclass(frozen=True)
class BehaviorGateObservation:
    protected_owner_ids: tuple[str, ...]
    jointly_coverable_protected_owner_ids: tuple[str, ...]
    unique_owner_ids: tuple[str, ...]
    cap_hit_count: int
    malformed_row_count: int
    row_count: int
    duplicate_count: int


@dataclass(frozen=True)
class BehaviorGateVerdict:
    accepted: bool
    reasons: tuple[str, ...]


def _sha256_tokens(tokens: Sequence[int]) -> str:
    encoded = b"".join(int(token).to_bytes(8, "big", signed=False) for token in tokens)
    return hashlib.sha256(encoded).hexdigest()


def _training_prefix(image: FrontierImage) -> tuple[tuple[int, ...], PrefixReceipt]:
    duplicate_orders = {
        event.duplicate_generated_order for event in image.duplicate_events
    }
    raw = tuple(image.generated_token_ids)
    terminal_removed = image.stop_reason == "im_end" and bool(raw)
    body_end = len(raw) - 1 if terminal_removed else len(raw)
    removed_positions: set[int] = set()
    for row in image.rows:
        if (
            row.token_start < 0
            or row.token_end <= row.token_start
            or row.token_end > body_end
        ):
            raise ValueError("frontier row span escapes the natural pre-terminal body")
        if tuple(raw[row.token_start : row.token_end]) != row.token_ids:
            raise ValueError(
                "frontier row span tokens differ from the natural trajectory"
            )
        if row.generated_order in duplicate_orders:
            removed_positions.update(range(row.token_start, row.token_end))
    kept = tuple(
        token
        for index, token in enumerate(raw[:body_end])
        if index not in removed_positions
    )
    return kept, PrefixReceipt(
        image_id=image.image_id,
        raw_natural_token_sha256=_sha256_tokens(raw),
        training_prefix_token_sha256=_sha256_tokens(kept),
        raw_natural_token_count=len(raw),
        training_prefix_token_count=len(kept),
        removed_duplicate_row_orders=tuple(sorted(duplicate_orders)),
        terminal_removed=terminal_removed,
    )


def _row_coordinate_offsets(tokens: tuple[int, ...]) -> tuple[int, int, int, int]:
    offsets = tuple(
        index
        for index, token in enumerate(tokens)
        if _COORD_TOKEN_START <= token < _COORD_TOKEN_END_EXCLUSIVE
    )
    if len(offsets) != 4:
        raise ValueError("selected native row lacks four canonical coordinate tokens")
    return offsets  # type: ignore[return-value]


def _segment(
    skeleton: Any,
    *,
    image_id: int,
    segment_id: str,
    role: base.LogicalRole,
    input_ids: tuple[int, ...],
    bindings: tuple[OnPolicyBinding, ...],
) -> base.LogicalPanelSegment:
    if is_dataclass(skeleton) and not isinstance(skeleton, type):
        encoded = replace(
            cast(Any, skeleton), example_id=segment_id, input_ids=input_ids
        )
        object.__setattr__(encoded, "human13_on_policy_bindings", bindings)
    else:
        values = dict(vars(skeleton))
        values.update(
            example_id=segment_id,
            input_ids=input_ids,
            human13_on_policy_bindings=bindings,
        )
        encoded = SimpleNamespace(**values)
    return base.LogicalPanelSegment(segment_id, image_id, role, encoded)


def materialize_on_policy_segments(
    frontier_images: Mapping[int, FrontierImage],
    skeletons: Mapping[int, Any],
    *,
    selected_scores: Mapping[int, CandidateScore],
    arm_id: ArmId,
    required_margin: float,
    global_max_length: int = base.GLOBAL_MAX_LENGTH,
) -> MaterializedOnPolicySegments:
    if arm_id not in {"O-Full-Safe", "O-First-Safe"}:
        raise ValueError("unknown on-policy arm")
    if required_margin <= 0:
        raise ValueError("on-policy treatment margin must be positive")
    if set(frontier_images) != set(skeletons) or not set(selected_scores) <= set(
        frontier_images
    ):
        raise ValueError("frontier, skeleton, and selected-score images differ")
    segments: list[base.LogicalPanelSegment] = []
    receipts: list[PrefixReceipt] = []
    for image_id in sorted(selected_scores):
        image = frontier_images[image_id]
        score = selected_scores[image_id]
        if (
            score.path.image_id != image_id
            or score.path.owner_id not in image.uncovered_h_owner_ids
        ):
            raise ValueError(
                "selected candidate is outside the current uncovered H frontier"
            )
        if score.path.token_ids != next(
            (
                alias.token_ids
                for alias in image.candidate_aliases
                if alias.owner_id == score.path.owner_id
                and alias.row_id == score.path.alias_id
            ),
            None,
        ):
            raise ValueError("selected candidate differs from the current native alias")
        skeleton = skeletons[image_id]
        prompt_count = int(skeleton.prompt_token_count)
        prompt = tuple(int(token) for token in skeleton.input_ids[:prompt_count])
        prefix, prefix_receipt = _training_prefix(image)
        receipts.append(prefix_receipt)
        row = tuple(score.path.token_ids)
        row_start = len(prompt) + len(prefix)
        if arm_id == "O-First-Safe":
            if score.first_bottleneck_index is None:
                raise ValueError("O-First-Safe candidate has no HF blocker")
            aligned_reserve = max(0.0, float(score.max_surface_margin_drift or 0.0))
            primary = OnPolicyBinding(
                "first_bottleneck",
                score.path.owner_id,
                score.path.owner_id,
                (row_start + score.first_bottleneck_index,),
                (row[score.first_bottleneck_index],),
                required_margin=required_margin + aligned_reserve,
            )
        else:
            primary = OnPolicyBinding(
                "full_row_ce",
                score.path.owner_id,
                score.path.owner_id,
                tuple(row_start + index for index in range(len(row))),
                row,
            )
        x1, y1, x2, y2 = _row_coordinate_offsets(row)
        rectangle = OnPolicyBinding(
            "rectangle",
            score.path.owner_id,
            score.path.owner_id,
            (row_start + x2, row_start + y2),
            (row[x2], row[y2]),
            lower_coordinate_token_ids=(row[x1], row[y1]),
        )
        segments.append(
            _segment(
                skeleton,
                image_id=image_id,
                segment_id=f"on-policy:{arm_id}:{image_id}:{score.path.owner_id}",
                role="h1_independent",
                input_ids=(*prompt, *prefix, *row),
                bindings=(primary, rectangle),
            )
        )
        row_by_order = {row.generated_order: row for row in image.rows}
        for event_index, event in enumerate(image.duplicate_events):
            duplicate_row = row_by_order[event.duplicate_generated_order]
            common = 0
            for left, right in zip(duplicate_row.token_ids, row, strict=False):
                if left != right:
                    break
                common += 1
            if common >= min(len(duplicate_row.token_ids), len(row)):
                raise ValueError("duplicate and selected rows have no differing token")
            decision_prefix = tuple(
                image.generated_token_ids[: duplicate_row.token_start]
            )
            duplicate_token = duplicate_row.token_ids[common]
            selected_token = row[common]
            token_index = len(prompt) + len(decision_prefix) + common
            binding = OnPolicyBinding(
                "duplicate_pair",
                f"dup:{image_id}:{event_index}",
                score.path.owner_id,
                (token_index,),
                (selected_token,),
                required_margin=required_margin,
                duplicate_token_id=duplicate_token,
            )
            segments.append(
                _segment(
                    skeleton,
                    image_id=image_id,
                    segment_id=f"on-policy:duplicate:{image_id}:{event_index}",
                    role="duplicate_event",
                    input_ids=(*prompt, *decision_prefix, *row[: common + 1]),
                    bindings=(binding,),
                )
            )
    result = MaterializedOnPolicySegments(arm_id, tuple(segments), tuple(receipts))
    result.preflight(global_max_length)
    return result


def _sites_for_pack(item: base.PackedPanelMicroStep) -> tuple[OnPolicyLossSite, ...]:
    packed_by_id = {segment.example_id: segment for segment in item.pack.segments}
    sites: list[OnPolicyLossSite] = []
    for logical in item.logical_segments:
        packed = packed_by_id[logical.segment_id]
        input_ids = tuple(logical.encoded_example.input_ids)
        bindings = getattr(logical.encoded_example, "human13_on_policy_bindings", ())
        if not bindings:
            raise ValueError("on-policy segment lacks typed bindings")
        for binding in bindings:
            if any(
                index <= 0 or index >= len(input_ids) for index in binding.token_indices
            ):
                raise ValueError("on-policy target index escapes its segment")
            expected = tuple(input_ids[index] for index in binding.token_indices)
            if expected != binding.target_token_ids:
                raise ValueError("on-policy binding target tokens drifted")
            sites.append(
                OnPolicyLossSite(
                    binding.objective,
                    binding.unit_id,
                    binding.owner_id,
                    logical.image_id,
                    logical.segment_id,
                    tuple(packed.start + index - 1 for index in binding.token_indices),
                    binding.target_token_ids,
                    binding.required_margin,
                    binding.lower_coordinate_token_ids,
                    binding.duplicate_token_id,
                )
            )
    return tuple(sites)


def _denominators(
    sites_by_pack: Mapping[int, Sequence[OnPolicyLossSite]],
) -> OnPolicyDenominators:
    sites = tuple(site for group in sites_by_pack.values() for site in group)
    positives = {
        site.owner_id
        for site in sites
        if site.objective in {"full_row_ce", "first_bottleneck"}
    }
    rectangles = {site.unit_id for site in sites if site.objective == "rectangle"}
    duplicate_event_counts: dict[int, int] = {}
    for site in sites:
        if site.objective == "duplicate_pair":
            duplicate_event_counts[site.image_id] = (
                duplicate_event_counts.get(site.image_id, 0) + 1
            )
    if not positives or not rectangles:
        raise ValueError("on-policy payload requires positive and rectangle sites")
    return OnPolicyDenominators(
        len(positives),
        len(rectangles),
        len(duplicate_event_counts),
        tuple(sorted(duplicate_event_counts.items())),
    )


def build_on_policy_payload(
    materialized: MaterializedOnPolicySegments,
    *,
    arm_id: ArmId,
    expected_vocab_size: int,
    vocab_groups: Any,
    global_max_length: int,
    required_margin: float,
    rectangle_margin: float,
    duplicate_margin: float,
) -> OnPolicyPayload:
    loss_config = OnPolicyLossConfig(
        float(required_margin), float(rectangle_margin), float(duplicate_margin)
    )
    if any(
        value <= 0
        for value in (
            loss_config.required_margin,
            loss_config.rectangle_margin,
            loss_config.duplicate_margin,
        )
    ):
        raise ValueError("on-policy loss margins must be positive")
    if materialized.arm_id != arm_id:
        raise ValueError("on-policy materialized arm differs")
    materialized.preflight(global_max_length)
    packed = base.plan_panel_packs(
        materialized.segments, global_max_length=global_max_length
    )
    sites_by_pack = {
        item.pack.pack_index: _sites_for_pack(item) for item in packed.packs
    }
    denominators = _denominators(sites_by_pack)
    steps: list[SupervisedMicroStep] = []
    for item in packed.packs:
        sites = sites_by_pack[item.pack.pack_index]
        positions = tuple(
            sorted({position for site in sites for position in site.logits_positions})
        )
        steps.append(
            SupervisedMicroStep(
                pack=item.pack,
                encoded_examples=item.encoded_examples,
                position_inputs=item.position_inputs,
                token_sequence=build_token_sequence_from_packed_supervision(
                    item.pack, ()
                ),
                vocab_groups=vocab_groups,
                metadata={
                    "human13_on_policy_sites": sites,
                    "human13_on_policy_denominators": denominators,
                    "human13_on_policy_loss_config": loss_config,
                },
                expected_vocab_size=expected_vocab_size,
                calibration_metadata=base.Human13CompactLogitsMetadata(positions),
            )
        )
    return OnPolicyPayload(
        arm_id,
        materialized.segments,
        packed,
        sites_by_pack,
        tuple(steps),
        denominators,
        materialized.prefix_receipts,
        loss_config,
    )


def on_policy_loss_context_factory(
    micro_step: SupervisedMicroStep, forward_result: Any
) -> OnPolicyPackContext:
    sites = (micro_step.metadata or {}).get("human13_on_policy_sites")
    if not isinstance(sites, tuple) or not sites:
        raise ValueError("on-policy micro-step lacks typed sites")
    return OnPolicyPackContext(
        forward_result.logits, forward_result.logits_position_ids, sites
    )


def _selected_logits(
    context: OnPolicyPackContext, site: OnPolicyLossSite
) -> torch.Tensor:
    if context.logits_position_ids is None:
        rows = site.logits_positions
    else:
        lookup = {
            position: index
            for index, position in enumerate(context.logits_position_ids)
        }
        if any(position not in lookup for position in site.logits_positions):
            raise ValueError("compact logits omit an on-policy site")
        rows = tuple(lookup[position] for position in site.logits_positions)
    return (
        context.logits[0]
        .index_select(
            0, torch.tensor(rows, dtype=torch.long, device=context.logits.device)
        )
        .float()
    )


@dataclass(frozen=True)
class OnPolicyLossRunner:
    denominators: OnPolicyDenominators
    required_margin: float
    rectangle_margin: float
    duplicate_margin: float

    @property
    def loss_config(self) -> OnPolicyLossConfig:
        return OnPolicyLossConfig(
            float(self.required_margin),
            float(self.rectangle_margin),
            float(self.duplicate_margin),
        )

    def prepare_planned_step(
        self,
        micro_steps: Sequence[SupervisedMicroStep],
        *,
        denominator_gatherer: Any | None = None,
        world_size: int = 1,
        rank: int = 0,
    ) -> OnPolicyLossPlan:
        del denominator_gatherer, rank
        if world_size != 1 or not micro_steps:
            raise ValueError("on-policy step requires world-size one physical packs")
        configs = {
            (step.metadata or {}).get("human13_on_policy_loss_config")
            for step in micro_steps
        }
        if configs != {self.loss_config}:
            raise ValueError("on-policy runner loss configuration differs from payload")
        return OnPolicyLossPlan(self.denominators, len(micro_steps), self.loss_config)

    def compute_micro_step(
        self,
        context: OnPolicyPackContext,
        plan: OnPolicyLossPlan,
        *,
        local_micro_step_index: int,
    ) -> LossBundle:
        del local_micro_step_index
        zero = context.logits.sum() * 0.0
        numerators = {name: zero for name in ("positive", "rectangle", "duplicate")}
        counts = {name: 0 for name in numerators}
        for site in context.sites:
            logits = _selected_logits(context, site)
            targets = torch.tensor(site.target_token_ids, device=logits.device)
            if site.objective == "full_row_ce":
                result = owner_mean_masked_row_cross_entropy(
                    logits.unsqueeze(0),
                    targets.unsqueeze(0),
                    torch.ones_like(targets, dtype=torch.bool).unsqueeze(0),
                )
                numerators["positive"] = numerators["positive"] + result.numerator
                counts["positive"] += len(targets)
            elif site.objective == "first_bottleneck":
                result = first_bottleneck_argmax_hinge(
                    logits,
                    targets,
                    torch.ones_like(targets, dtype=torch.bool),
                    required_margin=float(site.required_margin or self.required_margin),
                )
                numerators["positive"] = numerators["positive"] + result.numerator
                counts["positive"] += 1
            elif site.objective == "rectangle":
                valid = torch.zeros_like(logits, dtype=torch.bool)
                for row_index, lower in enumerate(site.lower_coordinate_token_ids):
                    if not _COORD_TOKEN_START <= lower < _COORD_TOKEN_END_EXCLUSIVE:
                        raise ValueError(
                            "rectangle lower coordinate is outside vocabulary"
                        )
                    valid_end = min(int(logits.shape[-1]), _COORD_TOKEN_END_EXCLUSIVE)
                    if lower + 1 < valid_end:
                        valid[row_index, lower + 1 : valid_end] = True
                result = rectangle_valid_argmax_hinge(
                    logits,
                    valid,
                    row_ids=torch.zeros(
                        len(targets), dtype=torch.long, device=logits.device
                    ),
                    required_margin=self.rectangle_margin,
                )
                numerators["rectangle"] = numerators["rectangle"] + result.numerator
                counts["rectangle"] += len(targets)
            elif site.objective == "duplicate_pair":
                if site.duplicate_token_id is None:
                    raise ValueError("duplicate pair lacks its rejected token")
                result = image_balanced_duplicate_token_contrast(
                    logits,
                    duplicate_token_ids=torch.tensor(
                        (site.duplicate_token_id,), device=logits.device
                    ),
                    selected_token_ids=targets,
                    image_ids=torch.tensor((site.image_id,), device=logits.device),
                    required_margin=self.duplicate_margin,
                )
                event_count = dict(plan.denominators.duplicate_events_by_image).get(
                    site.image_id
                )
                if event_count is None or event_count <= 0:
                    raise ValueError("duplicate event lacks its image denominator")
                numerators["duplicate"] = (
                    numerators["duplicate"] + result.numerator / event_count
                )
                counts["duplicate"] += 1
        denominator_values = {
            "positive": plan.denominators.positive_owners,
            "rectangle": plan.denominators.rectangle_rows,
            "duplicate": plan.denominators.duplicate_images,
        }
        terms: list[LossTermResult] = []
        for name in ("positive", "rectangle", "duplicate"):
            if counts[name] == 0:
                continue
            denominator = denominator_values[name]
            raw = numerators[name] / denominator
            terms.append(
                LossTermResult(
                    name=name,
                    raw_loss=raw,
                    weighted_loss=raw,
                    weight=1.0,
                    segment_mean_numerator=numerators[name],
                    denominator=SegmentBalancedDenominator(
                        term_name=name,
                        denominator_scope="planned_step",
                        eligible_segment_count=denominator,
                        selected_atom_count=counts[name],
                        skipped_segment_count=0,
                        context_count=plan.micro_step_count,
                    ),
                    reducer_name="human13_on_policy_complete_panel",
                    selected_count=counts[name],
                    skipped_count=0,
                    math_dtype="float32",
                    token_weighted_diagnostic=raw.detach(),
                    diagnostics={},
                )
            )
        total = sum((term.weighted_loss for term in terms), zero)
        finite = bool(torch.isfinite(total).item())
        return LossBundle(
            total_loss=total,
            terms=tuple(terms),
            metrics={"loss/total": float(total.detach())},
            counts={"count/sites": sum(counts.values())},
            diagnostics={},
            finite_status={"total_loss": "finite" if finite else "nonfinite"},
        )

    def finalize_planned_step(
        self, micro_loss_artifacts: Sequence[Mapping[str, Any]], plan: OnPolicyLossPlan
    ) -> dict[str, Any]:
        return {
            "total_loss": sum(
                float(item["total_loss"]) for item in micro_loss_artifacts
            ),
            "micro_step_count": plan.micro_step_count,
        }


def evaluate_behavior_gate(
    prior: BehaviorGateObservation, proposed: BehaviorGateObservation
) -> BehaviorGateVerdict:
    reasons: list[str] = []
    protected = set(prior.protected_owner_ids)
    if not protected <= set(proposed.jointly_coverable_protected_owner_ids):
        reasons.append("protected_owner_loss")
    if len(proposed.unique_owner_ids) < len(prior.unique_owner_ids):
        reasons.append("unique_owner_decrease")
    if proposed.cap_hit_count:
        reasons.append("cap_hit")
    if proposed.malformed_row_count > prior.malformed_row_count:
        reasons.append("malformed_increase")
    if proposed.row_count > 1.2 * prior.row_count:
        reasons.append("row_burden")
    if proposed.duplicate_count > prior.duplicate_count + 2:
        reasons.append("duplicate_burden")
    return BehaviorGateVerdict(not reasons, tuple(reasons))


__all__ = [
    "BehaviorGateObservation",
    "BehaviorGateVerdict",
    "MaterializedOnPolicySegments",
    "OnPolicyLossConfig",
    "OnPolicyLossRunner",
    "OnPolicyPackContext",
    "OnPolicyPayload",
    "build_on_policy_payload",
    "evaluate_behavior_gate",
    "materialize_on_policy_segments",
    "on_policy_loss_context_factory",
]
