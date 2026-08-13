"""Packed R1/R2 payload and exact two-pass loss for the Human-13 successor."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass, replace
import math
from types import SimpleNamespace
from typing import Any, Literal, Mapping, Sequence

import torch
import torch.nn.functional as F

from scripts.research.build_human13_row_contrast_successor import (
    COORD_TOKEN_END_EXCLUSIVE,
    COORD_TOKEN_START,
    Human13RowContrastLedger,
    SuccessorRow,
    _image_context,
)
from scripts.research import run_human13_k_union_overfit as base
from scripts.research.human13_gradient_preservation import (
    GradientProjectionError,
    project_and_write_gradients,
)
from src.losses.human13_k_union import (
    image_balanced_four_coordinate_unlikelihood,
    rectangle_valid_argmax_hinge,
)
from src.losses.normalizers import SegmentBalancedDenominator
from src.losses.runner import LossBundle, LossTermResult
from src.supervision import build_token_sequence_from_packed_supervision
from src.training.supervised_trainer import SupervisedMicroStep


SuccessorObjective = Literal[
    "union_candidate",
    "replay_ce",
    "row_duplicate",
    "row_candidate",
    "fallback_ul",
    "rectangle",
    "watch_ce",
]


@dataclass(frozen=True)
class SuccessorBinding:
    objective: SuccessorObjective
    unit_id: str
    row_id: str
    event_id: str | None
    owner_id: str | None
    token_indices: tuple[int, ...]
    lower_coordinate_token_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class MaterializedSuccessorSegments:
    segments: tuple[base.LogicalPanelSegment, ...]
    ledger_manifest_sha256: str

    def preflight(self, global_max_length: int = base.GLOBAL_MAX_LENGTH) -> None:
        for segment in self.segments:
            if segment.encoded_length > global_max_length:
                raise ValueError(
                    f"successor segment {segment.segment_id} length "
                    f"{segment.encoded_length} exceeds {global_max_length}"
                )


@dataclass(frozen=True)
class SuccessorLossSite:
    objective: SuccessorObjective
    unit_id: str
    row_id: str
    event_id: str | None
    owner_id: str | None
    image_id: int
    segment_id: str
    logits_positions: tuple[int, ...]
    target_token_ids: tuple[int, ...]
    lower_coordinate_token_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class SuccessorDenominators:
    union_images: int
    replay_owners: int
    contrast_images: int
    contrast_events_by_image: tuple[tuple[int, int], ...]
    fallback_images: int
    fallback_events_by_image: tuple[tuple[int, int], ...]
    rectangle_rows: int
    watch_owners: int


@dataclass(frozen=True)
class SuccessorPayload:
    arm_id: Literal["R1", "R2"]
    selected_segments: tuple[base.LogicalPanelSegment, ...]
    packed_plan: base.PackedPanelPlan
    sites_by_pack: dict[int, tuple[SuccessorLossSite, ...]]
    micro_steps: tuple[SupervisedMicroStep, ...]
    denominators: SuccessorDenominators
    duplicate_margin: float
    rectangle_margin: float
    gradient_projection: bool


@dataclass(frozen=True)
class SuccessorPackContext:
    logits: torch.Tensor
    logits_position_ids: tuple[int, ...] | None
    sites: tuple[SuccessorLossSite, ...]


@dataclass(frozen=True)
class WeightedScore:
    segment_id: str
    coefficient: float


@dataclass(frozen=True)
class SuccessorLossPlan:
    denominators: SuccessorDenominators
    micro_step_count: int
    union_weights: tuple[WeightedScore, ...]
    union_reference_nll: tuple[tuple[int, float], ...]
    contrast_weights: tuple[WeightedScore, ...]
    contrast_reference_losses: tuple[tuple[str, float], ...]
    score_forward_count: int
    parameter_versions: tuple[tuple[int, int], ...]


def materialize_successor_segments(
    ledger: Human13RowContrastLedger,
    skeletons: Mapping[int, Any],
    *,
    manifest: Any | None = None,
    global_max_length: int = base.GLOBAL_MAX_LENGTH,
) -> MaterializedSuccessorSegments:
    """Materialize only branch-relevant rows under their exact prefixes."""

    if not skeletons:
        raise ValueError("successor materialization requires processor skeletons")
    positives_by_image: dict[int, list[Any]] = {}
    for positive in ledger.positive_rows:
        image_id = _row_image_id(positive.owner_id)
        positives_by_image.setdefault(image_id, []).append(positive)
    watches_by_image: dict[int, list[Any]] = {}
    for positive in ledger.g_watch_rows:
        watches_by_image.setdefault(_row_image_id(positive.owner_id), []).append(
            positive
        )

    segments: list[base.LogicalPanelSegment] = []
    canonical_by_image: dict[int, tuple[base.LogicalPanelSegment, ...]] = {}
    canonical_rows_by_image: dict[int, Mapping[str, SuccessorRow]] = {}
    if manifest is not None:
        from scripts.research.human13_live_segments import materialize_segments

        canonical = materialize_segments(
            manifest,
            skeletons,
            global_max_length=global_max_length,
        )
        canonical_rows_by_image = {
            image.image_id: _image_context(image).row_by_id for image in manifest.images
        }
        for segment in canonical.segments:
            if segment.role in {"a4_union", "source_replay"}:
                canonical_by_image.setdefault(segment.image_id, ())
                canonical_by_image[segment.image_id] = (
                    *canonical_by_image[segment.image_id],
                    segment,
                )
    for image_id, skeleton in sorted(skeletons.items()):
        prompt_count = int(getattr(skeleton, "prompt_token_count"))
        prompt = tuple(int(x) for x in skeleton.input_ids[:prompt_count])
        if len(prompt) != prompt_count:
            raise ValueError(f"invalid prompt token count for image {image_id}")

        if canonical_by_image.get(image_id):
            row_by_id = canonical_rows_by_image[image_id]
            for canonical in canonical_by_image[image_id]:
                bindings: list[SuccessorBinding] = []
                for old in canonical.encoded_example.human13_row_bindings:
                    row = row_by_id.get(old.manifest_row_id)
                    if row is None:
                        raise ValueError(
                            f"canonical segment row {old.manifest_row_id} is absent "
                            "from the successor ledger"
                        )
                    selected = tuple(
                        old.token_start + offset
                        for offset, enabled in enumerate(old.target_token_mask)
                        if enabled
                    )
                    if canonical.role == "a4_union":
                        bindings.append(
                            SuccessorBinding(
                                "union_candidate",
                                old.unit_id,
                                row.row_id,
                                None,
                                old.unit_id,
                                selected,
                            )
                        )
                    else:
                        bindings.append(
                            SuccessorBinding(
                                "replay_ce",
                                old.unit_id,
                                row.row_id,
                                None,
                                old.unit_id,
                                selected,
                            )
                        )
                        bindings.append(
                            SuccessorBinding(
                                "watch_ce",
                                old.unit_id,
                                row.row_id,
                                None,
                                old.unit_id,
                                tuple(
                                    old.token_start + offset
                                    for offset in row.coordinate_offsets
                                ),
                            )
                        )
                    bindings.append(
                        _rectangle_binding(row, old.token_start, unit_id=old.unit_id)
                    )
                object.__setattr__(
                    canonical.encoded_example,
                    "human13_successor_bindings",
                    tuple(bindings),
                )
                segments.append(canonical)
        else:
            # Test-only/synthetic fallback. Real execution always supplies the
            # sealed manifest and therefore reuses exact A4/replay materialization.
            for positive in positives_by_image.get(image_id, ()):
                row = positive.row
                start = len(prompt)
                if positive.stratum == "H":
                    segments.append(
                        _segment(
                            skeleton,
                            image_id,
                            "a4_union",
                            f"successor:{image_id}:union:{positive.owner_id}",
                            prompt + row.token_ids,
                            (
                                SuccessorBinding(
                                    "union_candidate",
                                    positive.owner_id,
                                    row.row_id,
                                    None,
                                    positive.owner_id,
                                    tuple(
                                        start + index
                                        for index in range(len(row.token_ids))
                                    ),
                                ),
                                _rectangle_binding(
                                    row, start, unit_id=positive.owner_id
                                ),
                            ),
                        )
                    )
                elif positive.stratum == "G":
                    segments.append(
                        _segment(
                            skeleton,
                            image_id,
                            "source_replay",
                            f"successor:{image_id}:replay:{positive.owner_id}",
                            prompt + row.token_ids,
                            (
                                SuccessorBinding(
                                    "replay_ce",
                                    positive.owner_id,
                                    row.row_id,
                                    None,
                                    positive.owner_id,
                                    tuple(
                                        start + index
                                        for index in range(len(row.token_ids))
                                    ),
                                ),
                                _rectangle_binding(
                                    row, start, unit_id=positive.owner_id
                                ),
                                SuccessorBinding(
                                    "watch_ce",
                                    positive.owner_id,
                                    row.row_id,
                                    None,
                                    positive.owner_id,
                                    tuple(
                                        start + index
                                        for index in row.coordinate_offsets
                                    ),
                                ),
                            ),
                        )
                    )

        for event in (item for item in ledger.events if item.image_id == image_id):
            duplicate = event.duplicate_row
            start = len(prompt) + len(event.prefix_token_ids)
            if event.contrast_branch == "fallback":
                duplicate_binding = SuccessorBinding(
                    "fallback_ul",
                    event.event_id,
                    duplicate.row_id,
                    event.event_id,
                    None,
                    tuple(start + index for index in duplicate.coordinate_offsets),
                )
            else:
                duplicate_binding = SuccessorBinding(
                    "row_duplicate",
                    event.event_id,
                    duplicate.row_id,
                    event.event_id,
                    None,
                    _score_indices(duplicate, start, event.contrast_branch),
                )
            segments.append(
                _segment(
                    skeleton,
                    image_id,
                    "duplicate_event",
                    f"successor:{image_id}:event:{_short_id(event.event_id)}:duplicate",
                    prompt + event.prefix_token_ids + duplicate.token_ids,
                    (duplicate_binding,),
                )
            )
            if event.contrast_branch == "fallback":
                continue
            normalized_duplicate = _normalize(duplicate.category)
            for group in event.candidate_groups:
                if (
                    event.contrast_branch == "same_description"
                    and _normalize(group.category) != normalized_duplicate
                ):
                    continue
                for alias_index, row in enumerate(group.rows):
                    row_start = len(prompt) + len(event.prefix_token_ids)
                    binding = SuccessorBinding(
                        "row_candidate",
                        event.event_id,
                        row.row_id,
                        event.event_id,
                        group.owner_id,
                        _score_indices(row, row_start, event.contrast_branch),
                    )
                    segments.append(
                        _segment(
                            skeleton,
                            image_id,
                            "duplicate_event",
                            (
                                f"successor:{image_id}:event:{_short_id(event.event_id)}:"
                                f"candidate:{group.owner_id}:{alias_index}"
                            ),
                            prompt + event.prefix_token_ids + row.token_ids,
                            (binding,),
                        )
                    )
    result = MaterializedSuccessorSegments(tuple(segments), ledger.manifest_sha256)
    result.preflight(global_max_length)
    if not result.segments:
        raise ValueError("successor materialization emitted no segments")
    if len({segment.segment_id for segment in result.segments}) != len(result.segments):
        raise ValueError("successor materialization emitted duplicate segment IDs")
    return result


def build_successor_payload(
    materialized: MaterializedSuccessorSegments,
    *,
    arm_id: str,
    expected_vocab_size: int,
    vocab_groups: Any,
    global_max_length: int,
    duplicate_margin: float,
    rectangle_margin: float,
) -> SuccessorPayload:
    if arm_id not in {"R1", "R2"}:
        raise ValueError("successor payload admits only R1/R2")
    materialized.preflight(global_max_length)
    packed = base.plan_panel_packs(
        materialized.segments, global_max_length=global_max_length
    )
    sites_by_pack = {
        item.pack.pack_index: _sites_for_pack(item) for item in packed.packs
    }
    denominators = _denominators(sites_by_pack)
    micro_steps: list[SupervisedMicroStep] = []
    for item in packed.packs:
        pack_index = item.pack.pack_index
        sites = sites_by_pack[pack_index]
        positions = tuple(
            sorted({position for site in sites for position in site.logits_positions})
        )
        sequence = build_token_sequence_from_packed_supervision(item.pack, ())
        micro_steps.append(
            SupervisedMicroStep(
                pack=item.pack,
                encoded_examples=item.encoded_examples,
                position_inputs=item.position_inputs,
                token_sequence=sequence,
                vocab_groups=vocab_groups,
                metadata={
                    "human13_successor_sites": sites,
                    "human13_successor_denominators": denominators,
                },
                expected_vocab_size=expected_vocab_size,
                calibration_metadata=base.Human13CompactLogitsMetadata(positions),
            )
        )
    return SuccessorPayload(
        arm_id=arm_id,
        selected_segments=materialized.segments,
        packed_plan=packed,
        sites_by_pack=sites_by_pack,
        micro_steps=tuple(micro_steps),
        denominators=denominators,
        duplicate_margin=float(duplicate_margin),
        rectangle_margin=float(rectangle_margin),
        gradient_projection=arm_id == "R2",
    )


def successor_loss_context_factory(
    micro_step: SupervisedMicroStep, forward_result: Any
) -> SuccessorPackContext:
    sites = (micro_step.metadata or {}).get("human13_successor_sites")
    if not isinstance(sites, tuple) or not sites:
        raise ValueError("successor micro-step has no typed loss sites")
    return SuccessorPackContext(
        logits=forward_result.logits,
        logits_position_ids=forward_result.logits_position_ids,
        sites=sites,
    )


@dataclass(frozen=True)
class SuccessorLossRunner:
    arm_id: str
    denominators: SuccessorDenominators
    model: Any
    score_forward: Any
    duplicate_margin: float
    rectangle_margin: float

    def __post_init__(self) -> None:
        if self.arm_id not in {"R1", "R2"}:
            raise ValueError("successor loss runner admits only R1/R2")

    def prepare_planned_step(
        self,
        micro_steps: Sequence[SupervisedMicroStep],
        *,
        denominator_gatherer: Any | None = None,
        world_size: int = 1,
        rank: int = 0,
    ) -> SuccessorLossPlan:
        del denominator_gatherer, rank
        if world_size != 1:
            raise ValueError("successor training remains world-size one")
        checked = tuple(micro_steps)
        if not checked:
            raise ValueError("successor planned step requires physical packs")
        before = _parameter_versions(self.model)
        scores: dict[str, tuple[SuccessorLossSite, float]] = {}
        with torch.no_grad():
            for micro_step in checked:
                result = self.score_forward(self.model, micro_step)
                context = successor_loss_context_factory(micro_step, result)
                for site in context.sites:
                    if site.objective not in {
                        "union_candidate",
                        "row_duplicate",
                        "row_candidate",
                    }:
                        continue
                    if site.segment_id in scores:
                        raise ValueError("score pass segment identity is repeated")
                    scores[site.segment_id] = (
                        site,
                        float(_score_for_site(context, site).item()),
                    )
                del context, result
        after = _parameter_versions(self.model)
        if before != after:
            raise ValueError("successor parameters changed during score pass")

        union_weights: list[WeightedScore] = []
        union_reference: list[tuple[int, float]] = []
        union_sites = [
            item for item in scores.values() if item[0].objective == "union_candidate"
        ]
        for image_id in sorted({site.image_id for site, _score in union_sites}):
            group = sorted(
                (
                    (site, score)
                    for site, score in union_sites
                    if site.image_id == image_id
                ),
                key=lambda item: item[0].segment_id,
            )
            values = torch.tensor(
                [score for _site, score in group], dtype=torch.float32
            )
            weights = torch.softmax(values, dim=0)
            union_reference.append(
                (image_id, float((-torch.logsumexp(values, 0)).item()))
            )
            union_weights.extend(
                WeightedScore(site.segment_id, float(weight))
                for (site, _score), weight in zip(group, weights.tolist(), strict=True)
            )

        contrast_weights: list[WeightedScore] = []
        contrast_reference: list[tuple[str, float]] = []
        duplicate_sites = [
            (site, score)
            for site, score in scores.values()
            if site.objective == "row_duplicate"
        ]
        for duplicate_site, duplicate_score in sorted(
            duplicate_sites, key=lambda item: item[0].event_id or ""
        ):
            event_id = duplicate_site.event_id
            candidates = sorted(
                (
                    (site, score)
                    for site, score in scores.values()
                    if site.objective == "row_candidate" and site.event_id == event_id
                ),
                key=lambda item: (item[0].owner_id or "", item[0].segment_id),
            )
            if not candidates:
                raise ValueError(f"row contrast event {event_id} has no candidate")
            owner_groups: dict[str, list[tuple[SuccessorLossSite, float]]] = {}
            for site, score in candidates:
                owner_groups.setdefault(str(site.owner_id), []).append((site, score))
            owner_scores: list[tuple[str, float]] = []
            alias_weights: dict[str, list[tuple[SuccessorLossSite, float]]] = {}
            for owner_id, aliases in sorted(owner_groups.items()):
                values = torch.tensor(
                    [score for _site, score in aliases], dtype=torch.float32
                )
                weights = torch.softmax(values, dim=0)
                score = float(
                    (torch.logsumexp(values, 0) - math.log(len(aliases))).item()
                )
                owner_scores.append((owner_id, score))
                alias_weights[owner_id] = list(
                    zip((site for site, _ in aliases), weights.tolist(), strict=True)
                )
            owner_values = torch.tensor(
                [score for _owner, score in owner_scores], dtype=torch.float32
            )
            owner_probs = torch.softmax(owner_values, dim=0)
            valid_mass = torch.logsumexp(owner_values, 0)
            argument = (
                torch.tensor(self.duplicate_margin + duplicate_score) - valid_mass
            )
            slope = float(torch.sigmoid(argument).item())
            reference = float(F.softplus(argument).item())
            contrast_reference.append((str(event_id), reference))
            event_count = dict(self.denominators.contrast_events_by_image)[
                duplicate_site.image_id
            ]
            slope /= event_count
            contrast_weights.append(WeightedScore(duplicate_site.segment_id, slope))
            for (owner_id, _score), owner_weight in zip(
                owner_scores, owner_probs.tolist(), strict=True
            ):
                for alias_site, alias_weight in alias_weights[owner_id]:
                    contrast_weights.append(
                        WeightedScore(
                            alias_site.segment_id,
                            -slope * float(owner_weight) * float(alias_weight),
                        )
                    )
        return SuccessorLossPlan(
            denominators=self.denominators,
            micro_step_count=len(checked),
            union_weights=tuple(union_weights),
            union_reference_nll=tuple(union_reference),
            contrast_weights=tuple(contrast_weights),
            contrast_reference_losses=tuple(contrast_reference),
            score_forward_count=len(checked),
            parameter_versions=before,
        )

    def compute_micro_step(
        self,
        context: SuccessorPackContext,
        plan: SuccessorLossPlan,
        *,
        local_micro_step_index: int,
    ) -> LossBundle:
        del local_micro_step_index
        if _parameter_versions(self.model) != plan.parameter_versions:
            raise ValueError("successor parameter state changed before replay")
        union_map = {item.segment_id: item.coefficient for item in plan.union_weights}
        contrast_map = {
            item.segment_id: item.coefficient for item in plan.contrast_weights
        }
        zero = context.logits.sum() * 0.0
        numerators: dict[str, torch.Tensor] = {
            "union": zero,
            "replay": zero,
            "row_contrast": zero,
            "fallback": zero,
            "rectangle": zero,
        }
        selected_counts = {name: 0 for name in numerators}
        for site in context.sites:
            if site.objective == "union_candidate":
                numerators["union"] = numerators["union"] - union_map[
                    site.segment_id
                ] * _score_for_site(context, site)
                selected_counts["union"] += len(site.logits_positions)
            elif site.objective == "replay_ce":
                numerators["replay"] = numerators["replay"] + _mean_ce(context, site)
                selected_counts["replay"] += len(site.logits_positions)
            elif site.objective in {"row_duplicate", "row_candidate"}:
                numerators["row_contrast"] = numerators["row_contrast"] + contrast_map[
                    site.segment_id
                ] * _score_for_site(context, site)
                selected_counts["row_contrast"] += len(site.logits_positions)
            elif site.objective == "fallback_ul":
                selected = _selected_logits(context, site).unsqueeze(0)
                targets = torch.tensor(
                    site.target_token_ids, device=selected.device
                ).unsqueeze(0)
                result = image_balanced_four_coordinate_unlikelihood(
                    selected,
                    targets,
                    torch.tensor([site.image_id], device=selected.device),
                )
                count = dict(plan.denominators.fallback_events_by_image)[site.image_id]
                numerators["fallback"] = (
                    numerators["fallback"] + result.raw_loss / count
                )
                selected_counts["fallback"] += 4
            elif site.objective == "rectangle":
                selected = _selected_logits(context, site)
                valid = _rectangle_masks(
                    selected.shape[-1], site.lower_coordinate_token_ids, selected.device
                )
                result = rectangle_valid_argmax_hinge(
                    selected,
                    valid,
                    row_ids=torch.zeros(
                        len(site.logits_positions),
                        dtype=torch.long,
                        device=selected.device,
                    ),
                    required_margin=self.rectangle_margin,
                )
                numerators["rectangle"] = numerators["rectangle"] + result.raw_loss
                selected_counts["rectangle"] += len(site.logits_positions)

        denominators = {
            "union": plan.denominators.union_images,
            "replay": plan.denominators.replay_owners,
            "row_contrast": plan.denominators.contrast_images,
            "fallback": plan.denominators.fallback_images,
            "rectangle": plan.denominators.rectangle_rows,
        }
        terms: list[LossTermResult] = []
        for name in ("union", "replay", "row_contrast", "fallback", "rectangle"):
            denominator = denominators[name]
            if denominator <= 0 or selected_counts[name] == 0:
                continue
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
                        selected_atom_count=selected_counts[name],
                        skipped_segment_count=0,
                        context_count=plan.micro_step_count,
                    ),
                    reducer_name="human13_successor_complete_panel",
                    selected_count=selected_counts[name],
                    skipped_count=0,
                    math_dtype="float32",
                    token_weighted_diagnostic=raw.detach(),
                    diagnostics={"two_pass_exact": name in {"union", "row_contrast"}},
                )
            )
        total = sum((term.weighted_loss for term in terms), zero)
        finite = bool(torch.isfinite(total).item())
        return LossBundle(
            total_loss=total,
            terms=tuple(terms),
            metrics={"loss/total": float(total.detach())},
            counts={"count/packs": 1, "count/sites": sum(selected_counts.values())},
            diagnostics={"two_pass_exact": True},
            finite_status={"total_loss": "finite" if finite else "nonfinite"},
        )

    def finalize_planned_step(
        self,
        micro_loss_artifacts: Sequence[Mapping[str, Any]],
        plan: SuccessorLossPlan,
    ) -> dict[str, Any]:
        artifacts = tuple(micro_loss_artifacts)
        names = ("union", "replay", "row_contrast", "fallback", "rectangle")
        terms = []
        for name in names:
            found = [
                term
                for artifact in artifacts
                for term in artifact.get("terms", ())
                if term.get("name") == name
            ]
            if not found:
                continue
            representative = dict(found[0])
            for field in ("raw_loss", "weighted_loss", "segment_mean_numerator"):
                representative[field] = sum(float(item[field]) for item in found)
            representative["selected_count"] = sum(
                int(item["selected_count"]) for item in found
            )
            terms.append(representative)
        return {
            "total_loss": sum(float(item["total_loss"]) for item in artifacts),
            "micro_step_count": len(artifacts),
            "terms": terms,
            "two_pass_exact": True,
            "score_forward_count": plan.score_forward_count,
            "replay_forward_count": plan.micro_step_count,
            "union": {
                "candidate_count": len(plan.union_weights),
                "reference_nll_by_image": {
                    str(image_id): value for image_id, value in plan.union_reference_nll
                },
            },
            "row_contrast": {
                "event_count": len(plan.contrast_reference_losses),
                "reference_losses": dict(plan.contrast_reference_losses),
            },
        }


def watch_loss_for_context(
    context: SuccessorPackContext, denominators: SuccessorDenominators
) -> torch.Tensor:
    total = context.logits.sum() * 0.0
    for site in context.sites:
        if site.objective == "watch_ce":
            total = total + _mean_ce(context, site) / denominators.watch_owners
    return total


def build_gradient_projection_handler(
    payload: SuccessorPayload,
    *,
    epsilon: float,
    tolerance: float,
):
    """Return the R2 complete-panel gradient projection hook.

    The trainer invokes this exactly once after the complete R1 gradient has
    been accumulated and admitted, and before clipping/AdamW.  The callback
    clears those gradients only after copying them, replays only G-coordinate
    watch sites at the same parameter state, then writes the projected full
    gradient back to the trainable surface.
    """

    if not payload.gradient_projection or payload.arm_id != "R2":
        raise ValueError("gradient projection handler requires an R2 payload")

    def handler(
        qwen_forward: Any,
        model: Any,
        micro_steps: Sequence[SupervisedMicroStep],
        plan: SuccessorLossPlan,
        runtime: Any,
        planned_step_id: int,
    ) -> Mapping[str, Any]:
        if plan.denominators != payload.denominators:
            raise GradientProjectionError(
                "projection plan denominators differ from the sealed payload"
            )
        named_parameters = tuple(
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
        if not named_parameters:
            raise GradientProjectionError("projection has no trainable parameters")
        r1_gradients = {
            name: (
                torch.zeros_like(parameter, memory_format=torch.preserve_format)
                if parameter.grad is None
                else parameter.grad.detach().clone()
            )
            for name, parameter in named_parameters
        }
        runtime.zero_gradients(planned_step_id=planned_step_id)

        watch_forward_count = 0
        for micro_step in micro_steps:
            sites = (micro_step.metadata or {}).get("human13_successor_sites", ())
            if not any(site.objective == "watch_ce" for site in sites):
                continue
            result = qwen_forward(model, micro_step)
            context = successor_loss_context_factory(micro_step, result)
            loss = watch_loss_for_context(context, payload.denominators)
            if not bool(torch.isfinite(loss).item()):
                raise GradientProjectionError("G-coordinate watch loss is non-finite")
            runtime.backward(
                loss,
                planned_step_id=planned_step_id,
                sync_gradients=True,
            )
            watch_forward_count += 1
            del loss, context, result
        if watch_forward_count == 0:
            raise GradientProjectionError("projection found no G-coordinate watch pack")

        watch_gradients = {
            name: (
                torch.zeros_like(parameter, memory_format=torch.preserve_format)
                if parameter.grad is None
                else parameter.grad.detach().clone()
            )
            for name, parameter in named_parameters
        }
        receipt = project_and_write_gradients(
            named_parameters,
            r1_gradients=r1_gradients,
            watch_gradients=watch_gradients,
            epsilon=epsilon,
            tolerance=tolerance,
            world_size=int(runtime.world_size),
        )
        return {
            **asdict(receipt),
            "watch_forward_count": watch_forward_count,
            "planned_step_id": int(planned_step_id),
        }

    return handler


def _row_image_id(owner_id: str) -> int:
    parts = str(owner_id).split(":")
    if len(parts) < 3 or parts[0] != "gt":
        raise ValueError(f"owner ID does not encode a Human-13 image: {owner_id}")
    return int(parts[1])


def _rectangle_binding(
    row: SuccessorRow, start: int, *, unit_id: str
) -> SuccessorBinding:
    x1, y1, x2, y2 = row.coordinate_offsets
    return SuccessorBinding(
        "rectangle",
        unit_id,
        row.row_id,
        None,
        row.owner_id,
        (start + x2, start + y2),
        (row.token_ids[x1], row.token_ids[y1]),
    )


def _score_indices(row: SuccessorRow, start: int, branch: str) -> tuple[int, ...]:
    offsets = (
        row.coordinate_offsets
        if branch == "same_description"
        else (*row.description_offsets, *row.coordinate_offsets)
    )
    return tuple(start + index for index in offsets)


def _segment(
    skeleton: Any,
    image_id: int,
    role: str,
    segment_id: str,
    input_ids: tuple[int, ...],
    bindings: tuple[SuccessorBinding, ...],
) -> base.LogicalPanelSegment:
    if is_dataclass(skeleton):
        encoded = replace(skeleton, example_id=segment_id, input_ids=input_ids)
        object.__setattr__(encoded, "human13_successor_bindings", bindings)
    else:
        values = dict(vars(skeleton))
        values.update(
            example_id=segment_id,
            input_ids=input_ids,
            human13_successor_bindings=bindings,
        )
        encoded = SimpleNamespace(**values)
    return base.LogicalPanelSegment(segment_id, image_id, role, encoded)


def _sites_for_pack(item: base.PackedPanelMicroStep) -> tuple[SuccessorLossSite, ...]:
    packed_by_id = {segment.example_id: segment for segment in item.pack.segments}
    sites: list[SuccessorLossSite] = []
    for logical in item.logical_segments:
        packed = packed_by_id[logical.segment_id]
        bindings = getattr(logical.encoded_example, "human13_successor_bindings", ())
        if not bindings:
            raise ValueError("successor segment has no typed bindings")
        input_ids = tuple(logical.encoded_example.input_ids)
        for binding in bindings:
            if any(
                index <= 0 or index >= len(input_ids) for index in binding.token_indices
            ):
                raise ValueError("successor target index escapes its segment")
            sites.append(
                SuccessorLossSite(
                    objective=binding.objective,
                    unit_id=binding.unit_id,
                    row_id=binding.row_id,
                    event_id=binding.event_id,
                    owner_id=binding.owner_id,
                    image_id=logical.image_id,
                    segment_id=logical.segment_id,
                    logits_positions=tuple(
                        packed.start + index - 1 for index in binding.token_indices
                    ),
                    target_token_ids=tuple(
                        input_ids[index] for index in binding.token_indices
                    ),
                    lower_coordinate_token_ids=binding.lower_coordinate_token_ids,
                )
            )
    return tuple(sites)


def _denominators(
    sites_by_pack: Mapping[int, Sequence[SuccessorLossSite]],
) -> SuccessorDenominators:
    sites = tuple(site for values in sites_by_pack.values() for site in values)
    union_images = {
        site.image_id for site in sites if site.objective == "union_candidate"
    }
    replay_owners = {site.owner_id for site in sites if site.objective == "replay_ce"}
    contrast_events = {
        site.event_id: site.image_id
        for site in sites
        if site.objective == "row_duplicate"
    }
    fallback_events = {
        site.event_id: site.image_id
        for site in sites
        if site.objective == "fallback_ul"
    }
    rectangles = {site.unit_id for site in sites if site.objective == "rectangle"}
    watches = {site.owner_id for site in sites if site.objective == "watch_ce"}
    if not union_images or not replay_owners or not rectangles or not watches:
        raise ValueError("successor payload is missing a required objective family")
    return SuccessorDenominators(
        union_images=len(union_images),
        replay_owners=len(replay_owners),
        contrast_images=len(set(contrast_events.values())),
        contrast_events_by_image=_counts_by_image(contrast_events.values()),
        fallback_images=len(set(fallback_events.values())),
        fallback_events_by_image=_counts_by_image(fallback_events.values()),
        rectangle_rows=len(rectangles),
        watch_owners=len(watches),
    )


def _counts_by_image(values: Sequence[int] | Any) -> tuple[tuple[int, int], ...]:
    materialized = tuple(int(value) for value in values)
    return tuple(
        (value, materialized.count(value)) for value in sorted(set(materialized))
    )


def _mean_log_score(
    context: SuccessorPackContext, site: SuccessorLossSite
) -> torch.Tensor:
    selected = _selected_logits(context, site)
    targets = torch.tensor(
        site.target_token_ids, dtype=torch.long, device=selected.device
    )
    return torch.log_softmax(selected, dim=-1).gather(1, targets.unsqueeze(1)).mean()


def _score_for_site(
    context: SuccessorPackContext, site: SuccessorLossSite
) -> torch.Tensor:
    selected = _selected_logits(context, site)
    targets = torch.tensor(
        site.target_token_ids, dtype=torch.long, device=selected.device
    )
    values = torch.log_softmax(selected, dim=-1).gather(1, targets.unsqueeze(1))
    return values.sum() if site.objective == "union_candidate" else values.mean()


def _mean_ce(context: SuccessorPackContext, site: SuccessorLossSite) -> torch.Tensor:
    return -_mean_log_score(context, site)


def _selected_logits(
    context: SuccessorPackContext, site: SuccessorLossSite
) -> torch.Tensor:
    if context.logits_position_ids is None:
        rows = site.logits_positions
    else:
        lookup = {
            position: index
            for index, position in enumerate(context.logits_position_ids)
        }
        missing = [
            position for position in site.logits_positions if position not in lookup
        ]
        if missing:
            raise ValueError(f"compact logits omit successor positions: {missing}")
        rows = tuple(lookup[position] for position in site.logits_positions)
    indices = torch.tensor(rows, dtype=torch.long, device=context.logits.device)
    return context.logits[0].index_select(0, indices).float()


def _rectangle_masks(
    vocab_size: int, lower_tokens: tuple[int, ...], device: torch.device
) -> torch.Tensor:
    if len(lower_tokens) != 2:
        raise ValueError("rectangle site requires x1/y1 token identities")
    token_ids = torch.arange(vocab_size, device=device)
    masks = []
    for lower in lower_tokens:
        if not COORD_TOKEN_START <= lower < COORD_TOKEN_END_EXCLUSIVE:
            raise ValueError("rectangle lower coordinate token is invalid")
        masks.append(
            (token_ids >= max(COORD_TOKEN_START, lower + 1))
            & (token_ids < min(COORD_TOKEN_END_EXCLUSIVE, vocab_size))
        )
    result = torch.stack(masks)
    if not bool(result.any(dim=1).all().item()):
        raise ValueError("rectangle site has no positive-extent coordinate")
    return result


def _parameter_versions(model: Any) -> tuple[tuple[int, int], ...]:
    return tuple(
        (id(parameter), int(parameter._version)) for parameter in model.parameters()
    )


def _short_id(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _normalize(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


__all__ = [
    "MaterializedSuccessorSegments",
    "SuccessorBinding",
    "SuccessorDenominators",
    "SuccessorLossPlan",
    "SuccessorLossRunner",
    "SuccessorLossSite",
    "SuccessorPackContext",
    "SuccessorPayload",
    "build_successor_payload",
    "materialize_successor_segments",
    "successor_loss_context_factory",
    "watch_loss_for_context",
]
