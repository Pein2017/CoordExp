"""Pure candidate scoring and continuation selection for Human-13."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from scripts.research.compare_clean_rollout_owner_coverage import (
    _global_matches,
    iou_xyxy,
)
from src.eval.detection_categories import normalize_coco_category_name


@dataclass(frozen=True)
class CandidatePath:
    image_id: int
    owner_id: str
    alias_id: str
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class SurfaceEvidence:
    surface: str
    logits: tuple[tuple[float, ...], ...]
    target_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class TokenDecision:
    position: int
    target_token_id: int
    actual_argmax_token_id: int
    competitor_token_id: int
    target_logit: float
    competitor_logit: float
    strict_margin: float
    tie_count: int


@dataclass(frozen=True)
class PackedCandidateScore:
    path: CandidatePath
    sites: tuple[TokenDecision, ...]
    barrier: float
    first_bottleneck_index: int | None


@dataclass(frozen=True)
class AlignedSurfaceSite:
    position: int
    packed: TokenDecision
    hf: TokenDecision
    strict_margin_drift: float
    argmax_disagreement: bool
    bottleneck_disagreement: bool


@dataclass(frozen=True)
class CandidateScore:
    path: CandidatePath
    hf_sites: tuple[TokenDecision, ...]
    packed_sites: tuple[TokenDecision, ...]
    aligned_sites: tuple[AlignedSurfaceSite, ...]
    hf_barrier: float
    first_bottleneck_index: int | None
    packed_first_bottleneck_index: int | None
    first_bottleneck_disagreement: bool | None
    packed_barrier: float | None
    max_surface_margin_drift: float | None
    surface_rank_disagreement: bool | None


@dataclass(frozen=True)
class ContinuationOutcome:
    owner_id: str
    hf_barrier: float
    protected_coverable: bool
    unique_owner_delta: int
    termination_status: str
    cap_hit: bool
    duplicate_increase: int
    malformed_increase: int
    row_count: int
    generated_tokens: int


@dataclass
class _FlowEdge:
    to: int
    reverse: int
    capacity: int
    cost: int
    owner_index: int = -1
    prediction_index: int = -1


def score_candidate(
    path: CandidatePath,
    *,
    packed: SurfaceEvidence | None,
    hf: SurfaceEvidence | None,
) -> CandidateScore:
    """Score a complete row while reserving all decisions for the HF surface."""

    if hf is None or hf.surface != "hf_fp32_sdpa":
        raise ValueError("decision-owning HF fp32/SDPA evidence is required")
    hf_sites = _decisions(path, hf)
    packed_sites: tuple[TokenDecision, ...] | None = None
    if packed is not None:
        if packed.surface != "packed_bf16_fa2":
            raise ValueError("packed evidence surface identity differs")
        packed_sites = _decisions(path, packed)
        if len(packed.logits[0]) != len(hf.logits[0]):
            raise ValueError("packed/HF vocabulary sizes differ")
    barrier = sum(max(0.0, -site.strict_margin) for site in hf_sites)
    first = next(
        (site.position for site in hf_sites if site.strict_margin <= 0.0), None
    )
    packed_barrier = (
        None
        if packed_sites is None
        else sum(max(0.0, -site.strict_margin) for site in packed_sites)
    )
    packed_first = (
        None
        if packed_sites is None
        else next(
            (site.position for site in packed_sites if site.strict_margin <= 0.0),
            None,
        )
    )
    aligned = (
        ()
        if packed_sites is None
        else tuple(
            AlignedSurfaceSite(
                position=hf_site.position,
                packed=packed_site,
                hf=hf_site,
                strict_margin_drift=float(
                    abs(hf_site.strict_margin - packed_site.strict_margin)
                ),
                argmax_disagreement=(
                    hf_site.actual_argmax_token_id != packed_site.actual_argmax_token_id
                ),
                bottleneck_disagreement=(
                    (hf_site.strict_margin <= 0.0) != (packed_site.strict_margin <= 0.0)
                ),
            )
            for hf_site, packed_site in zip(hf_sites, packed_sites, strict=True)
        )
    )
    drift = (
        None
        if packed_sites is None
        else max(site.strict_margin_drift for site in aligned)
    )
    disagreement = (
        None
        if packed_sites is None
        else any(
            site.argmax_disagreement or site.bottleneck_disagreement for site in aligned
        )
    )
    return CandidateScore(
        path=path,
        hf_sites=hf_sites,
        packed_sites=() if packed_sites is None else packed_sites,
        aligned_sites=aligned,
        hf_barrier=float(barrier),
        first_bottleneck_index=first,
        packed_first_bottleneck_index=packed_first,
        first_bottleneck_disagreement=(
            None if packed_sites is None else first != packed_first
        ),
        packed_barrier=None if packed_barrier is None else float(packed_barrier),
        max_surface_margin_drift=None if drift is None else float(drift),
        surface_rank_disagreement=disagreement,
    )


def score_packed_candidate(
    path: CandidatePath, packed: SurfaceEvidence
) -> PackedCandidateScore:
    """Create packed-only prefilter evidence without requiring HF execution."""

    if packed.surface != "packed_bf16_fa2":
        raise ValueError("packed evidence surface identity differs")
    sites = _decisions(path, packed)
    return PackedCandidateScore(
        path=path,
        sites=sites,
        barrier=float(sum(max(0.0, -site.strict_margin) for site in sites)),
        first_bottleneck_index=next(
            (site.position for site in sites if site.strict_margin <= 0.0), None
        ),
    )


def shortlist_candidates(
    scores: Sequence[CandidateScore], *, limit: int = 4
) -> tuple[CandidateScore, ...]:
    """Keep the minimum-HF-barrier alias per owner, then two to four owners."""

    if limit < 2 or limit > 4:
        raise ValueError("shortlist limit must be between two and four")
    by_owner: dict[str, CandidateScore] = {}
    for score in scores:
        prior = by_owner.get(score.path.owner_id)
        key = (score.hf_barrier, score.path.alias_id)
        if prior is None or key < (prior.hf_barrier, prior.path.alias_id):
            by_owner[score.path.owner_id] = score
    return tuple(
        sorted(
            by_owner.values(),
            key=lambda item: (
                item.hf_barrier,
                item.path.owner_id,
                item.path.alias_id,
            ),
        )[:limit]
    )


def packed_prefilter(
    scores: Sequence[PackedCandidateScore], *, aliases_per_owner: int = 2
) -> tuple[PackedCandidateScore, ...]:
    """Bound HF work without promoting packed scores to a decision surface."""

    if aliases_per_owner < 1 or aliases_per_owner > 2:
        raise ValueError("aliases_per_owner must be one or two, at most two")
    grouped: dict[str, list[PackedCandidateScore]] = {}
    for score in scores:
        grouped.setdefault(score.path.owner_id, []).append(score)
    retained: list[PackedCandidateScore] = []
    for owner_id in sorted(grouped):
        retained.extend(
            sorted(
                grouped[owner_id],
                key=lambda item: (item.barrier, item.path.alias_id),
            )[:aliases_per_owner]
        )
    return tuple(retained)


def select_continuation(
    outcomes: Sequence[ContinuationOutcome],
) -> ContinuationOutcome:
    """Choose only a protected, positive-gain, naturally terminated continuation."""

    eligible = tuple(
        outcome
        for outcome in outcomes
        if outcome.protected_coverable
        and outcome.unique_owner_delta > 0
        and outcome.termination_status == "natural_im_end"
        and not outcome.cap_hit
    )
    if not eligible:
        raise ValueError("no eligible forced continuation")
    return min(
        eligible,
        key=lambda item: (
            -item.unique_owner_delta,
            item.duplicate_increase,
            item.malformed_increase,
            item.row_count,
            item.generated_tokens,
            item.hf_barrier,
            item.owner_id,
        ),
    )


def protected_owner_coverable(
    owners: Sequence[tuple[str, str, tuple[float, float, float, float]]],
    predictions: Sequence[tuple[str, tuple[float, float, float, float]]],
    *,
    protected_owner_ids: Sequence[str],
    threshold: float = 0.5,
) -> bool:
    """Return whether a maximum-cardinality matching can cover every protected owner."""

    owner_ids = tuple(owner_id for owner_id, _, _ in owners)
    protected = set(protected_owner_ids)
    if protected - set(owner_ids):
        raise ValueError("protected owner is absent from owner table")
    gt = [
        (normalize_coco_category_name(category), bbox) for _, category, bbox in owners
    ]
    pred = [
        (normalize_coco_category_name(category), bbox) for category, bbox in predictions
    ]
    maximum_cardinality = len(_global_matches(gt, pred, threshold))
    matches = _priority_matching(
        owners, predictions, protected=protected, threshold=threshold
    )
    matched_protected = {
        owner_ids[owner_index]
        for owner_index, _prediction_index in matches
        if owner_ids[owner_index] in protected
    }
    return len(matches) == maximum_cardinality and matched_protected == protected


def _priority_matching(
    owners: Sequence[tuple[str, str, tuple[float, float, float, float]]],
    predictions: Sequence[tuple[str, tuple[float, float, float, float]]],
    *,
    protected: set[str],
    threshold: float,
) -> tuple[tuple[int, int], ...]:
    source = 0
    owner_start = 1
    prediction_start = owner_start + len(owners)
    sink = prediction_start + len(predictions)
    graph: list[list[_FlowEdge]] = [[] for _ in range(sink + 1)]

    def add_edge(
        left: int,
        right: int,
        cost: int,
        *,
        owner_index: int = -1,
        prediction_index: int = -1,
    ) -> None:
        forward = _FlowEdge(
            right,
            len(graph[right]),
            1,
            cost,
            owner_index,
            prediction_index,
        )
        reverse = _FlowEdge(left, len(graph[left]), 0, -cost)
        graph[left].append(forward)
        graph[right].append(reverse)

    for owner_index in range(len(owners)):
        add_edge(source, owner_start + owner_index, 0)
    for prediction_index in range(len(predictions)):
        add_edge(prediction_start + prediction_index, sink, 0)
    protected_bonus = (max(len(owners), len(predictions)) + 1) * 1_000_000_001
    for owner_index, (owner_id, category, bbox) in enumerate(owners):
        normalized = normalize_coco_category_name(category)
        for prediction_index, (pred_category, pred_bbox) in enumerate(predictions):
            if normalized != normalize_coco_category_name(pred_category):
                continue
            overlap = iou_xyxy(bbox, pred_bbox)
            if overlap < threshold:
                continue
            reward = round(overlap * 1_000_000_000)
            if owner_id in protected:
                reward += protected_bonus
            add_edge(
                owner_start + owner_index,
                prediction_start + prediction_index,
                -reward,
                owner_index=owner_index,
                prediction_index=prediction_index,
            )

    while True:
        distances: list[int | None] = [None] * len(graph)
        previous: list[tuple[int, int] | None] = [None] * len(graph)
        distances[source] = 0
        for _ in range(len(graph) - 1):
            changed = False
            for left, edges in enumerate(graph):
                if distances[left] is None:
                    continue
                for edge_index, edge in enumerate(edges):
                    if edge.capacity <= 0:
                        continue
                    candidate = distances[left] + edge.cost
                    if distances[edge.to] is None or candidate < distances[edge.to]:
                        distances[edge.to] = candidate
                        previous[edge.to] = (left, edge_index)
                        changed = True
            if not changed:
                break
        if distances[sink] is None:
            break
        node = sink
        while node != source:
            link = previous[node]
            if link is None:
                raise RuntimeError("matching predecessor is missing")
            left, edge_index = link
            edge = graph[left][edge_index]
            edge.capacity = 0
            graph[node][edge.reverse].capacity = 1
            node = left

    return tuple(
        sorted(
            (edge.owner_index, edge.prediction_index)
            for owner_index in range(len(owners))
            for edge in graph[owner_start + owner_index]
            if edge.owner_index >= 0 and edge.capacity == 0
        )
    )


def _decisions(
    path: CandidatePath, evidence: SurfaceEvidence
) -> tuple[TokenDecision, ...]:
    if not path.token_ids or len(evidence.logits) != len(path.token_ids):
        raise ValueError("candidate/logit length or shape differs")
    if evidence.target_token_ids != path.token_ids:
        raise ValueError("surface target token IDs differ from candidate")
    decisions: list[TokenDecision] = []
    vocab_size: int | None = None
    for position, (row, target) in enumerate(
        zip(evidence.logits, path.token_ids, strict=True)
    ):
        if vocab_size is None:
            vocab_size = len(row)
        if len(row) != vocab_size or vocab_size < 2:
            raise ValueError("surface logits have inconsistent vocabulary shape")
        if target < 0 or target >= vocab_size:
            raise ValueError("target token ID is outside vocabulary")
        if any(not math.isfinite(float(value)) for value in row):
            raise ValueError("surface logits must be finite")
        actual = max(range(vocab_size), key=lambda token_id: (row[token_id], -token_id))
        competitors = [token_id for token_id in range(vocab_size) if token_id != target]
        competitor = max(competitors, key=lambda token_id: (row[token_id], -token_id))
        maximum = max(row)
        tie_count = sum(float(value) == float(maximum) for value in row)
        decisions.append(
            TokenDecision(
                position=position,
                target_token_id=target,
                actual_argmax_token_id=actual,
                competitor_token_id=competitor,
                target_logit=float(row[target]),
                competitor_logit=float(row[competitor]),
                strict_margin=float(row[target] - row[competitor]),
                tie_count=tie_count,
            )
        )
    return tuple(decisions)


__all__ = [
    "CandidatePath",
    "CandidateScore",
    "ContinuationOutcome",
    "AlignedSurfaceSite",
    "PackedCandidateScore",
    "SurfaceEvidence",
    "TokenDecision",
    "packed_prefilter",
    "protected_owner_coverable",
    "score_candidate",
    "score_packed_candidate",
    "select_continuation",
    "shortlist_candidates",
]
