from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Literal, Mapping, Sequence


Stage2TrieCandidateSource = Literal[
    "valid_rollout",
    "fallback_gt_fn_append_only",
]

Stage2TrieSpanRole = Literal[
    "matched_clean",
    "inserted_fn",
    "recovered_fn",
    "neutral_fp",
    "weak_positive_fp",
    "fallback_fn",
]

Stage2TrieTokenSemanticRole = Literal[
    "text",
    "struct",
    "desc",
    "coord",
    "eos",
]

_CANDIDATE_SOURCES = frozenset({"valid_rollout", "fallback_gt_fn_append_only"})
_SPAN_ROLES = frozenset(
    {
        "matched_clean",
        "inserted_fn",
        "recovered_fn",
        "neutral_fp",
        "weak_positive_fp",
        "fallback_fn",
    }
)
_TOKEN_SEMANTIC_ROLES = frozenset({"text", "struct", "desc", "coord", "eos"})
_TOKEN_SEMANTIC_ROLE_PRECEDENCE = {
    "text": 0,
    "desc": 1,
    "struct": 2,
    "coord": 3,
    "eos": 4,
}


@dataclass(frozen=True)
class Stage2TrieObjectSpan:
    """Object-level token span annotation for Stage-2 trie diagnostics."""

    role: Stage2TrieSpanRole
    token_start: int
    token_end: int
    object_iou: float | None
    support_count: int
    loss_weight: float

    def __post_init__(self) -> None:
        """Validate span boundaries and non-negative accounting fields."""

        _validate_membership("role", self.role, _SPAN_ROLES)
        _validate_span_bounds(self.token_start, self.token_end)
        _validate_non_negative_int("support_count", self.support_count)
        _validate_optional_finite_non_negative("object_iou", self.object_iou)
        _validate_finite_non_negative("loss_weight", self.loss_weight)


@dataclass(frozen=True)
class Stage2TrieCandidate:
    """One repaired rollout-correction continuation candidate."""

    sample_id: str
    rollout_index: int
    source: Stage2TrieCandidateSource
    token_ids: Sequence[int]
    loss_weight: float
    object_spans: Sequence[Stage2TrieObjectSpan]

    def __post_init__(self) -> None:
        """Validate source, target tokens, and candidate-level source weight."""

        _validate_string("sample_id", self.sample_id)
        _validate_int("rollout_index", self.rollout_index)
        _validate_membership("source", self.source, _CANDIDATE_SOURCES)
        _validate_finite_non_negative("loss_weight", self.loss_weight)

        token_ids = tuple(self.token_ids)
        if not token_ids:
            raise ValueError("Stage-2 trie candidate token_ids must be non-empty")
        for index, token_id in enumerate(token_ids):
            _validate_token_id(f"token_ids[{index}]", token_id)
        object.__setattr__(self, "token_ids", token_ids)

        object_spans = tuple(self.object_spans)
        for index, span in enumerate(object_spans):
            if not isinstance(span, Stage2TrieObjectSpan):
                raise TypeError(
                    "Stage-2 trie object_spans must contain Stage2TrieObjectSpan "
                    f"instances; object_spans[{index}] has type {type(span).__name__}"
                )
        object.__setattr__(self, "object_spans", object_spans)


@dataclass(frozen=True)
class Stage2TrieTokenTarget:
    """Multiple-positive next-token target for one segment-local label position.

    ``position`` is local to the encoded segment. The predicting logit row
    inside that segment is ``position - 1``; packed-row offsets are applied by
    the teacher-forcing objective consumer.
    """

    position: int
    positive_token_ids: tuple[int, ...]
    source_weights: tuple[float, ...]
    semantic_role: str

    def __post_init__(self) -> None:
        """Validate target position, token alternatives, and source weights."""

        _validate_positive_int("position", self.position)
        _validate_membership("semantic_role", self.semantic_role, _TOKEN_SEMANTIC_ROLES)
        if not self.positive_token_ids:
            raise ValueError(
                "Stage-2 trie token target positive_token_ids must be non-empty"
            )
        if len(self.positive_token_ids) != len(self.source_weights):
            raise ValueError(
                "Stage-2 trie token target positive_token_ids and source_weights "
                "must have equal lengths"
            )
        if len(set(self.positive_token_ids)) != len(self.positive_token_ids):
            raise ValueError(
                "Stage-2 trie token target positive_token_ids must be unique"
            )

        for index, token_id in enumerate(self.positive_token_ids):
            _validate_token_id(f"positive_token_ids[{index}]", token_id)
        for index, weight in enumerate(self.source_weights):
            _validate_finite_non_negative(f"source_weights[{index}]", weight)


@dataclass(frozen=True)
class Stage2TrieSpanScoreRecord:
    """Serializable span-score diagnostic record."""

    sample_id: str
    rollout_index: int
    candidate_source: str
    span_role: str
    object_role: str
    token_start: int
    token_end: int
    mean_token_logprob: float | None
    min_token_logprob: float | None
    object_iou: float | None
    support_count: int
    loss_weight: float

    def __post_init__(self) -> None:
        """Validate strict JSON-safe span-score payload fields."""

        _validate_string("sample_id", self.sample_id)
        _validate_int("rollout_index", self.rollout_index)
        _validate_membership("candidate_source", self.candidate_source, _CANDIDATE_SOURCES)
        _validate_membership("span_role", self.span_role, _SPAN_ROLES)
        _validate_membership("object_role", self.object_role, _SPAN_ROLES)
        _validate_span_bounds(self.token_start, self.token_end)
        _validate_optional_finite("mean_token_logprob", self.mean_token_logprob)
        _validate_optional_finite("min_token_logprob", self.min_token_logprob)
        _validate_optional_finite_non_negative("object_iou", self.object_iou)
        _validate_non_negative_int("support_count", self.support_count)
        _validate_finite_non_negative("loss_weight", self.loss_weight)


@dataclass(frozen=True)
class Stage2TrieSummary:
    """Aggregate target statistics used for metrics."""

    candidate_count: int
    fallback_candidate_count: int
    fallback_loss_weight_sum: float
    weak_positive_fp_count: int
    target_positions: int
    branch_points: int
    max_branching_factor: int


@dataclass(frozen=True)
class Stage2TrieTargets:
    """Compiled Stage-2 trie supervision sidecar."""

    token_targets: tuple[Stage2TrieTokenTarget, ...]
    span_score_records: tuple[Stage2TrieSpanScoreRecord, ...]
    summary: Stage2TrieSummary


def compile_stage2_trie_targets(
    candidates: list[Stage2TrieCandidate],
    *,
    label_position_start: int = 0,
    semantic_role_by_position: Mapping[int, str] | None = None,
    extra_token_targets: Sequence[Stage2TrieTokenTarget] = (),
) -> Stage2TrieTargets:
    """Compile repaired Stage-2 candidates into merged next-token targets.

    ``label_position_start`` is the segment-local encoded-token label position
    for the first candidate token. Emitted ``Stage2TrieTokenTarget.position``
    values stay segment-local, and each target is predicted by the in-segment
    logit row at ``position - 1``.
    """

    _validate_non_negative_int("label_position_start", label_position_start)
    if candidates and label_position_start <= 0:
        raise ValueError(
            "Stage-2 trie label_position_start must be > 0 for non-empty candidates"
        )

    positive_by_position: dict[int, dict[int, float]] = defaultdict(dict)
    role_by_position = _normalize_semantic_role_map(semantic_role_by_position)
    span_score_records: list[Stage2TrieSpanScoreRecord] = []

    # ...collect token positives only while candidates share the teacher prefix.
    teacher_token_ids = tuple(candidates[0].token_ids) if candidates else ()
    for local_index, _teacher_token_id in enumerate(teacher_token_ids):
        teacher_prefix = teacher_token_ids[: int(local_index)]
        position = label_position_start + int(local_index)

        for candidate in candidates:
            if int(local_index) >= len(candidate.token_ids):
                continue
            if tuple(candidate.token_ids[: int(local_index)]) != teacher_prefix:
                continue

            token_id = int(candidate.token_ids[int(local_index)])
            token_loss_weight = _candidate_token_loss_weight(
                candidate=candidate,
                local_index=int(local_index),
            )
            if token_loss_weight <= 0.0:
                continue

            previous_weight = positive_by_position[position].get(token_id, 0.0)
            positive_by_position[position][token_id] = max(
                previous_weight,
                token_loss_weight,
            )

    # ...merge explicit terminal/stop targets into the same deterministic stream.
    for extra_target in extra_token_targets:
        if not isinstance(extra_target, Stage2TrieTokenTarget):
            raise TypeError(
                "Stage-2 trie extra_token_targets must contain "
                "Stage2TrieTokenTarget instances"
            )

        for token_id, source_weight in zip(
            extra_target.positive_token_ids,
            extra_target.source_weights,
        ):
            previous_weight = positive_by_position[extra_target.position].get(
                int(token_id),
                0.0,
            )
            positive_by_position[extra_target.position][int(token_id)] = max(
                float(previous_weight),
                float(source_weight),
            )

        role_by_position[int(extra_target.position)] = _merge_semantic_roles(
            role_by_position.get(int(extra_target.position), "text"),
            str(extra_target.semantic_role),
        )

    # ...preserve span diagnostics for every repaired candidate in the group.
    for candidate in candidates:
        span_score_records.extend(
            _compile_span_score_records(
                candidate,
                label_position_start=label_position_start,
            )
        )

    # ...materialize sorted targets for reproducible tie handling.
    token_targets: list[Stage2TrieTokenTarget] = []
    for position in sorted(positive_by_position):
        token_weights = positive_by_position[position]
        ordered_token_weights = sorted(token_weights.items(), key=lambda item: item[0])
        token_targets.append(
            Stage2TrieTokenTarget(
                position=position,
                positive_token_ids=tuple(
                    token_id for token_id, _ in ordered_token_weights
                ),
                source_weights=tuple(
                    weight for _, weight in ordered_token_weights
                ),
                semantic_role=_semantic_role_for_position(
                    position,
                    semantic_role_by_position=role_by_position,
                ),
            )
        )

    # ...summarize fallback, weak-positive, and branching surfaces.
    branching_factors = [len(target.positive_token_ids) for target in token_targets]
    summary = Stage2TrieSummary(
        candidate_count=len(candidates),
        fallback_candidate_count=sum(
            1
            for candidate in candidates
            if candidate.source == "fallback_gt_fn_append_only"
        ),
        fallback_loss_weight_sum=sum(
            candidate.loss_weight
            for candidate in candidates
            if candidate.source == "fallback_gt_fn_append_only"
        ),
        weak_positive_fp_count=sum(
            1
            for candidate in candidates
            for span in candidate.object_spans
            if span.role == "weak_positive_fp"
        ),
        target_positions=len(token_targets),
        branch_points=sum(1 for factor in branching_factors if factor > 1),
        max_branching_factor=max(branching_factors, default=0),
    )

    return Stage2TrieTargets(
        token_targets=tuple(token_targets),
        span_score_records=tuple(span_score_records),
        summary=summary,
    )


def compile_stage2_trie_targets_for_rollout_group(
    candidates: Sequence[Stage2TrieCandidate],
    *,
    label_position_start: int,
    semantic_role_by_position: Mapping[int, str] | None = None,
    extra_token_targets: Sequence[Stage2TrieTokenTarget] = (),
) -> Stage2TrieTargets:
    """Compile all repaired candidates for one original Stage-2 example.

    ``label_position_start`` is the segment-local encoded-token label position
    for the first candidate token. Emitted target positions remain
    segment-local; packed-row offsets are applied by the objective consumer.
    """

    candidate_list = list(candidates)
    if not candidate_list:
        raise ValueError("Stage-2 trie rollout group candidates must be non-empty")

    sample_ids = {candidate.sample_id for candidate in candidate_list}
    if len(sample_ids) != 1:
        expected = ", ".join(sorted(sample_ids))
        raise ValueError(
            "Stage-2 trie rollout group candidates must share exactly one "
            f"sample_id; got {expected}"
        )

    return compile_stage2_trie_targets(
        candidate_list,
        label_position_start=label_position_start,
        semantic_role_by_position=semantic_role_by_position,
        extra_token_targets=extra_token_targets,
    )


def build_fp_object_span(
    token_start: int,
    token_end: int,
    policy_mode: str,
    support_count: int,
    weak_positive_weight: float,
    *,
    min_support_count: int = 1,
    require_explorer_support: bool = True,
) -> Stage2TrieObjectSpan:
    """Build a false-positive object span for the configured FP policy."""

    _validate_span_bounds(token_start, token_end)
    _validate_non_negative_int("support_count", support_count)
    _validate_finite_non_negative("weak_positive_weight", weak_positive_weight)
    _validate_positive_int("min_support_count", min_support_count)

    if policy_mode not in {"zero_loss_context", "weak_positive_context"}:
        raise ValueError(
            "Unknown Stage-2 false-positive policy mode "
            f"{policy_mode!r}; expected zero_loss_context or weak_positive_context"
        )

    support_satisfied = (not bool(require_explorer_support)) or (
        int(support_count) >= int(min_support_count)
    )
    if policy_mode == "weak_positive_context" and support_satisfied:
        return Stage2TrieObjectSpan(
            role="weak_positive_fp",
            token_start=token_start,
            token_end=token_end,
            object_iou=None,
            support_count=support_count,
            loss_weight=weak_positive_weight,
        )

    return Stage2TrieObjectSpan(
        role="neutral_fp",
        token_start=token_start,
        token_end=token_end,
        object_iou=None,
        support_count=support_count,
        loss_weight=0.0,
    )


def _candidate_token_loss_weight(
    *,
    candidate: Stage2TrieCandidate,
    local_index: int,
) -> float:
    """Resolve candidate or object-span weight for one local token."""

    covering_weights = [
        float(span.loss_weight)
        for span in candidate.object_spans
        if int(span.token_start) <= int(local_index) < int(span.token_end)
    ]
    if covering_weights:
        return float(max(covering_weights))

    return float(candidate.loss_weight)


def stage2_trie_span_score_record_to_json(
    record: Stage2TrieSpanScoreRecord,
) -> dict[str, object]:
    """Convert a span-score record to a JSON-serializable mapping."""

    payload = asdict(record)
    for key in (
        "mean_token_logprob",
        "min_token_logprob",
        "object_iou",
        "loss_weight",
    ):
        value = payload[key]
        if value is not None and not isfinite(value):
            raise ValueError(
                f"Stage-2 trie span-score field {key} must be finite for JSON"
            )

    return payload


def _compile_span_score_records(
    candidate: Stage2TrieCandidate,
    *,
    label_position_start: int,
) -> tuple[Stage2TrieSpanScoreRecord, ...]:
    """Compile candidate object spans into offset diagnostic records."""

    records: list[Stage2TrieSpanScoreRecord] = []
    for span in candidate.object_spans:
        records.append(
            Stage2TrieSpanScoreRecord(
                sample_id=candidate.sample_id,
                rollout_index=candidate.rollout_index,
                candidate_source=candidate.source,
                span_role=span.role,
                object_role=span.role,
                token_start=label_position_start + span.token_start,
                token_end=label_position_start + span.token_end,
                mean_token_logprob=None,
                min_token_logprob=None,
                object_iou=span.object_iou,
                support_count=span.support_count,
                loss_weight=span.loss_weight,
            )
        )

    return tuple(records)


def _semantic_role_for_position(
    position: int,
    *,
    semantic_role_by_position: Mapping[int, str],
) -> str:
    """Resolve the semantic role for one segment-local trie target position."""

    return str(semantic_role_by_position.get(int(position), "text"))


def _normalize_semantic_role_map(
    value: Mapping[int, str] | None,
) -> dict[int, str]:
    """Validate and normalize segment-local semantic role annotations."""

    if value is None:
        return {}

    out: dict[int, str] = {}
    for position, role in value.items():
        _validate_positive_int("semantic_role_by_position.position", position)
        _validate_membership(
            "semantic_role_by_position.role",
            role,
            _TOKEN_SEMANTIC_ROLES,
        )
        out[int(position)] = _merge_semantic_roles(out.get(int(position), "text"), role)

    return out


def _merge_semantic_roles(left: str, right: str) -> str:
    """Merge roles using EOS-first token-type precedence."""

    _validate_membership("semantic_role", left, _TOKEN_SEMANTIC_ROLES)
    _validate_membership("semantic_role", right, _TOKEN_SEMANTIC_ROLES)
    if _TOKEN_SEMANTIC_ROLE_PRECEDENCE[str(right)] > _TOKEN_SEMANTIC_ROLE_PRECEDENCE[
        str(left)
    ]:
        return str(right)

    return str(left)


def _validate_span_bounds(token_start: int, token_end: int) -> None:
    """Validate non-negative half-open span boundaries."""

    _validate_non_negative_int("token_start", token_start)
    _validate_non_negative_int("token_end", token_end)
    if token_start > token_end:
        raise ValueError(
            "Stage-2 trie span must satisfy token_start <= token_end; "
            f"got token_start={token_start}, token_end={token_end}"
        )


def _validate_membership(name: str, value: str, allowed_values: frozenset[str]) -> None:
    """Validate runtime membership for Literal-backed public fields."""

    _validate_string(name, value)
    if value not in allowed_values:
        expected = ", ".join(sorted(allowed_values))
        raise ValueError(
            f"Stage-2 trie {name} must be one of {expected}; got {value!r}"
        )


def _validate_string(name: str, value: object) -> None:
    """Validate a JSON string field."""

    if not isinstance(value, str):
        raise TypeError(f"Stage-2 trie {name} must be a string; got {value!r}")


def _validate_token_id(name: str, value: object) -> None:
    """Validate a token id as a non-bool, non-negative integer."""

    _validate_non_negative_int(name, value)


def _validate_positive_int(name: str, value: object) -> None:
    """Validate an integer field that must be strictly positive."""

    _validate_int(name, value)
    if value <= 0:
        raise ValueError(f"Stage-2 trie {name} must be > 0; got {value!r}")


def _validate_non_negative_int(name: str, value: object) -> None:
    """Validate an integer field that must be non-negative."""

    _validate_int(name, value)
    if value < 0:
        raise ValueError(f"Stage-2 trie {name} must be non-negative; got {value!r}")


def _validate_int(name: str, value: object) -> None:
    """Validate an integer field while rejecting bool."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Stage-2 trie {name} must be an int; got {value!r}")


def _validate_finite_non_negative(name: str, value: float) -> None:
    """Validate a numeric field that must be finite and non-negative."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Stage-2 trie {name} must be numeric; got {value!r}")
    if not isfinite(value):
        raise ValueError(f"Stage-2 trie {name} must be finite; got {value!r}")
    if value < 0:
        raise ValueError(f"Stage-2 trie {name} must be non-negative; got {value!r}")


def _validate_optional_finite(name: str, value: float | None) -> None:
    """Validate an optional numeric field that may be negative but must be finite."""

    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Stage-2 trie {name} must be numeric or None; got {value!r}")
    if not isfinite(value):
        raise ValueError(f"Stage-2 trie {name} must be finite; got {value!r}")


def _validate_optional_finite_non_negative(name: str, value: float | None) -> None:
    """Validate an optional numeric field that must be finite and non-negative."""

    if value is None:
        return
    _validate_finite_non_negative(name, value)


__all__ = [
    "Stage2TrieCandidateSource",
    "Stage2TrieSpanRole",
    "Stage2TrieTokenSemanticRole",
    "Stage2TrieObjectSpan",
    "Stage2TrieCandidate",
    "Stage2TrieTokenTarget",
    "Stage2TrieSpanScoreRecord",
    "Stage2TrieSummary",
    "Stage2TrieTargets",
    "compile_stage2_trie_targets",
    "compile_stage2_trie_targets_for_rollout_group",
    "build_fp_object_span",
    "stage2_trie_span_score_record_to_json",
]
