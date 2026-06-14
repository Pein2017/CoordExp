"""Template-aware rollout codecs for the Stage-2 policy seam.

This module intentionally stops at the pure policy and text-codec boundary. It
does not integrate with the trainer hot path, token-id span ownership, or
assignment logic. Legacy token-prefix compatibility remains owned by
``src.trainers.rollout_matching.parsing`` until the trainer is migrated.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from operator import index as integer_index
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Mapping, Protocol, Sequence

if TYPE_CHECKING:
    from src.trainers.rollout_matching.contracts import GTObject as _TrainerGTObject

Stage2RolloutTemplateFamily = Literal["compact_full", "coordjson"]

FALLBACK_GT_FN_APPEND_ONLY = "fallback_gt_fn_append_only"

BOX_START_TOKEN = "<|box_start|>"
COMPACT_FULL_FORMAT = "compact_full"
COORDJSON_FORMAT = "coordjson"
OBJECT_REF_START_TOKEN = "<|object_ref_start|>"

_COORD_TOKEN_RE = re.compile(r"^<\|coord_(\d{1,4})\|>$")


class Stage2RolloutTemplateMismatchError(ValueError):
    """Raised when rollout text belongs to the wrong template family."""


@dataclass(frozen=True)
class Stage2RolloutTemplatePolicy:
    """Resolved Stage-2 rollout template and parser/appender policy.

    :param template_family: Template family selected for rollout I/O.
    :param parser_id: Parser implementation identifier.
    :param append_policy_id: False-negative append policy identifier.
    :param invalid_rollout_policy: Policy used for invalid rollout outputs.
    :param fallback_loss_weight: Loss weight applied to fallback targets.
    :param strict_rollout_preflight: When true, reject compact-full rollout
        salvage/fallback paths instead of silently converting them into
        training targets.
    """

    template_family: Stage2RolloutTemplateFamily
    parser_id: str
    append_policy_id: str
    invalid_rollout_policy: str
    fallback_loss_weight: float = 1.0
    strict_rollout_preflight: bool = False

    @property
    def diagnostics_metadata(self) -> dict[str, object]:
        """Return stable diagnostics fields for artifacts or logs."""

        return {
            "resolved_rollout_template": self.template_family,
            "rollout_parser_id": self.parser_id,
            "rollout_append_policy_id": self.append_policy_id,
            "invalid_rollout_policy": self.invalid_rollout_policy,
            "fallback_loss_weight": float(self.fallback_loss_weight),
            "strict_rollout_preflight": bool(self.strict_rollout_preflight),
        }


@dataclass(frozen=True)
class Stage2RolloutObject:
    """Parsed or renderable rollout object at the Stage-2 codec seam.

    :param object_id: Stable object identifier within the rollout surface.
    :param index: Object index in the parsed/rendered sequence.
    :param desc: Object description.
    :param bbox_tokens: Optional compact coordinate tokens.
    :param bbox_norm1000: Optional numeric norm1000 bbox coordinates.
    :param geom_type: Geometry type. Compact-full rendering supports
        ``bbox_2d`` only in this slice.
    :param provenance: Source provenance label.
    :param metadata: Additional non-authoritative diagnostic metadata.
    """

    object_id: str
    index: int
    desc: str
    bbox_tokens: tuple[str, ...] | None = None
    bbox_norm1000: tuple[int, ...] | None = None
    geom_type: str = "bbox_2d"
    provenance: str = "rollout_accepted"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze caller-provided metadata snapshots."""

        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class Stage2RolloutParseResult:
    """Template-aware rollout parse result.

    :param template_family: Template family used for parsing.
    :param parser_id: Parser implementation identifier.
    :param response_text: Raw response text supplied to the parser.
    :param valid_objects: Parsed valid objects. Empty on invalid rollouts.
    :param invalid_rollout: Whether the rollout was malformed for this parser.
    :param empty_valid_object_set: Whether parsing succeeded but produced no
        valid objects.
    :param truncated: Whether the parser observed true output truncation.  For
        compact-full rollouts, terminal or padding suffixes such as chat stop
        markers and EOS padding must not be treated as truncation.
    :param fallback_reason: Optional fallback reason for compact-full policy.
    :param append_prefix_text: Typed legacy CoordJSON prefix for append-ready
        non-invalid rollouts. Compact-full parse results leave it unset.
    :param metadata: Additional diagnostics for artifacts or logs.
    :param response_token_ids: Compatibility response token ids used by Stage-2
        trainer logging and meta construction.
    :param prefix_token_ids: Compatibility prefix token ids used by legacy
        debug surfaces. Compact-full results keep this empty.
    :param prefix_text: Compatibility prefix text used by legacy debug surfaces.
        Compact-full results keep this empty.
    :param dropped_invalid: Count of parser-dropped invalid records.
    :param dropped_ambiguous: Count of parser-dropped ambiguous records.
    :param dropped_invalid_by_reason: Parser drop counters by reason.
    """

    template_family: Stage2RolloutTemplateFamily
    parser_id: str
    response_text: str
    valid_objects: tuple[Stage2RolloutObject, ...]
    invalid_rollout: bool
    empty_valid_object_set: bool
    truncated: bool
    fallback_reason: str | None = None
    append_prefix_text: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    response_token_ids: tuple[int, ...] = ()
    prefix_token_ids: tuple[int, ...] = ()
    prefix_text: str = ""
    dropped_invalid: int = 0
    dropped_ambiguous: int = 0
    dropped_invalid_by_reason: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze caller-provided metadata snapshots."""

        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(
            self,
            "response_token_ids",
            tuple(int(t) for t in self.response_token_ids),
        )
        object.__setattr__(
            self,
            "prefix_token_ids",
            tuple(int(t) for t in self.prefix_token_ids),
        )
        object.__setattr__(
            self,
            "dropped_invalid",
            int(self.dropped_invalid),
        )
        object.__setattr__(
            self,
            "dropped_ambiguous",
            int(self.dropped_ambiguous),
        )
        object.__setattr__(
            self,
            "dropped_invalid_by_reason",
            MappingProxyType(
                {
                    str(key): int(value)
                    for key, value in dict(self.dropped_invalid_by_reason).items()
                }
            ),
        )


@dataclass(frozen=True)
class Stage2RolloutTargetMetadata:
    """Metadata for a rendered Stage-2 rollout-correction target."""

    template_family: Stage2RolloutTemplateFamily
    parser_id: str
    append_policy_id: str
    rollout_context: str
    fallback_reason: str | None
    fallback_loss_weight: float
    counts_as_valid_rollout: bool


@dataclass(frozen=True)
class Stage2RolloutTarget:
    """Rendered target text plus policy metadata."""

    text: str
    metadata: Stage2RolloutTargetMetadata


@dataclass(frozen=True)
class Stage2RolloutDiagnosticSummary:
    """Pure diagnostic summary for Stage-2 rollout fallback behavior."""

    invalid_fallback_gt_fn_count: int
    invalid_fallback_gt_fn_rate: float
    empty_valid_object_rate: float
    fallback_loss_share: float
    fallback_dominance_warning: bool
    parse_truncated_rate: float
    parser_template_mismatch_rate: float


class _AppendObjectProtocol(Protocol):
    """Structural FN/GT object surface needed by rollout appenders."""

    geom_type: str
    points_norm1000: Sequence[object]
    desc: str


if TYPE_CHECKING:
    _AppendObjectLike = _TrainerGTObject | _AppendObjectProtocol
else:
    _AppendObjectLike = _AppendObjectProtocol


@dataclass(frozen=True)
class _LegacyAppendObject:
    """Import-light object compatible with legacy append serialization."""

    index: int
    geom_type: str
    points_norm1000: list[int]
    desc: str


def resolve_stage2_rollout_template_policy(
    rollout_template_family: str | None = None,
    *,
    custom_json_format: str | None = None,
    compact_decode_policy: str | None = None,
    rollout_decode_policy: str | None = None,
    invalid_rollout_policy: str | None = None,
    fallback_loss_weight: float = 1.0,
    strict_rollout_preflight: bool | str | int = False,
) -> Stage2RolloutTemplatePolicy:
    """Resolve the explicit Stage-2 rollout template policy.

    ``custom.json_format`` is accepted only as context for callers migrating
    legacy config surfaces. It never selects the rollout parser/appender.
    """

    # require an explicit rollout surface
    if rollout_template_family is None:
        raise ValueError(
            "rollout_template_family must be explicit; custom.json_format does not "
            "select the Stage-2 rollout parser"
        )

    template_family = str(rollout_template_family).strip().lower().replace("-", "_")
    _ = custom_json_format

    # validate shared numeric provenance
    try:
        resolved_fallback_loss_weight = float(fallback_loss_weight)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "stage2_rollout_correction.correction.fallback_loss_weight must be a float/int"
        ) from exc
    if not math.isfinite(resolved_fallback_loss_weight):
        raise ValueError(
            "stage2_rollout_correction.correction.fallback_loss_weight must be finite"
        )
    if resolved_fallback_loss_weight < 0.0:
        raise ValueError(
            "stage2_rollout_correction.correction.fallback_loss_weight must be >= 0"
        )
    if isinstance(strict_rollout_preflight, bool):
        resolved_strict_rollout_preflight = bool(strict_rollout_preflight)
    elif isinstance(strict_rollout_preflight, int) and strict_rollout_preflight in {
        0,
        1,
    }:
        resolved_strict_rollout_preflight = bool(strict_rollout_preflight)
    elif isinstance(strict_rollout_preflight, str):
        strict_raw = strict_rollout_preflight.strip().lower()
        if strict_raw in {"true", "1", "yes", "on"}:
            resolved_strict_rollout_preflight = True
        elif strict_raw in {"false", "0", "no", "off"}:
            resolved_strict_rollout_preflight = False
        else:
            raise ValueError(
                "stage2_rollout_correction.correction.strict_rollout_preflight must be boolean"
            )
    else:
        raise TypeError(
            "stage2_rollout_correction.correction.strict_rollout_preflight must be boolean"
        )

    if rollout_decode_policy is not None:
        raise ValueError(
            "stage2_rollout_correction.correction.rollout_decode_policy has been "
            "removed; Stage-2 rollouts no longer expose decode-policy selectors."
        )
    if compact_decode_policy is not None:
        raise ValueError(
            "stage2_rollout_correction.correction.compact_decode_policy has been "
            "removed; Stage-2 rollouts no longer expose decode-policy selectors."
        )
    invalid_policy = (
        None
        if invalid_rollout_policy is None
        else str(invalid_rollout_policy).strip().lower().replace("-", "_")
    )
    if invalid_policy is not None and invalid_policy not in {
        "abort",
        "dump_and_continue",
        FALLBACK_GT_FN_APPEND_ONLY,
    }:
        raise ValueError(
            "stage2_rollout_correction.correction.invalid_rollout_policy must be one of "
            "{'abort', 'dump_and_continue', 'fallback_gt_fn_append_only'}"
        )

    # resolve the canonical compact-full surface
    if template_family == COMPACT_FULL_FORMAT:
        if invalid_policy is not None and invalid_policy != FALLBACK_GT_FN_APPEND_ONLY:
            raise ValueError(
                "stage2_rollout_correction.correction.invalid_rollout_policy for compact_full must be "
                "'fallback_gt_fn_append_only' until alternate invalid-rollout "
                "behavior is implemented"
            )
        return Stage2RolloutTemplatePolicy(
            template_family="compact_full",
            parser_id="compact_full",
            append_policy_id="compact_full_fn_append",
            invalid_rollout_policy=invalid_policy or FALLBACK_GT_FN_APPEND_ONLY,
            fallback_loss_weight=resolved_fallback_loss_weight,
            strict_rollout_preflight=resolved_strict_rollout_preflight,
        )

    # resolve the explicit legacy CoordJSON surface
    if template_family == COORDJSON_FORMAT:
        if invalid_policy == FALLBACK_GT_FN_APPEND_ONLY:
            raise ValueError(
                "fallback_gt_fn_append_only is only valid for compact_full; "
                "coordjson must use 'abort' or 'dump_and_continue'"
            )
        return Stage2RolloutTemplatePolicy(
            template_family="coordjson",
            parser_id="coordjson_legacy",
            append_policy_id="coordjson_legacy_fn_append",
            invalid_rollout_policy=invalid_policy or "abort",
            fallback_loss_weight=resolved_fallback_loss_weight,
            strict_rollout_preflight=resolved_strict_rollout_preflight,
        )

    raise ValueError(
        "rollout_template_family must be one of {'compact_full', 'coordjson'}; "
        f"got {rollout_template_family!r}"
    )


def calculate_stage2_rollout_diagnostics(
    parse_results: Sequence[Stage2RolloutParseResult],
    *,
    parser_template_mismatch_count: int = 0,
    fallback_dominance_threshold: float = 0.4,
) -> Stage2RolloutDiagnosticSummary:
    """Calculate pure fallback/mismatch diagnostics from parse results.

    ``fallback_loss_share`` is the trainable fallback share over parse results.
    Parser/template mismatches are hard failures and are excluded from that
    trainable loss-share denominator.
    """

    # count observed rollout outcomes
    mismatch_count = max(0, int(parser_template_mismatch_count))
    total_count = int(len(parse_results)) + mismatch_count
    if total_count <= 0:
        return Stage2RolloutDiagnosticSummary(
            invalid_fallback_gt_fn_count=0,
            invalid_fallback_gt_fn_rate=0.0,
            empty_valid_object_rate=0.0,
            fallback_loss_share=0.0,
            fallback_dominance_warning=False,
            parse_truncated_rate=0.0,
            parser_template_mismatch_rate=0.0,
        )

    invalid_fallback_count = sum(
        1
        for result in parse_results
        if result.invalid_rollout and result.fallback_reason is not None
    )
    empty_valid_count = sum(1 for result in parse_results if result.empty_valid_object_set)
    truncated_count = sum(1 for result in parse_results if result.truncated)
    fallback_count = int(invalid_fallback_count) + int(empty_valid_count)

    # keep hard parser mismatches out of trainable loss-share accounting
    trainable_count = max(1, int(len(parse_results)))
    fallback_loss_share = float(fallback_count) / float(trainable_count)
    return Stage2RolloutDiagnosticSummary(
        invalid_fallback_gt_fn_count=int(invalid_fallback_count),
        invalid_fallback_gt_fn_rate=float(invalid_fallback_count) / float(total_count),
        empty_valid_object_rate=float(empty_valid_count) / float(total_count),
        fallback_loss_share=fallback_loss_share,
        fallback_dominance_warning=bool(
            fallback_loss_share > float(fallback_dominance_threshold)
        ),
        parse_truncated_rate=float(truncated_count) / float(total_count),
        parser_template_mismatch_rate=float(mismatch_count) / float(total_count),
    )


class CompactFullRolloutCodec:
    """Compact-full rollout parser and target renderer for Stage-2."""

    def __init__(self, policy: Stage2RolloutTemplatePolicy | None = None) -> None:
        self.policy = policy or resolve_stage2_rollout_template_policy("compact_full")
        if self.policy.template_family != "compact_full":
            raise ValueError("CompactFullRolloutCodec requires compact_full policy")

    def parse(self, raw_text: str) -> Stage2RolloutParseResult:
        """Parse compact-full rollout text without CoordJSON fallback."""

        from src.common.detection_sequence import parse_compact_detection_sequence

        # fail hard on cross-template text
        response_text = str(raw_text)
        if _looks_like_coordjson(response_text):
            raise Stage2RolloutTemplateMismatchError(
                "compact_full parser received CoordJSON/JSON-like rollout text"
            )

        parsed = parse_compact_detection_sequence(
            response_text,
            detection_sequence_format=COMPACT_FULL_FORMAT,
            salvage_malformed_rows=False,
        )
        fallback_reason = None
        if parsed is None:
            parsed = parse_compact_detection_sequence(
                response_text,
                detection_sequence_format=COMPACT_FULL_FORMAT,
                salvage_malformed_rows=True,
            )
            if parsed is not None:
                fallback_reason = "compact_full_salvage"
        if parsed is None:
            return self._parse_result(
                response_text=response_text,
                valid_objects=(),
                invalid_rollout=True,
                empty_valid_object_set=False,
                fallback_reason="malformed_compact_full",
            )

        objects = tuple(
            _stage2_object_from_compact_entry(index=index, entry=entry)
            for index, entry in enumerate(parsed.get("objects", []))
        )
        if not objects:
            return self._parse_result(
                response_text=response_text,
                valid_objects=(),
                invalid_rollout=False,
                empty_valid_object_set=True,
                fallback_reason="empty_valid_object_set",
            )

        return self._parse_result(
            response_text=response_text,
            valid_objects=objects,
            invalid_rollout=False,
            empty_valid_object_set=False,
            fallback_reason=fallback_reason,
        )

    def render_target(
        self,
        accepted_objects: Sequence[Stage2RolloutObject],
        fn_objects: Sequence[_AppendObjectLike],
    ) -> str:
        """Render accepted rollout objects plus FN objects as compact-full rows."""

        from src.common.detection_sequence import render_compact_detection_sequence

        payload = {
            "objects": [
                _compact_payload_from_stage2_object(obj) for obj in accepted_objects
            ]
        }
        payload["objects"].extend(_compact_payload_from_gt_object(obj) for obj in fn_objects)
        return render_compact_detection_sequence(
            payload,
            detection_sequence_format=COMPACT_FULL_FORMAT,
        )

    def build_rollout_correction_target(
        self,
        parse_result: Stage2RolloutParseResult,
        *,
        fn_objects: Sequence[_AppendObjectLike],
    ) -> Stage2RolloutTarget:
        """Build a rollout-correction target from a parse result and FN objects."""

        # validate parser/template alignment
        if parse_result.template_family != "compact_full":
            raise Stage2RolloutTemplateMismatchError(
                "compact_full appender received non-compact_full parse result"
            )

        # apply fallback policy for invalid or empty compact rollouts
        fallback_applies = bool(
            parse_result.invalid_rollout or parse_result.empty_valid_object_set
        )
        accepted_objects = () if fallback_applies else parse_result.valid_objects
        text = self.render_target(accepted_objects, fn_objects)
        metadata = Stage2RolloutTargetMetadata(
            template_family="compact_full",
            parser_id=self.policy.parser_id,
            append_policy_id=self.policy.append_policy_id,
            rollout_context=(
                FALLBACK_GT_FN_APPEND_ONLY
                if fallback_applies
                else "rollout_valid_with_fn_append"
            ),
            fallback_reason=parse_result.fallback_reason if fallback_applies else None,
            fallback_loss_weight=(
                float(self.policy.fallback_loss_weight) if fallback_applies else 0.0
            ),
            counts_as_valid_rollout=not fallback_applies,
        )
        return Stage2RolloutTarget(text=text, metadata=metadata)

    def _parse_result(
        self,
        *,
        response_text: str,
        valid_objects: tuple[Stage2RolloutObject, ...],
        invalid_rollout: bool,
        empty_valid_object_set: bool,
        fallback_reason: str | None,
    ) -> Stage2RolloutParseResult:
        """Return a compact-full parse result with stable metadata."""

        return Stage2RolloutParseResult(
            template_family="compact_full",
            parser_id=self.policy.parser_id,
            response_text=str(response_text),
            valid_objects=valid_objects,
            invalid_rollout=bool(invalid_rollout),
            empty_valid_object_set=bool(empty_valid_object_set),
            # Compact-full parsing strips chat/EOS suffixes before row parsing.
            # Those suffixes are normal with batched HF decoding, especially
            # when EOS is used as padding, so they are not evidence that the
            # output hit max_new_tokens or was cut mid-row.
            truncated=False,
            fallback_reason=fallback_reason,
            metadata=self.policy.diagnostics_metadata,
        )


class LegacyCoordJsonRolloutCodec:
    """Explicit legacy CoordJSON rollout parser and appender adapter.

    Token-id prefix compatibility remains owned by
    ``src.trainers.rollout_matching.parsing``. This adapter is intentionally
    text-level until the trainer hot path is migrated.
    """

    def __init__(self, policy: Stage2RolloutTemplatePolicy | None = None) -> None:
        self.policy = policy or resolve_stage2_rollout_template_policy("coordjson")
        if self.policy.template_family != "coordjson":
            raise ValueError("LegacyCoordJsonRolloutCodec requires coordjson policy")

    def parse(self, raw_text: str) -> Stage2RolloutParseResult:
        """Parse legacy CoordJSON rollout text without compact-full fallback."""

        from src.utils.coordjson_transpiler import parse_coordjson

        # fail hard on cross-template text
        response_text = str(raw_text)
        if _looks_like_compact_full(response_text):
            raise Stage2RolloutTemplateMismatchError(
                "coordjson parser received compact_full marker text"
            )

        parsed = parse_coordjson(
            response_text,
            mode="salvage",
            object_field_order="desc_first",
        )
        if bool(parsed.parse_failed) or parsed.objects_array_open_cut is None:
            return self._parse_result(
                response_text=response_text,
                valid_objects=(),
                prefix_text='{"objects": [',
                invalid_rollout=True,
                empty_valid_object_set=False,
                truncated=bool(parsed.truncated),
            )

        valid_objects = tuple(
            _stage2_object_from_coordjson_record(index=index, record=record)
            for index, record in enumerate(parsed.records)
        )
        prefix_text = _coordjson_append_ready_prefix(response_text, parsed)
        return self._parse_result(
            response_text=response_text,
            valid_objects=valid_objects,
            prefix_text=prefix_text,
            invalid_rollout=False,
            empty_valid_object_set=not valid_objects,
            truncated=bool(parsed.truncated),
        )

    def render_target(
        self,
        accepted_objects: Sequence[Stage2RolloutObject],
        fn_objects: Sequence[_AppendObjectLike],
    ) -> str:
        """Render a closed legacy CoordJSON target."""

        from src.trainers.rollout_matching.parsing import serialize_append_fragment

        prefix = '{"objects": ['
        if accepted_objects:
            accepted_gt = [
                _gt_object_from_stage2_object(index=index, obj=obj)
                for index, obj in enumerate(accepted_objects)
            ]
            prefix = prefix + serialize_append_fragment(
                fn_objects=accepted_gt,
                prefix_text=prefix,
                object_field_order="desc_first",
            )[:-2]
        validated_fn_objects = [
            _legacy_append_object_from_append_object(index=index, obj=obj)
            for index, obj in enumerate(fn_objects)
        ]
        return prefix + serialize_append_fragment(
            fn_objects=validated_fn_objects,
            prefix_text=prefix,
            object_field_order="desc_first",
        )

    def build_rollout_correction_target(
        self,
        parse_result: Stage2RolloutParseResult,
        *,
        fn_objects: Sequence[_AppendObjectLike],
    ) -> Stage2RolloutTarget:
        """Append FN objects to a legacy CoordJSON rollout prefix."""

        from src.trainers.rollout_matching.parsing import serialize_append_fragment

        # validate parser/template alignment
        if parse_result.template_family != "coordjson":
            raise Stage2RolloutTemplateMismatchError(
                "coordjson appender received non-coordjson parse result"
            )
        prefix_text = parse_result.append_prefix_text
        if prefix_text is None:
            if not parse_result.invalid_rollout:
                raise ValueError(
                    "coordjson appender requires append prefix text for non-invalid "
                    "parse results"
                )
            prefix_text = '{"objects": ['
        prefix_text = str(prefix_text)
        validated_fn_objects = [
            _legacy_append_object_from_append_object(index=index, obj=obj)
            for index, obj in enumerate(fn_objects)
        ]
        text = prefix_text + serialize_append_fragment(
            fn_objects=validated_fn_objects,
            prefix_text=prefix_text,
            object_field_order="desc_first",
        )
        return Stage2RolloutTarget(
            text=text,
            metadata=Stage2RolloutTargetMetadata(
                template_family="coordjson",
                parser_id=self.policy.parser_id,
                append_policy_id=self.policy.append_policy_id,
                rollout_context="legacy_coordjson_append",
                fallback_reason=None,
                fallback_loss_weight=0.0,
                counts_as_valid_rollout=not parse_result.invalid_rollout,
            ),
        )

    def _parse_result(
        self,
        *,
        response_text: str,
        valid_objects: tuple[Stage2RolloutObject, ...],
        prefix_text: str,
        invalid_rollout: bool,
        empty_valid_object_set: bool,
        truncated: bool,
    ) -> Stage2RolloutParseResult:
        """Return a CoordJSON parse result with legacy prefix metadata."""

        return Stage2RolloutParseResult(
            template_family="coordjson",
            parser_id=self.policy.parser_id,
            response_text=str(response_text),
            valid_objects=valid_objects,
            invalid_rollout=bool(invalid_rollout),
            empty_valid_object_set=bool(empty_valid_object_set),
            truncated=bool(truncated),
            fallback_reason=None,
            append_prefix_text=str(prefix_text),
            metadata=self.policy.diagnostics_metadata,
        )


def _looks_like_coordjson(text: str) -> bool:
    stripped = str(text).lstrip()
    if _looks_like_compact_full(stripped):
        return False
    return bool(
        stripped.startswith("{")
        or stripped.startswith("[")
        or '"objects"' in stripped
        or '"bbox_2d"' in stripped
        or '"poly"' in stripped
    )


def _looks_like_compact_full(text: str) -> bool:
    return OBJECT_REF_START_TOKEN in str(text) or BOX_START_TOKEN in str(text)


def _stage2_object_from_compact_entry(
    *,
    index: int,
    entry: Mapping[str, Any],
) -> Stage2RolloutObject:
    bbox_tokens = tuple(str(token) for token in entry.get("bbox_2d", ()))
    return Stage2RolloutObject(
        object_id=f"rollout[{int(index)}]",
        index=int(index),
        desc=str(entry.get("desc", "")),
        bbox_tokens=bbox_tokens,
        bbox_norm1000=_norm1000_from_coord_tokens(bbox_tokens),
        geom_type="bbox_2d",
        provenance="rollout_accepted",
    )


def _stage2_object_from_coordjson_record(
    *,
    index: int,
    record: Any,
) -> Stage2RolloutObject:
    values = _validate_norm1000_bbox(record.geometry_values)
    tokens = tuple(f"<|coord_{int(value)}|>" for value in values)
    return Stage2RolloutObject(
        object_id=f"objects[{int(record.index)}]",
        index=int(record.index),
        desc=str(record.desc),
        bbox_tokens=tokens if record.geometry_key == "bbox_2d" else None,
        bbox_norm1000=values if record.geometry_key == "bbox_2d" else None,
        geom_type=str(record.geometry_key),
        provenance="rollout_accepted",
        metadata={"record_index": int(index)},
    )


def _norm1000_from_coord_tokens(tokens: Sequence[str]) -> tuple[int, ...]:
    values: list[int] = []
    for token in tokens:
        match = _COORD_TOKEN_RE.fullmatch(str(token))
        if match is None:
            raise ValueError(f"invalid compact coordinate token: {token!r}")
        values.append(int(match.group(1)))
    return _validate_norm1000_bbox(values)


def _coord_tokens_from_norm1000(values: Sequence[int]) -> tuple[str, ...]:
    bbox = _validate_norm1000_bbox(values)
    return tuple(f"<|coord_{value}|>" for value in bbox)


def _validate_norm1000_bbox(values: Sequence[object]) -> tuple[int, int, int, int]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("norm1000 bbox must be a sequence of four integer values")
    if len(values) != 4:
        raise ValueError("norm1000 bbox must contain exactly four values")

    normalized: list[int] = []
    for raw_value in values:
        if isinstance(raw_value, bool):
            raise ValueError("norm1000 bbox values must not be bools")
        try:
            value = int(integer_index(raw_value))
        except TypeError:
            raise ValueError("norm1000 bbox values must be integer-like") from None
        if value < 0 or value > 999:
            raise ValueError("norm1000 bbox values must be in [0, 999]")
        normalized.append(value)

    return (normalized[0], normalized[1], normalized[2], normalized[3])


def _compact_payload_from_stage2_object(obj: Stage2RolloutObject) -> dict[str, object]:
    if obj.geom_type != "bbox_2d":
        raise ValueError("compact_full rollout codec supports bbox_2d objects only")
    bbox_tokens = obj.bbox_tokens
    if bbox_tokens is None:
        if obj.bbox_norm1000 is None:
            raise ValueError("compact_full object requires bbox tokens or norm1000 bbox")
        bbox_tokens = _coord_tokens_from_norm1000(obj.bbox_norm1000)
    else:
        _norm1000_from_coord_tokens(bbox_tokens)
    return {
        "desc": str(obj.desc),
        "bbox_2d": list(bbox_tokens),
    }


def _compact_payload_from_gt_object(obj: _AppendObjectLike) -> dict[str, object]:
    if obj.geom_type != "bbox_2d":
        raise ValueError("compact_full rollout codec supports bbox_2d GT objects only")
    return {
        "desc": str(obj.desc),
        "bbox_2d": list(_coord_tokens_from_norm1000(obj.points_norm1000)),
    }


def _gt_object_from_stage2_object(
    *, index: int, obj: Stage2RolloutObject
) -> _LegacyAppendObject:
    if obj.geom_type != "bbox_2d":
        raise ValueError("legacy CoordJSON target rendering supports bbox_2d here")
    if obj.bbox_norm1000 is None:
        if obj.bbox_tokens is None:
            raise ValueError("CoordJSON object requires bbox tokens or norm1000 bbox")
        points = list(_norm1000_from_coord_tokens(obj.bbox_tokens))
    else:
        points = list(_validate_norm1000_bbox(obj.bbox_norm1000))
    return _LegacyAppendObject(
        index=int(index),
        geom_type="bbox_2d",
        points_norm1000=points,
        desc=str(obj.desc),
    )


def _legacy_append_object_from_append_object(
    *, index: int, obj: _AppendObjectLike
) -> _LegacyAppendObject:
    if obj.geom_type != "bbox_2d":
        raise ValueError("legacy CoordJSON target rendering supports bbox_2d here")
    points = list(_validate_norm1000_bbox(obj.points_norm1000))
    return _LegacyAppendObject(
        index=int(index),
        geom_type="bbox_2d",
        points_norm1000=points,
        desc=str(obj.desc),
    )


def _coordjson_append_ready_prefix(text: str, parsed: Any) -> str:
    start = int(parsed.container_start or 0)
    cut_candidates = [int(parsed.objects_array_open_cut)]
    cut_candidates.extend(int(value) for value in parsed.record_end_cuts if int(value) > 0)
    cut = max(cut_candidates)
    return str(text)[start:cut]


__all__ = [
    "CompactFullRolloutCodec",
    "FALLBACK_GT_FN_APPEND_ONLY",
    "LegacyCoordJsonRolloutCodec",
    "Stage2RolloutDecodePolicy",
    "Stage2RolloutDiagnosticSummary",
    "Stage2RolloutObject",
    "Stage2RolloutParseResult",
    "Stage2RolloutTarget",
    "Stage2RolloutTargetMetadata",
    "Stage2RolloutTemplateFamily",
    "Stage2RolloutTemplateMismatchError",
    "Stage2RolloutTemplatePolicy",
    "calculate_stage2_rollout_diagnostics",
    "resolve_stage2_rollout_template_policy",
]
