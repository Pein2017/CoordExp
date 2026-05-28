"""Shared parser-policy guards for metric-bearing inference artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Literal, Mapping, Sequence

ParserPolicy = Literal["strict", "diagnostic"]


@dataclass(frozen=True)
class DetectionParserResult:
    """Parsed detection output plus the policy that makes it metric-safe or not."""

    predictions: tuple[Mapping[str, Any], ...]
    parser_id: str
    parser_policy: ParserPolicy
    metric_bearing: bool
    salvage_recovered: bool = False
    errors: tuple[str, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.parser_policy not in {"strict", "diagnostic"}:
            raise ValueError(
                "parser_policy must be one of: strict, diagnostic "
                f"(got {self.parser_policy!r})"
            )
        if not isinstance(self.metric_bearing, bool):
            raise TypeError("metric_bearing must be a bool")
        if not isinstance(self.salvage_recovered, bool):
            raise TypeError("salvage_recovered must be a bool")
        if not isinstance(self.parser_id, str) or not self.parser_id:
            raise ValueError("parser_id must be a non-empty string")
        if self.parser_policy == "diagnostic" and self.metric_bearing:
            raise ValueError("diagnostic parser results cannot be metric-bearing")
        if self.salvage_recovered and self.metric_bearing:
            raise ValueError(
                "salvage-recovered parser results cannot be metric-bearing"
            )
        object.__setattr__(self, "predictions", tuple(self.predictions))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors))

    def to_artifact_metadata(self) -> dict[str, Any]:
        """Return compact parser provenance fields for artifacts and summaries."""

        return {
            "parser_id": self.parser_id,
            "parser_policy": self.parser_policy,
            "metric_bearing": bool(self.metric_bearing),
            "salvage_recovered": bool(self.salvage_recovered),
            "parser_error_count": len(self.errors),
        }


def strict_parser_result(
    *,
    predictions: Sequence[Mapping[str, Any]],
    parser_id: str,
    errors: Sequence[str] = (),
    diagnostics: Mapping[str, Any] | None = None,
) -> DetectionParserResult:
    """Build a strict parser result that may feed official metrics."""

    return DetectionParserResult(
        predictions=tuple(predictions),
        parser_id=parser_id,
        parser_policy="strict",
        metric_bearing=True,
        salvage_recovered=False,
        errors=tuple(errors),
        diagnostics=dict(diagnostics or {}),
    )


def diagnostic_parser_result(
    *,
    predictions: Sequence[Mapping[str, Any]],
    parser_id: str,
    errors: Sequence[str] = (),
    diagnostics: Mapping[str, Any] | None = None,
    salvage_recovered: bool = True,
) -> DetectionParserResult:
    """Build a non-metric diagnostic/salvage parser result."""

    return DetectionParserResult(
        predictions=tuple(predictions),
        parser_id=parser_id,
        parser_policy="diagnostic",
        metric_bearing=False,
        salvage_recovered=bool(salvage_recovered),
        errors=tuple(errors),
        diagnostics=dict(diagnostics or {}),
    )


def require_metric_bearing(
    result: DetectionParserResult,
    *,
    consumer: str,
) -> DetectionParserResult:
    """Fail fast when a metric/export consumer is handed diagnostic parser output."""

    if not isinstance(result, DetectionParserResult):
        raise TypeError("result must be a DetectionParserResult")
    if not result.metric_bearing:
        raise ValueError(
            "metric_bearing=false: "
            f"{consumer} requires strict metric-bearing parser output "
            f"(parser_id={result.parser_id!r}, parser_policy={result.parser_policy!r})"
        )
    if result.parser_policy != "strict":
        raise ValueError(
            f"{consumer} requires strict parser output "
            f"(parser_policy={result.parser_policy!r})"
        )
    if result.salvage_recovered:
        raise ValueError(f"{consumer} rejects salvage-recovered parser output")
    return result


@dataclass(frozen=True)
class Stage2ParsedRolloutPredictions:
    """Metric-bearing Stage-2 eval parser adapter output."""

    parse: Any
    pred_meta: tuple[Any, ...]
    preds: tuple[Any, ...]
    pred_objects_dump: tuple[dict[str, Any], ...]
    parser_result: DetectionParserResult


def _stage2_parser_result_from_parse(
    *,
    parse: Any,
    parser_id: str,
    predictions: Sequence[Mapping[str, Any]],
) -> DetectionParserResult:
    dropped_invalid = int(getattr(parse, "dropped_invalid", 0) or 0)
    dropped_ambiguous = int(getattr(parse, "dropped_ambiguous", 0) or 0)
    invalid_rollout = bool(getattr(parse, "invalid_rollout", False))
    truncated = bool(getattr(parse, "truncated", False))
    empty_valid_object_set = bool(getattr(parse, "empty_valid_object_set", False))
    fallback_reason_raw = getattr(parse, "fallback_reason", None)
    fallback_reason = str(fallback_reason_raw or "").strip()
    strict_policy_advertised = (
        str(getattr(parse, "parser_policy", "") or "").strip().lower() == "strict"
    )
    salvage_recovered = (
        (
            str(parser_id).strip().lower() == "coordjson"
            and not strict_policy_advertised
        )
        or fallback_reason == "compact_full_salvage"
    )

    diagnostics = {
        "dropped_invalid": int(dropped_invalid),
        "dropped_ambiguous": int(dropped_ambiguous),
        "invalid_rollout": bool(invalid_rollout),
        "truncated": bool(truncated),
        "empty_valid_object_set": bool(empty_valid_object_set),
        "fallback_reason": fallback_reason or None,
        "salvage_recovered": bool(salvage_recovered),
    }
    errors: list[str] = []
    if invalid_rollout:
        errors.append("invalid_rollout")
    if dropped_invalid > 0:
        errors.append("dropped_invalid")
    if dropped_ambiguous > 0:
        errors.append("dropped_ambiguous")
    if truncated:
        errors.append("truncated")
    if empty_valid_object_set:
        errors.append("empty_valid_object_set")
    if fallback_reason:
        errors.append(f"fallback:{fallback_reason}")
    if salvage_recovered:
        errors.append("salvage_parser_policy")

    if errors:
        return diagnostic_parser_result(
            predictions=tuple(predictions),
            parser_id=parser_id,
            errors=tuple(errors),
            diagnostics=diagnostics,
            salvage_recovered=bool(salvage_recovered),
        )
    return strict_parser_result(
        predictions=tuple(predictions),
        parser_id=parser_id,
        diagnostics=diagnostics,
    )


def parse_stage2_detection_rollout_predictions(
    *,
    tokenizer: Any,
    response_token_ids: Sequence[int],
    response_text: str | None,
    rollout_template_policy: Any,
    object_field_order: str,
    coord_id_to_bin: Mapping[int, int],
    gt_object_factory: Callable[..., Any],
    compact_rollout_codec_factory: Callable[[Any], Any],
    parse_rollout_for_matching_fn: Callable[..., Any],
    points_from_coord_tokens_fn: Callable[..., Any],
) -> Stage2ParsedRolloutPredictions:
    """Parse Stage-2 eval rollout text/tokens into metric-ready prediction objects.

    The helper owns the parser-branch semantics without importing trainer-local
    matching or target classes; callers inject the few trainer-owned factories
    needed for object construction and legacy coord-token parsing.
    """

    resp_ids = tuple(int(t) for t in response_token_ids)
    pred_meta: list[Any] = []
    preds: list[Any] = []
    pred_objs_dump: list[dict[str, Any]] = []

    if getattr(rollout_template_policy, "template_family", None) == "compact_full":
        text = str(response_text or "")
        if not text:
            text = tokenizer.decode(
                [int(t) for t in resp_ids],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        parse = compact_rollout_codec_factory(rollout_template_policy).parse(text)
        geometry_drop_count = 0
        for pobj in list(parse.valid_objects):
            if pobj.geom_type != "bbox_2d" or pobj.bbox_norm1000 is None:
                geometry_drop_count += 1
                continue
            try:
                pts = [int(x) for x in pobj.bbox_norm1000]
            except (TypeError, ValueError):
                geometry_drop_count += 1
                continue
            if len(pts) != 4 or pts[2] <= pts[0] or pts[3] <= pts[1]:
                geometry_drop_count += 1
                continue
            pred = gt_object_factory(
                index=int(pobj.index),
                geom_type="bbox_2d",
                points_norm1000=pts,
                desc=str(pobj.desc),
            )
            pred_meta.append(pred)
            preds.append(pred)
            pred_objs_dump.append(
                {
                    "key": str(pobj.object_id),
                    "index": int(pobj.index),
                    "geom_type": "bbox_2d",
                    "points_norm1000": list(pts),
                    "desc": str(pobj.desc),
                }
            )
        if geometry_drop_count:
            drop_reasons = dict(parse.dropped_invalid_by_reason)
            drop_reasons["bbox_invalid"] = int(
                drop_reasons.get("bbox_invalid", 0)
            ) + int(geometry_drop_count)
            parse = replace(
                parse,
                dropped_invalid=(
                    int(parse.dropped_invalid) + int(geometry_drop_count)
                ),
                dropped_invalid_by_reason=drop_reasons,
                empty_valid_object_set=not preds,
                fallback_reason=("empty_valid_object_set" if not preds else None),
            )
        parse = replace(parse, response_token_ids=resp_ids)
        parser_id = str(getattr(parse, "parser_id", "") or "compact_full")
    else:
        parse = parse_rollout_for_matching_fn(
            tokenizer=tokenizer,
            response_token_ids=resp_ids,
            object_field_order=object_field_order,
        )
        parsed_pred_meta = list(parse.valid_objects)
        for pobj in parsed_pred_meta:
            pts = points_from_coord_tokens_fn(
                response_token_ids=parse.response_token_ids,
                coord_token_indices=pobj.coord_token_indices,
                coord_id_to_bin=coord_id_to_bin,
            )
            if pts is None:
                continue
            pred_meta.append(pobj)
            preds.append(
                gt_object_factory(
                    index=int(pobj.index),
                    geom_type=pobj.geom_type,
                    points_norm1000=pts,
                    desc="",
                )
            )
            pred_objs_dump.append(
                {
                    "key": str(getattr(pobj, "key", "") or ""),
                    "index": int(pobj.index),
                    "geom_type": str(pobj.geom_type),
                    "points_norm1000": list(pts),
                    "desc": str(getattr(pobj, "desc", "") or ""),
                }
            )
        parser_id = str(getattr(parse, "parser_id", "") or "coordjson")

    return Stage2ParsedRolloutPredictions(
        parse=parse,
        pred_meta=tuple(pred_meta),
        preds=tuple(preds),
        pred_objects_dump=tuple(pred_objs_dump),
        parser_result=_stage2_parser_result_from_parse(
            parse=parse,
            parser_id=parser_id,
            predictions=tuple(pred_objs_dump),
        ),
    )


__all__ = [
    "DetectionParserResult",
    "ParserPolicy",
    "Stage2ParsedRolloutPredictions",
    "diagnostic_parser_result",
    "parse_stage2_detection_rollout_predictions",
    "require_metric_bearing",
    "strict_parser_result",
]
