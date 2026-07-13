"""Canonical parse-and-score evidence for spatial-scope predictions."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from collections.abc import Mapping
from typing import Any, Literal

from src.analysis.spatial_scope_history.cohort_ledger import sha256_payload
from src.analysis.spatial_scope_history.execution_evidence import (
    ExecutionEvidenceEnvelope,
)
from src.analysis.spatial_scope_history.spatial import (
    SpatialGrid,
    SpatialGridSpec,
    spatial_variant_mode_for_arm,
)
from src.common.errors import ArtifactContractError, DataContractError
from src.data.geometry import coord_bins_to_pixel_xyxy
from src.eval.detection_categories import normalize_coco_category_name
from src.inference.backend import (
    DecodeResult,
    TokenTrace,
    canonical_float32_logprob,
)
from src.inference.parsing import (
    PARSER_ID,
    PARSER_POLICY,
    parse_compact_object_box_closed,
)
from src.inference.scoring import (
    PRED_SCORE_VERSION,
    SCORE_POLICY,
    SCORE_POLICY_FINGERPRINT,
    score_prediction,
)
from src.templates.renderer import (
    BOX_END_TOKEN,
    BOX_START_TOKEN,
    OBJECT_REF_END_TOKEN,
    OBJECT_REF_START_TOKEN,
)


CANONICAL_PARSE_SCORE_RECEIPT_SCHEMA_VERSION = (
    "spatial_scope_history.canonical_parse_score_receipt.v1"
)
CoordinateSource = Literal["source_canvas", "spatial_local_canvas"]
PixelBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class CanonicalParseScoreReceipt:
    """Immutable proof that one prediction came from canonical parse and score."""

    execution_evidence_fingerprint: str
    request_id: str
    decode_receipt_fingerprint: str
    decode_result_sha256: str
    raw_generated_text_sha256: str
    parser_text_sha256: str
    generated_token_identifiers_sha256: str
    token_trace_sha256: str
    full_token_trace_sha256: str
    parser_id: str
    parser_policy: str
    parse_row_index: int
    parse_status: str
    prediction_validity: Literal["accepted_metric_prediction"]
    drop_reason: None
    object_span_id: str
    generated_row_index: int
    span_char_start: int
    span_char_end: int
    raw_span_text: str
    raw_span_sha256: str
    category_span_text: str
    category_text: str
    normalized_category_name: str
    coordinate_tokens: tuple[str, str, str, str]
    coordinate_bins: tuple[int, int, int, int]
    coordinate_source: CoordinateSource
    coordinate_extent_width: int
    coordinate_extent_height: int
    parsed_bbox_xyxy: PixelBox
    score_policy_id: str
    score_policy_fingerprint: str
    prediction_score_version: int
    selected_generated_step_indices: tuple[int, ...]
    selected_token_ids: tuple[int, ...]
    selected_token_text: tuple[str, ...]
    selected_logprobs_float32: tuple[float, ...]
    selected_token_replay_sha256: str
    score: float
    receipt_sha256: str | None = None
    schema_version: str = CANONICAL_PARSE_SCORE_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CANONICAL_PARSE_SCORE_RECEIPT_SCHEMA_VERSION:
            _fail("parse-and-score receipt schema is unsupported", "schema")
        for field_name in (
            "execution_evidence_fingerprint",
            "decode_receipt_fingerprint",
            "decode_result_sha256",
            "raw_generated_text_sha256",
            "parser_text_sha256",
            "generated_token_identifiers_sha256",
            "token_trace_sha256",
            "full_token_trace_sha256",
            "raw_span_sha256",
            "score_policy_fingerprint",
            "selected_token_replay_sha256",
        ):
            _require_sha256(getattr(self, field_name), field=field_name)
        if self.parser_id != PARSER_ID or self.parser_policy != PARSER_POLICY:
            _fail("receipt does not name the canonical parser", "parser_identity")
        if self.prediction_validity != "accepted_metric_prediction":
            _fail(
                "only accepted predictions may enter metric normalization", "validity"
            )
        if self.drop_reason is not None:
            _fail("accepted prediction cannot carry a drop reason", "drop_reason")
        if self.parse_status not in {"accepted", "accepted_with_drops"}:
            _fail("accepted prediction has an invalid parse status", "parse_status")
        _require_nonnegative_integer(self.parse_row_index, field="parse_row_index")
        _require_nonnegative_integer(
            self.generated_row_index, field="generated_row_index"
        )
        if self.object_span_id != (
            f"{self.request_id}:span-{self.generated_row_index}"
        ):
            _fail("object span identity differs from generated row index", "row_index")
        if self.span_char_start < 0 or self.span_char_end <= self.span_char_start:
            _fail("generated object span is empty or malformed", "span")
        if not self.object_span_id.strip() or not self.category_text.strip():
            _fail("object span and category text must be nonempty", "text")
        if self.span_char_end - self.span_char_start != len(
            self.raw_span_text
        ) or self.raw_span_sha256 != _sha256_text(self.raw_span_text):
            _fail("raw object span text or digest is inconsistent", "raw_span")
        expected_raw_span = (
            OBJECT_REF_START_TOKEN
            + self.category_span_text
            + OBJECT_REF_END_TOKEN
            + BOX_START_TOKEN
            + "".join(self.coordinate_tokens)
            + BOX_END_TOKEN
        )
        if (
            self.raw_span_text != expected_raw_span
            or self.category_text != self.category_span_text.strip()
        ):
            _fail("category and coordinate evidence differ from raw span", "raw_span")
        if (
            normalize_coco_category_name(self.category_text)
            != self.normalized_category_name
        ):
            _fail("normalized category differs from parser category text", "category")
        if self.coordinate_source not in {"source_canvas", "spatial_local_canvas"}:
            _fail("coordinate source is unsupported", "coordinate_source")
        if self.coordinate_extent_width <= 0 or self.coordinate_extent_height <= 0:
            _fail("coordinate extents must be positive", "coordinate_extent")
        expected_tokens = tuple(f"<|coord_{value}|>" for value in self.coordinate_bins)
        if self.coordinate_tokens != expected_tokens:
            _fail("coordinate tokens differ from coordinate bins", "coordinate_tokens")
        expected_box = tuple(
            float(value)
            for value in coord_bins_to_pixel_xyxy(
                self.coordinate_bins,
                image_width=self.coordinate_extent_width,
                image_height=self.coordinate_extent_height,
                field="parse_score_receipt.coordinate_bins",
            )
        )
        if self.parsed_bbox_xyxy != expected_box:
            _fail("parsed box does not reproduce from coordinate bins", "bbox")
        if (
            self.score_policy_id != SCORE_POLICY["id"]
            or self.score_policy_fingerprint != SCORE_POLICY_FINGERPRINT
            or self.prediction_score_version != PRED_SCORE_VERSION
        ):
            _fail("receipt does not name the canonical score policy", "score_policy")
        selected_count = int(SCORE_POLICY["selected_token_count"])
        selected_fields = (
            self.selected_generated_step_indices,
            self.selected_token_ids,
            self.selected_token_text,
            self.selected_logprobs_float32,
        )
        if any(len(values) != selected_count for values in selected_fields):
            _fail("selected-token replay has the wrong cardinality", "score_replay")
        expected_selected_text = (
            OBJECT_REF_START_TOKEN,
            OBJECT_REF_END_TOKEN,
            BOX_START_TOKEN,
            *self.coordinate_tokens,
            BOX_END_TOKEN,
        )
        if (
            self.selected_token_text != expected_selected_text
            or tuple(sorted(self.selected_generated_step_indices))
            != self.selected_generated_step_indices
            or len(set(self.selected_generated_step_indices)) != selected_count
        ):
            _fail(
                "selected-token replay does not match parsed span order", "score_replay"
            )
        replay_payload = self.selected_token_replay_payload()
        if self.selected_token_replay_sha256 != sha256_payload(replay_payload):
            _fail("selected-token replay digest is invalid", "score_replay_digest")
        recomputed_score = _score_from_logprobs(self.selected_logprobs_float32)
        if self.score != recomputed_score:
            _fail("prediction score does not exactly recompute", "score_recompute")
        if not 0.0 < self.score <= 1.0:
            _fail("prediction score must lie in (0, 1]", "score_range")
        if self.receipt_sha256 is None:
            object.__setattr__(
                self, "receipt_sha256", sha256_payload(self.identity_payload())
            )
        _require_sha256(self.receipt_sha256, field="receipt_sha256")
        if self.receipt_sha256 != sha256_payload(self.identity_payload()):
            _fail("parse-and-score receipt digest is invalid", "receipt_digest")

    def selected_token_replay_payload(self) -> dict[str, Any]:
        return {
            "generated_step_indices": list(self.selected_generated_step_indices),
            "score_policy_fingerprint": self.score_policy_fingerprint,
            "selected_logprobs_float32": list(self.selected_logprobs_float32),
            "token_ids": list(self.selected_token_ids),
            "token_text": list(self.selected_token_text),
        }

    def identity_payload(self) -> dict[str, Any]:
        payload = self.to_artifact_dict()
        payload.pop("receipt_sha256")
        return payload

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "execution_evidence_fingerprint": self.execution_evidence_fingerprint,
            "request_id": self.request_id,
            "decode_receipt_fingerprint": self.decode_receipt_fingerprint,
            "decode_result_sha256": self.decode_result_sha256,
            "raw_generated_text_sha256": self.raw_generated_text_sha256,
            "parser_text_sha256": self.parser_text_sha256,
            "generated_token_identifiers_sha256": self.generated_token_identifiers_sha256,
            "token_trace_sha256": self.token_trace_sha256,
            "full_token_trace_sha256": self.full_token_trace_sha256,
            "parser_id": self.parser_id,
            "parser_policy": self.parser_policy,
            "parse_row_index": self.parse_row_index,
            "parse_status": self.parse_status,
            "prediction_validity": self.prediction_validity,
            "drop_reason": self.drop_reason,
            "object_span_id": self.object_span_id,
            "generated_row_index": self.generated_row_index,
            "span_char_start": self.span_char_start,
            "span_char_end": self.span_char_end,
            "raw_span_text": self.raw_span_text,
            "raw_span_sha256": self.raw_span_sha256,
            "category_span_text": self.category_span_text,
            "category_text": self.category_text,
            "normalized_category_name": self.normalized_category_name,
            "coordinate_tokens": list(self.coordinate_tokens),
            "coordinate_bins": list(self.coordinate_bins),
            "coordinate_source": self.coordinate_source,
            "coordinate_extent_width": self.coordinate_extent_width,
            "coordinate_extent_height": self.coordinate_extent_height,
            "parsed_bbox_xyxy": list(self.parsed_bbox_xyxy),
            "score_policy_id": self.score_policy_id,
            "score_policy_fingerprint": self.score_policy_fingerprint,
            "prediction_score_version": self.prediction_score_version,
            "selected_generated_step_indices": list(
                self.selected_generated_step_indices
            ),
            "selected_token_ids": list(self.selected_token_ids),
            "selected_token_text": list(self.selected_token_text),
            "selected_logprobs_float32": list(self.selected_logprobs_float32),
            "selected_token_replay_sha256": self.selected_token_replay_sha256,
            "score": self.score,
            "receipt_sha256": self.receipt_sha256,
        }

    @classmethod
    def from_artifact_dict(
        cls, value: Mapping[str, Any]
    ) -> CanonicalParseScoreReceipt:
        """Rehydrate a receipt while re-running canonical replay validation."""

        fields = {
            "category_span_text",
            "category_text",
            "coordinate_bins",
            "coordinate_extent_height",
            "coordinate_extent_width",
            "coordinate_source",
            "coordinate_tokens",
            "decode_receipt_fingerprint",
            "decode_result_sha256",
            "drop_reason",
            "execution_evidence_fingerprint",
            "full_token_trace_sha256",
            "generated_row_index",
            "generated_token_identifiers_sha256",
            "normalized_category_name",
            "object_span_id",
            "parse_row_index",
            "parse_status",
            "parsed_bbox_xyxy",
            "parser_id",
            "parser_policy",
            "parser_text_sha256",
            "prediction_score_version",
            "prediction_validity",
            "raw_generated_text_sha256",
            "raw_span_sha256",
            "raw_span_text",
            "receipt_sha256",
            "request_id",
            "schema_version",
            "score",
            "score_policy_fingerprint",
            "score_policy_id",
            "selected_generated_step_indices",
            "selected_logprobs_float32",
            "selected_token_ids",
            "selected_token_replay_sha256",
            "selected_token_text",
            "span_char_end",
            "span_char_start",
            "token_trace_sha256",
        }
        observed = set(value)
        if observed != fields:
            _fail(
                "parse-and-score receipt keys are not exact",
                "artifact_keys",
            )
        payload = dict(value)
        for field_name in (
            "coordinate_bins",
            "coordinate_tokens",
            "parsed_bbox_xyxy",
            "selected_generated_step_indices",
            "selected_logprobs_float32",
            "selected_token_ids",
            "selected_token_text",
        ):
            payload[field_name] = tuple(payload[field_name])
        return cls(**payload)


def build_canonical_parse_score_receipts(
    *,
    execution_evidence: ExecutionEvidenceEnvelope,
    decode_result: DecodeResult,
) -> tuple[CanonicalParseScoreReceipt, ...]:
    """Replay canonical parser and scorer over one envelope-bound decode result."""

    _validate_decode_result_association(
        execution_evidence=execution_evidence,
        decode_result=decode_result,
    )
    coordinate_source, width, height = _coordinate_frame(execution_evidence)
    parse_row_index = execution_evidence.scheduled_request.schedule_index
    parsed = parse_compact_object_box_closed(
        decode_result.parser_text,
        row_id=execution_evidence.request_id,
        row_index=parse_row_index,
        image_width=width,
        image_height=height,
    )
    canonical_trace = [_canonical_trace(row) for row in decode_result.token_trace]
    result_sha256 = sha256_payload(decode_result.to_artifact_dict())
    full_trace_sha256 = sha256_payload(
        [_token_trace_payload(row) for row in canonical_trace]
    )
    receipts: list[CanonicalParseScoreReceipt] = []
    for prediction in parsed.predictions:
        scored = score_prediction(
            row_id=execution_evidence.request_id,
            prediction=prediction,
            token_trace=canonical_trace,
        )
        replay = scored.replay
        coordinate_tokens = tuple(
            str(item["text"]) for item in prediction["coord_token_spans"]
        )
        raw_span_text = str(prediction["raw_span_text"])
        category_span_text = raw_span_text[
            len(OBJECT_REF_START_TOKEN) : raw_span_text.index(OBJECT_REF_END_TOKEN)
        ]
        coordinate_bins = tuple(int(value) for value in prediction["coord_bins"])
        selected_logprobs = tuple(float(value) for value in replay["selected_logprobs"])
        replay_payload = {
            "generated_step_indices": [
                int(value) for value in replay["generated_step_indices"]
            ],
            "score_policy_fingerprint": SCORE_POLICY_FINGERPRINT,
            "selected_logprobs_float32": list(selected_logprobs),
            "token_ids": [int(value) for value in replay["token_ids"]],
            "token_text": [str(value) for value in replay["token_text"]],
        }
        receipts.append(
            CanonicalParseScoreReceipt(
                execution_evidence_fingerprint=execution_evidence.fingerprint,
                request_id=execution_evidence.request_id,
                decode_receipt_fingerprint=execution_evidence.decode_receipt_fingerprint,
                decode_result_sha256=result_sha256,
                raw_generated_text_sha256=_sha256_text(
                    decode_result.raw_generated_text
                ),
                parser_text_sha256=_sha256_text(decode_result.parser_text),
                generated_token_identifiers_sha256=(
                    execution_evidence.generated_output_token_sha256
                ),
                token_trace_sha256=execution_evidence.token_trace_sha256,
                full_token_trace_sha256=full_trace_sha256,
                parser_id=parsed.parser_id,
                parser_policy=parsed.parser_policy,
                parse_row_index=parsed.row_index,
                parse_status=parsed.parse_status,
                prediction_validity="accepted_metric_prediction",
                drop_reason=None,
                object_span_id=str(prediction["object_span_id"]),
                generated_row_index=int(prediction["generated_order"]),
                span_char_start=int(prediction["char_start"]),
                span_char_end=int(prediction["char_end"]),
                raw_span_text=raw_span_text,
                raw_span_sha256=str(prediction["raw_span_sha256"]),
                category_span_text=category_span_text,
                category_text=str(prediction["description"]),
                normalized_category_name=normalize_coco_category_name(
                    prediction["description"]
                ),
                coordinate_tokens=coordinate_tokens,  # type: ignore[arg-type]
                coordinate_bins=coordinate_bins,  # type: ignore[arg-type]
                coordinate_source=coordinate_source,
                coordinate_extent_width=width,
                coordinate_extent_height=height,
                parsed_bbox_xyxy=tuple(float(value) for value in prediction["bbox"]),  # type: ignore[arg-type]
                score_policy_id=str(SCORE_POLICY["id"]),
                score_policy_fingerprint=SCORE_POLICY_FINGERPRINT,
                prediction_score_version=PRED_SCORE_VERSION,
                selected_generated_step_indices=tuple(
                    replay_payload["generated_step_indices"]
                ),
                selected_token_ids=tuple(replay_payload["token_ids"]),
                selected_token_text=tuple(replay_payload["token_text"]),
                selected_logprobs_float32=selected_logprobs,
                selected_token_replay_sha256=sha256_payload(replay_payload),
                score=_score_from_logprobs(selected_logprobs),
            )
        )
    return tuple(receipts)


def validate_parse_score_receipt_association(
    receipt: CanonicalParseScoreReceipt,
    *,
    execution_evidence: ExecutionEvidenceEnvelope,
) -> None:
    """Fail closed if a parse receipt is replayed with another execution."""

    if not isinstance(receipt, CanonicalParseScoreReceipt):
        _fail("normalization requires a canonical parse-and-score receipt", "type")
    expected = {
        "execution_evidence_fingerprint": execution_evidence.fingerprint,
        "request_id": execution_evidence.request_id,
        "decode_receipt_fingerprint": execution_evidence.decode_receipt_fingerprint,
        "generated_token_identifiers_sha256": (
            execution_evidence.generated_output_token_sha256
        ),
        "token_trace_sha256": execution_evidence.token_trace_sha256,
    }
    mismatches = {
        field: {"expected": value, "observed": getattr(receipt, field)}
        for field, value in expected.items()
        if getattr(receipt, field) != value
    }
    if mismatches:
        raise DataContractError(
            "parse-and-score receipt belongs to another execution envelope",
            code="analysis.parse_score_receipt.association",
            context={"mismatches": mismatches},
        )


def _validate_decode_result_association(
    *,
    execution_evidence: ExecutionEvidenceEnvelope,
    decode_result: DecodeResult,
) -> None:
    try:
        decode_result.validate_for_scored()
    except Exception as exc:
        if isinstance(exc, (DataContractError, ArtifactContractError)):
            raise
        raise DataContractError(
            "decode result failed canonical scored-result validation",
            code="analysis.parse_score_receipt.decode_result",
            cause=exc,
        ) from exc
    receipt = decode_result.execution_receipt
    if receipt is None:
        _fail("decode result has no execution receipt", "decode_receipt")
    expected = {
        "request_id": execution_evidence.request_id,
        "receipt_fingerprint": execution_evidence.decode_receipt_fingerprint,
        "generated_token_identifiers_hash": (
            execution_evidence.generated_output_token_sha256
        ),
        "canonical_float32_score_trace_hash": execution_evidence.token_trace_sha256,
    }
    mismatches = {
        field: {"expected": value, "observed": getattr(receipt, field)}
        for field, value in expected.items()
        if getattr(receipt, field) != value
    }
    if mismatches:
        raise DataContractError(
            "decode result differs from its execution envelope",
            code="analysis.parse_score_receipt.decode_association",
            context={"mismatches": mismatches},
        )
    non_padding_trace = [row for row in decode_result.token_trace if not row.is_pad]
    if decode_result.generated_token_ids != [row.token_id for row in non_padding_trace]:
        _fail(
            "generated token identifiers differ from non-padding token trace",
            "token_trace_identifiers",
        )
    if [row.step_index for row in decode_result.token_trace] != list(
        range(len(decode_result.token_trace))
    ):
        _fail("token trace step indices are not canonical", "token_trace_steps")
    for row in decode_result.token_trace:
        if (
            row.backend != decode_result.backend
            or row.backend_mode != decode_result.backend_mode
            or row.response_family != decode_result.response_family
        ):
            _fail(
                "token trace runtime identity differs from decode result", "token_trace"
            )
    if decode_result.raw_generated_text != "".join(
        row.token_text for row in non_padding_trace
    ):
        _fail(
            "raw generated output differs from exact non-padding token trace text",
            "raw_output_trace",
        )
    expected_parser_text = _derive_parser_text(decode_result)
    if decode_result.parser_text != expected_parser_text:
        _fail("parser text does not derive from raw generated output", "parser_text")


def _coordinate_frame(
    execution_evidence: ExecutionEvidenceEnvelope,
) -> tuple[CoordinateSource, int, int]:
    mode = spatial_variant_mode_for_arm(execution_evidence.arm)
    if mode is None:
        return (
            "source_canvas",
            execution_evidence.source_width,
            execution_evidence.source_height,
        )
    if (
        execution_evidence.grid_provenance.canonical_spatial_spec_sha256
        != SpatialGridSpec().fingerprint
    ):
        _fail("execution does not use the frozen spatial grid", "spatial_grid")
    cell_index = execution_evidence.canonical_cell_index
    if cell_index is None:
        _fail("spatial execution has no cell index", "spatial_cell")
    plan = SpatialGrid.build(
        source_width=execution_evidence.source_width,
        source_height=execution_evidence.source_height,
    ).plan(cell_index=cell_index, variant_mode=mode)
    return "spatial_local_canvas", plan.output_width, plan.output_height


def _derive_parser_text(decode_result: DecodeResult) -> str:
    if decode_result.strip_policy == "none":
        return decode_result.raw_generated_text
    if decode_result.strip_policy != "terminal_im_end":
        _fail("decode result has an unsupported strip policy", "strip_policy")
    stop_traces = [
        row for row in decode_result.token_trace if row.is_stop and not row.is_pad
    ]
    if len(stop_traces) != 1:
        _fail("terminal strip requires exactly one stop token", "strip_stop")
    stop_text = stop_traces[0].token_text
    if not stop_text or not decode_result.raw_generated_text.endswith(stop_text):
        _fail("terminal stop text is absent from raw output", "strip_stop")
    return decode_result.raw_generated_text[: -len(stop_text)]


def _canonical_trace(trace: TokenTrace) -> TokenTrace:
    logprob = trace.logprob
    if logprob is not None:
        logprob = canonical_float32_logprob(
            logprob,
            error_code="analysis.parse_score_receipt.nonfinite_logprob",
            context={"step_index": trace.step_index, "token_id": trace.token_id},
        )
    return replace(trace, logprob=logprob)


def _token_trace_payload(trace: TokenTrace) -> dict[str, Any]:
    return {
        "backend": trace.backend,
        "backend_mode": trace.backend_mode,
        "is_pad": trace.is_pad,
        "is_stop": trace.is_stop,
        "logprob": trace.logprob,
        "response_family": trace.response_family,
        "step_index": trace.step_index,
        "token_id": trace.token_id,
        "token_text": trace.token_text,
    }


def _score_from_logprobs(values: tuple[float, ...]) -> float:
    for value in values:
        if not math.isfinite(value) or value > 0.0:
            _fail("selected logprobs must be finite natural-log probabilities", "score")
    return math.exp(sum(values) / len(values))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require_sha256(value: object, *, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail("receipt field must be a lowercase SHA-256 digest", "sha256", field=field)


def _require_nonnegative_integer(value: object, *, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail("receipt field must be a nonnegative integer", "integer", field=field)


def _fail(message: str, suffix: str, **context: Any) -> None:
    raise DataContractError(
        message,
        code=f"analysis.parse_score_receipt.{suffix}",
        context=context,
    )


__all__ = [
    "CANONICAL_PARSE_SCORE_RECEIPT_SCHEMA_VERSION",
    "CanonicalParseScoreReceipt",
    "build_canonical_parse_score_receipts",
    "validate_parse_score_receipt_association",
]
