#!/usr/bin/env python3
"""Compare frozen Source@B16 with treatment Source@B16 panel realizations.

``Source@B16`` means Source decoding projected to at most the first sixteen
complete object rows.  This experiment-local reader validates every worker
manifest and declared batch artifact, then reads only each rollout's
``source_b16.projected_parser_evidence`` for owner and parser analysis.  Raw
completion parser output never participates in the comparison.

Annotated physical owners are matched by normalized category and deterministic
one-to-one bounding-box assignment at intersection-over-union thresholds 0.30
and 0.50.  Unmatched predictions remain review-needed evidence; this analyzer
does not label them hallucinations.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from scripts.research.analyze_individual_trajectory_union_support import (  # noqa: E402
    _malformed_before_budget,
    _parsed_rows,
    load_generation7_annotations,
)
from scripts.research.collect_vllm_trajectory_panel import (  # noqa: E402
    SOURCE_B16_ACCEPTED_STATUSES,
    SOURCE_B16_INELIGIBLE_STATUSES,
    SOURCE_B16_ROW_BUDGET,
    SOURCE_B16_STATUSES,
    _sha256_json,
)
from scripts.research.compare_clean_rollout_owner_coverage import (  # noqa: E402
    _global_matches,
    iou_xyxy,
)


SCHEMA_VERSION = "source_b16_treatment_owner_ledger.v1"
PANEL_SCHEMA_VERSION = "coordexp_vllm_trajectory_panel.v2"
WORKER_MANIFEST_SCHEMA_VERSION = "vllm_trajectory_panel_worker_manifest.v1"
INTERSECTION_OVER_UNION_THRESHOLDS = (0.30, 0.50)
EXPECTED_FULL_PANEL_IMAGE_COUNT = 2432
EXPECTED_FULL_PANEL_WORKER_COUNT = 8
EXPECTED_FULL_PANEL_IMAGE_BATCH_SIZE = 16
EXPECTED_MAXIMUM_SEQUENCES = 32
EXPECTED_SOURCE_B16_MAXIMUM_NEW_TOKENS = 2048
EXPECTED_BACKEND = "vllm"
EXPECTED_MODEL_DTYPE = "bf16"
EXPECTED_SAMPLING_ORDER = "request_major"
ALLOWED_SOURCE_B16_REPETITION_PENALTIES = frozenset({1.0, 1.1})
WORKER_DIRECTORY_PATTERN = re.compile(r"worker-(\d+)-of-(\d+)$")
OBJECT_COUNT_BANDS = (
    "one_to_three_annotated_objects",
    "four_to_seven_annotated_objects",
    "eight_to_fifteen_annotated_objects",
    "sixteen_or_more_annotated_objects",
)


class OwnerLedgerError(ValueError):
    """Raised when a panel cannot support an exact owner comparison."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OwnerLedgerError(f"{context} is not an object")
    return value


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OwnerLedgerError(f"invalid JSON object: {path}") from exc
    return dict(_mapping(value, str(path)))


def _canonical_image_id(value: Any, context: str) -> str:
    if isinstance(value, bool) or value is None:
        raise OwnerLedgerError(f"{context} has no usable image_id")
    if isinstance(value, (int, str)) and str(value).strip():
        return str(value).strip()
    raise OwnerLedgerError(f"{context} has no usable image_id")


def _natural_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def _owner_key(owner_id: str) -> tuple[tuple[int, int | str], str]:
    image_id = owner_id.split(":", 1)[0]
    return _natural_key(image_id), owner_id


def _nonnegative_integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OwnerLedgerError(f"{context} must be a nonnegative integer")
    return value


def _positive_integer(value: Any, context: str) -> int:
    parsed = _nonnegative_integer(value, context)
    if parsed == 0:
        raise OwnerLedgerError(f"{context} must be positive")
    return parsed


def _token_ids(value: Any, context: str) -> list[int]:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise OwnerLedgerError(f"{context} must contain integer token IDs")
    return list(value)


def _exact_number(value: Any, expected: float) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and float(value) == expected
    )


def _object_count_band(object_count: int) -> str:
    if 1 <= object_count <= 3:
        return OBJECT_COUNT_BANDS[0]
    if 4 <= object_count <= 7:
        return OBJECT_COUNT_BANDS[1]
    if 8 <= object_count <= 15:
        return OBJECT_COUNT_BANDS[2]
    if object_count >= 16:
        return OBJECT_COUNT_BANDS[3]
    raise OwnerLedgerError("candidate images must contain at least one annotation")


def _load_candidate_cohort(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    rows: dict[str, dict[str, Any]] = {}
    for line_number, raw in enumerate(
        source.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not raw.strip():
            raise OwnerLedgerError(f"blank candidate JSONL row {line_number}: {source}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OwnerLedgerError(
                f"invalid candidate JSONL row {line_number}: {source}"
            ) from exc
        row = dict(_mapping(value, f"candidate JSONL row {line_number}"))
        image_id = _canonical_image_id(
            row.get("image_id"), f"candidate JSONL row {line_number}"
        )
        if image_id in rows:
            raise OwnerLedgerError(f"candidate JSONL duplicates image_id {image_id}")
        width = _positive_integer(row.get("width"), f"candidate image {image_id} width")
        height = _positive_integer(
            row.get("height"), f"candidate image {image_id} height"
        )
        objects = row.get("objects")
        if not isinstance(objects, list) or not objects:
            raise OwnerLedgerError(f"candidate image {image_id} has no annotations")
        rows[image_id] = {
            "image_id": image_id,
            "width": width,
            "height": height,
            "object_count": len(objects),
            "object_count_band": _object_count_band(len(objects)),
        }
    if not rows:
        raise OwnerLedgerError("candidate JSONL is empty")

    annotations = load_generation7_annotations(source)
    for image_id in rows:
        owners = annotations.get(image_id)
        if not owners or len(owners) != rows[image_id]["object_count"]:
            raise OwnerLedgerError(
                f"candidate owner records do not cover image {image_id} exactly"
            )
        rows[image_id]["owners"] = [dict(owner) for owner in owners]
    extra = set(annotations) - set(rows)
    if extra:
        raise OwnerLedgerError(
            f"candidate annotation loader returned extra images: {sorted(extra)[:8]}"
        )
    return {
        "path": source,
        "sha256": _sha256_file(source),
        "rows": rows,
    }


def _validate_panel_config(
    config: Mapping[str, Any],
    *,
    path: Path,
    worker_index: int,
    worker_count: int,
    image_batch_size: int,
    require_full_panel: bool,
) -> dict[str, Any]:
    repetition_penalty = config.get("repetition_penalty")
    validated_repetition_penalty = next(
        (
            allowed
            for allowed in ALLOWED_SOURCE_B16_REPETITION_PENALTIES
            if _exact_number(repetition_penalty, allowed)
        ),
        None,
    )
    if (
        config.get("backend") != EXPECTED_BACKEND
        or config.get("model_dtype") != EXPECTED_MODEL_DTYPE
        or config.get("decode_mode") != "source_b16"
        or config.get("panel_mode") != "source_b16"
        or config.get("source_b16_row_budget") != SOURCE_B16_ROW_BUDGET
        or config.get("sample_count") != 1
        or config.get("sample_index_range") is not None
        or config.get("sampling_is_not_infer_config") is not True
        or config.get("sampling_order") != EXPECTED_SAMPLING_ORDER
        or config.get("request_seed") is not None
        or not _exact_number(config.get("temperature"), 0.0)
        or not _exact_number(config.get("top_p"), 1.0)
        or validated_repetition_penalty is None
    ):
        raise OwnerLedgerError(f"non-canonical Source@B16 config: {path}")
    if (
        config.get("worker_index") != worker_index
        or config.get("worker_count") != worker_count
        or config.get("image_batch_size") != image_batch_size
    ):
        raise OwnerLedgerError(f"worker/config identity mismatch: {path}")
    max_new_tokens = config.get("max_new_tokens")
    _positive_integer(max_new_tokens, f"{path}.config.max_new_tokens")
    if require_full_panel and (
        worker_count != EXPECTED_FULL_PANEL_WORKER_COUNT
        or image_batch_size != EXPECTED_FULL_PANEL_IMAGE_BATCH_SIZE
        or config.get("max_num_seqs") != EXPECTED_MAXIMUM_SEQUENCES
        or max_new_tokens != EXPECTED_SOURCE_B16_MAXIMUM_NEW_TOKENS
    ):
        raise OwnerLedgerError(
            f"Source@B16 artifact changed frozen batch policy: {path}"
        )
    resolved_fingerprint = config.get("resolved_fingerprint")
    if not isinstance(resolved_fingerprint, str) or not resolved_fingerprint:
        raise OwnerLedgerError(f"Source@B16 config lacks resolved fingerprint: {path}")
    infer_config_path = config.get("infer_config_path")
    if not isinstance(infer_config_path, str) or not infer_config_path:
        raise OwnerLedgerError(f"Source@B16 config lacks infer-config path: {path}")
    return {
        "infer_config_path": infer_config_path,
        "resolved_fingerprint": resolved_fingerprint,
        "backend": EXPECTED_BACKEND,
        "model_dtype": EXPECTED_MODEL_DTYPE,
        "decode_mode": "source_b16",
        "panel_mode": "source_b16",
        "source_b16_row_budget": SOURCE_B16_ROW_BUDGET,
        "temperature": 0.0,
        "top_p": 1.0,
        "repetition_penalty": validated_repetition_penalty,
        "sample_count": 1,
        "sample_index_range": None,
        "sampling_is_not_infer_config": True,
        "sampling_order": EXPECTED_SAMPLING_ORDER,
        "request_seed": None,
        "max_new_tokens": max_new_tokens,
        "image_batch_size": image_batch_size,
        "max_num_seqs": config.get("max_num_seqs"),
        "worker_count": worker_count,
    }


def _model_identity_surfaces(
    model: Mapping[str, Any], *, path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    execution = dict(
        _mapping(
            model.get("execution_model_identity"),
            f"{path}.execution_model_identity",
        )
    )
    source = _mapping(execution.get("source_identity"), f"{path}.source_identity")
    base = _mapping(source.get("base"), f"{path}.source_identity.base")
    adapter = _mapping(source.get("adapter"), f"{path}.source_identity.adapter")
    embedding = _mapping(
        source.get("embedding_delta"), f"{path}.source_identity.embedding_delta"
    )
    semantic = _mapping(
        embedding.get("semantic_identity"),
        f"{path}.embedding_delta.semantic_identity",
    )
    tokenizer = dict(
        _mapping(model.get("tokenizer_identity"), f"{path}.tokenizer_identity")
    )
    processor = dict(
        _mapping(model.get("processor_identity"), f"{path}.processor_identity")
    )
    token_ids = semantic.get("token_ids")
    token_strings = semantic.get("token_strings")
    if not isinstance(token_ids, list) or not isinstance(token_strings, list):
        raise OwnerLedgerError(f"model lacks selected-token schema: {path}")
    invariants = {
        "base_model_fingerprint": base.get("fingerprint"),
        "base_model_version": base.get("version"),
        "base_config_sha256": semantic.get("base_config_sha256"),
        "tokenizer_sha256": semantic.get("tokenizer_sha256"),
        "generation_config_fingerprint": model.get("generation_config_fingerprint"),
        "tokenizer_identity": tokenizer,
        "processor_identity": processor,
        "selected_token_schema": {
            key: semantic.get(key)
            for key in (
                "semantics",
                "tensor_dtype",
                "tensor_key",
                "tensor_shape",
                "tie_word_embeddings",
                "token_ids",
                "token_strings",
            )
        },
    }
    if any(
        invariants.get(key) in {None, ""}
        for key in (
            "base_model_fingerprint",
            "base_config_sha256",
            "tokenizer_sha256",
            "generation_config_fingerprint",
        )
    ):
        raise OwnerLedgerError(f"model invariant identity is incomplete: {path}")
    payload = {
        "execution_model_identity_sha256": _sha256_json(execution),
        "source_identity_sha256": _sha256_json(source),
        "composition_key": execution.get("composition_key"),
        "snapshot_fingerprint": execution.get("snapshot_fingerprint"),
        "receipt_fingerprint": execution.get("receipt_fingerprint"),
        "adapter_payload_fingerprint": adapter.get("fingerprint"),
        "selected_token_embedding_payload_fingerprint": embedding.get("fingerprint"),
    }
    if any(value in {None, ""} for value in payload.values()):
        raise OwnerLedgerError(f"model payload identity is incomplete: {path}")
    return execution, invariants, payload


def _validate_prompt_and_image_identity(
    *,
    payload: Mapping[str, Any],
    row: Mapping[str, Any],
    artifact_path: Path,
    candidate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    image_id = _canonical_image_id(row.get("image_id"), f"{artifact_path}.rollout")
    example_id = row.get("example_id")
    if not isinstance(example_id, str) or not example_id:
        raise OwnerLedgerError(f"rollout lacks example_id for image {image_id}")
    metadata_map = _mapping(
        payload.get("prompt_metadata"), f"{artifact_path}.prompt_metadata"
    )
    metadata = _mapping(
        metadata_map.get(example_id),
        f"{artifact_path}.prompt_metadata[{example_id}]",
    )
    row_prompt = _token_ids(
        row.get("prompt_token_ids"), f"{artifact_path}:{image_id}.prompt_token_ids"
    )
    metadata_prompt = _token_ids(
        metadata.get("prompt_token_ids"),
        f"{artifact_path}:{image_id}.metadata.prompt_token_ids",
    )
    row_prompt_hash = row.get("prompt_token_ids_sha256")
    metadata_prompt_hash = metadata.get("prompt_token_ids_sha256")
    if (
        row_prompt != metadata_prompt
        or row_prompt_hash != _sha256_json(row_prompt)
        or metadata_prompt_hash != _sha256_json(metadata_prompt)
        or row_prompt_hash != metadata_prompt_hash
    ):
        raise OwnerLedgerError(f"prompt identity mismatch for image {image_id}")
    row_source_hash = row.get("source_image_file_sha256")
    metadata_source_hash = metadata.get("source_image_file_sha256")
    if (
        not isinstance(row_source_hash, str)
        or not row_source_hash
        or row_source_hash != metadata_source_hash
    ):
        raise OwnerLedgerError(f"source image identity mismatch for image {image_id}")
    executed_rgb_hash = row.get("executed_rgb_sha256")
    if not isinstance(executed_rgb_hash, str) or not executed_rgb_hash:
        raise OwnerLedgerError(
            f"executed image identity is missing for image {image_id}"
        )
    width = _positive_integer(metadata.get("width"), f"metadata width for {image_id}")
    height = _positive_integer(
        metadata.get("height"), f"metadata height for {image_id}"
    )
    if row.get("image_width") != width or row.get("image_height") != height:
        raise OwnerLedgerError(f"image dimension mismatch for image {image_id}")
    if candidate is not None and (
        width != candidate["width"] or height != candidate["height"]
    ):
        raise OwnerLedgerError(
            f"candidate/panel image dimension mismatch for image {image_id}"
        )
    return {
        "example_id": example_id,
        "prompt_token_ids_sha256": str(row_prompt_hash),
        "source_image_file_sha256": row_source_hash,
        "executed_rgb_sha256": executed_rgb_hash,
        "image_width": width,
        "image_height": height,
    }


def _validated_parser_evidence(
    value: Any,
    *,
    context: str,
) -> tuple[dict[str, Any], list[Mapping[str, Any]], int, int]:
    parser = dict(_mapping(value, context))
    parse_status = parser.get("parse_status")
    if parse_status not in {
        "accepted",
        "accepted_with_drops",
        "all_spans_dropped",
        "empty",
        "malformed",
    }:
        raise OwnerLedgerError(f"{context} has an invalid parse status")
    valid_count = _nonnegative_integer(
        parser.get("valid_prediction_count"), f"{context}.valid_prediction_count"
    )
    predictions = parser.get("predictions")
    if not isinstance(predictions, list) or any(
        not isinstance(prediction, Mapping) for prediction in predictions
    ):
        raise OwnerLedgerError(f"{context}.predictions is not an object list")
    if len(predictions) != valid_count:
        raise OwnerLedgerError(f"{context} prediction list/count mismatch")
    dropped_count = _nonnegative_integer(
        parser.get("dropped_prediction_count"),
        f"{context}.dropped_prediction_count",
    )
    dropped = parser.get("dropped_predictions")
    if not isinstance(dropped, list) or any(
        not isinstance(prediction, Mapping) for prediction in dropped
    ):
        raise OwnerLedgerError(f"{context}.dropped_predictions is not an object list")
    if len(dropped) != dropped_count:
        raise OwnerLedgerError(f"{context} dropped-prediction list/count mismatch")
    return parser, predictions, valid_count, dropped_count


def _validate_projection(
    *,
    row: Mapping[str, Any],
    image_id: str,
    artifact_path: Path,
) -> dict[str, Any]:
    if (
        row.get("decode_mode") != "source_b16"
        or row.get("trajectory_id") != "source-b16"
    ):
        raise OwnerLedgerError(
            f"invalid Source@B16 rollout identity for image {image_id}"
        )
    stop_reason = row.get("stop_reason")
    if stop_reason not in {"im_end", "length"}:
        raise OwnerLedgerError(f"invalid Source@B16 stop reason for image {image_id}")
    raw_token_ids = _token_ids(
        row.get("generated_token_ids"),
        f"{artifact_path}:{image_id}.generated_token_ids",
    )
    if row.get("generated_token_ids_sha256") != _sha256_json(raw_token_ids):
        raise OwnerLedgerError(
            f"raw generated-token hash mismatch for image {image_id}"
        )

    receipt = _mapping(row.get("source_b16"), f"{artifact_path}:{image_id}.source_b16")
    status = receipt.get("status")
    if (
        status not in SOURCE_B16_STATUSES
        or receipt.get("row_budget") != SOURCE_B16_ROW_BUDGET
    ):
        raise OwnerLedgerError(
            f"invalid Source@B16 receipt status for image {image_id}"
        )
    projected_count = _nonnegative_integer(
        receipt.get("projected_valid_complete_row_count"),
        f"{artifact_path}:{image_id}.projected row count",
    )
    raw_count = _nonnegative_integer(
        receipt.get("raw_valid_complete_row_count"),
        f"{artifact_path}:{image_id}.raw row count",
    )
    if projected_count > SOURCE_B16_ROW_BUDGET:
        raise OwnerLedgerError(
            f"Source@B16 projection exceeds row budget for image {image_id}"
        )
    projected_token_ids = _token_ids(
        receipt.get("projected_token_ids"),
        f"{artifact_path}:{image_id}.projected_token_ids",
    )
    if receipt.get("projected_token_ids_sha256") != _sha256_json(projected_token_ids):
        raise OwnerLedgerError(f"projected token hash mismatch for image {image_id}")
    if not isinstance(receipt.get("projected_text"), str):
        raise OwnerLedgerError(f"Source@B16 projection lacks text for image {image_id}")
    token_limit_before_budget = receipt.get("token_limit_before_budget")
    natural_end_before_budget = receipt.get("natural_end_before_budget")
    if not isinstance(token_limit_before_budget, bool) or not isinstance(
        natural_end_before_budget, bool
    ):
        raise OwnerLedgerError(
            f"Source@B16 receipt lacks boolean health for image {image_id}"
        )
    expected_token_limit = (
        stop_reason == "length" and projected_count < SOURCE_B16_ROW_BUDGET
    )
    expected_natural_end = (
        stop_reason == "im_end" and projected_count < SOURCE_B16_ROW_BUDGET
    )
    if token_limit_before_budget is not expected_token_limit:
        raise OwnerLedgerError(f"token-limit health mismatch for image {image_id}")
    if natural_end_before_budget is not expected_natural_end:
        raise OwnerLedgerError(f"natural-end health mismatch for image {image_id}")
    if status == "accepted_budget" and projected_count != SOURCE_B16_ROW_BUDGET:
        raise OwnerLedgerError(
            f"accepted budget is not exactly sixteen rows for image {image_id}"
        )
    if status != "accepted_budget" and projected_count >= SOURCE_B16_ROW_BUDGET:
        raise OwnerLedgerError(
            f"pre-budget Source@B16 status has sixteen rows for image {image_id}"
        )
    if status == "failed_token_limit_before_budget" and stop_reason != "length":
        raise OwnerLedgerError(
            f"token-limit status lacks length stop for image {image_id}"
        )
    if status == "accepted_natural_end" and stop_reason != "im_end":
        raise OwnerLedgerError(
            f"accepted natural-end status lacks im_end stop for image {image_id}"
        )
    if raw_count < projected_count:
        raise OwnerLedgerError(
            f"raw parser count is below projected count for image {image_id}"
        )

    projected_context = f"{artifact_path}:{image_id}.projected_parser_evidence"
    parser, projected_predictions, parser_reported_count, projected_dropped_count = (
        _validated_parser_evidence(
            receipt.get("projected_parser_evidence"), context=projected_context
        )
    )
    if parser_reported_count != projected_count:
        raise OwnerLedgerError(f"projected parser count mismatch for image {image_id}")
    parser_row = {
        "image_id": image_id,
        "trajectory_id": "source-b16",
        "decode_mode": "greedy",
        "image_width": row["image_width"],
        "image_height": row["image_height"],
        "predictions": parser,
    }
    parsed_rows, parser_evidence = _parsed_rows(parser_row)
    projected_prediction_ids: list[str] = []
    for fallback_index, prediction in enumerate(projected_predictions):
        generated_index = prediction.get(
            "generated_order", prediction.get("row_index", fallback_index)
        )
        try:
            generated_index = int(generated_index)
        except (TypeError, ValueError):
            generated_index = fallback_index
        projected_prediction_ids.append(
            str(
                prediction.get(
                    "object_span_id",
                    prediction.get(
                        "prediction_id", f"source-b16:row-{generated_index}"
                    ),
                )
            )
        )
    if len(set(projected_prediction_ids)) != len(projected_prediction_ids):
        raise OwnerLedgerError(
            f"duplicate projected prediction ID for image {image_id}"
        )
    parsed_prediction_ids = {str(item["prediction_id"]) for item in parsed_rows}
    if not parsed_prediction_ids.issubset(projected_prediction_ids):
        raise OwnerLedgerError(
            f"matching parser introduced a prediction ID for image {image_id}"
        )
    owner_matching_ineligible_prediction_ids = [
        prediction_id
        for prediction_id in projected_prediction_ids
        if prediction_id not in parsed_prediction_ids
    ]
    if parser_evidence.get("valid_prediction_count") != len(parsed_rows):
        raise OwnerLedgerError(f"matching parser count mismatch for image {image_id}")
    parser_status = str(parser["parse_status"])
    if parser_status not in {"accepted", "empty"} or projected_dropped_count != 0:
        raise OwnerLedgerError(
            f"projected Source@B16 parser is not a clean prefix for image {image_id}"
        )
    raw_context = f"{artifact_path}:{image_id}.raw_parser_evidence"
    raw_parser, raw_predictions, raw_reported_count, raw_dropped_count = (
        _validated_parser_evidence(
            receipt.get("raw_parser_evidence"), context=raw_context
        )
    )
    if raw_reported_count != raw_count:
        raise OwnerLedgerError(f"raw parser count mismatch for image {image_id}")

    failure_value = receipt.get("failure_parser_evidence")
    failure_parser: dict[str, Any] | None = None
    failure_predictions: list[Mapping[str, Any]] = []
    failure_dropped_count = 0
    if failure_value is not None:
        (
            failure_parser,
            failure_predictions,
            _,
            failure_dropped_count,
        ) = _validated_parser_evidence(
            failure_value,
            context=f"{artifact_path}:{image_id}.failure_parser_evidence",
        )
        if status != "failed_invalid_before_budget":
            raise OwnerLedgerError(
                f"non-invalid Source@B16 status has failure evidence for image {image_id}"
            )

    dropped_prediction_count = 0
    malformed_row_count = 0
    failure_parser_status: str | None = None
    if status == "failed_invalid_before_budget":
        health_parser = failure_parser if failure_parser is not None else raw_parser
        health_predictions = (
            failure_predictions if failure_parser is not None else raw_predictions
        )
        dropped_prediction_count = (
            failure_dropped_count if failure_parser is not None else raw_dropped_count
        )
        failure_parser_status = str(health_parser["parse_status"])
        if dropped_prediction_count == 0:
            raise OwnerLedgerError(
                f"invalid-before-budget status lacks failure parser evidence for image {image_id}"
            )
        malformed_row_count = _malformed_before_budget(
            health_parser,
            health_predictions,
            SOURCE_B16_ROW_BUDGET,
        )
        if malformed_row_count == 0:
            raise OwnerLedgerError(
                f"invalid-before-budget status lacks a pre-budget failure for image {image_id}"
            )
    elif status in {"accepted_natural_end", "failed_token_limit_before_budget"}:
        if raw_dropped_count != 0:
            raise OwnerLedgerError(
                f"clean pre-budget status has raw parser drops for image {image_id}"
            )
    if status == "accepted_natural_end" and raw_parser != parser:
        raise OwnerLedgerError(
            f"natural-end projected parser differs from raw parser for image {image_id}"
        )
    eligible = status in SOURCE_B16_ACCEPTED_STATUSES
    return {
        "parsed_rows": parsed_rows,
        "owner_matching_ineligible_prediction_ids": (
            owner_matching_ineligible_prediction_ids
        ),
        "health": {
            "source_b16_status": status,
            "source_b16_eligible": eligible,
            "raw_stop_reason": stop_reason,
            "projected_first_sixteen_complete_row_count": projected_count,
            "prediction_count": projected_count,
            "owner_matching_eligible_prediction_count": len(parsed_rows),
            "owner_matching_ineligible_prediction_count": len(
                owner_matching_ineligible_prediction_ids
            ),
            "parser_status": parser_status,
            "raw_parser_status": str(raw_parser["parse_status"]),
            "raw_dropped_prediction_count": raw_dropped_count,
            "failure_parser_status": failure_parser_status,
            "dropped_prediction_count": dropped_prediction_count,
            "malformed_row_count": malformed_row_count,
            "invalid_before_budget_count": int(
                status == "failed_invalid_before_budget"
            ),
            "token_limit_before_budget_count": int(token_limit_before_budget),
            "natural_end_before_budget_count": int(natural_end_before_budget),
            "truncation_count": int(token_limit_before_budget),
        },
    }


def _status_summary(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(entry["health"]["source_b16_status"]) for entry in entries)
    ineligible = [
        str(entry["image_id"])
        for entry in entries
        if not bool(entry["health"]["source_b16_eligible"])
    ]
    return {
        "status_counts": dict(sorted(counts.items())),
        "accepted_count": sum(
            counts[status] for status in SOURCE_B16_ACCEPTED_STATUSES
        ),
        "ineligible_count": sum(
            counts[status] for status in SOURCE_B16_INELIGIBLE_STATUSES
        ),
        "ineligible_image_ids": ineligible,
    }


def _validate_declared_status_summary(
    value: Any,
    expected: Mapping[str, Any],
    context: str,
) -> None:
    observed = _mapping(value, context)
    try:
        observed_ineligible = [
            _canonical_image_id(item, context)
            for item in observed.get("ineligible_image_ids", [])
        ]
    except TypeError as exc:
        raise OwnerLedgerError(f"{context} has invalid ineligible image IDs") from exc
    comparable = {
        "status_counts": dict(observed.get("status_counts", {})),
        "accepted_count": observed.get("accepted_count"),
        "ineligible_count": observed.get("ineligible_count"),
        "ineligible_image_ids": observed_ineligible,
    }
    if comparable != dict(expected):
        raise OwnerLedgerError(f"{context} disagrees with batch artifacts")


def _validate_generation_health(
    value: Any,
    entries: Sequence[Mapping[str, Any]],
    context: str,
) -> None:
    health = _mapping(value, context)
    stop_counts = Counter(str(entry["health"]["raw_stop_reason"]) for entry in entries)
    expected_stop_counts = {
        "im_end": stop_counts["im_end"],
        "length": stop_counts["length"],
    }
    if (
        health.get("completion_count") != len(entries)
        or health.get("stop_reason_counts") != expected_stop_counts
        or health.get("natural_closure_count") != expected_stop_counts["im_end"]
    ):
        raise OwnerLedgerError(f"{context} disagrees with batch artifacts")


def _load_source_b16_panel(
    panel_root: str | Path,
    *,
    candidate_rows: Mapping[str, Mapping[str, Any]],
    require_full_panel: bool,
) -> dict[str, Any]:
    root = Path(panel_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    worker_directories = sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir() and WORKER_DIRECTORY_PATTERN.fullmatch(path.name)
        ),
        key=lambda path: path.name,
    )
    if not worker_directories:
        raise OwnerLedgerError(f"no Source@B16 worker directories under {root}")

    manifests: list[dict[str, Any]] = []
    images: dict[str, dict[str, Any]] = {}
    artifact_provenance: list[dict[str, Any]] = []
    worker_partition: list[dict[str, Any]] = []
    worker_indices: set[int] = set()
    declared_worker_count: int | None = None
    panel_model_identity: dict[str, Any] | None = None
    panel_model_identity_sha256: str | None = None
    panel_model_invariants: dict[str, Any] | None = None
    panel_model_payload_identity: dict[str, Any] | None = None
    panel_config_identity: dict[str, Any] | None = None

    for worker_directory in worker_directories:
        match = WORKER_DIRECTORY_PATTERN.fullmatch(worker_directory.name)
        assert match is not None
        directory_worker_index = int(match.group(1))
        directory_worker_count = int(match.group(2))
        manifest_path = worker_directory / "manifest.json"
        if not manifest_path.is_file():
            raise OwnerLedgerError(f"worker lacks manifest: {worker_directory}")
        manifest = _read_json_object(manifest_path)
        if manifest.get("schema_version") != WORKER_MANIFEST_SCHEMA_VERSION:
            raise OwnerLedgerError(f"wrong worker manifest schema: {manifest_path}")
        worker_index = _nonnegative_integer(
            manifest.get("worker_index"), f"{manifest_path}.worker_index"
        )
        worker_count = _positive_integer(
            manifest.get("worker_count"), f"{manifest_path}.worker_count"
        )
        if (
            worker_index != directory_worker_index
            or worker_count != directory_worker_count
        ):
            raise OwnerLedgerError(
                f"worker directory/manifest mismatch: {manifest_path}"
            )
        if worker_index in worker_indices:
            raise OwnerLedgerError(
                f"duplicate worker index {worker_index} under {root}"
            )
        worker_indices.add(worker_index)
        if declared_worker_count is None:
            declared_worker_count = worker_count
        elif declared_worker_count != worker_count:
            raise OwnerLedgerError(f"worker-count mismatch under {root}")
        if (
            manifest.get("panel_mode") != "source_b16"
            or manifest.get("decode_modes") != ["source_b16"]
            or manifest.get("source_b16_row_budget") != SOURCE_B16_ROW_BUDGET
        ):
            raise OwnerLedgerError(
                f"non-canonical Source@B16 manifest: {manifest_path}"
            )
        image_batch_size = _positive_integer(
            manifest.get("image_batch_size"), f"{manifest_path}.image_batch_size"
        )
        if require_full_panel and (
            worker_count != EXPECTED_FULL_PANEL_WORKER_COUNT
            or image_batch_size != EXPECTED_FULL_PANEL_IMAGE_BATCH_SIZE
            or manifest.get("max_num_seqs") != EXPECTED_MAXIMUM_SEQUENCES
        ):
            raise OwnerLedgerError(
                f"Source@B16 manifest changed frozen batch policy: {manifest_path}"
            )
        batches = manifest.get("batches")
        if not isinstance(batches, list) or not batches:
            raise OwnerLedgerError(f"worker manifest has no batches: {manifest_path}")
        batch_indices = [
            _nonnegative_integer(
                _mapping(batch, f"{manifest_path}.batch").get("batch_index"),
                f"{manifest_path}.batch_index",
            )
            for batch in batches
        ]
        if batch_indices != list(range(len(batches))):
            raise OwnerLedgerError(
                f"worker batch indices are incomplete: {manifest_path}"
            )

        declared_artifacts: set[Path] = set()
        worker_entries: list[dict[str, Any]] = []
        for batch in batches:
            batch = _mapping(batch, f"{manifest_path}.batch")
            batch_index = int(batch["batch_index"])
            raw_image_ids = batch.get("image_ids")
            if not isinstance(raw_image_ids, list) or not raw_image_ids:
                raise OwnerLedgerError(
                    f"batch lacks image IDs: {manifest_path}/{batch_index}"
                )
            batch_image_ids = [
                _canonical_image_id(item, f"{manifest_path}.batch[{batch_index}]")
                for item in raw_image_ids
            ]
            if len(set(batch_image_ids)) != len(batch_image_ids):
                raise OwnerLedgerError(
                    f"batch duplicates image IDs: {manifest_path}/{batch_index}"
                )
            artifacts = _mapping(
                batch.get("artifacts"),
                f"{manifest_path}.batch[{batch_index}].artifacts",
            )
            artifact_entry = _mapping(
                artifacts.get("source_b16"),
                f"{manifest_path}.batch[{batch_index}].source_b16",
            )
            relative_path = artifact_entry.get("path")
            expected_sha256 = artifact_entry.get("sha256")
            if not isinstance(relative_path, str) or not relative_path:
                raise OwnerLedgerError(
                    f"batch artifact lacks path: {manifest_path}/{batch_index}"
                )
            artifact_path = (worker_directory / relative_path).resolve()
            if not artifact_path.is_relative_to(worker_directory.resolve()):
                raise OwnerLedgerError(
                    f"batch artifact escapes worker directory: {artifact_path}"
                )
            if artifact_path in declared_artifacts:
                raise OwnerLedgerError(
                    f"batch artifact declared twice: {artifact_path}"
                )
            declared_artifacts.add(artifact_path)
            if not artifact_path.is_file() or expected_sha256 != _sha256_file(
                artifact_path
            ):
                raise OwnerLedgerError(f"batch artifact hash mismatch: {artifact_path}")

            payload = _read_json_object(artifact_path)
            if payload.get("schema_version") != PANEL_SCHEMA_VERSION:
                raise OwnerLedgerError(
                    f"wrong Source@B16 panel schema: {artifact_path}"
                )
            config = _mapping(payload.get("config"), f"{artifact_path}.config")
            config_identity = _validate_panel_config(
                config,
                path=artifact_path,
                worker_index=worker_index,
                worker_count=worker_count,
                image_batch_size=image_batch_size,
                require_full_panel=require_full_panel,
            )
            if panel_config_identity is None:
                panel_config_identity = config_identity
            elif panel_config_identity != config_identity:
                raise OwnerLedgerError(
                    f"panel config mismatch across artifacts under {root}"
                )
            model = _mapping(
                payload.get("model_identity"), f"{artifact_path}.model_identity"
            )
            execution_model, model_invariants, model_payload_identity = (
                _model_identity_surfaces(model, path=artifact_path)
            )
            execution_model_sha256 = model_payload_identity[
                "execution_model_identity_sha256"
            ]
            if panel_model_identity is None:
                panel_model_identity = execution_model
                panel_model_identity_sha256 = execution_model_sha256
                panel_model_invariants = model_invariants
                panel_model_payload_identity = model_payload_identity
            elif (
                panel_model_identity != execution_model
                or panel_model_identity_sha256 != execution_model_sha256
                or panel_model_invariants != model_invariants
                or panel_model_payload_identity != model_payload_identity
            ):
                raise OwnerLedgerError(
                    f"model identity mismatch across artifacts under {root}"
                )

            rollouts = payload.get("rollouts")
            if (
                not isinstance(rollouts, list)
                or not rollouts
                or payload.get("rollout_count") != len(rollouts)
            ):
                raise OwnerLedgerError(
                    f"artifact lacks consistent rollout rows: {artifact_path}"
                )
            prompt_metadata = _mapping(
                payload.get("prompt_metadata"), f"{artifact_path}.prompt_metadata"
            )
            artifact_example_ids: set[str] = set()
            artifact_entries: list[dict[str, Any]] = []
            for raw_row in rollouts:
                row = _mapping(raw_row, f"{artifact_path}.rollout")
                image_id = _canonical_image_id(
                    row.get("image_id"), f"{artifact_path}.rollout"
                )
                candidate = candidate_rows.get(image_id)
                if image_id in images:
                    raise OwnerLedgerError(
                        f"duplicate Source@B16 image {image_id} under {root}"
                    )
                identity = _validate_prompt_and_image_identity(
                    payload=payload,
                    row=row,
                    artifact_path=artifact_path,
                    candidate=candidate,
                )
                projection = _validate_projection(
                    row=row,
                    image_id=image_id,
                    artifact_path=artifact_path,
                )
                entry = {
                    "image_id": image_id,
                    "identity": identity,
                    "artifact_path": str(artifact_path),
                    **projection,
                }
                images[image_id] = entry
                artifact_entries.append(entry)
                artifact_example_ids.add(str(identity["example_id"]))
            if artifact_example_ids != {str(key) for key in prompt_metadata}:
                raise OwnerLedgerError(
                    f"prompt metadata does not cover exact artifact rows: {artifact_path}"
                )
            if [
                str(entry["image_id"]) for entry in artifact_entries
            ] != batch_image_ids:
                raise OwnerLedgerError(
                    f"manifest/artifact image order mismatch: {artifact_path}"
                )
            batch_summary = _status_summary(artifact_entries)
            _validate_declared_status_summary(
                batch.get("source_b16"),
                batch_summary,
                f"{manifest_path}.batch[{batch_index}].source_b16",
            )
            generation_health = _mapping(
                batch.get("generation_health"),
                f"{manifest_path}.batch[{batch_index}].generation_health",
            )
            _validate_generation_health(
                generation_health.get("source_b16"),
                artifact_entries,
                f"{manifest_path}.batch[{batch_index}].generation_health.source_b16",
            )
            worker_entries.extend(artifact_entries)
            worker_partition.append(
                {
                    "worker_index": worker_index,
                    "worker_count": worker_count,
                    "batch_index": batch_index,
                    "image_ids": batch_image_ids,
                }
            )
            artifact_provenance.append(
                {
                    "worker_index": worker_index,
                    "batch_index": batch_index,
                    "path": str(artifact_path),
                    "sha256": str(expected_sha256),
                    "image_count": len(artifact_entries),
                }
            )

        actual_artifacts = {
            path.resolve() for path in worker_directory.glob("source_b16-batch-*.json")
        }
        if declared_artifacts != actual_artifacts:
            missing = sorted(
                str(path) for path in declared_artifacts - actual_artifacts
            )
            extra = sorted(str(path) for path in actual_artifacts - declared_artifacts)
            raise OwnerLedgerError(
                f"worker manifest/artifact set mismatch: missing={missing[:4]} extra={extra[:4]}"
            )
        image_count = _positive_integer(
            manifest.get("image_count"), f"{manifest_path}.image_count"
        )
        if (
            image_count != len(worker_entries)
            or manifest.get("completed_image_count") != image_count
        ):
            raise OwnerLedgerError(f"worker image count mismatch: {manifest_path}")
        worker_summary = _status_summary(worker_entries)
        _validate_declared_status_summary(
            manifest.get("source_b16"), worker_summary, f"{manifest_path}.source_b16"
        )
        expected_status = (
            "completed_with_source_b16_ineligible"
            if worker_summary["ineligible_count"]
            else "completed"
        )
        if manifest.get("status") != expected_status:
            raise OwnerLedgerError(f"worker manifest is not complete: {manifest_path}")
        completion_health = _mapping(
            manifest.get("completion_health"), f"{manifest_path}.completion_health"
        )
        _validate_generation_health(
            {
                "completion_count": len(worker_entries),
                "stop_reason_counts": completion_health.get("stop_reason_counts"),
                "natural_closure_count": completion_health.get("natural_closure_count"),
            },
            worker_entries,
            f"{manifest_path}.completion_health",
        )
        manifests.append(
            {
                "path": str(manifest_path),
                "sha256": _sha256_file(manifest_path),
                "worker_index": worker_index,
                "image_count": image_count,
            }
        )

    assert declared_worker_count is not None
    expected_worker_indices = set(range(declared_worker_count))
    if (
        worker_indices != expected_worker_indices
        or len(worker_directories) != declared_worker_count
    ):
        raise OwnerLedgerError(
            f"missing exact worker cohort under {root}: observed={sorted(worker_indices)} "
            f"expected={sorted(expected_worker_indices)}"
        )
    if require_full_panel and len(images) != EXPECTED_FULL_PANEL_IMAGE_COUNT:
        raise OwnerLedgerError(
            f"panel has {len(images)} images, expected {EXPECTED_FULL_PANEL_IMAGE_COUNT}"
        )
    analysis_images = set(candidate_rows)
    if not analysis_images <= set(images):
        missing = sorted(analysis_images - set(images), key=_natural_key)
        raise OwnerLedgerError(
            f"analysis candidate JSONL is not a panel subset: missing={missing[:8]}"
        )
    assert panel_model_identity_sha256 is not None
    assert panel_model_invariants is not None
    assert panel_model_payload_identity is not None
    assert panel_config_identity is not None
    return {
        "root": root,
        "images": images,
        "execution_model_identity_sha256": panel_model_identity_sha256,
        "model_invariant_identity": panel_model_invariants,
        "model_payload_identity": panel_model_payload_identity,
        "config_identity": panel_config_identity,
        "worker_partition": worker_partition,
        "manifests": sorted(manifests, key=lambda item: int(item["worker_index"])),
        "artifacts": sorted(
            artifact_provenance,
            key=lambda item: (int(item["worker_index"]), int(item["batch_index"])),
        ),
    }


def _match_projected_rows(
    parsed_rows: Sequence[Mapping[str, Any]],
    owners: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
    owner_matching_ineligible_prediction_ids: Sequence[str] = (),
) -> dict[str, Any]:
    ground_truth = [
        (str(owner["category"]), tuple(float(value) for value in owner["bbox"]))
        for owner in owners
    ]
    predictions = [
        (str(row["category"]), tuple(float(value) for value in row["bbox"]))
        for row in parsed_rows
    ]
    matches = _global_matches(ground_truth, predictions, threshold)
    matched_owner_indices = {owner_index for owner_index, _, _ in matches}
    matched_prediction_indices = {
        prediction_index for _, prediction_index, _ in matches
    }
    matched_owner_ids = {
        str(owners[owner_index]["owner_id"]) for owner_index in matched_owner_indices
    }
    unmatched_prediction_indices = [
        index
        for index in range(len(predictions))
        if index not in matched_prediction_indices
    ]
    duplicate_prediction_ids: list[str] = []
    for prediction_index in unmatched_prediction_indices:
        prediction_category, prediction_box = predictions[prediction_index]
        owner_candidates = sorted(
            (
                (
                    float(iou_xyxy(owner_box, prediction_box)),
                    str(owners[owner_index]["owner_id"]),
                )
                for owner_index, (owner_category, owner_box) in enumerate(ground_truth)
                if owner_category == prediction_category
                and iou_xyxy(owner_box, prediction_box) >= threshold
            ),
            key=lambda item: (-item[0], item[1]),
        )
        if owner_candidates and owner_candidates[0][1] in matched_owner_ids:
            duplicate_prediction_ids.append(
                str(parsed_rows[prediction_index]["prediction_id"])
            )
    unmatched_prediction_ids = [
        str(parsed_rows[index]["prediction_id"])
        for index in unmatched_prediction_indices
    ]
    review_needed_prediction_ids = [
        *unmatched_prediction_ids,
        *(str(item) for item in owner_matching_ineligible_prediction_ids),
    ]
    return {
        "matched_owner_ids": sorted(matched_owner_ids, key=_owner_key),
        "matched_owner_count": len(matched_owner_ids),
        "matched_prediction_count": len(matched_prediction_indices),
        "unmatched_prediction_ids": review_needed_prediction_ids,
        "unmatched_prediction_count": len(review_needed_prediction_ids),
        "review_needed_prediction_ids": review_needed_prediction_ids,
        "review_needed_prediction_count": len(review_needed_prediction_ids),
        "owner_matching_ineligible_prediction_ids": list(
            owner_matching_ineligible_prediction_ids
        ),
        "owner_matching_ineligible_prediction_count": len(
            owner_matching_ineligible_prediction_ids
        ),
        "duplicate_prediction_ids": duplicate_prediction_ids,
        "duplicate_prediction_count": len(duplicate_prediction_ids),
    }


def _threshold_key(threshold: float) -> str:
    return f"{threshold:.2f}"


def _panel_image_result(
    entry: Mapping[str, Any],
    owners: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    health = dict(entry["health"])
    if not health["source_b16_eligible"]:
        matches: dict[str, Any] | None = None
    else:
        matches = {
            _threshold_key(threshold): _match_projected_rows(
                entry["parsed_rows"],
                owners,
                threshold=threshold,
                owner_matching_ineligible_prediction_ids=entry[
                    "owner_matching_ineligible_prediction_ids"
                ],
            )
            for threshold in INTERSECTION_OVER_UNION_THRESHOLDS
        }
    return {
        "health": health,
        "matching_by_intersection_over_union": matches,
    }


def _per_image_comparison(
    *,
    candidate: Mapping[str, Any],
    source_entry: Mapping[str, Any],
    treatment_entry: Mapping[str, Any],
) -> dict[str, Any]:
    owners = candidate["owners"]
    source = _panel_image_result(source_entry, owners)
    treatment = _panel_image_result(treatment_entry, owners)
    source_eligible = bool(source["health"]["source_b16_eligible"])
    treatment_eligible = bool(treatment["health"]["source_b16_eligible"])
    if not source_eligible:
        exclusion_reason = "source_ineligible"
    elif not treatment_eligible:
        exclusion_reason = "treatment_ineligible"
    else:
        exclusion_reason = None
    by_threshold: dict[str, Any] | None = None
    if exclusion_reason is None:
        by_threshold = {}
        for threshold in INTERSECTION_OVER_UNION_THRESHOLDS:
            key = _threshold_key(threshold)
            source_ids = set(
                source["matching_by_intersection_over_union"][key]["matched_owner_ids"]
            )
            treatment_ids = set(
                treatment["matching_by_intersection_over_union"][key][
                    "matched_owner_ids"
                ]
            )
            retained = sorted(source_ids & treatment_ids, key=_owner_key)
            lost = sorted(source_ids - treatment_ids, key=_owner_key)
            gained = sorted(treatment_ids - source_ids, key=_owner_key)
            by_threshold[key] = {
                "source_owner_ids": sorted(source_ids, key=_owner_key),
                "treatment_owner_ids": sorted(treatment_ids, key=_owner_key),
                "retained_source_owner_ids": retained,
                "lost_source_owner_ids": lost,
                "gained_annotated_owner_ids": gained,
                "retained_source_owner_count": len(retained),
                "lost_source_owner_count": len(lost),
                "gained_annotated_owner_count": len(gained),
                "net_owner_delta": len(treatment_ids) - len(source_ids),
            }
    return {
        "image_id": str(candidate["image_id"]),
        "annotated_object_count": int(candidate["object_count"]),
        "object_count_band": str(candidate["object_count_band"]),
        "source": source,
        "treatment": treatment,
        "owner_comparison": {
            "eligible": exclusion_reason is None,
            "exclusion_reason": exclusion_reason,
            "by_intersection_over_union": by_threshold,
        },
    }


def _panel_health_summary(
    records: Sequence[Mapping[str, Any]],
    arm: str,
) -> dict[str, Any]:
    health_rows = [record[arm]["health"] for record in records]
    status_counts = Counter(str(row["source_b16_status"]) for row in health_rows)
    stop_counts = Counter(str(row["raw_stop_reason"]) for row in health_rows)
    return {
        "image_count": len(health_rows),
        "source_b16_status_counts": dict(sorted(status_counts.items())),
        "eligible_image_count": sum(
            bool(row["source_b16_eligible"]) for row in health_rows
        ),
        "invalid_before_budget_count": sum(
            int(row["invalid_before_budget_count"]) for row in health_rows
        ),
        "token_limit_before_budget_count": sum(
            int(row["token_limit_before_budget_count"]) for row in health_rows
        ),
        "natural_end_before_budget_count": sum(
            int(row["natural_end_before_budget_count"]) for row in health_rows
        ),
        "raw_stop_reason_counts": dict(sorted(stop_counts.items())),
        "prediction_count": sum(int(row["prediction_count"]) for row in health_rows),
        "owner_matching_eligible_prediction_count": sum(
            int(row["owner_matching_eligible_prediction_count"]) for row in health_rows
        ),
        "owner_matching_ineligible_prediction_count": sum(
            int(row["owner_matching_ineligible_prediction_count"])
            for row in health_rows
        ),
        "dropped_prediction_count": sum(
            int(row["dropped_prediction_count"]) for row in health_rows
        ),
        "malformed_row_count": sum(
            int(row["malformed_row_count"]) for row in health_rows
        ),
        "truncation_count": sum(int(row["truncation_count"]) for row in health_rows),
    }


def _comparison_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [record for record in records if record["owner_comparison"]["eligible"]]
    source_ineligible = [
        record
        for record in records
        if not record["source"]["health"]["source_b16_eligible"]
    ]
    treatment_ineligible = [
        record
        for record in records
        if not record["treatment"]["health"]["source_b16_eligible"]
    ]
    owner_summaries: dict[str, Any] = {}
    compared_prediction_health: dict[str, Any] = {}
    for threshold in INTERSECTION_OVER_UNION_THRESHOLDS:
        key = _threshold_key(threshold)
        source_owner_ids: set[str] = set()
        treatment_owner_ids: set[str] = set()
        retained_owner_ids: set[str] = set()
        lost_owner_ids: set[str] = set()
        gained_owner_ids: set[str] = set()
        for record in eligible:
            comparison = record["owner_comparison"]["by_intersection_over_union"][key]
            source_owner_ids.update(comparison["source_owner_ids"])
            treatment_owner_ids.update(comparison["treatment_owner_ids"])
            retained_owner_ids.update(comparison["retained_source_owner_ids"])
            lost_owner_ids.update(comparison["lost_source_owner_ids"])
            gained_owner_ids.update(comparison["gained_annotated_owner_ids"])
        owner_summaries[key] = {
            "owner_denominator_image_count": len(eligible),
            "source_owner_ids": sorted(source_owner_ids, key=_owner_key),
            "treatment_owner_ids": sorted(treatment_owner_ids, key=_owner_key),
            "retained_source_owner_ids": sorted(retained_owner_ids, key=_owner_key),
            "lost_source_owner_ids": sorted(lost_owner_ids, key=_owner_key),
            "gained_annotated_owner_ids": sorted(gained_owner_ids, key=_owner_key),
            "source_owner_count": len(source_owner_ids),
            "treatment_owner_count": len(treatment_owner_ids),
            "retained_source_owner_count": len(retained_owner_ids),
            "lost_source_owner_count": len(lost_owner_ids),
            "gained_annotated_owner_count": len(gained_owner_ids),
            "net_owner_delta": len(treatment_owner_ids) - len(source_owner_ids),
        }
        by_arm: dict[str, Any] = {}
        for arm in ("source", "treatment"):
            match_rows = [
                record[arm]["matching_by_intersection_over_union"][key]
                for record in eligible
            ]
            by_arm[arm] = {
                "scope": "joined_source_and_treatment_eligible_images",
                "prediction_count": sum(
                    int(record[arm]["health"]["prediction_count"])
                    for record in eligible
                ),
                "owner_matching_eligible_prediction_count": sum(
                    int(
                        record[arm]["health"][
                            "owner_matching_eligible_prediction_count"
                        ]
                    )
                    for record in eligible
                ),
                "owner_matching_ineligible_prediction_count": sum(
                    int(
                        record[arm]["health"][
                            "owner_matching_ineligible_prediction_count"
                        ]
                    )
                    for record in eligible
                ),
                "matched_prediction_count": sum(
                    int(row["matched_prediction_count"]) for row in match_rows
                ),
                "unmatched_prediction_count": sum(
                    int(row["unmatched_prediction_count"]) for row in match_rows
                ),
                "review_needed_prediction_count": sum(
                    int(row["review_needed_prediction_count"]) for row in match_rows
                ),
                "duplicate_prediction_count": sum(
                    int(row["duplicate_prediction_count"]) for row in match_rows
                ),
                "invalid_before_budget_count": sum(
                    int(record[arm]["health"]["invalid_before_budget_count"])
                    for record in eligible
                ),
                "dropped_prediction_count": sum(
                    int(record[arm]["health"]["dropped_prediction_count"])
                    for record in eligible
                ),
                "malformed_row_count": sum(
                    int(record[arm]["health"]["malformed_row_count"])
                    for record in eligible
                ),
                "truncation_count": sum(
                    int(record[arm]["health"]["truncation_count"])
                    for record in eligible
                ),
            }
        compared_prediction_health[key] = by_arm
    return {
        "candidate_image_count": len(records),
        "source_eligible_image_count": sum(
            bool(record["source"]["health"]["source_b16_eligible"])
            for record in records
        ),
        "treatment_eligible_image_count": sum(
            bool(record["treatment"]["health"]["source_b16_eligible"])
            for record in records
        ),
        "joined_owner_comparison_eligible_image_count": len(eligible),
        "source_ineligible_image_count": len(source_ineligible),
        "source_ineligible_image_ids": [
            record["image_id"] for record in source_ineligible
        ],
        "treatment_ineligible_image_count": len(treatment_ineligible),
        "treatment_ineligible_image_ids": [
            record["image_id"] for record in treatment_ineligible
        ],
        "source_ineligible_excluded_from_owner_denominator_count": len(
            source_ineligible
        ),
        "source_panel_health": _panel_health_summary(records, "source"),
        "treatment_panel_health": _panel_health_summary(records, "treatment"),
        "owner_comparison_by_intersection_over_union": owner_summaries,
        "compared_prediction_health_by_intersection_over_union": compared_prediction_health,
    }


def _panel_provenance(panel: Mapping[str, Any]) -> dict[str, Any]:
    ordered_image_ids = [
        image_id
        for batch in panel["worker_partition"]
        for image_id in batch["image_ids"]
    ]
    return {
        "root": str(panel["root"]),
        "full_panel_image_count": len(panel["images"]),
        "worker_manifest_count": len(panel["manifests"]),
        "batch_artifact_count": len(panel["artifacts"]),
        "execution_model_identity_sha256": panel["execution_model_identity_sha256"],
        "model_payload_identity": dict(panel["model_payload_identity"]),
        "model_invariant_identity": dict(panel["model_invariant_identity"]),
        "model_invariant_identity_sha256": _sha256_json(
            panel["model_invariant_identity"]
        ),
        "config_identity": dict(panel["config_identity"]),
        "worker_partition_sha256": _sha256_json(panel["worker_partition"]),
        "ordered_image_ids_sha256": _sha256_json(ordered_image_ids),
        "worker_manifests": list(panel["manifests"]),
        "batch_artifacts": list(panel["artifacts"]),
    }


def analyze_source_b16_treatments(
    *,
    candidate_jsonl: str | Path,
    source_panel_root: str | Path,
    treatment_panel_roots: Mapping[str, str | Path],
    require_full_panel: bool = True,
) -> dict[str, Any]:
    """Validate panel roots and return paired owner-ledger comparisons."""

    if not treatment_panel_roots:
        raise OwnerLedgerError("at least one treatment panel is required")
    if any(not str(name).strip() for name in treatment_panel_roots):
        raise OwnerLedgerError("treatment names must be non-empty")
    cohort = _load_candidate_cohort(candidate_jsonl)
    candidate_rows = cohort["rows"]
    source_panel = _load_source_b16_panel(
        source_panel_root,
        candidate_rows=candidate_rows,
        require_full_panel=require_full_panel,
    )
    treatments: dict[str, Any] = {}
    for name, root in treatment_panel_roots.items():
        treatment_panel = _load_source_b16_panel(
            root,
            candidate_rows=candidate_rows,
            require_full_panel=require_full_panel,
        )
        if set(treatment_panel["images"]) != set(source_panel["images"]):
            raise OwnerLedgerError(
                f"Source/treatment full-panel cohort mismatch for treatment {name}"
            )
        if treatment_panel["worker_partition"] != source_panel["worker_partition"]:
            raise OwnerLedgerError(
                f"Source/treatment worker partition mismatch for treatment {name}"
            )
        source_contract = {
            key: value
            for key, value in source_panel["config_identity"].items()
            if key not in {"infer_config_path", "resolved_fingerprint"}
        }
        treatment_contract = {
            key: value
            for key, value in treatment_panel["config_identity"].items()
            if key not in {"infer_config_path", "resolved_fingerprint"}
        }
        if source_contract != treatment_contract:
            raise OwnerLedgerError(
                f"Source/treatment Source@B16 execution contract mismatch for treatment {name}"
            )
        if (
            treatment_panel["model_invariant_identity"]
            != source_panel["model_invariant_identity"]
        ):
            raise OwnerLedgerError(
                f"Source/treatment model invariant mismatch for treatment {name}"
            )
        for image_id in sorted(source_panel["images"], key=_natural_key):
            source_identity = source_panel["images"][image_id]["identity"]
            treatment_identity = treatment_panel["images"][image_id]["identity"]
            if source_identity != treatment_identity:
                differing = sorted(
                    key
                    for key in set(source_identity) | set(treatment_identity)
                    if source_identity.get(key) != treatment_identity.get(key)
                )
                raise OwnerLedgerError(
                    f"Source/treatment image or prompt mismatch for image {image_id}: "
                    f"{', '.join(differing)}"
                )
        records = [
            _per_image_comparison(
                candidate=candidate_rows[image_id],
                source_entry=source_panel["images"][image_id],
                treatment_entry=treatment_panel["images"][image_id],
            )
            for image_id in sorted(candidate_rows, key=_natural_key)
        ]
        band_summaries = {
            band: _comparison_summary(
                [record for record in records if record["object_count_band"] == band]
            )
            for band in OBJECT_COUNT_BANDS
        }
        treatments[str(name)] = {
            "panel": _panel_provenance(treatment_panel),
            "aggregate": _comparison_summary(records),
            "object_count_band_summaries": band_summaries,
            "per_image": records,
        }
    analyzer_path = Path(__file__).resolve()
    return {
        "schema_version": SCHEMA_VERSION,
        "policy": {
            "source_baseline": (
                "Source decoding projected to the first sixteen complete object rows or "
                "an earlier natural image-end"
            ),
            "owner_matching_uses_projected_parser_evidence_only": True,
            "invalid_health_parser_evidence": (
                "failure_parser_evidence when present, otherwise raw_parser_evidence; "
                "neither surface contributes owner matches"
            ),
            "intersection_over_union_thresholds": list(
                INTERSECTION_OVER_UNION_THRESHOLDS
            ),
            "owner_matching": (
                "normalized-category constrained, deterministic one-to-one, "
                "maximum-cardinality then maximum-intersection-over-union bounding-box matching"
            ),
            "source_ineligible_images_are_excluded_not_empty": True,
            "treatment_ineligible_images_are_excluded_not_losses": True,
            "unmatched_predictions_are_review_needed_not_hallucinations": True,
            "model_identity_gate": (
                "each panel root must contain one exact payload identity; Source and treatment "
                "adapter and selected-token embedding payloads may differ, while base/config, "
                "tokenizer, processor, selected-token schema, prompt, image, and batch-partition "
                "invariants must agree"
            ),
            "candidate_jsonl_scope": (
                "analysis filter only; every panel remains the frozen full input realization"
            ),
        },
        "inputs": {
            "candidate_jsonl": {
                "path": str(cohort["path"]),
                "sha256": cohort["sha256"],
                "image_count": len(candidate_rows),
            },
            "source_panel": _panel_provenance(source_panel),
            "analyzer_path": str(analyzer_path),
            "analyzer_sha256": _sha256_file(analyzer_path),
        },
        "treatments": treatments,
    }


def _treatment_argument(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("treatment must be NAME=PATH")
    return name.strip(), Path(raw_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument("--source-panel-root", type=Path, required=True)
    parser.add_argument(
        "--treatment",
        type=_treatment_argument,
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Treatment label and Source@B16-style panel root; repeat for multiple panels.",
    )
    parser.add_argument(
        "--allow-bounded-panel",
        action="store_true",
        help=(
            "Permit a deliberately bounded panel instead of requiring the full 2,432-image "
            "production contract. Panel identity and Source-treatment equivalence remain "
            "validated."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    treatments: dict[str, Path] = {}
    for name, path in args.treatment:
        if name in treatments:
            raise OwnerLedgerError(f"duplicate treatment name: {name}")
        treatments[name] = path
    result = analyze_source_b16_treatments(
        candidate_jsonl=args.candidate_jsonl,
        source_panel_root=args.source_panel_root,
        treatment_panel_roots=treatments,
        require_full_panel=not args.allow_bounded_panel,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            _jsonable(result),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
