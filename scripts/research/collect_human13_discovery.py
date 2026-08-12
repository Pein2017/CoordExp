#!/usr/bin/env python3
"""Acquire exact Human-13 Source/K discovery trajectories.

The default path is plan-only.  Explicit Source and K execution are separate,
publish one fresh terminal artifact root atomically, and never resume, retry,
or overwrite.  The loader at the bottom is deliberately runtime-free: it turns
two completed artifact roots into the strict ``ImageInput`` values consumed by
the frozen manifest builder.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Literal


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from scripts.research.build_human13_k_union_manifest import (  # noqa: E402
    EXPECTED_IMAGE_IDENTITIES,
    EXPECTED_K_SEEDS,
    PANEL_PATH,
    ImageInput,
    PredictionRowInput,
    RequestIdentity,
    TrajectoryInput,
    default_binding,
    load_frozen_panel,
)
from scripts.research.collect_human13_k16_vllm import (  # noqa: E402
    PlannedBatch,
    plan_panel_requests,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONFIG_PATH = (
    REPO_ROOT / "configs/coordexp_swift/infer/"
    "qwen3_vl_2b_human13_discovery_source_hf_batch1.yaml"
)
K_CONFIG_PATH = (
    REPO_ROOT / "configs/coordexp_swift/infer/"
    "qwen3_vl_2b_human13_discovery_k16_vllm_batch4.yaml"
)
COLLECTION_SCHEMA_VERSION = "human13_discovery_collection.v1"
RECORD_SCHEMA_VERSION = "human13_discovery_trajectory.v1"
RECEIPT_NAME = "receipt.json"
RECORDS_NAME = "trajectories.jsonl"
_PARSER_ID = "compact-object-box-closed-v1"
_PARSER_POLICY = "compact_object_box_closed_only"
_ALLOWED_PARSE_STATUSES = {
    "accepted",
    "accepted_with_drops",
    "empty",
    "all_spans_dropped",
}
_DIGEST_LENGTH = 64
_SOURCE_POLICY = {
    "n": 1,
    "temperature": 0.0,
    "top_p": 1.0,
    "repetition_penalty": 1.0,
    "max_new_tokens": 3084,
}
_K_POLICY = {
    "n": 1,
    "temperature": 0.4,
    "top_p": 0.95,
    "repetition_penalty": 1.10,
    "max_new_tokens": 512,
}


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_normalized(value: object) -> object:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _field(value: Mapping[str, object] | object, name: str) -> object:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _require_digest(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _DIGEST_LENGTH
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_keys(
    value: Mapping[str, object], expected: set[str], *, field: str
) -> None:
    if set(value) != expected:
        raise ValueError(f"{field} keys do not match the discovery schema")


def _prompt_policy_fingerprint(resolved: Any) -> str:
    from src.config.fingerprint import sha256_json
    from src.inference.prompt import TEMPLATE_ID

    return sha256_json(
        {
            "template": resolved.config.template.model_dump(mode="json"),
            "template_id": TEMPLATE_ID,
        }
    )


def validate_exact_config(mode: Literal["source", "k"], resolved: Any) -> None:
    """Fail before runtime when a config differs from the frozen discovery identity."""

    binding = default_binding()
    config = resolved.config
    common = {
        "base_model": config.model.base_model,
        "dtype": config.model.dtype,
        "input_jsonl": config.data.input_jsonl,
        "assistant_format": config.template.assistant_format,
        "object_ordering": config.template.object_ordering,
        "adapter_path": None if config.adapter is None else config.adapter.path,
        "embedding_delta_path": (
            None if config.embedding_delta is None else config.embedding_delta.path
        ),
        "artifact_root": config.run.artifact_root,
        "collision_policy": config.run.collision_policy,
        "prompt_policy_fingerprint": _prompt_policy_fingerprint(resolved),
        "scoring": config.scoring.enabled,
        "token_trace": config.artifacts.write_token_trace,
        "parse_diagnostics": config.artifacts.write_parse_diagnostics,
    }
    expected_common = {
        "base_model": binding.source.base_model_path,
        "dtype": "fp32",
        "input_jsonl": str(Path(PANEL_PATH).resolve()),
        "assistant_format": "object_box_closed",
        "object_ordering": "geo_sorted_xy",
        "adapter_path": str(Path(binding.source.checkpoint_path) / "adapter"),
        "embedding_delta_path": str(
            Path(binding.source.checkpoint_path) / "special_token_embeddings"
        ),
        "artifact_root": binding.artifact_root,
        "collision_policy": "fail",
        "prompt_policy_fingerprint": binding.surface.prompt_policy_fingerprint,
        "scoring": True,
        "token_trace": True,
        "parse_diagnostics": True,
    }
    if common != expected_common:
        raise ValueError("discovery config differs from the frozen Source identity")
    generation = config.generation
    if mode == "source":
        observed = (
            config.backend.type,
            generation.batch_size,
            generation.n,
            generation.temperature,
            generation.top_p,
            generation.repetition_penalty,
            generation.max_new_tokens,
            config.debug.smoke,
            config.debug.dry_run,
        )
        expected = ("hf", 1, 1, 0.0, 1.0, 1.0, 3084, True, False)
        if observed != expected:
            raise ValueError("Source discovery config does not match exact HF batch1")
        return
    observed = (
        config.backend.type,
        generation.batch_size,
        generation.n,
        generation.temperature,
        generation.top_p,
        generation.repetition_penalty,
        generation.max_new_tokens,
        config.debug.dry_run,
    )
    # The generic vLLM config remains a deterministic session scaffold.  The
    # experiment-local explicit requests below own temperature/top-p and seeds.
    expected = ("vllm", 4, 1, 0.0, 1.0, 1.10, 512, False)
    if observed != expected:
        raise ValueError("K discovery config does not match exact vLLM batch4 scaffold")


def load_exact_config(mode: Literal["source", "k"], path: str | Path) -> Any:
    from src.config.inference import load_infer_config

    resolved = load_infer_config(Path(path).resolve(strict=True))
    validate_exact_config(mode, resolved)
    return resolved


def dry_run_plan(
    *,
    source_config_path: str | Path = SOURCE_CONFIG_PATH,
    k_config_path: str | Path = K_CONFIG_PATH,
    panel_path: str | Path = PANEL_PATH,
) -> dict[str, object]:
    """Return the exact plan without importing a model runtime or writing artifacts."""

    load_exact_config("source", source_config_path)
    load_exact_config("k", k_config_path)
    frozen = load_frozen_panel(panel_path)
    if tuple(row.image_id for row in frozen) != tuple(
        image_id for image_id, _ in EXPECTED_IMAGE_IDENTITIES
    ):
        raise ValueError("frozen panel image order changed")
    return {
        "schema_version": COLLECTION_SCHEMA_VERSION,
        "status": "plan_only",
        "binding": _json_normalized(asdict(default_binding())),
        "source_config": str(Path(source_config_path).resolve()),
        "k_config": str(Path(k_config_path).resolve()),
        "source": {
            "backend": "hf",
            "image_count": 13,
            "physical_batch_count": 13,
            "physical_batch_size": 1,
            "request_count": 13,
            **_SOURCE_POLICY,
        },
        "k": {
            "backend": "vllm",
            "image_count": 13,
            "physical_batch_count": 52,
            "physical_batch_size": 4,
            "request_count": 208,
            "n": 1,
            "seeds": list(EXPECTED_K_SEEDS),
            **{key: value for key, value in _K_POLICY.items() if key != "n"},
        },
        "actions": {
            "model_imports": 0,
            "model_loads": 0,
            "engine_opens": 0,
            "gpu_allocations": 0,
            "artifact_writes": 0,
        },
    }


def _normalized_trace(
    token_trace: Sequence[Mapping[str, object] | object],
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for expected_index, item in enumerate(token_trace):
        step_index = _field(item, "step_index")
        token_id = _field(item, "token_id")
        token_text = _field(item, "token_text")
        is_stop = _field(item, "is_stop")
        is_pad = _field(item, "is_pad")
        if step_index != expected_index:
            raise ValueError("token trace step indices are not contiguous")
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise ValueError("token trace contains an invalid token id")
        if not isinstance(token_text, str) or not token_text:
            raise ValueError("token trace contains empty token text")
        if not isinstance(is_stop, bool) or not isinstance(is_pad, bool):
            raise ValueError("token trace stop/pad flags must be boolean")
        rows.append(
            {
                "step_index": step_index,
                "token_id": token_id,
                "token_text": token_text,
                "is_stop": is_stop,
                "is_pad": is_pad,
            }
        )
    if not rows:
        raise ValueError("token trace is empty")
    return tuple(rows)


def _trace_character_boundaries(
    *,
    parser_text: str,
    token_trace: Sequence[Mapping[str, object] | object],
) -> tuple[tuple[dict[str, object], ...], dict[int, int]]:
    trace = _normalized_trace(token_trace)
    non_pad = tuple(row for row in trace if not bool(row["is_pad"]))
    stop_indices = [index for index, row in enumerate(non_pad) if row["is_stop"]]
    if stop_indices and stop_indices != [len(non_pad) - 1]:
        raise ValueError("token trace has a non-terminal stop token")
    body = non_pad[:-1] if stop_indices else non_pad
    if "".join(str(row["token_text"]) for row in body) != parser_text:
        raise ValueError("token trace text does not exactly reconstruct parser text")
    boundaries = {0: 0}
    cursor = 0
    for token_index, row in enumerate(body):
        cursor += len(str(row["token_text"]))
        if cursor in boundaries:
            raise ValueError("token trace has ambiguous character boundaries")
        boundaries[cursor] = token_index + 1
    return non_pad, boundaries


def _text_occurrences(text: str, needle: str) -> tuple[int, ...]:
    starts: list[int] = []
    offset = 0
    while True:
        found = text.find(needle, offset)
        if found < 0:
            break
        starts.append(found)
        offset = found + 1
    return tuple(starts)


def align_char_span_to_token_span(
    *,
    parser_text: str,
    token_trace: Sequence[Mapping[str, object] | object],
    span_text: str,
    char_start: int | None,
    char_end: int | None,
) -> tuple[int, int]:
    """Map one parser span to exact generated-token boundaries.

    Absolute parser positions are authoritative.  A text-only fallback is
    accepted only when the text occurs exactly once, so repeated rows cannot be
    assigned by an arbitrary ``str.find`` result.
    """

    _, boundaries = _trace_character_boundaries(
        parser_text=parser_text,
        token_trace=token_trace,
    )
    if not span_text:
        raise ValueError("alignment span text must be non-empty")
    if char_start is None or char_end is None:
        occurrences = _text_occurrences(parser_text, span_text)
        if len(occurrences) != 1:
            raise ValueError("text-only token-span alignment is ambiguous or missing")
        char_start = occurrences[0]
        char_end = char_start + len(span_text)
    if (
        isinstance(char_start, bool)
        or not isinstance(char_start, int)
        or isinstance(char_end, bool)
        or not isinstance(char_end, int)
        or not 0 <= char_start < char_end <= len(parser_text)
    ):
        raise ValueError("parser character span is invalid")
    if parser_text[char_start:char_end] != span_text:
        raise ValueError("parser character span text does not match its evidence")
    if char_start not in boundaries or char_end not in boundaries:
        raise ValueError("parser character span does not align to token boundaries")
    return boundaries[char_start], boundaries[char_end]


def _terminal_index(
    *,
    token_ids: tuple[int, ...],
    stop_reason: object,
    trace: tuple[dict[str, object], ...],
) -> int | None:
    non_pad = tuple(row for row in trace if not bool(row["is_pad"]))
    traced_ids = tuple(int(row["token_id"]) for row in non_pad)
    if traced_ids != token_ids:
        raise ValueError("generated token ids differ from non-padding token trace")
    stop_indices = [index for index, row in enumerate(non_pad) if row["is_stop"]]
    if stop_reason == "im_end":
        if stop_indices != [len(token_ids) - 1]:
            raise ValueError("im_end result lacks one exact terminal token")
        return stop_indices[0]
    if stop_reason == "length":
        if stop_indices:
            raise ValueError("length result unexpectedly contains a terminal token")
        return None
    raise ValueError("result stop reason is outside the exact supported contract")


def trajectory_input_from_decode_result(
    *,
    image_id: int,
    trajectory_id: str,
    request: RequestIdentity,
    result: Mapping[str, object] | object,
    image_width: int,
    image_height: int,
) -> tuple[TrajectoryInput, dict[str, object]]:
    """Project one production DecodeResult through parser/token-span evidence."""

    from src.inference.parsing import parse_compact_object_box_closed

    token_ids_value = _field(result, "generated_token_ids")
    if not isinstance(token_ids_value, Sequence) or isinstance(
        token_ids_value, (str, bytes)
    ):
        raise ValueError("result generated token ids are missing")
    token_ids = tuple(int(value) for value in token_ids_value)
    if not token_ids or any(value < 0 for value in token_ids):
        raise ValueError("result generated token ids are invalid")
    parser_text = _field(result, "parser_text")
    if not isinstance(parser_text, str):
        raise ValueError("result parser text is missing")
    trace_value = _field(result, "token_trace")
    if not isinstance(trace_value, Sequence) or isinstance(trace_value, (str, bytes)):
        raise ValueError("result token trace is missing")
    trace = _normalized_trace(trace_value)
    stop_reason = _field(result, "stop_reason")
    terminal = _terminal_index(
        token_ids=token_ids,
        stop_reason=stop_reason,
        trace=trace,
    )
    _trace_character_boundaries(parser_text=parser_text, token_trace=trace)
    parsed = parse_compact_object_box_closed(
        parser_text,
        row_id=trajectory_id,
        row_index=0,
        image_width=image_width,
        image_height=image_height,
    )
    if parsed.parser_id != _PARSER_ID or parsed.parser_policy != _PARSER_POLICY:
        raise ValueError("parser identity differs from the frozen compact parser")
    if parsed.parse_status not in _ALLOWED_PARSE_STATUSES:
        raise ValueError("parser status is outside the strict discovery contract")
    rows: list[PredictionRowInput] = []
    for expected_order, prediction in enumerate(parsed.predictions):
        generated_order = prediction.get("generated_order")
        if generated_order != expected_order:
            raise ValueError("parser complete rows are not in generated order")
        raw_span_text = prediction.get("raw_span_text")
        char_start = prediction.get("char_start")
        char_end = prediction.get("char_end")
        if not isinstance(raw_span_text, str):
            raise ValueError("complete row lacks raw parser span evidence")
        token_start, token_end = align_char_span_to_token_span(
            parser_text=parser_text,
            token_trace=trace,
            span_text=raw_span_text,
            char_start=char_start if isinstance(char_start, int) else None,
            char_end=char_end if isinstance(char_end, int) else None,
        )
        coord_spans = prediction.get("coord_token_spans")
        if not isinstance(coord_spans, list) or len(coord_spans) != 4:
            raise ValueError("complete row lacks four coordinate span records")
        final_coord = coord_spans[-1]
        if not isinstance(final_coord, Mapping):
            raise ValueError("final coordinate span is invalid")
        coordinate_start, coordinate_end = align_char_span_to_token_span(
            parser_text=parser_text,
            token_trace=trace,
            span_text=str(final_coord.get("text", "")),
            char_start=(
                int(final_coord["char_start"])
                if isinstance(final_coord.get("char_start"), int)
                else None
            ),
            char_end=(
                int(final_coord["char_end"])
                if isinstance(final_coord.get("char_end"), int)
                else None
            ),
        )
        if coordinate_end != coordinate_start + 1:
            raise ValueError("final coordinate does not align to exactly one token")
        bbox = prediction.get("bbox")
        category = prediction.get("description")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not isinstance(category, str)
            or not category
        ):
            raise ValueError("complete row category/box evidence is invalid")
        rows.append(
            PredictionRowInput(
                row_id=f"{trajectory_id}:row:{expected_order:03d}",
                row_index=expected_order,
                category=category,
                bbox=tuple(float(value) for value in bbox),
                token_start=token_start,
                token_end=token_end,
                final_coordinate_token_index=coordinate_start,
                parser_status="complete",
                geometry_valid=True,
                row_terminated=True,
            )
        )
    trajectory = TrajectoryInput(
        trajectory_id=trajectory_id,
        request=request,
        token_ids=token_ids,
        terminal_token_index=terminal,
        stop_reason=str(stop_reason),
        parser_status=parsed.parse_status,
        rows=tuple(rows),
    )
    return trajectory, {
        "parser_id": parsed.parser_id,
        "parser_policy": parsed.parser_policy,
        "parse_status": parsed.parse_status,
        "dropped_predictions": parsed.dropped_predictions,
    }


def dispatch_source_batches(
    base_requests: Mapping[int, object],
    execute: Callable[[int, tuple[object, ...]], Sequence[object]],
) -> tuple[object, ...]:
    """Dispatch thirteen successive physical HF batches of exactly one request."""

    expected_ids = tuple(image_id for image_id, _ in EXPECTED_IMAGE_IDENTITIES)
    if tuple(base_requests) != expected_ids:
        raise ValueError("Source base requests do not cover the canonical panel order")
    results: list[object] = []
    for image_id in expected_ids:
        batch = (base_requests[image_id],)
        observed = tuple(execute(image_id, batch))
        if len(observed) != 1:
            raise ValueError("Source execution must return exactly one batch1 result")
        results.extend(observed)
    return tuple(results)


def dispatch_k_batches(
    base_requests: Mapping[int, object],
    execute: Callable[[PlannedBatch, object], Sequence[object]],
) -> tuple[object, ...]:
    """Dispatch 52 successive physical vLLM batches of four explicit requests."""

    expected_ids = tuple(image_id for image_id, _ in EXPECTED_IMAGE_IDENTITIES)
    if tuple(base_requests) != expected_ids:
        raise ValueError("K base requests do not cover the canonical panel order")
    results: list[object] = []
    for batch in plan_panel_requests():
        observed = tuple(execute(batch, base_requests[batch.image_id]))
        if len(observed) != 4:
            raise ValueError("K execution must return four results per physical batch")
        results.extend(observed)
    if len(results) != 208:
        raise ValueError("K execution did not return all 208 explicit results")
    return tuple(results)


def _atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing stale temporary artifact: {temporary}")
    temporary.write_bytes(_canonical_bytes(dict(value)))
    os.replace(temporary, path)


def execute_atomically(
    *,
    output_root: str | Path,
    mode: Literal["source", "k"],
    operation: Callable[[Path], object],
) -> object:
    """Run once in a staging root and atomically publish terminal status."""

    target = Path(output_root).expanduser().resolve()
    staging = target.with_name(f".{target.name}.staging")
    if target.exists() or staging.exists():
        raise FileExistsError(
            f"refusing to overwrite or resume discovery root: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    try:
        result = operation(staging)
        receipt_path = staging / RECEIPT_NAME
        if not receipt_path.is_file():
            raise RuntimeError("atomic discovery operation did not publish a receipt")
        receipt = _load_json_object(receipt_path)
        if receipt.get("status") != "completed":
            raise RuntimeError(
                "atomic discovery operation did not reach completed status"
            )
        if target.exists():
            raise FileExistsError(f"refusing to overwrite discovery root: {target}")
        os.replace(staging, target)
        return result
    except BaseException as exc:
        try:
            _atomic_write_json(
                staging / RECEIPT_NAME,
                {
                    "schema_version": COLLECTION_SCHEMA_VERSION,
                    "status": "failed",
                    "mode": mode,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                },
            )
            if not target.exists():
                os.replace(staging, target)
        except BaseException:
            pass
        raise


def _request_from_dict(value: Mapping[str, object]) -> RequestIdentity:
    _require_keys(value, set(RequestIdentity.__dataclass_fields__), field="request")
    request = RequestIdentity(
        backend=str(value["backend"]),
        backend_version=str(value["backend_version"]),
        mode=str(value["mode"]),  # type: ignore[arg-type]
        n=int(value["n"]),
        seed=None if value["seed"] is None else int(value["seed"]),
        physical_batch_index=int(value["physical_batch_index"]),
        temperature=float(value["temperature"]),
        top_p=float(value["top_p"]),
        repetition_penalty=float(value["repetition_penalty"]),
        max_new_tokens=int(value["max_new_tokens"]),
    )
    expected: tuple[object, ...]
    if request.mode == "source_greedy":
        expected = ("hf", 1, None, 0, 0.0, 1.0, 1.0, 3084)
    elif request.mode == "k_sampled" and request.seed in EXPECTED_K_SEEDS:
        assert request.seed is not None
        expected = (
            "vllm",
            1,
            request.seed,
            (request.seed - 21001) // 4,
            0.4,
            0.95,
            1.10,
            512,
        )
    else:
        raise ValueError("trajectory request is outside the frozen recipe")
    observed = (
        request.backend,
        request.n,
        request.seed,
        request.physical_batch_index,
        request.temperature,
        request.top_p,
        request.repetition_penalty,
        request.max_new_tokens,
    )
    if observed != expected or not request.backend_version:
        raise ValueError("trajectory request does not match the frozen recipe")
    return request


def _trajectory_from_dict(value: Mapping[str, object]) -> TrajectoryInput:
    _require_keys(value, set(TrajectoryInput.__dataclass_fields__), field="trajectory")
    request_value = value["request"]
    if not isinstance(request_value, Mapping):
        raise ValueError("trajectory request must be an object")
    rows_value = value["rows"]
    if not isinstance(rows_value, list):
        raise ValueError("trajectory rows must be a list")
    rows: list[PredictionRowInput] = []
    for row_value in rows_value:
        if not isinstance(row_value, Mapping):
            raise ValueError("trajectory row must be an object")
        _require_keys(
            row_value,
            set(PredictionRowInput.__dataclass_fields__),
            field="prediction row",
        )
        bbox = row_value["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("prediction row bbox must contain four values")
        rows.append(
            PredictionRowInput(
                row_id=str(row_value["row_id"]),
                row_index=int(row_value["row_index"]),
                category=str(row_value["category"]),
                bbox=tuple(float(item) for item in bbox),
                token_start=int(row_value["token_start"]),
                token_end=int(row_value["token_end"]),
                final_coordinate_token_index=int(
                    row_value["final_coordinate_token_index"]
                ),
                parser_status=str(row_value["parser_status"]),
                geometry_valid=bool(row_value["geometry_valid"]),
                row_terminated=bool(row_value["row_terminated"]),
            )
        )
    token_ids = value["token_ids"]
    if not isinstance(token_ids, list):
        raise ValueError("trajectory token_ids must be a list")
    terminal = value["terminal_token_index"]
    return TrajectoryInput(
        trajectory_id=str(value["trajectory_id"]),
        request=_request_from_dict(request_value),
        token_ids=tuple(int(item) for item in token_ids),
        terminal_token_index=None if terminal is None else int(terminal),
        stop_reason=str(value["stop_reason"]),
        parser_status=str(value["parser_status"]),
        rows=tuple(rows),
    )


def _validate_record(
    value: Mapping[str, object],
    *,
    expected_mode: Literal["source", "k"],
    frozen_by_id: Mapping[int, object],
) -> TrajectoryInput:
    _require_keys(
        value,
        {
            "schema_version",
            "mode",
            "image_id",
            "panel_row_sha256",
            "image_sha256",
            "prompt_identity",
            "execution_model_identity",
            "session_identity",
            "runtime_counters",
            "token_trace",
            "parse",
            "trajectory",
        },
        field="trajectory record",
    )
    if (
        value["schema_version"] != RECORD_SCHEMA_VERSION
        or value["mode"] != expected_mode
    ):
        raise ValueError("trajectory record schema/mode does not match collection")
    image_id = int(value["image_id"])
    frozen = frozen_by_id.get(image_id)
    if frozen is None:
        raise ValueError("trajectory image is outside the frozen Human-13 panel")
    if value["panel_row_sha256"] != getattr(frozen, "panel_row_sha256"):
        raise ValueError("trajectory panel-row SHA-256 does not match")
    if value["image_sha256"] != getattr(frozen, "image_sha256"):
        raise ValueError("trajectory image-content SHA-256 does not match")
    prompt = value["prompt_identity"]
    if not isinstance(prompt, Mapping):
        raise ValueError("trajectory prompt identity must be an object")
    _require_keys(
        prompt,
        {
            "prompt_policy_fingerprint",
            "prompt_token_ids_sha256",
            "chat_text_sha256",
        },
        field="prompt identity",
    )
    if (
        prompt["prompt_policy_fingerprint"]
        != default_binding().surface.prompt_policy_fingerprint
    ):
        raise ValueError("trajectory prompt policy identity changed")
    _require_digest(prompt["prompt_token_ids_sha256"], field="prompt token identity")
    _require_digest(prompt["chat_text_sha256"], field="chat text identity")
    execution = value["execution_model_identity"]
    if not isinstance(execution, Mapping) or execution.get("source_identity") != (
        _json_normalized(asdict(default_binding().source))
    ):
        raise ValueError("trajectory execution-model Source identity changed")
    session = value["session_identity"]
    if not isinstance(session, Mapping):
        raise ValueError("trajectory session identity must be an object")
    expected_backend = "hf" if expected_mode == "source" else "vllm"
    if session.get("backend") != expected_backend:
        raise ValueError("trajectory session backend differs from mode")
    _require_digest(session.get("session_identity_sha256"), field="session identity")
    runtime = value["runtime_counters"]
    if not isinstance(runtime, Mapping):
        raise ValueError("trajectory runtime counters must be an object")
    expected_batch_size = 1 if expected_mode == "source" else 4
    if runtime.get("physical_batch_size") != expected_batch_size:
        raise ValueError("trajectory physical batch size changed")
    parse = value["parse"]
    if not isinstance(parse, Mapping):
        raise ValueError("trajectory parse evidence must be an object")
    _require_keys(
        parse,
        {"parser_id", "parser_policy", "parse_status", "dropped_predictions"},
        field="parse evidence",
    )
    if (
        parse["parser_id"] != _PARSER_ID
        or parse["parser_policy"] != _PARSER_POLICY
        or parse["parse_status"] not in _ALLOWED_PARSE_STATUSES
    ):
        raise ValueError("trajectory parser status or identity is invalid")
    trajectory_value = value["trajectory"]
    if not isinstance(trajectory_value, Mapping):
        raise ValueError("trajectory payload must be an object")
    trajectory = _trajectory_from_dict(trajectory_value)
    if trajectory.parser_status != parse["parse_status"]:
        raise ValueError("trajectory/parser status differs")
    expected_request_mode = (
        "source_greedy" if expected_mode == "source" else "k_sampled"
    )
    if trajectory.request.mode != expected_request_mode:
        raise ValueError("trajectory request mode differs from collection")
    expected_trajectory_id = (
        f"human13:{image_id}:source"
        if expected_mode == "source"
        else f"human13:{image_id}:k16:{trajectory.request.seed}"
    )
    if trajectory.trajectory_id != expected_trajectory_id:
        raise ValueError("trajectory ID differs from its image/request identity")
    if (
        runtime.get("physical_batch_index") != trajectory.request.physical_batch_index
        or runtime.get("request_id") != trajectory.trajectory_id
    ):
        raise ValueError("trajectory runtime request identity changed")
    if runtime.get("generated_token_count") != len(trajectory.token_ids):
        raise ValueError("trajectory runtime generated-token count changed")
    batch_elapsed = runtime.get("batch_elapsed_seconds")
    if (
        isinstance(batch_elapsed, bool)
        or not isinstance(batch_elapsed, (int, float))
        or batch_elapsed < 0
    ):
        raise ValueError("trajectory runtime elapsed time is invalid")
    trace_value = value["token_trace"]
    if not isinstance(trace_value, list):
        raise ValueError("trajectory token trace must be a list")
    trace = _normalized_trace(trace_value)
    terminal = _terminal_index(
        token_ids=trajectory.token_ids,
        stop_reason=trajectory.stop_reason,
        trace=trace,
    )
    if trajectory.terminal_token_index != terminal:
        raise ValueError("trajectory terminal token index differs from token trace")
    return trajectory


def _load_json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"artifact must contain a JSON object: {path}")
    return value


def _load_collection(
    root: str | Path,
    *,
    expected_mode: Literal["source", "k"],
    frozen_by_id: Mapping[int, object],
) -> tuple[dict[str, object], tuple[tuple[int, TrajectoryInput], ...]]:
    artifact_root = Path(root).expanduser().resolve(strict=True)
    receipt = _load_json_object(artifact_root / RECEIPT_NAME)
    expected_receipt_keys = {
        "schema_version",
        "status",
        "mode",
        "binding",
        "config_identity",
        "session_identity",
        "runtime_counters",
        "records_file",
        "records_sha256",
        "record_count",
    }
    _require_keys(receipt, expected_receipt_keys, field="collection receipt")
    if receipt["schema_version"] != COLLECTION_SCHEMA_VERSION:
        raise ValueError("collection receipt schema differs")
    if receipt["status"] != "completed":
        raise ValueError("collection receipt is not completed")
    if receipt["mode"] != expected_mode:
        raise ValueError("collection receipt mode differs")
    if receipt["binding"] != _json_normalized(asdict(default_binding())):
        raise ValueError("collection binding differs from frozen Human-13 identity")
    config_identity = receipt["config_identity"]
    expected_config_path = (
        SOURCE_CONFIG_PATH if expected_mode == "source" else K_CONFIG_PATH
    )
    if not isinstance(config_identity, Mapping):
        raise ValueError("collection config identity must be an object")
    _require_keys(config_identity, {"path", "fingerprint"}, field="config identity")
    expected_config_fingerprint = load_exact_config(
        expected_mode, expected_config_path
    ).fingerprint
    if (
        config_identity["path"] != str(expected_config_path)
        or config_identity["fingerprint"] != expected_config_fingerprint
    ):
        raise ValueError("collection config identity differs from the exact config")
    session_identity = receipt["session_identity"]
    expected_backend = "hf" if expected_mode == "source" else "vllm"
    if not isinstance(session_identity, Mapping):
        raise ValueError("collection session identity must be an object")
    receipt_session_digest = _require_digest(
        session_identity.get("session_identity_sha256"),
        field="collection session identity",
    )
    if session_identity.get("backend") != expected_backend:
        raise ValueError("collection session identity backend differs from mode")
    runtime_counters = receipt["runtime_counters"]
    expected_count = 13 if expected_mode == "source" else 208
    expected_batches = 13 if expected_mode == "source" else 52
    if not isinstance(runtime_counters, Mapping) or any(
        (
            runtime_counters.get("image_count") != 13,
            runtime_counters.get("physical_batch_count") != expected_batches,
            runtime_counters.get("request_count") != expected_count,
            runtime_counters.get("retry_count") != 0,
            runtime_counters.get("resume_count") != 0,
        )
    ):
        raise ValueError("collection runtime counters differ from the exact execution")
    generated_count = runtime_counters.get("generated_token_count")
    wall_seconds = runtime_counters.get("wall_seconds")
    if (
        isinstance(generated_count, bool)
        or not isinstance(generated_count, int)
        or generated_count < expected_count
        or isinstance(wall_seconds, bool)
        or not isinstance(wall_seconds, (int, float))
        or wall_seconds < 0
    ):
        raise ValueError("collection runtime counters are invalid")
    if receipt["records_file"] != RECORDS_NAME:
        raise ValueError("collection records filename differs")
    records_path = artifact_root / RECORDS_NAME
    payload = records_path.read_bytes()
    if _sha256_bytes(payload) != receipt["records_sha256"]:
        raise ValueError("collection records SHA-256 mismatch")
    records: list[tuple[int, TrajectoryInput]] = []
    for line_number, raw_line in enumerate(payload.splitlines(), start=1):
        if not raw_line.strip():
            continue
        value = json.loads(raw_line)
        if not isinstance(value, Mapping):
            raise ValueError(f"trajectory record line {line_number} is not an object")
        record_session = value.get("session_identity")
        if (
            not isinstance(record_session, Mapping)
            or record_session.get("session_identity_sha256") != receipt_session_digest
        ):
            raise ValueError(
                "trajectory session identity differs from collection receipt"
            )
        image_id = int(value.get("image_id", -1))
        records.append(
            (
                image_id,
                _validate_record(
                    value,
                    expected_mode=expected_mode,
                    frozen_by_id=frozen_by_id,
                ),
            )
        )
    if len(records) != int(receipt["record_count"]):
        raise ValueError("collection receipt/result coverage differs")
    if len(records) != expected_count:
        suffix = (
            "one Source trajectory per image"
            if expected_mode == "source"
            else "sixteen K trajectories per image"
        )
        raise ValueError(f"collection requires {suffix}")
    return receipt, tuple(records)


def artifacts_to_image_inputs(
    *,
    source_root: str | Path,
    k_root: str | Path,
    panel_path: str | Path = PANEL_PATH,
) -> tuple[ImageInput, ...]:
    """Purely load completed Source/K roots into thirteen strict ImageInputs."""

    frozen = load_frozen_panel(panel_path)
    frozen_by_id = {row.image_id: row for row in frozen}
    _, source_records = _load_collection(
        source_root,
        expected_mode="source",
        frozen_by_id=frozen_by_id,
    )
    _, k_records = _load_collection(
        k_root,
        expected_mode="k",
        frozen_by_id=frozen_by_id,
    )
    source_by_image: dict[int, TrajectoryInput] = {}
    for image_id, trajectory in source_records:
        if image_id in source_by_image:
            raise ValueError("each image requires one and only one Source trajectory")
        source_by_image[image_id] = trajectory
    sampled_by_image: dict[int, list[TrajectoryInput]] = {
        row.image_id: [] for row in frozen
    }
    for image_id, trajectory in k_records:
        sampled_by_image.setdefault(image_id, []).append(trajectory)
    images: list[ImageInput] = []
    for row in frozen:
        source = source_by_image.get(row.image_id)
        if source is None:
            raise ValueError("full panel is missing a Source trajectory")
        sampled = sorted(
            sampled_by_image.get(row.image_id, []),
            key=lambda item: (
                -1 if item.request.seed is None else item.request.seed,
                item.trajectory_id,
            ),
        )
        seeds = tuple(item.request.seed for item in sampled)
        if seeds != EXPECTED_K_SEEDS:
            raise ValueError(
                f"image {row.image_id} requires sixteen unique declared K seeds"
            )
        images.append(
            ImageInput(
                image_id=row.image_id,
                owners=row.owners,
                source=source,
                sampled=tuple(sampled),
                panel_row_sha256=row.panel_row_sha256,
                image_sha256=row.image_sha256,
            )
        )
    return tuple(images)


def _session_identity(receipt: Any) -> dict[str, object]:
    payload = receipt.to_artifact_dict()
    identity = {
        "backend": payload.get("backend"),
        "backend_version": payload.get("backend_version"),
        "backend_mode": payload.get("backend_mode"),
        "response_family": payload.get("response_family"),
        "model_identity": payload.get("model_identity"),
        "tokenizer_identity": payload.get("tokenizer_identity"),
        "processor_identity": payload.get("processor_identity"),
        "generation_config_fingerprint": payload.get("generation_config_fingerprint"),
        "execution_model_identity": payload.get("execution_model_identity"),
        "effective_settings": payload.get("effective_settings"),
    }
    identity["session_identity_sha256"] = _sha256_bytes(_canonical_bytes(identity))
    return identity


def _scientific_request(
    *, mode: Literal["source", "k"], seed: int | None, backend_version: str
) -> RequestIdentity:
    if mode == "source":
        return RequestIdentity(
            backend="hf",
            backend_version=backend_version,
            mode="source_greedy",
            seed=None,
            physical_batch_index=0,
            **_SOURCE_POLICY,
        )
    if seed not in EXPECTED_K_SEEDS:
        raise ValueError("K request seed is outside the exact contract")
    assert seed is not None
    return RequestIdentity(
        backend="vllm",
        backend_version=backend_version,
        mode="k_sampled",
        seed=seed,
        physical_batch_index=(seed - 21001) // 4,
        **_K_POLICY,
    )


def _record_from_result(
    *,
    mode: Literal["source", "k"],
    image_id: int,
    frozen: object,
    prompt_metadata: Mapping[str, object],
    execution_model_identity: Mapping[str, object],
    session_identity: Mapping[str, object],
    result: object,
    request: RequestIdentity,
    batch_elapsed_seconds: float,
) -> dict[str, object]:
    trajectory_id = (
        f"human13:{image_id}:source"
        if mode == "source"
        else f"human13:{image_id}:k16:{request.seed}"
    )
    trajectory, parse = trajectory_input_from_decode_result(
        image_id=image_id,
        trajectory_id=trajectory_id,
        request=request,
        result=result,
        image_width=int(prompt_metadata["width"]),
        image_height=int(prompt_metadata["height"]),
    )
    trace = _normalized_trace(_field(result, "token_trace"))  # type: ignore[arg-type]
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "mode": mode,
        "image_id": image_id,
        "panel_row_sha256": getattr(frozen, "panel_row_sha256"),
        "image_sha256": getattr(frozen, "image_sha256"),
        "prompt_identity": {
            "prompt_policy_fingerprint": (
                default_binding().surface.prompt_policy_fingerprint
            ),
            "prompt_token_ids_sha256": prompt_metadata["prompt_token_ids_sha256"],
            "chat_text_sha256": prompt_metadata["chat_text_sha256"],
        },
        "execution_model_identity": dict(execution_model_identity),
        "session_identity": dict(session_identity),
        "runtime_counters": {
            "physical_batch_index": request.physical_batch_index,
            "physical_batch_size": 1 if mode == "source" else 4,
            "request_id": trajectory_id,
            "generated_token_count": len(trajectory.token_ids),
            "batch_elapsed_seconds": float(batch_elapsed_seconds),
        },
        "token_trace": list(trace),
        "parse": parse,
        "trajectory": asdict(trajectory),
    }


def _write_completed_collection(
    *,
    staging: Path,
    mode: Literal["source", "k"],
    records: Sequence[Mapping[str, object]],
    config_identity: Mapping[str, object],
    session_identity: Mapping[str, object],
    runtime_counters: Mapping[str, object],
) -> None:
    payload = b"".join(_canonical_bytes(dict(record)) for record in records)
    (staging / RECORDS_NAME).write_bytes(payload)
    _atomic_write_json(
        staging / RECEIPT_NAME,
        {
            "schema_version": COLLECTION_SCHEMA_VERSION,
            "status": "completed",
            "mode": mode,
            "binding": _json_normalized(asdict(default_binding())),
            "config_identity": dict(config_identity),
            "session_identity": dict(session_identity),
            "runtime_counters": dict(runtime_counters),
            "records_file": RECORDS_NAME,
            "records_sha256": _sha256_bytes(payload),
            "record_count": len(records),
        },
    )


def _vllm_sampling_params(request: Any, *, stop_token_id: int) -> Any:
    from vllm import SamplingParams

    return SamplingParams(
        n=1,
        seed=request.seed,
        temperature=0.4,
        top_p=0.95,
        top_k=0,
        repetition_penalty=1.10,
        max_tokens=512,
        logprobs=0,
        stop_token_ids=[int(stop_token_id)],
        ignore_eos=False,
        detokenize=True,
        skip_special_tokens=False,
        spaces_between_special_tokens=True,
    )


def _execute_runtime(
    *,
    mode: Literal["source", "k"],
    output_root: str | Path,
    config_path: str | Path,
    panel_path: str | Path = PANEL_PATH,
) -> Path:
    """Explicit live path.  The CLI never reaches this without an execute flag."""

    def operation(staging: Path) -> None:
        from src.config.fingerprint import sha256_json
        from src.data import load_raw_examples
        from src.inference.backend import (
            GenerationPolicy,
            cuda_peak_memory_snapshot,
            open_backend_session,
            update_decode_performance_receipt,
        )
        from src.inference.runtime import assemble_frontend
        from src.inference.vllm_backend import (
            _close_prompt_images,
            _restore_native_request_order,
        )
        from scripts.research.run_current_seeded_sampled_rollouts import (
            _build_requests,
            physical_image_id,
        )

        resolved = load_exact_config(mode, config_path)
        frozen = load_frozen_panel(panel_path)
        frozen_by_id = {row.image_id: row for row in frozen}
        examples = list(load_raw_examples(resolved.config.data.input_jsonl))
        image_ids = tuple(int(physical_image_id(example)) for example in examples)
        expected_ids = tuple(image_id for image_id, _ in EXPECTED_IMAGE_IDENTITIES)
        if image_ids != expected_ids:
            raise ValueError("runtime input rows differ from canonical Human-13 order")
        execution_model = None
        if mode == "k":
            from src.inference.pipeline import _resolve_execution_model_for_run

            execution_model = _resolve_execution_model_for_run(resolved)
            if not isinstance(execution_model, Mapping):
                raise RuntimeError(
                    "K execution lacks a validated execution-model receipt"
                )
        frontend = assemble_frontend(
            resolved.config,
            generation_config_fingerprint=sha256_json(
                resolved.config.generation.model_dump(mode="json")
            ),
            execution_model=execution_model,
        )
        if frontend.qwen.tokenizer_sha256 != default_binding().surface.tokenizer_sha256:
            raise ValueError("runtime tokenizer SHA-256 differs from frozen identity")
        requests, prompt_by_example = _build_requests(
            resolved.config, frontend, examples
        )
        base_by_image: dict[int, object] = {}
        prompt_by_image: dict[int, Mapping[str, object]] = {}
        for image_id, example, request in zip(
            image_ids, examples, requests, strict=True
        ):
            policy = GenerationPolicy(
                max_new_tokens=3084 if mode == "source" else 512,
                repetition_penalty=1.0 if mode == "source" else 1.10,
                temperature=0.0,
                top_p=1.0,
                include_raw_model_logprob=False,
            )
            canonical_id = (
                f"human13:{image_id}:source" if mode == "source" else str(image_id)
            )
            base_by_image[image_id] = replace(
                request,
                request_id=canonical_id,
                generation_policy=policy,
            )
            prompt_by_image[image_id] = prompt_by_example[str(example.example_id)]
        contexts: list[tuple[int, object, RequestIdentity, float]] = []
        started = time.perf_counter()
        with open_backend_session(frontend.launch) as session:
            backend_version = str(session.receipt.backend_version)
            if mode == "source":

                def decode_source(
                    image_id: int, batch: tuple[object, ...]
                ) -> Sequence[object]:
                    batch_started = time.perf_counter()
                    results = tuple(session.decode(batch))
                    elapsed = time.perf_counter() - batch_started
                    contexts.append(
                        (
                            image_id,
                            results[0],
                            _scientific_request(
                                mode="source",
                                seed=None,
                                backend_version=backend_version,
                            ),
                            elapsed,
                        )
                    )
                    return results

                dispatch_source_batches(base_by_image, decode_source)
            else:

                def decode_k(
                    batch: PlannedBatch, base_request: object
                ) -> Sequence[object]:
                    scientific_requests = tuple(
                        replace(
                            base_request,
                            request_id=request.request_id,
                        )
                        for request in batch.requests
                    )
                    native_requests = tuple(
                        replace(request, request_id=str(index))
                        for index, request in enumerate(scientific_requests)
                    )
                    prompts, media_hashes = session._generation_prompts(native_requests)
                    batch_started = time.perf_counter()
                    try:
                        native_outputs = session._engine.generate(
                            prompts,
                            [
                                _vllm_sampling_params(
                                    request,
                                    stop_token_id=session._im_end_token_id(),
                                )
                                for request in batch.requests
                            ],
                            use_tqdm=False,
                        )
                    finally:
                        _close_prompt_images(prompts)
                    elapsed = time.perf_counter() - batch_started
                    ordered = _restore_native_request_order(native_outputs, 4)
                    results: list[object] = []
                    for (
                        planned,
                        native_request,
                        scientific_request,
                        native,
                        media_hash,
                    ) in zip(
                        batch.requests,
                        native_requests,
                        scientific_requests,
                        ordered,
                        media_hashes,
                        strict=True,
                    ):
                        result = session._materialize_result(
                            request=native_request,
                            native_output=native,
                            executed_media_sha256=media_hash,
                            raw_logprobs=None,
                        )
                        result = replace(
                            result, request_id=scientific_request.request_id
                        )
                        result.validate_for_request(scientific_request, session.receipt)
                        results.append(result)
                        contexts.append(
                            (
                                batch.image_id,
                                result,
                                _scientific_request(
                                    mode="k",
                                    seed=planned.seed,
                                    backend_version=backend_version,
                                ),
                                elapsed,
                            )
                        )
                    session._record_live_operational_smoke(
                        requests=scientific_requests,
                        results=results,
                    )
                    allocated, reserved = cuda_peak_memory_snapshot(__import__("torch"))
                    session._receipt = update_decode_performance_receipt(
                        session.receipt,
                        request_count=4,
                        generated_token_count=sum(
                            len(result.generated_token_ids) for result in results
                        ),
                        elapsed_seconds=elapsed,
                        peak_cuda_memory_allocated_bytes=allocated,
                        peak_cuda_memory_reserved_bytes=reserved,
                    )
                    return tuple(results)

                dispatch_k_batches(base_by_image, decode_k)
        elapsed_total = time.perf_counter() - started
        session_identity = _session_identity(session.receipt)
        execution_identity: dict[str, object] = {
            "mode": "dynamic_hf" if mode == "source" else "materialized_vllm",
            "source_identity": _json_normalized(asdict(default_binding().source)),
        }
        if execution_model is not None:
            execution_identity["receipt"] = dict(execution_model)
        records = [
            _record_from_result(
                mode=mode,
                image_id=image_id,
                frozen=frozen_by_id[image_id],
                prompt_metadata=prompt_by_image[image_id],
                execution_model_identity=execution_identity,
                session_identity=session_identity,
                result=result,
                request=request,
                batch_elapsed_seconds=batch_elapsed,
            )
            for image_id, result, request, batch_elapsed in contexts
        ]
        expected_count = 13 if mode == "source" else 208
        if len(records) != expected_count:
            raise ValueError("runtime result coverage is incomplete")
        runtime_counters = {
            "image_count": 13,
            "physical_batch_count": 13 if mode == "source" else 52,
            "request_count": expected_count,
            "generated_token_count": sum(
                int(record["runtime_counters"]["generated_token_count"])  # type: ignore[index]
                for record in records
            ),
            "wall_seconds": elapsed_total,
            "retry_count": 0,
            "resume_count": 0,
        }
        if mode == "k":
            from scripts.research.collect_human13_k16_vllm import cache_telemetry

            runtime_counters["cache_telemetry"] = cache_telemetry(session)
        _write_completed_collection(
            staging=staging,
            mode=mode,
            records=records,
            config_identity={
                "path": str(Path(config_path).resolve()),
                "fingerprint": resolved.fingerprint,
            },
            session_identity=session_identity,
            runtime_counters=runtime_counters,
        )

    execute_atomically(output_root=output_root, mode=mode, operation=operation)
    return Path(output_root).expanduser().resolve()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--execute-source", action="store_true")
    modes.add_argument("--execute-k", action="store_true")
    modes.add_argument("--validate-artifacts", action="store_true")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--k-root", type=Path)
    parser.add_argument("--source-config", type=Path, default=SOURCE_CONFIG_PATH)
    parser.add_argument("--k-config", type=Path, default=K_CONFIG_PATH)
    parser.add_argument("--panel", type=Path, default=Path(PANEL_PATH))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.execute_source or args.execute_k:
        if args.output_root is None:
            raise SystemExit("explicit execution requires --output-root")
        mode: Literal["source", "k"] = "source" if args.execute_source else "k"
        config_path = args.source_config if mode == "source" else args.k_config
        path = _execute_runtime(
            mode=mode,
            output_root=args.output_root,
            config_path=config_path,
            panel_path=args.panel,
        )
        print(json.dumps({"status": "completed", "mode": mode, "root": str(path)}))
        return 0
    if args.validate_artifacts:
        if args.source_root is None or args.k_root is None:
            raise SystemExit("artifact validation requires --source-root and --k-root")
        images = artifacts_to_image_inputs(
            source_root=args.source_root,
            k_root=args.k_root,
            panel_path=args.panel,
        )
        print(
            json.dumps(
                {
                    "status": "admitted",
                    "image_count": len(images),
                    "source_count": len(images),
                    "k_count": sum(len(image.sampled) for image in images),
                },
                sort_keys=True,
            )
        )
        return 0
    print(
        json.dumps(
            dry_run_plan(
                source_config_path=args.source_config,
                k_config_path=args.k_config,
                panel_path=args.panel,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
