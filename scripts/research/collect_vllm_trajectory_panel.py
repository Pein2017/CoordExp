#!/usr/bin/env python3
"""Collect sixteen sampled trajectories, with an optional greedy comparison, using vLLM.

This is an experiment-local collector for the dense-enumeration trajectory
panel.  One process owns one visible GPU and one vLLM engine.  Images are
sharded across workers; vLLM owns scheduling within each worker.  Completed
image batches are written as immutable per-mode artifacts so a long panel does
not live only in process memory.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from scripts.research.run_current_seeded_sampled_rollouts import (  # noqa: E402
    _build_requests,
    physical_image_id,
    select_examples,
)
from src.templates.renderer import BOX_END_TOKEN  # noqa: E402


DEFAULT_CONFIG = Path(
    "configs/coordexp_swift/infer/research/"
    "qwen3_vl_2b_step4887_candidate_pool_2432_vllm_panel.yaml"
)
SAMPLED_COUNT = 16
SOURCE_B16_ROW_BUDGET = 16
SOURCE_B16_ALLOWED_REPETITION_PENALTIES = frozenset({1.0, 1.1})
SOURCE_B16_ACCEPTED_STATUSES = frozenset({"accepted_budget", "accepted_natural_end"})
SOURCE_B16_INELIGIBLE_STATUSES = frozenset(
    {"failed_invalid_before_budget", "failed_token_limit_before_budget"}
)
SOURCE_B16_STATUSES = SOURCE_B16_ACCEPTED_STATUSES | SOURCE_B16_INELIGIBLE_STATUSES
SCHEMA_VERSION = "coordexp_vllm_trajectory_panel.v2"


def _decode_modes(*, sampled_only: bool, source_b16: bool = False) -> tuple[str, ...]:
    """Return the artifact modes required by the selected panel contract."""

    if sampled_only and source_b16:
        raise ValueError("source_b16 and sampled_only are mutually exclusive")
    if source_b16:
        return ("source_b16",)
    return ("sampled",) if sampled_only else ("greedy", "sampled")


def shard_examples(
    examples: Sequence[Any], *, worker_index: int, worker_count: int
) -> list[Any]:
    """Return the stable image shard owned by one worker."""

    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    if not 0 <= worker_index < worker_count:
        raise ValueError("worker_index must be in [0, worker_count)")
    return list(examples[worker_index::worker_count])


def _chunks(values: Sequence[Any], size: int) -> list[list[Any]]:
    if size <= 0:
        raise ValueError("image_batch_size must be positive")
    return [list(values[offset : offset + size]) for offset in range(0, len(values), size)]


def parse_image_ids(value: str | None) -> set[str] | None:
    if value is None:
        return None
    image_ids = {piece.strip() for piece in value.split(",") if piece.strip()}
    if not image_ids:
        raise ValueError("image_ids must contain at least one non-empty id")
    return image_ids


def sampling_params_kwargs(
    *,
    decode_mode: str,
    sample_count: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    max_new_tokens: int,
    stop_token_id: int,
    seed: int | None,
) -> dict[str, Any]:
    """Build the experiment-local vLLM sampling policy."""

    if decode_mode not in {"greedy", "sampled", "source_b16"}:
        raise ValueError("decode_mode must be greedy, sampled, or source_b16")
    if sample_count != SAMPLED_COUNT:
        raise ValueError(f"sample_count must be exactly {SAMPLED_COUNT}")
    if decode_mode == "source_b16":
        if repetition_penalty not in SOURCE_B16_ALLOWED_REPETITION_PENALTIES:
            raise ValueError(
                "source_b16 requires repetition_penalty to be exactly 1.0 or 1.1"
            )
    elif repetition_penalty != 1.0:
        raise ValueError("greedy and sampled trajectory panels require repetition_penalty=1.0")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if decode_mode in {"greedy", "source_b16"}:
        if temperature != 0.0:
            raise ValueError("greedy temperature must be zero")
        kwargs: dict[str, Any] = {
            "n": 1,
            "temperature": 0.0,
            "top_p": 1.0,
        }
    else:
        if temperature != 0.4 or top_p != 0.95:
            raise ValueError("sampled panel requires temperature=0.4 and top_p=0.95")
        kwargs = {
            "n": sample_count,
            "temperature": temperature,
            "top_p": top_p,
        }
    kwargs.update(
        {
            "top_k": 0,
            "repetition_penalty": repetition_penalty,
            "max_tokens": max_new_tokens,
            "stop_token_ids": [int(stop_token_id)],
            "ignore_eos": False,
            "detokenize": True,
            "skip_special_tokens": False,
            "spaces_between_special_tokens": True,
        }
    )
    if seed is not None:
        kwargs["seed"] = int(seed)
    return kwargs


def _sampling_params(**kwargs: Any) -> Any:
    from vllm import SamplingParams

    return SamplingParams(**sampling_params_kwargs(**kwargs))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_completions(native_output: Any, *, expected_count: int) -> list[Any]:
    completions = getattr(native_output, "outputs", None)
    if not isinstance(completions, Sequence) or len(completions) != expected_count:
        raise RuntimeError(
            f"vLLM returned {0 if completions is None else len(completions)} completions; "
            f"expected {expected_count}"
        )
    try:
        ordered = sorted(completions, key=lambda item: int(item.index))
        indices = [int(item.index) for item in ordered]
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError("vLLM completions lack integer sample indices") from exc
    if indices != list(range(expected_count)):
        raise RuntimeError(f"vLLM completion indices are not 0..{expected_count - 1}: {indices}")
    return ordered


def _materialize_completion(
    *,
    completion: Any,
    tokenizer: Any,
    stop_token_id: int,
) -> tuple[list[int], str, str]:
    token_ids = [int(value) for value in getattr(completion, "token_ids", ())]
    if not token_ids:
        raise RuntimeError("vLLM returned an empty completion")
    stop_positions = [index for index, value in enumerate(token_ids) if value == stop_token_id]
    finish_reason = str(getattr(completion, "finish_reason", "") or "")
    if stop_positions:
        if stop_positions != [len(token_ids) - 1] or finish_reason != "stop":
            raise RuntimeError("vLLM returned inconsistent terminal stop-token evidence")
        parser_ids = token_ids[:-1]
        stop_reason = "im_end"
    else:
        if finish_reason != "length":
            raise RuntimeError("vLLM completion has neither a terminal stop nor length finish")
        parser_ids = token_ids
        stop_reason = "length"
    text = str(
        tokenizer.decode(
            parser_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    )
    return token_ids, text, stop_reason


def _decode_token_ids(tokenizer: Any, token_ids: Sequence[int]) -> str:
    return str(
        tokenizer.decode(
            list(token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    )


def _source_b16_receipt(
    *,
    token_ids: Sequence[int],
    parser_text: str,
    stop_reason: str,
    tokenizer: Any,
    stop_token_id: int,
    row_id: str,
    image_width: int,
    image_height: int,
    raw_parser: Any,
) -> dict[str, Any]:
    """Project one raw greedy completion onto its clean Source@B16 prefix."""

    from src.inference.parsing import parse_compact_object_box_closed

    raw_token_ids = [int(value) for value in token_ids]
    if stop_reason == "im_end":
        if not raw_token_ids or raw_token_ids[-1] != stop_token_id:
            raise RuntimeError("source_b16 lacks terminal im_end token evidence")
        parser_token_ids = raw_token_ids[:-1]
    elif stop_reason == "length":
        parser_token_ids = raw_token_ids
    else:
        raise RuntimeError(f"unsupported source_b16 stop reason: {stop_reason}")
    if _decode_token_ids(tokenizer, parser_token_ids) != parser_text:
        raise RuntimeError("source_b16 parser token/text parity failed")

    try:
        box_end_ids = tokenizer.encode(BOX_END_TOKEN, add_special_tokens=False)
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError("source_b16 tokenizer cannot encode box-end grammar token") from exc
    if (
        not isinstance(box_end_ids, Sequence)
        or isinstance(box_end_ids, (str, bytes))
        or len(box_end_ids) != 1
    ):
        raise RuntimeError("source_b16 box-end grammar token must have exactly one token id")
    box_end_token_id = int(box_end_ids[0])
    if _decode_token_ids(tokenizer, [box_end_token_id]) != BOX_END_TOKEN:
        raise RuntimeError("source_b16 box-end grammar token does not round-trip")

    def parse(text: str) -> Any:
        return parse_compact_object_box_closed(
            text,
            row_id=row_id,
            row_index=0,
            image_width=image_width,
            image_height=image_height,
        )

    empty_parser = parse("")
    projected_token_ids: list[int] = []
    projected_text = ""
    projected_parser = empty_parser
    projected_end_offset = 0
    projected_count = 0
    failure_parser: Any | None = None
    for end_offset, token_id in enumerate(parser_token_ids, start=1):
        if token_id != box_end_token_id:
            continue
        candidate_token_ids = parser_token_ids[:end_offset]
        candidate_text = _decode_token_ids(tokenizer, candidate_token_ids)
        candidate_parser = parse(candidate_text)
        if candidate_parser.dropped_prediction_count:
            failure_parser = candidate_parser
            break
        candidate_count = candidate_parser.valid_prediction_count
        if candidate_count < projected_count:
            raise RuntimeError("source_b16 parser row count regressed at a box-end token")
        projected_token_ids = candidate_token_ids
        projected_text = candidate_text
        projected_parser = candidate_parser
        projected_end_offset = end_offset
        projected_count = candidate_count
        if projected_count == SOURCE_B16_ROW_BUDGET:
            break
        if projected_count > SOURCE_B16_ROW_BUDGET:
            raise RuntimeError("source_b16 exceeded its row budget before projection")

    raw_count = int(raw_parser.valid_prediction_count)
    raw_evidence = raw_parser.to_artifact_dict()
    if projected_count == SOURCE_B16_ROW_BUDGET:
        status = "accepted_budget"
    elif failure_parser is not None or raw_parser.dropped_prediction_count:
        status = "failed_invalid_before_budget"
    elif stop_reason == "im_end":
        projected_token_ids = list(parser_token_ids)
        projected_text = parser_text
        projected_parser = raw_parser
        projected_end_offset = len(parser_token_ids)
        projected_count = raw_count
        status = "accepted_natural_end"
    else:
        status = "failed_token_limit_before_budget"

    receipt: dict[str, Any] = {
        "status": status,
        "row_budget": SOURCE_B16_ROW_BUDGET,
        "raw_valid_complete_row_count": raw_count,
        "projected_valid_complete_row_count": projected_count,
        "projected_token_end_offset_exclusive": projected_end_offset,
        "projected_token_ids": projected_token_ids,
        "projected_token_ids_sha256": _sha256_json(projected_token_ids),
        "projected_text": projected_text,
        "projected_parser_evidence": projected_parser.to_artifact_dict(),
        "token_limit_before_budget": (
            stop_reason == "length" and projected_count < SOURCE_B16_ROW_BUDGET
        ),
        "natural_end_before_budget": (
            stop_reason == "im_end" and projected_count < SOURCE_B16_ROW_BUDGET
        ),
        "raw_parser_evidence": raw_evidence,
    }
    if failure_parser is not None:
        receipt["failure_parser_evidence"] = failure_parser.to_artifact_dict()
    return receipt


def _source_b16_receipt_is_valid(
    row: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    tokenizer: Any | None,
    stop_token_id: int | None,
) -> bool:
    """Return whether a persisted Source@B16 receipt exactly replays from raw evidence."""

    from src.inference.parsing import parse_compact_object_box_closed

    status = receipt.get("status")
    if (
        tokenizer is None
        or stop_token_id is None
        or status not in SOURCE_B16_STATUSES
    ):
        return False
    try:
        raw_token_ids = [int(value) for value in row["generated_token_ids"]]
        image_width = int(row["image_width"])
        image_height = int(row["image_height"])
        example_id = row["example_id"]
        trajectory_id = row["trajectory_id"]
        generated_text = row["generated_text"]
    except (KeyError, TypeError, ValueError):
        return False
    if (
        not isinstance(example_id, str)
        or not example_id
        or trajectory_id != "source-b16"
        or row.get("decode_mode") != "source_b16"
        or not isinstance(generated_text, str)
        or image_width <= 0
        or image_height <= 0
        or not raw_token_ids
        or row.get("generated_token_ids_sha256") != _sha256_json(raw_token_ids)
    ):
        return False
    if row.get("stop_reason") == "im_end":
        if raw_token_ids.count(stop_token_id) != 1 or raw_token_ids[-1] != stop_token_id:
            return False
        parser_token_ids = raw_token_ids[:-1]
    elif row.get("stop_reason") == "length":
        if stop_token_id in raw_token_ids:
            return False
        parser_token_ids = raw_token_ids
    else:
        return False
    try:
        if _decode_token_ids(tokenizer, parser_token_ids) != generated_text:
            return False
        replayed_raw_parser = parse_compact_object_box_closed(
            generated_text,
            row_id=f"{example_id}:{trajectory_id}",
            row_index=0,
            image_width=image_width,
            image_height=image_height,
        )
    except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError):
        return False
    try:
        replayed_receipt = _source_b16_receipt(
            token_ids=raw_token_ids,
            parser_text=generated_text,
            stop_reason=str(row["stop_reason"]),
            tokenizer=tokenizer,
            stop_token_id=stop_token_id,
            row_id=f"{example_id}:{trajectory_id}",
            image_width=image_width,
            image_height=image_height,
            raw_parser=replayed_raw_parser,
        )
    except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError):
        return False
    return dict(receipt) == replayed_receipt


def _source_b16_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize complete Source@B16 receipts without treating ineligibility as failure."""

    status_counts: dict[str, int] = {}
    ineligible_image_ids: list[Any] = []
    for row in rows:
        receipt = row.get("source_b16")
        if not isinstance(receipt, Mapping):
            raise RuntimeError("source_b16 rollout lacks a receipt")
        status = receipt.get("status")
        if status not in SOURCE_B16_STATUSES:
            raise RuntimeError(f"source_b16 rollout has unknown status: {status!r}")
        status_counts[str(status)] = status_counts.get(str(status), 0) + 1
        if status in SOURCE_B16_INELIGIBLE_STATUSES:
            image_id = row.get("image_id")
            if image_id not in ineligible_image_ids:
                ineligible_image_ids.append(image_id)
    accepted_count = sum(
        status_counts.get(status, 0) for status in SOURCE_B16_ACCEPTED_STATUSES
    )
    ineligible_count = sum(
        status_counts.get(status, 0) for status in SOURCE_B16_INELIGIBLE_STATUSES
    )
    return {
        "status_counts": dict(sorted(status_counts.items())),
        "accepted_count": accepted_count,
        "ineligible_count": ineligible_count,
        "ineligible_image_ids": ineligible_image_ids,
    }


def _source_b16_worker_summary(batches: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate batch-level Source@B16 eligibility evidence in manifest order."""

    status_counts: dict[str, int] = {}
    ineligible_image_ids: list[Any] = []
    for batch in batches:
        summary = batch.get("source_b16")
        if not isinstance(summary, Mapping):
            raise RuntimeError("source_b16 batch lacks eligibility summary")
        counts = summary.get("status_counts")
        if not isinstance(counts, Mapping):
            raise RuntimeError("source_b16 batch lacks status counts")
        for status, count in counts.items():
            if status not in SOURCE_B16_STATUSES or not isinstance(count, int) or count < 0:
                raise RuntimeError("source_b16 batch has invalid status counts")
            status_counts[status] = status_counts.get(status, 0) + count
        image_ids = summary.get("ineligible_image_ids")
        if not isinstance(image_ids, list):
            raise RuntimeError("source_b16 batch lacks ineligible image ids")
        for image_id in image_ids:
            if image_id not in ineligible_image_ids:
                ineligible_image_ids.append(image_id)
    accepted_count = sum(
        status_counts.get(status, 0) for status in SOURCE_B16_ACCEPTED_STATUSES
    )
    ineligible_count = sum(
        status_counts.get(status, 0) for status in SOURCE_B16_INELIGIBLE_STATUSES
    )
    return {
        "status_counts": dict(sorted(status_counts.items())),
        "accepted_count": accepted_count,
        "ineligible_count": ineligible_count,
        "ineligible_image_ids": ineligible_image_ids,
    }


def _artifact_config(
    *,
    infer_config: Path,
    resolved_fingerprint: str,
    model_dtype: str,
    decode_mode: str,
    max_new_tokens: int,
    request_seed: int | None,
    worker_index: int,
    worker_count: int,
    image_batch_size: int,
    sampled_only: bool,
    source_b16: bool = False,
    repetition_penalty: float = 1.0,
) -> dict[str, Any]:
    panel_mode = (
        "source_b16" if source_b16 else "sampled_only" if sampled_only else "paired"
    )
    greedy = decode_mode in {"greedy", "source_b16"}
    config: dict[str, Any] = {
        "infer_config_path": str(infer_config.resolve()),
        "resolved_fingerprint": resolved_fingerprint,
        "model_dtype": model_dtype,
        "backend": "vllm",
        "decode_mode": decode_mode,
        "panel_mode": panel_mode,
        "temperature": 0.0 if greedy else 0.4,
        "top_p": 1.0 if greedy else 0.95,
        "repetition_penalty": repetition_penalty,
        "max_new_tokens": max_new_tokens,
        "sample_count": 1 if greedy else SAMPLED_COUNT,
        "sample_index_range": None if greedy else [0, SAMPLED_COUNT - 1],
        "sampling_is_not_infer_config": True,
        "sampling_order": "request_major",
        "worker_index": worker_index,
        "worker_count": worker_count,
        "image_batch_size": image_batch_size,
        "max_num_seqs": 32,
    }
    if source_b16:
        config["source_b16_row_budget"] = SOURCE_B16_ROW_BUDGET
    if request_seed is not None:
        config["request_seed"] = int(request_seed)
    return config


def build_artifact(
    *,
    config: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    prompt_metadata: Mapping[str, Any],
    rollouts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build and validate one immutable batch artifact."""

    mode = config.get("decode_mode")
    if mode not in {"greedy", "sampled", "source_b16"}:
        raise ValueError("artifact decode_mode is required")
    rows = [dict(row) for row in rollouts]
    if not rows:
        raise ValueError("artifact must contain rollout rows")
    by_example: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("decode_mode") != mode:
            raise ValueError("rollout mode differs from artifact mode")
        example_id = str(row.get("example_id", ""))
        if not example_id or example_id not in prompt_metadata:
            raise ValueError("rollout lacks matching prompt metadata")
        metadata = prompt_metadata[example_id]
        if not isinstance(metadata, Mapping):
            raise ValueError("rollout prompt metadata is not an object")
        if row.get("source_image_file_sha256") != metadata.get(
            "source_image_file_sha256"
        ):
            raise ValueError("rollout source image identity differs from prompt metadata")
        executed_rgb_sha256 = row.get("executed_rgb_sha256")
        if not isinstance(executed_rgb_sha256, str) or len(executed_rgb_sha256) != 64:
            raise ValueError("rollout lacks executed RGB image identity")
        by_example.setdefault(example_id, []).append(row)
    expected_per_image = 1 if mode in {"greedy", "source_b16"} else SAMPLED_COUNT
    for example_id, image_rows in by_example.items():
        if len(image_rows) != expected_per_image:
            raise ValueError(f"{example_id} has {len(image_rows)} {mode} rows")
        if mode == "sampled" and [row.get("sample_index") for row in image_rows] != list(
            range(SAMPLED_COUNT)
        ):
            raise ValueError(f"{example_id} sampled rows are not sample_index 0..15")
    if set(by_example) != set(prompt_metadata):
        raise ValueError("prompt metadata does not cover exact artifact images")
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_mode": "experiment_local_vllm_trajectory_panel",
        "config": dict(config),
        "model_identity": dict(model_identity),
        "prompt_metadata": dict(prompt_metadata),
        "rollout_count": len(rows),
        "rollouts": rows,
    }


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _run_native_generation(
    *,
    session: Any,
    requests: Sequence[Any],
    decode_mode: str,
    max_new_tokens: int,
    request_seed: int | None,
    repetition_penalty: float = 1.0,
) -> tuple[tuple[Any, ...], tuple[str, ...], float]:
    from src.inference.vllm_backend import (
        _close_prompt_images,
        _restore_native_request_order,
    )

    greedy = decode_mode in {"greedy", "source_b16"}
    prompts, media_hashes = session._generation_prompts(requests)
    started_at = time.perf_counter()
    try:
        outputs = session._engine.generate(
            prompts,
            _sampling_params(
                decode_mode=decode_mode,
                sample_count=SAMPLED_COUNT,
                temperature=0.0 if greedy else 0.4,
                top_p=1.0 if greedy else 0.95,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
                stop_token_id=session._im_end_token_id(),
                seed=None if greedy else request_seed,
            ),
            use_tqdm=False,
        )
    finally:
        _close_prompt_images(prompts)
    elapsed_seconds = time.perf_counter() - started_at
    return (
        _restore_native_request_order(outputs, len(requests)),
        media_hashes,
        elapsed_seconds,
    )


def _record_research_live_decode(
    *,
    session: Any,
    requests: Sequence[Any],
    rows: Sequence[Mapping[str, Any]],
    decode_mode: str,
) -> None:
    """Bind experiment-local multi-completion evidence to live preflight."""

    settings = dict(session.receipt.effective_settings)
    value = settings.get("runtime_preflight")
    if not isinstance(value, Mapping):
        raise RuntimeError("vLLM session lacks upstream runtime preflight evidence")
    preflight = dict(value)
    if preflight.get("status") not in {
        "ready_for_engine_construction",
        "passed_live_decode",
    }:
        raise RuntimeError(
            f"unexpected vLLM runtime preflight status: {preflight.get('status')}"
        )
    previous = preflight.get("research_panel_live_decode")
    evidence = (
        dict(previous)
        if isinstance(previous, Mapping)
        else {"scope": "experiment_local_multi_completion", "calls": []}
    )
    calls = list(evidence.get("calls") or [])
    calls.append(
        {
            "decode_mode": decode_mode,
            "request_count": len(requests),
            "completion_count": len(rows),
            "generated_token_count": sum(
                len(row.get("generated_token_ids", ())) for row in rows
            ),
            "length_finished_completion_count": sum(
                row.get("stop_reason") == "length" for row in rows
            ),
        }
    )
    evidence["calls"] = calls
    evidence["request_count"] = sum(int(call["request_count"]) for call in calls)
    evidence["completion_count"] = sum(
        int(call["completion_count"]) for call in calls
    )
    evidence["generated_token_count"] = sum(
        int(call["generated_token_count"]) for call in calls
    )
    preflight["status"] = "passed_live_decode"
    preflight["live_decode"] = {
        "scope": "experiment_local_multi_completion",
        "first_decode_mode": calls[0]["decode_mode"],
        "first_request_count": calls[0]["request_count"],
        "first_completion_count": calls[0]["completion_count"],
    }
    preflight["research_panel_live_decode"] = evidence
    settings["runtime_preflight"] = preflight
    session._receipt = replace(session.receipt, effective_settings=settings)


def _generation_health(
    rows: Sequence[Mapping[str, Any]], *, elapsed_seconds: float
) -> dict[str, Any]:
    generated_token_count = sum(len(row.get("generated_token_ids", ())) for row in rows)
    completion_count = len(rows)
    stop_counts = {
        reason: sum(row.get("stop_reason") == reason for row in rows)
        for reason in ("im_end", "length")
    }
    parser_status_counts: dict[str, int] = {}
    for row in rows:
        predictions = row.get("predictions")
        status = (
            str(predictions.get("parse_status", "unknown"))
            if isinstance(predictions, Mapping)
            else "unknown"
        )
        parser_status_counts[status] = parser_status_counts.get(status, 0) + 1
    return {
        "completion_count": completion_count,
        "generated_token_count": generated_token_count,
        "stop_reason_counts": stop_counts,
        "natural_closure_count": stop_counts["im_end"],
        "parser_status_counts": parser_status_counts,
        "elapsed_seconds": elapsed_seconds,
        "completions_per_second": completion_count / elapsed_seconds,
        "routes_per_second": completion_count / elapsed_seconds,
        "generated_tokens_per_second": generated_token_count / elapsed_seconds,
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return dict(value)


def _stable_session_identity(receipt: Any) -> dict[str, Any]:
    """Remove rank-local observations from cross-process artifact identity."""

    payload = dict(receipt.to_artifact_dict())
    settings_value = payload.get("effective_settings")
    if not isinstance(settings_value, Mapping):
        raise RuntimeError("backend receipt lacks effective settings")
    settings = dict(settings_value)
    settings.pop("runtime_preflight", None)
    settings.pop("performance", None)
    payload["effective_settings"] = settings
    return payload


def _validated_resume_batch(
    *,
    worker_root: Path,
    entry: Mapping[str, Any],
    expected_batch_index: int,
    expected_examples: Sequence[Any],
    resolved_fingerprint: str,
    model_identity: Mapping[str, Any],
    decode_modes: Sequence[str] = ("greedy", "sampled"),
    tokenizer: Any | None = None,
    stop_token_id: int | None = None,
    expected_repetition_penalty: float = 1.0,
) -> dict[str, Any] | None:
    """Return a fail-closed completed batch entry, otherwise regenerate it."""

    try:
        if entry.get("batch_index") != expected_batch_index:
            return None
        expected_ids = [physical_image_id(example) for example in expected_examples]
        if entry.get("image_ids") != expected_ids:
            return None
        modes = tuple(decode_modes)
        if (
            not modes
            or len(set(modes)) != len(modes)
            or any(mode not in {"greedy", "sampled", "source_b16"} for mode in modes)
        ):
            raise ValueError("resume decode modes are invalid")
        parts = entry.get("artifacts")
        if not isinstance(parts, Mapping) or set(parts) != set(modes):
            return None
        recomputed_health: dict[str, Any] = {}
        source_b16_summary: dict[str, Any] | None = None
        expected_panel_mode = (
            "source_b16"
            if modes == ("source_b16",)
            else "sampled_only"
            if modes == ("sampled",)
            else "paired"
        )
        for mode in modes:
            part = parts[mode]
            if not isinstance(part, Mapping):
                return None
            name = part.get("path")
            expected_hash = part.get("sha256")
            if not isinstance(name, str) or not isinstance(expected_hash, str):
                return None
            path = worker_root / name
            if not path.is_file() or _sha256_file(path) != expected_hash:
                return None
            artifact = _load_json_object(path)
            config = artifact.get("config")
            rows = artifact.get("rollouts")
            if (
                artifact.get("schema_version") != SCHEMA_VERSION
                or artifact.get("model_identity") != dict(model_identity)
                or not isinstance(config, Mapping)
                or config.get("resolved_fingerprint") != resolved_fingerprint
                or config.get("decode_mode") != mode
                or config.get("panel_mode", "paired") != expected_panel_mode
                or config.get("repetition_penalty") != expected_repetition_penalty
                or not isinstance(rows, list)
            ):
                return None
            row_ids = list(dict.fromkeys(row.get("image_id") for row in rows))
            if row_ids != expected_ids:
                return None
            expected_count = len(expected_ids) * (
                1 if mode in {"greedy", "source_b16"} else SAMPLED_COUNT
            )
            if len(rows) != expected_count:
                return None
            if mode != "source_b16" and any(
                row.get("stop_reason") == "length" for row in rows
            ):
                return None
            if mode == "source_b16" and any(
                not isinstance(row.get("source_b16"), Mapping)
                or not _source_b16_receipt_is_valid(
                    row,
                    row["source_b16"],
                    tokenizer=tokenizer,
                    stop_token_id=stop_token_id,
                )
                for row in rows
            ):
                return None
            if mode == "source_b16":
                source_b16_summary = _source_b16_summary(rows)
                if entry.get("source_b16") != source_b16_summary:
                    return None
            if mode == "sampled":
                for offset in range(0, len(rows), SAMPLED_COUNT):
                    if [row.get("sample_index") for row in rows[offset : offset + SAMPLED_COUNT]] != list(
                        range(SAMPLED_COUNT)
                    ):
                        return None
            old_health = entry.get("generation_health", {}).get(mode)
            if not isinstance(old_health, Mapping):
                return None
            elapsed = float(old_health.get("elapsed_seconds", 0.0))
            if elapsed <= 0.0:
                return None
            health = _generation_health(rows, elapsed_seconds=elapsed)
            for field in (
                "completion_count",
                "generated_token_count",
                "stop_reason_counts",
                "natural_closure_count",
                "parser_status_counts",
            ):
                if old_health.get(field) != health[field]:
                    return None
            recomputed_health[mode] = health
        resumed_entry: dict[str, Any] = {
            "batch_index": expected_batch_index,
            "image_ids": expected_ids,
            "artifacts": {mode: dict(parts[mode]) for mode in modes},
            "generation_health": recomputed_health,
            "resumed": True,
        }
        if source_b16_summary is not None:
            resumed_entry["source_b16"] = source_b16_summary
        return resumed_entry
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _open_research_vllm_session(launch: Any) -> Any:
    """Open max_num_seqs=32 through the upstream live-preflight contract."""

    from src.inference.vllm_backend import (
        open_vllm_backend_session,
    )

    session = open_vllm_backend_session(launch)
    settings = session.receipt.effective_settings
    engine_kwargs = settings.get("engine_kwargs")
    if not isinstance(engine_kwargs, Mapping) or engine_kwargs.get("max_num_seqs") != 32:
        session.close()
        raise ValueError("research panel opener requires max_num_seqs=32")
    preflight = settings.get("runtime_preflight")
    if not isinstance(preflight, Mapping) or preflight.get("status") != (
        "ready_for_engine_construction"
    ):
        session.close()
        raise RuntimeError("research panel opener lacks ready live preflight")
    return session


def resolve_panel_execution_model(
    resolved: Any,
    *,
    resolver: Callable[[Any], Mapping[str, Any] | None] | None = None,
    validator: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve and validate the composed checkpoint required by vLLM."""

    if resolver is None:
        from src.inference.pipeline import _resolve_execution_model_for_run

        resolver = _resolve_execution_model_for_run
    if validator is None:
        from src.inference.execution_model import validate_execution_model_receipt

        validator = validate_execution_model_receipt
    execution_model = resolver(resolved)
    if not isinstance(execution_model, Mapping):
        raise RuntimeError("vLLM panel requires a resolved execution-model identity")
    validated = dict(validator(execution_model))
    return validated


def _rollout_rows(
    *,
    native_outputs: Sequence[Any],
    requests: Sequence[Any],
    examples: Sequence[Any],
    media_hashes: Sequence[str],
    session: Any,
    decode_mode: str,
    request_seed: int | None,
) -> list[dict[str, Any]]:
    from src.inference.parsing import parse_compact_object_box_closed

    rows: list[dict[str, Any]] = []
    expected_count = 1 if decode_mode in {"greedy", "source_b16"} else SAMPLED_COUNT
    for native, request, example, media_hash in zip(
        native_outputs, requests, examples, media_hashes, strict=True
    ):
        prompt_ids = [int(value) for value in (native.prompt_token_ids or ())]
        if tuple(prompt_ids) != request.expected_executed_prompt_token_ids:
            raise RuntimeError(f"prompt token parity failed for {example.example_id}")
        for sample_index, completion in enumerate(
            _ordered_completions(native, expected_count=expected_count)
        ):
            token_ids, text, stop_reason = _materialize_completion(
                completion=completion,
                tokenizer=session._tokenizer,
                stop_token_id=session._im_end_token_id(),
            )
            trajectory_id = (
                "greedy"
                if decode_mode == "greedy"
                else "source-b16"
                if decode_mode == "source_b16"
                else f"sample-{sample_index:02d}"
            )
            image_width = int(example.image.width)
            image_height = int(example.image.height)
            parsed = parse_compact_object_box_closed(
                text,
                row_id=f"{example.example_id}:{trajectory_id}",
                row_index=0,
                image_width=image_width,
                image_height=image_height,
            )
            row: dict[str, Any] = {
                "image_id": physical_image_id(example),
                "example_id": str(example.example_id),
                "trajectory_id": trajectory_id,
                "decode_mode": decode_mode,
                "generated_token_ids": token_ids,
                "generated_token_ids_sha256": _sha256_json(token_ids),
                "generated_text": text,
                "stop_reason": stop_reason,
                "prompt_token_ids": prompt_ids,
                "prompt_token_ids_sha256": _sha256_json(prompt_ids),
                "observed_image_grid_thw": None,
                "source_image_file_sha256": request.image_sha256,
                "executed_rgb_sha256": media_hash,
                "predictions": parsed.to_artifact_dict(),
            }
            if decode_mode == "source_b16":
                row["image_width"] = image_width
                row["image_height"] = image_height
                row["source_b16"] = _source_b16_receipt(
                    token_ids=token_ids,
                    parser_text=text,
                    stop_reason=stop_reason,
                    tokenizer=session._tokenizer,
                    stop_token_id=session._im_end_token_id(),
                    row_id=f"{example.example_id}:{trajectory_id}",
                    image_width=image_width,
                    image_height=image_height,
                    raw_parser=parsed,
                )
            if decode_mode == "sampled":
                row["sample_index"] = sample_index
            if request_seed is not None:
                row["request_seed"] = int(request_seed)
            rows.append(row)
    return rows


def collect_panel(
    *,
    infer_config: Path,
    output_root: Path,
    worker_index: int,
    worker_count: int,
    image_batch_size: int,
    image_ids: set[str] | None,
    max_images: int | None,
    request_seed: int | None,
    resume: bool,
    sampled_only: bool = False,
    source_b16: bool = False,
    source_b16_repetition_penalty: float = 1.0,
) -> Path:
    from src.config.fingerprint import sha256_json
    from src.config.inference import load_infer_config
    from src.data import load_raw_examples
    from src.inference.backend import open_backend_session
    from src.inference.runtime import assemble_frontend

    resolved = load_infer_config(infer_config.resolve(strict=True))
    config = resolved.config
    if config.backend.type != "vllm":
        raise ValueError("collector requires a vLLM inference config")
    if config.generation.batch_size != 32:
        raise ValueError("collector requires generation.batch_size=32 (vLLM max_num_seqs)")
    if sampled_only and source_b16:
        raise ValueError("source_b16 and sampled_only are mutually exclusive")
    if not source_b16 and source_b16_repetition_penalty != 1.0:
        raise ValueError(
            "source_b16_repetition_penalty may differ from 1.0 only with --source-b16"
        )
    if source_b16_repetition_penalty not in SOURCE_B16_ALLOWED_REPETITION_PENALTIES:
        raise ValueError("source_b16_repetition_penalty must be exactly 1.0 or 1.1")
    config_repetition_penalty = float(config.generation.repetition_penalty)
    if source_b16 and config_repetition_penalty != source_b16_repetition_penalty:
        raise ValueError(
            "source_b16 repetition penalty mismatch: "
            f"infer config has {config_repetition_penalty}, CLI requested "
            f"{source_b16_repetition_penalty}"
        )
    effective_repetition_penalty = (
        source_b16_repetition_penalty if source_b16 else 1.0
    )
    required_max_new_tokens = 2048 if source_b16 else 1024
    if config.generation.max_new_tokens != required_max_new_tokens:
        raise ValueError(f"collector requires max_new_tokens={required_max_new_tokens}")
    selected_examples = select_examples(
        list(load_raw_examples(config.data.input_jsonl)), image_ids
    )
    examples = shard_examples(
        selected_examples,
        worker_index=worker_index,
        worker_count=worker_count,
    )
    if max_images is not None:
        if max_images <= 0:
            raise ValueError("max_images must be positive")
        examples = examples[:max_images]
    if not examples:
        raise ValueError("worker shard contains no images")
    decode_modes = _decode_modes(sampled_only=sampled_only, source_b16=source_b16)
    panel_mode = "source_b16" if source_b16 else "sampled_only" if sampled_only else "paired"

    worker_root = output_root / f"worker-{worker_index:02d}-of-{worker_count:02d}"
    if worker_root.exists() and any(worker_root.iterdir()) and not resume:
        raise ValueError(f"refusing non-empty worker output root: {worker_root}")
    worker_root.mkdir(parents=True, exist_ok=True)
    old_manifest: dict[str, Any] = {}
    manifest_path = worker_root / "manifest.json"
    if resume and manifest_path.is_file():
        try:
            candidate = _load_json_object(manifest_path)
            if (
                candidate.get("worker_index") == worker_index
                and candidate.get("worker_count") == worker_count
                and candidate.get("image_count") == len(examples)
                and candidate.get("panel_mode", "paired") == panel_mode
            ):
                old_manifest = candidate
        except (OSError, ValueError, json.JSONDecodeError):
            old_manifest = {}
    execution_model = resolve_panel_execution_model(resolved)
    frontend = assemble_frontend(
        config,
        generation_config_fingerprint=sha256_json(config.generation.model_dump(mode="json")),
        execution_model=execution_model,
    )
    requests, prompt_metadata = _build_requests(config, frontend, examples)
    for metadata in prompt_metadata.values():
        metadata["source_image_file_sha256"] = metadata.pop("image_sha256")
        metadata["source_image_hash_semantics"] = "source_file_bytes"
    chunks = _chunks(list(range(len(examples))), image_batch_size)
    manifest: dict[str, Any] = {
        "schema_version": "vllm_trajectory_panel_worker_manifest.v1",
        "worker_index": worker_index,
        "worker_count": worker_count,
        "image_count": len(examples),
        "image_batch_size": image_batch_size,
        "max_num_seqs": 32,
        "panel_mode": panel_mode,
        "decode_modes": list(decode_modes),
        "trajectory_contract": (
            "one_deterministic_greedy_raw_trajectory_projected_to_source_b16_per_image"
            if source_b16
            else
            "sixteen_sampled_trajectories_per_image"
            if sampled_only
            else "one_greedy_plus_sixteen_sampled_trajectories_per_image"
        ),
        "runtime_contract": "upstream_live_preflight_plus_research_multi_completion",
        "repetition_penalty": effective_repetition_penalty,
        "zero_truncation_required": not source_b16,
        "completed_image_count": 0,
        "status": "running",
        "batches": [],
    }
    if source_b16:
        manifest["source_b16_row_budget"] = SOURCE_B16_ROW_BUDGET
        manifest["source_b16"] = _source_b16_worker_summary([])
    old_batches = {
        entry.get("batch_index"): entry
        for entry in old_manifest.get("batches", [])
        if isinstance(entry, Mapping)
    }
    overall_started_at = time.perf_counter()
    generated_new_batch = False
    with open_backend_session(frontend.launch, opener=_open_research_vllm_session) as session:
        model_identity = _stable_session_identity(session.receipt)
        for batch_index, indices in enumerate(chunks):
            batch_examples = [examples[index] for index in indices]
            batch_requests = [requests[index] for index in indices]
            resumed_batch = _validated_resume_batch(
                worker_root=worker_root,
                entry=old_batches.get(batch_index, {}),
                expected_batch_index=batch_index,
                expected_examples=batch_examples,
                resolved_fingerprint=resolved.fingerprint,
                model_identity=model_identity,
                decode_modes=decode_modes,
                tokenizer=session._tokenizer,
                stop_token_id=session._im_end_token_id(),
                expected_repetition_penalty=effective_repetition_penalty,
            )
            if resumed_batch is not None:
                manifest["batches"].append(resumed_batch)
                manifest["completed_image_count"] += len(batch_examples)
                if source_b16:
                    manifest["source_b16"] = _source_b16_worker_summary(manifest["batches"])
                _atomic_write_json(manifest_path, manifest)
                continue
            batch_prompt_metadata = {
                str(example.example_id): prompt_metadata[str(example.example_id)]
                for example in batch_examples
            }
            written: dict[str, dict[str, str]] = {}
            mode_health: dict[str, Any] = {}
            batch_length_count = 0
            batch_source_b16_summary: dict[str, Any] | None = None
            for decode_mode in decode_modes:
                generated_new_batch = True
                outputs, media_hashes, elapsed_seconds = _run_native_generation(
                    session=session,
                    requests=batch_requests,
                    decode_mode=decode_mode,
                    max_new_tokens=config.generation.max_new_tokens,
                    request_seed=request_seed if decode_mode == "sampled" else None,
                    repetition_penalty=effective_repetition_penalty,
                )
                rows = _rollout_rows(
                    native_outputs=outputs,
                    requests=batch_requests,
                    examples=batch_examples,
                    media_hashes=media_hashes,
                    session=session,
                    decode_mode=decode_mode,
                    request_seed=request_seed if decode_mode == "sampled" else None,
                )
                _record_research_live_decode(
                    session=session,
                    requests=batch_requests,
                    rows=rows,
                    decode_mode=decode_mode,
                )
                health = _generation_health(rows, elapsed_seconds=elapsed_seconds)
                mode_health[decode_mode] = health
                if decode_mode != "source_b16":
                    batch_length_count += int(health["stop_reason_counts"]["length"])
                else:
                    if any(
                        not _source_b16_receipt_is_valid(
                            row,
                            row["source_b16"],
                            tokenizer=session._tokenizer,
                            stop_token_id=session._im_end_token_id(),
                        )
                        for row in rows
                    ):
                        raise RuntimeError("source_b16 receipt failed immediate replay validation")
                    batch_source_b16_summary = _source_b16_summary(rows)
                artifact = build_artifact(
                    config=_artifact_config(
                        infer_config=infer_config,
                        resolved_fingerprint=resolved.fingerprint,
                        model_dtype=str(config.model.dtype),
                        decode_mode=decode_mode,
                        max_new_tokens=config.generation.max_new_tokens,
                        request_seed=request_seed if decode_mode == "sampled" else None,
                        worker_index=worker_index,
                        worker_count=worker_count,
                        image_batch_size=image_batch_size,
                        sampled_only=sampled_only,
                        source_b16=source_b16,
                        repetition_penalty=effective_repetition_penalty,
                    ),
                    model_identity=model_identity,
                    prompt_metadata=batch_prompt_metadata,
                    rollouts=rows,
                )
                path = worker_root / f"{decode_mode}-batch-{batch_index:05d}.json"
                _atomic_write_json(path, artifact)
                written[decode_mode] = {
                    "path": path.name,
                    "sha256": _sha256_file(path),
                }
            batch_entry: dict[str, Any] = {
                "batch_index": batch_index,
                "image_ids": [physical_image_id(example) for example in batch_examples],
                "artifacts": written,
                "generation_health": mode_health,
            }
            if batch_source_b16_summary is not None:
                batch_entry["source_b16"] = batch_source_b16_summary
            manifest["batches"].append(batch_entry)
            manifest["completed_image_count"] += len(batch_examples)
            if source_b16:
                manifest["source_b16"] = _source_b16_worker_summary(manifest["batches"])
            manifest["wall_elapsed_seconds_current_process"] = (
                time.perf_counter() - overall_started_at
            )
            manifest["length_finished_completion_count"] = sum(
                int(mode["stop_reason_counts"]["length"])
                for batch in manifest["batches"]
                for mode in batch["generation_health"].values()
            )
            if batch_length_count:
                manifest["status"] = "failed_truncation"
            _atomic_write_json(manifest_path, manifest)
            if batch_length_count:
                session.close()
                manifest["runtime_receipt"] = session.receipt.to_artifact_dict()
                _atomic_write_json(manifest_path, manifest)
                raise RuntimeError(
                    f"batch {batch_index} produced {batch_length_count} length-finished "
                    "completions; zero-truncation panel collection stopped"
                )
    current_runtime_receipt = session.receipt.to_artifact_dict()
    previous_runtime_receipt = old_manifest.get("runtime_receipt")
    if not generated_new_batch and isinstance(previous_runtime_receipt, Mapping):
        manifest["runtime_receipt"] = dict(previous_runtime_receipt)
        manifest["resume_session_receipt"] = current_runtime_receipt
    else:
        manifest["runtime_receipt"] = current_runtime_receipt
    if source_b16:
        manifest["source_b16"] = _source_b16_worker_summary(manifest["batches"])
        manifest["status"] = (
            "completed_with_source_b16_ineligible"
            if manifest["source_b16"]["ineligible_count"]
            else "completed"
        )
    else:
        manifest["status"] = "completed"
    manifest["wall_elapsed_seconds_current_process"] = (
        time.perf_counter() - overall_started_at
    )
    total_completions = sum(
        int(mode["completion_count"])
        for batch in manifest["batches"]
        for mode in batch["generation_health"].values()
    )
    total_tokens = sum(
        int(mode["generated_token_count"])
        for batch in manifest["batches"]
        for mode in batch["generation_health"].values()
    )
    generation_elapsed_seconds = sum(
        float(mode["elapsed_seconds"])
        for batch in manifest["batches"]
        for mode in batch["generation_health"].values()
    )
    stop_reason_counts = {
        reason: sum(
            int(mode["stop_reason_counts"][reason])
            for batch in manifest["batches"]
            for mode in batch["generation_health"].values()
        )
        for reason in ("im_end", "length")
    }
    parser_status_counts: dict[str, int] = {}
    for batch in manifest["batches"]:
        for mode in batch["generation_health"].values():
            for status, count in mode["parser_status_counts"].items():
                parser_status_counts[status] = parser_status_counts.get(status, 0) + int(count)
    manifest["completion_health"] = {
        "stop_reason_counts": stop_reason_counts,
        "natural_closure_count": stop_reason_counts["im_end"],
        "parser_status_counts": parser_status_counts,
    }
    manifest["throughput"] = {
        "completion_count": total_completions,
        "generated_token_count": total_tokens,
        "generation_elapsed_seconds": generation_elapsed_seconds,
        "completions_per_second": total_completions / generation_elapsed_seconds,
        "routes_per_second": total_completions / generation_elapsed_seconds,
        "generated_tokens_per_second": total_tokens / generation_elapsed_seconds,
    }
    _atomic_write_json(manifest_path, manifest)
    return manifest_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infer-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--worker-count", type=int, default=8)
    parser.add_argument("--image-batch-size", type=int, default=16)
    parser.add_argument(
        "--image-ids",
        help="Optional comma-separated physical or example IDs selected before sharding.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        help="Bound this worker for an 8-16 image smoke; omit for the full shard.",
    )
    parser.add_argument(
        "--request-seed",
        type=int,
        help="Optional vLLM request seed metadata; sample_index remains trajectory identity.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip only completed panel artifacts that pass identity/hash/count validation.",
    )
    panel_mode = parser.add_mutually_exclusive_group()
    panel_mode.add_argument(
        "--sampled-only",
        action="store_true",
        help=(
            "Collect exactly sixteen sampled trajectories per image; do not run or require "
            "greedy artifacts."
        ),
    )
    panel_mode.add_argument(
        "--source-b16",
        action="store_true",
        help=(
            "Collect one deterministic greedy raw trajectory per image and retain only its "
            "clean first-sixteen-row Source@B16 prefix as baseline evidence."
        ),
    )
    parser.add_argument(
        "--source-b16-repetition-penalty",
        type=float,
        choices=sorted(SOURCE_B16_ALLOWED_REPETITION_PENALTIES),
        default=1.0,
        help=(
            "Explicit Source@B16 repetition penalty. It must equal the inference-config "
            "value; other panel modes remain fixed at 1.0."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    image_ids = parse_image_ids(args.image_ids)
    collect_panel(
        infer_config=args.infer_config,
        output_root=args.output_root,
        worker_index=args.worker_index,
        worker_count=args.worker_count,
        image_batch_size=args.image_batch_size,
        image_ids=image_ids,
        max_images=args.max_images,
        request_seed=args.request_seed,
        resume=args.resume,
        sampled_only=args.sampled_only,
        source_b16=args.source_b16,
        source_b16_repetition_penalty=args.source_b16_repetition_penalty,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
