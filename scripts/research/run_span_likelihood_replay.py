#!/usr/bin/env python3
"""Teacher-forced raw-model span likelihood replay for the three-checkpoint sampled panel.

One full-sequence forward per rollout recovers the raw (unmodified lm-head)
chosen-token log-probabilities for every generated token.  Every parsed object
span is then sliced out of that single result, so the number of forwards equals
the number of sequences, never the number of spans.

Only the raw channel is produced.  Policy likelihood (temperature / top-p /
repetition-penalty adjusted) is deliberately absent from the output schema; the
one place a policy quantity appears is the read-only cross-validation probe
against an independent greedy artifact, and it never reaches a span row.

The rollouts replayed here were sampled at temperature 0.4 / top_p 0.95 /
repetition_penalty 1.0.  The infer configs' ``generation:`` block describes a
different, prior greedy panel and must not be used to describe these rollouts.
Teacher-forced raw likelihood does not depend on either decode policy.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SCHEMA_VERSION = "span_likelihood_replay.v1"

DEFAULT_ROLLOUT_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-29-three-checkpoint-human-refined12-max3084"
)
DEFAULT_OUTPUT_DIRNAME = "likelihood-mining-v1"
OUTPUT_JSONL_NAME = "span-likelihood.jsonl"
RECEIPT_NAME = "replay-receipt.json"

#: Checkpoint -> infer config, in the order the task states them.
CHECKPOINT_CONFIGS: dict[str, str] = {
    "sorted": "qwen3_vl_2b_desc_first_geo_sorted_step4887_human_refined12_hf_fp32.yaml",
    "random": "qwen3_vl_2b_desc_first_random_step4887_human_refined12_hf_fp32.yaml",
    "permutation": (
        "qwen3_vl_2b_desc_first_random_permutation_bundle_step4887_human_refined12_hf_fp32.yaml"
    ),
}
INFER_CONFIG_DIR = Path("configs/coordexp_swift/infer")
SHARD_INDICES = (0, 1)

#: ``teacher_forced_chosen_token_logprobs`` extends only ``input_ids`` and
#: ``attention_mask`` when it concatenates the continuation.  Any positional or
#: cache key in the processor output would silently mis-position every generated
#: token, so the key set is gated rather than assumed.
ALLOWED_NATIVE_INPUT_KEYS = frozenset(
    {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
)

#: Schema wrappers are near-deterministic under these checkpoints.  An
#: off-by-one on prompt width slides wrappers onto content positions and this
#: collapses immediately, so it is a per-sequence alignment gate.
WRAPPER_MEDIAN_LOGPROB_FLOOR = -0.5

#: Tolerance for the sliced-coordinate log-softmax against the full-sequence
#: gather.  Both are FP32 row-wise softmaxes of the same logits; only kernel
#: shape differs.
COORD_CONSISTENCY_ATOL = 1e-6

#: Independent greedy artifact used for the end-to-end plumbing cross-check.
CROSS_VALIDATION_CHECKPOINT = "sorted"
CROSS_VALIDATION_RUN_DIR = Path(
    "/data/CoordExp/outputs/coordexp_swift/infer/val200/"
    "qwen3-vl-2b-desc-first-geo-sorted-step4887-human-refined12-hf-fp32"
)
CROSS_VALIDATION_REPETITION_PENALTY = 1.10
CROSS_VALIDATION_POLICY_ATOL = 1e-4

#: Frozen output field order.  Adding, removing or renaming a key here changes
#: the artifact contract.
ROW_FIELDS: tuple[str, ...] = (
    "checkpoint",
    "example_id",
    "image_id",
    "seed",
    "object_span_id",
    "generated_order",
    "trajectory_row_count",
    "token_start",
    "token_end",
    "row_token_count",
    "row_entry_logprob",
    "description_logprobs",
    "description_token_count",
    "description_mean",
    "coord_logprobs",
    "coord_mean",
    "coord_min",
    "coord_argmin_index",
    "coord_ranks",
    "coord_entropies",
    "wrapper_logprobs",
    "wrapper_mean",
    "row_logprob_sum",
    "row_logprob_mean",
    "description",
    "coord_bins",
    "bbox",
)


class ReplayContractError(RuntimeError):
    """A verification gate failed; no partial decision-bearing score is emitted."""

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(f"{message} | context: {json.dumps(context, sort_keys=True, default=str)}")
        self.context = context


def _fail(message: str, **context: Any) -> None:
    raise ReplayContractError(message, **context)


def _load_producer_module() -> Any:
    """Load the script that produced these rollouts, to reuse its prompt path."""

    path = REPO_ROOT / "scripts" / "research" / "run_current_seeded_sampled_rollouts.py"
    spec = importlib.util.spec_from_file_location(
        "_coordexp_seeded_sampled_rollouts_producer", path
    )
    if spec is None or spec.loader is None:
        _fail("cannot load the rollout producer module", path=str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Rollout artifact loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RolloutSequence:
    checkpoint: str
    shard: int
    example_id: str
    image_id: int
    seed: int
    prompt_token_ids: tuple[int, ...]
    prompt_token_ids_sha256: str
    generated_token_ids: tuple[int, ...]
    generated_token_ids_sha256: str
    generated_text: str
    observed_image_grid_thw: tuple[int, int, int] | None
    executed_media_sha256: str
    predictions: tuple[Mapping[str, Any], ...]

    @property
    def row_id(self) -> str:
        return f"{self.example_id}:seed-{self.seed}"


def _sha256_json_list(values: Sequence[int]) -> str:
    """Match the producer's ``_sha256_json`` over an integer list."""

    return hashlib.sha256(
        json.dumps(
            [int(value) for value in values],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()


def load_checkpoint_shards(
    root: Path, checkpoint: str
) -> list[tuple[int, Mapping[str, Any]]]:
    shards: list[tuple[int, Mapping[str, Any]]] = []
    for shard in SHARD_INDICES:
        path = root / checkpoint / "sampled" / f"shard-{shard}.json"
        if not path.is_file():
            _fail("sampled shard is missing", checkpoint=checkpoint, path=str(path))
        payload = json.loads(path.read_text(encoding="utf-8"))
        shards.append((shard, payload))
    return shards


def sequences_from_shard(
    checkpoint: str, shard: int, payload: Mapping[str, Any]
) -> list[RolloutSequence]:
    rows: list[RolloutSequence] = []
    for row in payload["rollouts"]:
        prompt_ids = tuple(int(value) for value in row["prompt_token_ids"])
        generated_ids = tuple(int(value) for value in row["generated_token_ids"])
        grid = row.get("observed_image_grid_thw")
        rows.append(
            RolloutSequence(
                checkpoint=checkpoint,
                shard=shard,
                example_id=str(row["example_id"]),
                image_id=int(row["image_id"]),
                seed=int(row["seed"]),
                prompt_token_ids=prompt_ids,
                prompt_token_ids_sha256=str(row["prompt_token_ids_sha256"]),
                generated_token_ids=generated_ids,
                generated_token_ids_sha256=str(row["generated_token_ids_sha256"]),
                generated_text=str(row["generated_text"]),
                observed_image_grid_thw=(
                    None if grid is None else tuple(int(v) for v in grid)
                ),
                executed_media_sha256=str(row["executed_media_sha256"]),
                predictions=tuple(row["predictions"]["predictions"]),
            )
        )
    return rows


def verify_stored_hashes(sequence: RolloutSequence) -> None:
    prompt_sha = _sha256_json_list(sequence.prompt_token_ids)
    if prompt_sha != sequence.prompt_token_ids_sha256:
        _fail(
            "stored prompt_token_ids_sha256 does not match prompt_token_ids",
            checkpoint=sequence.checkpoint,
            example_id=sequence.example_id,
            seed=sequence.seed,
            recomputed=prompt_sha,
            stored=sequence.prompt_token_ids_sha256,
        )
    generated_sha = _sha256_json_list(sequence.generated_token_ids)
    if generated_sha != sequence.generated_token_ids_sha256:
        _fail(
            "stored generated_token_ids_sha256 does not match generated_token_ids",
            checkpoint=sequence.checkpoint,
            example_id=sequence.example_id,
            seed=sequence.seed,
            recomputed=generated_sha,
            stored=sequence.generated_token_ids_sha256,
        )


# ---------------------------------------------------------------------------
# Token trace and span partition (alignment owned by src.inference.scoring)
# ---------------------------------------------------------------------------


class TokenTextCache:
    """Per-token text is the single-token decode, matching ``_decode_tokens``."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._cache: dict[int, str] = {}

    def text(self, token_id: int) -> str:
        cached = self._cache.get(token_id)
        if cached is None:
            cached = str(self._tokenizer.decode([int(token_id)], skip_special_tokens=False))
            self._cache[token_id] = cached
        return cached


def build_token_trace(
    sequence: RolloutSequence, cache: TokenTextCache
) -> list[Any]:
    from src.inference.backend import LikelihoodPair, TokenTrace

    traces = [
        TokenTrace(
            step_index=index,
            token_id=int(token_id),
            token_text=cache.text(int(token_id)),
            likelihood=LikelihoodPair(policy_logprob=None, raw_model_logprob=None),
            is_stop=False,
            is_pad=False,
            backend="hf",
            backend_mode="generate",
            response_family="hf",
        )
        for index, token_id in enumerate(sequence.generated_token_ids)
    ]
    joined = "".join(trace.token_text for trace in traces)
    if joined != sequence.generated_text:
        _fail(
            "per-token decode concatenation does not reproduce stored generated_text",
            checkpoint=sequence.checkpoint,
            example_id=sequence.example_id,
            seed=sequence.seed,
            joined_length=len(joined),
            stored_length=len(sequence.generated_text),
        )
    return traces


@dataclass(frozen=True)
class SpanPartition:
    object_span_id: str
    generated_order: int
    token_start: int
    token_end: int
    wrapper_indices: tuple[int, int, int, int]
    description_indices: tuple[int, ...]
    coord_indices: tuple[int, int, int, int]


def partition_span(
    *,
    sequence: RolloutSequence,
    prediction: Mapping[str, Any],
    token_trace: list[Any],
) -> SpanPartition:
    from src.inference.scoring import (
        _locate_object_interval,
        _token_char_ranges,
        _trace_for_span,
    )
    from src.templates.renderer import (
        BOX_END_TOKEN,
        BOX_START_TOKEN,
        OBJECT_REF_END_TOKEN,
        OBJECT_REF_START_TOKEN,
    )

    wrapper_order = (
        OBJECT_REF_START_TOKEN,
        OBJECT_REF_END_TOKEN,
        BOX_START_TOKEN,
        BOX_END_TOKEN,
    )
    object_span_id = str(prediction.get("object_span_id"))
    where = {
        "checkpoint": sequence.checkpoint,
        "example_id": sequence.example_id,
        "seed": sequence.seed,
        "object_span_id": object_span_id,
    }

    token_start, token_end, span_char_start = _locate_object_interval(
        row_id=sequence.row_id, prediction=dict(prediction), token_trace=token_trace
    )
    token_ranges = _token_char_ranges(token_trace, token_start, token_end)

    def resolve(span: Mapping[str, Any]) -> int:
        trace = _trace_for_span(
            dict(span),
            row_id=sequence.row_id,
            token_trace=token_trace,
            token_ranges=token_ranges,
            span_char_start=span_char_start,
            object_span_id=object_span_id,
        )
        index = int(trace.step_index)
        if token_trace[index] is not trace:
            _fail("resolved token trace index is not self-consistent", index=index, **where)
        return index

    schema_spans = list(prediction.get("schema_spans") or [])
    if len(schema_spans) != len(wrapper_order):
        _fail(
            "parsed row does not carry exactly four schema wrapper spans",
            schema_span_count=len(schema_spans),
            **where,
        )
    by_text: dict[str, int] = {}
    for span in schema_spans:
        text = str(span.get("text"))
        if text in by_text:
            _fail("duplicate schema wrapper span text", wrapper=text, **where)
        by_text[text] = resolve(span)
    if set(by_text) != set(wrapper_order):
        _fail(
            "schema wrapper span texts do not match the compact object schema",
            observed=sorted(by_text),
            expected=list(wrapper_order),
            **where,
        )
    wrapper_indices = tuple(by_text[text] for text in wrapper_order)
    ref_start, ref_end, box_start, box_end = wrapper_indices

    coord_spans = sorted(
        list(prediction.get("coord_token_spans") or []),
        key=lambda item: int(item["char_start"]),
    )
    if len(coord_spans) != 4:
        _fail(
            "parsed row does not carry exactly four coordinate token spans",
            coord_span_count=len(coord_spans),
            **where,
        )
    coord_indices = tuple(resolve(span) for span in coord_spans)

    if token_start != ref_start:
        _fail(
            "object interval does not open on the object_ref_start token",
            token_start=token_start,
            ref_start=ref_start,
            **where,
        )
    if token_end != box_end + 1:
        _fail(
            "object interval does not close on the box_end token",
            token_end=token_end,
            box_end=box_end,
            **where,
        )
    if not ref_start < ref_end < box_start < box_end:
        _fail(
            "schema wrapper tokens are not in compact object schema order",
            wrapper_indices=list(wrapper_indices),
            **where,
        )
    description_indices = tuple(range(ref_start + 1, ref_end))
    if not description_indices:
        _fail("object span has no description tokens between the ref markers", **where)
    expected_coords = tuple(range(box_start + 1, box_end))
    if coord_indices != expected_coords:
        _fail(
            "coordinate tokens are not exactly the contiguous run between the box markers",
            coord_indices=list(coord_indices),
            expected=list(expected_coords),
            **where,
        )

    return SpanPartition(
        object_span_id=object_span_id,
        generated_order=int(prediction["generated_order"]),
        token_start=int(token_start),
        token_end=int(token_end),
        wrapper_indices=wrapper_indices,  # type: ignore[arg-type]
        description_indices=description_indices,
        coord_indices=coord_indices,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# GPU replay
# ---------------------------------------------------------------------------


def check_native_input_keys(native_inputs: Mapping[str, Any], **where: Any) -> tuple[str, ...]:
    keys = tuple(sorted(str(key) for key in native_inputs))
    unexpected = sorted(set(keys) - ALLOWED_NATIVE_INPUT_KEYS)
    if unexpected:
        _fail(
            "processor emitted native inputs the teacher-forced concat path does not extend; "
            "positional or cache keys would silently mis-position every generated token",
            unexpected_keys=unexpected,
            observed_keys=list(keys),
            **where,
        )
    if "input_ids" not in keys or "attention_mask" not in keys:
        _fail("processor native inputs lack input_ids/attention_mask", observed_keys=list(keys), **where)
    return keys


@dataclass(frozen=True)
class SequenceReplay:
    chosen_logprobs: Any  # torch.Tensor, CPU float32, shape (generated_steps,)
    coord_positions: tuple[int, ...]
    coord_ranks: dict[int, int]
    coord_entropies: dict[int, float]
    coord_consistency_max_abs_diff: float
    vocab_size: int


def _forward_prediction_logits(
    *,
    model: Any,
    native_prompt_inputs: Mapping[str, Any],
    generated_token_ids: Sequence[int],
) -> Any:
    """Mirror ``teacher_forced_chosen_token_logprobs`` up to, but not past, the gather.

    The canonical helper gathers and discards the logits, so it physically
    cannot yield coordinate ranks or entropies.  This reproduces its exact
    concat/slice, keeps the logits, and is checked bitwise against the canonical
    helper on oracle sequences.
    """

    import torch

    from src.inference.hf_backend import (
        _require_rank_three_tensor,
        _require_rank_two_tensor,
    )

    input_ids = _require_rank_two_tensor(
        native_prompt_inputs.get("input_ids"), field="input_ids"
    )
    if int(input_ids.shape[0]) != 1:
        _fail("teacher-forced replay accepts exactly one prompt", shape=list(input_ids.shape))
    generated = torch.tensor(
        [int(token_id) for token_id in generated_token_ids],
        dtype=input_ids.dtype,
        device=input_ids.device,
    ).unsqueeze(0)
    if int(generated.shape[1]) == 0:
        _fail("teacher-forced replay requires generated token ids")
    forward_inputs = dict(native_prompt_inputs)
    forward_inputs["input_ids"] = torch.cat((input_ids, generated), dim=1)
    attention_mask = forward_inputs.get("attention_mask")
    if isinstance(attention_mask, torch.Tensor):
        extension = torch.ones(
            (1, int(generated.shape[1])),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        forward_inputs["attention_mask"] = torch.cat((attention_mask, extension), dim=1)
    with torch.inference_mode():
        outputs = model(**forward_inputs, return_dict=True, use_cache=False)
    logits = _require_rank_three_tensor(getattr(outputs, "logits", None), field="logits")
    prompt_width = int(input_ids.shape[1])
    prediction_logits = logits[
        :, prompt_width - 1 : prompt_width - 1 + int(generated.shape[1]), :
    ]
    if int(prediction_logits.shape[1]) != int(generated.shape[1]):
        _fail(
            "teacher-forced logits do not cover every generated token",
            generated_steps=int(generated.shape[1]),
            logit_steps=int(prediction_logits.shape[1]),
        )
    return prediction_logits, generated


def replay_sequence(
    *,
    model: Any,
    native_prompt_inputs: Mapping[str, Any],
    generated_token_ids: Sequence[int],
    coord_positions: Sequence[int],
) -> SequenceReplay:
    import torch
    from torch.nn import functional as F

    prediction_logits, generated = _forward_prediction_logits(
        model=model,
        native_prompt_inputs=native_prompt_inputs,
        generated_token_ids=generated_token_ids,
    )
    vocab_size = int(prediction_logits.shape[-1])
    # Identical op sequence to ``teacher_forced_chosen_token_logprobs``.
    chosen = (
        F.log_softmax(prediction_logits.float(), dim=-1)
        .gather(2, generated.unsqueeze(-1))
        .squeeze(0)
        .squeeze(-1)
    )
    chosen_cpu = chosen.detach().to(device="cpu", dtype=torch.float32).clone()
    del chosen

    ranks: dict[int, int] = {}
    entropies: dict[int, float] = {}
    max_abs_diff = 0.0
    positions = tuple(int(position) for position in coord_positions)
    if positions:
        index = torch.tensor(positions, dtype=torch.long, device=prediction_logits.device)
        coord_logits = prediction_logits[0].index_select(0, index).float()
        coord_ids = generated[0].index_select(0, index).unsqueeze(1)
        coord_logp = F.log_softmax(coord_logits, dim=-1)
        chosen_logp = coord_logp.gather(1, coord_ids).squeeze(1)
        chosen_logit = coord_logits.gather(1, coord_ids)
        rank_values = (coord_logits > chosen_logit).sum(dim=1)
        entropy_values = -(coord_logp.exp() * coord_logp).sum(dim=1)
        reference = chosen_cpu.index_select(0, index.to("cpu"))
        max_abs_diff = float(
            (chosen_logp.detach().to(device="cpu", dtype=torch.float32) - reference)
            .abs()
            .max()
            .item()
        )
        rank_list = [int(value) for value in rank_values.tolist()]
        entropy_list = [float(value) for value in entropy_values.tolist()]
        ranks = dict(zip(positions, rank_list, strict=True))
        entropies = dict(zip(positions, entropy_list, strict=True))
        del coord_logits, coord_logp, chosen_logp, chosen_logit, rank_values, entropy_values
    del prediction_logits, generated

    if max_abs_diff > COORD_CONSISTENCY_ATOL:
        _fail(
            "sliced coordinate log-softmax disagrees with the full-sequence gather",
            max_abs_diff=max_abs_diff,
            atol=COORD_CONSISTENCY_ATOL,
        )
    return SequenceReplay(
        chosen_logprobs=chosen_cpu,
        coord_positions=positions,
        coord_ranks=ranks,
        coord_entropies=entropies,
        coord_consistency_max_abs_diff=max_abs_diff,
        vocab_size=vocab_size,
    )


# ---------------------------------------------------------------------------
# Row emission
# ---------------------------------------------------------------------------


def _finite(values: Iterable[float], **where: Any) -> list[float]:
    out = [float(value) for value in values]
    for value in out:
        if not math.isfinite(value):
            _fail("replayed logprob is not finite", value=value, **where)
        if value > 0.0:
            _fail("replayed logprob is not a natural-log probability", value=value, **where)
    return out


def build_span_row(
    *,
    sequence: RolloutSequence,
    prediction: Mapping[str, Any],
    partition: SpanPartition,
    replay: SequenceReplay,
    trajectory_row_count: int,
) -> dict[str, Any]:
    where = {
        "checkpoint": sequence.checkpoint,
        "example_id": sequence.example_id,
        "seed": sequence.seed,
        "object_span_id": partition.object_span_id,
    }
    logprobs = replay.chosen_logprobs
    if int(logprobs.shape[0]) != len(sequence.generated_token_ids):
        _fail(
            "replayed logprob tensor length does not match generated_token_ids",
            logprob_steps=int(logprobs.shape[0]),
            generated_steps=len(sequence.generated_token_ids),
            **where,
        )

    def at(index: int) -> float:
        return float(logprobs[index].item())

    wrapper_logprobs = _finite((at(i) for i in partition.wrapper_indices), **where)
    description_logprobs = _finite((at(i) for i in partition.description_indices), **where)
    coord_logprobs = _finite((at(i) for i in partition.coord_indices), **where)
    row_logprobs = _finite(
        (at(i) for i in range(partition.token_start, partition.token_end)), **where
    )

    coord_ranks: list[int] = []
    coord_entropies: list[float] = []
    for index in partition.coord_indices:
        if index not in replay.coord_ranks or index not in replay.coord_entropies:
            _fail("coordinate position missing from the replay distribution slice", position=index, **where)
        coord_ranks.append(int(replay.coord_ranks[index]))
        entropy = float(replay.coord_entropies[index])
        if not math.isfinite(entropy) or entropy < 0.0:
            _fail("coordinate entropy is not a finite non-negative value", entropy=entropy, **where)
        coord_entropies.append(entropy)

    if wrapper_logprobs[0] != at(partition.token_start):
        _fail("row entry logprob does not match the opening object_ref_start token", **where)
    coord_min = min(coord_logprobs)
    row = {
        "checkpoint": sequence.checkpoint,
        "example_id": sequence.example_id,
        "image_id": int(sequence.image_id),
        "seed": int(sequence.seed),
        "object_span_id": partition.object_span_id,
        "generated_order": int(partition.generated_order),
        "trajectory_row_count": int(trajectory_row_count),
        "token_start": int(partition.token_start),
        "token_end": int(partition.token_end),
        "row_token_count": int(partition.token_end - partition.token_start),
        "row_entry_logprob": wrapper_logprobs[0],
        "description_logprobs": description_logprobs,
        "description_token_count": len(description_logprobs),
        "description_mean": sum(description_logprobs) / len(description_logprobs),
        "coord_logprobs": coord_logprobs,
        "coord_mean": sum(coord_logprobs) / 4.0,
        "coord_min": coord_min,
        "coord_argmin_index": int(coord_logprobs.index(coord_min)),
        "coord_ranks": coord_ranks,
        "coord_entropies": coord_entropies,
        "wrapper_logprobs": wrapper_logprobs,
        "wrapper_mean": sum(wrapper_logprobs) / 4.0,
        "row_logprob_sum": sum(row_logprobs),
        "row_logprob_mean": sum(row_logprobs) / len(row_logprobs),
        "description": str(prediction["description"]),
        "coord_bins": [int(value) for value in prediction["coord_bins"]],
        "bbox": [float(value) for value in prediction["bbox"]],
    }
    if tuple(row) != ROW_FIELDS:
        _fail("emitted row field set drifted from the frozen schema", observed=list(row), **where)
    return row


# ---------------------------------------------------------------------------
# Cross-validation against the independent greedy artifact
# ---------------------------------------------------------------------------


def load_greedy_traces(run_dir: Path) -> dict[str, list[Mapping[str, Any]]]:
    path = run_dir / "pred_token_trace.jsonl"
    if not path.is_file():
        _fail("greedy cross-validation token trace is missing", path=str(path))
    rows: dict[str, list[Mapping[str, Any]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("trace_type") != "generated_token":
                continue
            rows.setdefault(str(record["row_id"]), []).append(record)
    for row_id, records in rows.items():
        records.sort(key=lambda item: int(item["generated_step_index"]))
        expected = list(range(len(records)))
        if [int(item["generated_step_index"]) for item in records] != expected:
            _fail("greedy trace step indices are not a dense 0..n-1 range", row_id=row_id)
    return rows


def _greedy_prompt_shas(run_dir: Path) -> dict[str, str]:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    return {
        str(entry["row_id"]): str(entry["backend_executed_prompt_token_ids_sha256"])
        for entry in manifest["prompt_trace"]
    }


def _greedy_media_shas(run_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with (run_dir / "image_plan.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            out[str(record["example_id"])] = str(record["executed_media_sha256"])
    return out


def _policy_logprobs_with_repetition_penalty(
    *,
    prediction_logits: Any,
    prompt_token_ids: Sequence[int],
    generated_token_ids: Sequence[int],
    penalty: float,
) -> list[float]:
    """Best-effort reconstruction of the greedy run's stored policy logprob.

    Uses the same ``RepetitionPenaltyLogitsProcessor`` class HF ``generate``
    installs, fed the unpadded running context.  The production run decoded in
    batches of four with left padding, so its processor also saw pad tokens for
    shorter prompts; that is an acknowledged, unresolved confound.
    """

    import torch
    from torch.nn import functional as F
    from transformers import RepetitionPenaltyLogitsProcessor

    processor = RepetitionPenaltyLogitsProcessor(penalty=float(penalty))
    device = prediction_logits.device
    context = [int(value) for value in prompt_token_ids]
    out: list[float] = []
    for step, token_id in enumerate(int(value) for value in generated_token_ids):
        step_logits = prediction_logits[:, step, :].float().clone()
        context_ids = torch.tensor([context], dtype=torch.long, device=device)
        processed = processor(context_ids, step_logits)
        out.append(float(F.log_softmax(processed, dim=-1)[0, token_id].item()))
        context.append(token_id)
    return out


def run_cross_validation(
    *,
    session: Any,
    native_by_example: Mapping[str, Mapping[str, Any]],
    executed_prompt_by_example: Mapping[str, tuple[int, ...]],
    media_sha_by_example: Mapping[str, str],
    run_dir: Path,
    row_limit: int,
    attempt_policy_rows: int,
) -> dict[str, Any]:
    import torch

    traces = load_greedy_traces(run_dir)
    prompt_shas = _greedy_prompt_shas(run_dir)
    media_shas = _greedy_media_shas(run_dir)
    from src.inference.backend import token_ids_sha256

    row_ids = sorted(row_id for row_id in traces if row_id in native_by_example)
    if len(row_ids) < row_limit:
        _fail(
            "greedy cross-validation has fewer replayable rows than required",
            available=len(row_ids),
            required=row_limit,
        )
    selected = row_ids[:row_limit] if row_limit > 0 else row_ids

    per_row: list[dict[str, Any]] = []
    policy_attempts = 0
    for row_id in selected:
        records = traces[row_id]
        kept = [item for item in records if not bool(item["is_pad"])]
        trace_ids = [int(item["token_id"]) for item in kept]
        native = native_by_example[row_id]

        prompt_sha = token_ids_sha256(executed_prompt_by_example[row_id])
        prompt_parity = prompt_sha == prompt_shas.get(row_id)
        media_parity = media_sha_by_example[row_id] == media_shas.get(row_id)

        prediction_logits, generated = _forward_prediction_logits(
            model=session._model,
            native_prompt_inputs=native,
            generated_token_ids=trace_ids,
        )
        length_match = int(prediction_logits.shape[1]) == len(trace_ids)
        chosen_ids_match = [
            int(value) for value in generated[0].tolist()
        ] == trace_ids
        argmax_ids = [int(value) for value in prediction_logits[0].argmax(dim=-1).tolist()]
        mismatches = [
            {"step": step, "trace_token_id": trace_ids[step], "raw_argmax_token_id": argmax_ids[step]}
            for step in range(len(trace_ids))
            if argmax_ids[step] != trace_ids[step]
        ]

        entry: dict[str, Any] = {
            "row_id": row_id,
            "trace_token_count": len(trace_ids),
            "padded_trace_rows_dropped": len(records) - len(kept),
            "prompt_token_parity": bool(prompt_parity),
            "executed_prompt_token_ids_sha256": prompt_sha,
            "greedy_manifest_prompt_token_ids_sha256": prompt_shas.get(row_id),
            "executed_media_sha256": media_sha_by_example[row_id],
            "greedy_image_plan_executed_media_sha256": media_shas.get(row_id),
            "image_identity_parity": bool(media_parity),
            "logprob_tensor_length_matches_trace": bool(length_match),
            "teacher_forced_chosen_ids_equal_trace_token_ids": bool(chosen_ids_match),
            "raw_argmax_agreement_count": len(trace_ids) - len(mismatches),
            "raw_argmax_agreement_rate": (
                (len(trace_ids) - len(mismatches)) / len(trace_ids) if trace_ids else None
            ),
            "raw_argmax_mismatches": mismatches[:20],
            "raw_argmax_mismatch_count": len(mismatches),
        }

        if policy_attempts < attempt_policy_rows:
            policy_attempts += 1
            stored = [item["logprob"] for item in kept]
            try:
                reproduced = _policy_logprobs_with_repetition_penalty(
                    prediction_logits=prediction_logits,
                    prompt_token_ids=executed_prompt_by_example[row_id],
                    generated_token_ids=trace_ids,
                    penalty=CROSS_VALIDATION_REPETITION_PENALTY,
                )
                diffs = [
                    abs(float(a) - float(b))
                    for a, b in zip(stored, reproduced, strict=True)
                    if a is not None
                ]
                entry["policy_logprob_reproduction"] = {
                    "attempted": True,
                    "repetition_penalty": CROSS_VALIDATION_REPETITION_PENALTY,
                    "compared_steps": len(diffs),
                    "max_abs_diff": max(diffs) if diffs else None,
                    "median_abs_diff": statistics.median(diffs) if diffs else None,
                    "within_atol_fraction": (
                        sum(1 for d in diffs if d <= CROSS_VALIDATION_POLICY_ATOL) / len(diffs)
                        if diffs
                        else None
                    ),
                    "atol": CROSS_VALIDATION_POLICY_ATOL,
                }
            except Exception as exc:  # noqa: BLE001
                entry["policy_logprob_reproduction"] = {
                    "attempted": True,
                    "error": f"{type(exc).__name__}: {exc}"[:400],
                }
        del prediction_logits, generated
        torch.cuda.empty_cache()
        per_row.append(entry)

    gates = {
        "prompt_token_parity": all(item["prompt_token_parity"] for item in per_row),
        "image_identity_parity": all(item["image_identity_parity"] for item in per_row),
        "logprob_tensor_length_matches_trace": all(
            item["logprob_tensor_length_matches_trace"] for item in per_row
        ),
        "teacher_forced_chosen_ids_equal_trace_token_ids": all(
            item["teacher_forced_chosen_ids_equal_trace_token_ids"] for item in per_row
        ),
    }
    total_steps = sum(item["trace_token_count"] for item in per_row)
    total_agree = sum(item["raw_argmax_agreement_count"] for item in per_row)
    return {
        "status": "passed" if all(gates.values()) else "failed",
        "artifact": str(run_dir / "pred_token_trace.jsonl"),
        "artifact_decode": "greedy, repetition_penalty 1.10, raw_model_logprob disabled",
        "replayed_row_count": len(per_row),
        "gates": gates,
        "gate_c_note": (
            "Acceptance 3(c) as literally written is a construction invariant: teacher forcing "
            "feeds the trace token ids, so the chosen token is the trace token by definition. It "
            "is asserted anyway. The substantive plumbing evidence is raw_argmax_agreement_rate "
            "below: the greedy run selected argmax of the repetition-penalised logits, so raw "
            "argmax should agree at nearly every position and would collapse under any prompt "
            "width, position, or slice error."
        ),
        "raw_argmax_agreement_rate_overall": (total_agree / total_steps) if total_steps else None,
        "raw_argmax_agreement_steps": total_agree,
        "raw_argmax_total_steps": total_steps,
        "policy_reproduction_confound": (
            "The greedy production run decoded with batch_size 4 and left padding, so HF's "
            "RepetitionPenaltyLogitsProcessor saw the padded input_ids (including pad tokens) at "
            "every step. This replay is batch-of-one and unpadded, so exact policy reproduction "
            "is not guaranteed; residual disagreement is reported, not tuned away."
        ),
        "rows": per_row,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _resolved_config(config_dir: Path, filename: str) -> Any:
    from src.config.inference import load_infer_config

    return load_infer_config((config_dir / filename).resolve(strict=True))


def run_checkpoint(
    *,
    checkpoint: str,
    rollout_root: Path,
    config_dir: Path,
    producer: Any,
    limit_sequences: int | None,
    cross_validation_rows: int,
    policy_rows: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    from src.config.fingerprint import sha256_json
    from src.data import load_raw_examples
    from src.inference.backend import open_backend_session
    from src.inference.hf_backend import (
        _model_vocabulary_size,
        teacher_forced_chosen_token_logprobs,
    )
    from src.inference.runtime import assemble_frontend

    started = time.perf_counter()
    resolved = _resolved_config(config_dir, CHECKPOINT_CONFIGS[checkpoint])
    config = resolved.config

    shards = load_checkpoint_shards(rollout_root, checkpoint)
    sequences_by_shard: dict[int, list[RolloutSequence]] = {}
    shard_config_meta: list[dict[str, Any]] = []
    for shard, payload in shards:
        sequences_by_shard[shard] = sequences_from_shard(checkpoint, shard, payload)
        shard_config_meta.append(
            {
                "shard": shard,
                "rollout_count": int(payload["rollout_count"]),
                "producer_resolved_fingerprint": payload["config"]["resolved_fingerprint"],
                "producer_infer_config_path": payload["config"]["infer_config_path"],
                "decode_mode": payload["config"]["decode_mode"],
                "temperature": payload["config"]["temperature"],
                "top_p": payload["config"]["top_p"],
                "repetition_penalty": payload["config"]["repetition_penalty"],
                "max_new_tokens": payload["config"]["max_new_tokens"],
            }
        )

    raw_examples_all = list(load_raw_examples(config.data.input_jsonl))
    frontend = assemble_frontend(
        config,
        generation_config_fingerprint=sha256_json(config.generation.model_dump(mode="json")),
    )
    tokenizer_cache = TokenTextCache(frontend.qwen.tokenizer)

    rows: list[dict[str, Any]] = []
    gate_counters = {
        "sequences": 0,
        "spans": 0,
        "generated_tokens": 0,
        "stored_hash_checks": 0,
        "prompt_parity_checks": 0,
        "image_grid_checks": 0,
        "media_sha_checks": 0,
        "logprob_length_checks": 0,
        "span_interval_checks": 0,
        "single_token_span_checks": 0,
    }
    wrapper_medians: list[float] = []
    native_key_sets: set[tuple[str, ...]] = set()
    coord_consistency_max = 0.0
    determinism: dict[str, Any] = {"status": "not_run"}
    oracle: dict[str, Any] = {"status": "not_run"}
    cross_validation: dict[str, Any] = {"status": "not_applicable"}
    vocab_sizes: set[int] = set()
    receipt_artifact: dict[str, Any] = {}

    session_cm = open_backend_session(frontend.launch)
    with session_cm as session:
        receipt_artifact = session.receipt.to_artifact_dict()
        model_vocab = _model_vocabulary_size(session._model, session._tokenizer)
        cv_native: dict[str, Mapping[str, Any]] = {}
        cv_prompt: dict[str, tuple[int, ...]] = {}
        cv_media: dict[str, str] = {}

        emitted = 0
        for shard, sequences in sorted(sequences_by_shard.items()):
            example_ids = []
            for sequence in sequences:
                if sequence.example_id not in example_ids:
                    example_ids.append(sequence.example_id)
            selected_examples = [
                example
                for example in raw_examples_all
                if str(example.example_id) in set(example_ids)
            ]
            if len(selected_examples) != len(example_ids):
                _fail(
                    "shard example ids are not all present in the config input jsonl",
                    checkpoint=checkpoint,
                    shard=shard,
                    shard_examples=len(example_ids),
                    resolved_examples=len(selected_examples),
                )
            requests, _prompt_meta = producer._build_requests(
                config, frontend, selected_examples
            )
            request_by_example = {
                str(example.example_id): request
                for example, request in zip(selected_examples, requests, strict=True)
            }

            by_example: dict[str, list[RolloutSequence]] = {}
            for sequence in sequences:
                by_example.setdefault(sequence.example_id, []).append(sequence)

            for example_id in example_ids:
                request = request_by_example[example_id]
                (
                    native_inputs,
                    executed_ids,
                    observed_grids,
                    media_sha,
                ) = session._materialize_native_inputs((request,))
                native_key_sets.add(
                    check_native_input_keys(
                        native_inputs, checkpoint=checkpoint, example_id=example_id
                    )
                )
                cv_native[example_id] = native_inputs
                cv_prompt[example_id] = tuple(executed_ids[0])
                cv_media[example_id] = str(media_sha[0])

                for sequence in sorted(by_example[example_id], key=lambda item: item.seed):
                    if limit_sequences is not None and emitted >= limit_sequences:
                        break
                    where = {
                        "checkpoint": checkpoint,
                        "example_id": sequence.example_id,
                        "seed": sequence.seed,
                    }
                    verify_stored_hashes(sequence)
                    gate_counters["stored_hash_checks"] += 1

                    if tuple(executed_ids[0]) != sequence.prompt_token_ids:
                        _fail(
                            "executed prompt tokens differ from the stored rollout prompt",
                            executed_count=len(executed_ids[0]),
                            stored_count=len(sequence.prompt_token_ids),
                            **where,
                        )
                    gate_counters["prompt_parity_checks"] += 1
                    observed_grid = (
                        None if observed_grids[0] is None else tuple(int(v) for v in observed_grids[0])
                    )
                    if observed_grid != sequence.observed_image_grid_thw:
                        _fail(
                            "observed image grid differs from the stored rollout grid",
                            observed=observed_grid,
                            stored=sequence.observed_image_grid_thw,
                            **where,
                        )
                    gate_counters["image_grid_checks"] += 1
                    if str(media_sha[0]) != sequence.executed_media_sha256:
                        _fail(
                            "executed media sha differs from the stored rollout media sha",
                            observed=str(media_sha[0]),
                            stored=sequence.executed_media_sha256,
                            **where,
                        )
                    gate_counters["media_sha_checks"] += 1

                    token_trace = build_token_trace(sequence, tokenizer_cache)
                    partitions = [
                        partition_span(
                            sequence=sequence, prediction=prediction, token_trace=token_trace
                        )
                        for prediction in sequence.predictions
                    ]
                    gate_counters["span_interval_checks"] += len(partitions)
                    gate_counters["single_token_span_checks"] += 8 * len(partitions)
                    coord_positions = sorted(
                        {index for part in partitions for index in part.coord_indices}
                    )

                    replay = replay_sequence(
                        model=session._model,
                        native_prompt_inputs=native_inputs,
                        generated_token_ids=sequence.generated_token_ids,
                        coord_positions=coord_positions,
                    )
                    vocab_sizes.add(replay.vocab_size)
                    coord_consistency_max = max(
                        coord_consistency_max, replay.coord_consistency_max_abs_diff
                    )
                    if int(replay.chosen_logprobs.shape[0]) != len(sequence.generated_token_ids):
                        _fail(
                            "replayed logprob tensor length does not match generated_token_ids",
                            logprob_steps=int(replay.chosen_logprobs.shape[0]),
                            generated_steps=len(sequence.generated_token_ids),
                            **where,
                        )
                    gate_counters["logprob_length_checks"] += 1

                    if determinism["status"] == "not_run":
                        second = replay_sequence(
                            model=session._model,
                            native_prompt_inputs=native_inputs,
                            generated_token_ids=sequence.generated_token_ids,
                            coord_positions=coord_positions,
                        )
                        identical = bool(
                            torch.equal(replay.chosen_logprobs, second.chosen_logprobs)
                        )
                        determinism = {
                            "status": "passed" if identical else "failed",
                            "bitwise_identical": identical,
                            "example_id": sequence.example_id,
                            "seed": sequence.seed,
                            "generated_steps": len(sequence.generated_token_ids),
                        }
                        if not identical:
                            _fail("same-sequence replay is not bitwise identical", **where)
                        del second

                    if oracle["status"] == "not_run":
                        reference = teacher_forced_chosen_token_logprobs(
                            model=session._model,
                            native_prompt_inputs=native_inputs,
                            generated_token_ids=sequence.generated_token_ids,
                        )
                        reference_cpu = reference.detach().to(
                            device="cpu", dtype=torch.float32
                        )
                        identical = bool(torch.equal(replay.chosen_logprobs, reference_cpu))
                        oracle = {
                            "status": "passed" if identical else "failed",
                            "bitwise_identical_to_canonical_helper": identical,
                            "canonical_helper": (
                                "src.inference.hf_backend."
                                "teacher_forced_chosen_token_logprobs"
                            ),
                            "example_id": sequence.example_id,
                            "seed": sequence.seed,
                            "max_abs_diff": float(
                                (replay.chosen_logprobs - reference_cpu).abs().max().item()
                            ),
                        }
                        if not identical:
                            _fail(
                                "local replay slice disagrees with the canonical helper",
                                max_abs_diff=oracle["max_abs_diff"],
                                **where,
                            )
                        del reference, reference_cpu

                    trajectory_row_count = len(sequence.predictions)
                    sequence_wrappers: list[float] = []
                    for prediction, partition in zip(
                        sequence.predictions, partitions, strict=True
                    ):
                        row = build_span_row(
                            sequence=sequence,
                            prediction=prediction,
                            partition=partition,
                            replay=replay,
                            trajectory_row_count=trajectory_row_count,
                        )
                        sequence_wrappers.extend(row["wrapper_logprobs"])
                        rows.append(row)
                    if sequence_wrappers:
                        median = statistics.median(sequence_wrappers)
                        wrapper_medians.append(median)
                        if median < WRAPPER_MEDIAN_LOGPROB_FLOOR:
                            _fail(
                                "median schema-wrapper logprob is implausibly low; the generated-"
                                "token slice is probably misaligned",
                                wrapper_median=median,
                                floor=WRAPPER_MEDIAN_LOGPROB_FLOOR,
                                **where,
                            )
                    gate_counters["sequences"] += 1
                    gate_counters["spans"] += trajectory_row_count
                    gate_counters["generated_tokens"] += len(sequence.generated_token_ids)
                    emitted += 1
                    del replay
                if limit_sequences is not None and emitted >= limit_sequences:
                    break
            if limit_sequences is not None and emitted >= limit_sequences:
                break

        if checkpoint == CROSS_VALIDATION_CHECKPOINT and cross_validation_rows > 0:
            cross_validation = run_cross_validation(
                session=session,
                native_by_example=cv_native,
                executed_prompt_by_example=cv_prompt,
                media_sha_by_example=cv_media,
                run_dir=CROSS_VALIDATION_RUN_DIR,
                row_limit=cross_validation_rows,
                attempt_policy_rows=policy_rows,
            )

        cv_native.clear()

    del frontend
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    summary = {
        "checkpoint": checkpoint,
        "infer_config_path": str((config_dir / CHECKPOINT_CONFIGS[checkpoint]).resolve()),
        "resolved_config_fingerprint": resolved.fingerprint,
        "generation_config_fingerprint": sha256_json(config.generation.model_dump(mode="json")),
        "infer_config_generation_block_describes_a_prior_greedy_panel": (
            config.generation.model_dump(mode="json")
        ),
        "replayed_rollout_decode": {
            "note": (
                "These rollouts were sampled at temperature 0.4 / top_p 0.95 / "
                "repetition_penalty 1.0 by run_current_seeded_sampled_rollouts.py. Raw "
                "teacher-forced likelihood does not depend on that decode policy."
            ),
            "shards": shard_config_meta,
        },
        "sequence_count": gate_counters["sequences"],
        "span_count": gate_counters["spans"],
        "generated_token_count": gate_counters["generated_tokens"],
        "gates": {
            "stored_token_id_sha256_recomputed": gate_counters["stored_hash_checks"],
            "executed_prompt_token_parity": gate_counters["prompt_parity_checks"],
            "observed_image_grid_parity": gate_counters["image_grid_checks"],
            "executed_media_sha256_parity": gate_counters["media_sha_checks"],
            "logprob_length_equals_generated_token_count": gate_counters["logprob_length_checks"],
            "contiguous_object_intervals_resolved": gate_counters["span_interval_checks"],
            "single_token_schema_and_coord_spans_resolved": gate_counters[
                "single_token_span_checks"
            ],
            "all_gates_passed": True,
        },
        "native_input_key_sets": sorted(list(keys) for keys in native_key_sets),
        "wrapper_median_logprob": {
            "floor": WRAPPER_MEDIAN_LOGPROB_FLOOR,
            "sequences": len(wrapper_medians),
            "min": min(wrapper_medians) if wrapper_medians else None,
            "median": statistics.median(wrapper_medians) if wrapper_medians else None,
            "max": max(wrapper_medians) if wrapper_medians else None,
        },
        "coord_slice_vs_full_gather_max_abs_diff": coord_consistency_max,
        "logits_vocab_dim": sorted(vocab_sizes),
        "model_vocabulary_size": model_vocab,
        "determinism_check": determinism,
        "canonical_helper_oracle_check": oracle,
        "cross_validation": cross_validation,
        "backend_session_receipt": receipt_artifact,
        "wall_seconds": time.perf_counter() - started,
    }
    return rows, summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-root", type=Path, default=DEFAULT_ROLLOUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--checkpoints",
        default=",".join(CHECKPOINT_CONFIGS),
        help="Comma-separated checkpoints to replay.",
    )
    parser.add_argument("--config-dir", type=Path, default=REPO_ROOT / INFER_CONFIG_DIR)
    parser.add_argument(
        "--limit-sequences",
        type=int,
        default=None,
        help="Smoke only: replay at most N sequences per checkpoint.",
    )
    parser.add_argument("--cross-validation-rows", type=int, default=12)
    parser.add_argument(
        "--policy-reproduction-rows",
        type=int,
        default=1,
        help="Cross-validation rows on which to attempt policy-logprob reproduction.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    checkpoints = [piece.strip() for piece in str(args.checkpoints).split(",") if piece.strip()]
    unknown = sorted(set(checkpoints) - set(CHECKPOINT_CONFIGS))
    if unknown:
        parser.error(f"unknown checkpoints: {unknown}")

    output_dir = args.output_dir or (args.rollout_root / DEFAULT_OUTPUT_DIRNAME)
    jsonl_path = output_dir / OUTPUT_JSONL_NAME
    receipt_path = output_dir / RECEIPT_NAME
    if jsonl_path.exists() and not args.force:
        raise ReplayContractError(
            "refusing to overwrite an existing span-likelihood artifact; pass --force",
            path=str(jsonl_path),
        )

    import torch

    if not torch.cuda.is_available():
        _fail("CUDA is required for the replay stage")

    producer = _load_producer_module()
    started = time.perf_counter()
    all_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for checkpoint in checkpoints:
        rows, summary = run_checkpoint(
            checkpoint=checkpoint,
            rollout_root=args.rollout_root,
            config_dir=args.config_dir,
            producer=producer,
            limit_sequences=args.limit_sequences,
            cross_validation_rows=args.cross_validation_rows,
            policy_rows=args.policy_reproduction_rows,
        )
        all_rows.extend(rows)
        summaries[checkpoint] = summary
        print(
            f"[{checkpoint}] sequences={summary['sequence_count']} "
            f"spans={summary['span_count']} tokens={summary['generated_token_count']} "
            f"wall={summary['wall_seconds']:.1f}s",
            flush=True,
        )

    all_rows.sort(
        key=lambda row: (
            row["checkpoint"],
            row["example_id"],
            row["seed"],
            row["generated_order"],
        )
    )
    seen_keys: set[tuple[str, str, int, int]] = set()
    for row in all_rows:
        key = (row["checkpoint"], row["example_id"], row["seed"], row["generated_order"])
        if key in seen_keys:
            _fail("duplicate span row key", key=list(key))
        seen_keys.add(key)

    output_dir.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")

    wall = time.perf_counter() - started
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_unix": time.time(),
        "artifact": {
            "path": str(jsonl_path),
            "row_count": len(all_rows),
            "row_fields": list(ROW_FIELDS),
            "sort_key": ["checkpoint", "example_id", "seed", "generated_order"],
            "likelihood_channel": "raw",
            "likelihood_definition": "fp32_log_softmax_unmodified_lm_head_logits",
            "policy_likelihood_present": False,
        },
        "rollout_root": str(args.rollout_root),
        "device": {
            "requested_visible_devices": _visible_devices(),
            "torch_device": "cuda",
            "device_name": torch.cuda.get_device_name(0),
            "device_count": torch.cuda.device_count(),
        },
        "dtype": {
            "model_dtype": "fp32",
            "logprob_dtype": "float32",
            "note": "log_softmax is applied to .float() logits, matching the canonical helper.",
        },
        "replay_design": {
            "forwards_per_sequence": 1,
            "why_not_the_canonical_helper_in_bulk": (
                "src.inference.hf_backend.teacher_forced_chosen_token_logprobs gathers and then "
                "discards the logits, so it cannot yield coord_ranks or coord_entropies. The local "
                "slice reproduces its exact concat/slice/log_softmax op sequence and is checked "
                "bitwise against it per checkpoint (canonical_helper_oracle_check)."
            ),
            "coord_rank_definition": (
                "0-based strict-greater rank, (logits > chosen_logit).sum(); same tie rule as "
                "HFChosenTokenEvidence.candidate_vocab_rank, which is the 1-based form."
            ),
            "coord_entropy_definition": (
                "-(p * log p).sum() in nats over the full raw predictive distribution at that "
                "position; support is the lm-head output dimension, including any padded rows."
            ),
        },
        "per_checkpoint": summaries,
        "totals": {
            "sequences": sum(s["sequence_count"] for s in summaries.values()),
            "spans": sum(s["span_count"] for s in summaries.values()),
            "generated_tokens": sum(s["generated_token_count"] for s in summaries.values()),
        },
        "wall_seconds": wall,
    }
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {jsonl_path} ({len(all_rows)} rows) and {receipt_path} in {wall:.1f}s")
    return 0


def _visible_devices() -> str | None:
    import os

    return os.environ.get("CUDA_VISIBLE_DEVICES")


if __name__ == "__main__":
    raise SystemExit(main())
