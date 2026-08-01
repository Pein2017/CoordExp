#!/usr/bin/env python3
"""Replay forced-continuation prefixes and score bbox coordinate likelihood.

The score matches the 2026-07-29 sampled-span likelihood probe: the raw-model
FP32 teacher-forced log-probability mean over the four coordinate tokens.  The
rendering confidence is a checkpoint-specific percentile against the frozen
K=16 union-cluster medoid distribution used by the earlier coordinate-
confidence visualization.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import time
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence, cast

from scripts.research.run_current_seeded_sampled_rollouts import physical_image_id
from scripts.research.run_local_branch_causal_value import (
    hash_prefix_token_ids,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-30-iterative-forced-continue-exact-native/runs-exact-native"
)
DEFAULT_REFERENCE = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-29-three-checkpoint-human-refined12-max3084/likelihood-mining-v1/"
    "cluster-confidence.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-30-iterative-forced-continue-exact-native/"
    "post-native-prefix-coordinate-confidence-six-arms"
)
CHECKPOINTS = ("sorted", "random", "permutation")
RP_ARTIFACTS = (("rp1.0", "rp1p00.json"), ("rp1.1", "rp1p10.json"))
MAX_POST_NATIVE_PREDICTIONS = 60


class CoordinateReplayError(RuntimeError):
    """A source/replay identity invariant failed."""


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CoordinateReplayError(f"expected object payload: {path}")
    return payload


def _find_subsequence(
    haystack: Sequence[int], needle: Sequence[int], start: int = 0
) -> int:
    if not needle:
        raise CoordinateReplayError("refusing to locate an empty token sequence")
    limit = len(haystack) - len(needle) + 1
    for index in range(start, max(start, limit)):
        if list(haystack[index : index + len(needle)]) == list(needle):
            return index
    raise CoordinateReplayError(
        f"token subsequence not found: start={start} needle_length={len(needle)}"
    )


def _prediction_records(
    *,
    tokenizer: Any,
    image_id: str,
    row_index: int,
    predictions: Sequence[Mapping[str, Any]],
    token_ids: Sequence[int],
    global_token_start: int,
    event_type: str,
    force_ordinal: int | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    search_start = 0
    for prediction_index, prediction in enumerate(predictions):
        coord_spans = list(prediction.get("coord_token_spans") or [])
        if len(coord_spans) != 4:
            raise CoordinateReplayError(
                f"{image_id}: row {row_index} prediction {prediction_index} has "
                f"{len(coord_spans)} coordinate spans"
            )
        coord_ids = [
            int(tokenizer.convert_tokens_to_ids(str(span["text"])))
            for span in coord_spans
        ]
        local_start = _find_subsequence(token_ids, coord_ids, search_start)
        coord_positions = [global_token_start + local_start + offset for offset in range(4)]
        search_start = local_start + 4
        records.append(
            {
                "image_id": image_id,
                "prediction_id": (
                    f"{image_id}:row-{row_index}:prediction-{prediction_index}"
                ),
                "generated_row_index": row_index,
                "prediction_index_within_row": prediction_index,
                "event_type": event_type,
                "force_ordinal": force_ordinal,
                "description": str(prediction["description"]),
                "bbox_pixel_xyxy": [int(value) for value in prediction["bbox"]],
                "coord_bins": [int(value) for value in prediction["coord_bins"]],
                "coord_token_ids": coord_ids,
                "coord_token_positions": coord_positions,
            }
        )
    return records


def _post_native_records(case: Mapping[str, Any], tokenizer: Any) -> list[dict[str, Any]]:
    image_id = str(case["image_id"])
    executed = [int(value) for value in case["executed_completion_token_ids"]]
    rebuilt: list[int] = []
    records: list[dict[str, Any]] = []
    observed_native = False

    for event in case["events"]:
        event_type = str(event["event_type"])
        if event_type == "natural_segment":
            segment = event["segment"]
            segment_ids = [int(value) for value in segment.get("raw_generated_token_ids") or []]
            segment_start = len(rebuilt)
            rebuilt.extend(segment_ids)
            if not observed_native:
                observed_native = True
                continue
            if bool(segment.get("segment_is_clean_complete_rows")):
                row_search_start = 0
                for row in segment.get("complete_rows") or []:
                    row_ids = [int(value) for value in row["raw_generated_token_ids"]]
                    row_local_start = _find_subsequence(
                        segment_ids, row_ids, row_search_start
                    )
                    row_search_start = row_local_start + len(row_ids)
                    records.extend(
                        _prediction_records(
                            tokenizer=tokenizer,
                            image_id=image_id,
                            row_index=int(row["row_index"]),
                            predictions=list(row.get("parsed_predictions") or []),
                            token_ids=row_ids,
                            global_token_start=segment_start + row_local_start,
                            event_type="post_force_natural",
                            force_ordinal=None,
                        )
                    )
            else:
                predictions = list(segment.get("parsed_predictions") or [])
                first_row_index = int(event["complete_row_count_after"]) - len(predictions)
                search_start = 0
                for offset, prediction in enumerate(predictions):
                    partial = _prediction_records(
                        tokenizer=tokenizer,
                        image_id=image_id,
                        row_index=first_row_index + offset,
                        predictions=[prediction],
                        token_ids=segment_ids[search_start:],
                        global_token_start=segment_start + search_start,
                        event_type="post_force_natural",
                        force_ordinal=None,
                    )
                    record = partial[0]
                    search_start = record["coord_token_positions"][-1] - segment_start + 1
                    records.append(record)
        elif event_type == "forced_row" and bool(event.get("accepted_complete_row")):
            row = event.get("row")
            if row is None:
                continue
            row_ids = [int(value) for value in row["raw_generated_token_ids"]]
            row_start = len(rebuilt)
            rebuilt.extend(row_ids)
            records.extend(
                _prediction_records(
                    tokenizer=tokenizer,
                    image_id=image_id,
                    row_index=int(event["row_index"]),
                    predictions=list(row.get("parsed_predictions") or []),
                    token_ids=row_ids,
                    global_token_start=row_start,
                    event_type="forced_row",
                    force_ordinal=int(event["force_index"]) + 1,
                )
            )

    if not observed_native:
        raise CoordinateReplayError(f"{image_id}: missing native natural segment")
    # A terminal malformed/incomplete forced attempt is retained in the executed
    # completion but intentionally has no accepted-row event.  Accepted event
    # tokens must therefore be an exact prefix; the unparsed terminal suffix is
    # outside this visualization's prediction set.
    if executed[: len(rebuilt)] != rebuilt:
        raise CoordinateReplayError(
            f"{image_id}: accepted event tokens are not an exact completion prefix"
        )
    expected_post = int(case["final_snapshot"]["prediction_count"]) - int(
        case["native_snapshot"]["prediction_count"]
    )
    if len(records) != expected_post:
        raise CoordinateReplayError(
            f"{image_id}: post-native prediction mismatch: {len(records)} != {expected_post}"
        )
    return records


def _reference_values(path: Path, checkpoint: str) -> list[float]:
    payload = _read_json(path)
    clusters = payload.get("clusters", {}).get(checkpoint)
    if not isinstance(clusters, list) or not clusters:
        raise CoordinateReplayError(f"no reference clusters for {checkpoint}")
    values = sorted(
        float(cluster["likelihood"]["medoid"]["coord_mean"])
        for cluster in clusters
    )
    if not all(math.isfinite(value) for value in values):
        raise CoordinateReplayError(f"non-finite reference coordinate likelihood for {checkpoint}")
    return values


def _reference_percentile(value: float, ordered: Sequence[float]) -> float:
    if len(ordered) == 1:
        return 0.5
    if value <= ordered[0]:
        return 0.0
    if value >= ordered[-1]:
        return 1.0
    left = bisect.bisect_left(ordered, value)
    right = bisect.bisect_right(ordered, value)
    if right > left:
        position = (left + right - 1) / 2.0
    else:
        position = float(left)
    return position / (len(ordered) - 1)


def _request_for_example(
    *,
    raw: Any,
    source_index: int,
    config: Any,
    frontend: Any,
    plan: Any,
) -> Any:
    from src.inference.backend import DecodeRequest, GenerationPolicy
    from src.inference.pipeline import _template_config
    from src.inference.prompt import build_prompt_record

    record = build_prompt_record(
        raw,
        _template_config(config),
        processor=frontend.qwen.processor,
        row_index=source_index,
        merged_visual_tokens=plan.merged_visual_tokens,
        object_order_seed=config.template.object_order_seed,
    )
    return DecodeRequest(
        request_id=str(raw.example_id),
        chat_text=record.chat_text,
        input_prompt_token_ids=tuple(record.input_prompt_token_ids),
        expected_executed_prompt_token_ids=tuple(record.expected_executed_prompt_token_ids),
        image_path=plan.image_path,
        declared_image_width=plan.declared_width,
        declared_image_height=plan.declared_height,
        decoded_image_width=plan.decoded_width,
        decoded_image_height=plan.decoded_height,
        image_sha256=plan.image_content_sha256,
        expected_image_grid_thw=cast(
            tuple[int, int, int], tuple(plan.expected_image_grid_thw)
        ),
        logical_transform_id=plan.logical_transform_id,
        generation_policy=GenerationPolicy(max_new_tokens=1),
    )


def _score_checkpoint(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    from src.config.fingerprint import sha256_json
    from src.config.inference import load_infer_config
    from src.data import load_raw_examples
    from src.inference.backend import open_backend_session
    from src.inference.hf_backend import (
        HFBackendSession,
        teacher_forced_chosen_token_logprobs,
    )
    from src.inference.image_plan import plan_image_batch
    from src.inference.pipeline import _processor_config
    from src.inference.runtime import assemble_frontend

    checkpoint = str(args.checkpoint)
    artifacts: list[tuple[str, Path, dict[str, Any]]] = []
    for rp_label, filename in RP_ARTIFACTS:
        path = (args.source_root / checkpoint / filename).resolve(strict=True)
        artifacts.append((rp_label, path, _read_json(path)))
    config_paths = {str(payload["config"]["infer_config"]) for _, _, payload in artifacts}
    if len(config_paths) != 1:
        raise CoordinateReplayError(f"{checkpoint}: RP artifacts use different configs")
    config_path = Path(config_paths.pop()).resolve(strict=True)
    resolved = load_infer_config(config_path)
    config = resolved.config.model_copy(
        update={"model": resolved.config.model.model_copy(update={"dtype": "fp32"})}
    )
    if config.backend.type != "hf":
        raise CoordinateReplayError("coordinate replay requires backend.type: hf")
    if not torch.cuda.is_available():
        raise CoordinateReplayError("CUDA is required for coordinate replay")
    torch.cuda.set_device(0)

    raw_examples = list(load_raw_examples(config.data.input_jsonl))
    frontend = assemble_frontend(
        config,
        generation_config_fingerprint=sha256_json(config.generation.model_dump(mode="json")),
    )
    plans = plan_image_batch(
        raw_examples,
        components=frontend.qwen,
        processor_config=_processor_config(config),
        row_indices=list(range(len(raw_examples))),
    )
    plan_by_id = {str(plan.row_id): plan for plan in plans.rows}
    raw_by_image = {str(physical_image_id(raw)): (index, raw) for index, raw in enumerate(raw_examples)}
    references = _reference_values(args.reference_cluster_confidence, checkpoint)

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    with open_backend_session(frontend.launch) as opened:
        if not isinstance(opened, HFBackendSession):
            raise CoordinateReplayError("coordinate replay opened a non-HF session")
        tokenizer = opened._tokenizer
        if tokenizer is None:
            raise CoordinateReplayError("HF session lacks tokenizer")
        for rp_label, source_path, payload in artifacts:
            for case in payload["cases"]:
                image_id = str(case["image_id"])
                source_index, raw = raw_by_image[image_id]
                plan = plan_by_id[str(raw.example_id)]
                request = _request_for_example(
                    raw=raw,
                    source_index=source_index,
                    config=config,
                    frontend=frontend,
                    plan=plan,
                )
                native_inputs, executed_prompt_ids, _, _ = opened._materialize_native_inputs(
                    (request,)
                )
                if len(executed_prompt_ids) != 1:
                    raise CoordinateReplayError(f"{image_id}: unexpected prompt batch")
                if hash_prefix_token_ids(executed_prompt_ids[0]) != str(
                    case["prompt_token_ids_sha256"]
                ):
                    raise CoordinateReplayError(f"{image_id}: prompt token hash mismatch")

                records = _post_native_records(case, tokenizer)[
                    :MAX_POST_NATIVE_PREDICTIONS
                ]
                if not records:
                    continue
                coord_positions = sorted(
                    {position for record in records for position in record["coord_token_positions"]}
                )
                prefix_end = max(coord_positions) + 1
                generated_ids = [
                    int(value) for value in case["executed_completion_token_ids"][:prefix_end]
                ]
                chosen_logprobs = teacher_forced_chosen_token_logprobs(
                    model=opened._model,
                    native_prompt_inputs=native_inputs,
                    generated_token_ids=generated_ids,
                )
                for post_order, record in enumerate(records, 1):
                    positions = record.pop("coord_token_positions")
                    logprobs = [float(chosen_logprobs[position]) for position in positions]
                    observed_ids = [generated_ids[position] for position in positions]
                    if observed_ids != record["coord_token_ids"]:
                        raise CoordinateReplayError(
                            f"{record['prediction_id']}: coordinate token identity mismatch"
                        )
                    coord_mean = mean(logprobs)
                    rows.append(
                        {
                            "checkpoint": checkpoint,
                            "arm": f"{checkpoint}-{rp_label}",
                            "repetition_penalty": float(
                                payload["config"]["repetition_penalty"]
                            ),
                            "source_artifact": str(source_path),
                            "post_native_prediction_order": post_order,
                            **record,
                            "coord_logprobs": logprobs,
                            "coord_mean": coord_mean,
                            "coord_min": min(logprobs),
                            "coord_confidence_percentile": _reference_percentile(
                                coord_mean, references
                            ),
                            "confidence_reference": (
                                "checkpoint-specific K16 union-cluster medoid coord_mean CDF"
                            ),
                        }
                    )
                del chosen_logprobs

    elapsed = time.perf_counter() - started
    expected_arms = {f"{checkpoint}-rp1.0", f"{checkpoint}-rp1.1"}
    if {row["arm"] for row in rows} != expected_arms:
        raise CoordinateReplayError(f"{checkpoint}: missing scored arm")
    receipt = {
        "schema_version": "forced_continue_coord_likelihood_replay.v1",
        "checkpoint": checkpoint,
        "score_definition": (
            "FP32 raw-model teacher-forced mean log probability over four coordinate tokens"
        ),
        "color_confidence_definition": (
            "percentile of coord_mean against the checkpoint-specific K16 union-cluster "
            "medoid distribution from the 2026-07-29 likelihood probe"
        ),
        "reference_cluster_confidence": str(args.reference_cluster_confidence.resolve()),
        "max_post_native_predictions_scored_per_case": MAX_POST_NATIVE_PREDICTIONS,
        "row_count": len(rows),
        "case_count": len({(row["arm"], row["image_id"]) for row in rows}),
        "arm_counts": {
            arm: sum(row["arm"] == arm for row in rows) for arm in sorted(expected_arms)
        },
        "wall_seconds": elapsed,
        "device": torch.cuda.get_device_name(0),
    }
    return rows, receipt


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", choices=CHECKPOINTS, required=True)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--reference-cluster-confidence", type=Path, default=DEFAULT_REFERENCE
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.source_root = args.source_root.resolve(strict=True)
    args.reference_cluster_confidence = args.reference_cluster_confidence.resolve(
        strict=True
    )
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"coord-likelihood-{args.checkpoint}.jsonl"
    receipt_path = args.output_dir / f"replay-receipt-{args.checkpoint}.json"
    if (output_path.exists() or receipt_path.exists()) and not args.force:
        raise FileExistsError(
            f"refusing to overwrite {output_path} or {receipt_path}; pass --force"
        )
    rows, receipt = _score_checkpoint(args)
    _write_jsonl(output_path, rows)
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
