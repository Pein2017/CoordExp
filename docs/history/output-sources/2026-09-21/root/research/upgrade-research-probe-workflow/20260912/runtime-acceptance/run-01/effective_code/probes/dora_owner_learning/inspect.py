"""Inspect a selected cohort with an existing profile, without model weights."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.artifacts import publish_json_exclusive
from src.config.fingerprint import sha256_file, sha256_json
from src.config.inference import load_research_infer_config
from src.data import RawExample, load_raw_examples
from src.inference.runtime import assemble_frontend
from src.qwen.native import prepare_native_inputs

from .runtime import DEFAULT_CONFIG


def select_examples(
    rows: Sequence[RawExample],
    *,
    example_ids: Sequence[str] | None = None,
    count: int | None = None,
    seed: int | None = None,
) -> tuple[RawExample, ...]:
    """Select once and pass the returned ordered tuple unchanged to both arms."""
    by_id = {row.example_id: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("source example IDs must be unique")
    if example_ids is not None:
        if count is not None or seed is not None:
            raise ValueError("explicit IDs cannot be combined with count or seed")
        if isinstance(example_ids, str) or not example_ids:
            raise ValueError("explicit IDs must be a nonempty sequence")
        if len(set(example_ids)) != len(example_ids):
            raise ValueError("requested example IDs must be unique")
        unknown = [key for key in example_ids if key not in by_id]
        if unknown:
            raise ValueError(f"unknown example IDs: {unknown}")
        return tuple(by_id[key] for key in example_ids)
    if type(count) is not int or count <= 0:
        raise ValueError("selection requires explicit IDs or a positive count")
    if type(seed) is not int:
        raise ValueError("count selection requires an explicit integer seed")
    if count > len(rows):
        raise ValueError("selection count exceeds the source population")
    return tuple(random.Random(seed).sample(list(rows), count))


def inspect_examples(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    input_path: str | Path | None = None,
    example_ids: Sequence[str] | None = None,
    count: int | None = None,
    seed: int | None = None,
    target_max_length: int | None = None,
) -> dict[str, Any]:
    """Return input evidence only; fixed Source256 admission remains separate."""
    resolved = load_research_infer_config(config_path)
    config = resolved.config
    source = Path(input_path or config.data.input_jsonl).resolve(strict=True)
    source_hash = sha256_file(source)
    population = load_raw_examples(source)
    if sha256_file(source) != source_hash:
        raise ValueError("input source changed during inspection")
    selected = select_examples(population, example_ids=example_ids, count=count, seed=seed)
    indices = {row.example_id: index for index, row in enumerate(population)}
    if target_max_length is not None and (
        type(target_max_length) is not int or target_max_length <= 0
    ):
        raise ValueError("target maximum length must be a positive integer")

    from src.inference.inputs import plan_examples

    frontend = assemble_frontend(
        config,
        generation_config_fingerprint=sha256_json(config.generation.model_dump(mode="json")),
    )
    plans = plan_examples(
        selected, config=config, components=frontend.qwen,
        row_indices=[indices[row.example_id] for row in selected],
        target_max_length=target_max_length,
    )
    native = prepare_native_inputs(
        frontend.qwen.processor, tuple(plan.request for plan in plans),
        device="cpu", record_media_identity=True,
    )
    inspected = []
    for index, plan in enumerate(plans):
        target = None if plan.target is None else {
            **plan.target.to_artifact_dict(), "input_ids": list(plan.target.input_ids),
        }
        inspected.append({
            "example_id": plan.prompt.example_id,
            "source_row_index": plan.prompt.row_index,
            "prompt": plan.prompt.to_artifact_dict(),
            "image": plan.image.to_artifact_dict(),
            "native_prompt_token_ids": list(native.prompt_token_ids[index]),
            "native_image_grid": list(native.image_grids[index]),
            "executed_media_sha256": native.media_sha256[index],
            "target": target,
        })
    membership = [row.example_id for row in selected]
    return {
        "schema_version": "probe_input_inspection.v1",
        "scope": "input-inspection-only",
        "model_weights_loaded": False,
        "config_path": str(Path(config_path).resolve()),
        "config_fingerprint": resolved.fingerprint,
        "configured_input_path": str(config.data.input_jsonl),
        "source": {"path": str(source), "sha256": source_hash, "population_count": len(population)},
        "selection": {"mode": "ids" if example_ids is not None else "count_seed",
                      "seed": seed, "example_ids": membership,
                      "sha256": sha256_json({"source_sha256": source_hash, "example_ids": membership})},
        "native_tensor_shapes": {key: list(value.shape) for key, value in native.inputs.items()
                                 if hasattr(value, "shape")},
        "rows": inspected,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input", type=Path, help="Explicit data source; leaves the saved profile unchanged")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--ids", nargs="+")
    selection.add_argument("--count", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--target-max-length", type=int, help="Also inspect annotated targets with this limit")
    parser.add_argument("--output", type=Path, help="Write a new JSON file instead of stdout")
    args = parser.parse_args()
    result = inspect_examples(
        args.config, input_path=args.input, example_ids=args.ids,
        count=args.count, seed=args.seed, target_max_length=args.target_max_length,
    )
    if args.output is None:
        print(json.dumps(result, indent=2, allow_nan=False))
    else:
        publish_json_exclusive(args.output, result)
        print(json.dumps({"output": str(args.output), "selected_count": len(result["rows"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
