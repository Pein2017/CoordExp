#!/usr/bin/env python3
"""Score one fixed-prefix coordinate decision for the person-25 closeout.

The probe keeps the historical random-order checkpoint, image, prompt, and
raw sampled person-25 donor row fixed.  It then teacher-forces the partial
current row ``<|object_ref_start|>person<|box_start|>0,298,93`` and records
the next-token distribution.  The output is intentionally a small JSON
artifact: it measures coordinate probability competition, not hidden states or
full rollout quality.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research import run_historical_random_sorted_image2299_screen as historical
from scripts.research import run_person25_commit_closeout as closeout


SCHEMA_VERSION = "person25_y2_coordinate_competition.v1"
DEFAULT_OUTPUT_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-18-person25-dominant-owner-commit-and-persistence-closeout"
)
DEFAULT_SOURCE_ROOT = closeout.DEFAULT_SOURCE_ROOT
DEFAULT_DONOR_SEED = 0
DEFAULT_EARLIER_OWNER = 2
DEFAULT_CURRENT_COORDINATES = (0, 298, 93)
DEFAULT_BROAD_BAND = (500, 600)
DEFAULT_OBSERVED_BAND = (514, 576)


def _sha256_ids(values: Sequence[int]) -> str:
    return hashlib.sha256(
        json.dumps([int(value) for value in values], separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        return float("-inf")
    maximum = max(float(value) for value in values)
    if not math.isfinite(maximum):
        return maximum
    return maximum + math.log(math.fsum(math.exp(float(value) - maximum) for value in values))


def _normalized_probabilities(logits: Sequence[float], temperature: float) -> list[float]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = [float(value) / float(temperature) for value in logits]
    normalizer = _logsumexp(scaled)
    return [math.exp(value - normalizer) for value in scaled]


def _band_values(token_to_coordinate: Mapping[int, int], low: int, high: int) -> list[int]:
    if low > high:
        raise ValueError("band lower bound must not exceed upper bound")
    return [int(token_id) for token_id, coordinate in token_to_coordinate.items() if low <= int(coordinate) <= high]


def aggregate_coordinate_competition(
    logits: Sequence[float],
    coordinate_token_ids: Mapping[int, int],
    *,
    target_coordinate: int = 999,
    broad_band: tuple[int, int] = DEFAULT_BROAD_BAND,
    observed_band: tuple[int, int] = DEFAULT_OBSERVED_BAND,
    temperature: float = 0.4,
    top_k: int = 20,
) -> dict[str, Any]:
    """Aggregate a vocabulary logit vector without model-specific behavior.

    ``coordinate_token_ids`` maps coordinate value (0..999) to its vocabulary
    token identifier.  Masses are reported both against the full vocabulary
    and conditionally within the coordinate-token set, so a low coordinate
    probability cannot be mistaken for a high coordinate-relative preference.
    """

    values = [float(value) for value in logits]
    if not values:
        raise ValueError("logits must be non-empty")
    if target_coordinate not in coordinate_token_ids:
        raise ValueError(f"target coordinate {target_coordinate} is missing")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    token_to_coordinate = {int(token_id): int(coordinate) for coordinate, token_id in coordinate_token_ids.items()}
    if len(token_to_coordinate) != len(coordinate_token_ids):
        raise ValueError("coordinate token identifiers must be unique")
    if any(token_id < 0 or token_id >= len(values) for token_id in token_to_coordinate):
        raise ValueError("coordinate token identifier is outside the logits vocabulary")

    raw_probabilities = _normalized_probabilities(values, 1.0)
    temperature_probabilities = _normalized_probabilities(values, temperature)
    coordinate_items = sorted(token_to_coordinate.items(), key=lambda item: item[1])
    coordinate_token_values = [token_id for token_id, _ in coordinate_items]
    raw_coordinate_mass = math.fsum(raw_probabilities[token_id] for token_id in coordinate_token_values)
    temperature_coordinate_mass = math.fsum(
        temperature_probabilities[token_id] for token_id in coordinate_token_values
    )

    def mass(probabilities: Sequence[float], low: int, high: int) -> float:
        token_ids = _band_values(token_to_coordinate, low, high)
        return math.fsum(probabilities[token_id] for token_id in token_ids)

    def band_record(probabilities: Sequence[float], low: int, high: int) -> dict[str, float]:
        full_mass = mass(probabilities, low, high)
        denominator = raw_coordinate_mass if probabilities is raw_probabilities else temperature_coordinate_mass
        return {
            "coordinate_low_inclusive": int(low),
            "coordinate_high_inclusive": int(high),
            "probability_mass_over_full_vocabulary": float(full_mass),
            "probability_mass_given_coordinate_token": float(full_mass / denominator) if denominator else 0.0,
        }

    def margin(low: int, high: int, *, scaled_temperature: float) -> float:
        band_logits = [values[token_id] / scaled_temperature for token_id in _band_values(token_to_coordinate, low, high)]
        target_logit = values[int(coordinate_token_ids[target_coordinate])] / scaled_temperature
        return float(_logsumexp(band_logits) - target_logit)

    ranked = sorted(coordinate_items, key=lambda item: values[item[0]], reverse=True)[:top_k]
    top_coordinates = [
        {
            "coordinate": int(coordinate),
            "token_id": int(token_id),
            "logit": float(values[token_id]),
            "raw_probability": float(raw_probabilities[token_id]),
            "temperature_0_4_probability": float(temperature_probabilities[token_id]),
        }
        for token_id, coordinate in ranked
    ]
    target_token_id = int(coordinate_token_ids[target_coordinate])
    raw_coordinate_argmax_token, raw_coordinate_argmax_value = max(
        coordinate_items, key=lambda item: values[item[0]]
    )
    temperature_coordinate_argmax_token, temperature_coordinate_argmax_value = max(
        coordinate_items, key=lambda item: values[item[0]] / temperature
    )
    return {
        "vocabulary_size": len(values),
        "coordinate_token_count": len(coordinate_items),
        "temperature": float(temperature),
        "target": {
            "coordinate": int(target_coordinate),
            "token_id": target_token_id,
            "logit": float(values[target_token_id]),
            "raw_probability": float(raw_probabilities[target_token_id]),
            "temperature_0_4_probability": float(temperature_probabilities[target_token_id]),
        },
        "coordinate_vocab_mass": {
            "raw_probability": float(raw_coordinate_mass),
            "temperature_0_4_probability": float(temperature_coordinate_mass),
        },
        "bands": {
            f"{broad_band[0]}_{broad_band[1]}": {
                "raw": band_record(raw_probabilities, *broad_band),
                "temperature_0_4": band_record(temperature_probabilities, *broad_band),
                "raw_logsumexp_margin_band_vs_coord999": margin(*broad_band, scaled_temperature=1.0),
                "temperature_0_4_logsumexp_margin_band_vs_coord999": margin(
                    *broad_band, scaled_temperature=temperature
                ),
            },
            f"{observed_band[0]}_{observed_band[1]}": {
                "raw": band_record(raw_probabilities, *observed_band),
                "temperature_0_4": band_record(temperature_probabilities, *observed_band),
                "raw_logsumexp_margin_band_vs_coord999": margin(*observed_band, scaled_temperature=1.0),
                "temperature_0_4_logsumexp_margin_band_vs_coord999": margin(
                    *observed_band, scaled_temperature=temperature
                ),
            },
        },
        "coordinate_argmax": {
            "raw": {
                "coordinate": int(raw_coordinate_argmax_value),
                "token_id": int(raw_coordinate_argmax_token),
            },
            "temperature_0_4": {
                "coordinate": int(temperature_coordinate_argmax_value),
                "token_id": int(temperature_coordinate_argmax_token),
            },
        },
        "top_coordinate_tokens_by_raw_logit": top_coordinates,
    }


def _tokenize_exact(tokenizer: Any, text: str, expected_count: int | None = None) -> list[int]:
    ids = historical._tokenize_row(tokenizer, text)
    if expected_count is not None and len(ids) != expected_count:
        raise RuntimeError(f"unexpected token count {len(ids)} for {text!r}; expected {expected_count}")
    return ids


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    source_root = Path(args.source_root).expanduser().resolve(strict=True)
    checkpoint = Path(args.checkpoint or historical.DEFAULT_CHECKPOINTS["random"]).expanduser().resolve(strict=True)
    base_model = Path(args.base_model).expanduser().resolve(strict=True)
    image_path = Path(args.image).expanduser().resolve(strict=True)
    jsonl_path = Path(args.jsonl).expanduser().resolve(strict=True)
    output_root = Path(args.output_root).expanduser().resolve()
    output_path = output_root / "y2-fixed-prefix-coordinate-competition.json"
    if output_path.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {output_path}; pass --force")
    packet = historical.select_image2299_rows(jsonl_path, image_path)
    prompt = historical.verify_historical_prompt("sorted")
    donor = closeout.source_sample(source_root, owner_rank=2, seed=DEFAULT_DONOR_SEED)
    common_attestation = closeout.common_donor_attestation(source_root, seed=DEFAULT_DONOR_SEED)
    earlier_owner = int(args.earlier_owner)
    by_person = packet["by_person_rank"]
    history_rows = []
    for rank in (0, 1, earlier_owner):
        obj = by_person[rank]
        bbox = [historical._coord_value(str(token)) for token in obj["bbox_2d"]]
        history_rows.append(historical.render_legacy_row(str(obj["desc"]), bbox))

    model, processor, coord_receipt = historical._load_historical_model(
        base_model, checkpoint, str(args.device)
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": prompt["system"]}]},
        {"role": "user", "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": prompt["user"]},
        ]},
    ]
    base_inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    tokenizer = processor.tokenizer
    base_prompt_ids = [int(value) for value in base_inputs["input_ids"][0].tolist()]
    history_ids: list[int] = []
    for row in history_rows:
        history_ids.extend(_tokenize_exact(tokenizer, row, expected_count=7))
    donor_ids = [int(value) for value in donor["generated_token_ids"]]
    if len(donor_ids) != 7:
        raise RuntimeError("person-25 donor must contain exactly seven raw token IDs")
    prefix_ids = history_ids + donor_ids
    current_values = tuple(int(value) for value in args.current_coordinates)
    current_row_text = (
        "<|object_ref_start|>person<|box_start|>"
        + "".join(historical.coord_token(value) for value in current_values)
    )
    current_row_ids = _tokenize_exact(tokenizer, current_row_text, expected_count=6)
    full_sequence = base_prompt_ids + prefix_ids + current_row_ids
    logits = historical._forward_batch(
        model,
        base_inputs,
        [full_sequence],
        str(args.device),
        logits_to_keep=1,
    )
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != 1:
        raise RuntimeError(f"unexpected final-logit shape: {getattr(logits, 'shape', None)}")
    vocabulary_logits = [float(value) for value in logits[0, 0].tolist()]
    coordinate_token_ids: dict[int, int] = {}
    for coordinate in range(1000):
        token_id = tokenizer.convert_tokens_to_ids(historical.coord_token(coordinate))
        if token_id is None or int(token_id) < 0:
            raise RuntimeError(f"missing coordinate token {coordinate}")
        coordinate_token_ids[coordinate] = int(token_id)
    aggregate = aggregate_coordinate_competition(vocabulary_logits, coordinate_token_ids)
    for item in aggregate["top_coordinate_tokens_by_raw_logit"]:
        item["token"] = historical.coord_token(int(item["coordinate"]))
    aggregate["target"]["token"] = historical.coord_token(999)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "question": "At a fixed current-row prefix, how strongly does the coordinate-token distribution favor the observed 500-600 bands over coordinate 999?",
        "model": {
            "base_model": str(base_model),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": historical.sha256_file(checkpoint / "adapter_model.safetensors"),
            "adapter_role": "random",
            "runtime_dtype": "torch.float32",
            "attention_implementation": "eager",
            "device": str(args.device),
        },
        "image": {
            "image_id": historical.EXPECTED_IMAGE_ID,
            "path": str(image_path),
            "sha256": packet["image_sha256"],
            "source_jsonl": str(jsonl_path),
            "source_jsonl_full_file_sha256": packet["jsonl_sha256"],
            "source_jsonl_record_sha256": packet["record_sha256"],
            "source_line_index_1_based": packet["source_line_index"],
        },
        "prompt": {"hash": prompt["hash"], "payload": prompt["payload"]},
        "prefix": {
            "history_person_ranks": [0, 1, earlier_owner],
            "history_rows": history_rows,
            "history_token_ids": history_ids,
            "donor_seed": DEFAULT_DONOR_SEED,
            "donor_source_path": donor["source_path"],
            "donor_raw_token_ids": donor_ids,
            "donor_raw_token_ids_sha256": _sha256_ids(donor_ids),
            "donor_common_history_attestation": common_attestation,
            "prefix_token_ids": prefix_ids,
            "prefix_token_ids_sha256": _sha256_ids(prefix_ids),
            "base_prompt_token_count": len(base_prompt_ids),
            "current_row_prefix_text": current_row_text,
            "current_row_prefix_token_ids": current_row_ids,
            "current_row_prefix_token_ids_sha256": _sha256_ids(current_row_ids),
            "full_model_input_ids_sha256": _sha256_ids(full_sequence),
        },
        "coordinate_token_map_sha256": _sha256_ids([coordinate_token_ids[i] for i in range(1000)]),
        "coord_offset_load_receipt": coord_receipt,
        "distribution": aggregate,
        "execution": {
            "torch_version": torch.__version__,
            "image_forward": True,
            "logits_to_keep": 1,
            "teacher_forced_next_token_only": True,
        },
    }
    historical._atomic_json_dump(output_path, payload, force=bool(args.force))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--base-model", type=Path, default=historical.DEFAULT_BASE_MODEL)
    parser.add_argument("--image", type=Path, default=historical.DEFAULT_IMAGE)
    parser.add_argument("--jsonl", type=Path, default=historical.DEFAULT_JSONL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--earlier-owner", type=int, default=DEFAULT_EARLIER_OWNER)
    parser.add_argument("--current-coordinates", type=int, nargs=3, default=DEFAULT_CURRENT_COORDINATES)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_probe(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
