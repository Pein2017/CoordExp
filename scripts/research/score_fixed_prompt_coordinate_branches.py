#!/usr/bin/env python3
"""Compare clean and degraded coordinate branches at one fixed prompt.

This is a small, auditable probe for a specific research question: when two
sampled full-image rollouts share the same base prompt and row prefix, does
the language model already assign useful probability mass to the physical
coordinate neighborhood, even when one sampled branch emits a poor extent?

The bundle checks are deliberately strict.  A result from a different image,
history policy, prompt, row, category, or model composition is not a valid
paired comparison.  The executable path uses the current CoordExp-Swift
runtime and loads the base Qwen3-VL model, the Parameter-Efficient Fine-Tuning
DoRA adapter, and the selected special-token embedding delta through their
owner modules.  All probability calculations are performed from CPU
``float32`` logits.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


SCHEMA_VERSION = "fixed_prompt_coordinate_branches.v1"
FULL_BAG_K = "FULL_BAG_K"
FRESH_BASE_PROMPT = "fresh_base_prompt_per_call"
COORDINATE_SLOT_NAMES = ("x1", "y1", "x2", "y2")
DEFAULT_INFER_CONFIG = Path(
    "configs/coordexp_swift/infer/"
    "qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_step4887_val200.yaml"
)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read JSON {path}: {exc}") from exc


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _bundle_prompt_evidence(bundle: Mapping[str, Any]) -> Mapping[str, Any]:
    evidence = _as_mapping(bundle.get("execution_evidence"), "execution_evidence")
    return _as_mapping(evidence.get("executed_prompt_evidence"), "executed_prompt_evidence")


def _bundle_arm(bundle: Mapping[str, Any]) -> Mapping[str, Any]:
    evidence = _as_mapping(bundle.get("execution_evidence"), "execution_evidence")
    arm = evidence.get("arm")
    if not isinstance(arm, Mapping):
        arm = _as_mapping(_as_mapping(bundle.get("scheduled_request"), "scheduled_request").get("arm"), "scheduled arm")
    return arm


def _bundle_image_id(bundle: Mapping[str, Any]) -> str:
    evidence = _as_mapping(bundle.get("execution_evidence"), "execution_evidence")
    scheduled = _as_mapping(bundle.get("scheduled_request"), "scheduled_request")
    value = evidence.get("image_id", scheduled.get("image_id"))
    if value is None:
        raise ValueError("bundle is missing image_id")
    return str(value)


def _bundle_model_identity(bundle: Mapping[str, Any]) -> Mapping[str, Any]:
    decode = _as_mapping(bundle.get("decode_result"), "decode_result")
    return _as_mapping(decode.get("model_identity"), "decode_result.model_identity")


def expected_model_composition_from_config(config: Any) -> dict[str, str | None]:
    """Project the current inference configuration into comparable identities."""

    adapter = getattr(config, "adapter", None)
    delta = getattr(config, "embedding_delta", None)
    return {
        "base_model_path": str(Path(config.model.base_model).expanduser().resolve()),
        "adapter_path": None if adapter is None else str(Path(adapter.path).expanduser().resolve()),
        "adapter_type": None if adapter is None else str(adapter.type),
        "adapter_name": None if adapter is None else str(adapter.name),
        "embedding_delta_path": None if delta is None else str(Path(delta.path).expanduser().resolve()),
    }


def _bundle_composition_paths(bundle: Mapping[str, Any]) -> dict[str, str | None]:
    identity = _bundle_model_identity(bundle)
    base = _as_mapping(identity.get("base"), "model_identity.base")
    adapter = identity.get("adapter")
    embedding = identity.get("embedding_delta")
    adapter_map = adapter if isinstance(adapter, Mapping) else {}
    embedding_map = embedding if isinstance(embedding, Mapping) else {}
    nested_identity = embedding_map.get("identity")
    if isinstance(nested_identity, Mapping):
        embedding_map = nested_identity
    return {
        "base_model_path": str(Path(str(base.get("path"))).expanduser().resolve()) if base.get("path") else None,
        "adapter_path": str(Path(str(adapter_map.get("adapter_path"))).expanduser().resolve())
        if adapter_map.get("adapter_path")
        else None,
        "adapter_type": str(adapter_map.get("adapter_type")) if adapter_map.get("adapter_type") else None,
        "adapter_name": str(adapter_map.get("adapter_name")) if adapter_map.get("adapter_name") else None,
        "embedding_delta_path": str(Path(str(embedding_map.get("delta_path"))).expanduser().resolve())
        if embedding_map.get("delta_path")
        else None,
    }


def _row_receipt(bundle: Mapping[str, Any], row_index: int = 0) -> Mapping[str, Any]:
    receipts = bundle.get("parse_score_receipts")
    if not isinstance(receipts, list):
        raise ValueError("bundle parse_score_receipts must be a list")
    matches = []
    for receipt in receipts:
        # ``generated_row_index`` is the row ordinal in the decoded sequence.
        # ``parse_row_index`` is a parser-internal source position and is not
        # guaranteed to start at zero for a sampled rollout.
        if isinstance(receipt, Mapping) and int(receipt.get("generated_row_index", -1)) == row_index:
            matches.append(receipt)
    if len(matches) != 1:
        raise ValueError(f"bundle must contain exactly one parse row {row_index}, found {len(matches)}")
    return matches[0]


def _generated_tokens(bundle: Mapping[str, Any]) -> list[int]:
    decode = _as_mapping(bundle.get("decode_result"), "decode_result")
    values = decode.get("generated_token_ids")
    if not isinstance(values, list) or not values:
        raise ValueError("bundle decode_result.generated_token_ids must be non-empty")
    return [int(value) for value in values]


def _coordinate_slots(bundle: Mapping[str, Any], row: Mapping[str, Any]) -> tuple[list[int], list[int]]:
    selected = row.get("selected_token_ids")
    steps = row.get("selected_generated_step_indices")
    if not isinstance(selected, list) or len(selected) < 7:
        raise ValueError("row lacks the eight selected tokens needed for four coordinates")
    if not isinstance(steps, list) or len(steps) < 7:
        raise ValueError("row lacks selected generated step indices")
    coordinates = row.get("coordinate_bins")
    if not isinstance(coordinates, list) or len(coordinates) != 4:
        raise ValueError("row coordinate_bins must contain x1, y1, x2, and y2")
    return [int(value) for value in selected[3:7]], [int(value) for value in steps[3:7]]


def validate_bundle_pair(
    case: Mapping[str, Any],
    clean_bundle: Mapping[str, Any],
    bad_bundle: Mapping[str, Any],
    *,
    expected_model_composition: Mapping[str, str | None] | None = None,
    allow_counterfactual_model_composition: bool = False,
    counterfactual_purpose: str | None = None,
) -> dict[str, Any]:
    """Validate all conditions that make a clean/degraded pair interpretable."""

    if counterfactual_purpose is not None:
        counterfactual_purpose = str(counterfactual_purpose).strip()
        if not counterfactual_purpose:
            raise ValueError("counterfactual purpose must be non-empty when provided")
    if allow_counterfactual_model_composition and expected_model_composition is None:
        raise ValueError(
            "counterfactual model composition mode requires the current inference model composition"
        )
    if allow_counterfactual_model_composition and counterfactual_purpose is None:
        raise ValueError(
            "counterfactual model composition mode requires a non-empty counterfactual purpose"
        )
    if counterfactual_purpose is not None and not allow_counterfactual_model_composition:
        raise ValueError(
            "counterfactual purpose requires explicit counterfactual model composition mode"
        )

    case_image = str(case.get("image_id"))
    case_category = str(case.get("category", "")).strip().lower()
    if not case_image or not case_category:
        raise ValueError("case requires non-empty image_id and category")
    if _bundle_image_id(clean_bundle) != case_image or _bundle_image_id(bad_bundle) != case_image:
        raise ValueError("clean and bad bundles must match case image_id")
    for label, bundle in (("clean", clean_bundle), ("bad", bad_bundle)):
        arm = _bundle_arm(bundle)
        if str(arm.get("arm_code")) != FULL_BAG_K:
            raise ValueError(f"{label} bundle is not Full-Image K-Rollout Independent Bagging")
        if str(arm.get("history_policy")) != FRESH_BASE_PROMPT:
            raise ValueError(f"{label} bundle does not use fresh base prompt per call")
        row = _row_receipt(bundle)
        if int(row.get("generated_row_index", -1)) != 0:
            raise ValueError(f"{label} row is not generated row index 0")
        category = str(row.get("normalized_category_name") or row.get("category_text") or "").strip().lower()
        if category != case_category:
            raise ValueError(f"{label} category {category!r} does not match case {case_category!r}")
        prompt = _bundle_prompt_evidence(bundle)
        decode = _as_mapping(bundle.get("decode_result"), "decode_result")
        prompt_ids = decode.get("prompt_token_ids")
        if not isinstance(prompt_ids, list) or not prompt_ids:
            raise ValueError(f"{label} bundle has no prompt token ids")
        record_ids = prompt.get("prompt_token_ids")
        if isinstance(record_ids, list) and [int(value) for value in record_ids] != [int(value) for value in prompt_ids]:
            raise ValueError(f"{label} stored prompt record ids differ from decode prompt ids")
    clean_prompt = _bundle_prompt_evidence(clean_bundle)
    bad_prompt = _bundle_prompt_evidence(bad_bundle)
    for field in ("full_prompt_fingerprint",):
        if not clean_prompt.get(field) or clean_prompt.get(field) != bad_prompt.get(field):
            raise ValueError(f"clean and bad bundles do not share {field}")
    clean_ids = [int(value) for value in _as_mapping(clean_bundle["decode_result"], "decode_result")["prompt_token_ids"]]
    bad_ids = [int(value) for value in _as_mapping(bad_bundle["decode_result"], "decode_result")["prompt_token_ids"]]
    if clean_ids != bad_ids:
        raise ValueError("clean and bad bundles do not share exact prompt token ids")
    clean_identity = _bundle_composition_paths(clean_bundle)
    bad_identity = _bundle_composition_paths(bad_bundle)
    if clean_identity != bad_identity:
        raise ValueError("clean and bad bundles have different model composition")
    executed_identity = None if expected_model_composition is None else dict(expected_model_composition)
    composition_differs = executed_identity is not None and clean_identity != executed_identity
    if composition_differs and not allow_counterfactual_model_composition:
        raise ValueError(
            "bundle model composition does not match the current inference configuration: "
            f"observed={clean_identity!r}, expected={executed_identity!r}"
        )
    clean_row = _row_receipt(clean_bundle)
    bad_row = _row_receipt(bad_bundle)
    clean_token_ids, clean_steps = _coordinate_slots(clean_bundle, clean_row)
    bad_token_ids, bad_steps = _coordinate_slots(bad_bundle, bad_row)
    if clean_steps != bad_steps:
        raise ValueError("clean and bad coordinate step indices differ")
    clean_generated = _generated_tokens(clean_bundle)
    bad_generated = _generated_tokens(bad_bundle)
    first_slot = next((index for index, (a, b) in enumerate(zip(clean_token_ids, bad_token_ids)) if a != b), None)
    if first_slot is None:
        raise ValueError("clean and bad rows have no differing coordinate slot")
    step = clean_steps[first_slot]
    if step <= 0 or step >= len(clean_generated) or step >= len(bad_generated):
        raise ValueError("first differing coordinate step lies outside generated token ids")
    if clean_generated[:step] != bad_generated[:step]:
        raise ValueError("clean and bad branches do not share the generated prefix before the first differing coordinate")
    return {
        "case_name": str(case.get("name", "")),
        "image_id": case_image,
        "category": case_category,
        "role": str(case.get("role", "primary")),
        "full_prompt_fingerprint": str(clean_prompt["full_prompt_fingerprint"]),
        "prompt_token_ids_sha256": _sha256_json(clean_ids),
        "prompt_token_count": len(clean_ids),
        # Keep the source composition distinct from the model used for this
        # replay.  In ordinary mode they are equal; counterfactual mode is an
        # explicit, auditable exception rather than a weakened comparison.
        "source_model_composition": clean_identity,
        "executed_model_composition": executed_identity,
        "counterfactual_model_composition": {
            "enabled": bool(allow_counterfactual_model_composition),
            "source_and_executed_composition_differ": bool(composition_differs),
            "purpose": counterfactual_purpose,
        },
        # Backward-compatible alias retained for the first probe readers.
        "model_composition": clean_identity,
        "first_differing_coordinate_slot_index": int(first_slot),
        "first_differing_coordinate_slot": COORDINATE_SLOT_NAMES[first_slot],
        "first_differing_generated_step": int(step),
        "clean_coordinate_token_id": int(clean_token_ids[first_slot]),
        "bad_coordinate_token_id": int(bad_token_ids[first_slot]),
        "clean_coordinate_bins": [int(value) for value in clean_row["coordinate_bins"]],
        "bad_coordinate_bins": [int(value) for value in bad_row["coordinate_bins"]],
        "shared_generated_prefix_token_ids": clean_generated[:step],
        "clean_request_id": str(clean_bundle.get("request_id", "")),
        "bad_request_id": str(bad_bundle.get("request_id", "")),
    }


def _resolve_bundle_reference(reference: Any, *, bundle_root: Path | None) -> Path:
    if isinstance(reference, Mapping):
        if reference.get("path"):
            reference = reference["path"]
        elif reference.get("request_id"):
            reference = str(reference["request_id"])
        else:
            raise ValueError("bundle reference object needs path or request_id")
    if not isinstance(reference, str) or not reference:
        raise ValueError("bundle reference must be a path or request id")
    candidate = Path(reference).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    if bundle_root is None:
        raise ValueError(f"bundle reference {reference!r} is not a file and no bundle_root was provided")
    matches: list[Path] = []
    for path in bundle_root.rglob("terminal-output-bundle.json"):
        try:
            bundle = _read_json(path)
        except ValueError:
            continue
        if str(bundle.get("request_id")) == reference:
            matches.append(path)
    if len(matches) != 1:
        raise ValueError(f"request id {reference!r} resolved to {len(matches)} bundles")
    return matches[0].resolve()


def load_cases(path: Path) -> tuple[list[dict[str, Any]], Path | None]:
    payload = _read_json(path)
    if isinstance(payload, list):
        return [dict(_as_mapping(item, "case")) for item in payload], None
    payload_map = _as_mapping(payload, "cases document")
    values = payload_map.get("cases")
    if not isinstance(values, list) or not values:
        raise ValueError("cases document must contain a non-empty cases list")
    bundle_root = payload_map.get("bundle_root")
    return [dict(_as_mapping(item, "case")) for item in values], None if bundle_root is None else Path(str(bundle_root)).expanduser().resolve()


def _reference_bins(box: Sequence[float], width: int, height: int) -> list[int]:
    if len(box) != 4 or width <= 0 or height <= 0:
        raise ValueError("reference box and image dimensions are invalid")
    return [
        max(0, min(999, int(round(float(value) * 1000.0 / (width if index % 2 == 0 else height)))))
        for index, value in enumerate(box)
    ]


def _softmax_fp32(logits: Sequence[float], temperature: float = 1.0) -> list[float]:
    if not logits:
        raise ValueError("logits must be non-empty")
    if temperature <= 0 or not math.isfinite(float(temperature)):
        raise ValueError("temperature must be finite and positive")
    scaled = [float(value) / float(temperature) for value in logits]
    maximum = max(scaled)
    values = [math.exp(value - maximum) for value in scaled]
    denominator = math.fsum(values)
    return [value / denominator for value in values]


def _top_p_probability_distribution(
    logits: Sequence[float], *, temperature: float, top_p: float
) -> tuple[list[float], list[bool], list[int | None]]:
    """Apply the source categorical sampler's temperature and nucleus cutoff.

    The first token crossing the cumulative ``top_p`` boundary is retained,
    matching the standard transformers nucleus-filter convention.  Ranks are
    inherited from the temperature-scaled logits; top-p only removes tokens.
    """

    if not 0 < float(top_p) <= 1:
        raise ValueError("top_p must be in (0, 1]")
    untruncated = _softmax_fp32(logits, temperature)
    order = sorted(range(len(logits)), key=lambda index: (-float(logits[index]), index))
    keep = [False] * len(logits)
    cumulative = 0.0
    for index in order:
        keep[index] = True
        cumulative += untruncated[index]
        if cumulative >= float(top_p):
            break
    retained = [untruncated[index] if keep[index] else 0.0 for index in range(len(logits))]
    denominator = math.fsum(retained)
    if denominator <= 0:
        raise ValueError("top-p filtering retained no probability mass")
    probabilities = [value / denominator for value in retained]
    retained_rank: list[int | None] = [None] * len(logits)
    rank = 0
    for index in order:
        if keep[index]:
            rank += 1
            retained_rank[index] = rank
    return probabilities, keep, retained_rank


def summarize_coordinate_logits(
    logits: Sequence[float],
    coordinate_token_ids: Mapping[int, int],
    *,
    clean_coordinate: int,
    bad_coordinate: int,
    reference_coordinate: int,
    window_radii: Iterable[int] = (4, 8, 16, 32),
) -> dict[str, Any]:
    """Report full-vocabulary and coordinate-conditional competition."""

    values = [float(value) for value in logits]
    probabilities = _softmax_fp32(values, temperature=1.0)
    policy_probabilities, policy_included, policy_ranks = _top_p_probability_distribution(
        values, temperature=0.4, top_p=0.95
    )
    coordinate_items = sorted((int(bin_value), int(token_id)) for bin_value, token_id in coordinate_token_ids.items())
    if not coordinate_items or any(token_id < 0 or token_id >= len(values) for _, token_id in coordinate_items):
        raise ValueError("coordinate token ids are invalid for the vocabulary")
    coordinate_mass = math.fsum(probabilities[token_id] for _, token_id in coordinate_items)
    conditional = {
        bin_value: probabilities[token_id] / coordinate_mass if coordinate_mass else 0.0
        for bin_value, token_id in coordinate_items
    }
    policy_coordinate_mass = math.fsum(policy_probabilities[token_id] for _, token_id in coordinate_items)
    policy_conditional = {
        bin_value: policy_probabilities[token_id] / policy_coordinate_mass if policy_coordinate_mass else 0.0
        for bin_value, token_id in coordinate_items
    }
    sorted_full = sorted(range(len(values)), key=lambda index: values[index], reverse=True)
    full_rank = {token_id: index + 1 for index, token_id in enumerate(sorted_full)}
    sorted_coordinate = sorted(coordinate_items, key=lambda pair: values[pair[1]], reverse=True)
    coordinate_rank = {bin_value: index + 1 for index, (bin_value, _) in enumerate(sorted_coordinate)}
    entropy = -math.fsum(value * math.log(value) for value in conditional.values() if value > 0)
    policy_entropy = -math.fsum(value * math.log(value) for value in policy_conditional.values() if value > 0)
    coord_argmax = max(coordinate_items, key=lambda pair: values[pair[1]])[0]

    def token_record(coordinate: int) -> dict[str, Any]:
        if coordinate not in coordinate_token_ids:
            raise ValueError(f"coordinate {coordinate} missing from coordinate token map")
        token_id = int(coordinate_token_ids[coordinate])
        return {
            "coordinate_bin": int(coordinate),
            "token_id": token_id,
            "logit_float32": values[token_id],
            # Keep the compact v1 fields alongside the explicitly labelled
            # policy views below for downstream readers of the first probe.
            "probability_over_full_vocabulary": probabilities[token_id],
            "probability_given_coordinate_token": conditional[coordinate],
            "coordinate_token_rank": coordinate_rank[coordinate],
            "raw_temperature_1": {
                "probability_over_full_vocabulary": probabilities[token_id],
                "full_vocabulary_rank": full_rank[token_id],
                "probability_given_coordinate_token": conditional[coordinate],
                "coordinate_token_rank": coordinate_rank[coordinate],
            },
            "source_sampling_policy_temperature_0_4_top_p_0_95": {
                "probability_over_full_vocabulary_after_top_p": policy_probabilities[token_id],
                "included_by_top_p": policy_included[token_id],
                "sampling_policy_rank_if_included": policy_ranks[token_id],
                "probability_given_retained_coordinate_token": policy_conditional[coordinate],
            },
        }

    windows: dict[str, Any] = {}
    for radius in window_radii:
        radius = int(radius)
        if radius < 0:
            raise ValueError("window radii must be non-negative")
        low = max(0, reference_coordinate - radius)
        high = min(999, reference_coordinate + radius)
        bins = [bin_value for bin_value, _ in coordinate_items if low <= bin_value <= high]
        full = math.fsum(probabilities[coordinate_token_ids[bin_value]] for bin_value in bins)
        cond = math.fsum(conditional[bin_value] for bin_value in bins)
        policy_full = math.fsum(policy_probabilities[coordinate_token_ids[bin_value]] for bin_value in bins)
        policy_cond = math.fsum(policy_conditional[bin_value] for bin_value in bins)
        windows[f"plus_or_minus_{radius}_bins"] = {
            "low_inclusive": low,
            "high_inclusive": high,
            "probability_mass_over_full_vocabulary": full,
            "probability_mass_given_coordinate_token": cond,
            "source_sampling_policy_probability_mass_over_full_vocabulary_after_top_p": policy_full,
            "source_sampling_policy_probability_mass_given_retained_coordinate_token": policy_cond,
            "greedy_coordinate_argmax_inside": low <= coord_argmax <= high,
            "clean_coordinate_inside": low <= clean_coordinate <= high,
            "bad_coordinate_inside": low <= bad_coordinate <= high,
        }
    endpoints = (clean_coordinate, bad_coordinate, reference_coordinate)
    low, high = min(endpoints), max(endpoints)
    bins = [bin_value for bin_value, _ in coordinate_items if low <= bin_value <= high]
    windows["clean_bad_reference_union"] = {
        "low_inclusive": low,
        "high_inclusive": high,
        "probability_mass_over_full_vocabulary": math.fsum(probabilities[coordinate_token_ids[b]] for b in bins),
        "probability_mass_given_coordinate_token": math.fsum(conditional[b] for b in bins),
        "source_sampling_policy_probability_mass_over_full_vocabulary_after_top_p": math.fsum(policy_probabilities[coordinate_token_ids[b]] for b in bins),
        "source_sampling_policy_probability_mass_given_retained_coordinate_token": math.fsum(policy_conditional[b] for b in bins),
    }
    return {
        "distribution_semantics": {
            "raw_temperature_1": "Full vocabulary softmax at temperature 1.0; no truncation.",
            "source_sampling_policy_temperature_0_4_top_p_0_95": "Temperature 0.4 followed by top-p 0.95 truncation over the full vocabulary, then renormalization. Temperature changes sharpness but not rank; top-p removes tokens.",
        },
        "coordinate_vocabulary_probability_mass": coordinate_mass,
        "coordinate_entropy_nats": entropy,
        "source_sampling_policy_coordinate_vocabulary_probability_mass_after_top_p": policy_coordinate_mass,
        "source_sampling_policy_coordinate_entropy_nats_after_top_p": policy_entropy,
        "coordinate_argmax_bin": int(coord_argmax),
        "coordinate_argmax": token_record(int(coord_argmax)),
        "clean": token_record(clean_coordinate),
        "bad": token_record(bad_coordinate),
        "reference": token_record(reference_coordinate),
        "reference_windows": windows,
    }


def pair_coordinate_branch_reports(
    clean_prefix_reports: Sequence[Mapping[str, Any]],
    degraded_prefix_reports: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Pair the four conditional reports without dropping later-slot drift.

    Each list is produced by ``_score_branch_trajectory``.  Keeping the
    pairing explicit prevents a first-divergence-only artifact from hiding a
    later ``y2`` or ``x2`` change that occurs after an earlier harmless
    coordinate difference.
    """

    if len(clean_prefix_reports) != len(COORDINATE_SLOT_NAMES):
        raise ValueError("clean trajectory must contain exactly four coordinate reports")
    if len(degraded_prefix_reports) != len(COORDINATE_SLOT_NAMES):
        raise ValueError("degraded trajectory must contain exactly four coordinate reports")
    paired: list[dict[str, Any]] = []
    for expected_slot, (clean, degraded) in enumerate(zip(clean_prefix_reports, degraded_prefix_reports)):
        clean_slot = int(clean.get("slot_index", -1))
        degraded_slot = int(degraded.get("slot_index", -1))
        if clean_slot != expected_slot or degraded_slot != expected_slot:
            raise ValueError("coordinate trajectory reports are not ordered by slot")
        if str(clean.get("slot")) != COORDINATE_SLOT_NAMES[expected_slot] or str(degraded.get("slot")) != COORDINATE_SLOT_NAMES[expected_slot]:
            raise ValueError("coordinate trajectory reports contain an unexpected slot name")
        paired.append(
            {
                "slot_index": expected_slot,
                "slot": COORDINATE_SLOT_NAMES[expected_slot],
                "clean_prefix": dict(clean),
                "degraded_prefix": dict(degraded),
            }
        )
    return paired


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True, help="JSON case document")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON artifact")
    parser.add_argument("--infer-config", type=Path, default=DEFAULT_INFER_CONFIG)
    parser.add_argument("--device", default="cuda:0", help="Torch device, for example cuda:1 or cpu")
    # FP32 is the safe default for a probability-comparison probe.  BF16 is
    # opt-in only; never silently replace a caller's requested dtype.
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="fp32")
    parser.add_argument(
        "--allow-counterfactual-model-composition",
        action="store_true",
        help=(
            "Explicitly allow source bundles from a different model composition; "
            "all non-composition pair checks remain strict."
        ),
    )
    parser.add_argument(
        "--counterfactual-purpose",
        help="Plain-English purpose recorded with an explicitly counterfactual model comparison.",
    )
    parser.add_argument("--force", action="store_true", help="Allow replacing an existing output")
    args = parser.parse_args()
    if args.allow_counterfactual_model_composition and not args.counterfactual_purpose:
        parser.error("--counterfactual-purpose is required with --allow-counterfactual-model-composition")
    if args.counterfactual_purpose and not args.allow_counterfactual_model_composition:
        parser.error("--counterfactual-purpose requires --allow-counterfactual-model-composition")
    return args


def _build_chat_text(processor: Any, config: Any, image_path: Path) -> str:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": config.template.prompt.system}]},
        {"role": "user", "content": [{"type": "image", "image": str(image_path)}, {"type": "text", "text": config.template.prompt.user}]},
    ]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if not isinstance(rendered, str) or not rendered:
        raise RuntimeError("processor.apply_chat_template did not return a non-empty prompt")
    return rendered


def _forward_last_logits(session: Any, native_inputs: Mapping[str, Any], generated_prefix: Sequence[int]) -> list[float]:
    """Run one exact teacher-forced branch prefix and return CPU FP32 logits."""

    import torch

    device = native_inputs["input_ids"].device
    prefix = torch.tensor([list(map(int, generated_prefix))], dtype=torch.long, device=device)
    model_inputs = dict(native_inputs)
    model_inputs["input_ids"] = torch.cat((native_inputs["input_ids"], prefix), dim=1)
    if "attention_mask" in model_inputs:
        model_inputs["attention_mask"] = torch.cat((native_inputs["attention_mask"], torch.ones_like(prefix)), dim=1)
    with torch.inference_mode():
        output = session._model(**model_inputs)
    return output.logits[0, -1].detach().to(device="cpu", dtype=torch.float32).tolist()


def _score_branch_trajectory(
    *,
    session: Any,
    native_inputs: Mapping[str, Any],
    bundle: Mapping[str, Any],
    clean_bins: Sequence[int],
    bad_bins: Sequence[int],
    reference_bins: Sequence[int],
    coordinate_token_ids: Mapping[int, int],
    branch_name: str,
) -> list[dict[str, Any]]:
    """Score all four slots while preserving the branch prefix at each slot."""

    row = _row_receipt(bundle)
    _, steps = _coordinate_slots(bundle, row)
    generated = _generated_tokens(bundle)
    coordinates = [int(value) for value in row["coordinate_bins"]]
    records: list[dict[str, Any]] = []
    for slot, step in enumerate(steps):
        if step < 0 or step >= len(generated):
            raise ValueError(f"{branch_name} coordinate step {step} is outside generated token ids")
        logits = _forward_last_logits(session, native_inputs, generated[:step])
        records.append(
            {
                "slot_index": slot,
                "slot": COORDINATE_SLOT_NAMES[slot],
                "generation_step": int(step),
                "branch_coordinate_bin": coordinates[slot],
                "conditioned_generated_prefix_token_count": int(step),
                "conditioned_generated_prefix_token_ids": generated[:step],
                "coordinate_report": summarize_coordinate_logits(
                    logits,
                    coordinate_token_ids,
                    clean_coordinate=int(clean_bins[slot]),
                    bad_coordinate=int(bad_bins[slot]),
                    reference_coordinate=int(reference_bins[slot]),
                ),
            }
        )
    return records


def _score_case(
    case: Mapping[str, Any],
    clean_bundle: Mapping[str, Any],
    bad_bundle: Mapping[str, Any],
    *,
    session: Any,
    processor: Any,
    config: Any,
    allow_counterfactual_model_composition: bool = False,
    counterfactual_purpose: str | None = None,
) -> dict[str, Any]:
    import torch
    from PIL import Image
    from src.config.fingerprint import sha256_file
    from src.inference.backend import DecodeRequest, GenerationPolicy

    identity = validate_bundle_pair(
        case,
        clean_bundle,
        bad_bundle,
        expected_model_composition=expected_model_composition_from_config(config),
        allow_counterfactual_model_composition=allow_counterfactual_model_composition,
        counterfactual_purpose=counterfactual_purpose,
    )
    review_image_path = Path(str(case.get("image_path", ""))).expanduser().resolve(strict=True)
    replay_image_path = Path(str(case.get("replay_image_path") or review_image_path)).expanduser().resolve(strict=True)
    with Image.open(replay_image_path) as image:
        width, height = (int(value) for value in image.size)
    # The decode request contract and the historical execution bundle use the
    # source image *file* SHA-256.  The backend separately records the
    # canonical decoded RGB hash as executed-media evidence.
    image_sha = sha256_file(replay_image_path)
    clean_evidence = _as_mapping(clean_bundle.get("execution_evidence"), "clean execution_evidence")
    expected_sha = str(clean_evidence.get("source_image_sha256") or _as_mapping(clean_bundle.get("scheduled_request"), "scheduled_request").get("image_sha256", ""))
    if expected_sha and expected_sha != image_sha:
        raise ValueError(
            f"image bytes differ from clean bundle for case {case.get('name')}; "
            "set case.replay_image_path to the exact replay image bytes while "
            "keeping case.image_path as the review/display image"
        )
    chat_text = _build_chat_text(processor, config, replay_image_path)
    bundle_prompt = _bundle_prompt_evidence(clean_bundle)
    payload_json = bundle_prompt.get("prompt_record_payload_json")
    if isinstance(payload_json, str):
        payload = _read_json_text(payload_json)
        recorded_text = payload.get("full_chat_text") if isinstance(payload, Mapping) else None
        if isinstance(recorded_text, str) and recorded_text != chat_text:
            raise ValueError("processor-reconstructed full chat text differs from stored prompt")
    prompt_ids = tuple(int(value) for value in _as_mapping(clean_bundle.get("decode_result"), "decode_result")["prompt_token_ids"])
    request = DecodeRequest(
        request_id=f"fixed-prompt-coordinate:{case.get('name', 'case')}",
        chat_text=chat_text,
        input_prompt_token_ids=prompt_ids,
        expected_executed_prompt_token_ids=prompt_ids,
        image_path=str(replay_image_path),
        declared_image_width=width,
        declared_image_height=height,
        decoded_image_width=width,
        decoded_image_height=height,
        image_sha256=image_sha,
        generation_policy=GenerationPolicy(max_new_tokens=8, repetition_penalty=1.0),
    )
    native_inputs, executed_ids, observed_grids, media_sha = session._materialize_native_inputs((request,))
    if tuple(executed_ids[0]) != prompt_ids:
        raise RuntimeError("processor executed prompt token ids differ from stored bundle ids")
    tokenizer = session._tokenizer
    coordinate_tokens = {
        coordinate: int(tokenizer.convert_tokens_to_ids(f"<|coord_{coordinate}|>"))
        for coordinate in range(1000)
    }
    ref = _reference_bins(case["reference_box_xyxy"], width, height)
    clean_bins = identity["clean_coordinate_bins"]
    bad_bins = identity["bad_coordinate_bins"]
    clean_trajectory = _score_branch_trajectory(
        session=session,
        native_inputs=native_inputs,
        bundle=clean_bundle,
        clean_bins=clean_bins,
        bad_bins=bad_bins,
        reference_bins=ref,
        coordinate_token_ids=coordinate_tokens,
        branch_name="clean",
    )
    degraded_trajectory = _score_branch_trajectory(
        session=session,
        native_inputs=native_inputs,
        bundle=bad_bundle,
        clean_bins=clean_bins,
        bad_bins=bad_bins,
        reference_bins=ref,
        coordinate_token_ids=coordinate_tokens,
        branch_name="degraded",
    )
    paired_trajectory = pair_coordinate_branch_reports(clean_trajectory, degraded_trajectory)
    slot = int(identity["first_differing_coordinate_slot_index"])
    return {
        **identity,
        "image_path": str(review_image_path),
        "replay_image_path": str(replay_image_path),
        "image_dimensions": {"width": width, "height": height},
        "reference_box_xyxy_pixels": [float(value) for value in case["reference_box_xyxy"]],
        "reference_box_xyxy_coordinate_bins": ref,
        "observed_prompt_token_ids_sha256": _sha256_json(list(executed_ids[0])),
        "observed_image_grid_thw": None if observed_grids[0] is None else list(observed_grids[0]),
        "executed_media_sha256": media_sha[0],
        "coordinate_reports_by_branch": {
            "clean_prefix": clean_trajectory,
            "degraded_prefix": degraded_trajectory,
        },
        "coordinate_reports_by_slot": paired_trajectory,
        "first_differing_coordinate_report": {
            "slot": COORDINATE_SLOT_NAMES[slot],
            "clean_prefix": clean_trajectory[slot],
            "degraded_prefix": degraded_trajectory[slot],
        },
    }


def _read_json_text(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"stored prompt_record_payload_json is invalid JSON: {exc}") from exc


def main() -> int:
    args = _parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {output}; pass --force")
    cases, bundle_root = load_cases(args.cases.expanduser().resolve(strict=True))
    if not cases:
        raise SystemExit("cases document is empty")
    bundles: list[tuple[dict[str, Any], Mapping[str, Any], Mapping[str, Any], Path, Path]] = []
    for case in cases:
        root = bundle_root
        clean_path = _resolve_bundle_reference(
            case.get("clean_bundle")
            or case.get("clean_bundle_path")
            or case.get("clean_request_id"),
            bundle_root=root,
        )
        bad_path = _resolve_bundle_reference(
            case.get("degraded_bundle")
            or case.get("bad_bundle")
            or case.get("bad_bundle_path")
            or case.get("bad_request_id"),
            bundle_root=root,
        )
        bundles.append((case, _read_json(clean_path), _read_json(bad_path), clean_path, bad_path))
    import torch
    from src.config.fingerprint import sha256_json
    from src.config.inference import load_infer_config
    from src.inference.backend import open_backend_session
    from src.inference.runtime import assemble_frontend

    infer_config_path = args.infer_config.expanduser().resolve(strict=True)
    resolved = load_infer_config(infer_config_path)
    config = resolved.config.model_copy(update={"model": resolved.config.model.model_copy(update={"dtype": "fp32" if args.dtype == "fp32" else "bf16"})})
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
    frontend = assemble_frontend(config, generation_config_fingerprint=sha256_json(config.generation.model_dump(mode="json")))
    results = []
    with open_backend_session(frontend.launch) as session:
        for case, clean_bundle, bad_bundle, clean_path, bad_path in bundles:
            result = _score_case(
                case,
                clean_bundle,
                bad_bundle,
                session=session,
                processor=frontend.qwen.processor,
                config=config,
                allow_counterfactual_model_composition=args.allow_counterfactual_model_composition,
                counterfactual_purpose=args.counterfactual_purpose,
            )
            result["clean_bundle_path"] = str(clean_path)
            result["bad_bundle_path"] = str(bad_path)
            results.append(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "config": {
            "infer_config_path": str(infer_config_path),
            "resolved_fingerprint": resolved.fingerprint,
            "dtype": config.model.dtype,
            "device": str(args.device),
        },
        "executed_model_composition": expected_model_composition_from_config(config),
        "counterfactual_model_composition": {
            "enabled": bool(args.allow_counterfactual_model_composition),
            "purpose": args.counterfactual_purpose,
            "declaration": (
                "Source clean/degraded bundles are intentionally rescored with the current "
                "inference model composition."
                if args.allow_counterfactual_model_composition
                else "Source bundles and current inference model composition must match."
            ),
        },
        "case_count": len(results),
        "cases": results,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
