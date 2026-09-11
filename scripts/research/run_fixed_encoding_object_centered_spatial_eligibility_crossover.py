#!/usr/bin/env python3
"""Run the fixed-encoding object-centered spatial-eligibility crossover.

This is an experiment-local teacher-forced scorer.  It encodes each image once,
replays the captured visual feature streams, and changes only the eligibility
of already-computed image-token *keys* in the language decoder attention mask.
It deliberately does not modify the shared model/backend or run generation.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.visual_support_counterfactual import (  # noqa: E402
    FeatureBundle,
    FeatureReplayController,
    build_merged_support_mask,
    capture_feature_bundle,
    feature_bundle_fingerprint,
    tensor_sha256,
    validate_feature_layout,
)


OBJECT_REF_START = 151646
OBJECT_REF_END = 151647
BOX_START = 151648
BOX_END = 151649
COORDINATE_TOKEN_START = 151670
COORDINATE_TOKEN_END_EXCLUSIVE = 152670

DEFAULT_CONFIG = Path(
    "/data/CoordExp/.worktrees/research-probes/configs/coordexp_infras/infer/"
    "qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_step4887_val200.yaml"
)
DEFAULT_SOURCE_JSONL = Path(
    "/data/CoordExp/.worktrees/coordexp-infras/outputs/coordexp_swift/infer/"
    "val200_inputs/coco_val200_len12000.rebased_images.coord.jsonl"
)
DEFAULT_LEDGER = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-13-spatial-scope-history-disentanglement/readiness-v2/"
    "audit-augmented-ledger.jsonl"
)
DEFAULT_IMAGE_IDS = ("139", "632", "2299", "9400", "12120", "12639")

FROZEN_TARGETS: dict[str, dict[str, Any]] = {
    "139": {"annotation_id": "1669970", "category": "vase", "stratum": "different-category"},
    "632": {"annotation_id": "1661908", "category": "book", "stratum": "same-category"},
    "2299": {"annotation_id": "2008221", "category": "person", "stratum": "same-category"},
    "9400": {"annotation_id": "1292246", "category": "person", "stratum": "different-category"},
    "12120": {"annotation_id": "2030099", "category": "person", "stratum": "different-category"},
    "12639": {"annotation_id": "543629", "category": "person", "stratum": "same-category"},
}


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256_ints(values: Sequence[int]) -> str:
    return _sha256_json([int(value) for value in values])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve(strict=True).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_float_list(values: Iterable[Any]) -> list[float]:
    return [float(value) for value in values]


def build_causal_key_eligibility_mask(
    *,
    sequence_length: int,
    image_key_positions: Sequence[int],
    eligible_image_positions: Sequence[int] | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return a boolean [1,1,S,S] causal mask with image-key gating.

    ``True`` means the key is visible.  Non-image keys remain visible to every
    query; only image-token *keys* are restricted.  This is intentionally not a
    pixel or feature mask.
    """

    length = int(sequence_length)
    if length <= 0:
        raise ValueError("sequence_length must be positive")
    image = {int(value) for value in image_key_positions}
    if any(value < 0 or value >= length for value in image):
        raise ValueError("image key position is outside the sequence")
    eligible = image if eligible_image_positions is None else {int(value) for value in eligible_image_positions}
    if not eligible.issubset(image):
        raise ValueError("eligible image keys must be a subset of image keys")
    causal = torch.tril(torch.ones((length, length), dtype=torch.bool, device=device))
    blocked = image.difference(eligible)
    if blocked:
        blocked_tensor = torch.tensor(sorted(blocked), dtype=torch.long, device=device)
        causal[:, blocked_tensor] = False
    return causal.unsqueeze(0).unsqueeze(0)


def derive_explicit_position_ids(
    model: Any,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_grid_thw: torch.Tensor,
) -> torch.Tensor:
    """Derive Qwen's position identifiers once from the ordinary 2-D mask."""

    owner = getattr(model, "model", model)
    if not callable(getattr(owner, "get_rope_index", None)):
        raise TypeError("Qwen model does not expose get_rope_index")
    with torch.no_grad():
        position_ids, _rope_deltas = owner.get_rope_index(
            input_ids,
            image_grid_thw,
            None,
            attention_mask=attention_mask,
        )
    if not isinstance(position_ids, torch.Tensor) or position_ids.ndim != 3:
        raise ValueError("Qwen get_rope_index returned invalid position_ids")
    return position_ids.detach().clone()


def translate_mask_to_center(
    mask: torch.Tensor,
    *,
    temporal: int,
    merged_height: int,
    merged_width: int,
    source_center: tuple[int, int],
    destination_center: tuple[int, int],
) -> torch.Tensor:
    """Translate an exact binary support shape between merged-grid centers."""

    expected = int(temporal) * int(merged_height) * int(merged_width)
    if mask.ndim != 1 or int(mask.numel()) != expected or mask.dtype != torch.bool:
        raise ValueError("mask does not match merged-grid shape")
    dr = int(destination_center[0]) - int(source_center[0])
    dc = int(destination_center[1]) - int(source_center[1])
    source = mask.reshape(int(temporal), int(merged_height), int(merged_width))
    result = torch.zeros_like(source)
    for frame in range(int(temporal)):
        rows, cols = torch.where(source[frame])
        for row, col in zip(rows.tolist(), cols.tolist(), strict=True):
            new_row, new_col = int(row) + dr, int(col) + dc
            if not (0 <= new_row < int(merged_height) and 0 <= new_col < int(merged_width)):
                raise ValueError("translated support clips the merged grid")
            result[frame, new_row, new_col] = True
    if int(result.sum()) != int(mask.sum()):
        raise ValueError("translated support changed selected-token count")
    return result.reshape(-1)


def object_center_cell(
    bbox_xyxy: Sequence[float], *, image_width: int, image_height: int,
    merged_height: int, merged_width: int,
) -> tuple[int, int]:
    """Map a pixel-space box center to one merged-grid cell."""

    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    column = min(int(merged_width) - 1, max(0, int(cx / max(float(image_width), 1.0) * merged_width)))
    row = min(int(merged_height) - 1, max(0, int(cy / max(float(image_height), 1.0) * merged_height)))
    return row, column


def build_translated_competitor_mask(
    target_mask: torch.Tensor,
    *,
    target_bbox_xyxy: Sequence[float],
    competitor_bbox_xyxy: Sequence[float],
    image_width: int,
    image_height: int,
    temporal: int,
    merged_height: int,
    merged_width: int,
) -> torch.Tensor:
    source_center = object_center_cell(
        target_bbox_xyxy,
        image_width=image_width,
        image_height=image_height,
        merged_height=merged_height,
        merged_width=merged_width,
    )
    destination_center = object_center_cell(
        competitor_bbox_xyxy,
        image_width=image_width,
        image_height=image_height,
        merged_height=merged_height,
        merged_width=merged_width,
    )
    competitor = translate_mask_to_center(
        target_mask,
        temporal=temporal,
        merged_height=merged_height,
        merged_width=merged_width,
        source_center=source_center,
        destination_center=destination_center,
    )
    if bool((target_mask & competitor).any()):
        raise ValueError("translated competitor support overlaps target support")
    if int(competitor.sum()) != int(target_mask.sum()):
        raise ValueError("translated competitor support count differs")
    if not bool(competitor.reshape(int(temporal), int(merged_height), int(merged_width))[:, destination_center[0], destination_center[1]].all()):
        raise ValueError("translated competitor support does not contain competitor center")
    return competitor


def build_row_tokens(
    *, description_token_ids: Sequence[int], coordinate_bins: Sequence[int]
) -> list[int]:
    """Compose the canonical object-reference and four-coordinate row."""

    if len(coordinate_bins) != 4 or any(not 0 <= int(value) <= 999 for value in coordinate_bins):
        raise ValueError("coordinate_bins must contain four values in [0,999]")
    description = [int(value) for value in description_token_ids]
    if not description:
        raise ValueError("description_token_ids must not be empty")
    return [OBJECT_REF_START, *description, OBJECT_REF_END, BOX_START, *[COORDINATE_TOKEN_START + int(v) for v in coordinate_bins], BOX_END]


def split_row_phases(row_tokens: Sequence[int], *, description_length: int) -> dict[str, list[int]]:
    """Return row-relative indices for all frozen score phases."""

    row = [int(value) for value in row_tokens]
    expected = 1 + int(description_length) + 1 + 1 + 4 + 1
    if len(row) != expected or row[0] != OBJECT_REF_START or row[-1] != BOX_END:
        raise ValueError("row tokens do not match canonical wrapper")
    desc_start = 1
    desc_end = desc_start + int(description_length)
    box_start = desc_end + 1
    coord_start = box_start + 1
    return {
        "row_entry": [0],
        "description": list(range(desc_start, desc_end)),
        "geometry": list(range(box_start, len(row))),
        "x1": [coord_start],
        "y1": [coord_start + 1],
        "x2": [coord_start + 2],
        "y2": [coord_start + 3],
        "full_row": list(range(len(row))),
    }


def first_differing_description_index(
    left: Sequence[int], right: Sequence[int], *, description_length: int
) -> int | None:
    """Return a shared lexical position, not a lexical/boundary mismatch.

    If one description is only a strict token-prefix of the other, the next
    divergence is between a lexical token and ``OBJECT_REF_END``.  That is not
    a first-differing *description-token* comparison, so this metric is left
    undefined rather than mixing two generation phases.
    """

    left_phases = split_row_phases(left, description_length=description_length)
    right_phases = split_row_phases(right, description_length=len(right) - 8)
    for left_index, right_index in zip(
        left_phases["description"], right_phases["description"], strict=False
    ):
        if int(left[left_index]) != int(right[right_index]):
            return left_index
    return None


def score_row_log_likelihoods(
    logits: torch.Tensor,
    *,
    prefix_length: int,
    row_tokens: Sequence[int],
    description_length: int,
    terminal_token_id: int | None = None,
) -> dict[str, Any]:
    """Score teacher-forced row phases from full-vocabulary logits."""

    if logits.ndim != 2 or logits.shape[0] < prefix_length + len(row_tokens) - 1:
        raise ValueError("logits do not cover the teacher-forced row")
    row = [int(value) for value in row_tokens]
    log_probs = torch.log_softmax(logits.to(dtype=torch.float32), dim=-1)
    selected = torch.stack(
        [log_probs[prefix_length + index - 1, token] for index, token in enumerate(row)], dim=0
    )
    phases = split_row_phases(row, description_length=description_length)
    result: dict[str, Any] = {}
    for phase, indices in phases.items():
        values = selected[torch.tensor(indices, dtype=torch.long, device=selected.device)]
        result[phase] = {"sum": float(values.sum()), "mean": float(values.mean()), "count": len(indices)}
    result["token_log_probabilities"] = [float(value) for value in selected.detach().cpu().tolist()]
    row_steps = torch.stack(
        [logits.to(dtype=torch.float32)[prefix_length + index - 1] for index in range(len(row))]
    )
    row_token_tensor = torch.tensor(row, dtype=torch.long, device=row_steps.device)
    row_token_logits = row_steps.gather(1, row_token_tensor.unsqueeze(1)).squeeze(1)
    result["selected_token_ranks"] = [
        int(value)
        for value in (1 + (row_steps > row_token_logits.unsqueeze(1)).sum(dim=1)).detach().cpu().tolist()
    ]
    result["top_prediction_token_ids"] = [
        int(torch.argmax(logits.to(dtype=torch.float32)[prefix_length + index - 1]).item())
        for index in range(len(row))
    ]
    if terminal_token_id is not None:
        terminal = log_probs[prefix_length - 1, int(terminal_token_id)]
        result["row_entry_vs_terminal"] = {
            "row_entry_log_probability": float(selected[0]),
            "terminal_log_probability": float(terminal),
            "row_entry_minus_terminal": float(selected[0] - terminal),
        }
    return result


def compute_crossover_and_release(
    *,
    full_scores: Mapping[str, Mapping[str, Mapping[str, float]]],
    target_scores: Mapping[str, Mapping[str, Mapping[str, float]]],
    competitor_scores: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> dict[str, Any]:
    phases = sorted(set(full_scores) & set(target_scores) & set(competitor_scores))
    output: dict[str, Any] = {}
    for phase in phases:
        target_full = float(full_scores[phase]["target"]["mean"])
        competitor_full = float(full_scores[phase]["competitor"]["mean"])
        target_target = float(target_scores[phase]["target"]["mean"])
        competitor_target = float(target_scores[phase]["competitor"]["mean"])
        target_competitor = float(competitor_scores[phase]["target"]["mean"])
        competitor_competitor = float(competitor_scores[phase]["competitor"]["mean"])
        gamma_target = target_target - competitor_target
        gamma_competitor = target_competitor - competitor_competitor
        output[phase] = {
            "gamma_full": target_full - competitor_full,
            "gamma_target": gamma_target,
            "gamma_competitor": gamma_competitor,
            "crossover": gamma_target - gamma_competitor,
            "target_release": target_target - target_full,
            "competitor_release": competitor_competitor - competitor_full,
        }
    return output


def classify_case(
    crossover: Mapping[str, Mapping[str, float]], *, no_op_drift: float,
    require_description_reversal: bool | None = None,
) -> str:
    row_phase = crossover.get("full_row", {})
    geometry_phase = crossover.get("geometry", {})
    row = float(row_phase.get("crossover", 0.0))
    geometry = float(geometry_phase.get("crossover", 0.0))
    owner_release = max(
        float(crossover.get("full_row", {}).get("target_release", 0.0)),
        float(crossover.get("full_row", {}).get("competitor_release", 0.0)),
    )
    different_description = (
        "first_differing_description" in crossover and "description" in crossover
        if require_description_reversal is None
        else bool(require_description_reversal)
    )
    effect_floor = 10 * float(no_op_drift)
    threshold_passed = (
        row >= max(0.10, effect_floor)
        and geometry >= max(0.10, effect_floor)
        and owner_release >= max(0.05, effect_floor)
    )
    full_row_reversal = (
        float(row_phase.get("gamma_target", 0.0)) > effect_floor
        and float(row_phase.get("gamma_competitor", 0.0)) < -effect_floor
    )
    geometry_reversal = (
        float(geometry_phase.get("gamma_target", 0.0)) > effect_floor
        and float(geometry_phase.get("gamma_competitor", 0.0)) < -effect_floor
    )
    if threshold_passed and full_row_reversal and geometry_reversal:
        if different_description:
            description_reversal = (
                float(crossover["description"]["gamma_target"]) > effect_floor
                and float(crossover["description"]["gamma_competitor"]) < -effect_floor
            )
            first_difference = crossover.get("first_differing_description")
            first_difference_reversal = first_difference is None or (
                float(first_difference["gamma_target"]) > effect_floor
                and float(first_difference["gamma_competitor"]) < -effect_floor
            )
            if not description_reversal or not first_difference_reversal:
                return "route_to_phase_specific_discriminator"
        return "promote_bounded_free_row_switch_replay"
    if row < 0.10:
        return "close_hard_post_vision_image_token_key_eligibility"
    return "inconclusive"


def classify_panel(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the frozen six-case decision rule without promoting one case alone."""

    invalid = [
        str(item.get("image_id"))
        for item in results
        if item.get("classification") == "invalid_no_op_trust_gate"
    ]
    eligible = [item for item in results if isinstance(item.get("crossover"), Mapping)]
    promoted = [
        item
        for item in eligible
        if item.get("classification") == "promote_bounded_free_row_switch_replay"
    ]
    phase_specific = [
        item
        for item in eligible
        if item.get("classification") == "route_to_phase_specific_discriminator"
    ]
    closed = [
        item
        for item in eligible
        if item.get("classification")
        == "close_hard_post_vision_image_token_key_eligibility"
    ]
    summary: dict[str, Any] = {
        "requested_case_count": len(results),
        "eligible_case_count": len(eligible),
        "invalid_no_op_case_ids": invalid,
        "promoting_case_ids": [str(item.get("image_id")) for item in promoted],
        "phase_specific_case_ids": [str(item.get("image_id")) for item in phase_specific],
        "closed_case_ids": [str(item.get("image_id")) for item in closed],
    }
    if invalid and not eligible:
        summary["classification"] = "invalid_no_op_trust_gate"
        return summary
    if len(promoted) >= 2:
        strongest = max(
            promoted,
            key=lambda item: float(item["crossover"]["full_row"]["crossover"]),
        )
        summary.update(
            {
                "classification": "promote_one_bounded_free_row_switch_replay",
                "strongest_case_image_id": str(strongest.get("image_id")),
            }
        )
        return summary
    if len(phase_specific) >= 2:
        summary["classification"] = "route_to_one_phase_specific_discriminator"
        return summary
    if len(eligible) >= 2 and len(closed) == len(eligible):
        summary["classification"] = "close_hard_post_vision_image_token_key_eligibility"
        return summary
    summary["classification"] = (
        "inconclusive_isolated_owner_specific_case"
        if len(promoted) == 1
        else "inconclusive_mixed_or_insufficient_panel"
    )
    return summary


def select_highest_unrestricted_competitor(
    candidates: Sequence[Mapping[str, Any]], *, target_mean_full_row: float
) -> Mapping[str, Any] | None:
    """Choose B before restricted scores, requiring unrestricted B > A."""

    eligible = [
        candidate
        for candidate in candidates
        if float(candidate["unrestricted_mean_full_row"]) > float(target_mean_full_row)
    ]
    return max(eligible, key=lambda candidate: float(candidate["unrestricted_mean_full_row"])) if eligible else None


def assess_noop_trust_gate(
    *,
    implicit_token_log_probs: Sequence[float],
    explicit_token_log_probs: Sequence[float],
    all_allowed_token_log_probs: Sequence[float],
    implicit_ranks: Sequence[int],
    explicit_ranks: Sequence[int],
    all_allowed_ranks: Sequence[int],
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Verify explicit-position and all-allowed 4-D paths are no-ops."""

    def drift(left: Sequence[float], right: Sequence[float]) -> float:
        if len(left) != len(right):
            raise ValueError("no-op log-probability vectors have different lengths")
        return max((abs(float(a) - float(b)) for a, b in zip(left, right, strict=True)), default=0.0)

    explicit_drift = drift(implicit_token_log_probs, explicit_token_log_probs)
    all_allowed_drift = drift(implicit_token_log_probs, all_allowed_token_log_probs)
    ranks_equal = (
        list(map(int, implicit_ranks))
        == list(map(int, explicit_ranks))
        == list(map(int, all_allowed_ranks))
    )
    return {"passed": bool(ranks_equal and explicit_drift <= tolerance and all_allowed_drift <= tolerance), "token_ranks_equal": ranks_equal, "explicit_max_abs_logprob_drift": explicit_drift, "all_allowed_max_abs_logprob_drift": all_allowed_drift, "tolerance": float(tolerance)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infer-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-jsonl", type=Path, default=DEFAULT_SOURCE_JSONL)
    parser.add_argument("--audit-ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-ids", nargs="+", default=list(DEFAULT_IMAGE_IDS))
    parser.add_argument("--halo", type=int, default=1)
    return parser


def _load_ledger(path: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    with path.expanduser().resolve(strict=True).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("final_state") not in {"accepted", "crowd"}:
                continue
            result.setdefault(str(item.get("image_id")), []).append(item)
    return result


def _object_pixel_box(raw_object: Any, ledger_by_id: Mapping[str, Any], *, width: int, height: int) -> list[float]:
    ledger = ledger_by_id.get(str(raw_object.object_id))
    if ledger is None:
        ledger = ledger_by_id.get(f"coco-ann:{raw_object.object_id}")
    if isinstance(ledger, Mapping) and isinstance(ledger.get("source_canvas_box_xyxy"), list):
        return _as_float_list(ledger["source_canvas_box_xyxy"])
    bins = [int(value) for value in raw_object.bbox]
    return [bins[0] * width / 999.0, bins[1] * height / 999.0, bins[2] * width / 999.0, bins[3] * height / 999.0]


def _row_token_ids(tokenizer: Any, raw_object: Any) -> list[int]:
    description = tokenizer.encode(str(raw_object.description), add_special_tokens=False)
    return build_row_tokens(description_token_ids=description, coordinate_bins=raw_object.bbox)


def _score_model_sequence(
    *, model: Any, model_inputs: Mapping[str, Any], input_ids: torch.Tensor,
    attention_mask: torch.Tensor, position_ids: torch.Tensor | None,
    custom_mask: torch.Tensor | None, features: FeatureBundle,
    merge_size: int, grid_thw: Sequence[int],
) -> torch.Tensor:
    replay = FeatureReplayController(
        model=model,
        recipient=features,
        donor=features,
        grid_thw=grid_thw,
        merge_size=merge_size,
        mode="clean",
    )
    device = input_ids.device
    kwargs = {
        key: value.to(device=device) if isinstance(value, torch.Tensor) else value
        for key, value in model_inputs.items()
    }
    kwargs.update({"input_ids": input_ids, "attention_mask": attention_mask, "use_cache": False, "logits_to_keep": 0, "return_dict": True})
    if position_ids is not None:
        kwargs["position_ids"] = position_ids
    if custom_mask is not None:
        kwargs["attention_mask"] = custom_mask
    with replay:
        with torch.inference_mode():
            output = model(**kwargs)
    replay.validate_completed(expected_feature_calls=1)
    return output.logits[0].detach().to(device="cpu", dtype=torch.float32).contiguous()


def run(args: argparse.Namespace) -> dict[str, Any]:
    from src.config.fingerprint import sha256_json
    from src.config.inference import load_infer_config
    from src.data import load_raw_examples
    from src.inference.image_plan import materialize_image_plan_batch, verify_processor_model_vision_parity
    from src.inference.pipeline import _processor_config, _template_config
    from src.inference.prompt import build_prompt_record
    from src.inference.runtime import assemble_runtime

    if int(args.halo) != 1:
        raise SystemExit("this frozen unit requires halo=1")
    from scripts.research.run_sampled_rescue_transition import _temporary_cwd

    config_path = Path(args.infer_config).resolve(strict=True)
    with _temporary_cwd(config_path.parents[3]):
        resolved = load_infer_config(config_path)
    config = resolved.config.model_copy(update={"model": resolved.config.model.model_copy(update={"dtype": "fp32"})})
    raw_rows = load_raw_examples(Path(args.source_jsonl).resolve(strict=True))
    raw_by_image = {
        str(raw.metadata["source"]["image_id"]): raw
        for raw in raw_rows
        if isinstance(raw.metadata.get("source"), Mapping)
        and raw.metadata["source"].get("image_id") is not None
    }
    ledger = _load_ledger(Path(args.audit_ledger))
    runtime = assemble_runtime(config, source_gate_root=Path(args.infer_config).resolve(strict=True).parents[3])
    qwen = runtime.qwen
    qwen.model.eval()
    verify_processor_model_vision_parity(processor_identity=qwen.processor_identity, model_config=qwen.model.config)
    image_token_id = getattr(qwen.model.config, "image_token_id", None)
    if image_token_id is None:
        image_token_id = qwen.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if image_token_id is None or int(image_token_id) < 0:
        raise SystemExit("could not resolve Qwen image placeholder token id")
    template = _template_config(resolved.config)
    results: list[dict[str, Any]] = []
    for image_id in [str(value) for value in args.image_ids]:
        raw = raw_by_image.get(image_id)
        if raw is None:
            results.append({"image_id": image_id, "classification": "inconclusive", "reason": "source image absent"})
            continue
        target_spec = FROZEN_TARGETS.get(image_id)
        if target_spec is None:
            raise SystemExit(f"image {image_id} is not in the frozen target cohort")
        objects_by_id = {str(obj.object_id): obj for obj in raw.objects}
        target = objects_by_id.get(str(target_spec["annotation_id"]))
        if target is None:
            raise SystemExit(f"target annotation {target_spec['annotation_id']} absent for image {image_id}")
        ledger_by_id = {str(item.get("object_identifier", "")): item for item in ledger.get(image_id, [])}
        target_box = _object_pixel_box(target, ledger_by_id, width=raw.image.width, height=raw.image.height)
        base_prompt = build_prompt_record(raw, template, processor=qwen.processor, row_index=0)
        target_row = _row_token_ids(qwen.tokenizer, target)
        prompt_ids = [int(v) for v in base_prompt.prompt_token_ids]
        all_rows = [(obj, _row_token_ids(qwen.tokenizer, obj), _object_pixel_box(obj, ledger_by_id, width=raw.image.width, height=raw.image.height)) for obj in raw.objects if str(obj.object_id) != str(target.object_id)]
        image_plan = materialize_image_plan_batch([raw], components=qwen, processor_config=_processor_config(resolved.config), materialize=True, row_indices=[0])
        model_inputs = image_plan.model_inputs_by_row_id[raw.example_id]
        grid_thw = [int(value) for value in model_inputs["image_grid_thw"].reshape(-1, 3)[0].tolist()]
        merge_size = int(qwen.processor_identity.merge_size)
        features = capture_feature_bundle(qwen.model, model_inputs)
        layout = validate_feature_layout(features, features, grid_thw=grid_thw, merge_size=merge_size)
        target_support = build_merged_support_mask(bbox_xyxy=target_box, image_width=raw.image.width, image_height=raw.image.height, layout=layout, halo=1)
        target_center = object_center_cell(target_box, image_width=raw.image.width, image_height=raw.image.height, merged_height=layout.merged_height, merged_width=layout.merged_width)
        candidates: list[dict[str, Any]] = []
        for obj, row, box in all_rows:
            if (target_spec["stratum"] == "same-category") != (str(obj.description) == str(target_spec["category"])):
                continue
            try:
                competitor_mask = build_translated_competitor_mask(target_support, target_bbox_xyxy=target_box, competitor_bbox_xyxy=box, image_width=raw.image.width, image_height=raw.image.height, temporal=layout.temporal, merged_height=layout.merged_height, merged_width=layout.merged_width)
            except ValueError:
                continue
            candidates.append({"object": obj, "row": row, "box": box, "mask": competitor_mask})
        if not candidates:
            results.append({"image_id": image_id, "classification": "inconclusive", "reason": "no valid translated competitor"})
            continue
        scores = _score_candidate_pool(qwen.model, model_inputs, features, grid_thw, merge_size, prompt_ids, target_row, candidates, target_support, target_center, image_token_id=int(image_token_id))
        selected = select_highest_unrestricted_competitor(
            [
                {
                    **item,
                    "unrestricted_mean_full_row": item["competitor_score"]["full_row"]["mean"],
                }
                for item in scores
            ],
            target_mean_full_row=scores[0]["target_score"]["full_row"]["mean"] if scores else 0.0,
        )
        if selected is None:
            results.append({"image_id": image_id, "classification": "inconclusive", "reason": "no competitor outranks target", "candidate_scores": scores})
            continue
        chosen = next(item for item in candidates if str(item["object"].object_id) == str(selected["object_id"]))
        result = _run_case(qwen.model, model_inputs, features, grid_thw, merge_size, prompt_ids, target_row, chosen["row"], chosen["mask"], target_support, image_token_id=int(image_token_id), image_grid_thw=model_inputs["image_grid_thw"], terminal_token_id=qwen.tokenizer.eos_token_id)
        result.update({"image_id": image_id, "target_annotation_id": str(target.object_id), "competitor_annotation_id": str(chosen["object"].object_id), "candidate_scores": scores, "feature_fingerprint": feature_bundle_fingerprint(features), "image_grid_thw": grid_thw, "merge_size": merge_size, "target_mask_count": int(target_support.sum()), "competitor_mask_count": int(chosen["mask"].sum()), "mask_disjoint": not bool((target_support & chosen["mask"]).any())})
        results.append(result)
    receipt = {"schema_version": "fixed_encoding_object_centered_spatial_eligibility_crossover.v1", "unit_id": "2026-07-15-fixed-encoding-object-centered-spatial-eligibility-crossover", "model_dtype": "torch.float32", "source_jsonl": str(Path(args.source_jsonl).resolve()), "source_jsonl_sha256": _sha256_file(Path(args.source_jsonl)), "audit_ledger": str(Path(args.audit_ledger).resolve()), "audit_ledger_sha256": _sha256_file(Path(args.audit_ledger)), "results": results, "panel_decision": classify_panel(results), "config_sha256": sha256_json(config.model_dump(mode="json"))}
    return receipt


def _score_candidate_pool(model: Any, model_inputs: Mapping[str, Any], features: FeatureBundle, grid_thw: Sequence[int], merge_size: int, prompt_ids: Sequence[int], target_row: Sequence[int], candidates: Sequence[Mapping[str, Any]], target_mask: torch.Tensor, target_center: tuple[int, int], *, image_token_id: int) -> list[dict[str, Any]]:
    rows = [{"object": c["object"], "row": c["row"], "box": c["box"], "mask": c["mask"]} for c in candidates]
    out: list[dict[str, Any]] = []
    target_score = _score_one_row(model, model_inputs, features, grid_thw, merge_size, prompt_ids, target_row, eligible_mask=None, image_token_id=image_token_id, position_mode="implicit")
    for item in rows:
        item_score = _score_one_row(model, model_inputs, features, grid_thw, merge_size, prompt_ids, item["row"], eligible_mask=None, image_token_id=image_token_id, position_mode="implicit")
        out.append({"object_id": str(item["object"].object_id), "category": str(item["object"].description), "competitor_score": item_score, "target_score": target_score, "mask_indices": [int(v) for v in torch.where(item["mask"])[0].tolist()]})
    return out


def _score_one_row(model: Any, model_inputs: Mapping[str, Any], features: FeatureBundle, grid_thw: Sequence[int], merge_size: int, prompt_ids: Sequence[int], row: Sequence[int], *, eligible_mask: torch.Tensor | None, image_token_id: int, position_mode: str = "explicit", terminal_token_id: int | None = None) -> dict[str, Any]:
    device = next(model.parameters()).device
    ids = torch.tensor([list(prompt_ids) + [int(v) for v in row]], dtype=torch.long, device=device)
    ordinary = torch.ones_like(ids, dtype=torch.long)
    image_positions = [int(v) for v in torch.where(ids[0] == int(image_token_id))[0].tolist()] if image_token_id is not None else []
    if not image_positions:
        raise ValueError("row prompt contains no Qwen image placeholder keys")
    if eligible_mask is not None and len(image_positions) != int(eligible_mask.numel()):
        raise ValueError("eligibility mask length does not match image placeholder count")
    position_ids = None if position_mode == "implicit" else derive_explicit_position_ids(model, input_ids=ids, attention_mask=ordinary, image_grid_thw=model_inputs["image_grid_thw"].to(device=device))
    custom = None if eligible_mask is None else build_causal_key_eligibility_mask(sequence_length=ids.shape[1], image_key_positions=image_positions, eligible_image_positions=[image_positions[index] for index in torch.where(eligible_mask)[0].tolist()], device=device)
    if position_mode == "custom" and custom is None:
        raise ValueError("custom position mode requires an eligibility mask")
    logits = _score_model_sequence(model=model, model_inputs=model_inputs, input_ids=ids, attention_mask=ordinary, position_ids=position_ids, custom_mask=custom, features=features, merge_size=merge_size, grid_thw=grid_thw)
    return score_row_log_likelihoods(logits, prefix_length=len(prompt_ids), row_tokens=row, description_length=len(row) - 8, terminal_token_id=terminal_token_id)


def _run_case(model: Any, model_inputs: Mapping[str, Any], features: FeatureBundle, grid_thw: Sequence[int], merge_size: int, prompt_ids: Sequence[int], target_row: Sequence[int], competitor_row: Sequence[int], competitor_mask: torch.Tensor, target_mask: torch.Tensor, *, image_token_id: int, image_grid_thw: torch.Tensor, terminal_token_id: int | None) -> dict[str, Any]:
    target_row = [int(value) for value in target_row]
    competitor_row = [int(value) for value in competitor_row]
    device = next(model.parameters()).device
    base_ids = torch.tensor([list(prompt_ids) + target_row], dtype=torch.long, device=device)
    ordinary = torch.ones_like(base_ids, dtype=torch.long)
    explicit = derive_explicit_position_ids(model, input_ids=base_ids, attention_mask=ordinary, image_grid_thw=image_grid_thw.to(device=device))
    image_positions = [int(v) for v in torch.where(base_ids[0] == int(image_token_id))[0].tolist()]
    full_mask = build_causal_key_eligibility_mask(sequence_length=base_ids.shape[1], image_key_positions=image_positions, eligible_image_positions=image_positions, device=device)
    arms = {"implicit_2d": (None, None, "implicit"), "explicit_2d": (ordinary, explicit, "explicit"), "all_allowed_4d": (full_mask, explicit, "custom"), "target_eligibility": (target_mask, explicit, "custom"), "competitor_eligibility": (competitor_mask, explicit, "custom")}
    phase_scores: dict[str, dict[str, Any]] = {}
    for arm_name, (mask_or_attention, positions, position_mode) in arms.items():
        target_score = _score_one_row(model, model_inputs, features, grid_thw, merge_size, prompt_ids, target_row, eligible_mask=(None if arm_name in {"implicit_2d", "explicit_2d"} else (target_mask if arm_name == "target_eligibility" else competitor_mask if arm_name == "competitor_eligibility" else torch.ones_like(target_mask))), image_token_id=image_token_id, position_mode=position_mode, terminal_token_id=terminal_token_id)
        competitor_score = _score_one_row(model, model_inputs, features, grid_thw, merge_size, prompt_ids, competitor_row, eligible_mask=(None if arm_name in {"implicit_2d", "explicit_2d"} else (target_mask if arm_name == "target_eligibility" else competitor_mask if arm_name == "competitor_eligibility" else torch.ones_like(target_mask))), image_token_id=image_token_id, position_mode=position_mode, terminal_token_id=terminal_token_id)
        phase_scores[arm_name] = {"target": target_score, "competitor": competitor_score}
        first_diff = first_differing_description_index(
            target_row,
            competitor_row,
            description_length=len(target_row) - 8,
        )
        if first_diff is not None:
            target_token = target_score["token_log_probabilities"][first_diff]
            competitor_token = competitor_score["token_log_probabilities"][first_diff if first_diff < len(competitor_score["token_log_probabilities"]) else -1]
            phase_scores[arm_name]["first_differing_description"] = {
                "target": {"sum": float(target_token), "mean": float(target_token), "count": 1},
                "competitor": {"sum": float(competitor_token), "mean": float(competitor_token), "count": 1},
            }
    no_op_drift = _maximum_noop_drift(phase_scores["implicit_2d"], phase_scores["explicit_2d"], phase_scores["all_allowed_4d"])
    target_noop = assess_noop_trust_gate(
        implicit_token_log_probs=phase_scores["implicit_2d"]["target"]["token_log_probabilities"],
        explicit_token_log_probs=phase_scores["explicit_2d"]["target"]["token_log_probabilities"],
        all_allowed_token_log_probs=phase_scores["all_allowed_4d"]["target"]["token_log_probabilities"],
        implicit_ranks=phase_scores["implicit_2d"]["target"]["selected_token_ranks"],
        explicit_ranks=phase_scores["explicit_2d"]["target"]["selected_token_ranks"],
        all_allowed_ranks=phase_scores["all_allowed_4d"]["target"]["selected_token_ranks"],
    )
    competitor_noop = assess_noop_trust_gate(
        implicit_token_log_probs=phase_scores["implicit_2d"]["competitor"]["token_log_probabilities"],
        explicit_token_log_probs=phase_scores["explicit_2d"]["competitor"]["token_log_probabilities"],
        all_allowed_token_log_probs=phase_scores["all_allowed_4d"]["competitor"]["token_log_probabilities"],
        implicit_ranks=phase_scores["implicit_2d"]["competitor"]["selected_token_ranks"],
        explicit_ranks=phase_scores["explicit_2d"]["competitor"]["selected_token_ranks"],
        all_allowed_ranks=phase_scores["all_allowed_4d"]["competitor"]["selected_token_ranks"],
    )
    if not target_noop["passed"] or not competitor_noop["passed"]:
        return {"classification": "invalid_no_op_trust_gate", "no_op_trust_gate": {"target": target_noop, "competitor": competitor_noop}, "arms": phase_scores, "no_op_max_abs_logprob_drift": no_op_drift}
    interactions = compute_crossover_and_release(
        full_scores=_phase_score_map(phase_scores["all_allowed_4d"]),
        target_scores=_phase_score_map(phase_scores["target_eligibility"]),
        competitor_scores=_phase_score_map(phase_scores["competitor_eligibility"]),
    )
    target_description = target_row[1:-7]
    competitor_description = competitor_row[1:-7]
    classification = classify_case(
        interactions,
        no_op_drift=no_op_drift,
        require_description_reversal=target_description != competitor_description,
    )
    return {"classification": classification, "no_op_trust_gate": {"target": target_noop, "competitor": competitor_noop}, "no_op_max_abs_logprob_drift": no_op_drift, "arms": phase_scores, "crossover": interactions, "target_mask_indices": [int(v) for v in torch.where(target_mask)[0].tolist()], "competitor_mask_indices": [int(v) for v in torch.where(competitor_mask)[0].tolist()], "position_ids_sha256": tensor_sha256(explicit)}


def _phase_score_map(arm: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Transpose owner-major row scores into phase-major interaction scores."""

    target = arm["target"]
    competitor = arm["competitor"]
    phases = set(target) & set(competitor)
    excluded = {
        "token_log_probabilities",
        "selected_token_ranks",
        "top_prediction_token_ids",
        "row_entry_vs_terminal",
    }
    result = {
        phase: {"target": target[phase], "competitor": competitor[phase]}
        for phase in sorted(phases - excluded)
    }
    first_difference = arm.get("first_differing_description")
    if isinstance(first_difference, Mapping):
        result["first_differing_description"] = {
            "target": first_difference["target"],
            "competitor": first_difference["competitor"],
        }
    return result


def _maximum_noop_drift(implicit: Mapping[str, Any], explicit: Mapping[str, Any], custom: Mapping[str, Any]) -> float:
    values: list[float] = []
    for owner in ("target", "competitor"):
        for left, right in ((implicit[owner], explicit[owner]), (implicit[owner], custom[owner])):
            a = torch.tensor(left["token_log_probabilities"], dtype=torch.float32)
            b = torch.tensor(right["token_log_probabilities"], dtype=torch.float32)
            values.append(float(torch.max(torch.abs(a - b))))
    return max(values, default=0.0)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "receipt.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
