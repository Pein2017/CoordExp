#!/usr/bin/env python3
"""Run the frozen image-19432 non-boundary x1-to-x2 grammar probe.

This one-time runner deliberately tests only whether translated box-width
behavior extends beyond two frozen annotated chair left edges.  Synthetic
coordinates are not background, physical-owner, or boundary-absence evidence.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import scripts.research.run_fixed_prefix_complete_box_coherence as complete  # noqa: E402


UNIT_ID = "2026-07-17-object-specific-geometry-transport-and-cross-row-influence-horizon"
SCHEMA_VERSION = "fixed-prefix-nonboundary-box-grammar-image19432.v1"
IMAGE_ID = "19432"
FIXED_Y1 = 123
FIXTURE_PATH = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-16-fixed-prefix-complete-box-coherence-and-coordinate-release-factorial/"
    "fixture-v5-20260716e/target-fixture.json"
)
FIXTURE_SHA256 = "cbf9040a4237f58c5365816b4b1e08b932bd18aa66f68515e114826f7ebb8f7a"
EXPECTED_IMAGE_SHA256 = "815ff42f8543e6da5fb5d92372ce0b001f5f14b973f9fddeb6a956319f64277d"
EXPECTED_PROMPT_SHA256 = "250f52fc3dbc6e405a7d3bc1f90bb2a6c391be6812409383fec30aa0e1d0f253"
EXPECTED_IMAGE_SIZE = (1152, 864)
SUPPORT_TOLERANCE = math.log(10.0)
VALID_RIGHT_MASS_FLOOR = 0.80
REAL_SHIFT_FLOOR = 20.0
KENDALL_TAU_FLOOR = 2.0 / 3.0
SLOPE_RANGE = (0.5, 1.5)
WIDTH_RANGE_CEILING = 50.0
BOX_END_TOKEN_ID = 151649
CUES: tuple[tuple[str, int, str], ...] = (
    ("real_target_left_edge", 537, "real_edge"),
    ("synthetic_inter_edge_nonboundary", 608, "synthetic_nonboundary"),
    ("real_adjacent_left_edge", 643, "real_edge"),
    ("synthetic_right_dense_field_nonboundary", 700, "synthetic_nonboundary"),
)
PROHIBITED_CLAIMS = (
    "physical_owner",
    "background",
    "object_absence",
    "physical_or_visual_boundary_independence",
    "coverage",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve(strict=True).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_float32(values: torch.Tensor) -> str:
    tensor = values.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def load_and_validate_fixture(path: Path = FIXTURE_PATH) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    observed_hash = sha256_file(resolved)
    if observed_hash != FIXTURE_SHA256:
        raise ValueError(f"frozen fixture drifted: {observed_hash}")
    fixture = json.loads(resolved.read_text(encoding="utf-8"))
    target = fixture.get("targets", {}).get("C", {})
    image = target.get("image", {})
    row = target.get("row", {})
    if str(image.get("image_id")) != IMAGE_ID:
        raise ValueError("frozen fixture image identifier drifted")
    if (int(image.get("width", -1)), int(image.get("height", -1))) != EXPECTED_IMAGE_SIZE:
        raise ValueError("frozen fixture image dimensions drifted")
    if str(image.get("image_sha256")) != EXPECTED_IMAGE_SHA256:
        raise ValueError("frozen fixture image digest drifted")
    if str(row.get("pre_x1_prompt_token_ids_sha256")) != EXPECTED_PROMPT_SHA256:
        raise ValueError("frozen pre-x1 prompt digest drifted")
    if list(map(int, row.get("source_row_coordinates", []))) != [552, 123, 660, 357]:
        raise ValueError("frozen source row drifted")
    references = {str(item.get("label")): list(map(int, item.get("box", []))) for item in target.get("references", [])}
    if references.get("target_owner") != [537, 121, 651, 349]:
        raise ValueError("target chair reference drifted")
    if references.get("adjacent_owner") != [643, 126, 730, 350]:
        raise ValueError("adjacent chair reference drifted")
    chair_left_edges = sorted(
        int(item["bbox"][0])
        for item in image.get("source_objects", [])
        if str(item.get("description")) == "chair"
    )
    for name, cue, kind in CUES:
        if kind == "synthetic_nonboundary" and min(abs(cue - edge) for edge in chair_left_edges) < (43 if cue == 700 else 35):
            raise ValueError(f"synthetic cue {name} is too close to a frozen chair left edge")
    return {
        "path": str(resolved),
        "sha256": observed_hash,
        "fixture": fixture,
        "target": target,
        "chair_left_edges": chair_left_edges,
    }


def support_admission(history_scores: Mapping[str, float]) -> dict[str, Any]:
    required = {name for name, _, _ in CUES}
    if set(history_scores) != required:
        raise ValueError("history scores must contain the exact frozen four arms")
    real_names = ("real_target_left_edge", "real_adjacent_left_edge")
    best_real = max(float(history_scores[name]) for name in real_names)
    arms: dict[str, Any] = {}
    for name, cue, kind in CUES:
        score = float(history_scores[name])
        gap = best_real - score
        arms[name] = {
            "x1": cue,
            "kind": kind,
            "history_score": score,
            "best_real_history_score": best_real,
            "best_real_minus_arm": gap,
            "probability_ratio_to_best_real": math.exp(-gap),
            "passed": bool(gap <= SUPPORT_TOLERANCE),
        }
    return {
        "threshold_log": SUPPORT_TOLERANCE,
        "reference": "best_supported_real_edge_arm",
        "arms": arms,
        "admitted_arms": sorted(name for name, value in arms.items() if value["passed"]),
        "passed": all(value["passed"] for value in arms.values()),
    }


def x2_distribution_metrics(coordinate_logits: torch.Tensor, *, x1: int) -> dict[str, Any]:
    logits = coordinate_logits.detach().to(device="cpu", dtype=torch.float32).flatten()
    if logits.numel() != 1000 or not bool(torch.isfinite(logits).all()):
        raise ValueError("x2 coordinate logits must be a finite 1000-vector")
    if not 0 <= int(x1) <= 998:
        raise ValueError("x1 must leave at least one valid right boundary")
    probabilities = torch.softmax(logits, dim=0)
    valid = probabilities[int(x1) + 1 :]
    valid_mass = float(valid.sum())
    invalid_mass = float(probabilities[: int(x1) + 1].sum())
    if valid_mass <= 0.0:
        raise ValueError("valid right-boundary mass is zero")
    valid_bins = torch.arange(int(x1) + 1, 1000, dtype=torch.float32)
    expected_right_valid = float((valid_bins * valid).sum() / valid_mass)
    expected_width = expected_right_valid - float(x1)
    width_pmf = torch.zeros(999, dtype=torch.float32)
    width_pmf[: 999 - int(x1)] = valid / valid_mass
    top_values, top_indices = torch.topk(logits, k=20)
    return {
        "x1": int(x1),
        "coordinate_logits_float32": [float(value) for value in logits.tolist()],
        "coordinate_logits_float32_sha256": sha256_float32(logits),
        "valid_right_mass": valid_mass,
        "invalid_right_mass": invalid_mass,
        "expected_right_given_valid": expected_right_valid,
        "expected_width_given_valid": expected_width,
        "width_pmf_float32": [float(value) for value in width_pmf.tolist()],
        "width_pmf_float32_sha256": sha256_float32(width_pmf),
        "width_pmf_sum": float(width_pmf.sum()),
        "valid_width_admitted": bool(valid_mass >= VALID_RIGHT_MASS_FLOOR),
        "top20_x2": [
            {"coordinate_bin": int(index), "logit_float32": float(value)}
            for value, index in zip(top_values, top_indices)
        ],
    }


def jensen_shannon_divergence(left: Sequence[float], right: Sequence[float]) -> float:
    p = torch.as_tensor(left, dtype=torch.float64)
    q = torch.as_tensor(right, dtype=torch.float64)
    if p.shape != q.shape or p.ndim != 1 or bool((p < 0).any()) or bool((q < 0).any()):
        raise ValueError("Jensen-Shannon inputs must be same-shaped nonnegative vectors")
    if float(p.sum()) <= 0.0 or float(q.sum()) <= 0.0:
        raise ValueError("Jensen-Shannon inputs must have positive mass")
    p = p / p.sum()
    q = q / q.sum()
    midpoint = 0.5 * (p + q)
    left_term = torch.where(p > 0, p * (torch.log(p) - torch.log(midpoint)), 0.0).sum()
    right_term = torch.where(q > 0, q * (torch.log(q) - torch.log(midpoint)), 0.0).sum()
    return float(0.5 * (left_term + right_term))


def kendall_tau(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        raise ValueError("Kendall correlation needs matching sequences")
    concordant = discordant = 0
    for left, right in itertools.combinations(range(len(x_values)), 2):
        product = (float(x_values[right]) - float(x_values[left])) * (float(y_values[right]) - float(y_values[left]))
        if product > 0:
            concordant += 1
        elif product < 0:
            discordant += 1
    pairs = len(x_values) * (len(x_values) - 1) // 2
    return float((concordant - discordant) / pairs)


def ordinary_least_squares_slope(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        raise ValueError("slope needs matching sequences")
    mean_x = statistics.fmean(map(float, x_values))
    mean_y = statistics.fmean(map(float, y_values))
    denominator = sum((float(value) - mean_x) ** 2 for value in x_values)
    if denominator == 0.0:
        raise ValueError("slope x values are constant")
    return float(sum((float(x) - mean_x) * (float(y) - mean_y) for x, y in zip(x_values, y_values)) / denominator)


def panel_statistics(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    names = [name for name, _, _ in CUES]
    if set(metrics) != set(names):
        raise ValueError("panel metrics must contain the frozen arms")
    x_values = [int(metrics[name]["x1"]) for name in names]
    expected = [float(metrics[name]["expected_right_given_valid"]) for name in names]
    widths = [float(metrics[name]["expected_width_given_valid"]) for name in names]
    absolute_distributions = {
        name: torch.softmax(torch.tensor(metrics[name]["coordinate_logits_float32"], dtype=torch.float64), dim=0).tolist()
        for name in names
    }
    pairs: list[dict[str, Any]] = []
    for left, right in itertools.combinations(names, 2):
        pairs.append({
            "left": left,
            "right": right,
            "absolute_x2_jensen_shannon": jensen_shannon_divergence(absolute_distributions[left], absolute_distributions[right]),
            "translated_width_jensen_shannon": jensen_shannon_divergence(metrics[left]["width_pmf_float32"], metrics[right]["width_pmf_float32"]),
        })
    absolute_median = statistics.median(item["absolute_x2_jensen_shannon"] for item in pairs)
    width_median = statistics.median(item["translated_width_jensen_shannon"] for item in pairs)
    return {
        "x1_values_in_frozen_arm_order": x_values,
        "expected_right_given_valid_in_frozen_arm_order": expected,
        "expected_width_given_valid_in_frozen_arm_order": widths,
        "real_adjacent_minus_target_expected_right": float(
            metrics["real_adjacent_left_edge"]["expected_right_given_valid"]
            - metrics["real_target_left_edge"]["expected_right_given_valid"]
        ),
        "kendall_tau_x1_expected_right": kendall_tau(x_values, expected),
        "ordinary_least_squares_slope": ordinary_least_squares_slope(x_values, expected),
        "expected_width_range": max(widths) - min(widths),
        "pairwise_divergences": pairs,
        "median_absolute_x2_jensen_shannon": absolute_median,
        "median_translated_width_jensen_shannon": width_median,
    }


def classify_panel(
    *,
    execution_trust_passed: bool,
    support: Mapping[str, Any],
    metrics: Mapping[str, Mapping[str, Any]],
    statistics_payload: Mapping[str, Any],
) -> dict[str, Any]:
    support_passed = bool(support.get("passed"))
    valid_passed = all(bool(metrics[name]["valid_width_admitted"]) for name, _, _ in CUES)
    gates = {
        "execution_trust": bool(execution_trust_passed),
        "all_arm_history_support": support_passed,
        "all_arm_valid_right_mass": valid_passed,
        "real_shift": float(statistics_payload["real_adjacent_minus_target_expected_right"]) >= REAL_SHIFT_FLOOR,
        "kendall_tau": float(statistics_payload["kendall_tau_x1_expected_right"]) + 1e-12 >= KENDALL_TAU_FLOOR,
        "slope": SLOPE_RANGE[0] <= float(statistics_payload["ordinary_least_squares_slope"]) <= SLOPE_RANGE[1],
        "width_range": float(statistics_payload["expected_width_range"]) <= WIDTH_RANGE_CEILING,
        "translated_width_more_stable": float(statistics_payload["median_translated_width_jensen_shannon"]) < float(statistics_payload["median_absolute_x2_jensen_shannon"]),
    }
    if not gates["execution_trust"]:
        classification = "execution_trust_failure"
    elif not gates["all_arm_history_support"]:
        classification = "unidentified_control_support_failure"
    elif not gates["all_arm_valid_right_mass"]:
        classification = "unidentified_valid_width_failure"
    elif all(gates.values()):
        classification = "simple_translation_grammar_compatible"
    else:
        classification = "simple_translation_grammar_insufficient"
    return {
        "classification": classification,
        "gates": gates,
        "admitted_positive_mechanism_handle": classification == "simple_translation_grammar_compatible",
        "claim_scope": "translation-grammar compatibility beyond two frozen annotated chair left edges",
        "prohibited_claims": list(PROHIBITED_CLAIMS),
    }


def _selected_support(logits: torch.Tensor, coordinate: int) -> dict[str, Any]:
    return complete._log_probability_summary(logits, complete.coordinate_token_id(int(coordinate)))


def _parse_greedy_suffix(token_ids: Sequence[int]) -> dict[str, Any]:
    values = [int(value) for value in token_ids]
    coordinates: list[int] = []
    failure: str | None = None
    for index in range(2):
        if index >= len(values):
            failure = f"missing_coordinate_{index}"
            break
        try:
            coordinates.append(complete.coordinate_bin(values[index]))
        except ValueError:
            failure = f"non_coordinate_at_{index}"
            break
    natural_close = len(values) >= 3 and values[2] == BOX_END_TOKEN_ID
    if failure is None and not natural_close:
        failure = "missing_natural_box_end"
    return {
        "generated_token_ids": values,
        "released_x2_y2": coordinates if len(coordinates) == 2 else None,
        "natural_box_close": natural_close,
        "parser_valid": failure is None,
        "failure": failure,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    fixture_receipt = load_and_validate_fixture(Path(args.fixture))
    target = fixture_receipt["target"]
    fixture = fixture_receipt["fixture"]
    config_path = Path(args.infer_config).expanduser().resolve(strict=True)
    source_jsonl = Path(args.source_jsonl).expanduser().resolve(strict=True)

    from scripts.research.run_batch_coordinate_logit_invariance import _direct_full_prefix_logits
    from src.config.fingerprint import sha256_json
    from src.config.inference import load_infer_config
    from src.data import load_raw_examples
    from src.inference.backend import DecodeGenerationPolicy, DecodeRequest, HFGenerateBackend
    from src.inference.image_plan import materialize_image_plan_batch, verify_processor_model_vision_parity
    from src.inference.pipeline import _processor_config, _template_config, _tokenizer_identity
    from src.inference.prompt import AssistantContinuation, build_prompt_record
    from src.inference.runtime import assemble_runtime

    with complete._temporary_cwd(config_path.parents[3]):
        resolved = load_infer_config(config_path)
    config = resolved.config.model_copy(update={"model": resolved.config.model.model_copy(update={"dtype": "fp32"})})
    runtime = assemble_runtime(config, source_gate_root=config_path.parents[3])
    qwen = runtime.qwen
    qwen.model.eval()
    actual_dtypes = sorted({str(parameter.dtype) for parameter in qwen.model.parameters()})
    if actual_dtypes != ["torch.float32"]:
        raise RuntimeError(f"float32 execution contract failed: {actual_dtypes!r}")
    attention = str(getattr(qwen.model.config, "_attn_implementation", None) or getattr(qwen.model.config, "attn_implementation", None) or "unknown")
    if attention != "sdpa":
        raise RuntimeError(f"SDPA execution contract failed: {attention!r}")
    verify_processor_model_vision_parity(processor_identity=qwen.processor_identity, model_config=getattr(qwen.model, "config", qwen.model))

    raw = next((item for item in load_raw_examples(source_jsonl) if str(item.metadata.get("source", {}).get("image_id")) == IMAGE_ID), None)
    if raw is None:
        raise ValueError("source JSONL lacks image 19432")
    image_path = Path(raw.image.path).expanduser().resolve(strict=True)
    if (raw.image.width, raw.image.height) != EXPECTED_IMAGE_SIZE or sha256_file(image_path) != EXPECTED_IMAGE_SHA256:
        raise ValueError("live canonical image identity drifted")
    row = target["row"]
    generated_prefix = list(map(int, row["pre_x1_generated_token_ids"]))
    frozen_prompt = list(map(int, row["pre_x1_prompt_token_ids"]))
    continuation_text = qwen.tokenizer.decode(generated_prefix, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    prompt = build_prompt_record(raw, _template_config(config), processor=qwen.processor, row_index=0, assistant_continuation=AssistantContinuation(text=continuation_text))
    if list(prompt.prompt_token_ids) != frozen_prompt or complete._json_hash(frozen_prompt) != EXPECTED_PROMPT_SHA256:
        raise RuntimeError("exact pre-x1 prompt reconstruction failed")
    plan = materialize_image_plan_batch([raw], components=qwen, processor_config=_processor_config(config), materialize=True, row_indices=[0])
    model_inputs = plan.model_inputs_by_row_id[prompt.row_id]
    tensor_batch_sizes = {
        key: int(value.shape[0])
        for key, value in model_inputs.items()
        if key in {"image_grid_thw", "input_ids", "attention_mask"} and isinstance(value, torch.Tensor) and value.ndim > 0
    }
    if any(value != 1 for value in tensor_batch_sizes.values()):
        raise RuntimeError(f"physical batch-size contract failed: {tensor_batch_sizes!r}")
    model_identity = dict(runtime.model_identity)
    tokenizer_identity = _tokenizer_identity(qwen)
    generation_fingerprint = sha256_json(config.generation.model_dump(mode="json"))
    backend = HFGenerateBackend(model=qwen.model, tokenizer=qwen.tokenizer, model_identity=model_identity, tokenizer_identity=tokenizer_identity, generation_config_fingerprint=generation_fingerprint)
    policy = DecodeGenerationPolicy.greedy(max_new_tokens=3, repetition_penalty=1.0)

    x1_request = DecodeRequest(request_id="image19432-grammar:x1-support", prompt_token_ids=frozen_prompt, model_inputs=model_inputs, generation_policy=policy)
    x1_logits_list, _, x1_runtime = _direct_full_prefix_logits(backend=backend, requests=[x1_request])
    x1_logits = x1_logits_list[0]
    records: dict[str, Any] = {}
    history_scores: dict[str, float] = {}
    for name, x1, kind in CUES:
        y1_prefix = [*frozen_prompt, complete.coordinate_token_id(x1)]
        y1_request = DecodeRequest(request_id=f"image19432-grammar:{name}:y1-support", prompt_token_ids=y1_prefix, model_inputs=model_inputs, generation_policy=policy)
        y1_logits_list, _, y1_runtime = _direct_full_prefix_logits(backend=backend, requests=[y1_request])
        y1_logits = y1_logits_list[0]
        x2_prefix = [*y1_prefix, complete.coordinate_token_id(FIXED_Y1)]
        x2_request = DecodeRequest(request_id=f"image19432-grammar:{name}:x2", prompt_token_ids=x2_prefix, model_inputs=model_inputs, generation_policy=policy)
        x2_logits_list, _, x2_runtime = _direct_full_prefix_logits(backend=backend, requests=[x2_request])
        x1_support = _selected_support(x1_logits, x1)
        y1_support = _selected_support(y1_logits, FIXED_Y1)
        history_score = float(x1_support["selected_full_vocabulary_logprob_float32"] + y1_support["selected_full_vocabulary_logprob_float32"])
        history_scores[name] = history_score
        metrics = x2_distribution_metrics(
            x2_logits_list[0][complete.COORDINATE_TOKEN_START:complete.COORDINATE_TOKEN_END_EXCLUSIVE],
            x1=x1,
        )
        greedy_request = DecodeRequest(request_id=f"image19432-grammar:{name}:greedy", prompt_token_ids=x2_prefix, model_inputs=model_inputs, generation_policy=policy)
        greedy_result = backend.generate_batch([greedy_request], model_identity=model_identity, tokenizer_identity=tokenizer_identity, generation_config_fingerprint=generation_fingerprint)[0]
        greedy = _parse_greedy_suffix(greedy_result.generated_token_ids)
        greedy["generated_token_text"] = list(map(str, qwen.tokenizer.convert_ids_to_tokens(greedy_result.generated_token_ids)))
        records[name] = {
            "arm": name,
            "kind": kind,
            "x1": x1,
            "fixed_y1": FIXED_Y1,
            "history_score": history_score,
            "x1_support": x1_support,
            "y1_support": y1_support,
            "x2_distribution": metrics,
            "secondary_greedy_release": greedy,
            "runtime_receipts": {"y1": y1_runtime, "x2": x2_runtime},
        }
    support = support_admission(history_scores)
    metric_map = {name: value["x2_distribution"] for name, value in records.items()}
    stats = panel_statistics(metric_map)
    decision = classify_panel(execution_trust_passed=True, support=support, metrics=metric_map, statistics_payload=stats)
    return {
        "schema_version": SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "conclusion_scope": "NonboundaryCoordinateToRightBoundaryTranslationGrammarCompatibility",
        "fixture": {"path": fixture_receipt["path"], "sha256": fixture_receipt["sha256"], "cue_arms": [{"name": n, "x1": x, "kind": k} for n, x, k in CUES], "fixed_y1": FIXED_Y1, "chair_left_edges": fixture_receipt["chair_left_edges"]},
        "image_identity": {"image_id": IMAGE_ID, "path": str(image_path), "width": raw.image.width, "height": raw.image.height, "sha256": sha256_file(image_path)},
        "input_identity": {"config_path": str(config_path), "config_sha256": sha256_file(config_path), "source_jsonl": str(source_jsonl), "source_jsonl_sha256": sha256_file(source_jsonl), "prompt_token_ids_sha256": complete._json_hash(frozen_prompt), "prompt_token_count": len(frozen_prompt)},
        "runtime_contract": {"dtype": "torch.float32", "attention_implementation": attention, "physical_batch_size": 1, "model_input_tensor_batch_sizes": tensor_batch_sizes, "repetition_penalty": 1.0, "model_identity": model_identity, "tokenizer_identity": tokenizer_identity, "generation_config_fingerprint": generation_fingerprint, "x1_runtime_receipt": x1_runtime},
        "support_admission": support,
        "records": records,
        "panel_statistics": stats,
        "panel_decision": decision,
        "interpretation_contract": {"permitted": ["translation-grammar compatibility beyond two frozen annotated chair left edges"], "prohibited": list(PROHIBITED_CLAIMS), "greedy_sidecar_is_primary": False},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=FIXTURE_PATH)
    parser.add_argument("--infer-config", type=Path, default=complete.DEFAULT_CONFIG)
    parser.add_argument("--source-jsonl", type=Path, default=complete.DEFAULT_SOURCE_JSONL)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload["runner_sha256"] = sha256_file(Path(__file__))
    payload["normalized_argv"] = list(sys.argv[1:] if argv is None else argv)
    output = output_dir / "receipt.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
