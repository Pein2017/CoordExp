#!/usr/bin/env python3
"""CPU-only token/row stop-pattern diagnosis for recovered second-fit readbacks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path("/data/CoordExp/.worktrees/research-probes")
sys.path.insert(0, str(REPO))

from probes.source_rweak_row_cross.run import native_record
from probes.training_set_completion.readback_selectors import flatten_raw_rows, iou_xyxy

B = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum")
OUT = B / "second-fit-stop-patterns-v1"
RECOVERY = B / "second-fit-readback-recovery-v1"
READBACKS = {step: RECOVERY / f"readback-step-{step}.json" for step in (16, 32, 64)}
RECOVERY_MANIFEST = RECOVERY / "manifest.json"
RECOVERY_RESULT = RECOVERY / "result.json"
BANK = B / "stage03-repair-route-bank-v1/bank.json"
BANK_ROUTES = B / "stage03-repair-route-bank-v1/runtime-route-list.json"
BANK_SUMMARY = B / "stage03-repair-route-bank-v1/summary.json"
BANK_ACCEPTANCE = B / "stage03-repair-route-bank-v1/root-acceptance.json"
OWNER_PLAN = B / "stage03-single-owner-repair-v1/owner-plan.json"
PARENT16 = B / "first-fit-v1/readback-step-16.json"
TARGET_V3 = B / "target-owners-complete-v3.json"
PARSER_SOURCE = Path("src/inference/parsing.py").resolve()
NATIVE_WRAPPER_SOURCE = Path("probes/source_rweak_row_cross/run.py").resolve()
SELECTOR_SOURCE = Path("probes/training_set_completion/readback_selectors.py").resolve()
TOKENIZER_ROOT = Path("/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent")
CAP = 3084
EOS = 151645
OBJECT_START, OBJECT_END, BOX_START, BOX_END = 151646, 151647, 151648, 151649


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def binding(path: Path, expected: str | None = None, expected_size: int | None = None) -> dict[str, Any]:
    path = path.resolve(strict=True)
    actual = file_hash(path)
    size = path.stat().st_size
    if expected is not None and actual != expected:
        raise ValueError(f"hash mismatch: {path}")
    if expected_size is not None and size != expected_size:
        raise ValueError(f"size mismatch: {path}")
    return {"path": str(path), "sha256": actual, "size_bytes": size}


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def golden(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "example_id": case["row_id"], "gt": [], "image_height": case["image_height"],
        "image_path": case["image_path"], "image_width": case["image_width"],
        "row_id": case["row_id"], "row_index": case["row_index"],
    }


def lcp(left: list[int], right: list[int]) -> int:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right))


def strict_rows(ids: list[int]) -> list[dict[str, Any]]:
    rows, index = [], 0
    while index < len(ids):
        if ids[index] != OBJECT_START:
            index += 1
            continue
        start = index
        try:
            object_end = ids.index(OBJECT_END, start + 1)
        except ValueError:
            break
        box_start = object_end + 1
        box_end = box_start + 5
        if box_start >= len(ids) or ids[box_start] != BOX_START or box_end >= len(ids) or ids[box_end] != BOX_END:
            index = start + 1
            continue
        coords = [token - 151670 for token in ids[box_start + 1:box_end]]
        signature = ids[start:box_end + 1]
        rows.append({
            "token_start": start, "token_end_exclusive": box_end + 1,
            "token_ids_sha256": digest(signature), "token_ids": signature,
            "category_token_ids": ids[start + 1:object_end], "coord_bins_1000": coords,
            "raw_axes_valid": len(coords) == 4 and coords[0] < coords[2] and coords[1] < coords[3],
        })
        index = box_end + 1
    return rows


def common_row_prefix(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> int:
    return lcp([x["token_ids_sha256"] for x in left], [x["token_ids_sha256"] for x in right])


def eventual_periodic_tail(ids: list[int], *, max_period: int = 64, min_cycles: int = 8, min_span: int = 100) -> dict[str, Any] | None:
    candidates = []
    for period in range(1, min(max_period, len(ids) // min_cycles) + 1):
        last_mismatch = -1
        for index in range(period, len(ids)):
            if ids[index] != ids[index - period]:
                last_mismatch = index
        start = max(0, last_mismatch - period + 1)
        span = len(ids) - start
        if span >= max(min_span, min_cycles * period):
            candidates.append((start, period, span))
    if not candidates:
        return None
    start, period, span = min(candidates, key=lambda x: (x[0], x[1]))
    pattern = ids[start:start + period]
    if OBJECT_START in pattern:
        rotation = pattern.index(OBJECT_START)
        pattern = pattern[rotation:] + pattern[:rotation]
    return {
        "token_start_zero_based": start, "token_end_exclusive": len(ids),
        "period_tokens": period, "periodic_span_tokens": span,
        "cycles_including_partial": span / period,
        "period_token_ids": pattern, "period_token_ids_sha256": digest(pattern),
    }


def token_offsets(tokenizer: Any, text: str, ids: list[int]) -> list[tuple[int, int]]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if encoded["input_ids"] != ids:
        raise ValueError("saved token IDs do not match tokenizer re-encode")
    return [tuple(x) for x in encoded["offset_mapping"]]


def char_to_token(offsets: list[tuple[int, int]], char_start: int) -> int:
    for index, (start, end) in enumerate(offsets):
        if end > char_start or start >= char_start:
            return index
    return len(offsets)


def projected_native_rows(parsed: dict[str, Any], offsets: list[tuple[int, int]]) -> list[dict[str, Any]]:
    valid, dropped = flatten_raw_rows(parsed)
    rows = []
    for item in valid + dropped:
        raw = item["raw"]
        row = {
            "generated_order": int(item["generated_order"]), "status": item["status"],
            "description": item.get("description"), "coord_bins_1000": item.get("coord_bins_1000"),
            "raw_span_sha256": item.get("raw_span_sha256"), "drop_code": item.get("drop_code"),
            "drop_reason": item.get("drop_reason"), "char_start": raw.get("char_start"), "char_end": raw.get("char_end"),
        }
        if row["char_start"] is not None:
            row["token_start_zero_based"] = char_to_token(offsets, int(row["char_start"]))
            row["token_end_exclusive"] = char_to_token(offsets, int(row["char_end"]))
        rows.append(row)
    return sorted(rows, key=lambda x: x["generated_order"])


def first_duplicate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    seen = {}
    for row in rows:
        key = row["raw_span_sha256"]
        if key in seen:
            return {"generated_order": row["generated_order"], "repeats_generated_order": seen[key], "raw_span_sha256": key}
        seen[key] = row["generated_order"]
    return None


def longest_identical_run(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    best = (1, 0, 0)
    index = 0
    while index < len(rows):
        end = index + 1
        while end < len(rows) and rows[end]["raw_span_sha256"] == rows[index]["raw_span_sha256"]:
            end += 1
        if end - index > best[0]:
            best = (end - index, index, end - 1)
        index = end
    repetitions, start, end = best
    if repetitions < 2:
        return None
    representative = rows[start]
    return {
        "repetitions": repetitions,
        "start_generated_order": representative["generated_order"],
        "end_generated_order": rows[end]["generated_order"],
        "raw_span_sha256": representative["raw_span_sha256"],
        "status": representative["status"], "drop_code": representative.get("drop_code"),
        "description": representative.get("description"), "coord_bins_1000": representative.get("coord_bins_1000"),
    }


def target_match(valid_rows: list[dict[str, Any]], plan: dict[str, Any] | None, first_anomaly: int | None, sustained: int | None) -> dict[str, Any] | None:
    if plan is None or plan.get("action") != "append_absent_gt_before_eos_then_release":
        return None
    candidates = []
    for row in valid_rows:
        bins = row.get("coord_bins_1000")
        if not isinstance(bins, list) or len(bins) != 4:
            continue
        candidates.append((iou_xyxy(bins, plan["reference_bins"]), row))
    overlap, row = max(candidates, default=(0.0, None), key=lambda x: x[0])
    appeared = row is not None and overlap >= 0.5
    order = row["generated_order"] if appeared else None
    return {
        "owner_id": str(plan["owner_id"]), "category": plan["category"],
        "reference_bins": plan["reference_bins"], "diagnostic_iou50_appeared": appeared,
        "best_iou": overlap, "best_generated_order": row["generated_order"] if row else None,
        "best_description": row.get("description") if row else None,
        "best_coord_bins_1000": row.get("coord_bins_1000") if row else None,
        "exact_geometry": bool(appeared and row["coord_bins_1000"] == plan["reference_bins"]),
        "category_exact": bool(appeared and row.get("description") == plan["category"]),
        "before_first_observable_anomaly": bool(appeared and (first_anomaly is None or order < first_anomaly)),
        "before_sustained_repeat_or_collapse": bool(appeared and (sustained is None or order < sustained)),
        "claim_boundary": "Class-agnostic normalized-bin IoU is diagnostic only; it does not establish physical owner truth.",
    }


def build() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_ROOT), local_files_only=True, use_fast=True)
    manifest = load(RECOVERY_MANIFEST)
    recovery_result = load(RECOVERY_RESULT)
    bank = load(BANK)
    bank_routes = load(BANK_ROUTES)
    bank_summary = load(BANK_SUMMARY)
    bank_acceptance = load(BANK_ACCEPTANCE)
    owner_plan = load(OWNER_PLAN)
    parent = load(PARENT16)
    target = load(TARGET_V3)
    routes = {int(x["image_id"]): x for x in manifest["routes"]}
    parent_rows = {int(x["image_id"]): x for x in parent["rows"]}
    bank_by_image = {int(x["image_id"]): x for x in bank_routes}
    plan_by_image = {int(x["image_id"]): x for x in owner_plan["rows"]}
    target_by_key = {(int(x["image_id"]), str(x["owner_id"])): x for x in target["records"]}

    primary_paths = [*READBACKS.values(), RECOVERY_MANIFEST, RECOVERY_RESULT, BANK, BANK_ROUTES,
                     BANK_SUMMARY, BANK_ACCEPTANCE, OWNER_PLAN, PARENT16, TARGET_V3,
                     PARSER_SOURCE, NATIVE_WRAPPER_SOURCE, SELECTOR_SOURCE, TOKENIZER_ROOT / "tokenizer.json"]
    source_bindings = {"primary": [binding(path) for path in primary_paths], "raw_rows": []}
    output_rows = []
    parent_parsed_rows = {}
    for image_id, saved in parent_rows.items():
        case = routes[image_id]["case"]
        offsets = token_offsets(tokenizer, saved["raw_decode_text"], saved["generated_token_ids"])
        parsed = native_record(saved["raw_decode_text"], case, golden(case), saved["decode_stop_reason"])
        parent_parsed_rows[image_id] = projected_native_rows(parsed, offsets)

    for step, readback_path in READBACKS.items():
        readback = load(readback_path)
        if readback["status"] != "completed_unscored" or readback["recovery"]["checkpoint_step"] != step:
            raise ValueError("readback identity mismatch")
        raw_binding_by_image = {}
        for item in readback["recovery"]["per_image_rows"]:
            bound = binding(Path(item["path"]), item["sha256"], item["size_bytes"])
            raw = load(Path(item["path"]))
            raw_binding_by_image[int(raw["image_id"])] = bound
            source_bindings["raw_rows"].append(bound)
        for saved in sorted(readback["rows"], key=lambda x: int(x["image_id"])):
            image_id = int(saved["image_id"])
            ids = saved["generated_token_ids"]
            if saved["generated_token_ids_sha256"] != digest(ids):
                raise ValueError("token hash mismatch")
            raw_source = load(Path(raw_binding_by_image[image_id]["path"]))
            for key in ("generated_token_ids", "raw_decode_text", "decode_stop_reason", "route_id"):
                if raw_source[key] != saved[key]:
                    raise ValueError(f"raw/readback mismatch: {step}/{image_id}/{key}")
            offsets = token_offsets(tokenizer, saved["raw_decode_text"], ids)
            case = routes[image_id]["case"]
            parsed = native_record(saved["raw_decode_text"], case, golden(case), saved["decode_stop_reason"])
            rows = projected_native_rows(parsed, offsets)
            valid_rows = [x for x in rows if x["status"] == "parsed_valid"]
            dropped_rows = [x for x in rows if x["status"] == "parser_dropped"]
            duplicate = first_duplicate(rows)
            run = longest_identical_run(rows)
            period = eventual_periodic_tail(ids)
            if period is not None:
                decoded = tokenizer.decode(period["period_token_ids"], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                period["period_decoded_text"] = decoded[:240]
                period_order = next((x["generated_order"] for x in rows if x.get("token_start_zero_based", CAP + 1) <= period["token_start_zero_based"] < x.get("token_end_exclusive", -1)), None)
                period["containing_generated_order"] = period_order
            first_drop = min((x["generated_order"] for x in dropped_rows), default=None)
            first_axis = min((x["generated_order"] for x in dropped_rows if x.get("drop_code") == "data.bbox_order"), default=None)
            first_malformed = min((x["generated_order"] for x in dropped_rows if x.get("drop_code") != "data.bbox_order"), default=None)
            anomaly_candidates = [x for x in (first_drop, duplicate["generated_order"] if duplicate else None) if x is not None]
            first_anomaly = min(anomaly_candidates, default=None)
            sustained_candidates = []
            if run and run["repetitions"] >= 3:
                sustained_candidates.append(run["start_generated_order"])
            if period and period.get("containing_generated_order") is not None:
                sustained_candidates.append(period["containing_generated_order"])
            sustained = min(sustained_candidates, default=None)
            parent_native_hashes = {x["raw_span_sha256"] for x in parent_parsed_rows[image_id] if x["status"] == "parsed_valid"}
            pre_anomaly_valid = [x for x in valid_rows if first_anomaly is None or x["generated_order"] < first_anomaly]
            strict = strict_rows(ids)
            parent_strict = strict_rows(parent_rows[image_id]["generated_token_ids"])
            bank_strict = strict_rows(bank_by_image[image_id]["continuation_token_ids"])
            is_cap = saved["decode_stop_reason"] == "length" and len(ids) == CAP
            repeated_evidence = bool(period or (run and run["repetitions"] >= 3))
            if is_cap and period and period.get("containing_generated_order") in {x["generated_order"] for x in dropped_rows}:
                family = "capped_structural_token_loop"
            elif is_cap and run and run["repetitions"] >= 3:
                family = "capped_exact_row_repeat"
            elif is_cap:
                family = "capped_mixed_collapse"
            elif dropped_rows:
                family = "natural_eos_with_parser_debt"
            else:
                family = "natural_eos_valid_rows"
            timing = next(x for x in recovery_result["per_image_timing"] if int(x["image_id"]) == image_id and int(x["step"]) == step)
            record = {
                "schema": "training_set_completion.second_fit_stop_pattern.output.v1",
                "step": step, "image_id": image_id, "route_id": saved["route_id"],
                "source_raw_row": raw_binding_by_image[image_id],
                "stop_reason": saved["decode_stop_reason"], "natural_eos": saved["decode_stop_reason"] == "im_end" and ids[-1] == EOS,
                "capped": is_cap, "token_count": len(ids), "elapsed_seconds": timing["elapsed_seconds"],
                "token_rate_per_second": timing["token_rate_per_second"],
                "native_parser": {
                    "valid_rows": len(valid_rows), "dropped_rows": len(dropped_rows),
                    "rawaxis_invalid_rows": sum(x.get("drop_code") == "data.bbox_order" for x in dropped_rows),
                    "malformed_rows_or_spans": sum(x.get("drop_code") != "data.bbox_order" for x in dropped_rows),
                    "first_dropped_generated_order": first_drop, "first_rawaxis_invalid_generated_order": first_axis,
                    "first_malformed_generated_order": first_malformed,
                },
                "strict_token_grammar": {"complete_rows": len(strict), "rawaxis_invalid_rows": sum(not x["raw_axes_valid"] for x in strict)},
                "prefix": {
                    "parent16_common_tokens": lcp(ids, parent_rows[image_id]["generated_token_ids"]),
                    "parent16_common_complete_rows": common_row_prefix(strict, parent_strict),
                    "training_bank_common_tokens": lcp(ids, bank_by_image[image_id]["continuation_token_ids"]),
                    "training_bank_common_complete_rows": common_row_prefix(strict, bank_strict),
                },
                "first_observable_anomaly_generated_order": first_anomaly,
                "first_exact_duplicate_row": duplicate,
                "longest_identical_row_run": run,
                "eventual_exact_token_period": period,
                "sustained_repeat_or_collapse_generated_order": sustained,
                "exact_repeat_or_token_loop_observed": repeated_evidence,
                "valid_rows_before_first_anomaly": len(pre_anomaly_valid),
                "novel_valid_rows_before_first_anomaly": sum(x["raw_span_sha256"] not in parent_native_hashes for x in pre_anomaly_valid),
                "planned_missing_gt_geometry": target_match(valid_rows, plan_by_image.get(image_id), first_anomaly, sustained),
                "observed_failure_family": family,
                "claim_boundary": "Parser-valid and IoU-matched rows are diagnostic candidates, not physical-owner or false-positive rulings.",
            }
            output_rows.append(record)

    output_rows.sort(key=lambda x: (x["step"], x["image_id"]))
    appended_plans = [x for x in owner_plan["rows"] if x.get("action") == "append_absent_gt_before_eos_then_release"]
    if len(appended_plans) != 9:
        raise ValueError("expected nine appended GT plans")
    for item in appended_plans:
        target_record = target_by_key[(int(item["image_id"]), str(item["owner_id"]))]
        if target_record["reference_coord_bins_1000"] != item["reference_bins"]:
            raise ValueError("owner plan/target geometry mismatch")

    dose_summaries = []
    for step in (16, 32, 64):
        rows = [x for x in output_rows if x["step"] == step]
        planned = [x["planned_missing_gt_geometry"] for x in rows if x["planned_missing_gt_geometry"] is not None]
        dose_summaries.append({
            "step": step, "images": len(rows), "capped_images": sum(x["capped"] for x in rows),
            "natural_eos_images": sum(x["natural_eos"] for x in rows), "tokens": sum(x["token_count"] for x in rows),
            "native_valid_rows": sum(x["native_parser"]["valid_rows"] for x in rows),
            "native_dropped_rows": sum(x["native_parser"]["dropped_rows"] for x in rows),
            "rawaxis_invalid_rows": sum(x["native_parser"]["rawaxis_invalid_rows"] for x in rows),
            "malformed_rows_or_spans": sum(x["native_parser"]["malformed_rows_or_spans"] for x in rows),
            "capped_with_exact_repeat_or_token_loop": sum(x["capped"] and x["exact_repeat_or_token_loop_observed"] for x in rows),
            "planned_gt_geometry_iou50_appeared": sum(x["diagnostic_iou50_appeared"] for x in planned),
            "planned_gt_geometry_exact": sum(x["exact_geometry"] for x in planned),
            "planned_gt_geometry_before_first_anomaly": sum(x["before_first_observable_anomaly"] for x in planned),
            "planned_gt_geometry_before_sustained_repeat_or_collapse": sum(x["before_sustained_repeat_or_collapse"] for x in planned),
            "planned_gt_denominator": len(planned),
        })

    eos_active_tokens = sum(
        weight for route in bank_routes
        for token, weight in zip(route["continuation_token_ids"], route["ce_weights"], strict=True)
        if token == EOS
    )
    active_tokens = sum(sum(route["ce_weights"]) for route in bank_routes)
    eos_images = [int(route["image_id"]) for route in bank_routes if any(token == EOS and weight for token, weight in zip(route["continuation_token_ids"], route["ce_weights"], strict=True))]
    training_eos = {
        "routes": len(bank_routes), "continuation_tokens": sum(len(x["continuation_token_ids"]) for x in bank_routes),
        "active_ce_tokens": active_tokens, "literal_eos_occurrences": sum(sum(token == EOS for token in x["continuation_token_ids"]) for x in bank_routes),
        "positive_eos_tokens": eos_active_tokens, "positive_eos_images": eos_images,
        "positive_eos_image_fraction": len(eos_images) / len(bank_routes),
        "positive_eos_active_token_fraction": eos_active_tokens / active_tokens,
        "association_limit": "One positive EOS token and rising cap prevalence coexist, but this observational comparison does not establish EOS-supervision causality or an attention/KV mechanism.",
    }
    if active_tokens != bank_summary["active_ce_tokens"] or active_tokens != bank_acceptance["active_ce_tokens"]:
        raise ValueError("bank active-token mismatch")
    if eos_images != bank_summary["eos_positive_image_ids"] or eos_images != bank_acceptance["positive_eos_images"]:
        raise ValueError("bank EOS-image mismatch")

    cap_rows = [x for x in output_rows if x["capped"]]
    natural_rows = [x for x in output_rows if x["natural_eos"]]
    if len(cap_rows) != 18 or not all(x["exact_repeat_or_token_loop_observed"] for x in cap_rows):
        raise ValueError("cap/repetition invariant differs")
    if any(x["first_exact_duplicate_row"] for x in natural_rows):
        raise ValueError("unexpected exact row duplicate in natural-EOS output")
    expected_doses = {
        16: (4, 2, 0, 2, 2),
        32: (6, 6, 0, 4, 6),
        64: (8, 9, 4, 8, 9),
    }
    for dose in dose_summaries:
        expected = expected_doses[dose["step"]]
        observed = (dose["capped_images"], dose["planned_gt_geometry_iou50_appeared"], dose["planned_gt_geometry_exact"], dose["planned_gt_geometry_before_first_anomaly"], dose["planned_gt_geometry_before_sustained_repeat_or_collapse"])
        if observed != expected:
            raise ValueError(f"dose invariant differs: {dose['step']}: {observed}")
    control_210457 = [x for x in output_rows if x["image_id"] == 210457]
    if [(x["step"], x["stop_reason"], x["token_count"]) for x in control_210457] != [(16, "length", 3084), (32, "im_end", 48), (64, "im_end", 48)]:
        raise ValueError("210457 counterexample mismatch")
    if [x["prefix"]["training_bank_common_tokens"] for x in control_210457[1:]] != [48, 48]:
        raise ValueError("210457 bank route mismatch")

    report = {
        "schema": "training_set_completion.second_fit_stop_patterns.v1",
        "status": "candidate",
        "question": "Which observable token/row patterns accompany long or capped native second-fit outputs, and do the nine appended missing-GT geometries appear before those patterns?",
        "artifact_validity": "comparable_saved_native_readbacks",
        "dose_summaries": dose_summaries,
        "training_eos_supervision": training_eos,
        "observations": [
            "All 18 capped outputs contain an exact repeated-row run or an eventual exact token loop; none of the 15 natural-EOS outputs contains an exact duplicate parsed/raw span.",
            "The nine appended GT geometries appear by normalized-bin IoU>=0.5 in 2/9, 6/9, and 9/9 images at steps16/32/64. At step64, 8/9 appear before the first parser anomaly and all9 before sustained repetition/collapse.",
            "Four step64 geometries are exact coordinate-token matches. These matches are diagnostic and do not establish physical identity.",
            "Image210457 is capped at step16, then emits the exact 48-token bank route with natural EOS at steps32 and64, so dose alone does not force long output.",
        ],
        "interpretation": {
            "supported": "The missing-geometry teaching signal is often expressed before later repetition or structural collapse; the dominant observable failure in capped outputs is continuation/format degeneration rather than uniform failure to emit the appended geometry.",
            "not_established": "The evidence does not identify an EOS-logit, attention, or KV-cache mechanism, and it does not label unmatched valid-looking rows as false positives.",
        },
        "bounded_suggestions": [
            "Preserve and score the parser-valid prefix through the appended-geometry row separately from the degenerate suffix; the saved outputs show that target geometry can precede the stop failure.",
            "If another fit is considered, make positive terminal EOS support across routes and cap/repeat/parser-debt readback explicit acceptance variables; the current bank has only 1 positive EOS token among 1,556 active CE tokens and 1/11 positive-EOS images. This is motivated by association, not a causal claim.",
        ],
        "image_dose_index": [
            {"image_id": image_id, "doses": [
                {k: row[k] for k in ("step", "stop_reason", "token_count", "observed_failure_family", "first_observable_anomaly_generated_order", "sustained_repeat_or_collapse_generated_order", "planned_missing_gt_geometry")}
                for row in output_rows if row["image_id"] == image_id
            ]}
            for image_id in sorted(routes)
        ],
        "source_bindings": source_bindings,
        "limits": [
            "Rows are re-parsed from saved text with the native parser; normalized 0..1000 bins are used for geometry comparisons.",
            "A native parser drop is an observed syntax/geometry failure. A parser-valid novel row remains physically unreviewed.",
            "Exact periodic-tail detection requires at least 100 tokens and eight cycles with period <=64 tokens; exact repeated-row runs are reported independently.",
        ],
    }
    return report, output_rows


def validate(report: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    if report["schema"] != "training_set_completion.second_fit_stop_patterns.v1" or report["status"] != "candidate":
        raise ValueError("report identity")
    if len(rows) != 33 or {(x["step"], x["image_id"]) for x in rows} != {(step, image) for step in (16, 32, 64) for image in (25274, 59571, 99937, 210457, 219546, 323322, 351017, 388795, 417044, 477415, 528944)}:
        raise ValueError("row identity")
    if [x["capped_images"] for x in report["dose_summaries"]] != [4, 6, 8]:
        raise ValueError("cap counts")
    if [x["planned_gt_geometry_iou50_appeared"] for x in report["dose_summaries"]] != [2, 6, 9]:
        raise ValueError("target appearance counts")
    for source in report["source_bindings"]["primary"] + report["source_bindings"]["raw_rows"]:
        binding(Path(source["path"]), source["sha256"], source["size_bytes"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.write == args.validate:
        parser.error("choose exactly one of --write or --validate")
    if args.write:
        report, rows = build()
        validate(report, rows)
        report_path, rows_path = OUT / "report.json", OUT / "per-output.jsonl"
        report_path.write_bytes(canonical(report))
        rows_path.write_bytes(b"".join(canonical(row) for row in rows))
        receipt = {
            "schema": "training_set_completion.second_fit_stop_patterns.receipt.v1", "status": "candidate",
            "report": binding(report_path), "per_output": binding(rows_path), "producer": binding(Path(__file__)),
            "output_count": len(rows), "dose_summaries": report["dose_summaries"],
            "validation_command": f"python {Path(__file__)} --validate",
        }
        (OUT / "receipt.json").write_bytes(canonical(receipt))
        print(json.dumps({"status": "candidate", "dose_summaries": report["dose_summaries"]}, sort_keys=True))
    else:
        report = load(OUT / "report.json")
        rows = [json.loads(line) for line in (OUT / "per-output.jsonl").read_text().splitlines() if line.strip()]
        validate(report, rows)
        receipt = load(OUT / "receipt.json")
        binding(OUT / "report.json", receipt["report"]["sha256"], receipt["report"]["size_bytes"])
        binding(OUT / "per-output.jsonl", receipt["per_output"]["sha256"], receipt["per_output"]["size_bytes"])
        binding(Path(__file__), receipt["producer"]["sha256"], receipt["producer"]["size_bytes"])
        print(json.dumps({"status": "valid", "output_count": len(rows), "dose_summaries": report["dose_summaries"]}, sort_keys=True))


if __name__ == "__main__":
    main()
