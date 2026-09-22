#!/usr/bin/env python3
"""Build the CPU-only positive-branch input package.

This package is deliberately source-bound.  It reads the already accepted
native-escape records and the already materialized Stable50 reference inputs,
but never imports a model runtime or a tokenizer.  Token boundaries are found
from the original action-id marker sequence; decoded strings are copied from
the producer records rather than reconstructed by re-tokenization.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


BASE = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration")
NATIVE_ROOT = BASE / "2026-09-11-native-escape-witness"
PACKET_PATH = NATIVE_ROOT / "packet-v3.json"
LEAD_ACCEPTANCE_PATH = NATIVE_ROOT / "lead-acceptance.json"
FULL_RECORDS_PATH = NATIVE_ROOT / "full" / "records.jsonl"
NATIVE_CANDIDATE_MANIFEST_PATH = NATIVE_ROOT / "candidate_manifest.json"
STABLE50_INPUTS_PATH = BASE / "2026-09-11-stable50-geometric-dedup" / "inputs.json"
OUTPUT_ROOT = BASE / "2026-09-11-positive-branch-vs-repeat-event" / "input-preparation"
MANIFEST_PATH = OUTPUT_ROOT / "candidate_manifest.json"

# Qwen coordinate/object markers in the frozen action-token vocabulary.  These
# IDs are copied from the source records; they are not looked up through a
# tokenizer.
OBJECT_REF_START = 151646
OBJECT_REF_END = 151647
BOX_START = 151648
BOX_END = 151649
IM_END = 151645
COORD_BASE = 151670

COHORT = ("351017-c01", "417044-c01", "477415-c02")
EXPECTED_W = {
    "351017-c01": {"description": "wine glass", "coord_bins": [281, 361, 362, 502]},
    "417044-c01": {"description": "donut", "coord_bins": [276, 253, 349, 308]},
    "477415-c02": {"description": "person", "coord_bins": [249, 333, 362, 609]},
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_ids(ids: Sequence[int]) -> str:
    return sha256_bytes(json.dumps(list(ids), separators=(",", ":")).encode("utf-8"))


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> Any:
    require(path.is_file(), f"missing source file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def file_binding(path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    observed = sha256_file(path)
    if expected_sha256 is not None:
        require(observed == expected_sha256, f"source hash mismatch: {path}")
    return {"path": str(path), "sha256": observed, "size_bytes": path.stat().st_size}


def json_line_records(path: Path) -> list[tuple[int, bytes, Mapping[str, Any]]]:
    result: list[tuple[int, bytes, Mapping[str, Any]]] = []
    for index, raw in enumerate(path.read_bytes().splitlines(keepends=True)):
        if not raw.strip():
            continue
        result.append((index, raw, json.loads(raw.decode("utf-8"))))
    return result


def first_record(records: Sequence[tuple[int, bytes, Mapping[str, Any]]], *, job_id: str, case_id: str) -> tuple[int, bytes, Mapping[str, Any]]:
    matches = [item for item in records if item[2].get("job_id") == job_id and item[2].get("case_id") == case_id]
    require(len(matches) == 1, f"expected one {case_id}/{job_id} record, got {len(matches)}")
    return matches[0]


def packet_case(packet: Mapping[str, Any], case_id: str) -> Mapping[str, Any]:
    matches = [case for case in packet["cases"] if case.get("case_id") == case_id]
    require(len(matches) == 1, f"expected one packet case {case_id}")
    return matches[0]


def candidate_entry(native_manifest: Mapping[str, Any], candidate_id: str) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    case_id = candidate_id.split("-", 1)[0]
    matches = [case for case in native_manifest["cases"] if case.get("case_id") == case_id]
    require(len(matches) == 1, f"candidate manifest missing case {case_id}")
    case = matches[0]
    candidates = [candidate for candidate in case["candidates"] if candidate.get("candidate_id") == candidate_id]
    require(len(candidates) == 1, f"candidate manifest missing candidate {candidate_id}")
    return case, candidates[0]


def _first_index(values: Sequence[int], value: int, start: int) -> int:
    try:
        return list(values).index(value, start)
    except ValueError as exc:
        raise ValueError(f"marker {value} is missing after token {start}") from exc


def complete_row_span(ids: Sequence[int], start: int, *, label: str) -> dict[str, Any]:
    """Find one complete canonical row directly in original token IDs."""

    values = [int(token) for token in ids]
    require(0 <= start < len(values), f"{label}: row start outside sequence")
    require(values[start] == OBJECT_REF_START, f"{label}: row does not begin with object_ref_start")
    object_end = _first_index(values, OBJECT_REF_END, start + 1)
    box_start = _first_index(values, BOX_START, object_end + 1)
    box_end = _first_index(values, BOX_END, box_start + 1)
    coords = values[box_start + 1 : box_end]
    require(len(coords) == 4, f"{label}: complete row must contain four coordinate IDs")
    bins = [coord - COORD_BASE for coord in coords]
    require(all(0 <= value <= 999 for value in bins), f"{label}: coordinate ID outside coord_0..coord_999")
    end = box_end + 1
    return {
        "token_start": int(start),
        "token_end_exclusive": int(end),
        "token_length": int(end - start),
        "coordinate_positions": [box_start + 1 + offset for offset in range(4)],
        "coord_bins": bins,
        "ids": values[start:end],
    }


def free_complete_rows(ids: Sequence[int], start: int, *, label: str) -> tuple[list[dict[str, Any]], int | None]:
    """Scan a source continuation from ``h+c`` to the first terminal marker."""

    values = [int(token) for token in ids]
    cursor = int(start)
    rows: list[dict[str, Any]] = []
    terminal: int | None = None
    while cursor < len(values):
        if values[cursor] == IM_END:
            terminal = cursor
            require(cursor == len(values) - 1, f"{label}: EOS is not terminal")
            break
        row = complete_row_span(values, cursor, label=f"{label}/row-{len(rows)}")
        rows.append(row)
        cursor = row["token_end_exclusive"]
    require(terminal is not None, f"{label}: source continuation has no terminal im_end")
    require(rows, f"{label}: source continuation has no complete free row")
    return rows, terminal


def parser_prediction(record: Mapping[str, Any], generated_order: int, *, label: str) -> Mapping[str, Any]:
    parsed = record.get("parsed")
    require(isinstance(parsed, Mapping), f"{label}: missing parsed source evidence")
    matches = [row for row in parsed.get("pred", []) if row.get("generated_order") == generated_order]
    require(len(matches) == 1, f"{label}: parsed row {generated_order} is not unique")
    return matches[0]


def image_identity(source_case: Mapping[str, Any]) -> dict[str, Any]:
    plan = source_case["image_plan"]
    path = Path(source_case["image_path"])
    observed = sha256_file(path)
    require(observed == plan["image_content_sha256"], f"image hash mismatch: {path}")
    return {
        "row_id": source_case["row_id"],
        "row_index": int(source_case["row_index"]),
        "image_id": int(source_case["input_record"]["image_id"]),
        "image_path": str(path),
        "image_sha256": observed,
        "image_width": int(source_case["image_width"]),
        "image_height": int(source_case["image_height"]),
        "observed_image_grid_thw": list(plan["observed_image_grid_thw"]),
        "merged_visual_tokens": int(plan["merged_visual_tokens"]),
        "logical_transform_id": plan["logical_transform_id"],
        "backend_prompt_token_count": int(plan["backend_prompt_token_count"]),
        "executed_media_sha256": plan.get("executed_media_sha256"),
    }


def selected_adapter_identity(packet: Mapping[str, Any], stable_inputs: Mapping[str, Any]) -> dict[str, Any]:
    packet_config = packet["config"]
    stable_model = stable_inputs["model"]
    current = stable_model["current_adapter"]
    anchor_path = str(packet["anchor_adapter"])
    require(str(current["root"]) == anchor_path, "Stable50 current adapter is not packet.anchor_adapter")
    old_adapter = packet_config["adapter"]
    require(str(old_adapter["path"]) != anchor_path, "packet old source adapter unexpectedly equals Stable50 anchor")
    files = []
    for entry in current.get("files", []):
        files.append({
            "relative_path": entry["relative_path"],
            "sha256": entry["sha256"],
            "size_bytes": int(entry["size_bytes"]),
        })
    # Recheck the actual anchor files without importing a checkpoint/runtime.
    for entry in files:
        path = Path(anchor_path) / entry["relative_path"]
        require(sha256_file(path) == entry["sha256"], f"Stable50 anchor file hash mismatch: {path}")
    return {
        "base_model": packet_config["model"]["base_model"],
        "dtype": packet_config["model"]["dtype"],
        "processor": copy.deepcopy(packet_config["model"]["processor"]),
        "backend": copy.deepcopy(packet_config["backend"]),
        "embedding_delta": copy.deepcopy(packet_config["embedding_delta"]),
        "declared_packet_adapter": {
            "role": "old_source_adapter_not_effective",
            "name": old_adapter["name"],
            "type": old_adapter["type"],
            "path": old_adapter["path"],
        },
        "effective_adapter": {
            "role": "Stable50_anchor_adapter",
            "path": anchor_path,
            "kind": current["kind"],
            "fingerprint": current["fingerprint"],
            "file_count": int(current["file_count"]),
            "files": files,
        },
        "stable50_inputs_schema": stable_inputs["schema"],
        "stable50_anchor_training_receipt_sha256": stable_inputs["anchor_training_receipt_sha256"],
    }


def compact_normal_reference(reference: Mapping[str, Any], *, source_input_binding: Mapping[str, Any], index: int) -> dict[str, Any]:
    group = reference["group"]
    image_path = Path(group["image_path"])
    observed_image_sha = sha256_file(image_path)
    require(observed_image_sha == group["image_content_sha256"], f"normal image hash mismatch: {image_path}")
    ids = list(reference["action_ids"])
    require(sha256_ids(ids) == reference["action_ids_sha256"], f"normal action hash mismatch: {reference['key']}")
    prompt_ids = list(reference["prompt_token_ids"])
    require(sha256_ids(prompt_ids) == group["prompt_token_ids_sha256"], f"normal prompt hash mismatch: {reference['key']}")
    layout = copy.deepcopy(reference["initial_layout"])
    require(layout["action_token_count"] == len(ids), f"normal action/layout length mismatch: {reference['key']}")
    return {
        "source_index": int(index),
        "key": reference["key"],
        "example_id": reference["example_id"],
        "image_id": str(reference["image_id"]),
        "image": {
            "image_path": str(image_path),
            "image_sha256": observed_image_sha,
            "executed_media_sha256": group.get("executed_media_sha256"),
            "image_width": int(reference["image_width"]),
            "image_height": int(reference["image_height"]),
            "row_id": reference["case"]["row_id"],
            "row_index": int(reference["case"]["row_index"]),
            "observed_image_grid_thw": list(group["observed_image_grid_thw"]),
        },
        "prompt_token_ids": prompt_ids,
        "prompt_token_ids_sha256": group["prompt_token_ids_sha256"],
        "action_ids": ids,
        "action_ids_sha256": reference["action_ids_sha256"],
        "stop_reason": reference["stop_reason"],
        # This is the exact precomputed Stable50 layout.  In particular, do
        # not normalize away parser drops or geometry-invalid exclusions.
        "initial_layout": layout,
        "source": {
            "input_path": source_input_binding["path"],
            "input_sha256": source_input_binding["sha256"],
            "schema": "stable50_geometric_dedup.inputs.v1",
        },
    }


def positive_case(
    *,
    packet: Mapping[str, Any],
    native_manifest: Mapping[str, Any],
    records: Sequence[tuple[int, bytes, Mapping[str, Any]]],
    candidate_id: str,
    packet_binding: Mapping[str, Any],
    full_binding: Mapping[str, Any],
    candidate_binding: Mapping[str, Any],
    visual_binding: Mapping[str, Any],
    visual_note: str,
) -> dict[str, Any]:
    case_id = candidate_id.split("-", 1)[0]
    packet_case_value = packet_case(packet, case_id)
    native_case, candidate = candidate_entry(native_manifest, candidate_id)
    record_index, raw_line, record = first_record(records, job_id=f"h_plus_{candidate_id}", case_id=case_id)
    require(record["kind"] == "h_plus_c", f"{candidate_id}: source record is not h_plus_c")
    require(record["packet_sha256"] == packet_binding["sha256"], f"{candidate_id}: source record packet identity changed")
    require(record["source_row_id"] == packet_case_value["source_case"]["row_id"], f"{candidate_id}: source row identity changed")
    h_ids = list(packet_case_value["h_ids"])
    require(h_ids == list(record["h_ids"]), f"{candidate_id}: packet h differs from full record h")
    require(h_ids == list(native_case["h"]["ids"]), f"{candidate_id}: candidate manifest h differs from packet h")
    c_ids = list(candidate["c_ids"])
    require(c_ids == list(packet_case_value["candidates"][0 if candidate_id.endswith("-c01") else 1]["c_ids"]), f"{candidate_id}: packet candidate c differs")
    forced = record.get("forced_candidate")
    require(isinstance(forced, Mapping), f"{candidate_id}: missing forced candidate evidence")
    require(c_ids == list(forced["c_ids"]), f"{candidate_id}: candidate c differs from forced record c")
    require(record["action_ids"][: len(record["prefix_ids"])] == record["prefix_ids"], f"{candidate_id}: action/prefix partition mismatch")
    require(list(record["prefix_ids"]) == h_ids + c_ids, f"{candidate_id}: source prefix is not literal h+c")
    require(record["action_ids"][: len(h_ids)] == h_ids, f"{candidate_id}: h is not action prefix")
    require(sha256_ids(c_ids) == candidate["c_ids_sha256"] == forced["c_ids_sha256"], f"{candidate_id}: c hash mismatch")
    require(candidate["c_text"] == forced["c_text"], f"{candidate_id}: c text differs from forced record")
    require(candidate["c_text_sha256"] == forced["c_text_sha256"], f"{candidate_id}: c text hash differs from forced record")

    c_row = complete_row_span(c_ids, 0, label=f"{candidate_id}/c")
    require(c_row["token_end_exclusive"] == len(c_ids), f"{candidate_id}: c includes suffix outside complete row")
    require(c_row["coord_bins"] == list(candidate["coord_bins"]), f"{candidate_id}: c coordinates changed")
    require(candidate["c_text"].startswith("<|object_ref_start|>") and candidate["c_text"].endswith("<|box_end|>"), f"{candidate_id}: c grammar boundary missing")
    require(candidate["c_text"].count("<|object_ref_start|>") == 1 and candidate["c_text"].count("<|box_end|>") == 1, f"{candidate_id}: c grammar count changed")

    free_rows, eos_position = free_complete_rows(record["action_ids"], len(record["prefix_ids"]), label=f"{candidate_id}/fresh-free")
    w_row = free_rows[0]
    expected = EXPECTED_W[candidate_id]
    require(w_row["coord_bins"] == expected["coord_bins"], f"{candidate_id}: fresh w coordinate bins changed")
    # The original candidate's ``source_generated_order`` belongs to the
    # translated source continuation.  The fresh h+c record has its own
    # parser order: h rows, then the forced c row, then the first free w row.
    fresh_c_order = int(packet_case_value["h_complete_row_count"])
    parsed_c = parser_prediction(record, fresh_c_order, label=f"{candidate_id}/c")
    require(parsed_c["raw_span_text"] == candidate["c_text"], f"{candidate_id}: fresh c parser row differs")
    parsed_w = parser_prediction(record, fresh_c_order + 1, label=f"{candidate_id}/w")
    require(parsed_w["description"] == expected["description"], f"{candidate_id}: fresh w description changed")
    require(parsed_w["coord_bins"] == expected["coord_bins"], f"{candidate_id}: parser w coordinates changed")
    require(parsed_w["raw_span_text"].startswith("<|object_ref_start|>") and parsed_w["raw_span_text"].endswith("<|box_end|>"), f"{candidate_id}: w grammar boundary missing")
    require(sha256_text(parsed_w["raw_span_text"]) == parsed_w["raw_span_sha256"], f"{candidate_id}: w raw span hash mismatch")
    w_ids = list(w_row["ids"])
    require(w_ids == list(record["action_ids"])[w_row["token_start"] : w_row["token_end_exclusive"]], f"{candidate_id}: w source slice mismatch")
    require(w_row["token_start"] == len(record["prefix_ids"]), f"{candidate_id}: w is not immediately after h+c")
    require(record["action_ids"][-1] == IM_END and record["native_eos_observed"] is True, f"{candidate_id}: source EOS evidence missing")
    require(eos_position == len(record["action_ids"]) - 1, f"{candidate_id}: EOS position is not terminal")

    h_len = len(h_ids)
    c_start = h_len
    c_end = c_start + len(c_ids)
    w_start = w_row["token_start"]
    w_end = w_row["token_end_exclusive"]
    require([c_start, c_end] == [len(h_ids), len(h_ids) + len(c_ids)], f"{candidate_id}: c offsets drifted")
    require([w_start, w_end] == [len(record["prefix_ids"]), len(record["prefix_ids"]) + len(w_ids)], f"{candidate_id}: w offsets drifted")
    c_positions = list(range(c_start, c_end))
    w_positions = list(range(w_start, w_end))
    require(IM_END not in c_ids and IM_END not in w_ids, f"{candidate_id}: EOS entered c/w literal")
    require(eos_position not in c_positions and eos_position not in w_positions, f"{candidate_id}: EOS entered c/w mask")

    source_case = image_identity(packet_case_value["source_case"])
    line_sha = sha256_bytes(raw_line)
    return {
        "candidate_id": candidate_id,
        "case_id": case_id,
        "admission": "root_visually_admitted_conditional_protection_witness",
        "visual_admission": {
            "receipt": {"path": visual_binding["path"], "sha256": visual_binding["sha256"]},
            "candidate_visual": copy.deepcopy(candidate.get("visual_receipt")),
            "lead_note": visual_note,
            "scope": "root visual admission of this c and the first complete free row w; no suffix-wide correctness claim",
        },
        "image": source_case,
        "prompt": {
            "token_ids": list(packet_case_value["prompt_token_ids"]),
            "length": len(packet_case_value["prompt_token_ids"]),
            "ids_sha256": sha256_ids(packet_case_value["prompt_token_ids"]),
        },
        "h": {
            "job_id": packet_case_value["h_source_job_id"],
            "history_condition": "original_native",
            "token_ids": h_ids,
            "length": len(h_ids),
            "ids_sha256": sha256_ids(h_ids),
            "text": native_case["h"].get("text"),
            "text_sha256": native_case["h"].get("text_sha256"),
        },
        "c": {
            "candidate_id": candidate_id,
            "description": candidate["description"],
            "coord_bins": list(candidate["coord_bins"]),
            "token_ids": c_ids,
            "length": len(c_ids),
            "ids_sha256": sha256_ids(c_ids),
            "text": candidate["c_text"],
            "text_sha256": candidate["c_text_sha256"],
            "source_row_id": candidate["source_row_id"],
            "source_generated_order": int(candidate["source_generated_order"]),
            "complete_row_grammar": {
                "start": "object_ref_start",
                "end": "box_end",
                "token_start_in_c": 0,
                "token_end_exclusive_in_c": len(c_ids),
                "includes_object_ref_start": True,
                "includes_box_end": True,
            },
        },
        "w": {
            "selection": "first_complete_free_row_after_exact_h_plus_c",
            "description": parsed_w["description"],
            "coord_bins": list(parsed_w["coord_bins"]),
            "token_ids": w_ids,
            "length": len(w_ids),
            "ids_sha256": sha256_ids(w_ids),
            "text": parsed_w["raw_span_text"],
            "text_sha256": parsed_w["raw_span_sha256"],
            "parser_row": {
                "generated_order": int(parsed_w["generated_order"]),
                "object_span_id": parsed_w.get("object_span_id"),
                "bbox_pixel_xyxy": parsed_w.get("bbox"),
                "parser_disposition": parsed_w.get("parser_disposition"),
            },
            "complete_row_grammar": {
                "start": "object_ref_start",
                "end": "box_end",
                "token_start_in_source_action": int(w_start),
                "token_end_exclusive_in_source_action": int(w_end),
                "includes_object_ref_start": True,
                "includes_box_end": True,
            },
        },
        "continuation_offsets": {
            "index_frame": "source action_ids / continuation-relative; index 0 is the first token after the prompt",
            "h_span": [0, h_len],
            "c_target_span": [c_start, c_end],
            "w_only_kl_span": [w_start, w_end],
            "c_target_positions": c_positions,
            "w_only_kl_positions": w_positions,
            "target_spans_cover_only_c_and_w": True,
            "terminal_eos_id": IM_END,
            "terminal_eos_position_in_source_action": int(eos_position),
            "eos_in_c_target": False,
            "eos_in_w_only_kl": False,
        },
        "source_record": {
            "path": full_binding["path"],
            "file_sha256": full_binding["sha256"],
            "line_index_zero_based": int(record_index),
            "line_sha256_including_newline": line_sha,
            "job_id": record["job_id"],
            "kind": record["kind"],
            "request_id": record["request_id"],
            "source_row_id": record["source_row_id"],
            "packet_sha256": record["packet_sha256"],
            "action_ids_length": len(record["action_ids"]),
            "action_ids_sha256": sha256_ids(record["action_ids"]),
            "prefix_ids_length": len(record["prefix_ids"]),
            "prefix_ids_sha256": sha256_ids(record["prefix_ids"]),
            "native_eos_observed": bool(record["native_eos_observed"]),
            "stop_reason": record["stop_reason"],
        },
        "source_bindings": {
            "packet_v3": copy.deepcopy(packet_binding),
            "native_candidate_manifest": copy.deepcopy(candidate_binding),
            "lead_acceptance": {"path": str(LEAD_ACCEPTANCE_PATH), "sha256": sha256_file(LEAD_ACCEPTANCE_PATH)},
        },
    }


def build_manifest() -> dict[str, Any]:
    packet = load_json(PACKET_PATH)
    lead = load_json(LEAD_ACCEPTANCE_PATH)
    native_manifest = load_json(NATIVE_CANDIDATE_MANIFEST_PATH)
    stable_inputs = load_json(STABLE50_INPUTS_PATH)
    records = json_line_records(FULL_RECORDS_PATH)

    packet_sha = sha256_file(PACKET_PATH)
    lead_sha = sha256_file(LEAD_ACCEPTANCE_PATH)
    full_sha = sha256_file(FULL_RECORDS_PATH)
    candidate_sha = sha256_file(NATIVE_CANDIDATE_MANIFEST_PATH)
    stable_sha = sha256_file(STABLE50_INPUTS_PATH)
    require(packet_sha == lead["source_hashes"][str(PACKET_PATH)], "packet-v3 hash differs from lead acceptance")
    require(full_sha == lead["source_hashes"][str(FULL_RECORDS_PATH)], "full records hash differs from lead acceptance")
    require(candidate_sha == packet["candidate_manifest_sha256"], "native candidate manifest hash differs from packet-v3")
    source_bindings = {
        "packet_v3": file_binding(PACKET_PATH, expected_sha256=packet_sha),
        "lead_acceptance": file_binding(LEAD_ACCEPTANCE_PATH, expected_sha256=lead_sha),
        "full_records": file_binding(FULL_RECORDS_PATH, expected_sha256=full_sha),
        "native_candidate_manifest": file_binding(NATIVE_CANDIDATE_MANIFEST_PATH, expected_sha256=candidate_sha),
        "stable50_inputs": file_binding(STABLE50_INPUTS_PATH, expected_sha256=stable_sha),
        "visual_admission_receipt": {
            "path": str(packet["visual_admission_receipt"]),
            "sha256": sha256_file(Path(packet["visual_admission_receipt"])),
        },
    }
    require(source_bindings["visual_admission_receipt"]["sha256"] == packet["visual_admission_receipt_sha256"], "visual receipt hash mismatch")
    selected_adapter = selected_adapter_identity(packet, stable_inputs)

    normals = [
        compact_normal_reference(reference, source_input_binding=source_bindings["stable50_inputs"], index=index)
        for index, reference in enumerate(stable_inputs["reference_cases"])
    ]
    require(len(normals) == 56, "Stable50 normal reference count changed")
    positive_images = {str(case["image"]["image_id"]) for case in []}
    positives = [
        positive_case(
            packet=packet,
            native_manifest=native_manifest,
            records=records,
            candidate_id=candidate_id,
            packet_binding=source_bindings["packet_v3"],
            full_binding=source_bindings["full_records"],
            candidate_binding=source_bindings["native_candidate_manifest"],
            visual_binding=source_bindings["visual_admission_receipt"],
            visual_note=lead["visual_adjudication"]["notes"][candidate_id],
        )
        for candidate_id in COHORT
    ]
    positive_images = {str(case["image"]["image_id"]) for case in positives}
    normal_images = {str(case["image_id"]) for case in normals}
    require(len(positive_images) == 3, "positive cohort reuses an image")
    require(not positive_images.intersection(normal_images), "positive and normal image populations overlap")

    invalid_normals = [case for case in normals if case["initial_layout"]["invalid_geometry_rows"]]
    parser_drop_normals = [case for case in normals if case["initial_layout"]["parser_drops"]]
    excluded_positions = {
        case["image_id"]: [
            position
            for row_index in case["initial_layout"]["invalid_geometry_rows"]
            for row in case["initial_layout"]["parser_drop_rows"]
            if row.get("generated_order") == row_index
            for position in row.get("token_positions", [])
        ]
        for case in invalid_normals
    }
    return {
        "schema": "positive_branch_vs_repeat_event.input_preparation.v1",
        "status": "candidate_only",
        "immutable_candidate_manifest": True,
        "claim_boundary": {
            "scope": "CPU-only source packaging for three root-admitted conditional protection witnesses",
            "not_claimed": [
                "no model inference or training",
                "no natural retrieval or owner-complete guarantee",
                "no suffix-wide correctness or deployment promotion",
            ],
        },
        "research_identity": {
            "experiment": "positive branch vs repeat event",
            "positive_cohort": list(COHORT),
            "positive_selection": "root visually admitted c followed by the first complete free row w in the corresponding fresh h+c record",
            "normal_protection": "exact 56 existing Stable50 reference cases, including parser and geometry-invalid mask evidence",
        },
        "source_bindings": source_bindings,
        "model_identity": selected_adapter,
        "positives": positives,
        "normals": {
            "source": {
                "path": source_bindings["stable50_inputs"]["path"],
                "sha256": source_bindings["stable50_inputs"]["sha256"],
                "schema": stable_inputs["schema"],
            },
            "count": len(normals),
            "cases": normals,
            "mask_summary": {
                "reference_count": len(normals),
                "parser_clean_reference_count": len(normals) - len(parser_drop_normals),
                "parser_nonclean_reference_count": len(parser_drop_normals),
                "geometry_invalid_reference_count": len(invalid_normals),
                "geometry_invalid_reference_image_ids": [case["image_id"] for case in invalid_normals],
                "geometry_invalid_excluded_positions": excluded_positions,
                "note": "Do not treat all normal references as parser-clean; 360573 has one dropped 9-token geometry-invalid row excluded from KL.",
            },
        },
        "counts": {
            "positive_count": len(positives),
            "normal_count": len(normals),
            "positive_c_tokens": sum(len(case["c"]["token_ids"]) for case in positives),
            "positive_w_tokens": sum(len(case["w"]["token_ids"]) for case in positives),
            "positive_h_tokens": sum(len(case["h"]["token_ids"]) for case in positives),
            "normal_action_tokens": sum(len(case["action_ids"]) for case in normals),
            "normal_kl_positions": sum(len(case["initial_layout"]["kl_positions"]) for case in normals),
        },
        "build_receipt": {
            "builder_path": str(Path(__file__).resolve()),
            "builder_sha256": sha256_file(Path(__file__).resolve()),
            "no_model_import": True,
            "no_tokenizer_import": True,
            "no_gpu_calls": True,
            "row_boundary_method": "direct original action-id marker spans; no retokenization",
            "packet_v3_status": packet["status"],
            "lead_acceptance_status": lead["status"],
            "stable50_schema": stable_inputs["schema"],
        },
    }


def validate_manifest(manifest: Mapping[str, Any], *, check_sources: bool = False) -> None:
    require(manifest.get("schema") == "positive_branch_vs_repeat_event.input_preparation.v1", "wrong package schema")
    require(manifest.get("status") == "candidate_only", "package is not candidate-only")
    require(manifest.get("immutable_candidate_manifest") is True, "package is not immutable")
    positives = manifest.get("positives")
    require(isinstance(positives, list) and len(positives) == 3, "positive count is not three")
    require([case.get("candidate_id") for case in positives] == list(COHORT), "positive cohort/order changed")
    positive_images: set[str] = set()
    for case in positives:
        candidate_id = str(case["candidate_id"])
        h = case["h"]
        c = case["c"]
        w = case["w"]
        offsets = case["continuation_offsets"]
        h_ids, c_ids, w_ids = list(h["token_ids"]), list(c["token_ids"]), list(w["token_ids"])
        require(h["length"] == len(h_ids) and h["ids_sha256"] == sha256_ids(h_ids), f"{candidate_id}: h literal/hash invalid")
        require(c["length"] == len(c_ids) and c["ids_sha256"] == sha256_ids(c_ids), f"{candidate_id}: c literal/hash invalid")
        require(w["length"] == len(w_ids) and w["ids_sha256"] == sha256_ids(w_ids), f"{candidate_id}: w literal/hash invalid")
        c_span = offsets["c_target_span"]
        w_span = offsets["w_only_kl_span"]
        require(c_span == [len(h_ids), len(h_ids) + len(c_ids)], f"{candidate_id}: shifted c target boundary")
        require(w_span == [len(h_ids) + len(c_ids), len(h_ids) + len(c_ids) + len(w_ids)], f"{candidate_id}: shifted w KL boundary")
        require(offsets["c_target_positions"] == list(range(*c_span)), f"{candidate_id}: c positions do not cover c exactly")
        require(offsets["w_only_kl_positions"] == list(range(*w_span)), f"{candidate_id}: w positions do not cover w exactly")
        require(offsets["target_spans_cover_only_c_and_w"] is True, f"{candidate_id}: target coverage flag changed")
        require(offsets["eos_in_c_target"] is False and offsets["eos_in_w_only_kl"] is False, f"{candidate_id}: EOS policy changed")
        require(IM_END not in c_ids + w_ids, f"{candidate_id}: extra EOS is inside c/w target literals")
        require(c["complete_row_grammar"]["token_start_in_c"] == 0, f"{candidate_id}: c row start shifted")
        require(c["complete_row_grammar"]["token_end_exclusive_in_c"] == len(c_ids), f"{candidate_id}: c row end shifted")
        require(w["complete_row_grammar"]["token_end_exclusive_in_source_action"] - w["complete_row_grammar"]["token_start_in_source_action"] == len(w_ids), f"{candidate_id}: w row span length changed")
        require(w["complete_row_grammar"]["includes_object_ref_start"] is True and w["complete_row_grammar"]["includes_box_end"] is True, f"{candidate_id}: w grammar boundary changed")
        require(w["coord_bins"] == EXPECTED_W[candidate_id]["coord_bins"], f"{candidate_id}: expected w coordinates changed")
        require(w["description"] == EXPECTED_W[candidate_id]["description"], f"{candidate_id}: expected w description changed")
        require(case["source_record"]["native_eos_observed"] is True, f"{candidate_id}: source EOS evidence absent")
        require(case["source_record"]["stop_reason"] == "im_end", f"{candidate_id}: source stop is not im_end")
        eos_position = int(offsets["terminal_eos_position_in_source_action"])
        require(eos_position not in offsets["c_target_positions"] + offsets["w_only_kl_positions"], f"{candidate_id}: EOS is silently supervised")
        image_id = str(case["image"]["image_id"])
        require(image_id not in positive_images, f"positive image repeated: {image_id}")
        positive_images.add(image_id)
    normals = manifest.get("normals", {}).get("cases", [])
    require(len(normals) == 56, "normal reference count is not 56")
    normal_images = {str(case["image_id"]) for case in normals}
    require(not positive_images.intersection(normal_images), "positive image overlaps normal reference")
    require(manifest["normals"]["mask_summary"]["geometry_invalid_reference_image_ids"] == ["360573"], "geometry-invalid normal identity changed")
    require(manifest["normals"]["mask_summary"]["parser_nonclean_reference_count"] == 1, "normal parser status was over-cleaned")
    for case in normals:
        ids = list(case["action_ids"])
        require(case["action_ids_sha256"] == sha256_ids(ids), f"normal action hash mismatch: {case['key']}")
        layout = case["initial_layout"]
        require(layout["action_token_count"] == len(ids), f"normal layout count mismatch: {case['key']}")
        require(set(layout["kl_positions"]).issubset(set(range(len(ids)))), f"normal KL positions out of range: {case['key']}")
    if check_sources:
        bindings = manifest["source_bindings"]
        for binding in bindings.values():
            if isinstance(binding, Mapping) and "path" in binding and "sha256" in binding:
                require(sha256_file(Path(binding["path"])) == binding["sha256"], f"bound source changed: {binding['path']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args()
    manifest = build_manifest()
    validate_manifest(manifest, check_sources=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sha256": sha256_file(args.output), "counts": manifest["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
