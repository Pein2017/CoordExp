"""Build the reviewed stage-03 forced-history repair route bank on CPU.

The bank preserves every generated token.  Reviewed source-prefix rows and the
single root-accepted forced GT row are the only object fields eligible for CE;
all generated suffix objects remain context with zero CE weight.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from probes.dora_owner_learning.geometric_dedup import (
    _character_span_to_token_interval,
    _exact_token_text_frame,
)
from probes.training_set_completion.route_bank import _load_tokenizer, _mask_fields


SCHEMA = "training_set_completion.stage03_repair_route_bank.v1"
RUNTIME_SCHEMA = f"{SCHEMA}.runtime_route_list.v1"
IMAGE_IDS = (25274, 59571, 99937, 210457, 219546, 323322, 351017, 388795, 417044, 477415, 528944)
RETAINED_IMAGE_ID = 210457
CONDITIONAL_EOS_IMAGE_ID = 323322
EOS = 151645


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def binding(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    require(resolved.is_file(), f"bound source is not a file: {resolved}")
    return {"path": str(resolved), "sha256": file_hash(resolved), "size_bytes": resolved.stat().st_size}


def verify_binding(value: Mapping[str, Any], label: str) -> None:
    require(set(value) == {"path", "sha256", "size_bytes"}, f"{label} binding fields")
    require(binding(value["path"]) == dict(value), f"{label} bytes changed")


def _verify_adapter(value: Mapping[str, Any]) -> None:
    root = Path(value["root"]).resolve(strict=True)
    files = value.get("files", [])
    require(value.get("kind") == "dora_adapter" and files and value.get("file_count") == len(files), "parent adapter identity")
    for item in files:
        path = root / item["relative_path"]
        require(path.is_file() and path.stat().st_size == item["size_bytes"] and file_hash(path) == item["sha256"], "parent adapter file changed")


def publish(path: Path, value: Any) -> None:
    require(not path.exists(), f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical(value)
    path.write_bytes(payload)
    require(path.read_bytes() == payload, f"publication readback differs: {path}")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _content_hash(value: Mapping[str, Any]) -> str:
    return digest({key: item for key, item in value.items() if key != "content_sha256"})


def _atomic_targets(catalog: Mapping[str, Any]) -> dict[int, set[str]]:
    require(catalog.get("status") == "lead_accepted_target_reference_catalog", "target v3 is not lead accepted")
    records = catalog.get("records", [])
    require(catalog.get("atomic_target_count") == 228 and len(records) == 228, "target v3 atomic denominator")
    by_image: dict[int, set[str]] = defaultdict(set)
    for row in records:
        image, owner = int(row["image_id"]), str(row["owner_id"])
        require(image in IMAGE_IDS and owner not in by_image[image], "target v3 identity")
        by_image[image].add(owner)
    require(set(by_image) == set(IMAGE_IDS) and sum(map(len, by_image.values())) == 228, "target v3 cohort")
    return by_image


def _validate_repair_manifest(manifest: Mapping[str, Any]) -> None:
    require(manifest.get("schema") == "training_set_completion.stage03_single_owner_repair.v1", "repair manifest schema")
    require(manifest.get("status") == "candidate_ready" and manifest.get("content_sha256") == _content_hash(manifest), "repair manifest content")
    require(len(manifest.get("requests", [])) == 40 and len(manifest.get("prefixes", {})) == 11, "repair request/prefix denominator")
    require(manifest.get("runtime", {}).get("cap") == 3084 and manifest.get("runtime", {}).get("eos") == EOS, "repair cap/EOS")
    require(set(int(key) for key in manifest["prefixes"]) == set(IMAGE_IDS), "repair prefix cohort")


def _row_path(repair_root: Path, request_id: str) -> Path:
    return repair_root / "rows" / f"{hashlib.sha256(request_id.encode()).hexdigest()}.json"


def _validate_generated_row(row: Mapping[str, Any], request: Mapping[str, Any], manifest: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    prefix = list(row.get("forced_prefix_token_ids", []))
    suffix = list(row.get("generated_suffix_token_ids", []))
    full = list(row.get("continuation_token_ids", []))
    if row.get("manifest_sha256") != manifest.get("content_sha256"):
        reasons.append("manifest_mismatch")
    if row.get("request") != request or int(row.get("image_id", -1)) != int(request["image_id"]):
        reasons.append("request_mismatch")
    if full != prefix + suffix or int(row.get("prefix_token_count", -1)) != len(prefix) or int(row.get("suffix_token_count", -1)) != len(suffix):
        reasons.append("literal_prefix_suffix_mismatch")
    expected = manifest["prefixes"][str(request["image_id"])]
    if prefix != expected.get("forced_prefix") or row.get("forced_prefix_sha256") != digest(prefix):
        reasons.append("bound_prefix_mismatch")
    if row.get("continuation_token_ids_sha256") != digest(full) or row.get("generated_suffix_sha256") != digest(suffix):
        reasons.append("token_hash_mismatch")
    cap = int(manifest["runtime"]["cap"])
    if row.get("assistant_token_cap") != cap or len(full) > cap:
        reasons.append("cap_invalid")
    if row.get("decode_stop_reason") != "im_end" or not suffix or suffix[-1] != EOS or EOS in suffix[:-1] or EOS in prefix:
        reasons.append("not_natural_terminal_eos")
    return reasons


def choose_release(rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Choose shortest valid suffix, then the lower declared temperature."""
    requests = {str(item["request_id"]): item for item in manifest["requests"]}
    require(len(rows) == 4 and len({str(row["request"]["request_id"]) for row in rows}) == 4, "four repair candidates/image")
    cards = []
    eligible = []
    for row in rows:
        request_id = str(row["request"]["request_id"])
        require(request_id in requests, "repair row request outside manifest")
        reasons = _validate_generated_row(row, requests[request_id], manifest)
        card = {
            "request_id": request_id,
            "temperature": float(requests[request_id]["temperature"]),
            "suffix_token_count": len(row.get("generated_suffix_token_ids", [])),
            "eligible": not reasons,
            "rejection_reasons": reasons,
        }
        cards.append(card)
        if not reasons:
            eligible.append((card["suffix_token_count"], card["temperature"], request_id, row))
    require(eligible, f"no cap-valid natural-EOS release for image {rows[0].get('image_id')}")
    _, _, request_id, selected = min(eligible, key=lambda item: item[:3])
    return selected, {
        "criterion": "shortest generated suffix among cap-valid natural-terminal-EOS rows; tie lower temperature then request_id",
        "selected_request_id": request_id,
        "candidates": sorted(cards, key=lambda item: item["request_id"]),
    }


def _native_parse(ids: Sequence[int], tokenizer: Any, record: Mapping[str, Any], stop_reason: str) -> tuple[str, list[tuple[int, int]], dict[str, Any]]:
    from src.eval.native_rows import native_detection_record as native_record

    text, spans = _exact_token_text_frame(ids, tokenizer)
    parsed = native_record(text, record["case"], record["golden"], stop_reason)
    return text, spans, parsed


def _object_index(parsed: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    result = {}
    for row in [*parsed.get("pred", []), *parsed.get("dropped_predictions", [])]:
        order = int(row["generated_order"])
        require(order not in result, "duplicate parsed raw order")
        result[order] = dict(row)
    return result


def _interval(text: str, spans: Sequence[tuple[int, int]], row: Mapping[str, Any]) -> list[int]:
    left, right = _character_span_to_token_interval(int(row["char_start"]), int(row["char_end"]), text=text, token_spans=spans)
    return list(range(left, right))


def _adapt_decision(decision: Mapping[str, Any]) -> dict[str, Any]:
    raw = decision["raw"]
    return {
        "proposal_id": decision["proposal_id"],
        "root_owner_id": decision.get("effective_owner_id"),
        "reviewed_owner_id": decision.get("effective_owner_id"),
        "reviewed": True,
        "physical_status": decision["physical_status"],
        "root_repeat_after_alias": decision["physical_status"] == "repeat",
        "parser_status": decision["packet_raw_row_status"],
        "raw_geometry_invalid": raw.get("raw_axes_preserved") is not True or raw.get("status") != "parsed_valid",
        "effective_extent": decision["effective_extent"],
        "effective_class": decision["effective_class"],
        "effective_direct_CE": decision["effective_direct_CE"],
        "coord_bins_1000": list(raw["coord_bins_1000"]),
    }


def _mask_reviewed_prefix(
    *, ids: Sequence[int], tokenizer: Any, record: Mapping[str, Any], stop_reason: str,
    decisions: Sequence[Mapping[str, Any]], targets: set[str], prefix_source_token_count: int,
    coordinate_ids: Sequence[int],
) -> tuple[list[int], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    text, spans, parsed = _native_parse(ids, tokenizer, record, stop_reason)
    objects = _object_index(parsed)
    weights = [0] * len(ids)
    trace, boxes = [], []
    seen_orders = set()
    for decision in sorted(decisions, key=lambda item: int(item["generated_order"])):
        order = int(decision["generated_order"])
        require(order in objects, f"reviewed prefix row missing from literal parse: {decision['proposal_id']}")
        pred = dict(objects[order]); raw = decision["raw"]
        row_positions = _interval(text, spans, pred)
        require(row_positions and max(row_positions) < prefix_source_token_count, "reviewed row extends outside literal source prefix")
        pred["coord_bins"] = list(raw["coord_bins_1000"])
        pred["description"] = raw["description"]
        pred["_token_ids"] = list(ids)
        require(list(objects[order].get("coord_bins", raw["coord_bins_1000"])) == list(raw["coord_bins_1000"]), "reviewed coordinate differs from literal row")
        require(objects[order].get("description", raw["description"]) == raw["description"], "reviewed description differs from literal row")
        active, card = _mask_fields(
            acquisition_pred=pred, text=text, token_spans=spans, review=_adapt_decision(decision),
            targets=targets, coordinate_ids=coordinate_ids,
        )
        # Atomic owner membership is the final authority even if an upstream
        # projection accidentally marks a crowd/non-target row positive.
        if decision.get("effective_owner_id") not in targets:
            active = set(); card["positive_positions"] = []; card["reason"] = "masked_non_atomic_owner"
        for position in active:
            weights[position] = 1
        if active:
            coordinate_positions = list(card["coordinate_positions"])
            require(len(coordinate_positions) == 4 and set(coordinate_positions) <= active, "active geometry coordinate binding")
            boxes.append({
                "x1_position": coordinate_positions[0], "y1_position": coordinate_positions[1],
                "x2_position": coordinate_positions[2], "y2_position": coordinate_positions[3],
                "expected_bins": list(card["expected_bins"]),
            })
        card.update({"kind": "reviewed_source_prefix", "image_id": int(decision["image_id"]), "generated_order": order})
        trace.append(card); seen_orders.add(order)
    return weights, trace, boxes, {"text": text, "spans": spans, "parsed": parsed, "objects": objects, "seen_orders": seen_orders}


def _forced_row_card(
    *, ids: Sequence[int], parse_frame: Mapping[str, Any], forced: Mapping[str, Any], prefix_token_count: int,
    targets: set[str], coordinate_ids: Sequence[int], accepted: Mapping[str, Any], weights: list[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    appended = list(forced["appended_row_token_ids"])
    expected_positions = list(range(prefix_token_count - len(appended), prefix_token_count))
    require(list(ids)[expected_positions[0]:prefix_token_count] == appended, "forced row literal IDs changed")
    matches = []
    for order, row in parse_frame["objects"].items():
        if _interval(parse_frame["text"], parse_frame["spans"], row) == expected_positions:
            matches.append((order, row))
    require(len(matches) == 1, "forced row does not map to one exact raw object span")
    order, pred = matches[0]
    pred = dict(pred); pred["coord_bins"] = list(forced["reference_bins"]); pred["description"] = forced["category"]; pred["_token_ids"] = list(ids)
    review = {
        "proposal_id": f"stage03:forced:image-{accepted['image_id']:012d}:owner-{accepted['owner_id']}",
        "root_owner_id": accepted["owner_id"], "reviewed_owner_id": accepted["owner_id"], "reviewed": True,
        "physical_status": "true_unique", "root_repeat_after_alias": False, "parser_status": "parsed_valid",
        "raw_geometry_invalid": False, "effective_extent": "reasonable", "effective_class": "verified",
        "effective_direct_CE": {"bbox": "positive", "description": "positive"},
        "coord_bins_1000": list(forced["reference_bins"]),
    }
    require(str(accepted["owner_id"]) == str(forced["owner_id"]) and list(accepted["reference_bins"]) == list(forced["reference_bins"]), "forced review differs from manifest owner/reference")
    require(str(forced["owner_id"]) in targets, "forced owner is outside atomic target v3")
    active, card = _mask_fields(
        acquisition_pred=pred, text=parse_frame["text"], token_spans=parse_frame["spans"], review=review,
        targets=targets, coordinate_ids=coordinate_ids,
    )
    require(active and set(card["row_token_positions"]) == set(expected_positions), "forced row mask span")
    for position in active:
        weights[position] = 1
    positions = list(card["coordinate_positions"])
    box = {"x1_position": positions[0], "y1_position": positions[1], "x2_position": positions[2], "y2_position": positions[3], "expected_bins": list(card["expected_bins"])}
    card.update({"kind": "root_accepted_forced_gt", "image_id": int(accepted["image_id"]), "generated_order": int(order), "acceptance_decision": "positive_bbox_and_description"})
    return card, box


def _suffix_trace(*, ids: Sequence[int], parse_frame: Mapping[str, Any], prefix_token_count: int, excluded_orders: set[int]) -> list[dict[str, Any]]:
    result = []
    for order, row in sorted(parse_frame["objects"].items()):
        if order in excluded_orders:
            continue
        positions = _interval(parse_frame["text"], parse_frame["spans"], row)
        if positions and min(positions) >= prefix_token_count:
            result.append({
                "kind": "unreviewed_generated_suffix", "generated_order": order,
                "row_token_positions": positions, "positive_positions": [], "reason": "suffix_rows_are_context_only",
            })
    return result


def _source_record_map(manifest: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    result = {int(row["image_id"]): row for row in manifest["records"]}
    require(set(result) == set(IMAGE_IDS), "repair source record cohort")
    return result


def _readback_map(path: Path) -> dict[int, Mapping[str, Any]]:
    payload = read_json(path)
    rows = {int(row["image_id"]): row for row in payload["rows"]}
    require(set(rows) == set(IMAGE_IDS), "readback cohort")
    return rows


def _selected_rows(repair_root: Path, manifest: Mapping[str, Any]) -> tuple[dict[int, Mapping[str, Any]], dict[int, Any], list[dict[str, Any]]]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    bindings = []
    for request in manifest["requests"]:
        path = _row_path(repair_root, request["request_id"])
        row = read_json(path)
        grouped[int(request["image_id"])].append(row)
        bindings.append(binding(path))
    require(set(grouped) == set(IMAGE_IDS) - {RETAINED_IMAGE_ID}, "generated repair image cohort")
    selected, selection = {}, {}
    for image in sorted(grouped):
        selected[image], selection[image] = choose_release(grouped[image], manifest)
    return selected, selection, sorted(bindings, key=lambda item: item["path"])


def _validate_forced_acceptance(value: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    require(value.get("schema") == "stage03.forced_owner_reference_acceptance.v1" and value.get("status") == "lead_accepted", "forced reference acceptance status")
    rows = {int(row["image_id"]): row for row in value.get("rows", [])}
    required = set(IMAGE_IDS) - {RETAINED_IMAGE_ID}
    require(set(rows) == required and len(rows) == 10, "forced reference acceptance denominator")
    for image, row in rows.items():
        forced = manifest["prefixes"][str(image)]["forced_owner"]
        require(
            row.get("class") == "verified_existing_gt"
            and row.get("extent") == "reasonable_visible_gt_extent"
            and row.get("direct_CE") == {"bbox": "positive", "description": "positive"},
            "forced reference is not positive accepted",
        )
        require(str(row["owner_id"]) == str(forced["owner_id"]) and list(row["reference_bins"]) == list(forced["reference_bins"]), "forced acceptance identity/reference")
    return rows


def _allow_323322_eos(
    *, suffix: Sequence[int], source_decisions: Sequence[Mapping[str, Any]], forced_owner_id: str, targets: set[str],
) -> tuple[bool, set[str], bool]:
    known = {str(row["effective_owner_id"]) for row in source_decisions} | {str(forced_owner_id)}
    debt = any(
        row["effective_owner_id"] not in targets
        or row["physical_status"] != "true_unique"
        or row["effective_extent"] != "reasonable"
        or row["effective_class"] != "verified"
        or row["effective_direct_CE"] != {"bbox": "positive", "description": "positive"}
        for row in source_decisions
    )
    allowed = list(suffix) == [EOS] and len(source_decisions) == 6 and known == targets and not debt
    return allowed, known, debt


def _route_record(
    *, image: int, ids: list[int], weights: list[int], boxes: list[dict[str, Any]], trace: list[dict[str, Any]],
    source_record: Mapping[str, Any], route_id: str, eos_positive: bool, provenance: Mapping[str, Any],
) -> dict[str, Any]:
    require(len(ids) == len(weights) and ids and any(weights), "route literal/mask lengths")
    require(ids[-1] == EOS, "route lacks terminal EOS")
    eos_positions = [index for index, token in enumerate(ids) if token == EOS and weights[index]]
    require(eos_positions == ([len(ids) - 1] if eos_positive else []), "route EOS mask contract")
    for box in boxes:
        positions = [box[key] for key in ("x1_position", "y1_position", "x2_position", "y2_position")]
        require(all(weights[position] == 1 for position in positions), "trusted box coordinate is masked")
    return {
        "route_id": route_id, "image_id": image, "example_id": source_record["example_id"],
        "continuation_token_ids": ids, "continuation_token_ids_sha256": digest(ids),
        "ce_weights": weights, "ce_weights_sha256": digest(weights),
        "trusted_boxes": boxes, "trusted_complete_support_endpoint": eos_positive,
        "proposal_trace_sha256": digest(trace), "provenance": dict(provenance),
    }


def build(
    *, repair_root: Path, mask_decisions: Path, mask_source_manifest: Path, target_catalog: Path,
    mask_acceptance: Path, forced_acceptance: Path, consumer_policy_correction: Path,
    producer_acceptance: Path, retained_route: Path, output: Path,
) -> dict[str, Any]:
    final_names = ("bank.json", "proposal-trace.jsonl", "runtime-route-list.json", "summary.json")
    require(not any((output / name).exists() for name in final_names), f"final output exists: {output}")
    superseded = []
    if output.exists():
        unexpected = [item for item in output.iterdir() if not item.is_dir() or not item.name.startswith("superseded-attempt-")]
        require(not unexpected, f"unexpected preexisting output: {unexpected}")
        for directory in sorted(output.iterdir()):
            superseded.append({"directory": str(directory), "files": [binding(path) for path in sorted(directory.iterdir()) if path.is_file()]})
    manifest_path = repair_root / "manifest.json"; result_path = repair_root / "result.json"
    manifest = read_json(manifest_path); _validate_repair_manifest(manifest)
    result = read_json(result_path)
    require(result.get("status") == "candidate_ready" and result.get("request_count") == 40, "repair result incomplete")
    correction = read_json(consumer_policy_correction)
    require(correction.get("status") == "candidate_ready", "consumer policy correction status")
    require(correction.get("scientific_consumer_policy", {}).get("empty_assistant_prefix") is False, "forced histories must be nonempty for consumers")
    require(correction.get("manifest") == binding(manifest_path) and correction.get("result") == binding(result_path), "consumer correction repair bindings")
    producer_acceptance_payload = read_json(producer_acceptance)
    require(
        producer_acceptance_payload.get("schema") == "stage03.root_acceptance.v1"
        and producer_acceptance_payload.get("status") == "lead_accepted_conditional_acquisition_only"
        and producer_acceptance_payload.get("rows_redecoded_reparsed") == 40
        and producer_acceptance_payload.get("forced_prefix_boundaries_checked") == 40
        and producer_acceptance_payload.get("natural_suffix_eos") == 40
        and producer_acceptance_payload.get("empty_assistant_prefix") is False,
        "repair producer is not root accepted",
    )
    require(producer_acceptance_payload.get("manifest") == binding(manifest_path), "producer acceptance manifest binding")
    require(producer_acceptance_payload.get("correction_receipt") == binding(consumer_policy_correction), "producer acceptance correction binding")
    retained_payload = read_json(retained_route)
    require(retained_payload.get("status") == "retained_no_new_decode" and retained_payload.get("image_id") == RETAINED_IMAGE_ID, "retained route receipt")
    require(correction.get("retained_route") == binding(retained_route), "consumer correction retained-route binding")
    for source in manifest["sources"].values():
        verify_binding(source, "repair manifest source")
    decisions = read_jsonl(mask_decisions)
    require(decisions and all(row.get("schema", "").startswith("training_set_completion.stage03_reviewed_prefix_decision") for row in decisions), "mask decision schema")
    source_manifest = read_json(mask_source_manifest)
    require(source_manifest.get("schema") == "training_set_completion.stage03_mask_preparation_source_manifest.v2", "mask source manifest schema")
    mask_acceptance_payload = read_json(mask_acceptance)
    require(mask_acceptance_payload.get("status") == "lead_accepted_prefix_projection_only", "mask projection is not root accepted")
    require(mask_acceptance_payload.get("rows") == len(decisions) == 195, "mask projection denominator")
    require(mask_acceptance_payload.get("source_manifest_sha256") == file_hash(mask_source_manifest), "mask acceptance source-manifest binding")
    catalog = read_json(target_catalog); targets = _atomic_targets(catalog)
    forced_payload = read_json(forced_acceptance); accepted = _validate_forced_acceptance(forced_payload, manifest)
    selected, selections, row_bindings = _selected_rows(repair_root, manifest)
    require(sorted(correction.get("forced_row_files", []), key=lambda item: item["path"]) == row_bindings, "consumer correction row bindings")
    source_records = _source_record_map(manifest)
    step16_path = Path(manifest["sources"]["step16"]["path"]); step32_path = Path(manifest["sources"]["step32"]["path"])
    step16, step32 = _readback_map(step16_path), _readback_map(step32_path)
    base_model = Path(manifest["model_config"]["model"]["base_model"])
    tokenizer = _load_tokenizer(base_model)
    coordinate_ids = list(manifest["coordinate_token_ids"])
    by_image_step: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in decisions:
        by_image_step[(int(row["image_id"]), int(row["step"]))].append(row)
    routes, all_trace = [], []
    crosscheck_323322 = None
    for image in IMAGE_IDS:
        record = source_records[image]
        if image == RETAINED_IMAGE_ID:
            ids = list(step32[image]["generated_token_ids"])
            require(ids == list(manifest["prefixes"][str(image)]["continuation"]), "210457 retained step32 IDs changed")
            require(retained_payload.get("route", {}).get("generated_token_ids") == ids, "210457 retained receipt IDs changed")
            source_decisions = by_image_step[(image, 32)]
            weights, trace, boxes, frame = _mask_reviewed_prefix(
                ids=ids, tokenizer=tokenizer, record=record, stop_reason=step32[image]["decode_stop_reason"],
                decisions=source_decisions, targets=targets[image], prefix_source_token_count=len(ids) - 1,
                coordinate_ids=coordinate_ids,
            )
            require(set(row["effective_owner_id"] for row in source_decisions) == targets[image], "210457 reviewed target coverage")
            weights[-1] = 1
            trace = [{"image_id": image, **card} for card in trace]
            route = _route_record(
                image=image, ids=ids, weights=weights, boxes=boxes, trace=trace, source_record=record,
                route_id="stage03:image-000000210457:retained-step32-complete",
                eos_positive=True,
                provenance={"kind": "retained_exact_step32_complete_route", "source": binding(step32_path), "source_token_ids_sha256": digest(ids)},
            )
        else:
            raw = selected[image]; ids = list(raw["continuation_token_ids"]); prefix = list(raw["forced_prefix_token_ids"])
            info = manifest["prefixes"][str(image)]; forced = info["forced_owner"]
            source_step = 32 if info["prefix_rule"] == "exact_step32_first_six_raw_rows" else 16
            appended_len = len(forced["appended_row_token_ids"]); source_count = len(prefix) - appended_len
            source_decisions = [row for row in by_image_step[(image, source_step)] if int(row["generated_order"]) < (6 if image == CONDITIONAL_EOS_IMAGE_ID else 10**9)]
            weights, trace, boxes, frame = _mask_reviewed_prefix(
                ids=ids, tokenizer=tokenizer, record=record, stop_reason=raw["decode_stop_reason"],
                decisions=source_decisions, targets=targets[image], prefix_source_token_count=source_count,
                coordinate_ids=coordinate_ids,
            )
            forced_card, forced_box = _forced_row_card(
                ids=ids, parse_frame=frame, forced=forced, prefix_token_count=len(prefix), targets=targets[image],
                coordinate_ids=coordinate_ids, accepted=accepted[image], weights=weights,
            )
            trace.append(forced_card); boxes.append(forced_box)
            excluded = set(frame["seen_orders"]) | {int(forced_card["generated_order"])}
            trace.extend(_suffix_trace(ids=ids, parse_frame=frame, prefix_token_count=len(prefix), excluded_orders=excluded))
            # Generated suffix objects are never positive; only the separately
            # gated terminal EOS for 323322 can be active in the suffix.
            require(not any(weights[len(prefix):]), "generated suffix object received CE")
            eos_positive = False
            if image == CONDITIONAL_EOS_IMAGE_ID:
                eos_positive, known, debt = _allow_323322_eos(
                    suffix=raw["generated_suffix_token_ids"], source_decisions=source_decisions,
                    forced_owner_id=str(forced["owner_id"]), targets=targets[image],
                )
                if eos_positive:
                    weights[-1] = 1
                source32_ids = list(step32[image]["generated_token_ids"])
                _, _, source32_parsed = _native_parse(source32_ids, tokenizer, record, step32[image]["decode_stop_reason"])
                source32_objects = _object_index(source32_parsed)
                sixth_end = max(_interval(*_exact_token_text_frame(source32_ids, tokenizer), source32_objects[index])[-1] for index in range(6)) + 1
                source16_ids = list(step16[image]["generated_token_ids"])
                crosscheck_323322 = {
                    "prefix_rule": info["prefix_rule"], "source_row_count": 6,
                    "manifest_source_prefix_token_count": source_count,
                    "step32_first_six_token_count": sixth_end,
                    "matches_step32_first_six": prefix[:source_count] == source32_ids[:sixth_end] and source_count == sixth_end,
                    "matches_step16_same_length": prefix[:source_count] == source16_ids[:source_count],
                    "step32_first_six_token_ids_sha256": digest(source32_ids[:sixth_end]),
                    "manifest_source_prefix_token_ids_sha256": digest(prefix[:source_count]),
                    "step16_same_length_token_ids_sha256": digest(source16_ids[:source_count]),
                    "seven_known_atomic_owner_ids": sorted(known), "no_prefix_debt": not debt,
                    "selected_suffix_is_eos_only": list(raw["generated_suffix_token_ids"]) == [EOS],
                    "eos_positive": eos_positive,
                }
                require(crosscheck_323322["matches_step32_first_six"], "323322 prefix is not exact step32 first six")
            require(not any(weights[index] for index in range(len(prefix), len(ids) - (1 if eos_positive else 0))), "suffix row/non-EOS CE leakage")
            trace = [{"image_id": image, **card} for card in trace]
            route = _route_record(
                image=image, ids=ids, weights=weights, boxes=boxes, trace=trace, source_record=record,
                route_id=f"stage03:image-{image:012d}:reviewed-repair",
                eos_positive=eos_positive,
                provenance={
                    "kind": "forced_history_single_owner_repair", "selected_raw_row": binding(_row_path(repair_root, raw["request"]["request_id"])),
                    "selection": selections[image], "prefix_rule": info["prefix_rule"],
                    "generation_history_empty_assistant_prefix": False,
                    "source_manifest_empty_assistant_prefix_field_ignored": manifest["runtime"].get("empty_assistant_prefix"),
                    "forced_acceptance": binding(forced_acceptance),
                },
            )
        routes.append(route)
        all_trace.extend({"route_id": route["route_id"], "image_id": image, **card} for card in trace)
    require(crosscheck_323322 is not None, "missing 323322 cross-check")
    runtime_path = output / "runtime-route-list.json"
    trace_path = output / "proposal-trace.jsonl"
    # Trace is written first so the bank can bind the final bytes.
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    require(not trace_path.exists(), "trace exists")
    trace_path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in all_trace))
    require(read_jsonl(trace_path) == all_trace, "trace publication readback")
    publish(runtime_path, routes)
    sources = {
        "repair_manifest": binding(manifest_path), "repair_result": binding(result_path), "repair_rows": row_bindings,
        "mask_decisions": binding(mask_decisions), "mask_source_manifest": binding(mask_source_manifest),
        "mask_projection_acceptance": binding(mask_acceptance),
        "target_catalog_v3": binding(target_catalog), "forced_owner_acceptance": binding(forced_acceptance),
        "consumer_policy_correction": binding(consumer_policy_correction), "retained_210457_route": binding(retained_route),
        "repair_producer_acceptance": binding(producer_acceptance),
        "step16_readback": binding(step16_path), "step32_readback": binding(step32_path),
        "parent16_generator": manifest["parent_adapter"], "producer": binding(Path(__file__)),
    }
    bank = {
        "schema": SCHEMA, "status": "candidate_ready", "sources": sources,
        "policy": {
            "purpose": "bounded one-owner teaching bank; not physical clean-route promotion",
            "selection": "shortest cap-valid natural-terminal-EOS generated suffix; tie lower temperature",
            "object_positive_authority": "exact reviewed source-prefix decisions with atomic-v3 membership plus one root-accepted forced GT row",
            "generated_suffix_rows": "all CE masked",
            "eos": "positive only for retained complete 210457 and conditionally complete 323322; masked otherwise",
            "forced_generation_empty_assistant_prefix": False,
        },
        "target_atomic_count": 228, "crowd_count_separate": catalog.get("crowd_count_separate"),
        "route_count": len(routes), "runtime_route_list": binding(runtime_path), "proposal_trace": binding(trace_path),
        "routes": [{
            "route_id": row["route_id"], "image_id": row["image_id"],
            "continuation_token_ids_sha256": row["continuation_token_ids_sha256"],
            "ce_weights_sha256": row["ce_weights_sha256"],
            "active_ce_tokens": sum(row["ce_weights"]), "trusted_box_count": len(row["trusted_boxes"]),
            "eos_positive": row["trusted_complete_support_endpoint"], "proposal_trace_sha256": row["proposal_trace_sha256"],
        } for row in routes],
        "cross_checkpoint_323322_prefix": crosscheck_323322,
        "superseded_attempts": superseded,
    }
    bank["content_sha256"] = _content_hash(bank)
    publish(output / "bank.json", bank)
    summary = {
        "schema": f"{SCHEMA}.summary.v1", "status": "candidate_ready", "bank": binding(output / "bank.json"),
        "runtime_route_list": binding(runtime_path), "route_count": len(routes),
        "active_ce_tokens": sum(sum(row["ce_weights"]) for row in routes),
        "trusted_boxes": sum(len(row["trusted_boxes"]) for row in routes),
        "eos_positive_image_ids": [row["image_id"] for row in routes if row["trusted_complete_support_endpoint"]],
        "selected_requests": {str(image): selections[image]["selected_request_id"] for image in sorted(selections)},
        "limits": ["Generated suffix object rows are unreviewed context with zero CE.", "The bank is a teaching route set, not a physical clean-route claim."],
    }
    summary["content_sha256"] = _content_hash(summary)
    publish(output / "summary.json", summary)
    validate(output / "bank.json")
    return bank


def validate(bank_path: Path) -> dict[str, Any]:
    bank = read_json(bank_path)
    require(bank.get("schema") == SCHEMA and bank.get("status") == "candidate_ready", "bank schema/status")
    require(bank.get("content_sha256") == _content_hash(bank), "bank content hash")
    for key, source in bank["sources"].items():
        if key in ("repair_rows", "parent16_generator"):
            continue
        verify_binding(source, key)
    for source in bank["sources"]["repair_rows"]:
        verify_binding(source, "repair row")
    _verify_adapter(bank["sources"]["parent16_generator"])
    routes = read_json(Path(bank["runtime_route_list"]["path"])); verify_binding(bank["runtime_route_list"], "runtime route list")
    trace = read_jsonl(Path(bank["proposal_trace"]["path"])); verify_binding(bank["proposal_trace"], "proposal trace")
    require(isinstance(routes, list) and len(routes) == len(IMAGE_IDS) and [row["image_id"] for row in routes] == list(IMAGE_IDS), "runtime route list cohort/order")
    trace_by_route: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for card in trace:
        trace_by_route[card["route_id"]].append(card)
    for route in routes:
        ids, weights = route["continuation_token_ids"], route["ce_weights"]
        require(route["continuation_token_ids_sha256"] == digest(ids) and route["ce_weights_sha256"] == digest(weights), "route literal/mask hash")
        require(len(ids) == len(weights) and all(weight in (0, 1) for weight in weights), "route mask domain")
        stored_trace = [{key: value for key, value in card.items() if key != "route_id"} for card in trace_by_route[route["route_id"]]]
        require(route["proposal_trace_sha256"] == digest(stored_trace), "route trace hash")
        eos = [index for index, token in enumerate(ids) if token == EOS and weights[index]]
        require(eos == ([len(ids) - 1] if route["trusted_complete_support_endpoint"] else []), "route EOS validation")
        for box in route["trusted_boxes"]:
            positions = [box[key] for key in ("x1_position", "y1_position", "x2_position", "y2_position")]
            require(all(weights[position] == 1 for position in positions), "trusted box mask validation")
    require(bank["cross_checkpoint_323322_prefix"]["matches_step32_first_six"] is True, "323322 prefix validation")
    require([row["image_id"] for row in routes if row["trusted_complete_support_endpoint"]] in ([210457], [210457, 323322]), "EOS-positive scope")
    return {"status": "candidate_valid", "route_count": len(routes), "trace_count": len(trace), "bank_sha256": file_hash(bank_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--repair-root", type=Path, required=True)
    build_parser.add_argument("--mask-decisions", type=Path, required=True)
    build_parser.add_argument("--mask-source-manifest", type=Path, required=True)
    build_parser.add_argument("--mask-acceptance", type=Path, required=True)
    build_parser.add_argument("--target-catalog", type=Path, required=True)
    build_parser.add_argument("--forced-acceptance", type=Path, required=True)
    build_parser.add_argument("--consumer-policy-correction", type=Path, required=True)
    build_parser.add_argument("--producer-acceptance", type=Path, required=True)
    build_parser.add_argument("--retained-route", type=Path, required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("--bank", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        value = build(
            repair_root=args.repair_root, mask_decisions=args.mask_decisions,
            mask_source_manifest=args.mask_source_manifest, target_catalog=args.target_catalog,
            mask_acceptance=args.mask_acceptance, forced_acceptance=args.forced_acceptance,
            consumer_policy_correction=args.consumer_policy_correction, producer_acceptance=args.producer_acceptance,
            retained_route=args.retained_route,
            output=args.output,
        )
        print(json.dumps({"schema": value["schema"], "status": value["status"], "routes": value["route_count"]}))
    else:
        print(json.dumps(validate(args.bank), sort_keys=True))


if __name__ == "__main__":
    main()
