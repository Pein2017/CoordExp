"""CPU-only masked coherent-route bank from reviewed literal acquisition rows.

The review extraction is the only truth authority.  This module preserves every
raw token and assigns CE positions mechanically; it never constructs a repair
row, deletes a bad row, or infers a physical owner from geometry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from probes.dora_owner_learning.geometric_dedup import _character_span_to_token_interval, _exact_token_text_frame
from probes.training_set_completion.training import coordinate_token_table

SCHEMA = "training_set_completion.first_masked_route_bank.v2"
RUNTIME_ROUTES_SCHEMA = "training_set_completion.first_masked_route_bank.runtime_routes.v1"
IMAGE_IDS = (25274, 59571, 99937, 210457, 219546, 323322, 351017, 388795, 417044, 477415, 528944)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def binding(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": file_hash(path), "size_bytes": path.stat().st_size}


def _verify_binding(value: Mapping[str, Any], label: str) -> None:
    require(set(value) == {"path", "sha256", "size_bytes"} and binding(value["path"]) == dict(value), f"{label} bytes changed")


def publish(path: Path, value: Any) -> None:
    require(not path.exists(), f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value))
    require(path.read_bytes() == _canonical(value), "bank publication readback")


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _load_tokenizer(base_model: Path) -> Any:
    from src.qwen.runtime_loading import QwenLoadOptions, load_qwen_components_from_options
    return load_qwen_components_from_options(QwenLoadOptions(base_model=str(base_model), dtype="fp32", attn_implementation="sdpa", load_model=False)).tokenizer


def _owner(row: Mapping[str, Any]) -> str | None:
    value = row.get("root_owner_id") or row.get("reviewed_owner_id")
    return str(value) if isinstance(value, str) and value else None


def target_catalog(registry: Mapping[str, Any]) -> tuple[dict[int, set[str]], dict[str, int]]:
    require(registry.get("computed_counts", {}).get("atomic_candidate_total") == 220, "expected 220 atomic targets")
    require(len(registry.get("gt_atomic", [])) == 169 and len(registry.get("gt_crowd", [])) == 6, "GT/crowd denominator")
    require(len(registry.get("admitted_new_owner_registry", [])) == 50, "new-owner denominator")
    prior = [row for row in registry.get("prior_reviewed_supports", []) if row.get("support_role") != "gt_atomic"]
    require(len(prior) == 1, "expected exactly one prior non-GT atomic target")
    by_image: dict[int, set[str]] = defaultdict(set)
    for row in registry["gt_atomic"]:
        by_image[int(row["image_id"])].add(str(row["owner_id"]))
    for row in registry["admitted_new_owner_registry"]:
        require(row.get("admitted_target_candidate") is True and row.get("root_held") is False, "new target is not admitted")
        declaration = row.get("declarations", [{}])[0]
        by_image[int(declaration["image_id"])].add(str(row["owner_id"]))
    by_image[int(prior[0]["image_id"])].add(str(prior[0]["owner_id"]))
    flattened = [owner for owners in by_image.values() for owner in owners]
    require(len(flattened) == len(set(flattened)) == 220 and set(by_image) <= set(IMAGE_IDS), "atomic target identity/denominator")
    return by_image, {"atomic": 220, "gt_atomic": 169, "prior_non_gt": 1, "new": 50, "crowd_separate": 6}


def _root_reference_boxes(source_manifest: Mapping[str, Any]) -> dict[str, list[int]]:
    """Read only explicit root reference-box declarations, never a visual hint."""
    result: dict[str, list[int]] = {}
    for item in source_manifest.get("packets_and_reviews", []):
        reference = item.get("root_override")
        if not isinstance(reference, Mapping) or not reference.get("path"):
            continue
        payload = json.loads(Path(reference["path"]).read_text())
        declared = payload.get("root_reviewed_reference_boxes", payload.get("reviewed_reference_boxes", []))
        if isinstance(declared, Mapping):
            declared = [{"owner_id": owner, **box} for owner, box in declared.items() if isinstance(box, Mapping)]
        for box in declared if isinstance(declared, list) else []:
            owner, bins = box.get("owner_id"), box.get("coord_bins_1000", box.get("bbox_coord_bins_1000"))
            require(isinstance(owner, str) and isinstance(bins, list) and len(bins) == 4 and all(type(value) is int and 0 <= value <= 1000 for value in bins), "invalid root reviewed reference box")
            require(owner not in result, "duplicate root reviewed reference box")
            result[owner] = list(bins)
    return result


def project_target_owners(registry: Mapping[str, Any], proposal_rows: Sequence[Mapping[str, Any]], *, root_references: Mapping[str, Sequence[int]] = {}) -> dict[str, Any]:
    """Keep all 220 owners, including any reference-box gap as an explicit hold."""
    by_owner: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in proposal_rows:
        owner = _owner(row)
        if owner:
            by_owner[owner].append(row)
    records = []
    for row in registry["gt_atomic"]:
        records.append({"image_id": int(row["image_id"]), "owner_id": str(row["owner_id"]), "role": "gt_atomic", "reference_status": "qualified_gt", "reference_coord_bins_1000": list(row["bbox_coord_bins_1000"]), "category": row.get("category_name"), "physical_match": "separate_owner_metric"})
    for row in registry["prior_reviewed_supports"]:
        if row.get("support_role") != "gt_atomic":
            records.append({"image_id": int(row["image_id"]), "owner_id": str(row["owner_id"]), "role": "prior_non_gt", "reference_status": "qualified_prior", "reference_coord_bins_1000": list(row["bbox"]["coord_bins_1000"]), "category": row.get("description"), "physical_match": "separate_owner_metric"})
    for row in registry["admitted_new_owner_registry"]:
        owner = str(row["owner_id"]); declaration = row["declarations"][0]
        candidates = [item for item in by_owner.get(owner, []) if item.get("reviewed") and item.get("physical_status") == "true_unique" and item.get("parser_status") == "parsed_valid" and item.get("effective_extent") == "reasonable" and not item.get("raw_geometry_invalid", False)]
        candidates.sort(key=lambda item: str(item["proposal_id"]))
        if owner in root_references:
            reference = {"reference_status": "qualified_root_reviewed_reference", "reference_coord_bins_1000": list(root_references[owner]), "reference_proposal_id": None}
        elif candidates:
            reference = {"reference_status": "qualified_reviewed_member", "reference_coord_bins_1000": list(candidates[0]["coord_bins_1000"]), "reference_proposal_id": candidates[0]["proposal_id"]}
        else:
            reference = {"reference_status": "unresolved_reference_hold", "reference_coord_bins_1000": None, "reference_proposal_id": None}
        records.append({"image_id": int(declaration["image_id"]), "owner_id": owner, "role": "new", "category": None if "unknown" in row.get("classes", []) else None, "physical_match": "separate_owner_metric", **reference})
    require(len(records) == 220 and len({row["owner_id"] for row in records}) == 220, "target-owner projection denominator")
    return {"schema": f"{SCHEMA}.target_owners.v1", "status": "candidate_ready", "atomic_target_count": 220, "crowd_count_separate": 6, "records": sorted(records, key=lambda row: (row["image_id"], row["owner_id"]))}


def _valid_for_coverage(row: Mapping[str, Any], targets: set[str]) -> bool:
    return (_owner(row) in targets and row.get("physical_status") in ("true_unique", "repeat")
            and row.get("parser_status") == "parsed_valid" and not row.get("raw_geometry_invalid", False)
            and row.get("effective_extent") == "reasonable")


def _route_score(rows: Sequence[Mapping[str, Any]], targets: set[str]) -> tuple[int, int, int, float]:
    covered = {_owner(row) for row in rows if _valid_for_coverage(row, targets)}
    bad = sum(row.get("physical_status") in ("false", "repeat", "invalid") or row.get("effective_extent") == "wrong" or row.get("parser_status") != "parsed_valid" for row in rows)
    unknown = sum(row.get("physical_status") == "unknown" for row in rows)
    return (-len(covered), bad, unknown, float(rows[0]["temperature"]))


def choose_route(rows: Sequence[Mapping[str, Any]], targets: set[str]) -> tuple[str, dict[str, Any]]:
    by_request: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_request[str(row["request_id"])].append(row)
    require(len(by_request) == 4, "exactly four acquisition policies/image")
    request, route = min(by_request.items(), key=lambda item: _route_score(item[1], targets))
    score = _route_score(route, targets)
    return request, {"request_id": request, "distinct_target_owners": -score[0], "confirmed_bad_raw_rows": score[1], "unknown_raw_rows": score[2], "temperature": score[3], "criteria": "maximize valid/reasonable trusted target coverage; tie fewer false/repeat/invalid/wrong-extent rows, then fewer unknowns, then lower temperature"}


def _interval(text: str, token_spans: Sequence[tuple[int, int]], start: int, end: int) -> list[int]:
    left, right = _character_span_to_token_interval(start, end, text=text, token_spans=token_spans)
    return list(range(left, right))


def _mask_fields(*, acquisition_pred: Mapping[str, Any], text: str, token_spans: Sequence[tuple[int, int]], review: Mapping[str, Any], targets: set[str], coordinate_ids: Sequence[int]) -> tuple[set[int], dict[str, Any]]:
    """Return positions to supervise, retaining all other raw tokens in history."""
    object_positions = _interval(text, token_spans, int(acquisition_pred["char_start"]), int(acquisition_pred["char_end"]))
    schema_positions: list[int] = []
    for span in [*acquisition_pred.get("schema_spans", []), *acquisition_pred.get("coord_token_spans", [])]:
        schema_positions.extend(_interval(text, token_spans, int(span["char_start"]), int(span["char_end"])))
    description = acquisition_pred.get("description_predicted", acquisition_pred.get("description"))
    require(isinstance(description, str), "raw object lacks literal description")
    schema = acquisition_pred.get("schema_spans", [])
    require(len(schema) >= 2, "raw object lacks description delimiters")
    description_positions = _interval(text, token_spans, int(schema[0]["char_end"]), int(schema[1]["char_start"]))
    expected_bins = list(review.get("coord_bins_1000") or acquisition_pred.get("coord_bins") or [])
    coord_positions: list[int] = []
    for span in acquisition_pred.get("coord_token_spans", []):
        coord_positions.extend(_interval(text, token_spans, int(span["char_start"]), int(span["char_end"])))
    require(len(expected_bins) == len(coord_positions) == 4 and [review.get("coord_bins_1000", acquisition_pred.get("coord_bins"))[index] for index in range(4)] == expected_bins, "raw coordinate count")
    require([int(acquisition_pred["_token_ids"][position]) for position in coord_positions] == [coordinate_ids[value] for value in expected_bins], "trusted coordinate positions differ from raw tokens")
    base = (_owner(review) in targets and review.get("physical_status") == "true_unique" and not review.get("root_repeat_after_alias", False)
            and review.get("parser_status") == "parsed_valid" and not review.get("raw_geometry_invalid", False)
            and review.get("effective_extent") == "reasonable" and bool(review.get("reviewed")))
    effective_class = review.get("effective_class", review.get("class"))
    require(effective_class in ("verified", "unknown", "wrong"), "effective class is not a reviewed value")
    decision = review.get("effective_direct_CE")
    geometry = description_ok = False
    if base and isinstance(decision, str):
        geometry = decision in ("positive", "correct")
        description_ok = decision == "positive" and effective_class == "verified"
    elif base and isinstance(decision, Mapping):
        geometry = decision.get("bbox") == "positive"
        description_ok = geometry and decision.get("description") == "positive" and effective_class == "verified"
    positions = set(schema_positions if geometry else [])
    if description_ok:
        positions.update(description_positions)
    return positions, {"proposal_id": review["proposal_id"], "owner_id": _owner(review), "row_token_positions": object_positions, "schema_positions": schema_positions, "description_positions": description_positions, "coordinate_positions": coord_positions, "expected_bins": expected_bins, "positive_positions": sorted(positions), "reason": "positive_fields" if positions else "masked_by_review_or_nonpositive_field", "review_authority": bool(review.get("reviewed")), "description_predicted": description, "effective_class": effective_class}


def _effective_class_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Require every extracted review row to carry its final class decision."""
    overrides = []
    for row in rows:
        effective_class = row.get("effective_class")
        require(effective_class in ("verified", "unknown", "wrong"), "missing or invalid effective class")
        if effective_class != row.get("class"):
            overrides.append({"proposal_id": str(row["proposal_id"]), "class": row.get("class"), "effective_class": effective_class})
    return {"reviewed_row_count": len(rows), "effective_class_row_count": len(rows), "override_count": len(overrides), "overrides": sorted(overrides, key=lambda row: row["proposal_id"])}


def project_runtime_routes(bank: Mapping[str, Any], *, bank_path: Path, selected_raw: Mapping[str, Mapping[str, Any]], acquisition_records: Mapping[int, Mapping[str, Any]], acquisition_manifest_path: Path) -> dict[str, Any]:
    """Project literal continuations and reviewed coordinate supervision for training."""
    routes = []
    for route in bank["routes"]:
        raw, source_record = selected_raw[route["route_id"]], acquisition_records[route["image_id"]]
        case = source_record["case"]
        plan = case["image_plan"]
        require(raw["example_id"] == route["example_id"] == source_record["example_id"] and raw["prompt_token_ids"] == source_record["prompt_token_ids"], "selected request differs from bound acquisition record")
        require(raw["executed_media_sha256"] == plan["executed_media_sha256"] and raw["observed_image_grid_thw"] == plan["observed_image_grid_thw"], "selected request differs from bound image")
        weights = route["ce_weights"]
        boxes = []
        for card in route["proposal_trace"]:
            if not card["positive_positions"]:
                continue
            positions, bins = card["coordinate_positions"], card["expected_bins"]
            require(len(positions) == len(bins) == 4 and all(weights[position] == 1 for position in positions), "positive trace lacks active trusted coordinates")
            boxes.append({"x1_position": positions[0], "y1_position": positions[1], "x2_position": positions[2], "y2_position": positions[3], "expected_bins": bins})
        routes.append({"route_id": route["route_id"], "image_id": route["image_id"], "example_id": route["example_id"], "case": case, "image_identity": {"image_path": case["image_path"], "image_content_sha256": plan["image_content_sha256"], "executed_media_sha256": plan["executed_media_sha256"], "observed_image_grid_thw": plan["observed_image_grid_thw"]}, "prompt_token_ids": raw["prompt_token_ids"], "prompt_token_ids_sha256": raw["prompt_token_ids_sha256"], "continuation_token_ids": route["generated_token_ids"], "continuation_token_ids_sha256": route["generated_token_ids_sha256"], "ce_weights": weights, "ce_weights_sha256": route["ce_weights_sha256"], "trusted_boxes": boxes, "trusted_complete_support_endpoint": route["eos_positive"], "provenance": {"bank": binding(bank_path), "acquisition_manifest": binding(acquisition_manifest_path), "selected_request": route["selected_route"], "selected_raw_row": {"request_id": raw["request"]["request_id"], "generated_token_ids_sha256": raw["generated_token_ids_sha256"], "prompt_token_ids_sha256": raw["prompt_token_ids_sha256"], "executed_media_sha256": raw["executed_media_sha256"], "observed_image_grid_thw": raw["observed_image_grid_thw"]}, "proposal_trace_sha256": digest(route["proposal_trace"])}})
    value = {"schema": RUNTIME_ROUTES_SCHEMA, "status": "candidate_ready", "bank": binding(bank_path), "routes": routes}
    value["content_sha256"] = digest(value)
    return value


def build(*, acquisition_rows: Path, extraction_root: Path, output: Path, base_model: Path) -> dict[str, Any]:
    """Build one immutable bank only after a canonical extraction directory is supplied."""
    source_manifest = json.loads((extraction_root / "source-manifest.json").read_text())
    receipt = json.loads((extraction_root / "extraction-receipt.json").read_text())
    require(receipt.get("schema") == "training_set_completion.stage01_review_extraction.v2", "requires canonical review-extraction v2")
    registry_path = extraction_root / "target-candidate-registry.json"
    registry = json.loads(registry_path.read_text())
    targets, counts = target_catalog(registry)
    acquisitions = _rows(acquisition_rows)
    acquisition_manifest_path = acquisition_rows.parent / "manifest.json"
    acquisition_manifest = json.loads(acquisition_manifest_path.read_text())
    require(all(row.get("manifest_sha256") == acquisition_manifest.get("content_sha256") for row in acquisitions), "acquisition rows do not bind their manifest")
    acquisition_records = {int(row["image_id"]): row for row in acquisition_manifest.get("records", [])}
    require(set(acquisition_records) == set(IMAGE_IDS), "acquisition manifest record denominator")
    _verify_binding(source_manifest["acquisition"], "bound acquisition rows")
    require(binding(acquisition_rows) == source_manifest["acquisition"], "requested acquisition differs from extraction")
    for item in source_manifest.get("packets_and_reviews", []):
        _verify_binding(item["packet"], "review packet")
        _verify_binding(item["review"], "review")
        if item.get("root_override") is not None:
            _verify_binding(item["root_override"], "root override")
    by_request = {str(row["request"]["request_id"]): row for row in acquisitions}
    require(len(by_request) == 44, "acquisition request denominator")
    coordinate_ids = coordinate_token_table(base_model)["ids"]
    tokenizer = _load_tokenizer(base_model)
    route_rows: dict[int, list[dict[str, Any]]] = {}
    source_rows: list[dict[str, Any]] = []
    for image_id in IMAGE_IDS:
        path = extraction_root / "proposal-rows" / f"image-{image_id:012d}.jsonl"
        rows = _rows(path); require(rows and all(int(row["image_id"]) == image_id for row in rows), "per-image extraction row identity")
        source_rows.append(binding(path)); route_rows[image_id] = rows
    effective_class_audit = _effective_class_audit([row for rows in route_rows.values() for row in rows])
    target_owner_path = output.parent / "target-owners.json"
    target_owner_projection = project_target_owners(registry, [row for rows in route_rows.values() for row in rows], root_references=_root_reference_boxes(source_manifest))
    publish(target_owner_path, target_owner_projection)
    routes = []; selected_raw: dict[str, Mapping[str, Any]] = {}
    for image_id in IMAGE_IDS:
        selected_request, selection = choose_route(route_rows[image_id], targets.get(image_id, set()))
        raw = by_request[selected_request]
        selected_raw[selected_request] = raw
        ids = list(raw["generated_token_ids"]); text, spans = _exact_token_text_frame(ids, tokenizer)
        require(text == raw["raw_decode_text"], "raw decode differs from exact original token frame")
        reviews = {str(row["proposal_id"]): row for row in route_rows[image_id] if str(row["request_id"]) == selected_request}
        weights = [0] * len(ids); trace = []; covered = set()
        for pred in raw["parsed"]["pred"]:
            proposal_id = f"{selected_request}:p{pred['generated_order']}"; require(proposal_id in reviews, "parsed proposal missing extraction review")
            pred = dict(pred); pred["_token_ids"] = ids
            positions, card = _mask_fields(acquisition_pred=pred, text=text, token_spans=spans, review=reviews[proposal_id], targets=targets.get(image_id, set()), coordinate_ids=coordinate_ids)
            for pos in positions: weights[pos] = 1
            if _valid_for_coverage(reviews[proposal_id], targets.get(image_id, set())): covered.add(_owner(reviews[proposal_id]))
            trace.append(card)
        debt = any(row.get("physical_status") in ("unknown", "false", "repeat", "invalid") or row.get("effective_extent") == "wrong" or row.get("parser_status") != "parsed_valid" for row in reviews.values())
        eos = ids[-1] == 151645 and covered == targets.get(image_id, set()) and not debt
        if eos: weights[-1] = 1
        routes.append({"route_id": selected_request, "image_id": image_id, "example_id": raw["example_id"], "generated_token_ids": ids, "generated_token_ids_sha256": digest(ids), "ce_weights": weights, "ce_weights_sha256": digest(weights), "proposal_trace": trace, "covered_target_owner_ids": sorted(covered), "target_owner_ids": sorted(targets.get(image_id, set())), "eos_positive": eos, "selected_route": selection})
    bank = {"schema": SCHEMA, "status": "candidate_ready", "sources": {"acquisition_rows": binding(acquisition_rows), "acquisition_manifest": binding(acquisition_manifest_path), "extraction_registry": binding(registry_path), "extraction_receipt": binding(extraction_root / "extraction-receipt.json"), "extraction_source_manifest": binding(extraction_root / "source-manifest.json"), "proposal_rows": source_rows, "producer": binding(Path(__file__))}, "target_owners": binding(target_owner_path), "target_counts": counts, "crowds_separate": len(registry["gt_crowd"]), "effective_class_audit": effective_class_audit, "routes": routes}
    bank["content_sha256"] = digest(bank)
    validate_bank(bank)
    publish(output, bank)
    runtime_routes = project_runtime_routes(bank, bank_path=output, selected_raw=selected_raw, acquisition_records=acquisition_records, acquisition_manifest_path=acquisition_manifest_path)
    validate_runtime_routes(runtime_routes, bank=bank)
    publish(output.parent / "runtime-routes.json", runtime_routes)
    return bank


def validate_bank(bank: Mapping[str, Any]) -> None:
    require(bank.get("schema") == SCHEMA and bank.get("content_sha256") == digest({k: v for k, v in bank.items() if k != "content_sha256"}), "bank schema/content")
    sources = bank.get("sources", {})
    require(set(sources) == {"acquisition_rows", "acquisition_manifest", "extraction_registry", "extraction_receipt", "extraction_source_manifest", "proposal_rows", "producer"}, "bank source identity fields")
    for label in ("acquisition_rows", "acquisition_manifest", "extraction_registry", "extraction_receipt", "extraction_source_manifest", "producer"):
        _verify_binding(sources[label], label)
    require(isinstance(sources["proposal_rows"], list) and len(sources["proposal_rows"]) == len(IMAGE_IDS), "proposal source bindings")
    for source in sources["proposal_rows"]:
        _verify_binding(source, "proposal rows")
    _verify_binding(bank["target_owners"], "target-owner projection")
    require(bank.get("target_counts") == {"atomic": 220, "gt_atomic": 169, "prior_non_gt": 1, "new": 50, "crowd_separate": 6} and bank.get("crowds_separate") == 6, "target counts")
    routes = bank.get("routes", []); require([row["image_id"] for row in routes] == list(IMAGE_IDS), "all 11 routes")
    audit = bank.get("effective_class_audit", {})
    require(audit.get("reviewed_row_count") == audit.get("effective_class_row_count") == 932 and audit.get("override_count") == len(audit.get("overrides", [])), "effective class audit")
    for route in routes:
        ids, weights = route["generated_token_ids"], route["ce_weights"]
        require(len(ids) == len(weights) and all(type(item) is int and item in (0, 1) for item in weights), "literal token/mask identity")
        require(route.get("generated_token_ids_sha256") == digest(ids) and route.get("ce_weights_sha256") == digest(weights), "literal token/mask hashes")
        for card in route["proposal_trace"]:
            require(card.get("effective_class") in ("verified", "unknown", "wrong"), "trace effective class")
            if card["positive_positions"]:
                require(card["review_authority"], "positive CE lacks review authority")
            require(all(weights[pos] == 1 for pos in card["positive_positions"]), "trace/mask mismatch")
            if card["effective_class"] != "verified":
                require(all(weights[pos] == 0 for pos in card["description_positions"]), "non-verified effective class has description CE")
        require(not route["eos_positive"] or weights[-1] == 1, "EOS trace/mask mismatch")
        require(route["eos_positive"] or weights[-1] == 0, "EOS CE on incomplete/debt route")


def validate_runtime_routes(value: Mapping[str, Any], *, bank: Mapping[str, Any]) -> None:
    require(value.get("schema") == RUNTIME_ROUTES_SCHEMA and value.get("content_sha256") == digest({key: item for key, item in value.items() if key != "content_sha256"}), "runtime route schema/content")
    require(value.get("bank", {}).get("sha256") == file_hash(value["bank"]["path"]), "runtime route bank bytes changed")
    source_by_id = {row["route_id"]: row for row in bank["routes"]}
    require([row["image_id"] for row in value.get("routes", [])] == list(IMAGE_IDS), "runtime all 11 routes")
    for route in value["routes"]:
        source = source_by_id.get(route["route_id"])
        require(source is not None and route["continuation_token_ids"] == source["generated_token_ids"] and route["ce_weights"] == source["ce_weights"], "runtime literal route differs from bank")
        from probes.training_set_completion.training import validate_route
        validate_route(route, eos_token_id=151645)
        require(route["provenance"].get("bank") == value["bank"] and route["provenance"].get("selected_request") == source["selected_route"], "runtime route provenance differs from bank")
        require(route["continuation_token_ids_sha256"] == digest(route["continuation_token_ids"]) and route["ce_weights_sha256"] == digest(route["ce_weights"]), "runtime token/mask hashes")
        expected_boxes = [{"x1_position": card["coordinate_positions"][0], "y1_position": card["coordinate_positions"][1], "x2_position": card["coordinate_positions"][2], "y2_position": card["coordinate_positions"][3], "expected_bins": card["expected_bins"]} for card in source["proposal_trace"] if card["positive_positions"]]
        require(route["trusted_boxes"] == expected_boxes and route["trusted_complete_support_endpoint"] == source["eos_positive"], "runtime trusted supervision differs from bank")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition-rows", type=Path, required=True)
    parser.add_argument("--extraction-root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(acquisition_rows=args.acquisition_rows, extraction_root=args.extraction_root, base_model=args.base_model, output=args.output)


if __name__ == "__main__":
    main()
