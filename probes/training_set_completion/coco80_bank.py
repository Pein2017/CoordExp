"""Build the immutable COCO-80 teacher for the dual-start comparison.

This deliberately re-materializes every retained object row.  The 232-row
teacher is evidence only: its continuation suffixes, KV cache, logits and
probabilities are not inputs to this bank.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from probes.training_set_completion.complete_bank import EOS, _field_ids
from probes.training_set_completion.training import coordinate_token_table, validate_route
from src.eval.detection_categories import COCO_80_CLASS_NAMES


ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-14-training-set-completion-curriculum"
)
PREPARATION = ROOT / "dual-start-coco80-preparation-v1/preparation.json"
SOURCE_BANK = ROOT / "third-complete-bank-preparation-v2/bank.json"
OUTPUT = ROOT / "dual-start-coco80-teacher-v1"

SCHEMA = "training_set_completion.dual_start_coco80_teacher.v1"
ADMISSION_SCHEMA = SCHEMA + ".cpu_admission"
TARGET_VERSION = "coco80-source232-trusted-description-subset-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "sha256": file_sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def publish(path: Path, value: Any) -> None:
    require(not path.exists() and not path.is_symlink(), f"publication collision: {path}")
    data = canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    require(path.read_bytes() == data, f"publication readback: {path}")


def _checked_binding(value: Any, *, name: str) -> Path:
    require(isinstance(value, Mapping), f"{name} binding")
    path = Path(str(value.get("path", ""))).resolve(strict=True)
    require(binding(path) == dict(value), f"{name} bytes changed")
    return path


def _owner_key(entry: Mapping[str, Any]) -> tuple[int, str]:
    image_id, owner_id = entry.get("image_id"), entry.get("owner_id")
    require(type(image_id) is int and isinstance(owner_id, str) and owner_id, "owner identity")
    return image_id, owner_id


def _require_coco80_description(value: Any, *, owner_id: str) -> str:
    """Fail closed against the exact prompt-bound COCO-80 category registry."""

    require(isinstance(value, str) and value, f"included COCO description: {owner_id}")
    require(
        value in COCO_80_CLASS_NAMES,
        f"included description outside literal COCO-80: {owner_id}: {value!r}",
    )
    return value


def _load_preparation(path: Path) -> dict[str, Any]:
    preparation = read(path)
    require(
        preparation.get("schema") == "training_set_completion.dual_start_preparation.v1",
        "unexpected preparation schema",
    )
    require(
        preparation.get("status") == "prepared_inputs_not_training_admission",
        "preparation is not a CPU-only input inventory",
    )
    require(preparation.get("target_version") == TARGET_VERSION, "target version")
    owners = preparation.get("owners")
    require(isinstance(owners, list) and len(owners) == 232, "source232 ledger")
    keys = [_owner_key(item) for item in owners if isinstance(item, Mapping)]
    require(len(keys) == len(owners) == len(set(keys)), "source232 duplicate owner")
    decisions = defaultdict(int)
    for entry in owners:
        decisions[str(entry.get("decision"))] += 1
        require(
            entry.get("decision")
            in {"included", "confirmed_out_of_scope", "pending_category_or_scope"},
            "unsupported source232 decision",
        )
    require(
        dict(decisions)
        == {
            "included": 218,
            "confirmed_out_of_scope": 1,
            "pending_category_or_scope": 13,
        },
        "source232 partition",
    )
    require(preparation.get("source_population") == 232, "source232 population")
    require(preparation.get("physical_history_population") == 248, "current-known248 population")
    return preparation


def _selected_by_image(preparation: Mapping[str, Any]) -> dict[int, list[dict[str, Any]]]:
    selected: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for original in preparation["owners"]:
        if original["decision"] != "included":
            continue
        entry = copy.deepcopy(dict(original))
        require(entry.get("source_description_ce_positive") is True, "included description is not trusted")
        _require_coco80_description(entry.get("source_description"), owner_id=str(entry["owner_id"]))
        bins = entry.get("reference_bins")
        require(
            isinstance(bins, list)
            and len(bins) == 4
            and all(type(value) is int for value in bins)
            and bins[0] < bins[2]
            and bins[1] < bins[3],
            "included reference geometry",
        )
        selected[int(entry["image_id"])].append(entry)
    for image_id, entries in selected.items():
        entries.sort(key=lambda entry: int(entry["source_order"]))
        orders = [int(entry["source_order"]) for entry in entries]
        require(len(orders) == len(set(orders)), f"duplicate source order: {image_id}")
    require(len(selected) == 11 and sum(map(len, selected.values())) == 218, "included218/11")
    return dict(selected)


def _source_trace_by_owner(route: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    provenance = route.get("provenance")
    require(isinstance(provenance, Mapping), "source route provenance")
    trace = provenance.get("trace")
    require(isinstance(trace, list), "source route trace")
    result = {str(card.get("owner_id")): dict(card) for card in trace if isinstance(card, Mapping)}
    require(len(result) == len(trace), "source route duplicate/malformed owner")
    return result


def reassemble_route(
    source_route: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any,
    coordinate_ids: Sequence[int],
) -> dict[str, Any]:
    """Build one retained-only route and prove each row came from its own source card."""

    image_id = source_route.get("image_id")
    require(type(image_id) is int, "source route image")
    source_trace = _source_trace_by_owner(source_route)
    source_order = [str(card["owner_id"]) for card in source_route["provenance"]["trace"]]
    source_position = {owner: index for index, owner in enumerate(source_order)}
    selected_owners = [str(entry.get("owner_id")) for entry in selected]
    require(len(selected_owners) == len(set(selected_owners)), "duplicate selected owner")
    require(set(selected_owners) <= set(source_trace), "selected owner absent from source route")
    require(
        [source_position[owner] for owner in selected_owners]
        == sorted(source_position[owner] for owner in selected_owners),
        "retained owner order differs from source route",
    )
    require(
        [int(entry.get("source_order", -1)) for entry in selected]
        == sorted(int(entry.get("source_order", -1)) for entry in selected),
        "selected ledger order differs",
    )

    continuation: list[int] = []
    weights: list[int] = []
    boxes: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    for order, entry in enumerate(selected):
        owner = str(entry["owner_id"])
        source_card = source_trace[owner]
        source_fields = source_card.get("edited_fields")
        require(isinstance(source_fields, Mapping), f"source fields: {owner}")
        description = entry["source_description"]
        bins = list(entry["reference_bins"])
        require(
            source_fields.get("selected_description") == description,
            f"cross-object description: {owner}",
        )
        require(
            source_fields.get("catalog_reference_coord_bins_1000") == bins,
            f"cross-object geometry: {owner}",
        )
        require(entry.get("source_description_ce_positive") is True, f"untrusted description: {owner}")
        description_ids = tokenizer.encode(description, add_special_tokens=False)
        require(
            description_ids
            and tokenizer.decode(description_ids, skip_special_tokens=False) == description,
            f"literal tokenization: {owner}",
        )
        start = len(continuation)
        continuation.extend(_field_ids(tokenizer, description, bins, list(coordinate_ids)))
        weights.extend([1] * (len(description_ids) + 8))
        positions = list(range(start + 1 + len(description_ids) + 2, start + 1 + len(description_ids) + 6))
        boxes.append(
            {
                "x1_position": positions[0],
                "y1_position": positions[1],
                "x2_position": positions[2],
                "y2_position": positions[3],
                "expected_bins": bins,
            }
        )
        card = copy.deepcopy(source_card)
        card["order"] = order
        card["source_order"] = int(entry["source_order"])
        card["source_trace"] = copy.deepcopy(source_card)
        card["edited_fields"] = {
            "description_source": "coco80_trusted_source_description",
            "selected_description": description,
            "description_ce_positive": True,
            "description_source_proposal_id": source_fields.get("description_source_proposal_id"),
            "description_authority": "preparation.coco80_source232_trusted",
            "description_token_positions": list(range(start + 1, start + 1 + len(description_ids))),
            "catalog_reference_coord_bins_1000": bins,
            "geometry_replaced": source_fields.get("geometry_replaced"),
        }
        trace.append(card)
    continuation.append(EOS)
    weights.append(1)
    case = copy.deepcopy(source_route["case"])
    plan = case["image_plan"]
    route = {
        "route_id": f"dual-start-coco80:image-{image_id:012d}",
        "image_id": image_id,
        "example_id": source_route["example_id"],
        "case": case,
        "image_identity": {
            "image_path": case["image_path"],
            "image_content_sha256": plan["image_content_sha256"],
            "executed_media_sha256": plan["executed_media_sha256"],
            "observed_image_grid_thw": plan["observed_image_grid_thw"],
        },
        "prompt_token_ids": copy.deepcopy(source_route["prompt_token_ids"]),
        "continuation_token_ids": continuation,
        "ce_weights": weights,
        "trusted_boxes": boxes,
        "trusted_complete_support_endpoint": True,
        "provenance": {
            "synthetic_teacher": True,
            "source_bank_route_id": source_route["route_id"],
            "source_owner_ids": source_order,
            "selected_owner_ids": selected_owners,
            "trace": trace,
            "legacy_suffix_state": "invalidated_by_row_removal_reassemble_all_conditionals",
        },
    }
    validate_route(route, eos_token_id=EOS, coordinate_token_ids=coordinate_ids)
    return route


def _parser_roundtrip(
    routes: Sequence[Mapping[str, Any]], *, acquisition_path: Path, tokenizer: Any
) -> dict[str, Any]:
    """Use the production compact parser on the newly materialized rows."""

    from probes.source_rweak_row_cross.run import native_record

    acquisition = read(acquisition_path)
    goldens = {int(record["image_id"]): record["golden"] for record in acquisition["records"]}
    parsed_rows = 0
    for route in routes:
        text = tokenizer.decode(route["continuation_token_ids"], skip_special_tokens=False)
        parsed = native_record(text, route["case"], goldens[int(route["image_id"])], "im_end")
        require(parsed.get("parse_status") == "accepted" and parsed.get("metric_bearing") is True, "new teacher parser status")
        require(parsed.get("dropped_prediction_count") == 0, "new teacher parser drops")
        require(len(parsed["pred"]) == len(route["trusted_boxes"]), "new teacher parser row count")
        for parsed_row, box, card in zip(parsed["pred"], route["trusted_boxes"], route["provenance"]["trace"]):
            fields = card["edited_fields"]
            require(parsed_row["coord_bins"] == box["expected_bins"] == fields["catalog_reference_coord_bins_1000"], "parser geometry identity")
            require(parsed_row["description"] == fields["selected_description"], "parser description identity")
        parsed_rows += len(parsed["pred"])
    require(parsed_rows == 218, "parser retained denominator")
    return {
        "parser": "src.inference.parsing.parse_compact_object_box_closed",
        "images": len(routes),
        "parsed_rows": parsed_rows,
        "malformed_rows": 0,
        "dropped_rows": 0,
    }


def validate_bank(bank: Mapping[str, Any], *, verify_sources: bool = True) -> dict[str, Any]:
    require(bank.get("schema") == SCHEMA, "teacher schema")
    expected = {key: value for key, value in bank.items() if key != "content_sha256"}
    require(bank.get("content_sha256") == digest(expected), "teacher content identity")
    require(bank.get("status") == "candidate_ready", "teacher lifecycle")
    require(bank.get("target_version") == TARGET_VERSION, "teacher target version")
    sources = bank.get("sources")
    require(isinstance(sources, Mapping), "teacher sources")
    preparation_path = _checked_binding(sources.get("preparation"), name="preparation") if verify_sources else Path(sources["preparation"]["path"])
    source_bank_binding = dict(sources.get("source_bank", {}))
    source_bank_content_sha = source_bank_binding.pop("bank_content_sha256", None)
    require(isinstance(source_bank_content_sha, str) and source_bank_content_sha, "source bank content binding")
    source_bank_path = _checked_binding(source_bank_binding, name="source_bank") if verify_sources else Path(source_bank_binding["path"])
    preparation = _load_preparation(preparation_path)
    source_bank = read(source_bank_path)
    require(source_bank.get("content_sha256") == source_bank_content_sha, "source bank content identity")
    source_routes = {int(route["image_id"]): route for route in source_bank.get("routes", [])}
    selected = _selected_by_image(preparation)
    routes = bank.get("routes")
    require(isinstance(routes, list) and len(routes) == 11, "teacher route count")
    observed_images = [route.get("image_id") for route in routes]
    require(observed_images == list(preparation["image_ids"]), "teacher image order")
    owners: list[tuple[int, str]] = []
    for route in routes:
        image_id = int(route["image_id"])
        require(image_id in source_routes and image_id in selected, "teacher route image")
        source_route = source_routes[image_id]
        require(route["case"] == source_route["case"], "teacher case identity")
        require(route["prompt_token_ids"] == source_route["prompt_token_ids"], "teacher prompt identity")
        expected_owners = [str(entry["owner_id"]) for entry in selected[image_id]]
        actual_owners = route["provenance"].get("selected_owner_ids")
        require(actual_owners == expected_owners, "teacher owner selection/order")
        require(route["provenance"].get("source_bank_route_id") == source_route["route_id"], "teacher source route")
        require(len(route["trusted_boxes"]) == len(expected_owners), "teacher row count")
        validate_route(route, eos_token_id=EOS)
        for box, card, entry in zip(route["trusted_boxes"], route["provenance"]["trace"], selected[image_id]):
            fields = card["edited_fields"]
            require(card.get("owner_id") == entry["owner_id"], "teacher trace owner")
            require(card.get("source_order") == entry["source_order"], "teacher trace source order")
            require(fields.get("selected_description") == entry["source_description"], "teacher trace description")
            require(fields.get("catalog_reference_coord_bins_1000") == entry["reference_bins"] == box["expected_bins"], "teacher trace geometry")
            owners.append((image_id, str(entry["owner_id"])))
    require(len(owners) == len(set(owners)) == 218, "teacher global included218")
    ledger = bank.get("source232_partition_ledger")
    require(ledger == preparation["owners"], "teacher source232 ledger")
    return {
        "routes": len(routes),
        "images": len(observed_images),
        "owner_rows": len(owners),
        "source232_partition": {
            "included": 218,
            "confirmed_out_of_scope": 1,
            "pending_category_or_scope": 13,
        },
    }


def build(
    *,
    preparation_path: Path = PREPARATION,
    source_bank_path: Path = SOURCE_BANK,
    output: Path = OUTPUT,
) -> dict[str, Any]:
    """Build, cold-parse, validate and publish the retained-only teacher."""

    from probes.training_set_completion.route_bank import _load_tokenizer

    require(not (output / "bank.json").exists(), f"bank collision: {output / 'bank.json'}")
    preparation_path = preparation_path.resolve(strict=True)
    source_bank_path = source_bank_path.resolve(strict=True)
    preparation = _load_preparation(preparation_path)
    source_bank = read(source_bank_path)
    require(source_bank.get("schema") == "training_set_completion.complete_synthetic_teacher_bank.v1", "source bank schema")
    require(source_bank.get("status") == "candidate_ready", "source bank lifecycle")
    source_routes = {int(route["image_id"]): route for route in source_bank.get("routes", [])}
    require(len(source_routes) == 11, "source bank route count")
    selected = _selected_by_image(preparation)
    require(set(source_routes) == set(selected) == set(preparation["image_ids"]), "source/current image identity")

    first_fit_manifest = Path(source_bank["sources"]["first_fit_preparation"]["path"]).resolve(strict=True)
    require(binding(first_fit_manifest) == source_bank["sources"]["first_fit_preparation"], "source tokenizer manifest changed")
    first_fit = read(first_fit_manifest)
    base_model = Path(first_fit["model_config"]["model"]["base_model"])
    tokenizer = _load_tokenizer(base_model)
    coordinate_ids = coordinate_token_table(base_model)["ids"]
    routes = [
        reassemble_route(source_routes[image_id], selected[image_id], tokenizer=tokenizer, coordinate_ids=coordinate_ids)
        for image_id in preparation["image_ids"]
    ]
    acquisition_path = Path(first_fit["acquisition_manifest"]["path"]).resolve(strict=True)
    require(binding(acquisition_path) == first_fit["acquisition_manifest"], "source acquisition changed")
    parser_receipt = _parser_roundtrip(routes, acquisition_path=acquisition_path, tokenizer=tokenizer)
    bank = {
        "schema": SCHEMA,
        "status": "candidate_ready",
        "target_version": TARGET_VERSION,
        "sources": {
            "preparation": binding(preparation_path),
            "source_bank": {
                **binding(source_bank_path),
                "bank_content_sha256": source_bank["content_sha256"],
            },
            "first_fit_preparation": binding(first_fit_manifest),
            "acquisition_manifest": binding(acquisition_path),
            "producer": binding(Path(__file__)),
        },
        "source232_partition_ledger": copy.deepcopy(preparation["owners"]),
        "legacy_suffix_state": {
            "source": "third-complete-bank-preparation-v2",
            "status": "invalidated",
            "reason": "row deletion creates a new synthetic teacher; KV/logits/probabilities are not reusable",
        },
        "fixed_owner_count": 218,
        "image_count": 11,
        "parser_receipt": parser_receipt,
        "routes": routes,
        "content_sha256": None,
    }
    bank["content_sha256"] = digest({key: value for key, value in bank.items() if key != "content_sha256"})
    checks = validate_bank(bank)
    admission = {
        "schema": ADMISSION_SCHEMA,
        "status": "cpu_admitted_not_training_admission",
        "teacher_bank": {"path": str((output / "bank.json").resolve()), "sha256": digest(bank), "size_bytes": len(canonical(bank))},
        "checks": {**checks, "parser_roundtrip": parser_receipt},
        "non_authorizations": [
            "no model forward or GPU work executed",
            "no training manifest or optimizer state produced",
            "no old suffix KV/logits/probabilities reused",
        ],
    }
    publish(output / "bank.json", bank)
    # Bind the published bytes, rather than the in-memory canonical payload.
    admission["teacher_bank"] = binding(output / "bank.json")
    publish(output / "admission.json", admission)
    return bank


def verify_published(*, bank_path: Path, receipt_path: Path) -> dict[str, Any]:
    """Re-run the strengthened CPU admission without changing immutable bank bytes."""

    from probes.training_set_completion.route_bank import _load_tokenizer
    import src.eval.detection_categories as detection_categories

    bank_path = bank_path.resolve(strict=True)
    require(not receipt_path.exists() and not receipt_path.is_symlink(), f"verification receipt collision: {receipt_path}")
    bank = read(bank_path)
    checks = validate_bank(bank)
    first_fit_path = _checked_binding(bank["sources"]["first_fit_preparation"], name="first_fit_preparation")
    acquisition_path = _checked_binding(bank["sources"]["acquisition_manifest"], name="acquisition_manifest")
    first_fit = read(first_fit_path)
    base_model = Path(first_fit["model_config"]["model"]["base_model"])
    tokenizer = _load_tokenizer(base_model)
    coordinate_ids = coordinate_token_table(base_model)["ids"]
    for route in bank["routes"]:
        validate_route(route, eos_token_id=EOS, coordinate_token_ids=coordinate_ids)
    parser_receipt = _parser_roundtrip(bank["routes"], acquisition_path=acquisition_path, tokenizer=tokenizer)
    require(parser_receipt == bank["parser_receipt"], "published bank parser receipt changed")
    receipt = {
        "schema": ADMISSION_SCHEMA,
        "status": "cpu_admitted_not_training_admission",
        "version": 2,
        "teacher_bank": binding(bank_path),
        "sources": {
            "preparation": binding(Path(bank["sources"]["preparation"]["path"])),
            "producer": binding(Path(__file__)),
            "coco80_literal_registry": binding(Path(detection_categories.__file__)),
        },
        "checks": {
            **checks,
            "literal_coco80_membership": "all 218 retained descriptions are exact COCO_80_CLASS_NAMES members",
            "coordinate_token_position_validation": "all trusted positions match the bound coordinate token table",
            "parser_roundtrip": parser_receipt,
        },
        "non_authorizations": [
            "no model forward or GPU work executed",
            "no training manifest or optimizer state produced",
            "no old suffix KV/logits/probabilities reused",
        ],
    }
    publish(receipt_path, receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, default=PREPARATION)
    parser.add_argument("--source-bank", type=Path, default=SOURCE_BANK)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--verify-bank", type=Path, default=None, help="cold-validate an already published bank")
    parser.add_argument("--admission-output", type=Path, default=None, help="new receipt path for --verify-bank")
    args = parser.parse_args(argv)
    if args.verify_bank is not None:
        receipt_path = args.admission_output or args.verify_bank.parent / "admission-v2.json"
        receipt = verify_published(bank_path=args.verify_bank, receipt_path=receipt_path)
        print(json.dumps({"bank": str(args.verify_bank.resolve()), "receipt": str(receipt_path.resolve()), "checks": receipt["checks"]}, sort_keys=True))
        return 0
    bank = build(
        preparation_path=args.preparation,
        source_bank_path=args.source_bank,
        output=args.output,
    )
    print(json.dumps({"output": str(args.output.resolve()), "owner_rows": bank["fixed_owner_count"], "routes": len(bank["routes"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
