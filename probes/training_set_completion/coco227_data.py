"""Build the immutable 227-owner CE-normalization teacher.

The new teacher is deliberately an append-only continuation reconstruction:
every old-218 object field is reproduced from the admitted old teacher, and
only the nine already verified COCO-80 current-known owners are appended per
image before EOS.  It does not reuse an old continuation suffix, cache, logits,
or probabilities.
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from probes.training_set_completion import coco80_bank
from probes.training_set_completion.complete_bank import EOS, _field_ids
from probes.training_set_completion.training import coordinate_token_table, validate_route
from src.eval.detection_categories import COCO_80_CLASS_NAMES


ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-15-coco227-ce-normalization"
)
OLD_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-14-training-set-completion-curriculum"
)
OLD_BANK = OLD_ROOT / "dual-start-coco80-teacher-v1/bank.json"
EVALUATION_PREPARATION = OLD_ROOT / "dual-start-evaluation-preparation-v1/preparation-v4.json"
OUTPUT = ROOT / "data-v1"

SCHEMA = "training_set_completion.coco227_ce_normalization_teacher.v1"
ADMISSION_SCHEMA = SCHEMA + ".cpu_admission"
TARGET_VERSION = "coco227-old218-plus-current-known-verified-coco80-v1"
OLD_COUNT = 218
NEW_COUNT = 9
OWNER_COUNT = OLD_COUNT + NEW_COUNT
IMAGE_COUNT = 11

require = coco80_bank.require
canonical = coco80_bank.canonical
digest = coco80_bank.digest
binding = coco80_bank.binding
read = coco80_bank.read
publish = coco80_bank.publish
_checked_binding = coco80_bank._checked_binding


def _owner_key(row: Mapping[str, Any]) -> tuple[int, str]:
    image_id, owner_id = row.get("image_id"), row.get("owner_id")
    require(type(image_id) is int and isinstance(owner_id, str) and owner_id, "owner identity")
    return image_id, owner_id


def _literal_coco80(description: Any, *, owner_id: str) -> str:
    require(isinstance(description, str) and description, f"COCO description missing: {owner_id}")
    require(description in COCO_80_CLASS_NAMES, f"description outside literal COCO-80: {owner_id}: {description!r}")
    return description


def _old_records(bank: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Project the old teacher trace to a provenance-preserving owner ledger."""

    require(bank.get("schema") == coco80_bank.SCHEMA, "old218 teacher schema")
    require(bank.get("status") == "candidate_ready", "old218 teacher status")
    routes = bank.get("routes")
    require(isinstance(routes, list) and len(routes) == IMAGE_COUNT, "old218 route count")
    result: list[dict[str, Any]] = []
    for route in routes:
        image_id = route.get("image_id")
        require(type(image_id) is int, "old218 image identity")
        trace = route.get("provenance", {}).get("trace")
        require(isinstance(trace, list), "old218 trace")
        for card in trace:
            require(isinstance(card, Mapping), "old218 trace card")
            fields = card.get("edited_fields")
            require(isinstance(fields, Mapping), "old218 trace fields")
            owner_id = str(card.get("owner_id", ""))
            description = _literal_coco80(fields.get("selected_description"), owner_id=owner_id)
            bins = fields.get("catalog_reference_coord_bins_1000")
            require(
                isinstance(bins, list)
                and len(bins) == 4
                and all(type(value) is int for value in bins)
                and bins[0] < bins[2]
                and bins[1] < bins[3],
                f"old218 geometry: {owner_id}",
            )
            result.append(
                {
                    "image_id": image_id,
                    "owner_id": owner_id,
                    "description": description,
                    "reference_coord_bins_1000": list(bins),
                    "source_card": copy.deepcopy(dict(card)),
                }
            )
    keys = [_owner_key(item) for item in result]
    require(len(result) == OLD_COUNT and len(keys) == len(set(keys)), "old218 owner population")
    return result


def additions_from_sources(
    *, old_bank: Mapping[str, Any], evaluation_preparation: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Return the only permitted additions: verified current-known COCO owners."""

    require(
        evaluation_preparation.get("schema")
        == "training_set_completion.dual_start_paired_evaluation.v1.preparation",
        "evaluation preparation schema",
    )
    require(evaluation_preparation.get("status") == "candidate_ready", "evaluation preparation status")
    old_keys = {_owner_key(row) for row in _old_records(old_bank)}
    ledgers = evaluation_preparation.get("ledgers")
    require(isinstance(ledgers, Mapping), "evaluation ledgers")
    current = ledgers.get("current-known248")
    require(isinstance(current, list) and len(current) == 248, "current-known248 population")
    current_keys = [_owner_key(row) for row in current if isinstance(row, Mapping)]
    require(len(current_keys) == len(current) == len(set(current_keys)), "current-known248 identities")
    statuses = Counter(str(row.get("class_status")) for row in current)
    require(
        statuses == {"verified_coco80": 227, "unknown": 19, "verified_non_coco": 2},
        "current-known248 frozen class partition",
    )
    current_by_key = {_owner_key(row): row for row in current}
    require(old_keys <= set(current_by_key), "old218 absent from current-known248")
    for key in old_keys:
        old = next(item for item in _old_records(old_bank) if _owner_key(item) == key)
        current_row = current_by_key[key]
        require(current_row.get("class_status") == "verified_coco80", f"old218 current class: {key}")
        require(old["description"] == current_row.get("description"), f"old218 current description: {key}")
        require(old["reference_coord_bins_1000"] == current_row.get("reference_coord_bins_1000"), f"old218 current geometry: {key}")
    additions = []
    for row in current:
        require(isinstance(row, Mapping), "current-known entry")
        key = _owner_key(row)
        if key in old_keys:
            continue
        if row.get("class_status") != "verified_coco80":
            continue
        owner_id = key[1]
        description = _literal_coco80(row.get("description"), owner_id=owner_id)
        bins = row.get("reference_coord_bins_1000")
        require(
            isinstance(bins, list)
            and len(bins) == 4
            and all(type(value) is int for value in bins)
            and bins[0] < bins[2]
            and bins[1] < bins[3],
            f"new9 geometry: {owner_id}",
        )
        additions.append(
            {
                "image_id": key[0],
                "owner_id": owner_id,
                "description": description,
                "reference_coord_bins_1000": list(bins),
                "class_status": "verified_coco80",
                "source_current_known248": copy.deepcopy(dict(row)),
            }
        )
    additions.sort(key=lambda row: (int(row["image_id"]), str(row["owner_id"])))
    keys = [_owner_key(item) for item in additions]
    require(len(additions) == NEW_COUNT and len(keys) == len(set(keys)), "verified current-known additions")
    require(
        Counter(str(item["description"]) for item in additions)
        == {"person": 1, "donut": 7, "chair": 1},
        "new9 frozen description population",
    )
    return additions


def _addition_card(*, row: Mapping[str, Any], order: int, positions: list[int]) -> dict[str, Any]:
    owner_id = str(row["owner_id"])
    return {
        "owner_id": owner_id,
        "order": order,
        "source_kind": "current_known248_verified_coco80_append",
        "source_order": {"stable_owner_id_order": order},
        "source_decision": {
            "class_status": "verified_coco80",
            "ledger": "current-known248",
            "selection": "not_old218_verified_coco80",
        },
        "edited_fields": {
            "description_source": "current_known248_verified_description",
            "selected_description": row["description"],
            "description_ce_positive": True,
            "description_authority": "evaluation-preparation-v4.current-known248",
            "description_token_positions": positions,
            "catalog_reference_coord_bins_1000": list(row["reference_coord_bins_1000"]),
            "geometry_replaced": False,
        },
        "source_trace": {
            "ledger": "current-known248",
            "entry": copy.deepcopy(dict(row["source_current_known248"])),
        },
    }


def reassemble_route(
    old_route: Mapping[str, Any],
    additions: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any,
    coordinate_ids: Sequence[int],
) -> dict[str, Any]:
    """Recompute one old-prefix-plus-new-suffix route and its trusted positions."""

    image_id = old_route.get("image_id")
    require(type(image_id) is int, "old route image")
    old_trace = old_route.get("provenance", {}).get("trace")
    require(isinstance(old_trace, list), "old route trace")
    additions = list(additions)
    require(all(int(row["image_id"]) == image_id for row in additions), "cross-image append")
    addition_ids = [str(row["owner_id"]) for row in additions]
    require(addition_ids == sorted(addition_ids), "append order is not stable owner-ID order")
    old_owner_ids = [str(card.get("owner_id")) for card in old_trace]
    require(len(old_owner_ids) == len(set(old_owner_ids)), "old route duplicate owner")
    require(not set(old_owner_ids) & set(addition_ids), "old/new owner overlap")

    continuation: list[int] = []
    weights: list[int] = []
    boxes: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []

    for order, source_card in enumerate(old_trace):
        require(isinstance(source_card, Mapping), "old trace card")
        fields = source_card.get("edited_fields")
        require(isinstance(fields, Mapping), "old trace fields")
        owner_id = str(source_card.get("owner_id", ""))
        description = _literal_coco80(fields.get("selected_description"), owner_id=owner_id)
        bins = list(fields.get("catalog_reference_coord_bins_1000", []))
        start = len(continuation)
        ids = tokenizer.encode(description, add_special_tokens=False)
        require(ids and tokenizer.decode(ids, skip_special_tokens=False) == description, f"literal tokenization: {owner_id}")
        continuation.extend(_field_ids(tokenizer, description, bins, list(coordinate_ids)))
        weights.extend([1] * (len(ids) + 8))
        positions = list(range(start + 1 + len(ids) + 2, start + 1 + len(ids) + 6))
        source_positions = fields.get("description_token_positions")
        require(source_positions == list(range(start + 1, start + 1 + len(ids))), f"old literal description positions: {owner_id}")
        expected_box = {
            "x1_position": positions[0],
            "y1_position": positions[1],
            "x2_position": positions[2],
            "y2_position": positions[3],
            "expected_bins": bins,
        }
        require(old_route["trusted_boxes"][order] == expected_box, f"old literal trusted position: {owner_id}")
        boxes.append(expected_box)
        trace.append(copy.deepcopy(dict(source_card)))

    old_prefix = list(continuation)
    require(old_prefix == list(old_route["continuation_token_ids"][:-1]), "old literal continuation prefix")
    require(weights == list(old_route["ce_weights"][:-1]), "old literal CE prefix")

    for relative_order, row in enumerate(additions):
        owner_id = str(row["owner_id"])
        description = _literal_coco80(row.get("description"), owner_id=owner_id)
        bins = list(row.get("reference_coord_bins_1000", []))
        start = len(continuation)
        ids = tokenizer.encode(description, add_special_tokens=False)
        require(ids and tokenizer.decode(ids, skip_special_tokens=False) == description, f"literal tokenization: {owner_id}")
        continuation.extend(_field_ids(tokenizer, description, bins, list(coordinate_ids)))
        weights.extend([1] * (len(ids) + 8))
        positions = list(range(start + 1 + len(ids) + 2, start + 1 + len(ids) + 6))
        boxes.append(
            {
                "x1_position": positions[0],
                "y1_position": positions[1],
                "x2_position": positions[2],
                "y2_position": positions[3],
                "expected_bins": bins,
            }
        )
        trace.append(_addition_card(row=row, order=len(old_trace) + relative_order, positions=list(range(start + 1, start + 1 + len(ids)))))
    continuation.append(EOS)
    weights.append(1)
    case = copy.deepcopy(old_route["case"])
    plan = case["image_plan"]
    route = {
        "route_id": f"coco227-ce-normalization:image-{image_id:012d}",
        "image_id": image_id,
        "example_id": old_route["example_id"],
        "case": case,
        "image_identity": {
            "image_path": case["image_path"],
            "image_content_sha256": plan["image_content_sha256"],
            "executed_media_sha256": plan["executed_media_sha256"],
            "observed_image_grid_thw": plan["observed_image_grid_thw"],
        },
        "prompt_token_ids": copy.deepcopy(old_route["prompt_token_ids"]),
        "continuation_token_ids": continuation,
        "ce_weights": weights,
        "trusted_boxes": boxes,
        "trusted_complete_support_endpoint": True,
        "provenance": {
            "synthetic_teacher": True,
            "old218_source_bank_route_id": old_route["route_id"],
            "old218_owner_ids": old_owner_ids,
            "new9_owner_ids": addition_ids,
            "selected_owner_ids": old_owner_ids + addition_ids,
            "old218_literal_prefix_token_sha256": digest(old_prefix),
            "trace": trace,
            "legacy_suffix_state": "invalidated_by_append_and_full_reassembly; KV/logits/probabilities are not reusable",
        },
    }
    validate_route(route, eos_token_id=EOS, coordinate_token_ids=coordinate_ids)
    return route


def _parser_roundtrip(routes: Sequence[Mapping[str, Any]], *, acquisition_path: Path, tokenizer: Any) -> dict[str, Any]:
    """Cold-parse freshly constructed continuations with the production parser."""

    from src.eval.native_rows import native_detection_record as native_record

    acquisition = read(acquisition_path)
    goldens = {int(record["image_id"]): record["golden"] for record in acquisition["records"]}
    parsed_rows = 0
    for route in routes:
        text = tokenizer.decode(route["continuation_token_ids"], skip_special_tokens=False)
        parsed = native_record(text, route["case"], goldens[int(route["image_id"])], "im_end")
        require(parsed.get("parse_status") == "accepted" and parsed.get("metric_bearing") is True, "teacher parser status")
        require(parsed.get("dropped_prediction_count") == 0, "teacher parser drops")
        require(len(parsed["pred"]) == len(route["trusted_boxes"]), "teacher parser row count")
        for prediction, box, card in zip(parsed["pred"], route["trusted_boxes"], route["provenance"]["trace"]):
            fields = card["edited_fields"]
            require(prediction["coord_bins"] == box["expected_bins"] == fields["catalog_reference_coord_bins_1000"], "parser geometry identity")
            require(prediction["description"] == fields["selected_description"], "parser description identity")
        parsed_rows += len(parsed["pred"])
    require(parsed_rows == OWNER_COUNT, "teacher parser denominator")
    return {
        "parser": "src.inference.parsing.parse_compact_object_box_closed",
        "images": len(routes),
        "parsed_rows": parsed_rows,
        "malformed_rows": 0,
        "dropped_rows": 0,
    }


def _by_image(rows: Sequence[Mapping[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[int(row["image_id"])].append(dict(row))
    return dict(result)


def validate_bank(bank: Mapping[str, Any], *, verify_sources: bool = True) -> dict[str, Any]:
    require(bank.get("schema") == SCHEMA, "teacher schema")
    require(bank.get("status") == "candidate_ready", "teacher status")
    require(bank.get("target_version") == TARGET_VERSION, "teacher target version")
    expected = {key: value for key, value in bank.items() if key != "content_sha256"}
    require(bank.get("content_sha256") == digest(expected), "teacher content identity")
    sources = bank.get("sources")
    require(isinstance(sources, Mapping), "teacher sources")
    old_path = _checked_binding(sources["old218_teacher_bank"], name="old218 teacher") if verify_sources else Path(sources["old218_teacher_bank"]["path"])
    prep_path = _checked_binding(sources["evaluation_preparation_v4"], name="evaluation preparation") if verify_sources else Path(sources["evaluation_preparation_v4"]["path"])
    old_bank, evaluation_preparation = read(old_path), read(prep_path)
    old_routes = {int(route["image_id"]): route for route in old_bank["routes"]}
    additions = additions_from_sources(old_bank=old_bank, evaluation_preparation=evaluation_preparation)
    additions_by_image = _by_image(additions)
    routes = bank.get("routes")
    require(isinstance(routes, list) and len(routes) == IMAGE_COUNT, "teacher route count")
    expected_images = [int(route["image_id"]) for route in old_bank["routes"]]
    require([route.get("image_id") for route in routes] == expected_images, "teacher image order")
    observed_old: list[tuple[int, str]] = []
    observed_new: list[tuple[int, str]] = []
    for route in routes:
        image_id = int(route["image_id"])
        old_route = old_routes[image_id]
        old_trace = old_route["provenance"]["trace"]
        trace = route.get("provenance", {}).get("trace")
        require(isinstance(trace, list) and trace[: len(old_trace)] == old_trace, "old218 literal trace prefix")
        require(route["case"] == old_route["case"], "teacher case identity")
        require(route["prompt_token_ids"] == old_route["prompt_token_ids"], "teacher prompt identity")
        require(route["provenance"].get("old218_source_bank_route_id") == old_route["route_id"], "old route provenance")
        old_ids = [str(card["owner_id"]) for card in old_trace]
        new_rows = additions_by_image.get(image_id, [])
        new_ids = [str(row["owner_id"]) for row in new_rows]
        require(route["provenance"].get("old218_owner_ids") == old_ids, "old owner order")
        require(route["provenance"].get("new9_owner_ids") == new_ids, "new owner order")
        require(route["provenance"].get("selected_owner_ids") == old_ids + new_ids, "full owner order")
        require(len(route["trusted_boxes"]) == len(old_ids) + len(new_ids), "route owner count")
        require(route["continuation_token_ids"][: -1][: len(old_route["continuation_token_ids"]) - 1] == old_route["continuation_token_ids"][:-1], "old literal continuation")
        require(route["ce_weights"][: -1][: len(old_route["ce_weights"]) - 1] == old_route["ce_weights"][:-1], "old literal CE weights")
        require(route["trusted_boxes"][: len(old_trace)] == old_route["trusted_boxes"], "old literal positions")
        validate_route(route, eos_token_id=EOS)
        for card in trace:
            fields = card.get("edited_fields", {})
            _literal_coco80(fields.get("selected_description"), owner_id=str(card.get("owner_id")))
        observed_old.extend((image_id, item) for item in old_ids)
        observed_new.extend((image_id, item) for item in new_ids)
    require(len(observed_old) == len(set(observed_old)) == OLD_COUNT, "old218 observed population")
    require(len(observed_new) == len(set(observed_new)) == NEW_COUNT, "new9 observed population")
    require(bank.get("old218_owner_ids") == [{"image_id": image, "owner_id": owner} for image, owner in observed_old], "old218 top-level ledger")
    require(bank.get("new9_owner_ids") == [{"image_id": image, "owner_id": owner} for image, owner in observed_new], "new9 top-level ledger")
    return {
        "routes": len(routes),
        "images": len(expected_images),
        "owner_rows": OWNER_COUNT,
        "old218_rows": OLD_COUNT,
        "new9_rows": NEW_COUNT,
        "new9_description_population": dict(sorted(Counter(str(row["description"]) for row in additions).items())),
        "current_known248_partition": {"verified_coco80": 227, "unknown": 19, "verified_non_coco": 2},
    }


def build(
    *,
    old_bank_path: Path = OLD_BANK,
    evaluation_preparation_path: Path = EVALUATION_PREPARATION,
    output: Path = OUTPUT,
) -> dict[str, Any]:
    """Build, cold-parse, validate, and publish the new teacher once."""

    from probes.training_set_completion.route_bank import _load_tokenizer

    require(not (output / "bank.json").exists(), f"teacher collision: {output / 'bank.json'}")
    old_bank_path = old_bank_path.resolve(strict=True)
    evaluation_preparation_path = evaluation_preparation_path.resolve(strict=True)
    old_bank, evaluation_preparation = read(old_bank_path), read(evaluation_preparation_path)
    old_records = _old_records(old_bank)
    additions = additions_from_sources(old_bank=old_bank, evaluation_preparation=evaluation_preparation)
    source_first_fit = old_bank["sources"]["first_fit_preparation"]
    first_fit_path = _checked_binding(source_first_fit, name="old first-fit preparation")
    first_fit = read(first_fit_path)
    model_root = Path(first_fit["model_config"]["model"]["base_model"])
    tokenizer = _load_tokenizer(model_root)
    coordinate_ids = coordinate_token_table(model_root)["ids"]
    old_routes = {int(route["image_id"]): route for route in old_bank["routes"]}
    additions_by_image = _by_image(additions)
    routes = [
        reassemble_route(old_routes[image_id], additions_by_image.get(image_id, []), tokenizer=tokenizer, coordinate_ids=coordinate_ids)
        for image_id in [int(route["image_id"]) for route in old_bank["routes"]]
    ]
    acquisition_path = _checked_binding(old_bank["sources"]["acquisition_manifest"], name="old acquisition")
    parser_receipt = _parser_roundtrip(routes, acquisition_path=acquisition_path, tokenizer=tokenizer)
    old_owner_ids = [{"image_id": row["image_id"], "owner_id": row["owner_id"]} for row in old_records]
    new_owner_ids = [{"image_id": row["image_id"], "owner_id": row["owner_id"]} for row in additions]
    bank = {
        "schema": SCHEMA,
        "status": "candidate_ready",
        "target_version": TARGET_VERSION,
        "sources": {
            "old218_teacher_bank": binding(old_bank_path),
            "evaluation_preparation_v4": binding(evaluation_preparation_path),
            "first_fit_preparation": binding(first_fit_path),
            "acquisition_manifest": binding(acquisition_path),
            "producer": binding(Path(__file__)),
        },
        "old218_owner_ids": old_owner_ids,
        "new9_owner_ids": new_owner_ids,
        "route_additions": [
            {"image_id": image_id, "owner_ids": [str(row["owner_id"]) for row in additions_by_image.get(image_id, [])]}
            for image_id in [int(route["image_id"]) for route in old_bank["routes"]]
        ],
        "legacy_suffix_state": {
            "source": "dual-start-coco80-teacher-v1",
            "status": "invalidated",
            "reason": "appending owners creates a new synthetic teacher; old KV/logits/probabilities are not reusable",
        },
        "fixed_owner_count": OWNER_COUNT,
        "old218_owner_count": OLD_COUNT,
        "new9_owner_count": NEW_COUNT,
        "image_count": IMAGE_COUNT,
        "parser_receipt": parser_receipt,
        "routes": routes,
        "content_sha256": None,
    }
    bank["content_sha256"] = digest({key: value for key, value in bank.items() if key != "content_sha256"})
    checks = validate_bank(bank)
    new9_ledger = {
        "schema": SCHEMA + ".new9_ledger",
        "status": "source_derived_verified_current_known_only",
        "teacher_bank_content_sha256": bank["content_sha256"],
        "source_evaluation_preparation": binding(evaluation_preparation_path),
        "owners": additions,
    }
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
    publish(output / "new9-ledger.json", new9_ledger)
    admission["teacher_bank"] = binding(output / "bank.json")
    admission["new9_ledger"] = binding(output / "new9-ledger.json")
    publish(output / "admission-v1.json", admission)
    return bank


def verify_published(*, bank_path: Path, receipt_path: Path) -> dict[str, Any]:
    """Re-run strong CPU checks against a published immutable bank."""

    from probes.training_set_completion.route_bank import _load_tokenizer

    bank_path = bank_path.resolve(strict=True)
    require(not receipt_path.exists() and not receipt_path.is_symlink(), f"verification receipt collision: {receipt_path}")
    bank = read(bank_path)
    checks = validate_bank(bank)
    first_fit_path = _checked_binding(bank["sources"]["first_fit_preparation"], name="first-fit preparation")
    first_fit = read(first_fit_path)
    model_root = Path(first_fit["model_config"]["model"]["base_model"])
    tokenizer = _load_tokenizer(model_root)
    coordinate_ids = coordinate_token_table(model_root)["ids"]
    for route in bank["routes"]:
        validate_route(route, eos_token_id=EOS, coordinate_token_ids=coordinate_ids)
    acquisition_path = _checked_binding(bank["sources"]["acquisition_manifest"], name="acquisition")
    parser_receipt = _parser_roundtrip(bank["routes"], acquisition_path=acquisition_path, tokenizer=tokenizer)
    require(parser_receipt == bank["parser_receipt"], "published parser receipt")
    receipt = {
        "schema": ADMISSION_SCHEMA,
        "status": "cpu_admitted_not_training_admission",
        "version": 1,
        "teacher_bank": binding(bank_path),
        "sources": {
            "old218_teacher_bank": binding(Path(bank["sources"]["old218_teacher_bank"]["path"])),
            "evaluation_preparation_v4": binding(Path(bank["sources"]["evaluation_preparation_v4"]["path"])),
            "producer": binding(Path(__file__)),
        },
        "checks": {**checks, "coordinate_token_position_validation": "all 227 trusted positions match the bound coordinate token table", "parser_roundtrip": parser_receipt},
        "non_authorizations": ["no model forward or GPU work executed", "no training manifest or optimizer state produced"],
    }
    publish(receipt_path, receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank218", type=Path, default=OLD_BANK)
    parser.add_argument("--evaluation-preparation", type=Path, default=EVALUATION_PREPARATION)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--verify-bank", type=Path)
    parser.add_argument("--admission-output", type=Path)
    args = parser.parse_args(argv)
    if args.verify_bank is not None:
        receipt = verify_published(
            bank_path=args.verify_bank,
            receipt_path=args.admission_output or args.verify_bank.parent / "admission-v2.json",
        )
        print({"bank": str(args.verify_bank.resolve()), "receipt": receipt["teacher_bank"]})
        return 0
    bank = build(old_bank_path=args.bank218, evaluation_preparation_path=args.evaluation_preparation, output=args.output)
    print({"output": str(args.output.resolve()), "owners": bank["fixed_owner_count"], "routes": len(bank["routes"])})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
