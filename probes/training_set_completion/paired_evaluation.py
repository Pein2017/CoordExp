"""Saved-natural-readback scoring for the COCO-80 dual-start comparison.

The evaluator deliberately separates annotation matching from physical review.
An annotation-unmatched prediction is never promoted to a physical false
positive without an exact, hash-identified prior review record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from probes.training_set_completion import readback_selectors
from src.data.geometry import parse_source_bbox_tokens
from src.eval.detection_categories import COCO_80_CLASS_NAMES


ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-14-training-set-completion-curriculum"
)
PREPARATION = ROOT / "dual-start-coco80-preparation-v1/preparation.json"
TEACHER_BANK = ROOT / "dual-start-coco80-teacher-v1/bank.json"
ANNOTATIONS = ROOT / "annotations-with-unlabeled-v4/annotations.jsonl"
REVIEW_INDEX = ROOT / "fourth-fit-review-extraction-v1/evidence-index.json"
REVIEW_RESULTS = ROOT / "fourth-fit-review-extraction-v1/full-review-results.jsonl"
OUTPUT = ROOT / "dual-start-evaluation-preparation-v1"

SCHEMA = "training_set_completion.dual_start_paired_evaluation.v1"
PREPARATION_SCHEMA = SCHEMA + ".preparation"
IOU_PRIMARY = 0.5
IOU_DIAGNOSTIC = 0.8
CAP = 3084


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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


def _target(
    *,
    image_id: int,
    owner_id: str,
    bins: Sequence[int],
    description: str | None,
    class_status: str,
) -> dict[str, Any]:
    require(type(image_id) is int and isinstance(owner_id, str) and owner_id, "target identity")
    checked = list(bins)
    require(
        len(checked) == 4
        and all(type(value) is int for value in checked)
        and checked[0] < checked[2]
        and checked[1] < checked[3],
        f"target geometry: {owner_id}",
    )
    require(description is None or isinstance(description, str), "target description")
    return {
        "image_id": image_id,
        "owner_id": owner_id,
        "reference_coord_bins_1000": checked,
        "description": description,
        "class_status": class_status,
    }


def _verified_class_status(description: str | None) -> str:
    return "verified_coco80" if description in COCO_80_CLASS_NAMES else "verified_non_coco"


def _scoped_ledger(bank: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    for route in bank["routes"]:
        image_id = int(route["image_id"])
        for card in route["provenance"]["trace"]:
            fields = card["edited_fields"]
            description = fields["selected_description"]
            require(description in COCO_80_CLASS_NAMES, "scoped teacher description outside literal COCO-80")
            result.append(
                _target(
                    image_id=image_id,
                    owner_id=str(card["owner_id"]),
                    bins=fields["catalog_reference_coord_bins_1000"],
                    description=description,
                    class_status=_verified_class_status(description),
                )
            )
    require(len(result) == 218, "scoped218 denominator")
    return result


def _historical_ledger(preparation: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    for entry in preparation["owners"]:
        included = entry["decision"] == "included"
        result.append(
            _target(
                image_id=int(entry["image_id"]),
                owner_id=str(entry["owner_id"]),
                bins=entry["reference_bins"],
                description=entry["source_description"] if entry.get("source_description_ce_positive") else None,
                class_status=_verified_class_status(entry["source_description"]) if included else "historical_class_unresolved",
            )
        )
    require(len(result) == 232, "historical232 denominator")
    return result


def _current_known_ledger(annotation_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in annotation_rows:
        image_id = row.get("image_id")
        require(type(image_id) is int, "current annotation image")
        for index, item in enumerate(row.get("objects", [])):
            owner_id = item.get("coco_ann_id")
            require(owner_id is not None, "current GT owner")
            result.append(
                _target(
                    image_id=image_id,
                    owner_id=str(owner_id),
                    bins=parse_source_bbox_tokens(item.get("bbox_2d"), field=f"current.{image_id}.objects[{index}].bbox_2d"),
                    description=item.get("desc"),
                    class_status=_verified_class_status(item.get("desc")),
                )
            )
        for index, item in enumerate(row.get("unlabeled", [])):
            status = item.get("class_status")
            require(status in {"verified", "unknown"}, "current unlabeled class status")
            result.append(
                _target(
                    image_id=image_id,
                    owner_id=str(item.get("stable_owner_id")),
                    bins=item.get("bbox_2d_bins_1000"),
                    description=item.get("desc") if status == "verified" else None,
                    class_status=_verified_class_status(item.get("desc")) if status == "verified" else "unknown",
                )
            )
    require(len(result) == 248, "current-known248 denominator")
    return result


def _validate_ledger(name: str, rows: Sequence[Mapping[str, Any]], *, expected: int, image_ids: Sequence[int]) -> None:
    keys = [(int(row["image_id"]), str(row["owner_id"])) for row in rows]
    require(len(rows) == expected and len(keys) == len(set(keys)), f"{name} owner denominator")
    require(set(image for image, _ in keys) == set(image_ids), f"{name} image coverage")


def build_preparation(
    *,
    preparation_path: Path = PREPARATION,
    teacher_bank_path: Path = TEACHER_BANK,
    annotations_path: Path = ANNOTATIONS,
    output: Path = OUTPUT,
    review_index_path: Path = REVIEW_INDEX,
    review_results_path: Path = REVIEW_RESULTS,
    artifact_name: str = "preparation.json",
) -> dict[str, Any]:
    """Freeze evaluation ledgers and the evidence boundary before any new readback."""

    require(Path(artifact_name).name == artifact_name, "preparation artifact must be one filename")
    artifact_path = output / artifact_name
    require(not artifact_path.exists(), f"preparation collision: {artifact_path}")
    preparation_path = preparation_path.resolve(strict=True)
    teacher_bank_path = teacher_bank_path.resolve(strict=True)
    annotations_path = annotations_path.resolve(strict=True)
    review_index_path = review_index_path.resolve(strict=True)
    review_results_path = review_results_path.resolve(strict=True)
    preparation, bank = read(preparation_path), read(teacher_bank_path)
    annotations = read_jsonl(annotations_path)
    require(preparation.get("target_version") == "coco80-source232-trusted-description-subset-v1", "preparation target version")
    require(bank.get("schema") == "training_set_completion.dual_start_coco80_teacher.v1", "teacher bank schema")
    image_ids = list(preparation["image_ids"])
    scoped = _scoped_ledger(bank)
    historical = _historical_ledger(preparation)
    current = _current_known_ledger(annotations)
    _validate_ledger("scoped218", scoped, expected=218, image_ids=image_ids)
    _validate_ledger("historical232", historical, expected=232, image_ids=image_ids)
    _validate_ledger("current-known248", current, expected=248, image_ids=image_ids)
    first_fit = Path(bank["sources"]["first_fit_preparation"]["path"]).resolve(strict=True)
    first_fit_manifest = read(first_fit)
    acquisition = Path(bank["sources"]["acquisition_manifest"]["path"]).resolve(strict=True)
    value = {
        "schema": PREPARATION_SCHEMA,
        "status": "candidate_ready",
        "image_ids": image_ids,
        "sources": {
            "dual_start_preparation": binding(preparation_path),
            "teacher_bank": binding(teacher_bank_path),
            "current_annotations": binding(annotations_path),
            "first_fit_preparation": binding(first_fit),
            "acquisition_manifest": binding(acquisition),
            "prior_review_evidence_index": binding(review_index_path),
            "prior_review_results": binding(review_results_path),
            "producer": binding(Path(__file__)),
        },
        "tokenizer_root": str(Path(first_fit_manifest["model_config"]["model"]["base_model"]).resolve(strict=True)),
        "readback_contract": {
            "new_dual_start_admission": "probes.training_set_completion.dual_start.load_admitted_readback_rows(result_path, arm, step)",
            "conditioning": "original bound image/prompt/media/grid; empty assistant prefix; natural greedy output",
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": 1.0,
            "max_new_tokens": CAP,
            "required_rows_per_readback": 11,
            "legacy": "requires an explicit source-admitted historical diagnostic receipt; it cannot stand in for a new dual-start arm.",
        },
        "matching_contract": {
            "primary": "class-agnostic cardinality-first one-to-one IoU >= 0.5",
            "diagnostic": "class-agnostic cardinality-first one-to-one IoU >= 0.8",
            "implementation": "probes.training_set_completion.readback_selectors.one_to_one_matches -> src.eval.assignment.global_matches with shared category",
        },
        "metric_contract": {
            "primary": "scoped218 FN count and FN rate",
            "annotation_relative_micro": "precision/F1: TP from scoped218 IoU0.5 matches; prediction denominator is every valid parsed prediction in this saved readback; annotation-unmatched is not a physical FP",
            "physical": "a current-known248 IoU0.5 one-to-one match is an accepted physical owner; otherwise exact image_id + raw_span_sha256 prior review may provide owner/class/extent facts. Repeat is recomputed from current generated order, and unresolved unmatched predictions remain physical unknown.",
            "outside_coco80": "exact literal membership in COCO_80_CLASS_NAMES; a violation remains separate even when its box owner-matches",
            "no_scalar_composite": True,
        },
        "ledgers": {
            "scoped218": scoped,
            "historical232": historical,
            "current-known248": current,
        },
        "content_sha256": None,
    }
    value["content_sha256"] = digest({key: item for key, item in value.items() if key != "content_sha256"})
    publish(artifact_path, value)
    return value


def _load_exact_review_map(preparation: Mapping[str, Any]) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, Any]]:
    sources = preparation["sources"]
    index_path = _checked_binding(sources["prior_review_evidence_index"], name="prior review evidence index")
    results_path = _checked_binding(sources["prior_review_results"], name="prior review results")
    index = read(index_path)
    require(isinstance(index, list) and all(isinstance(item, Mapping) for item in index), "prior review index schema")
    for item in index:
        path = Path(str(item.get("path", ""))).resolve(strict=True)
        require(file_sha256(path) == item.get("sha256"), f"prior review evidence changed: {path}")
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in read_jsonl(results_path):
        raw, decision = record.get("raw_row"), record.get("decision")
        if not isinstance(raw, Mapping) or not isinstance(decision, Mapping):
            continue
        image_id, span = record.get("image_id"), raw.get("raw_span_sha256")
        if type(image_id) is int and isinstance(span, str) and span:
            grouped[(image_id, span)].append(dict(decision))
    exact: dict[tuple[int, str], dict[str, Any]] = {}
    for key, decisions in grouped.items():
        # Reuse requires a stable physical/class/extent ruling across every
        # historical occurrence of the byte-identical span; disagreements stay unknown.
        projections = {
            (
                item.get("physical_status"),
                item.get("class"),
                item.get("extent"),
                item.get("owner_id"),
            )
            for item in decisions
        }
        if len(projections) == 1:
            exact[key] = decisions[0]
    return exact, {"evidence_index": binding(index_path), "review_results": binding(results_path), "exact_key_count": len(exact)}


def _by_image(rows: Iterable[Mapping[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[int(row["image_id"])].append(dict(row))
    return dict(result)


def _class_correct(*, target: Mapping[str, Any], prediction: Mapping[str, Any]) -> bool | None:
    if not str(target.get("class_status", "")).startswith("verified_"):
        return None
    return prediction.get("description") == target.get("description")


def _ledger_image(
    targets: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]], *, threshold: float
) -> dict[str, Any]:
    matches = readback_selectors.one_to_one_matches(list(targets), list(predictions), threshold)
    target_by_owner = {str(item["owner_id"]): item for item in targets}
    prediction_by_id = {str(item["prediction_id"]): item for item in predictions}
    matched_owner_ids = {str(item["reference_owner_id"]) for item in matches}
    matched_prediction_ids = {str(item["prediction_id"]) for item in matches}
    class_values = [
        _class_correct(target=target_by_owner[str(item["reference_owner_id"])], prediction=prediction_by_id[str(item["prediction_id"])])
        for item in matches
    ]
    return {
        "target_count": len(targets),
        "matched_count": len(matches),
        "fn_count": len(targets) - len(matches),
        "fn_rate": (len(targets) - len(matches)) / len(targets) if targets else 0.0,
        "covered_owner_ids": [str(item["reference_owner_id"]) for item in matches],
        "missing_owner_ids": [str(item["owner_id"]) for item in targets if str(item["owner_id"]) not in matched_owner_ids],
        "matched_prediction_ids": [str(item["prediction_id"]) for item in matches],
        "annotation_unmatched_prediction_ids": [str(item["prediction_id"]) for item in predictions if str(item["prediction_id"]) not in matched_prediction_ids],
        "matches": matches,
        "class_correct_count": sum(value is True for value in class_values),
        "class_wrong_count": sum(value is False for value in class_values),
        "class_unknown_count": sum(value is None for value in class_values),
    }


def _raw_debt(dropped: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reasons = Counter(str(item.get("drop_reason")) for item in dropped)
    geometry = sum(
        count
        for reason, count in reasons.items()
        if "geometry" in reason.lower() or "bbox" in reason.lower()
    )
    return {
        "parser_dropped_total": len(dropped),
        "geometry_invalid": geometry,
        "malformed_non_geometry": len(dropped) - geometry,
        "drop_reasons": dict(sorted(reasons.items())),
    }


def _matchable_rows_with_geometry_debt(parsed: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep every native-parser row visible before class-agnostic matching."""

    valid, dropped = readback_selectors.flatten_raw_rows(parsed)
    matchable = []
    for row in valid:
        bins = row.get("coord_bins_1000")
        geometry_ok = (
            isinstance(bins, list)
            and len(bins) == 4
            and all(type(value) is int for value in bins)
            and 0 <= bins[0] < bins[2] <= 999
            and 0 <= bins[1] < bins[3] <= 999
        )
        if geometry_ok:
            matchable.append(row)
        else:
            dropped.append(
                {
                    **row,
                    "status": "consumer_geometry_invalid",
                    "drop_reason": "consumer_geometry_invalid_after_native_parse",
                    "drop_code": "evaluation.geometry_invalid",
                }
            )
    return matchable, dropped


def _physical_statuses(
    *,
    image_id: int,
    predictions: Sequence[Mapping[str, Any]],
    current_matches: Sequence[Mapping[str, Any]],
    current_targets: Sequence[Mapping[str, Any]],
    exact_reviews: Mapping[tuple[int, str], Mapping[str, Any]],
) -> dict[str, Any]:
    """Retain exact review facts while deriving repeat only from this readback.

    A current-known one-to-one match is an accepted owner judgement.  Exact
    historical reviews are used only for an otherwise unmatched byte-identical
    span.  In particular, an old ``repeat`` label is a fact about the old
    generation order, not an inheritable status for a new occurrence.
    """

    current_by_owner = {str(target["owner_id"]): target for target in current_targets}
    current_by_prediction = {str(match["prediction_id"]): match for match in current_matches}
    provisional: list[dict[str, Any]] = []
    for prediction in predictions:
        prediction_id = str(prediction["prediction_id"])
        span = prediction.get("raw_span_sha256")
        decision = exact_reviews.get((image_id, str(span))) if isinstance(span, str) else None
        match = current_by_prediction.get(prediction_id)
        owner_id: str | None = None
        owner_basis: str | None = None
        status: str
        if match is not None:
            owner_id = str(match["reference_owner_id"])
            require(owner_id in current_by_owner, "current match absent from ledger")
            owner_basis = "current_known_iou_0_5"
            status = "matched_current_known_owner"
        elif decision is not None and isinstance(decision.get("owner_id"), str) and decision["owner_id"]:
            # An exact review can carry stable owner/geometry/class facts, but
            # its historical repeat label is deliberately not copied below.
            owner_id = str(decision["owner_id"])
            owner_basis = "exact_prior_review"
            status = "exact_prior_reviewed_owner"
        elif decision is not None and decision.get("physical_status") == "false":
            status = "confirmed_false_positive_exact_prior_review"
        elif decision is not None and decision.get("physical_status") == "invalid_output":
            status = "reviewed_invalid_output_exact_prior_review"
        else:
            status = "unknown_no_accepted_owner_judgment"
        exact_facts = None
        if decision is not None:
            exact_facts = {
                key: decision.get(key)
                for key in ("physical_status", "owner_id", "class", "extent")
            }
        current_owner = None
        if owner_basis == "current_known_iou_0_5":
            target = current_by_owner[owner_id]
            current_owner = {
                "owner_id": owner_id,
                "description": target.get("description"),
                "class_status": target.get("class_status"),
                "iou": match["iou"],
                "class_correct": prediction.get("description") == target.get("description"),
            }
        provisional.append(
            {
                "prediction_id": prediction_id,
                "generated_order": prediction["generated_order"],
                "raw_span_sha256": span,
                "physical_status": status,
                "physical_owner_id": owner_id,
                "physical_owner_basis": owner_basis,
                "current_owner_match": current_owner,
                "exact_prior_review_reused": decision is not None,
                "exact_prior_review_facts": exact_facts,
            }
        )

    # A repeated physical owner is determined from this run's generated order.
    # Rows with only geometric duplicate evidence never gain a physical owner.
    by_owner: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in provisional:
        if isinstance(row["physical_owner_id"], str):
            by_owner[row["physical_owner_id"]].append(row)
    for owner_rows in by_owner.values():
        owner_rows.sort(key=lambda row: int(row["generated_order"]))
        for occurrence, row in enumerate(owner_rows):
            row["owner_occurrence_in_current_readback"] = occurrence + 1
            if occurrence:
                row["physical_status"] = "recomputed_repeat_of_accepted_owner"

    counts: Counter[str] = Counter(str(row["physical_status"]) for row in provisional)
    review_class = Counter(
        str(row["exact_prior_review_facts"]["class"])
        for row in provisional
        if isinstance(row["exact_prior_review_facts"], Mapping) and row["exact_prior_review_facts"].get("class") is not None
    )
    review_extent = Counter(
        str(row["exact_prior_review_facts"]["extent"])
        for row in provisional
        if isinstance(row["exact_prior_review_facts"], Mapping) and row["exact_prior_review_facts"].get("extent") is not None
    )
    return {
        "rows": provisional,
        "matched_current_known_owner": counts["matched_current_known_owner"],
        "confirmed_fp": counts["confirmed_false_positive_exact_prior_review"],
        "physical_unknown": counts["unknown_no_accepted_owner_judgment"],
        "reviewed_physical_repeat": counts["recomputed_repeat_of_accepted_owner"],
        "reviewed_invalid_output": counts["reviewed_invalid_output_exact_prior_review"],
        "physical_status_counts": dict(sorted(counts.items())),
        "exact_prior_review_class_counts": dict(sorted(review_class.items())),
        "exact_prior_review_extent_counts": dict(sorted(review_extent.items())),
    }


def _outside_coco80(predictions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "prediction_id": row["prediction_id"],
            "generated_order": row["generated_order"],
            "description": row.get("description"),
        }
        for row in predictions
        if row.get("description") not in COCO_80_CLASS_NAMES
    ]


LEGACY_ADMISSION_SCHEMA = SCHEMA + ".legacy_readback_admission"


def _require_natural_rows(
    rows: Sequence[Mapping[str, Any]], *, routes: Mapping[int, Mapping[str, Any]], policy: Mapping[str, Any]
) -> None:
    """Validate saved legacy rows against their own bound route and policy."""

    expected_ids = set(routes)
    require(len(rows) == len(expected_ids) and {int(row.get("image_id", -1)) for row in rows} == expected_ids, "legacy readback image cohort")
    require(
        policy == {
            "assistant_token_cap": CAP,
            "empty_assistant_prefix": True,
            "repetition_penalty": 1.0,
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
        },
        "legacy natural decode policy",
    )
    for row in rows:
        image_id = int(row["image_id"])
        route = routes[image_id]
        ids = row.get("generated_token_ids")
        require(isinstance(ids, list) and ids and len(ids) <= CAP, "legacy generated token budget")
        require(row.get("generated_token_ids_sha256") == digest(ids), "legacy generated token hash")
        require(row.get("empty_assistant_prefix") is True, "legacy empty assistant prefix")
        require(row.get("prompt_token_ids") == route.get("prompt_token_ids"), "legacy prompt identity")
        identity = route.get("image_identity")
        if not isinstance(identity, Mapping):
            plan = route.get("case", {}).get("image_plan", {})
            identity = {
                "executed_media_sha256": plan.get("executed_media_sha256"),
                "observed_image_grid_thw": plan.get("observed_image_grid_thw"),
            }
        require(
            row.get("executed_media_sha256") == identity.get("executed_media_sha256")
            and row.get("observed_image_grid_thw") == identity.get("observed_image_grid_thw"),
            "legacy media/grid identity",
        )
        stop = row.get("decode_stop_reason")
        require(
            (stop == "im_end" and ids[-1] == 151645 and 151645 not in ids[:-1])
            or (stop == "length" and len(ids) == CAP and 151645 not in ids),
            "legacy readback stop contract",
        )


def admit_legacy_readback(*, kind: str, readback_path: Path) -> dict[str, Any]:
    """Bind one known historical source without treating it as a dual-start arm.

    This is intentionally narrow: it exists to replay retained raw evidence,
    while every new dual-start score goes through ``dual_start``'s collection
    admission instead.
    """

    readback_path = readback_path.resolve(strict=True)
    readback = read(readback_path)
    if kind == "fourth-fit-step256":
        require(
            readback.get("schema") == "training_set_completion.masked_coherent_route.native_readback.v1",
            "fourth-fit legacy readback schema",
        )
        manifest_path = _checked_binding(readback.get("manifest"), name="fourth-fit legacy manifest")
        manifest = read(manifest_path)
        routes = {int(route["image_id"]): route for route in manifest.get("routes", [])}
        require(len(routes) == 11, "fourth-fit legacy routes")
        _require_natural_rows(readback.get("rows", []), routes=routes, policy=readback.get("policy", {}))
        adapter = readback.get("adapter")
        require(isinstance(adapter, Mapping) and isinstance(adapter.get("fingerprint"), str) and adapter.get("fingerprint"), "fourth-fit adapter fingerprint")
        loaded = readback.get("loaded_model", {}).get("model_identity", {}).get("adapter", {})
        require(
            isinstance(loaded, Mapping) and loaded.get("adapter_path") == adapter.get("root") and loaded.get("merged_adapters", []) == [],
            "fourth-fit loaded adapter identity",
        )
        require(readback.get("recovery", {}).get("checkpoint_step") == 256, "fourth-fit legacy checkpoint")
        return {
            "schema": LEGACY_ADMISSION_SCHEMA,
            "status": "source_admitted_historical_diagnostic",
            "legacy_kind": kind,
            "disposition": "Historical fourth-fit step256 evidence only; it is not a dual-start arm baseline or endpoint.",
            "readback": binding(readback_path),
            "sources": {"manifest": binding(manifest_path), "producer": binding(Path(__file__))},
            "conditioning": {"policy": readback["policy"], "route_count": len(routes), "adapter_fingerprint": adapter["fingerprint"]},
        }
    if kind == "intervening-n16":
        require(readback.get("schema") == "training_set_completion.anchor_readback_projection.v1", "N16 projection schema")
        raw_path = _checked_binding(readback.get("source"), name="N16 projection raw source")
        raw_rows = read_jsonl(raw_path)
        manifest_path = raw_path.parent / "manifest.json"
        manifest = read(manifest_path)
        require(manifest.get("schema") == "training_set_completion.stage01_acquisition.v1", "N16 source manifest")
        require(all(row.get("manifest_sha256") == manifest.get("content_sha256") for row in raw_rows), "N16 raw manifest identity")
        policy = manifest.get("policy")
        require(
            isinstance(policy, Mapping)
            and policy.get("assistant_token_cap") == CAP
            and policy.get("empty_assistant_prefix") is True
            and policy.get("repetition_penalty") == 1.0
            and policy.get("temperatures") == [0.0, 0.1, 0.3, 0.7]
            and policy.get("top_k") == 0
            and policy.get("top_p") == 1.0,
            "N16 source natural policy",
        )
        routes = {int(route["image_id"]): route for route in manifest.get("records", [])}
        require(len(routes) == 11, "N16 source routes")
        projected_rows = readback.get("rows", [])
        matched_source_rows = []
        for projected in projected_rows:
            candidates = [
                raw
                for raw in raw_rows
                if raw.get("image_id") == projected.get("image_id")
                and raw.get("generated_token_ids_sha256") == projected.get("generated_token_ids_sha256")
            ]
            require(len(candidates) == 1, "N16 projection source identity")
            raw = candidates[0]
            matched_source_rows.append(raw)
            for field in (
                "decode_stop_reason",
                "empty_assistant_prefix",
                "executed_media_sha256",
                "generated_token_ids",
                "generated_token_ids_sha256",
                "image_id",
                "observed_image_grid_thw",
                "prompt_token_ids",
                "raw_decode_text",
            ):
                require(projected.get(field) == raw.get(field), f"N16 projection field: {field}")
            require(raw.get("request", {}).get("kind") == "greedy" and raw["request"].get("temperature") == 0.0, "N16 greedy request")
        _require_natural_rows(
            projected_rows,
            routes=routes,
            policy={
                "assistant_token_cap": policy["assistant_token_cap"],
                "empty_assistant_prefix": policy["empty_assistant_prefix"],
                "repetition_penalty": policy["repetition_penalty"],
                "temperature": 0.0,
                "top_k": policy["top_k"],
                "top_p": policy["top_p"],
            },
        )
        receipt_values = {
            json.dumps(raw["model_receipt"], sort_keys=True): raw["model_receipt"]
            for raw in matched_source_rows
        }
        model_receipt_paths = [
            _checked_binding(receipt, name="N16 model receipt")
            for _, receipt in sorted(receipt_values.items())
        ]
        adapter_paths = {
            read(path).get("model_identity", {}).get("adapter", {}).get("adapter_path")
            for path in model_receipt_paths
        }
        require(
            len(adapter_paths) == 1
            and isinstance(next(iter(adapter_paths)), str)
            and next(iter(adapter_paths)).endswith("/full-fixedP-N16-v2/adapter"),
            "N16 intervening adapter identity",
        )
        adapter_path = next(iter(adapter_paths))
        return {
            "schema": LEGACY_ADMISSION_SCHEMA,
            "status": "source_admitted_historical_diagnostic",
            "legacy_kind": kind,
            "disposition": "Intervening N16 diagnostic only; this source is not original step2444 and must never serve as dual-start arm B.",
            "readback": binding(readback_path),
            "sources": {
                "raw_rows": binding(raw_path),
                "manifest": binding(manifest_path),
                "model_receipts": [binding(path) for path in model_receipt_paths],
                "producer": binding(Path(__file__)),
            },
            "conditioning": {
                "policy": {"assistant_token_cap": CAP, "empty_assistant_prefix": True, "repetition_penalty": 1.0, "temperature": 0.0, "top_k": 0, "top_p": 1.0},
                "route_count": len(routes),
                "adapter_path": adapter_path,
            },
        }
    raise ValueError(f"unsupported legacy kind: {kind}")


def _load_legacy_admission(path: Path, *, readback_path: Path) -> dict[str, Any]:
    admission_path = path.resolve(strict=True)
    admission = read(admission_path)
    require(admission.get("schema") == LEGACY_ADMISSION_SCHEMA, "legacy admission schema")
    require(admission.get("status") == "source_admitted_historical_diagnostic", "legacy admission status")
    require(admission.get("readback") == binding(readback_path), "legacy admission readback identity")
    _checked_binding(admission.get("sources", {}).get("producer"), name="legacy admission producer")
    return {"legacy_admission": binding(admission_path), "legacy_kind": admission["legacy_kind"], "disposition": admission["disposition"]}


def score_readback(
    *,
    preparation_path: Path,
    label: str,
    readback_path: Path | None = None,
    legacy_admission_path: Path | None = None,
    runtime_result_path: Path | None = None,
    arm: str | None = None,
    step: int | None = None,
) -> dict[str, Any]:
    """Score one literal saved readback against all frozen ledgers."""

    preparation_path = preparation_path.resolve(strict=True)
    preparation = read(preparation_path)
    require(
        (readback_path is None) != (runtime_result_path is None),
        "provide exactly one of a legacy readback or dual-start runtime result",
    )
    if runtime_result_path is not None:
        require(legacy_admission_path is None, "dual-start runtime result cannot use legacy admission")
        require(isinstance(arm, str) and arm in {"A", "B"} and type(step) is int, "dual-start arm and step")
        from probes.training_set_completion import dual_start

        runtime_result_path = runtime_result_path.resolve(strict=True)
        rows, collection = dual_start.load_admitted_readback_rows(runtime_result_path, arm=arm, step=step)
        readback = {"rows": rows}
        readback_source = {
            "runtime_collection": binding(runtime_result_path),
            "trial": collection["trial"],
            "arm": arm,
            "step": step,
            "admission": "probes.training_set_completion.dual_start.load_admitted_readback_rows",
        }
    else:
        require(readback_path is not None and legacy_admission_path is not None, "legacy scoring needs explicit legacy admission")
        readback_path = readback_path.resolve(strict=True)
        readback = read(readback_path)
        readback_source = {
            "readback": binding(readback_path),
            **_load_legacy_admission(legacy_admission_path, readback_path=readback_path),
        }
    require(preparation.get("schema") == PREPARATION_SCHEMA, "evaluation preparation schema")
    require(isinstance(label, str) and label, "evaluation label")
    sources = preparation["sources"]
    # The scored ledger is self-contained, but its source identities remain a
    # hard admission boundary for every later endpoint readback.
    _checked_binding(sources["dual_start_preparation"], name="dual-start preparation")
    _checked_binding(sources["teacher_bank"], name="teacher bank")
    _checked_binding(sources["current_annotations"], name="current annotations")
    _checked_binding(sources["first_fit_preparation"], name="first-fit preparation")
    acquisition_path = _checked_binding(sources["acquisition_manifest"], name="acquisition")
    acquisition = read(acquisition_path)
    records = {int(row["image_id"]): row for row in acquisition["records"]}
    require(set(records) == set(preparation["image_ids"]), "acquisition image identity")
    tokenizer = readback_selectors._load_tokenizer(preparation["tokenizer_root"])
    rows = readback.get("rows")
    require(isinstance(rows, list) and len(rows) == 11, "readback eleven rows")
    readback_selectors.validate_unique_image_rows(rows)
    by_id = {int(row["image_id"]): row for row in rows}
    require(set(by_id) == set(preparation["image_ids"]), "readback image identity")
    exact_reviews, review_receipt = _load_exact_review_map(preparation)
    ledgers = preparation["ledgers"]
    ledger_by_image = {name: _by_image(rows) for name, rows in ledgers.items()}
    per_image = []
    aggregate: dict[str, Counter[str]] = {name: Counter() for name in ledgers}
    aggregate_raw: Counter[str] = Counter()
    all_valid = 0
    scoped_tp = 0
    for image_id in preparation["image_ids"]:
        saved = by_id[image_id]
        ids = saved.get("generated_token_ids")
        require(isinstance(ids, list) and all(type(item) is int and item >= 0 for item in ids), f"saved ids: {image_id}")
        require(saved.get("generated_token_ids_sha256") == digest(ids), f"saved ID hash: {image_id}")
        text = saved.get("raw_decode_text")
        require(isinstance(text, str), f"saved text: {image_id}")
        require(
            tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) == text,
            f"saved decode identity: {image_id}",
        )
        from probes.source_rweak_row_cross.run import native_record

        parsed = native_record(text, records[image_id]["case"], records[image_id]["golden"], saved.get("decode_stop_reason", "unknown"))
        valid, dropped = _matchable_rows_with_geometry_debt(parsed)
        ledger_rows = {name: _ledger_image(targets.get(image_id, []), valid, threshold=IOU_PRIMARY) for name, targets in ledger_by_image.items()}
        diagnostic = {name: _ledger_image(targets.get(image_id, []), valid, threshold=IOU_DIAGNOSTIC) for name, targets in ledger_by_image.items()}
        scoped = ledger_rows["scoped218"]
        all_valid += len(valid)
        scoped_tp += scoped["matched_count"]
        for name, result in ledger_rows.items():
            aggregate[name].update(
                target_count=result["target_count"],
                matched_count=result["matched_count"],
                fn_count=result["fn_count"],
                class_correct_count=result["class_correct_count"],
                class_wrong_count=result["class_wrong_count"],
                class_unknown_count=result["class_unknown_count"],
            )
        raw = _raw_debt(dropped)
        physical = _physical_statuses(
            image_id=image_id,
            predictions=valid,
            current_matches=ledger_rows["current-known248"]["matches"],
            current_targets=ledger_by_image["current-known248"].get(image_id, []),
            exact_reviews=exact_reviews,
        )
        termination = readback_selectors.termination_metrics(len(ids), ids, saved.get("decode_stop_reason"), cap=CAP)
        duplicate_candidates = readback_selectors.pairwise_iou95(valid)
        outside = _outside_coco80(valid)
        aggregate_raw.update(
            {
                key: value
                for key, value in raw.items()
                if type(value) is int
            }
        )
        aggregate_raw.update(
            valid_prediction_count=len(valid),
            eos_debt=int(termination["eos_debt"]),
            cap_debt=int(termination["cap_debt"]),
            confirmed_fp=physical["confirmed_fp"],
            physical_unknown=physical["physical_unknown"],
            matched_current_known_owner=physical["matched_current_known_owner"],
            reviewed_physical_repeat=physical["reviewed_physical_repeat"],
            reviewed_invalid_output=physical["reviewed_invalid_output"],
            duplicate_candidate_pairs=len(duplicate_candidates),
            outside_literal_coco80=len(outside),
        )
        per_image.append(
            {
                "image_id": image_id,
                "route_id": saved.get("route_id"),
                "saved_readback": {"generated_token_ids_sha256": saved["generated_token_ids_sha256"], "decode_stop_reason": saved.get("decode_stop_reason")},
                "raw": {"valid_prediction_count": len(valid), **raw, **termination},
                "ledgers_iou_0_5": ledger_rows,
                "ledgers_iou_0_8_diagnostic": diagnostic,
                "duplicate_candidates_iou_gt_0_95": duplicate_candidates,
                "duplicate_candidate_note": "candidate overlap only; it is not a physical-repeat ruling without exact review evidence.",
                "physical": physical,
                "outside_literal_coco80_protocol_violations": outside,
            }
        )
    ledger_aggregate = {}
    for name, values in aggregate.items():
        target_count, matched_count = values["target_count"], values["matched_count"]
        ledger_aggregate[name] = {
            **dict(values),
            "fn_rate": (values["fn_count"] / target_count) if target_count else 0.0,
            "covered_owner_ids": sorted(
                owner
                for row in per_image
                for owner in row["ledgers_iou_0_5"][name]["covered_owner_ids"]
            ),
            "missing_owner_ids": sorted(
                owner
                for row in per_image
                for owner in row["ledgers_iou_0_5"][name]["missing_owner_ids"]
            ),
        }
        require(target_count == {"scoped218": 218, "historical232": 232, "current-known248": 248}[name], f"aggregate {name} denominator")
    annotation_unmatched = all_valid - scoped_tp
    annotation_precision = scoped_tp / all_valid if all_valid else 0.0
    annotation_f1 = (2 * scoped_tp / (218 + all_valid)) if 218 + all_valid else 0.0
    return {
        "schema": SCHEMA,
        "status": "saved_readback_scored_not_model_selection",
        "label": label,
        "sources": {
            "preparation": binding(preparation_path),
            "readback_admission": readback_source,
            "review_reuse": review_receipt,
            "producer": binding(Path(__file__)),
        },
        "matching": preparation["matching_contract"],
        "metric_contract": preparation["metric_contract"],
        "per_image": per_image,
        "aggregate": {
            "ledgers_iou_0_5": ledger_aggregate,
            "annotation_relative_micro_scoped218": {
                "tp": scoped_tp,
                "prediction_denominator_all_valid_parsed_rows": all_valid,
                "annotation_unmatched_prediction_count": annotation_unmatched,
                "fn_scoped218": 218 - scoped_tp,
                "precision": annotation_precision,
                "f1": annotation_f1,
                "meaning": "annotation-relative only; annotation-unmatched is not a physical false positive",
            },
            "raw_and_physical": dict(aggregate_raw),
        },
        "disposition": "No scalar composite and no model choice are produced. Compare each arm initial baseline with its endpoint, then compare A versus B across the complete ledger.",
    }


def compare(*, baseline: Mapping[str, Any], endpoint: Mapping[str, Any], label: str) -> dict[str, Any]:
    """Report per-ledger retained/gained/lost ownership without a scalar ranking."""

    require(baseline.get("schema") == endpoint.get("schema") == SCHEMA, "comparison schema")
    require(
        baseline.get("sources", {}).get("preparation") == endpoint.get("sources", {}).get("preparation"),
        "comparison evaluation preparation differs",
    )
    require(baseline.get("matching") == endpoint.get("matching"), "comparison matching contract differs")
    require(baseline.get("metric_contract") == endpoint.get("metric_contract"), "comparison metric contract differs")

    def coverage(result: Mapping[str, Any], ledger: str) -> dict[int, set[str]]:
        rows = result.get("per_image")
        require(isinstance(rows, list), "comparison per-image rows")
        values: dict[int, set[str]] = {}
        for row in rows:
            require(isinstance(row, Mapping) and type(row.get("image_id")) is int, "comparison image identity")
            image_id = int(row["image_id"])
            require(image_id not in values, "comparison duplicate image")
            ledgers = row.get("ledgers_iou_0_5")
            require(isinstance(ledgers, Mapping) and isinstance(ledgers.get(ledger), Mapping), "comparison ledger missing")
            owners = ledgers[ledger].get("covered_owner_ids")
            require(isinstance(owners, list) and all(isinstance(owner, str) and owner for owner in owners), "comparison owner identities")
            require(len(owners) == len(set(owners)), "comparison duplicate owner within image")
            values[image_id] = set(owners)
        return values

    per_ledger = {}
    per_image: dict[int, dict[str, Any]] = {}
    for name in ("scoped218", "historical232", "current-known248"):
        before_by_image, after_by_image = coverage(baseline, name), coverage(endpoint, name)
        require(set(before_by_image) == set(after_by_image), "comparison image cohort differs")
        before = {(image_id, owner) for image_id, owners in before_by_image.items() for owner in owners}
        after = {(image_id, owner) for image_id, owners in after_by_image.items() for owner in owners}
        per_ledger[name] = {
            "retained": [{"image_id": image_id, "owner_id": owner} for image_id, owner in sorted(before & after)],
            "gained": [{"image_id": image_id, "owner_id": owner} for image_id, owner in sorted(after - before)],
            "lost": [{"image_id": image_id, "owner_id": owner} for image_id, owner in sorted(before - after)],
            "retained_count": len(before & after),
            "gained_count": len(after - before),
            "lost_count": len(before - after),
        }
        for image_id in sorted(before_by_image):
            item = per_image.setdefault(image_id, {"image_id": image_id, "ledgers_iou_0_5": {}})
            image_before, image_after = before_by_image[image_id], after_by_image[image_id]
            item["ledgers_iou_0_5"][name] = {
                "retained": sorted(image_before & image_after),
                "gained": sorted(image_after - image_before),
                "lost": sorted(image_before - image_after),
                "retained_count": len(image_before & image_after),
                "gained_count": len(image_after - image_before),
                "lost_count": len(image_before - image_after),
            }
    return {
        "schema": SCHEMA + ".comparison",
        "status": "paired_saved_readback_comparison",
        "label": label,
        "baseline": baseline["sources"]["readback_admission"],
        "endpoint": endpoint["sources"]["readback_admission"],
        "per_ledger": per_ledger,
        "per_image": [per_image[image_id] for image_id in sorted(per_image)],
        "disposition": "No scalar composite or automatic winner; root evaluates the complete ledger.",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--preparation", type=Path, default=PREPARATION)
    prepare.add_argument("--teacher-bank", type=Path, default=TEACHER_BANK)
    prepare.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    prepare.add_argument("--review-index", type=Path, default=REVIEW_INDEX)
    prepare.add_argument("--review-results", type=Path, default=REVIEW_RESULTS)
    prepare.add_argument("--output", type=Path, default=OUTPUT)
    prepare.add_argument("--artifact", default="preparation.json", help="new filename inside --output")
    score = commands.add_parser("score")
    score.add_argument("--preparation", type=Path, default=OUTPUT / "preparation.json")
    score.add_argument("--readback", type=Path)
    score.add_argument("--legacy-admission", type=Path)
    score.add_argument("--runtime-result", type=Path)
    score.add_argument("--arm", choices=("A", "B"))
    score.add_argument("--step", type=int)
    score.add_argument("--label", required=True)
    score.add_argument("--output", type=Path, required=True)
    legacy = commands.add_parser("admit-legacy")
    legacy.add_argument("--kind", choices=("fourth-fit-step256", "intervening-n16"), required=True)
    legacy.add_argument("--readback", type=Path, required=True)
    legacy.add_argument("--output", type=Path, required=True)
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("--baseline", type=Path, required=True)
    compare_parser.add_argument("--endpoint", type=Path, required=True)
    compare_parser.add_argument("--label", required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        value = build_preparation(
            preparation_path=args.preparation,
            teacher_bank_path=args.teacher_bank,
            annotations_path=args.annotations,
            output=args.output,
            review_index_path=args.review_index,
            review_results_path=args.review_results,
            artifact_name=args.artifact,
        )
        print(json.dumps({"output": str((args.output / args.artifact).resolve()), "ledgers": {name: len(rows) for name, rows in value["ledgers"].items()}}, sort_keys=True))
        return 0
    if args.command == "admit-legacy":
        value = admit_legacy_readback(kind=args.kind, readback_path=args.readback)
    elif args.command == "score":
        value = score_readback(
            preparation_path=args.preparation,
            readback_path=args.readback,
            legacy_admission_path=args.legacy_admission,
            runtime_result_path=args.runtime_result,
            arm=args.arm,
            step=args.step,
            label=args.label,
        )
    else:
        value = compare(baseline=read(args.baseline), endpoint=read(args.endpoint), label=args.label)
    publish(args.output, value)
    print(json.dumps({"output": str(args.output.resolve()), "status": value["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
