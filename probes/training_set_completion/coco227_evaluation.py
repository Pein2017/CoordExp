"""Scoped saved-readback evaluation for the COCO-227 CE comparison.

The primary old218/new9 values are projections of one joint 227-owner
class-agnostic assignment.  A prediction therefore cannot earn credit in both
subsets.  Historical232 and current-known248 remain independently matched
diagnostic ledgers and physical review evidence remains separate from
annotation matching.
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from probes.training_set_completion import coco227_data, paired_evaluation as prior_eval, readback_selectors
from src.eval.detection_categories import COCO_80_CLASS_NAMES


ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-15-coco227-ce-normalization"
)
OLD_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-14-training-set-completion-curriculum"
)
DATA = ROOT / "data-v1/bank.json"
OLD_EVALUATION_PREPARATION = OLD_ROOT / "dual-start-evaluation-preparation-v1/preparation-v4.json"
OLD_RUNTIME_RESULT = OLD_ROOT / "dual-start-v3/readback-result.json"
OUTPUT = ROOT / "evaluation-v1"

SCHEMA = "training_set_completion.coco227_ce_normalization_evaluation.v1"
PREPARATION_SCHEMA = SCHEMA + ".preparation"
SOURCE0_ADMISSION_SCHEMA = SCHEMA + ".source0_admission"
READBACK_ADMISSION_SCHEMA = SCHEMA + ".readback_admission"
IOU_PRIMARY = prior_eval.IOU_PRIMARY
IOU_DIAGNOSTIC = prior_eval.IOU_DIAGNOSTIC
CAP = prior_eval.CAP
IMAGE_COUNT = 11
OWNER_COUNTS = {"scoped227": 227, "old218": 218, "new9": 9, "historical232": 232, "current-known248": 248}
SAVED_STEPS = (8, 16, 32, 64, 128, 256)

require = prior_eval.require
canonical = prior_eval.canonical
digest = prior_eval.digest
binding = prior_eval.binding
read = prior_eval.read
publish = prior_eval.publish
_checked_binding = prior_eval._checked_binding


def _by_image(rows: Iterable[Mapping[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    return prior_eval._by_image(rows)


def _target_from_card(image_id: int, card: Mapping[str, Any]) -> dict[str, Any]:
    fields = card.get("edited_fields")
    require(isinstance(fields, Mapping), "teacher trace fields")
    description = fields.get("selected_description")
    require(description in COCO_80_CLASS_NAMES, "teacher literal COCO-80 description")
    return prior_eval._target(
        image_id=image_id,
        owner_id=str(card.get("owner_id", "")),
        bins=fields.get("catalog_reference_coord_bins_1000", []),
        description=description,
        class_status="verified_coco80",
    )


def _scoped_ledgers(bank: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    coco227_data.validate_bank(bank)
    scoped, old, new = [], [], []
    for route in bank["routes"]:
        image_id = int(route["image_id"])
        old_ids = set(route["provenance"]["old218_owner_ids"])
        new_ids = set(route["provenance"]["new9_owner_ids"])
        require(not old_ids & new_ids, "old/new overlap")
        for card in route["provenance"]["trace"]:
            target = _target_from_card(image_id, card)
            owner_id = str(target["owner_id"])
            scoped.append(target)
            if owner_id in old_ids:
                old.append(target)
            elif owner_id in new_ids:
                new.append(target)
            else:
                raise ValueError(f"teacher owner outside old/new partition: {image_id}:{owner_id}")
    for name, rows, expected in (("scoped227", scoped, 227), ("old218", old, 218), ("new9", new, 9)):
        keys = [(int(row["image_id"]), str(row["owner_id"])) for row in rows]
        require(len(rows) == expected and len(keys) == len(set(keys)), f"{name} population")
    require(
        {(row["image_id"], row["owner_id"]) for row in scoped}
        == {(row["image_id"], row["owner_id"]) for row in old} | {(row["image_id"], row["owner_id"]) for row in new},
        "scoped227 partition union",
    )
    return {"scoped227": scoped, "old218": old, "new9": new}


def _validate_reference_ledger(name: str, rows: Sequence[Mapping[str, Any]], *, expected: int, image_ids: Sequence[int], all_images: bool) -> None:
    keys = [(int(row["image_id"]), str(row["owner_id"])) for row in rows]
    require(len(rows) == expected and len(keys) == len(set(keys)), f"{name} owner population")
    images = {image for image, _ in keys}
    require(images <= set(image_ids) and (images == set(image_ids) if all_images else bool(images)), f"{name} image cohort")


def build_preparation(
    *,
    teacher_bank_path: Path = DATA,
    old_evaluation_preparation_path: Path = OLD_EVALUATION_PREPARATION,
    output: Path = OUTPUT,
    artifact_name: str = "preparation.json",
) -> dict[str, Any]:
    """Freeze five ledgers and inherited review evidence for this trial."""

    require(Path(artifact_name).name == artifact_name, "preparation artifact name")
    artifact = output / artifact_name
    require(not artifact.exists() and not artifact.is_symlink(), f"preparation collision: {artifact}")
    teacher_bank_path = teacher_bank_path.resolve(strict=True)
    old_evaluation_preparation_path = old_evaluation_preparation_path.resolve(strict=True)
    bank, old_preparation = read(teacher_bank_path), read(old_evaluation_preparation_path)
    coco227_data.validate_bank(bank)
    require(old_preparation.get("schema") == prior_eval.PREPARATION_SCHEMA, "old evaluation preparation schema")
    scoped = _scoped_ledgers(bank)
    image_ids = [int(route["image_id"]) for route in bank["routes"]]
    old_ledgers = old_preparation.get("ledgers")
    require(isinstance(old_ledgers, Mapping), "old evaluation ledgers")
    historical = copy.deepcopy(old_ledgers.get("historical232"))
    current = copy.deepcopy(old_ledgers.get("current-known248"))
    require(isinstance(historical, list) and isinstance(current, list), "old historical/current ledgers")
    _validate_reference_ledger("scoped227", scoped["scoped227"], expected=227, image_ids=image_ids, all_images=True)
    _validate_reference_ledger("old218", scoped["old218"], expected=218, image_ids=image_ids, all_images=True)
    _validate_reference_ledger("new9", scoped["new9"], expected=9, image_ids=image_ids, all_images=False)
    _validate_reference_ledger("historical232", historical, expected=232, image_ids=image_ids, all_images=True)
    _validate_reference_ledger("current-known248", current, expected=248, image_ids=image_ids, all_images=True)
    current_by_key = {(int(row["image_id"]), str(row["owner_id"])): row for row in current}
    for row in scoped["scoped227"]:
        source = current_by_key.get((int(row["image_id"]), str(row["owner_id"])))
        require(source is not None, "teacher owner absent current-known248")
        require(source["class_status"] == "verified_coco80", "teacher owner current class")
        require(source["description"] == row["description"], "teacher/current literal description")
        require(source["reference_coord_bins_1000"] == row["reference_coord_bins_1000"], "teacher/current geometry")
    old_sources = old_preparation["sources"]
    value = {
        "schema": PREPARATION_SCHEMA,
        "status": "candidate_ready",
        "image_ids": image_ids,
        "sources": {
            "teacher_bank": binding(teacher_bank_path),
            "old_evaluation_preparation_v4": binding(old_evaluation_preparation_path),
            "current_annotations": copy.deepcopy(old_sources["current_annotations"]),
            "first_fit_preparation": copy.deepcopy(old_sources["first_fit_preparation"]),
            "acquisition_manifest": copy.deepcopy(old_sources["acquisition_manifest"]),
            "prior_review_evidence_index": copy.deepcopy(old_sources["prior_review_evidence_index"]),
            "prior_review_results": copy.deepcopy(old_sources["prior_review_results"]),
            "producer": binding(Path(__file__)),
        },
        "tokenizer_root": old_preparation["tokenizer_root"],
        "readback_contract": {
            "conditioning": "bound original image/prompt/media/grid; empty assistant prefix; natural greedy output",
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": 1.0,
            "max_new_tokens": CAP,
            "eos_token_id": 151645,
            "required_rows_per_readback": IMAGE_COUNT,
            "source0": "only prior dual-start-v3 A step256 via strict old runtime admission",
            "new_endpoints": "only a bound coco227 readback admission generated by the readback owner",
        },
        "matching_contract": {
            "primary_joint227": "class-agnostic cardinality-first one-to-one IoU >= 0.5 over scoped227; old218/new9 are a disjoint projection of this one assignment",
            "diagnostic": "class-agnostic cardinality-first one-to-one IoU >= 0.8",
            "implementation": "probes.training_set_completion.readback_selectors.one_to_one_matches -> src.eval.assignment.global_matches with shared category",
        },
        "metric_contract": {
            "primary": "joint scoped227 FN count/rate; old218 retained/lost and new9 coverage/FN are projections of the same joint assignment",
            "annotation_relative_micro": "precision/F1: TP from scoped227 joint IoU0.5 matches; prediction denominator is every valid parsed prediction in the saved readback; annotation-unmatched is not a physical false positive",
            "physical": "current-known248 IoU0.5 matches are accepted owners; exact prior review only supplies byte-identical otherwise-unmatched evidence; repeat is recomputed by current generated order",
            "outside_coco80": "exact literal membership in COCO_80_CLASS_NAMES; violation remains separate even when geometry matches",
            "clean_complete": "227 joint matches, 227 valid rows, correct scoped descriptions, natural EOS, and zero malformed/geometry-invalid/non-COCO/duplicate-candidate/physical-error burden",
            "no_scalar_composite": True,
        },
        "ledgers": {**scoped, "historical232": historical, "current-known248": current},
        "content_sha256": None,
    }
    value["content_sha256"] = digest({key: item for key, item in value.items() if key != "content_sha256"})
    publish(artifact, value)
    return value


def validate_preparation(preparation: Mapping[str, Any], *, verify_sources: bool = True) -> dict[str, Any]:
    require(preparation.get("schema") == PREPARATION_SCHEMA, "evaluation preparation schema")
    require(preparation.get("status") == "candidate_ready", "evaluation preparation status")
    require(preparation.get("content_sha256") == digest({key: item for key, item in preparation.items() if key != "content_sha256"}), "evaluation preparation content identity")
    image_ids = preparation.get("image_ids")
    require(isinstance(image_ids, list) and len(image_ids) == IMAGE_COUNT and len(set(image_ids)) == IMAGE_COUNT, "evaluation image cohort")
    sources = preparation.get("sources")
    require(isinstance(sources, Mapping), "evaluation sources")
    bank_path = _checked_binding(sources["teacher_bank"], name="teacher bank") if verify_sources else Path(sources["teacher_bank"]["path"])
    old_path = _checked_binding(sources["old_evaluation_preparation_v4"], name="old evaluation preparation") if verify_sources else Path(sources["old_evaluation_preparation_v4"]["path"])
    bank, old_preparation = read(bank_path), read(old_path)
    scoped = _scoped_ledgers(bank)
    ledgers = preparation.get("ledgers")
    require(isinstance(ledgers, Mapping), "evaluation ledgers")
    for name in ("scoped227", "old218", "new9"):
        require(ledgers.get(name) == scoped[name], f"{name} teacher ledger identity")
    require(ledgers.get("historical232") == old_preparation["ledgers"]["historical232"], "historical232 ledger identity")
    require(ledgers.get("current-known248") == old_preparation["ledgers"]["current-known248"], "current-known248 ledger identity")
    for name, all_images in (("scoped227", True), ("old218", True), ("new9", False), ("historical232", True), ("current-known248", True)):
        _validate_reference_ledger(name, ledgers[name], expected=OWNER_COUNTS[name], image_ids=image_ids, all_images=all_images)
    return {"images": IMAGE_COUNT, **{name: len(ledgers[name]) for name in OWNER_COUNTS}}


def _joint_partition(
    *,
    joint: Mapping[str, Any],
    targets: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    owner_ids: set[str],
    name: str,
) -> dict[str, Any]:
    """Project a joint assignment without recomputing a subset matcher."""

    target_by_owner = {str(row["owner_id"]): row for row in targets}
    predicted_by_id = {str(row["prediction_id"]): row for row in predictions}
    subset_targets = [row for row in targets if str(row["owner_id"]) in owner_ids]
    require(len(subset_targets) == len(owner_ids), f"{name} target partition")
    matches = [item for item in joint["matches"] if str(item["reference_owner_id"]) in owner_ids]
    matched_owners = {str(item["reference_owner_id"]) for item in matches}
    matched_predictions = {str(item["prediction_id"]) for item in matches}
    require(len(matched_owners) == len(matches) == len(matched_predictions), f"{name} joint one-to-one partition")
    class_values = [
        prior_eval._class_correct(
            target=target_by_owner[str(item["reference_owner_id"])],
            prediction=predicted_by_id[str(item["prediction_id"])],
        )
        for item in matches
    ]
    return {
        "matching_basis": "projection_of_joint_scoped227_iou_0_5_assignment",
        "target_count": len(subset_targets),
        "matched_count": len(matches),
        "fn_count": len(subset_targets) - len(matches),
        "fn_rate": (len(subset_targets) - len(matches)) / len(subset_targets) if subset_targets else 0.0,
        "covered_owner_ids": [str(item["reference_owner_id"]) for item in matches],
        "missing_owner_ids": [str(row["owner_id"]) for row in subset_targets if str(row["owner_id"]) not in matched_owners],
        "matched_prediction_ids": [str(item["prediction_id"]) for item in matches],
        "joint_annotation_unmatched_prediction_ids": list(joint["annotation_unmatched_prediction_ids"]),
        "matches": matches,
        "class_correct_count": sum(value is True for value in class_values),
        "class_wrong_count": sum(value is False for value in class_values),
        "class_unknown_count": sum(value is None for value in class_values),
    }


def _validate_saved_rows(rows: Sequence[Mapping[str, Any]], *, routes: Mapping[int, Mapping[str, Any]], arm: str, step: int) -> None:
    require(len(rows) == IMAGE_COUNT, "saved readback row count")
    readback_selectors.validate_unique_image_rows(rows)
    require({int(row.get("image_id", -1)) for row in rows} == set(routes), "saved readback image cohort")
    for row in rows:
        image_id = int(row["image_id"])
        route = routes[image_id]
        require(row.get("arm") == arm and row.get("step") == step and row.get("checkpoint_step") == step, "saved readback arm/step")
        require(row.get("empty_assistant_prefix") is True and row.get("temperature") == 0.0 and row.get("top_p") == 1.0 and row.get("top_k") == 0 and row.get("repetition_penalty") == 1.0, "saved readback decode policy")
        require(row.get("max_new_tokens") == CAP, "saved readback cap")
        ids = row.get("generated_token_ids")
        require(isinstance(ids, list) and ids and len(ids) <= CAP and all(type(item) is int and item >= 0 for item in ids), "saved generated tokens")
        require(row.get("generated_token_ids_sha256") == digest(ids), "saved generated token hash")
        require(row.get("prompt_token_ids") == route["prompt_token_ids"], "saved prompt identity")
        require(row.get("executed_media_sha256") == route["image_identity"]["executed_media_sha256"], "saved media identity")
        require(row.get("observed_image_grid_thw") == route["image_identity"]["observed_image_grid_thw"], "saved image grid identity")
        stop = row.get("decode_stop_reason")
        require((stop == "im_end" and ids[-1] == 151645 and 151645 not in ids[:-1]) or (stop == "length" and len(ids) == CAP and 151645 not in ids), "saved decode stop")


def admit_source0(*, runtime_result_path: Path = OLD_RUNTIME_RESULT) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Strictly re-admit only the accepted prior A-final256 source rows."""

    from probes.training_set_completion import dual_start

    runtime_result_path = runtime_result_path.resolve(strict=True)
    require(runtime_result_path == OLD_RUNTIME_RESULT.resolve(strict=True), "source0 must be prior dual-start-v3 runtime result")
    rows, collection = dual_start.load_admitted_readback_rows(runtime_result_path, arm="A", step=256)
    source_adapter = Path(OLD_ROOT / "dual-start-v3/A/training/checkpoints/step-00256/adapter").resolve(strict=True)
    require(all(Path(str(row.get("adapter", {}).get("root", ""))).resolve() == source_adapter for row in rows), "source0 adapter path")
    require(len({str(row["adapter"]["fingerprint"]) for row in rows}) == 1, "source0 adapter fingerprint")
    row_paths = [
        OLD_ROOT / "dual-start-v3/readback/A/new-step-256/rows" / f"image-{int(row['image_id']):012d}.json"
        for row in rows
    ]
    require(all(path.is_file() for path in row_paths), "source0 durable rows")
    admission = {
        "schema": SOURCE0_ADMISSION_SCHEMA,
        "status": "source0_admitted_common_prior_A_final256",
        "arm": "source0",
        "step": 0,
        "source_step": 256,
        "runtime_collection": binding(runtime_result_path),
        "trial": copy.deepcopy(collection["trial"]),
        "source_adapter": {"root": str(source_adapter), "fingerprint": rows[0]["adapter"]["fingerprint"]},
        "row_files": [binding(path) for path in row_paths],
        "row_hashes": [{"image_id": row["image_id"], "generated_token_ids_sha256": row["generated_token_ids_sha256"]} for row in rows],
        "admission": "probes.training_set_completion.dual_start.load_admitted_readback_rows(result_path, arm='A', step=256)",
        "disposition": "Common pre-training source only; original rows/provenance are retained without relabeling.",
        "producer": binding(Path(__file__)),
    }
    return admission, rows


def _load_source0_admission(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    value = read(path.resolve(strict=True))
    require(value.get("schema") == SOURCE0_ADMISSION_SCHEMA and value.get("status") == "source0_admitted_common_prior_A_final256", "source0 admission")
    fresh, rows = admit_source0(runtime_result_path=Path(value["runtime_collection"]["path"]))
    require(value == fresh, "source0 admission bytes no longer reproduce")
    return value, rows


def _source_for_endpoint(admission_path: Path, rows_path: Path, *, preparation: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    admission_path, rows_path = admission_path.resolve(strict=True), rows_path.resolve(strict=True)
    admission = read(admission_path)
    require(admission.get("schema") == READBACK_ADMISSION_SCHEMA and admission.get("status") == "admitted_natural_readback", "endpoint readback admission")
    arm, step = admission.get("arm"), admission.get("step")
    require(arm in {"S", "T"} and step in SAVED_STEPS, "endpoint arm/step")
    require(admission.get("rows") == binding(rows_path), "endpoint saved rows binding")
    require(admission.get("teacher_bank") == preparation["sources"]["teacher_bank"], "endpoint teacher identity")
    for name in ("trial", "training_manifest"):
        _checked_binding(admission.get(name), name=f"endpoint {name}")
    routes = {int(route["image_id"]): route for route in read(Path(preparation["sources"]["teacher_bank"]["path"]))["routes"]}
    rows_value = read(rows_path)
    rows = rows_value.get("rows") if isinstance(rows_value, Mapping) else rows_value
    require(isinstance(rows, list), "endpoint rows payload")
    _validate_saved_rows(rows, routes=routes, arm=arm, step=step)
    require(all(row.get("training_manifest") == admission["training_manifest"] and row.get("trial") == admission["trial"] for row in rows), "endpoint row provenance")
    require(all(row.get("adapter") == admission.get("adapter") for row in rows), "endpoint adapter identity")
    return admission, [dict(row) for row in rows]


def score_admitted_rows(
    *,
    preparation_path: Path,
    label: str,
    rows: Sequence[Mapping[str, Any]],
    readback_admission: Mapping[str, Any],
) -> dict[str, Any]:
    """Score already-admitted rows. Callers must use a source admission wrapper."""

    preparation_path = preparation_path.resolve(strict=True)
    preparation = read(preparation_path)
    validate_preparation(preparation)
    require(isinstance(label, str) and label, "score label")
    arm, step = readback_admission.get("arm"), readback_admission.get("step")
    require(arm in {"S", "T", "source0", "qualification"} and type(step) is int, "score arm/step")
    bank = read(Path(preparation["sources"]["teacher_bank"]["path"]))
    routes = {int(route["image_id"]): route for route in bank["routes"]}
    # source0 carries old route identities, so strict old admission has already
    # checked its prompt/media; direct validator is intentionally for new S/T only.
    if arm in {"S", "T"}:
        _validate_saved_rows(rows, routes=routes, arm=arm, step=step)
    require(len(rows) == IMAGE_COUNT, "scoring row count")
    readback_selectors.validate_unique_image_rows(rows)
    by_id = {int(row["image_id"]): row for row in rows}
    require(set(by_id) == set(preparation["image_ids"]), "scoring image cohort")
    sources = preparation["sources"]
    acquisition_path = _checked_binding(sources["acquisition_manifest"], name="acquisition")
    acquisition = read(acquisition_path)
    records = {int(item["image_id"]): item for item in acquisition["records"]}
    tokenizer = readback_selectors._load_tokenizer(preparation["tokenizer_root"])
    exact_reviews, review_receipt = prior_eval._load_exact_review_map(preparation)
    ledgers = preparation["ledgers"]
    ledger_by_image = {name: _by_image(value) for name, value in ledgers.items()}
    per_image = []
    aggregate = {name: Counter() for name in OWNER_COUNTS}
    raw_total: Counter[str] = Counter()
    all_valid = 0
    scoped_tp = 0
    for image_id in preparation["image_ids"]:
        saved = by_id[image_id]
        ids, text = saved.get("generated_token_ids"), saved.get("raw_decode_text")
        require(isinstance(ids, list) and all(type(item) is int and item >= 0 for item in ids), f"saved ids: {image_id}")
        require(saved.get("generated_token_ids_sha256") == digest(ids), f"saved token hash: {image_id}")
        require(isinstance(text, str) and tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) == text, f"saved decode identity: {image_id}")
        from src.eval.native_rows import native_detection_record as native_record

        parsed = native_record(text, records[image_id]["case"], records[image_id]["golden"], saved.get("decode_stop_reason", "unknown"))
        valid, dropped = prior_eval._matchable_rows_with_geometry_debt(parsed)
        joint = prior_eval._ledger_image(ledger_by_image["scoped227"].get(image_id, []), valid, threshold=IOU_PRIMARY)
        old_ids = {str(row["owner_id"]) for row in ledger_by_image["old218"].get(image_id, [])}
        new_ids = {str(row["owner_id"]) for row in ledger_by_image["new9"].get(image_id, [])}
        require(not old_ids & new_ids and old_ids | new_ids == {str(row["owner_id"]) for row in ledger_by_image["scoped227"].get(image_id, [])}, "joint subset partition")
        primary = {
            "scoped227": joint,
            "old218": _joint_partition(joint=joint, targets=ledger_by_image["scoped227"].get(image_id, []), predictions=valid, owner_ids=old_ids, name="old218"),
            "new9": _joint_partition(joint=joint, targets=ledger_by_image["scoped227"].get(image_id, []), predictions=valid, owner_ids=new_ids, name="new9"),
            "historical232": prior_eval._ledger_image(ledger_by_image["historical232"].get(image_id, []), valid, threshold=IOU_PRIMARY),
            "current-known248": prior_eval._ledger_image(ledger_by_image["current-known248"].get(image_id, []), valid, threshold=IOU_PRIMARY),
        }
        diagnostic = {name: prior_eval._ledger_image(targets.get(image_id, []), valid, threshold=IOU_DIAGNOSTIC) for name, targets in ledger_by_image.items() if name in {"scoped227", "historical232", "current-known248"}}
        require(primary["old218"]["matched_count"] + primary["new9"]["matched_count"] == joint["matched_count"], "joint old/new credit sum")
        all_valid += len(valid)
        scoped_tp += joint["matched_count"]
        for name, result in primary.items():
            aggregate[name].update(target_count=result["target_count"], matched_count=result["matched_count"], fn_count=result["fn_count"], class_correct_count=result["class_correct_count"], class_wrong_count=result["class_wrong_count"], class_unknown_count=result["class_unknown_count"])
        raw = prior_eval._raw_debt(dropped)
        physical = prior_eval._physical_statuses(image_id=image_id, predictions=valid, current_matches=primary["current-known248"]["matches"], current_targets=ledger_by_image["current-known248"].get(image_id, []), exact_reviews=exact_reviews)
        termination = readback_selectors.termination_metrics(len(ids), ids, saved.get("decode_stop_reason"), cap=CAP)
        duplicates = readback_selectors.pairwise_iou95(valid)
        outside = prior_eval._outside_coco80(valid)
        raw_total.update({key: value for key, value in raw.items() if type(value) is int})
        raw_total.update(valid_prediction_count=len(valid), eos_debt=int(termination["eos_debt"]), cap_debt=int(termination["cap_debt"]), confirmed_fp=physical["confirmed_fp"], physical_unknown=physical["physical_unknown"], matched_current_known_owner=physical["matched_current_known_owner"], reviewed_physical_repeat=physical["reviewed_physical_repeat"], reviewed_invalid_output=physical["reviewed_invalid_output"], duplicate_candidate_pairs=len(duplicates), outside_literal_coco80=len(outside))
        per_image.append({"image_id": image_id, "route_id": saved.get("route_id"), "saved_readback": {"generated_token_ids_sha256": saved["generated_token_ids_sha256"], "decode_stop_reason": saved.get("decode_stop_reason")}, "raw": {"valid_prediction_count": len(valid), **raw, **termination}, "ledgers_iou_0_5": primary, "ledgers_iou_0_8_diagnostic": diagnostic, "duplicate_candidates_iou_gt_0_95": duplicates, "duplicate_candidate_note": "candidate overlap remains non-adjudicated physical evidence, but makes the clean-output condition fail.", "physical": physical, "outside_literal_coco80_protocol_violations": outside})
    ledger_aggregate = {}
    for name, counter in aggregate.items():
        require(counter["target_count"] == OWNER_COUNTS[name], f"aggregate {name} denominator")
        ledger_aggregate[name] = {**dict(counter), "fn_rate": counter["fn_count"] / counter["target_count"], "covered_owners": [{"image_id": row["image_id"], "owner_id": owner} for row in per_image for owner in row["ledgers_iou_0_5"][name]["covered_owner_ids"]], "missing_owners": [{"image_id": row["image_id"], "owner_id": owner} for row in per_image for owner in row["ledgers_iou_0_5"][name]["missing_owner_ids"]]}
    annotation_unmatched = all_valid - scoped_tp
    physical_error = raw_total["confirmed_fp"] + raw_total["physical_unknown"] + raw_total["reviewed_physical_repeat"] + raw_total["reviewed_invalid_output"]
    clean_failures = {
        "joint227_fn": 227 - scoped_tp,
        "valid_prediction_count_debt": abs(all_valid - 227),
        "scoped227_class_wrong": aggregate["scoped227"]["class_wrong_count"],
        "parser_dropped": raw_total["parser_dropped_total"],
        "geometry_invalid": raw_total["geometry_invalid"],
        "outside_literal_coco80": raw_total["outside_literal_coco80"],
        "duplicate_candidate_pairs": raw_total["duplicate_candidate_pairs"],
        "physical_error": physical_error,
        "eos_debt": raw_total["eos_debt"],
        "cap_debt": raw_total["cap_debt"],
    }
    clean_failures = {key: value for key, value in clean_failures.items() if value}
    return {"schema": SCHEMA, "status": "saved_readback_scored_not_model_selection", "label": label, "arm": arm, "step": step, "sources": {"preparation": binding(preparation_path), "readback_admission": copy.deepcopy(dict(readback_admission)), "review_reuse": review_receipt, "producer": binding(Path(__file__))}, "matching": preparation["matching_contract"], "metric_contract": preparation["metric_contract"], "per_image": per_image, "aggregate": {"ledgers_iou_0_5": ledger_aggregate, "annotation_relative_micro_scoped227": {"tp": scoped_tp, "prediction_denominator_all_valid_parsed_rows": all_valid, "annotation_unmatched_prediction_count": annotation_unmatched, "fn_scoped227": 227 - scoped_tp, "precision": scoped_tp / all_valid if all_valid else 0.0, "f1": 2 * scoped_tp / (227 + all_valid) if 227 + all_valid else 0.0, "meaning": "annotation-relative only; annotation-unmatched is not a physical false positive"}, "raw_and_physical": dict(raw_total)}, "clean_completion": {"clean_complete": not clean_failures, "failure_counts": clean_failures, "definition": preparation["metric_contract"]["clean_complete"]}, "disposition": "No scalar composite or automatic arm choice. The final judgment uses joint227, old/new partition, historical/current, and complete-output errors."}


def score_qualification_rows(
    *,
    preparation_path: Path,
    label: str,
    rows: Sequence[Mapping[str, Any]],
    readback_admission: Mapping[str, Any],
) -> dict[str, Any]:
    """Score a readback-owner-admitted common-source qualification readback.

    Qualification is outside the S/T trial and must retain that identity in
    output.  The readback owner verifies live conditioning before calling this
    narrow scoring wrapper.
    """

    require(readback_admission.get("arm") == "qualification", "qualification arm")
    require(readback_admission.get("step") == 0, "qualification source step")
    require(readback_admission.get("source_kind") == "qualification_live_common_source", "qualification source kind")
    return score_admitted_rows(
        preparation_path=preparation_path,
        label=label,
        rows=rows,
        readback_admission=readback_admission,
    )


def compare(*, baseline: Mapping[str, Any], endpoint: Mapping[str, Any], label: str) -> dict[str, Any]:
    """Compare image-qualified ownership retention without a composite score."""

    require(baseline.get("schema") == endpoint.get("schema") == SCHEMA, "comparison schema")
    require(baseline.get("sources", {}).get("preparation") == endpoint.get("sources", {}).get("preparation"), "comparison preparation")
    require(baseline.get("matching") == endpoint.get("matching"), "comparison matching contract")
    require(baseline.get("metric_contract") == endpoint.get("metric_contract"), "comparison metric contract")

    def coverage(result: Mapping[str, Any], ledger: str) -> dict[int, set[str]]:
        values = {}
        for row in result.get("per_image", []):
            require(type(row.get("image_id")) is int and row["image_id"] not in values, "comparison image identity")
            owners = row.get("ledgers_iou_0_5", {}).get(ledger, {}).get("covered_owner_ids")
            require(isinstance(owners, list) and len(owners) == len(set(owners)), "comparison owner coverage")
            values[row["image_id"]] = set(owners)
        return values

    per_ledger, per_image = {}, {}
    for ledger in OWNER_COUNTS:
        before, after = coverage(baseline, ledger), coverage(endpoint, ledger)
        require(set(before) == set(after), "comparison image cohort")
        before_all = {(image, owner) for image, owners in before.items() for owner in owners}
        after_all = {(image, owner) for image, owners in after.items() for owner in owners}
        per_ledger[ledger] = {"retained": [{"image_id": image, "owner_id": owner} for image, owner in sorted(before_all & after_all)], "gained": [{"image_id": image, "owner_id": owner} for image, owner in sorted(after_all - before_all)], "lost": [{"image_id": image, "owner_id": owner} for image, owner in sorted(before_all - after_all)], "retained_count": len(before_all & after_all), "gained_count": len(after_all - before_all), "lost_count": len(before_all - after_all)}
        for image in sorted(before):
            values = per_image.setdefault(image, {"image_id": image, "ledgers_iou_0_5": {}})["ledgers_iou_0_5"]
            values[ledger] = {"retained": sorted(before[image] & after[image]), "gained": sorted(after[image] - before[image]), "lost": sorted(before[image] - after[image]), "retained_count": len(before[image] & after[image]), "gained_count": len(after[image] - before[image]), "lost_count": len(before[image] - after[image])}
    return {"schema": SCHEMA + ".comparison", "status": "paired_saved_readback_comparison", "label": label, "baseline": baseline["sources"]["readback_admission"], "endpoint": endpoint["sources"]["readback_admission"], "per_ledger": per_ledger, "per_image": [per_image[image] for image in sorted(per_image)], "disposition": "No scalar composite or automatic winner; compare the complete ledger."}


def sustained_clean_milestone(scores: Sequence[Mapping[str, Any]], *, arm: str) -> dict[str, Any]:
    """Return the first saved clean point which stays clean through step256."""

    by_step = {int(score.get("step", -1)): score for score in scores}
    require(set(by_step) == set(SAVED_STEPS) and len(by_step) == len(scores), "trajectory saved steps")
    require(all(score.get("arm") == arm for score in scores) and arm in {"S", "T"}, "trajectory arm")
    clean = {step: bool(by_step[step].get("clean_completion", {}).get("clean_complete")) for step in SAVED_STEPS}
    milestone = next((step for step in SAVED_STEPS if all(clean[later] for later in SAVED_STEPS if later >= step)), None)
    return {"schema": SCHEMA + ".sustained_clean_milestone", "status": "trajectory_observation_not_model_selection", "arm": arm, "saved_steps": list(SAVED_STEPS), "clean_by_step": clean, "earliest_sustained_clean_milestone": milestone, "disposition": "not_attained_by256" if milestone is None else "sampled_saved_checkpoint_milestone"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--teacher-bank", type=Path, default=DATA)
    prepare.add_argument("--old-evaluation-preparation", type=Path, default=OLD_EVALUATION_PREPARATION)
    prepare.add_argument("--output", type=Path, default=OUTPUT)
    prepare.add_argument("--artifact", default="preparation.json")
    admit = commands.add_parser("admit-source0")
    admit.add_argument("--runtime-result", type=Path, default=OLD_RUNTIME_RESULT)
    admit.add_argument("--output", type=Path, required=True)
    source0 = commands.add_parser("score-source0")
    source0.add_argument("--preparation", type=Path, default=OUTPUT / "preparation.json")
    source0.add_argument("--admission", type=Path, required=True)
    source0.add_argument("--label", default="source0-prior-A-final256")
    source0.add_argument("--output", type=Path, required=True)
    score = commands.add_parser("score-endpoint")
    score.add_argument("--preparation", type=Path, default=OUTPUT / "preparation.json")
    score.add_argument("--admission", type=Path, required=True)
    score.add_argument("--rows", type=Path, required=True)
    score.add_argument("--label", required=True)
    score.add_argument("--output", type=Path, required=True)
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("--baseline", type=Path, required=True)
    compare_parser.add_argument("--endpoint", type=Path, required=True)
    compare_parser.add_argument("--label", required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    trajectory = commands.add_parser("trajectory")
    trajectory.add_argument("--arm", choices=("S", "T"), required=True)
    trajectory.add_argument("--score", action="append", type=Path, required=True)
    trajectory.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        value = build_preparation(teacher_bank_path=args.teacher_bank, old_evaluation_preparation_path=args.old_evaluation_preparation, output=args.output, artifact_name=args.artifact)
        print({"output": str((args.output / args.artifact).resolve()), "status": value["status"]})
        return 0
    elif args.command == "admit-source0":
        value, _ = admit_source0(runtime_result_path=args.runtime_result)
    elif args.command == "score-source0":
        admission, rows = _load_source0_admission(args.admission)
        value = score_admitted_rows(preparation_path=args.preparation, label=args.label, rows=rows, readback_admission=admission)
    elif args.command == "score-endpoint":
        preparation = read(args.preparation.resolve(strict=True))
        admission, rows = _source_for_endpoint(args.admission, args.rows, preparation=preparation)
        value = score_admitted_rows(preparation_path=args.preparation, label=args.label, rows=rows, readback_admission=admission)
    elif args.command == "compare":
        value = compare(baseline=read(args.baseline), endpoint=read(args.endpoint), label=args.label)
    else:
        value = sustained_clean_milestone([read(path) for path in args.score], arm=args.arm)
    publish(args.output, value)
    print({"output": str(args.output.resolve()), "status": value["status"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
