"""Immutable literal supervision bank for the fixed-dose row-feedback pilot.

This module only binds already admitted native N16 evidence.  It deliberately
does not infer a new owner, repair a row, or interpret current natural output.
The runtime receives visible tokens only and owns S/F slot insertion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


BASE = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration")
N16_ROOT = BASE / "2026-09-12-native-owner-scale-and-state" / "scale"
TRAIN_INPUT = N16_ROOT / "training/preparation/inputs-v2.json"
ACQUISITION = N16_ROOT / "acquisition-full-v2.json"
FINAL_REVIEWS = N16_ROOT / "visual-review-full-v2/final-reviews.json"
TRAINING_COMPLETION = N16_ROOT / "training/preparation/training-completion-v2.json"
FIT_ANCHOR_ADAPTER = N16_ROOT / "training/full-fixedP-N16-v2/adapter"
ACQUISITION_PACKETS = (
    N16_ROOT / "preparation/acquisition-v2-remainder.json",
    N16_ROOT / "preparation/acquisition-v2-execfix.json",
)
SUPERSEDED_BANK = BASE / "2026-09-13-row-feedback-pilot/data/supervision-bank.json"
CURRENT_BANK = BASE / "2026-09-13-row-feedback-pilot/data-v2/supervision-bank.json"

ROW_START = 151646
ROW_END = 151649
SCHEMA = "row_feedback.supervision_bank.v2"
SMOKE_SCHEMA = "row_feedback.technical_smoke.v2"
PROTECTION_SCHEMA = "row_feedback.protection_records.v1"


def _json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binding(path: str | Path) -> dict[str, str]:
    resolved = Path(path).resolve()
    return {"path": str(resolved), "sha256": file_hash(resolved)}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_bound(reference: Mapping[str, str]) -> Any:
    path = Path(reference["path"])
    _require(path.is_absolute(), "source binding path must be absolute")
    _require(path.exists(), f"missing bound source: {path}")
    _require(file_hash(path) == reference["sha256"], f"source hash drift: {path}")
    return json.loads(path.read_text())


def _complete_rows(token_ids: Sequence[int], *, label: str) -> None:
    """Fail closed: the bank must preserve literal, complete visible rows."""
    _require(token_ids, f"{label} must be nonempty")
    index = 0
    while index < len(token_ids):
        _require(token_ids[index] == ROW_START, f"{label} is not row-aligned at {index}")
        try:
            end = list(token_ids).index(ROW_END, index + 1)
        except ValueError as exc:
            raise ValueError(f"{label} ends before box-end") from exc
        middle = token_ids[index + 1:end]
        _require(ROW_START not in middle and ROW_END not in middle,
                 f"{label} has nested row delimiters")
        index = end + 1


def _record_hash(record: Mapping[str, Any]) -> str:
    return _json_digest({
        "record_id": record["record_id"],
        "prompt_token_ids": record["prompt_token_ids"],
        "visible_history_token_ids": record["visible_history_token_ids"],
        "visible_target_token_ids": record["visible_target_token_ids"],
        "supervision_groups": record["supervision_groups"],
        "image": record["image"],
        "teacher": record["teacher"],
        "prefix_w_provenance": record["prefix_w_provenance"],
    })


def _validate_record(record: Mapping[str, Any]) -> None:
    h, c, w = (record["literal_rows"][key]["token_ids"] for key in ("h", "c", "w"))
    _complete_rows(h, label=f"{record['record_id']} h")
    _complete_rows(c, label=f"{record['record_id']} c")
    _complete_rows(w, label=f"{record['record_id']} w")
    _require(record["visible_history_token_ids"] == h + c,
             f"{record['record_id']} history must be literal h+c")
    _require(record["visible_target_token_ids"] == w,
             f"{record['record_id']} target must be literal post-c w")
    _require(record["teacher"]["visible_history_token_ids"] == h + c,
             f"{record['record_id']} teacher history mapping")
    _require(record["teacher"]["visible_target_token_ids"] == w,
             f"{record['record_id']} teacher target mapping")
    _require(record["teacher"]["mapping"] == "visible_target_ordinal",
             f"{record['record_id']} teacher must map visible ordinals")
    groups = record["supervision_groups"]
    _require(groups["entry_c"]["visible_history_token_ids"] == h
             and groups["entry_c"]["visible_target_token_ids"] == c,
             f"{record['record_id']} entry-c provenance")
    _require(groups["entry_c"]["native_teacher_protection"] == "forbidden_supervised_repaired_target",
             f"{record['record_id']} repaired c must not receive native KL")
    _require(groups["post_completion_w"]["visible_history_token_ids"] == h + c
             and groups["post_completion_w"]["visible_target_token_ids"] == w,
             f"{record['record_id']} post-c w provenance")
    _require(record["evidence_roles"]["c"] == "root_admitted_physical_single_owner_absent_from_h",
             f"{record['record_id']} c evidence role")
    _require(record["evidence_roles"]["w"] == "root_admitted_physical_single_owner_nonstrict_successor",
             f"{record['record_id']} w evidence role")
    _require(record["prefix_w_provenance"]["rollout_adapter_fingerprint"]
             != record["teacher"]["source_adapter"]["fingerprint"],
             f"{record['record_id']} older rollout producer conflated with N16 teacher")
    _require(record["record_sha256"] == _record_hash(record),
             f"{record['record_id']} record hash")


def validate_bank(bank: Mapping[str, Any]) -> None:
    _require(bank["schema"] == SCHEMA, "unexpected supervision-bank schema")
    _require(bank["status"] == "candidate_awaiting_root_protocol_acceptance",
             "bank status must not claim protocol acceptance")
    records = bank["records"]
    _require(8 <= len(records) <= 16, "bank must contain 8..16 transitions")
    record_ids = [record["record_id"] for record in records]
    _require(len(record_ids) == len(set(record_ids)), "duplicate supervision record ID")
    image_ids = {record["example_id"] for record in records}
    _require(len(image_ids) >= 4, "bank needs four distinct images")
    _require(bank["denominators"] == {
        "transitions": len(records), "distinct_images": len(image_ids),
        "accepted_c": len(records), "accepted_w": len(records),
        "unknown_or_unmatched_as_negative": 0,
    }, "bank denominators")
    roles = bank["source_roles"]
    producer = roles["prefix_w_rollout_producer"]
    anchor, teacher = roles["fit_anchor"], roles["native_teacher"]
    _require(producer["adapter"]["fingerprint"] != anchor["adapter"]["fingerprint"],
             "older rollout producer must differ from fit anchor")
    _require(teacher["adapter"] == anchor["adapter"],
             "native teacher must bind the frozen N16 fit-anchor identity")
    _require(len(producer["acquisition_packets"]) == 2,
             "accepted rows must bind both original acquisition packets")
    packet_by_hash = {packet["sha256"]: packet for packet in producer["acquisition_packets"]}
    for packet in packet_by_hash.values():
        loaded = _read_bound(packet)
        _require(loaded["adapter"] == producer["adapter"],
                 "original acquisition packet/producer adapter identity")
    for record in records:
        _require(record["prefix_w_provenance"]["acquisition_packet"]["sha256"] in packet_by_hash,
                 f"{record['record_id']} unknown acquisition packet")
        _validate_record(record)
    without_hash = dict(bank)
    bank_hash = without_hash.pop("bank_sha256")
    _require(bank_hash == _json_digest(without_hash), "bank hash")


def validate_protection_records(packet: Mapping[str, Any]) -> None:
    """Validate visible IDs/masks without accepting stale reference values."""
    _require(packet["schema"] == PROTECTION_SCHEMA, "unexpected protection-record schema")
    _require(packet["status"] == "fresh_n16_teacher_cache_required", "protection cache status")
    _require(packet["probabilities"] == "absent_runtime_must_generate", "stored probabilities forbidden")
    anchor = _read_bound(packet["sources"]["anchor_input"])
    manifest = _read_bound(packet["sources"]["anchor_manifest"])
    inputs = _read_bound(packet["sources"]["native_n16_training_input"])
    bank = _read_bound(packet["sources"]["supervision_bank"])
    validate_bank(bank)
    _require(anchor["manifest"] == packet["sources"]["anchor_manifest"], "anchor/manifest binding")
    selected = packet["normal_keys"]
    _require(selected == inputs["normal_keys"], "inputs/protection key identity")
    _require(len(selected) == len(set(selected)) == 54, "exact selected54 normal keys")
    source_adapter = anchor["stable50_adapter"]
    teacher = packet["fresh_n16_teacher"]
    _require(teacher == bank["source_roles"]["native_teacher"]["adapter"], "bank/N16 teacher identity")
    _require(teacher["fingerprint"] != source_adapter["fingerprint"], "stale Stable50 teacher identity")
    source_by_key = {row["key"]: row for row in manifest["normals"]["cases"]}
    _require(set(selected).issubset(source_by_key), "normal key outside anchor corpus")
    records = packet["records"]
    _require([row["key"] for row in records] == selected, "record order/key identity")
    action_total = position_total = 0
    for record in records:
        source = source_by_key[record["key"]]
        _require(record["example_id"] == source["example_id"] and record["image"] == source["image"],
                 f"image identity drift: {record['key']}")
        _require(record["prompt_token_ids"] == source["prompt_token_ids"], f"prompt drift: {record['key']}")
        _require(record["action_ids"] == source["action_ids"], f"action drift: {record['key']}")
        _require(record["kl_positions"] == source["initial_layout"]["kl_positions"],
                 f"KL mask drift: {record['key']}")
        _require(record["action_ids_sha256"] == source["action_ids_sha256"],
                 f"action hash drift: {record['key']}")
        _require(record["prompt_token_ids_sha256"] == source["prompt_token_ids_sha256"],
                 f"prompt hash drift: {record['key']}")
        _require(record["kl_positions"] and max(record["kl_positions"]) < len(record["action_ids"]),
                 f"invalid KL positions: {record['key']}")
        action_total += len(record["action_ids"])
        position_total += len(record["kl_positions"])
    _require(packet["denominators"] == {
        "records": 54, "distinct_images": 54, "visible_action_tokens": action_total,
        "protected_positions": position_total,
    } == {"records": 54, "distinct_images": 54, "visible_action_tokens": 5768,
          "protected_positions": 5759}, "selected54 denominators")
    exception = packet["mask_exception"]
    source_exception = source_by_key[exception["key"]]["initial_layout"]
    _require(exception["excluded_action_positions"] == list(range(75, 84)), "excluded9 positions")
    _require(source_exception["invalid_geometry_rows"] == [8]
             and source_exception["parser_drop_rows"][0]["reason"] == "geometry_invalid"
             and source_exception["parser_drop_rows"][0]["token_positions"] == exception["excluded_action_positions"],
             "excluded9 rationale")
    without_hash = dict(packet)
    packet_hash = without_hash.pop("protection_records_sha256")
    _require(packet_hash == _json_digest(without_hash), "protection-record hash")


def build_bank() -> dict[str, Any]:
    """Join the exact vetted older-anchor h/c/w bank without re-admission."""
    sources = {
        "native_n16_training_input": binding(TRAIN_INPUT),
        "native_n16_acquisition": binding(ACQUISITION),
        "native_n16_final_reviews": binding(FINAL_REVIEWS),
        "native_n16_training_completion": binding(TRAINING_COMPLETION),
    }
    train, acquisition, reviews, completion = (_read_bound(reference) for reference in sources.values())
    _require(train["schema"] == "parallel_owner_training.inputs.v1", "N16 input schema")
    _require(train["status"] == "prepared_no_model_execution", "N16 input status")
    _require(reviews["status"] == "lead_accepted_physical_bank_admission", "N16 physical admission")
    _require(reviews["counts"] == {"accept": 16, "neutral": 140}, "N16 final review census")
    _require(completion["input"] == sources["native_n16_training_input"], "N16 completion input binding")
    fit_anchor = completion["saved_adapter"]
    _require(fit_anchor["root"] == str(FIT_ANCHOR_ADAPTER), "unexpected N16 fit-anchor root")
    _require(Path(fit_anchor["root"]).is_dir(), "N16 fit anchor missing")
    acquisition_packets = [binding(path) for path in ACQUISITION_PACKETS]
    packet_by_hash = {packet["sha256"]: packet for packet in acquisition_packets}
    packet_payloads = {packet["sha256"]: _read_bound(packet) for packet in acquisition_packets}
    producer_adapter = packet_payloads[acquisition_packets[0]["sha256"]]["adapter"]
    _require(all(payload["adapter"] == producer_adapter for payload in packet_payloads.values()),
             "source acquisition packets disagree on rollout adapter")
    _require(producer_adapter["fingerprint"] != fit_anchor["fingerprint"],
             "source rollout producer and N16 fit anchor must differ")

    positives = {row["case_id"]: row for row in train["positive_records"]}
    conditionals = {row["case_id"]: row for row in train["conditional_records"]}
    _require(len(positives) == len(conditionals) == 16, "exact N16 package count")
    _require(set(positives) == set(conditionals), "N16 c/w case identity")
    acquired = {row["job_id"]: row for row in acquisition["rows"]}

    records: list[dict[str, Any]] = []
    for case_id, positive in positives.items():
        conditional = conditionals[case_id]
        acquired_row = acquired.get(case_id)
        decision = reviews["decisions"].get(case_id)
        _require(acquired_row is not None and decision is not None, f"missing N16 evidence for {case_id}")
        _require(decision["status"] == "accept", f"unadmitted N16 package: {case_id}")
        _require(decision["c_single_owner_absent_from_h"] is True,
                 f"unproven c physical role: {case_id}")
        _require(decision["w_single_owner_nonduplicate"] is True,
                 f"unproven w physical role: {case_id}")
        _require(acquired_row["local_w"]["status"] == "candidate_local_w",
                 f"non-immediate or duplicate w: {case_id}")
        h, c, w = positive["prefix_token_ids"], positive["target_token_ids"], conditional["target_token_ids"]
        _require(conditional["prefix_token_ids"] == h + c, f"post-c history mismatch: {case_id}")
        _require(acquired_row["local_w"]["w_token_ids"] == w, f"acquired w mismatch: {case_id}")
        source_packet = packet_by_hash.get(acquired_row["packet_sha256"])
        _require(source_packet is not None, f"unbound source packet for {case_id}")
        _require(positive["image"] == conditional["image"], f"image identity mismatch: {case_id}")
        record = {
            "record_id": f"{case_id}:h+c->w",
            "case_id": case_id,
            "example_id": positive["example_id"],
            "prompt_token_ids": positive["prompt_token_ids"],
            "prompt_token_ids_sha256": positive["prompt_token_ids_sha256"],
            "image": positive["image"],
            "literal_rows": {
                "h": {"token_ids": h, "token_ids_sha256": _json_digest(h),
                      "role": "vetted_older_anchor_rollout_history"},
                "c": {"token_ids": c, "token_ids_sha256": _json_digest(c),
                      "role": "root_admitted_physical_single_owner"},
                "w": {"token_ids": w, "token_ids_sha256": _json_digest(w),
                      "text": acquired_row["local_w"]["w_text"],
                      "role": "vetted_older_anchor_completed_post_c_successor"},
            },
            "visible_history_token_ids": h + c,
            "visible_target_token_ids": w,
            "supervision_groups": {
                "entry_c": {
                    "source_record_id": positive["record_id"],
                    "visible_history_token_ids": h,
                    "visible_target_token_ids": c,
                    "role": "literal_root_admitted_entry_c",
                    "native_teacher_protection": "forbidden_supervised_repaired_target",
                },
                "post_completion_w": {
                    "source_record_id": conditional["record_id"],
                    "visible_history_token_ids": h + c,
                    "visible_target_token_ids": w,
                    "role": "literal_root_admitted_post_completion_successor_w",
                    "native_teacher_protection": "root_recipe_unfrozen_matching_N16_only",
                },
            },
            "evidence_roles": {
                "h": "vetted_older_anchor_rollout_prefix_not_N16_on_policy_history",
                "c": "root_admitted_physical_single_owner_absent_from_h",
                "w": "root_admitted_physical_single_owner_nonstrict_successor",
                "gt_unmatched_or_unknown": "neutral_excluded_never_negative",
            },
            "teacher": {
                "source_adapter": fit_anchor,
                "source_training_completion": sources["native_n16_training_completion"],
                "visible_history_token_ids": h + c,
                "visible_target_token_ids": w,
                "mapping": "visible_target_ordinal",
                "claim_boundary": "frozen N16 protection is cross-protocol distillation over older-anchor h+c provenance, not same-physical-input parity",
            },
            "prefix_w_provenance": {
                "acquisition_packet": source_packet,
                "acquisition_row_id": acquired_row["job_id"],
                "rollout_adapter_fingerprint": producer_adapter["fingerprint"],
                "role": "literal h and local w were produced by the vetted older-anchor acquisition adapter",
            },
            "admission_evidence": {
                "decision": decision,
                "acquisition_local_w": {
                    key: acquired_row["local_w"][key]
                    for key in ("status", "strict_duplicate_threshold_exclusive", "max_any_class_prefix_iou", "w_token_ids_sha256")
                },
            },
        }
        record["record_sha256"] = _record_hash(record)
        records.append(record)

    records.sort(key=lambda row: row["record_id"])
    bank = {
        "schema": SCHEMA,
        "status": "candidate_awaiting_root_protocol_acceptance",
        "claim_boundary": "Literal vetted older-anchor histories and witnesses train a new N16-start S/F slot protocol; this is neither current-N16 on-policy history nor prior slot success or a natural-output claim.",
        "sources": sources,
        "source_roles": {
            "prefix_w_rollout_producer": {
                "role": "vetted older-anchor acquisition producer for literal h and local w",
                "adapter": producer_adapter,
                "acquisition_packets": acquisition_packets,
            },
            "fit_anchor": {
                "role": "frozen N16 adapter from which both S and F fit arms start",
                "adapter": fit_anchor,
                "training_completion": sources["native_n16_training_completion"],
            },
            "native_teacher": {
                "role": "frozen N16 visible-target distribution provider; no older Stable50 probabilities",
                "adapter": fit_anchor,
                "training_completion": sources["native_n16_training_completion"],
            },
        },
        "supersedes": {
            "artifact": binding(SUPERSEDED_BANK),
            "reason": "v1 incorrectly attributed vetted older-anchor h/w rollout provenance to N16; literal token IDs and physical admission are unchanged",
        },
        "records": records,
        "train_image_ids": sorted({record["example_id"] for record in records}),
        "denominators": {
            "transitions": len(records), "distinct_images": len({record["example_id"] for record in records}),
            "accepted_c": len(records), "accepted_w": len(records),
            "unknown_or_unmatched_as_negative": 0,
        },
        "selection": {
            "policy": "exact previously lead-admitted native N16 bank; no fill, re-ranking, or threshold relaxation",
            "maximum_transitions": 16, "minimum_transitions": 8, "minimum_distinct_images": 4,
        },
    }
    bank["bank_sha256"] = _json_digest(bank)
    validate_bank(bank)
    return bank


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def materialize_bank(output_dir: str | Path) -> dict[str, str]:
    """Publish immutable full-bank and one-record technical-smoke bindings."""
    bank = build_bank()
    output_dir = Path(output_dir).resolve()
    _require(not output_dir.exists(), f"refuse to overwrite data output: {output_dir}")
    output_dir.mkdir(parents=True)
    bank_path = output_dir / "supervision-bank.json"
    smoke_path = output_dir / "technical-smoke.json"
    receipt_path = output_dir / "materialization-receipt.json"
    _write_exclusive(bank_path, bank)
    smoke = {
        "schema": SMOKE_SCHEMA,
        "status": "admitted_technical_smoke_only",
        "claim_boundary": "One older-anchor-provenance, source-admitted record permits causal replay/gradient wiring only; it is not a fit or endpoint result.",
        "bank": binding(bank_path),
        "record": bank["records"][0],
        "denominators": {"transitions": 1, "distinct_images": 1, "scientific_fit_updates": 0},
    }
    smoke["smoke_sha256"] = _json_digest(smoke)
    _write_exclusive(smoke_path, smoke)
    receipt = {
        "schema": "row_feedback.supervision_materialization_receipt.v1",
        "status": "cpu_valid_candidate_bank_and_technical_smoke_written",
        "bank": binding(bank_path), "technical_smoke": binding(smoke_path),
        "records": len(bank["records"]), "distinct_images": len(bank["train_image_ids"]),
        "claim_boundary": bank["claim_boundary"],
    }
    _write_exclusive(receipt_path, receipt)
    return {"bank": str(bank_path), "technical_smoke": str(smoke_path), "receipt": str(receipt_path)}


def build_protection_records(bank_path: str | Path = CURRENT_BANK) -> dict[str, Any]:
    """Freeze the inherited visible normal corpus for fresh N16 references."""
    bank_reference = binding(bank_path)
    bank = _read_bound(bank_reference)
    validate_bank(bank)
    inputs = _read_bound(bank["sources"]["native_n16_training_input"])
    anchor_reference = inputs["anchor_input"]
    anchor = _read_bound(anchor_reference)
    manifest_reference = anchor["manifest"]
    manifest = _read_bound(manifest_reference)
    keys = list(inputs["normal_keys"])
    normal_by_key = {row["key"]: row for row in manifest["normals"]["cases"]}
    _require(len(keys) == 54 and set(keys).issubset(normal_by_key), "selected54 anchor normal keys")
    records = []
    for key in keys:
        source = normal_by_key[key]
        records.append({
            "key": key, "example_id": source["example_id"], "image": source["image"],
            "prompt_token_ids": source["prompt_token_ids"],
            "prompt_token_ids_sha256": source["prompt_token_ids_sha256"],
            "action_ids": source["action_ids"], "action_ids_sha256": source["action_ids_sha256"],
            "kl_positions": source["initial_layout"]["kl_positions"],
        })
    bank_images = {record["example_id"] for record in bank["records"]}
    overlap = sorted(bank_images.intersection({record["example_id"] for record in records}))
    packet = {
        "schema": PROTECTION_SCHEMA,
        "status": "fresh_n16_teacher_cache_required",
        "claim_boundary": "Frozen visible normal IDs and masks only. Fresh current-N16 reference distributions must be generated by runtime; no old Stable50 probabilities, margins, or floors are carried forward.",
        "sources": {
            "supervision_bank": bank_reference,
            "native_n16_training_input": bank["sources"]["native_n16_training_input"],
            "anchor_input": anchor_reference,
            "anchor_manifest": manifest_reference,
        },
        "fresh_n16_teacher": bank["source_roles"]["native_teacher"]["adapter"],
        "normal_keys": keys,
        "records": records,
        "probabilities": "absent_runtime_must_generate",
        "mask_exception": {
            "key": "coco2017_train_000000360573",
            "excluded_action_positions": list(range(75, 84)),
            "rationale": "original layout marks row 8 geometry-invalid and parser-dropped; preserve its nine-token exclusion",
        },
        "train_image_overlap": {
            "count": len(overlap), "example_ids": overlap,
            "policy": "reported_only; selected54 masks and IDs are unchanged",
        },
        "denominators": {
            "records": len(records), "distinct_images": len({record["example_id"] for record in records}),
            "visible_action_tokens": sum(len(record["action_ids"]) for record in records),
            "protected_positions": sum(len(record["kl_positions"]) for record in records),
        },
    }
    packet["protection_records_sha256"] = _json_digest(packet)
    validate_protection_records(packet)
    return packet


def materialize_protection_records(output_dir: str | Path, *, bank_path: str | Path = CURRENT_BANK) -> str:
    output = Path(output_dir).resolve() / "protection-records.json"
    _require(not output.exists(), f"refuse to overwrite protection records: {output}")
    packet = build_protection_records(bank_path)
    _write_exclusive(output, packet)
    return str(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--materialize-protection-records", action="store_true")
    args = parser.parse_args()
    if args.materialize_protection_records:
        print(json.dumps({"protection_records": materialize_protection_records(args.output_dir)}, sort_keys=True))
    else:
        print(json.dumps(materialize_bank(args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
