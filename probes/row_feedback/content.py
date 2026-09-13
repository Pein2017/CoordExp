"""Frozen exposed-case packet and receipts for feedback-content interventions.

This module does not load a model.  It binds reviewed h/c/w evidence to the
runtime owner's feedback-source override API and classifies only mechanically
identifiable outcomes.  Physical owner interpretation remains a review step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


BOX_END = 151649
VISIBLE_BUDGET = 3084

ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-12-native-owner-scale-and-state"
)
INPUTS = ROOT / "scale/training/preparation/inputs-v2.json"
LEDGER = ROOT / "evaluation/candidate-trusted-target-ledger-v5.json"
ACQUISITION = ROOT / "scale/acquisition-full-v2.json"
SEP12_SELECTION = ROOT / "evaluation/selection.json"
SEP13_CONFIRMATION = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-13-owner-successor-scale-throughput/evaluation/confirmation-selection.json"
)
SUPPLY_POOL = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-13-owner-successor-scale-throughput/supply/pool.json"
)
SELECTION_REMAINDER = ROOT / "scale/preparation/selection-v2-remainder.json"
SUPERVISION_BANK = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-13-row-feedback-pilot/data-v2/supervision-bank.json"
)
OLDER_HISTORY_PRODUCER = "024e46a512491b15d8715218c9fe7970707e7449f8b354b40ae37122c2c7ac9b"
OLDER_HISTORY_PRODUCER_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-10-selective-owner-learning-autonomous/positive7-support50-81/training/adapter"
)
N16_FIT_ANCHOR = "a7dcb56ea71ee8ab37a944778b7947c78ca22dfd321a0dcc59dde5227acecc80"

CASE_SPECS = (
    ("cup-210457-672025", 210457, "672025"),
    ("spoon-219546-708465", 219546, "708465"),
    ("donut-417044-1079494", 417044, "1079494"),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _one(rows: Sequence[Mapping[str, Any]], *, field: str, value: str) -> Mapping[str, Any]:
    matches = [row for row in rows if str(row.get(field)) == value]
    require(len(matches) == 1, f"expected exactly one {field}={value}, got {len(matches)}")
    return matches[0]


def _top_level_id_set(document: Mapping[str, Any], key: str) -> set[int]:
    values = document.get(key)
    require(isinstance(values, list), f"selection has no top-level {key}")
    return {int(value) for value in values}


def _source(path: Path, role: str) -> dict[str, Any]:
    require(path.is_file(), f"missing source: {path}")
    return {"path": str(path), "role": role, "sha256": _sha256_file(path)}


def _target_summary(target: Mapping[str, Any]) -> dict[str, Any]:
    prediction = target["prediction"]
    scoring = target["scoring_object"]
    return {
        "category": prediction["description"],
        "coord_bins": list(prediction["coord_bins"]),
        "owner_id": str(scoring["object_id"]),
        "pixel_bbox": list(target["target_bbox_pixels"]),
        "token_ids": list(target["token_ids"]),
        "token_ids_sha256": target["token_ids_sha256"],
    }


def _case(
    case_id: str,
    image_id: int,
    owner_id: str,
    *,
    positive_rows: Sequence[Mapping[str, Any]],
    conditional_rows: Sequence[Mapping[str, Any]],
    ledger_rows: Sequence[Mapping[str, Any]],
    acquisition_rows: Sequence[Mapping[str, Any]],
    bank_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    record_base = f"coco2017_train_{image_id:012d}:{owner_id}"
    positive = _one(positive_rows, field="record_id", value=f"{record_base}:c")
    conditional = _one(conditional_rows, field="record_id", value=f"{record_base}:w_kl")
    reviewed = _one(ledger_rows, field="record_id", value=f"{record_base}:c")
    acquired = _one(acquisition_rows, field="job_id", value=record_base)
    bank = _one(bank_rows, field="case_id", value=record_base)

    review = reviewed["final_review"]
    decision = review["decision"]
    require(review["status"] == "accept", f"{case_id}: target is not accepted")
    require(decision["c_single_owner_absent_from_h"] is True, f"{case_id}: c owner not established")
    require(decision["w_single_owner_nonduplicate"] is True, f"{case_id}: w owner not established")

    h = list(positive["prefix_token_ids"])
    c = list(positive["target_token_ids"])
    hc = list(conditional["prefix_token_ids"])
    w = list(conditional["target_token_ids"])
    require(h + c == hc, f"{case_id}: positive h+c does not equal conditional prefix")
    require(c[-1] == w[-1] == BOX_END, f"{case_id}: c/w must close at BOX_END")
    require(c.count(BOX_END) == w.count(BOX_END) == 1, f"{case_id}: c/w must be one row")
    require(reviewed["image_id"] == image_id, f"{case_id}: ledger image mismatch")
    require(str(reviewed["owner_id"]) == owner_id, f"{case_id}: ledger owner mismatch")
    require(acquired["local_w"]["w_token_ids"] == w, f"{case_id}: acquisition w mismatch")
    require(bank["literal_rows"]["h"]["token_ids"] == h, f"{case_id}: bank-v2 h mismatch")
    require(bank["literal_rows"]["c"]["token_ids"] == c, f"{case_id}: bank-v2 c mismatch")
    require(bank["literal_rows"]["w"]["token_ids"] == w, f"{case_id}: bank-v2 w mismatch")
    producer = bank["prefix_w_provenance"]["rollout_adapter_fingerprint"]
    require(producer == OLDER_HISTORY_PRODUCER, f"{case_id}: unexpected h/w producer")

    recipient_history = h + c
    donor_replay = recipient_history + w
    recipient_occurrence = recipient_history.count(BOX_END) - 1
    donor_occurrence = donor_replay.count(BOX_END) - 1

    return {
        "case_id": case_id,
        "image": positive["image"],
        "prompt_token_ids": list(positive["prompt_token_ids"]),
        "prompt_token_ids_sha256": positive["prompt_token_ids_sha256"],
        "case_provenance": {
            "bank_v2_record_id": bank["record_id"],
            "bank_v2_record_sha256": bank["record_sha256"],
            "older_acquisition_packet": bank["prefix_w_provenance"]["acquisition_packet"],
            "history_and_w_rollout_producer_adapter_fingerprint": producer,
            "history_role": bank["literal_rows"]["h"]["role"],
            "w_role": bank["literal_rows"]["w"]["role"],
            "c_role": bank["literal_rows"]["c"]["role"],
        },
        "recipient": {
            "reviewed_c": _target_summary(reviewed["target"]),
            "history_token_ids": recipient_history,
            "history_token_ids_sha256": _sha256_json(recipient_history),
            "feedback_boundary": {
                "box_end_occurrence": recipient_occurrence,
                "visible_boundary_index": len(recipient_history) - 1,
                "visible_token_id": BOX_END,
            },
        },
        "wrong_owner_donor": {
            "identity": "reviewed_distinct_owner_id_unavailable",
            "category": acquired["local_w"]["w_description"],
            "coord_bins": [token - 151670 for token in w[-5:-1]],
            "pixel_bbox": list(acquired["local_w"]["w_bbox_xyxy_pixels"]),
            "row_token_ids": w,
            "row_token_ids_sha256": acquired["local_w"]["w_token_ids_sha256"],
            "replay_history_token_ids": recipient_history,
            "replay_target_token_ids": w,
            "feedback_boundary": {
                "box_end_occurrence": donor_occurrence,
                "visible_boundary_index": len(donor_replay) - 1,
                "visible_token_id": BOX_END,
            },
            "review_decision_sha256": review["decision_sha256"],
            "review_reason": decision["reason"],
        },
        "same_owner_alternate_geometry": {
            "status": "unsupported_not_registered_for_this_case_evidence",
            "use": False,
        },
        "decisions": {
            "immediate": "first complete generated row after recipient boundary",
            "delayed": "second complete generated row after recipient boundary; inapplicable if absent",
            "requires_physical_review": True,
        },
    }


def build_packet(*, endpoint_selection: Path | None = None) -> dict[str, Any]:
    inputs = _load(INPUTS)
    ledger = _load(LEDGER)
    acquisition = _load(ACQUISITION)
    bank = _load(SUPERVISION_BANK)
    sep12 = _load(SEP12_SELECTION)
    confirmation = _load(SEP13_CONFIRMATION)
    pool = _load(SUPPLY_POOL)

    cases = [
        _case(
            case_id,
            image_id,
            owner_id,
            positive_rows=inputs["positive_records"],
            conditional_rows=inputs["conditional_records"],
            ledger_rows=ledger["targets"],
            acquisition_rows=acquisition["rows"],
            bank_rows=bank["records"],
        )
        for case_id, image_id, owner_id in CASE_SPECS
    ]
    image_ids = {case["image"]["image_id"] for case in cases}
    require(len(cases) <= 4 and len(image_ids) == len(cases), "content cases must be at most four distinct images")
    require(not image_ids & _top_level_id_set(sep12, "image_ids"), "case overlaps Sep12 evaluation image_ids")
    require(not image_ids & _top_level_id_set(confirmation, "image_ids"), "case overlaps Sep13 confirmation image_ids")
    require(image_ids <= _top_level_id_set(sep12, "excluded_image_ids"), "case missing from Sep12 exposed exclusions")
    require(image_ids <= _top_level_id_set(confirmation, "excluded_image_ids"), "case missing from confirmation exclusions")
    require(image_ids <= _top_level_id_set(pool, "image_ids"), "case missing from exposed supply pool")
    require(bank["schema"] == "row_feedback.supervision_bank.v2", "wrong supervision bank schema")
    require(bank["source_roles"]["prefix_w_rollout_producer"]["adapter"]["fingerprint"] == OLDER_HISTORY_PRODUCER, "bank-v2 producer drift")
    require(Path(bank["source_roles"]["prefix_w_rollout_producer"]["adapter"]["root"]).resolve() == OLDER_HISTORY_PRODUCER_ROOT.resolve(), "bank-v2 producer root drift")
    require(bank["source_roles"]["fit_anchor"]["adapter"]["fingerprint"] == N16_FIT_ANCHOR, "bank-v2 fit anchor drift")

    sources = [
        _source(INPUTS, "h_c_w_token_and_media_records"),
        _source(LEDGER, "physically_reviewed_c_w_distinct_owner_evidence"),
        _source(ACQUISITION, "reviewed_w_geometry_and_tokens"),
        _source(SEP12_SELECTION, "old_exposed_role_exclusion"),
        _source(SEP13_CONFIRMATION, "other_task_confirmation_exclusion"),
        _source(SUPPLY_POOL, "exposed_supply_role_exclusion"),
        _source(SELECTION_REMAINDER, "training_admission_superset_exclusion"),
        _source(SUPERVISION_BANK, "provenance_authority_and_literal_h_c_w_crosscheck"),
    ]
    endpoint_binding: dict[str, Any]
    if endpoint_selection is None:
        endpoint_binding = {"status": "pending_active_evidence_immutable_32_8_selection"}
        status = "candidate_pending_endpoint_binding"
    else:
        endpoint = _load(endpoint_selection)
        endpoint_ids = _top_level_id_set(endpoint, "image_ids")
        require(endpoint_ids, "endpoint selection exposes no image_ids")
        require(not image_ids & endpoint_ids, "content case overlaps immutable endpoint image_ids")
        endpoint_binding = {
            "status": "bound_disjoint",
            "path": str(endpoint_selection),
            "sha256": _sha256_file(endpoint_selection),
            "image_count": len(endpoint_ids),
        }
        status = "frozen_ready_for_runtime"

    packet: dict[str, Any] = {
        "schema": "row_feedback.content_packet.v2",
        "status": status,
        "provenance_boundary": {
            "literal_history_and_w_producer": {
                "role": "vetted older positive7-support50-81 rollout producer; these are not N16 on-policy histories",
                "adapter_fingerprint": OLDER_HISTORY_PRODUCER,
                "adapter_root": str(OLDER_HISTORY_PRODUCER_ROOT),
            },
            "fit_anchor": {
                "role": "N16 adapter from which the trained S/F slot arms start",
                "adapter_fingerprint": N16_FIT_ANCHOR,
            },
            "claim": "older-anchor h/c/w evidence is replayed through a trained N16-start F runtime; it is not evidence of native N16 history production",
        },
        "claim_boundary": {
            "positive": "a selective reviewed change supports use of the trained F feedback route",
            "dependence_only": "generic corruption or unreviewed divergence shows route dependence only",
            "inconclusive": "no effect, missing delayed row, invalid intervention, or failed review cannot disprove memory or veto the paired architecture pilot",
            "native_incapacity_claim": False,
        },
        "runtime_contract": {
            "arm": "F",
            "decode": {"temperature": 0.0, "top_p": 1.0, "repetition_penalty": 1.0},
            "visible_generated_token_budget": VISIBLE_BUDGET,
            "boundary_index_semantics": "zero-based BOX_END occurrence over supplied assistant history plus generated visible tokens; prompt excluded",
            "capture": "runtime feedback_sources returns detached final-postnorm u before fixed F normalization",
            "replacement": "override substitutes source u at exactly one boundary; runtime reapplies the fixed RMS rule",
            "fail_closed": ["missing_boundary", "duplicate_boundary", "out_of_range_boundary", "more_or_fewer_than_one_override", "S_arm_override"],
        },
        "arms": [
            {"name": "correct_f", "operation": "capture recipient source u; no override"},
            {"name": "exact_self_replay", "operation": "override recipient boundary with its captured source u"},
            {"name": "wrong_owner", "operation": "override recipient boundary with reviewed same-image same-class W source u"},
        ],
        "execution_order": [
            "replay h+c with supplied w and capture the final W feedback source",
            "generate correct_f from h+c for 3084 visible tokens and capture the final C feedback source",
            "generate exact_self_replay from h+c with captured C source at the final C boundary",
            "generate wrong_owner from h+c with captured W source at the final C boundary",
        ],
        "mechanical_acceptance": {
            "required": [
                "correct_f and exact_self_replay visible token IDs are identical",
                "each replay applies exactly one override at the registered recipient boundary",
                "captured C and W source receipts are finite and have identical shape and dtype",
                "captured C and W source SHA256 values differ",
                "all three arms retain the full 3084 visible-token allowance",
            ],
            "wrong_owner_caveat": "same image and class, reviewed as a distinct physical owner; donor owner ID is unavailable and donor comes from the later canonical W row",
            "future_completion_caveat": "the W source is captured after completing h+c+w and transplanted to the earlier C boundary; this is a diagnostic intervention, not a causal deployable state-writing rule",
        },
        "endpoint_binding": endpoint_binding,
        "sources": sources,
        "cases": cases,
    }
    packet["packet_sha256"] = _sha256_json(packet)
    return packet


def runtime_call_plan(case: Mapping[str, Any]) -> dict[str, Any]:
    """Build exact public-runtime kwargs without loading a model."""
    recipient = case["recipient"]
    donor = case["wrong_owner_donor"]
    return {
        "prompt_ids": list(case["prompt_token_ids"]),
        "recipient_boundary": dict(recipient["feedback_boundary"]),
        "donor_boundary": dict(donor["feedback_boundary"]),
        "donor_capture": {
            "history_ids": list(donor["replay_history_token_ids"]),
            "target_ids": list(donor["replay_target_token_ids"]),
            "arm": "F",
            "capture_feedback_sources": True,
        },
        "correct_f": {
            "history_ids": list(recipient["history_token_ids"]),
            "arm": "F",
            "max_visible_tokens": VISIBLE_BUDGET,
            "capture_feedback_sources": True,
        },
        "replay": {
            "history_ids": list(recipient["history_token_ids"]),
            "arm": "F",
            "max_visible_tokens": VISIBLE_BUDGET,
            "capture_feedback_sources": False,
        },
    }


def tensor_receipt(tensor: Any) -> dict[str, Any]:
    """Return a content identity without retaining a GPU tensor."""
    import torch

    require(isinstance(tensor, torch.Tensor), "feedback source must be a torch.Tensor")
    detached = tensor.detach().cpu().contiguous()
    require(detached.numel() > 0, "feedback source tensor is empty")
    finite = bool(torch.isfinite(detached.float()).all().item())
    require(finite, "feedback source tensor is non-finite")
    raw = detached.view(torch.uint8).numpy().tobytes()
    rms = math.sqrt(float(detached.float().square().mean().item()))
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "numel": detached.numel(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "rms": rms,
        "finite": finite,
    }


def classify_mechanical_outcome(
    *,
    correct_visible_ids: Sequence[int],
    self_visible_ids: Sequence[int],
    wrong_visible_ids: Sequence[int],
    self_override_count: int,
    wrong_override_count: int,
    correct_source_receipt: Mapping[str, Any],
    wrong_source_receipt: Mapping[str, Any],
    physical_review: str = "pending",
) -> dict[str, Any]:
    """Classify the intervention without converting divergence into semantics."""
    reasons: list[str] = []
    if self_override_count != 1 or wrong_override_count != 1:
        reasons.append("override_count")
    if list(correct_visible_ids) != list(self_visible_ids):
        reasons.append("self_replay_mismatch")
    if correct_source_receipt.get("shape") != wrong_source_receipt.get("shape"):
        reasons.append("source_shape")
    if correct_source_receipt.get("dtype") != wrong_source_receipt.get("dtype"):
        reasons.append("source_dtype")
    if correct_source_receipt.get("finite") is not True or wrong_source_receipt.get("finite") is not True:
        reasons.append("source_nonfinite")
    if correct_source_receipt.get("sha256") == wrong_source_receipt.get("sha256"):
        reasons.append("source_not_distinct")
    if reasons:
        return {"status": "technical_invalid", "reasons": reasons, "claim": "none"}
    if list(correct_visible_ids) == list(wrong_visible_ids):
        return {
            "status": "valid_no_visible_effect",
            "reasons": [],
            "claim": "inconclusive_for_memory_and_architecture",
        }
    if physical_review == "selective_owner_consistent":
        return {
            "status": "valid_selective_reviewed_change",
            "reasons": [],
            "claim": "supports_content_use_by_the_trained_F_route",
        }
    return {
        "status": "valid_visible_dependence",
        "reasons": [],
        "claim": "route_dependence_only_pending_or_nonspecific_physical_review",
    }


def verify_packet(packet: Mapping[str, Any]) -> None:
    expected = dict(packet)
    digest = expected.pop("packet_sha256", None)
    require(digest == _sha256_json(expected), "packet digest mismatch")
    for source in packet["sources"]:
        require(_sha256_file(Path(source["path"])) == source["sha256"], f"source drift: {source['path']}")
    require(len(packet["cases"]) <= 4, "too many content cases")
    require(packet["runtime_contract"]["visible_generated_token_budget"] == VISIBLE_BUDGET, "visible budget drift")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--endpoint-selection", type=Path)
    verify = sub.add_parser("verify")
    verify.add_argument("packet", type=Path)
    args = parser.parse_args(argv)

    if args.command == "prepare":
        packet = build_packet(endpoint_selection=args.endpoint_selection)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": packet["status"], "packet_sha256": packet["packet_sha256"], "path": str(args.output)}, sort_keys=True))
    else:
        packet = _load(args.packet)
        verify_packet(packet)
        print(json.dumps({"status": "verified", "packet_sha256": packet["packet_sha256"], "path": str(args.packet)}, sort_keys=True))


if __name__ == "__main__":
    main()
