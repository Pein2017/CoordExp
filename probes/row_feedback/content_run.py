"""Run the frozen, exposed three-case feedback-content diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from probes.row_feedback import content


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _one(rows: Sequence[Mapping[str, Any]], *, field: str, value: str) -> Mapping[str, Any]:
    matches = [row for row in rows if str(row.get(field)) == value]
    require(len(matches) == 1, f"expected one {field}={value}, got {len(matches)}")
    return matches[0]


def _source_at(result: Mapping[str, Any], boundary: int) -> torch.Tensor:
    sources = result.get("feedback_sources")
    require(isinstance(sources, Mapping) and boundary in sources, f"feedback source {boundary} missing")
    source = sources[boundary]
    require(isinstance(source, torch.Tensor), f"feedback source {boundary} is not a tensor")
    return source


def _boundary_at(result: Mapping[str, Any], boundary: int) -> Mapping[str, Any]:
    rows = [row for row in result["feedback_boundaries"] if row["boundary_index"] == boundary]
    require(len(rows) == 1, f"feedback boundary {boundary} missing or duplicated")
    return rows[0]


def _override_receipt(result: Mapping[str, Any]) -> dict[str, Any]:
    applied = [
        {"boundary_index": row["boundary_index"], "visible_boundary_index": row["visible_boundary_index"]}
        for row in result["feedback_boundaries"]
        if row["override_applied"]
    ]
    return {"count": len(applied), "boundaries": applied}


def _generation_receipt(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "arm": result["arm"],
        "visible_token_ids": list(result["visible_token_ids"]),
        "text": result["text"],
        "finish_reason": result["finish_reason"],
        "eos": result["eos"],
        "cap": result["cap"],
        "visible_generated_tokens": result["visible_generated_tokens"],
        "internal_slot_count": result["internal_slot_count"],
        "physical_token_count": result["physical_token_count"],
        "model_forwards": result["model_forwards"],
        "image_forwards": result["image_forwards"],
        "slot_work": result["slot_work"],
        "feedback_boundaries": result["feedback_boundaries"],
        "timing": result["timing"],
        "decode_contract": result["decode_contract"],
    }


def _run_case(
    *,
    runtime_api: Any,
    qwen: Any,
    materialized: Mapping[str, Any],
    case: Mapping[str, Any],
) -> dict[str, Any]:
    plan = content.runtime_call_plan(case)
    recipient_boundary = int(plan["recipient_boundary"]["box_end_occurrence"])
    donor_boundary = int(plan["donor_boundary"]["box_end_occurrence"])
    recipient_visible_index = int(plan["recipient_boundary"]["visible_boundary_index"])

    # replay_visible normally retains graphs for training.  The diagnostic must
    # capture only detached inference sources, including the future W source.
    with torch.inference_mode():
        donor = runtime_api.replay_visible(
            qwen,
            materialized["inputs"],
            prompt_ids=plan["prompt_ids"],
            **plan["donor_capture"],
        )
        donor_source = _source_at(donor, donor_boundary)
        correct = runtime_api.generate_visible(
            qwen,
            materialized["inputs"],
            prompt_ids=plan["prompt_ids"],
            **plan["correct_f"],
        )
        correct_source = _source_at(correct, recipient_boundary)
        self_override = runtime_api.FeedbackSourceOverride(
            source=correct_source.detach().clone(),
            visible_boundary_index=recipient_visible_index,
        )
        wrong_override = runtime_api.FeedbackSourceOverride(
            source=donor_source.detach().clone(),
            visible_boundary_index=recipient_visible_index,
        )
        exact_self = runtime_api.generate_visible(
            qwen,
            materialized["inputs"],
            prompt_ids=plan["prompt_ids"],
            feedback_source_overrides={recipient_boundary: self_override},
            **plan["replay"],
        )
        wrong_owner = runtime_api.generate_visible(
            qwen,
            materialized["inputs"],
            prompt_ids=plan["prompt_ids"],
            feedback_source_overrides={recipient_boundary: wrong_override},
            **plan["replay"],
        )

    for name, result in (("correct_f", correct), ("exact_self_replay", exact_self), ("wrong_owner", wrong_owner)):
        require(result["decode_contract"]["max_visible_tokens"] == content.VISIBLE_BUDGET,
                f"{name} visible allowance changed")
    donor_meta = _boundary_at(donor, donor_boundary)
    correct_meta = _boundary_at(correct, recipient_boundary)
    donor_source_receipt = content.tensor_receipt(donor_source)
    correct_source_receipt = content.tensor_receipt(correct_source)
    require(donor_meta["native_source_sha256"] == donor_source_receipt["sha256"], "donor source receipt mismatch")
    require(correct_meta["native_source_sha256"] == correct_source_receipt["sha256"], "correct source receipt mismatch")

    self_receipt = _override_receipt(exact_self)
    wrong_receipt = _override_receipt(wrong_owner)
    expected_override = [{
        "boundary_index": recipient_boundary,
        "visible_boundary_index": recipient_visible_index,
    }]
    require(self_receipt["boundaries"] == expected_override, "self override site changed")
    require(wrong_receipt["boundaries"] == expected_override, "wrong-owner override site changed")
    classification = content.classify_mechanical_outcome(
        correct_visible_ids=correct["visible_token_ids"],
        self_visible_ids=exact_self["visible_token_ids"],
        wrong_visible_ids=wrong_owner["visible_token_ids"],
        self_override_count=self_receipt["count"],
        wrong_override_count=wrong_receipt["count"],
        correct_source_receipt=correct_source_receipt,
        wrong_source_receipt=donor_source_receipt,
    )
    return {
        "schema": "row_feedback.content_case_receipt.v1",
        "case_id": case["case_id"],
        "image_id": case["image"]["image_id"],
        "exposed_case": True,
        "non_gating": True,
        "future_completion_caveat": (
            "W feedback is captured after completing h+c+w and transplanted to the earlier C boundary; "
            "this is a diagnostic intervention, not a causal deployable state-writing rule."
        ),
        "registered_boundaries": {
            "recipient": plan["recipient_boundary"],
            "donor": plan["donor_boundary"],
        },
        "sources": {
            "correct_c": {"tensor": correct_source_receipt, "boundary": correct_meta},
            "wrong_owner_w": {"tensor": donor_source_receipt, "boundary": donor_meta},
        },
        "arms": {
            "correct_f": _generation_receipt(correct),
            "exact_self_replay": {
                **_generation_receipt(exact_self),
                "override": self_receipt,
                "exact_visible_equality_to_correct": (
                    exact_self["visible_token_ids"] == correct["visible_token_ids"]
                ),
            },
            "wrong_owner": {**_generation_receipt(wrong_owner), "override": wrong_receipt},
        },
        "mechanical_classification": classification,
        "semantic_review": "pending_physical_review_if_visible_outputs_diverge",
    }


def run_diagnostic(
    *,
    packet_path: Path,
    adapter_path: Path,
    output: Path,
    device: torch.device,
    runtime_api: Any | None = None,
) -> Mapping[str, Any]:
    """Execute the accepted fixed-three diagnostic and persist JSON receipts."""
    if runtime_api is None:
        from probes.row_feedback import runtime as runtime_api
    require(not output.exists(), "content diagnostic output already exists")
    packet = _load(packet_path)
    content.verify_packet(packet)
    require(packet["status"] == "frozen_ready_for_runtime", "content packet is not frozen")
    require([case["case_id"] for case in packet["cases"]] == [spec[0] for spec in content.CASE_SPECS],
            "content case set/order changed")
    bank_source = _one(
        packet["sources"], field="role", value="provenance_authority_and_literal_h_c_w_crosscheck"
    )
    bank_path = Path(bank_source["path"])
    require(_file_sha256(bank_path) == bank_source["sha256"], "bound supervision bank changed")
    bank = _load(bank_path)

    output.mkdir(parents=True)
    started = time.monotonic()
    phase = "load"
    try:
        qwen, frontend, config, identity = runtime_api.load_feedback_policy(
            adapter_path=adapter_path,
            device=device,
        )
        cases: list[dict[str, Any]] = []
        for index, case in enumerate(packet["cases"]):
            phase = f"materialize:{case['case_id']}"
            record = _one(
                bank["records"],
                field="record_id",
                value=case["case_provenance"]["bank_v2_record_id"],
            )
            materialized = runtime_api.materialize_record(qwen, frontend, config, record)
            require(materialized["prompt_ids"] == case["prompt_token_ids"], "materialized prompt differs from packet")
            phase = f"execute:{case['case_id']}"
            receipt = _run_case(
                runtime_api=runtime_api,
                qwen=qwen,
                materialized=materialized,
                case=case,
            )
            case_path = output / f"case-{index:02d}-{case['case_id']}.json"
            case_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            cases.append(receipt)
        invalid = [case["case_id"] for case in cases if case["mechanical_classification"]["status"] == "technical_invalid"]
        result = {
            "schema": "row_feedback.content_run_receipt.v1",
            "status": "completed_with_technical_invalid_cases" if invalid else "completed_non_gating_content_diagnostic",
            "claim_boundary": "Exposed-case supporting diagnostic only; results cannot gate the paired fit or establish native incapacity.",
            "packet": {
                "path": str(packet_path.resolve()),
                "file_sha256": _file_sha256(packet_path),
                "packet_sha256": packet["packet_sha256"],
            },
            "adapter_path": str(adapter_path.resolve()),
            "device": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "loaded_identity": identity,
            "case_count": len(cases),
            "technical_invalid_cases": invalid,
            "case_receipts": [
                {
                    "case_id": case["case_id"],
                    "path": f"case-{index:02d}-{case['case_id']}.json",
                    "sha256": _file_sha256(output / f"case-{index:02d}-{case['case_id']}.json"),
                }
                for index, case in enumerate(cases)
            ],
            "wall_seconds": time.monotonic() - started,
        }
        (output / "receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result
    except BaseException as exc:
        failure = {
            "schema": "row_feedback.content_run_failure.v1",
            "status": "failed",
            "phase": phase,
            "error": f"{type(exc).__name__}: {exc}",
            "packet_path": str(packet_path.resolve()),
            "adapter_path": str(adapter_path.resolve()),
            "wall_seconds": time.monotonic() - started,
        }
        (output / "failure.json").write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    require(device.type == "cuda", "content diagnostic CLI requires a CUDA device")
    result = run_diagnostic(
        packet_path=args.packet,
        adapter_path=args.adapter,
        output=args.output,
        device=device,
    )
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
