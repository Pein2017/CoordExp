"""Explicit-D endpoint for the fixed positive-progress matched control.

This task-local producer emits only the selected A-at-k17 control D.  It uses
the accepted C endpoint as pinned code/evidence and never rewrites its global
arm, relabels a C artifact, or executes another A/Stable endpoint.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback
from typing import Any
from collections.abc import Mapping, Sequence


WORKTREE = Path("/data/CoordExp/.worktrees/research-probes")
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

from probes.dora_owner_learning import margin_preserved_endpoint as accepted  # noqa: E402
from probes.dora_owner_learning.candidate_opportunity import (  # noqa: E402
    file_hash,
    indexed,
    require,
    score,
)
from probes.dora_owner_learning.entrance_ce_eval import aggregate_scores  # noqa: E402
from probes.dora_owner_learning.geometric_dedup_eval import overlap_counts  # noqa: E402
from probes.dora_owner_learning.route_access import checkpoint_config  # noqa: E402
from src.adapters.dora import inspect_dora_adapter_payload  # noqa: E402


RAW_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-11-positive-progress-matched-control"
)
PREPARATION = RAW_ROOT / "endpoint-preparation"
SELECTION = RAW_ROOT / "selection.json"
SELECTION_SHA256 = "f65c38c0f04de31eccdad2a265e84aeae8756e03d7e8a41e590a9b6c10028dda"

ACCEPTED_SOURCE = WORKTREE / "probes/dora_owner_learning/margin_preserved_endpoint.py"
ACCEPTED_SOURCE_SHA256 = "c35d2dfbeafe45e2f5f2f58114f39fd67b632a73264ceffd58c561c47344d5b4"
TRAINER_SOURCE = WORKTREE / "probes/dora_owner_learning/positive_progress_matched_train.py"

C_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-11-margin-preserved-positive-branch"
)
C_PACKET = C_ROOT / "endpoint-preparation/packet.json"
C_PACKET_SHA256 = "5af898bc136e542c4e6b296e686389399b20bb44d3b27adb337080b4a03a6801"
C_CONSUMER = C_ROOT / "endpoint-C/consumer.json"
C_CONSUMER_SHA256 = "c0f1f73c107956051c821b1d038a7434ea76a958c6c8a13a3716dde385c0533e"
C_CONDITIONAL = C_ROOT / "endpoint-C/conditional-consumer.json"
C_CONDITIONAL_SHA256 = "e505811cbcca8f57248e28dadef4904f05f8f4f272752e39b66e4867d6f302bc"
C_RESULT = C_ROOT / "endpoint-C/result.json"
C_RESULT_SHA256 = "1d7dcd76adaf333a80e2479da95a8e58db5964f6a756aab3d6890f578554bd99"
C_RESOURCES = C_ROOT / "endpoint-C/resources.json"
C_RESOURCES_SHA256 = "af110b27e1ee2cf8bc6afd2b0fa673100c9dbaa30da65921f2d2a231f209148c"

ARM = "D"
SELECTED_UPDATES = 17
SELECTED_SCORE_STEP = 18
SELECTED_STATE_SHA256 = "80c9c39c42e9e76dfeda8b10abf7a66928bbf4fd1e1a7c1df79af6a47e1c5dd4"
SELECTED_SUM_POSITIVE_NLL = 21.453418254852295
TARGET_C_SUM_POSITIVE_NLL = 21.61498737335205
WORLD_SIZE = accepted.WORLD_SIZE
NATURAL_PER_RANK = accepted.NATURAL_PER_RANK
CAP = accepted.CAP
EOS = accepted.EOS
PAD = accepted.PAD

PACKET_SCHEMA = "positive_progress_matched_endpoint.packet.v1"
NATURAL_SCHEMA = "positive_progress_matched_endpoint.natural.v1"
CONDITIONAL_SCHEMA = "positive_progress_matched_endpoint.conditional.v1"
TERMINAL_SCHEMA = "positive_progress_matched_endpoint.terminal.v1"
OUTER_EXIT_SCHEMA = "positive_progress_matched_endpoint.outer_exit.v1"
RESULT_SCHEMA = "positive_progress_matched_endpoint.result.v1"

ForwardBudgetExceeded = accepted.ForwardBudgetExceeded
forward_budget_hook = accepted.forward_budget_hook
load_json = accepted.load_json
load_jsonl = accepted.load_jsonl
publish = accepted.publish
digest = accepted.digest


def accepted_base() -> Any:
    accepted.source(ACCEPTED_SOURCE, ACCEPTED_SOURCE_SHA256)
    return accepted


def source(path: Path, sha256: str) -> dict[str, Any]:
    return accepted_base().source(path, sha256)


def load_old_packet() -> dict[str, Any]:
    return accepted_base().load_old_packet()


def validate_selection_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    require(
        value.get("schema") == "positive_progress_matched_control.selection.v1"
        and value.get("status") == "lead_accepted_selection"
        and value.get("arm") == ARM,
        "lead-accepted D selection required",
    )
    require(
        value.get("optimizer_updates") == SELECTED_UPDATES
        and value.get("selected_scalar_step") == SELECTED_SCORE_STEP
        and value.get("selected", {}).get("optimizer_updates") == SELECTED_UPDATES
        and value.get("selected", {}).get("scores_measured_before_update")
        == SELECTED_SCORE_STEP,
        "selected k17/post-update score-before18 identity",
    )
    require(
        value.get("selected", {}).get("adapter_state_sha256") == SELECTED_STATE_SHA256
        and value.get("selected", {}).get("state_artifact")
        == value.get("selected_post_update_state_oracle"),
        "selected A state identity",
    )
    require(
        value.get("selection_metric") == "sum3(-sum_logprob)"
        and value.get("natural_endpoint_fields_used") is False,
        "natural endpoint fields must not enter selection",
    )
    require(
        value.get("selected", {}).get("sum_positive_nll")
        == SELECTED_SUM_POSITIVE_NLL
        and value.get("target_C32_sum_positive_nll") == TARGET_C_SUM_POSITIVE_NLL
        and value.get("relative_sum_residual", 1.0)
        <= value.get("relative_residual_feasibility_bound", 0.0)
        == 0.1,
        "selected scalar feasibility identity",
    )
    require(
        value.get("case_order") == ["351017-c01", "417044-c01", "477415-c02"]
        and len(value.get("curve", [])) == 33
        and len(value.get("recreation_step_oracles", [])) == SELECTED_UPDATES,
        "complete selected A curve/oracles",
    )
    require(
        value.get("selected_adapter_payload_retained") is False
        and value.get("gpu_launch_authorized") is False
        and value.get("endpoint_read_authorized") is False,
        "selection is CPU evidence, not a launch grant",
    )
    return dict(value)


def validate_selection() -> dict[str, Any]:
    source(SELECTION, SELECTION_SHA256)
    return validate_selection_payload(load_json(SELECTION))


def validate_retained_ledgers(old_packet: Mapping[str, Any]) -> dict[str, Any]:
    base = accepted_base()
    base.validate_retained_baselines(old_packet)
    for path, sha256 in (
        (C_PACKET, C_PACKET_SHA256),
        (C_CONSUMER, C_CONSUMER_SHA256),
        (C_CONDITIONAL, C_CONDITIONAL_SHA256),
        (C_RESULT, C_RESULT_SHA256),
        (C_RESOURCES, C_RESOURCES_SHA256),
    ):
        source(path, sha256)
    c_packet = load_json(C_PACKET)
    base.validate_packet(c_packet)
    c_rows = load_json(C_CONSUMER)
    base.validate_natural_rows(c_rows, old_packet, arm="C", retained=False)
    c_conditional = load_json(C_CONDITIONAL)
    base.validate_conditional_rows(c_conditional, c_packet)
    result, resources = load_json(C_RESULT), load_json(C_RESOURCES)
    require(
        result.get("schema") == accepted.RESULT_SCHEMA
        and result.get("arm") == "C"
        and resources.get("arm") == "C"
        and resources.get("model_loads") == 8
        and resources.get("continuations") == 390,
        "retained C endpoint identity",
    )
    return {
        "A_natural": 384,
        "C_natural": len(c_rows),
        "C_conditional": len(c_conditional),
        "C_arm": "C",
        "C_result_sha256": C_RESULT_SHA256,
    }


def build_conditional_jobs(old_packet: Mapping[str, Any]) -> list[Any]:
    old = accepted_base().old_endpoint()
    jobs = []
    for index, case in enumerate(old_packet["conditional_cases"]):
        h_ids, c_ids = list(case["h_ids"]), list(case["c_ids"])
        common = {
            "arm": ARM,
            "case_id": str(case["case_id"]),
            "candidate_id": str(case["candidate_id"]),
        }
        h_rows = int(case["h_complete_row_count"])
        jobs.append(old.ConditionalJob(
            rank=2 * index,
            kind="h_only",
            prefix_ids=h_ids,
            prefix_row_count=h_rows,
            forced_candidate=None,
            budget=CAP - len(h_ids),
            **common,
        ))
        jobs.append(old.ConditionalJob(
            rank=2 * index + 1,
            kind="h_plus_c",
            prefix_ids=h_ids + c_ids,
            prefix_row_count=h_rows + 1,
            forced_candidate={
                "candidate_id": case["candidate_id"],
                "description": case.get("description"),
                "c_ids": c_ids,
            },
            budget=CAP - len(h_ids) - len(c_ids),
            **common,
        ))
    require([job.rank for job in jobs] == list(range(6)), "fixed D conditional6 partition")
    return jobs


def packet_base(old_packet: Mapping[str, Any]) -> dict[str, Any]:
    selection = validate_selection()
    validate_retained_ledgers(old_packet)
    jobs = build_conditional_jobs(old_packet)
    bounds = accepted_base().rank_bounds(jobs)
    require(
        sum(row["max_generated_tokens"] for row in bounds) == 120_000
        and all(row["max_worker_seconds"] == 1500 for row in bounds),
        "D endpoint resource allocation",
    )
    return {
        "schema": PACKET_SCHEMA,
        "status": "pending_actual_D17_receipt_and_cold_check",
        "arm": ARM,
        "selection": {
            "path": str(SELECTION),
            "sha256": SELECTION_SHA256,
            "optimizer_updates": SELECTED_UPDATES,
            "adapter_state_sha256": SELECTED_STATE_SHA256,
            "sum_positive_nll": selection["selected"]["sum_positive_nll"],
            "target_C32_sum_positive_nll": selection["target_C32_sum_positive_nll"],
            "relative_sum_residual": selection["relative_sum_residual"],
        },
        "model": old_packet["model"],
        "config": old_packet["config"],
        "eval_records": old_packet["eval_records"],
        "eval_shards": old_packet["eval_shards"],
        "protected_targets": old_packet["protected_targets"],
        "conditional_cases": old_packet["conditional_cases"],
        "conditional_jobs": [job.to_dict() for job in jobs],
        "policy": copy.deepcopy(load_json(C_PACKET)["policy"]),
        "resource_bounds": {
            "workers": 8,
            "natural_calls": 384,
            "conditional_calls": 6,
            "total_calls": 390,
            "model_loads": 8,
            "global_max_generated_tokens": 120_000,
            "ranks": bounds,
        },
        "panels": {
            "reference56": {"images": 56, "role": "training protection"},
            "train256": {"images": 256, "role": "exposed train-side read"},
            "dev128": {"images": 128, "role": "already exposed development screen; not holdout"},
        },
        "claim_boundary": (
            "One actual D=A@k17 exposed384+conditional6 read, primarily paired against "
            "retained C32, with retained A32/Stable50 ledgers; no causal or fresh-generalization claim."
        ),
    }


def pending_payload() -> dict[str, Any]:
    old_packet = load_old_packet()
    selection = validate_selection()
    retained = validate_retained_ledgers(old_packet)
    base = packet_base(old_packet)
    return {
        "schema": "positive_progress_matched_endpoint.pending_bindings.v1",
        "status": "pending_actual_D17_receipt_and_cold_check",
        "unresolved": ["actual_D17_training_receipt", "actual_D17_cold_check"],
        "selection": source(SELECTION, SELECTION_SHA256),
        "selected": {
            "optimizer_updates": selection["optimizer_updates"],
            "scores_measured_before_update": selection["selected_scalar_step"],
            "adapter_state_sha256": selection["selected"]["adapter_state_sha256"],
            "sum_positive_nll": selection["selected"]["sum_positive_nll"],
            "target_C32_sum_positive_nll": selection["target_C32_sum_positive_nll"],
            "relative_sum_residual": selection["relative_sum_residual"],
        },
        "retained": retained,
        "verified_sources": {
            "accepted_endpoint_producer": source(ACCEPTED_SOURCE, ACCEPTED_SOURCE_SHA256),
            "old_endpoint_producer": source(accepted.OLD_PRODUCER, accepted.OLD_PRODUCER_SHA256),
            "old_endpoint_packet": source(accepted.OLD_PACKET, accepted.OLD_PACKET_SHA256),
            "A_consumer": source(accepted.RETAINED_A_CONSUMER, accepted.RETAINED_A_CONSUMER_SHA256),
            "C_packet": source(C_PACKET, C_PACKET_SHA256),
            "C_consumer": source(C_CONSUMER, C_CONSUMER_SHA256),
            "C_conditional": source(C_CONDITIONAL, C_CONDITIONAL_SHA256),
            "C_result": source(C_RESULT, C_RESULT_SHA256),
            "C_resources": source(C_RESOURCES, C_RESOURCES_SHA256),
        },
        "endpoint": {
            "arm": ARM,
            "natural": 384,
            "conditional": 6,
            "policy": base["policy"],
            "resource_bounds": base["resource_bounds"],
            "panels": base["panels"],
        },
        "final_packet_rule": (
            "packet.json remains absent until root accepts actual D17 state parity, saved adapter, "
            "final scores/reference diagnostics, and matching cold reload."
        ),
    }


def verify_training_receipt(receipt_path: Path) -> dict[str, Any]:
    from probes.dora_owner_learning import positive_progress_matched_train as trainer

    require(receipt_path.is_absolute(), "D receipt path must be absolute")
    source(TRAINER_SOURCE, file_hash(TRAINER_SOURCE))
    observed = trainer.verify_receipt(receipt_path.parent)
    require(observed == load_json(receipt_path), "D trainer verifier/file differs")
    return observed


def validate_d_training_metadata(
    receipt: Mapping[str, Any], cold: Mapping[str, Any], selection: Mapping[str, Any]
) -> None:
    require(receipt.get("arm") == ARM and cold.get("arm") == ARM, "only D training/cold identities accepted")
    require(
        receipt.get("schema") == "positive_progress_matched_train.receipt.v1"
        and receipt.get("status") == "completed"
        and receipt.get("optimizer_updates") == SELECTED_UPDATES
        and receipt.get("final_adapter_state_sha256") == SELECTED_STATE_SHA256,
        "complete selected D17 receipt required",
    )
    require(
        receipt.get("selection")
        == {"path": str(SELECTION), "sha256": SELECTION_SHA256}
        and receipt.get("selected_state_oracle", {}).get("adapter_sha256")
        == SELECTED_STATE_SHA256,
        "D receipt selection/state oracle",
    )
    require(
        receipt.get("final_live_positive_scores") == selection["selected"]["positive_scores"]
        and receipt.get("sum_positive_nll") == SELECTED_SUM_POSITIVE_NLL,
        "D17 final positive scores/Q",
    )
    reference_records = receipt.get("final_reference_records", [])
    require(
        isinstance(reference_records, list)
        and len(reference_records) == WORLD_SIZE
        and [row.get("rank") for row in reference_records] == list(range(WORLD_SIZE)),
        "D17 final reference rank coverage",
    )
    reference_rows = 0
    for reference in reference_records:
        source(Path(reference["path"]), reference["sha256"])
        reference_rows += len(load_json(Path(reference["path"]))["records"])
    require(
        reference_rows == 56
        and receipt.get("margin_signal", {}).get("final_post_update", {}).get("normal_count") == 56,
        "D17 final56 reference/margin diagnostics",
    )
    resources = receipt.get("resources", {})
    expected = selection["training_counter_bounds"]
    require(
        all(resources.get(key) == expected[key] for key in (
            "model_loads", "model_forwards", "image_forwards", "backwards",
            "synchronized_backwards", "final_reference_forwards",
        ))
        and resources.get("reference_forwards") == expected["source_reference_forwards"]
        and resources.get("negative_samples", 0) == 0
        and resources.get("raw_sampled_tokens", 0) == 0,
        "D17 exact training counters",
    )
    require(
        cold.get("schema") == "positive_progress_matched_train.cold_check.v1"
        and cold.get("status") == "passed"
        and cold.get("optimizer_updates") == SELECTED_UPDATES
        and cold.get("saved_adapter") == receipt.get("saved_adapter")
        and cold.get("positive_scores") == receipt.get("final_live_positive_scores")
        and cold.get("model_loads") == 1
        and cold.get("score_forwards") == 3,
        "matching D17 cold reload",
    )
    require(
        all(value == 0 for row in cold.get("live_score_deltas", {}).values() for value in row.values()),
        "D17 cold positive deltas must be exact zero",
    )


def validate_d_training(
    receipt_path: Path, cold_path: Path, old_packet: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    require(receipt_path.is_absolute() and cold_path.is_absolute(), "D binding paths must be absolute")
    receipt, cold = load_json(receipt_path), load_json(cold_path)
    selection = validate_selection()
    validate_d_training_metadata(receipt, cold, selection)
    require(
        Path(cold["training_receipt"]["path"]).resolve() == receipt_path.resolve()
        and cold["training_receipt"]["sha256"] == file_hash(receipt_path),
        "D cold/training receipt identity",
    )
    require(verify_training_receipt(receipt_path) == receipt, "D native sealed receipt verification")
    composition = receipt["composition"]
    require(
        composition["base_model_path"] == old_packet["model"]["base_model_path"]
        and composition["source_embedding"] == old_packet["model"]["source_embedding"]
        and composition["source_adapter"] == old_packet["model"]["current_adapter"]
        and composition["unmerged"] is True
        and composition["dtype"] == "fp32"
        and composition["attention_implementation"] == "sdpa",
        "D training composition differs from Stable50",
    )
    adapter = receipt["saved_adapter"]
    require(
        inspect_dora_adapter_payload(adapter["root"], composition["base_model_path"]) == adapter,
        "D saved adapter semantic identity",
    )
    return receipt, cold


def build_packet_payload(receipt_path: Path, cold_path: Path) -> dict[str, Any]:
    old_packet = load_old_packet()
    receipt, cold = validate_d_training(receipt_path, cold_path, old_packet)
    result = packet_base(old_packet)
    result.update(
        status="ready_for_one_D_endpoint_requires_root_gpu_grant",
        sources={
            "producer": source(Path(__file__).resolve(), file_hash(Path(__file__).resolve())),
            "accepted_endpoint_producer": source(ACCEPTED_SOURCE, ACCEPTED_SOURCE_SHA256),
            "D_training_verifier": source(TRAINER_SOURCE, file_hash(TRAINER_SOURCE)),
            "selection": source(SELECTION, SELECTION_SHA256),
            "old_endpoint_producer": source(accepted.OLD_PRODUCER, accepted.OLD_PRODUCER_SHA256),
            "old_endpoint_packet": source(accepted.OLD_PACKET, accepted.OLD_PACKET_SHA256),
            "A_consumer": source(accepted.RETAINED_A_CONSUMER, accepted.RETAINED_A_CONSUMER_SHA256),
            "A_conditional_reduction": source(
                accepted.RETAINED_A_CONDITIONAL_REDUCTION,
                accepted.RETAINED_A_CONDITIONAL_REDUCTION_SHA256,
            ),
            "C_packet": source(C_PACKET, C_PACKET_SHA256),
            "C_consumer": source(C_CONSUMER, C_CONSUMER_SHA256),
            "C_conditional": source(C_CONDITIONAL, C_CONDITIONAL_SHA256),
            "C_result": source(C_RESULT, C_RESULT_SHA256),
            "C_resources": source(C_RESOURCES, C_RESOURCES_SHA256),
            "D_training_receipt": source(receipt_path, file_hash(receipt_path)),
            "D_cold_check": source(cold_path, file_hash(cold_path)),
        },
        candidate={
            "arm": ARM,
            "optimizer_updates": SELECTED_UPDATES,
            "training_receipt": {"path": str(receipt_path), "sha256": file_hash(receipt_path)},
            "cold_check": {"path": str(cold_path), "sha256": file_hash(cold_path)},
            "saved_adapter": receipt["saved_adapter"],
            "cold_model_identity": cold["model_identity"],
            "sum_positive_nll": receipt["sum_positive_nll"],
            "final_margin_signal": receipt["margin_signal"]["final_post_update"],
        },
    )
    return result


def validate_packet(packet: Mapping[str, Any]) -> None:
    require(
        packet.get("schema") == PACKET_SCHEMA
        and packet.get("arm") == ARM
        and packet.get("status") == "ready_for_one_D_endpoint_requires_root_gpu_grant",
        "only final D endpoint packet accepted",
    )
    for item in packet["sources"].values():
        source(Path(item["path"]), item["sha256"])
    expected = build_packet_payload(
        Path(packet["candidate"]["training_receipt"]["path"]),
        Path(packet["candidate"]["cold_check"]["path"]),
    )
    require(dict(packet) == expected, "D packet differs from frozen bindings")


def prepare_final_packet(receipt_path: Path, cold_path: Path, out: Path) -> dict[str, Any]:
    require(out.is_absolute() and not out.exists(), "fresh absolute final D packet output")
    payload = build_packet_payload(receipt_path, cold_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    publish(out, payload)
    return payload


def runtime_conditional_jobs(packet: Mapping[str, Any], rank: int | None = None) -> list[Any]:
    old = accepted_base().old_endpoint()
    rows = packet["conditional_jobs"]
    if rank is not None:
        rows = [row for row in rows if row["rank"] == rank]
    jobs = [old.ConditionalJob(**dict(row)) for row in rows]
    require(all(job.arm == ARM for job in jobs), "only D conditional jobs")
    return jobs


def validate_d_natural_rows(
    rows: Sequence[Mapping[str, Any]], packet: Mapping[str, Any]
) -> None:
    expected = [
        (rank, str(example_id))
        for rank, shard in enumerate(packet["eval_shards"])
        for example_id in shard
    ]
    observed = [(int(row["shard"]), str(row["example_id"])) for row in rows]
    require(len(rows) == len({example_id for _, example_id in observed}) == 384, "exact384 D identity")
    require(observed == expected, "exact384 D ordered shard identity")
    require(
        all(row.get("arm") == ARM and row.get("schema") == NATURAL_SCHEMA for row in rows),
        "only D natural rows accepted",
    )
    frozen = indexed(packet["eval_records"], "example_id")
    shards = {example_id: rank for rank, ids in enumerate(packet["eval_shards"]) for example_id in ids}
    for row in rows:
        case = frozen[row["example_id"]]
        plan = case["case"]["image_plan"]
        identity = {
            "prompt_token_ids_sha256": digest(case["prompt_token_ids"]),
            "executed_media_sha256": plan["executed_media_sha256"],
            "observed_image_grid_thw": plan["observed_image_grid_thw"],
        }
        require(
            row["request_id"] == row["example_id"] == case["example_id"]
            and row["shard"] == shards[row["example_id"]]
            and row["image_id"] == case["image_id"]
            and row["split"] == case["split"],
            "D request/scored order differs",
        )
        require(
            all(row.get(key) == value for key, value in identity.items())
            and row.get("batch_identity_sha256") == digest(identity),
            "D prompt/media/grid differs",
        )
        card = score(row["parsed"], seed=None, length=len(row["action_ids"]), stop=row["stop_reason"])
        require(
            row["score"] == card and row["overlap_counts"] == overlap_counts(row["parsed"]),
            "D natural score differs",
        )


def validate_d_conditional_rows(
    rows: Sequence[Mapping[str, Any]], packet: Mapping[str, Any]
) -> None:
    old = accepted_base().old_endpoint()
    jobs = {
        (job.rank, f"{job.kind}:{job.candidate_id}"): job
        for job in runtime_conditional_jobs(packet)
    }
    require(
        len(rows) == 6
        and [(int(row["rank"]), str(row["job_id"])) for row in rows] == sorted(jobs),
        "exact D conditional6 ordered identity",
    )
    frozen = indexed(packet["eval_records"], "example_id")
    for row in rows:
        require(
            row.get("arm") == ARM and row.get("schema") == CONDITIONAL_SCHEMA,
            "only D conditional rows accepted",
        )
        job = jobs[(int(row["rank"]), str(row["job_id"]))]
        old.validate_conditional_artifact(row, job)
        example_id = next(
            case["example_id"]
            for case in packet["conditional_cases"]
            if case["case_id"] == job.case_id
        )
        case = frozen[example_id]
        plan = case["case"]["image_plan"]
        identity = {
            "prompt_token_ids_sha256": digest(case["prompt_token_ids"]),
            "executed_media_sha256": plan["executed_media_sha256"],
            "observed_image_grid_thw": plan["observed_image_grid_thw"],
        }
        require(
            row["request_id"] == example_id
            and all(row.get(key) == value for key, value in identity.items())
            and row.get("batch_identity_sha256") == digest(identity),
            "D conditional request/prompt/media/grid differs",
        )


def execute(packet_path: Path, out: Path, shard: int) -> None:
    import torch
    from probes.dora_owner_learning.runtime import load_policy
    from probes.source_rweak_row_cross.run import build_requests, native_record
    from src.config.inference import InferConfig
    from src.qwen.generation import NativeGenerationPolicy, generate_continuations
    from src.qwen.native import prepare_native_inputs

    require(packet_path.is_absolute() and out.is_absolute(), "D execution paths must be absolute")
    packet = load_json(packet_path)
    validate_packet(packet)
    require(
        0 <= shard < WORLD_SIZE
        and os.environ.get("CUDA_VISIBLE_DEVICES") == str(shard)
        and torch.cuda.device_count() == 1,
        "single assigned D GPU identity",
    )
    adapter = packet["candidate"]["saved_adapter"]
    receipt = packet["candidate"]["training_receipt"]
    run = out / f"shard-{shard}"
    require(not run.exists(), "occupied D endpoint shard")
    run.mkdir(parents=True)
    packet_sha = file_hash(packet_path)
    terminal = {
        "schema": TERMINAL_SCHEMA,
        "status": "running",
        "arm": ARM,
        "shard": shard,
        "pid": os.getpid(),
        "packet_sha256": packet_sha,
        "training_receipt_sha256": receipt["sha256"],
        "adapter_fingerprint": adapter["fingerprint"],
        "model_loads": 0,
        "continuations": 0,
        "natural_continuations": 0,
        "conditional_continuations": 0,
        "new_tokens": 0,
        "model_forwards": 0,
        "image_forwards": 0,
    }
    publish(run / "launch.json", dict(terminal))
    started = time.monotonic()
    handles: list[Any] = []
    alarm = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("D endpoint worker timeout")))
        signal.alarm(accepted.MAX_WORKER_SECONDS)
        config = checkpoint_config(InferConfig.model_validate(packet["config"]), adapter["root"])
        torch.cuda.reset_peak_memory_stats()
        qwen, identity = load_policy(config, device=torch.device("cuda:0"))
        terminal["model_loads"] = 1
        live = identity["model_identity"]["adapter"]
        require(
            live["adapter_path"] == adapter["root"]
            and live["merged_adapters"] == []
            and inspect_dora_adapter_payload(
                live["adapter_path"], packet["model"]["base_model_path"]
            ) == adapter,
            "loaded D adapter identity",
        )
        require(
            identity["effective_settings"]["observed_model_dtype"]["parameter_dtype_names"]
            == ["torch.float32"]
            and identity["effective_settings"]["observed_attn_implementation"] == "sdpa",
            "live D numerics",
        )
        publish(run / "model.json", {
            "arm": ARM,
            "identity": identity,
            "exported_adapter": adapter,
            "training_receipt": receipt,
            "cold_check": packet["candidate"]["cold_check"],
        })
        publish(run / "config.json", config.model_dump(mode="json"))
        qwen.model.eval()
        for parameter in qwen.model.parameters():
            parameter.requires_grad_(False)
        counters = {"model": 0, "image": 0}
        handles.append(qwen.model.register_forward_pre_hook(forward_budget_hook(counters)))
        visuals = [module for name, module in qwen.model.named_modules() if name.endswith("visual")]
        require(len(visuals) == 1, "ambiguous visual module")
        handles.append(
            visuals[0].register_forward_pre_hook(
                lambda *_: counters.__setitem__("image", counters["image"] + 1)
            )
        )
        policy = NativeGenerationPolicy(
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.0,
            use_model_defaults=False,
        )
        by_id = indexed(packet["eval_records"], "example_id")

        def batch_for(frozen: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
            requests, _ = build_requests(qwen, packet["config"], [frozen["case"]])
            require(list(requests[0].expected_token_ids) == frozen["prompt_token_ids"], "D request prompt")
            batch = prepare_native_inputs(
                qwen.processor,
                requests,
                device=torch.device("cuda:0"),
                record_media_identity=True,
            )
            plan = frozen["case"]["image_plan"]
            require(
                list(batch.prompt_token_ids[0]) == frozen["prompt_token_ids"]
                and batch.media_sha256[0] == plan["executed_media_sha256"]
                and list(batch.image_grids[0]) == plan["observed_image_grid_thw"],
                "live D prompt/media/grid",
            )
            observed = {
                "prompt_token_ids_sha256": digest(frozen["prompt_token_ids"]),
                "executed_media_sha256": batch.media_sha256[0],
                "observed_image_grid_thw": list(batch.image_grids[0]),
            }
            observed["batch_identity_sha256"] = digest(observed)
            return batch, observed

        with (run / "natural-rows.jsonl").open("x") as stream:
            for example_id in packet["eval_shards"][shard]:
                frozen = by_id[example_id]
                batch, batch_identity = batch_for(frozen)
                with torch.inference_mode():
                    generated = generate_continuations(
                        qwen.model,
                        batch,
                        extensions=[[]],
                        budgets=[CAP],
                        eos_token_id=EOS,
                        pad_token_id=qwen.tokenizer.pad_token_id,
                        policy=policy,
                        trace="none",
                    )[0]
                require(generated.request_id == example_id, "D natural request association")
                ids = list(generated.token_ids)
                accepted._checked_action(ids, generated.stop_reason, CAP)
                text = qwen.tokenizer.decode(ids, skip_special_tokens=False)
                parsed = native_record(text, frozen["case"], frozen["golden"], generated.stop_reason)
                row = {
                    "schema": NATURAL_SCHEMA,
                    "arm": ARM,
                    "shard": shard,
                    "packet_sha256": packet_sha,
                    "training_receipt_sha256": receipt["sha256"],
                    "adapter_fingerprint": adapter["fingerprint"],
                    "request_id": generated.request_id,
                    "example_id": example_id,
                    "image_id": frozen["image_id"],
                    "split": frozen["split"],
                    "action_ids": ids,
                    "prefix_ids": [],
                    "forced_ids": [],
                    "remaining_budget": CAP,
                    "text": text,
                    "stop_reason": generated.stop_reason,
                    "parsed": parsed,
                    "score": score(parsed, seed=None, length=len(ids), stop=generated.stop_reason),
                    "overlap_counts": overlap_counts(parsed),
                    **batch_identity,
                }
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                terminal["natural_continuations"] += 1
                terminal["continuations"] += 1
                terminal["new_tokens"] += len(ids)

        old = accepted_base().old_endpoint()
        with (run / "conditional-rows.jsonl").open("x") as stream:
            for job in runtime_conditional_jobs(packet, shard):
                example_id = next(
                    case["example_id"]
                    for case in packet["conditional_cases"]
                    if case["case_id"] == job.case_id
                )
                frozen = by_id[example_id]
                batch, batch_identity = batch_for(frozen)
                with torch.inference_mode():
                    generated = generate_continuations(
                        qwen.model,
                        batch,
                        extensions=[job.prefix_ids],
                        budgets=[job.budget],
                        eos_token_id=EOS,
                        pad_token_id=qwen.tokenizer.pad_token_id,
                        policy=policy,
                        trace="none",
                    )[0]
                require(generated.request_id == example_id, "D conditional request association")
                free_ids = list(generated.token_ids)
                accepted._checked_action(free_ids, generated.stop_reason, job.budget)
                text = qwen.tokenizer.decode(job.prefix_ids + free_ids, skip_special_tokens=False)
                parsed = native_record(text, frozen["case"], frozen["golden"], generated.stop_reason)
                row = old.conditional_artifact(
                    job,
                    free_ids=free_ids,
                    stop_reason=generated.stop_reason,
                    parsed=parsed,
                )
                row.update(
                    schema=CONDITIONAL_SCHEMA,
                    packet_sha256=packet_sha,
                    training_receipt_sha256=receipt["sha256"],
                    adapter_fingerprint=adapter["fingerprint"],
                    request_id=generated.request_id,
                    text=text,
                    **batch_identity,
                )
                old.validate_conditional_artifact(row, job)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                terminal["conditional_continuations"] += 1
                terminal["continuations"] += 1
                terminal["new_tokens"] += len(free_ids)

        terminal.update(model_forwards=counters["model"], image_forwards=counters["image"])
        require(
            terminal["natural_continuations"] == 48
            and terminal["conditional_continuations"]
            == packet["resource_bounds"]["ranks"][shard]["conditional_calls"],
            "D exact runtime calls",
        )
        terminal.update(status="completed", exit_code=0)
    except BaseException as exc:
        model_forwards = locals().get("counters", {}).get("model", 0)
        terminal.update(
            status="failed",
            exit_code=1,
            error=repr(exc),
            traceback=traceback.format_exc(),
            failure={
                "type": type(exc).__name__,
                "budget_exhausted": isinstance(exc, ForwardBudgetExceeded),
                "completed_natural_records": terminal["natural_continuations"],
                "completed_conditional_records": terminal["conditional_continuations"],
                "completed_record_tokens": terminal["new_tokens"],
                "admitted_model_forwards": model_forwards,
                "partial_unpublished_forward_count": max(0, model_forwards - terminal["new_tokens"]),
            },
        )
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, alarm)
        for handle in handles:
            handle.remove()
        terminal.update(
            elapsed_seconds=time.monotonic() - started,
            model_forwards=locals().get("counters", {}).get("model", 0),
            image_forwards=locals().get("counters", {}).get("image", 0),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()
            if torch.cuda.is_initialized() else 0,
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved()
            if torch.cuda.is_initialized() else 0,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        )
        if terminal["status"] == "completed":
            try:
                accepted.validate_terminal_resources(
                    terminal, packet["resource_bounds"]["ranks"][shard]
                )
            except BaseException as exc:
                terminal.update(
                    status="failed",
                    exit_code=1,
                    error=repr(exc),
                    traceback=traceback.format_exc(),
                )
                publish(run / "terminal.json", terminal)
                raise
        publish(run / "terminal.json", terminal)


def launch(packet_path: Path, out: Path) -> dict[str, Any]:
    require(packet_path.is_absolute() and out.is_absolute(), "D launcher paths must be absolute")
    validate_packet(load_json(packet_path))
    launcher = out / "launcher"
    require(
        not launcher.exists()
        and not any((out / f"shard-{rank}").exists() for rank in range(WORLD_SIZE)),
        "occupied D endpoint launch",
    )
    launcher.mkdir(parents=True)
    command_base = [
        sys.executable,
        str(Path(__file__).resolve()),
        "execute",
        "--packet", str(packet_path),
        "--out", str(out),
    ]
    processes = []
    for shard in range(WORLD_SIZE):
        command = [*command_base, "--shard", str(shard)]
        log = (launcher / f"shard-{shard}.log").open("x")
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(shard)
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, env=environment
        )
        processes.append((shard, process, log, command))
    group = {
        "schema": "positive_progress_matched_endpoint.group_launch.v1",
        "arm": ARM,
        "packet": {"path": str(packet_path), "sha256": file_hash(packet_path)},
        "output_root": str(out),
        "workers": [
            {"shard": shard, "pid": process.pid, "command": command,
             "cuda_visible_devices": str(shard)}
            for shard, process, _, command in processes
        ],
        "automatic_retry": False,
    }
    publish(launcher / "group-launch.json", group)
    exits = []
    for shard, process, log, command in processes:
        exit_code = process.wait()
        log.close()
        row = {
            "schema": OUTER_EXIT_SCHEMA,
            "arm": ARM,
            "shard": shard,
            "pid": process.pid,
            "exit_code": exit_code,
            "command": command,
        }
        publish(launcher / f"shard-{shard}-outer-exit.json", row)
        exits.append(row)
    completion = {
        "schema": "positive_progress_matched_endpoint.group_completion.v1",
        "arm": ARM,
        "status": "completed" if all(row["exit_code"] == 0 for row in exits) else "failed",
        "outer_exits": exits,
        "automatic_retry": False,
    }
    publish(launcher / "group-completion.json", completion)
    require(completion["status"] == "completed", "D worker failed; no automatic retry")
    return completion


def validate_outer_exits(launcher: Path) -> list[dict[str, Any]]:
    rows = [load_json(path) for path in sorted(launcher.glob("shard-*-outer-exit.json"))]
    require(
        len(rows) == 8
        and {row.get("shard") for row in rows} == set(range(8))
        and all(
            row.get("schema") == OUTER_EXIT_SCHEMA
            and row.get("arm") == ARM
            and row.get("exit_code") == 0
            for row in rows
        ),
        "complete successful D outer exits",
    )
    rows = sorted(rows, key=lambda row: row["shard"])
    completion = load_json(launcher / "group-completion.json")
    require(
        completion.get("status") == "completed"
        and completion.get("arm") == ARM
        and completion.get("outer_exits") == rows
        and completion.get("automatic_retry") is False,
        "D group completion/outer exits",
    )
    return rows


def d_vs_c_owner_counts(
    c_scores: Sequence[Mapping[str, Any]], d_scores: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, int]]:
    return accepted_base().owner_change_counts(c_scores, d_scores)


def _panel_ids(label: str, rows: Sequence[Mapping[str, Any]]) -> set[str]:
    if label == "reference56":
        ids = {row["example_id"] for row in rows if row["split"] == "reference56"}
        require(len(ids) == 56, "reference56 identity")
        return ids
    if label == "train256":
        ids = {row["example_id"] for row in rows if row["split"] != "dev128"}
        require(len(ids) == 256, "train256 identity")
        return ids
    if label == "dev128":
        ids = {row["example_id"] for row in rows if row["split"] == "dev128"}
        require(len(ids) == 128, "dev128 identity")
        return ids
    require(label == "union384", "unknown D endpoint panel")
    ids = {row["example_id"] for row in rows}
    require(len(ids) == 384, "union384 identity")
    return ids


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "quality": aggregate_scores([row["score"] for row in rows]),
        "burden": accepted_base().burden(rows),
    }


def reduce_endpoint(
    packet: Mapping[str, Any], d_rows: Sequence[Mapping[str, Any]],
    d_conditional: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    validate_d_natural_rows(d_rows, packet)
    validate_d_conditional_rows(d_conditional, packet)
    a_rows = load_json(accepted.RETAINED_A_CONSUMER)
    accepted.validate_natural_rows(a_rows, packet, arm="A", retained=True)
    c_rows = load_json(C_CONSUMER)
    accepted.validate_natural_rows(c_rows, packet, arm="C", retained=False)
    stable = accepted.stable_rows(packet)
    by_arm = {
        "Stable50": indexed(stable, "example_id"),
        "A": indexed(a_rows, "example_id"),
        "C": indexed(c_rows, "example_id"),
        ARM: indexed(list(d_rows), "example_id"),
    }
    panels = {}
    for label in ("reference56", "train256", "dev128", "union384"):
        ids = _panel_ids(label, d_rows)
        chosen = {
            arm: [rows[example_id] for example_id in sorted(ids)]
            for arm, rows in by_arm.items()
        }
        panels[label] = {
            "images": len(ids),
            **{arm: _summary(chosen[arm]) for arm in ("Stable50", "A", "C", ARM)},
            "D_vs_C_owner_changes": d_vs_c_owner_counts(
                [row["score"] for row in chosen["C"]],
                [row["score"] for row in chosen[ARM]],
            ),
            "D_vs_A_owner_changes": accepted.owner_change_counts(
                [row["score"] for row in chosen["A"]],
                [row["score"] for row in chosen[ARM]],
            ),
            "D_vs_Stable50_owner_changes": accepted.owner_change_counts(
                [row["score"] for row in chosen["Stable50"]],
                [row["score"] for row in chosen[ARM]],
            ),
        }
    by_image = {
        arm: {str(row["image_id"]): row for row in rows.values()}
        for arm, rows in by_arm.items()
    }
    positives = {
        image_id: {
            arm: {
                "quality": by_image[arm][image_id]["score"],
                "burden": accepted.burden([by_image[arm][image_id]]),
            }
            for arm in ("Stable50", "A", "C", ARM)
        }
        for image_id in ("351017", "417044", "477415")
    }
    diagnostic = {
        arm: {
            "quality": by_image[arm]["39654"]["score"],
            "burden": accepted.burden([by_image[arm]["39654"]]),
        }
        for arm in ("Stable50", "A", "C", ARM)
    }
    old = accepted_base().old_endpoint()
    _, witness_reducer = old._witness_modules()
    d_cells = [
        witness_reducer.reduce_record(row)
        for row in sorted(d_conditional, key=lambda row: row["rank"])
    ]
    a_conditional = load_json(accepted.RETAINED_A_CONDITIONAL_REDUCTION)
    c_conditional = load_json(C_RESULT)["conditional"]["C"]
    require(
        [(row["case_id"], row["job_id"]) for row in a_conditional["cells"]]
        == [(row["case_id"], row["job_id"]) for row in c_conditional["cells"]]
        == [(row["case_id"], row["job_id"]) for row in d_cells],
        "A/C/D conditional scored order",
    )
    per_image = []
    for row in d_rows:
        example_id = row["example_id"]
        per_image.append({
            "example_id": example_id,
            "image_id": row["image_id"],
            "split": row["split"],
            "D_vs_C": accepted.owner_changes(by_arm["C"][example_id]["score"], row["score"]),
            "D_vs_A": accepted.owner_changes(by_arm["A"][example_id]["score"], row["score"]),
            "D_vs_Stable50": accepted.owner_changes(
                by_arm["Stable50"][example_id]["score"], row["score"]
            ),
        })
    return {
        "schema": RESULT_SCHEMA,
        "arm": ARM,
        "primary_orientation": "D_after_vs_C_before; gained means D-only and lost means C-only",
        "selection": packet["selection"],
        "panels": panels,
        "positive_natural_cases": positives,
        "diagnostic_39654": diagnostic,
        "per_image_owner_changes": per_image,
        "conditional": {
            "A": a_conditional,
            "C": c_conditional,
            ARM: {
                "schema": "positive_progress_matched_endpoint.conditional_reduction.v1",
                "arm": ARM,
                "cells": d_cells,
                "claim_boundary": "Forced c is excluded from free-continuation counts.",
            },
        },
        "D_final_margin_signal": packet["candidate"]["final_margin_signal"],
        "retained_C_result": packet["sources"]["C_result"],
        "claim_boundary": packet["claim_boundary"],
    }


def merge(packet_path: Path, out: Path, *, verify: bool) -> dict[str, Any]:
    from probes.source_rweak_row_cross.run import native_record
    from tokenizers import Tokenizer

    packet = load_json(packet_path)
    validate_packet(packet)
    packet_sha = file_hash(packet_path)
    adapter = packet["candidate"]["saved_adapter"]
    receipt = packet["candidate"]["training_receipt"]
    validate_outer_exits(out / "launcher")
    tokenizer = Tokenizer.from_file(packet["model"]["base_model_path"] + "/tokenizer.json")
    frozen = indexed(packet["eval_records"], "example_id")
    jobs = {
        (job.rank, f"{job.kind}:{job.candidate_id}"): job
        for job in runtime_conditional_jobs(packet)
    }
    natural, conditional, terminals = [], [], []
    old = accepted_base().old_endpoint()
    for shard in range(WORLD_SIZE):
        run = out / f"shard-{shard}"
        terminal = load_json(run / "terminal.json")
        terminals.append(terminal)
        require(
            terminal.get("schema") == TERMINAL_SCHEMA
            and terminal.get("status") == "completed"
            and terminal.get("exit_code") == 0
            and terminal.get("arm") == ARM
            and terminal.get("shard") == shard
            and terminal.get("packet_sha256") == packet_sha
            and terminal.get("training_receipt_sha256") == receipt["sha256"]
            and terminal.get("adapter_fingerprint") == adapter["fingerprint"],
            "D terminal identity",
        )
        accepted.validate_terminal_resources(
            terminal, packet["resource_bounds"]["ranks"][shard]
        )
        model = load_json(run / "model.json")
        require(
            model["arm"] == ARM
            and model["exported_adapter"] == adapter
            and model["training_receipt"] == receipt
            and model["cold_check"] == packet["candidate"]["cold_check"]
            and model["identity"]["model_identity"]["adapter"]["adapter_path"]
            == adapter["root"]
            and model["identity"]["model_identity"]["adapter"]["merged_adapters"] == [],
            "cold loaded D composition",
        )
        rows = load_jsonl(run / "natural-rows.jsonl")
        require(
            [row["example_id"] for row in rows] == packet["eval_shards"][shard],
            "D shard order",
        )
        for row in rows:
            old._cold_natural(row, frozen[row["example_id"]], tokenizer)
        natural.extend(rows)
        for row in load_jsonl(run / "conditional-rows.jsonl"):
            require(
                row.get("schema") == CONDITIONAL_SCHEMA and row.get("arm") == ARM,
                "D conditional schema/arm",
            )
            job = jobs[(shard, row["job_id"])]
            old.validate_conditional_artifact(row, job)
            example_id = next(
                case["example_id"]
                for case in packet["conditional_cases"]
                if case["case_id"] == job.case_id
            )
            case = frozen[example_id]
            require(
                row["request_id"] == example_id
                and row["adapter_fingerprint"] == adapter["fingerprint"]
                and tokenizer.decode(row["action_ids"], skip_special_tokens=False)
                == row["text"],
                "D conditional request/adapter/token text",
            )
            require(
                native_record(row["text"], case["case"], case["golden"], row["stop_reason"])
                == row["parsed"],
                "cold D conditional parser",
            )
            conditional.append(row)
    validate_d_natural_rows(natural, packet)
    validate_d_conditional_rows(conditional, packet)
    total_tokens = sum(row["new_tokens"] for row in terminals)
    require(
        total_tokens
        == sum(len(row["action_ids"]) for row in natural)
        + sum(len(row["free_ids"]) for row in conditional)
        and total_tokens <= 120_000,
        "D global generated-token counter/bound",
    )
    resources = {
        "schema": "positive_progress_matched_endpoint.resources.v1",
        "arm": ARM,
        "model_loads": sum(row["model_loads"] for row in terminals),
        "continuations": sum(row["continuations"] for row in terminals),
        "new_tokens": total_tokens,
        "model_forwards": sum(row["model_forwards"] for row in terminals),
        "image_forwards": sum(row["image_forwards"] for row in terminals),
        "allocated_gpu_seconds": sum(row["elapsed_seconds"] for row in terminals),
        "shards": terminals,
    }
    require(
        resources["model_loads"] == 8
        and resources["continuations"] == resources["image_forwards"] == 390
        and resources["new_tokens"] == resources["model_forwards"],
        "D global endpoint counters",
    )
    result = reduce_endpoint(packet, natural, conditional)
    outputs = {
        "consumer.json": natural,
        "conditional-consumer.json": conditional,
        "result.json": result,
        "resources.json": resources,
    }
    if verify:
        require(
            all(load_json(out / name) == value for name, value in outputs.items()),
            "cold merged D endpoint differs",
        )
    else:
        for name, value in outputs.items():
            publish(out / name, value)
    return {
        "status": "verified" if verify else "merged",
        "arm": ARM,
        "natural": len(natural),
        "conditional": len(conditional),
        "new_tokens": total_tokens,
        "consumer_sha256": file_hash(out / "consumer.json"),
        "result_sha256": file_hash(out / "result.json"),
    }


def proposed_commands_payload() -> dict[str, Any]:
    script = str(Path(__file__).resolve())
    receipt = str(RAW_ROOT / "training/receipt.json")
    cold = str(RAW_ROOT / "training/cold-check.json")
    packet = str(PREPARATION / "packet.json")
    endpoint_root = str(RAW_ROOT / "endpoint-D")
    return {
        "schema": "positive_progress_matched_endpoint.proposed_commands.v1",
        "status": "not_executable_until_actual_D17_receipt_and_cold_are_root_accepted",
        "unverified_pending_bindings": {"training_receipt": receipt, "cold_check": cold},
        "commands_after_binding_and_separate_gpu_grant": {
            "prepare": [
                "python", script, "prepare",
                "--training-receipt", receipt,
                "--cold-check", cold,
                "--out", packet,
            ],
            "launch_once": [
                "python", script, "launch", "--packet", packet, "--out", endpoint_root,
            ],
            "merge": [
                "python", script, "merge", "--packet", packet, "--out", endpoint_root,
            ],
            "verify": [
                "python", script, "verify", "--packet", packet, "--out", endpoint_root,
            ],
        },
        "retry_policy": "none",
    }


def cpu_preflight_payload() -> dict[str, Any]:
    packet = load_old_packet()
    selection = validate_selection()
    retained = validate_retained_ledgers(packet)
    base = packet_base(packet)
    return {
        "schema": "positive_progress_matched_endpoint.cpu_preflight.v1",
        "status": "passed_pending_actual_D17_bindings",
        "producer": source(Path(__file__).resolve(), file_hash(Path(__file__).resolve())),
        "tests": source(
            Path(__file__).with_name("tests") / "test_positive_progress_matched_endpoint.py",
            file_hash(Path(__file__).with_name("tests") / "test_positive_progress_matched_endpoint.py"),
        ),
        "selection": source(SELECTION, SELECTION_SHA256),
        "selection_summary": {
            "optimizer_updates": selection["optimizer_updates"],
            "adapter_state_sha256": selection["selected"]["adapter_state_sha256"],
            "sum_positive_nll": selection["selected"]["sum_positive_nll"],
            "target_C32_sum_positive_nll": selection["target_C32_sum_positive_nll"],
            "relative_sum_residual": selection["relative_sum_residual"],
        },
        "retained": retained,
        "resource_bounds": base["resource_bounds"],
        "verified_scope": (
            "Real selection and retained A/C/Stable cohort, C/A labels and hashes, exact "
            "order/prompt/media/grid/native scores, conditional partition, policy and allocation."
        ),
        "excluded_scope": "No D receipt, adapter, model load, generation, rendering, or endpoint result.",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    pending_parser = subparsers.add_parser("prepare-pending")
    pending_parser.add_argument("--out", type=Path, default=PREPARATION / "pending-bindings.json")
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--training-receipt", type=Path, required=True)
    prepare_parser.add_argument("--cold-check", type=Path, required=True)
    prepare_parser.add_argument("--out", type=Path, default=PREPARATION / "packet.json")
    for command in ("launch", "merge", "verify"):
        child = subparsers.add_parser(command)
        child.add_argument("--packet", type=Path, required=True)
        child.add_argument("--out", type=Path, required=True)
    execute_parser = subparsers.add_parser("execute")
    execute_parser.add_argument("--packet", type=Path, required=True)
    execute_parser.add_argument("--out", type=Path, required=True)
    execute_parser.add_argument("--shard", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-pending":
        require(args.out.is_absolute() and not args.out.exists(), "fresh absolute pending output")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        publish(args.out, pending_payload())
        publish(args.out.parent / "cpu-preflight.json", cpu_preflight_payload())
        publish(args.out.parent / "proposed-commands.json", proposed_commands_payload())
        print(args.out)
    elif args.command == "prepare":
        prepare_final_packet(args.training_receipt, args.cold_check, args.out)
        print(args.out)
    elif args.command == "execute":
        execute(args.packet, args.out, args.shard)
    elif args.command == "launch":
        print(json.dumps(launch(args.packet, args.out), indent=2))
    else:
        print(json.dumps(merge(args.packet, args.out, verify=args.command == "verify"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
