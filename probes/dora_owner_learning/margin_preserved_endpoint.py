"""One explicitly-C endpoint for the frozen margin-preserved positive branch.

The accepted positive-branch endpoint is reused as an immutable source of the
384 cases, prompt/media identities, parser, scorer, and conditional partition.
This module does not relabel or rerun A.  A final packet can only be made from
an actual, verified full-C32 receipt and its matching cold-check artifact.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
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


WORKTREE = Path("/data/CoordExp/.worktrees/research-probes")
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

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
from src.artifacts import publish_json_exclusive  # noqa: E402
from probes.dora_owner_learning import positive_branch_endpoint


SOURCE_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-11-positive-branch-vs-repeat-event"
)
OLD_PRODUCER = Path(positive_branch_endpoint.__file__)
# Current maintained engine identity; original producer bytes stay in the source archive.
OLD_PRODUCER_SHA256 = "6269a0749d89469dfba0f18a2ddbef774d26fe37ec110c2f697560b2dbf3102b"
OLD_PACKET = SOURCE_ROOT / "endpoint-preparation/packet.json"
OLD_PACKET_SHA256 = "560006e73f3f0fc416e7d58751fe96c936aba9478fb6b320ab6118db2bcd5053"
RETAINED_A_ROOT = SOURCE_ROOT / "endpoint-A"
RETAINED_A_CONSUMER = RETAINED_A_ROOT / "consumer.json"
RETAINED_A_CONSUMER_SHA256 = "d61454f058793fd40338f19d1ebeac9023a3889df58912c3b960224dd8b91a37"
RETAINED_A_CONDITIONAL = RETAINED_A_ROOT / "conditional-consumer.json"
RETAINED_A_CONDITIONAL_SHA256 = "f1fe1a515427d9cdfcf2dc653d01b145daa0e7e6060c8932328ddad04d1f08c7"
RETAINED_A_CONDITIONAL_REDUCTION = RETAINED_A_ROOT / "conditional-reduction.json"
RETAINED_A_CONDITIONAL_REDUCTION_SHA256 = "5372fc3022524b55633cc71e6203e4805a43cb144ee03eb37863aa274d26d11e"
RETAINED_A_REDUCTION = RETAINED_A_ROOT / "reduction.json"
RETAINED_A_REDUCTION_SHA256 = "85a52bdf762db624b52f88594d3feba088aea89b929ae577a8e90e606b0b38b4"
RETAINED_A_RESOURCES = RETAINED_A_ROOT / "resources.json"
RETAINED_A_RESOURCES_SHA256 = "ef489eef52124e4a95a8ba615225bc88fc34f498637b808f8b183825d988a751"
TRAINER_SOURCE = WORKTREE / "probes/dora_owner_learning/margin_preserved_train.py"

OUTPUT_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-11-margin-preserved-positive-branch"
)
PREPARATION_ROOT = OUTPUT_ROOT / "endpoint-preparation"

ARM = "C"
WORLD_SIZE = 8
NATURAL_PER_RANK = 48
EOS = 151645
PAD = 151643
CAP = 3084
MAX_GLOBAL_GENERATED_TOKENS = 120_000
MAX_GENERATED_TOKENS_PER_WORKER = 15_000
MAX_WORKER_SECONDS = 1500
MAX_CUDA_BYTES = 24 * 1024**3
MAX_RSS_BYTES = 24 * 1024**3

PACKET_SCHEMA = "margin_preserved_endpoint.packet.v1"
NATURAL_SCHEMA = "margin_preserved_endpoint.natural.v1"
CONDITIONAL_SCHEMA = "margin_preserved_endpoint.conditional.v1"
TERMINAL_SCHEMA = "margin_preserved_endpoint.terminal.v1"
OUTER_EXIT_SCHEMA = "margin_preserved_endpoint.outer_exit.v1"
RESULT_SCHEMA = "margin_preserved_endpoint.result.v1"


class ForwardBudgetExceeded(RuntimeError):
    """Raised by the live pre-forward hook before a worker exceeds its allocation."""


def digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def publish(path: Path, value: Any) -> None:
    publish_json_exclusive(path, value)
    require(load_json(path) == value, f"cold publication mismatch: {path}")


def source(path: Path, sha256: str) -> dict[str, Any]:
    require(path.is_file() and file_hash(path) == sha256, f"source changed: {path}")
    return {"path": str(path), "sha256": sha256, "size_bytes": path.stat().st_size}


def old_endpoint() -> Any:
    source(OLD_PRODUCER, OLD_PRODUCER_SHA256)
    return positive_branch_endpoint


def load_old_packet() -> dict[str, Any]:
    source(OLD_PACKET, OLD_PACKET_SHA256)
    return old_endpoint().read_retained_packet(
        OLD_PACKET, expected_sha256=OLD_PACKET_SHA256,
        archive_manifest=Path("/data/CoordExp/docs/history/output-sources/2026-09-21/manifest.json"),
    )


def _flattened_identity(packet: Mapping[str, Any]) -> list[tuple[int, str]]:
    return [
        (rank, str(example_id))
        for rank, shard in enumerate(packet["eval_shards"])
        for example_id in shard
    ]


def validate_natural_rows(
    rows: Sequence[Mapping[str, Any]],
    packet: Mapping[str, Any],
    *,
    arm: str,
    retained: bool,
) -> None:
    require(arm in ("A", ARM), "natural validation supports retained A or actual C")
    expected = _flattened_identity(packet)
    observed = [(int(row["shard"]), str(row["example_id"])) for row in rows]
    require(
        len(rows) == len({example_id for _, example_id in observed}) == 384,
        "exact384 natural identity",
    )
    require(observed == expected, "exact384 ordered shard identity")
    label = "retained A arm" if retained else "actual C arm"
    require(all(row.get("arm") == arm for row in rows), f"{label} differs")
    if retained:
        require(
            all(row.get("packet_sha256") == OLD_PACKET_SHA256 for row in rows),
            "retained A source packet identity",
        )
    else:
        require(
            all(row.get("schema") == NATURAL_SCHEMA for row in rows),
            "actual C natural schema",
        )

    frozen = indexed(packet["eval_records"], "example_id")
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
            and int(row["shard"])
            == next(i for i, shard in enumerate(packet["eval_shards"]) if row["example_id"] in shard)
            and row["image_id"] == case["image_id"]
            and row["split"] == case["split"],
            "natural request/scored order differs",
        )
        require(
            all(row.get(key) == value for key, value in identity.items())
            and row.get("batch_identity_sha256") == digest(identity),
            "natural prompt/media/grid differs",
        )
        card = score(
            row["parsed"], seed=None, length=len(row["action_ids"]), stop=row["stop_reason"]
        )
        require(
            row["score"] == card and row["overlap_counts"] == overlap_counts(row["parsed"]),
            "natural score differs",
        )


def validate_retained_baselines(packet: Mapping[str, Any]) -> dict[str, Any]:
    bindings = (
        (RETAINED_A_CONSUMER, RETAINED_A_CONSUMER_SHA256),
        (RETAINED_A_CONDITIONAL, RETAINED_A_CONDITIONAL_SHA256),
        (RETAINED_A_CONDITIONAL_REDUCTION, RETAINED_A_CONDITIONAL_REDUCTION_SHA256),
        (RETAINED_A_REDUCTION, RETAINED_A_REDUCTION_SHA256),
        (RETAINED_A_RESOURCES, RETAINED_A_RESOURCES_SHA256),
    )
    for path, sha256 in bindings:
        source(path, sha256)
    rows = load_json(RETAINED_A_CONSUMER)
    validate_natural_rows(rows, packet, arm="A", retained=True)
    conditional = load_json(RETAINED_A_CONDITIONAL)
    conditional_reduction = load_json(RETAINED_A_CONDITIONAL_REDUCTION)
    require(
        len(conditional) == 6
        and all(row.get("arm") == "A" for row in conditional)
        and conditional_reduction.get("arm") == "A"
        and len(conditional_reduction.get("cells", [])) == 6,
        "retained A conditional6 identity",
    )
    by_image = {str(row["image_id"]): row for row in rows}
    positive = {image_id: by_image[image_id]["score"]["50"]["tp"] for image_id in ("351017", "417044", "477415")}
    require(positive == {"351017": 12, "417044": 11, "477415": 16}, "retained A positive TP50 baselines")
    diagnostic = by_image["39654"]
    diagnostic_summary = {
        "split": diagnostic["split"],
        "tp50": diagnostic["score"]["50"]["tp"],
        "strict_repeats": diagnostic["score"]["strict_repeats"],
        "parser_drops": diagnostic["score"]["parser_drops"],
        "cap": diagnostic["score"]["cap"],
    }
    require(
        diagnostic_summary
        == {"split": "dev128", "tp50": 0, "strict_repeats": 194, "parser_drops": 144, "cap": 1},
        "retained A 39654 diagnostic",
    )
    return {
        "natural_rows": 384,
        "arm": "A",
        "split_counts": {"reference56": 56, "train256": 256, "dev128": 128},
        "positive_A_tp50": positive,
        "diagnostic_39654": diagnostic_summary,
    }


def build_conditional_jobs(packet: Mapping[str, Any]) -> list[Any]:
    old = old_endpoint()
    cases = packet["conditional_cases"]
    require(len(cases) == 3, "exact three conditional cases")
    jobs = []
    for index, case in enumerate(cases):
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
    require([job.rank for job in jobs] == list(range(6)), "conditional rank partition")
    return jobs


def rank_bounds(jobs: Sequence[Any]) -> list[dict[str, int]]:
    by_rank = {job.rank: job for job in jobs}
    return [
        {
            "rank": rank,
            "natural_calls": NATURAL_PER_RANK,
            "conditional_calls": int(rank in by_rank),
            "max_calls": NATURAL_PER_RANK + int(rank in by_rank),
            "max_generated_tokens": MAX_GENERATED_TOKENS_PER_WORKER,
            "max_model_forwards": MAX_GENERATED_TOKENS_PER_WORKER,
            "max_image_forwards": NATURAL_PER_RANK + int(rank in by_rank),
            "max_model_loads": 1,
            "max_worker_seconds": MAX_WORKER_SECONDS,
            "max_cuda_bytes": MAX_CUDA_BYTES,
            "max_rss_bytes": MAX_RSS_BYTES,
        }
        for rank in range(WORLD_SIZE)
    ]


def packet_base(old_packet: Mapping[str, Any]) -> dict[str, Any]:
    require(dict(old_packet) == load_old_packet(), "retained endpoint packet content changed")
    validate_retained_baselines(old_packet)
    jobs = build_conditional_jobs(old_packet)
    bounds = rank_bounds(jobs)
    require(
        len(bounds) == WORLD_SIZE
        and sum(row["max_generated_tokens"] for row in bounds)
        == MAX_GLOBAL_GENERATED_TOKENS,
        "eight worker allocations must exactly sum to the global token bound",
    )
    return {
        "schema": PACKET_SCHEMA,
        "status": "pending_actual_C32_receipt_and_cold_check",
        "arm": ARM,
        "model": old_packet["model"],
        "config": old_packet["config"],
        "eval_records": old_packet["eval_records"],
        "eval_shards": old_packet["eval_shards"],
        "protected_targets": old_packet["protected_targets"],
        "conditional_cases": old_packet["conditional_cases"],
        "conditional_jobs": [job.to_dict() for job in jobs],
        "policy": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": 1.0,
            "use_model_defaults": False,
            "trace": "none",
            "eos_token_id": EOS,
            "pad_token_id": PAD,
            "total_action_cap": CAP,
        },
        "resource_bounds": {
            "workers": WORLD_SIZE,
            "natural_calls": 384,
            "conditional_calls": 6,
            "total_calls": 390,
            "model_loads": 8,
            "global_max_generated_tokens": MAX_GLOBAL_GENERATED_TOKENS,
            "ranks": bounds,
        },
        "panels": {
            "reference56": {"images": 56, "role": "training protection"},
            "train256": {"images": 256, "role": "exposed train-side read"},
            "dev128": {"images": 128, "role": "already exposed development screen; not holdout"},
        },
        "claim_boundary": (
            "One actual C exposed384 natural plus fixed conditional6 read versus immutable "
            "Stable50 and retained A; no new A run, visual acceptance, held-out, or promotion claim."
        ),
    }


def pending_payload() -> dict[str, Any]:
    old_packet = load_old_packet()
    retained = validate_retained_baselines(old_packet)
    base = packet_base(old_packet)
    return {
        "schema": "margin_preserved_endpoint.pending_bindings.v1",
        "status": "pending_actual_C32_receipt_and_cold_check",
        "unresolved": ["actual_C32_training_receipt", "actual_C32_cold_check"],
        "verified": {
            "old_endpoint_producer": source(OLD_PRODUCER, OLD_PRODUCER_SHA256),
            "old_endpoint_packet": source(OLD_PACKET, OLD_PACKET_SHA256),
            "retained_A_consumer": source(RETAINED_A_CONSUMER, RETAINED_A_CONSUMER_SHA256),
            "retained_A_conditional": source(RETAINED_A_CONDITIONAL, RETAINED_A_CONDITIONAL_SHA256),
            "retained_A_conditional_reduction": source(
                RETAINED_A_CONDITIONAL_REDUCTION, RETAINED_A_CONDITIONAL_REDUCTION_SHA256
            ),
            "retained_A_reduction": source(RETAINED_A_REDUCTION, RETAINED_A_REDUCTION_SHA256),
            "retained_A_resources": source(RETAINED_A_RESOURCES, RETAINED_A_RESOURCES_SHA256),
        },
        "retained_A_checks": retained,
        "frozen_endpoint": {
            "arm": ARM,
            "natural": 384,
            "conditional": 6,
            "resource_bounds": base["resource_bounds"],
            "policy": base["policy"],
            "panels": base["panels"],
        },
        "final_packet_rule": (
            "Do not publish packet.json until both actual full-C32 receipt and its cold check "
            "pass native validators; this record does not identify or verify a candidate adapter."
        ),
    }


def proposed_commands_payload() -> dict[str, Any]:
    script = str(Path(__file__).resolve())
    receipt = str(OUTPUT_ROOT / "full-C/receipt.json")
    cold = str(OUTPUT_ROOT / "full-C/cold-check.json")
    packet = str(PREPARATION_ROOT / "packet.json")
    endpoint_out = str(OUTPUT_ROOT / "endpoint-C")
    return {
        "schema": "margin_preserved_endpoint.proposed_commands.v1",
        "status": "not_executable_until_actual_C32_receipt_and_cold_check_are_lead_accepted",
        "unverified_pending_bindings": {
            "training_receipt": receipt,
            "cold_check": cold,
        },
        "commands_after_binding_and_separate_root_gpu_grant": {
            "prepare_final_packet_cpu": [
                "python", script, "prepare",
                "--training-receipt", receipt,
                "--cold-check", cold,
                "--out", packet,
            ],
            "launch_once_all8": [
                "python", script, "launch", "--packet", packet, "--out", endpoint_out,
            ],
            "merge_cpu": [
                "python", script, "merge", "--packet", packet, "--out", endpoint_out,
            ],
            "verify_cpu": [
                "python", script, "verify", "--packet", packet, "--out", endpoint_out,
            ],
        },
        "retry_policy": "none",
    }


def cpu_preflight_payload() -> dict[str, Any]:
    packet = load_old_packet()
    retained = validate_retained_baselines(packet)
    base = packet_base(packet)
    return {
        "schema": "margin_preserved_endpoint.cpu_preflight.v1",
        "status": "passed_pending_actual_C32_bindings",
        "producer": source(Path(__file__).resolve(), file_hash(Path(__file__).resolve())),
        "test_source": source(
            Path(__file__).with_name("tests") / "test_margin_preserved_endpoint.py",
            file_hash(Path(__file__).with_name("tests") / "test_margin_preserved_endpoint.py"),
        ),
        "training_verifier_source": source(TRAINER_SOURCE, file_hash(TRAINER_SOURCE)),
        "old_endpoint_producer_sha256": OLD_PRODUCER_SHA256,
        "old_endpoint_packet_sha256": OLD_PACKET_SHA256,
        "retained_A_consumer_sha256": RETAINED_A_CONSUMER_SHA256,
        "retained_A_checks": retained,
        "natural_order_sha256": digest(_flattened_identity(packet)),
        "resource_bounds": base["resource_bounds"],
        "verified_scope": (
            "Real source cohort, retained A arm/hash/order, prompt-media-grid identities, "
            "native scores, conditional forced/free partition, policy, and endpoint bounds."
        ),
        "excluded_scope": "No C receipt, C adapter, model load, generation, or GPU endpoint is claimed.",
    }


def validate_c_training_metadata(
    receipt: Mapping[str, Any], cold: Mapping[str, Any]
) -> None:
    require(receipt.get("arm") == ARM and cold.get("arm") == ARM, "only C training/cold identities are accepted")
    require(
        receipt.get("schema") == "margin_preserved_train.receipt.v1"
        and receipt.get("status") == "completed"
        and receipt.get("scientific_status") == "candidate"
        and receipt.get("mode") == "full"
        and receipt.get("updates") == 32
        and receipt.get("margin_weight") == 10.0
        and receipt.get("stop_reason") == "fixed_32_updates",
        "actual C32 full receipt required",
    )
    require(
        cold.get("schema") == "margin_preserved_train.cold_check.v1"
        and cold.get("status") == "passed"
        and cold.get("mode") == "full",
        "actual C32 cold check required",
    )
    require(cold.get("saved_adapter") == receipt.get("saved_adapter"), "C cold/saved adapter differs")
    observed = cold.get("positive_scores", {})
    expected = receipt.get("final_live_positive_scores", {})
    require(set(observed) == set(expected) and len(observed) == 3, "C cold/final positive identities differ")
    for candidate_id in expected:
        require(
            observed[candidate_id]["token_count"] == expected[candidate_id]["token_count"]
            and observed[candidate_id]["argmax_target_tokens"]
            == expected[candidate_id]["argmax_target_tokens"],
            "C cold/final positive discrete scores differ",
        )
        require(
            all(
                abs(observed[candidate_id][key] - expected[candidate_id][key]) <= 1e-5
                for key in ("sum_logprob", "mean_logprob", "mean_target_margin", "min_target_margin")
            ),
            "C cold/final positive numeric scores differ",
        )
    require(
        cold.get("model_loads") == 1
        and cold.get("score_forwards") == 3
        and all(
            abs(value) <= 1e-5
            for differences in cold.get("live_score_deltas", {}).values()
            for value in differences.values()
        ),
        "C cold score counters/deltas differ",
    )


def verify_training_receipt(receipt_path: Path) -> dict[str, Any]:
    """Use the trainer's package import so its relative imports retain context."""

    from probes.dora_owner_learning import margin_preserved_train as trainer

    require(receipt_path.is_absolute(), "training receipt path must be absolute")
    source(TRAINER_SOURCE, file_hash(TRAINER_SOURCE))
    observed = trainer.verify_receipt(receipt_path.parent)
    require(observed == load_json(receipt_path), "trainer receipt verifier/file differs")
    return observed


def validate_c_training(
    receipt_path: Path, cold_path: Path, old_packet: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    require(receipt_path.is_absolute() and cold_path.is_absolute(), "C binding paths must be absolute")
    receipt, cold = load_json(receipt_path), load_json(cold_path)
    validate_c_training_metadata(receipt, cold)
    require(
        Path(cold["training_receipt"]["path"]).resolve() == receipt_path.resolve()
        and cold["training_receipt"]["sha256"] == file_hash(receipt_path),
        "C cold/training receipt identity differs",
    )
    require(verify_training_receipt(receipt_path) == receipt, "C sealed receipt verifier differs")
    composition = receipt["composition"]
    require(
        composition["base_model_path"] == old_packet["model"]["base_model_path"]
        and composition["source_embedding"] == old_packet["model"]["source_embedding"]
        and composition["source_adapter"] == old_packet["model"]["current_adapter"]
        and composition["unmerged"] is True
        and composition["dtype"] == "fp32"
        and composition["attention_implementation"] == "sdpa",
        "C training composition differs from retained Stable50",
    )
    adapter = receipt["saved_adapter"]
    require(
        inspect_dora_adapter_payload(adapter["root"], composition["base_model_path"]) == adapter,
        "C adapter semantic fingerprint differs",
    )
    return receipt, cold


def build_packet_payload(receipt_path: Path, cold_path: Path) -> dict[str, Any]:
    old_packet = load_old_packet()
    receipt, cold = validate_c_training(receipt_path, cold_path, old_packet)
    result = packet_base(old_packet)
    result.update(
        status="ready_for_one_C_endpoint_requires_root_gpu_grant",
        sources={
            "producer": source(Path(__file__).resolve(), file_hash(Path(__file__).resolve())),
            "C_training_verifier": source(TRAINER_SOURCE, file_hash(TRAINER_SOURCE)),
            "old_endpoint_producer": source(OLD_PRODUCER, OLD_PRODUCER_SHA256),
            "old_endpoint_packet": source(OLD_PACKET, OLD_PACKET_SHA256),
            "retained_A_consumer": source(RETAINED_A_CONSUMER, RETAINED_A_CONSUMER_SHA256),
            "retained_A_conditional": source(RETAINED_A_CONDITIONAL, RETAINED_A_CONDITIONAL_SHA256),
            "retained_A_conditional_reduction": source(
                RETAINED_A_CONDITIONAL_REDUCTION, RETAINED_A_CONDITIONAL_REDUCTION_SHA256
            ),
            "retained_A_reduction": source(RETAINED_A_REDUCTION, RETAINED_A_REDUCTION_SHA256),
            "retained_A_resources": source(RETAINED_A_RESOURCES, RETAINED_A_RESOURCES_SHA256),
            "C_training_receipt": source(receipt_path, file_hash(receipt_path)),
            "C_cold_check": source(cold_path, file_hash(cold_path)),
        },
        candidate={
            "arm": ARM,
            "training_receipt": {"path": str(receipt_path), "sha256": file_hash(receipt_path)},
            "cold_check": {"path": str(cold_path), "sha256": file_hash(cold_path)},
            "saved_adapter": receipt["saved_adapter"],
            "cold_model_identity": cold["model_identity"],
            "final_margin_signal": receipt["margin_signal"]["final_post_update"],
        },
    )
    return result


def validate_packet(packet: Mapping[str, Any]) -> None:
    require(packet.get("schema") == PACKET_SCHEMA and packet.get("arm") == ARM, "only C endpoint packet accepted")
    require(
        packet.get("status") == "ready_for_one_C_endpoint_requires_root_gpu_grant",
        "final C endpoint packet is not ready",
    )
    for item in packet["sources"].values():
        source(Path(item["path"]), item["sha256"])
    expected = build_packet_payload(
        Path(packet["candidate"]["training_receipt"]["path"]),
        Path(packet["candidate"]["cold_check"]["path"]),
    )
    require(dict(packet) == expected, "C endpoint packet differs from frozen bindings")


def prepare_final_packet(receipt_path: Path, cold_path: Path, out: Path) -> dict[str, Any]:
    require(out.is_absolute(), "final endpoint packet output must be absolute")
    payload = build_packet_payload(receipt_path, cold_path)
    require(not out.exists(), "occupied final C endpoint packet")
    out.parent.mkdir(parents=True, exist_ok=True)
    publish(out, payload)
    return payload


def runtime_conditional_jobs(packet: Mapping[str, Any], rank: int | None = None) -> list[Any]:
    old = old_endpoint()
    rows = packet["conditional_jobs"]
    if rank is not None:
        rows = [row for row in rows if row["rank"] == rank]
    jobs = [old.ConditionalJob(**dict(row)) for row in rows]
    require(all(job.arm == ARM for job in jobs), "only C conditional jobs accepted")
    return jobs


def _checked_action(ids: list[int], stop: str, budget: int) -> None:
    require(ids and len(ids) <= budget and PAD not in ids and EOS not in ids[:-1], "action token corruption")
    require(
        (stop == "im_end" and ids[-1] == EOS)
        or (stop == "length" and len(ids) == budget and EOS not in ids),
        "action terminal/budget corruption",
    )


def validate_terminal_resources(terminal: Mapping[str, Any], bound: Mapping[str, int]) -> None:
    require(terminal["model_loads"] == bound["max_model_loads"] == 1, "endpoint model-load bound")
    require(
        terminal["natural_continuations"] == bound["natural_calls"]
        and terminal["conditional_continuations"] == bound["conditional_calls"]
        and terminal["continuations"] == terminal["image_forwards"] == bound["max_calls"],
        "endpoint continuation/image count",
    )
    require(
        terminal["new_tokens"] == terminal["model_forwards"]
        and terminal["new_tokens"] <= bound["max_generated_tokens"]
        and terminal["model_forwards"] <= bound["max_model_forwards"],
        "endpoint token/model-forward count",
    )
    require(0 <= terminal["elapsed_seconds"] <= bound["max_worker_seconds"], "endpoint wall bound")
    require(
        0 <= terminal["peak_cuda_allocated_bytes"] <= bound["max_cuda_bytes"]
        and 0 <= terminal["peak_cuda_reserved_bytes"] <= bound["max_cuda_bytes"],
        "endpoint CUDA bound",
    )
    require(0 <= terminal["peak_rss_bytes"] <= bound["max_rss_bytes"], "endpoint RSS bound")


def forward_budget_hook(counters: dict[str, int]) -> Any:
    """Count an admitted model forward; reject the next one at the hard cap."""

    def hook(*_: Any) -> None:
        if counters["model"] >= MAX_GENERATED_TOKENS_PER_WORKER:
            raise ForwardBudgetExceeded(
                "worker reached 15000 model forwards; withheld work before next forward"
            )
        counters["model"] += 1

    return hook


def execute(packet_path: Path, out: Path, shard: int) -> None:
    import torch
    from probes.dora_owner_learning.runtime import load_policy
    from probes.source_rweak_row_cross.run import build_requests, native_record
    from src.config.inference import InferConfig
    from src.qwen.generation import NativeGenerationPolicy, generate_continuations
    from src.qwen.native import prepare_native_inputs

    require(packet_path.is_absolute() and out.is_absolute(), "execution paths must be absolute")
    packet = load_json(packet_path)
    validate_packet(packet)
    require(
        0 <= shard < WORLD_SIZE
        and os.environ.get("CUDA_VISIBLE_DEVICES") == str(shard)
        and torch.cuda.device_count() == 1,
        "single assigned GPU identity",
    )
    adapter = packet["candidate"]["saved_adapter"]
    receipt = packet["candidate"]["training_receipt"]
    run = out / f"shard-{shard}"
    require(not run.exists(), "occupied endpoint shard")
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
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("endpoint worker timeout")))
        signal.alarm(MAX_WORKER_SECONDS)
        config = checkpoint_config(InferConfig.model_validate(packet["config"]), adapter["root"])
        # No-argument reset is required before CUDA initialization in torch 2.9.1.
        # It also makes the recorded peak cover model loading and the full worker lifecycle.
        torch.cuda.reset_peak_memory_stats()
        qwen, identity = load_policy(config, device=torch.device("cuda:0"))
        terminal["model_loads"] = 1
        live = identity["model_identity"]["adapter"]
        require(
            live["adapter_path"] == adapter["root"] and live["merged_adapters"] == [],
            "stale/merged loaded C adapter",
        )
        require(
            identity["effective_settings"]["observed_model_dtype"]["parameter_dtype_names"]
            == ["torch.float32"]
            and identity["effective_settings"]["observed_attn_implementation"] == "sdpa",
            "live endpoint numerics differ",
        )
        require(
            inspect_dora_adapter_payload(live["adapter_path"], packet["model"]["base_model_path"])
            == adapter,
            "live C adapter fingerprint differs",
        )
        publish(
            run / "model.json",
            {
                "arm": ARM,
                "identity": identity,
                "exported_adapter": adapter,
                "training_receipt": receipt,
                "cold_check": packet["candidate"]["cold_check"],
            },
        )
        publish(run / "config.json", config.model_dump(mode="json"))
        qwen.model.eval()
        for parameter in qwen.model.parameters():
            parameter.requires_grad_(False)
        counters = {"model": 0, "image": 0}
        handles.append(qwen.model.register_forward_pre_hook(forward_budget_hook(counters)))
        visuals = [module for name, module in qwen.model.named_modules() if name.endswith("visual")]
        require(len(visuals) == 1, "ambiguous visual module")
        handles.append(visuals[0].register_forward_pre_hook(lambda *_: counters.__setitem__("image", counters["image"] + 1)))
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
            require(list(requests[0].expected_token_ids) == frozen["prompt_token_ids"], "request prompt differs")
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
                "live prompt/media/grid differs",
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
                require(generated.request_id == example_id, "natural request association differs")
                ids = list(generated.token_ids)
                _checked_action(ids, generated.stop_reason, CAP)
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

        old = old_endpoint()
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
                require(generated.request_id == example_id, "conditional request association differs")
                free_ids = list(generated.token_ids)
                _checked_action(free_ids, generated.stop_reason, job.budget)
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
                require(row["arm"] == ARM, "actual C conditional arm")
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
            "runtime exact continuation counts",
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
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        )
        if terminal["status"] == "completed":
            try:
                validate_terminal_resources(
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
    require(packet_path.is_absolute() and out.is_absolute(), "launcher paths must be absolute")
    validate_packet(load_json(packet_path))
    launcher = out / "launcher"
    require(not launcher.exists() and not any((out / f"shard-{i}").exists() for i in range(WORLD_SIZE)), "occupied C endpoint launch")
    launcher.mkdir(parents=True)
    command_base = [sys.executable, str(Path(__file__).resolve()), "execute", "--packet", str(packet_path), "--out", str(out)]
    processes: list[tuple[int, subprocess.Popen[Any], Any, list[str]]] = []
    for shard in range(WORLD_SIZE):
        command = [*command_base, "--shard", str(shard)]
        log = (launcher / f"shard-{shard}.log").open("x")
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(shard)
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        processes.append((shard, process, log, command))
    group_launch = {
        "schema": "margin_preserved_endpoint.group_launch.v1",
        "packet": {"path": str(packet_path), "sha256": file_hash(packet_path)},
        "output_root": str(out),
        "workers": [
            {"shard": shard, "pid": process.pid, "command": command, "cuda_visible_devices": str(shard)}
            for shard, process, _, command in processes
        ],
        "automatic_retry": False,
    }
    publish(launcher / "group-launch.json", group_launch)
    exits = []
    for shard, process, log, command in processes:
        exit_code = process.wait()
        log.close()
        record = {
            "schema": OUTER_EXIT_SCHEMA,
            "shard": shard,
            "pid": process.pid,
            "exit_code": exit_code,
            "command": command,
        }
        publish(launcher / f"shard-{shard}-outer-exit.json", record)
        exits.append(record)
    completion = {
        "schema": "margin_preserved_endpoint.group_completion.v1",
        "status": "completed" if all(row["exit_code"] == 0 for row in exits) else "failed",
        "outer_exits": exits,
        "automatic_retry": False,
    }
    publish(launcher / "group-completion.json", completion)
    require(completion["status"] == "completed", "one or more C endpoint workers failed; no automatic retry")
    return completion


def validate_outer_exits(launcher: Path) -> list[dict[str, Any]]:
    paths = sorted(launcher.glob("shard-*-outer-exit.json"))
    rows = [load_json(path) for path in paths]
    require(
        len(rows) == WORLD_SIZE
        and {row.get("shard") for row in rows} == set(range(WORLD_SIZE))
        and all(row.get("schema") == OUTER_EXIT_SCHEMA and row.get("exit_code") == 0 for row in rows),
        "complete successful eight outer exits required",
    )
    completion = load_json(launcher / "group-completion.json")
    require(
        completion.get("status") == "completed"
        and completion.get("outer_exits") == sorted(rows, key=lambda row: row["shard"])
        and completion.get("automatic_retry") is False,
        "group completion differs from outer exits",
    )
    return sorted(rows, key=lambda row: row["shard"])


def validate_conditional_rows(
    rows: Sequence[Mapping[str, Any]], packet: Mapping[str, Any]
) -> None:
    old = old_endpoint()
    jobs = {
        (job.rank, f"{job.kind}:{job.candidate_id}"): job
        for job in runtime_conditional_jobs(packet)
    }
    require(
        len(rows) == 6
        and [(int(row["rank"]), str(row["job_id"])) for row in rows]
        == sorted(jobs),
        "actual C conditional6 ordered identity",
    )
    frozen = indexed(packet["eval_records"], "example_id")
    for row in rows:
        require(
            row.get("schema") == CONDITIONAL_SCHEMA and row.get("arm") == ARM,
            "actual C conditional schema/arm",
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
            "conditional request/prompt/media/grid differs",
        )


def owner_change_counts(
    before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, int]]:
    require(len(before) == len(after), "paired owner score lengths differ")
    result = {}
    for threshold in ("50", "60", "80"):
        pairs = [
            (set(left[threshold]["owners"]), set(right[threshold]["owners"]))
            for left, right in zip(before, after, strict=True)
        ]
        result[threshold] = {
            "gained": sum(len(right - left) for left, right in pairs),
            "lost": sum(len(left - right) for left, right in pairs),
            "retained": sum(len(left & right) for left, right in pairs),
        }
    return result


def owner_changes(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, dict[str, list[str]]]:
    result = {}
    for threshold in ("50", "60", "80"):
        left = set(before[threshold]["owners"])
        right = set(after[threshold]["owners"])
        result[threshold] = {
            "gained": sorted(right - left),
            "lost": sorted(left - right),
            "retained": sorted(left & right),
        }
    return result


def burden(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reasons = Counter(
        dropped["reason"] for row in rows for dropped in row["parsed"]["dropped_predictions"]
    )
    parsed = sum(row["score"]["parsed_prediction_count"] for row in rows)
    drops = sum(row["score"]["parser_drops"] for row in rows)
    result = {
        "row_starts": parsed + drops,
        "parsed_predictions": parsed,
        "strict_repeats": sum(row["score"]["strict_repeats"] for row in rows),
        "invalid_predictions": sum(row["score"]["invalid_predictions"] for row in rows),
        "parser_drops": drops,
        "parser_drop_reasons": dict(sorted(reasons.items())),
        "geometry_invalid_drops": reasons["geometry_invalid"],
        "other_malformed_drops": drops - reasons["geometry_invalid"],
        "eos": sum(row["stop_reason"] == "im_end" for row in rows),
        "caps": sum(row["stop_reason"] == "length" for row in rows),
        "generated_tokens": sum(len(row["action_ids"]) for row in rows),
    }
    require(result["eos"] + result["caps"] == len(rows), "natural EOS/cap accounting")
    require(sum(reasons.values()) == drops, "natural parser-drop reason accounting")
    return result


def stable_rows(packet: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    shards = {example_id: rank for rank, ids in enumerate(packet["eval_shards"]) for example_id in ids}
    for frozen in packet["eval_records"]:
        rows.append(
            {
                "arm": "Stable50",
                "example_id": frozen["example_id"],
                "image_id": frozen["image_id"],
                "split": frozen["split"],
                "shard": shards[frozen["example_id"]],
                "action_ids": frozen["stable_ids"],
                "stop_reason": frozen["stable_score"]["stop_reason"],
                "parsed": frozen["stable_parsed"],
                "score": frozen["stable_score"],
                "overlap_counts": frozen["stable_overlap_counts"],
            }
        )
    return rows


def _panel_ids(label: str, rows: Sequence[Mapping[str, Any]]) -> set[str]:
    if label == "reference56":
        ids = {str(row["example_id"]) for row in rows if row["split"] == "reference56"}
        require(len(ids) == 56, "reference56 panel identity")
        return ids
    if label == "train256":
        ids = {str(row["example_id"]) for row in rows if row["split"] != "dev128"}
        require(len(ids) == 256, "train256 panel identity")
        return ids
    if label == "dev128":
        ids = {str(row["example_id"]) for row in rows if row["split"] == "dev128"}
        require(len(ids) == 128, "dev128 panel identity")
        return ids
    require(label == "union384", "unknown endpoint panel")
    ids = {str(row["example_id"]) for row in rows}
    require(len(ids) == 384, "union384 panel identity")
    return ids


def _endpoint_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {"quality": aggregate_scores([row["score"] for row in rows]), "burden": burden(rows)}


def reduce_endpoint(
    packet: Mapping[str, Any], c_rows: Sequence[Mapping[str, Any]], c_conditional: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    validate_natural_rows(c_rows, packet, arm=ARM, retained=False)
    a_rows = load_json(RETAINED_A_CONSUMER)
    validate_natural_rows(a_rows, packet, arm="A", retained=True)
    stable = stable_rows(packet)
    by_arm = {
        "Stable50": indexed(stable, "example_id"),
        "A": indexed(a_rows, "example_id"),
        ARM: indexed(list(c_rows), "example_id"),
    }
    validate_conditional_rows(c_conditional, packet)
    panels = {}
    for label in ("reference56", "train256", "dev128", "union384"):
        ids = _panel_ids(label, c_rows)
        selected = {
            arm: [rows[example_id] for example_id in sorted(ids)]
            for arm, rows in by_arm.items()
        }
        panels[label] = {
            "images": len(ids),
            "Stable50": _endpoint_summary(selected["Stable50"]),
            "A": _endpoint_summary(selected["A"]),
            ARM: _endpoint_summary(selected[ARM]),
            "C_vs_Stable50_owner_changes": owner_change_counts(
                [row["score"] for row in selected["Stable50"]],
                [row["score"] for row in selected[ARM]],
            ),
            "C_vs_A_owner_changes": owner_change_counts(
                [row["score"] for row in selected["A"]],
                [row["score"] for row in selected[ARM]],
            ),
        }

    by_image = {
        arm: {str(row["image_id"]): row for row in rows.values()}
        for arm, rows in by_arm.items()
    }
    positives = {}
    for image_id in ("351017", "417044", "477415"):
        positives[image_id] = {
            arm: {
                "tp50": by_image[arm][image_id]["score"]["50"]["tp"],
                "quality": by_image[arm][image_id]["score"],
                "burden": burden([by_image[arm][image_id]]),
            }
            for arm in ("Stable50", "A", ARM)
        }
        require(positives[image_id]["A"]["tp50"] == {"351017": 12, "417044": 11, "477415": 16}[image_id], "positive retained A baseline")

    diagnostic = {
        arm: {
            "quality": by_image[arm]["39654"]["score"],
            "burden": burden([by_image[arm]["39654"]]),
        }
        for arm in ("Stable50", "A", ARM)
    }
    old = old_endpoint()
    _, witness_reducer = old._witness_modules()
    c_cells = [witness_reducer.reduce_record(row) for row in sorted(c_conditional, key=lambda row: row["rank"])]
    a_conditional = load_json(RETAINED_A_CONDITIONAL_REDUCTION)
    require(
        [(row["case_id"], row["job_id"]) for row in a_conditional["cells"]]
        == [(row["case_id"], row["job_id"]) for row in c_cells],
        "A/C conditional scored order differs",
    )
    per_image = []
    for row in c_rows:
        example_id = row["example_id"]
        per_image.append(
            {
                "example_id": example_id,
                "image_id": row["image_id"],
                "split": row["split"],
                "C_vs_Stable50": owner_changes(
                    by_arm["Stable50"][example_id]["score"], row["score"]
                ),
                "C_vs_A": owner_changes(by_arm["A"][example_id]["score"], row["score"]),
            }
        )
    return {
        "schema": RESULT_SCHEMA,
        "arm": ARM,
        "panels": panels,
        "positive_natural_cases": positives,
        "diagnostic_39654": diagnostic,
        "per_image_owner_changes": per_image,
        "conditional": {
            "A": a_conditional,
            ARM: {
                "schema": "margin_preserved_endpoint.conditional_reduction.v1",
                "arm": ARM,
                "cells": c_cells,
                "claim_boundary": "Forced c is excluded from all free-continuation counts.",
            },
        },
        "training_final_margin_signal": packet["candidate"]["final_margin_signal"],
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
    jobs = {(job.rank, f"{job.kind}:{job.candidate_id}"): job for job in runtime_conditional_jobs(packet)}
    natural: list[dict[str, Any]] = []
    conditional: list[dict[str, Any]] = []
    terminals = []
    old = old_endpoint()
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
            "C endpoint terminal identity",
        )
        validate_terminal_resources(terminal, packet["resource_bounds"]["ranks"][shard])
        model = load_json(run / "model.json")
        require(
            model["arm"] == ARM
            and model["exported_adapter"] == adapter
            and model["training_receipt"] == receipt
            and model["cold_check"] == packet["candidate"]["cold_check"]
            and model["identity"]["model_identity"]["adapter"]["adapter_path"] == adapter["root"]
            and model["identity"]["model_identity"]["adapter"]["merged_adapters"] == [],
            "cold C model/composition identity",
        )
        rows = load_jsonl(run / "natural-rows.jsonl")
        require([row["example_id"] for row in rows] == packet["eval_shards"][shard], "C shard natural order")
        for row in rows:
            old._cold_natural(row, frozen[row["example_id"]], tokenizer)
        natural.extend(rows)
        for row in load_jsonl(run / "conditional-rows.jsonl"):
            require(row.get("schema") == CONDITIONAL_SCHEMA and row.get("arm") == ARM, "actual C conditional schema/arm")
            job = jobs[(shard, row["job_id"])]
            old.validate_conditional_artifact(row, job)
            example_id = next(case["example_id"] for case in packet["conditional_cases"] if case["case_id"] == job.case_id)
            case = frozen[example_id]
            require(row["request_id"] == example_id and row["adapter_fingerprint"] == adapter["fingerprint"], "conditional request/adapter identity")
            require(tokenizer.decode(row["action_ids"], skip_special_tokens=False) == row["text"], "conditional token/text differs")
            reparsed = native_record(row["text"], case["case"], case["golden"], row["stop_reason"])
            require(reparsed == row["parsed"], "cold C conditional parser differs")
            conditional.append(row)
    validate_natural_rows(natural, packet, arm=ARM, retained=False)
    validate_conditional_rows(conditional, packet)
    total_tokens = sum(terminal["new_tokens"] for terminal in terminals)
    require(
        total_tokens
        == sum(len(row["action_ids"]) for row in natural)
        + sum(len(row["free_ids"]) for row in conditional),
        "global token counter",
    )
    require(total_tokens <= packet["resource_bounds"]["global_max_generated_tokens"], "global 120000 generated-token bound")
    resources = {
        "schema": "margin_preserved_endpoint.resources.v1",
        "arm": ARM,
        "model_loads": sum(row["model_loads"] for row in terminals),
        "continuations": sum(row["continuations"] for row in terminals),
        "new_tokens": total_tokens,
        "model_forwards": sum(row["model_forwards"] for row in terminals),
        "image_forwards": sum(row["image_forwards"] for row in terminals),
        "allocated_gpu_seconds": sum(row["elapsed_seconds"] for row in terminals),
        "shards": terminals,
    }
    require(resources["model_loads"] == 8 and resources["continuations"] == resources["image_forwards"] == 390, "global endpoint counts")
    result = reduce_endpoint(packet, natural, conditional)
    outputs = {
        "consumer.json": natural,
        "conditional-consumer.json": conditional,
        "result.json": result,
        "resources.json": resources,
    }
    if verify:
        require(all(load_json(out / name) == value for name, value in outputs.items()), "cold merged C endpoint differs")
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    pending = subparsers.add_parser("prepare-pending")
    pending.add_argument("--out", type=Path, default=PREPARATION_ROOT / "pending-bindings.json")
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--training-receipt", type=Path, required=True)
    prepare.add_argument("--cold-check", type=Path, required=True)
    prepare.add_argument("--out", type=Path, default=PREPARATION_ROOT / "packet.json")
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
        require(args.out.is_absolute() and not args.out.exists(), "pending output must be fresh absolute path")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        publish(args.out, pending_payload())
        publish(args.out.parent / "proposed-commands.json", proposed_commands_payload())
        publish(args.out.parent / "cpu-preflight.json", cpu_preflight_payload())
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
