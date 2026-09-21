"""Thin endpoint consumer for the fixed positive-vs-repeat-event experiment.

This task-local entrypoint reuses the established native 384 consumer and the
closed escape-witness parser/reducer.  It does not train, select examples, or
interpret visual quality.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback
from typing import Any

WORKTREE = Path("/data/CoordExp/.worktrees/research-probes")
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

from src.artifacts import publish_json_exclusive  # noqa: E402
from src.adapters.dora import inspect_dora_adapter_payload  # noqa: E402
from probes.dora_owner_learning.candidate_opportunity import (  # noqa: E402
    file_hash,
    indexed,
    require,
    score,
)
from probes.dora_owner_learning.entrance_ce_eval import (  # noqa: E402
    aggregate_scores,
    owner_change,
)
from probes.dora_owner_learning.geometric_dedup_eval import (  # noqa: E402
    overlap_counts,
    reduce_records,
)
from probes.dora_owner_learning.route_access import checkpoint_config  # noqa: E402


ROOT = Path(__file__).resolve().parent
PREPARATION = ROOT / "endpoint-preparation"
GEOM = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-stable50-geometric-dedup/inputs.json")
INPUT_MANIFEST = ROOT / "input-preparation" / "candidate_manifest.json"
INPUT_ADMISSION = ROOT / "input-admission.json"
WITNESS = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-native-escape-witness")
WITNESS_RUNNER = WITNESS / "run_probe.py"
WITNESS_REDUCER = WITNESS / "reduce.py"

GEOM_SHA256 = "749fee8e60bee1a8a06e5deb6f018ac6270df2dbd0c2feac7b2eb667c9f83fc4"
INPUT_MANIFEST_SHA256 = "2870e777965007b5b06487fbb33b8408d1a992f4c90e2b4bd1f0f564c1e4aa3b"
INPUT_ADMISSION_SHA256 = "3c5ff7cd84c7faabd05c246d4a1691b5f9124ed7c3a5a697fcfd7b4ef2f38d85"
WITNESS_PACKET_SHA256 = "de4eec6b0db7ae9d066f8746179ad024aadf34d65bf50df093d76c949f846fe3"
EOS, PAD, CAP = 151645, 151643, 3084
ARMS = ("A", "B")
WORLD_SIZE = 8
NATURAL_PER_RANK = 48
WORKER_SECONDS = 3600
CUDA_LIMIT = 12 * 1024**3
RSS_LIMIT = 16 * 1024**3


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def publish(path: Path, payload: Any) -> None:
    publish_json_exclusive(path, payload)
    require(json.loads(path.read_text()) == payload, f"cold publication mismatch: {path}")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def load_path_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ConditionalJob:
    def __init__(self, *, arm: str, rank: int, case_id: str, candidate_id: str,
                 kind: str, prefix_ids: list[int], prefix_row_count: int,
                 forced_candidate: dict[str, Any] | None, budget: int) -> None:
        self.arm = arm
        self.rank = rank
        self.case_id = case_id
        self.candidate_id = candidate_id
        self.kind = kind
        self.prefix_ids = prefix_ids
        self.prefix_row_count = prefix_row_count
        self.forced_candidate = forced_candidate
        self.budget = budget

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))


def build_conditional_jobs(cases: Sequence[Mapping[str, Any]], *, arm: str = "A") -> list[ConditionalJob]:
    require(arm in ARMS, "invalid arm")
    require(len(cases) == 3, "exact three conditional cases")
    jobs: list[ConditionalJob] = []
    for index, case in enumerate(cases):
        h, c = list(case["h_ids"]), list(case["c_ids"])
        h_rows = int(case.get("h_complete_row_count", max(1, len(h) // 9)))
        common = dict(arm=arm, case_id=str(case["case_id"]),
                      candidate_id=str(case["candidate_id"]))
        jobs.append(ConditionalJob(rank=2 * index, kind="h_only", prefix_ids=h,
                                   prefix_row_count=h_rows, forced_candidate=None,
                                   budget=CAP - len(h), **common))
        jobs.append(ConditionalJob(rank=2 * index + 1, kind="h_plus_c", prefix_ids=h + c,
                                   prefix_row_count=h_rows + 1,
                                   forced_candidate={"candidate_id": case["candidate_id"],
                                                     "description": case.get("description"),
                                                     "c_ids": c},
                                   budget=CAP - len(h) - len(c), **common))
    require([job.rank for job in jobs] == list(range(6)), "conditional rank balance")
    return jobs


def rank_bounds(jobs: Sequence[ConditionalJob]) -> list[dict[str, int]]:
    by_rank = {job.rank: job for job in jobs}
    return [{
        "rank": rank,
        "natural_calls": NATURAL_PER_RANK,
        "conditional_calls": int(rank in by_rank),
        "max_calls": NATURAL_PER_RANK + int(rank in by_rank),
        "max_generated_tokens": NATURAL_PER_RANK * CAP + (by_rank[rank].budget if rank in by_rank else 0),
        "max_model_forwards": NATURAL_PER_RANK * CAP + (by_rank[rank].budget if rank in by_rank else 0),
        "max_image_forwards": NATURAL_PER_RANK + int(rank in by_rank),
        "max_model_loads": 1,
        "max_worker_seconds": WORKER_SECONDS,
        "max_cuda_bytes": CUDA_LIMIT,
        "max_rss_bytes": RSS_LIMIT,
    } for rank in range(WORLD_SIZE)]


def runtime_conditional_jobs(
    packet: Mapping[str, Any], *, arm: str, rank: int | None = None
) -> list[ConditionalJob]:
    require(arm in ARMS, "invalid runtime arm")
    rows = packet["conditional_jobs"]
    if rank is not None:
        rows = [row for row in rows if row["rank"] == rank]
    return [ConditionalJob(**{**row, "arm": arm}) for row in rows]


def validate_terminal_resources(
    terminal: Mapping[str, Any], bound: Mapping[str, int]
) -> None:
    require(
        terminal["model_loads"] == bound["max_model_loads"] == 1,
        "endpoint model-load bound failure",
    )
    require(
        terminal["natural_continuations"] == bound["natural_calls"]
        and terminal["conditional_continuations"] == bound["conditional_calls"]
        and terminal["continuations"]
        == terminal["natural_continuations"] + terminal["conditional_continuations"]
        == terminal["image_forwards"]
        <= bound["max_calls"]
        and terminal["image_forwards"] <= bound["max_image_forwards"],
        "endpoint continuation/image bound failure",
    )
    require(
        terminal["new_tokens"] == terminal["model_forwards"]
        and terminal["new_tokens"] <= bound["max_generated_tokens"]
        and terminal["model_forwards"] <= bound["max_model_forwards"],
        "endpoint token/model-forward bound failure",
    )
    require(
        0 <= terminal["elapsed_seconds"] <= bound["max_worker_seconds"],
        "endpoint wall bound failure",
    )
    require(
        0 <= terminal["peak_cuda_allocated_bytes"] <= bound["max_cuda_bytes"]
        and 0 <= terminal["peak_cuda_reserved_bytes"] <= bound["max_cuda_bytes"],
        "endpoint CUDA bound failure",
    )
    require(
        0 <= terminal["peak_rss_bytes"] <= bound["max_rss_bytes"],
        "endpoint RSS bound failure",
    )


def _source(path: Path, expected: str) -> dict[str, Any]:
    require(path.is_file() and file_hash(path) == expected, f"source changed: {path}")
    return {"path": str(path), "sha256": expected, "size_bytes": path.stat().st_size}


def prepare_packet_payload() -> dict[str, Any]:
    sources = {
        "geometric_inputs": _source(GEOM, GEOM_SHA256),
        "input_manifest": _source(INPUT_MANIFEST, INPUT_MANIFEST_SHA256),
        "input_admission": _source(INPUT_ADMISSION, INPUT_ADMISSION_SHA256),
        "witness_runner": _source(WITNESS_RUNNER, file_hash(WITNESS_RUNNER)),
        "witness_reducer": _source(WITNESS_REDUCER, file_hash(WITNESS_REDUCER)),
        "producer": _source(Path(__file__).resolve(), file_hash(Path(__file__).resolve())),
    }
    geom, manifest, admission = map(load_json, (GEOM, INPUT_MANIFEST, INPUT_ADMISSION))
    require(geom["schema"] == "stable50_geometric_dedup.inputs.v1", "wrong geometric packet")
    require(admission["schema"] == "positive_branch_vs_repeat_event.input_admission.v1" and
            admission["status"] == "lead_accepted_inputs" and
            admission["manifest_sha256"] == INPUT_MANIFEST_SHA256, "input admission missing")
    require(manifest["schema"] == "positive_branch_vs_repeat_event.input_preparation.v1", "wrong input manifest")
    require(len(geom["eval_records"]) == 384 and list(map(len, geom["eval_shards"])) == [48] * 8,
            "wrong exposed384")
    validate_exact_natural_population(
        [{"example_id": eid, "shard": rank} for rank, ids in enumerate(geom["eval_shards"]) for eid in ids],
        geom["eval_shards"],
    )
    generation = geom["config"]["generation"]
    require(generation["max_new_tokens"] == CAP and generation["temperature"] == 0 and
            generation["top_p"] == 1 and generation["repetition_penalty"] == 1 and
            geom["config"]["backend"]["hf"] == {"attn_implementation": "sdpa", "patch_embed_linearization": "enabled"} and
            geom["config"]["model"]["dtype"] == "fp32", "native policy changed")
    require(manifest["model_identity"]["base_model"] == geom["model"]["base_model_path"] and
            manifest["model_identity"]["embedding_delta"]["path"] == geom["model"]["source_embedding"]["root"],
            "model/embedding identity differs")

    frozen = indexed(geom["eval_records"], "example_id")
    witness_packet_path = Path(manifest["source_bindings"]["packet_v3"]["path"])
    require(manifest["source_bindings"]["packet_v3"]["sha256"] == WITNESS_PACKET_SHA256 and
            file_hash(witness_packet_path) == WITNESS_PACKET_SHA256, "witness packet changed")
    witness_cases = {row["case_id"]: row for row in load_json(witness_packet_path)["cases"]}
    cases = []
    for item in manifest["positives"]:
        case_id = str(item["case_id"])
        example_id = item["image"]["row_id"]
        base = frozen[example_id]
        old = witness_cases[case_id]
        candidate = next(row for row in old["candidates"] if row["candidate_id"] == item["candidate_id"])
        require(item["prompt"]["token_ids"] == base["prompt_token_ids"] == old["prompt_token_ids"],
                f"{case_id}: prompt differs")
        plan = base["case"]["image_plan"]
        require(item["image"]["image_path"] == base["case"]["image_path"] and
                item["image"]["image_sha256"] == plan["image_content_sha256"] and
                item["image"]["executed_media_sha256"] == plan["executed_media_sha256"] and
                item["image"]["observed_image_grid_thw"] == plan["observed_image_grid_thw"],
                f"{case_id}: image identity differs")
        require(item["h"]["token_ids"] == old["h_ids"] and item["c"]["token_ids"] == candidate["c_ids"],
                f"{case_id}: literal h/c differs")
        cases.append({
            "case_id": case_id, "example_id": example_id, "candidate_id": item["candidate_id"],
            "description": item["c"]["description"], "h_ids": item["h"]["token_ids"],
            "h_ids_sha256": item["h"]["ids_sha256"], "h_complete_row_count": old["h_complete_row_count"],
            "c_ids": item["c"]["token_ids"], "c_ids_sha256": item["c"]["ids_sha256"],
            "image": item["image"], "prompt": item["prompt"],
        })
    require([case["candidate_id"] for case in cases] == admission["positives"], "positive order differs")
    jobs = build_conditional_jobs(cases)
    return {
        "schema": "positive_branch_vs_repeat_event.endpoint_packet.v1",
        "status": "candidate_ready_for_exported_adapters_no_gpu_grant",
        "sources": sources,
        "model": geom["model"], "config": geom["config"],
        "eval_records": geom["eval_records"], "eval_shards": geom["eval_shards"],
        "protected_targets": geom["protected_targets"],
        "conditional_cases": cases,
        "conditional_jobs": [job.to_dict() for job in jobs],
        "rank_bounds": rank_bounds(jobs),
        "policy": {"temperature": 0.0, "top_p": 1.0, "top_k": 0,
                   "repetition_penalty": 1.0, "use_model_defaults": False,
                   "trace": "none", "eos_token_id": EOS, "total_action_cap": CAP},
        "claim_boundary": "Exposed384 and fixed conditional read only; no visual acceptance, Source parity, training, checkpoint promotion, or held-out claim.",
    }


def validate_packet(packet: Mapping[str, Any]) -> None:
    require(packet.get("schema") == "positive_branch_vs_repeat_event.endpoint_packet.v1", "wrong endpoint packet")
    for source in packet["sources"].values():
        require(file_hash(Path(source["path"])) == source["sha256"], f"source changed: {source['path']}")
    require(dict(packet) == prepare_packet_payload(), "endpoint packet content differs from admitted sources")
    geom = load_json(GEOM)
    require(packet["eval_records"] == geom["eval_records"] and packet["eval_shards"] == geom["eval_shards"],
            "exposed384 differs from geometric packet")
    validate_exact_natural_population(
        [{"example_id": eid, "shard": rank} for rank, ids in enumerate(packet["eval_shards"]) for eid in ids],
        packet["eval_shards"],
    )
    jobs = [ConditionalJob(**row) for row in packet["conditional_jobs"]]
    require(packet["rank_bounds"] == rank_bounds(jobs), "rank work bounds differ")


def validate_adapter_export(receipt: Mapping[str, Any], *, arm: str,
                            inspect_payload: Callable[[str | Path, str | Path | None], dict[str, Any]] = inspect_dora_adapter_payload) -> dict[str, Any]:
    require(arm in ARMS and receipt.get("schema") == "repeat_recovery_train.receipt.v1", "wrong training receipt")
    require(receipt.get("status") == "completed" and receipt.get("scientific_status") == "candidate" and
            receipt.get("arm") == arm and receipt.get("mode") == "full" and receipt.get("updates") == 32 and
            receipt.get("stop_reason") == "fixed_32_updates", "incomplete/wrong full endpoint")
    adapter = receipt.get("saved_adapter")
    require(isinstance(adapter, Mapping) and Path(str(adapter.get("root", ""))).is_absolute(), "missing saved adapter")
    for item in adapter.get("files", []):
        path = Path(adapter["root"]) / item["relative_path"]
        require(path.is_file() and path.stat().st_size == item["size_bytes"] and file_hash(path) == item["sha256"],
                "adapter payload changed")
    require(len(adapter.get("files", [])) == adapter.get("file_count"), "adapter file count differs")
    composition = receipt.get("composition", {})
    observed = inspect_payload(adapter["root"], composition.get("base_model_path"))
    require(observed == adapter, "saved adapter inspection differs")
    return dict(adapter)


def validate_adapter_receipt(receipt: Mapping[str, Any], *, arm: str, packet: Mapping[str, Any]) -> dict[str, Any]:
    adapter = validate_adapter_export(receipt, arm=arm)
    composition = receipt["composition"]
    require(composition["base_model_path"] == packet["model"]["base_model_path"] and
            composition["source_embedding"] == packet["model"]["source_embedding"] and
            composition["source_adapter"] == packet["model"]["current_adapter"] and
            composition["unmerged"] is True and composition["dtype"] == "fp32" and
            composition["attention_implementation"] == "sdpa", "training composition differs")
    for key in ("input", "manifest", "input_admission", "protocol", "code_identity"):
        ref = receipt[key]
        require(Path(ref["path"]).is_file() and file_hash(Path(ref["path"])) == ref["sha256"], f"training ref changed: {key}")
    for key, packet_key in (("manifest", "input_manifest"), ("input_admission", "input_admission")):
        ref = receipt[key]
        frozen = packet["sources"][packet_key]
        require(
            Path(ref["path"]).resolve() == Path(frozen["path"]).resolve()
            and ref["sha256"] == frozen["sha256"],
            f"training {key} is not the admitted endpoint input",
        )
    return adapter


def _witness_modules() -> tuple[Any, Any]:
    return (load_path_module("endpoint_witness_runner", WITNESS_RUNNER),
            load_path_module("endpoint_witness_reducer", WITNESS_REDUCER))


def conditional_artifact(job: ConditionalJob, *, free_ids: list[int], stop_reason: str,
                         parsed: Mapping[str, Any]) -> dict[str, Any]:
    runner, _ = _witness_modules()
    partition = runner.partition_parser_rows(parsed, prefix_row_count=job.prefix_row_count)
    identity = runner.credit_identity({"kind": job.kind, "candidate": job.forced_candidate}, partition)
    return {
        "schema": "positive_branch_vs_repeat_event.endpoint_conditional.v1",
        "arm": job.arm, "rank": job.rank, "case_id": job.case_id,
        "candidate_id": job.candidate_id, "job_id": f"{job.kind}:{job.candidate_id}",
        "kind": job.kind, "prefix_ids": list(job.prefix_ids), "free_ids": list(free_ids),
        "action_ids": list(job.prefix_ids) + list(free_ids), "budget": job.budget,
        "stop_reason": stop_reason, "native_eos_observed": stop_reason == "im_end",
        "eos_is_positive_outcome": False, "forced_candidate": job.forced_candidate,
        "parsed": dict(parsed), "parser_partition": partition, "credit_identity": identity,
    }


def validate_conditional_artifact(record: Mapping[str, Any], job: ConditionalJob) -> None:
    runner, _ = _witness_modules()
    require(record["arm"] == job.arm, "conditional runtime arm differs")
    require(record["prefix_ids"] == job.prefix_ids and record["action_ids"] == job.prefix_ids + record["free_ids"],
            "literal prefix/free partition differs")
    runner.validate_suffix(record["free_ids"], budget=job.budget, stop=record["stop_reason"])
    expected_partition = runner.partition_parser_rows(
        record["parsed"], prefix_row_count=job.prefix_row_count
    )
    expected_credit = runner.credit_identity(
        {"kind": job.kind, "candidate": job.forced_candidate}, expected_partition
    )
    require(record["parser_partition"] == expected_partition, "conditional parser partition differs")
    require(record["credit_identity"] == expected_credit, "conditional credit identity differs")
    credit = record["credit_identity"]
    require(credit["forced_candidate_in_free_counts"] is False, "forced candidate entered free credit")
    require(credit["forced_candidate_row_count"] == (1 if job.kind == "h_plus_c" else 0), "forced row count differs")
    require(record["parser_partition"]["prefix_row_count"] == job.prefix_row_count, "prefix row count differs")


def validate_exact_natural_population(records: Sequence[Mapping[str, Any]], shards: Sequence[Sequence[str]]) -> None:
    expected = [(rank, eid) for rank, ids in enumerate(shards) for eid in ids]
    observed = [(int(row["shard"]), str(row["example_id"])) for row in records]
    require(len(expected) == 384 and len(records) == len({eid for _, eid in observed}) == 384, "exact384 identity failure")
    require(observed == expected, "exact384 ordered shard identity failure")


def pair_owner_changes(a: Sequence[Mapping[str, Any]], b: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    left, right = indexed(list(a), "example_id"), indexed(list(b), "example_id")
    require(set(left) == set(right), "paired natural IDs differ")
    result = []
    for eid in sorted(left):
        result.append({"example_id": eid, "B_vs_A": {
            threshold: owner_change(right[eid]["score"][threshold]["owners"], left[eid]["score"][threshold]["owners"])
            for threshold in ("50", "60", "80")
        }})
    return result


def panel_ids(label: str, rows: Sequence[Mapping[str, Any]]) -> set[str]:
    require(label in ("union384", "train256", "dev128"), "unknown endpoint panel")
    if label == "union384":
        return {str(row["example_id"]) for row in rows}
    if label == "train256":
        return {str(row["example_id"]) for row in rows if row["split"] != "dev128"}
    return {str(row["example_id"]) for row in rows if row["split"] == "dev128"}


def endpoint_aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary = aggregate_scores([row["score"] for row in rows])
    summary["stop_reasons"] = {
        stop: sum(row["stop_reason"] == stop for row in rows)
        for stop in ("im_end", "length")
    }
    summary["eos"] = summary["stop_reasons"]["im_end"]
    require(summary["eos"] + summary["cap"] == len(rows), "endpoint stop accounting differs")
    return summary


def _checked_action(ids: list[int], stop: str, budget: int) -> None:
    require(ids and len(ids) <= budget and PAD not in ids and EOS not in ids[:-1], "action token corruption")
    require((stop == "im_end" and ids[-1] == EOS) or (stop == "length" and len(ids) == budget and EOS not in ids),
            "action terminal/budget corruption")


def execute(packet_path: Path, receipt_path: Path, out: Path, *, arm: str, shard: int) -> None:
    import torch
    from probes.dora_owner_learning.runtime import load_policy
    from probes.source_rweak_row_cross.run import build_requests, native_record
    from src.config.inference import InferConfig
    from src.qwen.generation import NativeGenerationPolicy, generate_continuations
    from src.qwen.native import prepare_native_inputs

    require(packet_path.is_absolute() and receipt_path.is_absolute() and out.is_absolute(), "execution paths must be absolute")
    packet = load_json(packet_path)
    validate_packet(packet)
    require(arm in ARMS and 0 <= shard < WORLD_SIZE and os.environ.get("CUDA_VISIBLE_DEVICES") == str(shard) and
            torch.cuda.device_count() == 1, "single assigned GPU identity")
    receipt = load_json(receipt_path)
    adapter = validate_adapter_receipt(receipt, arm=arm, packet=packet)
    run = out / f"shard-{shard}"
    require(not run.exists(), "occupied endpoint shard")
    run.mkdir(parents=True)
    packet_sha, receipt_sha = file_hash(packet_path), file_hash(receipt_path)
    terminal = {"schema": "positive_branch_vs_repeat_event.endpoint_terminal.v1", "status": "running",
                "arm": arm, "shard": shard, "pid": os.getpid(), "packet_sha256": packet_sha,
                "adapter_receipt_sha256": receipt_sha, "adapter_fingerprint": adapter["fingerprint"],
                "model_loads": 0, "continuations": 0,
                "natural_continuations": 0, "conditional_continuations": 0, "new_tokens": 0,
                "model_forwards": 0, "image_forwards": 0}
    publish(run / "launch.json", dict(terminal))
    started = time.monotonic()
    handles = []
    alarm = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("endpoint worker timeout")))
        signal.alarm(WORKER_SECONDS)
        config = checkpoint_config(InferConfig.model_validate(packet["config"]), adapter["root"])
        torch.cuda.reset_peak_memory_stats()
        qwen, identity = load_policy(config, device=torch.device("cuda:0"))
        terminal["model_loads"] = 1
        live = identity["model_identity"]["adapter"]
        require(live["adapter_path"] == adapter["root"] and live["merged_adapters"] == [], "stale/merged loaded adapter")
        require(identity["effective_settings"]["observed_model_dtype"]["parameter_dtype_names"] == ["torch.float32"] and
                identity["effective_settings"]["observed_attn_implementation"] == "sdpa", "live numerics differ")
        require(inspect_dora_adapter_payload(live["adapter_path"], packet["model"]["base_model_path"]) == adapter,
                "live exported adapter fingerprint differs")
        publish(run / "model.json", {"identity": identity, "exported_adapter": adapter,
                                      "adapter_receipt": str(receipt_path), "adapter_receipt_sha256": receipt_sha})
        publish(run / "config.json", config.model_dump(mode="json"))
        qwen.model.eval()
        counters = {"model": 0, "image": 0}
        handles.append(qwen.model.register_forward_pre_hook(lambda *_: counters.__setitem__("model", counters["model"] + 1)))
        visuals = [module for name, module in qwen.model.named_modules() if name.endswith("visual")]
        require(len(visuals) == 1, "ambiguous visual module")
        handles.append(visuals[0].register_forward_pre_hook(lambda *_: counters.__setitem__("image", counters["image"] + 1)))
        policy = NativeGenerationPolicy(temperature=0.0, top_p=1.0, top_k=0, repetition_penalty=1.0, use_model_defaults=False)
        by_id = indexed(packet["eval_records"], "example_id")

        def batch_for(frozen: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
            requests, _ = build_requests(qwen, packet["config"], [frozen["case"]])
            require(list(requests[0].expected_token_ids) == frozen["prompt_token_ids"], "request prompt differs")
            batch = prepare_native_inputs(qwen.processor, requests, device=torch.device("cuda:0"), record_media_identity=True)
            plan = frozen["case"]["image_plan"]
            require(list(batch.prompt_token_ids[0]) == frozen["prompt_token_ids"] and
                    batch.media_sha256[0] == plan["executed_media_sha256"] and
                    list(batch.image_grids[0]) == plan["observed_image_grid_thw"], "live prompt/media/grid differs")
            batch_receipt = {
                "prompt_token_ids_sha256": digest(frozen["prompt_token_ids"]),
                "executed_media_sha256": batch.media_sha256[0],
                "observed_image_grid_thw": list(batch.image_grids[0]),
            }
            batch_receipt["batch_identity_sha256"] = digest(batch_receipt)
            return batch, batch_receipt

        natural_path, conditional_path = run / "natural-rows.jsonl", run / "conditional-rows.jsonl"
        with natural_path.open("x") as natural_stream:
            for eid in packet["eval_shards"][shard]:
                frozen = by_id[eid]
                batch, batch_receipt = batch_for(frozen)
                with torch.inference_mode():
                    generated = generate_continuations(qwen.model, batch, extensions=[[]], budgets=[CAP],
                        eos_token_id=EOS, pad_token_id=qwen.tokenizer.pad_token_id, policy=policy, trace="none")[0]
                require(generated.request_id == eid, "natural request association differs")
                ids = list(generated.token_ids)
                _checked_action(ids, generated.stop_reason, CAP)
                text = qwen.tokenizer.decode(ids, skip_special_tokens=False)
                parsed = native_record(text, frozen["case"], frozen["golden"], generated.stop_reason)
                card = score(parsed, seed=None, length=len(ids), stop=generated.stop_reason)
                row = {"schema": "positive_branch_vs_repeat_event.endpoint_natural.v1", "arm": arm,
                       "shard": shard, "packet_sha256": packet_sha, "adapter_receipt_sha256": receipt_sha,
                       "adapter_fingerprint": adapter["fingerprint"],
                       "request_id": generated.request_id, "example_id": eid,
                       "image_id": frozen["image_id"], "split": frozen["split"],
                       "action_ids": ids, "prefix_ids": [], "forced_ids": [], "remaining_budget": CAP,
                       "text": text, "stop_reason": generated.stop_reason, "parsed": parsed,
                       "score": card, "overlap_counts": overlap_counts(parsed), **batch_receipt}
                natural_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                natural_stream.flush()
                os.fsync(natural_stream.fileno())
                terminal["natural_continuations"] += 1
                terminal["continuations"] += 1
                terminal["new_tokens"] += len(ids)
        selected = runtime_conditional_jobs(packet, arm=arm, rank=shard)
        with conditional_path.open("x") as conditional_stream:
            for job in selected:
                frozen = by_id[next(case["example_id"] for case in packet["conditional_cases"] if case["case_id"] == job.case_id)]
                batch, batch_receipt = batch_for(frozen)
                with torch.inference_mode():
                    generated = generate_continuations(qwen.model, batch, extensions=[job.prefix_ids], budgets=[job.budget],
                        eos_token_id=EOS, pad_token_id=qwen.tokenizer.pad_token_id, policy=policy, trace="none")[0]
                require(generated.request_id == frozen["example_id"], "conditional request association differs")
                free = list(generated.token_ids)
                _checked_action(free, generated.stop_reason, job.budget)
                text = qwen.tokenizer.decode(job.prefix_ids + free, skip_special_tokens=False)
                parsed = native_record(text, frozen["case"], frozen["golden"], generated.stop_reason)
                record = conditional_artifact(job, free_ids=free, stop_reason=generated.stop_reason, parsed=parsed)
                record.update(
                    packet_sha256=packet_sha,
                    adapter_receipt_sha256=receipt_sha,
                    adapter_fingerprint=adapter["fingerprint"],
                    request_id=generated.request_id,
                    text=text,
                    **batch_receipt,
                )
                validate_conditional_artifact(record, job)
                conditional_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                conditional_stream.flush()
                os.fsync(conditional_stream.fileno())
                terminal["conditional_continuations"] += 1
                terminal["continuations"] += 1
                terminal["new_tokens"] += len(free)
        bound = packet["rank_bounds"][shard]
        require(terminal["natural_continuations"] == 48 and terminal["conditional_continuations"] == bound["conditional_calls"] and
                terminal["continuations"] == counters["image"] <= bound["max_calls"] and
                terminal["new_tokens"] == counters["model"] <= bound["max_generated_tokens"], "runtime counter/bound failure")
        terminal.update(status="completed", exit_code=0)
    except BaseException as exc:
        terminal.update(status="failed", exit_code=1, error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, alarm)
        for handle in handles:
            handle.remove()
        terminal.update(elapsed_seconds=time.monotonic() - started,
                        model_forwards=locals().get("counters", {}).get("model", 0),
                        image_forwards=locals().get("counters", {}).get("image", 0),
                        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
                        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0,
                        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        publish(run / "terminal.json", terminal)


def _cold_natural(row: dict[str, Any], frozen: Mapping[str, Any], tokenizer: Any) -> dict[str, Any]:
    from probes.source_rweak_row_cross.run import native_record
    require(row["request_id"] == row["example_id"] == frozen["example_id"], "natural request association differs")
    require(row["prefix_ids"] == row["forced_ids"] == [] and row["remaining_budget"] == CAP, "natural forcing/budget")
    _checked_action(row["action_ids"], row["stop_reason"], CAP)
    require(tokenizer.decode(row["action_ids"], skip_special_tokens=False) == row["text"], "natural token/text differs")
    parsed = native_record(row["text"], frozen["case"], frozen["golden"], row["stop_reason"])
    require(parsed == row["parsed"], "cold natural parser differs")
    plan = frozen["case"]["image_plan"]
    identity = {"prompt_token_ids_sha256": digest(frozen["prompt_token_ids"]),
                "executed_media_sha256": plan["executed_media_sha256"],
                "observed_image_grid_thw": plan["observed_image_grid_thw"]}
    require(all(row[key] == value for key, value in identity.items()) and
            row["batch_identity_sha256"] == digest(identity), "cold natural prompt/media/grid differs")
    card = score(parsed, seed=None, length=len(row["action_ids"]), stop=row["stop_reason"])
    require(card == row["score"] and overlap_counts(parsed) == row["overlap_counts"], "cold natural score differs")
    return row


def merge(packet_path: Path, receipt_path: Path, out: Path, *, arm: str, verify: bool = False) -> dict[str, Any]:
    from probes.source_rweak_row_cross.run import native_record
    from tokenizers import Tokenizer
    packet = load_json(packet_path)
    validate_packet(packet)
    receipt = load_json(receipt_path)
    adapter = validate_adapter_receipt(receipt, arm=arm, packet=packet)
    packet_sha, receipt_sha = file_hash(packet_path), file_hash(receipt_path)
    tokenizer = Tokenizer.from_file(packet["model"]["base_model_path"] + "/tokenizer.json")
    by_id = indexed(packet["eval_records"], "example_id")
    natural = []
    conditional = []
    terminals = []
    jobs = {
        (
            job.rank,
            f"{job.kind}:{job.candidate_id}",
        ): job
        for job in runtime_conditional_jobs(packet, arm=arm)
    }
    for shard in range(8):
        run = out / f"shard-{shard}"
        terminal = load_json(run / "terminal.json")
        terminals.append(terminal)
        require(terminal["status"] == "completed" and terminal["exit_code"] == 0 and terminal["arm"] == arm and
                terminal["packet_sha256"] == packet_sha and terminal["adapter_receipt_sha256"] == receipt_sha,
                "endpoint terminal identity")
        require(terminal["adapter_fingerprint"] == adapter["fingerprint"], "terminal adapter differs")
        validate_terminal_resources(terminal, packet["rank_bounds"][shard])
        model = load_json(run / "model.json")
        require(
            model["exported_adapter"] == adapter
            and model["adapter_receipt_sha256"] == receipt_sha
            and model["identity"]["model_identity"]["adapter"]["adapter_path"] == adapter["root"]
            and model["identity"]["model_identity"]["adapter"]["merged_adapters"] == [],
            "cold loaded adapter identity differs",
        )
        rows = load_jsonl(run / "natural-rows.jsonl")
        require([row["example_id"] for row in rows] == packet["eval_shards"][shard], "shard natural order differs")
        require(
            all(row["adapter_fingerprint"] == adapter["fingerprint"] for row in rows),
            "natural adapter identity differs",
        )
        natural.extend(_cold_natural(row, by_id[row["example_id"]], tokenizer) for row in rows)
        for row in load_jsonl(run / "conditional-rows.jsonl"):
            job = jobs[(shard, row["job_id"])]
            validate_conditional_artifact(row, job)
            frozen = by_id[next(case["example_id"] for case in packet["conditional_cases"] if case["case_id"] == job.case_id)]
            require(
                row["request_id"] == frozen["example_id"]
                and row["adapter_fingerprint"] == adapter["fingerprint"],
                "conditional request/adapter identity differs",
            )
            require(
                tokenizer.decode(row["action_ids"], skip_special_tokens=False) == row["text"],
                "conditional token/text differs",
            )
            reparsed = native_record(
                row["text"], frozen["case"], frozen["golden"], row["stop_reason"]
            )
            require(reparsed == row["parsed"], "cold conditional parser differs")
            plan = frozen["case"]["image_plan"]
            identity = {"prompt_token_ids_sha256": digest(frozen["prompt_token_ids"]),
                        "executed_media_sha256": plan["executed_media_sha256"],
                        "observed_image_grid_thw": plan["observed_image_grid_thw"]}
            require(all(row[key] == value for key, value in identity.items()) and
                    row["batch_identity_sha256"] == digest(identity), "cold conditional prompt/media/grid differs")
            conditional.append(row)
    validate_exact_natural_population(natural, packet["eval_shards"])
    require(len(conditional) == 6 and {(row["rank"], row["job_id"]) for row in conditional} == set(jobs), "conditional6 identity")
    cold = [{"example_id": row["example_id"], "image_id": row["image_id"], "split": row["split"],
             "score": row["score"], "overlap_counts": row["overlap_counts"], **row} for row in natural]
    reduction = reduce_records(cold, packet)
    _, witness_reducer = _witness_modules()
    conditional_reduction = {
        "schema": "positive_branch_vs_repeat_event.endpoint_conditional_reduction.v1", "arm": arm,
        "cells": [witness_reducer.reduce_record(row) for row in sorted(conditional, key=lambda row: row["rank"])],
        "claim_boundary": "Forced c excluded; updated-model conditional output has no Source parity assertion or visual acceptance.",
    }
    resources = {"arm": arm, "adapter": adapter, "model_loads": sum(t["model_loads"] for t in terminals),
                 "continuations": sum(t["continuations"] for t in terminals), "new_tokens": sum(t["new_tokens"] for t in terminals),
                 "model_forwards": sum(t["model_forwards"] for t in terminals), "image_forwards": sum(t["image_forwards"] for t in terminals),
                 "allocated_gpu_seconds": sum(t["elapsed_seconds"] for t in terminals), "shards": terminals}
    if verify:
        require(load_json(out / "consumer.json") == natural and load_json(out / "conditional-consumer.json") == conditional and
                load_json(out / "reduction.json") == reduction and load_json(out / "conditional-reduction.json") == conditional_reduction and
                load_json(out / "resources.json") == resources, "cold merged endpoint differs")
    else:
        publish(out / "consumer.json", natural)
        publish(out / "conditional-consumer.json", conditional)
        publish(out / "reduction.json", reduction)
        publish(out / "conditional-reduction.json", conditional_reduction)
        publish(out / "resources.json", resources)
    return {"status": "verified" if verify else "merged", "arm": arm, "natural": 384,
            "conditional": 6, "consumer_sha256": file_hash(out / "consumer.json") if (out / "consumer.json").exists() else None}


def pair(a_out: Path, b_out: Path, out: Path) -> dict[str, Any]:
    a = load_json(a_out / "consumer.json")
    b = load_json(b_out / "consumer.json")
    require(all(row["arm"] == "A" for row in a) and all(row["arm"] == "B" for row in b), "paired arm labels differ")
    paired = pair_owner_changes(a, b)
    panels = {}
    for label in ("union384", "train256", "dev128"):
        ids = panel_ids(label, a)
        aa = [row for row in a if row["example_id"] in ids]
        bb = [row for row in b if row["example_id"] in ids]
        changes = [row["B_vs_A"] for row in paired if row["example_id"] in ids]
        panels[label] = {"images": len(ids), "A": endpoint_aggregate(aa), "B": endpoint_aggregate(bb),
                         "B_vs_A_owner_changes": {t: {k: sum(len(row[t][k]) for row in changes) for k in ("gained", "lost", "retained")}
                                                  for t in ("50", "60", "80")}}
    a_cond, b_cond = load_json(a_out / "conditional-reduction.json"), load_json(b_out / "conditional-reduction.json")
    require([(row["case_id"], row["job_id"]) for row in a_cond["cells"]] ==
            [(row["case_id"], row["job_id"]) for row in b_cond["cells"]], "paired conditional jobs differ")
    payload = {"schema": "positive_branch_vs_repeat_event.endpoint_pair.v1", "A": str(a_out), "B": str(b_out),
               "A_consumer_sha256": file_hash(a_out / "consumer.json"), "B_consumer_sha256": file_hash(b_out / "consumer.json"),
               "panels": panels, "per_image": paired,
               "conditional": {"A": a_cond, "B": b_cond},
               "claim_boundary": "B-vs-A paired exposed384 and fixed conditional read; unmatched predictions are not hallucination labels."}
    publish(out, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out", type=Path, default=PREPARATION / "packet.json")
    for command in ("execute", "merge", "verify"):
        p = sub.add_parser(command)
        p.add_argument("--packet", type=Path, required=True)
        p.add_argument("--adapter-receipt", type=Path, required=True)
        p.add_argument("--arm", choices=ARMS, required=True)
        p.add_argument("--out", type=Path, required=True)
        if command == "execute":
            p.add_argument("--shard", type=int, required=True)
    p = sub.add_parser("pair")
    p.add_argument("--a", type=Path, required=True)
    p.add_argument("--b", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        payload = prepare_packet_payload()
        require(not args.out.exists(), "occupied endpoint packet")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        publish(args.out, payload)
        print(args.out)
    elif args.command == "execute":
        execute(args.packet, args.adapter_receipt, args.out, arm=args.arm, shard=args.shard)
    elif args.command in ("merge", "verify"):
        print(json.dumps(merge(
            args.packet,
            args.adapter_receipt,
            args.out,
            arm=args.arm,
            verify=args.command == "verify",
        )))
    else:
        pair(args.a, args.b, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
