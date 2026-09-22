"""Acquire bounded native escape witnesses from exact original-image prefixes.

The packet owns visual admission and literal token sequences. This runner only
validates frozen inputs, loads unchanged Stable50, and records fresh natural,
h-only, and h-plus-c continuations. It never selects candidates or trains.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import time
import traceback
from typing import Any


WORKTREE = Path.cwd()
if not (WORKTREE / "src").is_dir():
    WORKTREE = Path("/data/CoordExp/.worktrees/research-probes")
if str(WORKTREE) not in os.sys.path:
    os.sys.path.insert(0, str(WORKTREE))

from probes.dora_owner_learning.candidate_opportunity import file_hash  # noqa: E402
from probes.dora_owner_learning.route_access import checkpoint_config, publish  # noqa: E402


SCHEMA = "native_escape_witness.v1"
RECORD_SCHEMA = "native_escape_witness.record.v1"
EOS = 151645
PAD = 151643
CAP = 3084
HARD_WALL_SECONDS = 1500
COHORT = ("351017", "417044", "477415", "502725")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def token_ids(value: Any, *, field: str, allow_empty: bool = True,
              allow_eos: bool = False) -> list[int]:
    require(isinstance(value, list), f"{field} must be a list")
    require(allow_empty or bool(value), f"{field} must be nonempty")
    require(all(type(v) is int and v >= 0 for v in value),
            f"{field} must contain nonnegative integer token IDs")
    require(PAD not in value, f"{field} contains pad token")
    require(allow_eos or EOS not in value, f"{field} contains EOS")
    return list(value)


def validate_natural_action(value: Any, *, field: str) -> list[int]:
    require(isinstance(value, list) and value, f"{field} must be nonempty")
    require(all(type(v) is int and v >= 0 for v in value), f"{field} IDs invalid")
    require(PAD not in value, f"{field} contains pad token")
    require(EOS not in value[:-1], f"{field} contains early EOS")
    require(value[-1] == EOS or len(value) == CAP,
            f"{field} must end in EOS or fill cap")
    require(len(value) <= CAP, f"{field} exceeds cap")
    return list(value)


def validate_complete_row(candidate: Mapping[str, Any], *, case_id: str) -> dict[str, Any]:
    candidate_id = candidate.get("candidate_id")
    require(isinstance(candidate_id, str) and candidate_id,
            f"{case_id}: candidate_id is required")
    ids = token_ids(candidate.get("c_ids"), field=f"{case_id}/{candidate_id}.c_ids",
                    allow_empty=False)
    require(candidate.get("geometry_valid") is True,
            f"{case_id}/{candidate_id}: candidate geometry must be valid")
    require(candidate.get("source_job_id") == "early_original_translated",
            f"{case_id}/{candidate_id}: wrong source job")
    require(candidate.get("source_image_condition") == "original",
            f"{case_id}/{candidate_id}: candidate must come from original image")
    require(candidate.get("visual_admission") == "root_accepted",
            f"{case_id}/{candidate_id}: root visual admission is required")
    raw = candidate.get("raw_span_text")
    require(isinstance(raw, str) and raw.startswith("<|object_ref_start|>") and
            raw.endswith("<|box_end|>"),
            f"{case_id}/{candidate_id}: c is not one complete row")
    require(candidate.get("raw_span_sha256") == hashlib.sha256(raw.encode()).hexdigest(),
            f"{case_id}/{candidate_id}: raw row hash differs")
    require(candidate.get("c_ids_sha256") == digest_json(ids),
            f"{case_id}/{candidate_id}: c token hash differs")
    order = candidate.get("source_free_complete_order")
    require(type(order) is int and 0 <= order < 5,
            f"{case_id}/{candidate_id}: source order outside first five")
    max_iou = candidate.get("max_class_blind_iou_to_h")
    require(isinstance(max_iou, (int, float)) and float(max_iou) <= 0.95,
            f"{case_id}/{candidate_id}: overlaps h above threshold")
    return dict(candidate, c_ids=ids)


def validate_case(case: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    require(isinstance(case, Mapping), f"case[{index}] must be an object")
    case_id = case.get("case_id")
    require(case_id == COHORT[index], f"case[{index}] must be {COHORT[index]}")
    source = case.get("source_case")
    require(isinstance(source, Mapping), f"{case_id}.source_case is required")
    require(Path(str(source.get("image_path", ""))).is_absolute(),
            f"{case_id}: image_path must be absolute")
    require(source.get("row_id"), f"{case_id}: source row_id required")
    prompt = token_ids(case.get("prompt_token_ids"), field=f"{case_id}.prompt_token_ids",
                       allow_empty=False, allow_eos=True)
    baseline = validate_natural_action(case.get("baseline_action_ids"),
                                       field=f"{case_id}.baseline_action_ids")
    h_ids = token_ids(case.get("h_ids"), field=f"{case_id}.h_ids", allow_empty=False)
    require(case.get("h_ids_sha256") == digest_json(h_ids),
            f"{case_id}: h token hash differs")
    require(case.get("h_source_job_id") == "early_original_native",
            f"{case_id}: h is not original-image native early prefix")
    require(type(case.get("h_complete_row_count")) is int and
            case["h_complete_row_count"] > 0,
            f"{case_id}: h_complete_row_count invalid")
    raw_expected = case.get("expected_h_only_first512_ids")
    require(isinstance(raw_expected, list) and
            all(type(v) is int and v >= 0 for v in raw_expected),
            f"{case_id}.expected_h_only_first512_ids invalid")
    require(PAD not in raw_expected and EOS not in raw_expected[:-1],
            f"{case_id}.expected_h_only_first512_ids has invalid stop tokens")
    expected_h_free = list(raw_expected)
    require(len(expected_h_free) <= 512, f"{case_id}: h parity suffix exceeds 512")
    candidates = [validate_complete_row(c, case_id=case_id)
                  for c in case.get("candidates", [])]
    require(len(candidates) <= 2, f"{case_id}: more than two candidates")
    orders = [c["source_free_complete_order"] for c in candidates]
    require(orders == sorted(orders), f"{case_id}: candidate order is nondeterministic")
    require(len({c["candidate_id"] for c in candidates}) == len(candidates),
            f"{case_id}: duplicate candidate IDs")
    for candidate in candidates:
        require(len(h_ids) + len(candidate["c_ids"]) < CAP,
                f"{case_id}/{candidate['candidate_id']}: no free-token room")
    return {
        "case_id": case_id,
        "source_case": source,
        "prompt_token_ids": prompt,
        "baseline_action_ids": baseline,
        "golden": case.get("golden"),
        "h_ids": h_ids,
        "h_ids_sha256": case["h_ids_sha256"],
        "h_complete_row_count": case["h_complete_row_count"],
        "expected_h_only_first512_ids": expected_h_free,
        "candidates": candidates,
    }


def validate_packet(packet: Mapping[str, Any], *, verify_sources: bool = True) -> dict[str, Any]:
    require(packet.get("schema") == SCHEMA, f"packet schema must be {SCHEMA}")
    require(packet.get("cap") == CAP, f"packet cap must be {CAP}")
    source_packet = packet.get("source_packet")
    require(isinstance(source_packet, str) and Path(source_packet).is_absolute(),
            "source_packet must be absolute")
    require(file_hash(Path(source_packet)) == packet.get("source_packet_sha256"),
            "source packet bytes changed")
    require(packet.get("runner_sha256") == file_hash(Path(__file__).resolve()),
            "runner bytes changed")
    manifest = packet.get("candidate_manifest")
    require(isinstance(manifest, str) and Path(manifest).is_absolute(),
            "candidate_manifest must be absolute")
    require(file_hash(Path(manifest)) == packet.get("candidate_manifest_sha256"),
            "candidate manifest bytes changed")
    source_files = packet.get("source_files")
    require(isinstance(source_files, Mapping) and source_files,
            "source_files must be a nonempty hash map")
    if verify_sources:
        for raw_path, expected in source_files.items():
            path = Path(raw_path)
            require(path.is_absolute() and path.is_file(), f"missing source file: {path}")
            require(isinstance(expected, str) and HEX64.fullmatch(expected),
                    f"bad source hash: {path}")
            require(file_hash(path) == expected, f"source file bytes changed: {path}")
    anchor = Path(str(packet.get("anchor_adapter", "")))
    require(anchor.is_absolute() and anchor.exists(), "anchor_adapter unavailable")
    raw_cases = packet.get("cases")
    require(isinstance(raw_cases, list) and len(raw_cases) == 4,
            "packet must contain frozen four cases")
    cases = [validate_case(case, index=i) for i, case in enumerate(raw_cases)]
    config = packet.get("config")
    require(isinstance(config, Mapping), "source config is required")
    limits = packet.get("execution_limits")
    require(isinstance(limits, Mapping), "execution_limits required")
    return {"cases": cases, "config": config, "anchor_adapter": anchor}


def jobs_for_case(case: Mapping[str, Any]) -> list[dict[str, Any]]:
    jobs = [
        {"job_id": "natural_anchor", "prefix_ids": [], "budget": CAP,
         "kind": "natural_anchor", "candidate": None},
        {"job_id": "h_only", "prefix_ids": list(case["h_ids"]),
         "budget": CAP - len(case["h_ids"]), "kind": "h_only", "candidate": None},
    ]
    for candidate in case["candidates"]:
        prefix = list(case["h_ids"]) + list(candidate["c_ids"])
        jobs.append({
            "job_id": f"h_plus_{candidate['candidate_id']}",
            "prefix_ids": prefix,
            "budget": CAP - len(prefix),
            "kind": "h_plus_c",
            "candidate": candidate,
        })
    return jobs


def row_ledger(parsed: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [dict(row, parser_disposition="valid") for row in parsed.get("pred", [])]
    rows.extend(dict(row, parser_disposition="dropped")
                for row in parsed.get("dropped_predictions", []))
    return sorted(rows, key=lambda row: row.get("generated_order", -1))


def partition_parser_rows(parsed: Mapping[str, Any], *, prefix_row_count: int) -> dict[str, Any]:
    rows = row_ledger(parsed)
    prefix_rows = [row for row in rows if row.get("generated_order", -1) < prefix_row_count]
    free_rows = [row for row in rows if row.get("generated_order", -1) >= prefix_row_count]
    invalid_complete = [
        row for row in free_rows
        if row["parser_disposition"] == "dropped"
        and isinstance(row.get("raw_span_text"), str)
        and row["raw_span_text"].endswith("<|box_end|>")
    ]
    incomplete = [
        row for row in free_rows
        if row["parser_disposition"] == "dropped" and row not in invalid_complete
    ]
    valid_free = [row for row in free_rows if row["parser_disposition"] == "valid"]
    return {
        "prefix_row_count": prefix_row_count,
        "observed_prefix_rows": prefix_rows,
        "free_rows": free_rows,
        "valid_free_rows": valid_free,
        "complete_invalid_free_rows": invalid_complete,
        "incomplete_free_fragments": incomplete,
        "counts": {
            "all_free_rows_or_fragments": len(free_rows),
            "valid_complete_free_rows": len(valid_free),
            "invalid_complete_free_rows": len(invalid_complete),
            "incomplete_free_fragments": len(incomplete),
        },
    }


def credit_identity(job: Mapping[str, Any], partition: Mapping[str, Any]) -> dict[str, Any]:
    candidate = job.get("candidate")
    return {
        "natural_owner_recovery_eligible": job["kind"] in ("natural_anchor", "h_only"),
        "conditional_free_suffix_review_eligible": job["kind"] == "h_plus_c",
        "forced_candidate_row_count": 1 if candidate is not None else 0,
        "forced_candidate_in_free_counts": False,
        "credited_free_generated_orders": [
            row.get("generated_order") for row in partition["valid_free_rows"]
        ],
        "candidate_id": candidate.get("candidate_id") if candidate else None,
    }


def validate_suffix(suffix: list[int], *, budget: int, stop: str) -> None:
    require(len(suffix) <= budget, "suffix exceeds budget")
    require(PAD not in suffix, "suffix contains pad token")
    require(EOS not in suffix[:-1], "suffix contains early EOS")
    if stop == "im_end":
        require(bool(suffix) and suffix[-1] == EOS, "im_end suffix must end in EOS")
    elif stop == "length":
        require(len(suffix) == budget and EOS not in suffix, "length suffix must fill budget")
    else:
        raise ValueError(f"unsupported stop reason: {stop}")


def resolve_effective_config(packet: Mapping[str, Any], anchor: Path) -> Any:
    from src.config.inference import InferConfig

    config = InferConfig.model_validate(packet["config"])
    require(config.backend.type == "hf" and config.model.dtype == "fp32",
            "requires HF FP32")
    require(config.backend.hf.attn_implementation == "sdpa" and
            config.backend.hf.patch_embed_linearization == "enabled",
            "requires SDPA and patch linearization")
    effective = checkpoint_config(config, anchor)
    require(effective.adapter.path == str(anchor), "effective adapter differs")
    return effective


def batch_identity(batch: Any) -> dict[str, Any]:
    value = {
        "request_ids": list(batch.request_ids),
        "prompt_token_ids": [list(v) for v in batch.prompt_token_ids],
        "image_grids": [list(v) if v is not None else None for v in batch.image_grids],
        "media_sha256": (
            list(batch.media_sha256) if batch.media_sha256 is not None else None
        ),
    }
    return {"value": value, "identity_sha256": digest_json(value)}


def rss_peak_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate_record(record: Mapping[str, Any], *, case: Mapping[str, Any],
                    job: Mapping[str, Any], packet_sha256: str) -> None:
    require(record.get("schema") == RECORD_SCHEMA, "record schema differs")
    require(record.get("packet_sha256") == packet_sha256, "record packet differs")
    require(record.get("case_id") == case["case_id"], "record case differs")
    require(record.get("job_id") == job["job_id"], "record job differs")
    require(record.get("prefix_ids") == job["prefix_ids"], "record prefix differs")
    require(record.get("action_ids") == job["prefix_ids"] + record.get("free_ids", []),
            "record action is not prefix plus free suffix")
    validate_suffix(record["free_ids"], budget=job["budget"], stop=record["stop_reason"])
    if job["kind"] == "natural_anchor":
        require(record.get("baseline_match") is True, "natural anchor parity failed")
    if job["kind"] == "h_only":
        expected = case["expected_h_only_first512_ids"]
        require(record["free_ids"][:len(expected)] == expected,
                "h-only first512 parity failed")
    credit = record.get("credit_identity")
    require(credit.get("forced_candidate_in_free_counts") is False,
            "forced candidate entered free credit")
    if job["kind"] == "h_plus_c":
        require(credit.get("forced_candidate_row_count") == 1,
                "candidate forced-row accounting differs")


def selected_jobs(case: Mapping[str, Any], *, smoke: bool) -> list[dict[str, Any]]:
    jobs = jobs_for_case(case)
    if not smoke:
        return jobs
    require(case["case_id"] == COHORT[0], "smoke is the full 351017 case")
    require(case["candidates"], "smoke requires an admitted candidate")
    # The smoke is one complete case: parity, h-only, and both selected
    # force-and-release cells. Its four records are the final rank-0 records
    # and are never regenerated.
    return jobs


def run(packet_path: Path, out_dir: Path, *, rank: int, smoke: bool,
        reuse_records: Path | None) -> dict[str, Any]:
    require(packet_path.is_absolute() and packet_path.is_file(), "packet path invalid")
    require(out_dir.is_absolute() and not out_dir.exists(), "output dir occupied or relative")
    require(type(rank) is int and 0 <= rank < 4, "rank must be 0..3")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == str(rank),
            "CUDA_VISIBLE_DEVICES must equal physical rank")
    packet = json.loads(packet_path.read_text())
    plan = validate_packet(packet)
    case = plan["cases"][rank]
    complete_jobs = selected_jobs(case, smoke=smoke)
    jobs = list(complete_jobs)
    reused: list[dict[str, Any]] = []
    packet_sha = file_hash(packet_path)
    if reuse_records is not None:
        require(not smoke, "smoke cannot reuse records")
        require(reuse_records.is_absolute() and reuse_records.is_file(),
                "reuse records invalid")
        reused = read_jsonl(reuse_records)
        by_id = {job["job_id"]: job for job in jobs}
        require(len(reused) == len({row["job_id"] for row in reused}),
                "duplicate reused job")
        for record in reused:
            require(record.get("case_id") == case["case_id"], "reused case differs")
            require(record.get("job_id") in by_id, "reused job outside case plan")
            validate_record(record, case=case, job=by_id[record["job_id"]],
                            packet_sha256=packet_sha)
        reused_ids = {row["job_id"] for row in reused}
        jobs = [job for job in jobs if job["job_id"] not in reused_ids]
    require(jobs or reused, "no selected or reusable records")
    selected_token_bound = sum(job["budget"] for job in jobs)
    total_bound = selected_token_bound + sum(len(row["free_ids"]) for row in reused)
    require(total_bound <= packet["execution_limits"]["selected_generated_token_bound"],
            "selected work exceeds packet generated-token bound")

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir()
    records_path = out_dir / "records.jsonl"
    terminal: dict[str, Any] = {
        "schema": "native_escape_witness.terminal.v1",
        "status": "running", "exit_code": None, "rank": rank, "smoke": smoke,
        "packet_sha256": packet_sha, "selected_new_jobs": len(jobs),
        "reused_jobs": len(reused), "selected_generated_token_bound": selected_token_bound,
        "model_loads": 0, "continuations": 0, "new_tokens": 0,
        "model_forwards": 0, "image_forwards": 0, "started_unix": time.time(),
    }
    publish(out_dir / "launch.json", {
        "schema": "native_escape_witness.launch.v1", "pid": os.getpid(),
        "rank": rank, "smoke": smoke, "packet": str(packet_path),
        "packet_sha256": packet_sha, "jobs": [job["job_id"] for job in jobs],
        "reuse_records": str(reuse_records) if reuse_records else None,
    })
    effective = resolve_effective_config(packet, plan["anchor_adapter"])
    publish(out_dir / "config.json", {
        "schema": "native_escape_witness.config.v1",
        "source_config": packet["config"],
        "effective_config": effective.model_dump(mode="json"),
        "policy": {"temperature": 0.0, "top_p": 1.0, "top_k": 0,
                   "repetition_penalty": 1.0, "use_model_defaults": False,
                   "eos_token_id": EOS},
    })

    handles: list[Any] = []
    error: BaseException | None = None
    torch_module: Any = None
    counters = {"model_forwards": 0, "image_forwards": 0}
    old_alarm = signal.getsignal(signal.SIGALRM)
    started = time.monotonic()
    try:
        def expired(_signum: int, _frame: Any) -> None:
            raise TimeoutError(f"hard wall exceeded {HARD_WALL_SECONDS} seconds")

        signal.signal(signal.SIGALRM, expired)
        signal.alarm(HARD_WALL_SECONDS)
        import torch
        torch_module = torch
        from probes.dora_owner_learning.runtime import load_policy
        from probes.source_rweak_row_cross.run import build_requests, native_record
        from src.qwen.generation import NativeGenerationPolicy, generate_continuations
        from src.qwen.native import prepare_native_inputs

        require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
                "exactly one visible CUDA device required")
        qwen, model_identity = load_policy(effective, device=torch.device("cuda:0"))
        terminal["model_loads"] = 1
        require(model_identity["effective_settings"]["observed_attn_implementation"] == "sdpa",
                "live attention is not SDPA")
        require(model_identity["effective_settings"]["observed_model_dtype"]
                ["parameter_dtype_names"] == ["torch.float32"],
                "live parameters are not FP32")
        adapter = model_identity["model_identity"]["adapter"]
        require(adapter.get("adapter_path") == str(plan["anchor_adapter"]),
                "live adapter is not Stable50")
        require(adapter.get("merged_adapters") == [], "unexpected merged adapter")
        require(qwen.tokenizer.convert_tokens_to_ids("<|im_end|>") == EOS,
                "tokenizer EOS differs")
        h_text = qwen.tokenizer.decode(case["h_ids"], skip_special_tokens=False)
        h_parsed = native_record(h_text, {"row_id": case["source_case"]["row_id"]},
                                 case["golden"], "length")
        require(len(row_ledger(h_parsed)) == case["h_complete_row_count"] and
                not h_parsed.get("dropped_predictions"),
                "literal h does not parse as the frozen complete-row sequence")
        for candidate in case["candidates"]:
            require(qwen.tokenizer.decode(candidate["c_ids"], skip_special_tokens=False) ==
                    candidate["raw_span_text"],
                    f"{candidate['candidate_id']}: literal c token decode differs")
            c_parsed = native_record(candidate["raw_span_text"],
                                     {"row_id": case["source_case"]["row_id"]},
                                     case["golden"], "length")
            require(len(c_parsed.get("pred", [])) == 1 and
                    not c_parsed.get("dropped_predictions"),
                    f"{candidate['candidate_id']}: literal c does not parse as one valid row")
        publish(out_dir / "model.json", {
            "schema": "native_escape_witness.model.v1", "rank": rank,
            "identity": model_identity, "eos_token_id": EOS,
            "pad_token_id": qwen.tokenizer.pad_token_id,
        })
        qwen.model.eval()
        for parameter in qwen.model.parameters():
            parameter.requires_grad_(False)
        visuals = [module for name, module in qwen.model.named_modules()
                   if name.endswith("visual")]
        require(len(visuals) == 1, "ambiguous visual module")

        def model_hook(*_: Any) -> None:
            counters["model_forwards"] += 1

        def image_hook(*_: Any) -> None:
            counters["image_forwards"] += 1

        handles.append(qwen.model.register_forward_pre_hook(model_hook))
        handles.append(visuals[0].register_forward_pre_hook(image_hook))
        requests, _ = build_requests(qwen, packet["config"], [case["source_case"]])
        batch = prepare_native_inputs(qwen.processor, requests, device=torch.device("cuda:0"),
                                      record_media_identity=True)
        require(list(batch.prompt_token_ids[0]) == case["prompt_token_ids"],
                "live prompt IDs differ")
        expected_grid = case["source_case"]["image_plan"]["observed_image_grid_thw"]
        require(list(batch.image_grids[0]) == expected_grid, "live image grid differs")
        identity = batch_identity(batch)
        publish(out_dir / "batch.json", {"schema": "native_escape_witness.batch.v1",
                                         **identity})
        policy = NativeGenerationPolicy(temperature=0.0, top_p=1.0,
                                        repetition_penalty=1.0, top_k=0,
                                        use_model_defaults=False)
        with records_path.open("x", encoding="utf-8") as stream:
            for reused_record in reused:
                stored = dict(reused_record, reuse={
                    "source": str(reuse_records),
                    "source_record_sha256": digest_json(reused_record),
                })
                stream.write(json.dumps(stored, ensure_ascii=False, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            for job in jobs:
                before_model = counters["model_forwards"]
                before_image = counters["image_forwards"]
                tick = time.monotonic()
                with torch.inference_mode():
                    result = generate_continuations(
                        qwen.model, batch, extensions=[job["prefix_ids"]],
                        budgets=[job["budget"]], eos_token_id=EOS,
                        pad_token_id=qwen.tokenizer.pad_token_id, policy=policy,
                        trace="none",
                    )[0]
                suffix = list(result.token_ids)
                validate_suffix(suffix, budget=job["budget"], stop=result.stop_reason)
                action = job["prefix_ids"] + suffix
                text = qwen.tokenizer.decode(action, skip_special_tokens=False)
                parsed = native_record(text, {"row_id": case["source_case"]["row_id"]},
                                       case["golden"], result.stop_reason)
                prefix_rows = case["h_complete_row_count"] + (1 if job["candidate"] else 0)
                if job["kind"] == "natural_anchor":
                    prefix_rows = 0
                partition = partition_parser_rows(parsed, prefix_row_count=prefix_rows)
                record = {
                    "schema": RECORD_SCHEMA, "packet_sha256": packet_sha,
                    "case_id": case["case_id"],
                    "source_row_id": case["source_case"]["row_id"],
                    "job_id": job["job_id"], "kind": job["kind"],
                    "request_id": result.request_id, "prefix_ids": job["prefix_ids"],
                    "h_ids": case["h_ids"], "forced_candidate": job["candidate"],
                    "free_ids": suffix, "action_ids": action, "budget": job["budget"],
                    "text": text, "stop_reason": result.stop_reason,
                    "native_eos_observed": result.stop_reason == "im_end",
                    "eos_is_positive_outcome": False,
                    "parsed": parsed, "parser_partition": partition,
                    "credit_identity": credit_identity(job, partition),
                    "baseline_match": action == case["baseline_action_ids"]
                    if job["kind"] == "natural_anchor" else None,
                    "elapsed_seconds": time.monotonic() - tick,
                    "model_forwards": counters["model_forwards"] - before_model,
                    "image_forwards": counters["image_forwards"] - before_image,
                    "batch_identity_sha256": identity["identity_sha256"],
                }
                validate_record(record, case=case, job=job, packet_sha256=packet_sha)
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                terminal["continuations"] += 1
                terminal["new_tokens"] += len(suffix)
                terminal["model_forwards"] = counters["model_forwards"]
                terminal["image_forwards"] = counters["image_forwards"]

        durable = read_jsonl(records_path)
        expected_ids = {job["job_id"] for job in complete_jobs}
        require({row["job_id"] for row in durable} == expected_ids,
                "cold readback job set differs")
        by_id = {job["job_id"]: job for job in complete_jobs}
        for record in durable:
            validate_record(record, case=case, job=by_id[record["job_id"]],
                            packet_sha256=packet_sha)
        readback = {
            "schema": "native_escape_witness.readback.v1", "status": "accepted",
            "record_count": len(durable), "job_ids": sorted(expected_ids),
            "records_sha256": file_hash(records_path),
            "reused_record_count": len(reused),
        }
        publish(out_dir / "readback.json", readback)
        terminal["readback"] = readback
        terminal["status"] = "completed"
        terminal["exit_code"] = 0
    except BaseException as exc:
        error = exc
        terminal["status"] = "failed"
        terminal["exit_code"] = 1
        terminal["error"] = repr(exc)
        terminal["traceback"] = traceback.format_exc()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)
        for handle in handles:
            try:
                handle.remove()
            except Exception:
                pass
        if torch_module is not None and torch_module.cuda.is_initialized():
            try:
                torch_module.cuda.synchronize(torch_module.device("cuda:0"))
            except BaseException as exc:
                terminal["cuda_sync_error"] = repr(exc)
            try:
                terminal["peak_cuda_allocated_bytes"] = int(
                    torch_module.cuda.max_memory_allocated(torch_module.device("cuda:0")))
                terminal["peak_cuda_reserved_bytes"] = int(
                    torch_module.cuda.max_memory_reserved(torch_module.device("cuda:0")))
            except BaseException as exc:
                terminal["peak_cuda_error"] = repr(exc)
        terminal["peak_rss_bytes"] = rss_peak_bytes()
        terminal["elapsed_seconds"] = time.monotonic() - started
        bound_errors = []
        if terminal["model_forwards"] > selected_token_bound:
            bound_errors.append("model_forwards")
        if terminal["image_forwards"] > len(jobs):
            bound_errors.append("image_forwards")
        if terminal["new_tokens"] > selected_token_bound:
            bound_errors.append("new_tokens")
        if terminal.get("peak_cuda_allocated_bytes", 0) > 12 * 1024**3:
            bound_errors.append("cuda")
        if terminal["peak_rss_bytes"] > 16 * 1024**3:
            bound_errors.append("rss")
        if terminal["elapsed_seconds"] > HARD_WALL_SECONDS + 5:
            bound_errors.append("wall")
        terminal["bound_errors"] = bound_errors
        if bound_errors and terminal["status"] == "completed":
            terminal["status"] = "failed"
            terminal["exit_code"] = 1
            error = ValueError(f"runtime bounds exceeded: {bound_errors}")
        publish(out_dir / "terminal.json", terminal)
    if error is not None:
        raise error
    return terminal


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--reuse-records", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args.packet, args.out_dir, rank=args.rank, smoke=args.smoke,
        reuse_records=args.reuse_records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
