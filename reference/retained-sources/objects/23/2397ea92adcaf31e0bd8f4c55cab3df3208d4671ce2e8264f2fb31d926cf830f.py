"""Bounded native probe for small-box drifting repetition.

The packet is prepared by the research lead.  This module only validates the
packet, loads the frozen Source/Stable50 native policy once, and acquires the
literal continuation cells named by the packet.  It deliberately does not
select cases, translate histories, parse geometry into a new metric, train, or
manage recovery.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
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

# This output directory is outside the checkout.  The runner is normally
# invoked with the research-probes worktree as cwd, but retaining the fallback
# makes the absolute CLI entry point deterministic as well.
_WORKTREE = Path.cwd()
if not (_WORKTREE / "src").is_dir():
    _fallback = Path("/data/CoordExp/.worktrees/research-probes")
    if (_fallback / "src").is_dir():
        _WORKTREE = _fallback
if str(_WORKTREE) not in os.sys.path:
    os.sys.path.insert(0, str(_WORKTREE))

from probes.dora_owner_learning.candidate_opportunity import (  # noqa: E402
    file_hash,
)
from probes.dora_owner_learning.route_access import (  # noqa: E402
    checkpoint_config,
    publish,
)


SCHEMA = "small_owner_repeat_origin.v1"
RECORD_SCHEMA = "small_owner_repeat_origin.record.v1"
EOS = 151645
PAD = 151643
CAP = 3084
CONDITIONAL_BUDGET = 512
HARD_WALL_SECONDS = 1500
MAX_CASES = 8
MAX_JOBS_PER_CASE = 9
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_CONDITIONS = ("original", "donor")
_HISTORY_CONDITIONS = ("native", "translated")
_BOUNDARIES = ("early", "late")


def require(condition: Any, message: str) -> None:
    """Raise a plain, testable contract error for a failed probe invariant."""

    if not condition:
        raise ValueError(message)


def _is_int(value: Any) -> bool:
    return type(value) is int


def _token_ids(value: Any, *, field: str, allow_empty: bool = True) -> list[int]:
    require(isinstance(value, list), f"{field} must be a list")
    require(allow_empty or bool(value), f"{field} must be nonempty")
    require(
        all(_is_int(token) and token >= 0 for token in value),
        f"{field} must contain nonnegative integer token IDs",
    )
    return list(value)


def _validate_action_ids(ids: Any, *, field: str) -> list[int]:
    values = _token_ids(ids, field=field, allow_empty=False)
    require(len(values) <= CAP, f"{field} exceeds action cap")
    require(PAD not in values, f"{field} contains pad token")
    require(EOS not in values[:-1], f"{field} contains EOS before its end")
    require(
        values[-1] == EOS or len(values) == CAP,
        f"{field} must terminate with EOS or exactly fill the action cap",
    )
    return values


def _validate_extension(ids: Any, *, field: str) -> list[int]:
    values = _token_ids(ids, field=field)
    require(len(values) <= CAP, f"{field} exceeds action cap")
    require(PAD not in values, f"{field} contains pad token")
    require(EOS not in values, f"{field} cannot contain EOS")
    return values


def _dimensions(case: Mapping[str, Any], *, label: str) -> tuple[int, int]:
    require(isinstance(case, Mapping), f"{label} must be an object")
    width, height = case.get("image_width"), case.get("image_height")
    require(
        _is_int(width) and width > 0 and _is_int(height) and height > 0,
        f"{label} image dimensions must be positive integers",
    )
    return int(width), int(height)


def _row_id(case: Mapping[str, Any], *, label: str) -> str:
    value = case.get("row_id")
    require(isinstance(value, str) and bool(value), f"{label}.row_id must be nonempty")
    return value


def _image_path(case: Mapping[str, Any], *, label: str) -> Path:
    value = case.get("image_path")
    require(isinstance(value, str) and bool(value), f"{label}.image_path is required")
    path = Path(value)
    require(path.is_absolute(), f"{label}.image_path must be absolute")
    return path


def _case_image_digest(case: Mapping[str, Any], *, label: str) -> str | None:
    plan = case.get("image_plan")
    if not isinstance(plan, Mapping) or "image_content_sha256" not in plan:
        return None
    digest = plan["image_content_sha256"]
    require(isinstance(digest, str) and _HEX64.fullmatch(digest),
            f"{label}.image_plan.image_content_sha256 is invalid")
    return digest


def _validate_image_plan(case: Mapping[str, Any], *, label: str) -> Mapping[str, Any]:
    plan = case.get("image_plan")
    require(isinstance(plan, Mapping), f"{label}.image_plan is required")
    for field in ("image_content_sha256", "observed_image_grid_thw",
                  "logical_transform_id", "merged_visual_tokens",
                  "backend_prompt_token_count"):
        require(field in plan, f"{label}.image_plan.{field} is required")
    grid = plan["observed_image_grid_thw"]
    require(isinstance(grid, (list, tuple)) and len(grid) == 3 and
            all(_is_int(value) and value > 0 for value in grid),
            f"{label}.image_plan.observed_image_grid_thw is invalid")
    require(_is_int(plan["merged_visual_tokens"]) and plan["merged_visual_tokens"] > 0,
            f"{label}.image_plan.merged_visual_tokens is invalid")
    require(_is_int(plan["backend_prompt_token_count"]) and
            plan["backend_prompt_token_count"] > 0,
            f"{label}.image_plan.backend_prompt_token_count is invalid")
    require(isinstance(plan["logical_transform_id"], str) and
            bool(plan["logical_transform_id"]),
            f"{label}.image_plan.logical_transform_id is invalid")
    return plan


def _check_optional_prompt_binding(
    case: Mapping[str, Any], expected: Sequence[int], *, label: str
) -> None:
    """Check packet-side prompt fields when a case carries them.

    Historical source cases usually carry the prompt width in image_plan while
    the new packet carries the complete literal IDs.  Donor plans may copy
    source shape/grid metadata, so no inferred-media field is treated as a
    historical execution receipt here.
    """

    for owner, value in ((label, case.get("prompt_token_ids")),
                         (f"{label}.image_plan", (case.get("image_plan") or {}).get("prompt_token_ids")
                          if isinstance(case.get("image_plan"), Mapping) else None)):
        if value is not None:
            require(_token_ids(value, field=f"{owner}.prompt_token_ids") == list(expected),
                    f"{owner}.prompt_token_ids differs from packet prompt IDs")


def _expected_job_tuples() -> set[tuple[str, str, str]]:
    return {
        ("natural", "original", "native"),
        *{
            (boundary, image, history)
            for boundary in _BOUNDARIES
            for image in _IMAGE_CONDITIONS
            for history in _HISTORY_CONDITIONS
        },
    }


def _job_tuple(job: Mapping[str, Any]) -> tuple[str, str, str]:
    boundary, image, history = (
        job.get("boundary"),
        job.get("image_condition"),
        job.get("history_condition"),
    )
    require(boundary in ("natural", "early", "late"), "job boundary is invalid")
    require(image in _IMAGE_CONDITIONS, "job image_condition is invalid")
    require(history in _HISTORY_CONDITIONS, "job history_condition is invalid")
    return str(boundary), str(image), str(history)


def validate_job(job: Mapping[str, Any], *, case_id: str) -> dict[str, Any]:
    """Validate one packet job and return its normalized literal fields."""

    require(isinstance(job, Mapping), f"{case_id}: job must be an object")
    job_id = job.get("job_id")
    require(isinstance(job_id, str) and bool(job_id), f"{case_id}: job_id is required")
    boundary, image, history = _job_tuple(job)
    extension = _validate_extension(job.get("extension_ids"),
                                    field=f"{case_id}/{job_id}.extension_ids")
    budget = job.get("budget")
    require(_is_int(budget) and 0 <= budget <= CAP,
            f"{case_id}/{job_id}.budget must be in [0,{CAP}]")
    if (boundary, image, history) == ("natural", "original", "native"):
        require(not extension, f"{case_id}/{job_id}: natural extension must be empty")
        require(budget == CAP, f"{case_id}/{job_id}: natural budget must be {CAP}")
    elif (boundary, image, history) == ("natural", "donor", "native"):
        require(job_id == "donor_natural",
                f"{case_id}: the optional donor natural control must be donor_natural")
        require(not extension,
                f"{case_id}/{job_id}: donor natural extension must be empty")
        require(budget == CAP,
                f"{case_id}/{job_id}: donor natural budget must be {CAP}")
    else:
        require(boundary in _BOUNDARIES,
                f"{case_id}/{job_id}: conditional job must use early or late boundary")
        expected_budget = min(CONDITIONAL_BUDGET, CAP - len(extension))
        require(budget == expected_budget,
                f"{case_id}/{job_id}: budget must be min({CONDITIONAL_BUDGET}, remaining cap)")
    return {
        "job_id": job_id,
        "image_condition": image,
        "history_condition": history,
        "boundary": boundary,
        "extension_ids": extension,
        "budget": int(budget),
    }


def _validate_case(case: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    require(isinstance(case, Mapping), f"case[{index}] must be an object")
    case_id = case.get("case_id")
    require(isinstance(case_id, str) and bool(case_id), f"case[{index}].case_id is required")
    source = case.get("source_case")
    donor = case.get("donor_case")
    source_row = _row_id(source, label=f"{case_id}.source_case")
    donor_row = _row_id(donor, label=f"{case_id}.donor_case")
    require(source_row != donor_row, f"{case_id}: source and donor row IDs must differ")
    require(_dimensions(source, label=f"{case_id}.source_case") ==
            _dimensions(donor, label=f"{case_id}.donor_case"),
            f"{case_id}: source and donor image dimensions differ")
    _image_path(source, label=f"{case_id}.source_case")
    _image_path(donor, label=f"{case_id}.donor_case")

    prompt_ids = _token_ids(case.get("prompt_token_ids"),
                            field=f"{case_id}.prompt_token_ids", allow_empty=False)
    source_plan = _validate_image_plan(source, label=f"{case_id}.source_case")
    donor_plan = _validate_image_plan(donor, label=f"{case_id}.donor_case")
    require(tuple(source_plan["observed_image_grid_thw"]) ==
            tuple(donor_plan["observed_image_grid_thw"]),
            f"{case_id}: packet source/donor image_grid_thw differ")
    require(source_plan["backend_prompt_token_count"] == len(prompt_ids) and
            donor_plan["backend_prompt_token_count"] == len(prompt_ids),
            f"{case_id}: packet prompt token width differs from image plan")
    require(source_plan["merged_visual_tokens"] == donor_plan["merged_visual_tokens"],
            f"{case_id}: packet source/donor visual token shape differs")
    baseline_ids = _validate_action_ids(case.get("baseline_action_ids"),
                                        field=f"{case_id}.baseline_action_ids")
    require(isinstance(case.get("golden"), Mapping), f"{case_id}.golden must be an object")
    _check_optional_prompt_binding(source, prompt_ids, label=f"{case_id}.source_case")
    _check_optional_prompt_binding(donor, prompt_ids, label=f"{case_id}.donor_case")

    jobs = case.get("jobs")
    require(isinstance(jobs, list) and MAX_JOBS_PER_CASE <= len(jobs) <= MAX_JOBS_PER_CASE + 1,
            f"{case_id}: expected {MAX_JOBS_PER_CASE} jobs plus at most one donor natural control")
    normalized_jobs = [validate_job(job, case_id=case_id) for job in jobs]
    require(len({job["job_id"] for job in normalized_jobs}) == len(normalized_jobs),
            f"{case_id}: duplicate job IDs")
    tuples = [(job["boundary"], job["image_condition"], job["history_condition"])
              for job in normalized_jobs]
    optional = [item for item in tuples
                if item == ("natural", "donor", "native")]
    require(len(optional) <= 1,
            f"{case_id}: at most one donor_natural control is allowed")
    factorial = {item for item in tuples
                 if item != ("natural", "donor", "native")}
    require(factorial == _expected_job_tuples() and
            len(tuples) == MAX_JOBS_PER_CASE + len(optional),
            f"{case_id}: missing or extra factorial job cells")

    # Source and donor hashes are verified again by NativeRequest at live
    # preparation.  This CPU check catches a changed image before model load
    # when the packet supplied the corresponding raw-byte digest.
    for label, item in ((f"{case_id}.source_case", source), (f"{case_id}.donor_case", donor)):
        expected_digest = _case_image_digest(item, label=label)
        if expected_digest is not None:
            path = _image_path(item, label=label)
            require(path.is_file(), f"{label}.image_path does not exist")
            require(file_hash(path) == expected_digest, f"{label}: image bytes changed")

    return {
        "case_id": case_id,
        "source_case": source,
        "donor_case": donor,
        "source_row_id": source_row,
        "donor_row_id": donor_row,
        "prompt_token_ids": prompt_ids,
        "baseline_action_ids": baseline_ids,
        "golden": case["golden"],
        "jobs": normalized_jobs,
    }


def _anchor_path(value: Any) -> Path:
    if isinstance(value, str):
        candidate = value
    elif isinstance(value, Mapping):
        candidate = value.get("root", value.get("path"))
    else:
        candidate = None
    require(isinstance(candidate, str) and bool(candidate), "anchor_adapter path is required")
    path = Path(candidate)
    require(path.is_absolute(), "anchor_adapter path must be absolute")
    require(path.exists(), "anchor_adapter path does not exist")
    return path


def _validate_source_files(source_files: Any) -> dict[str, str]:
    require(isinstance(source_files, Mapping) and bool(source_files),
            "source_files must be a nonempty absolute-path hash map")
    observed: dict[str, str] = {}
    for raw_path, expected in source_files.items():
        require(isinstance(raw_path, str) and Path(raw_path).is_absolute(),
                "source_files keys must be absolute paths")
        require(isinstance(expected, str) and _HEX64.fullmatch(expected),
                f"source_files hash is invalid: {raw_path}")
        path = Path(raw_path)
        require(path.is_file(), f"source file does not exist: {raw_path}")
        actual = file_hash(path)
        require(actual == expected, f"source file hash changed: {raw_path}")
        observed[raw_path] = actual
    return observed


def _resolve_effective_config(packet: Mapping[str, Any], anchor: Path) -> Any:
    from src.config.inference import InferConfig

    source = packet.get("config")
    require(isinstance(source, Mapping), "config must be the source-config JSON object")
    try:
        config = InferConfig.model_validate(source)
    except Exception as exc:  # pydantic's concrete error is useful to callers
        raise ValueError(f"invalid source config: {exc}") from exc
    require(config.adapter is not None and config.embedding_delta is not None,
            "Source policy requires adapter and Source embedding delta")
    require(config.backend.type == "hf" and config.model.dtype == "fp32",
            "probe requires HF FP32 source configuration")
    require(config.backend.hf.attn_implementation == "sdpa" and
            config.backend.hf.patch_embed_linearization == "enabled",
            "probe requires FP32 SDPA patch-linearization source configuration")
    effective = checkpoint_config(config, anchor)
    source_dump = config.model_dump(mode="json")
    effective_dump = effective.model_dump(mode="json")
    for field in ("model", "data", "template", "backend", "generation", "scoring",
                  "artifacts", "debug", "embedding_delta", "run"):
        require(effective_dump[field] == source_dump[field],
                f"anchor switch changed source config field: {field}")
    require(effective_dump["adapter"]["path"] == str(anchor),
            "effective config does not name anchor_adapter")
    return effective


def validate_packet(packet: Mapping[str, Any], *, verify_sources: bool = True) -> dict[str, Any]:
    """Validate packet semantics without loading a model.

    The returned value is normalized only for the fields owned by this runner;
    golden records and source/donor case payloads remain opaque.
    """

    require(isinstance(packet, Mapping), "packet must be a JSON object")
    require(packet.get("schema") == SCHEMA, f"packet schema must be {SCHEMA}")
    if verify_sources:
        _validate_source_files(packet.get("source_files"))
    anchor = _anchor_path(packet.get("anchor_adapter"))
    raw_cases = packet.get("cases")
    require(isinstance(raw_cases, list) and 0 < len(raw_cases) <= MAX_CASES,
            f"packet cases must contain 1..{MAX_CASES} cases")
    normalized_cases = [_validate_case(case, index=i) for i, case in enumerate(raw_cases)]
    require(len({case["case_id"] for case in normalized_cases}) == len(normalized_cases),
            "duplicate packet case IDs")
    return {
        "schema": SCHEMA,
        "anchor_adapter": anchor,
        "cases": normalized_cases,
        "source_files": packet.get("source_files"),
        "config": packet.get("config"),
    }


def expected_job_ids(
    cases: Sequence[Mapping[str, Any]], *, job_id: str | None = None,
    smoke: bool = False,
) -> set[tuple[str, str]]:
    require(not (smoke and job_id is not None), "--smoke and --job-id are mutually exclusive")
    expected = set()
    for case in cases:
        for job in case["jobs"]:
            if smoke and not (
                (job["boundary"], job["image_condition"], job["history_condition"])
                == ("natural", "original", "native")
                or job["boundary"] == "early"
            ):
                continue
            expected.add((str(case["case_id"]), str(job["job_id"])))
    if job_id is not None:
        all_jobs = {
            (str(case["case_id"]), str(job["job_id"])): (case, job)
            for case in cases for job in case["jobs"]
        }
        selected = [key for key in all_jobs if key[1] == job_id]
        require(len(selected) == 1, f"unknown or ambiguous --job-id: {job_id}")
        case_id, selected_id = selected[0]
        case, job = all_jobs[(case_id, selected_id)]
        require((job["boundary"], job["image_condition"], job["history_condition"]) ==
                ("natural", "original", "native"),
                "--job-id smoke must select original native natural")
        return {selected[0]}
    return expected


def select_batch(image_condition: str, source_batch: Any, donor_batch: Any) -> Any:
    """Select the already-prepared native batch without changing its tensors."""

    require(image_condition in _IMAGE_CONDITIONS, "unknown image condition")
    return source_batch if image_condition == "original" else donor_batch


def _expected_request_id(case: Mapping[str, Any], job: Mapping[str, Any]) -> str:
    return case["source_row_id"] if job["image_condition"] == "original" else case["donor_row_id"]


def validate_record(
    record: Mapping[str, Any],
    *,
    case: Mapping[str, Any],
    job: Mapping[str, Any],
    packet_sha256: str,
    require_baseline: bool = True,
) -> dict[str, Any]:
    """Validate one durable JSONL record, including prefix/free accounting."""

    require(isinstance(record, Mapping), "record must be an object")
    require(record.get("schema") == RECORD_SCHEMA, "record schema mismatch")
    require(record.get("packet_sha256") == packet_sha256, "record packet identity mismatch")
    require(record.get("case_id") == case["case_id"], "record case identity mismatch")
    require(record.get("job_id") == job["job_id"], "record job identity mismatch")
    for field in ("image_condition", "history_condition", "boundary"):
        require(record.get(field) == job[field], f"record {field} mismatch")
    extension = _validate_extension(record.get("extension_ids"), field="record.extension_ids")
    require(extension == job["extension_ids"], "record extension differs from packet job")
    prefix = _token_ids(record.get("prefix_ids"), field="record.prefix_ids")
    require(prefix == extension, "record prefix/free partition has a shifted prefix")
    free = _token_ids(record.get("free_ids"), field="record.free_ids")
    action = _token_ids(record.get("action_ids"), field="record.action_ids", allow_empty=False)
    require(action == extension + free, "record action_ids differ from prefix + free_ids")
    require(len(free) <= job["budget"], "record free suffix exceeds job budget")
    stop = record.get("stop_reason", record.get("stop"))
    require(stop in ("im_end", "length"), "record stop reason is invalid")
    if stop == "length":
        require(len(free) == job["budget"], "length stop did not consume the job budget")
    else:
        require(bool(free) and free[-1] == EOS, "EOS stop is missing terminal EOS")
    require(record.get("text") and isinstance(record.get("text"), str),
            "record text is required")
    require(record.get("request_id") == _expected_request_id(case, job),
            "record request ID is not the selected image request")
    require(record.get("source_row_id") == case["source_row_id"] and
            record.get("donor_row_id") == case["donor_row_id"],
            "record source/donor row identity mismatch")
    donor = job["image_condition"] == "donor"
    owner_eligible = (
        job["image_condition"] == "original"
        and job["history_condition"] == "native"
    )
    expected_scope = (
        "counterfactual_donor_image"
        if donor
        else "source_image"
        if job["history_condition"] == "native"
        else "forced_history_diagnostic"
    )
    require(record.get("metrics_scope") == expected_scope,
            "record metrics scope marker mismatch")
    require(record.get("donor_metrics_counterfactual") is donor,
            "donor counterfactual marker mismatch")
    require(record.get("owner_preservation_eligible") is owner_eligible,
            "owner-preservation eligibility marker mismatch")
    if job["boundary"] == "natural" and job["image_condition"] == "original" and job["history_condition"] == "native":
        require(record.get("baseline_match") is (action == case["baseline_action_ids"]),
                "natural baseline_match marker is inconsistent")
        if require_baseline:
            require(action == case["baseline_action_ids"],
                    "original natural result differs from frozen baseline")
    return dict(record)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"missing records JSONL: {path}")
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            require(bool(line.strip()), f"blank records JSONL line {number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid records JSONL line {number}: {exc}") from exc
            require(isinstance(value, dict), f"records JSONL line {number} is not an object")
            records.append(value)
    return records


def validate_readback(
    records: Sequence[Mapping[str, Any]],
    *,
    cases: Sequence[Mapping[str, Any]],
    packet_sha256: str,
    job_id: str | None = None,
    smoke: bool = False,
    require_baseline: bool = True,
) -> dict[str, Any]:
    """Validate durable records and reject partial, duplicate, or extra cells."""

    by_case = {case["case_id"]: case for case in cases}
    expected = expected_job_ids(cases, job_id=job_id, smoke=smoke)
    seen: set[tuple[str, str]] = set()
    for record in records:
        key = (record.get("case_id"), record.get("job_id"))
        require(key not in seen, f"duplicate result cell: {key}")
        seen.add(key)
        require(key in expected, f"unexpected result cell: {key}")
        case = by_case.get(key[0])
        require(case is not None, f"unknown result case: {key[0]}")
        job = next(job for job in case["jobs"] if job["job_id"] == key[1])
        validate_record(record, case=case, job=job, packet_sha256=packet_sha256,
                        require_baseline=require_baseline)
    missing = expected - seen
    require(not missing, f"missing result cells: {sorted(missing)}")
    return {
        "expected_cells": len(expected),
        "observed_cells": len(seen),
        "job_ids": sorted(f"{case}:{job}" for case, job in seen),
    }


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _append_jsonl(path: Path, value: Mapping[str, Any], stream: Any) -> None:
    stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def _batch_identity(batch: Any) -> dict[str, Any]:
    prompts = [list(row) for row in batch.prompt_token_ids]
    grids = [None if grid is None else list(grid) for grid in batch.image_grids]
    media = None if batch.media_sha256 is None else list(batch.media_sha256)
    identity = {
        "request_ids": list(batch.request_ids),
        "prompt_token_ids": prompts,
        "image_grid_thw": grids,
        "media_sha256": media,
    }
    return {**identity, "identity_sha256": _digest(identity)}


def _rss_peak_bytes() -> int:
    # ru_maxrss is KiB on Linux, which is the execution platform for this
    # route.  Keep the conversion local rather than hiding it in a runtime.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def _resolve_job(case: Mapping[str, Any], job_id: str) -> Mapping[str, Any]:
    matches = [job for job in case["jobs"] if job["job_id"] == job_id]
    require(len(matches) == 1, f"unknown job {job_id} in case {case['case_id']}")
    return matches[0]


def run_probe(packet_path: str | Path, out_dir: str | Path, *, rank: int,
              job_id: str | None = None, smoke: bool = False) -> dict[str, Any]:
    """Execute the bounded native route; callers decide whether to launch it."""

    packet_path = Path(packet_path).expanduser()
    out_dir = Path(out_dir).expanduser()
    require(packet_path.is_absolute(), "--packet must be an absolute path")
    require(out_dir.is_absolute(), "--out-dir must be an absolute path")
    packet_path = packet_path.resolve()
    out_dir = out_dir.resolve()
    require(_is_int(rank) and 0 <= rank <= 7, "rank must be an integer in 0..7")
    require(not (smoke and job_id is not None), "--smoke and --job-id are mutually exclusive")
    require(packet_path.is_file(), f"missing packet: {packet_path}")
    require(not out_dir.exists(), f"output directory is occupied: {out_dir}")
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet_plan = validate_packet(packet, verify_sources=True)
    all_cases = packet_plan["cases"]
    require(rank < len(all_cases),
            f"rank {rank} has no assigned case (packet contains {len(all_cases)} cases)")
    # One rank owns one case.  This is the eight-rank bounded route; keeping
    # the assignment positional avoids inventing another packet sharding
    # contract and makes rank 4's smoke deterministically select its case.
    cases = [all_cases[rank]]
    packet_sha256 = file_hash(packet_path)
    expected_cells = expected_job_ids(cases, job_id=job_id, smoke=smoke)
    if job_id is None:
        selected = [
            (case, job)
            for case in cases
            for job in case["jobs"]
            if not smoke or job["boundary"] in ("natural", "early")
        ]
    else:
        selected = [(case, _resolve_job(case, job_id))
                    for case in cases if any(j["job_id"] == job_id for j in case["jobs"])]
        require(len(selected) == 1, f"unknown or ambiguous --job-id: {job_id}")
    selected_budget_total = sum(job["budget"] for _, job in selected)
    selected_job_count = len(selected)
    # This is a pre-GPU assignment guard.  It prevents a rank from silently
    # using another visible device; the caller owns CUDA process launch.
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == str(rank),
            "CUDA_VISIBLE_DEVICES must equal --rank before native execution")
    effective_config = _resolve_effective_config(packet, packet_plan["anchor_adapter"])

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir()
    records_path = out_dir / "records.jsonl"
    terminal: dict[str, Any] = {
        "schema": "small_owner_repeat_origin.terminal.v1",
        "status": "running",
        "exit_code": None,
        "rank": rank,
        "job_id": job_id,
        "smoke": smoke,
        "pid": os.getpid(),
        "packet_sha256": packet_sha256,
        "expected_cells": len(expected_cells),
        "selected_job_count": selected_job_count,
        "selected_budget_total": selected_budget_total,
        "model_loads": 0,
        "continuations": 0,
        "new_tokens": 0,
        "model_forwards": 0,
        "image_forwards": 0,
        "peak_cuda_allocated_bytes": None,
        "peak_cuda_reserved_bytes": None,
        "started_unix": time.time(),
    }
    publish(out_dir / "launch.json", {
        "schema": "small_owner_repeat_origin.launch.v1",
        "pid": os.getpid(),
        "rank": rank,
        "job_id": job_id,
        "smoke": smoke,
        "packet": str(packet_path),
        "packet_sha256": packet_sha256,
        "expected_cells": len(expected_cells),
    })
    publish(out_dir / "config.json", {
        "schema": "small_owner_repeat_origin.config.v1",
        "source_config": packet["config"],
        "effective_config": effective_config.model_dump(mode="json"),
        "anchor_adapter": str(packet_plan["anchor_adapter"]),
        "policy": {
            "temperature": 0.0,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
            "top_k": 0,
            "use_model_defaults": False,
            "eos_token_id": EOS,
            "trace": "none",
        },
    })

    handles: list[Any] = []
    old_alarm = signal.getsignal(signal.SIGALRM)
    started = time.monotonic()
    records: list[dict[str, Any]] = []
    counters = {"model_forwards": 0, "image_forwards": 0}
    error: BaseException | None = None
    torch_module: Any = None
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
                "native probe requires exactly one visible CUDA device")
        qwen, model_identity = load_policy(effective_config, device=torch.device("cuda:0"))
        terminal["model_loads"] = 1
        torch.cuda.reset_peak_memory_stats(torch.device("cuda:0"))
        require(model_identity["effective_settings"]["observed_attn_implementation"] == "sdpa",
                "live model attention is not SDPA")
        require(model_identity["effective_settings"]["observed_model_dtype"]["parameter_dtype_names"] ==
                ["torch.float32"], "live model parameters are not FP32")
        adapter_identity = model_identity["model_identity"]["adapter"]
        require(adapter_identity.get("adapter_path") == str(packet_plan["anchor_adapter"]),
                "live adapter is not packet anchor Stable50")
        require(adapter_identity.get("merged_adapters") == [],
                "live adapter unexpectedly merged")
        publish(out_dir / "model.json", {
            "schema": "small_owner_repeat_origin.model.v1",
            "rank": rank,
            "device": "cuda:0",
            "eos_token_id": EOS,
            "pad_token_id": qwen.tokenizer.pad_token_id,
            "identity": model_identity,
        })
        require(qwen.tokenizer.convert_tokens_to_ids("<|im_end|>") == EOS,
                "live tokenizer EOS identity differs")
        require(type(qwen.tokenizer.pad_token_id) is int and qwen.tokenizer.pad_token_id >= 0,
                "live tokenizer pad identity is invalid")
        qwen.model.eval()
        for parameter in qwen.model.parameters():
            parameter.requires_grad_(False)

        model = qwen.model
        visuals = [module for name, module in model.named_modules() if name.endswith("visual")]
        require(len(visuals) == 1, "ambiguous visual module for forward counters")

        def model_hook(*_: Any) -> None:
            counters["model_forwards"] += 1

        def image_hook(*_: Any) -> None:
            counters["image_forwards"] += 1

        handles.append(model.register_forward_pre_hook(model_hook))
        handles.append(visuals[0].register_forward_pre_hook(image_hook))

        # Build and prepare both image conditions once.  The prepared tensors
        # are passed unchanged to each fresh continuation call.
        raw_config = packet["config"]
        source_cases = [case["source_case"] for case in cases]
        donor_cases = [case["donor_case"] for case in cases]
        source_requests, _ = build_requests(qwen, raw_config, source_cases)
        donor_requests, _ = build_requests(qwen, raw_config, donor_cases)
        source_batch = prepare_native_inputs(qwen.processor, source_requests,
                                             device=torch.device("cuda:0"),
                                             record_media_identity=True)
        donor_batch = prepare_native_inputs(qwen.processor, donor_requests,
                                            device=torch.device("cuda:0"),
                                            record_media_identity=True)
        source_identity = _batch_identity(source_batch)
        donor_identity = _batch_identity(donor_batch)
        require(len(source_batch.request_ids) == len(cases) == len(donor_batch.request_ids),
                "prepared batch count differs from packet cases")
        for index, case in enumerate(cases):
            prompt = list(case["prompt_token_ids"])
            require(list(source_batch.prompt_token_ids[index]) == prompt and
                    list(donor_batch.prompt_token_ids[index]) == prompt,
                    f"{case['case_id']}: source/donor prompt token IDs differ")
            source_grid = source_batch.image_grids[index]
            donor_grid = donor_batch.image_grids[index]
            require(source_grid is not None and donor_grid is not None and
                    tuple(source_grid) == tuple(donor_grid),
                    f"{case['case_id']}: source/donor image_grid_thw differ")
            require(tuple(source_grid) == tuple(case["source_case"]["image_plan"]["observed_image_grid_thw"]),
                    f"{case['case_id']}: live source image grid differs from packet plan")
        publish(out_dir / "batches.json", {
            "schema": "small_owner_repeat_origin.batches.v1",
            "source": source_identity,
            "donor": donor_identity,
            "prompt_token_ids": {case["case_id"]: case["prompt_token_ids"] for case in cases},
            "grid_provenance": {
                case["case_id"]: {
                    "source": case["source_case"].get("image_plan", {}).get("grid_provenance"),
                    "donor": case["donor_case"].get("image_plan", {}).get("grid_provenance"),
                }
                for case in cases
            },
        })

        policy = NativeGenerationPolicy(temperature=0.0, top_p=1.0,
                                        repetition_penalty=1.0, top_k=0,
                                        use_model_defaults=False)
        with records_path.open("x", encoding="utf-8") as stream:
            # Natural is always first, so no contrast can be admitted if exact
            # Source replay fails.  The packet's other job order is irrelevant.
            ordered = sorted(
                selected,
                key=lambda pair: (
                    0
                    if (
                        pair[1]["boundary"],
                        pair[1]["image_condition"],
                        pair[1]["history_condition"],
                    )
                    == ("natural", "original", "native")
                    else 1
                    if pair[1]["boundary"] == "natural"
                    else 2,
                    pair[0]["case_id"],
                    pair[1]["job_id"],
                ),
            )
            for case, job in ordered:
                batch = select_batch(job["image_condition"], source_batch, donor_batch)
                expected_request_id = _expected_request_id(case, job)
                index = next(i for i, value in enumerate(cases)
                             if value["case_id"] == case["case_id"])
                require(batch.request_ids[index] == expected_request_id,
                        f"{case['case_id']}/{job['job_id']}: request association differs")
                before_model = counters["model_forwards"]
                before_image = counters["image_forwards"]
                tick = time.monotonic()
                with torch.inference_mode():
                    result = generate_continuations(
                        model,
                        # Rank ownership is one case, hence this prepared
                        # batch has exactly one row.  Passing it directly
                        # preserves every processor tensor unchanged and each
                        # call starts without a persisted KV cache.
                        batch,
                        extensions=[job["extension_ids"]],
                        budgets=[job["budget"]],
                        eos_token_id=EOS,
                        pad_token_id=qwen.tokenizer.pad_token_id,
                        policy=policy,
                        trace="none",
                    )[0]
                elapsed = time.monotonic() - tick
                require(result.request_id == expected_request_id,
                        f"{case['case_id']}/{job['job_id']}: native result request association differs")
                suffix = list(result.token_ids)
                action = job["extension_ids"] + suffix
                stop = result.stop_reason
                _validate_suffix(suffix, budget=job["budget"], stop=stop)
                text_value = qwen.tokenizer.decode(action, skip_special_tokens=False)
                parsed = native_record(text_value, {"row_id": case["source_row_id"]},
                                       case["golden"], stop)
                donor = job["image_condition"] == "donor"
                baseline_match = action == case["baseline_action_ids"] if (
                    job["boundary"], job["image_condition"], job["history_condition"]
                ) == ("natural", "original", "native") else None
                record = {
                    "schema": RECORD_SCHEMA,
                    "packet_sha256": packet_sha256,
                    "case_id": case["case_id"],
                    "source_row_id": case["source_row_id"],
                    "donor_row_id": case["donor_row_id"],
                    "job_id": job["job_id"],
                    "request_id": result.request_id,
                    "image_condition": job["image_condition"],
                    "history_condition": job["history_condition"],
                    "boundary": job["boundary"],
                    "extension_ids": list(job["extension_ids"]),
                    "prefix_ids": list(job["extension_ids"]),
                    "free_ids": suffix,
                    "action_ids": action,
                    "budget": job["budget"],
                    "text": text_value,
                    "stop": stop,
                    "stop_reason": stop,
                    "parsed": parsed,
                    "metrics_scope": (
                        "counterfactual_donor_image"
                        if donor
                        else "source_image"
                        if job["history_condition"] == "native"
                        else "forced_history_diagnostic"
                    ),
                    "donor_metrics_counterfactual": donor,
                    "owner_preservation_eligible": (
                        job["image_condition"] == "original"
                        and job["history_condition"] == "native"
                    ),
                    "baseline_match": baseline_match,
                    "baseline_discrepancy": (
                        None
                        if baseline_match is not False
                        else {
                            "expected_action_ids": case["baseline_action_ids"],
                            "observed_action_ids": action,
                        }
                    ),
                    "batch_identity_sha256": (
                        donor_identity["identity_sha256"] if donor else source_identity["identity_sha256"]
                    ),
                    "elapsed_seconds": elapsed,
                    "model_forwards": counters["model_forwards"] - before_model,
                    "image_forwards": counters["image_forwards"] - before_image,
                }
                validate_record(record, case=case, job=job, packet_sha256=packet_sha256,
                                require_baseline=False)
                _append_jsonl(records_path, record, stream)
                records.append(record)
                terminal["continuations"] += 1
                terminal["new_tokens"] += len(suffix)
                terminal["model_forwards"] = counters["model_forwards"]
                terminal["image_forwards"] = counters["image_forwards"]
                if baseline_match is False:
                    terminal["baseline_discrepancy"] = {
                        "case_id": case["case_id"],
                        "job_id": job["job_id"],
                        "expected_action_ids": case["baseline_action_ids"],
                        "observed_action_ids": action,
                    }
                    # The record is durable before the fail-closed decision.
                    raise ValueError("original natural result differs from frozen baseline")

        durable = read_jsonl(records_path)
        readback = validate_readback(durable, cases=cases, packet_sha256=packet_sha256,
                                     job_id=job_id, smoke=smoke, require_baseline=True)
        publish(out_dir / "readback.json", {
            "schema": "small_owner_repeat_origin.readback.v1",
            **readback,
        })
        terminal["readback"] = readback
        terminal["status"] = "completed"
        terminal["exit_code"] = 0
    except BaseException as exc:
        error = exc
        terminal["status"] = "failed"
        terminal["exit_code"] = 1
        terminal["error"] = repr(exc)
        terminal["traceback"] = traceback.format_exc()
        # Preserve a bounded readback diagnostic for partial or baseline-
        # rejected runs without replacing the original failure.
        if records_path.is_file():
            try:
                durable = read_jsonl(records_path)
                terminal["readback_observed_cells"] = len(durable)
                terminal["readback_error"] = None
                validate_readback(durable, cases=cases, packet_sha256=packet_sha256,
                                 job_id=job_id, smoke=smoke, require_baseline=False)
                terminal["readback_complete"] = True
            except BaseException as readback_exc:
                terminal["readback_error"] = repr(readback_exc)
                terminal["readback_complete"] = False
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)
        for handle in handles:
            try:
                handle.remove()
            except Exception:
                pass
        cuda_sync_error = None
        peak_cuda_allocated = None
        peak_cuda_reserved = None
        if torch_module is not None and torch_module.cuda.is_initialized():
            try:
                torch_module.cuda.synchronize(torch_module.device("cuda:0"))
            except BaseException as sync_exc:
                cuda_sync_error = repr(sync_exc)
            try:
                peak_cuda_allocated = int(torch_module.cuda.max_memory_allocated(
                    torch_module.device("cuda:0")))
                peak_cuda_reserved = int(torch_module.cuda.max_memory_reserved(
                    torch_module.device("cuda:0")))
            except BaseException as peak_exc:
                terminal["peak_cuda_error"] = repr(peak_exc)
        bound_errors = []
        if counters["model_forwards"] > selected_budget_total + selected_job_count:
            bound_errors.append("model_forwards_exceed_budget_plus_job_tolerance")
        if counters["image_forwards"] > selected_job_count:
            bound_errors.append("image_forwards_exceed_selected_job_count")
        if terminal["new_tokens"] > selected_budget_total:
            bound_errors.append("new_tokens_exceed_selected_budget")
        if terminal["continuations"] > selected_job_count:
            bound_errors.append("continuations_exceed_selected_job_count")
        if bound_errors:
            terminal["counter_bound_error"] = bound_errors
            if error is None:
                error = ValueError("probe counter bound exceeded: " + ", ".join(bound_errors))
                terminal["status"] = "failed"
                terminal["exit_code"] = 1
                terminal["error"] = repr(error)
        terminal.update({
            "elapsed_seconds": time.monotonic() - started,
            "host_peak_rss_bytes": _rss_peak_bytes(),
            "finished_unix": time.time(),
            "model_forwards": counters["model_forwards"],
            "image_forwards": counters["image_forwards"],
            "peak_cuda_allocated_bytes": peak_cuda_allocated,
            "peak_cuda_reserved_bytes": peak_cuda_reserved,
            "cuda_synchronize_error": cuda_sync_error,
            "counter_bounds": {
                "model_forwards_max": selected_budget_total + selected_job_count,
                "image_forwards_max": selected_job_count,
                "new_tokens_max": selected_budget_total,
                "continuations_max": selected_job_count,
            },
        })
        publish(out_dir / "terminal.json", terminal)
    if error is not None:
        raise error
    return terminal


def _validate_suffix(suffix: Sequence[int], *, budget: int, stop: str) -> None:
    values = _token_ids(list(suffix), field="native suffix")
    require(len(values) <= budget, "native suffix exceeds requested budget")
    require(stop in ("im_end", "length"), "native continuation stop is invalid")
    if stop == "im_end":
        require(bool(values) and values[-1] == EOS, "native EOS stop lacks terminal EOS")
        require(EOS not in values[:-1] and PAD not in values, "native suffix has EOS/pad corruption")
    else:
        require(len(values) == budget, "native length stop did not consume budget")
        require(EOS not in values and PAD not in values, "native length suffix has EOS/pad corruption")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True, type=Path, help="absolute packet.json")
    parser.add_argument("--out-dir", required=True, type=Path, help="new immutable output directory")
    parser.add_argument("--rank", required=True, type=int, choices=range(8))
    parser.add_argument("--job-id", default=None,
                        help="optional single original/native/natural smoke job")
    parser.add_argument("--smoke", action="store_true",
                        help="run the assigned case's natural plus four early factorial jobs")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        run_probe(args.packet, args.out_dir, rank=args.rank, job_id=args.job_id,
                  smoke=args.smoke)
    except BaseException as exc:
        print(f"small-owner-repeat-origin: {exc}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI only
    raise SystemExit(main())
