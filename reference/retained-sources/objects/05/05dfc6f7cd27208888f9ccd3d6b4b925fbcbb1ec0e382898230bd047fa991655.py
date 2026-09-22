"""CPU contract tests for the small-owner-repeat-origin runner.

These tests exercise packet and durable-cell invariants only.  They do not
instantiate a model, call CUDA, or assert a model-quality outcome.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


RUNNER_PATH = Path(__file__).with_name("run_probe.py")
spec = importlib.util.spec_from_file_location("small_owner_repeat_origin_runner", RUNNER_PATH)
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jobs() -> list[dict]:
    result = [{
        "job_id": "natural-original-native",
        "image_condition": "original",
        "history_condition": "native",
        "boundary": "natural",
        "extension_ids": [],
        "budget": runner.CAP,
    }]
    for boundary in ("early", "late"):
        for image in ("original", "donor"):
            for history in ("native", "translated"):
                result.append({
                    "job_id": f"{boundary}-{image}-{history}",
                    "image_condition": image,
                    "history_condition": history,
                    "boundary": boundary,
                    "extension_ids": [101, 102] if boundary == "early" else [201, 202, 203],
                    "budget": runner.CONDITIONAL_BUDGET,
                })
    return result


@pytest.fixture()
def packet(tmp_path: Path) -> dict:
    source_image = tmp_path / "source.jpg"
    donor_image = tmp_path / "donor.jpg"
    source_image.write_bytes(b"source-image-bytes")
    donor_image.write_bytes(b"donor-image-bytes")
    adapter = tmp_path / "stable50"
    adapter.mkdir()
    adapter_file = adapter / "adapter_config.json"
    adapter_file.write_text("{}", encoding="utf-8")

    def case(row_id: str, image: Path, image_hash: str) -> dict:
        return {
            "row_id": row_id,
            "image_path": str(image),
            "image_width": 640,
            "image_height": 480,
            "input_record": {},
            "row_index": 0,
            "image_plan": {
                "image_content_sha256": image_hash,
                "observed_image_grid_thw": [1, 2, 2],
                "logical_transform_id": "identity",
                "merged_visual_tokens": 4,
                "backend_prompt_token_count": 3,
            },
        }

    source = case("source-row", source_image, _sha(source_image))
    donor = case("donor-row", donor_image, _sha(donor_image))
    return {
        "schema": runner.SCHEMA,
        "config": {},
        "anchor_adapter": str(adapter),
        "source_files": {
            str(source_image): _sha(source_image),
            str(donor_image): _sha(donor_image),
            str(adapter_file): _sha(adapter_file),
        },
        "cases": [{
            "case_id": "case-0",
            "source_case": source,
            "donor_case": donor,
            "prompt_token_ids": [11, 12, 13],
            "baseline_action_ids": [runner.EOS],
            "golden": {},
            "jobs": _jobs(),
        }],
    }


def _normalized(packet: dict) -> dict:
    # The fixture has real image files and hashes, so this is the same
    # pre-model validation path used by the CLI.
    return runner.validate_packet(packet, verify_sources=True)


def _record(case: dict, job: dict, packet_sha: str, *, suffix: list[int] | None = None) -> dict:
    suffix = [runner.EOS] if suffix is None else suffix
    action = list(job["extension_ids"]) + suffix
    donor = job["image_condition"] == "donor"
    return {
        "schema": runner.RECORD_SCHEMA,
        "packet_sha256": packet_sha,
        "case_id": case["case_id"],
        "source_row_id": case["source_row_id"],
        "donor_row_id": case["donor_row_id"],
        "job_id": job["job_id"],
        "request_id": case["donor_row_id"] if donor else case["source_row_id"],
        "image_condition": job["image_condition"],
        "history_condition": job["history_condition"],
        "boundary": job["boundary"],
        "extension_ids": list(job["extension_ids"]),
        "prefix_ids": list(job["extension_ids"]),
        "free_ids": suffix,
        "action_ids": action,
        "budget": job["budget"],
        "text": "synthetic",
        "stop": "im_end",
        "stop_reason": "im_end",
        "parsed": {},
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
        "baseline_match": (
            action == case["baseline_action_ids"]
            if (job["boundary"], job["image_condition"], job["history_condition"])
            == ("natural", "original", "native")
            else None
        ),
    }


def test_packet_identity_prompt_and_dimensions_are_frozen(packet: dict):
    plan = _normalized(packet)
    case = plan["cases"][0]
    assert case["source_row_id"] != case["donor_row_id"]
    assert case["prompt_token_ids"] == [11, 12, 13]
    assert {tuple(job[k] for k in ("boundary", "image_condition", "history_condition"))
            for job in case["jobs"]} == runner._expected_job_tuples()

    changed = copy.deepcopy(packet)
    changed["cases"][0]["donor_case"]["image_width"] += 1
    with pytest.raises(ValueError, match="dimensions differ"):
        runner.validate_packet(changed, verify_sources=False)

    changed = copy.deepcopy(packet)
    changed["cases"][0]["donor_case"]["prompt_token_ids"] = [99]
    with pytest.raises(ValueError, match="prompt_token_ids differs"):
        runner.validate_packet(changed, verify_sources=False)


def test_extension_free_partition_and_minimum_remaining_budget(packet: dict):
    plan = _normalized(packet)
    case = plan["cases"][0]
    job = next(job for job in case["jobs"] if job["boundary"] == "early")
    packet_sha = "a" * 64
    record = _record(case, job, packet_sha)
    runner.validate_record(record, case=case, job=job, packet_sha256=packet_sha)

    shifted = copy.deepcopy(record)
    shifted["prefix_ids"] = shifted["prefix_ids"] + [999]
    with pytest.raises(ValueError, match="shifted prefix"):
        runner.validate_record(shifted, case=case, job=job, packet_sha256=packet_sha)

    long_extension = [7] * (runner.CAP - 3)
    long_job = runner.validate_job({
        "job_id": "late-long",
        "image_condition": "original",
        "history_condition": "translated",
        "boundary": "late",
        "extension_ids": long_extension,
        "budget": 3,
    }, case_id="case-0")
    assert long_job["budget"] == 3
    long_record = _record(case, long_job, packet_sha, suffix=[1, 2, 3])
    long_record["stop"] = long_record["stop_reason"] = "length"
    runner.validate_record(long_record, case=case, job=long_job, packet_sha256=packet_sha)

    too_long = copy.deepcopy(long_record)
    too_long["free_ids"] = [1, 2, 3, 4]
    too_long["action_ids"] = long_extension + too_long["free_ids"]
    with pytest.raises(ValueError, match="exceeds job budget"):
        runner.validate_record(too_long, case=case, job=long_job, packet_sha256=packet_sha)


def test_image_condition_selects_prepared_batch_without_replacement():
    source = object()
    donor = object()
    assert runner.select_batch("original", source, donor) is source
    assert runner.select_batch("donor", source, donor) is donor
    with pytest.raises(ValueError, match="unknown image condition"):
        runner.select_batch("translated", source, donor)


def test_translated_source_record_cannot_claim_native_owner_preservation(packet: dict):
    plan = _normalized(packet)
    case = plan["cases"][0]
    job = next(
        job
        for job in case["jobs"]
        if job["boundary"] == "early"
        and job["image_condition"] == "original"
        and job["history_condition"] == "translated"
    )
    packet_sha = "c" * 64
    record = _record(case, job, packet_sha)
    record["parsed"] = {"raw_parser_output": "preserved"}
    assert record["owner_preservation_eligible"] is False
    runner.validate_record(record, case=case, job=job, packet_sha256=packet_sha)

    old_claim = copy.deepcopy(record)
    old_claim["owner_preservation_eligible"] = True
    with pytest.raises(ValueError, match="owner-preservation eligibility"):
        runner.validate_record(old_claim, case=case, job=job, packet_sha256=packet_sha)
    assert old_claim["parsed"] == {"raw_parser_output": "preserved"}


def test_readback_rejects_partial_duplicate_and_extra_cells(packet: dict):
    plan = _normalized(packet)
    cases = plan["cases"]
    case = cases[0]
    packet_sha = "b" * 64
    records = [_record(case, job, packet_sha) for job in case["jobs"]]
    assert runner.validate_readback(records, cases=cases, packet_sha256=packet_sha) == {
        "expected_cells": 9,
        "observed_cells": 9,
        "job_ids": sorted(f"case-0:{job['job_id']}" for job in case["jobs"]),
    }

    with pytest.raises(ValueError, match="missing result cells"):
        runner.validate_readback(records[:-1], cases=cases, packet_sha256=packet_sha)

    with pytest.raises(ValueError, match="duplicate result cell"):
        runner.validate_readback(records + [records[0]], cases=cases, packet_sha256=packet_sha)

    extra = copy.deepcopy(records[0])
    extra["job_id"] = "not-in-packet"
    with pytest.raises(ValueError, match="unexpected result cell"):
        runner.validate_readback(records[:-1] + [extra], cases=cases, packet_sha256=packet_sha)


def test_only_authorized_donor_natural_control_may_add_a_tenth_cell(packet: dict):
    packet_with_control = copy.deepcopy(packet)
    packet_with_control["cases"][0]["jobs"].append({
        "job_id": "donor_natural",
        "image_condition": "donor",
        "history_condition": "native",
        "boundary": "natural",
        "extension_ids": [],
        "budget": runner.CAP,
    })
    plan = _normalized(packet_with_control)
    case = plan["cases"][0]
    packet_sha = "d" * 64
    records = [_record(case, job, packet_sha) for job in case["jobs"]]
    readback = runner.validate_readback(records, cases=plan["cases"], packet_sha256=packet_sha)
    assert readback["expected_cells"] == readback["observed_cells"] == 10
    assert runner.expected_job_ids([case], smoke=True) == {
        ("case-0", "natural-original-native"),
        *(('case-0', job_id) for job_id in (
            "early-original-native",
            "early-original-translated",
            "early-donor-native",
            "early-donor-translated",
        )),
    }

    unexpected = copy.deepcopy(packet)
    unexpected["cases"][0]["jobs"].append({
        "job_id": "donor_late_extra",
        "image_condition": "donor",
        "history_condition": "native",
        "boundary": "natural",
        "extension_ids": [],
        "budget": runner.CAP,
    })
    with pytest.raises(ValueError, match="donor natural control|at most one donor_natural|factorial job cells"):
        runner.validate_packet(unexpected, verify_sources=False)


def test_smoke_expected_cells_are_natural_plus_four_early_cells(packet: dict):
    plan = _normalized(packet)
    case = plan["cases"]
    expected = runner.expected_job_ids(case, smoke=True)
    assert len(expected) == 5
    assert sum(1 for _, job in expected if job.startswith("early-")) == 4
    assert runner.expected_job_ids(case, job_id="natural-original-native") == {
        ("case-0", "natural-original-native")
    }
    with pytest.raises(ValueError, match="mutually exclusive"):
        runner.expected_job_ids(case, job_id="natural-original-native", smoke=True)
