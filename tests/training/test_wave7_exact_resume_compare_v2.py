from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_wave7_exact_resume_compare as v1_test  # noqa: E402

from scripts.probes.coordexp_swift import (  # noqa: E402
    wave7_exact_resume_compare_v2 as compare,
)
from src.artifacts.checkpoint_payload import (  # noqa: E402
    build_inference_checkpoint_payload_identity,
    write_inference_checkpoint_payload_manifest,
)
from src.artifacts.training_state import load_training_state_manifest  # noqa: E402


SCRIPT = (
    v1_test.REPO_ROOT / "scripts/probes/coordexp_swift/wave7_exact_resume_compare_v2.py"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: Any) -> None:
    v1_test._json(path, value)


def _upgrade_checkpoint(checkpoint_dir: Path) -> dict[str, Any]:
    adapter_config = checkpoint_dir / "adapter/adapter_config.json"
    _json(
        adapter_config,
        {
            "base_model_name_or_path": None,
            "lora_alpha": 4,
            "peft_type": "LORA",
            "r": 2,
            "target_modules": ["q_proj"],
            "use_dora": True,
        },
    )
    write_inference_checkpoint_payload_manifest(checkpoint_dir)
    return build_inference_checkpoint_payload_identity(checkpoint_dir)


def _upgrade_logging(
    run_dir: Path,
    *,
    eval_duration_seconds: float,
) -> None:
    path = run_dir / "logging.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if "acc_top5" not in row:
            row["acc_top5"] = 0.90
        atom_count = 10_000
        row["accuracy_stats"] = {
            "top1_correct": round(float(row["acc_top1"]) * atom_count),
            "top5_correct": round(float(row["acc_top5"]) * atom_count),
            "atom_count": atom_count,
        }
        if row["split"] == "eval":
            row["eval_duration_seconds"] = eval_duration_seconds
    path.write_text(
        "".join(
            json.dumps(row, allow_nan=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _upgrade_run_events(run_dir: Path) -> None:
    path = run_dir / "run.json"
    run = json.loads(path.read_text(encoding="utf-8"))
    events = run["measurement"]["checkpoint_publication_events"]
    for event in events:
        step = int(event["step"])
        checkpoint_dir = run_dir / f"checkpoints/step-{step}"
        event.update(
            schema=compare.EVENT_SCHEMA,
            schema_version=compare.EVENT_SCHEMA_VERSION,
            inference_payload_identity=build_inference_checkpoint_payload_identity(
                checkpoint_dir
            ),
            committed_progress={
                "schema": compare.PROGRESS_SCHEMA,
                "schema_version": compare.PROGRESS_SCHEMA_VERSION,
                "completed_steps": step,
                "consumed_packs": step * 2,
                "optimizer_update_status": "applied",
                "finite_status": "finite",
            },
        )
    latest = events[-1]["committed_progress"]
    run["completed_steps"] = latest["completed_steps"]
    run["consumed_packs"] = latest["consumed_packs"]
    run["checkpoint_event_count"] = len(events)
    run["final_optimizer_update_status"] = latest["optimizer_update_status"]
    run["final_finite_status"] = latest["finite_status"]
    _json(path, run)


def _bind_v2_source(marker: Path, receipt: Path) -> None:
    marker_value = json.loads(marker.read_text(encoding="utf-8"))
    receipt_value = json.loads(receipt.read_text(encoding="utf-8"))
    identity = {
        "path": str(SCRIPT),
        "sha256": _sha256(SCRIPT),
        "size": SCRIPT.stat().st_size,
    }
    for payload in (marker_value, receipt_value):
        payload["source_hashes"] = [
            row for row in payload["source_hashes"] if row["path"] != str(SCRIPT)
        ] + [identity]
    _json(marker, marker_value)
    marker_hash = _sha256(marker)
    for field in ("expected_file_sha256", "file_sha256", "final_file_sha256"):
        receipt_value["marker"][field] = marker_hash
    _json(receipt, receipt_value)


def _rebuild_interrupt_evidence(paths: dict[str, Path]) -> None:
    manifest = load_training_state_manifest(paths["parent"] / "checkpoints/step-3")
    marker, receipt = v1_test._write_interrupt_evidence(
        paths["parent"].parent, paths["parent"], manifest
    )
    assert marker == paths["marker"]
    assert receipt == paths["receipt"]
    _bind_v2_source(marker, receipt)


def _mutate_accuracy(
    run_dir: Path,
    *,
    split: str,
    step: int,
    delta: int,
) -> None:
    path = run_dir / "logging.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if row["split"] == split and row["step"] == step:
            row["accuracy_stats"]["top1_correct"] += delta
            row["acc_top1"] = (
                row["accuracy_stats"]["top1_correct"]
                / row["accuracy_stats"]["atom_count"]
            )
    path.write_text(
        "".join(
            json.dumps(row, allow_nan=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _fixture(root: Path, *, mutation: str | None = None) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths = v1_test._fixture(root)
    for run_dir in (paths["reference"], paths["parent"], paths["child"]):
        for checkpoint_dir in sorted((run_dir / "checkpoints").glob("step-*")):
            _upgrade_checkpoint(checkpoint_dir)
    _upgrade_logging(paths["reference"], eval_duration_seconds=1.0)
    _upgrade_logging(paths["parent"], eval_duration_seconds=2.0)
    _upgrade_logging(paths["child"], eval_duration_seconds=3.0)
    if mutation == "parent_accuracy_numerator":
        _mutate_accuracy(paths["parent"], split="train", step=3, delta=1)
    elif mutation == "child_accuracy_numerator":
        _mutate_accuracy(paths["child"], split="train", step=4, delta=1)
    elif mutation == "best_value_and_numerator":
        _mutate_accuracy(paths["parent"], split="eval", step=3, delta=1)
        alias = json.loads(
            (paths["parent"] / "checkpoints/best.json").read_text(encoding="utf-8")
        )
        alias["value"] = 0.7501
        _json(paths["parent"] / "checkpoints/best.json", alias)
    elif mutation in {
        "best_missing_accuracy_stats",
        "best_value_tolerance_laundering",
    }:
        log_path = paths["parent"] / "logging.jsonl"
        rows = [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
        ]
        for row in rows:
            if row["split"] == "eval" and row["step"] == 3:
                if mutation == "best_missing_accuracy_stats":
                    row.pop("accuracy_stats")
                else:
                    row["acc_top1"] = 0.75000001
        log_path.write_text(
            "".join(
                json.dumps(row, allow_nan=False, sort_keys=True, separators=(",", ":"))
                + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )
        if mutation == "best_value_tolerance_laundering":
            alias_path = paths["parent"] / "checkpoints/best.json"
            alias = json.loads(alias_path.read_text(encoding="utf-8"))
            alias["value"] = 0.75000001
            _json(alias_path, alias)
    elif mutation == "best_wrong_selector":
        alias_path = paths["parent"] / "checkpoints/best.json"
        alias = json.loads(alias_path.read_text(encoding="utf-8"))
        alias["selector"] = "loss/total"
        alias["value"] = 0.5
        _json(alias_path, alias)
    elif mutation == "best_missing_value":
        alias_path = paths["parent"] / "checkpoints/best.json"
        alias = json.loads(alias_path.read_text(encoding="utf-8"))
        alias.pop("value")
        _json(alias_path, alias)
    elif mutation == "best_nonfinite_value":
        alias_path = paths["parent"] / "checkpoints/best.json"
        alias_path.write_text(
            '{"checkpoint_path":"checkpoints/step-3","selector":"acc_top1",'
            '"step":3,"value":NaN}\n',
            encoding="utf-8",
        )
    for run_dir in (paths["reference"], paths["parent"], paths["child"]):
        _upgrade_run_events(run_dir)
    if mutation == "committed_progress_mismatch":
        path = paths["parent"] / "run.json"
        run = json.loads(path.read_text(encoding="utf-8"))
        run["measurement"]["checkpoint_publication_events"][0]["committed_progress"][
            "consumed_packs"
        ] += 1
        _json(path, run)
    elif mutation == "committed_progress_pair_mismatch":
        path = paths["parent"] / "run.json"
        run = json.loads(path.read_text(encoding="utf-8"))
        progress = run["measurement"]["checkpoint_publication_events"][0][
            "committed_progress"
        ]
        progress["consumed_packs"] += 1
        run["consumed_packs"] = progress["consumed_packs"]
        _json(path, run)
    elif mutation == "inference_identity_mismatch":
        path = paths["parent"] / "run.json"
        run = json.loads(path.read_text(encoding="utf-8"))
        run["measurement"]["checkpoint_publication_events"][0][
            "inference_payload_identity"
        ]["aggregate_digest"] = "f" * 64
        _json(path, run)
    _rebuild_interrupt_evidence(paths)
    if mutation == "rank_state_corrupt":
        rank = (
            paths["parent"]
            / "checkpoints/step-3/training_state/rank-00000/rng-python.bin"
        )
        rank.write_bytes(rank.read_bytes() + b"corrupt")
    paths["output"] = root / "comparison-v2.json"
    return paths


def _legacy_fixture(root: Path) -> dict[str, Path]:
    paths = v1_test._fixture(root)
    run_path = paths["parent"] / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["completed_steps"] = 0
    run["consumed_packs"] = 0
    run["checkpoint_event_count"] = 0
    _json(run_path, run)
    log_path = paths["parent"] / "logging.jsonl"
    rows = [
        json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        if row["split"] == "eval":
            row["eval_duration_seconds"] = 2.0
    log_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    _rebuild_interrupt_evidence(paths)
    paths["output"] = root / "legacy-r4-v2-diagnostic.json"
    return paths


def _common_args(paths: dict[str, Path], output: Path) -> list[str]:
    return [
        "--uninterrupted-run-dir",
        str(paths["reference"]),
        "--interrupted-parent-run-dir",
        str(paths["parent"]),
        "--interruption-marker",
        str(paths["marker"]),
        "--termination-receipt",
        str(paths["receipt"]),
        "--output",
        str(output),
        "--expected-source-sha256",
        _sha256(SCRIPT),
        "--expected-interrupt-source-sha256",
        v1_test.INTERRUPT_SOURCE_SHA256,
        "--expected-provenance-sha256",
        v1_test.PROVENANCE_SHA256,
    ]


def _run_final(paths: dict[str, Path]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "compare",
            *_common_args(paths, paths["output"]),
            "--resume-child-run-dir",
            str(paths["child"]),
        ],
        cwd=v1_test.REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _run_pre_child(
    paths: dict[str, Path],
) -> tuple[subprocess.CompletedProcess[str], Path]:
    output = paths["output"].with_name("pre-child.json")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "pre-child", *_common_args(paths, output)],
        cwd=v1_test.REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result, output


def test_v1_source_and_tests_remain_byte_frozen() -> None:
    assert _sha256(v1_test.SCRIPT) == compare.V1_SOURCE_SHA256
    assert _sha256(Path(v1_test.__file__)) == (
        "2dc19c32fedf0e4b7e93010a6f1f7b027bcebe606f17116de799e46143d2d5ca"
    )


def test_r5_final_accepts_timing_drift_with_authoritative_exact_stats(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)

    result = _run_final(paths)

    assert result.returncode == 0, result.stderr
    receipt = json.loads(paths["output"].read_text(encoding="utf-8"))
    assert receipt["schema"] == compare.FINAL_RECEIPT_SCHEMA
    assert receipt["status"] == "passed"
    assert receipt["mismatches"] == []
    assert receipt["checkpoint_selection"]["status"] == "passed"
    assert receipt["checkpoint_selection"]["identity_fields"] == [
        "checkpoint_path",
        "selector",
        "step",
        "value",
        "eligibility",
    ]
    assert all(
        row["authoritative"] and row["matched"]
        for row in receipt["event_progress_comparisons"]
    )
    assert all(
        row["authoritative"]
        for row in receipt["log_comparison"]["accuracy_sufficient_statistics"]
    )
    assert (
        "all declared *_duration_seconds fields"
        in receipt["log_comparison"]["excluded_from_semantic_equality"]
    )


def test_selection_value_and_integer_stats_drift_fail_exactly(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path, mutation="best_value_and_numerator")

    result = _run_final(paths)

    assert result.returncode == 1
    receipt = json.loads(paths["output"].read_text(encoding="utf-8"))
    assert receipt["checkpoint_selection"]["status"] == "failed"
    codes = {row["code"] for row in receipt["mismatches"]}
    assert "wave7_compare_v2.accuracy_sufficient_statistics" in codes
    assert "wave7_compare_v2.checkpoint_selection" in codes


@pytest.mark.parametrize(
    "mutation",
    [
        "best_value_tolerance_laundering",
        "best_missing_accuracy_stats",
        "best_wrong_selector",
    ],
)
def test_best_selection_requires_exact_authoritative_integer_metric(
    tmp_path: Path, mutation: str
) -> None:
    paths = _fixture(tmp_path, mutation=mutation)

    result = _run_final(paths)

    assert result.returncode == 1
    receipt = json.loads(paths["output"].read_text(encoding="utf-8"))
    assert receipt["checkpoint_selection"]["status"] == "failed"
    parent = receipt["checkpoint_selection"]["eligibility"]["interrupted_parent"]
    assert parent["eligible"] is False
    assert parent["metric_matches_alias"] is False
    assert "wave7_compare_v2.checkpoint_selection" in {
        row["code"] for row in receipt["mismatches"]
    }


@pytest.mark.parametrize(
    "mutation",
    ["best_missing_value", "best_nonfinite_value"],
)
def test_best_selection_rejects_missing_or_nonfinite_value(
    tmp_path: Path, mutation: str
) -> None:
    paths = _fixture(tmp_path, mutation=mutation)

    result = _run_final(paths)

    assert result.returncode == 1
    receipt = json.loads(paths["output"].read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert receipt["mismatches"]


@pytest.mark.parametrize(
    "mutation",
    ["parent_accuracy_numerator", "child_accuracy_numerator"],
)
def test_final_exact_accuracy_integer_mutation_fails(
    tmp_path: Path, mutation: str
) -> None:
    paths = _fixture(tmp_path, mutation=mutation)

    result = _run_final(paths)

    assert result.returncode == 1
    receipt = json.loads(paths["output"].read_text(encoding="utf-8"))
    assert "wave7_compare_v2.accuracy_sufficient_statistics" in {
        row["code"] for row in receipt["mismatches"]
    }


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ("committed_progress_mismatch", "wave7_compare_v2.committed_progress"),
        ("inference_identity_mismatch", "wave7_compare_v2.inference_payload_identity"),
    ],
)
def test_r5_publication_identity_mutations_fail(
    tmp_path: Path, mutation: str, code: str
) -> None:
    paths = _fixture(tmp_path, mutation=mutation)

    result = _run_final(paths)

    assert result.returncode == 1
    receipt = json.loads(paths["output"].read_text(encoding="utf-8"))
    assert code in {row["code"] for row in receipt["mismatches"]}


def test_r5_cross_run_committed_progress_must_match_exactly(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, mutation="committed_progress_pair_mismatch")

    result = _run_final(paths)

    assert result.returncode == 1
    receipt = json.loads(paths["output"].read_text(encoding="utf-8"))
    assert "wave7_compare_v2.committed_progress_pair" in {
        row["code"] for row in receipt["mismatches"]
    }


def test_legacy_r4_stale_progress_is_diagnostic_not_relabelled_pass(
    tmp_path: Path,
) -> None:
    paths = _legacy_fixture(tmp_path)

    result = _run_final(paths)

    assert result.returncode == 1
    receipt = json.loads(paths["output"].read_text(encoding="utf-8"))
    codes = {row["code"] for row in receipt["mismatches"]}
    assert "wave7_compare.run_status" not in codes
    assert "wave7_compare.run_count_mismatch" not in codes
    assert "wave7_compare.logging_value" not in codes
    assert "wave7_compare_v2.legacy_event_not_r5_evidence" in codes
    assert "wave7_compare_v2.legacy_accuracy_not_r5_evidence" in codes
    progress = receipt["run_contract"]["interrupted_parent_progress"]
    assert progress["observed_top_level"]["completed_steps"] == 0
    assert progress["effective"]["completed_steps"] == 3


def test_pre_child_receipt_is_absent_only_and_digest_verifiable(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)

    first, output = _run_pre_child(paths)

    assert first.returncode == 0, first.stderr
    before = output.read_bytes()
    receipt = json.loads(before)
    assert receipt["schema"] == compare.PRE_CHILD_RECEIPT_SCHEMA
    assert receipt["child_launch_authorized"] is True
    assert receipt["event_progress_comparisons"] == [
        {
            "authoritative": True,
            "left": {
                "completed_steps": 3,
                "consumed_packs": 6,
                "finite_status": "finite",
                "optimizer_update_status": "applied",
                "schema": compare.PROGRESS_SCHEMA,
                "schema_version": compare.PROGRESS_SCHEMA_VERSION,
            },
            "left_role": "uninterrupted",
            "matched": True,
            "right": {
                "completed_steps": 3,
                "consumed_packs": 6,
                "finite_status": "finite",
                "optimizer_update_status": "applied",
                "schema": compare.PROGRESS_SCHEMA,
                "schema_version": compare.PROGRESS_SCHEMA_VERSION,
            },
            "right_role": "interrupted_parent",
            "step": 3,
        }
    ]
    verify = compare.main(
        [
            "verify-pre-child",
            "--receipt",
            str(output),
            "--expected-payload-sha256",
            receipt["receipt_payload_sha256"],
        ]
    )
    assert verify == 0
    assert (
        compare.main(
            [
                "verify-pre-child",
                "--receipt",
                str(output),
                "--expected-payload-sha256",
                "f" * 64,
            ]
        )
        == 1
    )
    second, _ = _run_pre_child(paths)
    assert second.returncode == 2
    assert output.read_bytes() == before


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        (
            "parent_accuracy_numerator",
            "wave7_compare_v2.accuracy_sufficient_statistics",
        ),
        ("rank_state_corrupt", None),
    ],
)
def test_pre_child_rejects_exact_accuracy_and_rank_state_corruption(
    tmp_path: Path,
    mutation: str,
    expected_code: str | None,
) -> None:
    paths = _fixture(tmp_path, mutation=mutation)

    result, output = _run_pre_child(paths)

    assert result.returncode == 1
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["child_launch_authorized"] is False
    assert receipt["mismatches"]
    if expected_code is not None:
        assert expected_code in {row["code"] for row in receipt["mismatches"]}
