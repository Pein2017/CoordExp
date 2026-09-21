from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path


REPO = Path("/data/CoordExp")
ROOT = REPO / "outputs/coco_refinement/probes/wave2-bounded-matrix-20260717-063044"
FULL_RECEIPT = REPO / "outputs/coco_refinement/probes/wave2-full-commit-20260717-060642/probe-receipt.json"
MULTITASK = Path("/tmp/test_task26_multitask_probe.py")
INVALID = Path("/tmp/test_task26_invalid_probe.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def extract_json_line(output: str, prefix: str) -> dict | None:
    for line in output.splitlines():
        if line.startswith(prefix + " "):
            return json.loads(line[len(prefix) + 1 :])
    return None


def run(name: str, argv: list[str], log_name: str) -> dict:
    start = time.perf_counter()
    process = subprocess.run(
        argv,
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONPATH": "/data/CoordExp/tests/coco_refinement:/data/CoordExp"},
    )
    elapsed = time.perf_counter() - start
    log_path = ROOT / "logs" / log_name
    log_path.write_text(process.stdout, encoding="utf-8")
    matches = re.findall(r"(\d+) passed", process.stdout)
    passed = int(matches[-1]) if matches else 0
    print(json.dumps({"name": name, "exit": process.returncode, "passed": passed, "seconds": elapsed, "log": str(log_path)}, sort_keys=True), flush=True)
    return {
        "name": name,
        "argv": argv,
        "command": " ".join(argv),
        "exit_code": process.returncode,
        "seconds": elapsed,
        "passed": passed,
        "log_path": str(log_path),
        "log_sha256": sha256(log_path),
        "output": process.stdout,
    }


def main() -> None:
    if ROOT.exists():
        raise RuntimeError(f"artifact root exists: {ROOT}")
    (ROOT / "logs").mkdir(parents=True)
    (ROOT / "scripts").mkdir()
    shutil.copy2(MULTITASK, ROOT / "scripts" / MULTITASK.name)
    shutil.copy2(INVALID, ROOT / "scripts" / INVALID.name)
    shutil.copy2(Path(__file__), ROOT / "scripts" / Path(__file__).name)

    native = [
        "conda", "run", "-n", "ms", "pytest", "-vv",
        "tests/coco_refinement/test_commit_api.py::test_paused_worker_captures_all_drafts_and_lost_retry_never_recaptures",
        "tests/coco_refinement/test_commit_api.py::test_real_paused_publication_keeps_http_editable_and_reconciles_newer_draft",
        "tests/coco_refinement/test_commit_api.py::test_failed_terminal_observer_keeps_http_editable_until_restart_repair",
        "tests/coco_refinement/test_commit_api.py::test_post_sqlite_pre_queue_terminal_keeps_http_editable_and_identity_stable",
        "tests/coco_refinement/test_adapters.py::test_terminal_success_retires_exact_captured_draft_idempotently_across_restart",
        "tests/coco_refinement/test_adapters.py::test_terminal_success_merges_only_identity_into_newer_draft_without_resurrection",
        "tests/coco_refinement/test_adapters.py::test_later_unrelated_draft_rebinds_to_terminal_project_generation_and_verifies",
        "tests/coco_refinement/test_runtime.py::test_startup_replays_published_terminal_before_strict_sqlite_attestation",
    ]
    store_runtime = [
        "conda", "run", "-n", "ms", "pytest", "-vv",
        "tests/label_studio_coco_refinement/test_store.py::test_stale_batch_member_fails_all_members_without_replacement",
        "tests/label_studio_coco_refinement/test_store.py::test_lost_enqueue_response_reopens_as_same_queued_batch",
        "tests/label_studio_coco_refinement/test_store.py::test_terminal_batch_retry_does_not_republish_or_allocate_again",
        "tests/label_studio_coco_refinement/test_runtime.py::test_lost_response_retry_queries_status_without_recapturing",
        "tests/label_studio_coco_refinement/test_runtime.py::test_clean_stop_then_restart_recovers_queued_batch",
    ]
    restart_cuts = [
        "conda", "run", "-n", "ms", "pytest", "-q",
        "tests/label_studio_coco_refinement/test_store.py::test_startup_recovery_rolls_back_every_pre_replacement_cut",
        "tests/label_studio_coco_refinement/test_store.py::test_startup_recovery_commits_every_post_replacement_cut_exactly_once",
        "tests/label_studio_coco_refinement/test_store.py::test_prepared_allocation_is_never_reused_after_rollback",
    ]
    scratch_multitask = [
        "conda", "run", "-n", "ms", "pytest", "-q", "-s",
        str(ROOT / "scripts" / MULTITASK.name),
        "--basetemp", str(ROOT / "pytest-tmp-multitask"),
    ]
    scratch_invalid = [
        "conda", "run", "-n", "ms", "pytest", "-q", "-s",
        str(ROOT / "scripts" / INVALID.name),
        "--basetemp", str(ROOT / "pytest-tmp-invalid"),
    ]

    started = datetime.now(UTC).isoformat()
    results = [
        run("existing_native_8", native, "existing-native-8.log"),
        run("existing_store_runtime_5", store_runtime, "existing-store-runtime-5.log"),
        run("existing_restart_cuts_14", restart_cuts, "existing-restart-cuts-14.log"),
        run("scratch_multitask", scratch_multitask, "scratch-multitask.log"),
        run("scratch_invalid", scratch_invalid, "scratch-invalid.log"),
    ]
    multitask_receipts = {
        "paused": extract_json_line(results[3]["output"], "paused"),
        "final": extract_json_line(results[3]["output"], "final"),
    }
    invalid_receipt = extract_json_line(results[4]["output"], "invalid")
    for result in results:
        result.pop("output")

    head = subprocess.check_output(["git", "rev-parse", "--short=8", "HEAD"], cwd=REPO, text=True).strip()
    full_receipt = json.loads(FULL_RECEIPT.read_text())
    existing_count = sum(value["passed"] for value in results[:3])
    final = multitask_receipts["final"] or {}
    invalid_value = invalid_receipt or {}
    checks = {
        "head_fa49e7fe": head == "fa49e7fe",
        "existing_27_passed": existing_count == 27 and all(value["exit_code"] == 0 for value in results[:3]),
        "scratch_multitask_passed": results[3]["exit_code"] == 0 and results[3]["passed"] == 1,
        "scratch_invalid_passed": results[4]["exit_code"] == 0 and results[4]["passed"] == 1,
        "captured_newer_preserved": final.get("captured_authority") == "draft" and final.get("captured_generation") == 1 and final.get("captured_id", 0) < 0,
        "uncaptured_later_preserved": final.get("uncaptured_authority") == "draft" and final.get("uncaptured_generation") == 1 and final.get("uncaptured_has_id") is False,
        "multitask_source_tree_unchanged": final.get("source_tree_before_sha256") == final.get("source_tree_after_sha256"),
        "multitask_image_tree_unchanged": final.get("image_tree_before_sha256") == final.get("image_tree_after_sha256"),
        "invalid_all_or_nothing": invalid_value.get("terminal") == "failed" and invalid_value.get("terminal_generation") == 0 and invalid_value.get("draft1_same") is True and invalid_value.get("draft3_same") is True and invalid_value.get("draft_count") == 2 and invalid_value.get("working_same") is True and invalid_value.get("manifest_same") is True,
        "invalid_source_tree_unchanged": invalid_value.get("source_tree_before_sha256") == invalid_value.get("source_tree_after_sha256"),
        "invalid_image_tree_unchanged": invalid_value.get("image_tree_before_sha256") == invalid_value.get("image_tree_after_sha256"),
        "response_loss_restart_covered": results[0]["exit_code"] == 0 and results[1]["exit_code"] == 0 and results[2]["exit_code"] == 0,
        "full_size_receipt_passed": full_receipt.get("status") == "passed" and all(full_receipt.get("checks", {}).values()),
    }
    scripts = {
        path.name: {"path": str(path), "sha256": sha256(path)}
        for path in sorted((ROOT / "scripts").iterdir())
    }
    receipt = {
        "schema_version": 1,
        "kind": "coco_refinement_wave2_bounded_matrix",
        "started_at": started,
        "completed_at": datetime.now(UTC).isoformat(),
        "head": head,
        "root": str(ROOT),
        "existing_test_count": existing_count,
        "total_test_count": sum(value["passed"] for value in results),
        "commands": results,
        "scripts": scripts,
        "multitask_receipts": multitask_receipts,
        "invalid_receipt": invalid_receipt,
        "tiny_tree_hashes": {
            "multitask_source_before": final.get("source_tree_before_sha256"),
            "multitask_source_after": final.get("source_tree_after_sha256"),
            "multitask_images_before": final.get("image_tree_before_sha256"),
            "multitask_images_after": final.get("image_tree_after_sha256"),
            "invalid_source_before": invalid_value.get("source_tree_before_sha256"),
            "invalid_source_after": invalid_value.get("source_tree_after_sha256"),
            "invalid_images_before": invalid_value.get("image_tree_before_sha256"),
            "invalid_images_after": invalid_value.get("image_tree_after_sha256"),
        },
        "full_size_complement": {
            "path": str(FULL_RECEIPT),
            "sha256": sha256(FULL_RECEIPT),
            "status": full_receipt.get("status"),
            "checks": full_receipt.get("checks"),
        },
        "checks": checks,
        "status": "passed" if all(checks.values()) else "failed",
    }
    atomic_json(ROOT / "probe-receipt.json", receipt)
    print(json.dumps({"receipt": str(ROOT / "probe-receipt.json"), "status": receipt["status"], "checks": checks}, sort_keys=True), flush=True)
    if receipt["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
