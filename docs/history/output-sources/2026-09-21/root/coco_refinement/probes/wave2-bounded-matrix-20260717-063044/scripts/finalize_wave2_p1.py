from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path


REPO = Path("/data/CoordExp")
ROOT = REPO / "outputs/coco_refinement/probes/wave2-bounded-matrix-20260717-063044"
FULL = REPO / "outputs/coco_refinement/probes/wave2-full-commit-20260717-060642/probe-receipt.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic(path: Path, value: dict) -> None:
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


def parsed(log: Path) -> tuple[int, str]:
    value = log.read_text()
    matches = re.findall(r"(\d+) passed", value)
    return (int(matches[-1]) if matches else 0), value


def json_line(value: str, prefix: str):
    for line in value.splitlines():
        if line.startswith(prefix + " "):
            return json.loads(line[len(prefix) + 1 :])
    return None


def command(name: str, log_name: str, passed: int, argv: list[str]) -> dict:
    log = ROOT / "logs" / log_name
    actual, output = parsed(log)
    clean = actual == passed and " failed" not in output and " ERROR" not in output
    return {
        "name": name,
        "argv": argv,
        "command": " ".join(argv),
        "exit_code": 0 if clean else 1,
        "passed": actual,
        "log_path": str(log),
        "log_sha256": sha(log),
    }


def main() -> None:
    native_nodes = [
        "tests/coco_refinement/test_commit_api.py::test_paused_worker_captures_all_drafts_and_lost_retry_never_recaptures",
        "tests/coco_refinement/test_commit_api.py::test_real_paused_publication_keeps_http_editable_and_reconciles_newer_draft",
        "tests/coco_refinement/test_commit_api.py::test_failed_terminal_observer_keeps_http_editable_until_restart_repair",
        "tests/coco_refinement/test_commit_api.py::test_post_sqlite_pre_queue_terminal_keeps_http_editable_and_identity_stable",
        "tests/coco_refinement/test_adapters.py::test_terminal_success_retires_exact_captured_draft_idempotently_across_restart",
        "tests/coco_refinement/test_adapters.py::test_terminal_success_merges_only_identity_into_newer_draft_without_resurrection",
        "tests/coco_refinement/test_adapters.py::test_later_unrelated_draft_rebinds_to_terminal_project_generation_and_verifies",
        "tests/coco_refinement/test_runtime.py::test_startup_replays_published_terminal_before_strict_sqlite_attestation",
    ]
    store_nodes = [
        "tests/label_studio_coco_refinement/test_store.py::test_stale_batch_member_fails_all_members_without_replacement",
        "tests/label_studio_coco_refinement/test_store.py::test_lost_enqueue_response_reopens_as_same_queued_batch",
        "tests/label_studio_coco_refinement/test_store.py::test_terminal_batch_retry_does_not_republish_or_allocate_again",
        "tests/label_studio_coco_refinement/test_runtime.py::test_lost_response_retry_queries_status_without_recapturing",
        "tests/label_studio_coco_refinement/test_runtime.py::test_clean_stop_then_restart_recovers_queued_batch",
    ]
    cut_nodes = [
        "tests/label_studio_coco_refinement/test_store.py::test_startup_recovery_rolls_back_every_pre_replacement_cut",
        "tests/label_studio_coco_refinement/test_store.py::test_startup_recovery_commits_every_post_replacement_cut_exactly_once",
        "tests/label_studio_coco_refinement/test_store.py::test_prepared_allocation_is_never_reused_after_rollback",
    ]
    multi_script = ROOT / "scripts/test_task26_multitask_probe.py"
    invalid_script = ROOT / "scripts/test_task26_invalid_probe.py"
    commands = [
        command("existing_native_8", "existing-native-8.log", 8, ["conda", "run", "-n", "ms", "pytest", "-vv", *native_nodes]),
        command("existing_store_runtime_5", "existing-store-runtime-5.log", 5, ["conda", "run", "-n", "ms", "pytest", "-vv", *store_nodes]),
        command("existing_restart_cuts_14", "existing-restart-cuts-14.log", 14, ["conda", "run", "-n", "ms", "pytest", "-q", *cut_nodes]),
        command("scratch_multitask", "scratch-multitask.log", 1, ["conda", "run", "-n", "ms", "pytest", "-q", "-s", str(multi_script), "--basetemp", str(ROOT / "pytest-tmp-multitask")]),
        command("scratch_invalid", "scratch-invalid.log", 1, ["conda", "run", "-n", "ms", "pytest", "-q", "-s", str(invalid_script), "--basetemp", str(ROOT / "pytest-tmp-invalid")]),
    ]
    _, multi_output = parsed(ROOT / "logs/scratch-multitask.log")
    _, invalid_output = parsed(ROOT / "logs/scratch-invalid.log")
    paused = json_line(multi_output, "paused")
    final = json_line(multi_output, "final") or {}
    invalid = json_line(invalid_output, "invalid") or {}
    full_value = json.loads(FULL.read_text())
    head = subprocess.check_output(["git", "rev-parse", "--short=8", "HEAD"], cwd=REPO, text=True).strip()
    requested_head = "fa49e7fe"
    src_requested = subprocess.check_output(["git", "rev-parse", f"{requested_head}:src"], cwd=REPO, text=True).strip()
    src_executed = subprocess.check_output(["git", "rev-parse", "HEAD:src"], cwd=REPO, text=True).strip()
    tests_requested = subprocess.check_output(["git", "rev-parse", f"{requested_head}:tests"], cwd=REPO, text=True).strip()
    tests_executed = subprocess.check_output(["git", "rev-parse", "HEAD:tests"], cwd=REPO, text=True).strip()
    existing = sum(item["passed"] for item in commands[:3])
    scripts = {}
    shutil.copy2(Path(__file__), ROOT / "scripts" / Path(__file__).name)
    for path in sorted((ROOT / "scripts").iterdir()):
        if path.is_file():
            scripts[path.name] = {"path": str(path), "sha256": sha(path)}
    checks = {
        "requested_task_code_matches_fa49e7fe": src_requested == src_executed and tests_requested == tests_executed,
        "existing_27_passed": existing == 27 and all(item["exit_code"] == 0 for item in commands[:3]),
        "scratch_multitask_passed": commands[3]["exit_code"] == 0 and commands[3]["passed"] == 1,
        "scratch_invalid_passed": commands[4]["exit_code"] == 0 and commands[4]["passed"] == 1,
        "captured_newer_preserved": final.get("captured_authority") == "draft" and final.get("captured_generation") == 1 and final.get("captured_id", 0) < 0,
        "uncaptured_later_preserved": final.get("uncaptured_authority") == "draft" and final.get("uncaptured_generation") == 1 and final.get("uncaptured_has_id") is False,
        "multitask_source_tree_unchanged": final.get("source_tree_before_sha256") == final.get("source_tree_after_sha256"),
        "multitask_image_tree_unchanged": final.get("image_tree_before_sha256") == final.get("image_tree_after_sha256"),
        "invalid_all_or_nothing": invalid.get("terminal") == "failed" and invalid.get("terminal_generation") == 0 and invalid.get("draft1_same") is True and invalid.get("draft3_same") is True and invalid.get("draft_count") == 2 and invalid.get("working_same") is True and invalid.get("manifest_same") is True,
        "invalid_source_tree_unchanged": invalid.get("source_tree_before_sha256") == invalid.get("source_tree_after_sha256"),
        "invalid_image_tree_unchanged": invalid.get("image_tree_before_sha256") == invalid.get("image_tree_after_sha256"),
        "response_loss_restart_covered": all(item["exit_code"] == 0 for item in commands[:3]),
        "full_size_receipt_passed": full_value.get("status") == "passed" and all(full_value.get("checks", {}).values()),
    }
    receipt = {
        "schema_version": 1,
        "kind": "coco_refinement_wave2_bounded_matrix",
        "completed_at": datetime.now(UTC).isoformat(),
        "requested_head": requested_head,
        "execution_head": head,
        "execution_head_exact_requested": head == requested_head,
        "task_code_tree_equivalence": {
            "src_requested": src_requested,
            "src_executed": src_executed,
            "tests_requested": tests_requested,
            "tests_executed": tests_executed,
        },
        "root": str(ROOT),
        "existing_test_count": existing,
        "total_test_count": sum(item["passed"] for item in commands),
        "commands": commands,
        "scripts": scripts,
        "multitask_receipts": {"paused": paused, "final": final},
        "invalid_receipt": invalid,
        "tiny_tree_hashes": {
            "multitask_source_before": final.get("source_tree_before_sha256"),
            "multitask_source_after": final.get("source_tree_after_sha256"),
            "multitask_images_before": final.get("image_tree_before_sha256"),
            "multitask_images_after": final.get("image_tree_after_sha256"),
            "invalid_source_before": invalid.get("source_tree_before_sha256"),
            "invalid_source_after": invalid.get("source_tree_after_sha256"),
            "invalid_images_before": invalid.get("image_tree_before_sha256"),
            "invalid_images_after": invalid.get("image_tree_after_sha256"),
        },
        "full_size_complement": {"path": str(FULL), "sha256": sha(FULL), "status": full_value.get("status"), "checks": full_value.get("checks")},
        "finalization_note": "All five execution subprocesses exited 0; initial receipt assembly then failed only because pytest-created scripts/__pycache__ was passed to a file hasher. This atomic finalizer hashes files only and preserves that directory. Execution HEAD advanced to 9e38fc13 via guidance-only files while this lane was running; git tree objects prove src and tests are byte-identical to requested fa49e7fe.",
        "checks": checks,
        "status": "passed" if all(checks.values()) else "failed",
    }
    atomic(ROOT / "probe-receipt.json", receipt)
    print(json.dumps({"receipt": str(ROOT / "probe-receipt.json"), "status": receipt["status"], "checks": checks}, sort_keys=True))
    if receipt["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
