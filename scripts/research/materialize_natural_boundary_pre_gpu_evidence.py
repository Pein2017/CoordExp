#!/usr/bin/env python3
"""Run and write the exact CPU evidence needed by the pre-GPU sealer.

This command is deliberately narrow: it runs the frozen focused test file list
and the installed-Qwen CPU attention probe, then writes two canonical,
write-once JSON artifacts.  It never loads the production checkpoint or uses a
GPU.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import natural_boundary_attention_actuators as attention  # noqa: E402
from scripts.research import seal_natural_boundary_pre_gpu_receipt as sealer  # noqa: E402


DEFAULT_OUTPUT_ROOT = sealer.DEFAULT_OUTPUT_ROOT.parent / "pre-gpu-evidence-v4"
DEFAULT_TEST_RECEIPT = DEFAULT_OUTPUT_ROOT / "focused-tests.json"
DEFAULT_MASK_PROBE = DEFAULT_OUTPUT_ROOT / "installed-qwen-mask-probe.json"

FOCUSED_TEST_PATHS = (
    REPO_ROOT / "tests/research/test_finalize_natural_boundary_routing_history_evidence.py",
    REPO_ROOT / "tests/research/test_build_natural_boundary_owner_admission_census.py",
    REPO_ROOT / "tests/research/test_plan_natural_boundary_owner_support_completion.py",
    REPO_ROOT / "tests/research/test_natural_boundary_attention_actuators.py",
    REPO_ROOT / "tests/research/test_natural_boundary_residual_actuators.py",
    REPO_ROOT / "tests/research/test_run_natural_boundary_routing_history_probe.py",
    REPO_ROOT / "tests/research/test_run_natural_boundary_support_completion.py",
    REPO_ROOT / "tests/research/test_run_s_primary_natural_boundary_gate.py",
    REPO_ROOT / "tests/research/test_seal_natural_boundary_pre_gpu_receipt.py",
    REPO_ROOT / "tests/research/test_materialize_natural_boundary_pre_gpu_evidence.py",
)


class EvidenceError(RuntimeError):
    """Raised when CPU evidence cannot be materialized cleanly."""


def _write_json_once(path: Path, document: dict[str, Any]) -> dict[str, str]:
    payload = sealer.canonical_json_bytes(document) + b"\n"
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise EvidenceError(f"immutable evidence already exists with different bytes: {path}")
    else:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    return {"path": str(path), "sha256": sealer.sha256_file(path)}


def _test_command(paths: Sequence[Path]) -> list[str]:
    return [sys.executable, "-m", "pytest", "-q", *(str(path) for path in paths)]


def run_focused_tests(paths: Sequence[Path] = FOCUSED_TEST_PATHS) -> dict[str, Any]:
    resolved: list[Path] = []
    for path in paths:
        candidate = path.resolve()
        if candidate.is_symlink() or not candidate.is_file():
            raise EvidenceError(f"focused test is not a regular file: {candidate}")
        resolved.append(candidate)
    command = _test_command(resolved)
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    output = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip())
    passed_match = re.search(r"(?P<count>\d+) passed", output)
    receipt = {
        "schema_version": "natural_boundary_focused_tests.v1",
        "status": "passed" if completed.returncode == 0 and passed_match else "failed",
        "passed": bool(completed.returncode == 0 and passed_match),
        "exit_code": int(completed.returncode),
        "fail_count": 0 if completed.returncode == 0 else None,
        "errors": 0 if completed.returncode == 0 else None,
        "failures": 0 if completed.returncode == 0 else None,
        "passed_test_count": int(passed_match.group("count")) if passed_match else None,
        "command": " ".join(command),
        "cwd": str(REPO_ROOT),
        "focused_tests": [
            {"path": str(path), "sha256": sealer.sha256_file(path)} for path in resolved
        ],
        "output": output,
        "gpu_used": False,
        "production_model_loaded": False,
    }
    if not receipt["passed"]:
        raise EvidenceError(f"focused tests did not pass: {output}")
    return receipt


def materialize(
    *,
    test_receipt_path: Path = DEFAULT_TEST_RECEIPT,
    mask_probe_path: Path = DEFAULT_MASK_PROBE,
) -> dict[str, Any]:
    test_receipt = run_focused_tests()
    test_ref = _write_json_once(test_receipt_path, test_receipt)
    mask_probe = attention.run_installed_qwen_cpu_probe()
    mask_ref = _write_json_once(mask_probe_path, mask_probe)
    if mask_probe.get("status") != "passed":
        raise EvidenceError(
            f"installed-Qwen CPU mask probe did not pass: {json.dumps(mask_probe, sort_keys=True)}"
        )
    return {"status": "passed", "focused_tests": test_ref, "mask_probe": mask_ref}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-receipt", type=Path, default=DEFAULT_TEST_RECEIPT)
    parser.add_argument("--mask-probe", type=Path, default=DEFAULT_MASK_PROBE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = materialize(
            test_receipt_path=args.test_receipt,
            mask_probe_path=args.mask_probe,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    except (EvidenceError, OSError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
