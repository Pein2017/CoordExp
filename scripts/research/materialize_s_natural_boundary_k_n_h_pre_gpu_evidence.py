#!/usr/bin/env python3
"""Materialize current-schema CPU-only evidence for the S K/N/H pre-GPU sealer."""

from __future__ import annotations

import argparse
from importlib import metadata
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import seal_s_natural_boundary_k_n_h_pre_gpu_receipt as sealer  # noqa: E402


class EvidenceError(RuntimeError):
    """Raised when current CPU evidence cannot be produced or sealed safely."""


def _producer_ref() -> dict[str, Any]:
    path = sealer.CODE_ROLE_PATHS[sealer.EVIDENCE_MATERIALIZER_ROLE].resolve(strict=True)
    return {
        "path": str(path),
        "sha256": sealer.sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _write_json_once(path: str | Path, document: Mapping[str, Any]) -> dict[str, Any]:
    target = sealer._absolute(path, "evidence output")
    payload = sealer.canonical_json_bytes(document) + b"\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        if target.is_symlink() or not target.is_file() or target.read_bytes() != payload:
            raise EvidenceError(f"immutable evidence already exists with different bytes: {target}")
        return {"path": str(target), "sha256": sealer.sha256_bytes(payload), "byte_identical": True}
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        raise
    return {"path": str(target), "sha256": sealer.sha256_bytes(payload), "byte_identical": False}


def _focused_argv() -> list[str]:
    return [
        str(Path(sys.executable).resolve(strict=True)),
        "-m",
        "pytest",
        "-q",
        *(str(path) for path in sealer.REQUIRED_FOCUSED_TEST_PATHS),
    ]


def run_focused_tests() -> dict[str, Any]:
    paths = tuple(sealer.REQUIRED_FOCUSED_TEST_PATHS)
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise EvidenceError(f"focused test is not an exact regular file: {path}")
    completed = subprocess.run(
        _focused_argv(), cwd=sealer.REPO_ROOT, check=False, capture_output=True, text=True
    )
    stdout, stderr = completed.stdout, completed.stderr
    output = stdout + stderr
    matches = re.findall(r"(?m)(?<!\d)(\d+) passed(?:,|\s|$)", output)
    if completed.returncode != 0 or len(matches) != 1:
        raise EvidenceError(
            f"focused tests did not produce one passing result (exit={completed.returncode}): {output}"
        )
    count = int(matches[0])
    if count < len(paths):
        raise EvidenceError(f"focused test pass count {count} is smaller than exact suite size {len(paths)}")
    return {
        "schema_version": sealer.FOCUSED_TEST_EVIDENCE_SCHEMA_VERSION,
        "status": "passed",
        "producer": _producer_ref(),
        "command": sealer.FOCUSED_TEST_COMMAND,
        "execution_argv": _focused_argv(),
        "cwd": str(sealer.REPO_ROOT),
        "exit_code": int(completed.returncode),
        "passed_test_count": count,
        "focused_tests": [
            {"path": str(path.resolve(strict=True)), "sha256": sealer.sha256_file(path)}
            for path in paths
        ],
        "stdout": stdout,
        "stderr": stderr,
        "output": output,
        "output_sha256": sealer.sha256_bytes(output.encode("utf-8")),
    }


def observe_runtime_evidence() -> dict[str, Any]:
    try:
        runtime = {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "torch_version": metadata.version("torch"),
            "transformers_version": metadata.version("transformers"),
            "backend": "hf",
            "dtype": "fp32",
            "attn_implementation": "sdpa",
            "no_training": True,
        }
    except metadata.PackageNotFoundError as exc:
        raise EvidenceError(f"installed runtime package is unavailable: {exc.name}") from exc
    return {
        "schema_version": sealer.RUNTIME_EVIDENCE_SCHEMA_VERSION,
        "status": "passed",
        "producer": _producer_ref(),
        "cpu_only": True,
        "gpu_used": False,
        "model_loaded": False,
        "runtime": runtime,
        "forced_math": {"enabled": True, "backend": "MATH"},
        "device_policy": sealer.DEVICE_POLICY,
    }


def materialize(*, focused_test_receipt: str | Path, runtime_evidence: str | Path) -> dict[str, Any]:
    focused = run_focused_tests()
    focused_ref = _write_json_once(focused_test_receipt, focused)
    runtime_ref = _write_json_once(runtime_evidence, observe_runtime_evidence())
    return {"status": "passed", "focused_test_receipt": focused_ref, "runtime_evidence": runtime_ref}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--focused-test-receipt", type=Path, required=True)
    parser.add_argument("--runtime-evidence", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = materialize(
            focused_test_receipt=args.focused_test_receipt,
            runtime_evidence=args.runtime_evidence,
        )
    except (EvidenceError, OSError, ValueError) as exc:
        print(f"blocked: {exc}", file=sys.stderr)
        return 2
    print(sealer.canonical_json_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
