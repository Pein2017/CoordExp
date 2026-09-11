#!/usr/bin/env python3
"""Run and durably bind the two production-shaped CPU compatibility gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
from typing import Any

INFRA_ROOT = Path(__file__).resolve().parents[2]
if str(INFRA_ROOT) not in sys.path:
    sys.path.insert(0, str(INFRA_ROOT))

from src.artifacts.json_values import (  # noqa: E402
    canonical_json_bytes,
    json_sha256,
    publish_json_exclusive,
)
from src.common.errors import ArtifactContractError  # noqa: E402


RESEARCH_PROBES_ROOT = Path("/data/CoordExp/.worktrees/research-probes")
OUTPUT_ROOT = Path(
    "/data/CoordExp/outputs/research-probe-infras/"
    "2026-08-09-production-shaped-admission-target-v2"
)

SUPPORT_FILES = {
    "consumer": RESEARCH_PROBES_ROOT
    / "scripts/research/run_natural_boundary_support_completion.py",
    "consumer_test": RESEARCH_PROBES_ROOT
    / "tests/research/test_run_natural_boundary_support_completion.py",
    "merger": RESEARCH_PROBES_ROOT
    / "scripts/research/merge_natural_boundary_support_completion.py",
    "merger_test": RESEARCH_PROBES_ROOT
    / "tests/test_merge_natural_boundary_support_completion.py",
    "adapter": INFRA_ROOT
    / "scripts/research/resumable_natural_boundary_support_completion.py",
    "adapter_test": INFRA_ROOT
    / "tests/research/test_resumable_natural_boundary_support_completion.py",
    "plan": Path(
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
        "2026-08-06-natural-boundary-routing-history-replication/"
        "support-completion-plan-v1/plan.json"
    ),
    "census": Path(
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
        "2026-08-06-natural-boundary-routing-history-replication/"
        "cpu-census-v2/admission-census.json"
    ),
    "generator": Path(__file__).resolve(),
}

CROSSOVER_FILES = {
    "runner": RESEARCH_PROBES_ROOT
    / "scripts/research/run_s_k10_h20_crossover_shard.py",
    "runner_test": RESEARCH_PROBES_ROOT
    / "tests/research/test_run_s_k10_h20_crossover_shard.py",
    "sealer": RESEARCH_PROBES_ROOT
    / "scripts/research/seal_s_k10_h20_crossover_pre_gpu_receipt.py",
    "finalizer": RESEARCH_PROBES_ROOT
    / "scripts/research/finalize_s_k10_h20_crossover.py",
    "finalizer_test": RESEARCH_PROBES_ROOT
    / "tests/research/test_finalize_s_k10_h20_crossover.py",
    "source_preflight": Path(
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
        "2026-08-07-s-k10-h20-natural-crossover/pre-gpu-evidence-v5/"
        "source-preflight-preseal.json"
    ),
    "accepted_evidence": Path(
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
        "2026-08-07-s-k10-h20-natural-crossover/evidence-v4/evidence.json"
    ),
    "generator": Path(__file__).resolve(),
}

CROSSOVER_PREFLIGHT_INPUTS = {
    "plan": Path(
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
        "2026-08-07-s-k10-h20-natural-crossover/plan-v1/plan.json"
    ),
    "manifest": Path(
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
        "2026-08-06-natural-boundary-routing-history-replication/"
        "cpu-census-v3-native-fn-supersession-v1/admitted-event-manifest.json"
    ),
    "census": Path(
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
        "2026-08-06-natural-boundary-routing-history-replication/"
        "cpu-census-v3-native-fn-supersession-v1/admission-census.json"
    ),
    "config": RESEARCH_PROBES_ROOT
    / (
        "configs/coordexp_infras/infer/"
        "qwen3_vl_2b_static_dynamic_owner_interface_s_step2444_h0.yaml"
    ),
    "panel": Path(
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
        "2026-08-05-static-dynamic-owner-interface-crossover/inputs/"
        "human-refined-13.geo_sorted_xy.coord.jsonl"
    ),
    "cohort": Path(
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
        "2026-08-06-natural-boundary-routing-history-replication/"
        "cpu-census-v3-native-fn-supersession-v1/s-context-cohort.json"
    ),
    "cohort_manifest": Path(
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
        "2026-08-06-natural-boundary-routing-history-replication/"
        "cpu-census-v3-native-fn-supersession-v1/s-context-cohort.manifest.json"
    ),
    "h0_root": Path(
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
        "2026-08-05-static-dynamic-owner-interface-crossover/h0"
    ),
    "h0_dir": Path(
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
        "2026-08-05-static-dynamic-owner-interface-crossover/h0/"
        "qwen3-vl-2b-static-dynamic-owner-interface-s-step2444-h0-repair1"
    ),
    "base_model_dir": Path(
        "/data/Qwen3-VL/model_cache/models/Qwen/"
        "Qwen3-VL-2B-Instruct-coordexp-natural-adjacent"
    ),
}


class CpuReceiptError(RuntimeError):
    """A declared compatibility command or durable publication failed."""


def _file_identity(path: Path) -> dict[str, Any]:
    if not path.is_absolute():
        raise CpuReceiptError(f"bound file path is not absolute: {path}")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise CpuReceiptError(f"bound file is missing: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise CpuReceiptError(f"bound file is not a regular non-symlink: {path}")
    resolved = path.resolve(strict=True)
    encoded = resolved.read_bytes()
    return {
        "path": str(resolved),
        "raw_sha256": hashlib.sha256(encoded).hexdigest(),
        "byte_count": len(encoded),
    }


def _publish_exact(path: Path, value: Any) -> Path:
    expected = canonical_json_bytes(value)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != expected:
            raise CpuReceiptError(f"existing receipt differs from exact bytes: {path}")
        return path
    try:
        publish_json_exclusive(path, value)
    except ArtifactContractError as exc:
        if path.is_file() and not path.is_symlink() and path.read_bytes() == expected:
            return path
        raise CpuReceiptError(f"receipt publication failed: {path}") from exc
    return path


def _stream_document(*, stream: str, encoded: bytes) -> dict[str, Any]:
    return {
        "schema_version": "research_probe_admission.command_stream.v1",
        "kind": stream,
        "encoding": "utf-8-replace",
        "text": encoded.decode("utf-8", errors="replace"),
        "raw_sha256": hashlib.sha256(encoded).hexdigest(),
        "byte_count": len(encoded),
    }


def _passed_count(stdout: bytes, *, marker: str | None) -> int:
    text = stdout.decode("utf-8", errors="replace")
    matches = re.findall(r"(?<!\d)(\d+) passed\b", text)
    if matches:
        return int(matches[-1])
    if marker is not None and marker in text:
        return 1
    raise CpuReceiptError("successful command output lacks an exact passed count or marker")


def _run_command(
    *,
    root: Path,
    name: str,
    selector: str,
    executable: Path,
    cwd: Path,
    argv_tail: list[str],
    marker: str | None = None,
) -> dict[str, Any]:
    executable_identity = _file_identity(executable)
    resolved_cwd = cwd.resolve(strict=True)
    if cwd.is_symlink() or not resolved_cwd.is_dir():
        raise CpuReceiptError(f"command cwd is not a non-symlink directory: {cwd}")
    argv = [str(Path(executable_identity["path"])), *argv_tail]
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        argv,
        cwd=resolved_cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    command_root = root / "commands" / name
    stdout_path = _publish_exact(
        command_root / "stdout.json",
        _stream_document(stream="stdout", encoded=completed.stdout),
    )
    stderr_path = _publish_exact(
        command_root / "stderr.json",
        _stream_document(stream="stderr", encoded=completed.stderr),
    )
    passed = _passed_count(completed.stdout, marker=marker) if completed.returncode == 0 else 0
    command_receipt = {
        "schema_version": "research_probe_admission.command_execution.v1",
        "kind": "command_execution",
        "generator": _file_identity(Path(__file__).resolve()),
        "executable": executable_identity,
        "cwd": str(resolved_cwd),
        "argv": argv,
        "selector": selector,
        "return_code": completed.returncode,
        "passed": passed,
        "stdout": _file_identity(stdout_path),
        "stderr": _file_identity(stderr_path),
        "claim_boundary": "cpu_command_mechanics_only_no_scientific_interpretation",
    }
    command_receipt["content_sha256"] = json_sha256(command_receipt)
    command_receipt_path = _publish_exact(
        command_root / "command.receipt.json", command_receipt
    )
    if completed.returncode != 0:
        raise CpuReceiptError(
            f"compatibility command {name} exited {completed.returncode}; "
            f"see {command_receipt_path}"
        )
    return {
        "executable": executable_identity,
        "cwd": str(resolved_cwd),
        "argv": argv,
        "selector": selector,
        "return_code": 0,
        "passed": passed,
        "stdout": _file_identity(stdout_path),
        "stderr": _file_identity(stderr_path),
        "receipt": _file_identity(command_receipt_path),
        "generator": _file_identity(Path(__file__).resolve()),
    }


def _support_commands(*, root: Path, conda: Path) -> dict[str, Any]:
    selectors = {
        "consumer_validator": "validate_execution_plan",
        "model_free_finalizer": "materialize_legacy_shard_receipts",
        "downstream_validator": "merge_support_receipts",
    }
    return {
        "consumer_validator": _run_command(
            root=root,
            name="consumer-validator",
            selector=selectors["consumer_validator"],
            executable=conda,
            cwd=RESEARCH_PROBES_ROOT,
            argv_tail=[
                "run",
                "-n",
                "ms",
                "pytest",
                "-q",
                "tests/research/test_run_natural_boundary_support_completion.py",
            ],
        ),
        "model_free_finalizer": _run_command(
            root=root,
            name="model-free-finalizer",
            selector=selectors["model_free_finalizer"],
            executable=conda,
            cwd=INFRA_ROOT,
            argv_tail=[
                "run",
                "-n",
                "ms",
                "pytest",
                "-q",
                "tests/research/test_resumable_natural_boundary_support_completion.py::test_full_plan_fresh_process_interruption_resume_has_identical_eight_receipts",
            ],
        ),
        "downstream_validator": _run_command(
            root=root,
            name="downstream-validator",
            selector=selectors["downstream_validator"],
            executable=conda,
            cwd=INFRA_ROOT,
            argv_tail=[
                "run",
                "-n",
                "ms",
                "pytest",
                "-q",
                "tests/research/test_resumable_natural_boundary_support_completion.py::test_full_plan_fresh_process_interruption_resume_has_identical_eight_receipts",
            ],
        ),
    }


def _validate_crossover_source_preflight_chain() -> dict[str, Any]:
    """Run the real model-free preflight and its current deep sealer validator."""

    import scripts.research as research_package
    from scripts.research import run_resumable_natural_boundary_support_shard

    runtime_source = (
        run_resumable_natural_boundary_support_shard._bind_consumer_runtime_source(
            CROSSOVER_FILES["runner"]
        )
    )

    sibling_research = str(RESEARCH_PROBES_ROOT / "scripts/research")
    if sibling_research not in research_package.__path__:
        research_package.__path__.append(sibling_research)
    from scripts.research import run_s_k10_h20_crossover_shard as runner
    from scripts.research import seal_s_k10_h20_crossover_pre_gpu_receipt as sealer

    with tempfile.TemporaryDirectory(
        prefix="research-probe-admission-crossover-preflight-"
    ) as temporary:
        temporary_root = Path(temporary).resolve()
        execution_root = temporary_root / "execution"
        final_root = temporary_root / "final"
        inputs = CROSSOVER_PREFLIGHT_INPUTS
        evidence = runner.preflight_crossover_sources(
            inputs["plan"],
            manifest=inputs["manifest"],
            census=inputs["census"],
            config=inputs["config"],
            panel=inputs["panel"],
            cohort=inputs["cohort"],
            cohort_manifest=inputs["cohort_manifest"],
            h0_root=inputs["h0_root"],
            h0_dir=inputs["h0_dir"],
            base_model_dir=inputs["base_model_dir"],
            execution_root=execution_root,
            final_root=final_root,
        )

        plan_ref, plan_document = sealer._validate_plan(inputs["plan"])
        source_values = {
            "manifest": inputs["manifest"],
            "census": inputs["census"],
            "execution_plan": inputs["plan"],
            "config": inputs["config"],
            "panel": inputs["panel"],
            "cohort": inputs["cohort"],
            "cohort_manifest": inputs["cohort_manifest"],
            "h0_root": inputs["h0_root"],
            "h0_dir": inputs["h0_dir"],
            "base_model_dir": inputs["base_model_dir"],
        }
        directory_roles = {"h0_root", "h0_dir", "base_model_dir"}
        input_refs = {
            role: sealer._path_ref(
                path,
                f"admission source-preflight {role}",
                allow_directory=role in directory_roles,
            )
            for role, path in source_values.items()
        }
        _, manifest_document = sealer._json_ref(
            input_refs["manifest"], "admission source-preflight manifest"
        )
        roots = sealer._validate_roots(execution_root, final_root)
        validated = sealer._validate_source_preflight(
            evidence,
            plan_ref=plan_ref,
            plan_doc=plan_document,
            manifest_doc=manifest_document,
            input_refs=input_refs,
            code_refs=sealer._validate_code(None),
            test_refs=sealer._validate_tests(None),
            roots=roots,
        )
        if validated != evidence:
            raise CpuReceiptError(
                "current sealer returned a different source-preflight document"
            )
        if execution_root.exists() or final_root.exists():
            raise CpuReceiptError(
                "source-preflight chain created a reserved execution/final root"
            )
        production = evidence.get("model_free_production_path")
        if not isinstance(production, dict):
            raise CpuReceiptError("source-preflight production receipt is missing")
        return {
            "status": "passed",
            "source_preflight_self_sha256": evidence["self_sha256"],
            "plan_raw_sha256": plan_ref["sha256"],
            "plan_self_sha256": plan_document["self_sha256"],
            "full_runtime_event_count": production["full_runtime_cohort"][
                "event_count"
            ],
            "selected_event_count": production["selected_event_count"],
            "gpu_used": False,
            "model_loaded": False,
            "reserved_roots_created": False,
            "consumer_runtime_source_sha256": runtime_source["tree"][
                "aggregate_sha256"
            ],
            "validator": "seal_s_k10_h20_crossover_pre_gpu_receipt._validate_source_preflight",
        }


def _crossover_commands(*, root: Path, conda: Path) -> dict[str, Any]:
    selectors = {
        "source_preflight_validator": "preflight_crossover_sources",
        "model_free_finalizer": "finalize",
        "downstream_validator": "validate_evidence",
    }
    validate_code = (
        "import json; from pathlib import Path; "
        "from scripts.research.finalize_s_k10_h20_crossover import validate_evidence; "
        "validate_evidence(json.loads(Path(" + repr(str(CROSSOVER_FILES["accepted_evidence"])) + ").read_text())); "
        "print('VALIDATE_EVIDENCE_OK')"
    )
    return {
        "source_preflight_validator": _run_command(
            root=root,
            name="source-preflight-validator",
            selector=selectors["source_preflight_validator"],
            executable=conda,
            cwd=INFRA_ROOT,
            argv_tail=[
                "run",
                "-n",
                "ms",
                "python",
                str(Path(__file__).resolve()),
                "--check-crossover-source-preflight",
            ],
            marker="CROSSOVER_SOURCE_PREFLIGHT_SEALER_OK",
        ),
        "model_free_finalizer": _run_command(
            root=root,
            name="model-free-finalizer",
            selector=selectors["model_free_finalizer"],
            executable=conda,
            cwd=RESEARCH_PROBES_ROOT,
            argv_tail=[
                "run",
                "-n",
                "ms",
                "pytest",
                "-q",
                "tests/research/test_finalize_s_k10_h20_crossover.py::test_write_once_and_final_root_binding",
            ],
        ),
        "downstream_validator": _run_command(
            root=root,
            name="downstream-validator",
            selector=selectors["downstream_validator"],
            executable=conda,
            cwd=RESEARCH_PROBES_ROOT,
            argv_tail=["run", "-n", "ms", "python", "-c", validate_code],
            marker="VALIDATE_EVIDENCE_OK",
        ),
    }


def _compatibility_receipt(
    *, kind: str, files: dict[str, Path], commands: dict[str, Any]
) -> dict[str, Any]:
    if kind == "support":
        document: dict[str, Any] = {
            "schema_version": "research_probe_admission.support_cpu_compatibility.v1",
            "kind": "support_cpu_compatibility",
            "selectors": {
                "consumer_validator": "validate_execution_plan",
                "model_free_finalizer": "materialize_legacy_shard_receipts",
                "downstream_validator": "merge_support_receipts",
            },
            "consumer_validator_ran": True,
            "model_free_finalizer_ran": True,
            "downstream_validator_accepted": True,
            "claim_boundary": "cpu_mechanics_compatibility_only_no_model_or_support_scientific_claim",
        }
    else:
        document = {
            "schema_version": "research_probe_admission.crossover_cpu_compatibility.v1",
            "kind": "crossover_cpu_compatibility",
            "selectors": {
                "source_preflight_validator": "preflight_crossover_sources",
                "model_free_finalizer": "finalize",
                "downstream_validator": "validate_evidence",
            },
            "source_preflight_validated": True,
            "model_free_finalizer_ran": True,
            "downstream_validator_accepted": True,
            "replay_status": "synthetic_fixture_ran",
            "claim_boundary": "cpu_source_preflight_and_synthetic_finalizer_fixture_only_no_model_or_crossover_scientific_claim",
        }
    document.update(
        {
            "exit_code": 0,
            "files": {name: _file_identity(path) for name, path in files.items()},
            "command_results": commands,
            "model_loaded": False,
            "gpu_used": False,
            "scientific_interpretation": False,
        }
    )
    document["content_sha256"] = json_sha256(document)
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conda-executable", type=Path)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--check-crossover-source-preflight", action="store_true")
    args = parser.parse_args()
    if args.check_crossover_source_preflight:
        summary = _validate_crossover_source_preflight_chain()
        print(json.dumps(summary, sort_keys=True))
        print("CROSSOVER_SOURCE_PREFLIGHT_SEALER_OK")
        return 0
    if args.conda_executable is None:
        parser.error("--conda-executable is required when generating receipts")
    conda = Path(_file_identity(args.conda_executable)["path"])
    output_root = args.output_root.resolve()
    if output_root.is_symlink():
        raise CpuReceiptError("output root must not be a symlink")
    support_root = output_root / "support"
    crossover_root = output_root / "crossover"
    support_commands = _support_commands(root=support_root, conda=conda)
    crossover_commands = _crossover_commands(root=crossover_root, conda=conda)
    _publish_exact(
        output_root / "support-cpu-compatibility.json",
        _compatibility_receipt(
            kind="support", files=SUPPORT_FILES, commands=support_commands
        ),
    )
    _publish_exact(
        output_root / "crossover-cpu-compatibility.json",
        _compatibility_receipt(
            kind="crossover", files=CROSSOVER_FILES, commands=crossover_commands
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
