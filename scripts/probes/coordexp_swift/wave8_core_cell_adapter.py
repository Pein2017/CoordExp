#!/usr/bin/env python3
"""Validate native core evidence and publish one Wave 8 matrix cell.

This adapter is CPU-side only.  It does not launch a model, CUDA work, cache
preparation, training, or evaluation; it converts already-passed native
receipts into the exact generic cell contract owned by the Wave 8 matrix.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probes.coordexp_swift import (  # noqa: E402
    wave2_packed_parity as wave2,
    wave3_zero_weight_gpu as wave3,
    wave7_exact_resume_postrun as wave7_postrun,
    wave7_exact_resume_sequence as wave7_sequence,
    wave8_compatibility_matrix as matrix,
)
from src.qwen.parity import canonical_json_bytes  # noqa: E402


CELL_KINDS = (
    "exact_resume",
    "packed_parity_all_layer_fa2",
    "protected_loss",
)


class Wave8CoreCellError(RuntimeError):
    def __init__(self, message: str, *, code: str) -> None:
        self.code = code
        super().__init__(message)


def _load_native_artifacts(
    paths: Sequence[Path], *, schemas: Mapping[str, str]
) -> dict[str, tuple[Path, dict[str, Any]]]:
    if len(paths) != len(schemas):
        raise Wave8CoreCellError(
            "native receipt inventory is not exact",
            code="wave8_core_cell.native_inventory",
        )
    by_schema = {schema: name for name, schema in schemas.items()}
    result: dict[str, tuple[Path, dict[str, Any]]] = {}
    for requested in paths:
        source = matrix._regular_file(Path(requested), owner="native receipt")
        payload = matrix._load_json_file(source, owner="native receipt")
        matrix._require_canonical_file(source, payload, owner="native receipt")
        name = by_schema.get(payload.get("schema"))
        if name is None or name in result:
            raise Wave8CoreCellError(
                "native receipt schema inventory is not exact",
                code="wave8_core_cell.native_schema",
            )
        result[name] = (source, payload)
    if set(result) != set(schemas):
        raise Wave8CoreCellError(
            "native receipt schema inventory is incomplete",
            code="wave8_core_cell.native_inventory",
        )
    return result


def _validate_packed(native_receipts: Sequence[Path]) -> None:
    artifacts = _load_native_artifacts(
        native_receipts,
        schemas={
            "plan": wave2.PARITY_PLAN_SCHEMA,
            "receipt": wave2.PARITY_RECEIPT_SCHEMA,
        },
    )
    plan = wave2.validate_parity_plan(artifacts["plan"][1])
    receipt_path, receipt = artifacts["receipt"]
    validated = wave2.validate_parity_receipt(
        receipt,
        expected_plan=plan,
        expected_receipt_target=receipt_path,
    )
    if validated.get("terminal_status") != "passed":
        raise Wave8CoreCellError(
            "packed parity native receipt did not pass",
            code="wave8_core_cell.native_not_passed",
        )


def _validate_protected_loss(native_receipts: Sequence[Path]) -> None:
    artifacts = _load_native_artifacts(
        native_receipts,
        schemas={
            "plan": wave3.PLAN_SCHEMA,
            "receipt": wave3.RECEIPT_SCHEMA,
        },
    )
    plan = wave3.validate_plan(artifacts["plan"][1])
    validated = wave3.validate_receipt(
        artifacts["receipt"][1], expected_plan=plan
    )
    if validated.get("status") != "passed":
        raise Wave8CoreCellError(
            "protected-loss native receipt did not pass",
            code="wave8_core_cell.native_not_passed",
        )


def _validate_exact_resume(
    plan: Mapping[str, Any], native_receipts: Sequence[Path]
) -> None:
    artifacts = _load_native_artifacts(
        native_receipts,
        schemas={
            "wave7_sequence": wave7_sequence.RECEIPT_SCHEMA,
            "wave7_comparison": wave7_postrun.FINAL_COMPARISON_SCHEMA,
            "wave7_postrun": wave7_postrun.RECEIPT_SCHEMA,
        },
    )
    inputs = plan.get("inputs")
    if not isinstance(inputs, Mapping):
        raise Wave8CoreCellError(
            "matrix plan has no exact-resume input bindings",
            code="wave8_core_cell.plan_binding",
        )
    for name, (path, _) in artifacts.items():
        binding = inputs.get(name)
        if (
            not isinstance(binding, Mapping)
            or Path(str(binding.get("path", ""))).resolve(strict=False) != path
        ):
            raise Wave8CoreCellError(
                "exact-resume native receipt is not bound by the matrix plan",
                code="wave8_core_cell.plan_binding",
            )
    result = matrix._validate_wave7_evidence(
        wave7_root=Path(str(plan["wave7_root"])),
        sequence_path=artifacts["wave7_sequence"][0],
        comparison_path=artifacts["wave7_comparison"][0],
        postrun_path=artifacts["wave7_postrun"][0],
    )
    if result.get("status") != "passed":
        raise Wave8CoreCellError(
            "exact-resume native receipts did not pass",
            code="wave8_core_cell.native_not_passed",
        )


def _validate_native(
    *, plan: Mapping[str, Any], cell: str, native_receipts: Sequence[Path]
) -> None:
    if cell == "packed_parity_all_layer_fa2":
        _validate_packed(native_receipts)
    elif cell == "protected_loss":
        _validate_protected_loss(native_receipts)
    elif cell == "exact_resume":
        _validate_exact_resume(plan, native_receipts)
    else:
        raise Wave8CoreCellError(
            "unsupported Wave 8 core cell",
            code="wave8_core_cell.unsupported_cell",
        )


def publish_cell(
    *, plan_path: Path, cell: str, native_receipts: Sequence[Path]
) -> dict[str, Any]:
    """Validate one native evidence set and publish its absent-only cell."""

    if cell not in CELL_KINDS:
        raise Wave8CoreCellError(
            "unsupported Wave 8 core cell",
            code="wave8_core_cell.unsupported_cell",
        )
    plan = matrix.validate_plan(Path(plan_path))
    cells = plan.get("cells")
    if not isinstance(cells, Mapping) or not isinstance(cells.get(cell), Mapping):
        raise Wave8CoreCellError(
            "matrix plan does not own the requested cell",
            code="wave8_core_cell.plan_cell",
        )
    cell_spec = cells[cell]
    if cell_spec.get("schema") != matrix.CELL_SCHEMAS[cell]:
        raise Wave8CoreCellError(
            "matrix plan cell schema drifted",
            code="wave8_core_cell.plan_cell",
        )
    target = Path(str(cell_spec.get("path", ""))).resolve(strict=False)
    matrix.assert_absent_artifact_target(target)
    _validate_native(
        plan=plan,
        cell=cell,
        native_receipts=tuple(Path(path) for path in native_receipts),
    )
    body = {
        "schema": matrix.CELL_SCHEMAS[cell],
        "status": "passed",
        "cell": cell,
        "plan_payload_sha256": plan["plan_payload_sha256"],
        "baseline_sha256": matrix.PINNED_BASELINE_SHA256,
        "identity_bundle_sha256": plan["identity_bundle_sha256"],
        "checks": {check: True for check in matrix.CELL_CHECKS[cell]},
        "claim_scope": matrix.NONPERFORMANCE_CLAIM,
    }
    receipt = matrix._signed(body, hash_field="receipt_payload_sha256")
    matrix._publish_absent(target, receipt)
    matrix._validate_cell(plan, cell)
    return receipt


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--cell", choices=CELL_KINDS, required=True)
    parser.add_argument(
        "--native-receipt",
        type=Path,
        action="append",
        required=True,
        help="native plan or terminal receipt path; repeat for the exact inventory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        receipt = publish_cell(
            plan_path=args.plan,
            cell=args.cell,
            native_receipts=args.native_receipt,
        )
    except Exception as exc:
        error = {
            "status": "failed",
            "error_code": str(getattr(exc, "code", type(exc).__name__)),
            "error_type": type(exc).__name__,
        }
        sys.stderr.buffer.write(canonical_json_bytes(error) + b"\n")
        return 2
    sys.stdout.buffer.write(canonical_json_bytes(receipt) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
