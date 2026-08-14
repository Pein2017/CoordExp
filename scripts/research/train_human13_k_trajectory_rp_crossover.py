#!/usr/bin/env python3
"""Fail-closed cell and acquisition-node runner for the Human13 RP screen."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import importlib
import json
from pathlib import Path
import sys
from typing import Any, Protocol, TextIO, cast

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.human13_rp_crossover_matrix_contracts import (
    ARM_IDS,
    DRY_RUN_COUNTER_KEYS,
    EVALUATION_RPS,
    PHASE_MATRIX,
    PHASE_QUALIFICATION,
    AcquisitionKey,
    CellKey,
    CellReceipt,
    CellSpec,
    NodeTerminalReceipt,
    validate_node_cell_specs,
)
from scripts.research.human13_rp_crossover_runtime import (
    CellRuntimeServices,
    run_cell,
)
from scripts.research import launch_human13_k_trajectory_rp_crossover as launcher


NODE_DRY_RUN_SCHEMA = "human13_rp_crossover_node_dry_run.v1"
NODE_RUNTIME_FACTORY_CONTRACT = "human13_rp_crossover_node_runtime_factory.v1"
FACTORY_STATUS_ABSENT = "absent"
FACTORY_STATUS_DECLARED = "factory_declared"
FACTORY_STATUS_READY = "execution_ready"
_COMPONENTS_BY_ARM = {
    "A": ("trajectory",),
    "B": ("trajectory", "compiler"),
    "C": ("trajectory", "compiler", "preservation"),
}


class NodeRuntimeServices(Protocol):
    """Injected production owner for one acquisition and its independent cells."""

    def acquire_cell_specs(
        self, node_plan: Mapping[str, Any]
    ) -> Sequence[CellSpec]: ...

    def services_for_cell(self, spec: CellSpec) -> CellRuntimeServices: ...

    def write_cell_receipt(self, receipt: CellReceipt) -> None: ...

    def write_node_terminal_receipt(
        self, path: str, payload: Mapping[str, Any]
    ) -> None: ...


NodeRuntimeFactory = Callable[[Mapping[str, Any]], NodeRuntimeServices]


def node_runtime_factory_plan_contract() -> dict[str, Any]:
    """The frozen, runtime-free contract every node runtime factory declares.

    Reading it must not load a model, allocate a GPU, or touch an artifact: it
    is the cheapest honest evidence that a ``module:callable`` reference is a
    real node runtime factory rather than a syntactically valid placeholder.
    """

    return {
        "schema_version": NODE_RUNTIME_FACTORY_CONTRACT,
        "arms": list(ARM_IDS),
        "evaluation_rps": list(EVALUATION_RPS),
        "max_updates": 1,
        "retry_policy": "none",
        "world_size": 1,
        "requires_user_model_gpu_authority": True,
        "actions": {key: 0 for key in DRY_RUN_COUNTER_KEYS},
    }


def declare_node_runtime_factory(factory: Any) -> Any:
    """Attach the frozen node runtime factory contract to a real factory."""

    if not callable(factory):
        raise TypeError("a node runtime factory must be callable")
    factory.node_runtime_factory_contract = NODE_RUNTIME_FACTORY_CONTRACT
    factory.plan_contract = node_runtime_factory_plan_contract
    return factory


def validate_node_runtime_factory(factory: Any) -> NodeRuntimeFactory:
    """Fail closed unless the factory declares the frozen runtime-free contract."""

    if not callable(factory):
        raise TypeError("runtime_factory reference is not callable")
    if (
        getattr(factory, "node_runtime_factory_contract", None)
        != NODE_RUNTIME_FACTORY_CONTRACT
    ):
        raise ValueError(
            "runtime factory does not declare the frozen node runtime contract"
        )
    plan_contract = getattr(factory, "plan_contract", None)
    if not callable(plan_contract):
        raise ValueError(
            "runtime factory must expose a runtime-free plan_contract() callable"
        )
    declared = plan_contract()
    if not isinstance(declared, Mapping) or dict(declared) != (
        node_runtime_factory_plan_contract()
    ):
        raise ValueError(
            "runtime factory plan contract differs from the frozen node contract"
        )
    return cast(NodeRuntimeFactory, factory)


def inspect_runtime_factory(reference: str | None) -> dict[str, Any]:
    """Report whether a factory reference is only declared or truly importable."""

    if reference is None:
        return {"reference": None, "status": FACTORY_STATUS_ABSENT, "detail": None}
    try:
        _resolve_runtime_factory(reference)
    except Exception as error:  # a placeholder reference is never ready
        return {
            "reference": reference,
            "status": FACTORY_STATUS_DECLARED,
            "detail": f"{type(error).__name__}: {error}",
        }
    return {"reference": reference, "status": FACTORY_STATUS_READY, "detail": None}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    entry = parser.add_mutually_exclusive_group(required=True)
    entry.add_argument("--cell-spec", type=Path)
    entry.add_argument("--dag-plan", type=Path)
    parser.add_argument("--node-id")
    parser.add_argument("--receipt-path")
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--runtime-factory")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--user-model-gpu-authority", action="store_true")
    return parser


def _load_spec(path: Path) -> CellSpec:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return CellSpec.from_dict(payload)


def _load_mapping(path: Path, *, label: str) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must contain one JSON object")
    return payload


def build_dry_run_plan(spec: CellSpec) -> dict[str, object]:
    """Describe one requested cell without loading a model or creating outputs."""

    return {
        "schema_version": "human13_rp_crossover_cell_dry_run.v1",
        "mode": "dry_run",
        "cell_spec_sha256": spec.content_sha256,
        "cell_key": spec.cell_key.to_dict(),
        "execution_ready": False,
        "requires_explicit_model_gpu_authority": True,
        "actions": {key: 0 for key in DRY_RUN_COUNTER_KEYS},
    }


def _validate_node_plan(
    dag_plan: Mapping[str, Any],
    *,
    node_id: str | None,
    receipt_path: str | None,
    repo_root: Path | None,
    max_updates: int | None,
) -> Mapping[str, Any]:
    if dag_plan.get("schema_version") != launcher.DAG_PLAN_SCHEMA:
        raise ValueError("dag plan schema_version differs")
    if dag_plan.get("unit_id") != launcher.UNIT_ID:
        raise ValueError("dag plan unit_id differs")
    if dag_plan.get("mode") != "dry_run" or dag_plan.get("actions") != {
        key: 0 for key in DRY_RUN_COUNTER_KEYS
    }:
        raise ValueError("node runner requires a sealed zero-action DAG plan")
    if not node_id:
        raise ValueError("node invocation requires --node-id")
    if not receipt_path:
        raise ValueError("node invocation requires --receipt-path")
    if repo_root is None or repo_root.expanduser().resolve(strict=True) != (
        launcher.REPO_ROOT.resolve(strict=True)
    ):
        raise ValueError("node invocation repo root differs from this checkout")
    if max_updates != 1:
        raise ValueError("node invocation max_updates must be exactly 1")

    acquisitions = dag_plan.get("acquisitions")
    if not isinstance(acquisitions, list) or len(acquisitions) != 8:
        raise ValueError("sealed DAG must contain exactly eight acquisition nodes")
    matches = [item for item in acquisitions if item.get("node_id") == node_id]
    if len(matches) != 1:
        raise ValueError("node_id must select exactly one sealed acquisition node")
    node = matches[0]
    if node.get("receipt_path") != receipt_path:
        raise ValueError("receipt_path differs from the selected sealed node")

    acquisition_key = AcquisitionKey.from_dict(node["acquisition_key"])
    if node.get("acquisition_key_sha256") != acquisition_key.content_sha256:
        raise ValueError("selected node acquisition identity differs")
    if (
        tuple(node.get("seeds", ())) != acquisition_key.seeds
        or node.get("training_rp") != acquisition_key.training_rp
    ):
        raise ValueError("selected node sealed seed identity differs")
    phase = node.get("phase")
    expected_arms = ARM_IDS if phase == PHASE_MATRIX else ("C",)
    if phase not in {PHASE_MATRIX, PHASE_QUALIFICATION}:
        raise ValueError("selected node phase differs")
    cells = node.get("cells")
    if not isinstance(cells, list) or tuple(
        cell.get("cell_key", {}).get("arm_id") for cell in cells
    ) != tuple(expected_arms):
        raise ValueError("selected node cell arms differ from its phase contract")
    for cell, arm_id in zip(cells, expected_arms, strict=True):
        cell_key = CellKey.from_dict(cell["cell_key"])
        if cell_key.acquisition_key != acquisition_key or cell_key.arm_id != arm_id:
            raise ValueError("selected node cell identity differs")
        if cell.get("cell_key_sha256") != cell_key.content_sha256:
            raise ValueError("selected node cell digest differs")
        if tuple(cell.get("objective_components", ())) != _COMPONENTS_BY_ARM[arm_id]:
            raise ValueError("selected node objective components differ")
        if not cell.get("adamw_config_sha256") or not cell.get(
            "fresh_optimizer_identity_sha256"
        ):
            raise ValueError("selected node cell optimizer identities are missing")
        if (
            cell.get("shared_evidence_group_sha256") != acquisition_key.content_sha256
            or cell.get("source") != "fresh"
            or cell.get("optimizer") != "fresh_adamw"
            or cell.get("world_size") != 1
            or cell.get("max_updates") != 1
            or cell.get("retry_policy") != "none"
            or tuple(cell.get("evaluation_rps", ())) != EVALUATION_RPS
        ):
            raise ValueError("selected node cell execution contract differs")
    if len({cell["adamw_config_sha256"] for cell in cells}) != 1:
        raise ValueError(
            "selected node cells must bind one declared AdamW configuration identity"
        )
    optimizer_identities = [cell["fresh_optimizer_identity_sha256"] for cell in cells]
    if len(set(optimizer_identities)) != len(optimizer_identities):
        raise ValueError(
            "selected node cells must have an independent fresh optimizer identity"
        )
    return node


def build_node_dry_run_plan(node: Mapping[str, Any]) -> dict[str, object]:
    """Describe one acquisition node without acquiring evidence or writing outputs."""

    return {
        "schema_version": NODE_DRY_RUN_SCHEMA,
        "mode": "dry_run",
        "node_id": node["node_id"],
        "phase": node["phase"],
        "acquisition_key_sha256": node["acquisition_key_sha256"],
        "cells": [
            {
                "cell_key_sha256": cell["cell_key_sha256"],
                "arm_id": cell["cell_key"]["arm_id"],
                "output_root": cell["output_root"],
            }
            for cell in node["cells"]
        ],
        "execution_ready": False,
        "requires_explicit_model_gpu_authority": True,
        "actions": {key: 0 for key in DRY_RUN_COUNTER_KEYS},
    }


def _validate_acquired_specs(
    node: Mapping[str, Any], specs: Sequence[CellSpec]
) -> tuple[CellSpec, ...]:
    """Admit exactly the cells this sealed node planned, with their identities.

    Node-level relations (one shared acquisition evidence, one declared AdamW
    configuration, per-cell optimizer identities, and the nested objective byte
    identities) are delegated to the shared contract validator, so the runner
    and the durable node terminal cannot disagree.
    """

    bound = tuple(specs)
    planned_cells = tuple(node["cells"])
    if len(bound) != len(planned_cells):
        raise ValueError("acquired CellSpecs do not cover the selected node exactly")
    acquisition_key = AcquisitionKey.from_dict(node["acquisition_key"])
    for spec, planned in zip(bound, planned_cells, strict=True):
        if not isinstance(spec, CellSpec) or CellSpec.from_dict(spec.to_dict()) != spec:
            raise ValueError("node runtime must return canonical CellSpec records")
        planned_key = CellKey.from_dict(planned["cell_key"])
        if (
            spec.cell_key != planned_key
            or spec.output_root != planned["output_root"]
            or spec.expected_objective_components
            != tuple(planned["objective_components"])
        ):
            raise ValueError("acquired CellSpec differs from the selected sealed cell")
        if spec.shared_evidence.seeds != acquisition_key.seeds:
            raise ValueError(
                "acquired evidence seeds differ from the sealed acquisition"
            )
        if (
            spec.adamw_config_sha256 != planned["adamw_config_sha256"]
            or spec.fresh_optimizer_identity_sha256
            != planned["fresh_optimizer_identity_sha256"]
        ):
            raise ValueError(
                "acquired CellSpec optimizer identities differ from the sealed cell"
            )
    return validate_node_cell_specs(acquisition_key, node["phase"], bound)


def _resolve_runtime_factory(reference: str) -> NodeRuntimeFactory:
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("runtime_factory must use module:callable syntax")
    factory = getattr(importlib.import_module(module_name), attribute)
    return validate_node_runtime_factory(factory)


def _execute_node(
    node: Mapping[str, Any],
    *,
    receipt_path: str,
    runtime_factory: NodeRuntimeFactory,
) -> dict[str, Any]:
    """Run one acquisition node and persist exactly one typed terminal receipt.

    The terminal carries the exact typed ``CellSpec`` records this node
    acquired next to their ``CellReceipt`` outcomes, so the aggregate publisher
    never has to re-derive what was actually run.
    """

    runtime = runtime_factory(node)
    specs: tuple[CellSpec, ...] = ()
    receipts: list[CellReceipt] = []
    failure: BaseException | None = None
    try:
        specs = _validate_acquired_specs(node, runtime.acquire_cell_specs(node))
        services_seen: list[CellRuntimeServices] = []
        for spec in specs:
            services = runtime.services_for_cell(spec)
            if any(services is previous for previous in services_seen):
                raise ValueError("every cell requires independent runtime services")
            services_seen.append(services)
            receipt = run_cell(
                spec,
                services=services,
                receipt_writer=runtime.write_cell_receipt,
            )
            receipts.append(receipt)
    except BaseException as error:
        failure = error

    terminal = NodeTerminalReceipt(
        node_id=node["node_id"],
        phase=node["phase"],
        acquisition_key=AcquisitionKey.from_dict(node["acquisition_key"]),
        status="failed" if failure is not None else "succeeded",
        cell_specs=specs,
        cell_receipts=tuple(receipts),
        failure_reason=(
            None if failure is None else f"{type(failure).__name__}: {failure}"
        ),
    ).to_dict()
    runtime.write_node_terminal_receipt(receipt_path, terminal)
    if failure is not None:
        raise failure
    return terminal


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    services_factory: Callable[[CellSpec], CellRuntimeServices] | None = None,
    receipt_writer: Callable[[CellReceipt], None] | None = None,
    node_runtime_factory: NodeRuntimeFactory | None = None,
    stdout: TextIO | None = None,
) -> int:
    """Run a zero-action plan or one explicitly authorized injected runtime."""

    args = _parser().parse_args(argv)
    output = sys.stdout if stdout is None else stdout
    if args.cell_spec is not None:
        if any(
            value is not None
            for value in (
                args.node_id,
                args.receipt_path,
                args.repo_root,
                args.max_updates,
                args.runtime_factory,
            )
        ):
            raise ValueError("single-cell invocation cannot include node-runner flags")
        spec = _load_spec(args.cell_spec)
        if not args.execute:
            payload: Mapping[str, Any] = build_dry_run_plan(spec)
        else:
            if args.user_model_gpu_authority is not True:
                raise PermissionError("execute requires explicit model/GPU authority")
            if services_factory is None or receipt_writer is None:
                raise RuntimeError(
                    "execute requires an injected production runtime adapter and receipt writer"
                )
            services = services_factory(spec)
            payload = run_cell(
                spec, services=services, receipt_writer=receipt_writer
            ).to_dict()
    else:
        dag_plan = _load_mapping(args.dag_plan, label="dag plan")
        node = _validate_node_plan(
            dag_plan,
            node_id=args.node_id,
            receipt_path=args.receipt_path,
            repo_root=args.repo_root,
            max_updates=args.max_updates,
        )
        if not args.execute:
            payload = build_node_dry_run_plan(node)
        else:
            if args.user_model_gpu_authority is not True:
                raise PermissionError("execute requires explicit model/GPU authority")
            if node_runtime_factory is not None and args.runtime_factory is not None:
                raise ValueError("runtime factory was provided twice")
            factory = (
                None
                if node_runtime_factory is None
                else validate_node_runtime_factory(node_runtime_factory)
            )
            if factory is None and args.runtime_factory is not None:
                factory = _resolve_runtime_factory(args.runtime_factory)
            if factory is None:
                raise RuntimeError("execute requires a production node runtime factory")
            payload = _execute_node(
                node, receipt_path=args.receipt_path, runtime_factory=factory
            )

    json.dump(payload, output, sort_keys=True)
    output.write("\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
