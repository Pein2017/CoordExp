#!/usr/bin/env python3
"""Bounded transactional controller for the two Human-13 on-policy arms.

The module is CPU-safe until a caller supplies an explicit live runtime and
execution authority.  The runtime is the narrow composition seam for the
existing model, packed payload, HF scoring/decode, checkpoint, and analyzer
adapters; this controller owns only iteration and transaction ordering.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass, replace
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Protocol, Sequence, cast

import yaml


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from scripts.research.build_human13_on_policy_frontier import (  # noqa: E402
    CheckpointIdentity,
    Human13FrontierIteration,
)
from scripts.research.human13_frontier_selection import (  # noqa: E402
    CandidateScore,
)
from scripts.research.human13_on_policy_live import (  # noqa: E402
    BehaviorGateObservation,
    BehaviorGateVerdict,
    evaluate_behavior_gate,
)
from scripts.research.human13_proposal_checkpoint import (  # noqa: E402
    private_proposal_checkpoint,
)


SCHEMA_VERSION = "human13_on_policy_arm.v1"
RECEIPT_SCHEMA_VERSION = "human13_on_policy_runtime.v1"
UNIT_ID = "2026-08-13-human13-on-policy-first-bottleneck-successor"
CONFIG_ROOT = Path("configs/coordexp_swift/research/human13_on_policy_successor")
ARM_FILES = ("01_o_full_safe.yaml", "02_o_first_safe.yaml")
ARM_IDS = ("O-Full-Safe", "O-First-Safe")
SOURCE_CHECKPOINT_PATH = (
    "/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/"
    "2026-08-05-closeout/artifacts/training/four-coordinate-xy/"
    "checkpoints/step-2444"
)
SOURCE_INFER_CONFIG = (
    "configs/coordexp_swift/infer/"
    "qwen3_vl_2b_static_dynamic_owner_interface_s_step2444_h0.yaml"
)
MANIFEST_SHA256 = "a8f88716c1227054ab29dc698f89462c9369c47c8d6415de3783c0937f60a6fb"
DECISION_SURFACE = "hf_fp32_sdpa_batch1"
ZERO_ACTIONS = {
    "model_loads": 0,
    "forwards": 0,
    "backwards": 0,
    "optimizer_steps": 0,
    "clean_decodes": 0,
    "checkpoint_writes": 0,
    "gpu_allocations": 0,
}


class OnPolicySuccessorError(RuntimeError):
    """Raised before an unbound or semantically invalid live mutation."""


@dataclass(frozen=True)
class OnPolicyArmConfig:
    schema_version: str
    unit_id: str
    arm_id: str
    arm_name: str
    updates: bool
    base_arm_config: str
    manifest_path: str
    manifest_sha256: str
    source_trajectory_path: str
    k_trajectory_path: str
    source_infer_config: str
    output_root: str
    world_size: int
    max_attempts: int
    one_update_per_ledger: bool
    fresh_source_each_arm: bool
    fresh_optimizer_each_arm: bool
    decision_surface: str
    shortlist_min: int
    shortlist_max: int
    required_margin: float
    rectangle_margin: float
    duplicate_margin: float
    global_max_length: int
    final_repetition_penalty: float
    refresh_after_accepted: int
    refresh_batch_size: int
    refresh_repetition_penalty: float


@dataclass(frozen=True)
class FrontierHandle:
    artifact_path: str
    artifact_sha256: str
    frontier: Human13FrontierIteration


@dataclass(frozen=True)
class RuntimeIdentity:
    source_checkpoint_path: str
    world_size: int
    optimizer_update_count: int


@dataclass(frozen=True)
class CandidateSelectionReceipt:
    decision_surface: str
    eligible_owner_count: int
    shortlisted: tuple[CandidateScore, ...]
    forced_continuation_count: int
    selected: CandidateScore


@dataclass(frozen=True)
class OneUpdateReceipt:
    ledger_sha256: str
    applied_optimizer_updates: int
    finite: bool


@dataclass(frozen=True)
class PrivateProposal:
    proposal_id: str
    checkpoint: CheckpointIdentity
    source_ledger_sha256: str


@dataclass(frozen=True)
class CleanDecodeReceipt:
    decision_surface: str
    checkpoint: CheckpointIdentity
    artifact_path: str
    artifact_sha256: str
    observation: BehaviorGateObservation


@dataclass(frozen=True)
class AttemptReceipt:
    attempt_index: int
    ledger_path: str
    ledger_sha256: str
    selected_owner_id: str
    selected_alias_id: str
    proposal_checkpoint_path: str
    proposed_decode_path: str
    decision: str
    decision_reasons: tuple[str, ...]
    transaction_before_sha256: str
    transaction_after_sha256: str
    accepted_checkpoint_path: str | None
    next_ledger_path: str | None
    rollback_decode_path: str | None


@dataclass(frozen=True)
class OneIterationResult:
    receipt: AttemptReceipt
    next_frontier: FrontierHandle | None
    accepted: bool


@dataclass(frozen=True)
class LoopReceipt:
    schema_version: str
    unit_id: str
    arm_id: str
    output_root: str
    stop_reason: str
    attempted_update_count: int
    accepted_update_count: int
    final_frontier_path: str
    final_checkpoint_path: str
    attempts: tuple[AttemptReceipt, ...]


class OnPolicyRuntime(Protocol):
    """Live composition seam; implementations reuse the existing model spine."""

    transaction: Any

    def execution_identity(self) -> RuntimeIdentity: ...

    def initial_frontier(self, config: OnPolicyArmConfig) -> FrontierHandle: ...

    def frontier_observation(
        self, frontier: FrontierHandle
    ) -> BehaviorGateObservation: ...

    def select_candidate(
        self, config: OnPolicyArmConfig, frontier: FrontierHandle
    ) -> CandidateSelectionReceipt | None: ...

    def apply_one_update(
        self,
        config: OnPolicyArmConfig,
        frontier: FrontierHandle,
        selection: CandidateSelectionReceipt,
    ) -> OneUpdateReceipt: ...

    def write_private_proposal(
        self,
        config: OnPolicyArmConfig,
        frontier: FrontierHandle,
        selection: CandidateSelectionReceipt,
        update: OneUpdateReceipt,
        *,
        proposal_run_dir: Path,
    ) -> CheckpointIdentity: ...

    def clean_decode(
        self,
        config: OnPolicyArmConfig,
        checkpoint: CheckpointIdentity,
        *,
        purpose: str,
    ) -> CleanDecodeReceipt: ...

    def publish_accepted_checkpoint(
        self,
        config: OnPolicyArmConfig,
        frontier: FrontierHandle,
        proposal: PrivateProposal,
    ) -> CheckpointIdentity: ...

    def materialize_next_frontier(
        self,
        config: OnPolicyArmConfig,
        prior: FrontierHandle,
        proposed_decode: CleanDecodeReceipt,
        accepted_checkpoint: CheckpointIdentity,
    ) -> FrontierHandle: ...

    def write_attempt_receipt(
        self, config: OnPolicyArmConfig, receipt: AttemptReceipt
    ) -> None: ...


RuntimeFactory = Callable[[OnPolicyArmConfig, Path], OnPolicyRuntime]


def load_on_policy_config(
    path: str | Path, *, repo_root: str | Path
) -> OnPolicyArmConfig:
    target = Path(path).expanduser().resolve(strict=True)
    raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise OnPolicySuccessorError("on-policy config must be an object")
    fields = tuple(OnPolicyArmConfig.__dataclass_fields__)
    if set(raw) != set(fields):
        raise OnPolicySuccessorError("on-policy config fields differ")
    config = OnPolicyArmConfig(
        schema_version=str(raw["schema_version"]),
        unit_id=str(raw["unit_id"]),
        arm_id=str(raw["arm_id"]),
        arm_name=str(raw["arm_name"]),
        updates=bool(raw["updates"]),
        base_arm_config=str(raw["base_arm_config"]),
        manifest_path=str(raw["manifest_path"]),
        manifest_sha256=str(raw["manifest_sha256"]),
        source_trajectory_path=str(raw["source_trajectory_path"]),
        k_trajectory_path=str(raw["k_trajectory_path"]),
        source_infer_config=str(raw["source_infer_config"]),
        output_root=str(raw["output_root"]),
        world_size=int(raw["world_size"]),
        max_attempts=int(raw["max_attempts"]),
        one_update_per_ledger=bool(raw["one_update_per_ledger"]),
        fresh_source_each_arm=bool(raw["fresh_source_each_arm"]),
        fresh_optimizer_each_arm=bool(raw["fresh_optimizer_each_arm"]),
        decision_surface=str(raw["decision_surface"]),
        shortlist_min=int(raw["shortlist_min"]),
        shortlist_max=int(raw["shortlist_max"]),
        required_margin=float(raw["required_margin"]),
        rectangle_margin=float(raw["rectangle_margin"]),
        duplicate_margin=float(raw["duplicate_margin"]),
        global_max_length=int(raw["global_max_length"]),
        final_repetition_penalty=float(raw["final_repetition_penalty"]),
        refresh_after_accepted=int(raw["refresh_after_accepted"]),
        refresh_batch_size=int(raw["refresh_batch_size"]),
        refresh_repetition_penalty=float(raw["refresh_repetition_penalty"]),
    )
    _validate_config(config, repo_root=repo_root)
    return config


def materialize_on_policy_plans(
    config_root: str | Path = CONFIG_ROOT, *, repo_root: str | Path
) -> tuple[OnPolicyArmConfig, ...]:
    root = Path(config_root)
    root = root if root.is_absolute() else Path(repo_root) / root
    names = tuple(path.name for path in sorted(root.glob("*.yaml")))
    if names != ARM_FILES:
        raise OnPolicySuccessorError("on-policy plan set must contain exact two arms")
    configs = tuple(
        load_on_policy_config(root / name, repo_root=repo_root) for name in ARM_FILES
    )
    if tuple(config.arm_id for config in configs) != ARM_IDS:
        raise OnPolicySuccessorError("on-policy arms are not in canonical order")
    if len({config.output_root for config in configs}) != 2:
        raise OnPolicySuccessorError("on-policy arms require unique output roots")
    return configs


def successor_model_config(config: OnPolicyArmConfig, *, repo_root: str | Path) -> Any:
    """Project an O arm onto the accepted language-DoRA/AdamW model seam."""

    from scripts.research.materialize_human13_k_union_configs import load_arm_config

    root = Path(repo_root).resolve()
    base_path = Path(config.base_arm_config)
    base_path = base_path if base_path.is_absolute() else root / base_path
    base = load_arm_config(base_path.resolve(strict=True))
    return replace(
        base,
        unit_id=config.unit_id,
        arm_id=config.arm_id,
        arm_name=config.arm_name,
        milestones=tuple(range(config.max_attempts + 1)),
    )


def build_dry_run_receipt(
    config: OnPolicyArmConfig, *, repo_root: str | Path
) -> dict[str, Any]:
    from scripts.research.human13_live_model import (
        build_human13_live_model_plan,
        validate_human13_live_model_plan,
    )

    output = Path(config.output_root).expanduser()
    if output.exists():
        raise OnPolicySuccessorError(f"output root already exists: {output}")
    plan = build_human13_live_model_plan(
        successor_model_config(config, repo_root=repo_root)
    )
    validation = validate_human13_live_model_plan(plan)
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "mode": "dry_run",
        "execution_ready": False,
        "arm_id": config.arm_id,
        "max_attempts": config.max_attempts,
        "output_root": config.output_root,
        "output_root_fresh": True,
        "manifest_sha256": config.manifest_sha256,
        "model_plan": _artifact(plan),
        "source_validation": _artifact(validation),
        "actions": dict(ZERO_ACTIONS),
    }


def run_one_iteration(
    config: OnPolicyArmConfig,
    *,
    runtime: OnPolicyRuntime,
    frontier: FrontierHandle,
    selection: CandidateSelectionReceipt,
    attempt_index: int,
    force_reject: bool = False,
) -> OneIterationResult:
    """Execute exactly one ledger-bound transactional proposal."""

    _validate_selection(config, frontier, selection)
    prior_observation = runtime.frontier_observation(frontier)
    snapshot = runtime.transaction.begin()
    transaction_closed = False
    try:
        update = runtime.apply_one_update(config, frontier, selection)
        if (
            update.ledger_sha256 != frontier.artifact_sha256
            or update.applied_optimizer_updates != 1
            or update.finite is not True
        ):
            raise OnPolicySuccessorError(
                "runtime did not apply exactly one finite ledger-bound update"
            )

        proposal_identity: CheckpointIdentity | None = None

        def write_proposal(run_dir: Path) -> str | Path:
            nonlocal proposal_identity
            proposal_identity = runtime.write_private_proposal(
                config,
                frontier,
                selection,
                update,
                proposal_run_dir=run_dir,
            )
            return proposal_identity.path

        proposal_parent = Path(config.output_root) / "private"
        with private_proposal_checkpoint(
            proposal_parent, writer=write_proposal
        ) as checkpoint_path:
            if (
                proposal_identity is None
                or Path(proposal_identity.path) != checkpoint_path
            ):
                raise OnPolicySuccessorError("private proposal identity/path differ")
            proposal = PrivateProposal(
                proposal_id=f"attempt-{attempt_index}",
                checkpoint=proposal_identity,
                source_ledger_sha256=frontier.artifact_sha256,
            )
            proposed_decode = runtime.clean_decode(
                config, proposal.checkpoint, purpose="proposal_gate"
            )
            _validate_clean_decode(config, proposed_decode, proposal.checkpoint)
            gate = evaluate_behavior_gate(
                prior_observation, proposed_decode.observation
            )
            if force_reject:
                gate = BehaviorGateVerdict(
                    False, (*gate.reasons, "forced_rejection_drill")
                )
            if gate.accepted:
                transaction_receipt = runtime.transaction.accept(snapshot)
                transaction_closed = True
                accepted_checkpoint = runtime.publish_accepted_checkpoint(
                    config, frontier, proposal
                )
                if Path(accepted_checkpoint.path) == checkpoint_path:
                    raise OnPolicySuccessorError(
                        "accepted checkpoint cannot remain the private proposal"
                    )
                next_frontier = runtime.materialize_next_frontier(
                    config, frontier, proposed_decode, accepted_checkpoint
                )
                _validate_next_frontier(frontier, next_frontier, accepted_checkpoint)
                receipt = AttemptReceipt(
                    attempt_index=attempt_index,
                    ledger_path=frontier.artifact_path,
                    ledger_sha256=frontier.artifact_sha256,
                    selected_owner_id=selection.selected.path.owner_id,
                    selected_alias_id=selection.selected.path.alias_id,
                    proposal_checkpoint_path=proposal.checkpoint.path,
                    proposed_decode_path=proposed_decode.artifact_path,
                    decision="accepted",
                    decision_reasons=(),
                    transaction_before_sha256=transaction_receipt.before_state_digest,
                    transaction_after_sha256=transaction_receipt.after_state_digest,
                    accepted_checkpoint_path=accepted_checkpoint.path,
                    next_ledger_path=next_frontier.artifact_path,
                    rollback_decode_path=None,
                )
            else:
                transaction_receipt = runtime.transaction.reject(snapshot)
                transaction_closed = True
                receipt = AttemptReceipt(
                    attempt_index=attempt_index,
                    ledger_path=frontier.artifact_path,
                    ledger_sha256=frontier.artifact_sha256,
                    selected_owner_id=selection.selected.path.owner_id,
                    selected_alias_id=selection.selected.path.alias_id,
                    proposal_checkpoint_path=proposal.checkpoint.path,
                    proposed_decode_path=proposed_decode.artifact_path,
                    decision="rejected",
                    decision_reasons=gate.reasons,
                    transaction_before_sha256=transaction_receipt.before_state_digest,
                    transaction_after_sha256=transaction_receipt.after_state_digest,
                    accepted_checkpoint_path=None,
                    next_ledger_path=None,
                    rollback_decode_path=None,
                )
                next_frontier = None
        if receipt.decision == "rejected":
            rollback = runtime.clean_decode(
                config, frontier.frontier.checkpoint, purpose="rollback_reproduction"
            )
            _validate_clean_decode(config, rollback, frontier.frontier.checkpoint)
            receipt = replace(receipt, rollback_decode_path=rollback.artifact_path)
            if set(rollback.observation.unique_owner_ids) != set(
                prior_observation.unique_owner_ids
            ):
                receipt = replace(
                    receipt,
                    decision="rollback_failed",
                    decision_reasons=(
                        *receipt.decision_reasons,
                        "rollback_owner_non_reproduction",
                    ),
                )
                runtime.write_attempt_receipt(config, receipt)
                raise OnPolicySuccessorError(
                    "rollback clean decode did not reproduce the prior owner set"
                )
        runtime.write_attempt_receipt(config, receipt)
        return OneIterationResult(
            receipt=receipt,
            next_frontier=next_frontier,
            accepted=receipt.decision == "accepted",
        )
    except BaseException:
        if not transaction_closed:
            runtime.transaction.reject(snapshot)
        raise


def run_on_policy_loop(
    config: OnPolicyArmConfig,
    *,
    runtime: OnPolicyRuntime | None,
    execute_authorized: bool,
    force_reject_attempt: int | None = None,
) -> LoopReceipt:
    """Run the bounded loop through an injected production-shaped runtime."""

    if execute_authorized is not True:
        raise OnPolicySuccessorError("execution requires explicit model/GPU authority")
    if runtime is None:
        raise OnPolicySuccessorError("execution requires an injected live runtime")
    output = Path(config.output_root).expanduser()
    if output.exists():
        raise OnPolicySuccessorError(f"output root already exists: {output}")
    identity = runtime.execution_identity()
    if (
        identity.source_checkpoint_path != SOURCE_CHECKPOINT_PATH
        or identity.world_size != 1
        or identity.optimizer_update_count != 0
    ):
        raise OnPolicySuccessorError(
            "runtime is not a fresh Source/world-one/fresh-optimizer assembly"
        )
    frontier = runtime.initial_frontier(config)
    if (
        frontier.frontier.iteration != 0
        or frontier.frontier.checkpoint.path != SOURCE_CHECKPOINT_PATH
    ):
        raise OnPolicySuccessorError("initial frontier is not bound to fresh Source")

    attempts: list[AttemptReceipt] = []
    accepted = 0
    updated_ledgers: set[str] = set()
    stop_reason = "attempt_cap"
    for attempt_index in range(config.max_attempts):
        if frontier.artifact_sha256 in updated_ledgers:
            raise OnPolicySuccessorError("one ledger cannot drive more than one update")
        selection = runtime.select_candidate(config, frontier)
        if selection is None:
            stop_reason = "no_candidate"
            break
        updated_ledgers.add(frontier.artifact_sha256)
        result = run_one_iteration(
            config,
            runtime=runtime,
            frontier=frontier,
            selection=selection,
            attempt_index=attempt_index,
            force_reject=force_reject_attempt == attempt_index,
        )
        attempts.append(result.receipt)
        if not result.accepted:
            stop_reason = "scientific_rejection"
            break
        if result.next_frontier is None:
            raise OnPolicySuccessorError("accepted update lacks its next frontier")
        frontier = result.next_frontier
        accepted += 1
    return LoopReceipt(
        schema_version=RECEIPT_SCHEMA_VERSION,
        unit_id=config.unit_id,
        arm_id=config.arm_id,
        output_root=config.output_root,
        stop_reason=stop_reason,
        attempted_update_count=len(attempts),
        accepted_update_count=accepted,
        final_frontier_path=frontier.artifact_path,
        final_checkpoint_path=frontier.frontier.checkpoint.path,
        attempts=tuple(attempts),
    )


def execute_cli(
    *,
    config_path: str | Path,
    repo_root: str | Path,
    execute: bool,
    authority: bool,
    runtime: OnPolicyRuntime | None = None,
    runtime_factory: RuntimeFactory | None = None,
    force_reject_attempt: int | None = None,
) -> Mapping[str, Any]:
    """Dry-run by default; compose a real runtime only after explicit authority."""

    root = Path(repo_root).resolve()
    config = load_on_policy_config(config_path, repo_root=root)
    if not execute:
        return build_dry_run_receipt(config, repo_root=root)
    if authority is not True:
        raise OnPolicySuccessorError("execution requires explicit model/GPU authority")
    if runtime is not None and runtime_factory is not None:
        raise OnPolicySuccessorError("provide a runtime or runtime factory, not both")
    live_runtime = runtime
    if live_runtime is None and runtime_factory is not None:
        live_runtime = runtime_factory(config, root)
    return _artifact(
        run_on_policy_loop(
            config,
            runtime=live_runtime,
            execute_authorized=True,
            force_reject_attempt=force_reject_attempt,
        )
    )


def _validate_config(config: OnPolicyArmConfig, *, repo_root: str | Path) -> None:
    if (
        config.schema_version != SCHEMA_VERSION
        or config.unit_id != UNIT_ID
        or config.arm_id not in ARM_IDS
        or config.updates is not True
        or config.manifest_sha256 != MANIFEST_SHA256
        or config.source_infer_config != SOURCE_INFER_CONFIG
        or config.world_size != 1
        or config.max_attempts != 8
        or config.one_update_per_ledger is not True
        or config.fresh_source_each_arm is not True
        or config.fresh_optimizer_each_arm is not True
        or config.decision_surface != DECISION_SURFACE
        or (config.shortlist_min, config.shortlist_max) != (2, 4)
        or config.global_max_length != 12_000
        or config.final_repetition_penalty != 1.0
        or config.refresh_after_accepted != 2
        or config.refresh_batch_size != 4
        or config.refresh_repetition_penalty != 1.10
    ):
        raise OnPolicySuccessorError("on-policy frozen contract differs")
    if any(
        not math.isfinite(value) or value <= 0
        for value in (
            config.required_margin,
            config.rectangle_margin,
            config.duplicate_margin,
        )
    ):
        raise OnPolicySuccessorError("on-policy margins must be finite and positive")
    root = Path(repo_root).resolve()
    for label, raw_path in (
        ("base arm config", config.base_arm_config),
        ("source inference config", config.source_infer_config),
    ):
        path = Path(raw_path)
        path = path if path.is_absolute() else root / path
        if not path.resolve(strict=True).is_file():
            raise OnPolicySuccessorError(f"{label} is unavailable")
    for label, raw_path in (
        ("manifest", config.manifest_path),
        ("Source trajectory", config.source_trajectory_path),
        ("K trajectory", config.k_trajectory_path),
    ):
        path = Path(raw_path).expanduser().resolve(strict=True)
        if not path.is_file():
            raise OnPolicySuccessorError(f"{label} is unavailable")
    if _sha256_file(Path(config.manifest_path)) != config.manifest_sha256:
        raise OnPolicySuccessorError("manifest SHA-256 drifted")
    if not Path(config.output_root).expanduser().is_absolute():
        raise OnPolicySuccessorError("output root must be absolute")


def _validate_selection(
    config: OnPolicyArmConfig,
    frontier: FrontierHandle,
    selection: CandidateSelectionReceipt,
) -> None:
    if (
        selection.decision_surface != DECISION_SURFACE
        or not config.shortlist_min
        <= len(selection.shortlisted)
        <= config.shortlist_max
        or selection.forced_continuation_count != len(selection.shortlisted)
        or selection.selected not in selection.shortlisted
        or selection.eligible_owner_count < len(selection.shortlisted)
        or len({item.path.owner_id for item in selection.shortlisted})
        != len(selection.shortlisted)
    ):
        raise OnPolicySuccessorError(
            "candidate selection receipt is not decision-grade"
        )
    if (
        any(
            item.path.image_id
            not in {image.image_id for image in frontier.frontier.images}
            for item in selection.shortlisted
        )
        and frontier.frontier.images
    ):
        raise OnPolicySuccessorError(
            "candidate selection is outside the current frontier"
        )


def _validate_clean_decode(
    config: OnPolicyArmConfig,
    receipt: CleanDecodeReceipt,
    checkpoint: CheckpointIdentity,
) -> None:
    if (
        receipt.decision_surface != config.decision_surface
        or receipt.checkpoint != checkpoint
        or not receipt.artifact_path
        or len(receipt.artifact_sha256) != 64
    ):
        raise OnPolicySuccessorError(
            "clean decode is not bound to the HF decision surface"
        )


def _validate_next_frontier(
    prior: FrontierHandle,
    proposed: FrontierHandle,
    accepted_checkpoint: CheckpointIdentity,
) -> None:
    if (
        proposed.frontier.iteration != prior.frontier.iteration + 1
        or proposed.frontier.checkpoint != accepted_checkpoint
        or proposed.frontier.previous_frontier_sha256 != prior.artifact_sha256
        or proposed.artifact_sha256 == prior.artifact_sha256
    ):
        raise OnPolicySuccessorError(
            "accepted decode did not create the sole next ledger"
        )


def _load_runtime_factory(spec: str) -> RuntimeFactory:
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise OnPolicySuccessorError("runtime factory must use module:callable")
    factory = getattr(importlib.import_module(module_name), attribute, None)
    if not callable(factory):
        raise OnPolicySuccessorError("runtime factory is not callable")
    return cast(RuntimeFactory, factory)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(value: Any) -> dict[str, Any]:
    method = getattr(value, "to_artifact_dict", None)
    if callable(method):
        artifact = method()
        if not isinstance(artifact, Mapping):
            raise OnPolicySuccessorError("artifact projection must be an object")
        return {str(key): item for key, item in artifact.items()}
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(cast(Any, value))
    return {"repr": repr(value)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--user-model-gpu-authority", action="store_true")
    parser.add_argument(
        "--runtime-factory",
        help="production composition hook as importable module:callable",
    )
    parser.add_argument("--force-reject-attempt", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    factory = (
        None
        if args.runtime_factory is None
        else _load_runtime_factory(args.runtime_factory)
    )
    receipt = execute_cli(
        config_path=args.config,
        repo_root=args.repo_root,
        execute=args.execute,
        authority=args.user_model_gpu_authority,
        runtime_factory=factory,
        force_reject_attempt=args.force_reject_attempt,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AttemptReceipt",
    "CandidateSelectionReceipt",
    "CleanDecodeReceipt",
    "FrontierHandle",
    "LoopReceipt",
    "OneIterationResult",
    "OneUpdateReceipt",
    "OnPolicyArmConfig",
    "OnPolicyRuntime",
    "OnPolicySuccessorError",
    "PrivateProposal",
    "RuntimeIdentity",
    "build_dry_run_receipt",
    "execute_cli",
    "load_on_policy_config",
    "materialize_on_policy_plans",
    "run_on_policy_loop",
    "run_one_iteration",
    "successor_model_config",
]
