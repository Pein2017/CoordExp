"""Production composition for the bounded Human-13 on-policy controller.

This module owns no new scoring, loss, parsing, matching, inference, or
checkpoint math.  It binds the existing experiment-local seams into the
``OnPolicyRuntime`` protocol consumed by
``train_human13_on_policy_successor.run_one_iteration``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, is_dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any, Protocol, cast

from scripts.research.build_human13_on_policy_frontier import (
    CheckpointIdentity,
    Human13FrontierIteration,
    build_frontier_iteration,
    canonical_write,
    natural_pre_stop_prefix,
)
from scripts.research.human13_on_policy_live import BehaviorGateObservation
from scripts.research.human13_training_transaction import (
    TrainingStateTransaction,
    UpdateCounter,
)
from scripts.research.train_human13_on_policy_successor import (
    AttemptReceipt,
    CandidateSelectionReceipt,
    CleanDecodeReceipt,
    DurableArtifact,
    FrontierHandle,
    OneUpdateReceipt,
    OnPolicyArmConfig,
    PrivateProposal,
    RuntimeIdentity,
    _canonical_bytes as _controller_canonical_bytes,
    successor_model_config,
)


@dataclass
class RuntimeBuildState:
    repo_root: Path
    output_root: Path
    config_sha256: str
    manifest: Any
    manifest_sha256: str
    assembly: Any
    skeletons: Mapping[int, Any]
    vocab_groups: Any
    transaction: Any
    update_counter: UpdateCounter
    source_checkpoint: CheckpointIdentity
    active_frontier: FrontierHandle | None
    source_records: Mapping[int, Mapping[str, Any]] = field(default_factory=dict)
    continuation_caps: Mapping[int, int] = field(default_factory=dict)
    pending_decodes: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    pending_outputs: dict[str, tuple[Mapping[str, Any], ...]] = field(
        default_factory=dict
    )
    selected_scores: dict[str, Any] = field(default_factory=dict)
    staged_prior_frontier: FrontierHandle | None = None
    decode_serial: int = 0


class RuntimeServices(Protocol):
    def initial_frontier(self, state: RuntimeBuildState) -> FrontierHandle: ...

    def select_candidate(
        self, state: RuntimeBuildState, frontier: FrontierHandle
    ) -> CandidateSelectionReceipt | None: ...

    def apply_one_update(
        self,
        state: RuntimeBuildState,
        frontier: FrontierHandle,
        selection: CandidateSelectionReceipt,
    ) -> OneUpdateReceipt: ...

    def write_private_proposal(
        self,
        state: RuntimeBuildState,
        frontier: FrontierHandle,
        selection: CandidateSelectionReceipt,
        update: OneUpdateReceipt,
        *,
        proposal_run_dir: Path,
    ) -> CheckpointIdentity: ...

    def clean_decode(
        self,
        state: RuntimeBuildState,
        checkpoint: CheckpointIdentity,
        *,
        purpose: str,
    ) -> CleanDecodeReceipt: ...

    def publish_accepted_checkpoint(
        self,
        state: RuntimeBuildState,
        frontier: FrontierHandle,
        proposal: PrivateProposal,
    ) -> CheckpointIdentity: ...

    def materialize_next_frontier(
        self,
        state: RuntimeBuildState,
        prior: FrontierHandle,
        proposed_decode: CleanDecodeReceipt,
        accepted_checkpoint: CheckpointIdentity,
    ) -> FrontierHandle: ...

    def write_attempt_receipt(
        self, state: RuntimeBuildState, receipt: AttemptReceipt
    ) -> DurableArtifact: ...

    def abort_staged_acceptance(
        self,
        state: RuntimeBuildState,
        accepted_checkpoint: CheckpointIdentity | None,
        next_frontier: FrontierHandle | None,
        receipt_artifact: DurableArtifact | None,
    ) -> None: ...


class Human13OnPolicyRuntime:
    """Typed adapter from the controller protocol to production services."""

    def __init__(
        self,
        config: OnPolicyArmConfig,
        state: RuntimeBuildState,
        services: RuntimeServices,
    ) -> None:
        self._config = config
        self._state = state
        self._services = services
        self.transaction = state.transaction

    def _bind(self, config: OnPolicyArmConfig) -> None:
        if config != self._config:
            raise ValueError("runtime received a different arm config")

    def execution_identity(self) -> RuntimeIdentity:
        return RuntimeIdentity(
            source_checkpoint_path=self._state.source_checkpoint.path,
            world_size=int(self._state.assembly.plan.world_size),
            optimizer_update_count=int(self._state.update_counter.value),
        )

    def initial_frontier(self, config: OnPolicyArmConfig) -> FrontierHandle:
        self._bind(config)
        return self._services.initial_frontier(self._state)

    def frontier_observation(self, frontier: FrontierHandle) -> BehaviorGateObservation:
        return _frontier_observation(frontier.frontier, manifest=self._state.manifest)

    def select_candidate(
        self, config: OnPolicyArmConfig, frontier: FrontierHandle
    ) -> CandidateSelectionReceipt | None:
        self._bind(config)
        return self._services.select_candidate(self._state, frontier)

    def apply_one_update(
        self,
        config: OnPolicyArmConfig,
        frontier: FrontierHandle,
        selection: CandidateSelectionReceipt,
    ) -> OneUpdateReceipt:
        self._bind(config)
        return self._services.apply_one_update(self._state, frontier, selection)

    def write_private_proposal(
        self,
        config: OnPolicyArmConfig,
        frontier: FrontierHandle,
        selection: CandidateSelectionReceipt,
        update: OneUpdateReceipt,
        *,
        proposal_run_dir: Path,
    ) -> CheckpointIdentity:
        self._bind(config)
        return self._services.write_private_proposal(
            self._state,
            frontier,
            selection,
            update,
            proposal_run_dir=proposal_run_dir,
        )

    def clean_decode(
        self,
        config: OnPolicyArmConfig,
        checkpoint: CheckpointIdentity,
        *,
        purpose: str,
    ) -> CleanDecodeReceipt:
        self._bind(config)
        return self._services.clean_decode(self._state, checkpoint, purpose=purpose)

    def publish_accepted_checkpoint(
        self,
        config: OnPolicyArmConfig,
        frontier: FrontierHandle,
        proposal: PrivateProposal,
    ) -> CheckpointIdentity:
        self._bind(config)
        return self._services.publish_accepted_checkpoint(
            self._state, frontier, proposal
        )

    def materialize_next_frontier(
        self,
        config: OnPolicyArmConfig,
        prior: FrontierHandle,
        proposed_decode: CleanDecodeReceipt,
        accepted_checkpoint: CheckpointIdentity,
    ) -> FrontierHandle:
        self._bind(config)
        return self._services.materialize_next_frontier(
            self._state, prior, proposed_decode, accepted_checkpoint
        )

    def write_attempt_receipt(
        self, config: OnPolicyArmConfig, receipt: AttemptReceipt
    ) -> DurableArtifact:
        self._bind(config)
        return self._services.write_attempt_receipt(self._state, receipt)

    def abort_staged_acceptance(
        self,
        config: OnPolicyArmConfig,
        accepted_checkpoint: CheckpointIdentity | None,
        next_frontier: FrontierHandle | None,
        receipt_artifact: DurableArtifact | None,
    ) -> None:
        self._bind(config)
        self._services.abort_staged_acceptance(
            self._state,
            accepted_checkpoint,
            next_frontier,
            receipt_artifact,
        )


class ProductionRuntimeServices:
    """Composition-only implementation over existing Human-13 live seams."""

    def __init__(self, config: OnPolicyArmConfig) -> None:
        self._config = config

    def initial_frontier(self, state: RuntimeBuildState) -> FrontierHandle:
        from scripts.research.human13_live_eval import (
            current_decodes_from_outputs,
            source_outputs_from_manifest,
            write_outputs_jsonl,
        )

        if state.active_frontier is not None:
            raise ValueError("initial frontier was already materialized")
        outputs = source_outputs_from_manifest(
            manifest=state.manifest,
            manifest_sha256=state.manifest_sha256,
            source_discovery_records=state.source_records,
            arm_id=self._config.arm_id,
            run_id=f"human13-{self._config.arm_id.lower()}-source",
            run_root=str(state.output_root),
            resolved_arm_plan_sha256=state.config_sha256,
            resolved_config_sha256=state.config_sha256,
            checkpoint_path=state.source_checkpoint.path,
        )
        outputs_path = state.output_root / "decodes" / "source.jsonl"
        write_outputs_jsonl(outputs_path, outputs)
        decodes = current_decodes_from_outputs(
            manifest=state.manifest,
            manifest_sha256=state.manifest_sha256,
            outputs=outputs,
            checkpoint=state.source_checkpoint,
        )
        frontier = build_frontier_iteration(
            state.manifest,
            manifest_path=self._config.manifest_path,
            iteration=0,
            checkpoint=state.source_checkpoint,
            decodes=decodes,
        )
        path = state.output_root / "frontiers" / "iteration-000.json"
        digest = canonical_write(frontier, path)
        handle = FrontierHandle(
            str(path),
            digest,
            frontier,
            source_decode_artifact_sha256=_sha256_file(path=outputs_path),
        )
        state.active_frontier = handle
        return handle

    def select_candidate(
        self, state: RuntimeBuildState, frontier: FrontierHandle
    ) -> CandidateSelectionReceipt | None:
        from scripts.research.human13_continuation_projection import (
            project_continuation,
            select_projected_continuation,
        )
        from scripts.research.human13_forced_continuation import (
            forced_complete_row_then_natural_continuation,
        )
        from scripts.research.human13_hf_census import (
            open_checkpoint_hf_census_scorer,
        )
        from scripts.research.human13_on_policy_scoring import (
            score_on_policy_frontier_candidates,
        )

        _require_active(state, frontier)
        images = {image.image_id: image for image in frontier.frontier.images}
        with open_checkpoint_hf_census_scorer(
            repo_root=state.repo_root,
            checkpoint_path=frontier.frontier.checkpoint.path,
        ) as hf_scorer:
            scored = score_on_policy_frontier_candidates(
                frontier_images=images,
                prompt_skeletons=state.skeletons,
                packed_model=state.assembly.model,
                packed_runtime=state.assembly.runtime,
                tokenizer=state.assembly.components.tokenizer,
                hf_scorer=hf_scorer,
                aliases_per_owner=2,
                shortlist_limit=self._config.shortlist_max,
                global_max_length=self._config.global_max_length,
            )
            shortlist = tuple(
                score
                for image_id in sorted(scored.shortlist_by_image)
                for score in scored.shortlist_by_image[image_id]
            )
            iteration_dir = (
                state.output_root
                / "iterations"
                / f"attempt-{frontier.frontier.iteration:03d}"
            )
            if len(shortlist) < self._config.shortlist_min:
                _write_candidate_receipt(
                    iteration_dir / "candidate-scoring.json",
                    frontier=frontier,
                    scored=scored,
                    projections=(),
                    selected=None,
                )
                return None
            cross_by_path = {
                (
                    item.path.image_id,
                    item.path.owner_id,
                    item.path.alias_id,
                ): item
                for item in scored.cross_surface_receipts
            }
            encoded_by_segment = {
                segment.segment_id: segment.encoded_example
                for segment in scored.prepared.packed_plan.logical_segments
            }
            manifest_images = {
                int(image.image_id): image for image in state.manifest.images
            }
            projections = []
            for score in shortlist:
                key = (score.path.image_id, score.path.owner_id, score.path.alias_id)
                cross = cross_by_path[key]
                encoded = encoded_by_segment[cross.segment_id]
                session, native_inputs = hf_scorer.exact_history_context(encoded)
                native_inputs = _prompt_only_native_inputs(
                    native_inputs, prompt_count=int(encoded.prompt_token_count)
                )
                natural = natural_pre_stop_prefix(images[score.path.image_id])
                skeleton = state.skeletons[score.path.image_id]
                source_trajectory = state.source_records[score.path.image_id][
                    "trajectory"
                ]
                if not isinstance(source_trajectory, Mapping):
                    raise ValueError("Source trajectory must remain a mapping")
                source_rows = source_trajectory["rows"]
                source_token_ids = source_trajectory["token_ids"]
                if not isinstance(source_rows, list) or not isinstance(
                    source_token_ids, list
                ):
                    raise ValueError("Source trajectory shape drifted")
                continuation_cap = state.continuation_caps[score.path.image_id]
                result = forced_complete_row_then_natural_continuation(
                    session=session,
                    native_inputs=native_inputs,
                    natural_prefix_token_ids=natural,
                    forced_row_token_ids=score.path.token_ids,
                    tokenizer=state.assembly.components.tokenizer,
                    image_width=int(skeleton.image_encoding.width),
                    image_height=int(skeleton.image_encoding.height),
                    repetition_penalty=self._config.final_repetition_penalty,
                    continuation_cap=continuation_cap,
                    source_row_count=len(source_rows),
                    source_token_count=len(source_token_ids),
                    current_checkpoint_payload_sha256=(
                        frontier.frontier.checkpoint.payload_sha256
                    ),
                )
                projections.append(
                    project_continuation(
                        manifest_images[score.path.image_id],
                        images[score.path.image_id],
                        score,
                        result,
                        expected_continuation_cap=continuation_cap,
                        expected_repetition_penalty=(
                            self._config.final_repetition_penalty
                        ),
                        current_checkpoint_payload_sha256=(
                            frontier.frontier.checkpoint.payload_sha256
                        ),
                    )
                )
        try:
            selected = select_projected_continuation(projections)
        except ValueError:
            selected = None
        _write_candidate_receipt(
            iteration_dir / "candidate-scoring.json",
            frontier=frontier,
            scored=scored,
            projections=tuple(projections),
            selected=selected,
        )
        if selected is None:
            return None
        state.selected_scores[frontier.artifact_sha256] = selected.score
        return CandidateSelectionReceipt(
            decision_surface=self._config.decision_surface,
            eligible_owner_count=sum(
                len(image.uncovered_h_owner_ids) for image in images.values()
            ),
            shortlisted=shortlist,
            forced_continuation_count=len(projections),
            selected=selected.score,
            continuation_projection_sha256s=tuple(
                item.artifact_sha256 for item in projections
            ),
            continuation_outcomes=tuple(item.outcome for item in projections),
        )

    def apply_one_update(
        self,
        state: RuntimeBuildState,
        frontier: FrontierHandle,
        selection: CandidateSelectionReceipt,
    ) -> OneUpdateReceipt:
        from scripts.research.human13_on_policy_live import (
            OnPolicyLossRunner,
            build_on_policy_payload,
            materialize_on_policy_segments,
            on_policy_loss_context_factory,
        )
        from scripts.research.train_human13_live_arm import build_training_schedule
        from src.training.supervised_trainer import SupervisedTrainer

        _require_active(state, frontier)
        score = state.selected_scores.get(frontier.artifact_sha256)
        if score != selection.selected:
            raise ValueError("selected continuation differs from scored frontier")
        image_id = selection.selected.path.image_id
        image = next(
            item for item in frontier.frontier.images if item.image_id == image_id
        )
        reserve = max(0.0, float(selection.selected.max_surface_margin_drift or 0.0))
        required_margin = self._config.required_margin + reserve
        materialized = materialize_on_policy_segments(
            {image_id: image},
            {image_id: state.skeletons[image_id]},
            selected_scores={image_id: selection.selected},
            arm_id=cast(Any, self._config.arm_id),
            required_margin=required_margin,
            global_max_length=self._config.global_max_length,
        )
        payload = build_on_policy_payload(
            materialized,
            arm_id=cast(Any, self._config.arm_id),
            expected_vocab_size=int(
                state.assembly.components.token_identity.tokenizer_vocab_size
            ),
            vocab_groups=state.vocab_groups,
            global_max_length=self._config.global_max_length,
            required_margin=required_margin,
            rectangle_margin=self._config.rectangle_margin,
            duplicate_margin=self._config.duplicate_margin,
        )
        payload = _attach_on_policy_image_processor(payload, state.assembly.components)
        runner = OnPolicyLossRunner(
            payload.denominators,
            required_margin=required_margin,
            rectangle_margin=self._config.rectangle_margin,
            duplicate_margin=self._config.duplicate_margin,
        )
        before_optimizer = int(state.assembly.runtime.optimizer_step_count)
        before_scheduler = int(state.assembly.runtime.scheduler_step_count)
        trainer = SupervisedTrainer(
            model=state.assembly.model,
            schedule=build_training_schedule(
                pack_count=len(payload.micro_steps), max_updates=1
            ),
            pack_stream=payload.micro_steps,
            loss_context_factory=on_policy_loss_context_factory,
            loss_runner=cast(Any, runner),
            runtime=state.assembly.runtime,
            on_checkpoint=lambda _event, _observation: None,
            on_final=lambda _event, _observation: None,
        )
        result = trainer.run()
        observation = result.latest_observation
        if (
            result.completed_steps != 1
            or observation is None
            or observation.optimizer_update_status != "applied"
            or observation.finite_status != "finite"
            or int(state.assembly.runtime.optimizer_step_count) != before_optimizer + 1
            or int(state.assembly.runtime.scheduler_step_count) != before_scheduler + 1
        ):
            raise RuntimeError(
                "on-policy SupervisedTrainer did not apply one finite update"
            )
        state.update_counter.value += 1
        _write_receipt(
            state.output_root
            / "iterations"
            / f"attempt-{frontier.frontier.iteration:03d}"
            / "one-update.json",
            {
                "schema_version": "human13_on_policy_one_update.v1",
                "frontier_sha256": frontier.artifact_sha256,
                "required_margin": required_margin,
                "candidate_surface_reserve": reserve,
                "payload": {
                    "logical_segment_count": _payload_counts(payload)[0],
                    "physical_pack_count": _payload_counts(payload)[1],
                },
                "training_result": _jsonable(result),
            },
        )
        return OneUpdateReceipt(frontier.artifact_sha256, 1, True)

    def write_private_proposal(
        self,
        state: RuntimeBuildState,
        frontier: FrontierHandle,
        selection: CandidateSelectionReceipt,
        update: OneUpdateReceipt,
        *,
        proposal_run_dir: Path,
    ) -> CheckpointIdentity:
        from scripts.research.human13_live_eval import checkpoint_payload_sha256
        from scripts.research.human13_live_model import (
            build_human13_checkpoint_kwargs,
            build_human13_checkpoint_writer,
            readback_human13_checkpoint,
        )

        del selection
        if update.ledger_sha256 != frontier.artifact_sha256:
            raise ValueError("private proposal update differs from frontier")
        writer = build_human13_checkpoint_writer(proposal_run_dir)
        written = writer.write_checkpoint(
            step=1,
            model=state.assembly.model,
            **build_human13_checkpoint_kwargs(state.assembly),
        )
        readback = readback_human13_checkpoint(
            written.checkpoint_dir, expected_step=1, assembly=state.assembly
        )
        identity = CheckpointIdentity(
            str(written.checkpoint_dir),
            checkpoint_payload_sha256(written.checkpoint_dir),
        )
        _write_receipt(
            state.output_root
            / "iterations"
            / f"attempt-{frontier.frontier.iteration:03d}"
            / "private-proposal.json",
            {
                "schema_version": "human13_on_policy_private_proposal.v1",
                "frontier_sha256": frontier.artifact_sha256,
                "checkpoint": _jsonable(identity),
                "readback": _jsonable(readback),
                "accepted": False,
            },
        )
        return identity

    def clean_decode(
        self,
        state: RuntimeBuildState,
        checkpoint: CheckpointIdentity,
        *,
        purpose: str,
    ) -> CleanDecodeReceipt:
        from scripts.research.human13_live_eval import (
            checkpoint_payload_sha256,
            current_decodes_from_outputs,
            evaluate_hf_checkpoint,
            write_outputs_jsonl,
        )

        active = state.active_frontier
        if active is None:
            raise ValueError("clean decode requires an active accepted frontier")
        if checkpoint_payload_sha256(checkpoint.path) != checkpoint.payload_sha256:
            raise ValueError("clean decode checkpoint payload drifted")
        serial = state.decode_serial
        state.decode_serial += 1
        outputs = evaluate_hf_checkpoint(
            manifest=state.manifest,
            manifest_sha256=state.manifest_sha256,
            checkpoint_path=checkpoint.path,
            arm_id=self._config.arm_id,
            milestone=active.frontier.iteration + 1,
            run_id=f"human13-{self._config.arm_id.lower()}",
            run_root=str(state.output_root),
            resolved_arm_plan_sha256=state.config_sha256,
            resolved_config_sha256=state.config_sha256,
            source_config_path=state.repo_root / self._config.source_infer_config,
        )
        path = (
            state.output_root
            / "decodes"
            / f"{serial:03d}-{purpose}-after-{active.frontier.iteration:03d}.jsonl"
        )
        write_outputs_jsonl(path, outputs)
        decodes = current_decodes_from_outputs(
            manifest=state.manifest,
            manifest_sha256=state.manifest_sha256,
            outputs=outputs,
            checkpoint=checkpoint,
        )
        prospective = build_frontier_iteration(
            state.manifest,
            manifest_path=self._config.manifest_path,
            iteration=active.frontier.iteration + 1,
            checkpoint=checkpoint,
            decodes=decodes,
            previous=active.frontier,
            previous_path=active.artifact_path,
        )
        observation = _frontier_observation(prospective, manifest=state.manifest)
        cap_hits = sum(
            str(item.get("stop_reason")) in {"length", "max_tokens", "cap_hit"}
            for item in outputs
        )
        observation = replace(
            observation,
            cap_hit_count=int(cap_hits),
        )
        state.pending_decodes[checkpoint.path] = tuple(decodes)
        state.pending_outputs[checkpoint.path] = tuple(outputs)
        return CleanDecodeReceipt(
            decision_surface=self._config.decision_surface,
            checkpoint=checkpoint,
            artifact_path=str(path),
            artifact_sha256=_sha256_file(path),
            observation=observation,
            current_decodes=tuple(decodes),
        )

    def publish_accepted_checkpoint(
        self,
        state: RuntimeBuildState,
        frontier: FrontierHandle,
        proposal: PrivateProposal,
    ) -> CheckpointIdentity:
        from scripts.research.human13_live_eval import checkpoint_payload_sha256
        from scripts.research.human13_live_model import readback_human13_checkpoint
        from scripts.research.human13_proposal_checkpoint import (
            promote_private_proposal_checkpoint,
        )

        _require_active(state, frontier)
        state.staged_prior_frontier = state.active_frontier
        step = frontier.frontier.iteration + 1
        target = promote_private_proposal_checkpoint(
            proposal.checkpoint.path,
            accepted_run_dir=state.output_root / "accepted",
            accepted_step=step,
        )
        try:
            readback_human13_checkpoint(
                target, expected_step=step, assembly=state.assembly
            )
            digest = checkpoint_payload_sha256(target)
            if digest != proposal.checkpoint.payload_sha256:
                raise RuntimeError(
                    "accepted checkpoint bytes differ from private proposal"
                )
        except BaseException:
            shutil.rmtree(target)
            raise
        identity = CheckpointIdentity(str(target), digest)
        return identity

    def materialize_next_frontier(
        self,
        state: RuntimeBuildState,
        prior: FrontierHandle,
        proposed_decode: CleanDecodeReceipt,
        accepted_checkpoint: CheckpointIdentity,
    ) -> FrontierHandle:
        _require_active(state, prior)
        if (
            proposed_decode.checkpoint.payload_sha256
            != accepted_checkpoint.payload_sha256
        ):
            raise ValueError("accepted checkpoint differs from gated proposal bytes")
        decodes = proposed_decode.current_decodes
        if state.pending_decodes.get(proposed_decode.checkpoint.path) != decodes:
            raise ValueError(
                "accepted proposal differs from its same-output decode cache"
            )
        rebound = tuple(
            replace(decode, checkpoint=accepted_checkpoint) for decode in decodes
        )
        frontier = build_frontier_iteration(
            state.manifest,
            manifest_path=self._config.manifest_path,
            iteration=prior.frontier.iteration + 1,
            checkpoint=accepted_checkpoint,
            decodes=rebound,
            previous=prior.frontier,
            previous_path=prior.artifact_path,
        )
        path = (
            state.output_root / "frontiers" / f"iteration-{frontier.iteration:03d}.json"
        )
        digest = canonical_write(frontier, path)
        handle = FrontierHandle(
            str(path),
            digest,
            frontier,
            source_decode_artifact_sha256=proposed_decode.artifact_sha256,
        )
        state.staged_prior_frontier = prior
        state.active_frontier = handle
        return handle

    def write_attempt_receipt(
        self, state: RuntimeBuildState, receipt: AttemptReceipt
    ) -> DurableArtifact:
        path = (
            state.output_root
            / "iterations"
            / f"attempt-{receipt.attempt_index:03d}"
            / "attempt-receipt.json"
        )
        _write_canonical_attempt(path, receipt)
        return DurableArtifact(str(path), _sha256_file(path))

    def abort_staged_acceptance(
        self,
        state: RuntimeBuildState,
        accepted_checkpoint: CheckpointIdentity | None,
        next_frontier: FrontierHandle | None,
        receipt_artifact: DurableArtifact | None,
    ) -> None:
        if receipt_artifact is not None:
            _unlink_staged_file(
                receipt_artifact.path,
                expected_parent=state.output_root / "iterations",
                expected_sha256=receipt_artifact.sha256,
            )
        if next_frontier is not None:
            _unlink_staged_file(
                f"{next_frontier.artifact_path}.sha256",
                expected_parent=state.output_root / "frontiers",
            )
            _unlink_staged_file(
                next_frontier.artifact_path,
                expected_parent=state.output_root / "frontiers",
                expected_sha256=next_frontier.artifact_sha256,
            )
        if accepted_checkpoint is not None:
            _remove_staged_checkpoint(state, accepted_checkpoint)
        if state.staged_prior_frontier is not None:
            state.active_frontier = state.staged_prior_frontier
            state.staged_prior_frontier = None


StateBuilder = Callable[[OnPolicyArmConfig, Path], RuntimeBuildState]


def build_runtime(
    config: OnPolicyArmConfig,
    repo_root: Path,
    *,
    _state_builder: StateBuilder | None = None,
    _services: RuntimeServices | None = None,
) -> Human13OnPolicyRuntime:
    """Assemble the fresh Source BF16/FA2 runtime used by the real CLI."""

    root = Path(repo_root).expanduser().resolve()
    builder = _state_builder or _build_production_state
    state = builder(config, root)
    services = _services or ProductionRuntimeServices(config)
    return Human13OnPolicyRuntime(config, state, services)


def _build_production_state(
    config: OnPolicyArmConfig, repo_root: Path
) -> RuntimeBuildState:
    from scripts.research.human13_live_eval import checkpoint_payload_sha256
    from scripts.research.human13_live_model import (
        assemble_human13_live_model,
        build_human13_live_model_plan,
        build_human13_processor_skeletons,
    )
    from scripts.research.train_human13_live_arm import (
        _build_vocab_groups,
        _load_sealed_manifest,
    )

    output = Path(config.output_root).expanduser()
    if output.exists():
        raise FileExistsError(f"on-policy output root already exists: {output}")
    sealed = _load_sealed_manifest(config.manifest_path)
    if sealed.manifest_sha256 != config.manifest_sha256:
        raise ValueError("sealed manifest differs from on-policy config")
    model_plan = build_human13_live_model_plan(
        successor_model_config(config, repo_root=repo_root)
    )
    assembly = assemble_human13_live_model(
        model_plan,
        pack_count=1,
        repo_root=repo_root,
    )
    skeletons = build_human13_processor_skeletons(
        sealed.manifest,
        assembly.components,
        repo_root=repo_root,
    )
    named_parameters = tuple(
        (name, parameter)
        for name, parameter in assembly.model.named_parameters()
        if parameter.requires_grad
    )
    if not named_parameters:
        raise ValueError("on-policy runtime has no trainable language-DoRA parameters")
    counter = UpdateCounter()
    transaction = TrainingStateTransaction(
        named_parameters,
        optimizer=assembly.optimizer,
        scheduler=assembly.scheduler,
        update_counter=counter,
        runtime=assembly.runtime,
        capture_cuda=True,
    )
    source_records = _load_source_records(config.source_trajectory_path)
    return RuntimeBuildState(
        repo_root=repo_root,
        output_root=output,
        config_sha256=_config_sha256(config),
        manifest=sealed.manifest,
        manifest_sha256=sealed.manifest_sha256,
        assembly=assembly,
        skeletons=skeletons,
        vocab_groups=_build_vocab_groups(assembly.components),
        transaction=transaction,
        update_counter=counter,
        source_checkpoint=CheckpointIdentity(
            assembly.plan.source.checkpoint_path,
            checkpoint_payload_sha256(assembly.plan.source.checkpoint_path),
        ),
        active_frontier=None,
        source_records=source_records,
        continuation_caps=_continuation_caps(source_records),
    )


def _frontier_observation(
    frontier: Human13FrontierIteration,
    *,
    manifest: Any | None = None,
) -> BehaviorGateObservation:
    jointly_coverable = _jointly_coverable_protected(frontier, manifest=manifest)
    owners = tuple(
        sorted(
            {
                owner_id
                for image in frontier.images
                for owner_id in image.canonical_owner_ids
            }
        )
    )
    return BehaviorGateObservation(
        protected_owner_ids=frontier.protected_owner_ids,
        jointly_coverable_protected_owner_ids=jointly_coverable,
        unique_owner_ids=owners,
        cap_hit_count=0,
        malformed_row_count=sum(image.malformed_row_count for image in frontier.images),
        row_count=sum(len(image.rows) for image in frontier.images),
        duplicate_count=sum(len(image.duplicate_events) for image in frontier.images),
    )


def _jointly_coverable_protected(
    frontier: Human13FrontierIteration, *, manifest: Any | None
) -> tuple[str, ...]:
    if manifest is None:
        return tuple(
            sorted(
                {
                    owner_id
                    for image in frontier.images
                    for owner_id in image.constrained_protected_owner_ids
                }
            )
        )
    from scripts.research.human13_frontier_selection import (
        protected_owner_coverable,
    )

    manifest_images = {int(image.image_id): image for image in manifest.images}
    if set(manifest_images) != {image.image_id for image in frontier.images}:
        raise ValueError("behavior observation manifest/frontier images differ")
    protected = set(frontier.protected_owner_ids)
    coverable: set[str] = set()
    for image in frontier.images:
        source = manifest_images[image.image_id]
        owners = tuple(
            (
                str(owner.owner_id),
                str(owner.category),
                _bbox4(owner.bbox),
            )
            for owner in source.owners
        )
        protected_here = tuple(
            owner_id for owner_id, _, _ in owners if owner_id in protected
        )
        predictions = tuple(
            (row.category, row.bbox)
            for row in sorted(image.rows, key=lambda item: item.generated_order)
        )
        if protected_owner_coverable(
            owners,
            predictions,
            protected_owner_ids=protected_here,
        ):
            coverable.update(protected_here)
    return tuple(sorted(coverable))


def _bbox4(values: Any) -> tuple[float, float, float, float]:
    result = tuple(float(value) for value in values)
    if len(result) != 4:
        raise ValueError("owner bbox must contain four coordinates")
    return cast(tuple[float, float, float, float], result)


def _payload_counts(payload: Any) -> tuple[int, int]:
    selected_segments = getattr(payload, "selected_segments", None)
    micro_steps = getattr(payload, "micro_steps", None)
    if not isinstance(selected_segments, tuple) or not isinstance(micro_steps, tuple):
        raise ValueError("on-policy payload lacks selected segments or micro-steps")
    return len(selected_segments), len(micro_steps)


def _prompt_only_native_inputs(
    native_inputs: Mapping[str, Any], *, prompt_count: int
) -> dict[str, Any]:
    import torch

    input_ids = native_inputs.get("input_ids")
    if (
        not isinstance(input_ids, torch.Tensor)
        or input_ids.ndim != 2
        or int(input_ids.shape[0]) != 1
        or int(input_ids.shape[1]) < prompt_count
    ):
        raise ValueError("HF exact-history inputs do not contain the canonical prompt")
    result = {
        key: value
        for key, value in native_inputs.items()
        if key not in {"position_ids", "cache_position", "rope_deltas"}
    }
    result["input_ids"] = input_ids[:, :prompt_count]
    attention = result.get("attention_mask")
    if isinstance(attention, torch.Tensor):
        result["attention_mask"] = attention[:, :prompt_count]
    return result


def _attach_on_policy_image_processor(payload: Any, components: Any) -> Any:
    from src.training.pipeline import (
        _attach_image_processors_to_micro_steps,
        _qwen_image_processor,
    )

    original = tuple(payload.micro_steps)
    attached = _attach_image_processors_to_micro_steps(
        original, image_processor=_qwen_image_processor(components)
    )
    for before_step, after_step in zip(original, attached, strict=True):
        for before, after in zip(
            before_step.encoded_examples, after_step.encoded_examples, strict=True
        ):
            bindings = getattr(before, "human13_on_policy_bindings", None)
            if bindings is None:
                raise ValueError("image processor lost on-policy bindings")
            object.__setattr__(after, "human13_on_policy_bindings", bindings)
    return replace(payload, micro_steps=attached)


def _require_active(state: RuntimeBuildState, frontier: FrontierHandle) -> None:
    if state.active_frontier != frontier:
        raise ValueError("runtime operation is not bound to the active frontier")


def _load_source_records(path: str | Path) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    for line in (
        Path(path).resolve(strict=True).read_text(encoding="utf-8").splitlines()
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        image_id = value.get("image_id") if isinstance(value, Mapping) else None
        if (
            isinstance(image_id, bool)
            or not isinstance(image_id, int)
            or image_id in result
        ):
            raise ValueError("Source records require unique integer image IDs")
        result[image_id] = value
    if len(result) != 13:
        raise ValueError("Source records must cover the full 13-image panel")
    return result


def _continuation_caps(
    source_records: Mapping[int, Mapping[str, Any]],
) -> dict[int, int]:
    from scripts.research.human13_forced_continuation import source_continuation_cap

    caps: dict[int, int] = {}
    for image_id, record in source_records.items():
        trajectory = record.get("trajectory")
        if not isinstance(trajectory, Mapping):
            raise ValueError("Source trajectory must be a mapping")
        rows = trajectory.get("rows")
        token_ids = trajectory.get("token_ids")
        if not isinstance(rows, list) or not isinstance(token_ids, list):
            raise ValueError("Source trajectory rows and token_ids must be lists")
        caps[int(image_id)] = source_continuation_cap(
            source_row_count=len(rows), source_token_count=len(token_ids)
        )
    return caps


def _config_sha256(config: OnPolicyArmConfig) -> str:
    encoded = json.dumps(
        asdict(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    from scripts.research.train_human13_live_arm import write_immutable_receipt

    write_immutable_receipt(path, payload)


def _write_candidate_receipt(
    path: Path,
    *,
    frontier: FrontierHandle,
    scored: Any,
    projections: tuple[Any, ...],
    selected: Any | None,
) -> None:
    _write_receipt(
        path,
        {
            "schema_version": "human13_on_policy_candidate_selection.v1",
            "frontier_sha256": frontier.artifact_sha256,
            "scoring": _jsonable(scored.receipt),
            "cross_surface": _jsonable(scored.cross_surface_receipts),
            "continuations": _jsonable(projections),
            "selected": _jsonable(selected),
        },
    )


def _write_canonical_attempt(path: Path, receipt: AttemptReceipt) -> None:
    encoded = _controller_canonical_bytes(receipt)
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ValueError(f"attempt receipt already exists: {target}") from exc
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _unlink_staged_file(
    path: str | Path,
    *,
    expected_parent: Path,
    expected_sha256: str | None = None,
) -> None:
    target = Path(path).expanduser()
    parent = expected_parent.expanduser().resolve()
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"staged artifact is unavailable: {target}")
    resolved = target.resolve(strict=True)
    if not resolved.is_relative_to(parent):
        raise ValueError(f"staged artifact escaped its owned root: {resolved}")
    if expected_sha256 is not None and _sha256_file(resolved) != expected_sha256:
        raise ValueError(f"staged artifact payload drifted: {resolved}")
    resolved.unlink()


def _remove_staged_checkpoint(
    state: RuntimeBuildState, checkpoint: CheckpointIdentity
) -> None:
    from scripts.research.human13_live_eval import checkpoint_payload_sha256

    target = Path(checkpoint.path).expanduser()
    accepted = (state.output_root / "accepted" / "checkpoints").resolve()
    if target.is_symlink() or not target.is_dir():
        raise ValueError("staged accepted checkpoint is unavailable")
    resolved = target.resolve(strict=True)
    if resolved.parent != accepted or not resolved.name.startswith("step-"):
        raise ValueError("staged accepted checkpoint escaped its owned root")
    if checkpoint_payload_sha256(resolved) != checkpoint.payload_sha256:
        raise ValueError("staged accepted checkpoint payload drifted")
    shutil.rmtree(resolved)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).resolve(strict=True).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(cast(Any, value)))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    method = getattr(value, "to_artifact_dict", None)
    if callable(method):
        return _jsonable(method())
    if isinstance(value, SimpleNamespace):
        return _jsonable(vars(value))
    raise TypeError(f"unsupported receipt value: {type(value).__name__}")


__all__ = [
    "Human13OnPolicyRuntime",
    "ProductionRuntimeServices",
    "RuntimeBuildState",
    "build_runtime",
]
