from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from scripts.research.build_human13_on_policy_frontier import (
    CheckpointIdentity,
    FrontierImage,
    FrontierRow,
    Human13FrontierIteration,
)
from scripts.research.human13_on_policy_runtime import (
    Human13OnPolicyRuntime,
    ProductionRuntimeServices,
    RuntimeBuildState,
    _config_sha256,
    _continuation_caps,
    _frontier_observation,
    _payload_counts,
    build_runtime,
)
from scripts.research.human13_frontier_selection import (
    CandidatePath,
    CandidateScore,
    ContinuationOutcome,
)
from scripts.research.human13_on_policy_live import BehaviorGateObservation
from scripts.research.human13_training_transaction import UpdateCounter
from scripts.research.train_human13_on_policy_successor import (
    AttemptReceipt,
    CandidateSelectionReceipt,
    CleanDecodeReceipt,
    DurableArtifact,
    FrontierHandle,
    OneUpdateReceipt,
    PrivateProposal,
    RuntimeIdentity,
    load_on_policy_config,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "configs/coordexp_swift/research/human13_on_policy_successor/02_o_first_safe.yaml"
)
SOURCE = (
    "/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/"
    "2026-08-05-closeout/artifacts/training/four-coordinate-xy/checkpoints/step-2444"
)


def _score() -> CandidateScore:
    return CandidateScore(
        path=CandidatePath(1, "h0", "row0", (10, 11)),
        hf_sites=(),
        packed_sites=(),
        aligned_sites=(),
        hf_barrier=0.0,
        first_bottleneck_index=None,
        packed_first_bottleneck_index=None,
        first_bottleneck_disagreement=None,
        packed_barrier=0.0,
        max_surface_margin_drift=0.0,
        surface_rank_disagreement=False,
    )


def _attempt_receipt() -> AttemptReceipt:
    return AttemptReceipt(
        attempt_index=0,
        ledger_path="frontier.json",
        ledger_sha256="e" * 64,
        selected_owner_id="h0",
        selected_alias_id="row0",
        proposal_checkpoint_path="private",
        proposal_checkpoint_sha256="c" * 64,
        proposed_decode_path="decode.jsonl",
        proposed_decode_sha256="f" * 64,
        decision="accepted",
        decision_reasons=(),
        transaction_before_sha256="1" * 64,
        transaction_after_sha256="2" * 64,
        accepted_checkpoint_path="accepted",
        accepted_checkpoint_sha256="c" * 64,
        next_ledger_path="next.json",
        next_ledger_sha256="3" * 64,
        rollback_decode_path=None,
        rollback_decode_sha256=None,
    )


def _outcome() -> ContinuationOutcome:
    return ContinuationOutcome(
        owner_id="h0",
        hf_barrier=0.0,
        protected_coverable=True,
        unique_owner_delta=1,
        termination_status="natural_im_end",
        cap_hit=False,
        duplicate_increase=0,
        malformed_increase=0,
        row_count=1,
        generated_tokens=2,
    )


def _iteration(checkpoint: CheckpointIdentity) -> Human13FrontierIteration:
    return Human13FrontierIteration(
        schema_version="human13_on_policy_frontier.v1",
        manifest_path="manifest.json",
        manifest_sha256="a" * 64,
        panel_sha256="b" * 64,
        iteration=0,
        checkpoint=checkpoint,
        previous_frontier_path=None,
        previous_frontier_sha256=None,
        protected_owner_ids=("g0",),
        protected_owner_ages=(("g0", 0),),
        images=(),
    )


class _Services:
    def __init__(self, frontier: FrontierHandle):
        self.frontier = frontier
        self.calls: list[str] = []

    def initial_frontier(self, state):
        self.calls.append("initial")
        state.active_frontier = self.frontier
        return self.frontier

    def select_candidate(self, state, frontier):
        self.calls.append("select")
        score = _score()
        return CandidateSelectionReceipt(
            decision_surface="hf_fp32_sdpa_batch1",
            eligible_owner_count=1,
            shortlisted=(score,),
            forced_continuation_count=1,
            selected=score,
            continuation_projection_sha256s=("f" * 64,),
            continuation_outcomes=(_outcome(),),
        )

    def apply_one_update(self, state, frontier, selection):
        self.calls.append("update")
        state.update_counter.value += 1
        return OneUpdateReceipt(frontier.artifact_sha256, 1, True)

    def write_private_proposal(
        self, state, frontier, selection, update, *, proposal_run_dir
    ):
        self.calls.append("private")
        checkpoint = proposal_run_dir / "checkpoints" / "step-1"
        checkpoint.mkdir(parents=True)
        return CheckpointIdentity(str(checkpoint), "c" * 64)

    def clean_decode(self, state, checkpoint, *, purpose):
        self.calls.append(f"decode:{purpose}")
        return CleanDecodeReceipt(
            decision_surface="hf_fp32_sdpa_batch1",
            checkpoint=checkpoint,
            artifact_path="decode.jsonl",
            artifact_sha256="f" * 64,
            observation=BehaviorGateObservation((), (), (), 0, 0, 0, 0),
            current_decodes=(),
        )

    def publish_accepted_checkpoint(self, state, frontier, proposal):
        self.calls.append("publish")
        return CheckpointIdentity("accepted", proposal.checkpoint.payload_sha256)

    def materialize_next_frontier(
        self, state, prior, proposed_decode, accepted_checkpoint
    ):
        self.calls.append("advance")
        return self.frontier

    def write_attempt_receipt(self, state, receipt):
        self.calls.append("receipt")
        return DurableArtifact("attempt.json", "a" * 64)

    def abort_staged_acceptance(
        self, state, accepted_checkpoint, next_frontier, receipt_artifact
    ):
        self.calls.append("abort")


def _state(tmp_path: Path) -> RuntimeBuildState:
    checkpoint = CheckpointIdentity(SOURCE, "c" * 64)
    return RuntimeBuildState(
        repo_root=ROOT,
        output_root=tmp_path / "run",
        config_sha256="d" * 64,
        manifest=SimpleNamespace(images=()),
        manifest_sha256="a" * 64,
        assembly=SimpleNamespace(
            plan=SimpleNamespace(world_size=1),
            runtime=SimpleNamespace(optimizer_step_count=0),
        ),
        skeletons={},
        vocab_groups=SimpleNamespace(),
        transaction=SimpleNamespace(),
        update_counter=UpdateCounter(),
        source_checkpoint=checkpoint,
        active_frontier=None,
    )


def test_build_runtime_uses_injected_state_builder_without_model_actions(
    tmp_path: Path,
) -> None:
    config = replace(
        load_on_policy_config(CONFIG, repo_root=ROOT),
        output_root=str(tmp_path / "fresh"),
    )
    state = _state(tmp_path)
    calls = []

    runtime = build_runtime(
        config,
        ROOT,
        _state_builder=lambda received, root: calls.append((received, root)) or state,
        _services=cast(Any, SimpleNamespace()),
    )

    assert isinstance(runtime, Human13OnPolicyRuntime)
    assert calls == [(config, ROOT)]
    assert runtime.execution_identity() == RuntimeIdentity(SOURCE, 1, 0)


def test_config_hash_binds_the_parsed_output_root_not_a_hardcoded_arm_file(
    tmp_path: Path,
) -> None:
    config = load_on_policy_config(CONFIG, repo_root=ROOT)
    vertical = replace(config, output_root=str(tmp_path / "vertical"))
    pilot = replace(config, output_root=str(tmp_path / "pilot"))

    assert _config_sha256(vertical) == _config_sha256(vertical)
    assert _config_sha256(vertical) != _config_sha256(pilot)


def test_runtime_is_a_thin_typed_adapter_over_production_services(
    tmp_path: Path,
) -> None:
    config = replace(
        load_on_policy_config(CONFIG, repo_root=ROOT),
        output_root=str(tmp_path / "run"),
    )
    state = _state(tmp_path)
    frontier = FrontierHandle(
        "frontier.json", "e" * 64, _iteration(state.source_checkpoint)
    )
    services = _Services(frontier)
    runtime = Human13OnPolicyRuntime(config, state, services)

    assert runtime.initial_frontier(config) == frontier
    assert runtime.frontier_observation(frontier).unique_owner_ids == ()
    selection = runtime.select_candidate(config, frontier)
    assert selection is not None
    assert selection.decision_surface == "hf_fp32_sdpa_batch1"
    update = runtime.apply_one_update(config, frontier, selection)
    proposal_dir = tmp_path / "proposal"
    proposal_checkpoint = runtime.write_private_proposal(
        config,
        frontier,
        selection,
        update,
        proposal_run_dir=proposal_dir,
    )
    decoded = runtime.clean_decode(config, proposal_checkpoint, purpose="proposal_gate")
    proposal = PrivateProposal("p0", proposal_checkpoint, frontier.artifact_sha256)
    accepted = runtime.publish_accepted_checkpoint(config, frontier, proposal)
    runtime.materialize_next_frontier(config, frontier, decoded, accepted)
    runtime.write_attempt_receipt(config, _attempt_receipt())
    runtime.abort_staged_acceptance(config, accepted, frontier, None)

    assert services.calls == [
        "initial",
        "select",
        "update",
        "private",
        "decode:proposal_gate",
        "publish",
        "advance",
        "receipt",
        "abort",
    ]


def test_runtime_rejects_foreign_config_instance(tmp_path: Path) -> None:
    config = replace(
        load_on_policy_config(CONFIG, repo_root=ROOT),
        output_root=str(tmp_path / "run"),
    )
    state = _state(tmp_path)
    frontier = FrontierHandle(
        "frontier.json", "e" * 64, _iteration(state.source_checkpoint)
    )
    runtime = Human13OnPolicyRuntime(config, state, _Services(frontier))

    with pytest.raises(ValueError, match="different arm config"):
        runtime.initial_frontier(replace(config, arm_name="foreign"))


def test_frontier_observation_aggregates_the_same_frontier_only() -> None:
    observation = _frontier_observation(
        _iteration(CheckpointIdentity(SOURCE, "c" * 64))
    )

    assert observation.protected_owner_ids == ("g0",)
    assert observation.jointly_coverable_protected_owner_ids == ()
    assert observation.unique_owner_ids == ()
    assert observation.cap_hit_count == 0
    assert observation.malformed_row_count == 0
    assert observation.row_count == 0
    assert observation.duplicate_count == 0


def test_frontier_observation_uses_joint_protected_coverability_not_canonical_id() -> (
    None
):
    checkpoint = CheckpointIdentity(SOURCE, "c" * 64)
    image = FrontierImage(
        image_id=1,
        trajectory_id="decode:1",
        generated_token_ids=(10, 11, 99),
        parser="compact_object_box_closed_only",
        parser_status="accepted",
        stop_reason="im_end",
        rows=(FrontierRow(0, "person", (0.0, 0.0, 10.0, 10.0), 0, 2, (10, 11)),),
        canonical_owner_ids=(),
        constrained_protected_owner_ids=(),
        covered_h_owner_ids=(),
        uncovered_h_owner_ids=(),
        candidate_aliases=(),
        duplicate_events=(),
        terminal_token_index=2,
        malformed_row_count=0,
    )
    frontier = replace(_iteration(checkpoint), images=(image,))
    manifest = SimpleNamespace(
        images=(
            SimpleNamespace(
                image_id=1,
                owners=(
                    SimpleNamespace(
                        owner_id="g0",
                        category="person",
                        bbox=(0.0, 0.0, 10.0, 10.0),
                    ),
                ),
            ),
        )
    )

    observation = _frontier_observation(frontier, manifest=manifest)

    assert observation.protected_owner_ids == ("g0",)
    assert observation.jointly_coverable_protected_owner_ids == ("g0",)


def test_continuation_caps_are_bound_per_source_image() -> None:
    records = {
        7: {"trajectory": {"rows": [{}, {}, {}], "token_ids": list(range(20))}},
        8: {"trajectory": {"rows": [{}] * 300, "token_ids": list(range(5))}},
    }

    assert _continuation_caps(records) == {7: 532, 8: 600}


def test_payload_receipt_counts_use_live_selected_segment_field() -> None:
    payload = SimpleNamespace(
        selected_segments=(object(), object()), micro_steps=(object(),)
    )

    assert _payload_counts(payload) == (2, 1)


@pytest.mark.parametrize(
    "record",
    [
        {},
        {"trajectory": []},
        {"trajectory": {"rows": "not-rows", "token_ids": []}},
        {"trajectory": {"rows": [], "token_ids": "not-tokens"}},
    ],
)
def test_continuation_caps_reject_malformed_source_records(record) -> None:
    with pytest.raises(ValueError, match="Source trajectory"):
        _continuation_caps({7: record})


def test_production_service_writes_controller_canonical_attempt_receipt(
    tmp_path: Path,
) -> None:
    config = replace(
        load_on_policy_config(CONFIG, repo_root=ROOT),
        output_root=str(tmp_path / "run"),
    )
    state = _state(tmp_path)
    receipt = _attempt_receipt()

    artifact = ProductionRuntimeServices(config).write_attempt_receipt(state, receipt)

    path = Path(artifact.path)
    assert json.loads(path.read_text(encoding="utf-8")) == json.loads(
        json.dumps(asdict(receipt))
    )
    assert (
        path.read_bytes()
        == (
            json.dumps(
                asdict(receipt),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode()
    )


def test_production_abort_removes_only_staged_acceptance_and_restores_frontier(
    tmp_path: Path,
) -> None:
    from scripts.research.human13_live_eval import checkpoint_payload_sha256

    config = replace(
        load_on_policy_config(CONFIG, repo_root=ROOT),
        output_root=str(tmp_path / "run"),
    )
    state = _state(tmp_path)
    service = ProductionRuntimeServices(config)
    prior = FrontierHandle("prior.json", "1" * 64, _iteration(state.source_checkpoint))
    checkpoint_path = state.output_root / "accepted/checkpoints/step-1"
    (checkpoint_path / "adapter").mkdir(parents=True)
    (checkpoint_path / "special_token_embeddings").mkdir()
    (checkpoint_path / "adapter/weights.bin").write_bytes(b"adapter")
    (checkpoint_path / "special_token_embeddings/delta.bin").write_bytes(b"delta")
    accepted = CheckpointIdentity(
        str(checkpoint_path), checkpoint_payload_sha256(checkpoint_path)
    )
    frontier_path = state.output_root / "frontiers/iteration-001.json"
    frontier_path.parent.mkdir(parents=True)
    frontier_path.write_bytes(b"frontier")
    Path(f"{frontier_path}.sha256").write_text("digest\n", encoding="ascii")
    next_frontier = FrontierHandle(
        str(frontier_path),
        hashlib.sha256(b"frontier").hexdigest(),
        replace(_iteration(accepted), iteration=1),
    )
    receipt_artifact = service.write_attempt_receipt(state, _attempt_receipt())
    state.active_frontier = next_frontier
    state.staged_prior_frontier = prior

    service.abort_staged_acceptance(state, accepted, next_frontier, receipt_artifact)

    assert state.active_frontier == prior
    assert state.staged_prior_frontier is None
    assert not checkpoint_path.exists()
    assert not frontier_path.exists()
    assert not Path(f"{frontier_path}.sha256").exists()
    assert not Path(receipt_artifact.path).exists()
