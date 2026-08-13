from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil
from typing import cast

import pytest
import torch

from scripts.research.build_human13_on_policy_frontier import (
    CheckpointIdentity,
    CurrentDecode,
    CurrentPrediction,
    FrontierImage,
    FrontierRow,
    Human13FrontierIteration,
)
from scripts.research.human13_frontier_selection import CandidatePath, CandidateScore
from scripts.research.human13_on_policy_live import BehaviorGateObservation
from scripts.research.human13_training_transaction import (
    TrainingStateTransaction,
    UpdateCounter,
)
from scripts.research.train_human13_on_policy_successor import (
    CandidateSelectionReceipt,
    CleanDecodeReceipt,
    DurableArtifact,
    FrontierHandle,
    OneUpdateReceipt,
    OnPolicyArmConfig,
    OnPolicySuccessorError,
    RuntimeIdentity,
    build_dry_run_receipt,
    execute_cli,
    load_on_policy_config,
    materialize_on_policy_plans,
    run_on_policy_loop,
)
from scripts.research.human13_proposal_checkpoint import (
    promote_private_proposal_checkpoint,
)
from scripts.research.human13_live_eval import checkpoint_payload_sha256


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs/coordexp_swift/research/human13_on_policy_successor"
SOURCE_CHECKPOINT = (
    "/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/"
    "2026-08-05-closeout/artifacts/training/four-coordinate-xy/checkpoints/step-2444"
)


def _candidate(owner_id: str) -> CandidateScore:
    path = CandidatePath(7, owner_id, f"alias-{owner_id}", (1, 2, 3, 4))
    return CandidateScore(path, (), (), (), 1.0, 1, 1, False, 1.0, 0.0, False)


def _frontier(
    iteration: int,
    checkpoint: CheckpointIdentity,
    *,
    previous_sha256: str | None,
    decodes: tuple[CurrentDecode, ...] = (),
) -> Human13FrontierIteration:
    return Human13FrontierIteration(
        schema_version="human13_on_policy_frontier.v1",
        manifest_path="manifest.json",
        manifest_sha256="a" * 64,
        panel_sha256="b" * 64,
        iteration=iteration,
        checkpoint=checkpoint,
        previous_frontier_path=None if iteration == 0 else "previous.json",
        previous_frontier_sha256=previous_sha256,
        protected_owner_ids=("g0",),
        protected_owner_ages=(("g0", iteration),),
        images=tuple(_frontier_image(decode) for decode in decodes),
    )


def _decodes(checkpoint: CheckpointIdentity) -> tuple[CurrentDecode, ...]:
    return tuple(
        CurrentDecode(
            image_id=image_id,
            trajectory_id=f"decode:{image_id}",
            generated_token_ids=(1000 + image_id, 2000 + image_id, 999),
            predictions=(
                CurrentPrediction(
                    generated_order=0,
                    category="person",
                    bbox=(1.0, 2.0, 3.0, 4.0),
                    token_start=0,
                    token_end=2,
                ),
            ),
            parser="compact_object_box_closed_only",
            parser_status="complete",
            stop_reason="im_end",
            checkpoint=checkpoint,
            terminal_token_index=2,
            malformed_row_count=0,
        )
        for image_id in range(1, 14)
    )


def _frontier_image(decode: CurrentDecode) -> FrontierImage:
    prediction = decode.predictions[0]
    return FrontierImage(
        image_id=decode.image_id,
        trajectory_id=decode.trajectory_id,
        generated_token_ids=decode.generated_token_ids,
        parser=decode.parser,
        parser_status=decode.parser_status,
        stop_reason=decode.stop_reason,
        rows=(
            FrontierRow(
                generated_order=prediction.generated_order,
                category=prediction.category,
                bbox=prediction.bbox,
                token_start=prediction.token_start,
                token_end=prediction.token_end,
                token_ids=decode.generated_token_ids[
                    prediction.token_start : prediction.token_end
                ],
            ),
        ),
        canonical_owner_ids=("g0",),
        constrained_protected_owner_ids=("g0",),
        covered_h_owner_ids=(),
        uncovered_h_owner_ids=("h0",),
        candidate_aliases=(),
        duplicate_events=(),
        terminal_token_index=decode.terminal_token_index,
        malformed_row_count=decode.malformed_row_count,
    )


def _write_artifact(path: Path, value: object) -> DurableArtifact:
    payload = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return DurableArtifact(str(path), hashlib.sha256(payload).hexdigest())


class _RecordingTransaction:
    def __init__(self, transaction: TrainingStateTransaction, events: list[str]):
        self._transaction = transaction
        self._events = events

    def begin(self):
        self._events.append("transaction:begin")
        return self._transaction.begin()

    def accept(self, snapshot):
        self._events.append("transaction:accept")
        return self._transaction.accept(snapshot)

    def reject(self, snapshot):
        self._events.append("transaction:reject")
        return self._transaction.reject(snapshot)

    def state_digest(self):
        return self._transaction.state_digest()


class _Runtime:
    def __init__(
        self,
        output_root: Path,
        *,
        accept: bool,
        endless: bool = False,
        fail_at: str | None = None,
    ):
        self.output_root = output_root
        self.accept = accept
        self.endless = endless
        self.fail_at = fail_at
        self.events: list[str] = []
        self.published: list[str] = []
        self.written_receipts = []
        self.staged_paths: list[Path] = []
        self.model = torch.nn.Linear(1, 1, bias=False)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda _: 1.0
        )
        self.counter = UpdateCounter()
        transaction = TrainingStateTransaction(
            (("adapter.language.weight", cast(torch.nn.Parameter, self.model.weight)),),
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            update_counter=self.counter,
        )
        self.transaction = _RecordingTransaction(transaction, self.events)

    def execution_identity(self) -> RuntimeIdentity:
        return RuntimeIdentity(
            source_checkpoint_path=SOURCE_CHECKPOINT,
            world_size=1,
            optimizer_update_count=self.counter.value,
        )

    def initial_frontier(self, config: OnPolicyArmConfig) -> FrontierHandle:
        del config
        self.events.append("frontier:initial")
        checkpoint = CheckpointIdentity(SOURCE_CHECKPOINT, "c" * 64)
        return FrontierHandle(
            artifact_path="frontier-0.json",
            artifact_sha256="0" * 64,
            frontier=_frontier(0, checkpoint, previous_sha256=None),
            source_decode_artifact_sha256=None,
        )

    def frontier_observation(self, frontier: FrontierHandle):
        self.events.append(f"observation:{frontier.frontier.iteration}")
        owners = ("g0",) if frontier.frontier.iteration == 0 else ("g0", "h0")
        return BehaviorGateObservation(
            protected_owner_ids=("g0",),
            jointly_coverable_protected_owner_ids=("g0",),
            unique_owner_ids=owners,
            cap_hit_count=0,
            malformed_row_count=0,
            row_count=2,
            duplicate_count=0,
        )

    def select_candidate(self, config: OnPolicyArmConfig, frontier: FrontierHandle):
        del config
        self.events.append(f"candidate:{frontier.frontier.iteration}")
        if not self.endless and frontier.frontier.iteration > 0:
            return None
        shortlisted = (_candidate("h0"), _candidate("h1"))
        return CandidateSelectionReceipt(
            decision_surface="hf_fp32_sdpa_batch1",
            eligible_owner_count=2,
            shortlisted=shortlisted,
            forced_continuation_count=2,
            selected=shortlisted[0],
        )

    def apply_one_update(self, config, frontier, selection):
        del config, selection
        self.events.append(f"update:{frontier.frontier.iteration}")
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.model(torch.ones((1, 1))).sum()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.counter.value += 1
        return OneUpdateReceipt(
            ledger_sha256=frontier.artifact_sha256,
            applied_optimizer_updates=1,
            finite=True,
        )

    def write_private_proposal(
        self, config, frontier, selection, update, *, proposal_run_dir: Path
    ):
        del config, selection, update
        self.events.append(f"proposal:private:{frontier.frontier.iteration}")
        checkpoint = proposal_run_dir / "checkpoints" / "step-1"
        (checkpoint / "adapter").mkdir(parents=True)
        (checkpoint / "special_token_embeddings").mkdir()
        (checkpoint / "adapter" / "weights.bin").write_bytes(b"adapter")
        (checkpoint / "special_token_embeddings" / "delta.bin").write_bytes(b"delta")
        return CheckpointIdentity(
            str(checkpoint),
            checkpoint_payload_sha256(checkpoint),
        )

    def clean_decode(self, config, checkpoint, *, purpose: str):
        self.events.append(f"decode:{purpose}")
        if purpose == "rollback_reproduction" or self.accept:
            owners = ("g0", "h0") if purpose == "proposal_gate" else ("g0",)
            protected = ("g0",)
        else:
            owners = ("h0",)
            protected = ()
        observation = BehaviorGateObservation(
            protected_owner_ids=("g0",),
            jointly_coverable_protected_owner_ids=protected,
            unique_owner_ids=owners,
            cap_hit_count=0,
            malformed_row_count=0,
            row_count=2,
            duplicate_count=0,
        )
        decodes = _decodes(checkpoint)
        artifact = _write_artifact(
            Path(config.output_root)
            / "decodes"
            / f"{purpose}-{self.counter.value}.jsonl",
            [asdict(decode) for decode in decodes],
        )
        return CleanDecodeReceipt(
            decision_surface="hf_fp32_sdpa_batch1",
            checkpoint=checkpoint,
            artifact_path=artifact.path,
            artifact_sha256=artifact.sha256,
            observation=observation,
            current_decodes=decodes,
        )

    def publish_accepted_checkpoint(self, config, frontier, proposal):
        self.events.append(f"proposal:publish:{frontier.frontier.iteration}")
        path = promote_private_proposal_checkpoint(
            proposal.checkpoint.path,
            accepted_run_dir=Path(config.output_root) / "accepted",
            accepted_step=frontier.frontier.iteration + 1,
        )
        self.staged_paths.append(path)
        self.published.append(str(path))
        if self.fail_at == "publish":
            raise RuntimeError("injected publish failure")
        return CheckpointIdentity(str(path), checkpoint_payload_sha256(path))

    def materialize_next_frontier(
        self, config, prior, proposed_decode, accepted_checkpoint
    ):
        del config
        self.events.append(f"frontier:advance:{prior.frontier.iteration}")
        iteration = prior.frontier.iteration + 1
        frontier = _frontier(
            iteration,
            accepted_checkpoint,
            previous_sha256=prior.artifact_sha256,
            decodes=proposed_decode.current_decodes,
        )
        artifact = _write_artifact(
            self.output_root / "frontiers" / f"frontier-{iteration}.json",
            asdict(frontier),
        )
        self.staged_paths.append(Path(artifact.path))
        if self.fail_at == "frontier":
            raise RuntimeError("injected frontier failure")
        return FrontierHandle(
            artifact_path=artifact.path,
            artifact_sha256=artifact.sha256,
            frontier=frontier,
            source_decode_artifact_sha256=proposed_decode.artifact_sha256,
        )

    def write_attempt_receipt(self, config, receipt):
        self.events.append(f"receipt:{receipt.decision}")
        self.written_receipts.append(receipt)
        artifact = _write_artifact(
            Path(config.output_root)
            / "receipts"
            / f"attempt-{receipt.attempt_index}-{receipt.decision}.json",
            asdict(receipt),
        )
        self.staged_paths.append(Path(artifact.path))
        if self.fail_at == "receipt":
            raise RuntimeError("injected receipt failure")
        return artifact

    def abort_staged_acceptance(
        self, config, accepted_checkpoint, next_frontier, receipt_artifact
    ):
        del config, accepted_checkpoint, next_frontier, receipt_artifact
        self.events.append("acceptance:abort")
        for path in reversed(self.staged_paths):
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        self.staged_paths.clear()
        for directory in (
            self.output_root / "accepted" / "checkpoints",
            self.output_root / "accepted",
            self.output_root / "frontiers",
            self.output_root / "receipts",
        ):
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()


class _BadRollbackRuntime(_Runtime):
    def clean_decode(self, config, checkpoint, *, purpose: str):
        receipt = super().clean_decode(config, checkpoint, purpose=purpose)
        if purpose != "rollback_reproduction":
            return receipt
        observation = replace(receipt.observation, unique_owner_ids=("wrong",))
        return replace(receipt, observation=observation)


def _config(arm_id: str = "O-First-Safe"):
    filename = (
        "02_o_first_safe.yaml" if arm_id == "O-First-Safe" else "01_o_full_safe.yaml"
    )
    return load_on_policy_config(CONFIG_ROOT / filename, repo_root=ROOT)


def test_exact_two_configs_bind_fresh_world_one_hf_surface_and_zero_actions(
    tmp_path: Path,
) -> None:
    plans = materialize_on_policy_plans(CONFIG_ROOT, repo_root=ROOT)

    assert tuple(plan.arm_id for plan in plans) == ("O-Full-Safe", "O-First-Safe")
    assert len({plan.output_root for plan in plans}) == 2
    for config in plans:
        config = replace(config, output_root=str(tmp_path / config.arm_id))
        receipt = build_dry_run_receipt(config, repo_root=ROOT)
        assert config.world_size == 1
        assert config.max_attempts == 8
        assert config.one_update_per_ledger is True
        assert config.fresh_source_each_arm is True
        assert config.fresh_optimizer_each_arm is True
        assert config.decision_surface == "hf_fp32_sdpa_batch1"
        assert receipt["actions"] == {
            "model_loads": 0,
            "forwards": 0,
            "backwards": 0,
            "optimizer_steps": 0,
            "clean_decodes": 0,
            "checkpoint_writes": 0,
            "gpu_allocations": 0,
        }
        assert receipt["output_root_fresh"] is True
        assert receipt["model_plan"]["world_size"] == 1
        assert receipt["model_plan"]["arm_id"] == config.arm_id


def test_execute_fails_closed_without_authority_or_injected_runtime(
    tmp_path: Path,
) -> None:
    config = replace(_config(), output_root=str(tmp_path / "run"))

    with pytest.raises(OnPolicySuccessorError, match="explicit model/GPU authority"):
        run_on_policy_loop(config, runtime=None, execute_authorized=False)
    with pytest.raises(OnPolicySuccessorError, match="injected live runtime"):
        run_on_policy_loop(config, runtime=None, execute_authorized=True)


def test_accept_only_after_durable_artifacts_and_advances_from_one_decode(
    tmp_path: Path,
) -> None:
    config = replace(_config(), output_root=str(tmp_path / "accepted-run"))
    runtime = _Runtime(Path(config.output_root), accept=True)

    receipt = run_on_policy_loop(config, runtime=runtime, execute_authorized=True)

    assert receipt.stop_reason == "no_candidate"
    assert receipt.attempted_update_count == 1
    assert receipt.accepted_update_count == 1
    assert runtime.counter.value == 1
    assert (
        runtime.events.index("proposal:publish:0")
        < runtime.events.index("frontier:advance:0")
        < runtime.events.index("receipt:accepted")
        < runtime.events.index("transaction:accept")
    )
    assert runtime.events.count("decode:proposal_gate") == 1
    assert "decode:rollback_reproduction" not in runtime.events
    assert receipt.attempts[0].proposal_checkpoint_path.startswith(
        str(Path(config.output_root) / "private")
    )
    accepted_path = receipt.attempts[0].accepted_checkpoint_path
    assert accepted_path is not None
    assert accepted_path.startswith(str(Path(config.output_root) / "accepted"))
    attempt = receipt.attempts[0]
    assert attempt.proposal_checkpoint_sha256
    assert attempt.accepted_checkpoint_sha256 == attempt.proposal_checkpoint_sha256
    assert attempt.proposed_decode_sha256
    assert attempt.next_ledger_sha256


@pytest.mark.parametrize("fail_at", ("publish", "frontier", "receipt"))
def test_acceptance_artifact_failure_restores_full_transaction_and_cleans_staging(
    tmp_path: Path,
    fail_at: str,
) -> None:
    config = replace(_config(), output_root=str(tmp_path / f"fail-{fail_at}"))
    runtime = _Runtime(Path(config.output_root), accept=True, fail_at=fail_at)
    before = runtime.transaction.state_digest()

    with pytest.raises(RuntimeError, match=f"injected {fail_at} failure"):
        run_on_policy_loop(config, runtime=runtime, execute_authorized=True)

    assert runtime.transaction.state_digest() == before
    assert runtime.counter.value == 0
    assert "transaction:accept" not in runtime.events
    assert "transaction:reject" in runtime.events
    assert "acceptance:abort" in runtime.events
    assert all(not path.exists() for path in runtime.staged_paths)
    assert not (Path(config.output_root) / "accepted" / "checkpoints").exists()
    assert not any((Path(config.output_root) / "frontiers").glob("*"))
    assert not any((Path(config.output_root) / "receipts").glob("*"))


def test_reject_restores_before_reproduction_and_never_publishes(
    tmp_path: Path,
) -> None:
    config = replace(_config(), output_root=str(tmp_path / "rejected-run"))
    runtime = _Runtime(Path(config.output_root), accept=False)
    before = runtime.model.weight.detach().clone()

    receipt = run_on_policy_loop(config, runtime=runtime, execute_authorized=True)

    assert receipt.stop_reason == "scientific_rejection"
    assert receipt.attempted_update_count == 1
    assert receipt.accepted_update_count == 0
    assert runtime.counter.value == 0
    assert torch.equal(runtime.model.weight, before)
    assert runtime.published == []
    assert runtime.events.index("transaction:reject") < runtime.events.index(
        "decode:rollback_reproduction"
    )
    assert list((Path(config.output_root) / "private").iterdir()) == []
    assert receipt.attempts[0].accepted_checkpoint_path is None
    assert receipt.attempts[0].accepted_checkpoint_sha256 is None
    assert receipt.attempts[0].next_ledger_sha256 is None
    assert receipt.attempts[0].transaction_before_sha256 == (
        receipt.attempts[0].transaction_after_sha256
    )


def test_loop_attempts_at_most_eight_and_updates_each_ledger_once(
    tmp_path: Path,
) -> None:
    config = replace(_config(), output_root=str(tmp_path / "bounded-run"))
    runtime = _Runtime(Path(config.output_root), accept=True, endless=True)

    receipt = run_on_policy_loop(config, runtime=runtime, execute_authorized=True)

    assert receipt.stop_reason == "attempt_cap"
    assert receipt.attempted_update_count == 8
    assert receipt.accepted_update_count == 8
    updated_ledgers = [
        event.removeprefix("update:")
        for event in runtime.events
        if event.startswith("update:")
    ]
    assert updated_ledgers == [str(index) for index in range(8)]


def test_rollback_non_reproduction_persists_a_terminal_attempt_receipt(
    tmp_path: Path,
) -> None:
    config = replace(_config(), output_root=str(tmp_path / "bad-rollback"))
    runtime = _BadRollbackRuntime(Path(config.output_root), accept=False)

    with pytest.raises(OnPolicySuccessorError, match="did not reproduce"):
        run_on_policy_loop(config, runtime=runtime, execute_authorized=True)

    assert len(runtime.written_receipts) == 1
    terminal = runtime.written_receipts[0]
    assert terminal.decision == "rollback_failed"
    assert terminal.decision_reasons[-1] == "rollback_owner_non_reproduction"
    assert terminal.rollback_decode_path.endswith("rollback_reproduction-0.jsonl")
    assert terminal.rollback_decode_sha256


class _BadAcceptedCheckpointRuntime(_Runtime):
    def publish_accepted_checkpoint(self, config, frontier, proposal):
        del config, frontier, proposal
        return CheckpointIdentity(
            str(self.output_root / "accepted" / "missing"), "f" * 64
        )


def test_nonexistent_accepted_checkpoint_fails_closed_and_restores(
    tmp_path: Path,
) -> None:
    config = replace(_config(), output_root=str(tmp_path / "missing-checkpoint"))
    runtime = _BadAcceptedCheckpointRuntime(Path(config.output_root), accept=True)
    before = runtime.transaction.state_digest()

    with pytest.raises(OnPolicySuccessorError, match="accepted checkpoint"):
        run_on_policy_loop(config, runtime=runtime, execute_authorized=True)

    assert runtime.transaction.state_digest() == before
    assert "transaction:accept" not in runtime.events


class _TamperedAcceptedCheckpointRuntime(_Runtime):
    def publish_accepted_checkpoint(self, config, frontier, proposal):
        identity = super().publish_accepted_checkpoint(config, frontier, proposal)
        (Path(identity.path) / "adapter" / "weights.bin").write_bytes(b"tampered")
        return identity


def test_accepted_checkpoint_payload_must_match_private_proposal(
    tmp_path: Path,
) -> None:
    config = replace(_config(), output_root=str(tmp_path / "tampered-checkpoint"))
    runtime = _TamperedAcceptedCheckpointRuntime(Path(config.output_root), accept=True)
    before = runtime.transaction.state_digest()

    with pytest.raises(OnPolicySuccessorError, match="accepted checkpoint"):
        run_on_policy_loop(config, runtime=runtime, execute_authorized=True)

    assert runtime.transaction.state_digest() == before
    assert "transaction:accept" not in runtime.events


class _StaleFrontierRuntime(_Runtime):
    def materialize_next_frontier(
        self, config, prior, proposed_decode, accepted_checkpoint
    ):
        handle = super().materialize_next_frontier(
            config, prior, proposed_decode, accepted_checkpoint
        )
        stale = replace(handle.frontier, images=())
        artifact = _write_artifact(Path(handle.artifact_path), asdict(stale))
        return replace(handle, artifact_sha256=artifact.sha256, frontier=stale)


def test_empty_runtime_reported_frontier_cannot_claim_same_decode_lineage(
    tmp_path: Path,
) -> None:
    config = replace(_config(), output_root=str(tmp_path / "empty-frontier"))
    runtime = _StaleFrontierRuntime(Path(config.output_root), accept=True)
    before = runtime.transaction.state_digest()

    with pytest.raises(OnPolicySuccessorError, match="same clean decode"):
        run_on_policy_loop(config, runtime=runtime, execute_authorized=True)

    assert runtime.transaction.state_digest() == before
    assert "transaction:accept" not in runtime.events


def test_execute_cli_dry_run_never_requires_a_runtime() -> None:
    receipt = execute_cli(
        config_path=CONFIG_ROOT / "02_o_first_safe.yaml",
        repo_root=ROOT,
        execute=False,
        authority=False,
        runtime=None,
    )

    assert receipt["mode"] == "dry_run"
    assert all(count == 0 for count in receipt["actions"].values())
