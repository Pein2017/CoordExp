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
from scripts.research.human13_frontier_selection import (
    CandidatePath,
    CandidateScore,
    ContinuationOutcome,
)
from scripts.research.human13_on_policy_live import BehaviorGateObservation
from scripts.research.human13_training_transaction import (
    TrainingStateTransaction,
    UpdateCounter,
)
from scripts.research.train_human13_on_policy_successor import (
    CandidateSelectionReceipt,
    CandidateKey,
    CleanDecodeReceipt,
    DurableArtifact,
    FrontierHandle,
    OneUpdateReceipt,
    OnPolicyArmConfig,
    OnPolicySuccessorError,
    RuntimeIdentity,
    RetryChildLedgerHandle,
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


def _outcome(owner_id: str) -> ContinuationOutcome:
    return ContinuationOutcome(
        owner_id=owner_id,
        hf_barrier=1.0,
        protected_coverable=True,
        unique_owner_delta=1,
        termination_status="natural_im_end",
        cap_hit=False,
        duplicate_increase=0,
        malformed_increase=0,
        row_count=2,
        generated_tokens=8,
    )


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


def _decodes(
    checkpoint: CheckpointIdentity, *, parser_status: str = "accepted"
) -> tuple[CurrentDecode, ...]:
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
            parser_status=parser_status,
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
        parser_status: str = "accepted",
    ):
        self.output_root = output_root
        self.accept = accept
        self.endless = endless
        self.fail_at = fail_at
        self.parser_status = parser_status
        self.events: list[str] = []
        self.published: list[str] = []
        self.written_receipts = []
        self.written_receipt_artifacts: list[DurableArtifact] = []
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

    def select_candidate(
        self,
        config: OnPolicyArmConfig,
        frontier: FrontierHandle,
        *,
        attempt_index: int = 0,
        retry_child: RetryChildLedgerHandle | None = None,
    ):
        del config
        excluded_candidates = (
            () if retry_child is None else retry_child.ledger.excluded_candidates
        )
        excluded_alias_ids = tuple(item.alias_id for item in excluded_candidates)
        self.events.append(
            "candidate:"
            f"{frontier.frontier.iteration}:{attempt_index}:"
            f"{','.join(excluded_alias_ids)}"
        )
        if not self.endless and frontier.frontier.iteration > 0:
            return None
        shortlisted = tuple(
            _candidate(owner_id)
            for owner_id in ("h0", "h1", "h2", "h3")
            if CandidateKey(7, owner_id, f"alias-{owner_id}")
            not in set(excluded_candidates)
        )
        if not shortlisted:
            return None
        candidate_ledger = _write_artifact(
            self.output_root
            / "iterations"
            / f"attempt-{attempt_index:03d}"
            / "candidate-scoring.json",
            {
                "frontier_sha256": frontier.artifact_sha256,
                "excluded_candidates": [asdict(item) for item in excluded_candidates],
                "selected_alias_id": shortlisted[0].path.alias_id,
            },
        )
        return CandidateSelectionReceipt(
            decision_surface="hf_fp32_sdpa_batch1",
            source_frontier_sha256=frontier.artifact_sha256,
            candidate_ledger_path=(
                candidate_ledger.path
                if retry_child is None
                else retry_child.artifact_path
            ),
            candidate_ledger_sha256=(
                candidate_ledger.sha256
                if retry_child is None
                else retry_child.artifact_sha256
            ),
            eligible_owner_count=len(shortlisted),
            shortlisted=shortlisted,
            forced_continuation_count=len(shortlisted),
            selected=shortlisted[0],
            continuation_projection_sha256s=tuple(
                f"{index + 1:x}" * 64 for index in range(len(shortlisted))
            ),
            continuation_outcomes=tuple(
                _outcome(item.path.owner_id) for item in shortlisted
            ),
            attempt_index=attempt_index,
            selection_evidence_path=candidate_ledger.path,
            selection_evidence_sha256=candidate_ledger.sha256,
            excluded_candidates=excluded_candidates,
            retry_depth=0 if retry_child is None else retry_child.ledger.retry_depth,
        )

    def apply_one_update(self, config, frontier, selection):
        del config
        self.events.append(f"update:{frontier.frontier.iteration}")
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.model(torch.ones((1, 1))).sum()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.counter.value += 1
        return OneUpdateReceipt(
            ledger_sha256=selection.candidate_ledger_sha256,
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
        decodes = _decodes(checkpoint, parser_status=self.parser_status)
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
        self.written_receipt_artifacts.append(artifact)
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


class _RngConsumingRollbackRuntime(_Runtime):
    def clean_decode(self, config, checkpoint, *, purpose: str):
        receipt = super().clean_decode(config, checkpoint, purpose=purpose)
        if purpose == "rollback_reproduction":
            torch.rand(1)
        return receipt


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


@pytest.mark.parametrize(
    "mutation",
    (
        lambda receipt: replace(
            receipt, continuation_projection_sha256s=("1" * 64, "1" * 64)
        ),
        lambda receipt: replace(
            receipt,
            continuation_outcomes=(
                replace(receipt.continuation_outcomes[0], owner_id="wrong"),
                receipt.continuation_outcomes[1],
            ),
        ),
        lambda receipt: replace(
            receipt,
            continuation_outcomes=(
                replace(receipt.continuation_outcomes[0], cap_hit=True),
                receipt.continuation_outcomes[1],
            ),
        ),
        lambda receipt: replace(receipt, selection_evidence_sha256="e" * 64),
    ),
)
def test_selection_requires_unique_path_aligned_eligible_continuation_evidence(
    tmp_path: Path,
    mutation,
) -> None:
    config = replace(_config(), output_root=str(tmp_path / "bad-selection"))
    runtime = _Runtime(Path(config.output_root), accept=True)
    frontier = runtime.initial_frontier(config)
    selection = runtime.select_candidate(
        config, frontier, attempt_index=0, retry_child=None
    )
    assert selection is not None

    with pytest.raises(OnPolicySuccessorError, match="candidate selection"):
        from scripts.research.train_human13_on_policy_successor import run_one_iteration

        run_one_iteration(
            config,
            runtime=runtime,
            frontier=frontier,
            selection=mutation(selection),
            attempt_index=0,
        )


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


@pytest.mark.parametrize(
    "parser_status",
    ("accepted", "accepted_with_drops", "empty", "all_spans_dropped"),
)
def test_clean_decode_preserves_every_canonical_frontier_parser_status(
    tmp_path: Path,
    parser_status: str,
) -> None:
    config = replace(_config(), output_root=str(tmp_path / parser_status))
    runtime = _Runtime(
        Path(config.output_root), accept=True, parser_status=parser_status
    )

    receipt = run_on_policy_loop(config, runtime=runtime, execute_authorized=True)

    assert receipt.accepted_update_count == 1
    assert runtime.written_receipts[0].decision == "accepted"


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


def test_reject_continues_past_three_while_an_eligible_candidate_remains(
    tmp_path: Path,
) -> None:
    config = replace(_config(), output_root=str(tmp_path / "rejected-run"))
    runtime = _Runtime(Path(config.output_root), accept=False)
    before = runtime.model.weight.detach().clone()

    receipt = run_on_policy_loop(config, runtime=runtime, execute_authorized=True)

    assert receipt.stop_reason == "repeated_rejection_no_eligible_candidate"
    assert receipt.attempted_update_count == 4
    assert receipt.accepted_update_count == 0
    assert runtime.counter.value == 0
    assert torch.equal(runtime.model.weight, before)
    assert runtime.published == []
    assert runtime.events.index("transaction:reject") < runtime.events.index(
        "decode:rollback_reproduction"
    )
    assert list((Path(config.output_root) / "private").iterdir()) == []
    assert [attempt.selected_alias_id for attempt in receipt.attempts] == [
        "alias-h0",
        "alias-h1",
        "alias-h2",
        "alias-h3",
    ]
    assert [event for event in runtime.events if event.startswith("candidate:")] == [
        "candidate:0:0:",
        "candidate:0:1:alias-h0",
        "candidate:0:2:alias-h0,alias-h1",
        "candidate:0:3:alias-h0,alias-h1,alias-h2",
        "candidate:0:4:alias-h0,alias-h1,alias-h2,alias-h3",
    ]
    for attempt in receipt.attempts:
        assert attempt.accepted_checkpoint_path is None
        assert attempt.accepted_checkpoint_sha256 is None
        assert attempt.next_ledger_sha256 is None
        assert attempt.transaction_before_sha256 == attempt.transaction_after_sha256

    first, second = receipt.attempts[:2]
    child_path = Path(second.ledger_path)
    assert child_path.parent.name == "candidate-ledgers"
    assert child_path.name == f"{second.ledger_sha256}.json"
    child = json.loads(child_path.read_text(encoding="utf-8"))
    assert child["source_frontier_sha256"] == "0" * 64
    assert child["parent_candidate_ledger_sha256"] == first.ledger_sha256
    assert (
        child["rejected_attempt_sha256"]
        == hashlib.sha256(
            Path(runtime.written_receipt_artifacts[0].path).read_bytes()
        ).hexdigest()
    )
    assert child["rollback_decode_sha256"] == first.rollback_decode_sha256
    assert child["restored_transaction_sha256"] == first.transaction_after_sha256
    assert child["excluded_candidates"] == [
        {"alias_id": "alias-h0", "image_id": 7, "owner_id": "h0"}
    ]
    grandchild = json.loads(Path(receipt.attempts[2].ledger_path).read_text())
    expected_evidence = (
        Path(config.output_root)
        / "iterations"
        / "attempt-001"
        / "candidate-scoring.json"
    )
    assert grandchild["parent_selection_evidence_path"] == str(expected_evidence)
    assert (
        grandchild["parent_selection_evidence_sha256"]
        == hashlib.sha256(expected_evidence.read_bytes()).hexdigest()
    )


def test_rollback_reproduction_rng_is_restored_before_retry_child(
    tmp_path: Path,
) -> None:
    config = replace(_config(), output_root=str(tmp_path / "rng-safe-retry"))
    runtime = _RngConsumingRollbackRuntime(Path(config.output_root), accept=False)
    before = runtime.transaction.state_digest()

    receipt = run_on_policy_loop(config, runtime=runtime, execute_authorized=True)

    assert receipt.stop_reason == "repeated_rejection_no_eligible_candidate"
    assert receipt.attempted_update_count == 4
    assert runtime.transaction.state_digest() == before


class _WrongRetryChildRuntime(_Runtime):
    def select_candidate(self, config, frontier, *, attempt_index=0, retry_child=None):
        selection = super().select_candidate(
            config,
            frontier,
            attempt_index=attempt_index,
            retry_child=retry_child,
        )
        if selection is None or retry_child is None:
            return selection
        forged = _write_artifact(
            self.output_root / "forged-child.json", {"forged": True}
        )
        return replace(
            selection,
            candidate_ledger_path=forged.path,
            candidate_ledger_sha256=forged.sha256,
        )


def test_retry_selection_must_use_the_exact_controller_child(tmp_path: Path) -> None:
    config = replace(_config(), output_root=str(tmp_path / "forged-retry"))
    runtime = _WrongRetryChildRuntime(Path(config.output_root), accept=False)

    with pytest.raises(OnPolicySuccessorError, match="retry child"):
        run_on_policy_loop(config, runtime=runtime, execute_authorized=True)


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


def test_execute_cli_dry_run_never_requires_a_runtime(tmp_path: Path) -> None:
    source = CONFIG_ROOT / "02_o_first_safe.yaml"
    fresh_config = tmp_path / source.name
    fresh_config.write_text(
        source.read_text(encoding="utf-8").replace(
            _config().output_root, str(tmp_path / "fresh-dry-run")
        ),
        encoding="utf-8",
    )
    receipt = execute_cli(
        config_path=fresh_config,
        repo_root=ROOT,
        execute=False,
        authority=False,
        runtime=None,
    )

    assert receipt["mode"] == "dry_run"
    assert all(count == 0 for count in receipt["actions"].values())
