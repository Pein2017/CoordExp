"""Wave-3 tasks 3.3/3.4: real two-rank gloo parity and fail-closed gating.

Every test here runs `world_size=2` over a real `gloo` CPU process group with
a real `DistributedDataParallel` module, because the thing under test IS the
backend's mean-gradient reduction: a hand-rolled all-reduce would only test
the test. The rank shards carry **unequal** eligible-segment counts (rank 0
contributes 1 segment, rank 1 contributes 3), so every term's planned-step
`segment_balanced` denominator is genuinely global and every rank's
differentiable contribution is a genuine partial.

Declared numeric tolerances
---------------------------
* **distributed vs world-size-one** (`_GRAD_RTOL` / `_GRAD_ATOL`): the two
  paths sum the same fp32 terms in different orders (per-rank partial sums
  through a gloo all-reduce, versus one in-process accumulation), so they are
  equal only up to fp32 reassociation. `rtol=1e-5, atol=1e-6` is ~2 orders of
  magnitude above the observed deltas on this fixture and ~2 orders below the
  smallest quantity being compared.
* **gate ablation vs base-CE-only** (`torch.equal`, bitwise): same inputs,
  same rank shards, same collective schedule, and the ablation contributes no
  operation to the objective graph at all (Wave-2 `detached_diagnostic`), so
  the two objectives are the *same sequence of floating point operations*.
  Anything short of bitwise equality would mean the diagnostic perturbed the
  optimized objective.
"""

from __future__ import annotations

from contextlib import nullcontext
from datetime import timedelta
import multiprocessing as mp
from queue import Empty
import socket
import traceback
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist

from src.coordinate_targets import CoordinateLossTarget
from src.losses import LossContext, LossRunner, TokenVocabularyGroups
from src.losses.coord_gaussian_rps import CoordGaussianRPSLoss
from src.packing.planner import PackedSegment
from src.supervision import TokenAtom, TokenSequence


_WORLD_SIZE = 2
_PROCESS_GROUP_TIMEOUT_SECONDS = 30
_JOIN_TIMEOUT_SECONDS = 240
_FEATURE_DIM = 4
_VOCAB_SIZE = 8
_COORDINATE_IDS = (1, 2, 3, 4)
_GATE_GROUPS = ("desc_text", "schema", "coordinate", "eos")
_LEARNING_RATE = 0.1

# See the module docstring for why these values, and why the ablation arm
# uses bitwise equality instead.
_GRAD_RTOL = 1e-5
_GRAD_ATOL = 1e-6

_ARMS: dict[str, tuple[float, float, float]] = {
    # arm -> (base_ce weight, token_type_gate weight, coord_gaussian_rps weight)
    "base_ce_only": (1.0, 0.0, 0.0),
    "enabled_gate": (1.0, 0.1, 0.0),
    "coord_auxiliary": (1.0, 0.1, 0.5),
}


# --------------------------------------------------------------------------
# Deterministic rank shards
# --------------------------------------------------------------------------


def _vocab_groups() -> TokenVocabularyGroups:
    return TokenVocabularyGroups(
        vocab_size=_VOCAB_SIZE,
        desc_text=(0,),
        schema=(5,),
        coordinate=_COORDINATE_IDS,
        eos=(6,),
        blocked=(7,),
    )


def _coordinate_atom(
    *,
    pack_index: int,
    segment_index: int,
    target_position: int,
    token_id: int,
    slot_index: int,
) -> TokenAtom:
    return TokenAtom(
        pack_index=pack_index,
        segment_index=segment_index,
        example_index=segment_index,
        example_id=f"pack{pack_index}-seg{segment_index}",
        target_position=target_position,
        token_id=token_id,
        token_type="coordinate",
        text=f"<|coord_{token_id - 1}|>",
        logical_target_position=target_position,
        object_id="obj-1",
        field=f"bbox[{slot_index}]",
        source="wave3",
        coordinate_target=CoordinateLossTarget(bbox=(2, 3, 8, 13), slot_index=slot_index),
    )


def _plain_atom(
    *,
    pack_index: int,
    segment_index: int,
    target_position: int,
    token_id: int,
    token_type: str,
) -> TokenAtom:
    return TokenAtom(
        pack_index=pack_index,
        segment_index=segment_index,
        example_index=segment_index,
        example_id=f"pack{pack_index}-seg{segment_index}",
        target_position=target_position,
        token_id=token_id,
        token_type=token_type,
        text="x",
        logical_target_position=target_position,
        source="wave3",
    )


def _segment(*, pack_index: int, segment_index: int, start: int, end: int) -> PackedSegment:
    return PackedSegment(
        pack_index=pack_index,
        segment_index=segment_index,
        example_index=segment_index,
        example_id=f"pack{pack_index}-seg{segment_index}",
        start=start,
        end=end,
    )


def _shard_token_sequence(shard_index: int) -> TokenSequence:
    """Shard 0 owns 1 eligible segment; shard 1 owns 3 (unequal by design)."""

    if shard_index == 0:
        return TokenSequence(
            pack_index=0,
            input_ids=tuple(0 for _ in range(4)),
            segments=(_segment(pack_index=0, segment_index=0, start=0, end=4),),
            atoms=(
                _plain_atom(
                    pack_index=0,
                    segment_index=0,
                    target_position=1,
                    token_id=0,
                    token_type="desc_text",
                ),
                _coordinate_atom(
                    pack_index=0,
                    segment_index=0,
                    target_position=2,
                    token_id=2,
                    slot_index=0,
                ),
            ),
            spans=(),
        )
    return TokenSequence(
        pack_index=1,
        input_ids=tuple(0 for _ in range(6)),
        segments=(
            _segment(pack_index=1, segment_index=0, start=0, end=2),
            _segment(pack_index=1, segment_index=1, start=2, end=4),
            _segment(pack_index=1, segment_index=2, start=4, end=6),
        ),
        atoms=(
            _plain_atom(
                pack_index=1,
                segment_index=0,
                target_position=1,
                token_id=6,
                token_type="eos",
            ),
            _coordinate_atom(
                pack_index=1,
                segment_index=1,
                target_position=3,
                token_id=3,
                slot_index=1,
            ),
            _coordinate_atom(
                pack_index=1,
                segment_index=2,
                target_position=5,
                token_id=4,
                slot_index=2,
            ),
        ),
        spans=(),
    )


def _shard_features(shard_index: int) -> torch.Tensor:
    """Deterministic per-shard model inputs, identical in every process."""

    token_sequence = _shard_token_sequence(shard_index)
    length = len(token_sequence.input_ids)
    values = torch.linspace(
        -1.0 + 0.37 * shard_index,
        1.0 - 0.19 * shard_index,
        steps=length * _FEATURE_DIM,
        dtype=torch.float32,
    )
    return values.reshape(1, length, _FEATURE_DIM)


def _build_model() -> torch.nn.Module:
    torch.manual_seed(20260820)
    return torch.nn.Linear(_FEATURE_DIM, _VOCAB_SIZE, bias=True).float()


def _runner_for_arm(arm: str) -> LossRunner:
    base_weight, gate_weight, coord_weight = _ARMS[arm]
    coord_term = (
        CoordGaussianRPSLoss(
            gaussian_weight=0.5,
            rps_weight=0.2,
            temperature=1.0,
            gaussian_r95_axis_fraction=0.5,
            gaussian_r95_cap_bins=4,
            gaussian_r95_min_bins=1,
            gaussian_r95_fallback_bins=4,
        )
        if coord_weight > 0.0
        else None
    )
    return LossRunner(
        base_ce_weight=base_weight,
        token_type_gate_weight=gate_weight,
        token_type_gate_groups=_GATE_GROUPS,
        coord_gaussian_rps_weight=coord_weight,
        coord_gaussian_rps=coord_term,
    )


# --------------------------------------------------------------------------
# Runtime plumbing
# --------------------------------------------------------------------------


class _Accelerator:
    """Minimal Accelerate stand-in over an already-prepared module."""

    def __init__(self, *, process_index: int, num_processes: int) -> None:
        self.process_index = process_index
        self.num_processes = num_processes
        self.device = torch.device("cpu")
        self.is_main_process = process_index == 0
        self.distributed_type = SimpleNamespace(
            name="MULTI_GPU" if num_processes > 1 else "NO"
        )
        self.gradient_accumulation_steps = 1
        self.mixed_precision = "no"
        self.scaler = None

    def prepare(self, *objects: Any) -> tuple[Any, ...]:
        return objects

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def no_sync(self, model: Any) -> Any:
        no_sync = getattr(model, "no_sync", None)
        return no_sync() if callable(no_sync) else nullcontext()


def _train_runtime(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    world_size: int,
    rank: int,
    gatherer: Any | None,
) -> Any:
    from src.config.models import RuntimeBatchResolution, RuntimeConfig
    from src.runtime.train_runtime import TrainRuntime

    return TrainRuntime(
        runtime_config=RuntimeConfig.model_validate(
            {"seed": 17, "determinism": {"mode": "legacy"}}
        ),
        runtime_batch=RuntimeBatchResolution(
            world_size=world_size,
            effective_batch_size=world_size,
            resolved_grad_accum_steps=1,
        ),
        model=model,
        optimizer=optimizer,
        scheduler=None,
        expected_mixed_precision="no",
        accelerator=_Accelerator(process_index=rank, num_processes=world_size),
        rank_report_gatherer=gatherer,
    )


def _loss_context(model: torch.nn.Module, shard_index: int) -> LossContext:
    logits = model(_shard_features(shard_index))
    return LossContext(
        logits=logits,
        token_sequence=_shard_token_sequence(shard_index),
        vocab_groups=_vocab_groups(),
    )


def _run_planned_step(
    *,
    runner: LossRunner,
    runtime: Any,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    shard_indices: tuple[int, ...],
    world_size: int,
    rank: int,
    objective: str = "bundle",
    gather_metrics: bool = False,
) -> dict[str, Any]:
    """One planned step through the production runtime boundary."""

    planned_step_id = 1
    optimizer.zero_grad(set_to_none=True)
    denominator_gatherer = (
        (
            lambda payload: runtime.gather_loss_denominators(
                payload, planned_step_id=planned_step_id
            )
        )
        if world_size > 1
        else None
    )
    token_sequences = tuple(_shard_token_sequence(index) for index in shard_indices)
    if world_size > 1:
        plan = runner.prepare_planned_step(
            token_sequences,
            denominator_gatherer=denominator_gatherer,
            world_size=world_size,
            rank=rank,
        )
    else:
        plan = runner.prepare_planned_step(token_sequences)

    micro_artifacts: list[dict[str, Any]] = []
    pre_decision = None
    for local_index, shard_index in enumerate(shard_indices):
        sync_gradients = local_index == len(shard_indices) - 1
        with runtime.accumulation_context(sync_gradients=sync_gradients):
            context = _loss_context(model, shard_index)
            bundle = runner.compute_micro_step(
                context, plan, local_micro_step_index=local_index
            )
            micro_artifacts.append(bundle.to_artifact_dict())
            pre_decision = runtime.pre_backward(
                bundle, planned_step_id=planned_step_id
            )
            if not pre_decision.should_call_backward:
                break
            if objective == "base_only":
                # Independent base-CE-only objective for the gate-ablation
                # comparison: nothing but the protected base term.
                loss = bundle.term_by_name("base_ce").backward_contribution
            else:
                loss = bundle.backward_loss
            runtime.backward(
                loss,
                planned_step_id=planned_step_id,
                sync_gradients=sync_gradients,
            )

    finalized = runner.finalize_planned_step(tuple(micro_artifacts), plan)
    result: dict[str, Any] = {
        "pre_decision": pre_decision.to_artifact_dict(),
        "finalized_metrics": dict(finalized["metrics"]),
        "finalized_total_loss": float(finalized["total_loss"]),
        "finalized_terms": {
            str(term["name"]): {
                "raw_loss": float(term["raw_loss"]),
                "weighted_loss": float(term["weighted_loss"]),
                "backward_contribution": float(term["backward_contribution"]),
                "eligible_segment_count": int(
                    term["denominator"]["eligible_segment_count"]
                ),
            }
            for term in finalized["terms"]
        },
        "accuracy_stats": dict(finalized["accuracy_stats"]),
        "denominators": {
            name: denominator.eligible_segment_count
            for name, denominator in plan.denominators.items()
        },
        "backend_gradient_scale": float(plan.backend_gradient_scale),
    }
    if pre_decision is not None and pre_decision.should_call_backward:
        post_decision = runtime.post_backward(planned_step_id=planned_step_id)
        result["post_decision"] = post_decision.to_artifact_dict()
        result["gradients"] = [
            None if parameter.grad is None else parameter.grad.detach().clone()
            for parameter in model.parameters()
        ]
        if post_decision.should_call_optimizer_step:
            runtime.clip_gradients(planned_step_id=planned_step_id)
            runtime.optimizer_step(planned_step_id=planned_step_id)
    else:
        result["gradients"] = [
            None if parameter.grad is None else parameter.grad.detach().clone()
            for parameter in model.parameters()
        ]
    result["parameters"] = [
        parameter.detach().clone() for parameter in model.parameters()
    ]
    if gather_metrics:
        result["reduced_metrics"] = dict(
            runtime.gather_metrics(
                finalized["metrics"],
                planned_step_id=planned_step_id,
                split="train",
                accuracy_stats=finalized["accuracy_stats"],
            )["metrics"]
        )
    return result


def _reference_world_size_one(arm: str, *, objective: str = "bundle") -> dict[str, Any]:
    """The equivalent single-rank batch: both shards as one planned step."""

    model = _build_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=_LEARNING_RATE)
    runtime = _train_runtime(
        model=model, optimizer=optimizer, world_size=1, rank=0, gatherer=None
    )
    return _run_planned_step(
        runner=_runner_for_arm(arm),
        runtime=runtime,
        model=runtime.model,
        optimizer=runtime.optimizer,
        shard_indices=(0, 1),
        world_size=1,
        rank=0,
        objective=objective,
        gather_metrics=True,
    )


# --------------------------------------------------------------------------
# Distributed workers
# --------------------------------------------------------------------------


def _init_process_group(rank: int, port: int) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=_WORLD_SIZE,
        timeout=timedelta(seconds=_PROCESS_GROUP_TIMEOUT_SECONDS),
    )


def _destroy_process_group_best_effort() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _distributed_run(*, arm: str, rank: int, objective: str = "bundle") -> dict[str, Any]:
    from src.training.control_plane import _build_rank_report_gatherer

    model = torch.nn.parallel.DistributedDataParallel(_build_model())
    optimizer = torch.optim.SGD(model.parameters(), lr=_LEARNING_RATE)
    gatherer = _build_rank_report_gatherer(_WORLD_SIZE)
    assert gatherer is not None
    try:
        runtime = _train_runtime(
            model=model,
            optimizer=optimizer,
            world_size=_WORLD_SIZE,
            rank=rank,
            gatherer=gatherer,
        )
        return _run_planned_step(
            runner=_runner_for_arm(arm),
            runtime=runtime,
            model=runtime.model,
            optimizer=runtime.optimizer,
            shard_indices=(rank,),
            world_size=_WORLD_SIZE,
            rank=rank,
            objective=objective,
            gather_metrics=True,
        )
    finally:
        close = getattr(gatherer, "close", None)
        if callable(close):
            close()


def _parity_worker(rank: int, port: int, output: mp.Queue) -> None:
    try:
        _init_process_group(rank, port)
        for arm in _ARMS:
            distributed = _distributed_run(arm=arm, rank=rank)
            reference = _reference_world_size_one(arm)

            assert distributed["backend_gradient_scale"] == float(_WORLD_SIZE)
            # Unequal rank shards, one shared global denominator per term.
            assert distributed["denominators"]["base_ce"] == 4
            assert reference["denominators"]["base_ce"] == 4
            assert distributed["denominators"]["token_type_gate"] == 4
            if arm == "coord_auxiliary":
                assert distributed["denominators"]["coord_gaussian_rps"] == 3
                assert reference["denominators"]["coord_gaussian_rps"] == 3
            else:
                assert "coord_gaussian_rps" not in distributed["denominators"]

            # Parameter gradients match the world-size-one batch.
            _assert_tensor_lists_close(
                distributed["gradients"], reference["gradients"], arm=arm, what="grad"
            )
            # ... and so does the state after exactly one optimizer update.
            _assert_tensor_lists_close(
                distributed["parameters"],
                reference["parameters"],
                arm=arm,
                what="param",
            )
            assert (
                distributed["post_decision"]["optimizer_update_status"]
                == reference["post_decision"]["optimizer_update_status"]
                == "ready_to_step"
            )
            # Reduced semantic telemetry matches too: no backend factor
            # survives into the row, at either world size.
            #
            # `_RANK_LOCAL_SHARD_METRIC_KEYS` are excluded because they are
            # rank-local shard descriptors that PRE-DATE this change and are
            # untouched by it: `count/packs`/`count/examples` count this
            # rank's own micro-steps, and `token_weighted_diag` is a
            # per-rank token mean that the train reducer averages unweighted.
            # Every objective, accuracy, finite and globally-denominated
            # count key IS compared.
            compared = {
                key: value
                for key, value in reference["reduced_metrics"].items()
                if not _is_rank_local_shard_metric_key(key)
            }
            assert any(key.startswith("loss/") for key in compared)
            for key, reference_value in compared.items():
                observed = distributed["reduced_metrics"][key]
                assert abs(observed - reference_value) <= max(
                    _GRAD_ATOL, _GRAD_RTOL * abs(reference_value)
                ), (arm, key, observed, reference_value)
        output.put((rank, "ok"))
    except BaseException:
        output.put((rank, traceback.format_exc()))
        raise
    finally:
        _destroy_process_group_best_effort()


def _gate_ablation_worker(rank: int, port: int, output: mp.Queue) -> None:
    try:
        _init_process_group(rank, port)
        ablation = _distributed_run(arm="base_ce_only", rank=rank)
        base_only = _distributed_run(
            arm="base_ce_only", rank=rank, objective="base_only"
        )

        gate = ablation["finalized_terms"]["token_type_gate"]
        base = ablation["finalized_terms"]["base_ce"]
        # The diagnostic really ran (finite, non-trivial) and stayed at
        # weighted zero without an objective edge.
        assert gate["raw_loss"] != 0.0
        assert gate["raw_loss"] == gate["raw_loss"]  # finite: not NaN
        assert gate["weighted_loss"] == 0.0
        assert gate["backward_contribution"] == 0.0
        assert ablation["finalized_total_loss"] == base["weighted_loss"]
        assert ablation["finalized_metrics"]["loss/total"] == pytest.approx(
            ablation["finalized_metrics"]["loss/base_ce"]
        )
        assert ablation["pre_decision"]["finite_status"] == "finite"

        # Bitwise identity against the base-CE-only objective (see the module
        # docstring for why bitwise is the right bar here).
        assert base_only["finalized_terms"]["base_ce"]["raw_loss"] == base["raw_loss"]
        for observed, expected in zip(
            ablation["gradients"], base_only["gradients"], strict=True
        ):
            assert observed is not None and expected is not None
            assert torch.equal(observed, expected)
        for observed, expected in zip(
            ablation["parameters"], base_only["parameters"], strict=True
        ):
            assert torch.equal(observed, expected)
        assert (
            ablation["post_decision"]["optimizer_update_status"]
            == base_only["post_decision"]["optimizer_update_status"]
            == "ready_to_step"
        )
        assert ablation["pre_decision"]["all_ranks_safe"] is True
        output.put((rank, "ok"))
    except BaseException:
        output.put((rank, traceback.format_exc()))
        raise
    finally:
        _destroy_process_group_best_effort()


def _non_finite_gate_worker(rank: int, port: int, output: mp.Queue) -> None:
    """Inject a non-finite gate diagnostic on rank 1 only (entry-audit F-2)."""

    from src.losses.token_type_gate import TokenTypeGateLoss

    original_per_atom_loss = TokenTypeGateLoss.per_atom_loss
    try:
        _init_process_group(rank, port)

        def poisoned(self: Any, context: Any) -> torch.Tensor:
            losses = original_per_atom_loss(self, context)
            if rank == 1 and losses.numel() > 0:
                poisoned_losses = losses.clone()
                poisoned_losses[0] = float("nan")
                return poisoned_losses
            return losses

        TokenTypeGateLoss.per_atom_loss = poisoned  # type: ignore[method-assign]
        result = _distributed_run(arm="base_ce_only", rank=rank)

        decision = result["pre_decision"]
        # ONE unsafe decision, converged on EVERY rank -- including rank 0,
        # whose own shard is entirely finite.
        assert decision["all_ranks_safe"] is False
        assert decision["should_call_backward"] is False
        assert decision["should_call_optimizer_step"] is False
        assert decision["finite_status"] == "non_finite"
        assert decision["optimizer_update_status"] == "skipped_non_finite_scalar"
        assert decision["ranks"] == [0, 1]
        assert decision["reason_codes"] == ["rank1:non_finite_scalar"]
        # Backward never ran on either rank, so no gradient exists and no
        # optimizer update could have been applied.
        assert all(gradient is None for gradient in result["gradients"])
        assert "post_decision" not in result
        # The unsafe term is the zero-WEIGHTED protected diagnostic: its
        # weighted value is a literal 0.0, so only RAW-keyed finite
        # derivation can see it (entry-audit F-2).
        rank_diagnostics = decision["rank_diagnostics"]
        unsafe = next(item for item in rank_diagnostics if item["rank"] == 1)
        assert unsafe["terms"]["token_type_gate"]["finite"] is False
        assert unsafe["terms"]["token_type_gate"]["weighted_loss"] == 0.0
        assert unsafe["terms"]["base_ce"]["finite"] is True
        output.put((rank, "ok"))
    except BaseException:
        output.put((rank, traceback.format_exc()))
        raise
    finally:
        TokenTypeGateLoss.per_atom_loss = original_per_atom_loss  # type: ignore[method-assign]
        _destroy_process_group_best_effort()


_RANK_LOCAL_SHARD_METRIC_KEYS = frozenset({"count/packs", "count/examples"})


def _is_rank_local_shard_metric_key(key: str) -> bool:
    return key in _RANK_LOCAL_SHARD_METRIC_KEYS or key.endswith("/token_weighted_diag")


def _assert_tensor_lists_close(
    observed: list[torch.Tensor | None],
    expected: list[torch.Tensor | None],
    *,
    arm: str,
    what: str,
) -> None:
    assert len(observed) == len(expected)
    for index, (left, right) in enumerate(zip(observed, expected, strict=True)):
        assert left is not None and right is not None, (arm, what, index)
        assert torch.allclose(left, right, rtol=_GRAD_RTOL, atol=_GRAD_ATOL), (
            arm,
            what,
            index,
            float((left - right).abs().max()),
        )


# --------------------------------------------------------------------------
# Drivers
# --------------------------------------------------------------------------


def _have_gloo() -> bool:
    if not dist.is_available():
        return False
    available = getattr(dist, "is_gloo_available", None)
    return True if available is None else bool(available())


def _find_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _drive_two_ranks(target: Any) -> None:
    context = mp.get_context("spawn")
    port = _find_free_tcp_port()
    output: mp.Queue = context.Queue()
    processes = [
        context.Process(target=target, args=(rank, port, output), daemon=False)
        for rank in range(_WORLD_SIZE)
    ]
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join(timeout=_JOIN_TIMEOUT_SECONDS)
        alive = [process for process in processes if process.is_alive()]
        if alive:
            for process in alive:
                process.terminate()
            pytest.fail(
                "two-rank worker hung; "
                f"alive_pids={[process.pid for process in alive]}"
            )
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)

    messages: dict[int, str] = {}
    while True:
        try:
            rank, message = output.get_nowait()
        except Empty:
            break
        messages[int(rank)] = str(message)
    for rank in range(_WORLD_SIZE):
        assert messages.get(rank) == "ok", messages.get(rank, "<no report>")
    assert [process.exitcode for process in processes] == [0] * _WORLD_SIZE


@pytest.mark.skipif(not _have_gloo(), reason="requires torch.distributed gloo backend")
def test_two_rank_gradients_and_update_match_world_size_one_for_every_arm() -> None:
    """Task 3.3: base CE only, enabled gate, and positive-weight coord aux."""

    _drive_two_ranks(_parity_worker)


@pytest.mark.skipif(not _have_gloo(), reason="requires torch.distributed gloo backend")
def test_two_rank_gate_ablation_preserves_the_base_ce_only_step_exactly() -> None:
    """Task 3.4 (first half): finite diagnostic changes nothing."""

    _drive_two_ranks(_gate_ablation_worker)


@pytest.mark.skipif(not _have_gloo(), reason="requires torch.distributed gloo backend")
def test_two_rank_non_finite_gate_diagnostic_skips_backward_on_every_rank() -> None:
    """Task 3.4 (second half) / entry-audit F-2 distributed cover."""

    _drive_two_ranks(_non_finite_gate_worker)
