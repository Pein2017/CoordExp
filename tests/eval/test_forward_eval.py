from __future__ import annotations

import json
import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from src.artifacts.run_writer import RunWriter
from src.common.errors import LossContractError, RuntimeContractError
from src.config.models import RuntimeBatchResolution, RuntimeConfig
from src.eval.forward import (
    EVAL_REDUCTION_DISJOINT_SHARD,
    EVAL_REDUCTION_REPLICATED,
    ForwardEvalObservation,
    ForwardEvalRunner,
    _finalize_disjoint_shard_scalars,
    _prepare_disjoint_shard_scalars,
    partition_eval_micro_steps_for_rank,
    resolve_active_eval_reduction_mode,
)
from src.losses import LossRunner, TokenVocabularyGroups
from src.losses.context import LossContext
from src.losses.runner import _weighted_average
from src.packing.planner import PackedSegment
from src.runtime.train_runtime import TrainRuntime
from src.supervision import TokenAtom, TokenSequence
from src.training.supervised_trainer import SupervisedMicroStep


def test_forward_eval_returns_one_pure_wide_observation_and_restores_mode(
    tmp_path: Path,
) -> None:
    log: list[str] = []
    model = FakeModel(log)
    runtime = FakeEvalRuntime(log)
    runner = ForwardEvalRunner(
        model=model,
        micro_step_stream=(
            _micro_step(0, encoded_examples=("a", "b")),
            _micro_step(1, encoded_examples=("c",)),
        ),
        loss_runner=StreamingFakeLossRunner(log),
        eval_source={"path": "tests/fixtures/eval.jsonl"},
        qwen_forward=_qwen_forward(log),
        loss_context_factory=_loss_context(log),
        runtime=runtime,
    )

    result = runner.run(planned_step_id=4, trigger_reasons=("milestone_40pct",))

    assert isinstance(result, ForwardEvalObservation)
    assert result.planned_step_id == 4
    assert result.split == "eval"
    assert result.trigger_reasons == ("milestone_40pct",)
    assert result.example_count == 3
    assert result.pack_count == 2
    assert result.accuracy_stats == {
        "top1_correct": 2,
        "top5_correct": 3,
        "atom_count": 4,
    }
    assert result.scalars == {
        "acc_top1": 0.5,
        "acc_top5": 0.75,
        "loss/total": 1.25,
    }
    assert result.to_logging_row() == {
        "step": 4,
        "split": "eval",
        "trigger_reasons": ["milestone_40pct"],
        "example_count": 3,
        "pack_count": 2,
        "accuracy_stats": {
            "top1_correct": 2,
            "top5_correct": 3,
            "atom_count": 4,
        },
        "acc_top1": 0.5,
        "acc_top5": 0.75,
        "loss/total": 1.25,
    }
    assert json.loads(json.dumps(result.to_logging_row())) == result.to_logging_row()
    assert model.training is True
    assert runtime.gathered == [(4, "eval", dict(FakeLossBundle.metrics))]
    assert list(tmp_path.iterdir()) == []
    assert log == [
        "model.eval",
        "streaming.prepare:2",
        "runtime.move:4:0",
        "forward:0:grad=False:training=False",
        "context:0",
        "streaming.loss:0",
        "runtime.move:4:1",
        "forward:1:grad=False:training=False",
        "context:1",
        "streaming.loss:1",
        "streaming.finalize:2",
        "runtime.gather:4:eval",
        "model.train:True",
    ]


def test_forward_eval_streaming_counts_and_wide_scalars() -> None:
    log: list[str] = []
    result = ForwardEvalRunner(
        model=FakeModel(log),
        micro_step_stream=(
            _micro_step(0, encoded_examples=("a", "b")),
            _micro_step(1, encoded_examples=("c",)),
            _micro_step(2, encoded_examples=("d",)),
        ),
        loss_runner=StreamingFakeLossRunner(log),
        eval_source={"path": "tests/fixtures/eval.jsonl"},
        qwen_forward=_qwen_forward(log),
        loss_context_factory=_loss_context(log),
    ).run(planned_step_id=136, trigger_reasons=("every_fraction:0.4",))

    assert result.example_count == 4
    assert result.pack_count == 3
    assert result.scalars["loss/total"] == pytest.approx(1.25)
    assert "streaming.prepare:3" in log
    assert "streaming.finalize:3" in log


def test_forward_eval_rejects_empty_stream_before_metric_reduction() -> None:
    log: list[str] = []
    runtime = FakeEvalRuntime(log)
    runner = ForwardEvalRunner(
        model=FakeModel(log),
        micro_step_stream=(),
        loss_runner=StreamingFakeLossRunner(log),
        eval_source={"path": "tests/fixtures/eval.jsonl"},
        qwen_forward=_qwen_forward(log),
        loss_context_factory=_loss_context(log),
        runtime=runtime,
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        runner.run(planned_step_id=1, trigger_reasons=("scheduled",))
    assert exc_info.value.code == "eval_forward.empty_stream"
    assert runtime.gathered == []


def test_forward_eval_leaves_nonfinite_values_for_writer_normalization() -> None:
    runner = ForwardEvalRunner(
        model=object(),
        micro_step_stream=(_micro_step(0),),
        loss_runner=NonfiniteLossRunner([]),
        eval_source={"path": "tests/fixtures/eval.jsonl"},
        qwen_forward=_qwen_forward([]),
        loss_context_factory=_loss_context([]),
    )

    row = runner.run(planned_step_id=2, trigger_reasons=("scheduled",)).to_logging_row()

    assert math.isnan(row["loss/total"])
    assert math.isinf(row["diagnostic/max_logit"])
    assert "non_finite_fields" not in row


def test_forward_eval_has_no_writer_or_event_coupling() -> None:
    import inspect
    import src.eval.forward as module

    source = inspect.getsource(module)
    assert "RunArtifactManager" not in source
    assert "MetricStreamEvent" not in source
    assert "write_" not in source
    assert "summary_path" not in source


def test_forward_eval_requires_explicit_source_before_consuming_stream() -> None:
    consumed = False

    def stream() -> Any:
        nonlocal consumed
        consumed = True
        yield _micro_step(0)

    runner = ForwardEvalRunner(
        model=object(),
        micro_step_stream=stream(),
        loss_runner=StreamingFakeLossRunner([]),
        eval_source=None,
        qwen_forward=_qwen_forward([]),
        loss_context_factory=_loss_context([]),
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        runner.run(planned_step_id=1, trigger_reasons=("scheduled",))
    assert exc_info.value.code == "eval_forward.source_required"
    assert consumed is False


def test_forward_eval_restores_model_mode_when_forward_raises() -> None:
    log: list[str] = []
    model = FakeModel(log)
    runner = ForwardEvalRunner(
        model=model,
        micro_step_stream=(_micro_step(0),),
        loss_runner=StreamingFakeLossRunner(log),
        eval_source={"path": "tests/fixtures/eval.jsonl"},
        qwen_forward=_raising_forward(log),
        loss_context_factory=_loss_context(log),
    )

    with pytest.raises(RuntimeError, match="synthetic forward failure"):
        runner.run(planned_step_id=4, trigger_reasons=("scheduled",))
    assert model.training is True
    assert log == [
        "model.eval",
        "streaming.prepare:1",
        "forward_raises:0:grad=False:training=False",
        "model.train:True",
    ]


def test_forward_eval_restores_model_mode_when_eval_raises() -> None:
    log: list[str] = []
    model = EvalRaisesModel(log)
    runner = ForwardEvalRunner(
        model=model,
        micro_step_stream=(_micro_step(0),),
        loss_runner=StreamingFakeLossRunner(log),
        eval_source={"path": "tests/fixtures/eval.jsonl"},
        qwen_forward=_qwen_forward(log),
        loss_context_factory=_loss_context(log),
    )

    with pytest.raises(RuntimeError, match="synthetic eval failure"):
        runner.run(planned_step_id=4, trigger_reasons=("scheduled",))
    assert model.training is True
    assert log == ["model.eval_raises", "model.train:True"]


@dataclass
class FakeForwardResult:
    logits: torch.Tensor
    receipt: dict[str, Any]


class FakeModel:
    def __init__(self, log: list[str]) -> None:
        self.log = log
        self.training = True

    def eval(self) -> None:
        self.log.append("model.eval")
        self.training = False

    def train(self, mode: bool = True) -> None:
        self.log.append(f"model.train:{mode}")
        self.training = mode


class EvalRaisesModel(FakeModel):
    def eval(self) -> None:
        self.log.append("model.eval_raises")
        self.training = False
        raise RuntimeError("synthetic eval failure")


class FakeEvalRuntime:
    def __init__(self, log: list[str]) -> None:
        self.log = log
        self.gathered: list[tuple[int, str, dict[str, float]]] = []

    def move_micro_step(
        self,
        micro_step: SupervisedMicroStep,
        *,
        planned_step_id: int,
        local_micro_step_index: int,
    ) -> SupervisedMicroStep:
        self.log.append(f"runtime.move:{planned_step_id}:{local_micro_step_index}")
        return micro_step

    def gather_metrics(
        self,
        metrics: dict[str, float],
        *,
        planned_step_id: int,
        split: str,
        accuracy_stats: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        self.log.append(f"runtime.gather:{planned_step_id}:{split}")
        self.gathered.append((planned_step_id, split, dict(metrics)))
        return {
            "metrics": dict(metrics),
            "accuracy_stats": dict(accuracy_stats or {}),
            "reduction": "single_rank",
        }


class FakeLossRunner:
    def __init__(self, log: list[str]) -> None:
        self.log = log

    def compute(self, contexts: tuple[Any, ...]) -> "FakeLossBundle":
        self.log.append(f"loss:{len(contexts)}:grad={torch.is_grad_enabled()}")
        return FakeLossBundle()


class FakeLossBundle:
    metrics = {"acc_top1": 0.5, "acc_top5": 0.75, "loss/total": 1.25}
    accuracy_stats = {"top1_correct": 1, "top5_correct": 1, "atom_count": 1}

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "total_loss": 1.25,
            "metrics": dict(self.metrics),
            "accuracy_stats": dict(self.accuracy_stats),
        }


class StreamingFakeLossRunner(FakeLossRunner):
    def compute(self, contexts: tuple[Any, ...]) -> FakeLossBundle:
        del contexts
        raise AssertionError("streaming eval path must not retain all contexts")

    def prepare_planned_step(
        self, micro_steps: tuple[SupervisedMicroStep, ...]
    ) -> dict[str, int]:
        self.log.append(f"streaming.prepare:{len(micro_steps)}")
        return {"micro_step_count": len(micro_steps)}

    def compute_micro_step(
        self, context: Any, plan: dict[str, int], *, local_micro_step_index: int
    ) -> FakeLossBundle:
        del context, plan
        self.log.append(f"streaming.loss:{local_micro_step_index}")
        return FakeLossBundle()

    def finalize_planned_step(
        self, micro_loss_artifacts: tuple[dict[str, Any], ...], plan: dict[str, int]
    ) -> dict[str, Any]:
        del plan
        self.log.append(f"streaming.finalize:{len(micro_loss_artifacts)}")
        return {
            "metrics": dict(FakeLossBundle.metrics),
            "accuracy_stats": {
                "top1_correct": 2,
                "top5_correct": 3,
                "atom_count": 4,
            },
        }


class NonfiniteLossRunner(StreamingFakeLossRunner):
    def finalize_planned_step(
        self, micro_loss_artifacts: tuple[dict[str, Any], ...], plan: dict[str, int]
    ) -> dict[str, Any]:
        del micro_loss_artifacts, plan
        return {
            "metrics": {
                "loss/total": float("nan"),
                "diagnostic/max_logit": float("inf"),
            }
        }


def _qwen_forward(log: list[str]):
    def forward(model: FakeModel, micro_step: SupervisedMicroStep) -> FakeForwardResult:
        index = int(str(micro_step.pack).split("-")[1])
        log.append(
            f"forward:{index}:grad={torch.is_grad_enabled()}:training={getattr(model, 'training', None)}"
        )
        return FakeForwardResult(torch.zeros(1, 2, 3), {"pack": index})

    return forward


def _raising_forward(log: list[str]):
    def forward(model: FakeModel, micro_step: SupervisedMicroStep) -> FakeForwardResult:
        index = int(str(micro_step.pack).split("-")[1])
        log.append(
            f"forward_raises:{index}:grad={torch.is_grad_enabled()}:training={getattr(model, 'training', None)}"
        )
        raise RuntimeError("synthetic forward failure")

    return forward


def _loss_context(log: list[str]):
    def factory(
        micro_step: SupervisedMicroStep, forward_result: FakeForwardResult
    ) -> dict[str, Any]:
        index = int(str(micro_step.pack).split("-")[1])
        log.append(f"context:{index}")
        return {"pack": micro_step.pack, "shape": tuple(forward_result.logits.shape)}

    return factory


def _micro_step(
    index: int, *, encoded_examples: tuple[str, ...] = ("example",)
) -> SupervisedMicroStep:
    return SupervisedMicroStep(
        pack=f"pack-{index}",
        encoded_examples=encoded_examples,
        position_inputs=f"positions-{index}",
        token_sequence=object(),
        vocab_groups=object(),
    )


# --- Wave 4: rank-sharded eval.forward with exact aggregation --------------
#
# These fixtures exercise the REAL LossRunner and REAL TrainRuntime (the
# same production reduction machinery training uses), driving two or more
# ranks concurrently through a small thread-based simulated collective so
# the exact same cross-rank gather/reduce code path is proven, not a
# reimplementation of it.


@dataclass(frozen=True)
class _FakePack:
    pack_index: int


def _groups() -> TokenVocabularyGroups:
    return TokenVocabularyGroups(
        vocab_size=8,
        desc_text=(7,),
        schema=(1, 2),
        coordinate=(3, 4),
        eos=(5,),
        blocked=(0, 6),
    )


def _wave4_logits(rows: tuple[tuple[float, ...], ...]) -> torch.Tensor:
    return torch.tensor((rows,), dtype=torch.float32)


def _wave4_segment(
    segment_index: int, start: int, end: int, *, pack_index: int
) -> PackedSegment:
    return PackedSegment(
        pack_index=pack_index,
        segment_index=segment_index,
        example_index=segment_index,
        example_id=f"pack{pack_index}-ex{segment_index}",
        start=start,
        end=end,
    )


def _wave4_atom(
    *,
    pack_index: int,
    segment_index: int,
    target_position: int,
    token_id: int,
    token_type: str = "desc_text",
) -> TokenAtom:
    return TokenAtom(
        pack_index=pack_index,
        segment_index=segment_index,
        example_index=segment_index,
        example_id=f"pack{pack_index}-ex{segment_index}",
        target_position=target_position,
        token_id=token_id,
        token_type=token_type,
        text="x",
        logical_target_position=target_position,
        object_id=None,
        field=None,
        source="unit",
        coordinate_target=None,
    )


def _wave4_context(
    logits: torch.Tensor,
    segments: tuple[PackedSegment, ...],
    atoms: tuple[TokenAtom, ...],
    *,
    pack_index: int,
) -> LossContext:
    pack_length = int(logits.shape[1])
    return LossContext(
        logits=logits,
        token_sequence=TokenSequence(
            pack_index=pack_index,
            input_ids=tuple(0 for _ in range(pack_length)),
            segments=segments,
            atoms=atoms,
            spans=(),
        ),
        vocab_groups=_groups(),
        logits_position_ids=None,
    )


def _wave4_micro_step(
    context: LossContext, *, num_examples: int
) -> SupervisedMicroStep:
    pack_index = context.token_sequence.pack_index
    return SupervisedMicroStep(
        pack=_FakePack(pack_index=pack_index),
        encoded_examples=tuple(f"pack{pack_index}-ex{i}" for i in range(num_examples)),
        position_inputs=None,
        token_sequence=context.token_sequence,
        vocab_groups=_groups(),
    )


_WAVE4_NUM_EXAMPLES_BY_PACK = {0: 2, 1: 1, 2: 2}


def _wave4_loss_runner() -> LossRunner:
    # Protected base CE is pinned to exactly 1.0 by the `forbid` zero policy
    # (enforced at LossRunner composition); the gate's 0.5 supplies the
    # non-trivial configured weight these reduction tests exercise.
    return LossRunner(
        base_ce_weight=1.0,
        token_type_gate_weight=0.5,
        token_type_gate_groups=("desc_text", "schema", "coordinate", "eos"),
    )


def _wave4_loss_runner_with_globally_unselected_gate_term() -> LossRunner:
    """`token_type_gate` configured against the valid V1 token type
    "schema" -- present only on pack 2's atom (see `_WAVE4_CONTEXTS_BY_PACK`)
    -- while this test's eval set is restricted to packs 0 and 1
    (`_WAVE4_PACKS_ZERO_AND_ONE`, whose atoms are `desc_text`/`eos`/
    `coordinate`, never `schema`). This is real production `LossRunner`
    configuration (`token_type_gate_groups` must be a non-empty closed set
    of V1 token types, per `_validate_token_type_groups` -- an empty group
    set is rejected outright, and every valid V1 type is used by SOME pack
    in the full 3-pack fixture, so no group is globally unused there), not a
    fixture shortcut: the gate term is still computed unconditionally every
    micro-step (`compute_micro_step` always calls
    `_compute_token_term_contribution` for `token_type_gate`), it just
    legitimately selects zero atoms on every pack in this restricted set, on
    every rank.
    """

    return LossRunner(
        base_ce_weight=1.0,
        token_type_gate_weight=0.5,
        token_type_gate_groups=("schema",),
    )


class _StaticQwenForward:
    """`qwen_forward` returns each pack's precomputed context unchanged."""

    def __init__(
        self, contexts_by_pack: Mapping[int, LossContext] | None = None
    ) -> None:
        self._contexts_by_pack = (
            _WAVE4_CONTEXTS_BY_PACK if contexts_by_pack is None else contexts_by_pack
        )

    def __call__(self, model: Any, micro_step: SupervisedMicroStep) -> LossContext:
        assert isinstance(micro_step.token_sequence, TokenSequence)
        return self._contexts_by_pack[micro_step.pack.pack_index]


def _wave4_loss_context_factory(
    micro_step: SupervisedMicroStep, forward_result: Any
) -> Any:
    del micro_step
    return forward_result


class _ThreadedRankCollective:
    """Simulates the production bounded-gatherer collective with threads.

    Each `.gather` call blocks until every rank has submitted its payload
    for the same sequential round, then every caller receives the same
    tuple of per-rank payloads -- exactly the shape
    `TrainRuntime._gather_rank_reports` expects from `rank_report_gatherer`.
    """

    def __init__(self, world_size: int) -> None:
        self.world_size = world_size
        self._barrier = threading.Barrier(world_size, action=self._swap)
        self._incoming: list[Any] = [None] * world_size
        self._outgoing: tuple[Any, ...] = ()

    def _swap(self) -> None:
        self._outgoing = tuple(self._incoming)
        self._incoming = [None] * self.world_size

    def __call__(self, local_report: Any) -> tuple[Any, ...]:
        rank = int(local_report["rank"])
        self._incoming[rank] = local_report
        self._barrier.wait()
        return self._outgoing


class _WorldSizeAccelerator:
    def __init__(self, *, process_index: int, num_processes: int) -> None:
        self.process_index = process_index
        self.num_processes = num_processes
        self.device = torch.device("cpu")
        self.is_main_process = process_index == 0
        self.distributed_type = SimpleNamespace(name="NO")
        self.gradient_accumulation_steps = 1
        self.mixed_precision = "no"
        self.scaler = None

    def prepare(self, *objects: Any) -> tuple[Any, ...]:
        return objects


def _train_runtime_for_rank(
    *, rank: int, world_size: int, collective: _ThreadedRankCollective
) -> TrainRuntime:
    return TrainRuntime(
        runtime_config=RuntimeConfig(seed=17),
        runtime_batch=RuntimeBatchResolution(
            world_size=world_size,
            effective_batch_size=world_size,
            resolved_grad_accum_steps=1,
        ),
        model=torch.nn.Linear(1, 1),
        optimizer=None,
        scheduler=None,
        expected_mixed_precision="no",
        accelerator=_WorldSizeAccelerator(process_index=rank, num_processes=world_size),
        rank_report_gatherer=collective if world_size > 1 else None,
    )


def _run_rank(
    *,
    micro_steps: tuple[SupervisedMicroStep, ...],
    reduction_mode: str,
    world_size: int,
    rank: int,
    runtime: Any,
    results: dict[int, ForwardEvalObservation],
    errors: dict[int, BaseException],
    contexts_by_pack: Mapping[int, LossContext] | None = None,
    loss_runner: LossRunner | None = None,
) -> None:
    try:
        runner = ForwardEvalRunner(
            model=object(),
            micro_step_stream=iter(micro_steps),
            loss_runner=loss_runner
            if loss_runner is not None
            else _wave4_loss_runner(),
            eval_source={"path": "fixture.jsonl"},
            qwen_forward=_StaticQwenForward(contexts_by_pack),
            loss_context_factory=_wave4_loss_context_factory,
            runtime=runtime,
            reduction_mode=reduction_mode,
            world_size=world_size,
            rank=rank,
        )
        results[rank] = runner.run(planned_step_id=4, trigger_reasons=("scheduled",))
    except BaseException as exc:  # noqa: BLE001 - surfaced to the driving thread
        errors[rank] = exc


_WAVE4_CONTEXTS_BY_PACK = {
    0: _wave4_context(
        _wave4_logits(
            (
                (0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                (0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 1.0),
                (0.0, 0.0, 0.0, 1.0, 5.0, 7.0, 0.0, 2.0),
                (0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 0.0, 1.0),
            )
        ),
        (
            _wave4_segment(0, 0, 2, pack_index=0),
            _wave4_segment(1, 2, 4, pack_index=0),
        ),
        (
            _wave4_atom(pack_index=0, segment_index=0, target_position=1, token_id=7),
            _wave4_atom(
                pack_index=0,
                segment_index=1,
                target_position=3,
                token_id=5,
                token_type="eos",
            ),
        ),
        pack_index=0,
    ),
    1: _wave4_context(
        _wave4_logits(
            (
                (0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 1.0),
                (0.0, 0.0, 0.0, 1.0, 6.0, 0.0, 0.0, 2.0),
            )
        ),
        (_wave4_segment(0, 0, 2, pack_index=1),),
        (
            _wave4_atom(
                pack_index=1,
                segment_index=0,
                target_position=1,
                token_id=3,
                token_type="coordinate",
            ),
        ),
        pack_index=1,
    ),
    2: _wave4_context(
        _wave4_logits(
            (
                (0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                (0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 1.0),
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 1.0),
            )
        ),
        (_wave4_segment(0, 0, 3, pack_index=2),),
        (
            _wave4_atom(
                pack_index=2,
                segment_index=0,
                target_position=1,
                token_id=1,
                token_type="schema",
            ),
            _wave4_atom(
                pack_index=2,
                segment_index=0,
                target_position=2,
                token_id=5,
                token_type="eos",
            ),
        ),
        pack_index=2,
    ),
}
_WAVE4_ALL_PACKS = tuple(
    _wave4_micro_step(
        _WAVE4_CONTEXTS_BY_PACK[pack_index],
        num_examples=_WAVE4_NUM_EXAMPLES_BY_PACK[pack_index],
    )
    for pack_index in sorted(_WAVE4_CONTEXTS_BY_PACK)
)

# Restricted to packs 0 and 1 only (drops pack 2, the sole carrier of the
# "schema" token type) -- used by
# `_wave4_loss_runner_with_globally_unselected_gate_term`'s companion test.
_WAVE4_PACKS_ZERO_AND_ONE = tuple(
    step for step in _WAVE4_ALL_PACKS if step.pack.pack_index in (0, 1)
)

# Same 3-pack layout as `_WAVE4_CONTEXTS_BY_PACK`, except pack 1's logits at
# its target position are NaN -- a genuine non-finite loss contribution
# produced by the REAL `BaseTokenCE`/`TokenTypeGateLoss` cross-entropy
# computation (F.cross_entropy on a NaN logits row yields NaN), not an
# injected fake scalar. Pack 1 is assigned to rank 1 under the standard
# world_size=2 partition, so rank 0's shard stays entirely finite while
# rank 1's shard is entirely non-finite -- decisive for the finite/*
# cross-rank AND reducer (Opus HOLD P1-1).
_WAVE4_NONFINITE_CONTEXTS_BY_PACK = {
    0: _WAVE4_CONTEXTS_BY_PACK[0],
    1: _wave4_context(
        _wave4_logits(
            (
                # Atom below targets position 1, so its causal logits row
                # (target_position - 1, per TokenAtom.causal_logits_position)
                # is row 0 -- that is the row set to NaN, not row 1.
                (float("nan"),) * 8,
                (0.0, 0.0, 0.0, 1.0, 6.0, 0.0, 0.0, 2.0),
            )
        ),
        (_wave4_segment(0, 0, 2, pack_index=1),),
        (
            _wave4_atom(
                pack_index=1,
                segment_index=0,
                target_position=1,
                token_id=3,
                token_type="coordinate",
            ),
        ),
        pack_index=1,
    ),
    2: _WAVE4_CONTEXTS_BY_PACK[2],
}
_WAVE4_NONFINITE_ALL_PACKS = tuple(
    _wave4_micro_step(
        _WAVE4_NONFINITE_CONTEXTS_BY_PACK[pack_index],
        num_examples=_WAVE4_NUM_EXAMPLES_BY_PACK[pack_index],
    )
    for pack_index in sorted(_WAVE4_NONFINITE_CONTEXTS_BY_PACK)
)


def _replicated_reference_row(
    micro_steps: tuple[SupervisedMicroStep, ...] = _WAVE4_ALL_PACKS,
    *,
    contexts_by_pack: Mapping[int, LossContext] | None = None,
    loss_runner: LossRunner | None = None,
) -> dict[str, Any]:
    runner = ForwardEvalRunner(
        model=object(),
        micro_step_stream=iter(micro_steps),
        loss_runner=loss_runner if loss_runner is not None else _wave4_loss_runner(),
        eval_source={"path": "fixture.jsonl"},
        qwen_forward=_StaticQwenForward(contexts_by_pack),
        loss_context_factory=_wave4_loss_context_factory,
        runtime=None,
        reduction_mode=EVAL_REDUCTION_REPLICATED,
        world_size=1,
        rank=0,
    )
    observation = runner.run(planned_step_id=4, trigger_reasons=("scheduled",))
    return observation.to_logging_row()


def _run_sharded_two_ranks(
    micro_steps: tuple[SupervisedMicroStep, ...] = _WAVE4_ALL_PACKS,
    *,
    contexts_by_pack: Mapping[int, LossContext] | None = None,
    loss_runner: LossRunner | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Runs rank0 (packs 0,2) and rank1 (pack 1) concurrently, disjoint-shard mode."""

    world_size = 2
    collective = _ThreadedRankCollective(world_size)
    rank0_steps = partition_eval_micro_steps_for_rank(
        micro_steps, rank=0, world_size=world_size
    )
    rank1_steps = partition_eval_micro_steps_for_rank(
        micro_steps, rank=1, world_size=world_size
    )
    # Independently-computed expected split (sequence position modulo
    # world_size), not the partition function under test -- decisive for
    # both the default 3-pack fixture (evaluates to [0, 2] / [1], matching
    # this helper's own docstring) and any custom `micro_steps` subset.
    expected_rank0 = [
        step.pack.pack_index
        for index, step in enumerate(micro_steps)
        if index % world_size == 0
    ]
    expected_rank1 = [
        step.pack.pack_index
        for index, step in enumerate(micro_steps)
        if index % world_size == 1
    ]
    assert [step.pack.pack_index for step in rank0_steps] == expected_rank0
    assert [step.pack.pack_index for step in rank1_steps] == expected_rank1

    results: dict[int, ForwardEvalObservation] = {}
    errors: dict[int, BaseException] = {}
    threads = [
        threading.Thread(
            target=_run_rank,
            kwargs={
                "micro_steps": rank0_steps,
                "reduction_mode": EVAL_REDUCTION_DISJOINT_SHARD,
                "world_size": world_size,
                "rank": 0,
                "runtime": _train_runtime_for_rank(
                    rank=0, world_size=world_size, collective=collective
                ),
                "results": results,
                "errors": errors,
                "contexts_by_pack": contexts_by_pack,
                "loss_runner": loss_runner,
            },
        ),
        threading.Thread(
            target=_run_rank,
            kwargs={
                "micro_steps": rank1_steps,
                "reduction_mode": EVAL_REDUCTION_DISJOINT_SHARD,
                "world_size": world_size,
                "rank": 1,
                "runtime": _train_runtime_for_rank(
                    rank=1, world_size=world_size, collective=collective
                ),
                "results": results,
                "errors": errors,
                "contexts_by_pack": contexts_by_pack,
                "loss_runner": loss_runner,
            },
        ),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    if errors:
        raise errors[min(errors)]
    assert set(results) == {0, 1}
    return results[0].to_logging_row(), results[1].to_logging_row()


def test_disjoint_shard_eval_reproduces_full_replicated_row_with_unequal_rank_atom_counts() -> (
    None
):
    replicated_row = _replicated_reference_row()
    rank0_row, rank1_row = _run_sharded_two_ranks()

    # Every rank derives the identical global row on its own.
    assert rank0_row == rank1_row

    sharded_row = rank0_row
    assert sharded_row["example_count"] == replicated_row["example_count"] == 5
    assert sharded_row["pack_count"] == replicated_row["pack_count"] == 3
    assert sharded_row["acc_top1"] == replicated_row["acc_top1"]
    assert sharded_row["acc_top5"] == replicated_row["acc_top5"]
    assert sharded_row["loss/total"] == pytest.approx(
        replicated_row["loss/total"], rel=1e-5
    )
    for name in ("base_ce", "token_type_gate"):
        assert sharded_row[f"loss/{name}"] == pytest.approx(
            replicated_row[f"loss/{name}"], rel=1e-5
        )
        assert sharded_row[f"loss/{name}/token_weighted_diag"] == pytest.approx(
            replicated_row[f"loss/{name}/token_weighted_diag"], rel=1e-5
        )
        # Globally merged, single-emission fields -- exactly equal, never
        # rank-summed (would otherwise be doubled by world_size=2 sharding).
        assert (
            sharded_row[f"loss/{name}/segment_count"]
            == replicated_row[f"loss/{name}/segment_count"]
        )
    assert (
        sharded_row["count/supervised_atoms"]
        == replicated_row["count/supervised_atoms"]
    )
    assert (
        sharded_row["count/eligible_segments"]
        == replicated_row["count/eligible_segments"]
    )
    assert (
        sharded_row["count/skipped_segments"]
        == replicated_row["count/skipped_segments"]
    )
    # count/packs, count/examples are rank-local sums over the disjoint
    # shards, distinct from the row-level example_count/pack_count fields
    # (count/examples counts distinct segment example_ids -- 2 + 1 + 1 = 4
    # here, since pack2's two atoms share one segment -- while
    # example_count sums encoded_examples per pack -- 2 + 1 + 2 = 5).
    assert sharded_row["count/packs"] == replicated_row["count/packs"] == 3
    assert sharded_row["count/examples"] == replicated_row["count/examples"] == 4
    assert set(sharded_row) == set(replicated_row)
    for key in sharded_row:
        assert not key.endswith("__weight__")


def test_disjoint_shard_eval_total_sums_uncompensated_local_contributions() -> None:
    """Wave 3: each shard reports its own SEMANTIC partial contribution.

    Before the semantic/backward separation the reducer recovered the global
    value as `mean_r(W * c_r)`, i.e. it relied on the mean cancelling a
    backend compensation factor baked into telemetry. It now reduces
    `sum_r(c_r)` over uncompensated contributions, which is the same number
    without the detour through a backend-only factor. Proven against the
    independently computed replicated reference.
    """

    replicated_row = _replicated_reference_row()
    rank0_row, rank1_row = _run_sharded_two_ranks()
    sharded_row = rank0_row

    assert sharded_row["loss/total"] == pytest.approx(
        replicated_row["loss/total"], rel=1e-5
    )


def test_disjoint_shard_globally_zero_selected_term_matches_replicated_zero_weight_convention() -> (
    None
):
    """Independent re-review finding (2026-08-04): `_weighted_average`
    (`src/losses/runner.py`) -- the exact replicated-path computation this
    sharded reduction must reproduce -- resolves a zero-total-weight average
    to `0.0`, never an error, so `_finalize_disjoint_shard_scalars` was
    changed to match that convention instead of raising
    `eval_forward.token_weighted_diag_zero_weight`. This is a pure-function
    proof of that local convention match, using a hand-built
    `loss_artifact`/`scalars` payload.

    CORRECTED same-day follow-up (end-to-end review, see
    `test_disjoint_shard_eval_globally_zero_selected_term_is_rejected_identically_to_replicated`
    below): driving the identical "globally zero selected term" scenario
    through the REAL `LossRunner.prepare_planned_step` shows it can never
    actually reach this code -- a pre-existing, Wave-4-independent
    "at least one eligible segment" check in
    `_build_denominator_from_token_sequences` fails closed first, on every
    rank, before any forward/compute happens, in BOTH reduction modes
    identically. The `weight <= 0.0` branch this test exercises is
    therefore defensive/unreachable code in real production use, not a live
    correctness fix -- kept as harmless convention-matching code (per this
    session's explicit instruction), the same disposition already given to
    `_prepare_disjoint_shard_scalars`'s own dead `local_value is None`
    branch (task 4.2's P2-D). This test remains a valid, narrow proof that
    the helper function itself matches `_weighted_average`'s convention;
    it does not by itself establish production reachability -- see the
    end-to-end test below for that.
    """

    reference = _weighted_average([(0.0, 0), (0.0, 0)])
    assert reference == 0.0

    loss_artifact = {"terms": [{"name": "coordinate_gaussian", "selected_count": 0}]}
    scalars = {
        "loss/total": 1.0,
        "loss/coordinate_gaussian/token_weighted_diag": 0.0,
    }
    per_rank = [
        _prepare_disjoint_shard_scalars(
            scalars, loss_artifact, example_count=2, pack_count=1
        )
        for _ in range(2)
    ]
    # Cross-rank sum, exactly what `TrainRuntime._reduce_eval_sum_metric`
    # does for these keys.
    summed = {key: sum(rank[key] for rank in per_rank) for key in per_rank[0]}

    result, example_count, pack_count = _finalize_disjoint_shard_scalars(summed)

    assert result["loss/coordinate_gaussian/token_weighted_diag"] == reference == 0.0
    assert example_count == 4
    assert pack_count == 2
    assert "loss/coordinate_gaussian/token_weighted_diag/__weight__" not in result


def test_disjoint_shard_eval_globally_zero_selected_term_is_rejected_identically_to_replicated() -> (
    None
):
    """End-to-end correction (2026-08-04 follow-up review): this test drives
    the REAL `ForwardEvalRunner` + `LossRunner` + `TrainRuntime`
    bounded-collective path (not hand-summed helpers), with
    `token_type_gate` configured against a valid V1 token type ("schema")
    that is globally unmatched on the restricted 2-pack eval set
    (`_WAVE4_PACKS_ZERO_AND_ONE`).

    Driving this through the real production `LossRunner` reveals that a
    globally-zero-selected term is actually UNREACHABLE past
    `prepare_planned_step`: `_build_denominator_from_token_sequences`
    (`src/losses/runner.py`) has a pre-existing, Wave-4-independent
    "at least one eligible segment" fail-closed check that fires on the
    RANK-LOCAL token sequences before any forward/compute happens -- and
    since `eligible_segment_count > 0` necessarily implies
    `selected_atom_count > 0` for that same rank (a segment only becomes
    eligible by having at least one selected atom), no rank can ever reach
    `compute_micro_step`/`finalize_planned_step` with a term whose rank-local
    `selected_count` is zero. This holds identically for `replicated` mode
    (a single "rank" over the full set) and every rank of `disjoint_shard`
    mode, so a *globally* zero-selected term (every rank locally zero) means
    EVERY rank fails this same pre-existing check -- symmetrically, with the
    identical error code, in both reduction modes.

    Consequently, `_finalize_disjoint_shard_scalars`'s `weight <= 0.0`
    handling (the fix from the prior same-day review, kept unchanged
    per that review's explicit instruction: it is harmless, defensive code
    that mirrors `_weighted_average`'s own zero-weight convention) can never
    actually be exercised by any real `LossRunner`-driven call -- the exact
    same "assessed and confirmed unreachable, left as defensive code" class
    of finding as `_prepare_disjoint_shard_scalars`'s own `local_value is
    None` branch (task 4.2's P2-D). The decisive, honest end-to-end proof is
    therefore symmetric REJECTION, not a successful row: both modes must
    raise the identical pre-existing `loss.segment_balanced_zero_eligible`
    error for this input, never a silent divergence (one mode succeeding
    where the other crashes, or a mode-specific error code).
    """

    loss_runner_factory = _wave4_loss_runner_with_globally_unselected_gate_term
    micro_steps = _WAVE4_PACKS_ZERO_AND_ONE

    with pytest.raises(LossContractError) as replicated_excinfo:
        _replicated_reference_row(micro_steps, loss_runner=loss_runner_factory())
    assert replicated_excinfo.value.code == "loss.segment_balanced_zero_eligible"

    with pytest.raises(LossContractError) as sharded_excinfo:
        _run_sharded_two_ranks(micro_steps, loss_runner=loss_runner_factory())
    assert sharded_excinfo.value.code == "loss.segment_balanced_zero_eligible"


def _rows_equal_treating_nan_as_equal(
    a: Mapping[str, Any], b: Mapping[str, Any]
) -> bool:
    if set(a) != set(b):
        return False
    for key in a:
        left, right = a[key], b[key]
        if isinstance(left, float) and isinstance(right, float):
            if math.isnan(left) and math.isnan(right):
                continue
        if left != right:
            return False
    return True


def test_disjoint_shard_eval_nonfinite_shard_reduces_finite_flags_by_and_not_mean(
    tmp_path: Path,
) -> None:
    """Opus HOLD P1-1/P1-2: pack 1 (rank 1's entire shard) contributes a
    genuine non-finite loss (real BaseTokenCE/TokenTypeGateLoss cross
    entropy over a NaN logits row); rank 0's shard stays entirely finite.
    Full-row parity, including finite/* and writer-facing null/
    non_finite_fields normalization, must match the world_size=1 replicated
    reference exactly. The old plain-mean reducer produced 0.5 here
    (mean(1.0, 0.0)); this test fails against that code.
    """

    replicated_row = _replicated_reference_row(
        _WAVE4_NONFINITE_ALL_PACKS, contexts_by_pack=_WAVE4_NONFINITE_CONTEXTS_BY_PACK
    )
    rank0_row, rank1_row = _run_sharded_two_ranks(
        _WAVE4_NONFINITE_ALL_PACKS, contexts_by_pack=_WAVE4_NONFINITE_CONTEXTS_BY_PACK
    )

    # Every rank still derives the identical global row on its own (NaN
    # fields compared as equal-to-NaN, not spuriously unequal).
    assert _rows_equal_treating_nan_as_equal(rank0_row, rank1_row)
    sharded_row = rank0_row

    # Ground truth: the replicated evaluator's own finite/* flags are 0.0
    # (non-finite), never a fractional value, because a non-finite
    # contribution anywhere in the full set makes the corresponding global
    # scalar non-finite.
    assert replicated_row["finite/total_loss"] == 0.0
    assert replicated_row["finite/base_ce"] == 0.0
    assert replicated_row["finite/token_type_gate"] == 0.0
    assert math.isnan(replicated_row["loss/total"])

    # Decisive parity: the fixed AND/min reducer reproduces the same 0.0 --
    # NOT the old plain-mean's 0.5 (rank 0 finite=1.0, rank 1 non-finite=0.0).
    for key in ("finite/total_loss", "finite/base_ce", "finite/token_type_gate"):
        assert sharded_row[key] == replicated_row[key] == 0.0
        assert sharded_row[key] != pytest.approx(0.5)
    assert math.isnan(sharded_row["loss/total"])

    # Writer-facing non-finite/null behavior (real RunWriter.append_logging_row,
    # unchanged code) must match between the sharded and replicated rows.
    sharded_writer = RunWriter.initialize(
        run_dir=tmp_path / "sharded",
        run_id="sharded",
        run_name="sharded",
        artifact_root=tmp_path,
        collision_outcome="created",
        created_at="now",
        config_fingerprint="fp",
        resolved_config={},
        world_size=2,
        resolved_max_steps=5,
    )
    replicated_writer = RunWriter.initialize(
        run_dir=tmp_path / "replicated",
        run_id="replicated",
        run_name="replicated",
        artifact_root=tmp_path,
        collision_outcome="created",
        created_at="now",
        config_fingerprint="fp",
        resolved_config={},
        world_size=1,
        resolved_max_steps=5,
    )
    sharded_writer.append_logging_row(sharded_row)
    replicated_writer.append_logging_row(replicated_row)
    sharded_written = json.loads(
        sharded_writer.logging_path.read_text().splitlines()[0]
    )
    replicated_written = json.loads(
        replicated_writer.logging_path.read_text().splitlines()[0]
    )

    assert sharded_written["loss/total"] is None
    assert replicated_written["loss/total"] is None
    assert sharded_written["finite/total_loss"] == 0.0
    assert replicated_written["finite/total_loss"] == 0.0
    assert (
        sharded_written["non_finite_fields"] == replicated_written["non_finite_fields"]
    )
    assert "loss/total" in sharded_written["non_finite_fields"]
    assert set(sharded_written) == set(replicated_written)
    for key in sharded_written:
        assert sharded_written[key] == replicated_written[key], key


def test_replicated_fallback_when_pack_count_below_world_size_keeps_pre_change_semantics() -> (
    None
):
    world_size = 4
    assert (
        resolve_active_eval_reduction_mode(pack_count=3, world_size=world_size)
        == EVAL_REDUCTION_REPLICATED
    )

    collective = _ThreadedRankCollective(world_size)
    results: dict[int, ForwardEvalObservation] = {}
    errors: dict[int, BaseException] = {}
    threads = [
        threading.Thread(
            target=_run_rank,
            kwargs={
                "micro_steps": _WAVE4_ALL_PACKS,
                "reduction_mode": EVAL_REDUCTION_REPLICATED,
                "world_size": world_size,
                "rank": rank,
                "runtime": _train_runtime_for_rank(
                    rank=rank, world_size=world_size, collective=collective
                ),
                "results": results,
                "errors": errors,
            },
        )
        for rank in range(world_size)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    if errors:
        raise errors[min(errors)]
    assert set(results) == set(range(world_size))

    replicated_row = _replicated_reference_row()
    for rank in range(world_size):
        row = results[rank].to_logging_row()
        assert row == replicated_row
        # No count is multiplied by world_size (4): still 3 packs, 5 examples.
        assert row["pack_count"] == 3
        assert row["example_count"] == 5


def test_resolve_active_eval_reduction_mode_defaults_to_auto_sharding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("COORDEXP_SWIFT_EVAL_REDUCTION_MODE", raising=False)
    assert (
        resolve_active_eval_reduction_mode(pack_count=10, world_size=4)
        == EVAL_REDUCTION_DISJOINT_SHARD
    )
    assert resolve_active_eval_reduction_mode(pack_count=1, world_size=1) == (
        EVAL_REDUCTION_REPLICATED
    )
    assert resolve_active_eval_reduction_mode(pack_count=1, world_size=4) == (
        EVAL_REDUCTION_REPLICATED
    )


def test_resolve_active_eval_reduction_mode_auto_control_activates_sharding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COORDEXP_SWIFT_EVAL_REDUCTION_MODE", "auto")
    assert (
        resolve_active_eval_reduction_mode(pack_count=8, world_size=4)
        == EVAL_REDUCTION_DISJOINT_SHARD
    )
    # The automatic pack_count < world_size fallback overrides "auto".
    assert (
        resolve_active_eval_reduction_mode(pack_count=2, world_size=4)
        == EVAL_REDUCTION_REPLICATED
    )
    # world_size == 1 never shards regardless of the control.
    assert (
        resolve_active_eval_reduction_mode(pack_count=8, world_size=1)
        == EVAL_REDUCTION_REPLICATED
    )


def test_resolve_active_eval_reduction_mode_rejects_invalid_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COORDEXP_SWIFT_EVAL_REDUCTION_MODE", "sharded")
    with pytest.raises(RuntimeContractError) as exc_info:
        resolve_active_eval_reduction_mode(pack_count=8, world_size=4)
    assert exc_info.value.code == "eval_forward.reduction_control_invalid"


def test_partition_eval_micro_steps_for_rank_is_deterministic_and_disjoint() -> None:
    world_size = 2
    shards = [
        partition_eval_micro_steps_for_rank(
            _WAVE4_ALL_PACKS, rank=rank, world_size=world_size
        )
        for rank in range(world_size)
    ]
    all_indices = [step.pack.pack_index for shard in shards for step in shard]
    assert sorted(all_indices) == [0, 1, 2]
    assert [step.pack.pack_index for step in shards[0]] == [0, 2]
    assert [step.pack.pack_index for step in shards[1]] == [1]


def _bare_micro_step_with_pack_index(pack_index: int) -> SupervisedMicroStep:
    return SupervisedMicroStep(
        pack=_FakePack(pack_index=pack_index),
        encoded_examples=(f"ex-{pack_index}",),
        position_inputs=None,
        token_sequence=object(),
        vocab_groups=object(),
    )


def test_partition_eval_micro_steps_for_rank_uses_sequence_position_not_pack_index_value() -> (
    None
):
    # Opus HOLD P2-A: pack_index values are deliberately non-contiguous and
    # chosen so that partitioning by pack_index VALUE (the old, rejected
    # behavior: pack_index % world_size) would produce a DIFFERENT split
    # than partitioning by sequence POSITION (the correct behavior):
    # position parity  = [0, 1, 0] -> rank0={pos0,pos2}, rank1={pos1}
    # pack_index parity = [7%2=1, 8%2=0, 5%2=1] -> rank0={pos1}, rank1={pos0,pos2}
    # These disagree, so this fixture is decisive between the two schemes.
    micro_steps = (
        _bare_micro_step_with_pack_index(7),
        _bare_micro_step_with_pack_index(8),
        _bare_micro_step_with_pack_index(5),
    )
    world_size = 2
    shards = [
        partition_eval_micro_steps_for_rank(
            micro_steps, rank=rank, world_size=world_size
        )
        for rank in range(world_size)
    ]
    # Every pack covered exactly once, by its position in the given order.
    assert [step.pack.pack_index for step in shards[0]] == [7, 5]
    assert [step.pack.pack_index for step in shards[1]] == [8]
    all_covered = [step.pack.pack_index for shard in shards for step in shard]
    assert sorted(all_covered) == sorted(step.pack.pack_index for step in micro_steps)
    # The pack_index-value-based (rejected) split would have been the exact
    # opposite assignment -- assert this fixture does NOT match it.
    assert [step.pack.pack_index for step in shards[0]] != [8]
    assert [step.pack.pack_index for step in shards[1]] != [7, 5]


def test_partition_eval_micro_steps_for_rank_still_validates_pack_identity_fail_closed() -> (
    None
):
    micro_steps = (
        SupervisedMicroStep(
            pack=object(),
            encoded_examples=("ex",),
            position_inputs=None,
            token_sequence=object(),
            vocab_groups=object(),
        ),
    )
    with pytest.raises(RuntimeContractError) as exc_info:
        partition_eval_micro_steps_for_rank(micro_steps, rank=0, world_size=2)
    assert exc_info.value.code == "eval_forward.pack_identity_missing"


def test_forward_eval_runner_rejects_disjoint_shard_with_world_size_one() -> None:
    with pytest.raises(RuntimeContractError) as exc_info:
        ForwardEvalRunner(
            model=object(),
            micro_step_stream=iter(()),
            loss_runner=_wave4_loss_runner(),
            eval_source={"path": "fixture.jsonl"},
            reduction_mode=EVAL_REDUCTION_DISJOINT_SHARD,
            world_size=1,
            rank=0,
        )
    assert exc_info.value.code == "eval_forward.disjoint_shard_requires_multi_rank"


def test_forward_eval_runner_rejects_non_streaming_loss_runner_in_disjoint_shard_mode() -> (
    None
):
    with pytest.raises(RuntimeContractError) as exc_info:
        ForwardEvalRunner(
            model=object(),
            micro_step_stream=iter(()),
            loss_runner=FakeLossRunner([]),
            eval_source={"path": "fixture.jsonl"},
            reduction_mode=EVAL_REDUCTION_DISJOINT_SHARD,
            world_size=2,
            rank=0,
        )
    assert exc_info.value.code == "eval_forward.loss_runner_requires_streaming_protocol"


def test_forward_eval_runner_rejects_non_streaming_loss_runner_in_replicated_mode() -> (
    None
):
    # The non-streaming batch eval path has been deleted: construction must
    # fail closed for any loss runner lacking the streaming protocol, even
    # in the default replicated reduction mode.
    with pytest.raises(RuntimeContractError) as exc_info:
        ForwardEvalRunner(
            model=object(),
            micro_step_stream=iter(()),
            loss_runner=FakeLossRunner([]),
            eval_source={"path": "fixture.jsonl"},
        )
    assert exc_info.value.code == "eval_forward.loss_runner_requires_streaming_protocol"


def test_forward_eval_runner_rejects_unknown_reduction_mode() -> None:
    with pytest.raises(RuntimeContractError) as exc_info:
        ForwardEvalRunner(
            model=object(),
            micro_step_stream=iter(()),
            loss_runner=_wave4_loss_runner(),
            eval_source={"path": "fixture.jsonl"},
            reduction_mode="sharded",
        )
    assert exc_info.value.code == "eval_forward.reduction_mode_invalid"


def test_best_checkpoint_selector_consumes_exact_acc_top1_from_sharded_row() -> None:
    rank0_row, _rank1_row = _run_sharded_two_ranks()
    replicated_row = _replicated_reference_row()
    # Mirrors pipeline._checkpoint_handler's selector read exactly.
    BEST_EVAL_SELECTOR_NAME = "acc_top1"
    assert rank0_row[BEST_EVAL_SELECTOR_NAME] == replicated_row[BEST_EVAL_SELECTOR_NAME]
    assert isinstance(rank0_row[BEST_EVAL_SELECTOR_NAME], float)
