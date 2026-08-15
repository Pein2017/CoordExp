"""Focused CPU tests for the Human-13 RP-crossover live pack materializer.

Frozen failure-mode matrix (Task 8 materializer; written before implementation).
Every row is closed by a CPU test in this file unless marked ``live``.

| # | Invariant | Executable owner / choke point | Minimal counterexample | Closing evidence |
|---|---|---|---|---|
| 1 | Typed admitted Task2 input only | ``plan_live_packs`` publication check | a ``SimpleNamespace`` or bare token list stands in for ``AdmittedPublication`` | ``test_plan_rejects_untyped_or_mismatched_inputs`` |
| 2 | Prompt identity is canonical, never self-authored | ``plan_live_packs`` skeleton/prompt join | skeleton prompt differs from the sealed ``ArtifactIdentity.prompt_token_ids`` | ``test_plan_rejects_untyped_or_mismatched_inputs`` |
| 3 | One isolated no-padding causal segment per trajectory | ``plan_live_packs`` segment build | segment omits the generated history or pads to a fixed length | ``test_segments_are_exact_prompt_then_generated_history`` |
| 4 | No cross-trajectory attention | pack planner FA2/MRoPE evidence check | ``cu_seqlens``/segment boundaries do not equal the segment ends | ``test_pack_evidence_binds_cu_seqlens_and_mrope_resets`` |
| 5 | Requested row is ``target_index - 1`` with exact prompt offset | ``plan_live_packs`` local position derivation | row for token ``t`` is taken at ``prompt+t`` (oracleized) or at ``t`` | ``test_requested_rows_are_exact_causal_predecessors`` |
| 6 | Sealed request/token order | ``materialize_live_packs`` sealed gather | rows follow pack order instead of ``AcquisitionGroup`` order | ``test_packed_raw_logits_replay_in_sealed_order`` |
| 7 | Local -> physical binding is receipted | ``LivePackPlan.row_bindings`` | binding drops pack index, segment id, or packed position | ``test_requested_rows_are_exact_causal_predecessors`` |
| 8 | Pack split invariance | ``plan_live_packs`` + ``materialize_live_packs`` | a different ``global_max_length`` changes the emitted rows | ``test_pack_split_invariance`` |
| 9 | Ephemeral tensor-backed ``PackedRawLogits`` | ``materialize_live_packs`` | full-vocab rows are converted with ``tolist``/``numpy`` or kept as Python floats | ``test_full_vocab_rows_are_never_pythonized`` |
| 10 | Streaming bound per image | ``stream_panel_live_packs`` | two images' tensors are alive at once | ``test_stream_releases_each_image_before_the_next`` |
| 11 | Gradient-carrying policy log probs equal the sealed processed transform | ``materialize_live_packs`` transform check | replay uses RP/temperature but the loss path uses raw log-softmax | ``test_policy_logprobs_carry_gradient_and_match_sealed_transform`` |
| 12 | One packed physical mapping for Task3 and Task4 | ``plan_live_packs`` compiler co-planning | compiler sites are forwarded in a second, independent pack plan | ``test_compiler_sites_coexist_in_one_packed_mapping`` |
| 13 | Exact global denominator applied once | ``combine_trajectory_numerators`` | per-image numerators are each divided by ``N*K`` | ``test_scored_token_partition_and_single_global_denominator`` |
| 14 | Ledger decisions stay detached | ``combine_trajectory_numerators`` | advantage tensors carry gradient into the selector | ``test_scored_token_partition_and_single_global_denominator`` |
| 15 | Fail closed on ledger/acquisition drift | ``plan_live_packs`` ledger binding | credit ledger names a different acquisition group SHA-256 | ``test_credit_ledger_lineage_fails_closed`` |
| 16 | Fail closed on compiler ledger drift | ``plan_live_packs`` compiler binding | compiler segment prompt/prefix digest differs from the admitted site | ``test_compiler_ledger_lineage_fails_closed`` |
| 17 | Fail closed on wrong vocabulary | ``materialize_live_packs`` forward validation | forward returns a vocabulary the tokenizer does not own | ``test_forward_result_fails_closed`` |
| 18 | Fail closed on position gaps/duplicates/reorder | ``materialize_live_packs`` forward validation | forward omits, duplicates, or invents a compact position | ``test_forward_result_fails_closed`` |
| 19 | Fail closed on nonfinite logits | ``materialize_live_packs`` forward validation | forward returns ``inf``/``nan`` rows | ``test_forward_result_fails_closed`` |
| 20 | Fail closed on pack overflow | ``plan_live_packs`` preflight | a trajectory segment exceeds the configured pack limit | ``test_pack_overflow_fails_closed`` |
| 21 | (superseded by rows 24-26 under task 6.2) packed FA2/MRoPE seam retired | ``default_live_packed_forward`` | — | rows 24-26 |
| 22 | Import stays runtime-free | module import | importing the module imports Torch or loads a model | ``test_module_import_is_runtime_free`` |
| 23 | Live no-update parity and memory on the exact surface | v5 parity-only qualification | CPU fixtures stand in for model-quality evidence | live (Task 6 vertical; not closed here) |

Task 6.2 execution-surface correction (frozen before implementation):

| # | Invariant | Executable owner / choke point | Minimal counterexample | Closing evidence |
|---|---|---|---|---|
| 24 | Score-function rows come only from the exact fp32/SDPA history surface | ``default_exact_history_forward`` surface gate | a bf16-parameter or FA2-configured model is supplied | ``test_default_forward_fails_closed_off_the_exact_surface`` |
| 25 | BF16/FA2 packed forward is retired for score-function evidence | ``default_live_packed_forward`` | the packed FA2 seam is called or injected after task 6.2 | ``test_fa2_packed_forward_is_retired_for_score_function_evidence`` |
| 26 | Rows are batch-one per segment with census causal/history/MRoPE semantics | ``default_exact_history_forward`` | a multi-segment packed forward or wrong local rows produce the logits | ``test_default_forward_runs_batch_one_exact_history_per_segment`` |
| 27 | Exact rows are invariant to pack partitioning | planner + default exact forward | a different ``global_max_length`` changes any row or policy log prob | ``test_exact_rows_are_invariant_to_pack_partitioning`` |
| 28 | Autograd reaches real DoRA trainables through the exact surface | materialization + real peft wrapper | policy log probs detach or grads land on a copy | ``test_gradients_reach_dora_trainables_through_exact_forward`` |
| 29 | MRoPE positions come from the real nested Qwen owner | ``_derive_qwen_position_ids`` chain walk | wrapper-depth resolution regresses to one hop | ``test_default_forward_runs_batch_one_exact_history_per_segment`` |
| 30 | Denominator, lifecycle, order, and leakage rules unchanged by the swap | rows 6, 8-14 owners | surface swap alters gather order, release, ``N*K``, or diagnostics | rows 6, 8-14 tests re-run green |
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import scripts.research.collect_human13_rp_crossover as acquisition_adapter
import scripts.research.human13_greedy_compiler as compiler
import scripts.research.human13_rp_crossover_live_packs as packs
import scripts.research.human13_trajectory_credit as credit


IMAGE_TOKEN_ID = 151655
NATURAL_STOP = acquisition_adapter.NATURAL_STOP_TOKEN_ID
PROMPT_IDS = (10, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID)
VOCAB_SIZE = 151936
IMAGE_ID = 1584
UNIFORM_LOGPROB = -math.log(VOCAB_SIZE)


@dataclass(frozen=True)
class FakeImagePlan:
    merge_size: int = 2


@dataclass(frozen=True)
class FakeImageEncoding:
    image_grid_thw: tuple[int, int, int] = (1, 4, 4)
    merged_visual_tokens: int = 4
    plan: FakeImagePlan = field(default_factory=FakeImagePlan)
    pixel_values: torch.Tensor = field(default_factory=lambda: torch.zeros((16, 2)))


@dataclass(frozen=True)
class Skeleton:
    example_id: str
    input_ids: tuple[int, ...]
    prompt_token_count: int
    image_pad_physical_start: int = 1
    image_pad_physical_end: int = 5
    image_grid_thw: tuple[int, int, int] = (1, 4, 4)
    merge_size: int = 2
    image_token_id: int = IMAGE_TOKEN_ID
    image_encoding: FakeImageEncoding = field(default_factory=FakeImageEncoding)


def _skeleton(image_id: int = IMAGE_ID) -> Skeleton:
    return Skeleton(
        example_id=f"human13:{image_id}:prompt",
        input_ids=PROMPT_IDS,
        prompt_token_count=len(PROMPT_IDS),
    )


def _generated(index: int) -> tuple[int, ...]:
    """Deliberately unequal trajectory lengths inside one sealed K16 group."""

    body = tuple(500 + 100 * step + index for step in range(index % 3))
    return (*body, 900 + index, NATURAL_STOP)


def _request_receipt(
    request: acquisition_adapter.AcquisitionRequest,
) -> acquisition_adapter.NativeRequestReceipt:
    return acquisition_adapter.NativeRequestReceipt(
        request_id=request.request_id,
        seed=request.seed,
        physical_batch_index=request.physical_batch_index,
        request_order_in_batch=request.request_order_in_batch,
        sampling_params=acquisition_adapter.expected_native_sampling_evidence(request),
        prompt_token_ids_sha256=acquisition_adapter.token_ids_sha256(PROMPT_IDS),
        model_id="model",
        model_identity_sha256="c" * 64,
        session_identity_sha256="d" * 64,
    )


def _output_receipt(
    request: acquisition_adapter.AcquisitionRequest, order: int
) -> acquisition_adapter.NativeOutputReceipt:
    generated = _generated(order)
    return acquisition_adapter.NativeOutputReceipt(
        native_request_receipt_sha256=_request_receipt(request).content_sha256,
        request_id=request.request_id,
        seed=request.seed,
        physical_batch_index=request.physical_batch_index,
        request_order_in_batch=request.request_order_in_batch,
        prompt_token_ids=PROMPT_IDS,
        source_sha256="a" * 64,
        manifest_sha256="b" * 64,
        model_id="model",
        tokenizer_id="tokenizer",
        processor_id="processor",
        processor_order=("repetition_penalty", "temperature", "log_softmax"),
        sampler_backend_id="vllm:test",
        generated_token_ids=generated,
        processed_logprobs=(UNIFORM_LOGPROB,) * len(generated),
        terminal_kind="natural_stop",
    )


def _publication(
    *, repetition_penalty: float = 1.10, image_id: int = IMAGE_ID
) -> acquisition_adapter.AdmittedPublication:
    from scripts.research.human13_rp_policy import validate_acquisition_group_replay

    plan = acquisition_adapter.plan_acquisition_group(
        image_id=image_id,
        repetition_penalty=repetition_penalty,
        seed_group_id="qualification",
    )
    order = iter(range(acquisition_adapter.REQUESTS_PER_IMAGE))

    def execute_batch(
        batch: acquisition_adapter.AcquisitionBatch, params: tuple[Any, ...]
    ) -> acquisition_adapter.NativeBatchReceipt:
        del params
        return acquisition_adapter.NativeBatchReceipt(
            requests=tuple(_request_receipt(item) for item in batch.requests),
            outputs=tuple(
                _output_receipt(item, next(order)) for item in batch.requests
            ),
        )

    execution = acquisition_adapter.execute_acquisition_group(
        plan=plan, execute_batch=execute_batch
    )
    parity = validate_acquisition_group_replay(
        execution.group, execution.group, acquisition_adapter.ReplayTolerance()
    )
    binding = acquisition_adapter._publication_binding(
        execution=execution, replayed=execution.group, replay_receipt=parity
    )
    return acquisition_adapter.AdmittedPublication(
        binding=binding,
        execution=execution,
        replayed_group=execution.group,
        parity_receipt=parity,
    )


class _NoPythonizeTensor(torch.Tensor):
    """Fails the test if any full-vocabulary row leaves the tensor domain."""

    def tolist(self) -> Any:  # pragma: no cover - tripwire
        raise AssertionError("full-vocabulary rows must not be converted to Python")

    def numpy(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - tripwire
        raise AssertionError("full-vocabulary rows must not be converted to Python")


class FakeForward:
    """Emits a deterministic row per requested packed causal position."""

    def __init__(
        self,
        *,
        mode: str = "uniform",
        vocab_size: int = VOCAB_SIZE,
        parameter: torch.Tensor | None = None,
        drop_last: bool = False,
        duplicate_first: bool = False,
        reverse: bool = False,
        nonfinite: bool = False,
        tripwire: bool = False,
    ) -> None:
        self.mode = mode
        self.vocab_size = vocab_size
        self.parameter = parameter
        self.drop_last = drop_last
        self.duplicate_first = duplicate_first
        self.reverse = reverse
        self.nonfinite = nonfinite
        self.tripwire = tripwire
        self.calls: list[tuple[int, tuple[int, ...]]] = []

    def __call__(
        self,
        model: Any,
        runtime: Any,
        tokenizer: Any,
        packed: Any,
        positions: tuple[int, ...],
    ) -> Any:
        del model, runtime, tokenizer
        self.calls.append((packed.pack.pack_index, tuple(positions)))
        returned = tuple(positions)
        if self.drop_last:
            returned = returned[:-1]
        if self.duplicate_first:
            returned = (returned[0], *returned[:-1])
        if self.reverse:
            returned = tuple(reversed(returned))
        rows = torch.zeros((len(returned), self.vocab_size), dtype=torch.float32)
        if self.mode in {"position", "local"}:
            for row, position in enumerate(returned):
                segment = next(
                    item
                    for item in packed.pack.segments
                    if item.start <= position < item.end
                )
                if self.mode == "position":
                    rows[row, 0] = float(position)
                    rows[row, 1] = float(packed.pack.pack_index)
                else:
                    # split-invariant: only segment-local content decides the row
                    rows[row, 0] = float(position - segment.start)
                    rows[row, 1] = float(packed.pack.input_ids[position])
        if self.nonfinite:
            rows[0, 0] = float("inf")
        if self.parameter is not None:
            rows = rows + self.parameter
        logits = rows.unsqueeze(0)
        if self.tripwire:
            logits = logits.as_subclass(_NoPythonizeTensor)
        return SimpleNamespace(logits=logits, logits_position_ids=returned)


def packed_chosen_ids(group: Any) -> tuple[int, ...]:
    return tuple(
        token.chosen_token_id
        for item in group.trajectories
        for token in item.generated_tokens
    )


def _plan(
    publication: acquisition_adapter.AdmittedPublication | None = None,
    **kwargs: Any,
) -> packs.LivePackPlan:
    return packs.plan_live_packs(
        publication=publication if publication is not None else _publication(),
        skeleton=_skeleton(),
        **kwargs,
    )


def _materialize(plan: packs.LivePackPlan, forward: FakeForward, **kwargs: Any) -> Any:
    return packs.materialize_live_packs(
        plan=plan,
        expected_vocab_size=VOCAB_SIZE,
        packed_forward=forward,
        **kwargs,
    )


# --------------------------------------------------------------------------
# Rows 1-3, 5, 7: typed inputs, exact segments, causal predecessor mapping
# --------------------------------------------------------------------------


def test_plan_rejects_untyped_or_mismatched_inputs() -> None:
    publication = _publication()
    with pytest.raises(packs.LivePackContractError, match="AdmittedPublication"):
        packs.plan_live_packs(
            publication=SimpleNamespace(execution=publication.execution),
            skeleton=_skeleton(),
        )
    with pytest.raises(packs.LivePackContractError, match="prompt"):
        packs.plan_live_packs(
            publication=publication,
            skeleton=replace(
                _skeleton(), input_ids=(11, *PROMPT_IDS[1:]), prompt_token_count=5
            ),
        )
    with pytest.raises(packs.LivePackContractError, match="skeleton"):
        packs.plan_live_packs(publication=publication, skeleton=object())


def test_prepublication_replay_plan_accepts_only_the_exact_acquisition_execution() -> (
    None
):
    publication = _publication()

    plan = packs.plan_live_packs(
        execution=publication.execution,
        skeleton=_skeleton(),
    )

    assert plan.acquisition_group_sha256 == publication.execution.group.content_sha256
    with pytest.raises(packs.LivePackContractError, match="exactly one"):
        packs.plan_live_packs(
            publication=publication,
            execution=publication.execution,
            skeleton=_skeleton(),
        )


def test_segments_are_exact_prompt_then_generated_history() -> None:
    plan = _plan()
    assert len(plan.trajectory_bindings) == 16
    lengths = {binding.generated_token_count for binding in plan.trajectory_bindings}
    assert len(lengths) > 1, "the fixture must exercise unequal trajectory lengths"

    segments = {
        segment.segment_id: segment for segment in plan.packed_plan.logical_segments
    }
    trajectories = {
        item.identity.request_id: item
        for item in plan.publication.execution.group.trajectories
    }
    for binding in plan.trajectory_bindings:
        trajectory = trajectories[binding.request_id]
        encoded = segments[binding.segment_id].encoded_example
        assert tuple(encoded.input_ids) == (
            *PROMPT_IDS,
            *trajectory.identity.generated_token_ids,
        )
        assert binding.prompt_token_count == len(PROMPT_IDS)
        assert binding.chosen_token_ids == trajectory.identity.generated_token_ids


def test_requested_rows_are_exact_causal_predecessors() -> None:
    plan = _plan()
    prompt_count = len(PROMPT_IDS)
    starts = {
        segment.example_id: (pack.pack.pack_index, segment.start)
        for pack in plan.packed_plan.packs
        for segment in pack.pack.segments
    }
    for binding in plan.trajectory_bindings:
        assert binding.local_causal_positions == tuple(
            prompt_count - 1 + index for index in range(binding.generated_token_count)
        )
        # no prefix oracleization: the requested row always precedes its token
        assert all(
            local < prompt_count + index
            for index, local in enumerate(binding.local_causal_positions)
        )

    assert len(plan.row_bindings) == sum(
        binding.generated_token_count for binding in plan.trajectory_bindings
    )
    for row in plan.row_bindings:
        pack_index, start = starts[row.segment_id]
        assert (row.pack_index, row.packed_causal_position) == (
            pack_index,
            start + row.local_causal_position,
        )

    forward = FakeForward(mode="position")
    materialized = _materialize(plan, forward)
    logits = materialized.packed_raw_logits.logits
    for index, row in enumerate(plan.row_bindings):
        assert float(logits[index, 0]) == float(row.packed_causal_position)
        assert float(logits[index, 1]) == float(row.pack_index)
    materialized.release()


# --------------------------------------------------------------------------
# Rows 4, 6, 8: attention isolation, sealed order, pack split invariance
# --------------------------------------------------------------------------


def test_pack_evidence_binds_cu_seqlens_and_mrope_resets() -> None:
    plan = _plan(global_max_length=32)
    assert len(plan.pack_requests) > 1
    for pack, request in zip(plan.packed_plan.packs, plan.pack_requests, strict=True):
        ends: list[int] = []
        running = 0
        for segment in pack.pack.segments:
            running += segment.end - segment.start
            ends.append(running)
        assert request.segment_boundaries == (0, *ends)
        assert request.cu_seq_lens == (0, *ends)
        assert request.mrope_reset_points == tuple(
            segment.start for segment in pack.pack.segments
        )
        assert request.pack_length == running  # no padding token exists


def test_packed_raw_logits_replay_in_sealed_order() -> None:
    publication = _publication()
    plan = _plan(publication, global_max_length=32)
    materialized = _materialize(plan, FakeForward())
    sampled = publication.execution.group
    packed = materialized.packed_raw_logits
    assert packed.request_ids == tuple(
        item.identity.request_id
        for item in sampled.trajectories
        for _ in item.generated_tokens
    )
    assert packed.token_indices == tuple(
        token.token_index
        for item in sampled.trajectories
        for token in item.generated_tokens
    )
    replayed, parity = acquisition_adapter.replay_acquisition_group(
        sampled=sampled, packed=packed
    )
    assert parity.admitted is True
    assert parity.token_count == len(packed.request_ids)
    assert parity.group_mean_absolute_error_nats <= 0.002
    assert tuple(
        token.chosen_token_id
        for item in replayed.trajectories
        for token in item.generated_tokens
    ) == packed_chosen_ids(sampled)
    materialized.release()


def test_pack_split_invariance() -> None:
    publication = _publication()
    single = _materialize(
        _plan(publication, global_max_length=12_000), FakeForward(mode="local")
    )
    split_plan = _plan(publication, global_max_length=32)
    split = _materialize(split_plan, FakeForward(mode="local"))
    assert len(split_plan.pack_requests) > len(single.plan.pack_requests)
    assert single.packed_raw_logits.request_ids == split.packed_raw_logits.request_ids
    assert (
        single.packed_raw_logits.token_indices == split.packed_raw_logits.token_indices
    )
    assert torch.equal(single.packed_raw_logits.logits, split.packed_raw_logits.logits)
    for request_id, values in single.policy_logprobs.items():
        assert torch.allclose(values, split.policy_logprobs[request_id])
    assert single.receipt.requested_row_count == split.receipt.requested_row_count
    assert single.receipt.pack_count == 1
    assert split.receipt.pack_count == split.receipt.forward_count > 1
    single.release()
    split.release()


# --------------------------------------------------------------------------
# Rows 9-11: ephemeral tensors, streaming bound, sealed processed transform
# --------------------------------------------------------------------------


def test_full_vocab_rows_are_never_pythonized() -> None:
    plan = _plan()
    materialized = _materialize(plan, FakeForward(tripwire=True))
    packed = materialized.packed_raw_logits
    assert isinstance(packed.logits, torch.Tensor)
    assert packed.logits.shape == (len(plan.row_bindings), VOCAB_SIZE)
    assert packed.logits.requires_grad is False
    assert materialized.receipt.vocab_size == VOCAB_SIZE
    assert materialized.receipt.sealed_row_bytes == (
        len(plan.row_bindings) * VOCAB_SIZE * 4
    )
    assert materialized.receipt.compact_row_bytes > 0
    materialized.release()
    with pytest.raises(packs.LivePackContractError, match="released"):
        materialized.packed_raw_logits


def test_stream_releases_each_image_before_the_next() -> None:
    publications = (
        _publication(image_id=1584),
        _publication(image_id=2299),
    )
    panel = credit.TrajectoryCreditPanelAcquisition(publications)
    skeletons = {1584: _skeleton(1584), 2299: _skeleton(2299)}
    seen = []
    for materialized in packs.stream_panel_live_packs(
        acquisition=panel,
        skeletons=skeletons,
        expected_vocab_size=VOCAB_SIZE,
        packed_forward=FakeForward(),
    ):
        assert materialized.released is False
        seen.append(materialized)
    assert [item.plan.image_id for item in seen] == [1584, 2299]
    assert [item.released for item in seen] == [True, True]


def test_policy_logprobs_carry_gradient_and_match_sealed_transform() -> None:
    from scripts.research.human13_rp_policy import processed_policy_logprobs

    parameter = torch.arange(VOCAB_SIZE, dtype=torch.float32).mul(1e-4)
    parameter.requires_grad_(True)
    publication = _publication()
    plan = _plan(publication)
    materialized = _materialize(plan, FakeForward(parameter=parameter))
    trajectories = {
        item.identity.request_id: item
        for item in publication.execution.group.trajectories
    }
    for request_id, values in materialized.policy_logprobs.items():
        trajectory = trajectories[request_id]
        assert values.requires_grad is True
        assert values.shape == (len(trajectory.generated_tokens),)
        rows = materialized.packed_raw_logits.logits
        offsets = [
            index
            for index, row in enumerate(plan.row_bindings)
            if row.request_id == request_id
        ]
        for token, offset in zip(trajectory.generated_tokens, offsets, strict=True):
            reference = processed_policy_logprobs(
                rows[offset], token, trajectory.policy_contract
            )[token.chosen_token_id]
            assert torch.allclose(
                values[token.token_index].detach(), reference, atol=1e-6
            )
    total = torch.stack(
        [values.sum() for values in materialized.policy_logprobs.values()]
    ).sum()
    total.backward()
    assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    materialized.release()


def test_processed_transform_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        packs, "_processed_chosen_logprob", lambda *a, **k: torch.tensor(-1.0)
    )
    with pytest.raises(packs.LivePackContractError, match="processed"):
        _materialize(_plan(), FakeForward())


# --------------------------------------------------------------------------
# Rows 12-16: one packed mapping for Task3/Task4 and content-bound ledgers
# --------------------------------------------------------------------------


def _credit_ledger(
    publications: tuple[acquisition_adapter.AdmittedPublication, ...],
    *,
    acquisition_sha256: str | None = None,
    image_group_sha256: str | None = None,
) -> credit.TrajectoryCreditLedger:
    panel = credit.TrajectoryCreditPanelAcquisition(publications)
    images = []
    for publication in publications:
        group = publication.replayed_group
        immediate = tuple(
            -0.5 if index % 2 == 0 else 0.0 for index in range(len(group.trajectories))
        )
        trajectories = []
        for index, trajectory in enumerate(group.trajectories):
            count = len(trajectory.generated_tokens)
            body_indices = tuple(range(count - 1))
            outcome = "legacy_m" if index == 0 else "unmatched"
            scored = outcome != "legacy_m"
            body = credit.RowCredit(
                request_id=trajectory.identity.request_id,
                generated_order=0,
                position_index=0,
                outcome=outcome,
                matched_owner_id=None,
                owner_stratum=None,
                token_indices=body_indices,
                immediate_credit=0.0,
                return_to_go=immediate[index],
                unclamped_advantage=0.0,
                advantage=0.0,
                scored=scored,
                tokens=tuple(
                    credit.TokenCredit(
                        request_id=trajectory.identity.request_id,
                        token_index=token_index,
                        row_position=0,
                        outcome=outcome,
                        advantage=0.0,
                        scored=scored,
                    )
                    for token_index in body_indices
                ),
            )
            stop = credit.RowCredit(
                request_id=trajectory.identity.request_id,
                generated_order=1,
                position_index=1,
                outcome="natural_stop",
                matched_owner_id=None,
                owner_stratum=None,
                token_indices=(count - 1,),
                immediate_credit=immediate[index],
                return_to_go=immediate[index],
                unclamped_advantage=0.0,
                advantage=0.0,
                scored=True,
                tokens=(
                    credit.TokenCredit(
                        request_id=trajectory.identity.request_id,
                        token_index=count - 1,
                        row_position=1,
                        outcome="natural_stop",
                        advantage=0.0,
                        scored=True,
                    ),
                ),
            )
            trajectories.append(
                credit.TrajectoryLedger(
                    request_id=trajectory.identity.request_id,
                    acquisition_trajectory_sha256=trajectory.content_sha256,
                    policy_contract_sha256=trajectory.policy_contract.content_sha256,
                    token_count=count,
                    terminal_kind=trajectory.terminal_kind,
                    rows=(body, stop),
                )
            )
        fixed = []
        for position in (0, 1):
            row_returns = tuple(
                trajectory.rows[position].return_to_go for trajectory in trajectories
            )
            fixed.append(row_returns)
        rebuilt = []
        for index, trajectory in enumerate(trajectories):
            rows = []
            for position, row in enumerate(trajectory.rows):
                row_returns = fixed[position]
                baseline = (sum(row_returns) - row_returns[index]) / (
                    len(trajectories) - 1
                )
                raw = row_returns[index] - baseline
                advantage = min(raw, 0.0) if row.outcome == "natural_stop" else raw
                rows.append(
                    replace(
                        row,
                        unclamped_advantage=raw,
                        advantage=advantage,
                        tokens=tuple(
                            replace(token, advantage=advantage) for token in row.tokens
                        ),
                    )
                )
            rebuilt.append(replace(trajectory, rows=tuple(rows)))
        images.append(
            credit.ImageCreditLedger(
                image_id=publication.execution.plan.image_id,
                acquisition_group_sha256=(image_group_sha256 or group.content_sha256),
                trusted_owner_ids=("g0", "h0"),
                legacy_m_owner_ids=("m0",),
                owner_weight=0.5,
                trajectories=tuple(rebuilt),
                position_returns=tuple(fixed),
            )
        )
    return credit._construct_trajectory_credit_ledger(
        source_sha256="a" * 64,
        manifest_sha256="b" * 64,
        acquisition_sha256=acquisition_sha256 or panel.content_sha256,
        logical_image_count=len(images),
        logical_k=len(images[0].trajectories),
        images=tuple(images),
    )


def test_scored_token_partition_and_single_global_denominator() -> None:
    publication = _publication()
    ledger = _credit_ledger((publication,))
    plan = _plan(publication, credit_ledger=ledger)
    scored = ledger.scored_tokens
    assert plan.scored_token_indices == tuple(range(len(scored)))

    parameter = torch.zeros(VOCAB_SIZE, dtype=torch.float32, requires_grad=True)
    materialized = _materialize(plan, FakeForward(parameter=parameter))
    numerator = credit._trajectory_score_function_numerator_for_test(
        materialized.policy_logprobs,
        ledger,
        token_indices=plan.scored_token_indices,
    )
    combined = packs.combine_trajectory_numerators((numerator,), ledger)
    assert torch.allclose(combined, numerator / ledger.logical_denominator)
    assert ledger.logical_denominator == 16
    combined.backward()
    assert parameter.grad is not None
    materialized.release()


def test_incremental_backward_releases_each_graph_before_next_image() -> None:
    """A consumer that first stacks panel numerators fails this iterator contract."""

    parameter = torch.nn.Parameter(torch.tensor(2.0))
    released: list[int] = []

    def steps():
        for image_id in (1584, 2299):
            if image_id == 2299:
                assert released == [1584]
            yield packs.StreamingObjectiveStep(
                image_id=image_id,
                trajectory_numerator=parameter * float(image_id == 1584) + parameter,
                compiler_numerator=parameter * 0.5,
                release=lambda image_id=image_id: released.append(image_id),
            )

    receipt = packs.backward_incremental_objectives(
        steps(),
        trajectory_denominator=32,
        compiler_image_denominator=2,
        include_compiler=True,
    )

    assert receipt.backward_count == 2
    assert receipt.released_graph_count == 2
    assert released == [1584, 2299]
    assert parameter.grad is not None


def test_manifest_absent_compiler_site_contributes_differentiable_zero() -> None:
    """Image 14439 has no H owner/site but still owns one global-N summand."""

    parameter = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
    observed_losses: list[torch.Tensor] = []

    receipt = packs.backward_incremental_objectives(
        iter(
            (
                packs.StreamingObjectiveStep(
                    image_id=14439,
                    trajectory_numerator=parameter * 3.0,
                    compiler_numerator=None,
                    compiler_absent_reason="no_trusted_remaining",
                    release=lambda: None,
                ),
            )
        ),
        trajectory_denominator=13 * 16,
        compiler_image_denominator=13,
        include_compiler=True,
        backward=observed_losses.append,
    )

    assert receipt.image_ids == (14439,)
    assert len(observed_losses) == 1
    loss = observed_losses[0]
    assert loss.dtype is parameter.dtype
    assert loss.device == parameter.device
    loss.backward()
    assert parameter.grad == pytest.approx(torch.tensor(3.0 / (13 * 16)))


@pytest.mark.parametrize(
    ("compiler", "absent_reason", "message"),
    (
        (None, None, "absent compiler numerator requires"),
        (torch.tensor(0.0, requires_grad=True), "no_trusted_remaining", "present"),
    ),
)
def test_incremental_backward_rejects_missing_or_mixed_compiler_evidence(
    compiler: torch.Tensor | None,
    absent_reason: str | None,
    message: str,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    with pytest.raises(packs.LivePackContractError, match=message):
        packs.backward_incremental_objectives(
            iter(
                (
                    packs.StreamingObjectiveStep(
                        image_id=14439,
                        trajectory_numerator=parameter,
                        compiler_numerator=compiler,
                        compiler_absent_reason=absent_reason,
                        release=lambda: None,
                    ),
                )
            ),
            trajectory_denominator=208,
            compiler_image_denominator=13,
            include_compiler=True,
        )


def test_credit_ledger_lineage_fails_closed() -> None:
    publication = _publication()
    with pytest.raises(packs.LivePackContractError, match="credit ledger"):
        _plan(
            publication,
            credit_ledger=_credit_ledger((publication,), image_group_sha256="f" * 64),
        )
    with pytest.raises(packs.LivePackContractError, match="credit ledger"):
        _plan(publication, credit_ledger=SimpleNamespace(images=()))


def _compiler_ledger(
    publication: acquisition_adapter.AdmittedPublication,
    *,
    prefix: tuple[int, ...],
    site_id: str = "site-0",
    segment_id: str = "human13rp-compiler:1584:site-0",
) -> compiler.CompilerLedger:
    site = compiler.CompilerSite(
        site_id=site_id,
        packed_segment_id=segment_id,
        image_id=publication.execution.plan.image_id,
        source_decode_sha256="1" * 64,
        alias_bank_sha256="2" * 64,
        generated_token_index=len(prefix),
        local_causal_position=len(PROMPT_IDS) + len(prefix) - 1,
        bad_token_id=777,
        compact_token_ids=(555, 777),
        repeated_token_ids=(),
        prompt_token_count=len(PROMPT_IDS),
        prompt_token_sha256=acquisition_adapter.token_ids_sha256(PROMPT_IDS),
        source_prefix_token_count=len(prefix),
        source_prefix_token_sha256=acquisition_adapter.token_ids_sha256(prefix),
        source_history_token_count=len(prefix),
        source_history_sha256="3" * 64,
        alias_children=(compiler.AliasChild("h0", "a0", 555),),
        valid_token_ids=(555,),
        valid_token_weights=(1.0,),
    )
    return compiler.CompilerLedger(
        source_sha256="a" * 64,
        manifest_sha256="b" * 64,
        acquisition_sha256=credit.TrajectoryCreditPanelAcquisition(
            (publication,)
        ).content_sha256,
        trajectory_credit_sha256="4" * 64,
        repetition_penalty=publication.execution.plan.repetition_penalty,
        logical_image_count=1,
        frozen_alias_count=compiler.EXACT_ALIAS_COUNT,
        alias_bank_sha256="2" * 64,
        source_panel_sha256="5" * 64,
        images=(
            compiler.CompilerImageLedger(
                image_id=publication.execution.plan.image_id,
                source_decode_sha256="1" * 64,
                site=site,
                absent_reason=None,
            ),
        ),
    )


def test_compiler_sites_coexist_in_one_packed_mapping() -> None:
    publication = _publication()
    prefix = (555, 666)
    request = packs.CompilerSegmentRequest(
        site_id="site-0",
        image_id=IMAGE_ID,
        segment_id="human13rp-compiler:1584:site-0",
        token_ids=(*PROMPT_IDS, *prefix),
        local_causal_position=len(PROMPT_IDS) + len(prefix) - 1,
    )
    ledger = _compiler_ledger(publication, prefix=prefix)
    plan = _plan(
        publication,
        compiler_segments=(request,),
        compiler_ledger=ledger,
        global_max_length=32,
    )
    assert len(plan.compiler_row_bindings) == 1
    binding = plan.compiler_row_bindings[0]
    assert binding.site_id == "site-0"
    assert binding.pack_index in {item.pack_index for item in plan.pack_requests}
    pack_request = next(
        item for item in plan.pack_requests if item.pack_index == binding.pack_index
    )
    assert binding.packed_causal_position in pack_request.compact_positions
    assert pack_request.compact_positions == tuple(
        sorted(set(pack_request.compact_positions))
    )

    forward = FakeForward(mode="position")
    materialized = _materialize(plan, forward)
    row = materialized.compiler_rows["site-0"]
    assert row.pack_index == binding.pack_index
    assert row.logits_position_ids == (binding.packed_causal_position,)
    assert row.raw_logits.shape == (1, VOCAB_SIZE)
    assert float(row.raw_logits[0, 0]) == float(binding.packed_causal_position)
    # one packed physical mapping: trajectory rows and compiler rows share packs
    assert {call[0] for call in forward.calls} == {
        item.pack_index for item in plan.pack_requests
    }
    assert len(forward.calls) == len(plan.pack_requests)
    materialized.release()


def test_compiler_ledger_lineage_fails_closed() -> None:
    publication = _publication()
    prefix = (555, 666)
    ledger = _compiler_ledger(publication, prefix=prefix)
    good = packs.CompilerSegmentRequest(
        site_id="site-0",
        image_id=IMAGE_ID,
        segment_id="human13rp-compiler:1584:site-0",
        token_ids=(*PROMPT_IDS, *prefix),
        local_causal_position=len(PROMPT_IDS) + len(prefix) - 1,
    )
    with pytest.raises(packs.LivePackContractError, match="compiler"):
        _plan(
            publication,
            compiler_segments=(replace(good, token_ids=(*PROMPT_IDS, 555, 999)),),
            compiler_ledger=ledger,
        )
    with pytest.raises(packs.LivePackContractError, match="compiler"):
        _plan(
            publication,
            compiler_segments=(replace(good, local_causal_position=0),),
            compiler_ledger=ledger,
        )
    with pytest.raises(packs.LivePackContractError, match="compiler"):
        _plan(
            publication,
            compiler_segments=(replace(good, site_id="site-9"),),
            compiler_ledger=ledger,
        )
    with pytest.raises(packs.LivePackContractError, match="prompt"):
        _plan(
            publication,
            compiler_segments=(replace(good, token_ids=(1, 2, 3, 4, 5, 6, 7)),),
        )
    cross_rp = _compiler_ledger(_publication(repetition_penalty=1.0), prefix=prefix)
    with pytest.raises(packs.LivePackContractError, match="RP"):
        _plan(publication, compiler_segments=(good,), compiler_ledger=cross_rp)


# --------------------------------------------------------------------------
# Rows 17-22: fail-closed forward, overflow, real Qwen seam, runtime-free import
# --------------------------------------------------------------------------


def test_forward_result_fails_closed() -> None:
    plan = _plan()
    with pytest.raises(packs.LivePackContractError, match="vocabulary"):
        _materialize(plan, FakeForward(vocab_size=VOCAB_SIZE - 1))
    with pytest.raises(packs.LivePackContractError, match="position"):
        _materialize(plan, FakeForward(drop_last=True))
    with pytest.raises(packs.LivePackContractError, match="position"):
        _materialize(plan, FakeForward(duplicate_first=True))
    with pytest.raises(packs.LivePackContractError, match="finite"):
        _materialize(plan, FakeForward(nonfinite=True))
    with pytest.raises(packs.LivePackContractError, match="logits"):
        packs.materialize_live_packs(
            plan=plan,
            expected_vocab_size=VOCAB_SIZE,
            packed_forward=lambda *a: SimpleNamespace(
                logits=None, logits_position_ids=()
            ),
        )
    # a reordered compact return is remapped, not silently accepted positionally
    reordered = _materialize(plan, FakeForward(mode="position", reverse=True))
    for index, row in enumerate(plan.row_bindings):
        assert float(reordered.packed_raw_logits.logits[index, 0]) == float(
            row.packed_causal_position
        )
    reordered.release()


def test_pack_overflow_fails_closed() -> None:
    with pytest.raises(packs.LivePackContractError, match="pack"):
        _plan(global_max_length=6)


class _ExactRopeOwner:
    """Qwen3VLModel-shaped nested owner of exact multimodal MRoPE positions."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def get_rope_index(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
        video_grid_thw: torch.Tensor | None,
        *,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        self.calls.append(
            {
                "input_ids": input_ids,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "attention_mask": attention_mask,
            }
        )
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        return positions.view(1, 1, -1).expand(3, 1, -1), None


class _ExactSurfaceModel(torch.nn.Module):
    """fp32/SDPA conditional-generation shape with the rope owner at .model."""

    def __init__(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        attn_implementation: str = "sdpa",
    ) -> None:
        super().__init__()
        self.model = _ExactRopeOwner()
        self.config = SimpleNamespace(_attn_implementation=attn_implementation)
        self.scale = torch.nn.Parameter(torch.zeros(1, dtype=dtype))
        self.forward_calls: list[dict[str, Any]] = []

    def forward(self, **kwargs: Any) -> SimpleNamespace:
        self.forward_calls.append(kwargs)
        requested = kwargs["logits_to_keep"].tolist()
        input_ids = kwargs["input_ids"][0].tolist()
        rows = torch.zeros((1, len(requested), VOCAB_SIZE), dtype=torch.float32)
        for row, local in enumerate(requested):
            # split-invariant: only segment-local content decides the row
            rows[0, row, 0] = float(local)
            rows[0, row, 1] = float(input_ids[local])
        return SimpleNamespace(logits=rows + self.scale.to(dtype=torch.float32))


def _exact_runtime() -> SimpleNamespace:
    return SimpleNamespace(accelerator=SimpleNamespace(device=torch.device("cpu")))


class _VocabTokenizer:
    def __len__(self) -> int:
        return VOCAB_SIZE


def test_default_forward_runs_batch_one_exact_history_per_segment() -> None:
    model = _ExactSurfaceModel()
    plan = _plan()
    materialized = packs.materialize_live_packs(
        plan=plan,
        expected_vocab_size=VOCAB_SIZE,
        model=model,
        runtime=_exact_runtime(),
        tokenizer=_VocabTokenizer(),
    )

    segments_with_rows = {row.segment_id for row in plan.row_bindings}
    assert len(model.forward_calls) == len(segments_with_rows)
    rope = model.model
    assert len(rope.calls) == len(model.forward_calls)
    by_segment = {
        segment.example_id: segment
        for pack in plan.packed_plan.packs
        for segment in pack.pack.segments
    }
    examples = {
        example.example_id: example
        for pack in plan.packed_plan.packs
        for example in pack.encoded_examples
    }
    for call, rope_call in zip(model.forward_calls, rope.calls, strict=True):
        # batch-one exact history: the forward sees one segment's own tokens
        input_ids = call["input_ids"]
        assert input_ids.shape[0] == 1
        segment_tokens = tuple(input_ids[0].tolist())
        owners = [
            example
            for example in examples.values()
            if tuple(example.input_ids) == segment_tokens
        ]
        assert owners, "forward input_ids are not any segment's exact history"
        assert torch.equal(
            call["attention_mask"], torch.ones_like(input_ids)
        )
        expected_positions = (
            torch.arange(input_ids.shape[1]).view(1, 1, -1).expand(3, 1, -1)
        )
        assert torch.equal(call["position_ids"], expected_positions)
        assert call["use_cache"] is False
        assert rope_call["input_ids"] is input_ids
        segment = by_segment[owners[0].example_id]
        for local in call["logits_to_keep"].tolist():
            assert 0 <= local < segment.end - segment.start

    # gathered rows land at their packed causal positions in sealed order
    raw = materialized.packed_raw_logits
    for row_binding, row in zip(
        plan.row_bindings, raw.logits.unbind(0), strict=True
    ):
        assert row[0].item() == float(row_binding.local_causal_position)
    materialized.release()


@pytest.mark.parametrize(
    ("dtype", "attn_implementation", "match"),
    [
        (torch.bfloat16, "sdpa", "fp32/SDPA"),
        (torch.float32, "flash_attention_2", "fp32/SDPA"),
    ],
)
def test_default_forward_fails_closed_off_the_exact_surface(
    dtype: torch.dtype,
    attn_implementation: str,
    match: str,
) -> None:
    model = _ExactSurfaceModel(dtype=dtype, attn_implementation=attn_implementation)
    plan = _plan()

    with pytest.raises(packs.LivePackContractError, match=match):
        packs.materialize_live_packs(
            plan=plan,
            expected_vocab_size=VOCAB_SIZE,
            model=model,
            runtime=_exact_runtime(),
            tokenizer=_VocabTokenizer(),
        )

    assert model.forward_calls == []


def test_fa2_packed_forward_is_retired_for_score_function_evidence() -> None:
    plan = _plan()

    with pytest.raises(packs.LivePackContractError, match="task 6.2"):
        packs.default_live_packed_forward(
            _ExactSurfaceModel(),
            _exact_runtime(),
            _VocabTokenizer(),
            plan.packed_plan.packs[0],
            plan.pack_requests[0].compact_positions,
        )

    with pytest.raises(packs.LivePackContractError, match="task 6.2"):
        packs.materialize_live_packs(
            plan=plan,
            expected_vocab_size=VOCAB_SIZE,
            model=_ExactSurfaceModel(),
            runtime=_exact_runtime(),
            tokenizer=_VocabTokenizer(),
            packed_forward=packs.default_live_packed_forward,
        )


def test_exact_rows_are_invariant_to_pack_partitioning() -> None:
    publication = _publication()
    wide = packs.plan_live_packs(publication=publication, skeleton=_skeleton())
    narrow = packs.plan_live_packs(
        publication=publication,
        skeleton=_skeleton(),
        global_max_length=16,
    )
    assert len(narrow.pack_requests) > len(wide.pack_requests)

    outputs = []
    for plan in (wide, narrow):
        materialized = packs.materialize_live_packs(
            plan=plan,
            expected_vocab_size=VOCAB_SIZE,
            model=_ExactSurfaceModel(),
            runtime=_exact_runtime(),
            tokenizer=_VocabTokenizer(),
        )
        raw = materialized.packed_raw_logits
        outputs.append(
            (
                raw.request_ids,
                raw.token_indices,
                raw.logits.detach().clone(),
                {
                    request_id: value.detach().clone()
                    for request_id, value in materialized.policy_logprobs.items()
                },
            )
        )
        materialized.release()

    assert outputs[0][0] == outputs[1][0]
    assert outputs[0][1] == outputs[1][1]
    assert torch.equal(outputs[0][2], outputs[1][2])
    for request_id, values in outputs[0][3].items():
        assert torch.equal(values, outputs[1][3][request_id])


def test_gradients_reach_dora_trainables_through_exact_forward() -> None:
    from peft import LoraConfig, get_peft_model

    class _CondGen(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = _ExactRopeOwner()
            self.language_head = torch.nn.Linear(4, 4)
            self.config = SimpleNamespace(
                _attn_implementation="sdpa",
                to_dict=lambda: {"model_type": "qwen3_vl"},
            )

        def forward(self, **kwargs: Any) -> SimpleNamespace:
            requested = kwargs["logits_to_keep"]
            base = self.language_head(
                torch.ones(4, dtype=torch.float32)
            ).sum()
            logits = base + torch.zeros(
                (1, int(requested.numel()), VOCAB_SIZE), dtype=torch.float32
            )
            return SimpleNamespace(logits=logits)

    wrapped = get_peft_model(
        _CondGen(),  # type: ignore[arg-type]
        LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["language_head"],
            use_dora=True,
        ),
    )
    plan = _plan()

    materialized = packs.materialize_live_packs(
        plan=plan,
        expected_vocab_size=VOCAB_SIZE,
        model=wrapped,
        runtime=_exact_runtime(),
        tokenizer=_VocabTokenizer(),
    )
    total = torch.stack(
        [value.sum() for value in materialized.policy_logprobs.values()]
    ).sum()
    total.backward()

    trainable_grads = [
        parameter.grad
        for name, parameter in wrapped.named_parameters()
        if parameter.requires_grad and "lora" in name.lower()
    ]
    assert trainable_grads
    assert any(grad is not None for grad in trainable_grads)
    materialized.release()


def test_module_import_is_runtime_free() -> None:
    code = (
        "import sys;"
        "import scripts.research.human13_rp_crossover_live_packs as m;"
        "print(m.SCHEMA_VERSION, 'torch' in sys.modules, 'vllm' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.split() == [packs.SCHEMA_VERSION, "False", "False"]


def test_receipt_is_immutable_and_content_addressed() -> None:
    plan = _plan(global_max_length=32)
    materialized = _materialize(plan, FakeForward())
    receipt = materialized.receipt
    assert receipt.schema_version == packs.SCHEMA_VERSION
    assert receipt.image_id == IMAGE_ID
    assert receipt.acquisition_group_sha256 == plan.acquisition_group_sha256
    assert receipt.forward_count == receipt.pack_count == len(plan.pack_requests)
    assert receipt.packed_token_count == sum(
        item.pack_length for item in plan.pack_requests
    )
    assert receipt.content_sha256 == receipt.content_sha256
    assert receipt.to_dict()["content_sha256"] == receipt.content_sha256
    with pytest.raises(Exception):
        receipt.pack_count = 99  # type: ignore[misc]
    materialized.release()
