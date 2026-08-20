#!/usr/bin/env python
"""Two-rank CPU gloo reduction-and-update probe for Wave 3.

`standardize-coordexp-swift-supervised-losses`, task 3.6. Runs ONE planned
optimizer step over a real `gloo` CPU process group with a real
`DistributedDataParallel` module and unequal rank-local eligible-segment
counts, then re-runs the equivalent world-size-one batch in-process, and
compares:

* per-term raw / weighted semantic values and the planned-step total (which
  must carry NO backend mean-gradient compensation);
* the global `segment_balanced` denominators;
* every parameter gradient and the parameter state after exactly one
  optimizer update (a per-arm digest plus the max absolute delta);
* the reduced train metric row.

Three arms are exercised: base CE only (the named zero-weight gate
ablation), enabled gate at `0.1`, and a positive-weight coordinate auxiliary.
The receipt also records the number of `torch.distributed` collective
operations each rank issued.

Smallest real component set: `LossRunner`, `TrainRuntime`, the production
bounded rank-report gatherer, a real DDP-wrapped `nn.Linear`, and a real SGD
optimizer. NO Qwen model, NO tokenizer, NO GPU, NO cache read or write, and
no artifact anywhere except the `--output` path, which MUST NOT already
exist.

Exit code is `0` only when every parity check is inside the declared
tolerances; any failure exits `1` with the findings inside the JSON receipt
and on stdout.

Usage:
    conda run -n ms python \\
        scripts/probes/coordexp_swift/losses_wave3_two_rank_probe.py \\
        --output /absolute/path/to/receipt.json
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import timedelta
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
from queue import Empty
import socket
import sys
import traceback
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from src.config.models import RuntimeBatchResolution, RuntimeConfig  # noqa: E402
from src.coordinate_targets import CoordinateLossTarget  # noqa: E402
from src.losses import LossContext, LossRunner, TokenVocabularyGroups  # noqa: E402
from src.losses.coord_gaussian_rps import CoordGaussianRPSLoss  # noqa: E402
from src.packing.planner import PackedSegment  # noqa: E402
from src.runtime.train_runtime import TrainRuntime  # noqa: E402
from src.supervision import TokenAtom, TokenSequence  # noqa: E402
from src.training.control_plane import _build_rank_report_gatherer  # noqa: E402


WORLD_SIZE = 2
PROCESS_GROUP_TIMEOUT_SECONDS = 60
JOIN_TIMEOUT_SECONDS = 600
FEATURE_DIM = 4
VOCAB_SIZE = 8
COORDINATE_IDS = (1, 2, 3, 4)
GATE_GROUPS = ("desc_text", "schema", "coordinate", "eos")
LEARNING_RATE = 0.1

# Declared tolerances. The distributed and world-size-one paths sum the same
# fp32 quantities in different orders (per-rank partials through a gloo
# all-reduce versus one in-process accumulation), so they agree only up to
# fp32 reassociation.
VALUE_RTOL = 1e-5
VALUE_ATOL = 1e-6

ARMS: dict[str, tuple[float, float, float]] = {
    "base_ce_only": (1.0, 0.0, 0.0),
    "enabled_gate": (1.0, 0.1, 0.0),
    "coord_auxiliary": (1.0, 0.1, 0.5),
}

# `count/packs`, `count/examples` and the `token_weighted_diag` family are
# rank-local shard descriptors whose train-side cross-rank reduction predates
# this change and is untouched by it; they are reported but not parity-gated.
UNGATED_METRIC_KEYS = frozenset({"count/packs", "count/examples"})


# ---------------------------------------------------------------------------
# Deterministic rank shards (shard 0: 1 eligible segment; shard 1: 3)
# ---------------------------------------------------------------------------


def _vocab_groups() -> TokenVocabularyGroups:
    return TokenVocabularyGroups(
        vocab_size=VOCAB_SIZE,
        desc_text=(0,),
        schema=(5,),
        coordinate=COORDINATE_IDS,
        eos=(6,),
        blocked=(7,),
    )


def _atom(
    *,
    pack_index: int,
    segment_index: int,
    target_position: int,
    token_id: int,
    token_type: str,
    slot_index: int | None = None,
) -> TokenAtom:
    coordinate_target = (
        None
        if slot_index is None
        else CoordinateLossTarget(bbox=(2, 3, 8, 13), slot_index=slot_index)
    )
    return TokenAtom(
        pack_index=pack_index,
        segment_index=segment_index,
        example_index=segment_index,
        example_id=f"pack{pack_index}-seg{segment_index}",
        target_position=target_position,
        token_id=token_id,
        token_type=token_type,
        text="x" if slot_index is None else f"<|coord_{token_id - 1}|>",
        logical_target_position=target_position,
        object_id=None if slot_index is None else "obj-1",
        field=None if slot_index is None else f"bbox[{slot_index}]",
        source="wave3_probe",
        coordinate_target=coordinate_target,
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
    if shard_index == 0:
        return TokenSequence(
            pack_index=0,
            input_ids=tuple(0 for _ in range(4)),
            segments=(_segment(pack_index=0, segment_index=0, start=0, end=4),),
            atoms=(
                _atom(
                    pack_index=0,
                    segment_index=0,
                    target_position=1,
                    token_id=0,
                    token_type="desc_text",
                ),
                _atom(
                    pack_index=0,
                    segment_index=0,
                    target_position=2,
                    token_id=2,
                    token_type="coordinate",
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
            _atom(
                pack_index=1,
                segment_index=0,
                target_position=1,
                token_id=6,
                token_type="eos",
            ),
            _atom(
                pack_index=1,
                segment_index=1,
                target_position=3,
                token_id=3,
                token_type="coordinate",
                slot_index=1,
            ),
            _atom(
                pack_index=1,
                segment_index=2,
                target_position=5,
                token_id=4,
                token_type="coordinate",
                slot_index=2,
            ),
        ),
        spans=(),
    )


def _shard_features(shard_index: int) -> torch.Tensor:
    length = len(_shard_token_sequence(shard_index).input_ids)
    values = torch.linspace(
        -1.0 + 0.37 * shard_index,
        1.0 - 0.19 * shard_index,
        steps=length * FEATURE_DIM,
        dtype=torch.float32,
    )
    return values.reshape(1, length, FEATURE_DIM)


def _build_model() -> torch.nn.Module:
    torch.manual_seed(20260820)
    return torch.nn.Linear(FEATURE_DIM, VOCAB_SIZE, bias=True).float()


def _runner_for_arm(arm: str) -> LossRunner:
    base_weight, gate_weight, coord_weight = ARMS[arm]
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
        token_type_gate_groups=GATE_GROUPS,
        coord_gaussian_rps_weight=coord_weight,
        coord_gaussian_rps=coord_term,
    )


# ---------------------------------------------------------------------------
# Runtime plumbing
# ---------------------------------------------------------------------------


class _Accelerator:
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
) -> TrainRuntime:
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
    return LossContext(
        logits=model(_shard_features(shard_index)),
        token_sequence=_shard_token_sequence(shard_index),
        vocab_groups=_vocab_groups(),
    )


def _tensor_digest(tensors: list[torch.Tensor]) -> str:
    hasher = hashlib.sha256()
    for tensor in tensors:
        hasher.update(tensor.detach().contiguous().float().numpy().tobytes())
    return hasher.hexdigest()


def _run_planned_step(
    *,
    arm: str,
    runtime: TrainRuntime,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    shard_indices: tuple[int, ...],
    world_size: int,
    rank: int,
) -> dict[str, Any]:
    runner = _runner_for_arm(arm)
    planned_step_id = 1
    optimizer.zero_grad(set_to_none=True)
    token_sequences = tuple(_shard_token_sequence(index) for index in shard_indices)
    if world_size > 1:
        plan = runner.prepare_planned_step(
            token_sequences,
            denominator_gatherer=lambda payload: runtime.gather_loss_denominators(
                payload, planned_step_id=planned_step_id
            ),
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
            bundle = runner.compute_micro_step(
                _loss_context(model, shard_index),
                plan,
                local_micro_step_index=local_index,
            )
            micro_artifacts.append(bundle.to_artifact_dict())
            pre_decision = runtime.pre_backward(bundle, planned_step_id=planned_step_id)
            if not pre_decision.should_call_backward:
                raise RuntimeError(f"{arm}: pre-backward gate refused a finite step")
            runtime.backward(
                bundle.backward_loss,
                planned_step_id=planned_step_id,
                sync_gradients=sync_gradients,
            )

    finalized = runner.finalize_planned_step(tuple(micro_artifacts), plan)
    post_decision = runtime.post_backward(planned_step_id=planned_step_id)
    gradients = [
        parameter.grad.detach().clone()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if len(gradients) != len(list(model.parameters())):
        raise RuntimeError(f"{arm}: a parameter received no gradient")
    if not post_decision.should_call_optimizer_step:
        raise RuntimeError(f"{arm}: post-backward gate refused a finite step")
    runtime.clip_gradients(planned_step_id=planned_step_id)
    runtime.optimizer_step(planned_step_id=planned_step_id)
    parameters = [parameter.detach().clone() for parameter in model.parameters()]
    reduced = runtime.gather_metrics(
        finalized["metrics"],
        planned_step_id=planned_step_id,
        split="train",
        accuracy_stats=finalized["accuracy_stats"],
    )["metrics"]

    return {
        "arm": arm,
        "world_size": world_size,
        "backend_gradient_scale": float(plan.backend_gradient_scale),
        "denominator_scope": plan.denominator_scope,
        "denominators": {
            name: {
                "eligible_segment_count": denominator.eligible_segment_count,
                "selected_atom_count": denominator.selected_atom_count,
                "skipped_segment_count": denominator.skipped_segment_count,
                "context_count": denominator.context_count,
            }
            for name, denominator in plan.denominators.items()
        },
        "terms": {
            str(term["name"]): {
                "raw_loss": float(term["raw_loss"]),
                "weighted_loss": float(term["weighted_loss"]),
                "weight": float(term["weight"]),
                "backward_contribution": float(term["backward_contribution"]),
                "backend_gradient_scale": float(term["backend_gradient_scale"]),
            }
            for term in finalized["terms"]
        },
        "total_loss": float(finalized["total_loss"]),
        "backward_loss": float(finalized["backward_loss"]),
        "accuracy_stats": dict(finalized["accuracy_stats"]),
        "reduced_metrics": {str(key): float(value) for key, value in reduced.items()},
        "gradient_digest": _tensor_digest(gradients),
        "parameter_digest": _tensor_digest(parameters),
        "gradients": [tensor.flatten().tolist() for tensor in gradients],
        "parameters": [tensor.flatten().tolist() for tensor in parameters],
        "finite_status": pre_decision.finite_status,
        "optimizer_update_status": post_decision.optimizer_update_status,
    }


def _reference_world_size_one(arm: str) -> dict[str, Any]:
    model = _build_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    runtime = _train_runtime(
        model=model, optimizer=optimizer, world_size=1, rank=0, gatherer=None
    )
    return _run_planned_step(
        arm=arm,
        runtime=runtime,
        model=runtime.model,
        optimizer=runtime.optimizer,
        shard_indices=(0, 1),
        world_size=1,
        rank=0,
    )


# ---------------------------------------------------------------------------
# Parity comparison
# ---------------------------------------------------------------------------


def _within_tolerance(observed: float, expected: float) -> bool:
    return abs(observed - expected) <= max(VALUE_ATOL, VALUE_RTOL * abs(expected))


def _compare_arm(
    arm: str,
    rank_records: tuple[dict[str, Any], ...],
    reference: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    """Compare the two-rank planned step against the world-size-one batch.

    Each rank's finalized `raw_loss`/`weighted_loss` is that rank's own
    PARTIAL semantic contribution (rank-local numerator over the shared
    global denominator), so the global semantic value is their SUM -- which
    is exactly what the runtime metric reducer computes and what this
    comparison re-derives independently.
    """

    findings: list[str] = []
    deltas: dict[str, Any] = {}
    distributed = rank_records[0]

    for rank, record in enumerate(rank_records):
        if record["backend_gradient_scale"] != float(WORLD_SIZE):
            findings.append(
                f"{arm}: rank {rank} backend_gradient_scale "
                f"{record['backend_gradient_scale']} != {float(WORLD_SIZE)}"
            )
    if reference["backend_gradient_scale"] != 1.0:
        findings.append(f"{arm}: world-size-one backend_gradient_scale must be 1.0")

    for rank, record in enumerate(rank_records):
        for name, reference_denominator in reference["denominators"].items():
            observed = record["denominators"].get(name)
            if observed != reference_denominator:
                findings.append(
                    f"{arm}: rank {rank} denominator {name} {observed} != "
                    f"{reference_denominator}"
                )
        extra_terms = set(record["denominators"]) - set(reference["denominators"])
        if extra_terms:
            findings.append(
                f"{arm}: rank {rank} built extra denominators {sorted(extra_terms)}"
            )

    term_deltas: dict[str, dict[str, float]] = {}
    for name, reference_term in reference["terms"].items():
        if any(name not in record["terms"] for record in rank_records):
            findings.append(f"{arm}: term {name} missing from a distributed rank")
            continue
        term_deltas[name] = {}
        for field in ("raw_loss", "weighted_loss"):
            pooled = sum(record["terms"][name][field] for record in rank_records)
            delta = abs(pooled - reference_term[field])
            term_deltas[name][f"pooled_{field}"] = delta
            if not _within_tolerance(pooled, reference_term[field]):
                findings.append(
                    f"{arm}: term {name} pooled {field} {pooled} != "
                    f"{reference_term[field]} (delta {delta})"
                )
        # Per rank: the semantic value must NOT carry the backend factor, and
        # the backward contribution must carry it exactly once.
        for rank, record in enumerate(rank_records):
            observed_term = record["terms"][name]
            expected_contribution = observed_term["weighted_loss"] * float(WORLD_SIZE)
            if not _within_tolerance(
                observed_term["backward_contribution"], expected_contribution
            ):
                findings.append(
                    f"{arm}: rank {rank} term {name} backward_contribution "
                    f"{observed_term['backward_contribution']} != weighted x "
                    f"world_size ({expected_contribution})"
                )
    deltas["terms"] = term_deltas

    pooled_total = sum(record["total_loss"] for record in rank_records)
    total_delta = abs(pooled_total - reference["total_loss"])
    deltas["pooled_total_loss"] = total_delta
    deltas["pooled_total_loss_value"] = pooled_total
    if not _within_tolerance(pooled_total, reference["total_loss"]):
        findings.append(
            f"{arm}: pooled total_loss {pooled_total} != "
            f"{reference['total_loss']} (delta {total_delta})"
        )

    gradient_delta = _max_flat_delta(distributed["gradients"], reference["gradients"])
    parameter_delta = _max_flat_delta(distributed["parameters"], reference["parameters"])
    deltas["max_abs_gradient_delta"] = gradient_delta
    deltas["max_abs_parameter_delta"] = parameter_delta
    if gradient_delta > VALUE_ATOL + VALUE_RTOL * _max_abs(reference["gradients"]):
        findings.append(f"{arm}: gradient parity delta {gradient_delta} exceeds tolerance")
    if parameter_delta > VALUE_ATOL + VALUE_RTOL * _max_abs(reference["parameters"]):
        findings.append(
            f"{arm}: parameter-update parity delta {parameter_delta} exceeds tolerance"
        )

    metric_deltas: dict[str, float] = {}
    for key, reference_value in reference["reduced_metrics"].items():
        if key in UNGATED_METRIC_KEYS or key.endswith("/token_weighted_diag"):
            continue
        observed = distributed["reduced_metrics"].get(key)
        if observed is None:
            findings.append(f"{arm}: reduced metric {key} missing from the distributed row")
            continue
        metric_deltas[key] = abs(observed - reference_value)
        if not _within_tolerance(observed, reference_value):
            findings.append(
                f"{arm}: reduced metric {key} {observed} != {reference_value}"
            )
    deltas["reduced_metrics"] = metric_deltas

    if distributed["accuracy_stats"] != reference["accuracy_stats"]:
        # Rank-local stats differ by construction; the REDUCED row is what
        # must match, and it is compared above via acc_top1/acc_top5.
        deltas["rank_local_accuracy_stats_differ"] = True
    return findings, deltas


def _max_flat_delta(observed: list[list[float]], expected: list[list[float]]) -> float:
    if len(observed) != len(expected):
        return float("inf")
    worst = 0.0
    for left, right in zip(observed, expected, strict=True):
        if len(left) != len(right):
            return float("inf")
        for a, b in zip(left, right, strict=True):
            worst = max(worst, abs(a - b))
    return worst


def _max_abs(values: list[list[float]]) -> float:
    worst = 0.0
    for row in values:
        for value in row:
            worst = max(worst, abs(value))
    return worst


# ---------------------------------------------------------------------------
# Distributed driver
# ---------------------------------------------------------------------------


def _worker(rank: int, port: int, output: mp.Queue) -> None:
    original_all_gather = dist.all_gather
    collective_calls = {"count": 0}

    def counted_all_gather(*args: Any, **kwargs: Any) -> Any:
        collective_calls["count"] += 1
        return original_all_gather(*args, **kwargs)

    gatherer = None
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=WORLD_SIZE,
            timeout=timedelta(seconds=PROCESS_GROUP_TIMEOUT_SECONDS),
        )
        dist.all_gather = counted_all_gather  # type: ignore[assignment]
        gatherer = _build_rank_report_gatherer(WORLD_SIZE)
        if gatherer is None:
            raise RuntimeError("production rank-report gatherer is unavailable")
        arms: dict[str, Any] = {}
        for arm in ARMS:
            model = torch.nn.parallel.DistributedDataParallel(_build_model())
            optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
            runtime = _train_runtime(
                model=model,
                optimizer=optimizer,
                world_size=WORLD_SIZE,
                rank=rank,
                gatherer=gatherer,
            )
            arms[arm] = _run_planned_step(
                arm=arm,
                runtime=runtime,
                model=runtime.model,
                optimizer=runtime.optimizer,
                shard_indices=(rank,),
                world_size=WORLD_SIZE,
                rank=rank,
            )
        output.put(
            (
                rank,
                "ok",
                {"arms": arms, "collective_op_count": collective_calls["count"]},
            )
        )
    except BaseException:
        output.put((rank, traceback.format_exc(), None))
        raise
    finally:
        dist.all_gather = original_all_gather  # type: ignore[assignment]
        close = getattr(gatherer, "close", None)
        if callable(close):
            close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _find_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _run_two_ranks() -> dict[int, dict[str, Any]]:
    context = mp.get_context("spawn")
    port = _find_free_tcp_port()
    output: mp.Queue = context.Queue()
    processes = [
        context.Process(target=_worker, args=(rank, port, output), daemon=False)
        for rank in range(WORLD_SIZE)
    ]
    for process in processes:
        process.start()
    payloads: dict[int, dict[str, Any]] = {}
    errors: dict[int, str] = {}
    try:
        for process in processes:
            process.join(timeout=JOIN_TIMEOUT_SECONDS)
        alive = [process for process in processes if process.is_alive()]
        if alive:
            for process in alive:
                process.terminate()
            raise RuntimeError(
                f"two-rank probe hung; alive_pids={[item.pid for item in alive]}"
            )
    finally:
        while True:
            try:
                rank, status, payload = output.get_nowait()
            except Empty:
                break
            if status == "ok" and payload is not None:
                payloads[int(rank)] = payload
            else:
                errors[int(rank)] = str(status)
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    if errors:
        raise RuntimeError(
            "two-rank probe worker failed:\n"
            + "\n".join(f"rank {rank}:\n{message}" for rank, message in errors.items())
        )
    if sorted(payloads) != list(range(WORLD_SIZE)):
        raise RuntimeError(f"missing worker payloads: got ranks {sorted(payloads)}")
    exit_codes = [process.exitcode for process in processes]
    if exit_codes != [0] * WORLD_SIZE:
        raise RuntimeError(f"worker exit codes {exit_codes}")
    return payloads


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="strict-JSON receipt path; MUST NOT already exist",
    )
    args = parser.parse_args()
    output_path: Path = args.output
    if output_path.exists():
        print(f"[result] FAILED: --output already exists: {output_path}")
        return 1
    payloads = _run_two_ranks()
    findings: list[str] = []
    arms_receipt: dict[str, Any] = {}
    for arm in ARMS:
        reference = _reference_world_size_one(arm)
        rank0 = payloads[0]["arms"][arm]
        rank1 = payloads[1]["arms"][arm]
        arm_findings, deltas = _compare_arm(arm, (rank0, rank1), reference)
        findings.extend(arm_findings)
        # Every rank must derive the identical reduced row and identical
        # post-update parameters.
        if rank0["reduced_metrics"] != rank1["reduced_metrics"]:
            findings.append(f"{arm}: reduced metric rows diverged across ranks")
        if rank0["parameter_digest"] != rank1["parameter_digest"]:
            findings.append(f"{arm}: post-update parameter digests diverged across ranks")
        if rank0["gradient_digest"] != rank1["gradient_digest"]:
            findings.append(f"{arm}: gradient digests diverged across ranks")
        # Guard against a degenerate (accidentally replicated) contrast: the
        # rank shards MUST carry unequal eligible-segment counts and
        # therefore unequal rank-local semantic contributions.
        if rank0["terms"]["base_ce"]["raw_loss"] == rank1["terms"]["base_ce"]["raw_loss"]:
            findings.append(
                f"{arm}: rank-local base_ce contributions are equal; the unequal-"
                "shard contrast degenerated"
            )
        arms_receipt[arm] = {
            "configured_weights": {
                "base_ce": ARMS[arm][0],
                "token_type_gate": ARMS[arm][1],
                "coord_gaussian_rps": ARMS[arm][2],
            },
            "distributed_rank0": _public_arm_view(rank0),
            "distributed_rank1": _public_arm_view(rank1),
            "world_size_one_reference": _public_arm_view(reference),
            "parity_deltas": deltas,
            "findings": arm_findings,
        }

    receipt = {
        "probe": "losses_wave3_two_rank_probe",
        "change": "standardize-coordexp-swift-supervised-losses",
        "task": "3.6",
        "world_size": WORLD_SIZE,
        "backend": "gloo",
        "device": "cpu",
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_initialized": bool(torch.cuda.is_initialized()),
        "declared_tolerances": {"rtol": VALUE_RTOL, "atol": VALUE_ATOL},
        "ungated_metric_keys": sorted(UNGATED_METRIC_KEYS) + ["*/token_weighted_diag"],
        "collective_op_count_by_rank": {
            str(rank): int(payloads[rank]["collective_op_count"])
            for rank in sorted(payloads)
        },
        "arms": arms_receipt,
        "findings": findings,
        "status": "FAILED" if findings else "OK",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if findings:
        print(f"[result] FAILED with {len(findings)} parity finding(s)")
        return 1
    print(
        "[result] OK: 3 arms, two-rank gloo unequal-segment planned step matches "
        f"the world-size-one batch within rtol={VALUE_RTOL} atol={VALUE_ATOL}; "
        f"receipt written to {output_path}"
    )
    return 0


def _public_arm_view(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key not in {"gradients", "parameters"}
    }


if __name__ == "__main__":
    raise SystemExit(main())
