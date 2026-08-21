"""P1-3 probe: zero-eligible ranks converge, over a REAL gloo collective.

Task 2.2 of OpenSpec change `close-coordexp-swift-review-p1s`.

What this proves that the unit tests cannot: the fixed
`_resolve_streaming_denominators` decision is made AFTER the cross-rank
denominator gather, driven by the PRODUCTION gather stack, so a rank whose
shard has zero eligible segments for a composed term does not abort into a
collective its peers are already inside.

Production surfaces exercised end to end (nothing about the gather is
re-implemented here):

    torch.distributed gloo process group (real, file:// rendezvous)
      -> src.training.control_plane._build_rank_report_gatherer
      -> src.runtime.train_runtime.TrainRuntime.gather_loss_denominators
      -> src.training.supervised_trainer._runtime_loss_denominator_gatherer
      -> src.losses.LossRunner.from_config(...).prepare_planned_step

The only stand-in is the accelerator object handed to `TrainRuntime`: a real
`accelerate.Accelerator` cannot declare `distributed_type == MULTI_GPU` on a
CPU-only gloo launch, and the accelerator is not the surface under test (it
supplies rank/world-size/device only). Everything on the denominator path is
production code.

Cases, in order (the order matters: both ranks must call the gatherer the same
number of times so its sequence counter stays aligned):

  1. CONTROL - both ranks have nonzero eligible segments. `prepare_planned_step`
     succeeds on both ranks and the denominators are the GLOBAL sums.
  2. FAILURE, production config - rank 0's shard carries segments with no
     supervised atoms at all (the real shape a shard takes when every atom of a
     segment is omitted at the segment boundary), rank 1's is normal. Under the
     production `LossesConfig` the gate groups are pinned to the full canonical
     tuple, so this is the only zero-eligible shard a config-validated run can
     produce; both protected terms are zero on rank 0 and the converged failure
     names the first canonical term.
  3. FAILURE, narrowed gate - the scenario the unit tests pin: rank 0's shard
     has atoms but none of the gated token type, so `base_ce` is nonzero while
     `token_type_gate` is zero. Requires a directly constructed `LossRunner`
     (a narrowed group tuple is rejected by `LossesConfig`), and proves the
     converged failure is per-term, not per-shard.

In both failure cases every rank must converge the identical typed
`loss.segment_balanced_zero_eligible` failure, built from gathered facts,
without a hang.

The parent process enforces a HARD TIMEOUT and fails loudly on a hang: a
hang-detection probe must never itself hang. The bound (90s) is deliberately
below the gloo control-group timeout (120s) so a genuine desync is reported by
this probe rather than by the backend.

Sensitivity: `--simulate-pre-fix` restores the pre-fix rank-local raise by
monkeypatching `_build_denominator_from_token_sequences` inside the child (no
source edit), so the probe can be OBSERVED to fail for the right reason. In
that mode the probe's own PASS/FAIL is inverted: a converged failure would mean
the probe is blind. Pair it with a short `COORDEXP_PROBE_JOIN_TIMEOUT` so the
sensitivity run does not wait out the full bound.

Usage:
    PYTHONDONTWRITEBYTECODE=1 python \
        scripts/probes/coordexp_swift/review_p1s_zero_eligible_gloo_probe.py
    PYTHONDONTWRITEBYTECODE=1 COORDEXP_PROBE_JOIN_TIMEOUT=30 python \
        scripts/probes/coordexp_swift/review_p1s_zero_eligible_gloo_probe.py \
        --simulate-pre-fix
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

from src.common.errors import LossContractError  # noqa: E402
from src.config.models import (  # noqa: E402
    LossesConfig,
    RuntimeBatchResolution,
    RuntimeConfig,
)
from src.losses import LossRunner  # noqa: E402
from src.packing.planner import PackedSegment  # noqa: E402
from src.runtime.train_runtime import TrainRuntime  # noqa: E402
from src.supervision import TokenAtom, TokenSequence  # noqa: E402
from src.training.control_plane import _build_rank_report_gatherer  # noqa: E402
from src.training.supervised_trainer import (  # noqa: E402
    _runtime_loss_denominator_gatherer,
)

WORLD_SIZE = 2
JOIN_TIMEOUT_SECONDS = float(os.environ.get("COORDEXP_PROBE_JOIN_TIMEOUT", "90"))
EXPECTED_CODE = "loss.segment_balanced_zero_eligible"
ZERO_ELIGIBLE_RANK = 0


# ---------------------------------------------------------------------------
# Accelerator stand-in (the ONLY fake; supplies rank/world/device only)
# ---------------------------------------------------------------------------


class _GlooLaunchAccelerator:
    def __init__(self, *, rank: int, world_size: int) -> None:
        self.process_index = rank
        self.num_processes = world_size
        self.device = torch.device("cpu")
        self.is_main_process = rank == 0
        self.distributed_type = SimpleNamespace(name="MULTI_GPU")
        self.gradient_accumulation_steps = 1
        self.mixed_precision = "no"
        self.scaler = None
        self.prepare_calls = 0

    def prepare(self, *objects: Any) -> tuple[Any, ...]:
        self.prepare_calls += 1
        return tuple(objects)

    @contextmanager
    def no_sync(self, model: Any) -> Any:
        yield

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def unscale_gradients(self, optimizer: Any = None) -> None:
        raise AssertionError("probe never reaches the optimizer boundary")


# ---------------------------------------------------------------------------
# Production-config loss runner and real token-sequence shards
# ---------------------------------------------------------------------------


def _config_loss_runner() -> LossRunner:
    """The production composition path: strict `LossesConfig` -> `from_config`.

    `LossesConfig` pins `token_type_gate.groups` to the full canonical tuple,
    so the gate is eligible wherever `base_ce` is.
    """

    config = LossesConfig.model_validate(
        {
            "normalizer": "segment_balanced",
            "protected": {
                "base_ce": {"weight": 1.0},
                "token_type_gate": {
                    "weight": 0.1,
                    "mode": "enabled",
                    "groups": ["desc_text", "schema", "coordinate", "eos"],
                },
            },
        }
    )
    return LossRunner.from_config(config)


def _narrowed_gate_loss_runner() -> LossRunner:
    """Gate narrowed to one group, as the unit tests pin it.

    Constructed directly because `LossesConfig` rejects a narrowed group tuple;
    this is the only way to separate gate eligibility from `base_ce`
    eligibility, which is what makes the per-term converged failure visible.
    """

    return LossRunner(
        base_ce_weight=1.0,
        token_type_gate_weight=0.1,
        token_type_gate_groups=("coordinate",),
    )


def _segment(segment_index: int, start: int, end: int) -> PackedSegment:
    return PackedSegment(
        pack_index=0,
        segment_index=segment_index,
        example_index=segment_index,
        example_id=f"ex-{segment_index}",
        start=start,
        end=end,
    )


def _atom(
    *,
    segment_index: int,
    target_position: int,
    token_id: int,
    token_type: str,
) -> TokenAtom:
    return TokenAtom(
        pack_index=0,
        segment_index=segment_index,
        example_index=segment_index,
        example_id=f"ex-{segment_index}",
        target_position=target_position,
        token_id=token_id,
        token_type=token_type,
        text="x",
        logical_target_position=target_position,
        object_id=None,
        field=None,
        source="probe",
        coordinate_target=None,
    )


def _token_sequence(*, token_type: str, token_id: int) -> TokenSequence:
    return TokenSequence(
        pack_index=0,
        input_ids=(0, 0, 0, 0),
        segments=(_segment(0, 0, 2), _segment(1, 2, 4)),
        atoms=(
            _atom(
                segment_index=0,
                target_position=1,
                token_id=token_id,
                token_type=token_type,
            ),
            _atom(
                segment_index=1,
                target_position=3,
                token_id=token_id + 1,
                token_type=token_type,
            ),
        ),
        spans=(),
    )


def _atom_free_shard() -> tuple[TokenSequence, ...]:
    """Segments with no supervised atoms -> zero eligible for every term."""

    return (
        TokenSequence(
            pack_index=0,
            input_ids=(0, 0, 0, 0),
            segments=(_segment(0, 0, 2), _segment(1, 2, 4)),
            atoms=(),
            spans=(),
        ),
    )


def _ungated_atoms_shard() -> tuple[TokenSequence, ...]:
    """Atoms present, none of them coordinate -> zero eligible for the gate."""

    return (_token_sequence(token_type="desc_text", token_id=7),)


def _nonzero_shard() -> tuple[TokenSequence, ...]:
    return (_token_sequence(token_type="coordinate", token_id=3),)


# ---------------------------------------------------------------------------
# Child process
# ---------------------------------------------------------------------------


def _emit(record: dict[str, Any], *, out_dir: str, rank: int, name: str) -> None:
    line = json.dumps(record, sort_keys=True)
    print(line, flush=True)
    path = Path(out_dir) / f"rank{rank}.{name}.json"
    path.write_text(line + "\n", encoding="utf-8")


def _zero_eligible_case(
    runner: LossRunner,
    runtime: TrainRuntime,
    *,
    rank: int,
    out_dir: str,
    name: str,
    planned_step_id: int,
    zero_shard: tuple[TokenSequence, ...],
    expected_term: str,
) -> None:
    """One failure case: rank 0 zero-eligible, rank 1 normal, both converge."""

    gatherer = _runtime_loss_denominator_gatherer(
        runtime, planned_step_id=planned_step_id, world_size=WORLD_SIZE
    )
    shard = zero_shard if rank == ZERO_ELIGIBLE_RANK else _nonzero_shard()
    raised: LossContractError | None = None
    try:
        runner.prepare_planned_step(
            shard,
            denominator_gatherer=gatherer,
            world_size=WORLD_SIZE,
            rank=rank,
        )
    except LossContractError as error:
        raised = error
    if raised is None:
        _emit(
            {
                "case": name,
                "rank": rank,
                "code": None,
                "error": "prepare_planned_step did NOT fail the planned step",
            },
            out_dir=out_dir,
            rank=rank,
            name=name,
        )
        raise AssertionError(f"{name}: zero-eligible planned step must fail")
    _emit(
        {
            "case": name,
            "rank": rank,
            "code": raised.code,
            "message": str(raised),
            "term": raised.context.get("term"),
            "zero_eligible_ranks": raised.context.get("zero_eligible_ranks"),
            "context": {key: raised.context[key] for key in sorted(raised.context)},
        },
        out_dir=out_dir,
        rank=rank,
        name=name,
    )
    assert raised.code == EXPECTED_CODE, raised.code
    assert raised.context.get("term") == expected_term, raised.context
    assert raised.context.get("zero_eligible_ranks") == [ZERO_ELIGIBLE_RANK], (
        raised.context
    )


def _install_pre_fix_local_raise() -> None:
    """Restore the pre-fix rank-local raise, without touching the source tree.

    The pre-fix code raised inside `_build_denominator_from_token_sequences`,
    i.e. BEFORE `_resolve_streaming_denominators` reached the gather. Patching
    the module global reproduces exactly that ordering, so the probe can be
    observed failing for the right reason (desync/hang), not merely passing.
    """

    import src.losses.runner as runner_module

    original = runner_module._build_denominator_from_token_sequences

    def _pre_fix(
        term_name: str,
        token_sequences: tuple[TokenSequence, ...],
        *,
        token_types: tuple[str, ...] | None,
    ) -> Any:
        denominator = original(term_name, token_sequences, token_types=token_types)
        if denominator.eligible_segment_count == 0:
            raise LossContractError(
                "segment_balanced reducer requires at least one eligible segment",
                code=EXPECTED_CODE,
                context={
                    "term": term_name,
                    "context_count": len(token_sequences),
                    "selected_atom_count": denominator.selected_atom_count,
                    "skipped_segment_count": denominator.skipped_segment_count,
                },
            )
        return denominator

    runner_module._build_denominator_from_token_sequences = _pre_fix


def _child(rank: int, init_file: str, out_dir: str, simulate_pre_fix: bool) -> None:
    exit_code = 1
    try:
        if simulate_pre_fix:
            _install_pre_fix_local_raise()
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            world_size=WORLD_SIZE,
            rank=rank,
        )
        accelerator = _GlooLaunchAccelerator(rank=rank, world_size=WORLD_SIZE)
        runtime = TrainRuntime(
            runtime_config=RuntimeConfig.model_validate(
                {"seed": 17, "determinism": {"mode": "legacy"}}
            ),
            runtime_batch=RuntimeBatchResolution(
                world_size=WORLD_SIZE,
                effective_batch_size=WORLD_SIZE,
                resolved_grad_accum_steps=1,
            ),
            model=torch.nn.Linear(1, 1, bias=False),
            optimizer=None,
            scheduler=None,
            expected_mixed_precision="no",
            max_grad_norm=None,
            accelerator=accelerator,
            # THE production gatherer: real torch.distributed gloo control group.
            rank_report_gatherer=_build_rank_report_gatherer(WORLD_SIZE),
        )
        runner = _config_loss_runner()
        narrowed_runner = _narrowed_gate_loss_runner()

        # -- case 1: control, every rank nonzero eligible --------------------
        control_gatherer = _runtime_loss_denominator_gatherer(
            runtime, planned_step_id=101, world_size=WORLD_SIZE
        )
        plan = runner.prepare_planned_step(
            _nonzero_shard(),
            denominator_gatherer=control_gatherer,
            world_size=WORLD_SIZE,
            rank=rank,
        )
        control = {
            "case": "control_all_ranks_nonzero",
            "rank": rank,
            "denominator_scope": plan.denominator_scope,
            "backend_gradient_scale": float(plan.backend_gradient_scale),
            "eligible_segment_count": {
                name: int(denominator.eligible_segment_count)
                for name, denominator in sorted(plan.denominators.items())
            },
            "selected_atom_count": {
                name: int(denominator.selected_atom_count)
                for name, denominator in sorted(plan.denominators.items())
            },
            "context_count": {
                name: int(denominator.context_count)
                for name, denominator in sorted(plan.denominators.items())
            },
        }
        _emit(control, out_dir=out_dir, rank=rank, name="control")
        assert plan.denominator_scope == "planned_step_global", plan.denominator_scope
        # Global sums over both ranks (each shard: 2 eligible segments,
        # 2 selected atoms, 1 context).
        for name in ("base_ce", "token_type_gate"):
            denominator = plan.denominators[name]
            assert int(denominator.eligible_segment_count) == 4, (name, denominator)
            assert int(denominator.selected_atom_count) == 4, (name, denominator)
            assert int(denominator.context_count) == 2, (name, denominator)
        assert float(plan.backend_gradient_scale) == 2.0

        # -- case 2: rank 0 has no supervised atoms at all --------------------
        _zero_eligible_case(
            runner,
            runtime,
            rank=rank,
            out_dir=out_dir,
            name="zero_eligible_config",
            planned_step_id=102,
            zero_shard=_atom_free_shard(),
            expected_term="base_ce",
        )

        # -- case 3: rank 0 has atoms, none of them gated --------------------
        _zero_eligible_case(
            narrowed_runner,
            runtime,
            rank=rank,
            out_dir=out_dir,
            name="zero_eligible_gate",
            planned_step_id=103,
            zero_shard=_ungated_atoms_shard(),
            expected_term="token_type_gate",
        )
        exit_code = 0
    except BaseException:  # noqa: BLE001 - probe child reports and exits
        traceback.print_exc()
        exit_code = 1
    finally:
        if dist.is_available() and dist.is_initialized():
            try:
                dist.barrier()
            except BaseException:  # noqa: BLE001
                traceback.print_exc()
            try:
                dist.destroy_process_group()
            except BaseException:  # noqa: BLE001
                traceback.print_exc()
    sys.exit(exit_code)


# ---------------------------------------------------------------------------
# Parent process
# ---------------------------------------------------------------------------


def main() -> int:
    started = time.monotonic()
    simulate_pre_fix = "--simulate-pre-fix" in sys.argv[1:]
    scratch = Path(
        os.environ.get("COORDEXP_PROBE_SCRATCH", "/tmp/coordexp-swift-probes")
    )
    run_dir = scratch / f"zero-eligible-gloo-{uuid.uuid4().hex}"
    run_dir.mkdir(parents=True, exist_ok=True)
    init_file = run_dir / "rendezvous"  # fresh per run: a stale store hangs
    print(f"[parent] run_dir={run_dir}", flush=True)
    print(f"[parent] torch={torch.__version__} world_size={WORLD_SIZE}", flush=True)
    print(
        f"[parent] join_timeout_seconds={JOIN_TIMEOUT_SECONDS:.0f} "
        f"simulate_pre_fix={simulate_pre_fix}",
        flush=True,
    )
    if simulate_pre_fix:
        print(
            "[parent] SENSITIVITY RUN: the pre-fix rank-local raise is restored "
            "in-process; a FAIL below is the expected outcome and this run's "
            "process exit code is inverted.",
            flush=True,
        )

    context = mp.get_context("spawn")
    processes = [
        context.Process(
            target=_child,
            args=(rank, str(init_file), str(run_dir), simulate_pre_fix),
            name=f"probe-rank{rank}",
        )
        for rank in range(WORLD_SIZE)
    ]
    for process in processes:
        process.start()

    deadline = time.monotonic() + JOIN_TIMEOUT_SECONDS
    hung: list[str] = []
    for process in processes:
        process.join(timeout=max(0.0, deadline - time.monotonic()))
    for process in processes:
        if process.is_alive():
            hung.append(process.name)
            process.terminate()
            process.join(timeout=10.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=10.0)

    elapsed = time.monotonic() - started
    exit_codes = {process.name: process.exitcode for process in processes}
    print(f"[parent] exit_codes={exit_codes}", flush=True)
    print(f"[parent] wall_seconds={elapsed:.2f}", flush=True)

    verdicts: list[tuple[str, bool, str]] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        verdicts.append((name, bool(passed), detail))

    check(
        f"no_hang_within_{JOIN_TIMEOUT_SECONDS:.0f}s",
        not hung,
        f"hung={hung}" if hung else "both children joined",
    )
    check(
        "both_children_exit_zero",
        all(code == 0 for code in exit_codes.values()),
        str(exit_codes),
    )

    def _load(name: str) -> dict[int, dict[str, Any]]:
        loaded: dict[int, dict[str, Any]] = {}
        for rank in range(WORLD_SIZE):
            path = run_dir / f"rank{rank}.{name}.json"
            if path.exists():
                loaded[rank] = json.loads(path.read_text(encoding="utf-8"))
        return loaded

    controls = _load("control")
    check(
        "control_case_reported_by_both_ranks",
        len(controls) == WORLD_SIZE,
        f"ranks={sorted(controls)}",
    )
    if len(controls) == WORLD_SIZE:
        check(
            "control_denominators_are_global_sums",
            all(
                row["denominator_scope"] == "planned_step_global"
                and row["eligible_segment_count"] == {"base_ce": 4, "token_type_gate": 4}
                and row["selected_atom_count"] == {"base_ce": 4, "token_type_gate": 4}
                and row["backend_gradient_scale"] == 2.0
                for row in controls.values()
            ),
            json.dumps(controls, sort_keys=True),
        )

    for case_name, expected_term in (
        ("zero_eligible_config", "base_ce"),
        ("zero_eligible_gate", "token_type_gate"),
    ):
        failures = _load(case_name)
        check(
            f"{case_name}.reported_by_both_ranks",
            len(failures) == WORLD_SIZE,
            f"ranks={sorted(failures)}",
        )
        if len(failures) != WORLD_SIZE:
            continue
        codes = {rank: row["code"] for rank, row in failures.items()}
        check(
            f"{case_name}.both_ranks_raise_the_expected_typed_code",
            all(code == EXPECTED_CODE for code in codes.values()),
            json.dumps(codes, sort_keys=True),
        )
        contexts = [failures[rank]["context"] for rank in sorted(failures)]
        check(
            f"{case_name}.both_rank_contexts_are_identical",
            contexts[0] == contexts[1],
            json.dumps(contexts[0], sort_keys=True),
        )
        messages = {failures[rank]["message"] for rank in failures}
        check(
            f"{case_name}.both_rank_messages_are_identical",
            len(messages) == 1,
            json.dumps(sorted(messages)),
        )
        check(
            f"{case_name}.term_is_{expected_term}",
            all(row["term"] == expected_term for row in failures.values()),
            json.dumps({rank: row["term"] for rank, row in failures.items()}),
        )
        check(
            f"{case_name}.zero_eligible_ranks_names_the_zero_rank",
            all(
                row["zero_eligible_ranks"] == [ZERO_ELIGIBLE_RANK]
                for row in failures.values()
            ),
            json.dumps(
                {rank: row["zero_eligible_ranks"] for rank, row in failures.items()},
                sort_keys=True,
            ),
        )

    print("", flush=True)
    print("[parent] --- verdicts ---", flush=True)
    for name, passed, detail in verdicts:
        print(f"[parent] {'PASS' if passed else 'FAIL'} {name} :: {detail}", flush=True)
    overall = all(passed for _, passed, _ in verdicts)
    if hung:
        print(
            "[parent] HANG DETECTED: children did not converge the failure within "
            f"{JOIN_TIMEOUT_SECONDS:.0f}s; terminated {hung}",
            flush=True,
        )
    print(f"[parent] VERDICT: {'PASS' if overall else 'FAIL'}", flush=True)
    print(f"[parent] wall_seconds={time.monotonic() - started:.2f}", flush=True)
    if simulate_pre_fix:
        print(
            "[parent] SENSITIVITY VERDICT: "
            + (
                "FAIL - the probe did NOT notice the pre-fix desync"
                if overall
                else "PASS - the probe fails on the pre-fix ordering"
            ),
            flush=True,
        )
        return 1 if overall else 0
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
