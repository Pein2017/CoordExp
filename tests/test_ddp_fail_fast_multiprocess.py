from __future__ import annotations

from datetime import timedelta
import multiprocessing as mp
from pathlib import Path
from queue import Empty
import socket
import tempfile
import types
from typing import Callable

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments

from src.trainers import with_final_checkpoint
from src.trainers.stage2_rollout_aligned import RolloutMatchingSFTTrainer
from src.trainers.stage2_two_channel import Stage2ABTrainingTrainer
from src.utils.ddp_fail_fast import ddp_rank0_coordinated_fail_fast


_WORLD_SIZE = 2
_JOIN_TIMEOUT_S = 25.0
_PG_TIMEOUT_S = 5.0
_FAIL_EXIT_CODE = 17


class _TinyCheckpointDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int):
        x = torch.tensor([float(idx), 1.0, 2.0, 3.0], dtype=torch.float32)
        y = torch.tensor(int(idx) % 2, dtype=torch.long)
        return {"x": x, "labels": y}


class _TinyCheckpointModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 2)

    def forward(self, x=None, labels=None):  # type: ignore[override]
        logits = self.proj(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, 2), labels.view(-1))
        return {"loss": loss, "logits": logits}


def _have_gloo() -> bool:
    if not dist.is_available():
        return False
    fn = getattr(dist, "is_gloo_available", None)
    if fn is None:
        return True
    return bool(fn())


def _find_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _init_cpu_pg(*, rank: int, port: int) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{int(port)}",
        rank=int(rank),
        world_size=int(_WORLD_SIZE),
        timeout=timedelta(seconds=float(_PG_TIMEOUT_S)),
    )


def _destroy_pg_best_effort() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _run_two_rank_workers(
    worker: Callable[[int, int, mp.Queue], None],
) -> tuple[list[int], dict[int, str]]:
    ctx = mp.get_context("spawn")
    port = _find_free_tcp_port()
    out_q: mp.Queue = ctx.Queue()

    procs = [
        ctx.Process(target=worker, args=(int(rank), int(port), out_q), daemon=False)
        for rank in range(_WORLD_SIZE)
    ]
    for proc in procs:
        proc.start()

    try:
        for proc in procs:
            proc.join(timeout=float(_JOIN_TIMEOUT_S))

        alive = [proc for proc in procs if proc.is_alive()]
        if alive:
            for proc in alive:
                proc.terminate()
            for proc in alive:
                proc.join(timeout=5.0)
            pytest.fail(
                "DDP worker hang detected (join timeout exceeded); "
                f"alive_pids={[int(proc.pid or -1) for proc in alive]}"
            )
    finally:
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5.0)

    messages: dict[int, str] = {}
    while True:
        try:
            rank, msg = out_q.get_nowait()
        except Empty:
            break
        messages[int(rank)] = str(msg)

    exit_codes = [int(proc.exitcode or 0) for proc in procs]
    return exit_codes, messages


def _worker_stage2_metric_nonrank0_failure(
    rank: int,
    port: int,
    out_q: mp.Queue,
) -> None:
    _init_cpu_pg(rank=int(rank), port=int(port))
    original_all_reduce = dist.all_reduce
    try:
        trainer = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
        trainer.model = types.SimpleNamespace(device=torch.device("cpu"))

        def _patched_all_reduce(tensor: torch.Tensor, op: object = None) -> None:
            if int(rank) == 1:
                raise RuntimeError("synthetic-rank1-allreduce-failure")
            original_all_reduce(tensor, op=op)

        dist.all_reduce = _patched_all_reduce  # type: ignore[assignment]

        trainer._reduce_stage2_pending_metrics_global({"stage2/raw_rollouts": 1.0})
        out_q.put((int(rank), "unexpected-success"))
    except Exception as exc:
        out_q.put((int(rank), f"{exc.__class__.__name__}: {exc}"))
        raise SystemExit(int(_FAIL_EXIT_CODE))
    finally:
        dist.all_reduce = original_all_reduce  # type: ignore[assignment]
        _destroy_pg_best_effort()


def _worker_rollout_metric_nonrank0_failure(
    rank: int,
    port: int,
    out_q: mp.Queue,
) -> None:
    _init_cpu_pg(rank=int(rank), port=int(port))
    original_all_reduce = dist.all_reduce
    try:
        trainer = RolloutMatchingSFTTrainer.__new__(RolloutMatchingSFTTrainer)
        trainer.model = types.SimpleNamespace(device=torch.device("cpu"))

        def _patched_all_reduce(tensor: torch.Tensor, op: object = None) -> None:
            if int(rank) == 1:
                raise RuntimeError("synthetic-rank1-allreduce-failure")
            original_all_reduce(tensor, op=op)

        dist.all_reduce = _patched_all_reduce  # type: ignore[assignment]

        trainer._reduce_train_rollout_log_payload_global({"train/samples_total": 1.0})
        out_q.put((int(rank), "unexpected-success"))
    except Exception as exc:
        out_q.put((int(rank), f"{exc.__class__.__name__}: {exc}"))
        raise SystemExit(int(_FAIL_EXIT_CODE))
    finally:
        dist.all_reduce = original_all_reduce  # type: ignore[assignment]
        _destroy_pg_best_effort()


def _worker_rank0_side_effect_failure(
    rank: int,
    port: int,
    out_q: mp.Queue,
) -> None:
    _init_cpu_pg(rank=int(rank), port=int(port))
    try:
        def _rank0_only_side_effect() -> None:
            if int(rank) == 0:
                raise RuntimeError("synthetic-rank0-side-effect-failure")
            return None

        ddp_rank0_coordinated_fail_fast(
            where="test/rank0-side-effect",
            fn_rank0_only=_rank0_only_side_effect,
            barrier=dist.barrier,
        )
        out_q.put((int(rank), "unexpected-success"))
    except Exception as exc:
        out_q.put((int(rank), f"{exc.__class__.__name__}: {exc}"))
        raise SystemExit(int(_FAIL_EXIT_CODE))
    finally:
        _destroy_pg_best_effort()


def _worker_checkpoint_save_rank0_failure(
    rank: int,
    port: int,
    out_q: mp.Queue,
) -> None:
    _init_cpu_pg(rank=int(rank), port=int(port))
    try:
        trainer_cls = with_final_checkpoint(Trainer)
        output_dir = (
            Path(tempfile.gettempdir()) / f"coordexp-checkpoint-ddp-fail-fast-{int(port)}"
        )
        args = TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=2,
            save_strategy="steps",
            save_steps=1,
            logging_steps=1,
            report_to=[],
            save_only_model=True,
            save_safetensors=True,
            use_cpu=True,
            remove_unused_columns=False,
            ddp_timeout=int(_PG_TIMEOUT_S),
        )
        trainer = trainer_cls(
            model=_TinyCheckpointModel(),
            args=args,
            train_dataset=_TinyCheckpointDataset(),
        )
        try:
            trainer.callback_handler.callbacks = []
        except Exception:
            pass
        trainer.state.global_step = 1

        original_save_model = trainer.save_model

        def _patched_save_model(output_dir=None, _internal_call=False):
            if int(rank) == 0:
                raise RuntimeError("synthetic-rank0-save-model-failure")
            return original_save_model(output_dir, _internal_call)

        trainer.save_model = _patched_save_model  # type: ignore[method-assign]

        try:
            trainer._save_checkpoint(trainer.model, trial=None)  # type: ignore[misc]
        except TypeError:
            trainer._save_checkpoint(  # type: ignore[misc,call-arg]
                trainer.model, trial=None, metrics=None
            )
        out_q.put((int(rank), "unexpected-success"))
    except Exception as exc:
        out_q.put((int(rank), f"{exc.__class__.__name__}: {exc}"))
        raise SystemExit(int(_FAIL_EXIT_CODE))
    finally:
        _destroy_pg_best_effort()


@pytest.mark.skipif(not _have_gloo(), reason="requires torch.distributed gloo backend")
def test_cpu_ddp_stage2_metric_aggregation_nonrank0_failure_exits_all_ranks() -> None:
    exit_codes, messages = _run_two_rank_workers(_worker_stage2_metric_nonrank0_failure)

    assert len(messages) == _WORLD_SIZE, messages
    assert all(code == int(_FAIL_EXIT_CODE) for code in exit_codes), exit_codes

    assert "stage2-ab metric all-reduce failed" in messages[0]
    assert "stage2-ab metric all-reduce failed" in messages[1]
    assert "rank=0/2" in messages[0]
    assert "rank=1/2" in messages[1]


@pytest.mark.skipif(not _have_gloo(), reason="requires torch.distributed gloo backend")
def test_cpu_ddp_rollout_metric_aggregation_nonrank0_failure_exits_all_ranks() -> None:
    exit_codes, messages = _run_two_rank_workers(_worker_rollout_metric_nonrank0_failure)

    assert len(messages) == _WORLD_SIZE, messages
    assert all(code == int(_FAIL_EXIT_CODE) for code in exit_codes), exit_codes

    assert "rollout metric all-reduce failed" in messages[0]
    assert "rollout metric all-reduce failed" in messages[1]
    assert "rank=0/2" in messages[0]
    assert "rank=1/2" in messages[1]


@pytest.mark.skipif(not _have_gloo(), reason="requires torch.distributed gloo backend")
def test_cpu_ddp_rank0_side_effect_failure_terminates_all_ranks() -> None:
    exit_codes, messages = _run_two_rank_workers(_worker_rank0_side_effect_failure)

    assert len(messages) == _WORLD_SIZE, messages
    assert all(code == int(_FAIL_EXIT_CODE) for code in exit_codes), exit_codes

    assert "DDP fail-fast abort (rank0 side effect)" in messages[0]
    assert "DDP fail-fast abort (rank0 side effect)" in messages[1]
    assert "where=test/rank0-side-effect" in messages[0]
    assert "where=test/rank0-side-effect" in messages[1]


@pytest.mark.skipif(not _have_gloo(), reason="requires torch.distributed gloo backend")
def test_cpu_ddp_checkpoint_save_rank0_failure_terminates_all_ranks() -> None:
    exit_codes, messages = _run_two_rank_workers(_worker_checkpoint_save_rank0_failure)

    assert len(messages) == _WORLD_SIZE, messages
    assert all(code == int(_FAIL_EXIT_CODE) for code in exit_codes), exit_codes

    assert "DDP fail-fast abort (rank0 side effect)" in messages[0]
    assert "DDP fail-fast abort (rank0 side effect)" in messages[1]
    assert "where=checkpoint/save_model" in messages[0]
    assert "where=checkpoint/save_model" in messages[1]
