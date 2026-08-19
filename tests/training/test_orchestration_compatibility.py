"""Pre-move characterization of the training-orchestration compatibility ledger.

Wave 0 of the OpenSpec change ``decompose-coordexp-swift-training-orchestration``
freezes the exact behavior of every surface named in the design's compatibility
ledger *before* any production code moves.  Later waves replay these fixtures.

Rules that govern this module (design decision 12 and plan Task 1 Step 3):

* A failing characterization assertion is a baseline defect.  Investigate the
  production behavior and repair the derivation; never regenerate a fixture to
  make a red assertion green.
* Ordered phase/collective evidence is captured at the same seam that
  ``tests/training/test_pipeline_phase_convergence.py`` already instruments:
  the real ``_run_rank_converged_phase`` boundary and the real rank-report
  gatherer built by ``_build_rank_report_gatherer`` over a real gloo transport.
  Only module-level attributes of ``src.training.pipeline`` are substituted, and
  only for surfaces that would otherwise require a GPU, a model, or a dataset.
* Values that are legitimately environment-derived (absolute temporary paths,
  wall-clock stamps, monotonic durations, source-content digests) are either
  normalized against a declared placeholder or asserted against a live
  recomputation.  They are never frozen as constants.

Regenerate the fixture tree only from an untouched predecessor checkout with::

    conda run -n ms python -c "import sys; sys.path[:0]=['.', 'tests/training']; \
import test_orchestration_compatibility as m; m.freeze_fixtures()"
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass, field
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
from queue import Empty
import socket
import tempfile
import time
from types import SimpleNamespace
from typing import Any

import pytest
import torch.distributed as dist

import src.train as train_entrypoint
from src.losses.vocab import TokenVocabularyGroups
from src.packing.planner import PackedSegment
from src.qwen import parity as parity_identity
from src.supervision import TokenAtom, TokenSequence
from src.supervision.tokens import TokenSpan
from src.training import control_plane, execution_plan, pack_cache, pipeline
from src.training.pack_cache import (
    PACKING_CACHE_MATERIALIZATION_STRATEGY,
    PACKING_CACHE_VERSION,
    cache_dir_for_fingerprint,
    write_micro_step_cache,
)
from src.training.supervised_trainer import (
    CompletedStepObservation,
    SupervisedMicroStep,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "training_orchestration"
LEGACY_MICRO_STEP_PICKLE = FIXTURE_ROOT / "supervised_micro_step_legacy.pkl"

#: Placeholder substituted for the harness-owned temporary root inside frozen
#: bytes.  The substitution is textual and is declared in the fixture itself.
PATH_PLACEHOLDER = "<root>"

_WORLD_SIZE = 2
_CHARACTERIZATION_DEADLINE_SECONDS = 180.0
_CHARACTERIZATION_POLL_SECONDS = 0.1
_TERMINATE_GRACE_SECONDS = 5.0

_PIPELINE_TRACE_SCOPE = (
    "pipeline_entry_through_accelerator_runtime_preflight; initialized training "
    "is stubbed because it loads model weights"
)

_MATERIALIZATION = {
    "strategy": PACKING_CACHE_MATERIALIZATION_STRATEGY,
    "workers": 1,
}
_AUGMENTATION = {
    "split": "train",
    "mode": "disabled",
    "policy": "geometry_flips",
    "enabled": False,
    "seed": 7,
    "input_example_count": 1,
    "output_example_count": 1,
    "presentation_count": 1,
    "object_ordering": "source_order",
}


# ---------------------------------------------------------------------------
# Fixture input/output helpers (plan Task 1 helper spine)
# ---------------------------------------------------------------------------


def load_fixture(name: str) -> Any:
    """Load one frozen strict-JSON characterization fixture by file name."""

    path = FIXTURE_ROOT / (name if name.endswith(".json") else f"{name}.json")
    return json.loads(path.read_text(encoding="utf-8"))


def load_binary_fixture_tree(name: str) -> dict[str, bytes]:
    """Load one frozen byte tree as ``{posix relative path: exact bytes}``."""

    root = FIXTURE_ROOT / name
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _write_json_fixture(name: str, payload: Any) -> Path:
    path = FIXTURE_ROOT / (name if name.endswith(".json") else f"{name}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _write_binary_fixture_tree(name: str, tree: Mapping[str, bytes]) -> Path:
    root = FIXTURE_ROOT / name
    for existing in sorted(root.rglob("*"), reverse=True):
        if existing.is_file():
            existing.unlink()
        else:
            existing.rmdir()
    for relative, payload in sorted(tree.items()):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    return root


def _volatile(value: Any) -> str:
    """Replace one environment-derived value with its declared type token."""

    return f"<volatile:{type(value).__name__}>"


def _normalize_paths(value: Any, *, root: str) -> Any:
    """Replace the harness temporary root with the declared placeholder."""

    if isinstance(value, str):
        return value.replace(root, PATH_PLACEHOLDER)
    if isinstance(value, Mapping):
        return {
            _normalize_paths(key, root=root): _normalize_paths(item, root=root)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_paths(item, root=root) for item in value]
    return value


# ---------------------------------------------------------------------------
# Shared real-shaped supervision inputs
# ---------------------------------------------------------------------------


def _characterized_token_sequence() -> TokenSequence:
    """Build one real two-segment ``TokenSequence`` with real atoms and a span."""

    segments = (
        PackedSegment(
            pack_index=0,
            segment_index=0,
            example_index=0,
            example_id="ex-0",
            start=0,
            end=4,
        ),
        PackedSegment(
            pack_index=0,
            segment_index=1,
            example_index=1,
            example_id="ex-1",
            start=4,
            end=8,
        ),
    )
    first = TokenAtom(
        pack_index=0,
        segment_index=0,
        example_index=0,
        example_id="ex-0",
        target_position=2,
        token_id=11,
        token_type="desc_text",
        text="a",
        logical_target_position=2,
    )
    second = TokenAtom(
        pack_index=0,
        segment_index=0,
        example_index=0,
        example_id="ex-0",
        target_position=3,
        token_id=12,
        token_type="desc_text",
        text="a",
        logical_target_position=3,
    )
    third = TokenAtom(
        pack_index=0,
        segment_index=1,
        example_index=1,
        example_id="ex-1",
        target_position=6,
        token_id=13,
        token_type="schema",
        text="b",
        logical_target_position=6,
    )
    span = TokenSpan(
        pack_index=0,
        segment_index=0,
        example_index=0,
        example_id="ex-0",
        token_type="desc_text",
        atoms=(first, second),
        text="a",
    )
    return TokenSequence(
        pack_index=0,
        input_ids=(1, 2, 3, 4, 5, 6, 7, 8),
        segments=segments,
        atoms=(first, second, third),
        spans=(span,),
    )


def _characterized_vocab_groups() -> TokenVocabularyGroups:
    return TokenVocabularyGroups(
        vocab_size=16,
        desc_text=(11, 12),
        schema=(13,),
        coordinate=(14,),
        eos=(15,),
        blocked=(0,),
    )


def characterized_micro_step() -> SupervisedMicroStep:
    """One real-shaped micro-step exercising the restricted pickle allowlist."""

    return SupervisedMicroStep(
        pack="pack-0",
        encoded_examples=("example-0",),
        position_inputs="positions-0",
        token_sequence=_characterized_token_sequence(),
        vocab_groups=_characterized_vocab_groups(),
        metadata={"pack_id": 0, "augmentation_receipt": dict(_AUGMENTATION)},
        fa2_model_dtype="bf16",
    )


def _characterization_determinants(*, purpose: str, split: str) -> dict[str, Any]:
    """Build a valid determinant registry through the production builders."""

    semantic: dict[str, Any] = {
        "version": PACKING_CACHE_VERSION,
        "split": split,
        "dataset": {"purpose": purpose, "split": split},
        "template": {"purpose": purpose},
        "packing": {"global_max_length": 8},
        "processor": {"purpose": purpose},
        "ordering": {"purpose": purpose},
        "augmentation": dict(_AUGMENTATION),
        "qwen": {
            "processor_identity": {"purpose": purpose},
            "token_identity": {"purpose": purpose},
            "encoding_identity": {"purpose": purpose},
            "model_config_assets": {"purpose": purpose},
            "processor_assets": {"purpose": purpose},
            "tokenizer_assets": {"purpose": purpose},
        },
        "realized_vocab_groups": {
            "vocab_size": 16,
            "desc_text": [11, 12],
            "schema": [13],
            "coordinate": [14],
            "eos": [15],
            "blocked": [0],
        },
        "micro_step_runtime_config": {
            "fa2_model_dtype": "bf16",
            "capture_fa2_branch": False,
            "require_fa2_branch_proof": False,
        },
        "micro_step_schema": pack_cache._supervised_micro_step_schema_identity(),
    }
    entries = pack_cache._build_determinant_entries(semantic)
    return {
        **semantic,
        "registry_schema_version": (
            pack_cache.PACKING_CACHE_DETERMINANT_REGISTRY_VERSION
        ),
        "determinants": entries,
        "aggregate_fingerprint": pack_cache._registry_entries_fingerprint(entries),
        "code_identity": pack_cache._registry_code_identity(entries),
    }


def publish_characterized_pack_cache(cache_root: Path) -> tuple[Path, str]:
    """Publish one real pack cache through the production writer path."""

    determinants = _characterization_determinants(
        purpose="orchestration-characterization",
        split="train",
    )
    fingerprint = str(determinants["aggregate_fingerprint"])
    cache_dir = cache_dir_for_fingerprint(cache_root, fingerprint)
    write_micro_step_cache(
        cache_dir,
        (characterized_micro_step(), characterized_micro_step()),
        cache_root=cache_root,
        fingerprint=fingerprint,
        determinants=determinants,
        materialization=dict(_MATERIALIZATION),
        determinant_revalidator=lambda: determinants,
        augmentation=dict(_AUGMENTATION),
        chunk_size=1,
    )
    return cache_dir, fingerprint


# ---------------------------------------------------------------------------
# Ordered pipeline phase/collective characterization (two real ranks)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CharacterizedPipeline:
    """The frozen model-free orchestration evidence for one two-rank entry."""

    result: dict[str, Any]
    phase_order: dict[str, Any]
    collective_order: dict[str, Any]
    cache_admission: dict[str, Any]
    per_rank_results: dict[str, dict[str, Any]] = field(default_factory=dict)


def _have_gloo() -> bool:
    if not dist.is_available():
        return False
    available = getattr(dist, "is_gloo_available", None)
    return True if available is None else bool(available())


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _characterization_config(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        run=SimpleNamespace(
            name="characterization-run",
            artifact_root=str(root / "artifacts"),
            output_dir="run",
            collision_policy="fail",
        ),
        runtime=SimpleNamespace(
            seed=17,
            determinism=SimpleNamespace(mode="legacy"),
        ),
        training=SimpleNamespace(precision="no"),
        model=SimpleNamespace(attn_implementation="flash_attention_2"),
        data=SimpleNamespace(train=object(), eval=None),
    )


def _runtime_baseline_receipt() -> dict[str, Any]:
    return {
        "schema_version": 3,
        "baseline_sha256": "a" * 64,
        "attention_backend": "flash_attention_2",
        "admitted": True,
        "mismatches": [],
        "reference_only": {},
    }


def _bounded_phase_trace(trace: Mapping[str, Any]) -> dict[str, Any]:
    """Project the preflight phase trace without wall-clock or RSS values."""

    projected: dict[str, Any] = {}
    for phase, receipt in trace.items():
        entry: dict[str, Any] = {"status": receipt["status"]}
        details = receipt.get("details")
        if isinstance(details, Mapping):
            entry["details"] = dict(details)
        failure = receipt.get("failure")
        if isinstance(failure, Mapping):
            entry["failure"] = dict(failure)
        entry["resource_snapshot_keys"] = sorted(receipt["resource_snapshot"])
        projected[phase] = entry
    return projected


def _bounded_cache_admission(preflight: Mapping[str, Any]) -> dict[str, Any]:
    """Project the cache admission receipt to its non-derived contract."""

    train_cache = preflight["train_cache"]
    return {
        "cache_root_receipt": dict(preflight["cache_root_receipt"]),
        "eval_cache": preflight["eval_cache"],
        "eval_reduction": dict(preflight["eval_reduction"]),
        "rank": int(preflight["rank"]),
        "world_size": int(preflight["world_size"]),
        "train_cache": {
            "augmentation": dict(train_cache["augmentation"]),
            "build_status": train_cache["build_status"],
            "chunk_count": int(train_cache["chunk_count"]),
            "chunk_size": int(train_cache["chunk_size"]),
            "format_version": train_cache["format_version"],
            "key_set": sorted(train_cache),
            "materialization": dict(train_cache["materialization"]),
            "micro_step_count": int(train_cache["micro_step_count"]),
            "phase_receipt": {
                phase: {
                    key: value
                    for key, value in receipt.items()
                    if key != "duration_seconds"
                }
                for phase, receipt in train_cache["phase_receipt"].items()
            },
            "status": train_cache["status"],
        },
        "train_micro_step_count": len(preflight["train_micro_steps"]),
        "phase_trace": _bounded_phase_trace(preflight["phase_trace"]),
    }


def _characterization_worker(
    rank: int,
    preflight_port: int,
    accelerator_port: int,
    root: str,
    output: "mp.Queue[Any]",
) -> None:
    """Run one real rank of the model-free facade and record its ordered trace."""

    task_root = Path(root)
    config_path = (task_root / "config.yaml").resolve()
    config = _characterization_config(task_root)
    resolved = SimpleNamespace(
        config=config,
        fingerprint="config-fingerprint",
        entry_config_path=config_path,
        to_artifact_dict=lambda: {"characterization": True},
    )
    token_identity = SimpleNamespace(tokenizer_vocab_size=32)
    components = SimpleNamespace(token_identity=token_identity, tokenizer=object())
    fingerprint = (task_root / "fingerprint.txt").read_text(encoding="utf-8").strip()

    phase_order: list[str] = []
    collective_order: list[dict[str, Any]] = []
    captured_preflight: dict[str, Any] = {}
    gatherer_serial = {"count": 0}

    real_converged_phase = control_plane._run_rank_converged_phase
    real_gatherer_factory = control_plane._build_rank_report_gatherer
    real_preflight_resolver = pipeline._resolve_model_free_training_preflight

    def recording_converged_phase(phase: str, **kwargs: Any) -> Any:
        phase_order.append(phase)
        return real_converged_phase(phase, **kwargs)

    def recording_gatherer_factory(world_size: int) -> Any:
        gatherer = real_gatherer_factory(world_size)
        if gatherer is None:
            return None
        gatherer_serial["count"] += 1
        index = gatherer_serial["count"]

        def wrapped(report: Any) -> tuple[Any, ...]:
            entry = {"event": "all_gather", "gatherer_index": index}
            if isinstance(report, Mapping):
                entry["kind"] = report.get("kind")
                entry["split"] = report.get("split")
            collective_order.append(entry)
            return tuple(gatherer(report))

        def close() -> None:
            collective_order.append({"event": "close", "gatherer_index": index})
            close_inner = getattr(gatherer, "close", None)
            if callable(close_inner):
                close_inner()
            wrapped.closed = True  # type: ignore[attr-defined]

        wrapped.close = close  # type: ignore[attr-defined]
        wrapped.closed = False  # type: ignore[attr-defined]
        return wrapped

    def capturing_preflight(**kwargs: Any) -> Any:
        preflight = real_preflight_resolver(**kwargs)
        captured_preflight.update(_bounded_cache_admission(preflight))
        return preflight

    def build_accelerator(precision: str) -> object:
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://127.0.0.1:{accelerator_port}",
            rank=rank,
            world_size=_WORLD_SIZE,
        )
        return SimpleNamespace(
            process_index=rank,
            num_processes=_WORLD_SIZE,
            is_main_process=rank == 0,
        )

    def run_initialized_training(**kwargs: Any) -> dict[str, Any]:
        # Model-bearing training is out of CPU characterization scope.  The
        # facade contract under test is that it returns this owner's mapping
        # unchanged, so the stub returns the exact production-shaped keys.
        writer = kwargs["writer"]
        run_directory = kwargs["run_directory"]
        if writer is not None:
            writer.finalize(
                status="completed",
                updated_at="2026-01-01T00:00:00Z",
                completed_steps=1,
                consumed_packs=2,
                checkpoint_event_count=0,
                optimizer_update_status="applied",
                finite_status="finite",
            )
        return {
            "run_dir": str(run_directory.run_dir),
            "run_id": kwargs["run_id"],
            "resolved_config_fingerprint": kwargs["resolved_config"].fingerprint,
            "completed_steps": 1,
            "consumed_micro_steps": 2,
            "scheduled_event_counts": {"checkpoint": 0, "eval": 0},
        }

    try:
        os.environ.update(
            {
                "GLOO_SOCKET_IFNAME": "lo",
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(preflight_port),
                "RANK": str(rank),
                "LOCAL_RANK": str(rank),
                "WORLD_SIZE": str(_WORLD_SIZE),
                "COORDEXP_SWIFT_PACK_CACHE_ROOT": str(task_root / "cache-root"),
            }
        )
        for name in (
            "COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE",
            "COORDEXP_SWIFT_EVAL_REDUCTION_MODE",
            "COORDEXP_SWIFT_PROFILE_SYNC_TIMINGS",
        ):
            os.environ.pop(name, None)

        execution_plan.load_train_config = lambda path: resolved
        pipeline.collect_execution_provenance = lambda **kwargs: {"schema_version": 1}
        pipeline.require_pinned_runtime_baseline = (
            lambda **kwargs: _runtime_baseline_receipt()
        )
        pipeline.load_qwen_components = lambda config, *, load_model: (
            (_ for _ in ()).throw(
                AssertionError("model load must stay outside CPU characterization")
            )
            if load_model
            else components
        )
        pipeline.build_token_vocabulary_groups = lambda *args, **kwargs: object()
        pipeline.resolve_qwen_runtime_controls = lambda *args, **kwargs: object()
        pipeline.build_packing_cache_fingerprint = lambda *args, **kwargs: fingerprint
        pipeline.resolve_planned_step_schedule = (
            lambda *args, **kwargs: SimpleNamespace(
                resolved_max_steps=1,
                runtime_batch=SimpleNamespace(
                    world_size=int(kwargs["world_size"]),
                    resolved_grad_accum_steps=1,
                    effective_batch_size=int(kwargs["world_size"]),
                ),
            )
        )
        control_plane._run_rank_converged_phase = recording_converged_phase
        control_plane._build_rank_report_gatherer = recording_gatherer_factory
        pipeline._resolve_model_free_training_preflight = capturing_preflight
        pipeline._build_accelerator = build_accelerator
        pipeline.validate_accelerator_runtime = lambda *args, **kwargs: None
        pipeline._run_initialized_training = run_initialized_training

        result = pipeline.run_training_pipeline(config_path)
        run_state = json.loads(
            (task_root / "artifacts" / "run" / "run.json").read_text(encoding="utf-8")
        )
        if dist.is_initialized():
            dist.destroy_process_group()
        output.put(
            {
                "rank": rank,
                "result": dict(result),
                "phase_order": list(phase_order),
                "collective_order": list(collective_order),
                "cache_admission": dict(captured_preflight),
                "run_state_phase_order": list(
                    run_state["measurement"]["phase_order"]
                ),
                "run_state_phase_status": {
                    name: receipt["status"]
                    for name, receipt in run_state["measurement"]["phases"].items()
                },
                "run_state_status": run_state["status"],
                "run_state_materializations": dict(run_state["materializations"]),
                "run_state_policy_identity_names": sorted(
                    run_state["policy_identities"]
                ),
            }
        )
    except BaseException as exc:  # pragma: no cover - reported to the parent
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        output.put(
            {
                "rank": rank,
                "worker_error": f"{type(exc).__name__}: {exc}",
                "phase_order": list(phase_order),
                "collective_order": list(collective_order),
            }
        )


def _reap(processes: Sequence[Any], output: Any, *, expected: int) -> list[Any]:
    deadline = time.monotonic() + _CHARACTERIZATION_DEADLINE_SECONDS
    messages: list[Any] = []
    timed_out = False
    try:
        while True:
            while len(messages) < expected:
                try:
                    messages.append(output.get_nowait())
                except Empty:
                    break
            alive = []
            for process in processes:
                process.join(timeout=0)
                if process.is_alive():
                    alive.append(process)
            if not alive and len(messages) >= expected:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            try:
                messages.append(
                    output.get(timeout=min(_CHARACTERIZATION_POLL_SECONDS, remaining))
                )
            except Empty:
                continue
    except BaseException:
        # Never leak a spawned rank on an unexpected failure, a KeyboardInterrupt,
        # or a pytest timeout: mirror `_reap_spawned_rank_processes` in
        # `tests/training/test_pipeline_cache_preflight.py`.
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=_TERMINATE_GRACE_SECONDS)
        raise

    if timed_out:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=_TERMINATE_GRACE_SECONDS)
        raise AssertionError(
            "two-rank orchestration characterization timed out after "
            f"{_CHARACTERIZATION_DEADLINE_SECONDS:.1f}s with "
            f"{len(messages)}/{expected} results"
        )
    return messages


def exercise_characterized_pipeline() -> CharacterizedPipeline:
    """Run the real two-rank model-free facade and return its ordered evidence."""

    with tempfile.TemporaryDirectory(prefix="coordexp-orchestration-") as raw_root:
        root = Path(raw_root).resolve()
        (root / "config.yaml").write_text("characterization: true\n", encoding="utf-8")
        cache_root = root / "cache-root"
        _, fingerprint = publish_characterized_pack_cache(cache_root)
        (root / "fingerprint.txt").write_text(fingerprint, encoding="utf-8")

        context = mp.get_context("spawn")
        output: "mp.Queue[Any]" = context.Queue()
        preflight_port = _free_port()
        accelerator_port = _free_port()
        while accelerator_port == preflight_port:
            accelerator_port = _free_port()
        processes = [
            context.Process(
                target=_characterization_worker,
                args=(rank, preflight_port, accelerator_port, str(root), output),
                daemon=False,
            )
            for rank in range(_WORLD_SIZE)
        ]
        for process in processes:
            process.start()
        messages = _reap(processes, output, expected=_WORLD_SIZE)
        exitcodes = [process.exitcode for process in processes]

        failures = [message for message in messages if "worker_error" in message]
        if failures:
            raise AssertionError(
                "two-rank orchestration characterization failed: "
                + json.dumps(failures, indent=2, sort_keys=True, default=str)
            )
        if exitcodes != [0, 0]:
            raise AssertionError(
                f"two-rank characterization exit codes were {exitcodes}"
            )
        by_rank = {int(message["rank"]): message for message in messages}
        if sorted(by_rank) != [0, 1]:
            raise AssertionError(f"missing rank results: {sorted(by_rank)}")

        root_text = str(root)
        per_rank_results = {
            str(rank): _normalize_paths(message["result"], root=root_text)
            for rank, message in by_rank.items()
        }
        return CharacterizedPipeline(
            result=per_rank_results["0"],
            phase_order={
                "scope": _PIPELINE_TRACE_SCOPE,
                "world_size": _WORLD_SIZE,
                "converged_phase_order": {
                    str(rank): list(message["phase_order"])
                    for rank, message in by_rank.items()
                },
                "run_state_phase_order": list(by_rank[0]["run_state_phase_order"]),
                "run_state_phase_status": dict(by_rank[0]["run_state_phase_status"]),
                "run_state_status": by_rank[0]["run_state_status"],
                "run_state_policy_identity_names": list(
                    by_rank[0]["run_state_policy_identity_names"]
                ),
            },
            collective_order={
                "scope": _PIPELINE_TRACE_SCOPE,
                "world_size": _WORLD_SIZE,
                "transport": (
                    "src.training.pipeline._build_rank_report_gatherer over gloo"
                ),
                "per_rank": {
                    str(rank): list(message["collective_order"])
                    for rank, message in by_rank.items()
                },
            },
            cache_admission=_normalize_paths(
                {
                    "per_rank": {
                        str(rank): message["cache_admission"]
                        for rank, message in by_rank.items()
                    },
                    "run_state_materializations": by_rank[0][
                        "run_state_materializations"
                    ],
                },
                root=root_text,
            ),
            per_rank_results=per_rank_results,
        )


# ---------------------------------------------------------------------------
# RunWriter byte characterization
# ---------------------------------------------------------------------------


def _characterized_resource_snapshot(scale: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "cpu": {
            "scope": "current_process",
            "max_rss_bytes": 100 * scale,
            "io_read_bytes": 10 * scale,
            "io_write_bytes": 20 * scale,
        },
        "gpu": {
            "scope": "current_process_current_device",
            "initialized": False,
            "unavailable_reason": "cuda_not_initialized",
        },
    }


def exercise_characterized_run_writer(tmp_path: Path) -> dict[str, bytes]:
    """Drive one representative RunWriter lifetime and return its exact bytes."""

    from src.artifacts.run_writer import RunWriter

    root = Path(tmp_path).resolve()
    run_dir = root / "artifacts" / "run"
    writer = RunWriter.initialize(
        run_dir=run_dir,
        run_id="characterized-run-000000000000",
        run_name="characterized-run",
        artifact_root=root / "artifacts",
        collision_outcome="created",
        created_at="2026-01-01T00:00:00Z",
        config_fingerprint="c" * 64,
        resolved_config={"characterization": True, "nested": {"value": 1}},
        world_size=2,
        resolved_max_steps=None,
        provenance={"schema_version": 1, "source": "characterization"},
        measurement_context={
            "comparison_arm": "unclassified",
            "wall_clock_scope": "training_entry_to_terminal_artifact",
            "warmup_exclusion_steps": 1,
            "workload_identity": "c" * 64,
            "world_size": 2,
        },
        entry_started_at="2026-01-01T00:00:00Z",
        segment_id="segment-00000000000000000000000000000000",
    )
    writer.bind_schedule(resolved_max_steps=5)
    writer.bind_policy_identity(
        "upstream_runtime_baseline",
        {"schema_version": 3, "baseline_sha256": "a" * 64, "admitted": True},
    )
    writer.bind_policy_identity(
        "cache",
        {"schema_version": 1, "root": {"resolved_root": "<cache>", "source": "default"}},
    )
    writer.bind_forward_input_provider_mode(
        "synchronous",
        resolution={"schema_version": 1, "resolved_mode": "synchronous"},
    )
    writer.bind_materialization(
        "train",
        cache_format_version=PACKING_CACHE_VERSION,
        semantic_fingerprint="f" * 64,
        determinant_digest="d" * 64,
    )
    writer.begin_phase(
        "config_provenance_resolution",
        started_at="2026-01-01T00:00:01Z",
        resources=_characterized_resource_snapshot(1),
    )
    writer.finish_phase(
        "config_provenance_resolution",
        status="completed",
        completed_at="2026-01-01T00:00:02Z",
        duration_seconds=1.0,
        resources=_characterized_resource_snapshot(2),
    )
    writer.record_completed_phase(
        "cache_identity_resolution",
        completed_at="2026-01-01T00:00:03Z",
        duration_seconds=0.5,
    )
    writer.record_phase_not_run(
        "cache_preparation",
        reason="single_process_preparation_precedes_training_launch",
    )
    writer.record_phase_not_run("cache_publication", reason="immutable_cache_hit")
    writer.begin_phase("cache_admission", started_at="2026-01-01T00:00:04Z")
    writer.finish_phase(
        "cache_admission",
        status="completed",
        completed_at="2026-01-01T00:00:05Z",
        duration_seconds=1.0,
    )
    writer.observe_resources(_characterized_resource_snapshot(3))
    writer.begin_phase("first_optimizer_step", started_at="2026-01-01T00:00:06Z")
    writer.append_logging_row(
        {
            "step": 1,
            "split": "train",
            "micro_step_count": 2,
            "optimizer_update_status": "applied",
            "finite_status": "finite",
            "loss/total": 1.5,
            "acc_top1": 0.5,
            "accuracy_stats": {"top1_correct": 1, "top5_correct": 2, "atom_count": 2},
        }
    )
    writer.append_logging_row(
        {
            "step": 1,
            "split": "eval",
            "example_count": 2,
            "pack_count": 1,
            "acc_top1": 0.6,
            "trigger_reasons": ["scheduled"],
        }
    )
    writer.finish_phase(
        "first_optimizer_step",
        status="completed",
        completed_at="2026-01-01T00:00:07Z",
        duration_seconds=1.0,
    )
    writer.record_phase_summary(
        "steady_state",
        status="completed",
        completed_at="2026-01-01T00:00:08Z",
        duration_seconds=2.0,
        duration_scope="sum_of_accepted_all_rank_max_step_durations",
        accepted_measured_steps=1,
        expected_measured_steps=1,
    )
    writer.record_phase_summary(
        "evaluation_execution",
        status="not_run",
        completed_at=None,
        duration_seconds=0.0,
        duration_scope="sum_of_all_rank_max_evaluation_event_durations",
        event_count=0,
        reason="eval_not_scheduled",
    )
    writer.record_warning("characterized_warning", count=2)
    writer.record_checkpoint_publication_event(
        step=1,
        status="failed",
        started_at="2026-01-01T00:00:09Z",
        completed_at="2026-01-01T00:00:10Z",
        duration_seconds=1.0,
        is_final=False,
        exact_training_state_enabled=False,
        checkpoint_identity=None,
        inference_payload_identity=None,
        committed_progress=None,
        failure_code="characterized_publication_failure",
    )
    writer.write_final(step=1)
    writer.write_best(
        step=1,
        selector="acc_top1",
        value=0.5,
        optimizer_update_status="applied",
        finite_status="finite",
        checkpoint_committed=True,
    )
    writer.finalize(
        status="completed",
        updated_at="2026-01-01T00:00:11Z",
        completed_steps=1,
        consumed_packs=2,
        checkpoint_event_count=1,
        optimizer_update_status="applied",
        finite_status="finite",
    )

    root_text = str(root)
    tree: dict[str, bytes] = {}
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        payload = path.read_bytes()
        tree[path.relative_to(run_dir).as_posix()] = payload.replace(
            root_text.encode("utf-8"), PATH_PLACEHOLDER.encode("utf-8")
        )
    return tree


# ---------------------------------------------------------------------------
# Completed-step logging rows
# ---------------------------------------------------------------------------


class _CharacterizationRuntime:
    """The minimal reduction surface the completed-step callback consumes."""

    is_main_process = True
    world_size = 1

    class _Accelerator:
        is_main_process = True
        num_processes = 1
        process_index = 0

    accelerator = _Accelerator()

    def gather_metrics(self, metrics: Mapping[str, Any], **kwargs: Any) -> Any:
        result: dict[str, Any] = {"metrics": dict(metrics)}
        accuracy_stats = kwargs.get("accuracy_stats")
        if isinstance(accuracy_stats, Mapping):
            result["accuracy_stats"] = dict(accuracy_stats)
        return result


def _characterized_observation(step: int) -> CompletedStepObservation:
    return CompletedStepObservation(
        planned_step_id=step,
        micro_step_count=2,
        loss_bundle_artifact={
            "metrics": {"loss/total": 1.0 + step, "acc_top1": 0.5, "acc_top5": 1.0},
            "accuracy_stats": {
                "top1_correct": 1,
                "top5_correct": 2,
                "atom_count": 2,
            },
        },
        optimizer_update_status="applied",
        finite_status="finite",
        scheduler_artifact={"learning_rates": [{"group_index": 0, "lr": 1e-5}]},
        step_duration_seconds=0.25,
        input_build_seconds=0.05,
        input_wait_seconds=0.01,
    )


def exercise_characterized_completed_step_rows(tmp_path: Path) -> dict[str, Any]:
    """Drive the current completed-step callback and return its exact rows."""

    from src.artifacts.run_writer import RunWriter

    root = Path(tmp_path).resolve()
    writer = RunWriter.initialize(
        run_dir=root / "run",
        run_id="rows-000000000000",
        run_name="rows",
        artifact_root=root,
        collision_outcome="created",
        created_at="2026-01-01T00:00:00Z",
        config_fingerprint="c" * 64,
        resolved_config={},
        world_size=1,
        resolved_max_steps=3,
        segment_id="segment-00000000000000000000000000000000",
    )
    lifecycle: dict[str, Any] = {
        "active_phase": None,
        "consumed_packs": 0,
        "measurement_warmup_steps": 1,
        "resolved_max_steps": 3,
        "expected_measured_steps": 2,
        "accepted_measured_steps": 0,
        "steady_state_duration_seconds": 0.0,
        "steady_state_rank_resources": None,
    }
    handle = pipeline._train_logging_handler(
        writer,
        lifecycle,
        _CharacterizationRuntime(),
    )
    for step in (1, 2, 3):
        handle(_characterized_observation(step))
    rows = [
        json.loads(line)
        for line in writer.logging_path.read_text(encoding="utf-8").splitlines()
    ]
    return {
        "rows": rows,
        "lifecycle": {
            "accepted_measured_steps": lifecycle["accepted_measured_steps"],
            "completed_steps": lifecycle["completed_steps"],
            "consumed_packs": lifecycle["consumed_packs"],
            "finite_status": lifecycle["finite_status"],
            "optimizer_update_status": lifecycle["optimizer_update_status"],
            "steady_state_duration_seconds": lifecycle[
                "steady_state_duration_seconds"
            ],
        },
    }


# ---------------------------------------------------------------------------
# Cache preparation receipt
# ---------------------------------------------------------------------------


def exercise_characterized_cache_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Characterize ``prepare_training_pack_caches`` receipt projection."""

    root = Path(tmp_path).resolve()
    config = SimpleNamespace(
        runtime=SimpleNamespace(seed=17, determinism=SimpleNamespace(mode="legacy")),
        model=SimpleNamespace(attn_implementation="flash_attention_2"),
        packing=SimpleNamespace(
            global_max_length=8,
            policy="source_order_next_fit",
            window_size=None,
            lookahead=None,
            seed=17,
            worker_count=1,
            cursor_byte_budget=65_536,
            max_packs_per_fragment=None,
            fragment_item_budget=1_024,
            fragment_byte_budget=4_194_304,
        ),
        data=SimpleNamespace(train=object(), eval=object(), train_order="source_order"),
        template=SimpleNamespace(object_ordering="geo_sorted"),
    )
    resolved = SimpleNamespace(
        config=config,
        fingerprint="config-fingerprint",
        entry_config_path=root / "config.yaml",
    )
    components = SimpleNamespace(token_identity=object(), tokenizer=object())
    train_cache = {
        "status": "complete",
        "build_status": "built",
        "cache_dir": root / "train-cache",
        "format_version": PACKING_CACHE_VERSION,
        "fingerprint": "a" * 64,
        "manifest_path": root / "train-cache" / "manifest.json",
        "manifest_sha256": "b" * 64,
        "micro_step_count": 11,
        "phase_receipt": {
            "cache_preparation": {"status": "completed", "duration_seconds": 0.1},
            "cache_publication": {"status": "completed", "duration_seconds": 0.2},
            "cache_admission": {"status": "completed", "duration_seconds": 0.3},
        },
    }
    eval_cache = {
        **train_cache,
        "cache_dir": root / "eval-cache",
        "fingerprint": "c" * 64,
        "manifest_path": root / "eval-cache" / "manifest.json",
        "manifest_sha256": "d" * 64,
        "micro_step_count": 3,
        "phase_receipt": {
            "cache_preparation": {
                "status": "not_run_cache_hit",
                "duration_seconds": 0.0,
            },
            "cache_publication": {
                "status": "not_run_cache_hit",
                "duration_seconds": 0.0,
            },
            "cache_admission": {"status": "completed", "duration_seconds": 0.4},
        },
    }
    monkeypatch.setattr(pipeline, "load_train_config", lambda path: resolved)
    monkeypatch.setattr(
        pipeline,
        "collect_execution_provenance",
        lambda **kwargs: {"schema_version": 1},
    )
    monkeypatch.setattr(
        pipeline,
        "require_pinned_runtime_baseline",
        lambda **kwargs: _runtime_baseline_receipt(),
    )
    monkeypatch.setattr(
        pipeline,
        "load_qwen_components",
        lambda config, *, load_model: components,
    )
    monkeypatch.setattr(
        pipeline, "build_token_vocabulary_groups", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        pipeline,
        "_resolve_or_build_train_pack_cache",
        lambda *args, **kwargs: train_cache,
    )
    monkeypatch.setattr(
        pipeline, "_resolve_eval_pack_cache", lambda *args, **kwargs: eval_cache
    )
    monkeypatch.setenv("COORDEXP_SWIFT_PACK_CACHE_ROOT", str(root / "cache-root"))

    result = pipeline.prepare_training_pack_caches(root / "config.yaml")
    receipt = _normalize_paths(copy.deepcopy(result), root=str(root))
    # Wall-clock stamps, live durations, and process RSS are environment-derived.
    # Their presence, key set, and type are the contract; their values are not.
    measurement = receipt["measurement"]
    measurement["started_at"] = _volatile(measurement["started_at"])
    measurement["completed_at"] = _volatile(measurement["completed_at"])
    measurement["duration_seconds"] = _volatile(measurement["duration_seconds"])
    entry_phase = measurement["phases"]["config_provenance_resolution"]
    entry_phase["duration_seconds"] = _volatile(entry_phase["duration_seconds"])
    high_water = measurement["resource_high_water"]
    high_water["cpu"] = {
        key: value if key == "scope" else _volatile(value)
        for key, value in high_water["cpu"].items()
    }
    return {
        "receipt": receipt,
        "aggregate_phase_cases": {
            "all_hit": pipeline._aggregate_cache_phase(
                [eval_cache, eval_cache], "cache_preparation"
            ),
            "mixed": pipeline._aggregate_cache_phase(
                [train_cache, eval_cache], "cache_preparation"
            ),
            "all_built": pipeline._aggregate_cache_phase(
                [train_cache, train_cache], "cache_preparation"
            ),
        },
    }


# ---------------------------------------------------------------------------
# Causal logits selection
# ---------------------------------------------------------------------------


def exercise_characterized_causal_logits() -> dict[str, Any]:
    """Characterize causal-logit position selection at its current owner.

    Wave 1 deletes ``supervised_trainer._logits_positions_to_keep`` in favour of
    ``TokenSequence.causal_logits_positions`` (design decision 6).  The frozen
    manifest's ``harness_seam_repoints`` rule re-points this call to the new
    owner and requires the selected positions and the empty/absent-atom results
    to replay unchanged; ``causal_logits.json`` is not a declared flip.
    """

    def select_causal_positions(
        micro_step: SupervisedMicroStep,
    ) -> tuple[int, ...] | None:
        return TokenSequence.causal_logits_positions(micro_step.token_sequence)

    sequence = _characterized_token_sequence()
    populated = SupervisedMicroStep(
        pack="pack-0",
        encoded_examples=(),
        position_inputs="positions",
        token_sequence=sequence,
        vocab_groups=_characterized_vocab_groups(),
    )
    empty_atoms = SupervisedMicroStep(
        pack="pack-0",
        encoded_examples=(),
        position_inputs="positions",
        token_sequence=SimpleNamespace(atoms=()),
        vocab_groups=_characterized_vocab_groups(),
    )
    no_atoms_attribute = SupervisedMicroStep(
        pack="pack-0",
        encoded_examples=(),
        position_inputs="positions",
        token_sequence=SimpleNamespace(),
        vocab_groups=_characterized_vocab_groups(),
    )
    return {
        "atom_causal_logits_positions": [
            atom.causal_logits_position for atom in sequence.atoms
        ],
        "atom_target_positions": [atom.target_position for atom in sequence.atoms],
        "populated": list(select_causal_positions(populated)),
        "empty_atoms": select_causal_positions(empty_atoms),
        "missing_atoms_attribute": select_causal_positions(no_atoms_attribute),
        "duplicate_free_sorted": True,
    }


# ---------------------------------------------------------------------------
# Generic identity helpers currently owned by src.qwen.parity
# ---------------------------------------------------------------------------


def _bounded_failure(exc: BaseException) -> dict[str, Any]:
    code = getattr(exc, "code", None)
    context = getattr(exc, "context", None)
    return {
        "type": type(exc).__name__,
        "code": code if isinstance(code, str) else None,
        "context_keys": sorted(context) if isinstance(context, Mapping) else None,
    }


def _capture_failure(body: Any) -> dict[str, Any]:
    try:
        body()
    except BaseException as exc:
        return _bounded_failure(exc)
    raise AssertionError("expected a bounded identity failure")


def exercise_characterized_identity(tmp_path: Path) -> dict[str, Any]:
    """Characterize the generic identity helpers and their failure contracts."""

    root = Path(tmp_path).resolve()
    (root / "owned").mkdir(parents=True, exist_ok=True)
    source = root / "owned" / "source.txt"
    source.write_bytes(b"characterized-source-bytes\n")

    payload = {"b": 2, "a": [1, {"z": None, "y": True}], "unicode": "é"}
    weight_identity_body = {
        "schema": parity_identity.MODEL_WEIGHT_IDENTITY_SCHEMA,
        "root": "/models/characterized-base",
        "mode": "standalone_safetensors",
        "index": None,
        "declaration_count": None,
        "shards": [
            {
                "path": "model.safetensors",
                "size_bytes": 4,
                "sha256": "e" * 64,
            }
        ],
        "shard_count": 1,
        "total_bytes": 4,
        "bounds": {
            "max_index_bytes": parity_identity.MAX_WEIGHT_INDEX_BYTES,
            "max_declarations": parity_identity.MAX_WEIGHT_DECLARATIONS,
            "max_shards": parity_identity.MAX_WEIGHT_SHARDS,
            "max_shard_bytes": parity_identity.MAX_WEIGHT_SHARD_BYTES,
            "max_total_bytes": parity_identity.MAX_WEIGHT_TOTAL_BYTES,
        },
    }
    weight_identity = {
        **weight_identity_body,
        "aggregate_sha256": parity_identity.sha256_json(weight_identity_body),
    }

    drifted_body = copy.deepcopy(weight_identity_body)
    drifted_body["shards"][0]["sha256"] = "1" * 64
    drifted_identity = {
        **drifted_body,
        "aggregate_sha256": parity_identity.sha256_json(drifted_body),
    }

    published = root / "published.json"
    parity_identity.write_strict_json_atomic(published, {"published": True})

    validated: Any
    try:
        validated = parity_identity.validate_model_weight_identity(weight_identity)
    except BaseException as exc:
        validated = _bounded_failure(exc)

    return {
        "canonical_json_bytes": parity_identity.canonical_json_bytes(payload).decode(
            "utf-8"
        ),
        "sha256_json": parity_identity.sha256_json(payload),
        "sha256_file": parity_identity.sha256_file(source),
        "source_owner_identity": parity_identity.source_owner_identity(
            root, ["owned/source.txt"]
        ),
        "repo_identity_key_set": sorted(parity_identity.repo_identity(REPO_ROOT)),
        "write_strict_json_atomic_bytes": published.read_bytes().decode("utf-8"),
        "validate_model_weight_identity": validated,
        "failures": {
            "canonical_json_bytes_non_json": _capture_failure(
                lambda: parity_identity.canonical_json_bytes({"value": object()})
            ),
            "canonical_json_bytes_non_finite": _capture_failure(
                lambda: parity_identity.canonical_json_bytes({"value": float("nan")})
            ),
            "sha256_file_missing": _capture_failure(
                lambda: parity_identity.sha256_file(root / "absent.txt")
            ),
            "assert_absent_artifact_target_occupied": _capture_failure(
                lambda: parity_identity.assert_absent_artifact_target(published)
            ),
            "assert_absent_artifact_target_missing_parent": _capture_failure(
                lambda: parity_identity.assert_absent_artifact_target(
                    root / "absent-dir" / "target.json"
                )
            ),
            "write_strict_json_atomic_collision": _capture_failure(
                lambda: parity_identity.write_strict_json_atomic(
                    published, {"published": True}
                )
            ),
            "validate_model_weight_identity_incomplete": _capture_failure(
                lambda: parity_identity.validate_model_weight_identity(
                    {"schema": parity_identity.MODEL_WEIGHT_IDENTITY_SCHEMA}
                )
            ),
            "assert_model_weight_identity_equal_drift": _capture_failure(
                lambda: parity_identity.assert_model_weight_identity_equal(
                    weight_identity, drifted_identity
                )
            ),
            "assert_model_weight_identity_equal_corrupt_fingerprint": _capture_failure(
                lambda: parity_identity.assert_model_weight_identity_equal(
                    weight_identity,
                    {**weight_identity, "aggregate_sha256": "0" * 64},
                )
            ),
            "source_owner_identity_escape": _capture_failure(
                lambda: parity_identity.source_owner_identity(root, ["../escape.txt"])
            ),
        },
        "assert_absent_artifact_target_success": str(
            parity_identity.assert_absent_artifact_target(root / "new-target.json")
        ).replace(str(root), PATH_PLACEHOLDER),
    }


# ---------------------------------------------------------------------------
# Legacy SupervisedMicroStep pickle bytes
# ---------------------------------------------------------------------------


def _decoded_micro_step_values(micro_step: Any) -> dict[str, Any]:
    sequence = micro_step.token_sequence
    return {
        "type_module": type(micro_step).__module__,
        "type_name": type(micro_step).__name__,
        "pack": micro_step.pack,
        "encoded_examples": list(micro_step.encoded_examples),
        "position_inputs": micro_step.position_inputs,
        "metadata": None if micro_step.metadata is None else dict(micro_step.metadata),
        "forward_device": micro_step.forward_device,
        "expected_vocab_size": micro_step.expected_vocab_size,
        "extra_model_kwargs": micro_step.extra_model_kwargs,
        "fa2_branch_evidence": micro_step.fa2_branch_evidence,
        "fa2_model_dtype": micro_step.fa2_model_dtype,
        "capture_fa2_branch": micro_step.capture_fa2_branch,
        "require_fa2_branch_proof": micro_step.require_fa2_branch_proof,
        "fa2_branch_proof_policy": micro_step.fa2_branch_proof_policy,
        "token_sequence": sequence.to_artifact_dict(),
        "vocab_groups": {
            "vocab_size": micro_step.vocab_groups.vocab_size,
            "desc_text": list(micro_step.vocab_groups.desc_text),
            "schema": list(micro_step.vocab_groups.schema),
            "coordinate": list(micro_step.vocab_groups.coordinate),
            "eos": list(micro_step.vocab_groups.eos),
            "blocked": list(micro_step.vocab_groups.blocked),
        },
    }


def build_legacy_micro_step_pickle(destination: Path) -> Path:
    """Publish one pack cache and copy its real chunk pickle to ``destination``."""

    with tempfile.TemporaryDirectory(prefix="coordexp-legacy-pkl-") as raw_root:
        cache_root = Path(raw_root).resolve() / "cache-root"
        determinants = _characterization_determinants(
            purpose="legacy-micro-step-pickle",
            split="train",
        )
        fingerprint = str(determinants["aggregate_fingerprint"])
        cache_dir = cache_dir_for_fingerprint(cache_root, fingerprint)
        manifest = write_micro_step_cache(
            cache_dir,
            (characterized_micro_step(),),
            cache_root=cache_root,
            fingerprint=fingerprint,
            determinants=determinants,
            materialization=dict(_MATERIALIZATION),
            determinant_revalidator=lambda: determinants,
            augmentation=dict(_AUGMENTATION),
            chunk_size=1,
        )
        chunk_relative = str(manifest["chunks"][0]["path"])
        payload = (cache_dir / chunk_relative).read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return destination


def restricted_load(payload: bytes) -> tuple[Any, ...]:
    """Decode one cache chunk through the production restricted unpickler."""

    import io

    return pack_cache._RestrictedCacheUnpickler(io.BytesIO(payload)).load()


def load_legacy_micro_step_pickle() -> tuple[Any, ...]:
    """Load the frozen legacy chunk through the production restricted unpickler."""

    return restricted_load(LEGACY_MICRO_STEP_PICKLE.read_bytes())


def wave1_revised_legacy_micro_step_expectation() -> dict[str, Any]:
    """The frozen Wave-0 fixture plus exactly the two Wave-1 declared deltas.

    The fixture bytes are never rewritten.  Wave 1 declares (manifest
    ``declared_flips``) that the record's canonical owner becomes
    ``src.training.micro_steps``, so the decoded class ``__module__`` changes and
    the restricted allowlist gains the canonical module path while keeping the
    historical one.  Everything derived from the frozen pickle bytes - the
    digest, byte length, embedded module paths, and schema identity - must stay
    exactly equal.
    """

    expected = load_fixture("legacy_micro_step.json")
    for decoded in expected["decoded"]:
        decoded["type_module"] = "src.training.micro_steps"
    expected["restricted_pickle_allowlist"] = sorted(
        {
            *expected["restricted_pickle_allowlist"],
            "src.training.micro_steps:SupervisedMicroStep",
        }
    )
    return expected


def exercise_characterized_legacy_micro_step() -> dict[str, Any]:
    """Characterize the frozen legacy pickle bytes, module path, and values."""

    payload = LEGACY_MICRO_STEP_PICKLE.read_bytes()
    decoded = load_legacy_micro_step_pickle()
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "byte_length": len(payload),
        "pickle_module_paths": sorted(
            {
                name.decode("utf-8")
                for name in (
                    b"src.training.supervised_trainer",
                    b"src.training.micro_steps",
                    b"src.supervision.tokens",
                    b"src.packing.planner",
                    b"src.losses.vocab",
                )
                if name in payload
            }
        ),
        "micro_step_count": len(decoded),
        "decoded": [_decoded_micro_step_values(step) for step in decoded],
        "schema_identity": pack_cache._supervised_micro_step_schema_identity(),
        "restricted_pickle_allowlist": sorted(
            f"{module}:{name}" for module, name in pack_cache._ALLOWED_PICKLE_GLOBALS
        ),
    }


# ---------------------------------------------------------------------------
# Entrypoint result mapping
# ---------------------------------------------------------------------------


def exercise_characterized_entrypoint(capsys: Any) -> dict[str, Any]:
    """Characterize ``src.train.main`` result mapping with an injected runner."""

    observed_paths: list[str] = []

    def runner(config_path: Path) -> Mapping[str, Any]:
        observed_paths.append(str(config_path))
        return {
            "run_dir": "/artifacts/run",
            "run_id": "characterized-run-000000000000",
            "resolved_config_fingerprint": "config-fingerprint",
            "completed_steps": 1,
            "consumed_micro_steps": 2,
            "scheduled_event_counts": {"checkpoint": 0, "eval": 0},
        }

    exit_code = train_entrypoint.main(
        ["--config", "configs/characterization.yaml"],
        runner=runner,
    )
    printed = capsys.readouterr().out
    return {
        "exit_code": exit_code,
        "runner_config_paths": observed_paths,
        "stdout": printed,
        "summary": json.loads(printed),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _have_gloo(), reason="requires torch.distributed gloo backend")
def test_pipeline_characterization_matches_frozen_result_and_order() -> None:
    observed = exercise_characterized_pipeline()

    assert observed.result == load_fixture("pipeline_result.json")
    assert observed.phase_order == load_fixture("phase_order.json")
    assert observed.collective_order == load_fixture("collective_order.json")
    assert observed.cache_admission == load_fixture("cache_admission.json")
    assert observed.per_rank_results["0"] == observed.per_rank_results["1"]


def test_run_writer_characterization_matches_exact_bytes(tmp_path: Path) -> None:
    observed = exercise_characterized_run_writer(tmp_path)

    assert observed == load_binary_fixture_tree("run_writer")


def test_run_writer_characterization_is_byte_reproducible(tmp_path: Path) -> None:
    first = exercise_characterized_run_writer(tmp_path / "a")
    second = exercise_characterized_run_writer(tmp_path / "b")

    assert first == second


def test_completed_step_rows_match_frozen_bytes(tmp_path: Path) -> None:
    observed = exercise_characterized_completed_step_rows(tmp_path)

    assert observed == load_fixture("completed_step_rows.json")


def test_cache_preparation_receipt_matches_frozen_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = exercise_characterized_cache_preparation(tmp_path, monkeypatch)

    assert observed == load_fixture("cache_preparation.json")


def test_causal_logits_selection_matches_frozen_positions() -> None:
    observed = exercise_characterized_causal_logits()

    assert observed == load_fixture("causal_logits.json")


def test_generic_identity_success_and_failure_cases_match_frozen_contract(
    tmp_path: Path,
) -> None:
    observed = exercise_characterized_identity(tmp_path)

    assert observed == load_fixture("identity_cases.json")


def test_legacy_micro_step_pickle_bytes_and_decoded_values_are_frozen() -> None:
    """Wave-1 declared flip: canonical owner module path and allowlist."""

    frozen = load_fixture("legacy_micro_step.json")
    observed = exercise_characterized_legacy_micro_step()

    assert observed == wave1_revised_legacy_micro_step_expectation()
    # Everything the frozen bytes determine is unchanged by the owner move.
    assert observed["sha256"] == frozen["sha256"]
    assert observed["byte_length"] == frozen["byte_length"]
    assert observed["pickle_module_paths"] == frozen["pickle_module_paths"]
    assert observed["schema_identity"] == frozen["schema_identity"]


def test_legacy_micro_step_pickle_carries_the_historical_module_path() -> None:
    payload = LEGACY_MICRO_STEP_PICKLE.read_bytes()

    assert b"src.training.supervised_trainer" in payload
    assert b"src.training.micro_steps" not in payload


def test_legacy_micro_step_pickle_loads_through_the_restricted_unpickler() -> None:
    """Wave-1 declared flip: the historical path resolves to the new owner.

    The frozen bytes still name ``src.training.supervised_trainer``; the
    restricted unpickler resolves that historical global through the
    compatibility re-export, so the decoded class now reports the canonical
    owner module.  The decoded values stay equal.
    """

    decoded = load_legacy_micro_step_pickle()

    assert len(decoded) == 1
    assert type(decoded[0]).__module__ == "src.training.micro_steps"
    assert decoded[0] == characterized_micro_step()


def test_legacy_micro_step_pickle_is_the_real_writer_output() -> None:
    """Wave-1 declared flip: the writer now publishes the canonical path.

    New publications pickle under ``src.training.micro_steps``, so their bytes
    are intentionally different from the frozen legacy payload and must never be
    characterized as byte-identical to it (design decision 4).  What the frozen
    node proved - that the fixture is real writer output, reproducibly - is kept
    by regenerating twice and decoding through the production reader.
    """

    with tempfile.TemporaryDirectory(prefix="coordexp-legacy-check-") as raw_root:
        first = build_legacy_micro_step_pickle(Path(raw_root) / "first.pkl")
        second = build_legacy_micro_step_pickle(Path(raw_root) / "second.pkl")
        payload = first.read_bytes()
        replay = second.read_bytes()

    assert payload == replay
    assert b"src.training.micro_steps" in payload
    assert b"src.training.supervised_trainer" not in payload
    assert payload != LEGACY_MICRO_STEP_PICKLE.read_bytes()
    assert restricted_load(payload)[0] == characterized_micro_step()
    assert pickle.loads(payload)[0] == characterized_micro_step()


def test_entrypoint_result_mapping_matches_frozen_summary(capsys: Any) -> None:
    observed = exercise_characterized_entrypoint(capsys)

    assert observed == load_fixture("entrypoint_result.json")


def test_prepare_training_pack_caches_remains_importable_from_the_facade() -> None:
    assert pipeline.prepare_training_pack_caches.__module__ == "src.training.pipeline"
    assert callable(pipeline.prepare_training_pack_caches)


# ---------------------------------------------------------------------------
# Fixture freezing
# ---------------------------------------------------------------------------


def freeze_fixtures() -> list[str]:
    """Derive and write every fixture from untouched production behavior."""

    written: list[str] = []
    FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)

    build_legacy_micro_step_pickle(LEGACY_MICRO_STEP_PICKLE)
    written.append(str(LEGACY_MICRO_STEP_PICKLE.relative_to(REPO_ROOT)))
    written.append(
        str(
            _write_json_fixture(
                "legacy_micro_step.json", exercise_characterized_legacy_micro_step()
            ).relative_to(REPO_ROOT)
        )
    )

    with tempfile.TemporaryDirectory(prefix="coordexp-freeze-writer-") as raw:
        tree = exercise_characterized_run_writer(Path(raw))
    with tempfile.TemporaryDirectory(prefix="coordexp-freeze-writer-") as raw:
        replay = exercise_characterized_run_writer(Path(raw))
    if tree != replay:
        raise AssertionError("RunWriter characterization is not byte-reproducible")
    written.append(
        str(_write_binary_fixture_tree("run_writer", tree).relative_to(REPO_ROOT))
    )

    with tempfile.TemporaryDirectory(prefix="coordexp-freeze-rows-") as raw:
        rows = exercise_characterized_completed_step_rows(Path(raw))
    written.append(
        str(
            _write_json_fixture("completed_step_rows.json", rows).relative_to(REPO_ROOT)
        )
    )

    monkeypatch = pytest.MonkeyPatch()
    try:
        with tempfile.TemporaryDirectory(prefix="coordexp-freeze-prep-") as raw:
            preparation = exercise_characterized_cache_preparation(
                Path(raw), monkeypatch
            )
    finally:
        monkeypatch.undo()
    written.append(
        str(
            _write_json_fixture("cache_preparation.json", preparation).relative_to(
                REPO_ROOT
            )
        )
    )

    written.append(
        str(
            _write_json_fixture(
                "causal_logits.json", exercise_characterized_causal_logits()
            ).relative_to(REPO_ROOT)
        )
    )

    with tempfile.TemporaryDirectory(prefix="coordexp-freeze-identity-") as raw:
        identity = exercise_characterized_identity(Path(raw))
    written.append(
        str(
            _write_json_fixture("identity_cases.json", identity).relative_to(REPO_ROOT)
        )
    )

    entry_capture = _StdoutCapture()
    with entry_capture:
        entrypoint = exercise_characterized_entrypoint(entry_capture)
    written.append(
        str(
            _write_json_fixture("entrypoint_result.json", entrypoint).relative_to(
                REPO_ROOT
            )
        )
    )

    characterized = exercise_characterized_pipeline()
    replayed = exercise_characterized_pipeline()
    if characterized.phase_order != replayed.phase_order:
        raise AssertionError("two-rank phase order is not reproducible")
    if characterized.collective_order != replayed.collective_order:
        raise AssertionError("two-rank collective order is not reproducible")
    if characterized.result != replayed.result:
        raise AssertionError("facade result mapping is not reproducible")
    for name, payload in (
        ("pipeline_result.json", characterized.result),
        ("phase_order.json", characterized.phase_order),
        ("collective_order.json", characterized.collective_order),
        ("cache_admission.json", characterized.cache_admission),
    ):
        written.append(str(_write_json_fixture(name, payload).relative_to(REPO_ROOT)))
    return written


class _StdoutCapture:
    """A minimal ``capsys``-shaped stdout capture usable outside pytest."""

    def __init__(self) -> None:
        self._buffer: Any = None
        self._redirect: Any = None

    def __enter__(self) -> "_StdoutCapture":
        import contextlib
        import io

        self._buffer = io.StringIO()
        self._redirect = contextlib.redirect_stdout(self._buffer)
        self._redirect.__enter__()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._redirect.__exit__(*exc)

    def readouterr(self) -> SimpleNamespace:
        return SimpleNamespace(out=self._buffer.getvalue(), err="")


if __name__ == "__main__":  # pragma: no cover - fixture derivation entrypoint
    for entry in freeze_fixtures():
        print(entry)
