from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import gc
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import re
import time
from typing import Any
import weakref

import pytest

from src.config.loader import load_train_config
from src.config.models import RuntimeBatchResolution
from src.training import pack_cache
from src.training.pack_cache import (
    DEFAULT_PACK_CACHE_MATERIALIZATION_WORKERS,
    PackingCacheInvalidError,
    PACKING_CACHE_MATERIALIZATION_STRATEGY,
    build_packing_cache_materialization,
    build_packing_cache_determinants,
    build_packing_cache_fingerprint,
    load_all_micro_steps_from_cache,
    load_cache_manifest,
    load_rank_micro_steps_from_cache,
    write_micro_step_cache as _write_micro_step_cache,
)
from src.training.schedule import ResolvedStepSchedule
from src.training.supervised_trainer import SupervisedMicroStep


FIXTURE_CONFIG = Path("tests/fixtures/smoke/qwen3_vl_single_image_pack/config.yaml")
UNIT_DETERMINANTS = {"purpose": "unit-test"}
UNIT_FINGERPRINT = hashlib.sha256(
    json.dumps(
        UNIT_DETERMINANTS,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
).hexdigest()
DISABLED_AUGMENTATION = {
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


def write_micro_step_cache(*args: Any, **kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("augmentation", DISABLED_AUGMENTATION)
    kwargs.setdefault("materialization", build_packing_cache_materialization())
    return _write_micro_step_cache(*args, **kwargs)


def build_repeating_micro_step_stream(
    base_micro_steps: Sequence[SupervisedMicroStep],
    schedule: ResolvedStepSchedule,
    *,
    rank: int = 0,
    world_size: int = 1,
) -> Iterator[SupervisedMicroStep]:
    """Reference oracle: the rank-local training presentation order a live
    pack stream would produce, used to prove rank-selective cache loading
    (`load_rank_micro_steps_from_cache`) matches it exactly."""

    if not base_micro_steps:
        raise ValueError("base_micro_steps must contain at least one micro-step")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank must be inside world_size")
    if schedule.runtime_batch.world_size != world_size:
        raise ValueError("stream world_size must match schedule runtime_batch")
    total_rank_local_micro_steps = (
        schedule.resolved_max_steps
        * schedule.runtime_batch.resolved_grad_accum_steps
    )

    def iter_repeated() -> Iterator[SupervisedMicroStep]:
        for index in range(total_rank_local_micro_steps):
            planned_step_index = (
                index // schedule.runtime_batch.resolved_grad_accum_steps
            )
            local_accum_index = (
                index % schedule.runtime_batch.resolved_grad_accum_steps
            )
            global_micro_step_index = (
                planned_step_index * schedule.runtime_batch.effective_batch_size
                + local_accum_index * world_size
                + rank
            )
            yield base_micro_steps[global_micro_step_index % len(base_micro_steps)]

    return iter_repeated()


def _overlap_writer_process(
    cache_dir_text: str,
    pause_phase: str | None,
    paused: Any,
    release: Any,
    started: Any,
    results: Any,
) -> None:
    cache_dir = Path(cache_dir_text)
    started.set()
    if pause_phase == "stage":
        real_dump = pack_cache.pickle.dump

        def paused_dump(*args: Any, **kwargs: Any) -> Any:
            paused.set()
            if not release.wait(timeout=30):
                raise RuntimeError("stage pause timed out")
            return real_dump(*args, **kwargs)

        pack_cache.pickle.dump = paused_dump
    elif pause_phase == "backup":
        real_replace = pack_cache.os.replace

        def paused_replace(source: Any, destination: Any) -> None:
            real_replace(source, destination)
            if Path(source) == cache_dir and ".backup-" in Path(destination).name:
                paused.set()
                if not release.wait(timeout=30):
                    raise RuntimeError("backup pause timed out")

        pack_cache.os.replace = paused_replace
    try:
        manifest = write_micro_step_cache(
            cache_dir,
            tuple(_micro_step(index) for index in range(4)),
            fingerprint=UNIT_FINGERPRINT,
            determinants=UNIT_DETERMINANTS,
            chunk_size=2,
        )
        loaded = load_all_micro_steps_from_cache(
            cache_dir, expected_fingerprint=UNIT_FINGERPRINT
        )
        results.put(
            (
                "ok",
                manifest["fingerprint"],
                [step.metadata["pack_id"] for step in loaded],
            )
        )
    except Exception as exc:
        results.put(("error", type(exc).__name__, str(exc)))


def test_packing_cache_fingerprint_tracks_data_template_and_encoding_only(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "train.coord.jsonl"
    dataset.write_text('{"example_id":"ex-0"}\n', encoding="utf-8")
    config = load_train_config(FIXTURE_CONFIG).config
    config = config.model_copy(
        update={
            "data": config.data.model_copy(
                update={
                    "train": config.data.train.model_copy(
                        update={"path": str(dataset)}
                    )
                }
            )
        }
    )
    components = FakeComponents()

    baseline = build_packing_cache_fingerprint(
        config,
        components,
        dataset=config.data.train,
        split="train",
    )
    changed_epochs = build_packing_cache_fingerprint(
        config.model_copy(
            update={"training": config.training.model_copy(update={"epochs": 7})}
        ),
        components,
        dataset=config.data.train,
        split="train",
    )
    changed_prompt = build_packing_cache_fingerprint(
        config.model_copy(
            update={
                "template": config.template.model_copy(
                    update={
                        "prompt": config.template.prompt.model_copy(
                            update={"user": "Changed prompt."}
                        )
                    }
                )
            }
        ),
        components,
        dataset=config.data.train,
        split="train",
    )

    assert changed_epochs == baseline
    assert changed_prompt != baseline


def test_packing_cache_fingerprint_is_timestamp_independent_and_tracks_content(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "train.coord.jsonl"
    dataset.write_text('{"example_id":"ex-a"}\n', encoding="utf-8")
    stat = dataset.stat()
    config = load_train_config(FIXTURE_CONFIG).config
    config = config.model_copy(
        update={
            "data": config.data.model_copy(
                update={
                    "train": config.data.train.model_copy(
                        update={"path": str(dataset)}
                    )
                }
            )
        }
    )
    components = FakeComponents()

    baseline = build_packing_cache_fingerprint(
        config,
        components,
        dataset=config.data.train,
        split="train",
    )
    baseline_determinants = build_packing_cache_determinants(
        config,
        components,
        dataset=config.data.train,
        split="train",
    )
    assert "mtime_ns" not in baseline_determinants["dataset"]

    # Touch: only the modification time changes (content and size identical).
    # The semantic fingerprint MUST remain identical (dataset timestamps are
    # not semantic identity).
    future_ns = int((time.time() + 5) * 1_000_000_000)
    os.utime(dataset, ns=(future_ns, future_ns))
    assert dataset.stat().st_mtime_ns != stat.st_mtime_ns
    assert dataset.stat().st_size == stat.st_size

    touched = build_packing_cache_fingerprint(
        config,
        components,
        dataset=config.data.train,
        split="train",
    )
    assert touched == baseline

    # Byte content change MUST change the fingerprint, independent of the
    # timestamp (forced back to the original value here to isolate the
    # effect to content alone).
    dataset.write_text('{"example_id":"ex-b"}\n', encoding="utf-8")
    os.utime(dataset, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    changed_content = build_packing_cache_fingerprint(
        config,
        components,
        dataset=config.data.train,
        split="train",
    )

    assert dataset.stat().st_size == stat.st_size
    assert dataset.stat().st_mtime_ns == stat.st_mtime_ns
    assert changed_content != baseline


def test_packing_cache_fingerprint_tracks_image_pad_token_id(tmp_path: Path) -> None:
    dataset = tmp_path / "train.coord.jsonl"
    dataset.write_text('{"example_id":"ex-0"}\n', encoding="utf-8")
    config = load_train_config(FIXTURE_CONFIG).config
    config = config.model_copy(
        update={
            "data": config.data.model_copy(
                update={
                    "train": config.data.train.model_copy(
                        update={"path": str(dataset)}
                    )
                }
            )
        }
    )

    baseline = build_packing_cache_fingerprint(
        config,
        FakeComponents(image_pad_token_id=151655),
        dataset=config.data.train,
        split="train",
    )
    changed_image_pad = build_packing_cache_fingerprint(
        config,
        FakeComponents(image_pad_token_id=151700),
        dataset=config.data.train,
        split="train",
    )

    assert changed_image_pad != baseline


def test_packing_cache_determinants_include_code_identity(tmp_path: Path) -> None:
    dataset = tmp_path / "train.coord.jsonl"
    dataset.write_text('{"example_id":"ex-0"}\n', encoding="utf-8")
    config = load_train_config(FIXTURE_CONFIG).config
    config = config.model_copy(
        update={
            "data": config.data.model_copy(
                update={
                    "train": config.data.train.model_copy(
                        update={"path": str(dataset)}
                    )
                }
            )
        }
    )

    determinants = build_packing_cache_determinants(
        config,
        FakeComponents(),
        dataset=config.data.train,
        split="train",
    )

    code_identity = determinants["code_identity"]
    assert set(code_identity) == {
        "augmentation_factory",
        "augmentation_geometry",
        "augmentation_processor",
        "coordinate_targets",
        "template_spans",
        "template_renderer",
        "qwen_encoding",
        "qwen_images",
        "qwen_positions",
        "qwen_fa2",
        "qwen_forward",
        "packing_planner",
        "packing_supervision",
        "supervision_tokens",
    }
    for payload in code_identity.values():
        assert payload["sha256"]
        assert len(payload["sha256"]) == 64
        assert payload["path"].startswith("src/")


def test_packing_cache_fingerprint_tracks_augmentation_config_and_seed(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "train.coord.jsonl"
    dataset.write_text('{"example_id":"ex-0"}\n', encoding="utf-8")
    config = load_train_config(FIXTURE_CONFIG).config
    config = config.model_copy(
        update={
            "data": config.data.model_copy(
                update={
                    "train": config.data.train.model_copy(
                        update={"path": str(dataset)}
                    )
                }
            )
        }
    )
    components = FakeComponents()

    baseline = build_packing_cache_fingerprint(
        config,
        components,
        dataset=config.data.train,
        split="train",
    )
    enabled = build_packing_cache_fingerprint(
        _config_with_geometry_flips(config, horizontal_prob=1.0),
        components,
        dataset=config.data.train,
        split="train",
    )
    changed_probability = build_packing_cache_fingerprint(
        _config_with_geometry_flips(config, horizontal_prob=0.5),
        components,
        dataset=config.data.train,
        split="train",
    )
    changed_seed = build_packing_cache_fingerprint(
        _config_with_geometry_flips(
            config.model_copy(
                update={"runtime": config.runtime.model_copy(update={"seed": 99})}
            ),
            horizontal_prob=1.0,
        ),
        components,
        dataset=config.data.train,
        split="train",
    )
    determinants = build_packing_cache_determinants(
        _config_with_geometry_flips(config, horizontal_prob=1.0),
        components,
        dataset=config.data.train,
        split="train",
    )

    assert enabled != baseline
    assert changed_probability != enabled
    assert changed_seed != enabled
    assert determinants["augmentation"]["policy"] == "geometry_flips"
    assert determinants["augmentation"]["train"]["geometry_flips"]["enabled"] is True


def test_packing_cache_fingerprint_ignores_materialization_worker_count(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "train.coord.jsonl"
    dataset.write_text('{"example_id":"ex-0"}\n', encoding="utf-8")
    config = load_train_config(FIXTURE_CONFIG).config
    config = config.model_copy(
        update={
            "data": config.data.model_copy(
                update={
                    "train": config.data.train.model_copy(
                        update={"path": str(dataset)}
                    )
                }
            )
        }
    )
    components = FakeComponents()
    fingerprint = build_packing_cache_fingerprint(
        config,
        components,
        dataset=config.data.train,
        split="train",
    )
    determinants = build_packing_cache_determinants(
        config,
        components,
        dataset=config.data.train,
        split="train",
    )

    default_cache = tmp_path / "default-cache"
    override_cache = tmp_path / "override-cache"
    default_manifest = write_micro_step_cache(
        default_cache,
        (_micro_step(0),),
        fingerprint=fingerprint,
        determinants=determinants,
    )
    override_manifest = write_micro_step_cache(
        override_cache,
        (_micro_step(0),),
        fingerprint=fingerprint,
        determinants=determinants,
        materialization={
            "strategy": PACKING_CACHE_MATERIALIZATION_STRATEGY,
            "workers": 3,
        },
    )

    assert default_manifest["fingerprint"] == fingerprint
    assert override_manifest["fingerprint"] == fingerprint
    assert default_manifest["materialization"]["workers"] != (
        override_manifest["materialization"]["workers"]
    )


def test_micro_step_cache_manifest_records_default_materialization_workers(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"

    manifest = write_micro_step_cache(
        cache_dir,
        (_micro_step(0),),
        fingerprint=UNIT_FINGERPRINT,
        determinants={"purpose": "unit-test"},
    )

    assert manifest["materialization"] == {
        "strategy": PACKING_CACHE_MATERIALIZATION_STRATEGY,
        "workers": DEFAULT_PACK_CACHE_MATERIALIZATION_WORKERS,
    }
    assert load_cache_manifest(
        cache_dir, expected_fingerprint=UNIT_FINGERPRINT, level="payloads"
    )["materialization"] == manifest["materialization"]


def test_micro_step_cache_manifest_records_explicit_materialization_override(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"

    manifest = write_micro_step_cache(
        cache_dir,
        (_micro_step(0),),
        fingerprint=UNIT_FINGERPRINT,
        determinants={"purpose": "unit-test"},
        materialization={
            "strategy": PACKING_CACHE_MATERIALIZATION_STRATEGY,
            "workers": 4,
        },
    )

    assert manifest["materialization"] == {
        "strategy": PACKING_CACHE_MATERIALIZATION_STRATEGY,
        "workers": 4,
    }


def test_micro_step_cache_manifest_requires_current_provenance(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    write_micro_step_cache(
        cache_dir,
        (_micro_step(0),),
        fingerprint=UNIT_FINGERPRINT,
        determinants={"purpose": "unit-test"},
    )
    manifest_path = cache_dir / "manifest.json"
    invalid_manifest = json.loads(manifest_path.read_text())
    invalid_manifest.pop("materialization")
    manifest_path.write_text(json.dumps(invalid_manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="materialization"):
        load_cache_manifest(cache_dir, expected_fingerprint=UNIT_FINGERPRINT, level="payloads")


@pytest.mark.parametrize("augmentation", [None, {}])
def test_cache_writer_rejects_missing_or_empty_augmentation_receipt(
    tmp_path: Path, augmentation: Any
) -> None:
    with pytest.raises(ValueError, match="augmentation.*non-empty"):
        _write_micro_step_cache(
            tmp_path / "cache",
            (_micro_step(0),),
            fingerprint=UNIT_FINGERPRINT,
            determinants=UNIT_DETERMINANTS,
            materialization=build_packing_cache_materialization(),
            augmentation=augmentation,
        )


def test_cache_writer_requires_explicit_materialization(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="materialization"):
        _write_micro_step_cache(
            tmp_path / "cache",
            (_micro_step(0),),
            fingerprint=UNIT_FINGERPRINT,
            determinants=UNIT_DETERMINANTS,
            augmentation=DISABLED_AUGMENTATION,
        )


@pytest.mark.parametrize("mutation", ["omit", "empty"])
def test_cache_reader_rejects_missing_or_empty_augmentation_receipt(
    tmp_path: Path, mutation: str
) -> None:
    cache_dir = tmp_path / "cache"
    write_micro_step_cache(
        cache_dir,
        (_micro_step(0),),
        fingerprint=UNIT_FINGERPRINT,
        determinants=UNIT_DETERMINANTS,
    )
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if mutation == "omit":
        manifest.pop("augmentation")
    else:
        manifest["augmentation"] = {}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(PackingCacheInvalidError, match="augmentation.*non-empty"):
        load_cache_manifest(cache_dir, expected_fingerprint=UNIT_FINGERPRINT, level="payloads")


def test_micro_step_cache_loads_exact_rank_local_training_order(tmp_path: Path) -> None:
    micro_steps = tuple(_micro_step(index) for index in range(10))
    cache_dir = tmp_path / "cache"
    manifest = write_micro_step_cache(
        cache_dir,
        micro_steps,
        fingerprint=UNIT_FINGERPRINT,
        determinants={"purpose": "unit-test"},
        chunk_size=3,
    )
    schedule = _schedule(
        resolved_max_steps=2,
        grad_accum_steps=2,
        world_size=2,
        effective_batch_size=4,
    )

    rank0 = load_rank_micro_steps_from_cache(
        cache_dir,
        expected_fingerprint=UNIT_FINGERPRINT,
        schedule=schedule,
        rank=0,
        world_size=2,
    )
    rank1 = load_rank_micro_steps_from_cache(
        cache_dir,
        expected_fingerprint=UNIT_FINGERPRINT,
        schedule=schedule,
        rank=1,
        world_size=2,
    )

    assert manifest["micro_step_count"] == 10
    assert [step.metadata["pack_id"] for step in rank0] == [0, 2, 4, 6]
    assert [step.metadata["pack_id"] for step in rank1] == [1, 3, 5, 7]


def test_micro_step_cache_wraps_tail_presentations_by_pack_count(tmp_path: Path) -> None:
    micro_steps = tuple(_micro_step(index) for index in range(3))
    cache_dir = tmp_path / "cache"
    write_micro_step_cache(
        cache_dir,
        micro_steps,
        fingerprint=UNIT_FINGERPRINT,
        determinants={"purpose": "unit-test"},
        chunk_size=2,
    )
    schedule = _schedule(
        resolved_max_steps=3,
        grad_accum_steps=1,
        world_size=2,
        effective_batch_size=2,
    )

    rank1 = load_rank_micro_steps_from_cache(
        cache_dir,
        expected_fingerprint=UNIT_FINGERPRINT,
        schedule=schedule,
        rank=1,
        world_size=2,
    )

    assert [step.metadata["pack_id"] for step in rank1] == [1, 0, 2]


def test_micro_step_cache_loads_complete_sequence_for_eval(tmp_path: Path) -> None:
    micro_steps = tuple(_micro_step(index) for index in range(7))
    cache_dir = tmp_path / "cache"
    write_micro_step_cache(
        cache_dir,
        micro_steps,
        fingerprint=hashlib.sha256(
            json.dumps(
                {"purpose": "unit-test", "split": "eval.forward"},
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
        determinants={"purpose": "unit-test", "split": "eval.forward"},
        chunk_size=3,
    )

    loaded = load_all_micro_steps_from_cache(
        cache_dir,
        expected_fingerprint=hashlib.sha256(
            json.dumps(
                {"purpose": "unit-test", "split": "eval.forward"},
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
    )

    assert [step.metadata["pack_id"] for step in loaded] == list(range(7))


def test_cached_rank_local_steps_must_not_be_sharded_again(tmp_path: Path) -> None:
    micro_steps = tuple(_micro_step(index) for index in range(2048))
    cache_dir = tmp_path / "cache"
    write_micro_step_cache(
        cache_dir,
        micro_steps,
        fingerprint=UNIT_FINGERPRINT,
        determinants={"purpose": "unit-test"},
        chunk_size=256,
    )
    schedule = _schedule(
        resolved_max_steps=2,
        grad_accum_steps=16,
        world_size=8,
        effective_batch_size=128,
    )

    rank0 = load_rank_micro_steps_from_cache(
        cache_dir,
        expected_fingerprint=UNIT_FINGERPRINT,
        schedule=schedule,
        rank=0,
        world_size=8,
    )

    assert [step.metadata["pack_id"] for step in rank0[:16]] == list(range(0, 128, 8))
    double_sharded = build_repeating_micro_step_stream(
        rank0,
        schedule,
        rank=0,
        world_size=8,
    )
    assert [next(double_sharded).metadata["pack_id"] for _ in range(16)] != list(
        range(0, 128, 8)
    )


def test_micro_step_cache_rejects_corrupt_required_chunk(tmp_path: Path) -> None:
    micro_steps = tuple(_micro_step(index) for index in range(4))
    cache_dir = tmp_path / "cache"
    write_micro_step_cache(
        cache_dir,
        micro_steps,
        fingerprint=UNIT_FINGERPRINT,
        determinants={"purpose": "unit-test"},
        chunk_size=2,
    )
    corrupt_chunk = cache_dir / "chunks" / "chunk-00000.pkl"
    with corrupt_chunk.open("wb") as handle:
        pickle.dump((_micro_step(99),), handle)
    schedule = _schedule(
        resolved_max_steps=1,
        grad_accum_steps=1,
        world_size=1,
        effective_batch_size=1,
    )

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_rank_micro_steps_from_cache(
            cache_dir,
            expected_fingerprint=UNIT_FINGERPRINT,
            schedule=schedule,
            rank=0,
            world_size=1,
        )


def test_micro_step_cache_manifest_rejects_chunk_gaps(tmp_path: Path) -> None:
    micro_steps = tuple(_micro_step(index) for index in range(4))
    cache_dir = tmp_path / "cache"
    write_micro_step_cache(
        cache_dir,
        micro_steps,
        fingerprint=UNIT_FINGERPRINT,
        determinants={"purpose": "unit-test"},
        chunk_size=2,
    )
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["chunks"][1]["start"] = 3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    schedule = _schedule(
        resolved_max_steps=1,
        grad_accum_steps=1,
        world_size=1,
        effective_batch_size=1,
    )

    with pytest.raises(ValueError, match="contiguous"):
        load_rank_micro_steps_from_cache(
            cache_dir,
            expected_fingerprint=UNIT_FINGERPRINT,
            schedule=schedule,
            rank=0,
            world_size=1,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda manifest: manifest.__setitem__("version", "old-version"), "version"),
        (lambda manifest: manifest.__setitem__("status", "writing"), "complete"),
        (lambda manifest: manifest.__setitem__("fingerprint", "other"), "fingerprint"),
        (lambda manifest: manifest.__setitem__("determinants", None), "determinants"),
        (lambda manifest: manifest.__setitem__("augmentation", None), "augmentation"),
        (lambda manifest: manifest.__setitem__("micro_step_count", 5), "cover"),
        (lambda manifest: manifest["chunks"][0].__setitem__("count", 1), "count"),
        (lambda manifest: manifest["chunks"][0].__setitem__("sha256", "z" * 64), "sha256"),
        (lambda manifest: manifest["chunks"][0].__setitem__("path", "../escape.pkl"), "path"),
    ],
)
def test_cache_manifest_rejects_invalid_current_contract(
    tmp_path: Path, mutation: Any, match: str
) -> None:
    cache_dir = tmp_path / "cache"
    write_micro_step_cache(
        cache_dir,
        tuple(_micro_step(index) for index in range(4)),
        fingerprint=UNIT_FINGERPRINT,
        determinants={"purpose": "unit-test"},
        chunk_size=2,
    )
    path = cache_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    mutation(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_cache_manifest(cache_dir, expected_fingerprint=UNIT_FINGERPRINT, level="payloads")


def test_cache_manifest_rejects_corrupt_json(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    write_micro_step_cache(
        cache_dir,
        (_micro_step(0),),
        fingerprint=UNIT_FINGERPRINT,
        determinants={"purpose": "unit-test"},
    )
    (cache_dir / "manifest.json").write_text("{", encoding="utf-8")

    with pytest.raises(PackingCacheInvalidError):
        load_cache_manifest(cache_dir, expected_fingerprint=UNIT_FINGERPRINT, level="payloads")


def test_cache_reader_rejects_corrupt_pickle_with_matching_declared_hash(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    write_micro_step_cache(
        cache_dir,
        (_micro_step(0),),
        fingerprint=UNIT_FINGERPRINT,
        determinants={"purpose": "unit-test"},
    )
    chunk_path = cache_dir / "chunks" / "chunk-00000.pkl"
    chunk_path.write_bytes(b"not-a-pickle")
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["chunks"][0]["sha256"] = _sha256(chunk_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="unreadable"):
        load_all_micro_steps_from_cache(
            cache_dir, expected_fingerprint=UNIT_FINGERPRINT
        )


@pytest.mark.parametrize("reader", ["manifest", "rank", "all"])
@pytest.mark.parametrize("failure", [ModuleNotFoundError, AttributeError, RuntimeError])
def test_cache_readers_normalize_state_restoration_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reader: str,
    failure: type[Exception],
) -> None:
    cache_dir = tmp_path / "cache"
    write_micro_step_cache(
        cache_dir,
        (_micro_step(0),),
        fingerprint=UNIT_FINGERPRINT,
        determinants=UNIT_DETERMINANTS,
    )
    schedule = _schedule(
        resolved_max_steps=1,
        grad_accum_steps=1,
        world_size=1,
        effective_batch_size=1,
    )

    def fail_restore(_self: Any) -> Any:
        raise failure("injected restoration failure")

    monkeypatch.setattr(pack_cache._RestrictedCacheUnpickler, "load", fail_restore)
    with pytest.raises(PackingCacheInvalidError, match="unreadable"):
        if reader == "manifest":
            load_cache_manifest(
                cache_dir,
                expected_fingerprint=UNIT_FINGERPRINT,
                level="payloads",
            )
        elif reader == "rank":
            load_rank_micro_steps_from_cache(
                cache_dir,
                expected_fingerprint=UNIT_FINGERPRINT,
                schedule=schedule,
                rank=0,
                world_size=1,
            )
        else:
            load_all_micro_steps_from_cache(
                cache_dir, expected_fingerprint=UNIT_FINGERPRINT
            )


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_cache_reader_does_not_normalize_process_interrupts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt: type[BaseException],
) -> None:
    cache_dir = tmp_path / "cache"
    write_micro_step_cache(
        cache_dir,
        (_micro_step(0),),
        fingerprint=UNIT_FINGERPRINT,
        determinants=UNIT_DETERMINANTS,
    )

    def interrupt_restore(_self: Any) -> Any:
        raise interrupt()

    monkeypatch.setattr(
        pack_cache._RestrictedCacheUnpickler, "load", interrupt_restore
    )
    with pytest.raises(interrupt):
        load_all_micro_steps_from_cache(
            cache_dir, expected_fingerprint=UNIT_FINGERPRINT
        )


def test_restricted_cache_unpickler_rejects_reduce_before_side_effect(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    write_micro_step_cache(
        cache_dir,
        (_micro_step(0),),
        fingerprint=UNIT_FINGERPRINT,
        determinants=UNIT_DETERMINANTS,
    )
    side_effect = tmp_path / "must-not-exist"

    class MaliciousPayload:
        def __reduce__(self) -> Any:
            return os.system, (f"touch {side_effect}",)

    chunk_path = cache_dir / "chunks" / "chunk-00000.pkl"
    with chunk_path.open("wb") as handle:
        pickle.dump((MaliciousPayload(),), handle)
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["chunks"][0]["sha256"] = _sha256(chunk_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(PackingCacheInvalidError, match="unreadable"):
        load_all_micro_steps_from_cache(
            cache_dir, expected_fingerprint=UNIT_FINGERPRINT
        )
    assert not side_effect.exists()


def test_cache_manifest_rejects_missing_chunk(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    write_micro_step_cache(
        cache_dir,
        (_micro_step(0),),
        fingerprint=UNIT_FINGERPRINT,
        determinants={"purpose": "unit-test"},
    )
    (cache_dir / "chunks" / "chunk-00000.pkl").unlink()

    with pytest.raises(ValueError, match="missing"):
        load_cache_manifest(cache_dir, expected_fingerprint=UNIT_FINGERPRINT, level="payloads")


@pytest.mark.parametrize("reader", ["manifest", "rank", "all"])
def test_cache_readers_reject_mutated_determinants_with_unchanged_fingerprint(
    tmp_path: Path, reader: str
) -> None:
    cache_dir = tmp_path / "cache"
    write_micro_step_cache(
        cache_dir,
        (_micro_step(0),),
        fingerprint=UNIT_FINGERPRINT,
        determinants=UNIT_DETERMINANTS,
    )
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["determinants"]["purpose"] = "tampered"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    schedule = _schedule(
        resolved_max_steps=1,
        grad_accum_steps=1,
        world_size=1,
        effective_batch_size=1,
    )

    with pytest.raises(ValueError, match="canonical determinants"):
        if reader == "manifest":
            load_cache_manifest(
                cache_dir,
                expected_fingerprint=UNIT_FINGERPRINT,
                level="payloads",
            )
        elif reader == "rank":
            load_rank_micro_steps_from_cache(
                cache_dir,
                expected_fingerprint=UNIT_FINGERPRINT,
                schedule=schedule,
                rank=0,
                world_size=1,
            )
        else:
            load_all_micro_steps_from_cache(
                cache_dir, expected_fingerprint=UNIT_FINGERPRINT
            )


@pytest.mark.parametrize("reader", ["manifest", "rank", "all"])
def test_cache_readers_require_explicit_materialization_strategy(
    tmp_path: Path, reader: str
) -> None:
    cache_dir = tmp_path / "cache"
    write_micro_step_cache(
        cache_dir,
        (_micro_step(0),),
        fingerprint=UNIT_FINGERPRINT,
        determinants=UNIT_DETERMINANTS,
    )
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["materialization"].pop("strategy")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    schedule = _schedule(
        resolved_max_steps=1,
        grad_accum_steps=1,
        world_size=1,
        effective_batch_size=1,
    )

    with pytest.raises(PackingCacheInvalidError, match="strategy"):
        if reader == "manifest":
            load_cache_manifest(
                cache_dir,
                expected_fingerprint=UNIT_FINGERPRINT,
                level="payloads",
            )
        elif reader == "rank":
            load_rank_micro_steps_from_cache(
                cache_dir,
                expected_fingerprint=UNIT_FINGERPRINT,
                schedule=schedule,
                rank=0,
                world_size=1,
            )
        else:
            load_all_micro_steps_from_cache(
                cache_dir, expected_fingerprint=UNIT_FINGERPRINT
            )


def test_rank_reader_skips_corrupt_chunk_outside_required_set(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    write_micro_step_cache(
        cache_dir,
        tuple(_micro_step(index) for index in range(4)),
        fingerprint=UNIT_FINGERPRINT,
        determinants={"purpose": "unit-test"},
        chunk_size=2,
    )
    # chunk-00001.pkl covers indices [2, 4); the rank below requires only
    # index 0, so this corrupt chunk must never be read or decoded.
    (cache_dir / "chunks" / "chunk-00001.pkl").write_bytes(b"corrupt")
    schedule = _schedule(
        resolved_max_steps=1,
        grad_accum_steps=1,
        world_size=1,
        effective_batch_size=1,
    )

    selected = load_rank_micro_steps_from_cache(
        cache_dir,
        expected_fingerprint=UNIT_FINGERPRINT,
        schedule=schedule,
        rank=0,
        world_size=1,
    )

    assert [step.metadata["pack_id"] for step in selected] == [0]


# A corrupted REQUIRED chunk still failing closed is covered by
# test_micro_step_cache_rejects_corrupt_required_chunk below.


def test_rank_reader_releases_validated_unselected_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache"
    write_micro_step_cache(
        cache_dir,
        tuple(_micro_step(index) for index in range(4)),
        fingerprint=UNIT_FINGERPRINT,
        determinants={"purpose": "unit-test"},
        chunk_size=2,
    )
    schedule = _schedule(
        resolved_max_steps=1,
        grad_accum_steps=1,
        world_size=1,
        effective_batch_size=1,
    )
    real_load = pack_cache._RestrictedCacheUnpickler.load
    loaded_refs: dict[int, weakref.ReferenceType[SupervisedMicroStep]] = {}

    def tracking_load(unpickler: Any) -> Any:
        payload = real_load(unpickler)
        for step in payload:
            loaded_refs[step.metadata["pack_id"]] = weakref.ref(step)
        return payload

    monkeypatch.setattr(pack_cache._RestrictedCacheUnpickler, "load", tracking_load)
    selected = load_rank_micro_steps_from_cache(
        cache_dir,
        expected_fingerprint=UNIT_FINGERPRINT,
        schedule=schedule,
        rank=0,
        world_size=1,
    )
    gc.collect()

    assert selected[0].metadata["pack_id"] == 0
    assert loaded_refs[0]() is selected[0]
    # Index 1 shares chunk-00000 with the required index 0, so it is still
    # decoded (chunk-level granularity) and released once unselected.
    assert loaded_refs[1]() is None
    # Indices 2 and 3 live in chunk-00001, which does not intersect the
    # rank's required set ({0}) and is never read or decoded at all.
    assert 2 not in loaded_refs
    assert 3 not in loaded_refs


@pytest.mark.parametrize(
    ("resolved_max_steps", "grad_accum_steps", "world_size"),
    [
        (1, 1, 1),
        (2, 3, 1),
        (3, 2, 4),
        (2, 5, 3),
    ],
)
def test_rank_selective_loading_matches_full_pass_and_repeating_stream_oracle(
    tmp_path: Path,
    resolved_max_steps: int,
    grad_accum_steps: int,
    world_size: int,
) -> None:
    total_micro_steps = 17
    micro_steps = tuple(_micro_step(index) for index in range(total_micro_steps))
    cache_dir = tmp_path / "cache"
    write_micro_step_cache(
        cache_dir,
        micro_steps,
        fingerprint=UNIT_FINGERPRINT,
        determinants={"purpose": "unit-test"},
        chunk_size=3,
    )
    schedule = _schedule(
        resolved_max_steps=resolved_max_steps,
        grad_accum_steps=grad_accum_steps,
        world_size=world_size,
        effective_batch_size=grad_accum_steps * world_size,
    )
    oracle_all = load_all_micro_steps_from_cache(
        cache_dir, expected_fingerprint=UNIT_FINGERPRINT
    )

    for rank in range(world_size):
        rank_selective = load_rank_micro_steps_from_cache(
            cache_dir,
            expected_fingerprint=UNIT_FINGERPRINT,
            schedule=schedule,
            rank=rank,
            world_size=world_size,
        )
        full_pass = load_rank_micro_steps_from_cache(
            cache_dir,
            expected_fingerprint=UNIT_FINGERPRINT,
            schedule=schedule,
            rank=rank,
            world_size=world_size,
            _force_full_chunk_pass=True,
        )
        oracle = tuple(
            build_repeating_micro_step_stream(
                oracle_all, schedule, rank=rank, world_size=world_size
            )
        )
        rank_selective_ids = [step.metadata["pack_id"] for step in rank_selective]
        full_pass_ids = [step.metadata["pack_id"] for step in full_pass]
        oracle_ids = [step.metadata["pack_id"] for step in oracle]

        # Rank-selective chunk skipping (the new default) MUST produce the
        # exact same sequence as a full validated digest-and-payload pass
        # (the internal force-full-pass control, reachable post-Wave-2 only
        # through this test/benchmark-only kwarg) and as the independent
        # repeating-stream oracle.
        assert rank_selective_ids == full_pass_ids
        assert rank_selective_ids == oracle_ids


@pytest.mark.parametrize("payload_kind", ["list", "wrong-length-tuple"])
def test_cache_reader_rejects_invalid_chunk_payload_shape_or_count(
    tmp_path: Path, payload_kind: str
) -> None:
    cache_dir = tmp_path / "cache"
    write_micro_step_cache(
        cache_dir,
        (_micro_step(0),),
        fingerprint=UNIT_FINGERPRINT,
        determinants={"purpose": "unit-test"},
    )
    chunk_path = cache_dir / "chunks" / "chunk-00000.pkl"
    payload = (
        [_micro_step(0)]
        if payload_kind == "list"
        else (_micro_step(0), _micro_step(1))
    )
    with chunk_path.open("wb") as handle:
        pickle.dump(payload, handle)
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["chunks"][0]["sha256"] = _sha256(chunk_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="tuple|payload length"):
        load_all_micro_steps_from_cache(
            cache_dir, expected_fingerprint=UNIT_FINGERPRINT
        )


def test_failed_cache_publication_preserves_previous_valid_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache"
    old_determinants = {"purpose": "old-unit-test"}
    new_determinants = {"purpose": "new-unit-test"}
    old_fingerprint = _fingerprint(old_determinants)
    new_fingerprint = _fingerprint(new_determinants)
    write_micro_step_cache(
        cache_dir,
        (_micro_step(0),),
        fingerprint=old_fingerprint,
        determinants=old_determinants,
    )
    real_replace = os.replace

    def fail_stage_publish(source: Any, destination: Any) -> None:
        if Path(destination) == cache_dir and re.search(r"\.stage-", Path(source).name):
            raise OSError("injected publication failure")
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_stage_publish)
    with pytest.raises(OSError, match="injected publication failure"):
        write_micro_step_cache(
            cache_dir,
            (_micro_step(1),),
            fingerprint=new_fingerprint,
            determinants=new_determinants,
        )

    loaded = load_all_micro_steps_from_cache(
        cache_dir, expected_fingerprint=old_fingerprint
    )
    assert [step.metadata["pack_id"] for step in loaded] == [0]
    assert not list(tmp_path.glob(".cache.stage-*"))
    assert not list(tmp_path.glob(".cache.backup-*"))


def test_cache_writer_removes_stale_stage_and_backup_siblings(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    stale_stage = tmp_path / ".cache.stage-crashed"
    stale_backup = tmp_path / ".cache.backup-crashed"
    stale_stage.mkdir()
    stale_backup.mkdir()
    (stale_stage / "manifest.json").write_text('{"status":"complete"}')
    (stale_backup / "manifest.json").write_text('{"status":"complete"}')

    write_micro_step_cache(
        cache_dir,
        (_micro_step(0),),
        fingerprint=UNIT_FINGERPRINT,
        determinants=UNIT_DETERMINANTS,
    )

    assert not stale_stage.exists()
    assert not stale_backup.exists()
    assert load_cache_manifest(
        cache_dir, expected_fingerprint=UNIT_FINGERPRINT, level="payloads"
    )["status"] == "complete"


@pytest.mark.parametrize("pause_phase", ["stage", "backup"])
def test_same_fingerprint_writers_serialize_across_processes(
    tmp_path: Path, pause_phase: str
) -> None:
    cache_dir = tmp_path / "cache"
    if pause_phase == "backup":
        write_micro_step_cache(
            cache_dir,
            (_micro_step(99),),
            fingerprint=UNIT_FINGERPRINT,
            determinants=UNIT_DETERMINANTS,
        )
        (cache_dir / "chunks" / "chunk-00000.pkl").write_bytes(b"invalid")

    context = multiprocessing.get_context("spawn")
    paused = context.Event()
    release = context.Event()
    started_a = context.Event()
    started_b = context.Event()
    unused_paused_b = context.Event()
    unused_release_b = context.Event()
    results = context.Queue()
    writer_a = context.Process(
        target=_overlap_writer_process,
        args=(str(cache_dir), pause_phase, paused, release, started_a, results),
    )
    writer_b = context.Process(
        target=_overlap_writer_process,
        args=(
            str(cache_dir),
            None,
            unused_paused_b,
            unused_release_b,
            started_b,
            results,
        ),
    )
    writer_a.start()
    assert started_a.wait(timeout=20)
    assert paused.wait(timeout=20)
    writer_b.start()
    assert started_b.wait(timeout=20)
    time.sleep(0.2)
    assert writer_b.is_alive(), "writer B must block on the cache-root lock"

    release.set()
    writer_a.join(timeout=30)
    writer_b.join(timeout=30)
    assert writer_a.exitcode == 0
    assert writer_b.exitcode == 0
    outcomes = [results.get(timeout=2), results.get(timeout=2)]
    assert outcomes == [
        ("ok", UNIT_FINGERPRINT, [0, 1, 2, 3]),
        ("ok", UNIT_FINGERPRINT, [0, 1, 2, 3]),
    ]
    assert not list(tmp_path.glob(".cache.stage-*"))
    assert not list(tmp_path.glob(".cache.backup-*"))
    assert (tmp_path / ".cache.lock").is_file()


def _micro_step(index: int) -> SupervisedMicroStep:
    return SupervisedMicroStep(
        pack=f"pack-{index}",
        encoded_examples=(f"example-{index}",),
        position_inputs=f"positions-{index}",
        token_sequence=f"tokens-{index}",
        vocab_groups=f"vocab-{index}",
        metadata={"pack_id": index},
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fingerprint(determinants: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            determinants,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _schedule(
    *,
    resolved_max_steps: int,
    grad_accum_steps: int,
    world_size: int,
    effective_batch_size: int,
) -> ResolvedStepSchedule:
    return ResolvedStepSchedule(
        resolved_max_steps=resolved_max_steps,
        packs_per_epoch=1,
        requested_pack_presentations=resolved_max_steps * effective_batch_size,
        actual_pack_presentations=resolved_max_steps * effective_batch_size,
        tail_fill_pack_count=0,
        runtime_batch=RuntimeBatchResolution(
            world_size=world_size,
            effective_batch_size=effective_batch_size,
            resolved_grad_accum_steps=grad_accum_steps,
        ),
        events={"checkpoint": (), "eval.forward": (), "final": ()},
    )


@dataclass(frozen=True)
class FakeIdentity:
    payload: dict[str, Any]

    def to_artifact_dict(self) -> dict[str, Any]:
        return dict(self.payload)


class FakeTokenizer:
    chat_template = "fake-chat-template"

    def __init__(self, *, image_pad_token_id: int) -> None:
        self.image_pad_token_id = image_pad_token_id

    def convert_tokens_to_ids(self, token: str) -> int | None:
        if token == "<|image_pad|>":
            return self.image_pad_token_id
        return None


class FakeProcessor:
    chat_template = "fake-chat-template"


class FakeComponents:
    processor_identity = FakeIdentity({"processor": "fake", "patch_size": 16})
    token_identity = FakeIdentity({"tokenizer": "fake", "vocab_size": 99})
    base_model_path = Path("/tmp/fake-model")
    package_versions = {"transformers": "unit-test"}

    def __init__(self, *, image_pad_token_id: int = 151655) -> None:
        self.tokenizer = FakeTokenizer(image_pad_token_id=image_pad_token_id)
        self.processor = FakeProcessor()


def _config_with_geometry_flips(
    config: Any,
    *,
    horizontal_prob: float = 0.0,
    vertical_prob: float = 0.0,
) -> Any:
    return config.model_copy(
        update={
            "data": config.data.model_copy(
                update={
                    "augmentation": config.data.augmentation.model_copy(
                        update={
                            "train": config.data.augmentation.train.model_copy(
                                update={
                                    "geometry_flips": (
                                        config.data.augmentation.train.geometry_flips.model_copy(
                                            update={
                                                "enabled": True,
                                                "horizontal_prob": horizontal_prob,
                                                "vertical_prob": vertical_prob,
                                            }
                                        )
                                    )
                                }
                            )
                        }
                    )
                }
            )
        }
    )


def test_manifest_admission_defers_payload_work_to_one_eager_rank_load(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    write_micro_step_cache(
        cache_dir,
        (_micro_step(0), _micro_step(1)),
        fingerprint=UNIT_FINGERPRINT,
        determinants=UNIT_DETERMINANTS,
        chunk_size=1,
    )
    corrupt_chunk = cache_dir / "chunks" / "chunk-00000.pkl"
    corrupt_chunk.write_bytes(b"corrupted-after-preparation")

    manifest = load_cache_manifest(
        cache_dir,
        expected_fingerprint=UNIT_FINGERPRINT,
        level="manifest",
    )

    assert manifest["micro_step_count"] == 2
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_rank_micro_steps_from_cache(
            cache_dir,
            expected_fingerprint=UNIT_FINGERPRINT,
            schedule=_schedule(
                resolved_max_steps=1,
                grad_accum_steps=1,
                world_size=1,
                effective_batch_size=1,
            ),
            rank=0,
            world_size=1,
        )


def test_load_cache_manifest_requires_known_explicit_level(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    write_micro_step_cache(
        cache_dir,
        (_micro_step(0),),
        fingerprint=UNIT_FINGERPRINT,
        determinants=UNIT_DETERMINANTS,
    )

    with pytest.raises(ValueError, match="verification level"):
        load_cache_manifest(
            cache_dir,
            expected_fingerprint=UNIT_FINGERPRINT,
            level="unknown",
        )
