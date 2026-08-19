from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import ctypes
import errno
import gc
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import shutil
import time
from typing import Any
import weakref

import pytest

from src.config.loader import load_train_config
from src.config.models import RuntimeBatchResolution
from src.losses.vocab import TokenVocabularyGroups
from src.training import pack_cache
from src.training.pack_cache import (
    DEFAULT_PACK_CACHE_MATERIALIZATION_WORKERS,
    EvalCacheEntry,
    EvalCacheShard,
    PackingCacheInvalidError,
    PACKING_CACHE_MATERIALIZATION_STRATEGY,
    build_packing_cache_materialization,
    build_packing_cache_determinants,
    build_packing_cache_fingerprint,
    load_all_micro_steps_from_cache as _load_all_micro_steps_from_cache,
    load_cache_manifest as _load_cache_manifest,
    load_rank_eval_micro_steps_from_cache as _load_rank_eval_micro_steps_from_cache,
    load_rank_micro_steps_from_cache as _load_rank_micro_steps_from_cache,
    write_micro_step_cache as _write_micro_step_cache,
)
from src.training.schedule import ResolvedStepSchedule
from src.training.supervised_trainer import SupervisedMicroStep


FIXTURE_CONFIG = Path("tests/fixtures/smoke/qwen3_vl_single_image_pack/config.yaml")
FAKE_VOCAB_GROUPS = TokenVocabularyGroups(
    vocab_size=6,
    desc_text=(0, 1),
    schema=(2,),
    coordinate=(3,),
    eos=(4,),
    blocked=(5,),
)
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


def _unit_determinants(
    *,
    purpose: str = "unit-test",
    split: str = "train",
) -> dict[str, Any]:
    semantic_payload = {
        "version": pack_cache.PACKING_CACHE_VERSION,
        "split": split,
        "dataset": {"purpose": purpose, "split": split},
        "template": {"purpose": purpose},
        "packing": {"global_max_length": 8},
        "processor": {"purpose": purpose},
        "ordering": {"purpose": purpose},
        "augmentation": {"split": split, "purpose": purpose},
        "qwen": {
            "processor_identity": {"purpose": purpose},
            "token_identity": {"purpose": purpose},
            "encoding_identity": {"purpose": purpose},
            "model_config_assets": {"purpose": purpose},
            "processor_assets": {"purpose": purpose},
            "tokenizer_assets": {"purpose": purpose},
        },
        "realized_vocab_groups": {
            "vocab_size": 6,
            "desc_text": [0, 1],
            "schema": [2],
            "coordinate": [3],
            "eos": [4],
            "blocked": [5],
        },
        "micro_step_runtime_config": {
            "fa2_model_dtype": "bf16",
            "capture_fa2_branch": False,
            "require_fa2_branch_proof": False,
        },
        "micro_step_schema": pack_cache._supervised_micro_step_schema_identity(),
    }
    entries = pack_cache._build_determinant_entries(semantic_payload)
    return {
        **semantic_payload,
        "registry_schema_version": (
            pack_cache.PACKING_CACHE_DETERMINANT_REGISTRY_VERSION
        ),
        "determinants": entries,
        "aggregate_fingerprint": pack_cache._registry_entries_fingerprint(entries),
        "code_identity": pack_cache._registry_code_identity(entries),
    }


UNIT_DETERMINANTS = _unit_determinants()
UNIT_FINGERPRINT = pack_cache._determinant_fingerprint(UNIT_DETERMINANTS)
EVAL_UNIT_DETERMINANTS = _unit_determinants(
    purpose="unit-test-eval",
    split="eval.forward",
)
EVAL_UNIT_FINGERPRINT = pack_cache._determinant_fingerprint(EVAL_UNIT_DETERMINANTS)


def write_micro_step_cache(*args: Any, **kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("cache_root", _cache_root_from_cache_dir(args[0]))
    kwargs.setdefault("augmentation", DISABLED_AUGMENTATION)
    kwargs.setdefault("materialization", build_packing_cache_materialization())
    kwargs.setdefault("determinant_revalidator", lambda: kwargs["determinants"])
    return _write_micro_step_cache(*args, **kwargs)


def load_cache_manifest(*args: Any, **kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("cache_root", _cache_root_from_cache_dir(args[0]))
    return _load_cache_manifest(*args, **kwargs)


def load_rank_micro_steps_from_cache(
    *args: Any,
    **kwargs: Any,
) -> tuple[SupervisedMicroStep, ...]:
    kwargs.setdefault("cache_root", _cache_root_from_cache_dir(args[0]))
    return _load_rank_micro_steps_from_cache(*args, **kwargs)


def load_rank_eval_micro_steps_from_cache(
    *args: Any,
    **kwargs: Any,
) -> EvalCacheShard:
    kwargs.setdefault("cache_root", _cache_root_from_cache_dir(args[0]))
    return _load_rank_eval_micro_steps_from_cache(*args, **kwargs)


def load_all_micro_steps_from_cache(
    *args: Any,
    **kwargs: Any,
) -> tuple[SupervisedMicroStep, ...]:
    kwargs.setdefault("cache_root", _cache_root_from_cache_dir(args[0]))
    return _load_all_micro_steps_from_cache(*args, **kwargs)


def _cache_root_from_cache_dir(cache_dir: str | Path) -> Path:
    return Path(cache_dir).parent.parent


def _cache_dir(cache_root: Path, fingerprint: str = UNIT_FINGERPRINT) -> Path:
    return pack_cache.cache_dir_for_fingerprint(cache_root, fingerprint)


def _write_valid_dataset(path: Path, *, example_id: str) -> None:
    image_path = path.parent / "image.bin"
    image_path.write_bytes(b"unit-test-image")
    path.write_text(
        json.dumps(
            {
                "example_id": example_id,
                "image": {"path": image_path.name, "width": 1, "height": 1},
                "objects": [
                    {
                        "object_id": "object-0",
                        "description": "object",
                        "bbox": [0, 0, 1, 1],
                    }
                ],
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


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
        schedule.resolved_max_steps * schedule.runtime_batch.resolved_grad_accum_steps
    )

    def iter_repeated() -> Iterator[SupervisedMicroStep]:
        for index in range(total_rank_local_micro_steps):
            planned_step_index = (
                index // schedule.runtime_batch.resolved_grad_accum_steps
            )
            local_accum_index = index % schedule.runtime_batch.resolved_grad_accum_steps
            global_micro_step_index = (
                planned_step_index * schedule.runtime_batch.effective_batch_size
                + local_accum_index * world_size
                + rank
            )
            yield base_micro_steps[global_micro_step_index % len(base_micro_steps)]

    return iter_repeated()


def _overlap_writer_process(
    cache_dir_text: str,
    pause_publish: bool,
    paused: Any,
    release: Any,
    started: Any,
    results: Any,
) -> None:
    cache_dir = Path(cache_dir_text)
    started.set()
    real_publish = pack_cache._publish_micro_step_cache
    published = False

    def tracked_publish(*args: Any, **kwargs: Any) -> Any:
        nonlocal published
        published = True
        if pause_publish:
            paused.set()
            if not release.wait(timeout=30):
                raise RuntimeError("publication pause timed out")
        return real_publish(*args, **kwargs)

    pack_cache._publish_micro_step_cache = tracked_publish
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
                published,
                manifest["fingerprint"],
                [step.metadata["pack_id"] for step in loaded],
                _cache_tree_digest(cache_dir),
            )
        )
    except Exception as exc:
        results.put(("error", type(exc).__name__, str(exc)))


def test_packing_cache_fingerprint_tracks_data_template_and_pack_policy(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "train.coord.jsonl"
    _write_valid_dataset(dataset, example_id="ex-0")
    config = load_train_config(FIXTURE_CONFIG).config
    config = config.model_copy(
        update={
            "data": config.data.model_copy(
                update={
                    "train": config.data.train.model_copy(update={"path": str(dataset)})
                }
            )
        }
    )
    components = FakeComponents(tmp_path / "fake-model")

    baseline = build_packing_cache_fingerprint(
        config,
        components,
        dataset=config.data.train,
        split="train",
        vocab_groups=FAKE_VOCAB_GROUPS,
    )
    changed_epochs = build_packing_cache_fingerprint(
        config.model_copy(
            update={"training": config.training.model_copy(update={"epochs": 7})}
        ),
        components,
        dataset=config.data.train,
        split="train",
        vocab_groups=FAKE_VOCAB_GROUPS,
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
        vocab_groups=FAKE_VOCAB_GROUPS,
    )
    changed_packing_policy = build_packing_cache_fingerprint(
        config.model_copy(
            update={
                "packing": config.packing.model_copy(
                    update={"policy": "window_binpack", "window_size": 32}
                )
            }
        ),
        components,
        dataset=config.data.train,
        split="train",
        vocab_groups=FAKE_VOCAB_GROUPS,
    )
    changed_planner_worker_count = build_packing_cache_fingerprint(
        config.model_copy(
            update={"packing": config.packing.model_copy(update={"worker_count": 8})}
        ),
        components,
        dataset=config.data.train,
        split="train",
        vocab_groups=FAKE_VOCAB_GROUPS,
    )

    assert changed_epochs == baseline
    assert changed_prompt != baseline
    assert changed_packing_policy != baseline
    assert changed_planner_worker_count != baseline


def test_cache_dir_for_fingerprint_uses_v3_namespace(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache-root"
    fingerprint = "a" * 64

    assert pack_cache.cache_dir_for_fingerprint(cache_root, fingerprint) == (
        cache_root.resolve() / "coordexp-swift-pack-cache-v3" / fingerprint
    )


@pytest.mark.parametrize(
    "fingerprint",
    ("", "abc123", "A" * 64, "g" * 64, "a" * 63, "../" + "a" * 64),
)
def test_cache_dir_for_fingerprint_rejects_noncanonical_hashes(
    tmp_path: Path,
    fingerprint: str,
) -> None:
    with pytest.raises(ValueError, match="64 lowercase hexadecimal"):
        pack_cache.cache_dir_for_fingerprint(tmp_path, fingerprint)


def test_cache_dir_for_fingerprint_rejects_traversal_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="traversal"):
        pack_cache.cache_dir_for_fingerprint(
            tmp_path / "nested" / "..",
            UNIT_FINGERPRINT,
        )


@pytest.mark.parametrize(
    "path_kind",
    ("arbitrary", "wrong-version", "wrong-fingerprint", "traversal"),
)
def test_cache_writer_rejects_noncanonical_target_before_creating_directories(
    tmp_path: Path,
    path_kind: str,
) -> None:
    cache_root = tmp_path / "must-remain-absent"
    if path_kind == "arbitrary":
        cache_dir = cache_root / UNIT_FINGERPRINT
    elif path_kind == "wrong-version":
        cache_dir = cache_root / "coordexp-swift-pack-cache-v2" / UNIT_FINGERPRINT
    else:
        cache_dir = cache_root / pack_cache.PACKING_CACHE_VERSION / ("f" * 64)
        if path_kind == "traversal":
            cache_dir = (
                cache_root
                / "nested"
                / ".."
                / pack_cache.PACKING_CACHE_VERSION
                / UNIT_FINGERPRINT
            )

    with pytest.raises(ValueError, match="canonical|traversal"):
        write_micro_step_cache(
            cache_dir,
            (_micro_step(0),),
            fingerprint=UNIT_FINGERPRINT,
            determinants=UNIT_DETERMINANTS,
        )

    assert not cache_root.exists()


@pytest.mark.parametrize(
    "operation",
    ("manifest", "rank", "all", "complete", "writer"),
)
def test_public_cache_apis_reject_alternate_well_shaped_root_before_consumption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    selected_cache_root = tmp_path / "canonical-root"
    cache_dir = _cache_dir(selected_cache_root)
    write_micro_step_cache(
        cache_dir,
        (_micro_step(0),),
        fingerprint=UNIT_FINGERPRINT,
        determinants=UNIT_DETERMINANTS,
    )
    alternate_cache_root = tmp_path / "alternate-root"
    copied_dir = _cache_dir(alternate_cache_root)
    copied_dir.parent.mkdir(parents=True)
    shutil.copytree(cache_dir, copied_dir)
    assert {
        path.relative_to(cache_dir).as_posix(): path.read_bytes()
        for path in cache_dir.rglob("*")
        if path.is_file()
    } == {
        path.relative_to(copied_dir).as_posix(): path.read_bytes()
        for path in copied_dir.rglob("*")
        if path.is_file()
    }

    def fail_if_consumed(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("noncanonical cache reached manifest consumption")

    monkeypatch.setattr(pack_cache, "_load_validated_manifest", fail_if_consumed)
    schedule = _schedule(
        resolved_max_steps=1,
        grad_accum_steps=1,
        world_size=1,
        effective_batch_size=1,
    )
    if operation == "complete":
        assert not pack_cache.cache_is_complete(
            copied_dir,
            cache_root=selected_cache_root,
            fingerprint=UNIT_FINGERPRINT,
        )
        return
    expected_error = ValueError if operation == "writer" else PackingCacheInvalidError
    with pytest.raises(expected_error, match="canonical"):
        if operation == "manifest":
            pack_cache.load_cache_manifest(
                copied_dir,
                cache_root=selected_cache_root,
                expected_fingerprint=UNIT_FINGERPRINT,
                level="payloads",
            )
        elif operation == "rank":
            pack_cache.load_rank_micro_steps_from_cache(
                copied_dir,
                cache_root=selected_cache_root,
                expected_fingerprint=UNIT_FINGERPRINT,
                schedule=schedule,
                rank=0,
                world_size=1,
            )
        elif operation == "all":
            pack_cache.load_all_micro_steps_from_cache(
                copied_dir,
                cache_root=selected_cache_root,
                expected_fingerprint=UNIT_FINGERPRINT,
            )
        else:
            pack_cache.write_micro_step_cache(
                copied_dir,
                (_micro_step(0),),
                cache_root=selected_cache_root,
                fingerprint=UNIT_FINGERPRINT,
                determinants=UNIT_DETERMINANTS,
                materialization=build_packing_cache_materialization(),
                determinant_revalidator=lambda: UNIT_DETERMINANTS,
                augmentation=DISABLED_AUGMENTATION,
            )


@pytest.mark.parametrize("symlink_component", ("cache-root", "version", "fingerprint"))
@pytest.mark.parametrize(
    "operation",
    ("manifest", "rank", "all", "complete", "writer"),
)
def test_public_cache_apis_reject_symlinked_cache_components_before_consumption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    symlink_component: str,
    operation: str,
) -> None:
    real_cache_root = tmp_path / "real-root"
    real_cache_dir = _cache_dir(real_cache_root)
    write_micro_step_cache(
        real_cache_dir,
        (_micro_step(0),),
        fingerprint=UNIT_FINGERPRINT,
        determinants=UNIT_DETERMINANTS,
    )
    selected_cache_root = tmp_path / "selected-root"
    if symlink_component == "cache-root":
        selected_cache_root.symlink_to(real_cache_root, target_is_directory=True)
    elif symlink_component == "version":
        selected_cache_root.mkdir()
        (selected_cache_root / pack_cache.PACKING_CACHE_VERSION).symlink_to(
            real_cache_root / pack_cache.PACKING_CACHE_VERSION,
            target_is_directory=True,
        )
    else:
        (selected_cache_root / pack_cache.PACKING_CACHE_VERSION).mkdir(parents=True)
        (
            selected_cache_root / pack_cache.PACKING_CACHE_VERSION / UNIT_FINGERPRINT
        ).symlink_to(real_cache_dir, target_is_directory=True)
    candidate = (
        selected_cache_root / pack_cache.PACKING_CACHE_VERSION / UNIT_FINGERPRINT
    )

    def fail_if_consumed(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("symlinked cache reached manifest consumption")

    monkeypatch.setattr(pack_cache, "_load_validated_manifest", fail_if_consumed)
    schedule = _schedule(
        resolved_max_steps=1,
        grad_accum_steps=1,
        world_size=1,
        effective_batch_size=1,
    )
    if operation == "complete":
        assert not pack_cache.cache_is_complete(
            candidate,
            cache_root=selected_cache_root,
            fingerprint=UNIT_FINGERPRINT,
        )
        return
    expected_error = ValueError if operation == "writer" else PackingCacheInvalidError
    with pytest.raises(expected_error, match="symlink"):
        if operation == "manifest":
            pack_cache.load_cache_manifest(
                candidate,
                cache_root=selected_cache_root,
                expected_fingerprint=UNIT_FINGERPRINT,
                level="payloads",
            )
        elif operation == "rank":
            pack_cache.load_rank_micro_steps_from_cache(
                candidate,
                cache_root=selected_cache_root,
                expected_fingerprint=UNIT_FINGERPRINT,
                schedule=schedule,
                rank=0,
                world_size=1,
            )
        elif operation == "all":
            pack_cache.load_all_micro_steps_from_cache(
                candidate,
                cache_root=selected_cache_root,
                expected_fingerprint=UNIT_FINGERPRINT,
            )
        else:
            pack_cache.write_micro_step_cache(
                candidate,
                (_micro_step(0),),
                cache_root=selected_cache_root,
                fingerprint=UNIT_FINGERPRINT,
                determinants=UNIT_DETERMINANTS,
                materialization=build_packing_cache_materialization(),
                determinant_revalidator=lambda: UNIT_DETERMINANTS,
                augmentation=DISABLED_AUGMENTATION,
            )


def test_packing_cache_fingerprint_is_timestamp_independent_and_tracks_content(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "train.coord.jsonl"
    _write_valid_dataset(dataset, example_id="ex-a")
    stat = dataset.stat()
    config = load_train_config(FIXTURE_CONFIG).config
    config = config.model_copy(
        update={
            "data": config.data.model_copy(
                update={
                    "train": config.data.train.model_copy(update={"path": str(dataset)})
                }
            )
        }
    )
    components = FakeComponents(tmp_path / "fake-model")

    baseline = build_packing_cache_fingerprint(
        config,
        components,
        dataset=config.data.train,
        split="train",
        vocab_groups=FAKE_VOCAB_GROUPS,
    )
    baseline_determinants = build_packing_cache_determinants(
        config,
        components,
        dataset=config.data.train,
        split="train",
        vocab_groups=FAKE_VOCAB_GROUPS,
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
        vocab_groups=FAKE_VOCAB_GROUPS,
    )
    assert touched == baseline

    # Byte content change MUST change the fingerprint, independent of the
    # timestamp (forced back to the original value here to isolate the
    # effect to content alone).
    _write_valid_dataset(dataset, example_id="ex-b")
    os.utime(dataset, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    changed_content = build_packing_cache_fingerprint(
        config,
        components,
        dataset=config.data.train,
        split="train",
        vocab_groups=FAKE_VOCAB_GROUPS,
    )

    assert dataset.stat().st_size == stat.st_size
    assert dataset.stat().st_mtime_ns == stat.st_mtime_ns
    assert changed_content != baseline


def test_packing_cache_fingerprint_tracks_image_pad_token_id(tmp_path: Path) -> None:
    dataset = tmp_path / "train.coord.jsonl"
    _write_valid_dataset(dataset, example_id="ex-0")
    config = load_train_config(FIXTURE_CONFIG).config
    config = config.model_copy(
        update={
            "data": config.data.model_copy(
                update={
                    "train": config.data.train.model_copy(update={"path": str(dataset)})
                }
            )
        }
    )

    baseline = build_packing_cache_fingerprint(
        config,
        FakeComponents(tmp_path / "fake-model", image_pad_token_id=151655),
        dataset=config.data.train,
        split="train",
        vocab_groups=FAKE_VOCAB_GROUPS,
    )
    changed_image_pad = build_packing_cache_fingerprint(
        config,
        FakeComponents(tmp_path / "fake-model", image_pad_token_id=151700),
        dataset=config.data.train,
        split="train",
        vocab_groups=FAKE_VOCAB_GROUPS,
    )

    assert changed_image_pad != baseline


def test_packing_cache_registry_exposes_source_owner_identities(tmp_path: Path) -> None:
    dataset = tmp_path / "train.coord.jsonl"
    _write_valid_dataset(dataset, example_id="ex-0")
    config = load_train_config(FIXTURE_CONFIG).config
    config = config.model_copy(
        update={
            "data": config.data.model_copy(
                update={
                    "train": config.data.train.model_copy(update={"path": str(dataset)})
                }
            )
        }
    )

    determinants = build_packing_cache_determinants(
        config,
        FakeComponents(tmp_path / "fake-model"),
        dataset=config.data.train,
        split="train",
        vocab_groups=FAKE_VOCAB_GROUPS,
    )

    code_identity = determinants["code_identity"]
    assert set(code_identity) == set(pack_cache.PACKING_CACHE_DETERMINANT_OWNERS)
    for payload in code_identity.values():
        assert payload["sha256"]
        assert len(payload["sha256"]) == 64
        assert payload["path"].startswith("src/")


def test_packing_cache_fingerprint_tracks_augmentation_config_and_seed(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "train.coord.jsonl"
    _write_valid_dataset(dataset, example_id="ex-0")
    config = load_train_config(FIXTURE_CONFIG).config
    config = config.model_copy(
        update={
            "data": config.data.model_copy(
                update={
                    "train": config.data.train.model_copy(update={"path": str(dataset)})
                }
            )
        }
    )
    components = FakeComponents(tmp_path / "fake-model")

    baseline = build_packing_cache_fingerprint(
        config,
        components,
        dataset=config.data.train,
        split="train",
        vocab_groups=FAKE_VOCAB_GROUPS,
    )
    enabled = build_packing_cache_fingerprint(
        _config_with_geometry_flips(config, horizontal_prob=1.0),
        components,
        dataset=config.data.train,
        split="train",
        vocab_groups=FAKE_VOCAB_GROUPS,
    )
    changed_probability = build_packing_cache_fingerprint(
        _config_with_geometry_flips(config, horizontal_prob=0.5),
        components,
        dataset=config.data.train,
        split="train",
        vocab_groups=FAKE_VOCAB_GROUPS,
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
        vocab_groups=FAKE_VOCAB_GROUPS,
    )
    determinants = build_packing_cache_determinants(
        _config_with_geometry_flips(config, horizontal_prob=1.0),
        components,
        dataset=config.data.train,
        split="train",
        vocab_groups=FAKE_VOCAB_GROUPS,
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
    _write_valid_dataset(dataset, example_id="ex-0")
    config = load_train_config(FIXTURE_CONFIG).config
    config = config.model_copy(
        update={
            "data": config.data.model_copy(
                update={
                    "train": config.data.train.model_copy(update={"path": str(dataset)})
                }
            )
        }
    )
    components = FakeComponents(tmp_path / "fake-model")
    fingerprint = build_packing_cache_fingerprint(
        config,
        components,
        dataset=config.data.train,
        split="train",
        vocab_groups=FAKE_VOCAB_GROUPS,
    )
    determinants = build_packing_cache_determinants(
        config,
        components,
        dataset=config.data.train,
        split="train",
        vocab_groups=FAKE_VOCAB_GROUPS,
    )

    default_cache = _cache_dir(tmp_path / "default-cache", fingerprint)
    override_cache = _cache_dir(tmp_path / "override-cache", fingerprint)
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
    assert (
        default_manifest["materialization"]["workers"]
        != (override_manifest["materialization"]["workers"])
    )


def test_micro_step_cache_manifest_records_default_materialization_workers(
    tmp_path: Path,
) -> None:
    cache_dir = _cache_dir(tmp_path)

    manifest = write_micro_step_cache(
        cache_dir,
        (_micro_step(0),),
        fingerprint=UNIT_FINGERPRINT,
        determinants=UNIT_DETERMINANTS,
    )

    assert manifest["materialization"] == {
        "strategy": PACKING_CACHE_MATERIALIZATION_STRATEGY,
        "workers": DEFAULT_PACK_CACHE_MATERIALIZATION_WORKERS,
    }
    assert (
        load_cache_manifest(
            cache_dir, expected_fingerprint=UNIT_FINGERPRINT, level="payloads"
        )["materialization"]
        == manifest["materialization"]
    )


def test_micro_step_cache_manifest_records_explicit_materialization_override(
    tmp_path: Path,
) -> None:
    cache_dir = _cache_dir(tmp_path)

    manifest = write_micro_step_cache(
        cache_dir,
        (_micro_step(0),),
        fingerprint=UNIT_FINGERPRINT,
        determinants=UNIT_DETERMINANTS,
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
    cache_dir = _cache_dir(tmp_path)
    write_micro_step_cache(
        cache_dir,
        (_micro_step(0),),
        fingerprint=UNIT_FINGERPRINT,
        determinants=UNIT_DETERMINANTS,
    )
    manifest_path = cache_dir / "manifest.json"
    invalid_manifest = json.loads(manifest_path.read_text())
    invalid_manifest.pop("materialization")
    manifest_path.write_text(json.dumps(invalid_manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="materialization"):
        load_cache_manifest(
            cache_dir, expected_fingerprint=UNIT_FINGERPRINT, level="payloads"
        )


@pytest.mark.parametrize("augmentation", [None, {}])
def test_cache_writer_rejects_missing_or_empty_augmentation_receipt(
    tmp_path: Path, augmentation: Any
) -> None:
    with pytest.raises(ValueError, match="augmentation.*non-empty"):
        _write_micro_step_cache(
            _cache_dir(tmp_path),
            (_micro_step(0),),
            cache_root=tmp_path,
            fingerprint=UNIT_FINGERPRINT,
            determinants=UNIT_DETERMINANTS,
            materialization=build_packing_cache_materialization(),
            determinant_revalidator=lambda: UNIT_DETERMINANTS,
            augmentation=augmentation,
        )


def test_cache_writer_requires_explicit_materialization(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="materialization"):
        _write_micro_step_cache(
            _cache_dir(tmp_path),
            (_micro_step(0),),
            cache_root=tmp_path,
            fingerprint=UNIT_FINGERPRINT,
            determinants=UNIT_DETERMINANTS,
            determinant_revalidator=lambda: UNIT_DETERMINANTS,
            augmentation=DISABLED_AUGMENTATION,
        )


def test_v3_cache_writer_rejects_missing_determinant_registry(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="determinant registry version"):
        _write_micro_step_cache(
            _cache_dir(tmp_path),
            (_micro_step(0),),
            cache_root=tmp_path,
            fingerprint="a" * 64,
            determinants={"purpose": "legacy-unit-test"},
            materialization=build_packing_cache_materialization(),
            determinant_revalidator=lambda: UNIT_DETERMINANTS,
            augmentation=DISABLED_AUGMENTATION,
        )


def test_v3_cache_manifest_rejects_missing_determinant_registry(
    tmp_path: Path,
) -> None:
    cache_dir = _cache_dir(tmp_path)
    write_micro_step_cache(
        cache_dir,
        (_micro_step(0),),
        fingerprint=UNIT_FINGERPRINT,
        determinants=UNIT_DETERMINANTS,
    )
    path = cache_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["determinants"] = {"purpose": "legacy-unit-test"}
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(PackingCacheInvalidError, match="determinant registry version"):
        load_cache_manifest(
            cache_dir,
            expected_fingerprint=UNIT_FINGERPRINT,
            level="manifest",
        )


@pytest.mark.parametrize("mutation", ["omit", "empty"])
def test_cache_reader_rejects_missing_or_empty_augmentation_receipt(
    tmp_path: Path, mutation: str
) -> None:
    cache_dir = _cache_dir(tmp_path)
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
        load_cache_manifest(
            cache_dir, expected_fingerprint=UNIT_FINGERPRINT, level="payloads"
        )


def test_micro_step_cache_loads_exact_rank_local_training_order(tmp_path: Path) -> None:
    micro_steps = tuple(_micro_step(index) for index in range(10))
    cache_dir = _cache_dir(tmp_path)
    manifest = write_micro_step_cache(
        cache_dir,
        micro_steps,
        fingerprint=UNIT_FINGERPRINT,
        determinants=UNIT_DETERMINANTS,
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


def test_micro_step_cache_wraps_tail_presentations_by_pack_count(
    tmp_path: Path,
) -> None:
    micro_steps = tuple(_micro_step(index) for index in range(3))
    cache_dir = _cache_dir(tmp_path)
    write_micro_step_cache(
        cache_dir,
        micro_steps,
        fingerprint=UNIT_FINGERPRINT,
        determinants=UNIT_DETERMINANTS,
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
    cache_dir = _cache_dir(tmp_path, EVAL_UNIT_FINGERPRINT)
    write_micro_step_cache(
        cache_dir,
        micro_steps,
        fingerprint=EVAL_UNIT_FINGERPRINT,
        determinants=EVAL_UNIT_DETERMINANTS,
        chunk_size=3,
    )

    loaded = load_all_micro_steps_from_cache(
        cache_dir,
        expected_fingerprint=EVAL_UNIT_FINGERPRINT,
    )

    assert [step.metadata["pack_id"] for step in loaded] == list(range(7))


def test_cached_rank_local_steps_must_not_be_sharded_again(tmp_path: Path) -> None:
    micro_steps = tuple(_micro_step(index) for index in range(2048))
    cache_dir = _cache_dir(tmp_path)
    write_micro_step_cache(
        cache_dir,
        micro_steps,
        fingerprint=UNIT_FINGERPRINT,
        determinants=UNIT_DETERMINANTS,
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
    cache_dir = _cache_dir(tmp_path)
    write_micro_step_cache(
        cache_dir,
        micro_steps,
        fingerprint=UNIT_FINGERPRINT,
        determinants=UNIT_DETERMINANTS,
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


def test_cache_reader_decodes_authenticated_snapshot_after_chunk_path_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = _cache_dir(tmp_path)
    write_micro_step_cache(
        cache_dir,
        (_micro_step(0),),
        fingerprint=UNIT_FINGERPRINT,
        determinants=UNIT_DETERMINANTS,
    )
    chunk_path = cache_dir / "chunks" / "chunk-00000.pkl"
    swapped_path = chunk_path.with_name("swapped.pkl")
    swapped_path.write_bytes(pickle.dumps((_micro_step(99),)))
    real_load = pack_cache._RestrictedCacheUnpickler.load
    swap_count = 0

    def swap_before_decode(unpickler: Any) -> Any:
        nonlocal swap_count
        if swap_count == 0:
            os.replace(swapped_path, chunk_path)
        swap_count += 1
        return real_load(unpickler)

    monkeypatch.setattr(
        pack_cache._RestrictedCacheUnpickler,
        "load",
        swap_before_decode,
    )

    loaded = load_all_micro_steps_from_cache(
        cache_dir,
        expected_fingerprint=UNIT_FINGERPRINT,
    )

    assert swap_count == 1
    assert [step.metadata["pack_id"] for step in loaded] == [0]
    with pytest.raises(PackingCacheInvalidError, match="checksum mismatch"):
        load_all_micro_steps_from_cache(
            cache_dir,
            expected_fingerprint=UNIT_FINGERPRINT,
        )


def test_cache_reader_rejects_in_cache_chunk_symlink(tmp_path: Path) -> None:
    cache_dir = _cache_dir(tmp_path)
    write_micro_step_cache(
        cache_dir,
        (_micro_step(0),),
        fingerprint=UNIT_FINGERPRINT,
        determinants=UNIT_DETERMINANTS,
    )
    chunk_path = cache_dir / "chunks" / "chunk-00000.pkl"
    backing_path = chunk_path.with_name("chunk-backing.pkl")
    chunk_path.rename(backing_path)
    chunk_path.symlink_to(backing_path.name)

    with pytest.raises(PackingCacheInvalidError, match="symlink"):
        load_all_micro_steps_from_cache(
            cache_dir,
            expected_fingerprint=UNIT_FINGERPRINT,
        )


def test_cache_reader_rejects_chunk_snapshot_over_byte_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = _cache_dir(tmp_path)
    write_micro_step_cache(
        cache_dir,
        (_micro_step(0),),
        fingerprint=UNIT_FINGERPRINT,
        determinants=UNIT_DETERMINANTS,
    )
    chunk_path = cache_dir / "chunks" / "chunk-00000.pkl"
    monkeypatch.setattr(
        pack_cache,
        "_MAX_PACK_CACHE_CHUNK_SNAPSHOT_BYTES",
        chunk_path.stat().st_size - 1,
    )

    with pytest.raises(PackingCacheInvalidError, match="unreadable"):
        load_all_micro_steps_from_cache(
            cache_dir,
            expected_fingerprint=UNIT_FINGERPRINT,
        )


def test_cache_reader_rejects_short_chunk_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = _cache_dir(tmp_path)
    write_micro_step_cache(
        cache_dir,
        (_micro_step(0),),
        fingerprint=UNIT_FINGERPRINT,
        determinants=UNIT_DETERMINANTS,
    )
    monkeypatch.setattr(pack_cache.os, "read", lambda _fd, _size: b"")

    with pytest.raises(PackingCacheInvalidError, match="unreadable"):
        load_all_micro_steps_from_cache(
            cache_dir,
            expected_fingerprint=UNIT_FINGERPRINT,
        )


def test_cache_reader_rejects_chunk_metadata_change_during_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = _cache_dir(tmp_path)
    write_micro_step_cache(
        cache_dir,
        (_micro_step(0),),
        fingerprint=UNIT_FINGERPRINT,
        determinants=UNIT_DETERMINANTS,
    )
    real_fstat = pack_cache.os.fstat
    fstat_count = 0

    class ChangedStat:
        def __init__(self, original: os.stat_result) -> None:
            self._original = original

        def __getattr__(self, name: str) -> Any:
            value = getattr(self._original, name)
            return value + 1 if name == "st_ctime_ns" else value

    def changing_fstat(fd: int) -> os.stat_result | ChangedStat:
        nonlocal fstat_count
        observed = real_fstat(fd)
        fstat_count += 1
        return ChangedStat(observed) if fstat_count == 2 else observed

    monkeypatch.setattr(pack_cache.os, "fstat", changing_fstat)

    with pytest.raises(PackingCacheInvalidError, match="unreadable"):
        load_all_micro_steps_from_cache(
            cache_dir,
            expected_fingerprint=UNIT_FINGERPRINT,
        )
    assert fstat_count == 2


def test_micro_step_cache_manifest_rejects_chunk_gaps(tmp_path: Path) -> None:
    micro_steps = tuple(_micro_step(index) for index in range(4))
    cache_dir = _cache_dir(tmp_path)
    write_micro_step_cache(
        cache_dir,
        micro_steps,
        fingerprint=UNIT_FINGERPRINT,
        determinants=UNIT_DETERMINANTS,
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
        (
            lambda manifest: manifest["chunks"][0].__setitem__("sha256", "z" * 64),
            "sha256",
        ),
        (
            lambda manifest: manifest["chunks"][0].__setitem__("path", "../escape.pkl"),
            "path",
        ),
    ],
)
def test_cache_manifest_rejects_invalid_current_contract(
    tmp_path: Path, mutation: Any, match: str
) -> None:
    cache_dir = _cache_dir(tmp_path)
    write_micro_step_cache(
        cache_dir,
        tuple(_micro_step(index) for index in range(4)),
        fingerprint=UNIT_FINGERPRINT,
        determinants=UNIT_DETERMINANTS,
        chunk_size=2,
    )
    path = cache_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    mutation(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_cache_manifest(
            cache_dir, expected_fingerprint=UNIT_FINGERPRINT, level="payloads"
        )


def test_cache_manifest_rejects_corrupt_json(tmp_path: Path) -> None:
    cache_dir = _cache_dir(tmp_path)
    write_micro_step_cache(
        cache_dir,
        (_micro_step(0),),
        fingerprint=UNIT_FINGERPRINT,
        determinants=UNIT_DETERMINANTS,
    )
    (cache_dir / "manifest.json").write_text("{", encoding="utf-8")

    with pytest.raises(PackingCacheInvalidError):
        load_cache_manifest(
            cache_dir, expected_fingerprint=UNIT_FINGERPRINT, level="payloads"
        )


def test_cache_reader_rejects_corrupt_pickle_with_matching_declared_hash(
    tmp_path: Path,
) -> None:
    cache_dir = _cache_dir(tmp_path)
    write_micro_step_cache(
        cache_dir,
        (_micro_step(0),),
        fingerprint=UNIT_FINGERPRINT,
        determinants=UNIT_DETERMINANTS,
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
    cache_dir = _cache_dir(tmp_path)
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
    cache_dir = _cache_dir(tmp_path)
    write_micro_step_cache(
        cache_dir,
        (_micro_step(0),),
        fingerprint=UNIT_FINGERPRINT,
        determinants=UNIT_DETERMINANTS,
    )

    def interrupt_restore(_self: Any) -> Any:
        raise interrupt()

    monkeypatch.setattr(pack_cache._RestrictedCacheUnpickler, "load", interrupt_restore)
    with pytest.raises(interrupt):
        load_all_micro_steps_from_cache(
            cache_dir, expected_fingerprint=UNIT_FINGERPRINT
        )


def test_restricted_cache_unpickler_rejects_reduce_before_side_effect(
    tmp_path: Path,
) -> None:
    cache_dir = _cache_dir(tmp_path)
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
    cache_dir = _cache_dir(tmp_path)
    write_micro_step_cache(
        cache_dir,
        (_micro_step(0),),
        fingerprint=UNIT_FINGERPRINT,
        determinants=UNIT_DETERMINANTS,
    )
    (cache_dir / "chunks" / "chunk-00000.pkl").unlink()

    with pytest.raises(ValueError, match="missing"):
        load_cache_manifest(
            cache_dir, expected_fingerprint=UNIT_FINGERPRINT, level="payloads"
        )


@pytest.mark.parametrize("reader", ["manifest", "rank", "all"])
def test_cache_readers_reject_mutated_determinants_with_unchanged_fingerprint(
    tmp_path: Path, reader: str
) -> None:
    cache_dir = _cache_dir(tmp_path)
    write_micro_step_cache(
        cache_dir,
        (_micro_step(0),),
        fingerprint=UNIT_FINGERPRINT,
        determinants=UNIT_DETERMINANTS,
    )
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["determinants"]["dataset"]["purpose"] = "tampered"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    schedule = _schedule(
        resolved_max_steps=1,
        grad_accum_steps=1,
        world_size=1,
        effective_batch_size=1,
    )

    with pytest.raises(ValueError, match="determinant.*mirror"):
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
    cache_dir = _cache_dir(tmp_path)
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
    cache_dir = _cache_dir(tmp_path)
    write_micro_step_cache(
        cache_dir,
        tuple(_micro_step(index) for index in range(4)),
        fingerprint=UNIT_FINGERPRINT,
        determinants=UNIT_DETERMINANTS,
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
    cache_dir = _cache_dir(tmp_path)
    write_micro_step_cache(
        cache_dir,
        tuple(_micro_step(index) for index in range(4)),
        fingerprint=UNIT_FINGERPRINT,
        determinants=UNIT_DETERMINANTS,
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
    cache_dir = _cache_dir(tmp_path)
    write_micro_step_cache(
        cache_dir,
        micro_steps,
        fingerprint=UNIT_FINGERPRINT,
        determinants=UNIT_DETERMINANTS,
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


def test_eval_manifest_declares_canonical_ordinal_index(tmp_path: Path) -> None:
    cache_dir = _cache_dir(tmp_path, EVAL_UNIT_FINGERPRINT)
    manifest = write_micro_step_cache(
        cache_dir,
        tuple(_micro_step(index) for index in range(5)),
        fingerprint=EVAL_UNIT_FINGERPRINT,
        determinants=EVAL_UNIT_DETERMINANTS,
        chunk_size=2,
    )

    assert manifest["eval_ordinal_index"] == [
        {"ordinal": 0, "chunk_index": 0, "chunk_offset": 0},
        {"ordinal": 1, "chunk_index": 0, "chunk_offset": 1},
        {"ordinal": 2, "chunk_index": 1, "chunk_offset": 0},
        {"ordinal": 3, "chunk_index": 1, "chunk_offset": 1},
        {"ordinal": 4, "chunk_index": 2, "chunk_offset": 0},
    ]


def test_eval_rank_assignments_are_disjoint_exact_cover_with_fewer_packs_than_ranks(
    tmp_path: Path,
) -> None:
    cache_dir = _cache_dir(tmp_path, EVAL_UNIT_FINGERPRINT)
    write_micro_step_cache(
        cache_dir,
        tuple(_micro_step(index) for index in range(3)),
        fingerprint=EVAL_UNIT_FINGERPRINT,
        determinants=EVAL_UNIT_DETERMINANTS,
        chunk_size=1,
    )

    shards = tuple(
        load_rank_eval_micro_steps_from_cache(
            cache_dir,
            expected_fingerprint=EVAL_UNIT_FINGERPRINT,
            rank=rank,
            world_size=5,
        )
        for rank in range(5)
    )

    assert [shard.canonical_ordinals for shard in shards] == [(0,), (1,), (2,), (), ()]
    flattened = [entry.canonical_ordinal for shard in shards for entry in shard.entries]
    assert sorted(flattened) == list(range(3))
    assert len(flattened) == len(set(flattened))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("ordinal", 4, "canonical and contiguous"),
        ("chunk_index", 9, "unknown chunk"),
        ("chunk_offset", 9, "outside its chunk"),
    ],
)
def test_eval_selective_loader_validates_every_manifest_index_declaration(
    tmp_path: Path,
    field: str,
    value: int,
    message: str,
) -> None:
    cache_dir = _cache_dir(tmp_path, EVAL_UNIT_FINGERPRINT)
    write_micro_step_cache(
        cache_dir,
        tuple(_micro_step(index) for index in range(4)),
        fingerprint=EVAL_UNIT_FINGERPRINT,
        determinants=EVAL_UNIT_DETERMINANTS,
        chunk_size=1,
    )
    manifest_file = cache_dir / pack_cache.PACKING_CACHE_MANIFEST
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    # Ordinal 3 belongs to rank 1 below. Rank 0 must still reject its malformed
    # declaration before reading any rank-local payload.
    manifest["eval_ordinal_index"][3][field] = value
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(PackingCacheInvalidError, match=message):
        load_rank_eval_micro_steps_from_cache(
            cache_dir,
            expected_fingerprint=EVAL_UNIT_FINGERPRINT,
            rank=0,
            world_size=2,
        )


def test_eval_corrupt_other_rank_chunk_is_skipped_only_by_unassigned_rank(
    tmp_path: Path,
) -> None:
    cache_dir = _cache_dir(tmp_path, EVAL_UNIT_FINGERPRINT)
    write_micro_step_cache(
        cache_dir,
        tuple(_micro_step(index) for index in range(4)),
        fingerprint=EVAL_UNIT_FINGERPRINT,
        determinants=EVAL_UNIT_DETERMINANTS,
        chunk_size=1,
    )
    # Ordinal/chunk 1 belongs only to rank 1 under modulo-2 assignment.
    (cache_dir / "chunks" / "chunk-00001.pkl").write_bytes(b"corrupt")

    rank_zero = load_rank_eval_micro_steps_from_cache(
        cache_dir,
        expected_fingerprint=EVAL_UNIT_FINGERPRINT,
        rank=0,
        world_size=2,
    )
    assert rank_zero.canonical_ordinals == (0, 2)
    assert [step.metadata["pack_id"] for step in rank_zero.micro_steps] == [0, 2]

    with pytest.raises(PackingCacheInvalidError, match="checksum mismatch"):
        load_rank_eval_micro_steps_from_cache(
            cache_dir,
            expected_fingerprint=EVAL_UNIT_FINGERPRINT,
            rank=1,
            world_size=2,
        )
    # The compatibility reference is deliberately full hydration and cannot
    # become a silent production fallback for the failed responsible rank.
    with pytest.raises(PackingCacheInvalidError, match="checksum mismatch"):
        pack_cache._load_all_eval_micro_steps_from_cache_for_test(
            cache_dir,
            cache_root=_cache_root_from_cache_dir(cache_dir),
            expected_fingerprint=EVAL_UNIT_FINGERPRINT,
        )


def test_eval_selective_and_full_hydration_are_row_exact(tmp_path: Path) -> None:
    cache_dir = _cache_dir(tmp_path, EVAL_UNIT_FINGERPRINT)
    write_micro_step_cache(
        cache_dir,
        tuple(_micro_step(index) for index in range(7)),
        fingerprint=EVAL_UNIT_FINGERPRINT,
        determinants=EVAL_UNIT_DETERMINANTS,
        chunk_size=3,
    )
    full = pack_cache._load_all_eval_micro_steps_from_cache_for_test(
        cache_dir,
        cache_root=_cache_root_from_cache_dir(cache_dir),
        expected_fingerprint=EVAL_UNIT_FINGERPRINT,
    )

    def row(entry: EvalCacheEntry) -> dict[str, Any]:
        step = entry.micro_step
        return {
            "canonical_ordinal": entry.canonical_ordinal,
            "pack_id": step.metadata["pack_id"],
            "encoded_examples": step.encoded_examples,
            "position_inputs": step.position_inputs,
            "token_sequence": step.token_sequence,
            "vocab_groups": step.vocab_groups,
            "metadata": step.metadata,
        }

    for rank in range(3):
        selective = load_rank_eval_micro_steps_from_cache(
            cache_dir,
            expected_fingerprint=EVAL_UNIT_FINGERPRINT,
            rank=rank,
            world_size=3,
        )
        full_reference_rows = [
            row(entry) for entry in full if entry.canonical_ordinal % 3 == rank
        ]
        assert [row(entry) for entry in selective.entries] == full_reference_rows


@pytest.mark.parametrize("payload_kind", ["list", "wrong-length-tuple"])
def test_cache_reader_rejects_invalid_chunk_payload_shape_or_count(
    tmp_path: Path, payload_kind: str
) -> None:
    cache_dir = _cache_dir(tmp_path)
    write_micro_step_cache(
        cache_dir,
        (_micro_step(0),),
        fingerprint=UNIT_FINGERPRINT,
        determinants=UNIT_DETERMINANTS,
    )
    chunk_path = cache_dir / "chunks" / "chunk-00000.pkl"
    payload = (
        [_micro_step(0)] if payload_kind == "list" else (_micro_step(0), _micro_step(1))
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


def test_existing_conflicting_target_fails_closed_without_byte_changes(
    tmp_path: Path,
) -> None:
    old_determinants = _unit_determinants(purpose="old-unit-test")
    new_determinants = _unit_determinants(purpose="new-unit-test")
    old_fingerprint = _fingerprint(old_determinants)
    new_fingerprint = _fingerprint(new_determinants)
    old_cache_dir = _cache_dir(tmp_path, old_fingerprint)
    cache_dir = _cache_dir(tmp_path, new_fingerprint)
    write_micro_step_cache(
        old_cache_dir,
        (_micro_step(0),),
        fingerprint=old_fingerprint,
        determinants=old_determinants,
    )
    old_cache_dir.rename(cache_dir)
    before = _cache_tree_digest(cache_dir)

    with pytest.raises(PackingCacheInvalidError) as exc_info:
        write_micro_step_cache(
            cache_dir,
            (_micro_step(1),),
            fingerprint=new_fingerprint,
            determinants=new_determinants,
        )

    _assert_collision_message(
        str(exc_info.value),
        target=cache_dir,
        fingerprint=new_fingerprint,
        validation_category="current_publication_invalid",
    )
    assert _cache_tree_digest(cache_dir) == before
    # Public readers now enforce that the directory basename matches the
    # requested manifest fingerprint. Restore this byte-identical fixture to
    # its original canonical path before proving the old payload is intact.
    cache_dir.rename(old_cache_dir)
    loaded = load_all_micro_steps_from_cache(
        old_cache_dir, expected_fingerprint=old_fingerprint
    )
    assert [step.metadata["pack_id"] for step in loaded] == [0]
    assert not list(cache_dir.parent.glob(f".{cache_dir.name}.stage-*"))
    assert not list(cache_dir.parent.glob(f".{cache_dir.name}.backup-*"))


@pytest.mark.parametrize("mutation", ["incomplete", "corrupt"])
def test_existing_invalid_target_fails_closed_without_byte_changes(
    tmp_path: Path,
    mutation: str,
) -> None:
    cache_dir = _cache_dir(tmp_path)
    write_micro_step_cache(
        cache_dir,
        (_micro_step(0),),
        fingerprint=UNIT_FINGERPRINT,
        determinants=UNIT_DETERMINANTS,
    )
    if mutation == "incomplete":
        path = cache_dir / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["status"] = "writing"
        path.write_text(json.dumps(manifest), encoding="utf-8")
    else:
        (cache_dir / "chunks" / "chunk-00000.pkl").write_bytes(b"corrupt")
    before = _cache_tree_digest(cache_dir)

    with pytest.raises(PackingCacheInvalidError) as exc_info:
        write_micro_step_cache(
            cache_dir,
            (_micro_step(1),),
            fingerprint=UNIT_FINGERPRINT,
            determinants=UNIT_DETERMINANTS,
        )

    _assert_collision_message(
        str(exc_info.value),
        target=cache_dir,
        fingerprint=UNIT_FINGERPRINT,
        validation_category="current_publication_invalid",
    )
    assert _cache_tree_digest(cache_dir) == before
    assert not list(cache_dir.parent.glob(f".{cache_dir.name}.stage-*"))
    assert not list(cache_dir.parent.glob(f".{cache_dir.name}.backup-*"))


def test_collision_message_does_not_embed_nested_validation_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = _cache_dir(tmp_path)
    cache_dir.mkdir(parents=True)
    secret_shaped_text = "SECRET_TOKEN=" + "x" * 10_000

    def fail_validation(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise PackingCacheInvalidError(secret_shaped_text)

    monkeypatch.setattr(pack_cache, "load_cache_manifest", fail_validation)

    with pytest.raises(PackingCacheInvalidError) as exc_info:
        write_micro_step_cache(
            cache_dir,
            (_micro_step(0),),
            fingerprint=UNIT_FINGERPRINT,
            determinants=UNIT_DETERMINANTS,
        )

    message = str(exc_info.value)
    assert secret_shaped_text not in message
    assert "detail_type=PackingCacheInvalidError" in message


def test_valid_existing_target_is_byte_preserving_hit(tmp_path: Path) -> None:
    cache_dir = _cache_dir(tmp_path)
    first = write_micro_step_cache(
        cache_dir,
        (_micro_step(0),),
        fingerprint=UNIT_FINGERPRINT,
        determinants=UNIT_DETERMINANTS,
    )
    before = _cache_tree_digest(cache_dir)

    hit = write_micro_step_cache(
        cache_dir,
        (_micro_step(99),),
        fingerprint=UNIT_FINGERPRINT,
        determinants=UNIT_DETERMINANTS,
    )

    assert hit == first
    assert _cache_tree_digest(cache_dir) == before
    loaded = load_all_micro_steps_from_cache(
        cache_dir,
        expected_fingerprint=UNIT_FINGERPRINT,
    )
    assert [step.metadata["pack_id"] for step in loaded] == [0]


def test_publication_uses_atomic_no_replace_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = _cache_dir(tmp_path)
    real_install = pack_cache._install_staged_cache_no_replace

    def collide_before_install(stage: Path, target: Path) -> None:
        target.mkdir()
        (target / "collision-marker").write_bytes(b"external-writer")
        real_install(stage, target)

    monkeypatch.setattr(
        pack_cache, "_install_staged_cache_no_replace", collide_before_install
    )

    with pytest.raises(PackingCacheInvalidError) as exc_info:
        write_micro_step_cache(
            cache_dir,
            (_micro_step(0),),
            fingerprint=UNIT_FINGERPRINT,
            determinants=UNIT_DETERMINANTS,
        )

    _assert_collision_message(
        str(exc_info.value),
        target=cache_dir,
        fingerprint=UNIT_FINGERPRINT,
        validation_category="target_already_exists",
    )
    assert (cache_dir / "collision-marker").read_bytes() == b"external-writer"
    assert not list(cache_dir.parent.glob(f".{cache_dir.name}.stage-*"))


def test_publication_revalidates_determinants_immediately_before_install(
    tmp_path: Path,
) -> None:
    cache_dir = _cache_dir(tmp_path)
    changed = _unit_determinants(purpose="changed-during-publication")

    with pytest.raises(PackingCacheInvalidError, match="determinants drifted"):
        _write_micro_step_cache(
            cache_dir,
            (_micro_step(0),),
            cache_root=tmp_path,
            fingerprint=UNIT_FINGERPRINT,
            determinants=UNIT_DETERMINANTS,
            chunk_size=1,
            materialization=build_packing_cache_materialization(workers=1),
            determinant_revalidator=lambda: changed,
            augmentation=DISABLED_AUGMENTATION,
        )

    assert not cache_dir.exists()
    assert not list(cache_dir.parent.glob(f".{cache_dir.name}.stage-*"))


def test_publication_fails_closed_on_hash_matching_forbidden_global_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = _cache_dir(tmp_path)

    class _ForbiddenGlobalPayload:
        def __reduce__(self) -> Any:
            return os.system, ("true",)

    real_dump = pickle.dump

    def stage_chunk_as_forbidden_global(
        _obj: Any, handle: Any, *args: Any, **kwargs: Any
    ) -> None:
        real_dump((_ForbiddenGlobalPayload(),), handle, *args, **kwargs)

    monkeypatch.setattr(pickle, "dump", stage_chunk_as_forbidden_global)

    with pytest.raises(ValueError, match="unreadable"):
        _write_micro_step_cache(
            cache_dir,
            (_micro_step(0),),
            cache_root=tmp_path,
            fingerprint=UNIT_FINGERPRINT,
            determinants=UNIT_DETERMINANTS,
            chunk_size=1,
            materialization=build_packing_cache_materialization(workers=1),
            determinant_revalidator=lambda: UNIT_DETERMINANTS,
            augmentation=DISABLED_AUGMENTATION,
        )

    assert not cache_dir.exists()
    assert not list(cache_dir.parent.glob(f".{cache_dir.name}.stage-*"))


@pytest.mark.parametrize("failure", ("unavailable", errno.ENOSYS, errno.EOPNOTSUPP))
def test_no_replace_unavailable_cleans_stage_and_leaves_target_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str | int,
) -> None:
    cache_dir = _cache_dir(tmp_path)

    class _FakeRenameAt2:
        def __call__(self, *_args: Any) -> int:
            ctypes.set_errno(int(failure))
            return -1

    class _FakeLibc:
        if failure != "unavailable":
            renameat2 = _FakeRenameAt2()

    monkeypatch.setattr(
        pack_cache.ctypes, "CDLL", lambda *_args, **_kwargs: _FakeLibc()
    )

    with pytest.raises(OSError) as exc_info:
        write_micro_step_cache(
            cache_dir,
            (_micro_step(0),),
            fingerprint=UNIT_FINGERPRINT,
            determinants=UNIT_DETERMINANTS,
        )

    expected_errno = errno.ENOSYS if failure == "unavailable" else failure
    assert exc_info.value.errno == expected_errno
    assert not cache_dir.exists()
    assert not list(cache_dir.parent.glob(f".{cache_dir.name}.stage-*"))


def test_cache_writer_leaves_unowned_stage_and_backup_siblings_untouched(
    tmp_path: Path,
) -> None:
    cache_dir = _cache_dir(tmp_path)
    stale_stage = cache_dir.parent / f".{cache_dir.name}.stage-crashed"
    stale_backup = cache_dir.parent / f".{cache_dir.name}.backup-crashed"
    cache_dir.parent.mkdir(parents=True)
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

    assert stale_stage.exists()
    assert stale_backup.exists()
    assert (
        load_cache_manifest(
            cache_dir, expected_fingerprint=UNIT_FINGERPRINT, level="payloads"
        )["status"]
        == "complete"
    )


def test_same_fingerprint_writers_publish_once_then_return_byte_identical_hit(
    tmp_path: Path,
) -> None:
    cache_dir = _cache_dir(tmp_path)
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
        args=(str(cache_dir), True, paused, release, started_a, results),
    )
    writer_b = context.Process(
        target=_overlap_writer_process,
        args=(
            str(cache_dir),
            False,
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
    assert [outcome[0] for outcome in outcomes] == ["ok", "ok"]
    assert sum(bool(outcome[1]) for outcome in outcomes) == 1
    assert {outcome[2] for outcome in outcomes} == {UNIT_FINGERPRINT}
    assert {tuple(outcome[3]) for outcome in outcomes} == {(0, 1, 2, 3)}
    assert len({outcome[4] for outcome in outcomes}) == 1
    assert not list(cache_dir.parent.glob(f".{cache_dir.name}.stage-*"))
    assert not list(cache_dir.parent.glob(f".{cache_dir.name}.backup-*"))
    assert (cache_dir.parent / f".{cache_dir.name}.lock").is_file()


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


def _cache_tree_digest(cache_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(cache_dir.rglob("*")):
        if not path.is_file():
            continue
        digest.update(path.relative_to(cache_dir).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _fingerprint(determinants: dict[str, Any]) -> str:
    return pack_cache.packing_cache_fingerprint_from_determinants(determinants)


def _assert_collision_message(
    message: str,
    *,
    target: Path,
    fingerprint: str,
    validation_category: str,
) -> None:
    assert f"target={target}" in message
    assert f"current_version={pack_cache.PACKING_CACHE_VERSION}" in message
    assert f"fingerprint={fingerprint}" in message
    assert f"validation_category={validation_category}" in message
    assert "automatic_recovery=unavailable" in message


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
    package_versions = {"transformers": "unit-test"}

    def __init__(
        self,
        base_model_path: Path,
        *,
        image_pad_token_id: int = 151655,
    ) -> None:
        base_model_path.mkdir(parents=True, exist_ok=True)
        (base_model_path / "config.json").write_text(
            '{"model_type":"unit-test"}\n', encoding="utf-8"
        )
        self.base_model_path = base_model_path
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
    cache_dir = _cache_dir(tmp_path)
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
    cache_dir = _cache_dir(tmp_path)
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
