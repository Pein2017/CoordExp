from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import pickle
from typing import Any

import pytest

from src.config.loader import load_train_config
from src.config.models import RuntimeBatchResolution
from src.training.pipeline import build_repeating_micro_step_stream
from src.training.pack_cache import (
    DEFAULT_PACK_CACHE_MATERIALIZATION_WORKERS,
    PACKING_CACHE_MATERIALIZATION_STRATEGY,
    build_packing_cache_determinants,
    build_packing_cache_fingerprint,
    load_all_micro_steps_from_cache,
    load_cache_manifest,
    load_rank_micro_steps_from_cache,
    write_micro_step_cache,
)
from src.training.schedule import ResolvedStepSchedule
from src.training.supervised_trainer import SupervisedMicroStep


FIXTURE_CONFIG = Path("tests/fixtures/smoke/qwen3_vl_single_image_pack/config.yaml")


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


def test_packing_cache_fingerprint_tracks_jsonl_content_not_only_stat(
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

    default_cache = tmp_path / "default-cache"
    override_cache = tmp_path / "override-cache"
    default_manifest = write_micro_step_cache(
        default_cache,
        (_micro_step(0),),
        fingerprint=fingerprint,
        determinants={"purpose": "unit-test"},
    )
    override_manifest = write_micro_step_cache(
        override_cache,
        (_micro_step(0),),
        fingerprint=fingerprint,
        determinants={"purpose": "unit-test"},
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
        fingerprint="abc123",
        determinants={"purpose": "unit-test"},
    )

    assert manifest["materialization"] == {
        "strategy": PACKING_CACHE_MATERIALIZATION_STRATEGY,
        "workers": DEFAULT_PACK_CACHE_MATERIALIZATION_WORKERS,
    }
    assert load_cache_manifest(cache_dir)["materialization"] == manifest["materialization"]


def test_micro_step_cache_manifest_records_explicit_materialization_override(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"

    manifest = write_micro_step_cache(
        cache_dir,
        (_micro_step(0),),
        fingerprint="abc123",
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


def test_legacy_micro_step_cache_manifest_without_materialization_still_loads(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    write_micro_step_cache(
        cache_dir,
        (_micro_step(0),),
        fingerprint="abc123",
        determinants={"purpose": "unit-test"},
    )
    manifest_path = cache_dir / "manifest.json"
    legacy_manifest = json.loads(manifest_path.read_text())
    legacy_manifest.pop("materialization")
    manifest_path.write_text(json.dumps(legacy_manifest), encoding="utf-8")

    loaded_manifest = load_cache_manifest(cache_dir)
    loaded_steps = load_all_micro_steps_from_cache(cache_dir)

    assert "materialization" not in loaded_manifest
    assert [step.metadata["pack_id"] for step in loaded_steps] == [0]


def test_micro_step_cache_loads_exact_rank_local_training_order(tmp_path: Path) -> None:
    micro_steps = tuple(_micro_step(index) for index in range(10))
    cache_dir = tmp_path / "cache"
    manifest = write_micro_step_cache(
        cache_dir,
        micro_steps,
        fingerprint="abc123",
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
        schedule=schedule,
        rank=0,
        world_size=2,
    )
    rank1 = load_rank_micro_steps_from_cache(
        cache_dir,
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
        fingerprint="abc123",
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
        fingerprint="abc123",
        determinants={"purpose": "unit-test", "split": "eval.forward"},
        chunk_size=3,
    )

    loaded = load_all_micro_steps_from_cache(cache_dir)

    assert [step.metadata["pack_id"] for step in loaded] == list(range(7))


def test_cached_rank_local_steps_must_not_be_sharded_again(tmp_path: Path) -> None:
    micro_steps = tuple(_micro_step(index) for index in range(2048))
    cache_dir = tmp_path / "cache"
    write_micro_step_cache(
        cache_dir,
        micro_steps,
        fingerprint="abc123",
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
        fingerprint="abc123",
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
        fingerprint="abc123",
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
            schedule=schedule,
            rank=0,
            world_size=1,
        )


def _micro_step(index: int) -> SupervisedMicroStep:
    return SupervisedMicroStep(
        pack=f"pack-{index}",
        encoded_examples=(f"example-{index}",),
        position_inputs=f"positions-{index}",
        token_sequence=f"tokens-{index}",
        vocab_groups=f"vocab-{index}",
        metadata={"pack_id": index},
    )


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
