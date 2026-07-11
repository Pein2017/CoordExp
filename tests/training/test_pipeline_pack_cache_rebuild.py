from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from threading import Condition, Thread
from types import SimpleNamespace

import pytest

from src.artifacts.run_writer import RunWriter
from src.training import pipeline
from src.training.pack_cache import (
    PACKING_CACHE_MATERIALIZATION_STRATEGY,
    PACKING_CACHE_VERSION,
    load_all_micro_steps_from_cache,
    load_rank_micro_steps_from_cache,
    write_micro_step_cache,
)
from src.training.supervised_trainer import SupervisedMicroStep


DETERMINANTS = {"purpose": "resolver-integration"}
AUGMENTATION_RECEIPT = {
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
MATERIALIZATION = {
    "strategy": PACKING_CACHE_MATERIALIZATION_STRATEGY,
    "workers": 1,
}
FINGERPRINT = hashlib.sha256(
    json.dumps(
        DETERMINANTS,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
).hexdigest()


def _micro_step(index: int) -> SupervisedMicroStep:
    return SupervisedMicroStep(
        pack=f"pack-{index}",
        encoded_examples=(f"example-{index}",),
        position_inputs=f"positions-{index}",
        token_sequence=f"tokens-{index}",
        vocab_groups=f"vocab-{index}",
        metadata={
            "pack_id": index,
            "augmentation_receipt": dict(AUGMENTATION_RECEIPT),
        },
    )


class _SingleAccelerator:
    num_processes = 1


class _SharedCollective:
    def __init__(self) -> None:
        self.condition = Condition()
        self.value: object | None = None

    def accelerator(self, *, rank: int) -> SimpleNamespace:
        def broadcast(values: list[object], from_process: int = 0) -> None:
            with self.condition:
                if rank == from_process:
                    self.value = values[0]
                    self.condition.notify_all()
                else:
                    self.condition.wait_for(lambda: self.value is not None, timeout=2.0)
                    values[0] = self.value

        return SimpleNamespace(
            num_processes=2,
            process_index=rank,
            broadcast_object_list=broadcast,
        )


def _install_resolver_identity(
    monkeypatch: pytest.MonkeyPatch, cache_root: Path
) -> None:
    monkeypatch.setenv("COORDEXP_SWIFT_PACK_CACHE_ROOT", str(cache_root))
    monkeypatch.setattr(
        pipeline, "build_packing_cache_fingerprint", lambda *args, **kwargs: FINGERPRINT
    )
    monkeypatch.setattr(
        pipeline,
        "build_packing_cache_determinants",
        lambda *args, **kwargs: dict(DETERMINANTS),
    )


def _resolve(
    tmp_path: Path,
    *,
    rank: int,
    build_micro_steps: object,
    accelerator: object | None = None,
) -> dict[str, object]:
    return pipeline._resolve_or_build_pack_cache(
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        repo_root=tmp_path,
        dataset=SimpleNamespace(),
        split="train",
        accelerator=accelerator or _SingleAccelerator(),
        rank=rank,
        build_micro_steps=build_micro_steps,
        materialization_workers=1,
    )


@pytest.mark.parametrize("damage", ["old_version", "checksum_corruption"])
def test_rank_zero_rebuilds_invalid_cache_and_subsequent_all_read_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    cache_root = tmp_path / "cache-root"
    _install_resolver_identity(monkeypatch, cache_root)
    cache_dir = cache_root / FINGERPRINT
    write_micro_step_cache(
        cache_dir,
        (_micro_step(0),),
        fingerprint=FINGERPRINT,
        determinants=DETERMINANTS,
        materialization=MATERIALIZATION,
        augmentation=AUGMENTATION_RECEIPT,
    )
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if damage == "old_version":
        manifest["version"] = "coordexp-swift-pack-cache-v1"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    else:
        chunk_path = cache_dir / manifest["chunks"][0]["path"]
        chunk_path.write_bytes(chunk_path.read_bytes() + b"corrupt")

    builds: list[int] = []
    result = _resolve(
        tmp_path,
        rank=0,
        build_micro_steps=lambda workers: builds.append(workers)
        or (_micro_step(10), _micro_step(11)),
    )

    assert builds == [1]
    assert result["build_status"] == "built"
    assert result["format_version"] == PACKING_CACHE_VERSION
    loaded = load_all_micro_steps_from_cache(
        result["cache_dir"], expected_fingerprint=FINGERPRINT
    )
    assert [step.metadata["pack_id"] for step in loaded] == [10, 11]
    rank_loaded = load_rank_micro_steps_from_cache(
        result["cache_dir"],
        expected_fingerprint=FINGERPRINT,
        schedule=SimpleNamespace(
            resolved_max_steps=1,
            runtime_batch=SimpleNamespace(
                world_size=1,
                resolved_grad_accum_steps=1,
                effective_batch_size=1,
            ),
        ),
        rank=0,
        world_size=1,
    )
    assert [step.metadata["pack_id"] for step in rank_loaded] == [10]
    assert not list(cache_root.glob(f".{FINGERPRINT}.stage-*"))
    assert not list(cache_root.glob(f".{FINGERPRINT}.backup-*"))


def test_peer_blocks_on_shared_success_then_strict_reads_published_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_root = tmp_path / "cache-root"
    _install_resolver_identity(monkeypatch, cache_root)
    collective = _SharedCollective()
    peer_result: dict[str, object] = {}
    peer_errors: list[BaseException] = []

    def run_peer() -> None:
        try:
            peer_result.update(
                _resolve(
                    tmp_path,
                    rank=1,
                    accelerator=collective.accelerator(rank=1),
                    build_micro_steps=lambda workers: (_ for _ in ()).throw(
                        AssertionError("peer must not build")
                    ),
                )
            )
        except BaseException as exc:
            peer_errors.append(exc)

    peer = Thread(target=run_peer)
    peer.start()
    assert peer.is_alive()
    main_result = _resolve(
        tmp_path,
        rank=0,
        accelerator=collective.accelerator(rank=0),
        build_micro_steps=lambda workers: (_micro_step(20),),
    )
    peer.join(timeout=2.0)

    assert not peer.is_alive()
    assert peer_errors == []
    assert main_result["build_status"] == "built"
    assert peer_result["build_status"] == "waited"
    assert peer_result["manifest_sha256"] == main_result["manifest_sha256"]
    loaded = load_all_micro_steps_from_cache(
        peer_result["cache_dir"], expected_fingerprint=FINGERPRINT
    )
    assert [step.metadata["pack_id"] for step in loaded] == [20]


def test_rank_zero_build_failure_is_broadcast_as_identical_named_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_root = tmp_path / "cache-root"
    _install_resolver_identity(monkeypatch, cache_root)
    collective = _SharedCollective()
    errors: dict[int, BaseException] = {}

    def resolve_rank(rank: int) -> None:
        try:
            _resolve(
                tmp_path,
                rank=rank,
                accelerator=collective.accelerator(rank=rank),
                build_micro_steps=(
                    lambda workers: (_ for _ in ()).throw(OSError("disk full"))
                    if rank == 0
                    else lambda workers: (_ for _ in ()).throw(
                        AssertionError("peer must not build")
                    )
                ),
            )
        except BaseException as exc:
            errors[rank] = exc

    peer = Thread(target=resolve_rank, args=(1,))
    peer.start()
    resolve_rank(0)
    peer.join(timeout=2.0)

    assert not peer.is_alive()
    assert set(errors) == {0, 1}
    assert all(
        getattr(error, "code", None) == "training.pack_cache_resolution_failed"
        for error in errors.values()
    )
    assert str(errors[0]) == str(errors[1])


def test_shared_cache_hit_descriptor_releases_peer_for_strict_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_root = tmp_path / "cache-root"
    _install_resolver_identity(monkeypatch, cache_root)
    write_micro_step_cache(
        cache_root / FINGERPRINT,
        (_micro_step(25),),
        fingerprint=FINGERPRINT,
        determinants=DETERMINANTS,
        materialization=MATERIALIZATION,
        augmentation=AUGMENTATION_RECEIPT,
    )
    collective = _SharedCollective()
    results: dict[int, dict[str, object]] = {}

    def resolve_rank(rank: int) -> None:
        results[rank] = _resolve(
            tmp_path,
            rank=rank,
            accelerator=collective.accelerator(rank=rank),
            build_micro_steps=lambda workers: (_ for _ in ()).throw(
                AssertionError("cache hit must not build")
            ),
        )

    peer = Thread(target=resolve_rank, args=(1,))
    peer.start()
    resolve_rank(0)
    peer.join(timeout=2.0)

    assert not peer.is_alive()
    assert results[0]["build_status"] == "hit"
    assert results[1]["build_status"] == "waited"
    assert results[0]["manifest_sha256"] == results[1]["manifest_sha256"]


def test_rank_zero_cache_hit_reuses_its_single_strict_manifest_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_root = tmp_path / "cache-root"
    _install_resolver_identity(monkeypatch, cache_root)
    write_micro_step_cache(
        cache_root / FINGERPRINT,
        (_micro_step(30),),
        fingerprint=FINGERPRINT,
        determinants=DETERMINANTS,
        materialization=MATERIALIZATION,
        augmentation=AUGMENTATION_RECEIPT,
    )
    real_load = pipeline.load_cache_manifest
    successful_reads: list[Path] = []

    def counting_load(
        cache_dir: Path, *, expected_fingerprint: str
    ) -> dict[str, object]:
        manifest = real_load(cache_dir, expected_fingerprint=expected_fingerprint)
        successful_reads.append(Path(cache_dir))
        return manifest

    monkeypatch.setattr(pipeline, "load_cache_manifest", counting_load)
    result = _resolve(
        tmp_path,
        rank=0,
        build_micro_steps=lambda workers: (_ for _ in ()).throw(
            AssertionError("cache hit must not build")
        ),
    )

    assert result["build_status"] == "hit"
    assert successful_reads == [cache_root / FINGERPRINT]


def test_resolver_binding_is_self_contained_after_physical_cache_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_root = tmp_path / "cache-root"
    _install_resolver_identity(monkeypatch, cache_root)
    resolved_cache = _resolve(
        tmp_path,
        rank=0,
        build_micro_steps=lambda workers: (_micro_step(40),),
    )
    writer = RunWriter.initialize(
        run_dir=tmp_path / "run",
        run_id="run",
        run_name="integration",
        artifact_root=tmp_path,
        collision_outcome="created",
        created_at="now",
        config_fingerprint="config-fingerprint",
        resolved_config={},
        world_size=1,
    )

    pipeline._bind_cache_materialization(writer, "train", resolved_cache)
    expected_binding = {
        "cache_format_version": PACKING_CACHE_VERSION,
        "semantic_fingerprint": FINGERPRINT,
        "determinant_digest": resolved_cache["determinants_sha256"],
    }
    assert writer.read_run()["materializations"]["train"] == expected_binding
    assert not set(expected_binding).intersection(
        {"cache", "cache_path", "manifest_path", "manifest_sha256", "chunk_sha256s"}
    )

    shutil.rmtree(cache_root)

    assert not cache_root.exists()
    assert writer.read_run()["materializations"]["train"] == expected_binding


def test_same_dataset_train_and_eval_resolve_distinct_role_materializations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_root = tmp_path / "cache-root"
    monkeypatch.setenv("COORDEXP_SWIFT_PACK_CACHE_ROOT", str(cache_root))
    fingerprints: dict[str, str] = {}

    def determinants(*args: object, split: str, **kwargs: object) -> dict[str, object]:
        return {"purpose": "resolver-role-integration", "split": split}

    def fingerprint(*args: object, split: str, **kwargs: object) -> str:
        payload = determinants(split=split)
        digest = hashlib.sha256(
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        fingerprints[split] = digest
        return digest

    monkeypatch.setattr(pipeline, "build_packing_cache_determinants", determinants)
    monkeypatch.setattr(pipeline, "build_packing_cache_fingerprint", fingerprint)
    train_receipt = {
        **AUGMENTATION_RECEIPT,
        "mode": "static_stochastic_view",
        "enabled": True,
        "policy_version": "geometry-flips-v1",
        "horizontal_prob": 0.5,
        "vertical_prob": 0.2,
        "transform_counts": {"identity": 1, "hflip": 0, "vflip": 0, "hvflip": 0},
        "random_object_order_presentations": [],
    }
    eval_receipt = {**AUGMENTATION_RECEIPT, "split": "eval.forward"}

    def step_with_receipt(
        index: int, receipt: dict[str, object]
    ) -> SupervisedMicroStep:
        step = _micro_step(index)
        return SupervisedMicroStep(
            **{
                **step.__dict__,
                "metadata": {"pack_id": index, "augmentation_receipt": receipt},
            }
        )

    monkeypatch.setattr(
        pipeline,
        "build_base_micro_steps",
        lambda *args, **kwargs: (step_with_receipt(50, train_receipt),),
    )
    monkeypatch.setattr(
        pipeline,
        "_build_micro_steps_for_dataset",
        lambda *args, **kwargs: (step_with_receipt(60, eval_receipt),),
    )
    shared_dataset = SimpleNamespace(path=tmp_path / "shared.jsonl", sample_limit=1)
    config = SimpleNamespace(
        data=SimpleNamespace(train=shared_dataset, eval=shared_dataset)
    )
    accelerator = _SingleAccelerator()

    train = pipeline._resolve_or_build_train_pack_cache(
        config,
        SimpleNamespace(),
        SimpleNamespace(),
        repo_root=tmp_path,
        accelerator=accelerator,
    )
    evaluated = pipeline._resolve_eval_pack_cache(
        config,
        SimpleNamespace(),
        SimpleNamespace(),
        repo_root=tmp_path,
        accelerator=accelerator,
    )

    assert evaluated is not None
    assert train["fingerprint"] == fingerprints["train"]
    assert evaluated["fingerprint"] == fingerprints["eval.forward"]
    assert train["fingerprint"] != evaluated["fingerprint"]
    assert train["augmentation"]["enabled"] is True
    assert evaluated["augmentation"]["enabled"] is False
    assert evaluated["augmentation"]["split"] == "eval.forward"
