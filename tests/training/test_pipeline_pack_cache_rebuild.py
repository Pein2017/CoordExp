from __future__ import annotations

import json
from pathlib import Path
import shutil
from threading import Condition, Thread
from types import SimpleNamespace

import pytest

from src.artifacts.run_writer import RunWriter
from src.common.errors import RuntimeContractError
from src.training import cache_workflow, pack_cache
from src.training.pack_cache import (
    PACKING_CACHE_MATERIALIZATION_STRATEGY,
    PACKING_CACHE_VERSION,
    cache_dir_for_fingerprint,
    load_all_micro_steps_from_cache,
    write_micro_step_cache,
)
from src.training.supervised_trainer import SupervisedMicroStep


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


def _synthetic_registry_determinants(
    purpose: str, *, split: str = "train"
) -> dict[str, object]:
    semantic: dict[str, object] = {
        "version": PACKING_CACHE_VERSION,
        "split": split,
        "dataset": {"purpose": purpose},
        "template": {"purpose": purpose},
        "packing": {"purpose": purpose},
        "processor": {"purpose": purpose},
        "ordering": {"purpose": purpose},
        "augmentation": {**AUGMENTATION_RECEIPT, "split": split},
        "qwen": {
            "processor_identity": {"purpose": purpose},
            "token_identity": {"purpose": purpose},
            "encoding_identity": {"purpose": purpose},
            "model_config_assets": {"purpose": purpose},
            "processor_assets": {"purpose": purpose},
            "tokenizer_assets": {"purpose": purpose},
        },
        "realized_vocab_groups": {"purpose": purpose},
        "micro_step_runtime_config": {
            "fa2_model_dtype": "no",
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


DETERMINANTS = _synthetic_registry_determinants("resolver-integration")
FINGERPRINT = str(DETERMINANTS["aggregate_fingerprint"])


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
        cache_workflow, "build_packing_cache_fingerprint", lambda *args, **kwargs: FINGERPRINT
    )
    monkeypatch.setattr(
        cache_workflow,
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
    return cache_workflow._resolve_or_build_pack_cache(
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
        verification_level="payloads",
    )


@pytest.mark.parametrize("damage", ["old_version", "checksum_corruption"])
def test_rank_zero_rejects_occupied_invalid_cache_without_rebuilding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    cache_root = tmp_path / "cache-root"
    _install_resolver_identity(monkeypatch, cache_root)
    cache_dir = cache_dir_for_fingerprint(cache_root, FINGERPRINT)
    write_micro_step_cache(
        cache_dir,
        (_micro_step(0),),
        cache_root=cache_root,
        fingerprint=FINGERPRINT,
        determinants=DETERMINANTS,
        materialization=MATERIALIZATION,
        determinant_revalidator=lambda: DETERMINANTS,
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
    before = {
        path.relative_to(cache_dir): path.read_bytes()
        for path in cache_dir.rglob("*")
        if path.is_file()
    }

    builds: list[int] = []
    with pytest.raises(RuntimeContractError) as exc_info:
        _resolve(
            tmp_path,
            rank=0,
            build_micro_steps=lambda workers: builds.append(workers)
            or (_micro_step(10), _micro_step(11)),
        )

    assert exc_info.value.code == "training.pack_cache_immutable_collision"
    assert exc_info.value.context["cache_root"] == str(cache_root)
    assert exc_info.value.context["expected_cache_target"] == str(cache_dir)
    assert exc_info.value.context["cache_version"] == PACKING_CACHE_VERSION
    assert exc_info.value.context["fingerprint"] == FINGERPRINT
    assert exc_info.value.context["automatic_recovery"] == "unavailable"
    assert "preparation_command" not in exc_info.value.context
    assert builds == []
    after = {
        path.relative_to(cache_dir): path.read_bytes()
        for path in cache_dir.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_preparation_publishes_cache_before_distributed_peer_strict_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_root = tmp_path / "cache-root"
    _install_resolver_identity(monkeypatch, cache_root)
    collective = _SharedCollective()
    main_result = _resolve(
        tmp_path,
        rank=0,
        build_micro_steps=lambda workers: (_micro_step(20),),
    )
    peer_result = _resolve(
        tmp_path,
        rank=1,
        accelerator=collective.accelerator(rank=1),
        build_micro_steps=lambda workers: (_ for _ in ()).throw(
            AssertionError("peer must not build")
        ),
    )

    assert main_result["build_status"] == "built"
    assert peer_result["build_status"] == "waited"
    assert peer_result["manifest_sha256"] == main_result["manifest_sha256"]
    loaded = load_all_micro_steps_from_cache(
        peer_result["cache_dir"],
        cache_root=tmp_path / "cache-root",
        expected_fingerprint=FINGERPRINT,
    )
    assert [step.metadata["pack_id"] for step in loaded] == [20]


def test_preparation_revalidates_determinants_after_build_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root = tmp_path / "cache-root"
    monkeypatch.setenv("COORDEXP_SWIFT_PACK_CACHE_ROOT", str(cache_root))
    determinant_state = {"purpose": "before-build"}

    def determinants(*args: object, **kwargs: object) -> dict[str, object]:
        return _synthetic_registry_determinants(determinant_state["purpose"])

    monkeypatch.setattr(cache_workflow, "build_packing_cache_determinants", determinants)
    initial_determinants = determinants()
    initial_fingerprint = str(initial_determinants["aggregate_fingerprint"])
    canonical_target = cache_dir_for_fingerprint(cache_root, initial_fingerprint)

    def mutate_determinant_during_build(
        workers: int,
    ) -> tuple[SupervisedMicroStep, ...]:
        assert workers == 1
        determinant_state["purpose"] = "dataset-image-or-asset-mutated-during-build"
        return (_micro_step(21),)

    with pytest.raises(RuntimeContractError) as exc_info:
        _resolve(
            tmp_path,
            rank=0,
            build_micro_steps=mutate_determinant_during_build,
        )

    assert exc_info.value.code == "training.pack_cache_resolution_failed"
    assert not canonical_target.exists()
    assert not list(canonical_target.parent.glob(f".{canonical_target.name}.stage-*"))


def test_distributed_cache_miss_fails_fast_with_preparation_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_root = tmp_path / "cache-root"
    _install_resolver_identity(monkeypatch, cache_root)
    collective = _SharedCollective()
    for rank in (0, 1):
        with pytest.raises(RuntimeContractError) as exc_info:
            _resolve(
                tmp_path,
                rank=rank,
                accelerator=collective.accelerator(rank=rank),
                build_micro_steps=lambda workers: (_ for _ in ()).throw(
                    AssertionError("distributed rank must not build")
                ),
            )
        assert exc_info.value.code == "training.pack_cache_not_prepared"
        assert "python -m src.prepare_train_cache" in str(exc_info.value)


def test_shared_cache_hit_descriptor_releases_peer_for_strict_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_root = tmp_path / "cache-root"
    _install_resolver_identity(monkeypatch, cache_root)
    write_micro_step_cache(
        cache_dir_for_fingerprint(cache_root, FINGERPRINT),
        (_micro_step(25),),
        cache_root=cache_root,
        fingerprint=FINGERPRINT,
        determinants=DETERMINANTS,
        materialization=MATERIALIZATION,
        determinant_revalidator=lambda: DETERMINANTS,
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
        cache_dir_for_fingerprint(cache_root, FINGERPRINT),
        (_micro_step(30),),
        cache_root=cache_root,
        fingerprint=FINGERPRINT,
        determinants=DETERMINANTS,
        materialization=MATERIALIZATION,
        determinant_revalidator=lambda: DETERMINANTS,
        augmentation=AUGMENTATION_RECEIPT,
    )
    real_load = cache_workflow.load_cache_manifest
    successful_reads: list[Path] = []

    def counting_load(
        cache_dir: Path,
        *,
        cache_root: Path,
        expected_fingerprint: str,
        level: str,
    ) -> dict[str, object]:
        manifest = real_load(
            cache_dir,
            cache_root=cache_root,
            expected_fingerprint=expected_fingerprint,
            level=level,
        )
        successful_reads.append(Path(cache_dir))
        return manifest

    monkeypatch.setattr(cache_workflow, "load_cache_manifest", counting_load)
    result = _resolve(
        tmp_path,
        rank=0,
        build_micro_steps=lambda workers: (_ for _ in ()).throw(
            AssertionError("cache hit must not build")
        ),
    )

    assert result["build_status"] == "hit"
    assert successful_reads == [cache_dir_for_fingerprint(cache_root, FINGERPRINT)]


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

    cache_workflow._bind_cache_materialization(writer, "train", resolved_cache)
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
        payload = _synthetic_registry_determinants(
            "resolver-role-integration", split=split
        )
        fingerprints[split] = str(payload["aggregate_fingerprint"])
        return payload

    monkeypatch.setattr(cache_workflow, "build_packing_cache_determinants", determinants)
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
        cache_workflow,
        "build_base_micro_steps",
        lambda *args, **kwargs: (step_with_receipt(50, train_receipt),),
    )
    monkeypatch.setattr(
        cache_workflow,
        "_build_micro_steps_for_dataset",
        lambda *args, **kwargs: (step_with_receipt(60, eval_receipt),),
    )
    shared_dataset = SimpleNamespace(path=tmp_path / "shared.jsonl", sample_limit=1)
    config = SimpleNamespace(
        data=SimpleNamespace(train=shared_dataset, eval=shared_dataset)
    )
    accelerator = _SingleAccelerator()

    train = cache_workflow._resolve_or_build_train_pack_cache(
        config,
        SimpleNamespace(),
        SimpleNamespace(),
        repo_root=tmp_path,
        accelerator=accelerator,
        verification_level="payloads",
    )
    evaluated = cache_workflow._resolve_eval_pack_cache(
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
