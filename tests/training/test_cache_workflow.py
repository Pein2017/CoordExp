"""Cache preparation, admission, hydration, and the fail-before-build gate.

Wave 3 of ``decompose-coordexp-swift-training-orchestration`` moves the
higher-level cache operations out of the training assembly facade.  These tests
own the moved surface directly: one-process preparation and its receipt, worker
resolution, split aggregation, absent-target publication, model-free
fingerprint/admission, rank-local hydration, image-processor attachment, the
bounded actionable failures, and ``--require-all-hit``.

Every cache target these tests touch lives under a pytest temporary directory.
No production cache root is read or written.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import src.prepare_train_cache as prepare_cli
from src.common.errors import RuntimeContractError
from src.qwen import QwenImageEncoding
from src.training import cache_workflow, pack_cache
from src.training.pack_cache import (
    PACKING_CACHE_MATERIALIZATION_STRATEGY,
    PACKING_CACHE_VERSION,
    cache_dir_for_fingerprint,
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

#: Every build-capable callable the verification route must never reach.
BUILD_CAPABLE_CALLABLES = (
    "_resolve_or_build_pack_cache",
    "_resolve_or_build_train_pack_cache",
    "_resolve_eval_pack_cache",
    "build_base_micro_steps",
    "_build_micro_steps_for_dataset",
    "_build_encoded_examples_for_dataset",
    "_encode_examples_with_fork_process_pool",
    "_render_and_encode_example",
    "_materialize_raw_examples_for_dataset",
    "_materialize_pack_plan",
    "write_micro_step_cache",
)


def _synthetic_registry_determinants(
    purpose: str, *, split: str = "train"
) -> dict[str, Any]:
    semantic: dict[str, Any] = {
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


TRAIN_DETERMINANTS = _synthetic_registry_determinants("cache-workflow-train")
EVAL_DETERMINANTS = _synthetic_registry_determinants(
    "cache-workflow-eval", split="eval.forward"
)
TRAIN_FINGERPRINT = str(TRAIN_DETERMINANTS["aggregate_fingerprint"])
EVAL_FINGERPRINT = str(EVAL_DETERMINANTS["aggregate_fingerprint"])


def _micro_step(index: int, *, split: str = "train") -> SupervisedMicroStep:
    return SupervisedMicroStep(
        pack=f"pack-{index}",
        encoded_examples=(f"example-{index}",),
        position_inputs=f"positions-{index}",
        token_sequence=f"tokens-{index}",
        vocab_groups=f"vocab-{index}",
        metadata={
            "split": split,
            "pack_id": index,
            "augmentation_receipt": {**AUGMENTATION_RECEIPT, "split": split},
        },
    )


class _SingleAccelerator:
    num_processes = 1


def _publish(
    cache_root: Path,
    *,
    fingerprint: str,
    determinants: dict[str, Any],
    micro_steps: tuple[SupervisedMicroStep, ...],
) -> Path:
    cache_dir = cache_dir_for_fingerprint(cache_root, fingerprint)
    write_micro_step_cache(
        cache_dir,
        micro_steps,
        cache_root=cache_root,
        fingerprint=fingerprint,
        determinants=determinants,
        materialization=MATERIALIZATION,
        determinant_revalidator=lambda: determinants,
        augmentation=AUGMENTATION_RECEIPT,
    )
    return cache_dir


# ---------------------------------------------------------------------------
# Worker resolution
# ---------------------------------------------------------------------------


def test_worker_resolution_defaults_to_the_declared_materialization_workers() -> None:
    assert cache_workflow._resolve_pack_cache_materialization_workers(None) == (
        pack_cache.DEFAULT_PACK_CACHE_MATERIALIZATION_WORKERS
    )
    assert cache_workflow._resolve_pack_cache_materialization_workers(3) == 3


@pytest.mark.parametrize("workers", [0, -1, True, 1.5, "8", None.__class__])
def test_worker_resolution_rejects_non_positive_or_non_integer_workers(
    workers: Any,
) -> None:
    with pytest.raises(RuntimeContractError) as caught:
        cache_workflow._resolve_pack_cache_materialization_workers(workers)

    assert caught.value.code == "training.pack_cache_workers_invalid"


# ---------------------------------------------------------------------------
# Split aggregation
# ---------------------------------------------------------------------------


def _phase_receipt(status: str, seconds: float) -> dict[str, Any]:
    return {
        "phase_receipt": {
            "cache_preparation": {"status": status, "duration_seconds": seconds}
        }
    }


def test_split_aggregation_reports_not_run_only_when_every_split_hit() -> None:
    hit = _phase_receipt("not_run_cache_hit", 0.0)
    built = _phase_receipt("completed", 0.25)

    assert cache_workflow._aggregate_cache_phase([hit, hit], "cache_preparation") == {
        "status": "not_run",
        "reason": "all_cache_hits",
        "duration_seconds": 0.0,
    }
    assert cache_workflow._aggregate_cache_phase(
        [built, built], "cache_preparation"
    ) == {"status": "completed", "duration_seconds": 0.5}
    assert cache_workflow._aggregate_cache_phase([built, hit], "cache_preparation") == {
        "status": "completed",
        "duration_seconds": 0.25,
        "reason": "mixed_cache_hits_and_builds",
    }


# ---------------------------------------------------------------------------
# Absent-target publication and admission
# ---------------------------------------------------------------------------


def _install_identity(
    monkeypatch: pytest.MonkeyPatch,
    cache_root: Path,
    *,
    fingerprint: str = TRAIN_FINGERPRINT,
    determinants: dict[str, Any] | None = None,
) -> None:
    monkeypatch.setenv("COORDEXP_SWIFT_PACK_CACHE_ROOT", str(cache_root))
    monkeypatch.setattr(
        cache_workflow,
        "build_packing_cache_fingerprint",
        lambda *args, **kwargs: fingerprint,
    )
    monkeypatch.setattr(
        cache_workflow,
        "build_packing_cache_determinants",
        lambda *args, **kwargs: dict(determinants or TRAIN_DETERMINANTS),
    )


def test_absent_target_is_published_then_admitted_in_one_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_root = tmp_path / "cache-root"
    _install_identity(monkeypatch, cache_root)
    workers: list[int] = []

    result = cache_workflow._resolve_or_build_pack_cache(
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        repo_root=tmp_path,
        dataset=SimpleNamespace(),
        split="train",
        accelerator=_SingleAccelerator(),
        rank=0,
        build_micro_steps=lambda count: workers.append(count) or (_micro_step(0),),
        materialization_workers=2,
        verification_level="payloads",
    )

    assert workers == [2]
    assert result["build_status"] == "built"
    assert result["fingerprint"] == TRAIN_FINGERPRINT
    assert result["micro_step_count"] == 1
    assert result["phase_receipt"]["cache_preparation"]["status"] == "completed"
    assert result["phase_receipt"]["cache_publication"]["status"] == "completed"
    assert result["phase_receipt"]["cache_admission"]["status"] == "completed"
    assert Path(result["cache_dir"]).is_relative_to(cache_root)


def test_existing_target_is_admitted_without_a_second_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_root = tmp_path / "cache-root"
    _install_identity(monkeypatch, cache_root)
    _publish(
        cache_root,
        fingerprint=TRAIN_FINGERPRINT,
        determinants=TRAIN_DETERMINANTS,
        micro_steps=(_micro_step(0),),
    )

    result = cache_workflow._resolve_or_build_pack_cache(
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        repo_root=tmp_path,
        dataset=SimpleNamespace(),
        split="train",
        accelerator=_SingleAccelerator(),
        rank=0,
        build_micro_steps=lambda count: (_ for _ in ()).throw(
            AssertionError("a cache hit must not build")
        ),
        materialization_workers=1,
        verification_level="payloads",
    )

    assert result["build_status"] == "hit"
    assert result["phase_receipt"]["cache_preparation"]["status"] == (
        "not_run_cache_hit"
    )


def test_model_free_admission_returns_the_bounded_hit_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_root = tmp_path / "cache-root"
    _install_identity(monkeypatch, cache_root)
    cache_dir = _publish(
        cache_root,
        fingerprint=TRAIN_FINGERPRINT,
        determinants=TRAIN_DETERMINANTS,
        micro_steps=(_micro_step(0), _micro_step(1)),
    )

    admitted = cache_workflow._admit_model_free_pack_cache(
        SimpleNamespace(),
        SimpleNamespace(),
        vocab_groups=SimpleNamespace(),
        dataset=SimpleNamespace(),
        split="train",
        cache_root=cache_root,
        config_path=tmp_path / "config.yaml",
        verification_level="payloads",
        resolved_fingerprint=TRAIN_FINGERPRINT,
    )

    assert admitted["cache_dir"] == cache_dir
    assert admitted["build_status"] == "hit"
    assert admitted["micro_step_count"] == 2
    assert admitted["phase_receipt"]["cache_preparation"]["status"] == (
        "not_run_cache_hit"
    )
    assert admitted["phase_receipt"]["cache_admission"]["verification_level"] == (
        "payloads"
    )


# ---------------------------------------------------------------------------
# Bounded actionable failures
# ---------------------------------------------------------------------------


def test_absent_target_failure_names_the_single_process_preparation_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_root = tmp_path / "cache-root"
    _install_identity(monkeypatch, cache_root)
    config_path = tmp_path / "config.yaml"

    with pytest.raises(RuntimeContractError) as caught:
        cache_workflow._admit_model_free_pack_cache(
            SimpleNamespace(),
            SimpleNamespace(),
            vocab_groups=SimpleNamespace(),
            dataset=SimpleNamespace(),
            split="train",
            cache_root=cache_root,
            config_path=config_path,
            verification_level="manifest",
            resolved_fingerprint=TRAIN_FINGERPRINT,
        )

    context = caught.value.context
    assert caught.value.code == "training.pack_cache_not_prepared"
    assert context["validation_category"] == "expected_target_missing"
    assert context["automatic_recovery"] == "single_process_preparation_required"
    assert context["preparation_argv"] == [
        "python",
        "-m",
        "src.prepare_train_cache",
        "--config",
        str(config_path.expanduser().resolve()),
    ]
    assert "src.prepare_train_cache" in context["preparation_command"]


def test_occupied_invalid_target_fails_closed_without_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_root = tmp_path / "cache-root"
    _install_identity(monkeypatch, cache_root)
    cache_dir = _publish(
        cache_root,
        fingerprint=TRAIN_FINGERPRINT,
        determinants=TRAIN_DETERMINANTS,
        micro_steps=(_micro_step(0),),
    )
    manifest_file = cache_dir / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["version"] = "coordexp-swift-pack-cache-v1"
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeContractError) as caught:
        cache_workflow._admit_model_free_pack_cache(
            SimpleNamespace(),
            SimpleNamespace(),
            vocab_groups=SimpleNamespace(),
            dataset=SimpleNamespace(),
            split="train",
            cache_root=cache_root,
            config_path=tmp_path / "config.yaml",
            verification_level="manifest",
            resolved_fingerprint=TRAIN_FINGERPRINT,
        )

    assert caught.value.code == "training.pack_cache_immutable_collision"
    assert caught.value.context["validation_category"] == "retired_or_unknown_version"
    assert caught.value.context["automatic_recovery"] == "unavailable"


# ---------------------------------------------------------------------------
# Rank-local hydration
# ---------------------------------------------------------------------------


def test_rank_local_eval_hydration_returns_the_canonical_ordinal_assignment(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache-root"
    cache_dir = _publish(
        cache_root,
        fingerprint=EVAL_FINGERPRINT,
        determinants=EVAL_DETERMINANTS,
        micro_steps=tuple(_micro_step(index, split="eval.forward") for index in range(4)),
    )

    micro_steps, reduction_mode, total = (
        cache_workflow._hydrate_eval_micro_steps_from_cache(
            {
                "cache_dir": cache_dir,
                "fingerprint": EVAL_FINGERPRINT,
                "micro_step_count": 4,
            },
            cache_root=cache_root,
            rank=0,
            world_size=1,
            rank_report_gatherer=None,
        )
    )

    assert total == 4
    assert len(micro_steps) == 4
    assert reduction_mode == "replicated"


def test_rank_local_eval_hydration_rejects_a_non_positive_declared_count(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeContractError) as caught:
        cache_workflow._hydrate_eval_micro_steps_from_cache(
            {
                "cache_dir": tmp_path / "absent",
                "fingerprint": EVAL_FINGERPRINT,
                "micro_step_count": 0,
            },
            cache_root=tmp_path,
            rank=0,
            world_size=1,
            rank_report_gatherer=None,
        )

    assert caught.value.code == "training.eval_cache_count_invalid"


# ---------------------------------------------------------------------------
# Image-processor attachment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _EncodedExample:
    example_id: str
    image_encoding: Any


def test_image_processor_attachment_rebinds_every_qwen_image_encoding() -> None:
    encoding = QwenImageEncoding(
        plan=SimpleNamespace(example_id="with-image"),
        pixel_values=None,
        image_grid_thw_tensor=None,
        image_processor=None,
    )
    micro_step = SupervisedMicroStep(
        pack="pack",
        encoded_examples=(
            _EncodedExample("with-image", encoding),
            _EncodedExample("without-image", None),
        ),
        position_inputs="positions",
        token_sequence="tokens",
        vocab_groups="vocab",
        metadata={"pack_id": 0, "augmentation_receipt": dict(AUGMENTATION_RECEIPT)},
    )
    processor = object()

    attached = cache_workflow._attach_image_processors_to_micro_steps(
        (micro_step,), image_processor=processor
    )

    assert attached[0].encoded_examples[0].image_encoding.image_processor is processor
    assert attached[0].encoded_examples[1].image_encoding is None
    assert micro_step.encoded_examples[0].image_encoding.image_processor is None


def test_image_processor_attachment_fails_closed_without_a_runtime_processor() -> None:
    with pytest.raises(RuntimeContractError) as caught:
        cache_workflow._attach_image_processors_to_micro_steps(
            (), image_processor=None
        )

    assert caught.value.code == "training.qwen_image_processor_missing"


# ---------------------------------------------------------------------------
# One-process preparation
# ---------------------------------------------------------------------------


def _install_preparation_stubs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    train_cache: dict[str, Any],
    eval_cache: dict[str, Any] | None,
) -> None:
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
        data=SimpleNamespace(
            train=object(),
            eval=None if eval_cache is None else object(),
            train_order="source_order",
        ),
        template=SimpleNamespace(object_ordering="geo_sorted"),
    )
    resolved = SimpleNamespace(
        config=config,
        fingerprint="config-fingerprint",
        entry_config_path=tmp_path / "config.yaml",
    )
    monkeypatch.setattr(cache_workflow, "load_train_config", lambda path: resolved)
    monkeypatch.setattr(
        cache_workflow,
        "_establish_converged_runtime_determinism",
        lambda *args, **kwargs: {"mode": "legacy"},
    )
    monkeypatch.setattr(
        cache_workflow,
        "_runtime_determinism_run_policy",
        lambda converged, **kwargs: dict(converged),
    )
    monkeypatch.setattr(
        cache_workflow,
        "collect_execution_provenance",
        lambda **kwargs: {"schema_version": 1},
    )
    monkeypatch.setattr(
        cache_workflow,
        "require_pinned_runtime_baseline",
        lambda **kwargs: {"schema_version": 1, "baseline_sha256": "a" * 64},
    )
    monkeypatch.setattr(
        cache_workflow,
        "load_qwen_components",
        lambda config, *, load_model: SimpleNamespace(
            token_identity=object(), tokenizer=object()
        ),
    )
    monkeypatch.setattr(
        cache_workflow, "build_token_vocabulary_groups", lambda *a, **k: object()
    )
    monkeypatch.setattr(
        cache_workflow,
        "_resolve_or_build_train_pack_cache",
        lambda *a, **k: train_cache,
    )
    monkeypatch.setattr(
        cache_workflow, "_resolve_eval_pack_cache", lambda *a, **k: eval_cache
    )
    monkeypatch.setenv("COORDEXP_SWIFT_PACK_CACHE_ROOT", str(tmp_path / "cache-root"))


def _split_cache(root: Path, name: str, *, built: bool) -> dict[str, Any]:
    status = "completed" if built else "not_run_cache_hit"
    return {
        "status": "complete",
        "build_status": "built" if built else "hit",
        "cache_dir": root / name,
        "format_version": PACKING_CACHE_VERSION,
        "fingerprint": TRAIN_FINGERPRINT if name == "train" else EVAL_FINGERPRINT,
        "manifest_path": root / name / "manifest.json",
        "manifest_sha256": "b" * 64,
        "determinants_sha256": "c" * 64,
        "micro_step_count": 3,
        "phase_receipt": {
            "cache_preparation": {"status": status, "duration_seconds": 0.1},
            "cache_publication": {"status": status, "duration_seconds": 0.2},
            "cache_admission": {"status": "completed", "duration_seconds": 0.3},
        },
    }


def test_one_process_preparation_is_model_free_and_covers_both_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "caches"
    _install_preparation_stubs(
        monkeypatch,
        tmp_path,
        train_cache=_split_cache(root, "train", built=True),
        eval_cache=_split_cache(root, "eval", built=False),
    )

    result = cache_workflow.prepare_training_pack_caches(tmp_path / "config.yaml")

    assert result["model_loaded"] is False
    assert result["train"]["build_status"] == "built"
    assert result["eval"]["build_status"] == "hit"
    assert result["policy_identities"]["cache"]["train_fingerprint"] == (
        TRAIN_FINGERPRINT
    )
    assert result["policy_identities"]["cache"]["eval_fingerprint"] == EVAL_FINGERPRINT
    phases = result["measurement"]["phases"]
    assert phases["cache_preparation"]["reason"] == "mixed_cache_hits_and_builds"
    assert result["measurement"]["context"]["wall_clock_scope"] == (
        "prepare_training_pack_caches_entry_to_return"
    )


def test_one_process_preparation_reports_an_absent_eval_split_as_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "caches"
    _install_preparation_stubs(
        monkeypatch,
        tmp_path,
        train_cache=_split_cache(root, "train", built=False),
        eval_cache=None,
    )

    result = cache_workflow.prepare_training_pack_caches(tmp_path / "config.yaml")

    assert result["eval"] is None
    assert result["policy_identities"]["cache"]["eval_fingerprint"] is None
    assert result["measurement"]["phases"]["cache_preparation"] == {
        "status": "not_run",
        "reason": "all_cache_hits",
        "duration_seconds": 0.1,
    }


def test_facade_reexport_delegates_to_the_cache_workflow_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.training import pipeline

    root = tmp_path / "caches"
    _install_preparation_stubs(
        monkeypatch,
        tmp_path,
        train_cache=_split_cache(root, "train", built=True),
        eval_cache=_split_cache(root, "eval", built=True),
    )

    facade = pipeline.prepare_training_pack_caches(tmp_path / "config.yaml")
    owner = cache_workflow.prepare_training_pack_caches(tmp_path / "config.yaml")

    assert set(facade) == set(owner)
    assert facade["train"] == owner["train"]
    assert facade["eval"] == owner["eval"]


# ---------------------------------------------------------------------------
# `--require-all-hit`: fail before any build-capable call
# ---------------------------------------------------------------------------


def _install_build_sentinels(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    reached: list[str] = []

    for name in BUILD_CAPABLE_CALLABLES:

        def sentinel(*args: Any, _name: str = name, **kwargs: Any) -> Any:
            reached.append(_name)
            raise AssertionError(
                f"cache verification reached the build-capable callable {_name}"
            )

        monkeypatch.setattr(cache_workflow, name, sentinel)
    return reached


def _install_verification_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cache_root: Path,
    *,
    with_eval: bool = True,
) -> None:
    _install_preparation_stubs(
        monkeypatch,
        tmp_path,
        train_cache=_split_cache(cache_root, "train", built=True),
        eval_cache=_split_cache(cache_root, "eval", built=True) if with_eval else None,
    )
    monkeypatch.setenv("COORDEXP_SWIFT_PACK_CACHE_ROOT", str(cache_root))
    monkeypatch.setattr(
        cache_workflow,
        "build_packing_cache_fingerprint",
        lambda config, components, *, dataset, split, vocab_groups: (
            TRAIN_FINGERPRINT if split == "train" else EVAL_FINGERPRINT
        ),
    )
    monkeypatch.setattr(
        cache_workflow,
        "build_packing_cache_determinants",
        lambda config, components, *, dataset, split, vocab_groups: dict(
            TRAIN_DETERMINANTS if split == "train" else EVAL_DETERMINANTS
        ),
    )


def test_require_all_hit_validates_two_existing_targets_without_building(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_root = tmp_path / "cache-root"
    train_dir = _publish(
        cache_root,
        fingerprint=TRAIN_FINGERPRINT,
        determinants=TRAIN_DETERMINANTS,
        micro_steps=(_micro_step(0), _micro_step(1)),
    )
    eval_dir = _publish(
        cache_root,
        fingerprint=EVAL_FINGERPRINT,
        determinants=EVAL_DETERMINANTS,
        micro_steps=(_micro_step(0, split="eval.forward"),),
    )
    _install_verification_identity(monkeypatch, tmp_path, cache_root)
    reached = _install_build_sentinels(monkeypatch)
    before = sorted(path.name for path in cache_root.iterdir())

    result = cache_workflow.prepare_training_pack_caches(
        tmp_path / "config.yaml", require_all_hit=True
    )

    assert reached == []
    assert result["cache_materialization_authorized"] is False
    assert result["verification_level"] == "payloads"
    assert result["verified_splits"] == ["train", "eval.forward"]
    assert result["train"]["cache_dir"] == str(train_dir)
    assert result["eval"]["cache_dir"] == str(eval_dir)
    assert result["train"]["build_status"] == "hit"
    assert result["eval"]["build_status"] == "hit"
    assert result["measurement"]["phases"]["cache_preparation"]["status"] == "not_run"
    assert result["measurement"]["phases"]["cache_publication"]["status"] == "not_run"
    assert sorted(path.name for path in cache_root.iterdir()) == before


@pytest.mark.parametrize("absent_split", ["train", "eval.forward"])
def test_require_all_hit_fails_before_any_build_when_a_target_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, absent_split: str
) -> None:
    cache_root = tmp_path / "cache-root"
    cache_root.mkdir()
    if absent_split != "train":
        _publish(
            cache_root,
            fingerprint=TRAIN_FINGERPRINT,
            determinants=TRAIN_DETERMINANTS,
            micro_steps=(_micro_step(0),),
        )
    _install_verification_identity(monkeypatch, tmp_path, cache_root)
    reached = _install_build_sentinels(monkeypatch)
    before = sorted(path.name for path in cache_root.iterdir())

    with pytest.raises(RuntimeContractError) as caught:
        cache_workflow.prepare_training_pack_caches(
            tmp_path / "config.yaml", require_all_hit=True
        )

    assert reached == []
    assert caught.value.code == "training.pack_cache_not_prepared"
    assert caught.value.context["split"] == absent_split
    assert caught.value.context["validation_category"] == "expected_target_missing"
    assert sorted(path.name for path in cache_root.iterdir()) == before


def test_require_all_hit_fails_before_any_build_when_a_target_is_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_root = tmp_path / "cache-root"
    _publish(
        cache_root,
        fingerprint=TRAIN_FINGERPRINT,
        determinants=TRAIN_DETERMINANTS,
        micro_steps=(_micro_step(0),),
    )
    eval_dir = _publish(
        cache_root,
        fingerprint=EVAL_FINGERPRINT,
        determinants=EVAL_DETERMINANTS,
        micro_steps=(_micro_step(0, split="eval.forward"),),
    )
    manifest = json.loads((eval_dir / "manifest.json").read_text(encoding="utf-8"))
    chunk_path = eval_dir / manifest["chunks"][0]["path"]
    chunk_path.write_bytes(chunk_path.read_bytes() + b"corrupt")
    _install_verification_identity(monkeypatch, tmp_path, cache_root)
    reached = _install_build_sentinels(monkeypatch)

    with pytest.raises(RuntimeContractError) as caught:
        cache_workflow.prepare_training_pack_caches(
            tmp_path / "config.yaml", require_all_hit=True
        )

    assert reached == []
    assert caught.value.code == "training.pack_cache_immutable_collision"
    assert caught.value.context["split"] == "eval.forward"
    assert caught.value.context["automatic_recovery"] == "unavailable"


def test_require_all_hit_fails_closed_when_no_eval_split_is_declared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_root = tmp_path / "cache-root"
    _publish(
        cache_root,
        fingerprint=TRAIN_FINGERPRINT,
        determinants=TRAIN_DETERMINANTS,
        micro_steps=(_micro_step(0),),
    )
    _install_verification_identity(monkeypatch, tmp_path, cache_root, with_eval=False)
    reached = _install_build_sentinels(monkeypatch)

    with pytest.raises(RuntimeContractError) as caught:
        cache_workflow.prepare_training_pack_caches(
            tmp_path / "config.yaml", require_all_hit=True
        )

    assert reached == []
    assert caught.value.code == (
        "training.pack_cache_verification_split_undeclared"
    )
    assert caught.value.context["required_splits"] == ["train", "eval.forward"]


def test_cache_preflight_record_exposes_only_the_bounded_verification_bundle(
    tmp_path: Path,
) -> None:
    preflight = cache_workflow.CachePreflight(
        train_fingerprint=TRAIN_FINGERPRINT,
        eval_fingerprint=EVAL_FINGERPRINT,
        train_target=tmp_path / "train",
        eval_target=tmp_path / "eval",
        receipt={"model_loaded": False},
    )

    assert [field for field in preflight.__dataclass_fields__] == [
        "train_fingerprint",
        "eval_fingerprint",
        "train_target",
        "eval_target",
        "receipt",
    ]
    assert preflight.to_receipt_dict() == {"model_loaded": False}
    with pytest.raises(Exception):
        preflight.train_fingerprint = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def test_cli_forwards_require_all_hit_and_names_a_verification_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[bool] = []

    def fake(path: Path, *, require_all_hit: bool = False) -> dict[str, Any]:
        calls.append(require_all_hit)
        return {"model_loaded": False, "cache_materialization_authorized": False}

    monkeypatch.setattr(prepare_cli, "prepare_training_pack_caches", fake)
    receipt_path = tmp_path / "verification.json"

    assert (
        prepare_cli.main(
            [
                "--config",
                str(tmp_path / "config.yaml"),
                "--receipt",
                str(receipt_path),
                "--require-all-hit",
            ]
        )
        == 0
    )

    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert calls == [True]
    assert payload["schema"] == "coordexp-swift-pack-cache-verification-receipt-v1"
    assert payload["terminal_status"] == "completed"
    assert payload["result"]["cache_materialization_authorized"] is False


def test_cli_default_route_remains_the_only_build_capable_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[bool] = []

    def fake(path: Path, *, require_all_hit: bool = False) -> dict[str, Any]:
        calls.append(require_all_hit)
        return {"model_loaded": False}

    monkeypatch.setattr(prepare_cli, "prepare_training_pack_caches", fake)
    receipt_path = tmp_path / "preparation.json"

    assert (
        prepare_cli.main(
            ["--config", str(tmp_path / "config.yaml"), "--receipt", str(receipt_path)]
        )
        == 0
    )

    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert calls == [False]
    assert payload["schema"] == "coordexp-swift-pack-cache-preparation-receipt-v1"


def test_cli_publishes_a_bounded_failed_verification_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(path: Path, *, require_all_hit: bool = False) -> dict[str, Any]:
        raise RuntimeContractError(
            "secret-shaped diagnostic",
            code="training.pack_cache_not_prepared",
        )

    monkeypatch.setattr(prepare_cli, "prepare_training_pack_caches", fail)
    receipt_path = tmp_path / "failed.json"

    with pytest.raises(RuntimeContractError):
        prepare_cli.main(
            [
                "--config",
                str(tmp_path / "config.yaml"),
                "--receipt",
                str(receipt_path),
                "--require-all-hit",
            ]
        )

    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "coordexp-swift-pack-cache-verification-receipt-v1"
    assert payload["terminal_status"] == "failed"
    assert payload["result"] is None
    assert payload["failure"]["error_code"] == "training.pack_cache_not_prepared"
    assert "secret-shaped" not in receipt_path.read_text(encoding="utf-8")
