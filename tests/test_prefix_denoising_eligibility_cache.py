from __future__ import annotations

from types import SimpleNamespace

import pytest

import src.detection.runtime as runtime_mod
import src.detection.prefix_denoising.dataset as prefix_dataset_mod
from src.config.schema import PrefixDenoisingConfig
from src.detection.prefix_denoising.dataset import PrefixDenoisingTrainingDataset
from src.detection.prefix_denoising.types import (
    HybridPrefixDenoisingSample,
    PrefixDenoisingPackingEstimate,
    PrefixDenoisingSegment,
)


def _rows(count: int = 2) -> list[dict[str, object]]:
    return [
        {
            "images": [f"image_{idx}.jpg"],
            "objects": [{"bbox_2d": [10, 20, 30, 40], "label": "thing"}],
        }
        for idx in range(count)
    ]


def _segment(segment_id: str, branch_id: str, *, length: int) -> PrefixDenoisingSegment:
    input_ids = tuple(range(length))
    return PrefixDenoisingSegment(
        segment_id=segment_id,
        branch_id=branch_id,  # type: ignore[arg-type]
        input_ids=input_ids,
        labels=tuple(-100 if idx == 0 else token for idx, token in enumerate(input_ids)),
        attention_mask=tuple(1 for _ in input_ids),
        supervised_positions=tuple(range(1, length)),
        ce_denominator=max(length - 1, 0),
    )


def _ok_sample(base_sample_id: str, *, total_length: int) -> HybridPrefixDenoisingSample:
    clean_len = total_length // 2
    noisy_len = total_length - clean_len
    clean = _segment(f"{base_sample_id}:clean", "clean_full", length=clean_len)
    noisy = _segment(f"{base_sample_id}:noisy", "noisy_full", length=noisy_len)
    return HybridPrefixDenoisingSample(
        ok=True,
        hybrid_sample_id=f"{base_sample_id}:hybrid",
        base_sample_id=base_sample_id,
        clean_full=clean,
        noisy_full=noisy,
        kl_sites=(),
    )


def test_prefix_denoising_eligibility_cache_rank0_writes_rank1_loads(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def _estimate_sample(*_args: object, **_kwargs: object):
        calls.append(f"estimate-{len(calls)}")
        return PrefixDenoisingPackingEstimate(ok=True, total_length=10 + len(calls))

    monkeypatch.setattr(
        prefix_dataset_mod,
        "estimate_hybrid_prefix_denoising_packing",
        _estimate_sample,
    )
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    cache_root = tmp_path / "eligibility"

    rank0 = PrefixDenoisingTrainingDataset(
        _rows(2),
        swift_template=SimpleNamespace(tokenizer=object()),
        image_root=tmp_path,
        user_prompt="detect",
        system_prompt=None,
        prefix_denoising=PrefixDenoisingConfig(enabled=True),
        max_length=12000,
        dataset_name="unit",
        seed=17,
        eligibility_cache_dir=cache_root,
        eligibility_cache_fingerprint={"case": "ranked-cache"},
        eligibility_cache_wait_timeout_s=0.1,
        eligibility_precompute_workers=1,
    )

    assert len(rank0) == 2
    assert calls == ["estimate-0", "estimate-1"]
    assert list(cache_root.glob("*/eligibility.json"))
    assert list(cache_root.glob("*/progress.json"))

    def _unexpected_build(*_args: object, **_kwargs: object):
        raise AssertionError("rank 1 should load eligibility cache without rebuilding")

    monkeypatch.setattr(
        prefix_dataset_mod,
        "estimate_hybrid_prefix_denoising_packing",
        _unexpected_build,
    )
    monkeypatch.setenv("RANK", "1")

    rank1 = PrefixDenoisingTrainingDataset(
        _rows(2),
        swift_template=SimpleNamespace(tokenizer=object()),
        image_root=tmp_path,
        user_prompt="detect",
        system_prompt=None,
        prefix_denoising=PrefixDenoisingConfig(enabled=True),
        max_length=12000,
        dataset_name="unit",
        seed=17,
        eligibility_cache_dir=cache_root,
        eligibility_cache_fingerprint={"case": "ranked-cache"},
        eligibility_cache_wait_timeout_s=0.1,
        eligibility_precompute_workers=1,
    )

    assert len(rank1) == 2
    assert rank1.prefix_denoising_dataset_summary()["eligible_rows"] == 2


def test_prefix_denoising_eligibility_cache_reuses_compatible_loss_and_trial_fingerprint(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def _estimate_sample(*_args: object, **_kwargs: object):
        calls.append(f"estimate-{len(calls)}")
        return PrefixDenoisingPackingEstimate(ok=True, total_length=10 + len(calls))

    monkeypatch.setattr(
        prefix_dataset_mod,
        "estimate_hybrid_prefix_denoising_packing",
        _estimate_sample,
    )
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    cache_root = tmp_path / "eligibility"
    fingerprint = {
        "schema_version": "unit",
        "dataset_jsonl": "train.jsonl",
        "run_name": "first-trial",
        "save_delay_steps": 600,
        "prefix_denoising": {
            "enabled": True,
            "noise": {
                "center_shift_frac": 0.08,
                "uniform_scale_range": [0.92, 1.08],
            },
            "current_object_kl": {
                "weight": 0.05,
                "window_radius": 8,
                "num_objects_per_image": 1,
            },
        },
    }

    first = PrefixDenoisingTrainingDataset(
        _rows(2),
        swift_template=SimpleNamespace(tokenizer=object()),
        image_root=tmp_path,
        user_prompt="detect",
        system_prompt=None,
        prefix_denoising=PrefixDenoisingConfig(enabled=True),
        max_length=12000,
        dataset_name="unit",
        seed=17,
        eligibility_cache_dir=cache_root,
        eligibility_cache_fingerprint=fingerprint,
        eligibility_cache_wait_timeout_s=0.1,
        eligibility_precompute_workers=1,
    )
    assert len(first) == 2
    assert calls == ["estimate-0", "estimate-1"]

    def _unexpected_build(*_args: object, **_kwargs: object):
        raise AssertionError("compatible cache should avoid eligibility rebuild")

    monkeypatch.setattr(
        prefix_dataset_mod,
        "estimate_hybrid_prefix_denoising_packing",
        _unexpected_build,
    )
    kl_only_changed = {
        **fingerprint,
        "run_name": "second-trial",
        "save_delay_steps": 200,
        "prefix_denoising": {
            **fingerprint["prefix_denoising"],
            "current_object_kl": {
                "weight": 0.0,
                "window_radius": 4,
                "num_objects_per_image": 3,
            },
        },
    }

    second = PrefixDenoisingTrainingDataset(
        _rows(2),
        swift_template=SimpleNamespace(tokenizer=object()),
        image_root=tmp_path,
        user_prompt="detect",
        system_prompt=None,
        prefix_denoising=PrefixDenoisingConfig(enabled=True),
        max_length=12000,
        dataset_name="unit",
        seed=17,
        eligibility_cache_dir=cache_root,
        eligibility_cache_fingerprint=kl_only_changed,
        eligibility_cache_wait_timeout_s=0.1,
        eligibility_precompute_workers=1,
    )

    assert len(second) == 2
    assert len(list(cache_root.glob("*/eligibility.json"))) == 2


def test_prefix_denoising_eligibility_cache_reuses_after_jsonl_mtime_drift(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def _estimate_sample(*_args: object, **_kwargs: object):
        calls.append(f"estimate-{len(calls)}")
        return PrefixDenoisingPackingEstimate(ok=True, total_length=10 + len(calls))

    monkeypatch.setattr(
        prefix_dataset_mod,
        "estimate_hybrid_prefix_denoising_packing",
        _estimate_sample,
    )
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    cache_root = tmp_path / "eligibility"
    fingerprint = {
        "schema_version": "prefix_denoising_runtime_eligibility_v1",
        "dataset_jsonl": "train.coord.jsonl",
        "dataset_jsonl_size": 1234,
        "dataset_jsonl_mtime_ns": 111,
        "dataset_split": "detection_train",
        "sample_limit": None,
        "prefix_denoising": {
            "enabled": True,
            "noise": {
                "center_shift_frac": 0.08,
                "uniform_scale_range": [0.92, 1.08],
            },
            "current_object_kl": {
                "weight": 0.05,
                "window_radius": 8,
                "num_objects_per_image": 1,
            },
        },
    }

    first = PrefixDenoisingTrainingDataset(
        _rows(2),
        swift_template=SimpleNamespace(tokenizer=object()),
        image_root=tmp_path,
        user_prompt="detect",
        system_prompt=None,
        prefix_denoising=PrefixDenoisingConfig(enabled=True),
        max_length=12000,
        dataset_name="unit",
        seed=17,
        eligibility_cache_dir=cache_root,
        eligibility_cache_fingerprint=fingerprint,
        eligibility_cache_wait_timeout_s=0.1,
        eligibility_precompute_workers=1,
    )
    assert len(first) == 2
    assert calls == ["estimate-0", "estimate-1"]

    def _unexpected_build(*_args: object, **_kwargs: object):
        raise AssertionError("mtime-only drift should not rebuild eligibility cache")

    monkeypatch.setattr(
        prefix_dataset_mod,
        "estimate_hybrid_prefix_denoising_packing",
        _unexpected_build,
    )
    mtime_only_changed = {
        **fingerprint,
        "dataset_jsonl_mtime_ns": 222,
    }

    second = PrefixDenoisingTrainingDataset(
        _rows(2),
        swift_template=SimpleNamespace(tokenizer=object()),
        image_root=tmp_path,
        user_prompt="detect",
        system_prompt=None,
        prefix_denoising=PrefixDenoisingConfig(enabled=True),
        max_length=12000,
        dataset_name="unit",
        seed=17,
        eligibility_cache_dir=cache_root,
        eligibility_cache_fingerprint=mtime_only_changed,
        eligibility_cache_wait_timeout_s=0.1,
        eligibility_precompute_workers=1,
    )

    assert len(second) == 2
    assert len(list(cache_root.glob("*/eligibility.json"))) == 2


def test_prefix_denoising_runtime_wires_static_cache_root_to_eligibility_cache(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def _from_jsonl(jsonl_path: object, **kwargs: object) -> object:
        captured["jsonl_path"] = jsonl_path
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        runtime_mod,
        "PrefixDenoisingTrainingDataset",
        SimpleNamespace(from_jsonl=staticmethod(_from_jsonl)),
    )
    cache_root = tmp_path / "static-packing"
    cfg = SimpleNamespace(
        prefix_denoising=PrefixDenoisingConfig(enabled=True),
        data=SimpleNamespace(image_root=tmp_path),
        template={"max_length": 12000},
        global_max_length=None,
        training={
            "packing": True,
            "packing_length_precompute_workers": 7,
            "packing_wait_timeout_s": 11.0,
            "static_packing_cache": {"root_dir": str(cache_root)},
        },
    )

    dataset = runtime_mod.build_detection_dataset(
        tmp_path / "train.jsonl",
        swift_template=SimpleNamespace(tokenizer=object()),
        training_config=cfg,  # type: ignore[arg-type]
        custom_config=SimpleNamespace(user_prompt="detect"),
        system_prompt=None,
        seed=17,
        sample_limit=None,
        dataset_name="train",
    )

    assert dataset is not None
    assert captured["eligibility_cache_dir"] == (
        cache_root / "prefix_denoising_eligibility"
    )
    assert captured["eligibility_cache_wait_timeout_s"] == 11.0
    assert captured["eligibility_precompute_workers"] == 7
    fingerprint = captured["eligibility_cache_fingerprint"]
    assert isinstance(fingerprint, dict)
    assert fingerprint["dataset_split"] == "train"
    assert fingerprint["sample_limit"] is None
    assert fingerprint["fast_estimator"]["schema_version"].startswith("qwen3_vl_")


def test_runtime_eligibility_fingerprint_tracks_chat_template_and_processor_shape(
    tmp_path,
) -> None:
    cache_root = tmp_path / "static-packing"
    cfg = SimpleNamespace(
        prefix_denoising=PrefixDenoisingConfig(enabled=True),
        data=SimpleNamespace(image_root=tmp_path),
        template={"max_length": 12000, "template": "qwen3-vl"},
        detection_template=SimpleNamespace(
            id="compact_full",
            bbox_format="coord_tokens",
            object_field_order="desc_first",
        ),
        training={
            "packing": True,
            "static_packing_cache": {"root_dir": str(cache_root)},
        },
    )
    common = {
        "jsonl_path": tmp_path / "train.jsonl",
        "training_config": cfg,
        "custom_config": SimpleNamespace(user_prompt="detect"),
        "system_prompt": None,
        "max_length": 12000,
        "seed": 17,
        "sample_limit": None,
        "dataset_name": "train",
    }
    template_a = SimpleNamespace(
        tokenizer=SimpleNamespace(chat_template="chat-a"),
        image_processor=SimpleNamespace(patch_size=16, merge_size=2),
    )
    template_b = SimpleNamespace(
        tokenizer=SimpleNamespace(chat_template="chat-b"),
        image_processor=SimpleNamespace(patch_size=14, merge_size=4),
    )

    fingerprint_a = runtime_mod._prefix_denoising_eligibility_cache_kwargs(
        swift_template=template_a,
        **common,
    )["eligibility_cache_fingerprint"]
    fingerprint_b = runtime_mod._prefix_denoising_eligibility_cache_kwargs(
        swift_template=template_b,
        **common,
    )["eligibility_cache_fingerprint"]

    assert fingerprint_a["fast_estimator"] != fingerprint_b["fast_estimator"]
    assert fingerprint_a["fast_estimator"]["qwen_vl_patch_size"] == 16
    assert fingerprint_a["fast_estimator"]["qwen_vl_merge_size"] == 2
    assert (
        fingerprint_a["fast_estimator"]["tokenizer_chat_template_sha256"]
        != fingerprint_b["fast_estimator"]["tokenizer_chat_template_sha256"]
    )
