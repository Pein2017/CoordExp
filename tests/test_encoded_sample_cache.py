from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from src.config.schema import CoordTokensConfig
from src.datasets.dense_caption import BaseCaptionDataset
from src.datasets.wrappers.packed_caption import build_static_packed_dataset


class _CountingTemplate:
    max_pixels = 786432
    max_length = 64
    system = None

    def __init__(self) -> None:
        self.encode_calls = 0

    def encode(
        self, merged: Dict[str, Any], return_length: bool = True
    ) -> Dict[str, Any]:
        self.encode_calls += 1
        count = len((merged.get("assistant_payload") or {}).get("objects") or [])
        return {
            "input_ids": [count, count + 1],
            "labels": [count, count + 1],
            "length": 2,
        }


def _record(*, image: str, desc: str) -> dict[str, Any]:
    return {
        "images": [image],
        "width": 32,
        "height": 32,
        "objects": [
            {
                "desc": desc,
                "bbox_2d": [
                    "<|coord_0|>",
                    "<|coord_0|>",
                    "<|coord_1|>",
                    "<|coord_1|>",
                ],
            }
        ],
    }


def _cache_request(tmp_path, *, policy: str = "error") -> dict[str, Any]:
    return {
        "enabled": True,
        "root_dir": str(tmp_path / "encoded-cache"),
        "ineligible_policy": policy,
        "wait_timeout_s": 5.0,
        "max_resident_shards": 4,
        "dataset_split": "train",
        "dataset_jsonl": "train.jsonl",
        "fingerprint": {"cache_schema_version": 1, "dataset": "toy", "split": "train"},
    }


def _dataset(
    *,
    template: _CountingTemplate,
    tmp_path,
    dataset_name: str,
    object_ordering: str = "sorted",
    detection_sequence_format: str = "coordjson",
    policy: str = "error",
) -> BaseCaptionDataset:
    return BaseCaptionDataset(
        base_records=[
            _record(image="a.png", desc="alpha"),
            _record(image="b.png", desc="beta"),
        ],
        template=template,
        user_prompt="prompt",
        emit_norm="none",
        json_format="standard",
        dataset_name=dataset_name,
        seed=123,
        coord_tokens=CoordTokensConfig(enabled=True, skip_bbox_norm=True),
        object_ordering=object_ordering,  # type: ignore[arg-type]
        object_field_order="desc_first",
        detection_sequence_format=detection_sequence_format,
        encoded_sample_cache=_cache_request(tmp_path, policy=policy),
    )


def test_encoded_sample_cache_builds_once_and_reuses_without_hot_path_encode(
    tmp_path,
) -> None:
    template_a = _CountingTemplate()
    ds_a = _dataset(template=template_a, tmp_path=tmp_path, dataset_name="train_a")
    info_a = ds_a.get_encoded_sample_cache_info()
    assert info_a is not None
    assert info_a["status"] == "built"
    assert template_a.encode_calls == 2

    sample_a = ds_a[0]
    assert sample_a["dataset"] == "train_a"
    assert template_a.encode_calls == 2

    template_b = _CountingTemplate()
    ds_b = _dataset(template=template_b, tmp_path=tmp_path, dataset_name="train_b")
    info_b = ds_b.get_encoded_sample_cache_info()
    assert info_b is not None
    assert info_b["status"] == "reused"
    assert template_b.encode_calls == 0

    sample_b = ds_b[0]
    assert sample_b["dataset"] == "train_b"
    assert sample_b["base_idx"] == 0
    assert template_b.encode_calls == 0


def test_encoded_sample_cache_reattaches_runtime_metadata_for_shared_root_reuse(
    tmp_path,
) -> None:
    first = _dataset(
        template=_CountingTemplate(),
        tmp_path=tmp_path,
        dataset_name="first_namespace",
    )
    first_sample = first[0]

    second = _dataset(
        template=_CountingTemplate(),
        tmp_path=tmp_path,
        dataset_name="second_namespace",
    )
    second_sample = second[0]

    assert first_sample["dataset"] == "first_namespace"
    assert second_sample["dataset"] == "second_namespace"
    assert int(first_sample["sample_id"]) != int(second_sample["sample_id"])
    assert second_sample["base_idx"] == 0


def test_encoded_sample_cache_bypasses_ineligible_random_ordering_when_requested(
    tmp_path,
) -> None:
    template = _CountingTemplate()
    ds = _dataset(
        template=template,
        tmp_path=tmp_path,
        dataset_name="train_a",
        object_ordering="random",
        policy="bypass",
    )

    info = ds.get_encoded_sample_cache_info()
    assert info is not None
    assert info["status"] == "bypassed"
    assert "random object ordering" in str(info["reason"]).lower()
    assert template.encode_calls == 0

    _ = ds[0]
    assert template.encode_calls == 1


def test_encoded_sample_cache_rejects_ineligible_random_ordering_by_default(
    tmp_path,
) -> None:
    with pytest.raises(ValueError, match="object_ordering='sorted'"):
        _dataset(
            template=_CountingTemplate(),
            tmp_path=tmp_path,
            dataset_name="train_a",
            object_ordering="random",
            policy="error",
        )


def test_encoded_sample_cache_request_normalizes_typed_fields(tmp_path) -> None:
    from src.datasets.encoded_sample_cache import EncodedSampleCacheRequest

    request = EncodedSampleCacheRequest.from_mapping(_cache_request(tmp_path))

    assert request.enabled is True
    assert request.root_dir == (tmp_path / "encoded-cache").resolve()
    assert request.ineligible_policy == "error"
    assert request.wait_timeout_s == pytest.approx(5.0)
    assert request.max_resident_shards == 4
    assert request.dataset_split == "train"
    assert request.dataset_jsonl == "train.jsonl"
    assert request.fingerprint["cache_schema_version"] == 1


def test_encoded_sample_cache_setup_rejects_enabled_request_without_root_dir() -> None:
    from src.datasets.encoded_sample_cache import setup_encoded_sample_cache_for_dataset

    request = {
        "enabled": True,
        "ineligible_policy": "bypass",
        "fingerprint": {"cache_schema_version": 1, "dataset": "toy"},
    }

    with pytest.raises(ValueError, match="root_dir"):
        setup_encoded_sample_cache_for_dataset(
            SimpleNamespace(object_ordering="random"),
            request,
        )


def test_encoded_sample_cache_request_allows_disabled_request_without_root_dir() -> None:
    from src.datasets.encoded_sample_cache import setup_encoded_sample_cache_for_dataset

    store, info = setup_encoded_sample_cache_for_dataset(
        SimpleNamespace(object_ordering="random"),
        {"enabled": False, "fingerprint": {"cache_schema_version": 1}},
    )

    assert store is None
    assert info is None


def test_encoded_sample_cache_setup_ignores_disabled_stale_derived_fields() -> None:
    from src.datasets.encoded_sample_cache import setup_encoded_sample_cache_for_dataset

    store, info = setup_encoded_sample_cache_for_dataset(
        SimpleNamespace(object_ordering="random"),
        {
            "enabled": False,
            "fingerprint": {"cache_schema_version": 1},
            "cache_dir": "/tmp/stale",
            "manifest_path": "/tmp/stale/old-manifest.json",
        },
    )

    assert store is None
    assert info is None


def test_encoded_sample_cache_request_rejects_fingerprint_digest_mismatch(
    tmp_path,
) -> None:
    from src.datasets.encoded_sample_cache import EncodedSampleCacheRequest

    payload = _cache_request(tmp_path)
    payload["fingerprint_sha256"] = "not-the-canonical-digest"

    with pytest.raises(ValueError, match="fingerprint_sha256"):
        EncodedSampleCacheRequest.from_mapping(payload)


def test_encoded_sample_cache_request_rejects_cache_dir_mismatch(tmp_path) -> None:
    from src.datasets.encoded_sample_cache import EncodedSampleCacheRequest

    payload = EncodedSampleCacheRequest.from_mapping(
        _cache_request(tmp_path)
    ).to_mapping()
    payload["cache_dir"] = str(tmp_path / "elsewhere")

    with pytest.raises(ValueError, match="cache_dir"):
        EncodedSampleCacheRequest.from_mapping(payload)


def test_encoded_sample_cache_request_rejects_manifest_path_mismatch(tmp_path) -> None:
    from src.datasets.encoded_sample_cache import EncodedSampleCacheRequest

    payload = EncodedSampleCacheRequest.from_mapping(
        _cache_request(tmp_path)
    ).to_mapping()
    payload["manifest_path"] = str(Path(payload["cache_dir"]) / "other.json")

    with pytest.raises(ValueError, match="manifest_path"):
        EncodedSampleCacheRequest.from_mapping(payload)


def test_encoded_sample_cache_manifest_roundtrips_serialized_payload(tmp_path) -> None:
    from src.datasets.encoded_sample_cache import EncodedSampleCacheManifest

    ds = _dataset(
        template=_CountingTemplate(),
        tmp_path=tmp_path,
        dataset_name="train_a",
    )
    info = ds.get_encoded_sample_cache_info()
    assert info is not None

    manifest_path = Path(str(info["manifest_path"]))
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = EncodedSampleCacheManifest.from_mapping(payload, path=manifest_path)

    assert manifest.status == "complete"
    assert manifest.fingerprint_sha256 == info["fingerprint_sha256"]
    assert manifest.shard_count == int(info["shard_count"])
    assert manifest.payload_keys == tuple(info["payload_keys"])
    assert manifest.to_mapping() == payload


@pytest.mark.parametrize(
    ("mutation", "error_pattern"),
    [
        (lambda payload: payload.pop("shards"), "complete manifest missing shards"),
        (
            lambda payload: payload.update({"fingerprint_sha256": ""}),
            "complete manifest fingerprint_sha256",
        ),
        (
            lambda payload: payload.update({"shard_size": 0}),
            "complete manifest shard_size",
        ),
        (
            lambda payload: payload.update(
                {"shards": [{**payload["shards"][0], "count": 999}]}
            ),
            "complete manifest shard count",
        ),
        (
            lambda payload: payload.update(
                {"shards": [{**payload["shards"][0], "file": "../outside.pt"}]}
            ),
            "complete manifest shard file",
        ),
        (
            lambda payload: payload.update(
                {"shards": [{**payload["shards"][0], "file": "/tmp/outside.pt"}]}
            ),
            "complete manifest shard file",
        ),
        (
            lambda payload: payload.update(
                {"shards": [{**payload["shards"][0], "file": "nested/shard.pt"}]}
            ),
            "complete manifest shard file",
        ),
        (
            lambda payload: payload.update(
                {"shards": [payload["shards"][0], {**payload["shards"][0]}]}
            ),
            "complete manifest duplicate shard",
        ),
        (
            lambda payload: payload.update(
                {
                    "num_samples": 3,
                    "shards": [
                        {
                            **payload["shards"][0],
                            "shard_index": 0,
                            "start": 0,
                            "end": 2,
                            "count": 2,
                        },
                        {
                            **payload["shards"][0],
                            "shard_index": 1,
                            "start": 1,
                            "end": 2,
                            "count": 1,
                        },
                    ],
                }
            ),
            "complete manifest shard range",
        ),
        (
            lambda payload: payload.update({"num_samples": payload["num_samples"] + 1}),
            "complete manifest shard count",
        ),
    ],
)
def test_encoded_sample_cache_manifest_rejects_malformed_complete_payloads(
    tmp_path,
    mutation,
    error_pattern: str,
) -> None:
    from src.datasets.encoded_sample_cache import EncodedSampleCacheManifest

    ds = _dataset(
        template=_CountingTemplate(),
        tmp_path=tmp_path,
        dataset_name="train_a",
    )
    info = ds.get_encoded_sample_cache_info()
    assert info is not None

    manifest_path = Path(str(info["manifest_path"]))
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutation(payload)

    with pytest.raises((TypeError, ValueError), match=error_pattern):
        EncodedSampleCacheManifest.from_mapping(payload, path=manifest_path)


def test_encoded_sample_cache_rejects_corrupt_complete_manifest_before_reuse(
    tmp_path,
) -> None:
    ds = _dataset(
        template=_CountingTemplate(),
        tmp_path=tmp_path,
        dataset_name="train_a",
    )
    info = ds.get_encoded_sample_cache_info()
    assert info is not None

    manifest_path = Path(str(info["manifest_path"]))
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.pop("shards")
    payload.pop("num_samples")
    payload.pop("shard_size")
    payload["fingerprint_sha256"] = ""
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="complete manifest"):
        _dataset(
            template=_CountingTemplate(),
            tmp_path=tmp_path,
            dataset_name="train_b",
        )


def test_detection_sequence_format_participates_in_static_packing_surfaces(
    tmp_path,
) -> None:
    ds = _dataset(
        template=_CountingTemplate(),
        tmp_path=tmp_path,
        dataset_name="train_a",
        detection_sequence_format="compact_full",
    )

    info = ds._static_packing_precompute_info()

    assert (
        info["fingerprint_surfaces"]["detection_sequence_format"]
        == "compact_full"
    )


def test_static_packing_reuses_cache_backed_dataset_after_full_length_probe(
    tmp_path,
) -> None:
    template = _CountingTemplate()
    ds = _dataset(template=template, tmp_path=tmp_path, dataset_name="train_a")
    assert template.encode_calls == 2

    packed = build_static_packed_dataset(
        ds,
        template=template,
        packing_length=4,
        cache_dir=tmp_path / "static-packing",
        fingerprint={"dataset": "toy", "split": "train"},
        world_size=1,
        train_dataloader_shuffle=False,
    )

    assert len(packed) > 0
    plan_encode_calls = template.encode_calls
    assert plan_encode_calls > 2
    first_pack = packed[0]
    assert isinstance(first_pack, list)
    assert template.encode_calls == plan_encode_calls


def test_static_packing_prefers_cache_backed_length_helper_over_dataset_getitem(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template = _CountingTemplate()
    ds = _dataset(template=template, tmp_path=tmp_path, dataset_name="train_a")
    assert template.encode_calls == 2

    def _boom(self, index: int):
        raise AssertionError("static packing should not call dataset.__getitem__ for length planning")

    monkeypatch.setattr(type(ds), "__getitem__", _boom)

    packed = build_static_packed_dataset(
        ds,
        template=template,
        packing_length=4,
        cache_dir=tmp_path / "static-packing",
        fingerprint={"dataset": "toy", "split": "train"},
        world_size=1,
        train_dataloader_shuffle=False,
    )

    assert len(packed) > 0


def test_encoded_sample_cache_evicts_old_shards_when_residency_is_bounded(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.datasets import encoded_sample_cache as cache_mod

    request = _cache_request(tmp_path)
    request["max_resident_shards"] = 1

    monkeypatch.setattr(cache_mod, "_DEFAULT_ENCODED_SAMPLE_SHARD_SIZE", 1)

    ds = BaseCaptionDataset(
        base_records=[
            _record(image="a.png", desc="alpha"),
            _record(image="b.png", desc="beta"),
            _record(image="c.png", desc="gamma"),
        ],
        template=_CountingTemplate(),
        user_prompt="prompt",
        emit_norm="none",
        json_format="standard",
        dataset_name="train_a",
        seed=123,
        coord_tokens=CoordTokensConfig(enabled=True, skip_bbox_norm=True),
        object_ordering="sorted",
        object_field_order="desc_first",
        encoded_sample_cache=request,
    )

    store = ds._encoded_sample_cache
    assert store is not None
    assert store.info()["max_resident_shards"] == 1

    first = ds[0]
    second = ds[1]
    third = ds[2]
    again_first = ds[0]

    assert first["dataset"] == "train_a"
    assert second["dataset"] == "train_a"
    assert third["dataset"] == "train_a"
    assert again_first["dataset"] == "train_a"
    assert len(store._loaded_shards) <= 1


def test_encoded_sample_cache_supports_concurrent_shard_loads(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.datasets import encoded_sample_cache as cache_mod

    request = _cache_request(tmp_path)
    request["max_resident_shards"] = 2

    monkeypatch.setattr(cache_mod, "_DEFAULT_ENCODED_SAMPLE_SHARD_SIZE", 1)

    ds = BaseCaptionDataset(
        base_records=[
            _record(image="a.png", desc="alpha"),
            _record(image="b.png", desc="beta"),
            _record(image="c.png", desc="gamma"),
            _record(image="d.png", desc="delta"),
        ],
        template=_CountingTemplate(),
        user_prompt="prompt",
        emit_norm="none",
        json_format="standard",
        dataset_name="train_a",
        seed=123,
        coord_tokens=CoordTokensConfig(enabled=True, skip_bbox_norm=True),
        object_ordering="sorted",
        object_field_order="desc_first",
        encoded_sample_cache=request,
    )

    store = ds._encoded_sample_cache
    assert store is not None
    assert store.thread_safe is True

    def _load(index: int) -> str:
        return str(ds[index]["dataset"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(_load, [0, 1, 2, 3, 0, 2, 1, 3]))

    assert results == ["train_a"] * 8
    assert len(store._loaded_shards) <= 2
