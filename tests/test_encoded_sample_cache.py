from __future__ import annotations

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


def test_static_packing_can_consume_cache_backed_dataset_without_reencoding(
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
    first_pack = packed[0]
    assert isinstance(first_pack, list)
    assert template.encode_calls == 2
