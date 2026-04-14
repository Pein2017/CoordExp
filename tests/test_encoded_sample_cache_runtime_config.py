from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.config.schema import CoordTokensConfig
from src.sft import (
    _attach_encoded_sample_cache_run_metadata,
    _build_encoded_sample_cache_fingerprint,
    _parse_encoded_sample_cache_config,
)


class _Template:
    def __init__(self, max_length: int = 128) -> None:
        self.max_length = int(max_length)


def _custom_config() -> SimpleNamespace:
    return SimpleNamespace(
        user_prompt="prompt",
        emit_norm="none",
        json_format="standard",
        bbox_format="xyxy",
        object_ordering="sorted",
        object_field_order="desc_first",
        use_summary=False,
        offline_max_pixels=786432,
        coord_tokens=CoordTokensConfig(enabled=True, skip_bbox_norm=True),
    )


def test_parse_encoded_sample_cache_config_uses_output_dir_default_root(tmp_path) -> None:
    cfg = _parse_encoded_sample_cache_config(
        {"encoded_sample_cache": {"enabled": True}},
        SimpleNamespace(output_dir=str(tmp_path / "out")),
    )

    assert cfg.enabled is True
    assert cfg.root_dir == str((tmp_path / "out" / "cache" / "encoded_samples").resolve())
    assert cfg.ineligible_policy == "error"
    assert cfg.wait_timeout_s == pytest.approx(7200.0)


def test_parse_encoded_sample_cache_config_rejects_negative_wait_timeout() -> None:
    with pytest.raises(ValueError, match="encoded_sample_cache.wait_timeout_s"):
        _parse_encoded_sample_cache_config(
            {
                "encoded_sample_cache": {
                    "enabled": True,
                    "root_dir": "/tmp/cache",
                    "wait_timeout_s": -1,
                }
            },
            SimpleNamespace(output_dir="out"),
        )


def test_encoded_sample_cache_fingerprint_tracks_dataset_identity(tmp_path) -> None:
    train_jsonl = tmp_path / "train.jsonl"
    train_jsonl.write_text('{"id": 1}\n', encoding="utf-8")

    fingerprint = _build_encoded_sample_cache_fingerprint(
        training_config=SimpleNamespace(
            global_max_length=1024,
            template={"system": "sys", "truncation_strategy": "raise"},
        ),
        custom_config=_custom_config(),
        template=_Template(max_length=128),
        train_args=SimpleNamespace(max_model_len=512),
        dataset_seed=7,
        dataset_jsonl=str(train_jsonl),
        dataset_split="train",
        dataset_mode="dense",
        sample_limit=64,
        system_prompt_dense="sys",
        system_prompt_summary=None,
    )

    assert fingerprint["dataset_split"] == "train"
    assert fingerprint["dataset_jsonl"] == str(train_jsonl)
    assert fingerprint["sample_limit"] == 64
    source = fingerprint["dataset_source_jsonl"]
    assert isinstance(source, dict)
    assert source["raw_path"] == str(train_jsonl)


def test_encoded_sample_cache_fingerprint_tracks_bbox_format(tmp_path) -> None:
    train_jsonl = tmp_path / "train.jsonl"
    train_jsonl.write_text('{"id": 1}\n', encoding="utf-8")

    xyxy = _build_encoded_sample_cache_fingerprint(
        training_config=SimpleNamespace(
            global_max_length=1024,
            template={"system": "sys", "truncation_strategy": "raise"},
        ),
        custom_config=_custom_config(),
        template=_Template(max_length=128),
        train_args=SimpleNamespace(max_model_len=512),
        dataset_seed=7,
        dataset_jsonl=str(train_jsonl),
        dataset_split="train",
        dataset_mode="dense",
        sample_limit=64,
        system_prompt_dense="sys",
        system_prompt_summary=None,
    )
    cxcy_logw_logh = _build_encoded_sample_cache_fingerprint(
        training_config=SimpleNamespace(
            global_max_length=1024,
            template={"system": "sys", "truncation_strategy": "raise"},
        ),
        custom_config=SimpleNamespace(
            **{**_custom_config().__dict__, "bbox_format": "cxcy_logw_logh"}
        ),
        template=_Template(max_length=128),
        train_args=SimpleNamespace(max_model_len=512),
        dataset_seed=7,
        dataset_jsonl=str(train_jsonl),
        dataset_split="train",
        dataset_mode="dense",
        sample_limit=64,
        system_prompt_dense="sys",
        system_prompt_summary=None,
    )

    assert xyxy["custom_bbox_format"] == "xyxy"
    assert cxcy_logw_logh["custom_bbox_format"] == "cxcy_logw_logh"
    assert xyxy != cxcy_logw_logh


def test_encoded_sample_cache_fingerprint_tracks_prompt_variant_and_template_hash(
    tmp_path,
) -> None:
    train_jsonl = tmp_path / "train.jsonl"
    train_jsonl.write_text('{"id": 1}\n', encoding="utf-8")

    default_fp = _build_encoded_sample_cache_fingerprint(
        training_config=SimpleNamespace(
            global_max_length=1024,
            template={"system": "sys", "truncation_strategy": "raise"},
        ),
        custom_config=SimpleNamespace(
            **{**_custom_config().__dict__, "bbox_format": "cxcy_logw_logh"},
            extra={"prompt_variant": "default"},
        ),
        template=_Template(max_length=128),
        train_args=SimpleNamespace(max_model_len=512),
        dataset_seed=7,
        dataset_jsonl=str(train_jsonl),
        dataset_split="train",
        dataset_mode="dense",
        sample_limit=64,
        system_prompt_dense="sys",
        system_prompt_summary=None,
    )
    lvis_fp = _build_encoded_sample_cache_fingerprint(
        training_config=SimpleNamespace(
            global_max_length=1024,
            template={"system": "sys", "truncation_strategy": "raise"},
        ),
        custom_config=SimpleNamespace(
            **{**_custom_config().__dict__, "bbox_format": "cxcy_logw_logh"},
            extra={"prompt_variant": "lvis_stage1_federated"},
        ),
        template=_Template(max_length=128),
        train_args=SimpleNamespace(max_model_len=512),
        dataset_seed=7,
        dataset_jsonl=str(train_jsonl),
        dataset_split="train",
        dataset_mode="dense",
        sample_limit=64,
        system_prompt_dense="sys",
        system_prompt_summary=None,
    )

    assert default_fp["custom_prompt_variant"] == "default"
    assert lvis_fp["custom_prompt_variant"] == "lvis_stage1_federated"
    assert isinstance(default_fp["custom_prompt_template_hash"], str)
    assert isinstance(lvis_fp["custom_prompt_template_hash"], str)
    assert default_fp["custom_prompt_template_hash"] != lvis_fp["custom_prompt_template_hash"]


def test_attach_encoded_sample_cache_run_metadata_scopes_train_and_eval() -> None:
    meta: dict[str, object] = {}
    _attach_encoded_sample_cache_run_metadata(
        meta,
        train_cache_info={"status": "built", "root_dir": "/tmp/train"},
        eval_cache_info={"status": "reused", "root_dir": "/tmp/eval"},
    )

    block = meta["encoded_sample_cache"]
    assert isinstance(block, dict)
    assert block["train"]["status"] == "built"
    assert block["eval"]["status"] == "reused"
