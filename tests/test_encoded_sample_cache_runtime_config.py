from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.config.schema import CoordTokensConfig
from src.datasets.encoded_sample_cache import EncodedSampleCacheRequest
from src.sft import (
    _attach_encoded_sample_cache_run_metadata,
    _build_encoded_sample_cache_bypass_info,
    _build_encoded_sample_cache_fingerprint,
    _build_encoded_sample_cache_request,
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
        detection_sequence_format="coordjson",
        object_ordering="sorted",
        object_field_order="desc_first",
        use_summary=False,
        offline_max_pixels=10485760,
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


def test_build_encoded_sample_cache_request_returns_canonical_payload(tmp_path) -> None:
    train_jsonl = tmp_path / "train.jsonl"
    train_jsonl.write_text('{"id": 1}\n', encoding="utf-8")
    runtime_cfg = _parse_encoded_sample_cache_config(
        {
            "encoded_sample_cache": {
                "enabled": True,
                "root_dir": str(tmp_path / "cache"),
                "ineligible_policy": "bypass",
                "wait_timeout_s": 5,
                "max_resident_shards": 2,
            }
        },
        SimpleNamespace(output_dir=str(tmp_path / "out")),
    )

    payload = _build_encoded_sample_cache_request(
        runtime_cfg=runtime_cfg,
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

    assert payload is not None
    request = EncodedSampleCacheRequest.from_mapping(payload)
    assert payload == request.to_mapping()
    assert payload["fingerprint_sha256"]
    assert payload["cache_dir"] == str(
        Path(payload["root_dir"]) / payload["fingerprint_sha256"]
    )
    assert payload["manifest_path"] == str(Path(payload["cache_dir"]) / "manifest.json")


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
    cxcywh = _build_encoded_sample_cache_fingerprint(
        training_config=SimpleNamespace(
            global_max_length=1024,
            template={"system": "sys", "truncation_strategy": "raise"},
        ),
        custom_config=SimpleNamespace(
            **{**_custom_config().__dict__, "bbox_format": "cxcywh"}
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
    assert cxcywh["custom_bbox_format"] == "cxcywh"
    assert xyxy != cxcy_logw_logh
    assert cxcy_logw_logh != cxcywh


def test_encoded_sample_cache_fingerprint_tracks_detection_sequence_format(
    tmp_path,
) -> None:
    train_jsonl = tmp_path / "train.jsonl"
    train_jsonl.write_text('{"id": 1}\n', encoding="utf-8")

    coordjson = _build_encoded_sample_cache_fingerprint(
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
    compact = _build_encoded_sample_cache_fingerprint(
        training_config=SimpleNamespace(
            global_max_length=1024,
            template={"system": "sys", "truncation_strategy": "raise"},
        ),
        custom_config=SimpleNamespace(
            **{**_custom_config().__dict__, "detection_sequence_format": "compact_full"}
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

    assert coordjson["custom_detection_sequence_format"] == "stage1_json_pretty"
    assert compact["custom_detection_sequence_format"] == "compact"
    assert coordjson != compact


def test_encoded_sample_cache_fingerprint_canonicalizes_legacy_compact_format(
    tmp_path,
) -> None:
    train_jsonl = tmp_path / "train.jsonl"
    train_jsonl.write_text('{"id": 1}\n', encoding="utf-8")
    common_training = SimpleNamespace(
        global_max_length=1024,
        template={"system": "sys", "truncation_strategy": "raise"},
    )
    common_kwargs = dict(
        training_config=common_training,
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

    compact = _build_encoded_sample_cache_fingerprint(
        custom_config=SimpleNamespace(
            **{**_custom_config().__dict__, "detection_sequence_format": "compact"}
        ),
        **common_kwargs,
    )
    compact_full = _build_encoded_sample_cache_fingerprint(
        custom_config=SimpleNamespace(
            **{**_custom_config().__dict__, "detection_sequence_format": "compact_full"}
        ),
        **common_kwargs,
    )

    assert compact["custom_detection_sequence_format"] == "compact"
    assert compact == compact_full


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


def test_encoded_sample_cache_fingerprint_tracks_template_and_field_order_axes(
    tmp_path,
) -> None:
    train_jsonl = tmp_path / "train.jsonl"
    train_jsonl.write_text('{"id": 1}\n', encoding="utf-8")
    common_template = _Template(max_length=128)
    common_train_args = SimpleNamespace(max_model_len=512)

    def fingerprint(
        *,
        template_id: str = "compact_object_box_closed",
        object_field_order: str = "desc_first",
        global_max_length: int = 12000,
        prompt_variant: str = "default",
    ) -> dict[str, object]:
        return _build_encoded_sample_cache_fingerprint(
            training_config=SimpleNamespace(
                global_max_length=global_max_length,
                template={"system": "sys", "truncation_strategy": "raise"},
                detection_template={"id": template_id},
            ),
            custom_config=SimpleNamespace(
                **{
                    **_custom_config().__dict__,
                    "detection_sequence_format": "compact",
                    "object_field_order": object_field_order,
                    "extra": {"prompt_variant": prompt_variant},
                }
            ),
            template=common_template,
            train_args=common_train_args,
            dataset_seed=7,
            dataset_jsonl=str(train_jsonl),
            dataset_split="train",
            dataset_mode="dense",
            sample_limit=64,
            system_prompt_dense="sys",
            system_prompt_summary=None,
        )

    baseline = fingerprint()
    geometry_first = fingerprint(object_field_order="geometry_first")
    object_closed = fingerprint(template_id="compact_object_closed")
    longer = fingerprint(global_max_length=16000)
    prompt_changed = fingerprint(prompt_variant="lvis_stage1_federated")
    legacy_alias = fingerprint(template_id="compact_full")

    assert baseline["detection_template_id"] == "compact_object_box_closed"
    assert baseline["tokenizer_id"] == "unknown_tokenizer"
    assert baseline["custom_object_field_order"] == "desc_first"
    assert baseline["global_max_length"] == 12000
    assert baseline["custom_prompt_template_hash"]
    assert geometry_first["custom_object_field_order"] == "geometry_first"
    assert object_closed["detection_template_id"] == "compact_object_closed"
    assert longer["global_max_length"] == 16000
    assert prompt_changed["custom_prompt_template_hash"] != baseline["custom_prompt_template_hash"]
    assert legacy_alias["detection_template_id"] == "compact"
    assert baseline != geometry_first
    assert baseline != object_closed
    assert baseline != longer
    assert baseline != prompt_changed


def test_encoded_sample_cache_fingerprint_tracks_normalized_hierarchy_axes(
    tmp_path,
) -> None:
    train_jsonl = tmp_path / "train.jsonl"
    train_jsonl.write_text('{"id": 1}\n', encoding="utf-8")
    common_template = _Template(max_length=128)
    common_train_args = SimpleNamespace(model="model-a", max_model_len=512)

    def fingerprint(
        *,
        pipeline_id: str = "stage1_standard_sft",
        objective_id: str = "standard_ce",
        sample_factory_id: str = "detection_sequence",
        template_id: str = "compact_object_box_closed",
        object_ordering: str = "sorted",
        object_field_order: str = "desc_first",
        bbox_format: str = "xyxy",
        coordinate_surface: str = "coord_token",
        strict_parse: bool = True,
        prompt_variant: str = "default",
    ) -> dict[str, object]:
        return _build_encoded_sample_cache_fingerprint(
            training_config=SimpleNamespace(
                global_max_length=12000,
                pipeline={"id": pipeline_id},
                objective={"id": objective_id},
                sample_factory={
                    "id": sample_factory_id,
                    "target_sequence": {
                        "object_ordering": object_ordering,
                        "object_field_order": object_field_order,
                        "bbox_format": bbox_format,
                        "coordinate_surface": coordinate_surface,
                        "strict_parse": strict_parse,
                    },
                },
                detection_template={"id": template_id},
                prompt={"variant": prompt_variant},
                template={"system": "sys", "truncation_strategy": "raise"},
            ),
            custom_config=SimpleNamespace(
                **{
                    **_custom_config().__dict__,
                    "object_ordering": object_ordering,
                    "object_field_order": object_field_order,
                    "bbox_format": bbox_format,
                    "extra": {"prompt_variant": prompt_variant},
                }
            ),
            template=common_template,
            train_args=common_train_args,
            dataset_seed=7,
            dataset_jsonl=str(train_jsonl),
            dataset_split="train",
            dataset_mode="dense",
            sample_limit=64,
            system_prompt_dense="sys",
            system_prompt_summary=None,
        )

    baseline = fingerprint()
    research = fingerprint(
        pipeline_id="stage1_research_teacher_forcing",
        objective_id="research_teacher_forcing",
    )
    random_order = fingerprint(object_ordering="random_permutation")
    geometry_first = fingerprint(object_field_order="geometry_first")
    bbox_changed = fingerprint(bbox_format="cxcy_logw_logh")
    strict_changed = fingerprint(strict_parse=False)
    prompt_changed = fingerprint(prompt_variant="coco_80")

    assert baseline["pipeline_id"] == "stage1_standard_sft"
    assert baseline["objective_id"] == "standard_ce"
    assert baseline["sample_factory_id"] == "detection_sequence"
    assert baseline["sample_factory_target_sequence_object_ordering"] == "sorted"
    assert baseline["sample_factory_target_sequence_object_field_order"] == "desc_first"
    assert baseline["sample_factory_target_sequence_bbox_format"] == "xyxy"
    assert baseline["sample_factory_target_sequence_coordinate_surface"] == "coord_token"
    assert baseline["sample_factory_target_sequence_strict_parse"] is True
    assert baseline["prompt_variant"] == "default"
    assert baseline["prompt_template_hash"] == baseline["custom_prompt_template_hash"]
    assert baseline["chat_template_identity"] == "unknown_chat_template"
    assert baseline["tokenizer_id"] == "model-a"
    assert baseline["packing_length"] == 128
    assert "custom_object_field_order" in baseline
    assert "coord_tokens" in baseline
    assert baseline != research
    assert baseline != random_order
    assert baseline != geometry_first
    assert baseline != bbox_changed
    assert baseline != strict_changed
    assert baseline != prompt_changed


def test_encoded_sample_cache_fingerprint_tracks_tokenizer_identity(
    tmp_path,
) -> None:
    train_jsonl = tmp_path / "train.jsonl"
    train_jsonl.write_text('{"id": 1}\n', encoding="utf-8")
    common_custom = _custom_config()
    common_kwargs = dict(
        custom_config=common_custom,
        template=_Template(max_length=128),
        dataset_seed=7,
        dataset_jsonl=str(train_jsonl),
        dataset_split="train",
        dataset_mode="dense",
        sample_limit=64,
        system_prompt_dense="sys",
        system_prompt_summary=None,
    )

    model_a = _build_encoded_sample_cache_fingerprint(
        training_config=SimpleNamespace(
            global_max_length=12000,
            template={"system": "sys", "truncation_strategy": "raise"},
            model={"model": "model-a"},
        ),
        train_args=SimpleNamespace(model="fallback-model", max_model_len=512),
        **common_kwargs,
    )
    model_b = _build_encoded_sample_cache_fingerprint(
        training_config=SimpleNamespace(
            global_max_length=12000,
            template={"system": "sys", "truncation_strategy": "raise"},
            model={"model": "model-b"},
        ),
        train_args=SimpleNamespace(model="fallback-model", max_model_len=512),
        **common_kwargs,
    )
    fallback = _build_encoded_sample_cache_fingerprint(
        training_config=SimpleNamespace(
            global_max_length=12000,
            template={"system": "sys", "truncation_strategy": "raise"},
        ),
        train_args=SimpleNamespace(model="fallback-model", max_model_len=512),
        **common_kwargs,
    )

    assert model_a["tokenizer_id"] == "model-a"
    assert model_b["tokenizer_id"] == "model-b"
    assert fallback["tokenizer_id"] == "fallback-model"
    assert model_a != model_b
    assert model_a != fallback


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


def test_build_encoded_sample_cache_bypass_info_records_reason() -> None:
    request = EncodedSampleCacheRequest.from_mapping(
        {
            "enabled": True,
            "ineligible_policy": "bypass",
            "wait_timeout_s": 5,
            "dataset_split": "train",
            "dataset_jsonl": "train.jsonl",
            "fingerprint": {"dataset_split": "train"},
            "root_dir": "/tmp/cache",
        }
    ).to_mapping()

    info = _build_encoded_sample_cache_bypass_info(
        request,
        reason="unit_test_ineligible_surface",
    )

    assert info["enabled"] is True
    assert info["status"] == "bypassed"
    assert info["policy"] == "bypass"
    assert info["reason"] == "unit_test_ineligible_surface"
