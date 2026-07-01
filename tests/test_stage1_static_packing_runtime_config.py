from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from src.callbacks import DatasetEpochCallback
from src.config.loader import ConfigLoader
from src.config.schema import DetectionTrainingConfig
from src.datasets.wrappers.packed_caption import _fingerprint_diff_keys
import src.sft as sft_module
from src.sft import (
    PackingRuntimeConfig,
    StaticPackingCacheRuntimeConfig,
    _append_dataset_epoch_callback,
    _build_static_packing_fingerprint,
    _pipeline_base_callbacks,
    _pipeline_data_collator,
    _parse_static_packing_cache_config,
    _resolve_static_packing_cache_dir,
    _parse_encoded_sample_cache_config,
    _parse_packing_config,
    _recompute_gas_for_packing,
    _resolve_model_checkpoint_path,
    _validate_stage1_static_packing_policy,
    _validate_static_packing_accumulation_windows,
)


class _Template:
    def __init__(self, max_length: int):
        self.max_length = int(max_length)


class _EpochDataset:
    def __init__(self) -> None:
        self.epochs: list[int] = []

    def set_epoch(self, epoch: int) -> None:
        self.epochs.append(int(epoch))


class _UnsafeLengthDataset:
    def __init__(self) -> None:
        self.records = [{"length": 2}, {"length": 3}, {"length": 4}]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        return dict(self.records[int(index)])

    def _static_packing_precompute_info(self) -> dict[str, object]:
        return {"thread_safe": False, "reason": "template mutation"}


def test_parse_packing_config_defaults_to_static_mode() -> None:
    cfg = _parse_packing_config(
        training_cfg={"packing": True},
        template=_Template(max_length=256),
        train_args=SimpleNamespace(max_model_len=512),
    )

    assert cfg.enabled is True
    assert cfg.mode == "static"
    assert cfg.packing_length == 256
    assert cfg.eval_packing is True
    assert cfg.wait_timeout_s == pytest.approx(7200.0)
    assert cfg.length_cache_persist_every is None
    assert cfg.length_precompute_workers == 8


def test_parse_packing_config_accepts_static_mode() -> None:
    cfg = _parse_packing_config(
        training_cfg={"packing": True, "packing_mode": "static"},
        template=_Template(max_length=128),
        train_args=SimpleNamespace(max_model_len=0),
    )

    assert cfg.mode == "static"
    assert cfg.packing_length == 128


def test_parse_packing_config_allows_disabling_eval_packing() -> None:
    cfg = _parse_packing_config(
        training_cfg={"packing": True, "packing_mode": "static", "eval_packing": False},
        template=_Template(max_length=128),
        train_args=SimpleNamespace(max_model_len=0),
    )

    assert cfg.eval_packing is False


def test_parse_packing_config_accepts_static_cache_runtime_knobs() -> None:
    cfg = _parse_packing_config(
        training_cfg={
            "packing": True,
            "packing_mode": "static",
            "packing_wait_timeout_s": 1234,
            "packing_length_cache_persist_every": 2048,
            "packing_length_precompute_workers": 6,
        },
        template=_Template(max_length=128),
        train_args=SimpleNamespace(max_model_len=0),
    )

    assert cfg.wait_timeout_s == pytest.approx(1234.0)
    assert cfg.length_cache_persist_every == 2048
    assert cfg.length_precompute_workers == 6


def test_parse_packing_config_rejects_negative_wait_timeout() -> None:
    with pytest.raises(ValueError, match="packing_wait_timeout_s"):
        _parse_packing_config(
            training_cfg={
                "packing": True,
                "packing_mode": "static",
                "packing_wait_timeout_s": -1,
            },
            template=_Template(max_length=128),
            train_args=SimpleNamespace(max_model_len=0),
        )


def test_parse_packing_config_rejects_nonpositive_length_cache_persist_every() -> None:
    with pytest.raises(ValueError, match="packing_length_cache_persist_every"):
        _parse_packing_config(
            training_cfg={
                "packing": True,
                "packing_mode": "static",
                "packing_length_cache_persist_every": 0,
            },
            template=_Template(max_length=128),
            train_args=SimpleNamespace(max_model_len=0),
        )


def test_parse_packing_config_rejects_nonpositive_length_precompute_workers() -> None:
    with pytest.raises(ValueError, match="packing_length_precompute_workers"):
        _parse_packing_config(
            training_cfg={
                "packing": True,
                "packing_mode": "static",
                "packing_length_precompute_workers": 0,
            },
            template=_Template(max_length=128),
            train_args=SimpleNamespace(max_model_len=0),
        )


def test_parse_encoded_sample_cache_config_accepts_residency_bound() -> None:
    cfg = _parse_encoded_sample_cache_config(
        training_cfg={
            "encoded_sample_cache": {"enabled": True, "max_resident_shards": 3}
        },
        train_args=SimpleNamespace(output_dir="output/run"),
    )

    assert cfg.enabled is True
    assert cfg.max_resident_shards == 3


def test_parse_encoded_sample_cache_config_rejects_nonpositive_residency_bound() -> None:
    with pytest.raises(ValueError, match="max_resident_shards"):
        _parse_encoded_sample_cache_config(
            training_cfg={
                "encoded_sample_cache": {"enabled": True, "max_resident_shards": 0}
            },
            train_args=SimpleNamespace(output_dir="output/run"),
        )


def test_parse_static_packing_cache_config_defaults_to_auto_root() -> None:
    cfg = _parse_static_packing_cache_config({})

    assert cfg == StaticPackingCacheRuntimeConfig(root_dir=None)


def test_parse_static_packing_cache_config_accepts_explicit_root_dir(
    tmp_path: Path,
) -> None:
    cfg = _parse_static_packing_cache_config(
        {"static_packing_cache": {"root_dir": str(tmp_path / "cache_root")}}
    )

    assert cfg.root_dir == str((tmp_path / "cache_root").resolve())




def test_parse_packing_config_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="packing_mode"):
        _parse_packing_config(
            training_cfg={"packing": True, "packing_mode": "foobar"},
            template=_Template(max_length=128),
            train_args=SimpleNamespace(max_model_len=0),
        )


def test_parse_packing_config_warns_on_deprecated_dynamic_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warning_logs: list[str] = []

    def _capture_warning(message: str, *args: object) -> None:
        if args:
            warning_logs.append(message % args)
        else:
            warning_logs.append(message)

    monkeypatch.setattr(sft_module.logger, "warning", _capture_warning)

    cfg = _parse_packing_config(
        training_cfg={"packing": True, "packing_mode": "dynamic"},
        template=_Template(max_length=128),
        train_args=SimpleNamespace(max_model_len=0),
    )

    assert cfg.mode == "dynamic"
    assert any("packing_mode=dynamic is deprecated" in msg for msg in warning_logs)


def test_validate_stage1_static_packing_policy_rejects_dynamic_mode() -> None:
    with pytest.raises(
        ValueError,
        match="deprecated and unsupported for Stage-1",
    ):
        _validate_stage1_static_packing_policy(
            packing_cfg=PackingRuntimeConfig(enabled=True, mode="dynamic"),
            trainer_variant=None,
        )


def test_validate_stage1_static_packing_policy_allows_eval_packing() -> None:
    _validate_stage1_static_packing_policy(
        packing_cfg=PackingRuntimeConfig(
            enabled=True,
            mode="static",
            eval_packing=True,
        ),
        trainer_variant=None,
    )


def test_validate_stage1_static_packing_policy_rejects_unsupported_detection_template() -> None:
    with pytest.raises(ValueError, match="dataset-level static packing only supports"):
        _validate_stage1_static_packing_policy(
            packing_cfg=PackingRuntimeConfig(
                enabled=True,
                mode="static",
                packing_length=128,
            ),
            trainer_variant=None,
            training_config=SimpleNamespace(),
            custom_config=SimpleNamespace(
                detection_sequence_format="compact_no_desc",
                object_ordering="sorted",
            ),
        )


def test_validate_stage1_static_packing_policy_rejects_recursive_objective() -> None:
    for objective_variant in (
        "random_permutation_et_rmp_ce",
        "prefix_rollin_et_rmp_ce",
    ):
        with pytest.raises(
            ValueError,
            match="trie target metadata preservation|target-position offsets",
        ):
            _validate_stage1_static_packing_policy(
                packing_cfg=PackingRuntimeConfig(
                    enabled=True,
                    mode="static",
                    packing_length=128,
                ),
                trainer_variant=None,
                training_config=SimpleNamespace(
                    objective=SimpleNamespace(
                        variant=objective_variant,
                        state_weighting="legacy_row_mean_prefix_mixture_equivalence",
                        normalization="legacy_row_mean_equivalence",
                    )
                ),
                custom_config=SimpleNamespace(
                    detection_sequence_format="compact_full",
                    object_ordering="random",
                ),
            )


def test_validate_stage1_static_packing_policy_skips_rollout_matching_variants() -> None:
    _validate_stage1_static_packing_policy(
        packing_cfg=PackingRuntimeConfig(enabled=True, mode="dynamic"),
        trainer_variant="stage2_rollout_correction",
    )


def test_main_validates_static_packing_before_dataset_materialization() -> None:
    source = Path(sft_module.__file__).read_text(encoding="utf-8")
    main_source = source[source.index("def main(") :]

    dataset_position = main_source.index("dataset = BaseCaptionDataset.from_jsonl")
    assert main_source.index("packing_cfg = _parse_packing_config") < dataset_position
    assert main_source.index("_validate_stage1_static_packing_policy") < dataset_position


def test_append_dataset_epoch_callback_registers_set_epoch_datasets() -> None:
    dataset = _EpochDataset()
    callbacks = _append_dataset_epoch_callback([], dataset)

    assert len(callbacks) == 1
    assert isinstance(callbacks[0], DatasetEpochCallback)

    callbacks[0].on_epoch_begin(
        args=SimpleNamespace(),
        state=SimpleNamespace(epoch=3.0, global_step=0),
        control=SimpleNamespace(),
    )
    assert dataset.epochs == [3]


def test_pipeline_base_callbacks_tolerates_new_swift_pipeline_api() -> None:
    assert _pipeline_base_callbacks(SimpleNamespace()) == []


def test_pipeline_base_callbacks_copies_legacy_callbacks() -> None:
    original = [object()]
    copied = _pipeline_base_callbacks(SimpleNamespace(callbacks=original))

    assert copied == original
    assert copied is not original


def test_pipeline_data_collator_uses_legacy_pipeline_method() -> None:
    collator = object()
    pipeline = SimpleNamespace(_get_data_collator=lambda: collator)

    assert _pipeline_data_collator(pipeline, SimpleNamespace()) is collator


def test_pipeline_data_collator_falls_back_to_template_collator() -> None:
    def template_collator(batch, *, padding_to=None):
        return {"batch": batch, "padding_to": padding_to}

    template = SimpleNamespace(max_length=128, data_collator=template_collator)
    pipeline = SimpleNamespace(template=template)
    train_args = SimpleNamespace(training_args=SimpleNamespace(tuner_type="longlora"))

    collator = _pipeline_data_collator(pipeline, train_args)

    assert collator(["sample"]) == {"batch": ["sample"], "padding_to": 128}


def test_static_packing_fingerprint_includes_dataset_source_identity(
    tmp_path: Path,
) -> None:
    train_jsonl = tmp_path / "train.jsonl"
    train_jsonl.write_text('{"id": 1}\n', encoding="utf-8")

    packing_cfg = _parse_packing_config(
        training_cfg={"packing": True, "packing_mode": "static"},
        template=_Template(max_length=128),
        train_args=SimpleNamespace(max_model_len=0),
    )

    fingerprint = _build_static_packing_fingerprint(
        training_config=SimpleNamespace(
            global_max_length=1024,
            template={"system": "sys", "truncation_strategy": "raise"},
            training={"train_dataloader_shuffle": True},
        ),
        custom_config=SimpleNamespace(
            user_prompt="prompt",
            emit_norm="none",
            json_format="standard",
            object_ordering="none",
            object_field_order="geometry_first",
            use_summary=False,
            system_prompt_dense=None,
            system_prompt_summary=None,
        ),
        template=_Template(max_length=128),
        train_args=SimpleNamespace(max_model_len=512),
        dataset_seed=7,
        packing_cfg=packing_cfg,
        train_jsonl=str(train_jsonl),
        train_sample_limit=32,
    )

    assert fingerprint["dataset_split"] == "train"
    assert fingerprint["train_sample_limit"] == 32
    assert fingerprint["dataset_jsonl"] == str(train_jsonl)
    assert fingerprint["custom_train_jsonl"] == str(train_jsonl)
    contract = fingerprint["detection_packing_contract"]
    assert contract["schema_version"] == "detection_packing_fingerprint_v1"
    assert contract["metadata"]["template_id"] == "stage1_json_pretty"
    assert contract["metadata"]["objective_variant"] == "sorted_sft"
    assert contract["metadata"]["normalization_policy"] == "token_mean"
    source = fingerprint["dataset_source_jsonl"]
    assert isinstance(source, dict)
    assert source["raw_path"] == str(train_jsonl)
    assert source["resolved_path"] == str(train_jsonl.resolve())
    assert source["exists"] is True
    assert source["size_bytes"] == train_jsonl.stat().st_size


def test_static_packing_fingerprint_tracks_eval_split_inputs(
    tmp_path: Path,
) -> None:
    val_jsonl = tmp_path / "val.jsonl"
    val_jsonl.write_text('{"id": 1}\n', encoding="utf-8")

    packing_cfg = _parse_packing_config(
        training_cfg={"packing": True, "packing_mode": "static"},
        template=_Template(max_length=128),
        train_args=SimpleNamespace(max_model_len=0),
    )

    fingerprint = _build_static_packing_fingerprint(
        training_config=SimpleNamespace(
            global_max_length=1024,
            template={"system": "sys", "truncation_strategy": "raise"},
            training={"train_dataloader_shuffle": True},
        ),
        custom_config=SimpleNamespace(
            user_prompt="prompt",
            emit_norm="none",
            json_format="standard",
            object_ordering="none",
            object_field_order="geometry_first",
            use_summary=False,
            system_prompt_dense=None,
            system_prompt_summary=None,
        ),
        template=_Template(max_length=128),
        train_args=SimpleNamespace(max_model_len=512),
        dataset_seed=7,
        packing_cfg=packing_cfg,
        train_jsonl=str(val_jsonl),
        dataset_split="eval",
        eval_sample_limit=99,
        eval_sample_with_replacement=False,
    )

    assert fingerprint["dataset_split"] == "eval"
    assert fingerprint["eval_sample_limit"] == 99
    assert fingerprint["eval_sample_with_replacement"] is False


def test_static_packing_fingerprint_tracks_offline_pixels_and_coord_tokens() -> None:
    packing_cfg = _parse_packing_config(
        training_cfg={"packing": True, "packing_mode": "static"},
        template=_Template(max_length=128),
        train_args=SimpleNamespace(max_model_len=0),
    )

    common_training = SimpleNamespace(
        global_max_length=1024,
        template={"system": "sys", "truncation_strategy": "raise"},
        training={"train_dataloader_shuffle": True},
    )
    base_custom = dict(
        user_prompt="prompt",
        emit_norm="none",
        json_format="standard",
        object_ordering="none",
        object_field_order="geometry_first",
        use_summary=False,
        system_prompt_dense=None,
        system_prompt_summary=None,
    )

    fingerprint_a = _build_static_packing_fingerprint(
        training_config=common_training,
        custom_config=SimpleNamespace(
            **base_custom,
            offline_max_pixels=1048576,
            coord_tokens={"enabled": True, "skip_bbox_norm": True},
        ),
        template=_Template(max_length=128),
        train_args=SimpleNamespace(max_model_len=512),
        dataset_seed=7,
        packing_cfg=packing_cfg,
        train_jsonl="train.jsonl",
    )
    fingerprint_b = _build_static_packing_fingerprint(
        training_config=common_training,
        custom_config=SimpleNamespace(
            **base_custom,
            offline_max_pixels=2097152,
            coord_tokens={"enabled": False},
        ),
        template=_Template(max_length=128),
        train_args=SimpleNamespace(max_model_len=512),
        dataset_seed=7,
        packing_cfg=packing_cfg,
        train_jsonl="train.jsonl",
    )

    assert fingerprint_a["custom_offline_max_pixels"] == 1048576
    assert fingerprint_a["coord_tokens"] == {
        "enabled": True,
        "skip_bbox_norm": True,
    }
    assert fingerprint_a["custom_coord_mode"] == "coord_tokens"
    assert fingerprint_b["custom_offline_max_pixels"] == 2097152
    assert fingerprint_b["coord_tokens"] == {"enabled": False}
    assert fingerprint_b["custom_coord_mode"] == "norm1000_text"
    assert fingerprint_a != fingerprint_b


def test_static_packing_fingerprint_tracks_bbox_parameterization() -> None:
    packing_cfg = _parse_packing_config(
        training_cfg={"packing": True, "packing_mode": "static"},
        template=_Template(max_length=128),
        train_args=SimpleNamespace(max_model_len=0),
    )

    training_cfg = SimpleNamespace(
        global_max_length=1024,
        template={"system": "sys", "truncation_strategy": "raise"},
        training={"train_dataloader_shuffle": True},
    )
    common_custom = dict(
        user_prompt="prompt",
        emit_norm="none",
        json_format="standard",
        object_ordering="sorted",
        object_field_order="desc_first",
        use_summary=False,
        system_prompt_dense=None,
        system_prompt_summary=None,
        offline_max_pixels=1048576,
        coord_tokens={"enabled": True, "skip_bbox_norm": True},
    )

    xyxy = _build_static_packing_fingerprint(
        training_config=training_cfg,
        custom_config=SimpleNamespace(**common_custom, bbox_format="xyxy"),
        template=_Template(max_length=128),
        train_args=SimpleNamespace(max_model_len=512),
        dataset_seed=7,
        packing_cfg=packing_cfg,
        train_jsonl="train.jsonl",
    )
    cxcy_logw_logh = _build_static_packing_fingerprint(
        training_config=training_cfg,
        custom_config=SimpleNamespace(
            **common_custom, bbox_format="cxcy_logw_logh"
        ),
        template=_Template(max_length=128),
        train_args=SimpleNamespace(max_model_len=512),
        dataset_seed=7,
        packing_cfg=packing_cfg,
        train_jsonl="train.jsonl",
    )
    cxcywh = _build_static_packing_fingerprint(
        training_config=training_cfg,
        custom_config=SimpleNamespace(
            **common_custom, bbox_format="cxcywh"
        ),
        template=_Template(max_length=128),
        train_args=SimpleNamespace(max_model_len=512),
        dataset_seed=7,
        packing_cfg=packing_cfg,
        train_jsonl="train.jsonl",
    )

    assert xyxy["custom_bbox_format"] == "xyxy"
    assert cxcy_logw_logh["custom_bbox_format"] == "cxcy_logw_logh"
    assert cxcywh["custom_bbox_format"] == "cxcywh"
    assert xyxy != cxcy_logw_logh
    assert cxcy_logw_logh != cxcywh


def test_static_packing_fingerprint_tracks_detection_sequence_format() -> None:
    packing_cfg = _parse_packing_config(
        training_cfg={"packing": True, "packing_mode": "static"},
        template=_Template(max_length=128),
        train_args=SimpleNamespace(max_model_len=0),
    )

    training_cfg = SimpleNamespace(
        global_max_length=1024,
        template={"system": "sys", "truncation_strategy": "raise"},
        training={"train_dataloader_shuffle": True},
    )
    common_custom = dict(
        user_prompt="prompt",
        emit_norm="none",
        json_format="standard",
        bbox_format="xyxy",
        object_ordering="sorted",
        object_field_order="desc_first",
        use_summary=False,
        system_prompt_dense=None,
        system_prompt_summary=None,
        offline_max_pixels=1048576,
        coord_tokens={"enabled": True, "skip_bbox_norm": True},
    )

    coordjson = _build_static_packing_fingerprint(
        training_config=training_cfg,
        custom_config=SimpleNamespace(
            **common_custom,
            detection_sequence_format="coordjson",
        ),
        template=_Template(max_length=128),
        train_args=SimpleNamespace(max_model_len=512),
        dataset_seed=7,
        packing_cfg=packing_cfg,
        train_jsonl="train.jsonl",
    )
    compact = _build_static_packing_fingerprint(
        training_config=training_cfg,
        custom_config=SimpleNamespace(
            **common_custom,
            detection_sequence_format="compact_full",
        ),
        template=_Template(max_length=128),
        train_args=SimpleNamespace(max_model_len=512),
        dataset_seed=7,
        packing_cfg=packing_cfg,
        train_jsonl="train.jsonl",
    )

    assert coordjson["custom_detection_sequence_format"] == "stage1_json_pretty"
    assert compact["custom_detection_sequence_format"] == "compact"
    assert coordjson != compact


def test_static_packing_fingerprint_canonicalizes_legacy_compact_format() -> None:
    packing_cfg = _parse_packing_config(
        training_cfg={"packing": True, "packing_mode": "static"},
        template=_Template(max_length=128),
        train_args=SimpleNamespace(max_model_len=0),
    )
    training_cfg = SimpleNamespace(
        global_max_length=1024,
        template={"system": "sys", "truncation_strategy": "raise"},
        training={"train_dataloader_shuffle": True},
    )
    common_custom = dict(
        user_prompt="prompt",
        emit_norm="none",
        json_format="standard",
        bbox_format="xyxy",
        object_ordering="sorted",
        object_field_order="desc_first",
        use_summary=False,
        system_prompt_dense=None,
        system_prompt_summary=None,
        offline_max_pixels=1048576,
        coord_tokens={"enabled": True, "skip_bbox_norm": True},
    )

    compact = _build_static_packing_fingerprint(
        training_config=training_cfg,
        custom_config=SimpleNamespace(
            **common_custom,
            detection_sequence_format="compact",
        ),
        template=_Template(max_length=128),
        train_args=SimpleNamespace(max_model_len=512),
        dataset_seed=7,
        packing_cfg=packing_cfg,
        train_jsonl="train.jsonl",
    )
    compact_full = _build_static_packing_fingerprint(
        training_config=training_cfg,
        custom_config=SimpleNamespace(
            **common_custom,
            detection_sequence_format="compact_full",
        ),
        template=_Template(max_length=128),
        train_args=SimpleNamespace(max_model_len=512),
        dataset_seed=7,
        packing_cfg=packing_cfg,
        train_jsonl="train.jsonl",
    )

    assert compact["custom_detection_sequence_format"] == "compact"
    assert compact == compact_full


def test_static_packing_fingerprint_tracks_prompt_variant_and_template_hash() -> None:
    packing_cfg = _parse_packing_config(
        training_cfg={"packing": True, "packing_mode": "static"},
        template=_Template(max_length=128),
        train_args=SimpleNamespace(max_model_len=0),
    )

    training_cfg = SimpleNamespace(
        global_max_length=1024,
        template={"system": "sys", "truncation_strategy": "raise"},
        training={"train_dataloader_shuffle": True},
    )
    common_custom = dict(
        user_prompt="prompt",
        emit_norm="none",
        json_format="standard",
        bbox_format="cxcy_logw_logh",
        object_ordering="sorted",
        object_field_order="geometry_first",
        use_summary=False,
        system_prompt_dense=None,
        system_prompt_summary=None,
        offline_max_pixels=1048576,
        coord_tokens={"enabled": True, "skip_bbox_norm": True},
    )

    default_fp = _build_static_packing_fingerprint(
        training_config=training_cfg,
        custom_config=SimpleNamespace(
            **common_custom,
            extra={"prompt_variant": "default"},
        ),
        template=_Template(max_length=128),
        train_args=SimpleNamespace(max_model_len=512),
        dataset_seed=7,
        packing_cfg=packing_cfg,
        train_jsonl="train.jsonl",
    )
    lvis_fp = _build_static_packing_fingerprint(
        training_config=training_cfg,
        custom_config=SimpleNamespace(
            **common_custom,
            extra={"prompt_variant": "lvis_stage1_federated"},
        ),
        template=_Template(max_length=128),
        train_args=SimpleNamespace(max_model_len=512),
        dataset_seed=7,
        packing_cfg=packing_cfg,
        train_jsonl="train.jsonl",
    )

    assert default_fp["custom_prompt_variant"] == "default"
    assert lvis_fp["custom_prompt_variant"] == "lvis_stage1_federated"
    assert isinstance(default_fp["custom_prompt_template_hash"], str)
    assert isinstance(lvis_fp["custom_prompt_template_hash"], str)
    assert default_fp["custom_prompt_template_hash"] != lvis_fp["custom_prompt_template_hash"]


def test_static_packing_fingerprint_tracks_template_and_field_order_axes() -> None:
    packing_cfg = _parse_packing_config(
        training_cfg={"packing": True, "packing_mode": "static"},
        template=_Template(max_length=128),
        train_args=SimpleNamespace(max_model_len=0),
    )
    common_training = dict(
        global_max_length=12000,
        template={"system": "sys", "truncation_strategy": "raise"},
        training={"train_dataloader_shuffle": True},
    )
    common_custom = dict(
        user_prompt="prompt",
        emit_norm="none",
        json_format="standard",
        bbox_format="xyxy",
        object_ordering="sorted",
        detection_sequence_format="compact",
        use_summary=False,
        system_prompt_dense=None,
        system_prompt_summary=None,
        offline_max_pixels=1048576,
        coord_tokens={"enabled": True, "skip_bbox_norm": True},
        extra={"prompt_variant": "default"},
    )

    def fingerprint(
        *,
        template_id: str = "compact_object_box_closed",
        object_field_order: str = "desc_first",
        global_max_length: int = 12000,
        prompt_variant: str = "default",
        packing_length: int = 128,
    ) -> dict[str, object]:
        local_packing_cfg = _parse_packing_config(
            training_cfg={"packing": True, "packing_mode": "static"},
            template=_Template(max_length=packing_length),
            train_args=SimpleNamespace(max_model_len=0),
        )
        training_cfg = SimpleNamespace(
            **{
                **common_training,
                "global_max_length": global_max_length,
                "detection_template": {"id": template_id},
            }
        )
        custom_cfg = SimpleNamespace(
            **{
                **common_custom,
                "object_field_order": object_field_order,
                "extra": {"prompt_variant": prompt_variant},
            }
        )
        return _build_static_packing_fingerprint(
            training_config=training_cfg,
            custom_config=custom_cfg,
            template=_Template(max_length=128),
            train_args=SimpleNamespace(max_model_len=512),
            dataset_seed=7,
            packing_cfg=local_packing_cfg,
            train_jsonl="train.jsonl",
        )

    baseline = fingerprint()
    geometry_first = fingerprint(object_field_order="geometry_first")
    object_closed = fingerprint(template_id="compact_object_closed")
    longer = fingerprint(global_max_length=16000)
    longer_packing = fingerprint(packing_length=256)
    prompt_changed = fingerprint(prompt_variant="lvis_stage1_federated")

    assert baseline["detection_template_id"] == "compact_object_box_closed"
    assert baseline["custom_object_field_order"] == "desc_first"
    assert baseline["global_max_length"] == 12000
    assert baseline["packing_length"] == 128
    assert baseline["custom_prompt_template_hash"]
    assert geometry_first["custom_object_field_order"] == "geometry_first"
    assert object_closed["detection_template_id"] == "compact_object_closed"
    assert longer["global_max_length"] == 16000
    assert longer_packing["packing_length"] == 256
    assert prompt_changed["custom_prompt_template_hash"] != baseline["custom_prompt_template_hash"]
    assert baseline != geometry_first
    assert baseline != object_closed
    assert baseline != longer
    assert baseline != longer_packing
    assert baseline != prompt_changed


def test_static_packing_fingerprint_tracks_normalized_hierarchy_axes() -> None:
    common_custom = dict(
        user_prompt="prompt",
        emit_norm="none",
        json_format="standard",
        bbox_format="xyxy",
        object_ordering="sorted",
        detection_sequence_format="compact",
        use_summary=False,
        system_prompt_dense=None,
        system_prompt_summary=None,
        offline_max_pixels=1048576,
        coord_tokens={"enabled": True, "skip_bbox_norm": True},
        extra={"prompt_variant": "default"},
    )

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
        packing_length: int = 128,
    ) -> dict[str, object]:
        packing_cfg = _parse_packing_config(
            training_cfg={"packing": True, "packing_mode": "static"},
            template=_Template(max_length=packing_length),
            train_args=SimpleNamespace(max_model_len=0),
        )
        training_cfg = SimpleNamespace(
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
            training={"train_dataloader_shuffle": True},
        )
        custom_cfg = SimpleNamespace(
            **{
                **common_custom,
                "object_ordering": object_ordering,
                "object_field_order": object_field_order,
                "bbox_format": bbox_format,
                "extra": {"prompt_variant": prompt_variant},
            }
        )
        return _build_static_packing_fingerprint(
            training_config=training_cfg,
            custom_config=custom_cfg,
            template=_Template(max_length=128),
            train_args=SimpleNamespace(model="model-a", max_model_len=512),
            dataset_seed=7,
            packing_cfg=packing_cfg,
            train_jsonl="train.jsonl",
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
    packing_changed = fingerprint(packing_length=256)

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
    assert baseline != packing_changed


def test_fingerprint_diff_keys_reports_missing_vs_null() -> None:
    differing = _fingerprint_diff_keys(
        {"custom_prompt_variant": None},
        {},
    )

    assert differing == ["custom_prompt_variant"]


def test_resolve_static_packing_cache_dir_defaults_to_dataset_local_root(
    tmp_path: Path,
) -> None:
    train_jsonl = tmp_path / "dataset" / "train.jsonl"
    train_jsonl.parent.mkdir(parents=True, exist_ok=True)
    train_jsonl.write_text('{"id": 1}\n', encoding="utf-8")

    packing_cfg = _parse_packing_config(
        training_cfg={"packing": True, "packing_mode": "static"},
        template=_Template(max_length=128),
        train_args=SimpleNamespace(max_model_len=0),
    )

    cache_dir = _resolve_static_packing_cache_dir(
        runtime_cfg=StaticPackingCacheRuntimeConfig(),
        training_config=SimpleNamespace(global_max_length=12000),
        train_args=SimpleNamespace(output_dir=str(tmp_path / "out_v003")),
        dataset_jsonl=str(train_jsonl),
        dataset_split="train",
        packing_cfg=packing_cfg,
    )

    assert cache_dir == (
        train_jsonl.parent
        / "cache"
        / "static_packing"
        / "global_max_length_12000"
        / "train"
    ).resolve()


@pytest.mark.parametrize(
    ("config_relpath", "expected_ordering"),
    [
        ("configs/stage1/lvis_bbox_max60_1024.yaml", "sorted"),
        ("configs/stage1/smoke/lvis_bbox_max60_1024.yaml", "sorted"),
    ],
)
def test_stage1_profiles_pin_cache_parity_and_ordering(
    config_relpath: str,
    expected_ordering: str,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = ConfigLoader.load_materialized_training_config(str(repo_root / config_relpath))

    assert cfg.training["seed"] == 17
    assert cfg.custom.object_ordering == expected_ordering


def test_lvis_stage1_config_keeps_canonical_recipe_and_desc_first_sorted_contract() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = ConfigLoader.load_materialized_training_config(
        str(repo_root / "configs/stage1/lvis_bbox_max60_1024.yaml")
    )

    assert cfg.training["optimizer"] == "multimodal_token_embeddings_adapter"
    assert cfg.training["run_name"] == "epoch_4-hard_ce_soft_ce_w1-2b"
    assert cfg.custom.train_jsonl == "public_data/lvis/rescale_32_1024_bbox_max60/train.coord.jsonl"
    assert cfg.custom.val_jsonl == "public_data/lvis/rescale_32_1024_bbox_max60/val.coord.jsonl"
    assert cfg.custom.val_sample_limit == 512
    assert cfg.custom.dump_conversation_text is True
    assert cfg.custom.object_ordering == "sorted"
    assert cfg.custom.object_field_order == "desc_first"
    assert cfg.custom.extra["prompt_variant"] == "lvis_stage1_federated"
    resolved_model_checkpoint = _resolve_model_checkpoint_path(cfg)
    assert resolved_model_checkpoint == str(cfg.model["model"]).strip()
    assert resolved_model_checkpoint
    assert cfg.custom.coord_soft_ce_w1.enabled is True
    assert cfg.custom.coord_soft_ce_w1.ce_weight == pytest.approx(1.0)
    assert cfg.custom.coord_soft_ce_w1.soft_ce_weight == pytest.approx(1.0)

    assert cfg.custom.coord_soft_ce_w1.w1_weight == pytest.approx(1.0)
    assert cfg.custom.coord_soft_ce_w1.gate_weight == pytest.approx(5.0)
    assert cfg.custom.bbox_geo.enabled is False
    assert cfg.custom.bbox_size_aux.enabled is False
    assert cfg.training["artifact_subdir"] == "stage1/lvis_bbox_max60_1024_coord_softce_w1"
    assert cfg.training["output_dir"] == "./output/stage1/lvis_bbox_max60_1024_coord_softce_w1"
    assert cfg.training["logging_dir"] == "./tb/stage1/lvis_bbox_max60_1024_coord_softce_w1"


def test_object_ref_close_box_close_standard_sft_config_contract() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    configs = {
        "desc_first_prod": ConfigLoader.load_materialized_training_config(
            str(
                repo_root
                / "configs/stage1/profiles/2b/pure_ce_coco80_desc_first_1024_object_ref_close_box_close_sorted_packed_natural_adjacent.yaml"
            )
        ),
        "desc_first_smoke": ConfigLoader.load_materialized_training_config(
            str(
                repo_root
                / "configs/stage1/smoke/pure_ce_coco80_desc_first_1024_object_ref_close_box_close_sorted_packed_natural_adjacent_tiny.yaml"
            )
        ),
        "geometry_first_prod": ConfigLoader.load_materialized_training_config(
            str(
                repo_root
                / "configs/stage1/profiles/2b/pure_ce_coco80_geometry_first_1024_object_ref_close_box_close_sorted_packed_natural_adjacent.yaml"
            )
        ),
        "geometry_first_smoke": ConfigLoader.load_materialized_training_config(
            str(
                repo_root
                / "configs/stage1/smoke/pure_ce_coco80_geometry_first_1024_object_ref_close_box_close_sorted_packed_natural_adjacent_tiny.yaml"
            )
        ),
    }

    for name, cfg in configs.items():
        assert isinstance(cfg, DetectionTrainingConfig)
        assert cfg.global_max_length == 12000
        assert cfg.template["max_length"] == 12000
        assert cfg.model["model"] == (
            "model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent"
        )
        assert cfg.pipeline.id == "stage1_standard_sft"
        assert cfg.objective.id == "standard_ce"
        assert cfg.objective.auxiliaries.coord_soft_ce.enabled is False
        assert cfg.objective.auxiliaries.geometry.enabled is False
        assert cfg.sample_factory.target_sequence.object_ordering == "sorted"
        expected_field_order = (
            "geometry_first" if "geometry_first" in name else "desc_first"
        )
        assert cfg.sample_factory.target_sequence.object_field_order == (
            expected_field_order
        )
        assert cfg.detection_template.id == "compact_object_box_closed"
        assert cfg.evaluation.expected_template == "compact_object_box_closed"
        assert cfg.training["packing"] is True
        assert cfg.training["packing_mode"] == "static"
        assert cfg.training["eval_packing"] is True
        assert cfg.tuner["freeze_llm"] is False
        assert cfg.tuner["freeze_vit"] is True
        assert cfg.tuner["freeze_aligner"] is True
        assert cfg.token_embeddings_adapter.enabled is True
        assert set(cfg.token_embeddings_adapter.groups) == {
            "coord_geometry",
            "compact_structure",
        }
        structural = cfg.token_embeddings_adapter.groups["compact_structure"]
        assert structural.tokens == (
            "<|object_ref_start|>",
            "<|object_ref_end|>",
            "<|box_start|>",
            "<|box_end|>",
        )
        assert dict(structural.expected_ids) == {
            "<|object_ref_start|>": 151646,
            "<|object_ref_end|>": 151647,
            "<|box_start|>": 151648,
            "<|box_end|>": 151649,
        }
        assert "custom" not in cfg.to_mapping()

    assert configs["desc_first_prod"].training["num_train_epochs"] == 4
    assert configs["geometry_first_prod"].training["num_train_epochs"] == 4
    assert configs["desc_first_smoke"].training["max_steps"] == 2
    assert configs["geometry_first_smoke"].training["max_steps"] == 2
    assert configs["desc_first_smoke"].debug.train_sample_limit == 32
    assert configs["geometry_first_smoke"].debug.val_sample_limit == 8


def test_coord_gaussian_rps_desc_first_standard_sft_config_contract() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    configs = {
        "prod": ConfigLoader.load_materialized_training_config(
            str(
                repo_root
                / "configs/stage1/profiles/2b/coord_gaussian_rps_coco80_desc_first_1024_object_ref_close_box_close_sorted_packed_natural_adjacent.yaml"
            )
        ),
        "smoke": ConfigLoader.load_materialized_training_config(
            str(
                repo_root
                / "configs/stage1/smoke/coord_gaussian_rps_coco80_desc_first_1024_object_ref_close_box_close_sorted_packed_natural_adjacent_tiny.yaml"
            )
        ),
        "launch_smoke": ConfigLoader.load_materialized_training_config(
            str(
                repo_root
                / "configs/stage1/smoke/coord_gaussian_rps_coco80_desc_first_1024_object_ref_close_box_close_sorted_packed_natural_adjacent_8gpu_smoke.yaml"
            )
        ),
    }

    for cfg in configs.values():
        assert isinstance(cfg, DetectionTrainingConfig)
        assert cfg.pipeline.id == "stage1_standard_sft"
        assert cfg.objective.id == "standard_ce"
        assert cfg.objective.auxiliaries.coord_soft_ce.enabled is False
        aux = cfg.objective.auxiliaries.coord_gaussian_rps
        assert aux.enabled is True
        assert aux.ce_weight == pytest.approx(1.0)
        assert aux.gaussian_weight == pytest.approx(0.5)
        assert aux.rps_weight == pytest.approx(0.2)
        assert aux.gaussian_r95_axis_fraction == pytest.approx(0.04)
        assert aux.gaussian_r95_cap_bins == 8
        assert aux.gaussian_r95_min_bins == 1
        assert aux.gaussian_r95_fallback_bins == 8
        assert aux.type_gate.enabled is True
        assert aux.type_gate.mode == "allowed_type_mass"
        assert aux.type_gate.weights.struct == pytest.approx(1.0)
        assert aux.type_gate.weights.coord == pytest.approx(1.0)
        assert aux.type_gate.weights.desc == pytest.approx(1.0)
        assert aux.type_gate.weights.eos == pytest.approx(1.0)
        assert cfg.sample_factory.target_sequence.object_ordering == "sorted"
        assert cfg.sample_factory.target_sequence.object_field_order == "desc_first"
        assert cfg.sample_factory.target_sequence.bbox_format == "xyxy"
        assert cfg.detection_template.id == "compact_object_box_closed"
        assert cfg.evaluation.expected_template == "compact_object_box_closed"
        assert cfg.data.image_root == "public_data/coco/rescale_32_1024_bbox_max60"

    assert configs["prod"].training["num_train_epochs"] == 4
    assert configs["prod"].training["logging_steps"] == 10
    assert configs["prod"].training["eval_steps"] == 100
    assert configs["smoke"].training["max_steps"] == 2
    assert configs["smoke"].debug.train_sample_limit == 32
    assert configs["smoke"].debug.val_sample_limit == 8
    assert configs["launch_smoke"].training["max_steps"] == 12
    assert configs["launch_smoke"].training["eval_steps"] == 4
    assert configs["launch_smoke"].training["save_steps"] == 4
    assert configs["launch_smoke"].training["save_strategy"] == "best"
    assert configs["launch_smoke"].training["save_delay_steps"] == 0
    assert configs["launch_smoke"].debug.train_sample_limit == 512
    assert configs["launch_smoke"].debug.val_sample_limit == 128
    assert configs["launch_smoke"].global_max_length == 6000
    assert configs["launch_smoke"].template["max_length"] == 6000


def test_lvis_stage1_smoke_config_only_overrides_runtime_limits() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = ConfigLoader.load_materialized_training_config(
        str(repo_root / "configs/stage1/smoke/lvis_bbox_max60_1024.yaml")
    )

    assert cfg.custom.train_jsonl == "public_data/lvis/rescale_32_1024_bbox_max60/train.coord.jsonl"
    assert cfg.custom.val_jsonl == "public_data/lvis/rescale_32_1024_bbox_max60/val.coord.jsonl"
    assert cfg.custom.object_ordering == "sorted"
    assert cfg.custom.object_field_order == "desc_first"
    assert cfg.custom.extra["prompt_variant"] == "lvis_stage1_federated"
    assert cfg.custom.coord_soft_ce_w1.ce_weight == pytest.approx(1.0)
    assert cfg.custom.coord_soft_ce_w1.soft_ce_weight == pytest.approx(1.0)
    assert cfg.custom.coord_soft_ce_w1.w1_weight == pytest.approx(1.0)
    assert cfg.custom.bbox_geo.enabled is False
    assert cfg.custom.bbox_size_aux.enabled is False
    assert cfg.training["max_steps"] == 2
    assert cfg.custom.train_sample_limit == 32
    assert cfg.custom.val_sample_limit == 8
    assert (
        cfg.training["run_name"]
        == "smoke_2steps-stage1-lvis_bbox_max60_1024-hard_ce_soft_ce_w1"
    )
    assert cfg.training["artifact_subdir"] == "stage1/smoke/lvis_bbox_max60_1024_coord_softce_w1"
    assert cfg.training["output_dir"] == "./output/stage1/smoke/lvis_bbox_max60_1024_coord_softce_w1"
    assert cfg.training["logging_dir"] == "./tb/stage1/smoke/lvis_bbox_max60_1024_coord_softce_w1"


def test_stage2_rollout_correction_prod_config_keeps_residual_pipeline_contract() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = ConfigLoader.load_materialized_training_config(
        str(
            repo_root
            / "configs/stage2/rollout_correction/prod/coco1024_online_residual_correction_vllm_tail_append.yaml"
        )
    )

    assert cfg.pipeline.id == "stage2_rollout_correction"
    assert not hasattr(cfg, "custom")
    assert cfg.rollout_matching.eval_detection.metrics == "coco"
    assert "stage2_rollout_correction" in cfg.training["output_dir"]
    objective = {
        module.name: module
        for module in cfg.stage2_rollout_correction.pipeline.objective
    }
    assert list(objective) == ["residual_set_correction"]
    assert objective["residual_set_correction"].application["preset"] == "rollout_self_prefix"


def test_stage2_rollout_correction_smoke_config_resolves_residual_pipeline() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = ConfigLoader.load_materialized_training_config(
        str(
            repo_root
            / "configs/stage2/rollout_correction/smoke/compact_full_hf_1step.yaml"
        )
    )

    objective = {
        module.name: module
        for module in cfg.stage2_rollout_correction.pipeline.objective
    }
    assert cfg.training["run_name"] == "compact_full_hf_1step"
    assert "stage2_rollout_correction" in cfg.training["output_dir"]
    assert list(objective) == ["residual_set_correction"]


def test_representative_raw_leaves_still_author_model_run_name_and_artifact_subdir() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    raw_paths = [
        repo_root / "configs/stage1/lvis_bbox_max60_1024.yaml",
        repo_root
        / "configs/stage2/rollout_correction/prod/coco1024_online_residual_correction_vllm_tail_append.yaml",
    ]

    for path in raw_paths:
        payload = ConfigLoader.load_yaml_with_extends(str(path))
        payload = ConfigLoader._materialize_training_artifact_paths(payload)
        assert payload["model"]["model"]
        assert payload["training"]["run_name"]
        assert payload["training"]["output_dir"]


def test_shared_dataset_and_prompt_facets_materialize_through_minimal_stage1_leaf(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    leaf = tmp_path / "stage1_leaf.yaml"
    leaf.write_text(
        yaml.safe_dump(
            {
                "extends": [
                    str(
                        repo_root
                        / "configs/stage1/_shared/coord_soft_ce_gate_4b.yaml"
                    ),
                    str(
                        repo_root
                        / "configs/_shared/datasets/coco_768_bbox_max60.yaml"
                    ),
                    str(
                        repo_root
                        / "configs/_shared/prompts/coco80_desc_first.yaml"
                    ),
                ],
                "training": {
                    "run_name": "toy-stage1-coco768",
                    "artifact_subdir": "stage1/test/coco768",
                },
                "model": {
                    "model": "model_cache/models/Qwen/Qwen3-VL-4B-Instruct-coordexp"
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = ConfigLoader.load_materialized_training_config(str(leaf))

    assert (
        cfg.custom.train_jsonl
        == "public_data/coco/rescale_32_768_bbox_max60/train.coord.jsonl"
    )
    assert (
        cfg.custom.val_jsonl
        == "public_data/coco/rescale_32_768_bbox_max60/val.coord.jsonl"
    )
    assert cfg.custom.object_ordering == "sorted"
    assert cfg.custom.object_field_order == "desc_first"
    assert cfg.custom.extra["prompt_variant"] == "coco_80"
    assert cfg.template["max_pixels"] == 10485760
    assert cfg.custom.offline_max_pixels == 10485760
    assert cfg.training["artifact_subdir"] == "stage1/test/coco768"
    assert cfg.training["output_dir"] == "./output/stage1/test/coco768"
    assert cfg.training["logging_dir"] == "./tb/stage1/test/coco768"


def test_future_dataset_facet_can_use_existing_typed_keys_without_loader_changes(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_facet = tmp_path / "future_dataset.yaml"
    dataset_facet.write_text(
        yaml.safe_dump(
            {
                "template": {"max_pixels": 123456},
                "custom": {
                    "train_jsonl": "public_data/future/train.coord.jsonl",
                    "val_jsonl": "public_data/future/val.coord.jsonl",
                    "offline_max_pixels": 123456,
                    "object_ordering": "sorted",
                },
            }
        ),
        encoding="utf-8",
    )
    leaf = tmp_path / "future_leaf.yaml"
    leaf.write_text(
        yaml.safe_dump(
            {
                "extends": [
                    str(
                        repo_root
                        / "configs/stage1/_shared/coord_soft_ce_gate_4b.yaml"
                    ),
                    str(dataset_facet),
                    str(
                        repo_root
                        / "configs/_shared/prompts/coco80_geometry_first.yaml"
                    ),
                ],
                "training": {
                    "run_name": "toy-stage1-future",
                    "artifact_subdir": "stage1/test/future",
                },
                "model": {
                    "model": "model_cache/models/Qwen/Qwen3-VL-4B-Instruct-coordexp"
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = ConfigLoader.load_materialized_training_config(str(leaf))

    assert cfg.custom.train_jsonl == "public_data/future/train.coord.jsonl"
    assert cfg.custom.val_jsonl == "public_data/future/val.coord.jsonl"
    assert cfg.custom.offline_max_pixels == 123456
    assert cfg.template["max_pixels"] == 123456
    assert cfg.custom.object_ordering == "sorted"
    assert cfg.custom.object_field_order == "geometry_first"
    assert cfg.custom.extra["prompt_variant"] == "coco_80"


def test_recompute_gas_for_packing_uses_effective_batch_size() -> None:
    nested = SimpleNamespace(gradient_accumulation_steps=2)
    args = SimpleNamespace(
        gradient_accumulation_steps=2,
        training_args=nested,
    )

    new_gas = _recompute_gas_for_packing(
        train_args=args,
        training_cfg={"effective_batch_size": 32},
        original_per_device_bs=4,
        original_gas=2,
        world_size=2,
    )

    assert new_gas == 16
    assert args.gradient_accumulation_steps == 16
    assert nested.gradient_accumulation_steps == 16


def test_recompute_gas_for_packing_preserves_implied_global_batch() -> None:
    args = SimpleNamespace(
        gradient_accumulation_steps=3,
        training_args=None,
    )

    new_gas = _recompute_gas_for_packing(
        train_args=args,
        training_cfg={},
        original_per_device_bs=4,
        original_gas=3,
        world_size=2,
    )

    assert new_gas == 12
    assert args.gradient_accumulation_steps == 12


def test_recompute_gas_for_packing_logs_requested_vs_realized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = SimpleNamespace(
        gradient_accumulation_steps=1,
        training_args=None,
    )
    info_logs: list[str] = []

    def _capture_info(message: str, *args: object) -> None:
        if args:
            info_logs.append(message % args)
        else:
            info_logs.append(message)

    monkeypatch.setattr(sft_module.logger, "info", _capture_info)

    new_gas = _recompute_gas_for_packing(
        train_args=args,
        training_cfg={"effective_batch_size": 8},
        original_per_device_bs=1,
        original_gas=1,
        world_size=2,
    )

    assert new_gas == 4
    assert any(
        "requested_global_packs_per_step=8" in msg for msg in info_logs
    )
    assert any(
        "realized_global_packs_per_step=8" in msg for msg in info_logs
    )


def test_recompute_gas_for_packing_raises_when_effective_batch_not_divisible_by_world_size() -> None:
    args = SimpleNamespace(
        gradient_accumulation_steps=1,
        training_args=None,
    )

    with pytest.raises(
        ValueError,
        match="must be divisible by world_size",
    ):
        _recompute_gas_for_packing(
            train_args=args,
            training_cfg={"effective_batch_size": 7},
            original_per_device_bs=1,
            original_gas=1,
            world_size=3,
        )


def test_validate_static_packing_accumulation_windows_warns_on_partial_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warning_logs: list[str] = []

    def _capture_warning(message: str, *args: object) -> None:
        if args:
            warning_logs.append(message % args)
        else:
            warning_logs.append(message)

    monkeypatch.setattr(sft_module.logger, "warning", _capture_warning)

    _validate_static_packing_accumulation_windows(
        packing_cfg=PackingRuntimeConfig(enabled=True, mode="static"),
        trainer_variant=None,
        per_rank_batches_est=15,
        gradient_accumulation_steps=32,
        world_size=1,
        dataloader_drop_last=True,
    )

    assert any(
        "per_rank_batches_est=15 is smaller than gradient_accumulation_steps=32"
        in msg
        for msg in warning_logs
    )


def test_validate_static_packing_accumulation_windows_warns_on_remainder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warning_logs: list[str] = []

    def _capture_warning(message: str, *args: object) -> None:
        if args:
            warning_logs.append(message % args)
        else:
            warning_logs.append(message)

    monkeypatch.setattr(sft_module.logger, "warning", _capture_warning)

    _validate_static_packing_accumulation_windows(
        packing_cfg=PackingRuntimeConfig(
            enabled=True,
            mode="static",
        ),
        trainer_variant=None,
        per_rank_batches_est=15,
        gradient_accumulation_steps=8,
        world_size=1,
        dataloader_drop_last=False,
    )

    assert any(
        "is not divisible by gradient_accumulation_steps=8" in msg
        for msg in warning_logs
    )


@pytest.mark.parametrize(
    ("config_name", "expected_ordering"),
    [
        ("../lvis_bbox_max60_1024.yaml", "sorted"),
        ("../smoke/lvis_bbox_max60_1024.yaml", "sorted"),
    ],
)
def test_stage1_leaves_pin_ordering_cache_seed_and_paths(
    config_name: str,
    expected_ordering: str,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = ConfigLoader.load_materialized_training_config(
        str(repo_root / "configs" / "stage1" / "ablation" / config_name)
    )

    training = cfg.training
    custom = cfg.custom
    template = cfg.template

    assert custom.object_ordering == expected_ordering
    assert training["seed"] == 17
    assert template["max_pixels"] == 1048576
    assert "rescale_32_1024_bbox_max60/train.coord.jsonl" in str(custom.train_jsonl)


def test_static_packing_avoids_thread_pool_for_unsafe_length_helper_in_distributed_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.datasets.wrappers import packed_caption as packed_mod

    dataset = _UnsafeLengthDataset()

    monkeypatch.setattr(packed_mod.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(packed_mod.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(packed_mod.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(packed_mod.torch.cuda, "is_initialized", lambda: False)

    class _BoomPool:
        def __init__(self, *args, **kwargs):
            raise AssertionError("unsafe static precompute should not use thread pool")

    monkeypatch.setattr(packed_mod.concurrent.futures, "ThreadPoolExecutor", _BoomPool)

    packed = packed_mod.build_static_packed_dataset(
        dataset,
        template=_Template(max_length=16),
        packing_length=8,
        cache_dir=tmp_path / "static-packing",
        fingerprint={"dataset": "unsafe"},
        world_size=1,
        train_dataloader_shuffle=False,
        length_precompute_workers=4,
    )

    assert len(packed) > 0
