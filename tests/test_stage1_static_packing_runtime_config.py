from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.callbacks import DatasetEpochCallback
from src.config.loader import ConfigLoader
import src.sft as sft_module
from src.sft import (
    PackingRuntimeConfig,
    _append_dataset_epoch_callback,
    _build_static_packing_fingerprint,
    _parse_encoded_sample_cache_config,
    _parse_packing_config,
    _recompute_gas_for_packing,
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


def test_validate_stage1_static_packing_policy_skips_rollout_matching_variants() -> None:
    _validate_stage1_static_packing_policy(
        packing_cfg=PackingRuntimeConfig(enabled=True, mode="dynamic"),
        trainer_variant="stage2_two_channel",
    )


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
        fusion_config_path=None,
    )

    assert fingerprint["dataset_split"] == "train"
    assert fingerprint["dataset_jsonl"] == str(train_jsonl)
    assert fingerprint["custom_train_jsonl"] == str(train_jsonl)
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
        fusion_config_path=None,
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
        fusion_config_path=None,
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
        fusion_config_path=None,
    )

    assert fingerprint_a["custom_offline_max_pixels"] == 1048576
    assert fingerprint_a["coord_tokens"] == {
        "enabled": True,
        "skip_bbox_norm": True,
    }
    assert fingerprint_b["custom_offline_max_pixels"] == 2097152
    assert fingerprint_b["coord_tokens"] == {"enabled": False}
    assert fingerprint_a != fingerprint_b


@pytest.mark.parametrize(
    ("config_relpath", "expected_ordering"),
    [
        (
            "configs/stage1/ablation/2b_coord_ce_soft_ce_w1_gate_coco80_desc_first_sorted_order.yaml",
            "sorted",
        ),
        (
            "configs/stage1/ablation/2b_coord_ce_soft_ce_w1_gate_coco80_desc_first_random_order.yaml",
            "random",
        ),
    ],
)
def test_stage1_ablation_profiles_pin_cache_parity_and_ordering(
    config_relpath: str,
    expected_ordering: str,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = ConfigLoader.load_materialized_training_config(str(repo_root / config_relpath))

    assert cfg.training["seed"] == 17
    assert cfg.training["encoded_sample_cache"]["enabled"] is False
    assert cfg.custom.object_ordering == expected_ordering
    assert expected_ordering in cfg.training["run_name"]
    assert expected_ordering in cfg.training["output_dir"]
    assert expected_ordering in cfg.training["logging_dir"]


def test_lvis_stage1_config_keeps_canonical_recipe_and_desc_first_sorted_contract() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = ConfigLoader.load_materialized_training_config(
        str(repo_root / "configs/stage1/lvis_bbox_max60_1024.yaml")
    )

    assert cfg.training["optimizer"] == "multimodal_coord_offset"
    assert (
        cfg.training["run_name"]
        == "epoch_2-hard_ce_soft_ce_w1_ciou_bbox_size"
    )
    assert cfg.custom.train_jsonl == "public_data/lvis/rescale_32_1024_bbox_max60/train.coord.jsonl"
    assert cfg.custom.val_jsonl == "public_data/lvis/rescale_32_1024_bbox_max60/val.coord.jsonl"
    assert cfg.custom.val_sample_limit == 512
    assert cfg.custom.dump_conversation_text is True
    assert cfg.custom.object_ordering == "sorted"
    assert cfg.custom.object_field_order == "desc_first"
    assert cfg.custom.extra["prompt_variant"] == "lvis_stage1_federated"
    assert cfg.custom.coord_soft_ce_w1.enabled is True
    assert cfg.custom.coord_soft_ce_w1.ce_weight == pytest.approx(1.0)
    assert cfg.custom.coord_soft_ce_w1.soft_ce_weight == pytest.approx(1.0)
    assert cfg.custom.coord_soft_ce_w1.w1_weight == pytest.approx(1.0)
    assert cfg.custom.coord_soft_ce_w1.gate_weight == pytest.approx(5.0)
    assert cfg.custom.bbox_geo.enabled is True
    assert cfg.custom.bbox_geo.smoothl1_weight == pytest.approx(0.01)
    assert cfg.custom.bbox_geo.ciou_weight == pytest.approx(1.0)
    assert cfg.custom.bbox_size_aux.enabled is True
    assert cfg.custom.bbox_size_aux.log_wh_weight == pytest.approx(0.05)
    assert cfg.training["output_dir"] == "./output/stage1/lvis_bbox_max60_1024"
    assert cfg.training["logging_dir"] == "./tb/stage1/lvis_bbox_max60_1024"


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
    assert cfg.custom.bbox_geo.enabled is True
    assert cfg.custom.bbox_geo.ciou_weight == pytest.approx(1.0)
    assert cfg.custom.bbox_size_aux.enabled is True
    assert cfg.training["max_steps"] == 2
    assert cfg.custom.train_sample_limit == 32
    assert cfg.custom.val_sample_limit == 8
    assert (
        cfg.training["run_name"]
        == "smoke_2steps-stage1-lvis_bbox_max60_1024-hard_ce_soft_ce_w1_ciou_bbox_size"
    )
    assert cfg.training["output_dir"] == "./output/stage1/smoke/lvis_bbox_max60_1024"
    assert cfg.training["logging_dir"] == "./tb/stage1/smoke/lvis_bbox_max60_1024"


def test_lvis_stage2_config_keeps_same_data_contract_with_stage2_prompt() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = ConfigLoader.load_materialized_training_config(
        str(repo_root / "configs/stage2_two_channel/lvis_bbox_max60_1024.yaml")
    )

    assert (
        cfg.model["model"]
        == "output/stage1/lvis_bbox_max60_1024/epoch_4-stage1-lvis_bbox_max60_1024-hard_ce_soft_ce_w1_ciou_bbox_size-merged"
    )
    assert (
        cfg.training["run_name"]
        == "epoch_2-stage2-lvis_bbox_max60_1024-hard_ce_soft_ce_w1_ciou_bbox_size"
    )
    assert cfg.custom.train_jsonl == "public_data/lvis/rescale_32_1024_bbox_max60/train.coord.jsonl"
    assert cfg.custom.val_jsonl == "public_data/lvis/rescale_32_1024_bbox_max60/val.coord.jsonl"
    assert cfg.custom.object_ordering == "sorted"
    assert cfg.custom.object_field_order == "desc_first"
    assert cfg.custom.extra["prompt_variant"] == "lvis_stage2_federated"
    assert cfg.rollout_matching.eval_detection.metrics == "f1ish"
    objective = {module.name: module for module in cfg.stage2_ab.pipeline.objective}
    assert objective["bbox_geo"].config["smoothl1_weight"] == pytest.approx(0.0)
    assert objective["bbox_geo"].config["ciou_weight"] == pytest.approx(1.0)
    assert objective["coord_reg"].config["coord_ce_weight"] == pytest.approx(1.0)
    assert objective["coord_reg"].config["soft_ce_weight"] == pytest.approx(1.0)
    assert objective["coord_reg"].config["w1_weight"] == pytest.approx(1.0)


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
        (
            "2b_coord_ce_soft_ce_w1_gate_coco80_desc_first_sorted_order.yaml",
            "sorted",
        ),
        (
            "2b_coord_ce_soft_ce_w1_gate_coco80_desc_first_random_order.yaml",
            "random",
        ),
    ],
)
def test_stage1_ablation_leaves_pin_ordering_cache_seed_and_paths(
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
    assert training["encoded_sample_cache"]["enabled"] is False
    assert training["seed"] == 17
    assert expected_ordering in str(training["run_name"])
    assert expected_ordering in str(training["output_dir"])
    assert expected_ordering in str(training["logging_dir"])
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
