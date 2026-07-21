from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from src.common.errors import ConfigContractError
from src.config.inference import load_infer_config
from src.inference.data_parallel import plan_data_parallel_shards


RANDOM_VAL200_CONFIG = Path(
    "configs/coordexp_swift/infer/"
    "qwen3_vl_2b_desc_first_random_pure_ce_typegate_dora_r16a32_"
    "step4887_val200_hf_fp32.yaml"
)


def test_random_val200_config_carries_training_order_seed() -> None:
    config = load_infer_config(RANDOM_VAL200_CONFIG).config

    assert config.template.object_ordering == "random"
    assert config.template.object_order_seed == 17


def test_random_inference_ordering_rejects_missing_seed(tmp_path: Path) -> None:
    payload = yaml.safe_load(Path("configs/coordexp_swift/infer/base.yaml").read_text())
    payload["template"]["object_ordering"] = "random"
    config_path = tmp_path / "random-without-seed.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigContractError) as exc_info:
        load_infer_config(config_path)

    assert exc_info.value.code == "config.random_object_order_seed_required"


def test_worker_canonicalizes_unindexed_cuda_device() -> None:
    from src.inference.worker import _model_first_parameter_device

    model = SimpleNamespace(
        parameters=lambda: iter((SimpleNamespace(device="cuda"),))
    )

    assert _model_first_parameter_device(model) == "cuda:0"


def test_merge_accepts_canonical_equivalent_unindexed_cuda_device() -> None:
    from src.inference.merge import _validate_worker_metadata_matches_rank_plan

    plan = plan_data_parallel_shards(
        row_ids=("row-0",),
        per_device_batch_size=1,
        visible_cuda_tokens=("7",),
    )
    metadata = {
        "rank": 0,
        "world_size": 1,
        "assigned_parent_visible_device_token": "7",
        "worker_cuda_visible_devices": "7",
        "worker_logical_device": "cuda:0",
        "cuda_device_count": 1,
        "cuda_current_device": 0,
        "model_first_parameter_device": "cuda",
    }

    _validate_worker_metadata_matches_rank_plan(
        worker_metadata=metadata,
        rank_plan=plan.ranks[0],
    )
