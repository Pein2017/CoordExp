from __future__ import annotations

from pathlib import Path

import pytest

from src.config.inference import InferConfig, load_infer_config


@pytest.mark.parametrize(
    "config_path",
    (
        Path("configs/infer/production.yaml"),
        Path("configs/infer/vllm.yaml"),
        Path("configs/infer/smoke/two_gpu_qwen3_vl_2b_coco.yaml"),
    ),
)
def test_neutral_canonical_infer_configs_load_with_strict_loader(
    config_path: Path,
) -> None:
    resolved = load_infer_config(config_path)

    assert isinstance(resolved.config, InferConfig)
    assert resolved.entry_config_path == config_path.resolve()
    assert resolved.config.schema_version == 1
    assert resolved.fingerprint


@pytest.mark.parametrize("dtype", ["fp16", "fp32"])
def test_vllm_config_rejects_every_non_bf16_dtype(tmp_path: Path, dtype: str) -> None:
    config = tmp_path / "vllm.yaml"
    config.write_text(
        "\n".join(
            (
                "schema_version: 1",
                f"extends: {Path('configs/infer/vllm.yaml').resolve()}",
                "model:",
                f"  dtype: {dtype}",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    from src.common.errors import ConfigContractError

    with pytest.raises(ConfigContractError) as exc_info:
        load_infer_config(config)

    assert exc_info.value.code == "config.vllm_dtype_unqualified"


def test_canonical_vllm_config_uses_an_executed_concurrency_mode() -> None:
    resolved = load_infer_config("configs/infer/vllm.yaml")

    assert resolved.config.model.dtype == "bf16"
    assert resolved.config.generation.batch_size == 4
