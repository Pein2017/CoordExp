from __future__ import annotations

from src.augmentation.factory import build_augmentation_processor
from src.augmentation.processor import (
    GeometryFlipAugmentationProcessor,
    NoopAugmentationProcessor,
)
from src.config.loader import load_train_config


FIXTURE_CONFIG = "tests/fixtures/smoke/qwen3_vl_single_image_pack/config.yaml"


def test_factory_uses_noop_when_train_geometry_flips_disabled() -> None:
    config = load_train_config(FIXTURE_CONFIG).config

    processor = build_augmentation_processor(config, split="train")

    assert isinstance(processor, NoopAugmentationProcessor)


def test_factory_uses_geometry_flips_only_for_train_split() -> None:
    config = load_train_config(FIXTURE_CONFIG).config
    config = config.model_copy(
        update={
            "data": config.data.model_copy(
                update={
                    "augmentation": config.data.augmentation.model_copy(
                        update={
                            "train": config.data.augmentation.train.model_copy(
                                update={
                                    "geometry_flips": (
                                        config.data.augmentation.train.geometry_flips.model_copy(
                                            update={
                                                "enabled": True,
                                                "horizontal_prob": 1.0,
                                            }
                                        )
                                    )
                                }
                            )
                        }
                    )
                }
            )
        }
    )

    train_processor = build_augmentation_processor(config, split="train")
    eval_processor = build_augmentation_processor(config, split="eval.forward")

    assert isinstance(train_processor, GeometryFlipAugmentationProcessor)
    assert isinstance(eval_processor, NoopAugmentationProcessor)
