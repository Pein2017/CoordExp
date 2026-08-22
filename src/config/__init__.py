"""Configuration package."""

from src.config.loader import load_train_config
from src.config.writer import write_resolved_config_artifacts

__all__ = ["load_train_config", "write_resolved_config_artifacts"]
