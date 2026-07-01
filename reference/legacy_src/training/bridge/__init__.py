"""Bridge contracts between trainer batches and semantic objectives."""

from src.training.bridge.coordinate_mapper import PredictionCoordinateMapper
from src.training.bridge.loss_bridge import (
    TrainerLossBridge,
    TrainerLossBridgeResult,
    TrainerLossBridgeSettings,
)

__all__ = [
    "PredictionCoordinateMapper",
    "TrainerLossBridge",
    "TrainerLossBridgeResult",
    "TrainerLossBridgeSettings",
]
