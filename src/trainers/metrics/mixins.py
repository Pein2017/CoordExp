"""Compatibility re-exports for trainer metric/loss mixins.

Concern-specific implementations live in sibling modules. Keep this module as
the legacy import surface for existing trainer setup and tests.
"""

from src.trainers.metrics.aggregate_tokens import AggregateTokenTypeMetricsMixin
from src.trainers.metrics.batch_contract import (
    GradAccumLossScaleMixin,
    _resolve_embedding_rows,
    _resolve_text_vocab_size,
    _validate_batch_contract,
)
from src.trainers.metrics.coord_losses import CoordSoftCEW1LossMixin
from src.trainers.metrics.recursive_detection import RecursiveDetectionCEMixin
from src.trainers.metrics.sft_gaussian_coord_soft_ce import (
    SFTGaussianCoordSoftCELossMixin,
)
from src.trainers.metrics.structural_close import SFTStructuralCloseLossMixin
from src.trainers.metrics.teacher_forcing import TeacherForcingObjectiveMixin
from src.trainers.monitoring.instability import InstabilityMonitorMixin

__all__ = [
    "GradAccumLossScaleMixin",
    "_resolve_embedding_rows",
    "_resolve_text_vocab_size",
    "_validate_batch_contract",
    "SFTStructuralCloseLossMixin",
    "RecursiveDetectionCEMixin",
    "SFTGaussianCoordSoftCELossMixin",
    "TeacherForcingObjectiveMixin",
    "AggregateTokenTypeMetricsMixin",
    "CoordSoftCEW1LossMixin",
    "InstabilityMonitorMixin",
]
