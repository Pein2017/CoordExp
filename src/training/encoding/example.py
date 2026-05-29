"""Example-level encoding containers."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.training.encoding.model_inputs import ModelInputBundle
from src.training.encoding.view import EncodedDetectionView
from src.training.sidecars import TrainingSidecars


@dataclass(frozen=True, slots=True)
class EncodedTrainingExample:
    """Minimal encoded example bundle for bridge/runner handoff.

    :param detection_view: Authoritative tokenized detection view.
    :param model_inputs: Strict backend model-input bundle.
    :param sidecars: Non-forwarded sidecar payloads.
    """

    detection_view: EncodedDetectionView
    model_inputs: ModelInputBundle
    sidecars: TrainingSidecars = field(default_factory=TrainingSidecars)

    def __post_init__(self) -> None:
        """Validate composed encoding contract types."""

        if type(self.detection_view) is not EncodedDetectionView:
            raise TypeError("detection_view must be EncodedDetectionView")
        if type(self.model_inputs) is not ModelInputBundle:
            raise TypeError("model_inputs must be ModelInputBundle")
        if type(self.sidecars) is not TrainingSidecars:
            raise TypeError("sidecars must be TrainingSidecars")


__all__ = [
    "EncodedTrainingExample",
]
