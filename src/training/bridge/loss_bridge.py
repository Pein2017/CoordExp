"""Trainer-to-objective loss bridge."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch

from src.trainers.batch_extras import BatchExtras
from src.trainers.teacher_forcing.forwards import prepare_forward_inputs
from src.training.bridge.coordinate_mapper import PredictionCoordinateMapper
from src.training.encoding.model_inputs import (
    ModelInputBundle,
    SIDECAR_ONLY_KEYS,
)
from src.training.objectives.runner import ObjectiveRunner
from src.training.objectives.types import ObjectiveRunResult, ObjectiveSpec
from src.training.sidecars import TrainingSidecars
from src.training.supervision.batch import SupervisionBatch
from src.training.teacher_forcing.constants import TEACHER_FORCING_TARGET_IR_KEY


@dataclass(frozen=True, slots=True)
class TrainerLossBridgeSettings:
    """Optional bridge behavior switches.

    :param runner_owns_loss: Whether labels and legacy loss inputs are stripped
        before model forwarding.
    :param packing_enabled: Whether Qwen packing metadata is required by forward
        preparation.
    :param allow_logits_projection: Whether ``logits_to_keep`` projection is
        accepted. Projection is intentionally disabled by default for Task 8.
    """

    runner_owns_loss: bool = True
    packing_enabled: bool = False
    allow_logits_projection: bool = False

    def __post_init__(self) -> None:
        """Validate settings with plain booleans only."""

        for field_name in (
            "runner_owns_loss",
            "packing_enabled",
            "allow_logits_projection",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"{field_name} must be a plain bool")


@dataclass(frozen=True, slots=True)
class TrainerLossBridgeResult:
    """Loss bridge output returned to trainer integrations."""

    loss: torch.Tensor
    outputs: Any
    objective_result: ObjectiveRunResult
    model_inputs: ModelInputBundle
    coordinate_mapper: PredictionCoordinateMapper
    training_sidecars: TrainingSidecars


class TrainerLossBridge:
    """Run one model forward and delegate semantic loss to ``ObjectiveRunner``."""

    def __init__(
        self,
        *,
        objective_runner: ObjectiveRunner | None = None,
        settings: TrainerLossBridgeSettings | None = None,
    ) -> None:
        """Initialize the bridge with optional runner and settings overrides."""

        self._objective_runner = objective_runner or ObjectiveRunner()
        self._settings = settings or TrainerLossBridgeSettings()

    def compute_loss(
        self,
        *,
        model: Any,
        raw_batch: Mapping[str, Any],
        batch_extras: BatchExtras | None = None,
        training_sidecars: TrainingSidecars | None = None,
        supervision: SupervisionBatch,
        objectives: Sequence[ObjectiveSpec],
        sample_id_to_batch_index: Mapping[str, int] | None = None,
    ) -> TrainerLossBridgeResult:
        """Return runner-owned objective loss for one raw trainer batch."""

        # establish the validated model-input bundle after dropping sidecars.
        if not isinstance(raw_batch, Mapping):
            raise TypeError("raw_batch must be a mapping")
        if batch_extras is not None and type(batch_extras) is not BatchExtras:
            raise TypeError("batch_extras must be a BatchExtras or None")
        if training_sidecars is not None and type(training_sidecars) is not TrainingSidecars:
            raise TypeError("training_sidecars must be a TrainingSidecars or None")
        if type(supervision) is not SupervisionBatch:
            raise TypeError("supervision must be a SupervisionBatch")
        if not self._settings.runner_owns_loss:
            raise NotImplementedError(
                "runner_owns_loss=False is not implemented in TrainerLossBridge; "
                "the bridge currently returns ObjectiveRunner-owned losses only"
            )

        resolved_sidecars = self._extract_training_sidecars(
            raw_batch,
            batch_extras=batch_extras,
            training_sidecars=training_sidecars,
        )
        model_inputs = ModelInputBundle.from_mapping(
            self._strip_sidecars(raw_batch),
            runner_owns_loss=self._settings.runner_owns_loss,
        )
        self._reject_logits_projection(model_inputs)

        # prepare Qwen-compatible forwarded kwargs without reimplementing metadata rules.
        ignored_keys = [
            key
            for key in model_inputs.payload
            if model_inputs.classification_for(key) != "forwarded"
        ]
        core_model, inputs_for_model, _model_type = prepare_forward_inputs(
            model=model,
            inputs=model_inputs.payload,
            ignored_keys=ignored_keys,
            packing_enabled=self._settings.packing_enabled,
            where="TrainerLossBridge",
        )

        # call the model exactly once and require full, unsliced logits.
        outputs = core_model(**inputs_for_model)
        logits = self._extract_logits(outputs)
        self._validate_full_logits(
            logits=logits,
            input_ids=inputs_for_model.get("input_ids"),
        )

        # construct bridge-level coordinates before delegating objective math.
        coordinate_mapper = PredictionCoordinateMapper.from_logits(
            logits,
            supervision=supervision,
            sample_id_to_batch_index=sample_id_to_batch_index,
        )
        objective_result = self._objective_runner.run(
            logits=logits,
            supervision=supervision,
            objectives=tuple(objectives),
            label_rows=coordinate_mapper.label_rows,
        )

        # return only the runner-owned objective loss, ignoring model-provided loss.
        return TrainerLossBridgeResult(
            loss=objective_result.loss,
            outputs=outputs,
            objective_result=objective_result,
            model_inputs=model_inputs,
            coordinate_mapper=coordinate_mapper,
            training_sidecars=resolved_sidecars,
        )

    def _strip_sidecars(self, raw_batch: Mapping[str, Any]) -> dict[str, Any]:
        """Return raw batch values that are eligible for bundle validation."""

        return {
            key: value
            for key, value in raw_batch.items()
            if key not in SIDECAR_ONLY_KEYS
        }

    def _extract_training_sidecars(
        self,
        raw_batch: Mapping[str, Any],
        *,
        batch_extras: BatchExtras | None,
        training_sidecars: TrainingSidecars | None,
    ) -> TrainingSidecars:
        """Return semantic sidecars carried by the trainer batch."""

        raw_sidecars = raw_batch.get("training_sidecars")
        if training_sidecars is not None:
            sidecars = training_sidecars
        elif type(raw_sidecars) is TrainingSidecars:
            sidecars = raw_sidecars
        else:
            sidecars = TrainingSidecars()

        teacher_forcing_target_ir = raw_batch.get(TEACHER_FORCING_TARGET_IR_KEY)
        if batch_extras is not None and batch_extras.teacher_forcing_target_ir is not None:
            teacher_forcing_target_ir = batch_extras.teacher_forcing_target_ir

        if teacher_forcing_target_ir is None:
            return sidecars

        supervision = replace(
            sidecars.supervision,
            teacher_forcing_target_ir=teacher_forcing_target_ir,
        )
        return replace(sidecars, supervision=supervision)

    def _reject_logits_projection(self, model_inputs: ModelInputBundle) -> None:
        """Reject logits projection unless an explicit future setting enables it."""

        if (
            "logits_to_keep" in model_inputs.bridge_auxiliaries()
        ):
            if self._settings.allow_logits_projection:
                raise NotImplementedError(
                    "logits_to_keep projection is not implemented in "
                    "TrainerLossBridge"
                )
            raise ValueError(
                "logits_to_keep requires explicit logits projection support; "
                "TrainerLossBridge preserves full logits by default"
            )

    def _extract_logits(self, outputs: Any) -> torch.Tensor:
        """Return logits from model outputs."""

        logits = getattr(outputs, "logits", None)
        if not isinstance(logits, torch.Tensor):
            raise TypeError("model outputs must expose logits as a torch.Tensor")

        return logits

    def _validate_full_logits(
        self,
        *,
        logits: torch.Tensor,
        input_ids: Any,
    ) -> None:
        """Validate logits preserve the model time dimension when possible."""

        if not isinstance(input_ids, torch.Tensor):
            return
        if logits.ndim == 3 and input_ids.ndim >= 2:
            expected = tuple(int(dim) for dim in input_ids.shape[:2])
            actual = tuple(int(dim) for dim in logits.shape[:2])
        elif logits.ndim == 2 and input_ids.ndim == 1:
            expected = (int(input_ids.shape[0]),)
            actual = (int(logits.shape[0]),)
        elif logits.ndim == 2 and input_ids.ndim >= 2 and int(input_ids.shape[0]) == 1:
            expected = (int(input_ids.shape[1]),)
            actual = (int(logits.shape[0]),)
        else:
            if logits.ndim == 2 and input_ids.ndim >= 2 and int(input_ids.shape[0]) > 1:
                raise ValueError(
                    "TrainerLossBridge requires full logits for batched input_ids; "
                    "2D logits are ambiguous for multi-sample batches. "
                    "Sliced logits are not supported."
                )
            return

        if actual != expected:
            raise ValueError(
                "TrainerLossBridge requires full logits aligned to input_ids; "
                f"got logits prefix {actual} for input_ids prefix {expected}. "
                "Sliced logits are not supported."
            )
