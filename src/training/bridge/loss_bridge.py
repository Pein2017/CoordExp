"""Trainer-to-objective loss bridge."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch

from src.metrics.events import MetricEvent
from src.trainers.batch_extras import BatchExtras
from src.trainers.teacher_forcing.forwards import prepare_forward_inputs
from src.training.bridge.coordinate_mapper import PredictionCoordinateMapper
from src.training.coverage_ledger.head import CoverageLedgerHead
from src.training.coverage_ledger.loss import (
    CoverageLedgerLossConfig,
    compute_coverage_ledger_loss,
)
from src.training.coverage_ledger.metrics import (
    COVERAGE_ACCURACY_KEY,
    COVERAGE_AUC_KEY,
    coverage_ledger_metric_events,
)
from src.training.coverage_ledger.qwen_capture import CoverageLedgerForwardCapture
from src.training.coverage_ledger.sidecars import CoverageLedgerSidecar
from src.training.coverage_ledger.visual_regions import (
    map_norm1000_bbox_to_visual_token_region,
    pool_object_visual_embeddings,
)
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
    coverage_ledger: Any | None = None

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
    metric_events: tuple[MetricEvent, ...] = ()


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

        # call the model exactly once and require full, unsliced logits.
        coverage_ledger_enabled = self._coverage_ledger_enabled()
        coverage_ledger_sidecar: CoverageLedgerSidecar | None = None
        coverage_ledger_head: CoverageLedgerHead | None = None
        if coverage_ledger_enabled:
            coverage_ledger_sidecar = self._require_coverage_ledger_sidecar(
                resolved_sidecars
            )
            coverage_ledger_head = self._require_coverage_ledger_head(model)
            captured = CoverageLedgerForwardCapture().capture(
                model=model,
                inputs=model_inputs.payload,
                ignored_keys=ignored_keys,
                packing_enabled=self._settings.packing_enabled,
                where="TrainerLossBridge",
            )
            outputs = captured
            logits = captured.logits
        else:
            core_model, inputs_for_model, _model_type = prepare_forward_inputs(
                model=model,
                inputs=model_inputs.payload,
                ignored_keys=ignored_keys,
                packing_enabled=self._settings.packing_enabled,
                where="TrainerLossBridge",
            )
            outputs = core_model(**inputs_for_model)
            logits = self._extract_logits(outputs)
        self._validate_full_logits(
            logits=logits,
            input_ids=model_inputs.payload.get("input_ids"),
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
        loss = objective_result.loss
        metric_events = tuple(objective_result.metric_events)

        if coverage_ledger_enabled:
            assert coverage_ledger_sidecar is not None
            assert coverage_ledger_head is not None
            visual_config = self._resolve_coverage_ledger_visual_config(model)
            regions = tuple(
                map_norm1000_bbox_to_visual_token_region(
                    entry.bbox_norm1000_xyxy,
                    image_grid_thw=coverage_ledger_sidecar.image_grid_thw,
                    processed_width=coverage_ledger_sidecar.processed_width,
                    processed_height=coverage_ledger_sidecar.processed_height,
                    patch_size=visual_config["patch_size"],
                    spatial_merge_size=visual_config["spatial_merge_size"],
                )
                for entry in coverage_ledger_sidecar.object_entries
            )
            pooled_visual_object_embeddings = pool_object_visual_embeddings(
                captured.image_embeds,
                regions,
            )
            coverage_ledger_result = compute_coverage_ledger_loss(
                head=coverage_ledger_head,
                final_hidden_states=captured.final_hidden_states,
                pooled_visual_object_embeddings=pooled_visual_object_embeddings,
                sidecar=coverage_ledger_sidecar,
                config=self._coverage_ledger_loss_config(),
                sample_id_to_batch_index=sample_id_to_batch_index,
            )
            loss = loss + coverage_ledger_result.weighted_loss
            metric_events = metric_events + self._filter_coverage_ledger_metric_events(
                coverage_ledger_metric_events(coverage_ledger_result)
            )

        # return only the runner-owned objective loss, ignoring model-provided loss.
        return TrainerLossBridgeResult(
            loss=loss,
            outputs=outputs,
            objective_result=objective_result,
            model_inputs=model_inputs,
            coordinate_mapper=coordinate_mapper,
            training_sidecars=resolved_sidecars,
            metric_events=metric_events,
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
        if training_sidecars is not None and type(raw_sidecars) is TrainingSidecars:
            self._validate_training_sidecars_match(
                explicit_sidecars=training_sidecars,
                raw_sidecars=raw_sidecars,
            )

        if training_sidecars is not None:
            sidecars = training_sidecars
        elif type(raw_sidecars) is TrainingSidecars:
            sidecars = raw_sidecars
        else:
            sidecars = TrainingSidecars()

        teacher_forcing_target_ir = self._resolve_teacher_forcing_target_ir(
            sidecars=sidecars,
            raw_batch=raw_batch,
            batch_extras=batch_extras,
        )
        if teacher_forcing_target_ir is None:
            return sidecars
        if self._teacher_forcing_target_ir_equal(
            sidecars.supervision.teacher_forcing_target_ir,
            teacher_forcing_target_ir,
        ):
            return sidecars

        supervision = replace(
            sidecars.supervision,
            teacher_forcing_target_ir=teacher_forcing_target_ir,
        )
        return replace(sidecars, supervision=supervision)

    def _validate_training_sidecars_match(
        self,
        *,
        explicit_sidecars: TrainingSidecars,
        raw_sidecars: TrainingSidecars,
    ) -> None:
        """Validate explicit and raw full sidecars do not disagree."""

        if self._sidecar_payload_equal(explicit_sidecars, raw_sidecars):
            return

        raise ValueError(
            "conflicting teacher_forcing_target_ir sources: "
            "training_sidecars and raw_batch.training_sidecars"
        )

    def _resolve_teacher_forcing_target_ir(
        self,
        *,
        sidecars: TrainingSidecars,
        raw_batch: Mapping[str, Any],
        batch_extras: BatchExtras | None,
    ) -> Any:
        """Resolve teacher-forcing IR with semantic sidecars as canonical."""

        semantic_ir = sidecars.supervision.teacher_forcing_target_ir
        raw_ir = raw_batch.get(TEACHER_FORCING_TARGET_IR_KEY)
        batch_ir = (
            None
            if batch_extras is None
            else batch_extras.teacher_forcing_target_ir
        )

        sources: list[tuple[str, Any]] = []
        if semantic_ir is not None:
            sources.append(("training_sidecars", semantic_ir))
        if raw_ir is not None:
            sources.append(("raw_batch", raw_ir))
        if batch_ir is not None:
            sources.append(("batch_extras", batch_ir))

        for index, (left_name, left_value) in enumerate(sources):
            for right_name, right_value in sources[index + 1 :]:
                if not self._teacher_forcing_target_ir_equal(left_value, right_value):
                    raise ValueError(
                        "conflicting teacher_forcing_target_ir sources: "
                        f"{left_name} and {right_name}"
                    )

        if semantic_ir is not None:
            return semantic_ir
        if batch_ir is not None:
            return batch_ir
        return raw_ir

    def _coverage_ledger_enabled(self) -> bool:
        """Return whether the bridge-local coverage ledger auxiliary is enabled."""

        cfg = self._settings.coverage_ledger
        if cfg is None:
            return False
        if isinstance(cfg, Mapping):
            return bool(cfg.get("enabled", False))
        return bool(getattr(cfg, "enabled", False))

    def _require_coverage_ledger_sidecar(
        self,
        sidecars: TrainingSidecars,
    ) -> CoverageLedgerSidecar:
        """Return the one V0 coverage-ledger sidecar required by enabled config."""

        payloads = tuple(
            payload
            for payload in sidecars.supervision.payloads
            if type(payload) is CoverageLedgerSidecar
        )
        if len(payloads) != 1:
            raise ValueError(
                "coverage_ledger.enabled=true requires exactly one "
                f"CoverageLedgerSidecar; got {len(payloads)}"
            )
        return payloads[0]

    def _require_coverage_ledger_head(self, model: Any) -> CoverageLedgerHead:
        """Return the one trainable coverage-ledger head attached to the model."""

        heads: list[CoverageLedgerHead] = []
        direct_head = getattr(model, "coverage_ledger_head", None)
        if isinstance(direct_head, CoverageLedgerHead):
            heads.append(direct_head)

        named_modules = getattr(model, "named_modules", None)
        if callable(named_modules):
            for name, module in named_modules():
                if (
                    (name == "coverage_ledger_head" or name.endswith(".coverage_ledger_head"))
                    and isinstance(module, CoverageLedgerHead)
                    and all(id(module) != id(existing) for existing in heads)
                ):
                    heads.append(module)

        if len(heads) != 1:
            raise ValueError(
                "coverage_ledger.enabled=true requires exactly one "
                f"coverage_ledger_head; got {len(heads)}"
            )
        return heads[0]

    def _coverage_ledger_loss_config(self) -> CoverageLedgerLossConfig:
        """Translate bridge settings into the coverage-ledger loss config."""

        cfg = self._settings.coverage_ledger
        return CoverageLedgerLossConfig(
            coverage_weight=self._coverage_ledger_float(
                cfg,
                "coverage_weight",
                default=0.1,
            ),
            region_anchor_weight=self._coverage_ledger_float(
                cfg,
                "region_anchor_weight",
                default=0.1,
            ),
            temperature=self._coverage_ledger_float(
                cfg,
                "temperature",
                default=0.2,
            ),
            pos_weight=self._coverage_ledger_float(
                cfg,
                "pos_weight",
                default=1.0,
            ),
        )

    def _filter_coverage_ledger_metric_events(
        self,
        events: tuple[MetricEvent, ...],
    ) -> tuple[MetricEvent, ...]:
        """Apply optional bridge metric toggles to ledger diagnostic events."""

        cfg = self._settings.coverage_ledger
        log_auc = self._coverage_ledger_bool(cfg, "log_auc", default=True)
        log_accuracy = self._coverage_ledger_bool(
            cfg,
            "log_accuracy",
            default=True,
        )
        filtered = []
        for event in events:
            if event.key == COVERAGE_AUC_KEY and not log_auc:
                continue
            if event.key == COVERAGE_ACCURACY_KEY and not log_accuracy:
                continue
            filtered.append(event)
        return tuple(filtered)

    def _resolve_coverage_ledger_visual_config(self, model: Any) -> dict[str, int]:
        """Resolve Qwen visual patch and merge sizes from the active model tree."""

        candidates = self._iter_model_config_candidates(model)
        patch_size = self._first_positive_int_attr(candidates, "patch_size")
        spatial_merge_size = self._first_positive_int_attr(
            candidates,
            "spatial_merge_size",
        )
        missing = []
        if patch_size is None:
            missing.append("patch_size")
        if spatial_merge_size is None:
            missing.append("spatial_merge_size")
        if missing:
            raise ValueError(
                "coverage_ledger.enabled=true requires Qwen visual config fields "
                + ", ".join(missing)
            )
        return {
            "patch_size": patch_size,
            "spatial_merge_size": spatial_merge_size,
        }

    def _iter_model_config_candidates(self, model: Any) -> tuple[Any, ...]:
        """Return model/config/visual candidates without assuming wrapper shape."""

        seen: set[int] = set()
        stack: list[Any] = [model]
        candidates: list[Any] = []
        while stack:
            current = stack.pop(0)
            if current is None or id(current) in seen:
                continue
            seen.add(id(current))
            candidates.append(current)
            config = getattr(current, "config", None)
            if config is not None and id(config) not in seen:
                candidates.append(config)
            visual = getattr(current, "visual", None)
            if visual is not None and id(visual) not in seen:
                candidates.append(visual)
                visual_config = getattr(visual, "config", None)
                if visual_config is not None and id(visual_config) not in seen:
                    candidates.append(visual_config)
            get_base_model = getattr(current, "get_base_model", None)
            if callable(get_base_model):
                try:
                    base_model = get_base_model()
                except TypeError:
                    base_model = None
                if base_model is not None:
                    stack.append(base_model)
            for attr_name in ("module", "base_model", "model"):
                child = getattr(current, attr_name, None)
                if child is not None:
                    stack.append(child)
        return tuple(candidates)

    @staticmethod
    def _first_positive_int_attr(candidates: tuple[Any, ...], name: str) -> int | None:
        for candidate in candidates:
            value = getattr(candidate, name, None)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return int(value)
        return None

    @staticmethod
    def _coverage_ledger_float(cfg: Any, field_name: str, *, default: float) -> float:
        if cfg is None:
            return float(default)
        value = cfg.get(field_name, default) if isinstance(cfg, Mapping) else getattr(cfg, field_name, default)
        return float(value)

    @staticmethod
    def _coverage_ledger_bool(cfg: Any, field_name: str, *, default: bool) -> bool:
        if cfg is None:
            return bool(default)
        value = cfg.get(field_name, default) if isinstance(cfg, Mapping) else getattr(cfg, field_name, default)
        return bool(value)

    @staticmethod
    def _teacher_forcing_target_ir_equal(left: Any, right: Any) -> bool:
        """Return whether two source payloads are equivalent."""

        if left is right:
            return True
        try:
            return bool(left == right)
        except (RuntimeError, TypeError, ValueError):
            return False

    @staticmethod
    def _sidecar_payload_equal(left: TrainingSidecars, right: TrainingSidecars) -> bool:
        """Return whether two full semantic sidecar payloads are equivalent."""

        if left is right:
            return True
        try:
            return bool(left == right)
        except (RuntimeError, TypeError, ValueError):
            return False

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
