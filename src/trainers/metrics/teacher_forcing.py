from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Mapping, MutableMapping, Sequence

from src.detection.token_types import build_compact_token_type_groups
from src.training.bridge import TrainerLossBridge
from src.training.objectives.types import ObjectiveSpec
from src.training.supervision.batch import SupervisionBatch
from src.training.supervision.distributions import TeacherForcingTargetDistribution
from src.training.supervision.spans import SupervisionSpan
from src.training.teacher_forcing.ir import TeacherForcingTargetIR
from src.training.teacher_forcing.vocab import RoleVocab


class TeacherForcingObjectiveMixin:
    """Trainer mixin that routes Stage-1 target-IR supervision through ObjectiveRunner."""

    def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch=None):
        from src.detection.dataset import strip_non_model_detection_sidecars
        from src.trainers.batch_extras import maybe_pop_and_stash_batch_extras

        if not isinstance(inputs, MutableMapping):
            raise TypeError("teacher_forcing objective requires dict-like inputs")
        extras = maybe_pop_and_stash_batch_extras(self, inputs)
        target_irs = _require_teacher_forcing_irs(extras.teacher_forcing_target_ir)
        sample_ids = _resolve_sample_ids(inputs.get("sample_id"), count=len(target_irs))

        strip_non_model_detection_sidecars(inputs)
        input_ids = inputs.get("input_ids")
        role_vocab = _resolve_role_vocab(self)
        supervision, sample_id_to_batch_index = _build_teacher_forcing_supervision(
            target_irs=target_irs,
            sample_ids=sample_ids,
        )
        objective_cfg = getattr(self, "teacher_forcing_objective_cfg", None)
        bridge = TrainerLossBridge()
        result = bridge.compute_loss(
            model=model,
            raw_batch=inputs,
            batch_extras=extras,
            supervision=supervision,
            objectives=(
                ObjectiveSpec(
                    "teacher_forcing",
                    config={
                        "input_ids": input_ids,
                        "role_vocab": role_vocab,
                        "coverage_strength": _coverage_strength(objective_cfg),
                        "token_type_mass_enabled": _token_type_mass_enabled(
                            objective_cfg
                        ),
                        "token_type_mass_weight": _token_type_mass_weight(
                            objective_cfg
                        ),
                    },
                ),
            ),
            sample_id_to_batch_index=sample_id_to_batch_index,
        )
        return (result.loss, result.outputs) if return_outputs else result.loss


def _require_teacher_forcing_irs(payload: Any) -> tuple[TeacherForcingTargetIR, ...]:
    if payload is None:
        raise ValueError("teacher_forcing objective requires teacher_forcing_target_ir sidecars")
    values = tuple(payload if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)) else (payload,))
    if not values:
        raise ValueError("teacher_forcing_target_ir sidecar batch is empty")
    for value in values:
        if type(value) is not TeacherForcingTargetIR:
            raise TypeError("teacher_forcing_target_ir entries must be TeacherForcingTargetIR")
    return values


def _resolve_sample_ids(payload: Any, *, count: int) -> tuple[str, ...]:
    if payload is None:
        return tuple(f"sample-{index}" for index in range(count))
    if not isinstance(payload, (str, bytes)) and callable(
        getattr(payload, "tolist", None)
    ):
        payload = payload.tolist()
    if isinstance(payload, (str, bytes)):
        values = (str(payload),)
    elif isinstance(payload, Sequence):
        values = tuple(str(item) for item in payload)
    else:
        values = (str(payload),)
    if len(values) != count:
        raise ValueError(
            "teacher_forcing sample_id sidecar length must match "
            f"teacher_forcing_target_ir length; got sample_id={len(values)} "
            f"target_ir={count}"
        )
    return values


def _build_teacher_forcing_supervision(
    *,
    target_irs: tuple[TeacherForcingTargetIR, ...],
    sample_ids: tuple[str, ...],
) -> tuple[SupervisionBatch, dict[str, int]]:
    spans: list[SupervisionSpan] = []
    sample_id_to_batch_index: dict[str, int] = {}
    for batch_index, (sample_id, target_ir) in enumerate(zip(sample_ids, target_irs, strict=True)):
        if sample_id in sample_id_to_batch_index:
            raise ValueError(f"teacher_forcing duplicate sample ID: {sample_id!r}")
        sample_id_to_batch_index[sample_id] = batch_index
        shifted_ir = _with_batch_index(target_ir, batch_index=batch_index)
        spans.append(
            SupervisionSpan(
                sample_id=sample_id,
                role="schema",
                label_positions=tuple(atom.target_position for atom in shifted_ir.atoms),
                distribution=TeacherForcingTargetDistribution(target_ir=shifted_ir),
                provenance="teacher_forcing_target_ir",
            )
        )
    return (
        SupervisionBatch(spans=tuple(spans), batch_id="teacher_forcing"),
        sample_id_to_batch_index,
    )


def _with_batch_index(
    target_ir: TeacherForcingTargetIR,
    *,
    batch_index: int,
) -> TeacherForcingTargetIR:
    if all(int(atom.batch_index) == int(batch_index) for atom in target_ir.atoms):
        return target_ir
    return replace(
        target_ir,
        atoms=tuple(
            replace(atom, batch_index=int(batch_index)) for atom in target_ir.atoms
        ),
    )


def _resolve_role_vocab(trainer: Any) -> RoleVocab:
    role_vocab = getattr(trainer, "teacher_forcing_role_vocab", None)
    if type(role_vocab) is RoleVocab:
        return role_vocab
    tokenizer = _resolve_tokenizer(trainer)
    groups = build_compact_token_type_groups(tokenizer)
    if len(groups.eos) != 1:
        raise ValueError("teacher_forcing role vocab requires exactly one STOP token")
    return RoleVocab(
        schema_token_ids=groups.struct,
        text_token_ids=groups.desc,
        coord_token_ids=groups.coord,
        stop_token_id=next(iter(groups.eos)),
    )


def _resolve_tokenizer(trainer: Any) -> object:
    tokenizer = getattr(trainer, "tokenizer", None)
    if tokenizer is not None:
        return tokenizer
    processing_class = getattr(trainer, "processing_class", None)
    tokenizer = getattr(processing_class, "tokenizer", None)
    if tokenizer is not None:
        return tokenizer
    template = getattr(trainer, "template", None)
    tokenizer = getattr(template, "tokenizer", None)
    if tokenizer is not None:
        return tokenizer
    raise ValueError("teacher_forcing objective requires a trainer tokenizer")


def _coverage_strength(objective_cfg: Any) -> float:
    modules = getattr(objective_cfg, "modules", None)
    coverage = getattr(modules, "within_valid_coverage", None)
    value = getattr(coverage, "coverage_strength", 0.0)
    return float(value or 0.0)


def _token_type_mass_enabled(objective_cfg: Any) -> bool:
    token_type_mass = _token_type_mass_cfg(objective_cfg)
    if token_type_mass is None:
        return False
    value = _cfg_get(token_type_mass, "enabled", False)
    if type(value) is not bool:
        raise TypeError("token_type_mass.enabled must be a bool")
    return value


def _token_type_mass_weight(objective_cfg: Any) -> float:
    token_type_mass = _token_type_mass_cfg(objective_cfg)
    if token_type_mass is None:
        return 1.0
    value = _cfg_get(token_type_mass, "weight", 1.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("token_type_mass.weight must be a finite numeric scalar")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("token_type_mass.weight must be finite")
    if parsed < 0.0:
        raise ValueError("token_type_mass.weight must be >= 0")
    return parsed


def _token_type_mass_cfg(objective_cfg: Any) -> Any:
    terms = _cfg_get(objective_cfg, "terms", None)
    return _cfg_get(terms, "token_type_mass", None)


def _cfg_get(value: Any, name: str, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


__all__ = ["TeacherForcingObjectiveMixin"]
