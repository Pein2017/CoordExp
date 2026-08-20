#!/usr/bin/env python3
"""Frozen owner-wise witness and qualification dose mechanics for the RP screen.

This owner turns the sealed Source clean-greedy surfaces into the exact records
the already implemented preservation module consumes.  It never re-derives
owners, matches boxes, decodes, or defines a loss: owner rows, parser spans,
owner classes, and compiler boundaries all arrive sealed from their existing
owners, and every margin is read from one injected HF fp32/SDPA batch-one logit
surface.

Semantics are frozen by the owning research unit:

* one constraint per ``(trusted owner, Source RP membership)``, so an owner
  emitted on both Source surfaces yields two witnesses and legacy-M owners stay
  audit-only without a Jacobian;
* eligible tokens are exactly the sealed parser span ``[token_start,
  token_end)``;
* ``z = P_r(raw_logits)`` uses the exact sign-aware repetition-penalty
  transform over the full vocabulary with no temperature division, and
  ``m_t = z[y_t] - max_{v != y_t} z[v]``;
* the witness is the minimum ``m_t`` with ties broken by the smallest generated
  token index and competitor ties by the smallest token id, and the margin must
  be finite and Source-greedy;
* ``(y, v*)`` are then frozen and ``J = d(z[y] - z[v*])/d theta`` is taken over
  the frozen :class:`ParameterLayout` and flattened to float64.

Importing this module is Torch-only; no model, engine, or artifact is touched.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any, Protocol

import torch

from scripts.research.human13_adamw_proposal_preservation import (
    LEGACY_M_MEMBERSHIP,
    LEGACY_M_OWNER_CLASS,
    TRUSTED_OWNER_CLASS,
    WEAKEST_MARGIN_SELECTION,
    WITNESS_FIRST_ORDER_TOLERANCE,
    FrozenWitnessBank,
    OwnerWitness,
    ParameterLayout,
    WitnessBinding,
    jacobian_sha256,
)


SCHEMA_VERSION = "human13_rp_crossover_witness.v1"
JVP_FD_TOLERANCE = WITNESS_FIRST_ORDER_TOLERANCE
MEMBERSHIP_BY_RP = {1.0: "u_intersect_s_1.0", 1.10: "u_intersect_s_1.10"}
RP_TEXT_BY_RP = {1.0: "1.0", 1.10: "1.10"}
GREEDY_MARGIN_TOLERANCE = 1.0e-6


class WitnessMeasurementError(ValueError):
    """Raised when a witness or dose mechanic cannot be measured fail-closed."""


@dataclass(frozen=True)
class SealedOwnerRow:
    """One sealed Source parser/matcher row and its half-open token span."""

    owner_id: str
    owner_class: str
    token_start: int
    token_end: int

    def __post_init__(self) -> None:
        if not isinstance(self.owner_id, str) or not self.owner_id:
            raise WitnessMeasurementError("sealed owner row requires an owner id")
        if self.owner_class not in {TRUSTED_OWNER_CLASS, LEGACY_M_OWNER_CLASS}:
            raise WitnessMeasurementError("sealed owner row class is not admitted")
        for field in ("token_start", "token_end"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise WitnessMeasurementError(f"sealed owner row {field} is invalid")
        if self.token_end <= self.token_start:
            raise WitnessMeasurementError(
                "sealed owner row span must be a nonempty half-open range"
            )

    @property
    def token_indices(self) -> tuple[int, ...]:
        return tuple(range(self.token_start, self.token_end))


@dataclass(frozen=True)
class SealedSourceDecode:
    """One RP-specific sealed Source clean-greedy decode over one image."""

    image_id: int
    repetition_penalty: float
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    owner_rows: tuple[SealedOwnerRow, ...]
    compiler_token_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.image_id, bool) or not isinstance(self.image_id, int):
            raise WitnessMeasurementError("sealed decode image_id must be an integer")
        if float(self.repetition_penalty) not in MEMBERSHIP_BY_RP:
            raise WitnessMeasurementError(
                "sealed decode repetition penalty must be exactly 1.0 or 1.10"
            )
        object.__setattr__(self, "repetition_penalty", float(self.repetition_penalty))
        for field in ("prompt_token_ids", "generated_token_ids"):
            tokens = tuple(getattr(self, field))
            if not tokens or any(
                isinstance(token, bool) or not isinstance(token, int) or token < 0
                for token in tokens
            ):
                raise WitnessMeasurementError(f"sealed decode {field} is invalid")
            object.__setattr__(self, field, tokens)
        rows = tuple(self.owner_rows)
        if any(not isinstance(row, SealedOwnerRow) for row in rows):
            raise WitnessMeasurementError("sealed decode requires typed owner rows")
        if len({(row.owner_id, row.owner_class) for row in rows}) != len(rows):
            raise WitnessMeasurementError("sealed decode owner rows are duplicated")
        limit = len(self.generated_token_ids)
        if any(row.token_end > limit for row in rows):
            raise WitnessMeasurementError(
                "sealed owner span escapes its generated token sequence"
            )
        object.__setattr__(self, "owner_rows", rows)
        compiler = tuple(self.compiler_token_indices)
        if any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < limit
            for index in compiler
        ):
            raise WitnessMeasurementError(
                "compiler boundary index is outside the decode"
            )
        object.__setattr__(self, "compiler_token_indices", compiler)

    @property
    def source_membership(self) -> str:
        return MEMBERSHIP_BY_RP[self.repetition_penalty]

    @property
    def surface_key(self) -> str:
        return f"{self.image_id}|{RP_TEXT_BY_RP[self.repetition_penalty]}"

    def history_token_ids(self, token_index: int) -> tuple[int, ...]:
        return (*self.prompt_token_ids, *self.generated_token_ids[:token_index])


@dataclass(frozen=True)
class WitnessSite:
    """One frozen witness site with its chosen and competitor token frozen."""

    image_id: int
    owner_id: str
    source_membership: str
    owner_class: str
    token_index: int
    chosen_token_id: int
    competitor_token_id: int
    margin_value: float

    @property
    def canonical_key(self) -> str:
        return f"{self.source_membership}|{self.image_id}|{self.owner_id}"

    @property
    def is_constraint(self) -> bool:
        return self.owner_class == TRUSTED_OWNER_CLASS


class MarginLogitSurface(Protocol):
    """The injected HF fp32/SDPA batch-one raw-logit surface."""

    def named_trainable_parameters(
        self,
    ) -> Sequence[tuple[str, torch.nn.Parameter]]: ...

    def raw_logit_rows(
        self, decode: SealedSourceDecode, token_indices: Sequence[int]
    ) -> torch.Tensor: ...


def processed_logits(
    raw_row: torch.Tensor,
    *,
    history_token_ids: Sequence[int],
    repetition_penalty: float,
) -> torch.Tensor:
    """Return ``z = P_r(raw_logits)`` with no temperature and no normalization.

    The transform mirrors the sealed sampling processor exactly: it is applied
    once per repeated token type, is sign aware, and keeps the full vocabulary.
    Unlike the replay policy transform it deliberately stops before temperature
    division and ``log_softmax`` because a witness margin is a logit difference.
    """

    if not isinstance(raw_row, torch.Tensor) or raw_row.ndim != 1:
        raise WitnessMeasurementError("a witness row must be a 1-D logit tensor")
    logits = raw_row.to(dtype=torch.float32)
    penalty = float(repetition_penalty)
    if penalty not in MEMBERSHIP_BY_RP:
        raise WitnessMeasurementError("witness repetition penalty is not admitted")
    history = tuple(history_token_ids)
    vocab_size = int(logits.shape[0])
    if any(token >= vocab_size for token in history):
        raise WitnessMeasurementError("history token is outside the logit vocabulary")
    if penalty == 1.0 or not history:
        return logits
    indices = torch.tensor(sorted(set(history)), dtype=torch.long, device=logits.device)
    selected = logits.index_select(0, indices)
    penalized = torch.where(selected < 0, selected * penalty, selected / penalty)
    return logits.scatter(0, indices, penalized)


def _margin_and_competitor(
    processed: torch.Tensor, chosen_token_id: int
) -> tuple[torch.Tensor, int]:
    """Return ``z[y] - max_{v != y} z[v]`` and the smallest tied competitor id."""

    if chosen_token_id >= int(processed.shape[0]):
        raise WitnessMeasurementError("chosen token is outside the logit vocabulary")
    masked = processed.detach().clone()
    masked[chosen_token_id] = float("-inf")
    best = torch.max(masked)
    if not bool(torch.isfinite(best).item()):
        raise WitnessMeasurementError("witness row has no finite competitor token")
    # ties resolve to the smallest token id
    competitor = int(torch.nonzero(masked == best, as_tuple=False)[0].item())
    return processed[chosen_token_id] - processed[competitor], competitor


def median(values: Sequence[float]) -> float:
    """Median with the even-cardinality mean of the two middle values."""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise WitnessMeasurementError("a median requires at least one value")
    if any(not math.isfinite(value) for value in ordered):
        raise WitnessMeasurementError("median inputs must be finite")
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def jvp_finite_difference_error(
    bank: FrozenWitnessBank,
    *,
    applied_delta: torch.Tensor,
    realized: Mapping[str, float],
) -> float:
    """Return ``max |J . Delta - (m_tilde(theta+Delta) - m_tilde(theta))|``.

    ``Delta`` is the parameter change that was actually applied and audited, and
    the finite difference is taken at unit step from the frozen Source margin.
    There is no optional small-step or random-direction probe.
    """

    if not isinstance(bank, FrozenWitnessBank):
        raise WitnessMeasurementError("certification requires a frozen witness bank")
    delta = applied_delta.to(dtype=torch.float64).reshape(-1)
    if delta.numel() != bank.layout.total_numel:
        raise WitnessMeasurementError(
            "applied delta does not match the frozen parameter layout"
        )
    expected = {item.canonical_key for item in bank.constraints}
    if not isinstance(realized, Mapping) or set(realized) != expected:
        raise WitnessMeasurementError(
            "realized margins must cover exactly the frozen constraints"
        )
    worst = 0.0
    for _index, constraint, jacobian in bank.stream_constraints():
        predicted = float(jacobian @ delta)
        finite_difference = float(realized[constraint.canonical_key]) - float(
            constraint.margin_value
        )
        if not math.isfinite(predicted) or not math.isfinite(finite_difference):
            raise WitnessMeasurementError(
                "witness certification measured a non-finite value"
            )
        worst = max(worst, abs(predicted - finite_difference))
    if not worst >= 0.0:
        raise WitnessMeasurementError("witness certification produced no measurement")
    return worst


class WitnessMeasurement:
    """One frozen witness/dose measurement bound to sealed Source surfaces."""

    def __init__(
        self,
        *,
        decodes: Sequence[SealedSourceDecode],
        surface: MarginLogitSurface,
    ) -> None:
        bound = tuple(decodes)
        if not bound or any(not isinstance(item, SealedSourceDecode) for item in bound):
            raise WitnessMeasurementError(
                "witness measurement requires typed sealed Source decodes"
            )
        if len({item.surface_key for item in bound}) != len(bound):
            raise WitnessMeasurementError("sealed Source surfaces are duplicated")
        for name in ("named_trainable_parameters", "raw_logit_rows"):
            if not callable(getattr(surface, name, None)):
                raise WitnessMeasurementError(
                    f"margin surface does not expose {name}()"
                )
        self._decodes = bound
        self._surface = surface
        self._sites: tuple[WitnessSite, ...] | None = None

    # -- surface helpers ----------------------------------------------------

    def _parameters(self) -> tuple[tuple[str, torch.nn.Parameter], ...]:
        bound = tuple(self._surface.named_trainable_parameters())
        if not bound:
            raise WitnessMeasurementError("margin surface exposes no trainables")
        return bound

    @property
    def parameter_layout(self) -> ParameterLayout:
        return ParameterLayout.from_named_parameters(self._parameters())

    def _rows(
        self, decode: SealedSourceDecode, token_indices: Sequence[int]
    ) -> torch.Tensor:
        indices = tuple(token_indices)
        rows = self._surface.raw_logit_rows(decode, indices)
        if (
            not isinstance(rows, torch.Tensor)
            or rows.ndim != 2
            or int(rows.shape[0]) != len(indices)
        ):
            raise WitnessMeasurementError(
                "margin surface did not return one row per requested token"
            )
        return rows

    def _processed(
        self, decode: SealedSourceDecode, token_index: int, row: torch.Tensor
    ) -> torch.Tensor:
        return processed_logits(
            row,
            history_token_ids=decode.history_token_ids(token_index),
            repetition_penalty=decode.repetition_penalty,
        )

    # -- witness selection --------------------------------------------------

    def _select(self) -> tuple[WitnessSite, ...]:
        sites: list[WitnessSite] = []
        for decode in self._decodes:
            for row in decode.owner_rows:
                indices = row.token_indices
                logits = self._rows(decode, indices)
                best: tuple[float, int, int] | None = None
                for offset, token_index in enumerate(indices):
                    processed = self._processed(decode, token_index, logits[offset])
                    margin, competitor = _margin_and_competitor(
                        processed, decode.generated_token_ids[token_index]
                    )
                    value = float(margin.detach())
                    if not math.isfinite(value):
                        raise WitnessMeasurementError(
                            f"witness margin for {row.owner_id} is not finite"
                        )
                    if value < -GREEDY_MARGIN_TOLERANCE:
                        raise WitnessMeasurementError(
                            "witness margin for "
                            f"{row.owner_id} is not Source-greedy "
                            f"(rp={decode.repetition_penalty:g}, "
                            f"membership={decode.source_membership}, "
                            f"token_index={token_index}, "
                            f"chosen_token_id={decode.generated_token_ids[token_index]}, "
                            f"competitor_token_id={competitor}, "
                            f"margin={value:.9g})"
                        )
                    # ties resolve to the smallest generated token index
                    if best is None or value < best[0]:
                        best = (value, token_index, competitor)
                assert best is not None
                margin_value, token_index, competitor = best
                sites.append(
                    WitnessSite(
                        image_id=decode.image_id,
                        owner_id=row.owner_id,
                        source_membership=(
                            decode.source_membership
                            if row.owner_class == TRUSTED_OWNER_CLASS
                            else LEGACY_M_MEMBERSHIP
                        ),
                        owner_class=row.owner_class,
                        token_index=token_index,
                        chosen_token_id=decode.generated_token_ids[token_index],
                        competitor_token_id=competitor,
                        margin_value=margin_value,
                    )
                )
        keys = [item.canonical_key for item in sites]
        if len(set(keys)) != len(keys):
            raise WitnessMeasurementError(
                "one owner and Source membership may hold only one witness"
            )
        return tuple(sites)

    @property
    def sites(self) -> tuple[WitnessSite, ...]:
        if self._sites is None:
            self._sites = self._select()
        return self._sites

    @property
    def witness_sites(self) -> tuple[WitnessSite, ...]:
        return tuple(item for item in self.sites if item.is_constraint)

    @property
    def audit_only_sites(self) -> tuple[WitnessSite, ...]:
        return tuple(item for item in self.sites if not item.is_constraint)

    # -- frozen bank --------------------------------------------------------

    def _jacobian(self, site: WitnessSite) -> torch.Tensor:
        decode = self._decode_for(site)
        parameters = self._parameters()
        rows = self._rows(decode, (site.token_index,))
        processed = self._processed(decode, site.token_index, rows[0])
        margin = processed[site.chosen_token_id] - processed[site.competitor_token_id]
        grads = torch.autograd.grad(
            margin,
            [parameter for _name, parameter in parameters],
            allow_unused=True,
            retain_graph=False,
        )
        flat = torch.cat(
            [
                (
                    torch.zeros(parameter.numel(), dtype=torch.float64)
                    if grad is None
                    else grad.detach().reshape(-1).to(dtype=torch.float64)
                )
                for (_name, parameter), grad in zip(parameters, grads, strict=True)
            ]
        )
        if not bool(torch.isfinite(flat).all()):
            raise WitnessMeasurementError(
                f"witness Jacobian for {site.canonical_key} is not finite"
            )
        return flat

    def _decode_for(self, site: WitnessSite) -> SealedSourceDecode:
        for decode in self._decodes:
            membership = (
                decode.source_membership if site.is_constraint else LEGACY_M_MEMBERSHIP
            )
            if decode.image_id == site.image_id and membership == (
                site.source_membership
            ):
                return decode
        raise WitnessMeasurementError(
            f"no sealed Source surface owns {site.canonical_key}"
        )

    def freeze_witness_bank(self, *, binding: WitnessBinding) -> FrozenWitnessBank:
        """Freeze the owner-wise bank with a lazy, digest-checked Jacobian reader."""

        if not isinstance(binding, WitnessBinding):
            raise WitnessMeasurementError("a typed witness binding is required")
        layout = self.parameter_layout
        digests: dict[str, str] = {}
        for site in self.witness_sites:
            jacobian = self._jacobian(site)
            if jacobian.numel() != layout.total_numel:
                raise WitnessMeasurementError(
                    "witness Jacobian does not match the frozen parameter layout"
                )
            digests[site.canonical_key] = jacobian_sha256(jacobian)
            del jacobian
        by_key = {site.canonical_key: site for site in self.sites}
        witnesses = tuple(
            OwnerWitness(
                image_id=str(site.image_id),
                owner_id=site.owner_id,
                source_membership=site.source_membership,
                owner_class=site.owner_class,
                token_id=site.chosen_token_id,
                margin_selection=WEAKEST_MARGIN_SELECTION,
                margin_value=site.margin_value,
                detached=True,
                jacobian_sha256=digests.get(site.canonical_key),
            )
            for site in self.sites
        )

        def provider(witness: OwnerWitness) -> torch.Tensor:
            site = by_key.get(witness.canonical_key)
            if site is None or not site.is_constraint:
                raise WitnessMeasurementError(
                    f"no frozen witness site owns {witness.canonical_key}"
                )
            return self._jacobian(site)

        constraints = tuple(
            sorted(
                (item for item in witnesses if item.is_constraint),
                key=lambda item: (item.source_membership, item.image_id, item.owner_id),
            )
        )
        audit_only = tuple(
            sorted(
                (item for item in witnesses if not item.is_constraint),
                key=lambda item: (item.source_membership, item.image_id, item.owner_id),
            )
        )
        return FrozenWitnessBank(
            binding=binding,
            layout=layout,
            constraints=constraints,
            audit_only=audit_only,
            jacobian_provider=provider,
        )

    # -- measured margins ---------------------------------------------------

    def margin_values(self) -> dict[str, float]:
        """Re-maximize the competitor at every frozen constraint site."""

        measured: dict[str, float] = {}
        for site in self.witness_sites:
            decode = self._decode_for(site)
            with torch.no_grad():
                rows = self._rows(decode, (site.token_index,))
                processed = self._processed(decode, site.token_index, rows[0])
                margin, _competitor = _margin_and_competitor(
                    processed, site.chosen_token_id
                )
            value = float(margin)
            if not math.isfinite(value):
                raise WitnessMeasurementError(
                    f"realized margin for {site.canonical_key} is not finite"
                )
            measured[site.canonical_key] = value
        return measured

    @property
    def dose_sites(self) -> tuple[tuple[int, float, int], ...]:
        """The deduplicated compiler/trusted-witness union, legacy-M excluded."""

        collected: list[tuple[int, float, int]] = []
        for decode in self._decodes:
            for token_index in decode.compiler_token_indices:
                collected.append(
                    (decode.image_id, decode.repetition_penalty, token_index)
                )
        for site in self.witness_sites:
            decode = self._decode_for(site)
            collected.append(
                (decode.image_id, decode.repetition_penalty, site.token_index)
            )
        return tuple(sorted(set(collected)))

    def dose_site_margins(self) -> dict[str, float]:
        """Score every dose site on its own RP surface."""

        by_surface = {decode.surface_key: decode for decode in self._decodes}
        measured: dict[str, float] = {}
        for image_id, repetition_penalty, token_index in self.dose_sites:
            decode = by_surface[f"{image_id}|{RP_TEXT_BY_RP[repetition_penalty]}"]
            with torch.no_grad():
                rows = self._rows(decode, (token_index,))
                processed = self._processed(decode, token_index, rows[0])
                margin, _competitor = _margin_and_competitor(
                    processed, decode.generated_token_ids[token_index]
                )
            value = float(margin)
            if not math.isfinite(value):
                raise WitnessMeasurementError("a dose-site margin is not finite")
            measured[
                f"{image_id}|{RP_TEXT_BY_RP[repetition_penalty]}|{token_index}"
            ] = value
        return measured

    def teacher_forced_greedy_change_count(self) -> int:
        """Count sealed decode tokens whose processed argmax is no longer sealed."""

        changed = 0
        for decode in self._decodes:
            indices = tuple(range(len(decode.generated_token_ids)))
            with torch.no_grad():
                rows = self._rows(decode, indices)
                for offset, token_index in enumerate(indices):
                    processed = self._processed(decode, token_index, rows[offset])
                    if not bool(torch.isfinite(processed).all().item()):
                        raise WitnessMeasurementError(
                            "teacher-forced greedy row is not finite"
                        )
                    best = torch.max(processed)
                    # ties resolve to the smallest token id
                    argmax = int(
                        torch.nonzero(processed == best, as_tuple=False)[0].item()
                    )
                    if argmax != decode.generated_token_ids[token_index]:
                        changed += 1
        return changed

    def teacher_forced_greedy_token_ids(self) -> tuple[tuple[int, ...], ...]:
        """Return processed teacher-forced argmax IDs for each sealed decode.

        This is an observation surface for cross-surface behavioral
        reconciliation.  It does not alter the strict sampler-to-replay parity
        contract, which remains owned by the shared-surface replay receipts.
        """

        result: list[tuple[int, ...]] = []
        for decode in self._decodes:
            indices = tuple(range(len(decode.generated_token_ids)))
            with torch.no_grad():
                rows = self._rows(decode, indices)
                argmax_ids: list[int] = []
                for offset, token_index in enumerate(indices):
                    processed = self._processed(decode, token_index, rows[offset])
                    if not bool(torch.isfinite(processed).all().item()):
                        raise WitnessMeasurementError(
                            "teacher-forced greedy row is not finite"
                        )
                    best = torch.max(processed)
                    argmax_ids.append(
                        int(torch.nonzero(processed == best, as_tuple=False)[0].item())
                    )
            result.append(tuple(argmax_ids))
        return tuple(result)


def decision_margin_dose_statistics(
    *, source_margins: Mapping[str, float], applied_margins: Mapping[str, float]
) -> tuple[float, float]:
    """Return ``(median |source margin|, median |margin displacement|)``."""

    if set(source_margins) != set(applied_margins) or not source_margins:
        raise WitnessMeasurementError(
            "dose-site margins must cover exactly the same sealed sites"
        )
    keys = sorted(source_margins)
    return (
        median([abs(float(source_margins[key])) for key in keys]),
        median(
            [
                abs(float(applied_margins[key]) - float(source_margins[key]))
                for key in keys
            ]
        ),
    )


def mirror_parameter_values(
    *,
    source: Sequence[tuple[str, Any]],
    target: Sequence[tuple[str, torch.nn.Parameter]],
) -> None:
    """Copy trained parameter values onto the margin surface fail-closed."""

    values = dict(source)
    bound = tuple(target)
    if set(values) != {name for name, _ in bound} or len(values) != len(bound):
        raise WitnessMeasurementError(
            "margin surface trainables differ from the trained parameter names"
        )
    with torch.no_grad():
        for name, parameter in bound:
            incoming = values[name]
            if tuple(incoming.shape) != tuple(parameter.shape):
                raise WitnessMeasurementError(
                    f"margin surface parameter {name} shape differs"
                )
            parameter.copy_(incoming.detach().to(dtype=parameter.dtype))


__all__ = [
    "GREEDY_MARGIN_TOLERANCE",
    "JVP_FD_TOLERANCE",
    "MEMBERSHIP_BY_RP",
    "RP_TEXT_BY_RP",
    "SCHEMA_VERSION",
    "MarginLogitSurface",
    "SealedOwnerRow",
    "SealedSourceDecode",
    "WitnessMeasurement",
    "WitnessMeasurementError",
    "WitnessSite",
    "decision_margin_dose_statistics",
    "jvp_finite_difference_error",
    "median",
    "mirror_parameter_values",
    "processed_logits",
]
