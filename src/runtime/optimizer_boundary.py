"""Runtime-owned optimizer-boundary receipt and post-wrapper consensus.

This module owns the *truth* of one planned step's optimizer boundary. It has
exactly two responsibilities:

1. `AppliedUpdateReceipt` - the single bounded record of what actually
   happened at the boundary. It is produced ONLY by its own named
   constructors, so no reporter, session, or exception handler can synthesize
   `attempted`/`applied`/`step_was_skipped` from a traceback.
2. `reduce_post_wrapper_reports` - the bounded all-rank consensus over the
   immediate post-wrapper skip booleans (`all_skipped | none_skipped |
   mixed`). This is an optimizer-CORRECTNESS collective, not a
   metric/observability one, and it must complete before the scheduler or any
   row handling.

The closed pre-wrapper decision (`apply | scaler_skip | not_attempted`, or a
terminal unsafe outcome) is reduced in `src/runtime/finite_gates.py` because it
rides the existing all-rank gradient/overflow report; this module never opens a
second gradient-scan collective.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from src.common.errors import RuntimeContractError

# The closed normal action set. It MUST NOT grow a fourth terminal member:
# terminal outcomes are carried by `terminal`/`terminal_reason` instead.
OPTIMIZER_BOUNDARY_ACTIONS: tuple[str, ...] = ("apply", "scaler_skip", "not_attempted")

# The closed post-wrapper consensus outcomes.
POST_WRAPPER_OUTCOMES: tuple[str, ...] = ("all_skipped", "none_skipped", "mixed")

# The closed, bounded composite mutation vocabulary.
MUTATION_STATES: tuple[str, ...] = (
    # No wrapper call and no unscale: nothing at the boundary moved.
    "unchanged",
    # Every rank's wrapper applied the update.
    "applied",
    # Every rank applied an update that the action did not sanction (a
    # `scaler_skip` whose wrapper nonetheless stepped on known non-finite
    # gradients).
    "applied_unsafe",
    # Every rank's wrapper was suppressed by GradScaler; parameters and the
    # underlying optimizer are untouched and the scaler is finalized.
    "scaler_suppressed",
    # Bounded unknown: rank-divergent, or an unfinalized GradScaler whose
    # per-optimizer stage and `found_inf` record already moved.
    "divergent_or_unknown",
)

TERMINAL_PRE_WRAPPER_SCALER_CANDIDACY_DIVERGENT = "pre_wrapper_scaler_candidacy_divergent"
TERMINAL_PRE_WRAPPER_MIXED_SCALER_OVERFLOW = "pre_wrapper_mixed_scaler_overflow"
TERMINAL_PRE_WRAPPER_UNRELATED_UNSAFE = "pre_wrapper_unrelated_unsafe"
# Declared fp16 with no reachable, enabled GradScaler on any rank. The launch
# gate refuses this state, so a boundary that still observes it is post-launch
# drift and must not be reclassified as the retained bf16/non-scaler path.
TERMINAL_PRE_WRAPPER_FP16_SCALER_MISSING = "pre_wrapper_fp16_scaler_missing"
TERMINAL_POST_WRAPPER_MIXED = "post_wrapper_mixed"
TERMINAL_POST_WRAPPER_APPLY_ALL_SKIPPED = "post_wrapper_apply_all_skipped"
TERMINAL_POST_WRAPPER_SCALER_SKIP_NONE_SKIPPED = "post_wrapper_scaler_skip_none_skipped"

TERMINAL_REASONS: tuple[str, ...] = (
    TERMINAL_PRE_WRAPPER_SCALER_CANDIDACY_DIVERGENT,
    TERMINAL_PRE_WRAPPER_MIXED_SCALER_OVERFLOW,
    TERMINAL_PRE_WRAPPER_UNRELATED_UNSAFE,
    TERMINAL_PRE_WRAPPER_FP16_SCALER_MISSING,
    TERMINAL_POST_WRAPPER_MIXED,
    TERMINAL_POST_WRAPPER_APPLY_ALL_SKIPPED,
    TERMINAL_POST_WRAPPER_SCALER_SKIP_NONE_SKIPPED,
)

APPLIED_UPDATE_STATUS = "applied"
SCALER_SKIP_UPDATE_STATUS = "skipped_scaler_overflow"

_MAX_STATUS_BYTES = 128

# Only the named constructors below may hand this token to the dataclass.
_CONSTRUCTOR_TOKEN = object()


def _checked_reason(reason: str) -> str:
    text = str(reason)
    if not text or not text.isascii() or len(text) > _MAX_STATUS_BYTES:
        raise RuntimeContractError(
            "optimizer boundary reason must be a bounded ASCII string",
            code="runtime.update_receipt_reason_invalid",
            context={"reason_length": len(text)},
        )
    return text


def _checked_group_count(group_count: int) -> int:
    if isinstance(group_count, bool) or not isinstance(group_count, int):
        raise RuntimeContractError(
            "optimizer group count must be an integer",
            code="runtime.update_receipt_group_count_invalid",
            context={"value_type": type(group_count).__name__},
        )
    if group_count < 0:
        raise RuntimeContractError(
            "optimizer group count must not be negative",
            code="runtime.update_receipt_group_count_invalid",
            context={"group_count": group_count},
        )
    return int(group_count)


@dataclass(frozen=True)
class AppliedUpdateReceipt:
    """The one bounded truth of a planned step's optimizer boundary.

    `applied` and `step_was_skipped` are nullable ON PURPOSE: a post-wrapper
    `mixed` consensus genuinely does not know which ranks applied their update,
    and inventing a boolean there would be a lie with durable consequences.
    """

    planned_step_id: int
    action: str | None
    attempted: bool
    applied: bool | None
    step_was_skipped: bool | None
    mutation_state: str
    optimizer_update_status: str
    terminal: bool
    terminal_reason: str | None
    reason: str | None
    post_wrapper_outcome: str | None
    group_learning_rates: tuple[float | None, ...]
    # The gate decision's finite_status at the moment the terminal converged.
    # It is always computed before a terminal reason is chosen, so terminal
    # constructors receive the known value; "unavailable" remains only for
    # receipts whose construction path genuinely never saw a gate decision.
    finite_status: str = "unavailable"
    construction_token: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.construction_token is not _CONSTRUCTOR_TOKEN:
            raise RuntimeContractError(
                "AppliedUpdateReceipt is runtime-owned: use one of its named "
                "constructors instead of synthesizing boundary booleans",
                code="runtime.update_receipt_direct_construction",
            )
        if self.action is not None and self.action not in OPTIMIZER_BOUNDARY_ACTIONS:
            raise RuntimeContractError(
                "optimizer boundary action is not in the closed action set",
                code="runtime.update_receipt_action_invalid",
                context={"action": self.action},
            )
        if self.mutation_state not in MUTATION_STATES:
            raise RuntimeContractError(
                "optimizer boundary mutation state is not in the closed set",
                code="runtime.update_receipt_mutation_state_invalid",
                context={"mutation_state": self.mutation_state},
            )
        if (
            self.post_wrapper_outcome is not None
            and self.post_wrapper_outcome not in POST_WRAPPER_OUTCOMES
        ):
            raise RuntimeContractError(
                "post-wrapper outcome is not in the closed consensus set",
                code="runtime.update_receipt_outcome_invalid",
                context={"post_wrapper_outcome": self.post_wrapper_outcome},
            )
        if self.terminal is (self.terminal_reason is None):
            raise RuntimeContractError(
                "a terminal receipt requires a terminal reason and a normal "
                "receipt must not carry one",
                code="runtime.update_receipt_terminal_invalid",
                context={
                    "terminal": self.terminal,
                    "terminal_reason": self.terminal_reason,
                },
            )
        if not self.attempted and self.applied:
            raise RuntimeContractError(
                "a receipt cannot report an applied update without a wrapper call",
                code="runtime.update_receipt_inconsistent",
            )
        if self.finite_status not in ("finite", "non_finite", "unavailable"):
            raise RuntimeContractError(
                "receipt finite_status is not in the closed set",
                code="runtime.update_receipt_finite_status_invalid",
                context={"finite_status": self.finite_status},
            )

    # -- normal completed boundaries --------------------------------------

    @classmethod
    def applied_update(
        cls,
        planned_step_id: int,
        *,
        group_learning_rates: Sequence[float | None],
        post_wrapper_outcome: str | None = None,
    ) -> "AppliedUpdateReceipt":
        """`apply` accepted: every rank applied the pre-call group LRs."""

        return cls(
            planned_step_id=int(planned_step_id),
            action="apply",
            attempted=True,
            applied=True,
            step_was_skipped=False,
            mutation_state="applied",
            optimizer_update_status=APPLIED_UPDATE_STATUS,
            terminal=False,
            terminal_reason=None,
            reason=None,
            post_wrapper_outcome=post_wrapper_outcome,
            group_learning_rates=tuple(
                None if value is None else float(value) for value in group_learning_rates
            ),
            construction_token=_CONSTRUCTOR_TOKEN,
        )

    @classmethod
    def scaler_skipped(
        cls,
        planned_step_id: int,
        group_count: int,
    ) -> "AppliedUpdateReceipt":
        """`scaler_skip` accepted: the wrapper ran only to finalize the scaler.

        No update was applied, so every configured group LR is JSON `null`
        even though this IS a completed planned-step boundary.
        """

        groups = _checked_group_count(group_count)
        return cls(
            planned_step_id=int(planned_step_id),
            action="scaler_skip",
            attempted=True,
            applied=False,
            step_was_skipped=True,
            mutation_state="scaler_suppressed",
            optimizer_update_status=SCALER_SKIP_UPDATE_STATUS,
            terminal=False,
            terminal_reason=None,
            reason=None,
            post_wrapper_outcome="all_skipped",
            group_learning_rates=(None,) * groups,
            construction_token=_CONSTRUCTOR_TOKEN,
        )

    @classmethod
    def not_attempted(
        cls,
        planned_step_id: int,
        group_count: int,
        reason: str,
    ) -> "AppliedUpdateReceipt":
        """Supported `not_attempted`: no wrapper call, nothing mutated.

        Used for pre-backward scalar rejection and the retained
        bf16/non-scaler post-backward rejection. Neither has unscaled, so the
        composite mutation state really is `unchanged`.
        """

        groups = _checked_group_count(group_count)
        status = _checked_reason(reason)
        return cls(
            planned_step_id=int(planned_step_id),
            action="not_attempted",
            attempted=False,
            applied=False,
            step_was_skipped=False,
            mutation_state="unchanged",
            optimizer_update_status=status,
            terminal=False,
            terminal_reason=None,
            reason=status,
            post_wrapper_outcome=None,
            group_learning_rates=(None,) * groups,
            construction_token=_CONSTRUCTOR_TOKEN,
        )

    # -- terminal boundaries ----------------------------------------------

    @classmethod
    def terminal_not_attempted(
        cls,
        planned_step_id: int,
        group_count: int,
        reason: str,
        *,
        unscale_completed: bool,
        finite_status: str = "unavailable",
    ) -> "AppliedUpdateReceipt":
        """Terminal BEFORE any wrapper call.

        The underlying optimizer and parameters are untouched - that much is
        known. But when exactly-once unscale already ran, GradScaler's
        per-optimizer stage and `found_inf` record have moved and may be
        unfinalized or rank-divergent, so the COMPOSITE state is
        `divergent_or_unknown`, never `unchanged`. `unscale_completed` is a
        fact the runtime observed, not a label the caller may choose.
        """

        groups = _checked_group_count(group_count)
        terminal_reason = _checked_reason(reason)
        return cls(
            planned_step_id=int(planned_step_id),
            action=None,
            attempted=False,
            applied=False,
            step_was_skipped=False,
            mutation_state=(
                "divergent_or_unknown" if bool(unscale_completed) else "unchanged"
            ),
            optimizer_update_status=_checked_reason(f"terminal_{terminal_reason}"),
            terminal=True,
            terminal_reason=terminal_reason,
            reason=terminal_reason,
            post_wrapper_outcome=None,
            group_learning_rates=(None,) * groups,
            finite_status=finite_status,
            construction_token=_CONSTRUCTOR_TOKEN,
        )

    @classmethod
    def terminal_post_wrapper(
        cls,
        planned_step_id: int,
        group_count: int,
        *,
        action: str,
        outcome: str,
        pre_call_learning_rates: Sequence[float | None],
        finite_status: str = "unavailable",
    ) -> "AppliedUpdateReceipt":
        """Terminal AFTER every rank's wrapper returned.

        `mixed` is the only branch that loses application truth. The two
        unanimous action-contradictory branches keep their KNOWN booleans, and
        `scaler_skip + none_skipped` keeps the identical pre-call LRs that were
        in fact applied on every rank.
        """

        groups = _checked_group_count(group_count)
        if action not in ("apply", "scaler_skip"):
            raise RuntimeContractError(
                "a post-wrapper terminal receipt requires the wrapper-calling action",
                code="runtime.update_receipt_action_invalid",
                context={"action": action},
            )
        if outcome not in POST_WRAPPER_OUTCOMES:
            raise RuntimeContractError(
                "post-wrapper outcome is not in the closed consensus set",
                code="runtime.update_receipt_outcome_invalid",
                context={"post_wrapper_outcome": outcome},
            )
        if outcome == "mixed":
            applied: bool | None = None
            skipped: bool | None = None
            mutation_state = "divergent_or_unknown"
            terminal_reason = TERMINAL_POST_WRAPPER_MIXED
            learning_rates: tuple[float | None, ...] = (None,) * groups
        elif action == "apply":
            # apply + all_skipped: known, and nothing was applied.
            applied = False
            skipped = True
            mutation_state = "scaler_suppressed"
            terminal_reason = TERMINAL_POST_WRAPPER_APPLY_ALL_SKIPPED
            learning_rates = (None,) * groups
        else:
            # scaler_skip + none_skipped: known, and every rank DID apply the
            # identical pre-call LRs to known non-finite gradients.
            applied = True
            skipped = False
            mutation_state = "applied_unsafe"
            terminal_reason = TERMINAL_POST_WRAPPER_SCALER_SKIP_NONE_SKIPPED
            learning_rates = tuple(
                None if value is None else float(value)
                for value in pre_call_learning_rates
            )
        return cls(
            planned_step_id=int(planned_step_id),
            action=action,
            attempted=True,
            applied=applied,
            step_was_skipped=skipped,
            mutation_state=mutation_state,
            optimizer_update_status=_checked_reason(f"terminal_{terminal_reason}"),
            terminal=True,
            terminal_reason=terminal_reason,
            reason=terminal_reason,
            post_wrapper_outcome=outcome,
            group_learning_rates=learning_rates,
            finite_status=finite_status,
            construction_token=_CONSTRUCTOR_TOKEN,
        )

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "planned_step_id": self.planned_step_id,
            "optimizer_boundary_action": self.action,
            "attempted": self.attempted,
            "applied": self.applied,
            "step_was_skipped": self.step_was_skipped,
            "mutation_state": self.mutation_state,
            "optimizer_update_status": self.optimizer_update_status,
            "terminal": self.terminal,
            "terminal_reason": self.terminal_reason,
            "post_wrapper_outcome": self.post_wrapper_outcome,
            "group_learning_rates": list(self.group_learning_rates),
            "finite_status": self.finite_status,
        }


class OptimizerBoundaryTerminal(RuntimeContractError):
    """Raised identically on every rank AFTER a terminal outcome converges.

    Carrying the receipt keeps the terminal row builder (task 3.7) from having
    to reconstruct boundary truth from an exception message.
    """

    def __init__(self, receipt: AppliedUpdateReceipt) -> None:
        if not isinstance(receipt, AppliedUpdateReceipt) or not receipt.terminal:
            raise RuntimeContractError(
                "OptimizerBoundaryTerminal requires a terminal update receipt",
                code="runtime.optimizer_boundary_terminal_invalid",
            )
        super().__init__(
            "optimizer boundary converged a terminal unsafe outcome",
            code="runtime.optimizer_boundary_terminal",
            context={
                "planned_step_id": receipt.planned_step_id,
                "terminal_reason": receipt.terminal_reason,
                "mutation_state": receipt.mutation_state,
                "attempted": receipt.attempted,
            },
        )
        self.receipt = receipt


@dataclass(frozen=True)
class RankPostWrapperReport:
    """One rank's immediate post-wrapper skip flag.

    Authoritative as an INPUT only: no rank may interpret or raise from its own
    value before `reduce_post_wrapper_reports` converges.
    """

    planned_step_id: int
    rank: int
    world_size: int
    action: str
    step_was_skipped: bool
    # An unreadable local skip flag is UNKNOWN truth, not a rank-local raise:
    # it travels into the same consensus so one rank can never leave its peers
    # waiting on a collective it abandoned.
    report_error_code: str | None = None


def reduce_post_wrapper_reports(
    reports: Sequence[RankPostWrapperReport],
) -> str:
    """Converge the closed `all_skipped | none_skipped | mixed` outcome."""

    checked = tuple(reports)
    if not checked:
        raise RuntimeContractError(
            "post-wrapper consensus requires at least one rank report",
            code="runtime.post_wrapper_empty_reports",
        )
    for report in checked:
        if not isinstance(report, RankPostWrapperReport):
            raise RuntimeContractError(
                "post-wrapper consensus requires typed rank reports",
                code="runtime.post_wrapper_report_invalid",
                context={"value_type": type(report).__name__},
            )
    planned_step_ids = {report.planned_step_id for report in checked}
    if len(planned_step_ids) != 1:
        raise RuntimeContractError(
            "post-wrapper reports must share planned_step_id",
            code="runtime.post_wrapper_planned_step",
            context={"planned_step_ids": sorted(planned_step_ids)},
        )
    world_sizes = {report.world_size for report in checked}
    if len(world_sizes) != 1:
        raise RuntimeContractError(
            "post-wrapper reports must share world_size",
            code="runtime.post_wrapper_world_size",
            context={"world_sizes": sorted(world_sizes)},
        )
    world_size = checked[0].world_size
    ranks = tuple(sorted(report.rank for report in checked))
    if ranks != tuple(range(world_size)):
        raise RuntimeContractError(
            "post-wrapper consensus must contain exactly one report per rank",
            code="runtime.post_wrapper_rank_coverage",
            context={"observed_ranks": list(ranks), "world_size": world_size},
        )
    actions = {report.action for report in checked}
    if len(actions) != 1:
        raise RuntimeContractError(
            "ranks entered the optimizer wrapper with different actions",
            code="runtime.post_wrapper_action_divergent",
            context={"observed_actions": sorted(actions)},
        )
    if any(report.report_error_code is not None for report in checked):
        # Some rank could not observe its own outcome, so the global truth is
        # unknown rather than unanimous.
        return "mixed"
    skipped = {bool(report.step_was_skipped) for report in checked}
    if skipped == {True}:
        return "all_skipped"
    if skipped == {False}:
        return "none_skipped"
    return "mixed"


__all__ = [
    "APPLIED_UPDATE_STATUS",
    "AppliedUpdateReceipt",
    "MUTATION_STATES",
    "OPTIMIZER_BOUNDARY_ACTIONS",
    "OptimizerBoundaryTerminal",
    "POST_WRAPPER_OUTCOMES",
    "RankPostWrapperReport",
    "SCALER_SKIP_UPDATE_STATUS",
    "TERMINAL_POST_WRAPPER_APPLY_ALL_SKIPPED",
    "TERMINAL_POST_WRAPPER_MIXED",
    "TERMINAL_POST_WRAPPER_SCALER_SKIP_NONE_SKIPPED",
    "TERMINAL_PRE_WRAPPER_FP16_SCALER_MISSING",
    "TERMINAL_PRE_WRAPPER_MIXED_SCALER_OVERFLOW",
    "TERMINAL_PRE_WRAPPER_SCALER_CANDIDACY_DIVERGENT",
    "TERMINAL_PRE_WRAPPER_UNRELATED_UNSAFE",
    "TERMINAL_REASONS",
    "reduce_post_wrapper_reports",
]
